/*************************************************************************\
 * Copyright (c) 2024 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: correct_coupling.c
 * purpose: skew-quadrupole correction of vertical dispersion
 *
 * Vertical dispersion eta_y at user-selected BPMs is driven to zero by
 * adjusting user-selected skew-quadrupole knobs. The response matrix
 *      R[i][j] = d eta_y(BPM_i) / d K1_j
 * is built by one-sided finite difference (perturbing each knob and
 * recomputing the periodic twiss solution), SVD-pseudo-inverted with a
 * user-controlled truncation, and the resulting step
 *      Delta K1 = - pinv(R) . eta_y
 * is applied (optionally scaled by an absolute |K1| limit). The loop
 * iterates until the RMS of eta_y at BPMs falls below `convergence` or
 * `n_iterations` is exhausted.
 *
 * The command depends on `twiss_output` having been issued previously,
 * so the lattice has a current twiss solution and an `etay` value at
 * every element.
 */
#include "mdb.h"
#include "matlib.h"
#include "matrixOp.h"
#include "track.h"
#include "correctionEngine.h"
#include "correctorStash.h"
#include "correct_coupling.h"

/* Per-step reassertion of skew-quadrupole values.  Populated lazily from the
 * existing knobs[] inventory at first save/reassert; reassert flag tracks the
 * namelist parameter reset_correctors_each_step. */
static CORRECTOR_STASH ccStash = { NULL, NULL, NULL, NULL, 0, 0, 1, 0, "coupling" };

/* Module-level state retained between setup/do/finish.  skewName/skewType/
 * skewItem are populated only by &compute_coupling_response_matrix and
 * &load_coupling_response_matrix; the &correct_coupling namelist uses the
 * family-list state defined immediately below. */
static long initialized = 0;
static char **skewName = NULL;      /* parsed name_pattern from compute/load */
static long nSkewName = 0;
static char **skewType = NULL;
static long nSkewType = 0;
static char **bpmName = NULL;
static long nBpmName = 0;
static char **bpmType = NULL;
static long nBpmType = 0;
static char *skewItem = NULL;       /* item used by compute_/load_ */

/* Family-list state from the &correct_coupling namelist.  Mirrors the
 * correct_lattice convention: parallel space-separated tokens for items,
 * lower/upper bounds, and a separate exclude list. */
static char **corrPatterns = NULL;       static long nCorrPatterns = 0;
static char *corrPatternStr = NULL;
static char **itemsList = NULL;          static long nItemsList = 0;
static double *lowerList = NULL;         static long nLowerList = 0;
static double *upperList = NULL;         static long nUpperList = 0;
static char **excludePatterns = NULL;    static long nExcludePatterns = 0;

/* Per-knob arrays sized nSkew. */
static char **knobItem = NULL;
static double *knobLower = NULL;
static double *knobUpper = NULL;
static long *knobFamily = NULL;

/* The knob/observable struct types live in correctionEngine.h as LRC_Knob and
 * LRC_Bpm. */

static SDDS_DATASET SDDSstrengthLog, SDDSetay, SDDSresponse, SDDSrmsLog;
static short SDDSstrengthLogInit = 0, SDDSetayInit = 0, SDDSresponseInit = 0, SDDSrmsLogInit = 0;

/* Module-level persistent state for the response matrix and the working buffers.
 * Promoted from static-locals in do_correct_coupling so the matrix can be
 * built in setup_correct_coupling (when use_perturbed_matrix=0) and reused
 * across multiple do_correct_coupling calls inside the steps loop. */
static LRC_Knob *knobs = NULL;
static LRC_Bpm *bpms = NULL;
static long nSkew = 0, nBpm = 0;
static double *etay = NULL, *etayPert = NULL, *dK = NULL;
static double **R = NULL;
/* Set when R holds a valid response matrix for the current knob/bpm sets. */
static short responseValid = 0;

/* Cross-plane steering response channel ------------------------------------
 * When cross_response_weight > 0, the response matrix gets additional rows
 * for the cross-plane steering response: H-corrector -> y at vertical BPMs,
 * and V-corrector -> x at horizontal BPMs.  Each (probe-corrector, target-BPM)
 * pair contributes one observable; the engine builds d(observable)/d(skew K1)
 * by the usual finite-difference + SVD machinery.  All cross-plane state lives
 * in module scope and is set up at setup_correct_coupling time when active. */
static LRC_Knob *hCorrs = NULL;
static long      nHCorr = 0;
static LRC_Knob *vCorrs = NULL;
static long      nVCorr = 0;
static LRC_Bpm  *crossXBpms = NULL;       /* BPMs read for x (V-corr -> x) */
static long      nCrossXBpm = 0;
static LRC_Bpm  *crossYBpms = NULL;       /* BPMs read for y (H-corr -> y) */
static long      nCrossYBpm = 0;
/* Saved specs/patterns so they survive the compute->load round-trip. */
static char *crossHSpec = NULL;
static char *crossVSpec = NULL;
static char **crossXBpmNamePat = NULL;  static long nCrossXBpmNamePat = 0;
static char **crossXBpmTypePat = NULL;  static long nCrossXBpmTypePat = 0;
static char **crossYBpmNamePat = NULL;  static long nCrossYBpmNamePat = 0;
static char **crossYBpmTypePat = NULL;  static long nCrossYBpmTypePat = 0;
static double crossKickMag       = 1e-5;  /* probe kick magnitude (rad) */
static double crossMeasNoise     = 0.0;
static double etayChannelWeight  = 1.0;
static double crossChannelWeight = 0.0;
/* Workspace for find_closed_orbit() per probe + the readout buffer. */
static TRAJECTORY *crossClorbBase = NULL;
static TRAJECTORY *crossClorbPert = NULL;
/* Total rows of R: nBpm (etay) + nHCorr*nCrossYBpm (H->y) + nVCorr*nCrossXBpm (V->x). */
static long nRowsEtay = 0, nRowsCrossHy = 0, nRowsCrossVx = 0, nRowsTotal = 0;
/* Per-row weight (length nRowsTotal); applied multiplicatively to R and y
 * before SVD solve. */
static double *rowWeight = NULL;
/* Flat residual + perturbation buffers sized nRowsTotal. */
static double *yResidFlat = NULL;
static double *yPertFlat  = NULL;

/****************************************************************************/
/* Parse a steering specification of the form
 *     "TYPE1/ITEM1[/NAMEPATTERN1], TYPE2/ITEM2[/NAMEPATTERN2], ..."
 * and collect every beamline element matching some entry.  NAMEPATTERN
 * defaults to "*" (any name).  Returns the number of probe correctors found
 * and allocates *knobs.  Each entry of *knobs records the element together
 * with the index/pointer to the item parameter that will be perturbed.
 *
 * The same element may appear multiple times in the spec (once per item)
 * if it provides both H- and V-steering; the caller should pass distinct
 * specs for the two planes.  Within a single spec, an element matched by
 * multiple entries is still added once per matching entry, because the
 * second entry refers to a different ITEM (otherwise the user has written
 * a redundant spec; we do not de-duplicate). */
static long collectSteeringKnobs(LINE_LIST *beamline, const char *spec,
                                 LRC_Knob **knobs) {
  long count = 0, capacity = 0;
  char *buf, *tok, *save1 = NULL;
  *knobs = NULL;
  if (!spec || !*spec) return 0;
  cp_str(&buf, (char *)spec);
  /* Split entries on ',' (allow surrounding spaces). */
  for (tok = strtok_r(buf, ",", &save1); tok; tok = strtok_r(NULL, ",", &save1)) {
    while (isspace((unsigned char)*tok)) tok++;
    char *end = tok + strlen(tok);
    while (end > tok && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    if (!*tok) continue;
    /* Split on '/': type, item, optional namepattern */
    char *typeStr = tok;
    char *slash1 = strchr(typeStr, '/');
    if (!slash1)
      bombElegant("collectSteeringKnobs: each entry must be TYPE/ITEM[/NAMEPATTERN]", NULL);
    *slash1 = '\0';
    char *itemStr = slash1 + 1;
    char *slash2 = strchr(itemStr, '/');
    char *nameStr = "*";
    if (slash2) {
      *slash2 = '\0';
      nameStr = slash2 + 1;
    }
    while (isspace((unsigned char)*typeStr)) typeStr++;
    while (isspace((unsigned char)*itemStr)) itemStr++;
    while (isspace((unsigned char)*nameStr)) nameStr++;
    if (!*typeStr || !*itemStr)
      bombElegant("collectSteeringKnobs: empty TYPE or ITEM in entry", NULL);

    ELEMENT_LIST *eptr = beamline->elem;
    while (eptr) {
      long paramIndex;
      if (wild_match(entity_name[eptr->type], typeStr) &&
          wild_match(eptr->name, nameStr) &&
          (paramIndex = confirm_parameter(itemStr, eptr->type)) >= 0 &&
          entity_description[eptr->type].parameter[paramIndex].type == IS_DOUBLE) {
        if (count == capacity) {
          capacity = capacity ? 2 * capacity : 32;
          *knobs = SDDS_Realloc(*knobs, sizeof(**knobs) * capacity);
        }
        (*knobs)[count].elem = eptr;
        (*knobs)[count].paramIndex = paramIndex;
        (*knobs)[count].valuePtr = (double *)(eptr->p_elem +
            entity_description[eptr->type].parameter[paramIndex].offset);
        (*knobs)[count].initialValue = *(*knobs)[count].valuePtr;
        count++;
      }
      eptr = eptr->succ;
    }
  }
  free(buf);
  return count;
}

/****************************************************************************/

/* Local RMS helper kept here (small, not worth promoting). */
static double ccRmsValue(double *v, long n) {
  long i;
  double s = 0;
  if (n <= 0) return 0;
  for (i = 0; i < n; i++) s += v[i] * v[i];
  return sqrt(s / n);
}

/* Read eta_y at every BPM, optionally adding simulated measurement noise.
 * Signature kept compatible with the existing call sites in the iteration
 * loop; an LRC_ReaderFn-compatible trampoline below delegates here. */
static void readEtayAtBpms(LRC_Bpm *bpms, long nBpm, double *etay) {
  long i;
  for (i = 0; i < nBpm; i++) {
    if (!bpms[i].elem->twiss) {
      fprintf(stderr, "correct_coupling: BPM %s has no twiss data; run twiss_output first\n",
              bpms[i].elem->name);
      exitElegant(1);
    }
    etay[i] = bpms[i].elem->twiss->etay;
    if (measurement_noise > 0)
      etay[i] += gauss_rn_lim(0.0, measurement_noise, measurement_noise_cutoff, random_3);
  }
}

/* Integer positions in beamline->elem for each cross-plane BPM, set up at
 * cross-plane setup time so the closed-orbit centroids can be indexed
 * directly. */
static long *crossXBpmIdx = NULL;
static long *crossYBpmIdx = NULL;

/* Compute the integer position of `target` in the beamline elem list. */
static long bpmElemIndex(LINE_LIST *beamline, ELEMENT_LIST *target) {
  ELEMENT_LIST *e = beamline->elem;
  long i = 0;
  while (e) {
    if (e == target) return i;
    e = e->succ;
    i++;
  }
  return -1;
}

/* Make sure beamline->matrix exists; find_closed_orbit needs it. */
static void ensureBeamlineMatrix(LINE_LIST *beamline, RUN *run) {
  if (!beamline->matrix) {
    if (beamline->elem_twiss)
      beamline->matrix = full_matrix(beamline->elem_twiss, run, 1);
    else
      beamline->matrix = full_matrix(beamline->elem, run, 1);
  }
}

/* find_closed_orbit wrapper with reasonable defaults for the cross-plane
 * probe.  Returns 1 on success, 0 on failure. */
static long findClorbOrFail(LINE_LIST *beamline, RUN *run, TRAJECTORY *clorb) {
  ensureBeamlineMatrix(beamline, run);
  return find_closed_orbit(clorb,
                           1e-12, 1e-10, 200,
                           beamline, beamline->matrix, run,
                           0.0,    /* dp */
                           1,      /* start_from_recirc */
                           0,      /* fixed_length */
                           NULL,   /* starting_point */
                           0.5, 1.0, 0, NULL,
                           0);
}

/* Apply a probe kick to one corrector and rebuild its element matrix.  Returns
 * the saved (pre-kick) value of the item for later restoration. */
static double crossProbeApply(LRC_Knob *probe, double kick, RUN *run, LINE_LIST *beamline) {
  double k0 = *probe->valuePtr;
  *probe->valuePtr = k0 + kick;
  if (probe->elem->matrix) {
    free_matrices(probe->elem->matrix);
    free(probe->elem->matrix);
    probe->elem->matrix = NULL;
  }
  compute_matrix(probe->elem, run, NULL);
  if (beamline->links)
    assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
  return k0;
}

static void crossProbeRestore(LRC_Knob *probe, double k0, RUN *run, LINE_LIST *beamline) {
  *probe->valuePtr = k0;
  if (probe->elem->matrix) {
    free_matrices(probe->elem->matrix);
    free(probe->elem->matrix);
    probe->elem->matrix = NULL;
  }
  compute_matrix(probe->elem, run, NULL);
  if (beamline->links)
    assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
}

/* Read the cross-plane response vector into obs[].  Each (probe corrector,
 * target BPM) pair contributes one observable: the change in cross-plane
 * centroid per radian of kick, measured with a TWO-SIDED finite difference
 * around the current lattice state.  Two-sided is essential to cancel the
 * second-order coupling that a lattice with nonzero sextupole strength shows
 * even when fully decoupled at the linear level (V kick -> y orbit -> through
 * sextupole -> x' kick -> x orbit, which scales as kick^2).  Layout:
 *   obs[0 .. nHCorr*nCrossYBpm - 1]  : H-corr -> y at vertical BPMs
 *   obs[nHCorr*nCrossYBpm .. ]      : V-corr -> x at horizontal BPMs
 * within each block, packed as obs[corrIdx*nBpmInBlock + bpmIdx].
 * Optionally adds Gaussian noise per reading of sigma crossMeasNoise (m).
 *
 * This runs serially on the calling rank: the engine's outer-parallel split
 * means only the rank that owns the current knob calls the reader, so no
 * collective MPI is performed inside.  Bracketing with
 * parallelTrackingBasedMatrices=0 is the caller's responsibility (see
 * compute_orbcor_matrices1p() for the canonical pattern). */
static long readCrossPlaneAtBpms(LINE_LIST *beamline, RUN *run, double *obs) {
  long iC, iB;
  long baseOffset = nHCorr * nCrossYBpm;

  /* H-corrector -> y at cross_y_bpms (two-sided difference per probe). */
  for (iC = 0; iC < nHCorr; iC++) {
    double k0 = crossProbeApply(&hCorrs[iC], +crossKickMag, run, beamline);
    if (!findClorbOrFail(beamline, run, crossClorbPert)) {
      fprintf(stderr, "correct_coupling: +kick closed-orbit search failed (H probe %s)\n",
              hCorrs[iC].elem->name);
      crossProbeRestore(&hCorrs[iC], k0, run, beamline);
      return 0;
    }
    crossProbeRestore(&hCorrs[iC], k0 - crossKickMag, run, beamline);
    if (!findClorbOrFail(beamline, run, crossClorbBase)) {
      fprintf(stderr, "correct_coupling: -kick closed-orbit search failed (H probe %s)\n",
              hCorrs[iC].elem->name);
      crossProbeRestore(&hCorrs[iC], k0, run, beamline);
      return 0;
    }
    crossProbeRestore(&hCorrs[iC], k0, run, beamline);
    for (iB = 0; iB < nCrossYBpm; iB++) {
      long idx = crossYBpmIdx[iB];
      double dY = (computeMonitorReading(crossYBpms[iB].elem, 2, crossClorbPert[idx].centroid, 0)
                   - computeMonitorReading(crossYBpms[iB].elem, 2, crossClorbBase[idx].centroid, 0))
                  / (2.0 * crossKickMag);
      if (crossMeasNoise > 0)
        dY += gauss_rn_lim(0.0, crossMeasNoise / fabs(crossKickMag),
                           measurement_noise_cutoff, random_3);
      obs[iC * nCrossYBpm + iB] = dY;
    }
  }
  /* V-corrector -> x at cross_x_bpms (two-sided difference per probe). */
  for (iC = 0; iC < nVCorr; iC++) {
    double k0 = crossProbeApply(&vCorrs[iC], +crossKickMag, run, beamline);
    if (!findClorbOrFail(beamline, run, crossClorbPert)) {
      fprintf(stderr, "correct_coupling: +kick closed-orbit search failed (V probe %s)\n",
              vCorrs[iC].elem->name);
      crossProbeRestore(&vCorrs[iC], k0, run, beamline);
      return 0;
    }
    crossProbeRestore(&vCorrs[iC], k0 - crossKickMag, run, beamline);
    if (!findClorbOrFail(beamline, run, crossClorbBase)) {
      fprintf(stderr, "correct_coupling: -kick closed-orbit search failed (V probe %s)\n",
              vCorrs[iC].elem->name);
      crossProbeRestore(&vCorrs[iC], k0, run, beamline);
      return 0;
    }
    crossProbeRestore(&vCorrs[iC], k0, run, beamline);
    for (iB = 0; iB < nCrossXBpm; iB++) {
      long idx = crossXBpmIdx[iB];
      double dX = (computeMonitorReading(crossXBpms[iB].elem, 0, crossClorbPert[idx].centroid, 0)
                   - computeMonitorReading(crossXBpms[iB].elem, 0, crossClorbBase[idx].centroid, 0))
                  / (2.0 * crossKickMag);
      if (crossMeasNoise > 0)
        dX += gauss_rn_lim(0.0, crossMeasNoise / fabs(crossKickMag),
                           measurement_noise_cutoff, random_3);
      obs[baseOffset + iC * nCrossXBpm + iB] = dX;
    }
  }
  return 1;
}

/* Context for the combined reader: the engine needs a RUN* and LINE_LIST* to
 * compute closed orbits; pass them through ctx. */
typedef struct {
  RUN *run;
  LINE_LIST *beamline;
} CombinedReaderCtx;

/* LRC_ReaderFn-compatible trampoline used by LRC_buildResponseMatrix.  Reads
 * the η_y channel first (into the first nBpm slots), then optionally appends
 * the cross-plane channel.  Layout matches the per-row weight array set up at
 * setup time. */
static void etayReader(long nObs, double *obs, void *ctx) {
  (void)nObs;
  readEtayAtBpms(bpms, nBpm, obs);
  if (crossChannelWeight > 0 && (nHCorr + nVCorr) > 0) {
    CombinedReaderCtx *cctx = (CombinedReaderCtx *)ctx;
    if (!readCrossPlaneAtBpms(cctx->beamline, cctx->run, obs + nBpm)) {
      /* Fill cross-plane block with zeros on failure so downstream RMS does
       * not contain NaNs; the SVD will still see the baseline state. */
      long i;
      for (i = nBpm; i < nObs; i++) obs[i] = 0;
    }
  }
}

/****************************************************************************/
/* Free anything previously allocated by setupCrossPlaneState; safe to call
 * even when nothing was allocated. */
static void freeCrossPlaneState(void) {
  if (hCorrs)        { free(hCorrs);        hCorrs = NULL; }
  if (vCorrs)        { free(vCorrs);        vCorrs = NULL; }
  if (crossXBpms)    { free(crossXBpms);    crossXBpms = NULL; }
  if (crossYBpms)    { free(crossYBpms);    crossYBpms = NULL; }
  if (crossXBpmIdx)  { free(crossXBpmIdx);  crossXBpmIdx = NULL; }
  if (crossYBpmIdx)  { free(crossYBpmIdx);  crossYBpmIdx = NULL; }
  if (crossClorbBase){ free(crossClorbBase);crossClorbBase = NULL; }
  if (crossClorbPert){ free(crossClorbPert);crossClorbPert = NULL; }
  nHCorr = nVCorr = nCrossXBpm = nCrossYBpm = 0;
  nRowsCrossHy = nRowsCrossVx = 0;
}

/* Walk the beamline, collect H/V probe correctors and cross-plane BPMs,
 * allocate the closed-orbit working trajectories.  Must be called after the
 * skew BPMs are known (nBpm) so the total row count is consistent. */
static void setupCrossPlaneState(LINE_LIST *beamline) {
  freeCrossPlaneState();
  /* Probe corrector lists.  At least one of (h, v) must be non-empty when the
   * cross channel is enabled; we let the caller decide what to do with that. */
  if (crossHSpec) {
    nHCorr = collectSteeringKnobs(beamline, crossHSpec, &hCorrs);
    if (nHCorr == 0)
      bombElegantVA("correct_coupling: cross_h_steering=\"%s\" matched zero correctors "
                    "in the beamline", crossHSpec);
  }
  if (crossVSpec) {
    nVCorr = collectSteeringKnobs(beamline, crossVSpec, &vCorrs);
    if (nVCorr == 0)
      bombElegantVA("correct_coupling: cross_v_steering=\"%s\" matched zero correctors "
                    "in the beamline", crossVSpec);
  }
  /* Cross-plane BPMs.  Vertical-reading BPMs for the H->y block; horizontal-
   * reading BPMs for the V->x block. */
  nCrossYBpm = LRC_collectBpms(beamline, crossYBpmNamePat, nCrossYBpmNamePat,
                               crossYBpmTypePat, nCrossYBpmTypePat, &crossYBpms);
  nCrossXBpm = LRC_collectBpms(beamline, crossXBpmNamePat, nCrossXBpmNamePat,
                               crossXBpmTypePat, nCrossXBpmTypePat, &crossXBpms);
  if (nHCorr > 0 && nCrossYBpm == 0)
    bombElegant("correct_coupling: H-probe correctors selected but no vertical-reading "
                "BPMs matched cross_y_bpm_name_pattern/cross_y_bpm_type_pattern", NULL);
  if (nVCorr > 0 && nCrossXBpm == 0)
    bombElegant("correct_coupling: V-probe correctors selected but no horizontal-reading "
                "BPMs matched cross_x_bpm_name_pattern/cross_x_bpm_type_pattern", NULL);
  /* Cache the integer position of each cross-plane BPM in the beamline. */
  long i;
  if (nCrossYBpm > 0) {
    crossYBpmIdx = tmalloc(sizeof(*crossYBpmIdx) * nCrossYBpm);
    for (i = 0; i < nCrossYBpm; i++) {
      crossYBpmIdx[i] = bpmElemIndex(beamline, crossYBpms[i].elem);
      if (crossYBpmIdx[i] < 0)
        bombElegant("correct_coupling: cross-plane Y-BPM not found in beamline", NULL);
    }
  }
  if (nCrossXBpm > 0) {
    crossXBpmIdx = tmalloc(sizeof(*crossXBpmIdx) * nCrossXBpm);
    for (i = 0; i < nCrossXBpm; i++) {
      crossXBpmIdx[i] = bpmElemIndex(beamline, crossXBpms[i].elem);
      if (crossXBpmIdx[i] < 0)
        bombElegant("correct_coupling: cross-plane X-BPM not found in beamline", NULL);
    }
  }
  /* Trajectories for closed-orbit search. */
  crossClorbBase = tmalloc(sizeof(*crossClorbBase) * (beamline->n_elems + 1));
  crossClorbPert = tmalloc(sizeof(*crossClorbPert) * (beamline->n_elems + 1));
  nRowsCrossHy = nHCorr * nCrossYBpm;
  nRowsCrossVx = nVCorr * nCrossXBpm;
}

/* Parse a whitespace-separated list of floating-point values; mirror of the
 * helper in correct_lattice.c. */
static double *parseDoubleList(const char *input0, long *nOut, const char *ctx) {
  *nOut = 0;
  if (input0 == NULL) return NULL;
  char *input;
  cp_str(&input, (char *)input0);
  char *ptr;
  double *vals = NULL;
  long n = 0;
  while ((ptr = get_token(input))) {
    double v;
    if (sscanf(ptr, "%le", &v) != 1)
      bombElegantVA("correct_coupling: %s contains non-numeric token \"%s\"", ctx, ptr);
    vals = SDDS_Realloc(vals, sizeof(*vals) * (n + 1));
    vals[n++] = v;
  }
  free(input);
  *nOut = n;
  return vals;
}

static int matchesExclude(const char *name) {
  long k;
  for (k = 0; k < nExcludePatterns; k++)
    if (wild_match((char *)name, excludePatterns[k])) return 1;
  return 0;
}

/* Walk the beamline; for each element not in the exclude list, find the first
 * family pattern matching its name and (if the element exposes the family's
 * item parameter as a double) record it as a skew knob.  Allocates knobs[]
 * and knobFamily[].  Returns the count. */
static long collectKnobsFromFamilies(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr = beamline->elem;
  long cap = 0;
  nSkew = 0;
  if (knobs)      { free(knobs);      knobs = NULL; }
  if (knobFamily) { free(knobFamily); knobFamily = NULL; }
  while (eptr) {
    long fam;
    for (fam = 0; fam < nCorrPatterns; fam++) {
      if (!wild_match(eptr->name, corrPatterns[fam])) continue;
      if (matchesExclude(eptr->name)) break;
      long paramIndex = confirm_parameter(itemsList[fam], eptr->type);
      if (paramIndex < 0 ||
          entity_description[eptr->type].parameter[paramIndex].type != IS_DOUBLE)
        break;
      if (nSkew == cap) {
        cap = cap ? 2 * cap : 32;
        knobs      = SDDS_Realloc(knobs,      sizeof(*knobs)      * cap);
        knobFamily = SDDS_Realloc(knobFamily, sizeof(*knobFamily) * cap);
      }
      knobs[nSkew].elem        = eptr;
      knobs[nSkew].paramIndex  = paramIndex;
      knobs[nSkew].valuePtr    = (double *)(eptr->p_elem +
          entity_description[eptr->type].parameter[paramIndex].offset);
      knobs[nSkew].initialValue = *knobs[nSkew].valuePtr;
      knobFamily[nSkew] = fam;
      nSkew++;
      break;
    }
    eptr = eptr->succ;
  }
  return nSkew;
}

/* Free the per-knob owned strings in knobItem[] and the array itself. */
static void freeKnobItem(void) {
  if (!knobItem) return;
  long j;
  for (j = 0; j < nSkew; j++)
    if (knobItem[j]) free(knobItem[j]);
  free(knobItem);
  knobItem = NULL;
}

/* Build per-knob (item, lower, upper) arrays from knobFamily + family lists.
 * knobItem[] owns its strings (cp_str'd) so per-knob items survive across
 * setup calls.  When no family list is supplied but knobItem[] is already
 * populated by load_coupling_response_matrix, it is preserved as-is. */
static void buildPerKnobItemAndBounds(void) {
  long j;
  if (knobLower) { free(knobLower); knobLower = NULL; }
  if (knobUpper) { free(knobUpper); knobUpper = NULL; }
  if (nSkew == 0) { freeKnobItem(); return; }
  if (nCorrPatterns > 0 && knobFamily) {
    freeKnobItem();
    knobItem = tmalloc(sizeof(*knobItem) * nSkew);
    for (j = 0; j < nSkew; j++) cp_str(&knobItem[j], itemsList[knobFamily[j]]);
    if (lowerList) {
      knobLower = tmalloc(sizeof(*knobLower) * nSkew);
      for (j = 0; j < nSkew; j++) knobLower[j] = lowerList[knobFamily[j]];
    }
    if (upperList) {
      knobUpper = tmalloc(sizeof(*knobUpper) * nSkew);
      for (j = 0; j < nSkew; j++) knobUpper[j] = upperList[knobFamily[j]];
    }
  } else if (knobItem == NULL) {
    knobItem = tmalloc(sizeof(*knobItem) * nSkew);
    for (j = 0; j < nSkew; j++) cp_str(&knobItem[j], skewItem ? skewItem : "K1");
  }
  /* else: preloaded per-knob knobItem[] preserved as-is */
}

/* When a preloaded inventory exists AND the user supplied a family list,
 * assign each preloaded knob to its first matching family; bomb on
 * item disagreements (the response-matrix columns are bound to skewItem). */
static void assignFamiliesToPreloadedKnobs(void) {
  long j, k;
  if (knobFamily) { free(knobFamily); knobFamily = NULL; }
  if (nSkew == 0 || nCorrPatterns == 0) return;
  knobFamily = tmalloc(sizeof(*knobFamily) * nSkew);
  for (j = 0; j < nSkew; j++) {
    knobFamily[j] = -1;
    for (k = 0; k < nCorrPatterns; k++) {
      if (!wild_match(knobs[j].elem->name, corrPatterns[k])) continue;
      if (matchesExclude(knobs[j].elem->name))
        bombElegantVA("correct_coupling: preloaded knob %s is matched by exclude pattern "
                      "but cannot be removed once the response matrix is built; "
                      "rebuild the matrix without this element",
                      knobs[j].elem->name);
      if (skewItem && itemsList && strcmp(skewItem, itemsList[k]) != 0)
        bombElegantVA("correct_coupling: preloaded knob %s would be assigned to family "
                      "\"%s\" whose item \"%s\" disagrees with the preloaded item \"%s\"",
                      knobs[j].elem->name, corrPatterns[k], itemsList[k], skewItem);
      knobFamily[j] = k;
      break;
    }
  }
}

/* Collect knob/BPM lists from the beamline and allocate the response-matrix
 * working buffers. Idempotent: subsequent calls are no-ops once knobs/bpms have
 * been collected.  Returns 1 on success; bombs if no matching knobs/BPMs. */
static long collectAndAllocate(LINE_LIST *beamline) {
  if (knobs == NULL) {
    if (nCorrPatterns == 0)
      bombElegant("correct_coupling: correction_elements must be supplied "
                  "(unless compute_coupling_response_matrix or "
                  "load_coupling_response_matrix has been issued)", NULL);
    collectKnobsFromFamilies(beamline);
    nBpm = LRC_collectBpms(beamline, bpmName, nBpmName, bpmType, nBpmType, &bpms);
    if (nSkew == 0)
      bombElegant("correct_coupling: no knobs matched correction_elements", NULL);
    if (nBpm == 0)
      bombElegant("correct_coupling: no BPMs matched bpm_name_pattern/bpm_type_pattern", NULL);
    buildPerKnobItemAndBounds();
  } else {
    assignFamiliesToPreloadedKnobs();
    buildPerKnobItemAndBounds();
    if (nCorrPatterns > 0 && (knobLower || knobUpper)) {
      long j;
      if (knobLower)
        for (j = 0; j < nSkew; j++)
          if (knobFamily[j] < 0) knobLower[j] = -DBL_MAX;
      if (knobUpper)
        for (j = 0; j < nSkew; j++)
          if (knobFamily[j] < 0) knobUpper[j] = DBL_MAX;
    }
  }
  /* Cross-plane setup runs alongside the skew setup so the flat buffers have
   * the right size.  Only run it when the channel is enabled, otherwise the
   * cross-row counts stay zero and the flat layout collapses to plain etay. */
  if (crossChannelWeight > 0 && hCorrs == NULL && vCorrs == NULL)
    setupCrossPlaneState(beamline);

  nRowsEtay   = nBpm;
  nRowsTotal  = nRowsEtay + nRowsCrossHy + nRowsCrossVx;

  /* Allocate any working buffer that's still NULL (load_coupling_response_matrix
   * may have populated some but not all -- e.g. it sets etay/etayPert/dK and R
   * but not yResidFlat/yPertFlat/rowWeight).  Each buffer is checked
   * individually so the union of "populated by load_ or by collectKnobs"
   * is fully provisioned before do_correct_coupling reads from any of them. */
  if (etay        == NULL) etay       = tmalloc(sizeof(*etay)       * nBpm);
  if (etayPert    == NULL) etayPert   = tmalloc(sizeof(*etayPert)   * nBpm);
  if (dK          == NULL) dK         = tmalloc(sizeof(*dK)         * nSkew);
  if (yResidFlat  == NULL) yResidFlat = tmalloc(sizeof(*yResidFlat) * nRowsTotal);
  if (yPertFlat   == NULL) yPertFlat  = tmalloc(sizeof(*yPertFlat)  * nRowsTotal);
  if (rowWeight   == NULL) {
    rowWeight = tmalloc(sizeof(*rowWeight) * nRowsTotal);
    long i;
    for (i = 0; i < nRowsEtay; i++) rowWeight[i] = etayChannelWeight;
    for (i = nRowsEtay; i < nRowsTotal; i++) rowWeight[i] = crossChannelWeight;
  }
  if (R == NULL) {
#if USE_MPI
    if (myid == 0)
      R = (double **)czarray_2d(sizeof(**R), nRowsTotal, nSkew);
#else
    R = (double **)czarray_2d(sizeof(**R), nRowsTotal, nSkew);
#endif
  }
  return 1;
}

/* Build the response matrix R[i,j] = d observable_i / d K1_j via the engine
 * (one-sided finite difference + MPI assembly) and, if the response_file is
 * open, write the matrix to it as one page tagged with iterTag (-1 for the
 * setup-time build, >=0 for in-loop builds).
 *
 * Rows are laid out per the flat observable convention:
 *   [0, nRowsEtay):                  etay at vertical BPMs
 *   [nRowsEtay, +nRowsCrossHy):      H-corrector -> y at vertical BPMs
 *   [...,         +nRowsCrossVx):    V-corrector -> x at horizontal BPMs */
static void buildResponseMatrix(RUN *run, LINE_LIST *beamline, long iterTag, double perturbation) {
  long i, j;
  CombinedReaderCtx ctx;
  ctx.run = run;
  ctx.beamline = beamline;

  /* Bracket the engine call with parallelTrackingBasedMatrices = 0, matching
   * compute_orbcor_matrices1p()'s pattern.  Two reasons:
   *   (1) the cross-plane reader, when active, calls find_closed_orbit per
   *       probe corrector, which can invoke collective tracking;
   *   (2) compute_matrix() inside LRC_retwiss (and inside the per-knob apply
   *       in do_correct_coupling) can also invoke collective tracking for
   *       element types that compute their matrix by tracking
   *       (KQUAD/TRACKING_MATRIX, APPLE, KICKMAP, high-order CSBEND, ...).
   * Either path expects every rank to be at the same call; the engine's
   * outer-parallel split puts only the knob's owning rank inside, so the
   * collective tracking would deadlock with no peers.  Saving and clearing
   * the flag avoids it without affecting any pure-analytic element. */
  long parTrackSave = parallelTrackingBasedMatrices;
  parallelTrackingBasedMatrices = 0;
  if (verbosity>2) {
    printf("   building coupling response matrix\n");
    fflush(stdout);
  }
  LRC_buildResponseMatrix(run, beamline, knobs, nSkew, nRowsTotal,
                          etayReader, &ctx, perturbation, R);
  parallelTrackingBasedMatrices = parTrackSave;
  responseValid = 1;
  if (verbosity > 1) {
    if (iterTag < 0)
      printf("  response matrix built in setup\n");
    else
      printf("  response matrix built before iteration %ld\n", iterTag);
    fflush(stdout);
  }

  if (SDDSresponseInit) {
    long row = 0;
    if (!SDDS_StartPage(&SDDSresponse, nBpm * nSkew) ||
        !SDDS_SetParameters(&SDDSresponse, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Iteration", iterTag, NULL))
      SDDS_Bomb("correct_coupling: error writing response_file page");
    for (i = 0; i < nBpm; i++)
      for (j = 0; j < nSkew; j++) {
        if (!SDDS_SetRowValues(&SDDSresponse, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                               "BPMName",   bpms[i].elem->name,
                               "sBPM",      bpms[i].elem->end_pos,
                               "SkewName",  knobs[j].elem->name,
                               "sSkew",     knobs[j].elem->end_pos,
                               "dEtayDK1",  R[i][j], NULL))
          SDDS_Bomb("correct_coupling: error setting response_file row");
        row++;
      }
    if (!SDDS_WritePage(&SDDSresponse))
      SDDS_Bomb("correct_coupling: error writing response_file page");
  }
}

/****************************************************************************/

void setup_correct_coupling(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&correct_coupling, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &correct_coupling);

  if (n_iterations < 0)
    bombElegant("correct_coupling: n_iterations must be >= 0", NULL);
  if (correction_fraction <= 0 || correction_fraction > 1)
    bombElegant("correct_coupling: correction_fraction must be in (0, 1]", NULL);
  if (change_tolerance < 0 || change_tolerance >= 1)
    bombElegant("correct_coupling: change_tolerance must be in [0, 1)", NULL);
  if (response_perturbation <= 0)
    bombElegant("correct_coupling: response_perturbation must be > 0", NULL);
  if (svd_threshold < 0)
    bombElegant("correct_coupling: svd_threshold must be >= 0", NULL);
  if (n_singular_values < 0)
    bombElegant("correct_coupling: n_singular_values must be >= 0", NULL);
  if (measurement_noise < 0)
    bombElegant("correct_coupling: measurement_noise must be >= 0", NULL);
  if (measurement_noise > 0 && measurement_noise_cutoff <= 0)
    bombElegant("correct_coupling: measurement_noise_cutoff must be > 0 when measurement_noise > 0", NULL);
  if (etay_weight < 0)
    bombElegant("correct_coupling: etay_weight must be >= 0", NULL);
  if (cross_response_weight < 0)
    bombElegant("correct_coupling: cross_response_weight must be >= 0", NULL);
  if (cross_response_weight > 0 && cross_steering_kick <= 0)
    bombElegant("correct_coupling: cross_steering_kick must be > 0 when cross_response_weight > 0", NULL);
  if (cross_measurement_noise < 0)
    bombElegant("correct_coupling: cross_measurement_noise must be >= 0", NULL);

  /* Family-list parsing (same convention as &correct_lattice and &correct_tunes).
   *
   * When a matrix was preloaded by load_/compute_coupling_response_matrix and
   * carried CorrectionElements/CorrectionItems metadata, corrPatterns/itemsList
   * are already populated.  Preserve them when the user didn't supply
   * correction_elements/items; when the user did supply, validate against the
   * preloaded values and bomb on disagreement (so the user can't accidentally
   * specify lower_limits aligned to a different family count than the matrix
   * was built with). */
  LRC_freePatternList(&excludePatterns,  &nExcludePatterns);
  if (lowerList) { free(lowerList); lowerList = NULL; }
  if (upperList) { free(upperList); upperList = NULL; }
  nLowerList = nUpperList = 0;

  if (correction_elements && *correction_elements) {
    /* User supplied correction_elements on this command. */
    if (responseValid && corrPatternStr && strcmp(corrPatternStr, correction_elements) != 0)
      bombElegantVA("correct_coupling: correction_elements=\"%s\" on this command "
                    "disagrees with the loaded matrix's CorrectionElements=\"%s\"; "
                    "either omit correction_elements (to inherit) or rebuild the "
                    "matrix with the desired family list",
                    correction_elements, corrPatternStr);
    /* Free and re-populate (idempotent when string matches preloaded). */
    LRC_freePatternList(&corrPatterns, &nCorrPatterns);
    if (corrPatternStr) { free(corrPatternStr); corrPatternStr = NULL; }
    LRC_freePatternList(&itemsList, &nItemsList);
    cp_str(&corrPatternStr, correction_elements);
    corrPatterns = addPatterns(&nCorrPatterns, corrPatternStr);
    if (nCorrPatterns == 0)
      bombElegant("correct_coupling: correction_elements parsed to zero patterns", NULL);
    itemsList = tmalloc(sizeof(*itemsList) * nCorrPatterns);
    nItemsList = nCorrPatterns;
    if (items && *items) {
      char *icopy;
      cp_str(&icopy, items);
      char *tok;
      long ii = 0;
      while ((tok = get_token(icopy)) && ii < nCorrPatterns) {
        cp_str(&itemsList[ii], tok);
        ii++;
      }
      free(icopy);
      if (ii != nCorrPatterns)
        bombElegantVA("correct_coupling: items has %ld entries but correction_elements has %ld",
                      ii, nCorrPatterns);
    } else {
      long ii;
      for (ii = 0; ii < nCorrPatterns; ii++)
        cp_str(&itemsList[ii], "K1");
    }
  } else if (items && *items) {
    /* User supplied items but not correction_elements.  Allowed only when
     * a preloaded family list exists; validate the count and contents. */
    if (!responseValid || nItemsList == 0)
      bombElegant("correct_coupling: items supplied without correction_elements; "
                  "requires a preloaded matrix that carries the family list", NULL);
    char *icopy;
    cp_str(&icopy, items);
    char *tok;
    long ii = 0;
    while ((tok = get_token(icopy))) {
      if (ii >= nItemsList || strcmp(tok, itemsList[ii]) != 0) {
        bombElegantVA("correct_coupling: items on this command disagrees with the "
                      "loaded matrix's CorrectionItems (entry %ld: \"%s\" vs \"%s\")",
                      ii, tok, ii < nItemsList ? itemsList[ii] : "(none)");
      }
      ii++;
    }
    if (ii != nItemsList)
      bombElegantVA("correct_coupling: items has %ld entries but loaded matrix has %ld families",
                    ii, nItemsList);
    free(icopy);
  } else if (!responseValid) {
    bombElegant("correct_coupling: correction_elements must be supplied "
                "(unless compute_coupling_response_matrix or "
                "load_coupling_response_matrix has been issued)", NULL);
  }

  /* Parse lower_limits/upper_limits and exclude regardless of whether the
   * family list came from this namelist or from a preload.  Length validation
   * uses nCorrPatterns (the effective family count). */
  lowerList = parseDoubleList(lower_limits, &nLowerList, "lower_limits");
  upperList = parseDoubleList(upper_limits, &nUpperList, "upper_limits");
  if (lowerList && nLowerList != nCorrPatterns)
    bombElegantVA("correct_coupling: lower_limits has %ld entries but correction_elements has %ld",
                  nLowerList, nCorrPatterns);
  if (upperList && nUpperList != nCorrPatterns)
    bombElegantVA("correct_coupling: upper_limits has %ld entries but correction_elements has %ld",
                  nUpperList, nCorrPatterns);
  if (lowerList && upperList) {
    long ii;
    for (ii = 0; ii < nCorrPatterns; ii++)
      if (lowerList[ii] > upperList[ii])
        bombElegantVA("correct_coupling: lower_limits[%ld]=%le > upper_limits[%ld]=%le",
                      ii, lowerList[ii], ii, upperList[ii]);
  }
  if (exclude && *exclude) {
    char *ecopy;
    cp_str(&ecopy, exclude);
    excludePatterns = addPatterns(&nExcludePatterns, ecopy);
    free(ecopy);
  }

  /* BPM patterns only needed when we're going to build the inventory ourselves. */
  if (!responseValid) {
    LRC_freePatternList(&bpmName, &nBpmName);
    LRC_freePatternList(&bpmType, &nBpmType);
    bpmName = addPatterns(&nBpmName, bpm_name_pattern);
    bpmType = addPatterns(&nBpmType, bpm_type_pattern);
  }

  /* Cross-plane channel weights are always taken from this namelist (they
   * default to off, so silent enable is impossible). */
  etayChannelWeight  = etay_weight;
  crossChannelWeight = cross_response_weight;
  /* Honor the per-correction reassert flag so vary_beamline()'s per-step
   * restoration knows what to do for coupling. */
  ccStash.reassert   = reset_correctors_each_step ? 1 : 0;
  /* Probe-kick and noise: override if the user gave a value (cross_steering_kick
   * always has a positive default, so it's always set; crossMeasNoise too). */
  crossKickMag       = cross_steering_kick;
  crossMeasNoise     = cross_measurement_noise;

  /* For the spec strings and BPM patterns: override the preloaded values only
   * when the user explicitly supplies them on this namelist (NULL = inherit).
   * This matches the bind_name_pattern convention from correct_lattice. */
  short specChanged = 0;
  if (cross_h_steering != NULL) {
    if (crossHSpec) free(crossHSpec);
    crossHSpec = NULL;
    if (*cross_h_steering) cp_str(&crossHSpec, cross_h_steering);
    specChanged = 1;
  }
  if (cross_v_steering != NULL) {
    if (crossVSpec) free(crossVSpec);
    crossVSpec = NULL;
    if (*cross_v_steering) cp_str(&crossVSpec, cross_v_steering);
    specChanged = 1;
  }
  if (cross_x_bpm_name_pattern != NULL) {
    LRC_freePatternList(&crossXBpmNamePat, &nCrossXBpmNamePat);
    crossXBpmNamePat = addPatterns(&nCrossXBpmNamePat, cross_x_bpm_name_pattern);
    specChanged = 1;
  }
  if (cross_x_bpm_type_pattern != NULL) {
    LRC_freePatternList(&crossXBpmTypePat, &nCrossXBpmTypePat);
    crossXBpmTypePat = addPatterns(&nCrossXBpmTypePat, cross_x_bpm_type_pattern);
    specChanged = 1;
  }
  if (cross_y_bpm_name_pattern != NULL) {
    LRC_freePatternList(&crossYBpmNamePat, &nCrossYBpmNamePat);
    crossYBpmNamePat = addPatterns(&nCrossYBpmNamePat, cross_y_bpm_name_pattern);
    specChanged = 1;
  }
  if (cross_y_bpm_type_pattern != NULL) {
    LRC_freePatternList(&crossYBpmTypePat, &nCrossYBpmTypePat);
    crossYBpmTypePat = addPatterns(&nCrossYBpmTypePat, cross_y_bpm_type_pattern);
    specChanged = 1;
  }
  /* For the cross-plane BPM TYPE patterns specifically: namelist defaults are
   * NULL (so that omitting them doesn't trip specChanged and clobber a
   * preloaded matrix), but the historical defaults "MONI HMON" / "MONI VMON"
   * are still what we want when the user is building the cross channel
   * directly here (i.e., no preload at all) and didn't supply patterns. */
  if (crossXBpmTypePat == NULL && nCrossXBpmTypePat == 0 && cross_response_weight > 0)
    crossXBpmTypePat = addPatterns(&nCrossXBpmTypePat, (char *)"MONI HMON");
  if (crossYBpmTypePat == NULL && nCrossYBpmTypePat == 0 && cross_response_weight > 0)
    crossYBpmTypePat = addPatterns(&nCrossYBpmTypePat, (char *)"MONI VMON");

  /* Validate that the cross-plane channel has something to work with: either
   * the user supplied (or a prior namelist set) spec strings here, or a
   * prior compute_/load_coupling_response_matrix populated the corrector
   * inventory.  This check follows the inheritance logic above so loaded
   * specs satisfy it without re-supplying. */
  if (cross_response_weight > 0 &&
      crossHSpec == NULL && crossVSpec == NULL &&
      hCorrs == NULL && vCorrs == NULL)
    bombElegant("correct_coupling: at least one of cross_h_steering / cross_v_steering "
                "must be supplied (or carried by a prior compute_/load_coupling_response_matrix) "
                "when cross_response_weight > 0", NULL);
  /* If the user changed any spec, invalidate the cached matrix + buffers so
   * the next collectAndAllocate rebuilds with the new inventory.  Otherwise
   * preserve whatever load_/compute_coupling_response_matrix set up. */
  if (specChanged) {
    freeCrossPlaneState();
    responseValid = 0;
    if (etay) {
      free(etay);     etay = NULL;
      free(etayPert); etayPert = NULL;
      free(dK);       dK = NULL;
      free(yResidFlat); yResidFlat = NULL;
      free(yPertFlat);  yPertFlat = NULL;
      free(rowWeight);  rowWeight = NULL;
#if USE_MPI
      if (R && myid == 0) { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#else
      if (R)              { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#endif
    }
  }

  /* Output file setup */
#if USE_MPI
  if (myid==0) {
#endif    
  if (SDDSstrengthLogInit)    { SDDS_Terminate(&SDDSstrengthLog);    SDDSstrengthLogInit = 0; }
  if (SDDSetayInit)      { SDDS_Terminate(&SDDSetay);      SDDSetayInit = 0; }
  if (SDDSresponseInit)  { SDDS_Terminate(&SDDSresponse);  SDDSresponseInit = 0; }
  if (SDDSrmsLogInit)    { SDDS_Terminate(&SDDSrmsLog);    SDDSrmsLogInit = 0; }
  
  if (strength_log) {
    char *fn = compose_filename(strength_log, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSstrengthLog, SDDS_BINARY, 0,
                                      "Element parameter changes applied by correct_coupling",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSstrengthLog, "ElementName", NULL, NULL,
                          "Skew quadrupole element name", NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "ElementOccurence", NULL, NULL,
                          "Occurrence number of the element", NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "s", "s", "m",
                          "Distance from start of beamline", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "ElementParameter", NULL, NULL,
                          "Name of the element parameter being adjusted",
                          NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "PreviousParameterValue", NULL, NULL,
                          "Element parameter value before this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "DeltaParameterValue", NULL, NULL,
                          "Change applied to the element parameter this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSstrengthLog, "ParameterValue", NULL, NULL,
                          "Element parameter value after this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_DefineSimpleParameter(&SDDSstrengthLog, "Iteration", NULL, SDDS_LONG) ||
        SDDS_DefineParameter(&SDDSstrengthLog, "Stage", NULL, NULL,
                             "uncorrected for all iterations except the last; corrected for the final iteration",
                             NULL, SDDS_STRING, NULL) < 0 ||
        !SDDS_WriteLayout(&SDDSstrengthLog)) {
      fprintf(stderr, "correct_coupling: unable to set up strength_log %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSstrengthLogInit = 1;
  }

  if (etay_file) {
    char *fn = compose_filename(etay_file, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSetay, SDDS_BINARY, 0,
                                      "Vertical dispersion at BPMs during correct_coupling",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSetay, "ElementName", NULL, NULL,
                          "BPM element name", NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSetay, "ElementOccurence", NULL, NULL,
                          "Occurrence number", NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDSetay, "s", "s", "m",
                          "Distance from start of beamline", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSetay, "etay", "$gc$r$by$n", "m",
                          "Vertical dispersion at this BPM", NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_DefineSimpleParameter(&SDDSetay, "Iteration", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDSetay, "etayRMS", "m", SDDS_DOUBLE) ||
        !SDDS_WriteLayout(&SDDSetay)) {
      fprintf(stderr, "correct_coupling: unable to set up etay_file %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSetayInit = 1;
  }

  if (response_file) {
    char *fn = compose_filename(response_file, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSresponse, SDDS_BINARY, 0,
                                      "Response matrix d eta_y / d K1 used in correct_coupling",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSresponse, "BPMName", NULL, NULL,
                          "BPM element name (observation point)", NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "sBPM", "s$bBPM$n", "m",
                          "BPM position", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "SkewName", NULL, NULL,
                          "Skew quadrupole element name (knob)", NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "sSkew", "s$bskew$n", "m",
                          "Skew quadrupole position", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "dEtayDK1", NULL, "m$a3$n",
                          "Response: d eta_y(BPM) / d K1(skew)", NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_DefineSimpleParameter(&SDDSresponse, "Iteration", NULL, SDDS_LONG) ||
        !SDDS_WriteLayout(&SDDSresponse)) {
      fprintf(stderr, "correct_coupling: unable to set up response_file %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSresponseInit = 1;
  }

  if (rms_log) {
    char *fn = compose_filename(rms_log, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSrmsLog, SDDS_BINARY, 0,
                                      "Per-iteration RMS residuals from correct_coupling",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSrmsLog, "Iteration", NULL, NULL,
                          "Iteration index within this correction call "
                          "(0 = initial state, k+1 = state after iteration k)",
                          NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "EtayRms", NULL, "m",
                          "RMS of etay at the vertical BPMs (unweighted)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "CrossRms", NULL, "m/rad",
                          "RMS of the cross-plane response (m per rad of probe kick); "
                          "0 when the cross-plane channel is off",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "WeightedRms", NULL, NULL,
                          "RMS of the weighted residual the SVD solver drives down",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineParameter(&SDDSrmsLog, "EtayWeight", NULL, NULL,
                             "Per-channel weight applied to etay residuals",
                             NULL, SDDS_DOUBLE, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSrmsLog, "CrossResponseWeight", NULL, NULL,
                             "Per-channel weight applied to cross-plane residuals",
                             NULL, SDDS_DOUBLE, NULL) < 0 ||
        !SDDS_WriteLayout(&SDDSrmsLog)) {
      fprintf(stderr, "correct_coupling: unable to set up rms_log %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSrmsLogInit = 1;
  }
#if USE_MPI
  }
#endif

  /* Populate the knob/BPM inventory at setup time regardless of the
   * use_perturbed_matrix setting, so that the per-step corrector-reassertion
   * machinery (correct_coupling_save_correctors) can snapshot the SQ values
   * on the very first vary_beamline() tick.  Without this, use_perturbed_matrix>0
   * leaves knobs[] == NULL until do_correct_coupling first runs (i.e. inside
   * step 1), the save hook becomes a no-op, and step 2 ends up inheriting
   * step 1's converged SQ values instead of restoring the baseline. */
  if (!responseValid)
    collectAndAllocate(beamline);

  /* If use_perturbed_matrix=0 (default) and no matrix has been preloaded by
   * compute_/load_coupling_response_matrix, build the response matrix once
   * now using the lattice state as-of namelist parse time.  With
   * use_perturbed_matrix>0, the matrix is (re)built inside do_correct_coupling
   * -- following the pattern of chrom.c. */
  if (!responseValid && use_perturbed_matrix == 0) {
    /* Ensure twiss is current before perturbing knobs. */
    LRC_retwiss(run, beamline, NULL);
    buildResponseMatrix(run, beamline, -1, response_perturbation);
  }

  initialized = 1;
}

/****************************************************************************/

/* Weighted RMS of a length-n vector using a per-row weight vector w.
 * Returns sqrt(sum_i (w_i * v_i)^2 / n). */
static double ccWeightedRms(const double *v, const double *w, long n) {
  long i;
  double s = 0;
  if (n <= 0) return 0;
  for (i = 0; i < n; i++) {
    double wi = w[i] * v[i];
    s += wi * wi;
  }
  return sqrt(s / n);
}

long do_correct_coupling(RUN *run, LINE_LIST *beamline) {
  long iter, i, j;
  double rms0, rms;
  double rmsEtay, rmsCross;
  double minSV, maxSV;
  long nUsedSV;
  CombinedReaderCtx readerCtx = { run, beamline };
  if (!initialized)
    return 0;

  /* Bracket the entire routine with parallelTrackingBasedMatrices = 0 (same
   * reasoning as in buildResponseMatrix above): both the optional cross-plane
   * reader's find_closed_orbit and the per-iteration compute_matrix() invoked
   * when applying corrections can hit tracking-based-matrix elements that
   * otherwise expect collective MPI participation from all ranks. */
  long parTrackSave = parallelTrackingBasedMatrices;
  parallelTrackingBasedMatrices = 0;

  /* Ensure twiss parameters are up to date */
  LRC_retwiss(run, beamline, NULL);

  /* Collect knobs/BPMs and allocate working buffers if not already done. With
   * use_perturbed_matrix=0 this was done at setup; with use_perturbed_matrix>0
   * it is done here on the first call. */
  collectAndAllocate(beamline);

  if (verbosity > 0) {
    printf("correct_coupling: %ld skew quad knobs, %ld BPM observation points\n",
           nSkew, nBpm);
    fflush(stdout);
  }
  if (verbosity > 1) {
    printf("  skew knobs:\n");
    for (j = 0; j < nSkew; j++)
      printf("    %s#%ld  s=%le  %s=%le\n", knobs[j].elem->name, knobs[j].elem->occurence,
             knobs[j].elem->end_pos, knobItem ? knobItem[j] : skewItem, *knobs[j].valuePtr);
    printf("  BPMs:\n");
    for (i = 0; i < nBpm; i++)
      printf("    %s#%ld  s=%le\n", bpms[i].elem->name, bpms[i].elem->occurence,
             bpms[i].elem->end_pos);
    fflush(stdout);
  }

  /* Keep the most recent strength_log page open until we know whether the next
   * iteration will run; that lets us tag the actually-final iteration with
   * Stage="corrected" without buffering or rewriting the file. */
  short strengthLogPageOpen = 0;

  /* For use_perturbed_matrix>=1, invalidate any cached response so the first
   * iteration triggers a fresh build using the lattice's current state. The
   * default use_perturbed_matrix=0 leaves responseValid set from setup. */
  if (use_perturbed_matrix >= 1)
    responseValid = 0;

  /* Working correction fraction; can be reduced by adaptive_step on RMS growth. */
  double currentFraction = correction_fraction;
  /* RMS from the previous iteration's entry, for adaptive comparison. */
  double prevRMS = -1.0;

  /* Start the rms_log page (one row per iteration index 0..n).  Row 0 logs
   * the initial state, row k+1 logs the state after iteration k.  Skipped
   * iterations leave fewer than n+1 rows. */
  long rmsLogRow = 0;
  if (SDDSrmsLogInit) {
    if (!SDDS_StartPage(&SDDSrmsLog, n_iterations + 1) ||
        !SDDS_SetParameters(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "EtayWeight",          (double)etayChannelWeight,
                            "CrossResponseWeight", (double)crossChannelWeight, NULL))
      SDDS_Bomb("correct_coupling: error starting rms_log page");
  }

  LRC_retwiss(run, beamline, NULL);
  /* Read the combined channel into yResidFlat: first nBpm slots hold etay,
   * the rest hold the cross-plane response if the channel is active.  Keep
   * a copy in etay[] so etay_file output and the strength-log diagnostics
   * keep their existing semantics. */
  etayReader(nRowsTotal, yResidFlat, &readerCtx);
  for (i = 0; i < nBpm; i++) etay[i] = yResidFlat[i];
  rmsEtay  = ccRmsValue(yResidFlat, nRowsEtay);
  rmsCross = (nRowsTotal > nRowsEtay)
             ? ccRmsValue(yResidFlat + nRowsEtay, nRowsTotal - nRowsEtay) : 0;
  rms0     = ccWeightedRms(yResidFlat, rowWeight, nRowsTotal);
  if (SDDSrmsLogInit) {
    if (!SDDS_SetRowValues(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, rmsLogRow,
                           "Iteration", (long)0,
                           "EtayRms",     rmsEtay,
                           "CrossRms",    rmsCross,
                           "WeightedRms", rms0, NULL))
      SDDS_Bomb("correct_coupling: error setting rms_log row 0");
    rmsLogRow++;
  }
  if (verbosity > 0) {
    if (crossChannelWeight > 0)
      printf("  initial RMS eta_y = %le m, RMS cross-plane = %le m/rad, weighted RMS = %le\n",
             rmsEtay, rmsCross, rms0);
    else
      printf("  initial RMS eta_y at BPMs = %le m\n", rmsEtay);
    fflush(stdout);
  }

  /* Start with the svd threshold equal to the nominal value */
  double workingSvdThreshold = svd_threshold;
  /* prevKnobBefore retains the knob values from the start of the previous
   * iteration's apply so the adaptive_step block can roll back a step that
   * made things worse, not just shrink the next step.  NULL until the first
   * iteration completes. */
  double *prevKnobBefore = NULL;
  for (iter = 0; iter < n_iterations; iter++) {
    if (iter!=0 && workingSvdThreshold>svd_threshold) {
      /* Back off somewhat from adjusted (increased) SVD threshold so we don't slow things
       * too much */
      workingSvdThreshold = workingSvdThreshold/sqrt(auto_sv_threshold_factor);
      if (workingSvdThreshold<svd_threshold)
	workingSvdThreshold = svd_threshold;
    }
    /* Adaptive backoff: if the previous iteration made RMS worse, halve the
     * correction fraction AND roll the knobs back to the state at the start
     * of that previous iteration, so this iteration tries a smaller step from
     * a known-better operating point rather than compounding the bad step. */
    if (adaptive_step && prevRMS >= 0 && rms0 > prevRMS) {
      currentFraction *= 0.5;
      if (verbosity > 0) {
        printf("  iteration %ld: RMS grew %le -> %le; halving correction_fraction to %le\n",
               iter, prevRMS, rms0, currentFraction);
        fflush(stdout);
      }
      if (prevKnobBefore && nSkew > 0) {
        for (j = 0; j < nSkew; j++) {
          *knobs[j].valuePtr = prevKnobBefore[j];
          if (knobs[j].elem->matrix) {
            free_matrices(knobs[j].elem->matrix);
            free(knobs[j].elem->matrix);
            knobs[j].elem->matrix = NULL;
          }
          compute_matrix(knobs[j].elem, run, NULL);
          change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                                   knobs[j].elem->type, prevKnobBefore[j], NULL, LOAD_FLAG_ABSOLUTE);
        }
        pushWarningSuppression();
        (void)LRC_retwiss_status(run, beamline, NULL, 0);
        popWarningSuppression();
        etayReader(nRowsTotal, yResidFlat, &readerCtx);
        for (i = 0; i < nBpm; i++) etay[i] = yResidFlat[i];
        rmsEtay  = ccRmsValue(yResidFlat, nRowsEtay);
        rmsCross = (nRowsTotal > nRowsEtay)
                   ? ccRmsValue(yResidFlat + nRowsEtay, nRowsTotal - nRowsEtay) : 0;
        rms0 = ccWeightedRms(yResidFlat, rowWeight, nRowsTotal);
        if (verbosity > 0) {
          printf("  iteration %ld: rolled back to previous-iteration values; RMS now %le\n",
                 iter, rms0);
          fflush(stdout);
        }
      }
    }
    prevRMS = rms0;

    /* Report eta_y at BPMs at the start of this iteration */
    if (SDDSetayInit) {
      if (verbosity>2) {
	printf("  Writing etay\n");
	fflush(stdout);
      }
      if (!SDDS_StartPage(&SDDSetay, nBpm) ||
          !SDDS_SetParameters(&SDDSetay, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Iteration", iter, "etayRMS", rms0, NULL))
        SDDS_Bomb("correct_coupling: error writing etay_file page");
      for (i = 0; i < nBpm; i++) {
        if (!SDDS_SetRowValues(&SDDSetay, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, i,
                               "ElementName",      bpms[i].elem->name,
                               "ElementOccurence", bpms[i].elem->occurence,
                               "s",                bpms[i].elem->end_pos,
                               "etay",             etay[i], NULL))
          SDDS_Bomb("correct_coupling: error setting etay_file row");
      }
      if (!SDDS_WritePage(&SDDSetay))
        SDDS_Bomb("correct_coupling: error writing etay_file page");
    }

    if (rms0 < convergence) {
      if (verbosity > 0) {
        if (crossChannelWeight > 0)
          printf("  iteration %ld: weighted RMS %le below convergence %le; stopping\n",
                 iter, rms0, convergence);
        else
          printf("  iteration %ld: RMS eta_y %le below convergence %le; stopping\n",
                 iter, rms0, convergence);
        fflush(stdout);
      }
      /* The previous iteration (if any) was therefore the actually-final one. */
      if (strengthLogPageOpen) {
        if (verbosity>2) {
  	  printf("  Writing strength log\n");
	  fflush(stdout);
        }
        if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Stage", "corrected", NULL) ||
            !SDDS_WritePage(&SDDSstrengthLog))
          SDDS_Bomb("correct_coupling: error writing strength_log page");
        strengthLogPageOpen = 0;
      }
      break;
    }

    /* Build the response matrix if needed. With use_perturbed_matrix=0 it was
     * built in setup and responseValid is already set; with use_perturbed_matrix=1
     * it is built before iter 0 of this call (we invalidated above); with
     * use_perturbed_matrix>1 it is rebuilt before each iteration. */
    if (!responseValid)
      buildResponseMatrix(run, beamline, iter, response_perturbation);

    /* Save the "before" knob values so we can roll back if applying the step
     * leaves the lattice unstable.  Allocated once per iteration. */
    double *knobBefore = tmalloc(sizeof(*knobBefore) * nSkew);
    for (j = 0; j < nSkew; j++) knobBefore[j] = *knobs[j].valuePtr;

    /* SVD retry loop: solve, apply, check stability.  On instability with
     * auto_sv_threshold enabled, restore baseline, bump workingSvdThreshold,
     * and re-solve.  Bail out either when stable or when retries exhausted
     * (or when the bumped threshold drops every singular value, in which
     * case no correction is applied and the lattice is trivially stable).
     * The working threshold is local to this iteration so each iter starts
     * with the user-requested namelist value. */
    double scale = 0;
    int retryUnstable = 0;
    short emitWarning = 1;  /* whether to print the unstable warning on final retwiss */
    long autoRetry;
    const long maxAutoRetries = 20;
    for (autoRetry = 0; autoRetry <= maxAutoRetries; autoRetry++) {
#if USE_MPI
      if (myid==0) {
#endif
        if (verbosity>3) {
          printf("  Solving for skew strengths using SVD\n");
          fflush(stdout);
        }
        double **Rweighted = (double **)czarray_2d(sizeof(double), nRowsTotal, nSkew);
        double *yWeighted  = tmalloc(sizeof(*yWeighted) * nRowsTotal);
        for (i = 0; i < nRowsTotal; i++) {
          double wi = rowWeight[i];
          yWeighted[i] = wi * yResidFlat[i];
          for (j = 0; j < nSkew; j++) Rweighted[i][j] = wi * R[i][j];
        }
        LRC_svdSolve(Rweighted, nRowsTotal, nSkew, yWeighted, dK,
                     workingSvdThreshold, n_singular_values,
                     &minSV, &maxSV, &nUsedSV);
        free(yWeighted);
        free_czarray_2d((void **)Rweighted, nRowsTotal, nSkew);
        if (verbosity > 1) {
          printf("  iteration %ld: SVD used %ld of %ld singular values; SV range [%le, %le]; threshold %le\n",
                 iter, nUsedSV, MIN(nSkew, nRowsTotal), minSV, maxSV, workingSvdThreshold);
          fflush(stdout);
        }
#if USE_MPI
      }
      MPI_Bcast(dK, nSkew, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nUsedSV, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#endif

      if (currentFraction != 1.0)
        for (j = 0; j < nSkew; j++) dK[j] *= currentFraction;

      scale = LRC_clampStepToLimitArray(knobs, dK, nSkew, knobLower, knobUpper);
      if (scale == 0 || nUsedSV == 0) {
        /* No step possible (limit binding or threshold dropped all SVs).
         * Restore to before-state and break out. */
        for (j = 0; j < nSkew; j++) {
          *knobs[j].valuePtr = knobBefore[j];
          if (knobs[j].elem->matrix) {
            free_matrices(knobs[j].elem->matrix);
            free(knobs[j].elem->matrix);
            knobs[j].elem->matrix = NULL;
          }
          compute_matrix(knobs[j].elem, run, NULL);
          change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                                   knobs[j].elem->type, knobBefore[j], NULL, LOAD_FLAG_ABSOLUTE);
        }
        LRC_retwiss(run, beamline, NULL);
        scale = 0;
        break;
      }

      /* Apply the step. */
      for (j = 0; j < nSkew; j++) {
        double after = knobBefore[j] + scale * dK[j];
        *knobs[j].valuePtr = after;
        if (knobs[j].elem->matrix) {
          free_matrices(knobs[j].elem->matrix);
          free(knobs[j].elem->matrix);
          knobs[j].elem->matrix = NULL;
        }
        compute_matrix(knobs[j].elem, run, NULL);
        change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                                 knobs[j].elem->type, after, NULL, LOAD_FLAG_ABSOLUTE);
      }

      /* Check stability.  Suppress the engine's "unstable twiss" warning AND
       * the twiss-routine's underlying "beamline unstable" warning while we
       * still have retries; only emit them on the last attempt. */
      short suppressWarn = auto_sv_threshold && (autoRetry < maxAutoRetries);
      if (suppressWarn) pushWarningSuppression();
      retryUnstable = LRC_retwiss_status(run, beamline, NULL, !suppressWarn);
      if (suppressWarn) popWarningSuppression();

      if (!retryUnstable) {
        emitWarning = 0;
        break;  /* stable -- accept this step */
      }
      if (!auto_sv_threshold) break;
      if (autoRetry == maxAutoRetries) {
        /* Retries exhausted and lattice still unstable.  Roll the knobs back
         * to the start-of-iteration values and abandon this iteration so the
         * outer adaptive_step logic can react. */
        if (verbosity > 0) {
          printf("  iteration %ld: auto_sv_threshold retries exhausted; "
                 "rolling back step and skipping this iteration\n", iter);
          fflush(stdout);
        }
        for (j = 0; j < nSkew; j++) {
          *knobs[j].valuePtr = knobBefore[j];
          if (knobs[j].elem->matrix) {
            free_matrices(knobs[j].elem->matrix);
            free(knobs[j].elem->matrix);
            knobs[j].elem->matrix = NULL;
          }
          compute_matrix(knobs[j].elem, run, NULL);
          change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                                   knobs[j].elem->type, knobBefore[j], NULL, LOAD_FLAG_ABSOLUTE);
        }
        pushWarningSuppression();
        (void)LRC_retwiss_status(run, beamline, NULL, 0);
        popWarningSuppression();
        scale = 0;  /* signal no-step-applied path */
        break;
      }

      /* Roll back and try again with a larger SV-cutoff threshold. */
      double newThreshold = workingSvdThreshold * auto_sv_threshold_factor;
      if (verbosity > 0) {
        printf("  iteration %ld: unstable twiss; rolling back, svd_threshold %le -> %le (retry %ld)\n",
               iter, workingSvdThreshold, newThreshold, autoRetry + 1);
        fflush(stdout);
      }
      workingSvdThreshold = newThreshold;
      for (j = 0; j < nSkew; j++) {
        *knobs[j].valuePtr = knobBefore[j];
        if (knobs[j].elem->matrix) {
          free_matrices(knobs[j].elem->matrix);
          free(knobs[j].elem->matrix);
          knobs[j].elem->matrix = NULL;
        }
        compute_matrix(knobs[j].elem, run, NULL);
        change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                                 knobs[j].elem->type, knobBefore[j], NULL, LOAD_FLAG_ABSOLUTE);
      }
      /* Retwiss to restore the baseline twiss before next attempt; suppress
       * warning since this is purely a restore-to-known-good. */
      pushWarningSuppression();
      (void)LRC_retwiss_status(run, beamline, NULL, 0);
      popWarningSuppression();
    }
    (void)emitWarning;  /* used only as a flag for whether to log */

    if (scale == 0) {
      if (verbosity > 0)
        printf("  iteration %ld: no correction applied (per-knob strength limit reached or all SVs dropped)\n",
               iter);
      free(knobBefore);
      if (strengthLogPageOpen) {
        if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Stage", "corrected", NULL) ||
            !SDDS_WritePage(&SDDSstrengthLog))
          SDDS_Bomb("correct_coupling: error writing strength_log page");
        strengthLogPageOpen = 0;
      }
      break;
    }
    if (scale < 1 && verbosity > 0) {
      printf("  iteration %ld: step scaled by %le to respect per-knob strength limits\n",
             iter, scale);
      fflush(stdout);
    }

    /* About to log this iteration's changes -- flush any previous still-open
     * page as Stage="uncorrected" (we now know the previous iter was NOT the
     * last one to apply changes). */
    if (strengthLogPageOpen) {
      if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Stage", "uncorrected", NULL) ||
          !SDDS_WritePage(&SDDSstrengthLog))
        SDDS_Bomb("correct_coupling: error writing strength_log page");
      strengthLogPageOpen = 0;
    }

    if (SDDSstrengthLogInit) {
      if (!SDDS_StartPage(&SDDSstrengthLog, nSkew) ||
          !SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Iteration", iter, NULL))
        SDDS_Bomb("correct_coupling: error writing strength_log page");
    }
    for (j = 0; j < nSkew; j++) {
      double before = knobBefore[j];
      double after  = *knobs[j].valuePtr;
      double step   = after - before;
      if (SDDSstrengthLogInit) {
        if (!SDDS_SetRowValues(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, j,
                               "ElementName",            knobs[j].elem->name,
                               "ElementOccurence",       knobs[j].elem->occurence,
                               "s",                      knobs[j].elem->end_pos,
                               "ElementParameter",       knobItem[j],
                               "PreviousParameterValue", before,
                               "DeltaParameterValue",    step,
                               "ParameterValue",         after, NULL))
          SDDS_Bomb("correct_coupling: error setting strength_log row");
      }
    }
    /* Transfer this iteration's "before" snapshot to prevKnobBefore so the
     * next iteration's adaptive_step block can roll back this iteration's
     * apply if it ends up making the residual worse. */
    if (prevKnobBefore) free(prevKnobBefore);
    prevKnobBefore = knobBefore;
    knobBefore = NULL;
    /* Don't write the page yet -- defer until the next iteration starts (when
     * we'll tag it "uncorrected") or until the loop exits (when we'll tag it
     * "corrected"). */
    if (SDDSstrengthLogInit)
      strengthLogPageOpen = 1;

    /* Final retwiss already done inside the retry loop; the state in memory
     * matches whatever we accepted.  Re-read both channels. */
    etayReader(nRowsTotal, yResidFlat, &readerCtx);
    for (i = 0; i < nBpm; i++) etay[i] = yResidFlat[i];
    rmsEtay  = ccRmsValue(yResidFlat, nRowsEtay);
    rmsCross = (nRowsTotal > nRowsEtay)
               ? ccRmsValue(yResidFlat + nRowsEtay, nRowsTotal - nRowsEtay) : 0;
    rms      = ccWeightedRms(yResidFlat, rowWeight, nRowsTotal);
    if (SDDSrmsLogInit) {
      if (!SDDS_SetRowValues(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, rmsLogRow,
                             "Iteration",   (long)(iter + 1),
                             "EtayRms",     rmsEtay,
                             "CrossRms",    rmsCross,
                             "WeightedRms", rms, NULL))
        SDDS_Bomb("correct_coupling: error setting rms_log row");
      rmsLogRow++;
    }
    if (verbosity > 0) {
      if (crossChannelWeight > 0)
        printf("  iteration %ld: RMS eta_y -> %le m, RMS cross -> %le m/rad, weighted RMS -> %le\n",
               iter, rmsEtay, rmsCross, rms);
      else
        printf("  iteration %ld: RMS eta_y -> %le m\n", iter, rmsEtay);
      fflush(stdout);
    }
    rms0 = rms;

    /* change_tolerance: stop if the relative improvement this iteration was
     * positive but smaller than the requested floor.  Negative improvement
     * (rms grew) is left to the adaptive_step block at the top of the next
     * iteration so it can roll back rather than terminate prematurely. */
    if (change_tolerance > 0 && prevRMS > 0 && rms <= prevRMS &&
        (prevRMS - rms) < change_tolerance * prevRMS) {
      if (verbosity > 0) {
        printf("  iteration %ld: relative improvement %le below change_tolerance %le; stopping\n",
               iter, (prevRMS - rms) / prevRMS, change_tolerance);
        fflush(stdout);
      }
      break;
    }

    /* For use_perturbed_matrix>1, force a fresh response build before the next
     * iteration; for 0 or 1, leave the matrix in place. */
    if (use_perturbed_matrix > 1)
      responseValid = 0;
  }
  if (prevKnobBefore) { free(prevKnobBefore); prevKnobBefore = NULL; }

  /* Loop completed normally (n_iterations exhausted). Any still-open page is
   * therefore from the actually-final iteration -- tag it "corrected". */
  if (strengthLogPageOpen) {
    if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Stage", "corrected", NULL) ||
        !SDDS_WritePage(&SDDSstrengthLog))
      SDDS_Bomb("correct_coupling: error writing strength_log page");
    strengthLogPageOpen = 0;
  }
  if (SDDSrmsLogInit) {
    if (!SDDS_WritePage(&SDDSrmsLog))
      SDDS_Bomb("correct_coupling: error writing rms_log page");
  }

  if (verbosity > 0) {
    if (crossChannelWeight > 0)
      printf("correct_coupling: final RMS eta_y = %le m, RMS cross = %le m/rad, weighted RMS = %le\n",
             rmsEtay, rmsCross, rms0);
    else
      printf("correct_coupling: final RMS eta_y at BPMs = %le m\n", rmsEtay);
    fflush(stdout);
  }

  parallelTrackingBasedMatrices = parTrackSave;
  return 1;
}

/****************************************************************************/

void finish_correct_coupling(void) {
  if (SDDSstrengthLogInit)    { SDDS_Terminate(&SDDSstrengthLog);    SDDSstrengthLogInit = 0; }
  if (SDDSetayInit)      { SDDS_Terminate(&SDDSetay);      SDDSetayInit = 0; }
  if (SDDSresponseInit)  { SDDS_Terminate(&SDDSresponse);  SDDSresponseInit = 0; }
  if (SDDSrmsLogInit)    { SDDS_Terminate(&SDDSrmsLog);    SDDSrmsLogInit = 0; }
  if (SDDSrmsLogInit)    { SDDS_Terminate(&SDDSrmsLog);    SDDSrmsLogInit = 0; }
  LRC_freePatternList(&skewName, &nSkewName);
  LRC_freePatternList(&skewType, &nSkewType);
  LRC_freePatternList(&bpmName,  &nBpmName);
  LRC_freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }
  if (crossHSpec) { free(crossHSpec); crossHSpec = NULL; }
  if (crossVSpec) { free(crossVSpec); crossVSpec = NULL; }
  LRC_freePatternList(&crossXBpmNamePat, &nCrossXBpmNamePat);
  LRC_freePatternList(&crossXBpmTypePat, &nCrossXBpmTypePat);
  LRC_freePatternList(&crossYBpmNamePat, &nCrossYBpmNamePat);
  LRC_freePatternList(&crossYBpmTypePat, &nCrossYBpmTypePat);
  /* Free module-level knob/BPM/working-buffer state. */
  if (etay)       { free(etay);       etay = NULL; }
  if (etayPert)   { free(etayPert);   etayPert = NULL; }
  if (dK)         { free(dK);         dK = NULL; }
  if (yResidFlat) { free(yResidFlat); yResidFlat = NULL; }
  if (yPertFlat)  { free(yPertFlat);  yPertFlat = NULL; }
  if (rowWeight)  { free(rowWeight);  rowWeight = NULL; }
#if USE_MPI
  if (R && myid == 0)  { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms = NULL; }
  freeCrossPlaneState();
  nRowsTotal = nRowsEtay = 0;
  LRC_freePatternList(&corrPatterns, &nCorrPatterns);
  if (corrPatternStr) { free(corrPatternStr); corrPatternStr = NULL; }
  LRC_freePatternList(&itemsList, &nItemsList);
  LRC_freePatternList(&excludePatterns, &nExcludePatterns);
  if (lowerList) { free(lowerList); lowerList = NULL; }
  if (upperList) { free(upperList); upperList = NULL; }
  nLowerList = nUpperList = 0;
  freeKnobItem();
  if (knobLower) { free(knobLower); knobLower = NULL; }
  if (knobUpper) { free(knobUpper); knobUpper = NULL; }
  if (knobFamily){ free(knobFamily);knobFamily = NULL; }
  nSkew = nBpm = 0;
  responseValid = 0;
  initialized = 0;
}

/****************************************************************************/
/* Free any preloaded module state (knobs/bpms/buffers/patterns), so a
 * subsequent compute_/load_ command starts from a clean slate. */
static void freeModuleState(void) {
  if (etay)       { free(etay);       etay = NULL; }
  if (etayPert)   { free(etayPert);   etayPert = NULL; }
  if (dK)         { free(dK);         dK = NULL; }
  if (yResidFlat) { free(yResidFlat); yResidFlat = NULL; }
  if (yPertFlat)  { free(yPertFlat);  yPertFlat = NULL; }
  if (rowWeight)  { free(rowWeight);  rowWeight = NULL; }
#if USE_MPI
  if (R && myid == 0)  { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, nRowsTotal, nSkew); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms = NULL; }
  freeCrossPlaneState();
  nRowsTotal = nRowsEtay = 0;
  LRC_freePatternList(&skewName, &nSkewName);
  LRC_freePatternList(&skewType, &nSkewType);
  LRC_freePatternList(&bpmName,  &nBpmName);
  LRC_freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }
  LRC_freePatternList(&corrPatterns, &nCorrPatterns);
  if (corrPatternStr) { free(corrPatternStr); corrPatternStr = NULL; }
  LRC_freePatternList(&itemsList, &nItemsList);
  LRC_freePatternList(&excludePatterns, &nExcludePatterns);
  if (lowerList) { free(lowerList); lowerList = NULL; }
  if (upperList) { free(upperList); upperList = NULL; }
  nLowerList = nUpperList = 0;
  freeKnobItem();
  if (knobLower) { free(knobLower); knobLower = NULL; }
  if (knobUpper) { free(knobUpper); knobUpper = NULL; }
  if (knobFamily){ free(knobFamily);knobFamily = NULL; }
  nSkew = nBpm = 0;
  responseValid = 0;
}

/* Write the current in-memory response matrix to an SDDS file in a format
 * load_coupling_response_matrix can read back. */
static void saveResponseMatrixToFile(char *filename, double pert) {
  SDDS_DATASET out;
  long i, j, row;
#if USE_MPI
  if (myid != 0) return;
#endif
  if (!SDDS_InitializeOutputElegant(&out, SDDS_BINARY, 0,
                                    "Coupling correction response matrix",
                                    NULL, filename) ||
      SDDS_DefineColumn(&out, "BPMName", NULL, NULL,
                        "BPM (observation point) element name",
                        NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "BPMOccurence", NULL, NULL,
                        "BPM element occurrence in the beamline",
                        NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "sBPM", "s$bBPM$n", "m",
                        "BPM position along the beamline",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "SkewName", NULL, NULL,
                        "Skew quadrupole (knob) element name",
                        NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "SkewOccurence", NULL, NULL,
                        "Skew element occurrence in the beamline",
                        NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "sSkew", "s$bskew$n", "m",
                        "Skew element position along the beamline",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "Coefficient", NULL, NULL,
                        "Response matrix entry: d eta_y(BPM)/d parameter(skew)",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "KnobParameter", NULL, NULL,
                        "Per-knob parameter name perturbed for this column "
                        "(supersedes ElementParameter parameter for multi-item "
                        "matrices)",
                        NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineParameter(&out, "ElementParameter", NULL, NULL,
                           "Legacy single-item field; for matrices with mixed "
                           "items the authoritative per-knob value lives in the "
                           "KnobParameter column",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CorrectionElements", NULL, NULL,
                           "Verbatim correction_elements string used to build "
                           "this matrix; consumed by load_coupling_response_matrix "
                           "so a subsequent correct_coupling can inherit the family "
                           "list and supply only lower_limits / upper_limits",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CorrectionItems", NULL, NULL,
                           "Whitespace-joined items list aligned with "
                           "CorrectionElements; consumed by load_coupling_response_matrix",
                           NULL, SDDS_STRING, NULL) < 0 ||
      !SDDS_DefineSimpleParameter(&out, "ResponsePerturbation", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&out, "nKnobs", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nBPMs", NULL, SDDS_LONG) ||
      !SDDS_WriteLayout(&out)) {
    fprintf(stderr, "compute_coupling_response_matrix: unable to open %s for output\n", filename);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  /* Build the joined items string from itemsList[] (one slot per family). */
  char *itemsJoined = NULL;
  {
    long total = 0, kk;
    for (kk = 0; kk < nItemsList; kk++) total += strlen(itemsList[kk]) + 1;
    if (total == 0) total = 1;
    itemsJoined = tmalloc(total + 1);
    itemsJoined[0] = '\0';
    for (kk = 0; kk < nItemsList; kk++) {
      if (kk > 0) strcat(itemsJoined, " ");
      strcat(itemsJoined, itemsList[kk]);
    }
  }
  if (!SDDS_StartPage(&out, nBpm * nSkew) ||
      !SDDS_SetParameters(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ElementParameter",     skewItem,
                          "CorrectionElements",   corrPatternStr ? corrPatternStr : "",
                          "CorrectionItems",      itemsJoined,
                          "ResponsePerturbation", pert,
                          "nKnobs",               nSkew,
                          "nBPMs",                nBpm, NULL))
    SDDS_Bomb("compute_coupling_response_matrix: error writing parameters");
  free(itemsJoined);
  row = 0;
  for (i = 0; i < nBpm; i++)
    for (j = 0; j < nSkew; j++) {
      if (!SDDS_SetRowValues(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                             "BPMName",       bpms[i].elem->name,
                             "BPMOccurence",  bpms[i].elem->occurence,
                             "sBPM",          bpms[i].elem->end_pos,
                             "SkewName",      knobs[j].elem->name,
                             "SkewOccurence", knobs[j].elem->occurence,
                             "sSkew",         knobs[j].elem->end_pos,
                             "Coefficient",   R[i][j],
                             "KnobParameter", knobItem ? knobItem[j] : skewItem,
                             NULL))
        SDDS_Bomb("compute_coupling_response_matrix: error setting row");
      row++;
    }
  if (!SDDS_WritePage(&out) || !SDDS_Terminate(&out))
    SDDS_Bomb("compute_coupling_response_matrix: error closing output file");
}

/* Save the cross-plane block of R to a separate file.  Schema: one row per
 * (cross observable, skew knob) pair, fully self-describing so
 * load_coupling_response_matrix can reconstruct the corrector/BPM inventory
 * from the file alone.  Only invoked when the cross-plane channel is active. */
static void saveCrossResponseMatrixToFile(char *filename, double pert) {
  SDDS_DATASET out;
  long iC, iB, j, row;
#if USE_MPI
  if (myid != 0) return;
#endif
  if (!SDDS_InitializeOutputElegant(&out, SDDS_BINARY, 0,
                                    "Cross-plane steering response matrix for correct_coupling",
                                    NULL, filename) ||
      SDDS_DefineColumn(&out, "Channel", NULL, NULL,
                        "Cross-plane block: \"yFromH\" or \"xFromV\"",
                        NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "CorrName", NULL, NULL,
                        "Probe corrector element name", NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "CorrOccurence", NULL, NULL,
                        "Probe corrector occurrence in the beamline", NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "CorrItem", NULL, NULL,
                        "Item perturbed on the probe corrector", NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "sCorr", "s$bcorr$n", "m",
                        "Probe corrector position", NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "BPMName", NULL, NULL,
                        "Target BPM element name", NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "BPMOccurence", NULL, NULL,
                        "Target BPM occurrence", NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "sBPM", "s$bBPM$n", "m",
                        "Target BPM position", NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "SkewName", NULL, NULL,
                        "Skew quadrupole (knob) element name", NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "SkewOccurence", NULL, NULL,
                        "Skew occurrence in the beamline", NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "sSkew", "s$bskew$n", "m",
                        "Skew position", NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "Coefficient", NULL, "m/rad",
                        "d(cross-plane centroid)/d(skew item)",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineParameter(&out, "ElementParameter", NULL, NULL,
                           "Name of the skew parameter that was perturbed",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossHSteering", NULL, NULL,
                           "Spec for H-plane probe correctors",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossVSteering", NULL, NULL,
                           "Spec for V-plane probe correctors",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossXBpmNamePattern", NULL, NULL,
                           "Name pattern for cross-plane X-reading BPMs",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossXBpmTypePattern", NULL, NULL,
                           "Type pattern for cross-plane X-reading BPMs",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossYBpmNamePattern", NULL, NULL,
                           "Name pattern for cross-plane Y-reading BPMs",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CrossYBpmTypePattern", NULL, NULL,
                           "Type pattern for cross-plane Y-reading BPMs",
                           NULL, SDDS_STRING, NULL) < 0 ||
      !SDDS_DefineSimpleParameter(&out, "CrossSteeringKick", "rad", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&out, "ResponsePerturbation", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&out, "nKnobs", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nHCorr", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nVCorr", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nCrossYBpm", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nCrossXBpm", NULL, SDDS_LONG) ||
      !SDDS_WriteLayout(&out)) {
    fprintf(stderr, "compute_coupling_response_matrix: unable to open %s for cross-plane output\n",
            filename);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  long nCrossRows = (nRowsCrossHy + nRowsCrossVx) * nSkew;
  if (!SDDS_StartPage(&out, nCrossRows) ||
      !SDDS_SetParameters(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ElementParameter",       skewItem,
                          "CrossHSteering",         crossHSpec ? crossHSpec : "",
                          "CrossVSteering",         crossVSpec ? crossVSpec : "",
                          "CrossXBpmNamePattern",   "",
                          "CrossXBpmTypePattern",   "",
                          "CrossYBpmNamePattern",   "",
                          "CrossYBpmTypePattern",   "",
                          "CrossSteeringKick",      crossKickMag,
                          "ResponsePerturbation",   pert,
                          "nKnobs",                 nSkew,
                          "nHCorr",                 nHCorr,
                          "nVCorr",                 nVCorr,
                          "nCrossYBpm",             nCrossYBpm,
                          "nCrossXBpm",             nCrossXBpm, NULL))
    SDDS_Bomb("compute_coupling_response_matrix: error writing cross-plane parameters");
  row = 0;
  /* H -> y block: rows nBpm .. nBpm + nHCorr*nCrossYBpm - 1 of R */
  for (iC = 0; iC < nHCorr; iC++) {
    for (iB = 0; iB < nCrossYBpm; iB++) {
      long rRow = nBpm + iC * nCrossYBpm + iB;
      for (j = 0; j < nSkew; j++) {
        if (!SDDS_SetRowValues(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                               "Channel",       "yFromH",
                               "CorrName",      hCorrs[iC].elem->name,
                               "CorrOccurence", hCorrs[iC].elem->occurence,
                               "CorrItem",
                                 entity_description[hCorrs[iC].elem->type].parameter[hCorrs[iC].paramIndex].name,
                               "sCorr",         hCorrs[iC].elem->end_pos,
                               "BPMName",       crossYBpms[iB].elem->name,
                               "BPMOccurence",  crossYBpms[iB].elem->occurence,
                               "sBPM",          crossYBpms[iB].elem->end_pos,
                               "SkewName",      knobs[j].elem->name,
                               "SkewOccurence", knobs[j].elem->occurence,
                               "sSkew",         knobs[j].elem->end_pos,
                               "Coefficient",   R[rRow][j], NULL))
          SDDS_Bomb("compute_coupling_response_matrix: error setting cross-plane row");
        row++;
      }
    }
  }
  /* V -> x block: rows nBpm + nHCorr*nCrossYBpm .. nRowsTotal - 1 of R */
  long baseRow = nBpm + nHCorr * nCrossYBpm;
  for (iC = 0; iC < nVCorr; iC++) {
    for (iB = 0; iB < nCrossXBpm; iB++) {
      long rRow = baseRow + iC * nCrossXBpm + iB;
      for (j = 0; j < nSkew; j++) {
        if (!SDDS_SetRowValues(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                               "Channel",       "xFromV",
                               "CorrName",      vCorrs[iC].elem->name,
                               "CorrOccurence", vCorrs[iC].elem->occurence,
                               "CorrItem",
                                 entity_description[vCorrs[iC].elem->type].parameter[vCorrs[iC].paramIndex].name,
                               "sCorr",         vCorrs[iC].elem->end_pos,
                               "BPMName",       crossXBpms[iB].elem->name,
                               "BPMOccurence",  crossXBpms[iB].elem->occurence,
                               "sBPM",          crossXBpms[iB].elem->end_pos,
                               "SkewName",      knobs[j].elem->name,
                               "SkewOccurence", knobs[j].elem->occurence,
                               "sSkew",         knobs[j].elem->end_pos,
                               "Coefficient",   R[rRow][j], NULL))
          SDDS_Bomb("compute_coupling_response_matrix: error setting cross-plane row");
        row++;
      }
    }
  }
  if (!SDDS_WritePage(&out) || !SDDS_Terminate(&out))
    SDDS_Bomb("compute_coupling_response_matrix: error closing cross-plane file");
}

/* compute_coupling_response_matrix command: build R from the lattice and
 * save it to a file. Leaves R, knobs, bpms, skewItem populated in module state
 * so a subsequent correct_coupling reuses them. */
void setup_compute_coupling_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&compute_coupling_response_matrix, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &compute_coupling_response_matrix);

  if (compute_coupling_response_matrix_struct.filename == NULL)
    bombElegant("compute_coupling_response_matrix: filename is required", NULL);
  if (compute_coupling_response_matrix_struct.correction_elements == NULL ||
      *compute_coupling_response_matrix_struct.correction_elements == 0)
    bombElegant("compute_coupling_response_matrix: correction_elements is required", NULL);
  if (compute_coupling_response_matrix_struct.response_perturbation <= 0)
    bombElegant("compute_coupling_response_matrix: response_perturbation must be > 0", NULL);
  if (compute_coupling_response_matrix_struct.measurement_noise < 0)
    bombElegant("compute_coupling_response_matrix: measurement_noise must be >= 0", NULL);
  if (compute_coupling_response_matrix_struct.measurement_noise > 0 &&
      compute_coupling_response_matrix_struct.measurement_noise_cutoff <= 0)
    bombElegant("compute_coupling_response_matrix: measurement_noise_cutoff must be > 0 when measurement_noise > 0",
                NULL);

  freeModuleState();
  /* Parse the new family-list inventory params (same convention as
   * &correct_coupling).  No lower/upper here -- limits are correct-time
   * concerns and live on &correct_coupling. */
  cp_str(&corrPatternStr, compute_coupling_response_matrix_struct.correction_elements);
  corrPatterns = addPatterns(&nCorrPatterns, corrPatternStr);
  if (nCorrPatterns == 0)
    bombElegant("compute_coupling_response_matrix: correction_elements parsed to zero patterns", NULL);
  itemsList = tmalloc(sizeof(*itemsList) * nCorrPatterns);
  nItemsList = nCorrPatterns;
  if (compute_coupling_response_matrix_struct.items &&
      *compute_coupling_response_matrix_struct.items) {
    char *icopy;
    cp_str(&icopy, compute_coupling_response_matrix_struct.items);
    char *tok;
    long ii = 0;
    while ((tok = get_token(icopy)) && ii < nCorrPatterns) {
      cp_str(&itemsList[ii], tok);
      ii++;
    }
    free(icopy);
    if (ii != nCorrPatterns)
      bombElegantVA("compute_coupling_response_matrix: items has %ld entries but correction_elements has %ld",
                    ii, nCorrPatterns);
  } else {
    long ii;
    for (ii = 0; ii < nCorrPatterns; ii++) cp_str(&itemsList[ii], "K1");
  }
  if (compute_coupling_response_matrix_struct.exclude &&
      *compute_coupling_response_matrix_struct.exclude) {
    char *ecopy;
    cp_str(&ecopy, compute_coupling_response_matrix_struct.exclude);
    excludePatterns = addPatterns(&nExcludePatterns, ecopy);
    free(ecopy);
  }
  bpmName = addPatterns(&nBpmName, compute_coupling_response_matrix_struct.bpm_name_pattern);
  bpmType = addPatterns(&nBpmType, compute_coupling_response_matrix_struct.bpm_type_pattern);
  /* Legacy skewItem records the first family's item for the
   * backward-compatibility ElementParameter parameter in the matrix file. */
  cp_str(&skewItem, itemsList[0]);

  /* Make verbosity available to the helpers (uses correct_coupling's global). */
  verbosity = compute_coupling_response_matrix_struct.verbosity;
  /* Propagate measurement noise into the globals consulted by readEtayAtBpms,
   * so a compute_coupling_response_matrix run can simulate noisy LOCO-style
   * matrix measurements. A later correct_coupling namelist will reset these
   * to its own values when it is parsed. */
  measurement_noise        = compute_coupling_response_matrix_struct.measurement_noise;
  measurement_noise_cutoff = compute_coupling_response_matrix_struct.measurement_noise_cutoff;

  /* If cross_filename is supplied, enable the cross-plane channel for this
   * build and stash the spec strings so the saved file can carry them. */
  short crossWanted = (compute_coupling_response_matrix_struct.cross_filename != NULL);
  if (crossWanted) {
    if (compute_coupling_response_matrix_struct.cross_h_steering == NULL &&
        compute_coupling_response_matrix_struct.cross_v_steering == NULL)
      bombElegant("compute_coupling_response_matrix: cross_filename set but neither "
                  "cross_h_steering nor cross_v_steering supplied", NULL);
    if (compute_coupling_response_matrix_struct.cross_steering_kick <= 0)
      bombElegant("compute_coupling_response_matrix: cross_steering_kick must be > 0", NULL);
    if (crossHSpec) { free(crossHSpec); crossHSpec = NULL; }
    if (crossVSpec) { free(crossVSpec); crossVSpec = NULL; }
    if (compute_coupling_response_matrix_struct.cross_h_steering)
      cp_str(&crossHSpec, compute_coupling_response_matrix_struct.cross_h_steering);
    if (compute_coupling_response_matrix_struct.cross_v_steering)
      cp_str(&crossVSpec, compute_coupling_response_matrix_struct.cross_v_steering);
    LRC_freePatternList(&crossXBpmNamePat, &nCrossXBpmNamePat);
    LRC_freePatternList(&crossXBpmTypePat, &nCrossXBpmTypePat);
    LRC_freePatternList(&crossYBpmNamePat, &nCrossYBpmNamePat);
    LRC_freePatternList(&crossYBpmTypePat, &nCrossYBpmTypePat);
    crossXBpmNamePat = addPatterns(&nCrossXBpmNamePat,
                                   compute_coupling_response_matrix_struct.cross_x_bpm_name_pattern);
    crossXBpmTypePat = addPatterns(&nCrossXBpmTypePat,
                                   compute_coupling_response_matrix_struct.cross_x_bpm_type_pattern);
    crossYBpmNamePat = addPatterns(&nCrossYBpmNamePat,
                                   compute_coupling_response_matrix_struct.cross_y_bpm_name_pattern);
    crossYBpmTypePat = addPatterns(&nCrossYBpmTypePat,
                                   compute_coupling_response_matrix_struct.cross_y_bpm_type_pattern);
    crossKickMag       = compute_coupling_response_matrix_struct.cross_steering_kick;
    crossMeasNoise     = compute_coupling_response_matrix_struct.cross_measurement_noise;
    crossChannelWeight = 1.0;   /* nonzero so the reader fills the cross-plane rows */
    etayChannelWeight  = 1.0;
  } else {
    /* No cross channel for this build. */
    crossChannelWeight = 0.0;
    etayChannelWeight  = 1.0;
  }

  collectAndAllocate(beamline);
  LRC_retwiss(run, beamline, NULL);
  buildResponseMatrix(run, beamline, -1,
                      compute_coupling_response_matrix_struct.response_perturbation);

  {
    char *fn = compose_filename(compute_coupling_response_matrix_struct.filename, run->rootname);
    saveResponseMatrixToFile(fn, compute_coupling_response_matrix_struct.response_perturbation);
    if (verbosity > 0) {
      printf("compute_coupling_response_matrix: wrote %ldx%ld eta_y response matrix to %s\n",
             nBpm, nSkew, fn);
      fflush(stdout);
    }
  }
  if (crossWanted) {
    char *fnx = compose_filename(compute_coupling_response_matrix_struct.cross_filename, run->rootname);
    saveCrossResponseMatrixToFile(fnx, compute_coupling_response_matrix_struct.response_perturbation);
    if (verbosity > 0) {
      printf("compute_coupling_response_matrix: wrote %ldx%ld cross-plane response matrix to %s "
             "(%ld H probes x %ld vmons + %ld V probes x %ld hmons)\n",
             nRowsCrossHy + nRowsCrossVx, nSkew, fnx,
             nHCorr, nCrossYBpm, nVCorr, nCrossXBpm);
      fflush(stdout);
    }
  }
}

/* load_coupling_response_matrix command: read a previously-saved R from a
 * file and populate module state so the next correct_coupling reuses it
 * without rebuilding. */
void setup_load_coupling_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  SDDS_DATASET in;
  char *fn;
  char *itemStr = NULL;
  long nRows;
  int32_t nKnobs = 0, nBPMs = 0;
  char **bpmNames = NULL, **skewNames = NULL;
  int32_t *bpmOccs = NULL, *skewOccs = NULL;
  double *coef = NULL;
  long i, j;

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&load_coupling_response_matrix, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &load_coupling_response_matrix);

  if (load_coupling_response_matrix_struct.filename == NULL)
    bombElegant("load_coupling_response_matrix: filename is required", NULL);

  fn = compose_filename(load_coupling_response_matrix_struct.filename, run->rootname);
  if (!SDDS_InitializeInputFromSearchPath(&in, fn) || SDDS_ReadPage(&in) != 1)
    SDDS_Bomb("load_coupling_response_matrix: cannot open or read reference file");

  if (SDDS_CheckColumn(&in, "BPMName",       NULL, SDDS_STRING,         stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "BPMOccurence",  NULL, SDDS_ANY_INTEGER_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "SkewName",      NULL, SDDS_STRING,         stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "SkewOccurence", NULL, SDDS_ANY_INTEGER_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "Coefficient",   NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "ElementParameter", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "nKnobs",     NULL, SDDS_ANY_INTEGER_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "nBPMs",      NULL, SDDS_ANY_INTEGER_TYPE, stdout) != SDDS_CHECK_OK)
    SDDS_Bomb("load_coupling_response_matrix: required columns/parameters missing");

  if (!SDDS_GetParameter(&in, "ElementParameter", &itemStr) ||
      !SDDS_GetParameterAsLong(&in, "nKnobs", &nKnobs) ||
      !SDDS_GetParameterAsLong(&in, "nBPMs",  &nBPMs))
    SDDS_Bomb("load_coupling_response_matrix: error reading parameters");
  nRows = SDDS_CountRowsOfInterest(&in);
  if (nRows != nKnobs * nBPMs)
    SDDS_Bomb("load_coupling_response_matrix: row count != nKnobs*nBPMs");
  if (!(bpmNames  = SDDS_GetColumn(&in, "BPMName")) ||
      !(bpmOccs   = SDDS_GetColumnInLong(&in, "BPMOccurence")) ||
      !(skewNames = SDDS_GetColumn(&in, "SkewName")) ||
      !(skewOccs  = SDDS_GetColumnInLong(&in, "SkewOccurence")) ||
      !(coef      = SDDS_GetColumnInDoubles(&in, "Coefficient")))
    SDDS_Bomb("load_coupling_response_matrix: error reading columns");
  /* Per-knob item column (KnobParameter) is the multi-item form; absent for
   * matrices written by older compute_coupling_response_matrix. */
  char **knobItemsF = NULL;
  if (SDDS_CheckColumn(&in, "KnobParameter", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    knobItemsF = SDDS_GetColumn(&in, "KnobParameter");
  /* Family-list metadata (CorrectionElements, CorrectionItems) written by
   * recent compute_coupling_response_matrix.  Optional: if absent, the
   * subsequent correct_coupling must either supply correction_elements
   * itself or run with no per-family bounds. */
  char *corrElemsF = NULL, *corrItemsF = NULL;
  if (SDDS_CheckParameter(&in, "CorrectionElements", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    SDDS_GetParameter(&in, "CorrectionElements", &corrElemsF);
  if (SDDS_CheckParameter(&in, "CorrectionItems", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    SDDS_GetParameter(&in, "CorrectionItems", &corrItemsF);
  SDDS_Terminate(&in);

  /* The file was written row-major with i (BPM) outer, j (skew) inner.
   * Rows 0..nKnobs-1 give the unique skews; rows 0, nKnobs, 2*nKnobs, ...
   * give the unique BPMs. Resolve each by (name, occurence) lookup in the
   * current beamline. */
  freeModuleState();
  cp_str(&skewItem, itemStr);
  /* Populate the family-list module state from the loaded metadata so that a
   * subsequent correct_coupling can inherit correction_elements/items and
   * only supply lower_limits/upper_limits aligned with the loaded count. */
  if (corrElemsF && *corrElemsF) {
    cp_str(&corrPatternStr, corrElemsF);
    corrPatterns = addPatterns(&nCorrPatterns, corrPatternStr);
    if (corrItemsF && *corrItemsF) {
      char *ic;
      cp_str(&ic, corrItemsF);
      char *tok;
      long ii = 0;
      itemsList = tmalloc(sizeof(*itemsList) * nCorrPatterns);
      nItemsList = nCorrPatterns;
      while ((tok = get_token(ic)) && ii < nCorrPatterns) {
        cp_str(&itemsList[ii], tok);
        ii++;
      }
      free(ic);
    }
    /* Seed the &correct_coupling namelist's correction_elements and items
     * globals (and their _default companions) so the next print_namelist
     * echoes the inherited values.  STICKY_NAMELIST_DEFAULTS in
     * processNamelist re-copies the _default into the live variable before
     * the user's entries are applied, so both must be updated to be
     * effective. */
    cp_str(&correction_elements, corrElemsF);
    cp_str(&correction_elements_default, corrElemsF);
    if (corrItemsF && *corrItemsF) {
      cp_str(&items, corrItemsF);
      cp_str(&items_default, corrItemsF);
    }
  }
  if (corrElemsF) free(corrElemsF);
  if (corrItemsF) free(corrItemsF);
  nSkew = nKnobs;
  nBpm  = nBPMs;

  knobs = SDDS_Realloc(NULL, sizeof(*knobs) * nSkew);
  knobItem = tmalloc(sizeof(*knobItem) * nSkew);
  for (j = 0; j < nSkew; j++) {
    ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline, skewNames[j], skewOccs[j]);
    if (!eptr) {
      fprintf(stderr, "load_coupling_response_matrix: cannot find skew %s#%d in current beamline\n",
              skewNames[j], skewOccs[j]);
      exitElegant(1);
    }
    char *thisItem = knobItemsF ? knobItemsF[j] : skewItem;
    cp_str(&knobItem[j], thisItem);
    knobs[j].elem = eptr;
    knobs[j].paramIndex = confirm_parameter(thisItem, eptr->type);
    if (knobs[j].paramIndex < 0 ||
        entity_description[eptr->type].parameter[knobs[j].paramIndex].type != IS_DOUBLE) {
      fprintf(stderr, "load_coupling_response_matrix: element %s has no double parameter %s\n",
              skewNames[j], thisItem);
      exitElegant(1);
    }
    knobs[j].valuePtr = (double *)(eptr->p_elem +
        entity_description[eptr->type].parameter[knobs[j].paramIndex].offset);
    knobs[j].initialValue = *knobs[j].valuePtr;
  }
  if (knobItemsF) {
    long jj;
    for (jj = 0; jj < nRows; jj++) free(knobItemsF[jj]);
    free(knobItemsF);
  }
  bpms = SDDS_Realloc(NULL, sizeof(*bpms) * nBpm);
  for (i = 0; i < nBpm; i++) {
    long rowI = i * nSkew;
    ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline, bpmNames[rowI], bpmOccs[rowI]);
    if (!eptr) {
      fprintf(stderr, "load_coupling_response_matrix: cannot find BPM %s#%d in current beamline\n",
              bpmNames[rowI], bpmOccs[rowI]);
      exitElegant(1);
    }
    bpms[i].elem = eptr;
  }

  /* Allocate working buffers and populate R. */
  etay     = tmalloc(sizeof(*etay)     * nBpm);
  etayPert = tmalloc(sizeof(*etayPert) * nBpm);
  dK       = tmalloc(sizeof(*dK)       * nSkew);
#if USE_MPI
  if (myid == 0)
    R = (double **)czarray_2d(sizeof(**R), nBpm, nSkew);
#else
  R = (double **)czarray_2d(sizeof(**R), nBpm, nSkew);
#endif
#if USE_MPI
  if (myid == 0)
#endif
  for (i = 0; i < nBpm; i++)
    for (j = 0; j < nSkew; j++)
      R[i][j] = coef[i * nSkew + j];

  responseValid = 1;

  if (load_coupling_response_matrix_struct.verbosity > 0) {
    printf("load_coupling_response_matrix: loaded %ldx%ld eta_y response matrix from %s (item=%s)\n",
           nBpm, nSkew, fn, skewItem);
    fflush(stdout);
  }

  /* If cross_filename is supplied, also load the cross-plane block.  Old files
   * with no cross-plane data simply omit this and behave as before. */
  if (load_coupling_response_matrix_struct.cross_filename) {
    SDDS_DATASET inx;
    char *fnx = compose_filename(load_coupling_response_matrix_struct.cross_filename, run->rootname);
    if (!SDDS_InitializeInputFromSearchPath(&inx, fnx) || SDDS_ReadPage(&inx) != 1)
      SDDS_Bomb("load_coupling_response_matrix: cannot open or read cross_filename");
    if (SDDS_CheckColumn(&inx, "Channel",       NULL, SDDS_STRING,            stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "CorrName",      NULL, SDDS_STRING,            stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "CorrOccurence", NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "CorrItem",      NULL, SDDS_STRING,            stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "BPMName",       NULL, SDDS_STRING,            stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "BPMOccurence",  NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "SkewName",      NULL, SDDS_STRING,            stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "SkewOccurence", NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckColumn(&inx, "Coefficient",   NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(&inx, "nKnobs",     NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(&inx, "nHCorr",     NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(&inx, "nVCorr",     NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(&inx, "nCrossYBpm", NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(&inx, "nCrossXBpm", NULL, SDDS_ANY_INTEGER_TYPE,  stdout) != SDDS_CHECK_OK)
      SDDS_Bomb("load_coupling_response_matrix: cross_filename missing required columns/parameters");

    int32_t nKnobsX = 0, nHCorrF = 0, nVCorrF = 0, nCrossYF = 0, nCrossXF = 0;
    char *hSpecStr = NULL, *vSpecStr = NULL;
    double kickMag = 1e-5;
    if (!SDDS_GetParameterAsLong(&inx, "nKnobs",     &nKnobsX) ||
        !SDDS_GetParameterAsLong(&inx, "nHCorr",     &nHCorrF) ||
        !SDDS_GetParameterAsLong(&inx, "nVCorr",     &nVCorrF) ||
        !SDDS_GetParameterAsLong(&inx, "nCrossYBpm", &nCrossYF) ||
        !SDDS_GetParameterAsLong(&inx, "nCrossXBpm", &nCrossXF) ||
        !SDDS_GetParameter(&inx, "CrossSteeringKick", &kickMag))
      SDDS_Bomb("load_coupling_response_matrix: error reading cross-plane parameters");
    if (nKnobsX != nSkew)
      bombElegant("load_coupling_response_matrix: cross_filename nKnobs disagrees with eta_y file", NULL);
    SDDS_GetParameter(&inx, "CrossHSteering", &hSpecStr);
    SDDS_GetParameter(&inx, "CrossVSteering", &vSpecStr);

    long nCrossRowsTotal = (long)(nHCorrF * nCrossYF + nVCorrF * nCrossXF);
    long nXRows = SDDS_CountRowsOfInterest(&inx);
    if (nXRows != nCrossRowsTotal * (long)nSkew)
      SDDS_Bomb("load_coupling_response_matrix: cross_filename row count != "
                "(nHCorr*nCrossYBpm + nVCorr*nCrossXBpm) * nKnobs");

    char **chanCol = SDDS_GetColumn(&inx, "Channel");
    char **cCorrNames = SDDS_GetColumn(&inx, "CorrName");
    int32_t *cCorrOccs = SDDS_GetColumnInLong(&inx, "CorrOccurence");
    char **cCorrItems = SDDS_GetColumn(&inx, "CorrItem");
    char **cBpmNames = SDDS_GetColumn(&inx, "BPMName");
    int32_t *cBpmOccs = SDDS_GetColumnInLong(&inx, "BPMOccurence");
    double *cCoef = SDDS_GetColumnInDoubles(&inx, "Coefficient");
    if (!chanCol || !cCorrNames || !cCorrOccs || !cCorrItems ||
        !cBpmNames || !cBpmOccs || !cCoef)
      SDDS_Bomb("load_coupling_response_matrix: error reading cross-plane columns");
    SDDS_Terminate(&inx);

    /* Resolve hCorrs / vCorrs and crossYBpms / crossXBpms from the file's
     * inventory.  The rows are ordered: first all H->y rows (iC outer, iB
     * inner, j innermost), then all V->x rows.  Within each block, the
     * corrector identity changes every nCrossYBpm/nCrossXBpm * nSkew rows. */
    freeCrossPlaneState();
    nHCorr = nHCorrF; nVCorr = nVCorrF;
    nCrossYBpm = nCrossYF; nCrossXBpm = nCrossXF;
    crossKickMag = kickMag;
    if (crossHSpec) { free(crossHSpec); crossHSpec = NULL; }
    if (crossVSpec) { free(crossVSpec); crossVSpec = NULL; }
    if (hSpecStr && *hSpecStr) cp_str(&crossHSpec, hSpecStr);
    if (vSpecStr && *vSpecStr) cp_str(&crossVSpec, vSpecStr);

    long iC;
    if (nHCorr > 0) {
      hCorrs = SDDS_Realloc(NULL, sizeof(*hCorrs) * nHCorr);
      for (iC = 0; iC < nHCorr; iC++) {
        long firstRow = iC * nCrossYBpm * nSkew;
        ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline,
                              cCorrNames[firstRow], cCorrOccs[firstRow]);
        if (!eptr) {
          fprintf(stderr, "load_coupling_response_matrix: cannot find H probe %s#%d in beamline\n",
                  cCorrNames[firstRow], cCorrOccs[firstRow]);
          exitElegant(1);
        }
        hCorrs[iC].elem = eptr;
        hCorrs[iC].paramIndex = confirm_parameter(cCorrItems[firstRow], eptr->type);
        if (hCorrs[iC].paramIndex < 0)
          bombElegant("load_coupling_response_matrix: H probe item not found on element", NULL);
        hCorrs[iC].valuePtr = (double *)(eptr->p_elem +
            entity_description[eptr->type].parameter[hCorrs[iC].paramIndex].offset);
        hCorrs[iC].initialValue = *hCorrs[iC].valuePtr;
      }
      crossYBpms = SDDS_Realloc(NULL, sizeof(*crossYBpms) * nCrossYBpm);
      crossYBpmIdx = tmalloc(sizeof(*crossYBpmIdx) * nCrossYBpm);
      long iB;
      for (iB = 0; iB < nCrossYBpm; iB++) {
        long firstRow = iB * nSkew;
        ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline,
                              cBpmNames[firstRow], cBpmOccs[firstRow]);
        if (!eptr) {
          fprintf(stderr, "load_coupling_response_matrix: cannot find cross-Y BPM %s#%d in beamline\n",
                  cBpmNames[firstRow], cBpmOccs[firstRow]);
          exitElegant(1);
        }
        crossYBpms[iB].elem = eptr;
        crossYBpmIdx[iB] = bpmElemIndex(beamline, eptr);
      }
    }
    long offsetV = nHCorr * nCrossYBpm * nSkew;
    if (nVCorr > 0) {
      vCorrs = SDDS_Realloc(NULL, sizeof(*vCorrs) * nVCorr);
      for (iC = 0; iC < nVCorr; iC++) {
        long firstRow = offsetV + iC * nCrossXBpm * nSkew;
        ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline,
                              cCorrNames[firstRow], cCorrOccs[firstRow]);
        if (!eptr) {
          fprintf(stderr, "load_coupling_response_matrix: cannot find V probe %s#%d in beamline\n",
                  cCorrNames[firstRow], cCorrOccs[firstRow]);
          exitElegant(1);
        }
        vCorrs[iC].elem = eptr;
        vCorrs[iC].paramIndex = confirm_parameter(cCorrItems[firstRow], eptr->type);
        if (vCorrs[iC].paramIndex < 0)
          bombElegant("load_coupling_response_matrix: V probe item not found on element", NULL);
        vCorrs[iC].valuePtr = (double *)(eptr->p_elem +
            entity_description[eptr->type].parameter[vCorrs[iC].paramIndex].offset);
        vCorrs[iC].initialValue = *vCorrs[iC].valuePtr;
      }
      crossXBpms = SDDS_Realloc(NULL, sizeof(*crossXBpms) * nCrossXBpm);
      crossXBpmIdx = tmalloc(sizeof(*crossXBpmIdx) * nCrossXBpm);
      long iB;
      for (iB = 0; iB < nCrossXBpm; iB++) {
        long firstRow = offsetV + iB * nSkew;
        ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline,
                              cBpmNames[firstRow], cBpmOccs[firstRow]);
        if (!eptr) {
          fprintf(stderr, "load_coupling_response_matrix: cannot find cross-X BPM %s#%d in beamline\n",
                  cBpmNames[firstRow], cBpmOccs[firstRow]);
          exitElegant(1);
        }
        crossXBpms[iB].elem = eptr;
        crossXBpmIdx[iB] = bpmElemIndex(beamline, eptr);
      }
    }
    crossClorbBase = tmalloc(sizeof(*crossClorbBase) * (beamline->n_elems + 1));
    crossClorbPert = tmalloc(sizeof(*crossClorbPert) * (beamline->n_elems + 1));
    nRowsCrossHy = nHCorr * nCrossYBpm;
    nRowsCrossVx = nVCorr * nCrossXBpm;

    /* Reallocate R to include cross-plane rows; copy etay block from the
     * previously-loaded coef[] (still in scope). */
    nRowsEtay = nBpm;
    nRowsTotal = nRowsEtay + nRowsCrossHy + nRowsCrossVx;
#if USE_MPI
    if (R && myid == 0) { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#else
    if (R) { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#endif
#if USE_MPI
    if (myid == 0)
#endif
    {
      R = (double **)czarray_2d(sizeof(**R), nRowsTotal, nSkew);
      for (i = 0; i < nBpm; i++)
        for (j = 0; j < nSkew; j++)
          R[i][j] = coef[i * nSkew + j];
      long row = 0;
      /* H -> y block */
      long iCorr, iBpm;
      for (iCorr = 0; iCorr < nHCorr; iCorr++)
        for (iBpm = 0; iBpm < nCrossYBpm; iBpm++) {
          long rRow = nBpm + iCorr * nCrossYBpm + iBpm;
          for (j = 0; j < nSkew; j++) R[rRow][j] = cCoef[row++];
        }
      /* V -> x block */
      long baseRow = nBpm + nHCorr * nCrossYBpm;
      for (iCorr = 0; iCorr < nVCorr; iCorr++)
        for (iBpm = 0; iBpm < nCrossXBpm; iBpm++) {
          long rRow = baseRow + iCorr * nCrossXBpm + iBpm;
          for (j = 0; j < nSkew; j++) R[rRow][j] = cCoef[row++];
        }
    }
    /* Default weights: leave correct_coupling free to override.  We do NOT
     * auto-enable cross_response_weight; the user explicitly opts in. */
    if (load_coupling_response_matrix_struct.verbosity > 0) {
      printf("load_coupling_response_matrix: loaded cross-plane matrix from %s "
             "(%ld H probes x %ld vmons + %ld V probes x %ld hmons)\n",
             fnx, nHCorr, nCrossYBpm, nVCorr, nCrossXBpm);
      if (crossHSpec) printf("  cross_h_steering carried in file: \"%s\"\n", crossHSpec);
      if (crossVSpec) printf("  cross_v_steering carried in file: \"%s\"\n", crossVSpec);
      fflush(stdout);
    }
    /* Free file-local buffers. */
    for (i = 0; i < nXRows; i++) {
      if (chanCol[i])     free(chanCol[i]);
      if (cCorrNames[i])  free(cCorrNames[i]);
      if (cCorrItems[i])  free(cCorrItems[i]);
      if (cBpmNames[i])   free(cBpmNames[i]);
    }
    free(chanCol);
    free(cCorrNames); free(cCorrOccs); free(cCorrItems);
    free(cBpmNames);  free(cBpmOccs);
    free(cCoef);
    if (hSpecStr) free(hSpecStr);
    if (vSpecStr) free(vSpecStr);
  }

  for (i = 0; i < nRows; i++) {
    if (bpmNames[i])  free(bpmNames[i]);
    if (skewNames[i]) free(skewNames[i]);
  }
  free(bpmNames); free(skewNames);
  free(bpmOccs);  free(skewOccs);
  free(coef);
  free(itemStr);
}

/* ============================================================ */
/* Per-step corrector reassertion hooks for vary_beamline().    */
/* See correctorStash.h for the cross-module contract.          */
/* ============================================================ */

/* Populate ccStash from the current knobs[] inventory and snapshot the
 * current skew K values.  No-op if no &correct_coupling has been issued. */
void correct_coupling_save_correctors(RUN *run, LINE_LIST *beamline) {
  long j;
  (void)run; (void)beamline;
  if (!initialized || knobs == NULL || nSkew == 0) return;
  corstash_clear(&ccStash);
  for (j = 0; j < nSkew; j++)
    corstash_add(&ccStash, knobs[j].elem, knobs[j].paramIndex);
  corstash_snapshot(&ccStash);
}

long correct_coupling_reassert_correctors(RUN *run, LINE_LIST *beamline) {
  return corstash_reassert(&ccStash, run, beamline);
}

void correct_coupling_invalidate_correctors(void) {
  corstash_clear(&ccStash);
}
