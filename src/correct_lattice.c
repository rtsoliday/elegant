/*************************************************************************\
 * Copyright (c) 2024 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: correct_lattice.c
 * purpose: normal-quadrupole correction of linear lattice functions.
 *
 * The lattice observables beta_x, beta_y, eta_x at user-selected BPMs are
 * driven toward design values (loaded from a reference twiss_output file)
 * by adjusting user-selected normal-quadrupole knobs. Each BPM contributes
 * three weighted rows to the SVD residual; the per-channel weights
 *      betax_weight, betay_weight, etax_weight
 * scale the residuals before stacking.  The response matrix
 *      R[i,j] = d obs_i / d K1_j
 * (with i indexing 3*nBpm observables and j indexing the quad knobs) is
 * built via the shared correctionEngine, SVD-pseudo-inverted, and
 * applied with the usual correction_fraction + per-knob signed-bound clamp
 * (per-family lower/upper limits set via the correction_elements family list).
 *
 * Three commands:
 *   compute_lattice_response_matrix  -- build R and save it to a file.
 *   load_lattice_response_matrix     -- load a previously-saved R.
 *   correct_lattice                  -- perform the correction (deferred to
 *                                       the simulation steps loop).
 *
 * If either compute_/load_ is issued, correct_lattice reuses the resulting R
 * and the knob/BPM/item lists rather than building from its own namelist.
 *
 * Both correct_coupling.c and this file delegate the response-matrix build,
 * SVD solve, strength clamping, knob/BPM collection, element lookup and
 * twiss recomputation to correctionEngine.c.
 */
#include "mdb.h"
#include "matlib.h"
#include "matrixOp.h"
#include "track.h"
#include "correctionEngine.h"
#include "correctorStash.h"
#include "correct_lattice.h"

/* Per-step reassertion stash; populated from knobs[] at save time. */
static CORRECTOR_STASH clStash = { NULL, NULL, NULL, NULL, 0, 0, 1, 0, "lattice" };

/* Module-level state retained between setup/do/finish.  quadName/quadType/
 * quadItem are populated only by &compute_lattice_response_matrix and
 * &load_lattice_response_matrix; the &correct_lattice namelist uses its own
 * family-list state defined further down. */
static long initialized = 0;
static char **quadName = NULL;       long nQuadName = 0;
static char **quadType = NULL;       long nQuadType = 0;
/* Parallel measurement-location/type lists.  Both arrays are nMeasPat
 * long; an element matches if some i satisfies wild_match(name,
 * measLoc[i]) && wild_match(type, measType[i]).  This generalizes the
 * old (bpm_name_pattern, bpm_type_pattern) Cartesian-product scheme
 * and lets the caller mix BPMs with non-BPM measurement points
 * (e.g. quadrupoles measured via LOCO). */
static char **measLoc  = NULL;
static char **measType = NULL;
static long   nMeasPat = 0;
static char *quadItem = NULL;        /* item used by compute_/load_ */

/* Family-list state populated from &correct_lattice's correction_elements/
 * items/lower_limits/upper_limits/exclude.  nFamily == nCorrPatterns.
 * itemsList[k] is the item for family k (always non-NULL after setup; default
 * "K1" when the user omits the items list).  lowerList/upperList are NULL
 * when the user omits the respective limit list. */
static char **corrPatterns = NULL;       static long nCorrPatterns = 0;
static char *corrPatternStr = NULL;      /* verbatim user string, for echoing */
static char **itemsList = NULL;          static long nItemsList = 0;
static double *lowerList = NULL;         static long nLowerList = 0;
static double *upperList = NULL;         static long nUpperList = 0;
static char **excludePatterns = NULL;    static long nExcludePatterns = 0;

/* Per-knob arrays sized nKnob; populated by buildPerKnobItemAndBounds() after
 * the knob inventory is finalised.  knobItem[j] is a non-owning pointer into
 * itemsList[] (when families are in use) or into quadItem (preloaded only). */
static char **knobItem = NULL;
static double *knobLower = NULL;         /* NULL ⇒ no lower bound on any knob */
static double *knobUpper = NULL;         /* NULL ⇒ no upper bound on any knob */
static long *knobFamily = NULL;          /* family index per knob, or -1 */

static SDDS_DATASET SDDSstrengthLog, SDDSresponse, SDDSrmsLog;
static short SDDSstrengthLogInit = 0, SDDSresponseInit = 0, SDDSrmsLogInit = 0;

/* Number of observables per BPM: betax, betay, etax. */
#define LCC_N_OBS 3

/* Persistent knob/BPM inventory and matrix, set by compute_/load_/setup. */
static LRC_Knob *knobs = NULL;
static LRC_Bpm  *bpms = NULL;
static long nKnob = 0, nBpm = 0;
static double *yMeas = NULL;     /* current observables, length LCC_N_OBS*nBpm */
static double *yPert = NULL;     /* perturbed observables */
static double *yTarget = NULL;   /* design observables from reference file */
static double *dK = NULL;        /* correction step, length nKnob */
static double **R = NULL;        /* response matrix, (LCC_N_OBS*nBpm) x nKnob */
static short responseValid = 0;

/* Knob grouping for bind_name_pattern: knobs sharing a group index get a
 * single bound dK and are summed into a single column of the SVD response
 * matrix.  knobGroup[j] is the group index of knob j; nGroup is the number
 * of distinct groups.  With bind_name_pattern==NULL every knob is its own
 * group (nGroup==nKnob), which reproduces the unbound behavior. */
static long *knobGroup = NULL;
static long  nGroup = 0;
static char **bindPatterns = NULL;
static long  nBindPatterns = 0;
/* Original user-supplied bind_name_pattern string, kept verbatim so it can
 * be saved into the response-matrix file and read back by load. */
static char *bindNamePatternStr = NULL;

/****************************************************************************/

static double lccRmsValue(double *v, long n) {
  long i;
  double s = 0;
  if (n <= 0) return 0;
  for (i = 0; i < n; i++) s += v[i] * v[i];
  return sqrt(s / n);
}

/* From an unweighted residual vector (layout: [bx,by,ex] per BPM), report the
 * per-channel RMS (in real units) and the weighted RMS (the metric the solver
 * actually drives down). */
static void computeRmsBreakdown(double *yResid, double *bxRms, double *byRms,
                                double *exRms, double *wRms) {
  long i;
  double sxB = 0, syB = 0, sE = 0, sw = 0;
  double wbx = betax_weight, wby = betay_weight, wex = etax_weight;
  for (i = 0; i < nBpm; i++) {
    double rx = yResid[LCC_N_OBS*i + 0];
    double ry = yResid[LCC_N_OBS*i + 1];
    double re = yResid[LCC_N_OBS*i + 2];
    sxB += rx * rx;
    syB += ry * ry;
    sE  += re * re;
    sw  += (wbx*rx)*(wbx*rx) + (wby*ry)*(wby*ry) + (wex*re)*(wex*re);
  }
  *bxRms = nBpm > 0 ? sqrt(sxB / nBpm) : 0.0;
  *byRms = nBpm > 0 ? sqrt(syB / nBpm) : 0.0;
  *exRms = nBpm > 0 ? sqrt(sE  / nBpm) : 0.0;
  *wRms  = nBpm > 0 ? sqrt(sw / (LCC_N_OBS * nBpm)) : 0.0;
}

/* Read (betax, betay, etax) at every BPM, in observable-major-by-BPM layout:
 * obs[3*i_bpm + 0] = betax, +1 = betay, +2 = etax. Adds optional measurement
 * noise (beta_measurement_noise to betax/betay, eta_measurement_noise to etax). */
static void readBetaEtaAtBpms(LRC_Bpm *bpms, long nBpm, double *obs) {
  long i;
  for (i = 0; i < nBpm; i++) {
    if (!bpms[i].elem->twiss) {
      fprintf(stderr, "correct_lattice: BPM %s has no twiss data; run twiss_output first\n",
              bpms[i].elem->name);
      exitElegant(1);
    }
    double bx = bpms[i].elem->twiss->betax;
    double by = bpms[i].elem->twiss->betay;
    double ex = bpms[i].elem->twiss->etax;
    if (beta_measurement_noise > 0) {
      bx *= (1+gauss_rn_lim(0.0, beta_measurement_noise, measurement_noise_cutoff, random_3));
      by *= (1+gauss_rn_lim(0.0, beta_measurement_noise, measurement_noise_cutoff, random_3));
    }
    if (eta_measurement_noise > 0)
      ex += gauss_rn_lim(0.0, eta_measurement_noise, measurement_noise_cutoff, random_3);
    obs[LCC_N_OBS * i + 0] = bx;
    obs[LCC_N_OBS * i + 1] = by;
    obs[LCC_N_OBS * i + 2] = ex;
  }
}

/* LRC_ReaderFn-compatible trampoline.  Reads (betax, betay, etax) at every
 * vertical-and-horizontal BPM in the module-scope bpms[] array.  nObs is the
 * total observable count (LCC_N_OBS * nBpm); the engine treats it opaquely. */
static void betaEtaReader(long nObs, double *obs, void *ctx) {
  (void)nObs; (void)ctx;
  readBetaEtaAtBpms(bpms, nBpm, obs);
}

/* Replace the module's binding state (bindNamePatternStr + parsed bindPatterns)
 * from a single user-supplied string.  Pass NULL to clear binding. */
static void setBindingFromString(const char *spec) {
  LRC_freePatternList(&bindPatterns, &nBindPatterns);
  if (bindNamePatternStr) { free(bindNamePatternStr); bindNamePatternStr = NULL; }
  if (spec && *spec) {
    cp_str(&bindNamePatternStr, (char *)spec);
    bindPatterns = addPatterns(&nBindPatterns, bindNamePatternStr);
  }
}

/****************************************************************************/
/* Assign every knob to a group.  Each entry of bind_name_pattern defines one
 * family group containing every knob whose element name matches that entry,
 * regardless of whether the matched knobs share an element name (so
 * S1:Q1, S2:Q1, ..., S40:Q1 collapse to a single Q1 family knob when
 * "*Q1" is one of the entries).  A knob matched by multiple bind patterns is
 * placed in the earliest-listed matching group (first match wins).  Knobs
 * not matched by any bind pattern get their own singleton group, preserving
 * the original per-knob behavior.  Must be called after knobs[] is populated
 * and after bindPatterns is set. */
static void computeKnobGroups(void) {
  long j, p;
  if (knobGroup) { free(knobGroup); knobGroup = NULL; }
  if (nKnob == 0) { nGroup = 0; return; }
  knobGroup = tmalloc(sizeof(*knobGroup) * nKnob);
  for (j = 0; j < nKnob; j++) knobGroup[j] = -1;
  nGroup = 0;
  for (p = 0; p < nBindPatterns; p++) {
    long familyGroup = -1;
    for (j = 0; j < nKnob; j++) {
      if (knobGroup[j] >= 0) continue;
      if (!wild_match(knobs[j].elem->name, bindPatterns[p])) continue;
      if (familyGroup < 0) familyGroup = nGroup++;
      knobGroup[j] = familyGroup;
    }
  }
  for (j = 0; j < nKnob; j++)
    if (knobGroup[j] < 0) knobGroup[j] = nGroup++;
}

/****************************************************************************/
/* Parse a whitespace-separated list of floating-point values into a freshly
 * allocated array.  *nOut receives the count.  Returns NULL when the input is
 * NULL/empty; bombs on any token that does not scan as a double. */
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
      bombElegantVA("correct_lattice: %s contains non-numeric token \"%s\"", ctx, ptr);
    vals = SDDS_Realloc(vals, sizeof(*vals) * (n + 1));
    vals[n++] = v;
  }
  free(input);
  *nOut = n;
  return vals;
}

/* Test whether element name `name` matches any pattern in the exclude list. */
static int matchesExclude(const char *name) {
  long k;
  for (k = 0; k < nExcludePatterns; k++)
    if (wild_match((char *)name, excludePatterns[k])) return 1;
  return 0;
}

/* Walk the beamline once; for each element not in the exclude list, find the
 * first family pattern matching its name and (if the element exposes the
 * family's item parameter as a double) record it as a knob.  Element types
 * lacking that item are silently skipped.  Allocates knobs[] and knobFamily[].
 * Returns the number of knobs collected. */
static long collectKnobsFromFamilies(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr = beamline->elem;
  long cap = 0;
  nKnob = 0;
  if (knobs)      { free(knobs);      knobs = NULL; }
  if (knobFamily) { free(knobFamily); knobFamily = NULL; }
  while (eptr) {
    long fam;
    for (fam = 0; fam < nCorrPatterns; fam++) {
      if (!wild_match(eptr->name, corrPatterns[fam])) continue;
      if (matchesExclude(eptr->name)) break;  /* first family matched; excluded */
      long paramIndex = confirm_parameter(itemsList[fam], eptr->type);
      if (paramIndex < 0 ||
          entity_description[eptr->type].parameter[paramIndex].type != IS_DOUBLE)
        break;  /* element matches the family but lacks the item; skip element */
      if (nKnob == cap) {
        cap = cap ? 2 * cap : 32;
        knobs      = SDDS_Realloc(knobs,      sizeof(*knobs)      * cap);
        knobFamily = SDDS_Realloc(knobFamily, sizeof(*knobFamily) * cap);
      }
      knobs[nKnob].elem        = eptr;
      knobs[nKnob].paramIndex  = paramIndex;
      knobs[nKnob].valuePtr    = (double *)(eptr->p_elem +
          entity_description[eptr->type].parameter[paramIndex].offset);
      knobs[nKnob].initialValue = *knobs[nKnob].valuePtr;
      knobFamily[nKnob] = fam;
      nKnob++;
      break;  /* first-match-wins; do not consider later families */
    }
    eptr = eptr->succ;
  }
  return nKnob;
}

/* Free the per-knob owned strings in knobItem[] and the array itself. */
static void freeKnobItem(void) {
  if (!knobItem) return;
  long j;
  for (j = 0; j < nKnob; j++)
    if (knobItem[j]) free(knobItem[j]);
  free(knobItem);
  knobItem = NULL;
}

/* Build the per-knob (item, lower, upper) arrays.  Assumes knobs[] and (when
 * families are in use) knobFamily[] are populated.  knobItem[] owns its
 * strings (cp_str'd) so the per-knob items survive after itemsList[] is
 * freed by setup_load_lattice_response_matrix or by a later setup call.
 *
 * When the caller supplied a family list (nCorrPatterns>0), the per-knob
 * (item, lower, upper) are rebuilt from it -- this is the only path through
 * which lower/upper become set.  When no family list was supplied but a
 * preloaded knobItem[] already exists (load_lattice_response_matrix populated
 * it from the matrix file, possibly with per-knob items), it is left intact.
 * Otherwise, every knob inherits quadItem (the legacy single-item field). */
static void buildPerKnobItemAndBounds(void) {
  long j;
  if (knobLower) { free(knobLower); knobLower = NULL; }
  if (knobUpper) { free(knobUpper); knobUpper = NULL; }
  if (nKnob == 0) { freeKnobItem(); return; }
  if (nCorrPatterns > 0 && knobFamily) {
    freeKnobItem();
    knobItem = tmalloc(sizeof(*knobItem) * nKnob);
    for (j = 0; j < nKnob; j++) cp_str(&knobItem[j], itemsList[knobFamily[j]]);
    if (lowerList) {
      knobLower = tmalloc(sizeof(*knobLower) * nKnob);
      for (j = 0; j < nKnob; j++) knobLower[j] = lowerList[knobFamily[j]];
    }
    if (upperList) {
      knobUpper = tmalloc(sizeof(*knobUpper) * nKnob);
      for (j = 0; j < nKnob; j++) knobUpper[j] = upperList[knobFamily[j]];
    }
  } else if (knobItem == NULL) {
    knobItem = tmalloc(sizeof(*knobItem) * nKnob);
    for (j = 0; j < nKnob; j++) cp_str(&knobItem[j], quadItem ? quadItem : "K1");
  }
  /* else: preloaded per-knob knobItem[] is preserved as-is */
}

/* When a preloaded inventory exists AND the user supplied a family list, the
 * family list serves only to assign items/bounds to the already-collected
 * knobs (by name match).  This routine fills knobFamily[] in that case and
 * additionally bombs if any preloaded knob would otherwise be associated with
 * a family whose item disagrees with the preloaded quadItem (since the
 * response matrix is column-bound to quadItem). */
static void assignFamiliesToPreloadedKnobs(void) {
  long j, k;
  if (knobFamily) { free(knobFamily); knobFamily = NULL; }
  if (nKnob == 0 || nCorrPatterns == 0) return;
  knobFamily = tmalloc(sizeof(*knobFamily) * nKnob);
  for (j = 0; j < nKnob; j++) {
    knobFamily[j] = -1;
    for (k = 0; k < nCorrPatterns; k++) {
      if (!wild_match(knobs[j].elem->name, corrPatterns[k])) continue;
      if (matchesExclude(knobs[j].elem->name))
        bombElegantVA("correct_lattice: preloaded knob %s is matched by exclude pattern "
                      "but cannot be removed once the response matrix is built; "
                      "rebuild the matrix without this element",
                      knobs[j].elem->name);
      if (quadItem && itemsList && strcmp(quadItem, itemsList[k]) != 0)
        bombElegantVA("correct_lattice: preloaded knob %s would be assigned to family "
                      "\"%s\" whose item \"%s\" disagrees with the preloaded item \"%s\"",
                      knobs[j].elem->name, corrPatterns[k], itemsList[k], quadItem);
      knobFamily[j] = k;
      break;
    }
    /* Unmatched preloaded knobs stay knobFamily[j] = -1 and get no per-knob
     * limit (their entry in knobLower/knobUpper will be unbounded). */
  }
}

/****************************************************************************/
/* Walk the beamline, collect knobs/BPMs (lazy), allocate working buffers. */
static long collectAndAllocate(LINE_LIST *beamline) {
  if (knobs == NULL) {
    /* No preloaded inventory ⇒ build the knob inventory from the family list. */
    if (nCorrPatterns == 0)
      bombElegant("correct_lattice: correction_elements must be supplied "
                  "(unless compute_lattice_response_matrix or "
                  "load_lattice_response_matrix has been issued)", NULL);
    collectKnobsFromFamilies(beamline);
    nBpm = LRC_collectBpmsParallel(beamline, measLoc, measType, nMeasPat, &bpms);
    if (nKnob == 0)
      bombElegant("correct_lattice: no knobs matched correction_elements", NULL);
    if (nBpm == 0)
      bombElegant("correct_lattice: no elements matched measurement_elements/measurement_types", NULL);
    buildPerKnobItemAndBounds();
  } else {
    /* Preloaded inventory; assign families (and per-knob bounds) if the user
     * supplied a family list on this correct_lattice command. */
    assignFamiliesToPreloadedKnobs();
    buildPerKnobItemAndBounds();
    /* Re-derive per-knob items/bounds.  For unmatched preloaded knobs, the
     * lower/upper entry stays unbounded by way of leaving lower/upper arrays
     * disabled below. */
    if (nCorrPatterns == 0) {
      /* No family list at all ⇒ leave knobLower/knobUpper NULL (unbounded). */
    } else if (knobLower || knobUpper) {
      /* Per-knob arrays have been built; for any j with knobFamily[j]==-1 we
       * need to clear that entry to "unbounded".  Sentinel: set the entry to
       * a value that disables the clamp on that side.  The asymmetric form of
       * LRC_clampStepToLimitArray inspects each knob independently, so we
       * encode "no bound" by placing the bound at +/-infty. */
      long j;
      if (knobLower)
        for (j = 0; j < nKnob; j++)
          if (knobFamily[j] < 0) knobLower[j] = -DBL_MAX;
      if (knobUpper)
        for (j = 0; j < nKnob; j++)
          if (knobFamily[j] < 0) knobUpper[j] = DBL_MAX;
    }
  }
  if (yMeas == NULL) {
    long nRows = LCC_N_OBS * nBpm;
    yMeas   = tmalloc(sizeof(*yMeas)   * nRows);
    yPert   = tmalloc(sizeof(*yPert)   * nRows);
    yTarget = tmalloc(sizeof(*yTarget) * nRows);
    dK      = tmalloc(sizeof(*dK)      * nKnob);
#if USE_MPI
    if (myid == 0)
      R = (double **)czarray_2d(sizeof(**R), nRows, nKnob);
#else
    R = (double **)czarray_2d(sizeof(**R), nRows, nKnob);
#endif
  }
  return 1;
}

static void freeModuleState(void) {
  if (yMeas)   { free(yMeas);   yMeas   = NULL; }
  if (yPert)   { free(yPert);   yPert   = NULL; }
  if (yTarget) { free(yTarget); yTarget = NULL; }
  if (dK)      { free(dK);      dK      = NULL; }
#if USE_MPI
  if (R && myid == 0) { free_czarray_2d((void **)R, LCC_N_OBS*nBpm, nKnob); R = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, LCC_N_OBS*nBpm, nKnob); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms  = NULL; }
  LRC_freePatternList(&quadName,    &nQuadName);
  LRC_freePatternList(&quadType,    &nQuadType);
  LRC_freePatternList(&measLoc,     &nMeasPat);
  /* measType is parallel to measLoc and shares the same count; free
   * separately but reset nMeasPat only once. */
  {
    long dummy = nMeasPat;
    LRC_freePatternList(&measType, &dummy);
    nMeasPat = 0;
  }
  LRC_freePatternList(&bindPatterns, &nBindPatterns);
  if (bindNamePatternStr) { free(bindNamePatternStr); bindNamePatternStr = NULL; }
  if (knobGroup) { free(knobGroup); knobGroup = NULL; }
  nGroup = 0;
  if (quadItem) { free(quadItem); quadItem = NULL; }
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
  if (knobFamily) { free(knobFamily); knobFamily = NULL; }
  nKnob = nBpm = 0;
  responseValid = 0;
}

/****************************************************************************/
/* Load the design target (betax, betay, etax at each BPM) from a twiss file.
 * Each BPM is matched by ElementName + (ElementOccurence if present). */
static void loadTargetFromReferenceFile(char *filename) {
  SDDS_DATASET in;
  char **nameCol = NULL;
  int32_t *occCol = NULL;
  double *bxCol = NULL, *byCol = NULL, *exCol = NULL;
  long nRows, i, j;
  short haveOccurence;

  if (!SDDS_InitializeInputFromSearchPath(&in, filename) || SDDS_ReadPage(&in) != 1)
    SDDS_Bomb("correct_lattice: cannot open or read reference_file");
  if (SDDS_CheckColumn(&in, "ElementName", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "betax", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "betay", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "etax",  NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
    SDDS_Bomb("correct_lattice: reference_file missing required columns ElementName/betax/betay/etax");
  haveOccurence = (SDDS_CheckColumn(&in, "ElementOccurence", NULL,
                                    SDDS_ANY_INTEGER_TYPE, NULL) == SDDS_CHECK_OK);

  nRows = SDDS_CountRowsOfInterest(&in);
  if (!(nameCol = SDDS_GetColumn(&in, "ElementName")) ||
      !(bxCol = SDDS_GetColumnInDoubles(&in, "betax")) ||
      !(byCol = SDDS_GetColumnInDoubles(&in, "betay")) ||
      !(exCol = SDDS_GetColumnInDoubles(&in, "etax")))
    SDDS_Bomb("correct_lattice: error reading reference_file columns");
  if (haveOccurence && !(occCol = SDDS_GetColumnInLong(&in, "ElementOccurence")))
    SDDS_Bomb("correct_lattice: error reading reference_file ElementOccurence");
  SDDS_Terminate(&in);

  /* For each of our BPMs, locate a matching row in the reference. */
  for (i = 0; i < nBpm; i++) {
    long match = -1;
    for (j = 0; j < nRows; j++) {
      if (strcmp(nameCol[j], bpms[i].elem->name) != 0) continue;
      if (haveOccurence && occCol[j] != bpms[i].elem->occurence) continue;
      match = j;
      break;
    }
    if (match < 0) {
      fprintf(stderr, "correct_lattice: BPM %s#%ld not found in reference_file\n",
              bpms[i].elem->name, bpms[i].elem->occurence);
      exitElegant(1);
    }
    yTarget[LCC_N_OBS * i + 0] = bxCol[match];
    yTarget[LCC_N_OBS * i + 1] = byCol[match];
    yTarget[LCC_N_OBS * i + 2] = exCol[match];
  }

  for (j = 0; j < nRows; j++) free(nameCol[j]);
  free(nameCol);
  if (occCol) free(occCol);
  free(bxCol); free(byCol); free(exCol);
}

/****************************************************************************/
/* Build R and (if response_file is open) write a diagnostic page. */
static void buildResponseMatrix(RUN *run, LINE_LIST *beamline, long iterTag, double perturbation) {
  long i, j;

  /* Bracket with parallelTrackingBasedMatrices = 0, matching
   * compute_orbcor_matrices1p()'s pattern.  Although betaEtaReader itself is
   * a pure twiss-memory read, the engine's per-knob LRC_retwiss invokes
   * update_twiss_parameters -> fill_in_matrices -> compute_matrix(), which
   * can dispatch to tracking-based matrix computation for elements like
   * KQUAD with TRACKING_MATRIX=1, APPLE, KICKMAP, or high-order CSBEND.
   * Those tracking paths expect every rank at the same call; the engine's
   * outer-parallel split puts only the knob's owning rank inside, so the
   * collective tracking would deadlock with no peers.  Clearing the flag
   * forces single-particle tracking on the calling rank and is a no-op for
   * pure-analytic elements. */
  long parTrackSave = parallelTrackingBasedMatrices;
  parallelTrackingBasedMatrices = 0;
  LRC_buildResponseMatrix(run, beamline, knobs, nKnob, LCC_N_OBS * nBpm,
                          betaEtaReader, NULL, perturbation, R);
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
    if (!SDDS_StartPage(&SDDSresponse, nBpm * nKnob) ||
        !SDDS_SetParameters(&SDDSresponse, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Iteration", iterTag, NULL))
      SDDS_Bomb("correct_lattice: error writing response_file page");
    for (i = 0; i < nBpm; i++)
      for (j = 0; j < nKnob; j++) {
        if (!SDDS_SetRowValues(&SDDSresponse, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                               "BPMName",   bpms[i].elem->name,
                               "sBPM",      bpms[i].elem->end_pos,
                               "QuadName",  knobs[j].elem->name,
                               "sQuad",     knobs[j].elem->end_pos,
                               "dBetaxDK1", R[LCC_N_OBS*i+0][j],
                               "dBetayDK1", R[LCC_N_OBS*i+1][j],
                               "dEtaxDK1",  R[LCC_N_OBS*i+2][j], NULL))
          SDDS_Bomb("correct_lattice: error setting response_file row");
        row++;
      }
    if (!SDDS_WritePage(&SDDSresponse))
      SDDS_Bomb("correct_lattice: error writing response_file page");
  }
}

/* Apply the per-channel weights to R and y in-place (rows are already laid out
 * by observable: row LCC_N_OBS*i+k is observable k at BPM i). */
static void applyWeights(double **Rw, long nRows, long nKnob, double *yw) {
  long i, j;
  double w[LCC_N_OBS];
  w[0] = betax_weight;
  w[1] = betay_weight;
  w[2] = etax_weight;
  for (i = 0; i < nRows; i++) {
    double wi = w[i % LCC_N_OBS];
    yw[i] *= wi;
    if (Rw)
      for (j = 0; j < nKnob; j++)
        Rw[i][j] *= wi;
  }
}

/****************************************************************************/

void setup_correct_lattice(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&correct_lattice, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &correct_lattice);

  if (reference_file == NULL)
    bombElegant("correct_lattice: reference_file is required", NULL);
  if (n_iterations < 0)
    bombElegant("correct_lattice: n_iterations must be >= 0", NULL);
  if (correction_fraction <= 0 || correction_fraction > 1)
    bombElegant("correct_lattice: correction_fraction must be in (0, 1]", NULL);
  if (change_tolerance < 0 || change_tolerance >= 1)
    bombElegant("correct_lattice: change_tolerance must be in [0, 1)", NULL);
  if (response_perturbation <= 0)
    bombElegant("correct_lattice: response_perturbation must be > 0", NULL);
  if (svd_threshold < 0)
    bombElegant("correct_lattice: svd_threshold must be >= 0", NULL);
  if (n_singular_values < 0)
    bombElegant("correct_lattice: n_singular_values must be >= 0", NULL);
  if (beta_measurement_noise < 0 || eta_measurement_noise < 0)
    bombElegant("correct_lattice: measurement_noise values must be >= 0", NULL);
  if ((beta_measurement_noise > 0 || eta_measurement_noise > 0) && measurement_noise_cutoff <= 0)
    bombElegant("correct_lattice: measurement_noise_cutoff must be > 0 when noise is enabled", NULL);
  if (betax_weight <= 0 || betay_weight <= 0 || etax_weight <= 0)
    bombElegant("correct_lattice: channel weights must be > 0", NULL);

  /* Per-step reassertion control honored by vary_beamline(). */
  clStash.reassert = reset_correctors_each_step ? 1 : 0;

  /* Family-list parsing.  Same convention as &correct_tunes:
   * whitespace/comma-separated, parallel to correction_elements.
   *
   * When a matrix was preloaded by load_/compute_lattice_response_matrix and
   * carried CorrectionElements/CorrectionItems metadata, corrPatterns/itemsList
   * are already populated.  Preserve them when the user didn't supply
   * correction_elements/items; when the user did supply, validate against the
   * preloaded values and bomb on disagreement. */
  LRC_freePatternList(&excludePatterns,  &nExcludePatterns);
  if (lowerList) { free(lowerList); lowerList = NULL; }
  if (upperList) { free(upperList); upperList = NULL; }
  nLowerList = nUpperList = 0;

  if (correction_elements && *correction_elements) {
    if (responseValid && corrPatternStr && strcmp(corrPatternStr, correction_elements) != 0)
      bombElegantVA("correct_lattice: correction_elements=\"%s\" on this command "
                    "disagrees with the loaded matrix's CorrectionElements=\"%s\"; "
                    "either omit correction_elements (to inherit) or rebuild the "
                    "matrix with the desired family list",
                    correction_elements, corrPatternStr);
    LRC_freePatternList(&corrPatterns, &nCorrPatterns);
    if (corrPatternStr) { free(corrPatternStr); corrPatternStr = NULL; }
    LRC_freePatternList(&itemsList, &nItemsList);
    cp_str(&corrPatternStr, correction_elements);
    corrPatterns = addPatterns(&nCorrPatterns, corrPatternStr);
    if (nCorrPatterns == 0)
      bombElegant("correct_lattice: correction_elements parsed to zero patterns", NULL);
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
        bombElegantVA("correct_lattice: items has %ld entries but correction_elements has %ld",
                      ii, nCorrPatterns);
    } else {
      long ii;
      for (ii = 0; ii < nCorrPatterns; ii++)
        cp_str(&itemsList[ii], "K1");
    }
  } else if (items && *items) {
    if (!responseValid || nItemsList == 0)
      bombElegant("correct_lattice: items supplied without correction_elements; "
                  "requires a preloaded matrix that carries the family list", NULL);
    char *icopy;
    cp_str(&icopy, items);
    char *tok;
    long ii = 0;
    while ((tok = get_token(icopy))) {
      if (ii >= nItemsList || strcmp(tok, itemsList[ii]) != 0)
        bombElegantVA("correct_lattice: items on this command disagrees with the "
                      "loaded matrix's CorrectionItems (entry %ld: \"%s\" vs \"%s\")",
                      ii, tok, ii < nItemsList ? itemsList[ii] : "(none)");
      ii++;
    }
    if (ii != nItemsList)
      bombElegantVA("correct_lattice: items has %ld entries but loaded matrix has %ld families",
                    ii, nItemsList);
    free(icopy);
  } else if (!responseValid) {
    bombElegant("correct_lattice: correction_elements must be supplied "
                "(unless compute_lattice_response_matrix or "
                "load_lattice_response_matrix has been issued)", NULL);
  }

  /* Bound lists and exclude apply regardless of whether the family list came
   * from this namelist or from a preload. */
  lowerList = parseDoubleList(lower_limits, &nLowerList, "lower_limits");
  upperList = parseDoubleList(upper_limits, &nUpperList, "upper_limits");
  if (lowerList && nLowerList != nCorrPatterns)
    bombElegantVA("correct_lattice: lower_limits has %ld entries but correction_elements has %ld",
                  nLowerList, nCorrPatterns);
  if (upperList && nUpperList != nCorrPatterns)
    bombElegantVA("correct_lattice: upper_limits has %ld entries but correction_elements has %ld",
                  nUpperList, nCorrPatterns);
  if (lowerList && upperList) {
    long ii;
    for (ii = 0; ii < nCorrPatterns; ii++)
      if (lowerList[ii] > upperList[ii])
        bombElegantVA("correct_lattice: lower_limits[%ld]=%le > upper_limits[%ld]=%le",
                      ii, lowerList[ii], ii, upperList[ii]);
  }
  if (exclude && *exclude) {
    char *ecopy;
    cp_str(&ecopy, exclude);
    excludePatterns = addPatterns(&nExcludePatterns, ecopy);
    free(ecopy);
  }

  /* Measurement-location/type patterns: parsed only when we're going
   * to build the inventory.  When a preloaded matrix exists the BPM
   * list is already populated and we leave the patterns alone. */
  if (!responseValid) {
    long nLoc = 0, nType = 0;
    LRC_freePatternList(&measLoc, &nMeasPat);
    {
      long dummy = nMeasPat;
      LRC_freePatternList(&measType, &dummy);
      nMeasPat = 0;
    }
    measLoc  = addPatterns(&nLoc,  measurement_elements);
    measType = addPatterns(&nType, measurement_types);
    if (nLoc != nType)
      bombElegantVA("correct_lattice: measurement_elements has %ld pattern(s) "
                    "but measurement_types has %ld; the two lists must be "
                    "parallel (same token count)", nLoc, nType);
    if (nLoc == 0)
      bombElegant("correct_lattice: measurement_elements / measurement_types "
                  "must be supplied (no measurement points selected)", NULL);
    nMeasPat = nLoc;
  }
  /* bind_name_pattern: if explicitly supplied on this command, it overrides
   * whatever was set by a prior compute_/load_lattice_response_matrix.
   * If NULL on this command, the binding from compute/load (if any) is
   * preserved -- this is how a pre-computed matrix can travel with its
   * intended binding into a subsequent correct_lattice. */
  if (bind_name_pattern != NULL)
    setBindingFromString(bind_name_pattern);

  /* SDDS output setup */
#if USE_MPI
  if (myid == 0) {
#endif
  if (SDDSstrengthLogInit) { SDDS_Terminate(&SDDSstrengthLog); SDDSstrengthLogInit = 0; }
  if (SDDSresponseInit)    { SDDS_Terminate(&SDDSresponse);    SDDSresponseInit    = 0; }
  if (SDDSrmsLogInit)      { SDDS_Terminate(&SDDSrmsLog);      SDDSrmsLogInit      = 0; }

  if (strength_log) {
    char *fn = compose_filename(strength_log, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSstrengthLog, SDDS_BINARY, 0,
                                      "Quadrupole parameter changes applied by correct_lattice",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSstrengthLog, "ElementName", NULL, NULL,
                          "Quadrupole element name", NULL, SDDS_STRING, 0) < 0 ||
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
      fprintf(stderr, "correct_lattice: unable to set up strength_log %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSstrengthLogInit = 1;
  }

  if (response_file) {
    char *fn = compose_filename(response_file, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSresponse, SDDS_BINARY, 0,
                                      "Lattice response matrix used by correct_lattice",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSresponse, "BPMName", NULL, NULL,
                          "BPM (observation point) element name",
                          NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "sBPM", "s$bBPM$n", "m",
                          "BPM position", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "QuadName", NULL, NULL,
                          "Quadrupole (knob) element name", NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "sQuad", "s$bquad$n", "m",
                          "Quad position", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "dBetaxDK1", NULL, "m$a2$n",
                          "Response: d beta_x(BPM)/d parameter(quad)", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "dBetayDK1", NULL, "m$a2$n",
                          "Response: d beta_y(BPM)/d parameter(quad)", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSresponse, "dEtaxDK1", NULL, "m$a2$n",
                          "Response: d eta_x(BPM)/d parameter(quad)", NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_DefineSimpleParameter(&SDDSresponse, "Iteration", NULL, SDDS_LONG) ||
        !SDDS_WriteLayout(&SDDSresponse)) {
      fprintf(stderr, "correct_lattice: unable to set up response_file %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSresponseInit = 1;
  }

  if (rms_log) {
    char *fn = compose_filename(rms_log, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSrmsLog, SDDS_BINARY, 0,
                                      "Per-iteration RMS residuals from correct_lattice",
                                      NULL, fn) ||
        SDDS_DefineColumn(&SDDSrmsLog, "Iteration", NULL, NULL,
                          "Iteration index within this correction call "
                          "(0 = initial state, k+1 = state after iteration k)",
                          NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "BetaxRms", NULL, "m",
                          "RMS of (betax - betax_target) at the BPMs (unweighted)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "BetayRms", NULL, "m",
                          "RMS of (betay - betay_target) at the BPMs (unweighted)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "EtaxRms", NULL, "m",
                          "RMS of (etax - etax_target) at the BPMs (unweighted)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDSrmsLog, "WeightedRms", NULL, NULL,
                          "RMS of the weighted residual the SVD solver drives down",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineParameter(&SDDSrmsLog, "BetaxWeight", NULL, NULL,
                             "Per-channel weight applied to betax residuals",
                             NULL, SDDS_DOUBLE, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSrmsLog, "BetayWeight", NULL, NULL,
                             "Per-channel weight applied to betay residuals",
                             NULL, SDDS_DOUBLE, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSrmsLog, "EtaxWeight", NULL, NULL,
                             "Per-channel weight applied to etax residuals",
                             NULL, SDDS_DOUBLE, NULL) < 0 ||
        !SDDS_WriteLayout(&SDDSrmsLog)) {
      fprintf(stderr, "correct_lattice: unable to set up rms_log %s\n", fn);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDSrmsLogInit = 1;
  }
#if USE_MPI
  }
#endif

  /* Build matrix at setup time if use_perturbed_matrix==0 and no preload. */
  if (!responseValid && use_perturbed_matrix == 0) {
    collectAndAllocate(beamline);
    LRC_retwiss(run, beamline, NULL);
    buildResponseMatrix(run, beamline, -1, response_perturbation);
  }

  /* Make sure we have buffers even if response was preloaded. */
  collectAndAllocate(beamline);
  /* Always load fresh targets from the reference file. */
  loadTargetFromReferenceFile(reference_file);
  /* Now that knobs[] is known, compute the bind-grouping. */
  computeKnobGroups();
  if (verbosity > 0 && nGroup != nKnob) {
    printf("correct_lattice: bind_name_pattern collapses %ld knobs into %ld bound groups\n",
           nKnob, nGroup);
    fflush(stdout);
  }

  initialized = 1;
}

/****************************************************************************/

long do_correct_lattice(RUN *run, LINE_LIST *beamline) {
  long iter, i, j;
  double rms0, rms;
  double minSV, maxSV;
  long nUsedSV;
  long nRows;

  if (!initialized)
    return 0;

  /* Bracket the entire routine with parallelTrackingBasedMatrices = 0.  The
   * per-iteration application of corrections recomputes the changed elements'
   * matrices via compute_matrix(), which for tracking-based-matrix element
   * types (KQUAD/TRACKING_MATRIX, APPLE, KICKMAP, high-order CSBEND, ...)
   * would otherwise dispatch to collective parallel tracking and deadlock
   * under Pelegant.  See same logic in buildResponseMatrix above. */
  long parTrackSave = parallelTrackingBasedMatrices;
  parallelTrackingBasedMatrices = 0;

  if (!(beamline->flags & BEAMLINE_TWISS_DONE)) {
    printWarning("correct_lattice: twiss_output must be issued before correct_lattice; nothing done",
                 NULL);
    parallelTrackingBasedMatrices = parTrackSave;
    return 1;
  }

  collectAndAllocate(beamline);
  nRows = LCC_N_OBS * nBpm;

  if (verbosity > 0) {
    printf("correct_lattice: %ld quad knobs, %ld BPM observation points (%ld observables total)\n",
           nKnob, nBpm, nRows);
    fflush(stdout);
  }

  short strengthLogPageOpen = 0;
  if (use_perturbed_matrix >= 1)
    responseValid = 0;
  double currentFraction = correction_fraction;
  double prevRMS = -1.0;
  long rmsLogRow = 0;
  double bxRmsCh, byRmsCh, exRmsCh, wRmsCh;

  /* Working buffers for the weighted SVD solve. */
  double *yResid = tmalloc(sizeof(*yResid) * nRows);
  double **Rweighted = (double **)czarray_2d(sizeof(double), nRows, nKnob);
  /* The SVD operates on the group-collapsed matrix (nRows x nGroup). When
   * no binding is in effect, nGroup == nKnob, knobGroup is the identity, and
   * Rgrouped/dKgroup are just renames of Rweighted/dK -- but allocating
   * separately keeps the logic uniform. */
  double **Rgrouped = (double **)czarray_2d(sizeof(double), nRows, nGroup);
  double *dKgroup = tmalloc(sizeof(*dKgroup) * nGroup);

  if (SDDSrmsLogInit) {
    if (!SDDS_StartPage(&SDDSrmsLog, n_iterations + 1) ||
        !SDDS_SetParameters(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "BetaxWeight", (double)betax_weight,
                            "BetayWeight", (double)betay_weight,
                            "EtaxWeight",  (double)etax_weight, NULL))
      SDDS_Bomb("correct_lattice: error starting rms_log page");
  }

  /* Initial measurement */
  LRC_retwiss(run, beamline, NULL);
  readBetaEtaAtBpms(bpms, nBpm, yMeas);
  for (i = 0; i < nRows; i++) yResid[i] = yMeas[i] - yTarget[i];
  rms0 = lccRmsValue(yResid, nRows);
  if (SDDSrmsLogInit) {
    computeRmsBreakdown(yResid, &bxRmsCh, &byRmsCh, &exRmsCh, &wRmsCh);
    if (!SDDS_SetRowValues(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, rmsLogRow,
                           "Iteration",   (long)0,
                           "BetaxRms",    bxRmsCh,
                           "BetayRms",    byRmsCh,
                           "EtaxRms",     exRmsCh,
                           "WeightedRms", wRmsCh, NULL))
      SDDS_Bomb("correct_lattice: error setting rms_log row 0");
    rmsLogRow++;
  }
  if (verbosity > 0) {
    printf("  initial weighted RMS residual = %le\n", rms0);
    fflush(stdout);
  }

  /* prevKnobBefore retains the knob values from the start of the previous
   * iteration's apply so the adaptive_step block can roll back a step that
   * made things worse, not just shrink the next step. */
  double *prevKnobBefore = NULL;
  for (iter = 0; iter < n_iterations; iter++) {
    if (adaptive_step && prevRMS >= 0 && rms0 > prevRMS) {
      currentFraction *= 0.5;
      if (verbosity > 0) {
        printf("  iteration %ld: RMS grew %le -> %le; halving correction_fraction to %le\n",
               iter, prevRMS, rms0, currentFraction);
        fflush(stdout);
      }
      if (prevKnobBefore && nKnob > 0) {
        for (j = 0; j < nKnob; j++) {
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
        readBetaEtaAtBpms(bpms, nBpm, yMeas);
        for (i = 0; i < nRows; i++) yResid[i] = yMeas[i] - yTarget[i];
        rms0 = lccRmsValue(yResid, nRows);
        if (verbosity > 0) {
          printf("  iteration %ld: rolled back to previous-iteration values; weighted RMS residual now %le\n",
                 iter, rms0);
          fflush(stdout);
        }
      }
    }
    prevRMS = rms0;

    if (rms0 < convergence) {
      if (verbosity > 0) {
        printf("  iteration %ld: weighted RMS residual %le below convergence %le; stopping\n",
               iter, rms0, convergence);
        fflush(stdout);
      }
      if (strengthLogPageOpen) {
        if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Stage", "corrected", NULL) ||
            !SDDS_WritePage(&SDDSstrengthLog))
          SDDS_Bomb("correct_lattice: error writing strength_log page");
        strengthLogPageOpen = 0;
      }
      break;
    }

    if (!responseValid)
      buildResponseMatrix(run, beamline, iter, response_perturbation);

    /* Save the "before" knob values to enable rollback if the applied step
     * destabilises the lattice. */
    double *knobBefore = tmalloc(sizeof(*knobBefore) * nKnob);
    for (j = 0; j < nKnob; j++) knobBefore[j] = *knobs[j].valuePtr;

    /* SVD retry loop: solve, apply, check stability; bump the SV-cutoff
     * threshold on instability when auto_sv_threshold is enabled.  The
     * working threshold is local to this iteration. */
    double workingSvdThreshold = svd_threshold;
    double scale = 0;
    int retryUnstable = 0;
    long autoRetry;
    const long maxAutoRetries = 20;
    for (autoRetry = 0; autoRetry <= maxAutoRetries; autoRetry++) {
#if USE_MPI
      if (myid == 0) {
#endif
        for (i = 0; i < nRows; i++) {
          yResid[i] = yMeas[i] - yTarget[i];
          for (j = 0; j < nKnob; j++)
            Rweighted[i][j] = R[i][j];
        }
        applyWeights(Rweighted, nRows, nKnob, yResid);
        long g;
        for (i = 0; i < nRows; i++)
          for (g = 0; g < nGroup; g++) Rgrouped[i][g] = 0;
        for (i = 0; i < nRows; i++)
          for (j = 0; j < nKnob; j++) Rgrouped[i][knobGroup[j]] += Rweighted[i][j];
        LRC_svdSolve(Rgrouped, nRows, nGroup, yResid, dKgroup,
                     workingSvdThreshold, n_singular_values,
                     &minSV, &maxSV, &nUsedSV);
        for (j = 0; j < nKnob; j++) dK[j] = dKgroup[knobGroup[j]];
        if (verbosity > 1) {
          printf("  iteration %ld: SVD used %ld of %ld singular values; SV range [%le, %le]; threshold %le\n",
                 iter, nUsedSV, MIN(nGroup, nRows), minSV, maxSV, workingSvdThreshold);
          fflush(stdout);
        }
#if USE_MPI
      }
      MPI_Bcast(dK, nKnob, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nUsedSV, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#endif

      if (currentFraction != 1.0)
        for (j = 0; j < nKnob; j++) dK[j] *= currentFraction;

      scale = LRC_clampStepToLimitArray(knobs, dK, nKnob, knobLower, knobUpper);
      if (scale == 0 || nUsedSV == 0) {
        /* No step possible; restore baseline state. */
        for (j = 0; j < nKnob; j++) {
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
      for (j = 0; j < nKnob; j++) {
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

      short suppressWarn = auto_sv_threshold && (autoRetry < maxAutoRetries);
      if (suppressWarn) pushWarningSuppression();
      retryUnstable = LRC_retwiss_status(run, beamline, NULL, !suppressWarn);
      if (suppressWarn) popWarningSuppression();

      if (!retryUnstable) break;
      if (!auto_sv_threshold) break;
      if (autoRetry == maxAutoRetries) {
        if (verbosity > 0) {
          printf("  iteration %ld: auto_sv_threshold retries exhausted; "
                 "rolling back step and skipping this iteration\n", iter);
          fflush(stdout);
        }
        for (j = 0; j < nKnob; j++) {
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
        scale = 0;
        break;
      }

      /* Roll back and bump threshold for next attempt. */
      double newThreshold = workingSvdThreshold * auto_sv_threshold_factor;
      if (verbosity > 0) {
        printf("  iteration %ld: unstable twiss; rolling back, svd_threshold %le -> %le (retry %ld)\n",
               iter, workingSvdThreshold, newThreshold, autoRetry + 1);
        fflush(stdout);
      }
      workingSvdThreshold = newThreshold;
      for (j = 0; j < nKnob; j++) {
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
    }

    if (scale == 0) {
      if (verbosity > 0)
        printf("  iteration %ld: no correction applied (per-knob strength limit reached or all SVs dropped)\n",
               iter);
      free(knobBefore);
      if (strengthLogPageOpen) {
        if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Stage", "corrected", NULL) ||
            !SDDS_WritePage(&SDDSstrengthLog))
          SDDS_Bomb("correct_lattice: error writing strength_log page");
        strengthLogPageOpen = 0;
      }
      break;
    }
    if (scale < 1 && verbosity > 0) {
      printf("  iteration %ld: step scaled by %le to respect per-knob strength limits\n",
             iter, scale);
      fflush(stdout);
    }

    if (strengthLogPageOpen) {
      if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Stage", "uncorrected", NULL) ||
          !SDDS_WritePage(&SDDSstrengthLog))
        SDDS_Bomb("correct_lattice: error writing strength_log page");
      strengthLogPageOpen = 0;
    }
    if (SDDSstrengthLogInit) {
      if (!SDDS_StartPage(&SDDSstrengthLog, nKnob) ||
          !SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Iteration", iter, NULL))
        SDDS_Bomb("correct_lattice: error writing strength_log page");
    }
    for (j = 0; j < nKnob; j++) {
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
          SDDS_Bomb("correct_lattice: error setting strength_log row");
      }
    }
    /* Transfer this iteration's "before" snapshot to prevKnobBefore so the
     * next iteration's adaptive_step block can roll back this iteration's
     * apply if it ends up making the residual worse. */
    if (prevKnobBefore) free(prevKnobBefore);
    prevKnobBefore = knobBefore;
    knobBefore = NULL;
    if (SDDSstrengthLogInit)
      strengthLogPageOpen = 1;

    /* Final retwiss already done inside the retry loop. */
    readBetaEtaAtBpms(bpms, nBpm, yMeas);
    for (i = 0; i < nRows; i++) yResid[i] = yMeas[i] - yTarget[i];
    rms = lccRmsValue(yResid, nRows);
    if (SDDSrmsLogInit) {
      computeRmsBreakdown(yResid, &bxRmsCh, &byRmsCh, &exRmsCh, &wRmsCh);
      if (!SDDS_SetRowValues(&SDDSrmsLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, rmsLogRow,
                             "Iteration",   (long)(iter + 1),
                             "BetaxRms",    bxRmsCh,
                             "BetayRms",    byRmsCh,
                             "EtaxRms",     exRmsCh,
                             "WeightedRms", wRmsCh, NULL))
        SDDS_Bomb("correct_lattice: error setting rms_log row");
      rmsLogRow++;
    }
    if (verbosity > 0) {
      printf("  iteration %ld: weighted RMS residual -> %le\n", iter, rms);
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

    if (use_perturbed_matrix > 1)
      responseValid = 0;
  }
  if (prevKnobBefore) { free(prevKnobBefore); prevKnobBefore = NULL; }

  if (SDDSrmsLogInit) {
    if (!SDDS_WritePage(&SDDSrmsLog))
      SDDS_Bomb("correct_lattice: error writing rms_log page");
  }

  if (strengthLogPageOpen) {
    if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Stage", "corrected", NULL) ||
        !SDDS_WritePage(&SDDSstrengthLog))
      SDDS_Bomb("correct_lattice: error writing strength_log page");
    strengthLogPageOpen = 0;
  }

  if (verbosity > 0) {
    printf("correct_lattice: final weighted RMS residual = %le\n", rms0);
    fflush(stdout);
  }

  free(yResid);
  free_czarray_2d((void **)Rweighted, nRows, nKnob);
  free_czarray_2d((void **)Rgrouped,  nRows, nGroup);
  free(dKgroup);
  parallelTrackingBasedMatrices = parTrackSave;
  return 1;
}

/****************************************************************************/

void finish_correct_lattice(void) {
  if (SDDSstrengthLogInit) { SDDS_Terminate(&SDDSstrengthLog); SDDSstrengthLogInit = 0; }
  if (SDDSresponseInit)    { SDDS_Terminate(&SDDSresponse);    SDDSresponseInit    = 0; }
  if (SDDSrmsLogInit)      { SDDS_Terminate(&SDDSrmsLog);      SDDSrmsLogInit      = 0; }
  freeModuleState();
  initialized = 0;
}

/****************************************************************************/
/* Write the current response matrix to an SDDS file in a format that
 * load_lattice_response_matrix can read back. */
static void saveResponseMatrixToFile(char *filename, double pert) {
  SDDS_DATASET out;
  long i, j, row;
#if USE_MPI
  if (myid != 0) return;
#endif
  if (!SDDS_InitializeOutputElegant(&out, SDDS_BINARY, 0,
                                    "Lattice correction response matrix",
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
      SDDS_DefineColumn(&out, "QuadName", NULL, NULL,
                        "Quadrupole (knob) element name",
                        NULL, SDDS_STRING, 0) < 0 ||
      SDDS_DefineColumn(&out, "QuadOccurence", NULL, NULL,
                        "Quadrupole element occurrence in the beamline",
                        NULL, SDDS_LONG, 0) < 0 ||
      SDDS_DefineColumn(&out, "sQuad", "s$bquad$n", "m",
                        "Quadrupole element position along the beamline",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "dBetaxDK1", NULL, "m$a2$n",
                        "Response: d beta_x(BPM) / d parameter(quad)",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "dBetayDK1", NULL, "m$a2$n",
                        "Response: d beta_y(BPM) / d parameter(quad)",
                        NULL, SDDS_DOUBLE, 0) < 0 ||
      SDDS_DefineColumn(&out, "dEtaxDK1", NULL, "m$a2$n",
                        "Response: d eta_x(BPM) / d parameter(quad)",
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
      SDDS_DefineParameter(&out, "BindNamePattern", NULL, NULL,
                           "Default bind_name_pattern carried by this matrix; "
                           "empty means no binding",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CorrectionElements", NULL, NULL,
                           "Verbatim correction_elements string used to build "
                           "this matrix; consumed by load_lattice_response_matrix "
                           "so a subsequent correct_lattice can inherit the family "
                           "list and supply only lower_limits / upper_limits",
                           NULL, SDDS_STRING, NULL) < 0 ||
      SDDS_DefineParameter(&out, "CorrectionItems", NULL, NULL,
                           "Whitespace-joined items list aligned with "
                           "CorrectionElements; consumed by load_lattice_response_matrix",
                           NULL, SDDS_STRING, NULL) < 0 ||
      !SDDS_DefineSimpleParameter(&out, "ResponsePerturbation", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&out, "nKnobs", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nBPMs", NULL, SDDS_LONG) ||
      !SDDS_WriteLayout(&out)) {
    fprintf(stderr, "compute_lattice_response_matrix: unable to open %s for output\n", filename);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  /* Build the joined items string aligned to itemsList[]. */
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
  if (!SDDS_StartPage(&out, nBpm * nKnob) ||
      !SDDS_SetParameters(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ElementParameter",     quadItem,
                          "BindNamePattern",      bindNamePatternStr ? bindNamePatternStr : "",
                          "CorrectionElements",   corrPatternStr ? corrPatternStr : "",
                          "CorrectionItems",      itemsJoined,
                          "ResponsePerturbation", pert,
                          "nKnobs",               nKnob,
                          "nBPMs",                nBpm, NULL))
    SDDS_Bomb("compute_lattice_response_matrix: error writing parameters");
  free(itemsJoined);
  row = 0;
  for (i = 0; i < nBpm; i++)
    for (j = 0; j < nKnob; j++) {
      if (!SDDS_SetRowValues(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                             "BPMName",       bpms[i].elem->name,
                             "BPMOccurence",  bpms[i].elem->occurence,
                             "sBPM",          bpms[i].elem->end_pos,
                             "QuadName",      knobs[j].elem->name,
                             "QuadOccurence", knobs[j].elem->occurence,
                             "sQuad",         knobs[j].elem->end_pos,
                             "dBetaxDK1",     R[LCC_N_OBS*i+0][j],
                             "dBetayDK1",     R[LCC_N_OBS*i+1][j],
                             "dEtaxDK1",      R[LCC_N_OBS*i+2][j],
                             "KnobParameter", knobItem ? knobItem[j] : quadItem,
                             NULL))
        SDDS_Bomb("compute_lattice_response_matrix: error setting row");
      row++;
    }
  if (!SDDS_WritePage(&out) || !SDDS_Terminate(&out))
    SDDS_Bomb("compute_lattice_response_matrix: error closing output file");
}

/* compute_lattice_response_matrix command */
void setup_compute_lattice_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&compute_lattice_response_matrix, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &compute_lattice_response_matrix);

  if (compute_lattice_response_matrix_struct.filename == NULL)
    bombElegant("compute_lattice_response_matrix: filename is required", NULL);
  if (compute_lattice_response_matrix_struct.correction_elements == NULL ||
      *compute_lattice_response_matrix_struct.correction_elements == 0)
    bombElegant("compute_lattice_response_matrix: correction_elements is required", NULL);
  if (compute_lattice_response_matrix_struct.response_perturbation <= 0)
    bombElegant("compute_lattice_response_matrix: response_perturbation must be > 0", NULL);
  if (compute_lattice_response_matrix_struct.measurement_noise < 0)
    bombElegant("compute_lattice_response_matrix: measurement_noise must be >= 0", NULL);
  if (compute_lattice_response_matrix_struct.measurement_noise > 0 &&
      compute_lattice_response_matrix_struct.measurement_noise_cutoff <= 0)
    bombElegant("compute_lattice_response_matrix: measurement_noise_cutoff must be > 0 when noise is enabled", NULL);

  freeModuleState();
  /* Parse the new family-list inventory params (same convention as
   * &correct_lattice).  No lower/upper here -- limits are correct-time
   * concerns and live on &correct_lattice. */
  cp_str(&corrPatternStr, compute_lattice_response_matrix_struct.correction_elements);
  corrPatterns = addPatterns(&nCorrPatterns, corrPatternStr);
  if (nCorrPatterns == 0)
    bombElegant("compute_lattice_response_matrix: correction_elements parsed to zero patterns", NULL);
  itemsList = tmalloc(sizeof(*itemsList) * nCorrPatterns);
  nItemsList = nCorrPatterns;
  if (compute_lattice_response_matrix_struct.items &&
      *compute_lattice_response_matrix_struct.items) {
    char *icopy;
    cp_str(&icopy, compute_lattice_response_matrix_struct.items);
    char *tok;
    long ii = 0;
    while ((tok = get_token(icopy)) && ii < nCorrPatterns) {
      cp_str(&itemsList[ii], tok);
      ii++;
    }
    free(icopy);
    if (ii != nCorrPatterns)
      bombElegantVA("compute_lattice_response_matrix: items has %ld entries but correction_elements has %ld",
                    ii, nCorrPatterns);
  } else {
    long ii;
    for (ii = 0; ii < nCorrPatterns; ii++) cp_str(&itemsList[ii], "K1");
  }
  if (compute_lattice_response_matrix_struct.exclude &&
      *compute_lattice_response_matrix_struct.exclude) {
    char *ecopy;
    cp_str(&ecopy, compute_lattice_response_matrix_struct.exclude);
    excludePatterns = addPatterns(&nExcludePatterns, ecopy);
    free(ecopy);
  }
  {
    long nLoc = 0, nType = 0;
    LRC_freePatternList(&measLoc, &nMeasPat);
    {
      long dummy = nMeasPat;
      LRC_freePatternList(&measType, &dummy);
      nMeasPat = 0;
    }
    measLoc  = addPatterns(&nLoc,
                           compute_lattice_response_matrix_struct.measurement_elements);
    measType = addPatterns(&nType,
                           compute_lattice_response_matrix_struct.measurement_types);
    if (nLoc != nType)
      bombElegantVA("compute_lattice_response_matrix: measurement_elements "
                    "has %ld pattern(s) but measurement_types has %ld; the "
                    "two lists must be parallel (same token count)",
                    nLoc, nType);
    if (nLoc == 0)
      bombElegant("compute_lattice_response_matrix: measurement_elements / "
                  "measurement_types must be supplied", NULL);
    nMeasPat = nLoc;
  }
  /* Legacy quadItem records the first family's item for the
   * backward-compatibility ElementParameter parameter in the matrix file. */
  cp_str(&quadItem, itemsList[0]);
  /* Stash the binding intent so it gets saved into the matrix file and is
   * also visible to a subsequent correct_lattice in this run. */
  setBindingFromString(compute_lattice_response_matrix_struct.bind_name_pattern);

  verbosity = compute_lattice_response_matrix_struct.verbosity;
  /* Propagate noise into the globals consulted by readBetaEtaAtBpms. */
  beta_measurement_noise   = compute_lattice_response_matrix_struct.measurement_noise;
  eta_measurement_noise    = compute_lattice_response_matrix_struct.measurement_noise;
  measurement_noise_cutoff = compute_lattice_response_matrix_struct.measurement_noise_cutoff;

  collectAndAllocate(beamline);
  LRC_retwiss(run, beamline, NULL);
  buildResponseMatrix(run, beamline, -1,
                      compute_lattice_response_matrix_struct.response_perturbation);

  {
    char *fn = compose_filename(compute_lattice_response_matrix_struct.filename, run->rootname);
    saveResponseMatrixToFile(fn, compute_lattice_response_matrix_struct.response_perturbation);
    if (verbosity > 0) {
      printf("compute_lattice_response_matrix: wrote %ldx%ld response matrix (%ld observables per BPM) to %s\n",
             LCC_N_OBS*nBpm, nKnob, (long)LCC_N_OBS, fn);
      if (bindNamePatternStr && *bindNamePatternStr)
        printf("compute_lattice_response_matrix: saved bind_name_pattern \"%s\" into the matrix file\n",
               bindNamePatternStr);
      fflush(stdout);
    }
  }
}

/* load_lattice_response_matrix command */
void setup_load_lattice_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  SDDS_DATASET in;
  char *fn;
  char *itemStr = NULL;
  long nRows, i, j;
  int32_t nKnobsF = 0, nBPMsF = 0;
  char **bpmNames = NULL, **quadNames = NULL;
  int32_t *bpmOccs = NULL, *quadOccs = NULL;
  double *dBx = NULL, *dBy = NULL, *dEx = NULL;

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&load_lattice_response_matrix, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &load_lattice_response_matrix);

  if (load_lattice_response_matrix_struct.filename == NULL)
    bombElegant("load_lattice_response_matrix: filename is required", NULL);

  fn = compose_filename(load_lattice_response_matrix_struct.filename, run->rootname);
  if (!SDDS_InitializeInputFromSearchPath(&in, fn) || SDDS_ReadPage(&in) != 1)
    SDDS_Bomb("load_lattice_response_matrix: cannot open or read reference file");

  if (SDDS_CheckColumn(&in, "BPMName",       NULL, SDDS_STRING,             stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "BPMOccurence",  NULL, SDDS_ANY_INTEGER_TYPE,   stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "QuadName",      NULL, SDDS_STRING,             stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "QuadOccurence", NULL, SDDS_ANY_INTEGER_TYPE,   stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "dBetaxDK1",     NULL, SDDS_ANY_FLOATING_TYPE,  stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "dBetayDK1",     NULL, SDDS_ANY_FLOATING_TYPE,  stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&in, "dEtaxDK1",      NULL, SDDS_ANY_FLOATING_TYPE,  stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "ElementParameter", NULL, SDDS_STRING,             stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "nKnobs",           NULL, SDDS_ANY_INTEGER_TYPE,   stdout) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&in, "nBPMs",            NULL, SDDS_ANY_INTEGER_TYPE,   stdout) != SDDS_CHECK_OK)
    SDDS_Bomb("load_lattice_response_matrix: required columns/parameters missing");

  if (!SDDS_GetParameter(&in, "ElementParameter", &itemStr) ||
      !SDDS_GetParameterAsLong(&in, "nKnobs", &nKnobsF) ||
      !SDDS_GetParameterAsLong(&in, "nBPMs",  &nBPMsF))
    SDDS_Bomb("load_lattice_response_matrix: error reading parameters");
  /* BindNamePattern is optional (matrices written by older versions of
   * compute_lattice_response_matrix won't have it).  Empty/absent -> no
   * preloaded binding; a subsequent correct_lattice can supply its own. */
  char *bindStr = NULL;
  if (SDDS_CheckParameter(&in, "BindNamePattern", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    SDDS_GetParameter(&in, "BindNamePattern", &bindStr);
  nRows = SDDS_CountRowsOfInterest(&in);
  if (nRows != (long)nKnobsF * (long)nBPMsF)
    SDDS_Bomb("load_lattice_response_matrix: row count != nKnobs*nBPMs");
  if (!(bpmNames  = SDDS_GetColumn(&in, "BPMName")) ||
      !(bpmOccs   = SDDS_GetColumnInLong(&in, "BPMOccurence")) ||
      !(quadNames = SDDS_GetColumn(&in, "QuadName")) ||
      !(quadOccs  = SDDS_GetColumnInLong(&in, "QuadOccurence")) ||
      !(dBx       = SDDS_GetColumnInDoubles(&in, "dBetaxDK1")) ||
      !(dBy       = SDDS_GetColumnInDoubles(&in, "dBetayDK1")) ||
      !(dEx       = SDDS_GetColumnInDoubles(&in, "dEtaxDK1")))
    SDDS_Bomb("load_lattice_response_matrix: error reading columns");
  /* Per-knob item column (KnobParameter) is the multi-item form; absent
   * for matrices written by older compute_lattice_response_matrix.  When
   * absent, every knob falls back to the legacy ElementParameter parameter. */
  char **knobItemsF = NULL;
  if (SDDS_CheckColumn(&in, "KnobParameter", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    knobItemsF = SDDS_GetColumn(&in, "KnobParameter");
  /* Family-list metadata (CorrectionElements, CorrectionItems) written by
   * recent compute_lattice_response_matrix.  Optional. */
  char *corrElemsF = NULL, *corrItemsF = NULL;
  if (SDDS_CheckParameter(&in, "CorrectionElements", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    SDDS_GetParameter(&in, "CorrectionElements", &corrElemsF);
  if (SDDS_CheckParameter(&in, "CorrectionItems", NULL, SDDS_STRING, NULL) == SDDS_CHECK_OK)
    SDDS_GetParameter(&in, "CorrectionItems", &corrItemsF);
  SDDS_Terminate(&in);

  freeModuleState();
  setBindingFromString(bindStr);
  if (bindStr) free(bindStr);
  cp_str(&quadItem, itemStr);
  /* Populate the family-list module state from the loaded metadata so that a
   * subsequent correct_lattice can inherit correction_elements/items and
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
    /* Seed the &correct_lattice namelist's correction_elements and items
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
  nKnob = nKnobsF;
  nBpm  = nBPMsF;

  knobs = SDDS_Realloc(NULL, sizeof(*knobs) * nKnob);
  knobItem = tmalloc(sizeof(*knobItem) * nKnob);
  for (j = 0; j < nKnob; j++) {
    ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline, quadNames[j], quadOccs[j]);
    if (!eptr) {
      fprintf(stderr, "load_lattice_response_matrix: cannot find quad %s#%d in current beamline\n",
              quadNames[j], quadOccs[j]);
      exitElegant(1);
    }
    /* knobItemsF stores one entry per (BPM, knob) row.  The first BPM's
     * block holds row index j for knob j (i==0 in the outer i*nKnob+j
     * formula used at save time). */
    char *thisItem = knobItemsF ? knobItemsF[j] : quadItem;
    cp_str(&knobItem[j], thisItem);
    knobs[j].elem = eptr;
    knobs[j].paramIndex = confirm_parameter(thisItem, eptr->type);
    if (knobs[j].paramIndex < 0 ||
        entity_description[eptr->type].parameter[knobs[j].paramIndex].type != IS_DOUBLE) {
      fprintf(stderr, "load_lattice_response_matrix: element %s has no double parameter %s\n",
              quadNames[j], thisItem);
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
    long rowI = i * nKnob;
    ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline, bpmNames[rowI], bpmOccs[rowI]);
    if (!eptr) {
      fprintf(stderr, "load_lattice_response_matrix: cannot find BPM %s#%d in current beamline\n",
              bpmNames[rowI], bpmOccs[rowI]);
      exitElegant(1);
    }
    bpms[i].elem = eptr;
  }

  /* Allocate working buffers and populate R. */
  long nMatRows = LCC_N_OBS * nBpm;
  yMeas   = tmalloc(sizeof(*yMeas)   * nMatRows);
  yPert   = tmalloc(sizeof(*yPert)   * nMatRows);
  yTarget = tmalloc(sizeof(*yTarget) * nMatRows);
  dK      = tmalloc(sizeof(*dK)      * nKnob);
#if USE_MPI
  if (myid == 0)
    R = (double **)czarray_2d(sizeof(double), nMatRows, nKnob);
#else
  R = (double **)czarray_2d(sizeof(double), nMatRows, nKnob);
#endif
#if USE_MPI
  if (myid == 0)
#endif
  for (i = 0; i < nBpm; i++)
    for (j = 0; j < nKnob; j++) {
      long k = i * nKnob + j;
      R[LCC_N_OBS*i+0][j] = dBx[k];
      R[LCC_N_OBS*i+1][j] = dBy[k];
      R[LCC_N_OBS*i+2][j] = dEx[k];
    }

  responseValid = 1;

  if (load_lattice_response_matrix_struct.verbosity > 0) {
    printf("load_lattice_response_matrix: loaded %ldx%ld response matrix from %s (item=%s)\n",
           nMatRows, nKnob, fn, quadItem);
    if (bindNamePatternStr && *bindNamePatternStr)
      printf("load_lattice_response_matrix: bind_name_pattern \"%s\" loaded with the matrix "
             "(override with correct_lattice's bind_name_pattern)\n", bindNamePatternStr);
    fflush(stdout);
  }

  for (i = 0; i < nRows; i++) {
    if (bpmNames[i])  free(bpmNames[i]);
    if (quadNames[i]) free(quadNames[i]);
  }
  free(bpmNames); free(quadNames);
  free(bpmOccs);  free(quadOccs);
  free(dBx); free(dBy); free(dEx);
  free(itemStr);
}

/* ============================================================ */
/* Per-step corrector reassertion hooks for vary_beamline().    */
/* ============================================================ */

void correct_lattice_save_correctors(RUN *run, LINE_LIST *beamline) {
  long j;
  (void)run; (void)beamline;
  if (!initialized || knobs == NULL || nKnob == 0) return;
  corstash_clear(&clStash);
  for (j = 0; j < nKnob; j++)
    corstash_add(&clStash, knobs[j].elem, knobs[j].paramIndex);
  corstash_snapshot(&clStash);
}

long correct_lattice_reassert_correctors(RUN *run, LINE_LIST *beamline) {
  return corstash_reassert(&clStash, run, beamline);
}

void correct_lattice_invalidate_correctors(void) {
  corstash_clear(&clStash);
}
