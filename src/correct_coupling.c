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
#include "correct_coupling.h"

/* Module-level state retained between setup/do/finish */
static long initialized = 0;
static char **skewName = NULL;      /* parsed name_pattern list */
static long nSkewName = 0;
static char **skewType = NULL;
static long nSkewType = 0;
static char **bpmName = NULL;
static long nBpmName = 0;
static char **bpmType = NULL;
static long nBpmType = 0;
static char *skewItem = NULL;

/* The knob/observable struct types live in correctionEngine.h as LRC_Knob and
 * LRC_Bpm. */

static SDDS_DATASET SDDSstrengthLog, SDDSetay, SDDSresponse;
static short SDDSstrengthLogInit = 0, SDDSetayInit = 0, SDDSresponseInit = 0;

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

/* LRC_ReaderFn-compatible trampoline used by LRC_buildResponseMatrix.
 * nObs is always 1 for coupling correction. */
static void etayReader(LRC_Bpm *bpms, long nBpm, long nObs, double *obs, void *ctx) {
  (void)nObs; (void)ctx;
  readEtayAtBpms(bpms, nBpm, obs);
}

/****************************************************************************/
/* Collect knob/BPM lists from the beamline and allocate the response-matrix
 * working buffers. Idempotent: subsequent calls are no-ops once knobs/bpms have
 * been collected.  Returns 1 on success; bombs if no matching knobs/BPMs. */
static long collectAndAllocate(LINE_LIST *beamline) {
  if (knobs == NULL) {
    nSkew = LRC_collectKnobs(beamline, skewName, nSkewName, skewType, nSkewType,
                             skewItem, &knobs);
    nBpm  = LRC_collectBpms (beamline, bpmName,  nBpmName,  bpmType,  nBpmType,  &bpms);
    if (nSkew == 0)
      bombElegant("correct_coupling: no skew quadrupole knobs matched name_pattern/type_pattern", NULL);
    if (nBpm == 0)
      bombElegant("correct_coupling: no BPMs matched bpm_name_pattern/bpm_type_pattern", NULL);
  }
  if (etay == NULL) {
    etay     = tmalloc(sizeof(*etay)     * nBpm);
    etayPert = tmalloc(sizeof(*etayPert) * nBpm);
    dK       = tmalloc(sizeof(*dK)       * nSkew);
#if USE_MPI
    if (myid == 0)
      R = (double **)czarray_2d(sizeof(**R), nBpm, nSkew);
#else
    R = (double **)czarray_2d(sizeof(**R), nBpm, nSkew);
#endif
  }
  return 1;
}

/* Build the response matrix R[i,j] = d eta_y(BPM_i)/d K1_j via the engine
 * (one-sided finite difference + MPI assembly) and, if the response_file is
 * open, write the matrix to it as one page tagged with iterTag (-1 for the
 * setup-time build, >=0 for in-loop builds). */
static void buildResponseMatrix(RUN *run, LINE_LIST *beamline, long iterTag, double perturbation) {
  long i, j;

  LRC_buildResponseMatrix(run, beamline, knobs, nSkew, bpms, nBpm, 1,
                          etayReader, NULL, perturbation, R);
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

  /* If the response matrix was already populated by compute_coupling_response_matrix
   * or load_coupling_response_matrix, the knob/BPM patterns and item live in the
   * preloaded module state, so the namelist patterns become optional. */
  if (!responseValid && name_pattern == NULL && type_pattern == NULL)
    bombElegant("correct_coupling: at least one of name_pattern, type_pattern must be supplied "
                "(unless compute_coupling_response_matrix or load_coupling_response_matrix has been issued)",
                NULL);
  if (!responseValid && (item == NULL || strlen(item) == 0))
    bombElegant("correct_coupling: item cannot be empty", NULL);
  if (n_iterations < 0)
    bombElegant("correct_coupling: n_iterations must be >= 0", NULL);
  if (correction_fraction <= 0 || correction_fraction > 1)
    bombElegant("correct_coupling: correction_fraction must be in (0, 1]", NULL);
  if (response_perturbation <= 0)
    bombElegant("correct_coupling: response_perturbation must be > 0", NULL);
  if (svd_threshold < 0)
    bombElegant("correct_coupling: svd_threshold must be >= 0", NULL);
  if (n_singular_values < 0)
    bombElegant("correct_coupling: n_singular_values must be >= 0", NULL);
  if (strength_limit < 0)
    bombElegant("correct_coupling: strength_limit must be >= 0", NULL);
  if (measurement_noise < 0)
    bombElegant("correct_coupling: measurement_noise must be >= 0", NULL);
  if (measurement_noise > 0 && measurement_noise_cutoff <= 0)
    bombElegant("correct_coupling: measurement_noise_cutoff must be > 0 when measurement_noise > 0", NULL);

  /* Only overwrite patterns/item if a preloaded matrix hasn't already set them.
   * This lets a prior compute_/load_coupling_response_matrix command provide
   * both the matrix and the implied knob/BPM/item lists. */
  if (!responseValid) {
    LRC_freePatternList(&skewName, &nSkewName);
    LRC_freePatternList(&skewType, &nSkewType);
    LRC_freePatternList(&bpmName,  &nBpmName);
    LRC_freePatternList(&bpmType,  &nBpmType);
    if (skewItem) { free(skewItem); skewItem = NULL; }

    skewName = addPatterns(&nSkewName, name_pattern);
    skewType = addPatterns(&nSkewType, type_pattern);
    bpmName  = addPatterns(&nBpmName,  bpm_name_pattern);
    bpmType  = addPatterns(&nBpmType,  bpm_type_pattern);
    cp_str(&skewItem, item);
  }

  /* Output file setup */
#if USE_MPI
  if (myid==0) {
#endif    
  if (SDDSstrengthLogInit)    { SDDS_Terminate(&SDDSstrengthLog);    SDDSstrengthLogInit = 0; }
  if (SDDSetayInit)      { SDDS_Terminate(&SDDSetay);      SDDSetayInit = 0; }
  if (SDDSresponseInit)  { SDDS_Terminate(&SDDSresponse);  SDDSresponseInit = 0; }
  
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
#if USE_MPI
  }
#endif

  /* If the user has selected use_perturbed_matrix=0 (the default) and no matrix
   * has already been preloaded by compute_/load_coupling_response_matrix,
   * build the response matrix once now, using the lattice state as-of namelist
   * parse time. With use_perturbed_matrix>0, the matrix is (re)built inside
   * do_correct_coupling instead -- following the pattern of chrom.c. */
  if (!responseValid && use_perturbed_matrix == 0) {
    collectAndAllocate(beamline);
    /* Ensure twiss is current before perturbing knobs. */
    LRC_retwiss(run, beamline, NULL);
    buildResponseMatrix(run, beamline, -1, response_perturbation);
  }

  initialized = 1;
}

/****************************************************************************/

long do_correct_coupling(RUN *run, LINE_LIST *beamline) {
  long iter, i, j;
  double rms0, rms;
  double minSV, maxSV;
  long nUsedSV;

  if (!initialized)
    return 0;

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
             knobs[j].elem->end_pos, skewItem, *knobs[j].valuePtr);
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

  LRC_retwiss(run, beamline, NULL);
  readEtayAtBpms(bpms, nBpm, etay);
  rms0 = ccRmsValue(etay, nBpm);
  if (verbosity > 0) {
    printf("  initial RMS eta_y at BPMs = %le m\n", rms0);
    fflush(stdout);
  }

  for (iter = 0; iter < n_iterations; iter++) {
    /* Adaptive backoff: if the previous iteration made RMS worse, halve the
     * correction fraction (chrom.c does the same thing for chromaticity). */
    if (adaptive_step && prevRMS >= 0 && rms0 > prevRMS) {
      currentFraction *= 0.5;
      if (verbosity > 0) {
        printf("  iteration %ld: RMS grew %le -> %le; halving correction_fraction to %le\n",
               iter, prevRMS, rms0, currentFraction);
        fflush(stdout);
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
    
#if USE_MPI
    if (myid==0) {
      /* only the master has the full response matrix */
#endif      
      /* SVD-pseudo-invert and solve. */
      if (verbosity>3) {
	printf("  Solving for skew strengths using SVD\n");
	fflush(stdout);
      }
      LRC_svdSolve(R, nBpm, nSkew, etay, dK,
	       svd_threshold, n_singular_values,
	       &minSV, &maxSV, &nUsedSV);
      if (verbosity > 1) {
	printf("  iteration %ld: SVD used %ld of %ld singular values; SV range [%le, %le]\n",
	       iter, nUsedSV, MIN(nSkew, nBpm), minSV, maxSV);
	fflush(stdout);
      }
#if USE_MPI
    }
    /* master shares dK with slaves */
    MPI_Bcast(dK, nSkew, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    
    /* Bake correction_fraction into dK; subsequent strength_limit clamping
     * tightens further if necessary. */
    if (currentFraction != 1.0)
      for (j = 0; j < nSkew; j++)
        dK[j] *= currentFraction;

    /* Apply, with strength-limit scaling. */
    double scale = LRC_clampStepToLimit(knobs, dK, nSkew, strength_limit);
    if (scale == 0) {
      if (verbosity > 0)
        printf("  iteration %ld: no correction applied (strength_limit reached)\n", iter);
      /* No further changes; previous iteration's page was actually-final. */
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
      printf("  iteration %ld: step scaled by %le to respect strength_limit=%le\n",
             iter, scale, strength_limit);
      fflush(stdout);
    }

    /* About to apply this iteration's changes -- flush any previous still-open
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
      double before = *knobs[j].valuePtr;
      double step = scale * dK[j];
      double after = before + step;
      *knobs[j].valuePtr = after;
      /* Recompute this element's matrix; persist the change into the lattice definition. */
      if (knobs[j].elem->matrix) {
        free_matrices(knobs[j].elem->matrix);
        free(knobs[j].elem->matrix);
        knobs[j].elem->matrix = NULL;
      }
      compute_matrix(knobs[j].elem, run, NULL);
      change_defined_parameter(knobs[j].elem->name, knobs[j].paramIndex,
                               knobs[j].elem->type, after, NULL, LOAD_FLAG_ABSOLUTE);
      if (SDDSstrengthLogInit) {
        if (!SDDS_SetRowValues(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, j,
                               "ElementName",            knobs[j].elem->name,
                               "ElementOccurence",       knobs[j].elem->occurence,
                               "s",                      knobs[j].elem->end_pos,
                               "ElementParameter",       skewItem,
                               "PreviousParameterValue", before,
                               "DeltaParameterValue",    step,
                               "ParameterValue",         after, NULL))
          SDDS_Bomb("correct_coupling: error setting strength_log row");
      }
    }
    /* Don't write the page yet -- defer until the next iteration starts (when
     * we'll tag it "uncorrected") or until the loop exits (when we'll tag it
     * "corrected"). */
    if (SDDSstrengthLogInit)
      strengthLogPageOpen = 1;

    /* Retwiss with all corrections in place. */
    LRC_retwiss(run, beamline, NULL);
    readEtayAtBpms(bpms, nBpm, etay);
    rms = ccRmsValue(etay, nBpm);
    if (verbosity > 0) {
      printf("  iteration %ld: RMS eta_y -> %le m\n", iter, rms);
      fflush(stdout);
    }
    rms0 = rms;

    /* For use_perturbed_matrix>1, force a fresh response build before the next
     * iteration; for 0 or 1, leave the matrix in place. */
    if (use_perturbed_matrix > 1)
      responseValid = 0;
  }

  /* Loop completed normally (n_iterations exhausted). Any still-open page is
   * therefore from the actually-final iteration -- tag it "corrected". */
  if (strengthLogPageOpen) {
    if (!SDDS_SetParameters(&SDDSstrengthLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Stage", "corrected", NULL) ||
        !SDDS_WritePage(&SDDSstrengthLog))
      SDDS_Bomb("correct_coupling: error writing strength_log page");
    strengthLogPageOpen = 0;
  }

  if (verbosity > 0) {
    printf("correct_coupling: final RMS eta_y at BPMs = %le m\n", rms0);
    fflush(stdout);
  }

  return 1;
}

/****************************************************************************/

void finish_correct_coupling(void) {
  if (SDDSstrengthLogInit)    { SDDS_Terminate(&SDDSstrengthLog);    SDDSstrengthLogInit = 0; }
  if (SDDSetayInit)      { SDDS_Terminate(&SDDSetay);      SDDSetayInit = 0; }
  if (SDDSresponseInit)  { SDDS_Terminate(&SDDSresponse);  SDDSresponseInit = 0; }
  LRC_freePatternList(&skewName, &nSkewName);
  LRC_freePatternList(&skewType, &nSkewType);
  LRC_freePatternList(&bpmName,  &nBpmName);
  LRC_freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }
  /* Free module-level knob/BPM/working-buffer state. */
  if (etay)     { free(etay);     etay = NULL; }
  if (etayPert) { free(etayPert); etayPert = NULL; }
  if (dK)       { free(dK);       dK = NULL; }
#if USE_MPI
  if (R && myid == 0)  { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms = NULL; }
  nSkew = nBpm = 0;
  responseValid = 0;
  initialized = 0;
}

/****************************************************************************/
/* Free any preloaded module state (knobs/bpms/buffers/patterns), so a
 * subsequent compute_/load_ command starts from a clean slate. */
static void freeModuleState(void) {
  if (etay)     { free(etay);     etay = NULL; }
  if (etayPert) { free(etayPert); etayPert = NULL; }
  if (dK)       { free(dK);       dK = NULL; }
#if USE_MPI
  if (R && myid == 0)  { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms = NULL; }
  LRC_freePatternList(&skewName, &nSkewName);
  LRC_freePatternList(&skewType, &nSkewType);
  LRC_freePatternList(&bpmName,  &nBpmName);
  LRC_freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }
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
      SDDS_DefineParameter(&out, "ElementParameter", NULL, NULL,
                           "Name of the element parameter that was perturbed (e.g. K1)",
                           NULL, SDDS_STRING, NULL) < 0 ||
      !SDDS_DefineSimpleParameter(&out, "ResponsePerturbation", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&out, "nKnobs", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&out, "nBPMs", NULL, SDDS_LONG) ||
      !SDDS_WriteLayout(&out)) {
    fprintf(stderr, "compute_coupling_response_matrix: unable to open %s for output\n", filename);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (!SDDS_StartPage(&out, nBpm * nSkew) ||
      !SDDS_SetParameters(&out, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ElementParameter",     skewItem,
                          "ResponsePerturbation", pert,
                          "nKnobs",               nSkew,
                          "nBPMs",                nBpm, NULL))
    SDDS_Bomb("compute_coupling_response_matrix: error writing parameters");
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
                             "Coefficient",   R[i][j], NULL))
        SDDS_Bomb("compute_coupling_response_matrix: error setting row");
      row++;
    }
  if (!SDDS_WritePage(&out) || !SDDS_Terminate(&out))
    SDDS_Bomb("compute_coupling_response_matrix: error closing output file");
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
  if (compute_coupling_response_matrix_struct.name_pattern == NULL &&
      compute_coupling_response_matrix_struct.type_pattern == NULL)
    bombElegant("compute_coupling_response_matrix: at least one of name_pattern, type_pattern is required",
                NULL);
  if (compute_coupling_response_matrix_struct.item == NULL ||
      strlen(compute_coupling_response_matrix_struct.item) == 0)
    bombElegant("compute_coupling_response_matrix: item cannot be empty", NULL);
  if (compute_coupling_response_matrix_struct.response_perturbation <= 0)
    bombElegant("compute_coupling_response_matrix: response_perturbation must be > 0", NULL);
  if (compute_coupling_response_matrix_struct.measurement_noise < 0)
    bombElegant("compute_coupling_response_matrix: measurement_noise must be >= 0", NULL);
  if (compute_coupling_response_matrix_struct.measurement_noise > 0 &&
      compute_coupling_response_matrix_struct.measurement_noise_cutoff <= 0)
    bombElegant("compute_coupling_response_matrix: measurement_noise_cutoff must be > 0 when measurement_noise > 0",
                NULL);

  freeModuleState();
  skewName = addPatterns(&nSkewName, compute_coupling_response_matrix_struct.name_pattern);
  skewType = addPatterns(&nSkewType, compute_coupling_response_matrix_struct.type_pattern);
  bpmName  = addPatterns(&nBpmName,  compute_coupling_response_matrix_struct.bpm_name_pattern);
  bpmType  = addPatterns(&nBpmType,  compute_coupling_response_matrix_struct.bpm_type_pattern);
  cp_str(&skewItem, compute_coupling_response_matrix_struct.item);

  /* Make verbosity available to the helpers (uses correct_coupling's global). */
  verbosity = compute_coupling_response_matrix_struct.verbosity;
  /* Propagate measurement noise into the globals consulted by readEtayAtBpms,
   * so a compute_coupling_response_matrix run can simulate noisy LOCO-style
   * matrix measurements. A later correct_coupling namelist will reset these
   * to its own values when it is parsed. */
  measurement_noise        = compute_coupling_response_matrix_struct.measurement_noise;
  measurement_noise_cutoff = compute_coupling_response_matrix_struct.measurement_noise_cutoff;

  collectAndAllocate(beamline);
  LRC_retwiss(run, beamline, NULL);
  buildResponseMatrix(run, beamline, -1,
                      compute_coupling_response_matrix_struct.response_perturbation);

  {
    char *fn = compose_filename(compute_coupling_response_matrix_struct.filename, run->rootname);
    saveResponseMatrixToFile(fn, compute_coupling_response_matrix_struct.response_perturbation);
    if (verbosity > 0) {
      printf("compute_coupling_response_matrix: wrote %ldx%ld response matrix to %s\n",
             nBpm, nSkew, fn);
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
  SDDS_Terminate(&in);

  /* The file was written row-major with i (BPM) outer, j (skew) inner.
   * Rows 0..nKnobs-1 give the unique skews; rows 0, nKnobs, 2*nKnobs, ...
   * give the unique BPMs. Resolve each by (name, occurence) lookup in the
   * current beamline. */
  freeModuleState();
  cp_str(&skewItem, itemStr);
  nSkew = nKnobs;
  nBpm  = nBPMs;

  knobs = SDDS_Realloc(NULL, sizeof(*knobs) * nSkew);
  for (j = 0; j < nSkew; j++) {
    ELEMENT_LIST *eptr = LRC_findElementByNameOccurence(beamline, skewNames[j], skewOccs[j]);
    if (!eptr) {
      fprintf(stderr, "load_coupling_response_matrix: cannot find skew %s#%d in current beamline\n",
              skewNames[j], skewOccs[j]);
      exitElegant(1);
    }
    knobs[j].elem = eptr;
    knobs[j].paramIndex = confirm_parameter(skewItem, eptr->type);
    if (knobs[j].paramIndex < 0 ||
        entity_description[eptr->type].parameter[knobs[j].paramIndex].type != IS_DOUBLE) {
      fprintf(stderr, "load_coupling_response_matrix: element %s has no double parameter %s\n",
              skewNames[j], skewItem);
      exitElegant(1);
    }
    knobs[j].valuePtr = (double *)(eptr->p_elem +
        entity_description[eptr->type].parameter[knobs[j].paramIndex].offset);
    knobs[j].initialValue = *knobs[j].valuePtr;
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
    printf("load_coupling_response_matrix: loaded %ldx%ld response matrix from %s (item=%s)\n",
           nBpm, nSkew, fn, skewItem);
    fflush(stdout);
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
