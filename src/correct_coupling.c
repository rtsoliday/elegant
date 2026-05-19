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

/* Per-iteration knob/observable inventories rebuilt on each invocation
 * (because do_correct_coupling may be called multiple times over the
 * course of a simulation). */
typedef struct {
  ELEMENT_LIST *elem;
  long paramIndex;        /* index into entity_description[type].parameter[] */
  double *valuePtr;       /* live pointer into elem->p_elem */
  double initialValue;    /* K1 at start of this do_correct_coupling call */
} SkewKnob;
typedef struct {
  ELEMENT_LIST *elem;
} BpmObs;

static SDDS_DATASET SDDSstrengthLog, SDDSetay, SDDSresponse;
static short SDDSstrengthLogInit = 0, SDDSetayInit = 0, SDDSresponseInit = 0;

/* Module-level persistent state for the response matrix and the working buffers.
 * Promoted from static-locals in do_correct_coupling so the matrix can be
 * built in setup_correct_coupling (when use_perturbed_matrix=0) and reused
 * across multiple do_correct_coupling calls inside the steps loop. */
static SkewKnob *knobs = NULL;
static BpmObs *bpms = NULL;
static long nSkew = 0, nBpm = 0;
static double *etay = NULL, *etayPert = NULL, *dK = NULL;
static double **R = NULL;
#if USE_MPI
static double **Rt = NULL;
#endif
/* Set when R holds a valid response matrix for the current knob/bpm sets. */
static short responseValid = 0;

static void freePatternList(char ***patterns, long *n) {
  long i;
  if (*patterns) {
    for (i = 0; i < *n; i++)
      free((*patterns)[i]);
    free(*patterns);
  }
  *patterns = NULL;
  *n = 0;
}

static long collectSkews(LINE_LIST *beamline, SkewKnob **knobs) {
  ELEMENT_LIST *eptr;
  long n = 0, cap = 0, paramIndex;
  *knobs = NULL;
  eptr = beamline->elem;
  while (eptr) {
    int nameOk = (nSkewName == 0) ||
                 matchesPatternList(skewName, nSkewName, eptr->name);
    int typeOk = (nSkewType == 0) ||
                 matchesPatternList(skewType, nSkewType, entity_name[eptr->type]);
    if (nameOk && typeOk) {
      if ((paramIndex = confirm_parameter(skewItem, eptr->type)) >= 0 &&
          entity_description[eptr->type].parameter[paramIndex].type == IS_DOUBLE) {
        if (n == cap) {
          cap = cap ? 2 * cap : 16;
          *knobs = SDDS_Realloc(*knobs, sizeof(**knobs) * cap);
        }
        (*knobs)[n].elem = eptr;
        (*knobs)[n].paramIndex = paramIndex;
        (*knobs)[n].valuePtr = (double *)(eptr->p_elem +
            entity_description[eptr->type].parameter[paramIndex].offset);
        (*knobs)[n].initialValue = *(*knobs)[n].valuePtr;
        n++;
      }
    }
    eptr = eptr->succ;
  }
  return n;
}

static long collectBpms(LINE_LIST *beamline, BpmObs **bpms) {
  ELEMENT_LIST *eptr;
  long n = 0, cap = 0;
  *bpms = NULL;
  eptr = beamline->elem;
  while (eptr) {
    int nameOk = (nBpmName == 0) ||
                 matchesPatternList(bpmName, nBpmName, eptr->name);
    int typeOk = (nBpmType == 0) ||
                 matchesPatternList(bpmType, nBpmType, entity_name[eptr->type]);
    if (nameOk && typeOk) {
      if (n == cap) {
        cap = cap ? 2 * cap : 64;
        *bpms = SDDS_Realloc(*bpms, sizeof(**bpms) * cap);
      }
      (*bpms)[n].elem = eptr;
      n++;
    }
    eptr = eptr->succ;
  }
  return n;
}

/* Force re-computation of one element's matrix and the periodic twiss. */
static void retwiss(RUN *run, LINE_LIST *beamline, ELEMENT_LIST *changed) {
  unsigned long unstable;
  if (changed && changed->matrix) {
    free_matrices(changed->matrix);
    free(changed->matrix);
    changed->matrix = NULL;
  }
  /* Force the one-turn matrix and periodic solution to be redone. */
  if (beamline->matrix) {
    free_matrices(beamline->matrix);
    free(beamline->matrix);
    beamline->matrix = NULL;
  }
  beamline->flags &= ~BEAMLINE_TWISS_DONE;
  beamline->flags &= ~BEAMLINE_RADINT_DONE;
  beamline->flags |= BEAMLINE_MATRICES_NEEDED;
  update_twiss_parameters(run, beamline, &unstable);
  if (unstable)
    printWarning("correct_coupling: unstable twiss solution encountered", NULL);
}

static double ccRmsValue(double *v, long n) {
  long i;
  double s = 0;
  if (n <= 0) return 0;
  for (i = 0; i < n; i++) s += v[i] * v[i];
  return sqrt(s / n);
}

static void readEtayAtBpms(BpmObs *bpms, long nBpm, double *etay) {
  long i;
  for (i = 0; i < nBpm; i++) {
    if (!bpms[i].elem->twiss) {
      fprintf(stderr, "correct_coupling: BPM %s has no twiss data; run twiss_output first\n",
              bpms[i].elem->name);
      exitElegant(1);
    }
    etay[i] = bpms[i].elem->twiss->etay;
  }
}

/* SVD pseudo-inverse: dK = -pinv(R) . y, where R is n_bpm x n_skew.
 *   threshold: drop singular values with s_i/s_max < threshold.
 *   nKeep: if > 0, additionally cap the number of retained singular values.
 *   minSV/maxSV/nUsed report the diagnostics of the retained SV set. */
static void svdSolve(double **R_data, long n_bpm, long n_skew,
                     double *y, double *dK,
                     double threshold, long nKeep,
                     double *minSV, double *maxSV, long *nUsed) {
  MAT *R = NULL, *Rinv = NULL, *yMat = NULL, *dKMat = NULL;
  VEC *S_used = NULL;
  int32_t usedSValues = 0;
  long i, j;

  R = matrix_get(n_bpm, n_skew);
  for (i = 0; i < n_bpm; i++)
    for (j = 0; j < n_skew; j++)
      Mij(R, i, j) = R_data[i][j];

  /* matrix_invert(A, weight, largestSValue, smallestSValue, minRatio,
   *               tikhonovAlpha, tikhonovN,
   *               deleteVectors, deleteVector, deletedVector,
   *               S_Vec, sValues, S_Vec_used, usedSValues,
   *               U_matrix, Vt_matrix, conditionNum)
   *   largestSValue: keep top N (0 = no cap)
   *   smallestSValue: remove smallest N (we use minRatio instead)
   *   minRatio: drop SVs with s/s_max < minRatio
   */
  Rinv = matrix_invert(R, NULL, (int32_t)nKeep, 0, threshold,
                       0, 0, 0, NULL, NULL,
                       NULL, NULL, &S_used, &usedSValues,
                       NULL, NULL, NULL);

  *nUsed = usedSValues;
  *minSV = 0;
  *maxSV = 0;
  if (S_used && usedSValues > 0) {
    *maxSV = S_used->ve[0];
    *minSV = S_used->ve[usedSValues - 1];
    /* (matrix_invert returns the retained singular values in descending order) */
  }

  /* dK = -Rinv . y */
  yMat = matrix_get(n_bpm, 1);
  for (i = 0; i < n_bpm; i++)
    Mij(yMat, i, 0) = y[i];
  dKMat = matrix_mult(Rinv, yMat);
  for (j = 0; j < n_skew; j++)
    dK[j] = -Mij(dKMat, j, 0);

  matrix_free(R);
  matrix_free(Rinv);
  matrix_free(yMat);
  matrix_free(dKMat);
  if (S_used) vec_free(S_used);
}

/* Largest uniform scale factor s in (0,1] such that |K1_j + s.dK_j| <=
 * strength_limit for every knob. If a knob is already past the limit in
 * the same direction the SVD wants to push it, the factor collapses to 0
 * and a warning is emitted. */
static double clampStepToLimit(SkewKnob *knobs, double *dK, long n,
                               double limit) {
  long j;
  double scale = 1.0;
  if (limit <= 0) return 1.0;
  for (j = 0; j < n; j++) {
    double K0 = *knobs[j].valuePtr;
    double K1 = K0 + dK[j];
    if (fabs(K1) <= limit) continue;
    if (dK[j] == 0) continue;
    /* Allowed signed step toward the relevant boundary. */
    double allowed = (dK[j] > 0 ? limit : -limit) - K0;
    double s_j = allowed / dK[j];
    if (s_j < 0) s_j = 0;
    if (s_j < scale) scale = s_j;
  }
  if (scale < 1e-12) {
    printWarning("correct_coupling: a knob is already at the strength_limit; no correction applied this iteration",
                 NULL);
    scale = 0;
  }
  return scale;
}

/****************************************************************************/
/* Collect knob/BPM lists from the beamline and allocate the response-matrix
 * working buffers. Idempotent: subsequent calls are no-ops once knobs/bpms have
 * been collected.  Returns 1 on success; bombs if no matching knobs/BPMs. */
static long collectAndAllocate(LINE_LIST *beamline) {
  if (knobs == NULL) {
    nSkew = collectSkews(beamline, &knobs);
    nBpm = collectBpms(beamline, &bpms);
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
    else
      Rt = (double **)czarray_2d(sizeof(**R), nSkew/n_processors+1, nBpm);
#else
    R = (double **)czarray_2d(sizeof(**R), nBpm, nSkew);
#endif
  }
  return 1;
}

/* Build the response matrix R[i,j] = d eta_y(BPM_i)/d K1_j by one-sided
 * finite difference. iterTag is the value written to the SDDS parameter
 * "Iteration" in the response-matrix output file: -1 indicates a setup-time
 * build (use_perturbed_matrix=0), >=0 indicates an in-loop build. */
static void buildResponseMatrix(RUN *run, LINE_LIST *beamline, long iterTag) {
  long i, j;
#if USE_MPI
  long jlocal = 0;
#endif

  /* Baseline eta_y, assuming twiss is current. */
  readEtayAtBpms(bpms, nBpm, etay);

  for (j = 0; j < nSkew; j++) {
#if USE_MPI
    if (j%n_processors != myid)
      continue;
#endif
    double k0 = *knobs[j].valuePtr;
    if (verbosity > 2) {
      printf("  Computing response for skew %ld of %ld\n", j, nSkew);
      fflush(stdout);
    }
    *knobs[j].valuePtr = k0 + response_perturbation;
    retwiss(run, beamline, knobs[j].elem);
    readEtayAtBpms(bpms, nBpm, etayPert);
#if USE_MPI
    if (myid == 0)
      for (i = 0; i < nBpm; i++)
        R[i][j] = (etayPert[i] - etay[i]) / response_perturbation;
    else
      for (i = 0; i < nBpm; i++)
        /* stored transposed so it ships compactly to the master */
        Rt[jlocal][i] = (etayPert[i] - etay[i]) / response_perturbation;
    jlocal++;
#else
    for (i = 0; i < nBpm; i++)
      R[i][j] = (etayPert[i] - etay[i]) / response_perturbation;
#endif
    *knobs[j].valuePtr = k0;
    retwiss(run, beamline, knobs[j].elem);
  }
  /* Re-read baseline eta_y to defend against retwiss drift (numerical noise). */
  readEtayAtBpms(bpms, nBpm, etay);

#if USE_MPI
  /* Share the matrix among processors. */
  if (verbosity > 1) {
    printf("  Sharing response matrix results among processors\n");
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    int sendingRank;
    long sendingj, ib;
    MPI_Status mpiStatus;
    double *recBuffer;
    recBuffer = malloc(sizeof(*recBuffer)*nBpm);
    for (j = 0; j < nSkew; j++) {
      if ((sendingRank = j%n_processors) == 0)
        continue;
      if (MPI_Recv(&sendingj, 1, MPI_LONG, sendingRank, 100, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(recBuffer, nBpm, MPI_DOUBLE, sendingRank, 100, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS) {
        printf("Error: MPI_Recv returns error retrieving data from processor %d\n", sendingRank);
        mpiAbort = MPI_ABORT_RESPONSE_MATRIX_SHARING;
      }
      for (ib = 0; ib < nBpm; ib++)
        R[ib][sendingj] = recBuffer[ib];
    }
    free(recBuffer);
  } else {
    jlocal = 0;
    for (j = 0; j < nSkew; j++) {
      if (j%n_processors == myid) {
        MPI_Send(&j, 1, MPI_LONG, 0, 100, MPI_COMM_WORLD);
        MPI_Send(Rt[jlocal], nBpm, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        jlocal++;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  responseValid = 1;
  if (verbosity > 1) {
    if (iterTag < 0)
      printf("  response matrix built in setup\n");
    else
      printf("  response matrix built before iteration %ld\n", iterTag);
    fflush(stdout);
  }

  /* Write a page to the response file if it is open. */
  if (SDDSresponseInit) {
    long row = 0;
    if (verbosity > 2) {
      printf("  Writing response matrix\n");
      fflush(stdout);
    }
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

  if (name_pattern == NULL && type_pattern == NULL)
    bombElegant("correct_coupling: at least one of name_pattern, type_pattern must be supplied", NULL);
  if (item == NULL || strlen(item) == 0)
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

  freePatternList(&skewName, &nSkewName);
  freePatternList(&skewType, &nSkewType);
  freePatternList(&bpmName,  &nBpmName);
  freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }

  skewName = addPatterns(&nSkewName, name_pattern);
  skewType = addPatterns(&nSkewType, type_pattern);
  bpmName  = addPatterns(&nBpmName,  bpm_name_pattern);
  bpmType  = addPatterns(&nBpmType,  bpm_type_pattern);
  cp_str(&skewItem, item);

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

  /* If the user has selected use_perturbed_matrix=0 (the default), build the
   * response matrix once now, using the lattice state as-of namelist parse
   * time. With use_perturbed_matrix>0, the matrix is (re)built inside
   * do_correct_coupling instead -- following the pattern of chrom.c. */
  responseValid = 0;
  if (use_perturbed_matrix == 0) {
    collectAndAllocate(beamline);
    /* Ensure twiss is current before perturbing knobs. */
    retwiss(run, beamline, NULL);
    buildResponseMatrix(run, beamline, -1);
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

  if (!(beamline->flags & BEAMLINE_TWISS_DONE)) {
    printWarning("correct_coupling: twiss_output must be issued before correct_coupling; nothing done",
                 NULL);
    return 1;
  }

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

  retwiss(run, beamline, NULL);
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
      buildResponseMatrix(run, beamline, iter);
    
#if USE_MPI
    if (myid==0) {
      /* only the master has the full response matrix */
#endif      
      /* SVD-pseudo-invert and solve. */
      if (verbosity>3) {
	printf("  Solving for skew strengths using SVD\n");
	fflush(stdout);
      }
      svdSolve(R, nBpm, nSkew, etay, dK,
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
    double scale = clampStepToLimit(knobs, dK, nSkew, strength_limit);
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
    retwiss(run, beamline, NULL);
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
  freePatternList(&skewName, &nSkewName);
  freePatternList(&skewType, &nSkewType);
  freePatternList(&bpmName,  &nBpmName);
  freePatternList(&bpmType,  &nBpmType);
  if (skewItem) { free(skewItem); skewItem = NULL; }
  /* Free module-level knob/BPM/working-buffer state. */
  if (etay)     { free(etay);     etay = NULL; }
  if (etayPert) { free(etayPert); etayPert = NULL; }
  if (dK)       { free(dK);       dK = NULL; }
#if USE_MPI
  if (R && myid == 0)  { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
  if (Rt && myid != 0) { free_czarray_2d((void **)Rt, nSkew/n_processors+1, nBpm); Rt = NULL; }
#else
  if (R) { free_czarray_2d((void **)R, nBpm, nSkew); R = NULL; }
#endif
  if (knobs) { free(knobs); knobs = NULL; }
  if (bpms)  { free(bpms);  bpms = NULL; }
  nSkew = nBpm = 0;
  responseValid = 0;
  initialized = 0;
}
