/*************************************************************************\
 * Copyright (c) 2024 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: correctionEngine.c
 * purpose: shared linear-response correction engine used by correct_coupling
 *          and correct_lattice.  See correctionEngine.h for the API.
 */

#include "mdb.h"
#include "matlib.h"
#include "matrixOp.h"
#include "track.h"
#include "correctionEngine.h"

/****************************************************************************/

void LRC_freePatternList(char ***patterns, long *n) {
  long i;
  if (*patterns) {
    for (i = 0; i < *n; i++)
      free((*patterns)[i]);
    free(*patterns);
  }
  *patterns = NULL;
  *n = 0;
}

/****************************************************************************/

long LRC_collectKnobs(LINE_LIST *beamline,
                      char **namePatterns, long nNamePatterns,
                      char **typePatterns, long nTypePatterns,
                      char *item, LRC_Knob **knobs) {
  ELEMENT_LIST *eptr;
  long n = 0, cap = 0, paramIndex;
  *knobs = NULL;
  eptr = beamline->elem;
  while (eptr) {
    int nameOk = (nNamePatterns == 0) ||
                 matchesPatternList(namePatterns, nNamePatterns, eptr->name);
    int typeOk = (nTypePatterns == 0) ||
                 matchesPatternList(typePatterns, nTypePatterns, entity_name[eptr->type]);
    if (nameOk && typeOk) {
      if ((paramIndex = confirm_parameter(item, eptr->type)) >= 0 &&
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

/****************************************************************************/

long LRC_collectBpms(LINE_LIST *beamline,
                     char **namePatterns, long nNamePatterns,
                     char **typePatterns, long nTypePatterns,
                     LRC_Bpm **bpms) {
  ELEMENT_LIST *eptr;
  long n = 0, cap = 0;
  *bpms = NULL;
  eptr = beamline->elem;
  while (eptr) {
    int nameOk = (nNamePatterns == 0) ||
                 matchesPatternList(namePatterns, nNamePatterns, eptr->name);
    int typeOk = (nTypePatterns == 0) ||
                 matchesPatternList(typePatterns, nTypePatterns, entity_name[eptr->type]);
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

/****************************************************************************/

long LRC_collectBpmsParallel(LINE_LIST *beamline,
                             char **locPatterns, char **typePatterns,
                             long nPatterns, LRC_Bpm **bpms) {
  ELEMENT_LIST *eptr;
  long n = 0, cap = 0, i;
  *bpms = NULL;
  if (nPatterns <= 0)
    return 0;
  eptr = beamline->elem;
  while (eptr) {
    int matched = 0;
    for (i = 0; i < nPatterns; i++) {
      if (wild_match(eptr->name, locPatterns[i]) &&
          wild_match(entity_name[eptr->type], typePatterns[i])) {
        matched = 1;
        break;
      }
    }
    if (matched) {
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

/****************************************************************************/

ELEMENT_LIST *LRC_findElementByNameOccurence(LINE_LIST *beamline,
                                             char *name, long occurence) {
  ELEMENT_LIST *eptr = beamline->elem;
  while (eptr) {
    if (strcmp(eptr->name, name) == 0 && eptr->occurence == occurence)
      return eptr;
    eptr = eptr->succ;
  }
  return NULL;
}

/****************************************************************************/

void LRC_retwiss(RUN *run, LINE_LIST *beamline, ELEMENT_LIST *changed) {
  (void)LRC_retwiss_status(run, beamline, changed, 1);
}

int LRC_retwiss_status(RUN *run, LINE_LIST *beamline, ELEMENT_LIST *changed,
                       int warnOnUnstable) {
  unsigned long unstable = 0;
  if (changed && changed->matrix) {
    free_matrices(changed->matrix);
    free(changed->matrix);
    changed->matrix = NULL;
  }
  if (beamline->matrix) {
    free_matrices(beamline->matrix);
    free(beamline->matrix);
    beamline->matrix = NULL;
  }
  beamline->flags &= ~BEAMLINE_TWISS_DONE;
  beamline->flags &= ~BEAMLINE_RADINT_DONE;
  beamline->flags |= BEAMLINE_MATRICES_NEEDED;
  update_twiss_parameters(run, beamline, &unstable);
  if (unstable && warnOnUnstable)
    printWarning("correctionEngine: unstable twiss solution encountered", NULL);
  return unstable ? 1 : 0;
}

/****************************************************************************/

void LRC_buildResponseMatrix(RUN *run, LINE_LIST *beamline,
                             LRC_Knob *knobs, long nKnob,
                             long nObs,
                             LRC_ReaderFn reader, void *ctx,
                             double perturbation,
                             double **R) {
  long nRows = nObs;
  double *baseline = tmalloc(sizeof(*baseline) * nRows);
  double *pert     = tmalloc(sizeof(*pert)     * nRows);
  long i, j;
#if USE_MPI
  long jlocal = 0;
  long nLocal = nKnob / n_processors + 1;
  double **Rt = NULL;
  if (myid != 0)
    Rt = (double **)czarray_2d(sizeof(double), nLocal, nRows);
#endif

  /* baseline measurement */
  reader(nObs, baseline, ctx);

  for (j = 0; j < nKnob; j++) {
#if USE_MPI
    if (j % n_processors != myid)
      continue;
#endif
    double k0 = *knobs[j].valuePtr;
    *knobs[j].valuePtr = k0 + perturbation;
    LRC_retwiss(run, beamline, knobs[j].elem);
    reader(nObs, pert, ctx);
#if USE_MPI
    if (myid == 0) {
      for (i = 0; i < nRows; i++)
        R[i][j] = (pert[i] - baseline[i]) / perturbation;
    } else {
      /* store transposed so we can ship it compactly to master */
      for (i = 0; i < nRows; i++)
        Rt[jlocal][i] = (pert[i] - baseline[i]) / perturbation;
      jlocal++;
    }
#else
    for (i = 0; i < nRows; i++)
      R[i][j] = (pert[i] - baseline[i]) / perturbation;
#endif
    *knobs[j].valuePtr = k0;
    LRC_retwiss(run, beamline, knobs[j].elem);
  }

  /* Re-read baseline after restoration, defending against retwiss drift. */
  reader(nObs, baseline, ctx);

#if USE_MPI
  /* Gather non-master columns onto the master. */
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    int sendingRank;
    long sendingj, ib;
    MPI_Status mpiStatus;
    double *recBuffer = malloc(sizeof(*recBuffer) * nRows);
    for (j = 0; j < nKnob; j++) {
      if ((sendingRank = j % n_processors) == 0)
        continue;
      if (MPI_Recv(&sendingj, 1, MPI_LONG, sendingRank, 100, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(recBuffer, nRows, MPI_DOUBLE, sendingRank, 100, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS) {
        printf("Error: MPI_Recv failure in LRC_buildResponseMatrix\n");
        mpiAbort = MPI_ABORT_RESPONSE_MATRIX_SHARING;
      }
      for (ib = 0; ib < nRows; ib++)
        R[ib][sendingj] = recBuffer[ib];
    }
    free(recBuffer);
  } else {
    jlocal = 0;
    for (j = 0; j < nKnob; j++) {
      if (j % n_processors == myid) {
        MPI_Send(&j, 1, MPI_LONG, 0, 100, MPI_COMM_WORLD);
        MPI_Send(Rt[jlocal], nRows, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        jlocal++;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (Rt)
    free_czarray_2d((void **)Rt, nLocal, nRows);
#endif

  free(baseline);
  free(pert);
}

/****************************************************************************/

void LRC_svdSolve(double **R_data, long nRows, long nCols,
                  double *y, double *dK,
                  double svdThreshold, long nKeep,
                  double *minSV, double *maxSV, long *nUsed) {
  MAT *R = NULL, *Rinv = NULL, *yMat = NULL, *dKMat = NULL;
  VEC *S_used = NULL;
  int32_t usedSValues = 0;
  long i, j;

  R = matrix_get(nRows, nCols);
  for (i = 0; i < nRows; i++)
    for (j = 0; j < nCols; j++)
      Mij(R, i, j) = R_data[i][j];

  /* matrix_invert returns the SVD pseudo-inverse with the named SV truncation.
   * Pass tikhonovN = -1 to disable Tikhonov regularization (the convention used
   * by the rest of elegant); tikhonovN = 0 would silently turn on Tikhonov with
   * alpha = s_max, heavily damping every singular value and producing the
   * crippled inverse that earlier versions of this engine accidentally used. */
  Rinv = matrix_invert(R, NULL, (int32_t)nKeep, 0, svdThreshold,
                       0, -1, 0, NULL, NULL,
                       NULL, NULL, &S_used, &usedSValues,
                       NULL, NULL, NULL);

  *nUsed = usedSValues;
  *minSV = 0;
  *maxSV = 0;
  if (S_used && usedSValues > 0) {
    /* matrix_invert returns retained singular values in descending order */
    *maxSV = S_used->ve[0];
    *minSV = S_used->ve[usedSValues - 1];
  }

  /* dK = -Rinv . y */
  yMat = matrix_get(nRows, 1);
  for (i = 0; i < nRows; i++)
    Mij(yMat, i, 0) = y[i];
  dKMat = matrix_mult(Rinv, yMat);
  for (j = 0; j < nCols; j++)
    dK[j] = -Mij(dKMat, j, 0);

  matrix_free(R);
  matrix_free(Rinv);
  matrix_free(yMat);
  matrix_free(dKMat);
  if (S_used) vec_free(S_used);
}

/****************************************************************************/

double LRC_clampStepToLimit(LRC_Knob *knobs, double *dK, long n, double strengthLimit) {
  long j;
  double scale = 1.0;
  if (strengthLimit <= 0) return 1.0;
  for (j = 0; j < n; j++) {
    double K0 = *knobs[j].valuePtr;
    double K1 = K0 + dK[j];
    if (fabs(K1) <= strengthLimit) continue;
    if (dK[j] == 0) continue;
    /* Allowed signed step toward the relevant boundary. */
    double allowed = (dK[j] > 0 ? strengthLimit : -strengthLimit) - K0;
    double s_j = allowed / dK[j];
    if (s_j < 0) s_j = 0;
    if (s_j < scale) scale = s_j;
  }
  if (scale < 1e-12) {
    printWarning("correctionEngine: a knob is already at strength_limit; no correction applied this iteration",
                 NULL);
    scale = 0;
  }
  return scale;
}

/* Asymmetric per-knob clamp.  Constraint:
 *     lower[j] <= K[j] + s*dK[j] <= upper[j]
 * Bounds are SIGNED values on K, not magnitudes.  Either array may be
 * NULL (then that side of the bound is inactive for every knob); if
 * both arrays are NULL no clamping is performed.  Returns the largest
 * uniform s in [0,1] satisfying every active bound, or 0 with a
 * warning when at least one active bound is already binding in the
 * step direction. */
double LRC_clampStepToLimitArray(LRC_Knob *knobs, double *dK, long n,
                                 const double *lower, const double *upper) {
  long j;
  double scale = 1.0;
  if (lower == NULL && upper == NULL) return 1.0;
  for (j = 0; j < n; j++) {
    if (dK[j] == 0) continue;
    double K0 = *knobs[j].valuePtr;
    double K1 = K0 + dK[j];
    if (upper && dK[j] > 0 && K1 > upper[j]) {
      double s_j = (upper[j] - K0) / dK[j];
      if (s_j < 0) s_j = 0;
      if (s_j < scale) scale = s_j;
    } else if (lower && dK[j] < 0 && K1 < lower[j]) {
      double s_j = (lower[j] - K0) / dK[j];
      if (s_j < 0) s_j = 0;
      if (s_j < scale) scale = s_j;
    }
  }
  if (scale < 1e-12) {
    printWarning("correctionEngine: a knob is already at its strength limit; no correction applied this iteration",
                 NULL);
    scale = 0;
  }
  return scale;
}
