/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: twiss.cc
 * purpose: computation of Twiss parameters
 *
 * Michael Borland, 1989
 */
#if USE_MPI
#  include "mpi.h" /* Defines the interface to MPI allowing the use of all MPI functions. */
#  if USE_MPE
#    include "mpe.h" /* Defines the MPE library */
#  endif
#endif
#include <atomic>
#include <complex>
#include <thread>
#include <vector>
#if defined(__APPLE__)
#  include <cmath>
#  define isnan(x) std::isnan(x)
#endif
#include "mdb.h"
#include "track.h"
#include "matlib.h"
#include "chromDefs.h"
#include "fftpackC.h"
#include "twiss.h"
#if defined(HAVE_GPU) && !USE_MPI
#  include "gpu/gpu_tune.h"
#endif
#include <stddef.h>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
void computeSDrivingTerms(LINE_LIST *beamline);
void SetSDrivingTermsRow(SDDS_DATASET *SDDSout, long i, long row, double position, const char *name, const char *type, long occurence, LINE_LIST *beamline);
void computeDrivingTerms(DRIVING_TERMS *drivingTerms, ELEMENT_LIST *eptr, TWISS *twiss0, double *tune, long n_periods);
void copy_doubles(double *target, double *source, long n);
double find_acceptance(ELEMENT_LIST *elem, long plane, RUN *run, char **name, double *end_pos);
void findChamberShapes(ELEMENT_LIST *elem);
void modify_rfca_matrices(ELEMENT_LIST *eptr, long order);
void reset_rfca_matrices(ELEMENT_LIST *eptr, long order);
void incrementRadIntegrals(RADIATION_INTEGRALS *radIntegrals, double *dI,
                           ELEMENT_LIST *elem,
                           double beta0, double alpha0, double gamma0, double eta0, double etap0,
                           double betay0, double alphay0, double gammay0, double etay0, double etapy0,
                           double *coord, double pCentral);
long determineScraperAperture(long plane, unsigned long direction, double position,
                              double offset, double centroid, double *apertureRet,
                              short scraperConvention);
#ifdef __cplusplus
extern "C" {
#endif
  void AddWigglerRadiationIntegrals(double length, long periods, double radius,
                                    double eta, double etap,
                                    double beta, double alpha,
                                    double *I1, double *I2, double *I3, double *I4, double *I5);
#ifdef __cplusplus
}
#endif
void LoadStartingTwissFromFile(double *betax, double *betay, double *alphax, double *alphay,
                               double *etax, double *etay, double *etaxp, double *etayp,
                               char *filename, char *elementName, long elementOccurrence);
void computeTuneShiftWithAmplitude(double dnuxdA[N_TSWA][N_TSWA], double dnuydA[N_TSWA][N_TSWA],
                                   double *nuxExtrema, double *nuyExtrema,
                                   TWISS *twiss, double *tune, VMATRIX *M, LINE_LIST *beamline,
                                   RUN *run, double *startingCoord, long nPeriods);
void computeTuneShiftWithAmplitudeM(double dnuxdA[N_TSWA][N_TSWA], double dnuydA[N_TSWA][N_TSWA],
                                    TWISS *twiss, double *tune, VMATRIX *M, long nPeriods);
void processTwissAnalysisRequests(ELEMENT_LIST *elem);

static long twissConcatOrder = 3;
static long doTuneShiftWithAmplitude = 0;
static SDDS_DATASET SDDSTswaTunes;
static long linearChromaticTrackingInitialized = 0;
void setLinearChromaticTrackingValues(LINE_LIST *beamline);

double effectiveEllipticalAperture(double a, double b, double x, double y);

SDDS_TABLE SDDS_SDrivingTerms;

#define TWISS_ANALYSIS_QUANTITIES 8
static const char *twissAnalysisQuantityName[TWISS_ANALYSIS_QUANTITIES] = {"betax", "betay", "etax", "etay", "alphax", "alphay", "etaxp", "etayp"};
static long twissAnalysisQuantityOffset[TWISS_ANALYSIS_QUANTITIES] = {
  offsetof(TWISS, betax),
  offsetof(TWISS, betay),
  offsetof(TWISS, etax),
  offsetof(TWISS, etay),
  offsetof(TWISS, alphax),
  offsetof(TWISS, alphay),
  offsetof(TWISS, etapx),
  offsetof(TWISS, etapx),
};
#define TWISS_ANALYSIS_AVE 0
#define TWISS_ANALYSIS_MIN 1
#define TWISS_ANALYSIS_MAX 2
#define TWISS_ANALYSIS_STATS 3
static const char *twissAnalysisStatName[TWISS_ANALYSIS_STATS] = {"ave", "min", "max"};
static long twissAnalysisStatCode[TWISS_ANALYSIS_STATS] = {
  TWISS_ANALYSIS_AVE, TWISS_ANALYSIS_MIN, TWISS_ANALYSIS_MAX};

typedef struct {
  char *startName, *endName, *matchName, *tag;
  double sStart, sEnd;
  long startOccurence, endOccurence;
  short initialized;
  long count;
  long twissMem[TWISS_ANALYSIS_STATS][TWISS_ANALYSIS_QUANTITIES];
} TWISS_ANALYSIS_REQUEST;

static long twissAnalysisRequests = 0;
static TWISS_ANALYSIS_REQUEST *twissAnalysisRequest = NULL;

static short mustResetRfcaMatrices = 0;
static short periodicTwissComputed = 0;
static TWISS lastPeriodicTwiss;
static short mirror;
static long nRfElem = 0;
static ELEMENT_LIST **rfElem = NULL;
static FILE *fpRf = NULL;
static FILE *fpBucket = NULL;

VMATRIX *compute_periodic_twiss(
                                double *betax, double *alphax, double *etax, double *etapx, double *NUx,
                                double *betay, double *alphay, double *etay, double *etapy, double *NUy,
                                ELEMENT_LIST *elem, double *clorb, RUN *run, unsigned long *unstable,
                                double *eta2, double *eta3) {
  VMATRIX *M, *M1;
  double cos_phi, sin_phi, **R, beta[2], alpha[2], phi[2];
  double ***T, ****Q, eta2f[6];
  long i, j, k;
  MATRIX *dispR, *dispM, *dispMInv, *dispEta;
  static short noticeCounter = 0;

  log_entry((char *)"compute_periodic_twiss");

  if (mirror != 0 && mirror != 1 && matched != 1)
    bombElegant("problem with value of mirror parameter in call to compute_periodic_twiss---may result from attempt to correct chromaticity when matched is not 1 in &twiss_output", NULL);

  *unstable = 0;

#if DEBUG
  startMatrixComputationTiming();
#endif
  if ((i = fill_in_matrices(elem, run))) {
    if (noticeCounter < 100) {
      printf((char *)"%ld matrices recomputed for periodic Twiss parameter computation\n", i);
      report_stats(stdout, (char *)"statistics: ");
      fflush(stdout);
      if (++noticeCounter == 100) {
        fputs("(Further notices discontinued)\n", stdout);
        fflush(stdout);
      }
    }
  }
#if DEBUG
  reportMatrixComputationTiming();
#endif

  if (cavities_are_drifts_if_matched) {
    if (run->always_change_p0)
      bombElegant((char *)"can't have run_setup/always_change_p0=1 and twiss_output/cavities_are_drifts_if_matched=1", NULL);
    modify_rfca_matrices(elem, run->default_order); /* replace rf cavities with drifts */
  }

  if (clorb) {
    /* use the closed orbit to compute the on-orbit R matrix */
#ifdef DEBUG
    printf("Using closed orbit to compute on-orbit R matrix in compute_periodic_twiss\n");
    printf("clorb=%le, %le, %le, %le, %le, %le\n",
           clorb[0], clorb[1], clorb[2], clorb[3], clorb[4], clorb[5]);
#endif
    M1 = (VMATRIX *)tmalloc(sizeof(*M1));
    initialize_matrices(M1, 1);
    for (i = 0; i < 6; i++) {
      M1->C[i] = clorb[i];
      M1->R[i][i] = 1;
    }
    M = append_full_matrix(elem, run, M1, twissConcatOrder);
    free_matrices(M1);
    free(M1);
    M1 = NULL;
    /*
      printf("Computed revolution matrix on closed orbit to %ld order\n",
      twissConcatOrder);
      fflush(stdout);
      printf("matrix concatenation for periodic Twiss computation:\n");
      fflush(stdout);
      printf("closed orbit at input:\n  ");
      fflush(stdout);
      for (i=0; i<6; i++)
      printf("%14.6e ", clorb[i]);
      fflush(stdout);
      printf("\nclosed orbit at output:\n  ");
      fflush(stdout);
      for (i=0; i<6; i++)
      printf("%14.6e ", M->C[i]);
      fflush(stdout);
      printf("\nR matrix:\n");
      fflush(stdout);
      for (i=0; i<6; i++) {
      printf("  ");
      fflush(stdout);
      for (j=0; j<6; j++)
      printf("%14.6e ", M->R[i][j]);
      fflush(stdout);
      printf("\n");
      fflush(stdout);
      }
    */
  } else {
    M = full_matrix(elem, run, twissConcatOrder);
    /*
      printf("Computed revolution matrix to %ld order\n", twissConcatOrder);
      fflush(stdout);
    */
  }
#ifdef DEBUG
  report_stats(stdout, "computed revolution matrix: ");
#endif

  if (mirror) {
    /* create reverse matrix, then concatenate with forward matrix */
    VMATRIX *Mr, *Mt;
    Mr = (VMATRIX *)tmalloc(sizeof(*Mr));
    initialize_matrices(Mr, 1);
    if (!reverse_matrix(Mr, M))
      bombElegant("Problem creating reverse matrix for mirror-image twiss parameters", NULL);
    Mt = (VMATRIX *)tmalloc(sizeof(*Mt));
    initialize_matrices(Mt, 1);
    concat_matrices(Mt, Mr, M, 0);
    R = (double **)czarray_2d(sizeof(double), 6, 6);
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        R[i][j] = Mt->R[i][j];
    free_matrices(Mr);
    free_matrices(Mt);
    T = NULL;
    Q = NULL;
  } else {
    R = M->R;
    T = M->T;
    Q = M->Q;
  }

  /* allocate matrices for computing dispersion, which I do
   * in 4-d using
   * eta[i] = Inv(I - R)[i][j] R[j][5]
   * eta2[i] = Inv(I-R)[i][j] Sum[0<=k<=j<=5] T[i][j][k]*eta[i]*eta[k]
   *  with eta[4]=0 and eta[5]=1
   * The dispersion is defined by, for example,
   *  x = x(delta=0) + delta*eta + delta^2*eta2 + delta^3*eta3 ...
   */
  m_alloc(&dispM, 4, 4);
  m_alloc(&dispMInv, 4, 4);
  m_alloc(&dispR, 4, 1);
  m_alloc(&dispEta, 4, 1);
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      dispM->a[i][j] = (i == j ? 1 : 0) - R[i][j];
    }
    dispR->a[i][0] = R[i][5];
  }
  if (m_det(dispM) == 0) {
    printf((char *)"Unable to compute dispersion: unstable\n");
    fflush(stdout);
    *unstable = 3; /* both planes */
  } else {
    double *eta;
    m_invert(dispMInv, dispM);
    m_mult(dispEta, dispMInv, dispR);
    eta = (double *)tmalloc(sizeof(*eta) * 6);
    eta[0] = *etax = dispEta->a[0][0];
    eta[1] = *etapx = dispEta->a[1][0];
    eta[2] = *etay = dispEta->a[2][0];
    eta[3] = *etapy = dispEta->a[3][0];
    eta[4] = 0;
    eta[5] = 1;
    if (eta2) {
      /* now do the second-order dispersion */
      for (i = 0; i < 4; i++) {
        dispR->a[i][0] = 0;
        if (T) {
          for (j = 0; j < 6; j++)
            for (k = 0; k <= j; k++)
              dispR->a[i][0] += T[i][j][k] * eta[j] * eta[k];
        }
      }
      m_mult(dispEta, dispMInv, dispR);
      for (i = 0; i < 4; i++)
        eta2f[i] = eta2[i] = dispEta->a[i][0];
      eta2f[4] = 0;
      eta2f[5] = 0;

      if (eta3) {
        long l;
        /* third-order dispersion */
        for (i = 0; i < 4; i++) {
          dispR->a[i][0] = 0;
          if (T) {
            for (j = 0; j < 6; j++)
              for (k = 0; k <= j; k++) {
                dispR->a[i][0] += T[i][j][k] * (eta2f[j] * eta[k] + eta[j] * eta2f[k]);
                if (Q)
                  for (l = 0; l <= k; l++)
                    dispR->a[i][0] += Q[i][j][k][l] * eta[j] * eta[k] * eta[l];
              }
          }
        }
        m_mult(dispEta, dispMInv, dispR);
        for (i = 0; i < 4; i++)
          eta3[i] = dispEta->a[i][0];
      }
    }
    free(eta);
  }
  m_free(&dispM);
  m_free(&dispEta);
  m_free(&dispR);
  m_free(&dispMInv);

  for (i = 0; i < 4; i += 2) {
    if (fabs(cos_phi = (R[i][i] + R[i + 1][i + 1]) / 2) > 1) {
      if (i == 0)
        printWarning((char *)"twiss_output: beamline unstable for x plane.", (char *)"Can't find periodic lattice functions.");
      else
        printWarning((char *)"twiss_output: beamline unstable for y plane.", (char *)"Can't find periodic lattice functions.");
      *unstable |= (i == 0 ? 1 : 2);
      sin_phi = 1e-6;
    }
    beta[i / 2] = fabs(R[i][i + 1] / sin(acos(cos_phi)));
    sin_phi = R[i][i + 1] / beta[i / 2];
    phi[i / 2] = atan2(sin_phi, cos_phi) / (mirror ? 2 : 1);
    if (phi[i / 2] < 0)
      phi[i / 2] += PIx2;
    alpha[i / 2] = (R[i][i] - R[i + 1][i + 1]) / (2 * sin_phi);
  }

  *betax = beta[0];
  *betay = beta[1];
  *alphax = alpha[0];
  *alphay = alpha[1];
  *NUx = phi[0] / PIx2;
  *NUy = phi[1] / PIx2;

  /* Copy the periodic values to the twiss_output input variables.  This makes them available
   * for bunched_beams via the use_twiss_command_values qualifier
   */
  periodicTwissComputed = 1;
  memset(&lastPeriodicTwiss, 0, sizeof(lastPeriodicTwiss));
  lastPeriodicTwiss.betax = beta[0];
  lastPeriodicTwiss.alphax = alpha[0];
  lastPeriodicTwiss.etax = etax[0];
  lastPeriodicTwiss.etapx = etapx[0];
  lastPeriodicTwiss.betay = beta[1];
  lastPeriodicTwiss.alphay = alpha[1];
  lastPeriodicTwiss.etay = etay[0];
  lastPeriodicTwiss.etapy = etapy[0];

  if (mirror)
    free_czarray_2d((void **)R, 6, 6);

  // report_stats(stdout, "Computed periodic Twiss functions\nstatistics: ");

  log_exit((char *)"compute_periodic_twiss");
#ifdef DEBUG
  report_stats(stdout, "exiting compute_periodic_twiss: ");
#endif
  return (M);
}

void propagate_twiss_parameters(TWISS *twiss0, double *tune, long *waists,
                                RADIATION_INTEGRALS *radIntegrals,
                                ELEMENT_LIST *elem, RUN *run, double *traj, double *finalTraj,
                                double *couplingFactor) {
  double beta[2], alpha[2], phi[2], eta[2], etap[2], gamma[2], refAlpha[2];
  double *func, path[6], path0[6], detR[2], length, sTotal;
  double **R = NULL, C[2], S[2], Cp[2], Sp[2], D[2], Dp[2], sin_dphi, cos_dphi, dphi;
  double lastRI[6];
  double lastI3pos = 0, lastI3neg = 0;
  // long n_mat_computed;
  long i, j, plane, hasMatrix, hasPath;
  // long otherPlane;
  VMATRIX *M1, *M2;
  MATRIX *dispM, *dispOld, *dispNew;
  VMATRIX *dispM1, *dispM2;
  ELEMENT_LIST *elemOrig;
  MULT *mult;
  double KnL;
  static long asinWarning = warningCountLimit;
  std::complex<double> kappa;
  log_entry("propagate_twiss_parameters");

#ifdef DEBUG
  report_stats(stdout, "started propagate_twiss_parameters: ");
#endif

  if (!twiss0)
    bombElegant((char *)"initial Twiss parameters not given (propagate_twiss_parameters())", NULL);
  elemOrig = elem;

  if (local_dispersion) {
    /* By default, dispersion is computed ignoring acceleration, i.e., using local deltaP/P */
    m_alloc(&dispM, 4, 4);
    m_alloc(&dispOld, 4, 1);
    m_alloc(&dispNew, 4, 1);
    dispM1 = dispM2 = NULL;
  } else {
    dispM1 = (VMATRIX *)tmalloc(sizeof(*dispM1));
    dispM2 = (VMATRIX *)tmalloc(sizeof(*dispM2));
    initialize_matrices(dispM1, 1);
    initialize_matrices(dispM2, 1);
  }

  for (plane = 0; plane < 2; plane++) {
    beta[plane] = *(&twiss0->betax + plane * TWISS_Y_OFFSET);
    refAlpha[plane] = alpha[plane] = *(&twiss0->alphax + plane * TWISS_Y_OFFSET);
    phi[plane] = *(&twiss0->phix + plane * TWISS_Y_OFFSET);
    eta[plane] = *(&twiss0->etax + plane * TWISS_Y_OFFSET);
    etap[plane] = *(&twiss0->etapx + plane * TWISS_Y_OFFSET);
  }

  M1 = (VMATRIX *)tmalloc(sizeof(*M1));
  M2 = (VMATRIX *)tmalloc(sizeof(*M2));
  initialize_matrices(M1, 1);
  initialize_matrices(M2, twissConcatOrder);
  if (traj) {
#ifdef DEBUG
    printf("using trajectory in propagate_twiss_parameters()\n");
    printf("traj=%le, %le, %le, %le, %le, %le\n",
           traj[0], traj[1], traj[2], traj[3], traj[4], traj[5]);
#endif
    for (i = 0; i < 6; i++) {
      path0[i] = path[i] = traj[i];
      M1->R[i][i] = 1;
    }
  } else {
    for (i = 0; i < 6; i++) {
      path0[i] = path[i] = 0;
      M1->R[i][i] = 1;
    }
  }

  if (!local_dispersion) {
    for (i = 0; i < 6; i++)
      dispM1->R[i][i] = 1;
    for (i = 0; i < 2; i++) {
      dispM1->C[2 * i + 0] = eta[i];
      dispM1->C[2 * i + 1] = etap[i];
    }
    dispM1->C[5] = 1;
  }

  // n_mat_computed = 0;

  /*
    printf("Twiss parameter computation on path %e, %e, %e, %e, %e, %e\n",
    path[0], path[1], path[2], path[3], path[4], path[5]);
  */

  if (radIntegrals) {
    for (i = 0; i < 6; i++)
      lastRI[i] = radIntegrals->RI[i] = 0;
    radIntegrals->I3pos = 0;
    radIntegrals->I3neg = 0;
  }
  waists[0] = waists[1] = 0;

  elem = elemOrig;
  sTotal = sStart;
  while (elem) {
    for (plane = 0; plane < 2; plane++)
      gamma[plane] = (1 + sqr(alpha[plane])) / beta[plane];
    if (entity_description[elem->type].flags & HAS_MATRIX) {
      if (!elem->matrix || !(elem->matrix->R) ||
          (elem->pred && elem->pred->Pref_output != elem->Pref_input) ||
          elem->type == T_TWISSELEMENT) {
        if (elem->type == T_TWISSELEMENT) {
          if (((TWISSELEMENT *)elem->p_elem)->disable) {
            if (!elem->matrix)
              elem->matrix = drift_matrix(0.0, 1);
          } else {
            if (!((TWISSELEMENT *)elem->p_elem)->computeOnce || !((TWISSELEMENT *)elem->p_elem)->transformComputed) {
              TWISS twissInput;
              if (((TWISSELEMENT *)elem->p_elem)->from0Values) {
                twissInput = ((TWISSELEMENT *)elem->p_elem)->twiss0;
              } else {
                twissInput.betax = beta[0];
                twissInput.betay = beta[1];
                twissInput.alphax = alpha[0];
                twissInput.alphay = alpha[1];
                twissInput.etax = eta[0];
                twissInput.etapx = etap[0];
                twissInput.etay = eta[1];
                twissInput.etapy = etap[1];
              }
              if (elem->matrix) {
                free_matrices(elem->matrix);
                free(elem->matrix);
                elem->matrix = NULL;
              }
              if (((TWISSELEMENT *)elem->p_elem)->verbose) {
                printf((char *)"Computing twiss transformation matrix for %s at z=%e m from lattice twiss parameters\n", elem->name, elem->end_pos);
                printf((char *)"  * Initial twiss parameters:\n");
                printf((char *)"  betax = %le  alphax = %le  etax = %le, etaxp = %le\n",
                       twissInput.betax, twissInput.alphax, twissInput.etax, twissInput.etapx);
                printf((char *)"  betay = %le  alphay = %le  etay = %le, etayp = %le\n",
                       twissInput.betay, twissInput.alphay, twissInput.etay, twissInput.etapy);
                printf((char *)"  * Final twiss parameters:\n");
                printf((char *)"  betax = %le  alphax = %le  etax = %le, etaxp = %le\n",
                       ((TWISSELEMENT *)elem->p_elem)->twiss.betax,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.alphax,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.etax,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.etapx);
                printf((char *)"  betax = %le  alphax = %le  etax = %le, etaxp = %le\n",
                       ((TWISSELEMENT *)elem->p_elem)->twiss.betay,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.alphay,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.etay,
                       ((TWISSELEMENT *)elem->p_elem)->twiss.etapy);
              }
              elem->matrix = twissTransformMatrix((TWISSELEMENT *)elem->p_elem,
                                                  &twissInput);
              if (((TWISSELEMENT *)elem->p_elem)->fromBeam)
                /* If user wants the transform from the beam, we don't regard the transform as having
                 * been computed in final form.  Hence, we set this flag to zero.  This allows twiss and moments
                 * computations to be done with a possibly valid matrix prior to tracking
                 */
                ((TWISSELEMENT *)elem->p_elem)->transformComputed = 0;
              else
                ((TWISSELEMENT *)elem->p_elem)->transformComputed = 1;
            }
          }
        } else {
          if (elem->matrix) {
            free_matrices(elem->matrix);
            free(elem->matrix);
            elem->matrix = NULL;
          }
          elem->matrix = compute_matrix(elem, run, NULL);
        }
        // n_mat_computed++;
      }
      hasMatrix = 1;
      /* Use matrix concatenation to include effect of beam path. */
      /* In addition to copying the path into the matrix centroid array, we check
       * to see if we really need to concatenate (hasPath=1).  Saves some time
       * when optimizing
       */
      for (i = 0; i < 6; i++)
        /* path0 (entrance) is needed for rad integrals */
        path0[i] = M1->C[i] = path[i];
      for (i = hasPath = 0; i < 4; i++)
        if (path0[i]) {
          hasPath = 1;
          break;
        }
      if (hasPath || path0[5]) {
        concat_matrices(M2, elem->matrix, M1, entity_description[elem->type].flags & HAS_RF_MATRIX ? CONCAT_EXCLUDE_S0 : 0);
        R = M2->R;
        /* record new centroids for beam path */
        for (i = 0; i < 6; i++)
          path[i] = M2->C[i];
        if (!local_dispersion) {
          concat_matrices(dispM2, M2, dispM1, 0);
          /* prevent accumulation of the nominal path-length since rf elements need differential pathlength */
          dispM2->C[4] -= M2->C[4];
          /* print_matrices(stdout, elem->name, M2); */
        }
      } else {
        R = elem->matrix->R;
        /* record new centroids for beam path */
        for (i = 0; i < 6; i++)
          path[i] = elem->matrix->C[i];
        path[4] += sTotal;
        if (!local_dispersion) {
          concat_matrices(dispM2, elem->matrix, dispM1, 0);
          /* prevent accumulation of the nominal path-length since rf elements need differential pathlength */
          dispM2->C[4] -= elem->matrix->C[4];
          /* print_matrices(stdout, elem->name, elem->matrix); */
        }
      }
      /*
        if (!local_dispersion) {
        printf("\n%20s dispM1->C: ", elem->name);
        for (i=0; i<6; i++)
        printf("%e,%c", dispM1->C[i], i==5?'\n':' ');
        printf("%20s dispM2->C: ", elem->name);
        for (i=0; i<6; i++)
        printf("%e,%c", dispM2->C[i], i==5?'\n':' ');
        }
      */

      for (plane = 0; plane < 2; plane++) {
        C[plane] = R[0 + 2 * plane][0 + 2 * plane];
        S[plane] = R[0 + 2 * plane][1 + 2 * plane];
        Cp[plane] = R[1 + 2 * plane][0 + 2 * plane];
        Sp[plane] = R[1 + 2 * plane][1 + 2 * plane];
      }
    } else {
      hasMatrix = 0;
      if (elem->pred)
        elem->Pref_input = elem->pred->Pref_output;
      if (elem->Pref_output <= 0)
        elem->Pref_output = elem->Pref_input;
      for (plane = 0; plane < 2; plane++) {
        C[plane] = Sp[plane] = 1;
        Cp[plane] = D[plane] = Dp[plane] = 0;
      }
      if (entity_description[elem->type].flags & HAS_LENGTH) {
        S[0] = S[1] = *((double *)elem->p_elem);
        path[0] += path[1] * S[0];
        path[2] += path[3] * S[1];
      } else {
        S[0] = S[1] = 0;
      }
    }
    if (!elem->twiss)
      elem->twiss = (TWISS *)tmalloc(sizeof(*elem->twiss));
    elem->twiss->periodic = matched;
    if (radIntegrals) {
      incrementRadIntegrals(radIntegrals, elem->twiss->dI,
                            elem,
                            beta[0], alpha[0], gamma[0], eta[0], etap[0],
                            beta[1], alpha[1], gamma[1], eta[1], etap[1],
                            path0, run->p_central);
      if (elem->type == T_MRADINTEGRALS) {
        double mrFactor = ((MRADINTEGRALS *)elem->p_elem)->factor;
        for (i = 0; i < 6; i++) {
          radIntegrals->RI[i] = (radIntegrals->RI[i] - lastRI[i]) * mrFactor + lastRI[i];
          lastRI[i] = radIntegrals->RI[i];
        }
        /* Rescale the sign-split I3 contributions by the same factor to keep
         * I3pos + I3neg == RI[2] consistent. */
        radIntegrals->I3pos = (radIntegrals->I3pos - lastI3pos) * mrFactor + lastI3pos;
        radIntegrals->I3neg = (radIntegrals->I3neg - lastI3neg) * mrFactor + lastI3neg;
        lastI3pos = radIntegrals->I3pos;
        lastI3neg = radIntegrals->I3neg;
      }
    }
    if (elem->type == T_ROTATE && !(((ROTATE *)elem->p_elem)->excludeOptics)) {
      if (fabs(((ROTATE *)elem->p_elem)->tilt - PI / 2.0) < 1e-6 ||
          fabs(((ROTATE *)elem->p_elem)->tilt - 3 * PI / 2.0) < 1e-6 ||
          fabs(((ROTATE *)elem->p_elem)->tilt + PI / 2.0) < 1e-6 ||
          fabs(((ROTATE *)elem->p_elem)->tilt + 3 * PI / 2.0) < 1e-6) {
        elem->twiss->betax = beta[1];
        elem->twiss->betay = beta[0];
        elem->twiss->alphax = alpha[1];
        elem->twiss->alphay = alpha[0];
        elem->twiss->phix = phi[1];
        elem->twiss->phiy = phi[0];
        elem->twiss->etax = eta[1];
        elem->twiss->etay = eta[0];
        elem->twiss->etapx = etap[1];
        elem->twiss->etapy = etap[0];
        SWAP_DOUBLE(beta[0], beta[1]);
        SWAP_DOUBLE(alpha[0], alpha[1]);
        SWAP_DOUBLE(eta[0], eta[1]);
        SWAP_DOUBLE(etap[0], etap[1]);
        SWAP_DOUBLE(phi[0], phi[1]);
        elem = elem->succ;
        continue;
      }
    }
    for (plane = 0; plane < 2; plane++) {
      // otherPlane = plane?0:1;
      detR[plane] = C[plane] * Sp[plane] - Cp[plane] * S[plane];
      /* set up pointers to Twiss functions */
      func = ((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0);
      /* store centroid position */
      *(((double *)elem->twiss) + (plane ? 1 : 0) + TWISS_CENT_OFFSET) = path[plane ? 2 : 0];
      /* calculate new beta and alpha */
      if (elem->type == T_REFLECT) {
        func[0] = beta[plane];
        func[1] = -alpha[plane];
        dphi = 0;
      } else {
        func[0] = (sqr(C[plane]) * beta[plane] - 2 * C[plane] * S[plane] * alpha[plane] + sqr(S[plane]) * gamma[plane]) / detR[plane];
        func[1] = (-C[plane] * Cp[plane] * beta[plane] +
                   (Sp[plane] * C[plane] + S[plane] * Cp[plane]) * alpha[plane] -
                   S[plane] * Sp[plane] * gamma[plane]) /
          detR[plane];
        /* use R12=S to find sin(dphi) */
        if ((sin_dphi = S[plane] / sqrt(beta[plane] * func[0])) > 1) {
          printWarning((char *)"twiss_output: argument of asin() is >1 when propagating twiss parameters", NULL);
          if (asinWarning > 0) {
            asinWarning--;
            printf((char *)"Argument of asin > 1 by %f (propagate_twiss)\n", sin_dphi - 1);
            printf((char *)"element is %s at z=%em\n", elem->name, elem->end_pos);
            printf((char *)"%c-plane matrix:  C = %e,  S = %e,  ",
                   (plane == 0 ? 'x' : 'y'), C[plane], S[plane]);
            printf((char *)"C' = %e,  S' = %e\n", Cp[plane], Sp[plane]);
            printf((char *)"beta0 = %e, func[0] = %e\n", beta[plane], func[0]);
            fflush(stdout);
          }
          sin_dphi = 1;
          cos_dphi = 0;
        } else if (sin_dphi < -1) {
          printWarning((char *)"twiss_output: argument of asin() is < -1 when propagating twiss parameters", NULL);
          if (asinWarning > 0) {
            asinWarning--;
            printf((char *)"Argument of asin < -1 by %f (propagate_twiss)\n", sin_dphi + 1);
            printf((char *)"element is %s at z=%em\n", elem->name, elem->end_pos);
            printf((char *)"%c-plane matrix:  C = %e,  S = %e,  ",
                   (plane == 0 ? 'x' : 'y'), C[plane], S[plane]);
            printf((char *)"C' = %e,  S' = %e\n", Cp[plane], Sp[plane]);
            printf((char *)"beta0 = %e, func[0] = %e\n", beta[plane], func[0]);
            fflush(stdout);
          }
          sin_dphi = -1;
          cos_dphi = 0;
        } else {
          /* use R11=C to find cos(dphi) */
          cos_dphi = sqrt(beta[plane] / func[0]) * C[plane] - alpha[plane] * sin_dphi;
        }
        if (entity_description[elem->type].flags & HAS_LENGTH) {
          if (*((double *)elem->p_elem) >= 0) {
            if ((dphi = atan2(sin_dphi, cos_dphi)) < 0)
              dphi += PIx2;
          } else {
            if ((dphi = atan2(sin_dphi, cos_dphi)) > 0)
              dphi -= PIx2;
          }
        } else if (elem->type == T_MALIGN) {
          MALIGN *mal;
          mal = (MALIGN *)elem->p_elem;
          dphi = atan2(sin_dphi, cos_dphi);
          if (dphi < 0 && mal->dz >= 0)
            dphi += PIx2;
        } else {
          if ((dphi = atan2(sin_dphi, cos_dphi)) < 0)
            dphi += PIx2;
        }
      }

      phi[plane] = func[2] = phi[plane] + dphi;
      beta[plane] = fabs(func[0]);

      alpha[plane] = func[1];
      if (SIGN(alpha[plane]) != SIGN(refAlpha[plane]) && refAlpha[plane] != 0 && alpha[plane] != 0) {
        /* if sign of alpha changes, it is called a "waist". */
        waists[plane]++;
        refAlpha[plane] = alpha[plane];
      }
    }

    if (local_dispersion) {
      /* compute dispersion function and slope */
      if (hasMatrix) {
        for (i = 0; i < 4; i++) {
          for (j = 0; j < 4; j++) {
            dispM->a[i][j] = R[i][j];
          }
        }
      } else {
        for (i = 0; i < 4; i++) {
          for (j = 0; j < 4; j++) {
            dispM->a[i][j] = i == j ? 1 : 0;
          }
        }
        dispM->a[0][1] = S[0];
        dispM->a[2][3] = S[1];
      }
      dispOld->a[0][0] = eta[0];
      dispOld->a[1][0] = etap[0];
      dispOld->a[2][0] = eta[1];
      dispOld->a[3][0] = etap[1];

      m_mult(dispNew, dispM, dispOld);

      plane = 0;
      func = ((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0);
      eta[plane] = func[3] = dispNew->a[0][0] + (hasMatrix ? R[0][5] : 0);
      etap[plane] = func[4] = dispNew->a[1][0] + (hasMatrix ? R[1][5] : 0);
      plane = 1;
      func = ((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0);
      eta[plane] = func[3] = dispNew->a[2][0] + (hasMatrix ? R[2][5] : 0);
      etap[plane] = func[4] = dispNew->a[3][0] + (hasMatrix ? R[3][5] : 0);
    } else {
      plane = 0;
      func = ((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0);
      eta[plane] = func[3] = dispM2->C[0];
      etap[plane] = func[4] = dispM2->C[1];
      plane = 1;
      func = ((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0);
      eta[plane] = func[3] = dispM2->C[2];
      etap[plane] = func[4] = dispM2->C[3];
      for (i = 0; i < 6; i++)
        dispM1->C[i] = dispM2->C[i];
    }

    if (elem->type == T_MARK && ((MARK *)elem->p_elem)->fitpoint)
      store_fitpoint_twiss_parameters((MARK *)elem->p_elem, elem->name, elem->occurence,
                                      elem->twiss, radIntegrals);

    sTotal = elem->end_pos;
    elem = elem->succ;
  }

  tune[0] = phi[0] / PIx2 * n_periods;
  tune[1] = phi[1] / PIx2 * n_periods;
  if (couplingFactor) {
    /* Compute linear coupling based on 187 of Handbook of Accelerator Physics and Engineering
     * and P.J. Bryant, "A Simple Theory for Weak Betatron Coupling", CERN 94-01, Vol 1., 207-217.
     */
    std::complex<double> integrand, phaseFactor;
    double ks, phase, y, K1, K1r, tilt;
    long q;
#ifdef DEBUG_COUPLING
    FILE *fp;
    fp = fopen((char *)"kappa.sdds", (char *)"w");
    fprintf(fp, (char *)"SDDS1\n");
    fprintf(fp, (char *)"&column name=ElementName type=string &end\n");
    fprintf(fp, (char *)"&column name=Position, type=double &end\n");
    fprintf(fp, (char *)"&column name=Length , type=double &end\n");
    fprintf(fp, (char *)"&column name=K1 , type=double &end\n");
    fprintf(fp, (char *)"&column name=Tilt , type=double &end\n");
    fprintf(fp, (char *)"&column name=ks , type=double &end\n");
    fprintf(fp, (char *)"&column name=phase1 type=double &end\n");
    fprintf(fp, (char *)"&column name=phase2 type=double &end\n");
    fprintf(fp, (char *)"&column name=phase type=double &end\n");
    fprintf(fp, (char *)"&column name=exp0, type=double &end\n");
    fprintf(fp, (char *)"&column name=exp1 , type=double &end\n");
    fprintf(fp, (char *)"&column name=i0 , type=double &end\n");
    fprintf(fp, (char *)"&column name=i1 , type=double &end\n");
    fprintf(fp, (char *)"&column name=kappa0 , type=double &end\n");
    fprintf(fp, (char *)"&column name=kappa1 , type=double &end\n");
    fprintf(fp, (char *)"&data mode=ascii no_row_counts=1 &end\n");
#endif
    kappa = std::complex<double>(0, 0);
    elem = elemOrig;
    q = (long)(tune[0] - tune[1] + 0.5);
    couplingFactor[1] = (tune[0] - tune[1]) - q;
    if (n_periods == 1) {
      while (elem) {
        if ((elem->type == T_QUAD || elem->type == T_KQUAD || elem->type == T_SEXT ||
             elem->type == T_KSEXT || elem->type == T_SOLE || elem->type == T_MULT) &&
            (length = *((double *)elem->p_elem))) {
          if (elem->pred) {
            beta[0] = (elem->twiss->betax + elem->pred->twiss->betax) / 2;
            beta[1] = (elem->twiss->betay + elem->pred->twiss->betay) / 2;
            alpha[0] = (elem->twiss->alphax + elem->pred->twiss->alphax) / 2;
            alpha[1] = (elem->twiss->alphay + elem->pred->twiss->alphay) / 2;
            phi[0] = (elem->twiss->phix + elem->pred->twiss->phix) / 2;
            phi[1] = (elem->twiss->phiy + elem->pred->twiss->phiy) / 2;
            y = (elem->twiss->Cy + elem->pred->twiss->Cy) / 2;
          } else {
            beta[0] = (elem->twiss->betax + twiss0->betax) / 2;
            beta[1] = (elem->twiss->betay + twiss0->betay) / 2;
            alpha[0] = (elem->twiss->alphax + twiss0->alphax) / 2;
            alpha[1] = (elem->twiss->alphay + twiss0->alphay) / 2;
            phi[0] = (elem->twiss->phix + twiss0->phix) / 2;
            phi[1] = (elem->twiss->phiy + twiss0->phiy) / 2;
            y = (elem->twiss->Cy + twiss0->Cy) / 2;
          }
          ks = K1r = K1 = 0;
          switch (elem->type) {
          case T_QUAD:
            K1r = (K1 = ((QUAD *)(elem->p_elem))->k1) * sin(2 * (tilt = ((QUAD *)(elem->p_elem))->tilt));
            break;
          case T_KQUAD:
            K1r = (K1 = ((KQUAD *)(elem->p_elem))->k1) * sin(2 * (tilt = ((KQUAD *)(elem->p_elem))->tilt));
            break;
          case T_SEXT:
            K1r = ((SEXT *)(elem->p_elem))->k2 * (y - ((SEXT *)(elem->p_elem))->dy) +
              ((SEXT *)(elem->p_elem))->j1;
            break;
          case T_KSEXT:
            K1r = ((KSEXT *)(elem->p_elem))->k2 * (y - ((SEXT *)(elem->p_elem))->dy) +
              ((KSEXT *)(elem->p_elem))->j1;
            break;
          case T_SOLE:
            ks = -((SOLE *)(elem->p_elem))->ks;
            break;
          case T_MULT:
            mult = (MULT *)elem->p_elem;
            if (mult->bore != 0) {
              double H;
              H = run->p_central * me_mks * c_mks / e_mks;
              KnL = dfactorial(mult->order) * mult->BTipL / (H * ipow(mult->bore, mult->order));
            } else {
              KnL = mult->KnL;
            }
            switch (mult->order) {
            case 1:
              K1r = (K1 = KnL / length) * sin(2 * (tilt = mult->tilt));
              break;
            case 2:
              K1r = KnL / length * (y - mult->dy);
              break;
            default:
              break;
            }
            break;
          }
          if (K1r != 0 || ks != 0) {
            integrand = (K1r + (alpha[0] / beta[0] - alpha[1] / beta[1]) * ks / 2 - std::complex<double>(0, 1) * (1 / beta[0] + 1 / beta[1]) * ks / 2.0) * sqrt(beta[0] * beta[1]) / PIx2 * length;
            phase = phi[0] - phi[1] - (tune[0] - tune[1] - q) * PIx2 * (elem->end_pos - length / 2) / sTotal;
            phaseFactor = cexpi(phase);
            kappa = kappa + integrand * phaseFactor;
          }
#ifdef DEBUG_COUPLING
          fprintf(fp, (char *)"%s %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
                  elem->name, elem->end_pos - length / 2 + i * sTotal, length, K1r, tilt, ks,
                  (phi[0] + i * (tune[0] / n_periods)) - (phi[1] + i * (tune[1] / n_periods)),
                  (tune[0] - tune[1] - q) * PIx2 * (elem->end_pos - length / 2 + i * sTotal) / (n_periods * sTotal),
                  phase, phaseFactor.real(), phaseFactor.imag(),
                  integrand.real(), integrand.imag(),
                  kappa.real(), kappa.imag());
#endif
        }
        elem = elem->succ;
      }
      if ((couplingFactor[0] = std::abs<double>(kappa)))
        couplingFactor[2] = sqr(couplingFactor[0]) / (sqr(couplingFactor[0]) + sqr(couplingFactor[1]));
      else
        couplingFactor[2] = 0;
#ifdef DEBUG_COUPLING
      fclose(fp);
#endif
    } else {
      couplingFactor[0] = couplingFactor[2] = 0;
    }
  }

  if (radIntegrals)
    radIntegrals->computed = 1;

  /*
    printf("beta, eta, alpha: %e, %e; %e, %e; %e, %e\n",
    beta[0], beta[1],
    eta[0], eta[1],
    alpha[0], alpha[1]);
    if (radIntegrals)
    printf("Radiation integrals: %e, %e, %e, %e, %e\n",
    radIntegrals->RI[0],
    radIntegrals->RI[1],
    radIntegrals->RI[2],
    radIntegrals->RI[3],
    radIntegrals->RI[4]);
  */

  if (local_dispersion) {
    m_free(&dispNew);
    m_free(&dispM);
    m_free(&dispOld);
  } else {
    free_matrices(dispM1);
    tfree(dispM1);
    dispM1 = NULL;
    free_matrices(dispM2);
    tfree(dispM2);
    dispM2 = NULL;
    /* fprintf(stderr, "*** Using global dispersion algorithm \n"); */
  }

  free_matrices(M1);
  tfree(M1);
  M1 = NULL;
  free_matrices(M2);
  tfree(M2);
  M2 = NULL;

  processTwissAnalysisRequests(elemOrig);

  //report_stats(stdout, "Finished propagating twiss parameters.\nstatistics: ");

  if (finalTraj)
    memcpy(finalTraj, path, 6 * sizeof(*finalTraj));

  log_exit("propagate_twiss_parameters");
}

static long twiss_initialized = 0;
static long SDDS_twiss_initialized = 0;
static SDDS_TABLE SDDS_twiss;
static long twiss_count = 0;

#define IC_S 0
#define IC_BETAX 1
#define IC_ALPHAX 2
#define IC_PHIX 3
#define IC_ETAX 4
#define IC_ETAPX 5
#define IC_APX 6
#define IC_BETAY 7
#define IC_ALPHAY 8
#define IC_PHIY 9
#define IC_ETAY 10
#define IC_ETAPY 11
#define IC_APY 12
#define IC_PCENTRAL 13
#define N_DOUBLE_COLUMNS 14
#define IC_ELEMENT 14
#define IC_OCCURENCE 15
#define IC_TYPE 16
#define IC_CHAMBER_SHAPE 17
#define N_COLUMNS 18

#define IC_I1 N_COLUMNS
#define IC_I2 (N_COLUMNS + 1)
#define IC_I3 (N_COLUMNS + 2)
#define IC_I4 (N_COLUMNS + 3)
#define IC_I5 (N_COLUMNS + 4)
#define N_COLUMNS_WRI (IC_I5 + 1)
static SDDS_DEFINITION column_definition[N_COLUMNS_WRI] = {
  {(char *)"s", (char *)"&column name=s, type=double, units=m, description=Distance &end"},
  {(char *)"betax", (char *)"&column name=betax, type=double, units=m, symbol=\"$gb$r$bx$n\", description=\"Horizontal beta-function\" &end"},
  {(char *)"alphax", (char *)"&column name=alphax, type=double, symbol=\"$ga$r$bx$n\", description=\"Horizontal alpha-function\" &end"},
  {(char *)"psix", (char *)"&column name=psix, type=double, units=rad, symbol=\"$gy$r$bx$n\", description=\"Horizontal phase advance\" &end"},
  {(char *)"etax", (char *)"&column name=etax, type=double, units=m, symbol=\"$gc$r$bx$n\", description=\"Horizontal dispersion\" &end"},
  {(char *)"etaxp", (char *)"&column name=etaxp, type=double, symbol=\"$gc$r$bx$n$a'$n\", description=\"Slope of horizontal dispersion\" &end"},
  {(char *)"xAperture", (char *)"&column name=xAperture, type=double, units=m, symbol=\"a$bx,eff$n\", description=\"Effective horizontal aperture\" &end"},
  {(char *)"betay", (char *)"&column name=betay, type=double, units=m, symbol=\"$gb$r$by$n\", description=\"Vertical beta-function\" &end"},
  {(char *)"alphay", (char *)"&column name=alphay, type=double, symbol=\"$ga$r$by$n\", description=\"Vertical alpha-function\" &end"},
  {(char *)"psiy", (char *)"&column name=psiy, type=double, units=rad, symbol=\"$gy$r$by$n\", description=\"Vertical phase advance\" &end"},
  {(char *)"etay", (char *)"&column name=etay, type=double, units=m, symbol=\"$gc$r$by$n\", description=\"Vertical dispersion\" &end"},
  {(char *)"etayp", (char *)"&column name=etayp, type=double, symbol=\"$gc$r$by$n$a'$n\", description=\"Slope of vertical dispersion\" &end"},
  {(char *)"yAperture", (char *)"&column name=yAperture, type=double, units=m, symbol=\"a$by,eff$n\", description=\"Effective vertical aperture\" &end"},
  {(char *)"pCentral0", (char *)"&column name=pCentral0, type=double, units=\"m$be$nc\", symbol=\"p$bcent$n\", description=\"Initial central momentum\" &end"},
  {(char *)"ElementName", (char *)"&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {(char *)"ElementOccurence", (char *)"&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {(char *)"ElementType", (char *)"&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
  {(char *)"ChamberShape", (char *)"&column name=ChamberShape type=string &end"},
  {(char *)"dI1", (char *)"&column name=dI1, type=double, description=\"Contribution to radiation integral 1\", units=m &end"},
  {(char *)"dI2", (char *)"&column name=dI2, type=double, description=\"Contribution to radiation integral 2\", units=1/m &end"},
  {(char *)"dI3", (char *)"&column name=dI3, type=double, description=\"Contribution to radiation integral 3\", units=1/m$a2$n &end"},
  {(char *)"dI4", (char *)"&column name=dI4, type=double, description=\"Contribution to radiation integral 4\", units=1/m &end"},
  {(char *)"dI5", (char *)"&column name=dI5, type=double, description=\"Contribution to radiation integral 5\", units=1/m &end"},
};

#define IP_STEP 0
#define IP_SVN (IP_STEP+1)
#define IP_PASSLENGTH (IP_SVN+1)
#define IP_NUX (IP_PASSLENGTH + 1)
#define IP_DNUXDP (IP_NUX + 1)
#define IP_DNUXDP2 (IP_DNUXDP + 1)
#define IP_DNUXDP3 (IP_DNUXDP2 + 1)
#define IP_AX (IP_DNUXDP3 + 1)
#define IP_AXLOC (IP_AX + 1)
#define IP_AXNAME (IP_AX + 2)
#define IP_NUY (IP_AX + 3)
#define IP_DNUYDP (IP_NUY + 1)
#define IP_DNUYDP2 (IP_DNUYDP + 1)
#define IP_DNUYDP3 (IP_DNUYDP2 + 1)
#define IP_AY (IP_DNUYDP3 + 1)
#define IP_AYLOC (IP_AY + 1)
#define IP_AYNAME (IP_AY + 2)
#define IP_DPHRANGE (IP_AY + 3)
#define IP_NUXUPPER (IP_DPHRANGE + 1)
#define IP_NUXLOWER (IP_NUXUPPER + 1)
#define IP_NUYUPPER (IP_NUXLOWER + 1)
#define IP_NUYLOWER (IP_NUYUPPER + 1)
#define IP_STAGE (IP_NUYLOWER + 1)
#define IP_PCENTRAL (IP_STAGE + 1)
#define IP_DBETAXDP (IP_PCENTRAL + 1)
#define IP_DBETAYDP (IP_DBETAXDP + 1)
#define IP_DALPHAXDP (IP_DBETAYDP + 1)
#define IP_DALPHAYDP (IP_DALPHAXDP + 1)
#define IP_ETAX2 (IP_DALPHAYDP + 1)
#define IP_ETAY2 (IP_ETAX2 + 1)
#define IP_ETAX3 (IP_ETAY2 + 1)
#define IP_ETAY3 (IP_ETAX3 + 1)
#define IP_ETAPX2 (IP_ETAY3 + 1)
#define IP_ETAPY2 (IP_ETAPX2 + 1)
#define IP_ETAPX3 (IP_ETAPY2 + 1)
#define IP_ETAPY3 (IP_ETAPX3 + 1)
#define IP_BETAXMIN (IP_ETAPY3 + 1)
#define IP_BETAXAVE (IP_BETAXMIN + 1)
#define IP_BETAXMAX (IP_BETAXAVE + 1)
#define IP_BETAYMIN (IP_BETAXMAX + 1)
#define IP_BETAYAVE (IP_BETAYMIN + 1)
#define IP_BETAYMAX (IP_BETAYAVE + 1)
#define IP_ETAXMAX (IP_BETAYMAX + 1)
#define IP_ETAYMAX (IP_ETAXMAX + 1)
#define IP_WAISTSX (IP_ETAYMAX + 1)
#define IP_WAISTSY (IP_WAISTSX + 1)
#define IP_DNUXDAX (IP_WAISTSY + 1)
#define IP_DNUXDAY (IP_DNUXDAX + 1)
#define IP_DNUYDAX (IP_DNUXDAY + 1)
#define IP_DNUYDAY (IP_DNUYDAX + 1)
#define IP_DNUXDAX2 (IP_DNUYDAY + 1)
#define IP_DNUXDAY2 (IP_DNUXDAX2 + 1)
#define IP_DNUXDAXAY (IP_DNUXDAY2 + 1)
#define IP_DNUYDAX2 (IP_DNUXDAXAY + 1)
#define IP_DNUYDAY2 (IP_DNUYDAX2 + 1)
#define IP_DNUYDAXAY (IP_DNUYDAY2 + 1)
#define IP_NUXTSWAMIN (IP_DNUYDAXAY + 1)
#define IP_NUXTSWAMAX (IP_NUXTSWAMIN + 1)
#define IP_NUYTSWAMIN (IP_NUXTSWAMAX + 1)
#define IP_NUYTSWAMAX (IP_NUYTSWAMIN + 1)
#define IP_COUPLINGINTEGRAL (IP_NUYTSWAMAX + 1)
#define IP_COUPLINGOFFSET (IP_COUPLINGINTEGRAL + 1)
#define IP_EMITRATIO (IP_COUPLINGOFFSET + 1)
#define IP_ALPHAC3 (IP_EMITRATIO + 1)
#define IP_ALPHAC2 (IP_ALPHAC3 + 1)
#define IP_ALPHAC (IP_ALPHAC2 + 1)
/* IP_ALPHAC must be the last item before the radiation-integral-related
 * items!
 */
#define IP_I1 IP_ALPHAC + 1
#define IP_I2 IP_ALPHAC + 2
#define IP_I3 IP_ALPHAC + 3
#define IP_I4 IP_ALPHAC + 4
#define IP_I5 IP_ALPHAC + 5
#define IP_EX0 IP_ALPHAC + 6
#define IP_ENX0 IP_ALPHAC + 7
#define IP_TAUX IP_ALPHAC + 8
#define IP_JX IP_ALPHAC + 9
#define IP_TAUY IP_ALPHAC + 10
#define IP_JY IP_ALPHAC + 11
#define IP_SIGMADELTA IP_ALPHAC + 12
#define IP_TAUDELTA IP_ALPHAC + 13
#define IP_JDELTA IP_ALPHAC + 14
#define IP_U0 IP_ALPHAC + 15
#define IP_I3POS IP_ALPHAC + 16
#define IP_I3NEG IP_ALPHAC + 17
#define IP_TAUP IP_ALPHAC + 18
#define IP_PEQ IP_ALPHAC + 19
#define IP_PSTLIMIT IP_ALPHAC + 20
#define N_PARAMETERS IP_PSTLIMIT + 1
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {(char *)"Step", (char *)"&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {(char *)"SVNVersion", (char *)"&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {(char *)"PassLength", (char *)"&parameter name=PassLength, type=double, description=\"Length of a single pass\" &end"},
  {(char *)"nux", (char *)"&parameter name=nux, symbol=\"$gn$r$bx$n\", type=double, units=\"1/(2$gp$r)\", description=\"Horizontal tune\" &end"},
  {(char *)"dnux/dp", (char *)"&parameter name=dnux/dp, symbol=\"$gx$r$bx$n\", type=double, units=\"1/(2$gp$r)\", description=\"Horizontal chromaticity\" &end"},
  {(char *)"dnux/dp2", (char *)"&parameter name=dnux/dp2, symbol=\"$gx$r$bx2$n\", type=double, units=\"1/(2$gp$r)\", description=\"Horizontal 2nd-order chromaticity\" &end"},
  {(char *)"dnux/dp3", (char *)"&parameter name=dnux/dp3, symbol=\"$gx$r$bx3$n\", type=double, units=\"1/(2$gp$r)\", description=\"Horizontal 3rd-order chromaticity\" &end"},
  {(char *)"Ax", (char *)"&parameter name=Ax, symbol=\"A$bx$n\", type=double, units=\"m\", description=\"Horizontal acceptance\" &end"},
  {(char *)"AxLocation", (char *)"&parameter name=AxLocation, type=double, units=\"m\", description=\"Location of horizontal acceptance limit\" &end"},
  {(char *)"AxElementName", (char *)"&parameter name=AxElementName, type=string, description=\"Name of element that limits horizontal acceptance\" &end"},
  {(char *)"nuy", (char *)"&parameter name=nuy, symbol=\"$gn$r$by$n\", type=double, units=\"1/(2$gp$r)\", description=\"Vertical tune\" &end"},
  {(char *)"dnuy/dp", (char *)"&parameter name=dnuy/dp, symbol=\"$gx$r$by$n\", type=double, units=\"1/(2$gp$r)\", description=\"Vertical chromaticity\" &end"},
  {(char *)"dnuy/dp2", (char *)"&parameter name=dnuy/dp2, symbol=\"$gx$r$by2$n\", type=double, units=\"1/(2$gp$r)\", description=\"Vertical 2nd-order chromaticity\" &end"},
  {(char *)"dnuy/dp3", (char *)"&parameter name=dnuy/dp3, symbol=\"$gx$r$by3$n\", type=double, units=\"1/(2$gp$r)\", description=\"Vertical 3rd-order chromaticity\" &end"},
  {(char *)"Ay", (char *)"&parameter name=Ay, symbol=\"A$by$n\", type=double, units=\"m\", description=\"Vertical acceptance\" &end"},
  {(char *)"AyLocation", (char *)"&parameter name=AyLocation, type=double, units=\"m\", description=\"Location of vertical acceptance limit\" &end"},
  {(char *)"AyElementName", (char *)"&parameter name=AyElementName, type=string, description=\"Name of element that limits vertical acceptance\" &end"},
  {(char *)"deltaHalfRange", (char *)"&parameter name=deltaHalfRange, symbol=\"$gDd$r/2\", type=double, description=\"Half range of momentum offset for chromatic tune spread evaluation\" &end"},
  {(char *)"nuxChromUpper", (char *)"&parameter name=nuxChromUpper, symbol=\"$gx$r$bu$n\", type=double, description=\"Upper limit of x tune due to chromaticity and deltaRange\" &end"},
  {(char *)"nuxChromLower", (char *)"&parameter name=nuxChromLower, symbol=\"$gx$r$bu$n\", type=double, description=\"Lower limit of x tune due to chromaticity and deltaRange\" &end"},
  {(char *)"nuyChromUpper", (char *)"&parameter name=nuyChromUpper, symbol=\"$gy$r$bu$n\", type=double, description=\"Upper limit of y tune due to chromaticity and deltaRange\" &end"},
  {(char *)"nuyChromLower", (char *)"&parameter name=nuyChromLower, symbol=\"$gy$r$bu$n\", type=double, description=\"Lower limit of y tune due to chromaticity and deltaRange\" &end"},
  {(char *)"Stage", (char *)"&parameter name=Stage, type=string, description=\"Stage of computation\" &end"},
  {(char *)"pCentral", (char *)"&parameter name=pCentral, type=double, units=\"m$be$nc\", description=\"Central momentum\" &end"},
  {(char *)"dbetax/dp", (char *)"&parameter name=dbetax/dp, units=m, type=double, description=\"Derivative of betax with momentum offset\" &end"},
  {(char *)"dbetay/dp", (char *)"&parameter name=dbetay/dp, units=m, type=double, description=\"Derivative of betay with momentum offset\" &end"},
  {(char *)"dalphax/dp", (char *)"&parameter name=dalphax/dp, type=double, description=\"Derivative of alphax with momentum offset\" &end"},
  {(char *)"dalphay/dp", (char *)"&parameter name=dalphay/dp, type=double, description=\"Derivative of alphay with momentum offset\" &end"},
  {(char *)"etax2", (char *)"&parameter name=etax2, symbol=\"$gc$r$bx2$n\", units=m, type=double, description=\"Second-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etay2", (char *)"&parameter name=etay2, symbol=\"$gc$r$by2$n\", units=m, type=double, description=\"Second-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etax3", (char *)"&parameter name=etax3, symbol=\"$gc$r$bx3$n\", units=m, type=double, description=\"Third-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etay3", (char *)"&parameter name=etay3, symbol=\"$gc$r$by3$n\", units=m, type=double, description=\"Third-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etaxp2", (char *)"&parameter name=etaxp2, symbol=\"$gc$r$bx2$n$a'$n\", units=m, type=double, description=\"Second-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etayp2", (char *)"&parameter name=etayp2, symbol=\"$gc$r$by2$n$a'$n\", units=m, type=double, description=\"Second-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etaxp3", (char *)"&parameter name=etaxp3, symbol=\"$gc$r$bx3$n$a'$n\", units=m, type=double, description=\"Third-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"etayp3", (char *)"&parameter name=etayp3, symbol=\"$gc$r$by3$n$a'$n\", units=m, type=double, description=\"Third-order dispersion (for matched or periodic case only)\" &end"},
  {(char *)"betaxMin", (char *)"&parameter name=betaxMin, type=double, units=m, description=\"Minimum betax\" &end"},
  {(char *)"betaxAve", (char *)"&parameter name=betaxAve, type=double, units=m, description=\"Average betax\" &end"},
  {(char *)"betaxMax", (char *)"&parameter name=betaxMax, type=double, units=m, description=\"Maximum betax\" &end"},
  {(char *)"betayMin", (char *)"&parameter name=betayMin, type=double, units=m, description=\"Minimum betay\" &end"},
  {(char *)"betayAve", (char *)"&parameter name=betayAve, type=double, units=m, description=\"Average betay\" &end"},
  {(char *)"betayMax", (char *)"&parameter name=betayMax, type=double, units=m, description=\"Maximum betay\" &end"},
  {(char *)"etaxMax", (char *)"&parameter name=etaxMax, type=double, units=m, description=\"Maximum absolute value of etax\" &end"},
  {(char *)"etayMax", (char *)"&parameter name=etayMax, type=double, units=m, description=\"Maximum absolute value of etay\" &end"},
  {(char *)"waistsx", (char *)"&parameter name=waistsx, type=long, description=\"Number of changes in the sign of alphax\" &end"},
  {(char *)"waistsy", (char *)"&parameter name=waistsy, type=long, description=\"Number of changes in the sign of alphay\" &end"},
  {(char *)"dnux/dAx", (char *)"&parameter name=dnux/dAx, type=double, description=\"Horizontal tune shift with horizontal amplitude\", units=1/m &end"},
  {(char *)"dnux/dAy", (char *)"&parameter name=dnux/dAy, type=double, description=\"Horizontal tune shift with vertical amplitude\", units=1/m &end"},
  {(char *)"dnuy/dAx", (char *)"&parameter name=dnuy/dAx, type=double, description=\"Vertical tune shift with horizontal amplitude\", units=1/m &end"},
  {(char *)"dnuy/dAy", (char *)"&parameter name=dnuy/dAy, type=double, description=\"Vertical tune shift with vertical amplitude\", units=1/m &end"},
  {(char *)"dnux/dAx2", (char *)"&parameter name=dnux/dAx2, type=double, description=\"Horizontal tune shift with horizontal amplitude\", units=1/m$a2$n &end"},
  {(char *)"dnux/dAy2", (char *)"&parameter name=dnux/dAy2, type=double, description=\"Horizontal tune shift with vertical amplitude\", units=1/m$a2$n &end"},
  {(char *)"dnux/dAxAy", (char *)"&parameter name=dnux/dAxAy, type=double, description=\"Horizontal tune shift with horizontal and vertical amplitude\", units=1/m$a2$n &end"},
  {(char *)"dnuy/dAx2", (char *)"&parameter name=dnuy/dAx2, type=double, description=\"Vertical tune shift with horizontal amplitude\", units=1/m$a2$n &end"},
  {(char *)"dnuy/dAy2", (char *)"&parameter name=dnuy/dAy2, type=double, description=\"Vertical tune shift with vertical amplitude\", units=1/m$a2$n &end"},
  {(char *)"dnuy/dAxAy", (char *)"&parameter name=dnuy/dAxAy, type=double, description=\"Vertical tune shift with horizontal and vertical amplitude\", units=1/m$a2$n &end"},
  {(char *)"nuxTswaLower", (char *)"&parameter name=nuxTswaLower, type=double, description=\"Minimum horizontal tune from tune-shift-with-amplitude calculations\", &end"},
  {(char *)"nuxTswaUpper", (char *)"&parameter name=nuxTswaUpper, type=double, description=\"Maximum horizontal tune from tune-shift-with-amplitude calculations\", &end"},
  {(char *)"nuyTswaLower", (char *)"&parameter name=nuyTswaLower, type=double, description=\"Minimum vertical tune from tune-shift-with-amplitude calculations\", &end"},
  {(char *)"nuyTswaUpper", (char *)"&parameter name=nuyTswaUpper, type=double, description=\"Maximum vertical tune from tune-shift-with-amplitude calculations\", &end"},
  {(char *)"couplingIntegral", (char *)"&parameter name=couplingIntegral, type=double, description=\"Coupling integral for difference resonance\" &end"},
  {(char *)"couplingDelta", (char *)"&parameter name=couplingDelta, type=double, description=\"Distance from difference resonance\" &end"},
  {(char *)"emittanceRatio", (char *)"&parameter name=emittanceRatio, type=double, description=\"Emittance ratio from coupling integral\" &end"},
  {(char *)"alphac3", (char *)"&parameter name=alphac3, symbol=\"$ga$r$bc3$n\", type=double, description=\"3rd-order momentum compaction factor\" &end"},
  {(char *)"alphac2", (char *)"&parameter name=alphac2, symbol=\"$ga$r$bc2$n\", type=double, description=\"2nd-order momentum compaction factor\" &end"},
  {(char *)"alphac", (char *)"&parameter name=alphac, symbol=\"$ga$r$bc$n\", type=double, description=\"Momentum compaction factor\" &end"},
  {(char *)"I1", (char *)"&parameter name=I1, type=double, description=\"Radiation integral 1\", units=m &end"},
  {(char *)"I2", (char *)"&parameter name=I2, type=double, description=\"Radiation integral 2\", units=1/m &end"},
  {(char *)"I3", (char *)"&parameter name=I3, type=double, description=\"Radiation integral 3\", units=1/m$a2$n &end"},
  {(char *)"I4", (char *)"&parameter name=I4, type=double, description=\"Radiation integral 4\", units=1/m &end"},
  {(char *)"I5", (char *)"&parameter name=I5, type=double, description=\"Radiation integral 5\", units=1/m &end"},
  {(char *)"ex0", (char *)"&parameter name=ex0, type=double, description=\"Damped horizontal emittance\", units=$gp$rm &end"},
  {(char *)"enx0", (char *)"&parameter name=enx0, type=double, units=\"m$be$nc $gp$rm\", description=\"Damped normalized horizontal emittance\""},
  {(char *)"taux", (char *)"&parameter name=taux, type=double, description=\"Horizontal damping time\", units=s &end"},
  {(char *)"Jx", (char *)"&parameter name=Jx, type=double, description=\"Horizontal damping partition number\" &end"},
  {(char *)"tauy", (char *)"&parameter name=tauy, type=double, description=\"Vertical damping time\", units=s &end"},
  {(char *)"Jy", (char *)"&parameter name=Jy, type=double, description=\"Vertical damping partition number\" &end"},
  {(char *)"Sdelta0", (char *)"&parameter name=Sdelta0, type=double, description=\"RMS fractional energy spread\" &end"},
  {(char *)"taudelta", (char *)"&parameter name=taudelta, type=double, description=\"Longitudinal damping time\", units=s &end"},
  {(char *)"Jdelta", (char *)"&parameter name=Jdelta, type=double, description=\"Longitudinal damping partition number\" &end"},
  {(char *)"U0", (char *)"&parameter name=U0, type=double, units=MeV, description=\"Energy loss per turn\" &end"},
  {(char *)"I3pos", (char *)"&parameter name=I3pos, type=double, units=1/m$a2$n, description=\"Third radiation integral over positive-bending regions (Sokolov-Ternov polarizing piece)\" &end"},
  {(char *)"I3neg", (char *)"&parameter name=I3neg, type=double, units=1/m$a2$n, description=\"Third radiation integral over negative-bending regions (Sokolov-Ternov depolarizing piece)\" &end"},
  {(char *)"tauP", (char *)"&parameter name=tauP, type=double, units=s, description=\"Sokolov-Ternov self-polarization time\" &end"},
  {(char *)"Peq", (char *)"&parameter name=Peq, type=double, description=\"Equilibrium polarization (planar-ring DK formula); 8/(5*sqrt(3)) for a single-direction planar ring, suppressed by wigglers and antibends\" &end"},
  {(char *)"PstLimit", (char *)"&parameter name=PstLimit, type=double, description=\"Ideal Sokolov-Ternov polarization limit 8/(5*sqrt(3))\" &end"},
};

#define IP_DNUXDJX 0
#define IP_DNUXDJY 1
#define IP_DNUYDJY 2
#define IP_H11001 3
#define IP_H00111 4
#define IP_H10100 5
#define IP_H10010 6
#define IP_H21000 7
#define IP_H30000 8
#define IP_H10110 9
#define IP_H10020 10
#define IP_H10200 11
#define IP_H20001 12
#define IP_H00201 13
#define IP_H10002 14
#define IP_H22000 15
#define IP_H11110 16
#define IP_H00220 17
#define IP_H31000 18
#define IP_H40000 19
#define IP_H20110 20
#define IP_H11200 21
#define IP_H20020 22
#define IP_H20200 23
#define IP_H00310 24
#define IP_H00400 25
#define N_DT_PARAMETERS (3 * 23 + 3)
static SDDS_DEFINITION driving_term_parameter_definition[N_DT_PARAMETERS] = {
  {(char *)"dnux/dJx", (char *)"&parameter name=dnux/dJx, type=double, description=\"Horizontal tune shift with horizontal invariant\", units=\"1/m\" &end"},
  {(char *)"dnux/dJy", (char *)"&parameter name=dnux/dJy, type=double, description=\"Horizontal tune shift with vertical invariant\", units=\"1/m\" &end"},
  {(char *)"dnuy/dJy", (char *)"&parameter name=dnuy/dJy, type=double, description=\"Vertical tune shift with vertical invariant\", units=\"1/m\" &end"},
  {(char *)"h11001", (char *)"&parameter name=h11001, type=double, description=\"Magnitude of chromatic driving term (x chromaticity)\", &end"},
  {(char *)"h00111", (char *)"&parameter name=h00111, type=double, description=\"Magnitude of chromatic driving term (y chromaticity)\", &end"},
  {(char *)"h10100", (char *)"&parameter name=h10100, type=double, description=\"Magnitude of coupling driving term (difference resonance)\", &end"},
  {(char *)"h10010", (char *)"&parameter name=h10010, type=double, description=\"Magnitude of chromatic driving term (sum resonance)\", &end"},
  {(char *)"h21000", (char *)"&parameter name=h21000, type=double, description=\"Magnitude of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"h30000", (char *)"&parameter name=h30000, type=double, description=\"Magnitude of geometric driving term (3 nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"h10110", (char *)"&parameter name=h10110, type=double, description=\"Magnitude of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"h10020", (char *)"&parameter name=h10020, type=double, description=\"Magnitude of geometric driving term (nux - 2 nuy)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"h10200", (char *)"&parameter name=h10200, type=double, description=\"Magnitude of geometric driving term (nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"h20001", (char *)"&parameter name=h20001, type=double, description=\"Magnitude of chromatic driving term (synchro-betatron resonances)\", &end"},
  {(char *)"h00201", (char *)"&parameter name=h00201, type=double, description=\"Magnitude of chromatic driving term (momentum-dependence of beta functions)\", &end"},
  {(char *)"h10002", (char *)"&parameter name=h10002, type=double, description=\"Magnitude of chromatic driving term (second order dispersion)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"h22000", (char *)"&parameter name=h22000, type=double, description=\"Magnitude of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"h11110", (char *)"&parameter name=h11110, type=double, description=\"Magnitude of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"h00220", (char *)"&parameter name=h00220, type=double, description=\"Magnitude of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"h31000", (char *)"&parameter name=h31000, type=double, description=\"Magnitude of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"h40000", (char *)"&parameter name=h40000, type=double, description=\"Magnitude of geometric driving term (4 nux)\", units=\"1/m\" &end"},
  {(char *)"h20110", (char *)"&parameter name=h20110, type=double, description=\"Magnitude of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"h11200", (char *)"&parameter name=h11200, type=double, description=\"Magnitude of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"h20020", (char *)"&parameter name=h20020, type=double, description=\"Magnitude of geometric driving term (2 nux - 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"h20200", (char *)"&parameter name=h20200, type=double, description=\"Magnitude of geometric driving term (2 nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"h00310", (char *)"&parameter name=h00310, type=double, description=\"Magnitude of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"h00400", (char *)"&parameter name=h00400, type=double, description=\"Magnitude of geometric driving term (4 nuy)\", units=\"1/m\" &end"},

  {(char *)"Reh11001", (char *)"&parameter name=Reh11001, type=double, description=\"Real part of chromatic driving term (x chromaticity)\", &end"},
  {(char *)"Reh00111", (char *)"&parameter name=Reh00111, type=double, description=\"Real part of chromatic driving term (y chromaticity)\", &end"},
  {(char *)"Reh10100", (char *)"&parameter name=Reh10100, type=double, description=\"Real part of coupling driving term (difference resonance)\", &end"},
  {(char *)"Reh10010", (char *)"&parameter name=Reh10010, type=double, description=\"Real part of chromatic driving term (sum resonance)\", &end"},
  {(char *)"Reh21000", (char *)"&parameter name=Reh21000, type=double, description=\"Real part of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Reh30000", (char *)"&parameter name=Reh30000, type=double, description=\"Real part of geometric driving term (3 nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Reh10110", (char *)"&parameter name=Reh10110, type=double, description=\"Real part of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Reh10020", (char *)"&parameter name=Reh10020, type=double, description=\"Real part of geometric driving term (nux - 2 nuy)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Reh10200", (char *)"&parameter name=Reh10200, type=double, description=\"Real part of geometric driving term (nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Reh20001", (char *)"&parameter name=Reh20001, type=double, description=\"Real part of chromatic driving term (synchro-betatron resonances)\", &end"},
  {(char *)"Reh00201", (char *)"&parameter name=Reh00201, type=double, description=\"Real part of chromatic driving term (momentum-dependence of beta functions)\", &end"},
  {(char *)"Reh10002", (char *)"&parameter name=Reh10002, type=double, description=\"Real part of chromatic driving term (second order dispersion)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Reh22000", (char *)"&parameter name=Reh22000, type=double, description=\"Real part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Reh11110", (char *)"&parameter name=Reh11110, type=double, description=\"Real part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Reh00220", (char *)"&parameter name=Reh00220, type=double, description=\"Real part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Reh31000", (char *)"&parameter name=Reh31000, type=double, description=\"Real part of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"Reh40000", (char *)"&parameter name=Reh40000, type=double, description=\"Real part of geometric driving term (4 nux)\", units=\"1/m\" &end"},
  {(char *)"Reh20110", (char *)"&parameter name=Reh20110, type=double, description=\"Real part of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"Reh11200", (char *)"&parameter name=Reh11200, type=double, description=\"Real part of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Reh20020", (char *)"&parameter name=Reh20020, type=double, description=\"Real part of geometric driving term (2 nux - 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Reh20200", (char *)"&parameter name=Reh20200, type=double, description=\"Real part of geometric driving term (2 nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Reh00310", (char *)"&parameter name=Reh00310, type=double, description=\"Real part of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Reh00400", (char *)"&parameter name=Reh00400, type=double, description=\"Real part of geometric driving term (4 nuy)\", units=\"1/m\" &end"},

  {(char *)"Imh11001", (char *)"&parameter name=Imh11001, type=double, description=\"Imaginary part of chromatic driving term (x chromaticity)\", &end"},
  {(char *)"Imh00111", (char *)"&parameter name=Imh00111, type=double, description=\"Imaginary part of chromatic driving term (y chromaticity)\", &end"},
  {(char *)"Imh10100", (char *)"&parameter name=Imh10100, type=double, description=\"Imaginary part of coupling driving term (difference resonance)\", &end"},
  {(char *)"Imh10010", (char *)"&parameter name=Imh10010, type=double, description=\"Imaginary part of chromatic driving term (sum resonance)\", &end"},
  {(char *)"Imh21000", (char *)"&parameter name=Imh21000, type=double, description=\"Imaginary part of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Imh30000", (char *)"&parameter name=Imh30000, type=double, description=\"Imaginary part of geometric driving term (3 nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Imh10110", (char *)"&parameter name=Imh10110, type=double, description=\"Imaginary part of geometric driving term (nux)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Imh10020", (char *)"&parameter name=Imh10020, type=double, description=\"Imaginary part of geometric driving term (nux - 2 nuy)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Imh10200", (char *)"&parameter name=Imh10200, type=double, description=\"Imaginary part of geometric driving term (nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Imh20001", (char *)"&parameter name=Imh20001, type=double, description=\"Imaginary part of chromatic driving term (synchro-betatron resonances)\", &end"},
  {(char *)"Imh00201", (char *)"&parameter name=Imh00201, type=double, description=\"Imaginary part of chromatic driving term (momentum-dependence of beta functions)\", &end"},
  {(char *)"Imh10002", (char *)"&parameter name=Imh10002, type=double, description=\"Imaginary part of chromatic driving term (second order dispersion)\", units=\"1/m$a1/2$n\" &end"},
  {(char *)"Imh22000", (char *)"&parameter name=Imh22000, type=double, description=\"Imaginary part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Imh11110", (char *)"&parameter name=Imh11110, type=double, description=\"Imaginary part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Imh00220", (char *)"&parameter name=Imh00220, type=double, description=\"Imaginary part of geometric driving term (amplitude-dependent tune)\", units=\"1/m\" &end"},
  {(char *)"Imh31000", (char *)"&parameter name=Imh31000, type=double, description=\"Imaginary part of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"Imh40000", (char *)"&parameter name=Imh40000, type=double, description=\"Imaginary part of geometric driving term (4 nux)\", units=\"1/m\" &end"},
  {(char *)"Imh20110", (char *)"&parameter name=Imh20110, type=double, description=\"Imaginary part of geometric driving term (2 nux)\", units=\"1/m\" &end"},
  {(char *)"Imh11200", (char *)"&parameter name=Imh11200, type=double, description=\"Imaginary part of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Imh20020", (char *)"&parameter name=Imh20020, type=double, description=\"Imaginary part of geometric driving term (2 nux - 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Imh20200", (char *)"&parameter name=Imh20200, type=double, description=\"Imaginary part of geometric driving term (2 nux + 2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Imh00310", (char *)"&parameter name=Imh00310, type=double, description=\"Imaginary part of geometric driving term (2 nuy)\", units=\"1/m\" &end"},
  {(char *)"Imh00400", (char *)"&parameter name=Imh00400, type=double, description=\"Imaginary part of geometric driving term (4 nuy)\", units=\"1/m\" &end"},
};

void dump_twiss_parameters(
                           LINE_LIST *beamline,
                           long n_elem,
                           double *tune,
                           RADIATION_INTEGRALS *radIntegrals,
                           double *chromaticity,
                           double *dbeta,
                           double *dalpha,
                           double *acceptance,
                           char **acceptanceElementName,
                           double *alphac,
                           long final_values_only_inner_scope,
                           long tune_corrected,
                           RUN *run) {
  double *data;
  long i, j, row_count;
  char *stage;
  TWISS twiss_ave, twiss_min, twiss_max;

  TWISS *twiss0;
  ELEMENT_LIST *elem;

  log_entry((char *)"dump_twiss_parameters");

  twiss0 = beamline->twiss0;
  elem = beamline->elem_twiss;

  if (!twiss0)
    bombElegant((char *)"Twiss data not computed prior to dump_twiss_parameters() call (1)", NULL);

  data = (double *)tmalloc(sizeof(*data) * N_DOUBLE_COLUMNS);

  if (tune_corrected == 1)
    stage = (char *)"tunes corrected";
  else
    stage = (char *)"tunes uncorrected";

  if (!SDDS_StartTable(&SDDS_twiss, final_values_only_inner_scope ? 1 : n_elem + 1)) {
    SDDS_SetError((char *)"Problem starting SDDS table (dump_twiss_parameters)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  compute_twiss_statistics(beamline, &twiss_ave, &twiss_min, &twiss_max);
  if (!SDDS_SetParameters(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IP_STEP, twiss_count, IP_STAGE, stage, IP_PASSLENGTH, beamline->revolution_length,
                          IP_NUX, tune[0], IP_DNUXDP, chromaticity[0], IP_AX, acceptance[0], IP_AXLOC, acceptance[2],
                          IP_AXNAME, acceptanceElementName[0],
                          IP_NUY, tune[1], IP_DNUYDP, chromaticity[1], IP_AY, acceptance[1], IP_AYLOC, acceptance[3],
                          IP_AYNAME, acceptanceElementName[1],
                          IP_DNUXDP2, beamline->chrom2[0],
                          IP_DNUYDP2, beamline->chrom2[1],
                          IP_DNUXDP3, beamline->chrom3[0],
                          IP_DNUYDP3, beamline->chrom3[1],
                          IP_DPHRANGE, beamline->chromDeltaHalfRange,
                          IP_NUXUPPER, beamline->tuneChromUpper[0],
                          IP_NUYUPPER, beamline->tuneChromUpper[1],
                          IP_NUXLOWER, beamline->tuneChromLower[0],
                          IP_NUYLOWER, beamline->tuneChromLower[1],
                          IP_COUPLINGINTEGRAL, beamline->couplingFactor[0],
                          IP_COUPLINGOFFSET, beamline->couplingFactor[1],
                          IP_EMITRATIO, beamline->couplingFactor[2],
                          IP_ALPHAC, alphac[0], IP_ALPHAC2, alphac[1], IP_ALPHAC3, alphac[2],
                          IP_DBETAXDP, dbeta[0], IP_DBETAYDP, dbeta[1],
                          IP_DALPHAXDP, dalpha[0], IP_DALPHAYDP, dalpha[1],
                          IP_BETAXMIN, twiss_min.betax, IP_BETAXAVE, twiss_ave.betax, IP_BETAXMAX, twiss_max.betax,
                          IP_BETAYMIN, twiss_min.betay, IP_BETAYAVE, twiss_ave.betay, IP_BETAYMAX, twiss_max.betay,
                          IP_ETAXMAX, MAX(fabs(twiss_min.etax), fabs(twiss_max.etax)),
                          IP_ETAYMAX, MAX(fabs(twiss_min.etay), fabs(twiss_max.etay)),
                          IP_WAISTSX, beamline->waists[0],
                          IP_WAISTSY, beamline->waists[1],
                          IP_DNUXDAX, beamline->dnux_dA[1][0],
                          IP_DNUXDAY, beamline->dnux_dA[0][1],
                          IP_DNUYDAX, beamline->dnuy_dA[1][0],
                          IP_DNUYDAY, beamline->dnuy_dA[0][1],
                          IP_DNUXDAX2, beamline->dnux_dA[2][0],
                          IP_DNUXDAY2, beamline->dnux_dA[0][2],
                          IP_DNUXDAXAY, beamline->dnux_dA[1][1],
                          IP_DNUYDAX2, beamline->dnuy_dA[2][0],
                          IP_DNUYDAY2, beamline->dnuy_dA[0][2],
                          IP_DNUYDAXAY, beamline->dnuy_dA[1][1],
                          IP_NUXTSWAMIN, beamline->nuxTswaExtrema[0],
                          IP_NUXTSWAMAX, beamline->nuxTswaExtrema[1],
                          IP_NUYTSWAMIN, beamline->nuyTswaExtrema[0],
                          IP_NUYTSWAMAX, beamline->nuyTswaExtrema[1],
                          IP_ETAX2, beamline->eta2[0],
                          IP_ETAPX2, beamline->eta2[1],
                          IP_ETAY2, beamline->eta2[2],
                          IP_ETAPY2, beamline->eta2[3],
                          IP_ETAX3, beamline->eta3[0],
                          IP_ETAPX3, beamline->eta3[1],
                          IP_ETAY3, beamline->eta3[2],
                          IP_ETAPY3, beamline->eta3[3],
                          IP_PCENTRAL, run->p_central, -1)) {
    SDDS_SetError((char *)"Problem setting SDDS parameters (dump_twiss_parameters 1)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (compute_driving_terms) {
    long offset;
    if (radiation_integrals)
      offset = IP_U0 + 1;
    else
      offset = IP_ALPHAC + 1;
    if (!SDDS_SetParameters(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                            IP_DNUXDJX + offset, beamline->drivingTerms.dnux_dJx,
                            IP_DNUXDJY + offset, beamline->drivingTerms.dnux_dJy,
                            IP_DNUYDJY + offset, beamline->drivingTerms.dnuy_dJy,
                            /* magnitudes of driving terms */
                            IP_H11001 + offset, beamline->drivingTerms.h11001[0],
                            IP_H00111 + offset, beamline->drivingTerms.h00111[0],
                            IP_H10010 + offset, beamline->drivingTerms.h10010[0],
                            IP_H10100 + offset, beamline->drivingTerms.h10100[0],
                            IP_H20001 + offset, beamline->drivingTerms.h20001[0],
                            IP_H00201 + offset, beamline->drivingTerms.h00201[0],
                            IP_H10002 + offset, beamline->drivingTerms.h10002[0],
                            IP_H21000 + offset, beamline->drivingTerms.h21000[0],
                            IP_H30000 + offset, beamline->drivingTerms.h30000[0],
                            IP_H10110 + offset, beamline->drivingTerms.h10110[0],
                            IP_H10020 + offset, beamline->drivingTerms.h10020[0],
                            IP_H10200 + offset, beamline->drivingTerms.h10200[0],
                            IP_H22000 + offset, beamline->drivingTerms.h22000[0],
                            IP_H11110 + offset, beamline->drivingTerms.h11110[0],
                            IP_H00220 + offset, beamline->drivingTerms.h00220[0],
                            IP_H31000 + offset, beamline->drivingTerms.h31000[0],
                            IP_H40000 + offset, beamline->drivingTerms.h40000[0],
                            IP_H20110 + offset, beamline->drivingTerms.h20110[0],
                            IP_H11200 + offset, beamline->drivingTerms.h11200[0],
                            IP_H20020 + offset, beamline->drivingTerms.h20020[0],
                            IP_H20200 + offset, beamline->drivingTerms.h20200[0],
                            IP_H00310 + offset, beamline->drivingTerms.h00310[0],
                            IP_H00400 + offset, beamline->drivingTerms.h00400[0],
                            /* real parts */
                            IP_H11001 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h11001[1],
                            IP_H00111 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h00111[1],
                            IP_H10010 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10010[1],
                            IP_H10100 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10100[1],
                            IP_H20001 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h20001[1],
                            IP_H00201 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h00201[1],
                            IP_H10002 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10002[1],
                            IP_H21000 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h21000[1],
                            IP_H30000 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h30000[1],
                            IP_H10110 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10110[1],
                            IP_H10020 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10020[1],
                            IP_H10200 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h10200[1],
                            IP_H22000 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h22000[1],
                            IP_H11110 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h11110[1],
                            IP_H00220 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h00220[1],
                            IP_H31000 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h31000[1],
                            IP_H40000 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h40000[1],
                            IP_H20110 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h20110[1],
                            IP_H11200 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h11200[1],
                            IP_H20020 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h20020[1],
                            IP_H20200 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h20200[1],
                            IP_H00310 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h00310[1],
                            IP_H00400 + (IP_H00400 - IP_H11001 + 1) + offset, beamline->drivingTerms.h00400[1],
                            /* imaginary parts */
                            IP_H11001 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h11001[2],
                            IP_H00111 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h00111[2],
                            IP_H10010 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10010[2],
                            IP_H10100 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10100[2],
                            IP_H20001 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h20001[2],
                            IP_H00201 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h00201[2],
                            IP_H10002 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10002[2],
                            IP_H21000 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h21000[2],
                            IP_H30000 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h30000[2],
                            IP_H10110 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10110[2],
                            IP_H10020 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10020[2],
                            IP_H10200 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h10200[2],
                            IP_H22000 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h22000[2],
                            IP_H11110 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h11110[2],
                            IP_H00220 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h00220[2],
                            IP_H31000 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h31000[2],
                            IP_H40000 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h40000[2],
                            IP_H20110 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h20110[2],
                            IP_H11200 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h11200[2],
                            IP_H20020 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h20020[2],
                            IP_H20200 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h20200[2],
                            IP_H00310 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h00310[2],
                            IP_H00400 + (IP_H00400 - IP_H11001 + 1) * 2 + offset, beamline->drivingTerms.h00400[2],
                            -1)) {
      SDDS_SetError((char *)"Problem setting SDDS parameters (dump_twiss_parameters 1.5)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (radIntegrals) {
    if (!SDDS_SetParameters(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                            IP_I1, radIntegrals->RI[0],
                            IP_I2, radIntegrals->RI[1],
                            IP_I3, radIntegrals->RI[2],
                            IP_I4, radIntegrals->RI[3],
                            IP_I5, radIntegrals->RI[4],
                            IP_EX0, radIntegrals->ex0,
                            IP_TAUX, radIntegrals->taux,
                            IP_JX, radIntegrals->Jx,
                            IP_TAUY, radIntegrals->tauy,
                            IP_JY, radIntegrals->Jy,
                            IP_SIGMADELTA, radIntegrals->sigmadelta,
                            IP_TAUDELTA, radIntegrals->taudelta,
                            IP_JDELTA, radIntegrals->Jdelta,
                            IP_U0, radIntegrals->Uo,
                            IP_ENX0, radIntegrals->ex0 * sqrt(sqr(run->p_central) + 1),
                            IP_I3POS, radIntegrals->I3pos,
                            IP_I3NEG, radIntegrals->I3neg,
                            IP_TAUP, radIntegrals->tauP,
                            IP_PEQ, radIntegrals->Peq,
                            IP_PSTLIMIT, radIntegrals->PstLimit,
                            -1)) {
      SDDS_SetError((char *)"Problem setting SDDS parameters (dump_twiss_parameters 2)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (!final_values_only_inner_scope) {
    ELEMENT_LIST *eptr;
    row_count = 0;
    eptr = elem;
    data[0] = eptr->beg_pos; /* position */
    copy_doubles(data + 1, (double *)twiss0, N_DOUBLE_COLUMNS - 2);
    data[N_DOUBLE_COLUMNS - 1] = elem->Pref_input;
    for (j = 0; j < N_DOUBLE_COLUMNS; j++)
      if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count, j, data[j], -1)) {
        SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count++,
                           IC_ELEMENT, "_BEG_", IC_OCCURENCE, (long)1, IC_TYPE, (char *)"MARK", -1)) {
      SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    if (s_dependent_driving_terms_file) {
      if (!SDDS_StartTable(&SDDS_SDrivingTerms, beamline->n_elems + 1)) {
        SDDS_SetError((char *)"Unable to start SDDS table (s_dependent_driving_terms_file)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      eptr = elem;
      i = 0;
      while (eptr) {
        SetSDrivingTermsRow(&SDDS_SDrivingTerms, i, i + 1, eptr->end_pos, eptr->name, entity_name[eptr->type], eptr->occurence, beamline);
        if (i == (beamline->n_elems - 1)) {
          SetSDrivingTermsRow(&SDDS_SDrivingTerms, i, 0, 0.0, "_BEG_", entity_name[T_MARK], 1, beamline);
        }
        i++;
        eptr = eptr->succ;
      }
      if (!SDDS_WriteTable(&SDDS_SDrivingTerms)) {
        SDDS_SetError((char *)"Unable to write SDDS table (s_dependent_driving_terms_file)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }

    i = 0;
    while (elem) {
      data[0] = elem->end_pos; /* position */
      data[N_DOUBLE_COLUMNS - 1] = elem->Pref_output;
      if (!elem->twiss)
        bombElegant((char *)"Twiss data not computed prior to dump_twiss_parameters() call (2)", NULL);
      copy_doubles(data + 1, (double *)elem->twiss, N_DOUBLE_COLUMNS - 2);
      for (j = 0; j < N_DOUBLE_COLUMNS; j++)
        if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count, j, data[j], -1)) {
          SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count,
                             IC_ELEMENT, elem->name, IC_OCCURENCE, elem->occurence,
                             IC_TYPE, entity_name[elem->type],
                             IC_CHAMBER_SHAPE, chamberShapeChoice[elem->chamberShape], -1)) {
        SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

      if (radIntegrals) {
        if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count,
                               IC_I1, elem->twiss->dI[0],
                               IC_I2, elem->twiss->dI[1],
                               IC_I3, elem->twiss->dI[2],
                               IC_I4, elem->twiss->dI[3],
                               IC_I5, elem->twiss->dI[4],
                               -1)) {
          SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      i++;
      row_count++;
      elem = elem->succ;
    }
    if (i != n_elem)
      bombElegant((char *)"element count error in dump_twiss_parameters()", NULL);
  } else {
    /* find final element */
    i = 0;
    while (1) {
      if (!elem->twiss)
        bombElegant((char *)"Twiss data not computed prior to dump_twiss_parameters() call (2)", NULL);
      i++;
      if (!elem->succ)
        break;
      elem = elem->succ;
    }
    if (i != n_elem)
      bombElegant((char *)"element count error in dump_twiss_parameters()", NULL);
    data[0] = elem->end_pos; /* position */
    data[N_DOUBLE_COLUMNS - 1] = elem->Pref_output;
    copy_doubles(data + 1, (double *)elem->twiss, N_DOUBLE_COLUMNS - 2);

    for (j = 0; j < N_DOUBLE_COLUMNS; j++)
      if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0, j, data[j], -1)) {
        SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                           IC_ELEMENT, elem->name, IC_OCCURENCE, elem->occurence,
                           IC_TYPE, entity_name[elem->type],
                           IC_CHAMBER_SHAPE, chamberShapeChoice[elem->chamberShape], -1)) {
      SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (radIntegrals) {
      if (!SDDS_SetRowValues(&SDDS_twiss, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                             IC_I1, elem->twiss->dI[0],
                             IC_I2, elem->twiss->dI[1],
                             IC_I3, elem->twiss->dI[2],
                             IC_I4, elem->twiss->dI[3],
                             IC_I5, elem->twiss->dI[4],
                             -1)) {
        SDDS_SetError((char *)"Problem setting SDDS rows (dump_twiss_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }

  if (!SDDS_WriteTable(&SDDS_twiss)) {
    SDDS_SetError((char *)"Unable to write Twiss parameter data (dump_twiss_parameters)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_twiss);
  if (!SDDS_EraseData(&SDDS_twiss)) {
    SDDS_SetError((char *)"Unable to erase Twiss parameter data (dump_twiss_parameters)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  free(data);
  if (tune_corrected)
    twiss_count++;

  log_exit((char *)"dump_twiss_parameters");
}

long get_twiss_mode(long *mode, TWISS *twissRet) {
  if (!twiss_initialized)
    return (0);
  if ((*mode = matched)) {
    if (periodicTwissComputed) {
      *mode = 0;
    } else
      return 0;
  }
  if (!matched) {
    twissRet->betax = beta_x;
    twissRet->alphax = alpha_x;
    twissRet->etax = eta_x;
    twissRet->etapx = etap_x;
    twissRet->betay = beta_y;
    twissRet->alphay = alpha_y;
    twissRet->etay = eta_y;
    twissRet->etapy = etap_y;
  } else
    memcpy(twissRet, &lastPeriodicTwiss, sizeof(*twissRet));
  return (1);
}

void setup_twiss_output(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *do_twiss_output,
                        long default_order) {

  log_entry((char *)"setup_twiss_output");

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&twiss_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &twiss_output);

  mirror = 0;
  if (matched == -1)
    mirror = 1;
  if (filename)
    filename = compose_filename(filename, run->rootname);
  if (s_dependent_driving_terms_file) {
    s_dependent_driving_terms_file = compose_filename(s_dependent_driving_terms_file, run->rootname);
    if (!filename)
      bombElegant("You must give a value for filename if you give one for s_dependent_driving_terms_file", NULL);
  }
  twissConcatOrder = concat_order;
  if (twissConcatOrder < default_order)
    twissConcatOrder = default_order;
  *do_twiss_output = output_at_each_step;
  if (higher_order_chromaticity && !matched)
    bombElegant((char *)"higher order chromaticity calculations available only for a matched (periodic) beamline", NULL);

  if (reference_file && matched)
    bombElegant((char *)"reference_file and matched=1 are incompatible", NULL);
  if (!matched) {
    if (reference_file) {
      if (reference_element && reference_element_occurrence < 0)
        bombElegant((char *)"invalid value of reference_element_occurrence---use 0 for last occurrence, >=1 for specific occurrence.", NULL);
      LoadStartingTwissFromFile(&beta_x, &beta_y, &alpha_x, &alpha_y,
                                &eta_x, &etap_x, &eta_y, &etap_y,
                                reference_file, reference_element,
                                reference_element_occurrence);
      if (reflect_reference_values) {
        alpha_x = -alpha_x;
        alpha_y = -alpha_y;
        etap_x = -etap_x;
        etap_y = -etap_y;
      }
      printf((char *)"Starting twiss parameters from reference file:\nbeta, alpha x: %le, %le\nbeta, alpha y: %le, %le\n",
             beta_x, alpha_x, beta_y, alpha_y);
      printf((char *)"eta, eta' x: %le, %le\neta, eta' y: %le, %le\n",
             eta_x, etap_x, eta_y, etap_y);
      fflush(stdout);
    }
    if (beta_x <= 0 || beta_y <= 0)
      bombElegant((char *)"invalid initial beta-functions given in twiss_output namelist", NULL);
  }

#if USE_MPI
  if (writePermitted) {
#endif
    if (filename) {
#if SDDS_MPI_IO
      SDDS_twiss.parallel_io = 0;
#endif
      SDDS_ElegantOutputSetup(&SDDS_twiss, filename, SDDS_BINARY, 1, (char *)"Twiss parameters",
                              run->runfile, run->lattice, parameter_definition,
                              (radiation_integrals ? N_PARAMETERS : IP_ALPHAC + 1),
                              column_definition, (radiation_integrals ? N_COLUMNS_WRI : N_COLUMNS), (char *)"setup_twiss_output",
                              SDDS_EOS_NEWFILE);
      if (compute_driving_terms) {
        long i;
        for (i = 0; i < N_DT_PARAMETERS; i++) {
          if (!SDDS_ProcessParameterString(&SDDS_twiss, driving_term_parameter_definition[i].text, 0) ||
              (SDDS_GetParameterIndex(&SDDS_twiss, driving_term_parameter_definition[i].name)) < 0) {
            printf("Unable to define SDDS parameter for driving terms--string was:\n%s\n",
                   driving_term_parameter_definition[i].text);
            fflush(stdout);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
            exitElegant(1);
          }
        }
      }
      if (!SDDS_WriteLayout(&SDDS_twiss)) {
        printf("Unable to write SDDS layout for file %s\n", filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      SDDS_twiss_initialized = 1;
      twiss_count = 0;
    } else
      SDDS_twiss_initialized = 0;
#if USE_MPI
  }
#endif
  twiss_initialized = 1;

#if USE_MPI
  if (writePermitted) {
#endif
    if (s_dependent_driving_terms_file) {
      if (!SDDS_InitializeOutputElegant(&SDDS_SDrivingTerms, SDDS_ASCII, 1L, NULL, NULL, s_dependent_driving_terms_file)) {
        SDDS_SetError((char *)"Unable set up SDDS file (s_dependent_driving_terms_file)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

      if (SDDS_DefineColumn(&SDDS_SDrivingTerms, "s", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "ElementType", NULL, NULL, NULL, NULL, SDDS_STRING, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f10010", NULL,
                            "1/m$a1/2$n", "f10010 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f10100", NULL,
                            "1/m$a1/2$n", "f10100 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f30000", NULL,
                            "1/m$a1/2$n", "f30000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f12000", NULL,
                            "1/m$a1/2$n", "f12000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f10200", NULL,
                            "1/m$a1/2$n", "f10200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f01200", NULL,
                            "1/m$a1/2$n", "f01200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f01110", NULL,
                            "1/m$a1/2$n", "f01110 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f00300", NULL,
                            "1/m$a1/2$n", "f00300 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f00120", NULL,
                            "1/m$a1/2$n", "f00120 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f20100", NULL,
                            "1/m$a1/2$n", "f20100 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f20010", NULL,
                            "1/m$a1/2$n", "f20010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "f11010", NULL,
                            "1/m$a1/2$n", "f11010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||

          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf10010", NULL,
                            "1/m$a1/2$n", "f10010 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf10100", NULL,
                            "1/m$a1/2$n", "f10100 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf30000", NULL,
                            "1/m$a1/2$n", "f30000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf12000", NULL,
                            "1/m$a1/2$n", "f12000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf10200", NULL,
                            "1/m$a1/2$n", "f10200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf01200", NULL,
                            "1/m$a1/2$n", "f01200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf01110", NULL,
                            "1/m$a1/2$n", "f01110 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf00300", NULL,
                            "1/m$a1/2$n", "f00300 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf00120", NULL,
                            "1/m$a1/2$n", "f00120 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf20100", NULL,
                            "1/m$a1/2$n", "f20100 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf20010", NULL,
                            "1/m$a1/2$n", "f20010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Imf11010", NULL,
                            "1/m$a1/2$n", "f11010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||

          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref10010", NULL,
                            "1/m$a1/2$n", "f10010 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref10100", NULL,
                            "1/m$a1/2$n", "f10100 Skew quadrupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref30000", NULL,
                            "1/m$a1/2$n", "f30000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref12000", NULL,
                            "1/m$a1/2$n", "f12000 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref10200", NULL,
                            "1/m$a1/2$n", "f10200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref01200", NULL,
                            "1/m$a1/2$n", "f01200 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref01110", NULL,
                            "1/m$a1/2$n", "f01110 Normal sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref00300", NULL,
                            "1/m$a1/2$n", "f00300 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref00120", NULL,
                            "1/m$a1/2$n", "f00120 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref20100", NULL,
                            "1/m$a1/2$n", "f20100 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref20010", NULL,
                            "1/m$a1/2$n", "f20010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          SDDS_DefineColumn(&SDDS_SDrivingTerms, "Ref11010", NULL,
                            "1/m$a1/2$n", "f11010 Skew sextupole-like RDT", NULL, SDDS_DOUBLE, 0) == -1 ||
          !SDDS_WriteLayout(&SDDS_SDrivingTerms)) {
        SDDS_SetError((char *)"Unable to define SDDS column (s_dependent_driving_terms_file)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
#if USE_MPI
  }
#endif

  beamline->sDrivingTerms.f10010 = beamline->sDrivingTerms.f10100 = NULL;
  beamline->sDrivingTerms.f30000 = beamline->sDrivingTerms.f12000 = NULL;
  beamline->sDrivingTerms.f10200 = beamline->sDrivingTerms.f01200 = NULL;
  beamline->sDrivingTerms.f01110 = beamline->sDrivingTerms.f00300 = NULL;
  beamline->sDrivingTerms.f00120 = beamline->sDrivingTerms.f20100 = NULL;
  beamline->sDrivingTerms.f20010 = beamline->sDrivingTerms.f11010 = NULL;

  /* beamline->flags &= ~BEAMLINE_BACKTRACKING; */
  beamline->flags |= BEAMLINE_MATRICES_NEEDED;
  beamline->flags |= BEAMLINE_TWISS_WANTED;
  if (radiation_integrals)
    beamline->flags |= BEAMLINE_RADINT_WANTED;

  beamline->chromDeltaHalfRange = chromatic_tune_spread_half_range;

  log_exit((char *)"setup_twiss_output");
}

void finish_twiss_output(LINE_LIST *beamline) {
  log_entry((char *)"finish_twiss_output");
  if (SDDS_twiss_initialized && !SDDS_Terminate(&SDDS_twiss)) {
    SDDS_SetError((char *)"Problem terminating SDDS output (finish_twiss_output)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDS_twiss_initialized = twiss_count = 0;
  if (doTuneShiftWithAmplitude && tune_shift_with_amplitude_struct.tune_output)
    SDDS_Terminate(&SDDSTswaTunes);
  doTuneShiftWithAmplitude = 0;

  if (s_dependent_driving_terms_file) {
    tfree(beamline->sDrivingTerms.f10010);
    tfree(beamline->sDrivingTerms.f10100);
    tfree(beamline->sDrivingTerms.f30000);
    tfree(beamline->sDrivingTerms.f12000);
    tfree(beamline->sDrivingTerms.f10200);
    tfree(beamline->sDrivingTerms.f01200);
    tfree(beamline->sDrivingTerms.f01110);
    tfree(beamline->sDrivingTerms.f00300);
    tfree(beamline->sDrivingTerms.f00120);
    tfree(beamline->sDrivingTerms.f20100);
    tfree(beamline->sDrivingTerms.f20010);
    tfree(beamline->sDrivingTerms.f11010);

    beamline->sDrivingTerms.f10010 = beamline->sDrivingTerms.f10100 = NULL;
    beamline->sDrivingTerms.f30000 = beamline->sDrivingTerms.f12000 = NULL;
    beamline->sDrivingTerms.f10200 = beamline->sDrivingTerms.f01200 = NULL;
    beamline->sDrivingTerms.f01110 = beamline->sDrivingTerms.f00300 = NULL;
    beamline->sDrivingTerms.f00120 = beamline->sDrivingTerms.f20100 = NULL;
    beamline->sDrivingTerms.f20010 = beamline->sDrivingTerms.f11010 = NULL;

    if (!SDDS_Terminate(&SDDS_SDrivingTerms)) {
      SDDS_SetError((char *)"Problem terminating SDDS s-dependent driving term output (finish_twiss_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  log_exit((char *)"finish_twiss_output");
}

long run_twiss_output(RUN *run, LINE_LIST *beamline, double *starting_coord, long tune_corrected) {
  ELEMENT_LIST *eptr, *elast;
  long n_elem, last_n_elem;
  unsigned long unstable;
  TWISS twiss_ave, twiss_min, twiss_max;

  /*
    if (beamline->flags&BEAMLINE_TWISS_CURRENT)
    return;
  */

  log_entry((char *)"run_twiss_output");

#ifdef DEBUG
  report_stats(stdout, "now in run_twiss_output\n");
  printf("tune_corrected = %ld\n", tune_corrected);
  {
    long i;
    printf("Starting coordinates: ");
    for (i = 0; i < 6; i++)
      printf("%21.15e%c", starting_coord[i], i == 5 ? '\n' : ' ');
  }
#endif
  if (tune_corrected == 0 && !output_before_tune_correction) {
    log_exit((char *)"run_twiss_output");
    return 1;
  }

  eptr = beamline->elem_twiss = beamline->elem;
  n_elem = last_n_elem = beamline->n_elems;
  while (eptr) {
    if (eptr->type == T_RECIRC && matched) {
      last_n_elem = n_elem;
      if (eptr->pred)
        beamline->elem_twiss = eptr->pred;
      else
        beamline->elem_twiss = eptr;
      beamline->elem_recirc = eptr;
    }
    eptr = eptr->succ;
    n_elem--;
  }
  n_elem = last_n_elem;
#ifdef DEBUG
  report_stats(stdout, "counted elements: ");
#endif
  compute_twiss_parameters(run, beamline, starting_coord, matched, radiation_integrals,
                           beta_x, alpha_x, eta_x, etap_x,
                           beta_y, alpha_y, eta_y, etap_y, &unstable);
  elast = beamline->elast;

  if (twissConcatOrder > 1) {
    printf((char *)"%s Twiss parameters (chromaticity valid for fully second-order calculation only!):\n",
           matched ? (char *)"periodic" : (char *)"final");
    printf((char *)"         beta          alpha           nu           eta          eta'       dnu/d(dp/p)   dbeta/(dp/p)     accept.\n");
    printf((char *)"          m                          1/2pi           m                         1/2pi            m          mm-mrad\n");
    printf((char *)"--------------------------------------------------------------------------------------------------------------------\n");
    printf((char *)"  x: %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n",
           elast->twiss->betax, elast->twiss->alphax, beamline->tune[0], elast->twiss->etax, elast->twiss->etapx,
           beamline->chromaticity[0], beamline->dbeta_dPoP[0], 1e6 * beamline->acceptance[0]);
    if (statistics) {
      compute_twiss_statistics(beamline, &twiss_ave, &twiss_min, &twiss_max);
      printf((char *)"ave: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_ave.betax, twiss_ave.alphax, (char *)"", twiss_ave.etax, twiss_ave.etapx);
      printf((char *)"min: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_min.betax, twiss_min.alphax, (char *)"", twiss_min.etax, twiss_min.etapx);
      printf((char *)"max: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_max.betax, twiss_max.alphax, (char *)"", twiss_max.etax, twiss_max.etapx);
    }
    printf((char *)"  y: %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n",
           elast->twiss->betay, elast->twiss->alphay, beamline->tune[1], elast->twiss->etay, elast->twiss->etapy,
           beamline->chromaticity[1], beamline->dbeta_dPoP[1], 1e6 * beamline->acceptance[1]);
    if (statistics) {
      printf((char *)"ave: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_ave.betay, twiss_ave.alphay, (char *)"", twiss_ave.etay, twiss_ave.etapy);
      printf((char *)"min: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_min.betay, twiss_min.alphay, (char *)"", twiss_min.etay, twiss_min.etapy);
      printf((char *)"max: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_max.betay, twiss_max.alphay, (char *)"", twiss_max.etay, twiss_max.etapy);
    }
    fflush(stdout);
  } else {
    printf((char *)"%s Twiss parameters:\n", matched ? (char *)"periodic" : (char *)"final");
    printf((char *)"         beta          alpha           nu           eta          eta'        accept.\n");
    printf((char *)"          m                          1/2pi           m                       mm-mrad\n");
    printf((char *)"---------------------------------------------------------------------------------------\n");
    printf((char *)"  x: %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n",
           elast->twiss->betax, elast->twiss->alphax, beamline->tune[0], elast->twiss->etax, elast->twiss->etapx,
           1e6 * beamline->acceptance[0]);
    if (statistics) {
      compute_twiss_statistics(beamline, &twiss_ave, &twiss_min, &twiss_max);
      printf((char *)"ave: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_ave.betax, twiss_ave.alphax, (char *)"", twiss_ave.etax, twiss_ave.etapx);
      printf((char *)"min: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_min.betax, twiss_min.alphax, (char *)"", twiss_min.etax, twiss_min.etapx);
      printf((char *)"max: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_max.betax, twiss_max.alphax, (char *)"", twiss_max.etax, twiss_max.etapx);
    }
    printf((char *)"  y: %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n",
           elast->twiss->betay, elast->twiss->alphay, beamline->tune[1], elast->twiss->etay, elast->twiss->etapy,
           1e6 * beamline->acceptance[1]);
    if (statistics) {
      printf((char *)"ave: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_ave.betay, twiss_ave.alphay, (char *)"", twiss_ave.etay, twiss_ave.etapy);
      printf((char *)"min: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_min.betay, twiss_min.alphay, (char *)"", twiss_min.etay, twiss_min.etapy);
      printf((char *)"max: %13.6e %13.6e %-13s %13.6e %13.6e\n",
             twiss_max.betay, twiss_max.alphay, (char *)"", twiss_max.etay, twiss_max.etapy);
    }
    fflush(stdout);
  }

  if (beamline->acc_limit_name[0]) {
    printf((char *)"x acceptance limited to %e by %s ending at %e m\n", beamline->acceptance[0], beamline->acc_limit_name[0], beamline->acceptance[2]);
    fflush(stdout);
  }
  if (beamline->acc_limit_name[1]) {
    printf((char *)"y acceptance limited to %e by %s ending at %e m\n", beamline->acceptance[1], beamline->acc_limit_name[1], beamline->acceptance[3]);
    fflush(stdout);
  }

  if (SDDS_twiss_initialized) {
    dump_twiss_parameters(beamline, n_elem,
                          beamline->tune,
                          radiation_integrals ? &(beamline->radIntegrals) : NULL,
                          beamline->chromaticity, beamline->dbeta_dPoP,
                          beamline->dalpha_dPoP,
                          beamline->acceptance,
                          beamline->acc_limit_name,
                          beamline->alpha, final_values_only, tune_corrected, run);
  }

  if (isnan(beamline->tune[0]) ||
      isnan(elast->twiss->betax) ||
      isnan(elast->twiss->etax) ||
      isnan(beamline->tune[1]) ||
      isnan(elast->twiss->betay) ||
      isnan(elast->twiss->etay))
    return 0;
  if (beamline->acc_limit_name[0]) {
    free(beamline->acc_limit_name[0]);
    beamline->acc_limit_name[0] = NULL;
  }
  if (beamline->acc_limit_name[1]) {
    free(beamline->acc_limit_name[1]);
    beamline->acc_limit_name[1] = NULL;
  }

  computeAllUndulatorBrightnesses(beamline);

  log_exit((char *)"run_twiss_output");
  return 1;
}

void compute_twiss_parameters(RUN *run, LINE_LIST *beamline, double *starting_coord,
                              long periodic,
                              long radiation_integrals_inner_scope,
                              double betax, double alphax, double etax, double etapx,
                              double betay, double alphay, double etay, double etapy,
                              unsigned long *unstable) {
  VMATRIX *M;
  double chromx, chromy, dbetax, dbetay, alpha1, alpha2, alpha3, dalphax, dalphay;
  double x_acc_z, y_acc_z;
  ELEMENT_LIST *eptr, *elast;
  char *x_acc_name, *y_acc_name;
  long i, j;
  double ending_coord[MAX_PROPERTIES_PER_PARTICLE];

  log_entry((char *)"compute_twiss_parameters");

  *unstable = 0;

#ifdef DEBUG
  if (starting_coord) {
    printf("compute_twiss_parameters: starting_coord = %le, %le, %le, %le, %le, %le\n",
           starting_coord[0],
           starting_coord[1],
           starting_coord[2],
           starting_coord[3],
           starting_coord[4],
           starting_coord[5]);
  }
#endif

  if (!beamline->twiss0)
    beamline->twiss0 = (TWISS *)tmalloc(sizeof(*beamline->twiss0));

  eptr = beamline->elem_twiss = beamline->elem;
  elast = eptr;
  while (eptr) {
    if (eptr->type == T_RECIRC && matched)
      beamline->elem_twiss = beamline->elem_recirc = eptr;
    elast = eptr;
    eptr = eptr->succ;
  }
  beamline->elast = elast;

  if (periodic) {
    if (beamline->matrix) {
      free_matrices(beamline->matrix);
      free(beamline->matrix);
      beamline->matrix = NULL;
    }

    beamline->matrix = compute_periodic_twiss(&betax, &alphax, &etax, &etapx, beamline->tune,
                                              &betay, &alphay, &etay, &etapy, beamline->tune + 1,
                                              beamline->elem_twiss, starting_coord, run,
                                              unstable,
                                              beamline->eta2, beamline->eta3);
#ifdef DEBUG
    printf((char *)"matched parameters computed--returned to compute_twiss_parameters\n");
    fflush(stdout);
    printf((char *)"beamline matrix has order %ld\n", beamline->matrix->order);
    fflush(stdout);
    printf((char *)"beamline matrix pointers:  %x, %x, %x, %x\n",
           beamline->matrix, beamline->matrix->C, beamline->matrix->R, beamline->matrix->T);
    fflush(stdout);
#endif
    if (twissConcatOrder >= 2 && !(beamline->matrix->T))
      bombElegant((char *)"logic error: T matrix is NULL on return from compute_periodic_twiss", NULL);
  } else {
    VMATRIX *M1;
    if (twissConcatOrder > 1 && starting_coord) {
      M1 = (VMATRIX *)tmalloc(sizeof(*M1));
      initialize_matrices(M1, twissConcatOrder);
      for (i = 0; i < 6; i++) {
        M1->C[i] = starting_coord[i];
        M1->R[i][i] = 1;
      }
      fill_in_matrices(beamline->elem_twiss, run);
      if (beamline->matrix) {
        free_matrices(beamline->matrix);
        free(beamline->matrix);
      }
      beamline->matrix = append_full_matrix(beamline->elem_twiss, run, M1, twissConcatOrder);
      free_matrices(M1);
      free(M1);
    } else
      beamline->matrix = full_matrix(beamline->elem_twiss, run, twissConcatOrder);
  }

  beamline->twiss0->betax = betax;
  beamline->twiss0->alphax = alphax;
  beamline->twiss0->phix = 0;
  beamline->twiss0->etax = etax;
  beamline->twiss0->etapx = etapx;
  beamline->twiss0->betay = betay;
  beamline->twiss0->alphay = alphay;
  beamline->twiss0->phiy = 0;
  beamline->twiss0->etay = etay;
  beamline->twiss0->etapy = etapy;
  if (starting_coord) {
    beamline->twiss0->Cx = starting_coord[0];
    beamline->twiss0->Cy = starting_coord[2];
  } else
    beamline->twiss0->Cx = beamline->twiss0->Cy = 0;

  if (twissConcatOrder >= 2 && !(beamline->matrix->T))
    bombElegant((char *)"logic error: beamline T matrix is NULL in compute_twiss_parameters", NULL);

#ifdef DEBUG
  printf((char *)"propagating parameters\n");
  fflush(stdout);
  printf((char *)"beamline matrix pointers:  %x, %x, %x, %x\n",
         beamline->matrix, beamline->matrix->C, beamline->matrix->R, beamline->matrix->T);
  fflush(stdout);
#endif

  beamline->radIntegrals.computed = 0;

  propagate_twiss_parameters(beamline->twiss0, beamline->tune, beamline->waists,
                             (radiation_integrals_inner_scope ? &(beamline->radIntegrals) : NULL),
                             beamline->elem_twiss, run, starting_coord, ending_coord,
                             beamline->couplingFactor);

  if (radiation_integrals_inner_scope)
    completeRadiationIntegralComputation(&(beamline->radIntegrals), beamline, run->p_central, ending_coord);

#ifdef DEBUG
  printf((char *)"finding acceptance\n");
  fflush(stdout);
#endif
  beamline->acceptance[0] = find_acceptance(beamline->elem_twiss, 0, run, &x_acc_name, &x_acc_z);
  beamline->acceptance[1] = find_acceptance(beamline->elem_twiss, 1, run, &y_acc_name, &y_acc_z);
  findChamberShapes(beamline->elem_twiss);

  beamline->acceptance[2] = x_acc_z;
  beamline->acceptance[3] = y_acc_z;

  if (x_acc_name) {
    if (beamline->acc_limit_name[0])
      free(beamline->acc_limit_name[0]);
    beamline->acc_limit_name[0] = NULL;
    cp_str(&beamline->acc_limit_name[0], x_acc_name);
  } else
    beamline->acc_limit_name[0] = NULL;
  if (y_acc_name) {
    if (beamline->acc_limit_name[1])
      free(beamline->acc_limit_name[1]);
    beamline->acc_limit_name[1] = NULL;
    cp_str(&beamline->acc_limit_name[1], y_acc_name);
  } else
    beamline->acc_limit_name[1] = NULL;

  beamline->twiss0->apx = beamline->elem_twiss->twiss->apx;
  beamline->twiss0->apy = beamline->elem_twiss->twiss->apy;

#ifdef DEBUG
  printf((char *)"computing chromaticities\n");
  fflush(stdout);
#endif

  chromx = chromy = 0;
  dbetax = dbetay = 0;
  dalphax = dalphay = 0;

  for (i = 0; i < N_TSWA; i++)
    for (j = 0; j < N_TSWA; j++)
      beamline->dnux_dA[i][j] = beamline->dnuy_dA[i][j] = 0;

  for (i = 0; i < 3; i++)
    beamline->drivingTerms.h21000[i] = beamline->drivingTerms.h30000[i] = beamline->drivingTerms.h10110[i] =
      beamline->drivingTerms.h10020[i] = beamline->drivingTerms.h10200[i] =
      beamline->drivingTerms.h22000[i] = beamline->drivingTerms.h11110[i] = beamline->drivingTerms.h00220[i] = beamline->drivingTerms.h31000[i] =
      beamline->drivingTerms.h40000[i] = beamline->drivingTerms.h20110[i] = beamline->drivingTerms.h11200[i] = beamline->drivingTerms.h20020[i] =
      beamline->drivingTerms.h20200[i] = beamline->drivingTerms.h00310[i] = beamline->drivingTerms.h00400[i] = 0;

  if (periodic) {
    if (!(M = beamline->matrix))
      bombElegant((char *)"logic error: revolution matrix is NULL in compute_twiss_parameters", NULL);

    if (twissConcatOrder > 1) {
#ifdef DEBUG
      printf((char *)"computing chromaticities\n");
      fflush(stdout);
#endif
      if (!(M->T))
        bombElegant((char *)"logic error: T matrix is NULL in compute_twiss_parameters", NULL);
      computeChromaticities(&chromx, &chromy,
                            &dbetax, &dbetay, &dalphax, &dalphay, beamline->twiss0,
                            beamline->elast->twiss, M);
      beamline->chromaticity[0] = chromx;
      beamline->chromaticity[1] = chromy;
      if (twissConcatOrder > 1 && higher_order_chromaticity)
        computeHigherOrderChromaticities(beamline, starting_coord, run, twissConcatOrder,
                                         higher_order_chromaticity_range /
                                         (higher_order_chromaticity_points - 1),
                                         higher_order_chromaticity_points, quick_higher_order_chromaticity);
      beamline->chromaticity[0] *= n_periods;
      beamline->chromaticity[1] *= n_periods;
      beamline->chrom2[0] *= n_periods;
      beamline->chrom3[0] *= n_periods;
      beamline->chrom2[1] *= n_periods;
      beamline->chrom3[1] *= n_periods;
      computeChromaticTuneLimits(beamline);
      if (doTuneShiftWithAmplitude)
        computeTuneShiftWithAmplitude(beamline->dnux_dA, beamline->dnuy_dA,
                                      beamline->nuxTswaExtrema, beamline->nuyTswaExtrema,
                                      beamline->twiss0, beamline->tune, M, beamline, run,
                                      starting_coord, n_periods);
      if (compute_driving_terms)
        computeDrivingTerms(&(beamline->drivingTerms), beamline->elem_twiss, beamline->twiss0, beamline->tune, n_periods);
      if (s_dependent_driving_terms_file)
        computeSDrivingTerms(beamline);
    }
  } else {
    M = beamline->matrix;
    elast = beamline->elast;

    if (!elast)
      bombElegant((char *)"logic error in compute_twiss_parameters--elast pointer is NULL", NULL);
    if (!elast->twiss)
      bombElegant((char *)"logic error in compute_twiss_parameters--elast->twiss pointer is NULL", NULL);

    betax = elast->twiss->betax;
    alphax = elast->twiss->alphax;
    etax = elast->twiss->etax;
    etapx = elast->twiss->etapx;
    betay = elast->twiss->betay;
    alphay = elast->twiss->alphay;
    etay = elast->twiss->etay;
    etapy = elast->twiss->etapy;

    if (twissConcatOrder >= 2) {
      if (!(M->T))
        bombElegant((char *)"logic error: T matrix is NULL in compute_twiss_parameters", NULL);
      computeChromaticities(&chromx, &chromy,
                            &dbetax, &dbetay, &dalphax, &dalphay, beamline->twiss0,
                            beamline->elast->twiss, M);
      beamline->chromaticity[0] = chromx;
      beamline->chromaticity[1] = chromy;
      if (twissConcatOrder > 1 && higher_order_chromaticity)
        computeHigherOrderChromaticities(beamline, starting_coord, run, twissConcatOrder,
                                         higher_order_chromaticity_range /
                                         (higher_order_chromaticity_points - 1),
                                         higher_order_chromaticity_points, quick_higher_order_chromaticity);
      beamline->chromaticity[0] *= n_periods;
      beamline->chromaticity[1] *= n_periods;
      beamline->chrom2[0] *= n_periods;
      beamline->chrom3[0] *= n_periods;
      beamline->chrom2[1] *= n_periods;
      beamline->chrom3[1] *= n_periods;
      computeChromaticTuneLimits(beamline);
      if (doTuneShiftWithAmplitude)
        computeTuneShiftWithAmplitude(beamline->dnux_dA, beamline->dnuy_dA,
                                      beamline->nuxTswaExtrema, beamline->nuyTswaExtrema,
                                      beamline->twiss0, beamline->tune, M, beamline, run,
                                      starting_coord, n_periods);
#ifdef DEBUG
      printf((char *)"chomaticities: %e, %e\n", chromx, chromy);
      fflush(stdout);
#endif
    }
  }
  beamline->dbeta_dPoP[0] = dbetax;
  beamline->dbeta_dPoP[1] = dbetay;
  beamline->dalpha_dPoP[0] = dalphax;
  beamline->dalpha_dPoP[1] = dalphay;

  alpha1 = alpha2 = alpha3 = 0.;
  if (beamline->matrix->C[4] != 0) {
    alpha1 = (beamline->matrix->R[4][5] +
              beamline->matrix->R[4][0] * elast->twiss->etax +
              beamline->matrix->R[4][1] * elast->twiss->etapx +
              beamline->matrix->R[4][2] * elast->twiss->etay +
              beamline->matrix->R[4][3] * elast->twiss->etapy) /
      beamline->matrix->C[4];
    if (beamline->matrix->T) {
      double eta[6];
      long k;
      eta[0] = elast->twiss->etax;
      eta[1] = elast->twiss->etapx;
      eta[2] = elast->twiss->etay;
      eta[3] = elast->twiss->etapy;
      eta[4] = 0;
      eta[5] = 1;
      for (j = 0; j < 4; j++)
        alpha2 += beamline->matrix->R[4][j] * beamline->eta2[j];
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          alpha2 += beamline->matrix->T[4][j][k] * eta[j] * eta[k];
      alpha2 /= beamline->matrix->C[4];

      if (beamline->matrix->Q) {
        long l;
        // for (j = 0; j < 6; ++j)
        //   for (k = 0; k <= j; ++k)
        //     for (l = 0; l <= k; ++l)
        //       printf("Q[4][%ld][%ld][%ld] = %lf\n", j, k, l, beamline->matrix->Q[4][j][k][l]);
        // for (j = 0; j < 6; ++j)
        //   for (k = 0; k <= j; ++k)
        //     printf("T[4][%ld][%ld] = %lf\n", j, k, beamline->matrix->T[4][j][k]);
        for (j = 0; j < 4; j++)
          alpha3 += beamline->matrix->R[4][j] * beamline->eta3[j];
        for (j = 0; j < 4; j++)
          for (k = 0; k <= j; k++)
            alpha3 += beamline->matrix->T[4][j][k] * (beamline->eta2[j] * beamline->eta2[k] +
                                                      eta[j] * beamline->eta2[k] +
                                                      beamline->eta2[j] * eta[k]);
        for (j = 0; j < 6; j++)
          for (k = 0; k <= j; k++)
            for (l = 0; l <= k; l++)
              alpha3 += beamline->matrix->Q[4][j][k][l] * eta[j] * eta[k] * eta[l];
        alpha3 /= beamline->matrix->C[4];
      } // alpha3
    }   // alpha2
  }     // alpha
  beamline->alpha[0] = alpha1;
  beamline->alpha[1] = alpha2;
  beamline->alpha[2] = alpha3;

  beamline->flags |= BEAMLINE_TWISS_DONE + BEAMLINE_TWISS_CURRENT;
  if (radiation_integrals_inner_scope)
    beamline->flags |= BEAMLINE_RADINT_DONE + BEAMLINE_RADINT_CURRENT;

  if (cavities_are_drifts_if_matched && mustResetRfcaMatrices)
    reset_rfca_matrices(beamline->elem, run->default_order);

  log_exit((char *)"compute_twiss_parameters");
}

void update_twiss_parameters(RUN *run, LINE_LIST *beamline, unsigned long *unstable) {
  unsigned long unstable0;
  static unsigned long nUpdates = 0;
  unsigned long updateInterval;
  nUpdates++;
  if (nUpdates<=100) {
    printf("Updating twiss parameters (%ld)\n", nUpdates);
  } else {
    updateInterval = sqrt(nUpdates);
    if (nUpdates%updateInterval==0) {
      printf("Updating twiss parameters (%ld)\n", nUpdates);
      fflush(stdout);
    }
  }
  if (linearChromaticTrackingInitialized) {
    /* This is a kludge to fool SREffects into thinking we've computed twiss parameters.
     * First, propagate the users parameters around the beamline.
     * Second, restore the global parameters in case the beamline isn't right (which is likely).
     */
    compute_twiss_parameters(run, beamline,
                             beamline->closed_orbit ? beamline->closed_orbit->centroid : NULL, 0, 0,
                             beamline->twiss0->betax, beamline->twiss0->alphax,
                             beamline->twiss0->etax, beamline->twiss0->etapx,
                             beamline->twiss0->betay, beamline->twiss0->alphay,
                             beamline->twiss0->etay, beamline->twiss0->etapy,
                             &unstable0);
    setLinearChromaticTrackingValues(beamline);

  } else {
    compute_twiss_parameters(run, beamline,
                             beamline->closed_orbit ? beamline->closed_orbit->centroid : NULL, matched,
                             radiation_integrals,
                             beta_x, alpha_x, eta_x, etap_x, beta_y, alpha_y, eta_y, etap_y,
                             &unstable0);
  }
  if (unstable)
    *unstable = unstable0;
}

void copy_doubles(double *target, double *source, long n) {
  log_entry((char *)"copy_doubles");
  while (n--)
    *target++ = *source++;
  log_exit((char *)"copy_doubles");
}

long has_aperture(ELEMENT_LIST *elem);

double find_acceptance(
                       ELEMENT_LIST *elem, long plane, RUN *run, char **name, double *z) {
  double beta, acceptance, tmp;
  double tube_aperture, aperture, centroid, offset;
  double other_centroid, a_tube, b_tube, aperture1;
  long aperture_set, elliptical_tube, tube_set, apIndex, acceptance_set;
  SCRAPER *scraper;
  SPEEDBUMP *speedbump;
  ELEMENT_LIST *ap_elem;
  TAPERAPC *taperApC;
  TAPERAPE *taperApE;
  TAPERAPR *taperApR;

  log_entry((char *)"find_acceptance");

  acceptance = tube_aperture = a_tube = b_tube = 0;
  elliptical_tube = tube_set = 0;
  acceptance_set = 0;
  *name = NULL;
  *z = -DBL_MAX;
  aperture = 10;
  while (elem) {
    beta = *(((double *)elem->twiss) + (plane ? TWISS_Y_OFFSET : 0));
    centroid = *(((double *)elem->twiss) + (plane ? 1 : 0) + TWISS_CENT_OFFSET);
    other_centroid = *(((double *)elem->twiss) + (plane ? 0 : 1) + TWISS_CENT_OFFSET);
    aperture = 0;
    aperture_set = 0;
    if (!has_aperture(elem) && has_aperture(elem->succ))
      ap_elem = elem->succ;
    else
      ap_elem = elem;
    switch (ap_elem->type) {
    case T_MAXAMP:
      if (((MAXAMP *)ap_elem->p_elem)->elliptical == 0) {
        elliptical_tube = tube_set = aperture_set = 0;
        if (plane)
          tube_aperture = ((MAXAMP *)ap_elem->p_elem)->y_max;
        else
          tube_aperture = ((MAXAMP *)ap_elem->p_elem)->x_max;
        if (tube_aperture > 0) {
          if ((aperture = tube_aperture - fabs(centroid)) < 0)
            aperture = 0;
          aperture_set = 1;
          tube_set = 1;
        }
      } else {
        elliptical_tube = 1;
        tube_set = aperture_set = 0;
        if (plane) {
          a_tube = ((MAXAMP *)ap_elem->p_elem)->y_max;
          b_tube = ((MAXAMP *)ap_elem->p_elem)->x_max;
        } else {
          a_tube = ((MAXAMP *)ap_elem->p_elem)->x_max;
          b_tube = ((MAXAMP *)ap_elem->p_elem)->y_max;
        }
        if (a_tube > 0) {
          aperture = effectiveEllipticalAperture(a_tube, b_tube, centroid, other_centroid);
          aperture_set = 1;
          tube_set = 1;
        }
      }
      break;
    case T_RCOL:
      if (plane) {
        if (((RCOL *)ap_elem->p_elem)->y_max > 0) {
          aperture = ((RCOL *)ap_elem->p_elem)->y_max;
          offset = ((RCOL *)ap_elem->p_elem)->dy;
          aperture_set = 1;
        }
      } else {
        if (((RCOL *)ap_elem->p_elem)->x_max > 0) {
          aperture = ((RCOL *)ap_elem->p_elem)->x_max;
          offset = ((RCOL *)ap_elem->p_elem)->dx;
          aperture_set = 1;
        }
      }
      if (aperture_set)
        aperture = aperture - fabs(centroid - offset);
      break;
    case T_ECOL:
      if (plane) {
        if (((ECOL *)ap_elem->p_elem)->y_max > 0) {
          aperture = effectiveEllipticalAperture(((ECOL *)ap_elem->p_elem)->y_max,
                                                 ((ECOL *)ap_elem->p_elem)->x_max,
                                                 centroid - ((ECOL *)ap_elem->p_elem)->dy,
                                                 other_centroid - ((ECOL *)ap_elem->p_elem)->dx);
        }
      } else {
        if (((ECOL *)ap_elem->p_elem)->x_max > 0) {
          aperture = effectiveEllipticalAperture(((ECOL *)ap_elem->p_elem)->x_max,
                                                 ((ECOL *)ap_elem->p_elem)->y_max,
                                                 centroid - ((ECOL *)ap_elem->p_elem)->dx,
                                                 other_centroid - ((ECOL *)ap_elem->p_elem)->dy);
        }
      }
      if (aperture > 0)
        aperture_set = 1;
      break;
    case T_SCRAPER:
      scraper = (SCRAPER *)ap_elem->p_elem;
      aperture_set = determineScraperAperture(plane, scraper->direction,
                                              scraper->position, plane ? scraper->dy : scraper->dx, centroid,
                                              &aperture, 1);
      break;
    case T_SPEEDBUMP:
      speedbump = (SPEEDBUMP *)ap_elem->p_elem;
      aperture_set = determineScraperAperture(plane, speedbump->direction,
                                              speedbump->position, plane ? speedbump->dy : speedbump->dx, centroid,
                                              &aperture, speedbump->scraperConvention);
      break;
    case T_TAPERAPC:
      taperApC = ((TAPERAPC *)ap_elem->p_elem);
      aperture_set = 1;
      apIndex = ap_elem == elem ? taperApC->e2Index : taperApC->e1Index;
      if (plane)
        aperture = effectiveEllipticalAperture(taperApC->r[apIndex],
                                               taperApC->r[apIndex],
                                               centroid - taperApC->dy,
                                               other_centroid - taperApC->dx);
      else
        aperture = effectiveEllipticalAperture(taperApC->r[apIndex],
                                               taperApC->r[apIndex],
                                               centroid - taperApC->dx,
                                               other_centroid - taperApC->dy);
      if (taperApC->sticky) {
        tube_set = elliptical_tube = 1;
        a_tube = b_tube = taperApC->r[apIndex];
      }
      break;
    case T_TAPERAPE:
      taperApE = ((TAPERAPE *)ap_elem->p_elem);
      aperture_set = 1;
      apIndex = ap_elem == elem ? taperApE->e2Index : taperApE->e1Index;
      if (plane)
        aperture = effectiveEllipticalAperture(taperApE->b[apIndex],
                                               taperApE->a[apIndex],
                                               centroid - taperApE->dy,
                                               other_centroid - taperApE->dx);
      else
        aperture = effectiveEllipticalAperture(taperApE->a[apIndex],
                                               taperApE->b[apIndex],
                                               centroid - taperApE->dx,
                                               other_centroid - taperApE->dy);
      if (taperApE->sticky) {
        tube_set = elliptical_tube = 1;
        if (plane) {
          a_tube = taperApE->b[apIndex];
          b_tube = taperApE->a[apIndex];
        } else {
          a_tube = taperApE->a[apIndex];
          b_tube = taperApE->b[apIndex];
        }
      }
      break;
    case T_TAPERAPR:
      taperApR = ((TAPERAPR *)ap_elem->p_elem);
      apIndex = ap_elem == elem ? taperApR->e2Index : taperApR->e1Index;
      aperture_set = 0;
      if (plane) {
        if (taperApR->ymax[apIndex] > 0) {
          aperture = taperApR->ymax[apIndex];
          offset = taperApR->dy;
          aperture_set = 1;
        }
      } else {
        if (taperApR->xmax[apIndex] > 0) {
          aperture = taperApR->xmax[apIndex];
          offset = taperApR->dx;
          aperture_set = 1;
        }
      }
      if (aperture_set)
        aperture = aperture - fabs(centroid - offset);
      if (taperApR->sticky) {
        elliptical_tube = 0;
        tube_aperture = aperture;
        tube_set = 1;
      }
      break;
    default:
      break;
    }
    if (run->apertureData.initialized) {
      /* Find aperture from the aperture data file */
      double dx, dy, xsize, ysize;
      if (interpolateApertureData(elem->end_pos, &(run->apertureData), &dx, &dy, &xsize, &ysize)) {
        centroid = *(((double *)elem->twiss) + (plane ? 1 : 0) + TWISS_CENT_OFFSET);
        if (plane) {
          aperture1 = ysize;
          offset = dy;
        } else {
          aperture1 = xsize;
          offset = dx;
        }
        aperture1 = aperture1 - fabs(centroid - offset);
        if (!aperture_set || aperture1 < aperture) {
          aperture = aperture1;
          aperture_set = 1;
        }
      }
    }
    if (tube_set) {
      aperture1 = 0;
      if (elliptical_tube) {
        if (a_tube > 0)
          aperture1 = effectiveEllipticalAperture(a_tube, b_tube, centroid, other_centroid);
      } else if (tube_aperture > 0)
        aperture1 = tube_aperture - fabs(centroid);
      if (aperture1 < 0)
        aperture1 = 0;
      if (!aperture_set || aperture1 < aperture)
        aperture = aperture1;
      aperture_set = 1;
    }

    if (!aperture_set)
      aperture = coordLimit;
    if (aperture > 0)
      tmp = sqr(aperture) / beta;
    else
      tmp = 0;
    if (tmp < acceptance || !acceptance_set) {
      *name = elem->name;
      *z = elem->end_pos;
      acceptance = tmp;
      acceptance_set = 1;
    }

    if (plane)
      elem->twiss->apy = aperture;
    else
      elem->twiss->apx = aperture;

    elem = elem->succ;
  }
  log_exit((char *)"find_acceptance");
  return (acceptance);
}

long has_aperture(ELEMENT_LIST *elem) {
  if (!elem)
    return (0);
  switch (elem->type) {
  case T_MAXAMP:
  case T_RCOL:
  case T_ECOL:
  case T_SCRAPER:
  case T_TAPERAPC:
  case T_TAPERAPR:
  case T_TAPERAPE:
    return (1);
  default:
    return (0);
  }
}

void modify_rfca_matrices(ELEMENT_LIST *eptr, long order)
/* Replace the matrices for rf cavities with drift matrices.
   Used prior to periodic twiss parameter computations. */
{
  while (eptr) {
    if (eptr->type == T_RFCA || eptr->type == T_MODRF || eptr->type == T_RFCW || eptr->type == T_RFDF) {
      eptr->savedMatrix = eptr->matrix;
      eptr->matrix = NULL;
      switch (eptr->type) {
      case T_RFCA:
        if (((RFCA *)eptr->p_elem)->change_p0)
          bombElegant((char *)"can't treat cavities like drifts when CHANGE_P0=1", NULL);
        eptr->matrix = drift_matrix(((RFCA *)eptr->p_elem)->length, order);
        break;
      case T_RFCW:
        if (((RFCW *)eptr->p_elem)->change_p0)
          bombElegant((char *)"can't treat cavities like drifts when CHANGE_P0=1", NULL);
        eptr->matrix = drift_matrix(((RFCW *)eptr->p_elem)->length, order);
        break;
      case T_MODRF:
        eptr->matrix = drift_matrix(((MODRF *)eptr->p_elem)->length, order);
        break;
      case T_RFDF:
        eptr->matrix = drift_matrix(((RFDF *)eptr->p_elem)->length, order);
        break;
      }
    }
    eptr = eptr->succ;
  }
  mustResetRfcaMatrices = 1;
}

void reset_rfca_matrices(ELEMENT_LIST *eptr, long order)
/* Delete the rf cavity matrices so they'll get updated */
{
  while (eptr) {
    if (eptr->type == T_RFCA || eptr->type == T_MODRF || eptr->type == T_RFCW || eptr->type == T_RFDF) {
      if (eptr->matrix) {
        free_matrices(eptr->matrix);
        tfree(eptr->matrix);
        eptr->matrix = NULL;
      }
      if (eptr->savedMatrix) {
        eptr->matrix = eptr->savedMatrix;
        eptr->savedMatrix = NULL;
      }
    }
    eptr = eptr->succ;
  }
}

#define ASSIGN_MINMAX(min, max, value) ((min > value ? min = value : 1), (max < value ? max = value : 1))

void compute_twiss_statistics(LINE_LIST *beamline, TWISS *twiss_ave, TWISS *twiss_min, TWISS *twiss_max) {
  ELEMENT_LIST *eptr;
  double dz, end_pos = sStart;
  long nElems;

  if (!twiss_ave) {
    printf((char *)"error: NULL twiss_ave pointer in compute_twiss_statistics\n");
    fflush(stdout);
    abort();
  }
  if (!twiss_min) {
    printf((char *)"error: NULL twiss_min pointer in compute_twiss_statistics\n");
    fflush(stdout);
    abort();
  }
  if (!twiss_max) {
    printf((char *)"error: NULL twiss_max pointer in compute_twiss_statistics\n");
    fflush(stdout);
    abort();
  }

  zero_memory(twiss_ave, sizeof(*twiss_ave));
  twiss_min->betax = twiss_min->alphax = twiss_min->etax = twiss_min->etapx = DBL_MAX;
  twiss_min->betay = twiss_min->alphay = twiss_min->etay = twiss_min->etapy = DBL_MAX;
  twiss_max->betax = twiss_max->alphax = twiss_max->etax = twiss_max->etapx = -DBL_MAX;
  twiss_max->betay = twiss_max->alphay = twiss_max->etay = twiss_max->etapy = -DBL_MAX;

  eptr = beamline->elem_twiss;
  nElems = 0;
  while (eptr) {
    ASSIGN_MINMAX(twiss_min->betax, twiss_max->betax, eptr->twiss->betax);
    ASSIGN_MINMAX(twiss_min->alphax, twiss_max->alphax, eptr->twiss->alphax);
    ASSIGN_MINMAX(twiss_min->etax, twiss_max->etax, eptr->twiss->etax);
    ASSIGN_MINMAX(twiss_min->etapx, twiss_max->etapx, eptr->twiss->etapx);
    ASSIGN_MINMAX(twiss_min->betay, twiss_max->betay, eptr->twiss->betay);
    ASSIGN_MINMAX(twiss_min->alphay, twiss_max->alphay, eptr->twiss->alphay);
    ASSIGN_MINMAX(twiss_min->etay, twiss_max->etay, eptr->twiss->etay);
    ASSIGN_MINMAX(twiss_min->etapy, twiss_max->etapy, eptr->twiss->etapy);
    if (eptr->pred && eptr->pred->twiss && eptr->twiss) {
      dz = (eptr->end_pos - eptr->pred->end_pos) / 2;
      twiss_ave->betax += (eptr->pred->twiss->betax + eptr->twiss->betax) * dz;
      twiss_ave->alphax += (eptr->pred->twiss->alphax + eptr->twiss->alphax) * dz;
      twiss_ave->etax += (eptr->pred->twiss->etax + eptr->twiss->etax) * dz;
      twiss_ave->etapx += (eptr->pred->twiss->etapx + eptr->twiss->etapx) * dz;
      twiss_ave->betay += (eptr->pred->twiss->betay + eptr->twiss->betay) * dz;
      twiss_ave->alphay += (eptr->pred->twiss->alphay + eptr->twiss->alphay) * dz;
      twiss_ave->etay += (eptr->pred->twiss->etay + eptr->twiss->etay) * dz;
      twiss_ave->etapy += (eptr->pred->twiss->etapy + eptr->twiss->etapy) * dz;
      nElems++;
    }
    end_pos = eptr->end_pos;
    eptr = eptr->succ;
  }
  if (nElems == 0) {
    twiss_ave->betax = (twiss_min->betax + twiss_max->betax) / 2;
    twiss_ave->alphax = (twiss_min->alphax + twiss_max->alphax) / 2;
    twiss_ave->etax = (twiss_min->etax + twiss_max->etax) / 2;
    twiss_ave->etapx = (twiss_min->etapx + twiss_max->etapx) / 2;
    twiss_ave->betay = (twiss_min->betay + twiss_max->betay) / 2;
    twiss_ave->alphay = (twiss_min->alphay + twiss_max->alphay) / 2;
    twiss_ave->etay = (twiss_min->etay + twiss_max->etay) / 2;
    twiss_ave->etapy = (twiss_min->etapy + twiss_max->etapy) / 2;
  } else if (end_pos) {
    twiss_ave->betax /= end_pos;
    twiss_ave->alphax /= end_pos;
    twiss_ave->etax /= end_pos;
    twiss_ave->etapx /= end_pos;
    twiss_ave->betay /= end_pos;
    twiss_ave->alphay /= end_pos;
    twiss_ave->etay /= end_pos;
    twiss_ave->etapy /= end_pos;
  }
}

void compute_twiss_percentiles(LINE_LIST *beamline, TWISS *twiss_p99, TWISS *twiss_p98, TWISS *twiss_p96) {
  ELEMENT_LIST *eptr;
  long iElem, i;
  double **data;
  TWISS *twiss_pXX[3];
  double percent[3] = {99, 98, 96};
  double value[3];

  if (!twiss_p99) {
    printf((char *)"error: NULL twiss_p99 pointer in compute_twiss_percentiles\n");
    fflush(stdout);
    abort();
  }
  if (!twiss_p98) {
    printf((char *)"error: NULL twiss_min pointer in compute_twiss_percentiles\n");
    fflush(stdout);
    abort();
  }
  if (!twiss_p96) {
    printf((char *)"error: NULL twiss_max pointer in compute_twiss_percentiles\n");
    fflush(stdout);
    abort();
  }

  data = (double **)czarray_2d(sizeof(double), 8, beamline->n_elems);
  eptr = beamline->elem_twiss;
  iElem = 0;
  while (eptr) {
    if (iElem >= beamline->n_elems)
      bombElegant("element counting error (compute_twiss_percentiles)", NULL);
    data[0][iElem] = eptr->twiss->betax;
    data[1][iElem] = eptr->twiss->alphax;
    data[2][iElem] = eptr->twiss->etax;
    data[3][iElem] = eptr->twiss->etapx;
    data[4][iElem] = eptr->twiss->betay;
    data[5][iElem] = eptr->twiss->alphay;
    data[6][iElem] = eptr->twiss->etay;
    data[7][iElem] = eptr->twiss->etapy;
    iElem++;
    eptr = eptr->succ;
  }

  twiss_pXX[0] = twiss_p99;
  twiss_pXX[1] = twiss_p98;
  twiss_pXX[2] = twiss_p96;

  compute_percentiles(value, percent, 3, data[0], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->betax = value[i];
#ifdef DEBUG
    printf("betax %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[1], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->alphax = value[i];
#ifdef DEBUG
    printf("alphax %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[2], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->etax = value[i];
#ifdef DEBUG
    printf("etax %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[3], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->etapx = value[i];
#ifdef DEBUG
    printf("etaxp %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[4], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->betay = value[i];
#ifdef DEBUG
    printf("betay %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[5], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->alphay = value[i];
#ifdef DEBUG
    printf("alphay %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[6], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->etay = value[i];
#ifdef DEBUG
    printf("etay %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  compute_percentiles(value, percent, 3, data[7], beamline->n_elems);
  for (i = 0; i < 3; i++) {
    twiss_pXX[i]->etapy = value[i];
#ifdef DEBUG
    printf("etayp %.0f %% = %g\n", percent[i], value[i]);
#endif
  }

  free_czarray_2d((void **)data, 8, beamline->n_elems);
}

void incrementRadIntegrals(RADIATION_INTEGRALS *radIntegrals, double *dI, ELEMENT_LIST *elem,
                           double beta0, double alpha0, double gamma0, double eta0, double etap0,
                           double betay0, double alphay0, double gammay0, double etay0, double etapy0,
                           double *coord, double pCentral) {
  /* compute contribution to radiation integrals */
  long isBend;
  BEND *bptr;
  KSBEND *kbptr;
  CSBEND *cbptr;
  CSRCSBEND *csrbptr;
  QUAD *qptr;
  KQUAD *qptrk;
  SEXT *sptr;
  KSEXT *sptrk;
  double length = 0.0, angle = 0.0, E1 = 0.0, E2 = 0.0, K1 = 0.0;
  double k2, rho, k, kl, kick = 0.0;
  double I1, I2, I3, I4, I5;
  double alpha1, gamma1, etap1, eta2, sin_kl, cos_kl;
  double etaAve = 0, etaK1_rhoAve = 0, HAve = 0, h, K2 = 0.0, dx = 0.0;
  static double coord0[6] = {0,0,0,0,0,0};

  I1 = I2 = I3 = I4 = I5 = 0;
  if (!coord)
    coord = &coord0[0];

#ifdef DEBUG
  fprintf(stderr, (char *)"incrementRadIntegrals: %s at %e: beta=%e, alpha=%e, eta=%e, etap=%e\n",
          elem->name, elem->end_pos, beta0, alpha0, eta0, etap0);
#endif
  if (elem->type == T_WIGGLER) {
    WIGGLER *wiggler;
    wiggler = (WIGGLER *)(elem->p_elem);
    AddWigglerRadiationIntegrals(wiggler->length, wiggler->poles, wiggler->radiusInternal,
                                 eta0, etap0,
                                 beta0, alpha0,
                                 &I1, &I2, &I3, &I4, &I5);
    radIntegrals->RI[0] += I1 * n_periods;
    radIntegrals->RI[1] += I2 * n_periods;
    radIntegrals->RI[2] += I3 * n_periods;
    radIntegrals->RI[3] += I4 * n_periods;
    radIntegrals->RI[4] += I5 * n_periods;
    /* Sokolov-Ternov: a symmetric wiggler contributes equally to + and -
     * half-periods so net polarizing contribution is zero, but it still
     * contributes to the spin-flip rate I3_tot. */
    radIntegrals->I3pos += 0.5 * I3 * n_periods;
    radIntegrals->I3neg += 0.5 * I3 * n_periods;
  } else if (elem->type == T_CWIGGLER) {
    CWIGGLER *wiggler;
    wiggler = (CWIGGLER *)(elem->p_elem);
    if (wiggler->BPeak[1]) {
      /* By */
      AddWigglerRadiationIntegrals(wiggler->length, wiggler->periods * 2,
                                   wiggler->radiusInternal[1],
                                   eta0, etap0, beta0, alpha0,
                                   &I1, &I2, &I3, &I4, &I5);
      radIntegrals->RI[0] += I1 * n_periods;
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->RI[3] += I4 * n_periods;
      radIntegrals->RI[4] += I5 * n_periods;
      radIntegrals->I3pos += 0.5 * I3 * n_periods;
      radIntegrals->I3neg += 0.5 * I3 * n_periods;
    }
    if (wiggler->BPeak[0]) {
      /* Bx */
      I1 = I2 = I3 = I4 = I5 = 0;
      AddWigglerRadiationIntegrals(wiggler->length, wiggler->periods * 2,
                                   wiggler->radiusInternal[0],
                                   etay0, etapy0, betay0, alphay0,
                                   &I1, &I2, &I3, &I4, &I5);
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->I3pos += 0.5 * I3 * n_periods;
      radIntegrals->I3neg += 0.5 * I3 * n_periods;
    }
  } else if (elem->type == T_APPLE) {
    APPLE *wiggler;
    wiggler = (APPLE *)(elem->p_elem);
    if (wiggler->BPeak[1]) {
      /* By */
      AddWigglerRadiationIntegrals(wiggler->length, wiggler->periods * 2,
                                   wiggler->radiusInternal[1],
                                   eta0, etap0, beta0, alpha0,
                                   &I1, &I2, &I3, &I4, &I5);
      radIntegrals->RI[0] += I1 * n_periods;
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->RI[3] += I4 * n_periods;
      radIntegrals->RI[4] += I5 * n_periods;
      radIntegrals->I3pos += 0.5 * I3 * n_periods;
      radIntegrals->I3neg += 0.5 * I3 * n_periods;
    }
    if (wiggler->BPeak[0]) {
      /* Bx */
      I1 = I2 = I3 = I4 = I5 = 0;
      AddWigglerRadiationIntegrals(wiggler->length, wiggler->periods * 2,
                                   wiggler->radiusInternal[0],
                                   etay0, etapy0, betay0, alphay0,
                                   &I1, &I2, &I3, &I4, &I5);
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->I3pos += 0.5 * I3 * n_periods;
      radIntegrals->I3neg += 0.5 * I3 * n_periods;
    }
  } else if (elem->type == T_UKICKMAP) {
    UKICKMAP *ukmap;
    ukmap = (UKICKMAP *)(elem->p_elem);
    if (ukmap->radiusInternal > 0) {
      AddWigglerRadiationIntegrals(ukmap->length, 2 * ukmap->periods, ukmap->radiusInternal,
                                   eta0, etap0,
                                   beta0, alpha0,
                                   &I1, &I2, &I3, &I4, &I5);
      radIntegrals->RI[0] += I1 * n_periods;
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->RI[3] += I4 * n_periods;
      radIntegrals->RI[4] += I5 * n_periods;
      radIntegrals->I3pos += 0.5 * I3 * n_periods;
      radIntegrals->I3neg += 0.5 * I3 * n_periods;
    }
  } else if (elem->type == T_CCBEND) {
    double ccbendAngle = ((CCBEND *)elem->p_elem)->angle;
    addCcbendRadiationIntegrals((CCBEND *)elem->p_elem, coord, pCentral, eta0, etap0, beta0, alpha0, &I1, &I2, &I3, &I4, &I5, elem);
    radIntegrals->RI[0] += I1;
    radIntegrals->RI[1] += I2;
    radIntegrals->RI[2] += I3;
    radIntegrals->RI[3] += I4;
    radIntegrals->RI[4] += I5;
    if (ccbendAngle > 0)
      radIntegrals->I3pos += I3;
    else if (ccbendAngle < 0)
      radIntegrals->I3neg += I3;
  } else if (elem->type == T_LGBEND) {
    LGBEND *lgbend;
    lgbend = (LGBEND *)elem->p_elem;
    if (lgbend->optimized != 1 && lgbend->optimizeFse) {
      double **oneParticle;
      oneParticle = (double **)czarray_2d(sizeof(**oneParticle), 1, totalPropertiesPerParticle);
      track_through_lgbend(oneParticle, 1, elem, lgbend, pCentral, NULL, 0, NULL, NULL, NULL, NULL, NULL, -1, -1);
    }
    addLgbendRadiationIntegrals((LGBEND *)elem->p_elem, coord, pCentral, eta0, etap0, beta0, alpha0, &I1, &I2, &I3, &I4, &I5, elem);
    radIntegrals->RI[0] += I1;
    radIntegrals->RI[1] += I2;
    radIntegrals->RI[2] += I3;
    radIntegrals->RI[3] += I4;
    radIntegrals->RI[4] += I5;
    {
      double lgbendAngle = ((LGBEND *)elem->p_elem)->angle;
      if (lgbendAngle > 0)
        radIntegrals->I3pos += I3;
      else if (lgbendAngle < 0)
        radIntegrals->I3neg += I3;
    }
  } else {
    MULT *mult = NULL;
    double KnMult = 0;
    isBend = 1;
    if (elem->type == T_MULT) {
      double H = 0;
      H = pCentral * me_mks * c_mks / e_mks;
      mult = (MULT *)elem->p_elem; /* only used if type==T_MULT */
      if ((length = mult->length) < 1e-6)
        length = 1e-6;
      if (mult->bore)
        KnMult = dfactorial(mult->order) * mult->BTipL / (H * ipow(mult->bore, mult->order) * length);
      else
        KnMult = mult->KnL / length;
    }
    if (elem->type == T_QUAD || elem->type == T_KQUAD || (elem->type == T_MULT && mult->order == 1)) {
      if (!coord && !coord[0]) {
        isBend = 0;
      } else {
        switch (elem->type) {
        case T_QUAD:
          qptr = (QUAD *)(elem->p_elem);
          length = qptr->length;
          kick = qptr->xkick * qptr->xKickCalibration;
          K1 = qptr->k1;
          dx = qptr->dx;
          break;
        case T_KQUAD:
          qptrk = (KQUAD *)(elem->p_elem);
          length = qptrk->length;
          kick = qptrk->xkick * qptrk->xKickCalibration;
          K1 = qptrk->k1;
          dx = qptrk->dx;
          break;
        case T_MULT:
          K1 = KnMult;
          dx = mult->dx;
          break;
        }
        if (!(h = K1 * (coord[0] - dx))) {
          isBend = 0;
        } else {
          /* The kick is subtracted because a positive angle bends toward the
           * right (negative x)
           */
          angle = length * h - kick;
          E1 = E2 = 0;
          isBend = 1;
        }
      }
    } else if (elem->type == T_SEXT || elem->type == T_KSEXT || (elem->type == T_MULT && mult->order == 2)) {
      if (!coord && !coord[0]) {
        isBend = 0;
      } else {
        switch (elem->type) {
        case T_SEXT:
          sptr = (SEXT *)(elem->p_elem);
          length = sptr->length;
          K2 = sptr->k2;
          dx = sptr->dx;
          break;
        case T_KSEXT:
          sptrk = (KSEXT *)(elem->p_elem);
          length = sptrk->length;
          K2 = sptrk->k2;
          dx = sptrk->dx;
          break;
        case T_MULT:
          K2 = KnMult;
          dx = mult->dx;
          break;
        }
        if (!(h = K2 * sqr(coord[0] - dx) / 2)) {
          isBend = 0;
        } else {
          K1 = K2 * (coord[0] - dx);
          angle = length * h;
          E1 = E2 = 0;
          isBend = 1;
        }
      }
    } else if (elem->type == T_SBEN || elem->type == T_RBEN) {
      bptr = (BEND *)(elem->p_elem);
      length = bptr->length;
      angle = bptr->angle;
      E1 = bptr->e[bptr->e1Index] * (bptr->edgeFlags & BEND_EDGE1_EFFECTS ? 1 : 0);
      E2 = bptr->e[bptr->e2Index] * (bptr->edgeFlags & BEND_EDGE2_EFFECTS ? 1 : 0);
      K1 = bptr->k1;
      K1 /= 1 + coord[5];
      if (angle)
	angle *= (1 + K1*length/angle*(coord[0]-bptr->dx))/(1 + coord[5]);
    } else if (elem->type == T_KSBEND) {
      kbptr = (KSBEND *)(elem->p_elem);
      length = kbptr->length;
      angle = kbptr->angle;
      E1 = kbptr->e[kbptr->e1Index] * (kbptr->flags & BEND_EDGE1_EFFECTS ? 1 : 0);
      E2 = kbptr->e[kbptr->e2Index] * (kbptr->flags & BEND_EDGE2_EFFECTS ? 1 : 0);
      K1 = kbptr->k1;
      K1 /= 1 + coord[5];
      if (angle)
	angle *= (1 + K1*length/angle*(coord[0]-kbptr->dx))/(1 + coord[5]);
    } else if (elem->type == T_CSBEND) {
      cbptr = (CSBEND *)(elem->p_elem);
      length = cbptr->length;
      angle = cbptr->angle;
      E1 = cbptr->e[cbptr->e1Index] * (cbptr->edgeFlags & BEND_EDGE1_EFFECTS ? 1 : 0);
      E2 = cbptr->e[cbptr->e2Index] * (cbptr->edgeFlags & BEND_EDGE2_EFFECTS ? 1 : 0);
      K1 = cbptr->k1;
      K1 /= 1 + coord[5];
      if (angle)
	angle *= (1 + K1*length/angle*(coord[0]-cbptr->dx))/(1 + coord[5]);
    } else if (elem->type == T_CSRCSBEND) {
      csrbptr = (CSRCSBEND *)(elem->p_elem);
      length = csrbptr->length;
      angle = csrbptr->angle;
      E1 = csrbptr->e[csrbptr->e1Index] * (csrbptr->edgeFlags & BEND_EDGE1_EFFECTS ? 1 : 0);
      E2 = csrbptr->e[csrbptr->e2Index] * (csrbptr->edgeFlags & BEND_EDGE2_EFFECTS ? 1 : 0);
      K1 = csrbptr->k1;
      K1 /= 1 + coord[5];
      if (angle)
	angle *= (1 + K1*length/angle*(coord[0]-csrbptr->dx))/(1 + coord[5]);
    } else {
      isBend = 0;
    }

    if (isBend && angle != 0) {
      double tanE1, tanE2, h2, l2, l3, l4;
      rho = length / angle;
      h = 1. / rho;
      h2 = h * h;
      l2 = length * length;
      l3 = l2 * length;
      l4 = l3 * length;
      tanE1 = tan(E1);
      tanE2 = tan(E2);

      k2 = K1 + 1. / (rho * rho);
      /* equations are from SLAC 1193 */
      k = sqrt(fabs(k2));
      kl = k * length;
      if (k2 < 0) {
        cos_kl = cosh(kl);
        sin_kl = sinh(kl);
      } else {
        sin_kl = sin(kl);
        cos_kl = cos(kl);
      }
      etap1 = etap0 + eta0 / rho * tanE1;
      eta2 = eta0 * cos_kl + etap1 * sin_kl / k + (1 - cos_kl) / (rho * k2);
      alpha1 = alpha0 - beta0 / rho * tanE1;
      gamma1 = (1 + sqr(alpha1)) / beta0;
      if (kl > 1e-2) {
        etaAve = eta0 * sin_kl / kl + etap1 * (1 - cos_kl) / (k2 * length) +
          (kl - sin_kl) / (k2 * kl * rho);
        etaK1_rhoAve = -etaAve * K1 / rho + (eta0 * tanE1 + eta2 * tanE2) / (2 * length) * h2;
        HAve = gamma1 * sqr(eta0) + 2 * alpha1 * eta0 * etap1 + beta0 * sqr(etap1) + 2 * angle * (-(gamma1 * eta0 + alpha1 * etap1) * (kl - sin_kl) / (k * k2 * l2) + (alpha1 * eta0 + beta0 * etap1) * (1 - cos_kl) / (k2 * l2)) + sqr(angle) * (gamma1 * (3 * kl - 4 * sin_kl + sin_kl * cos_kl) / (2 * k * k2 * k2 * l3) - alpha1 * sqr(1 - cos_kl) / (k2 * k2 * l3) + beta0 * (kl - cos_kl * sin_kl) / (2 * kl * k2 * l2));
      }
      if (kl <= 1e-2 || HAve < 0) {
        double kl2, kl4, kl6, kl8;
        double T0, T2, T4, T6, T8;
        kl2 = kl * kl;
        kl4 = kl2 * kl2;
        kl6 = kl4 * kl2;
        kl8 = kl6 * kl2;

        T0 = eta0 + (length * (3 * etap1 + h * length)) / 6.;
        T2 = (-20 * eta0 - length * (5 * etap1 + h * length)) / 120.;
        T4 = (42 * eta0 + length * (7 * etap1 + h * length)) / 5040.;
        T6 = (-72 * eta0 - length * (9 * etap1 + h * length)) / 362880.;
        T8 = 2.505210838544172e-8 * (110. * eta0 + length * (11. * etap1 + h * length));
        etaAve = T0 + T2 * kl2 + T4 * kl4 + T6 * kl6 + T8 * kl8;
        etaK1_rhoAve = -etaAve * K1 / rho + (eta0 * tanE1 + eta2 * tanE2) / (2 * length) * h2;

        T0 = (ipow2(eta0) * gamma1 -
              (eta0 * gamma1 * h * l2) / 3. +
              (gamma1 * h2 * l4) / 20. +
              beta0 * (ipow2(etap1) + etap1 * h * length +
                       (h2 * l2) / 3.) +
              alpha1 * (eta0 * (2 * etap1 + h * length) -
                        (h * l2 * (4 * etap1 + 3 * h * length)) / 12.));
        T2 = -(h * l2 *
               (70 * beta0 * etap1 - 14 * eta0 * gamma1 * length + 56 * beta0 * h * length +
                5 * gamma1 * h * l3 +
                7 * alpha1 * (10 * eta0 - length * (2 * etap1 + 5 * h * length)))) /
          (840. * length);
        T4 = (h * l2 *
              (-8 * eta0 * gamma1 * length + 7 * gamma1 * h * l3 +
               8 * beta0 * (7 * etap1 + 16 * h * length) +
               alpha1 * (56 * eta0 - length * (8 * etap1 + 63 * h * length)))) /
          (20160. * length);
        T6 = (-2.505210838544172e-7 * h * l2 *
              (-22. * eta0 * gamma1 * length + 51. * gamma1 * h * l3 +
               22. * beta0 * (9. * etap1 + 64. * h * length) +
               11. * alpha1 * (18. * eta0 - 1. * length * (2. * etap1 + 51. * h * length)))) /
          length;
        T8 = (9.635426302092969e-10 * h * l2 *
              (-52. * eta0 * gamma1 * length + 341. * gamma1 * h * l3 +
               52. * beta0 * (11. * etap1 + 256. * h * length) +
               13. * alpha1 * (44. * eta0 - 1. * length * (4. * etap1 + 341. * h * length)))) /
          length;
        HAve = T0 + T2 * kl2 + T4 * kl4 + T6 * kl6 + T8 * kl8;
      }
      I1 = etaAve * length / rho;
      I2 = length * h2;
      I3 = I2 / fabs(rho);
      I4 = I2 / rho * etaAve - 2 * length * etaK1_rhoAve;
      I5 = HAve * I3;
      radIntegrals->RI[0] += I1 * n_periods;
      radIntegrals->RI[1] += I2 * n_periods;
      radIntegrals->RI[2] += I3 * n_periods;
      radIntegrals->RI[3] += I4 * n_periods;
      radIntegrals->RI[4] += I5 * n_periods;
      /* Sokolov-Ternov: bin I3 by sign of bend angle.  Quadrupole/sextupole
       * "bends" from off-axis trajectories (T_QUAD, T_SEXT, T_MULT, etc.)
       * also enter this branch via the synthesized angle; binning by their
       * effective angle sign is the right treatment. */
      if (angle > 0)
        radIntegrals->I3pos += I3 * n_periods;
      else if (angle < 0)
        radIntegrals->I3neg += I3 * n_periods;
    }
  }
  if (dI) {
    dI[0] = I1 * n_periods;
    dI[1] = I2 * n_periods;
    dI[2] = I3 * n_periods;
    dI[3] = I4 * n_periods;
    dI[4] = I5 * n_periods;
  }
}

void completeRadiationIntegralComputation(RADIATION_INTEGRALS *RI, LINE_LIST *beamline, double Po, double *coord) {
  double Rce, gamma;

  if (coord)
    Po *= 1 + coord[5];
  RI->Pref = Po;
  gamma = sqrt(sqr(Po) + 1);
  Rce = sqr(particleCharge) / (1e7 * particleMass);
  RI->Uo = particleMassMV * Rce * RI->RI[1] * 2. / 3. * ipow4(gamma);
  RI->Jx = 1 - RI->RI[3] / RI->RI[1];
  RI->Jdelta = 3 - RI->Jx;
  RI->Jy = 1;
  RI->tauy = 1. / (Rce / 3 * ipow3(gamma) * c_mks / (beamline->revolution_length * n_periods) * RI->RI[1]);
  RI->taux = RI->tauy * RI->Jy / RI->Jx;
  RI->taudelta = RI->tauy * RI->Jy / RI->Jdelta;
  RI->sigmadelta = gamma * sqrt(55. / 32. / sqrt(3.) * hbar_mks / (particleMass * c_mks) * RI->RI[2] / (2 * RI->RI[1] + RI->RI[3]));
  RI->ex0 = sqr(gamma) * 55. / 32. / sqrt(3.) * hbar_mks / (particleMass * c_mks) * RI->RI[4] / (RI->RI[1] - RI->RI[3]);

  /* Sokolov-Ternov self-polarization for a planar ring with sign-correct
   * Derbenev-Kondratenko reduction of the third radiation integral:
   *   1/tau_p = (5*sqrt(3)/8) * r_e * hbar * gamma^5 * I3 / (m_e * C)
   *   P_eq    = (8/(5*sqrt(3))) * (I3pos - I3neg) / (I3pos + I3neg)
   *
   * The polarization rate is from Sokolov and Ternov, generalized to an
   * arbitrary lattice by replacing 1/|rho|^3 with its ring average I3/C:
   *   A.A. Sokolov and I.M. Ternov, Sov. Phys. Dokl. 8, 1203 (1964);
   *   V.N. Baier, "Radiative polarization of electrons in storage
   *     rings," Sov. Phys. Usp. 14, 695 (1972), Eq. (3.2).
   * The (I3pos - I3neg)/(I3pos + I3neg) form for P_eq is the planar-
   * ring reduction of the full Derbenev-Kondratenko formula:
   *   Ya.S. Derbenev and A.M. Kondratenko, Sov. Phys. JETP 37, 968 (1973).
   * Both formulas are collected in modern form in the review
   *   S.R. Mane, Yu.M. Shatunov, K. Yokoya, "Spin-polarized charged
   *     particle beams in high-energy accelerators," Rep. Prog. Phys.
   *     68, 1997 (2005), Section 4.
   *
   * The DK split reduces to the pure 8/(5*sqrt(3)) limit for a single-
   * direction planar ring.  Wigglers/antibends suppress P_eq. */
  {
    double C = beamline->revolution_length * n_periods;
    double i3 = RI->RI[2];
    double i3sum = RI->I3pos + RI->I3neg;
    RI->PstLimit = 8.0 / (5.0 * sqrt(3.0));
    if (i3 > 0 && C > 0)
      RI->tauP = (8.0 / (5.0 * sqrt(3.0))) * particleMass * C
                 / (particleRadius * hbar_mks * ipow5(gamma) * i3);
    else
      RI->tauP = 0;
    if (i3sum > 0)
      RI->Peq = RI->PstLimit * (RI->I3pos - RI->I3neg) / i3sum;
    else
      RI->Peq = 0;
  }

  RI->computed = 1;
}

void LoadStartingTwissFromFile(double *betax, double *betay, double *alphax, double *alphay,
                               double *etax, double *etaxp, double *etay, double *etayp,
                               char *filename_inner_scope, char *elementName, long elementOccurrence) {
  SDDS_DATASET SDDSin;
  long rows = 0, rowOfInterest;
  double *betaxData, *betayData = NULL, *alphaxData = NULL, *alphayData = NULL;
  double *etaxData = NULL, *etayData = NULL, *etaxpData = NULL, *etaypData = NULL;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename_inner_scope) ||
      SDDS_ReadPage(&SDDSin) != 1)
    SDDS_Bomb((char *)"problem reading Twiss reference file");
  if (SDDS_CheckColumn(&SDDSin, (char *)"betax", (char *)"m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"betay", (char *)"m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"alphax", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"alphay", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"etax", (char *)"m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"etay", (char *)"m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"etaxp", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"etayp", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"ElementName", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK)
    SDDS_Bomb((char *)"invalid/missing columns in Twiss reference file");
  if (elementName) {
    if (!SDDS_SetRowFlags(&SDDSin, 1) ||
        (rows = SDDS_MatchRowsOfInterest(&SDDSin, (char *)"ElementName", elementName, SDDS_AND)) <= 0)
      SDDS_Bomb((char *)"Problem finding data for beta function reference.  Check for existence of element.");
    if (elementOccurrence > 0 && elementOccurrence > rows)
      SDDS_Bomb((char *)"Too few occurrences of reference element in beta function reference file.");
  }
  if ((rows = SDDS_CountRowsOfInterest(&SDDSin)) < 1)
    SDDS_Bomb((char *)"No data in beta function reference file.");

  if (!(betaxData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"betax")) ||
      !(betayData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"betay")) ||
      !(alphaxData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"alphax")) ||
      !(alphayData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"alphay")) ||
      !(etaxData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"etax")) ||
      !(etayData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"etay")) ||
      !(etaxpData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"etaxp")) ||
      !(etaypData = SDDS_GetColumnInDoubles(&SDDSin, (char *)"etayp")))
    SDDS_Bomb((char *)"Problem getting data for beta function reference.");
  if (elementName && elementOccurrence > 0)
    rowOfInterest = elementOccurrence - 1;
  else
    rowOfInterest = rows - 1;
  *betax = betaxData[rowOfInterest];
  *betay = betayData[rowOfInterest];
  *alphax = alphaxData[rowOfInterest];
  *alphay = alphayData[rowOfInterest];
  *etax = etaxData[rowOfInterest];
  *etay = etayData[rowOfInterest];
  *etaxp = etaxpData[rowOfInterest];
  *etayp = etaypData[rowOfInterest];
  free(betaxData);
  free(betayData);
  free(alphaxData);
  free(alphayData);
  free(etaxData);
  free(etayData);
  free(etaxpData);
  free(etaypData);
}

void setupTuneShiftWithAmplitude(NAMELIST_TEXT *nltext, RUN *run) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&tune_shift_with_amplitude, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &tune_shift_with_amplitude);

  if (tune_shift_with_amplitude_struct.turns < 100 && tune_shift_with_amplitude_struct.turns != 0)
    bombElegant((char *)"too few turns requested (tune_shift_with_amplitude)", NULL);
  if (!tune_shift_with_amplitude_struct.turns && tune_shift_with_amplitude_struct.spread_only)
    bombElegant((char *)"spread_only requires turns!=0", NULL);
  if (tune_shift_with_amplitude_struct.turns) {
    if (tune_shift_with_amplitude_struct.x0 <= 0 ||
        tune_shift_with_amplitude_struct.y0 <= 0)
      bombElegant((char *)"x0 or y0 is zero or negative (tune_shift_with_amplitude)", NULL);
    if (!tune_shift_with_amplitude_struct.spread_only &&
        (tune_shift_with_amplitude_struct.x1 < tune_shift_with_amplitude_struct.x0 * 10 ||
         tune_shift_with_amplitude_struct.y1 < tune_shift_with_amplitude_struct.y0 * 10))
      bombElegant((char *)"x1 or y1 is too small (tune_shift_with_amplitude)", NULL);
  }
  doTuneShiftWithAmplitude = 1;
  if (tune_shift_with_amplitude_struct.tune_output) {
    char *filename_inner_scope;
    filename_inner_scope = compose_filename(tune_shift_with_amplitude_struct.tune_output, run->rootname);
    if (!SDDS_InitializeOutputElegant(&SDDSTswaTunes, SDDS_BINARY, 0, NULL, NULL,
                                      filename_inner_scope) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"x", (char *)"m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"y", (char *)"m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"Ax", (char *)"m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"Ay", (char *)"m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"nux", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"nuy", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"nuxError", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSTswaTunes, (char *)"nuyError", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSTswaTunes, (char *)"nuxRmsResidual", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSTswaTunes, (char *)"nuyRmsResidual", NULL, SDDS_DOUBLE) ||
        !SDDS_WriteLayout(&SDDSTswaTunes)) {
      printf((char *)"Unable set up file %s\n",
             tune_shift_with_amplitude_struct.tune_output);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (filename_inner_scope != tune_shift_with_amplitude_struct.tune_output)
      free(filename_inner_scope);
  }
}

double QElement(double ****Q, long i1, long i2, long i3, long i4) {
  if (i3 < i4)
    SWAP_LONG(i3, i4);
  if (i2 < i3)
    SWAP_LONG(i2, i3);
  if (i3 < i4)
    SWAP_LONG(i3, i4);
  if (i2 < i3 || i3 < i4)
    bombElegant((char *)"it didn't work", NULL);
  return Q[i1][i2][i3][i4];
}

void computeTuneShiftWithAmplitude(double dnux_dA[N_TSWA][N_TSWA], double dnuy_dA[N_TSWA][N_TSWA],
                                   double *xTuneExtrema, double *yTuneExtrema,
                                   TWISS *twiss, double *tune, VMATRIX *M, LINE_LIST *beamline,
                                   RUN *run, double *startingCoord, long nPeriods) {
  double tune1[2];
  double *Ax, *Ay;
  double *x0, *y0;
  double **xTune, **yTune;
  short **lost;
  double tuneLowerLimit[2], tuneUpperLimit[2];
  double result, maxResult;
  double x, y;
  long ix, iy, tries, gridSize, nLost = 0;
  double upperLimit[2], lowerLimit[2];
  /*  MATRIX *AxAy, *Coef, *Nu, *AxAyTr, *Mf, *MfInv, *AxAyTrNu; */
  long j, ix1, iy1;
  // long m;

  tune1[0] = tune1[1] = -1; /* suppress a compiler warning */

  if (tune_shift_with_amplitude_struct.turns == 0) {
    /* use the matrix only without tracking */
    xTuneExtrema[0] = xTuneExtrema[1] = 0;
    yTuneExtrema[0] = yTuneExtrema[1] = 0;
    computeTuneShiftWithAmplitudeM(dnux_dA, dnuy_dA, twiss, tune, M, nPeriods);
    return;
  }

  gridSize = tune_shift_with_amplitude_struct.grid_size;

  Ax = (double *)tmalloc(sizeof(*Ax) * gridSize);
  Ay = (double *)tmalloc(sizeof(*Ax) * gridSize);
  x0 = (double *)tmalloc(sizeof(*x0) * gridSize);
  y0 = (double *)tmalloc(sizeof(*y0) * gridSize);
  xTune = (double **)czarray_2d(sizeof(**xTune), gridSize, gridSize);
  yTune = (double **)czarray_2d(sizeof(**yTune), gridSize, gridSize);
  lost = (short **)czarray_2d(sizeof(**lost), gridSize, gridSize);

  if (tune_shift_with_amplitude_struct.tune_output &&
      !SDDS_StartPage(&SDDSTswaTunes, gridSize * gridSize)) {
    printf((char *)"Problem starting SDDS page for TSWA tune output\n");
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }

  /* use tracking and NAFF */
  tries = tune_shift_with_amplitude_struct.scaling_iterations;
  upperLimit[0] = upperLimit[1] = 1;
  lowerLimit[0] = lowerLimit[1] = 0;
  while (tries--) {
    // m = 0; /* number of tune points */
    tuneLowerLimit[0] = tuneLowerLimit[1] = 0;
    tuneUpperLimit[0] = tuneUpperLimit[1] = 0;
    nLost = 0;
    for (ix = 0; ix < gridSize; ix++) {
      x0[ix] =
        x = sqrt(ix *
                 sqr((tune_shift_with_amplitude_struct.x1 - tune_shift_with_amplitude_struct.x0)) / (gridSize - 1) +
                 sqr(tune_shift_with_amplitude_struct.x0));
      Ax[ix] = sqr(x) * (1 + sqr(twiss->alphax)) / twiss->betax;
      for (iy = 0; iy < gridSize; iy++) {
        lost[ix][iy] = 0;
        xTune[ix][iy] = -1;
        yTune[ix][iy] = -1;
        if ((tune_shift_with_amplitude_struct.sparse_grid &&
             !(ix == 0 || iy == 0 || ix == iy)) ||
            (tune_shift_with_amplitude_struct.lines_only &&
             !(ix == 0 || iy == 0)))
          continue;
        // m++;
        y0[iy] =
          y = sqrt(iy *
                   sqr((tune_shift_with_amplitude_struct.y1 - tune_shift_with_amplitude_struct.y0)) / (gridSize - 1) +
                   sqr(tune_shift_with_amplitude_struct.y0));
        Ay[iy] = sqr(y) * (1 + sqr(twiss->alphay)) / twiss->betay;
        if (ix == 0 && iy == 0) {
          tuneLowerLimit[0] = tuneLowerLimit[1] = 0;
          tuneUpperLimit[0] = tuneUpperLimit[1] = 0;
        } else {
          if ((ix1 = ix - 1) < 0)
            ix1 = 0;
          if ((iy1 = iy - 1) < 0)
            iy1 = 0;
          tuneLowerLimit[0] = xTune[ix1][iy1] - tune_shift_with_amplitude_struct.nux_roi_width / 2;
          tuneUpperLimit[0] = xTune[ix1][iy1] + tune_shift_with_amplitude_struct.nux_roi_width / 2;
          tuneLowerLimit[1] = yTune[ix1][iy1] - tune_shift_with_amplitude_struct.nux_roi_width / 2;
          tuneUpperLimit[1] = yTune[ix1][iy1] + tune_shift_with_amplitude_struct.nux_roi_width / 2;
        }
        if (!computeTunesFromTracking(tune1, NULL, M, beamline, run, startingCoord,
                                      x, y, 0,
                                      tune_shift_with_amplitude_struct.turns, 0, NULL,
                                      tuneLowerLimit, tuneUpperLimit, 0, nPeriods, CTFT_INCLUDE_X | CTFT_INCLUDE_Y)) {
          lost[ix][iy] = 1;
          nLost++;
        }
        xTune[ix][iy] = tune1[0];
        yTune[ix][iy] = tune1[1];
        if (tune_shift_with_amplitude_struct.verbose)
          printf((char *)"Tunes for TSWA: x=%e, y=%e, nux=%.15e, nuy=%.15e\n",
                 x, y, tune1[0], tune1[1]);
      }
    }
    if (tune_shift_with_amplitude_struct.verbose)
      printf((char *)"All tunes computed for TSWA (%ld particles lost)\n", nLost);

    maxResult = -DBL_MAX;
    if (!tune_shift_with_amplitude_struct.spread_only) {
      if (nLost) {
        printf((char *)"Amplitude too large for TSWA computation---particles lost\n");
      } else {
        for (ix = 0; ix < gridSize; ix++) {
          for (iy = 0; iy < gridSize; iy++) {
            if (ix == 0 && iy == 0)
              continue;
            if ((tune_shift_with_amplitude_struct.sparse_grid &&
                 !(ix == 0 || iy == 0 || ix == iy)) ||
                (tune_shift_with_amplitude_struct.lines_only &&
                 !(ix == 0 || iy == 0)))
              continue;
            result = fabs(xTune[ix][iy] - xTune[0][0]);
            if (result > maxResult)
              maxResult = result;
            result = fabs(yTune[ix][iy] - yTune[0][0]);
            if (result > maxResult)
              maxResult = result;
          }
        }
        if (tune_shift_with_amplitude_struct.verbose)
          printf((char *)"maximum tune change: %e\n", maxResult);
      }
      if (nLost || maxResult > tune_shift_with_amplitude_struct.scale_down_limit) {
        char warningBuffer[1024];
        if (upperLimit[0] > tune_shift_with_amplitude_struct.x1)
          upperLimit[0] = tune_shift_with_amplitude_struct.x1;
        if (upperLimit[1] > tune_shift_with_amplitude_struct.y1)
          upperLimit[1] = tune_shift_with_amplitude_struct.y1;
        tune_shift_with_amplitude_struct.x1 = (upperLimit[0] + 2 * lowerLimit[0]) / 3;
        tune_shift_with_amplitude_struct.y1 = (upperLimit[1] + 2 * lowerLimit[1]) / 3;
        snprintf(warningBuffer, 1024,
                 "Setting tune_shift_with_amplitude_struct.x1=%le and tune_shift_with_amplitude_struct.y1=%le.",
                 tune_shift_with_amplitude_struct.x1, tune_shift_with_amplitude_struct.y1);
        printWarning((char *)"tune_shift_with_amplitude: the amplitude specified for calculations too large.",
                     warningBuffer);
        if (tries == 0)
          tries = 1; /* ensures we don't exit on amplitude too large */
        continue;
      }
    }
    break;
  }

  if (tune_shift_with_amplitude_struct.tune_output) {
    if (!SDDS_SetParameters(&SDDSTswaTunes, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "nuxRmsResidual", -1.0,
                            "nuyRmsResidual", -1.0,
                            NULL)) {
      printf((char *)"Problem filling SDDS page for TSWA tune output\n");
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    for (ix = j = 0; ix < gridSize; ix++) {
      for (iy = 0; iy < gridSize; iy++) {
        if ((tune_shift_with_amplitude_struct.sparse_grid &&
             !(ix == 0 || iy == 0 || ix == iy)) ||
            (tune_shift_with_amplitude_struct.lines_only &&
             !(ix == 0 || iy == 0)))
          continue;
        if (!SDDS_SetRowValues(&SDDSTswaTunes, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, j,
                               (char *)"x", x0[ix], (char *)"Ax", Ax[ix],
                               (char *)"nux", xTune[ix][iy],
                               (char *)"y", y0[iy], (char *)"Ay", Ay[iy],
                               (char *)"nuy", yTune[ix][iy],
                               (char *)"nuxError", 0.0,
                               (char *)"nuyError", 0.0,
                               NULL)) {
          printf((char *)"Problem filling SDDS page for TSWA tune output\n");
          SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        j++;
      }
    }
  }

  if (nLost)
    for (ix = 0; ix < N_TSWA; ix++)
      for (iy = 0; iy < N_TSWA; iy++)
        dnux_dA[ix][iy] = dnuy_dA[ix][iy] = sqrt(DBL_MAX);
  else
    for (ix = 0; ix < N_TSWA; ix++)
      for (iy = 0; iy < N_TSWA; iy++)
        dnux_dA[ix][iy] = dnuy_dA[ix][iy] = 0;

  xTuneExtrema[0] = -(xTuneExtrema[1] = -DBL_MAX);
  yTuneExtrema[0] = -(yTuneExtrema[1] = -DBL_MAX);
  for (ix = 0; ix < gridSize; ix++) {
    for (iy = 0; iy < gridSize; iy++) {
      if ((tune_shift_with_amplitude_struct.sparse_grid &&
           !(ix == 0 || iy == 0 || ix == iy)) ||
          (tune_shift_with_amplitude_struct.lines_only &&
           !(ix == 0 || iy == 0)))
        continue;
      if (lost[ix][iy] ||
          xTune[ix][iy] <= 0.0 || xTune[ix][iy] >= 1.0 ||
          yTune[ix][iy] <= 0.0 || yTune[ix][iy] >= 1.0)
        continue;
      if (xTuneExtrema[0] > xTune[ix][iy])
        xTuneExtrema[0] = xTune[ix][iy];
      if (xTuneExtrema[1] < xTune[ix][iy])
        xTuneExtrema[1] = xTune[ix][iy];
      if (yTuneExtrema[0] > yTune[ix][iy])
        yTuneExtrema[0] = yTune[ix][iy];
      if (yTuneExtrema[1] < yTune[ix][iy])
        yTuneExtrema[1] = yTune[ix][iy];
    }
  }
  if (nLost && !tune_shift_with_amplitude_struct.exclude_lost_particles) {
    printf((char *)"%ld lost particles in tune tracking: setting spreads to 1\n", nLost);
    xTuneExtrema[0] = yTuneExtrema[0] = 0;
    xTuneExtrema[1] = yTuneExtrema[1] = 1.0;
  }

  if (tune_shift_with_amplitude_struct.verbose) {
    printf((char *)"xTune extrema: %21.15e, %21.15e, delta = %21.15e\n",
           xTuneExtrema[0], xTuneExtrema[1],
           fabs(xTuneExtrema[0] - xTuneExtrema[1]));
    printf((char *)"yTune extrema: %21.15e, %21.15e, delta = %21.15e\n",
           yTuneExtrema[0], yTuneExtrema[1],
           fabs(yTuneExtrema[0] - yTuneExtrema[1]));
  }

  if (!nLost && !tune_shift_with_amplitude_struct.spread_only) {
    long terms, iterm, points, maxOrder;
    int32_t *order[2];
    double *Axy[2], *nux, *nuy, *nuxDiff, *nuyDiff, *nuxCoef, *nuyCoef, nuxChiSqr, nuyChiSqr, nuxCondition, nuyCondition;

    if (tune_shift_with_amplitude_struct.order < 1)
      tune_shift_with_amplitude_struct.order = 1;
    maxOrder = tune_shift_with_amplitude_struct.order;
    if (tune_shift_with_amplitude_struct.lines_only) {
      /* Perform 1-D fits vs Ax and Ay.
       * No cross terms get included.
       */
      terms = 2 * maxOrder + 1;
      order[0] = (int32_t *)tmalloc(sizeof(*order[0]) * terms);
      order[1] = (int32_t *)tmalloc(sizeof(*order[0]) * terms);
      for (ix = iterm = 0; ix <= maxOrder; ix++, iterm++) {
        order[0][iterm] = ix;
        order[1][iterm] = 0;
      }
      for (iy = 1; iy <= maxOrder; iy++, iterm++) {
        if (iterm >= terms)
          bombElegant("term counting issues in TSWA computation", NULL);
        order[0][iterm] = 0;
        order[1][iterm] = iy;
      }
      points = 2 * gridSize - 1;
      nux = (double *)tmalloc(sizeof(*nux) * points);
      nuy = (double *)tmalloc(sizeof(*nuy) * points);
      Axy[0] = (double *)tmalloc(sizeof(*Axy[0]) * points);
      Axy[1] = (double *)tmalloc(sizeof(*Axy[1]) * points);
      for (ix = iterm = 0; ix < gridSize; ix++, iterm++) {
        if (iterm >= points)
          bombElegant("point counting issues in TSWA computation", NULL);
        nux[iterm] = xTune[ix][0];
        nuy[iterm] = yTune[ix][0];
        Axy[0][iterm] = Ax[ix];
        Axy[1][iterm] = Ay[0];
      }
      for (iy = 1; iy < gridSize; iy++, iterm++) {
        if (iterm >= points)
          bombElegant("point counting issues in TSWA computation", NULL);
        nux[iterm] = xTune[0][iy];
        nuy[iterm] = yTune[0][iy];
        Axy[0][iterm] = Ax[0];
        Axy[1][iterm] = Ay[iy];
      }
    } else {
      /* Perform 2-D fits vs Ax and Ay.
       */
      terms = (maxOrder + 1) * (maxOrder + 2) / 2;
      order[0] = (int32_t *)tmalloc(sizeof(*order[0]) * terms);
      order[1] = (int32_t *)tmalloc(sizeof(*order[1]) * terms);
      for (ix = iterm = 0; ix <= maxOrder; ix++) {
        for (iy = 0; (ix + iy) <= maxOrder; iy++, iterm++) {
          order[0][iterm] = ix;
          order[1][iterm] = iy;
        }
      }
      if (tune_shift_with_amplitude_struct.sparse_grid) {
        points = 3 * gridSize - 2;
        nux = (double *)tmalloc(sizeof(*nux) * points);
        nuy = (double *)tmalloc(sizeof(*nuy) * points);
        Axy[0] = (double *)tmalloc(sizeof(*Axy[0]) * points);
        Axy[1] = (double *)tmalloc(sizeof(*Axy[1]) * points);
        for (ix = iterm = 0; ix < gridSize; ix++) {
          for (iy = 0; iy < gridSize; iy++) {
            if (!(ix == 0 || iy == 0 || ix == iy))
              continue;
            if (iterm >= points)
              bombElegant("point counting issues in TSWA computation", NULL);
            nux[iterm] = xTune[ix][iy];
            nuy[iterm] = yTune[ix][iy];
            Axy[0][iterm] = Ax[ix];
            Axy[1][iterm] = Ay[iy];
            iterm++;
          }
        }
        if (iterm != points)
          bombElegant("point counting error in TSWA computation", NULL);
      } else {
        points = gridSize * gridSize;
        nux = (double *)tmalloc(sizeof(*nux) * points);
        nuy = (double *)tmalloc(sizeof(*nuy) * points);
        Axy[0] = (double *)tmalloc(sizeof(*Axy[0]) * points);
        Axy[1] = (double *)tmalloc(sizeof(*Axy[1]) * points);
        for (ix = iterm = 0; ix < gridSize; ix++) {
          for (iy = 0; iy < gridSize; iy++, iterm++) {
            if (iterm >= points)
              bombElegant("point counting issues in TSWA computation", NULL);
            nux[iterm] = xTune[ix][iy];
            nuy[iterm] = yTune[ix][iy];
            Axy[0][iterm] = Ax[ix];
            Axy[1][iterm] = Ay[iy];
          }
        }
      }
    }
    nuxDiff = (double *)tmalloc(sizeof(*nuxDiff) * points);
    nuyDiff = (double *)tmalloc(sizeof(*nuyDiff) * points);
    nuxCoef = (double *)tmalloc(sizeof(*nuxCoef) * terms);
    nuyCoef = (double *)tmalloc(sizeof(*nuyCoef) * terms);
    printf("Fitting for TSWA with %ld terms and %ld points:\n", terms, points);
    lsf2dPolyUnweighted(Axy, nux, points, order, terms, nuxCoef, &nuxChiSqr, &nuxCondition, nuxDiff);
    lsf2dPolyUnweighted(Axy, nuy, points, order, terms, nuyCoef, &nuyChiSqr, &nuyCondition, nuyDiff);
    printf("TSWA fit residuals: nux = %le, nuy = %le\n", sqrt(nuxChiSqr * (points - terms) / (1.0 * points)),
           sqrt(nuyChiSqr * (points - terms) / (1.0 * points)));

    for (iterm = 0; iterm < terms; iterm++) {
      ix = order[0][iterm];
      iy = order[1][iterm];
      if (ix < 0 || iy < 0)
        bombElegantVA((char *)"order invalid (%ld, %ld) after fitting for TSWA", ix, iy);
      if (ix >= N_TSWA || iy >= N_TSWA)
        continue;
      dnux_dA[ix][iy] = nuxCoef[iterm] * factorial(ix) * factorial(iy);
      dnuy_dA[ix][iy] = nuyCoef[iterm] * factorial(ix) * factorial(iy);
    }

    if (tune_shift_with_amplitude_struct.verbose) {
      printf((char *)"dnux/(dAx^i dAy^j):\n");
      for (ix = 0; ix <= maxOrder && ix < N_TSWA; ix++) {
        for (iy = 0; (ix + iy) <= maxOrder && (ix + iy) < N_TSWA; iy++)
          printf((char *)"%10.3e ", dnux_dA[ix][iy]);
        printf("\n");
      }
      printf((char *)"dnuy/(dAx^i dAy^j):\n");
      for (ix = 0; ix <= maxOrder && ix < N_TSWA; ix++) {
        for (iy = 0; (ix + iy) <= maxOrder && (ix + iy) < N_TSWA; iy++)
          printf((char *)"%10.3e ", dnuy_dA[ix][iy]);
        printf("\n");
      }
    }

    if (tune_shift_with_amplitude_struct.tune_output) {
      if (!SDDS_SetParameters(&SDDSTswaTunes, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "nuxRmsResidual", sqrt(nuxChiSqr * (points - terms) / (1.0 * points)),
                              "nuyRmsResidual", sqrt(nuyChiSqr * (points - terms) / (1.0 * points)),
                              NULL)) {
        printf((char *)"Problem filling SDDS page for TSWA tune output\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      for (ix = j = 0; ix < gridSize; ix++) {
        for (iy = 0; iy < gridSize; iy++) {
          if ((tune_shift_with_amplitude_struct.sparse_grid &&
               !(ix == 0 || iy == 0 || ix == iy)) ||
              (tune_shift_with_amplitude_struct.lines_only &&
               !(ix == 0 || iy == 0)))
            continue;
          if (!SDDS_SetRowValues(&SDDSTswaTunes, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, j,
                                 (char *)"nuxError", nuxDiff[j],
                                 (char *)"nuyError", nuyDiff[j],
                                 NULL)) {
            printf((char *)"Problem filling SDDS page for TSWA tune output\n");
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            exitElegant(1);
          }
          j++;
        }
      }
      if (!SDDS_WritePage(&SDDSTswaTunes)) {
        printf((char *)"Problem writing SDDS page for TSWA tune output\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    free(order[0]);
    free(order[1]);
    free(nux);
    free(nuy);
    free(nuxDiff);
    free(nuyDiff);
    free(nuxCoef);
    free(nuyCoef);
  } else {
    if (tune_shift_with_amplitude_struct.tune_output) {
      if (!SDDS_WritePage(&SDDSTswaTunes)) {
        printf((char *)"Problem writing SDDS page for TSWA tune output\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
  }
}

long computeTunesFromTracking(double *tune, double *amp, VMATRIX *M, LINE_LIST *beamline, RUN *run,
                              double *startingCoord,
                              double xAmplitude, double yAmplitude, double deltaOffset, long turns, long turnOffset,
                              double *endingCoord, double *tuneLowerLimit, double *tuneUpperLimit,
                              long allowLosses, long nPeriods,
                              unsigned long flags) {
  double **oneParticle, dummy;
  double frequency[4], amplitude[4], phase[4];
  double *x, *y, *xp, *yp, p;
  long i, one = 1;
  double CSave[6];

  /* A tracking loss returns before the final coordinates are copied below.
   * Define the output in that case so callers never read an uninitialized or
   * stale ending coordinate. */
  if (endingCoord)
    fill_double_array(endingCoord, 6, 0.0);

#ifdef DEBUG1
  static FILE *fpdeb = NULL;
  if (!fpdeb) {
    fpdeb = fopen((char *)"tuneTracking.sdds", (char *)"w");
    fprintf(fpdeb, (char *)"SDDS1\n");
    fprintf(fpdeb, (char *)"&column name=Pass type=long &end\n");
    fprintf(fpdeb, (char *)"&column name=x type=double &end\n");
    fprintf(fpdeb, (char *)"&column name=xp type=double &end\n");
    fprintf(fpdeb, (char *)"&column name=y type=double &end\n");
    fprintf(fpdeb, (char *)"&column name=yp type=double &end\n");
    fprintf(fpdeb, (char *)"&column name=t type=double &end\n");
    fprintf(fpdeb, (char *)"&column name=p type=double &end\n");
    fprintf(fpdeb, (char *)"&data mode=ascii &end\n");
  }
  fprintf(fpdeb, (char *)"%ld\n", turns);

#endif

#ifdef DEBUG
  printf((char *)"In computeTunesFromTracking: turns=%ld, xAmp=%le, yAmp=%le\n",
         turns, xAmplitude, yAmplitude);
  fflush(stdout);
#endif
  x = xp = y = yp = NULL;
  if (flags & CTFT_USE_MATRIX) {
    /* this is necessary because the concatenated matrix includes the closed orbit in
     * C.  We don't want to put this in at each turn.
     */
    for (i = 0; i < 6; i++) {
      CSave[i] = M->C[i];
      M->C[i] = 0;
    }
  }
  oneParticle = (double **)czarray_2d(sizeof(**oneParticle), 1, totalPropertiesPerParticle);
  if (!startingCoord)
    fill_double_array(oneParticle[0], totalPropertiesPerParticle, 0.0);
  else {
    memcpy(oneParticle[0], startingCoord, 6 * sizeof(**oneParticle));
    oneParticle[0][6] = 1;
  }
  if (!(flags & CTFT_INCLUDE_X))
    xAmplitude = 0;
  if (!(flags & CTFT_INCLUDE_Y))
    yAmplitude = 0;

  oneParticle[0][2] += yAmplitude;
  oneParticle[0][0] += xAmplitude;
  oneParticle[0][5] += deltaOffset;

#ifdef DEBUG
  printf((char *)"Starting coordinates: %21.15le, %21.15le, %21.15le, %21.15le, %21.15le, %21.15le\n",
         oneParticle[0][0], oneParticle[0][1],
         oneParticle[0][2], oneParticle[0][3],
         oneParticle[0][4], oneParticle[0][5]);
  fflush(stdout);
#endif

#ifdef DEBUG
  printf((char *)"Doing malloc: turns=%ld\n", turns);
  fflush(stdout);
#endif

  if (!(x = (double *)malloc(sizeof(*x) * turns)) || !(y = (double *)malloc(sizeof(*y) * turns)) ||
      !(xp = (double *)malloc(sizeof(*xp) * turns)) || !(yp = (double *)malloc(sizeof(*yp) * turns)))
    bombElegant((char *)"memory allocation failure (computeTunesFromTracking)", NULL);

#ifdef DEBUG
  printf((char *)"Did malloc\n");
  fflush(stdout);
#endif

  x[0] = oneParticle[0][0];
  xp[0] = oneParticle[0][1];
  y[0] = oneParticle[0][2];
  yp[0] = oneParticle[0][3];
  p = run->p_central;

#ifdef DEBUG
  printf((char *)"Starting to track\n");
  fflush(stdout);
#endif
#ifdef DEBUG1
  fprintf(fpdeb, (char *)"0 %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n",
          oneParticle[0][0], oneParticle[0][1], oneParticle[0][2], oneParticle[0][3], oneParticle[0][4], oneParticle[0][5]);
#endif

  for (i = 1; i < turns; i++) {
    if (flags & CTFT_USE_MATRIX) {
      long j;
      for (j = 0; j < nPeriods; j++)
        track_particles(oneParticle, M, oneParticle, one);
    } else {
      if (!do_tracking(NULL, oneParticle, 1, NULL, beamline, &p, (double **)NULL,
                       (BEAM_SUMS **)NULL, (long *)NULL,
                       (TRAJECTORY *)NULL, run, 0,
                       TEST_PARTICLES + (allowLosses ? TEST_PARTICLE_LOSSES : 0) + TIME_DEPENDENCE_OFF,
                       nPeriods, i - 1 + turnOffset, NULL, NULL, NULL, NULL, NULL)) {
        if (!allowLosses)
          printWarning((char *)"Test particle lost when computing tunes from tracking", NULL);
        return 0;
      }
    }
    if (isnan(oneParticle[0][0]) || isnan(oneParticle[0][1]) ||
        isnan(oneParticle[0][2]) || isnan(oneParticle[0][3]) ||
        isnan(oneParticle[0][4]) || isnan(oneParticle[0][5])) {
      if (!allowLosses)
        printWarning((char *)"Test particle lost when computing tunes from tracking", NULL);
      return 0;
    }
    x[i] = oneParticle[0][0];
    xp[i] = oneParticle[0][1];
    y[i] = oneParticle[0][2];
    yp[i] = oneParticle[0][3];
#ifdef DEBUG1
    fprintf(fpdeb, (char *)"%ld %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", i,
            oneParticle[0][0], oneParticle[0][1], oneParticle[0][2], oneParticle[0][3], oneParticle[0][4], oneParticle[0][5]);
#endif
  }
  if (endingCoord) {
    for (i = 0; i < 6; i++)
      endingCoord[i] = oneParticle[0][i];
  }
#ifdef DEBUG
  printf((char *)"Ending coordinates: %21.15le, %21.15le, %21.15le, %21.15le, %21.15le, %21.15le\n",
         oneParticle[0][0], oneParticle[0][1],
         oneParticle[0][2], oneParticle[0][3],
         oneParticle[0][4], oneParticle[0][5]);
  fflush(stdout);
#endif

#ifdef DEBUG
  printf((char *)"Performing NAFF (1)\n");
  fflush(stdout);
#endif
  if (flags & CTFT_INCLUDE_X &&
      PerformNAFF(&frequency[0], &amplitude[0], &phase[0],
                  &dummy, 0.0, 1.0, x, turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT | NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ? (tuneLowerLimit[0] > 0.5 ? 1 - tuneLowerLimit[0] : tuneLowerLimit[0]) : 0,
                  tuneUpperLimit ? (tuneUpperLimit[0] > 0.5 ? 1 - tuneUpperLimit[0] : tuneUpperLimit[0]) : 0) != 1) {
    printWarning((char *)"NAFF failed for tune analysis from tracking (x).\n", NULL);
    /*
      printf((char*)"Limits: %e, %e\n",
      tuneLowerLimit?(tuneLowerLimit[0]>0.5 ? 1-tuneLowerLimit[0] : tuneLowerLimit[0]):0,
      tuneUpperLimit?(tuneUpperLimit[0]>0.5 ? 1-tuneUpperLimit[0] : tuneUpperLimit[0]):0);
    */
    return 0;
  }
#ifdef DEBUG
  printf((char *)"Performing NAFF (2)\n");
  fflush(stdout);
#endif
  if (flags & CTFT_INCLUDE_X &&
      PerformNAFF(&frequency[1], &amplitude[1], &phase[1],
                  &dummy, 0.0, 1.0, xp, turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT | NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ? (tuneLowerLimit[0] > 0.5 ? 1 - tuneLowerLimit[0] : tuneLowerLimit[0]) : 0,
                  tuneUpperLimit ? (tuneUpperLimit[0] > 0.5 ? 1 - tuneUpperLimit[0] : tuneUpperLimit[0]) : 0) != 1) {
    printWarning((char *)"NAFF failed for tune analysis from tracking (xp).\n", NULL);
    return 0;
  }
#ifdef DEBUG
  printf((char *)"Performing NAFF (3)\n");
  fflush(stdout);
#endif
  if (flags & CTFT_INCLUDE_Y &&
      PerformNAFF(&frequency[2], &amplitude[2], &phase[2],
                  &dummy, 0.0, 1.0, y, turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT | NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ? (tuneLowerLimit[1] > 0.5 ? 1 - tuneLowerLimit[1] : tuneLowerLimit[1]) : 0,
                  tuneUpperLimit ? (tuneUpperLimit[1] > 0.5 ? 1 - tuneUpperLimit[1] : tuneUpperLimit[1]) : 0) != 1) {
    printWarning((char *)"NAFF failed for tune analysis from tracking (y).\n", NULL);
    return 0;
  }
#ifdef DEBUG
  printf((char *)"Performing NAFF (4)\n");
  fflush(stdout);
#endif
  if (flags & CTFT_INCLUDE_Y &&
      PerformNAFF(&frequency[3], &amplitude[3], &phase[3],
                  &dummy, 0.0, 1.0, yp, turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT | NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ? (tuneLowerLimit[1] > 0.5 ? 1 - tuneLowerLimit[1] : tuneLowerLimit[1]) : 0,
                  tuneUpperLimit ? (tuneUpperLimit[1] > 0.5 ? 1 - tuneUpperLimit[1] : tuneUpperLimit[1]) : 0) != 1) {
    printWarning((char *)"NAFF failed for tune analysis from tracking (yp).\n", NULL);
    return 0;
  }

#ifdef DEBUG
  printf((char *)"NAFF done\n");
  for (i = 0; i < 4; i++)
    printf((char *)"%ld: freq=%e, phase=%e, amp=%e\n",
           i, frequency[i], phase[i], amplitude[i]);
  fflush(stdout);
#endif

  if (flags & CTFT_INCLUDE_X)
    tune[0] = adjustTuneHalfPlane(frequency[0], phase[0], phase[1]);
  if (flags & CTFT_INCLUDE_Y)
    tune[1] = adjustTuneHalfPlane(frequency[2], phase[2], phase[3]);
  if (amp) {
    if (flags & CTFT_INCLUDE_X)
      amp[0] = amplitude[0];
    if (flags & CTFT_INCLUDE_Y)
      amp[1] = amplitude[2];
  }
#ifdef DEBUG
  printf((char *)"xtune = %e, ytune = %e\n", tune[0], tune[1]);
#endif

  free(x);
  free(y);
  free(xp);
  free(yp);
  free_czarray_2d((void **)oneParticle, 1, totalPropertiesPerParticle);
  if (flags & CTFT_USE_MATRIX) {
    M->C[i] = CSave[i];
  }
  return 1;
}

static long computeTunesFromHistory(double *tune, double *amp,
                                    const double *x, const double *xp,
                                    const double *y, const double *yp,
                                    long turns, double *tuneLowerLimit,
                                    double *tuneUpperLimit,
                                    unsigned long flags) {
  double frequency[4] = {0, 0, 0, 0};
  double amplitude[4] = {0, 0, 0, 0};
  double phase[4] = {0, 0, 0, 0};
  double dummy;

  if (flags & CTFT_INCLUDE_X &&
      PerformNAFF(&frequency[0], &amplitude[0], &phase[0],
                  &dummy, 0.0, 1.0, const_cast<double *>(x), turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT |
                    NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ?
                    (tuneLowerLimit[0] > 0.5 ?
                       1 - tuneLowerLimit[0] : tuneLowerLimit[0]) : 0,
                  tuneUpperLimit ?
                    (tuneUpperLimit[0] > 0.5 ?
                       1 - tuneUpperLimit[0] : tuneUpperLimit[0]) : 0) != 1)
    return 0;
  if (flags & CTFT_INCLUDE_X &&
      PerformNAFF(&frequency[1], &amplitude[1], &phase[1],
                  &dummy, 0.0, 1.0, const_cast<double *>(xp), turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT |
                    NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ?
                    (tuneLowerLimit[0] > 0.5 ?
                       1 - tuneLowerLimit[0] : tuneLowerLimit[0]) : 0,
                  tuneUpperLimit ?
                    (tuneUpperLimit[0] > 0.5 ?
                       1 - tuneUpperLimit[0] : tuneUpperLimit[0]) : 0) != 1)
    return 0;
  if (flags & CTFT_INCLUDE_Y &&
      PerformNAFF(&frequency[2], &amplitude[2], &phase[2],
                  &dummy, 0.0, 1.0, const_cast<double *>(y), turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT |
                    NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ?
                    (tuneLowerLimit[1] > 0.5 ?
                       1 - tuneLowerLimit[1] : tuneLowerLimit[1]) : 0,
                  tuneUpperLimit ?
                    (tuneUpperLimit[1] > 0.5 ?
                       1 - tuneUpperLimit[1] : tuneUpperLimit[1]) : 0) != 1)
    return 0;
  if (flags & CTFT_INCLUDE_Y &&
      PerformNAFF(&frequency[3], &amplitude[3], &phase[3],
                  &dummy, 0.0, 1.0, const_cast<double *>(yp), turns,
                  NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT |
                    NAFF_FREQ_ACCURACY_LIMIT,
                  0.0, 1, 200, 1e-12,
                  tuneLowerLimit ?
                    (tuneLowerLimit[1] > 0.5 ?
                       1 - tuneLowerLimit[1] : tuneLowerLimit[1]) : 0,
                  tuneUpperLimit ?
                    (tuneUpperLimit[1] > 0.5 ?
                       1 - tuneUpperLimit[1] : tuneUpperLimit[1]) : 0) != 1)
    return 0;

  if (flags & CTFT_INCLUDE_X)
    tune[0] = adjustTuneHalfPlane(frequency[0], phase[0], phase[1]);
  if (flags & CTFT_INCLUDE_Y)
    tune[1] = adjustTuneHalfPlane(frequency[2], phase[2], phase[3]);
  if (amp) {
    if (flags & CTFT_INCLUDE_X)
      amp[0] = amplitude[0];
    if (flags & CTFT_INCLUDE_Y)
      amp[1] = amplitude[2];
  }
  return 1;
}

long computeTunesFromTrackingBatch(double *tune, double *amp, VMATRIX *M,
                                   LINE_LIST *beamline, RUN *run,
                                   double **startingCoord,
                                   const double *xAmplitude,
                                   const double *yAmplitude,
                                   const double *deltaOffset,
                                   long particles, long turns,
                                   long turnOffset, double **endingCoord,
                                   long *valid, long *survived,
                                   double *tuneLowerLimit,
                                   double *tuneUpperLimit, long allowLosses,
                                   long nPeriods, unsigned long flags) {
  double **particle = NULL;
  double **trackingParticle = NULL;
  double *history = NULL, *x, *xp, *y, *yp;
  double CSave[6], p;
  unsigned char *seen = NULL;
  long *prepassSurvived = NULL;
  long i, ip, id, nActive, nValid;
  long fusedPrepass = 0;

  if (!tune || !beamline || !run || !valid || particles <= 0 || turns <= 1)
    return 0;
  particle = (double **)czarray_2d(sizeof(**particle), particles,
                                  totalPropertiesPerParticle);
  history = (double *)calloc(4 * particles * turns, sizeof(*history));
  seen = (unsigned char *)calloc(particles, sizeof(*seen));
  if (!particle || !history || !seen)
    bombElegant((char *)"memory allocation failure "
                "(computeTunesFromTrackingBatch)", NULL);
  x = history;
  xp = x + particles * turns;
  y = xp + particles * turns;
  yp = y + particles * turns;

  if (flags & CTFT_USE_MATRIX) {
    for (i = 0; i < 6; i++) {
      CSave[i] = M->C[i];
      M->C[i] = 0;
    }
  }

  for (ip = 0; ip < particles; ip++) {
    if (startingCoord && startingCoord[ip])
      memcpy(particle[ip], startingCoord[ip],
             6 * sizeof(**particle));
    particle[ip][particleIDIndex] = ip + 1;
    if (flags & CTFT_INCLUDE_X)
      particle[ip][0] += xAmplitude ? xAmplitude[ip] : 0;
    if (flags & CTFT_INCLUDE_Y)
      particle[ip][2] += yAmplitude ? yAmplitude[ip] : 0;
    particle[ip][5] += deltaOffset ? deltaOffset[ip] : 0;
    if (endingCoord && endingCoord[ip])
      memcpy(endingCoord[ip], particle[ip], 6 * sizeof(**endingCoord));
    valid[ip] = 1;
    if (survived)
      survived[ip] = 1;
    x[ip * turns] = particle[ip][0];
    xp[ip * turns] = particle[ip][1];
    y[ip * turns] = particle[ip][2];
    yp[ip * turns] = particle[ip][3];
  }

  p = run->p_central;
  nActive = particles;
#if defined(HAVE_GPU) && !USE_MPI
  if (!(flags & CTFT_USE_MATRIX) && allowLosses && nPeriods == 1) {
    long prepassCount = 0;

    prepassSurvived = (long *)calloc(
      (size_t)particles, sizeof(*prepassSurvived));
    if (!prepassSurvived)
      bombElegant((char *)"memory allocation failure for fused tune "
                  "prepass status", NULL);
    fusedPrepass = gpu_batched_tune_loss_prepass(
      beamline, p, particle[0], particles, totalPropertiesPerParticle,
      turns, turnOffset, prepassSurvived, &prepassCount);
    if (fusedPrepass) {
      trackingParticle = (double **)calloc(
        (size_t)(prepassCount > 0 ? prepassCount : 1),
        sizeof(*trackingParticle));
      if (!trackingParticle)
        bombElegant((char *)"memory allocation failure for fused tune "
                    "repair rows", NULL);
      for (ip = nActive = 0; ip < particles; ip++) {
        if (prepassSurvived[ip])
          trackingParticle[nActive++] = particle[ip];
        else {
          valid[ip] = 0;
          if (survived)
            survived[ip] = 0;
        }
      }
      if (nActive != prepassCount)
        bombElegant((char *)"inconsistent fused tune prepass survivor "
                    "count", NULL);
    }
  }
#endif
  if (!trackingParticle)
    trackingParticle = particle;
  for (i = 1; i < turns && nActive > 0; i++) {
    if (flags & CTFT_USE_MATRIX) {
      long period;
      for (period = 0; period < nPeriods; period++)
        track_particles(trackingParticle, M, trackingParticle, nActive);
    } else {
      nActive = do_tracking(
        NULL, trackingParticle, nActive, NULL, beamline, &p,
        (double **)NULL,
        (BEAM_SUMS **)NULL, (long *)NULL, (TRAJECTORY *)NULL, run, 0,
        TEST_PARTICLES + (allowLosses ? TEST_PARTICLE_LOSSES : 0) +
          TIME_DEPENDENCE_OFF,
        nPeriods, i - 1 + turnOffset, NULL, NULL, NULL, NULL, NULL);
    }
    memset(seen, 0, particles * sizeof(*seen));
    for (ip = 0; ip < nActive; ip++) {
      id = (long)trackingParticle[ip][particleIDIndex] - 1;
      if (id < 0 || id >= particles || seen[id])
        bombElegant((char *)"invalid or duplicate particle ID in batched "
                    "tune tracking", NULL);
      seen[id] = 1;
      x[id * turns + i] = trackingParticle[ip][0];
      xp[id * turns + i] = trackingParticle[ip][1];
      y[id * turns + i] = trackingParticle[ip][2];
      yp[id * turns + i] = trackingParticle[ip][3];
    }
    for (id = 0; id < particles; id++)
      if (valid[id] && !seen[id]) {
        valid[id] = 0;
        if (survived)
          survived[id] = 0;
      }
  }

  if (endingCoord) {
    for (ip = 0; ip < nActive; ip++) {
      id = (long)trackingParticle[ip][particleIDIndex] - 1;
      if (id >= 0 && id < particles && endingCoord[id])
        memcpy(endingCoord[id], trackingParticle[ip],
               6 * sizeof(**endingCoord));
    }
  }

  {
    std::atomic<long> nextParticle(0), validCount(0);
    std::vector<std::thread> workers;
    unsigned long workerCount = std::thread::hardware_concurrency();
    const char *threadSetting = getenv("ELEGANT_GPU_BATCHED_TUNE_THREADS");

    if (threadSetting && threadSetting[0]) {
      char *endptr = NULL;
      unsigned long requested = strtoul(threadSetting, &endptr, 10);
      if (endptr && !endptr[0] && requested)
        workerCount = requested;
    }
    if (!workerCount)
      workerCount = 1;
    if (workerCount > 16)
      workerCount = 16;
    if ((long)workerCount > particles)
      workerCount = particles;
    if (particles < 32)
      workerCount = 1;

    auto analyzeHistory = [&]() {
      long particleIndex;
      while ((particleIndex = nextParticle.fetch_add(1)) < particles) {
        if (!valid[particleIndex])
          continue;
        if (!computeTunesFromHistory(
              tune + 2 * particleIndex,
              amp ? amp + 2 * particleIndex : NULL,
              x + particleIndex * turns, xp + particleIndex * turns,
              y + particleIndex * turns, yp + particleIndex * turns, turns,
              tuneLowerLimit, tuneUpperLimit, flags)) {
          valid[particleIndex] = 0;
          continue;
        }
        validCount.fetch_add(1);
      }
    };

    for (unsigned long worker = 1; worker < workerCount; worker++)
      workers.emplace_back(analyzeHistory);
    analyzeHistory();
    for (auto &worker : workers)
      worker.join();
    nValid = validCount.load();
  }

  if (flags & CTFT_USE_MATRIX)
    for (i = 0; i < 6; i++)
      M->C[i] = CSave[i];
  if (fusedPrepass)
    free(trackingParticle);
  free(prepassSurvived);
  free(seen);
  free(history);
  free_czarray_2d((void **)particle, particles,
                  totalPropertiesPerParticle);
  return nValid;
}

double adjustTuneHalfPlane(double frequency, double phase0, double phase1) {
  if (fabs(phase0 - phase1) > PI) {
    if (phase0 < phase1)
      phase0 += PIx2;
    else
      phase1 += PIx2;
  }
  if (phase0 < phase1)
    return frequency;
  return 1 - frequency;
}

void computeTuneShiftWithAmplitudeM(double dnux_dA[N_TSWA][N_TSWA], double dnuy_dA[N_TSWA][N_TSWA],
                                    TWISS *twiss, double *tune, VMATRIX *M, long nPeriods) {
  VMATRIX M1, M2;
  long tplane, splane;
  long it, jt, is, js, ia;
  double alpha[2], beta[2], shift[2];
  double turns;
  double C, S, theta;

  initialize_matrices(&M1, 3);
  initialize_matrices(&M2, 3);
  copy_matrices(&M2, M);
  /* the matrix M is already formed for any closed orbit, so we don't want
   * to concatenate the closed orbit
   */
  for (it = 0; it < 6; it++)
    M2.C[it] = 0;
  concat_matrices(&M1, &M2, &M2, 0);
  turns = 2;

  while (turns < 32768) {
    copy_matrices(&M2, &M1);
    concat_matrices(&M1, &M2, &M2, 0);
    turns *= 2;
  }

  free_matrices(&M2);

  /*
    fprintf(stderr, (char*)"Tune check: \n%le %le\n%le %le\n",
    cos(PIx2*turns*tune[0]), (M1.R[0][0]+M1.R[1][1])/2,
    cos(PIx2*turns*tune[1]), (M1.R[2][2]+M1.R[3][3])/2);
  */

  alpha[0] = twiss->alphax;
  alpha[1] = twiss->alphay;
  beta[0] = twiss->betax;
  beta[1] = twiss->betay;

  for (it = 0; it < N_TSWA; it++)
    for (is = 0; is < N_TSWA; is++)
      dnux_dA[it][is] = dnuy_dA[it][is] = 0;
  dnux_dA[0][0] = tune[0];
  dnuy_dA[0][0] = tune[1];

  for (tplane = 0; tplane < 2; tplane++) {
    it = 2 * tplane;
    jt = it + 1;
    for (splane = 0; splane < 2; splane++) {
      is = 2 * splane;
      js = is + 1;
      shift[splane] = 0;
      /*
        fprintf(stderr, (char*)"tplane=%ld splane=%ld\n", tplane, splane);
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        it, it, is, is, QElement(M1.Q, it, it, is, is));
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        jt, jt, is, is, QElement(M1.Q, jt, jt, is, is));
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        it, it, is, js, QElement(M1.Q, it, it, is, js));
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        jt, jt, is, js, QElement(M1.Q, jt, jt, is, js));
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        it, it, js, js, QElement(M1.Q, it, it, js, js));
        fprintf(stderr, (char*)"Q%ld%ld%ld%ld = %e\n",
        jt, jt, js, js, QElement(M1.Q, jt, jt, js, js));
        fprintf(stderr, (char*)"\n");
      */
      for (ia = 0, theta = 0; ia < 72; ia++, theta += PIx2 / 72) {
        C = cos(theta);
        S = sin(theta) * alpha[splane] + C;
        shift[splane] +=
          (QElement(M1.Q, it, it, is, is) + QElement(M1.Q, jt, jt, is, is)) * beta[splane] * sqr(C) - (QElement(M1.Q, it, it, is, js) + QElement(M1.Q, jt, jt, is, js)) * C * S + (QElement(M1.Q, it, it, js, js) + QElement(M1.Q, jt, jt, js, js)) * sqr(S) / beta[splane];
      }
      shift[splane] /= 72;
    }
    if (tplane == 0) {
      /* nux */
      dnux_dA[1][0] = shift[0] / (-2 * PIx2 * sin(PIx2 * tune[0] * turns) * turns) / nPeriods;
      dnux_dA[0][1] = shift[1] / (-2 * PIx2 * sin(PIx2 * tune[0] * turns) * turns) / nPeriods;
    } else {
      /* nuy */
      dnuy_dA[1][0] = shift[0] / (-2 * PIx2 * sin(PIx2 * tune[1] * turns) * turns) / nPeriods;
      dnuy_dA[0][1] = shift[1] / (-2 * PIx2 * sin(PIx2 * tune[1] * turns) * turns) / nPeriods;
    }
  }

  free_matrices(&M1);
}

void store_fitpoint_twiss_parameters(MARK *fpt, char *name, long occurence, TWISS *twiss, RADIATION_INTEGRALS *radIntegrals) {
  long i, j;
  static char *twiss_name_suffix[14] = {
    (char *)"betax",
    (char *)"alphax",
    (char *)"nux",
    (char *)"etax",
    (char *)"etapx",
    (char *)"etaxp",
    (char *)"psix",
    (char *)"betay",
    (char *)"alphay",
    (char *)"nuy",
    (char *)"etay",
    (char *)"etapy",
    (char *)"etayp",
    (char *)"psiy",
  };
  static char s[200];
  if (!(fpt->init_flags & 1)) {
    fpt->twiss_mem = (long *)tmalloc(sizeof(*(fpt->twiss_mem)) * 20);
    fpt->init_flags |= 1;
    for (i = 0; i < 14; i++) {
      snprintf(s, 200, (char *)"%s#%ld.%s", name, occurence, twiss_name_suffix[i]);
      fpt->twiss_mem[i] = rpn_create_mem(s, 0);
    }
    for (j = 0; j < 6; j++, i++) {
      snprintf(s, 200, (char *)"%s#%ld.I%ld", name, occurence, j + 1);
      fpt->twiss_mem[i] = rpn_create_mem(s, 0);
    }
  }
  if (!twiss) {
    printf((char *)"twiss parameter pointer unexpectedly NULL\n");
    fflush(stdout);
    abort();
  }
  for (i = 0; i < 5; i++) {
    if (i == 2) {
      /* psi/(2*pi) */
      rpn_store(*((&twiss->betax) + i) / PIx2, NULL, fpt->twiss_mem[2]);
      rpn_store(*((&twiss->betay) + i) / PIx2, NULL, fpt->twiss_mem[9]);
      /* psi */
      rpn_store(*((&twiss->betax) + i), NULL, fpt->twiss_mem[6]);
      rpn_store(*((&twiss->betay) + i), NULL, fpt->twiss_mem[13]);
    } else if (i == 4) {
      /* store etaxp and etayp in under two names each: etapx and etaxp */
      rpn_store(*((&twiss->betax) + i), NULL, fpt->twiss_mem[i]);
      rpn_store(*((&twiss->betax) + i), NULL, fpt->twiss_mem[i + 1]);
      rpn_store(*((&twiss->betay) + i), NULL, fpt->twiss_mem[i + 7]);
      rpn_store(*((&twiss->betay) + i), NULL, fpt->twiss_mem[i + 7]);
    } else {
      rpn_store(*((&twiss->betax) + i), NULL, fpt->twiss_mem[i]);
      rpn_store(*((&twiss->betay) + i), NULL, fpt->twiss_mem[i + 7]);
    }
  }
  if (radIntegrals) {
    i = 14;
    for (j = 0; j < 6; j++, i++)
      rpn_store(radIntegrals->RI[j], NULL, fpt->twiss_mem[i]);
  }
}

void clearTwissAnalysisRequests() {
  long i;
  for (i = 0; i < twissAnalysisRequests; i++) {
    if (twissAnalysisRequest[i].startName)
      free(twissAnalysisRequest[i].startName);
    if (twissAnalysisRequest[i].endName)
      free(twissAnalysisRequest[i].endName);
    if (twissAnalysisRequest[i].matchName)
      free(twissAnalysisRequest[i].matchName);
    if (twissAnalysisRequest[i].tag)
      free(twissAnalysisRequest[i].tag);
  }
  free(twissAnalysisRequest);
  twissAnalysisRequests = 0;
}

void addTwissAnalysisRequest(char *tag, char *startName, char *endName,
                             char *matchName,
                             long startOccurence, long endOccurence,
                             double sStart, double sEnd) {
  long i;
  if (!tag || !strlen(tag))
    bombElegant((char *)"NULL or blank tag passed to addTwissAnalysisRequest", NULL);
  if (!(startName && strlen(startName) && endName && strlen(endName)) && !(matchName && strlen(matchName)) && sStart == sEnd)
    bombElegant((char *)"must have both startName and endName, or matchName, or sStart!=sEnd (addTwissAnalysisRequest)", NULL);
  if ((startName || endName) && matchName)
    bombElegant((char *)"can't have startName or endName with matchName (addTwissAnalysisRequest)", NULL);
  for (i = 0; i < twissAnalysisRequests; i++)
    if (strcmp(twissAnalysisRequest[i].tag, tag) == 0)
      bombElegant((char *)"duplicate tag names seen (addTwissAnalysisRequest)", NULL);
  if (!(twissAnalysisRequest =
        (TWISS_ANALYSIS_REQUEST *)SDDS_Realloc(twissAnalysisRequest, sizeof(*twissAnalysisRequest) * (twissAnalysisRequests + 1))) ||
      !SDDS_CopyString(&twissAnalysisRequest[twissAnalysisRequests].tag, tag))
    bombElegant((char *)"memory allocation failure (addTwissAnalysisRequest)", NULL);
  twissAnalysisRequest[twissAnalysisRequests].startName =
    twissAnalysisRequest[twissAnalysisRequests].endName =
    twissAnalysisRequest[twissAnalysisRequests].matchName = NULL;
  if ((startName &&
       !SDDS_CopyString(&twissAnalysisRequest[twissAnalysisRequests].startName, startName)) ||
      (endName &&
       !SDDS_CopyString(&twissAnalysisRequest[twissAnalysisRequests].endName, endName)) ||
      (matchName &&
       !SDDS_CopyString(&twissAnalysisRequest[twissAnalysisRequests].matchName, matchName)))
    bombElegant((char *)"memory allocation failure (addTwissAnalysisRequest)", NULL);
  if (matchName &&
      has_wildcards(twissAnalysisRequest[twissAnalysisRequests].matchName) &&
      strchr(twissAnalysisRequest[twissAnalysisRequests].matchName, '-'))
    twissAnalysisRequest[twissAnalysisRequests].matchName =
      expand_ranges(twissAnalysisRequest[twissAnalysisRequests].matchName);
  twissAnalysisRequest[twissAnalysisRequests].sStart = sStart;
  twissAnalysisRequest[twissAnalysisRequests].sEnd = sEnd;
  twissAnalysisRequest[twissAnalysisRequests].startOccurence = startOccurence;
  twissAnalysisRequest[twissAnalysisRequests].endOccurence = endOccurence;
  twissAnalysisRequest[twissAnalysisRequests].initialized = 0;
  twissAnalysisRequests++;
}

void processTwissAnalysisRequests(ELEMENT_LIST *elem) {
  long i, is, iq, count, firstTime;
  char buffer[1024];
  ELEMENT_LIST *elemOrig;
  double value;
  // double end_pos, start_pos;
  double twissData[TWISS_ANALYSIS_STATS][TWISS_ANALYSIS_QUANTITIES];

  elemOrig = elem;
  // start_pos = 0;

  for (i = 0; i < twissAnalysisRequests; i++) {
    /* initialize statistics buffers and rpn memories */
    firstTime = !twissAnalysisRequest[i].initialized;
    for (iq = 0; iq < TWISS_ANALYSIS_QUANTITIES; iq++) {
      for (is = 0; is < TWISS_ANALYSIS_STATS; is++)
        if (!twissAnalysisRequest[i].initialized) {
          snprintf(buffer, 1024, (char *)"%s.%s.%s", twissAnalysisRequest[i].tag,
                   twissAnalysisStatName[is], twissAnalysisQuantityName[iq]);
          twissAnalysisRequest[i].twissMem[is][iq] = rpn_create_mem(buffer, 0);
        }
      twissData[TWISS_ANALYSIS_AVE][iq] = 0;
      twissData[TWISS_ANALYSIS_MIN][iq] = DBL_MAX;
      twissData[TWISS_ANALYSIS_MAX][iq] = -DBL_MAX;
    }
    twissAnalysisRequest[i].initialized = 1;

    count = 0;
    // end_pos = 0;
    while (elem) {
      if (twissAnalysisRequest[i].sStart < twissAnalysisRequest[i].sEnd &&
          elem->end_pos > twissAnalysisRequest[i].sEnd)
        break;
      if (twissAnalysisRequest[i].sStart < twissAnalysisRequest[i].sEnd &&
          elem->end_pos < twissAnalysisRequest[i].sStart) {
        elem = elem->succ;
        continue;
      }
      if (twissAnalysisRequest[i].matchName || twissAnalysisRequest[i].startName) {
        if (twissAnalysisRequest[i].matchName) {
          if (!wild_match(elem->name, twissAnalysisRequest[i].matchName)) {
            elem = elem->succ;
            continue;
          }
        } else {
          if (!count &&
              !(twissAnalysisRequest[i].startName &&
                strcmp(twissAnalysisRequest[i].startName, elem->name) == 0 &&
                (twissAnalysisRequest[i].startOccurence <= 1 ||
                 twissAnalysisRequest[i].startOccurence == elem->occurence))) {
            elem = elem->succ;
            continue;
          }
        }
      }
      count++;
      /*
        if (count==1) {
        if (elem->pred) {
        start_pos = elem->pred->end_pos;
        } else {
        start_pos = 0;
        }
        }
      */
      if (twiss_analysis_struct.verbosity > 1 && firstTime) {
        printf((char *)"twiss analysis %s will include %s#%ld\n",
               twissAnalysisRequest[i].tag,
               elem->name, elem->occurence);
      }
      for (iq = 0; iq < TWISS_ANALYSIS_QUANTITIES; iq++) {
        value = *(double *)((char *)elem->twiss + twissAnalysisQuantityOffset[iq]);
        for (is = 0; is < TWISS_ANALYSIS_STATS; is++) {
          switch (twissAnalysisStatCode[is]) {
          case TWISS_ANALYSIS_AVE:
            twissData[is][iq] += value;
            break;
          case TWISS_ANALYSIS_MIN:
            if (twissData[is][iq] > value)
              twissData[is][iq] = value;
            break;
          case TWISS_ANALYSIS_MAX:
            if (twissData[is][iq] < value)
              twissData[is][iq] = value;
            break;
          }
        }
      }
      if (twissAnalysisRequest[i].endName &&
          strcmp(twissAnalysisRequest[i].endName, elem->name) == 0 &&
          (twissAnalysisRequest[i].endOccurence <= 1 ||
           twissAnalysisRequest[i].endOccurence == elem->occurence))
        break;
      elem = elem->succ;
    }
    if (!count) {
      printf((char *)"error: twiss analysis conditions never satisfied for request with tag %s\n",
             twissAnalysisRequest[i].tag);
      exitElegant(1);
    }
    for (iq = 0; iq < TWISS_ANALYSIS_QUANTITIES; iq++) {
      twissData[TWISS_ANALYSIS_AVE][iq] /= count;
      for (is = 0; is < TWISS_ANALYSIS_STATS; is++) {
        rpn_store(twissData[is][iq], NULL, twissAnalysisRequest[i].twissMem[is][iq]);
      }
    }
    if (twiss_analysis_struct.verbosity && firstTime) {
      printf((char *)"%ld elements included in computations for twiss analysis request with tag %s\n",
             count,
             twissAnalysisRequest[i].tag);
    }

#if DEBUG
    if (twissAnalysisRequest[i].matchName) {
      printf((char *)"%ld matches for %s\n",
             count, twissAnalysisRequest[i].tag);
      for (iq = 0; iq < TWISS_ANALYSIS_QUANTITIES; iq++) {
        printf((char *)"%s: ", twissAnalysisQuantityName[iq]);
        for (is = 0; is < TWISS_ANALYSIS_STATS; is++)
          printf((char *)"%s=%e%s",
                 twissAnalysisStatName[is],
                 twissData[is][iq],
                 (is == TWISS_ANALYSIS_STATS - 1 ? (char *)"\n" : (char *)", "));
      }
      fflush(stdout);
    }
#endif
    elem = elemOrig;
  }
}

void setupTwissAnalysisRequest(NAMELIST_TEXT *nltext, RUN *run,
                               LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&twiss_analysis, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &twiss_analysis);

  if (twiss_analysis_struct.clear) {
    clearTwissAnalysisRequests();
    if (!(twiss_analysis_struct.start_name && twiss_analysis_struct.end_name) &&
        !twiss_analysis_struct.match_name &&
        twiss_analysis_struct.s_start == twiss_analysis_struct.s_end)
      return;
  }

  if (twiss_analysis_struct.start_name &&
      !strlen(trim_spaces(str_toupper(twiss_analysis_struct.start_name))))
    bombElegant((char *)"start_name is blank", NULL);
  if (twiss_analysis_struct.end_name &&
      !strlen(trim_spaces(str_toupper(twiss_analysis_struct.end_name))))
    bombElegant((char *)"end_name is blank", NULL);
  if (twiss_analysis_struct.match_name &&
      !strlen(trim_spaces(str_toupper(twiss_analysis_struct.match_name))))
    bombElegant((char *)"match_name is blank", NULL);
  if ((twiss_analysis_struct.tag &&
       !strlen(trim_spaces(twiss_analysis_struct.tag))) ||
      !twiss_analysis_struct.tag)
    bombElegant((char *)"tag is blank", NULL);

  if (!(twiss_analysis_struct.start_name && twiss_analysis_struct.end_name) && !twiss_analysis_struct.match_name &&
      twiss_analysis_struct.s_start == twiss_analysis_struct.s_end)
    bombElegant((char *)"you must give start_name and end_name, or match_name, or s_start different from s_end", NULL);
  if (twiss_analysis_struct.s_start > twiss_analysis_struct.s_end)
    bombElegant((char *)"s_start>s_end", NULL);
  addTwissAnalysisRequest(twiss_analysis_struct.tag,
                          twiss_analysis_struct.start_name,
                          twiss_analysis_struct.end_name,
                          twiss_analysis_struct.match_name,
                          twiss_analysis_struct.start_occurence,
                          twiss_analysis_struct.end_occurence,
                          twiss_analysis_struct.s_start,
                          twiss_analysis_struct.s_end);
}

#ifdef __cplusplus
extern "C" {
#endif
  void setupLinearChromaticTracking(NAMELIST_TEXT *nltext, LINE_LIST *beamline);
#ifdef __cplusplus
}
#endif

void setupLinearChromaticTracking(NAMELIST_TEXT *nltext, LINE_LIST *beamline) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&setup_linear_chromatic_tracking, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &setup_linear_chromatic_tracking);

  if (setup_linear_chromatic_tracking_struct.nux[0] < 0)
    bombElegant((char *)"nux < 0", NULL);
  if (setup_linear_chromatic_tracking_struct.nuy[0] < 0)
    bombElegant((char *)"nuy < 0", NULL);

  beamline->twiss0 = (TWISS *)malloc(sizeof(TWISS));

  setLinearChromaticTrackingValues(beamline);
  linearChromaticTrackingInitialized = 1;
}

void setLinearChromaticTrackingValues(LINE_LIST *beamline) {
  beamline->tune[0] = setup_linear_chromatic_tracking_struct.nux[0];
  beamline->chromaticity[0] = setup_linear_chromatic_tracking_struct.nux[1];
  beamline->chrom2[0] = setup_linear_chromatic_tracking_struct.nux[2];
  beamline->chrom3[0] = setup_linear_chromatic_tracking_struct.nux[3];

  beamline->twiss0->betax = setup_linear_chromatic_tracking_struct.betax[0];
  beamline->dbeta_dPoP[0] = setup_linear_chromatic_tracking_struct.betax[1];

  beamline->twiss0->alphax = setup_linear_chromatic_tracking_struct.alphax[0];
  beamline->dalpha_dPoP[0] = setup_linear_chromatic_tracking_struct.alphax[1];

  beamline->twiss0->etax = setup_linear_chromatic_tracking_struct.etax[0];
  beamline->eta2[0] = setup_linear_chromatic_tracking_struct.etax[1];
  beamline->eta3[0] = 0.0;
  beamline->twiss0->etapx = setup_linear_chromatic_tracking_struct.etapx[0];
  beamline->eta2[1] = setup_linear_chromatic_tracking_struct.etapx[1];
  beamline->eta3[1] = 0.0;

  beamline->tune[1] = setup_linear_chromatic_tracking_struct.nuy[0];
  beamline->chromaticity[1] = setup_linear_chromatic_tracking_struct.nuy[1];
  beamline->chrom2[1] = setup_linear_chromatic_tracking_struct.nuy[2];
  beamline->chrom3[1] = setup_linear_chromatic_tracking_struct.nuy[3];

  beamline->twiss0->betay = setup_linear_chromatic_tracking_struct.betay[0];
  beamline->dbeta_dPoP[1] = setup_linear_chromatic_tracking_struct.betay[1];

  beamline->twiss0->alphay = setup_linear_chromatic_tracking_struct.alphay[0];
  beamline->dalpha_dPoP[1] = setup_linear_chromatic_tracking_struct.alphay[1];

  beamline->twiss0->etay = setup_linear_chromatic_tracking_struct.etay[0];
  beamline->eta2[2] = setup_linear_chromatic_tracking_struct.etay[1];
  beamline->eta3[2] = 0.0;
  beamline->twiss0->etapy = setup_linear_chromatic_tracking_struct.etapy[0];
  beamline->eta2[3] = setup_linear_chromatic_tracking_struct.etapy[1];
  beamline->eta3[3] = 0.0;

  beamline->alpha[0] = setup_linear_chromatic_tracking_struct.alphac[0];
  beamline->alpha[1] = setup_linear_chromatic_tracking_struct.alphac[1];
}

typedef struct {
  double betax, betay;
  double rbetax, rbetay; /* rbetax = sqrt(betax) */
  double betax2, betay2; /* betax2 = betax^2 */
  double phix, phiy;
  std::complex<double> px[5], py[5]; /* px[j]=(exp(i*phix))^j j>0 */
  double b2L, b3L, s;
} ELEMDATA;

void computeSDrivingTerms(LINE_LIST *beamline) {

  /* Skew quadrupole */
  std::complex<double> f10010, f10100;
  /* Normal sextupole */
  std::complex<double> f30000, f12000, f10200, f01200, f01110;
  /* Skew Sextupole */
  std::complex<double> f00300, f00120, f20100, f20010, f11010;

  std::complex<double> ii = std::complex<double>(0, 1);

  double tilt;
  double k2;                   /* k2 <- normal sext */
  double j1, j2;               /* j1 <- skew quad, j2 <- skew sext */
  double src_betax, src_betay; /* betas where the source is located */
  double obs_phix, obs_phiy;
  double src_phix, src_phiy;
  double delta_phix, delta_phiy; /* phase advance between source and observator */
  double qx, qy;                 /* tunes */

  int idx;

  ELEMENT_LIST *src_ptr, *obs_ptr;

  qx = beamline->tune[0];
  qy = beamline->tune[1];

  if (beamline->sDrivingTerms.f10010 == NULL) {
    beamline->sDrivingTerms.f10010 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f10100 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f30000 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f12000 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f10200 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f01200 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f01110 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f00300 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f00120 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f20100 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f20010 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
    beamline->sDrivingTerms.f11010 = (double(*)[3])malloc(sizeof(double[3]) * beamline->n_elems);
  }

  idx = 0;
  obs_ptr = beamline->elem_twiss;
  while (obs_ptr) { /* loop over each observation point */

    f10010 = f10100 = f30000 = f12000 =
      f10200 = f01200 = f01110 = f00300 =
      f00120 = f20100 = f20010 = f11010 = std::complex<double>(0, 0);

    src_ptr = beamline->elem_twiss;

    if (obs_ptr->pred) {
      obs_phix = (obs_ptr->twiss->phix + obs_ptr->pred->twiss->phix) / 2;
      obs_phiy = (obs_ptr->twiss->phiy + obs_ptr->pred->twiss->phiy) / 2;
    } else {
      obs_phix = (obs_ptr->twiss->phix + beamline->twiss0->phix) / 2;
      obs_phiy = (obs_ptr->twiss->phiy + beamline->twiss0->phiy) / 2;
    }

    while (src_ptr) {           /* loop over each source */
      k2 = j1 = j2 = tilt = 0.; /* get source strength */
      switch (src_ptr->type) {
      case T_SEXT:
        tilt = ((SEXT *)src_ptr->p_elem)->tilt;
        k2 = ((SEXT *)src_ptr->p_elem)->k2 * ((SEXT *)src_ptr->p_elem)->length;
        break;
      case T_KSEXT:
        tilt = ((KSEXT *)src_ptr->p_elem)->tilt;
        k2 = ((KSEXT *)src_ptr->p_elem)->k2 *
          ((KSEXT *)src_ptr->p_elem)->length;
        break;
      case T_KQUSE:
        tilt = ((KQUSE *)src_ptr->p_elem)->tilt;
        j1 = -((KQUSE *)src_ptr->p_elem)->k1 *
          ((KQUSE *)src_ptr->p_elem)->length * sin(2. * tilt);
        k2 = ((KQUSE *)src_ptr->p_elem)->k2 * ((KQUSE *)src_ptr->p_elem)->length;
        break;
      case T_SBEN:
      case T_RBEN:
        tilt = ((BEND *)src_ptr->p_elem)->tilt;
        j1 = -((BEND *)src_ptr->p_elem)->k1 *
          ((BEND *)src_ptr->p_elem)->length * sin(2. * tilt);
        k2 = ((BEND *)src_ptr->p_elem)->k2 * ((BEND *)src_ptr->p_elem)->length;
        break;
      case T_CSBEND:
        tilt = ((CSBEND *)src_ptr->p_elem)->tilt;
        j1 = -((CSBEND *)src_ptr->p_elem)->k1 *
          ((CSBEND *)src_ptr->p_elem)->length * sin(2. * tilt);
        k2 = ((CSBEND *)src_ptr->p_elem)->k2 *
          ((CSBEND *)src_ptr->p_elem)->length;
        break;
      case T_CCBEND:
        tilt = ((CCBEND *)src_ptr->p_elem)->tilt;
        j1 = -((CCBEND *)src_ptr->p_elem)->K1 *
          ((CCBEND *)src_ptr->p_elem)->length * sin(2. * tilt);
        k2 = ((CCBEND *)src_ptr->p_elem)->K2 *
          ((CCBEND *)src_ptr->p_elem)->length;
        break;
      case T_CSRCSBEND:
        tilt = ((CSRCSBEND *)src_ptr->p_elem)->tilt;
        j1 = -((CSRCSBEND *)src_ptr->p_elem)->k1 *
          ((CSRCSBEND *)src_ptr->p_elem)->length * sin(2. * tilt);
        k2 = ((CSRCSBEND *)src_ptr->p_elem)->k2 *
          ((CSRCSBEND *)src_ptr->p_elem)->length;
        break;
      case T_QUAD:
        tilt = ((QUAD *)src_ptr->p_elem)->tilt;
        j1 = -((QUAD *)src_ptr->p_elem)->k1 *
          ((QUAD *)src_ptr->p_elem)->length * sin(2. * tilt);
        break;
      case T_KQUAD:
        tilt = ((KQUAD *)src_ptr->p_elem)->tilt;
        j1 = -((KQUAD *)src_ptr->p_elem)->k1 *
          ((KQUAD *)src_ptr->p_elem)->length * sin(2. * tilt);
        break;
      default:
        break;
      }

      if (!(k2 || j1 || j2)) {
        src_ptr = src_ptr->succ;
        continue;
      }

      /* Apply rotation */
      j2 = -k2 * sin(3. * tilt);
      k2 *= cos(3. * tilt);

      if (src_ptr->pred) {
        src_betax = (src_ptr->twiss->betax + src_ptr->pred->twiss->betax) / 2;
        src_betay = (src_ptr->twiss->betay + src_ptr->pred->twiss->betay) / 2;
        src_phix = (src_ptr->twiss->phix + src_ptr->pred->twiss->phix) / 2;
        src_phiy = (src_ptr->twiss->phiy + src_ptr->pred->twiss->phiy) / 2;
      } else {
        src_betax = (src_ptr->twiss->betax + beamline->twiss0->betax) / 2;
        src_betay = (src_ptr->twiss->betay + beamline->twiss0->betay) / 2;
        src_phix = (src_ptr->twiss->phix + beamline->twiss0->phix) / 2;
        src_phiy = (src_ptr->twiss->phiy + beamline->twiss0->phiy) / 2;
      }

      delta_phix = obs_phix - src_phix;
      delta_phiy = obs_phiy - src_phiy;

      if (delta_phix < 0.) {
        delta_phix += 2. * M_PI * qx;
      }

      if (delta_phiy < 0.) {
        delta_phiy += 2. * M_PI * qy;
      }

      f10010 += j1 * sqrt(src_betax * src_betay) * exp(ii * (delta_phix - delta_phiy));
      f10100 += j1 * sqrt(src_betax * src_betay) * exp(ii * (delta_phix + delta_phiy));
      f30000 += k2 * src_betax * sqrt(src_betax) * exp(3. * ii * delta_phix);
      f12000 += k2 * src_betax * sqrt(src_betax) * exp(-ii * delta_phix);
      f10200 += k2 * sqrt(src_betax) * src_betay * exp(ii * (delta_phix + 2. * delta_phiy));
      f01200 += k2 * sqrt(src_betax) * src_betay * exp(ii * (2. * delta_phiy - delta_phix));
      f01110 += k2 * sqrt(src_betax) * src_betay * exp(-ii * delta_phix);
      f00300 += j2 * src_betay * sqrt(src_betay) * exp(ii * 3. * delta_phiy);
      f00120 += j2 * src_betay * sqrt(src_betay) * exp(-ii * delta_phiy);
      f20100 += j2 * src_betax * sqrt(src_betay) * exp(ii * (2. * delta_phix + delta_phiy));
      f20010 += j2 * src_betax * sqrt(src_betay) * exp(ii * (2. * delta_phix - delta_phiy));
      f11010 += j2 * src_betax * sqrt(src_betay) * exp(-ii * delta_phiy);

      src_ptr = src_ptr->succ;
    }
    f10010 /= 4. * (1. - exp(2. * M_PI * ii * (qx - qy)));
    f10100 /= 4. * (1. - exp(2. * M_PI * ii * (qx + qy)));
    f30000 /= 48. * (1. - exp(2. * M_PI * ii * 3. * qx));
    f12000 /= 16. * (1. - exp(2. * M_PI * ii * qx));
    f10200 /= 16. * (1. - exp(2. * M_PI * ii * (qx + 2. * qy)));
    f01200 /= 16. * (1. - exp(2. * M_PI * ii * (2. * qy - qx)));
    f01110 /= 8. * (1. - exp(-2. * M_PI * ii * qx));
    f00300 /= 48. * (1. - exp(2. * M_PI * ii * 3. * qy));
    f00120 /= 16. * (1. - exp(-2. * M_PI * ii * qy));
    f20100 /= 16. * (1. - exp(2. * M_PI * ii * (2. * qx + qy)));
    f20010 /= 16. * (1. - exp(2. * M_PI * ii * (2. * qx - qy)));
    f11010 /= 8. * (1. - exp(-2. * M_PI * ii * qy));

    beamline->sDrivingTerms.f10010[idx][0] = std::abs<double>(f10010);
    beamline->sDrivingTerms.f10100[idx][0] = std::abs<double>(f10100);
    beamline->sDrivingTerms.f30000[idx][0] = std::abs<double>(f30000);
    beamline->sDrivingTerms.f12000[idx][0] = std::abs<double>(f12000);
    beamline->sDrivingTerms.f10200[idx][0] = std::abs<double>(f10200);
    beamline->sDrivingTerms.f01200[idx][0] = std::abs<double>(f01200);
    beamline->sDrivingTerms.f01110[idx][0] = std::abs<double>(f01110);
    beamline->sDrivingTerms.f00300[idx][0] = std::abs<double>(f00300);
    beamline->sDrivingTerms.f00120[idx][0] = std::abs<double>(f00120);
    beamline->sDrivingTerms.f20100[idx][0] = std::abs<double>(f20100);
    beamline->sDrivingTerms.f20010[idx][0] = std::abs<double>(f20010);
    beamline->sDrivingTerms.f11010[idx][0] = std::abs<double>(f11010);

    beamline->sDrivingTerms.f10010[idx][1] = f10010.real();
    beamline->sDrivingTerms.f10100[idx][1] = f10100.real();
    beamline->sDrivingTerms.f30000[idx][1] = f30000.real();
    beamline->sDrivingTerms.f12000[idx][1] = f12000.real();
    beamline->sDrivingTerms.f10200[idx][1] = f10200.real();
    beamline->sDrivingTerms.f01200[idx][1] = f01200.real();
    beamline->sDrivingTerms.f01110[idx][1] = f01110.real();
    beamline->sDrivingTerms.f00300[idx][1] = f00300.real();
    beamline->sDrivingTerms.f00120[idx][1] = f00120.real();
    beamline->sDrivingTerms.f20100[idx][1] = f20100.real();
    beamline->sDrivingTerms.f20010[idx][1] = f20010.real();
    beamline->sDrivingTerms.f11010[idx][1] = f11010.real();

    beamline->sDrivingTerms.f10010[idx][2] = f10010.imag();
    beamline->sDrivingTerms.f10100[idx][2] = f10100.imag();
    beamline->sDrivingTerms.f30000[idx][2] = f30000.imag();
    beamline->sDrivingTerms.f12000[idx][2] = f12000.imag();
    beamline->sDrivingTerms.f10200[idx][2] = f10200.imag();
    beamline->sDrivingTerms.f01200[idx][2] = f01200.imag();
    beamline->sDrivingTerms.f01110[idx][2] = f01110.imag();
    beamline->sDrivingTerms.f00300[idx][2] = f00300.imag();
    beamline->sDrivingTerms.f00120[idx][2] = f00120.imag();
    beamline->sDrivingTerms.f20100[idx][2] = f20100.imag();
    beamline->sDrivingTerms.f20010[idx][2] = f20010.imag();
    beamline->sDrivingTerms.f11010[idx][2] = f11010.imag();

    idx++;
    obs_ptr = obs_ptr->succ;
  }
}

void computeDrivingTerms(DRIVING_TERMS *d, ELEMENT_LIST *elem, TWISS *twiss0, double *tune, long nPeriods)
/* Based on J. Bengtsson, SLS Note 9/97, March 7, 1997, with corrections per W. Guo (NSLS) */
/* Revised to follow C. X. Wang AOP-TN-2009-020 for second-order terms */
{
  std::complex<double> h11001, h00111, h20001, h00201, h10002, h10010, h10100;
  std::complex<double> h21000, h30000, h10110, h10020, h10200;
  std::complex<double> h22000, h11110, h00220, h31000, h40000;
  std::complex<double> h20110, h11200, h20020, h20200, h00310, h00400;
  std::complex<double> t1, t2;
  // std::complex<double>  t3, t4;
  std::complex<double> ii;
  std::complex<double> periodicFactor[9][9];
#define PF(i, j) (periodicFactor[4 + i][4 + j])
  double betax1, betay1, phix1, phiy1, etax1, termSign;
  double b2L, a2L, b3L, b4L, nux, nuy;
  ELEMENT_LIST *eptr1;
  double two = 2, three = 3, four = 4;
  ELEMDATA *ed = NULL;
  long nE = 0, iE, jE, i, j;
  // double sqrt8, sqrt2;
  double tilt;

  // sqrt8 = sqrt((double)8);
  // sqrt2 = sqrt((double)2);
  ii = std::complex<double>(0, 1);

  /* accumulate real and imaginary parts */
  h11001 = h00111 = h20001 = h00201 = h10002 = std::complex<double>(0, 0);
  h21000 = h30000 = h10110 = h10020 = h10200 = std::complex<double>(0, 0);
  h22000 = h11110 = h00220 = h31000 = h40000 = std::complex<double>(0, 0);
  h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = std::complex<double>(0, 0);
  h10100 = h10010 = std::complex<double>(0, 0);

  d->dnux_dJx = d->dnux_dJy = d->dnuy_dJy = 0;

  if (nPeriods != 1) {
    double a1, a2;
    for (i = 0; i < 9; i++) {
      for (j = 0; j < 9; j++) {
        a1 = PIx2 * (tune[0] * (i - 4) + tune[1] * (j - 4));
        a2 = a1 / nPeriods;
        periodicFactor[i][j] = (exp(ii * a1) - 1.0) / (exp(ii * a2) - 1.0);
      }
    }
  } else {
    for (i = 0; i < 9; i++)
      for (j = 0; j < 9; j++)
        periodicFactor[i][j] = 1;
  }

  eptr1 = elem;
  while (eptr1) {
    a2L = b2L = b3L = b4L = 0;
    switch (eptr1->type) {
    case T_SEXT:
      b3L = ((SEXT *)eptr1->p_elem)->k2 * ((SEXT *)eptr1->p_elem)->length / 2;
      break;
    case T_KSEXT:
      b3L = ((KSEXT *)eptr1->p_elem)->k2 * ((KSEXT *)eptr1->p_elem)->length / 2;
      break;
    case T_KQUSE:
      b2L = ((KQUSE *)eptr1->p_elem)->k1 * ((KQUSE *)eptr1->p_elem)->length;
      b3L = ((KQUSE *)eptr1->p_elem)->k2 * ((KQUSE *)eptr1->p_elem)->length / 2;
      break;
    case T_SBEN:
    case T_RBEN:
      b2L = ((BEND *)eptr1->p_elem)->k1 * ((BEND *)eptr1->p_elem)->length;
      b3L = ((BEND *)eptr1->p_elem)->k2 * ((BEND *)eptr1->p_elem)->length / 2;
      break;
    case T_CSBEND:
      b2L = ((CSBEND *)eptr1->p_elem)->k1 * ((CSBEND *)eptr1->p_elem)->length;
      b3L = ((CSBEND *)eptr1->p_elem)->k2 * ((CSBEND *)eptr1->p_elem)->length / 2;
      break;
    case T_CCBEND:
      b2L = ((CCBEND *)eptr1->p_elem)->K1 * ((CCBEND *)eptr1->p_elem)->length;
      b3L = ((CCBEND *)eptr1->p_elem)->K2 * ((CCBEND *)eptr1->p_elem)->length / 2;
      break;
    case T_CSRCSBEND:
      b2L = ((CSRCSBEND *)eptr1->p_elem)->k1 * ((CSRCSBEND *)eptr1->p_elem)->length;
      b3L = ((CSRCSBEND *)eptr1->p_elem)->k2 * ((CSRCSBEND *)eptr1->p_elem)->length / 2;
      break;
    case T_QUAD:
      tilt = ((QUAD *)eptr1->p_elem)->tilt;
      b2L = ((QUAD *)eptr1->p_elem)->k1 * ((QUAD *)eptr1->p_elem)->length * cos(2 * tilt);
      a2L = ((QUAD *)eptr1->p_elem)->k1 * ((QUAD *)eptr1->p_elem)->length * sin(2 * tilt);
      break;
    case T_KQUAD:
      tilt = ((KQUAD *)eptr1->p_elem)->tilt;
      b2L = ((KQUAD *)eptr1->p_elem)->k1 * ((KQUAD *)eptr1->p_elem)->length * cos(2 * tilt);
      a2L = ((KQUAD *)eptr1->p_elem)->k1 * ((KQUAD *)eptr1->p_elem)->length * sin(2 * tilt);
      break;
    case T_OCT:
      b4L = ((OCTU *)eptr1->p_elem)->k3 * ((OCTU *)eptr1->p_elem)->length / 6;
      break;
    case T_KOCT:
      b4L = ((KOCT *)eptr1->p_elem)->k3 * ((OCTU *)eptr1->p_elem)->length / 6;
      break;
    default:
      break;
    }
    if (a2L || b2L || b3L || b4L) {
      if (eptr1->pred) {
        betax1 = (eptr1->twiss->betax + eptr1->pred->twiss->betax) / 2;
        etax1 = (eptr1->twiss->etax + eptr1->pred->twiss->etax) / 2;
        phix1 = (eptr1->twiss->phix + eptr1->pred->twiss->phix) / 2;
        betay1 = (eptr1->twiss->betay + eptr1->pred->twiss->betay) / 2;
        phiy1 = (eptr1->twiss->phiy + eptr1->pred->twiss->phiy) / 2;
      } else {
        betax1 = (eptr1->twiss->betax + twiss0->betax) / 2;
        etax1 = (eptr1->twiss->etax + twiss0->etax) / 2;
        phix1 = (eptr1->twiss->phix + twiss0->phix) / 2;
        betay1 = (eptr1->twiss->betay + twiss0->betay) / 2;
        phiy1 = (eptr1->twiss->phiy + twiss0->phiy) / 2;
      }

      ed = (ELEMDATA *)SDDS_Realloc(ed, sizeof(*ed) * (nE + 1));
      ed[nE].s = eptr1->end_pos;
      ed[nE].b2L = b2L;
      ed[nE].b3L = b3L;
      ed[nE].betax = betax1;
      ed[nE].betax2 = sqr(betax1);
      ed[nE].rbetax = sqrt(betax1);
      ed[nE].phix = phix1;
      ed[nE].px[1] = exp(ii * phix1);
      ed[nE].px[2] = ed[nE].px[1] * ed[nE].px[1];
      ed[nE].px[3] = ed[nE].px[1] * ed[nE].px[2];
      ed[nE].px[4] = ed[nE].px[1] * ed[nE].px[3];
      ed[nE].betay = betay1;
      ed[nE].betay2 = sqr(betay1);
      ed[nE].rbetay = sqrt(betay1);
      ed[nE].phiy = phiy1;
      ed[nE].py[1] = exp(ii * phiy1);
      ed[nE].py[2] = ed[nE].py[1] * ed[nE].py[1];
      ed[nE].py[3] = ed[nE].py[1] * ed[nE].py[2];
      ed[nE].py[4] = ed[nE].py[1] * ed[nE].py[3];

      if (a2L) {
        /* linear coupling terms */
        h10010 += (a2L / 4) * ed[nE].rbetax * ed[nE].rbetay * ed[nE].px[1] / ed[nE].py[1] * PF(1, -1);
        h10100 += (a2L / 4) * ed[nE].rbetax * ed[nE].rbetay * ed[nE].px[1] * ed[nE].py[1] * PF(1, 1);
      }
      if (b2L || b3L) {
        /* first-order chromatic terms */
        /* h11001 and h00111 */
        h11001 += (b3L * betax1 * etax1 / 2 - b2L * betax1 / 4) * nPeriods;
        h00111 += (b2L * betay1 / 4 - b3L * betay1 * etax1 / 2) * nPeriods;

        /* h20001, h00201 */
        h20001 += (b3L * betax1 * etax1 / 2 - b2L * betax1 / 4) / 2 * ed[nE].px[2] * PF(2, 0);
        h00201 += (b2L * betay1 / 4 - b3L * betay1 * etax1 / 2) / 2 * ed[nE].py[2] * PF(0, 2);

        /* h10002 */
        h10002 += (b3L * ed[nE].rbetax * ipow2(etax1) - b2L * ed[nE].rbetax * etax1) / 2 * ed[nE].px[1] * PF(1, 0);
      }
      if (ed[nE].b3L) {
        /* first-order geometric terms from sextupoles */
        /* h21000 */
        h21000 += b3L * ed[nE].rbetax * betax1 / 8 * ed[nE].px[1] * PF(1, 0);

        /* h30000 */
        h30000 += b3L * ed[nE].rbetax * betax1 / 24 * ed[nE].px[3] * PF(3, 0);

        /* h10110 */
        h10110 += -b3L * ed[nE].rbetax * betay1 / 4 * ed[nE].px[1] * PF(1, 0);

        /* h10020 and h10200 */
        h10020 += -b3L * ed[nE].rbetax * betay1 / 8 * ed[nE].px[1] * conj(ed[nE].py[2]) * PF(1, -2);

        h10200 += -b3L * ed[nE].rbetax * betay1 / 8 * ed[nE].px[1] * ed[nE].py[2] * PF(1, 2);
      }
      if (b4L) {
        /* second-order terms from leading order effects of octupoles */
        /* Ignoring a large number of terms that are not also driven by sextupoles */

        d->dnux_dJx += 3 * b4L * ed[nE].betax2 / (8 * PI) * nPeriods;
        d->dnux_dJy -= 3 * b4L * betax1 * betay1 / (4 * PI) * nPeriods;
        d->dnuy_dJy += 3 * b4L * ed[nE].betay2 / (8 * PI) * nPeriods;

        h22000 += 3 * b4L * ed[nE].betax2 / 32 * nPeriods;
        h11110 += -3 * b4L * betax1 * betay1 / 8 * nPeriods;
        h00220 += 3 * b4L * ed[nE].betay2 / 32 * nPeriods;
        h31000 += b4L * ed[nE].betax2 / 16 * ed[nE].px[2] * PF(2, 0);
        h40000 += b4L * ed[nE].betax2 / 64 * ed[nE].px[4] * PF(4, 0);
        h20110 += -3 * b4L * betax1 * betay1 / 16 * ed[nE].px[2] * PF(2, 0);
        h11200 += -3 * b4L * betax1 * betay1 / 16 * ed[nE].py[2] * PF(0, 2);
        h20020 += -3 * b4L * betax1 * betay1 / 32 * ed[nE].px[2] * conj(ed[nE].py[2]) * PF(2, -2);
        h20200 += -3 * b4L * betax1 * betay1 / 32 * ed[nE].px[2] * ed[nE].py[2] * PF(2, 2);
        h00310 += b4L * ed[nE].betay2 / 16 * ed[nE].py[2] * PF(0, 2);
        h00400 += b4L * ed[nE].betay2 / 64 * ed[nE].py[4] * PF(0, 4);
      }
      nE++;
    }
    eptr1 = eptr1->succ;
  }

  /* Done with the leading-order quad and sext terms */
  d->h11001[0] = std::abs<double>(h11001);
  d->h11001[1] = h11001.real();
  d->h11001[2] = h11001.imag();
  d->h00111[0] = std::abs<double>(h00111);
  d->h00111[1] = h00111.real();
  d->h00111[2] = h00111.imag();
  d->h20001[0] = std::abs<double>(h20001);
  d->h20001[1] = h20001.real();
  d->h20001[2] = h20001.imag();
  d->h00201[0] = std::abs<double>(h00201);
  d->h00201[1] = h00201.real();
  d->h00201[2] = h00201.imag();
  d->h10002[0] = std::abs<double>(h10002);
  d->h10002[1] = h10002.real();
  d->h10002[2] = h10002.imag();

  d->h10100[0] = std::abs<double>(h10100);
  d->h10100[1] = h10100.real();
  d->h10100[2] = h10100.imag();
  d->h10010[0] = std::abs<double>(h10010);
  d->h10010[1] = h10010.real();
  d->h10010[2] = h10010.imag();

  d->h21000[0] = std::abs<double>(h21000);
  d->h21000[1] = h21000.real();
  d->h21000[2] = h21000.imag();
  d->h30000[0] = std::abs<double>(h30000);
  d->h30000[1] = h30000.real();
  d->h30000[2] = h30000.imag();
  d->h10110[0] = std::abs<double>(h10110);
  d->h10110[1] = h10110.real();
  d->h10110[2] = h10110.imag();
  d->h10020[0] = std::abs<double>(h10020);
  d->h10020[1] = h10020.real();
  d->h10020[2] = h10020.imag();
  d->h10200[0] = std::abs<double>(h10200);
  d->h10200[1] = h10200.real();
  d->h10200[2] = h10200.imag();

  if (!leading_order_driving_terms_only) {
    /* compute sextupole contributions to second-order terms */
    if (nPeriods != 1)
      bombElegant("Computating of higher-order driving terms not available when n_periods!=1", NULL);

    nux = tune[0];
    nuy = tune[1];
    for (iE = 0; iE < nE; iE++) {
      if (ed[iE].b3L) {
        for (jE = 0; jE < nE; jE++) {
          if (ed[jE].b3L) {
            d->dnux_dJx += ed[iE].b3L * ed[jE].b3L / (-16 * PI) * pow(ed[iE].betax * ed[jE].betax, 1.5) *
              (3 * cos(fabs(ed[iE].phix - ed[jE].phix) - PI * nux) / sin(PI * nux) + cos(fabs(3 * (ed[iE].phix - ed[jE].phix)) - 3 * PI * nux) / sin(3 * PI * nux));
            d->dnux_dJy += ed[iE].b3L * ed[jE].b3L / (8 * PI) * sqrt(ed[iE].betax * ed[jE].betax) * ed[iE].betay *
              (2 * ed[jE].betax * cos(fabs(ed[iE].phix - ed[jE].phix) - PI * nux) / sin(PI * nux) - ed[jE].betay * cos(fabs(ed[iE].phix - ed[jE].phix) + 2 * fabs(ed[iE].phiy - ed[jE].phiy) - PI * (nux + 2 * nuy)) / sin(PI * (nux + 2 * nuy)) + ed[jE].betay * cos(fabs(ed[iE].phix - ed[jE].phix) - 2 * fabs(ed[iE].phiy - ed[jE].phiy) - PI * (nux - 2 * nuy)) / sin(PI * (nux - 2 * nuy)));
            d->dnuy_dJy += ed[iE].b3L * ed[jE].b3L / (-16 * PI) * sqrt(ed[iE].betax * ed[jE].betax) * ed[iE].betay * ed[jE].betay *
              (4 * cos(fabs(ed[iE].phix - ed[jE].phix) - PI * nux) / sin(PI * nux) + cos(fabs(ed[iE].phix - ed[jE].phix) + 2 * fabs(ed[iE].phiy - ed[jE].phiy) - PI * (nux + 2 * nuy)) / sin(PI * (nux + 2 * nuy)) + cos(fabs(ed[iE].phix - ed[jE].phix) - 2 * fabs(ed[iE].phiy - ed[jE].phiy) - PI * (nux - 2 * nuy)) / sin(PI * (nux - 2 * nuy)));
            termSign = SIGN(ed[iE].s - ed[jE].s);
            if (termSign) {
              /* geometric terms */
              h22000 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betax * ed[jE].betax *
                (ed[iE].px[3] * conj(ed[jE].px[3]) + three * ed[iE].px[1] * conj(ed[jE].px[1]));
              h31000 += (1. / 32) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betax * ed[jE].betax *
                ed[iE].px[3] * conj(ed[jE].px[1]);
              t1 = conj(ed[iE].px[1]) * ed[jE].px[1];
              t2 = ed[iE].px[1] * conj(ed[jE].px[1]);
              h11110 += (1. / 16) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay *
                (ed[jE].betax * (t1 - conj(t1)) +
                 ed[jE].betay * ed[iE].py[2] * conj(ed[jE].py[2]) * (conj(t1) + t1));
              t1 = exp(-ii * (ed[iE].phix - ed[jE].phix));
              t2 = conj(t1);
              h11200 += (1. / 32) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay * exp(ii * (2 * ed[iE].phiy)) *
                (ed[jE].betax * (t1 - t2) +
                 two * ed[jE].betay * (t2 + t1));
              h40000 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betax * ed[jE].betax *
                ed[iE].px[3] * ed[jE].px[1];
              h20020 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay *
                (ed[jE].betax * conj(ed[iE].px[1] * ed[iE].py[2]) * ed[jE].px[3] - (ed[jE].betax + four * ed[jE].betay) * ed[iE].px[1] * ed[jE].px[1] * conj(ed[iE].py[2]));
              h20110 += (1. / 32) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay *
                (ed[jE].betax * (conj(ed[iE].px[1]) * ed[jE].px[3] -
                                 ed[iE].px[1] * ed[jE].px[1]) +
                 two * ed[jE].betay * ed[iE].px[1] * ed[jE].px[1] * ed[iE].py[2] * conj(ed[jE].py[2]));
              h20200 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay *
                (ed[jE].betax *
                 conj(ed[iE].px[1]) * ed[jE].px[3] * ed[iE].py[2] -
                 (ed[jE].betax - four * ed[jE].betay) *
                 ed[iE].px[1] * ed[jE].px[1] * ed[iE].py[2]);
              h00220 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay * ed[jE].betay *
                (ed[iE].px[1] * ed[iE].py[2] * conj(ed[jE].px[1] * ed[jE].py[2]) + four * ed[iE].px[1] * conj(ed[jE].px[1]) - conj(ed[iE].px[1] * ed[jE].py[2]) * ed[jE].px[1] * ed[iE].py[2]);
              h00310 += (1. / 32) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay * ed[jE].betay * ed[iE].py[2] *
                (ed[iE].px[1] * conj(ed[jE].px[1]) - conj(ed[iE].px[1]) * ed[jE].px[1]);
              h00400 += (1. / 64) * termSign * ii * ed[iE].b3L * ed[jE].b3L *
                ed[iE].rbetax * ed[jE].rbetax * ed[iE].betay * ed[jE].betay *
                ed[iE].px[1] * conj(ed[jE].px[1]) * ed[iE].py[2] * ed[jE].py[2];
            }
          }
        }
      }
    }
  }

  d->h22000[0] = std::abs<double>(h22000);
  d->h22000[1] = h22000.real();
  d->h22000[2] = h22000.imag();
  d->h11110[0] = std::abs<double>(h11110);
  d->h11110[1] = h11110.real();
  d->h11110[2] = h11110.imag();
  d->h00220[0] = std::abs<double>(h00220);
  d->h00220[1] = h00220.real();
  d->h00220[2] = h00220.imag();
  d->h31000[0] = std::abs<double>(h31000);
  d->h31000[1] = h31000.real();
  d->h31000[2] = h31000.imag();
  d->h40000[0] = std::abs<double>(h40000);
  d->h40000[1] = h40000.real();
  d->h40000[2] = h40000.imag();
  d->h20110[0] = std::abs<double>(h20110);
  d->h20110[1] = h20110.real();
  d->h20110[2] = h20110.imag();
  d->h11200[0] = std::abs<double>(h11200);
  d->h11200[1] = h11200.real();
  d->h11200[2] = h11200.imag();
  d->h20020[0] = std::abs<double>(h20020);
  d->h20020[1] = h20020.real();
  d->h20020[2] = h20020.imag();
  d->h20200[0] = std::abs<double>(h20200);
  d->h20200[1] = h20200.real();
  d->h20200[2] = h20200.imag();
  d->h00310[0] = std::abs<double>(h00310);
  d->h00310[1] = h00310.real();
  d->h00310[2] = h00310.imag();
  d->h00400[0] = std::abs<double>(h00400);
  d->h00400[1] = h00400.real();
  d->h00400[2] = h00400.imag();

  if (ed)
    free(ed);
}

void setup_rf_setup(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long do_twiss_output, long *do_rf_setup) {
  ELEMENT_LIST *eptr;

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&rf_setup, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &rf_setup);
  if (rf_setup_struct.disable)
    return;

  if ((rf_setup_struct.start_occurence > 0 && rf_setup_struct.end_occurence < 1) ||
      (rf_setup_struct.start_occurence < 1 && rf_setup_struct.end_occurence > 0) ||
      rf_setup_struct.start_occurence > rf_setup_struct.end_occurence)
    bombElegant("Invalid start and end occurence values", NULL);
  if (rf_setup_struct.s_start > rf_setup_struct.s_end)
    bombElegant("Invalid s_start and s_end values", NULL);
  if (rf_setup_struct.near_frequency > 0 && rf_setup_struct.harmonic > 1)
    bombElegant("Give non-zero near_frequency or harmonic value, not both", NULL);
  if (rf_setup_struct.bucket_half_height > 0 && rf_setup_struct.over_voltage > 0 && rf_setup_struct.total_voltage > 0)
    bombElegant("Give non-zero value for only one of bucket_half_height, over_voltage, or total_voltage", NULL);
  if (rf_setup_struct.bucket_half_height <= 0 && rf_setup_struct.over_voltage <= 0 && rf_setup_struct.total_voltage <= 0)
    bombElegant("Give non-zero value for one of bucket_half_height, over_voltage, or total_voltage", NULL);

  if (!do_twiss_output && rf_setup_struct.set_for_each_step)
    bombElegant("If set_for_each_step is non-zero, must also ask for twiss output at each step", NULL);
  if (do_twiss_output && !rf_setup_struct.set_for_each_step)
    bombElegant("If set_for_each_step is zero, cannot also ask for twiss output at each step", NULL);

  if (!matched || !radiation_integrals)
    bombElegant("Need to compute matched twiss parameters with radiation integrals for RF setup", NULL);

  if (rf_setup_struct.name) {
    str_toupper(rf_setup_struct.name);
    if (has_wildcards(rf_setup_struct.name) && strchr(rf_setup_struct.name, '-'))
      rf_setup_struct.name = expand_ranges(rf_setup_struct.name);
  }

  free(rfElem);
  nRfElem = 0;
  rfElem = NULL;

  if (!rf_setup_struct.output_only) {
    eptr = beamline->elem;

    while (eptr) {
      if ((eptr->type != T_RFCA && (eptr->type!=T_MODRF))
	  || (rf_setup_struct.name && !wild_match(eptr->name, rf_setup_struct.name))
	  || (rf_setup_struct.start_occurence > 0 && eptr->occurence < rf_setup_struct.start_occurence)
	  || (rf_setup_struct.end_occurence > 0 && eptr->occurence > rf_setup_struct.end_occurence)
	  || (rf_setup_struct.s_start > 0 && eptr->end_pos < rf_setup_struct.s_start)
	  || (rf_setup_struct.s_end > 0 && eptr->end_pos > rf_setup_struct.s_end)) {
        eptr = eptr->succ;
        continue;
      }
      rfElem = (ELEMENT_LIST **)SDDS_Realloc(rfElem, sizeof(*rfElem) * (nRfElem + 1));
      rfElem[nRfElem] = eptr;
      nRfElem++;
      eptr = eptr->succ;
    }

    if (nRfElem == 0)
      bombElegant("No RFCA or MODRF elements found meeting requirements", NULL);
  } else {
    if (rf_setup_struct.filename == NULL || strlen(rf_setup_struct.filename) == 0)
      bombElegant("No filename provided but output_only requested.", NULL);
  }

  if (fpRf)
    fclose(fpRf);
  fpRf = NULL;
  if (rf_setup_struct.filename) {
    rf_setup_struct.filename = compose_filename(rf_setup_struct.filename, run->rootname);
    fpRf = fopen(rf_setup_struct.filename, "w");
    fprintf(fpRf, "SDDS1\n&parameter name=PhiSynch, type=double, units=deg &end\n");
    fprintf(fpRf, "&parameter name=Voltage, type=double, units=V &end\n");
    fprintf(fpRf, "&parameter name=BucketHalfHeight, type=double &end\n");
    fprintf(fpRf, "&parameter name=nuSynch, type=double &end\n");
    fprintf(fpRf, "&parameter name=Sz0, type=double, units=m &end\n");
    fprintf(fpRf, "&parameter name=St0, type=double, units=s &end\n");
    fprintf(fpRf, "&parameter name=Frequency, type=double, units=Hz &end\n");
    fprintf(fpRf, "&parameter name=Harmonic, type=long &end\n");
    fprintf(fpRf, "&data mode=ascii &end\n");
  }

  if (fpBucket)
    fclose(fpBucket);
  fpBucket = NULL;
  if (rf_setup_struct.bucket_filename) {
    if (rf_setup_struct.bucket_points < 2)
      bombElegant("rf_setup bucket_points must be at least 2", NULL);
    rf_setup_struct.bucket_filename = compose_filename(rf_setup_struct.bucket_filename, run->rootname);
    fpBucket = fopen(rf_setup_struct.bucket_filename, "w");
    fprintf(fpBucket, "SDDS1\n");
    fprintf(fpBucket, "&parameter name=Frequency, type=double, units=Hz &end\n");
    fprintf(fpBucket, "&parameter name=Harmonic, type=long &end\n");
    fprintf(fpBucket, "&parameter name=Voltage, type=double, units=V &end\n");
    fprintf(fpBucket, "&parameter name=PhiSynch, type=double, units=deg &end\n");
    fprintf(fpBucket, "&parameter name=BucketHalfHeight, type=double &end\n");
    fprintf(fpBucket, "&column name=t, type=double, units=s, description=\"time relative to the synchronous particle\" &end\n");
    fprintf(fpBucket, "&column name=p, type=double, units=\"m$be$nc\", description=\"central momentum on the separatrix\" &end\n");
    fprintf(fpBucket, "&column name=delta, type=double, description=\"(p-pCentral)/pCentral on the separatrix\" &end\n");
    fprintf(fpBucket, "&data mode=ascii &end\n");
  }

  *do_rf_setup = 0;
  if (!rf_setup_struct.set_for_each_step)
    run_rf_setup(run, beamline, 1);
  else
    *do_rf_setup = 1;
}

void run_rf_setup(RUN *run, LINE_LIST *beamline, long writeToFile) {
  double beta, T0, frf, q, voltage;
  long harmonic, i;
  RFCA *rfca;
  MODRF *modrf;
  long iFreqRfca, iVoltRfca, iPhaseRfca;
  long iFreqModrf, iVoltModrf, iPhaseModrf;
  double phase;
  double St0, Sz0, nus, rfAcceptance = -1;
  double wrf, w0, Vdot, E;
  long h;
  double eta, gamma, dPOverP, pCentral;

  if (!(beamline->flags & BEAMLINE_TWISS_CURRENT) || !(beamline->flags & BEAMLINE_RADINT_CURRENT)) {
    if (output_before_tune_correction)
      bombElegant("twiss parameters and radiation integrals not up-to-date (do_rf_setup)", NULL);
    return;
  }

  if (beamline->revolution_length <= 0)
    bombElegant("Beamline length is undefined (do_rf_setup)", NULL);

  pCentral = run->p_central;
  dPOverP = 0;
  if (beamline->closed_orbit) {
    /*
      printf("Using closed orbit to compute orbit length: centroid[4] = %21.15e, centroid[5] = %21.15e\n",
      beamline->closed_orbit[beamline->n_elems-1].centroid[4],
      beamline->closed_orbit[beamline->n_elems-1].centroid[5]);
    */
    dPOverP = beamline->closed_orbit[beamline->n_elems - 1].centroid[5];
    pCentral = run->p_central * (1 + dPOverP);
    beta = pCentral / (gamma = sqrt(sqr(pCentral) + 1));
    T0 = beamline->closed_orbit[beamline->n_elems - 1].centroid[4] / (beta * c_mks);
  } else {
    beta = run->p_central / (gamma = sqrt(sqr(run->p_central) + 1));
    T0 = beamline->revolution_length / (beta * c_mks);
  }
  if (rf_setup_struct.harmonic > 0)
    harmonic = rf_setup_struct.harmonic;
  else
    harmonic = rf_setup_struct.near_frequency * T0 + 0.5;

  iFreqRfca = confirm_parameter((char *)"FREQ", T_RFCA);
  iVoltRfca = confirm_parameter((char *)"VOLT", T_RFCA);
  iPhaseRfca = confirm_parameter((char *)"PHASE", T_RFCA);
  iFreqModrf = confirm_parameter((char *)"FREQ", T_MODRF);
  iVoltModrf = confirm_parameter((char *)"VOLT", T_MODRF);
  iPhaseModrf = confirm_parameter((char *)"PHASE", T_MODRF);
  frf = harmonic / T0 * (1 + rf_setup_struct.fractional_frequency_change);
  printf("\nRf setup: frequency is %21.15e Hz (h=%ld)\n", frf, harmonic);
  for (i = 0; i < nRfElem; i++) {
    if (rfElem[i]->type == T_RFCA) {
      rfca = (RFCA *)(rfElem[i]->p_elem);
      rfca->freq = frf;
      change_defined_parameter(rfElem[i]->name, iFreqRfca, T_RFCA, frf, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
      change_used_parameter(beamline, rfElem[i]->name, iFreqRfca, T_RFCA, frf, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
    } else {
      modrf = (MODRF *)(rfElem[i]->p_elem);
      modrf->freq = frf;
      change_defined_parameter(rfElem[i]->name, iFreqModrf, T_MODRF, frf, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
      change_used_parameter(beamline, rfElem[i]->name, iFreqModrf, T_MODRF, frf, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
    }
    if (rfElem[i]->matrix) {
      free_matrices(rfElem[i]->matrix);
      free(rfElem[i]->matrix);
      rfElem[i]->matrix = NULL;
    }
  }

  if (beamline->alpha[0] == 0)
    bombElegant("alphac == 0. rf_setup can't handle this.", NULL);

  eta = beamline->alpha[0] - 1 / sqr(gamma);

  if (rf_setup_struct.bucket_half_height > 0) {
    double F, E;

    E = sqrt(sqr(pCentral) + 1) * particleMassMV;
    F = sqr(rf_setup_struct.bucket_half_height) / (beamline->radIntegrals.Uo / (PI * fabs(eta) * harmonic * E));
    q = (F + 2) / 2;
    voltage = (q = solveForOverVoltage(F, q)) * beamline->radIntegrals.Uo * 1e6 / nRfElem;
  } else if (rf_setup_struct.over_voltage)
    voltage = (q = rf_setup_struct.over_voltage) * beamline->radIntegrals.Uo * 1e6 / nRfElem;
  else {
    voltage = rf_setup_struct.total_voltage / nRfElem;
    q = rf_setup_struct.total_voltage / (beamline->radIntegrals.Uo * 1e6);
  }

  phase = 0;
  if (voltage) {
    if (eta > 0)
      phase = 180 - asin(1 / q) * 180 / PI;
    else
      phase = asin(1 / q) * 180 / PI;
    printf("Voltage per cavity is %21.15e V, overvoltage is %21.15e, phase is %21.15e deg\n\n", voltage, q, phase);
    fflush(stdout);
    for (i = 0; i < nRfElem; i++) {
      if (rfElem[i]->type == T_RFCA) {
        rfca = (RFCA *)(rfElem[i]->p_elem);
        rfca->volt = voltage;
        rfca->phase = phase;
        rfca->fiducial_seen = 0;
        change_defined_parameter(rfElem[i]->name, iVoltRfca, T_RFCA, voltage, NULL, LOAD_FLAG_ABSOLUTE);
        change_used_parameter(beamline, rfElem[i]->name, iVoltRfca, T_RFCA, voltage, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
        change_defined_parameter(rfElem[i]->name, iPhaseRfca, T_RFCA, phase + rf_setup_struct.phase_offset, NULL, LOAD_FLAG_ABSOLUTE);
        change_used_parameter(beamline, rfElem[i]->name, iPhaseRfca, T_RFCA, phase + rf_setup_struct.phase_offset, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
      } else {
        modrf = (MODRF *)(rfElem[i]->p_elem);
        modrf->volt = voltage;
        modrf->phase = phase;
        modrf->fiducial_seen = 0;
        change_defined_parameter(rfElem[i]->name, iVoltModrf, T_MODRF, voltage, NULL, LOAD_FLAG_ABSOLUTE);
        change_used_parameter(beamline, rfElem[i]->name, iVoltModrf, T_MODRF, voltage, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
        change_defined_parameter(rfElem[i]->name, iPhaseModrf, T_MODRF, phase + rf_setup_struct.phase_offset, NULL, LOAD_FLAG_ABSOLUTE);
        change_used_parameter(beamline, rfElem[i]->name, iPhaseModrf, T_MODRF, phase + rf_setup_struct.phase_offset, NULL, LOAD_FLAG_ABSOLUTE | LOAD_FLAG_VERBOSE);
      }
      printf("Set phase of %s to %21.15e deg\n",
             rfElem[i]->name, phase + rf_setup_struct.phase_offset);
    }
  }

  rfAcceptance = -1;
  wrf = frf * PIx2;
  w0 = PIx2 / T0;
  h = wrf / w0 + 0.5;
  Vdot = nRfElem * h * w0 * voltage * cos(phase * PI / 180);
  E = sqrt(sqr(pCentral) + 1) * particleMassMV * 1e6;

  if ((eta * Vdot) < 0) {
    nus = sqrt(eta / PIx2 * (-Vdot / w0) / E);
    St0 = fabs(beamline->radIntegrals.sigmadelta * eta / (nus * w0));
    Sz0 = St0 * c_mks;
    if ((q = nRfElem * voltage / (1e6 * beamline->radIntegrals.Uo)) > 1)
      rfAcceptance = sqrt(2 * beamline->radIntegrals.Uo * 1e6 / (PI * fabs(eta) * h * E) * (sqrt(q * q - 1) - acos(1 / q)));
  } else {
    q = rfAcceptance = nus = St0 = Sz0 = -1;
  }

  rpn_store(Sz0, NULL, rpn_create_mem((char *)"Sz0", 0));
  rpn_store(St0, NULL, rpn_create_mem((char *)"St0", 0));
  rpn_store(frf, NULL, rpn_create_mem((char *)"RfFrequency", 0));
  rpn_store(beamline->radIntegrals.sigmadelta, NULL, rpn_create_mem((char *)"Sdelta0", 0));

  if (writeToFile && fpRf) {
    fprintf(fpRf, "%21.15le\n%21.15le\n%21.15le\n%21.15le\n%21.15le\n%21.15le\n%21.15le\n%ld\n",
            phase, voltage * nRfElem, rfAcceptance, nus, Sz0, St0, frf, h);
  }

  if (writeToFile && fpBucket) {
    /* Write the rf-bucket separatrix in (t, p) space.  The separatrix through the
       unstable fixed point phi_u = pi - phi_s of the longitudinal Hamiltonian is
          delta(phi)^2 = Vtot/(pi h E) * [cos(phi_u) - cos(phi) + (phi_u - phi) sin(phi_s)] / eta,
       whose peak equals the bucket half height (rfAcceptance) computed above.
       phi is converted to time via t = (phi - phi_s)/wrf (relative to the
       synchronous particle) and to momentum via p = pCentral*(1 + delta). */
    int64_t np = rf_setup_struct.bucket_points;
    long ip;
    double phi_s = phase * PI / 180.0;
    double phi_u = PI - phi_s;
    double sins = sin(phi_s);
    double Vtot = voltage * nRfElem;
    double A = Vtot / (PI * h * E); /* delta^2 = A*bracket(phi)/eta */
    double phi2, dphi, phi, d2, delta, tval, a, b, fa, fm, m;
    long k, nRows = 0;
#define RFBUCKET_BRACKET(ph) (cos(phi_u) - cos(ph) + (phi_u - (ph)) * sins)

    if (rfAcceptance > 0 && Vtot > 0) {
      /* find phi2, the second zero of the bracket (the far edge of the bucket),
         on the opposite side of phi_s from phi_u, by bisection */
      a = phi_s;
      b = phi_u + ((phi_s > phi_u) ? 1.0 : -1.0) * PIx2;
      fa = RFBUCKET_BRACKET(a);
      for (k = 0; k < 200; k++) {
        m = 0.5 * (a + b);
        fm = RFBUCKET_BRACKET(m);
        if ((fm > 0) == (fa > 0)) {
          a = m;
          fa = fm;
        } else
          b = m;
      }
      phi2 = 0.5 * (a + b);
      nRows = 2 * np;
    }

    fprintf(fpBucket, "%21.15le\n%ld\n%21.15le\n%21.15le\n%21.15le\n%ld\n",
            frf, h, Vtot, phase, rfAcceptance, nRows);
    if (nRows) {
      dphi = (phi2 - phi_u) / (np - 1);
      /* upper branch: phi_u -> phi2 with delta >= 0 */
      for (ip = 0; ip < np; ip++) {
        phi = phi_u + ip * dphi;
        d2 = A * RFBUCKET_BRACKET(phi) / eta;
        delta = (d2 > 0) ? sqrt(d2) : 0.0;
        tval = (phi - phi_s) / wrf;
        fprintf(fpBucket, "%21.15le %21.15le %21.15le\n", tval, pCentral * (1 + delta), delta);
      }
      /* lower branch: phi2 -> phi_u with delta <= 0 (closes the contour) */
      for (ip = np - 1; ip >= 0; ip--) {
        phi = phi_u + ip * dphi;
        d2 = A * RFBUCKET_BRACKET(phi) / eta;
        delta = (d2 > 0) ? -sqrt(d2) : 0.0;
        tval = (phi - phi_s) / wrf;
        fprintf(fpBucket, "%21.15le %21.15le %21.15le\n", tval, pCentral * (1 + delta), delta);
      }
    }
#undef RFBUCKET_BRACKET
  }
}

void SetSDrivingTermsRow(SDDS_DATASET *SDDSout, long i, long row, double position, const char *name, const char *type, long occurence, LINE_LIST *beamline) {
  if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, row,
                         "s", position,
                         "ElementName", name,
                         "ElementType", type,
                         "ElementOccurence", occurence,
                         "f10010", beamline->sDrivingTerms.f10010[i][0],
                         "f10100", beamline->sDrivingTerms.f10100[i][0],
                         "f30000", beamline->sDrivingTerms.f30000[i][0],
                         "f12000", beamline->sDrivingTerms.f12000[i][0],
                         "f10200", beamline->sDrivingTerms.f10200[i][0],
                         "f01200", beamline->sDrivingTerms.f01200[i][0],
                         "f01110", beamline->sDrivingTerms.f01110[i][0],
                         "f00300", beamline->sDrivingTerms.f00300[i][0],
                         "f00120", beamline->sDrivingTerms.f00120[i][0],
                         "f20100", beamline->sDrivingTerms.f20100[i][0],
                         "f20010", beamline->sDrivingTerms.f20010[i][0],
                         "f11010", beamline->sDrivingTerms.f11010[i][0],

                         "Ref10010", beamline->sDrivingTerms.f10010[i][1],
                         "Ref10100", beamline->sDrivingTerms.f10100[i][1],
                         "Ref30000", beamline->sDrivingTerms.f30000[i][1],
                         "Ref12000", beamline->sDrivingTerms.f12000[i][1],
                         "Ref10200", beamline->sDrivingTerms.f10200[i][1],
                         "Ref01200", beamline->sDrivingTerms.f01200[i][1],
                         "Ref01110", beamline->sDrivingTerms.f01110[i][1],
                         "Ref00300", beamline->sDrivingTerms.f00300[i][1],
                         "Ref00120", beamline->sDrivingTerms.f00120[i][1],
                         "Ref20100", beamline->sDrivingTerms.f20100[i][1],
                         "Ref20010", beamline->sDrivingTerms.f20010[i][1],
                         "Ref11010", beamline->sDrivingTerms.f11010[i][1],

                         "Imf10010", beamline->sDrivingTerms.f10010[i][2],
                         "Imf10100", beamline->sDrivingTerms.f10100[i][2],
                         "Imf30000", beamline->sDrivingTerms.f30000[i][2],
                         "Imf12000", beamline->sDrivingTerms.f12000[i][2],
                         "Imf10200", beamline->sDrivingTerms.f10200[i][2],
                         "Imf01200", beamline->sDrivingTerms.f01200[i][2],
                         "Imf01110", beamline->sDrivingTerms.f01110[i][2],
                         "Imf00300", beamline->sDrivingTerms.f00300[i][2],
                         "Imf00120", beamline->sDrivingTerms.f00120[i][2],
                         "Imf20100", beamline->sDrivingTerms.f20100[i][2],
                         "Imf20010", beamline->sDrivingTerms.f20010[i][2],
                         "Imf11010", beamline->sDrivingTerms.f11010[i][2],
                         NULL)) {
    SDDS_SetError((char *)"Problem setting SDDS rows (s_dependent_driving_terms_file)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
}

long determineScraperAperture(long plane, unsigned long direction, double position,
                              double offset, double centroid, double *apertureRet,
                              short scraperConvention) {
  double a1 = coordLimit, a2 = -coordLimit;
  unsigned long flagPlus, flagMinus, flags;
  double aperture;
  long aperture_set = 0;

  flagPlus = plane ? DIRECTION_PLUS_Y : DIRECTION_PLUS_X;
  flagMinus = plane ? DIRECTION_MINUS_Y : DIRECTION_MINUS_X;
  flags = flagPlus + flagMinus;
  *apertureRet = 0;
  centroid -= offset;
  if (direction & flags) {
    if ((direction & flags) == flags) {
      /* inserted from both sides */
      a1 = a2 = 0;
      if (!scraperConvention)
        position = fabs(position);
      if (position < centroid)
        a1 = 0;
      else if (centroid < -position)
        a2 = 0;
      else {
        a1 = position - centroid;
        a2 = position + centroid;
      }
      aperture = MIN(a1, a2);
    } else if (direction & flagPlus) {
      /* inserted from positive side */
      if (centroid > position)
        aperture = 0;
      else
        aperture = position - centroid;
    } else {
      /* inserted from negative side */
      position *= scraperConvention ? -1 : 1;
      centroid *= -1;
      if (centroid > position)
        aperture = 0;
      else
        aperture = position - centroid;
    }
    *apertureRet = aperture;
    aperture_set = 1;
  }
  return aperture_set;
}

double effectiveEllipticalAperture(double a, double b, double x, double y) {
  double aperture = 0, position = 0;
  x = fabs(x);
  y = fabs(y);
  if (a > 0 && x > a)
    return 0;
  if (b > 0 && y > b)
    return 0;
  if (a > 0) {
    if (b > 0) {
      if ((sqr(x / a) + sqr(y / b)) >= 1)
        return 0;
      if ((position = sqr(a) * (1 - sqr(y / b))) < 0)
        return 0;
      position = sqrt(position);
    } else
      position = a;
    if ((aperture = position - x) < 0)
      aperture = 0;
  }
  return aperture;
}

void findChamberShapes(ELEMENT_LIST *elem) {
  short code;
  MAXAMP *maxamp;
  ECOL *ecol;
  TAPERAPE *tape;

  code = UNKNOWN_CHAMBER;
  while (elem) {
    switch (elem->type) {
    case T_MAXAMP:
      maxamp = (MAXAMP *)elem->p_elem;
      if (maxamp->elliptical) {
        if (maxamp->exponent <= 2 && maxamp->yExponent <= 2) {
          if (maxamp->x_max == maxamp->y_max)
            code = ROUND_CHAMBER;
          else
            code = ELLIPTICAL_CHAMBER;
        } else
          code = SUPERELLIPTICAL_CHAMBER;
      } else
        code = RECTANGULAR_CHAMBER;
      break;
    case T_RCOL:
      code = RECTANGULAR_CHAMBER;
      break;
    case T_ECOL:
      ecol = (ECOL *)elem->p_elem;
      if (ecol->exponent <= 2 && ecol->yExponent <= 2) {
        if (ecol->x_max == ecol->y_max)
          code = ROUND_CHAMBER;
        else
          code = ELLIPTICAL_CHAMBER;
      } else
        code = RECTANGULAR_CHAMBER;
      break;
    case T_TAPERAPR:
      code = RECTANGULAR_CHAMBER;
      break;
    case T_TAPERAPC:
      code = ROUND_CHAMBER;
      break;
    case T_TAPERAPE:
      tape = (TAPERAPE *)elem->p_elem;
      if (tape->xExponent <= 2 && tape->yExponent <= 2)
        code = ELLIPTICAL_CHAMBER;
      else
        code = SUPERELLIPTICAL_CHAMBER;
      break;
    default:
      break;
    }
    elem->chamberShape = code;
    elem = elem->succ;
  }
}
