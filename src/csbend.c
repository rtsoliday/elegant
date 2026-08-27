/************************************************************************* \
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: csbend.c
 * contents:  track_through_canonical_sbend()
 *
 *
 * Michael Borland, 1991, 1992.
 */
#include "mdb.h"
#include "track.h"
#include "csbend.h"
#ifdef HAVE_GPU
#  include "gpu_base.h"
#  include "gpu_csbend.h"
#  include "gpu_funcs.h"
#endif
#ifdef _OPENMP
#  include <omp.h>
#endif

#if !defined(HAVE_GPU)
#define VT static
#else
#define VT
#endif

/* global variables */
VT long expansionOrder1 = 11; /* order of expansion+1 */
VT long hasSkew = 0, hasNormal = 0;
VT double rho0, rho_actual, rad_coef = 0, isrConstant = 0;
VT double meanPhotonsPerRadian0, meanPhotonsPerMeter0, normalizedCriticalEnergy0;
VT long distributionBasedRadiation, includeOpeningAngle;
VT long photonCount = 0;
VT double energyCount = 0, radiansTotal = 0;
VT double **Fx_xy = NULL;
VT double **Fy_xy = NULL;
VT short refTrajectoryMode = 0;
VT long refTrajectoryPoints = 0;
VT double **refTrajectoryData = NULL;
#ifdef HAVE_GPU
static long csrGpuCsbendInnerCall = 0;
static long csrGpuCsbendDeviceEntryRequested = 0;
#endif

#define VTS static

void convolveArrays1(double *output, long n, double *a1, double *a2);
static long prepareParticleCoordinateHistogram(double **hist, long *maxBins,
                                               double *lower, double *upper,
                                               double *binSize, long *bins,
                                               double expansionFactor,
                                               double **particleCoord,
                                               long nParticles,
                                               long coordinateIndex);
VTS void dipoleFringeKHwang(double *Qf, double *Qi,
                        double rho, double inFringe, long higherOrder, double K1, double edge, double gap,
                        double fint, double Rhe);
VTS void dipoleFringeKHwangRLindberg(double *Qf, double *Qi,
                                 double rho, double inFringe, double K1, double edge,
                                 double gap, double fint, double Rhe);
VTS void curvedDipoleFringe(double *Qf, double *Qi, double rho, long inFringe, long edgeOrder, double K1, double edge,
                            double *integrals, unsigned short edgeFlip);

VTS void addRadiationKick(double *Qx, double *Qy, double *dPoP, double *sigmaDelta2,
                      double x, double y, double theta, double thetaf, double h0, double Fx, double Fy,
                      double ds, double radCoef, double dsISR, double isrCoef,
                      long distributionBased, long includeOpeningAngle,
                      double meanPhotonsPerMeter,
                      double normalizedCriticalEnergy, double Po);
VTS double pickNormalizedPhotonEnergy(double RN);

VTS long integrate_csbend_ordn(double *Qf, double *Qi, double *sigmaDelta2, double *S, double s, long n, long i, double rho0, double p0,
                           double *dz_lost, MULT_APERTURE_DATA *apData, short integration_order, ELEMENT_LIST *eptr);
VTS long integrate_csbend_ordn_expanded(double *Qf, double *Qi, double *sigmaDelta2, double s, long n, long i, double rho0, double p0,
                                    double *dz_lost, MULT_APERTURE_DATA *apData, short integration_order, ELEMENT_LIST *eptr);
VTS void convertFromCSBendCoords(double **part, int64_t np, double rho0,
                             double cos_ttilt, double sin_ttilt, long ctMode);
VTS void convertToCSBendCoords(double **part, int64_t np, double rho0,
                           double cos_ttilt, double sin_ttilt, long ctMode);
void applyFilterTable(double *function, long bins, double dt, long fValues,
                      double *fFreq, double *fReal, double *fImag);

long correctDistribution(double *array, long npoints, double desiredSum);

VTS void convertToDipoleCanonicalCoordinates(double *Qi, long expanded);
VTS void convertFromDipoleCanonicalCoordinates(double *Qi, long expanded);

VTS long inversePoissonCDF(double mu, double C);

VTS void setUpCsbendPhotonOutputFile(CSBEND *csbend, char *rootname, int64_t np);
VTS void logPhoton(double Ep, double x, double xp, double y, double yp, double theta, double thetaf, double rho);
VTS SDDS_DATASET *SDDSphotons;
VTS long photonRows;
VTS double photonLowEnergyCutoff;

#define RECORD_TRAJECTORY 1
#define SUBTRACT_TRAJECTORY 2

VTS void applySimpleDipoleEdgeKick(double *xp, double *yp, double x, double y, double delta, double rho, double ea,
                               double psi, double kickLimit, long expanded);

//VTS void computeCSBENDFields(double *Fx, double *Fy, double x, double y);
void computeCSBENDFieldCoefficients(double *b, double *c, double h1, long nonlinear, long expansionOrder);


static void computeCSBENDFields(double *restrict Fx, double *restrict Fy, const double x, const double y) {
  double yp[11];
  double sumFx = 0, sumFy = 0;
  long i, j;

  // B[xy] is H*F[xy]/rho0
  
  if (!hasSkew && !hasNormal) {
    *Fx = 0;
    *Fy = 1;
    return;
  }

  yp[0] = 1;
  for (i = 1; i < expansionOrder1; i++) {
    yp[i] = yp[i - 1] * y;
  }

  // This change is needed to specialize loops to help autovectorizer
  // Can debug with -ftree-vectorize -ftree-vectorizer-verbose=4 -fopt-info-vec-missed
  // Easier to use godbolt with bite sized pieces, like in https://godbolt.org/z/nMdKP488z

  /* Note: using expansionOrder-i here ensures that for x^i*y^j , i+j<=(expansionOrder1-1) */

  double xt = 1;
  if (hasSkew) {
    //j0=0,dj=1
    for (i = 0; i < expansionOrder1; i++) {
      for (j = 0; j < expansionOrder1 - i; j += 1) {
#if TURBO_RECIPROCALS
        sumFx += *(*Fx_xy + i*11 + j) * (xt * yp[j]);
        sumFy += *(*Fy_xy + i*11 + j) * (xt * yp[j]);
#else
        sumFx += *(*Fx_xy + i*11 + j) * xt * yp[j];
        sumFy += *(*Fy_xy + i*11 + j) * xt * yp[j];
#endif
      }
      xt *= x;
    }
  } else {
//    //j0=1,dj=2
//    for (i = 0; i < expansionOrder1; i++) {
//      for (j = 1; j < expansionOrder1 - i; j += 2)
//        sumFx += *(*Fx_xy + i*11 + j) * xt * yp[j];
//      for (j = 0; j < expansionOrder1 - i; j += 2)
//        sumFy += *(*Fy_xy + i*11 + j) * xt * yp[j];
//      xt *= x;
//    }
    //j0=1,dj=2
    for (i = 0; i < expansionOrder1; i++) {
      for (j = 0; j < expansionOrder1 - i; j++) {
#if TURBO_RECIPROCALS
        if (j & 1) {
          sumFx += *(*Fx_xy + i*11 + j) * (xt * yp[j]);
        } else {
          sumFy += *(*Fy_xy + i*11 + j) * (xt * yp[j]);
        }
#else
        if (j & 1) {
          sumFx += *(*Fx_xy + i*11 + j) * xt * yp[j];
        } else {
          sumFy += *(*Fy_xy + i*11 + j) * xt * yp[j];
        }
//        if (j & 1) {
//          sumFx += Fx_xy[i][j] * xt * yp[j];
//        } else {
//          sumFy += Fy_xy[i][j] * xt * yp[j];
//        }
#endif
      }
      xt *= x;
    }
  }
  *Fx = sumFx;
  *Fy = sumFy;
}

#ifdef HAVE_GPU
typedef struct {
  CSBEND *csbend;
  MULT_APERTURE_DATA *apertureData;
  ELEMENT_LIST *eptr;
  double Po;
  double zStart;
  double rho0;
  double rhoActual;
  double e1;
  double e2;
  double cosTilt;
  double sinTilt;
} CSBEND_OMP_TRACK_DATA;

/*
 * Deterministic, full-element CSBEND particle tracking for the batched tune
 * path. Beam-wide setup and misalignment transforms remain in
 * track_through_csbend(). Losses are flagged here and compacted in original
 * particle order after the parallel region.
 */
static long trackCsbendParticleOmp(double *coord,
                                   const CSBEND_OMP_TRACK_DATA *data,
                                   double *dzLost) {
  CSBEND *csbend = data->csbend;
  double Qi[MAX_PROPERTIES_PER_PARTICLE];
  double Qf[MAX_PROPERTIES_PER_PARTICLE];
  double x = coord[0], xp = coord[1];
  double y = coord[2], yp = coord[3];
  double s = coord[4], dp = coord[5];
  double *singleParticle = coord;
  long particleLost;

  if (csbend->edgeFlags & BEND_EDGE1_EFFECTS) {
    Qi[0] = x;
    Qi[1] = xp;
    Qi[2] = y;
    Qi[3] = yp;
    Qi[4] = 0;
    Qi[5] = dp;
    convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
    curvedDipoleFringe(
      Qf, Qi, data->rhoActual, -1, csbend->edge_order,
      csbend->b[1] / data->rho0, data->e1,
      csbend->fringeInt[csbend->e1Index], csbend->edgeFlip);
    convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
    x = Qf[0];
    xp = Qf[1];
    y = Qf[2];
    yp = Qf[3];
    s += Qf[4];
    dp = Qf[5];
  }

  Qi[0] = x;
  Qi[1] = xp;
  Qi[2] = y;
  Qi[3] = yp;
  Qi[4] = 0;
  Qi[5] = dp;
  convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
  if (csbend->expandHamiltonian)
    particleLost = !integrate_csbend_ordn_expanded(
      Qf, Qi, NULL, csbend->length, csbend->nSlices, -1, data->rho0,
      data->Po, dzLost, data->apertureData, csbend->integration_order,
      data->eptr);
  else
    particleLost = !integrate_csbend_ordn(
      Qf, Qi, NULL, NULL, csbend->length, csbend->nSlices, -1,
      data->rho0, data->Po, dzLost, data->apertureData,
      csbend->integration_order, data->eptr);

  if (csbend->fseCorrection == 1)
    Qf[4] -= csbend->fseCorrectionPathError;
  convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
  if (particleLost) {
    memcpy(coord, Qf, sizeof(*coord) * 6);
    convertFromCSBendCoords(
      &singleParticle, 1, data->rho0, data->cosTilt, data->sinTilt, 0);
    coord[4] = data->zStart + *dzLost;
    coord[5] = data->Po * (1 + coord[5]);
    return 0;
  }

  s += Qf[4];
  x = Qf[0];
  xp = Qf[1];
  y = Qf[2];
  yp = Qf[3];
  dp = Qf[5];

  if (csbend->edgeFlags & BEND_EDGE2_EFFECTS) {
    Qi[0] = x;
    Qi[1] = xp;
    Qi[2] = y;
    Qi[3] = yp;
    Qi[4] = 0;
    Qi[5] = dp;
    convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
    curvedDipoleFringe(
      Qf, Qi, data->rhoActual, 1, csbend->edge_order,
      csbend->b[1] / data->rho0, data->e2,
      csbend->fringeInt[csbend->e2Index], csbend->edgeFlip);
    convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
    x = Qf[0];
    xp = Qf[1];
    y = Qf[2];
    yp = Qf[3];
    s += Qf[4];
    dp = Qf[5];
  }

  coord[0] = x;
  coord[1] = xp;
  coord[2] = y;
  coord[3] = yp;
  coord[4] = s;
  coord[5] = dp;
  return 1;
}
#endif


void computeCSBENDFieldCoefficients(double *b, double *c,
                                    double h1, const long nonlinear, long expansionOrder) {
  long i;
  double h[20];

  if (expansionOrder == 0) {
    /* set the order to be <highestMultipole>+2 */
    for (i = 8; i >= 0; i--)
      if (b[i] || c[i])
        break;
    if ((expansionOrder = i + 2) < 4)
      /* make minimum value 4 for backward compatibility */
      expansionOrder = 4;
  }

  expansionOrder1 = expansionOrder + 1;
  if (expansionOrder1 > 11)
    bombElegant("expansion order >10 for CSBEND or CSRCSBEND", NULL);

  hasSkew = hasNormal = 0;
  for (i = 0; i < 9; i++) {
    if (b[i])
      hasNormal = 1;
    if (c[i])
      hasSkew = 1;
  }

  if (!Fx_xy)
    Fx_xy = (double **)czarray_2d(sizeof(double), 11, 11);
  if (!Fy_xy)
    Fy_xy = (double **)czarray_2d(sizeof(double), 11, 11);

  for (i = 0; i < expansionOrder1; i++) {
    memset(Fx_xy[i], 0, expansionOrder1 * sizeof(double));
    memset(Fy_xy[i], 0, expansionOrder1 * sizeof(double));
  }

  h[0] = 1;
  for (i = 1; i < 20; i++)
    h[i] = h[i - 1] * h1;

  Fx_xy[0][0] = c[0];
  Fy_xy[0][0] = 1 - b[0];

  /* these increments allow using the previous indexing from when c[0] and b[0] where the quadrupole etc. */
  b += 1;
  c += 1;

  Fx_xy[0][1] = b[0];
  Fy_xy[1][0] = b[0];
  Fy_xy[0][1] = c[0];
  Fx_xy[1][0] = -c[0];

  if (nonlinear) {
    Fy_xy[0][2] = -(h[1] * b[0]) / 2 - b[1] / 2;
    Fy_xy[0][3] = (h[2] * c[0]) / 6 - (h[1] * c[1]) / 3 - c[2] / 6;
    Fy_xy[0][4] = (h[3] * b[0]) / 24 - (h[2] * b[1]) / 24 + (h[1] * b[2]) / 12 + b[3] / 24;
    Fy_xy[0][5] = (-3 * h[4] * c[0]) / 40 + (h[3] * c[1]) / 20 - (h[2] * c[2]) / 40 + (h[1] * c[3]) / 40 + c[4] / 120;
    Fy_xy[0][6] = -(h[5] * b[0]) / 80 + (h[4] * b[1]) / 80 - (h[3] * b[2]) / 120 + (h[2] * b[3]) / 240 - (h[1] * b[4]) / 240 - b[5] / 720;
    Fy_xy[0][7] = (5 * h[6] * c[0]) / 112 - (h[5] * c[1]) / 40 + (17 * h[4] * c[2]) / 1680 - (h[3] * c[3]) / 280 + (h[2] * c[4]) / 840 -
                  (h[1] * c[5]) / 1260 - c[6] / 5040;
    Fy_xy[0][8] = (5 * h[7] * b[0]) / 896 - (5 * h[6] * b[1]) / 896 + (h[5] * b[2]) / 320 - (17 * h[4] * b[3]) / 13440 + (h[3] * b[4]) / 2240 -
                  (h[2] * b[5]) / 6720 + (h[1] * b[6]) / 10080 + b[7] / 40320;
    Fy_xy[0][9] = (-35 * h[8] * c[0]) / 1152 + (65 * h[7] * c[1]) / 4032 - (145 * h[6] * c[2]) / 24192 + (43 * h[5] * c[3]) / 24192 -
                  (11 * h[4] * c[4]) / 24192 + (h[3] * c[5]) / 9072 - (h[2] * c[6]) / 36288 + (h[1] * c[7]) / 72576;
    Fy_xy[0][10] = (-7 * h[9] * b[0]) / 2304 + (7 * h[8] * b[1]) / 2304 - (13 * h[7] * b[2]) / 8064 + (29 * h[6] * b[3]) / 48384 -
                   (43 * h[5] * b[4]) / 241920 + (11 * h[4] * b[5]) / 241920 - (h[3] * b[6]) / 90720 + (h[2] * b[7]) / 362880;
    Fy_xy[1][1] = h[1] * c[0] + c[1];
    Fy_xy[1][2] = (h[2] * b[0]) / 2 - (h[1] * b[1]) / 2 - b[2] / 2;
    Fy_xy[1][3] = -(h[3] * c[0]) / 2 + (h[2] * c[1]) / 2 - (h[1] * c[2]) / 3 - c[3] / 6;
    Fy_xy[1][4] = -(h[4] * b[0]) / 8 + (h[3] * b[1]) / 8 - (h[2] * b[2]) / 8 + (h[1] * b[3]) / 12 + b[4] / 24;
    Fy_xy[1][5] = (3 * h[5] * c[0]) / 8 - (9 * h[4] * c[1]) / 40 + (h[3] * c[2]) / 10 - (h[2] * c[3]) / 20 + (h[1] * c[4]) / 40 + c[5] / 120;
    Fy_xy[1][6] = (h[6] * b[0]) / 16 - (h[5] * b[1]) / 16 + (3 * h[4] * b[2]) / 80 - (h[3] * b[3]) / 60 + (h[2] * b[4]) / 120 - (h[1] * b[5]) / 240 -
                  b[6] / 720;
    Fy_xy[1][7] = (-5 * h[7] * c[0]) / 16 + (19 * h[6] * c[1]) / 112 - (11 * h[5] * c[2]) / 168 + (h[4] * c[3]) / 48 - (h[3] * c[4]) / 168 +
                  (h[2] * c[5]) / 504 - (h[1] * c[6]) / 1260 - c[7] / 5040;
    Fy_xy[1][8] = (-5 * h[8] * b[0]) / 128 + (5 * h[7] * b[1]) / 128 - (19 * h[6] * b[2]) / 896 + (11 * h[5] * b[3]) / 1344 - (h[4] * b[4]) / 384 +
                  (h[3] * b[5]) / 1344 - (h[2] * b[6]) / 4032 + (h[1] * b[7]) / 10080;
    Fy_xy[1][9] = (35 * h[9] * c[0]) / 128 - (55 * h[8] * c[1]) / 384 + (5 * h[7] * c[2]) / 96 - (5 * h[6] * c[3]) / 336 + (29 * h[5] * c[4]) / 8064 -
                  (19 * h[4] * c[5]) / 24192 + (h[3] * c[6]) / 6048 - (h[2] * c[7]) / 24192;
    Fy_xy[1][10] = (7 * h[10] * b[0]) / 256 - (7 * h[9] * b[1]) / 256 + (11 * h[8] * b[2]) / 768 - (h[7] * b[3]) / 192 + (h[6] * b[4]) / 672 -
                   (29 * h[5] * b[5]) / 80640 + (19 * h[4] * b[6]) / 241920 - (h[3] * b[7]) / 60480;
    Fy_xy[2][0] = b[1] / 2;
    Fy_xy[2][1] = -(h[2] * c[0]) + (h[1] * c[1]) / 2 + c[2] / 2;
    Fy_xy[2][2] = -(h[3] * b[0]) / 2 + (h[2] * b[1]) / 2 - (h[1] * b[2]) / 4 - b[3] / 4;
    Fy_xy[2][3] = h[4] * c[0] - (3 * h[3] * c[1]) / 4 + (5 * h[2] * c[2]) / 12 - (h[1] * c[3]) / 6 - c[4] / 12;
    Fy_xy[2][4] = (h[5] * b[0]) / 4 - (h[4] * b[1]) / 4 + (3 * h[3] * b[2]) / 16 - (5 * h[2] * b[3]) / 48 + (h[1] * b[4]) / 24 + b[5] / 48;
    Fy_xy[2][5] = (-9 * h[6] * c[0]) / 8 + (51 * h[5] * c[1]) / 80 - (21 * h[4] * c[2]) / 80 + (h[3] * c[3]) / 10 - (3 * h[2] * c[4]) / 80 +
                  (h[1] * c[5]) / 80 + c[6] / 240;
    Fy_xy[2][6] = (-3 * h[7] * b[0]) / 16 + (3 * h[6] * b[1]) / 16 - (17 * h[5] * b[2]) / 160 + (7 * h[4] * b[3]) / 160 - (h[3] * b[4]) / 60 +
                  (h[2] * b[5]) / 160 - (h[1] * b[6]) / 480 - b[7] / 1440;
    Fy_xy[2][7] = (5 * h[8] * c[0]) / 4 - (149 * h[7] * c[1]) / 224 + (167 * h[6] * c[2]) / 672 - (25 * h[5] * c[3]) / 336 + (13 * h[4] * c[4]) / 672 -
                  (5 * h[3] * c[5]) / 1008 + (h[2] * c[6]) / 720 - (h[1] * c[7]) / 2520;
    Fy_xy[2][8] = (5 * h[9] * b[0]) / 32 - (5 * h[8] * b[1]) / 32 + (149 * h[7] * b[2]) / 1792 - (167 * h[6] * b[3]) / 5376 +
                  (25 * h[5] * b[4]) / 2688 - (13 * h[4] * b[5]) / 5376 + (5 * h[3] * b[6]) / 8064 - (h[2] * b[7]) / 5760;
    Fy_xy[2][9] = (-175 * h[10] * c[0]) / 128 + (545 * h[9] * c[1]) / 768 - (65 * h[8] * c[2]) / 256 + (95 * h[7] * c[3]) / 1344 -
                  (265 * h[6] * c[4]) / 16128 + (163 * h[5] * c[5]) / 48384 - (31 * h[4] * c[6]) / 48384 + (h[3] * c[7]) / 8064;
    Fy_xy[2][10] = (-35 * h[11] * b[0]) / 256 + (35 * h[10] * b[1]) / 256 - (109 * h[9] * b[2]) / 1536 + (13 * h[8] * b[3]) / 512 -
                   (19 * h[7] * b[4]) / 2688 + (53 * h[6] * b[5]) / 32256 - (163 * h[5] * b[6]) / 483840 + (31 * h[4] * b[7]) / 483840;
    Fy_xy[3][0] = b[2] / 6;
    Fy_xy[3][1] = h[3] * c[0] - (h[2] * c[1]) / 2 + (h[1] * c[2]) / 6 + c[3] / 6;
    Fy_xy[3][2] = (h[4] * b[0]) / 2 - (h[3] * b[1]) / 2 + (h[2] * b[2]) / 4 - (h[1] * b[3]) / 12 - b[4] / 12;
    Fy_xy[3][3] = (-5 * h[5] * c[0]) / 3 + (13 * h[4] * c[1]) / 12 - (19 * h[3] * c[2]) / 36 + (7 * h[2] * c[3]) / 36 - (h[1] * c[4]) / 18 - c[5] / 36;
    Fy_xy[3][4] = (-5 * h[6] * b[0]) / 12 + (5 * h[5] * b[1]) / 12 - (13 * h[4] * b[2]) / 48 + (19 * h[3] * b[3]) / 144 - (7 * h[2] * b[4]) / 144 +
                  (h[1] * b[5]) / 72 + b[6] / 144;
    Fy_xy[3][5] = (21 * h[7] * c[0]) / 8 - (23 * h[6] * c[1]) / 16 + (9 * h[5] * c[2]) / 16 - (3 * h[4] * c[3]) / 16 + (7 * h[3] * c[4]) / 120 -
                  (h[2] * c[5]) / 60 + (h[1] * c[6]) / 240 + c[7] / 720;
    Fy_xy[3][6] = (7 * h[8] * b[0]) / 16 - (7 * h[7] * b[1]) / 16 + (23 * h[6] * b[2]) / 96 - (3 * h[5] * b[3]) / 32 + (h[4] * b[4]) / 32 -
                  (7 * h[3] * b[5]) / 720 + (h[2] * b[6]) / 360 - (h[1] * b[7]) / 1440;
    Fy_xy[3][7] = (-15 * h[9] * c[0]) / 4 + (63 * h[8] * c[1]) / 32 - (23 * h[7] * c[2]) / 32 + (139 * h[6] * c[3]) / 672 - (17 * h[5] * c[4]) / 336 +
                  (23 * h[4] * c[5]) / 2016 - (13 * h[3] * c[6]) / 5040 + (h[2] * c[7]) / 1680;
    Fy_xy[3][8] = (-15 * h[10] * b[0]) / 32 + (15 * h[9] * b[1]) / 32 - (63 * h[8] * b[2]) / 256 + (23 * h[7] * b[3]) / 256 -
                  (139 * h[6] * b[4]) / 5376 + (17 * h[5] * b[5]) / 2688 - (23 * h[4] * b[6]) / 16128 + (13 * h[3] * b[7]) / 40320;
    Fy_xy[3][9] = (1925 * h[11] * c[0]) / 384 - (1985 * h[10] * c[1]) / 768 + (2105 * h[9] * c[2]) / 2304 - (575 * h[8] * c[3]) / 2304 +
                  (65 * h[7] * c[4]) / 1152 - (115 * h[6] * c[5]) / 10368 + (41 * h[5] * c[6]) / 20736 - (7 * h[4] * c[7]) / 20736;
    Fy_xy[3][10] = (385 * h[12] * b[0]) / 768 - (385 * h[11] * b[1]) / 768 + (397 * h[10] * b[2]) / 1536 - (421 * h[9] * b[3]) / 4608 +
                   (115 * h[8] * b[4]) / 4608 - (13 * h[7] * b[5]) / 2304 + (23 * h[6] * b[6]) / 20736 - (41 * h[5] * b[7]) / 207360;
    Fy_xy[4][0] = b[3] / 24;
    Fy_xy[4][1] = -(h[4] * c[0]) + (h[3] * c[1]) / 2 - (h[2] * c[2]) / 6 + (h[1] * c[3]) / 24 + c[4] / 24;
    Fy_xy[4][2] = -(h[5] * b[0]) / 2 + (h[4] * b[1]) / 2 - (h[3] * b[2]) / 4 + (h[2] * b[3]) / 12 - (h[1] * b[4]) / 48 - b[5] / 48;
    Fy_xy[4][3] = (5 * h[6] * c[0]) / 2 - (3 * h[5] * c[1]) / 2 + (2 * h[4] * c[2]) / 3 - (11 * h[3] * c[3]) / 48 + (h[2] * c[4]) / 16 - (h[1] * c[5]) / 72 -
                  c[6] / 144;
    Fy_xy[4][4] = (5 * h[7] * b[0]) / 8 - (5 * h[6] * b[1]) / 8 + (3 * h[5] * b[2]) / 8 - (h[4] * b[3]) / 6 + (11 * h[3] * b[4]) / 192 -
                  (h[2] * b[5]) / 64 + (h[1] * b[6]) / 288 + b[7] / 576;
    Fy_xy[4][5] = (-21 * h[8] * c[0]) / 4 + (45 * h[7] * c[1]) / 16 - (17 * h[6] * c[2]) / 16 + (21 * h[5] * c[3]) / 64 - (29 * h[4] * c[4]) / 320 +
                  (11 * h[3] * c[5]) / 480 - (h[2] * c[6]) / 192 + (h[1] * c[7]) / 960;
    Fy_xy[4][6] = (-7 * h[9] * b[0]) / 8 + (7 * h[8] * b[1]) / 8 - (15 * h[7] * b[2]) / 32 + (17 * h[6] * b[3]) / 96 - (7 * h[5] * b[4]) / 128 +
                  (29 * h[4] * b[5]) / 1920 - (11 * h[3] * b[6]) / 2880 + (h[2] * b[7]) / 1152;
    Fy_xy[4][7] = (75 * h[10] * c[0]) / 8 - (39 * h[9] * c[1]) / 8 + (7 * h[8] * c[2]) / 4 - (439 * h[7] * c[3]) / 896 + (103 * h[6] * c[4]) / 896 -
                  (97 * h[5] * c[5]) / 4032 + (193 * h[4] * c[6]) / 40320 - (19 * h[3] * c[7]) / 20160;
    Fy_xy[4][8] = (75 * h[11] * b[0]) / 64 - (75 * h[10] * b[1]) / 64 + (39 * h[9] * b[2]) / 64 - (7 * h[8] * b[3]) / 32 + (439 * h[7] * b[4]) / 7168 -
                  (103 * h[6] * b[5]) / 7168 + (97 * h[5] * b[6]) / 32256 - (193 * h[4] * b[7]) / 322560;
    Fy_xy[4][9] = (-1925 * h[12] * c[0]) / 128 + (1975 * h[11] * c[1]) / 256 - (2075 * h[10] * c[2]) / 768 + (745 * h[9] * c[3]) / 1024 -
                  (165 * h[8] * c[4]) / 1024 + (425 * h[7] * c[5]) / 13824 - (145 * h[6] * c[6]) / 27648 + (23 * h[5] * c[7]) / 27648;
    Fy_xy[4][10] = (-385 * h[13] * b[0]) / 256 + (385 * h[12] * b[1]) / 256 - (395 * h[11] * b[2]) / 512 + (415 * h[10] * b[3]) / 1536 -
                   (149 * h[9] * b[4]) / 2048 + (33 * h[8] * b[5]) / 2048 - (85 * h[7] * b[6]) / 27648 + (29 * h[6] * b[7]) / 55296;
    Fy_xy[5][0] = b[4] / 120;
    Fy_xy[5][1] = h[5] * c[0] - (h[4] * c[1]) / 2 + (h[3] * c[2]) / 6 - (h[2] * c[3]) / 24 + (h[1] * c[4]) / 120 + c[5] / 120;
    Fy_xy[5][2] = (h[6] * b[0]) / 2 - (h[5] * b[1]) / 2 + (h[4] * b[2]) / 4 - (h[3] * b[3]) / 12 + (h[2] * b[4]) / 48 - (h[1] * b[5]) / 240 -
                  b[6] / 240;
    Fy_xy[5][3] = (-7 * h[7] * c[0]) / 2 + 2 * h[6] * c[1] - (5 * h[5] * c[2]) / 6 + (13 * h[4] * c[3]) / 48 - (17 * h[3] * c[4]) / 240 +
                  (11 * h[2] * c[5]) / 720 - (h[1] * c[6]) / 360 - c[7] / 720;
    Fy_xy[5][4] = (-7 * h[8] * b[0]) / 8 + (7 * h[7] * b[1]) / 8 - (h[6] * b[2]) / 2 + (5 * h[5] * b[3]) / 24 - (13 * h[4] * b[4]) / 192 +
                  (17 * h[3] * b[5]) / 960 - (11 * h[2] * b[6]) / 2880 + (h[1] * b[7]) / 1440;
    Fy_xy[5][5] = (189 * h[9] * c[0]) / 20 - (399 * h[8] * c[1]) / 80 + (147 * h[7] * c[2]) / 80 - (173 * h[6] * c[3]) / 320 +
                  (221 * h[5] * c[4]) / 1600 - (51 * h[4] * c[5]) / 1600 + (h[3] * c[6]) / 150 - (h[2] * c[7]) / 800;
    Fy_xy[5][6] = (63 * h[10] * b[0]) / 40 - (63 * h[9] * b[1]) / 40 + (133 * h[8] * b[2]) / 160 - (49 * h[7] * b[3]) / 160 +
                  (173 * h[6] * b[4]) / 1920 - (221 * h[5] * b[5]) / 9600 + (17 * h[4] * b[6]) / 3200 - (h[3] * b[7]) / 900;
    Fy_xy[5][7] = (-165 * h[11] * c[0]) / 8 + (213 * h[10] * c[1]) / 20 - (151 * h[9] * c[2]) / 40 + (663 * h[8] * c[3]) / 640 -
                  (151 * h[7] * c[4]) / 640 + (271 * h[6] * c[5]) / 5760 - (871 * h[5] * c[6]) / 100800 + (307 * h[4] * c[7]) / 201600;
    Fy_xy[5][8] = (-165 * h[12] * b[0]) / 64 + (165 * h[11] * b[1]) / 64 - (213 * h[10] * b[2]) / 160 + (151 * h[9] * b[3]) / 320 -
                  (663 * h[8] * b[4]) / 5120 + (151 * h[7] * b[5]) / 5120 - (271 * h[6] * b[6]) / 46080 + (871 * h[5] * b[7]) / 806400;
    Fy_xy[5][9] = (-13299 * h[13] * c[0]) / 128 + (13189 * h[12] * c[1]) / 256 - (4323 * h[11] * c[2]) / 256 + (4207 * h[10] * c[3]) / 1024 -
                  (12109 * h[9] * c[4]) / 15360 + (17051 * h[8] * c[5]) / 138240 - (1927 * h[7] * c[6]) / 120960 + (403 * h[6] * c[7]) / 241920;
    Fy_xy[5][10] = (-13299 * h[14] * b[0]) / 1280 + (13299 * h[13] * b[1]) / 1280 - (13189 * h[12] * b[2]) / 2560 +
                   (4323 * h[11] * b[3]) / 2560 - (4207 * h[10] * b[4]) / 10240 + (12109 * h[9] * b[5]) / 153600 - (17051 * h[8] * b[6]) / 1382400 +
                   (1927 * h[7] * b[7]) / 1209600;
    Fy_xy[6][0] = b[5] / 720;
    Fy_xy[6][1] = -(h[6] * c[0]) + (h[5] * c[1]) / 2 - (h[4] * c[2]) / 6 + (h[3] * c[3]) / 24 - (h[2] * c[4]) / 120 + (h[1] * c[5]) / 720 +
                  c[6] / 720;
    Fy_xy[6][2] = -(h[7] * b[0]) / 2 + (h[6] * b[1]) / 2 - (h[5] * b[2]) / 4 + (h[4] * b[3]) / 12 - (h[3] * b[4]) / 48 + (h[2] * b[5]) / 240 -
                  (h[1] * b[6]) / 1440 - b[7] / 1440;
    Fy_xy[6][3] = (14 * h[8] * c[0]) / 3 - (31 * h[7] * c[1]) / 12 + (37 * h[6] * c[2]) / 36 - (23 * h[5] * c[3]) / 72 + (29 * h[4] * c[4]) / 360 -
                  (73 * h[3] * c[5]) / 4320 + (13 * h[2] * c[6]) / 4320 - (h[1] * c[7]) / 2160;
    Fy_xy[6][4] = (7 * h[9] * b[0]) / 6 - (7 * h[8] * b[1]) / 6 + (31 * h[7] * b[2]) / 48 - (37 * h[6] * b[3]) / 144 + (23 * h[5] * b[4]) / 288 -
                  (29 * h[4] * b[5]) / 1440 + (73 * h[3] * b[6]) / 17280 - (13 * h[2] * b[7]) / 17280;
    Fy_xy[6][5] = (-63 * h[10] * c[0]) / 4 + (329 * h[9] * c[1]) / 40 - (119 * h[8] * c[2]) / 40 + (271 * h[7] * c[3]) / 320 -
                  (197 * h[6] * c[4]) / 960 + (17 * h[5] * c[5]) / 384 - (83 * h[4] * c[6]) / 9600 + (11 * h[3] * c[7]) / 7200;
    Fy_xy[6][6] = (-21 * h[11] * b[0]) / 8 + (21 * h[10] * b[1]) / 8 - (329 * h[9] * b[2]) / 240 + (119 * h[8] * b[3]) / 240 -
                  (271 * h[7] * b[4]) / 1920 + (197 * h[6] * b[5]) / 5760 - (17 * h[5] * b[6]) / 2304 + (83 * h[4] * b[7]) / 57600;
    Fy_xy[6][7] = (165 * h[12] * c[0]) / 4 - (339 * h[11] * c[1]) / 16 + (119 * h[10] * c[2]) / 16 - (193 * h[9] * c[3]) / 96 +
                  (43 * h[8] * c[4]) / 96 - (199 * h[7] * c[5]) / 2304 + (1213 * h[6] * c[6]) / 80640 - (11 * h[5] * c[7]) / 4480;
    Fy_xy[6][8] = (165 * h[13] * b[0]) / 32 - (165 * h[12] * b[1]) / 32 + (339 * h[11] * b[2]) / 128 - (119 * h[10] * b[3]) / 128 +
                  (193 * h[9] * b[4]) / 768 - (43 * h[8] * b[5]) / 768 + (199 * h[7] * b[6]) / 18432 - (1213 * h[6] * b[7]) / 645120;
    Fy_xy[6][9] = (56485 * h[14] * c[0]) / 384 - (55825 * h[13] * c[1]) / 768 + (54505 * h[12] * c[2]) / 2304 - (52435 * h[11] * c[3]) / 9216 +
                  (9887 * h[10] * c[4]) / 9216 - (27113 * h[9] * c[5]) / 165888 + (23489 * h[8] * c[6]) / 1161216 - (71 * h[7] * c[7]) / 36288;
    Fy_xy[6][10] = (11297 * h[15] * b[0]) / 768 - (11297 * h[14] * b[1]) / 768 + (11165 * h[13] * b[2]) / 1536 - (10901 * h[12] * b[3]) / 4608 +
                   (10487 * h[11] * b[4]) / 18432 - (9887 * h[10] * b[5]) / 92160 + (27113 * h[9] * b[6]) / 1658880 - (23489 * h[8] * b[7]) / 11612160;
    Fy_xy[7][0] = b[6] / 5040;
    Fy_xy[7][1] = h[7] * c[0] - (h[6] * c[1]) / 2 + (h[5] * c[2]) / 6 - (h[4] * c[3]) / 24 + (h[3] * c[4]) / 120 - (h[2] * c[5]) / 720 +
                  (h[1] * c[6]) / 5040 + c[7] / 5040;
    Fy_xy[7][2] = (h[8] * b[0]) / 2 - (h[7] * b[1]) / 2 + (h[6] * b[2]) / 4 - (h[5] * b[3]) / 12 + (h[4] * b[4]) / 48 - (h[3] * b[5]) / 240 +
                  (h[2] * b[6]) / 1440 - (h[1] * b[7]) / 10080;
    Fy_xy[7][3] = -6 * h[9] * c[0] + (13 * h[8] * c[1]) / 4 - (5 * h[7] * c[2]) / 4 + (3 * h[6] * c[3]) / 8 - (11 * h[5] * c[4]) / 120 +
                  (3 * h[4] * c[5]) / 160 - (11 * h[3] * c[6]) / 3360 + (h[2] * c[7]) / 2016;
    Fy_xy[7][4] = (-3 * h[10] * b[0]) / 2 + (3 * h[9] * b[1]) / 2 - (13 * h[8] * b[2]) / 16 + (5 * h[7] * b[3]) / 16 - (3 * h[6] * b[4]) / 32 +
                  (11 * h[5] * b[5]) / 480 - (3 * h[4] * b[6]) / 640 + (11 * h[3] * b[7]) / 13440;
    Fy_xy[7][5] = (99 * h[11] * c[0]) / 4 - (513 * h[10] * c[1]) / 40 + (183 * h[9] * c[2]) / 40 - (407 * h[8] * c[3]) / 320 + (19 * h[7] * c[4]) / 64 -
                  (39 * h[6] * c[5]) / 640 + (757 * h[5] * c[6]) / 67200 - (127 * h[4] * c[7]) / 67200;
    Fy_xy[7][6] = (33 * h[12] * b[0]) / 8 - (33 * h[11] * b[1]) / 8 + (171 * h[10] * b[2]) / 80 - (61 * h[9] * b[3]) / 80 + (407 * h[8] * b[4]) / 1920 -
                  (19 * h[7] * b[5]) / 384 + (13 * h[6] * b[6]) / 1280 - (757 * h[5] * b[7]) / 403200;
    Fy_xy[7][7] = (4719 * h[13] * c[0]) / 28 - (9339 * h[12] * c[1]) / 112 + (3047 * h[11] * c[2]) / 112 - (1471 * h[10] * c[3]) / 224 +
                  (199 * h[9] * c[4]) / 160 - (15331 * h[8] * c[5]) / 80640 + (13213 * h[7] * c[6]) / 564480 - (1229 * h[6] * c[7]) / 564480;
    Fy_xy[7][8] = (4719 * h[14] * b[0]) / 224 - (4719 * h[13] * b[1]) / 224 + (9339 * h[12] * b[2]) / 896 - (3047 * h[11] * b[3]) / 896 +
                  (1471 * h[10] * b[4]) / 1792 - (199 * h[9] * b[5]) / 1280 + (15331 * h[8] * b[6]) / 645120 - (13213 * h[7] * b[7]) / 4515840;
    Fy_xy[7][9] = (6721 * h[15] * c[0]) / 128 - (40755 * h[14] * c[1]) / 1792 + (28171 * h[13] * c[2]) / 5376 - (1375 * h[12] * c[3]) / 3072 -
                  (4741 * h[11] * c[4]) / 35840 + (14081 * h[10] * c[5]) / 215040 - (691 * h[9] * c[6]) / 43008 + (1819 * h[8] * c[7]) / 645120;
    Fy_xy[7][10] = (6721 * h[16] * b[0]) / 1280 - (6721 * h[15] * b[1]) / 1280 + (8151 * h[14] * b[2]) / 3584 - (28171 * h[13] * b[3]) / 53760 +
                   (275 * h[12] * b[4]) / 6144 + (4741 * h[11] * b[5]) / 358400 - (14081 * h[10] * b[6]) / 2150400 + (691 * h[9] * b[7]) / 430080;
    Fy_xy[8][0] = b[7] / 40320;
    Fy_xy[8][1] = -(h[8] * c[0]) + (h[7] * c[1]) / 2 - (h[6] * c[2]) / 6 + (h[5] * c[3]) / 24 - (h[4] * c[4]) / 120 + (h[3] * c[5]) / 720 -
                  (h[2] * c[6]) / 5040 + (h[1] * c[7]) / 40320;
    Fy_xy[8][2] = -(h[9] * b[0]) / 2 + (h[8] * b[1]) / 2 - (h[7] * b[2]) / 4 + (h[6] * b[3]) / 12 - (h[5] * b[4]) / 48 + (h[4] * b[5]) / 240 -
                  (h[3] * b[6]) / 1440 + (h[2] * b[7]) / 10080;
    Fy_xy[8][3] = (15 * h[10] * c[0]) / 2 - 4 * h[9] * c[1] + (3 * h[8] * c[2]) / 2 - (7 * h[7] * c[3]) / 16 + (5 * h[6] * c[4]) / 48 -
                  (h[5] * c[5]) / 48 + (h[4] * c[6]) / 280 - (43 * h[3] * c[7]) / 80640;
    Fy_xy[8][4] = (15 * h[11] * b[0]) / 8 - (15 * h[10] * b[1]) / 8 + h[9] * b[2] - (3 * h[8] * b[3]) / 8 + (7 * h[7] * b[4]) / 64 -
                  (5 * h[6] * b[5]) / 192 + (h[5] * b[6]) / 192 - (h[4] * b[7]) / 1120;
    Fy_xy[8][5] = (-297 * h[12] * c[0]) / 8 + (153 * h[11] * c[1]) / 8 - (27 * h[10] * c[2]) / 4 + (59 * h[9] * c[3]) / 32 - (67 * h[8] * c[4]) / 160 +
                  (53 * h[7] * c[5]) / 640 - (197 * h[6] * c[6]) / 13440 + (253 * h[5] * c[7]) / 107520;
    Fy_xy[8][6] = (-99 * h[13] * b[0]) / 16 + (99 * h[12] * b[1]) / 16 - (51 * h[11] * b[2]) / 16 + (9 * h[10] * b[3]) / 8 - (59 * h[9] * b[4]) / 192 +
                  (67 * h[8] * b[5]) / 960 - (53 * h[7] * b[6]) / 3840 + (197 * h[6] * b[7]) / 80640;
    Fy_xy[8][7] = (-22737 * h[14] * c[0]) / 112 + (2805 * h[13] * c[1]) / 28 - (3641 * h[12] * c[2]) / 112 + (3485 * h[11] * c[3]) / 448 -
                  (3257 * h[10] * c[4]) / 2240 + (4393 * h[9] * c[5]) / 20160 - (367 * h[8] * c[6]) / 14112 + (10291 * h[7] * c[7]) / 4515840;
    Fy_xy[8][8] = (-22737 * h[15] * b[0]) / 896 + (22737 * h[14] * b[1]) / 896 - (2805 * h[13] * b[2]) / 224 + (3641 * h[12] * b[3]) / 896 -
                  (3485 * h[11] * b[4]) / 3584 + (3257 * h[10] * b[5]) / 17920 - (4393 * h[9] * b[6]) / 161280 + (367 * h[8] * b[7]) / 112896;
    Fy_xy[8][9] = (-24167 * h[16] * c[0]) / 448 + (40755 * h[15] * c[1]) / 1792 - (25597 * h[14] * c[2]) / 5376 + (3355 * h[13] * c[3]) / 21504 +
                  (8327 * h[12] * c[4]) / 35840 - (57751 * h[11] * c[5]) / 645120 + (18455 * h[10] * c[6]) / 903168 - (13787 * h[9] * c[7]) / 4014080;
    Fy_xy[8][10] = (-24167 * h[17] * b[0]) / 4480 + (24167 * h[16] * b[1]) / 4480 - (8151 * h[15] * b[2]) / 3584 +
                   (25597 * h[14] * b[3]) / 53760 - (671 * h[13] * b[4]) / 43008 - (8327 * h[12] * b[5]) / 358400 + (57751 * h[11] * b[6]) / 6451200 -
                   (3691 * h[10] * b[7]) / 1806336;
    Fy_xy[9][0] = 0;
    Fy_xy[9][1] = h[9] * c[0] - (h[8] * c[1]) / 2 + (h[7] * c[2]) / 6 - (h[6] * c[3]) / 24 + (h[5] * c[4]) / 120 - (h[4] * c[5]) / 720 +
                  (h[3] * c[6]) / 5040 - (h[2] * c[7]) / 40320;
    Fy_xy[9][2] = (h[10] * b[0]) / 2 - (h[9] * b[1]) / 2 + (h[8] * b[2]) / 4 - (h[7] * b[3]) / 12 + (h[6] * b[4]) / 48 - (h[5] * b[5]) / 240 +
                  (h[4] * b[6]) / 1440 - (h[3] * b[7]) / 10080;
    Fy_xy[9][3] = (-55 * h[11] * c[0]) / 6 + (29 * h[10] * c[1]) / 6 - (16 * h[9] * c[2]) / 9 + (73 * h[8] * c[3]) / 144 - (17 * h[7] * c[4]) / 144 +
                  (5 * h[6] * c[5]) / 216 - (59 * h[5] * c[6]) / 15120 + (139 * h[4] * c[7]) / 241920;
    Fy_xy[9][4] = (-55 * h[12] * b[0]) / 24 + (55 * h[11] * b[1]) / 24 - (29 * h[10] * b[2]) / 24 + (4 * h[9] * b[3]) / 9 - (73 * h[8] * b[4]) / 576 +
                  (17 * h[7] * b[5]) / 576 - (5 * h[6] * b[6]) / 864 + (59 * h[5] * b[7]) / 60480;
    Fy_xy[9][5] = (-715 * h[13] * c[0]) / 8 + 44 * h[12] * c[1] - (341 * h[11] * c[2]) / 24 + (323 * h[10] * c[3]) / 96 - (59 * h[9] * c[4]) / 96 +
                  (101 * h[8] * c[5]) / 1152 - (379 * h[7] * c[6]) / 40320 + (197 * h[6] * c[7]) / 322560;
    Fy_xy[9][6] = (-715 * h[14] * b[0]) / 48 + (715 * h[13] * b[1]) / 48 - (22 * h[12] * b[2]) / 3 + (341 * h[11] * b[3]) / 144 -
                  (323 * h[10] * b[4]) / 576 + (59 * h[9] * b[5]) / 576 - (101 * h[8] * b[6]) / 6912 + (379 * h[7] * b[7]) / 241920;
    Fy_xy[9][7] = (-2145 * h[15] * c[0]) / 112 + (715 * h[14] * c[1]) / 112 - (1045 * h[12] * c[3]) / 1344 + (473 * h[11] * c[4]) / 1344 -
                  (193 * h[10] * c[5]) / 2016 + (535 * h[9] * c[6]) / 28224 - (2609 * h[8] * c[7]) / 903168;
    Fy_xy[9][8] = (-2145 * h[16] * b[0]) / 896 + (2145 * h[15] * b[1]) / 896 - (715 * h[14] * b[2]) / 896 + (1045 * h[12] * b[4]) / 10752 -
                  (473 * h[11] * b[5]) / 10752 + (193 * h[10] * b[6]) / 16128 - (535 * h[9] * b[7]) / 225792;
    Fy_xy[9][9] = (32175 * h[17] * c[0]) / 448 - (9295 * h[16] * c[1]) / 256 + (22165 * h[15] * c[2]) / 1792 - (202345 * h[14] * c[3]) / 64512 +
                  (39325 * h[13] * c[4]) / 64512 - (103345 * h[12] * c[5]) / 1161216 + (70477 * h[11] * c[6]) / 8128512 - (12937 * h[10] * c[7]) / 65028096;
    Fy_xy[9][10] = (6435 * h[18] * b[0]) / 896 - (6435 * h[17] * b[1]) / 896 + (1859 * h[16] * b[2]) / 512 - (4433 * h[15] * b[3]) / 3584 +
                   (40469 * h[14] * b[4]) / 129024 - (7865 * h[13] * b[5]) / 129024 + (20669 * h[12] * b[6]) / 2322432 - (70477 * h[11] * b[7]) / 81285120;
    Fy_xy[10][0] = 0;
    Fy_xy[10][1] = -(h[10] * c[0]) + (h[9] * c[1]) / 2 - (h[8] * c[2]) / 6 + (h[7] * c[3]) / 24 - (h[6] * c[4]) / 120 + (h[5] * c[5]) / 720 -
                   (h[4] * c[6]) / 5040 + (h[3] * c[7]) / 40320;
    Fy_xy[10][2] = -(h[11] * b[0]) / 2 + (h[10] * b[1]) / 2 - (h[9] * b[2]) / 4 + (h[8] * b[3]) / 12 - (h[7] * b[4]) / 48 + (h[6] * b[5]) / 240 -
                   (h[5] * b[6]) / 1440 + (h[4] * b[7]) / 10080;
    Fy_xy[10][3] = 11 * h[12] * c[0] - (23 * h[11] * c[1]) / 4 + (25 * h[10] * c[2]) / 12 - (7 * h[9] * c[3]) / 12 + (2 * h[8] * c[4]) / 15 -
                   (37 * h[7] * c[5]) / 1440 + (43 * h[6] * c[6]) / 10080 - (5 * h[5] * c[7]) / 8064;
    Fy_xy[10][4] = (11 * h[13] * b[0]) / 4 - (11 * h[12] * b[1]) / 4 + (23 * h[11] * b[2]) / 16 - (25 * h[10] * b[3]) / 48 + (7 * h[9] * b[4]) / 48 -
                   (h[8] * b[5]) / 30 + (37 * h[7] * b[6]) / 5760 - (43 * h[6] * b[7]) / 40320;
    Fy_xy[10][5] = (3861 * h[14] * c[0]) / 40 - (759 * h[13] * c[1]) / 16 + (1221 * h[12] * c[2]) / 80 - (115 * h[11] * c[3]) / 32 +
                   (521 * h[10] * c[4]) / 800 - (147 * h[9] * c[5]) / 1600 + (13 * h[8] * c[6]) / 1344 - (107 * h[7] * c[7]) / 179200;
    Fy_xy[10][6] = (1287 * h[15] * b[0]) / 80 - (1287 * h[14] * b[1]) / 80 + (253 * h[13] * b[2]) / 32 - (407 * h[12] * b[3]) / 160 +
                   (115 * h[11] * b[4]) / 192 - (521 * h[10] * b[5]) / 4800 + (49 * h[9] * b[6]) / 3200 - (13 * h[8] * b[7]) / 8064;
    Fy_xy[10][7] = (1287 * h[16] * c[0]) / 70 - (1287 * h[15] * c[1]) / 224 - (429 * h[14] * c[2]) / 1120 + (209 * h[13] * c[3]) / 224 -
                   (1111 * h[12] * c[4]) / 2800 + (21247 * h[11] * c[5]) / 201600 - (5801 * h[10] * c[6]) / 282240 + (17453 * h[9] * c[7]) / 5644800;
    Fy_xy[10][8] = (1287 * h[17] * b[0]) / 560 - (1287 * h[16] * b[1]) / 560 + (1287 * h[15] * b[2]) / 1792 + (429 * h[14] * b[3]) / 8960 -
                   (209 * h[13] * b[4]) / 1792 + (1111 * h[12] * b[5]) / 22400 - (21247 * h[11] * b[6]) / 1612800 + (5801 * h[10] * b[7]) / 2257920;
    Fy_xy[10][9] = (-170599 * h[18] * c[0]) / 2240 + (34463 * h[17] * c[1]) / 896 - (175747 * h[16] * c[2]) / 13440 +
                   (10153 * h[15] * c[3]) / 3072 - (49049 * h[14] * c[4]) / 76800 + (892309 * h[13] * c[5]) / 9676800 - (118217 * h[12] * c[6]) / 13547520 +
                   (72007 * h[11] * c[7]) / 541900800;
    Fy_xy[10][10] = (-170599 * h[19] * b[0]) / 22400 + (170599 * h[18] * b[1]) / 22400 - (34463 * h[17] * b[2]) / 8960 +
                    (175747 * h[16] * b[3]) / 134400 - (10153 * h[15] * b[4]) / 30720 + (49049 * h[14] * b[5]) / 768000 - (892309 * h[13] * b[6]) / 96768000 +
                    (118217 * h[12] * b[7]) / 135475200;

    Fx_xy[1][1] = b[1];
    Fx_xy[0][2] = (h[1] * c[0]) / 2 + c[1] / 2;
    Fx_xy[0][3] = (h[2] * b[0]) / 6 - (h[1] * b[1]) / 6 - b[2] / 6;
    Fx_xy[0][4] = -(h[3] * c[0]) / 8 + (h[2] * c[1]) / 8 - (h[1] * c[2]) / 12 - c[3] / 24;
    Fx_xy[0][5] = -(h[4] * b[0]) / 40 + (h[3] * b[1]) / 40 - (h[2] * b[2]) / 40 + (h[1] * b[3]) / 60 + b[4] / 120;
    Fx_xy[0][6] = (h[5] * c[0]) / 16 - (3 * h[4] * c[1]) / 80 + (h[3] * c[2]) / 60 - (h[2] * c[3]) / 120 + (h[1] * c[4]) / 240 + c[5] / 720;
    Fx_xy[0][7] = (h[6] * b[0]) / 112 - (h[5] * b[1]) / 112 + (3 * h[4] * b[2]) / 560 - (h[3] * b[3]) / 420 + (h[2] * b[4]) / 840 -
                  (h[1] * b[5]) / 1680 - b[6] / 5040;
    Fx_xy[0][8] = (-5 * h[7] * c[0]) / 128 + (19 * h[6] * c[1]) / 896 - (11 * h[5] * c[2]) / 1344 + (h[4] * c[3]) / 384 - (h[3] * c[4]) / 1344 +
                  (h[2] * c[5]) / 4032 - (h[1] * c[6]) / 10080 - c[7] / 40320;
    Fx_xy[0][9] = (-5 * h[8] * b[0]) / 1152 + (5 * h[7] * b[1]) / 1152 - (19 * h[6] * b[2]) / 8064 + (11 * h[5] * b[3]) / 12096 -
                  (h[4] * b[4]) / 3456 + (h[3] * b[5]) / 12096 - (h[2] * b[6]) / 36288 + (h[1] * b[7]) / 90720;
    Fx_xy[0][10] = (7 * h[9] * c[0]) / 256 - (11 * h[8] * c[1]) / 768 + (h[7] * c[2]) / 192 - (h[6] * c[3]) / 672 + (29 * h[5] * c[4]) / 80640 -
                   (19 * h[4] * c[5]) / 241920 + (h[3] * c[6]) / 60480 - (h[2] * c[7]) / 241920;
    Fx_xy[1][2] = -(h[2] * c[0]) + (h[1] * c[1]) / 2 + c[2] / 2;
    Fx_xy[1][3] = -(h[3] * b[0]) / 3 + (h[2] * b[1]) / 3 - (h[1] * b[2]) / 6 - b[3] / 6;
    Fx_xy[1][4] = (h[4] * c[0]) / 2 - (3 * h[3] * c[1]) / 8 + (5 * h[2] * c[2]) / 24 - (h[1] * c[3]) / 12 - c[4] / 24;
    Fx_xy[1][5] = (h[5] * b[0]) / 10 - (h[4] * b[1]) / 10 + (3 * h[3] * b[2]) / 40 - (h[2] * b[3]) / 24 + (h[1] * b[4]) / 60 + b[5] / 120;
    Fx_xy[1][6] = (-3 * h[6] * c[0]) / 8 + (17 * h[5] * c[1]) / 80 - (7 * h[4] * c[2]) / 80 + (h[3] * c[3]) / 30 - (h[2] * c[4]) / 80 +
                  (h[1] * c[5]) / 240 + c[6] / 720;
    Fx_xy[1][7] = (-3 * h[7] * b[0]) / 56 + (3 * h[6] * b[1]) / 56 - (17 * h[5] * b[2]) / 560 + (h[4] * b[3]) / 80 - (h[3] * b[4]) / 210 +
                  (h[2] * b[5]) / 560 - (h[1] * b[6]) / 1680 - b[7] / 5040;
    Fx_xy[1][8] = (5 * h[8] * c[0]) / 16 - (149 * h[7] * c[1]) / 896 + (167 * h[6] * c[2]) / 2688 - (25 * h[5] * c[3]) / 1344 +
                  (13 * h[4] * c[4]) / 2688 - (5 * h[3] * c[5]) / 4032 + (h[2] * c[6]) / 2880 - (h[1] * c[7]) / 10080;
    Fx_xy[1][9] = (5 * h[9] * b[0]) / 144 - (5 * h[8] * b[1]) / 144 + (149 * h[7] * b[2]) / 8064 - (167 * h[6] * b[3]) / 24192 +
                  (25 * h[5] * b[4]) / 12096 - (13 * h[4] * b[5]) / 24192 + (5 * h[3] * b[6]) / 36288 - (h[2] * b[7]) / 25920;
    Fx_xy[1][10] = (-35 * h[10] * c[0]) / 128 + (109 * h[9] * c[1]) / 768 - (13 * h[8] * c[2]) / 256 + (19 * h[7] * c[3]) / 1344 -
                   (53 * h[6] * c[4]) / 16128 + (163 * h[5] * c[5]) / 241920 - (31 * h[4] * c[6]) / 241920 + (h[3] * c[7]) / 40320;
    Fx_xy[2][0] = -c[1] / 2;
    Fx_xy[2][1] = b[2] / 2;
    Fx_xy[2][2] = (3 * h[3] * c[0]) / 2 - (3 * h[2] * c[1]) / 4 + (h[1] * c[2]) / 4 + c[3] / 4;
    Fx_xy[2][3] = (h[4] * b[0]) / 2 - (h[3] * b[1]) / 2 + (h[2] * b[2]) / 4 - (h[1] * b[3]) / 12 - b[4] / 12;
    Fx_xy[2][4] = (-5 * h[5] * c[0]) / 4 + (13 * h[4] * c[1]) / 16 - (19 * h[3] * c[2]) / 48 + (7 * h[2] * c[3]) / 48 - (h[1] * c[4]) / 24 - c[5] / 48;
    Fx_xy[2][5] = -(h[6] * b[0]) / 4 + (h[5] * b[1]) / 4 - (13 * h[4] * b[2]) / 80 + (19 * h[3] * b[3]) / 240 - (7 * h[2] * b[4]) / 240 +
                  (h[1] * b[5]) / 120 + b[6] / 240;
    Fx_xy[2][6] = (21 * h[7] * c[0]) / 16 - (23 * h[6] * c[1]) / 32 + (9 * h[5] * c[2]) / 32 - (3 * h[4] * c[3]) / 32 + (7 * h[3] * c[4]) / 240 -
                  (h[2] * c[5]) / 120 + (h[1] * c[6]) / 480 + c[7] / 1440;
    Fx_xy[2][7] = (3 * h[8] * b[0]) / 16 - (3 * h[7] * b[1]) / 16 + (23 * h[6] * b[2]) / 224 - (9 * h[5] * b[3]) / 224 + (3 * h[4] * b[4]) / 224 -
                  (h[3] * b[5]) / 240 + (h[2] * b[6]) / 840 - (h[1] * b[7]) / 3360;
    Fx_xy[2][8] = (-45 * h[9] * c[0]) / 32 + (189 * h[8] * c[1]) / 256 - (69 * h[7] * c[2]) / 256 + (139 * h[6] * c[3]) / 1792 -
                  (17 * h[5] * c[4]) / 896 + (23 * h[4] * c[5]) / 5376 - (13 * h[3] * c[6]) / 13440 + (h[2] * c[7]) / 4480;
    Fx_xy[2][9] = (-5 * h[10] * b[0]) / 32 + (5 * h[9] * b[1]) / 32 - (21 * h[8] * b[2]) / 256 + (23 * h[7] * b[3]) / 768 -
                  (139 * h[6] * b[4]) / 16128 + (17 * h[5] * b[5]) / 8064 - (23 * h[4] * b[6]) / 48384 + (13 * h[3] * b[7]) / 120960;
    Fx_xy[2][10] = (385 * h[11] * c[0]) / 256 - (397 * h[10] * c[1]) / 512 + (421 * h[9] * c[2]) / 1536 - (115 * h[8] * c[3]) / 1536 +
                   (13 * h[7] * c[4]) / 768 - (23 * h[6] * c[5]) / 6912 + (41 * h[5] * c[6]) / 69120 - (7 * h[4] * c[7]) / 69120;
    Fx_xy[3][0] = -c[2] / 6;
    Fx_xy[3][1] = b[3] / 6;
    Fx_xy[3][2] = -2 * h[4] * c[0] + h[3] * c[1] - (h[2] * c[2]) / 3 + (h[1] * c[3]) / 12 + c[4] / 12;
    Fx_xy[3][3] = (-2 * h[5] * b[0]) / 3 + (2 * h[4] * b[1]) / 3 - (h[3] * b[2]) / 3 + (h[2] * b[3]) / 9 - (h[1] * b[4]) / 36 - b[5] / 36;
    Fx_xy[3][4] = (5 * h[6] * c[0]) / 2 - (3 * h[5] * c[1]) / 2 + (2 * h[4] * c[2]) / 3 - (11 * h[3] * c[3]) / 48 + (h[2] * c[4]) / 16 - (h[1] * c[5]) / 72 -
                  c[6] / 144;
    Fx_xy[3][5] = (h[7] * b[0]) / 2 - (h[6] * b[1]) / 2 + (3 * h[5] * b[2]) / 10 - (2 * h[4] * b[3]) / 15 + (11 * h[3] * b[4]) / 240 -
                  (h[2] * b[5]) / 80 + (h[1] * b[6]) / 360 + b[7] / 720;
    Fx_xy[3][6] = (-7 * h[8] * c[0]) / 2 + (15 * h[7] * c[1]) / 8 - (17 * h[6] * c[2]) / 24 + (7 * h[5] * c[3]) / 32 - (29 * h[4] * c[4]) / 480 +
                  (11 * h[3] * c[5]) / 720 - (h[2] * c[6]) / 288 + (h[1] * c[7]) / 1440;
    Fx_xy[3][7] = -(h[9] * b[0]) / 2 + (h[8] * b[1]) / 2 - (15 * h[7] * b[2]) / 56 + (17 * h[6] * b[3]) / 168 - (h[5] * b[4]) / 32 +
                  (29 * h[4] * b[5]) / 3360 - (11 * h[3] * b[6]) / 5040 + (h[2] * b[7]) / 2016;
    Fx_xy[3][8] = (75 * h[10] * c[0]) / 16 - (39 * h[9] * c[1]) / 16 + (7 * h[8] * c[2]) / 8 - (439 * h[7] * c[3]) / 1792 + (103 * h[6] * c[4]) / 1792 -
                  (97 * h[5] * c[5]) / 8064 + (193 * h[4] * c[6]) / 80640 - (19 * h[3] * c[7]) / 40320;
    Fx_xy[3][9] = (25 * h[11] * b[0]) / 48 - (25 * h[10] * b[1]) / 48 + (13 * h[9] * b[2]) / 48 - (7 * h[8] * b[3]) / 72 + (439 * h[7] * b[4]) / 16128 -
                  (103 * h[6] * b[5]) / 16128 + (97 * h[5] * b[6]) / 72576 - (193 * h[4] * b[7]) / 725760;
    Fx_xy[3][10] = (-385 * h[12] * c[0]) / 64 + (395 * h[11] * c[1]) / 128 - (415 * h[10] * c[2]) / 384 + (149 * h[9] * c[3]) / 512 -
                   (33 * h[8] * c[4]) / 512 + (85 * h[7] * c[5]) / 6912 - (29 * h[6] * c[6]) / 13824 + (23 * h[5] * c[7]) / 69120;
    Fx_xy[4][0] = -c[3] / 24;
    Fx_xy[4][1] = b[4] / 24;
    Fx_xy[4][2] = (5 * h[5] * c[0]) / 2 - (5 * h[4] * c[1]) / 4 + (5 * h[3] * c[2]) / 12 - (5 * h[2] * c[3]) / 48 + (h[1] * c[4]) / 48 + c[5] / 48;
    Fx_xy[4][3] = (5 * h[6] * b[0]) / 6 - (5 * h[5] * b[1]) / 6 + (5 * h[4] * b[2]) / 12 - (5 * h[3] * b[3]) / 36 + (5 * h[2] * b[4]) / 144 -
                  (h[1] * b[5]) / 144 - b[6] / 144;
    Fx_xy[4][4] = (-35 * h[7] * c[0]) / 8 + (5 * h[6] * c[1]) / 2 - (25 * h[5] * c[2]) / 24 + (65 * h[4] * c[3]) / 192 - (17 * h[3] * c[4]) / 192 +
                  (11 * h[2] * c[5]) / 576 - (h[1] * c[6]) / 288 - c[7] / 576;
    Fx_xy[4][5] = (-7 * h[8] * b[0]) / 8 + (7 * h[7] * b[1]) / 8 - (h[6] * b[2]) / 2 + (5 * h[5] * b[3]) / 24 - (13 * h[4] * b[4]) / 192 +
                  (17 * h[3] * b[5]) / 960 - (11 * h[2] * b[6]) / 2880 + (h[1] * b[7]) / 1440;
    Fx_xy[4][6] = (63 * h[9] * c[0]) / 8 - (133 * h[8] * c[1]) / 32 + (49 * h[7] * c[2]) / 32 - (173 * h[6] * c[3]) / 384 + (221 * h[5] * c[4]) / 1920 -
                  (17 * h[4] * c[5]) / 640 + (h[3] * c[6]) / 180 - (h[2] * c[7]) / 960;
    Fx_xy[4][7] = (9 * h[10] * b[0]) / 8 - (9 * h[9] * b[1]) / 8 + (19 * h[8] * b[2]) / 32 - (7 * h[7] * b[3]) / 32 + (173 * h[6] * b[4]) / 2688 -
                  (221 * h[5] * b[5]) / 13440 + (17 * h[4] * b[6]) / 4480 - (h[3] * b[7]) / 1260;
    Fx_xy[4][8] = (-825 * h[11] * c[0]) / 64 + (213 * h[10] * c[1]) / 32 - (151 * h[9] * c[2]) / 64 + (663 * h[8] * c[3]) / 1024 -
                  (151 * h[7] * c[4]) / 1024 + (271 * h[6] * c[5]) / 9216 - (871 * h[5] * c[6]) / 161280 + (307 * h[4] * c[7]) / 322560;
    Fx_xy[4][9] = (-275 * h[12] * b[0]) / 192 + (275 * h[11] * b[1]) / 192 - (71 * h[10] * b[2]) / 96 + (151 * h[9] * b[3]) / 576 -
                  (221 * h[8] * b[4]) / 3072 + (151 * h[7] * b[5]) / 9216 - (271 * h[6] * b[6]) / 82944 + (871 * h[5] * b[7]) / 1451520;
    Fx_xy[4][10] = (-13299 * h[13] * c[0]) / 256 + (13189 * h[12] * c[1]) / 512 - (4323 * h[11] * c[2]) / 512 + (4207 * h[10] * c[3]) / 2048 -
                   (12109 * h[9] * c[4]) / 30720 + (17051 * h[8] * c[5]) / 276480 - (1927 * h[7] * c[6]) / 241920 + (403 * h[6] * c[7]) / 483840;
    Fx_xy[5][0] = -c[4] / 120;
    Fx_xy[5][1] = b[5] / 120;
    Fx_xy[5][2] = -3 * h[6] * c[0] + (3 * h[5] * c[1]) / 2 - (h[4] * c[2]) / 2 + (h[3] * c[3]) / 8 - (h[2] * c[4]) / 40 + (h[1] * c[5]) / 240 +
                  c[6] / 240;
    Fx_xy[5][3] = -(h[7] * b[0]) + h[6] * b[1] - (h[5] * b[2]) / 2 + (h[4] * b[3]) / 6 - (h[3] * b[4]) / 24 + (h[2] * b[5]) / 120 -
                  (h[1] * b[6]) / 720 - b[7] / 720;
    Fx_xy[5][4] = 7 * h[8] * c[0] - (31 * h[7] * c[1]) / 8 + (37 * h[6] * c[2]) / 24 - (23 * h[5] * c[3]) / 48 + (29 * h[4] * c[4]) / 240 -
                  (73 * h[3] * c[5]) / 2880 + (13 * h[2] * c[6]) / 2880 - (h[1] * c[7]) / 1440;
    Fx_xy[5][5] = (7 * h[9] * b[0]) / 5 - (7 * h[8] * b[1]) / 5 + (31 * h[7] * b[2]) / 40 - (37 * h[6] * b[3]) / 120 + (23 * h[5] * b[4]) / 240 -
                  (29 * h[4] * b[5]) / 1200 + (73 * h[3] * b[6]) / 14400 - (13 * h[2] * b[7]) / 14400;
    Fx_xy[5][6] = (-63 * h[10] * c[0]) / 4 + (329 * h[9] * c[1]) / 40 - (119 * h[8] * c[2]) / 40 + (271 * h[7] * c[3]) / 320 -
                  (197 * h[6] * c[4]) / 960 + (17 * h[5] * c[5]) / 384 - (83 * h[4] * c[6]) / 9600 + (11 * h[3] * c[7]) / 7200;
    Fx_xy[5][7] = (-9 * h[11] * b[0]) / 4 + (9 * h[10] * b[1]) / 4 - (47 * h[9] * b[2]) / 40 + (17 * h[8] * b[3]) / 40 - (271 * h[7] * b[4]) / 2240 +
                  (197 * h[6] * b[5]) / 6720 - (17 * h[5] * b[6]) / 2688 + (83 * h[4] * b[7]) / 67200;
    Fx_xy[5][8] = (495 * h[12] * c[0]) / 16 - (1017 * h[11] * c[1]) / 64 + (357 * h[10] * c[2]) / 64 - (193 * h[9] * c[3]) / 128 +
                  (43 * h[8] * c[4]) / 128 - (199 * h[7] * c[5]) / 3072 + (1213 * h[6] * c[6]) / 107520 - (33 * h[5] * c[7]) / 17920;
    Fx_xy[5][9] = (55 * h[13] * b[0]) / 16 - (55 * h[12] * b[1]) / 16 + (113 * h[11] * b[2]) / 64 - (119 * h[10] * b[3]) / 192 +
                  (193 * h[9] * b[4]) / 1152 - (43 * h[8] * b[5]) / 1152 + (199 * h[7] * b[6]) / 27648 - (1213 * h[6] * b[7]) / 967680;
    Fx_xy[5][10] = (11297 * h[14] * c[0]) / 128 - (11165 * h[13] * c[1]) / 256 + (10901 * h[12] * c[2]) / 768 - (10487 * h[11] * c[3]) / 3072 +
                   (9887 * h[10] * c[4]) / 15360 - (27113 * h[9] * c[5]) / 276480 + (23489 * h[8] * c[6]) / 1935360 - (71 * h[7] * c[7]) / 60480;
    Fx_xy[6][0] = -c[5] / 720;
    Fx_xy[6][1] = b[6] / 720;
    Fx_xy[6][2] = (7 * h[7] * c[0]) / 2 - (7 * h[6] * c[1]) / 4 + (7 * h[5] * c[2]) / 12 - (7 * h[4] * c[3]) / 48 + (7 * h[3] * c[4]) / 240 -
                  (7 * h[2] * c[5]) / 1440 + (h[1] * c[6]) / 1440 + c[7] / 1440;
    Fx_xy[6][3] = (7 * h[8] * b[0]) / 6 - (7 * h[7] * b[1]) / 6 + (7 * h[6] * b[2]) / 12 - (7 * h[5] * b[3]) / 36 + (7 * h[4] * b[4]) / 144 -
                  (7 * h[3] * b[5]) / 720 + (7 * h[2] * b[6]) / 4320 - (h[1] * b[7]) / 4320;
    Fx_xy[6][4] = (-21 * h[9] * c[0]) / 2 + (91 * h[8] * c[1]) / 16 - (35 * h[7] * c[2]) / 16 + (21 * h[6] * c[3]) / 32 - (77 * h[5] * c[4]) / 480 +
                  (21 * h[4] * c[5]) / 640 - (11 * h[3] * c[6]) / 1920 + (h[2] * c[7]) / 1152;
    Fx_xy[6][5] = (-21 * h[10] * b[0]) / 10 + (21 * h[9] * b[1]) / 10 - (91 * h[8] * b[2]) / 80 + (7 * h[7] * b[3]) / 16 - (21 * h[6] * b[4]) / 160 +
                  (77 * h[5] * b[5]) / 2400 - (21 * h[4] * b[6]) / 3200 + (11 * h[3] * b[7]) / 9600;
    Fx_xy[6][6] = (231 * h[11] * c[0]) / 8 - (1197 * h[10] * c[1]) / 80 + (427 * h[9] * c[2]) / 80 - (2849 * h[8] * c[3]) / 1920 +
                  (133 * h[7] * c[4]) / 384 - (91 * h[6] * c[5]) / 1280 + (757 * h[5] * c[6]) / 57600 - (127 * h[4] * c[7]) / 57600;
    Fx_xy[6][7] = (33 * h[12] * b[0]) / 8 - (33 * h[11] * b[1]) / 8 + (171 * h[10] * b[2]) / 80 - (61 * h[9] * b[3]) / 80 + (407 * h[8] * b[4]) / 1920 -
                  (19 * h[7] * b[5]) / 384 + (13 * h[6] * b[6]) / 1280 - (757 * h[5] * b[7]) / 403200;
    Fx_xy[6][8] = (4719 * h[13] * c[0]) / 32 - (9339 * h[12] * c[1]) / 128 + (3047 * h[11] * c[2]) / 128 - (1471 * h[10] * c[3]) / 256 +
                  (1393 * h[9] * c[4]) / 1280 - (15331 * h[8] * c[5]) / 92160 + (13213 * h[7] * c[6]) / 645120 - (1229 * h[6] * c[7]) / 645120;
    Fx_xy[6][9] = (1573 * h[14] * b[0]) / 96 - (1573 * h[13] * b[1]) / 96 + (3113 * h[12] * b[2]) / 384 - (3047 * h[11] * b[3]) / 1152 +
                  (1471 * h[10] * b[4]) / 2304 - (1393 * h[9] * b[5]) / 11520 + (15331 * h[8] * b[6]) / 829440 - (13213 * h[7] * b[7]) / 5806080;
    Fx_xy[6][10] = (47047 * h[15] * c[0]) / 1280 - (8151 * h[14] * c[1]) / 512 + (28171 * h[13] * c[2]) / 7680 - (1925 * h[12] * c[3]) / 6144 -
                   (4741 * h[11] * c[4]) / 51200 + (14081 * h[10] * c[5]) / 307200 - (691 * h[9] * c[6]) / 61440 + (1819 * h[8] * c[7]) / 921600;
    Fx_xy[7][0] = -c[6] / 5040;
    Fx_xy[7][1] = b[7] / 5040;
    Fx_xy[7][2] = -4 * h[8] * c[0] + 2 * h[7] * c[1] - (2 * h[6] * c[2]) / 3 + (h[5] * c[3]) / 6 - (h[4] * c[4]) / 30 + (h[3] * c[5]) / 180 -
                  (h[2] * c[6]) / 1260 + (h[1] * c[7]) / 10080;
    Fx_xy[7][3] = (-4 * h[9] * b[0]) / 3 + (4 * h[8] * b[1]) / 3 - (2 * h[7] * b[2]) / 3 + (2 * h[6] * b[3]) / 9 - (h[5] * b[4]) / 18 +
                  (h[4] * b[5]) / 90 - (h[3] * b[6]) / 540 + (h[2] * b[7]) / 3780;
    Fx_xy[7][4] = 15 * h[10] * c[0] - 8 * h[9] * c[1] + 3 * h[8] * c[2] - (7 * h[7] * c[3]) / 8 + (5 * h[6] * c[4]) / 24 - (h[5] * c[5]) / 24 +
                  (h[4] * c[6]) / 140 - (43 * h[3] * c[7]) / 40320;
    Fx_xy[7][5] = 3 * h[11] * b[0] - 3 * h[10] * b[1] + (8 * h[9] * b[2]) / 5 - (3 * h[8] * b[3]) / 5 + (7 * h[7] * b[4]) / 40 - (h[6] * b[5]) / 24 +
                  (h[5] * b[6]) / 120 - (h[4] * b[7]) / 700;
    Fx_xy[7][6] = (-99 * h[12] * c[0]) / 2 + (51 * h[11] * c[1]) / 2 - 9 * h[10] * c[2] + (59 * h[9] * c[3]) / 24 - (67 * h[8] * c[4]) / 120 +
                  (53 * h[7] * c[5]) / 480 - (197 * h[6] * c[6]) / 10080 + (253 * h[5] * c[7]) / 80640;
    Fx_xy[7][7] = (-99 * h[13] * b[0]) / 14 + (99 * h[12] * b[1]) / 14 - (51 * h[11] * b[2]) / 14 + (9 * h[10] * b[3]) / 7 - (59 * h[9] * b[4]) / 168 +
                  (67 * h[8] * b[5]) / 840 - (53 * h[7] * b[6]) / 3360 + (197 * h[6] * b[7]) / 70560;
    Fx_xy[7][8] = (-22737 * h[14] * c[0]) / 112 + (2805 * h[13] * c[1]) / 28 - (3641 * h[12] * c[2]) / 112 + (3485 * h[11] * c[3]) / 448 -
                  (3257 * h[10] * c[4]) / 2240 + (4393 * h[9] * c[5]) / 20160 - (367 * h[8] * c[6]) / 14112 + (10291 * h[7] * c[7]) / 4515840;
    Fx_xy[7][9] = (-7579 * h[15] * b[0]) / 336 + (7579 * h[14] * b[1]) / 336 - (935 * h[13] * b[2]) / 84 + (3641 * h[12] * b[3]) / 1008 -
                  (3485 * h[11] * b[4]) / 4032 + (3257 * h[10] * b[5]) / 20160 - (4393 * h[9] * b[6]) / 181440 + (367 * h[8] * b[7]) / 127008;
    Fx_xy[7][10] = (-24167 * h[16] * c[0]) / 560 + (8151 * h[15] * c[1]) / 448 - (25597 * h[14] * c[2]) / 6720 + (671 * h[13] * c[3]) / 5376 +
                   (8327 * h[12] * c[4]) / 44800 - (57751 * h[11] * c[5]) / 806400 + (3691 * h[10] * c[6]) / 225792 - (13787 * h[9] * c[7]) / 5017600;
    Fx_xy[8][0] = -c[7] / 40320;
    Fx_xy[8][1] = 0;
    Fx_xy[8][2] = (9 * h[9] * c[0]) / 2 - (9 * h[8] * c[1]) / 4 + (3 * h[7] * c[2]) / 4 - (3 * h[6] * c[3]) / 16 + (3 * h[5] * c[4]) / 80 -
                  (h[4] * c[5]) / 160 + (h[3] * c[6]) / 1120 - (h[2] * c[7]) / 8960;
    Fx_xy[8][3] = (3 * h[10] * b[0]) / 2 - (3 * h[9] * b[1]) / 2 + (3 * h[8] * b[2]) / 4 - (h[7] * b[3]) / 4 + (h[6] * b[4]) / 16 - (h[5] * b[5]) / 80 +
                  (h[4] * b[6]) / 480 - (h[3] * b[7]) / 3360;
    Fx_xy[8][4] = (-165 * h[11] * c[0]) / 8 + (87 * h[10] * c[1]) / 8 - 4 * h[9] * c[2] + (73 * h[8] * c[3]) / 64 - (17 * h[7] * c[4]) / 64 +
                  (5 * h[6] * c[5]) / 96 - (59 * h[5] * c[6]) / 6720 + (139 * h[4] * c[7]) / 107520;
    Fx_xy[8][5] = (-33 * h[12] * b[0]) / 8 + (33 * h[11] * b[1]) / 8 - (87 * h[10] * b[2]) / 40 + (4 * h[9] * b[3]) / 5 - (73 * h[8] * b[4]) / 320 +
                  (17 * h[7] * b[5]) / 320 - (h[6] * b[6]) / 96 + (59 * h[5] * b[7]) / 33600;
    Fx_xy[8][6] = (-2145 * h[13] * c[0]) / 16 + 66 * h[12] * c[1] - (341 * h[11] * c[2]) / 16 + (323 * h[10] * c[3]) / 64 - (59 * h[9] * c[4]) / 64 +
                  (101 * h[8] * c[5]) / 768 - (379 * h[7] * c[6]) / 26880 + (197 * h[6] * c[7]) / 215040;
    Fx_xy[8][7] = (-2145 * h[14] * b[0]) / 112 + (2145 * h[13] * b[1]) / 112 - (66 * h[12] * b[2]) / 7 + (341 * h[11] * b[3]) / 112 -
                  (323 * h[10] * b[4]) / 448 + (59 * h[9] * b[5]) / 448 - (101 * h[8] * b[6]) / 5376 + (379 * h[7] * b[7]) / 188160;
    Fx_xy[8][8] = (-19305 * h[15] * c[0]) / 896 + (6435 * h[14] * c[1]) / 896 - (3135 * h[12] * c[3]) / 3584 + (1419 * h[11] * c[4]) / 3584 -
                  (193 * h[10] * c[5]) / 1792 + (535 * h[9] * c[6]) / 25088 - (2609 * h[8] * c[7]) / 802816;
    Fx_xy[8][9] = (-2145 * h[16] * b[0]) / 896 + (2145 * h[15] * b[1]) / 896 - (715 * h[14] * b[2]) / 896 + (1045 * h[12] * b[4]) / 10752 -
                  (473 * h[11] * b[5]) / 10752 + (193 * h[10] * b[6]) / 16128 - (535 * h[9] * b[7]) / 225792;
    Fx_xy[8][10] = (57915 * h[17] * c[0]) / 896 - (16731 * h[16] * c[1]) / 512 + (39897 * h[15] * c[2]) / 3584 - (40469 * h[14] * c[3]) / 14336 +
                   (7865 * h[13] * c[4]) / 14336 - (20669 * h[12] * c[5]) / 258048 + (70477 * h[11] * c[6]) / 9031680 - (12937 * h[10] * c[7]) / 72253440;
    Fx_xy[9][0] = 0;
    Fx_xy[9][1] = 0;
    Fx_xy[9][2] = -5 * h[10] * c[0] + (5 * h[9] * c[1]) / 2 - (5 * h[8] * c[2]) / 6 + (5 * h[7] * c[3]) / 24 - (h[6] * c[4]) / 24 + (h[5] * c[5]) / 144 -
                  (h[4] * c[6]) / 1008 + (h[3] * c[7]) / 8064;
    Fx_xy[9][3] = (-5 * h[11] * b[0]) / 3 + (5 * h[10] * b[1]) / 3 - (5 * h[9] * b[2]) / 6 + (5 * h[8] * b[3]) / 18 - (5 * h[7] * b[4]) / 72 +
                  (h[6] * b[5]) / 72 - (h[5] * b[6]) / 432 + (h[4] * b[7]) / 3024;
    Fx_xy[9][4] = (55 * h[12] * c[0]) / 2 - (115 * h[11] * c[1]) / 8 + (125 * h[10] * c[2]) / 24 - (35 * h[9] * c[3]) / 24 + (h[8] * c[4]) / 3 -
                  (37 * h[7] * c[5]) / 576 + (43 * h[6] * c[6]) / 4032 - (25 * h[5] * c[7]) / 16128;
    Fx_xy[9][5] = (11 * h[13] * b[0]) / 2 - (11 * h[12] * b[1]) / 2 + (23 * h[11] * b[2]) / 8 - (25 * h[10] * b[3]) / 24 + (7 * h[9] * b[4]) / 24 -
                  (h[8] * b[5]) / 15 + (37 * h[7] * b[6]) / 2880 - (43 * h[6] * b[7]) / 20160;
    Fx_xy[9][6] = (1287 * h[14] * c[0]) / 8 - (1265 * h[13] * c[1]) / 16 + (407 * h[12] * c[2]) / 16 - (575 * h[11] * c[3]) / 96 +
                  (521 * h[10] * c[4]) / 480 - (49 * h[9] * c[5]) / 320 + (65 * h[8] * c[6]) / 4032 - (107 * h[7] * c[7]) / 107520;
    Fx_xy[9][7] = (1287 * h[15] * b[0]) / 56 - (1287 * h[14] * b[1]) / 56 + (1265 * h[13] * b[2]) / 112 - (407 * h[12] * b[3]) / 112 +
                  (575 * h[11] * b[4]) / 672 - (521 * h[10] * b[5]) / 3360 + (7 * h[9] * b[6]) / 320 - (65 * h[8] * b[7]) / 28224;
    Fx_xy[9][8] = (1287 * h[16] * c[0]) / 56 - (6435 * h[15] * c[1]) / 896 - (429 * h[14] * c[2]) / 896 + (1045 * h[13] * c[3]) / 896 -
                  (1111 * h[12] * c[4]) / 2240 + (21247 * h[11] * c[5]) / 161280 - (5801 * h[10] * c[6]) / 225792 + (17453 * h[9] * c[7]) / 4515840;
    Fx_xy[9][9] = (143 * h[17] * b[0]) / 56 - (143 * h[16] * b[1]) / 56 + (715 * h[15] * b[2]) / 896 + (143 * h[14] * b[3]) / 2688 -
                  (1045 * h[13] * b[4]) / 8064 + (1111 * h[12] * b[5]) / 20160 - (21247 * h[11] * b[6]) / 1451520 + (5801 * h[10] * b[7]) / 2032128;
    Fx_xy[9][10] = (-170599 * h[18] * c[0]) / 2240 + (34463 * h[17] * c[1]) / 896 - (175747 * h[16] * c[2]) / 13440 +
                   (10153 * h[15] * c[3]) / 3072 - (49049 * h[14] * c[4]) / 76800 + (892309 * h[13] * c[5]) / 9676800 - (118217 * h[12] * c[6]) / 13547520 +
                   (72007 * h[11] * c[7]) / 541900800;
    Fx_xy[10][0] = 0;
    Fx_xy[10][1] = 0;
    Fx_xy[10][2] = (11 * h[11] * c[0]) / 2 - (11 * h[10] * c[1]) / 4 + (11 * h[9] * c[2]) / 12 - (11 * h[8] * c[3]) / 48 + (11 * h[7] * c[4]) / 240 -
                   (11 * h[6] * c[5]) / 1440 + (11 * h[5] * c[6]) / 10080 - (11 * h[4] * c[7]) / 80640;
    Fx_xy[10][3] = (11 * h[12] * b[0]) / 6 - (11 * h[11] * b[1]) / 6 + (11 * h[10] * b[2]) / 12 - (11 * h[9] * b[3]) / 36 + (11 * h[8] * b[4]) / 144 -
                   (11 * h[7] * b[5]) / 720 + (11 * h[6] * b[6]) / 4320 - (11 * h[5] * b[7]) / 30240;
    Fx_xy[10][4] = (143 * h[13] * c[0]) / 4 - (275 * h[12] * c[1]) / 16 + (253 * h[11] * c[2]) / 48 - (55 * h[10] * c[3]) / 48 +
                   (11 * h[9] * c[4]) / 60 - (121 * h[8] * c[5]) / 5760 + (11 * h[7] * c[6]) / 8064 + (11 * h[6] * c[7]) / 161280;
    Fx_xy[10][5] = (143 * h[14] * b[0]) / 20 - (143 * h[13] * b[1]) / 20 + (55 * h[12] * b[2]) / 16 - (253 * h[11] * b[3]) / 240 +
                   (11 * h[10] * b[4]) / 48 - (11 * h[9] * b[5]) / 300 + (121 * h[8] * b[6]) / 28800 - (11 * h[7] * b[7]) / 40320;
    Fx_xy[10][6] = (-429 * h[15] * c[0]) / 80 + (143 * h[14] * c[1]) / 32 - (429 * h[13] * c[2]) / 160 + (209 * h[12] * c[3]) / 192 -
                   (1507 * h[11] * c[4]) / 4800 + (649 * h[10] * c[5]) / 9600 - (451 * h[9] * c[6]) / 40320 + (1529 * h[8] * c[7]) / 1075200;
    Fx_xy[10][7] = (-429 * h[16] * b[0]) / 560 + (429 * h[15] * b[1]) / 560 - (143 * h[14] * b[2]) / 224 + (429 * h[13] * b[3]) / 1120 -
                   (209 * h[12] * b[4]) / 1344 + (1507 * h[11] * b[5]) / 33600 - (649 * h[10] * b[6]) / 67200 + (451 * h[9] * b[7]) / 282240;
    Fx_xy[10][8] = (-21879 * h[17] * c[0]) / 560 + (34749 * h[16] * c[1]) / 1792 - (8151 * h[15] * c[2]) / 1280 + (2717 * h[14] * c[3]) / 1792 -
                   (5863 * h[13] * c[4]) / 22400 + (46651 * h[12] * c[5]) / 1612800 - (1133 * h[11] * c[6]) / 2257920 - (26851 * h[10] * c[7]) / 45158400;
    Fx_xy[10][9] = (-2431 * h[18] * b[0]) / 560 + (2431 * h[17] * b[1]) / 560 - (3861 * h[16] * b[2]) / 1792 + (2717 * h[15] * b[3]) / 3840 -
                   (2717 * h[14] * b[4]) / 16128 + (5863 * h[13] * b[5]) / 201600 - (46651 * h[12] * b[6]) / 14515200 + (1133 * h[11] * b[7]) / 20321280;
    Fx_xy[10][10] = (476333 * h[19] * c[0]) / 22400 - (14443 * h[18] * c[1]) / 1280 + (563849 * h[17] * c[2]) / 134400 -
                    (260117 * h[16] * c[3]) / 215040 + (1525381 * h[15] * c[4]) / 5376000 - (5336903 * h[14] * c[5]) / 96768000 +
                    (1191619 * h[13] * c[6]) / 135475200 - (5947469 * h[12] * c[7]) / 5419008000;
  }

  /*
  for (i=0; i<11; i++) 
    for (j=0; j<11; j++)
      printf("Fx[%ld][%ld] = %e   Fy[%ld][%ld] = %e\n", i, j, Fx_xy[i][j], i, j, Fy_xy[i][j]);
      */
}

long trackCSBENDWithLargeRadius(double **part, int64_t n_part, CSBEND *csbend, double p_error,
                                double Po, double **accepted,
                                double z_start, double *sigmaDelta2, char *rootname, MAXAMP *maxamp,
                                APCONTOUR *apContour, APERTURE_DATA *apFileData,
                                /* If iSlice non-negative, we do one step. The caller is responsible 
                                 * for handling the coordinates appropriately outside this routine. 
                                 * The element must have been previously optimized to determine FSE and X offsets.
                                 */
                                long iSlice,
                                ELEMENT_LIST *eptr)
{
  if (iSlice>=0)
    bombElegant("Error: One or more CSBENDs have angle = 0 or radius > 1e6 but radiation matrix was requested. Please convert element to an EDRIFT, KQUAD, or KSEXT as appropriate.\n", NULL);
  if (csbend->k1 != 0) {
    if (csbend->k2 == 0) {
      ELEMENT_LIST elem;
      KQUAD kquad;
      printWarningForTracking("CSBEND has angle = 0 or radius > 1e6 but non-zero K1.",
                              "Treated as KQUAD; higher multipoles ignored.");
      memset(&elem, 0, sizeof(elem));
      memset(&kquad, 0, sizeof(kquad));
      elem.p_elem = (void *)&kquad;
      elem.type = T_KQUAD;
      kquad.length = csbend->length;
      kquad.k1 = csbend->k1;
      kquad.tilt = csbend->tilt + csbend->etilt * csbend->etiltSign;
      kquad.dx = csbend->dx;
      kquad.dy = csbend->dy;
      kquad.dz = csbend->dz;
      kquad.synch_rad = csbend->synch_rad;
      kquad.isr = csbend->isr;
      kquad.isr1Particle = csbend->isr1Particle;
      kquad.nSlices = csbend->nSlices;
      kquad.integration_order = csbend->integration_order;
      return multipole_tracking2(part, n_part, &elem, p_error, Po, accepted, z_start, maxamp, NULL, apFileData, sigmaDelta2, -1);
    } else {
      /* K1 and K2 nonzero */
      ELEMENT_LIST elem;
      KQUSE kquse;
      printWarningForTracking("CSBEND has angle = 0 or radius > 1e6 but non-zero K1 and K2.",
                              "Treated as KQUSE; higher multipoles ignored.");
      memset(&elem, 0, sizeof(elem));
      memset(&kquse, 0, sizeof(kquse));
      elem.p_elem = (void *)&kquse;
      elem.type = T_KQUSE;
      kquse.length = csbend->length;
      kquse.k1 = csbend->k1;
      kquse.k2 = csbend->k2;
      kquse.tilt = csbend->tilt + csbend->etilt * csbend->etiltSign;
      kquse.dx = csbend->dx;
      kquse.dy = csbend->dy;
      kquse.dz = csbend->dz;
      kquse.synch_rad = csbend->synch_rad;
      kquse.isr = csbend->isr;
      kquse.isr1Particle = csbend->isr1Particle;
      kquse.nSlices = csbend->nSlices;
      kquse.integration_order = csbend->integration_order;
      return multipole_tracking2(part, n_part, &elem, p_error, Po, accepted, z_start, maxamp, NULL, apFileData, sigmaDelta2, -1);
    }
  } else if (csbend->k2 != 0) {
    /* K2 nonzero */
    ELEMENT_LIST elem;
    KSEXT ksext;
    printWarningForTracking("CSBEND has angle = 0 or radius > 1e6 but non-zero K1 and K2.",
                            "Treated as KSEXT; higher multipoles ignored.");
    memset(&elem, 0, sizeof(elem));
    memset(&ksext, 0, sizeof(ksext));
    elem.p_elem = (void *)&ksext;
    elem.type = T_KSEXT;
    ksext.length = csbend->length;
    ksext.k2 = csbend->k2;
    ksext.tilt = csbend->tilt + csbend->etilt * csbend->etiltSign;
    ksext.dx = csbend->dx;
    ksext.dy = csbend->dy;
    ksext.dz = csbend->dz;
    ksext.synch_rad = csbend->synch_rad;
    ksext.isr = csbend->isr;
    ksext.isr1Particle = csbend->isr1Particle;
    ksext.nSlices = csbend->nSlices;
    ksext.integration_order = csbend->integration_order;
    return multipole_tracking2(part, n_part, &elem, p_error, Po, accepted, z_start, maxamp, NULL, apFileData, sigmaDelta2, -1);
  } else {
    /* K1 and K2 zero */
    printWarningForTracking("CSBEND has radius > 1e6 with zero K1.",
                            "Treated as EDRIFT; higher multipoles are ignored.");
    exactDrift(part, n_part, csbend->length);
    return n_part;
  }
}

long track_through_csbend(double **part, int64_t n_part, CSBEND *csbend, double p_error,
                          double Po, double **accepted,
                          double z_start, double *sigmaDelta2, char *rootname, MAXAMP *maxamp,
                          APCONTOUR *apContour, APERTURE_DATA *apFileData,
                          /* If iSlice non-negative, we do one step. The caller is responsible 
                           * for handling the coordinates appropriately outside this routine. 
                           * The element must have been previously optimized to determine FSE and X offsets.
                           */
                          long iSlice,
                          ELEMENT_LIST *eptr) {
  double h;
  int64_t i_part, i_top;
  long particle_lost, j;
  double rho, s, Fx, Fy;
  double x, xp, y, yp, dp, dp0;
  double n, fse, dp_prime;
  double tilt, etilt, cos_ttilt, sin_ttilt, ttilt;
  double *coord, dz_lost;
  double angle, e1, e2, Kg1, Kg2;
  double psi1, psi2, he1, he2;
  double Qi[MAX_PROPERTIES_PER_PARTICLE], Qf[MAX_PROPERTIES_PER_PARTICLE];
  double dcoord_etilt[6];
  double dxi, dyi, dzi;
  double dxf, dyf, dzf;
  double delta_xp, xp0, yp0;
  double e1_kick_limit, e2_kick_limit;
  MULT_APERTURE_DATA apertureData;

  /*
  static FILE *fpdeb = NULL;
  if (!fpdeb) {
    fpdeb = fopen("apdebug.sdds", "w");
    fprintf(fpdeb, "SDDS1\n&column name=x type=float &end\n&column name=y type=float &end\n&data mode=ascii no_row_counts=1 &end\n");
  }
  */

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    i_part = gpu_track_through_csbend(n_part, csbend, p_error, Po, accepted,
                                      z_start, sigmaDelta2, rootname, maxamp, apContour, apFileData, iSlice, eptr);
#  ifdef GPU_VERIFY
    startCpuTimer();
    track_through_csbend(part, n_part, csbend, p_error, Po, accepted, z_start, sigmaDelta2, rootname, maxamp, apContour, apFileData, iSlice, eptr);
    compareGpuCpu(n_part, "track_through_csbend");
#  endif /* GPU_VERIFY */
    return i_part;
  }
#endif /* HAVE_GPU */

  if (!csbend)
    bombElegant("null CSBEND pointer (track_through_csbend)", NULL);

  if (iSlice >= 0 && csbend->referenceCorrection && csbend->refTrajectoryChangeSet == 0)
    bombElegant("One-step CSBEND tracking invoked but reference correction not completed first, which is a bug.", NULL);

  setUpCsbendPhotonOutputFile(csbend, rootname, n_part);

  if (csbend->edge_order > 1 && (csbend->edge_effects[csbend->e1Index] == 2 || csbend->edge_effects[csbend->e2Index] == 2) && csbend->hgap == 0)
    bombElegant("CSBEND has EDGE_ORDER>1 and EDGE[12]_EFFECTS==2, but HGAP=0. This gives undefined results.", NULL);

  if (csbend->referenceCorrection) {
    if (csbend->refTrajectoryChangeSet == 0 || csbend->refLength != csbend->length || csbend->refAngle != csbend->angle || csbend->refSlices != csbend->nSlices) {
      /* Figure out the reference trajectory offsets to suppress inaccuracy in the integrator */
      CSBEND csbend0;
      double **part0;
      TRACKING_CONTEXT tcontext;

      getTrackingContext(&tcontext);
      if (tcontext.elementOccurrence > 0) {
        printf("Determining reference trajectory for CSBEND %s#%ld at s=%e\n", tcontext.elementName, tcontext.elementOccurrence, tcontext.zStart);
      }

      if (csbend->refTrajectoryChange && csbend->refSlices) {
        /*
        printf("Freeing refTrajectoryChange=%p for CSBEND (%ld slices)\n", (void*)csbend->refTrajectoryChange,
               csbend->refSlices);
        fflush(stdout);
        */
        free_czarray_2d((void **)csbend->refTrajectoryChange, csbend->refSlices, 5);
        csbend->refTrajectoryChange = NULL;
        csbend->refSlices = 0;
      }

      part0 = (double **)czarray_2d(sizeof(double), 1, totalPropertiesPerParticle);
      memset(part0[0], 0, sizeof(**part0) * totalPropertiesPerParticle);
      memcpy(&csbend0, csbend, sizeof(*csbend));
      csbend0.dx = csbend0.dy = csbend0.dz = csbend0.fse = csbend0.etilt = csbend0.epitch = csbend0.eyaw =
        csbend0.isr = csbend0.synch_rad = csbend0.fseDipole = csbend0.fseQuadrupole = csbend0.xKick = csbend0.yKick = 0;

      csbend0.refTrajectoryChange = csbend->refTrajectoryChange = (double **)czarray_2d(sizeof(double), csbend->nSlices, 5);
      /*
      printf("Allocated refTrajectoryChange=%p for CSBEND\n", (void*)csbend->refTrajectoryChange);
      fflush(stdout);
      */
      csbend->refSlices = csbend0.refSlices = csbend0.nSlices;
      refTrajectoryPoints = csbend->nSlices;
      csbend0.refLength = csbend0.length;
      csbend0.refAngle = csbend0.angle;
      /* This forces us into the next branch on the next call to this routine */
      csbend0.refTrajectoryChangeSet = 1;
      setTrackingContext("csbend0", 0, T_CSBEND, "none", NULL, -1);
      track_through_csbend(part0, 1, &csbend0, p_error, Po, NULL, 0, NULL, NULL, maxamp, apContour, apFileData, -1, eptr);
      csbend->refTrajectoryChangeSet = 2; /* indicates that reference trajectory has been determined */

      csbend->refLength = csbend->length;
      csbend->refAngle = csbend->angle;
      free_czarray_2d((void **)part0, 1, totalPropertiesPerParticle);

      refTrajectoryData = csbend->refTrajectoryChange;
      refTrajectoryPoints = csbend->refSlices;
      refTrajectoryMode = SUBTRACT_TRAJECTORY;
    } else if (csbend->refTrajectoryChangeSet == 1) {
      /* indicates reference trajectory is about to be determined */
      refTrajectoryData = csbend->refTrajectoryChange;
      refTrajectoryPoints = csbend->refSlices;
      /*
      printf("refTrajectoryData = %p, refTrajectoryPoints = %ld with csbend->refTrajectoryChangeSet==1\n", 
             refTrajectoryData, refTrajectoryPoints);
      fflush(stdout);
      */
      refTrajectoryMode = RECORD_TRAJECTORY;
      csbend->refTrajectoryChangeSet = 2;
    } else {
      /* assume that reference trajectory already determined */
      refTrajectoryData = csbend->refTrajectoryChange;
      refTrajectoryPoints = csbend->refSlices;
      refTrajectoryMode = SUBTRACT_TRAJECTORY;
      /*
      printf("refTrajectoryData = %p, refTrajectoryPoints = %ld with csbend->refTrajectoryChangeSet==%ld\n", 
             refTrajectoryData, refTrajectoryPoints, csbend->refTrajectoryChangeSet);
      fflush(stdout);
      */
    }
  } else
    refTrajectoryMode = 0;

  if (csbend->angle == 0 || fabs(rho0 = csbend->length / csbend->angle)>1e6)
    return trackCSBENDWithLargeRadius(part, n_part, csbend, p_error, Po, accepted, z_start, sigmaDelta2, rootname,
                               maxamp, apContour, apFileData, iSlice, eptr);

  if (!(csbend->edgeFlags & BEND_EDGE_DETERMINED))
    bombElegant("CSBEND element doesn't have edge flags set.", NULL);

  if (csbend->integration_order != 2 && csbend->integration_order != 4 && csbend->integration_order != 6)
    bombElegant("CSBEND integration_order is invalid--must be 2, 4, or 6", NULL);

  if (csbend->use_bn) {
    csbend->b[0] = 0;
    csbend->b[1] = csbend->b1;
    csbend->b[2] = csbend->b2;
    csbend->b[3] = csbend->b3;
    csbend->b[4] = csbend->b4;
    csbend->b[5] = csbend->b5;
    csbend->b[6] = csbend->b6;
    csbend->b[7] = csbend->b7;
    csbend->b[8] = csbend->b8;
  } else {
    csbend->b[0] = 0;
    csbend->b[1] = csbend->k1 * rho0;
    csbend->b[2] = csbend->k2 * rho0;
    csbend->b[3] = csbend->k3 * rho0;
    csbend->b[4] = csbend->k4 * rho0;
    csbend->b[5] = csbend->k5 * rho0;
    csbend->b[6] = csbend->k6 * rho0;
    csbend->b[7] = csbend->k7 * rho0;
    csbend->b[8] = csbend->k8 * rho0;
  }
  for (j = 0; j < 9; j++)
    csbend->c[j] = 0;
  if (csbend->xReference > 0) {
    double term = 1 / csbend->xReference, f[8], g[8];
    long i;
    f[0] = csbend->f1;
    f[1] = csbend->f2;
    f[2] = csbend->f3;
    f[3] = csbend->f4;
    f[4] = csbend->f5;
    f[5] = csbend->f6;
    f[6] = csbend->f7;
    f[7] = csbend->f8;
    g[0] = csbend->g1;
    g[1] = csbend->g2;
    g[2] = csbend->g3;
    g[3] = csbend->g4;
    g[4] = csbend->g5;
    g[5] = csbend->g6;
    g[6] = csbend->g7;
    g[7] = csbend->g8;
    for (i = 0; i < 8; i++) {
      csbend->b[i + 1] += f[i] * term;
      csbend->c[i + 1] += g[i] * term;
      term *= (i + 2) / csbend->xReference;
    }
  }
  /* these adjustments ensure that we don't apply FSE+FSEDIPOLE twice for quadrupole and sextupole terms */
  csbend->b[1] *= (1 + csbend->fse + csbend->fseQuadrupole) / (1 + csbend->fse + csbend->fseDipole);
  csbend->c[1] *= (1 + csbend->fse + csbend->fseQuadrupole) / (1 + csbend->fse + csbend->fseDipole);
  csbend->b[2] *= (1 + csbend->fse) / (1 + csbend->fse + csbend->fseDipole);
  csbend->c[2] *= (1 + csbend->fse) / (1 + csbend->fse + csbend->fseDipole);

  csbend->b[0] = csbend->xKick / csbend->angle;
  csbend->c[0] = csbend->yKick / csbend->angle;

  he1 = csbend->h[csbend->e1Index];
  he2 = csbend->h[csbend->e2Index];
  if (csbend->angle < 0 && csbend->malignMethod == 0) {
    long i;
    angle = -csbend->angle;
    e1 = -csbend->e[csbend->e1Index];
    e2 = -csbend->e[csbend->e2Index];
    etilt = csbend->etilt * csbend->etiltSign;
    tilt = csbend->tilt + PI;
    rho0 = csbend->length / angle;
    for (i = 1; i < 9; i += 2) {
      csbend->b[i] *= -1;
      csbend->c[i] *= -1;
    }
  } else {
    angle = csbend->angle;
    e1 = csbend->e[csbend->e1Index];
    e2 = csbend->e[csbend->e2Index];
    etilt = csbend->etilt * csbend->etiltSign;
    tilt = csbend->tilt;
    rho0 = csbend->length / angle;
  }

  setupMultApertureData(&apertureData, -tilt, apContour, maxamp, apFileData, NULL, z_start + csbend->length / 2, eptr);

  fse = csbend->fse + csbend->fseDipole + (csbend->fseCorrection ? csbend->fseCorrectionValue : 0);
  h = 1 / rho0;
  n = -csbend->b[1] / h;
  if (fabs(fse + 1) < 1e-12)
    fse = -1 + 1e-12;
  rho_actual = 1 / ((1 + fse) * h);

  /*
  if (1) {
      TRACKING_CONTEXT tcontext;
      getTrackingContext(&tcontext);
      printf("Tracking %s#%ld: FSE=%le, FSE(User)=%le, FSE(Correction)=%le\n",
             tcontext.elementName, tcontext.elementOccurrence, fse, csbend->fse, csbend->fseCorrectionValue);
  }
  */

  e1_kick_limit = csbend->edge_kick_limit[csbend->e1Index];
  e2_kick_limit = csbend->edge_kick_limit[csbend->e2Index];
  if (csbend->kick_limit_scaling) {
    e1_kick_limit *= rho0 / rho_actual;
    e2_kick_limit *= rho0 / rho_actual;
  }
  if (e1_kick_limit > 0 || e2_kick_limit > 0) {
    printf("rho0=%e  rho_a=%e fse=%e e1_kick_limit=%e e2_kick_limit=%e\n",
           rho0, rho_actual, csbend->fse, e1_kick_limit, e2_kick_limit);
    fflush(stdout);
  }
  /* angles for fringe-field effects */
  Kg1 = 2 * csbend->hgap * (csbend->fint[csbend->e1Index] >= 0 ? csbend->fint[csbend->e1Index] : csbend->fintBoth) * SIGN(rho0);
  psi1 = Kg1 / fabs(rho_actual) / cos(e1) * (1 + sqr(sin(e1)));
  Kg2 = 2 * csbend->hgap * (csbend->fint[csbend->e2Index] >= 0 ? csbend->fint[csbend->e2Index] : csbend->fintBoth) * SIGN(rho0);
  psi2 = Kg2 / fabs(rho_actual) / cos(e2) * (1 + sqr(sin(e2)));
  if (csbend->length < 0) {
    psi1 *= -1;
    psi2 *= -1;
  }

  /* rad_coef is d((P-Po)/Po)/ds for the on-axis, on-momentum particle, where po is the momentum of
   * the central particle.
   */
  if (csbend->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) * sqr(1 + fse) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho0));
  else
    rad_coef = 0;
  /* isrConstant is the RMS increase in dP/P per meter due to incoherent SR.  */
  isrConstant = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895 / pow3(fabs(rho_actual)));
  if (!csbend->isr || (csbend->isr1Particle == 0 && n_part == 1))
    /* Minus sign here indicates that we accumulate ISR into sigmaDelta^2 but don't apply it to particles. */
    isrConstant *= -1;

  if ((distributionBasedRadiation = csbend->distributionBasedRadiation)) {
    /* Sands 5.15 */
    meanPhotonsPerRadian0 = 5.0 / (2.0 * sqrt(3)) * Po / 137.0359895;
    meanPhotonsPerMeter0 = (5 * c_mks * Po * particleMass * particleRadius) / (2 * sqrt(3) * hbar_mks * fabs(rho_actual));
    /* Critical energy normalized to reference energy, Sands 5.9 */
    normalizedCriticalEnergy0 = 3.0 / 2 * hbar_mks * c_mks * pow3(Po) / fabs(rho_actual) / (Po * particleMass * sqr(c_mks));
    /* fprintf(stderr, "Mean photons per radian expected: %le   ECritical/E: %le\n", 
            meanPhotonsPerRadian0, normalizedCriticalEnergy0);
    */
    includeOpeningAngle = csbend->includeOpeningAngle;
  }

  // Store and reload computed Fx_xy/Fy_xy based on function inputs
  // If we were using C++, there are many good memoization libraries, but alas...

  //  If debugging, enable tracking context
  //  TRACKING_CONTEXT context;
  //  getTrackingContext(&context);

  bool needsUpdate = false;
  if (!csbend->Fx_xy_ref) {
    // First execution, need to cache
    needsUpdate = true;
//    printf("CSBEND update needed because this is initial run for %s#%ld h:%e, nl:%ld order:%ld\n",
//           context.elementName, context.elementOccurrence, h, csbend->nonlinear, csbend->expansionOrder);
  } else if ((memcmp(csbend->b_ref, csbend->b, sizeof(double)*9) != 0) ||
            (memcmp(csbend->c_ref, csbend->c, sizeof(double)*9) != 0) ||
            (csbend->h_ref != h) ||
            (csbend->nonlinear_ref != csbend->nonlinear) ||
            (csbend->expansionOrder_ref != csbend->expansionOrder)) {
    // Need update due to new parameters (only 5 determine output)
    needsUpdate = true;
//    printf("CSBEND update needed because of parameter change for %s#%ldc\n",
//           context.elementName, context.elementOccurrence);
  }

  if (needsUpdate) {
    computeCSBENDFieldCoefficients(csbend->b, csbend->c, h, csbend->nonlinear, csbend->expansionOrder);
    // b_ref/c_ref already initialized because part of global struct
    memcpy(csbend->b_ref, csbend->b, sizeof(double)*9);
    memcpy(csbend->c_ref, csbend->c, sizeof(double)*9);
    csbend->expansionOrder_ref = csbend->expansionOrder;
    csbend->nonlinear_ref = csbend->nonlinear;
    csbend->h_ref = h;
    csbend->expansionOrder1_ref = expansionOrder1;
    csbend->hasNormal_ref = hasNormal;
    csbend->hasSkew_ref = hasSkew;
    if (!csbend->Fx_xy_ref)
      csbend->Fx_xy_ref = (double **)czarray_2d(sizeof(double), 11, 11);
    if (!csbend->Fy_xy_ref)
      csbend->Fy_xy_ref = (double **)czarray_2d(sizeof(double), 11, 11);
    memcpy(*(csbend->Fx_xy_ref), *(Fx_xy), sizeof(double)*11*11);
    memcpy(*(csbend->Fy_xy_ref), *(Fy_xy), sizeof(double)*11*11);
//    printf("Fresh CSBEND fields initialized h:%e nl:%ld order:%ld\n", h, csbend->nonlinear, csbend->expansionOrder);
//    printf("Set fresh CSBEND fields for array of size %ld %ld %ld\n Fy_xy[0][0]=%e  %e  \n", sizeof(csbend->b),
//           sizeof(Fx_xy), sizeof(csbend->Fx_xy_ref), (csbend->Fy_xy_ref)[0][0], Fy_xy[0][0]);
  } else {
    // Set pointers to cached valued
    if (!Fx_xy)
      bombElegant("unexpected null pointer in Fx_xy", NULL);
    if (!Fy_xy)
      bombElegant("unexpected null pointer in Fy_xy", NULL);
    memcpy(*(Fx_xy), *(csbend->Fx_xy_ref), sizeof(double)*11*11);
    memcpy(*(Fy_xy), *(csbend->Fy_xy_ref), sizeof(double)*11*11);
    //Fx_xy = csbend->Fx_xy_ref;
    //Fy_xy = csbend->Fy_xy_ref;
    expansionOrder1 = csbend->expansionOrder1_ref;
    hasNormal = csbend->hasNormal_ref;
    hasSkew = csbend->hasSkew_ref;
//    printf("Set CACHED CSBEND fields for %s#%ld -- h:%e nl:%ld order:%ld\n", context.elementName,
//           context.elementOccurrence, h, csbend->nonlinear, csbend->expansionOrder);
  }

  ttilt = tilt + etilt;
  if (ttilt == 0) {
    cos_ttilt = 1;
    sin_ttilt = 0;
  } else if (fabs(fabs(ttilt) - PI) < 1e-12) {
    cos_ttilt = -1;
    sin_ttilt = 0;
  } else if (fabs(ttilt - PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = 1;
  } else if (fabs(ttilt + PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = -1;
  } else {
    cos_ttilt = cos(ttilt);
    sin_ttilt = sin(ttilt);
  }

  dxi = dyi = dzi = 0;
  dxf = dyf = dzf = 0;
  if (csbend->malignMethod == 0) {
    computeEtiltCentroidOffset(dcoord_etilt, rho0, angle, etilt, tilt);

    dxi = -csbend->dx;
    dzi = csbend->dz;
    dyi = -csbend->dy;

    /* must use the original angle here because the translation is done after
     * the final rotation back
     */
    dxf = csbend->dx * cos(csbend->angle) + csbend->dz * sin(csbend->angle);
    dzf = csbend->dx * sin(csbend->angle) - csbend->dz * cos(csbend->angle);
    dyf = csbend->dy;
  } else {
    if (iSlice <= 0) {
      offsetParticlesForMisalignment(csbend->malignMethod, part, n_part,
                                     csbend->dx, csbend->dy, csbend->dz,
                                     csbend->epitch, csbend->eyaw, csbend->etilt, tilt, angle, csbend->length, 1);
    }
  }

  i_top = n_part - 1;
#if !defined(PARALLEL)
  multipoleKicksDone += n_part * csbend->nSlices * (csbend->integration_order == 4 ? 4 : 1);
#endif

  if (sigmaDelta2)
    *sigmaDelta2 = 0;

#ifdef HAVE_GPU
  if (gpuOmpTrackingEnabled(n_part) && !accepted &&
      globalLossCoordOffset <= 0 && !sigmaDelta2 && !spinCoordOffset &&
      !csbend->synch_rad && !csbend->isr &&
      !csbend->distributionBasedRadiation &&
      (!csbend->photonOutputFile || !csbend->photonOutputFile[0]) &&
      !csbend->referenceCorrection && refTrajectoryMode == 0 &&
      iSlice < 0 && csbend->malignMethod != 0 &&
      (!(csbend->edgeFlags & BEND_EDGE1_EFFECTS) ||
       csbend->edge_effects[csbend->e1Index] == 5) &&
      (!(csbend->edgeFlags & BEND_EDGE2_EFFECTS) ||
       csbend->edge_effects[csbend->e2Index] == 5)) {
    CSBEND_OMP_TRACK_DATA ompData;
    GPU_OMP_TRACKING_WORKSPACE *ompWorkspace;
    long ompParticles = n_part;
    long ompThreads = gpuGetOmpTrackingThreads();
#  if !defined(_OPENMP)
    (void)ompThreads;
#  endif

    ompWorkspace = gpuGetOmpTrackingWorkspace(ompParticles);
    memset(ompWorkspace->survived, 0,
           ompParticles * sizeof(*ompWorkspace->survived));
    ompData.csbend = csbend;
    ompData.apertureData = &apertureData;
    ompData.eptr = eptr;
    ompData.Po = Po;
    ompData.zStart = z_start;
    ompData.rho0 = rho0;
    ompData.rhoActual = rho_actual;
    ompData.e1 = e1;
    ompData.e2 = e2;
    ompData.cosTilt = cos_ttilt;
    ompData.sinTilt = sin_ttilt;
#  if defined(_OPENMP)
#    pragma omp taskloop num_tasks(ompThreads)
#  endif
    for (i_part = 0; i_part < ompParticles; i_part++) {
      double localDzLost = 0;
      ompWorkspace->survived[i_part] =
        trackCsbendParticleOmp(
          part[i_part], &ompData, &localDzLost) ? 1 : 0;
      ompWorkspace->lossOffset[i_part] = localDzLost;
    }
    i_top = gpuStableCompactParticles(
              part, ompParticles, ompWorkspace->survived) - 1;
    goto csbend_particle_loop_done;
  }
#endif

  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!part) {
      printf("error: null particle array found (working on particle %ld) (track_through_csbend)\n", i_part);
      fflush(stdout);
      abort();
    }
    if (!(coord = part[i_part])) {
      printf("error: null coordinate pointer for particle %ld (track_through_csbend)\n", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("error: null accepted particle pointer for particle %ld (track_through_csbend)\n", i_part);
      fflush(stdout);
      abort();
    }

    if (csbend->malignMethod == 0 && iSlice <= 0) {
      coord[4] += dzi * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      coord[0] = coord[0] + dxi + dzi * coord[1];
      coord[2] = coord[2] + dyi + dzi * coord[3];

      x = coord[0] * cos_ttilt + coord[2] * sin_ttilt;
      y = -coord[0] * sin_ttilt + coord[2] * cos_ttilt;
      xp = coord[1] * cos_ttilt + coord[3] * sin_ttilt;
      yp = -coord[1] * sin_ttilt + coord[3] * cos_ttilt;
      s = coord[4];
      dp = dp0 = coord[5];
      if (spinCoordOffset) {
	/* rotate spin into magnet coordinate system */
	/*
	fprintf(stderr, "Rotating spins by %le (%le, %le)\n", ttilt, cos_ttilt, sin_ttilt);
	fprintf(stderr, "Before: (%le, %le, %le)\n", coord[spinCoordOffset+0], coord[spinCoordOffset+1],
		coord[spinCoordOffset+2]);
	*/
	rotateSpinCoordinateSystem(coord+spinCoordOffset, cos_ttilt, sin_ttilt);
	/*
	fprintf(stderr, "After : (%le, %le, %le)\n", coord[spinCoordOffset+0], coord[spinCoordOffset+1],
		coord[spinCoordOffset+2]);
	*/
      }
    } else {
      x = coord[0];
      y = coord[2];
      xp = coord[1];
      yp = coord[3];
      s = coord[4];
      dp = dp0 = coord[5];
    }

    if (iSlice <= 0) {
      if (csbend->edgeFlags & BEND_EDGE1_EFFECTS) {
        delta_xp = 0;
        xp0 = xp;
        yp0 = yp;
        rho = (1 + dp) * rho_actual;
        if (csbend->edge_order <= 1 && csbend->edge_effects[csbend->e1Index] == 1) {
          /* apply edge focusing, nonsymplectic method */
          delta_xp = tan(e1) / rho * x;
          if (e1_kick_limit > 0 && fabs(delta_xp) > e1_kick_limit)
            delta_xp = SIGN(delta_xp) * e1_kick_limit;
          xp += delta_xp;
          yp -= tan(e1 - psi1 / (1 + dp)) / rho * y;
        } else if (csbend->edge_order >= 2 && csbend->edge_effects[csbend->e1Index] == 1) {
          /* apply edge focusing, nonsymplectic method */
          apply_edge_effects(&x, &xp, &y, &yp, rho, n, e1, he1, psi1 * (1 + dp), -1);
        } else if (csbend->edge_effects[csbend->e1Index] == 2) {
          /* K. Hwang's approach */
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          dipoleFringeKHwang(Qf, Qi, rho_actual, -1., csbend->edge_order, csbend->b[1] / rho0, e1, 2 * csbend->hgap,
                             csbend->fint[csbend->e1Index] >= 0 ? csbend->fint[csbend->e1Index] : csbend->fintBoth,
                             csbend->h[csbend->e1Index]);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          dp = Qf[5];
        } else if (csbend->edge_effects[csbend->e1Index] == 3) {
          /* simple-minded symplectic approach */
          applySimpleDipoleEdgeKick(&xp, &yp, x, y, dp, rho_actual, e1, psi1, e1_kick_limit, csbend->expandHamiltonian);
        } else if (csbend->edge_effects[csbend->e1Index] == 4) {
          /* K. Hwang's approach as symplectified by R. Lindberg */
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          dipoleFringeKHwangRLindberg(Qf, Qi, rho_actual, -1., csbend->b[1] / rho0, e1,
                                      2 * csbend->hgap,
                                      csbend->fint[csbend->e1Index] >= 0 ? csbend->fint[csbend->e1Index] : csbend->fintBoth,
                                      csbend->h[csbend->e1Index]);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          dp = Qf[5];
        } else if (csbend->edge_effects[csbend->e1Index] == 5) {
          /* New curved dipole treatment by R. Lindberg */
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          curvedDipoleFringe(Qf, Qi, rho_actual, -1, csbend->edge_order, csbend->b[1] / rho0, e1,
                             csbend->fringeInt[csbend->e1Index], csbend->edgeFlip);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          s += Qf[4];
          dp = Qf[5];
        }
        if (spinCoordOffset) {
	  /* Use the kick values to infer the spin rotations */
	  double P = Po*(1+coord[5]);
	  double gamma = sqrt(P*P+1);
	  performQuaternionRotation(coord+spinCoordOffset,
				    -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				    (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				    -(1+particleAnomalousMagneticMoment)*y/rho);
	}
      }
    }

    /* load input coordinates into arrays */
    Qi[0] = x;
    Qi[1] = xp;
    Qi[2] = y;
    Qi[3] = yp;
    Qi[4] = 0;
    Qi[5] = dp;

    if (iSlice <= 0) {
      if (csbend->edgeFlags & BEND_EDGE1_EFFECTS && e1 != 0 && rad_coef) {
        /* pre-adjust dp/p to anticipate error made by integrating over entire sector */
        computeCSBENDFields(&Fx, &Fy, x, y);

        dp_prime = -rad_coef * (sqr(Fx) + sqr(Fy)) * sqr(1 + dp) * sqrt(sqr(1 + x / rho0) + sqr(xp) + sqr(yp));
        Qi[5] -= dp_prime * x * tan(e1);
      }

      convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
    }

    if (csbend->expandHamiltonian)
      particle_lost = !integrate_csbend_ordn_expanded(Qf, Qi, sigmaDelta2, csbend->length, csbend->nSlices, iSlice, rho0, Po, &dz_lost,
                                                      &apertureData, csbend->integration_order, eptr);
    else
      particle_lost = !integrate_csbend_ordn(Qf, Qi, sigmaDelta2, spinCoordOffset?coord+spinCoordOffset:NULL,
					     csbend->length, csbend->nSlices, iSlice, rho0, Po, &dz_lost,
                                             &apertureData, csbend->integration_order, eptr);

    if (iSlice < 0 || iSlice == (csbend->nSlices - 1) || particle_lost) {
      if (csbend->fseCorrection == 1)
        Qf[4] -= csbend->fseCorrectionPathError;
      convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
    }

    if (particle_lost) {
      if (!part[i_top]) {
        printf("error: couldn't swap particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
               i_part, i_top);
        fflush(stdout);
        abort();
      }
      memcpy(part[i_part], Qf, sizeof(part[i_part][0]) * 6);
      convertFromCSBendCoords(part + i_part, 1, rho0, cos_ttilt, sin_ttilt, 0);
      swapParticles(part[i_part], part[i_top]);
      if (accepted) {
        if (!accepted[i_top]) {
          printf(
            "error: couldn't swap acceptance data for particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
            i_part, i_top);
          fflush(stdout);
          abort();
        }
        swapParticles(accepted[i_part], accepted[i_top]);
      }
      part[i_top][4] = z_start + dz_lost;
      part[i_top][5] = Po * (1 + part[i_top][5]);
      if (globalLossCoordOffset > 0)
        memcpy(part[i_top] + globalLossCoordOffset, Qf + globalLossCoordOffset, sizeof(double) * GLOBAL_LOSS_PROPERTIES_PER_PARTICLE);
      i_top--;
      i_part--;
      continue;
    }

    if (iSlice < 0 || iSlice == (csbend->nSlices - 1)) {
      if (csbend->edgeFlags & BEND_EDGE2_EFFECTS && e2 != 0 && rad_coef) {
        /* post-adjust dp/p to correct error made by integrating over entire sector */
        x = Qf[0];
        xp = Qf[1];
        y = Qf[2];
        yp = Qf[3];
        dp = Qf[5];

        computeCSBENDFields(&Fx, &Fy, x, y);

        dp_prime = -rad_coef * (sqr(Fx) + sqr(Fy)) * sqr(1 + dp) * sqrt(sqr(1 + x / rho0) + sqr(xp) + sqr(yp));
        Qf[5] -= dp_prime * x * tan(e2);
      }

      /* get final coordinates */
      if (rad_coef || isrConstant) {
        double p0, p1;
        double beta0, beta1;
        /* fix previous distance information to reflect new velocity--since distance
         * is really time-of-flight at the current velocity 
         */
        p0 = Po * (1 + dp0);
        beta0 = p0 / sqrt(sqr(p0) + 1);
        p1 = Po * (1 + Qf[5]);
        beta1 = p1 / sqrt(sqr(p1) + 1);
        s = beta1 * s / beta0 + Qf[4];
      } else
        s += Qf[4];
    } else
      s += Qf[4];
    x = Qf[0];
    xp = Qf[1];
    y = Qf[2];
    yp = Qf[3];
    dp = Qf[5];

    if (iSlice < 0 || iSlice == (csbend->nSlices - 1)) {
      if (csbend->edgeFlags & BEND_EDGE2_EFFECTS) {
        /* apply edge focusing */
        delta_xp = 0;
        xp0 = xp;
        yp0 = yp;
        rho = (1 + dp) * rho_actual;
        if (csbend->edge_order <= 1 && csbend->edge_effects[csbend->e2Index] == 1) {
          delta_xp = tan(e2) / rho * x;
          if (e2_kick_limit > 0 && fabs(delta_xp) > e2_kick_limit)
            delta_xp = SIGN(delta_xp) * e2_kick_limit;
          xp += delta_xp;
          yp -= tan(e2 - psi2 / (1 + dp)) / rho * y;
        } else if (csbend->edge_order >= 2 && csbend->edge_effects[csbend->e2Index] == 1) {
          apply_edge_effects(&x, &xp, &y, &yp, rho, n, e2, he2, psi2 * (1 + dp), 1);
        } else if (csbend->edge_effects[csbend->e2Index] == 2) {
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          dipoleFringeKHwang(Qf, Qi, rho_actual, 1., csbend->edge_order, csbend->b[1] / rho0, e2, 2 * csbend->hgap,
                             csbend->fint[csbend->e2Index] >= 0 ? csbend->fint[csbend->e2Index] : csbend->fintBoth,
                             csbend->h[csbend->e2Index]);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          dp = Qf[5];
        } else if (csbend->edge_effects[csbend->e2Index] == 3) {
          applySimpleDipoleEdgeKick(&xp, &yp, x, y, dp, rho_actual, e2, psi2, e2_kick_limit, csbend->expandHamiltonian);
        } else if (csbend->edge_effects[csbend->e2Index] == 4) {
          /* K. Hwang's approach as symplectified by R. Lindberg */
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          dipoleFringeKHwangRLindberg(Qf, Qi, rho_actual, 1., csbend->b[1] / rho0, e2, 2 * csbend->hgap,
                                      csbend->fint[csbend->e2Index] >= 0 ? csbend->fint[csbend->e2Index] : csbend->fintBoth,
                                      csbend->h[csbend->e2Index]);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          dp = Qf[5];
        } else if (csbend->edge_effects[csbend->e2Index] == 5) {
          /* New curved dipole treatment by R. Lindberg */
          /* load input coordinates into arrays */
          Qi[0] = x;
          Qi[1] = xp;
          Qi[2] = y;
          Qi[3] = yp;
          Qi[4] = 0;
          Qi[5] = dp;
          convertToDipoleCanonicalCoordinates(Qi, csbend->expandHamiltonian);
          curvedDipoleFringe(Qf, Qi, rho_actual, 1, csbend->edge_order, csbend->b[1] / rho0, e2,
                             csbend->fringeInt[csbend->e2Index], csbend->edgeFlip);
          /* retrieve coordinates from arrays */
          convertFromDipoleCanonicalCoordinates(Qf, csbend->expandHamiltonian);
          x = Qf[0];
          xp = Qf[1];
          y = Qf[2];
          yp = Qf[3];
          s += Qf[4];
          dp = Qf[5];
        }
        if (spinCoordOffset) {
	  /* Use the kick values to infer the spin rotations */
	  double P = Po*(1+coord[5]);
	  double gamma = sqrt(P*P+1);
	  performQuaternionRotation(coord+spinCoordOffset,
				    -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				    (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				    -(1+particleAnomalousMagneticMoment)*y/rho);
	}
      }
    }

    if (csbend->malignMethod == 0 && (iSlice < 0 || iSlice == (csbend->nSlices - 1))) {
      coord[0] = x * cos_ttilt - y * sin_ttilt + dcoord_etilt[0];
      coord[2] = x * sin_ttilt + y * cos_ttilt + dcoord_etilt[2];
      coord[1] = xp * cos_ttilt - yp * sin_ttilt + dcoord_etilt[1];
      coord[3] = xp * sin_ttilt + yp * cos_ttilt + dcoord_etilt[3];
      coord[4] = s + dcoord_etilt[4];
      coord[5] = dp;

      coord[0] += dxf + dzf * coord[1];
      coord[2] += dyf + dzf * coord[3];
      coord[4] += dzf * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      if (spinCoordOffset) {
	/* rotate spin out of  magnet coordinate system */
	/*
	fprintf(stderr, "Derotating spins by %le (%le, %le)\n", -ttilt, cos_ttilt, -sin_ttilt);
	fprintf(stderr, "Before: (%le, %le, %le)\n", coord[spinCoordOffset+0], coord[spinCoordOffset+1],
		coord[spinCoordOffset+2]);
	*/
	rotateSpinCoordinateSystem(coord+spinCoordOffset, cos_ttilt, -sin_ttilt);
	/*
	fprintf(stderr, "After : (%le, %le, %le)\n", coord[spinCoordOffset+0], coord[spinCoordOffset+1],
		coord[spinCoordOffset+2]);
	*/
      }
    } else {
      coord[0] = x;
      coord[2] = y;
      coord[1] = xp;
      coord[3] = yp;
      coord[4] = s;
      coord[5] = dp;
    }
  }

#ifdef HAVE_GPU
csbend_particle_loop_done:
#endif
  if (iSlice < 0 || iSlice == (csbend->nSlices - 1)) {
    if (csbend->malignMethod != 0)
      offsetParticlesForMisalignment(csbend->malignMethod, part, n_part,
                                     csbend->dx, csbend->dy, csbend->dz,
                                     csbend->epitch, csbend->eyaw, csbend->etilt, tilt, angle, csbend->length, 2);
  }

  if (distributionBasedRadiation) {
    radiansTotal += fabs(csbend->angle);
    /*
      fprintf(stderr, "%e radians, photons/particle=%e, photons/radian = %e, mean y = %e\n",
      radiansTotal, photonCount/(1.0*i_top), photonCount/radiansTotal/(1.0*i_top), energyCount/photonCount);
    */
    distributionBasedRadiation = 0;
  }

  /*
  for (i_part=i_top+1; i_part<n_part; i_part++) 
      fprintf(fpdeb, "%le %le\n", part[i_part][0], part[i_part][2]);
  */

  if (sigmaDelta2)
    /* Return average value for all particles */
    *sigmaDelta2 /= i_top + 1;

  if (csbend->photonOutputFile && !SDDS_UpdatePage(csbend->SDDSphotons, FLUSH_TABLE))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  return (i_top + 1);
}

void convertToDipoleCanonicalCoordinates(double *Qi, long expanded) {
  double f;
  if (expanded)
    f = (1 + Qi[5]);
  else
    f = (1 + Qi[5]) / sqrt(1 + sqr(Qi[1]) + sqr(Qi[3]));
  Qi[1] *= f;
  Qi[3] *= f;
}

void convertFromDipoleCanonicalCoordinates(double *Qi, long expanded) {
  double f;
  if (expanded)
    f = 1 / (1 + Qi[5]);
  else
    f = 1 / sqrt(sqr(1 + Qi[5]) - sqr(Qi[1]) - sqr(Qi[3]));
  Qi[1] *= f;
  Qi[3] *= f;
}

/* BETA is 2^(1/3) */
#define BETA 1.25992104989487316477

long integrate_csbend_ordn(
  double *Qf,                 /* final coordinates */
  double *Qi,                 /* initial coordinates */
  double *sigmaDelta2,        /* accumulate the energy spread increase for propagation of radiation matrix */
  double *spin,               /* if not NULL, three-component spin vector */
  double s,                   /* arc length */
  long n,                     /* number of slices */
  long iSlice,                         /* If <0, integrate the full magnet. If >=0, integrate just a single part and return.
                                       * This is needed to allow propagation of the radiation matrix. */
  double rho0,                /* nominal bending radius */
  double p0,                  /* central momentum */
  double *dz_lost,            /* return of loss position */
  MULT_APERTURE_DATA *apData, /* aperture data */
  short integration_order,    /* 2, 4, or 6 */
  ELEMENT_LIST *eptr
) {
  long i;
  double factor, f, phi, ds, dsh, dist;
  double Fx, Fy, x, y;
  double sine, cosi, tang;
  double sin_phi, cos_phi;
  //APCONTOUR *apContour;
  double fieldConversion;

  static double driftFrac2[2] = {
    0.5, 0.5};
  static double kickFrac2[2] = {
    1.0, 0.0};

  static double driftFrac4[4] = {
    0.5 / (2 - BETA), (1 - BETA) / (2 - BETA) / 2, (1 - BETA) / (2 - BETA) / 2, 0.5 / (2 - BETA)};
  static double kickFrac4[4] = {
    1. / (2 - BETA), -BETA / (2 - BETA), 1 / (2 - BETA), 0};

  /* From AOP-TN-2020-064 */
  static double driftFrac6[8] = {
    0.39225680523878,
    0.5100434119184585,
    -0.47105338540975655,
    0.0687531682525181,
    0.0687531682525181,
    -0.47105338540975655,
    0.5100434119184585,
    0.39225680523878,
  };
  static double kickFrac6[8] = {
    0.784513610477560, 0.235573213359357, -1.17767998417887, 1.3151863206839063,
    -1.17767998417887, 0.235573213359357, 0.784513610477560, 0};

  double *driftFrac = NULL, *kickFrac = NULL;
  long nSubsteps = 0;
  switch (integration_order) {
  case 2:
    nSubsteps = 2;
    driftFrac = driftFrac2;
    kickFrac = kickFrac2;
    break;
  case 4:
    nSubsteps = 4;
    driftFrac = driftFrac4;
    kickFrac = kickFrac4;
    break;
  case 6:
    nSubsteps = 8;
    driftFrac = driftFrac6;
    kickFrac = kickFrac6;
    break;
  default:
    bombElegantVA("invalid order %ld given for symplectic integrator", integration_order);
    break;
  }

#define X0 Qi[0]
#define XP0 Qi[1]
#define Y0 Qi[2]
#define YP0 Qi[3]
#define S0 Qi[4]
#define DPoP0 Qi[5]

#define X Qf[0]
#define QX Qf[1]
#define Y Qf[2]
#define QY Qf[3]
#define S Qf[4]
#define DPoP Qf[5]

  /* Factor to convert the normalzed field values (Fx,Fy) to (Bx, By) */  
  fieldConversion = -1/rho0/(particleCharge/(particleMass*c_mks*p0));
  
  if (refTrajectoryMode && refTrajectoryPoints != n)
    bombElegant("Problem with recorded reference trajectory for CSBEND element---has wrong number of points\n", NULL);
  if (!Qf)
    bombElegant("NULL final coordinates pointer ()", NULL);
  if (!Qi)
    bombElegant("NULL initial coordinates pointer (integrate_csbend_ordn)", NULL);
  if (n < 1)
    bombElegant("invalid number of steps (integrate_csbend_ordn)", NULL);

  memcpy(Qf, Qi, sizeof(*Qi) * 6);

  /*
  if (apData)
    apContour = apData->apContour;
  */

  dist = 0;
  s /= n;
  *dz_lost = 0; /* we'll accumulate this value even if the particle isn't lost */
  for (i = 0; i < n; i++) {
    long j;
    if (apData && !checkMultAperture(X, Y, i*s, apData)) {
      if (globalLossCoordOffset > 0) {
        double Xg, Yg, Zg, theta;
        double part[7];
        part[0] = Qf[0];
        part[2] = Qf[2];
        part[1] = part[3] = part[4] = part[5] = part[6] = 0; /* don't use these values */
        convertLocalCoordinatesToGlobal(&Zg, &Xg, &Yg, &theta, GLOBAL_LOCAL_MODE_SEG, part, eptr, 0.0, i, n);
        Qf[globalLossCoordOffset + 0] = Xg;
        Qf[globalLossCoordOffset + 1] = Zg;
        Qf[globalLossCoordOffset + 2] = theta;
      }
      return 0;
    }
    if (insideObstruction(Qf, GLOBAL_LOCAL_MODE_SEG, 0.0, i, n))
      return 0;
    for (j = 0; j < nSubsteps; j++) {
      /* do drift */
      dsh = s * driftFrac[j];
      f = sqr(1 + DPoP) - sqr(QY);
      if (f <= 0) {
        return 0;
      }
      f = sqrt(f);
      if (fabs(QX / f) > 1) {
        return 0;
      }
      sin_phi = QX / f;
      phi = asin(sin_phi);
      sine = sin(dsh / rho0 + phi);
      cosi = cos(dsh / rho0 + phi);
      if (cosi == 0) {
        return 0;
      }
      tang = sine / cosi;
      cos_phi = cos(phi);
      QX = f * sine;
      factor = (rho0 + X) * cos_phi / f * (tang - sin_phi / cos_phi);
      Y += QY * factor;
      dist += factor * (1 + DPoP);
      *dz_lost += dsh;
      f = cos_phi / cosi;
      X = rho0 * (f - 1) + f * X;

      if (kickFrac[j] == 0)
        break;
      /* do kick */
      ds = s * kickFrac[j];
      /* -- calculate the scaled fields */
      x = X;
      y = Y;

      computeCSBENDFields(&Fx, &Fy, x, y);

      /* --do kicks */
#if TURBO_RECIPROCALS
      double tmp = ds * (1 + X / rho0) / rho_actual;
      QX += -Fy * tmp;
      QY += Fx * tmp;
#else
      QX += -ds * (1 + X / rho0) * Fy / rho_actual;
      QY += ds * (1 + X / rho0) * Fx / rho_actual;
#endif

      if (spin) {
	/* --do spin, if needed */
	updateSpinQuaternionLocalFS(spin, Fx*fieldConversion, Fy*fieldConversion, 0, X, Y, QX, QY, ds, 1/rho0,
				    p0*(1+DPoP),  -particleCharge*particleRelSign,
				    particleMass, particleAnomalousMagneticMoment);
      }
      
      if (rad_coef || isrConstant) {
        if (!distributionBasedRadiation) {
#if TURBO_RECIPROCALS
          double f = (1 + X / rho0) / sqrt(sqr(1 + DPoP) - sqr(QX) - sqr(QY));
          double xp = QX * f;
          double yp = QY * f;
          double dsFactor = sqrt(sqr(1 + X / rho0) + sqr(xp) + sqr(yp));
#else
          double f = (1 + X * (1. / rho0)) / sqrt(sqr(1 + DPoP) - sqr(QX) - sqr(QY));
          double xp = QX * f;
          double yp = QY * f;
          double dsFactor = sqrt(sqr(1 + X * (1. / rho0)) + sqr(xp) + sqr(yp));
#endif
          double F2 = sqr(Fx) + sqr(Fy);
          double deltaFactor = sqr(1 + DPoP);
          double dsISR = s / (nSubsteps - 1);
#if TURBO_RECIPROCALS
          QX *= 1.0/(1 + DPoP);
          QY *= 1.0/(1 + DPoP);
#else
          QX /= (1 + DPoP);
          QY /= (1 + DPoP);
#endif
          if (rad_coef)
            DPoP -= rad_coef * deltaFactor * F2 * ds * dsFactor;
          if (isrConstant > 0)
            /* The minus sign is for consistency with the previous version. */
            DPoP -= isrConstant * deltaFactor * pow(F2, 0.75) * sqrt(dsISR * dsFactor) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
          if (sigmaDelta2)
            *sigmaDelta2 += sqr(isrConstant * deltaFactor) * pow(F2, 1.5) * dsISR * dsFactor;
          QX *= (1 + DPoP);
          QY *= (1 + DPoP);
        } else {
          addRadiationKick(&QX, &QY, &DPoP, sigmaDelta2,
                          X, Y, (i + 1. / 3) * s, s * n, 1. / rho0, Fx, Fy,
                          ds, rad_coef, s / (nSubsteps - 1), isrConstant,
                          distributionBasedRadiation, includeOpeningAngle,
                          meanPhotonsPerMeter0, normalizedCriticalEnergy0, p0);
        }
      }
    }

    if (refTrajectoryMode == RECORD_TRAJECTORY) {
      if (i >= refTrajectoryPoints) {
        TRACKING_CONTEXT context;
        getTrackingContext(&context);
        bombElegantVA("Problem with reference trajectory for %s#%ld: i=%ld, refTrajectoryPoints=%ld\n",
                      context.elementName, context.elementOccurrence, i, refTrajectoryPoints);
      }
      refTrajectoryData[i][0] = X;
      refTrajectoryData[i][1] = QX;
      refTrajectoryData[i][2] = Y;
      refTrajectoryData[i][3] = QY;
      refTrajectoryData[i][4] = dist - s;
      X = QX = Y = QY = dist = 0;
    }
    if (refTrajectoryMode == SUBTRACT_TRAJECTORY) {
      if (i >= refTrajectoryPoints) {
        TRACKING_CONTEXT context;
        getTrackingContext(&context);
        bombElegantVA("Problem with reference trajectory for %s#%ld: i=%ld, refTrajectoryPoints=%ld\n",
                      context.elementName, context.elementOccurrence, i, refTrajectoryPoints);
      }
      X -= refTrajectoryData[i][0];
      QX -= refTrajectoryData[i][1];
      Y -= refTrajectoryData[i][2];
      QY -= refTrajectoryData[i][3];
      dist -= refTrajectoryData[i][4];
    }
    if (iSlice >= 0)
      break;
  }

  *dz_lost = n * s;
  if (apData && !checkMultAperture(X, Y, i*s, apData)) {
    if (globalLossCoordOffset > 0) {
      double Xg, Yg, Zg, theta;
      double part[7];
      part[0] = Qf[0];
      part[2] = Qf[2];
      part[1] = part[3] = part[4] = part[5] = part[6] = 0; /* don't use these values */
      convertLocalCoordinatesToGlobal(&Zg, &Xg, &Yg, &theta, GLOBAL_LOCAL_MODE_SEG, part, eptr, 0.0, i, n);
      Qf[globalLossCoordOffset + 0] = Xg;
      Qf[globalLossCoordOffset + 1] = Zg;
      Qf[globalLossCoordOffset + 2] = theta;
    }
    return 0;
  }
  if (insideObstruction(Qf, GLOBAL_LOCAL_MODE_SEG, 0.0, i, n))
    return 0;

  Qf[4] += dist;
  return 1;
}

#if defined(HAVE_GPU) && defined(GPU_VERIFY)
long gpu_verify_csrcsbend_cpu_body_slice(double *coord, CSRCSBEND *csbend,
                                         double beta0, double sliceLength,
                                         double localRho0,
                                         double localRhoActual, double Po) {
  double Qi[6], Qf[6], dz_lost = 0;

  (void)localRhoActual;
  if (!coord || !csbend || beta0 == 0)
    return 0;
  Qi[0] = coord[0];
  Qi[1] = coord[1];
  Qi[2] = coord[2];
  Qi[3] = coord[3];
  Qi[4] = 0;
  Qi[5] = coord[5];
  convertToDipoleCanonicalCoordinates(Qi, 0);
  if (!integrate_csbend_ordn(Qf, Qi, NULL, NULL, sliceLength, 1, -1,
                             localRho0, Po, &dz_lost, NULL,
                             csbend->integration_order, NULL))
    return 0;
  convertFromDipoleCanonicalCoordinates(Qf, 0);
  coord[0] = Qf[0];
  coord[1] = Qf[1];
  coord[2] = Qf[2];
  coord[3] = Qf[3];
  coord[4] += Qf[4] / beta0;
  coord[5] = Qf[5];
  return 1;
}
#endif

long integrate_csbend_ordn_expanded(double *Qf, double *Qi, double *sigmaDelta2, double s, long n,
                                    long iSlice, double rho0, double p0, double *dz_lost,
                                    MULT_APERTURE_DATA *apData, short integration_order, ELEMENT_LIST *eptr)
/* The Hamiltonian in this case is approximated as
 * H = Hd + Hf, where Hd is the drift part and Hf is the field part.
 * Hd = Hd1 + Hd2 + Hd1, where
 * Hd1 = -0.5*(1+x/rho0)*(1+delta) 
 * Hd2 = 0.5*(qx^2+qy^2)/(1+delta)
 */
{
  long i;
  double ds, dsh, dist;
  double Fx, Fy, x, y;

  static double driftFrac2[2] = {
    0.5, 0.5};
  static double kickFrac2[2] = {
    1.0, 0.0};

  static double driftFrac4[4] = {
    0.5 / (2 - BETA), (1 - BETA) / (2 - BETA) / 2, (1 - BETA) / (2 - BETA) / 2, 0.5 / (2 - BETA)};
  static double kickFrac4[4] = {
    1. / (2 - BETA), -BETA / (2 - BETA), 1 / (2 - BETA), 0};

  /* From AOP-TN-2020-064 */
  static double driftFrac6[8] = {
    0.39225680523878,
    0.5100434119184585,
    -0.47105338540975655,
    0.0687531682525181,
    0.0687531682525181,
    -0.47105338540975655,
    0.5100434119184585,
    0.39225680523878,
  };
  static double kickFrac6[8] = {
    0.784513610477560, 0.235573213359357, -1.17767998417887, 1.3151863206839063,
    -1.17767998417887, 0.235573213359357, 0.784513610477560, 0};

  double *driftFrac = NULL, *kickFrac = NULL;
  long nSubsteps = 0;
  switch (integration_order) {
  case 2:
    nSubsteps = 2;
    driftFrac = driftFrac2;
    kickFrac = kickFrac2;
    break;
  case 4:
    nSubsteps = 4;
    driftFrac = driftFrac4;
    kickFrac = kickFrac4;
    break;
  case 6:
    nSubsteps = 8;
    driftFrac = driftFrac6;
    kickFrac = kickFrac6;
    break;
  default:
    bombElegantVA("invalid order %ld given for symplectic integrator", integration_order);
    break;
  }
#define X0 Qi[0]
#define XP0 Qi[1]
#define Y0 Qi[2]
#define YP0 Qi[3]
#define S0 Qi[4]
#define DPoP0 Qi[5]

#define X Qf[0]
#define QX Qf[1]
#define Y Qf[2]
#define QY Qf[3]
#define S Qf[4]
#define DPoP Qf[5]

  if (refTrajectoryMode && refTrajectoryPoints != n)
    bombElegant("Problem with recorded reference trajectory for CSBEND element---has wrong number of points\n", NULL);
  if (!Qf)
    bombElegant("NULL final coordinates pointer ()", NULL);
  if (!Qi)
    bombElegant("NULL initial coordinates pointer (integrate_csbend_ordn)", NULL);
  if (n < 1)
    bombElegant("invalid number of steps (integrate_csbend_ordn)", NULL);

  memcpy(Qf, Qi, sizeof(*Qi) * 6);

  dist = 0;
  s /= n;
  *dz_lost = 0; /* we'll accumulate this value even if the particle isn't lost */
  for (i = 0; i < n; i++) {
    long j;
    if ((apData && !checkMultAperture(X, Y, i*s, apData)) ||
        insideObstruction(Qf, GLOBAL_LOCAL_MODE_SEG, 0.0, i, n)) {
      return 0;
    }
    for (j = 0; j < nSubsteps; j++) {
      /* do drift */
      dsh = s * driftFrac[j];
      QX += dsh * (1 + DPoP) / (2 * rho0);
      dist += dsh * (1 + (sqr(QX) + sqr(QY)) / 2);
      *dz_lost += dsh;
      X += QX * dsh / (1 + DPoP);
      Y += QY * dsh / (1 + DPoP);
      QX += dsh * (1 + DPoP) / (2 * rho0);

      if (apData && !checkMultAperture(X, Y, i*s, apData)) {
        return 0;
      }

      if (kickFrac[j] == 0)
        break;
      /* do kick */
      ds = s * kickFrac[j];
      /* -- calculate the scaled fields */
      x = X;
      y = Y;

      computeCSBENDFields(&Fx, &Fy, x, y);

      /* --do kicks */
#if TURBO_RECIPROCALS
      QX += -(ds * (1 + X / rho0) / rho_actual) * Fy;
      QY += (ds * (1 + X / rho0) / rho_actual) * Fx;
#else
      QX += -ds * (1 + X / rho0) * Fy / rho_actual;
      QY += ds * (1 + X / rho0) * Fx / rho_actual;
#endif
      if (rad_coef || isrConstant) {
        if (!distributionBasedRadiation) {
#if TURBO_RECIPROCALS
          double f = (1 + X / rho0) / sqrt(sqr(1 + DPoP) - sqr(QX) - sqr(QY));
          double xp = QX * f;
          double yp = QY * f;
          double tmp = sqr(1 + X / rho0) / (sqr(1 + DPoP) - sqr(QX) - sqr(QY));
          double dsFactor = sqrt(sqr(1 + X / rho0) + sqr(QX) * tmp + sqr(QY) * tmp);
#else
          double f = (1 + X * (1. / rho0)) / sqrt(sqr(1 + DPoP) - sqr(QX) - sqr(QY));
          double xp = QX * f;
          double yp = QY * f;
          double dsFactor = sqrt(sqr(1 + X * (1. / rho0)) + sqr(xp) + sqr(yp));
#endif
          double F2 = sqr(Fx) + sqr(Fy);
          double deltaFactor = sqr(1 + DPoP);
          double dsISR = s / 3;
          QX /= (1 + DPoP);
          QY /= (1 + DPoP);
          if (rad_coef)
            DPoP -= rad_coef * deltaFactor * F2 * ds * dsFactor;
          if (isrConstant > 0)
            /* The minus sign is for consistency with the previous version. */
            DPoP -= isrConstant * deltaFactor * pow(F2, 0.75) * sqrt(dsISR * dsFactor) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
          if (sigmaDelta2)
            *sigmaDelta2 += sqr(isrConstant * deltaFactor) * pow(F2, 1.5) * dsISR * dsFactor;
          QX *= (1 + DPoP);
          QY *= (1 + DPoP);
        } else {
          addRadiationKick(&QX, &QY, &DPoP, sigmaDelta2,
                          X, Y, (i + 1. / 3) * s, s * n, 1. / rho0, Fx, Fy,
                          ds, rad_coef, s / 3, isrConstant,
                          distributionBasedRadiation, includeOpeningAngle,
                          meanPhotonsPerMeter0, normalizedCriticalEnergy0, p0);
        }
      }
    }

    if (refTrajectoryMode == RECORD_TRAJECTORY) {
      refTrajectoryData[i][0] = X;
      refTrajectoryData[i][1] = QX;
      refTrajectoryData[i][2] = Y;
      refTrajectoryData[i][3] = QY;
      refTrajectoryData[i][4] = dist - s;
      X = QX = Y = QY = dist = 0;
    }
    if (refTrajectoryMode == SUBTRACT_TRAJECTORY) {
      X -= refTrajectoryData[i][0];
      QX -= refTrajectoryData[i][1];
      Y -= refTrajectoryData[i][2];
      QY -= refTrajectoryData[i][3];
      dist -= refTrajectoryData[i][4];
    }
    if (iSlice >= 0)
      break;
  }
  if ((apData && !checkMultAperture(X, Y, i*s, apData)) ||
      insideObstruction(Qf, GLOBAL_LOCAL_MODE_SEG, 0.0, i, n)) {
    *dz_lost = n * s;
    return 0;
  }

  Qf[4] += dist;
  return 1;
}

CSR_LAST_WAKE csrWake;

static char *derbenevCriterionOption[N_DERBENEV_CRITERION_OPTIONS] = {
  "disable", "evaluate", "enforce"};

void readWakeFilterFile(long *values, double **freq, double **real, double **imag,
                        char *freqName, char *realName, char *imagName,
                        char *filename);

#ifdef GPU_VERIFY
static double csrVerifyEnvDouble(const char *name, double defaultValue) {
  char *endptr = NULL;
  const char *value = getenv(name);
  double result;

  if (!value || !*value)
    return defaultValue;
  result = strtod(value, &endptr);
  if (endptr == value)
    return defaultValue;
  return result;
}

static long csrVerifyEnvFlag(const char *name) {
  const char *value = getenv(name);

  return value && (!strcmp(value, "1") || !strcmp(value, "yes") ||
                   !strcmp(value, "true") || !strcmp(value, "on") ||
                   !strcmp(value, "YES") || !strcmp(value, "TRUE") ||
                   !strcmp(value, "ON"));
}

static long csrVerifyValuesClose(double cpu, double gpu,
                                 double absTol, double relTol,
                                 double *absDiffReturn,
                                 double *relDiffReturn) {
  double absDiff, relDiff, scale;

  if (isnan(cpu) || isnan(gpu)) {
    absDiff = relDiff = (isnan(cpu) && isnan(gpu)) ? 0 : DBL_MAX;
    if (absDiffReturn)
      *absDiffReturn = absDiff;
    if (relDiffReturn)
      *relDiffReturn = relDiff;
    return isnan(cpu) && isnan(gpu);
  }
  absDiff = fabs(cpu - gpu);
  scale = fmax(fabs(cpu), fabs(gpu));
  relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;
  if (absDiffReturn)
    *absDiffReturn = absDiff;
  if (relDiffReturn)
    *relDiffReturn = relDiff;
  return absDiff <= absTol || relDiff <= relTol;
}

static void csrCopyLastWakeForVerify(CSR_LAST_WAKE *target,
                                     const CSR_LAST_WAKE *source) {
  size_t bytes;

  memcpy(target, source, sizeof(*target));
  if (source->dGamma && source->bins > 0) {
    bytes = (size_t)source->bins * sizeof(*target->dGamma);
    target->dGamma = malloc(bytes);
    if (!target->dGamma)
      bombElegant("memory allocation failure copying CSR_LAST_WAKE dGamma for GPU verification", NULL);
    memcpy(target->dGamma, source->dGamma, bytes);
  }
}

static void csrReleaseLastWakeVerifyCopy(CSR_LAST_WAKE *wake) {
  free(wake->dGamma);
  wake->dGamma = NULL;
  if (wake->FdNorm) {
    free(wake->FdNorm);
    free(wake->xSaldin);
    wake->FdNorm = wake->xSaldin = NULL;
  }
  if (wake->StupakovFileActive) {
    if (!SDDS_Terminate(&wake->SDDS_Stupakov))
      bombElegant("problem terminating data file for Stupakov output from CSRDRIFT", NULL);
    wake->StupakovFileActive = 0;
  }
}

static void csrCompareLastWakeArray(const char *field,
                                    const double *cpu,
                                    const double *gpu,
                                    long n,
                                    double absTol,
                                    double relTol,
                                    long *mismatches,
                                    double *maxAbs,
                                    double *maxRel) {
  long i;

  if (n <= 0)
    return;
  if (!cpu || !gpu) {
    if (cpu != gpu) {
      fprintf(stderr,
              "elegant CUDA VERIFY CSR_LAST_WAKE mismatch %s pointer cpu=%p gpu=%p\n",
              field, (const void *)cpu, (const void *)gpu);
      (*mismatches)++;
    }
    return;
  }
  for (i = 0; i < n; i++) {
    double absDiff, relDiff;
    if (!csrVerifyValuesClose(cpu[i], gpu[i], absTol, relTol,
                              &absDiff, &relDiff)) {
      if (*mismatches < 10)
        fprintf(stderr,
                "elegant CUDA VERIFY CSR_LAST_WAKE mismatch %s[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                field, i, cpu[i], gpu[i], absDiff, relDiff);
      (*mismatches)++;
    }
    if (absDiff > *maxAbs)
      *maxAbs = absDiff;
    if (relDiff > *maxRel)
      *maxRel = relDiff;
  }
}

static void compareCSR_LAST_WAKE(const CSR_LAST_WAKE *gpuWake,
                                 const CSR_LAST_WAKE *cpuWake) {
  double absTol = csrVerifyEnvDouble("ELEGANT_GPU_WAKE_COMPARE_ABS",
                                     csrVerifyEnvDouble("ELEGANT_GPU_COMPARE_ABS", 1e-9));
  double relTol = csrVerifyEnvDouble("ELEGANT_GPU_WAKE_COMPARE_REL",
                                     csrVerifyEnvDouble("ELEGANT_GPU_COMPARE_REL", 1e-10));
  double maxAbs = 0, maxRel = 0;
  long mismatches = 0;

#  define CSR_COMPARE_LONG(field)                                             \
  do {                                                                        \
    if (cpuWake->field != gpuWake->field) {                                   \
      if (mismatches < 10)                                                    \
        fprintf(stderr,                                                       \
                "elegant CUDA VERIFY CSR_LAST_WAKE mismatch %s cpu=%ld gpu=%ld\n", \
                #field, (long)cpuWake->field, (long)gpuWake->field);          \
      mismatches++;                                                           \
    }                                                                         \
  } while (0)

#  define CSR_COMPARE_ULONG(field)                                            \
  do {                                                                        \
    if (cpuWake->field != gpuWake->field) {                                   \
      if (mismatches < 10)                                                    \
        fprintf(stderr,                                                       \
                "elegant CUDA VERIFY CSR_LAST_WAKE mismatch %s cpu=%lu gpu=%lu\n", \
                #field, (unsigned long)cpuWake->field,                        \
                (unsigned long)gpuWake->field);                               \
      mismatches++;                                                           \
    }                                                                         \
  } while (0)

#  define CSR_COMPARE_DOUBLE(field)                                           \
  do {                                                                        \
    double absDiff, relDiff;                                                  \
    if (!csrVerifyValuesClose(cpuWake->field, gpuWake->field, absTol, relTol, \
                              &absDiff, &relDiff)) {                         \
      if (mismatches < 10)                                                    \
        fprintf(stderr,                                                       \
                "elegant CUDA VERIFY CSR_LAST_WAKE mismatch %s cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n", \
                #field, cpuWake->field, gpuWake->field, absDiff, relDiff);    \
      mismatches++;                                                           \
    }                                                                         \
    if (absDiff > maxAbs)                                                     \
      maxAbs = absDiff;                                                       \
    if (relDiff > maxRel)                                                     \
      maxRel = relDiff;                                                       \
  } while (0)

  CSR_COMPARE_ULONG(lastMode);
  CSR_COMPARE_LONG(bins);
  CSR_COMPARE_LONG(valid);
  CSR_COMPARE_DOUBLE(dctBin);
  CSR_COMPARE_DOUBLE(s0);
  CSR_COMPARE_DOUBLE(ds0);
  CSR_COMPARE_DOUBLE(zLast);
  CSR_COMPARE_DOUBLE(z0);
  CSR_COMPARE_DOUBLE(S11);
  CSR_COMPARE_DOUBLE(S12);
  CSR_COMPARE_DOUBLE(S22);
  CSR_COMPARE_DOUBLE(rho);
  CSR_COMPARE_DOUBLE(bendingAngle);
  CSR_COMPARE_DOUBLE(Po);
  CSR_COMPARE_DOUBLE(perc68BunchLength);
  CSR_COMPARE_DOUBLE(perc90BunchLength);
  CSR_COMPARE_DOUBLE(peakToPeakWavelength);
  CSR_COMPARE_DOUBLE(rmsBunchLength);
  CSR_COMPARE_LONG(nSaldin);
  if (cpuWake->nSaldin)
    CSR_COMPARE_DOUBLE(lastFdNorm);
  CSR_COMPARE_LONG(SGOrder);
  CSR_COMPARE_LONG(SGHalfWidth);
  CSR_COMPARE_LONG(SGDerivHalfWidth);
  CSR_COMPARE_LONG(SGDerivOrder);
  CSR_COMPARE_DOUBLE(binRangeFactor);
  CSR_COMPARE_DOUBLE(GSConstant);
  CSR_COMPARE_DOUBLE(MPCharge);
  CSR_COMPARE_LONG(StupakovFileActive);
  CSR_COMPARE_LONG(StupakovOutputInterval);
  CSR_COMPARE_LONG(trapazoidIntegration);
  CSR_COMPARE_DOUBLE(lowFrequencyCutoff0);
  CSR_COMPARE_DOUBLE(lowFrequencyCutoff1);
  CSR_COMPARE_DOUBLE(highFrequencyCutoff0);
  CSR_COMPARE_DOUBLE(highFrequencyCutoff1);
  CSR_COMPARE_LONG(clipNegativeBins);
  CSR_COMPARE_LONG(wffValues);

  csrCompareLastWakeArray("dGamma", cpuWake->dGamma, gpuWake->dGamma,
                          cpuWake->bins == gpuWake->bins ? cpuWake->bins : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);
  csrCompareLastWakeArray("FdNorm", cpuWake->FdNorm, gpuWake->FdNorm,
                          cpuWake->nSaldin == gpuWake->nSaldin ? cpuWake->nSaldin : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);
  csrCompareLastWakeArray("xSaldin", cpuWake->xSaldin, gpuWake->xSaldin,
                          cpuWake->nSaldin == gpuWake->nSaldin ? cpuWake->nSaldin : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);
  csrCompareLastWakeArray("wffFreqValue", cpuWake->wffFreqValue,
                          gpuWake->wffFreqValue,
                          cpuWake->wffValues == gpuWake->wffValues ? cpuWake->wffValues : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);
  csrCompareLastWakeArray("wffRealFactor", cpuWake->wffRealFactor,
                          gpuWake->wffRealFactor,
                          cpuWake->wffValues == gpuWake->wffValues ? cpuWake->wffValues : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);
  csrCompareLastWakeArray("wffImagFactor", cpuWake->wffImagFactor,
                          gpuWake->wffImagFactor,
                          cpuWake->wffValues == gpuWake->wffValues ? cpuWake->wffValues : 0,
                          absTol, relTol, &mismatches, &maxAbs, &maxRel);

  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY CSR_LAST_WAKE failed: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (csrVerifyEnvFlag("ELEGANT_GPU_VERBOSE"))
    fprintf(stderr,
            "elegant CUDA VERIFY CSR_LAST_WAKE passed: maxAbs=%.3e maxRel=%.3e\n",
            maxAbs, maxRel);

#  undef CSR_COMPARE_LONG
#  undef CSR_COMPARE_ULONG
#  undef CSR_COMPARE_DOUBLE
}
#endif

#if defined(HAVE_GPU) && !USE_MPI
static long csrDriftUsesCsrState(CSRDRIFT *csrDrift) {
  return csrDrift && (csrDrift->csr || csrDrift->LSCBins);
}

static long csrDriftUsesStupakovOnly(CSRDRIFT *csrDrift) {
  return csrDrift && csrDrift->csr && csrDrift->useStupakov &&
         !csrDrift->LSCBins && !csrDrift->linearOptics;
}

static long csrDriftNeedsCsbendFinalPrep(CSRDRIFT *csrDrift) {
  if (!csrDriftUsesCsrState(csrDrift))
    return 0;
  return !csrDriftUsesStupakovOnly(csrDrift);
}

static long csrDriftCsbendFinalPrepNeededFollows(ELEMENT_LIST *eptr) {
  ELEMENT_LIST *next;

  if (!eptr)
    return 0;
  for (next = eptr->succ; next; next = next->succ) {
    if (next->type == T_CSRCSBEND)
      return 0;
    if (next->type == T_CSRDRIFT &&
        csrDriftNeedsCsbendFinalPrep((CSRDRIFT *)next->p_elem))
      return 1;
  }
  return 0;
}

static long csrDriftStupakovOnlyStateConsumerFollows(ELEMENT_LIST *eptr) {
  ELEMENT_LIST *next;

  if (!eptr)
    return 0;
  for (next = eptr->succ; next; next = next->succ) {
    if (next->type == T_CSRCSBEND)
      return 0;
    if (next->type == T_CSRDRIFT &&
        csrDriftUsesStupakovOnly((CSRDRIFT *)next->p_elem))
      return 1;
  }
  return 0;
}

static long csrResidentSimpleFinalTransformAvailable(
  CSRCSBEND *csbend, double e2, double psi2, double cos_ttilt,
  double sin_ttilt, double dxf, double dyf, double dzf,
  const double *dcoord_etilt) {
  long i;

  if (!csbend || !dcoord_etilt)
    return 0;
  if ((cos_ttilt != 1 && cos_ttilt != -1) || sin_ttilt != 0)
    return 0;
  if (dxf != 0 || dyf != 0 || dzf != 0)
    return 0;
  for (i = 0; i < 5; i++)
    if (dcoord_etilt[i] != 0)
      return 0;
  if (!(csbend->edgeFlags & BEND_EDGE2_EFFECTS))
    return 1;
  if (csbend->edge_order > 1)
    return 0;
  return csbend->edge_effects[csbend->e2Index] == 0 ||
         csbend->edge_effects[csbend->e2Index] == 1;
}

static long csrResidentSimpleInitialTransformAvailable(
  CSRCSBEND *csbend, double e1, double he1, double psi1,
  double cos_ttilt, double sin_ttilt, double dxi, double dyi,
  double dzi) {
  (void)e1;
  (void)he1;
  (void)psi1;

  if (!csbend)
    return 0;
  if ((cos_ttilt != 1 && cos_ttilt != -1) || sin_ttilt != 0)
    return 0;
  if (dxi != 0 || dyi != 0 || dzi != 0)
    return 0;
  if (!(csbend->edgeFlags & BEND_EDGE1_EFFECTS))
    return 1;
  if (csbend->edge_order > 1)
    return 0;
  return csbend->edge_effects[csbend->e1Index] == 0 ||
         csbend->edge_effects[csbend->e1Index] == 1;
}
#endif

#ifdef HAVE_GPU
long track_through_csbendCSR_cuda_resident_entry(
  double **part, int64_t n_part, CSRCSBEND *csbend, double p_error,
  double Po, double **accepted, double z_start, double z_end,
  CHARGE *charge, char *rootname, MAXAMP *maxamp, APCONTOUR *apContour,
  APERTURE_DATA *apFileData, ELEMENT_LIST *eptr) {
  long result;

  csrGpuCsbendInnerCall++;
  csrGpuCsbendDeviceEntryRequested++;
  result = track_through_csbendCSR(part, n_part, csbend, p_error, Po,
                                   accepted, z_start, z_end, charge,
                                   rootname, maxamp, apContour,
                                   apFileData, eptr);
  csrGpuCsbendDeviceEntryRequested--;
  csrGpuCsbendInnerCall--;
  return result;
}
#endif

long track_through_csbendCSR(double **part, int64_t n_part, CSRCSBEND *csbend, double p_error,
                             double Po, double **accepted, double z_start, double z_end,
                             CHARGE *charge, char *rootname, MAXAMP *maxamp, APCONTOUR *apContour,
                             APERTURE_DATA *apFileData, ELEMENT_LIST *eptr) {
  double h, n, he1, he2;
  static long csrWarning = 0;
  static double *beta0 = NULL, *ctHist = NULL, *ctHistDeriv = NULL;
  static double *dGamma = NULL, *T1 = NULL, *T2 = NULL, *denom = NULL, *chik = NULL, *grnk = NULL;
  static long maxParticles = 0, maxBins = 0;
#if defined(HAVE_GPU)
  static double *denomCachePointer = NULL;
  static long denomCacheBins = 0;
  static double denomCacheDct = 0;
#endif
  char particleLost;
  double x = 0, xp, y = 0, yp, p1, beta1, p0;
  double ctLower, ctUpper, dct, slippageLength, phiBend, slippageLength13;
  long diSlippage, diSlippage4;
  long nBins, nBinned = 0;
  int64_t i_part, i_top;
  long kick, j;
  double rho = 0.0, Fx, Fy;
  double fse, dp_prime;
  double tilt, etilt, cos_ttilt, sin_ttilt, ttilt;
  double *coord;
  double angle, e1, e2, Kg;
  double psi1, psi2;
  double Qi[6], Qf[6];
  double dcoord_etilt[MAX_PROPERTIES_PER_PARTICLE];
  double dxi, dyi, dzi;
  double dxf, dyf, dzf;
  double delta_xp;
  double macroParticleCharge, CSRConstant, gamma2, gamma3;
  long iBin, iBinBehind;
  long csrInhibit = 0;
#if defined(HAVE_GPU) && !USE_MPI
  long csrDgammaHostCurrent = 1;
#endif
#if defined(HAVE_GPU) && !USE_MPI
  long useGpuCsrResident = 0;
  long usedGpuCsrResident = 0;
  long gpuCsrResidentDeviceEntryDone = 0;
  long gpuCsrResidentFinalDone = 0;
  long skipGpuCsrDriftPrep = 0;
#endif
  double derbenevRatio = 0;
  long n_partMoreThanOne = 0;
  TRACKING_CONTEXT tContext;
  VMATRIX *Msection = NULL, *Me1 = NULL, *Me2 = NULL;
  static double accumulatedAngle = 0;
  short accumulatingAngle = 1;
  double dz_lost = 0;
  MULT_APERTURE_DATA apertureData;
#if USE_MPI
  double *buffer;
#endif
#ifdef DEBUG_IGF
  FILE *fpdeb;
  fpdeb = fopen("csr.sdds", "w");
  fprintf(fpdeb, "SDDS1\n&parameter name = Slice, type=long &end\n");
  fprintf(fpdeb, "&column name=s, type=double, units=m &end\n");
  fprintf(fpdeb, "&column name=iBin, type=long &end\n");
  fprintf(fpdeb, "&column name=Chi, type=double &end\n");
  fprintf(fpdeb, "&column name=G, units=V/m, type=double &end\n");
  fprintf(fpdeb, "&column name=dGamma, type=double &end\n");
  fprintf(fpdeb, "&data mode=ascii &end\n");
#endif

#ifdef HAVE_GPU
  if (getElementOnGpu() && !csrGpuCsbendInnerCall) {
    startGpuTimer();
    i_part = gpu_track_through_csbendCSR(n_part, csbend, p_error, Po, accepted, z_start, z_end, charge, rootname, maxamp, apContour, apFileData, eptr);
#  ifdef GPU_VERIFY
    startCpuTimer();
    /* Copy the csrWake global struct (it is reset below) */
    CSR_LAST_WAKE gpuCsrWake;
    csrCopyLastWakeForVerify(&gpuCsrWake, &csrWake);
    csrWake.FdNorm = NULL;          /* Reset doesn't deallocate */
    csrWake.xSaldin = NULL;
    csrWake.StupakovFileActive = 0; /* Reset doesn't close */

    track_through_csbendCSR(part, n_part, csbend, p_error, Po, accepted, z_start, z_end, charge, rootname, maxamp, apContour, apFileData, eptr);
    compareGpuCpu(n_part, "track_through_csbendCSR");
    compareCSR_LAST_WAKE(&gpuCsrWake, &csrWake);
    csrReleaseLastWakeVerifyCopy(&gpuCsrWake);
#  endif /* GPU_VERIFY */
    return i_part;
  }
#endif /* HAVE_GPU */

#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  gpu_clear_csr_wake_cpu_shadow();
#endif

  gamma2 = Po * Po + 1;
  gamma3 = pow(gamma2, 3. / 2);

#if USE_MPI
  if (distributedBeam)
    n_partMoreThanOne = 1; /* This is necessary to solve synchronization issue in parallel version*/
  else if (n_part > 1)
    n_partMoreThanOne = 1;
#else
  if (n_part > 1)
    n_partMoreThanOne = 1;
#endif

  if (!(csbend->edgeFlags & SAME_BEND_PRECEDES))
    accumulatedAngle = accumulatingAngle = 0;

  csrWake.valid = 0;
  refTrajectoryMode = 0;
  if (isSlave || !distributedBeam)
    reset_driftCSR();

  getTrackingContext(&tContext);

  if (!csbend)
    bombElegant("null CSRCSBEND pointer (track_through_csbend)", NULL);
  if (csbend->integratedGreensFunction && !csbend->steadyState)
    bombElegant("CSRCSBEND requires STEADYSTATE=1 if IGF=1.", NULL);
  if (csbend->edge_order > 1 && (csbend->edge_effects[csbend->e1Index] == 2 || csbend->edge_effects[csbend->e2Index] == 2) && csbend->hgap == 0)
    bombElegant("CSRCSBEND has EDGE_ORDER>1 and EDGE[12]_EFFECTS==2, but HGAP=0. This gives undefined results.", NULL);

  if (csbend->angle == 0) {
    if (!csbend->useMatrix)
      exactDrift(part, n_part, csbend->length);
    else {
      int64_t i;
      if (isSlave || !distributedBeam) {
        for (i = 0; i < n_part; i++) {
          part[i][0] += csbend->length * part[i][1];
          part[i][2] += csbend->length * part[i][3];
          part[i][4] += csbend->length;
        }
      }
    }
    return n_part;
  }

  if (csbend->integration_order != 2 && csbend->integration_order != 4 && csbend->integration_order != 6)
    bombElegant("CSBEND integration_order is invalid--must be either 2, 4, or 6", NULL);

  macroParticleCharge = 0;
  if (charge) {
    macroParticleCharge = charge->macroParticleCharge;
  } else if (csbend->bins && !csrWarning && csbend->csr) {
    printf("Warning: you asked for CSR on CSBEND but didn't give a CHARGE element\n");
    fflush(stdout);
    csrWarning = 1;
  }

  if ((nBins = csbend->bins) < 2)
    bombElegant("Less than 2 bins for CSR!", NULL);

  if (csbend->SGDerivHalfWidth <= 0)
    csbend->SGDerivHalfWidth = csbend->SGHalfWidth;
  if (csbend->SGDerivHalfWidth <= 0)
    csbend->SGDerivHalfWidth = 1;

  if (csbend->SGDerivOrder <= 0)
    csbend->SGDerivOrder = csbend->SGOrder;
  if (csbend->SGDerivOrder <= 0)
    csbend->SGDerivOrder = 1;

  if (isSlave || !distributedBeam)
    if (n_part > maxParticles &&
        (!(beta0 = SDDS_Realloc(beta0, sizeof(*beta0) * (maxParticles = n_part)))))
      bombElegant("Memory allocation failure (track_through_csbendCSR)", NULL);

  rho0 = csbend->length / csbend->angle;
  if (csbend->use_bn) {
    csbend->b[0] = 0;
    csbend->b[1] = csbend->b1;
    csbend->b[2] = csbend->b2;
    csbend->b[3] = csbend->b3;
    csbend->b[4] = csbend->b4;
    csbend->b[5] = csbend->b5;
    csbend->b[6] = csbend->b6;
    csbend->b[7] = csbend->b7;
    csbend->b[8] = csbend->b8;
  } else {
    csbend->b[0] = 0;
    csbend->b[1] = csbend->k1 * rho0;
    csbend->b[2] = csbend->k2 * rho0;
    csbend->b[3] = csbend->k3 * rho0;
    csbend->b[4] = csbend->k4 * rho0;
    csbend->b[5] = csbend->k5 * rho0;
    csbend->b[6] = csbend->k6 * rho0;
    csbend->b[7] = csbend->k7 * rho0;
    csbend->b[8] = csbend->k8 * rho0;
  }
  for (j = 0; j < 9; j++)
    csbend->c[j] = 0;

  he1 = csbend->h[csbend->e1Index];
  he2 = csbend->h[csbend->e2Index];
  if (csbend->angle < 0) {
    long i;
    angle = -csbend->angle;
    e1 = -csbend->e[csbend->e1Index];
    e2 = -csbend->e[csbend->e2Index];
    etilt = csbend->etilt * csbend->etiltSign;
    tilt = csbend->tilt + PI;
    rho0 = csbend->length / angle;
    for (i = 1; i < 9; i += 2) {
      csbend->b[i] *= -1;
      csbend->c[i] *= -1;
    }
  } else {
    angle = csbend->angle;
    e1 = csbend->e[csbend->e1Index];
    e2 = csbend->e[csbend->e2Index];
    etilt = csbend->etilt * csbend->etiltSign;
    tilt = csbend->tilt;
    rho0 = csbend->length / angle;
  }

  setupMultApertureData(&apertureData, -tilt, apContour, maxamp, apFileData, NULL, z_start + csbend->length / 2, eptr);

  if (rho0 > 1e6) {
    printWarningForTracking("CSRCSBEND has radius > 1e6 but non-zero K1.",
                            "Treated as EDRIFT.");
    exactDrift(part, n_part, csbend->length);
    return n_part;
  }

  h = 1 / rho0;
  n = -csbend->b[1] / h;
  fse = csbend->fse;
  if (fse > -1)
    rho_actual = 1 / ((1 + fse) * h);
  else
    rho_actual = 1e16 / h;

  /* angles for fringe-field effects */
  Kg = 2 * csbend->hgap * csbend->fint;
  psi1 = Kg / rho_actual / cos(e1) * (1 + sqr(sin(e1)));
  psi2 = Kg / rho_actual / cos(e2) * (1 + sqr(sin(e2)));
  if (csbend->length < 0) {
    psi1 *= -1;
    psi2 *= -1;
  }

  /* rad_coef is d((P-Po)/Po)/ds for the on-axis, on-momentum particle, where po is the momentum of
   * the central particle.
   */
  if (csbend->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) * sqr(1 + fse) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho0));
  else
    rad_coef = 0;
  /* isrConstant is the RMS increase in dP/P per meter due to incoherent SR.  */
  if (csbend->isr && (n_part > 1 || !csbend->isr1Particle))
    isrConstant = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) *
                                        137.0359895 / pow3(fabs(rho_actual)));
  else
    isrConstant = 0;

  distributionBasedRadiation = 0;

  if (csbend->useMatrix) {
    csbend->nonlinear = 0;
    Me1 = edge_matrix(e1, 1. / (rho0 / (1 + csbend->fse)), 0.0, n, -1, Kg, 1, 0, 0, csbend->length);
    Msection = bend_matrix(csbend->length / csbend->nSlices,
                           angle / csbend->nSlices, 0.0, 0.0,
                           0.0, 0.0, csbend->b[1] * h, 0.0,
                           0.0, 0.0, 0.0, 0.0, csbend->fse, 0.0, 0.0,
                           csbend->etilt * csbend->etiltSign, 1, 1, 0, 0, 0.0, 0.0);
    Me2 = edge_matrix(e2, 1. / (rho0 / (1 + csbend->fse)), 0.0, n, 1, Kg, 1, 0, 0, csbend->length);
  }
  computeCSBENDFieldCoefficients(csbend->b, csbend->c, h, csbend->nonlinear, csbend->expansionOrder);

  ttilt = tilt + etilt;
  if (ttilt == 0) {
    cos_ttilt = 1;
    sin_ttilt = 0;
  } else if (fabs(fabs(ttilt) - PI) < 1e-12) {
    cos_ttilt = -1;
    sin_ttilt = 0;
  } else if (fabs(ttilt - PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = 1;
  } else if (fabs(ttilt + PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = -1;
  } else {
    cos_ttilt = cos(ttilt);
    sin_ttilt = sin(ttilt);
  }

  if (etilt)
    computeEtiltCentroidOffset(dcoord_etilt, rho0, angle, etilt, tilt);
  else
    fill_double_array(dcoord_etilt, 6L, 0.0);

  dxi = -csbend->dx;
  dzi = csbend->dz;
  dyi = -csbend->dy;

  /* must use the original angle here because the translation is done after
   * the final rotation back
   */
  dxf = csbend->dx * cos(csbend->angle) + csbend->dz * sin(csbend->angle);
  dzf = csbend->dx * sin(csbend->angle) - csbend->dz * cos(csbend->angle);
  dyf = csbend->dy;

  if (isMaster) {
    if (csbend->particleOutputFile && strlen(csbend->particleOutputFile) && !csbend->particleFileActive) {
      char *filename;
      /* set up SDDS output file for particle coordinates inside bend */
      csbend->particleFileActive = 1;
      filename = compose_filename(csbend->particleOutputFile, rootname);
      csbend->SDDSpart = tmalloc(sizeof(*(csbend->SDDSpart)));
      if (!SDDS_InitializeOutputElegant(csbend->SDDSpart, SDDS_BINARY, 1,
                                        NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(csbend->SDDSpart, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSpart, "Pass", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSpart, "Kick", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSpart, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSpart, "Angle", NULL, SDDS_DOUBLE) ||
          (csbend->xIndex = SDDS_DefineColumn(csbend->SDDSpart, "x", NULL, "m",
                                              NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (csbend->xpIndex = SDDS_DefineColumn(csbend->SDDSpart, "xp", NULL, NULL,
                                               NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (csbend->tIndex = SDDS_DefineColumn(csbend->SDDSpart, "t", NULL, "s",
                                              NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (csbend->pIndex = SDDS_DefineColumn(csbend->SDDSpart, "p", NULL, "m$be$nc",
                                              NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          !SDDS_WriteLayout(csbend->SDDSpart)) {
        SDDS_SetError("Problem setting up particle output file for CSR");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      free(filename);
    }
  }

  if (isMaster) {
    if (csbend->histogramFile && strlen(csbend->histogramFile) && !csbend->wakeFileActive) {
      /* set up SDDS output file for CSR monitoring */
      char *filename;
      csbend->wakeFileActive = 1;
      filename = compose_filename(csbend->histogramFile, rootname);
      csbend->SDDSout = tmalloc(sizeof(*(csbend->SDDSout)));
      if (!SDDS_InitializeOutputElegant(csbend->SDDSout, SDDS_BINARY, 1, NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(csbend->SDDSout, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "Pass", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "Kick", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "Angle", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "SlippageLength", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "TotalBunchLength", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "BinSize", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "dsKick", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(csbend->SDDSout, "DerbenevRatio", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "s", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "LinearDensity", "C/s", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "LinearDensityDeriv", "C/s$a2$n", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "DeltaGamma", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "GammaDeriv", "1/m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "DeltaGammaT1", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(csbend->SDDSout, "DeltaGammaT2", NULL, SDDS_DOUBLE) ||
          !SDDS_WriteLayout(csbend->SDDSout)) {
        SDDS_SetError("Problem setting up wake output file for CSR");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      free(filename);
    }
  }
  if (csbend->wakeFilterFile && strlen(csbend->wakeFilterFile) && !csbend->wffValues)
    readWakeFilterFile(&csbend->wffValues,
                       &csbend->wffFreqValue, &csbend->wffRealFactor, &csbend->wffImagFactor,
                       csbend->wffFreqColumn, csbend->wffRealColumn, csbend->wffImagColumn,
                       csbend->wakeFilterFile);

  /*  prepare arrays for CSR integrals */
  nBins = csbend->bins;
  if (!(ctHist = SDDS_Realloc(ctHist, sizeof(*ctHist) * nBins)) ||
      !(ctHistDeriv = SDDS_Realloc(ctHistDeriv, sizeof(*ctHistDeriv) * nBins)) ||
      !(denom = SDDS_Realloc(denom, sizeof(*denom) * nBins)) ||
      !(T1 = SDDS_Realloc(T1, sizeof(*T1) * nBins)) ||
      !(T2 = SDDS_Realloc(T2, sizeof(*T2) * nBins)) ||
      !(dGamma = SDDS_Realloc(dGamma, sizeof(*dGamma) * nBins)))
    bombElegant("memory allocation failure (track_through_csbendCSR)", NULL);

#if !defined(PARALLEL)
  multipoleKicksDone += n_part * csbend->nSlices * (csbend->integration_order == 4 ? 4 : 1);
#endif

#if defined(HAVE_GPU) && !USE_MPI
  if (csrGpuCsbendDeviceEntryRequested) {
    if (n_partMoreThanOne &&
        !maxamp && !apContour && (!apFileData || !apFileData->initialized) &&
        gpu_csr_csbend_resident_available(csbend, n_part, nBins) &&
        csrResidentSimpleInitialTransformAvailable(
          csbend, e1, he1, psi1, cos_ttilt, sin_ttilt, dxi, dyi, dzi))
      gpuCsrResidentDeviceEntryDone =
        gpu_track_csbend_csr_enter_simple(
          n_part, Po, cos_ttilt,
          (csbend->edgeFlags & BEND_EDGE1_EFFECTS) &&
            csbend->edge_effects[csbend->e1Index] == 1,
          e1, psi1, rho_actual);
    if (!gpuCsrResidentDeviceEntryDone)
      forceParticlesToCpu("CSRCSBEND resident initial CPU fallback");
  }
#endif

#if defined(HAVE_GPU) && !USE_MPI
  if (!gpuCsrResidentDeviceEntryDone && (isSlave || !distributedBeam)) {
#else
  if (isSlave || !distributedBeam) {
#endif
    /* check particle data, transform coordinates, and handle edge effects */
    for (i_part = 0; i_part < n_part; i_part++) {
      if (!part) {
        printf("error: null particle array found (working on particle %ld) (track_through_csbend)\n", i_part);
        fflush(stdout);
        abort();
      }
      if (!(coord = part[i_part])) {
        printf("error: null coordinate pointer for particle %ld (track_through_csbend)\n", i_part);
        fflush(stdout);
        abort();
      }
      if (accepted && !accepted[i_part]) {
        printf("error: null accepted particle pointer for particle %ld (track_through_csbend)\n", i_part);
        fflush(stdout);
        abort();
      }

      /* adjust for element offsets */
      coord[4] += dzi * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      coord[0] = coord[0] + dxi + dzi * coord[1];
      coord[2] = coord[2] + dyi + dzi * coord[3];

      /* perform tilt transformations and save some data */
      x = coord[0] * cos_ttilt + coord[2] * sin_ttilt;
      y = -coord[0] * sin_ttilt + coord[2] * cos_ttilt;
      coord[0] = x;
      coord[2] = y;
      xp = coord[1] * cos_ttilt + coord[3] * sin_ttilt;
      yp = -coord[1] * sin_ttilt + coord[3] * cos_ttilt;
      coord[1] = xp;
      coord[3] = yp;
      p0 = Po * (1 + coord[5]);
      beta0[i_part] = p0 / sqrt(p0 * p0 + 1);
      coord[4] /= beta0[i_part];

#undef X
#undef Y
#define X coord[0]
#define Y coord[2]
#define XP coord[1]
#define YP coord[3]
#define CT coord[4]
#define DP coord[5]
      if (csbend->edgeFlags & BEND_EDGE1_EFFECTS) {
        /* apply edge focusing */
        if (csbend->useMatrix)
          track_particles(&coord, Me1, &coord, 1);
        else {
          rho = (1 + DP) * rho_actual;
          if (csbend->edge_order <= 1 && csbend->edge_effects[csbend->e1Index] == 1) {
            /* apply edge focusing, nonsymplectic method */
            delta_xp = tan(e1) / rho * X;
            XP += delta_xp;
            YP -= tan(e1 - psi1 / (1 + DP)) / rho * Y;
          } else if (csbend->edge_order >= 2 && csbend->edge_effects[csbend->e1Index] == 1)
            apply_edge_effects(&X, &XP, &Y, &YP, rho, n, e1, he1, psi1 * (1 + DP), -1);
          else if (csbend->edge_effects[csbend->e1Index] == 2) {
            rho = (1 + DP) * rho_actual;
            /* load input coordinates into arrays */
            Qi[0] = X;
            Qi[1] = XP;
            Qi[2] = Y;
            Qi[3] = YP;
            Qi[4] = 0;
            Qi[5] = DP;
            convertToDipoleCanonicalCoordinates(Qi, 0);
            dipoleFringeKHwang(Qf, Qi, rho_actual, -1., csbend->edge_order, csbend->b[1] / rho0, e1, 2 * csbend->hgap, csbend->fint, csbend->h[csbend->e1Index]);
            /* retrieve coordinates from arrays */
            convertFromDipoleCanonicalCoordinates(Qf, 0);
            X = Qf[0];
            XP = Qf[1];
            Y = Qf[2];
            YP = Qf[3];
            DP = Qf[5];
          } else if (csbend->edge_effects[csbend->e1Index] == 3) {
            applySimpleDipoleEdgeKick(&XP, &YP, X, Y, DP, rho_actual, e1, psi1, -1.0, 0);
          }
        }
      }

      if (csbend->edgeFlags & BEND_EDGE1_EFFECTS && e1 != 0 && rad_coef) {
        /* pre-adjust dp/p to anticipate error made by integrating over entire sector */
        computeCSBENDFields(&Fx, &Fy, X, Y);

        dp_prime = -rad_coef * (sqr(Fx) + sqr(Fy)) * sqr(1 + DP) *
                   sqrt(sqr(1 + X / rho0) + sqr(XP) + sqr(YP));
        DP -= dp_prime * X * tan(e1);
      }
    }
  }
#if defined(HAVE_GPU) && !USE_MPI
  useGpuCsrResident =
    n_partMoreThanOne &&
    !maxamp && !apContour && (!apFileData || !apFileData->initialized) &&
    gpu_csr_csbend_resident_available(csbend, n_part, nBins);
  if (useGpuCsrResident && !gpuCsrResidentDeviceEntryDone &&
      !gpu_csr_csbend_resident_begin(beta0, n_part))
    useGpuCsrResident = 0;
  if (useGpuCsrResident)
    usedGpuCsrResident = 1;
#endif
  if (csbend->csr && n_partMoreThanOne)
    CSRConstant = 2 * macroParticleCharge * particleCharge / pow(3 * rho0 * rho0, 1. / 3.) / (4 * PI * epsilon_o * particleMass * sqr(c_mks));
  else
    CSRConstant = 0;
  /* Now do the body of the sector dipole */
  phiBend = accumulatedAngle;
  i_top = n_part - 1;
  for (kick = 0; kick < (csbend->nSlices + 1); kick++) {
    if (!csbend->backtrack && kick == csbend->nSlices)
      break;
    if (isSlave || !distributedBeam) {
      if (!csbend->backtrack || kick != 0) {
#if defined(HAVE_GPU) && !USE_MPI
        long gpuCsrResidentBodyDone = 0;
        if (useGpuCsrResident) {
          gpuCsrResidentBodyDone =
            gpu_track_csbend_csr_body_slice(csbend, n_part,
                                            csbend->length / csbend->nSlices,
                                            rho0, rho_actual, Po);
          if (!gpuCsrResidentBodyDone) {
            forceParticlesToCpu("CSRCSBEND resident body fallback");
            if (gpuCsrResidentDeviceEntryDone &&
                !gpu_copy_csbend_csr_beta0(beta0, n_part))
              bombElegant("unable to copy CUDA CSRCSBEND beta0 for body fallback", NULL);
            useGpuCsrResident = 0;
          }
        }
        if (!gpuCsrResidentBodyDone) {
#endif
        for (i_part = 0; i_part <= i_top; i_part++) {
          coord = part[i_part];

          if (csbend->useMatrix) {
            track_particles(&coord, Msection, &coord, 1);
          } else {
            /* load input coordinates into arrays */
            Qi[0] = X;
            Qi[1] = XP;
            Qi[2] = Y;
            Qi[3] = YP;
            Qi[4] = 0;
            Qi[5] = DP;
            convertToDipoleCanonicalCoordinates(Qi, 0);

            particleLost = !integrate_csbend_ordn(Qf, Qi, NULL, NULL, csbend->length / csbend->nSlices, 1, -1, rho0, Po, 
                                                  &dz_lost, &apertureData, csbend->integration_order, eptr);

            /* retrieve coordinates from arrays */
            convertFromDipoleCanonicalCoordinates(Qf, 0);
            X = Qf[0];
            XP = Qf[1];
            Y = Qf[2];
            YP = Qf[3];
            DP = Qf[5];

            if (particleLost) {
              if (!part[i_top]) {
                printf("error: couldn't swap particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
                       i_part, i_top);
                fflush(stdout);
                abort();
              }
              memcpy(part[i_part], Qf, sizeof(part[i_part][0]) * 6);
              convertFromCSBendCoords(part + i_part, 1, rho0, cos_ttilt, sin_ttilt, 0);
              swapParticles(part[i_part], part[i_top]);
              if (accepted) {
                if (!accepted[i_top]) {
                  printf(
                    "error: couldn't swap acceptance data for particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
                    i_part, i_top);
                  fflush(stdout);
                  abort();
                }
                swapParticles(accepted[i_part], accepted[i_top]);
              }
              part[i_top][4] = z_start + dz_lost;
              part[i_top][5] = Po * (1 + part[i_top][5]);

              i_top--;
              i_part--;
            } else {
              if (rad_coef || isrConstant) {
                /* convert additional distance traveled to ct using mean velocity */
                p1 = Po * (1 + DP);
                beta1 = p1 / sqrt(p1 * p1 + 1);
                CT += Qf[4] * 2 / (beta0[i_part] + beta1);
                beta0[i_part] = beta1;
              } else
                CT += Qf[4] / beta0[i_part];
            }
          }
        }
        n_part = i_top + 1;
#if defined(HAVE_GPU) && !USE_MPI
        }
#endif
      }
    }

    if (csbend->backtrack && kick == csbend->nSlices)
      break;

    if (n_partMoreThanOne && csbend->derbenevCriterionMode) {
      /* evaluate Derbenev criterion from TESLA-FEL 1995-05: sigma_x/sigma_z << (R/sigma_z)^(1/3) */
      long code;
      double Sz, Sx;
      switch (code = match_string(csbend->derbenevCriterionMode, derbenevCriterionOption, N_DERBENEV_CRITERION_OPTIONS, 0)) {
      case DERBENEV_CRITERION_DISABLE:
        break;
      case DERBENEV_CRITERION_EVAL:
      case DERBENEV_CRITERION_ENFORCE:
#if !USE_MPI
        rms_emittance(part, 4, 5, n_part, &Sz, NULL, NULL, NULL, NULL);
        rms_emittance(part, 0, 1, n_part, &Sx, NULL, NULL, NULL, NULL);
#else
        if (distributedBeam) {
          /* The master will get the result from the rms_emittance routine */
          rms_emittance_p(part, 4, 5, n_part, &Sz, NULL, NULL, NULL, NULL, NULL);
          rms_emittance_p(part, 0, 1, n_part, &Sx, NULL, NULL, NULL, NULL, NULL);
        } else {
          rms_emittance(part, 4, 5, n_part, &Sz, NULL, NULL, NULL, NULL);
          rms_emittance(part, 0, 1, n_part, &Sx, NULL, NULL, NULL, NULL);
        }
#endif
        Sz = sqrt(Sz);
        Sx = sqrt(Sx);
        derbenevRatio = (Sx / Sz) / pow(rho0 / Sz, 1. / 3.);
        if (derbenevRatio > 0.1) {
          if (code == DERBENEV_CRITERION_EVAL) {
            printWarningForTracking("Using 1-D CSR formalism but Derbenev criterion not satisfied (ratio > 0.1).",
                                    "CSR applied regardless per setting of DERBENEV_CRITERION_MODE.");
          } else {
            csrInhibit = 1;
            printWarningForTracking("Using 1-D CSR formalism but Derbenev criterion not satisfied (ratio > 0.1).",
                                    "CSR not applied per setting of DERBENEV_CRITERION_MODE.");
          }
        }
        break;
      default:
        fprintf(stderr, "Error: invalid value for DERBENEV_CRITERION_MODE. Give 'disable', 'evaluate', or 'enforce'\n");
        exit(1);
        break;
      }
    }

#if (!USE_MPI)
    if (n_partMoreThanOne && !csrInhibit) {
#else
    if (!csrInhibit && (distributedBeam || (!distributedBeam && n_partMoreThanOne))) { /* n_part could be 0 for some processors, which could cause synchronization problem */
#endif
#if defined(HAVE_GPU) && !USE_MPI
      long useGpuCsrResidentKick = 0;
#endif
      /* compute CSR potential function */
      if (kick == 0 || !csbend->binOnce) {
	long lastMaxBins;
	lastMaxBins = maxBins;
        /* - first make a density histogram */
        ctLower = ctUpper = 0;
	if ((dct = c_mks*csbend->binSize) > 0) {
	  /* maxBins = nBins; */
	  nBins = 0;
	}
#if defined(HAVE_GPU) && !USE_MPI
        nBinned = -1;
        if (useGpuCsrResident) {
          double gpuCtLower = ctLower, gpuCtUpper = ctUpper, gpuDct = dct;
          long gpuNBins = nBins;
          if (gpu_prepare_csbend_csr_histogram_device(
                &gpuCtLower, &gpuCtUpper, &gpuDct, &gpuNBins,
                csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor,
                n_part, Po) >= 0) {
            if (gpuNBins > maxBins) {
              maxBins = gpuNBins;
              if (!(ctHist = SDDS_Realloc(ctHist, sizeof(*ctHist) * maxBins)) ||
                  !(ctHistDeriv = SDDS_Realloc(ctHistDeriv, sizeof(*ctHistDeriv) * maxBins)) ||
                  !(denom = SDDS_Realloc(denom, sizeof(*denom) * maxBins)) ||
                  !(T1 = SDDS_Realloc(T1, sizeof(*T1) * maxBins)) ||
                  !(T2 = SDDS_Realloc(T2, sizeof(*T2) * maxBins)) ||
                  !(dGamma = SDDS_Realloc(dGamma, sizeof(*dGamma) * maxBins)))
                bombElegant("memory allocation failure (track_through_csbendCSR)", NULL);
            }
            nBinned = gpu_compute_csbend_csr_histogram_device(ctHist,
                                                              n_part, gpuNBins,
                                                              gpuCtLower,
                                                              gpuDct);
            if (nBinned >= 0) {
              ctLower = gpuCtLower;
              ctUpper = gpuCtUpper;
              dct = gpuDct;
              nBins = gpuNBins;
            }
          }
          if (nBinned < 0) {
            forceParticlesToCpu("CSRCSBEND resident histogram fallback");
            if (gpuCsrResidentDeviceEntryDone &&
                !gpu_copy_csbend_csr_beta0(beta0, n_part))
              bombElegant("unable to copy CUDA CSRCSBEND beta0 for histogram fallback", NULL);
            useGpuCsrResident = 0;
          }
        }
        if (nBinned < 0) {
          ctLower = ctUpper = 0;
          if ((dct = c_mks*csbend->binSize) > 0)
            nBins = 0;
#endif
        nBinned = binParticleCoordinate(&ctHist, &maxBins,
                                        &ctLower, &ctUpper, &dct, &nBins,
                                        csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor,
                                        part, n_part, 4);
#if defined(HAVE_GPU) && !USE_MPI
        }
#endif
	if (lastMaxBins<maxBins) {
	  if (!(ctHistDeriv = SDDS_Realloc(ctHistDeriv, sizeof(*ctHistDeriv) * maxBins)) ||
	      !(denom = SDDS_Realloc(denom, sizeof(*denom) * maxBins)) ||
	      !(T1 = SDDS_Realloc(T1, sizeof(*T1) * maxBins)) ||
	      !(T2 = SDDS_Realloc(T2, sizeof(*T2) * maxBins)) ||
	      !(dGamma = SDDS_Realloc(dGamma, sizeof(*dGamma) * maxBins))) {
	    bombElegant("memory allocation failure (track_through_csbendCSR)", NULL);
	  }
	}
#if (!USE_MPI)
        if (nBinned != n_part) {
          printf("Only %ld of %ld particles binned for CSRCSBEND (z0=%le, kick=%ld, BRF=%le)\n",
                 nBinned, n_part, z_start, kick, csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor);
          printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
                 ctLower, ctUpper, dct, nBins, maxBins);
          fflush(stdout);
        }
#else
        if (distributedBeam) {
          if (USE_MPI) {
            long all_binned, result = 1, nBinned_total;

            if (isSlave || !distributedBeam) {
              result = ((nBinned == n_part) ? 1 : 0);
            }
            MPI_Allreduce(&result, &all_binned, 1, MPI_LONG, MPI_LAND, MPI_COMM_WORLD);
            MPI_Allreduce(&nBinned, &nBinned_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            nBinned = nBinned_total;
            if (!all_binned && isMaster) {
              printf("Not all particles binned for CSRCSBEND (z0=%le, kick=%ld, BRF=%le)\n",
                     z_start, kick,
                     csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor);
              printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
                     ctLower, ctUpper, dct, nBins, maxBins);
              fflush(stdout);
            }
          }

          if (USE_MPI) { /* Master needs to know the information to write the result */
            buffer = malloc(sizeof(double) * nBins);
            MPI_Allreduce(ctHist, buffer, nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            memcpy(ctHist, buffer, sizeof(double) * nBins);
            free(buffer);
          }
        }
#endif

        /* - smooth the histogram, normalize to get linear density, and 
           copy in preparation for taking derivative
           */
        if (csbend->highFrequencyCutoff0 > 0 || csbend->lowFrequencyCutoff0 >= 0) {
          long nz;
          nz = applyLHPassFilters(ctHist, nBins,
                                  csbend->lowFrequencyCutoff0, csbend->lowFrequencyCutoff1,
                                  csbend->highFrequencyCutoff0, csbend->highFrequencyCutoff1,
                                  csbend->clipNegativeBins);
          if (nz) {
            char warningText[1024];
            snprintf(warningText, 1024, "Negative values in %ld bins.", nz);
            printWarningForTracking("Low pass filter resulted in negative values.", warningText);
          }
        }
        if (csbend->SGHalfWidth > 0) {
          SavitzyGolaySmooth(ctHist, nBins, csbend->SGOrder, csbend->SGHalfWidth, csbend->SGHalfWidth, 0);
          correctDistribution(ctHist, nBins, 1.0 * nBinned);
        }
#if defined(HAVE_GPU)
        if (denomCachePointer != denom || denomCacheDct != dct || denomCacheBins < nBins) {
          for (iBin = 0; iBin < nBins; iBin++)
            denom[iBin] = pow(dct * iBin, 1. / 3.);
          denomCachePointer = denom;
          denomCacheDct = dct;
          denomCacheBins = nBins;
        }
        for (iBin = 0; iBin < nBins; iBin++)
          ctHistDeriv[iBin] = (ctHist[iBin] /= dct);
#else
        for (iBin = 0; iBin < nBins; iBin++) {
          denom[iBin] = pow(dct * iBin, 1. / 3.);
          ctHistDeriv[iBin] = (ctHist[iBin] /= dct);
        }
#endif
        /* - compute derivative with smoothing.  The deriv is w.r.t. index number and
         * I won't scale it now as it will just fall out in the integral 
         */
        SavitzyGolaySmooth(ctHistDeriv, nBins, csbend->SGDerivOrder,
                           csbend->SGDerivHalfWidth, csbend->SGDerivHalfWidth, 1);
      } else {
        ctLower += rho0 * angle / csbend->nSlices;
        ctUpper += rho0 * angle / csbend->nSlices;
      }

      phiBend += angle / csbend->nSlices;
      slippageLength = fabs(rho0 * ipow3(phiBend) / 24.0);
      slippageLength13 = pow(slippageLength, 1. / 3.);
      diSlippage = slippageLength / dct;
      diSlippage4 = 4 * slippageLength / dct;
      if (kick == 0 || !csbend->binOnce) {
        if (csbend->integratedGreensFunction) {
          /* Integrated Greens function method */
          double const2;
          double z, xmu, a, b, frac, const1;
          if (kick == 0) {
            if (!csbend->steadyState)
              bombElegant("Must have STEADY_STATE=1 when IGF=1\n", NULL);
            if (!(grnk = SDDS_Realloc(grnk, sizeof(*grnk) * nBins)) ||
                !(chik = SDDS_Realloc(chik, sizeof(*chik) * nBins)))
              bombElegant("memory allocation failure (track_through_csbendCSR)", NULL);
          }
          frac = 9.0 / 16.0;
          const1 = 6.0 - log(27.0 / 4.0);
          for (iBin = 0; iBin < nBins; iBin++) {
            z = iBin * dct;
            xmu = 3.0 * gamma3 * z / (2.0 * rho0);
            a = sqrt(xmu * xmu + 1.0);
            b = a + xmu;
            if (xmu < 1e-3)
              chik[iBin] = frac * const1 + 0.50 * ipow2(xmu) - (7.0 / 54.0) * ipow4(xmu) + (140.0 / 2187.0) * ipow6(xmu);
            else
              chik[iBin] = frac * (3.0 * (-2.0 * xmu * pow(b, 1.0 / 3.0) + pow(b, 2.0 / 3.0) + pow(b, 4.0 / 3.0)) +
                                   log(pow((1 - pow(b, 2.0 / 3.0)) / xmu, 2) / (1 + pow(b, 2.0 / 3.0) + pow(b, 4.0 / 3.0))));
          }
          const2 = (16.0 / 27.0) * (particleCharge / (4 * PI * epsilon_o)) / (gamma2 * dct);
          grnk[0] = const2 * (chik[1] - chik[0]);
          for (iBin = 1; iBin < nBins - 1; iBin++)
            grnk[iBin] = const2 * (chik[iBin + 1] - 2.0 * chik[iBin] + chik[iBin - 1]);
          grnk[nBins - 1] = 0;
        } else {
#ifdef HAVE_GPU
          long returnCsrWakeComponents =
            csbend->wakeFileActive &&
            ((!csbend->outputLastWakeOnly && kick % csbend->outputInterval == 0) ||
             (csbend->outputLastWakeOnly && kick == (csbend->nSlices - 1)));
          long copyCsrDgammaToHost = 1;
#  if !USE_MPI
          useGpuCsrResidentKick =
            !csbend->backtrack && !csbend->wffValues && CSRConstant &&
            useGpuCsrResident &&
            gpu_csr_csbend_resident_available(csbend, n_part, nBins);
          copyCsrDgammaToHost =
            !useGpuCsrResidentKick || returnCsrWakeComponents ||
            (kick == (csbend->nSlices - 1));
#  endif
          if (!gpu_compute_csbend_csr_wake(dGamma, T1, T2,
                                           ctHist, ctHistDeriv, denom,
                                           n_part, nBins, CSRConstant,
                                           csbend->length / csbend->nSlices,
                                           slippageLength13, dct,
                                           csbend->steadyState,
                                           csbend->trapazoidIntegration,
                                           diSlippage, diSlippage4,
                                           returnCsrWakeComponents,
                                           copyCsrDgammaToHost)) {
#  if !USE_MPI
            useGpuCsrResidentKick = 0;
#  endif
#endif
          for (iBin = 0; iBin < nBins; iBin++) {
            double term1, term2;
            long count;
            T1[iBin] = T2[iBin] = 0;
            term1 = term2 = 0;
            if (CSRConstant) {
              if (csbend->steadyState) {
                if (!csbend->integratedGreensFunction) {
                  if (!csbend->trapazoidIntegration) {
                    for (iBinBehind = iBin + 1; iBinBehind < nBins; iBinBehind++)
                      T1[iBin] += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
                  } else {
                    if ((iBinBehind = iBin + 1) < nBins)
                      term1 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
                    for (count = 0, iBinBehind = iBin + 1; iBinBehind < nBins; iBinBehind++, count++)
                      T1[iBin] += (term2 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin]);
                    if ((iBin + 1) < nBins)
                      T1[iBin] += 0.3 * sqr(denom[1]) * (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) / dct;
                    if (count > 1)
                      T1[iBin] -= (term1 + term2) / 2;
                  }
                }
              } else {
                /* Transient CSR */
                if (!csbend->trapazoidIntegration) {
                  for (iBinBehind = iBin + 1; iBinBehind <= (iBin + diSlippage) && iBinBehind < nBins; iBinBehind++)
                    T1[iBin] += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
                } else {
                  if ((iBinBehind = iBin + 1) < nBins && iBinBehind <= (iBin + diSlippage))
                    term1 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin] / 2;
                  for (count = 0, iBinBehind = iBin + 1; iBinBehind <= (iBin + diSlippage) && iBinBehind < nBins;
                       count++, iBinBehind++)
                    T1[iBin] += (term2 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin]);
                  if (diSlippage > 0 && (iBin + 1) < nBins)
                    T1[iBin] += 0.3 * sqr(denom[1]) * (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) / dct;
                  if (count > 1)
                    T1[iBin] -= (term1 + term2) / 2;
                }
                if ((iBin + diSlippage) < nBins)
                  T2[iBin] += ctHist[iBin + diSlippage];
                if ((iBin + diSlippage4) < nBins)
                  T2[iBin] -= ctHist[iBin + diSlippage4];
              }
              /* there is no negative sign here because my derivative is w.r.t. -s
                 in notation of Saldin, et. al. */
              T1[iBin] *= CSRConstant * csbend->length / csbend->nSlices;
              /* keep the negative sign on this term, which has no derivative */
              T2[iBin] *= -CSRConstant * csbend->length / csbend->nSlices / slippageLength13;
            }
            dGamma[iBin] = T1[iBin] + T2[iBin];
          }
#if defined(HAVE_GPU) && !USE_MPI
          csrDgammaHostCurrent = 1;
#endif
#ifdef HAVE_GPU
          } else {
#  if !USE_MPI
            csrDgammaHostCurrent = copyCsrDgammaToHost;
#  endif
          }
#endif
        }

        if (csbend->integratedGreensFunction) {
          convolveArrays1(dGamma, nBins, ctHist, grnk);
          for (iBin = 0; iBin < nBins; iBin++)
            dGamma[iBin] *= -macroParticleCharge / (particleMass * sqr(c_mks)) * csbend->length / csbend->nSlices;
#ifdef DEBUG_IGF
          fprintf(fpdeb, "%ld\n%ld\n", kick, nBins);
          for (iBin = 0; iBin < nBins; iBin++)
            fprintf(fpdeb, "%le %ld %le %le %le\n", iBin * dct, iBin, chik[iBin], grnk[iBin], dGamma[iBin]);
#endif
        }

        if (csbend->wffValues)
          applyFilterTable(dGamma, nBins, dct / c_mks, csbend->wffValues, csbend->wffFreqValue,
                           csbend->wffRealFactor, csbend->wffImagFactor);
      }
      if (isSlave || !distributedBeam) {
        if (CSRConstant) {
#if defined(HAVE_GPU) && !USE_MPI
          if (useGpuCsrResidentKick &&
              gpu_apply_csbend_csr_kick_device(n_part, nBins, ctLower, dct, Po, rho0)) {
            if (!csrDgammaHostCurrent && kick == (csbend->nSlices - 1)) {
              if (!gpu_copy_csbend_csr_dgamma(dGamma, nBins))
                bombElegant("unable to copy CUDA CSR dGamma for final CSRCSBEND state", NULL);
              csrDgammaHostCurrent = 1;
            }
          } else {
            if (useGpuCsrResident) {
              forceParticlesToCpu("CSRCSBEND resident kick fallback");
              if (gpuCsrResidentDeviceEntryDone &&
                  !gpu_copy_csbend_csr_beta0(beta0, n_part))
                bombElegant("unable to copy CUDA CSRCSBEND beta0 for kick fallback", NULL);
              useGpuCsrResident = 0;
            }
            if (!csrDgammaHostCurrent) {
              if (!gpu_copy_csbend_csr_dgamma(dGamma, nBins))
                bombElegant("unable to copy CUDA CSR dGamma for CPU CSR kick fallback", NULL);
              csrDgammaHostCurrent = 1;
            }
#endif
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(n_part))
#endif
          for (i_part = 0; i_part < n_part; i_part++) {
            long localBin, nBins1;
            double f;
            double *localCoord;
            nBins1 = nBins - 1;
            localCoord = part[i_part];
            /* apply CSR kick */
            localBin = (f = (localCoord[4] - ctLower) / dct);
            f -= localBin;
            if (localBin >= 0 && localBin < nBins1) {
              localCoord[5] += ((1 - f) * dGamma[localBin] + f * dGamma[localBin + 1]) / Po * (1 + localCoord[0] / rho0);
              /* This code probably should be uncommented, but makes very little difference.
              p1 = Po*(1+DP);
              beta1 = p1/sqrt(p1*p1+1);
              CT *= beta0[i_part]/beta1;
              beta0[i_part] = beta1;
              */
            }
          }
#if defined(HAVE_GPU) && !USE_MPI
          }
#endif
        }
      }

      if (csbend->particleFileActive && kick % csbend->particleOutputInterval == 0) {
        if (isMaster) {
          int64_t ip;
          /* dump particle data at this location */
          if (!SDDS_StartPage(csbend->SDDSpart, n_part) ||
              !SDDS_SetParameters(csbend->SDDSpart, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "Pass", -1, "Kick", kick, "pCentral", Po, "Angle", phiBend,
                                  NULL))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          convertFromCSBendCoords(part, n_part, rho0, cos_ttilt, sin_ttilt, 1);
          for (ip = 0; ip < n_part; ip++) {
            if (!SDDS_SetRowValues(csbend->SDDSpart, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                   ip,
                                   csbend->xIndex, part[ip][0],
                                   csbend->xpIndex, part[ip][1],
                                   csbend->tIndex, part[ip][4],
                                   csbend->pIndex, Po * (1 + part[ip][5]),
                                   -1))
              SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          }
          convertToCSBendCoords(part, n_part, rho0, cos_ttilt, sin_ttilt, 1);
          if (!SDDS_WritePage(csbend->SDDSpart))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (!inhibitFileSync)
            SDDS_DoFSync(csbend->SDDSpart);
        }
      }

      if (tContext.sliceAnalysis && tContext.sliceAnalysis->active &&
          kick != (csbend->nSlices - 1) &&
          (csbend->sliceAnalysisInterval == 0 ||
           kick % csbend->sliceAnalysisInterval == 0)) {
#if (!USE_MPI)
        double **sliceAnalysisPart = part;
#  if defined(HAVE_GPU)
        if (useGpuCsrResident)
          sliceAnalysisPart =
            copyParticlesToCpuReadOnly("CSRCSBEND resident slice analysis output");
#  endif
        convertFromCSBendCoords(sliceAnalysisPart, n_part, rho0, cos_ttilt, sin_ttilt, 1);
        performSliceAnalysisOutput(tContext.sliceAnalysis, sliceAnalysisPart, n_part,
                                   0, tContext.step, Po,
                                   macroParticleCharge * n_part,
                                   tContext.elementName,
                                   z_start + (kick * (z_end - z_start)) / (csbend->nSlices - 1),
                                   1);
        convertToCSBendCoords(sliceAnalysisPart, n_part, rho0, cos_ttilt, sin_ttilt, 1);
#else
        if (isMaster)
          printf("Pelegant does not support slice analysis output inside an element now.");

#endif
      }

      if (csbend->wakeFileActive &&
          ((!csbend->outputLastWakeOnly && kick % csbend->outputInterval == 0) ||
           (csbend->outputLastWakeOnly && kick == (csbend->nSlices - 1)))) {
        /* scale the linear density and its derivative to get C/s and C/s^2 
         * ctHist is already normalized to dct, but ctHistDeriv requires an additional factor
         */
        for (iBin = 0; iBin < nBins; iBin++) {
          ctHist[iBin] *= macroParticleCharge * c_mks;
          ctHistDeriv[iBin] *= macroParticleCharge * sqr(c_mks) / dct;
        }

        if (isMaster) {
          if (!SDDS_StartPage(csbend->SDDSout, nBins) ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, dGamma, nBins, "DeltaGamma") ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, T1, nBins, "DeltaGammaT1") ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, T2, nBins, "DeltaGammaT2") ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, ctHist, nBins, "LinearDensity") ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, ctHistDeriv, nBins, "LinearDensityDeriv") ||
              !SDDS_SetParameters(csbend->SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "Pass", -1, "Kick", kick, "dsKick", csbend->length / csbend->nSlices,
                                  "pCentral", Po, "Angle", phiBend, "SlippageLength", slippageLength,
                                  "TotalBunchLength", ctUpper - ctLower,
                                  "BinSize", dct,
                                  "DerbenevRatio", derbenevRatio, NULL))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        if (csbend->binOnce) {
          /* fix these arrays so they can be used again */
          ctHist[iBin] /= macroParticleCharge * c_mks;
          ctHistDeriv[iBin] /= macroParticleCharge * sqr(c_mks) / dct;
        }
        /* use T1 array to output s and T2 to output dGamma/ds */
        for (iBin = 0; iBin < nBins; iBin++) {
          T1[iBin] = ctLower - (ctLower + ctUpper) / 2.0 + dct * (iBin + 0.5);
          T2[iBin] = dGamma[iBin] / (csbend->length / csbend->nSlices);
        }
        if (isMaster) {
          if (!SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, T1, nBins, "s") ||
              !SDDS_SetColumn(csbend->SDDSout, SDDS_SET_BY_NAME, T2, nBins, "GammaDeriv") ||
              !SDDS_WritePage(csbend->SDDSout))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (!inhibitFileSync)
            SDDS_DoFSync(csbend->SDDSout);
        }
      }
    }
  }

#if defined(HAVE_GPU) && !USE_MPI
  skipGpuCsrDriftPrep =
    usedGpuCsrResident && !csrDriftCsbendFinalPrepNeededFollows(eptr);
  if (useGpuCsrResident) {
    if (csrResidentSimpleFinalTransformAvailable(
          csbend, e2, psi2, cos_ttilt, sin_ttilt,
          dxf, dyf, dzf, dcoord_etilt))
      gpuCsrResidentFinalDone =
        gpu_track_csbend_csr_finalize_simple(
          n_part, Po, cos_ttilt,
          (csbend->edgeFlags & BEND_EDGE2_EFFECTS) &&
            csbend->edge_effects[csbend->e2Index] == 1,
          e2, psi2, rho_actual);
#  if defined(GPU_VERIFY)
    forceParticlesToCpu("CSRCSBEND resident final CPU handoff");
#  else
    if (!gpuCsrResidentFinalDone || !skipGpuCsrDriftPrep)
      forceParticlesToCpu("CSRCSBEND resident final CPU handoff");
#  endif
    useGpuCsrResident = 0;
  }
#endif

  if (!csbend->binOnce && n_partMoreThanOne && !csrInhibit && !csbend->csrBlock
#if defined(HAVE_GPU) && !USE_MPI
      && !skipGpuCsrDriftPrep
#endif
      ) {
    /* prepare some data for use by CSRDRIFT element */
#ifdef DEBUG
    fprintf(stderr, "Preparing data for CSRDRIFT, in case one follows\n");
#endif
    csrWake.dctBin = dct;
    ctLower = ctUpper = dct = 0;

    nBinned = binParticleCoordinate(&ctHist, &maxBins,
                                    &ctLower, &ctUpper, &dct, &nBins,
                                    csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor,
                                    part, n_part, 4);
#if (!USE_MPI)
    if (nBinned != n_part) {
      printf("Only %ld of %ld particles binned for CSRCSBEND (z0=%le, end, BRF=%le)\n",
             nBinned, n_part, z_start, csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor);
      printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
             ctLower, ctUpper, dct, nBins, maxBins);
      fflush(stdout);
    }
#else
    if (USE_MPI && distributedBeam) {
      long all_binned, result = 1, nBinned_total;

      if (isSlave || !distributedBeam) {
        result = ((nBinned == n_part) ? 1 : 0);
      } else
        nBinned = 0;
      MPI_Allreduce(&result, &all_binned, 1, MPI_LONG, MPI_LAND, MPI_COMM_WORLD);
      MPI_Allreduce(&nBinned, &nBinned_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      nBinned = nBinned_total;
      if (!all_binned && isMaster) {
        printf("Not all particles binned for CSRCSBEND (z0=%le, kick=%ld, BRF=%le)\n",
               z_start, kick,
               csbend->binRangeFactor < 1.1 ? 1.1 : csbend->binRangeFactor);
        printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
               ctLower, ctUpper, dct, nBins, maxBins);
        fflush(stdout);
      }
      if (distributedBeam) { /* Master needs to know the information to write the result */
        buffer = malloc(sizeof(double) * nBins);
        MPI_Allreduce(ctHist, buffer, nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        memcpy(ctHist, buffer, sizeof(double) * nBins);
        free(buffer);
      }
    }
#endif
    csrWake.s0 = ctLower + dzf;
  } else {
    ctLower = ctUpper = dct = 0;
    csrWake.dctBin = dct;
    csrWake.s0 = ctLower + dzf;
  }

  i_top = n_part - 1;
  if ((isSlave || !distributedBeam)
#if defined(HAVE_GPU) && !USE_MPI
      && !gpuCsrResidentFinalDone
#endif
      ) {
    /* handle edge effects, and transform coordinates */
    for (i_part = 0; i_part <= i_top; i_part++) {
      coord = part[i_part];
      if (csbend->edgeFlags & BEND_EDGE2_EFFECTS && e2 != 0 && rad_coef) {
        /* post-adjust dp/p to correct error made by integrating over entire sector */
        computeCSBENDFields(&Fx, &Fy, X, Y);

        dp_prime = -rad_coef * (sqr(Fx) + sqr(Fy)) * sqr(1 + DP) *
                   sqrt(sqr(1 + X / rho0) + sqr(XP) + sqr(YP));
        DP -= dp_prime * X * tan(e2);
      }

      /* convert CT to distance traveled at final velocity */
      p1 = Po * (1 + DP);
      beta1 = p1 / sqrt(sqr(p1) + 1);
      coord[4] = CT * beta1;

      if (p1 <= 0) {
        if (!part[i_top]) {
          printf("error: couldn't swap particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
                 i_part, i_top);
          fflush(stdout);
          abort();
        }
        swapParticles(part[i_part], part[i_top]);
        if (accepted) {
          if (!accepted[i_top]) {
            printf(
              "error: couldn't swap acceptance data for particles %ld and %ld--latter is null pointer (track_through_csbend)\n",
              i_part, i_top);
            fflush(stdout);
            abort();
          }
          swapParticles(accepted[i_part], accepted[i_top]);
        }
        part[i_top][4] = z_start + dz_lost;
        part[i_top][5] = Po * (1 + part[i_top][5]);
        i_top--;
        i_part--;
        continue;
      }

      if (csbend->edgeFlags & BEND_EDGE2_EFFECTS) {
        if (csbend->useMatrix)
          track_particles(&coord, Me2, &coord, 1);
        else {
          /* apply edge focusing */
          rho = (1 + DP) * rho_actual;
          if (csbend->edge_order <= 1 && csbend->edge_effects[csbend->e2Index] == 1) {
            delta_xp = tan(e2) / rho * X;
            XP += delta_xp;
            YP -= tan(e2 - psi2 / (1 + DP)) / rho * Y;
          } else if (csbend->edge_order >= 2 && csbend->edge_effects[csbend->e2Index] == 1)
            apply_edge_effects(&X, &XP, &Y, &YP, rho, n, e2, he2, psi2 * (1 + DP), 1);
          else if (csbend->edge_effects[csbend->e2Index] == 2) {
            rho = (1 + DP) * rho_actual;
            /* load input coordinates into arrays */
            Qi[0] = X;
            Qi[1] = XP;
            Qi[2] = Y;
            Qi[3] = YP;
            Qi[4] = 0;
            Qi[5] = DP;
            convertToDipoleCanonicalCoordinates(Qi, 0);
            dipoleFringeKHwang(Qf, Qi, rho_actual, 1., csbend->edge_order, csbend->b[1] / rho0, e2, 2 * csbend->hgap, csbend->fint, csbend->h[csbend->e2Index]);
            /* retrieve coordinates from arrays */
            convertFromDipoleCanonicalCoordinates(Qf, 0);
            X = Qf[0];
            XP = Qf[1];
            Y = Qf[2];
            YP = Qf[3];
            DP = Qf[5];
          } else if (csbend->edge_effects[csbend->e2Index] == 3) {
            applySimpleDipoleEdgeKick(&XP, &YP, X, Y, DP, rho_actual, e2, psi2, -1.0, 0);
          }
        }
      }

      coord = part[i_part];
      x = X * cos_ttilt - Y * sin_ttilt + dcoord_etilt[0];
      y = X * sin_ttilt + Y * cos_ttilt + dcoord_etilt[2];
      xp = XP * cos_ttilt - YP * sin_ttilt + dcoord_etilt[1];
      yp = XP * sin_ttilt + YP * cos_ttilt + dcoord_etilt[3];
      X = x;
      Y = y;
      XP = xp;
      YP = yp;
      coord[0] += dxf + dzf * coord[1];
      coord[2] += dyf + dzf * coord[3];
      coord[4] += dzf * sqrt(1 + sqr(coord[1]) + sqr(coord[3])) + dcoord_etilt[4];
    }
    n_part = i_top + 1;
  }

  if (n_partMoreThanOne && !csbend->csrBlock
#if defined(HAVE_GPU) && !USE_MPI
      && !skipGpuCsrDriftPrep
#endif
      ) {
    /* prepare more data for CSRDRIFT */
    int64_t imin, imax;
    double S55;

#if !USE_MPI
    rms_emittance(part, 0, 1, i_top + 1, &csrWake.S11, &csrWake.S12, &csrWake.S22, NULL, NULL);
    rms_emittance(part, 4, 5, i_top + 1, &S55, NULL, NULL, NULL, NULL);
#else
    if (distributedBeam) {
      rms_emittance_p(part, 0, 1, i_top + 1, &csrWake.S11, &csrWake.S12, &csrWake.S22, NULL, NULL, NULL);
      rms_emittance_p(part, 4, 5, i_top + 1, &S55, NULL, NULL, NULL, NULL, NULL);
    } else {
      rms_emittance(part, 0, 1, i_top + 1, &csrWake.S11, &csrWake.S12, &csrWake.S22, NULL, NULL);
      rms_emittance(part, 4, 5, i_top + 1, &S55, NULL, NULL, NULL, NULL);
    }
#endif

    csrWake.perc68BunchLength = approximateBeamWidth(0.6826, part, i_top + 1, 4) / 2;
    csrWake.perc90BunchLength = approximateBeamWidth(0.9, part, i_top + 1, 4) / 2;

    csrWake.rmsBunchLength = sqrt(S55);

#ifdef DEBUG
    fprintf(stderr, "rms bunch length = %le, percentile bunch length (68, 90) = %le, %le\n",
            csrWake.rmsBunchLength, csrWake.perc68BunchLength, csrWake.perc90BunchLength);
#endif
#if defined(HAVE_GPU) && !USE_MPI
    if (!csrDgammaHostCurrent) {
      if (!gpu_copy_csbend_csr_dgamma(dGamma, nBins))
        bombElegant("unable to copy CUDA CSR dGamma for CSR_LAST_WAKE state", NULL);
      csrDgammaHostCurrent = 1;
    }
#endif
    if (macroParticleCharge) {
      index_min_max(&imin, &imax, csrWake.dGamma, csrWake.bins);
      csrWake.peakToPeakWavelength = 2 * fabs(1.0 * imax - imin) * dct;
    } else {
      csrWake.peakToPeakWavelength = csrWake.perc68BunchLength;
    }

    csrWake.valid = 1;
    csrWake.rho = rho_actual;
    csrWake.bendingAngle = accumulatingAngle ? fabs(phiBend) : fabs(angle);
    csrWake.Po = Po;
    csrWake.SGOrder = csbend->SGOrder;
    csrWake.SGDerivOrder = csbend->SGDerivOrder;
    csrWake.SGHalfWidth = csbend->SGHalfWidth;
    csrWake.SGDerivHalfWidth = csbend->SGDerivHalfWidth;
    csrWake.GSConstant = CSRConstant * pow(3 * rho0 * rho0, 1. / 3.) / 2; /* used for G. Stupakov's drift formulae */
    csrWake.MPCharge = macroParticleCharge;
    csrWake.binRangeFactor = csbend->binRangeFactor;
    csrWake.trapazoidIntegration = csbend->trapazoidIntegration;
    if (csbend->useMatrix) {
      free_matrices(Msection);
      free_matrices(Me1);
      free_matrices(Me2);
      free(Msection);
      free(Me1);
      free(Me2);
      Msection = Me1 = Me2 = NULL;
    }
  }
#if defined(HAVE_GPU) && !USE_MPI
  else if (n_partMoreThanOne && !csbend->csrBlock &&
           skipGpuCsrDriftPrep &&
           csrDriftStupakovOnlyStateConsumerFollows(eptr)) {
    csrWake.valid = 1;
    csrWake.rho = rho_actual;
    csrWake.bendingAngle = accumulatingAngle ? fabs(phiBend) : fabs(angle);
    csrWake.Po = Po;
    csrWake.SGOrder = csbend->SGOrder;
    csrWake.SGDerivOrder = csbend->SGDerivOrder;
    csrWake.SGHalfWidth = csbend->SGHalfWidth;
    csrWake.SGDerivHalfWidth = csbend->SGDerivHalfWidth;
    csrWake.GSConstant = CSRConstant * pow(3 * rho0 * rho0, 1. / 3.) / 2;
    csrWake.MPCharge = macroParticleCharge;
    csrWake.binRangeFactor = csbend->binRangeFactor;
    csrWake.trapazoidIntegration = csbend->trapazoidIntegration;
  }
#endif

  if (csbend->csrBlock)
    accumulatedAngle = 0;
  else
    /* accumulate the bending angle just in case the same type of dipole follows */
    accumulatedAngle += fabs(angle);

  /* prepare some data for CSRDRIFT */
  csrWake.dGamma = dGamma;
  csrWake.bins = nBins;
  csrWake.ds0 = csbend->length / csbend->nSlices;
  csrWake.zLast = csrWake.z0 = z_end;
  csrWake.highFrequencyCutoff0 = csbend->highFrequencyCutoff0;
  csrWake.highFrequencyCutoff1 = csbend->highFrequencyCutoff1;
  csrWake.lowFrequencyCutoff0 = csbend->lowFrequencyCutoff0;
  csrWake.lowFrequencyCutoff1 = csbend->lowFrequencyCutoff1;
  csrWake.clipNegativeBins = csbend->clipNegativeBins;
  csrWake.wffValues = csbend->wffValues;
  csrWake.wffFreqValue = csbend->wffFreqValue;
  csrWake.wffRealFactor = csbend->wffRealFactor;
  csrWake.wffImagFactor = csbend->wffImagFactor;

#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  if (csrWake.valid && csrWake.dGamma && csrWake.bins == nBins &&
      !csbend->integratedGreensFunction) {
    double *cpuDGamma = malloc((size_t)nBins * sizeof(*cpuDGamma));
    if (!cpuDGamma)
      bombElegant("memory allocation failure copying CSR_LAST_WAKE CPU shadow", NULL);
    if (gpu_copy_csr_wake_cpu_shadow(cpuDGamma, nBins)) {
      CSR_LAST_WAKE cpuCsrWake;
      if (csbend->wffValues)
        applyFilterTable(cpuDGamma, nBins, dct / c_mks,
                         csbend->wffValues, csbend->wffFreqValue,
                         csbend->wffRealFactor, csbend->wffImagFactor);
      memcpy(&cpuCsrWake, &csrWake, sizeof(cpuCsrWake));
      cpuCsrWake.dGamma = cpuDGamma;
      compareCSR_LAST_WAKE(&csrWake, &cpuCsrWake);
    }
    free(cpuDGamma);
  }
#endif

#if defined(MINIMIZE_MEMORY)
  /* leave dGamma out of this because that memory is used by CSRDRIFT */
  free(beta0);
  free(ctHist);
  free(ctHistDeriv);
  free(T1);
  free(T2);
  free(denom);
  if (grnk)
    free(grnk);
  if (chik)
    free(chik);
  beta0 = ctHist = ctHistDeriv = T1 = T2 = denom = NULL;
  maxBins = maxParticles = 0;
#  if defined(HAVE_GPU)
  denomCachePointer = NULL;
  denomCacheBins = 0;
  denomCacheDct = 0;
#  endif
#endif

#if (!USE_MPI)
  return (i_top + 1);
#else
  if (isSlave || !distributedBeam)
    return (i_top + 1);
  else
    return n_part; /* i_top is not defined for master */
#endif
}
#undef DEBUG_IGF

static long prepareParticleCoordinateHistogram(double **hist, long *maxBins,
                                               double *lower, double *upper,
                                               double *binSize, long *bins,
                                               double expansionFactor,
                                               double **particleCoord,
                                               long nParticles,
                                               long coordinateIndex) {
  long iBin, iParticle, count;
  double value;
#ifdef DEBUG
  fprintf(stderr, "binParticleCoordinate: maxBins = %ld, bins = %ld, binSize = %le\n",
	  *maxBins, *bins, *binSize);
#endif
  
  if (*binSize <= 0 && *bins < 1)
    return -1;
  if (*binSize > 0 && *bins > 1)
    return -2;

  /* if (*lower==*upper)  This condition will be removed */
  count = 0;
  if (isSlave || !distributedBeam) {
    /* find range of points */
    *upper = -(*lower = DBL_MAX);
    for (iParticle = 0; iParticle < nParticles; iParticle++) {
      if (isinf(value = particleCoord[iParticle][coordinateIndex]) ||
	  isnan(value))
	continue;
      count++;
      if (value < *lower)
        *lower = value;
      if (value > *upper)
        *upper = value;
    }
  }
#ifdef DEBUG
  fprintf(stderr, "count = %ld, lower = %le, upper = %le\n", count, *lower, *upper);
#endif
  
#if USE_MPI
  /* find the global maximum and minimum */
  if (distributedBeam) {
    if (isMaster)
      nParticles = 0;
    find_global_min_max(lower, upper, nParticles, MPI_COMM_WORLD);
  }
#endif

  if (*lower==DBL_MAX || *upper==-DBL_MAX) {
    *lower = -sqrt(DBL_MAX);
    *upper = sqrt(DBL_MAX);
    *binSize = 0;
    *bins = 10;
    return 0;
  } else if (expansionFactor > 1) {
    double center, range;
    center = (*lower + *upper) / 2;
    range = (*upper - *lower) * expansionFactor;
    *lower = center - range / 2;
    *upper = center + range / 2;
  }

  if (*binSize > 0)
    /* bin size given, so determine the number of bins */
    *bins = (*upper - *lower) / (*binSize);
  *binSize = (*upper - *lower) / (*bins);

  /* realloc if necessary */
  if (*bins > *maxBins &&
      !(*hist = SDDS_Realloc(*hist, sizeof(**hist) * (*maxBins = *bins))))
    bombElegant("Memory allocation failure (binParticleCoordinate)", NULL);

#ifdef DEBUG
  fprintf(stderr, "range: [%le, %le] (%le) : bins => %ld, binSize => %le; maxBins = %ld\n",
	  *lower, *upper, *upper-*lower, *bins, *binSize, *maxBins);
#endif

  for (iBin = 0; iBin < *bins; iBin++)
    (*hist)[iBin] = 0;
  return count;
}

long binParticleCoordinate(double **hist, long *maxBins,
                           double *lower, double *upper, double *binSize, long *bins,
                           double expansionFactor,
                           double **particleCoord, long nParticles, long coordinateIndex) {
  long iBin, iParticle, nBinned, status;
  if ((status = prepareParticleCoordinateHistogram(hist, maxBins, lower, upper,
                                                  binSize, bins, expansionFactor,
                                                  particleCoord, nParticles,
                                                  coordinateIndex)) < 0)
    return status;
  nBinned = 0;
  if (isSlave || !distributedBeam) {
    for (iParticle = nBinned = 0; iParticle < nParticles; iParticle++) {
      /* the coordinate of the bin center is (iBin+0.5)*(*binSize) + *lower */
      iBin = (particleCoord[iParticle][coordinateIndex] - *lower) / (*binSize);
      if (iBin < 0 || iBin > (*bins - 1))
        continue;
      (*hist)[iBin] += 1;
      nBinned++;
    }
  }
  return nBinned;
}

#if USE_MPI
long binParticleCoordinate_s(double **hist, long *maxBins,
                             double *lower, double *upper, double *binSize, long *bins,
                             double expansionFactor,
                             double **particleCoord, long nParticles, long coordinateIndex) {
  long iBin, iParticle, nBinned;
  double value;

  if (*binSize <= 0 && *bins < 1)
    return -1;
  if (*binSize > 0 && *bins > 1)
    return -2;

  /* if (*lower==*upper)  This condition will be removed */
  /* find range of points */
  *upper = -(*lower = DBL_MAX);
  for (iParticle = 0; iParticle < nParticles; iParticle++) {
    value = particleCoord[iParticle][coordinateIndex];
    if (value < *lower)
      *lower = value;
    if (value > *upper)
      *upper = value;
  }
  if (expansionFactor > 1) {
    double center, range;
    center = (*lower + *upper) / 2;
    range = (*upper - *lower) * expansionFactor;
    *lower = center - range / 2;
    *upper = center + range / 2;
  }

  if (*binSize > 0)
    /* bin size given, so determine the number of bins */
    *bins = (*upper - *lower) / (*binSize);
  *binSize = (*upper - *lower) / (*bins);

  /* realloc if necessary */
  if (*bins > *maxBins &&
      !(*hist = SDDS_Realloc(*hist, sizeof(**hist) * (*maxBins = *bins))))
    bombElegant("Memory allocation failure (binParticleCoordinate)", NULL);

  for (iBin = 0; iBin < *bins; iBin++)
    (*hist)[iBin] = 0;
  nBinned = 0;
  for (iParticle = nBinned = 0; iParticle < nParticles; iParticle++) {
    /* the coordinate of the bin center is (iBin+0.5)*(*binSize) + *lower */
    iBin = (particleCoord[iParticle][coordinateIndex] - *lower) / (*binSize);
    if (iBin < 0 || iBin > (*bins - 1))
      continue;
    (*hist)[iBin] += 1;
    nBinned++;
  }
  return nBinned;
}
#endif

void computeSaldinFdNorm(double **FdNorm, double **x, long *n, double sMax, long ns,
                         double Po, double radius, double angle, double dx, char *normMode);
long track_through_driftCSR_Stupakov(double **part, int64_t np, CSRDRIFT *csrDrift,
                                     double Po, double **accepted, double zStart, CHARGE *charge, char *rootname);

long track_through_driftCSR(double **part, int64_t np, CSRDRIFT *csrDrift,
                            double Po, double **accepted, double zStart,
                            double revolutionLength, CHARGE *charge, char *rootname) {
  int64_t iPart;
  long iKick, iBin, binned = 0, nKicks, iSpreadMode = 0;
  double *coord, p, beta, dz, ct0 = 0.0, factor, dz0, dzFirst;
  double ctmin, ctmax, spreadFactor, dct;
  double zTravel, attenuationLength, thetaRad = 0.0, sigmaZ, overtakingLength, criticalWavelength, wavelength = 0.0;
  static char *spreadMode[3] = {"full", "simple", "radiation-only"};
  static char *wavelengthMode[3] = {"sigmaz", "bunchlength", "peak-to-peak"};
  static char *bunchlengthMode[3] = {"rms", "68-percentile", "90-percentile"};
  unsigned long mode;
  long nBins1;
  TRACKING_CONTEXT tContext;
#if USE_MPI
  long np_total = 1, np_tmp = np, binned_total;
#endif

  if (csrDrift->LSCBins && !csrDrift->useStupakov)
    bombElegant("LSCBINS is nonzero on CSRDRIFT but USE_STUPAKOV is zero. This is not supported.", NULL);

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
#  ifdef GPU_VERIFY
    CSR_LAST_WAKE initCsrWake;
    memcpy(&initCsrWake, &csrWake, sizeof(CSR_LAST_WAKE));
#  endif
    startGpuTimer();
    iPart = gpu_track_through_driftCSR(np, csrDrift, Po, accepted, zStart, revolutionLength, charge, rootname);
#  ifdef GPU_VERIFY
    startCpuTimer();
    memcpy(&csrWake, &initCsrWake, sizeof(CSR_LAST_WAKE));
    track_through_driftCSR(part, np, csrDrift, Po, accepted, zStart, revolutionLength, charge, rootname);
    compareGpuCpu(np, "track_through_driftCSR");
#  endif /* GPU_VERIFY */
    return iPart;
  }
#endif /* HAVE_GPU */

  getTrackingContext(&tContext);

#if (!USE_MPI)
  if (np <= 1 || !csrWake.valid || !(csrDrift->csr || csrDrift->LSCBins)) {
#else
  if (distributedBeam) {
    if (isMaster)
      np_tmp = 0;
    MPI_Allreduce(&np_tmp, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  } else
    np_total = np;

  if (np_total <= 1 || !csrWake.valid || !(csrDrift->csr || csrDrift->LSCBins)) {
    if (isSlave || !distributedBeam) {
#endif
    if (csrDrift->linearOptics) {
      int64_t i;
      for (i = 0; i < np; i++) {
        part[i][0] += csrDrift->length * part[i][1];
        part[i][2] += csrDrift->length * part[i][3];
        part[i][4] += csrDrift->length;
      }
    } else
      exactDrift(part, np, csrDrift->length);
#if (USE_MPI)
  }
#endif
  return np;
}
nBins1 = csrWake.bins - 1;

mode =
  (csrDrift->spread ? CSRDRIFT_SPREAD : 0) +
  (csrDrift->useOvertakingLength ? CSRDRIFT_OVERTAKINGLENGTH : 0) +
  (csrDrift->useSaldin54 ? CSRDRIFT_SALDIN54 : 0) +
  (csrDrift->attenuationLength > 0 ? CSRDRIFT_ATTENUATIONLENGTH : 0) +
  (csrDrift->useStupakov ? CSRDRIFT_STUPAKOV : 0);
while ((zStart + 1e-12) < csrWake.zLast) {
  printWarningForTracking("Incrementing zStart by revolution length for CSRDRIFT",
                          "If you are not simulating a ring, this could be a problem!");
  zStart += revolutionLength;
}
if (bitsSet(mode) > 1) {
  printf("Error: Too many modes set for CSRDRIFT.\n");
  exitElegant(1);
}
if (csrWake.lastMode && csrWake.lastMode != mode) {
  printf("Error: CSRDRIFT mode changed between dipoles. Pick one mode following each dipole.\n");
  exitElegant(1);
}
csrWake.lastMode = mode;

if (mode & CSRDRIFT_STUPAKOV)
  return track_through_driftCSR_Stupakov(part, np, csrDrift, Po, accepted, zStart, charge, rootname);

printWarningForTracking("USE_STUPAKOV=1 is recommended for CSRDRIFT elements.",
                        "This is the most physical model available in elegant.");

dct = csrWake.dctBin;
if (csrDrift->dz > 0) {
  if ((nKicks = csrDrift->length / csrDrift->dz) < 1)
    nKicks = 1;
} else
  nKicks = csrDrift->nKicks;
if (nKicks <= 0)
  bombElegant("nKicks=0 in CSR drift.", NULL);
dz = (dz0 = csrDrift->length / nKicks) / 2;

sigmaZ = 0;
switch (match_string(csrDrift->bunchlengthMode, bunchlengthMode, 3, 0)) {
case 0:
  sigmaZ = csrWake.rmsBunchLength;
  break;
case 1:
  sigmaZ = csrWake.perc68BunchLength;
  break;
case 2:
  sigmaZ = csrWake.perc90BunchLength;
  break;
default:
  bombElegant("invalid bunchlength_mode for CSRDRIFT.  Use rms or percentile.", NULL);
}

overtakingLength = pow(24 * sigmaZ * csrWake.rho * csrWake.rho, 1. / 3.);

if (mode & CSRDRIFT_OVERTAKINGLENGTH)
  attenuationLength = overtakingLength * csrDrift->overtakingLengthMultiplier;
else
  attenuationLength = csrDrift->attenuationLength;

if (mode & CSRDRIFT_SPREAD) {
  iSpreadMode = 0;
  if (csrDrift->spreadMode &&
      (iSpreadMode = match_string(csrDrift->spreadMode, spreadMode, 3, 0)) < 0)
    bombElegant("invalid spread_mode for CSR DRIFT.  Use full, simple, or radiation-only", NULL);
  switch (match_string(csrDrift->wavelengthMode, wavelengthMode, 3, 0)) {
  case 0:
  case 1:
    /* bunch length */
    wavelength = sigmaZ;
    break;
  case 2:
    /* peak-to-peak */
    wavelength = csrWake.peakToPeakWavelength;
    break;
  default:
    bombElegant("invalid wavelength_mode for CSR DRIFT.  Use sigmaz or peak-to-peak", NULL);
    break;
  }
  criticalWavelength = 4.19 / ipow3(csrWake.Po) * csrWake.rho;
  if (!particleIsElectron)
    bombElegant("CSRDRIFT spread mode is not supported for particles other than electrons", NULL);
  thetaRad = 0.5463e-3 / (csrWake.Po * 0.511e-3) / pow(criticalWavelength / wavelength, 1. / 3.);
}

if (mode & CSRDRIFT_SALDIN54) {
  if (csrWake.FdNorm == NULL) {
    if (csrDrift->nSaldin54Points < 20)
      csrDrift->nSaldin54Points = 20;
    computeSaldinFdNorm(&csrWake.FdNorm, &csrWake.xSaldin, &csrWake.nSaldin,
                        2 * sigmaZ, csrDrift->nSaldin54Points, csrWake.Po, csrWake.rho, csrWake.bendingAngle, dz,
                        csrDrift->normMode);
    if (csrDrift->Saldin54Output) {
      long ix;
      if (!csrDrift->fpSaldin) {
	char *filename;
        filename = compose_filename(csrDrift->Saldin54Output, rootname);
        csrDrift->fpSaldin = fopen(filename, "w");
	free(filename);
        fprintf(csrDrift->fpSaldin, "SDDS1\n&column name=z, type=double &end\n&column name=Factor, type=double &end\n");
        fprintf(csrDrift->fpSaldin, "&data mode=ascii no_row_counts=1 &end\n");
      } else
        fprintf(csrDrift->fpSaldin, "\n");
      for (ix = 0; ix < csrWake.nSaldin; ix++)
        fprintf(csrDrift->fpSaldin, "%le %le\n", csrWake.xSaldin[ix], csrWake.FdNorm[ix]);
      fflush(csrDrift->fpSaldin);
    }
  }
}

dzFirst = zStart - csrWake.zLast;
zTravel = zStart - csrWake.z0; /* total distance traveled by radiation to reach this point */
#ifdef DEBUG
printf("CSR in drift:\n");
printf("zStart = %21.15le, zLast = %21.15le, zTravel = %21.15le\n", zStart, csrWake.zLast,
       zTravel);
printf("dzFirst = %21.15e, s0 = %21.15e\n", dzFirst, csrWake.s0);
#endif

for (iKick = 0; iKick < nKicks; iKick++) {
  /* first drift is dz=dz0/2, others are dz0 */
  if (iKick == 1)
    dz = dz0;
  zTravel += dz;

  ctmin = DBL_MAX;
  ctmax = -DBL_MAX;

  /* propagate particles forward, converting s to c*t=s/beta */
  if (isSlave || !distributedBeam) {
    for (iPart = 0; iPart < np; iPart++) {
      coord = part[iPart];
      coord[0] += coord[1] * dz;
      coord[2] += coord[3] * dz;
      p = Po * (1 + coord[5]);
      beta = p / sqrt(p * p + 1);
      if (csrDrift->linearOptics)
        coord[4] = (coord[4] + dz) / beta;
      else
        coord[4] = (coord[4] + dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]))) / beta;
#ifdef DEBUG
      if (coord[4] > ctmax)
        ctmax = coord[4];
      if (coord[4] < ctmin)
        ctmin = coord[4];
#endif
    }
  }

  factor = 1;
  if (csrWake.dGamma) {
    /* propagate wake forward */
    csrWake.s0 += dz + dzFirst; /* accumulates position of back end of the radiation pulse */
    ct0 = csrWake.s0;

    if (attenuationLength > 0) {
      /* attenuate wake */
      if ((factor = exp(-(dz + dzFirst) / attenuationLength)) < 1) {
        for (iBin = 0; iBin < csrWake.bins; iBin++)
          csrWake.dGamma[iBin] *= factor;
      }
    }
    /* factor to account for difference in drift lengths here and in
       * csrcsbend integration.  Use dz0 here because that is the
       * length integrated by each kick.  Add dzFirst to account for any
       * length we may have missed due to intervening non-drift elements.
       */
    factor = (dz0 + dzFirst) / csrWake.ds0;
  }
  if (mode & CSRDRIFT_SPREAD) {
    /* compute loss of on-axis field due to spread of beam using a simple-minded
       * computation of beam sizes */
    switch (iSpreadMode) {
    case 0: /* full */
      factor *= (spreadFactor =
                   sqrt(csrWake.S11 / (csrWake.S11 +
                                       2 * zTravel * csrWake.S12 +
                                       zTravel * zTravel * (sqr(thetaRad) + csrWake.S22))));
      break;
    case 1: /* simple */
      factor *= (spreadFactor =
                   sqrt(csrWake.S11 / (csrWake.S11 + zTravel * zTravel * (sqr(thetaRad) + csrWake.S22))));
      break;
    case 2: /* radiation only */
      factor *= (spreadFactor =
                   sqrt(csrWake.S11 / (csrWake.S11 + sqr(zTravel * thetaRad))));
      break;
    default:
      bombElegant("invalid spread code---programming error!", NULL);
      break;
    }
  }

  if (mode & CSRDRIFT_SALDIN54) {
    long code = 0;
    double f0 = 0;
    if (zTravel <= csrWake.xSaldin[csrWake.nSaldin - 1])
      factor *= (f0 = interp(csrWake.FdNorm, csrWake.xSaldin, csrWake.nSaldin, zTravel, 0, 1, &code));
    else
      factor = 0;
    csrWake.lastFdNorm = f0;
#ifdef DEBUG
    fprintf(csrWake.fpSaldin, "%le %le\n", zTravel, f0);
    fflush(csrWake.fpSaldin);
#endif
    if (!code) {
      char warningText[1024];
      snprintf(warningText, 1024, "zTravel = %le,  csrWake available up to %le\n",
               zTravel, csrWake.xSaldin[csrWake.nSaldin - 1]);
      printWarningForTracking("Interpolation failure for Saldin eq. 54.", warningText);
      factor = 0;
    }
  }

  dzFirst = 0;

  /* apply kick to each particle and convert back to normal coordinates */
  if (isSlave || !distributedBeam) {
    for (iPart = binned = 0; iPart < np; iPart++) {
      coord = part[iPart];
      if (csrWake.dGamma) {
        double f;
        iBin = (f = (coord[4] - ct0) / dct);
        f -= iBin;
        if (iBin >= 0 && iBin < nBins1) {
          coord[5] += ((1 - f) * csrWake.dGamma[iBin] + f * csrWake.dGamma[iBin + 1]) / Po * factor;
          binned++;
        }
      }
      p = (1 + coord[5]) * Po;
      beta = p / sqrt(p * p + 1);
      coord[4] = beta * coord[4];
    }
  }
#if USE_MPI
  if (isSlave && distributedBeam) {
    MPI_Allreduce(&binned, &binned_total, 1, MPI_LONG, MPI_SUM, workers);
  }
  if ((myid == 1) && (csrWake.dGamma && np_total != binned_total)) {
    dup2(fdStdout, fileno(stdout)); /* Let the first slave processor write the output */
    printf("only %ld of %ld particles binned for CSR drift %s (track_through_driftCSR)\n",
           binned_total, np_total, tContext.elementName);
#else
      if (csrWake.dGamma && np != binned) {
        printf("only %ld of %ld particles binned for CSR drift %s (track_through_driftCSR)\n",
               binned, np, tContext.elementName);
#endif
    printf("beam ct min, max = %21.15e, %21.15e\n",
           ctmin, ctmax);
    printf("wake ct0 = %21.15e, ct1 = %21.15e\n",
           ct0, ct0 + csrWake.dctBin * csrWake.bins);
    fflush(stdout);
#if USE_MPI
#  if defined(_WIN32)
    freopen("NUL", "w", stdout);
#  else
    if (!freopen("/dev/null", "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
#  endif
#endif
  }
}
/* do final drift of dz0/2 */
dz = dz0 / 2;
if (isSlave || !distributedBeam) {
  for (iPart = 0; iPart < np; iPart++) {
    coord = part[iPart];
    coord[0] += coord[1] * dz;
    coord[2] += coord[3] * dz;
    if (csrDrift->linearOptics)
      coord[4] += dz;
    else
      coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  }
}
csrWake.zLast = zStart + csrDrift->length;

if (csrWake.dGamma) {
  /* propagate wake forward */
  csrWake.s0 += dz;
  ct0 = csrWake.s0;

  if (attenuationLength > 0) {
    /* attenuate wake */
    if ((factor = exp(-dz / attenuationLength)) < 1) {
      for (iBin = 0; iBin < csrWake.bins; iBin++)
        csrWake.dGamma[iBin] *= factor;
    }
  }
}

return np;
}

/* this should be called before starting to track a beamline to make sure that
 * CSR drift elements upstream of all CSRBEND elements get treated like ordinary
 * drift spaces. */

long reset_driftCSR() {
  csrWake.lastMode = 0;
  if (csrWake.valid && csrWake.FdNorm) {
    printf("Last value of normalization factor for CSR wake was %le\n",
           csrWake.lastFdNorm);
  }
  csrWake.valid = csrWake.bins = 0;
  csrWake.dctBin = csrWake.s0 = csrWake.ds0 = csrWake.zLast =
    csrWake.z0 = csrWake.S11 = csrWake.S12 = csrWake.S22 = 0;
  csrWake.dGamma = NULL;
  csrWake.nSaldin = 0;
  if (csrWake.FdNorm) {
    free(csrWake.FdNorm);
    free(csrWake.xSaldin);
    csrWake.FdNorm = csrWake.xSaldin = NULL;
  }
  if (csrWake.StupakovFileActive) {
    if (!SDDS_Terminate(&csrWake.SDDS_Stupakov))
      bombElegant("problem terminating data file for Stupakov output from CSRDRIFT", NULL);
    csrWake.StupakovFileActive = 0;
  }
  return 1;
}

double SolveForPsiSaldin54(double xh, double sh);
double Saldin5354Factor(double xh, double sh, double phihm, double xhLowerLimit);

void computeSaldinFdNorm(double **FdNorm, double **x, long *n, double sMax, long ns,
                         double Po, double radius, double bendingAngle, double dx,
                         char *normMode) {
  double xEnd, sh, beta, gamma, xh, dx0;
  long ix, is;
  double phihs, phihm, xhLowerLimit, xUpperLimit, s, f, fx;
  double t1, t2, f0, fmax;
  char *allowedNormMode[2] = {"first", "peak"};

  gamma = sqrt(sqr(Po) + 1);
  beta = Po / gamma;

  if ((xEnd = sMax / (1 - beta)) > 1000 || isnan(xEnd) || isinf(xEnd)) {
    printWarningForTracking("The extent of the CSR drift wake decay was limited at 1km.", NULL);
    xEnd = 1000;
  }

  *n = 100;
  dx0 = xEnd / (100 * (*n));
  if (dx < dx0) {
    *n = xEnd / (100 * dx);
    if (*n > 100000) {
      *n = 100000;
      printWarningForTracking("The CSR drift wake decay table size hit the limit of 100k points.",
                              "Check results with a different CSR model.");
    }
  } else
    dx = dx0;
  fx = pow(xEnd / dx, 1. / (*n));

  if (!(*FdNorm = calloc(sizeof(**FdNorm), (*n))) ||
      !(*x = malloc(sizeof(**x) * (*n))))
    bombElegant("memory allocation failure (computeSaldinFdNorm)", NULL);

  for (ix = 0; ix < *n; ix++)
    (*x)[ix] = ix == 0 ? 0 : ipow(fx, ix - 1) * dx;
  for (is = 0; is < ns; is++) {
    /* don't use s=0 as it is singular */
    s = (is + 1.0) * sMax / ns;
    sh = s * ipow3(gamma) / radius;
    phihm = bendingAngle * gamma;
    t1 = 12 * sh;
    t2 = sqrt(64 + 144 * sh * sh);
    phihs = pow(t1 + t2, 1. / 3.) - pow(-t1 + t2, 1. / 3.);
    xhLowerLimit = -1;
    if (phihs > phihm)
      xhLowerLimit = sh - phihm - ipow3(phihm) / 6 + sqrt(sqr(ipow3(phihm) - 6 * sh) + 9 * ipow4(phihm)) / 6;
    xUpperLimit = 0.999 * s / (1 - beta);
    for (ix = 0; ix < *n; ix++) {
      if ((*x)[ix] >= xUpperLimit)
        break;
      xh = (*x)[ix] * gamma / radius;
      (*FdNorm)[ix] += Saldin5354Factor(xh, sh, phihm, xhLowerLimit);
    }
  }

  /* average over s */
  for (ix = 0; ix < *n; ix++)
    (*FdNorm)[ix] /= ns;

  /* get the first nonzero and also the maximum value of Fd */
  for (ix = f0 = fmax = 0; ix < *n; ix++) {
    f = (*FdNorm)[ix];
    if (f0 == 0 && f > 0)
      f0 = f;
    if (fmax < f)
      fmax = f;
  }
  if (fmax > f0 / 0.99) {
    char warningText[1024];
    snprintf(warningText, 1024, "%ld points, max/start-1 is %le.",
             ns, fmax / f0 - 1);
    printWarningForTracking("Possible problem with SALDIN54 drift mode: too few points.",
                            warningText);
  }
  switch (match_string(normMode, allowedNormMode, 2, 0)) {
  case 0:
    /* first */
    f = f0;
    break;
  case 1:
    /* peak */
    f = fmax;
    break;
  default:
    fprintf(stderr, "Error: unknown Saldin-54 normalization mode: %s\n", normMode);
    f = 0; /* suppress spurious compiler warning */
    exitElegant(1);
    break;
  }
  if (f)
    for (ix = 0; ix < *n; ix++)
      (*FdNorm)[ix] /= f;
  else
    for (ix = 0; ix < *n; ix++)
      (*FdNorm)[ix] = 0;
}

double SolveForPsiSaldin54(double xh, double sh) {
  double s_sum, s_diff2, bestSol;
  double solList[4] = {-1, -1, -1, -1};
  long nSols = 0, sol;

  s_sum = (-2 * xh - sqrt(-8 + 4 * pow(xh, 2) -
                          (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) /
                            pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                                  3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                                0.3333333333333333) +
                          2 * pow(2, 0.6666666666666666) *
                            pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                                  3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                                0.3333333333333333))) /
          2.;
  if (!isnan(s_sum)) {
    s_diff2 = (-16 + 8 * pow(xh, 2) + (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) / pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)), 0.3333333333333333) -
               2 * pow(2, 0.6666666666666666) *
                 pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                       3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                     0.3333333333333333) +
               (16 * (-3 * sh + pow(xh, 3))) /
                 sqrt(-8 + 4 * pow(xh, 2) -
                      (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) /
                        pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                              3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                            0.3333333333333333) +
                      2 * pow(2, 0.6666666666666666) *
                        pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                              3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                            0.3333333333333333))) /
              4.;
    if (s_diff2 >= 0) {
      solList[0] = s_sum + sqrt(s_diff2);
      solList[1] = s_sum + sqrt(s_diff2);
      nSols = 2;
    }
  }

  s_sum = (-2 * xh + sqrt(-8 + 4 * pow(xh, 2) -
                          (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) /
                            pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                                  3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                                0.3333333333333333) +
                          2 * pow(2, 0.6666666666666666) *
                            pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                                  3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                                0.3333333333333333))) /
          2.;
  if (!isnan(s_sum)) {
    s_diff2 = (-16 + 8 * pow(xh, 2) + (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) / pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)), 0.3333333333333333) -
               2 * pow(2, 0.6666666666666666) *
                 pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                       3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                     0.3333333333333333) -
               (16 * (-3 * sh + pow(xh, 3))) /
                 sqrt(-8 + 4 * pow(xh, 2) -
                      (4 * pow(2, 0.3333333333333333) * (-1 + pow(xh, 2))) /
                        pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                              3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                            0.3333333333333333) +
                      2 * pow(2, 0.6666666666666666) *
                        pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) +
                              3 * pow(xh, 4) + sqrt(4 * pow(-1 + pow(xh, 2), 3) + pow(2 + 9 * pow(sh, 2) - 3 * pow(xh, 2) - 6 * sh * pow(xh, 3) + 3 * pow(xh, 4), 2)),
                            0.3333333333333333))) /
              4.;

    if (s_diff2 >= 0) {
      solList[nSols] = s_sum + sqrt(s_diff2);
      solList[nSols + 1] = s_sum - sqrt(s_diff2);
      nSols += 2;
    }
  }
  bestSol = solList[0];
  for (sol = 0; sol < nSols; sol++) {
    if (solList[sol] > bestSol) {
      bestSol = solList[sol];
    }
  }
  return bestSol;
}

double Saldin5354Factor(double xh, double sh, double phihm, double xhLowerLimit) {
  double t1, t2, f, psi, psi2;
  if (xh < xhLowerLimit) {
    /* use Saldin 53 */
    t1 = (ipow3(phihm) + 3 * xh * sqr(phihm) - 6 * sh);
    t2 = 3 * (phihm + 2 * xh);
    f = 2 / (phihm + 2 * xh) * (1 + (t1 + t2) / sqrt(t1 * t1 + sqr(phihm * t2))) - 1 / sh;
  } else {
    if ((psi = SolveForPsiSaldin54(xh, sh)) >= 0) {
      psi2 = psi * psi;
      f = 4 * (2 * xh * (psi2 + 1) + psi * (psi2 + 2)) /
            (4 * xh * xh * (psi2 + 1) + 4 * xh * psi * (psi2 + 2) + psi2 * (psi2 + 4)) -
          1 / sh;
    } else
      return 0;
  }
  if (isnan(f) || isinf(f))
    f = 0;
  return f;
}

void exactDrift(double **part, int64_t np, double length) {
  int64_t i;
  double *coord;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    gpu_exactDrift(np, length);
#  ifdef GPU_VERIFY
    startCpuTimer();
    exactDrift(part, np, length);
    compareGpuCpu(np, "exactDrift");
#  endif /* GPU_VERIFY */
    return;
  }
#endif /* HAVE_GPU */

#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp taskloop if(gpuOmpTrackingRequested(np) && omp_in_parallel()) num_tasks(gpuGetOmpTrackingThreads())
#endif
  for (i = 0; i < np; i++) {
    coord = part[i];
    coord[0] += coord[1] * length;
    coord[2] += coord[3] * length;
    coord[4] += length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  }
}

double SolveForPhiStupakov(double x, double ds, double phim);
void DumpStupakovOutput(char *filename, SDDS_DATASET *SDDSout, long *active,
                        double zTravel, double *ctHist, double *ctHistDeriv,
                        double *dGamma, long nBins, double dct,
                        double MPCharge, double dz,
                        long nCaseC, long nCaseD1, long nCaseD2,
                        double x, double dsMax, double phi0, double phi1);

double SolveForPhiStupakovDiffSum = 0;
long SolveForPhiStupakovDiffCount = 0;

long track_through_driftCSR_Stupakov(double **part, int64_t np, CSRDRIFT *csrDrift,
                                     double Po, double **accepted, double zStart, CHARGE *charge, char *rootname) {
  int64_t iPart;
  long iKick, iBin, binned = 0, nKicks;
  long nCaseC, nCaseD1, nCaseD2;
  double ctLower, ctUpper, ds;
  long nBins, maxBins, nBinned, diBin;
  double *coord, p, beta, dz, factor, dz0, dzFirst;
  double zTravel, dct, zOutput;
  double *ctHist = NULL, *ctHistDeriv = NULL, *phiSoln = NULL;
  double length;
  long nBins1;
  double dsMax, x;
  TRACKING_CONTEXT tContext;
  LSCKICK lscKick;
#if USE_MPI
  long binned_total = 1, np_total = 1;
  double *buffer;
#endif

  getTrackingContext(&tContext);

  SolveForPhiStupakovDiffCount = 0;
  SolveForPhiStupakovDiffSum = 0;

  length = csrDrift->length;
  if (zStart != csrWake.zLast) {
    length += (dzFirst = zStart - csrWake.zLast);
    /* propagate beam back so we can tranverse the missing length including CSR
     */
    if (isSlave || !distributedBeam)
      for (iPart = 0; iPart < np; iPart++) {
        coord = part[iPart];
        coord[0] -= dzFirst * coord[1];
        coord[2] -= dzFirst * coord[3];
        if (csrDrift->linearOptics)
          coord[4] -= dzFirst;
        else
          coord[4] -= dzFirst * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      }
    zStart = csrWake.zLast;
  }
  zOutput = zStart; /* absolute coordinate used for output of data vs z or s */

  if (csrDrift->dz > 0) {
    if ((nKicks = length / csrDrift->dz + 0.5) < 1)
      nKicks = 1;
  } else
    nKicks = csrDrift->nKicks;
  if (nKicks <= 0)
    bombElegant("nKicks=0 in CSR drift.", NULL);
  dz = (dz0 = length / nKicks) / 2;

  zTravel = zStart - csrWake.z0; /* total distance traveled by radiation to reach this point */

  maxBins = nBins = csrWake.bins;
  nBins1 = nBins - 1;
  if (!(ctHist = SDDS_Malloc(sizeof(*ctHist) * nBins)) ||
      !(ctHistDeriv = SDDS_Malloc(sizeof(*ctHistDeriv) * nBins)) ||
      !(phiSoln = SDDS_Malloc(sizeof(*phiSoln) * nBins)))
    bombElegant("memory allocation failure (track_through_driftCSR)", NULL);

  if ((lscKick.bins = csrDrift->LSCBins) > 0) {
    lscKick.interpolate = csrDrift->LSCInterpolate;
    lscKick.radiusFactor = csrDrift->LSCRadiusFactor;
    lscKick.lowFrequencyCutoff0 = csrDrift->LSCLowFrequencyCutoff0;
    lscKick.lowFrequencyCutoff1 = csrDrift->LSCLowFrequencyCutoff1;
    lscKick.highFrequencyCutoff0 = csrDrift->LSCHighFrequencyCutoff0;
    lscKick.highFrequencyCutoff1 = csrDrift->LSCHighFrequencyCutoff1;
    lscKick.backtrack = 0;
  }
  for (iKick = 0; iKick < nKicks; iKick++) {
    /* first drift is dz=dz0/2, others are dz0 */
    if (iKick == 1)
      dz = dz0;
    zTravel += dz;
    zOutput += dz;

    x = zTravel / csrWake.rho;
    dsMax = csrWake.rho / 24 * pow(csrWake.bendingAngle, 3) * (csrWake.bendingAngle + 4 * x) / (csrWake.bendingAngle + x);
    /* propagate particles forward, converting s to c*t=s/beta */
    if (isSlave || !distributedBeam) {
      for (iPart = 0; iPart < np; iPart++) {
        coord = part[iPart];
        coord[0] += coord[1] * dz;
        coord[2] += coord[3] * dz;
        p = Po * (1 + coord[5]);
        beta = p / sqrt(p * p + 1);
        if (csrDrift->linearOptics)
          coord[4] = (coord[4] + dz) / beta;
        else
          coord[4] = (coord[4] + dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]))) / beta;
      }
    }
    /* bin the particle distribution */
    ctLower = ctUpper = dct = 0;
    nBinned = binParticleCoordinate(&ctHist, &maxBins,
                                    &ctLower, &ctUpper, &dct, &nBins,
                                    csrWake.binRangeFactor < 1.1 ? 1.1 : csrWake.binRangeFactor,
                                    part, np, 4);
#if USE_MPI
    if (distributedBeam) {
      if (isSlave)
        MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
      MPI_Allreduce(&nBinned, &binned_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    if (distributedBeam) { /* Master needs to know the information to write the result */
      buffer = malloc(sizeof(double) * nBins);
      MPI_Allreduce(ctHist, buffer, nBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      memcpy(ctHist, buffer, sizeof(double) * nBins);
      free(buffer);
    }
    if ((myid == 1) && (np_total != binned_total)) {
      dup2(fdStdout, fileno(stdout)); /* Let the first slave processor write the output */
      printf("Only %ld of %ld particles binned for CSRDRIFT (%s, BRF=%le, Stupakov)\n",
             binned_total, np_total,
             tContext.elementName, csrWake.binRangeFactor);
      fflush(stdout);
#else
      if (nBinned != np) {
        printf("Only %ld of %ld particles binned for CSRDRIFT (%s, BRF=%le, Stupakov)\n",
               nBinned, np,
               tContext.elementName, csrWake.binRangeFactor);
#endif
      printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
             ctLower, ctUpper, dct, nBins, maxBins);
      fflush(stdout);
#if USE_MPI
#  if defined(_WIN32)
      freopen("NUL", "w", stdout);
#  else
      if (!freopen("/dev/null", "w", stdout)) {
        perror("freopen failed");
        exit(EXIT_FAILURE);
      }
#  endif
#endif
    }

    /* - smooth the histogram, normalize to get linear density, and 
       copy in preparation for taking derivative
       */
    if (csrWake.highFrequencyCutoff0 > 0 || csrWake.lowFrequencyCutoff0 >= 0) {
      long nz;
      nz = applyLHPassFilters(ctHist, nBins,
                              csrWake.lowFrequencyCutoff0, csrWake.lowFrequencyCutoff1,
                              csrWake.highFrequencyCutoff0, csrWake.highFrequencyCutoff1,
                              csrWake.clipNegativeBins);
      if (nz) {
        char warningText[1024];
        snprintf(warningText, 1024, "Negative values in %ld bins.", nz);
        printWarningForTracking("Low pass filter resulted in negative values.",
                                warningText);
        fflush(stdout);
      }
    }

    if (csrWake.SGHalfWidth > 0) {
      SavitzkyGolaySmooth(ctHist, nBins, csrWake.SGOrder, csrWake.SGHalfWidth, csrWake.SGHalfWidth, 0);
#if (!USE_MPI)
      correctDistribution(ctHist, nBins, 1.0 * nBinned);
#else
        if (distributedBeam)
          correctDistribution(ctHist, nBins, 1.0 * binned_total);
        else
          correctDistribution(ctHist, nBins, 1.0 * nBinned);
#endif
    }
    for (iBin = 0; iBin < nBins; iBin++)
      ctHistDeriv[iBin] = (ctHist[iBin] /= dct);
    /* - compute derivative with smoothing.  The deriv is w.r.t. index number and
     * I won't scale it now as it will just fall out in the integral 
     */
    SavitzkyGolaySmooth(ctHistDeriv, nBins, csrWake.SGDerivOrder,
                        csrWake.SGDerivHalfWidth, csrWake.SGDerivHalfWidth, 1);

    /* Case C */
    nCaseC = 0;
    nCaseD1 = 0;
    nCaseD2 = 0;
    for (iBin = 0; iBin < nBins; iBin++) {
      double f;
      ds = csrWake.rho / 6 * sqr(csrWake.bendingAngle) * (csrWake.bendingAngle + 3 * x);
      diBin = ds / dct;
      if (iBin + diBin < nBins) {
        f = -1 / (csrWake.bendingAngle + 2 * x);
        csrWake.dGamma[iBin] = f * ctHist[iBin + diBin];
        nCaseC++;
      } else
        csrWake.dGamma[iBin] = 0;
    }
    /* Case D */
    for (iBin = 0; iBin < nBins; iBin++) {
      phiSoln[iBin] = -1;
      if ((ds = iBin * dct) > dsMax)
        break;
      phiSoln[iBin] = SolveForPhiStupakov(x, iBin * dct / csrWake.rho, csrWake.bendingAngle);
    }
    for (iBin = 0; iBin < nBins; iBin++) {
      long jBin, first, count;
      double term1 = 0, term2 = 0;
      diBin = dsMax / dct;
      if (iBin + diBin < nBins) {
        nCaseD1++;
        csrWake.dGamma[iBin] += ctHist[iBin + diBin] / (csrWake.bendingAngle + 2 * x);
      }
      first = 1;
      count = 0;
      for (jBin = iBin; jBin < nBins; jBin++) {
        double phi;
        if ((phi = phiSoln[jBin - iBin]) >= 0) {
          /* I put in a negative sign here because my s is opposite in direction to 
           * Saldin et al. and Stupakov, so my derivative has the opposite sign.
           * Note lack of ds factor here as I use the same one in my unnormalized derivative.
           */
          if (phi > 0) {
            /* ^^^ If I test phi+2*x here, I get noisy, unphysical results very close
             * to the dipole exit 
             */
            term2 = ctHistDeriv[jBin] / (phi + 2 * x);
            csrWake.dGamma[iBin] -= term2;
            if (first) {
              term1 = term2;
              first = 0;
            }
            count++;
            nCaseD2++;
          }
        } else
          break;
      }
      if (count > 1 && csrWake.trapazoidIntegration)
        /* trapazoid rule correction for ends */
        csrWake.dGamma[iBin] += (term1 + term2) / 2;
    }
    /* the minus sign adjusts for Stupakov using wake<0 to indicate energy gain
     */
    factor = -4 / csrWake.rho * csrWake.GSConstant * dz0;
    for (iBin = 0; iBin < nBins; iBin++)
      csrWake.dGamma[iBin] *= factor;

    if (csrWake.wffValues)
      applyFilterTable(csrWake.dGamma, nBins, dct / c_mks, csrWake.wffValues, csrWake.wffFreqValue,
                       csrWake.wffRealFactor, csrWake.wffImagFactor);

    if ((csrDrift->StupakovOutput || csrWake.StupakovFileActive) &&
        (csrDrift->StupakovOutputInterval < 2 || iKick % csrDrift->StupakovOutputInterval == 0)) {
      double x, dsMax, phi0, phi1;
      if (!csrWake.StupakovFileActive) {
        if (!SDDS_CopyString(&csrWake.StupakovOutput, csrDrift->StupakovOutput))
          bombElegant("string copying problem preparing Stupakov output for CSRDRIFT", NULL);
        csrWake.StupakovOutput = compose_filename(csrWake.StupakovOutput, rootname);
      }
      x = zTravel / csrWake.rho;
      dsMax = csrWake.rho / 24 * pow(csrWake.bendingAngle, 3) * (csrWake.bendingAngle + 4 * x) / (csrWake.bendingAngle + x);
      phi0 = SolveForPhiStupakov(x, 0.0, csrWake.bendingAngle);
      phi1 = SolveForPhiStupakov(x, dsMax / csrWake.rho * 0.999, csrWake.bendingAngle);

      /* note that the contents of ctHist and ctHistDeriv are corrupted by this operation */
      DumpStupakovOutput(csrWake.StupakovOutput, &csrWake.SDDS_Stupakov,
                         &csrWake.StupakovFileActive, zTravel,
                         ctHist, ctHistDeriv, csrWake.dGamma, nBins, dct, csrWake.MPCharge,
                         dz0, nCaseC, nCaseD1, nCaseD2,
                         x, dsMax / csrWake.rho, phi0, phi1);
    }

    /* apply kick to each particle and convert back to normal coordinates */
    if (isSlave || !distributedBeam) {
      for (iPart = binned = 0; iPart < np; iPart++) {
        double f;
        coord = part[iPart];
        iBin = (f = (coord[4] - ctLower) / dct);
        f -= iBin;
        if (iBin >= 0 && iBin < nBins1) {
          if (csrDrift->csr)
            coord[5] += ((1 - f) * csrWake.dGamma[iBin] + f * csrWake.dGamma[iBin + 1]) / Po;
          binned++;
        } else {
          printf("Particle out of bin range---not kicked: ct-ctLower=%21.15e, dct=%21.15e, iBin=%ld\n",
                 coord[4] - ctLower, dct, iBin);
        }
        p = (1 + coord[5]) * Po;
        beta = p / sqrt(p * p + 1);
        coord[4] = beta * coord[4];
      }
    }

    if (tContext.sliceAnalysis && tContext.sliceAnalysis->active &&
        (csrDrift->sliceAnalysisInterval == 0 ||
         iKick % csrDrift->sliceAnalysisInterval == 0)) {
#if USE_MPI
      /* This function will be parallelized in the future */
      printf("performSliceAnalysisOutput is not supported in parallel mode currently.\n");
      MPI_Barrier(MPI_COMM_WORLD); /* Make sure the information can be printed before aborting */
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      performSliceAnalysisOutput(tContext.sliceAnalysis, part, np,
                                 0, tContext.step, Po,
                                 csrWake.MPCharge * np,
                                 tContext.elementName,
                                 zOutput, 0);
    }
#if USE_MPI
    if (isSlave && distributedBeam) {
      MPI_Allreduce(&binned, &binned_total, 1, MPI_LONG, MPI_SUM, workers);
    }
    if ((myid == 1) && (np_total != binned_total)) {
      dup2(fdStdout, fileno(stdout)); /* Let the first slave processor write the output */
      printf("Only %ld of %ld particles kicked for CSRDRIFT (%s, BRF=%le, Stupakov)\n",
             binned_total, np_total,
             tContext.elementName, csrWake.binRangeFactor);
#else
      if (np != binned) {
        printf("Only %ld of %ld particles kicked for CSRDRIFT (%s, BRF=%le, Stupakov)\n",
               binned, np,
               tContext.elementName, csrWake.binRangeFactor);
#endif
      printf("ct min, max = %21.15e, %21.15e, dct = %21.15e, nBins=%ld, maxBins=%ld\n",
             ctLower, ctUpper, dct, nBins, maxBins);
      fflush(stdout);
#if USE_MPI
#  if defined(_WIN32)
      freopen("NUL", "w", stdout);
#  else
      if (!freopen("/dev/null", "w", stdout)) {
        perror("freopen failed");
        exit(EXIT_FAILURE);
      }
#  endif
#endif
    }

    if (csrDrift->LSCBins > 0)
      addLSCKick(part, np, &lscKick, Po, charge, dz, 0.0);
  }

  /* do final drift of dz0/2 */
  dz = dz0 / 2;
  if (isSlave || !distributedBeam)
    for (iPart = 0; iPart < np; iPart++) {
      coord = part[iPart];
      coord[0] += coord[1] * dz;
      coord[2] += coord[3] * dz;
      if (csrDrift->linearOptics)
        coord[4] += dz;
      else
        coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    }

  if (csrDrift->LSCBins > 0)
    addLSCKick(part, np, &lscKick, Po, charge, dz, 0.0);

  csrWake.zLast = zStart + length;
  free(ctHist);
  free(ctHistDeriv);
  free(phiSoln);
#if DEBUG
  if (SolveForPhiStupakovDiffCount)
    printf("Phi solution accuracy for %ld solutions: %le\n",
           SolveForPhiStupakovDiffCount, SolveForPhiStupakovDiffSum / SolveForPhiStupakovDiffCount);
#endif
  return np;
}

static double SolveForPhiStupakov_x, SolveForPhiStupakov_4x;

double SolveForPhiStupakovFn(double phi) {
  return phi * phi * phi * (phi + SolveForPhiStupakov_4x) / (phi + SolveForPhiStupakov_x);
}

/* solve for phi:  ds=phi^3/24*(phi+4*x)/(phi+x), where ds = (s-s')/rho */

double SolveForPhiStupakov(double x, double ds, double phim) {
  double phi;
  static double phiLast = -1;

  if (ds < 0)
    return -1;
  if (ds == 0)
    return 0;

  ds *= 24;
  SolveForPhiStupakov_x = x;
  SolveForPhiStupakov_4x = 4 * x;

  if (phiLast == -1)
    phiLast = phim / 2;

  /* try phim first */
  if (fabs(ds - SolveForPhiStupakovFn(phim)) < ds / 1e4) {
    phiLast = phim;
    return phim;
  }

  /* try a solution with Newton's method */
  phi = zeroNewton(SolveForPhiStupakovFn, ds, phiLast, phim / 1000, 3, ds / 1e4);
  if (phi < 0 || phi > phim || fabs(ds - SolveForPhiStupakovFn(phi)) > ds / 1e4)
    /* try a more plodding method */
    phi = zeroInterp(SolveForPhiStupakovFn, ds, 0, phim * 1.01, phim / 100, ds / 1e4);
  if (phi < 0 || phi > phim)
    return -1;
  phiLast = phi;
  SolveForPhiStupakovDiffCount++;
  SolveForPhiStupakovDiffSum += fabs(ds - SolveForPhiStupakovFn(phi));
  return phi;
}

/* this procedure destroys the contents of ctHist and ctHistDeriv ! */

void DumpStupakovOutput(char *filename, SDDS_DATASET *SDDSout, long *active,
                        double zTravel, double *ctHist, double *ctHistDeriv,
                        double *dGamma, long nBins, double dct,
                        double MPCharge, double dz,
                        long nCaseC, long nCaseD1, long nCaseD2,
                        double x, double dsMax, double phi0, double phi1) {
  long i;
  if (!*active) {
    if (!SDDS_InitializeOutputElegant(SDDSout, SDDS_BINARY, 1, NULL, NULL, filename) ||
        0 > SDDS_DefineParameter(SDDSout, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
        !SDDS_DefineSimpleParameter(SDDSout, "z", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDSout, "CaseC", "#", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDSout, "CaseD1", "#", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDSout, "CaseD2", "#", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDSout, "x", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDSout, "dsMax", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDSout, "phi0", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDSout, "phi1", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(SDDSout, "s", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(SDDSout, "LinearDensity", "C/s", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(SDDSout, "LinearDensityDeriv", "C/s$a2$n", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(SDDSout, "DeltaGamma", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(SDDSout, "GammaDeriv", "1/m", SDDS_DOUBLE) ||
        !SDDS_WriteLayout(SDDSout)) {
      SDDS_SetError("Problem setting up output file for CSRDRIFT (Stupakov mode)");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    *active = 1;
  }
  for (i = 0; i < nBins; i++) {
    ctHist[i] *= MPCharge * c_mks;
    ctHistDeriv[i] *= MPCharge * sqr(c_mks) / dct;
  }
  if (!SDDS_StartPage(SDDSout, nBins) ||
      !SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, dGamma, nBins, "DeltaGamma") ||
      !SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, ctHist, nBins, "LinearDensity") ||
      !SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, ctHistDeriv, nBins, "LinearDensityDeriv")) {
    SDDS_SetError("Problem writing to output file for CSRDRIFT (Stupakov mode)");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  /* use ctHist array for output of s and ctHistDeriv for dGamma/ds */
  for (i = 0; i < nBins; i++) {
    ctHist[i] = dct * (i + 0.5 - nBins / 2);
    ctHistDeriv[i] = dGamma[i] / dz;
  }
  if (!SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, ctHist, nBins, "s") ||
      !SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, ctHistDeriv, nBins, "GammaDeriv") ||
      !SDDS_SetParameters(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "z", zTravel, "CaseC", nCaseC,
                          "CaseD1", nCaseD1, "CaseD2", nCaseD2,
                          "x", x, "dsMax", dsMax, "phi0", phi0, "phi1", phi1,
                          NULL) ||
      !SDDS_WritePage(SDDSout)) {
    SDDS_SetError("Problem writing to output file for CSRDRIFT (Stupakov mode)");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDSout);
}

void apply_edge_effects(
  double *x, double *xp, double *y, double *yp,
  double rho, double n, double beta, double he, double psi, long which_edge)
/* Applies edge effects using non-symplectic K. L. Brown method to second order */
{
  double h, tan_beta, tan2_beta, sec_beta, sec2_beta, h2;
  double R21, R43;
  double T111, T133, T211, T441, T331, T221, T233, T243, T431, T432;
  double x0, xp0, y0, yp0;

  h = 1 / rho;
  R21 = h * (tan_beta = tan(beta));
  R43 = -h * tan(beta - psi);

  h2 = sqr(h);
  T111 = which_edge * h / 2 * (tan2_beta = sqr(tan_beta));
  sec_beta = 1. / cos(beta);
  sec2_beta =  sqr(sec_beta);
  T133 = -which_edge * h / 2 * sec2_beta;
  T211 = which_edge == -1 ? -n * h2 * tan_beta : -h2 * (n + tan2_beta / 2) * tan_beta;
  T441 = -(T331 = T221 = -which_edge * h * tan2_beta);
  T233 = which_edge == -1 ? h2 * (n + .5 + tan2_beta) * tan_beta : h2 * (n - tan2_beta / 2) * tan_beta;
  T243 = which_edge * h * tan2_beta;
  T431 = h2 * (2 * n + (which_edge == 1 ? sec2_beta : 0)) * tan_beta;
  T432 = which_edge * h * sec2_beta;
  if (he != 0) {
    double term;
    term = h / 2 * he * sec2_beta * sec_beta;
    T211 += term;
    T233 -= term;
    T431 -= 2 * term;
  }

  x0 = *x;
  xp0 = *xp;
  y0 = *y;
  yp0 = *yp;
  *x = x0 + T111 * sqr(x0) + T133 * sqr(y0);
  *xp = xp0 + R21 * x0 + T211 * sqr(x0) + T221 * x0 * xp0 + T233 * sqr(y0) + T243 * y0 * yp0;
  *y = y0 + T331 * x0 * y0;
  *yp = yp0 + R43 * y0 + T441 * yp0 * x0 + T431 * x0 * y0 + T432 * xp0 * y0;
}

/* dipole fringe effects tracking, based on work of Kilean Hwang. Not symplectic for edgeOrder>=2 */

void dipoleFringeKHwang(double *Qf, double *Qi,
                        double rho, double inFringe, long edgeOrder, double K1, double edge, double gap, double fint, double Rhe) {
  double dx, dpx, dy, dpy;
  double tan_edge, sin_edge, sec_edge, cos_edge;
  double x0, px0, y0, py0, dp0;
  /* double psi, Kg; */
  double k0, k3, k2;
  double k4, k5, k6;

  k0 = sqr(PI) / 6.;
  k2 = fint;
  k3 = 1.0 * 1. / 6.;
  /* Kg = gap*fint; */
  k4 = -1.0 * sqr(PI) / 3.;
  k5 = 0.0;
  k6 = -1.0;

  x0 = Qi[0];
  px0 = Qi[1];
  y0 = Qi[2];
  py0 = Qi[3];
  dp0 = Qi[5];
  dx = dpx = dy = dpy = 0;
  /* psi = Kg/rho/cos(edge)*(1+sqr(sin(edge))); */

  sec_edge = 1. / cos(edge);
  tan_edge = tan(edge);
  sin_edge = sin(edge);
  cos_edge = cos(edge);

  if (edgeOrder > 1) {

    /* entrance */
    if (inFringe == -1.) {
      dx = inFringe * ipow2(sec_edge) * ipow2(gap) * k0 / rho / (1 + dp0) + inFringe * ipow2(x0) * ipow2(tan_edge) / 2 / rho / (1 + dp0) - inFringe * ipow2(y0) * ipow2(sec_edge) / 2 / rho / (1 + dp0);
      dy = -inFringe * x0 * y0 * ipow2(tan_edge) / rho / (1 + dp0);
      dpx = -1. * ipow3(sec_edge) * sin_edge * ipow2(gap) * k0 / rho / rho / (1 + dp0) + tan_edge * x0 / rho + ipow2(y0) / 2 * (2 * ipow3(tan_edge)) / ipow2(rho) / (1 + dp0) + ipow2(y0) / 2 * (ipow1(tan_edge)) / ipow2(rho) / (1 + dp0) - inFringe * (x0 * px0 - y0 * py0) * ipow2(tan_edge) / rho / (1 + dp0) + k4 * ipow2(sin_edge) * ipow2(gap) / 2 / ipow3(cos_edge) / rho * Rhe - k5 * x0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe + k6 * (y0 * y0 - x0 * x0) / 2 / ipow3(cos_edge) / rho * Rhe;
      dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + ipow2(sin_edge)) * gap / (1 + dp0) / ipow2(rho) / ipow3(cos_edge) + inFringe * (x0 * py0 + y0 * px0) * ipow2(tan_edge) / rho / (1 + dp0) + inFringe * y0 * px0 / rho / (1 + dp0) + k3 * ipow3(y0) * (2. / 3. / cos_edge - 4. / 3. / ipow3(cos_edge)) / (1 + dp0) / rho / rho / gap + k6 * x0 * y0 / ipow3(cos_edge) / rho * Rhe;
    }
    /* exit */
    if (inFringe == 1.) {
      dx = inFringe * ipow2(sec_edge) * ipow2(gap) * k0 / rho / (1 + dp0) + inFringe * ipow2(x0) * ipow2(tan_edge) / 2 / rho / (1 + dp0) - inFringe * ipow2(y0) * ipow2(sec_edge) / 2 / rho / (1 + dp0);
      dy = -inFringe * x0 * y0 * ipow2(tan_edge) / rho / (1 + dp0);
      dpx = tan_edge * x0 / rho - ipow2(y0) / 2 * (1 * ipow3(tan_edge)) / ipow2(rho) / (1 + dp0) - ipow2(x0) / 2 * (1 * ipow3(tan_edge)) / ipow2(rho) / (1 + dp0) - inFringe * (x0 * px0 - y0 * py0) * ipow2(tan_edge) / rho / (1 + dp0) + k4 * ipow2(sin_edge) * ipow2(gap) / 2 / ipow3(cos_edge) / rho * Rhe - k5 * x0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe + k6 * (y0 * y0 - x0 * x0) / 2 / ipow3(cos_edge) / rho * Rhe;
      dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + ipow2(sin_edge)) * gap / (1 + dp0) / ipow2(rho) / ipow3(cos_edge) + inFringe * (x0 * py0 + y0 * px0) * ipow2(tan_edge) / rho / (1 + dp0) + inFringe * y0 * px0 / rho / (1 + dp0) + x0 * y0 * ipow2(sec_edge) * tan_edge / ipow2(rho) / (1 + dp0) + k3 * ipow3(y0) * (2. / 3. / cos_edge - 4. / 3. / ipow3(cos_edge)) / (1 + dp0) / rho / rho / gap - k5 * y0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe + k6 * x0 * y0 / ipow3(cos_edge) / rho * Rhe;
    }

  } else {
    /* linear terms in transverse coordinates only */

    /* entrance */
    if (inFringe == -1.) {
      dx = inFringe * ipow2(sec_edge) * ipow2(gap) * k0 / rho / (1 + dp0);
      dy = 0;
      dpx = -1. * ipow3(sec_edge) * sin_edge * ipow2(gap) * k0 / rho / rho / (1 + dp0) + tan_edge * x0 / rho + k4 * ipow2(sin_edge) * ipow2(gap) / 2 / ipow3(cos_edge) / rho * Rhe - k5 * x0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe;
      dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + ipow2(sin_edge)) * gap / (1 + dp0) / ipow2(rho) / ipow3(cos_edge);
    }

    /* exit */
    if (inFringe == 1.) {
      dx = inFringe * ipow2(sec_edge) * ipow2(gap) * k0 / rho / (1 + dp0);
      dy = 0;
      dpx = tan_edge * x0 / rho + k4 * ipow2(sin_edge) * ipow2(gap) / 2 / ipow3(cos_edge) / rho * Rhe - k5 * x0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe;
      dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + ipow2(sin_edge)) * gap / (1 + dp0) / ipow2(rho) / ipow3(cos_edge) - k5 * y0 * ipow1(sin_edge) * ipow1(gap) / ipow3(cos_edge) / rho * Rhe;
    }
  }

  Qf[0] = x0 + dx;
  Qf[1] = px0 + dpx;
  Qf[2] = y0 + dy;
  Qf[3] = py0 + dpy;
  Qf[5] = Qi[5];
  /*  printf("x %f y %f xp %f yp %f dp0 %f\n", *x, *y, *xp, *yp, dp0); */
}

/* Symplectic higher-order dipole fringe effects tracking, based on work of Kilean Hwang as further developed by Ryan Linberg */

void dipoleFringeKHwangRLindberg(double *Qf, double *Qi,
                                 double rho, double inFringe, double K1, double edge, double gap, double fint, double Rhe) {
  double tan_edge, sin_edge, sec_edge, cos_edge;
  double cos3_edge, sec2_edge, tan2_edge, tan3_edge;
  double x0, px0, y0, py0, dp0;
  double x1, px1, y1, py1;
  double x2, px2, y2, py2;
  double x3, px3, y3, py3;
  double x4, px4, y4, py4;
  double x5, px5, y5, py5;
  double k0, k3, k2;
  double k4, k5, k6;
  double t1, t2, rho2;

  k0 = sqr(PI) / 6.;
  k2 = fint;
  k3 = 1.0 * 1. / 6.;
  k4 = -1.0 * sqr(PI) / 3.;
  k5 = 0.0;
  k6 = -1.0;
  rho2 = sqr(rho);

  x0 = Qi[0];
  px0 = Qi[1];
  y0 = Qi[2];
  py0 = Qi[3];
  dp0 = Qi[5];

  sec_edge = 1. / cos(edge);
  tan_edge = tan(edge);
  sin_edge = sin(edge);
  cos_edge = cos(edge);

  sec2_edge = ipow2(sec_edge);
  cos3_edge = ipow3(cos_edge);
  tan2_edge = ipow2(tan_edge);
  tan3_edge = ipow3(tan_edge);

  if (inFringe == -1) {
    /* entrance */

    x1 = x0;
    px1 = px0 + tan_edge / rho * x0 + tan_edge / (2 * rho2 * (1 + dp0)) * sqr(y0) - tan3_edge / (rho2 * (1 + dp0)) * sqr(x0) - gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * x0 + k6 * sec2_edge * Rhe / (2 * rho) * (sqr(y0) - sqr(x0));
    y1 = y0;
    py1 = py0 - tan_edge / rho * y0 + tan_edge / (rho2 * (1 + dp0)) * x0 * y0 + gap * k2 * (1 + sqr(sin_edge)) / (rho2 * (1 + dp0) * cos3_edge) * y0 + 2 * k3 * (sqr(cos_edge) - 2) / (3 * gap * rho2 * cos3_edge) * ipow3(y0) + gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * y0 + k6 * ipow3(sec_edge) * Rhe / rho * x0 * y0;

    t1 = (1 + tan2_edge / (2 * rho * (1 + dp0)) * x1);
    x2 = x1 / t1;
    px2 = px1 * sqr(t1);
    y2 = y1;
    py2 = py1;

    t1 = sec2_edge / (rho * (1 + dp0));
    x3 = x2 + t1 / 2 * sqr(y2);
    px3 = px2;
    y3 = y2;
    py3 = py2 - t1 * y2 * px2;

    t1 = tan2_edge / (rho * (1 + dp0));
    t2 = exp(t1 * x3);
    x4 = x3;
    px4 = px3 - t1 * y3 * py3;
    y4 = y3 * t2;
    py4 = py3 / t2;

    t1 = sqr(gap) * k0 * sec2_edge / (rho * (1 + dp0));
    x5 = x4 - t1;
    px5 = px4 - t1 * tan_edge / rho + sqr(gap) * k4 * sqr(sin_edge) * Rhe / (2 * rho * cos3_edge);
    y5 = y4;
    py5 = py4;
  } else {
    /* exit */

    x1 = x0;
    px1 = px0 + tan_edge / rho * x0 + tan3_edge / (2 * rho2 * (1 + dp0)) * sqr(y0) + tan3_edge / (2 * rho2 * (1 + dp0)) * sqr(x0) - gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * x0 + k6 * sec2_edge * Rhe / (2 * rho) * (sqr(y0) - sqr(x0));
    y1 = y0;
    py1 = py0 - tan_edge / rho * y0 + tan3_edge / (rho2 * (1 + dp0)) * x0 * y0 + gap * k2 * (1 + sqr(sin_edge)) / (rho2 * (1 + dp0) * cos3_edge) * y0 + 2 * k3 * (sqr(cos_edge) - 2) / (3 * gap * rho2 * cos3_edge) * ipow3(y0) + gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * y0 + k6 * ipow3(sec_edge) * Rhe / rho * x0 * y0;

    t1 = (1 - tan2_edge / (2 * rho * (1 + dp0)) * x1);
    x2 = x1 / t1;
    px2 = px1 * sqr(t1);
    y2 = y1;
    py2 = py1;

    t1 = sec2_edge / (rho * (1 + dp0));
    x3 = x2 - t1 / 2 * sqr(y2);
    px3 = px2;
    y3 = y2;
    py3 = py2 + t1 * y2 * px2;

    t1 = -tan2_edge / (rho * (1 + dp0));
    t2 = exp(t1 * x3);
    x4 = x3;
    px4 = px3 - t1 * y3 * py3;
    y4 = y3 * t2;
    py4 = py3 / t2;

    x5 = x4 + sqr(gap) * k0 * sec2_edge / (rho * (1 + dp0));
    px5 = px4 + sqr(gap) * k4 * sqr(sin_edge) * Rhe / (2 * rho * cos3_edge);
    y5 = y4;
    py5 = py4;
  }

  Qf[0] = x5;
  Qf[1] = px5;
  Qf[2] = y5;
  Qf[3] = py5;
  Qf[5] = dp0;
}

/* this is used solely to convert coordinates inside the element for
 * the purpose of generating output.  It ignores misalignments.
 */

void convertFromCSBendCoords(double **part, int64_t np, double rho0,
                             double cos_ttilt, double sin_ttilt,
                             long ctMode) {
  int64_t ip;
  double x, y, xp, yp, *coord;

  for (ip = 0; ip < np; ip++) {
    coord = part[ip];

    x = X * cos_ttilt - Y * sin_ttilt;
    y = X * sin_ttilt + Y * cos_ttilt;
    xp = XP * cos_ttilt - YP * sin_ttilt;
    yp = XP * sin_ttilt + YP * cos_ttilt;

    X = x;
    Y = y;
    XP = xp;
    YP = yp;

    if (ctMode)
      coord[4] /= c_mks;
  }
}

/* this is used solely to undo the transformation done by 
 * convertFromCSBendCoords
 */

void convertToCSBendCoords(double **part, int64_t np, double rho0,
                           double cos_ttilt, double sin_ttilt, long ctMode) {
  int64_t ip;
  double x, y, xp, yp, *coord;

  for (ip = 0; ip < np; ip++) {
    coord = part[ip];

    x = X * cos_ttilt + Y * sin_ttilt;
    y = -X * sin_ttilt + Y * cos_ttilt;
    xp = XP * cos_ttilt + YP * sin_ttilt;
    yp = -XP * sin_ttilt + YP * cos_ttilt;

    X = x;
    Y = y;
    XP = xp;
    YP = yp;

    if (ctMode)
      coord[4] *= c_mks;
  }
}

#include "fftpackC.h"
long applyLowPassFilter(double *histogram, long bins,
                        double start, /* in units of Nyquist frequency */
                        double end    /* in units of Nyquist frequency */
) {
  long i, i1, i2;
  double fraction, dfraction, sum;
  double *realimag;
  long frequencies;

  if (!(realimag = (double *)malloc(sizeof(*realimag) * (bins + 2))))
    SDDS_Bomb("allocation failure");

  if (end < start)
    end = start;

  frequencies = bins / 2 + 1;
  realFFT2(realimag, histogram, bins, 0);

  i1 = start * frequencies;
  if (i1 < 0)
    i1 = 0;
  if (i1 > frequencies - 1)
    i1 = frequencies - 1;

  i2 = end * frequencies;
  if (i2 < 0)
    i2 = 0;
  if (i2 > frequencies - 1)
    i2 = frequencies - 1;

  dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
  fraction = 1;
  for (i = i1; i <= i2; i++) {
    realimag[2 * i] *= fraction;
    realimag[2 * i + 1] *= fraction;
    if ((fraction -= dfraction) < 0)
      fraction = 0;
  }
  for (; i < frequencies; i++) {
    realimag[2 * i] = 0;
    realimag[2 * i + 1] = 0;
  }

  realFFT2(realimag, realimag, bins, INVERSE_FFT);

  /* copy data to input buffer.
   * normalize to keep the sum constant
   * don't allow negative values 
   */
  for (i = sum = 0; i < bins; i++) {
    sum += histogram[i];
    histogram[i] = realimag[i];
  }
  free(realimag);
  return correctDistribution(histogram, bins, sum);
}

long applyLHPassFilters(double *histogram, long bins,
                        double startHP, /* in units of Nyquist frequency */
                        double endHP,   /* in units of Nyquist frequency */
                        double startLP, /* in units of Nyquist frequency */
                        double endLP,   /* in units of Nyquist frequency */
                        long clipNegative) {
  long i, i1, i2;
  double fraction, dfraction, sum;
  double *realimag;
  long frequencies;

  if (!(realimag = (double *)malloc(sizeof(*realimag) * (bins + 2))))
    SDDS_Bomb("allocation failure");

  if (endLP < startLP)
    endLP = startLP;
  if (endHP < startHP)
    endHP = startHP;

  frequencies = bins / 2 + 1;
  realFFT2(realimag, histogram, bins, 0);

  if (startLP > 0) {
    i1 = startLP * frequencies;
    if (i1 < 0)
      i1 = 0;
    if (i1 > frequencies - 1)
      i1 = frequencies - 1;

    i2 = endLP * frequencies;
    if (i2 < 0)
      i2 = 0;
    if (i2 > frequencies - 1)
      i2 = frequencies - 1;

    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 1;
    for (i = i1; i <= i2; i++) {
      realimag[2 * i] *= fraction;
      realimag[2 * i + 1] *= fraction;
      if ((fraction -= dfraction) < 0)
        fraction = 0;
    }
    for (; i < frequencies; i++) {
      realimag[2 * i] = 0;
      realimag[2 * i + 1] = 0;
    }
  }

  if (startHP > 0) {
    i1 = startHP * frequencies;
    if (i1 < 0)
      i1 = 0;
    if (i1 > frequencies - 1)
      i1 = frequencies - 1;

    i2 = endHP * frequencies;
    if (i2 < 0)
      i2 = 0;
    if (i2 > frequencies - 1)
      i2 = frequencies - 1;

    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 0;
    for (i = 0; i < i1; i++) {
      realimag[2 * i] = 0;
      realimag[2 * i + 1] = 0;
    }
    for (i = i1; i <= i2; i++) {
      realimag[2 * i] *= fraction;
      realimag[2 * i + 1] *= fraction;
      if ((fraction += dfraction) > 1)
        fraction = 1;
    }
  }

  realFFT2(realimag, realimag, bins, INVERSE_FFT);

  /* copy data to input buffer  */
  for (i = sum = 0; i < bins; i++) {
    sum += histogram[i];
    histogram[i] = realimag[i];
  }
  free(realimag);

  if (clipNegative)
    /* normalize to keep the sum constant
     * don't allow negative values 
     */
    return correctDistribution(histogram, bins, sum);
  else
    return 0;
}

long correctDistribution(double *array, long npoints, double desiredSum) {
  double sum, factor;
  long nz, i;
  for (i = nz = sum = 0; i < npoints; i++) {
    if (array[i] < 0) {
      nz++;
      array[i] = 0;
    }
    sum += array[i];
  }
  if (!sum)
    return nz;
  factor = desiredSum / sum;
  for (i = 0; i < npoints; i++)
    array[i] *= factor;
  return nz;
}

void computeEtiltCentroidOffset(double *dcoord_etilt, double rho0, double angle, double etilt, double tilt) {
  /* compute final offsets due to error-tilt of the magnet */
  /* see pages 90-93 of notebook 1 about this */
  double q1a, q2a, q3a;
  double q1b, q2b, q3b;
  double qp1, qp2, qp3;
  double dz, tan_alpha, k;

  if (!etilt) {
    fill_double_array(dcoord_etilt, 6L, 0.0);
    return;
  }

  etilt *= -1; /* consistent sign convention with TILT */

  q1a = (1 - cos(angle)) * rho0 * (cos(etilt) - 1);
  q2a = 0;
  q3a = (1 - cos(angle)) * rho0 * sin(etilt);
  qp1 = sin(angle) * cos(etilt);
  qp2 = cos(angle);
  k = sqrt(sqr(qp1) + sqr(qp2));
  qp1 /= k;
  qp2 /= k;
  qp3 = sin(angle) * sin(etilt) / k;
  tan_alpha = 1. / tan(angle) / cos(etilt);
  q1b = q1a * tan_alpha / (tan(angle) + tan_alpha);
  q2b = -q1b * tan(angle);
  dz = sqrt(sqr(q1b - q1a) + sqr(q2b - q2a));
  q3b = q3a + qp3 * dz;

  dcoord_etilt[0] = sqrt(sqr(q1b) + sqr(q2b));
  dcoord_etilt[1] = tan(atan(tan_alpha) - (PIo2 - angle));
  dcoord_etilt[2] = q3b;
  dcoord_etilt[3] = qp3;
  dcoord_etilt[4] = dz * sqrt(1 + sqr(qp3));
  dcoord_etilt[5] = 0;

#ifdef DEBUG
  printf("pre-tilt offsets due to ETILT=%le:  %le %le %le %le %le\n",
         etilt, dcoord_etilt[0], dcoord_etilt[1], dcoord_etilt[2],
         dcoord_etilt[3], dcoord_etilt[4]);
  fflush(stdout);
#endif

  /* rotate by tilt to get into same frame as bend equations. */
  rotateCoordinatesForMisalignment(dcoord_etilt, -tilt);
#ifdef DEBUG
  printf("offsets due to ETILT=%le:  %le %le %le %le %le\n",
         etilt, dcoord_etilt[0], dcoord_etilt[1], dcoord_etilt[2],
         dcoord_etilt[3], dcoord_etilt[4]);
  fflush(stdout);
#endif
}

void readWakeFilterFile(long *values,
                        double **freq, double **real, double **imag,
                        char *freqName, char *realName, char *imagName,
                        char *filename) {
  SDDS_DATASET SDDSin;
  long i;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename) || !SDDS_ReadPage(&SDDSin)) {
    fprintf(stderr, "Error: unable to open or read CSRCSBEND wake filter file %s\n", filename);
    exitElegant(1);
  }
  if ((*values = SDDS_RowCount(&SDDSin)) < 2) {
    fprintf(stderr, "Error: too little data in CSRCSBEND wake filter file %s\n", filename);
    exitElegant(1);
  }
  if (!freqName || !strlen(freqName))
    SDDS_Bomb("WFF_FREQ_COLUMN is blank in CSRCSBEND");
  if (SDDS_CheckColumn(&SDDSin, freqName, "Hz", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
    fprintf(stderr, "Error: column %s invalid in CSRCSBEND wake filter file %s---check existence, type, and units (Hz).\n",
            freqName, filename);
    exitElegant(1);
  }
  if (!realName || !strlen(realName))
    SDDS_Bomb("WFF_REAL_COLUMN is blank in CSRCSBEND");
  if (SDDS_CheckColumn(&SDDSin, realName, NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
    fprintf(stderr, "Error: column %s invalid in CSRCSBEND wake filter file %s---check existence and type.\n",
            realName, filename);
    exitElegant(1);
  }
  if (!imagName || !strlen(imagName))
    SDDS_Bomb("WFF_IMAG_COLUMN is blank in CSRCSBEND");
  if (SDDS_CheckColumn(&SDDSin, imagName, NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
    fprintf(stderr, "Error: column %s invalid in CSRCSBEND wake filter file %s---check existence and type.\n",
            imagName, filename);
    exitElegant(1);
  }
  if (!(*freq = SDDS_GetColumnInDoubles(&SDDSin, freqName)) ||
      !(*real = SDDS_GetColumnInDoubles(&SDDSin, realName)) ||
      !(*imag = SDDS_GetColumnInDoubles(&SDDSin, imagName))) {
    fprintf(stderr, "Problem getting data from CSRCSBEND wake filter file %s.\n", filename);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  for (i = 1; i < *values; i++) {
    if ((*freq)[i - 1] >= (*freq)[i]) {
      fprintf(stderr, "Error: frequency data is not monotonically increasing in CSRCSBEND wake filter file %s.\n", filename);
      exitElegant(1);
    }
  }
}

void applyFilterTable(double *function, long bins, double dx, long fValues,
                      double *fFreq, double *fReal, double *fImag) {
  long i, i1, i2;
  double f;
  double *realimag, dfrequency, length;
  long frequencies;
  double sum;

  if (!(realimag = (double *)malloc(sizeof(*realimag) * (bins + 2))))
    SDDS_Bomb("allocation failure");

  frequencies = bins / 2 + 1;
  length = dx * (bins - 1);
  dfrequency = 1.0 / length;
  realFFT2(realimag, function, bins, 0);

  for (i = 0; i < frequencies; i++) {
    long code;
    i1 = 2 * i + 0;
    i2 = 2 * i + 1;
    f = i * dfrequency;
    realimag[i1] *= interp(fReal, fFreq, fValues, f, 0, 1, &code);
    realimag[i2] *= interp(fImag, fFreq, fValues, f, 0, 1, &code);
  }
  realFFT2(realimag, realimag, bins, INVERSE_FFT);

  /* copy data to input buffer.
   */
  for (i = sum = 0; i < bins; i++)
    function[i] = realimag[i];
  free(realimag);
}

void addRadiationKick(double *Qx, double *Qy, double *dPoP, double *sigmaDelta2,
                      double x, double y, double theta, double thetaf, double h0, double Fx, double Fy,
                      double ds, double radCoef, double dsISR, double isrCoef,
                      long distributionBased, long includeOpeningAngle, double meanPhotonsPerMeter,
                      double normalizedCriticalEnergy0, double Po) {
  double f, xp, yp, F2, F, deltaFactor, dsFactor;

  f = (1 + x * h0) / sqrt(sqr(1 + *dPoP) - sqr(*Qx) - sqr(*Qy));
  xp = *Qx * f;
  yp = *Qy * f;
  dsFactor = sqrt(sqr(1 + x * h0) + sqr(xp) + sqr(yp));
  F2 = sqr(Fx) + sqr(Fy);

  if (!distributionBased) {
    deltaFactor = sqr(1 + *dPoP);
    *Qx /= (1 + *dPoP);
    *Qy /= (1 + *dPoP);
    if (radCoef)
      *dPoP -= radCoef * deltaFactor * F2 * ds * dsFactor;
    if (isrCoef > 0)
      /* The minus sign is for consistency with the previous version. */
      *dPoP -= isrCoef * deltaFactor * pow(F2, 0.75) * sqrt(dsISR * dsFactor) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
    if (sigmaDelta2)
      *sigmaDelta2 += sqr(isrCoef * deltaFactor) * pow(F2, 1.5) * dsISR * dsFactor;
    *Qx *= (1 + *dPoP);
    *Qy *= (1 + *dPoP);
  } else {
    double dtheta = 0, dphi = 0;
    double yph, logyph;
    double normalizedCriticalEnergy;
    double nMean, dDelta, thetaRms;
    long i, nEmitted;
    long rhoSign;
    F = sqrt(F2);
    rhoSign = SIGN(h0);
    /* Compute the mean number of photons emitted = meanPhotonsPerMeter*meters */
    /* Note that unlike the #photons/radian, this is independent of energy */
    nMean = meanPhotonsPerMeter * dsISR * dsFactor * F;
    /* Pick the actual number of photons emitted from Poisson distribution */
    nEmitted = inversePoissonCDF(nMean, random_2(1));
    /* Adjust normalized critical energy to local field strength (FSE is already included via rho_actual) */
    normalizedCriticalEnergy = normalizedCriticalEnergy0 * F;
    /* For each photon, pick its energy and emission angles */
    for (i = 0; i < nEmitted; i++) {
      /* Pick photon energy normalized to critical energy */
      yph = pickNormalizedPhotonEnergy(random_2(1));
      /* Multiply by critical energy normalized to central beam energy, adjusting for variation with
       * individual electron energy offset. Note that it goes like (1+delta)^2, not (1+delta)^3 
       * because the bending radius also depends on (1+delta) 
       */
      dDelta = normalizedCriticalEnergy * sqr(1 + *dPoP) * yph;
      photonCount++;
      energyCount += yph;
      /* Change the total electron momentum */
      *dPoP -= dDelta;
      if (includeOpeningAngle) {
        /* Compute rms spread in electron angle = (rms photon angle)*dDelta */
        logyph = log10(yph);
        thetaRms = dDelta * pow(10, -2.418673276661232e-01 + logyph * (-4.472680955382907e-01 + logyph * (-4.535350424882360e-02 - logyph * 6.181818621278201e-03))) / Po;
        /* Compute change in electron angle due to photon angle */
        dtheta = thetaRms * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
        dphi = thetaRms * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
        if (SDDSphotons)
          logPhoton(dDelta * Po, x, xp - dtheta / dDelta, y, yp - dphi / dDelta, theta, thetaf, 1 / h0);
        /* rhoSign factor is for backward compatibility */
        xp += dtheta * rhoSign;
        yp += dphi * rhoSign;
      } else {
        if (SDDSphotons)
          logPhoton(dDelta * Po, x, xp - dtheta / dDelta, y, yp - dphi / dDelta, theta, thetaf, 1 / h0);
      }
    }
    f = (1 + *dPoP) / sqrt(sqr(1 + x * h0) + sqr(xp) + sqr(yp));
    *Qx = xp * f;
    *Qy = yp * f;
  }
}

long inversePoissonCDF(double mu, double C) {
  double sum, expMinusMu, term;
  long r, rMax;

  r = 0;
  if ((rMax = 50 * mu) < 10)
    rMax = 10;
  expMinusMu = exp(-mu);
  term = sum = expMinusMu;
  while (r <= rMax && C >= sum) {
    term *= mu / (++r);
    sum += term;
  }
  /* fprintf(stderr, "inversePoissonCDF: r=%ld for mu=%e, C=%e\n", r, mu, C); */
  return r;
}

/* Return randomly-chosen photon energy normalized to the critical energy */
double pickNormalizedPhotonEnergy(double RN) {
  long interpCode;
  double value;
  static double ksiTable[200] = {
    1.000000000000000e-07,
    1.103351074554523e-07,
    1.217383310646075e-07,
    1.343200559586096e-07,
    1.482020747927429e-07,
    1.635189113578160e-07,
    1.804187271717404e-07,
    1.990651067932036e-07,
    2.196385495378513e-07,
    2.423384085535426e-07,
    2.673842889192374e-07,
    2.950186137528527e-07,
    3.255088868939687e-07,
    3.591505332405233e-07,
    3.962690515521040e-07,
    4.372236992560780e-07,
    4.824109227537678e-07,
    5.322685173365927e-07,
    5.872789374272963e-07,
    6.479745823257918e-07,
    7.149429940622619e-07,
    7.888329490825732e-07,
    8.703595415651299e-07,
    9.603117573208774e-07,
    1.059560346180786e-06,
    1.169066741218290e-06,
    1.289890855361064e-06,
    1.423201921186706e-06,
    1.570290408531488e-06,
    1.732581081736759e-06,
    1.911644944229382e-06,
    2.109214732888093e-06,
    2.327202951554652e-06,
    2.567720983189001e-06,
    2.833097369770374e-06,
    3.125899929822278e-06,
    3.448963034857157e-06,
    3.805415579829938e-06,
    4.198708930670648e-06,
    4.632648449870741e-06,
    5.111434736502657e-06,
    5.639704554113462e-06,
    6.222573518736223e-06,
    6.865680963315421e-06,
    7.575252265320169e-06,
    8.358158724774669e-06,
    9.221982709850737e-06,
    1.017508139438384e-05,
    1.122668092062936e-05,
    1.238696398672945e-05,
    1.366716918178818e-05,
    1.507968136669955e-05,
    1.663817388115531e-05,
    1.835773664119261e-05,
    2.025502748788774e-05,
    2.234840008736815e-05,
    2.465811862080412e-05,
    2.720654509486478e-05,
    3.001836976776838e-05,
    3.312079173855694e-05,
    3.654384295607505e-05,
    4.032066206311618e-05,
    4.448784496925588e-05,
    4.908569920644159e-05,
    5.415873276357582e-05,
    5.975605436563109e-05,
    6.593190648243325e-05,
    7.274602260436053e-05,
    8.026436452046312e-05,
    8.855971070475770e-05,
    9.771244913354602e-05,
    1.078111117026042e-04,
    1.189534517802125e-04,
    1.312473286188584e-04,
    1.448118705606887e-04,
    1.597782996280689e-04,
    1.762914808166189e-04,
    1.945112638414439e-04,
    2.146141870658145e-04,
    2.367947478384840e-04,
    2.612676279365202e-04,
    2.882697280504828e-04,
    3.180626634504074e-04,
    3.509347182037930e-04,
    3.872040386295810e-04,
    4.272217166086854e-04,
    4.713754443547046e-04,
    5.200925166529275e-04,
    5.738444084509028e-04,
    6.331514454110419e-04,
    6.985881553542330e-04,
    7.707878748140519e-04,
    8.504493031356108e-04,
    9.383435740575402e-04,
    1.035322090321881e-03,
    1.142323583631410e-03,
    1.260383487096962e-03,
    1.390644637378946e-03,
    1.534368742954572e-03,
    1.692947192375836e-03,
    1.867914439234830e-03,
    2.060964191854564e-03,
    2.273966197872862e-03,
    2.508982774009835e-03,
    2.768287894108917e-03,
    3.054391670052556e-03,
    3.370064913211098e-03,
    3.718364390303022e-03,
    4.102660002531562e-03,
    4.526671789275104e-03,
    4.994505863796759e-03,
    5.510692966986859e-03,
    6.080227102658105e-03,
    6.708621448554406e-03,
    7.401960924834442e-03,
    8.166960997457158e-03,
    9.011022498794720e-03,
    9.942316075907171e-03,
    1.096985907697749e-02,
    1.210360516825099e-02,
    1.335452196639184e-02,
    1.473471854468308e-02,
    1.625755786686624e-02,
    1.793779326078131e-02,
    1.979167811637735e-02,
    2.183715833862031e-02,
    2.409403673367596e-02,
    2.658418068941497e-02,
    2.933167681381903e-02,
    3.236312132092089e-02,
    3.570786033002121e-02,
    3.939830569107346e-02,
    4.347015239952341e-02,
    4.796281661196050e-02,
    5.291978786613488e-02,
    5.838910396032759e-02,
    6.442366592223334e-02,
    7.108188831787103e-02,
    7.842822357081021e-02,
    8.653385973120035e-02,
    9.547720673026440e-02,
    1.053448316706813e-01,
    1.162322544203413e-01,
    1.282449697899047e-01,
    1.414991970199129e-01,
    1.561232236887442e-01,
    1.722586122799903e-01,
    1.900616973883389e-01,
    2.097047392306722e-01,
    2.313778527830923e-01,
    2.552908366685073e-01,
    2.816753658394816e-01,
    3.107867644741113e-01,
    3.429067722728065e-01,
    3.783463152734567e-01,
    4.174487166108054e-01,
    4.605924179038830e-01,
    5.081949415377639e-01,
    5.607170865377987e-01,
    6.186676294134220e-01,
    6.826074965088431e-01,
    7.531554325044472e-01,
    8.309943513916955e-01,
    9.168782178988778e-01,
    1.011638437307961e+00,
    1.116191954411967e+00,
    1.231550862322391e+00,
    1.358832474428039e+00,
    1.499269100004806e+00,
    1.654219599620752e+00,
    1.825183916868001e+00,
    2.013817817791925e+00,
    2.221947831729780e+00,
    2.451587713539253e+00,
    2.704960411634972e+00,
    2.984519634347505e+00,
    3.292972664221137e+00,
    3.633303772560254e+00,
    4.008807416353689e+00,
    4.423119788364888e+00,
    4.880253623874576e+00,
    5.384631444881934e+00,
    5.941135706944927e+00,
    6.555154946882635e+00,
    7.232636842499024e+00,
    7.980135322277263e+00,
    8.804886289535018e+00,
    9.714875109915180e+00,
    1.071891743371295e+01,
    1.182672581369469e+01,
    1.304902401296616e+01,
    1.439764568247248e+01,
    1.588565738231238e+01,
    1.752745256838863e+01,
    1.933892408641842e+01,
    2.133760842747432e+01,
    2.354287285156119e+01,
    2.597604764912193e+01,
    2.866068635656761e+01,
    3.162277660168377e+01,
  };
  static double FTable[200] = {
    0.000000000000000e+00,
    1.916076787477782e-04,
    3.896006996482199e-04,
    5.941918318862451e-04,
    8.056009324383097e-04,
    1.024055848381587e-03,
    1.249790750550654e-03,
    1.483048166648730e-03,
    1.724078746036354e-03,
    1.973142196708657e-03,
    2.230505581886648e-03,
    2.496445345396121e-03,
    2.771247236692068e-03,
    3.055207274452791e-03,
    3.348630028390361e-03,
    3.651830594751359e-03,
    3.965134731031564e-03,
    4.288879835176022e-03,
    4.623413241840414e-03,
    4.969094094184835e-03,
    5.326293748409966e-03,
    5.695396753910589e-03,
    6.076799201662367e-03,
    6.470910424341261e-03,
    6.878153743490802e-03,
    7.298967431515415e-03,
    7.733803160081558e-03,
    8.183127443537616e-03,
    8.647422816544402e-03,
    9.127188753348749e-03,
    9.622940276393589e-03,
    1.013520903749105e-02,
    1.066454503150837e-02,
    1.121151744173274e-02,
    1.177671348365064e-02,
    1.236073899369794e-02,
    1.296422080554008e-02,
    1.358780748228750e-02,
    1.423216848805338e-02,
    1.489799412460975e-02,
    1.558599872144761e-02,
    1.629692120583610e-02,
    1.703152471578071e-02,
    1.779059568966145e-02,
    1.857494805017956e-02,
    1.938542355142036e-02,
    2.022289197174982e-02,
    2.108824911944652e-02,
    2.198242222447622e-02,
    2.290636998738409e-02,
    2.386108352008151e-02,
    2.484758298036615e-02,
    2.586692442491193e-02,
    2.692019946744418e-02,
    2.800853716992024e-02,
    2.913309895920114e-02,
    3.029508724043934e-02,
    3.149574455091229e-02,
    3.273635664445640e-02,
    3.401824527268329e-02,
    3.534277891084329e-02,
    3.671137126806664e-02,
    3.812548585471972e-02,
    3.958662612641901e-02,
    4.109634873515033e-02,
    4.265626122542306e-02,
    4.426802842804741e-02,
    4.593335936539433e-02,
    4.765402350963371e-02,
    4.943184769110832e-02,
    5.126872354805775e-02,
    5.316659282366442e-02,
    5.512746484065590e-02,
    5.715341374017695e-02,
    5.924658623150298e-02,
    6.140918640941097e-02,
    6.364349310252398e-02,
    6.595185823803101e-02,
    6.833671466591608e-02,
    7.080056062143326e-02,
    7.334597653614160e-02,
    7.597562485484224e-02,
    7.869225776043576e-02,
    8.149870143512994e-02,
    8.439787179667740e-02,
    8.739277614110098e-02,
    9.048652052440445e-02,
    9.368229400251835e-02,
    9.698338259579789e-02,
    1.003931731738744e-01,
    1.039151601553213e-01,
    1.075529299247917e-01,
    1.113101721273651e-01,
    1.151906862423360e-01,
    1.191983871317649e-01,
    1.233372898347266e-01,
    1.276115170379167e-01,
    1.320253087906072e-01,
    1.365830262538971e-01,
    1.412891370418868e-01,
    1.461482173909104e-01,
    1.511649654280279e-01,
    1.563442022251273e-01,
    1.616908577721921e-01,
    1.672099659551390e-01,
    1.729066816736924e-01,
    1.787862779608226e-01,
    1.848541324986933e-01,
    1.911157129897795e-01,
    1.975765981791920e-01,
    2.042424693223850e-01,
    2.111190968366426e-01,
    2.182123129823302e-01,
    2.255280364093208e-01,
    2.330722555939911e-01,
    2.408510146807283e-01,
    2.488703695479055e-01,
    2.571364147677684e-01,
    2.656552557248533e-01,
    2.744329918358574e-01,
    2.834756510338721e-01,
    2.927892168723024e-01,
    3.023795848158047e-01,
    3.122525396990814e-01,
    3.224136624350130e-01,
    3.328683532505147e-01,
    3.436217660758924e-01,
    3.546787751835536e-01,
    3.660438466342482e-01,
    3.777210511874600e-01,
    3.897139689097260e-01,
    4.020256374046105e-01,
    4.146583795759362e-01,
    4.276137967068266e-01,
    4.408926345108801e-01,
    4.544946994027770e-01,
    4.684186411623745e-01,
    4.826619073829413e-01,
    4.972205662043195e-01,
    5.120891723276200e-01,
    5.272605134477985e-01,
    5.427255018050126e-01,
    5.584729557729631e-01,
    5.744894096219003e-01,
    5.907588309168045e-01,
    6.072624513895929e-01,
    6.239785205673239e-01,
    6.408820784452591e-01,
    6.579446823895981e-01,
    6.751342192611512e-01,
    6.924146905220633e-01,
    7.097460264751286e-01,
    7.270839321656986e-01,
    7.443798099419868e-01,
    7.615807315005799e-01,
    7.786294876113157e-01,
    7.954647879517510e-01,
    8.120215670459850e-01,
    8.282314411609915e-01,
    8.440233375507971e-01,
    8.593244148619452e-01,
    8.740611430503441e-01,
    8.881606055288961e-01,
    9.015520524341847e-01,
    9.141687939214495e-01,
    9.259502059677074e-01,
    9.368438193470870e-01,
    9.468075155760183e-01,
    9.558117659236037e-01,
    9.638415906288208e-01,
    9.708980284014210e-01,
    9.769991556010101e-01,
    9.821804314564566e-01,
    9.864941142754962e-01,
    9.900075473722704e-01,
    9.928006021230259e-01,
    9.949622418923878e-01,
    9.965864066620354e-01,
    9.977674409043544e-01,
    9.985957390441925e-01,
    9.991538996011060e-01,
    9.995138052403904e-01,
    9.997348433078885e-01,
    9.998634876235176e-01,
    9.999340435105117e-01,
    9.999702897374164e-01,
    9.999876125809349e-01,
    9.999952573365697e-01,
    9.999983471293699e-01,
    9.999994807411241e-01,
    9.999998545219742e-01,
    9.999999640793696e-01,
    9.999999922833868e-01,
    9.999999985785893e-01,
    9.999999997790499e-01,
    9.999999999715266e-01,
    9.999999999970198e-01,
    9.999999999997560e-01,
    9.999999999999872e-01,
    1.000000000000000e+00,
  };
  value = interp(ksiTable, FTable, 200, RN, 0, 2, &interpCode);
  if (!interpCode)
    return ksiTable[0];
  return value;
}

void addCorrectorRadiationKick(double **coord, int64_t np, ELEMENT_LIST *elem, long type, double Po, double *sigmaDelta2, long disableISR) {
  double F2;
  double kick=0, length=0;
  double isrCoef, radCoef, dp, p, beta0, beta1, deltaFactor;
  short isr, sr;
  int64_t i;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    gpu_addCorrectorRadiationKick(np, elem, type, Po, sigmaDelta2, disableISR);
#  ifdef GPU_VERIFY
    startCpuTimer();
    addCorrectorRadiationKick(coord, np, elem, type, Po, sigmaDelta2, disableISR);
    compareGpuCpu(np, "addCorrectorRadiationKick");
#  endif /* GPU_VERIFY */
    return;
  }
#endif /* HAVE_GPU */

  if (!np)
    return;

  isr = sr = 0;

  switch (type) {
  case T_HCOR:
    kick = ((HCOR *)elem->p_elem)->kick;
    if ((length = ((HCOR *)elem->p_elem)->length) == 0)
      length = ((HCOR *)elem->p_elem)->lEffRad;
    if (((HCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((HCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  case T_VCOR:
    kick = ((VCOR *)elem->p_elem)->kick;
    if ((length = ((VCOR *)elem->p_elem)->length) == 0)
      length = ((VCOR *)elem->p_elem)->lEffRad;
    if (((VCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((VCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  case T_HVCOR:
    kick = sqrt(sqr(((HVCOR *)elem->p_elem)->xkick) + sqr(((HVCOR *)elem->p_elem)->ykick));
    if ((length = ((HVCOR *)elem->p_elem)->length) == 0)
      length = ((HVCOR *)elem->p_elem)->lEffRad;
    if (((HVCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((HVCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  case T_EHCOR:
    kick = ((EHCOR *)elem->p_elem)->kick;
    if ((length = ((EHCOR *)elem->p_elem)->length) == 0)
      length = ((EHCOR *)elem->p_elem)->lEffRad;
    if (((EHCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((EHCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  case T_EVCOR:
    kick = ((EVCOR *)elem->p_elem)->kick;
    if ((length = ((EVCOR *)elem->p_elem)->length) == 0)
      length = ((EVCOR *)elem->p_elem)->lEffRad;
    if (((EVCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((EVCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  case T_EHVCOR:
    kick = sqrt(sqr(((EHVCOR *)elem->p_elem)->xkick) + sqr(((EHVCOR *)elem->p_elem)->ykick));
    if ((length = ((EHVCOR *)elem->p_elem)->length) == 0)
      length = ((EHVCOR *)elem->p_elem)->lEffRad;
    if (((EHVCOR *)elem->p_elem)->synchRad) {
      sr = 1;
      if (((EHVCOR *)elem->p_elem)->isr)
        isr = 1;
    }
    break;
  }
  if (sr == 0 || length == 0)
    return;
  if (disableISR)
    isr = 0;
  radCoef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);

  F2 = sqr(kick / length);
  for (i = 0; i < np; i++) {
    dp = coord[i][5];
    p = Po * (1 + dp);
    beta0 = p / sqrt(sqr(p) + 1);
    deltaFactor = sqr(1 + dp);
    dp -= radCoef * deltaFactor * F2 * length;
    if (isr)
      dp += isrCoef * deltaFactor * pow(F2, 0.75) * sqrt(length) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
    if (sigmaDelta2)
      *sigmaDelta2 += sqr(isrCoef * deltaFactor) * pow(F2, 1.5) * length;
    p = Po * (1 + dp);
    beta1 = p / sqrt(sqr(p) + 1);
    coord[i][5] = dp;
    coord[i][4] = beta1 * coord[i][4] / beta0;
  }
  if (sigmaDelta2)
    *sigmaDelta2 /= np;
}

void convolveArrays1(double *output, long n, double *a1, double *a2) {
  long ib, ib1;
  for (ib = 0; ib < n; ib++) {
    output[ib] = 0;
    for (ib1 = ib; ib1 < n; ib1++)
      output[ib] += a1[ib1] * a2[ib1 - ib];
  }
}

void setUpCsbendPhotonOutputFile(CSBEND *csbend, char *rootname, int64_t np) {
  TRACKING_CONTEXT tc;
#if USE_MPI
  SDDSphotons = NULL;
  return;
#endif
  if (!csbend->photonOutputFile) {
    SDDSphotons = NULL;
    return;
  }
  photonLowEnergyCutoff = csbend->photonLowEnergyCutoff;
  getTrackingContext(&tc);
  if (!csbend->photonFileActive) {
    char *filename;
    filename = compose_filename(csbend->photonOutputFile, rootname);
    csbend->SDDSphotons = tmalloc(sizeof(SDDS_DATASET));
    if (!SDDS_InitializeOutputElegant(csbend->SDDSphotons, SDDS_BINARY, 1, NULL, NULL, filename) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "Step", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "Particles", NULL, NULL, "Number of charged particles", NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "LowEnergyCutoff", NULL, "eV", "Minimum photon energy included in output", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, tc.elementName) ||
        0 > SDDS_DefineParameter(csbend->SDDSphotons, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) ||
        !SDDS_DefineSimpleColumn(csbend->SDDSphotons, "Ep", "eV", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(csbend->SDDSphotons, "x", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(csbend->SDDSphotons, "xp", "", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(csbend->SDDSphotons, "y", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(csbend->SDDSphotons, "yp", "", SDDS_FLOAT) ||
        !SDDS_WriteLayout(csbend->SDDSphotons)) {
      SDDS_SetError("Problem setting up photon output file for CSBEND");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(filename);
    csbend->photonFileActive = 1;
  }
  if (!SDDS_StartPage(csbend->SDDSphotons, 10000) ||
      !SDDS_SetParameters(csbend->SDDSphotons, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Particles", np, "Step", tc.step,
                          "LowEnergyCutoff", photonLowEnergyCutoff, "ElementName", tc.elementName, "ElementOccurence", tc.elementOccurrence, NULL)) {
    SDDS_SetError("Problem setting up photon output file for CSBEND");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  photonRows = 0;
  SDDSphotons = csbend->SDDSphotons;
}

void logPhoton(double Ep, double x, double xp, double y, double yp, double thetar, double thetaf, double rho) {
  double Xi, Zi, thetai, phii;
  double L, R;
  double XBar, thetaBar, phiBar, yBar;

  if ((Ep *= me_mev * 1e6) < photonLowEnergyCutoff)
    return;

  /* emission */
  thetai = thetar - atan(xp);
  phii = atan(yp);
  Xi = -rho * (1 - cos(thetar)) + x * cos(thetar);
  Zi = (x + rho) * sin(thetar);

  /* intersection with exit plane */
  L = (Zi * cos(thetaf) - (rho + Xi) * sin(thetaf)) / cos(thetaf - thetai);
  R = ((rho + Xi) * cos(thetai) + Zi * sin(thetai)) / cos(thetaf - thetai);
  XBar = R - rho;
  thetaBar = thetai - thetaf;
  phiBar = phii;
  yBar = y + L * tan(phii);

  if (!SDDS_SetRowValues(SDDSphotons, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, photonRows++,
                         0, (float)Ep,
                         1, (float)XBar,
                         2, (float)(-tan(thetaBar)),
                         3, (float)yBar,
                         4, (float)tan(phiBar),
                         -1))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  if (photonRows % 10000 == 0) {
    if (!SDDS_UpdatePage(SDDSphotons, FLUSH_TABLE))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
}

static CSBEND csbendWorking;
static ELEMENT_LIST  *eptrWorking;
static long optimizationEvaluations;
static double **optParticle = NULL;

double csbend_fse_adjustment_penalty(double *value, long *invalid) {
  if (!optParticle)
    optParticle = (double **)czarray_2d(sizeof(**optParticle), 1, totalPropertiesPerParticle);
  memset(optParticle[0], 0, totalPropertiesPerParticle * sizeof(**optParticle));

  csbendWorking.fseCorrectionValue = *value;
  optimizationEvaluations++;
  if (!track_through_csbend(optParticle, 1, &csbendWorking, 0, 1e3, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, eptrWorking)) {
    *invalid = 1;
    return 0.0;
  }
  *invalid = 0;
  return fabs(optParticle[0][1]);
}

static long FSEOptimizationCount = 0;

void csbend_update_fse_adjustment(CSBEND *csbend, ELEMENT_LIST *eptr) {
  double fseUser = 0, fse = 0, stepSize = 1e-3, lowerLimit = -1, upperLimit = 1, acc;
  short disable = 0;
  if (csbend->fseCorrection &&
      (csbend->edge_effects[csbend->e1Index] == 2 || csbend->edge_effects[csbend->e2Index] == 2 ||
       csbend->edge_effects[csbend->e1Index] == 4 || csbend->edge_effects[csbend->e2Index] == 4)) {
    if (!optParticle)
      optParticle = (double **)czarray_2d(sizeof(**optParticle), 1, totalPropertiesPerParticle);
    fseUser = csbend->fse;
    csbend->fse = 0;
    memcpy(&csbendWorking, csbend, sizeof(csbendWorking));
    csbendWorking.dx = csbendWorking.dy = csbendWorking.dz = csbendWorking.etilt = csbendWorking.tilt = 0;
    csbendWorking.isr = csbendWorking.synch_rad = csbendWorking.fseCorrectionPathError = 0;
    eptrWorking = eptr;
    optimizationEvaluations = 0;
    if (simplexMin(&acc, &fse, &stepSize, &lowerLimit, &upperLimit, &disable, 1,
                   fabs(1e-14 * csbend->angle), fabs(1e-16 * csbend->angle),
                   csbend_fse_adjustment_penalty, NULL, 1500, 3, 12, 3.0, 1.0, 0) < 0) {
      bombElegantVA("failed to find FSE to center trajectory for csbend. accuracy acheived was %le.", acc);
    }
    csbend->fse = fseUser;
    csbend->fseCorrectionValue = fse;
    csbend->fseCorrectionPathError = optParticle[0][4] - csbend->length;
    if (++FSEOptimizationCount < 1000) {
      printf("FSE optimized to %le (%le net) for CSBEND after %ld evaluations, giving error of %le and path-length %s of %le\n",
             fse, fse + fseUser, optimizationEvaluations, acc,
             csbend->fseCorrection == 1 ? "adjustment" : "error", csbend->fseCorrectionPathError);
      fflush(stdout);
    } else {
      if (FSEOptimizationCount == 1000) {
        printf("FSE optimized to %le (%le net) for CSBEND after %ld evaluations, giving error of %le and path-length %s of %le\n",
               fse, fse + fseUser, optimizationEvaluations, acc,
               csbend->fseCorrection == 1 ? "adjustment" : "error", csbend->fseCorrectionPathError);
        printf("Suppressing further FSE optimization messages\n");
        fflush(stdout);
      } else {
        if (FSEOptimizationCount % 1000 == 0) {
          printf("FSE optimization done %ld times in total\n", FSEOptimizationCount);
          fflush(stdout);
        }
      }
    }
  }
}

void applySimpleDipoleEdgeKick(double *xp, double *yp, double x, double y, double delta, double rho, double ea, double psi,
                               double kickLimit, long expanded) {
  /*  Apply edge effects using a symplectic method based on linear K. L. Brown matrix */
  double Qi[6];
  double dqx, dqy;

  Qi[0] = x;
  Qi[1] = *xp;
  Qi[2] = y;
  Qi[3] = *yp;
  Qi[4] = 0;
  Qi[5] = delta;
  convertToDipoleCanonicalCoordinates(Qi, expanded);

  dqx = tan(ea) / rho * x;
  if (kickLimit > 0 && fabs(dqx) > kickLimit) {
    dqx = SIGN(dqx) * kickLimit;
  }
  dqy = -tan(ea - psi / (1 + delta)) / rho * y;

  Qi[1] += dqx;
  Qi[3] += dqy;

  convertFromDipoleCanonicalCoordinates(Qi, expanded);
  *xp = Qi[1];
  *yp = Qi[3];
}


void curvedDipoleFringe
(
 double *Qf,            // Final coordinates (output)
 double *Qi,            // Initial coordinates
 double rho,            // Bending radius 
 long inFringe,       // -1 => entrance, else exit
 long edgeOrder,        // =1 for linear only
 double K1,             // body K1
 double edge,           // edge angle
 double *integrals,     // fringe integrals K0, K1, ..., K6
 unsigned short edgeFlip
) 
{
  double dx, dpx, dy, dpy, dtau;
  double tan_edge, sec_edge;
  double x0, px0, y0, py0, dp0;
  double mx, my, delX, focX, focY;
  double x1, px1, y1, py1;
  double intK0, intK1, intK2, intK3, intK4, intK5, intK6;
  int signFix;
  
  x0 = Qi[0];
  px0 = Qi[1];
  y0 = Qi[2];
  py0 = Qi[3];
  dp0 = Qi[5];
  dx = dpx = dy = dpy = dtau = 0;

  sec_edge = 1. / cos(edge);
  tan_edge = tan(edge);

  signFix = 1;
  if (edgeFlip) {
    // static int counter = 0;
    signFix = -1;
    // printf("signFix invoked %d times\n", ++counter);
  }
  
  intK0 = integrals[0]*signFix;
  intK1 = integrals[1];
  intK2 = integrals[2];
  intK3 = integrals[3]*signFix;
  intK4 = integrals[4]*signFix;
  intK5 = integrals[5];
  intK6 = integrals[6];

  if (inFringe!=-1 && inFringe!=1) 
    bombElegantVA("Error: inFringe is invalid (%ld) in curvedDipoleFringe---please report to developers.", 
                  inFringe);

  if (edgeOrder > 1) {
    /* higher-order */
    if (inFringe == -1.) {
      /* entrance */
      mx = (intK3 + intK4)/(1.0 + dp0);
      my = intK3/(1.0 + dp0);
      delX = intK0/(1.0 + dp0);
      focX = intK1 + tan_edge/rho;
      focY = intK2/(1.0 + dp0);

      if(fabs(mx) < 1.0e-5) {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - 0.5*mx*(1.0 - mx/3.0 + mx*mx/12.0));
	px1 = px0*exp(mx) + focX*(x0*(1.0 + mx*mx/6.0) + dp0*delX*(0.5 + mx*mx/24.0) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*0.5*mx*(1.0 + mx/3.0 + mx*mx/12.0)
	  + focX*x0*x0*mx*(0.5 - (mx - 0.5*mx*mx)/3.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(1.0 + 0.05*mx*mx)/6.0
	  + focX*delX*x0*0.5*(1.0 + mx*mx/12.0)/(1.0 + dp0)
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*(1.0 - mx*(0.75- mx*(0.35 - 0.125*mx)))/(6.0*(1.0 + dp0));
      }
      else {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - exp(-mx))/mx;
	px1 = px0*exp(mx) + focX*(x0*sinh(mx)/mx + dp0*delX*(cosh(mx) - 1.0)/(mx*mx) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*(exp(mx) - 1.0 - mx)/mx
	  + focX*x0*x0*0.25*((exp(-2.0*mx) - 1.0)/mx + 2.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(sinh(mx) - mx)/(mx*mx*mx)
	  + focX*delX*x0*(cosh(mx) - 1.0)/(mx*mx*(1.0 + dp0))
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*0.25*(2.0 - (3.0 + exp(-2.0*mx) - 4.0*exp(-mx))/mx)/(mx*mx*(1.0 + dp0));
      }
      if(fabs(my) < 1.0e-5) {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*(1.0 + my*my/6.0);
	dtau += ( -my*py0*y0 + 0.5*(focY + focX*(my + my*my*(2.0 + my)/3.0)*y0*y0) )/(1.0 + dp0);
      }
      else {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*sinh(my)/my;
	dtau += ( -my*py0*y0 + 0.5*(focY + 0.5*focX*((exp(2.0*my) - 1.0)/my - 2.0))*y0*y0 )/(1.0 + dp0);
      }

      double sect2_rho = sec_edge*sec_edge/(rho*(1.0 + dp0));
      double tant_rho = tan_edge/(rho*(1.0 + dp0));
      double x2, y2, px2, py2;

      x2  =  x1;
      y2  =  y1*exp(tan_edge*tant_rho*x1);
      px2 = px1 - tan_edge*tant_rho*y1*py1 - ( tan_edge*(tan_edge*tant_rho/rho - K1) - 0.5*intK6 )*x1*x1;
      py2 = py1*exp(-tan_edge*tant_rho*x1);
      dtau += -tan_edge*tant_rho*tant_rho*x1*x1*x1/3.0 - tant_rho*tant_rho*rho*x1*y1*py1;

      x1  =  x2 + 0.5*sect2_rho*y2*y2;
      y1  =  y2;
      px1 = px2 + ( 0.5*tant_rho/rho - 0.5*intK5 - K1*tan_edge + 0.25*intK1*sect2_rho )*y2*y2;
      py1 = py2 + ( tant_rho/rho - intK5 - 2.0*K1*tan_edge + 0.5*intK1*sect2_rho )*x2*y2 - sect2_rho*y2*px2; 
      dtau += ( 0.5*sect2_rho*(0.25*(2.0*tan_edge*K1 + intK5)*y2*y2 - px2) + (0.5*tant_rho/rho + 0.25*intK1*sect2_rho)*x2 )*y2*y2/(1.0 + dp0);

      x2  =  x1/(1.0 + 0.5*tan_edge*tant_rho*x1);
      y2  =  y1;
      px2 = px1*(1.0 + 0.5*tan_edge*tant_rho*x1)*(1.0 + 0.5*tan_edge*tant_rho*x1);
      py2 = py1;
      dtau += 0.5*tant_rho*tant_rho*rho*x1*x1*px1;

      dx  = x2 - x0;
      dpx = px2-px0;
      dy  = y2 - y0;
      dpy = py2-py0;
    } else if (inFringe == 1.) {
      /* exit */
      mx = (intK3 + intK4)/(1.0 + dp0);
      my = intK3/(1.0 + dp0);
      delX = intK0/(1.0 + dp0);
      focX = intK1 + tan_edge/rho;
      focY = intK2/(1.0 + dp0);

      if(fabs(mx) < 1.0e-5) {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - 0.5*mx*(1.0 - mx/3.0 + mx*mx/12.0));
	px1 = px0*exp(mx) + focX*(x0*(1.0 + mx*mx/6.0) + dp0*delX*(0.5 + mx*mx/24.0) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*0.5*mx*(1.0 + mx/3.0 + mx*mx/12.0)
	  + focX*x0*x0*mx*(0.5 - (mx - 0.5*mx*mx)/3.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(1.0 + 0.05*mx*mx)/6.0
	  + focX*delX*x0*0.5*(1.0 + mx*mx/12.0)/(1.0 + dp0)
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*(1.0 - mx*(0.75- mx*(0.35 - 0.125*mx)))/(6.0*(1.0 + dp0));
      }
      else {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - exp(-mx))/mx;
	px1 = px0*exp(mx) + focX*(x0*sinh(mx)/mx + dp0*delX*(cosh(mx) - 1.0)/(mx*mx) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*(exp(mx) - 1.0 - mx)/mx
	  + focX*x0*x0*0.25*((exp(-2.0*mx) - 1.0)/mx + 2.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(sinh(mx) - mx)/(mx*mx*mx)
	  + focX*delX*x0*(cosh(mx) - 1.0)/(mx*mx*(1.0 + dp0))
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*0.25*(2.0 - (3.0 + exp(-2.0*mx) - 4.0*exp(-mx))/mx)/(mx*mx*(1.0 + dp0));
      }
      if(fabs(my) < 1.0e-5) {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*(1.0 + my*my/6.0);
	dtau += ( -my*py0*y0 + 0.5*(focY + focX*(my + my*my*(2.0 + my)/3.0)*y0*y0) )/(1.0 + dp0);
      }
      else {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*sinh(my)/my;
	dtau += ( -my*py0*y0 + 0.5*(focY + 0.5*focX*((exp(2.0*my) - 1.0)/my - 2.0))*y0*y0 )/(1.0 + dp0);
      }

      double sect2_rho = sec_edge*sec_edge/(rho*(1.0 + dp0));
      double tant_rho = tan_edge/(rho*(1.0 + dp0));
      double x2, y2, px2, py2;

      x2  =  x1;
      y2  =  y1*exp(-tan_edge*tant_rho*x1);
      px2 = px1 + tan_edge*tant_rho*y1*py1 + ( tan_edge*(0.5*tan_edge*tant_rho/rho + K1) + 0.5*intK6 )*x1*x1;
      py2 = py1*exp(tan_edge*tant_rho*x1);
      dtau += tan_edge*tant_rho*tant_rho*x1*x1*x1/6.0 + tant_rho*tant_rho*rho*x1*y1*py1;

      x1  =  x2 - 0.5*sect2_rho*y2*y2;
      y1  =  y2;
      px1 = px2 + ( 0.5*tan_edge*tan_edge*tant_rho/rho - 0.5*intK5 - K1*tan_edge - 0.25*intK1*sect2_rho )*y2*y2;
      py1 = py2 + ( tan_edge*tan_edge*tant_rho/rho - intK5 - 2.0*K1*tan_edge - 0.5*intK1*sect2_rho )*x2*y2
	+ sect2_rho*y2*px2; 
      dtau += (-0.5*sect2_rho*(0.25*(2.0*tan_edge*K1 + intK5)*y2*y2 - px2)
	       + (0.5*tan_edge*tan_edge*tant_rho/rho - 0.25*intK1*sect2_rho)*x2 )*y2*y2/(1.0 + dp0);

      x2  =  x1/(1.0 - 0.5*tan_edge*tant_rho*x1);
      y2  =  y1;
      px2 = px1*(1.0 - 0.5*tan_edge*tant_rho*x1)*(1.0 + 0.5*tan_edge*tant_rho*x1);
      py2 = py1;
      dtau += -0.5*tant_rho*tant_rho*rho*x1*x1*px1;

      dx  = x2 - x0;
      dpx = px2-px0;
      dy  = y2 - y0;
      dpy = py2-py0;
    }

  } else {
    /* linear terms in transverse coordinates only */

    if (inFringe == -1.) {
      /* entrance */

      mx = (intK3 + intK4)/(1.0 + dp0);
      my = intK3/(1.0 + dp0);
      delX = intK0/(1.0 + dp0);
      focX = intK1 + tan_edge/rho;
      focY = intK2/(1.0 + dp0);

      if(fabs(mx) < 1.0e-5) {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - 0.5*mx*(1.0 - mx/3.0 + mx*mx/12.0));
	px1 = px0*exp(mx) + focX*(x0*(1.0 + mx*mx/6.0) + dp0*delX*(0.5 + mx*mx/24.0) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*0.5*mx*(1.0 + mx/3.0 + mx*mx/12.0)
	  + focX*x0*x0*mx*(0.5 - (mx - 0.5*mx*mx)/3.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(1.0 + 0.05*mx*mx)/6.0
	  + focX*delX*x0*0.5*(1.0 + mx*mx/12.0)/(1.0 + dp0)
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*(1.0 - mx*(0.75- mx*(0.35 - 0.125*mx)))/(6.0*(1.0 + dp0));
      }
      else {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - exp(-mx))/mx;
	px1 = px0*exp(mx) + focX*(x0*sinh(mx)/mx + dp0*delX*(cosh(mx) - 1.0)/(mx*mx) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*(exp(mx) - 1.0 - mx)/mx
	  + focX*x0*x0*0.25*((exp(-2.0*mx) - 1.0)/mx + 2.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(sinh(mx) - mx)/(mx*mx*mx)
	  + focX*delX*x0*(cosh(mx) - 1.0)/(mx*mx*(1.0 + dp0))
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*0.25*(2.0 - (3.0 + exp(-2.0*mx) - 4.0*exp(-mx))/mx)/(mx*mx*(1.0 + dp0));
      }
      if(fabs(my) < 1.0e-5) {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*(1.0 + my*my/6.0);
	dtau += ( -my*py0*y0 + 0.5*(focY + focX*(my + my*my*(2.0 + my)/3.0)*y0*y0) )/(1.0 + dp0);
      }
      else {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*sinh(my)/my;
	dtau += ( -my*py0*y0 + 0.5*(focY + 0.5*focX*((exp(2.0*my) - 1.0)/my - 2.0))*y0*y0 )/(1.0 + dp0);
      }
      dx  = x1 - x0;
      dpx = px1-px0;
      dy  = y1 - y0;
      dpy = py1-py0;
    } else if (inFringe == 1.) {
      /* exit */

      mx = (intK3 + intK4)/(1.0 + dp0);
      my = intK3/(1.0 + dp0);
      delX = intK0/(1.0 + dp0);
      focX = intK1 + tan_edge/rho;
      focY = intK2/(1.0 + dp0);

      if(fabs(mx) < 1.0e-5) {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - 0.5*mx*(1.0 - mx/3.0 + mx*mx/12.0));
	px1 = px0*exp(mx) + focX*(x0*(1.0 + mx*mx/6.0) + dp0*delX*(0.5 + mx*mx/24.0) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*0.5*mx*(1.0 + mx/3.0 + mx*mx/12.0)
	  + focX*x0*x0*mx*(0.5 - (mx - 0.5*mx*mx)/3.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(1.0 + 0.05*mx*mx)/6.0
	  + focX*delX*x0*0.5*(1.0 + mx*mx/12.0)/(1.0 + dp0)
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*(1.0 - mx*(0.75- mx*(0.35 - 0.125*mx)))/(6.0*(1.0 + dp0));
      }
      else {
	x1  = x0*exp(-mx) + dp0*delX*(1.0 - exp(-mx))/mx;
	px1 = px0*exp(mx) + focX*(x0*sinh(mx)/mx + dp0*delX*(cosh(mx) - 1.0)/(mx*mx) );
	dtau = px0*(mx*x0 + delX)/(1.0 + dp0) + delX*px0*(exp(mx) - 1.0 - mx)/mx
	  + focX*x0*x0*0.25*((exp(-2.0*mx) - 1.0)/mx + 2.0)/(1.0 + dp0)
	  + dp0*focX*delX*(delX + mx*x0/(1.0 + dp0))*(sinh(mx) - mx)/(mx*mx*mx)
	  + focX*delX*x0*(cosh(mx) - 1.0)/(mx*mx*(1.0 + dp0))
	  + dp0*focX*delX*(2.0*mx*x0 - dp0*delX)*0.25*(2.0 - (3.0 + exp(-2.0*mx) - 4.0*exp(-mx))/mx)/(mx*mx*(1.0 + dp0));
      }
      if(fabs(my) < 1.0e-5) {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*(1.0 + my*my/6.0);
	dtau += ( -my*py0*y0 + 0.5*(focY + focX*(my + my*my*(2.0 + my)/3.0)*y0*y0) )/(1.0 + dp0);
      }
      else {
	y1  = y0*exp(my);
	py1 = py0*exp(-my) + y0*(focY - focX)*sinh(my)/my;
	dtau += ( -my*py0*y0 + 0.5*(focY + 0.5*focX*((exp(2.0*my) - 1.0)/my - 2.0))*y0*y0 )/(1.0 + dp0);
      }
      dx  = x1 - x0;
      dpx = px1-px0;
      dy  = y1 - y0;
      dpy = py1-py0;
    }
  }

  Qf[0] = x0 + dx;
  Qf[1] = px0 + dpx;
  Qf[2] = y0 + dy;
  Qf[3] = py0 + dpy;
  Qf[4] = -dtau;
  Qf[5] = Qi[5];
}
