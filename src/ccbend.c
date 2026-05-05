/************************************************************************* \
 * Copyright (c) 2018 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2018 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* routine: crbend()
 * purpose: tracking through canonical bend in cartesian coordinates
 *
 * Michael Borland, 2018
 */
#include "mdb.h"
#include "track.h"
#include "multipole.h"

static CCBEND ccbendCopy;
static ELEMENT_LIST *eptrCopy = NULL;
static double PoCopy, lastX, lastXp, lastRho;
/* These variables are used for optimization of the FSE and x offset */
static double xMin, xFinal, xAve, xMax, xError, xpInitial, xInitial, xpError;
#define OPTIMIZE_X 0x01UL
#define OPTIMIZE_XP 0x02UL
static unsigned long optimizationFlags = 0;
static long edgeMultActive[2];
#ifdef DEBUG
static short logHamiltonian = 0;
static FILE *fpHam = NULL;
#endif

void switchRbendPlane(double **particle, long n_part, double alpha, double Po);
void verticalRbendFringe(double **particle, long n_part, double alpha, double rho0, double K1, double K2, double gK,
                         double *fringeIntKn, short angleSign, short isExit, short fringeModel, short order);
double ccbend_trajectory_error(double *value, long *invalid);

long track_through_ccbend(
                          double **particle, /* initial/final phase-space coordinates */
                          long n_part,       /* number of particles */
                          ELEMENT_LIST *eptr,
                          CCBEND *ccbend,
                          double Po,
                          double **accepted,
                          double z_start,
                          double *sigmaDelta2, /* use for accumulation of energy spread for radiation matrix computation */
                          char *rootname,
                          MAXAMP *maxamp,
                          APCONTOUR *apContour,
                          APERTURE_DATA *apFileData,
                          /* If iPart non-negative, we do one step. The caller is responsible
                           * for handling the coordinates appropriately outside this routine.
                           * The element must have been previously optimized to determine FSE and X offsets.
                           */
                          long iPart,
                          /* If iFinalSlice is positive, we terminate integration inside the magnet. The caller is responsible
                           * for handling the coordinates appropriately outside this routine.
                           * The element must have been previously optimized to determine FSE and X offsets.
                           */
                          long iFinalSlice) {
  double KnL[9];
  long iTerm, nTerms;
  double dx, dy, dz; /* offsets of the multipole center */
  long nSlices, integ_order;
  long i_part, i_top;
  double *coef;
  double fse, etilt, epitch, eyaw, tilt, rad_coef, isr_coef, dzLoss = 0;
  double rho0, arcLength, length, angle, yaw, angleSign;
  MULTIPOLE_DATA *multData = NULL, *edge1MultData = NULL, *edge2MultData = NULL;
  long freeMultData = 0;
  MULT_APERTURE_DATA apertureData;
  double referenceKnL = 0;
  double gK[2];
  double fringeInt1[N_CCBEND_FRINGE_INT], fringeInt2[N_CCBEND_FRINGE_INT];
  double lastRho1;
  GLOBAL_BEAM_SUMS *beamSums = NULL;
  short disableSums = 1;
  TRACKING_CONTEXT context;
  getTrackingContext(&context);

  if (!particle)
    bombTracking("particle array is null (track_through_ccbend)");

  if (iPart >= 0 && ccbend->optimized != 1)
    bombTracking("Programming error: one-step mode invoked for unoptimized CCBEND.");
  if (iFinalSlice > 0 && iPart >= 0)
    bombTracking("Programming error: partial integration mode and one-step mode invoked together for CCBEND.");
  if (iFinalSlice > 0 && ccbend->optimized != 1 && (ccbend->optimizeFse || ccbend->optimizeDx))
    bombTracking("Programming error: partial integration mode invoked for unoptimized CCBEND.");
  if (iFinalSlice >= ccbend->nSlices)
    iFinalSlice = 0;

  if (N_CCBEND_FRINGE_INT != 8)
    bombTracking("Coding error detected: number of fringe integrals for CCBEND is different than expected.");
  if (ccbend->fringeModel < -1 || ccbend->fringeModel > 1)
    bombTracking("FRINGEMODEL parameter of CCBEND must be -1, 0, or 1.");

  if (!ccbend->edgeFlip) {
    memcpy(fringeInt1, ccbend->fringeInt1, N_CCBEND_FRINGE_INT * sizeof(fringeInt1[0]));
    memcpy(fringeInt2, ccbend->fringeInt2, N_CCBEND_FRINGE_INT * sizeof(fringeInt2[0]));
  } else {
    memcpy(fringeInt1, ccbend->fringeInt2, N_CCBEND_FRINGE_INT * sizeof(fringeInt1[0]));
    memcpy(fringeInt2, ccbend->fringeInt1, N_CCBEND_FRINGE_INT * sizeof(fringeInt2[0]));
    // Per R. Lindberg, these integrals change sign from entrance to exit orientation
    fringeInt1[0] *= -1;
    fringeInt1[3] *= -1;
    fringeInt1[5] *= -1;
    fringeInt2[0] *= -1;
    fringeInt2[3] *= -1;
    fringeInt2[5] *= -1;
  }

  ccbend->centroidsRequested = (ccbend->centroidOutputFile && strlen(ccbend->centroidOutputFile));
  if (ccbend->centroidsRequested && !ccbend->SDDScen
#if USE_MPI
      && myid==1 
#endif    
      ) {
    ccbend->SDDScen = tmalloc(sizeof(SDDS_DATASET));
    char *filename;
    filename = compose_filename(ccbend->centroidOutputFile, context.rootname);
    if (!SDDS_InitializeOutputElegant(ccbend->SDDScen, SDDS_BINARY, 1, NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(ccbend->SDDScen, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(ccbend->SDDScen, "Step", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(ccbend->SDDScen, "Pass", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(ccbend->SDDScen, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(ccbend->SDDScen, "ElementName", NULL, SDDS_STRING) ||
          !SDDS_DefineSimpleParameter(ccbend->SDDScen, "ElementOccurence", NULL, SDDS_LONG) ||
          SDDS_DefineColumn(ccbend->SDDScen, "Z", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "Slice", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "CX", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "theta", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "CY", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "phi", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "Cdelta", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(ccbend->SDDScen, "Particles", NULL, "", NULL, NULL, SDDS_LONG, 0) < 0 ||
          !SDDS_WriteLayout(ccbend->SDDScen)) {
        SDDS_SetError("Problem setting up centroid output file for CCBEND");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(filename);
  }

  if ((ccbend->optimizeFse || ccbend->optimizeDx) && ccbend->optimized != -1 && ccbend->angle != 0) {
    if (ccbend->optimized == 0 ||
        ccbend->length != ccbend->referenceData[0] ||
        ccbend->angle != ccbend->referenceData[1] ||
        ccbend->K1 != ccbend->referenceData[2] ||
        ccbend->K2 != ccbend->referenceData[3] ||
        ccbend->yaw != ccbend->referenceData[4] ||
        ccbend->tilt != ccbend->referenceData[5]) {
      double acc;
      double startValue[2], stepSize[2], lowerLimit[2], upperLimit[2];
      short disable[2] = {0, 0};
      PoCopy = lastRho1 = lastX = lastXp =
        xMin = xFinal = xAve = xMax = xError = xpInitial = xInitial = xpError = 0;
      if (ccbend->verbose > 5) {
        long ifr;
        printf("Fringe integrals used for %s:\n", eptr->name);
        for (ifr = 0; ifr < N_CCBEND_FRINGE_INT; ifr++)
          printf("I[%ld] = %e, %e\n", ifr, fringeInt1[ifr], fringeInt2[ifr]);
      }
      /*
        printf("Reoptimizing CCBEND due to changes: (optimized=%hd)\n", ccbend->optimized);
        printf("delta L: %le\n", ccbend->length-ccbend->referenceData[0]);
        printf("delta ANGLE: %le\n", ccbend->angle-ccbend->referenceData[1]);
        printf("delta K1: %le\n", ccbend->K1-ccbend->referenceData[2]);
        printf("delta K2: %le\n", ccbend->K2-ccbend->referenceData[3]);
        fflush(stdout);
      */
      if (iPart >= 0)
        bombTracking("Programming error: oneStep mode is incompatible with optmization for CCBEND.");
      optimizationFlags = OPTIMIZE_X | OPTIMIZE_XP;
      if (ccbend->optimized) {
        startValue[0] = ccbend->fseOffset;
        startValue[1] = ccbend->dxOffset;
        if (ccbend->optimizeFseOnce) {
          disable[0] = 1;
          optimizationFlags &= ~OPTIMIZE_XP;
        }
        if (ccbend->optimizeDxOnce) {
          disable[1] = 1;
          optimizationFlags &= ~OPTIMIZE_X;
        }
      } else {
        startValue[0] = startValue[1] = 0;
        if (!ccbend->optimizeFse) {
          disable[0] = 1;
          optimizationFlags &= ~OPTIMIZE_XP;
        }
        if (!ccbend->optimizeDx) {
          disable[1] = 1;
          optimizationFlags &= ~OPTIMIZE_X;
        }
      }
      if (!disable[0] || !disable[1]) {
        double **particle0;
        ccbend->optimized = -1; /* flag to indicate calls to track_through_ccbend will be for FSE optimization */
        memcpy(&ccbendCopy, ccbend, sizeof(ccbendCopy));
        if (ccbend->length < 0) {
          /* For backtracking. This seems to help improve results. Otherwise, the offsets are different, which doesn't
           * make sense. */
          ccbendCopy.length = fabs(ccbend->length);
          ccbendCopy.angle = -ccbend->angle;
          ccbendCopy.yaw = -ccbend->yaw;
        }
        eptrCopy = eptr;
        ccbendCopy.fse = ccbendCopy.fseDipole = ccbendCopy.fseQuadrupole = ccbendCopy.dx = ccbendCopy.dy = ccbendCopy.dz =
          ccbendCopy.ePitch = ccbendCopy.eYaw = ccbendCopy.eTilt =
          ccbendCopy.isr = ccbendCopy.synch_rad = ccbendCopy.isr1Particle =
          ccbendCopy.KnDelta = ccbendCopy.xKick = 0;
        memset(&ccbendCopy.referenceTrajectory[0], 0, 5 * sizeof(ccbendCopy.referenceTrajectory[0]));
        PoCopy = Po;
        stepSize[0] = 1e-3; /* FSE */
        stepSize[1] = 1e-4; /* X */
        lowerLimit[0] = lowerLimit[1] = -1;
        upperLimit[0] = upperLimit[1] = 1;
        if (simplexMin(&acc, startValue, stepSize, lowerLimit, upperLimit, disable, 2,
                       fabs(1e-15 * ccbend->length), fabs(1e-16 * ccbend->length),
                       ccbend_trajectory_error, NULL, 1500, 3, 12, 3.0, 1.0, 0) < 0) {
          bombElegantVA("failed to find FSE and x offset to center trajectory for ccbend. accuracy acheived was %le.", acc);
        }
        ccbend->fseOffset = startValue[0];
        ccbend->dxOffset = startValue[1];
        ccbend->xAdjust = xFinal;
        ccbend->KnDelta = ccbendCopy.KnDelta;
        ccbend->referenceData[0] = ccbend->length;
        ccbend->referenceData[1] = ccbend->angle;
        ccbend->referenceData[2] = ccbend->K1;
        ccbend->referenceData[3] = ccbend->K2;
        ccbend->referenceData[4] = ccbend->yaw;
        ccbend->referenceData[5] = ccbend->tilt;

        ccbend->optimized = 1;
        particle0 = (double **)czarray_2d(sizeof(**particle0), 1, totalPropertiesPerParticle);
        memset(particle0[0], 0, totalPropertiesPerParticle * sizeof(**particle));
        track_through_ccbend(particle0, 1, eptr, ccbend, Po, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, 0);
        for (int ii = 0; ii < 4; ii++)
          ccbend->referenceTrajectory[ii] = particle0[0][ii];
        ccbend->referenceTrajectory[4] = particle0[0][4] - ccbend->length;
        free_czarray_2d((void **)particle0, 1, totalPropertiesPerParticle);
        if (ccbend->verbose) {
          printf("CCBEND %s#%ld optimized: FSE=%21.15le, dx=%21.15le, accuracy=%21.15le\n",
                 eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, ccbend->fseOffset, ccbend->dxOffset, acc);
          printf("length = %21.15le, angle = %21.15le, K1 = %21.15le\nK2 = %21.15le, yaw = %21.15le\n",
                 ccbend->length, ccbend->angle, ccbend->K1, ccbend->K2, ccbend->yaw);
          if (ccbend->referenceCorrection & 1)
            printf("lengthCorrection = %21.15le\n", ccbend->referenceTrajectory[4]);
          if (ccbend->referenceCorrection & 2)
            printf("trajectory corrections: %21.15le, %21.15le, %21.15le, %21.15le\n",
                   ccbend->referenceTrajectory[0], ccbend->referenceTrajectory[1], ccbend->referenceTrajectory[2],
                   ccbend->referenceTrajectory[3]);
          printf("xMin = %21.15le, xMax = %21.15le, xAve = %21.15le, xInitial = %21.15le, xFinal = %21.15le, xpError = %21.15le\n",
                 xMin, xMax, xAve, xInitial, xFinal, xpError);
          fflush(stdout);
        }
      }
    }
  }

#ifdef DEBUG
  if (ccbend->optimized == 1) {
    char filename[1000];
    logHamiltonian = 1;
    snprintf(filename, 1000, "ccbend-%s.sdds", eptr->name);
    fpHam = fopen(filename, "w");
    fprintf(fpHam, "SDDS1\n&column name=z type=double units=m &end\n");
    fprintf(fpHam, "&column name=x type=double units=m &end\n");
    fprintf(fpHam, "&column name=y type=double units=m &end\n");
    fprintf(fpHam, "&column name=qx type=double &end\n");
    fprintf(fpHam, "&column name=qy type=double &end\n");
    fprintf(fpHam, "&column name=dH type=double &end\n");
    fprintf(fpHam, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  rad_coef = isr_coef = 0;

  nSlices = ccbend->nSlices;
  arcLength = ccbend->length;
  if (ccbend->optimized != 0)
    fse = ccbend->fse + ccbend->fseOffset;
  else
    fse = ccbend->fse;
  // NK: why not = {0}; initialization?
  for (iTerm = 0; iTerm < 9; iTerm++)
    KnL[iTerm] = 0;
  length = rho0 = 0; /* prevent compiler warnings */
  if ((angle = ccbend->angle) != 0) {
    rho0 = arcLength / angle;
    length = 2 * rho0 * sin(angle / 2);
    KnL[0] = (1 + fse + ccbend->fseDipole) / rho0 * length - ccbend->xKick;
  } else {
    /* TODO: Use KQUAD as substitute */
    bombTracking("Can't have zero ANGLE for CCBEND.");
  }
  KnL[1] = (1 + fse + ccbend->fseQuadrupole) * ccbend->K1 * length / (1 - ccbend->KnDelta);
  KnL[2] = (1 + fse) * ccbend->K2 * length / (1 - ccbend->KnDelta);
  KnL[3] = (1 + fse) * ccbend->K3 * length / (1 - ccbend->KnDelta);
  KnL[4] = (1 + fse) * ccbend->K4 * length / (1 - ccbend->KnDelta);
  KnL[5] = (1 + fse) * ccbend->K5 * length / (1 - ccbend->KnDelta);
  KnL[6] = (1 + fse) * ccbend->K6 * length / (1 - ccbend->KnDelta);
  KnL[7] = (1 + fse) * ccbend->K7 * length / (1 - ccbend->KnDelta);
  KnL[8] = (1 + fse) * ccbend->K8 * length / (1 - ccbend->KnDelta);
  if (angle < 0) {
    angleSign = -1;
    for (iTerm = 0; iTerm < 9; iTerm += 2)
      KnL[iTerm] *= -1;
    angle = -angle;
    rho0 = -rho0;
    yaw = -ccbend->yaw * (ccbend->edgeFlip ? -1 : 1);
    ccbend->extraTilt = PI;
  } else {
    angleSign = 1;
    ccbend->extraTilt = 0;
    yaw = ccbend->yaw * (ccbend->edgeFlip ? -1 : 1);
  }
  if (ccbend->systematic_multipoles || ccbend->edge_multipoles || ccbend->random_multipoles ||
      ccbend->edge1_multipoles || ccbend->edge2_multipoles) {
    /* Note that with referenceOrder=0, referenceKnL will always be positive if angle is nonzero */
    if (ccbend->referenceOrder == 0 && (referenceKnL = KnL[0]) == 0)
      bombElegant("REFERENCE_ORDER=0 but CCBEND ANGLE is zero", NULL);
    if (ccbend->referenceOrder == 1 && (referenceKnL = KnL[1]) == 0)
      bombElegant("REFERENCE_ORDER=1 but CCBEND K1 is zero", NULL);
    if (ccbend->referenceOrder == 2 && (referenceKnL = KnL[2]) == 0)
      bombElegant("REFERENCE_ORDER=2 but CCBEND K2 is zero", NULL);
    if (ccbend->referenceOrder < 0 || ccbend->referenceOrder > 2)
      bombElegant("REFERENCE_ORDER must be 0, 1, or 2 for CCBEND", NULL);
  }
  if (ccbend->edgeFlip == 0) {
    gK[0] = 2 * ccbend->fint1 * ccbend->hgap * (length < 0 ? -1 : 1);
    gK[1] = 2 * ccbend->fint2 * ccbend->hgap * (length < 0 ? -1 : 1);
  } else {
    gK[1] = 2 * ccbend->fint1 * ccbend->hgap * (length < 0 ? -1 : 1);
    gK[0] = 2 * ccbend->fint2 * ccbend->hgap * (length < 0 ? -1 : 1);
  }

  integ_order = ccbend->integration_order;
  if (ccbend->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
  if (!ccbend->isr || (ccbend->isr1Particle == 0 && n_part == 1))
    /* minus sign indicates we accumulate into sigmadelta^2 only, don't perturb particles */
    isr_coef *= -1;
  if (ccbend->length < 1e-6 && (ccbend->isr || ccbend->synch_rad)) {
    rad_coef = isr_coef = 0; /* avoid unphysical results */
  }
  if (!ccbend->multipolesInitialized) {
    /* read the data files for the error multipoles */
    readErrorMultipoleData(&(ccbend->systematicMultipoleData), ccbend->systematic_multipoles, 0);
    if ((ccbend->edge1_multipoles && !ccbend->edgeFlip) || (ccbend->edge2_multipoles && ccbend->edgeFlip)) {
      readErrorMultipoleData(&(ccbend->edge1MultipoleData),
                             ccbend->edgeFlip ? ccbend->edge2_multipoles : ccbend->edge1_multipoles, 0);
      if (ccbend->verbose && (ccbend->edgeFlip ? ccbend->edge2_multipoles : ccbend->edge1_multipoles) && eptr)
        printf("Using file %s for edge 1 of %s#%ld\n",
               ccbend->edgeFlip ? ccbend->edge2_multipoles : ccbend->edge1_multipoles,
               eptr->name, eptr->occurence);
    } else {
      readErrorMultipoleData(&(ccbend->edge1MultipoleData), ccbend->edge_multipoles, 0);
      if (ccbend->verbose && ccbend->edge_multipoles && eptr)
        printf("Using file %s for edge 1 of %s#%ld\n",
               ccbend->edge_multipoles, eptr->name, eptr->occurence);
    }
    if ((ccbend->edge2_multipoles && !ccbend->edgeFlip) || (ccbend->edge1_multipoles && ccbend->edgeFlip)) {
      readErrorMultipoleData(&(ccbend->edge2MultipoleData),
                             ccbend->edgeFlip ? ccbend->edge1_multipoles : ccbend->edge2_multipoles, 0);
      if (ccbend->verbose && (ccbend->edgeFlip ? ccbend->edge1_multipoles : ccbend->edge2_multipoles) && eptr)
        printf("Using file %s for edge 2 of %s#%ld\n",
               ccbend->edgeFlip ? ccbend->edge1_multipoles : ccbend->edge2_multipoles,
               eptr->name, eptr->occurence);
    } else {
      readErrorMultipoleData(&(ccbend->edge2MultipoleData), ccbend->edge_multipoles, 0);
      if (ccbend->verbose && ccbend->edge_multipoles && eptr)
        printf("Using file %s for edge 2 of %s#%ld\n",
               ccbend->edge_multipoles, eptr->name, eptr->occurence);
    }
    if (ccbend->optimized != -1)
      readErrorMultipoleData(&(ccbend->randomMultipoleData), ccbend->random_multipoles, 0);
    ccbend->multipolesInitialized = 1;
  }
  if (!ccbend->totalMultipolesComputed) {
    computeTotalErrorMultipoleFields(&(ccbend->totalMultipoleData),
                                     &(ccbend->systematicMultipoleData), ccbend->systematicMultipoleFactor,
                                     &(ccbend->edge1MultipoleData), &(ccbend->edge2MultipoleData),
                                     &(ccbend->randomMultipoleData), ccbend->randomMultipoleFactor,
                                     NULL, 0.0,
                                     referenceKnL,
                                     ccbend->referenceOrder, 0,
                                     ccbend->minMultipoleOrder, ccbend->maxMultipoleOrder);
    if (angleSign < 0) {
      long i;
      for (i = 0; i < ccbend->systematicMultipoleData.orders; i++) {
        if (ccbend->totalMultipoleData.order[i] % 2) {
          ccbend->totalMultipoleData.KnL[i] *= -1;
          ccbend->totalMultipoleData.JnL[i] *= -1;
        }
      }
      for (i = 0; i < ccbend->edge1MultipoleData.orders; i++) {
        if (ccbend->totalMultipoleData.order[i] % 2) {
          ccbend->edge1MultipoleData.KnL[i] *= -1;
          ccbend->edge1MultipoleData.JnL[i] *= -1;
        }
      }
      for (i = 0; i < ccbend->edge2MultipoleData.orders; i++) {
        if (ccbend->totalMultipoleData.order[i] % 2) {
          ccbend->edge2MultipoleData.KnL[i] *= -1;
          ccbend->edge2MultipoleData.JnL[i] *= -1;
        }
      }
    }
    ccbend->totalMultipolesComputed = 1;
  }
  multData = &(ccbend->totalMultipoleData);
  edge1MultData = &(ccbend->edge1MultipoleData);
  edge2MultData = &(ccbend->edge2MultipoleData);

  if (multData && !multData->initialized)
    multData = NULL;

  if (nSlices <= 0)
    bombTracking("nSlices<=0 in track_ccbend()");
  if (integ_order != 2 && integ_order != 4 && integ_order != 6)
    bombTracking("multipole integration_order must be 2, 4, or 6");

  if (!(coef = expansion_coefficients(0)))
    bombTracking("expansion_coefficients(0) returned NULL pointer (track_through_ccbend)");
  if (!(coef = expansion_coefficients(1)))
    bombTracking("expansion_coefficients(1) returned NULL pointer (track_through_ccbend)");
  if (!(coef = expansion_coefficients(2)))
    bombTracking("expansion_coefficients(2) returned NULL pointer (track_through_ccbend)");

  tilt = ccbend->tilt + ccbend->extraTilt;
  etilt = ccbend->eTilt;
  eyaw = ccbend->eYaw;
  epitch = ccbend->ePitch;
  dx = ccbend->dx;
  dy = ccbend->dy;
  dz = ccbend->dz * (ccbend->edgeFlip ? -1 : 1);

  if (tilt) {
    /* this is needed because the DX and DY offsets will be applied after the particles are
     * tilted into the magnet reference frame
     */
    dx = ccbend->dx * cos(tilt) + ccbend->dy * sin(tilt);
    dy = -ccbend->dx * sin(tilt) + ccbend->dy * cos(tilt);
  }

  setupMultApertureData(&apertureData, -tilt, apContour, maxamp, apFileData, NULL, z_start + length / 2, eptr);

  if (iPart <= 0) {
    /*
      printf("input before adjustments: %16.10le %16.10le %16.10le %16.10le %16.10le %16.10le\n",
      particle[0][0], particle[0][1], particle[0][2],
      particle[0][3], particle[0][4], particle[0][5]);
    */
    /* We perform the tilt first and separately for CCBEND because of the R-bend plane switch */
    if (tilt)
      rotateBeamCoordinatesForMisalignment(particle, n_part, tilt);
    /* save initial x, x' value of the possible reference particle for optimization of FSE and DX */
    xInitial = particle[0][0];
    xpInitial = particle[0][1];
    switchRbendPlane(particle, n_part, angle / 2 - yaw, Po);
    if (dx || dy || dz || eyaw || epitch)
      offsetParticlesForMisalignment(ccbend->malignMethod, particle, n_part,
                                     dx, dy, dz,
                                     epitch, eyaw, etilt, 0, 0, length, 1);
    if (ccbend->optimized)
      offsetBeamCoordinatesForMisalignment(particle, n_part, ccbend->dxOffset, 0, 0);
    verticalRbendFringe(particle, n_part, angle / 2 - yaw, rho0, KnL[1] / length, KnL[2] / length, gK[0],
                        &(fringeInt1[0]), angleSign, 0, ccbend->fringeModel, ccbend->edgeOrder);
    /*
      printf("input after adjustments: %16.10le %16.10le %16.10le %16.10le %16.10le %16.10le\n",
      particle[0][0], particle[0][1], particle[0][2],
      particle[0][3], particle[0][4], particle[0][5]);
      fflush(stdout);
    */
  }

  nTerms = 0;
  for (iTerm = 8; iTerm >= 0; iTerm--)
    if (KnL[iTerm]) {
      nTerms = iTerm + 1;
      break;
    }

  if (iPart>0 || iFinalSlice>0 || context.flags&TEST_PARTICLES || context.iPass<0 || ccbend->optimized!=1 ||
      !ccbend->centroidsRequested)
    disableSums = 1;
  else {
    disableSums = 0;
    if (!(beamSums=tmalloc((nSlices+1)*sizeof(*beamSums)))) 
      bombElegantVA("Problem allocating beam sums array for centroid output file for CCBEND %s", eptr->name);
  }

  if (sigmaDelta2)
    *sigmaDelta2 = 0;
  i_top = n_part - 1;
  edgeMultActive[0] = edgeMultActive[1] = 0;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!integrate_kick_KnL(particle[i_part], dx, dy, Po, rad_coef, isr_coef, KnL, nTerms,
                            integ_order, nSlices, iPart, iFinalSlice, length, multData, edge1MultData, edge2MultData,
                            &apertureData, &dzLoss, sigmaDelta2, &lastRho1, tilt, 0.0, eptr,
			    disableSums?NULL:&beamSums[0])) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start + dzLoss;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
    /*
      if (i_part==0 && ccbend->verbose && ccbend->optimized!=-1)
      printf("Edge multipoles active: %ld, %ld\n", edgeMultActive[0], edgeMultActive[1]);
    */
  }
  lastRho = lastRho1; /* make available for radiation integral calculations */
  multipoleKicksDone += (i_top + 1) * nSlices;
  if (multData)
    multipoleKicksDone += (i_top + 1) * nSlices * multData->orders;
  if (sigmaDelta2)
    *sigmaDelta2 /= i_top + 1;

  if ((iPart < 0 || iPart == (ccbend->nSlices - 1)) && iFinalSlice <= 0) {
    /*
      printf("output before adjustments: %16.10le %16.10le %16.10le %16.10le %16.10le %16.10le\n",
      particle[0][0], particle[0][1], particle[0][2],
      particle[0][3], particle[0][4], particle[0][5]);
    */
    verticalRbendFringe(particle, i_top + 1, angle / 2 + yaw, rho0, KnL[1] / length, KnL[2] / length, gK[1],
                        &(fringeInt2[0]), angleSign, 1, ccbend->fringeModel, ccbend->edgeOrder);
    if (ccbend->optimized)
      offsetBeamCoordinatesForMisalignment(particle, i_top + 1, ccbend->xAdjust, 0, 0);
    if (dx || dy || dz || etilt || eyaw || epitch)
      offsetParticlesForMisalignment(ccbend->malignMethod, particle, i_top + 1,
                                     dx, dy, dz,
                                     epitch, eyaw, etilt, 0, 0, length, 2);
    switchRbendPlane(particle, i_top + 1, angle / 2 + yaw, Po);
    if (ccbend->fringeModel) {
      /* If fringes are (possibly) extended, we use the difference between the initial and final
       * x coordinates including the fringe effects as the figure of merit */
      xFinal = particle[0][0]; /* This will also be used as the x adjustment to suppress offsets in tracking */
      xError = xInitial - xFinal;
    } else {
      /* For the original fringe model, we continue to try to center the beam in the magnet */
      /* xFinal is set in the trajectory error function as before, for backward compatibility */
      xError = xMax + xMin;
    }
    /* For x', want the initial and final angle errors to be equal and opposite, which ensures a symmetric
       trajectory. In practice, xpInitial=0, so the final angle is also zero.
    */
    xpError = xpInitial + particle[0][1];
    if (tilt)
      /* use n_part here so lost particles get rotated back */
      rotateBeamCoordinatesForMisalignment(particle, n_part, -tilt);
    if (ccbend->optimized == 1 && ccbend->referenceCorrection) {
      if (ccbend->referenceCorrection & 2) {
        for (i_part = 0; i_part <= i_top; i_part++)
          for (int ii = 0; ii < 4; ii++)
            particle[i_part][ii] -= ccbend->referenceTrajectory[ii];
      }
      if (ccbend->referenceCorrection & 1) {
        for (i_part = 0; i_part <= i_top; i_part++)
          particle[i_part][4] -= ccbend->referenceTrajectory[4];
      }
    }
    /*
      printf("output after adjustments: %16.10le %16.10le %16.10le %16.10le %16.10le %16.10le\n",
      particle[0][0], particle[0][1], particle[0][2],
      particle[0][3], particle[0][4], particle[0][5]);
      fflush(stdout);
    */
  } else if (iFinalSlice > 0) {
    if (tilt)
      /* use n_part here so lost particles get rotated back */
      rotateBeamCoordinatesForMisalignment(particle, n_part, -tilt);
  }
  
  if (ccbend->centroidsRequested && !disableSums) {
    long iRow = 0;
    if (ccbend->SDDScen) {
      if (!SDDS_StartPage(ccbend->SDDScen, nSlices+1)) {
	SDDS_SetError("Problem starting page for centroid output file for CCBEND");
	SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      if (!SDDS_SetParameters(ccbend->SDDScen, SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
			      "Step", context.step, "Pass", context.iPass, "pCentral", Po,
			      "ElementName", eptr->name, "ElementOccurence", eptr->occurence, NULL)) {
	SDDS_SetError("Problem preparing parameter data for centroid output file for CCBEND");
	SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
    }
    for (int iSlice=0; iSlice<=ccbend->nSlices; iSlice++, iRow++) {
#if USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &beamSums[iRow].n_part, 1, MPI_LONG, MPI_SUM, workers);
      MPI_Allreduce(MPI_IN_PLACE, beamSums[iRow].centroid, 6, MPI_DOUBLE, MPI_SUM, workers);
#endif
      if (beamSums[iRow].n_part) {
	for (int i=0; i<5; i++)
	  beamSums[iRow].centroid[i] /= beamSums[iRow].n_part;
      }
      if (ccbend->SDDScen &&
	  !SDDS_SetRowValues(ccbend->SDDScen, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iRow,
			     0, beamSums[iRow].centroid[4],
			     1, iSlice,
			     2, beamSums[iRow].centroid[0], 
			     3, beamSums[iRow].centroid[1], 
			     4, beamSums[iRow].centroid[2], 
			     5, beamSums[iRow].centroid[3], 
			     6, beamSums[iRow].centroid[5],
			     7, beamSums[iRow].n_part,
			     -1)) {
	SDDS_SetError("Problem preparing column data for centroid output file for CCBEND");
	SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
    }
    if (ccbend->SDDScen &&
	!SDDS_WritePage(ccbend->SDDScen)) {
      SDDS_SetError("Problem writing page for centroid output file for CCBEND");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(beamSums);
    beamSums = NULL;
  }


  if (angleSign < 0) {
    lastRho *= -1;
    lastX *= -1;
    lastXp *= -1;
  }

  if (freeMultData && !multData->copy) {
    if (multData->order)
      free(multData->order);
    if (multData->KnL)
      free(multData->KnL);
    free(multData);
  }

#ifdef DEBUG
  if (fpHam) {
    fclose(fpHam);
    fpHam = NULL;
  }
#endif

  log_exit("track_through_ccbend");
  return (i_top + 1);
}

static long findMaximumStepOrder(const long order2, MULTIPOLE_DATA *multData) {
  long j, maxOrder;
  maxOrder = order2;
  if (multData) {
    for (j = 0; j < (multData)->orders; j++) {
      if (maxOrder < (multData)->order[j])
        maxOrder = (multData)->order[j];
    }
  }
  return maxOrder;
}

/* beta is 2^(1/3) */
#define BETA 1.25992104989487316477

#define FILLX fillPowerArray
#define FILLY fillPowerArray
#define FILLXY fillPowerArrays
#define DO_MKICKS apply_canonical_multipole_kicks

#  define DO_MKICKS_RET apply_canonical_multipole_kicks_ret
#  define DO_MKICKS_NORET apply_canonical_multipole_kicks_noret
#  define DO_ALLKICKS_NORET apply_all_kicks_noret

int integrate_kick_KnL(double *restrict coord, /* coordinates of the particle */
                       const double dx,        /* misalignments, needed for aperture checks */
                       const double dy,
                       const double Po,
                       const double rad_coef,
                       const double isr_coef, /* radiation effects */
                       const double *KnLFull,
                       const long nTerms,            /* Number of terms, between 0 and 9 */
                       const long integration_order, /* 2, 4, or 6 */
                       const long n_parts,           /* NSLICES */
                       const long iPart,             /* If <0, integrate the full magnet. If >=0, integrate just a single part and return.
                                                      * This is needed to allow propagation of the radiation matrix. */
                       long iFinalSlice,             /* If >0, integrate to the indicated slice. Needed to allow extracting the
                                                      * interior matrix from tracking data. */
                       double drift,                 /* length of the full element */
                       /* error multipoles */
                       MULTIPOLE_DATA *multData,      /* body terms */
                       MULTIPOLE_DATA *edge1MultData, /* entrance */
                       MULTIPOLE_DATA *edge2MultData, /* exit */
                       MULT_APERTURE_DATA *apData,    /* aperture */
                       double *dsLoss,                /* if particle is lost, offset from start of element where this occurs */
                       double *sigmaDelta2,           /* accumulate the energy spread increase for propagation of radiation matrix */
                       double *lastRho,               /* needed for radiation integrals */
                       double refTilt,
                       const double dZOffset, /* offset of start of present segment relative to Z coordinate of entry plane */
                       ELEMENT_LIST *eptr,
		       GLOBAL_BEAM_SUMS *beamSums) {
  double p, qx, qy, denom, beta0, beta1, dp, s;
  double x, y, xp, yp, delta_qx, delta_qy;
  double xSum;
  long i_kick, step, iMult, nSum;
  double dsh;
  long maxOrder;

  // Only go up to K8 => 9 terms max length (same as in parent methods)
#  define MAX_MULT_ORDER 9
  // ccbend2 has maxOrder=19...
  static double xpow[MAX_EXTRA_ORDER];
  static double ypow[MAX_EXTRA_ORDER];

  double KnL[MAX_MULT_ORDER];
  double KnLActive[MAX_MULT_ORDER];
  int KnLIdx[MAX_MULT_ORDER];
  int kidx = 0;
  for (int i = 0; i < MAX_MULT_ORDER; i++) {
    KnL[i] = KnLFull[i] / n_parts;
    if (KnL[i]) {
      KnLActive[kidx] = KnL[i];
      KnLIdx[kidx] = i;
      kidx++;
    }
  }

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

  static double driftBuf[8] = {0};

#ifdef DEBUG1
  static FILE *fpdeb = NULL;
  static double dZOffsetLast = -1;
  if (fpdeb == NULL) {
    fpdeb = fopen("integ.sdds", "w");
    fprintf(fpdeb, "SDDS1\n&column name=z type=double units=m &end\n");
    fprintf(fpdeb, "&column name=s type=double units=m &end\n");
    fprintf(fpdeb, "&column name=x type=double units=m &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
  if (dZOffset <= dZOffsetLast)
    fprintf(fpdeb, "\n");
  dZOffsetLast = dZOffset;

  fprintf(fpdeb, "%le %le %le\n", dZOffset, coord[4], coord[0]);
#endif

  if (dZOffset < 0) {
    printf("coding error: dZOffset<0 (%le)\n", dZOffset);
    exit(1);
  }

  drift = drift / n_parts;

  double *driftFrac = NULL, *kickFrac = NULL;
  long nSubsteps = 0;
  switch (integration_order) {
  case 2:
    nSubsteps = 2;
    driftFrac = driftBuf;
    for (int i = 0; i < nSubsteps; i++)
      driftFrac[i] = drift * driftFrac2[i];
    kickFrac = kickFrac2;
    break;
  case 4:
    nSubsteps = 4;
    driftFrac = driftBuf;
    for (int i = 0; i < nSubsteps; i++)
      driftFrac[i] = drift * driftFrac4[i];
    kickFrac = kickFrac4;
    break;
  case 6:
    nSubsteps = 8;
    driftFrac = driftBuf;
    for (int i = 0; i < nSubsteps; i++)
      driftFrac[i] = drift * driftFrac6[i];
    kickFrac = kickFrac6;
    break;
  default:
    bombElegantVA("invalid order %ld given for symplectic integrator", integration_order);
    break;
  }

  x = coord[0];
  xp = coord[1];
  y = coord[2];
  yp = coord[3];
  s = 0;
  dp = coord[5];
  p = Po * (1 + dp);
  beta0 = p / sqrt(sqr(p) + 1);

#if defined(IEEE_MATH)
  if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
    return 0;
  }
#endif
  if (fabs(x) > coordLimit || fabs(y) > coordLimit ||
      fabs(xp) > slopeLimit || fabs(yp) > slopeLimit) {
    return 0;
  }

  /* calculate initial canonical momenta */
  denom = sqrt(1 + sqr(xp) + sqr(yp));
#if TURBO_RECIPROCALS
  qx = ((1 + dp) / denom) * xp;
  qy = ((1 + dp) / denom) * yp;
#else
  qx = (1 + dp) * xp / denom;
  qy = (1 + dp) * yp / denom;
#endif

  maxOrder = findMaximumOrder(1, nTerms, edge1MultData, edge2MultData, multData);
  int maxOrderSteps = findMaximumStepOrder(nTerms, multData);
  int maxOrderEdge1 = findMaximumStepOrder(1, edge1MultData);
  int maxOrderEdge2 = findMaximumStepOrder(1, edge2MultData);

  // static allocation
  if (maxOrder > MAX_EXTRA_ORDER - 1)
    bombElegantVA("max order %ld is greater than MAX_EXTRA_ORDER", maxOrder);

  if (iPart <= 0 && edge1MultData && edge1MultData->orders) {
    FILLXY(x, xpow, y, ypow, maxOrderEdge1);
    for (iMult = 0; iMult < edge1MultData->orders; iMult++) {
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      edge1MultData->order[iMult],
                      edge1MultData->KnL[iMult], 0);
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      edge1MultData->order[iMult],
                      edge1MultData->JnL[iMult], 1);
    }
    edgeMultActive[0] = 1;
  }
  /* we must do this to avoid numerical precision issues that may subtly change the results
   * when edge multipoles are enabled to disabled
   */
  denom = sqr(1 + dp) - sqr(qx) - sqr(qy);
  if (denom <= 0) {
    coord[0] = x;
    coord[2] = y;
    return 0;
  }
  denom = sqrt(denom);
#if TURBO_RECIPROCALS
  xp = qx * (1.0 / denom);
  yp = qy * (1.0 / denom);
#else
  xp = qx / denom;
  yp = qy / denom;
#endif
  *dsLoss = 0;
  if (iFinalSlice <= 0)
    iFinalSlice = n_parts;
  xMin = DBL_MAX;
  xMax = -DBL_MAX;
  xSum = x;
  nSum = 1;
  for (i_kick = 0; i_kick < iFinalSlice; i_kick++) {
    coord[0] = x;
    coord[1] = xp;
    coord[2] = y;
    coord[3] = yp;
    if (beamSums) {
      double X, Y, Z, theta;
      convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
				      GLOBAL_LOCAL_MODE_DZ, coord, eptr, dZOffset + i_kick * drift, i_kick, n_parts);
      
      beamSums[i_kick].centroid[0] += X;
      beamSums[i_kick].centroid[1] += theta;
      beamSums[i_kick].centroid[2] += Y;
      beamSums[i_kick].centroid[3] += atan(coord[3]);
      beamSums[i_kick].centroid[4] += Z;
      beamSums[i_kick].centroid[5] += coord[5];
      beamSums[i_kick].n_part += 1;
    }
#ifdef DEBUG
    double H0;
    if (logHamiltonian && fpHam) {
      double H;
      H = -sqrt(ipow2(1 + dp) - qx * qx - qy * qy) +
        KnL[0] / drift * x +
        KnL[1] / (2 * drift) * (x * x - y * y) +
        KnL[2] / (6 * drift) * (ipow3(x) - 3 * x * y * y);
      if (i_kick == 0)
        H0 = H;
      fprintf(fpHam, "%e %e %e %e %le %le\n", i_kick * drift, x, y, qx, qy, H - H0);
      if (i_kick == n_parts)
        fputs("\n", fpHam);
    }
#endif
    delta_qx = delta_qy = 0;
#ifdef DEBUG1
    fprintf(fpdeb, "%le %le %le\n", dZOffset + i_kick * drift, s + coord[4], x);
#endif
    if ((apData && !checkMultAperture(x + dx, y + dy, dZOffset + i_kick * drift, apData))) {
      coord[0] = x;
      coord[2] = y;
      if (globalLossCoordOffset > 0) {
        double X, Y, Z, theta;
        convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
                                        GLOBAL_LOCAL_MODE_DZ, coord, eptr, dZOffset + i_kick * drift, i_kick, n_parts);
        coord[globalLossCoordOffset + 0] = X;
        coord[globalLossCoordOffset + 1] = Z;
        coord[globalLossCoordOffset + 2] = theta;
      }
      return 0;
    }
    if (insideObstruction_xyz(x, xp, y, yp, coord[particleIDIndex],
                              globalLossCoordOffset > 0 ? coord + globalLossCoordOffset : NULL,
                              refTilt, GLOBAL_LOCAL_MODE_DZ, dZOffset + i_kick * drift, i_kick, n_parts)) {
      coord[0] = x;
      coord[2] = y;
      return 0;
    }

    for (step = 0; step < nSubsteps; step++) {
      dsh = driftFrac[step];
      s += dsh * sqrt(1 + sqr(xp) + sqr(yp));
      x += xp * dsh;
      y += yp * dsh;
      // if (!kickFrac[step])
      //   break;
      //  kickfrac is 0 on last substep only
      if (step == nSubsteps - 1) {
        *dsLoss = s;
        break;
      }

      FILLXY(x, xpow, y, ypow, maxOrderSteps);
      delta_qx = delta_qy = 0;
      for (int i = 0; i < kidx; i++) {
        double tx = 0;
        double ty = 0;
        DO_MKICKS_NORET(&tx, &ty, xpow, ypow,
                        KnLIdx[i], KnLActive[i] * kickFrac[step], 0);
        qx += tx;
        qy += ty;
        // if (step == nSubsteps - 2) {
        delta_qx += tx;
        delta_qy += ty;
        //}
      }


      if (multData) {
        /* do kicks for spurious multipoles */
        for (iMult = 0; iMult < multData->orders; iMult++) {
          if (multData->KnL && multData->KnL[iMult]) {
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                            multData->order[iMult],
                            multData->KnL[iMult] * kickFrac[step] / n_parts,
                            0);
          }
          if (multData->JnL && multData->JnL[iMult]) {
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                            multData->order[iMult],
                            multData->JnL[iMult] * kickFrac[step] / n_parts,
                            1);
          }
        }
      }

      // 3% runtime
      denom = sqr(1 + dp) - sqr(qx) - sqr(qy);
      if (denom <= 0) {
        coord[0] = x;
        coord[2] = y;
        *dsLoss = s;
        if (step > 0) {
          // Recompute lastRho from last step
          if (nTerms == 1)
            *lastRho = 1 / (KnL[0] / drift);
          else if (nTerms == 2)
            *lastRho = 1 / (KnL[0] / drift + lastX * (KnL[1] / drift));
          else if (nTerms > 2)
            *lastRho = 1 / (KnL[0] / drift + lastX * (KnL[1] / drift) + lastX * lastX * (KnL[2] / drift) / 2);
        }
        return 0;
      }

      // 8% runtime
      denom = sqrt(denom);
#if TURBO_RECIPROCALS
      xp = qx * (1.0 / denom);
      yp = qy * (1.0 / denom);
#else
      xp = qx / denom;
      yp = qy / denom;
#endif

      /* these three quantities are needed for radiation integrals */
      if (step == nSubsteps - 2) {
#if TURBO_RECIPROCALS
        if (nTerms == 1)
          *lastRho = drift / KnL[0];
        else if (nTerms == 2)
          *lastRho = drift / (KnL[0] + x * KnL[1]);
        else if (nTerms > 2)
          *lastRho = drift / (KnL[0] + x * KnL[1] + x * x * KnL[2] / 2);
#else
        if (nTerms == 1)
          *lastRho = 1 / (KnL[0] / drift);
        else if (nTerms == 2)
          *lastRho = 1 / (KnL[0] / drift + x * (KnL[1] / drift));
        else if (nTerms > 2)
          *lastRho = 1 / (KnL[0] / drift + x * (KnL[1] / drift) + x * x * (KnL[2] / drift) / 2);
#endif
      }
      lastX = x;
      lastXp = xp;
    }
    if ((rad_coef || isr_coef) && drift) {
      double deltaFactor, F2, dsFactor;
      qx /= (1 + dp);
      qy /= (1 + dp);
      deltaFactor = sqr(1 + dp);
      /* delta_qx and delta_qy are for the last step and have kickFrac[step-1] included, so remove it */
      delta_qx /= kickFrac[step - 1];
      delta_qy /= kickFrac[step - 1];
#if TURBO_RECIPROCALS
      F2 = (sqr(delta_qx) + sqr(delta_qy)) / sqr(drift);
#else
      F2 = sqr(delta_qx / drift) + sqr(delta_qy / drift);
#endif
      dsFactor = sqrt(1 + sqr(xp) + sqr(yp)) * drift;
      if (rad_coef)
        dp -= rad_coef * deltaFactor * F2 * dsFactor;
      if (isr_coef > 0)
        dp -= isr_coef * deltaFactor * pow(F2, 0.75) * sqrt(dsFactor) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
      if (sigmaDelta2)
        *sigmaDelta2 += sqr(isr_coef * deltaFactor) * pow(F2, 1.5) * dsFactor;
      qx *= (1 + dp);
      qy *= (1 + dp);
    }
    xSum += x;
    nSum++;
    if (x > xMax)
      xMax = x;
    if (x < xMin)
      xMin = x;
    if (iPart >= 0)
      break;
  }
  xFinal = x; /* may be needed for fringeModel==0, for backward compatibility */
  xAve = xSum / nSum;

  /*
    printf("x init, min, max, fin = %le, %le, %le, %le\n", x0, xMin, xMax, x);
    printf("xp init, fin = %le, %le\n", xp0, xp);
  */

#ifdef DEBUG1
  fprintf(fpdeb, "%le %le %le\n", dZOffset + i_kick * drift, s + coord[4], x);
#endif
  if ((apData && !checkMultAperture(x + dx, y + dy, dZOffset + i_kick * drift, apData))) {
    coord[0] = x;
    coord[2] = y;
    if (globalLossCoordOffset > 0) {
      double X, Y, Z, theta;
      convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
                                      GLOBAL_LOCAL_MODE_DZ, coord, eptr, dZOffset + i_kick * drift, i_kick, n_parts);
      coord[globalLossCoordOffset + 0] = X;
      coord[globalLossCoordOffset + 1] = Z;
      coord[globalLossCoordOffset + 2] = theta;
    }
    return 0;
  }
  if (insideObstruction_xyz(x, xp, y, yp, coord[particleIDIndex],
                            globalLossCoordOffset > 0 ? coord + globalLossCoordOffset : NULL,
                            refTilt, GLOBAL_LOCAL_MODE_DZ, dZOffset + i_kick * drift, i_kick, n_parts)) {
    coord[0] = x;
    coord[2] = y;
    return 0;
  }

  if ((iPart < 0 || iPart == n_parts) && (iFinalSlice == n_parts) && edge2MultData && edge2MultData->orders) {
    FILLXY(x, xpow, y, ypow, maxOrderEdge2);
    for (iMult = 0; iMult < edge2MultData->orders; iMult++) {
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      edge2MultData->order[iMult],
                      edge2MultData->KnL[iMult], 0);
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      edge2MultData->order[iMult],
                      edge2MultData->JnL[iMult], 1);
    }
    edgeMultActive[1] = 1;
  }
  denom = sqr(1 + dp) - sqr(qx) - sqr(qy);
  if (denom <= 0) {
    coord[0] = x;
    coord[2] = y;
    return 0;
  }

  denom = sqrt(denom);
#if TURBO_RECIPROCALS
  xp = qx * (1.0 / denom);
  yp = qy * (1.0 / denom);
#else
  xp = qx / denom;
  yp = qy / denom;
#endif


  coord[0] = x;
  coord[1] = xp;
  coord[2] = y;
  coord[3] = yp;
  if (rad_coef) {
    p = Po * (1 + dp);
    beta1 = p / sqrt(sqr(p) + 1);
    coord[4] = beta1 * (coord[4] / beta0 + 2 * s / (beta0 + beta1));
  } else
    coord[4] += s;
  coord[5] = dp;

  if (beamSums) {
    double X, Y, Z, theta;
    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
				    GLOBAL_LOCAL_MODE_DZ, coord, eptr, dZOffset + i_kick * drift, i_kick, n_parts);
    beamSums[i_kick].centroid[0] += X;
    beamSums[i_kick].centroid[1] += theta;
    beamSums[i_kick].centroid[2] += Y;
    beamSums[i_kick].centroid[3] += atan(coord[3]);
    beamSums[i_kick].centroid[4] += Z;
    beamSums[i_kick].centroid[5] += coord[5];
    beamSums[i_kick].n_part += 1;
  }

#if defined(IEEE_MATH)
  if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
    return 0;
  }
#endif
  if (fabs(x) > coordLimit || fabs(y) > coordLimit ||
      fabs(xp) > slopeLimit || fabs(yp) > slopeLimit) {
    return 0;
  }
  return 1;
}

#define USE_NEW_SWITCH_CODE 0

void switchRbendPlane(double **particle, long n_part, double alpha, double po)
/* transforms the reference plane to one that is at an angle alpha relative to the
 * initial plane. use alpha=theta/2, where theta is the total bend angle.
 */
{
#if USE_NEW_SWITCH_CODE
  long i;
  double cos_alpha, tan_alpha, magnet_s;

  tan_alpha = tan(alpha);
  cos_alpha = cos(alpha);

  for (i = 0; i < n_part; i++) {
    magnet_s = particle[i][0] * tan_alpha / (1.0 - particle[i][1] * tan_alpha);
    particle[i][0] = (particle[i][0] + particle[i][1] * magnet_s) / cos_alpha;
    particle[i][2] = particle[i][2] + particle[i][3] * magnet_s;

    particle[i][4] += sqrt(1.0 + sqr(particle[i][1]) + sqr(particle[i][3])) * magnet_s;

    particle[i][3] = particle[i][3] / (cos_alpha * (1.0 - particle[i][1] * tan_alpha));
    particle[i][1] = (particle[i][1] + tan_alpha) / (1.0 - particle[i][1] * tan_alpha);
  }
#else
  long i;
  double s, *coord, sin_alpha, cos_alpha, tan_alpha;
  double d, qx0, qy0, qz0, qx, qy, qz;

  tan_alpha = tan(alpha);
  cos_alpha = cos(alpha);
  sin_alpha = sin(alpha);

  for (i = 0; i < n_part; i++) {
    coord = particle[i];
    s = coord[0] * tan_alpha / (1 - coord[1] * tan_alpha);
    coord[0] = (coord[0] + coord[1] * s) / cos_alpha;
    coord[2] = (coord[2] + coord[3] * s);
    coord[4] += s;
    d = sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    qx0 = coord[1] * (1 + coord[5]) / d;
    qy0 = coord[3] * (1 + coord[5]) / d;
    qz0 = (1 + coord[5]) / d;
    qx = qx0 * cos_alpha + qz0 * sin_alpha;
    qy = qy0;
    qz = -qx0 * sin_alpha + qz0 * cos_alpha;
    coord[1] = qx / qz;
    coord[3] = qy / qz;
  }
#endif
}

void verticalRbendFringe(
                         double **particle, long n_part,
                         double alpha,        // edge angle relative to beam path
                         double rho0,         // bending radius (always positive)
                         double K1,           // interior gradient
                         double K2,           // interior sextupole
                         double gK,           // conventional soft-edge parameter
                         double *fringeIntKn, // fringe integrals: K0, I0, K2, I1, K4, K5, K6, K7
                         short angleSign,     // -1 or 1
                         short isExit,
                         short fringeModel,
                         short order) {

  if (fringeModel == -1)
    return;

  if (fringeModel == 0) {
    // old method
    long i;
    double c, d, e;
    if (order < 1)
      return;
    c = d = e = 0;
    if (gK != 0)
      alpha -= gK / fabs(rho0) / cos(alpha) * (1 + sqr(sin(alpha)));
    c = sin(alpha) / rho0;
    if (order > 1)
      d = sin(alpha) * K1;
    if (order > 2)
      e = sin(alpha) * K2 / 2;
    for (i = 0; i < n_part; i++) {
      double x, y;
      x = particle[i][0];
      y = particle[i][2];
      particle[i][3] -= y * (c + d * x + e * (x * x - y * y)) / (1 + particle[i][5]);
    }
  } else {
    double x1, px1, y1, py1, tau1, delta;
    double x2, px2, y2, py2, tau2;

    double intK0, intK2, intK4, intK5, intK6, intK7, intI0, intI1;
    double invRhoPlus, invRhoMinus, K1plus, K1minus;
    double tant, sect, sect3, sint, temp;
    double focX0, /* focXd, */ focY0, focYd, invP;
    double dispX, kickPx, expT;

    long i;
    if (isExit)
      alpha = -alpha;

    // Per R. Lindberg, some of the fringe integrals change sign with the sign of the bending angle.
    intK0 = fringeIntKn[0] * angleSign;
    intK2 = fringeIntKn[2];
    intK4 = fringeIntKn[4] * angleSign;
    intK5 = fringeIntKn[5] * angleSign;
    intK6 = fringeIntKn[6] * angleSign;
    intK7 = fringeIntKn[7];
    // shamelessly packed the In integrals in the slots unused by the Kn's
    intI0 = fringeIntKn[1];
    intI1 = fringeIntKn[3];

    if (isExit) {
      // exit fringe
      K1minus = K1;
      K1plus = 0;
      invRhoMinus = 1 / rho0;
      invRhoPlus = 0;
    } else {
      // entrance fringe
      K1plus = K1;
      K1minus = 0;
      invRhoPlus = 1 / rho0;
      invRhoMinus = 0;
    }

    tant = tan(alpha);
    sint = sin(alpha);
    sect = 1.0 / cos(alpha);
    sect3 = sect * sect * sect;
    focX0 = -tant * intK5 - 0.5 * intI0 * (2.0 - tant * tant);
    /* focXd = sect3 * intK7; */
    focY0 = -tant * (invRhoPlus - invRhoMinus) + tant * sect * sect * intK5 + 0.5 * intI0 * (2.0 + tant * tant);
    focYd = sect3 * ((1.0 + sint * sint) * intK2 - intK7);
    for (i = 0; i < n_part; i++) {
      x1 = particle[i][0];
      temp = sqrt(1.0 + sqr(particle[i][1]) + sqr(particle[i][3]));
      px1 = (1.0 + particle[i][5]) * particle[i][1] / temp;
      y1 = particle[i][2];
      py1 = (1.0 + particle[i][5]) * particle[i][3] / temp;
      tau1 = -particle[i][4] + sint * x1;
      delta = particle[i][5];
      px1 = px1 - (1.0 + delta) * sint;
      invP = 1.0 / (1.0 + delta);

      /** Symplectic update for Linear and Zeroth Order terms **/
      temp = sect * intI1 * invP;
      if (fabs(temp) < 1.0e-20)
        temp = 1.0e-20;
      dispX = -sect * intK0 * invP;
      kickPx = -tant * (intI1 + 0.5 * intK4);
      expT = exp(temp);

      x2 = expT * x1 + dispX * (expT - 1.0) / temp;
      y2 = y1 / expT;
      px2 = px1 / expT + kickPx * (1.0 - 1.0 / expT) / temp + focX0 * (x1 * (expT - 1.0 / expT) / (2.0 * temp) + dispX * (expT + 1.0 / expT - 2.0) / (2.0 * temp * temp));
      py2 = expT * py1 + y1 * (focY0 + focYd * invP) * (expT - 1.0 / expT) / (2.0 * temp);
      tau2 = tau1 + invP * (kickPx * (temp * x1 + dispX) * (1.0 + temp - expT) / (temp * temp) - px1 * dispX - (px1 * x1 - py1 * y1) * temp + (0.5 * focYd * invP + focY0 * (1.0 / (expT * expT) - 1.0 + 2.0 * temp) / (4.0 * temp)) * y1 * y1 - x1 * x1 * focX0 * (expT * expT - 1.0 - 2.0 * temp) / (4.0 * temp) - x1 * dispX * focX0 * (expT * expT - 2.0 * expT + 1.0) / (2.0 * temp * temp) - dispX * dispX * focX0 * (expT * expT - 4.0 * expT + 2.0 * temp + 3.0) / (4.0 * temp * temp * temp));

      /* x2 += 0.5*sect3*( intK5*(x1*x1 - y1*y1) + y1*y1*(invRhoPlus-invRhoMinus) )*invP;
         y2 += -sect*intK5*x1*y1*invP;
         px2 += -0.25*tant*(K1plus - K1minus)*(x1*x1 + y1*y1) - 0.5*intK6*(x1*x1 - sect*sect*y1*y1)
         + intK5*(sect*py1*y1 - sect3*px1*x1)*invP; // hereit
         py2 += (sect*sect*intK6 - 0.5*tant*(K1plus - K1minus))*x1*y1
         + ( intK5*(sect*py1*x1 + sect3*px1*y1) - sect3*(invRhoPlus - invRhoMinus)*px1*y1 )*invP;
         tau2 += ( sect*intK5*py1*x1*y1
         + sect3*(intK5*px1*(y1*y1-x1*x1) - px1*y1*y1*(invRhoPlus-invRhoMinus)) )*invP*invP; */

      if (order >= 2) {
        /** Symplectic update for second order terms **/
        x1 = x2;
        y1 = y2 * exp(-sect * intK5 * x2 * invP);
        px1 = px2 - (0.5 * intK6 + 0.25 * tant * (K1plus - K1minus)) * x2 * x2 + sect * intK5 * invP * py2 * y2;
        py1 = py2 * exp(sect * intK5 * x2 * invP);
        tau1 = tau2 + sect * intK5 * invP * invP * py2 * x2 * y2;

        x2 = x1 - 0.5 * sect3 * (intK5 - (invRhoPlus - invRhoMinus)) * invP * y1 * y1;
        y2 = y1;
        px2 = px1 + (0.5 * sect * sect * intK6 - 0.25 * tant * (K1plus - K1minus)) * y1 * y1;
        py2 = py1 + sect3 * (intK5 - (invRhoPlus - invRhoMinus)) * invP * px1 * y1 + (sect * sect * intK6 - 0.5 * tant * (K1plus - K1minus)) * x1 * y1;
        tau2 = tau1 + 0.5 * sect3 * (intK5 - (invRhoPlus - invRhoMinus)) * invP * invP * y1 * y1 * (px1 + (0.25 * sect * sect * intK6 - 0.125 * tant * (K1plus - K1minus)) * y1 * y1);

        temp = -0.5 * sect3 * intK5 * invP * x2;
        tau2 += temp * invP * px2 * x2;
        temp = 1.0 + temp;
        x2 = x2 / temp;
        px2 = px2 * temp * temp;
      }

      particle[i][0] = x2;
      particle[i][2] = y2;
      px2 = px2 + (1.0 + delta) * sint;
      temp = sqrt(sqr(1 + delta) - sqr(px2) - sqr(py2));
      particle[i][1] = px2 / temp;
      particle[i][3] = py2 / temp;
      particle[i][4] = -tau2 + sint * x2;
    }
  }
}

double ccbend_trajectory_error(double *value, long *invalid) {
  static double **particle = NULL;
  double result;

  *invalid = 0;
  if ((ccbendCopy.fseOffset = value[0]) >= 1 || value[0] <= -1) {
    *invalid = 1;
    return DBL_MAX;
  }
  if (value[1] >= 1 || value[1] <= -1) {
    *invalid = 1;
    return DBL_MAX;
  }
  if (!particle)
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
  memset(particle[0], 0, totalPropertiesPerParticle * sizeof(**particle));
  ccbendCopy.dxOffset = value[1];
  if (ccbendCopy.compensateKn)
    ccbendCopy.KnDelta = -ccbendCopy.fseOffset;
  /* printf("** fse = %le, dx = %le, x[0] = %le\n", value[0], value[1], particle[0][0]); fflush(stdout);  */
  if (!track_through_ccbend(particle, 1, eptrCopy, &ccbendCopy, PoCopy, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, -1)) {
    *invalid = 1;
    return DBL_MAX;
  }
  result = 0;
  if (optimizationFlags & OPTIMIZE_X)
    result += fabs(xError);
  if (optimizationFlags & OPTIMIZE_XP)
    result += fabs(xpError / ccbendCopy.angle);
  /*
    printf("part[0][0] = %le, part[0][1] = %le, xInitial=%le, xpInitial=%le, xFinal = %le, xError = %le, xpError = %le, result = %le\n",
    particle[0][0], particle[0][1], xInitial, xpInitial, xFinal, xError, xpError, result); fflush(stdout);
  */
  return result;
}

VMATRIX *determinePartialCcbendLinearMatrix(CCBEND *ccbend, double *startingCoord, double pCentral, long iFinalSlice)
/* This routine is used for getting the linear transport matrix from the start of the CCBEND to some interior slice.
 * We need this to compute radiation integrals.
 */
{
  double **coord;
  long n_track, i, j;
  VMATRIX *M;
  double **R, *C;
  double stepSize[6] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5};
  long ltmp1, ltmp2;
  double angle0[2];

#if USE_MPI
  long notSinglePart_saved = notSinglePart;

  /* All the particles should do the same thing for this routine. */
  notSinglePart = 0;
#endif

  coord = (double **)czarray_2d(sizeof(**coord), 1 + 6 * 4, totalPropertiesPerParticle);

  n_track = 4 * 6 + 1;
  for (j = 0; j < 6; j++)
    for (i = 0; i < n_track; i++)
      coord[i][j] = startingCoord ? startingCoord[j] : 0;

  /* particles 0 and 1 are for d/dx */
  coord[0][0] += stepSize[0];
  coord[1][0] -= stepSize[0];
  /* particles 2 and 3 are for d/dxp */
  coord[2][1] += stepSize[1];
  coord[3][1] -= stepSize[1];
  /* particles 4 and 5 are for d/dy */
  coord[4][2] += stepSize[2];
  coord[5][2] -= stepSize[2];
  /* particles 6 and 7 are for d/dyp */
  coord[6][3] += stepSize[3];
  coord[7][3] -= stepSize[3];
  /* particles 8 and 9 are for d/ds */
  coord[8][4] += stepSize[4];
  coord[9][4] -= stepSize[4];
  /* particles 10 and 11 are for d/delta */
  coord[10][5] += stepSize[5];
  coord[11][5] -= stepSize[5];

  /* particles 12 and 13 are for d/dx */
  coord[12][0] += 3 * stepSize[0];
  coord[13][0] -= 3 * stepSize[0];
  /* particles 14 and 15 are for d/dxp */
  coord[14][1] += 3 * stepSize[1];
  coord[15][1] -= 3 * stepSize[1];
  /* particles 16 and 17 are for d/dy */
  coord[16][2] += 3 * stepSize[2];
  coord[17][2] -= 3 * stepSize[2];
  /* particles 18 and 19 are for d/dyp */
  coord[18][3] += 3 * stepSize[3];
  coord[19][3] -= 3 * stepSize[3];
  /* particles 20 and 21 are for d/ds */
  coord[20][4] += 3 * stepSize[4];
  coord[21][4] -= 3 * stepSize[4];
  /* particles 22 and 23 are for d/delta */
  coord[22][5] += 3 * stepSize[5];
  coord[23][5] -= 3 * stepSize[5];
  /* particle n_track-1 is the reference particle (coordinates set above) */

  /* save radiation-related parameters */
  ltmp1 = ccbend->isr;
  ltmp2 = ccbend->synch_rad;

  ccbend->isr = ccbend->synch_rad = 0;
  track_through_ccbend(coord, n_track, NULL, ccbend, pCentral, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, iFinalSlice);

  ccbend->isr = ltmp1;
  ccbend->synch_rad = ltmp2;

  M = tmalloc(sizeof(*M));
  M->order = 1;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  /* Rotate the coordinates into the local frame by assuming that the reference particle
   * is on the reference trajectory. Not valid if errors are present, but not a bad approximation presumably.
   */
  for (j = 0; j < 2; j++)
    angle0[j] = atan(coord[n_track - 1][2 * j + 1]);
  for (i = 0; i < n_track; i++) {
    for (j = 0; j < 2; j++)
      coord[i][2 * j] = coord[i][2 * j] - coord[n_track - 1][2 * j];
    for (j = 0; j < 2; j++)
      coord[i][2 * j + 1] = tan(atan(coord[i][2 * j + 1]) - angle0[j]);
    coord[i][4] -= coord[n_track - 1][4];
  }

  for (i = 0; i < 6; i++) {
    /* i indexes the dependent quantity */

    /* Determine C[i] */
    C[i] = coord[n_track - 1][i];

    /* Compute R[i][j] */
    for (j = 0; j < 6; j++) {
      /* j indexes the initial coordinate value */
      R[i][j] =
        (27 * (coord[2 * j][i] - coord[2 * j + 1][i]) - (coord[2 * j + 12][i] - coord[2 * j + 13][i])) / (48 * stepSize[j]);
    }
  }

  free_czarray_2d((void **)coord, 1 + 4 * 6, totalPropertiesPerParticle);

#if USE_MPI
  notSinglePart = notSinglePart_saved;
#endif
  return M;
}

#undef DEBUG

void addCcbendRadiationIntegrals(CCBEND *ccbend, double *startingCoord, double pCentral,
                                 double eta0, double etap0, double beta0, double alpha0,
                                 double *I1, double *I2, double *I3, double *I4, double *I5, ELEMENT_LIST *elem) {
  long iSlice;
  VMATRIX *M;
  double gamma0, K1;
  double eta1, beta1, alpha1, etap1;
  double eta2, beta2, alpha2, etap2;
  double C, S, Cp, Sp, ds, H1, H2;
#ifdef DEBUG
  double s0;
  static FILE *fpcr = NULL;
  if (fpcr == NULL) {
    fpcr = fopen("ccbend-RI.sdds", "w");
    fprintf(fpcr, "SDDS1\n&column name=Slice type=short &end\n&column name=s type=double units=m &end\n");
    fprintf(fpcr, "&column name=betax type=double units=m &end\n");
    fprintf(fpcr, "&column name=etax type=double units=m &end\n");
    fprintf(fpcr, "&column name=etaxp type=double &end\n");
    fprintf(fpcr, "&column name=alphax type=double &end\n");
    fprintf(fpcr, "&column name=ElementName type=string &end\n");
    fprintf(fpcr, "&data mode=ascii no_row_counts=1 &end\n");
  }
  s0 = elem->end_pos - ccbend->length;
  fprintf(fpcr, "0 %le %le %le %le %le %s\n", s0, beta0, eta0, etap0, alpha0, elem->name);
#endif

  if (ccbend->tilt)
    bombElegant("Can't add radiation integrals for tilted CCBEND\n", NULL);

  gamma0 = (1 + alpha0 * alpha0) / beta0;

  beta1 = beta0;
  eta1 = eta0;
  etap1 = etap0;
  alpha1 = alpha0;
  H1 = (eta1 * eta1 + sqr(beta1 * etap1 + alpha1 * eta1)) / beta1;
  ds = ccbend->length / ccbend->nSlices; /* not really right... */
  for (iSlice = 1; iSlice <= ccbend->nSlices; iSlice++) {
    /* Determine matrix from start of element to exit of slice iSlice */
    M = determinePartialCcbendLinearMatrix(ccbend, startingCoord, pCentral, iSlice);
    C = M->R[0][0];
    Cp = M->R[1][0];
    S = M->R[0][1];
    Sp = M->R[1][1];

    /* propagate lattice functions from start of element to exit of this slice */
    beta2 = (sqr(C) * beta0 - 2 * C * S * alpha0 + sqr(S) * gamma0);
    alpha2 = -C * Cp * beta0 + (Sp * C + S * Cp) * alpha0 - S * Sp * gamma0;
    eta2 = C * eta0 + S * etap0 + M->R[0][5];
    etap2 = Cp * eta0 + Sp * etap0 + M->R[1][5];
#ifdef DEBUG
    s0 += ds;
    fprintf(fpcr, "%ld %le %le %le %le %le %s\n", iSlice, s0, beta2, eta2, etap2, alpha2, elem->name);
#endif

    /* Compute contributions to radiation integrals in this slice.
     * lastRho is saved by the routine track_through_ccbend(). Since determinePartialCcbendLinearMatrix()
     * puts the reference particle first, it is appropriate to the central trajectory.
     */
    *I1 += ds * (eta1 + eta2) / 2 / lastRho;
    *I2 += ds / sqr(lastRho);
    *I3 += ds / ipow3(fabs(lastRho));
    /* Compute effective K1 including the sextupole effect plus rotation.
     * lastX and lastXp are saved by track_through_ccbend().
     */
    K1 = (ccbend->K1 + (lastX - ccbend->dxOffset) * ccbend->K2) * cos(atan(lastXp));
    *I4 += ds * (eta1 + eta2) / 2 * (1 / ipow3(lastRho) + 2 * K1 / lastRho);
    H2 = (eta2 * eta2 + sqr(beta2 * etap2 + alpha2 * eta2)) / beta2;
    *I5 += ds / ipow3(fabs(lastRho)) * (H1 + H2) / 2;

    /* Save lattice functions as values at start of next slice */
    beta1 = beta2;
    alpha1 = alpha2;
    eta1 = eta2;
    etap1 = etap2;
    H1 = H2;
    free_matrices(M);
    free(M);
  }
}
