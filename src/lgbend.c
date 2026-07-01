/************************************************************************* \
* Copyright (c) 2022 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2022 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: lgbend()
 * purpose: tracking through canonical bend in cartesian coordinates
 * 
 * Michael Borland, 2022
 */
#include "mdb.h"
#include "track.h"
#include "multipole.h"

/* Used to share data with trajectory optimization penalty function */
static LGBEND lgbendCopy;
static ELEMENT_LIST *eptrCopy = NULL;
static double PoCopy;
/* Used to share data with the radiation integral computation */
static double lastRho, lastX, lastXp;

#ifdef DEBUG
static FILE *fpDeb = NULL;
static long iPart0;
#endif

void switchLgbendPlane(double **particle, long n_part, double dx, double alpha, double po, long exitPlane);
void lgbendFringe(double **particle, long n_part, double alpha, double invRhoPlus, double K1plus, double invRhoMinus, double K1minus,
                  LGBEND_SEGMENT *segment, short angleSign, short isExit, short edgeOrder, double Po);
static double lgBendTrajErrorFromOptim[2];
double lgbend_trajectory_error(double *value, long *invalid);

void storeLGBendOptimizedFSEValues(LGBEND *lgbend);
int retrieveLGBendOptimizedFSEValues(LGBEND *lgbend);
void readLGBendApertureData(LGBEND *lgbend);

long track_through_lgbend(
  double **particle, /* initial/final phase-space coordinates */
  long n_part,       /* number of particles */
  ELEMENT_LIST *eptr,
  LGBEND *lgbend,
  double Po,
  double **accepted,
  double z_start,
  double *sigmaDelta2, /* use for accumulation of energy spread for radiation matrix computation */
  char *rootname,
  MAXAMP *maxamp,
  APCONTOUR *apContour,
  APERTURE_DATA *apFileData,
  /* If iPart non-negative, we do one step. The assumption is that the caller is doing step-by-step
  * integration under outside control. The caller is responsible for handling the coordinates appropriately 
  * outside this routine. The element must have been previously optimized to determine FSE and X offsets.
  */
  long iPart,
  /* If iFinalSlice is positive, we terminate integration inside the magnet after the indicated number of
  * slices. The caller is responsible for handling the coordinates appropriately outside this routine. 
  * The element must have been previously optimized to determine FSE and X offsets.
  */
  long iFinalSlice) {
  double KnL[9];
  long iTerm, nTerms, iSegment, iSegment0;
  double dx, dy, dz; /* offsets of the multipole center */
  double eyaw, epitch;
  double dZOffset0, dZOffset; /* offset of start of first segment from entry plane, offset of interior point */
  long nSlices, integ_order;
  long i_part, i_top;
  double *coef;
  double fse, etilt, tilt, rad_coef, isr_coef, dzLoss = 0;
  double rho0, length, angle, angleSign, extraTilt;
  double entryAngle, exitAngle, entryPosition, exitPosition;
  MULT_APERTURE_DATA apertureData;
  double lastRho1;
  GLOBAL_BEAM_SUMS *beamSums = NULL;
  short disableSums = 1;
  TRACKING_CONTEXT context;
  getTrackingContext(&context);
  
#ifdef DEBUG
  if (!fpDeb) {
    char buffer[1024];
    snprintf(buffer, 1024, "%s.lgb", context.rootname);
    fpDeb = fopen(buffer, "w");
    fprintf(fpDeb, "SDDS1\n&column name=segment type=short &end\n");
    fprintf(fpDeb, "&column name=iPart type=short &end\n");
    fprintf(fpDeb, "&column name=z type=double units=m &end\n");
    fprintf(fpDeb, "&column name=s type=double units=m &end\n");
    fprintf(fpDeb, "&column name=x type=double units=m &end\n");
    fprintf(fpDeb, "&column name=xp type=double &end\n");
    fprintf(fpDeb, "&column name=y type=double units=m &end\n");
    fprintf(fpDeb, "&column name=yp type=double &end\n");
    fprintf(fpDeb, "&column name=part type=string &end\n");
    fprintf(fpDeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  if (lgbend->apertureDataFile && !(lgbend->localApertureData)) 
    readLGBendApertureData(lgbend);

  if (!particle)
    bombTracking("particle array is null (track_through_lgbend)");

  if (iPart >= 0 && lgbend->optimized != 1)
    bombTracking("Programming error: one-step mode invoked for unoptimized LGBEND.");
  if (iFinalSlice >= (lgbend->nSlices * lgbend->nSegments))
    iFinalSlice = 0; /* integrate the full magnet */
  if (iFinalSlice > 0 && iPart >= 0)
    bombTracking("Programming error: partial integration mode and one-step mode invoked together for LGBEND.");
  if (iFinalSlice > 0 && lgbend->optimized != 1 && lgbend->optimizeFse)
    bombTracking("Programming error: partial integration mode invoked for unoptimized LGBEND.");

  if (N_LGBEND_FRINGE_INT != 8)
    bombTracking("Coding error detected: number of fringe integrals for LGBEND is different than expected.");

  lgbend->centroidsRequested = (lgbend->centroidOutputFile && strlen(lgbend->centroidOutputFile));
  if (lgbend->centroidsRequested && !lgbend->SDDScen
#if USE_MPI      
      && myid==1 
#endif
      ) {
    lgbend->SDDScen = tmalloc(sizeof(SDDS_DATASET));
    char *filename;
    filename = compose_filename(lgbend->centroidOutputFile, context.rootname);
    if (!SDDS_InitializeOutputElegant(lgbend->SDDScen, SDDS_BINARY, 1, NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(lgbend->SDDScen, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(lgbend->SDDScen, "Step", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(lgbend->SDDScen, "Pass", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(lgbend->SDDScen, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(lgbend->SDDScen, "ElementName", NULL, SDDS_STRING) ||
          !SDDS_DefineSimpleParameter(lgbend->SDDScen, "ElementOccurence", NULL, SDDS_LONG) ||
          SDDS_DefineColumn(lgbend->SDDScen, "Z", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "Segment", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "Slice", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "CX", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "theta", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "CY", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "phi", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "Cdelta", NULL, "", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
          SDDS_DefineColumn(lgbend->SDDScen, "Particles", NULL, "", NULL, NULL, SDDS_LONG, 0) < 0 ||
          !SDDS_WriteLayout(lgbend->SDDScen)) {
        SDDS_SetError("Problem setting up centroid output file for LGBEND");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(filename);
  }
  
  if (lgbend->optimizeFse && !lgbend->optimized && lgbend->angle != 0) {
    double acc;
    double startValue[2], stepSize[2], lowerLimit[2], upperLimit[2];
    short disable[2];

    PoCopy = lastRho1 = lastX = lastXp = 0;
    if (iPart >= 0)
      bombTracking("Programming error: oneStep mode is incompatible with optmization for LGBEND.");
    if (!retrieveLGBendOptimizedFSEValues(lgbend)) {
      startValue[0] = lgbend->fseOpt[0];
      startValue[1] = lgbend->fseOpt[lgbend->nSegments - 1];
      lgbend->optimized = -1; /* flag to indicate calls to track_through_lgbend will be for FSE optimization */
      memcpy(&lgbendCopy, lgbend, sizeof(lgbendCopy));
      eptrCopy = eptr;
      lgbendCopy.fse = lgbendCopy.dx = lgbendCopy.dy = lgbendCopy.dz =
        lgbendCopy.etilt = lgbendCopy.epitch = lgbendCopy.eyaw = lgbendCopy.tilt =
          lgbendCopy.isr = lgbendCopy.synch_rad = lgbendCopy.isr1Particle = 0;
      PoCopy = Po;
      stepSize[0] = stepSize[1] = 1e-3;
      lowerLimit[0] = lowerLimit[1] = -1;
      upperLimit[0] = upperLimit[1] = 1;
      disable[0] = disable[1] = 0;
      if (simplexMin(&acc, startValue, stepSize, lowerLimit, upperLimit, disable, 2,
                     fabs(1e-15 * lgbend->length), fabs(1e-16 * lgbend->length),
                     lgbend_trajectory_error, NULL, 1500, 3, 12, 3.0, 1.0, 0) < 0) {
        bombElegantVA("failed to find FSE and x offset to center trajectory for lgbend. accuracy acheived was %le.", acc);
      }
      lgbend->fseOpt[0] = startValue[0];
      lgbend->fseOpt[lgbend->nSegments - 1] = startValue[1];
      if (lgbend->compensateKn) {
        lgbend->KnDelta[0] = -lgbend->fseOpt[0];
        lgbend->KnDelta[lgbend->nSegments - 1] = -lgbend->fseOpt[lgbend->nSegments - 1];
      }

      lgbend->optimized = 1;
      if (lgbend->verbose) {
        printf("LGBEND %s#%ld optimized: FSE[0]=%le, FSE[%ld]=%le, accuracy=%le (%le, %le)\n",
               eptr ? eptr->name : "?", eptr ? eptr->occurence : -1,
               lgbend->fseOpt[0], lgbend->nSegments - 1, lgbend->fseOpt[lgbend->nSegments - 1], acc,
	       lgBendTrajErrorFromOptim[0], lgBendTrajErrorFromOptim[1]);
	
        fflush(stdout);
      }
      storeLGBendOptimizedFSEValues(lgbend);
    }
  }
  
  rad_coef = isr_coef = 0;

  if (lgbend->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
  if (!lgbend->isr || (lgbend->isr1Particle == 0 && n_part == 1))
    /* minus sign indicates we accumulate into sigmadelta^2 only, don't perturb particles */
    isr_coef *= -1;
  if (lgbend->length < 1e-6 && (lgbend->isr || lgbend->synch_rad)) {
    rad_coef = isr_coef = 0; /* avoid unphysical results */
  }

  if (!(coef = expansion_coefficients(0)))
    bombTracking("expansion_coefficients(0) returned NULL pointer (track_through_lgbend)");
  if (!(coef = expansion_coefficients(1)))
    bombTracking("expansion_coefficients(1) returned NULL pointer (track_through_lgbend)");
  if (!(coef = expansion_coefficients(2)))
    bombTracking("expansion_coefficients(2) returned NULL pointer (track_through_lgbend)");

  nSlices = lgbend->nSlices;
  integ_order = lgbend->integration_order;
  if (nSlices <= 0)
    bombTracking("nSlices<=0 in track_lgbend()");
  if (integ_order != 2 && integ_order != 4 && integ_order != 6)
    bombTracking("multipole integration_order must be 2, 4, or 6");

  fse = lgbend->fse;

#ifdef DEBUG
  iPart0 = iPart;
  if (lgbend->optimized != -1 && iPart >= 0)
    fprintf(fpDeb, "0 %ld 0.0 %21.15le %21.15le %21.15le %21.15le %21.15le start\n",
            iPart0, particle[n_part - 1][4],
            particle[n_part - 1][0], particle[n_part - 1][1],
            particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
  if (iPart <= 0)
    exactDrift(particle, n_part, lgbend->predrift);
  dZOffset0 = lgbend->predrift * cos(lgbend->segment[0].entryAngle);
#ifdef DEBUG
  if (lgbend->optimized != -1 && iPart >= 0)
    fprintf(fpDeb, "0 %ld 0.0 %21.15le %21.15le %21.15le %21.15le %21.15le drift\n",
            iPart0, particle[n_part - 1][4],
            particle[n_part - 1][0], particle[n_part - 1][1],
            particle[n_part - 1][2], particle[n_part - 1][3]);
#endif

  double invRhoPlus, invRhoMinus, K1plus, K1minus;
  long nSegments;
  if (iFinalSlice > 0) {
    iSegment0 = 0;
    nSegments = (iFinalSlice - 1) / nSlices + 1;
    iFinalSlice -= nSlices * (nSegments - 1);
    if (nSegments <= 0 || nSegments > lgbend->nSegments || iFinalSlice <= 0) {
      bombTracking("segment account issue for LGBEND partial integration");
    }
  } else if (iPart >= 0) {
    // printf("Integrating one step LGBEND %s#%ld, iPart=%ld: ", eptr->name, eptr->occurence, iPart);
    iSegment0 = iPart / nSlices;
    nSegments = iSegment0 + 1;
    iPart -= iSegment0 * nSlices;
    // printf("iSeg0 = %ld, nSeg = %ld, iPart = %ld\n", iSegment0, nSegments, iPart);
    // print_elem(stdout, eptr);
    // fflush(stdout);
  } else {
    iSegment0 = 0;
    nSegments = lgbend->nSegments;
  }

  if (sigmaDelta2)
    *sigmaDelta2 = 0;
  i_top = n_part - 1;

  if (iPart>0 || iFinalSlice>0 || context.flags&TEST_PARTICLES || context.iPass<0 || lgbend->optimized!=1 ||
      !lgbend->centroidOutputFile || !strlen(lgbend->centroidOutputFile))
    disableSums = 1;
  else {
    disableSums = 0;
    if (!(beamSums=tmalloc((nSegments*(nSlices+1))*sizeof(*beamSums)))) 
      bombElegantVA("Problem allocating beam sums array for centroid output file for LGBEND %s", eptr->name);
  }

  entryAngle = entryPosition = 0; /* suppress compiler warning */
  for (iSegment = iSegment0; iSegment < nSegments; iSegment++) {
    for (iTerm = 0; iTerm < 9; iTerm++)
      KnL[iTerm] = 0;
    angle = lgbend->segment[iSegment].angle;
    length = lgbend->segment[iSegment].length;
    entryAngle = lgbend->segment[iSegment].entryAngle;
    entryPosition = lgbend->segment[iSegment].entryX;
    exitAngle = lgbend->segment[iSegment].exitAngle;
    exitPosition = lgbend->segment[iSegment].exitX;
    rho0 = length / (sin(entryAngle) + sin(angle - entryAngle));
    KnL[0] = (1 + fse + lgbend->fseOpt[iSegment]) / rho0 * length;
    KnL[1] = (1 + fse + lgbend->fseOpt[iSegment]) * lgbend->segment[iSegment].K1 * length / (1 - lgbend->KnDelta[iSegment]);
    KnL[2] = (1 + fse + lgbend->fseOpt[iSegment]) * lgbend->segment[iSegment].K2 * length / (1 - lgbend->KnDelta[iSegment]);
    if (angle < 0) {
      angleSign = -1;
      for (iTerm = 0; iTerm < 9; iTerm += 2)
        KnL[iTerm] *= -1;
      angle = -angle;
      rho0 = -rho0;
      entryAngle = -entryAngle;
      exitAngle = -exitAngle;
      extraTilt = PI;
      entryPosition = -entryPosition;
    } else {
      angleSign = 1;
      extraTilt = 0;
    }
    /*
    if (iPart>=0) {
      printf("angle=%le, length=%le, rho0=%le, KnL0=%le, KnL1=%le, KnL2=%le\n",
             angle, length, rho0, KnL[0], KnL[1], KnL[2]);
    }
    */
    tilt = lgbend->tilt + extraTilt;
    etilt = lgbend->etilt;
    dx = lgbend->dx;
    dy = lgbend->dy;
    dz = lgbend->dz * (lgbend->wasFlipped ? -1 : 1);
    eyaw = lgbend->eyaw;
    epitch = lgbend->epitch;

    if (tilt) {
      /* this is needed because the DX and DY offsets will be applied after the particles are
       * tilted into the magnet reference frame
       */
      dx = lgbend->dx * cos(tilt) + lgbend->dy * sin(tilt);
      dy = -lgbend->dx * sin(tilt) + lgbend->dy * cos(tilt);
    }

    setupMultApertureData(&apertureData, -tilt, apContour, maxamp, apFileData, lgbend->localApertureData,
                          z_start + lgbend->segment[iSegment].arcLengthStart + lgbend->segment[iSegment].arcLength / 2,
                          eptr);

#ifdef DEBUG
    if (lgbend->optimized != -1 && iPart >= 0)
      fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le segstart\n",
              iSegment, iPart0, (iSegment == 0 ? 0 : lgbend->segment[iSegment - 1].zAccumulated), particle[n_part - 1][4],
              particle[n_part - 1][0], particle[n_part - 1][1],
              particle[n_part - 1][2], particle[n_part - 1][3]);
#endif

    if (iSegment == 0 && iPart <= 0) {
      /* First full segment or first step of first segment, so do entrance transformations.
       * rotateBeamCoordinatesForMisalignment also rotates the spin when spinCoordOffset > 0. */
      if (tilt)
        rotateBeamCoordinatesForMisalignment(particle, n_part, tilt);
      switchLgbendPlane(particle, n_part, entryPosition, entryAngle, Po, 0);
      if (spinCoordOffset)
        rotateSpinsAboutYAxis(particle, n_part, entryAngle);
#ifdef DEBUG
      if (lgbend->optimized != -1 && iPart >= 0)
        fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le switch-lgbend-plane\n",
                iSegment, iPart0, (iSegment == 0 ? 0 : lgbend->segment[iSegment - 1].zAccumulated), particle[n_part - 1][4],
                particle[n_part - 1][0], particle[n_part - 1][1],
                particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
      /*
      if (dx || dy || dz)
        offsetBeamCoordinatesForMisalignment(particle, n_part, dx, dy, dz);
      if (etilt)
        rotateBeamCoordinatesForMisalignment(particle, n_part, etilt);
      */
      offsetParticlesForMisalignment(4, particle, n_part,
                                  dx, dy, dz,
                                  epitch, eyaw, etilt,
                                  0.0, 0.0, lgbend->segment[lgbend->nSegments - 1].zAccumulated, 1);
    }
    if (iPart <= 0) {
      /* Start of a segment */
      if (iSegment == 0) {
        invRhoMinus = 0.0;
        K1minus = 0.0;
      } else {
        invRhoMinus = sin(lgbend->segment[iSegment - 1].entryAngle) + sin(lgbend->segment[iSegment - 1].angle - lgbend->segment[iSegment - 1].entryAngle);
        invRhoMinus = invRhoMinus / lgbend->segment[iSegment - 1].length;
        K1minus = (1 + fse + lgbend->fseOpt[iSegment - 1]) * lgbend->segment[iSegment - 1].K1 / (1 - lgbend->KnDelta[iSegment - 1]);
      }
      if (lgbend->segment[iSegment].has1) {
        lgbendFringe(particle, n_part, entryAngle, 1.0 / rho0, KnL[1] / length, invRhoMinus, K1minus,
                     lgbend->segment + iSegment, angleSign, 0, lgbend->edgeOrder, Po);
#ifdef DEBUG
        if (lgbend->optimized != -1 && iPart >= 0)
          fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le fringe1\n",
                  iSegment, iPart0, (iSegment == 0 ? 0 : lgbend->segment[iSegment - 1].zAccumulated), particle[n_part - 1][4],
                  particle[n_part - 1][0], particle[n_part - 1][1],
                  particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
      }
    }

    nTerms = 0;
    for (iTerm = 8; iTerm >= 0; iTerm--)
      if (KnL[iTerm]) {
        nTerms = iTerm + 1;
        break;
      }
    if (nTerms == 0)
      nTerms = 1; /* might happen if FSE=-1 */

    dZOffset = dZOffset0 + (iSegment > 0 ? lgbend->segment[iSegment - 1].zAccumulated : 0);
    for (i_part = 0; i_part <= i_top; i_part++) {
      if (!integrate_kick_KnL(particle[i_part], dx, dy, Po, rad_coef, isr_coef, KnL, nTerms,
                              integ_order, nSlices, iPart,
                              iSegment == (nSegments - 1) ? iFinalSlice : 0,
                              length, NULL, NULL, NULL,
                              &apertureData, &dzLoss, sigmaDelta2, &lastRho1, tilt,
                              dZOffset, eptr, disableSums?NULL:&beamSums[iSegment*(nSlices+1)])) {
        swapParticles(particle[i_part], particle[i_top]);
        if (accepted)
          swapParticles(accepted[i_part], accepted[i_top]);
        particle[i_top][4] = z_start + lgbend->segment[iSegment].arcLengthStart + dzLoss;
        particle[i_top][5] = Po * (1 + particle[i_top][5]);
        i_top--;
        i_part--;
        continue;
      }
    }
    lastRho = lastRho1; /* make available for radiation integral calculation */
#ifdef DEBUG
    if (lgbend->optimized != -1 && iPart >= 0)
      fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le integrated\n",
              iSegment, iPart0, lgbend->segment[iSegment].zAccumulated, particle[n_part - 1][4],
              particle[n_part - 1][0], particle[n_part - 1][1],
              particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
    multipoleKicksDone += (i_top + 1) * nSlices;
    if (i_top >= 0) {
      lastX = particle[i_top][0];
      lastXp = particle[i_top][1];
    } else if (lgbend->optimized == -1)
      bombElegant("Particle lost while optimizing LGBEND FSE", NULL);

    if ((iPart < 0 && iFinalSlice <= 0) || (iPart == lgbend->nSlices - 1)) {
      /* end of a segment */
      if (iSegment == lgbend->nSegments - 1) {
        invRhoPlus = 0.0;
        K1plus = 0.0;
      } else {
        invRhoPlus = sin(lgbend->segment[iSegment + 1].entryAngle) + sin(lgbend->segment[iSegment + 1].angle - lgbend->segment[iSegment + 1].entryAngle);
        invRhoPlus = invRhoPlus / lgbend->segment[iSegment + 1].length;
        K1plus = (1 + fse + lgbend->fseOpt[iSegment + 1]) * lgbend->segment[iSegment + 1].K1 / (1 - lgbend->KnDelta[iSegment + 1]);
      }
      if (lgbend->segment[iSegment].has2) {
        lgbendFringe(particle, i_top + 1, -exitAngle, invRhoPlus, K1plus, 1.0 / rho0, KnL[1] / length,
                     lgbend->segment + iSegment, angleSign, 1, lgbend->edgeOrder, Po);
#ifdef DEBUG
        if (lgbend->optimized != -1 && iPart >= 0)
          fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le fringe2\n",
                  iSegment, iPart0, lgbend->segment[iSegment].zAccumulated, particle[n_part - 1][4],
                  particle[n_part - 1][0], particle[n_part - 1][1],
                  particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
      }
      if (iSegment == (lgbend->nSegments - 1)) {
        /* end of magnet */
        /*
        if (etilt)
          rotateBeamCoordinatesForMisalignment(particle, n_part, -etilt);
        if (dx || dy || dz)
          offsetBeamCoordinatesForMisalignment(particle, i_top+1, -dx, -dy, -dz);
        */
        offsetParticlesForMisalignment(4, particle, i_top + 1,
                                    dx, dy, dz,
                                    epitch, eyaw, etilt,
                                    0.0, 0.0, lgbend->segment[lgbend->nSegments - 1].zAccumulated, 2);
        switchLgbendPlane(particle, i_top + 1, -exitPosition, -exitAngle, Po, 1);
        if (spinCoordOffset)
          rotateSpinsAboutYAxis(particle, i_top + 1, -exitAngle);
#ifdef DEBUG
        if (lgbend->optimized != -1 && iPart >= 0)
          fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le switch-lgbend-plane\n",
                  iSegment, iPart0, lgbend->segment[iSegment].zAccumulated, particle[n_part - 1][4],
                  particle[n_part - 1][0], particle[n_part - 1][1],
                  particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
        exactDrift(particle, i_top + 1, lgbend->postdrift);
#ifdef DEBUG
        if (lgbend->optimized != -1 && iPart >= 0)
          fprintf(fpDeb, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le drift\n",
                  iSegment, iPart0, lgbend->segment[iSegment].zAccumulated, particle[n_part - 1][4],
                  particle[n_part - 1][0], particle[n_part - 1][1],
                  particle[n_part - 1][2], particle[n_part - 1][3]);
#endif
        if (tilt)
          /* use n_part here so lost particles get rotated back.
           * rotateBeamCoordinatesForMisalignment handles the spin rotation too. */
          rotateBeamCoordinatesForMisalignment(particle, n_part, -tilt);
        /*
          if (lgbend->optimized) {
          for (i_part=0; i_part<=i_top; i_part++)
          particle[i_part][4] += lgbend->lengthCorrection;
          }
        */
        /*
          printf("output after adjustments: %16.10le %16.10le %16.10le %16.10le %16.10le %16.10le\n",
          particle[n_part-1][0], particle[n_part-1][1], particle[n_part-1][2],
          particle[n_part-1][3], particle[n_part-1][4], particle[n_part-1][5]);
          fflush(stdout);
        */
      }
    } else if (iFinalSlice > 0) {
      if (tilt)
        /* use n_part here so lost particles get rotated back.
         * rotateBeamCoordinatesForMisalignment handles the spin rotation too. */
        rotateBeamCoordinatesForMisalignment(particle, n_part, -tilt);
    }

    if (angleSign < 0) {
      lastRho *= -1;
      /*
      lastX *= -1;
      lastXp *= -1;
      */
    }
  }

  if (sigmaDelta2)
    *sigmaDelta2 /= i_top + 1;
  
  if (lgbend->centroidsRequested && !disableSums) {
    long iRow = 0, iSDDSRow = 0;
    /* printf("entryX = %le, entryAngle = %le, dzOffset0 = %le, predrift = %le\n",
       lgbend->segment[0].entryX, lgbend->segment[0].entryAngle, dZOffset0, lgbend->predrift);
    */
    if (lgbend->SDDScen) {
      if (!SDDS_StartPage(lgbend->SDDScen, nSegments*nSlices+1)) {
	SDDS_SetError("Problem starting page for centroid output file for LGBEND");
	SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      if (!SDDS_SetParameters(lgbend->SDDScen, SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
			      "Step", context.step, "Pass", context.iPass, "pCentral", Po,
			      "ElementName", eptr->name, "ElementOccurence", eptr->occurence, NULL)) {
	SDDS_SetError("Problem preparing parameter data for centroid output file for LGBEND");
	SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
    }
    for (iSegment=0; iSegment<lgbend->nSegments; iSegment++) {
      for (int iSlice=0; iSlice<=lgbend->nSlices; iSlice++, iRow++) {
	if (iSlice==lgbend->nSlices && iSegment!=(lgbend->nSegments-1))
	  continue;
#if USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &beamSums[iRow].n_part, 1, MPI_LONG, MPI_SUM, workers);
	MPI_Allreduce(MPI_IN_PLACE, beamSums[iRow].centroid, 6, MPI_DOUBLE, MPI_SUM, workers);
#endif	  
	if (beamSums[iRow].n_part) {
	  for (int i=0; i<5; i++)
	    beamSums[iRow].centroid[i] /= beamSums[iRow].n_part;
	}
	if (lgbend->SDDScen &&
	    !SDDS_SetRowValues(lgbend->SDDScen, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iSDDSRow,
			       0, beamSums[iRow].centroid[4],
			       1, iSegment,
			       2, iSlice,
			       3, beamSums[iRow].centroid[0], 
			       4, beamSums[iRow].centroid[1], 
			       5, beamSums[iRow].centroid[2], 
			       6, beamSums[iRow].centroid[3], 
			       7, beamSums[iRow].centroid[5], 
                               8, beamSums[iRow].n_part,
			       -1)) {
	  SDDS_SetError("Problem preparing column data for centroid output file for LGBEND");
	  SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
	}
	iSDDSRow++;
      }
    }
    if (lgbend->SDDScen && !SDDS_WritePage(lgbend->SDDScen)) {
      SDDS_SetError("Problem writing page for centroid output file for LGBEND");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    free(beamSums);
    beamSums = NULL;
  }

  log_exit("track_through_lgbend");

  return (i_top + 1);
}

VMATRIX *determinePartialLgbendLinearMatrix(LGBEND *lgbend, double *startingCoord, double pCentral, long iFinalSlice)
/* This routine is used for getting the linear transport matrix from the start of the LGBEND to some interior slice.
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
  long distributedBeam_saved = distributedBeam;

  /* All the particles should do the same thing for this routine. */
  distributedBeam = 0;
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
  ltmp1 = lgbend->isr;
  ltmp2 = lgbend->synch_rad;

  lgbend->isr = lgbend->synch_rad = 0;
  track_through_lgbend(coord, n_track, NULL, lgbend, pCentral, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, iFinalSlice);

  lgbend->isr = ltmp1;
  lgbend->synch_rad = ltmp2;

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
  distributedBeam = distributedBeam_saved;
#endif
  return M;
}

void addLgbendRadiationIntegrals(LGBEND *lgbend, double *startingCoord, double pCentral,
                                 double eta0, double etap0, double beta0, double alpha0,
                                 double *I1, double *I2, double *I3, double *I4, double *I5, ELEMENT_LIST *elem) {
  long iSlice;
  VMATRIX *M;
  double gamma0, K1;
  double eta1, beta1, alpha1, etap1;
  double eta2, beta2, alpha2, etap2;
  double C, S, Cp, Sp, ds, H1, H2;
  long iSegment;
#ifdef DEBUG
  double s0;
  static FILE *fpcr = NULL;
#  if USE_MPI
  if (myid==1)
#  endif
    if (fpcr == NULL) {
      fpcr = fopen("lgbend-RI.sdds", "w");
      fprintf(fpcr, "SDDS1\n&column name=Segment type=short &end\n");
      fprintf(fpcr, "&column name=Slice type=short &end\n");
      fprintf(fpcr, "&column name=s type=double units=m &end\n");
      fprintf(fpcr, "&column name=x type=double units=m &end\n");
      fprintf(fpcr, "&column name=xp type=double &end\n");
      fprintf(fpcr, "&column name=betax type=double units=m &end\n");
      fprintf(fpcr, "&column name=etax type=double units=m &end\n");
      fprintf(fpcr, "&column name=etaxp type=double &end\n");
      fprintf(fpcr, "&column name=alphax type=double &end\n");
      fprintf(fpcr, "&column name=ElementName type=string &end\n");
      fprintf(fpcr, "&data mode=ascii no_row_counts=1 &end\n");
    }
  s0 = elem->end_pos - lgbend->length;
#  if USE_MPI
  if (myid==1)
#  endif
    fprintf(fpcr, "0 0 %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %s\n",
            s0, startingCoord ? startingCoord[0] : 0.0, startingCoord ? startingCoord[1] : 0.0,
            beta0, eta0, etap0, alpha0, elem->name);
#endif

  if (lgbend->tilt)
    bombElegant("Can't add radiation integrals for tilted LGBEND\n", NULL);

  gamma0 = (1 + alpha0 * alpha0) / beta0;

  beta1 = beta0;
  eta1 = eta0;
  etap1 = etap0;
  alpha1 = alpha0;
  H1 = (eta1 * eta1 + sqr(beta1 * etap1 + alpha1 * eta1)) / beta1;
  for (iSegment = 0; iSegment < lgbend->nSegments; iSegment++) {
    ds = lgbend->segment[iSegment].arcLength / lgbend->nSlices;
    for (iSlice = 1; iSlice <= lgbend->nSlices; iSlice++) {
      /* Determine matrix from start of element to exit of slice iSlice */
      M = determinePartialLgbendLinearMatrix(lgbend, startingCoord, pCentral, iSegment * lgbend->nSlices + iSlice);
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
#  if USE_MPI
      if (myid==1)
#  endif
        fprintf(fpcr, "%ld %ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %s\n",
                iSegment, iSlice, s0, lastX, lastXp, beta2, eta2, etap2, alpha2, elem->name);
#endif

      /* Compute contributions to radiation integrals in this slice.
       * lastRho is saved by the routine track_through_lgbend(). Since determinePartialLgbendLinearMatrix()
       * puts the reference particle first, it is appropriate to the central trajectory. 
       */
      *I1 += ds * (eta1 + eta2) / 2 / lastRho;
      *I2 += ds / sqr(lastRho);
      *I3 += ds / ipow3(fabs(lastRho));
      /* Compute effective K1 including the sextupole effect plus rotation.
       * lastX and lastXp are saved by track_through_lgbend().
       */
      K1 = (lgbend->segment[iSegment].K1 + (lastX - lgbend->dx) * lgbend->segment[iSegment].K2) * cos(atan(lastXp));
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
}

void lgbendFringe(
  double **particle, long n_part,
  double alpha,       // edge angle relative to beam path
  double invRhoPlus,  // inverse bending radius after fringe (always positive)
  double K1plus,      // interior gradient after fringe
  double invRhoMinus, // inverse bending radius before fringe (always positive)
  double K1minus,     // interior gradient before fringe
  LGBEND_SEGMENT *segment,
  short angleSign, // -1 or 1
  short isExit,
  short edgeOrder,
  double Po) {     // reference momentum (for spin tracking only)
  double x1, px1, y1, py1, tau1, delta;
  double x2, px2, y2, py2, tau2;

  /* Pre-fringe slope buffers reused across calls.  See verticalRbendFringe
   * in ccbend.c for the same pattern: save (xp, yp) per particle before
   * the orbit kick, then read them back after to compute the matching
   * spin rotation via applyFringeSpinKick(). */
  static double *xp0buf = NULL, *yp0buf = NULL;
  static long xp0bufMax = 0;
  long isf;
  if (spinCoordOffset) {
    if (n_part > xp0bufMax) {
      xp0buf = trealloc(xp0buf, sizeof(*xp0buf) * n_part);
      yp0buf = trealloc(yp0buf, sizeof(*yp0buf) * n_part);
      xp0bufMax = n_part;
    }
    for (isf = 0; isf < n_part; isf++) {
      xp0buf[isf] = particle[isf][1];
      yp0buf[isf] = particle[isf][3];
    }
  }

  double intK0, intK2, intK4, intK5, intK6, intK7, intI0, intI1;
  double tant, sect, sect3, sint, temp;
  double focX0, focY0, focYd, invP;
  double dispX, kickPx, expT;

  long i;
  if (isExit) {
    // exit fringe
    alpha = -alpha;
    intK0 = segment->fringeInt2K0;
    intI0 = segment->fringeInt2I0;
    intK2 = segment->fringeInt2K2;
    intI1 = segment->fringeInt2I1;
    intK4 = segment->fringeInt2K4;
    intK5 = segment->fringeInt2K5;
    intK6 = segment->fringeInt2K6;
    intK7 = segment->fringeInt2K7;
  } else {
    // entry fringe
    intK0 = segment->fringeInt1K0;
    intI0 = segment->fringeInt1I0;
    intK2 = segment->fringeInt1K2;
    intI1 = segment->fringeInt1I1;
    intK4 = segment->fringeInt1K4;
    intK5 = segment->fringeInt1K5;
    intK6 = segment->fringeInt1K6;
    intK7 = segment->fringeInt1K7;
  }

  // Per R. Lindberg, some of the fringe integrals change sign with the sign of the bending angle.
  intK0 *= angleSign;
  intK4 *= angleSign;
  intK5 *= angleSign;
  intK6 *= angleSign;

  tant = tan(alpha);
  sint = sin(alpha);
  sect = 1.0 / cos(alpha);
  sect3 = sect * sect * sect;
  focX0 = -tant * intK5 - 0.5 * intI0 * (2.0 - tant * tant);
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

    if (edgeOrder >= 2) {
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

  /* Spin precession from the integrated fringe field.  The body-bend
   * leakage term uses the current segment's rho (1/invRhoPlus for an
   * entrance fringe, 1/invRhoMinus for an exit fringe); whichever is
   * zero corresponds to "no body on that side" and contributes zero to
   * the spin rotation. */
  if (spinCoordOffset) {
    double body_invRho = isExit ? invRhoMinus : invRhoPlus;
    double body_rho_signed = (body_invRho != 0) ? (angleSign / body_invRho) : 0.0;
    for (isf = 0; isf < n_part; isf++) {
      double p_i = Po * (1 + particle[isf][5]);
      applyFringeSpinKick(particle[isf] + spinCoordOffset,
                          particle[isf][1], xp0buf[isf],
                          particle[isf][3], yp0buf[isf],
                          particle[isf][2],
                          body_rho_signed, p_i, particleAnomalousMagneticMoment);
    }
  }
}

void switchLgbendPlane(double **particle, long n_part, double dx, double alpha, double po, long exitPlane)
/* transforms the reference plane to one that is at an angle alpha relative to the
 * initial plane. 
 */
{
  long i;
  double s, *coord, sin_alpha, cos_alpha, tan_alpha;
  double d, qx0, qy0, qz0, qx, qy, qz;

  tan_alpha = tan(alpha);
  cos_alpha = cos(alpha);
  sin_alpha = sin(alpha);

  if (exitPlane)
    for (i = 0; i < n_part; i++) {
      coord = particle[i];
      coord[0] += dx;
    }

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

  if (!exitPlane)
    for (i = 0; i < n_part; i++) {
      coord = particle[i];
      coord[0] += dx;
    }
}

double lgbend_trajectory_error(double *value, long *invalid) {
  static double **particle = NULL;
  double result;

  *invalid = 0;
  if ((lgbendCopy.fseOpt[0] = value[0]) >= 1 || value[0] <= -1) {
    *invalid = 1;
    return DBL_MAX;
  }
  if ((lgbendCopy.fseOpt[lgbendCopy.nSegments - 1] = value[1]) >= 1 || value[1] <= -1) {
    *invalid = 1;
    return DBL_MAX;
  }
  if (!particle)
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
  memset(particle[0], 0, totalPropertiesPerParticle * sizeof(**particle));
  if (lgbendCopy.compensateKn) {
    lgbendCopy.KnDelta[0] = -lgbendCopy.fseOpt[0];
    lgbendCopy.KnDelta[lgbendCopy.nSegments - 1] = -lgbendCopy.fseOpt[lgbendCopy.nSegments - 1];
  }
  if (!track_through_lgbend(particle, 1, eptrCopy, &lgbendCopy, PoCopy, NULL, 0.0, NULL, NULL, NULL, NULL, NULL, -1, -1)) {
    *invalid = 1;
    return DBL_MAX;
  }
  lgBendTrajErrorFromOptim[0] = particle[0][0];
  lgBendTrajErrorFromOptim[1] = particle[0][1];
  result = fabs(particle[0][0]) + fabs(particle[0][1]);
  return result;
}

static long nStoredLGBENDs = 0;
LGBEND *storedLGBEND = NULL;

int retrieveLGBendOptimizedFSEValues(LGBEND *lgbend) {
  long i;
  for (i = 0; i < nStoredLGBENDs; i++) {
    if (strcmp(lgbend->configurationFile, storedLGBEND[i].configurationFile) == 0 &&
        lgbend->nSegments == storedLGBEND[i].nSegments &&
        lgbend->length == storedLGBEND[i].length &&
        lgbend->xVertex == storedLGBEND[i].xVertex &&
        lgbend->zVertex == storedLGBEND[i].zVertex &&
        lgbend->xEntry == storedLGBEND[i].xEntry &&
        lgbend->zEntry == storedLGBEND[i].zEntry &&
        lgbend->xExit == storedLGBEND[i].xExit &&
        lgbend->zExit == storedLGBEND[i].zExit &&
        lgbend->tilt == storedLGBEND[i].tilt &&
        lgbend->nSlices == storedLGBEND[i].nSlices &&
        lgbend->integration_order == storedLGBEND[i].integration_order &&
        lgbend->compensateKn == storedLGBEND[i].compensateKn &&
        lgbend->wasFlipped == storedLGBEND[i].wasFlipped) {
      lgbend->optimized = 1;
      lgbend->initialized = 1;
      lgbend->fseOpt[0] = storedLGBEND[i].fseOpt[0];
      lgbend->fseOpt[lgbend->nSegments - 1] = storedLGBEND[i].fseOpt[storedLGBEND[i].nSegments - 1];
      if (lgbend->compensateKn) {
        lgbend->KnDelta[0] = -lgbend->fseOpt[0];
        lgbend->KnDelta[lgbend->nSegments - 1] = -lgbend->fseOpt[lgbend->nSegments - 1];
      }
      return 1;
    }
  }
  return 0;
}

void storeLGBendOptimizedFSEValues(LGBEND *lgbend) {
  long i;
  if (!retrieveLGBendOptimizedFSEValues(lgbend)) {
    i = nStoredLGBENDs++;
    storedLGBEND = SDDS_Realloc(storedLGBEND, sizeof(*storedLGBEND) * nStoredLGBENDs);
    cp_str(&storedLGBEND[i].configurationFile, lgbend->configurationFile);
    storedLGBEND[i].fseOpt = tmalloc(sizeof(storedLGBEND[i].fseOpt[0]) * (lgbend->nSegments));
    storedLGBEND[i].nSegments = lgbend->nSegments;
    storedLGBEND[i].length = lgbend->length;
    storedLGBEND[i].xVertex = lgbend->xVertex;
    storedLGBEND[i].zVertex = lgbend->zVertex;
    storedLGBEND[i].xEntry = lgbend->xEntry;
    storedLGBEND[i].zEntry = lgbend->zEntry;
    storedLGBEND[i].xExit = lgbend->xExit;
    storedLGBEND[i].zExit = lgbend->zExit;
    storedLGBEND[i].tilt = lgbend->tilt;
    storedLGBEND[i].nSlices = lgbend->nSlices;
    storedLGBEND[i].integration_order = lgbend->integration_order;
    storedLGBEND[i].compensateKn = lgbend->compensateKn;
    storedLGBEND[i].wasFlipped = lgbend->wasFlipped;
    storedLGBEND[i].fseOpt[0] = lgbend->fseOpt[0];
    storedLGBEND[i].fseOpt[storedLGBEND[i].nSegments - 1] = lgbend->fseOpt[storedLGBEND[i].nSegments - 1];
  }
}


void readLGBendApertureData(LGBEND *lgbend)
{
  static htab *apertureDataHashTable = NULL;
  static long apertureDatasets = 0;
  static APERTURE_DATA **apertureDataset = NULL;
  
  char *filename;

  if (!(filename = findFileInSearchPath(lgbend->apertureDataFile)))
    bombElegantVA("failed to find LGBEND aperture file %s in search path\n", lgbend->apertureDataFile);
  
  if (!apertureDataHashTable) 
    apertureDataHashTable = hcreate(12);
  if (hcount(apertureDataHashTable)==0 || hfind(apertureDataHashTable, (unsigned char*)filename, strlen(filename))==FALSE) {
    /* Read file and add to hash table */
    if (!(apertureDataset = SDDS_Realloc(apertureDataset, sizeof(*apertureDataset)*(apertureDatasets+1))) ||
        !(apertureDataset[apertureDatasets] = malloc(sizeof(**apertureDataset))))
      bombElegantVA("Memory allocation error in readLGBendApertureData reading file %s\n", filename);
    readApertureInput(apertureDataset[apertureDatasets], filename, 1);
    hadd(apertureDataHashTable, (unsigned char *)filename, strlen(filename), (void*)apertureDataset[apertureDatasets]);
    lgbend->localApertureData = apertureDataset[apertureDatasets];
    apertureDatasets++;
  } else {
    lgbend->localApertureData = hstuff(apertureDataHashTable);
    free(filename);
  }
}
