#include "mdb.h"
#include "track.h"
//#include "math.h"
//#include "stdio.h"

OBSTRUCTION_DATASETS obstructionDataSets = {0, 0, 0, 0, {0.0, 0.0}, {-10.0, 10.0}, 0.0, NULL, 0.0, 0.0, 0, 0, NULL, NULL};

static long obstructionsInForce = 1;
void setObstructionsMode(long state) {
  obstructionsInForce = state;
}

void readObstructionInput(NAMELIST_TEXT *nltext, RUN *run) {
  SDDS_DATASET SDDSin;
  char s[16384];
  long code, nY, iY, nTotal;
  short hasCanGoFlag;
  double Y;

#include "obstructionData.h"

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&obstruction_data, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &obstruction_data);

  resetObstructionData(&obstructionDataSets);
  obstructionDataSets.YLimit[0] = y_limit[0];
  obstructionDataSets.YLimit[1] = y_limit[1];

  if (disable)
    return;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, input)) {
    sprintf(s, "Problem opening aperture input file %s", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if (!check_sdds_column(&SDDSin, "Z", "m") ||
      !check_sdds_column(&SDDSin, "X", "m")) {
    printf("Necessary data quantities (Z, X) have wrong units or are not present in %s\n",
           input);
    printf("Note that units must be \"m\" on all quantities\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (!check_sdds_parameter(&SDDSin, "ZCenter", "m") ||
      !check_sdds_parameter(&SDDSin, "XCenter", "m") ||
      !check_sdds_parameter(&SDDSin, "Superperiodicity", NULL)) {
    printf("Necessary data quantities (ZCenter, XCenter, Superperiodicity) have wrong type or units, or are not present in %s\n",
           input);
    fflush(stdout);
    exitElegant(1);
  }
  hasCanGoFlag = 0;
  if (check_sdds_parameter(&SDDSin, "CanGo", NULL))
    obstructionDataSets.hasCanGoFlag = hasCanGoFlag = 1;
  if ((obstructionDataSets.YSpacing = y_spacing) > 0 && !check_sdds_parameter(&SDDSin, "Y", "m")) {
    printf("Parameter Y has wrong type or units, or is not present in %s. Required when y_spacing>0.\n",
           input);
    fflush(stdout);
    exitElegant(1);
  }
  obstructionDataSets.periods = periods;

  nTotal = 0;
  nY = 0;
  while ((code = SDDS_ReadPage(&SDDSin)) > 0) {
    if (code == 1) {
      if (!SDDS_GetParameterAsDouble(&SDDSin, "ZCenter", &obstructionDataSets.center[0]) ||
          !SDDS_GetParameterAsDouble(&SDDSin, "XCenter", &obstructionDataSets.center[1]) ||
          !SDDS_GetParameterAsLong(&SDDSin, "Superperiodicity", &obstructionDataSets.superperiodicity)) {
        sprintf(s, "Problem getting data from page %ld of obstruction input file %s", code, input);
        SDDS_SetError(s);
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      printf("ZCenter = %le, XCenter = %le, Superperiodicity = %ld\n",
             obstructionDataSets.center[0], obstructionDataSets.center[1], (long)obstructionDataSets.superperiodicity);
    }
    Y = 0;
    iY = 0;
    if (obstructionDataSets.YSpacing) {
      if (!SDDS_GetParameterAsDouble(&SDDSin, "Y", &Y)) {
        sprintf(s, "Problem getting Y data from page %ld of obstruction input file %s", code, input);
        SDDS_SetError(s);
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      iY = Y / obstructionDataSets.YSpacing + SIGN(Y) * 0.5;
      if (fabs(iY - Y / obstructionDataSets.YSpacing) > 1e-6) {
        bombElegantVA("Problem with Y data (%le) from page %ld of obstruction input file %s: not close to multiple of declared spacing %le\n",
                      Y, code, input, obstructionDataSets.YSpacing);
      }
      for (iY = 0; iY < nY; iY++)
        if (Y == obstructionDataSets.YValue[iY])
          break;
      if (iY == nY) {
        obstructionDataSets.YValue = SDDS_Realloc(obstructionDataSets.YValue,
                                                  (nY + 1) * sizeof(double));
        obstructionDataSets.nDataSets = SDDS_Realloc(obstructionDataSets.nDataSets,
                                                     (nY + 1) * sizeof(long));
        obstructionDataSets.data = SDDS_Realloc(obstructionDataSets.data, (nY + 1) * sizeof(*(obstructionDataSets.data)));
        nY++;
        obstructionDataSets.YValue[iY] = Y;
        obstructionDataSets.nDataSets[iY] = 0;
        obstructionDataSets.data[iY] = NULL;
      }
    } else {
      /* No y spacing */
      if (nY == 0) {
        obstructionDataSets.YValue = SDDS_Realloc(obstructionDataSets.YValue,
                                                  (nY + 1) * sizeof(double));
        obstructionDataSets.nDataSets = SDDS_Realloc(obstructionDataSets.nDataSets,
                                                     (nY + 1) * sizeof(long));
        obstructionDataSets.data = SDDS_Realloc(obstructionDataSets.data, (nY + 1) * sizeof(*(obstructionDataSets.data)));
        nY = 1;
        obstructionDataSets.YValue[iY] = Y;
        obstructionDataSets.nDataSets[iY] = 0;
        obstructionDataSets.data[iY] = NULL;
      }
      iY = 0;
    }

    obstructionDataSets.data[iY] = SDDS_Realloc(obstructionDataSets.data[iY],
                                                sizeof(**(obstructionDataSets.data)) * (obstructionDataSets.nDataSets[iY] + 1));
    if ((obstructionDataSets.data[iY][obstructionDataSets.nDataSets[iY]].points = SDDS_RowCount(&SDDSin)) < 3) {
      printf("Obstruction input file %s has fewer than 3 rows on page %ld", input, code);
      fflush(stdout);
      exitElegant(1);
    }
    obstructionDataSets.data[iY][obstructionDataSets.nDataSets[iY]].canGo = 0;
    if (hasCanGoFlag &&
        !SDDS_GetParameterAsLong(&SDDSin, "CanGo",
                                 &obstructionDataSets.data[iY][obstructionDataSets.nDataSets[iY]].canGo)) {
      sprintf(s, "Problem getting CanGo parameter from page %ld of obstruction input file %s", code, input);
      SDDS_SetError(s);
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    if (!(obstructionDataSets.data[iY][obstructionDataSets.nDataSets[iY]].Z = SDDS_GetColumnInDoubles(&SDDSin, "Z")) ||
        !(obstructionDataSets.data[iY][obstructionDataSets.nDataSets[iY]].X = SDDS_GetColumnInDoubles(&SDDSin, "X"))) {
      sprintf(s, "Problem getting data from page %ld of obstruction input file %s", code, input);
      SDDS_SetError(s);
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }

    obstructionDataSets.nDataSets[iY] += 1;
    nTotal++;
  }
  if (code == 0 || nTotal < 1) {
    sprintf(s, "Problem reading obstruction  input file %s---seems to be empty", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  printf("%ld datasets read from obstruction file %s\n", nTotal, input);
  obstructionDataSets.nY = nY;
  if (obstructionDataSets.YSpacing > 0) {
    obstructionDataSets.iY0 = -1;
    find_min_max(&obstructionDataSets.YMin, &obstructionDataSets.YMax, obstructionDataSets.YValue, nY);
    printf("Found %ld y planes with Y:[%le, %le]\n", nY, obstructionDataSets.YMin, obstructionDataSets.YMax);
    for (iY = 0; iY < nY; iY++) {
      printf("Plane %ld: Y = %le, %ld contours\n", iY, obstructionDataSets.YValue[iY],
             obstructionDataSets.nDataSets[iY]);
      if (obstructionDataSets.YValue[iY] == 0)
        obstructionDataSets.iY0 = iY;
    }
    if (obstructionDataSets.iY0 == -1)
      bombElegantVA("Didn't find Y=0 data for file %s\n", input);
    if (nY > 1) {
      for (iY = 1; iY < nY; iY++) {
        double dY;
        if (obstructionDataSets.YValue[iY] <= obstructionDataSets.YValue[iY - 1])
          bombElegantVA("Data for y planes is not in increasing order. Use sddssort -parameter=Y %s to fix this.\n",
                        input);
        dY = obstructionDataSets.YValue[iY] - obstructionDataSets.YValue[iY - 1];
        if (fabs(dY / obstructionDataSets.YSpacing - 1) > 1e-5)
          bombElegantVA("Data for y planes is missing some planes with %le spacing.\n", obstructionDataSets.YSpacing);
      }
    }
  }

  obstructionDataSets.initialized = 1;

  /*
  if (1) {
    long i, j, k;
    FILE *fp;
    fp = fopen("obstruction.check", "w");
    fprintf(fp, "SDDS1\n&column name=Z type=double units=m &end\n");
    fprintf(fp, "&column name=X type=double units=m &end\n");
    fprintf(fp, "&parameter name=Y type=double units=m &end\n");
    fprintf(fp, "&data mode=ascii &end\n");
    for (i=0; i<obstructionDataSets.nY; i++) {
      for (j=0; j<obstructionDataSets.nDataSets[i]; j++) {
        fprintf(fp, "%le\n%ld\n", obstructionDataSets.YValue[i], obstructionDataSets.data[i][j].points);
        for (k=0; k<obstructionDataSets.data[i][j].points; k++) {
          fprintf(fp, "%le %le\n",
                  obstructionDataSets.data[i][j].Z[k],
                  obstructionDataSets.data[i][j].X[k]);
        }
      }
    }
    fclose(fp);
  }
  */

  return;
}

void logInside(double *part, short where) {
#ifdef DEBUG
  static FILE *fpInside = NULL;
  TRACKING_CONTEXT context;
  if (!fpInside) {
    char buffer[1024];
#  if USE_MPI
    snprintf(buffer, 1024, "insideObstruction-%04d.sdds", myid);
#  else
    snprintf(buffer, 1024, "insideObstruction.sdds");
#  endif
    fpInside = fopen(buffer, "w");
    fprintf(fpInside, "SDDS1\n");
    fprintf(fpInside, "&column name=x type=double units=m &end\n");
    fprintf(fpInside, "&column name=xp type=double &end\n");
    fprintf(fpInside, "&column name=y type=double units=m &end\n");
    fprintf(fpInside, "&column name=yp type=double &end\n");
    fprintf(fpInside, "&column name=Z type=double units=m &end\n");
    fprintf(fpInside, "&column name=X type=double units=m &end\n");
    fprintf(fpInside, "&column name=thetaX type=double units=m &end\n");
    fprintf(fpInside, "&column name=particleID type=long &end\n");
    fprintf(fpInside, "&column name=call type=short &end\n");
    fprintf(fpInside, "&column name=ElementName type=string &end\n");
    fprintf(fpInside, "&column name=ElementType type=string &end\n");
    fprintf(fpInside, "&data mode=ascii no_row_counts=1 &end\n");
  }
  getTrackingContext(&context);
  fprintf(fpInside, "%le %le %le %le %le %le %le %ld %hd %s %s\n",
          part[0], part[1], part[2], part[3],
          globalLossCoordOffset > 0 ? part[globalLossCoordOffset + 1] : 999.0,
          globalLossCoordOffset > 0 ? part[globalLossCoordOffset + 0] : 999.0,
          globalLossCoordOffset > 0 ? part[globalLossCoordOffset + 2] : 999.0,
          (long)part[6], where,
          context.element->name, entity_name[context.element->type]);
  fflush(fpInside);
#endif
}

/* Returns 0 if not obstructed */
long insideObstruction(double *part, short mode, double dz, long segment, long nSegments) {
  TRACKING_CONTEXT context;
  ELEMENT_LIST *eptr;
  /*
  static FILE *fpObs = NULL;
  static ELEMENT_LIST *lastEptr = NULL;
  */
  long ic, iperiod;
  short lost;
  double Z, X, Y, thetaX;

  /*
  if (!fpObs) {
    fpObs = fopen("globalPart.sdds", "w");
    fprintf(fpObs, "SDDS1\n&column name=Z type=double units=m &end\n");
    fprintf(fpObs, "&column name=X type=double units=m &end\n");
    fprintf(fpObs, "&column name=particleID type=long &end\n");
    fprintf(fpObs, "&data mode=ascii no_row_counts=1 &end\n");
  }
  */

  if (!obstructionDataSets.initialized) {
    /*
    printf("insideObstruction: obstructions not initialized\n");
    */
    return 0;
  }

  if (!obstructionsInForce) {
    /*
    printf("insideObstruction: obstructions not in force\n");
    */
    return 0;
  }

  getTrackingContext(&context);
  if (!(eptr = context.element)) {
    printf("No element pointer in insideObstruction()\n");
    return 0;
  }

  /*
  if (eptr!=lastEptr) {
    printf("%s#%04ld: Z=%le, X=%le, theta=%le\n", 
           eptr->name, eptr->occurence, eptr->floorCoord[2], eptr->floorCoord[0], eptr->floorAngle[0]);
    lastEptr = eptr;
  }
  */

  convertLocalCoordinatesToGlobal(&Z, &X, &Y, &thetaX, mode, part, eptr, dz, segment, nSegments);
  /*
    printf("Checking obstruction for particle %ld, x=%le, y=%le, s=%le, segment=%ld/%ld: Z=%le, X=%le, Y=%le\n",
           (long)part[6], part[0], part[2], part[4], segment, nSegments, Z, X, Y);
  */

  /* 
  fprintf(fpObs, "%le %le %ld\n", Z, X, (long)part[6]); 
  */

  if (obstructionDataSets.YLimit[0] < obstructionDataSets.YLimit[1] &&
      (Y < obstructionDataSets.YLimit[0] || Y > obstructionDataSets.YLimit[1])) {
    /*
    printf("Loss code 2 seen\n");
    fflush(stdout);
    */
    lost = 2;
  } else {
    long iY = 0;
    lost = 0;
    if (obstructionDataSets.YSpacing > 0) {
      /* round toward the midplane (Y=0) */
      if (Y <= obstructionDataSets.YMin)
        iY = 0;
      else if (Y >= obstructionDataSets.YMax)
        iY = obstructionDataSets.nY - 1;
      else {
        iY = ((long)(Y / obstructionDataSets.YSpacing)) + obstructionDataSets.iY0;
        if (iY < 0 || iY >= obstructionDataSets.nY)
          bombElegantVA("Failed to find valid iY value for Y=%le, YMin=%le, YMax=%le, dY=%le, iY0=%ld\n",
                        Y, obstructionDataSets.YMin, obstructionDataSets.YMax, obstructionDataSets.YSpacing,
                        obstructionDataSets.iY0);
      }
    }
    for (ic = 0; ic < obstructionDataSets.nDataSets[iY]; ic++) {
      for (iperiod = 0; iperiod < obstructionDataSets.periods; iperiod++) {
        if (pointIsInsideContour(Z, X,
                                 obstructionDataSets.data[iY][ic].Z,
                                 obstructionDataSets.data[iY][ic].X,
                                 obstructionDataSets.data[iY][ic].points,
                                 obstructionDataSets.center,
                                 (iperiod * PIx2) / obstructionDataSets.superperiodicity)) {
          if (!obstructionDataSets.data[iY][ic].canGo) {
            lost = 1;
            if (!obstructionDataSets.hasCanGoFlag)
              break;
          } else
            return 0;
        }
      }
    }
  }
  if (lost) {
    if (globalLossCoordOffset > 0) {
      /*
      printf("Saving global loss coordinates (1): X=%le, Z=%le, theta = %le\n",
             X, Z, thetaX);
      */
      part[globalLossCoordOffset + 0] = X;
      part[globalLossCoordOffset + 1] = Z;
      part[globalLossCoordOffset + 2] = thetaX;
    }
    logInside(part, lost);
    return 1;
  }
  return 0;
}

long insideObstruction_xyz(
  double x, /* local x coordinates */
  double xp,
  double y,
  double yp,
  long particleID,         /* for diagnostic purposes */
  double *lossCoordinates, /* (X, Z, thetaX) for return if wanted */
  double xyTilt,           /* tilt of element */
  short mode,
  double dz,
  long segment,  /* for segmented elements, the segment index */
  long nSegments /* For segmented elements, the number of segments.
                * If non-positive, the s value is used to determine the
                * longitudinal position more accurately.
                */
) {
  double part[MAX_PROPERTIES_PER_PARTICLE];
  double sin_tilt, cos_tilt;

  if (!obstructionDataSets.initialized || !obstructionsInForce)
    return 0;

  if (xyTilt) {
    sin_tilt = sin(xyTilt);
    cos_tilt = cos(xyTilt);
  } else {
    sin_tilt = 0;
    cos_tilt = 1;
  }
  memset(&part[0], 0, sizeof(double) * MAX_PROPERTIES_PER_PARTICLE);
  part[0] = x * cos_tilt - y * sin_tilt;
  part[1] = xp * cos_tilt - yp * sin_tilt;
  part[2] = x * sin_tilt + y * cos_tilt;
  part[3] = xp * sin_tilt - yp * cos_tilt;
  part[particleIDIndex] = particleID;
  if (insideObstruction(part, mode, dz, segment, nSegments)) {
    if (lossCoordinates && globalLossCoordOffset > 0)
      memcpy(lossCoordinates, part + globalLossCoordOffset, sizeof(double) * GLOBAL_LOSS_PROPERTIES_PER_PARTICLE);
    return 1;
  }
  return 0;
}

long insideObstruction_XYZ
  /* Used for elements with internal Cartesian coordinate system that
 * may be offset and rotated relative to global coordinates.
 * E.g., BRAT element.
 */
  (
    /* magnet-frame coordinates of particle */
    double X, double Y, double Z,
    /* coordinate offsets of the nominal entrance */
    double dXi, double dYi, double dZi,
    /* angle of the internal Z axis w.r.t. nominal incoming trajectory */
    double thetai,
    /* horizontal slope of particle in local coordinate system */
    double xp,
    /* return of global loss coordinates (X, Z, thetaX) */
    double *lossCoordinates) {
  TRACKING_CONTEXT context;
  double C, S;
  double X1, Y1, Z1, thetaX1;
  long ic, iperiod;
  short lost;

  /*
  static FILE *fp = NULL;
  if (!fp) {
    fp = fopen("obstructPath.sdds", "w");
    fprintf(fp, "SDDS1\n&column name=Z type=double units=m &end\n");
    fprintf(fp, "&column name=X type=double units=m &end\n");
    fprintf(fp, "&column name=dZi type=double units=m &end\n");
    fprintf(fp, "&column name=dXi type=double units=m &end\n");
    fprintf(fp, "&column name=Z1 type=double units=m &end\n");
    fprintf(fp, "&column name=X1 type=double units=m &end\n");
    fprintf(fp, "&column name=thetai type=double &end\n");
    fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
  }
  fprintf(fp, "%21.15e %21.15e %21.15e %21.15e ", Z, X, dZi, dXi);
  */

  if (!obstructionDataSets.initialized || !obstructionsInForce)
    return 0;

  X -= dXi;
  Y -= dYi;
  Z -= dZi;

  getTrackingContext(&context);
  if (context.element->pred)
    thetai -= context.element->pred->floorAngle[0];

  C = cos(thetai);
  S = sin(thetai);
  X1 = C * X - S * Z;
  Y1 = Y;
  Z1 = S * X + C * Z;
  thetaX1 = -thetai + atan(xp);
  if (context.element->pred) {
    X1 += context.element->pred->floorCoord[0];
    Y1 += context.element->pred->floorCoord[1];
    Z1 += context.element->pred->floorCoord[2];
  }

  /*
      fprintf(fp, "%21.15e %21.15e %21.15e\n", Z1, X1, thetai);
  */

  if (obstructionDataSets.YLimit[0] < obstructionDataSets.YLimit[1] &&
      (Y1 < obstructionDataSets.YLimit[0] || Y1 > obstructionDataSets.YLimit[1]))
    lost = 1;
  else {
    long iY = 0;
    if (obstructionDataSets.YSpacing > 0) {
      /* round toward the midplane (Y=0) */
      if (Y <= obstructionDataSets.YMin)
        iY = 0;
      else if (Y >= obstructionDataSets.YMax)
        iY = obstructionDataSets.nY - 1;
      else {
        iY = ((long)(Y / obstructionDataSets.YSpacing)) + obstructionDataSets.iY0;
        if (iY < 0 || iY >= obstructionDataSets.nY)
          bombElegantVA("Failed to find valid iY value for Y=%le, YMin=%le, YMax=%le, dY=%le, iY0=%ld\n",
                        Y, obstructionDataSets.YMin, obstructionDataSets.YMax, obstructionDataSets.YSpacing,
                        obstructionDataSets.iY0);
      }
    }
    lost = 0;
    for (iperiod = 0; iperiod < obstructionDataSets.periods; iperiod++) {
      for (ic = 0; ic < obstructionDataSets.nDataSets[iY]; ic++) {
        if (pointIsInsideContour(Z1, X1,
                                 obstructionDataSets.data[iY][ic].Z,
                                 obstructionDataSets.data[iY][ic].X,
                                 obstructionDataSets.data[iY][ic].points,
                                 obstructionDataSets.center,
                                 (iperiod * PIx2) / obstructionDataSets.superperiodicity)) {
          if (!obstructionDataSets.data[iY][ic].canGo) {
            lost = 1;
            if (!obstructionDataSets.hasCanGoFlag)
              break;
          } else
            return 0;
        }
      }
    }
  }
  if (lost) {
    if (lossCoordinates) {
      lossCoordinates[0] = X1;
      lossCoordinates[1] = Z1;
      lossCoordinates[2] = thetaX1;
    }
    return 1;
  }
  return 0;
}

long filterParticlesWithObstructions(double **coord, int64_t np, double **accepted, double z, double Po) {
  int64_t ip, itop;
  itop = np - 1;
  for (ip = 0; ip <= itop; ip++) {
    /* printf("filterParticlesWithObstructions: ip=%ld, itop=%ld\n", ip, itop); */
    if (insideObstruction(coord[ip], GLOBAL_LOCAL_MODE_END, 0.0, 0, 1)) {
      if (ip != itop)
        swapParticles(coord[ip], coord[itop]);
      coord[itop][4] = z;
      coord[itop][5] = Po * (1 + coord[itop][5]);
      if (accepted)
        swapParticles(accepted[ip], accepted[itop]);
      --itop;
      --ip;
    }
  }
  return itop + 1;
}

void resetObstructionData(OBSTRUCTION_DATASETS *obsData) {
  long i, j;
  if (obsData->initialized) {
    printf("Freeing old obstruction data\n"); fflush(stdout);
    for (j = 0; j < obsData->nY; j++) {
      for (i = 0; i < obsData->nDataSets[j]; i++) {
        free(obsData->data[j][i].X);
        free(obsData->data[j][i].Z);
      }
      free(obsData->data[j]);
      obsData->data[j] = NULL;
    }
    free(obsData->data);
    obsData->data = NULL;
    free(obsData->YValue);
    obsData->YValue = NULL;
    obsData->initialized = 0;
    obsData->YLimit[0] = -10;
    obsData->YLimit[1] = 10;
    printf("Done freeing old obstruction data\n"); fflush(stdout);
  }
}
