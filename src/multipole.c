/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: multipole()
 * purpose: apply kicks due to an arbitrary multipole
 * 
 * Michael Borland, 1991. 
 */
#include "mdb.h"
#include "track.h"
#include "multipole.h"
#ifdef HAVE_GPU
#  include "gpu_multipole.h"
#endif

unsigned long multipoleKicksDone = 0;
unsigned short expandHamiltonian = 0;


/* #define ODD(j) ((j) % 2) */

typedef struct {
  char *filename;
  MULTIPOLE_DATA data;
  short steering;
} STORED_MULTIPOLE_DATA;
static STORED_MULTIPOLE_DATA *storedMultipoleData = NULL;
static long nMultipoleDataSets = 0;
void applyRadialCanonicalMultipoleKicks(double *qx, double *qy,
                                        double *sum_Fx_return, double *sum_Fy_return,
                                        double *xpow, double *ypow,
                                        long order, double KnL, long skew);
long evaluateLostWithOpenSides(long code, double dx, double dy, double xsize, double ysize);


long findMaximumOrder(const long order, const long order2,
                      MULTIPOLE_DATA *edgeMultData, MULTIPOLE_DATA *steeringMultData,
                      MULTIPOLE_DATA *multData) {
  long i, j, maxOrder;
  MULTIPOLE_DATA *ptr[3];
  maxOrder = order > order2 ? order : order2;
  ptr[0] = edgeMultData;
  ptr[1] = steeringMultData;
  ptr[2] = multData;
  for (i = 0; i < 3; i++) {
    if (ptr[i]) {
      for (j = 0; j < (ptr[i])->orders; j++) {
        if (maxOrder < (ptr[i])->order[j])
          maxOrder = (ptr[i])->order[j];
      }
    }
  }
  return maxOrder;
}

void print_multipole_data(MULTIPOLE_DATA *md, char *label)
{
  printf("%s:\n", label);
  if (!md || !md->initialized) {
    printf("not initialized!\n");
    return;
  }
  printf("orders = %ld, referenceRadius = %le, referenceOrder = %ld\n",
	 (long)md->orders, md->referenceRadius, (long)md->referenceOrder);
  printf("%5s %13s %13s %13s %13s \n", "order", "an", "bn", "KnL", "JnL");
  for (int i=0; i<md->orders; i++) {
    printf("%5d %13.6e %13.6e %13.6e %13.6e\n", md->order[i],
	   md->an ? md->an[i] : NAN,
	   md->bn ? md->bn[i] : NAN,
	   md->KnL ? md->KnL[i] : NAN,
	   md->JnL ? md->JnL[i] : NAN );
  }
}

long searchForStoredMultipoleData(char *multFile) {
  long i;
  /* should use a hash table ! */
  for (i = 0; i < nMultipoleDataSets; i++)
    if (strcmp(multFile, storedMultipoleData[i].filename) == 0) {
      return i;
    }
  return -1;
}

void copyMultipoleDataset(MULTIPOLE_DATA *multData, long index) {
  memcpy(multData, &storedMultipoleData[index].data, sizeof(*multData));
  multData->copy = 1;
}

void addMultipoleDatasetToStore(MULTIPOLE_DATA *multData, char *filename) {
  printf("Adding file %s to multipole data store\n", filename);
  fflush(stdout);
  storedMultipoleData = SDDS_Realloc(storedMultipoleData, sizeof(*storedMultipoleData) * (nMultipoleDataSets + 1));
  memcpy(&storedMultipoleData[nMultipoleDataSets].data, multData, sizeof(*multData));
  storedMultipoleData[nMultipoleDataSets].filename = tmalloc(sizeof(*filename) * (strlen(filename) + 1));
  storedMultipoleData[nMultipoleDataSets].data.copy = 1;
  strcpy(storedMultipoleData[nMultipoleDataSets].filename, filename);
  nMultipoleDataSets++;
}

void readErrorMultipoleData(MULTIPOLE_DATA *multData,
                            char *multFile,
                            long steering /* 1 => systematic, 2 => random */
) {
  SDDS_DATASET SDDSin;
  char buffer[1024];
  short anCheck, bnCheck, normalCheck, skewCheck;
  long index;
  char warningBuffer[1024];

  if (!multFile || !strlen(multFile)) {
    multData->orders = 0;
    multData->initialized = 0;
    return;
  }
  if (multData->initialized)
    return;
  if ((index = searchForStoredMultipoleData(multFile)) >= 0) {
    copyMultipoleDataset(multData, index);
    return;
  }
  cp_str(&(multData->filename), multFile);
  snprintf(warningBuffer, 1024, "File is %s", multFile);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, multFile)) {
    printf("Problem opening file %s\n", multFile);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    fflush(stdout);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "order", NULL, SDDS_ANY_INTEGER_TYPE, NULL) != SDDS_CHECK_OK ||
      SDDS_CheckParameter(&SDDSin, "referenceRadius", "m", SDDS_ANY_FLOATING_TYPE, NULL) != SDDS_CHECK_OK) {
    printf("Problems with data in multipole file %s\n", multFile);
    fflush(stdout);
    exitElegant(1);
  }
  anCheck = SDDS_CheckColumn(&SDDSin, "an", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK;
  normalCheck = SDDS_CheckColumn(&SDDSin, "normal", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK;
  if (!anCheck && !normalCheck) {
    printf("Problems with data in multipole file %s: neither \"an\" nor \"normal\" column found\n", multFile);
    exitElegant(1);
  }
  if (anCheck && normalCheck) {
    printWarningForTracking("Multipole file has both \"an\" and \"normal\" columns. \"normal\" used.", warningBuffer);
    anCheck = 0;
  }

  bnCheck = SDDS_CheckColumn(&SDDSin, "bn", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK;
  skewCheck = SDDS_CheckColumn(&SDDSin, "skew", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK;
  if (steering != 1) {
    if (!bnCheck && !skewCheck) {
      printf("Problems with data in multipole file %s: neither \"bn\" nor \"skew\" column found\n", multFile);
      exitElegant(1);
    }
    if (bnCheck && skewCheck) {
      printWarningForTracking("Multipole file has both \"bn\" and \"skew\" columns. \"skew\" used.", warningBuffer);
      bnCheck = 0;
    }
  } else {
    if (bnCheck || skewCheck) {
      printWarningForTracking("Steering multipole file has systematic bn or skew columns, which are ignored.",
                              warningBuffer);
    }
  }

  if (SDDS_ReadPage(&SDDSin) != 1) {
    sprintf(buffer, "Problem reading multipole file %s\n", multFile);
    SDDS_SetError(buffer);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if ((multData->orders = SDDS_RowCount(&SDDSin)) <= 0) {
    printWarningForTracking("No rows of data in multipole file.", warningBuffer);
    SDDS_Terminate(&SDDSin);
    return;
  }
  if (!SDDS_GetParameterAsDouble(&SDDSin, "referenceRadius", &multData->referenceRadius) ||
      !(multData->order = SDDS_GetColumnInLong(&SDDSin, "order")) ||
      !(multData->an = SDDS_GetColumnInDoubles(&SDDSin, anCheck ? "an" : "normal"))) {
    sprintf(buffer, "Unable to read multipole data for file %s\n", multFile);
    SDDS_SetError(buffer);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  multData->referenceOrder = -1; /* assumed to be the order of the lowest main multipole */
  if (SDDS_CheckParameter(&SDDSin, "referenceOrder", NULL, SDDS_ANY_INTEGER_TYPE, NULL) != SDDS_CHECK_NONEXISTENT) {
    if (SDDS_CheckParameter(&SDDSin, "referenceOrder", NULL, SDDS_ANY_INTEGER_TYPE, NULL) != SDDS_CHECK_OK) {
      printf("Problems with data in multipole file %s---referenceOrder parameter should be integer type\n", multFile);
      fflush(stdout);
      exitElegant(1);
    }
    if (!SDDS_GetParameterAsLong(&SDDSin, "referenceOrder", &multData->referenceOrder) ||
        multData->referenceOrder < 0) {
      sprintf(buffer, "Unable to read referenceOrder data for file %s, or invalid value\n", multFile);
      SDDS_SetError(buffer);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
  if (steering != 1) {
    if (!(multData->bn = SDDS_GetColumnInDoubles(&SDDSin, bnCheck ? "bn" : "skew"))) {
      sprintf(buffer, "Unable to read multipole data for file %s\n", multFile);
      SDDS_SetError(buffer);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  } else {
    if (!(multData->bn = calloc(multData->orders, sizeof(*(multData->bn))))) {
      printf("Memory allocation failure (readErrorMultipoleData)\n");
      exitElegant(1);
    }
  }
  if (SDDS_ReadPage(&SDDSin) == 2) {
    printWarningForTracking("Multipole file has multiple pages, which are ignored.", warningBuffer);
  }
  SDDS_Terminate(&SDDSin);
  if (steering == 1) {
    long i, j;
    /* check for disallowed multipoles */
    for (i = 0; i < multData->orders; i++) {
      if (ODD(multData->order[i]) && (multData->an[i] != 0 || multData->bn[i] != 0)) {
        printf("Error: steering multipole file %s has disallowed odd orders.\n",
               multFile);
        exitElegant(1);
      }
    }
    /* normalize to n=0 if present */
    /* find i such that order[i] is 0 (dipole) */
    for (i = 0; i < multData->orders; i++) {
      if (multData->order[i] == 0)
        break;
    }
    if (multData->orders > 1 && i != multData->orders) {
      /* dipole present */
      if (!multData->an[i]) {
        printf("Steering multipole data in %s is invalid: an is zero for order=0\n",
               multFile);
        exitElegant(1);
      }
      /* normalize to dipole for normal and skew separately */
      for (j = 0; j < multData->orders; j++)
        if (j != i)
          multData->an[j] /= multData->an[i];
      /* remove the dipole data */
      for (j = i + 1; j < multData->orders; j++) {
        multData->an[j - 1] = multData->an[j];
        multData->order[j - 1] = multData->order[j];
      }
      multData->orders -= 1;
    }
    for (i = 0; i < multData->orders; i++)
      multData->bn[i] = multData->an[i] * ipow(-1.0, multData->order[i] / 2);
#ifdef DEBUG
    printf("Steering multipole data: \n");
    for (i = 0; i < multData->orders; i++)
      printf("%ld: %e %e\n", multData->order[i],
             multData->an[i], multData->bn[i]);
#endif
  }

  if (steering) {
    long i;
    if (!(multData->KnL = SDDS_Malloc(sizeof(*multData->KnL) * multData->orders)) ||
        !(multData->JnL = SDDS_Malloc(sizeof(*multData->JnL) * multData->orders)))
      bombTracking("memory allocation failure (readErrorMultipoleData)");
    for (i = 0; i < multData->orders; i++) {
      multData->KnL[i] =
        -1 * multData->an[i] * dfactorial(multData->order[i]) / ipow(multData->referenceRadius, multData->order[i]);
      multData->JnL[i] =
        -1 * multData->bn[i] * dfactorial(multData->order[i]) / ipow(multData->referenceRadius, multData->order[i]);
    }
  }

  multData->initialized = 1;
  multData->copy = 0;

  addMultipoleDatasetToStore(multData, multFile);
}

void initialize_fmultipole(FMULT *multipole) {
  SDDS_DATASET SDDSin;
  char buffer[1024], warningBuffer[1024];
  MULTIPOLE_DATA *multData;

  multData = &(multipole->multData);
  if (multData->initialized)
    return;
  if (!multipole->filename)
    bombTracking("FMULT element doesn't have filename");
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, multipole->filename)) {
    sprintf(buffer, "Problem opening file %s (FMULT)\n", multipole->filename);
    SDDS_SetError(buffer);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "order", NULL, SDDS_ANY_INTEGER_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "KnL", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "JnL", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
    bombTracking("problems with data in FMULT input file");
  if (SDDS_ReadPage(&SDDSin) != 1) {
    snprintf(buffer, 1024, "Problem reading FMULT file %s\n", multipole->filename);
    SDDS_SetError(buffer);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  snprintf(warningBuffer, 1024, "File is %s.", multipole->filename);
  if ((multData->orders = SDDS_RowCount(&SDDSin)) <= 0) {
    printWarningForTracking("No data in FMULT file.", warningBuffer);
    SDDS_Terminate(&SDDSin);
    return;
  }
  multData->JnL = NULL;
  if (!(multData->order = SDDS_GetColumnInLong(&SDDSin, "order")) ||
      !(multData->KnL = SDDS_GetColumnInDoubles(&SDDSin, "KnL")) ||
      !(multData->JnL = SDDS_GetColumnInDoubles(&SDDSin, "JnL"))) {
    snprintf(buffer, 1024, "Unable to read data for FMULT file %s\n", multipole->filename);
    SDDS_SetError(buffer);
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (SDDS_ReadPage(&SDDSin) == 2)
    printWarningForTracking("FMULT file has multiple pages, which are ignored.", warningBuffer);
  SDDS_Terminate(&SDDSin);
  multData->initialized = 1;
}

long fmultipole_tracking(
  double **particle, /* initial/final phase-space coordinates */
  long n_part,       /* number of particles */
  FMULT *multipole,  /* multipole structure */
  double p_error,    /* p_nominal/p_central */
  double Po,
  double **accepted,
  double z_start) {
  /* double dummy; */
  double dzLoss = 0;
  long nSlices;
  long i_part, i_top, is_lost = 0, i_order;
  double *coord;
  double drift;
  double x = 0.0, xp = 0.0, y = 0.0, yp = 0.0;
  double rad_coef;
  MULTIPOLE_DATA multData;
  double *KnLSave, *JnLSave;
  /* These are potentially accessed in integrate_kick_multipole_ord4, but in reality are irrelevant as long as all the
     KnL values are 0.
  */
  short skew[3] = {0, 0, 0};
  long order[3] = {1, 0, 0};
  double KnL[3] = {0, 0, 0};

  if (!particle)
    bombTracking("particle array is null (fmultipole_tracking)");

  if (!multipole)
    bombTracking("null MULT pointer (fmultipole_tracking)");

  if (!multipole->multData.initialized)
    initialize_fmultipole(multipole);

  if ((nSlices = multipole->n_kicks) <= 0) {
    if ((nSlices = multipole->nSlices) <= 0)
      bombTracking("N_KICKS<=0 and N_SLICES<=0 in fmultipole_tracking()");
  }

  drift = multipole->length;

  if (multipole->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  else
    rad_coef = 0;

  multData = multipole->multData; /* shares data stored in permanent structure! */
  KnLSave = tmalloc(sizeof(*KnLSave) * multData.orders);
  if (multData.JnL)
    JnLSave = tmalloc(sizeof(*JnLSave) * multData.orders);
  else
    JnLSave = NULL;
  for (i_order = 0; i_order < multData.orders; i_order++) {
    KnLSave[i_order] = multData.KnL[i_order];
    multData.KnL[i_order] *= (1 + multipole->fse) * multipole->factor;
    if (multData.JnL) {
      JnLSave[i_order] = multData.JnL[i_order];
      multData.JnL[i_order] *= (1 + multipole->fse) * multipole->factor;
    }
  }

  if (multipole->dx || multipole->dy || multipole->dz) {
    offsetBeamCoordinatesForMisalignment(particle, n_part, multipole->dx, multipole->dy, multipole->dz);
  }
  if (multipole->tilt)
    rotateBeamCoordinatesForMisalignment(particle, n_part, multipole->tilt);

  i_top = n_part - 1;
  multipoleKicksDone += (i_top + 1) * multData.orders * nSlices * 4;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!(coord = particle[i_part])) {
      printf("null coordinate pointer for particle %ld (fmultipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("null accepted coordinates pointer for particle %ld (fmultipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }

    is_lost = 0;
    if (!integrate_kick_multipole_ordn(coord, multipole->dx, multipole->dy, 0.0, 0.0, Po, rad_coef, 0.0,
                                       order, KnL, skew, nSlices, -1, drift, 4, &multData, NULL, NULL, NULL,
                                       &dzLoss, NULL, 0, multipole->tilt))
      is_lost = 1;

    x = coord[0];
    y = coord[2];
    xp = coord[1];
    yp = coord[3];

    if (is_lost || isnan(x) || isnan(y) || isnan(xp) || isnan(yp) ||
        FABS(x) > coordLimit || FABS(y) > coordLimit ||
        FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start + dzLoss;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
    }
  }

  /* Note that we undo misalignments for all particles, including lost particles */
  if (multipole->tilt)
    rotateBeamCoordinatesForMisalignment(particle, n_part, -multipole->tilt);
  if (multipole->dx || multipole->dy || multipole->dz)
    offsetBeamCoordinatesForMisalignment(particle, n_part, -multipole->dx, -multipole->dy, -multipole->dz);

  /* Restore the values so we don't change them permanently */
  for (i_order = 0; i_order < multData.orders; i_order++) {
    multData.KnL[i_order] = KnLSave[i_order];
    if (multData.JnL)
      multData.JnL[i_order] = JnLSave[i_order];
  }
  free(KnLSave);
  if (JnLSave)
    free(JnLSave);

  log_exit("fmultipole_tracking");
  return (i_top + 1);
}

long multipole_tracking(
  double **particle, /* initial/final phase-space coordinates */
  long n_part,       /* number of particles */
  MULT *multipole,   /* multipole structure */
  double p_error,    /* p_nominal/p_central */
  double Po,
  double **accepted,
  double z_start) {
  double KnL;   /* integrated strength = L/(B.rho)*(Dx^n(By))_o for central momentum */
  long order;   /* order (n) */
  long nSlices; /* number of parts to split multipole into */
  long i_part, i_kick, i, i_top, is_lost;
  double sum_Fx, sum_Fy, qx, qy;
  double *coord;
  double drift;
  double *coef;
  double x, xp, y, yp, s, dp;
  /* double ratio; */
  double rad_coef;
  double beta0, beta1, p;
  static long maxOrder = -1;
  static double *xpow = NULL, *ypow = NULL;

  log_entry("multipole_tracking");

  if (!particle)
    bombTracking("particle array is null (multipole_tracking)");

  if (!multipole)
    bombTracking("null MULT pointer (multipole_tracking)");
  expandHamiltonian = multipole->expandHamiltonian;

  if ((nSlices = multipole->nSlices) <= 0)
    bombTracking("N_SLICES (or N_KICKS) <=0  in multipole_tracking()");

  if ((order = multipole->order) < 0)
    bombTracking("order < 0 in multipole_tracking()");
  if (order > maxOrder || maxOrder == -1 || !xpow || !ypow) {
    xpow = SDDS_Realloc(xpow, sizeof(*xpow) * (order + 1));
    ypow = SDDS_Realloc(ypow, sizeof(*ypow) * (order + 1));
    maxOrder = order;
  }

  if (!(coef = expansion_coefficients(order)))
    bombTracking("expansion_coefficients() returned null pointer (multipole_tracking)");

  drift = multipole->length / nSlices / 2;
  if (multipole->bore)
    /* KnL = d^nB/dx^n * L/(B.rho) = n! B(a)/a^n * L/(B.rho) */
    KnL = dfactorial(multipole->order) * multipole->BTipL / ipow(multipole->bore, multipole->order) *
          (particleCharge / (particleMass * c_mks * Po)) * multipole->factor / nSlices;
  else
    KnL = multipole->KnL * multipole->factor / nSlices;

  if (KnL == 0) {
    if (multipole->length == 0)
      return n_part;
    exactDrift(particle, n_part, multipole->length);
    return n_part;
  }

  if (multipole->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  else
    rad_coef = 0;

  if (multipole->dx || multipole->dy || multipole->dz)
    offsetBeamCoordinatesForMisalignment(particle, n_part, multipole->dx, multipole->dy, multipole->dz);
  if (multipole->tilt)
    rotateBeamCoordinatesForMisalignment(particle, n_part, multipole->tilt);

  i_top = n_part - 1;
  multipoleKicksDone += (i_top + 1) * nSlices * 4;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!(coord = particle[i_part])) {
      printf("null coordinate pointer for particle %ld (multipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("null accepted coordinates pointer for particle %ld (multipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }
    if (KnL == 0) {
      coord[4] += multipole->length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      coord[0] += multipole->length * coord[1];
      coord[2] += multipole->length * coord[3];
      continue;
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
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
#endif
    if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
        FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }

    /* calculate initial canonical momenta */
    convertSlopesToMomenta(&qx, &qy, xp, yp, dp);
    is_lost = 0;
    for (i_kick = 0; i_kick < nSlices; i_kick++) {
      if (drift) {
        x += xp * drift * (i_kick ? 2 : 1);
        y += yp * drift * (i_kick ? 2 : 1);
        if (multipole->expandHamiltonian)
          s += (i_kick ? 2 : 1) * drift * (1 + (sqr(xp) + sqr(yp)) / 2);
        else
          s += (i_kick ? 2 : 1) * drift * sqrt(1 + sqr(xp) + sqr(yp));
      }
      fillPowerArray(x, xpow, order);
      fillPowerArray(y, ypow, order);
      /* now sum up the terms for the multipole expansion */
      for (i = sum_Fx = sum_Fy = 0; i <= order; i++) {
        if (ODD(i))
          sum_Fx += coef[i] * xpow[order - i] * ypow[i];
        else
          sum_Fy += coef[i] * xpow[order - i] * ypow[i];
      }
      /* apply kicks canonically */
      qx -= KnL * sum_Fy;
      qy += KnL * sum_Fx;
      if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp)) {
        is_lost = 1;
        break;
      }
      if (rad_coef && drift) {
        qx /= (1 + dp);
        qy /= (1 + dp);
        dp -= rad_coef * sqr(KnL * (1 + dp)) * (sqr(sum_Fy) + sqr(sum_Fx)) * sqrt(1 + sqr(xp) + sqr(yp)) / (2 * drift);
        qx *= (1 + dp);
        qy *= (1 + dp);
        if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp)) {
          is_lost = 1;
          break;
        }
      }
    }
    if (drift && !is_lost) {
      /* go through final drift */
      x += xp * drift;
      y += yp * drift;
      if (multipole->expandHamiltonian)
        s += drift * (1 + (sqr(xp) + sqr(yp)) / 2);
      else
        s += drift * sqrt(1 + sqr(xp) + sqr(yp));
    }

    coord[0] = x;
    coord[1] = xp;
    coord[2] = y;
    coord[3] = yp;
    coord[5] = dp;

    if (rad_coef) {
      p = Po * (1 + dp);
      beta1 = p / sqrt(sqr(p) + 1);
      coord[4] = beta1 * (coord[4] / beta0 + 2 * s / (beta0 + beta1));
    } else
      coord[4] += s;

#if defined(IEEE_MATH)
    if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
#endif
    if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
        FABS(xp) > slopeLimit || FABS(yp) > slopeLimit || is_lost) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
  }

  if (multipole->tilt)
    rotateBeamCoordinatesForMisalignment(particle, n_part, -multipole->tilt);
  if (multipole->dx || multipole->dy || multipole->dz)
    offsetBeamCoordinatesForMisalignment(particle, n_part, -multipole->dx, -multipole->dy, -multipole->dz);

  log_exit("multipole_tracking");
  return (i_top + 1);
}

long multipole_tracking2(
  double **particle,  /* initial/final phase-space coordinates */
  long n_part,        /* number of particles */
  ELEMENT_LIST *elem, /* element pointer */
  double p_error,     /* p_nominal/p_central */
  double Po,
  double **accepted,
  double z_start,
  /* From previous MAXAMP element */
  MAXAMP *maxamp,
  /* From previous APCONTOUR element with STICKY=1 */
  APCONTOUR *apcontour,
  /* from aperture_data command */
  APERTURE_DATA *apFileData,
  /* For return of accumulated change in sigmaDelta^2 */
  double *sigmaDelta2,
  /* if iSlice>=0, used for slice-by-slice integration */
  long iSlice) {
  double KnL[3] = {0, 0, 0};
  long order[3] = {0, 0, 0};
  short skew[3] = {0, 0, 0};
  double dx, dy, dz; /* offsets of the multipole center */
  long nSlices, n_kicks, integ_order, iOrder;
  long i_part, i_top;
  double *coord;
  double drift;
  double tilt, pitch, yaw, rad_coef, isr_coef, xkick, ykick, dzLoss = 0;
  KQUAD *kquad = NULL;
  KSEXT *ksext;
  KQUSE *kquse;
  KOCT *koct;
  DQCOR *dqcor;
  double lEffective = -1, lEnd = 0;
  double K0, J0;
  short doEndDrift = 0, malignMethod;

  MULTIPOLE_DATA *multData = NULL, *steeringMultData = NULL, *edgeMultData = NULL;
  long freeMultData = 0;
  MULT_APERTURE_DATA apertureData;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    i_part = gpu_multipole_tracking2(n_part, elem, p_error, Po, accepted, z_start, maxamp, apcontour, apFileData, sigmaDelta2, iSlice);
#  ifdef GPU_VERIFY
    startCpuTimer();
    multipole_tracking2(particle, n_part, elem, p_error, Po, accepted, z_start, maxamp, apcontour, apFileData, sigmaDelta2, iSlice);
    compareGpuCpu(n_part, "multipole_tracking2");
#  endif /* GPU_VERIFY */
    return i_part;
  }
#endif /* HAVE_GPU */

  log_entry("multipole_tracking2");

  if (!particle)
    bombTracking("particle array is null (multipole_tracking)");

  if (!elem)
    bombTracking("null element pointer (multipole_tracking2)");
  if (!elem->p_elem)
    bombTracking("null p_elem pointer (multipole_tracking2)");

  rad_coef = xkick = ykick = isr_coef = 0;
  pitch = yaw = tilt = 0;
  dx = dy = dz = 0;
  malignMethod = 0;

  switch (elem->type) {
  case T_KQUAD:
    kquad = ((KQUAD *)elem->p_elem);
    nSlices = kquad->nSlices;
    n_kicks = kquad->n_kicks;
    expandHamiltonian = kquad->expandHamiltonian;
    order[0] = 1;
    if ((lEffective = kquad->lEffective) <= 0)
      lEffective = kquad->length;
    else {
      lEnd = (kquad->length - lEffective) / 2;
      doEndDrift = 1;
    }
    if (kquad->bore)
      /* KnL = d^nB/dx^n * L/(B.rho) = n! B(a)/a^n * L/(B.rho) * (1+FSE) */
      KnL[0] = kquad->B / kquad->bore * (particleCharge / (particleMass * c_mks * Po)) * lEffective * (1 + kquad->fse);
    else
      KnL[0] = kquad->k1 * lEffective * (1 + kquad->fse);
    drift = lEffective;
    tilt = kquad->tilt;
    pitch = kquad->pitch;
    yaw = kquad->yaw;
    dx = kquad->dx;
    dy = kquad->dy;
    dz = kquad->dz;
    malignMethod = kquad->malignMethod;
    xkick = kquad->xkick * kquad->xKickCalibration;
    ykick = kquad->ykick * kquad->yKickCalibration;
    integ_order = kquad->integration_order;
    if (kquad->synch_rad)
      rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    if (!kquad->isr || (kquad->isr1Particle == 0 && n_part == 1))
      /* Minus sign indicates we accumulate into sigmaDelta^2 only, don't perturb particles */
      isr_coef *= -1;
    if (kquad->length < 1e-6 && (kquad->isr || kquad->synch_rad)) {
      rad_coef = isr_coef = 0; /* avoid unphysical results */
      printWarningForTracking("Quadrupole with length < 1e-6 has SYNCH_RAD=0 and ISR=0 forced to avoid unphysical results.",
                              NULL);
    }
    if (!kquad->multipolesInitialized) {
      /* read the data files for the error multipoles */
      readErrorMultipoleData(&(kquad->systematicMultipoleData),
                             kquad->systematic_multipoles, 0);
      readErrorMultipoleData(&(kquad->edgeMultipoleData),
                             kquad->edge_multipoles, 0);
      readErrorMultipoleData(&(kquad->randomMultipoleData),
                             kquad->random_multipoles, 0);
      readErrorMultipoleData(&(kquad->steeringMultipoleData),
                             kquad->steering_multipoles, 1);
      kquad->multipolesInitialized = 1;
    }
    if (!kquad->totalMultipolesComputed) {
      computeTotalErrorMultipoleFields(&(kquad->totalMultipoleData),
                                       &(kquad->systematicMultipoleData),
                                       kquad->systematicMultipoleFactor,
                                       &(kquad->edgeMultipoleData),
                                       NULL,
                                       &(kquad->randomMultipoleData),
                                       kquad->randomMultipoleFactor,
                                       &(kquad->steeringMultipoleData),
                                       kquad->steeringMultipoleFactor,
                                       KnL[0], 1, 1,
                                       kquad->minMultipoleOrder, kquad->maxMultipoleOrder);
      kquad->totalMultipolesComputed = 1;
    }
    multData = &(kquad->totalMultipoleData);
    edgeMultData = &(kquad->edgeMultipoleData);
    steeringMultData = &(kquad->steeringMultipoleData);
    /*
    print_multipole_data(multData, "KQUAD systematic");
    print_multipole_data(steeringMultData, "KQUAD steering");
    */
    break;
  case T_KSEXT:
    ksext = ((KSEXT *)elem->p_elem);
    n_kicks = ksext->n_kicks;
    nSlices = ksext->nSlices;
    expandHamiltonian = ksext->expandHamiltonian;
    order[0] = 2;
    if (ksext->bore)
      /* KnL = d^nB/dx^n * L/(B.rho) = n! B(a)/a^n * L/(B.rho) * (1+FSE) */
      KnL[0] = 2 * ksext->B / sqr(ksext->bore) * (particleCharge / (particleMass * c_mks * Po)) * ksext->length * (1 + ksext->fse);
    else
      KnL[0] = ksext->k2 * ksext->length * (1 + ksext->fse);
    drift = ksext->length;
    tilt = ksext->tilt;
    pitch = ksext->pitch;
    yaw = ksext->yaw;
    dx = ksext->dx;
    dy = ksext->dy;
    dz = ksext->dz;
    malignMethod = ksext->malignMethod;
    xkick = ksext->xkick * ksext->xKickCalibration;
    ykick = ksext->ykick * ksext->yKickCalibration;
    integ_order = ksext->integration_order;
    if (ksext->k1) {
      KnL[1] = ksext->k1 * ksext->length;
      order[1] = 1;
    }
    if (ksext->j1) {
      KnL[2] = ksext->j1 * ksext->length;
      order[2] = 1;
      skew[2] = 1;
    }
    if (ksext->synch_rad)
      rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    if (!ksext->isr || (ksext->isr1Particle == 0 && n_part == 1))
      /* Minus sign indicates we accumulate into sigmaDelta^2 only, don't perturb particles */
      isr_coef *= -1;
    if (ksext->length < 1e-6 && (ksext->isr || ksext->synch_rad)) {
      rad_coef = isr_coef = 0; /* avoid unphysical results */
      printWarningForTracking("Sextupole with length < 1e-6 has SYNCH_RAD=0 and ISR=0 forced to avoid unphysical results.",
                              NULL);
    }
    if (!ksext->multipolesInitialized) {
      /* read the data files for the error multipoles */
      readErrorMultipoleData(&(ksext->systematicMultipoleData),
                             ksext->systematic_multipoles, 0);
      readErrorMultipoleData(&(ksext->edgeMultipoleData),
                             ksext->edge_multipoles, 0);
      readErrorMultipoleData(&(ksext->randomMultipoleData),
                             ksext->random_multipoles, 0);
      readErrorMultipoleData(&(ksext->steeringMultipoleData),
                             ksext->steering_multipoles, 1);
      ksext->multipolesInitialized = 1;
    }
    if (!ksext->totalMultipolesComputed) {
      computeTotalErrorMultipoleFields(&(ksext->totalMultipoleData),
                                       &(ksext->systematicMultipoleData),
                                       ksext->systematicMultipoleFactor,
                                       &(ksext->edgeMultipoleData),
                                       NULL,
                                       &(ksext->randomMultipoleData),
                                       ksext->randomMultipoleFactor,
                                       &(ksext->steeringMultipoleData),
                                       ksext->steeringMultipoleFactor,
                                       KnL[0], 2, 1,
                                       ksext->minMultipoleOrder, ksext->maxMultipoleOrder);
      ksext->totalMultipolesComputed = 1;
    }
    multData = &(ksext->totalMultipoleData);
    edgeMultData = &(ksext->edgeMultipoleData);
    steeringMultData = &(ksext->steeringMultipoleData);
    break;
  case T_KOCT:
    koct = ((KOCT *)elem->p_elem);
    n_kicks = koct->n_kicks;
    nSlices = koct->nSlices;
    expandHamiltonian = koct->expandHamiltonian;
    order[0] = 3;
    if (koct->bore)
      /* KnL = d^nB/dx^n * L/(B.rho) = n! B(a)/a^n * L/(B.rho) * (1+FSE) */
      KnL[0] = 6 * koct->B / ipow3(koct->bore) * (particleCharge / (particleMass * c_mks * Po)) * koct->length * (1 + koct->fse);
    else
      KnL[0] = koct->k3 * koct->length * (1 + koct->fse);
    drift = koct->length;
    tilt = koct->tilt;
    pitch = koct->pitch;
    yaw = koct->yaw;
    dx = koct->dx;
    dy = koct->dy;
    dz = koct->dz;
    malignMethod = koct->malignMethod;
    integ_order = koct->integration_order;
    if (koct->synch_rad)
      rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    if (!koct->isr || (koct->isr1Particle == 0 && n_part == 1))
      /* Minus sign indicates we accumulate into sigmaDelta^2 only, don't perturb particles */
      isr_coef *= -1;
    if (koct->length < 1e-6 && (koct->isr || koct->synch_rad)) {
      rad_coef = isr_coef = 0; /* avoid unphysical results */
      printWarningForTracking("Octupole with length < 1e-6 has SYNCH_RAD=0 and ISR=0 forced to avoid unphysical results.",
                              NULL);
    }
    if (!koct->multipolesInitialized) {
      /* read the data files for the error multipoles */
      readErrorMultipoleData(&(koct->systematicMultipoleData),
                             koct->systematic_multipoles, 0);
      readErrorMultipoleData(&(koct->randomMultipoleData),
                             koct->random_multipoles, 0);
      koct->multipolesInitialized = 1;
    }
    if (!koct->totalMultipolesComputed) {
      computeTotalErrorMultipoleFields(&(koct->totalMultipoleData),
                                       &(koct->systematicMultipoleData),
                                       1.0,
                                       NULL,
                                       NULL,
                                       &(koct->randomMultipoleData),
                                       1.0,
                                       NULL,
                                       1.0,
                                       KnL[0], 3, 1,
                                       NULL, NULL);
      koct->totalMultipolesComputed = 1;
    }
    multData = &(koct->totalMultipoleData);
    break;
  case T_KQUSE:
    /* Implemented as a quadrupole with sextupole as a secondary multipole */
    kquse = ((KQUSE *)elem->p_elem);
    n_kicks = kquse->n_kicks;
    nSlices = kquse->nSlices;
    expandHamiltonian = kquse->expandHamiltonian;
    order[0] = 1;
    KnL[0] = kquse->k1 * kquse->length * (1 + kquse->fse1);
    drift = kquse->length;
    tilt = kquse->tilt;
    dx = kquse->dx;
    dy = kquse->dy;
    dz = kquse->dz;
    malignMethod = 0;
    integ_order = kquse->integration_order;
    if (kquse->synch_rad)
      rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    if (!kquse->isr || (kquse->isr1Particle == 0 && n_part == 1))
      /* Minus sign indicates we accumulate into sigmaDelta^2 only, don't perturb particles */
      isr_coef *= -1;
    if (kquse->length < 1e-6 && (kquse->isr || kquse->synch_rad)) {
      rad_coef = isr_coef = 0; /* avoid unphysical results */
      printWarningForTracking("KQUSE with length < 1e-6 has SYNCH_RAD=0 and ISR=0 forced to avoid unphysical results.", NULL);
    }
    KnL[1] = kquse->k2 * kquse->length * (1 + kquse->fse2);
    order[1] = 2;
    break;
  case T_DQCOR:
    dqcor = ((DQCOR *)elem->p_elem);
    n_kicks = 0;
    nSlices = dqcor->nSlices;
    expandHamiltonian = 0;
    KnL[0] = dqcor->k1 * dqcor->length * (1 + dqcor->fse);
    order[0] = 1;
    skew[0] = 0;
    KnL[1] = dqcor->j1 * dqcor->length * (1 + dqcor->fse);
    order[1] = 1;
    skew[1] = 1;
    tilt = dqcor->tilt;
    drift = dqcor->length;
    pitch = dqcor->pitch;
    yaw = dqcor->yaw;
    dx = dqcor->dx;
    dy = dqcor->dy;
    dz = dqcor->dz;
    malignMethod = dqcor->malignMethod;
    determineDQCorReferenceFrameTilt(dqcor);
    computeQuadSteeringStrengths(&K0, &J0,
				 dqcor->xKickReferenceFrame * dqcor->xKickCalibration,
				 dqcor->yKickReferenceFrame * dqcor->yKickCalibration,
				 dqcor->length, dqcor->K1ReferenceFrame);
    xkick = -K0*dqcor->length;
    ykick = -J0*dqcor->length;
    rotate_xy(&xkick, &ykick, -dqcor->phiReferenceFrame);
    integ_order = dqcor->integration_order;
    if (dqcor->synch_rad)
      rad_coef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    isr_coef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    if (!dqcor->isr || (dqcor->isr1Particle == 0 && n_part == 1))
      /* Minus sign indicates we accumulate into sigmaDelta^2 only, don't perturb particles */
      isr_coef *= -1;
    if (dqcor->length < 1e-6 && (dqcor->isr || dqcor->synch_rad)) {
      rad_coef = isr_coef = 0; /* avoid unphysical results */
      printWarningForTracking("DQCOR with length < 1e-6 has SYNCH_RAD=0 and ISR=0 forced to avoid unphysical results.",
                              NULL);
    }
    if (!dqcor->multipolesInitialized) {
      readErrorMultipoleData(&(dqcor->quadrupoleSystematicMultipoleData),
                             dqcor->quadrupoleSystematicMultipoles, 0);
      readErrorMultipoleData(&(dqcor->quadrupoleRandomMultipoleData),
                             dqcor->quadrupoleRandomMultipoles, 0);
      readErrorMultipoleData(&(dqcor->dipoleSystematicMultipoleData),
                             dqcor->dipoleSystematicMultipoles, 1);
      dqcor->multipolesInitialized = 1;
    }
    if (!dqcor->totalMultipolesComputed) {
      computeTotalErrorMultipoleFields(&(dqcor->totalMultipoleData),
                                       &(dqcor->quadrupoleSystematicMultipoleData),
				       dqcor->quadrupoleSystematicMultipoleFactor,
				       NULL,
                                       NULL,
                                       &(dqcor->quadrupoleRandomMultipoleData),
				       dqcor->quadrupoleRandomMultipoleFactor,
                                       &(dqcor->dipoleSystematicMultipoleData),
				       dqcor->dipoleSystematicMultipoleFactor,
                                       KnL[0], 1, 1,
                                       dqcor->minMultipoleOrder, dqcor->maxMultipoleOrder);
      dqcor->totalMultipolesComputed = 1;
    }
    multData = &(dqcor->totalMultipoleData);
    edgeMultData = NULL;
    steeringMultData = &(dqcor->dipoleSystematicMultipoleData);
    /*
    print_multipole_data(multData, "DQCOR systematic");
    print_multipole_data(steeringMultData, "DQCOR steering");
    */
    break;
  default:
    printf("error: multipole_tracking2() called for element %s--not supported!\n", elem->name);
    fflush(stdout);
    KnL[0] = dx = dy = dz = tilt = drift = 0;
    integ_order = order[0] = n_kicks = nSlices = 0;
    exitElegant(1);
    break;
  }
  if (multData && !multData->initialized)
    multData = NULL;

  if (order[0] <= 0)
    bombTracking("order <= 0 in multipole()");
  if (integ_order != 2 && integ_order != 4 && integ_order != 6)
    bombTracking("multipole integration_order must be 2, 4, or 6");

  for (iOrder = 0; iOrder < 3; iOrder++) {
    if (KnL[iOrder] && !expansion_coefficients(order[iOrder]))
      bombTracking("expansion_coefficients() returned null pointer (multipole_tracking)");
  }

  i_top = n_part - 1;

  if (n_kicks <= 0) {
    if (nSlices <= 0)
      bombTracking("N_KICKS<=0 and N_SLICES<=0 in multipole tracking");
  } else {
    if (integ_order > 2) {
      if ((nSlices = ceil(n_kicks / (1.0 * integ_order))) < 1)
        nSlices = 1;
      n_kicks = nSlices * integ_order;
    } else
      nSlices = n_kicks;
  }

  multipoleKicksDone += (i_top + 1) * n_kicks;
  if (multData)
    multipoleKicksDone += (i_top + 1) * n_kicks * multData->orders;

  setupMultApertureData(&apertureData, -tilt, apcontour, maxamp, apFileData, NULL, z_start + drift / 2, elem);

  if (iSlice <= 0) {
    offsetParticlesForMisalignment(malignMethod, particle, n_part, dx, dy, dz, pitch, yaw, 0.0, tilt, 0.0, drift, 1);

    if (doEndDrift) {
      exactDrift(particle, n_part, lEnd);
    }

    /* Fringe treatment, if any */
    switch (elem->type) {
    case T_KQUAD:
      if (kquad->edge1_effects > 0)
        quadFringe(particle, n_part, kquad->k1, kquad->fringeIntM, kquad->fringeIntP, kquad->length < 0, -1,
                   kquad->edge1_effects, kquad->edge1Linear, kquad->edge1NonlinearFactor);
      break;
    default:
      break;
    }
  }

  if (sigmaDelta2)
    *sigmaDelta2 = 0;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!(coord = particle[i_part])) {
      printf("null coordinate pointer for particle %ld (multipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("null accepted coordinates pointer for particle %ld (multipole_tracking)", i_part);
      fflush(stdout);
      abort();
    }

    if (!integrate_kick_multipole_ordn(coord, dx, dy, xkick, ykick,
                                       Po, rad_coef, isr_coef,
                                       order, KnL, skew,
                                       nSlices, iSlice, drift, integ_order,
                                       multData, edgeMultData, steeringMultData,
                                       &apertureData, &dzLoss, sigmaDelta2,
                                       elem->type == T_KQUAD ? kquad->radial : 0, tilt)) {
      swapParticles(particle[i_part], particle[i_top]);
      if (globalLossCoordOffset > 0) {
        double X, Y, Z, theta;
        convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta, GLOBAL_LOCAL_MODE_DZ, particle[i_top], elem,
                                        dzLoss, 0, 0);
        particle[i_top][globalLossCoordOffset + 0] = X;
        particle[i_top][globalLossCoordOffset + 1] = Z;
        particle[i_top][globalLossCoordOffset + 2] = theta;
      }
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start + dzLoss;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
  }
  if (sigmaDelta2)
    *sigmaDelta2 /= i_top + 1;

  if (iSlice < 0 || iSlice == (nSlices - 1)) {
    /* Fringe treatment, if any */
    switch (elem->type) {
    case T_KQUAD:
      if (kquad->edge2_effects > 0)
        quadFringe(particle, n_part, kquad->k1, kquad->fringeIntM, kquad->fringeIntP, kquad->length < 0, 1,
                   kquad->edge2_effects, kquad->edge2Linear, kquad->edge2NonlinearFactor);
      break;
    default:
      break;
    }

    if (doEndDrift) {
      exactDrift(particle, n_part, lEnd);
    }

    if (dx || dy || dz || tilt || pitch || yaw) 
      offsetParticlesForMisalignment(malignMethod, particle, n_part, dx, dy, dz, pitch, yaw, 0.0, tilt, 0.0, drift, 2);
  }

  if (freeMultData && multData->copy) {
    if (multData->order)
      free(multData->order);
    if (multData->KnL)
      free(multData->KnL);
    free(multData);
  }

  log_exit("multipole_tracking2");
  expandHamiltonian = 0;
  return (i_top + 1);
}

/* BETA is 2^(1/3) */
#define BETA 1.25992104989487316477

#define FILLX fillPowerArray
#define FILLY fillPowerArray

#define DO_MKICKS_RET apply_canonical_multipole_kicks_ret
#define DO_MKICKS_NORET apply_canonical_multipole_kicks_noret

int integrate_kick_multipole_ordn(double *coord, double dx, double dy, double xkick, double ykick,
                                  double Po, double rad_coef, double isr_coef,
                                  long *order, double *KnL, short *skew,
                                  long n_parts, long i_part, double drift,
                                  long integration_order,
                                  MULTIPOLE_DATA *multData, MULTIPOLE_DATA *edgeMultData,
                                  MULTIPOLE_DATA *steeringMultData,
                                  MULT_APERTURE_DATA *apData,
                                  double *dzLoss, double *sigmaDelta2,
                                  long radial,
                                  double refTilt /* used for obstruction evaluation only */
) {
  double p, qx, qy, beta0, beta1, dp, s;
  double x, y, xp, yp, delta_qx, delta_qy;
  long i_kick, step, imult, iOrder;
  double dsh;
  long maxOrder;
  double *xpow, *ypow;

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

  drift = drift / n_parts;
  xkick = xkick / n_parts;
  ykick = ykick / n_parts;

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
  if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
      FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
    return 0;
  }

  /* calculate initial canonical momenta */
  convertSlopesToMomenta(&qx, &qy, xp, yp, dp);

  maxOrder = findMaximumOrder(order[0], order[1] > order[2] ? order[1] : order[2], edgeMultData, steeringMultData, multData);
  xpow = tmalloc(sizeof(*xpow) * (maxOrder + 1));
  ypow = tmalloc(sizeof(*ypow) * (maxOrder + 1));
  // Core loop only goes to 3, ignore higher KnL
  double KnLp[3];
  for (int i = 0; i < 3; i++)
    KnLp[i] = KnL[i] / n_parts;

  if (i_part <= 0) {
    if (edgeMultData && edgeMultData->orders) {
      FILLX(x, xpow, maxOrder);
      FILLY(y, ypow, maxOrder);
      for (imult = 0; imult < edgeMultData->orders; imult++) {
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                        edgeMultData->order[imult],
                                        edgeMultData->KnL[imult], 0);
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                        edgeMultData->order[imult],
                                        edgeMultData->JnL[imult], 1);
      }
    }
  }

  /* We must do this in case steering or edge multipoles were run. We do it even if not in order
   * to avoid numerical precision issues that may subtly change the results
   */
  if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp))
    return 0;

  *dzLoss = 0;
  for (i_kick = 0; i_kick < n_parts; i_kick++) {
    if ((apData && !checkMultAperture(x + dx, y + dy, drift*i_kick, apData)) ||
        insideObstruction_xyz(x, xp, y, yp, coord[particleIDIndex],
                              globalLossCoordOffset > 0 ? coord + globalLossCoordOffset : NULL,
                              refTilt, GLOBAL_LOCAL_MODE_SEG, 0.0, i_kick, n_parts)) {
      coord[0] = x;
      coord[2] = y;
      return 0;
    }
    delta_qx = delta_qy = 0;
    for (step = 0; step < nSubsteps; step++) {
      if (drift) {
        dsh = drift * driftFrac[step];
        x += xp * dsh;
        y += yp * dsh;
        if (expandHamiltonian) {
          s += dsh * (1 + (sqr(xp) + sqr(yp)) / 2);
        } else {
          s += dsh * sqrt(1 + sqr(xp) + sqr(yp));
        }
        *dzLoss += dsh;
      }

      if (!kickFrac[step])
        break;

      FILLX(x, xpow, maxOrder);
      FILLY(y, ypow, maxOrder);

      delta_qx = delta_qy = 0;

      if (!radial) {
        for (iOrder = 0; iOrder < 3; iOrder++)
          if (KnL[iOrder])
            DO_MKICKS_RET(&qx, &qy, &delta_qx, &delta_qy, xpow, ypow,
                          order[iOrder], KnLp[iOrder] * kickFrac[step], skew[iOrder]);
      } else {
        applyRadialCanonicalMultipoleKicks(&qx, &qy, &delta_qx, &delta_qy, xpow, ypow,
                                           order[0], KnLp[0] * kickFrac[step], 0);
      }
      if (xkick)
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow, 0, -xkick * kickFrac[step], 0);
      if (ykick)
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow, 0, -ykick * kickFrac[step], 1);
      if (steeringMultData && steeringMultData->orders) {
        /* apply steering corrector multipoles */
        for (imult = 0; imult < steeringMultData->orders; imult++) {
          if (steeringMultData->KnL[imult])
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      steeringMultData->order[imult],
                      steeringMultData->KnL[imult] * xkick * kickFrac[step], 0);
          if (steeringMultData->JnL[imult])
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      steeringMultData->order[imult],
                      steeringMultData->JnL[imult] * ykick * kickFrac[step], 1);
        }
      }

      if (multData) {
        /* do kicks for spurious multipoles */
        for (imult = 0; imult < multData->orders; imult++) {
          if (multData->KnL && multData->KnL[imult]) {
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      multData->order[imult],
                      multData->KnL[imult] * kickFrac[step] / n_parts,
                      0);
          }
          if (multData->JnL && multData->JnL[imult]) {
            DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                      multData->order[imult],
                      multData->JnL[imult] * kickFrac[step] / n_parts,
                      1);
          }
        }
      }

      if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp))
        return 0;

      if ((rad_coef || isr_coef) && drift) {
        double deltaFactor, F2, dsFactor, dsISRFactor;
        qx /= (1 + dp);
        qy /= (1 + dp);
        deltaFactor = sqr(1 + dp);
        /* delta_qx and delta_qy are for the last step and have kickFrac[step-1] included, so remove it */
        delta_qx /= kickFrac[step];
        delta_qy /= kickFrac[step];
#if TURBO_RECIPROCALS
        F2 = (sqr(delta_qx - xkick) + sqr(delta_qy + ykick)) / sqr(drift);
#else
        F2 = sqr(delta_qx / drift - xkick / drift) + sqr(delta_qy / drift + ykick / drift);
#endif
        delta_qx = 0;
        delta_qy = 0;
        dsFactor = sqrt(1 + sqr(xp) + sqr(yp));
        dsISRFactor = dsFactor * drift / (nSubsteps - 1); /* recall that kickFrac may be negative */
        dsFactor *= drift * kickFrac[step];               /* that's ok here, since we don't take sqrt */
        if (rad_coef)
          dp -= rad_coef * deltaFactor * F2 * dsFactor;
        if (isr_coef > 0)
          dp -= isr_coef * deltaFactor * pow(F2, 0.75) * sqrt(dsISRFactor) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
        if (sigmaDelta2)
          *sigmaDelta2 += sqr(isr_coef * deltaFactor) * pow(F2, 1.5) * dsISRFactor;
        qx *= (1 + dp);
        qy *= (1 + dp);
        if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp))
          return 0;
      }
    }

    if (i_part >= 0)
      break;
  }

  if ((apData && !checkMultAperture(x + dx, y + dy, drift*i_kick, apData)) ||
      insideObstruction_xyz(x, xp, y, yp, coord[particleIDIndex],
                            globalLossCoordOffset > 0 ? coord + globalLossCoordOffset : NULL,
                            refTilt, GLOBAL_LOCAL_MODE_SEG, 0.0, i_kick, n_parts)) {
    coord[0] = x;
    coord[2] = y;
    return 0;
  }

  if (i_part < 0 || i_part == (n_parts - 1)) {
    if (edgeMultData && edgeMultData->orders) {
      FILLX(x, xpow, maxOrder);
      FILLY(y, ypow, maxOrder);
      for (imult = 0; imult < edgeMultData->orders; imult++) {
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                        edgeMultData->order[imult],
                                        edgeMultData->KnL[imult], 0);
        DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                        edgeMultData->order[imult],
                                        edgeMultData->JnL[imult], 1);
      }
    }
  }

  if (!convertMomentaToSlopes(&xp, &yp, qx, qy, dp))
    return 0;

  free(xpow);
  free(ypow);

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

#if defined(IEEE_MATH)
  if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
    return 0;
  }
#endif
  if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
      FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
    return 0;
  }
  return 1;
}

void applyRadialCanonicalMultipoleKicks(double *qx, double *qy,
                                        double *sum_Fx_return, double *sum_Fy_return,
                                        double *xpow, double *ypow,
                                        long order, double KnL, long skew) {
  long i;
  double sum_Fx, sum_Fy;
  double *coef;

  if (sum_Fx_return)
    *sum_Fx_return = 0;
  if (sum_Fy_return)
    *sum_Fy_return = 0;
  coef = expansion_coefficients(order);
  i = 0;
  /* now sum up the terms for the multipole expansion */
  for (sum_Fx = sum_Fy = 0; i <= order; i++) {
    if (ODD(i))
      sum_Fx -= coef[i - 1] * xpow[order - i] * ypow[i];
    else
      sum_Fy += coef[i] * xpow[order - i] * ypow[i];
  }
  if (skew) {
    SWAP_DOUBLE(sum_Fx, sum_Fy);
    sum_Fx = -sum_Fx;
  }
  /* add the kicks */
  *qx -= KnL * sum_Fy;
  *qy += KnL * sum_Fx;
  if (sum_Fx_return)
    *sum_Fx_return = sum_Fx;
  if (sum_Fy_return)
    *sum_Fy_return = sum_Fy;
}

void randomizeErrorMultipoleFields(MULTIPOLE_DATA *randomMult) {
  long i;
  double nFactorial, rpow, rn1, rn2;
  if (!randomMult || randomMult->randomized)
    return;
  for (i = 0; i < randomMult->orders; i++) {
    nFactorial = dfactorial(randomMult->order[i]);
    rpow = ipow(randomMult->referenceRadius, randomMult->order[i]);
    rn1 = gauss_rn_lim(0.0, 1.0, 2.0, random_1_elegant);
    rn2 = gauss_rn_lim(0.0, 1.0, 2.0, random_1_elegant);
    randomMult->anMod[i] = randomMult->an[i] * nFactorial * rn1 / rpow;
    randomMult->bnMod[i] = randomMult->bn[i] * nFactorial * rn2 / rpow;
  }
#ifdef DEBUG_RANDOMIZE
  printf("randomized multipoles\n");
#endif
  randomMult->randomized = 1;
}

void computeTotalErrorMultipoleFields(MULTIPOLE_DATA *totalMult,
                                      MULTIPOLE_DATA *systematicMult,
                                      double systematicMultFactor,
                                      MULTIPOLE_DATA *edge1Mult,
                                      MULTIPOLE_DATA *edge2Mult,
                                      MULTIPOLE_DATA *randomMult,
                                      double randomMultFactor,
                                      MULTIPOLE_DATA *steeringMult,
                                      double steeringMultFactor,
                                      double KmL, long rootOrder0,
                                      long orderCheck,
                                      short *minOrder, /* normal, skew */
                                      short *maxOrder  /* normal, skew */
) {
  long i, edge;
  MULTIPOLE_DATA *edgeMult;
  double sFactor = 0.0, rFactor = 0.0;
  long rootOrder[3]; /* systematic body, systematic edge, random */

  rootOrder[0] = rootOrder[1] = rootOrder[2] = rootOrder0;
  if (orderCheck) {
    if (systematicMult && systematicMult->initialized && systematicMult->referenceOrder >= 0 && rootOrder0 != systematicMult->referenceOrder)
      bombElegantVA("root order mismatch for multipole data file %s---expected %ld but found %ld in referenceOrder parameter",
                    systematicMult->filename, rootOrder0, systematicMult->referenceOrder);
    if (edge1Mult && edge1Mult->initialized && edge1Mult->referenceOrder >= 0 && rootOrder0 != edge1Mult->referenceOrder)
      bombElegantVA("root order mismatch for multipole data file %s---expected %ld but found %ld in referenceOrder parameter",
                    edge1Mult->filename, rootOrder0, edge1Mult->referenceOrder);
    if (edge2Mult && edge2Mult->initialized && edge2Mult->referenceOrder >= 0 && rootOrder0 != edge2Mult->referenceOrder)
      bombElegantVA("root order mismatch for multipole data file %s---expected %ld but found %ld in referenceOrder parameter",
                    edge2Mult->filename, rootOrder0, edge2Mult->referenceOrder);
    if (randomMult && randomMult->initialized && randomMult->referenceOrder >= 0 && rootOrder0 != randomMult->referenceOrder)
      bombElegantVA("root order mismatch for multipole data file %s---expected %ld but found %ld in referenceOrder parameter",
                    randomMult->filename, rootOrder0, randomMult->referenceOrder);
  } else {
    if (systematicMult && systematicMult->referenceOrder >= 0)
      rootOrder[0] = systematicMult->referenceOrder;
    if (edge1Mult && edge1Mult->referenceOrder >= 0)
      rootOrder[1] = edge1Mult->referenceOrder;
    if (edge2Mult && edge2Mult->referenceOrder >= 0 && edge2Mult->referenceOrder > rootOrder[1])
      rootOrder[1] = edge2Mult->referenceOrder;
    if (randomMult && randomMult->referenceOrder >= 0)
      rootOrder[2] = randomMult->referenceOrder;
  }
  if (steeringMult && steeringMult->initialized && steeringMult->referenceOrder >= 0 && steeringMult->referenceOrder != 0)
    bombElegantVA("root order error for multipole data file %s---expected 0 but found %ld in referenceOrder parameter",
                  randomMult->filename, randomMult->referenceOrder);

  if (!totalMult->initialized) {
    totalMult->initialized = 1;
    /* make a list of unique orders for random and systematic multipoles */
    if (systematicMult->orders && randomMult->orders &&
        systematicMult->orders != randomMult->orders) {
      fprintf(stderr, "Issue with files %s and %s\n",
              systematicMult->filename, randomMult->filename);
      bombTracking("The number of systematic and random multipole error orders must be the same for any given element");
    }
    if (systematicMult->orders)
      totalMult->orders = systematicMult->orders;
    else
      totalMult->orders = randomMult->orders;
    if (totalMult->orders) {
      if (!(totalMult->order = SDDS_Malloc(sizeof(*totalMult->order) * (totalMult->orders))))
        bombTracking("memory allocation failure (computeTotalMultipoleFields)");
      if (systematicMult->orders &&
          (!(systematicMult->anMod = SDDS_Malloc(sizeof(*systematicMult->anMod) * systematicMult->orders)) ||
           !(systematicMult->bnMod = SDDS_Malloc(sizeof(*systematicMult->bnMod) * systematicMult->orders)) ||
           !(systematicMult->KnL = SDDS_Malloc(sizeof(*systematicMult->KnL) * systematicMult->orders)) ||
           !(systematicMult->JnL = SDDS_Malloc(sizeof(*systematicMult->JnL) * systematicMult->orders))))
        bombTracking("memory allocation failure (computeTotalMultipoleFields)");
      if (randomMult->orders &&
          (!(randomMult->anMod = SDDS_Malloc(sizeof(*randomMult->anMod) * randomMult->orders)) ||
           !(randomMult->bnMod = SDDS_Malloc(sizeof(*randomMult->bnMod) * randomMult->orders)) ||
           !(randomMult->KnL = SDDS_Malloc(sizeof(*randomMult->KnL) * randomMult->orders)) ||
           !(randomMult->JnL = SDDS_Malloc(sizeof(*randomMult->JnL) * randomMult->orders))))
        bombTracking("memory allocation failure (computeTotalMultipoleFields");
      if (!(totalMult->KnL = SDDS_Malloc(sizeof(*totalMult->KnL) * totalMult->orders)) ||
          !(totalMult->JnL = SDDS_Malloc(sizeof(*totalMult->JnL) * totalMult->orders)))
        bombTracking("memory allocation failure (computeTotalMultipoleFields)");
      for (i = 0; i < totalMult->orders; i++) {
        if (systematicMult->orders && randomMult->orders &&
            systematicMult->order[i] != randomMult->order[i])
          bombTracking("multipole orders in systematic and random lists must match up for any given element.");
        if (systematicMult->orders) {
          totalMult->order[i] = systematicMult->order[i];
          systematicMult->anMod[i] = systematicMult->an[i] * dfactorial(systematicMult->order[i]) /
                                     ipow(systematicMult->referenceRadius, systematicMult->order[i]);
          systematicMult->bnMod[i] = systematicMult->bn[i] * dfactorial(systematicMult->order[i]) /
                                     ipow(systematicMult->referenceRadius, systematicMult->order[i]);
        } else {
          totalMult->order[i] = randomMult->order[i];
          /* anMod and bnMod will be computed later for randomized multipoles */
        }
      }
    }
    for (edge = 0; edge < 2; edge++) {
      edgeMult = edge ? edge2Mult : edge1Mult;
      if (edgeMult && edgeMult->orders) {
        if (!(edgeMult->anMod = SDDS_Malloc(sizeof(*edgeMult->anMod) * edgeMult->orders)) ||
            !(edgeMult->bnMod = SDDS_Malloc(sizeof(*edgeMult->bnMod) * edgeMult->orders)) ||
            !(edgeMult->KnL = SDDS_Malloc(sizeof(*edgeMult->KnL) * edgeMult->orders)) ||
            !(edgeMult->JnL = SDDS_Malloc(sizeof(*edgeMult->JnL) * edgeMult->orders)))
          bombTracking("memory allocation failure (computeTotalMultipoleFields");
        for (i = 0; i < edgeMult->orders; i++) {
          edgeMult->anMod[i] = edgeMult->an[i] * dfactorial(edgeMult->order[i]) /
                               ipow(edgeMult->referenceRadius, edgeMult->order[i]);
          edgeMult->bnMod[i] = edgeMult->bn[i] * dfactorial(edgeMult->order[i]) /
                               ipow(edgeMult->referenceRadius, edgeMult->order[i]);
        }
      }
    }
  }

  if (randomMult->orders)
    randomizeErrorMultipoleFields(randomMult);

  /* body multipoles:
   * compute normal (KnL) and skew (JnL) from an and bn
   * KnL = an*n!/r^n*(KmL*r^m/m!), 
   * JnL = bn*n!/r^n*(KmL*r^m/m!), where m is the root order 
   * of the magnet with strength KmL
   */
  if (systematicMult->orders)
    sFactor = KmL / dfactorial(rootOrder[0]) * ipow(systematicMult->referenceRadius, rootOrder[0]) * systematicMultFactor;
  if (randomMult->orders)
    rFactor = KmL / dfactorial(rootOrder[2]) * ipow(randomMult->referenceRadius, rootOrder[2]) * randomMultFactor;
  for (i = 0; i < totalMult->orders; i++) {
    totalMult->KnL[i] = totalMult->JnL[i] = 0;
    if (systematicMult->orders) {
      totalMult->KnL[i] += sFactor * systematicMult->anMod[i];
      totalMult->JnL[i] += sFactor * systematicMult->bnMod[i];
    }
    if (randomMult->orders) {
      totalMult->KnL[i] += rFactor * randomMult->anMod[i];
      totalMult->JnL[i] += rFactor * randomMult->bnMod[i];
    }
    if ((minOrder && minOrder[0] >= 0 && totalMult->order[i] < minOrder[0]) || (maxOrder && maxOrder[0] >= 0 && totalMult->order[i] > maxOrder[0]))
      totalMult->KnL[i] = 0;
    if ((minOrder && minOrder[1] >= 0 && totalMult->order[i] < minOrder[1]) || (maxOrder && maxOrder[1] >= 0 && totalMult->order[i] > maxOrder[1]))
      totalMult->JnL[i] = 0;
  }

  for (edge = 0; edge < 2; edge++) {
    edgeMult = edge ? edge2Mult : edge1Mult;
    if (edgeMult && edgeMult->orders) {
      /* edge multipoles: 
       * compute normal (KnL) and skew (JnL) from an and bn
       * KnL = an*n!/r^n*(KmL*r^m/m!), 
       * JnL = bn*n!/r^n*(KmL*r^m/m!), where m is the root order 
       * of the magnet with strength KmL
       */
      sFactor = KmL / dfactorial(rootOrder[1]) * ipow(edgeMult->referenceRadius, rootOrder[1]) * systematicMultFactor;
      for (i = 0; i < edgeMult->orders; i++) {
        edgeMult->KnL[i] = sFactor * edgeMult->anMod[i];
        edgeMult->JnL[i] = sFactor * edgeMult->bnMod[i];
        if ((minOrder && minOrder[0] >= 0 && totalMult->order[i] < minOrder[0]) || (maxOrder && maxOrder[0] >= 0 && totalMult->order[i] > maxOrder[0]))
          edgeMult->KnL[i] = 0;
        if ((minOrder && minOrder[1] >= 0 && totalMult->order[i] < minOrder[1]) || (maxOrder && maxOrder[1] >= 0 && totalMult->order[i] > maxOrder[1]))
          edgeMult->JnL[i] = 0;
      }
    }
  }

  if (steeringMult) {
    /* same for steering multipoles, but compute KnL/theta and JnL/theta (in this case m=0) */
    for (i = 0; i < steeringMult->orders; i++) {
      steeringMult->KnL[i] =
        -1 * steeringMult->an[i] * dfactorial(steeringMult->order[i]) / ipow(steeringMult->referenceRadius, steeringMult->order[i]) *
        steeringMultFactor;
      steeringMult->JnL[i] =
        -1 * steeringMult->bn[i] * dfactorial(steeringMult->order[i]) / ipow(steeringMult->referenceRadius, steeringMult->order[i]) *
        steeringMultFactor;
      if ((minOrder && minOrder[0] >= 0 && totalMult->order[i] < minOrder[0]) || (maxOrder && maxOrder[0] >= 0 && totalMult->order[i] > maxOrder[0]))
        steeringMult->KnL[i] = 0;
      if ((minOrder && minOrder[1] >= 0 && totalMult->order[i] < minOrder[1]) || (maxOrder && maxOrder[1] >= 0 && totalMult->order[i] > maxOrder[1]))
        steeringMult->JnL[i] = 0;
    }
  }
}

void setupMultApertureData
(
 MULT_APERTURE_DATA *apertureData,
  /* used to undo the tilt of the element when it's done for computational reasons, e.g., negative bend with TILT+=PI */
 double reverseTilt,
 APCONTOUR *apContour,
 MAXAMP *maxamp, 
 APERTURE_DATA *apFileData,      /* global aperture data vs s, from aperture_data command */
 APERTURE_DATA *localApFileData, /* local aperture data vs s (s=0 is start of element), from element definition */
 double zPosition,                /* arc length of reference location */
 ELEMENT_LIST *eptr
 ) {
  double x_max, y_max;

  apertureData->apContour = apContour;
  apertureData->localAperture = localApFileData;
  apertureData->reverseTilt = reverseTilt;
  apertureData->reverseTiltCS[0] = cos(reverseTilt);
  apertureData->reverseTiltCS[1] = sin(reverseTilt);
  apertureData->eptr = eptr;
  
  /* zPosition=_start+drift/2 */
  apertureData->xCen = apertureData->yCen = 0;
  x_max = y_max = apertureData->xMax = apertureData->yMax = 0;
  apertureData->elliptical = 0;
  apertureData->present = apertureData->openSide = 0;

  if (maxamp) {
    x_max = apertureData->xMax = maxamp->x_max;
    y_max = apertureData->yMax = maxamp->y_max;
    apertureData->elliptical = maxamp->elliptical;
    apertureData->present = x_max > 0 || y_max > 0;
    apertureData->xExponent =
      apertureData->yExponent = maxamp->exponent;
    if (maxamp->yExponent)
      apertureData->yExponent = maxamp->yExponent;
    apertureData->openSide = determineOpenSideCode(maxamp->openSide);
  }

  if (apFileData && apFileData->initialized) {
    /* If there is file-based aperture data, it may override MAXAMP data. */
    double xCenF, yCenF, xMaxF, yMaxF;
    apertureData->present = 1;
    if (interpolateApertureData(zPosition, apFileData,
                                &xCenF, &yCenF, &xMaxF, &yMaxF)) {
      if (x_max <= 0 || (x_max > fabs(xCenF + xMaxF) && x_max > fabs(xCenF - xMaxF))) {
        apertureData->xMax = xMaxF;
        apertureData->xCen = xCenF;
        apertureData->elliptical = 0;
        apertureData->openSide = 0;
      }
      if (y_max <= 0 || (y_max > fabs(yCenF + yMaxF) && y_max > fabs(yCenF - yMaxF))) {
        apertureData->yMax = yMaxF;
        apertureData->yCen = yCenF;
        apertureData->elliptical = 0;
        apertureData->openSide = 0;
      }
    }
  }
}

long checkMultAperture(double xInput, double yInput, double zLocal, MULT_APERTURE_DATA *apData) {
  double xa, yb;
  double x, y;
  if (!apData)
    return 1;

  x = xInput - apData->xCen;
  y = yInput - apData->yCen;

  if (apData->reverseTilt) {
    double x0, y0;
    x0 = x;
    y0 = y;
    x = x0 * apData->reverseTiltCS[0] + y0 * apData->reverseTiltCS[1];
    y = -x0 * apData->reverseTiltCS[1] + y0 * apData->reverseTiltCS[0];
  }

  if (apData->localAperture) {
    double xCenF, yCenF, xMaxF, yMaxF;
    if (interpolateApertureData(zLocal, apData->localAperture, &xCenF, &yCenF, &xMaxF, &yMaxF) && 
        (fabs(x-xCenF)>=xMaxF || fabs(y-yCenF)>=yMaxF))
      return 0;
  }

  if (apData->elliptical == 0 || apData->xMax <= 0 || apData->yMax <= 0) {
    /* rectangular or one-dimensional */
    if ((apData->xMax > 0 && fabs(x) > apData->xMax) ||
        (apData->yMax > 0 && fabs(y) > apData->yMax)) {
      if (apData->openSide == 0 ||
          evaluateLostWithOpenSides(apData->openSide, x, y, apData->xMax, apData->yMax))
        return 0;
    }
  } else {
    /* Elliptical or super-elliptical */
    xa = x / apData->xMax;
    yb = y / apData->yMax;
    if ((ipow(xa, apData->xExponent) + ipow(yb, apData->yExponent)) >= 1) {
      if (apData->openSide == 0 ||
          evaluateLostWithOpenSides(apData->openSide, x, y, apData->xMax, apData->yMax))
        return 0;
    }
  }

  if (apData->apContour && !checkApContour(x, y, apData->apContour, apData->eptr))
    return 0;

  return 1;
}

