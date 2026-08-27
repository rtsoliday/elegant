/*************************************************************************\
* Copyright (c) 2013 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2013 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "table.h"
#include "fftpackC.h"

void set_up_lrwake(LRWAKE *wakeData, RUN *run, long pass, long particles, CHARGE *charge, long nBunches);

void index_bunch_assignments(double **part, int64_t np, long idSlotsPerBunch, double P0,
                             /* return data: */
                             double **time,     /* (*time)[ip] is the arrival time of ip-th particle */
                             long **ibParticle, /* (*ibParticle)[ip] is the bunch assignment for ip-th particle */
                             int64_t ***ipBunch,   /* (*ipBunch)[ib][i] is the ip value of the i-th particle in the ib-th bunch */
                             int64_t **npBunch,    /* (*npBunch)[ib] is the number of particles in the ib-th bunch */
                             long *nBunches,    /* *nBunches is the number of bunches */
                             /* input data from previous call */
                             long lastNBunches /* Supply this only if the calling algorithm insists that the number of bunches not change */
) {
  int64_t ip;
  long ib;
  long ibMin, ibMax;
#if USE_MPI
  long ibMinGlobal, ibMaxGlobal;
#endif

#ifdef DEBUG
  printf("index_bunch_assignments called, np=%" PRId64 ", idSlotsPerBunch=%ld, lastNBunches=%ld\n",
         np, idSlotsPerBunch, lastNBunches);
  printf("pointer check: part %s, time %s, ibParticle %s, ipBunch %s, npBunch %s, nBunches %s\n",
         part ? "ok" : "NULL",
         time ? "ok" : "NULL",
         ibParticle ? "ok" : "NULL",
         ipBunch ? "ok" : "NULL",
         npBunch ? "ok" : "NULL",
         nBunches ? "ok" : "NULL");
  fflush(stdout);
#endif

  if (trajectoryTracking) {
    /* when tracking for trajectory, ignore bunch assignments */
    idSlotsPerBunch = -1;
  }

  if (idSlotsPerBunch <= 0) {
    ibMin = 0;
    *nBunches = 1;
#ifdef DEBUG
    printf("index_bunch_assignments: only one bunch\n");
    fflush(stdout);
#endif
  } else {
    ibMin = LONG_MAX;
    ibMax = LONG_MIN;
    for (ip = 0; ip < np; ip++) {
      ib = part[ip][bunchIndex];
      if (ib < ibMin)
        ibMin = ib;
      if (ib > ibMax)
        ibMax = ib;
    }
#if USE_MPI
#  ifdef DEBUG
    printf("Sharing bunch min/max data: %ld, %ld\n", ibMin, ibMax);
    fflush(stdout);
#  endif
    if (!partOnMaster) {
      MPI_Allreduce(&ibMin, &ibMinGlobal, 1, MPI_LONG, MPI_MIN, workers);
      MPI_Allreduce(&ibMax, &ibMaxGlobal, 1, MPI_LONG, MPI_MAX, workers);
      ibMin = ibMinGlobal;
      ibMax = ibMaxGlobal;
    }
#endif
    if (ibMin == LONG_MAX || ibMax == LONG_MIN)
      *nBunches = 0;
    else
      *nBunches = (ibMax - ibMin) + 1;
#ifdef DEBUG
    printf("nPPB=%ld, ibMin = %ld, ibMax = %ld, nBunches = %ld\n", idSlotsPerBunch, ibMin, ibMax, *nBunches);
    fflush(stdout);
#endif
  }
  if (*nBunches) {
    /* To prevent problems in LRWAKE, need to ensure that number of bunches does not increase */
    if (lastNBunches > 0 && *nBunches > lastNBunches) {
#ifdef DEBUG
      printf("Error: lastNBunches = %ld, *nBunches = %ld\n", lastNBunches, *nBunches);
      fflush(stdout);
#endif
#if USE_MPI
      mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
      return;
#else
      bombElegant("Error: number of bunches has increased.", NULL);
#endif
    }

#ifdef DEBUG
    printf("Performing bunch assignment, nBunches=%ld\n", *nBunches);
    fflush(stdout);
#endif
#if USE_MPI
    if (isSlave || !distributedBeam || partOnMaster) {
#  ifdef DEBUG
      printf("...performing bunch assignment\n");
      fflush(stdout);
#  endif
#endif
      if (np) {
#ifdef DEBUG
        printf("Doing branch for np!=0 (np=%" PRId64 ")\n", np);
        fflush(stdout);
#endif
        /* Compute time coordinate of each particle */
        if (!(*time = tmalloc(sizeof(**time) * np))) {
#if USE_MPI
          mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
          return;
#else
        bombElegant("Memory allocation problem in index_bunch_assignments\n", NULL);
#endif
        }
        computeTimeCoordinatesOnly(*time, P0, part, np);
#ifdef DEBUG
        printf("Computed time coordinates\n");
        fflush(stdout);
#endif
        *ibParticle = tmalloc(sizeof(**ibParticle) * np);
        if (ipBunch && npBunch) {
#ifdef DEBUG
          printf("Allocating cross-reference arrays\n");
          fflush(stdout);
#endif
          *npBunch = (int64_t *)tmalloc(sizeof(**npBunch) * (*nBunches));
          for (ib = 0; ib < (*nBunches); ib++)
            (*npBunch)[ib] = 0;
        }
#ifdef DEBUG
        printf("Looping over all particles\n");
        fflush(stdout);
#endif
        /* count the number of particles in each bunch so we can size arrays */
        for (ip = 0; ip < np; ip++) {
          if (*nBunches == 1)
            ib = 0;
          else
            ib = part[ip][bunchIndex] - ibMin;
          if (ib < 0 || ib >= (*nBunches)) {
#if USE_MPI
            mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
            return;
#else
          printf("Error: particle outside bunch: ib=%ld, nBunches=%ld, particleID=%ld\n", ib, *nBunches, (long)(part[ip][6]));
#endif
          }
          (*ibParticle)[ip] = ib;
          if (npBunch)
            (*npBunch)[ib] += 1;
        }
        if (ipBunch && npBunch) {
          *ipBunch = tmalloc(sizeof(**ipBunch) * (*nBunches));
          for (ib = 0; ib < *nBunches; ib++) {
            (*ipBunch)[ib] = (int64_t *)tmalloc(sizeof(***ipBunch) * (*npBunch)[ib]);
            (*npBunch)[ib] = 0; /* will use as index when assigning particles to bunches below */
          }
        }
        for (ip = 0; ip < np; ip++) {
          if (*nBunches == 1)
            ib = 0;
          else
            ib = (*ibParticle)[ip];
          if (ib < 0 || ib >= (*nBunches)) {
#if USE_MPI
            mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
            return;
#else
          printf("Error: particle outside bunch: ib=%ld, nBunches=%ld, particleID=%ld\n", ib, *nBunches, (long)(part[ip][6]));
#endif
          }
          if (ipBunch && npBunch) {
            (*ipBunch)[ib][(*npBunch)[ib]] = ip;
            (*npBunch)[ib] += 1;
          }
        }
      }

#if USE_MPI
    }
#endif
  }

#ifdef DEBUG
  printf("%ld bunches found (1)\n", *nBunches);
  printf("npBunch = %x\n", npBunch);
  printf("*npBunch = %x\n", *npBunch);
  fflush(stdout);
#if USE_MPI
  if (!partOnMaster || myid == 0) {
#endif
    for (ib = 0; ib < *nBunches; ib++)
      printf("npBunch[%ld] = %" PRId64 "\n", ib, (*npBunch)[ib]);
    fflush(stdout);
#if USE_MPI
  }
#endif
  printf("%ld bunches found (2)\n", *nBunches);
  fflush(stdout);
#endif

#if USE_MPI
  if (!partOnMaster) {
#  ifdef DEBUG
    printf("Waiting on barrier at end of index_bunch_assignment\n");
    fflush(stdout);
#  endif
    if (distributedBeam)
      MPI_Barrier(workers);
    else
      MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

#ifdef DEBUG
  printf("Leaving index_bunch_assignment\n");
  fflush(stdout);
#endif
}

void track_through_lrwake(double **part, int64_t np, LRWAKE *wakeData, double *P0Input,
                          RUN *run, long i_pass, CHARGE *charge) {
  double *tBunch = NULL; /* array for <t> for bunches */
  double *xBunch = NULL; /* array for Q*<x> for bnuches */
  double *yBunch = NULL; /* array for Q*<y> for bunches */
  double *QBunch = NULL; /* array for Q for bunches */
  static long lastNBunches = -1;

  double *VzBunch = NULL, *VxBunch = NULL, *VyBunch = NULL; /* arrays of voltage at each bunch */
  double *QxBunch = NULL, *QyBunch = NULL;                  /* quadrupole wake divided by probe particle displacement */
  double *time = NULL;                                      /* array to record arrival time of each particle */
  long ib, nBunches;
  int64_t ip;
  double factor, P0, rampFactor;
  long *ibParticle = NULL;
#if USE_MPI
  double *buffer = NULL;
#endif

#ifdef DEBUG
#  if DEBUG > 1
  static FILE *fph = NULL, *fpw = NULL;
  if (fph == NULL) {
    fph = fopen("lrwake.history", "w");
    fprintf(fph, "SDDS1\n");
    fprintf(fph, "&parameter name=Pass type=long &end\n");
    fprintf(fph, "&column name=ib type=long &end\n");
    fprintf(fph, "&column name=tMean units=s type=double &end\n");
    fprintf(fph, "&column name=QSum units=C type=double &end\n");
    fprintf(fph, "&data mode=ascii no_row_counts=1 &end\n");
  }
  if (fpw == NULL) {
    fpw = fopen("lrwake.wake", "w");
    fprintf(fpw, "SDDS1\n");
    fprintf(fpw, "&parameter name=Pass type=long &end\n");
    fprintf(fpw, "&column name=t units=s type=double &end\n");
    fprintf(fpw, "&column name=Vz units=V type=double &end\n");
    fprintf(fpw, "&data mode=ascii no_row_counts=1 &end\n");
  }
#  endif
#endif

  /* this element does nothing in single particle mode (e.g., trajectory, orbit, ..) */
  /*
#if USE_MPI
  if (distributedBeam==0)
    return;
#endif
*/

  if (isSlave || !distributedBeam) {
#ifdef DEBUG
    printf("Running track_through_lrwake, isSlave=%ld, distributedBeam=%ld\n", isSlave, distributedBeam);
#endif

    P0 = *P0Input;
    index_bunch_assignments(part, np, charge ? charge->idSlotsPerBunch : 0, P0, &time, &ibParticle, NULL, NULL, &nBunches, lastNBunches);
    lastNBunches = nBunches;

#ifdef DEBUG
    fputs("index_bunch_assignment returned\n", stdout);
    printf("np = %" PRId64 ", nBunches = %ld\n", np, nBunches);
#endif

    set_up_lrwake(wakeData, run, i_pass, np, charge, nBunches);

#ifdef DEBUG
    fputs("set_up_lrwake returned\n", stdout);
#endif

    rampFactor = 0;
    if (i_pass >= (wakeData->rampPasses - 1))
      rampFactor = 1;
    else
      rampFactor = (i_pass + 1.0) / wakeData->rampPasses;

    tBunch = tmalloc(sizeof(*tBunch) * nBunches);
    xBunch = tmalloc(sizeof(*xBunch) * nBunches);
    yBunch = tmalloc(sizeof(*yBunch) * nBunches);
    QBunch = tmalloc(sizeof(*QBunch) * nBunches);
    for (ib = 0; ib < nBunches; ib++)
      xBunch[ib] = yBunch[ib] = tBunch[ib] = QBunch[ib] = 0;

    for (ip = 0; ip < np; ip++) {
      ib = ibParticle[ip];
      tBunch[ib] += time[ip];
      QBunch[ib] += 1;
      xBunch[ib] += part[ip][0];
      yBunch[ib] += part[ip][2];
    }

#if USE_MPI
    /* sum data across processors */
#  ifdef MPI_DEBUG
    printf("Summing data across processors\n");
    fflush(stdout);
#  endif

    buffer = tmalloc(sizeof(*buffer) * nBunches);
    MPI_Allreduce(tBunch, buffer, nBunches, MPI_DOUBLE, MPI_SUM, workers);
    memcpy(tBunch, buffer, sizeof(*tBunch) * nBunches);

    MPI_Allreduce(QBunch, buffer, nBunches, MPI_DOUBLE, MPI_SUM, workers);
    memcpy(QBunch, buffer, sizeof(*QBunch) * nBunches);

    if (wakeData->xFactor && wakeData->WColumn[1]) {
      MPI_Allreduce(xBunch, buffer, nBunches, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(xBunch, buffer, sizeof(*xBunch) * nBunches);
    }

    if (wakeData->yFactor && wakeData->WColumn[2]) {
      MPI_Allreduce(yBunch, buffer, nBunches, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(yBunch, buffer, sizeof(*yBunch) * nBunches);
    }

    free(buffer);
    buffer = NULL;
#  ifdef MPI_DEBUG
    printf("Computing values within bunches\n");
    fflush(stdout);
#  endif
#endif

    /* Compute mean time and space position of particles within bunches, multiplied by charge */
    for (ib = 0; ib < nBunches; ib++) {
      if (QBunch[ib]) {
        /* QBunch[ib] holds particle count at this point */
        tBunch[ib] /= QBunch[ib];
        /* index_bunch_assignments() returns register-reduced time; add the per-step
           macro offset so bunch times stored in tHistory carry true absolute time.
           The wake lookup below uses tBunch-tHistory differences, which must span
           the macro inter-step gap; tBunch is never written back to coord[4].
           Zero unless step_frequency is in use, so legacy behavior is unchanged. */
        tBunch[ib] += trackingClockOffset();
        xBunch[ib] /= QBunch[ib];
        yBunch[ib] /= QBunch[ib];
        /* multiply by macro-particle charge so QBunch means what it says */
        QBunch[ib] *= charge->macroParticleCharge;
      }
    }

#ifdef MPI_DEBUG
    printf("Moving history data back\n");
    fflush(stdout);
#endif

    /* Move the existing history data back */
    memmove(wakeData->tHistory, wakeData->tHistory + nBunches, sizeof(*(wakeData->tHistory)) * (wakeData->nHistory - nBunches));
    memmove(wakeData->QHistory, wakeData->QHistory + nBunches, sizeof(*(wakeData->QHistory)) * (wakeData->nHistory - nBunches));
    memmove(wakeData->xHistory, wakeData->xHistory + nBunches, sizeof(*(wakeData->xHistory)) * (wakeData->nHistory - nBunches));
    memmove(wakeData->yHistory, wakeData->yHistory + nBunches, sizeof(*(wakeData->yHistory)) * (wakeData->nHistory - nBunches));

#ifdef MPI_DEBUG
    printf("Moving new data into buffer\n");
    fflush(stdout);
#endif
    /* Move the new history data into the buffer */
    memmove(wakeData->tHistory + wakeData->nHistory - nBunches, tBunch, sizeof(*tBunch) * nBunches);
    memmove(wakeData->QHistory + wakeData->nHistory - nBunches, QBunch, sizeof(*QBunch) * nBunches);
    memmove(wakeData->xHistory + wakeData->nHistory - nBunches, xBunch, sizeof(*xBunch) * nBunches);
    memmove(wakeData->yHistory + wakeData->nHistory - nBunches, yBunch, sizeof(*yBunch) * nBunches);

#ifdef DEBUG
#  if DEBUG > 1
    fprintf(fph, "%ld\n", i_pass);
    for (ib = 0; ib < wakeData->nHistory; ib++)
      fprintf(fph, "%ld %e %e\n", ib, wakeData->tHistory[ib], wakeData->QHistory[ib]);
    fprintf(fph, "\n");
    fflush(fph);
#  endif
#endif

#ifdef MPI_DEBUG
    printf("Computing wake function at each new bunch\n");
    fflush(stdout);
#endif
    /* Compute the wake function at each new bunch */
    VxBunch = tmalloc(sizeof(*VxBunch) * nBunches);
    VyBunch = tmalloc(sizeof(*VyBunch) * nBunches);
    QxBunch = tmalloc(sizeof(*QxBunch) * nBunches);
    QyBunch = tmalloc(sizeof(*QyBunch) * nBunches);
    VzBunch = tmalloc(sizeof(*VzBunch) * nBunches);
    for (ib = 0; ib < nBunches; ib++) {
      long ibh;
      VxBunch[ib] = VyBunch[ib] = VzBunch[ib] = QxBunch[ib] = QyBunch[ib] = 0;
#ifdef DEBUG
      printf("bin %ld: summing from %ld to %ld\n", ib, 0L, (wakeData->nHistory - nBunches + ib) - 1);
#endif
      for (ibh = 0; ibh < wakeData->nHistory - nBunches + ib; ibh++) {
        if (wakeData->QHistory[ibh]) {
          long it;
          double dt;
          dt = tBunch[ib] - wakeData->tHistory[ibh];
          /* interpolate wake functions */
          it = find_nearby_array_entry(wakeData->W[0], wakeData->wakePoints, dt);
          /*
#ifdef DEBUG
	printf("ib=%ld, ibh=%ld, dt=%le, it=%ld\n", ib, ibh, dt, it);
#endif
	*/
          if (wakeData->W[1])
            VxBunch[ib] += wakeData->xHistory[ibh] * wakeData->xFactor * wakeData->QHistory[ibh] *
                           linear_interpolation(wakeData->W[1], wakeData->W[0], wakeData->wakePoints, dt, it);
          if (wakeData->W[2])
            VyBunch[ib] += wakeData->yHistory[ibh] * wakeData->yFactor * wakeData->QHistory[ibh] *
                           linear_interpolation(wakeData->W[2], wakeData->W[0], wakeData->wakePoints, dt, it);
          if (wakeData->W[3])
            VzBunch[ib] += wakeData->zFactor * wakeData->QHistory[ibh] *
                           linear_interpolation(wakeData->W[3], wakeData->W[0], wakeData->wakePoints, dt, it);
          if (wakeData->W[4])
            QxBunch[ib] += wakeData->qxFactor * wakeData->QHistory[ibh] *
                           linear_interpolation(wakeData->W[4], wakeData->W[0], wakeData->wakePoints, dt, it);
          if (wakeData->W[5])
            QyBunch[ib] += wakeData->qyFactor * wakeData->QHistory[ibh] *
                           linear_interpolation(wakeData->W[5], wakeData->W[0], wakeData->wakePoints, dt, it);
        }
      }
    }

#ifdef DEBUG
#  if DEBUG > 1
    fprintf(fpw, "%ld\n", i_pass);
    for (ib = 0; ib < nBunches; ib++)
      fprintf(fpw, "%e %e\n", tBunch[ib], VzBunch[ib]);
    fprintf(fpw, "\n");
    fflush(fpw);
#  endif
#endif

#ifdef MPI_DEBUG
    printf("Imparting kicks to particles\n");
    fflush(stdout);
#endif
    for (ip = 0; ip < np; ip++) {
      /* Impart kick for each particle */
      double dgam, pz;
      factor = wakeData->factor * rampFactor;
      ib = ibParticle[ip];
      dgam = VzBunch[ib] / (1e6 * particleMassMV) * factor;
      add_to_particle_energy(part[ip], time[ip], P0, -dgam);
      pz = P0 * (1 + part[ip][5]) / sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
      part[ip][1] += (VxBunch[ib] + part[ip][0] * QxBunch[ib]) * factor / (1e6 * particleMassMV) / pz;
      part[ip][3] += (VyBunch[ib] + part[ip][2] * QyBunch[ib]) * factor / (1e6 * particleMassMV) / pz;
    }

    /* Free memory */
    if (tBunch)
      free(tBunch);
    if (xBunch)
      free(xBunch);
    if (yBunch)
      free(yBunch);
    if (QBunch)
      free(QBunch);
    if (VxBunch)
      free(VxBunch);
    if (VyBunch)
      free(VyBunch);
    if (VzBunch)
      free(VzBunch);
    if (QxBunch)
      free(QxBunch);
    if (QyBunch)
      free(QyBunch);
    free_bunch_index_memory(time, ibParticle, NULL, NULL, nBunches);
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

typedef struct {
  char *filename;
  long points;
  int32_t nColumns;
  char **columnName, **units;
  double **data;
} LRWAKE_DATA;

static LRWAKE_DATA *storedWake = NULL;
static long storedWakes = 0;
static char *expectedUnits[6] = {"s", "V/C/m", "V/C/m", "V/C", "V/C/m", "V/C/m"};

void set_up_lrwake(LRWAKE *wakeData, RUN *run, long pass, long particles, CHARGE *charge, long nBunches) {
  SDDS_DATASET SDDSin;
  double tmin, tmax;
  long iw, icol, ic;
#if SDDS_MPI_IO
  /* All the processes will read the wake file, but not in parallel.
   Zero the Memory when call  SDDS_InitializeInput */
  SDDSin.parallel_io = 0;
#endif
  if (!charge)
    bombElegant("LRWAKE given but no charge element in beamline.", NULL);

  if (wakeData->initialized)
    return;

  wakeData->initialized = 1;

  if (!wakeData->inputFile || !strlen(wakeData->inputFile))
    bombElegant("supply inputFile for LRWAKE element", NULL);

  for (icol = 0; icol < 6; icol++)
    if (wakeData->WColumn[icol] && !strlen(wakeData->WColumn[icol]))
      wakeData->WColumn[icol] = NULL;
  if (!wakeData->WColumn[0])
    bombElegant("supply tColumn for LRWAKE element", NULL);
  if (!wakeData->WColumn[1] && !wakeData->WColumn[2] && !wakeData->WColumn[3])
    bombElegant("supply at least one of WxColumn, WyColumn, or WzColumn for LRWAKE element", NULL);

  for (icol = 0; icol < 6; icol++)
    wakeData->W[icol] = NULL;

  for (iw = 0; iw < storedWakes; iw++) {
    if (strcmp(storedWake[iw].filename, wakeData->inputFile) == 0)
      break;
  }

  if (iw == storedWakes) {
    /* read new wake file */
#ifdef DEBUG
    fputs("Reading new wake file for LRWAKE\n", stdout);
#endif
    if (!(storedWake = SDDS_Realloc(storedWake, sizeof(*storedWake) * (storedWakes + 1)))) {
      printf("Error: memory allocation failur storing wake data from LRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }
    cp_str(&(storedWake[storedWakes].filename), wakeData->inputFile);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, wakeData->inputFile) || SDDS_ReadPage(&SDDSin) != 1) {
      printf("Error: unable to open or read LRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }
    if ((storedWake[storedWakes].points = SDDS_RowCount(&SDDSin)) < 0) {
      printf("Error: no data in LRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }
    if (storedWake[storedWakes].points < 2) {
      printf("Error: too little data in LRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }
    if (!(storedWake[storedWakes].columnName = SDDS_GetColumnNames(&SDDSin, &storedWake[storedWakes].nColumns)) ||
        storedWake[storedWakes].nColumns < 1) {
      printf("Error: too few columns in LRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }
    storedWake[storedWakes].data = tmalloc(sizeof(*(storedWake[storedWakes].data)) * storedWake[storedWakes].nColumns);
    storedWake[storedWakes].units = tmalloc(sizeof(*(storedWake[storedWakes].units)) * storedWake[storedWakes].nColumns);
#ifdef DEBUG
    fputs("Reading new wake file for LRWAKE (1)\n", stdout);
#endif
    for (ic = 0; ic < storedWake[storedWakes].nColumns; ic++) {
      storedWake[storedWakes].data[ic] = NULL;
      storedWake[storedWakes].units[ic] = NULL;
      if (SDDS_CheckColumn(&SDDSin, storedWake[storedWakes].columnName[ic], NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_WRONGTYPE &&
          (!(storedWake[storedWakes].data[ic] =
               SDDS_GetColumnInDoubles(&SDDSin, storedWake[storedWakes].columnName[ic])) ||
           !(SDDS_GetColumnInformation(&SDDSin, "units", &(storedWake[storedWakes].units[ic]), SDDS_BY_NAME, storedWake[storedWakes].columnName[ic])))) {
        printf("Error: problem reading column %s from LRWAKE file %s\n", storedWake[storedWakes].columnName[ic], wakeData->inputFile);
        exitElegant(1);
      }
    }
#ifdef DEBUG
    fputs("Reading new wake file for LRWAKE (2)\n", stdout);
#endif

    SDDS_Terminate(&SDDSin);
    storedWakes++;
  }

  /* search stored data for the required columns */
  for (iw = 0; iw < storedWakes; iw++) {
    /* Scan over all stored wake input files */
    if (strcmp(storedWake[iw].filename, wakeData->inputFile) == 0) {
#ifdef DEBUG
      printf("Using wakefile %ld (%s)\n", iw, storedWake[iw].filename);
#endif
      /* Found the right file */
      wakeData->wakePoints = storedWake[iw].points;
      for (icol = 0; icol < 6; icol++) {
#ifdef DEBUG
        printf("Looking for column %ld \n", icol);
#endif
        /* check for data for each type of data (t, Wx, Wy, Wz) */
        if (wakeData->WColumn[icol] && strlen(wakeData->WColumn[icol])) {
#ifdef DEBUG
          printf("Looking for column %s \n", wakeData->WColumn[icol]);
#endif
          /* user has requested this type of data */
          long ic;
          for (ic = 0; ic < storedWake[iw].nColumns; ic++) {
#ifdef DEBUG
            printf("Comparing data column %ld (%s) \n", ic, storedWake[iw].columnName[ic]);
#endif
            /* scan over all the columns for this file */
            if (strcmp(storedWake[iw].columnName[ic], wakeData->WColumn[icol]) == 0) {
              /* found the matching column, so check units */
#ifdef DEBUG
              printf("Found match\n");
#endif
              if (!storedWake[iw].units[ic] || !strlen(storedWake[iw].units[ic]) ||
                  strcmp(storedWake[iw].units[ic], expectedUnits[icol]) != 0) {
#ifdef DEBUG
                printf("Units don't match\n");
#endif
                printf("Expected units %s for column %s, but found %s, for LRWAKE file %s\n",
                       expectedUnits[icol], storedWake[iw].columnName[ic], storedWake[iw].units[ic], wakeData->inputFile);
                exitElegant(1);
              }
              /* copy the data pointer */
              wakeData->W[icol] = storedWake[iw].data[ic];
#ifdef DEBUG
              printf("Stored %x for slot %ld\n", wakeData->W[icol], icol);
#endif
              break;
            }
          }
        }
      }
      break;
    }
  }
#ifdef DEBUG
  printf("Completed seerch loop\n");
#endif

  for (icol = 0; icol < 6; icol++) {
    if (wakeData->WColumn[icol] && strlen(wakeData->WColumn[icol]) && !wakeData->W[icol]) {
      printf("Data for %s not found in LRWAKE file %s\n", wakeData->WColumn[icol], wakeData->inputFile);
      exitElegant(1);
    }
  }

  find_min_max(&tmin, &tmax, wakeData->W[0], wakeData->wakePoints);
  if (tmin >= tmax) {
    printf("Error: zero or negative time span in LRWAKE file %s\n", wakeData->inputFile);
    exitElegant(1);
  }
  if (tmin != 0) {
    printf("Error: LRWAKE function does not start at t=0 for file %s\n", wakeData->inputFile);
    exitElegant(1);
  }
  wakeData->dt = (tmax - tmin) / (wakeData->wakePoints - 1);

  wakeData->nHistory = wakeData->turnsToKeep * nBunches;

  wakeData->tHistory = tmalloc(sizeof(*(wakeData->tHistory)) * wakeData->nHistory);
  wakeData->xHistory = tmalloc(sizeof(*(wakeData->xHistory)) * wakeData->nHistory);
  wakeData->yHistory = tmalloc(sizeof(*(wakeData->yHistory)) * wakeData->nHistory);
  wakeData->QHistory = tmalloc(sizeof(*(wakeData->QHistory)) * wakeData->nHistory);
  for (iw = 0; iw < wakeData->nHistory; iw++)
    wakeData->tHistory[iw] = wakeData->xHistory[iw] = wakeData->yHistory[iw] = wakeData->QHistory[iw] = 0;
#ifdef DEBUG
  printf("Returing from lrwake setup\n");
#endif
}

void free_bunch_index_memory(double *time0, long *ibParticle, int64_t **ipBunch, int64_t *npBunch, long nBunches) {
  long iBunch;
  if (time0)
    free(time0);
  if (ibParticle)
    free(ibParticle);
  if (ipBunch) {
    for (iBunch = 0; iBunch < nBunches; iBunch++)
      free(ipBunch[iBunch]);
    free(ipBunch);
  }
  if (npBunch)
    free(npBunch);
}
