/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "table.h"
#include "fftpackC.h"

void set_up_trwake(TRWAKE *wakeData, RUN *run, long pass, long particles, CHARGE *charge);
void dumpTransverseTimeDistributions(double **posItime, long nb);

#ifdef HAVE_GPU
#  include <gpu_base.h>
#  include <gpu_trwake.h>
#endif
void track_through_trwake(double **part0, long np0, TRWAKE *wakeData, double Po,
                          RUN *run, long i_pass, CHARGE *charge) {
  double *posItime[2] = {NULL, NULL}; /* array for histogram of particle density times x, y*/
  double *Vtime = NULL;               /* array for voltage acting on each bin */
  long max_n_bins = 0;
  long *pbin = NULL;    /* array to record which bin each particle is in */
  double *time0 = NULL; /* array to record arrival time of each particle */
  double *time = NULL;  /* array to record arrival time of each particle, for working bucket */
  double *pz = NULL;
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long max_np = 0;
  long ib, nb = 0, n_binned = 0, plane;
  long iBucket, nBuckets, ip, np;
  double factor, tmin, tmean = 0, tmax, dt = 0, rampFactor = 1;
#if USE_MPI
  double *buffer;
#endif

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    long gpuBunchedWakeAction = gpu_trwake_bunched_mode_action(np0, wakeData, charge);
    if (gpuBunchedWakeAction == GPU_BUNCHED_WAKE_UNSUPPORTED) {
      part0 = forceParticlesToCpu("TRWAKE bunched beam CPU fallback");
    } else if (gpuBunchedWakeAction == GPU_BUNCHED_WAKE_SKIP) {
      return;
    } else {
      startGpuTimer();
      gpu_track_through_trwake(np0, wakeData, Po, run, i_pass, charge);
#  ifdef GPU_VERIFY
      startCpuTimer();
      track_through_trwake(part0, np0, wakeData, Po, run, i_pass, charge);
      compareGpuCpu(np0, "track_through_trwake");
#  endif
      return;
    }
  }
#endif

  set_up_trwake(wakeData, run, i_pass, np0, charge);

  if (i_pass >= (wakeData->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / wakeData->rampPasses;

  if (isSlave || !notSinglePart) {
    index_bunch_assignments(part0, np0, (charge && wakeData->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (wakeData->bunchedBeamMode && 
          ((wakeData->startBunch>=0 && iBucket<wakeData->startBunch) ||
           (wakeData->endBunch>=0 && iBucket>wakeData->endBunch)))
        continue;
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
        pz = trealloc(pz, sizeof(*pz) * np);
      } else {
        if ((np = npBucket[iBucket]) == 0)
          continue;
#ifdef DEBUG
        printf("WAKE: copying data to work array, iBucket=%ld, np=%ld\n", iBucket, np);
        fflush(stdout);
#endif
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)tmalloc(sizeof(*time) * np);
          pbin = trealloc(pbin, sizeof(*pbin) * np);
          pz = trealloc(pz, sizeof(*pz) * np);
          max_np = np;
        }
        for (ip = tmean = 0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          tmean += time[ip];
          pz[ip] = 0;
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
        if (np > 0)
          tmean /= np;
      }

      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#ifdef DEBUG
      printf("WAKE: tmin=%21.15le, tmax=%21.15le, tmax-tmin=%21.15le, np=%ld\n", tmin, tmax, tmax - tmin, np);
      fflush(stdout);
#endif
#if USE_MPI
      if (isSlave && notSinglePart)
        find_global_min_max(&tmin, &tmax, np, workers);
#endif
      if (isSlave || !notSinglePart) {
        if ((tmax - tmin) > (wakeData->t[wakeData->wakePoints - 1] - wakeData->t[0])) {
          fprintf(stderr, "The beam is longer than the transverse wake function.\nThis would produce unphysical results.\n");
          fprintf(stderr, "The beam length is %le s, while the wake length is %le s\n",
                  tmax - tmin, wakeData->t[wakeData->wakePoints - 1] - wakeData->t[0]);
          exitElegant(1);
        }

        dt = wakeData->dt;

        if (wakeData->n_bins) {
          tmin = tmean - dt * wakeData->n_bins / 2.0;
          nb = wakeData->n_bins;
        } else {
          nb = (tmax - tmin) / dt + 3;
          tmin -= dt;
          tmax += dt;
        }
#if HAVE_GPU
        if (nb > np)
          bombElegant("track_through_wake: gpu-elegant requires nbins<nparticles", NULL);
#endif
        if (tmin > tmax || nb <= 0) {
          printf("Problem with time coordinates in TRWAKE.  Po=%le\n", Po);
          exitElegant(1);
        }

        if (nb > max_n_bins) {
          posItime[0] = trealloc(posItime[0], sizeof(**posItime) * (max_n_bins = nb));
          posItime[1] = trealloc(posItime[1], sizeof(**posItime) * (max_n_bins = nb));
          Vtime = trealloc(Vtime, sizeof(*Vtime) * (max_n_bins + 1));
        }

        if (wakeData->tilt)
          rotateBeamCoordinatesForMisalignment(part, np, wakeData->tilt);

        n_binned = binTransverseTimeDistribution(posItime, pz, pbin, tmin, dt, nb, time, part, Po, np,
                                                 wakeData->dx, wakeData->dy,
                                                 wakeData->xDriveExponent, wakeData->yDriveExponent);
#if (!USE_MPI)
#  ifdef DEBUG
        dumpTransverseTimeDistributions(posItime, nb);
#  endif
#endif
      }
#if (!USE_MPI)
      if (n_binned != np) {
        char warningBuffer[1024];
        snprintf(warningBuffer, 1024, "Only %ld of %ld particles were binned. Consider setting N_BINS=0 to invoke autoscaling.",
                 n_binned, np);
        printWarningForTracking("Some particles not binned in TRWAKE.", warningBuffer);
      }
#else
      if (notSinglePart) {
        if (isSlave) {
          int all_binned, result = 1;

          result = ((n_binned == np) ? 1 : 0);
          MPI_Allreduce(&result, &all_binned, 1, MPI_INT, MPI_LAND, workers);
          if (!all_binned) {
            if (myid == 1) {
              dup2(fdStdout, fileno(stdout));
              printWarningForTracking("Some particles not binned in TRWAKE.",
                                      "Consider setting N_BINS=0 to invoke autoscaling.");
              if (!freopen("/dev/null", "w", stdout)) {
                perror("freopen failed");
                exit(EXIT_FAILURE);
              }
            }
          }
        }
      } else {
        if (n_binned != np) {
          dup2(fdStdout, fileno(stdout));
          printWarningForTracking("Some particles not binned in TRWAKE.",
                                  "Consider setting N_BINS=0 to invoke autoscaling.");
          if (!freopen("/dev/null", "w", stdout)) {
            perror("freopen failed");
            exit(EXIT_FAILURE);
          }
        }
      }
#endif
      if (isSlave || !notSinglePart) {
        for (plane = 0; plane < 2; plane++) {
          if (!wakeData->W[plane])
            continue;

#if USE_MPI
          if (isSlave && notSinglePart) {
            buffer = malloc(sizeof(double) * nb);
            if (nb<=0)
              bombElegantVA("Error in TRWAKE: number of bins is %ld\n", nb);
            MPI_Allreduce(posItime[plane], buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
            memcpy(posItime[plane], buffer, sizeof(double) * nb);
            free(buffer);
            buffer = NULL;
          }
#endif
          factor = wakeData->macroParticleCharge * particleRelSign * wakeData->factor;
          if (plane == 0)
            factor *= wakeData->xfactor * rampFactor;
          else
            factor *= wakeData->yfactor * rampFactor;
          if (!factor)
            continue;

          if (wakeData->smoothing && nb >= (2 * wakeData->SGHalfWidth + 1)) {
            if (!SavitzyGolaySmooth(posItime[plane], nb, wakeData->SGOrder,
                                    wakeData->SGHalfWidth, wakeData->SGHalfWidth, 0)) {
              fprintf(stderr, "Problem with smoothing for TRWAKE element (file %s)\n",
                      wakeData->inputFile);
              fprintf(stderr, "Parameters: nbins=%ld, order=%ld, half-width=%ld\n",
                      nb, wakeData->SGOrder, wakeData->SGHalfWidth);
              exitElegant(1);
            }
          }

          /* Do the convolution of the particle density and the wake function,
             V(T) = Integral[W(T-t)*I(t)dt, t={-infinity, T}]
             Note that T<0 is the head of the bunch.
             For the wake, the argument is the normal convention wherein larger
             arguments are later times.
             */
          Vtime[nb] = 0;
          convolveArrays(Vtime, nb,
                         posItime[plane], nb,
                         wakeData->W[plane], wakeData->wakePoints, wakeData->i0);

          for (ib = 0; ib < nb; ib++)
            Vtime[ib] *= factor;

          /* change particle transverse momenta to reflect voltage in relevant bin */
          applyTransverseWakeKicks(part, time, pz, pbin, np,
                                   Po, plane,
                                   Vtime, nb, tmin, dt, wakeData->interpolate,
                                   plane == 0 ? wakeData->xProbeExponent : wakeData->yProbeExponent);
        }

        if (wakeData->tilt)
          rotateBeamCoordinatesForMisalignment(part, np, -wakeData->tilt);

        if (nBuckets != 1) {
#ifdef DEBUG
          printf("WAKE: copying data back to main array\n");
          fflush(stdout);
#endif

          for (ip = 0; ip < np; ip++)
            memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);

#ifdef DEBUG
          printf("Done with bucket %ld\n", iBucket);
          fflush(stdout);
#endif
        }
      }
    }
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (part && part != part0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin)
    free(pbin);
  if (isSlave || !notSinglePart)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
  if (Vtime)
    free(Vtime);
  if (pz)
    free(pz);
  if (posItime[0])
    free(posItime[0]);
  if (posItime[1])
    free(posItime[1]);
}

void applyTransverseWakeKicks(double **part, double *time, double *pz, long *pbin, long np,
                              double Po, long plane,
                              double *Vtime, long nb, double tmin, double dt,
                              long interpolate, long exponent) {
  long ip, ib, offset;
  double dt1, Vinterp;

  offset = 2 * plane + 1; /* xp or yp index */
  for (ip = 0; ip < np; ip++) {
    if ((ib = pbin[ip]) >= 0 && ib <= nb - 1) {
      if (interpolate) {
        dt1 = time[ip] - (tmin + dt * ib); /* distance to bin center */
        if ((dt1 < 0 && ib) || ib == nb - 1) {
          ib--;
          dt1 += dt;
        }
        Vinterp = Vtime[ib] + (Vtime[ib + 1] - Vtime[ib]) / dt * dt1;
      } else
        Vinterp = Vtime[ib];
      if (exponent)
        Vinterp *= ipow(part[ip][offset - 1], exponent);
      if (Vinterp)
        part[ip][offset] += Vinterp / (1e6 * particleMassMV * particleRelSign) / pz[ip];
    }
  }
}

typedef struct {
  char *key;
  long points;
  double *t, *Wx, *Wy;
} WAKE_DATA;

static WAKE_DATA *storedWake = NULL;
static long storedWakes = 0;

void set_up_trwake(TRWAKE *wakeData, RUN *run, long pass, long particles, CHARGE *charge) {
  SDDS_DATASET SDDSin;
  double tmin, tmax;
  long iw;
  char *key;
#if SDDS_MPI_IO
  /* All the processes will read the wake file, but not in parallel.
   Zero the Memory when call  SDDS_InitializeInput */
  SDDSin.parallel_io = 0;
#endif

  if (charge) {
    wakeData->macroParticleCharge = charge->macroParticleCharge;
  } else if (pass == 0) {
    wakeData->macroParticleCharge = 0;
#if (!USE_MPI)
    if (wakeData->charge < 0)
      bombElegant("WAKE charge parameter should be non-negative.  Use change_particle to set particle charge state.", NULL);
    if (particles)
      wakeData->macroParticleCharge = wakeData->charge / particles;
#else
    if (isSlave) {
      long particles_total;
      MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, workers);
      if (particles_total)
        wakeData->macroParticleCharge = wakeData->charge / particles_total;
    }
#endif
  }

  if (wakeData->initialized)
    return;
  wakeData->initialized = 1;
  wakeData->W[0] = wakeData->W[1] = wakeData->t = NULL;

  if (wakeData->n_bins < 2 && wakeData->n_bins != 0)
    bombElegant("n_bins must be >=2 or 0 (autoscale) for TRWAKE element", NULL);

  if (!wakeData->inputFile || !wakeData->tColumn ||
      (!wakeData->WxColumn && !wakeData->WyColumn) ||
      !strlen(wakeData->inputFile) || !strlen(wakeData->tColumn))
    bombElegant("supply inputFile, tColumn, and WxColumn and/or WyColumn for TRWAKE element", NULL);
  if (wakeData->WxColumn && !strlen(wakeData->WxColumn))
    bombElegant("supply valid column name for WxColumn for TRWAKE element", NULL);
  if (wakeData->WyColumn && !strlen(wakeData->WyColumn))
    bombElegant("supply valid column name for WyColumn for TRWAKE element", NULL);

  key = tmalloc(sizeof(*key) * (strlen(wakeData->inputFile) + strlen(wakeData->tColumn) +
                                (wakeData->WxColumn ? strlen(wakeData->WxColumn) : 4) +
                                (wakeData->WyColumn ? strlen(wakeData->WyColumn) : 4) +
                                10));
  sprintf(key, "%s.%s.%s.%s", wakeData->inputFile, wakeData->tColumn,
          wakeData->WxColumn ? wakeData->WxColumn : "NULL",
          wakeData->WyColumn ? wakeData->WyColumn : "NULL");
  for (iw = 0; iw < storedWakes; iw++) {
    if (strcmp(storedWake[iw].key, key) == 0)
      break;
  }
  if (iw == storedWakes) {
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, wakeData->inputFile) || SDDS_ReadPage(&SDDSin) != 1 ||
        (wakeData->wakePoints = SDDS_RowCount(&SDDSin)) < 0 ||
        wakeData->wakePoints < 2) {
      fprintf(stderr, "Error: TRWAKE file %s is unreadable, or has insufficient data.\n", wakeData->inputFile);
      exitElegant(1);
    }
    if (SDDS_CheckColumn(&SDDSin, wakeData->tColumn, "s", SDDS_ANY_FLOATING_TYPE,
                         stdout) != SDDS_CHECK_OK) {
      fprintf(stderr, "Error: problem with time column for TRWAKE file %s---check existence, units, and type",
              wakeData->inputFile);
      exitElegant(1);
    }
    if (!(wakeData->t = SDDS_GetColumnInDoubles(&SDDSin, wakeData->tColumn))) {
      fprintf(stderr, "Error: unable to retrieve time data from TRWAKE file %s\n", wakeData->inputFile);
      exitElegant(1);
    }

    wakeData->W[0] = wakeData->W[1] = NULL;
    if (wakeData->WxColumn) {
      if (wakeData->xDriveExponent < 0 || wakeData->xProbeExponent < 0)
        bombElegant("Can't have xDriveExponent or xProbeExponent negative", NULL);
      switch (wakeData->xDriveExponent + wakeData->xProbeExponent) {
      case 1:
        if (SDDS_CheckColumn(&SDDSin, wakeData->WxColumn, "V/C/m", SDDS_ANY_FLOATING_TYPE,
                             stdout) != SDDS_CHECK_OK) {
          fprintf(stderr, "Error: problem (1) with Wx wake column for TRWAKE file %s---check existence, units, and type",
                  wakeData->inputFile);
          exitElegant(1);
        }
        break;
      case 0:
        if (SDDS_CheckColumn(&SDDSin, wakeData->WxColumn, "V/C", SDDS_ANY_FLOATING_TYPE,
                             stdout) != SDDS_CHECK_OK) {
          fprintf(stderr, "Error: problem (2) with Wx wake column for TRWAKE file %s---check existence, units, and type",
                  wakeData->inputFile);
          exitElegant(1);
        }
        break;
      default:
        /* Would be nice to be smarter here. */
        break;
      }
      if (!(wakeData->W[0] = SDDS_GetColumnInDoubles(&SDDSin, wakeData->WxColumn))) {
        fprintf(stderr, "Error: unable to retrieve Wx data from TRWAKE file %s\n", wakeData->inputFile);
        exitElegant(1);
      }
    }
    if (wakeData->WyColumn) {
      if (wakeData->yDriveExponent < 0 || wakeData->yProbeExponent < 0)
        bombElegant("Can't have yDriveExponent or yProbeExponent negative", NULL);
      switch (wakeData->yDriveExponent + wakeData->yProbeExponent) {
      case 1:
        if (SDDS_CheckColumn(&SDDSin, wakeData->WyColumn, "V/C/m", SDDS_ANY_FLOATING_TYPE,
                             stdout) != SDDS_CHECK_OK) {
          fprintf(stderr, "Error: problem (1) with Wy wake column for TRWAKE file %s---check existence, units, and type",
                  wakeData->inputFile);
          exitElegant(1);
        }
        break;
      case 0:
        if (SDDS_CheckColumn(&SDDSin, wakeData->WyColumn, "V/C", SDDS_ANY_FLOATING_TYPE,
                             stdout) != SDDS_CHECK_OK) {
          fprintf(stderr, "Error: problem (2) with Wy wake column for TRWAKE file %s---check existence, units, and type",
                  wakeData->inputFile);
          exitElegant(1);
        }
        break;
      default:
        /* Would be nice to be smarter here. */
        break;
      }
      if (!(wakeData->W[1] = SDDS_GetColumnInDoubles(&SDDSin, wakeData->WyColumn))) {
        fprintf(stderr, "Error: unable to retrieve Wy data from TRWAKE file %s\n", wakeData->inputFile);
        exitElegant(1);
      }
    }

    SDDS_Terminate(&SDDSin);

    /* record in wake storage */
    if (!(storedWake = SDDS_Realloc(storedWake, sizeof(*storedWake) * (storedWakes + 1))))
      SDDS_Bomb("Memory allocation failure (WAKE)");
    storedWake[storedWakes].key = key;
    storedWake[storedWakes].t = wakeData->t;
    storedWake[storedWakes].Wx = wakeData->W[0];
    storedWake[storedWakes].Wy = wakeData->W[1];
    storedWake[storedWakes].points = wakeData->wakePoints;
    wakeData->isCopy = 0;
    storedWakes++;
  } else {
    /* point to an existing wake */
    wakeData->t = storedWake[iw].t;
    wakeData->W[0] = storedWake[iw].Wx;
    wakeData->W[1] = storedWake[iw].Wy;
    wakeData->wakePoints = storedWake[iw].points;
    wakeData->isCopy = 1;
  }

  if (!wakeData->W[0] && !wakeData->W[1])
    bombElegant("no valid wake data for TRWAKE element", NULL);

  find_min_max(&tmin, &tmax, wakeData->t, wakeData->wakePoints);
#if USE_MPI
  if (isSlave)
    find_global_min_max(&tmin, &tmax, wakeData->wakePoints, workers);
#endif
  if (tmin == tmax)
    bombElegant("no time span in TRWAKE data", NULL);
  wakeData->dt = (tmax - tmin) / (wakeData->wakePoints - 1);
  wakeData->i0 = 0;
  if (tmin != 0) {
    if (!wakeData->acausalAllowed) {
      fprintf(stderr, "Error: TRWAKE function does not start at t=0 for file %s\n", wakeData->inputFile);
      fprintf(stderr, "If you really want this, set ACAUSAL_ALLOWED=1\n");
      exitElegant(1);
    } else {
      wakeData->i0 = -1;
      for (iw = 0; iw < wakeData->wakePoints; iw++) {
        if (fabs(wakeData->t[iw]) < 1e-6 * wakeData->dt) {
          wakeData->i0 = iw;
          break;
        }
      }
      if (wakeData->i0 == -1) {
        fprintf(stderr, "Error: TRWAKE function does have value at t=0 (within 1e-6*dt) for file %s\n", wakeData->inputFile);
        exitElegant(1);
      }
      if (fabs(tmin) > tmax) {
        fprintf(stderr, "Error: acausal WAKE function has |tmin|>tmax for file %s\n", wakeData->inputFile);
        exitElegant(1);
      }
    }
  }
}

void computeTimeCoordinatesOnly(double *time, double Po, double **part, long np) {
  long ip;
  double P;

  for (ip = 0; ip < np; ip++) {
    P = Po * (part[ip][5] + 1);
    time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
  }
}

/* This routine should call computeTimeCoordinatesOnly() first instead of duplicating
   the equations */

double computeTimeCoordinates(double *time, double Po, double **part, long np) {
  double tmean, P;
  long ip;
#ifdef USE_KAHAN
  double error = 0.0;
#endif

#if (!USE_MPI)
  for (ip = tmean = 0; ip < np; ip++) {
    P = Po * (part[ip][5] + 1);
#  ifndef USE_KAHAN
    tmean += (time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P));
#  else
    time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
    tmean = KahanPlus(tmean, time[ip], &error);
#  endif
  }
  return tmean / np;
#else /* MPI */
  tmean = 0;
  if (!partOnMaster) {
    long np_total = 0;
    double tmean_total;
    if (isSlave || !notSinglePart) {
      for (ip = 0; ip < np; ip++) {
        P = Po * (part[ip][5] + 1);
#  ifndef USE_KAHAN
        tmean += (time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P));
#  else
        time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
        tmean = KahanPlus(tmean, time[ip], &error);
#  endif
      }
    }
    tmean_total = DBL_MAX;
    if (notSinglePart) {
      if (isMaster) {
        tmean = 0;
        np = 0;
      }
      MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#  ifndef USE_KAHAN
      MPI_Allreduce(&tmean, &tmean_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
      tmean_total = KahanParallel(tmean, error, MPI_COMM_WORLD);
#  endif
    }
    if (tmean_total == DBL_MAX)
      bombElegant("invalid value of tmean_total in computeTimeCoordinates. Seek professional help!", NULL);
    if (np_total > 0)
      return tmean_total / np_total;
    return (double)0;
  } else { /* This serial part can be removed after all the upper level functions (e.g. wake) are parallelized */
    for (ip = tmean = 0; ip < np; ip++) {
      P = Po * (part[ip][5] + 1);
#  ifndef USE_KAHAN
      tmean += (time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P));
#  else
      time[ip] = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
      tmean = KahanPlus(tmean, time[ip], &error);
#  endif
    }
    return tmean / np;
  }
#endif
}

void computeDistanceCoordinates(double *time, double Po, double **part, long np) {
  double P, beta;
  long ip;

  for (ip = 0; ip < np; ip++) {
    P = Po * (part[ip][5] + 1);
    beta = P / sqrt(sqr(P) + 1);
    part[ip][4] = c_mks * beta * time[ip];
  }
}

long binTransverseTimeDistribution(double **posItime, double *pz, long *pbin, double tmin, double dt, long nb,
                                   double *time, double **part, double Po, long np,
                                   double dx, double dy, long xPower, long yPower) {
  long ip, ib, n_binned;
  for (ib = 0; ib < nb; ib++)
    posItime[0][ib] = posItime[1][ib] = 0;
  for (ip = n_binned = 0; ip < np; ip++) {
    pbin[ip] = -1;
    /* Bin CENTERS are at tmin+ib*dt */
    ib = (time[ip] - tmin) / dt + 0.5;
    if (ib < 0)
      continue;
    if (ib > nb - 1)
      continue;
    if (xPower == 1)
      posItime[0][ib] += part[ip][0] - dx;
    else if (xPower < 1)
      posItime[0][ib] += 1;
    else
      posItime[0][ib] += ipow(part[ip][0] - dx, xPower);
    if (yPower == 1)
      posItime[1][ib] += part[ip][2] - dy;
    else if (yPower < 1)
      posItime[1][ib] += 1;
    else
      posItime[1][ib] += ipow(part[ip][2] - dy, yPower);
    pbin[ip] = ib;
    if (pz)
      pz[ip] = Po * (1 + part[ip][5]) / sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
    n_binned++;
  }
  return n_binned;
}

void dumpTransverseTimeDistributions(double **posItime, long nb) {
  static FILE *fp = NULL;
  long ib;
  if (!fp) {
    fp = fopen("posItime.sdds", "w");
    fprintf(fp, "SDDS1\n&column name=Bin type=long &end\n&column name=xI, type=double &end\n&column name=yI, type=double &end\n");
    fprintf(fp, "&data mode=ascii &end\n");
  }
  fprintf(fp, "%ld\n", nb);
  for (ib = 0; ib < nb; ib++)
    fprintf(fp, "%ld %le %le\n", ib, posItime[0][ib], posItime[1][ib]);
  fflush(fp);
}
