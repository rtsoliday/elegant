/************************************************************************* \
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: ftrfmode.c
 * contents: track_through_ftrfmode(), set_up_ftrfmode(),
 *
 * Michael Borland, 1993, 2003
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu_rfmode.h"
#endif

void track_through_ftrfmode(
  double **part0, long np0, FTRFMODE *trfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  unsigned long *count = NULL;
  double *xsum = NULL;     /* sum of x coordinate in each bin = N*<x> */
  double *ysum = NULL;     /* sum of y coordinate in each bin = N*<y> */
  double *Vxbin = NULL;    /* array for voltage acting on each bin MV */
  double *Vybin = NULL;    /* array for voltage acting on each bin MV */
  double *Vzbin = NULL;    /* array for voltage acting on each bin MV */
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets, np;

  long max_np = 0;
  double *VxPrevious = NULL, *VyPrevious = NULL, *xPhasePrevious = NULL, *yPhasePrevious = NULL, tPrevious;
  long ip, ib;
  double tmin, tmax, last_tmax, tmean, dt, P;
  double Vxb, Vyb, V, omega, phase, t, k, omegaOverC, damping_factor, tau;
  double Px, Py, Pz;
  double Q, Qrp;
  long firstBin, lastBin, imode;
  double rampFactor;
#ifdef HAVE_GPU
  long gpuTracking = 0;
#endif
#if USE_MPI
  long firstBin_global, lastBin_global, np_total;
  ;
#endif
#ifdef DEBUG
  static FILE *fpdeb = NULL;
/*  if (!fpdeb) {
    fpdeb = fopen("ftrfmode.sdds", "w");
    fprintf(fpdeb, "SDDS1\n");
    fprintf(fpdeb, "&column name=ib, type=long, &end\n");
    fprintf(fpdeb, "&column name=N, type=long, &end\n");
    fprintf(fpdeb, "&column name=xSum, type=double, &end\n");
    fprintf(fpdeb, "&column name=ySum, type=double, &end\n");
    fprintf(fpdeb, "&column name=t, type=double, units=s &end\n");
    fprintf(fpdeb, "&column name=Vx, type=double, units=V &end\n");
    fprintf(fpdeb, "&column name=Vy, type=double, units=V &end\n");
    fprintf(fpdeb, "&column name=Vz, type=double, units=V &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
*/
#endif

#ifdef DEBUG
  printf("In track_through_ftrfmode\n");
  fflush(stdout);
#endif

  if (charge)
    trfmode->mp_charge = charge->macroParticleCharge;
  else
    bombElegant("CHARGE element required to use FTRFMODE", NULL);
#ifdef DEBUG
  printf("mp charge = %le\n", trfmode->mp_charge);
  fflush(stdout);
#endif

  if (!trfmode->initialized)
    bombElegant("track_through_ftrfmode called with uninitialized element", NULL);

#if USE_MPI
  if (myid == 1) {
#endif
    if (trfmode->outputFile && pass == 0 && !SDDS_StartPage(trfmode->SDDSout, n_passes))
      SDDS_Bomb("Problem starting page for FTRFMODE output file");
#if USE_MPI
  }
#endif

  if (trfmode->mp_charge == 0 || (trfmode->xfactor == 0 && trfmode->yfactor == 0))
    return;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    if (gpu_rfmode_single_bunch_supported(
          np0, trfmode->bunchedBeamMode, charge))
      gpuTracking = 1;
    else
      part0 = forceParticlesToCpu("FTRFMODE physical-bunch CPU fallback");
  }
#endif

#ifdef DEBUG
  printf("About to allocate memory\n");
  fflush(stdout);
#endif

  xsum = calloc(trfmode->n_bins, sizeof(*xsum));
  ysum = calloc(trfmode->n_bins, sizeof(*ysum));
  count = calloc(trfmode->n_bins, sizeof(*count));

  Vxbin = calloc(trfmode->n_bins, sizeof(*Vxbin));
  Vybin = calloc(trfmode->n_bins, sizeof(*Vybin));
  Vzbin = calloc(trfmode->n_bins, sizeof(*Vzbin));
  if (!(xsum && ysum && count && Vxbin && Vybin && Vzbin))
    bomb("Memory allocation failure in track_through_ftrfmod", NULL);

#ifdef DEBUG
  printf("Finished allocating memory\n");
  fflush(stdout);
#endif

  if (!(xsum && ysum && count))
    bomb("Memory allocation failure in track_through_ftrfmod", NULL);

  if (isSlave || !distributedBeam) {
#ifdef HAVE_GPU
    if (gpuTracking) {
      nBuckets = 1;
    } else
#endif
    {
#ifdef DEBUG
    printf("FTRFMODE: Determining bucket assignments\n");
    fflush(stdout);
#endif
    index_bunch_assignments(part0, np0, trfmode->bunchedBeamMode ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#ifdef DEBUG
    printf("FTRFMODE: Done determining bucket assignments\n");
    fflush(stdout);
#endif
#if USE_MPI
    if (mpiAbort)
      return;
#endif
    }
  } else
    nBuckets = 0;

#if USE_MPI
  /* Master needs to know the number of buckets */
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(&nBuckets, &iBucket, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (myid == 0)
    nBuckets = iBucket;
#endif
#ifdef DEBUG
  printf("RFMODE: nBuckets = %ld\n", nBuckets);
  fflush(stdout);
#endif

  if (trfmode->long_range_only) {
    VxPrevious = tmalloc(sizeof(*VxPrevious) * trfmode->modes);
    VyPrevious = tmalloc(sizeof(*VyPrevious) * trfmode->modes);
    xPhasePrevious = tmalloc(sizeof(*xPhasePrevious) * trfmode->modes);
    yPhasePrevious = tmalloc(sizeof(*yPhasePrevious) * trfmode->modes);
  }

  for (iBucket = 0; iBucket < nBuckets; iBucket++) {
    np = -1;
#if USE_MPI
    /* Master needs to know if this bucket has particles */
    if (isSlave || !distributedBeam) {
      if (nBuckets == 1)
        np = np0;
      else if (npBucket)
        np = npBucket[iBucket];
      else
        np = 0;
    } else {
      np = 0;
    }
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (np_total == 0)
      continue;
#endif

    /* these are mostly to suppress compiler warnings */
    tmin = DBL_MAX;
    tmax = -DBL_MAX;
    tPrevious = 0;
    last_tmax = -DBL_MAX;
    dt = 0;

    lastBin = 0;
    firstBin = trfmode->n_bins;

    if (isSlave || !distributedBeam) {
      if (nBuckets == 1) {
        part = part0;
        np = np0;
#ifdef HAVE_GPU
        if (!gpuTracking)
#endif
        {
          time = time0;
          pbin = (long *)trealloc(pbin, sizeof(*pbin) * (max_np = np));
        }
      } else {
        if (npBucket && (np = npBucket[iBucket]) > 0) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)trealloc(time, sizeof(*time) * np);
          pbin = (long *)trealloc(pbin, sizeof(*pbin) * np);
          max_np = np;
          for (ip = 0; ip < np; ip++) {
            time[ip] = time0[ipBucket[iBucket][ip]];
            memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
          }
        } else
          np = 0;
      }
#ifdef DEBUG
      printf("Working on bucket %ld of %ld, %ld particles\n", iBucket, nBuckets, np);
      fflush(stdout);
#endif
      tmean = 0;
#ifdef HAVE_GPU
      if (gpuTracking) {
        tmean = gpu_rfmode_time_mean(np, Po);
      } else
#endif
      if (isSlave) {
        for (ip = 0; ip < np; ip++) {
          tmean += time[ip];
        }
      }
#if USE_MPI
      if (distributedBeam) {
        long np_total = np;
        if (isSlave) {
          double t_total;
          MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
          MPI_Allreduce(&tmean, &t_total, 1, MPI_DOUBLE, MPI_SUM, workers);
          tmean = t_total;
        }
        if (np_total)
          tmean /= np_total;
        else
          tmean = 0;
      } else if (np != 0)
        tmean /= np;
      else
        tmean = 0;
#else
      if (np != 0) {
#  ifdef HAVE_GPU
        if (!gpuTracking)
#  endif
          tmean /= np;
      } else
        tmean = 0;
#endif
#ifdef DEBUG
      printf("tmean = %21.15e, bin_size = %le, n_bins = %ld\n", 
             tmean, trfmode->bin_size, trfmode->n_bins);
      fflush(stdout);
#endif

      tmin = tmean - trfmode->bin_size * trfmode->n_bins / 2.;
      tmax = tmean + trfmode->bin_size * trfmode->n_bins / 2.;
      if (iBucket > 0 && tmin < last_tmax) {
#if USE_MPI
        if (myid == 0)
#endif
          printWarningForTracking("Time range overlap between buckets for FTRFMODE.", "Consider using fewer bins or smaller bins.");
      }
      last_tmax = tmax;

      if (isSlave) {
#ifdef DEBUG
        printf("tmin = %21.15e, tmax = %21.15e\n", tmin, tmax);
        fflush(stdout);
#endif

        if (trfmode->long_range_only) {
          tPrevious = trfmode->last_t;
          for (imode = 0; imode < trfmode->modes; imode++) {
            VxPrevious[imode] = trfmode->Vx[imode];
            VyPrevious[imode] = trfmode->Vy[imode];
            xPhasePrevious[imode] = trfmode->lastPhasex[imode];
            yPhasePrevious[imode] = trfmode->lastPhasey[imode];
          }
        }

        dt = (tmax - tmin) / trfmode->n_bins;
#ifdef DEBUG
        printf("dt = %21.15e\n", dt);
        fflush(stdout);
#endif
        lastBin = -1;
        firstBin = trfmode->n_bins;
        for (ib = 0; ib < trfmode->n_bins; ib++)
          xsum[ib] = ysum[ib] = count[ib] = 0;

        if (isSlave) {
#ifdef HAVE_GPU
          if (gpuTracking) {
            gpu_trfmode_histogram(
              np, Po, trfmode->n_bins, tmin, dt,
              trfmode->dx, trfmode->dy, xsum, ysum, count,
              &firstBin, &lastBin);
          } else
#endif
          {
          for (ip = 0; ip < np; ip++) {
            pbin[ip] = -1;
            ib = (time[ip] - tmin) / dt;
            if (ib < 0)
              continue;
            if (ib > trfmode->n_bins - 1)
              continue;
            xsum[ib] += part[ip][0] - trfmode->dx;
            ysum[ib] += part[ip][2] - trfmode->dy;
            count[ib] += 1;
            pbin[ip] = ib;
            if (ib > lastBin)
              lastBin = ib;
            if (ib < firstBin)
              firstBin = ib;
          }
          }
        }
      }
#ifdef DEBUG
      printf("firstBin = %ld, lastBin = %ld\n", firstBin, lastBin);
      fflush(stdout);
#endif
    }

#if USE_MPI
    if (isMaster) {
      firstBin = trfmode->n_bins;
      lastBin = 0;
    }
    MPI_Allreduce(&lastBin, &lastBin_global, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&firstBin, &firstBin_global, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    lastBin = lastBin_global;
    firstBin = firstBin_global;
#  ifdef DEBUG
    printf("global: firstBin = %ld, lastBin = %ld\n", firstBin, lastBin);
    fflush(stdout);
#  endif
    if (isSlave || !distributedBeam) {
      double *dbuffer;
      unsigned long *lbuffer;

      dbuffer = (double *)calloc(lastBin - firstBin + 1, sizeof(double));
#  ifdef DEBUG
      printf("sharing x sums, firstBin=%ld, lastBin=%ld\n", firstBin, lastBin);
      fflush(stdout);
#  endif
      MPI_Allreduce(&xsum[firstBin], dbuffer, lastBin - firstBin + 1, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(xsum + firstBin, dbuffer, sizeof(double) * (lastBin - firstBin + 1));

#  ifdef DEBUG
      printf("sharing y sums\n");
      fflush(stdout);
#  endif
      MPI_Allreduce(&ysum[firstBin], dbuffer, lastBin - firstBin + 1, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(ysum + firstBin, dbuffer, sizeof(double) * (lastBin - firstBin + 1));
      free(dbuffer);

#  ifdef DEBUG
      printf("sharing counts\n");
      fflush(stdout);
#  endif
      lbuffer = (unsigned long *)calloc(lastBin - firstBin + 1, sizeof(unsigned long));
      MPI_Allreduce(&count[firstBin], lbuffer, lastBin - firstBin + 1, MPI_UNSIGNED_LONG, MPI_SUM, workers);
      memcpy(count + firstBin, lbuffer, sizeof(unsigned long) * (lastBin - firstBin + 1));
      free(lbuffer);
#  ifdef DEBUG
      printf("done sharing data\n");
      fflush(stdout);
#  endif
    }
#endif

    rampFactor = 0;
    if (pass > (trfmode->rampPasses - 1))
      rampFactor = 1;
    else
      rampFactor = (pass + 1.0) / trfmode->rampPasses;

    if (isSlave) {
      double last_t = trfmode->last_t; /* Save it and use it later for different modes */

      for (ib = firstBin; ib <= lastBin; ib++)
        Vxbin[ib] = Vybin[ib] = Vzbin[ib] = 0;
      for (imode = 0; imode < trfmode->modes; imode++) {
        if (trfmode->cutoffFrequency > 0 && (trfmode->omega[imode] > PIx2 * trfmode->cutoffFrequency))
          continue;
        if (!trfmode->doX[imode] && !trfmode->doY[imode])
          continue;

        trfmode->last_t = last_t;
#ifdef DEBUG
        printf("working on mode %ld, last_t = %21.15le\n", imode, last_t);
        fflush(stdout);
#endif
        for (ib = firstBin; ib <= lastBin; ib++) {
          if (!count[ib])
            continue;
          t = tmin + (ib + 0.5) * dt; /* middle arrival time for this bin */
          if (t < trfmode->last_t) {
            trfmode->last_t = t;
          }
          omega = trfmode->omega[imode];
          Q = trfmode->Q[imode] / (1 + trfmode->beta[imode]);
          tau = 2 * Q / omega;
          Qrp = sqrt(Q * Q - 0.25);
          k = omega / 2 * trfmode->Rs[imode] / trfmode->Q[imode];
          /* These adjustments per Zotter and Kheifets, 3.2.4, 3.3.2 */
          k *= Q / Qrp;
          omega *= Qrp / Q;
          omegaOverC = omega / c_mks;

          damping_factor = exp(-(t - trfmode->last_t) / tau);
          if (trfmode->doX[imode]) {
            /* -- x plane */
            /* advance the phasor */
            phase = trfmode->lastPhasex[imode] + omega * (t - trfmode->last_t);
            V = trfmode->Vx[imode] * damping_factor;
            trfmode->Vxr[imode] = V * cos(phase);
            trfmode->Vxi[imode] = V * sin(phase);
            trfmode->lastPhasex[imode] = phase;
            Vxb = 2 * k * trfmode->mp_charge * particleRelSign * xsum[ib] * trfmode->xfactor * rampFactor;

            /* add this cavity's contribution to this bin */
            if (trfmode->long_range_only) {
              double Vd = VxPrevious[imode] * exp(-(t - tPrevious) / tau);
              Vxbin[ib] += Vd * cos(xPhasePrevious[imode] + omega * (t - tPrevious));
              Vzbin[ib] += omegaOverC * (xsum[ib] / count[ib]) * Vd * sin(xPhasePrevious[imode] + omega * (t - tPrevious));
            } else {
              Vxbin[ib] += trfmode->Vxr[imode];
              Vzbin[ib] += omegaOverC * (xsum[ib] / count[ib]) * (trfmode->Vxi[imode] - Vxb / 2);
            }

            /* add beam-induced voltage to cavity voltage---it is imaginary as
             * the voltage is 90deg out of phase 
             */
            trfmode->Vxi[imode] -= Vxb;

            /* update the phasor */
            if (trfmode->Vxi[imode] == 0 && trfmode->Vxr[imode] == 0)
              trfmode->lastPhasex[imode] = 0;
            else
              trfmode->lastPhasex[imode] = atan2(trfmode->Vxi[imode], trfmode->Vxr[imode]);
            trfmode->Vx[imode] = sqrt(sqr(trfmode->Vxr[imode]) + sqr(trfmode->Vxi[imode]));
          }
          if (trfmode->doY[imode]) {
            /* -- y plane */
            /* advance the phasor */
            phase = trfmode->lastPhasey[imode] + omega * (t - trfmode->last_t);
            V = trfmode->Vy[imode] * damping_factor;
            trfmode->Vyr[imode] = V * cos(phase);
            trfmode->Vyi[imode] = V * sin(phase);
            trfmode->lastPhasey[imode] = phase;
            Vyb = 2 * k * trfmode->mp_charge * particleRelSign * ysum[ib] * trfmode->yfactor * rampFactor;

            /* add this cavity's contribution to this bin */
            if (trfmode->long_range_only) {
              double Vd = VyPrevious[imode] * exp(-(t - tPrevious) / tau);
              Vybin[ib] += Vd * cos(yPhasePrevious[imode] + omega * (t - tPrevious));
              Vzbin[ib] += omegaOverC * (ysum[ib] / count[ib]) * Vd * sin(yPhasePrevious[imode] + omega * (t - tPrevious));
            } else {
              Vybin[ib] += trfmode->Vyr[imode];
              Vzbin[ib] += omegaOverC * (ysum[ib] / count[ib]) * (trfmode->Vyi[imode] - Vyb / 2);
            }

            /* add beam-induced voltage to cavity voltage---it is imaginary as
             * the voltage is 90deg out of phase 
             */
            trfmode->Vyi[imode] -= Vyb;

            /* update the phasor */
            if (trfmode->Vyi[imode] == 0 && trfmode->Vyr[imode] == 0)
              trfmode->lastPhasey[imode] = 0;
            else
              trfmode->lastPhasey[imode] = atan2(trfmode->Vyi[imode], trfmode->Vyr[imode]);
            trfmode->Vy[imode] = sqrt(sqr(trfmode->Vyr[imode]) + sqr(trfmode->Vyi[imode]));
          }
          trfmode->last_t = t;
        } /* loop over bins */
      }   /* loop over modes */

#ifdef DEBUG
      if (fpdeb) {
        for (ib = firstBin; ib <= lastBin; ib++) {
          if (count[ib])
            fprintf(fpdeb, "%ld %ld %e %e %21.15e %e %e %e \n", ib, (long)count[ib], xsum[ib], ysum[ib], tmin + dt * ib, Vxbin[ib], Vybin[ib], Vzbin[ib]);
        }
        fflush(fpdeb);
      }
#endif

      /* change particle slopes to reflect voltage in relevant bin */
#ifdef HAVE_GPU
      if (gpuTracking) {
        if (firstBin <= lastBin)
          gpu_trfmode_apply_kicks(
            np, Po, trfmode->n_bins, tmin, dt,
            firstBin, lastBin, 0, trfmode->n_cavities,
            Vxbin, Vybin, Vzbin);
      } else
#endif
      for (ip = 0; ip < np; ip++) {
        if (pbin[ip] >= 0) {
          ib = pbin[ip];
          if (ib < firstBin || ib > lastBin)
            bombElegant("particle bin index outside of expected range---please report this bug", NULL);
          P = Po * (1 + part[ip][5]);
          Pz = P / sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3])) + trfmode->n_cavities * Vzbin[ib] / (1e6 * particleMassMV * particleRelSign);
          Px = part[ip][1] * Pz + trfmode->n_cavities * Vxbin[ib] / (1e6 * particleMassMV * particleRelSign);
          Py = part[ip][3] * Pz + trfmode->n_cavities * Vybin[ib] / (1e6 * particleMassMV * particleRelSign);
          P = sqrt(Pz * Pz + Px * Px + Py * Py);
          part[ip][1] = Px / Pz;
          part[ip][3] = Py / Pz;
          part[ip][5] = (P - Po) / Po;
          part[ip][4] = time[ip] * c_mks * P / sqrt(sqr(P) + 1);
        }
      }

      if (nBuckets != 1) {
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }
  }

#ifdef DEBUG
  printf("Done with FTRFMODE: last_t = %21.15le\n", trfmode->last_t);
  for (imode = 0; imode < trfmode->modes; imode++) {
    printf("imode=%ld, Vxr=%le, Vxi=%le, phasex=%le, Vx=%le\n", imode, trfmode->Vxr[imode], trfmode->Vxi[imode], trfmode->lastPhasex[imode],
           sqrt(sqr(trfmode->Vxr[imode]) + sqr(trfmode->Vxi[imode])));
    printf("imode=%ld, Vyr=%le, Vyi=%le, phasey=%le, Vy=%le\n", imode, trfmode->Vyr[imode], trfmode->Vyi[imode], trfmode->lastPhasey[imode],
           sqrt(sqr(trfmode->Vyr[imode]) + sqr(trfmode->Vyi[imode])));
  }
#endif

#if USE_MPI
  if (myid == 1) { /* We let the first slave dump the parameter */
#endif
    if (trfmode->outputFile) {
      for (imode = 0; imode < trfmode->modes; imode++) {
        if (!SDDS_SetRowValues(trfmode->SDDSout,
                               SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, pass,
                               trfmode->xModeIndex[imode], trfmode->Vx[imode],
                               trfmode->yModeIndex[imode], trfmode->Vy[imode], -1))
          SDDS_Bomb("Problem writing data to FTRFMODE output file");
      }

      if (!SDDS_SetRowValues(trfmode->SDDSout,
                             SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, pass,
                             "Pass", pass, NULL))
        SDDS_Bomb("Problem writing data to FTRFMODE output file");
      if ((trfmode->flushInterval < 1 || pass % trfmode->flushInterval == 0 || pass == (n_passes - 1)) &&
          !SDDS_UpdatePage(trfmode->SDDSout, 0))
        SDDS_Bomb("Problem writing data to FTRFMODE output file");
    }
#if USE_MPI
  }
#endif

  free(xsum);
  free(ysum);
  free(count);
  free(Vxbin);
  free(Vybin);
  free(Vzbin);

  if (VxPrevious)
    free(VxPrevious);
  if (xPhasePrevious)
    free(xPhasePrevious);
  if (VyPrevious)
    free(VyPrevious);
  if (yPhasePrevious)
    free(yPhasePrevious);
  if (part && part != part0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin)
    free(pbin);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
}

void set_up_ftrfmode(FTRFMODE *rfmode, char *element_name, double element_z, long n_passes,
                     RUN *run, long n_particles,
                     double Po, double total_length) {
  long imode;
  double T;
  SDDS_DATASET SDDSin;

  if (rfmode->initialized)
    return;
#if SDDS_MPI_IO
  if (isSlave)
#endif
    if (n_particles < 1)
      bombElegant("too few particles in set_up_ftrfmode()", NULL);
  if (rfmode->n_bins < 2)
    bombElegant("too few bins for FTRFMODE", NULL);
  if (rfmode->bin_size <= 0)
    bombElegant("bin size must be positive for FTRFMODE", NULL);
  if (!rfmode->filename ||
      !SDDS_InitializeInput(&SDDSin, rfmode->filename))
    bombElegant("unable to open file for FTRFMODE element", NULL);
  /* check existence and properties of required columns  */
  if (SDDS_CheckColumn(&SDDSin, "Frequency", "Hz", SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Frequency column for FTRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "Frequency", "Hz", SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Frequency column for FTRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "Q", NULL, SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Q column for FTRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (rfmode->useSymmData) {
    if (SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "$gW$r/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "Ohms/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "Ohm/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with ShuntImpedanceSymm column for FTRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
      exitElegant(1);
    }
  } else {
    if (SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "$gW$r/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "Ohms/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "Ohm/m", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with ShuntImpedance column for FTRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
      exitElegant(1);
    }
  }

  if (!SDDS_ReadPage(&SDDSin))
    SDDS_Bomb("unable to read page from file for FTRFMODE element");
  if ((rfmode->modes = SDDS_RowCount(&SDDSin)) < 1) {
    printf("Error: no data in FTRFMODE file %s\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "beta", NULL, SDDS_ANY_FLOATING_TYPE,
                       NULL) != SDDS_CHECK_NONEXISTENT) {
    if (SDDS_CheckColumn(&SDDSin, "beta", NULL, SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with \"beta\" column for FRFMODE file %s.  Check type and units.\n", rfmode->filename);
      exitElegant(1);
    }
  }
  if (SDDS_CheckColumn(&SDDSin, "xMode", NULL, SDDS_ANY_INTEGER_TYPE,
                       NULL) != SDDS_CHECK_NONEXISTENT) {
    if (SDDS_CheckColumn(&SDDSin, "xMode", NULL, SDDS_ANY_INTEGER_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with \"doX\" column for FTRFMODE file %s.  Check type and units.\n", rfmode->filename);
      exitElegant(1);
    }
  }
  if (SDDS_CheckColumn(&SDDSin, "yMode", NULL, SDDS_ANY_INTEGER_TYPE,
                       NULL) != SDDS_CHECK_NONEXISTENT) {
    if (SDDS_CheckColumn(&SDDSin, "yMode", NULL, SDDS_ANY_INTEGER_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with \"doY\" column for FTRFMODE file %s.  Check type and units.\n", rfmode->filename);
      exitElegant(1);
    }
  }
  if (!(rfmode->omega = SDDS_GetColumnInDoubles(&SDDSin, "Frequency")) ||
      !(rfmode->Q = SDDS_GetColumnInDoubles(&SDDSin, "Q")) ||
      (rfmode->useSymmData &&
       !(rfmode->Rs = SDDS_GetColumnInDoubles(&SDDSin, "ShuntImpedanceSymm"))) ||
      (!rfmode->useSymmData &&
       !(rfmode->Rs = SDDS_GetColumnInDoubles(&SDDSin, "ShuntImpedance"))))
    SDDS_Bomb("Problem getting data from FTRFMODE file");
  if (!(rfmode->beta = SDDS_GetColumnInDoubles(&SDDSin, "beta"))) {
    if (!(rfmode->beta = malloc(sizeof(*(rfmode->beta)) * rfmode->modes)))
      bombElegant("memory allocation failure (FTRFMODE)", NULL);
    for (imode = 0; imode < rfmode->modes; imode++)
      rfmode->beta[imode] = 0;
  }
  if (!(rfmode->doX = SDDS_GetColumnInLong(&SDDSin, "xMode"))) {
    if (!(rfmode->doX = malloc(sizeof(*(rfmode->doX)) * rfmode->modes)))
      bombElegant("memory allocation failure (FTRFMODE)", NULL);
    for (imode = 0; imode < rfmode->modes; imode++)
      rfmode->doX[imode] = 1;
  }
  if (!(rfmode->doY = SDDS_GetColumnInLong(&SDDSin, "yMode"))) {
    if (!(rfmode->doY = malloc(sizeof(*(rfmode->doY)) * rfmode->modes)))
      bombElegant("memory allocation failure (FTRFMODE)", NULL);
    for (imode = 0; imode < rfmode->modes; imode++)
      rfmode->doY[imode] = 1;
  }
  SDDS_Terminate(&SDDSin);

  if (!(rfmode->Vx = malloc(sizeof(*(rfmode->Vx)) * rfmode->modes)) ||
      !(rfmode->Vxr = malloc(sizeof(*(rfmode->Vxr)) * rfmode->modes)) ||
      !(rfmode->Vxi = malloc(sizeof(*(rfmode->Vxi)) * rfmode->modes)) ||
      !(rfmode->Vy = malloc(sizeof(*(rfmode->Vy)) * rfmode->modes)) ||
      !(rfmode->Vyr = malloc(sizeof(*(rfmode->Vyr)) * rfmode->modes)) ||
      !(rfmode->Vyi = malloc(sizeof(*(rfmode->Vyi)) * rfmode->modes)) ||
      !(rfmode->lastPhasex = malloc(sizeof(*(rfmode->lastPhasex)) * rfmode->modes)) ||
      !(rfmode->lastPhasey = malloc(sizeof(*(rfmode->lastPhasey)) * rfmode->modes)))
    bombElegant("memory allocation failure (FTRFMODE)", NULL);

  for (imode = 0; imode < rfmode->modes; imode++) {
    rfmode->omega[imode] *= PIx2;
    rfmode->Vx[imode] = rfmode->Vxr[imode] = rfmode->Vxi[imode] = 0;
    rfmode->lastPhasex[imode] = 0;
    rfmode->Vy[imode] = rfmode->Vyr[imode] = rfmode->Vyi[imode] = 0;
    rfmode->lastPhasey[imode] = 0;
  }

  for (imode = 0; imode < rfmode->modes; imode++) {
    if (rfmode->bin_size * rfmode->omega[imode] / PIx2 > 0.1) {
      T = rfmode->bin_size * rfmode->n_bins;
      rfmode->bin_size = 0.1 / (rfmode->omega[imode] / PIx2);
      rfmode->n_bins = T / rfmode->bin_size + 1;
      rfmode->bin_size = T / rfmode->n_bins;
      printf("The FTRFMODE %s bin size is too large for mode %ld--setting to %e and increasing to %ld bins\n",
             element_name, imode, rfmode->bin_size, rfmode->n_bins);
      fflush(stdout);
    }
  }

  for (imode = 0; imode < rfmode->modes; imode++) {
    if (rfmode->bin_size * rfmode->omega[imode] / PIx2 > 0.1) {
      printf("Error: FTRFMODE bin size adjustment failed\n");
      exitElegant(1);
    }
  }

  rfmode->last_t = element_z / c_mks;

#if (USE_MPI)
  if (myid == 1) { /* We let the first slave to dump the parameter */
                   /*
#ifndef DEBUG
    dup2(fdStdout,fileno(stdout));
#endif
    */
#endif
    if (rfmode->outputFile) {
      TRACKING_CONTEXT context;
      char *filename;
      getTrackingContext(&context);
      filename = compose_filename(rfmode->outputFile, context.rootname);
      if (!rfmode->SDDSout)
        rfmode->SDDSout = tmalloc(sizeof(*(rfmode->SDDSout)));
      if (!SDDS_InitializeOutputElegant(rfmode->SDDSout, SDDS_BINARY, 0, NULL, NULL,
                                        filename) ||
          !SDDS_DefineSimpleColumn(rfmode->SDDSout, "Pass", NULL, SDDS_LONG)) {
        fprintf(stderr, "Problem initializing file %s for FTRFMODE element\n", filename);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (!(rfmode->xModeIndex = malloc(sizeof(*(rfmode->xModeIndex)) * rfmode->modes)) ||
          !(rfmode->yModeIndex = malloc(sizeof(*(rfmode->yModeIndex)) * rfmode->modes))) {
        fprintf(stderr, "Memory allocation failure for TFRFMODE element\n");
        exitElegant(1);
      }
      for (imode = 0; imode < rfmode->modes; imode++) {
        char sx[100], sy[100];
        sprintf(sx, "VxMode%03ld", imode);
        sprintf(sy, "VyMode%03ld", imode);
        if ((rfmode->xModeIndex[imode] = SDDS_DefineColumn(rfmode->SDDSout, sx, NULL, "V", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
            ((rfmode->yModeIndex[imode] = SDDS_DefineColumn(rfmode->SDDSout, sy, NULL, "V", NULL, NULL, SDDS_DOUBLE, 0)) < 0)) {
          fprintf(stderr, "Problem initializing file %s for FTRFMODE element\n", filename);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
      if (!SDDS_WriteLayout(rfmode->SDDSout)) {
        fprintf(stderr, "Problem initializing file %s for FTRFMODE element\n", filename);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (filename != rfmode->outputFile)
        free(filename);
    }
#if (USE_MPI)
    /*
#ifndef MPI_DEBUG
#if defined(_WIN32)
    freopen("NUL","w",stdout); 
#else
    freopen("/dev/null","w",stdout);   
#endif
#endif
  */
  } /* We let the first slave to dump the parameter */
#endif

  SDDS_ClearErrors();

  rfmode->initialized = 1;
}
