/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: trfmode.c
 * contents: track_through_trfmode()
 *
 * Michael Borland, 1999
 */
#include "mdb.h"
#include "track.h"

void runBinlessTrfMode(double **part, long np, TRFMODE *trfmode, double Po,
                       char *element_name, double element_z, long pass, long n_passes,
                       CHARGE *charge);

void track_through_trfmode(
  double **part0, long np0, TRFMODE *trfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  unsigned long *count = NULL;
  double *xsum = NULL;  /* sum of x coordinate in each bin = N*<x> */
  double *ysum = NULL;  /* sum of y coordinate in each bin = N*<y> */
  double *Vxbin = NULL; /* array for voltage acting on each bin MV */
  double *Vybin = NULL; /* array for voltage acting on each bin MV */
  double *Vzbin = NULL; /* array for voltage acting on each bin MV */
  long max_n_bins = 0;
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets, np;
  long max_np = 0;
  double tPrevious, VxPrevious, xPhasePrevious, VyPrevious, yPhasePrevious;
  long ip, ib;
  double tmin, tmax, tmean, dt, P, last_tmax = -DBL_MAX;
  double Vxb, Vyb, V, omega, phase, t, k, omegaOverC, damping_factor, tau;
  double Px, Py, Pz;
  double Q, Qrp;
  long n_binned, firstBin, lastBin;
#ifdef DEBUG
  static FILE *fpdeb = NULL;
  static long debugPass = 0;
#endif
#if USE_MPI
  double *buffer, t_total;
  unsigned long *ulbuffer;
  long np_total, binned_total;
  MPI_Status mpiStatus;
#endif

  if (trfmode->binless) { /* This can't be done in parallel mode */
#if USE_MPI
    printf("binless in trfmode is not supported in the current parallel version.\n");
    printf("Please use serial version.\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 9);
#endif
    runBinlessTrfMode(part0, np0, trfmode, Po, element_name, element_z, pass, n_passes, charge);
    return;
  }

#if USE_MPI
  MPI_Allreduce(&np0, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
  if (myid == 0) {
    printf("np_total = %ld\n", np_total);
    fflush(stdout);
  }
#  endif
#endif
  if (charge) {
    trfmode->mp_charge = charge->macroParticleCharge;
  } else if (pass == 0) {
    trfmode->mp_charge = 0;
    if (trfmode->charge < 0)
      bombElegant((char *)"TRFMODE charge parameter should be non-negative. Use change_particle to set particle charge state.", NULL);
#if (!USE_MPI)
    if (np0)
      trfmode->mp_charge = trfmode->charge / np0;
#else
    if (notSinglePart) {
      if (np_total)
        trfmode->mp_charge = trfmode->charge / np_total;
    } else {
      if (np0)
        trfmode->mp_charge = trfmode->charge / np0;
    }
#endif
  }
#ifdef DEBUG
  printf("TRFMODE: np0=%ld, charge=%le, mp_charge=%le\n", np0, trfmode->charge, trfmode->mp_charge);
#endif

#ifdef DEBUG
  if (!fpdeb) {
    fpdeb = fopen("trfmode.debug", "w");
    fprintf(fpdeb, "SDDS1\n&parameter name=DebugPass type=long &end\n");
    fprintf(fpdeb, "&parameter name=nBinned type=long &end\n");
    fprintf(fpdeb, "&parameter name=Pass type=long &end\n");
    fprintf(fpdeb, "&parameter name=Bunch type=long &end\n");
    fprintf(fpdeb, "&parameter name=Label1, type=string &end\n");
    fprintf(fpdeb, "&column name=Bin , type=double &end\n");
    fprintf(fpdeb, "&column name=Counts , type=long &end\n");
    fprintf(fpdeb, "&column name=xSum , type=double &end\n");
    fprintf(fpdeb, "&column name=ySum , type=double &end\n");
    fprintf(fpdeb, "&column name=xVoltage , type=double &end\n");
    fprintf(fpdeb, "&column name=yVoltage , type=double &end\n");
    fprintf(fpdeb, "&data mode=ascii &end\n");
  }
#endif

  if (trfmode->fileInitialized && pass == 0) {
    long n = n_passes / trfmode->sample_interval;
    if (!SDDS_StartPage(trfmode->SDDSrec, n)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb((char *)"problem startingn page for TRFMODE record file");
    }
  }

  omega = PIx2 * trfmode->freq;
  if ((Q = trfmode->Q / (1 + trfmode->beta)) <= 0.5) {
    printf("The effective Q<=0.5 for TRFMODE.  Use the ZTRANSVERSE element.\n");
    fflush(stdout);
    exitElegant(1);
  }
  tau = 2 * Q / omega;
  Qrp = sqrt(Q * Q - 0.25);
  k = omega / 4 * trfmode->RaInternal / trfmode->Q;
  /* These adjustments per Zotter and Kheifets, 3.2.4, 3.3.2 */
  k *= Q / Qrp;
  omega *= Qrp / Q;
  omegaOverC = omega / c_mks;

  if (!trfmode->doX && !trfmode->doY)
    bombElegant("x and y turned off for TRFMODE---this shouldn't happen", NULL);

  if (trfmode->freq < 1e3 && trfmode->freq)
    printWarningForTracking("TRFMODE frequency is less than 1kHz.", "This may be an error. Consult manual for units.");

  if (!trfmode->initialized)
    bombElegant("track_through_trfmode called with uninitialized element", NULL);

  if (trfmode->n_bins > max_n_bins) {
    max_n_bins = trfmode->n_bins;
    xsum = trealloc(xsum, sizeof(*xsum) * max_n_bins);
    ysum = trealloc(ysum, sizeof(*ysum) * max_n_bins);
    count = trealloc(count, sizeof(*count) * max_n_bins);
    Vxbin = trealloc(Vxbin, sizeof(*Vxbin) * max_n_bins);
    Vybin = trealloc(Vybin, sizeof(*Vybin) * max_n_bins);
    Vzbin = trealloc(Vzbin, sizeof(*Vzbin) * max_n_bins);
  }
  for (ib = 0; ib < trfmode->n_bins; ib++) {
    Vxbin[ib] = Vybin[ib] = Vzbin[ib] = xsum[ib] = ysum[ib] = 0;
    count[ib] = 0;
  }

  if (isSlave || !notSinglePart) {
#ifdef DEBUG
    printf("TRFMODE: Determining bucket assignments IDSlotsPerBunch = %ld\n",
            (charge && trfmode->bunchedBeamMode) ? charge->idSlotsPerBunch : 0);
#endif
    index_bunch_assignments(part0, np0, (charge && trfmode->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#ifdef DEBUG
    printf("TRFMODE: Done determining bucket assignments. nBuckets=%ld\n", nBuckets);
    fflush(stdout);
#endif
#if USE_MPI
    if (mpiAbort)
      return;
#endif
  } else
    nBuckets = 1;

#if USE_MPI
  /* Master needs to know the number of buckets */
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 1)
    MPI_Send(&nBuckets, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBuckets, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
#endif
#ifdef DEBUG
  printf("TRFMODE: nBuckets = %ld\n", nBuckets);
  fflush(stdout);
#endif

  for (iBucket = 0; iBucket < nBuckets; iBucket++) {
    np = -1;
    tmean = DBL_MAX;
#ifdef DEBUG
    printf("working on bucket %ld of %ld\n", iBucket, nBuckets);
#endif

#if USE_MPI
    /* Master needs to know if this bucket has particles */
    if (isSlave || !notSinglePart) {
      if (nBuckets == 1)
        np = np0;
      else
        np = npBucket[iBucket];
    } else {
      np = 0;
    }
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#ifdef DEBUG
    printf("Total of %ld particles, %ld on this processor\n", np_total, np);
    fflush(stdout);
#endif
    if (np_total == 0)
      continue;
#endif

    if (isSlave || !notSinglePart) {
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = (long *)trealloc(pbin, sizeof(*pbin) * (max_np = np));
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

      tmean = 0;
      if (isSlave) {
        for (ip = 0; ip < np; ip++) {
          tmean += time[ip];
        }
      }
    }

#if USE_MPI
    if (!isSlave)
      tmean = np = 0;
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&tmean, &t_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tmean = t_total / np_total;
#else
    tmean /= np;
#endif

    tmin = tmean - trfmode->bin_size * trfmode->n_bins / 2.;
    tmax = tmean + trfmode->bin_size * trfmode->n_bins / 2.;
    if (iBucket > 0 && tmin < last_tmax) {
#if USE_MPI
      if (myid == 0)
#endif
        bombElegant("Error: time range overlap between buckets\n", NULL);
    }
    last_tmax = tmax;
    dt = (tmax - tmin) / trfmode->n_bins;

    n_binned = 0;
    lastBin = -1;
    firstBin = trfmode->n_bins;

    if (isSlave) {
      for (ib = 0; ib < trfmode->n_bins; ib++)
        xsum[ib] = ysum[ib] = count[ib] = 0;
      if (np == -1) {
        /* should never happen... */
        bombElegant("np==-1 in TRFMODE. Seek professional help!", NULL);
      }
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
        n_binned++;
      }
    }

#if USE_MPI
    if (!isSlave)
      n_binned = 0;
    MPI_Allreduce(&n_binned, &binned_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (binned_total != np_total && myid == 0) {
      char warningBuffer[1024];
      snprintf(warningBuffer, 1024, "%ld of %ld particles binned.", binned_total, np_total);
      printWarningForTracking("Some particles not binned in TRFMODE", warningBuffer);
    }
    if (isSlave) {
      long lastBin_global, firstBin_global;
      MPI_Allreduce(&lastBin, &lastBin_global, 1, MPI_LONG, MPI_MAX, workers);
      lastBin = lastBin_global;
      MPI_Allreduce(&firstBin, &firstBin_global, 1, MPI_LONG, MPI_MIN, workers);
      firstBin = firstBin_global;

      buffer = malloc(sizeof(double) * (lastBin - firstBin + 1));
      ulbuffer = malloc(sizeof(unsigned long) * (lastBin - firstBin + 1));
      MPI_Allreduce(xsum + firstBin, buffer, lastBin - firstBin + 1, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(xsum + firstBin, buffer, sizeof(double) * (lastBin - firstBin + 1));
      MPI_Allreduce(ysum + firstBin, buffer, lastBin - firstBin + 1, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(ysum + firstBin, buffer, sizeof(double) * (lastBin - firstBin + 1));
      MPI_Allreduce(count + firstBin, ulbuffer, lastBin - firstBin + 1, MPI_UNSIGNED_LONG, MPI_SUM, workers);
      memcpy(count + firstBin, ulbuffer, sizeof(unsigned long) * (lastBin - firstBin + 1));
      free(buffer);
      free(ulbuffer);
    }
#else
    if (n_binned != np) {
      char warningBuffer[1024];
      snprintf(warningBuffer, 1024, "%ld of %ld particles binned.", n_binned, np);
      printWarningForTracking("Some particles not binned in TRFMODE", warningBuffer);
    }
#endif

    if (pass <= (trfmode->rampPasses - 1))
      k *= (pass + 1.0) / trfmode->rampPasses;

    if (trfmode->single_pass) {
      trfmode->Vx = trfmode->Vy = 0;
      trfmode->last_t = tmin + 0.5 * dt;
      trfmode->last_xphase = trfmode->last_yphase = 0;
    }

    VxPrevious = trfmode->Vx;
    VyPrevious = trfmode->Vy;
    xPhasePrevious = trfmode->last_xphase;
    yPhasePrevious = trfmode->last_yphase;
    tPrevious = trfmode->last_t;

#if USE_MPI
    if (isSlave) {
#endif
      for (ib = firstBin; ib <= lastBin; ib++) {
        if (!trfmode->interpolate && count[ib] == 0)
          continue;

        t = tmin + (ib + 0.5) * dt; /* middle arrival time for this bin */

        /* advance cavity to this time */
        damping_factor = exp(-(t - trfmode->last_t) / tau);
        if (damping_factor > 1)
          printWarningForTracking("Damping factor >1 for TRFMODE.", "This may happen if the bunch and binning region are very long.");
        if (trfmode->doX) {
          /* -- x plane */
          phase = trfmode->last_xphase + omega * (t - trfmode->last_t);
          V = trfmode->Vx * damping_factor;
          trfmode->Vxr = V * cos(phase);
          trfmode->Vxi = V * sin(phase);
          trfmode->last_xphase = phase;
        }
        if (trfmode->doY) {
          /* -- y plane */
          phase = trfmode->last_yphase + omega * (t - trfmode->last_t);
          V = trfmode->Vy * damping_factor;
          trfmode->Vyr = V * cos(phase);
          trfmode->Vyi = V * sin(phase);
          trfmode->last_yphase = phase;
        }

        trfmode->last_t = t;
        Vxbin[ib] = Vybin[ib] = Vzbin[ib] = 0;

        /* compute beam-induced voltage for this bin */
        if (trfmode->doX) {
          /* -- x plane (NB: ramp factor is already in k) */
          Vxb = 2 * k * trfmode->mp_charge * particleRelSign * xsum[ib] * trfmode->xfactor;
          if (trfmode->long_range_only) {
            double Vd = VxPrevious * exp(-(t - tPrevious) / tau);
            Vxbin[ib] = Vd * cos(xPhasePrevious + omega * (t - tPrevious));
            if (count[ib])
              Vzbin[ib] += omegaOverC * (xsum[ib] / count[ib]) * Vd * sin(xPhasePrevious + omega * (t - tPrevious));
          } else {
            Vxbin[ib] = trfmode->Vxr;
            if (count[ib])
              Vzbin[ib] += omegaOverC * (xsum[ib] / count[ib]) * (trfmode->Vxi - Vxb / 2);
          }
          /* add beam-induced voltage to cavity voltage---it is imaginary as
           * the voltage is 90deg out of phase 
           */
          trfmode->Vxi -= Vxb;
          if (trfmode->Vxi == 0 && trfmode->Vxr == 0)
            trfmode->last_xphase = 0;
          else
            trfmode->last_xphase = atan2(trfmode->Vxi, trfmode->Vxr);
          trfmode->Vx = sqrt(sqr(trfmode->Vxr) + sqr(trfmode->Vxi));
        }
        if (trfmode->doY) {
          /* -- y plane (NB: ramp factor is already in k) */
          Vyb = 2 * k * trfmode->mp_charge * particleRelSign * ysum[ib] * trfmode->yfactor;
          if (trfmode->long_range_only) {
            double Vd = VyPrevious * exp(-(t - tPrevious) / tau);
            Vybin[ib] = Vd * cos(yPhasePrevious + omega * (t - tPrevious));
            if (count[ib])
              Vzbin[ib] += omegaOverC * (ysum[ib] / count[ib]) * Vd * sin(yPhasePrevious + omega * (t - tPrevious));
          } else {
            Vybin[ib] = trfmode->Vyr;
            if (count[ib])
              Vzbin[ib] += omegaOverC * (ysum[ib] / count[ib]) * (trfmode->Vyi - Vyb / 2);
          }
          /* add beam-induced voltage to cavity voltage---it is imaginary as
           * the voltage is 90deg out of phase 
           */
          trfmode->Vyi -= Vyb;
          if (trfmode->Vyi == 0 && trfmode->Vyr == 0)
            trfmode->last_yphase = 0;
          else
            trfmode->last_yphase = atan2(trfmode->Vyi, trfmode->Vyr);
          trfmode->Vy = sqrt(sqr(trfmode->Vyr) + sqr(trfmode->Vyi));
        }
      }

#if DEBUG
      fprintf(fpdeb, "%ld\n%ld\n%ld\n%ld\n\"Pass %ld  Bucket %ld\"\n%ld\n",
              debugPass++, n_binned, pass, iBucket, pass, iBucket, lastBin - firstBin + 1);
      for (ib = firstBin; ib <= lastBin; ib++) {
        fprintf(fpdeb, "%ld %ld %e %e %e %e\n",
                ib, count[ib], xsum[ib], ysum[ib],
                Vxbin[ib],
                Vybin[ib]);
      }
      fflush(fpdeb);
#endif

      if (pass >= trfmode->rigid_until_pass) {
        /* change particle slopes to reflect voltage in relevant bin */
        for (ip = 0; ip < np; ip++) {
          double Vx, Vy, Vz;
          if (pbin[ip] >= 0) {
            P = Po * (1 + part[ip][5]);
            if (trfmode->interpolate) {
              long ib1, ib2;
              double dt1;
              ib = pbin[ip];
              dt1 = time[ip] - (tmin + dt * (ib + 0.5));
              if (dt1 < 0) {
                ib1 = ib - 1;
                ib2 = ib;
              } else {
                ib1 = ib;
                ib2 = ib + 1;
              }
              if (ib2 > lastBin) {
                ib2--;
                ib1--;
              }
              if (ib1 < firstBin) {
                ib1++;
                ib2++;
              }
              dt1 = time[ip] - (tmin + dt * (ib1 + 0.5));
              Vx = Vxbin[ib1] + (Vxbin[ib2] - Vxbin[ib1]) / dt * dt1;
              Vy = Vybin[ib1] + (Vybin[ib2] - Vybin[ib1]) / dt * dt1;
              Vz = Vzbin[ib1] + (Vzbin[ib2] - Vzbin[ib1]) / dt * dt1;
            } else {
              Vx = Vxbin[pbin[ip]];
              Vy = Vybin[pbin[ip]];
              Vz = Vzbin[pbin[ip]];
            }
            Pz = P / sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3])) + trfmode->n_cavities * Vz / (1e6 * particleMassMV * particleRelSign);
            Px = part[ip][1] * Pz + trfmode->n_cavities * Vx / (1e6 * particleMassMV * particleRelSign);
            Py = part[ip][3] * Pz + trfmode->n_cavities * Vy / (1e6 * particleMassMV * particleRelSign);
            P = sqrt(Pz * Pz + Px * Px + Py * Py);
            part[ip][1] = Px / Pz;
            part[ip][3] = Py / Pz;
            part[ip][5] = (P - Po) / Po;
            part[ip][4] = time[ip] * c_mks * P / sqrt(sqr(P) + 1);
          }
        }
      }

      if (nBuckets != 1) {
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }

#if USE_MPI
    }
#endif
  }

  if (trfmode->record) {
#if USE_MPI
    if (myid == 1) { /* first slave will do output */
#endif
      printf("Setting TRFMODE record data for pass=%ld\n", pass);
      if ((pass % trfmode->sample_interval) == 0 &&
          (!SDDS_SetRowValues(trfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              (pass / trfmode->sample_interval),
                              (char *)"Pass", pass,
                              (char *)"t", trfmode->last_t,
                              (char *)"Vx", sqrt(sqr(trfmode->Vxr) + sqr(trfmode->Vxi)),
                              (char *)"Vy", sqrt(sqr(trfmode->Vyr) + sqr(trfmode->Vyi)),
                              NULL) ||
           !SDDS_UpdatePage(trfmode->SDDSrec, FLUSH_TABLE))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"problem setting up data for TRFMODE record file");
      }
      /*
      if (pass==n_passes-1 && !SDDS_Terminate(trfmode->SDDSrec)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char*)"problem writing data for TRFMODE record file");
      }
      */
#if USE_MPI
    }
#endif
  }

  free(xsum);
  free(ysum);
  free(count);
  free(Vxbin);
  free(Vybin);
  free(Vzbin);
  if (part && part != part0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin)
    free(pbin);
  if (isSlave || !notSinglePart)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
}

void set_up_trfmode(TRFMODE *trfmode, char *element_name, double element_z,
                    long n_passes, RUN *run, long n_particles) {
  double T;

  if (trfmode->initialized)
    return;

  trfmode->initialized = 1;
  trfmode->Vx = trfmode->Vxr = trfmode->Vxi = trfmode->last_xphase =
    trfmode->Vy = trfmode->Vyr = trfmode->Vyi = trfmode->last_yphase =
      trfmode->last_t = 0;

#if SDDS_MPI_IO
  if (isSlave)
#endif
    if (n_particles < 1)
      bombElegant("too few particles in set_up_trfmode()", NULL);
  if (trfmode->n_bins < 2)
    bombElegant("too few bins for TRFMODE", NULL);
  if (trfmode->bin_size <= 0 && !trfmode->binless)
    bombElegant("bin_size must be positive for TRFMODE", NULL);
  if (trfmode->Ra && trfmode->Rs)
    bombElegant("TRFMODE element may have only one of Ra or Rs nonzero.  Ra is just 2*Rs", NULL);
  if (trfmode->long_range_only) {
    if (trfmode->single_pass)
      bombElegant((char *)"single-pass and long-range modes are incompatible in TRFMODE", NULL);
  }
  if (trfmode->Ra)
    trfmode->RaInternal = trfmode->Ra;
  else
    trfmode->RaInternal = 2 * trfmode->Rs;
  if (trfmode->bin_size * trfmode->freq > 0.1) {
    T = trfmode->bin_size * trfmode->n_bins;
    trfmode->bin_size = 0.1 / trfmode->freq;
    trfmode->n_bins = T / trfmode->bin_size;
    printf("The TRFMODE %s bin size is too large--setting to %e and increasing to %ld bins\n",
           element_name, trfmode->bin_size, trfmode->n_bins);
    fflush(stdout);
  }
  trfmode->last_t = element_z / c_mks;
  trfmode->Vxr = trfmode->Vxi = trfmode->Vx = 0;
  trfmode->Vyr = trfmode->Vyi = trfmode->Vy = 0;
  trfmode->doX = trfmode->doY = 0;
  if (strcmp_ci(trfmode->plane, "BOTH") == 0)
    trfmode->doX = trfmode->doY = 1;
  else if (strcmp_ci(trfmode->plane, "X") == 0)
    trfmode->doX = 1;
  else if (strcmp_ci(trfmode->plane, "Y") == 0)
    trfmode->doY = 1;
  if (!trfmode->doX && !trfmode->doY)
    bombElegant("No planes selected for TRFMODE", NULL);

#if (USE_MPI)
  if (myid == 1) /* We let the first slave to dump the parameter */
#endif
    if (trfmode->record && !trfmode->fileInitialized) {
      /* long n = n_passes/trfmode->sample_interval; */
      char *filename;
      filename = compose_filename(trfmode->record, run->rootname);
      if (!trfmode->perParticleOutput && trfmode->binless) {
        if (!trfmode->SDDSrec)
          trfmode->SDDSrec = tmalloc(sizeof(*(trfmode->SDDSrec)));
        if (!SDDS_InitializeOutputElegant(trfmode->SDDSrec, SDDS_BINARY, 1, NULL, NULL, filename) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "VxMax", "V", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "VxRealMax", "V", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "VyMax", "V", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "VyRealMax", "V", SDDS_DOUBLE) ||
            !SDDS_WriteLayout(trfmode->SDDSrec)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb("problem setting up TRFMODE record file");
        }
      } else {
        if (!trfmode->SDDSrec)
          trfmode->SDDSrec = tmalloc(sizeof(*(trfmode->SDDSrec)));
        if (!SDDS_InitializeOutputElegant(trfmode->SDDSrec, SDDS_BINARY, 1, NULL, NULL, filename) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "Vx", "V", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(trfmode->SDDSrec, "Vy", "V", SDDS_DOUBLE) ||
            !SDDS_WriteLayout(trfmode->SDDSrec)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb("problem setting up TRFMODE record file");
        }
      }
      free(filename);
      trfmode->fileInitialized = 1;
    }
}

void runBinlessTrfMode(
  double **part, long np, TRFMODE *trfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  static TIMEDATA *tData;
  static long max_np = 0;
  long ip, ip0;
  double P;
  double Vxb, Vyb, Vzb, dVxb, dVyb, V, omega, phase, t, k, omegaOverC, damping_factor, tau;
  double Px, Py, Pz;
  double Q, Qrp;
  double x, y;
  double VxMax, VxRealMax, VyMax, VyRealMax;
  /* static long called = 0; */
#if DEBUG
  static FILE *fpdeb = NULL;
  double dphase, Vxr_last, Vxi_last;
  if (!fpdeb) {
    fpdeb = fopen("trfmode.deb", "w");
    fprintf(fpdeb, "SDDS1\n");
    fprintf(fpdeb, "&column name=t type=double units=s &end\n");
    fprintf(fpdeb, "&column name=last_t type=double units=s &end\n");
    fprintf(fpdeb, "&column name=cos_dphase type=double &end\n");
    fprintf(fpdeb, "&column name=sin_dphase type=double &end\n");
    fprintf(fpdeb, "&column name=VxrOld, type=double &end\n");
    fprintf(fpdeb, "&column name=VxiOld, type=double &end\n");
    fprintf(fpdeb, "&column name=VxrNew, type=double &end\n");
    fprintf(fpdeb, "&column name=VxiNew, type=double &end\n");
    fprintf(fpdeb, "&column name=x, type=double &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }

#endif

  if (np == 0)
    return;

  if (charge) {
    trfmode->mp_charge = charge->macroParticleCharge;
  } else if (pass == 0) {
    trfmode->mp_charge = 0;
    if (trfmode->charge < 0)
      bombElegant("TRFMODE charge parameter should be non-negative. Use change_particle to set particle charge state.", NULL);
    if (np)
      trfmode->mp_charge = trfmode->charge / np;
  }

  omega = PIx2 * trfmode->freq;
  if ((Q = trfmode->Q / (1 + trfmode->beta)) <= 0.5) {
    printf("The effective Q<=0.5 for TRFMODE.  Use the ZTRANSVERSE element.\n");
    fflush(stdout);
    exitElegant(1);
  }
  tau = 2 * Q / omega;
  Qrp = sqrt(Q * Q - 0.25);
  k = omega / 4 * trfmode->RaInternal / trfmode->Q;

  if (!trfmode->doX && !trfmode->doY)
    bombElegant("x and y turned off for TRFMODE---this shouldn't happen", NULL);

  if (trfmode->freq < 1e3 && trfmode->freq)
    printWarningForTracking("TRFMODE frequency is less than 1kHz", "This may be an error. Consult manual for units.");

  if (!trfmode->initialized)
    bombElegant("track_through_trfmode called with uninitialized element", NULL);

  if (np > max_np)
    tData = trealloc(tData, sizeof(*tData) * (max_np = np));

  for (ip = 0; ip < np; ip++) {
    P = Po * (part[ip][5] + 1);
    tData[ip].t = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
    tData[ip].ip = ip;
  }
  qsort(tData, np, sizeof(*tData), compTimeData);

  /* These adjustments per Zotter and Kheifets, 3.2.4, 3.3.2 */
  k *= Q / Qrp;
  omega *= Qrp / Q;
  omegaOverC = omega / c_mks;

  if (pass <= (trfmode->rampPasses - 1))
    k *= (pass + 1.0) / trfmode->rampPasses;

  if (trfmode->single_pass) {
    trfmode->Vx = trfmode->Vy = 0;
    trfmode->last_t = tData[0].t;
    trfmode->last_xphase = trfmode->last_yphase = 0;
  }

  if (trfmode->record && (trfmode->sample_interval < 2 || pass % trfmode->sample_interval == 0)) {
    if (!SDDS_StartPage(trfmode->SDDSrec, trfmode->perParticleOutput ? np : 1)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb("problem setting up TRFMODE record file");
    }
  }

  VxMax = VxRealMax = VyMax = VyRealMax = 0;
  for (ip0 = 0; ip0 < np; ip0++) {
    ip = tData[ip0].ip;
    x = part[ip][0] - trfmode->dx;
    y = part[ip][2] - trfmode->dy;
    t = tData[ip0].t;

    /* advance cavity to this time */
    damping_factor = exp(-(t - trfmode->last_t) / tau);
    if (trfmode->doX) {
      /* -- x plane */
      phase = trfmode->last_xphase + omega * (t - trfmode->last_t);
      V = trfmode->Vx * damping_factor;
      trfmode->Vxr = V * cos(phase);
      trfmode->Vxi = V * sin(phase);
#if DEBUG
      dphase = omega * (t - trfmode->last_t);
      Vxr_last = trfmode->Vxr;
      Vxi_last = trfmode->Vxi;
#endif
      trfmode->last_xphase = phase;
    }
    if (trfmode->doY) {
      /* -- y plane */
      phase = trfmode->last_yphase + omega * (t - trfmode->last_t);
      V = trfmode->Vy * damping_factor;
      trfmode->Vyr = V * cos(phase);
      trfmode->Vyi = V * sin(phase);
      trfmode->last_yphase = phase;
    }

    Vxb = Vyb = Vzb = 0;

    /* compute beam-induced voltage for this bin */
    if (trfmode->doX) {
      /* -- x plane */
      dVxb = 2 * k * trfmode->mp_charge * particleRelSign * x * trfmode->xfactor;
      Vxb = trfmode->Vxr;
      Vzb += omegaOverC * x * (trfmode->Vxi - dVxb / 2);
      /* add beam-induced voltage to cavity voltage---it is imaginary as
       * the voltage is 90deg out of phase 
       */
      trfmode->Vxi -= dVxb;
      if (trfmode->Vxi == 0 && trfmode->Vxr == 0)
        trfmode->last_xphase = 0;
      else
        trfmode->last_xphase = atan2(trfmode->Vxi, trfmode->Vxr);
      trfmode->Vx = sqrt(sqr(trfmode->Vxr) + sqr(trfmode->Vxi));
    }
    if (trfmode->doY) {
      /* -- y plane */
      dVyb = 2 * k * trfmode->mp_charge * particleRelSign * y * trfmode->yfactor;
      Vyb = trfmode->Vyr;
      Vzb += omegaOverC * y * (trfmode->Vyi - dVyb / 2);
      /* add beam-induced voltage to cavity voltage---it is imaginary as
       * the voltage is 90deg out of phase 
       */
      trfmode->Vyi -= dVyb;
      if (trfmode->Vyi == 0 && trfmode->Vyr == 0)
        trfmode->last_yphase = 0;
      else
        trfmode->last_yphase = atan2(trfmode->Vyi, trfmode->Vyr);
      trfmode->Vy = sqrt(sqr(trfmode->Vyr) + sqr(trfmode->Vyi));
    }
    if (pass >= trfmode->rigid_until_pass) {
      /* change particle slopes to reflect voltage in relevant bin */
      P = Po * (1 + part[ip][5]);
      Pz = P / sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3])) + trfmode->n_cavities * Vzb / (1e6 * particleMassMV * particleRelSign);
      Px = part[ip][1] * Pz + trfmode->n_cavities * Vxb / (1e6 * particleMassMV * particleRelSign);
      Py = part[ip][3] * Pz + trfmode->n_cavities * Vyb / (1e6 * particleMassMV * particleRelSign);
#if DEBUG
      fprintf(fpdeb, "%e %e %e %e %e %e %e %e %e\n",
              tData[ip0].t, trfmode->last_t, cos(dphase), sin(dphase),
              Vxr_last, Vxi_last,
              trfmode->Vxr, trfmode->Vxi, part[ip][0]);
#endif
      P = sqrt(Pz * Pz + Px * Px + Py * Py);
      part[ip][1] = Px / Pz;
      part[ip][3] = Py / Pz;
      part[ip][5] = (P - Po) / Po;
      part[ip][4] = tData[ip0].t * c_mks * P / sqrt(sqr(P) + 1);
    }

    if (trfmode->record && (trfmode->sample_interval < 2 || pass % trfmode->sample_interval == 0)) {
      if (!trfmode->perParticleOutput) {
        if (VxMax < fabs(trfmode->Vx))
          VxMax = fabs(trfmode->Vx);
        if (VyMax < fabs(trfmode->Vy))
          VyMax = fabs(trfmode->Vy);
        if (VxRealMax < fabs(trfmode->Vxr))
          VxRealMax = fabs(trfmode->Vxr);
        if (VyRealMax < fabs(trfmode->Vyr))
          VyRealMax = fabs(trfmode->Vyr);
      } else {
        if (!SDDS_SetRowValues(trfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                               ip0,
                               "Pass", pass, "t", tData[ip0].t,
                               "Vx", trfmode->Vx, "VxReal", trfmode->Vxr,
                               "Vy", trfmode->Vy, "VyReal", trfmode->Vyr,
                               NULL)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb("problem setting up TRFMODE record data (1)");
        }
      }
    }

    trfmode->last_t = t;
  }

  if (trfmode->record && (trfmode->sample_interval < 2 || pass % trfmode->sample_interval == 0)) {
    if (!trfmode->perParticleOutput) {
      if (!SDDS_SetRowValues(trfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             0,
                             "Pass", pass, "t", tData[0].t,
                             "VxMax", VxMax, "VxRealMax", VxRealMax,
                             "VyMax", VyMax, "VyRealMax", VyRealMax,
                             NULL)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb("problem setting up TRFMODE record data (2)");
      }
    }
    if (!SDDS_WritePage(trfmode->SDDSrec)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb("problem writing TRFMODE record data");
    }
  }

#if defined(MINIMIZE_MEMORY)
  free(tData);
  tData = NULL;
  max_np = 0;
#endif

  /* called = 1; */
}

int compTimeData(const void *tv1, const void *tv2) {
  double diff;
  diff = ((TIMEDATA *)tv1)->t - ((TIMEDATA *)tv2)->t;
  if (diff < 0)
    return -1;
  if (diff > 0)
    return 1;
  return 0;
}
