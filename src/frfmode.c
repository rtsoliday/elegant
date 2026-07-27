/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: frfmode.c
 * contents: track_through_frfmode(), set_up_frfmode()
 *
 * Michael Borland, 1993, 2003
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu_rfmode.h"
#endif

void track_through_frfmode(
  double **part0, long np0, FRFMODE *rfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  long *Ihist = NULL;  /* array for histogram of particle density */
  double *Vbin = NULL; /* array for voltage acting on each bin */
  long max_n_bins = 0;
  long max_np = 0;
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets, np;

  double *VPrevious = NULL, *phasePrevious = NULL, tPrevious;
  long ip, ib, firstBin, lastBin;
#ifdef DEBUG
  long n_binned;
#endif
  double tmin, tmax, last_tmax, tmean, dt;
  double Vb, V, omega, phase, t, k, damping_factor, tau;
  //double V_sum, Vr_sum, phase_sum, Vc, Vcr;
  /* long max_hist, nb2, n_occupied; */
  long imode;
  double Qrp, VbImagFactor, Q;
  double rampFactor;
#if USE_MPI
  long np_total;
  long nonEmptyBins = 0;
#endif
#ifdef HAVE_GPU
  long gpuTracking = 0;
#endif

#if DEBUG == 1 && USE_MPI == 1
  printf("FRFMODE(1): myid=%d, isSlave=%ld, distributedBeam=%ld, np0=%ld\n",
         myid, isSlave, distributedBeam, np0);
  fflush(stdout);
#endif
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (charge)
    rfmode->mp_charge = charge->macroParticleCharge;
  else
    bombElegant("CHARGE element required to use FRFMODE", NULL);

  if (!rfmode->initialized)
    bombElegant("track_through_rfmode called with uninitialized element", NULL);

#if (USE_MPI)
  if (myid == 1) { /* We let the first slave to dump the parameter */
#endif
    if (rfmode->outputFile && pass == 0 && !SDDS_StartPage(rfmode->SDDSout, n_passes))
      SDDS_Bomb("Problem starting page for FRFMODE output file");
#if (USE_MPI)
  }
#endif

  if (rfmode->mp_charge == 0 || rfmode->factor == 0)
    return;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    if (gpu_rfmode_single_bunch_supported(
          np0, rfmode->bunchedBeamMode, charge))
      gpuTracking = 1;
    else
      part0 = forceParticlesToCpu("FRFMODE physical-bunch CPU fallback");
  }
#endif

  if (rfmode->n_bins > max_n_bins) {
    Ihist = trealloc(Ihist, sizeof(*Ihist) * (max_n_bins = rfmode->n_bins));
    Vbin = trealloc(Vbin, sizeof(*Vbin) * max_n_bins);
  }

#if DEBUG == 1 && USE_MPI == 1
  printf("FRFMODE(2): myid=%d, isSlave=%ld, distributedBeam=%ld\n",
         myid, isSlave, distributedBeam);
  fflush(stdout);
#endif

  if (isSlave || !distributedBeam) {
#ifdef HAVE_GPU
    if (gpuTracking) {
      nBuckets = 1;
    } else
#endif
    {
#ifdef DEBUG
    printf("FRFMODE: Determining bucket assignments\n");
    fflush(stdout);
#endif
    index_bunch_assignments(part0, np0, rfmode->bunchedBeamMode ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif
#ifdef DEBUG
    printf("FRFMODE: Done determining bucket assignments\n");
    fflush(stdout);
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
  printf("FRFMODE: nBuckets = %ld\n", nBuckets);
#endif

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
    last_tmax = -DBL_MAX;
    dt = 0;

#ifdef DEBUG
    n_binned = 0;
#endif
    lastBin = 0;
    firstBin = rfmode->n_bins;

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
#  ifdef HAVE_GPU
      if (!gpuTracking)
#  endif
      {
        if (np != 0)
          tmean /= np;
        else
          tmean = 0;
      }
#endif

      tmin = tmean - rfmode->bin_size * rfmode->n_bins / 2.;
      tmax = tmean + rfmode->bin_size * rfmode->n_bins / 2.;
      if (iBucket > 0 && tmin < last_tmax) {
#if USE_MPI
        if (myid == 0)
#endif
          bombElegant("Error: time range overlap between buckets\n", NULL);
      }
      last_tmax = tmax;

      if (isSlave) {
#ifdef DEBUG
        printf("tmin = %le, tmax = %le, tmean = %le\n", tmin, tmax, tmean);
        fflush(stdout);
#endif

        if (rfmode->long_range_only) {
          VPrevious = tmalloc(sizeof(*VPrevious) * rfmode->modes);
          phasePrevious = tmalloc(sizeof(*phasePrevious) * rfmode->modes);
          for (imode = 0; imode < rfmode->modes; imode++) {
            VPrevious[imode] = rfmode->V[imode];
            phasePrevious[imode] = rfmode->last_phase[imode];
          }
        }
#ifdef DEBUG
        printf("Allocated arrays for previous values\n");
        fflush(stdout);
#endif

        for (ib = 0; ib < rfmode->n_bins; ib++) {
          Ihist[ib] = 0;
          Vbin[ib] = 0;
        }

#ifdef DEBUG
        printf("Zeroed-out histogram and voltage arrays.\n");
        fflush(stdout);
#endif
        dt = (tmax - tmin) / rfmode->n_bins;
#if USE_MPI
        nonEmptyBins = 0;
#endif
#ifdef HAVE_GPU
        if (gpuTracking) {
          gpu_rfmode_histogram(
            np, Po, rfmode->n_bins, tmin, dt,
            Ihist, &firstBin, &lastBin);
        } else
#endif
        for (ip = 0; ip < np; ip++) {
          pbin[ip] = -1;
          ib = (time[ip] - tmin) / dt;
          if (ib < 0)
            continue;
          if (ib > rfmode->n_bins - 1)
            continue;
          Ihist[ib] += 1;
#if USE_MPI
          if (Ihist[ib] == 1)
            nonEmptyBins++;
#endif
          pbin[ip] = ib;
          if (ib > lastBin)
            lastBin = ib;
          if (ib < firstBin)
            firstBin = ib;
#ifdef DEBUG
            n_binned++;
#endif
        }
#ifdef DEBUG
        printf("Binned %ld particles\n", n_binned);
        fflush(stdout);
#endif
      }
    }

#if USE_MPI
    if (nBuckets == 0) {
      /* Since nBuckets can never be 0, this code is not called. */
      /* There are unresolved issues with histogram_sums() introducing noise into the results. */
      histogram_sums(nonEmptyBins, firstBin, &lastBin, Ihist);
    } else {
      long firstBin_global, lastBin_global;
      if (myid == 0) {
        firstBin = rfmode->n_bins;
        lastBin = 0;
      }
      MPI_Allreduce(&lastBin, &lastBin_global, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&firstBin, &firstBin_global, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
      firstBin = firstBin_global;
      lastBin = lastBin_global;
      if (isSlave || !distributedBeam) {
        long *buffer;
        buffer = (long *)calloc(lastBin - firstBin + 1, sizeof(long));
        MPI_Allreduce(&Ihist[firstBin], buffer, lastBin - firstBin + 1, MPI_LONG, MPI_SUM, workers);
        memcpy(Ihist + firstBin, buffer, sizeof(long) * (lastBin - firstBin + 1));
        free(buffer);
      }
#  ifdef DEBUG
      printf("firstBin = %ld, lastBin = %ld\n", firstBin_global, lastBin_global);
      fflush(stdout);
#  endif
    }
#endif

    rampFactor = 0;
    if (pass > (rfmode->rampPasses - 1))
      rampFactor = 1;
    else
      rampFactor = (pass + 1.0) / rfmode->rampPasses;

    if (isSlave || !distributedBeam) {
      tPrevious = rfmode->last_t;

      for (ib = firstBin; ib <= lastBin; ib++) {
        t = tmin + (ib + 0.5) * dt; /* middle arrival time for this bin */
        if (!Ihist[ib])
          continue;

        for (imode = 0; imode < rfmode->modes; imode++) {
          if (rfmode->cutoffFrequency > 0 && (rfmode->omega[imode] > PIx2 * rfmode->cutoffFrequency))
            continue;

          //V_sum = Vr_sum = phase_sum = Vc = Vcr = 0;
          /* max_hist = n_occupied = 0; */
          /* n_occupied = 0; */
          /* nb2 = rfmode->n_bins/2; */

          omega = rfmode->omega[imode];
          Q = rfmode->Q[imode] / (1 + rfmode->beta[imode]);
          tau = 2 * Q / omega;
          k = omega / 2 * rfmode->Rs[imode] * rfmode->factor / rfmode->Q[imode];

          /* These adjustments per Zotter and Kheifets, 3.2.4 */
          Qrp = sqrt(Q * Q - 0.25);
          VbImagFactor = 1 / (2 * Qrp);
          omega *= Qrp / Q;

          /* advance cavity to this time */
          phase = rfmode->last_phase[imode] + omega * (t - rfmode->last_t);
          damping_factor = exp(-(t - rfmode->last_t) / tau);
          rfmode->last_phase[imode] = phase;
          V = rfmode->V[imode] * damping_factor;
          rfmode->Vr[imode] = V * cos(phase);
          rfmode->Vi[imode] = V * sin(phase);

          /* compute beam-induced voltage for this bin */
          Vb = 2 * k * particleRelSign * rfmode->mp_charge * Ihist[ib] * rampFactor;
          if (rfmode->long_range_only) {
            Vbin[ib] += VPrevious[imode] * exp(-(t - tPrevious) / tau) * cos(phasePrevious[imode] + omega * (t - tPrevious));
          } else {
            Vbin[ib] += rfmode->Vr[imode] - Vb / 2;
          }

          /* add beam-induced voltage to cavity voltage */
          rfmode->Vr[imode] -= Vb;
          rfmode->Vi[imode] -= Vb * VbImagFactor;
          rfmode->last_phase[imode] = atan2(rfmode->Vi[imode], rfmode->Vr[imode]);
          rfmode->V[imode] = sqrt(sqr(rfmode->Vr[imode]) + sqr(rfmode->Vi[imode]));

          //V_sum += Ihist[ib] * rfmode->V[imode];
          //Vr_sum += Ihist[ib] * rfmode->Vr[imode];
          //phase_sum += Ihist[ib] * rfmode->last_phase[imode];

#if (USE_MPI)
          if (myid == 1) { /* We let the first slave to dump the parameter */
#endif
            if (rfmode->outputFile &&
                !SDDS_SetRowValues(rfmode->SDDSout,
                                   SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, pass,
                                   rfmode->modeIndex[imode], rfmode->V[imode], -1))
              SDDS_Bomb("Problem writing data to FRFMODE output file");
#if USE_MPI
          }
#endif
        }
        rfmode->last_t = t;
      }

      if (rfmode->rigid_until_pass <= pass) {
        /* change particle momentum offsets to reflect voltage in relevant bin */
        /* also recompute slopes for new momentum to conserve transverse momentum */
#ifdef HAVE_GPU
        if (gpuTracking) {
          if (firstBin <= lastBin)
            gpu_rfmode_apply_kicks(
              np, Po, rfmode->n_bins, tmin, dt,
              firstBin, lastBin, 0, rfmode->n_cavities, Vbin);
        } else
#endif
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip = 0; ip < np; ip++) {
          if (pbin[ip] >= 0) {
            /* compute new momentum and momentum offset for this particle */
            double particleDgamma = rfmode->n_cavities * Vbin[pbin[ip]] / (1e6 * particleMassMV * particleRelSign);
            add_to_particle_energy(part[ip], time[ip], Po, particleDgamma);
          }
        }
      }

      if (rfmode->outputFile && iBucket == (nBuckets - 1)) {
#if (USE_MPI)
        if (myid == 1) { /* We let the first slave to dump the parameter */
#endif
          if (!SDDS_SetRowValues(rfmode->SDDSout,
                                 SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, pass,
                                 "Pass", pass, NULL))
            SDDS_Bomb("Problem writing data to FRFMODE output file");
          if ((rfmode->flushInterval < 1 || pass % rfmode->flushInterval == 0 || pass == (n_passes - 1)) &&
              !SDDS_UpdatePage(rfmode->SDDSout, 0))
            SDDS_Bomb("Problem writing data to FRFMODE output file");
#if USE_MPI
        }
#endif
      }

      if (nBuckets != 1) {
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

#if USE_MPI
#  ifdef DEBUG
  printf("FRFMODE: Waiting on barrier after main loop\n");
  fflush(stdout);
#  endif
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef DEBUG
  printf("FRFMODE: Preparing to free memory\n");
  fflush(stdout);
#endif
  if (Ihist)
    free(Ihist);
  if (Vbin)
    free(Vbin);
  if (part && part != part0 && max_np > 0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin)
    free(pbin);
  if ((isSlave || !distributedBeam)
#ifdef HAVE_GPU
      && !gpuTracking
#endif
      )
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
#ifdef DEBUG
  printf("FRFMODE: Returning\n");
  fflush(stdout);
#endif
}

void set_up_frfmode(FRFMODE *rfmode, char *element_name, double element_z, long n_passes,
                    RUN *run, long n_particles,
                    double Po, double total_length) {
  long imode;
  double T;
  SDDS_DATASET SDDSin;

  if (rfmode->initialized)
    return;

#if !SDDS_MPI_IO
  if (n_particles < 1)
    bombElegant("too few particles in set_up_frfmode()", NULL);
#endif

  if (rfmode->n_bins < 2)
    bombElegant("too few bins for FRFMODE", NULL);
  if (rfmode->bin_size <= 0)
    bombElegant("bin_size must be positive for FRFMODE", NULL);
  if (!rfmode->filename ||
      !SDDS_InitializeInput(&SDDSin, rfmode->filename))
    bombElegant("unable to open file for FRFMODE element", NULL);
  /* check existence and properties of required columns  */
  if (SDDS_CheckColumn(&SDDSin, "Frequency", "Hz", SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Frequency column for FRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "Frequency", "Hz", SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Frequency column for FRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "Q", NULL, SDDS_ANY_FLOATING_TYPE,
                       stdout) != SDDS_CHECK_OK) {
    printf("Error: problem with Q column for FRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, "beta", NULL, SDDS_ANY_FLOATING_TYPE,
                       NULL) != SDDS_CHECK_NONEXISTENT) {
    if (SDDS_CheckColumn(&SDDSin, "beta", NULL, SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with \"beta\" column for FRFMODE file %s.  Check type and units.\n", rfmode->filename);
      exit(1);
    }
  }
  if (rfmode->useSymmData) {
    if (SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "$gW$r", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "Ohms", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedanceSymm", "Ohm", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with ShuntImpedanceSymm column for FRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
      exit(1);
    }
  } else {
    if (SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "$gW$r", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "Ohms", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin, "ShuntImpedance", "Ohm", SDDS_ANY_FLOATING_TYPE,
                         NULL) != SDDS_CHECK_OK) {
      printf("Error: problem with ShuntImpedance column for FRFMODE file %s.  Check existence, type, and units.\n", rfmode->filename);
      exit(1);
    }
  }

  if (!SDDS_ReadPage(&SDDSin))
    SDDS_Bomb("unable to read page from file for FRFMODE element");
  if ((rfmode->modes = SDDS_RowCount(&SDDSin)) < 1) {
    printf("Error: no data in FRFMODE file %s\n", rfmode->filename);
    exitElegant(1);
  }
  if (!(rfmode->omega = SDDS_GetColumnInDoubles(&SDDSin, "Frequency")) ||
      !(rfmode->Q = SDDS_GetColumnInDoubles(&SDDSin, "Q")) ||
      (rfmode->useSymmData &&
       !(rfmode->Rs = SDDS_GetColumnInDoubles(&SDDSin, "ShuntImpedanceSymm"))) ||
      (!rfmode->useSymmData &&
       !(rfmode->Rs = SDDS_GetColumnInDoubles(&SDDSin, "ShuntImpedance"))))
    SDDS_Bomb("Problem getting data from FRFMODE file");
  if (!(rfmode->beta = SDDS_GetColumnInDoubles(&SDDSin, "beta"))) {
    if (!(rfmode->beta = malloc(sizeof(*(rfmode->beta)) * rfmode->modes)))
      bombElegant("memory allocation failure (FRFMODE)", NULL);
    for (imode = 0; imode < rfmode->modes; imode++)
      rfmode->beta[imode] = 0;
  }
  SDDS_Terminate(&SDDSin);

  if (!(rfmode->V = malloc(sizeof(*(rfmode->V)) * rfmode->modes)) ||
      !(rfmode->Vr = malloc(sizeof(*(rfmode->Vr)) * rfmode->modes)) ||
      !(rfmode->Vi = malloc(sizeof(*(rfmode->Vi)) * rfmode->modes)) ||
      !(rfmode->last_phase = malloc(sizeof(*(rfmode->last_phase)) * rfmode->modes)))
    bombElegant("memory allocation failure (FRFMODE)", NULL);

  for (imode = 0; imode < rfmode->modes; imode++) {
    rfmode->omega[imode] *= PIx2;
    rfmode->V[imode] = rfmode->Vr[imode] = rfmode->Vi[imode] = 0;
    rfmode->last_phase[imode] = 0;
  }

  for (imode = 0; imode < rfmode->modes; imode++) {
    if (rfmode->bin_size * rfmode->omega[imode] / PIx2 > 0.1) {
      T = rfmode->bin_size * rfmode->n_bins;
      rfmode->bin_size = 0.1 / (rfmode->omega[imode] / PIx2);
      rfmode->n_bins = T / rfmode->bin_size + 1;
      rfmode->bin_size = T / rfmode->n_bins;
      printf("The FRFMODE %s bin size is too large for mode %ld--setting to %e and increasing to %ld bins\n",
             element_name, imode, rfmode->bin_size, rfmode->n_bins);
      fflush(stdout);
    }
  }

  for (imode = 0; imode < rfmode->modes; imode++) {
    if (rfmode->bin_size * rfmode->omega[imode] / PIx2 > 0.1) {
      printf("Error: FRFMODE bin size adjustment failed\n");
      exitElegant(1);
    }
  }

  rfmode->last_t = element_z / c_mks;

#if (USE_MPI)
  if (myid == 1) { /* We let the first slave to dump the parameter */
                   /*
#ifndef MPI_DEBUG
    dup2(fdStdout,fileno(stdout));
#endif
    */
#endif
    if (rfmode->outputFile) {
      TRACKING_CONTEXT context;
      char *filename;
      getTrackingContext(&context);
      filename = compose_filename(rfmode->outputFile, context.rootname);
      if (!(rfmode->SDDSout))
        rfmode->SDDSout = tmalloc(sizeof(*(rfmode->SDDSout)));
      if (!SDDS_InitializeOutputElegant(rfmode->SDDSout, SDDS_BINARY, 0, NULL, NULL, filename) ||
          !SDDS_DefineSimpleColumn(rfmode->SDDSout, "Pass", NULL, SDDS_LONG)) {
        fprintf(stderr, "Problem initializing file %s for FRFMODE element\n", filename);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (!(rfmode->modeIndex = malloc(sizeof(*(rfmode->modeIndex)) * rfmode->modes))) {
        fprintf(stderr, "Memory allocation failure for FRFMODE element\n");
        exitElegant(1);
      }
      for (imode = 0; imode < rfmode->modes; imode++) {
        char s[100];
        sprintf(s, "VMode%03ld", imode);
        if ((rfmode->modeIndex[imode] = SDDS_DefineColumn(rfmode->SDDSout, s, NULL, "V", NULL, NULL,
                                                          SDDS_DOUBLE, 0)) < 0) {
          fprintf(stderr, "Problem initializing file %s for FRFMODE element\n", filename);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
      if (!SDDS_WriteLayout(rfmode->SDDSout)) {
        fprintf(stderr, "Problem initializing file %s for FRFMODE element\n", filename);
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

  rfmode->initialized = 1;
}
