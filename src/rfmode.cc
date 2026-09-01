/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: rfmode.c
 * contents: track_through_rfmode()
 *
 * Michael Borland, 1993
 */
#if USE_MPI
#  include "mpi.h" /* Defines the interface to MPI allowing the use of all MPI functions. */
#  if USE_MPE
#    include "mpe.h" /* Defines the MPE library */
#  endif
#endif
#include <complex>
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu_rfmode.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
long find_nearby_array_entry(double *entry, long n, double key);
double linear_interpolation(double *y, double *t, long n, double t0, long i);
#ifdef __cplusplus
}
#endif

void runBinlessRfMode(double **part, int64_t np, RFMODE *rfmode, double Po,
                      char *element_name, double element_z, long pass, long n_passes,
                      CHARGE *charge);
void fillBerencABMatrices(MATRIX *A, MATRIX *B, RFMODE *rfmode, double dt);

void track_through_rfmode(
  double **part0, int64_t np0, RFMODE *rfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  long *Ihist = NULL;  /* array for histogram of particle density */
  double *Vbin = NULL; /* array for voltage acting on each bin */
  long max_n_bins = 0;
  int64_t max_np = 0;
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets = 0, effectiveBuckets, jBucket, adjusting, outputing;
  int64_t np;
  double tOffset;
  /*
    static FILE *fpdeb = NULL;
    static FILE *fpdeb2 = NULL;
    */

  int64_t ip, n_binned = 0;
  long ib, lastBin = 0, firstBin = 0;
  double tmin = 0, tmax, last_tmax, tmean, dt = 0;
  double Vb, V, omega = 0, phase, t, k, damping_factor, tau;
  double VPrevious, tPrevious, phasePrevious;
  double clockNow = 0, clockOffsetPrevious = 0;
  double clockPhaseNow = 0, clockPhasePrevious = 0;
  double V_sum, Vr_sum, Vi_sum, Vg_sum, Vgr_sum, Vgi_sum, Vci_sum, Vcr_sum, Vc_sum;
  double Q_sum;
  long n_summed, max_hist=0, n_occupied;
  double Qrp, VbImagFactor, Q = 0;
  long deltaPass;
  int64_t np_total;
#ifdef HAVE_GPU
  long gpuTracking = 0;
#endif
#if USE_MPI
  long nonEmptyBins = 0;
  MPI_Status mpiStatus;
  int64_t n_binned_global = 0;
#endif

  /* These are here just to quash apparently spurious compiler warnings about possibly using uninitialzed variables */
  tOffset = last_tmax = k = tau = V_sum = Vr_sum = Vi_sum = Vg_sum = Vgr_sum = Vgi_sum = Vci_sum = Vcr_sum = Vc_sum = VbImagFactor = tmean = DBL_MAX;
  n_summed = n_occupied = LONG_MAX;

  /*
    if (!fpdeb) {
      fpdeb = fopen("rfmode.sdds", "w");
      fprintf(fpdeb, "SDDS1\n");
      fprintf(fpdeb, "&column name=t type=double units=s &end\n");
      fprintf(fpdeb, "&column name=VI type=double units=V &end\n");
      fprintf(fpdeb, "&column name=VQ type=double units=V &end\n");
      fprintf(fpdeb, "&column name=V type=double units=V &end\n");
      fprintf(fpdeb, "&column name=phase type=double &end\n");
      fprintf(fpdeb, "&column name=dV type=double units=V &end\n");
      fprintf(fpdeb, "&column name=dPhase type=double &end\n");
      fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
    }
    if (!fpdeb2) {
      fpdeb2 = fopen("rfmode2.sdds", "w");
      fprintf(fpdeb2, "SDDS1\n");
      fprintf(fpdeb2, "&column name=Pass type=long &end\n");
      fprintf(fpdeb2, "&column name=t type=double units=s &end\n");
      fprintf(fpdeb2, "&column name=dt type=double units=s &end\n");
      fprintf(fpdeb2, "&column name=V type=double units=V &end\n");
      fprintf(fpdeb2, "&column name=phase type=double units=V &end\n");
      fprintf(fpdeb2, "&column name=VReal type=double units=V &end\n");
      fprintf(fpdeb2, "&data mode=ascii no_row_counts=1 &end\n");
    }
    */

  if (rfmode->binless) { /* This can't be done in parallel mode */
#if USE_MPI
    printf((char *)"binless in rfmode is not supported in the current parallel version.\n");
    printf((char *)"Please use serial version.\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, T_RFMODE);
#endif
    runBinlessRfMode(part0, np0, rfmode, Po, element_name, element_z, pass, n_passes, charge);
    return;
  }

#if USE_MPI
  MPI_Allreduce(&np0, &np_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
  if (myid == 0) {
    printf("np_total = %" PRId64 "\n", np_total);
    fflush(stdout);
  }
#  endif
#endif
  if (charge) {
    rfmode->mp_charge = charge->macroParticleCharge;
  } else if (pass == 0) {
    rfmode->mp_charge = 0;
    if (rfmode->charge < 0)
      bombElegant((char *)"RFMODE charge parameter should be non-negative. Use change_particle to set particle charge state.", NULL);
#if (!USE_MPI)
    if (np0)
      rfmode->mp_charge = rfmode->charge / np0;
#else
    if (distributedBeam) {
      if (np_total)
        rfmode->mp_charge = rfmode->charge / np_total;
    } else {
      if (np0)
        rfmode->mp_charge = rfmode->charge / np0;
    }
#endif
  }
#ifdef DEBUG
  printf("RFMODE: np0=%" PRId64 ", charge=%le, mp_charge=%le\n", np0, rfmode->charge, rfmode->mp_charge);
#endif

  if (rfmode->fileInitialized && pass == 0) {
#if USE_MPI
    if (myid == 0) {
#endif
      if (rfmode->record && !SDDS_StartPage(rfmode->SDDSrec, rfmode->flush_interval)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"problem starting page for RFMODE record file");
      }
#if USE_MPI
    }
    if (myid == 1) {
#endif
      if (rfmode->driveFrequency > 0 && rfmode->feedbackRecordFile && !SDDS_StartPage(rfmode->SDDSfbrec, rfmode->flush_interval)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"problem starting page for RFMODE feedback record file");
      }
#if USE_MPI
    }
#endif
    rfmode->sample_counter = rfmode->fbSample = 0;
  }

  if (pass % rfmode->pass_interval)
    return;

  if (isMaster) {
    if (rfmode->freq < 1e3 && rfmode->freq)
      printWarningForTracking((char *)"RFMODE frequency is less than 1kHz.",
                              (char *)"This may be an error. Consult manual for units.");
  }

  if (rfmode->mp_charge == 0 && rfmode->voltageSetpoint == 0) {
#ifdef DEBUG
    printf("RFMODE: mp_charge=0, returning\n");
#endif
    return;
  }
  if (rfmode->detuned_until_pass > pass) {
    return;
  }

  if (!rfmode->initialized)
    bombElegant((char *)"track_through_rfmode called with uninitialized element", NULL);
  if (rfmode->Ra)
    rfmode->RaInternal = rfmode->Ra;
  else
    rfmode->RaInternal = 2 * rfmode->Rs;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    if (gpu_rfmode_single_bunch_supported(
          np0, rfmode->bunchedBeamMode, charge))
      gpuTracking = 1;
    else
      part0 = forceParticlesToCpu("RFMODE physical-bunch CPU fallback");
  }
#endif

  if (rfmode->n_bins > max_n_bins) {
#ifdef DEBUG
    printf("Allocating Ihist and Vbin to %ld bins, Ihist=%p, Vbin=%p\n", rfmode->n_bins, (void *)Ihist, (void *)Vbin);
#endif
    Ihist = (long *)trealloc(Ihist, sizeof(*Ihist) * (max_n_bins = rfmode->n_bins));
    Vbin = (double *)trealloc(Vbin, sizeof(*Vbin) * max_n_bins);
#ifdef DEBUG
    printf("Allocated Ihist and Vbin to %ld bins, Ihist=%p, Vbin=%p\n", max_n_bins, (void *)Ihist, (void *)Vbin);
#endif
  }

  if (isSlave || !distributedBeam) {
#ifdef HAVE_GPU
    if (gpuTracking) {
      nBuckets = 1;
    } else
#endif
    {
#ifdef DEBUG
    printf("RFMODE: Determining bucket assignments\n");
#endif
    index_bunch_assignments(part0, np0, (charge && rfmode->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif
    if (rfmode->bunchedBeamMode > 1) {
      if (rfmode->bunchInterval > 0) {
        /* Use pseudo-bunched beam mode---only one bunch is really present */
      } else
        bombElegantVA((char *)"RFMODE %s has invalid values for bunched_beam_mode (>1) and bunch_interval (<=0)\n", element_name);
    }
    }
  }

#if USE_MPI
  /* Master needs to know the number of buckets, check for consistency */
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 1)
    MPI_Send(&nBuckets, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBuckets, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
#endif
#ifdef DEBUG
  printf("RFMODE: nBuckets = %ld\n", nBuckets);
  fflush(stdout);
#endif

  effectiveBuckets = nBuckets == 1 ? rfmode->bunchedBeamMode : nBuckets;
  for (jBucket = 0; jBucket < effectiveBuckets; jBucket++) {
    np = -1;
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#  ifdef DEBUG
    printf("RFMODE: Waiting at top of loop\n");
    fflush(stdout);
#  endif
#endif
    if (nBuckets > 1)
      iBucket = jBucket;
    else
      iBucket = 0;
#ifdef DEBUG
    printf("iBucket = %ld, jBucket = %ld\n", iBucket, jBucket);
    fflush(stdout);
#endif
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
    MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
    printf("np_total = %" PRId64 "\n", np_total);
    fflush(stdout);
#  endif
    if (np_total == 0)
      continue;
#endif

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

      tmean = DBL_MAX;
#ifdef HAVE_GPU
      if (gpuTracking) {
        tmean = gpu_rfmode_time_mean(np, Po);
      } else
#endif
      if (isSlave) {
        for (ip = tmean = 0; ip < np; ip++) {
          tmean += time[ip];
        }
      }

#if USE_MPI
      if (distributedBeam) {
        if (isSlave) {
          double t_total;
          MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, workers);
          MPI_Allreduce(&tmean, &t_total, 1, MPI_DOUBLE, MPI_SUM, workers);
          tmean = t_total;
        }
        tmean /= np_total;
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
#ifdef DEBUG
      printf("computed tmean = %21.15le\n", tmean);
      fflush(stdout);
#endif

      tOffset = 0;
      if (nBuckets == 1 && jBucket) {
        tOffset = rfmode->bunchInterval * jBucket;
        tmean += tOffset;
      }

      tmin = tmean - rfmode->bin_size * rfmode->n_bins / 2.;
      tmax = tmean + rfmode->bin_size * rfmode->n_bins / 2.;

      if (np && iBucket > 0 && tmin < last_tmax) {
#if USE_MPI
        if (myid == 0)
#endif
          bombElegant("Error: time range overlap between buckets\n", NULL);
      }
      last_tmax = tmax;

      if (rfmode->driveFrequency > 0) {
        /* handle generator voltage, cavity feedback */
        if (!rfmode->fbRunning) {
          /* This next statement phases the generator to the first bunch at the desired phase */
          rfmode->tGenerator = rfmode->fbLastTickTime = tmean - (rfmode->readOffset + 0.5) / rfmode->driveFrequency;
          rfmode->fbNextTickTime = rfmode->fbLastTickTime + rfmode->updateInterval / rfmode->driveFrequency;
          rfmode->fbNextTickTimeError = 0;
          rfmode->fbRunning = 1;
        }
        while (tmean > rfmode->fbNextTickTime) {
          /* Need to advance the generator phasors to the next sample time before handling this bunch */
          double Vrl, Vil, omegaDrive, omegaRes, dt, Vgr, Vgi, Vbr, Vbi;
          double IgAmp, IgPhase;
          double VI, VQ;

#ifdef DEBUG
          printf("Advancing feedback ticks\n");
          fflush(stdout);
#endif
          /* Update the voltage using the cavity state-space model */
          m_mult(rfmode->Mt1, rfmode->A, rfmode->Viq);
          m_mult(rfmode->Mt2, rfmode->B, rfmode->Iiq);
          m_add(rfmode->Viq, rfmode->Mt1, rfmode->Mt2);

          rfmode->fbLastTickTime = rfmode->fbNextTickTime;
          rfmode->fbNextTickTime = KahanPlus(rfmode->fbNextTickTime, rfmode->updateInterval / rfmode->driveFrequency, &rfmode->fbNextTickTimeError);

          /** Do feedback **/

          /* Calculate the net voltage and phase at this time */
          omegaRes = PIx2 * rfmode->freq;
          omegaDrive = PIx2 * rfmode->driveFrequency;
          Q = rfmode->Q / (1 + rfmode->beta);
          tau = 2 * Q / omegaRes;
          /* - Calculate beam-induced voltage components (real, imag). */
          dt = rfmode->fbLastTickTime - rfmode->last_t;
          phase = rfmode->last_phase + omegaRes * dt;
          damping_factor = exp(-dt / tau);
#ifdef DEBUG
          printf("Advancing beamloading from %21.15le to %21.15le, tau = %le, DF=%le\n", rfmode->last_t, rfmode->fbLastTickTime, tau, damping_factor);
          printf("Before: Vb=%le\n", rfmode->V);
#endif
          Vrl = (Vbr = damping_factor * rfmode->V * cos(phase));
          Vil = (Vbi = damping_factor * rfmode->V * sin(phase));
          /* - Add generator voltage components (real, imag) */
          dt = rfmode->fbLastTickTime - rfmode->tGenerator;
          Vrl += (Vgr = rfmode->Viq->a[0][0] * cos(omegaDrive * dt) - rfmode->Viq->a[1][0] * sin(omegaDrive * dt));
          Vil += (Vgi = rfmode->Viq->a[0][0] * sin(omegaDrive * dt) + rfmode->Viq->a[1][0] * cos(omegaDrive * dt));

          /* - Compute total voltage amplitude and phase */
          V = sqrt(Vrl * Vrl + Vil * Vil);
          rfmode->fbVCavity = V;
          phase = atan2(Vil, Vrl);
          VI = cos(omegaDrive * dt) * Vrl + sin(omegaDrive * dt) * Vil;
          VQ = -sin(omegaDrive * dt) * Vrl + cos(omegaDrive * dt) * Vil;

          /* parametric receiver noise */
          if (rfmode->nNoise[I_NOISE_ALPHA_V] || rfmode->nNoise[I_NOISE_PHI_V]) {
            if (rfmode->nNoise[I_NOISE_ALPHA_V]) {
              /* amplitude */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_ALPHA_V], rfmode->nNoise[I_NOISE_ALPHA_V], rfmode->fbLastTickTime);
              V *= 1 + linear_interpolation(rfmode->fNoise[I_NOISE_ALPHA_V], rfmode->tNoise[I_NOISE_ALPHA_V], rfmode->nNoise[I_NOISE_ALPHA_V], rfmode->fbLastTickTime, ib);
            }
            if (rfmode->nNoise[I_NOISE_PHI_V]) {
              /* phase */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_PHI_V], rfmode->nNoise[I_NOISE_PHI_V], rfmode->fbLastTickTime);
              phase += linear_interpolation(rfmode->fNoise[I_NOISE_PHI_V], rfmode->tNoise[I_NOISE_PHI_V], rfmode->nNoise[I_NOISE_PHI_V], rfmode->fbLastTickTime, ib);
            }
            Vrl = V * cos(phase);
            Vil = V * sin(phase);
            VI = cos(omegaDrive * dt) * Vrl + sin(omegaDrive * dt) * Vil;
            VQ = -sin(omegaDrive * dt) * Vrl + cos(omegaDrive * dt) * Vil;
          }
          /* additive receiver noise */
          if (rfmode->nNoise[I_NOISE_I_V] || rfmode->nNoise[I_NOISE_Q_V]) {
            if (rfmode->nNoise[I_NOISE_I_V]) {
              /* In-phase */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_I_V], rfmode->nNoise[I_NOISE_I_V], rfmode->fbLastTickTime);
              VI += linear_interpolation(rfmode->fNoise[I_NOISE_I_V], rfmode->tNoise[I_NOISE_I_V], rfmode->nNoise[I_NOISE_I_V], rfmode->fbLastTickTime, ib);
            }
            if (rfmode->nNoise[I_NOISE_Q_V]) {
              /* Quadrature */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_Q_V], rfmode->nNoise[I_NOISE_Q_V], rfmode->fbLastTickTime);
              VQ += linear_interpolation(rfmode->fNoise[I_NOISE_Q_V], rfmode->tNoise[I_NOISE_Q_V], rfmode->nNoise[I_NOISE_Q_V], rfmode->fbLastTickTime, ib);
            }
            Vrl = VI * cos(omegaDrive * dt) - VQ * sin(omegaDrive * dt);
            Vil = VI * sin(omegaDrive * dt) + VQ * cos(omegaDrive * dt);
            V = sqrt(Vrl * Vrl + Vil * Vil);
            phase = atan2(Vil, Vrl);
          }

          if (rfmode->nAmplitudeFilters) {
            /* amplitude/phase feedback */

            /* Calculate updated generator amplitude and phase
                 - Compute errors for voltage amplitude and phase
                 - Run these through the IIR filters
                 - Add to nominal generator amplitude and phase
              */

            IgAmp = sqrt(sqr(rfmode->Ig0->a[0][0]) + sqr(rfmode->Ig0->a[1][0])) + applyIIRFilter(rfmode->amplitudeFilter, rfmode->nAmplitudeFilters,
                                                                                                 rfmode->lambdaA * (rfmode->voltageSetpoint + rfmode->setpointAdjustment - V));
            IgPhase = atan2(rfmode->Ig0->a[1][0], rfmode->Ig0->a[0][0]) + applyIIRFilter(rfmode->phaseFilter, rfmode->nPhaseFilters, PI / 180 * rfmode->phaseSetpoint - 3 * PI / 2 - phase);

            if (rfmode->muteGenerator >= 0 && rfmode->muteGenerator <= pass) {
              if (rfmode->muteGenerator == pass) {
                printf("Generator muted for RFMODE %s on pass %ld\n", element_name, pass);
                fflush(stdout);
              }
              IgAmp = 0;
            }
            IgAmp *= rfmode->generatorFactor;

            /* Calculate updated I/Q components for generator current */
            rfmode->Iiq->a[0][0] = IgAmp * cos(IgPhase);
            rfmode->Iiq->a[1][0] = IgAmp * sin(IgPhase);
          } else {
            /* I/Q feedback */
            double VISetpoint, VQSetpoint; /* equivalent in-phase and quadrature setpoints */
            double phaseg;
            double dII, dIQ;

            /* convert from V*sin(phi) to V*cos(phi) convention and account for fact that the
               * feedback is performed in the nominally empty part of each bucket (i.e., 180 degrees before 
               * the nominal beam phase)
               */
            phaseg = PI / 180 * rfmode->phaseSetpoint - 3 * PI / 2;
            VISetpoint = (rfmode->voltageSetpoint + rfmode->setpointAdjustment) * cos(phaseg);
            VQSetpoint = (rfmode->voltageSetpoint + rfmode->setpointAdjustment) * sin(phaseg);

            rfmode->Iiq->a[0][0] = rfmode->Ig0->a[0][0] + (dII = applyIIRFilter(rfmode->IFilter, rfmode->nIFilters, rfmode->lambdaA * (VISetpoint - VI)));
            rfmode->Iiq->a[1][0] = rfmode->Ig0->a[1][0] + (dIQ = applyIIRFilter(rfmode->QFilter, rfmode->nQFilters, rfmode->lambdaA * (VQSetpoint - VQ)));

            if (rfmode->muteGenerator >= 0 && rfmode->muteGenerator <= pass) {
              if (rfmode->muteGenerator == pass) {
                printf("Generator muted for RFMODE %s on pass %ld\n", element_name, pass);
                fflush(stdout);
              }
              rfmode->Iiq->a[0][0] = rfmode->Iiq->a[1][0] = 0;
            }

#ifdef DEBUG
            printf("dII = %le, dIQ = %le\n", dII, dIQ);
#endif
            IgAmp = sqrt(sqr(rfmode->Iiq->a[0][0]) + sqr(rfmode->Iiq->a[1][0]));
            IgPhase = atan2(rfmode->Iiq->a[1][0], rfmode->Iiq->a[0][0]);
          }

          /* Parametric generator noise */
          if (rfmode->nNoise[I_NOISE_ALPHA_GEN] || rfmode->nNoise[I_NOISE_PHI_GEN]) {
            if (rfmode->nNoise[I_NOISE_ALPHA_GEN]) {
              /* amplitude */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_ALPHA_GEN], rfmode->nNoise[I_NOISE_ALPHA_GEN], rfmode->fbLastTickTime);
              IgAmp *= 1 + linear_interpolation(rfmode->fNoise[I_NOISE_ALPHA_GEN], rfmode->tNoise[I_NOISE_ALPHA_GEN], rfmode->nNoise[I_NOISE_ALPHA_GEN], rfmode->fbLastTickTime, ib);
            }
            if (rfmode->nNoise[I_NOISE_PHI_GEN]) {
              /* phase */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_PHI_GEN], rfmode->nNoise[I_NOISE_PHI_GEN], rfmode->fbLastTickTime);
              IgPhase += linear_interpolation(rfmode->fNoise[I_NOISE_PHI_GEN], rfmode->tNoise[I_NOISE_PHI_GEN], rfmode->nNoise[I_NOISE_PHI_GEN], rfmode->fbLastTickTime, ib);
            }
            /* Calculate updated I/Q components for generator current */
            rfmode->Iiq->a[0][0] = IgAmp * cos(IgPhase);
            rfmode->Iiq->a[1][0] = IgAmp * sin(IgPhase);
          }
          if (rfmode->nNoise[I_NOISE_I_GEN] || rfmode->nNoise[I_NOISE_Q_GEN]) {
            /* Additive generator noise */
            if (rfmode->nNoise[I_NOISE_I_GEN]) {
              /* In-phase */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_I_GEN], rfmode->nNoise[I_NOISE_I_GEN], rfmode->fbLastTickTime);
              rfmode->Iiq->a[0][0] += linear_interpolation(rfmode->fNoise[I_NOISE_I_GEN], rfmode->tNoise[I_NOISE_I_GEN], rfmode->nNoise[I_NOISE_I_GEN], rfmode->fbLastTickTime, ib);
            }
            if (rfmode->nNoise[I_NOISE_Q_GEN]) {
              /* Quadrature */
              ib = find_nearby_array_entry(rfmode->tNoise[I_NOISE_Q_GEN], rfmode->nNoise[I_NOISE_Q_GEN], rfmode->fbLastTickTime);
              rfmode->Iiq->a[1][0] += linear_interpolation(rfmode->fNoise[I_NOISE_Q_GEN], rfmode->tNoise[I_NOISE_Q_GEN], rfmode->nNoise[I_NOISE_Q_GEN], rfmode->fbLastTickTime, ib);
            }
            IgAmp = sqrt(sqr(rfmode->Iiq->a[0][0]) + sqr(rfmode->Iiq->a[1][0]));
            IgPhase = atan2(rfmode->Iiq->a[1][0], rfmode->Iiq->a[0][0]);
          }

          if (rfmode->driveFrequency && rfmode->feedbackRecordFile) {
#if USE_MPI
            if (myid == 1) {
#endif
              if ((rfmode->fbSample + 1) % rfmode->flush_interval == 0) {
                if (!SDDS_UpdatePage(rfmode->SDDSfbrec, FLUSH_TABLE)) {
                  SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
                  SDDS_Bomb((char *)"problem flushing RFMODE feedback record file");
                }
              }
              if (!SDDS_SetRowValues(rfmode->SDDSfbrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                     rfmode->fbSample,
                                     (char *)"Pass", pass, (char *)"t", rfmode->fbLastTickTime,
                                     (char *)"fResonance", rfmode->freq,
                                     (char *)"fDrive", rfmode->driveFrequency,
                                     (char *)"VbReal", Vbr, (char *)"VbImag", Vbi,
                                     (char *)"VgReal", Vgr, (char *)"VgImag", Vgi,
                                     (char *)"VCavity", V, (char *)"PhaseCavity", phase,
                                     (char *)"IgAmplitude", IgAmp,
                                     (char *)"IgPhase", IgPhase,
                                     NULL)) {
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
                SDDS_Bomb((char *)"problem setting values for feedback record file");
              }
              /*
                  if ((rfmode->fbSample%1000==0 || (pass==(n_passes-1) && iBucket==(nBuckets-1)))
                  && !SDDS_UpdatePage(rfmode->SDDSfbrec, FLUSH_TABLE)) {
                  SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
                  printf("Warning: problem writing data for RFMODE feedback record file, row %ld\n", rfmode->fbSample);
                  }
                */
              rfmode->fbSample++;
#if USE_MPI
            }
#endif
          }

          /*
              if (isnan(rfmode->Iiq->a[0][0]) || isnan(rfmode->Iiq->a[1][0]) || isinf(rfmode->Iiq->a[0][0]) || isinf(rfmode->Iiq->a[1][0])) {
              printf("V = %le, setpoint = %le\n", V, rfmode->voltageSetpoint);
              printf("phase = %le, setpoints = %le\n", phase, rfmode->phaseg);
              printf("Viq = %le, %le\n", rfmode->Viq->a[0][0], rfmode->Viq->a[1][0]);
              printf("Iiq = %le, %le\n", rfmode->Iiq->a[0][0], rfmode->Iiq->a[1][0]);
              exit(1);
              }
            */
#ifdef DEBUG
          printf("After tick: Vb = %le V, Vbr = %le, Vbi = %le, phase = %21.15le, t=%21.15le s, tNext=%21.15le\n",
                 sqrt(Vbr * Vbr + Vbi * Vbi), Vbr, Vbi, atan2(Vbi, Vbr), rfmode->fbLastTickTime, rfmode->fbNextTickTime);
#endif
        } /* end while loop for feedback ticks */
      }

      if (isSlave) {
#ifdef DEBUG
        printf("tmin = %21.15le, tmax = %21.15le, tmean = %21.15le\n", tmin, tmax, tmean);
        fflush(stdout);
#endif

        dt = (tmax - tmin) / rfmode->n_bins;
        n_binned = lastBin = 0;
        firstBin = rfmode->n_bins;
#if USE_MPI
        nonEmptyBins = 0;
#endif
        if (np == -1) {
          /* This shouldn't happen, but compiler says it might... */
          bombElegant("np==-1 in track_through_rfmode. Seek professional help!\n", NULL);
        }
#ifdef HAVE_GPU
        if (gpuTracking) {
          n_binned = gpu_rfmode_histogram(
            np, Po, rfmode->n_bins, tmin, dt,
            Ihist, &firstBin, &lastBin);
        } else
#endif
        {
          for (ib = 0; ib < rfmode->n_bins; ib++)
            Ihist[ib] = 0;
          for (ip = 0; ip < np; ip++) {
            pbin[ip] = -1;
            ib = (long)((time[ip] + tOffset - tmin) / dt);
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
            n_binned++;
          }
        }
#if USE_MPI && MPI_DEBUG
        /*
	  printf("Histogram Ihist has %ld  particles, %ld binned particles, firstBin=%ld, lastBin=%ld\n",
		 np, n_binned, firstBin, lastBin);
	  for (ib=firstBin; ib<=lastBin; ib++) 
	    printf("Ihist[%ld] = %ld\n", ib, Ihist[ib]);
	  */
#endif
        V_sum = Vr_sum = Vi_sum = Q_sum = Vg_sum = Vgr_sum = Vgi_sum = Vci_sum = Vcr_sum = Vc_sum = 0;
        n_summed = max_hist = n_occupied = 0;

        /* find frequency and Q at this time */
        omega = PIx2 * rfmode->freq;
        if (rfmode->nFreq) {
          double omegaFactor;
          ib = find_nearby_array_entry(rfmode->tFreq, rfmode->nFreq, tmean);
          omega *= (omegaFactor = linear_interpolation(rfmode->fFreq, rfmode->tFreq, rfmode->nFreq, tmean, ib));
          /* keeps stored energy constant for constant R/Q */
          rfmode->V *= sqrt(omegaFactor);
        }
        Q = rfmode->Q / (1 + rfmode->beta);
        if (rfmode->nQ) {
          ib = find_nearby_array_entry(rfmode->tQ, rfmode->nQ, tmean);
          Q *= linear_interpolation(rfmode->fQ, rfmode->tQ, rfmode->nQ, tmean, ib);
        }
      }
#if (!USE_MPI)
      if (Q < 0.5) {
        printf((char *)"The effective Q<=0.5 for RFMODE.  Use the ZLONGIT element.\n");
        fflush(stdout);
        exitElegant(1);
      }
#else
      if (myid == 1) { /* Let the first slave processor write the output */
        if (Q < 0.5) {
          dup2(fdStdout, fileno(stdout));
          printf((char *)"The effective Q<=0.5 for RFMODE.  Use the ZLONGIT element.\n");
          fflush(stdout);
          if (!freopen("/dev/null", "w", stdout)) {
            perror("freopen failed");
            exit(EXIT_FAILURE);
          }
          MPI_Abort(MPI_COMM_WORLD, T_RFMODE);
        }
      }
#endif
      tau = 2 * Q / omega;
      k = omega / 4 * (rfmode->RaInternal) / rfmode->Q;
      if ((deltaPass = (pass - rfmode->detuned_until_pass)) <= (rfmode->rampPasses - 1))
        k *= (deltaPass + 1.0) / rfmode->rampPasses;

      /* These adjustments per Zotter and Kheifets, 3.2.4 */
      Qrp = sqrt(Q * Q - 0.25);
      VbImagFactor = 1 / (2 * Qrp);
      omega *= Qrp / Q;

      /* trackingClockOffset() is constant within a step; the macro inter-step gap
         it represents must enter the cavity damping (true elapsed time) but never
         the phase, the bin times, or the coord[4] write-back, which stay reduced. */
      clockNow = trackingClockOffset();
      /* Reduced clock phase for THIS mode's (Q-shifted) omega; a non-harmonic
         mode adds its across-turn difference to the reduced-time phase advance
         so CHANGE_T's whole-period removal is transparent (not just harmonics). */
      clockPhaseNow = trackingClockPhase(omega);

      if (rfmode->single_pass) {
        rfmode->V = rfmode->last_phase = 0;
        rfmode->last_t = tmin + 0.5 * dt;
        rfmode->last_clockOffset = clockNow;
        rfmode->last_clockPhase = clockPhaseNow;
      }
    }

#if USE_MPI
#  ifdef DEBUG
    printf("Checking for unbinned particles\n");
    fflush(stdout);
#  endif
    if (myid == 0)
      n_binned = np = 0;
    MPI_Allreduce(&n_binned, &n_binned_global, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
    printf("n_binned = %" PRId64 ", n_binned_global = %" PRId64 ", np = %" PRId64 ", np_total = %" PRId64 "\n",
           n_binned, n_binned_global, np, np_total);
    fflush(stdout);
#  endif
    if (n_binned_global != np_total) {
      TRACKING_CONTEXT tcontext;
      getTrackingContext(&tcontext);
      if (myid == 0) {
        printf("%" PRId64 " of %" PRId64 " particles outside of binning region in RFMODE %s #%ld. Consider increasing number of bins.\n",
               np_total - n_binned_global,
               np_total, tcontext.elementName, tcontext.elementOccurrence);
        printf("Also, check particleID assignments for bunch identification. Bunches should be on separate pages of the input file.\n");
        fflush(stdout);
      }
      if (!rfmode->allowUnbinnedParticles) {
        mpiAbort = MPI_ABORT_BUNCH_TOO_LONG_RFMODE;
        MPI_Abort(MPI_COMM_WORLD, T_RFMODE);
      }
    }
#else
    if (n_binned != np) {
      TRACKING_CONTEXT tcontext;
      getTrackingContext(&tcontext);
      if (!rfmode->allowUnbinnedParticles) {
        bombElegantVA((char *)"%" PRId64 " of %" PRId64 " particles  outside of binning region in RFMODE %s #%ld. Consider increasing number of bins. Also, particleID assignments should be checked.", np - n_binned, np, tcontext.elementName, tcontext.elementOccurrence);
      } else {
      printf("%" PRId64 " of %" PRId64 " particles  outside of binning region in RFMODE %s #%ld. Consider increasing number of bins. Also, particleID assignments should be checked.\n", np - n_binned, np, tcontext.elementName, tcontext.elementOccurrence);
        fflush(stdout);
      }
    }
#endif

#if USE_MPI
#  ifdef DEBUG
    printf("About to wait on barrier (0)\n");
    fflush(stdout);
#  endif
    MPI_Barrier(MPI_COMM_WORLD);
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
#  ifdef DEBUG
      printf("Finding first and last bin globally\n");
      fflush(stdout);
#  endif
      MPI_Allreduce(&lastBin, &lastBin_global, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&firstBin, &firstBin_global, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
      firstBin = firstBin_global;
      lastBin = lastBin_global;
#  ifdef DEBUG
      printf("bucket = %ld, firstBin = %ld, lastBin = %ld\n", iBucket, firstBin_global, lastBin_global);
      fflush(stdout);
      if (myid != 0) {
        printf("%" PRId64 " particles binned\n", n_binned);
        fflush(stdout);
      }
#  endif
      if (distributedBeam) {
        /* Sum the per-worker histograms.  Only slaves are in the "workers" communicator;
         * "workers" is MPI_COMM_NULL on the master, so it must never call this collective.
         * When the beam is NOT distributed (e.g. fewer particles than workers, so every rank
         * tracks the full beam), each rank already holds the complete histogram and no
         * cross-worker sum is needed. */
        if (isSlave) {
          long *buffer;
          buffer = (long *)calloc(lastBin - firstBin + 1, sizeof(long));
          MPI_Allreduce(&Ihist[firstBin], buffer, lastBin - firstBin + 1, MPI_LONG, MPI_SUM, workers);
          memcpy(Ihist + firstBin, buffer, sizeof(long) * (lastBin - firstBin + 1));
          free(buffer);
        }
      }
#  ifdef DEBUG
      printf("Summed histogram across processors\n");
      fflush(stdout);
      printf("%" PRId64 " particles binned\n", n_binned);
      for (ib = firstBin; ib <= lastBin; ib++)
        printf("%ld %ld\n", ib, Ihist[ib]);
      fflush(stdout);
#  endif
    }
#else
#  ifdef DEBUG
    printf("%" PRId64 " particles binned\n", n_binned);
    for (ib = firstBin; ib <= lastBin; ib++)
      printf("%ld %ld\n", ib, Ihist[ib]);
    fflush(stdout);
#  endif
#endif

    if (isSlave || !distributedBeam) {
      /* These values are fixed and can be used to compute the effect on the beam of
         * the "long-range" fields (previous turns) only
         */
      VPrevious = rfmode->V;
      tPrevious = rfmode->last_t;
      clockOffsetPrevious = rfmode->last_clockOffset;
      clockPhasePrevious = rfmode->last_clockPhase;
      phasePrevious = rfmode->last_phase;
#ifdef DEBUG
      printf("VPrevious = %le, tPrevious = %le, phasePrevious = %le, firstBin = %ld, lastBin = %ld\n",
             VPrevious, tPrevious, phasePrevious, firstBin, lastBin);
#endif

      if (rfmode->driveFrequency > 0) {
        /* compute generator voltage I and Q envelopes at bunch center */
        fillBerencABMatrices(rfmode->At, rfmode->Bt, rfmode, tmean - rfmode->fbLastTickTime);
        m_mult(rfmode->Mt1, rfmode->At, rfmode->Viq);
        m_mult(rfmode->Mt2, rfmode->Bt, rfmode->Iiq);
        m_add(rfmode->Mt3, rfmode->Mt1, rfmode->Mt2);
      }

      for (ib = firstBin; ib <= lastBin; ib++) {
        t = tmin + (ib + 0.5) * dt; /* middle arrival time for this bin */
        if (!Ihist[ib] && !rfmode->interpolate)
          continue;
        if (Ihist[ib] > max_hist)
          max_hist = Ihist[ib];
        if (Ihist[ib])
          n_occupied++;

          /* advance cavity to this time */
#ifdef DEBUG
        printf("Advancing beamloading from %21.15le to %21.15le, dt=%le, tau=%le, last_phase=%le\n",
               rfmode->last_t, t, t - rfmode->last_t, tau, rfmode->last_phase);
        printf("Before: Vb = %le V, Vbr = %le, Vbi = %le, phase = %21.15le, t=%21.15les\n",
               rfmode->V, rfmode->Vr, rfmode->Vi, fmod(rfmode->last_phase, PIx2), rfmode->last_t);
#endif
        phase = rfmode->last_phase + omega * (t - rfmode->last_t) + (clockPhaseNow - rfmode->last_clockPhase);
        damping_factor = exp(-((t - rfmode->last_t) + (clockNow - rfmode->last_clockOffset)) / tau);
        rfmode->last_t = t;
        rfmode->last_clockOffset = clockNow;
        rfmode->last_clockPhase = clockPhaseNow;
        rfmode->last_phase = phase;
        rfmode->V = V = rfmode->V * damping_factor;
        rfmode->Vr = V * cos(phase);
        rfmode->Vi = V * sin(phase);
#ifdef DEBUG
        printf("After: Vb = %le V, Vbr = %le, Vbi = %le, phase = %21.15le, t=%21.15les, tau=%le, DF=%le\n",
               rfmode->V, rfmode->Vr, rfmode->Vi, fmod(rfmode->last_phase, PIx2), rfmode->last_t, tau, damping_factor);
#endif

        /* compute beam-induced voltage for this bin */
        Vb = 2 * k * rfmode->mp_charge * particleRelSign * rfmode->pass_interval * Ihist[ib];
        if (rfmode->long_range_only)
          Vbin[ib] = VPrevious * (damping_factor = exp(-((t - tPrevious) + (clockNow - clockOffsetPrevious)) / tau)) * cos(phasePrevious + omega * (t - tPrevious) + (clockPhaseNow - clockPhasePrevious));
        else
          Vbin[ib] = rfmode->Vr - Vb / 2;

        if (rfmode->driveFrequency > 0) {
          /* add generator-induced voltage */
          double Vr, Vi, dt;
          dt = t - rfmode->tGenerator;
          Vg_sum += Ihist[ib] * sqrt(sqr(rfmode->Mt3->a[0][0]) + sqr(rfmode->Mt3->a[1][0]));
          Vr = rfmode->Mt3->a[0][0] * cos(PIx2 * rfmode->driveFrequency * dt) - rfmode->Mt3->a[1][0] * sin(PIx2 * rfmode->driveFrequency * dt);
          Vi = rfmode->Mt3->a[0][0] * sin(PIx2 * rfmode->driveFrequency * dt) + rfmode->Mt3->a[1][0] * cos(PIx2 * rfmode->driveFrequency * dt);
          Vbin[ib] += Vr;
          Vgi_sum += Ihist[ib] * Vi;
          Vgr_sum += Ihist[ib] * Vr;
          Vcr_sum += Ihist[ib] * (Vr + rfmode->Vr - Vb / 2);
          Vci_sum += Ihist[ib] * (Vi + rfmode->Vi - Vb * VbImagFactor / 2);
          Vc_sum += Ihist[ib] * sqrt(sqr(Vr + rfmode->Vr - Vb / 2) + sqr(Vi + rfmode->Vi - Vb * VbImagFactor / 2));
          /* fprintf(fpdeb2, "%ld %21.15le %21.15le %le %le %le\n", pass, t, dt, sqrt(Vi*Vi+Vr*Vr), atan2(Vi, Vr), Vr); */
        }

        /* add slice contribution to beam-induced voltage */
        rfmode->Vr -= Vb;
        rfmode->Vi -= Vb * VbImagFactor;
        rfmode->last_phase = atan2(rfmode->Vi, rfmode->Vr);
        rfmode->V = sqrt(sqr(rfmode->Vr) + sqr(rfmode->Vi));
#ifdef DEBUG
        printf("BL+: Vb = %le V, Vbr = %le, Vbi = %le, phase = %21.15le, t=%21.15les, tau=%le, DF=%le\n",
               rfmode->V, rfmode->Vr, rfmode->Vi, fmod(rfmode->last_phase, PIx2), rfmode->last_t, tau, damping_factor);
#endif

        V_sum += Ihist[ib] * rfmode->V;
        Vr_sum += Ihist[ib] * rfmode->Vr;
        Vi_sum += Ihist[ib] * rfmode->Vi;
        Q_sum += Ihist[ib] * rfmode->mp_charge * particleRelSign;
        n_summed += Ihist[ib];
      }
      /*
        fprintf(fpdeb, "\n");
        fflush(fpdeb);
        */
#ifdef DEBUG
      printf("Computed voltage values in bins\n");
      fflush(stdout);
#endif

      if (rfmode->rigid_until_pass <= pass) {
        /* change particle momentum offsets to reflect voltage in relevant bin */
        /* also recompute slopes for new momentum to conserve transverse momentum */
#ifdef HAVE_GPU
        if (gpuTracking) {
          if (firstBin <= lastBin)
            gpu_rfmode_apply_kicks(
              np, Po, rfmode->n_bins, tmin, dt,
              firstBin, lastBin, rfmode->interpolate,
              rfmode->n_cavities, Vbin);
        } else
#endif
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip = 0; ip < np; ip++) {
          double particleV;
          /* compute voltage seen by this particle */
          if (rfmode->interpolate) {
            long ib1, ib2;
            long particleBin;
            double dt1;
            particleBin = pbin[ip];
            dt1 = time[ip] + tOffset - (tmin + dt * (particleBin + 0.5));
            if (dt1 < 0) {
              ib1 = particleBin - 1;
              ib2 = particleBin;
            } else {
              ib1 = particleBin;
              ib2 = particleBin + 1;
            }
            if (ib2 > lastBin) {
              ib2--;
              ib1--;
            }
            if (ib1 < firstBin) {
              ib1++;
              ib2++;
            }
            dt1 = time[ip] + tOffset - (tmin + dt * (ib1 + 0.5));
            particleV = Vbin[ib1] + (Vbin[ib2] - Vbin[ib1]) / dt * dt1;
          } else {
            particleV = Vbin[pbin[ip]];
          }
          if (iBucket == jBucket) {
            double particleDgamma = rfmode->n_cavities * particleV / (1e6 * particleMassMV * particleRelSign);
            add_to_particle_energy(part[ip], time[ip], Po, particleDgamma);
          }
        }
      }

#ifdef DEBUG
      printf("Applied voltages to particles\n");
      fflush(stdout);
#endif

      if (nBuckets != 1 && iBucket == jBucket) {
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }

    adjusting = (rfmode->adjustmentFraction > 0 && pass >= rfmode->adjustmentStart && pass <= rfmode->adjustmentEnd &&
                 rfmode->adjustmentInterval > 0 && pass % rfmode->adjustmentInterval == 0 && jBucket == 0);
    if (adjusting && pass == rfmode->adjustmentStart)
      rfmode->setpointAdjustment = 0;
    outputing = (rfmode->record && (pass % rfmode->sample_interval) == 0);
    if (outputing || adjusting) {
#if USE_MPI
#  define SR_BUFLEN 17
      double sendBuffer[SR_BUFLEN], receiveBuffer[SR_BUFLEN];
#endif

#if USE_MPI
      if (myid == 0)
#endif
        if (outputing) {
          if (!SDDS_UpdatePage(rfmode->SDDSrec, FLUSH_TABLE)) {
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb((char *)"problem flushing RFMODE record file");
          }
        }

      np_total = np; /* Used by serial version */
#if (USE_MPI)
      /* Sum data across all cores */
      if (myid == 0) {
        memset(sendBuffer, 0, sizeof(sendBuffer[0]) * SR_BUFLEN);
      } else {
        sendBuffer[0] = n_binned;
        sendBuffer[1] = rfmode->V;
        sendBuffer[2] = rfmode->last_phase;
        sendBuffer[3] = rfmode->last_t;
        sendBuffer[4] = V_sum;
        sendBuffer[5] = Vr_sum;
        sendBuffer[6] = Vi_sum;
        sendBuffer[7] = np;
        sendBuffer[8] = Vgr_sum;
        sendBuffer[9] = Vgi_sum;
        sendBuffer[10] = Vcr_sum;
        sendBuffer[11] = Vci_sum;
        sendBuffer[12] = n_occupied;
        sendBuffer[13] = n_summed;
        sendBuffer[14] = Vg_sum;
        sendBuffer[15] = rfmode->fbVCavity;
        sendBuffer[16] = Vc_sum;
      }
      if (adjusting) {
        MPI_Allreduce(sendBuffer, receiveBuffer, SR_BUFLEN, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      } else
        MPI_Reduce(sendBuffer, receiveBuffer, SR_BUFLEN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (adjusting || (myid == 0)) {
        n_binned = receiveBuffer[0];
        if (myid == 0) {
          rfmode->fbVCavity = receiveBuffer[15] / (n_processors - 1);
          rfmode->V = receiveBuffer[1] / (n_processors - 1);
          rfmode->last_phase = receiveBuffer[2] / (n_processors - 1);
          rfmode->last_t = receiveBuffer[3] / (n_processors - 1);
        }
        V_sum = receiveBuffer[4];
        Vr_sum = receiveBuffer[5];
        Vi_sum = receiveBuffer[6];
        np_total = receiveBuffer[7];
        Vgr_sum = receiveBuffer[8];
        Vgi_sum = receiveBuffer[9];
        Vcr_sum = receiveBuffer[10];
        Vci_sum = receiveBuffer[11];
        Vc_sum = receiveBuffer[16];
        n_occupied = receiveBuffer[12];
        n_summed = receiveBuffer[13];
        Vg_sum = receiveBuffer[14];
      }
      if (myid == 0) {
#endif
#ifdef DEBUG
        printf("Writing record file\n");
        fflush(stdout);
#endif
        if (!SDDS_SetRowValues(rfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                               rfmode->sample_counter++,
                               (char *)"Bunch", jBucket, (char *)"Pass", pass, (char *)"NumberOccupied", n_occupied,
                               (char *)"FractionBinned", np_total ? (1.0 * n_binned) / np_total : 0.0,
                               (char *)"VPostBeam", rfmode->V,
                               (char *)"PhasePostBeam", rfmode->last_phase,
                               (char *)"tPostBeam", rfmode->last_t,
                               (char *)"V", n_summed ? V_sum / n_summed : 0.0,
                               (char *)"VReal", n_summed ? Vr_sum / n_summed : 0.0,
                               (char *)"Phase", n_summed ? atan2(Vi_sum / n_summed, Vr_sum / n_summed) : 0.0,
                               (char *)"Charge", rfmode->mp_charge * np_total,
                               NULL) ||
            (rfmode->driveFrequency > 0 &&
             !SDDS_SetRowValues(rfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                rfmode->sample_counter - 1,
                                (char *)"VGenerator", n_summed ? Vg_sum / n_summed : 0.0,
                                (char *)"PhaseGenerator", n_summed ? atan2(Vgi_sum / n_summed, Vgr_sum / n_summed) : 0.0,
                                (char *)"VCavity", n_summed ? Vc_sum / n_summed : 0.0,
                                (char *)"PhaseCavity", n_summed ? atan2(Vci_sum / n_summed, Vcr_sum / n_summed) : 0.0,
                                NULL))) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          printWarningForTracking((char *)"Problem setting up data for RFMODE record file", NULL);
        }
        /*
            if (rfmode->sample_counter%100==0 && !SDDS_UpdatePage(rfmode->SDDSrec, 0)) {
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
              printf("Warning: problem writing data for RFMODE record file, row %ld\n", rfmode->sample_counter);
            }
            */
#ifdef DEBUG
        printf("Done writing record file\n");
        fflush(stdout);
#endif
#if USE_MPI
      }
#endif
      if (adjusting) {
        /* adjust the voltage setpoint to get the voltage we really want */
        double VcEffective;
        VcEffective = n_summed ? Vc_sum / n_summed : 0.0;
        rfmode->setpointAdjustment += (rfmode->voltageSetpoint - VcEffective) * rfmode->adjustmentFraction;
        printf("Voltage setpoint adjustment changed to %le V on pass %ld\n", rfmode->setpointAdjustment, pass);
        fflush(stdout);
      }
    }

#if USE_MPI
#  ifdef DEBUG
    printf("Preparing to wait on barrier (1)\n");
    fflush(stdout);
#  endif
    MPI_Barrier(MPI_COMM_WORLD);
#  ifdef DEBUG
    printf("Finished waiting on barrier (1), iBucket=%ld, jBucket=%ld\n", iBucket, jBucket);
    fflush(stdout);
#  endif
#endif
  }

#if USE_MPI
#  ifdef DEBUG
  printf("Preparing to wait on barrier (2)\n");
  fflush(stdout);
#  endif
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef DEBUG
  printf("Finished waiting on barrier (2)\n");
  fflush(stdout);
#endif

  if (rfmode->record) {

#if USE_MPI
    if (myid == 0) {
#endif
      if (pass == n_passes - 1) {
        if (!SDDS_UpdatePage(rfmode->SDDSrec, FLUSH_TABLE)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"problem writing data for RFMODE record file");
        }
      }
#if USE_MPI
    }
#endif
  }

#ifdef DEBUG
  printf("RFMODE: Exited bunch loop\n");
  fflush(stdout);
#endif

  if (Ihist)
    free(Ihist);
  if (Vbin)
    free(Vbin);
  if (part && part != part0)
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
}

void set_up_rfmode(RFMODE *rfmode, char *element_name, double element_z, long n_passes,
                   RUN *run, int64_t n_particles,
                   double Po, double total_length) {
  long i;
  double T;
  TABLE data;

  if (rfmode->initialized)
    return;

  rfmode->V = rfmode->Vr = rfmode->Vi = rfmode->last_t = rfmode->last_phase = rfmode->last_omega = rfmode->last_Q = 0;
  rfmode->last_clockOffset = 0;
  rfmode->last_clockPhase = 0;

  rfmode->initialized = 1;
  if (rfmode->pass_interval <= 0)
    bombElegant((char *)"pass_interval <= 0 for RFMODE", NULL);
  if (rfmode->long_range_only) {
    if (rfmode->binless)
      bombElegantVA((char *)"Error: binless and long-range modes are incompatible in RFMODE (element %s)\n", element_name);
    if (rfmode->single_pass)
      bombElegantVA((char *)"single-pass and long-range modes are incompatible in RFMODE (element %s)\n", element_name);
  }
  if (rfmode->driveFrequency > 0) {
    if (rfmode->Qwaveform || rfmode->fwaveform)
      bombElegantVA((char *)"Error: Unfortunately, can't do Q_WAVEFORM or FREQ_WAVEFORM with RF feedback for RFMODE (element %s)\n", element_name);
    if (rfmode->binless)
      bombElegantVA((char *)"Error: Unfortunately, can't use BINLESS mode with RF feedback for RFMODE (element %s)\n", element_name);
  }

#if !SDDS_MPI_IO
  if (n_particles < 1)
    bombElegant((char *)"too few particles in set_up_rfmode()", NULL);
#endif
  if (rfmode->n_bins < 2)
    bombElegant((char *)"too few bins for RFMODE", NULL);
  if (rfmode->bin_size <= 0 && !rfmode->binless)
    bombElegant((char *)"bin_size must be positive for RFMODE", NULL);
  if (rfmode->Ra && rfmode->Rs)
    bombElegant((char *)"RFMODE element may have only one of Ra or Rs nonzero.  Ra is just 2*Rs", NULL);
  if (rfmode->Ra)
    rfmode->RaInternal = rfmode->Ra;
  else
    rfmode->RaInternal = 2 * rfmode->Rs;
  if (rfmode->bin_size * rfmode->freq > 0.1) {
    T = rfmode->bin_size * rfmode->n_bins;
    rfmode->bin_size = 0.1 / rfmode->freq;
    rfmode->n_bins = ((long)(T / rfmode->bin_size + 1));
    rfmode->bin_size = T / rfmode->n_bins;
    printf((char *)"The RFMODE %s bin size is too large--setting to %e and increasing to %ld bins\n",
           element_name, rfmode->bin_size, rfmode->n_bins);
    printf((char *)"Total span changed from %le to %le\n",
           T, rfmode->n_bins * rfmode->bin_size);
    fflush(stdout);
  }
  if (rfmode->sample_interval <= 0)
    rfmode->sample_interval = 1;
  if (!rfmode->fileInitialized) {
    if (rfmode->record && strlen(rfmode->record)==0)
      rfmode->record = NULL;
    if (rfmode->record) {
      char *filename;
      filename = compose_filename(rfmode->record, run->rootname);
#if (USE_MPI)
      if (myid == 0)
#endif
        if (!(rfmode->SDDSrec = (SDDS_DATASET *)tmalloc(sizeof(*(rfmode->SDDSrec)))) ||
            !SDDS_InitializeOutputElegant(rfmode->SDDSrec, SDDS_BINARY, 1, NULL, NULL, filename) ||
            SDDS_DefineParameter(rfmode->SDDSrec, (char *)"SVNVersion", NULL, NULL, (char *)"SVN version number", NULL, SDDS_STRING, (char *)SVN_VERSION) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"Bunch", NULL, NULL, (char *)"Bunch number", NULL, SDDS_LONG, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"Pass", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"NumberOccupied", NULL, NULL, (char *)"Number of bins that are occupied", NULL, SDDS_LONG, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"FractionBinned", NULL, NULL, (char *)"Fraction of particles that are binned", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"V", NULL, (char *)"V", (char *)"Beam-induced voltage seen by the bunch", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"VReal", NULL, (char *)"V", (char *)"Real part of beam-induced voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"Phase", NULL, (char *)"rad", (char *)"Phase of beam-induced voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"VPostBeam", NULL, (char *)"V", (char *)"Beam-induced voltage after bunch passage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"PhasePostBeam", NULL, (char *)"rad", (char *)"Phase of beam-induced voltage after bunch passage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"tPostBeam", NULL, (char *)"s", (char *)"Time at which VPostBeam and PhasePostBeam hold", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSrec, (char *)"Charge", NULL, (char *)"C", (char *)"Bunch charge", NULL, SDDS_DOUBLE, 0) < 0 ||
            (rfmode->driveFrequency > 0 &&
             (SDDS_DefineColumn(rfmode->SDDSrec, (char *)"VGenerator", NULL, (char *)"V", (char *)"Generator voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
              SDDS_DefineColumn(rfmode->SDDSrec, (char *)"PhaseGenerator", NULL, (char *)"rad", (char *)"Generator phase", NULL, SDDS_DOUBLE, 0) < 0 ||
              SDDS_DefineColumn(rfmode->SDDSrec, (char *)"VCavity", NULL, (char *)"V", (char *)"Net cavity voltage (if generator active)", NULL, SDDS_DOUBLE, 0) < 0 ||
              SDDS_DefineColumn(rfmode->SDDSrec, (char *)"PhaseCavity", NULL, (char *)"rad", (char *)"Phase of net cavity voltage (if generator active)", NULL, SDDS_DOUBLE, 0) < 0)) ||
            !SDDS_WriteLayout(rfmode->SDDSrec)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"problem setting up RFMODE record file");
        }
      free(filename);
    }
    if (rfmode->feedbackRecordFile && strlen(rfmode->feedbackRecordFile)==0)
      rfmode->feedbackRecordFile = NULL;
    if (rfmode->driveFrequency > 0 && rfmode->feedbackRecordFile) {
      char *filename;
      filename = compose_filename(rfmode->feedbackRecordFile, run->rootname);
#if (USE_MPI)
      if (myid == 1)
#endif
        if (!(rfmode->SDDSfbrec = (SDDS_DATASET *)tmalloc(sizeof(*(rfmode->SDDSfbrec)))) ||
            !SDDS_InitializeOutputElegant(rfmode->SDDSfbrec, SDDS_BINARY, 1, NULL, NULL, filename) ||
            SDDS_DefineParameter(rfmode->SDDSfbrec, (char *)"SVNVersion", NULL, NULL, (char *)"SVN version number", NULL, SDDS_STRING, (char *)SVN_VERSION) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"Pass", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"t", NULL, (char *)"s", (char *)"Time", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"fResonance", NULL, (char *)"Hz", (char *)"Cavity resonance frequency", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"fDrive", NULL, (char *)"Hz", (char *)"Rf drive frequency", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"VbReal", NULL, (char *)"V", (char *)"Real part of beam-induced voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"VbImag", NULL, (char *)"V", (char *)"Imaginary part of beam-induced voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"VgReal", NULL, (char *)"V", (char *)"Real part of generator-driven voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"VgImag", NULL, (char *)"V", (char *)"Imaginary part of generator-driven voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"VCavity", NULL, (char *)"V", (char *)"Magnitude of total cavity voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"PhaseCavity", NULL, (char *)"rad", (char *)"Phase of cavity voltage", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"IgAmplitude", NULL, (char *)"A", (char *)"Generator current amplitude", NULL, SDDS_DOUBLE, 0) < 0 ||
            SDDS_DefineColumn(rfmode->SDDSfbrec, (char *)"IgPhase", NULL, (char *)"rad", (char *)"Generator phase", NULL, SDDS_DOUBLE, 0) < 0 ||
            !SDDS_WriteLayout(rfmode->SDDSfbrec)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"problem setting up RFMODE feedback record file");
        }
      free(filename);
    }
    rfmode->fileInitialized = 1;
  }
  for (i = 0; i < 8; i++) {
    rfmode->nNoise[i] = 0;
    if (rfmode->noiseData[i] && strlen(rfmode->noiseData[i]) != 0) {
      if (!getTableFromSearchPath(&data, rfmode->noiseData[i], 1, 0)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        bombElegantVA((char *)"Problem reading RFMODE noise file %s\n", rfmode->noiseData[i]);
      }
      if (data.n_data <= 1)
        bombElegantVA((char *)"RFMODE noise file %s has too little data\n", rfmode->noiseData[i]);
      rfmode->tNoise[i] = data.c1;
      rfmode->fNoise[i] = data.c2;
      rfmode->nNoise[i] = data.n_data;
      tfree(data.xlab);
      tfree(data.ylab);
      tfree(data.title);
      tfree(data.topline);
      data.xlab = data.ylab = data.title = data.topline = NULL;
      data.c1 = data.c2 = NULL;
    }
  }
  if (rfmode->preload && (rfmode->charge || rfmode->preloadCharge)) {
    double To, fref, psi, I, charge;
    if (rfmode->preloadCharge)
      charge = rfmode->preloadCharge;
    else
      charge = rfmode->charge;
    if (rfmode->fwaveform || rfmode->Qwaveform)
      printWarningForTracking((char *)"Preloading of RFMODE doesn't work properly with frequency or Q waveforms unless the initial values of the frequency and Q factors are 1.", NULL);
    To = total_length / (Po * c_mks / sqrt(sqr(Po) + 1));
    if (rfmode->preloadHarmonic)
      fref = rfmode->preloadHarmonic / To;
    else
      fref = ((int)(rfmode->freq * To + 0.5)) / To;
    psi = atan(2 * (rfmode->freq - fref) / fref * (rfmode->Q / (1 + rfmode->beta)));
    I = charge / To;
    rfmode->V = I * rfmode->RaInternal / (1 + rfmode->beta) * cos(psi) * rfmode->preload_factor;
    rfmode->last_phase = psi - PI;
    printf((char *)"RFMODE %s at z=%fm preloaded:  %eV at %fdeg \n",
           element_name, element_z, rfmode->V, rfmode->last_phase * 180 / PI);
    fflush(stdout);
    printf((char *)"To=%21.15es, I = %21.15eA, fref = %21.15eHz, df = %21.15eHz, psi = %21.15e\n", To, I, fref, rfmode->freq - fref, psi);
    printf((char *)"Q = %le, Ra = %le, beta = %le\n", rfmode->Q, rfmode->RaInternal, rfmode->beta);
    fflush(stdout);
    rfmode->last_t = element_z / (Po * c_mks / sqrt(sqr(Po) + 1));
  } else if (rfmode->initial_V) {
    rfmode->last_phase = rfmode->initial_phase;
    rfmode->V = rfmode->initial_V;
    rfmode->last_t = rfmode->initial_t;
  } else
    rfmode->last_t = element_z / (Po * c_mks / sqrt(sqr(Po) + 1));
  /* last_t above is a register-reduced estimate at the start of the current step;
     record the matching clock offset so the first advance sees no macro gap. */
  rfmode->last_clockOffset = trackingClockOffset();
  rfmode->last_clockPhase = 0; /* omega not known here; first advance sees V=0 so any residual is harmless, then it is snapshotted with the real omega */
  rfmode->tFreq = rfmode->fFreq = NULL;
  rfmode->nFreq = 0;
  if (rfmode->fwaveform) {
    if (!getTableFromSearchPath(&data, rfmode->fwaveform, 1, 0))
      bombElegant((char *)"unable to read frequency waveform for RFMODE", NULL);
    if (data.n_data <= 1)
      bombElegant((char *)"RFMODE frequency table contains less than 2 points", NULL);
    rfmode->tFreq = data.c1;
    rfmode->fFreq = data.c2;
    rfmode->nFreq = data.n_data;
    for (i = 1; i < rfmode->nFreq; i++)
      if (rfmode->tFreq[i - 1] > rfmode->tFreq[i])
        bombElegant((char *)"time values are not monotonically increasing in RFMODE frequency waveform", NULL);
    tfree(data.xlab);
    tfree(data.ylab);
    tfree(data.title);
    tfree(data.topline);
    data.xlab = data.ylab = data.title = data.topline = NULL;
    data.c1 = data.c2 = NULL;
  }
  rfmode->tQ = rfmode->fQ = NULL;
  rfmode->nQ = 0;
  if (rfmode->Qwaveform) {
    if (!getTableFromSearchPath(&data, rfmode->Qwaveform, 1, 0))
      bombElegant((char *)"unable to read Q waveform for RFMODE", NULL);
    if (data.n_data <= 1)
      bombElegant((char *)"RFMODE Q table contains less than 2 points", NULL);
    rfmode->tQ = data.c1;
    rfmode->fQ = data.c2;
    rfmode->nQ = data.n_data;
    for (i = 1; i < rfmode->nQ; i++)
      if (rfmode->tQ[i - 1] > rfmode->tQ[i])
        bombElegant((char *)"time values are not monotonically increasing in RFMODE Q waveform", NULL);
    tfree(data.xlab);
    tfree(data.ylab);
    tfree(data.title);
    tfree(data.topline);
    data.xlab = data.ylab = data.title = data.topline = NULL;
    data.c1 = data.c2 = NULL;
  }

  if (rfmode->V)
    printf("Beam loading initialization: V = %le, phase = %le, V*cos(phase) = %le, V*sin(phase) = %le, last_t = %21.15e\n",
           rfmode->V, rfmode->last_phase, rfmode->V * cos(rfmode->last_phase), rfmode->V * sin(rfmode->last_phase), rfmode->last_t);

  if (rfmode->driveFrequency > 0) {
    /* Set up the generator and related data. */
    /* See T. Berenc, RF-TN-2015-001 */
    MATRIX *C;
    double deltaOmega, sigma, QL, k;
    //double decrement, Tsample;

    rfmode->lambdaA = 2 * (rfmode->beta + 1) / rfmode->RaInternal;
    m_alloc(&rfmode->A, 2, 2);
    m_alloc(&rfmode->B, 2, 2);
    m_alloc(&rfmode->At, 2, 2);
    m_alloc(&rfmode->Bt, 2, 2);
    m_alloc(&rfmode->Mt1, 2, 1);
    m_alloc(&rfmode->Mt2, 2, 1);
    m_alloc(&rfmode->Mt3, 2, 1);
    m_alloc(&rfmode->Ig0, 2, 1);
    m_alloc(&rfmode->Viq, 2, 1);
    m_alloc(&rfmode->Iiq, 2, 1);

    fillBerencABMatrices(rfmode->A, rfmode->B, rfmode, rfmode->updateInterval / rfmode->driveFrequency);

    /* convert from V*sin(phi) to V*cos(phi) convention and account for fact that the
     * feedback is performed in the nominally empty part of each bucket (i.e., 180 degrees before 
     * the nominal beam phase)
     */
    rfmode->phaseg = PI / 180 * rfmode->phaseSetpoint - PI / 2 - PI;
    rfmode->Viq->a[0][0] = rfmode->voltageSetpoint * cos(rfmode->phaseg) + rfmode->V * cos(rfmode->last_phase);
    rfmode->Viq->a[1][0] = rfmode->voltageSetpoint * sin(rfmode->phaseg) + rfmode->V * sin(rfmode->last_phase);
    /* We'll need these later in I/Q mode feedback */
    rfmode->V0 = rfmode->V;
    rfmode->last_phase0 = rfmode->last_phase;

    /* Compute nominal generator current */
    QL = rfmode->Q / (1 + rfmode->beta);
    deltaOmega = PIx2 * (rfmode->freq - rfmode->driveFrequency);
    sigma = PIx2 * rfmode->freq / (2 * QL);
    //Tsample = rfmode->updateInterval/rfmode->driveFrequency;
    //decrement = exp(-sigma*Tsample);
    k = PIx2 * rfmode->freq / 4 * (rfmode->RaInternal / rfmode->Q);
    m_alloc(&C, 2, 2);
    C->a[0][0] = C->a[1][1] = sigma / k;
    C->a[1][0] = -(C->a[0][1] = deltaOmega / k);
    m_mult(rfmode->Ig0, C, rfmode->Viq);
    m_copy(rfmode->Iiq, rfmode->Ig0);
    m_free(&C);

    rfmode->fbRunning = 0;
    rfmode->fbNextTickTime = 0;
    rfmode->fbLastTickTime = 0;

    /* Read FB filters */
    rfmode->nAmplitudeFilters = rfmode->nPhaseFilters = rfmode->nIFilters = rfmode->nQFilters = 0;
    if (rfmode->amplitudeFilterFile && !(rfmode->nAmplitudeFilters = readIIRFilter(rfmode->amplitudeFilter, 4, rfmode->amplitudeFilterFile)))
      bombElegantVA((char *)"Error: problem reading amplitude filter file for RFMODE %s\n", element_name);
    if (rfmode->phaseFilterFile && !(rfmode->nPhaseFilters = readIIRFilter(rfmode->phaseFilter, 4, rfmode->phaseFilterFile)))
      bombElegantVA((char *)"Error: problem reading phase filter file for RFMODE %s\n", element_name);
    if (rfmode->IFilterFile && !(rfmode->nIFilters = readIIRFilter(rfmode->IFilter, 4, rfmode->IFilterFile)))
      bombElegantVA((char *)"Error: problem reading I filter file for RFMODE %s\n", element_name);
    if (rfmode->QFilterFile && !(rfmode->nQFilters = readIIRFilter(rfmode->QFilter, 4, rfmode->QFilterFile)))
      bombElegantVA((char *)"Error: problem reading Q filter file for RFMODE %s\n", element_name);
    if (rfmode->nAmplitudeFilters && !rfmode->nPhaseFilters)
      bombElegantVA((char *)"Error: amplitude filters specified without phase filters for RFMODE %s\n", element_name);
    if (!rfmode->nAmplitudeFilters && rfmode->nPhaseFilters)
      bombElegantVA((char *)"Error: phase filters specified without amplitude filters for RFMODE %s\n", element_name);
    if (rfmode->nIFilters && !rfmode->nQFilters)
      bombElegantVA((char *)"Error: I filters specified without Q filters for RFMODE %s\n", element_name);
    if (!rfmode->nIFilters && rfmode->nQFilters)
      bombElegantVA((char *)"Error: Q filters specified without I filters for RFMODE %s\n", element_name);
    if ((rfmode->nAmplitudeFilters + rfmode->nPhaseFilters) != 0 && (rfmode->nIFilters + rfmode->nQFilters) != 0)
      bombElegantVA((char *)"Error: can't specify both amplitude/phase and I/Q feedback! RFMODE %s\n", element_name);
  }
}

void runBinlessRfMode(
  double **part, int64_t np, RFMODE *rfmode, double Po,
  char *element_name, double element_z, long pass, long n_passes,
  CHARGE *charge) {
  static TIMEDATA *tData;
  static long max_np = 0;
  int64_t ip, ip0;
  long ib;
  double P;
  double Vb, V, omega, phase, t, k, damping_factor, tau;
  double V_sum, Vr_sum, phase_sum;
  double Vc, Vcr, Q_sum, dgamma, Vb_sum;
  long n_summed, max_hist, n_occupied;
  double Qrp, VbImagFactor, Q;
  double tmean;
  double clockNow;
  double clockPhaseNow = 0;
  long deltaPass;
#if USE_MPI
  int64_t np_total;

  if (isSlave)
    MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, workers);
#endif

  if (charge) {
    rfmode->mp_charge = charge->macroParticleCharge;
  } else if (pass == 0) {
    rfmode->mp_charge = 0;
    if (rfmode->charge < 0)
      bombElegant((char *)"RFMODE charge parameter should be non-negative. Use change_particle to set particle charge state.", NULL);
#if !USE_MPI
    if (np)
      rfmode->mp_charge = rfmode->charge / np;
#else
    if (distributedBeam) {
      if (np_total)
        rfmode->mp_charge = rfmode->charge / np_total;
    } else {
      if (np)
        rfmode->mp_charge = rfmode->charge / np;
    }
#endif
  }

  if (pass % rfmode->pass_interval)
    return;

  if (rfmode->freq < 1e3 && rfmode->freq)
    printWarningForTracking((char *)"RFMODE frequency is less than 1kHz.",
                            (char *)"This may be an error. Consult manual for units.");

  if (rfmode->mp_charge == 0) {
    return;
  }
  if (rfmode->detuned_until_pass > pass) {
    return;
  }

  if (!rfmode->initialized)
    bombElegant((char *)"track_through_rfmode called with uninitialized element", NULL);

  if (np > max_np)
    tData = (TIMEDATA *)trealloc(tData, sizeof(*tData) * (max_np = np));

  tmean = 0;
  for (ip = 0; ip < np; ip++) {
    P = Po * (part[ip][5] + 1);
    tData[ip].t = part[ip][4] * sqrt(sqr(P) + 1) / (c_mks * P);
    tData[ip].ip = ip;
    if (isSlave) {
      tmean += tData[ip].t;
    }
  }
  qsort(tData, np, sizeof(*tData), compTimeData);

#if USE_MPI
  if (distributedBeam) {
    if (isSlave) {
      double t_total;

      MPI_Allreduce(&tmean, &t_total, 1, MPI_DOUBLE, MPI_SUM, workers);
      tmean = t_total;
    }
    tmean /= np_total;
  } else
    tmean /= np;
#else
  tmean /= np;
#endif

  V_sum = Vr_sum = phase_sum = Vc = Vcr = Q_sum = Vb_sum = 0;
  n_summed = max_hist = n_occupied = 0;

  /* find frequency and Q at this time */
  omega = PIx2 * rfmode->freq;
  if (rfmode->nFreq) {
    double omegaFactor;
    ib = find_nearby_array_entry(rfmode->tFreq, rfmode->nFreq, tmean);
    omega *= (omegaFactor = linear_interpolation(rfmode->fFreq, rfmode->tFreq, rfmode->nFreq, tmean, ib));
    /* keeps stored energy constant for constant R/Q */
    rfmode->V *= sqrt(omegaFactor);
  }
  Q = rfmode->Q / (1 + rfmode->beta);
  if (rfmode->nQ) {
    ib = find_nearby_array_entry(rfmode->tQ, rfmode->nQ, tmean);
    Q *= linear_interpolation(rfmode->fQ, rfmode->tQ, rfmode->nQ, tmean, ib);
  }
  if (Q < 0.5) {
    printf((char *)"The effective Q<=0.5 for RFMODE.  Use the ZLONGIT element.\n");
    fflush(stdout);
    exitElegant(1);
  }
  tau = 2 * Q / omega;
  k = omega / 4 * (rfmode->RaInternal) / rfmode->Q;
  if ((deltaPass = (pass - rfmode->detuned_until_pass)) <= (rfmode->rampPasses - 1))
    k *= (deltaPass + 1.0) / rfmode->rampPasses;

  /* These adjustments per Zotter and Kheifets, 3.2.4 */
  Qrp = sqrt(Q * Q - 0.25);
  VbImagFactor = 1 / (2 * Qrp);
  omega *= Qrp / Q;

  /* trackingClockOffset() is constant within a step; the macro inter-step gap
     it represents must enter the cavity damping (true elapsed time) but never
     the phase or the coord[4] write-back, which stay register-reduced. */
  clockNow = trackingClockOffset();
  clockPhaseNow = trackingClockPhase(omega);

  if (rfmode->single_pass) {
    rfmode->V = rfmode->last_phase = 0;
    rfmode->last_t = tData[0].t;
    rfmode->last_clockOffset = clockNow;
    rfmode->last_clockPhase = clockPhaseNow;
  }

  for (ip0 = 0; ip0 < np; ip0++) {
    ip = tData[ip0].ip;
    t = tData[ip0].t;

    /* advance cavity to this time */
    phase = rfmode->last_phase + omega * (t - rfmode->last_t) + (clockPhaseNow - rfmode->last_clockPhase);
    damping_factor = exp(-((t - rfmode->last_t) + (clockNow - rfmode->last_clockOffset)) / tau);
    rfmode->last_t = t;
    rfmode->last_clockOffset = clockNow;
    rfmode->last_clockPhase = clockPhaseNow;
    rfmode->last_phase = phase;
    V = rfmode->V * damping_factor;
    rfmode->Vr = V * cos(phase);
    rfmode->Vi = V * sin(phase);

    /* compute beam-induced voltage from this particle */
    Vb = 2 * k * rfmode->mp_charge * particleRelSign * rfmode->pass_interval;
    /* compute voltage seen by this particle */
    V = rfmode->Vr - Vb / 2;

    /* add beam-induced voltage to cavity voltage */
    rfmode->Vr -= Vb;
    rfmode->Vi -= Vb * VbImagFactor;
    rfmode->last_phase = atan2(rfmode->Vi, rfmode->Vr);
    rfmode->V = sqrt(sqr(rfmode->Vr) + sqr(rfmode->Vi));

    V_sum += rfmode->V;
    Vb_sum += Vb;
    Vr_sum += rfmode->Vr;
    phase_sum += rfmode->last_phase;
    Q_sum += rfmode->mp_charge * particleRelSign;
    n_summed += 1;

    if (rfmode->rigid_until_pass <= pass) {
      /* change the particle energy */
      dgamma = rfmode->n_cavities * V / (1e6 * particleMassMV * particleRelSign);
      add_to_particle_energy(part[ip], tData[ip0].t, Po, dgamma);
    }
  }

  if (rfmode->record) {
    if (pass == 0 && !SDDS_StartPage(rfmode->SDDSrec, rfmode->flush_interval)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb((char *)"problem starting page for RFMODE record file");
    }
    if ((pass % rfmode->sample_interval) == 0) {
      if (!SDDS_SetRowValues(rfmode->SDDSrec, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             (pass / rfmode->sample_interval),
                             (char *)"Pass", pass, (char *)"NumberOccupied", n_occupied,
                             (char *)"FractionBinned", 1.0,
                             (char *)"VPostBeam", rfmode->V, (char *)"PhasePostBeam", rfmode->last_phase,
                             (char *)"tPostBeam", rfmode->last_t,
                             (char *)"V", n_summed ? V_sum / n_summed : 0.0,
                             (char *)"VReal", n_summed ? Vr_sum / n_summed : 0.0,
                             (char *)"Phase", n_summed ? phase_sum / n_summed : 0.0,
                             NULL) ||
          !SDDS_UpdatePage(rfmode->SDDSrec, 0)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"problem setting up data for RFMODE record file");
      }
    }
    if (pass == n_passes - 1) {
      free(rfmode->SDDSrec);
      rfmode->SDDSrec = NULL;
      if (!SDDS_Terminate(rfmode->SDDSrec)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"problem writing data for RFMODE record file");
      }
    }
  }

#if defined(MINIMIZE_MEMORY)
  free(tData);
  tData = NULL;
  max_np = 0;
#endif
}

#if USE_MPI
typedef struct {
  long index; /* Records the real index in the whole histogram */
  long sum;
} SUB_HISTOGRAM;

void histogram_sums(long nonEmptyBins, long firstBin, long *lastBin, long *his) {
  bombElegant("error: retired function histogram_sums() was called. Seek professional help!", NULL);
#  if 0
  static long *nonEmptyArray = NULL;
  long lastBin_global, firstBin_global;
  long nonEmptyBins_total = 0, offset = 0;
  long i, j, map_j, ib;
  static SUB_HISTOGRAM *subHis; /* a compressed histogram with non-zero bins only */
  static long max_nonEmptyBins_total = 0;
  MPI_Status status;

#    ifdef DEBUG
  printf("histogram_sums 1\n"); fflush(stdout);
#    endif

  MPI_Reduce(lastBin, &lastBin_global, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&firstBin, &firstBin_global, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  if (isMaster) {
    *lastBin = lastBin_global;
    firstBin = firstBin_global; 
  }
#    ifdef DEBUG
  printf("histogram_sums 2\n"); fflush(stdout);
#    endif

  if (!nonEmptyArray)
    nonEmptyArray =(long*) tmalloc(sizeof(*nonEmptyArray)*n_processors);

#    ifdef DEBUG
  printf("histogram_sums 3\n"); fflush(stdout);
#    endif

  MPI_Gather(&nonEmptyBins,1,MPI_LONG,nonEmptyArray,1,MPI_LONG,0,MPI_COMM_WORLD);
  if (isMaster){
    for (i=1; i<n_processors; i++) 
      nonEmptyBins_total += nonEmptyArray[i];
  }
  MPI_Bcast(&nonEmptyBins_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#    ifdef DEBUG
  printf("histogram_sums 4: nonEmptyBins_total = %ld\n", nonEmptyBins_total); fflush(stdout);
#    endif
 
  if (nonEmptyBins_total>max_nonEmptyBins_total) {
    if (!(subHis = (SUB_HISTOGRAM*) trealloc(subHis, sizeof(*subHis)*(max_nonEmptyBins_total=nonEmptyBins_total))))
      bomb ("Memory allocation failure in track_through_ftrfmod", NULL);
  }

#    ifdef DEBUG
  printf("histogram_sums 5\n"); fflush(stdout);
#    endif
  if (isSlave) {
#    ifdef DEBUG
    printf("histogram_sums 6, firstBin=%ld, lastBin=%ld\n", firstBin, *lastBin); fflush(stdout);
#    endif
    for (i=0,ib=firstBin; ib<=*lastBin; ib++) {
      if (his[ib]){        
	subHis[i].index = ib;
	subHis[i].sum = his[ib]; 
	i++;
      }
    }
#    ifdef DEBUG
    printf("histogram_sums 6.1\n"); fflush(stdout);
#    endif
    MPI_Send(subHis, nonEmptyBins*sizeof(*subHis), MPI_BYTE, 0, 108, MPI_COMM_WORLD);
#    ifdef DEBUG
    printf("histogram_sums 6.2\n"); fflush(stdout);
#    endif
  }
  else {
#    ifdef DEBUG
    printf("histogram_sums 7\n"); fflush(stdout);
#    endif
    for (i=1; i<n_processors; i++) {
      if (i>1)
	offset += nonEmptyArray[i-1];
      MPI_Recv (&subHis[offset], nonEmptyArray[i]*sizeof(*subHis), MPI_BYTE, i, 108, MPI_COMM_WORLD, &status); 
      for (j=offset; j<nonEmptyArray[i]+offset; j++) {
#    define current subHis[j]
	map_j = current.index;
	his[map_j] += current.sum; 
      }
    }
#    ifdef DEBUG
    printf("histogram_sums 7.1\n"); fflush(stdout);
#    endif
      
    for (i=0, ib=firstBin; ib<=*lastBin; ib++) { 
      if (his[ib]) {
	subHis[i].index = ib;
	subHis[i].sum = his[ib];
	i++;  
      }
    }
#    ifdef DEBUG
    printf("histogram_sums 7.2\n"); fflush(stdout);
#    endif
    /* If there are overlapped bins between different processors, the number should be less than the original
       nonEmptyBins_total */
    nonEmptyBins_total = i;  
  }
#    ifdef DEBUG
  printf("histogram_sums 8\n"); fflush(stdout);
#    endif
  MPI_Bcast (&nonEmptyBins_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast (subHis, nonEmptyBins_total*sizeof(*subHis), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast (lastBin, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  if (isSlave) {
    for (i=0; i<nonEmptyBins_total; i++) {      
      his[subHis[i].index] = subHis[i].sum; 
    }
  }
#    ifdef DEBUG
  printf("histogram_sums 9\n"); fflush(stdout);
#    endif
#  endif
}
#endif

void fillBerencABMatrices(MATRIX *A, MATRIX *B, RFMODE *rfmode, double dt) {
  double deltaOmega, decrement, sigma, QL, Tsample, k, b1, alpha, beta, sin1, cos1;

  QL = rfmode->Q / (1 + rfmode->beta);
  deltaOmega = PIx2 * (rfmode->freq - rfmode->driveFrequency);
  sigma = PIx2 * rfmode->freq / (2 * QL);
  Tsample = dt;
  decrement = exp(-sigma * Tsample);

  sin1 = sin(deltaOmega * Tsample);
  cos1 = cos(deltaOmega * Tsample);
  A->a[0][0] = A->a[1][1] = decrement * cos1;
  A->a[0][1] = -(A->a[1][0] = decrement * sin1);

  k = PIx2 * rfmode->freq / 4 * (rfmode->RaInternal / rfmode->Q);
  b1 = k / (sqr(sigma) + sqr(deltaOmega));
  alpha = deltaOmega * decrement * sin1 - sigma * decrement * cos1 + sigma;
  beta = sigma * decrement * sin1 + deltaOmega * decrement * cos1 - deltaOmega;
  B->a[0][0] = B->a[1][1] = b1 * alpha;
  B->a[1][0] = -(B->a[0][1] = b1 * beta);
}
