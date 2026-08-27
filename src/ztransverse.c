/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: zlongit.c
 * contents: track_through_zlongit()
 *
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "track.h"
#include "table.h"
#include "fftpackC.h"

#if !USE_MPI
#  define WAKE_COLUMNS 5
static SDDS_DEFINITION wake_column[WAKE_COLUMNS] = {
  {"Deltat", "&column name=Deltat, symbol=\"$gD$rt\", units=s, type=double, description=\"Time after head of bunch\" &end"},
  {"Ix", "&column name=Ix, symbol=\"<I$bx$n>\", units=C*m$ad$n/s, type=double, description=\"Transverse horizontal moment\" &end"},
  {"Wx", "&column name=Wx, symbol=\"W$bx$n\", units=V/m$ap$n, type=double, description=\"Transverse horizontal wake\" &end"},
  {"Iy", "&column name=Iy, symbol=\"<I$by$n>\", units=C*m$ad$n/s, type=double, description=\"Transverse vertical moment\" &end"},
  {"Wy", "&column name=Wy, symbol=\"W$by$n\", units=V/m$ap$n, type=double, description=\"Transverse vertical wake\" &end"},
};

#  define WAKE_PARAMETERS 6
#  define BB_WAKE_PARAMETERS 6
#  define NBB_WAKE_PARAMETERS 3
static SDDS_DEFINITION wake_parameter[WAKE_PARAMETERS] = {
  {"Pass", "&parameter name=Pass, type=long &end"},
  {"Bunch", "&parameter name=Bunch, type=long &end"},
  {"q", "&parameter name=q, units=C, type=double, description=\"Total charge\" &end"},
  {"Rs", "&parameter name=Rs, symbol=\"R$bs$n\", units=\"$gW$r\", type=double, description=\"Broad-band impedance\" &end"},
  {"fo", "&parameter name=fo, symbol=\"f$bo$n\", units=Hz, type=double, description=\"Frequency of BB resonator\" &end"},
  {"Deltaf", "&parameter name=Deltaf, symbol=\"$gD$rf\", units=Hz, type=double, description=\"Frequency sampling interval\" &end"},
};
#endif

void set_up_ztransverse(ZTRANSVERSE *ztransverse, RUN *run, long pass, long particles, CHARGE *charge,
                        double timeSpan);
double *getTransverseImpedance(SDDS_DATASET *SDDSin, char *ZName);

void track_through_ztransverse(double **part0, int64_t np0, ZTRANSVERSE *ztransverse, double Po,
                               RUN *run, long i_pass, CHARGE *charge) {
  double *posItime[2] = {NULL, NULL}; /* array for particle density times x, y*/
  double *posIfreq = NULL;            /* array for FFT of particle density times x or y*/
  double *Vtime = NULL;               /* array for voltage acting on each bin */
  long max_n_bins = 0;
  long *pbin = NULL;    /* array to record which bin each particle is in */
  double *time0 = NULL; /* array to record arrival time of each particle */
  double *time = NULL;  /* array to record arrival time of each particle */
  double *pz = NULL;
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  int64_t max_np = 0;
  double *Vfreq = NULL, *Z = NULL;
  long nBuckets, iBucket;
  int64_t np;
#if USE_MPI
  long offset, length;
  double tmin_part, tmax_part;
  double *buffer;
#endif
  long ib, nb, nfreq, iReal, iImag, plane, first;
  /* long n_binned; */
  double factor, tmin, tmax, dt, userFactor[2], rampFactor = 1;
  //double tmean;
  //static long not_first_call = -1;
  int64_t ip;
  long i_pass0;
#if defined(DEBUG)
  FILE *fp;
#endif

  i_pass0 = i_pass;
  if ((i_pass -= ztransverse->startOnPass) < 0 || ztransverse->factor == 0)
    return;

#if defined(DEBUG) && USE_MPI
  printf("ZTRANSVERSE, myid = %d\n", myid);
  fflush(stdout);
#endif

  if (i_pass >= (ztransverse->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / ztransverse->rampPasses;

  //not_first_call += 1;

  if (isSlave || !distributedBeam) {
    index_bunch_assignments(part0, np0, (charge && ztransverse->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif

#ifdef DEBUG
    printf("%ld buckets\n", nBuckets);
    fflush(stdout);
    if (nBuckets > 1) {
      fflush(stdout);
      for (iBucket = 0; iBucket < nBuckets; iBucket++) {
        printf("bucket %ld: %" PRId64 " particles\n", iBucket, npBucket[iBucket]);
        fflush(stdout);
      }
    }
#endif

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (ztransverse->bunchedBeamMode && 
          ((ztransverse->startBunch>=0 && iBucket<ztransverse->startBunch) ||
           (ztransverse->endBunch>=0 && iBucket>ztransverse->endBunch)))
        continue;
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
        pz = trealloc(pz, sizeof(*pz) * np);
      } else {
        if (npBucket)
          np = npBucket[iBucket];
        else
          np = 0;
        if (np && (!ibParticle || !ipBucket || !time0)) {
#if USE_MPI
          mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
          return;
#else
          printf("Problem in index_bunch_assignments. Seek professional help.\n");
          exitElegant(1);
#endif
        }
#if !USE_MPI
        if (np == 0)
          continue;
#endif
#ifdef DEBUG
        printf("ZTRANSVERSE: copying data to work array, iBucket=%ld, np=%" PRId64 "\n", iBucket, np);
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
        for (ip = 0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
      }

      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#if USE_MPI
      find_global_min_max(&tmin, &tmax, np, workers);
      tmin_part = tmin;
      tmax_part = tmax;
      //tmean = computeAverage_p(time, np, workers);
#else
      //compute_average(&tmean, time, np);
#endif
      /* use np0 here since we need to compute the macroparticle charge */
      set_up_ztransverse(ztransverse, run, i_pass, np0, charge, tmax - tmin);

      nb = ztransverse->n_bins;
      dt = ztransverse->bin_size;
      tmin -= dt;
      tmax += dt;
#if USE_MPI && MPI_DEBUG
      printf("tmin_part = %21.15e  tmax_part = %21.15e\n", tmin_part, tmax_part);
      printf("tmin      = %21.15e  tmax      = %21.15e  dt = %21.15e\n", tmin, tmax, dt);
      fflush(stdout);
#endif
      if ((tmax - tmin) * 2 > nb * dt) {
        TRACKING_CONTEXT tcontext;
        getTrackingContext(&tcontext);
#if USE_MPI && defined(MPI_DEBUG)
        if (myid == 1)
          dup2(fdStdout, fileno(stdout));
#endif
        fprintf(stderr, "%s %s: Time span of bunch %ld (%21.15le s) is more than half the total time span (%21.15le s).\n",
                entity_name[tcontext.elementType],
                tcontext.elementName,
                iBucket, tmax - tmin, nb * dt);
        fprintf(stderr, "If using broad-band impedance, you should increase the number of bins and rerun.\n");
        fprintf(stderr, "If using file-based impedance, you should increase the number of data points or decrease the frequency resolution.\n");
        if (!ztransverse->allowLongBeam) {
#if USE_MPI
#  if MPI_DEBUG
          for (ip = 0; ip < np; ip++)
            printf("particle %5" PRId64 ": t=%21.15e, delta=%21.15e\n", ip, time[ip], part[ip][5]);
          printf("Issuing MPI abort from ZLONGIT\n");
          fflush(stdout);
#  endif
          mpiAbort = MPI_ABORT_BUNCH_TOO_LONG_ZTRANSVERSE;
          return;
#else
          exitElegant(1);
#endif
        }
      }

      if (nb > max_n_bins) {
        posItime[0] = trealloc(posItime[0], 2 * sizeof(**posItime) * (max_n_bins = nb));
        posItime[1] = trealloc(posItime[1], 2 * sizeof(**posItime) * (max_n_bins = nb));
        posIfreq = trealloc(posIfreq, 2 * sizeof(*posIfreq) * (max_n_bins = nb));
        Vtime = trealloc(Vtime, 2 * sizeof(*Vtime) * (max_n_bins + 1));
      }

      for (ib = 0; ib < nb; ib++)
        posItime[0][2 * ib] = posItime[0][2 * ib + 1] =
          posItime[1][2 * ib] = posItime[1][2 * ib + 1] = 0;

      /* make arrays of I(t)*x and I(t)*y */
      /*n_binned = binTransverseTimeDistribution(posItime, pz, pbin, tmin, dt, nb, 
                                               time, part, Po, np,
                                               ztransverse->dx, ztransverse->dy,
                                               ztransverse->xDriveExponent, ztransverse->yDriveExponent);*/
      binTransverseTimeDistribution(posItime, pz, pbin, tmin, dt, nb,
                                    time, part, Po, np,
                                    ztransverse->dx, ztransverse->dy,
                                    ztransverse->xDriveExponent, ztransverse->yDriveExponent);
      userFactor[0] = ztransverse->factor * ztransverse->xfactor * rampFactor;
      userFactor[1] = ztransverse->factor * ztransverse->yfactor * rampFactor;

      first = 1;
      for (plane = 0; plane < 2; plane++) {
#if USE_MPI
        if (isSlave) {
          offset = ((long)((tmin_part - tmin) / dt) - 1 ? (long)((tmin_part - tmin) / dt) - 1 : 0);
          length = ((long)((tmax_part - tmin_part) / dt) + 2 < nb ? (long)((tmax_part - tmin_part) / dt) + 2 : nb);
          if (offset < 0) {
            length -= offset;
            offset = 0;
          }
          if (offset >= nb) {
            offset = 0;
            length = nb;
          }
          if ((offset + length) > nb)
            length = nb - offset;
#  if MPI_DEBUG
          printf("plane = %ld, offset = %ld, length=%ld, nb=%ld\n", plane, offset, length, nb);
          fflush(stdout);
#  endif
          buffer = malloc(sizeof(double) * length);
          MPI_Allreduce(&posItime[plane][offset], buffer, length, MPI_DOUBLE, MPI_SUM, workers);
          memcpy(&posItime[plane][offset], buffer, sizeof(double) * length);
          free(buffer);
#  if MPI_DEBUG
          printf("posItime buffer shared\n");
          fflush(stdout);
#  endif
        }

#endif
        if (userFactor[plane] == 0) {
          for (ib = 0; ib < nb; ib++)
            Vtime[ib] = 0;
        } else {
          if (ztransverse->smoothing)
            SavitzyGolaySmooth(posItime[plane], nb, ztransverse->SGOrder,
                               ztransverse->SGHalfWidth, ztransverse->SGHalfWidth, 0);

#if MPI_DEBUG
          printf("Smoothing completed\n");
          fflush(stdout);
#endif

          /* Take the FFT of (x*I)(t) to get (x*I)(f) */
          memcpy(posIfreq, posItime[plane], 2 * nb * sizeof(*posIfreq));
          realFFT(posIfreq, nb, 0);

#if MPI_DEBUG
          printf("FFT completed\n");
          fflush(stdout);
#endif

          /* Compute V(f) = i*Z(f)*(x*I)(f), putting in a factor 
           * to normalize the current waveform
           */
          Vfreq = Vtime;
          factor = ztransverse->macroParticleCharge * particleRelSign / dt * userFactor[plane];
          Z = ztransverse->Z[plane];
          Vfreq[0] = posIfreq[0] * Z[0] * factor;
          nfreq = nb / 2 + 1;
          if (nb % 2 == 0)
            /* Nyquist term */
            Vfreq[nb - 1] = posIfreq[nb - 1] * Z[nb - 1] * factor;
          for (ib = 1; ib < nfreq - 1; ib++) {
            iImag = (iReal = 2 * ib - 1) + 1;
            /* The signs are chosen here to get agreement with TRFMODE.
               In particular, test particles following closely behind the 
               drive particle get defocused.
               */
            Vfreq[iReal] = (posIfreq[iReal] * Z[iReal] - posIfreq[iImag] * Z[iImag]) * factor;
            Vfreq[iImag] = (posIfreq[iReal] * Z[iImag] + posIfreq[iImag] * Z[iReal]) * factor;
          }
#if MPI_DEBUG
          printf("Product completed\n");
          fflush(stdout);
#endif

          /* Compute inverse FFT of V(f) to get V(t) */
          realFFT(Vfreq, nb, INVERSE_FFT);
          Vtime = Vfreq;

#if MPI_DEBUG
          printf("IFFT completed\n");
          fflush(stdout);
#endif

          /* change particle transverse momenta to reflect voltage in relevant bin */
          applyTransverseWakeKicks(part, time, pz, pbin, np,
                                   Po, plane,
                                   Vtime, nb, tmin, dt, ztransverse->interpolate,
                                   plane == 0 ? ztransverse->xProbeExponent : ztransverse->yProbeExponent);
#if MPI_DEBUG
          printf("Wake application completed\n");
          fflush(stdout);
#endif
        }

        if (ztransverse->SDDS_wake_initialized && ztransverse->wakes) {
          /* wake potential output */
          factor = ztransverse->macroParticleCharge * particleRelSign / dt;
          if ((ztransverse->wake_interval <= 0 || ((i_pass0 - ztransverse->wake_start) % ztransverse->wake_interval) == 0) &&
              i_pass0 >= ztransverse->wake_start && (ztransverse->wake_end<0 || i_pass0 <= ztransverse->wake_end)) {
            if (first && !SDDS_StartTable(ztransverse->SDDS_wake, nb)) {
              SDDS_SetError("Problem starting SDDS table for wake output (track_through_ztransverse)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            for (ib = 0; ib < nb; ib++) {
              if (!SDDS_SetRowValues(ztransverse->SDDS_wake, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                                     0, ib * dt,
                                     1 + plane * 2, posItime[plane][ib] * factor,
                                     2 + plane * 2, Vtime[ib], -1)) {
                SDDS_SetError("Problem setting rows of SDDS table for wake output (track_through_ztransverse)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
            }
            if (!SDDS_SetParameters(ztransverse->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                    "Pass", i_pass0, "q", ztransverse->macroParticleCharge * particleRelSign * np, NULL)) {
              SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_ztransverse)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            if (ztransverse->broad_band) {
              if (!SDDS_SetParameters(ztransverse->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                      "Rs", ztransverse->Rs, "fo", ztransverse->freq,
                                      "Deltaf", ztransverse->bin_size, NULL)) {
                SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_ztransverse)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
            }
            if (!first) {
              if (!SDDS_WriteTable(ztransverse->SDDS_wake)) {
                SDDS_SetError("Problem writing SDDS table for wake output (track_through_ztransverse)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
              if (!inhibitFileSync)
                SDDS_DoFSync(ztransverse->SDDS_wake);
            }
          }
        }
        first = 0;
#if MPI_DEBUG
        printf("plane = %ld completed completed\n", plane);
        fflush(stdout);
#endif
      }
#ifdef DEBUG
      printf("Done with both planes\n");
      fflush(stdout);
#endif

      if (nBuckets != 1) {
#ifdef DEBUG
        printf("ZTRANSVERSE: copying data back to main array\n");
        fflush(stdout);
#endif

        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);

#if USE_MPI
#  ifdef DEBUG
        printf("Preparing to wait on barrier at end of loop for bucket %ld\n", iBucket);
        fflush(stdout);
#  endif
#endif
#ifdef DEBUG
        printf("Done with bucket %ld\n", iBucket);
        fflush(stdout);
#endif
      }
#if USE_MPI
      MPI_Barrier(workers);
#endif
    }
  }

#ifdef DEBUG
  printf("Preparing to free memory\n");
  fflush(stdout);
#endif
  if (part && part != part0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
  if (pbin)
    free(pbin);
  if (pz)
    free(pz);
  if (posItime[0])
    free(posItime[0]);
  if (posItime[1])
    free(posItime[1]);
  if (Vtime)
    free(Vtime);
  if (posIfreq)
    free(posIfreq);

#if USE_MPI
  MPI_Barrier(workers);
#endif
#ifdef DEBUG
  printf("Done with ZTRANSVERSE\n");
  fflush(stdout);
#endif
}

void set_up_ztransverse(ZTRANSVERSE *ztransverse, RUN *run, long pass, long particles, CHARGE *charge,
                        double timeSpan) {
  long i, nfreq;
  double df;

  if (charge) {
    ztransverse->macroParticleCharge = charge->macroParticleCharge;
  } else if (pass == 0) {
    ztransverse->macroParticleCharge = 0;
    if (ztransverse->charge < 0)
      bombElegant("ZTRANSVERSE charge parameter should be non-negative. Use change_particle to set particle charge state.", NULL);
#if (!USE_MPI)
    if (particles)
      ztransverse->macroParticleCharge = ztransverse->charge / particles;
#else
    if (USE_MPI) {
      long particles_total;

      MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, workers);
      if (particles_total)
        ztransverse->macroParticleCharge = ztransverse->charge / particles_total;
    }
#endif
  }

  if (ztransverse->initialized)
    return;

  ztransverse->SDDS_wake_initialized = 0;

  if (ztransverse->broad_band) {
    /* Use impedance Z = -i*wr/w*Rs/(1 + i*Q(w/wr-wr/w))
       */
    double term;
    if (ztransverse->bin_size <= 0)
      bombElegant("bin_size must be positive for ZTRANSVERSE element", NULL);
    if (ztransverse->n_bins % 2 != 0)
      bombElegant("ZTRANSVERSE element must have n_bins divisible by 2", NULL);
    if (ztransverse->ZxReal || ztransverse->ZxImag ||
        ztransverse->ZyReal || ztransverse->ZyImag)
      bombElegant("can't specify both broad_band impedance and Z(f) files for ZTRANSVERSE element", NULL);

    optimizeBinSettingsForImpedance(timeSpan, ztransverse->freq, ztransverse->Q,
                                    &(ztransverse->bin_size), &(ztransverse->n_bins),
                                    ztransverse->max_n_bins);

    nfreq = ztransverse->n_bins / 2 + 1;
    ztransverse->Z[0] = tmalloc(sizeof(**(ztransverse->Z)) * ztransverse->n_bins);
    ztransverse->Z[1] = tmalloc(sizeof(**(ztransverse->Z)) * ztransverse->n_bins);
    /* df is the frequency spacing normalized to the resonant frequency */
    df = 1 / (ztransverse->n_bins * ztransverse->bin_size) / (ztransverse->freq);
    /* DC term of Z is pure real */
    ztransverse->Z[0][0] = 2 * ztransverse->Rs / ztransverse->Q;
    ztransverse->Z[1][0] = 2 * ztransverse->Rs / ztransverse->Q;
    for (i = 1; i < nfreq - 1; i++) {
      term = ztransverse->Q * (i * df - 1.0 / (i * df));
      /* imaginary part of Z */
      ztransverse->Z[0][2 * i] = ztransverse->Z[1][2 * i] =
        -ztransverse->Rs / (i * df) / (1 + term * term);
      /* real part of Z */
      ztransverse->Z[0][2 * i - 1] = ztransverse->Z[1][2 * i - 1] =
        term * ztransverse->Z[0][2 * i];
    }
    /* Nyquist term--real part of Z only */
    term = ztransverse->Q * (1.0 / (nfreq * df) - nfreq * df);
    ztransverse->Z[0][ztransverse->n_bins - 1] =
      ztransverse->Z[1][ztransverse->n_bins - 1] =
        -term * ztransverse->Rs / (nfreq * df) / (1 + term * term);
    df *= ztransverse->freq;
  } else {
    double *ZReal[2], *ZImag[2], *freqData;
    double df_spect;
    long n_spect;
    SDDS_DATASET SDDSin;
    if (!ztransverse->freqColumn || !ztransverse->inputFile)
      bombElegant("you must give an inputFile and freqColumn, or use a broad band model (ZTRANSVERSE)", NULL);
    if (!ztransverse->ZxReal && !ztransverse->ZxImag &&
        !ztransverse->ZyReal && !ztransverse->ZxImag)
      bombElegant("you must either give broad_band=1, or Z[xy]Real and/or Z[xy]Imag (ZTRANSVERSE)", NULL);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, ztransverse->inputFile) || !SDDS_ReadPage(&SDDSin)) {
      printf("unable to read file %s\n", ztransverse->inputFile);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if ((n_spect = SDDS_RowCount(&SDDSin)) < 4) {
      printf("too little data in %s\n", ztransverse->inputFile);
      fflush(stdout);
      exitElegant(1);
    }
    if (!power_of_2(n_spect - 1))
      bombElegant("number of spectrum points must be 2^n+1, n>1 (ZTRANSVERSE)", NULL);
    ZReal[0] = getTransverseImpedance(&SDDSin, ztransverse->ZxReal);
    ZImag[0] = getTransverseImpedance(&SDDSin, ztransverse->ZxImag);
    ZReal[1] = getTransverseImpedance(&SDDSin, ztransverse->ZyReal);
    ZImag[1] = getTransverseImpedance(&SDDSin, ztransverse->ZyImag);
    if (!(freqData = SDDS_GetColumnInDoubles(&SDDSin, ztransverse->freqColumn))) {
      printf("Unable to read column %s (ZTRANSVERSE)\n", ztransverse->freqColumn);
      fflush(stdout);
      exitElegant(1);
    }
    if (!checkPointSpacing(freqData, n_spect, 1e-6)) {
      printf("Frequency values are not equispaced (ZTRANSVERSE)\n");
      fflush(stdout);
      exitElegant(1);
    }
    if ((df_spect = (freqData[n_spect - 1] - freqData[0]) / (n_spect - 1)) <= 0) {
      printf("Zero or negative frequency spacing in %s (ZTRANSVERSE)\n",
             ztransverse->inputFile);
      fflush(stdout);
      exitElegant(1);
    }
    df = df_spect;
    nfreq = n_spect;
    ztransverse->n_bins = 2 * (n_spect - 1);
    ztransverse->bin_size = 1.0 / (ztransverse->n_bins * df_spect);
    if (!SDDS_Terminate(&SDDSin)) {
      printf("Error closing data set %s\n",
             ztransverse->inputFile);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (!(ztransverse->Z[0] =
            calloc(sizeof(*ztransverse->Z[0]), n_spect * 2)) ||
        !(ztransverse->Z[1] =
            calloc(sizeof(*ztransverse->Z[1]), n_spect * 2)))
      bombElegant("memory allocation failure (ZTRANSVERSE)", NULL);
    for (i = 0; i < n_spect; i++) {
      if (i == 0) {
        /* DC term needs a factor of two, because we assume that the impedance is created
         * by sddsfft'ing the impedance, which folds the frequency range. We compensate
         * for that (e.g., trwake2impedance) by dividing by two, but need to fix the
         * DC term. */
        ztransverse->Z[0][i] = 2 * ZReal[0][i];
        ztransverse->Z[1][i] = 2 * ZReal[1][i];
      } else if (i == n_spect - 1 && ztransverse->n_bins % 2 == 0) {
        /* Nyquist term */
        ztransverse->Z[0][2 * i - 1] = ZReal[0][i];
        ztransverse->Z[1][2 * i - 1] = ZReal[1][i];
      } else {
        /* real part of Z */
        ztransverse->Z[0][2 * i - 1] = ZReal[0][i];
        ztransverse->Z[1][2 * i - 1] = ZReal[1][i];
        /* imaginary part of Z */
        ztransverse->Z[0][2 * i] = ZImag[0][i];
        ztransverse->Z[1][2 * i] = ZImag[1][i];
      }
    }
    free(ZReal[0]);
    free(ZReal[1]);
    free(ZImag[0]);
    free(ZImag[1]);
  }

#if (!USE_MPI)
  /* Only the serial version will dump this part of output */
  if (ztransverse->wakes) {
    char *filename;
    filename = compose_filename(ztransverse->wakes, run->rootname);
    ztransverse->SDDS_wake = tmalloc(sizeof(*(ztransverse->SDDS_wake)));
    if (ztransverse->broad_band)
      SDDS_ElegantOutputSetup(ztransverse->SDDS_wake, filename, SDDS_BINARY,
                              1, "transverse wake",
                              run->runfile, run->lattice, wake_parameter, BB_WAKE_PARAMETERS,
                              wake_column, WAKE_COLUMNS, "set_up_ztransverse",
                              SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    else {
      SDDS_ElegantOutputSetup(ztransverse->SDDS_wake, filename, SDDS_BINARY,
                              1, "transverse wake",
                              run->runfile, run->lattice, wake_parameter, NBB_WAKE_PARAMETERS,
                              wake_column, WAKE_COLUMNS, "set_up_ztransverse",
                              SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    }
    free(filename);
    ztransverse->SDDS_wake_initialized = 1;
  }
#endif

  if (ztransverse->highFrequencyCutoff0 > 0) {
    applyLowPassFilterToImpedance(ztransverse->Z[0], nfreq,
                                  ztransverse->highFrequencyCutoff0,
                                  ztransverse->highFrequencyCutoff1);
    applyLowPassFilterToImpedance(ztransverse->Z[1], nfreq,
                                  ztransverse->highFrequencyCutoff0,
                                  ztransverse->highFrequencyCutoff1);
  }

#if 0
  if (!ztransverse->initialized) {
    static long index = 0;
    char buffer[1024];
    FILE *fp;
    sprintf(buffer, "ztransverse-%03d.sdds", index++);
    fp = fopen_e(buffer, "w", 0);
    fprintf(fp, "SDDS1\n&column name=f units=Hz type=double &end\n");
    fprintf(fp, "&column name=ZReal type=double &end\n");
    fprintf(fp, "&column name=ZImag type=double &end\n");
    fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
    for (i=0; i<nfreq; i++) 
      fprintf(fp, "%21.15e %21.15e %21.15e\n",
              i*df, 
              i==0?ztransverse->Z[0][i]:ztransverse->Z[0][2*i-1],
              i==0?                 0.0:ztransverse->Z[0][2*i]);

    fclose(fp);
  }
#endif

  ztransverse->initialized = 1;
}

double *getTransverseImpedance(SDDS_DATASET *SDDSin,
                               char *ZName) {
  long rows;
  double *Z;
  rows = SDDS_RowCount(SDDSin);
  if (!ZName || !strlen(ZName)) {
    Z = calloc(sizeof(*Z), rows);
    return Z;
  }
  if (!(Z = SDDS_GetColumnInDoubles(SDDSin, ZName))) {
    printf("Unable to read column %s (ZTRANSVERSE)\n",
           ZName);
    fflush(stdout);
    exitElegant(1);
  }
  return Z;
}

void optimizeBinSettingsForImpedance(double timeSpan, double freq, double Q,
                                     double *binSize, long *nBins, long maxBins) {
  long n_bins, maxBins2;
  double bin_size, factor;
  TRACKING_CONTEXT tcontext;

  n_bins = *nBins;
  bin_size = *binSize;
  getTrackingContext(&tcontext);

  if (maxBins <= 0)
    maxBins2 = pow(2, 20);
  else
    maxBins2 = pow(2, (long)(log(maxBins) / log(2) + 1));
  if (maxBins > 0 && maxBins != maxBins2)
    printf("Adjusted maximum number of bins for %s %s to %ld\n",
           entity_name[tcontext.elementType],
           tcontext.elementName, maxBins2);

  if (1 / (2 * freq * bin_size) < 10) {
    /* want maximum frequency in Z > 10*fResonance */
    printf("%s %s has excessive bin size for given resonance frequency\n",
           entity_name[tcontext.elementType],
           tcontext.elementName);
    bin_size = 1. / freq / 20;
    printf("  Bin size adjusted to %e\n", bin_size);
    fflush(stdout);
  }
  if (2 * timeSpan > bin_size * n_bins) {
    printf("%s %s has insufficient time span for initial bunch\n",
           entity_name[tcontext.elementType],
           tcontext.elementName);
    n_bins = pow(2,
                 (long)(log(2 * timeSpan * 1.05 / bin_size) / log(2) + 1));
    if (maxBins2 < n_bins) {
      fprintf(stderr, "  Maximum number of bins does not allow sufficient time span!\n");
#if USE_MPI
      mpiAbort = MPI_ABORT_BUNCH_TOO_LONG_ZTRANSVERSE;
      return;
#else
      exitElegant(1);
#endif
    }
    printf("  Number of bins adjusted to %ld\n",
           n_bins);
    fflush(stdout);
  }
  if (Q < 1)
    /* Ideally, want frequency resolution < fResonance/200 and < fResonanceWidth/200 */
    factor = 200 / (n_bins * bin_size * freq / Q);
  else
    factor = 200 / (n_bins * bin_size * freq);
  if (factor > 1) {
    printf("%s %s has too few bins or excessively small bin size for given frequency\n",
           entity_name[tcontext.elementType],
           tcontext.elementName);
    if (n_bins * factor > maxBins2) {
      if ((n_bins * factor) / maxBins2 > 50) {
        if (Q < 1)
          printf(" With %ld bins, the frequency resolution is only %.1g times the resonant frequency\n",
                 maxBins2, freq * maxBins2 * bin_size);
        else
          printf(" With %ld bins, the frequency resolution is only %.1g times the resonance width\n",
                 maxBins2, freq / Q * maxBins2 * bin_size);
        printf(" It isn't possible to model this situation accurately with %ld bins.  Consider the RFMODE or TRFMODE element.\n",
               maxBins2);
        printf(" Alternatively, consider increasing your bin size or maximum number of bins\n");
#if USE_MPI
        mpiAbort = 1;
        return;
#else
        exitElegant(1);
#endif
      }
      n_bins = maxBins2;
      printf("  Number of bins adjusted to %ld\n", n_bins);

    } else {
      n_bins = pow(2, (long)(log(n_bins * factor) / log(2) + 1));
      printf("  Number of bins adjusted to %ld\n",
             n_bins);
    }
    fflush(stdout);
  }

  *nBins = n_bins;
  *binSize = bin_size;
}
