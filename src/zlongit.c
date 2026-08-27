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

/* Sign conventions in this module:
 * V = I*Z 
 * V(t) = V(w)*exp(i*w*t)  I(t) = I(w)*exp(i*w*t)
 * For inductor:   Z = i*w*L
 * For capacitor:  Z = -i/(w*C)
 * For resistor:   Z = R
 * For resonator:  Z = R/(1+i*Q*(w/wr-wr/w))
 *
 * The minus sign to get energy loss instead of energy gain is 
 * used internally and must not be included in the impedance.
 *
 */

#include "mdb.h"
#include "track.h"
#include "table.h"
#include "fftpackC.h"

#define WAKE_COLUMNS 3
static SDDS_DEFINITION wake_column[WAKE_COLUMNS] = {
  {"Deltat", "&column name=Deltat, symbol=\"$gD$rt\", units=s, type=double, description=\"Time after head of bunch\" &end"},
  {"Wz", "&column name=Wz, symbol=\"W$bz$n\", units=V, type=double, description=\"Longitudinal wake\" &end"},
  {"LinearDensity", "&column name=LinearDensity, units=C/s, type=double &end"},
};

#define WAKE_PARAMETERS 6
#define BB_WAKE_PARAMETERS 6
#define NBB_WAKE_PARAMETERS 3
static SDDS_DEFINITION wake_parameter[WAKE_PARAMETERS] = {
  {"Pass", "&parameter name=Pass, type=long &end"},
  {"Bunch", "&parameter name=Bunch, type=long &end"},
  {"q", "&parameter name=q, units=C, type=double, description=\"Total charge\" &end"},
  {"Ra", "&parameter name=Ra, symbol=\"R$ba$n\", units=\"$gW$r\", type=double, description=\"Broad-band impedance\" &end"},
  {"fo", "&parameter name=fo, symbol=\"f$bo$n\", units=Hz, type=double, description=\"Frequency of BB resonator\" &end"},
  {"Deltaf", "&parameter name=Deltaf, symbol=\"$gD$rf\", units=Hz, type=double, description=\"Frequency sampling interval\" &end"},
};

void set_up_zlongit(ZLONGIT *zlongit, RUN *run, long pass, long particles, CHARGE *charge,
                    double timeSpan);

void track_through_zlongit(double **part0, int64_t np0, ZLONGIT *zlongit, double Po,
                           RUN *run, long i_pass, CHARGE *charge) {
  double *Itime = NULL;    /* array for histogram of particle density */
  double *Ifreq = NULL;    /* array for FFT of histogram of particle density */
  double *Vtime = NULL;    /* array for voltage acting on each bin */
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  int64_t max_np = 0;
  double *Vfreq, *Z;
  int64_t np = 0, ip, n_binned;
  long nfreq, iReal, iImag, ib, nb = 0;
  double factor, tmin, tmax, tmean, dt, dt1, dgam, rampFactor;
  long i_pass0;
  long iBucket, nBuckets;
#if USE_MPI
  static long not_first_call = -1;
  double *buffer = NULL;
  double tmin_part, tmax_part; /* record the actual tmin and tmax for particles to reduce communications */
  long offset = 0, length = 0;
  int64_t particles_total;
#endif

  /*
#if USE_MPI && defined(DEBUG)  
  static FILE *fpdeb = NULL;
  char s[100];
  if (!fpdeb) {
    sprintf(s, "zlongit.debu%02d", myid);
    fpdeb = fopen(s, "w");
    fprintf(fpdeb, "SDDS1\n&column name=time type=double &end\n&column name=dt1 type=double &end\n&column name=ib type=long &end\n");
    fprintf(fpdeb, "&column name=V0 type=double &end\n&column name=V1 type=double &end\n&column name=dgamma type=double &end\n&data mode=ascii no_row_counts=1 &end\n");
  }
#endif
  */

#ifdef USE_MPE /* use the MPE library */
  int event1a, event1b, event2a, event2b, event3a, event3b;
  event1a = MPE_Log_get_event_number();
  event1b = MPE_Log_get_event_number();
  event2a = MPE_Log_get_event_number();
  event2b = MPE_Log_get_event_number();
  event3a = MPE_Log_get_event_number();
  event3b = MPE_Log_get_event_number();
  MPE_Describe_state(event1a, event1b, "Histogram", "red");
  MPE_Describe_state(event2a, event2b, "Compute", "yellow");
  MPE_Describe_state(event3a, event3b, "Kick", "orange");
#endif

  i_pass0 = i_pass;
  if ((i_pass -= zlongit->startOnPass) < 0 || zlongit->factor == 0)
    return;

  rampFactor = 0;
  if (i_pass >= (zlongit->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / zlongit->rampPasses;

#if USE_MPI
    not_first_call += 1;
#endif

#if defined(DEBUG) && USE_MPI
  printf("ZLONGIT, myid = %d\n", myid);
#endif

  if (isSlave || !distributedBeam) {
    index_bunch_assignments(part0, np0, (charge && zlongit->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif

#ifdef DEBUG
    printf("%ld buckets\n", nBuckets);
    if (nBuckets > 1) {
      fflush(stdout);
      for (iBucket = 0; iBucket < nBuckets; iBucket++) {
        printf("bucket %ld: %" PRId64 " particles\n", iBucket, npBucket ? npBucket[iBucket] : 0);
        fflush(stdout);
      }
    }
#endif

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (zlongit->bunchedBeamMode && 
          ((zlongit->startBunch>=0 && iBucket<zlongit->startBunch) ||
           (zlongit->endBunch>=0 && iBucket>zlongit->endBunch)))
        continue;
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
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
        if (np > max_np) {
#ifdef DEBUG
          printf("ZLONGIT: setting up work arrays, iBucket=%ld, np=%" PRId64 "\n", iBucket, np);
          fflush(stdout);
#endif
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)trealloc(time, sizeof(*time) * np);
          pbin = trealloc(pbin, sizeof(*pbin) * np);
          max_np = np;
        }
        if (np > 0) {
#ifdef DEBUG
          printf("ZLONGIT: copying data to work array, iBucket=%ld, np=%" PRId64 "\n", iBucket, np);
          fflush(stdout);
#endif
          for (ip = 0; ip < np; ip++) {
            time[ip] = time0[ipBucket[iBucket][ip]];
            memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
          }
        }
      }

      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#if USE_MPI
      find_global_min_max(&tmin, &tmax, np, workers);
      tmin_part = tmin;
      tmax_part = tmax;
      tmean = computeAverage_p(time, np, workers);
#else
      compute_average(&tmean, time, np);
#endif

      /* use np0 here since we may need to compute the macroparticle charge */
      set_up_zlongit(zlongit, run, i_pass, np0, charge, tmax - tmin);
      nb = zlongit->n_bins;
      dt = zlongit->bin_size;

#ifdef DEBUG
      printf("Allocating histogram arrays, nb=%ld\n", nb);
      fflush(stdout);
#endif
      if (!Itime)
        Itime = tmalloc(2 * sizeof(*Itime) * nb);
      if (!Ifreq)
        Ifreq = tmalloc(2 * sizeof(*Ifreq) * nb);
      if (!Vtime)
        Vtime = tmalloc(2 * sizeof(*Vtime) * (nb + 1));

      if ((tmax - tmin) * 2 > nb * dt) {
        TRACKING_CONTEXT tcontext;
        getTrackingContext(&tcontext);
#if USE_MPI && !defined(MPI_DEBUG)
        if (myid == 1)
          dup2(fdStdout, fileno(stdout));
#endif
        printf("%s %s: Time span of bunch %ld (%21.15le->%21.15le, span %21.15le s) is more than half the total time span (%21.15le s).\n",
               entity_name[tcontext.elementType],
               tcontext.elementName, iBucket, tmin, tmax, tmax - tmin, nb * dt);
        printf("If using broad-band impedance, you should increase the number of bins (or use auto-scaling) and rerun.\n");
        printf("If using file-based impedance, you should increase the number of data points or decrease the frequency resolution.\n");
        if (!zlongit->allowLongBeam) {
#if USE_MPI
#  if MPI_DEBUG
          for (ip = 0; ip < np; ip++)
            printf("particle %5" PRId64 ": t=%21.15e, delta=%21.15e\n", ip, time[ip], part[ip][5]);
          printf("Issuing MPI abort from ZLONGIT\n");
          fflush(stdout);
#  endif
          mpiAbort = MPI_ABORT_BUNCH_TOO_LONG_ZLONGIT;
          return;
#else
          exitElegant(1);
#endif
        }
      }

      if (zlongit->reverseTimeOrder) {
        for (ip = 0; ip < np; ip++)
          time[ip] = 2 * tmean - time[ip];
      }

      if (zlongit->bin_centered) {
        /* ZTRANSVERSE/IMPEDANCE convention: bunch left-justified in grid,
         * bin centers at tmin + ib*dt. */
        tmin -= dt;
      } else {
        /* Legacy ZLONGIT convention: bunch centered in grid, bin
         * left-edges at tmin + ib*dt (bin centers at tmin + (ib+0.5)*dt). */
        tmin = tmean - dt * nb / 2.0;
      }

#ifdef DEBUG
      printf("tmin = %21.15e  tmax = %21.15e  dt = %21.15e  nb = %ld\n",
             tmin, tmax, dt, nb);
#endif

      for (ib = 0; ib < nb; ib++)
        Itime[2 * ib] = Itime[2 * ib + 1] = 0;

      n_binned = 0;
      for (ip = 0; ip < np; ip++) {
        pbin[ip] = -1;
        if (zlongit->bin_centered)
          ib = (long)((time[ip] - tmin) / dt + 0.5);
        else
          ib = (time[ip] - tmin) / dt;
        if (ib < 0)
          continue;
        if (ib > nb - 1)
          continue;
        if (zlongit->area_weight && ib > 1 && ib < (nb - 1)) {
          double dist, bin_center;
          if (zlongit->bin_centered)
            bin_center = ib * dt + tmin;
          else
            bin_center = (ib + 0.5) * dt + tmin;
          dist = (time[ip] - bin_center) / dt;
          Itime[ib] += 0.5;
          Itime[ib - 1] += 0.25 - 0.5 * dist;
          Itime[ib + 1] += 0.25 + 0.5 * dist;
        } else
          Itime[ib] += 1;
        pbin[ip] = ib;
        n_binned++;
      }
#if (!USE_MPI)
      if (n_binned != np) {
        char warningBuffer[1024];
        snprintf(warningBuffer, 1024, "Only %" PRId64 " of %" PRId64 " particles were binned; results suspect. Reduce impedance frequency spacing for file-based impedances or invoking auto-scaling for broad-band impedances.",
                 n_binned, np);
        printWarningForTracking("Not all particles binned in ZLONGIT.",
                                warningBuffer);
      }
#else
      if (USE_MPI) {
        int all_binned, result = 1;
        if (isSlave)
          result = ((n_binned == np) ? 1 : 0);

        MPI_Allreduce(&result, &all_binned, 1, MPI_INT, MPI_LAND, workers);
        if (!all_binned && !not_first_call) {
#  ifndef MPI_DEBUG
          if (myid == 1)
            dup2(fdStdout, fileno(stdout));
#  endif
          printf("*** Not all particles binned in ZLONGIT. This may produce unphysical results.  Your impedance needs smaller frequency\n");
          printf("    spacing to cover a longer time span. Invoking auto-scaling may help for broad-band impedances. \n");
          fflush(stdout);
#  ifndef MPI_DEBUG
#    if defined(_WIN32)
          if (myid == 1)
            freopen("NUL", "w", stdout);
#    else
          if (myid == 1)
            if (!freopen("/dev/null", "w", stdout)) {
              perror("freopen failed");
              exit(EXIT_FAILURE);
            }
#    endif
#  endif
        }
      }
#endif

#if USE_MPI
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
#  ifdef USE_MPE
      MPE_Log_event(event1a, 0, "start histogram");
#  endif
      if (isSlave) {
#  if MPI_DEBUG
        printf("histogram transfer: offset = %ld, length = %ld, nb = %ld\n", offset, length, nb);
        fflush(stdout);
#  endif
        buffer = malloc(sizeof(double) * length);
        MPI_Allreduce(&Itime[offset], buffer, length, MPI_DOUBLE, MPI_SUM, workers);
        memcpy(&Itime[offset], buffer, sizeof(double) * length);
        free(buffer);
        buffer = NULL;
      }
#  ifdef USE_MPE
      MPE_Log_event(event1b, 0, "end histogram");
#  endif
#  ifdef USE_MPE
      MPE_Log_event(event2a, 0, "start computation");
#  endif
#endif

      if (zlongit->smoothing)
        SavitzyGolaySmooth(Itime, nb, zlongit->SGOrder,
                           zlongit->SGHalfWidth, zlongit->SGHalfWidth, 0);

#ifdef DEBUG
      /* Output the time-binned data */
      if (0) {
        FILE *fp;
        fp = fopen("zlongit.tbin", "w");
        fprintf(fp, "SDDS1\n&column name=t type=double units=s &end\n&column name=I type=double &end\n&data mode=ascii &end\n");
        fprintf(fp, "%ld\n", nb);
        for (ib = 0; ib < nb; ib++)
          fprintf(fp, "%e %e\n",
                  ib * dt + tmin, Itime[ib] * zlongit->macroParticleCharge * particleRelSign / dt);
        fclose(fp);
      }
#endif

      /* Take the FFT of I(t) to get I(f) */
      memcpy(Ifreq, Itime, 2 * nb * sizeof(*Ifreq));
      realFFT(Ifreq, nb, 0);

      /* Compute V(f) = Z(f)*I(f), putting in a factor 
       * to normalize the current waveform.
       */
      Vfreq = Vtime;
      factor = zlongit->macroParticleCharge * particleRelSign / dt * zlongit->factor * rampFactor;
      Z = zlongit->Z;
      Vfreq[0] = Ifreq[0] * Z[0] * factor;
      nfreq = nb / 2 + 1;
      if (nb % 2 == 0)
        /* Nyquist term */
        Vfreq[nb - 1] = Ifreq[nb - 1] * Z[nb - 1] * factor;
      for (ib = 1; ib < nfreq - 1; ib++) {
        iImag = (iReal = 2 * ib - 1) + 1;
        Vfreq[iReal] = (Ifreq[iReal] * Z[iReal] - Ifreq[iImag] * Z[iImag]) * factor;
        Vfreq[iImag] = (Ifreq[iReal] * Z[iImag] + Ifreq[iImag] * Z[iReal]) * factor;
      }

      /* Compute inverse FFT of V(f) to get V(t) */
      realFFT(Vfreq, nb, INVERSE_FFT);
      Vtime = Vfreq;
#ifdef USE_MPE
      MPE_Log_event(event2b, 0, "end computation");
#endif

#if USE_MPI
      MPI_Allreduce(&np, &particles_total, 1, MPI_INT64_T, MPI_SUM, workers);
      if (myid == 1) {
#endif
        if (zlongit->SDDS_wake_initialized && zlongit->wakes) {
          /* wake potential output */
          factor = zlongit->macroParticleCharge * particleRelSign / dt;
          if ((zlongit->wake_interval <= 0 || ((i_pass0 - zlongit->wake_start) % zlongit->wake_interval) == 0) &&
              i_pass0 >= zlongit->wake_start && (zlongit->wake_end<0 || i_pass0 <= zlongit->wake_end)) {
            if (!SDDS_StartTable(zlongit->SDDS_wake, nb)) {
              SDDS_SetError("Problem starting SDDS table for wake output (track_through_zlongit)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            for (ib = 0; ib < nb; ib++) {
              if (!SDDS_SetRowValues(zlongit->SDDS_wake, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                                     0, ib * dt, 1, Vtime[ib], 2, Itime[ib] * factor, -1)) {
                SDDS_SetError("Problem setting rows of SDDS table for wake output (track_through_zlongit)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
            }
            if (!SDDS_SetParameters(zlongit->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                    "Pass", i_pass0, "Bunch", iBucket, "q",
#if USE_MPI
                                    zlongit->macroParticleCharge * particleRelSign * particles_total,
#else
                                  zlongit->macroParticleCharge * particleRelSign * np,
#endif
                                    NULL)) {
              SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_zlongit)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            if (zlongit->broad_band) {
              if (!SDDS_SetParameters(zlongit->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                      "Ra", zlongit->Ra, "fo", zlongit->freq, "Deltaf", 1. / dt, NULL)) {
                SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_zlongit)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
            }
            if (!SDDS_WriteTable(zlongit->SDDS_wake)) {
              SDDS_SetError("Problem writing SDDS table for wake output (track_through_zlongit)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            if (!inhibitFileSync)
              SDDS_DoFSync(zlongit->SDDS_wake);
          }
        }
#if USE_MPI
      }
#endif

#ifdef USE_MPE
      MPE_Log_event(event3a, 0, "start bunch kicks");
#endif
      /* put zero voltage in Vtime[nb] for use in interpolation */
      Vtime[nb] = 0;
      /* change particle momentum offsets to reflect voltage in relevant bin */
      for (ip = 0; ip < np; ip++) {
        if ((ib = pbin[ip]) >= 0 && ib <= (nb - 1)) {
          if (zlongit->interpolate && ib > 0) {
            /* bin_centered convention: bin center at tmin + dt*ib.
             * legacy convention: bin center at tmin + dt/2 + dt*ib. */
            if (zlongit->bin_centered)
              dt1 = time[ip] - (tmin + dt * ib);
            else
              dt1 = time[ip] - (tmin + dt / 2.0 + dt * ib);
            if (dt1 < 0)
              dt1 += dt;
            else
              ib += 1;
            if (ib < nb)
              dgam = (Vtime[ib - 1] + (Vtime[ib] - Vtime[ib - 1]) / dt * dt1) / (1e6 * particleMassMV * particleRelSign);
            else
              continue;
            /*
#if USE_MPI && defined(DEBUG)
            fprintf(fpdeb, "%21.15e %21.15e %ld %21.15e %21.15e %21.15e\n",
                    time[ip], dt1, ib, Vtime[ib-1], Vtime[ib], dgam);
#endif
            */
          } else
            dgam = Vtime[ib] / (1e6 * particleMassMV * particleRelSign);
          if (dgam) {
            if (zlongit->reverseTimeOrder)
              time[ip] = 2 * tmean - time[ip];
            /* Put in minus sign here as the voltage decelerates the beam */
            add_to_particle_energy(part[ip], time[ip], Po, -dgam);
          }
        }
      }
#ifdef USE_MPE
      MPE_Log_event(event3b, 0, "end bunch kicks");
#endif

      if (nBuckets != 1 && np > 0) {
#ifdef DEBUG
        printf("ZLONGIT: copying data back to main array\n");
        fflush(stdout);
#endif
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
#if USE_MPI
#  ifdef DEBUG
      printf("Preparing to wait on barrier at end of loop for bucket %ld\n", iBucket);
      fflush(stdout);
#  endif
      MPI_Barrier(workers);
#endif
#ifdef DEBUG
      printf("Done with bucket %ld\n", iBucket);
      fflush(stdout);
#endif
    }
  }

#if USE_MPI
#  ifdef DEBUG
  printf("Preparing to wait on barrier\n");
  fflush(stdout);
#  endif
  MPI_Barrier(workers);
#endif

  /*
#if USE_MPI && defined(DEBUG)
  fprintf(fpdeb, "\n");
#endif
  */

#ifdef DEBUG
  printf("Preparing to free memory, max_np=%" PRId64 ", nb=%ld, np0=%" PRId64 ", nBuckets=%ld\n",
         max_np, nb, np0, nBuckets);
  fflush(stdout);
#endif

  if (Itime)
    free(Itime);
  if (Vtime)
    free(Vtime);
  if (Ifreq)
    free(Ifreq);
  if (part && part != part0 && max_np > 0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin)
    free(pbin);

  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

#if USE_MPI
  MPI_Barrier(workers);
#endif

#ifdef DEBUG
  printf("Done with ZLONGIT\n");
  fflush(stdout);
#endif
}

void set_up_zlongit(ZLONGIT *zlongit, RUN *run, long pass, long particles, CHARGE *charge,
                    double timeSpan) {
  long i, nfreq;
  double df;

  if (charge) {
    zlongit->macroParticleCharge = charge->macroParticleCharge;
  } else if (pass == 0) {
    zlongit->macroParticleCharge = 0;
    if (zlongit->charge < 0)
      bombElegant("ZLONGIT charge parameter should be non-negative.  Use change_particle to set particle charge state.", NULL);
#if (!USE_MPI)
    if (particles)
      zlongit->macroParticleCharge = zlongit->charge / particles;
#else
    if (USE_MPI) {
      long particles_total;

      MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, workers);
      if (particles_total)
        zlongit->macroParticleCharge = zlongit->charge / particles_total;
    }
#endif
  }

  if (zlongit->initialized)
    return;

  if (zlongit->broad_band) {
    /* compute impedance for a resonator.  Recall that I use V(t) = Vo*exp(i*w*t) convention,
       * so the impedance is Z(w) = (Ra/2)*(1 + i*T)/(1+T^2), where T=Q*(wo/w-w/wo).
       * The imaginary and real parts are positive for small w.
       */
    double term;
    if (zlongit->bin_size <= 0)
      bombElegant("bin_size must be positive for ZLONGIT element", NULL);
    if (zlongit->Ra && zlongit->Rs)
      bombElegant("ZLONGIT element broad-band resonator may have only one of Ra or Rs nonzero.  Ra is just 2*Rs", NULL);
    if (!zlongit->Ra)
      zlongit->Ra = 2 * zlongit->Rs;
    if (zlongit->n_bins % 2 != 0)
      bombElegant("ZLONGIT element must have n_bins divisible by 2", NULL);
    if (zlongit->Zreal || zlongit->Zimag)
      bombElegant("can't specify both broad_band impedance and Z(f) files for ZLONGIT element", NULL);
    optimizeBinSettingsForImpedance(timeSpan, zlongit->freq, zlongit->Q,
                                    &(zlongit->bin_size), &(zlongit->n_bins), zlongit->max_n_bins);
    df = 1 / (zlongit->n_bins * zlongit->bin_size) / (zlongit->freq);
    nfreq = zlongit->n_bins / 2 + 1;
    printf("ZLONGIT has %ld frequency points with df=%e\n",
           nfreq, df);
    fflush(stdout);
    zlongit->Z = tmalloc(sizeof(*(zlongit->Z)) * zlongit->n_bins);
    zlongit->Z[0] = 0;                   /* DC term */
    zlongit->Z[zlongit->n_bins - 1] = 0; /* Nyquist term */
    for (i = 1; i < nfreq - 1; i++) {
      term = zlongit->Q * (1.0 / (i * df) - i * df);
      zlongit->Z[2 * i - 1] = zlongit->Ra / 2 / (1 + sqr(term)); /* Real part */
      zlongit->Z[2 * i] = zlongit->Z[2 * i - 1] * term;          /* Imaginary part */
    }
    if (0) {
      FILE *fp;
      fp = fopen("zbb.debug", "w");
      fputs("SDDS1\n&column name=Index, type=long &end\n", fp);
      fputs("&column name=zReal, type=double &end\n", fp);
      fputs("&column name=zImag, type=double &end\n", fp);
      fputs("&data mode=ascii, no_row_counts=1 &end\n", fp);
      fprintf(fp, "0 %e 0\n", zlongit->Z[0]);
      for (i = 1; i < nfreq - 1; i++)
        fprintf(fp, "%ld %e %e\n",
                i, zlongit->Z[2 * i - 1], zlongit->Z[2 * i]);
      fprintf(fp, "%ld %e 0\n", i, zlongit->Z[zlongit->n_bins - 1]);
      fclose(fp);
    }
    df *= zlongit->freq;
  } else {
    TABLE Zr_data, Zi_data;
    double *Zr = NULL, *Zi = NULL;
    double df_spect = 0;
    long n_spect = 0;
    if (!zlongit->Zreal && !zlongit->Zimag)
      bombElegant("you must either give broad_band=1, or Zreal and/or Zimag (ZLONGIT)", NULL);
    if (zlongit->Zreal && !getTableFromSearchPath(&Zr_data, zlongit->Zreal, 1, 0))
      bombElegantVA("unable to read real impedance function (ZLONGIT) from file %s\n", zlongit->Zreal);
    if (zlongit->Zimag && !getTableFromSearchPath(&Zi_data, zlongit->Zimag, 1, 0))
      bombElegantVA("unable to read imaginary impedance function (ZLONGIT) from file %s\n", zlongit->Zimag);
    if (zlongit->Zreal && !zlongit->Zimag) {
      if (!checkPointSpacing(Zr_data.c1, Zr_data.n_data, 1e-6))
        bombElegant("frequency values not equally spaced for real data (ZLONGIT)", NULL);
      Zr = Zr_data.c2;
      if ((n_spect = Zr_data.n_data) < 2)
        bombElegant("too little data in real impedance input file (ZLONGIT)", NULL);
      df_spect = Zr_data.c1[1] - Zr_data.c1[0];
      Zi = tmalloc(sizeof(*Zi) * n_spect);
      for (i = 0; i < n_spect; i++)
        Zi[i] = 0;
    } else if (zlongit->Zimag && !zlongit->Zreal) {
      if (!checkPointSpacing(Zi_data.c1, Zi_data.n_data, 1e-6))
        bombElegant("frequency values not equally spaced for real data (ZLONGIT)", NULL);
      Zi = Zi_data.c2;
      if ((n_spect = Zi_data.n_data) < 2)
        bombElegant("too little data in imaginary impedance input file (ZLONGIT)", NULL);
      df_spect = Zi_data.c1[1] - Zi_data.c1[0];
      Zr = tmalloc(sizeof(*Zr) * n_spect);
      for (i = 0; i < n_spect; i++)
        Zr[i] = 0;
    } else if (zlongit->Zimag && zlongit->Zreal) {
      if (!checkPointSpacing(Zr_data.c1, Zr_data.n_data, 1e-6))
        bombElegant("frequency values not equally spaced for real data (ZLONGIT)", NULL);
      if (!checkPointSpacing(Zi_data.c1, Zi_data.n_data, 1e-6))
        bombElegant("frequency values not equally spaced for real data (ZLONGIT)", NULL);
      if (Zi_data.n_data != Zr_data.n_data)
        bombElegant("real and imaginary impedance files have different amounts of data (ZLONGIT)", NULL);
      n_spect = Zi_data.n_data;
      df_spect = Zi_data.c1[1] - Zi_data.c1[0];
      if (df_spect != (Zi_data.c1[1] - Zi_data.c1[0]))
        bombElegant("real and imaginary impedance files have different frequency spacing (ZLONGIT)", NULL);
      Zi = Zi_data.c2;
      Zr = Zr_data.c2;
    }
    if (Zi[0])
      bombElegantVA("impedance spectrum has non-zero imaginary DC term: Z(0)=(%le, %le) (ZLONGIT)\n", Zr[0], Zi[0]);
    if (!power_of_2(n_spect - 1))
      bombElegant("number of spectrum points must be 2^n+1, n>1 (ZLONGIT)", NULL);
    zlongit->n_bins = 2 * (n_spect - 1);
    zlongit->bin_size = 1.0 / (zlongit->n_bins * df_spect);
    nfreq = n_spect;
    printf("Using Nb=%ld and dt=%e s (span of %e s) in ZLONGIT\n",
           zlongit->n_bins, zlongit->bin_size, zlongit->n_bins * zlongit->bin_size);
    fflush(stdout);
    zlongit->Z = tmalloc(sizeof(*zlongit->Z) * 2 * zlongit->n_bins);
    for (i = 0; i < n_spect; i++) {
      if (i == 0)
        /* DC term (multiply by 2 for consistency with sddsfft) */
        zlongit->Z[0] = 2 * Zr[0];
      else if (i == n_spect - 1 && zlongit->n_bins % 2 == 0)
        /* Nyquist term */
        zlongit->Z[2 * i - 1] = 2 * Zr[i];
      else {
        zlongit->Z[2 * i - 1] = Zr[i];
        zlongit->Z[2 * i] = Zi[i];
      }
    }
    df = df_spect;
  }

  zlongit->SDDS_wake_initialized = 0;
#if USE_MPI
  if (zlongit->SDDS_wake_initialized && !SDDS_Terminate(zlongit->SDDS_wake)) {
    SDDS_SetError("Problem terminating SDDS output (set_up_zlongit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (myid == 1) {
#endif
    if (zlongit->wakes) {
      char *filename;
      filename = compose_filename(zlongit->wakes, run->rootname);
      zlongit->SDDS_wake = tmalloc(sizeof(*(zlongit->SDDS_wake)));
      if (zlongit->broad_band)
        SDDS_ElegantOutputSetup(zlongit->SDDS_wake, filename, SDDS_BINARY, 1, "longitudinal wake",
                                run->runfile, run->lattice, wake_parameter, BB_WAKE_PARAMETERS,
                                wake_column, WAKE_COLUMNS, "set_up_zlongit", SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
      else {
        SDDS_ElegantOutputSetup(zlongit->SDDS_wake, filename, SDDS_BINARY, 1, "longitudinal wake",
                                run->runfile, run->lattice, wake_parameter, NBB_WAKE_PARAMETERS,
                                wake_column, WAKE_COLUMNS, "set_up_zlongit", SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
      }
      free(filename);
#if USE_MPI
      if (!SDDS_WriteLayout(zlongit->SDDS_wake)) {
#  ifndef MPI_DEBUG
        dup2(fdStdout, fileno(stdout));
#  endif
        printf("Error: unable to write layout for ZLONGIT wake file %s\n", zlongit->wakes);
        fflush(stdout);
#  ifndef MPI_DEBUG
        if (!freopen("/dev/null", "w", stdout)) {
          perror("freopen failed");
          exit(EXIT_FAILURE);
        }
#  endif
        MPI_Abort(MPI_COMM_WORLD, T_ZLONGIT);
        return;
      }
#endif
      zlongit->SDDS_wake_initialized = 1;
    }
#if USE_MPI
  }
#endif

  if (zlongit->highFrequencyCutoff0 > 0)
    applyLowPassFilterToImpedance(zlongit->Z, nfreq,
                                  zlongit->highFrequencyCutoff0,
                                  zlongit->highFrequencyCutoff1);

#if 0
    if (!zlongit->initialized) {
      FILE *fp;
      fp = fopen_e("zlongit.sdds", "w", 0);
      fprintf(fp, "SDDS1\n&column name=f units=Hz type=double &end\n");
      fprintf(fp, "&column name=ZReal type=double &end\n");
      fprintf(fp, "&column name=ZImag type=double &end\n");
      fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
      for (i=0; i<nfreq; i++) 
        if (i==0)
          fprintf(fp, "%21.15e %21.15e %21.15e\n",
                  i*df, zlongit->Z[0], 0.0);
        else 
          fprintf(fp, "%21.15e %21.15e %21.15e\n",
                  i*df, zlongit->Z[2*i-1], zlongit->Z[2*i]);
      fclose(fp);
    }
#endif

  zlongit->initialized = 1;
}

void applyLowPassFilterToImpedance(double *Z, long nfreq, double cutoff0, double cutoff1) {
  long i;
  double f;

  for (i = 1; i < nfreq - 1; i++) {
    f = (i * 1.0) / nfreq;
    if (f < cutoff0)
      continue;
    else if (f > cutoff1 || cutoff1 <= cutoff0)
      Z[2 * i - 1] = Z[2 * i] = 0;
    else {
      Z[2 * i - 1] *= 1 - (f - cutoff0) / (cutoff1 - cutoff0);
      Z[2 * i] *= 1 - (f - cutoff0) / (cutoff1 - cutoff0);
    }
  }
}

long checkPointSpacing(double *x, long n, double tolerance) {
  double dx, dx0, range;
  long i;

  if (n < 3)
    return 1;
  if ((range = x[n - 1] - x[0]) <= 0)
    return 0;
  dx0 = (x[1] - x[0]) / range;
  for (i = 1; i < n - 1; i++) {
    dx = (x[i + 1] - x[i]) / range;
    if (fabs(dx - dx0) > tolerance)
      return 0;
  }
  return 1;
}

#if USE_MPI
double computeAverage_p(double *data, int64_t np, MPI_Comm mpiComm) {
  double tSum = 0;
  int64_t ip;
  double error = 0.0;
  int64_t np_total;

  for (ip = tSum = 0; ip < np; ip++) {
    tSum = KahanPlus(tSum, data[ip], &error);
  }

  if (isMaster) {
    tSum = 0.0;
    np = 0;
  }
  MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, mpiComm);
  if (np_total > 0)
    return KahanParallel(tSum, error, mpiComm) / np_total;
  return 0.0;
}
#endif
