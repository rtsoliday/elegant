/*************************************************************************\
* Copyright (c) 2003 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2003 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: lsc.c 
 *
 * Michael Borland, 2003
 */
#include "mdb.h"
#include "track.h"

#include "fftpackC.h"
#ifdef HAVE_GPU
#  include "gpu_lsc.h"
#endif

void track_through_lscdrift(double **part, int64_t np, LSCDRIFT *LSC, double Po, CHARGE *charge) {
  static double *Itime = NULL; /* array for histogram of particle density */
  static double *Ifreq = NULL; /* array for FFT of histogram of particle density */
  static double *Vtime = NULL; /* array for voltage acting on each bin */
  static long max_n_bins = 0;
  static long *pbin = NULL;   /* array to record which bin each particle is in */
  static double *time = NULL; /* array to record arrival time of each particle */
  static long max_np = 0;
  double *Vfreq, ZImag;
  short kickMode = 0;
  long nb, n_binned = 0, nfreq, iReal, iImag;
  int64_t ib;
  double factor, tmin, tmax, dt, df, dk, a1, a2;
  double lengthLeft, Imin, Imax, kSC, Zmax;
  double Ia = 17045, Z0, length, k;
  double S11, S33, beamRadius;
  TRACKING_CONTEXT context;
  ELEMENT_LIST *eptr;
#if DEBUG
  FILE *fpd = NULL;
#endif
#if USE_MPI
  double *buffer;
#endif

  getTrackingContext(&context);
  eptr = context.element;
  if (eptr->pred && LSC->length == 0 && LSC->autoLEffective) {
    ELEMENT_LIST *pred;
    pred = eptr->pred;
    if (entity_description[pred->type].flags & HAS_LENGTH) {
      LSC->lEffective = ((DRIFT *)pred->p_elem)->length;
      printf("Set LEffective=%le m for LSCDRIFT %s following %s\n", LSC->lEffective, eptr->name, eptr->pred->name);
      fflush(stdout);
    } else
      return;
  }
#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    gpu_track_through_lscdrift(np, LSC, Po, charge);
#  ifdef GPU_VERIFY
    startCpuTimer();
    track_through_lscdrift(part, np, LSC, Po, charge);
    compareGpuCpu(np, "track_through_lscdrift");
#  endif /* GPU_VERIFY */
    return;
  }
#endif /* HAVE_GPU */

  if (LSC->lsc == 0) {
    if (isSlave || !distributedBeam)
      exactDrift(part, np, LSC->length);
    return;
  }

  if (!charge)
    bombElegant("No charge defined for LSC.  Insert a CHARGE element in the beamline.", NULL);

  Z0 = sqrt(mu_o / epsilon_o);
  if ((nb = LSC->bins) < 2) {
    printf("Error: LSC must have an BINS>=2\n");
    exitElegant(1);
  }
  if (nb % 2 == 1) {
    printf("Error: LSC must have an even number of bins\n");
    exitElegant(1);
  }

#if DEBUG
  printf("LSC: np=%" PRId64 ", nb=%ld, L=%le, bt=%d\n", np, nb, LSC->length, LSC->backtrack);
  fflush(stdout);
#endif

  if (nb > max_n_bins) {
    max_n_bins = nb;
    Itime = trealloc(Itime, 2 * sizeof(*Itime) * (max_n_bins + 1));
    Ifreq = trealloc(Ifreq, 2 * sizeof(*Ifreq) * (max_n_bins + 1));
    Vtime = trealloc(Vtime, 2 * sizeof(*Vtime) * (max_n_bins + 1));
  }

  if (!USE_MPI || !distributedBeam) {
    if (np > max_np) {
      pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
      time = trealloc(time, sizeof(*time) * max_np);
    }
  }
#if USE_MPI
  else if (USE_MPI && distributedBeam) {
    int64_t np_total;
    if (isSlave) {
      MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, workers);
      if (np_total > max_np) {
        /* if the total number of particles is increased, we do reallocation for every CPU */
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
        time = trealloc(time, sizeof(*time) * max_np);
        max_np = np_total; /* max_np should be the sum across all the processors */
      }
    }
  }
#endif
  if ((lengthLeft = fabs(LSC->length)) == 0) {
    lengthLeft = fabs(LSC->lEffective);
    kickMode = 1;
  }
  while (lengthLeft > 0) {
#if DEBUG
    printf("lengthLeft = %le\n", lengthLeft);
    fflush(stdout);
#endif
    /* compute time coordinates and make histogram */
    if (isSlave || !distributedBeam)
      computeTimeCoordinatesOnly(time, Po, part, np);
    find_min_max(&tmin, &tmax, time, np);
#if USE_MPI
    if (distributedBeam) {
      if (isMaster) {
        tmin = DBL_MAX;
        tmax = -DBL_MAX;
      }
      find_global_min_max(&tmin, &tmax, nb, MPI_COMM_WORLD);
    }
#endif
    dt = (tmax - tmin) / (nb - 3);
#if DEBUG
    printf("tmin=%e, tmax=%e, dt=%e\n",
           tmin, tmax, dt);
    fflush(stdout);
#endif
    if (isSlave || !distributedBeam)
      n_binned = binTimeDistribution(Itime, pbin, tmin, dt, nb, time, part, Po, np);

    if (!USE_MPI || !distributedBeam) {
      if (n_binned != np) {
        char warningText[1024];
        snprintf(warningText, 1024,
                 "Only %ld of %" PRId64 " particles were binned. This shouldn't happen.", n_binned, np);
        printWarningForTracking("Some particles were not binned in LSCDRIFT.",
                                warningText);
      }
    }
#if USE_MPI
#  if MPI_DEBUG
    else if (isSlave) {
      int all_binned, result = 1;

      result = ((n_binned == np) ? 1 : 0);
      MPI_Allreduce(&result, &all_binned, 1, MPI_INT, MPI_LAND, workers);
      if (!all_binned) {
        if (myid == 1) {
          /* This warning will be given only if the flag MPI_DEBUG is defined for the Pelegant to avoid communications */
          printWarningForTracking("Some particles were not binned in LSCDRIFT.", "This shouldn't happen.");
        }
      }
    }
#  endif
    if (isSlave && distributedBeam) {
      buffer = malloc(sizeof(double) * nb);
      MPI_Allreduce(Itime, buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
      memcpy(Itime, buffer, sizeof(double) * nb);
      free(buffer);
    }
#endif
    if (isSlave || !distributedBeam) {
      if (LSC->smoothing) {
        SavitzyGolaySmooth(Itime, nb, LSC->SGOrder, LSC->SGHalfWidth, LSC->SGHalfWidth, 0);
#if DEBUG
        printf("Smoothing completed\n");
        fflush(stdout);
#endif
      }
    }

    /* Compute kSC and length to drift */
    /* - find maximum current */
    find_min_max(&Imin, &Imax, Itime, nb);
#if USE_MPI
    if (distributedBeam) {
      if (isMaster) {
        Imin = DBL_MAX;
        Imax = -DBL_MAX;
      }
      find_global_min_max(&Imin, &Imax, nb, MPI_COMM_WORLD);
    }
#endif
#if DEBUG
    printf("Maximum particles/bin: %e    Q/MP: %e C    Imax: %e A\n",
           Imax, charge->macroParticleCharge, Imax * charge->macroParticleCharge / dt);
    fflush(stdout);
#endif
    Imax *= charge->macroParticleCharge / dt;
    /* - compute beam radius as the average rms beam size in x and y */
#if !USE_MPI
    rms_emittance(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL);
#else
    if (distributedBeam)
      rms_emittance_p(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL, NULL);
    else
      rms_emittance(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL);
#endif

    if ((beamRadius = (sqrt(S11) + sqrt(S33)) / 2 * LSC->radiusFactor) == 0) {
      printf("Error: beam radius is zero in LSCDRIFT: S11=%le, S33=%le, RADIUS_FACTOR=%le\n",
             S11, S33, LSC->radiusFactor);
      exitElegant(1);
    }
    /* - compute kSC */
    kSC = 2 / beamRadius * sqrt(Imax / ipow3(Po) / Ia);
    /* - compute length to drift */
    length = 0.1 / kSC;
    if (length > lengthLeft || kickMode)
      length = lengthLeft;
    /* - compute values for computing impedance */
    df = 1. / (dt * nb);
    dk = df * PIx2 / c_mks;

#if DEBUG
    printf("rb: %e m   LSC: I0=%e A    kSC=%e 1/m\nlength = %e m   dt = %e s    df = %e Hz   dk = %e 1/m\n",
           beamRadius, Imax, kSC, length, dt, df, dk);
    fflush(stdout);
#endif

    /* Take the FFT of I(t) to get I(f) */
    memcpy(Ifreq, Itime, 2 * nb * sizeof(*Ifreq));
    realFFT(Ifreq, nb, 0);
    nfreq = nb / 2 + 1;

    if (LSC->highFrequencyCutoff0 > 0) {
      /* apply low-pass filter */
      long i, i1, i2;
      double dfraction, fraction;
      i1 = LSC->highFrequencyCutoff0 * nfreq;
      if (i1 < 1)
        i1 = 1;
      i2 = LSC->highFrequencyCutoff1 * nfreq;
      if (i2 >= nfreq)
        i2 = nfreq - 1;
      dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
      fraction = 1;
      for (i = i1; i < i2; i++) {
        Ifreq[2 * i - 1] *= fraction;
        Ifreq[2 * i] *= fraction;
        if ((fraction -= dfraction) < 0)
          fraction = 0;
      }
      for (; i < nfreq - 1; i++) {
        Ifreq[2 * i - 1] = 0;
        Ifreq[2 * i] = 0;
      }
      /* kill the Nyquist term */
      Ifreq[nb - 1] = 0;
    }

    if (LSC->lowFrequencyCutoff0 >= 0) {
      /* apply high-pass filter */
      long i, i1, i2;
      double dfraction, fraction;
      i1 = LSC->lowFrequencyCutoff0 * nfreq;
      if (i1 < 1)
        i1 = 1;
      if (i1 >= nfreq)
        i1 = nfreq - 1;
      i2 = LSC->lowFrequencyCutoff1 * nfreq;
      if (i2 < i1)
        i2 = i1;
      if (i2 >= nfreq)
        i2 = nfreq - 1;
      dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
      fraction = 0;
      Ifreq[0] = 0;
      for (i = 1; i < i1; i++) {
        Ifreq[2 * i - 1] = 0;
        Ifreq[2 * i] = 0;
      }
      for (i = i1; i < i2; i++) {
        Ifreq[2 * i - 1] *= fraction;
        Ifreq[2 * i] *= fraction;
        if ((fraction += dfraction) > 1)
          fraction = 1;
      }
    }

    /* Compute V(f) = Z(f)*I(f), putting in a factor 
     * to normalize the current waveform.
     */
    Vfreq = Vtime;

    /* impedance is zero at DC */
    Vfreq[0] = 0;

    /* Nyquist term for current histogram is pure imaginary.
     * Since impedance is also imaginary, the product is pure real.
     * Hence, the Nyquist term we store is zero.
     */
    if (nb % 2 == 0)
      Vfreq[nb - 1] = 0;

    factor = charge->macroParticleCharge / dt;
    if (LSC->backtrack)
      factor *= -1;
    a2 = Z0 / (PI * sqr(beamRadius)) * length;
#if DEBUG
    printf("nfreq = %ld   a2 = %e Ohms/m\n", nfreq, a2);
    fflush(stdout);

    if (!fpd) {
      fpd = fopen_e("lscZ.sdds", "w", 0);
      fprintf(fpd, "SDDS1\n&column name=k type=double units=m &end\n");
      fprintf(fpd, "&column name=ZImag, type=double, units=Ohms &end\n");
      fprintf(fpd, "&data no_row_counts=1 mode=ascii &end\n");
    } else
      fprintf(fpd, "\n");
#endif
    Zmax = 0;
    if (isSlave || !distributedBeam) {
      for (ib = 1; ib < nfreq - 1; ib++) {
        k = ib * dk;
        a1 = k * beamRadius / Po;
        ZImag = a2 / k * (1 - a1 * dbesk1(a1));
#if DEBUG
        fprintf(fpd, "%e %e\n", k, ZImag);
#endif
        if (ZImag > Zmax)
          Zmax = ZImag;
        iImag = (iReal = 2 * ib - 1) + 1;
        /* There is a minus sign here because I use t<0 for the head */
        Vfreq[iReal] = Ifreq[iImag] * ZImag * factor;
        Vfreq[iImag] = -Ifreq[iReal] * ZImag * factor;
      }
    }
#if DEBUG
    printf("Maximum |Z| = %e Ohm\n", Zmax);
    fflush(stdout);
#endif

    /* Compute inverse FFT of V(f) to get V(t) */
    realFFT(Vfreq, nb, INVERSE_FFT);
    Vtime = Vfreq;

    /* put zero voltage in Vtime[nb] for use in interpolation */
    Vtime[nb] = 0;
    if (isSlave || !distributedBeam) {
      applyLongitudinalWakeKicks(part, time, pbin, np, Po, Vtime,
                                 nb, tmin, dt, LSC->interpolate);
      if (!kickMode) {
        /* advance particles to the next step */
        for (ib = 0; ib < np; ib++) {
          part[ib][4] += length * sqrt(1 + sqr(part[ib][1]) + sqr(part[ib][3])) * (LSC->backtrack ? -1 : 1);
          part[ib][0] += length * part[ib][1] * (LSC->backtrack ? -1 : 1);
          part[ib][2] += length * part[ib][3] * (LSC->backtrack ? -1 : 1);
        }
      }
    }
    lengthLeft -= length;
  }

#if DEBUG
  if (fpd)
    fclose(fpd);
#endif

#if defined(MINIMIZE_MEMORY)
  tfree(Itime);
  tfree(Vtime);
  if (pbin)
    tfree(pbin);
  if (time)
    tfree(time);
  Itime = Vtime = time = NULL;
  pbin = NULL;
  max_np = max_n_bins = 0;
#endif
}

void addLSCKick(double **part, int64_t np, LSCKICK *LSC, double Po, CHARGE *charge,
                double lengthScale, double dgammaOverGamma) {
  static double *Itime = NULL; /* array for histogram of particle density */
  static double *Ifreq = NULL; /* array for FFT of histogram of particle density */
  static double *Vtime = NULL; /* array for voltage acting on each bin */
  static long max_n_bins = 0;
  static long *pbin = NULL;   /* array to record which bin each particle is in */
  static double *time = NULL; /* array to record arrival time of each particle */
  static long max_np = 0;
  double *Vfreq, ZImag;
  long ib, nb, n_binned, nfreq, iReal, iImag;
  double factor, tmin, tmax, dt, df, dk, a1, a2;
  double Imin, Imax, kSC, Zmax;
  double Ia = 17045, Z0, length, k;
  double S11, S33, beamRadius;

  if (!charge)
    bombElegant("No charge defined for LSC.  Insert a CHARGE element in the beamline.", NULL);

  Z0 = sqrt(mu_o / epsilon_o);
  nb = LSC->bins;
  if (nb % 2 == 1) {
    printf("Error: LSC must have an even number of bins\n");
    exitElegant(1);
  }

#if DEBUG
  printf("%ld bins for LSC\n", nb);
  fflush(stdout);
#endif

  if (nb > max_n_bins) {
    max_n_bins = nb;
    Itime = trealloc(Itime, 2 * sizeof(*Itime) * (max_n_bins + 1));
    Ifreq = trealloc(Ifreq, 2 * sizeof(*Ifreq) * (max_n_bins + 1));
    Vtime = trealloc(Vtime, 2 * sizeof(*Vtime) * (max_n_bins + 1));
  }

  if (np > max_np) {
    pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
    time = trealloc(time, sizeof(*time) * max_np);
  }

  /* compute time coordinates and make histogram */
  computeTimeCoordinatesOnly(time, Po, part, np);
  find_min_max(&tmin, &tmax, time, np);
#if USE_MPI
  if (isSlave && distributedBeam)
    find_global_min_max(&tmin, &tmax, nb, workers);
#endif
  dt = (tmax - tmin) / (nb - 3);
#if DEBUG
  printf("tmin=%e, tmax=%e, dt=%e\n",
         tmin, tmax, dt);
  fflush(stdout);
#endif
  n_binned = binTimeDistribution(Itime, pbin, tmin, dt, nb, time, part, Po, np);
#if DEBUG
  printf("%ld of %" PRId64 " particles binned\n", n_binned, np);
  fflush(stdout);
#endif
  if (n_binned != np && !USE_MPI) { /* This will not be checked in Pelegant to avoid communications */
    char warningText[1024];
    snprintf(warningText, 1024,
             "Only %ld of %" PRId64 " particles were binned, which shouldn't happen.", n_binned, np);
    printWarningForTracking("Some particles were not binned in LSCKICK.",
                            warningText);
  }

  /* Compute kSC and length to drift */
  /* - compute values involved in binning and FFTs */
  df = 1. / (dt * nb);
  dk = df * PIx2 / c_mks;
  /* - find maximum current */
  find_min_max(&Imin, &Imax, Itime, nb);
#if USE_MPI
  if (isSlave && distributedBeam) {
    double *buffer;
    find_global_min_max(&tmin, &tmax, np, workers);
    buffer = malloc(sizeof(double) * nb);
    MPI_Allreduce(Itime, buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
    memcpy(Itime, buffer, sizeof(double) * nb);
    tfree(buffer);
    buffer = NULL;
  }
#endif
#if DEBUG
  printf("Maximum particles/bin: %e    Q/MP: %e C    Imax: %e A\n",
         Imax, charge->macroParticleCharge, Imax * charge->macroParticleCharge / dt);
  fflush(stdout);
#endif
  Imax *= charge->macroParticleCharge / dt;
  /* - compute beam radius as the average rms beam size in x and y */
#if !USE_MPI
  rms_emittance(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL);
#else
  if (distributedBeam)
    rms_emittance_p(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL, NULL);
  else
    rms_emittance(part, 0, 2, np, &S11, NULL, &S33, NULL, NULL);
#endif
  if ((beamRadius = (sqrt(S11) + sqrt(S33)) / 2 * LSC->radiusFactor) == 0) {
    printf("Error: beam radius is zero in LSCKICK\n");
    exitElegant(1);
  }
  /* - compute kSC */
  kSC = 2 / beamRadius * sqrt(Imax / ipow3(Po) / Ia);

  /* - compute maximum length that we should be traveling between kicks */
  lengthScale = fabs(lengthScale);
#if DEBUG
  printf("rb=%e m   I0=%e A    kSC=%e 1/m    dt=%e s    df=%e Hz   dk=%e 1/m\n",
         beamRadius, Imax, kSC, dt, df, dk);
  printf("lengthScale=%e m   dgamma/gamma=%e\n", lengthScale, dgammaOverGamma);
  fflush(stdout);
#endif
  length = 1 / kSC;
  if (isSlave || !distributedBeam) {
    if (dgammaOverGamma) {
      double length2;
      length2 = fabs(lengthScale / dgammaOverGamma);
      if (length2 < length)
        length = length2;
    }
    length /= 10;

    if (length < lengthScale) {
      /* length scale used by calling routine is too large, so refuse to do it */
      TRACKING_CONTEXT context;
      getTrackingContext(&context);
#if USE_MPI
      if (myid == 1)
        dup2(fdStdout, fileno(stdout)); /* Let the first slave processor write the output */
#endif
      printf("Error: distance between LSC kicks for %s at z=%e is too large.\n",
             context.elementName, context.zStart);
      printf("Suggest reducing distance between kicks by factor %e\n",
             lengthScale / length);
#if USE_MPI
      MPI_Abort(workers, 1);
#else
      exitElegant(1);
#endif
    }
  }
  /* Take the FFT of I(t) to get I(f) */
  memcpy(Ifreq, Itime, 2 * nb * sizeof(*Ifreq));
  realFFT(Ifreq, nb, 0);
  nfreq = nb / 2 + 1;

  if (LSC->highFrequencyCutoff0 > 0) {
    /* apply low-pass filter */
    long i, i1, i2;
    double dfraction, fraction;
    i1 = LSC->highFrequencyCutoff0 * nfreq;
    if (i1 < 1)
      i1 = 1;
    i2 = LSC->highFrequencyCutoff1 * nfreq;
    if (i2 >= nfreq)
      i2 = nfreq - 1;
    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 1;
    for (i = i1; i < i2; i++) {
      Ifreq[2 * i - 1] *= fraction;
      Ifreq[2 * i] *= fraction;
      if ((fraction -= dfraction) < 0)
        fraction = 0;
    }
    for (; i < nfreq - 1; i++) {
      Ifreq[2 * i - 1] = 0;
      Ifreq[2 * i] = 0;
    }
    /* kill the Nyquist term */
    Ifreq[nb - 1] = 0;
  }

  if (LSC->lowFrequencyCutoff0 >= 0) {
    /* apply high-pass filter */
    long i, i1, i2;
    double dfraction, fraction;
    i1 = LSC->lowFrequencyCutoff0 * nfreq;
    if (i1 < 1)
      i1 = 1;
    if (i1 >= nfreq)
      i1 = nfreq - 1;
    i2 = LSC->lowFrequencyCutoff1 * nfreq;
    if (i2 < i1)
      i2 = i1;
    if (i2 >= nfreq)
      i2 = nfreq - 1;
    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 0;
    Ifreq[0] = 0;
    for (i = 1; i < i1; i++) {
      Ifreq[2 * i - 1] = 0;
      Ifreq[2 * i] = 0;
    }
    for (i = i1; i < i2; i++) {
      Ifreq[2 * i - 1] *= fraction;
      Ifreq[2 * i] *= fraction;
      if ((fraction += dfraction) > 1)
        fraction = 1;
    }
  }

  /* Compute V(f) = Z(f)*I(f), putting in a factor 
   * to normalize the current waveform.
   */
  Vfreq = Vtime;

  /* impedance is zero at DC */
  Vfreq[0] = 0;

  /* Nyquist term for current histogram is pure imaginary.
   * Since impedance is also imaginary, the product is pure real.
   * Hence, the Nyquist term we store is zero.
   */
  if (nb % 2 == 0)
    Vfreq[nb - 1] = 0;

  factor = charge->macroParticleCharge / dt;
  if (LSC->backtrack)
    factor *= -1;
  a2 = Z0 / (PI * sqr(beamRadius)) * lengthScale;
#if DEBUG
  printf("nfreq = %ld   a2 = %e Ohms/m  factor = %le\n", nfreq, a2, factor);
  fflush(stdout);
#endif
  Zmax = 0;
  for (ib = 1; ib < nfreq - 1; ib++) {
    k = ib * dk;
    a1 = k * beamRadius / Po;
    ZImag = a2 / k * (1 - a1 * dbesk1(a1));
    if (ZImag > Zmax)
      Zmax = ZImag;
    iImag = (iReal = 2 * ib - 1) + 1;
    /* There is a minus sign here because I use t<0 for the head */
    Vfreq[iReal] = Ifreq[iImag] * ZImag * factor;
    Vfreq[iImag] = -Ifreq[iReal] * ZImag * factor;
  }
#if DEBUG
  printf("Maximum |Z| = %e Ohm\n", Zmax);
  fflush(stdout);
#endif

  /* Compute inverse FFT of V(f) to get V(t) */
  realFFT(Vfreq, nb, INVERSE_FFT);
  Vtime = Vfreq;

  /* put zero voltage in Vtime[nb] for use in interpolation */
  Vtime[nb] = 0;
  applyLongitudinalWakeKicks(part, time, pbin, np, Po, Vtime,
                             nb, tmin, dt, LSC->interpolate);

#if defined(MINIMIZE_MEMORY)
  tfree(Itime);
  tfree(Vtime);
  tfree(pbin);
  tfree(time);
  Itime = Vtime = time = NULL;
  pbin = NULL;
  max_np = max_n_bins = 0;
#endif
}
