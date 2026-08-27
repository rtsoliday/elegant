/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: simple_rfca.c
 * contents: simple_rf_cavity()
 *
 * Michael Borland, 1991
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu_simple_rfca.h"
#endif

void add_to_particle_energy2(double *coord, double timeOfFlight, double Po, double dgamma, double dpx, double dpy);

static char *fiducialModeChoice[4] = {
  "light",
  "tmean",
  "first",
  "pmaximum",
};

long trackRfCavityWithWakes(double **part, int64_t np, RFCA *rfca, double **accepted,
                            double *P_central, double zEnd, long iPass, RUN *run,
                            CHARGE *charge, WAKE *wake, TRWAKE *trwake, LSCKICK *LSCKick,
                            long wakesAtEnd);

unsigned long parseFiducialMode(char *modeString) {
  long code;

  if (!modeString)
    return FID_MODE_TMEAN;
  switch (code = match_string(modeString, fiducialModeChoice, 4, 0)) {
  case 0:
  case 1:
  case 2:
  case 3:
    return FID_MODE_LIGHT << code;
  default:
    return 0;
  }
}

static long fiducializationBunch = -1;
static int64_t idSlotsPerBunch = -1;
void setFiducializationBunch(long b, int64_t n) {
  fiducializationBunch = b;
  idSlotsPerBunch = n;
#if USE_MPI && MPI_DEBUG
  printf("Fiducialization will use bunch %ld, idSlotsPerBunch=%" PRId64 "\n",
         fiducializationBunch, idSlotsPerBunch);
  fflush(stdout);
#endif
}

/* --- tracking-clock register --------------------------------------------
 * Macro-scale absolute time (seconds) that is deliberately NOT carried in
 * part[][4] (=beta*c*t), so intra-bunch precision is preserved even in
 * long-timescale (multi-step / many-turn) simulations.  Time-dependent
 * consumers and time outputs reconstruct the true absolute time as
 *     t_true = part[][4]/(c*beta) + trackingClockOffset().
 *   base  : per-step macro offset (P1, set from run_control step_frequency);
 *           set once per step and persists.
 *   accum : per-turn CHANGE_T removals (P2); reset at the top of each
 *           do_tracking() call.
 * The base/accum split keeps the per-step base (set in track_beam before
 * do_tracking) safe from the accum reset performed inside do_tracking.
 */
static double trackingClockBase_s = 0.0;
static double trackingClockAccum_s = 0.0;   /* plain running sum -> trackingClockOffset() */
static double trackingClockAccumC_s = 0.0;  /* Neumaier compensation (low word) -> trackingClockPhase() only */

void setTrackingClockBase(double t_s) {
  trackingClockBase_s = t_s;
}

void resetTrackingClock(void) {
  trackingClockAccum_s = 0.0;
  trackingClockAccumC_s = 0.0;
}

void accumulateTrackingClock(double dt_s) {
  /* The plain sum trackingClockAccum_s drives trackingClockOffset() and is
   * bit-for-bit what it was before this change, so the linear-difference
   * consumers (HOM damping, ionEffects drift interval) are unaffected.  We
   * additionally carry a Neumaier compensation word so that after the sum
   * grows to many seconds the low-order bits it would otherwise shed are
   * still available to trackingClockPhase() -- that is what lets a
   * non-harmonic HOM recover an accurate fractional phase (see below). */
  double t = trackingClockAccum_s + dt_s;
  if (fabs(trackingClockAccum_s) >= fabs(dt_s))
    trackingClockAccumC_s += (trackingClockAccum_s - t) + dt_s;
  else
    trackingClockAccumC_s += (dt_s - t) + trackingClockAccum_s;
  trackingClockAccum_s = t;
}

double trackingClockOffset(void) {
  return trackingClockBase_s + trackingClockAccum_s;
}

/* Reduced clock phase for a time-dependent consumer of angular frequency
 * omega: returns (omega * trueOffset) wrapped into [0, 2*PI).  CHANGE_T
 * subtracts a whole number of RF fundamental periods from part[][4] each
 * turn and (Bundle P2) accumulates the removed time here; a consumer whose
 * frequency is NOT an integer harmonic of that fundamental cannot simply
 * drop the offset from its phase advance.  Instead it adds the difference
 * of this quantity across the gap.  The offset is reduced modulo the
 * consumer's own period 2*PI/omega *in the time domain first* (fmodl is an
 * exact remainder), so the large integer cycle count is removed before
 * scaling and the small fractional phase keeps full precision even when the
 * offset has grown to many seconds.  long double is used only for this
 * stateless reduction, never for the accumulated state, so it has no effect
 * on cross-rank determinism (the state is portable double arithmetic). */
double trackingClockPhase(double omega) {
  long double off, period, resid;
  if (omega == 0)
    return 0.0;
  off = (long double)trackingClockBase_s +
        ((long double)trackingClockAccum_s + (long double)trackingClockAccumC_s);
  period = (long double)PIx2 / (long double)omega;
  resid = fmodl(off, period);
  if (resid < 0)
    resid += period;
  return (double)((long double)omega * resid);
}

long getFiducializationPidRange(unsigned long mode, int64_t *startPID,
                                int64_t *endPID) {
  if (!startPID || !endPID)
    return 0;
  if (fiducializationBunch < 0 || idSlotsPerBunch <= 0 ||
      mode & FID_MODE_FULLBEAM) {
    *startPID = *endPID = -1;
  } else {
    *startPID = fiducializationBunch * idSlotsPerBunch + 1;
    *endPID = *startPID + idSlotsPerBunch - 1;
  }
  return 1;
}

double findFiducialTime(double **part, int64_t np, double s0, double sOffset,
                        double p0, unsigned long mode) {
  double tFid = 0.0;
  int64_t i;
  int64_t startPID, endPID;

  getFiducializationPidRange(mode, &startPID, &endPID);
#if USE_MPI
  mpiAbort = 0;
#  if MPI_DEBUG
  printf("startPID = %" PRId64 ", endPID = %" PRId64 "\n", startPID, endPID);
#  endif
#endif

  /* TEMPORARILY not using the GPU code until it can be updated to match the non-GPU code */
  /*
#ifdef HAVE_GPU
  if(getElementOnGpu()){
    startGpuTimer();
    tFid = gpu_findFiducialTime(np, s0, sOffset, p0, mode);
#ifdef GPU_VERIFY     
    startCpuTimer();
    findFiducialTime(part, np, s0, sOffset, p0, mode);
    compareGpuCpu(np, "findFiducialTime");
#endif
    return tFid;
  }
#endif
  */

  if (mode & FID_MODE_LIGHT) {
    printf("Performing \"light\" mode fiducialization with s0=%21.15e, sOffset=%21.15e\n",
           s0, sOffset);
    tFid = (s0 + sOffset) / c_mks;
  } else if (mode & FID_MODE_FIRST) {
#if (!USE_MPI)
    long found = 0;
    if (np) {
      for (i = 0; i < np; i++) {
        if ((startPID < 0 && endPID < 0) || (part[i][6] >= startPID && part[i][6] <= endPID)) {
          tFid = (part[i][4] + sOffset) / (c_mks * beta_from_delta(p0, part[i][5]));
          found = 1;
          break;
        }
      }
    }
    if (!found)
      bombElegant("No available particle for the FID_MODE_FIRST mode in findFiducialTime", NULL);
#else
    if (myid == 1) { /* If the first particle is lost, Pelegant and elegant will not have the same fiducial time */
      long found = 0;
      if (np) {
        for (i = 0; i < np; i++) {
          if ((startPID < 0 && endPID < 0) || (part[i][6] >= startPID && part[i][6] <= endPID)) {
            tFid = (part[i][4] + sOffset) / (c_mks * beta_from_delta(p0, part[i][5]));
            found = 1;
            break;
          }
        }
      }
      if (!found)
        mpiAbort = MPI_ABORT_RF_FIDUCIALIZATION_ERROR;
    }
    MPI_Bcast(&tFid, 1, MPI_DOUBLE, 1, MPI_COMM_WORLD);
#endif
  } else if (mode & FID_MODE_PMAX) {
    long ibest;
    int64_t i;
    double best;
#if (!USE_MPI)
    best = part[0][5];
#else /* np could be 0 for some of the slave processors */
    best = -DBL_MAX;
#endif
    ibest = -1;
    if (isSlave || !distributedBeam)
      for (i = 0; i < np; i++) {
        if (((startPID < 0 && endPID < 0) || (part[i][6] >= startPID && part[i][6] <= endPID)) && best < part[i][5]) {
          best = part[i][5];
          ibest = i;
        }
      }
#if (!USE_MPI)
    if (ibest >= 0 && ibest < np)
      tFid = (part[ibest][4] + sOffset) / (c_mks * beta_from_delta(p0, part[ibest][5]));
    else
      bombElegant("No available particle for RF cavity fiducialization", NULL);
#else
    if (distributedBeam) {
      double sBest;
      struct {
        double val;
        int rank;
      } in, out;
      MPI_Comm_rank(MPI_COMM_WORLD, &(in.rank));
      in.val = best;
      /* find the global best value and its location, i.e., on which processor */
      MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
      best = out.val;
      /* broadcast the part[ibest][4] corresponding to the global best value */
      if (in.rank == out.rank) {
        if (ibest >= 0 && ibest < np)
          sBest = part[ibest][4];
        else {
          sBest = -DBL_MAX;
          mpiAbort = MPI_ABORT_RF_FIDUCIALIZATION_ERROR;
        }
      }
      MPI_Bcast(&sBest, 1, MPI_DOUBLE, out.rank, MPI_COMM_WORLD);
      tFid = (sBest + sOffset) / (c_mks * beta_from_delta(p0, best));
    } else {
      if (ibest >= 0 && ibest < np)
        tFid = (part[ibest][4] + sOffset) / (c_mks * beta_from_delta(p0, part[ibest][5]));
      else {
        tFid = -DBL_MAX;
        mpiAbort = MPI_ABORT_RF_FIDUCIALIZATION_ERROR;
      }
    }
#endif
  } else if (mode & FID_MODE_TMEAN) {
    double tsum = 0;
    int64_t ip, nsum = 0;
#ifdef USE_KAHAN
    double error = 0.0;
#endif
#if USE_MPI && MPI_DEBUG
    printf("fiducializing using TMEAN, isSlave=%d, distributedBeam=%d\n", isSlave, distributedBeam);
    fflush(stdout);
#endif

    if (isSlave || !distributedBeam) {
#if USE_MPI && MPI_DEBUG
      printf("doing fiducilization sum for %ld particles\n", np);
      fflush(stdout);
#endif
      for (ip = tsum = 0; ip < np; ip++) {
        if ((startPID < 0 && endPID < 0) || (part[ip][6] >= startPID && part[ip][6] <= endPID)) {
#ifndef USE_KAHAN
          tsum += (part[ip][4] + sOffset) / (c_mks * beta_from_delta(p0, part[ip][5]));
#else
          tsum = KahanPlus(tsum, (part[ip][4] + sOffset) / (c_mks * beta_from_delta(p0, part[ip][5])), &error);
#endif
          nsum++;
        }
      }
    }
#if (!USE_MPI)
    if (nsum > 0)
      tFid = tsum / nsum;
    else
      bombElegant("No available particle for RF cavity fiducilization", NULL);
#else
#  if MPI_DEBUG
    printf("tsum = %le, nsum = %ld\n", tsum, nsum);
    fflush(stdout);
#  endif
    if (distributedBeam) {
      if (USE_MPI) {
        double tsum_total;
        int64_t nsum_total;

        if (isMaster) {
          tsum = 0.0;
          nsum = 0;
        }
        MPI_Allreduce(&nsum, &nsum_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#  ifndef USE_KAHAN
        MPI_Allreduce(&tsum, &tsum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
        tsum_total = KahanParallel(tsum, error, MPI_COMM_WORLD);
#  endif
#  if MPI_DEBUG
        printf("tsum total = %le, nsum total = %ld\n", tsum_total, nsum_total);
        fflush(stdout);
#  endif
        if (nsum_total > 0)
          tFid = tsum_total / nsum_total;
        else {
          tFid = -DBL_MAX;
          mpiAbort = MPI_ABORT_RF_FIDUCIALIZATION_ERROR;
        }
      }
    } else {
#  if MPI_DEBUG
      printf("Found %ld particles for fiducialization\n", np);
      fflush(stdout);
#  endif
      if (np > 0)
        tFid = tsum / np;
      else {
        tFid = -DBL_MAX;
        mpiAbort = MPI_ABORT_RF_FIDUCIALIZATION_ERROR;
      }
    }
#endif
  } else
    bombElegant("invalid fiducial mode in findFiducialTime", NULL);

#ifdef DEBUG
  printf("Fiducial time (mode %lx): %21.15e\n", mode, tFid);
#endif
  return tFid;
}

long simple_rf_cavity(
  double **part, int64_t np, RFCA *rfca, double **accepted, double *P_central, double zEnd) {

#ifdef HAVE_GPU
  int64_t nLeft;
#  ifdef GPU_VERIFY
  double P_central_input = *P_central;
#  endif

  if (getElementOnGpu()) {
    startGpuTimer();
    nLeft = gpu_trackRfCavityWithWakes(np, rfca, accepted, P_central, zEnd, 0,
                                       NULL, NULL, NULL, NULL, NULL, 0);
#  ifdef GPU_VERIFY
    if (getElementOnGpu()) {
      startCpuTimer();
      trackRfCavityWithWakes(part, np, rfca, accepted, &P_central_input, zEnd, 0,
                             NULL, NULL, NULL, NULL, NULL, 0);
      compareGpuCpu(np, "simple_rf_cavity");
      if (*P_central / P_central_input - 1 > 1e-12) {
        printf("simple_rf_cavity: Warning: GPU P_central=%le vs CPU=%le\n",
               *P_central, P_central_input);
      }
    }
#  endif
    return nLeft;
  }
#endif

  return trackRfCavityWithWakes(part, np, rfca, accepted, P_central, zEnd, 0,
                                NULL, NULL, NULL, NULL, NULL, 0);
}

static void trackRfCavityAddLSCKick(double **part, int64_t np,
                                    LSCKICK *LSCKick, double Po,
                                    CHARGE *charge, double length,
                                    double dgammaOverGammaAve) {
#ifdef HAVE_GPU
  if (gpu_track_rfcw_lsc_kick_only(part, np, LSCKick, Po, charge,
                                   length, dgammaOverGammaAve))
    return;
#endif
  addLSCKick(part, np, LSCKick, Po, charge, length, dgammaOverGammaAve);
}

long trackRfCavityWithWakes(
  double **part, int64_t np,
  RFCA *rfca,
  double **accepted,
  double *P_central, double zEnd,
  long iPass,
  RUN *run,
  CHARGE *charge,
  WAKE *wake,
  TRWAKE *trwake,
  LSCKICK *LSCKick,
  long wakesAtEnd) {
  int64_t ip;
  long same_dgamma, nKicks, linearize, ik;
  double timeOffset, dc4, x, xp;
  double P, gamma, gamma1, dgamma = 0.0, dgammaMax = 0.0, phase, length, dtLight, volt, To, dpr;
  double *coord, t, t0, omega, beta_i, tau, dt, tAve = 0, dgammaAve = 0;
  short useSRSModel = 0, twFocusing1 = 0, matrixMethod;
  double dgammaOverGammaAve = 0;
  long dgammaOverGammaNp = 0;
  long lockPhase = 0;
#ifdef USE_KAHAN
  double error = 0.0;
#endif
#if USE_MPI
  long np_total, np_tmp;
  long i;
  double error_sum = 0.0, *sumArray, *errorArray;
#endif

#ifdef DEBUG
  printf("Tracking through rf cavity with V=%le, Phase=%le\n",
         rfca->volt, rfca->phase);
  fflush(stdout);
#endif

  matrixMethod = 0;
  identifyRfcaBodyFocusModel(rfca, T_RFCA, &matrixMethod, &useSRSModel, &twFocusing1);

  if (rfca->freq < 1e3 && rfca->freq)
    printWarningForTracking("RFCA frequency is less than 1kHz.", "This may be an error. Consult manual for units.");
  if (fabs(rfca->volt) < 100 && rfca->volt)
    printWarningForTracking("RFCA voltage is less than 100V.", "This may be an error. Consult manual for units.");
  if (isSlave) {
    if (!part)
      bombElegant("NULL particle data pointer (trackRfCavityWithWakes)", NULL);
  }
  if (isSlave || !distributedBeam) {
    for (ip = 0; ip < np; ip++)
      if (!part[ip]) {
        fprintf(stderr, "NULL pointer for particle %" PRId64 " (trackRfCavityWithWakes)\n", ip);
        fflush(stderr);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        abort();
      }
  }
  if (!rfca)
    bombElegant("NULL rfca pointer (trackRfCavityWithWakes)", NULL);

#if (!USE_MPI)
  if (np <= 0) {
    log_exit("trackRfCavityWithWakes");
    return (np);
  }
#else
  if (distributedBeam) {
    if (isMaster)
      np_tmp = 0;
    else
      np_tmp = np;
    MPI_Allreduce(&np_tmp, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (np_total <= 0) {
      log_exit("trackRfCavityWithWakes");
      return (np_total);
    }
  } else if (np <= 0) {
    log_exit("trackRfCavityWithWakes");
    return (np);
  }
#endif
  if (rfca->change_t && rfca->Q)
    bombElegant("incompatible RF cavity parameters: change_t!=0 && Q!=0", NULL);

  length = rfca->length;

  if (rfca->volt == 0 && !wake && !trwake && !LSCKick) {
    if (rfca->length) {
      if (isSlave || !distributedBeam) {
        for (ip = 0; ip < np; ip++) {
          coord = part[ip];
          coord[0] += coord[1] * length;
          coord[2] += coord[3] * length;
          coord[4] += length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
        }
      }
    }
    log_exit("trackRfCavityWithWakes");
    return (np);
  }

  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  if (omega)
    tau = rfca->Q / omega;
  else
    tau = 0;

  nKicks = length ? rfca->nKicks : 1;
  if (nKicks <= 0) {
    matrixMethod = 1;
    nKicks = 1;
  }
  length /= nKicks;
  volt /= nKicks;
  dtLight = length / c_mks;

  if (rfca->phase_reference == 0) {
    rfca->phase_reference = unused_phase_reference();
#if defined(DEBUG)
    printf("RFCA assigned to phase reference %ld\n", rfca->phase_reference);
#endif
  }

  switch (get_phase_reference(&phase, rfca->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!rfca->fiducial_seen) {
      unsigned long mode;
      if (!(mode = parseFiducialMode(rfca->fiducial)))
        bombElegant("invalid fiducial mode for RFCA element", NULL);
      if (rfca->tReference != -1)
        t0 = rfca->tReference;
      else {
        if (!rfca->backtrack || matrixMethod)
          t0 = findFiducialTime(part, np, zEnd - rfca->length, length / 2, *P_central, mode);
        else {
          double beta, P;
          long ik0;
          /* t0 = findFiducialTime(part, np, zEnd, -(fabs(rfca->length)-fabs(length/2)), *P_central, mode); */
          t0 = findFiducialTime(part, np, 0.0, 0.0, *P_central, mode);
#if !USE_MPI && defined(DEBUG)
          printf("t0 = %21.15e\n", t0);
#endif
          P = *P_central;
          beta = beta_from_delta(P, 0.0);
          for (ik0 = 0; ik0 < nKicks; ik0++) {
            t0 -= fabs(length / 2) / (beta * c_mks);
            P += volt * sin(rfca->phase * PI / 180);
            beta = beta_from_delta(P, 0.0);
            if (ik0 != (nKicks - 1))
              t0 -= fabs(length / 2) / (beta * c_mks);
          }
#if !USE_MPI && defined(DEBUG)
          printf("t0 = %21.15e, P = %21.15e\n", t0, P);
#endif
        }
      }
#if !USE_MPI && defined(DEBUG)
      printf("t0 = %21.15e, part[0][4] = %21.15e, part[0][5] = %21.15e, P0=%21.15e, dP/P=%le\n",
             t0, part[0][4], part[0][5], *P_central, volt * sin(rfca->phase * PI / 180));
      fflush(stdout);
#endif
      rfca->phase_fiducial = -omega * t0;
      rfca->fiducial_seen = 1;
    }
    set_phase_reference(rfca->phase_reference, phase = rfca->phase_fiducial);
#if defined(DEBUG)
    printf("RFCA fiducial phase is %e\n", phase);
#endif
    break;
  default:
    bombElegant("unknown return value from get_phase_reference()", NULL);
    break;
  }

  if (omega) {
    t0 = -rfca->phase_fiducial / omega;
    rfca->t_fiducial = t0; /* in case the user asks us to save it */
    To = PIx2 / omega;
  } else
    t0 = To = 0;
  phase += rfca->phase * PI / 180.0;

  same_dgamma = 0;
  if (omega == 0 && tau == 0) {
    dgamma = volt * sin(phase);
    dgammaMax = volt;
    same_dgamma = 1;
  }

  timeOffset = 0;
  if (omega && rfca->change_t) {
    /* Bundle P2: derive the whole-RF-period removal from an MPI-consistent, first-bunch
     * reference (findFiducialTime) so every rank removes the SAME integer of periods, and
     * ACCUMULATE the removed time into the tracking-clock register instead of discarding it.
     * Non-harmonic time-dependent elements (RFMODE HOM damping, ionEffects, ...) then see the
     * true multi-turn elapsed time via trackingClockOffset().  This operates on the reduced
     * coord[4] (as the old part[0] reference did); do NOT add trackingClockOffset() here --
     * this is the producer that drives it.  findFiducialTime runs MPI collectives, so it must
     * be called by all ranks (incl. the master, which holds np==0 and contributes zero); the
     * old "isSlave || !distributedBeam" guard is therefore removed.  In the serial (non-
     * distributed) case an empty beam is skipped to preserve the historical no-op. */
    unsigned long changeTFidMode = parseFiducialMode(rfca->fiducial);
    double tRef;
    if (distributedBeam || np != 0) {
      tRef = findFiducialTime(part, np, 0.0, 0.0, *P_central, changeTFidMode);
      if (tRef > (0.9 * To)) {
        timeOffset = ((long)(tRef / To + 0.5)) * To;
        accumulateTrackingClock(timeOffset);
      }
    }
  }

  linearize = rfca->linearize;
  lockPhase = rfca->lockPhase;

  if (linearize || lockPhase) {
    tAve = 0;
    if (nKicks != 1)
      bombElegant("Must use n_kicks=1 for linearized rf cavity", NULL);

    if (isSlave || !distributedBeam) {
      for (ip = 0; ip < np; ip++) {
        coord = part[ip];
        P = *P_central * (1 + coord[5]);
        beta_i = P / (gamma = sqrt(sqr(P) + 1));
        t = (coord[4] + length / 2) / (c_mks * beta_i) - timeOffset;
#ifndef USE_KAHAN
        tAve += t;
#else
        tAve = KahanPlus(tAve, t, &error);
#endif
      }
    }
#if (!USE_MPI)
    tAve /= np;
#else
    if (distributedBeam) {
      if (USE_MPI) {
        double tAve_total = 0.0;
#  ifndef USE_KAHAN
        MPI_Allreduce(&tAve, &tAve_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tAve = tAve_total / np_total;
#  else
        sumArray = malloc(sizeof(double) * n_processors);
        errorArray = malloc(sizeof(double) * n_processors);

        MPI_Allgather(&tAve, 1, MPI_DOUBLE, sumArray, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        /* collect errors from all processors */
        MPI_Allgather(&error, 1, MPI_DOUBLE, errorArray, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (i = 1; i < n_processors; i++) {
          tAve_total = KahanPlus(tAve_total, sumArray[i], &error_sum);
          tAve_total = KahanPlus(tAve_total, errorArray[i], &error_sum);
        }
        tAve = tAve_total / np_total;

        free(sumArray);
        free(errorArray);
#  endif
      }
    } else
      tAve /= np;
#endif
    if (lockPhase) {
      phase = PI / 180 * rfca->phase;
      dgammaAve = volt * sin(phase);
    } else
      dgammaAve = volt * sin(omega * (tAve - timeOffset) + phase);
  }
  if (isSlave || !distributedBeam) {
    for (ip = 0; ip < np; ip++) {
      coord = part[ip];
      coord[0] -= rfca->dx;
      coord[2] -= rfca->dy;
    }
  }

  if (!matrixMethod) {
    double *inverseF;
    inverseF = calloc(sizeof(*inverseF), np);

    if (!rfca->backtrack) {
      for (ik = 0; ik < nKicks; ik++) {
        dgammaOverGammaAve = dgammaOverGammaNp = 0;
        if (isSlave || !distributedBeam) {
          for (ip = 0; ip < np; ip++) {
            dpr = 0;
            coord = part[ip];
            if (coord[5] == -1)
              continue;
            if (length)
              /* compute distance traveled to center of this section */
              dc4 = length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            else
              dc4 = 0;

            /* compute energy kick */
            P = *P_central * (1 + coord[5]);
            beta_i = P / (gamma = sqrt(sqr(P) + 1));
            t = (coord[4] + dc4) / (c_mks * beta_i) - timeOffset;
            if (ik == 0 && timeOffset && rfca->change_t)
              coord[4] = t * c_mks * beta_i - dc4;
            if ((dt = t - t0) < 0)
              dt = 0;
            if (!same_dgamma) {
              if (!linearize) {
                if (rfca->standingWave)
                  dgamma = volt * sin(omega * (t - (lockPhase ? tAve : 0) ) + phase) * (tau ? sqrt(1 - exp(-dt / tau)) : 1);
                else {
                  double phi, dfactor;
                  phi = omega * (t - (lockPhase ? tAve : 0) - ik * dtLight) + phase;
                  dfactor = tau ? sqrt(1 - exp(-dt / tau)) : 1;
                  dgamma = volt * sin(phi) * dfactor;
                  if (twFocusing1)
                    dpr = volt/(2*beta_i)*(omega/c_mks)*(1-beta_i)*cos(phi);
                }
              } else
                dgamma = dgammaAve + volt * omega * (t - tAve) * cos(omega * (tAve - timeOffset) + phase);
            }
            if (gamma) {
              dgammaOverGammaNp++;
              dgammaOverGammaAve += dgamma / gamma;
            }

            if (length) {
              if (rfca->end1Focus && ik == 0) {
                /* apply focus kick */
                inverseF[ip] = dgamma / (2 * gamma * length);
                coord[1] -= coord[0] * inverseF[ip];
                coord[3] -= coord[2] * inverseF[ip];
              }
              /* apply initial drift */
              coord[0] += coord[1] * length / 2;
              coord[2] += coord[3] * length / 2;
              coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            }

            /* apply energy kick */
            if (dpr!=0)
              add_to_particle_energy2(coord, t, *P_central, dgamma, dpr*coord[0], dpr*coord[2]);
            else
              add_to_particle_energy(coord, t, *P_central, dgamma);
            if ((gamma1 = gamma + dgamma) <= 1)
              coord[5] = -1;
            else
              /* compute inverse focal length for exit kick */
              inverseF[ip] = -dgamma / (2 * gamma1 * length);
          }
        }
        if (!wakesAtEnd) {
          /* do wakes */
          if (wake)
            track_through_wake(part, np, wake, P_central, run, iPass, charge);
          if (trwake)
            track_through_trwake(part, np, trwake, *P_central, run, iPass, charge);
          if (LSCKick) {
#if !USE_MPI
            if (dgammaOverGammaNp)
              dgammaOverGammaAve /= dgammaOverGammaNp;
#else
            if (distributedBeam) {
              double t1 = dgammaOverGammaAve;
              long t2 = dgammaOverGammaNp;
              MPI_Allreduce(&t1, &dgammaOverGammaAve, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
              MPI_Allreduce(&t2, &dgammaOverGammaNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
              if (dgammaOverGammaNp)
                dgammaOverGammaAve /= dgammaOverGammaNp;
            } else if (dgammaOverGammaNp)
              dgammaOverGammaAve /= dgammaOverGammaNp;

#endif
            trackRfCavityAddLSCKick(part, np, LSCKick, *P_central, charge,
                                    length, dgammaOverGammaAve);
          }
        }
        if (length) {
          if (isSlave || !distributedBeam) {
            /* apply final drift and focus kick if needed */
            for (ip = 0; ip < np; ip++) {
              coord = part[ip];
              coord[0] += coord[1] * length / 2;
              coord[2] += coord[3] * length / 2;
              coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
              if (rfca->end2Focus && (ik == nKicks - 1)) {
                coord[1] -= coord[0] * inverseF[ip];
                coord[3] -= coord[2] * inverseF[ip];
              }
            }
          }
        }

        if (wakesAtEnd) {
          /* do wakes */
          if (wake)
            track_through_wake(part, np, wake, P_central, run, iPass, charge);
          if (trwake)
            track_through_trwake(part, np, trwake, *P_central, run, iPass, charge);
          if (LSCKick) {
#if !USE_MPI
            if (dgammaOverGammaNp)
              dgammaOverGammaAve /= dgammaOverGammaNp;
#else
            if (dgammaOverGammaNp) {
              if (distributedBeam) {
                double t1 = dgammaOverGammaAve;
                long t2 = dgammaOverGammaNp;
                MPI_Allreduce(&t1, &dgammaOverGammaAve, 1, MPI_DOUBLE, MPI_SUM, workers);
                MPI_Allreduce(&t2, &dgammaOverGammaNp, 1, MPI_LONG, MPI_SUM, workers);
                dgammaOverGammaAve /= dgammaOverGammaNp;
              } else
                dgammaOverGammaAve /= dgammaOverGammaNp;
            }
#endif
            trackRfCavityAddLSCKick(part, np, LSCKick, *P_central, charge,
                                    length, dgammaOverGammaAve);
          }
        }
      }
#if defined(DEBUG)
      printf("part[0][4] = %21.15le, part[0][5] = %21.15e, P0=%21.15e\n", part[0][4], part[0][5], *P_central);
#endif
    } else { /* backtracking */
      double *dgammaSave = NULL, *tSave = NULL;
      for (ik = 0; ik < nKicks; ik++) {
        dgammaOverGammaAve = dgammaOverGammaNp = 0;
        if (isSlave || !distributedBeam) {
          dgammaSave = tmalloc(sizeof(*dgammaSave) * np);
          tSave = tmalloc(sizeof(*tSave) * np);
          for (ip = 0; ip < np; ip++) {
            coord = part[ip];
            if (coord[5] == -1)
              continue;
            if (length)
              /* compute distance traveled to center of this section */
              dc4 = length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            else
              dc4 = 0;

            /* compute energy kick */
            P = *P_central * (1 + coord[5]);
            beta_i = P / (gamma = sqrt(sqr(P) + 1));
            t = (coord[4] + dc4) / (c_mks * beta_i) - timeOffset;
            if (ik == 0 && timeOffset && rfca->change_t)
              coord[4] = t * c_mks * beta_i - dc4;
            if ((dt = t - t0) < 0)
              dt = 0;
            if (!same_dgamma) {
              if (!linearize)
                dgamma = volt * sin(omega * (t - (lockPhase ? tAve : 0) + (nKicks - 1 - ik) * dtLight) + phase) * (tau ? sqrt(1 - exp(-dt / tau)) : 1);
              else
                dgamma = dgammaAve + volt * omega * (t - tAve) * cos(omega * (tAve - timeOffset) + phase);
            }
            tSave[ip] = t;
            dgammaSave[ip] = dgamma;
            if (gamma) {
              dgammaOverGammaNp++;
              dgammaOverGammaAve += dgamma / gamma;
            }

            if (length) {
              if (rfca->end1Focus && ik == 0) {
                /* apply focus kick */
                inverseF[ip] = dgamma / (2 * gamma * length);
                coord[1] -= coord[0] * inverseF[ip];
                coord[3] -= coord[2] * inverseF[ip];
              }
              /* apply initial drift */
              coord[0] += coord[1] * length / 2;
              coord[2] += coord[3] * length / 2;
              coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            }
          }
        }

        /* do wakes */
        if (LSCKick) {
#if !USE_MPI
          if (dgammaOverGammaNp)
            dgammaOverGammaAve /= dgammaOverGammaNp;
#else
          if (distributedBeam) {
            double t1 = dgammaOverGammaAve;
            long t2 = dgammaOverGammaNp;
            MPI_Allreduce(&t1, &dgammaOverGammaAve, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&t2, &dgammaOverGammaNp, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (dgammaOverGammaNp)
              dgammaOverGammaAve /= dgammaOverGammaNp;
          } else if (dgammaOverGammaNp)
            dgammaOverGammaAve /= dgammaOverGammaNp;

#endif
          trackRfCavityAddLSCKick(part, np, LSCKick, *P_central, charge,
                                  length, dgammaOverGammaAve);
        }
        if (trwake)
          track_through_trwake(part, np, trwake, *P_central, run, iPass, charge);
        if (wake)
          track_through_wake(part, np, wake, P_central, run, iPass, charge);

        if (isSlave || !distributedBeam) {
          /* apply final drift and focus kick if needed */
          for (ip = 0; ip < np; ip++) {
            coord = part[ip];
            if (coord[5] == -1)
              continue;
            if (length)
              /* compute distance traveled to center of this section */
              dc4 = length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            else
              dc4 = 0;

            /* apply energy kick */
            t = tSave[ip];
            P = *P_central * (1 + coord[5]);
            beta_i = P / (gamma = sqrt(sqr(P) + 1));
            dgamma = dgammaSave[ip];
            add_to_particle_energy(coord, t, *P_central, dgamma);
            if ((gamma1 = gamma + dgamma) <= 1)
              coord[5] = -1;
            else
              /* compute inverse focal length for exit kick */
              inverseF[ip] = -dgamma / (2 * gamma1 * length);
            coord = part[ip];
            coord[0] += coord[1] * length / 2;
            coord[2] += coord[3] * length / 2;
            coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            if (rfca->end2Focus && (ik == nKicks - 1)) {
              coord[1] -= coord[0] * inverseF[ip];
              coord[3] -= coord[2] * inverseF[ip];
            }
          }
          free(dgammaSave);
          free(tSave);
        }
      }
#if defined(DEBUG)
      printf("part[0][4] = %21.15le, part[0][5] = %21.15e, P0=%21.15e\n", part[0][4], part[0][5], *P_central);
#endif
    }
    free(inverseF);
  } else { /* matrix method */
    double sin_phase = 0.0, cos_phase, inverseF;
    double R11 = 1, R21 = 0, R22, R12, dP, ds1;

    for (ik = 0; ik < nKicks; ik++) {
      dgammaOverGammaAve = dgammaOverGammaNp = 0;

      if (length < 0) {
        /* do wakes */
        if (wake)
          track_through_wake(part, np, wake, P_central, run, iPass, charge);
        if (trwake)
          track_through_trwake(part, np, trwake, *P_central, run, iPass, charge);
        if (LSCKick) {
          if (dgammaOverGammaNp)
#if !USE_MPI
            dgammaOverGammaAve /= dgammaOverGammaNp;
#else
            if (USE_MPI) {
              double t1 = dgammaOverGammaAve;
              long t2 = dgammaOverGammaNp;
              MPI_Allreduce(&t1, &dgammaOverGammaAve, 1, MPI_DOUBLE, MPI_SUM, workers);
              MPI_Allreduce(&t2, &dgammaOverGammaNp, 1, MPI_LONG, MPI_SUM, workers);
              dgammaOverGammaAve /= dgammaOverGammaNp;
            }
#endif
          trackRfCavityAddLSCKick(part, np, LSCKick, *P_central, charge,
                                  length, dgammaOverGammaAve);
        }
      }

      if (isSlave || !distributedBeam) {
        for (ip = 0; ip < np; ip++) {
          coord = part[ip];

          /* use matrix to propagate particles */
          /* compute energy change using phase of arrival at center of cavity */
          P = *P_central * (1 + coord[5]);
          beta_i = P / (gamma = sqrt(sqr(P) + 1));
          ds1 = length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
          t = (coord[4] + ds1) / (c_mks * beta_i) - timeOffset;
          if (timeOffset && rfca->change_t)
            coord[4] = t * c_mks * beta_i - ds1;
          if ((dt = t - t0) < 0)
            dt = 0;
          if (!same_dgamma) {
            if (!linearize) {
              sin_phase = sin(omega * (t - (lockPhase ? tAve : 0) - ik * dtLight) + phase);
              cos_phase = cos(omega * (t - (lockPhase ? tAve : 0) - ik * dtLight) + phase);
              dgamma = (dgammaMax = volt * (tau ? sqrt(1 - exp(-dt / tau)) : 1)) * sin_phase;
            } else {
              cos_phase = cos(omega * tAve + phase);
              sin_phase = omega * (t - tAve) * cos_phase;
              dgamma = (dgammaMax = volt * (tau ? sqrt(1 - exp(-dt / tau)) : 1)) * sin_phase +
                       dgammaAve;
            }
          }

          if (rfca->end1Focus && length && ik == 0) {
            /* apply end focus kick */
            inverseF = dgamma / (2 * gamma * length);
            coord[1] -= coord[0] * inverseF;
            coord[3] -= coord[2] * inverseF;
          }

          dP = sqrt(sqr(gamma + dgamma) - 1) - P;
          if (gamma) {
            dgammaOverGammaNp++;
            dgammaOverGammaAve += dgamma / gamma;
          }

          if (useSRSModel) {
            /* note that Rosenzweig and Serafini use gamma in places
               * where they should probably use momentum, but I'll keep
               * their expressions for now.
               */
            double alpha, sin_alpha, gammaf;
            gammaf = gamma + dgamma;
            if (fabs(sin_phase) > 1e-6)
              alpha = log(gammaf / gamma) / (2 * SQRT2 * sin_phase);
            else
              alpha = dgammaMax / gamma / (2 * SQRT2);
            R11 = cos(alpha);
            R22 = R11 * gamma / gammaf;
            R12 = 2 * SQRT2 * gamma * length / dgammaMax * (sin_alpha = sin(alpha));
            R21 = -sin_alpha * dgammaMax / (length * gammaf * 2 * SQRT2);
          } else {
            /* my original treatment used momentum for all 
               * computations, which I still think is correct
               */
            R22 = 1 / (1 + dP / P);
            if (fabs(dP / P) > 1e-14)
              R12 = length * (P / dP * log(1 + dP / P));
            else
              R12 = length;
          }

          coord[4] += ds1;
          x = coord[0];
          xp = coord[1];
          coord[0] = x * R11 + xp * R12;
          coord[1] = x * R21 + xp * R22;
          x = coord[2];
          xp = coord[3];
          coord[2] = x * R11 + xp * R12;
          coord[3] = x * R21 + xp * R22;
          coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
          coord[5] = (P + dP - (*P_central)) / (*P_central);

          if ((gamma += dgamma) <= 1)
            coord[5] = -1;
          if (rfca->end2Focus && length && ik == (nKicks - 1)) {
            inverseF = -dgamma / (2 * gamma * length);
            coord[1] -= coord[0] * inverseF;
            coord[3] -= coord[2] * inverseF;
          }
          /* adjust s for the new particle velocity */
          coord[4] = (P + dP) / gamma * coord[4] / beta_i;
        }
      }

      if (length >= 0) {
        /* do wakes */
        if (wake)
          track_through_wake(part, np, wake, P_central, run, iPass, charge);
        if (trwake)
          track_through_trwake(part, np, trwake, *P_central, run, iPass, charge);
        if (LSCKick) {
          if (dgammaOverGammaNp)
#if !USE_MPI
            dgammaOverGammaAve /= dgammaOverGammaNp;
#else
            if (USE_MPI) {
              double t1 = dgammaOverGammaAve;
              long t2 = dgammaOverGammaNp;
              MPI_Allreduce(&t1, &dgammaOverGammaAve, 1, MPI_DOUBLE, MPI_SUM, workers);
              MPI_Allreduce(&t2, &dgammaOverGammaNp, 1, MPI_LONG, MPI_SUM, workers);
              dgammaOverGammaAve /= dgammaOverGammaNp;
            }
#endif
          trackRfCavityAddLSCKick(part, np, LSCKick, *P_central, charge,
                                  length, dgammaOverGammaAve);
        }
      }
    }
  }

  if (isSlave || !distributedBeam) {
    for (ip = 0; ip < np; ip++) {
      coord = part[ip];
      coord[0] += rfca->dx;
      coord[2] += rfca->dy;
    }
    np = removeInvalidParticles(part, np, accepted, zEnd, *P_central);
  }
  if (rfca->change_p0) {
    do_match_energy(part, np, P_central, 0);
#if defined(DEBUG)
    printf("part[0][4] = %21.15le, part[0][5] = %21.15e, P0=%21.15e\n", part[0][4], part[0][5], *P_central);
#endif
  }

  return (np);
}

void add_to_particle_energy(double *coord, double timeOfFlight, double Po, double dgamma) {
  double gamma, gamma1, PRatio, P, P1, Pz1, Pz;

  P = Po * (1 + coord[5]);                     /* old momentum */
  gamma1 = (gamma = sqrt(P * P + 1)) + dgamma; /* new gamma */
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  P1 = sqrt(gamma1 * gamma1 - 1); /* new momentum */
  coord[5] = (P1 - Po) / Po;

  /* adjust s for the new particle velocity */
  coord[4] = timeOfFlight * c_mks * P1 / gamma1;

  /* adjust slopes so that Px and Py are conserved */
  Pz = P / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  Pz1 = sqrt(Pz * Pz + gamma1 * gamma1 - gamma * gamma);
  PRatio = Pz / Pz1;
  coord[1] *= PRatio;
  coord[3] *= PRatio;
}

void add_to_particle_energy2(double *coord, double timeOfFlight, double Po, double dgamma, double dpx, double dpy) {
  double gamma, gamma1, pz, P, P1;
  double px, py, beta, betaz;

  P = Po * (1 + coord[5]);                     /* old momentum */
  gamma1 = (gamma = sqrt(P * P + 1)) + dgamma; /* new gamma */
  beta = P/gamma;
  betaz = beta/sqrt(1 + coord[1]*coord[1] + coord[3]*coord[3]);
  px = coord[1]*betaz*gamma + dpx;
  py = coord[3]*betaz*gamma + dpy;

  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  P1 = sqrt(gamma1 * gamma1 - 1); /* new momentum */
  coord[5] = (P1 - Po) / Po;

  /* adjust s for the new particle velocity */
  coord[4] = timeOfFlight * c_mks * P1 / gamma1;

  pz = sqrt(P1*P1 - px*px - py*py);
  coord[1] = px/pz;
  coord[3] = py/pz;
}

long track_through_rfcw(double **part, int64_t np, RFCW *rfcw, double **accepted, double *P_central, double zEnd,
                        RUN *run, long i_pass, CHARGE *charge) {

#ifdef HAVE_GPU
  int64_t nLeft;
  if (getElementOnGpu()) {
    startGpuTimer();
#  ifdef GPU_VERIFY
    double P_central_input = *P_central;
#  endif
    nLeft = gpu_track_through_rfcw(np, rfcw, accepted, P_central, zEnd,
                                   run, i_pass, charge);
#  ifdef GPU_VERIFY
    startCpuTimer();
    track_through_rfcw(part, np, rfcw, accepted, &P_central_input, zEnd,
                       run, i_pass, charge);
    compareGpuCpu(np, "track_through_rfcw");
    if (*P_central / P_central_input - 1 > 1e-12) {
      printf("track_through_rfcw: Warning: GPU P_central=%le vs CPU=%le\n",
             *P_central, P_central_input);
    }
#  endif /* GPU_VERIFY */
    return nLeft;
  }
#endif /* HAVE_GPU */

  if (rfcw->cellLength <= 0)
    bombElegant("invalid cell length for RFCW", NULL);
  if (rfcw->length == 0)
    printWarningForTracking("RFCW element has zero length, so wakefields will scale to 0.", NULL);
  /* set up the RFCA, TRWAKE, and WAKE structures */
  rfcw->rfca.length = rfcw->length;
  rfcw->rfca.volt = rfcw->volt;
  rfcw->rfca.phase = rfcw->phase;
  rfcw->rfca.freq = rfcw->freq;
  rfcw->rfca.Q = rfcw->Q;
  rfcw->rfca.change_p0 = rfcw->change_p0;
  rfcw->rfca.change_t = rfcw->change_t;
  rfcw->rfca.tReference = -1;
  rfcw->rfca.end1Focus = rfcw->end1Focus;
  rfcw->rfca.end2Focus = rfcw->end2Focus;
  rfcw->rfca.standingWave = rfcw->standingWave;
  if (rfcw->bodyFocusModel)
    SDDS_CopyString(&rfcw->rfca.bodyFocusModel, rfcw->bodyFocusModel);
  else
    rfcw->rfca.bodyFocusModel = NULL;
  rfcw->rfca.nKicks = rfcw->length ? rfcw->nKicks : 1;
  rfcw->rfca.dx = rfcw->dx;
  rfcw->rfca.dy = rfcw->dy;
  rfcw->rfca.tReference = rfcw->tReference;
  rfcw->rfca.linearize = rfcw->linearize;
  if (!rfcw->initialized) {
    rfcw->rfca.phase_reference = rfcw->phase_reference;
    if (rfcw->fiducial)
      SDDS_CopyString(&rfcw->rfca.fiducial, rfcw->fiducial);
    else
      rfcw->rfca.fiducial = NULL;
  }

  rfcw->trwake.charge = 0;
  rfcw->trwake.bunchedBeamMode = 1;
  rfcw->trwake.startBunch = rfcw->trwake.endBunch = -1;
  rfcw->trwake.xfactor = rfcw->trwake.yfactor = rfcw->trwake.factor = 1;
  rfcw->trwake.n_bins = rfcw->n_bins;
  rfcw->trwake.interpolate = rfcw->interpolate;
  rfcw->trwake.smoothing = rfcw->smoothing;
  rfcw->trwake.SGHalfWidth = rfcw->SGHalfWidth;
  rfcw->trwake.SGOrder = rfcw->SGOrder;
  /* misalignment is taken care of by code before and after wake call */
  rfcw->trwake.dx = 0;
  rfcw->trwake.dy = 0;
  rfcw->trwake.acausalAllowed = rfcw->trwake.i0 = 0;
  rfcw->trwake.xDriveExponent = rfcw->trwake.yDriveExponent = 1;
  rfcw->trwake.xProbeExponent = rfcw->trwake.yProbeExponent = 0;
  if (!rfcw->initialized && rfcw->includeTrWake) {
    rfcw->trwake.initialized = 0;
    if (rfcw->wakeFile) {
      if (rfcw->trWakeFile || rfcw->zWakeFile)
        SDDS_Bomb("You can't give wakeFile along with trWakeFile or zWakeFile for RFCW element");
      SDDS_CopyString(&rfcw->trWakeFile, rfcw->wakeFile);
      SDDS_CopyString(&rfcw->zWakeFile, rfcw->wakeFile);
    }

    if (rfcw->WxColumn || rfcw->WyColumn) {
      if (!rfcw->trWakeFile)
        SDDS_Bomb("no input file for transverse wake for RFCW element");
      SDDS_CopyString(&rfcw->trwake.inputFile, rfcw->trWakeFile);
      if (!rfcw->tColumn)
        SDDS_Bomb("no tColumn value for wake for RFCW element");
      SDDS_CopyString(&rfcw->trwake.tColumn, rfcw->tColumn);
      if (rfcw->WxColumn)
        SDDS_CopyString(&rfcw->trwake.WxColumn, rfcw->WxColumn);
      if (rfcw->WyColumn)
        SDDS_CopyString(&rfcw->trwake.WyColumn, rfcw->WyColumn);
    } else
      rfcw->WxColumn = rfcw->WyColumn = NULL;
  }

  rfcw->wake.charge = 0;
  rfcw->wake.bunchedBeamMode = 1;
  rfcw->wake.startBunch = rfcw->wake.endBunch = -1;
  rfcw->wake.n_bins = rfcw->n_bins;
  rfcw->wake.acausalAllowed = rfcw->wake.i0 = 0;
  rfcw->wake.interpolate = rfcw->interpolate;
  rfcw->wake.smoothing = rfcw->smoothing;
  rfcw->wake.SGHalfWidth = rfcw->SGHalfWidth;
  rfcw->wake.SGOrder = rfcw->SGOrder;
  rfcw->wake.change_p0 = rfcw->change_p0;
  rfcw->wake.factor = 1;
  if (!rfcw->initialized && rfcw->includeZWake) {
    if (rfcw->WzColumn) {
      if (!rfcw->zWakeFile)
        SDDS_Bomb("no input file for z wake for RFCW element");
      SDDS_CopyString(&rfcw->wake.inputFile, rfcw->zWakeFile);
      if (!rfcw->tColumn)
        SDDS_Bomb("no tColumn value for wake for RFCW element");
      SDDS_CopyString(&rfcw->wake.tColumn, rfcw->tColumn);
      SDDS_CopyString(&rfcw->wake.WColumn, rfcw->WzColumn);
      rfcw->wake.initialized = 0;
    } else
      rfcw->wake.WColumn = NULL;
  }

  rfcw->LSCKick.bins = rfcw->LSCBins;
  rfcw->LSCKick.interpolate = rfcw->LSCInterpolate;
  rfcw->LSCKick.lowFrequencyCutoff0 = rfcw->LSCLowFrequencyCutoff0;
  rfcw->LSCKick.lowFrequencyCutoff1 = rfcw->LSCLowFrequencyCutoff1;
  rfcw->LSCKick.highFrequencyCutoff0 = rfcw->LSCHighFrequencyCutoff0;
  rfcw->LSCKick.highFrequencyCutoff1 = rfcw->LSCHighFrequencyCutoff1;
  rfcw->LSCKick.radiusFactor = rfcw->LSCRadiusFactor;
  rfcw->LSCKick.backtrack = 0;

  rfcw->initialized = 1;

  if (rfcw->WzColumn && rfcw->includeZWake)
    rfcw->wake.factor = rfcw->length / rfcw->cellLength / (rfcw->rfca.nKicks ? rfcw->rfca.nKicks : 1);
  if ((rfcw->WxColumn || rfcw->WyColumn) && rfcw->includeTrWake)
    rfcw->trwake.factor = rfcw->length / rfcw->cellLength / (rfcw->rfca.nKicks ? rfcw->rfca.nKicks : 1);
  if (rfcw->backtrack) {
    rfcw->LSCKick.backtrack = 1;
    rfcw->rfca.backtrack = 1;
  }

  np = trackRfCavityWithWakes(part, np, &rfcw->rfca, accepted, P_central, zEnd,
                              i_pass, run, charge,
                              (rfcw->WzColumn && rfcw->includeZWake) ? &rfcw->wake : NULL,
                              ((rfcw->WxColumn || rfcw->WyColumn) && rfcw->includeTrWake) ? &rfcw->trwake : NULL,
                              rfcw->doLSC ? &rfcw->LSCKick : NULL,
                              rfcw->wakesAtEnd);
  return np;
}

/* See H. Wiedemann, Particle Accelerator Physics I, 8.2.2 */
double rfAcceptance_Fq(double q) {
  return 2 * (sqrt(q * q - 1) - acos(1 / q));
}

double solveForOverVoltage(double F, double q0) {
  return zeroNewton(&rfAcceptance_Fq, F, q0, 1e-6, 1000, 1e-12);
}

void identifyRfcaBodyFocusModel(void *pElem, long type, short *matrixMethod, short *useSRSModel, short *twFocusing1)
{
  char *bodyFocusModel;
  short standingWave;
  long nKicks, i;

  nKicks = 0;
  bodyFocusModel = NULL;
  standingWave = 0;

  switch (type) {
  case T_RFCA:
    bodyFocusModel = ((RFCA*)pElem)->bodyFocusModel;
    standingWave = ((RFCA*)pElem)->standingWave;
    nKicks = ((RFCA*)pElem)->nKicks;
    break;
  case T_RFCW:
    bodyFocusModel = ((RFCW*)pElem)->bodyFocusModel;
    standingWave = ((RFCW*)pElem)->standingWave;
    nKicks = ((RFCW*)pElem)->nKicks;
    break;
  default:
    bombElegantVA("Invalid type %ld in identifyRfcaBodyFocusModel. Report to developers.", type);
    break;
  }

  if (bodyFocusModel) {
    char *modelName[3] = {"none", "srs", "tw1"};
    switch (match_string(bodyFocusModel, modelName, 3, 0)) {
    case 0:
      break;
    case 1:
      *useSRSModel = 1;
      *matrixMethod = 1;
      if (!standingWave) {
        printWarningForTracking("Forcing STANDING_WAVE=1 for BODY_FOCUS_MODEL=SRS in RFCA or RFCW",
                                NULL);
        if (type==T_RFCA)
          ((RFCA*)pElem)->standingWave = 1;
        else
          ((RFCW*)pElem)->standingWave = 1;
      }
      break;
    case 2:
      *twFocusing1 = 1;
      *matrixMethod = 0;
      if (nKicks<10)
        bombElegant("When using BODY_FOCUS_MODEL=TW1 for RFCA or RFCW, must have N_KICKS>=10", NULL);
      if (standingWave)
        bombElegant("Can't use BODY_FOCUS_MODEL=TW1 in standing wave mode for RFCA", NULL);
      break;
    default:
      fprintf(stderr, "Error: BODY_FOCUS_MODEL=%s not understood for RFC%c. Known models are \n", 
              bodyFocusModel, type==T_RFCA?'A':'W');
      for (i=0; i<3; i++)
        fprintf(stderr, "%s%c ", modelName[i], i==2?'\n':',');
      exitElegant(1);
      break;
    }
  }
}
