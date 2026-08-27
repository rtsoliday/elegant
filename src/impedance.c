/*************************************************************************\
* Copyright (c) 2026 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: impedance.c
 * contents: track_through_impedance(), set_up_impedance().
 *
 * Combined single-pass impedance element.  Performs at most one
 * index_bunch_assignments() and one binning sweep per call, then evaluates
 * any subset of {longitudinal monopole, x dipole, y dipole, x quadrupole,
 * y quadrupole} wakes from a single SDDS impedance file whose columns are
 * named in the element parameters.
 *
 * Sign conventions (same as ZLONGIT / ZTRANSVERSE):
 *   V(t) = V(w)*exp(i*w*t),  I(t) = I(w)*exp(i*w*t)
 *   For inductor:  Z = i*w*L
 *   For capacitor: Z = -i/(w*C)
 *   For resistor:  Z = R
 * The minus sign converting energy loss to energy gain is applied
 * internally and must not be included in the impedance.
 */

#include "mdb.h"
#include "track.h"
#include "table.h"
#include "fftpackC.h"
#ifdef HAVE_GPU
#  include "gpu_impedance.h"
#endif

typedef struct {
  long driveX, driveY;   /* drive exponents */
  long probeX, probeY;   /* probe exponents */
  long kickPlane;        /* 0=long energy, 1=xp, 2=yp */
  const char *vName;     /* output column for V_t(.) */
  const char *vText;     /* SDDS column header text */
} IMP_WAKE_INFO;

static const IMP_WAKE_INFO impedanceWakes[IMPEDANCE_N_WAKES] = {
  {0, 0, 0, 0, 0, "VtZL",
   "&column name=VtZL, units=V, type=double, description=\"Longitudinal monopole voltage\" &end"},
  {1, 0, 0, 0, 1, "VtZxD",
   "&column name=VtZxD, units=V, type=double, description=\"Transverse x dipole voltage\" &end"},
  {0, 1, 0, 0, 2, "VtZyD",
   "&column name=VtZyD, units=V, type=double, description=\"Transverse y dipole voltage\" &end"},
  {0, 0, 1, 0, 1, "VtZxQ",
   "&column name=VtZxQ, units=V/m, type=double, description=\"Transverse x quadrupole voltage\" &end"},
  {0, 0, 0, 1, 2, "VtZyQ",
   "&column name=VtZyQ, units=V/m, type=double, description=\"Transverse y quadrupole voltage\" &end"},
};

static void getWakeColumns(IMPEDANCE *imp, long iw, char **real, char **imag) {
  switch (iw) {
  case IMPEDANCE_ZL:  *real = imp->ZL_real;  *imag = imp->ZL_imag;  break;
  case IMPEDANCE_ZXD: *real = imp->ZxD_real; *imag = imp->ZxD_imag; break;
  case IMPEDANCE_ZYD: *real = imp->ZyD_real; *imag = imp->ZyD_imag; break;
  case IMPEDANCE_ZXQ: *real = imp->ZxQ_real; *imag = imp->ZxQ_imag; break;
  case IMPEDANCE_ZYQ: *real = imp->ZyQ_real; *imag = imp->ZyQ_imag; break;
  default:            *real = *imag = NULL;                         break;
  }
}

static double *getImpedanceColumn(SDDS_DATASET *SDDSin, char *colName, long nrows) {
  double *vec;
  long i;
  if (!colName || !strlen(colName)) {
    vec = tmalloc(sizeof(*vec) * nrows);
    for (i = 0; i < nrows; i++)
      vec[i] = 0;
    return vec;
  }
  vec = SDDS_GetColumnInDoubles(SDDSin, colName);
  if (!vec)
    bombElegantVA("IMPEDANCE: column %s not found in input file\n", colName);
  return vec;
}

void set_up_impedance(IMPEDANCE *imp, RUN *run, long pass, long particles, CHARGE *charge) {
  long i, n_spect = 0, anyEnabled = 0;
  double *freqData = NULL;
  double df_spect = 0;
  SDDS_DATASET SDDSin;

  if (!charge)
    bombElegant("IMPEDANCE element requires a CHARGE element upstream in the beamline.", NULL);
  imp->macroParticleCharge = charge->macroParticleCharge;

  if (imp->initialized)
    return;

  for (i = 0; i < IMPEDANCE_N_WAKES; i++) {
    imp->Z[i] = NULL;
    imp->enabled[i] = 0;
  }

  if (!imp->inputFile)
    bombElegant("IMPEDANCE element requires INPUTFILE", NULL);
  if (!imp->freqColumn)
    bombElegant("IMPEDANCE element requires FREQCOLUMN", NULL);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, imp->inputFile) ||
      !SDDS_ReadPage(&SDDSin)) {
    fprintf(stderr, "IMPEDANCE: unable to read file %s\n", imp->inputFile);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if ((n_spect = SDDS_RowCount(&SDDSin)) < 4)
    bombElegantVA("IMPEDANCE: too little data in input file %s\n", imp->inputFile);
  if (!power_of_2(n_spect - 1))
    bombElegant("IMPEDANCE: number of impedance points must be 2^n+1, n>1", NULL);

  freqData = SDDS_GetColumnInDoubles(&SDDSin, imp->freqColumn);
  if (!freqData)
    bombElegantVA("IMPEDANCE: frequency column %s not found in input file\n", imp->freqColumn);
  if (!checkPointSpacing(freqData, n_spect, 1e-6))
    bombElegant("IMPEDANCE: frequency values are not equally spaced", NULL);
  df_spect = (freqData[n_spect - 1] - freqData[0]) / (n_spect - 1);
  if (df_spect <= 0)
    bombElegant("IMPEDANCE: zero or negative frequency spacing", NULL);

  imp->n_bins = 2 * (n_spect - 1);
  imp->bin_size = 1.0 / (imp->n_bins * df_spect);
  imp->n_freq = n_spect;

  printf("IMPEDANCE using Nb=%ld, dt=%e s (span %e s) from %s\n",
         imp->n_bins, imp->bin_size, imp->n_bins * imp->bin_size, imp->inputFile);
  fflush(stdout);

  for (i = 0; i < IMPEDANCE_N_WAKES; i++) {
    char *creal = NULL, *cimag = NULL;
    double *Zr = NULL, *Zi = NULL;
    long ks;
    getWakeColumns(imp, i, &creal, &cimag);
    imp->enabled[i] = (creal || cimag) ? 1 : 0;
    if (!imp->enabled[i])
      continue;
    Zr = getImpedanceColumn(&SDDSin, creal, n_spect);
    Zi = getImpedanceColumn(&SDDSin, cimag, n_spect);
    if (Zi[0] != 0)
      bombElegantVA("IMPEDANCE: wake %ld has non-zero imaginary DC term: Z(0)=(%le,%le)\n",
                    i, Zr[0], Zi[0]);
    imp->Z[i] = tmalloc(sizeof(*imp->Z[i]) * imp->n_bins);
    for (ks = 0; ks < imp->n_bins; ks++)
      imp->Z[i][ks] = 0;
    imp->Z[i][0] = 2 * Zr[0];
    if (imp->n_bins % 2 == 0)
      imp->Z[i][imp->n_bins - 1] = 2 * Zr[n_spect - 1];
    for (ks = 1; ks < n_spect - 1; ks++) {
      imp->Z[i][2 * ks - 1] = Zr[ks];
      imp->Z[i][2 * ks]     = Zi[ks];
    }
    if (imp->highFrequencyCutoff0 > 0)
      applyLowPassFilterToImpedance(imp->Z[i], n_spect,
                                    imp->highFrequencyCutoff0,
                                    imp->highFrequencyCutoff1);
    free(Zr);
    free(Zi);
    anyEnabled = 1;
  }
  free(freqData);
  if (!SDDS_Terminate(&SDDSin))
    bombElegantVA("IMPEDANCE: error closing input file %s\n", imp->inputFile);

  if (!anyEnabled)
    bombElegant("IMPEDANCE element has no enabled wakes (specify at least one Z*_REAL or Z*_IMAG column)", NULL);

  imp->SDDS_wake_initialized = 0;
#if (!USE_MPI)
  if (imp->wakes) {
    char *filename;
    long nVCol = 0, k, j;
    SDDS_DEFINITION pardef[3];
    SDDS_DEFINITION *coldef;
    pardef[0].name = "Pass";
    pardef[0].text = "&parameter name=Pass, type=long &end";
    pardef[1].name = "Bunch";
    pardef[1].text = "&parameter name=Bunch, type=long &end";
    pardef[2].name = "q";
    pardef[2].text = "&parameter name=q, units=C, type=double, description=\"Total charge\" &end";
    for (k = 0; k < IMPEDANCE_N_WAKES; k++)
      if (imp->enabled[k])
        nVCol++;
    coldef = tmalloc(sizeof(*coldef) * (2 + nVCol));
    coldef[0].name = "Deltat";
    coldef[0].text = "&column name=Deltat, symbol=\"$gD$rt\", units=s, type=double, description=\"Time after bin start\" &end";
    coldef[1].name = "LinearDensity";
    coldef[1].text = "&column name=LinearDensity, units=C/s, type=double, description=\"Longitudinal current density\" &end";
    j = 2;
    for (k = 0; k < IMPEDANCE_N_WAKES; k++) {
      if (!imp->enabled[k])
        continue;
      coldef[j].name = (char *)impedanceWakes[k].vName;
      coldef[j].text = (char *)impedanceWakes[k].vText;
      j++;
    }
    filename = compose_filename(imp->wakes, run->rootname);
    imp->SDDS_wake = tmalloc(sizeof(*(imp->SDDS_wake)));
    SDDS_ElegantOutputSetup(imp->SDDS_wake, filename, SDDS_BINARY, 1,
                            "impedance wakes", run->runfile, run->lattice,
                            pardef, 3, coldef, 2 + nVCol,
                            "set_up_impedance",
                            SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    free(filename);
    free(coldef);
    imp->SDDS_wake_initialized = 1;
  }
#endif

  imp->initialized = 1;
}

/* V(f) = Z(f) * I(f) * factor, in the half-complex packed layout used by
 * realFFT().  Matches the multiply pattern in zlongit.c and ztransverse.c. */
static void applyImpedanceMultiplication(double *Ifreq, double *Z, double *Vfreq,
                                         long nb, double factor) {
  long ib, iReal, iImag, nfreq;
  Vfreq[0] = Ifreq[0] * Z[0] * factor;
  nfreq = nb / 2 + 1;
  if (nb % 2 == 0)
    Vfreq[nb - 1] = Ifreq[nb - 1] * Z[nb - 1] * factor;
  for (ib = 1; ib < nfreq - 1; ib++) {
    iImag = (iReal = 2 * ib - 1) + 1;
    Vfreq[iReal] = (Ifreq[iReal] * Z[iReal] - Ifreq[iImag] * Z[iImag]) * factor;
    Vfreq[iImag] = (Ifreq[iReal] * Z[iImag] + Ifreq[iImag] * Z[iReal]) * factor;
  }
}

void track_through_impedance(double **part0, long np0, IMPEDANCE *imp,
                             double Po, RUN *run, long i_pass, CHARGE *charge) {
  double *time0 = NULL, *time = NULL;
  double *pz = NULL;
  long *pbin = NULL;
  long *ibParticle = NULL;
  int64_t **ipBucket = NULL;
  int64_t *npBucket = NULL;
  double **part = NULL;
  long max_np = 0;
  long iBucket, nBuckets = 0, ib, nb, iw;
  int64_t np = 0, ip;
  double tmin, tmax, tmean = 0, dt, factor, rampFactor = 1;
  double *Itime = NULL, *xItime = NULL, *yItime = NULL;
  double *Ifreq = NULL;
  double *Vt[IMPEDANCE_N_WAKES] = {NULL};
#if !USE_MPI
  long i_pass0 = i_pass;
#endif
  long needTransverseX = 0, needTransverseY = 0;
  long needLongDriver = 0;
#if USE_MPI
  long offset = 0, length = 0;
  double tmin_part, tmax_part;
  double *buffer;
#endif

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    long action = gpu_impedance_bunched_mode_action(np0, imp, charge);
    if (action == GPU_BUNCHED_WAKE_UNSUPPORTED) {
      part0 = forceParticlesToCpu("IMPEDANCE option CPU fallback");
    } else if (action == GPU_BUNCHED_WAKE_SKIP) {
      return;
    } else {
      startGpuTimer();
      gpu_track_through_impedance(np0, imp, Po, run, i_pass, charge);
#  ifdef GPU_VERIFY
      startCpuTimer();
      track_through_impedance(part0, np0, imp, Po, run, i_pass, charge);
      compareGpuCpu(np0, "track_through_impedance");
#  endif
      return;
    }
  }
#endif

  if ((i_pass -= imp->startOnPass) < 0 || imp->factor == 0)
    return;

  if (imp->rampPasses <= 1 || i_pass >= (imp->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / imp->rampPasses;

  /* Force file load now so imp->enabled[] is populated before we use it. */
  set_up_impedance(imp, run, i_pass, np0, charge);

  needLongDriver = imp->enabled[IMPEDANCE_ZL] ||
                   imp->enabled[IMPEDANCE_ZXQ] ||
                   imp->enabled[IMPEDANCE_ZYQ];
  needTransverseX = imp->enabled[IMPEDANCE_ZXD];
  needTransverseY = imp->enabled[IMPEDANCE_ZYD];

  if (isSlave || !distributedBeam) {
    index_bunch_assignments(part0, np0,
                            (charge && imp->bunchedBeamMode) ? charge->idSlotsPerBunch : 0,
                            Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (imp->bunchedBeamMode &&
          ((imp->startBunch >= 0 && iBucket < imp->startBunch) ||
           (imp->endBunch   >= 0 && iBucket > imp->endBunch)))
        continue;

      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
        pz   = trealloc(pz,   sizeof(*pz)   * np);
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
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)trealloc(time, sizeof(*time) * np);
          pbin = trealloc(pbin, sizeof(*pbin) * np);
          pz   = trealloc(pz,   sizeof(*pz)   * np);
          max_np = np;
        }
        for (ip = 0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]],
                 sizeof(double) * totalPropertiesPerParticle);
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

      set_up_impedance(imp, run, i_pass, np0, charge);

      nb = imp->n_bins;
      dt = imp->bin_size;
      tmin -= dt;
      tmax += dt;

      if ((tmax - tmin) * 2 > nb * dt) {
        TRACKING_CONTEXT tcontext;
        getTrackingContext(&tcontext);
#if USE_MPI && !defined(MPI_DEBUG)
        if (myid == 1)
          dup2(fdStdout, fileno(stdout));
#endif
        printf("%s %s: Time span of bunch %ld (%21.15le s) exceeds half the total time span (%21.15le s).\n",
               entity_name[tcontext.elementType], tcontext.elementName,
               iBucket, tmax - tmin, nb * dt);
        printf("Increase the number of points in the impedance file (smaller frequency resolution) to extend the time span.\n");
        if (!imp->allowLongBeam) {
#if USE_MPI
          mpiAbort = MPI_ABORT_BUNCH_TOO_LONG_ZLONGIT;
          return;
#else
          exitElegant(1);
#endif
        }
      }

      if (needLongDriver  && !Itime)  Itime  = tmalloc(sizeof(*Itime)  * 2 * nb);
      if (needTransverseX && !xItime) xItime = tmalloc(sizeof(*xItime) * 2 * nb);
      if (needTransverseY && !yItime) yItime = tmalloc(sizeof(*yItime) * 2 * nb);
      if (!Ifreq)
        Ifreq = tmalloc(sizeof(*Ifreq) * 2 * nb);
      for (iw = 0; iw < IMPEDANCE_N_WAKES; iw++)
        if (imp->enabled[iw] && !Vt[iw])
          Vt[iw] = tmalloc(sizeof(*Vt[iw]) * 2 * (nb + 1));

      if (imp->reverseTimeOrder && imp->enabled[IMPEDANCE_ZL]) {
        for (ip = 0; ip < np; ip++)
          time[ip] = 2 * tmean - time[ip];
      }

      if (Itime)
        for (ib = 0; ib < 2 * nb; ib++) Itime[ib] = 0;
      if (xItime)
        for (ib = 0; ib < 2 * nb; ib++) xItime[ib] = 0;
      if (yItime)
        for (ib = 0; ib < 2 * nb; ib++) yItime[ib] = 0;

      /* Single binning pass.  Bin CENTERS at tmin + ib*dt, matching the
       * convention of binTransverseTimeDistribution(). */
      for (ip = 0; ip < np; ip++) {
        double tval = time[ip];
        pbin[ip] = -1;
        ib = (long)((tval - tmin) / dt + 0.5);
        if (ib < 0 || ib > nb - 1)
          continue;
        if (Itime) {
          if (imp->area_weight && ib > 0 && ib < (nb - 1)) {
            double dist = (tval - (ib * dt + tmin)) / dt;
            Itime[ib]     += 0.5;
            Itime[ib - 1] += 0.25 - 0.5 * dist;
            Itime[ib + 1] += 0.25 + 0.5 * dist;
          } else {
            Itime[ib] += 1;
          }
        }
        if (xItime)
          xItime[ib] += part[ip][0] - imp->dx;
        if (yItime)
          yItime[ib] += part[ip][2] - imp->dy;
        pbin[ip] = ib;
        pz[ip] = Po * (1 + part[ip][5]) /
                 sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
      }

#if USE_MPI
      if (isSlave) {
        offset = ((long)((tmin_part - tmin) / dt) - 1 ? (long)((tmin_part - tmin) / dt) - 1 : 0);
        length = ((long)((tmax_part - tmin_part) / dt) + 2 < nb ? (long)((tmax_part - tmin_part) / dt) + 2 : nb);
        if (offset < 0) { length -= offset; offset = 0; }
        if (offset >= nb) { offset = 0; length = nb; }
        if ((offset + length) > nb) length = nb - offset;
        {
          long hi;
          double *hPtr[3];
          hPtr[0] = Itime;
          hPtr[1] = xItime;
          hPtr[2] = yItime;
          for (hi = 0; hi < 3; hi++) {
            if (!hPtr[hi]) continue;
            buffer = malloc(sizeof(double) * length);
            MPI_Allreduce(&hPtr[hi][offset], buffer, length, MPI_DOUBLE, MPI_SUM, workers);
            memcpy(&hPtr[hi][offset], buffer, sizeof(double) * length);
            free(buffer);
          }
        }
      }
#endif

      if (imp->smoothing) {
        if (Itime)
          SavitzyGolaySmooth(Itime,  nb, imp->SGOrder, imp->SGHalfWidth, imp->SGHalfWidth, 0);
        if (xItime)
          SavitzyGolaySmooth(xItime, nb, imp->SGOrder, imp->SGHalfWidth, imp->SGHalfWidth, 0);
        if (yItime)
          SavitzyGolaySmooth(yItime, nb, imp->SGOrder, imp->SGHalfWidth, imp->SGHalfWidth, 0);
      }

      /* For each enabled wake, FFT driver -> Z multiply -> inverse FFT. */
      for (iw = 0; iw < IMPEDANCE_N_WAKES; iw++) {
        double *driver = NULL, channelFactor;
        if (!imp->enabled[iw])
          continue;
        if (impedanceWakes[iw].driveX == 1)      driver = xItime;
        else if (impedanceWakes[iw].driveY == 1) driver = yItime;
        else                                     driver = Itime;
        if (!driver)
          continue;
        memcpy(Ifreq, driver, 2 * nb * sizeof(*Ifreq));
        realFFT(Ifreq, nb, 0);
        switch (iw) {
        case IMPEDANCE_ZL:  channelFactor = imp->zFactor;  break;
        case IMPEDANCE_ZXD: channelFactor = imp->xFactor;  break;
        case IMPEDANCE_ZYD: channelFactor = imp->yFactor;  break;
        case IMPEDANCE_ZXQ: channelFactor = imp->qxFactor; break;
        case IMPEDANCE_ZYQ: channelFactor = imp->qyFactor; break;
        default:            channelFactor = 1.0;           break;
        }
        factor = imp->macroParticleCharge * particleRelSign / dt *
                 imp->factor * channelFactor * rampFactor;
        applyImpedanceMultiplication(Ifreq, imp->Z[iw], Vt[iw], nb, factor);
        realFFT(Vt[iw], nb, INVERSE_FFT);
        Vt[iw][nb] = 0;
      }

      /* Longitudinal kick (closely mirrors zlongit.c). */
      if (imp->enabled[IMPEDANCE_ZL]) {
        double *V = Vt[IMPEDANCE_ZL];
        for (ip = 0; ip < np; ip++) {
          long ibp = pbin[ip];
          double dgam, dt1;
          if (ibp < 0 || ibp > nb - 1)
            continue;
          if (imp->interpolate && ibp > 0) {
            dt1 = time[ip] - (tmin + dt * ibp);
            if (dt1 < 0)
              dt1 += dt;
            else
              ibp += 1;
            if (ibp >= nb)
              continue;
            dgam = (V[ibp - 1] + (V[ibp] - V[ibp - 1]) / dt * dt1) /
                   (1e6 * particleMassMV * particleRelSign);
          } else {
            dgam = V[ibp] / (1e6 * particleMassMV * particleRelSign);
          }
          if (dgam) {
            double tparticle = time[ip];
            if (imp->reverseTimeOrder)
              tparticle = 2 * tmean - tparticle;
            add_to_particle_energy(part[ip], tparticle, Po, -dgam);
          }
        }
        if (imp->reverseTimeOrder) {
          for (ip = 0; ip < np; ip++)
            time[ip] = 2 * tmean - time[ip];
        }
      }

      /* Transverse kicks (dipole + quadrupole). */
      for (iw = 1; iw < IMPEDANCE_N_WAKES; iw++) {
        long plane, exponent;
        if (!imp->enabled[iw])
          continue;
        plane = (impedanceWakes[iw].kickPlane == 1) ? 0 : 1;
        exponent = (plane == 0) ? impedanceWakes[iw].probeX
                                : impedanceWakes[iw].probeY;
        applyTransverseWakeKicks(part, time, pz, pbin, np, Po, plane,
                                 Vt[iw], nb, tmin, dt, imp->interpolate,
                                 exponent);
      }

#if (!USE_MPI)
      if (imp->SDDS_wake_initialized && imp->wakes &&
          (imp->wake_interval <= 0 ||
           ((i_pass0 - imp->wake_start) % imp->wake_interval) == 0) &&
          i_pass0 >= imp->wake_start && (imp->wake_end<0 || i_pass0 <= imp->wake_end)) {
        double convI = imp->macroParticleCharge * particleRelSign / dt;
        if (!SDDS_StartTable(imp->SDDS_wake, nb)) {
          SDDS_SetError("Problem starting SDDS table for IMPEDANCE wake output");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        for (ib = 0; ib < nb; ib++) {
          long col;
          double linDensity = Itime ? Itime[ib] * convI : 0.0;
          if (!SDDS_SetRowValues(imp->SDDS_wake,
                                 SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                                 0, ib * dt, 1, linDensity, -1)) {
            SDDS_SetError("Problem setting rows of IMPEDANCE wake table");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
          col = 2;
          for (iw = 0; iw < IMPEDANCE_N_WAKES; iw++) {
            if (!imp->enabled[iw])
              continue;
            if (!SDDS_SetRowValues(imp->SDDS_wake,
                                   SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                                   col, Vt[iw][ib], -1)) {
              SDDS_SetError("Problem setting rows of IMPEDANCE wake table");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            col++;
          }
        }
        if (!SDDS_SetParameters(imp->SDDS_wake,
                                SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Pass", i_pass0, "Bunch", iBucket,
                                "q", imp->macroParticleCharge * particleRelSign * np,
                                NULL)) {
          SDDS_SetError("Problem setting parameters of IMPEDANCE wake table");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!SDDS_WriteTable(imp->SDDS_wake)) {
          SDDS_SetError("Problem writing IMPEDANCE wake table");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!inhibitFileSync)
          SDDS_DoFSync(imp->SDDS_wake);
      }
#endif

      if (nBuckets != 1 && np > 0) {
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip],
                 sizeof(double) * totalPropertiesPerParticle);
      }
#if USE_MPI
      MPI_Barrier(workers);
#endif
    }
  }

#if USE_MPI
  MPI_Barrier(workers);
#endif

  if (Itime)  free(Itime);
  if (xItime) free(xItime);
  if (yItime) free(yItime);
  if (Ifreq)  free(Ifreq);
  for (iw = 0; iw < IMPEDANCE_N_WAKES; iw++)
    if (Vt[iw])
      free(Vt[iw]);
  if (part && part != part0 && max_np > 0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin) free(pbin);
  if (pz)   free(pz);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
#if USE_MPI
  MPI_Barrier(workers);
#endif
}
