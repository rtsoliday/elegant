/*************************************************************************\
* Copyright (c) 2026 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: cwake.c
 * contents: set_up_cwake(), track_through_cwake().
 *
 * A unified single-pass time-domain wake element that combines the
 * physics of WAKE (longitudinal monopole) and TRWAKE (transverse
 * dipole and quadrupole), and additionally supports two
 * position-independent ("constant") transverse deflecting channels.
 * One SDDS file holds every wake column the user wants to apply;
 * particle binning, bunch identification, and the per-bunch loop are
 * done only once across every enabled channel.
 *
 * Channel layout (indexed by CWAKE_*):
 *   WZ : longitudinal monopole       drive (0,0)  probe (0,0)  -> dgamma
 *   WX : x dipole                    drive (1,0)  probe (0,0)  -> dxp
 *   WY : y dipole                    drive (0,1)  probe (0,0)  -> dyp
 *   QX : x quadrupole                drive (0,0)  probe (1,0)  -> dxp
 *   QY : y quadrupole                drive (0,0)  probe (0,1)  -> dyp
 *   CX : x constant (position-indep) drive (0,0)  probe (0,0)  -> dxp
 *   CY : y constant (position-indep) drive (0,0)  probe (0,0)  -> dyp
 */

#include "mdb.h"
#include "track.h"
#include "table.h"
#ifdef HAVE_GPU
#  include "gpu_cwake.h"
#endif

typedef struct {
  long driveX, driveY;
  long probeX, probeY;
  long kickPlane;   /* 0 = longitudinal dgamma, 1 = xp, 2 = yp */
} CWAKE_INFO;

static const CWAKE_INFO cwakeInfo[CWAKE_N_WAKES] = {
  {0, 0, 0, 0, 0},  /* WZ */
  {1, 0, 0, 0, 1},  /* WX */
  {0, 1, 0, 0, 2},  /* WY */
  {0, 0, 1, 0, 1},  /* QX */
  {0, 0, 0, 1, 2},  /* QY */
  {0, 0, 0, 0, 1},  /* CX */
  {0, 0, 0, 0, 2},  /* CY */
};

static char *getCWakeColumn(CWAKE *cw, long iw) {
  switch (iw) {
  case CWAKE_WZ: return cw->WzColumn;
  case CWAKE_WX: return cw->WxColumn;
  case CWAKE_WY: return cw->WyColumn;
  case CWAKE_QX: return cw->QxColumn;
  case CWAKE_QY: return cw->QyColumn;
  case CWAKE_CX: return cw->CxColumn;
  case CWAKE_CY: return cw->CyColumn;
  }
  return NULL;
}

static double getCWakeChannelFactor(CWAKE *cw, long iw) {
  switch (iw) {
  case CWAKE_WZ: return cw->zFactor;
  case CWAKE_WX: return cw->xFactor;
  case CWAKE_WY: return cw->yFactor;
  case CWAKE_QX: return cw->qxFactor;
  case CWAKE_QY: return cw->qyFactor;
  case CWAKE_CX: return cw->cxFactor;
  case CWAKE_CY: return cw->cyFactor;
  }
  return 1.0;
}

void set_up_cwake(CWAKE *cw, RUN *run, long pass, long particles, CHARGE *charge) {
  SDDS_DATASET SDDSin;
  double tmin, tmax;
  long iw, nEnabled = 0;

  if (!charge)
    bombElegant("CWAKE element requires a CHARGE element upstream in the beamline.", NULL);
  cw->macroParticleCharge = charge->macroParticleCharge;

  if (cw->initialized)
    return;
  cw->initialized = 1;

  for (iw = 0; iw < CWAKE_N_WAKES; iw++) {
    cw->W[iw] = NULL;
    cw->enabled[iw] = 0;
  }
  cw->t = NULL;

  if (cw->n_bins < 2 && cw->n_bins != 0)
    bombElegant("N_BINS must be >=2 or 0 (autoscale) for CWAKE element", NULL);
  if (!cw->inputFile || !strlen(cw->inputFile))
    bombElegant("CWAKE element requires INPUTFILE", NULL);
  if (!cw->tColumn || !strlen(cw->tColumn))
    bombElegant("CWAKE element requires TCOLUMN", NULL);

  for (iw = 0; iw < CWAKE_N_WAKES; iw++) {
    char *colName = getCWakeColumn(cw, iw);
    if (colName && strlen(colName))
      nEnabled++;
  }
  if (!nEnabled)
    bombElegant("CWAKE element has no enabled channels (set at least one of WZCOLUMN, WXCOLUMN, WYCOLUMN, QXCOLUMN, QYCOLUMN, CXCOLUMN, CYCOLUMN)", NULL);

#if SDDS_MPI_IO
  SDDSin.parallel_io = 0;
#endif
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, cw->inputFile) ||
      SDDS_ReadPage(&SDDSin) != 1) {
    fprintf(stderr, "Error: unable to open or read CWAKE file %s\n", cw->inputFile);
    exitElegant(1);
  }
  if ((cw->wakePoints = SDDS_RowCount(&SDDSin)) < 2) {
    fprintf(stderr, "Error: too little data in CWAKE file %s\n", cw->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, cw->tColumn, "s", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
    fprintf(stderr, "Error: problem with time column %s in CWAKE file %s.  Check existence, type, and units.\n",
            cw->tColumn, cw->inputFile);
    exitElegant(1);
  }
  if (!(cw->t = SDDS_GetColumnInDoubles(&SDDSin, cw->tColumn))) {
    fprintf(stderr, "Error: problem retrieving time data from CWAKE file %s\n", cw->inputFile);
    exitElegant(1);
  }
  for (iw = 0; iw < CWAKE_N_WAKES; iw++) {
    char *colName = getCWakeColumn(cw, iw);
    if (!colName || !strlen(colName))
      continue;
    if (!(cw->W[iw] = SDDS_GetColumnInDoubles(&SDDSin, colName))) {
      fprintf(stderr, "Error: column %s not found in CWAKE file %s\n", colName, cw->inputFile);
      exitElegant(1);
    }
    cw->enabled[iw] = 1;
  }
  SDDS_Terminate(&SDDSin);

  find_min_max(&tmin, &tmax, cw->t, cw->wakePoints);
  if (tmin >= tmax) {
    fprintf(stderr, "Error: zero or negative time span in CWAKE file %s\n", cw->inputFile);
    exitElegant(1);
  }
  cw->dt = (tmax - tmin) / (cw->wakePoints - 1);
  cw->i0 = 0;
  if (tmin != 0) {
    if (!cw->acausalAllowed) {
      fprintf(stderr, "Error: CWAKE function does not start at t=0 for file %s\n", cw->inputFile);
      fprintf(stderr, "If you really want this, set ACAUSAL_ALLOWED=1\n");
      exitElegant(1);
    }
    cw->i0 = -1;
    for (iw = 0; iw < cw->wakePoints; iw++) {
      if (fabs(cw->t[iw]) < 1e-6 * cw->dt) {
        cw->i0 = iw;
        break;
      }
    }
    if (cw->i0 == -1) {
      fprintf(stderr, "Error: CWAKE function does not have a value at t=0 (within 1e-6*dt) for file %s\n", cw->inputFile);
      exitElegant(1);
    }
    if (fabs(tmin) > tmax) {
      fprintf(stderr, "Error: acausal CWAKE function has |tmin|>tmax for file %s\n", cw->inputFile);
      exitElegant(1);
    }
  }
}

void track_through_cwake(double **part0, long np0, CWAKE *cw, double *PoInput,
                         RUN *run, long i_pass, CHARGE *charge) {
  double *Itime = NULL, *xItime = NULL, *yItime = NULL;
  double *Vtime = NULL;
  long max_n_bins = 0;
  long *pbin = NULL;
  double *time0 = NULL, *time = NULL, *pz = NULL;
  double **part = NULL;
  long *ibParticle = NULL;
  long **ipBucket = NULL;
  long *npBucket = NULL;
  long ib, nb = 0;
  long iBucket, nBuckets = 0, max_np = 0, ip, np, iw;
  double tmin, tmax, tmean = 0, dt = 0, Po, rampFactor;
  long needI = 0, needXI = 0, needYI = 0, needTransverse = 0;
#if USE_MPI
  double *buffer;
#endif

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    double gpuPoInput = *PoInput;
    long action = gpu_cwake_bunched_mode_action(np0, cw, charge);
    if (action == GPU_BUNCHED_WAKE_UNSUPPORTED) {
      part0 = forceParticlesToCpu("CWAKE option CPU fallback");
    } else if (action == GPU_BUNCHED_WAKE_SKIP) {
      return;
    } else {
      startGpuTimer();
      gpu_track_through_cwake(np0, cw, &gpuPoInput, run, i_pass, charge);
#  ifdef GPU_VERIFY
      double cpuPoInput = *PoInput;
      startCpuTimer();
      track_through_cwake(part0, np0, cw, &cpuPoInput, run, i_pass, charge);
      if (cw->change_p0) {
        double diff = fabs(cpuPoInput - gpuPoInput);
        double scale = fabs(cpuPoInput) > fabs(gpuPoInput) ?
                       fabs(cpuPoInput) : fabs(gpuPoInput);
        if (diff > 1e-12 && diff > 1e-12 * scale) {
          fprintf(stderr,
                  "elegant CUDA VERIFY CWAKE P_central mismatch cpu=%21.15e gpu=%21.15e\n",
                  cpuPoInput, gpuPoInput);
          exit(1);
        }
      }
      compareGpuCpu(np0, "track_through_cwake");
#  endif
      *PoInput = gpuPoInput;
      return;
    }
  }
#endif

  /* Load wake file and populate enabled[] before we make allocation decisions. */
  set_up_cwake(cw, run, i_pass, np0, charge);

  needI = cw->enabled[CWAKE_WZ] ||
          cw->enabled[CWAKE_QX] || cw->enabled[CWAKE_QY] ||
          cw->enabled[CWAKE_CX] || cw->enabled[CWAKE_CY];
  needXI = cw->enabled[CWAKE_WX];
  needYI = cw->enabled[CWAKE_WY];
  needTransverse = needXI || needYI ||
                   cw->enabled[CWAKE_QX] || cw->enabled[CWAKE_QY] ||
                   cw->enabled[CWAKE_CX] || cw->enabled[CWAKE_CY];

  if (isSlave || !distributedBeam) {
    rampFactor = 0;
    if (cw->rampPasses <= 1 || i_pass >= (cw->rampPasses - 1))
      rampFactor = 1;
    else
      rampFactor = (i_pass + 1.0) / cw->rampPasses;
    Po = *PoInput;

    index_bunch_assignments(part0, np0,
                            (charge && cw->bunchedBeamMode) ? charge->idSlotsPerBunch : 0,
                            Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#if USE_MPI
    if (mpiAbort) return;
#endif

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (cw->bunchedBeamMode &&
          ((cw->startBunch >= 0 && iBucket < cw->startBunch) ||
           (cw->endBunch   >= 0 && iBucket > cw->endBunch)))
        continue;

      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = trealloc(pbin, sizeof(*pbin) * (max_np = np));
        if (needTransverse)
          pz = trealloc(pz, sizeof(*pz) * np);
        compute_average(&tmean, time, np);
      } else {
        if ((np = npBucket[iBucket]) == 0)
          continue;
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)tmalloc(sizeof(*time) * np);
          pbin = trealloc(pbin, sizeof(*pbin) * np);
          if (needTransverse)
            pz = trealloc(pz, sizeof(*pz) * np);
          max_np = np;
        }
        for (ip = 0, tmean = 0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          tmean += time[ip];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]],
                 sizeof(double) * totalPropertiesPerParticle);
        }
        if (np > 0) tmean /= np;
      }

      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#if USE_MPI
      if (isSlave && distributedBeam)
        find_global_min_max(&tmin, &tmax, np, workers);
#endif

      if (isSlave || !distributedBeam) {
        if ((tmax - tmin) > (cw->t[cw->wakePoints - 1] - cw->t[0])) {
          if (!cw->allowLongBeam) {
            fprintf(stderr, "Error: the beam is longer than the CWAKE wake function.\n");
            fprintf(stderr, "Beam length %le s, wake length %le s\n",
                    tmax - tmin, cw->t[cw->wakePoints - 1] - cw->t[0]);
            exitElegant(1);
          } else {
            char warningBuffer[1024];
            snprintf(warningBuffer, 1024,
                     "Beam length %le s exceeds wake length %le s in CWAKE.",
                     tmax - tmin, cw->t[cw->wakePoints - 1] - cw->t[0]);
            printWarningForTracking("CWAKE: beam longer than wake function.", warningBuffer);
          }
        }

        dt = cw->dt;
        if (cw->n_bins) {
          nb = cw->n_bins;
          tmin = tmean - dt * nb / 2.0;
        } else {
          nb = (tmax - tmin) / dt + 3;
          tmin -= dt;
          tmax += dt;
        }
        if (nb <= 0) {
          printWarningForTracking("CWAKE: zero or negative bin count.",
                                  "Likely an extremely long bunch.  Wake ignored.");
          continue;
        }

        if (nb > max_n_bins) {
          max_n_bins = nb;
          if (needI)  Itime  = trealloc(Itime,  sizeof(*Itime)  * max_n_bins);
          if (needXI) xItime = trealloc(xItime, sizeof(*xItime) * max_n_bins);
          if (needYI) yItime = trealloc(yItime, sizeof(*yItime) * max_n_bins);
          Vtime = trealloc(Vtime, sizeof(*Vtime) * (max_n_bins + 1));
        }

        if (cw->tilt)
          rotateBeamCoordinatesForMisalignment(part, np, cw->tilt);

        /* Single binning sweep: fills all needed histograms, pbin[], and pz[]. */
        if (Itime)  for (ib = 0; ib < nb; ib++) Itime[ib]  = 0;
        if (xItime) for (ib = 0; ib < nb; ib++) xItime[ib] = 0;
        if (yItime) for (ib = 0; ib < nb; ib++) yItime[ib] = 0;
        for (ip = 0; ip < np; ip++) {
          double tval = time[ip];
          long ibp;
          pbin[ip] = -1;
          ibp = (long)((tval - tmin) / dt + 0.5);
          if (ibp < 0 || ibp > nb - 1)
            continue;
          if (Itime)  Itime[ibp]  += 1;
          if (xItime) xItime[ibp] += part[ip][0] - cw->dx;
          if (yItime) yItime[ibp] += part[ip][2] - cw->dy;
          pbin[ip] = ibp;
          if (needTransverse)
            pz[ip] = Po * (1 + part[ip][5]) /
                     sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
        }
      }

#if USE_MPI
      if (isSlave && distributedBeam) {
        if (Itime)  { buffer = malloc(sizeof(double) * nb);
                      MPI_Allreduce(Itime,  buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
                      memcpy(Itime,  buffer, sizeof(double) * nb); free(buffer); }
        if (xItime) { buffer = malloc(sizeof(double) * nb);
                      MPI_Allreduce(xItime, buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
                      memcpy(xItime, buffer, sizeof(double) * nb); free(buffer); }
        if (yItime) { buffer = malloc(sizeof(double) * nb);
                      MPI_Allreduce(yItime, buffer, nb, MPI_DOUBLE, MPI_SUM, workers);
                      memcpy(yItime, buffer, sizeof(double) * nb); free(buffer); }
      }
#endif

      if (isSlave || !distributedBeam) {
        if (cw->smoothing && nb >= (2 * cw->SGHalfWidth + 1)) {
          if (Itime)
            SavitzyGolaySmooth(Itime,  nb, cw->SGOrder, cw->SGHalfWidth, cw->SGHalfWidth, 0);
          if (xItime)
            SavitzyGolaySmooth(xItime, nb, cw->SGOrder, cw->SGHalfWidth, cw->SGHalfWidth, 0);
          if (yItime)
            SavitzyGolaySmooth(yItime, nb, cw->SGOrder, cw->SGHalfWidth, cw->SGHalfWidth, 0);
        }

        /* For each enabled channel: convolve driver -> Vtime, scale, kick. */
        for (iw = 0; iw < CWAKE_N_WAKES; iw++) {
          double *driver = NULL, channelFactor, factor;
          if (!cw->enabled[iw])
            continue;
          if (cwakeInfo[iw].driveX == 1)      driver = xItime;
          else if (cwakeInfo[iw].driveY == 1) driver = yItime;
          else                                driver = Itime;
          if (!driver)
            continue;

          Vtime[nb] = 0;
          convolveArrays(Vtime, nb, driver, nb, cw->W[iw], cw->wakePoints, cw->i0);
          channelFactor = getCWakeChannelFactor(cw, iw);
          factor = cw->macroParticleCharge * particleRelSign *
                   cw->factor * channelFactor * rampFactor;
          for (ib = 0; ib < nb; ib++) Vtime[ib] *= factor;

          if (cwakeInfo[iw].kickPlane == 0) {
            applyLongitudinalWakeKicks(part, time, pbin, np, Po,
                                       Vtime, nb, tmin, dt, cw->interpolate);
            /* The longitudinal kick changes delta and therefore pz.  The
             * subsequent transverse kicks need the updated pz to match a
             * stacked WAKE + TRWAKE layout (each TRWAKE element recomputes
             * pz from the post-WAKE particle state). */
            if (needTransverse) {
              for (ip = 0; ip < np; ip++)
                pz[ip] = Po * (1 + part[ip][5]) /
                         sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
            }
          } else {
            long plane    = (cwakeInfo[iw].kickPlane == 1) ? 0 : 1;
            long exponent = (plane == 0) ? cwakeInfo[iw].probeX
                                         : cwakeInfo[iw].probeY;
            applyTransverseWakeKicks(part, time, pz, pbin, np, Po, plane,
                                     Vtime, nb, tmin, dt, cw->interpolate,
                                     exponent);
          }
        }

        if (cw->tilt)
          rotateBeamCoordinatesForMisalignment(part, np, -cw->tilt);

        if (nBuckets != 1 && np > 0) {
          for (ip = 0; ip < np; ip++)
            memcpy(part0[ipBucket[iBucket][ip]], part[ip],
                   sizeof(double) * totalPropertiesPerParticle);
        }
      }
    }
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (cw->change_p0 && cw->enabled[CWAKE_WZ])
    do_match_energy(part0, np0, PoInput, 0);

  if (part && part != part0)
    free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
  if (time && time != time0)
    free(time);
  if (pbin) free(pbin);
  if (pz)   free(pz);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
  if (Itime)  free(Itime);
  if (xItime) free(xItime);
  if (yItime) free(yItime);
  if (Vtime)  free(Vtime);
}
