/*************************************************************************\
 * Copyright (c) 2003 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2003 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: tfeedback.c
 *
 * Michael Borland, 2003
 */

#include <complex>
#include "mdb.h"
#include "track.h"

void propagateLfbCavity(double *V, double *Vp, double *VResidual, double dt0, TFBDRIVER *tfbd,
                        std::complex<double> Ig, std::complex<double> Zc);

void transverseFeedbackPickup(TFBPICKUP *tfbp, double **part0, long np0, long pass, double Po, long idSlotsPerBunch) {
  double sum, position, output;
  long i, j;
  long np;
  double *time0 = NULL;    /* array to record arrival time of each particle */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets = 0;
#if USE_MPI
  long npTotal;
  double sumTotal;
  MPI_Status mpiStatus;
#endif

#ifdef DEBUG
  printf("TFBPICKUP\n");
#endif

  /* this element does nothing in single particle mode (e.g., trajectory, orbit, ..) */
#if USE_MPI
  if (notSinglePart == 0)
    return;
#endif

  if ((tfbp->startPass > 0 && pass < tfbp->startPass) || (tfbp->endPass > 0 && pass > tfbp->endPass))
    return;

  if (tfbp->initialized == 0)
    initializeTransverseFeedbackPickup(tfbp);

  if (isSlave || !notSinglePart)
    index_bunch_assignments(part0, np0, tfbp->bunchedBeamMode ? idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBuckets, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
  else if (myid == 1)
    MPI_Send(&nBuckets, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
#endif
#ifdef DEBUG
  printf("TFBPICKUP: %ld bunches\n", nBuckets);
#endif

  if (tfbp->updateInterval > 1 && pass % tfbp->updateInterval != 0) {
    if (isSlave || !notSinglePart)
      free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
    return;
  }

  if (tfbp->nBunches == 0 || tfbp->nBunches != nBuckets) {
    if (tfbp->nBunches != nBuckets) {
      printf("Number of bunches has changed, re-initializing feedback pickup\n");
      fflush(stdout);
      for (i = 0; i < tfbp->nBunches; i++)
        free(tfbp->data[i]);
    }
    tfbp->nBunches = nBuckets;
    tfbp->data = (double **)SDDS_Realloc(tfbp->data, sizeof(*tfbp->data) * nBuckets);
    tfbp->filterOutput = (double *)SDDS_Realloc(tfbp->filterOutput, sizeof(*tfbp->filterOutput) * nBuckets);
    for (i = 0; i < nBuckets; i++) {
      if (!(tfbp->data[i] = (double *)calloc(TFB_FILTER_LENGTH, sizeof(**tfbp->data))))
        bombElegant("Memory allocation problem for TFBPICKUP", NULL);
      tfbp->filterOutput[i] = 0;
    }
    tfbp->pass0 = pass;
  }

  for (iBucket = 0; iBucket < nBuckets; iBucket++) {
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    sum = position = np = 0;
    if (isSlave || !notSinglePart) {
      if (nBuckets == 1) {
        np = np0;
        for (i = 0; i < np0; i++)
          sum += part0[i][tfbp->iPlane];
      } else {
        if (npBucket)
          np = npBucket[iBucket];
        else
          np = 0;
        for (i = 0; i < np; i++)
          sum += part0[ipBucket[iBucket][i]][tfbp->iPlane];
      }
    }

#if USE_MPI
    if (myid == 0)
      np = 0;
    MPI_Allreduce(&sum, &sumTotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (npTotal > 0)
      position = sumTotal / npTotal;
#else
    if (np > 0)
      position = sum / np;
#endif
    if (tfbp->iPlane == 0) {
      position -= tfbp->dx;
    } else if (tfbp->iPlane == 2) {
      position -= tfbp->dy;
    } else if (tfbp->iPlane == 4) {
      double beta;
      beta = Po / sqrt(Po * Po + 1);
      position /= beta * c_mks;
#ifdef DEBUG
      printf("position = %le, tRS=%ld, tR=%le\n",
             position, tfbp->tReferenceSet, tfbp->tReference);
#endif
      if (!tfbp->tReferenceSet) {
        tfbp->tReference = position;
        tfbp->tReferenceSet = 1;
      }
      position = tfbp->referenceFrequency * (position - tfbp->tReference);
      position = (position - (long)position);
      if (position > 0.5)
        position -= 1;
      position *= PIx2;
#ifdef DEBUG
      printf("positon ==> %le\n", position);
#endif
    }

    if (tfbp->rmsNoise) {
      double dposition;
      dposition = gauss_rn_lim(0.0, tfbp->rmsNoise, 2, random_3);
#if USE_MPI
      MPI_Bcast(&dposition, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      position += dposition;
    }
#ifdef DEBUG
    printf("TFBPICKUP: Putting value %e in slot %ld for bunch %ld\n",
           position, pass % tfbp->filterLength, iBucket);
#endif
    tfbp->data[iBucket][pass % tfbp->filterLength] = position;
    if ((pass - tfbp->pass0) < tfbp->filterLength) {
      tfbp->filterOutput[iBucket] = 0;
    } else {
      j = (pass - tfbp->pass0) / tfbp->updateInterval;
      for (i = output = 0; i < tfbp->filterLength; i++, j--) {
        output += tfbp->a[i] * tfbp->data[iBucket][j % tfbp->filterLength];
      }
#ifdef DEBUG
      printf("TFBPICKUP: filter output is %e\n", output);
#endif
      tfbp->filterOutput[iBucket] = output;
    }
  }

  if (isSlave || !notSinglePart)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
}

void initializeTransverseFeedbackPickup(TFBPICKUP *tfbp) {
  long i;
  double sum;

  if (tfbp->ID == NULL || !strlen(tfbp->ID))
    bombElegant("you must give an ID string for TFBPICKUP", NULL);

  for (i = sum = 0; i < TFB_FILTER_LENGTH; i++)
    sum += tfbp->a[i];
  if (fabs(sum) > 1e-6)
    printWarningForTracking((char *)"Sum of a[i] is nonzero for TFBPICKUP.", (char *)"This may produce unexpected results.");

  for (i = TFB_FILTER_LENGTH - 1; i >= 0; i--) {
    if (tfbp->a[i] != 0)
      break;
  }
  if (i < 0)
    bombElegant("All filter coefficients are zero for TFBPICKUP", NULL);
  tfbp->filterLength = i + 1;

  if (strcmp(tfbp->plane, "x") == 0 || strcmp(tfbp->plane, "X") == 0)
    tfbp->iPlane = 0;
  else if (strcmp(tfbp->plane, "y") == 0 || strcmp(tfbp->plane, "Y") == 0)
    tfbp->iPlane = 2;
  else if (strcmp(tfbp->plane, "delta") == 0 || strcmp(tfbp->plane, "DELTA") == 0)
    tfbp->iPlane = 5;
  else if (strcmp(tfbp->plane, "time") == 0 || strcmp(tfbp->plane, "TIME") == 0 ||
           strcmp(tfbp->plane, "phase") == 0 || strcmp(tfbp->plane, "PHASE") == 0) {
    tfbp->iPlane = 4;
    if (tfbp->referenceFrequency <= 0)
      bombElegant("PLANE parameter set to \"time\" or \"phase\", but REFERENCE_FREQUENCY is non-positive", NULL);
  } else
    bombElegant("PLANE parameter for TFBPICKUP must be x, y, delta, or time", NULL);

  if (tfbp->data && tfbp->nBunches) {
    for (i = 0; i < tfbp->nBunches; i++)
      if (tfbp->data)
        free(tfbp->data);
    tfbp->data = NULL;
  }
  tfbp->nBunches = 0;

  if (tfbp->updateInterval < 1)
    tfbp->updateInterval = 1;

  tfbp->initialized = 1;
}

void transverseFeedbackDriver(TFBDRIVER *tfbd, double **part0, long np0, LINE_LIST *beamline, long pass, long nPasses, char *rootname, double Po, long idSlotsPerBunch, CHARGE *charge) {
  double kick, nomKick;
  long i, j;
  double *time0 = NULL;    /* array to record arrival time of each particle */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long iBucket, nBuckets = 0;
  long rpass, updateInterval;
  double tAve, rfFactor, phase = 0;
  std::complex<double> Zc, Ig, iu;
  double V, Vp, tMax = -DBL_MAX;
  double qBunch;
  
#if USE_MPI
  MPI_Status mpiStatus;
  double buffer[5];
#endif

#if defined(DEBUG) || MPI_DEBUG
  printf("TFBDRIVER\n");
#endif

#if USE_MPI
  if (notSinglePart == 0)
    /* this element does nothing in single particle mode (e.g., trajectory, orbit, ..) */
    return;
#endif

  if ((tfbd->startPass > 0 && pass < tfbd->startPass) || (tfbd->endPass > 0 && pass > tfbd->endPass))
    return;

  if (isSlave || !notSinglePart)
    index_bunch_assignments(part0, np0, tfbd->bunchedBeamMode ? idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBuckets, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
  else if (myid == 1)
    MPI_Send(&nBuckets, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
#endif
#if defined(DEBUG) || MPI_DEBUG
  printf("TFBDRIVER: %ld bunches\n", nBuckets);
  fflush(stdout);
#endif

  if (tfbd->initialized == 0) {
    if (nPasses < tfbd->outputInterval)
      tfbd->outputInterval = nPasses;
    initializeTransverseFeedbackDriver(tfbd, beamline, nPasses * nBuckets, rootname);
    if (tfbd->nGainFactors && tfbd->nGainFactors != nBuckets)
      printWarningForTracking((char *)"Number of gain factors not equal to number of bunches", NULL);
  }

  if ((tfbd->pickup->iPlane == 5 || tfbd->pickup->iPlane == 4) && !tfbd->longitudinal)
    bombElegant("TFBDRIVER linked to TFBPICKUP with PLANE=delta or time, but driver not working on longitudinal plane", NULL);

  if (tfbd->startPass > 0 && tfbd->startPass != tfbd->pickup->startPass)
    bombElegantVA((char *)"TFBDRIVER linked to TFBPICKUP with different START_PASS value (%ld vs %ld).",
                  tfbd->startPass, tfbd->pickup->startPass);
  if (tfbd->endPass > 0 && tfbd->endPass != tfbd->pickup->endPass)
    bombElegantVA((char *)"TFBDRIVER linked to TFBPICKUP with different END_PASS value (%ld vs %ld).",
                  tfbd->endPass, tfbd->pickup->endPass);

  if ((updateInterval = tfbd->pickup->updateInterval * tfbd->updateInterval) <= 0)
    bombElegantVA((char *)"TFBDRIVER and TFBPICKUP with ID=%s have UPDATE_INTERVAL product of %d", tfbd->ID, updateInterval);
  if (pass % updateInterval != 0) {
    if (isSlave || !notSinglePart)
      free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
    return;
  }

  if (pass == 0)
    tfbd->dataWritten = 0;

  if (tfbd->nBunches == 0 || tfbd->nBunches != nBuckets) {
    if (tfbd->nBunches != nBuckets) {
      printf("Number of bunches has changed, re-initializing feedback driver.\n");
      fflush(stdout);
      for (i = 0; i < tfbd->nBunches; i++) {
        free(tfbd->driverSignal[i]);
        tfbd->driverSignal[i] = NULL;
      }
      free(tfbd->driverSignal);
    }
    tfbd->nBunches = nBuckets;
    if (!(tfbd->driverSignal = (double **)tmalloc(sizeof(*tfbd->driverSignal) * nBuckets)))
      bombElegant("memory allocation failure (transverseFeedbackDriver)", NULL);
    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      tfbd->driverSignal[iBucket] = (double *)calloc((tfbd->delay + 1 + TFB_FILTER_LENGTH), sizeof(**tfbd->driverSignal));
    }
    tfbd->pass0 = pass;
  }

  if (tfbd->nBunches != tfbd->pickup->nBunches)
    bombElegant("mismatch in number of buckets between TFBDRIVER and TFBPICKUP", NULL);

  if (tfbd->computeGeneratorCurrent) {
    Zc = std::complex<double>(tfbd->Zc[0], tfbd->Zc[1]);
    iu = std::complex<double>(0, 1);
  }

  for (iBucket = 0; iBucket < nBuckets; iBucket++) {
    nomKick = 0;
#if USE_MPI
#  if MPI_DEBUG
    printf("Waiting on barrier at top of bucket loop\n");
    fflush(stdout);
#  endif
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    kick = tfbd->pickup->filterOutput[iBucket] * tfbd->strength;
    rpass = (pass - tfbd->pass0) / updateInterval;
#if defined(DEBUG) || MPI_DEBUG
    printf("TFBDRIVER: pass %ld\nstoring kick %e in slot %ld based on filter output of %e\n",
           pass, kick, rpass % (tfbd->delay + tfbd->filterLength), tfbd->pickup->filterOutput[iBucket]);
#endif

    tfbd->driverSignal[iBucket][rpass % (tfbd->delay + tfbd->filterLength)] = kick;

    kick = 0;
    if (rpass >= tfbd->delay + tfbd->filterLength) {
      for (i = 0; i < tfbd->filterLength; i++) {
#ifdef DEBUG
        printf("TFBDRIVER: adding term a[%ld]=%e  *   %e\n",
               i, tfbd->a[i], tfbd->driverSignal[iBucket][(rpass - tfbd->delay - i) % (tfbd->delay + tfbd->filterLength)]);
        fflush(stdout);
#endif
        kick += tfbd->a[i] * tfbd->driverSignal[iBucket][(rpass - tfbd->delay - i) % (tfbd->delay + tfbd->filterLength)];
      }
    }
#if defined(DEBUG) || MPI_DEBUG
    printf("TFBDRIVER: kick = %le\n", kick);
    fflush(stdout);
#endif
    if (iBucket < tfbd->nGainFactors)
      kick *= tfbd->gainFactor[iBucket];
    if (tfbd->kickLimit > 0 && fabs(kick) > tfbd->kickLimit)
      kick = SIGN(kick) * tfbd->kickLimit;
    phase = tfbd->phase * PI / 180;

    if (tfbd->gainChargeScale>0) {
      if (!charge)
        bombElegant("TFBDRIVER GAIN_CHARGE_SCALE>0 but no CHARGE element seen.", NULL);
#if USE_MPI
      long npTotal=0, np1=0;
      if (myid!=0)
	np1 = npBucket[iBucket];
      MPI_Allreduce(&np1, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      qBunch = npTotal*charge->macroParticleCharge;
#else
      qBunch = npBucket[iBucket]*charge->macroParticleCharge;
#endif
      kick *= tfbd->gainFactor0 + qBunch/tfbd->gainChargeScale;
    } else {
      kick *= tfbd->gainFactor0;
    }
    if (isSlave || !notSinglePart) {
      tAve = 0;
      rfFactor = 1;

      if (tfbd->frequency > 0) {
        /* Determine average arrival time of bunch */
        if (nBuckets == 1) {
#if USE_MPI
          tAve = computeAverage_p(time0, np0, workers);
#else
          compute_average(&tAve, time0, np0);
#endif
        } else {
#if USE_MPI
          long npTotal = 0;
#endif
          double tSum, error;
          error = tSum = 0;
          for (i = 0; i < npBucket[iBucket]; i++)
            tSum = KahanPlus(tSum, time0[ipBucket[iBucket][i]], &error);
#if USE_MPI
          MPI_Allreduce(&npBucket[iBucket], &npTotal, 1, MPI_LONG, MPI_SUM, workers);
          if (npTotal)
            tAve = KahanParallel(tSum, error, workers) / npTotal;
#else
          if (npBucket[iBucket] > 0)
            tAve = tSum / npBucket[iBucket];
#endif
        }
      }

      if (tfbd->computeGeneratorCurrent) {
        /* method described in Berenc, RF-TN-2018-005; email from Berenc on 6/27/2018 */
        double Vkick, dt;
        long nc;
        if (!(tfbd->initialized & TFBDRIVER_CLOCK_INIT)) {
          /* align the clock to this bunch */
          tfbd->initialized |= TFBDRIVER_CLOCK_INIT;
          tfbd->lastTime = tAve + tfbd->clockOffset - 1. / tfbd->clockFrequency;
        }
        nc = ceil((tAve - tfbd->lastTime) * tfbd->clockFrequency);
        dt = nc / tfbd->clockFrequency;
        tfbd->thisTime = tfbd->lastTime + dt;
        if (tfbd->thisTime < tAve)
          bombElegant("clock alignment problem in TFBDRIVER. Consider reducing clock frequency.", NULL);
        dt -= tfbd->clockOffset; /* ensures that we are targetting the region where the bunch supposedly sits */
        Vkick = kick * Po * particleMassMV * 1e6;
        Ig = Vkick / Zc / (1 - exp(-tfbd->sigma * dt)) * exp(-iu * (dt * tfbd->omegag + tfbd->phase));
      }

      nomKick = kick * cos(phase);
      if (!tfbd->longitudinal) {
        j = tfbd->pickup->iPlane + 1;
        if (nBuckets == 1) {
          for (i = 0; i < np0; i++) {
            if (tfbd->frequency > 0)
              rfFactor = cos(PIx2 * tfbd->frequency * (time0[i] - tAve) + phase);
            part0[i][j] += kick / (1 + part0[i][5]);
          }
        } else {
          if (npBucket) {
            for (i = 0; i < npBucket[iBucket]; i++) {
              if (tfbd->frequency > 0)
                rfFactor = cos(PIx2 * tfbd->frequency * (time0[ipBucket[iBucket][i]] - tAve) + phase);
              part0[ipBucket[iBucket][i]][j] += kick * rfFactor / (1 + part0[ipBucket[iBucket][i]][5]);
            }
          }
        }
      } else { /* longitudinal */
        tMax = -DBL_MAX;
        if (nBuckets == 1) {
          for (i = 0; i < np0; i++) {
            if (time0[i] > tMax)
              tMax = time0[i];
            if (tfbd->computeGeneratorCurrent) {
              double dt0;
              dt0 = time0[i] - tfbd->lastTime;
              propagateLfbCavity(&V, &Vp, NULL, dt0, tfbd, Ig, Zc);
              part0[i][5] += V / (Po * particleMassMV * 1e6);
            } else { /* no generator sim */
              if (tfbd->frequency > 0)
                rfFactor = cos(PIx2 * tfbd->frequency * (time0[i] - tAve) + phase);
              part0[i][5] += kick * rfFactor;
            }
          }
        } else { /* multiple buckets */
          if (npBucket) {
#ifdef MPI_DEBUG
            printf("ib=%ld, tAve = %le, freq = %le, phase = %le\n, rfFactor = %le, np=%ld\n",
                   iBucket, tAve, tfbd->frequency, phase, rfFactor, npBucket[iBucket]);
            fflush(stdout);
#endif
            for (i = 0; i < npBucket[iBucket]; i++) {
              if (time0[ipBucket[iBucket][i]] > tMax)
                tMax = time0[ipBucket[iBucket][i]];
              if (tfbd->computeGeneratorCurrent) {
                double dt0;
                dt0 = time0[ipBucket[iBucket][i]] - tfbd->lastTime;
                propagateLfbCavity(&V, &Vp, NULL, dt0, tfbd, Ig, Zc);
                part0[ipBucket[iBucket][i]][5] += V / (Po * particleMassMV * 1e6);
              } else { /* no generator sim */
                if (tfbd->frequency > 0)
                  rfFactor = cos(PIx2 * tfbd->frequency * (time0[ipBucket[iBucket][i]] - tAve) + phase);
                part0[ipBucket[iBucket][i]][5] += kick * rfFactor;
              }
            }
          }
        }
      }
    }

#if defined(DEBUG) || MPI_DEBUG
    printf("TFBDRIVER: preparing for output for bunch %ld\n", iBucket);
    fflush(stdout);
#endif

#if USE_MPI
    if (myid == 0) {
      MPI_Recv(buffer, 4, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus);
      nomKick = buffer[0];
      tfbd->thisTime = buffer[1];
      Ig = std::complex<double>(buffer[2], buffer[3]);
    } else if (myid == 1) {
      buffer[0] = nomKick;
      buffer[1] = tfbd->thisTime;
      buffer[2] = Ig.real();
      buffer[3] = Ig.imag();
      MPI_Send(buffer, 4, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
#endif
    if (tfbd->computeGeneratorCurrent) {
#if USE_MPI
      if (myid == 0)
        tMax = -DBL_MAX;
      MPI_Allreduce(&tMax, &buffer[0], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      tMax = buffer[0];
#endif
      if (tMax > tfbd->thisTime)
        bombElegant("bunch extends past the clock tick. Increase the clock offset or decrease the clock frequency for TFBDRIVER.", NULL);
      propagateLfbCavity(&V, &Vp, &tfbd->VResidual, tfbd->thisTime - tfbd->lastTime, tfbd, Ig, Zc);
      tfbd->lastV = V;
      tfbd->lastVp = Vp;
      tfbd->lastIg = sqrt(sqr(Ig.real()) + sqr(Ig.imag()));
      tfbd->lastTime = tfbd->thisTime;
    }

#if USE_MPI
    if (myid == 0)
#endif
      if (tfbd->outputFile) {
        if ((tfbd->outputIndex + 1) % tfbd->outputInterval == 0) {
#ifdef DEBUG
          printf("Flushing output file\n");
          fflush(stdout);
#endif
          if (!SDDS_UpdatePage(tfbd->SDDSout, FLUSH_TABLE)) {
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb((char *)"problem flushing data for TFBDRIVER output file");
          }
          tfbd->dataWritten = 1;
        }
#ifdef DEBUG
        printf("Preparing to set rows\n");
        fflush(stdout);
#endif
        if (!SDDS_SetRowValues(tfbd->SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                               tfbd->outputIndex,
                               "Bunch", iBucket, "Pass", pass,
                               "PickupOutput", tfbd->pickup->filterOutput[iBucket],
                               "DriverOutput", nomKick, NULL) ||
            (tfbd->computeGeneratorCurrent &&
             !SDDS_SetRowValues(tfbd->SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, tfbd->outputIndex,
                                "GeneratorCurrent", tfbd->lastIg,
                                "Voltage", tfbd->lastV,
                                "VoltageDeriv", tfbd->lastVp,
                                "ResidualVoltage", tfbd->VResidual,
                                NULL))) {
          SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"problem writing data for TFBDRIVER output file");
        }
        tfbd->outputIndex++;
        tfbd->dataWritten = 0;
      }

#if defined(DEBUG) || MPI_DEBUG
    printf("TFBDRIVER: end of loop for bunch %ld\n", iBucket);
    fflush(stdout);
#endif
  }

#if defined(DEBUG) || MPI_DEBUG
  printf("TFBDRIVER: exited from loop over bunches\n");
  fflush(stdout);
#endif

#if USE_MPI
#  ifdef MPI_DEBUG
  printf("TFBDRIVER: Waiting on barrier after loop over bunches\n");
  fflush(stdout);
#  endif
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (isSlave || !notSinglePart)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

#if defined(DEBUG) || MPI_DEBUG
  printf("TFBDRIVER: end of routine\n");
  fflush(stdout);
#endif
}

void initializeTransverseFeedbackDriver(TFBDRIVER *tfbd, LINE_LIST *beamline, long nPasses, char *rootname) {
  ELEMENT_LIST *eptr;
  long pickupFound = 0, i;
  long count = 0;

  if (tfbd->ID == NULL || !strlen(tfbd->ID))
    bombElegant("you must give an ID string for TFBDRIVER", NULL);
  if (tfbd->gainFactorFile) {
    SDDS_DATASET SDDSgff;
    if (!strlen(tfbd->gainFactorFile))
      bombElegant("TFBDRIVER GAIN_FACTOR_FILE is invalid", NULL);
    if (!tfbd->gainFactorColumn || !strlen(tfbd->gainFactorColumn))
      bombElegant("TFBDRIVER GAIN_FACTOR_COLUMN is invalid", NULL);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSgff, tfbd->gainFactorFile) ||
        SDDS_ReadPage(&SDDSgff) != 1 || (tfbd->nGainFactors = SDDS_RowCount(&SDDSgff)) < 1 ||
        !(tfbd->gainFactor = SDDS_GetColumnInDoubles(&SDDSgff, tfbd->gainFactorColumn))) {
      fprintf(stderr, "Error: GAIN_FACTOR_FILE file %s is unreadable, has insufficient data, or is missing column %s.\n", tfbd->gainFactorFile, tfbd->gainFactorColumn);
      exitElegant(1);
    }
  }
  if (tfbd->RaOverQ > 0)
    count++;
  if (tfbd->QLoaded > 0)
    count++;
  if (count && count != 2)
    bombElegant("RAOVERQ and QLOADED must both be set to positive values, or else both set to zero in TFBDRIVER", NULL);
  if (count == 2) {
    if (!tfbd->longitudinal)
      bombElegant("Positive RAOVERQ and QLOADED given for TFBDRIVER in transverse plane", NULL);
    tfbd->computeGeneratorCurrent = 1;
    if (tfbd->frequency <= 0)
      bombElegant("FREQUENCY must be positive when RAOVERQ and QLOADED also set to positive values in TFBDRIVER", NULL);
    if (tfbd->clockFrequency <= 0)
      bombElegant("CLOCK_FREQUENCY must be positive when RAOVERQ and QLOADED also set to positive values in TFBDRIVER", NULL);
    if (tfbd->clockOffset <= 0)
      bombElegant("CLOCK_OFFSET must be positive when RAOVERQ and QLOADED also set to positive values in TFBDRIVER", NULL);
  } else
    tfbd->computeGeneratorCurrent = 0;

  for (i = TFB_FILTER_LENGTH - 1; i >= 0; i--) {
    if (tfbd->a[i] != 0)
      break;
  }
  if (i < 0)
    bombElegant("All filter coefficients are zero for TFBDRIVER", NULL);
  tfbd->filterLength = i + 1;

  eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_TFBPICKUP && strcmp(tfbd->ID, ((TFBPICKUP *)eptr->p_elem)->ID) == 0) {
      pickupFound = 1;
      tfbd->pickup = ((TFBPICKUP *)eptr->p_elem);
      break;
    }
    eptr = eptr->succ;
  }
  if (!pickupFound)
    bombElegant("pickup not found for TFBDRIVER", NULL);

  if (tfbd->delay < 0)
    bombElegant("TFBDRIVER delay is negative", NULL);
  if (tfbd->outputInterval < 1)
    bombElegant("TFBDRIVER output interval is less than 1", NULL);

#if USE_MPI
  if (myid == 0)
#endif
    if (tfbd->outputFile) {
      char *filename;
      filename = compose_filename(tfbd->outputFile, rootname);
      if (!tfbd->SDDSout)
        tfbd->SDDSout = (SDDS_DATASET *)tmalloc(sizeof(*(tfbd->SDDSout)));
      if (!SDDS_InitializeOutputElegant(tfbd->SDDSout, SDDS_BINARY, 1, NULL, NULL, filename) ||
          !SDDS_DefineSimpleColumn(tfbd->SDDSout, "Pass", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleColumn(tfbd->SDDSout, "Bunch", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleColumn(tfbd->SDDSout, "PickupOutput", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(tfbd->SDDSout, "DriverOutput", tfbd->longitudinal ? "" : "rad", SDDS_DOUBLE) ||
          (tfbd->computeGeneratorCurrent && (!SDDS_DefineSimpleColumn(tfbd->SDDSout, "GeneratorCurrent", "A", SDDS_DOUBLE) ||
                                             !SDDS_DefineSimpleColumn(tfbd->SDDSout, "Voltage", "V", SDDS_DOUBLE) ||
                                             !SDDS_DefineSimpleColumn(tfbd->SDDSout, "VoltageDeriv", "V/s", SDDS_DOUBLE) ||
                                             !SDDS_DefineSimpleColumn(tfbd->SDDSout, "ResidualVoltage", "V", SDDS_DOUBLE))) ||
          !SDDS_WriteLayout(tfbd->SDDSout) || !SDDS_StartPage(tfbd->SDDSout, tfbd->outputInterval)) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"Problem setting up TFBDRIVER output file");
      }
      free(filename);
    }
  tfbd->dataWritten = tfbd->outputIndex = 0;

  if (tfbd->driverSignal) {
    long i;
    for (i = 0; i < tfbd->nBunches; i++)
      free(tfbd->driverSignal[i]);
    free(tfbd->driverSignal);
    tfbd->driverSignal = NULL;
  }

  tfbd->nBunches = 0;

  if (tfbd->updateInterval < 1)
    tfbd->updateInterval = 1;

  tfbd->lastV = tfbd->lastIg = tfbd->lastTime = tfbd->lastVp = 0;

  if (tfbd->computeGeneratorCurrent) {
    /* compute quantities described in Berenc, RF-TN-2018-005 */
    double denom;
    tfbd->omegao = PIx2 * tfbd->frequency;
    tfbd->omegan = tfbd->omegao * sqrt(1 - 1 / (4 * sqr(tfbd->QLoaded)));
    if (tfbd->driveFrequency <= 0)
      tfbd->omegag = tfbd->omegan;
    else
      tfbd->omegag = PIx2 * tfbd->driveFrequency;
    tfbd->sigma = tfbd->omegao / (2 * tfbd->QLoaded);
    tfbd->k = (tfbd->omegao / 4) * tfbd->RaOverQ;
    denom = sqr(sqr(tfbd->omegao) - sqr(tfbd->omegag)) + sqr(2 * tfbd->sigma * tfbd->omegag);
    /* store real and imaginary parts in an array since track.h can't include complex.h */
    tfbd->Zc[0] = 4 * tfbd->k * tfbd->sigma * sqr(tfbd->omegag) / denom;
    tfbd->Zc[1] = 2 * tfbd->k * tfbd->omegag * (sqr(tfbd->omegao) - sqr(tfbd->omegag)) / denom;
    if (tfbd->clockOffset * tfbd->clockFrequency >= 1)
      bombElegant("TFBDRIVER clock offset must be less than clock period!", NULL);
  }

  tfbd->initialized = TFBDRIVER_MAIN_INIT;
}

void propagateLfbCavity(double *V, double *Vp, double *VResidualSum, double dt0, TFBDRIVER *tfbd,
                        std::complex<double> Ig, std::complex<double> Zc) {
  double alpha, alpha1, alpha2, beta, beta1, beta2;
  std::complex<double> iu, exp1, exp2;
  iu = std::complex<double>(0, 1);
  alpha1 = tfbd->lastV;
  alpha2 = -(Ig * Zc).real();
  alpha = alpha1 + alpha2;
  beta1 = (2 * tfbd->k * tfbd->lastIg - tfbd->sigma * tfbd->lastV - tfbd->lastVp) / tfbd->omegan;
  beta2 = (-sqr(tfbd->omegao) / tfbd->omegag * (Ig * Zc).imag() - tfbd->sigma * (Ig * Zc).real()) / tfbd->omegan;
  beta = beta1 + beta2;
  exp1 = Ig * Zc * exp(iu * tfbd->omegag * dt0);
  exp2 = (alpha + iu * beta) * exp((iu * tfbd->omegan - tfbd->sigma) * dt0);
  *V = (exp1 + exp2).real();
  *Vp = (iu * tfbd->omegag * exp1 + (iu * tfbd->omegan - tfbd->sigma) * exp2).real();
  if (VResidualSum)
    *VResidualSum += ((alpha1 + iu * beta1) * exp((iu * tfbd->omegan - tfbd->sigma) * dt0)).real();
}
