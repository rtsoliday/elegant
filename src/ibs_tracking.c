/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "zibs.h"
#include "SDDS.h"

static double tmp_safe_sqrt;
#define SAFE_SQRT(x) ((tmp_safe_sqrt = (x)) < 0 ? 0.0 : sqrt(tmp_safe_sqrt))
static TWISS *twiss0;

void init_IBS(ELEMENT_LIST *element);
void free_IBS(IBSCATTER *IBS);
void inflateEmittance(double **coord, double Po, double eta[4],
                      long offset, long istart, long iend, long *index, double factor);
void inflateEmittanceZ(double **coord, double Po, long isRing, double dt,
                       long istart, long iend, long *index, double zRate[3], double Duration);
void SDDS_IBScatterSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                         RUN *run, ELEMENT_LIST *elem, char *caller, long isRing);
void dump_IBScatter(SDDS_TABLE *SDDS_table, IBSCATTER *IBS, long pass);
void reset_IBS_output(ELEMENT_LIST *element);

void slicebeam(double **coord, int64_t np, double *time, double Po, long nslice, long *index, long *count, double *dt);
void zeroslice(long islice, IBSCATTER *IBS);
long computeSliceParameters(double C[6], double S[6][6], double **part, long *index, long start, long end, double Po);
void forth_propagate_twiss(IBSCATTER *IBS, long islice, double betax0, double alphax0,
                           double betay0, double alphay0, RUN *run);
void copy_twiss(TWISS *tp0, TWISS *tp1);

void track_IBS(double **part0, int64_t np0, ELEMENT_LIST *eptr, double Po,
               ELEMENT_LIST *element, CHARGE *charge, long i_pass, long n_passes, RUN *run) {
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long max_np = 0, np;
  int64_t ip;
  long iBucket, nBuckets = 0;

  long *index = NULL, *count = NULL;
  long istart, iend, ipart, icoord, ihcoord, islice;
  double aveCoord[6], S[6][6];
  double betax0, alphax0, betay0, alphay0;
  double randomNumber;
  double RNSigma[3];
  double bLength = 0, tLength, zRate[3];
  double eta[4];
  IBSCATTER *IBS;
#if USE_MPI
  long npTotal, countTotal;
  MPI_Status mpiStatus;
  long elemOffset, elemCount;
  double mpiTemp;
  /* printf("myid=%d, np=%ld, npTotal=%ld\n", myid, np, npTotal);  */
#endif

  IBS = (IBSCATTER *)eptr->p_elem;

  if (IBS->factor<=0)
    return;
  
  if (IBS->verbose) {
    printf("Starting IBSCATTER\n");
    fflush(stdout);
  }

  if (IBS->nslice < 1)
    bombElegant("IBSCATTER: NSLICE has to be an integer >= 1", NULL);
  if (IBS->bunchedBeamMode && !IBS->isRing)
    bombElegantVA("IBSCATTER %s: has BUNCHED_BEAM_MODE=1 BUT ISRING=0", element->name);
  if (!charge)
    bombElegant("IBSCATTER: no CHARGE element was given", NULL);

  if (!IBS->s)
    init_IBS(element);
  if (IBS->dT == 0)
    return;
  if (IBS->dT < 0)
    bombElegant("IBSCATTER failed because interval dT was negative. This shouldn't happen.", NULL);

  if (IBS->verbose) {
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    printf("Beginning IBSCATTER algorithm (tests passed)\n");
    fflush(stdout);
  }

  if (isSlave || !distributedBeam) {
    eta[0] = IBS->etax[IBS->elements - 1];
    eta[1] = IBS->etaxp[IBS->elements - 1];
    eta[2] = IBS->etay[IBS->elements - 1];
    eta[3] = IBS->etayp[IBS->elements - 1];

#ifdef DEBUG
    printf("bunchedBeamMode=%d, charge = %c, ID slots per bunch: %ld\n", IBS->bunchedBeamMode,
           charge ? 'Y' : 'N', (charge && IBS->bunchedBeamMode) ? charge->idSlotsPerBunch : 0);
#endif
    index_bunch_assignments(part0, np0, (charge && IBS->bunchedBeamMode) ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#ifdef DEBUG
    printf("%ld buckets\n", nBuckets);
    fflush(stdout);
    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      printf("bucket %ld: %" PRId64 " particles\n", iBucket, npBucket[iBucket]);
      fflush(stdout);
    }
#endif
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBuckets, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
  else if (myid == 1)
    MPI_Send(&nBuckets, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
#endif
  if (IBS->verbose) {
    printf("Bunch assignment completed\n");
    fflush(stdout);
  }

  if (IBS->verbose) {
    printf("%ld bunches identified\n", nBuckets);
    fflush(stdout);
  }

  for (iBucket = 0; iBucket < nBuckets; iBucket++) {
    np = -1;

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (IBS->verbose) {
      printf("Starting IBS for bunch %ld\n", iBucket);
      fflush(stdout);
    }

    if (isSlave || !distributedBeam) {
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
      } else {
        np = npBucket[iBucket];
#ifdef DEBUG
        printf("IBSCATTER: copying data to work array, iBucket=%ld, np=%ld\n", iBucket, np);
        fflush(stdout);
#endif
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)trealloc(time, sizeof(*time) * np);
          max_np = np;
        }
        for (ip = 0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
      }
    }

    if (charge) {
#if USE_MPI
      if (myid == 0)
        np = 0;
      if (np == -1)
        bombElegant("Error: np==-1 in IBSCATTER. Seek professional help!", NULL);
      MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      IBS->charge = charge->macroParticleCharge * npTotal;
      /* printf("myid=%d, np=%ld, npTotal=%ld, charge=%le\n", myid, np, npTotal, IBS->charge); */
      if (IBS->verbose) {
        printf("bunch %ld has charge=%e, np=%ld\n", iBucket, IBS->charge, npTotal);
        fflush(stdout);
      }
#else
      if (np == -1)
        bombElegant("Error: np==-1 in IBSCATTER. Seek professional help!", NULL);
      IBS->charge = charge->macroParticleCharge * np;
      if (IBS->verbose) {
        printf("bunch %ld has charge=%e, np=%ld\n", iBucket, IBS->charge, np);
        fflush(stdout);
      }
#endif
    }

#if USE_MPI
    if (npTotal == 0)
      continue;
#else
    if (np == 0)
      continue;
#endif
    if (!IBS->charge)
      continue;

    if (isSlave || !distributedBeam) {
      index = (long *)malloc(sizeof(long) * np);
      count = (long *)malloc(sizeof(long) * IBS->nslice);
      slicebeam(part, np, time, Po, IBS->nslice, index, count, &tLength);
      bLength = 0;
      if (IBS->nslice > 1) {
        bLength = c_mks * tLength / sqrt(4 * PI);
        /*
        if (IBS->isRing)
          bLength *= 2;
        */
      }
    }

    iend = 0;
    for (islice = 0; islice < IBS->nslice; islice++) {
      if (IBS->verbose) {
#if USE_MPI
        long countTemp;
        MPI_Barrier(MPI_COMM_WORLD);
	if (IBS->verbose>3) {
	  printf("myid=%d, isSlave=%ld, distributedBeam=%ld\n", myid, isSlave, distributedBeam);
	  fflush(stdout);
	}
        if (myid == 0)
          countTemp = 0;
        else {
          if (!count)
            bombElegant("Error: count array is null in track_IBS", NULL);
          countTemp = count[islice];
        }
        MPI_Allreduce(&countTemp, &countTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	printf("Starting computation of parameters for slice %ld for bunch %ld (%ld particles)\n", islice, iBucket, countTotal);
	fflush(stdout);
#else
        printf("Starting computation of parameters for slice %ld for bunch %ld (%ld particles)\n", islice, iBucket, count[islice]);
        fflush(stdout);
#endif
      }

#if USE_MPI
      countTotal = 0;
      if (myid == 0) {
	long dummyCount = 0;
	MPI_Allreduce(&dummyCount, &countTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	istart = iend = 0;
      } else {
        istart = iend;
        iend += count[islice];
        MPI_Allreduce(&count[islice], &countTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      }
      if (countTotal < 10) {
	printWarningForTracking("IBS suppressed due to fewer than 10 particles in slice.",
				"This may happen for end slices and may be resolved using more particles");
	zeroslice(islice, IBS);
	continue;
      }
#else
      istart = iend;
      iend += count[islice];
      if (count[islice] < 10) {
	printWarningForTracking("IBS suppressed due to fewer than 10 particles in slice.",
				"This may happen for end slices and may be resolved using more particles");
	zeroslice(islice, IBS);
	continue;
      }
#endif

      if (isSlave || !distributedBeam) {
        computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
      }
      if (IBS->verbose > 1) {
	printf("Finished computation of parameters for slice %ld for bunch %ld\n", islice, iBucket);
        fflush(stdout);
      }
#if USE_MPI
      if (IBS->verbose > 2) {
        if (myid == 0) {
          long i, j;
          MPI_Recv(&aveCoord[0], 6, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus);
          MPI_Recv(&S[0][0], 36, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus);
          printf("aveCoord = %le, %le, %le, %le, %le, %le\n",
                 aveCoord[0], aveCoord[1], aveCoord[2],
                 aveCoord[3], aveCoord[4], aveCoord[5]);
          for (i = 0; i < 6; i++) {
            printf("S[%ld]: ", i);
            for (j = 0; j < 6; j++)
              printf("  %le", S[i][j]);
            printf("\n");
          }
          fflush(stdout);
        }
        if (myid == 1) {
          MPI_Send(&aveCoord[0], 6, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
          MPI_Send(&S[0][0], 36, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
#endif

      if (isSlave || !distributedBeam) {
	if (IBS->verbose > 3) {
	  printf("Computing growth rate for slice %ld\n", islice);
	  fflush(stdout);
	}
        /*** Should update eta[i] here to use values from the sigma matrix ****/
        IBS->emitx0[islice] = correctedEmittance(S, eta, 0, 1, &betax0, &alphax0);
        IBS->emity0[islice] = correctedEmittance(S, eta, 2, 3, &betay0, &alphay0);
        IBS->emitl0[islice] = SAFE_SQRT(S[4][4] * S[5][5] - sqr(S[4][5]));
        IBS->sigmaDelta0[islice] = sqrt(S[5][5]);
        if (IBS->nslice == 1)
          bLength = sqrt(S[4][4]) * c_mks;
        IBS->sigmaz0[islice] = bLength;
	if (IBS->verbose > 3) {
	  printf("slice %ld, bLength=%le\n", islice, bLength);
	  fflush(stdout);
	}
	
        if (!IBS->forceMatchedTwiss) {
          forth_propagate_twiss(IBS, islice, betax0, alphax0, betay0, alphay0, run);
	  if (IBS->verbose > 3) {
	    printf("slice %ld, forth_propagate_twiss returned\n", islice);
	    fflush(stdout);
	  }
	}
#if USE_MPI
        IBS->icharge[islice] = (IBS->charge * countTotal) / npTotal;
	if (IBS->verbose > 3) {
	  printf("slice %ld, particles=%ld of %ld, charge=%le\n", islice, countTotal, npTotal, IBS->icharge[islice]);
	  fflush(stdout);
	}
#else
        IBS->icharge[islice] = IBS->charge * (double)count[islice] / (double)np;
	if (IBS->verbose > 3) {
	  printf("slice %ld, particles=%ld of %ld, charge=%le\n", islice, count[islice], np, IBS->icharge[islice]);
	  fflush(stdout);
	}
#endif
#if USE_MPI
        elemCount = 0;
        if (n_processors > 3)
          elemCount = IBS->elements / (n_processors - 1);
        if (elemCount > 5 && IBS->parallelIntegration) {
          elemOffset = (myid - 1) * elemCount;
          if (myid == (n_processors - 1))
            elemCount = IBS->elements - (n_processors - 2) * elemCount;
          if (elemOffset > 0) {
            elemOffset--;
            elemCount++;
          }
	  if (IBS->verbose > 3) {
	    printf("slice %ld, computing rate\n", islice);
	    fflush(stdout);
	  }
          IBSRate(fabs(IBS->icharge[islice] / particleCharge),
                  elemCount, 1, 0, IBS->isRing,
                  IBS->emitx0[islice], IBS->emity0[islice], IBS->sigmaDelta0[islice], bLength,
                  IBS->s + elemOffset, IBS->pCentral, IBS->betax[islice] + elemOffset, IBS->alphax[islice] + elemOffset,
                  IBS->betay[islice] + elemOffset, IBS->alphay[islice] + elemOffset,
                  IBS->etax + elemOffset, IBS->etaxp + elemOffset, IBS->etay + elemOffset, IBS->etayp + elemOffset,
                  IBS->xRateVsS[islice] + elemOffset, IBS->yRateVsS[islice] + elemOffset, IBS->zRateVsS[islice] + elemOffset,
                  &(IBS->xGrowthRate[islice]), &(IBS->yGrowthRate[islice]), &(IBS->zGrowthRate[islice]), 1,
                  IBS->s[IBS->elements - 1] - IBS->s[0]);
	  if (IBS->verbose > 3) {
	    printf("slice %ld, accumulating rate\n", islice);
	    fflush(stdout);
	  }
          MPI_Allreduce(&IBS->xGrowthRate[islice], &mpiTemp, 1, MPI_DOUBLE, MPI_SUM, workers);
          IBS->xGrowthRate[islice] = mpiTemp;
          MPI_Allreduce(&IBS->yGrowthRate[islice], &mpiTemp, 1, MPI_DOUBLE, MPI_SUM, workers);
          IBS->yGrowthRate[islice] = mpiTemp;
          MPI_Allreduce(&IBS->zGrowthRate[islice], &mpiTemp, 1, MPI_DOUBLE, MPI_SUM, workers);
          IBS->zGrowthRate[islice] = mpiTemp;
	  if (IBS->verbose > 3) {
	    printf("slice %ld, rate=%le, %le, %le\n", islice, IBS->xGrowthRate[islice], IBS->yGrowthRate[islice], IBS->zGrowthRate[islice]);
	    fflush(stdout);
	  }
        } else {
          IBSRate(fabs(IBS->icharge[islice] / particleCharge),
                  IBS->elements, 1, 0, IBS->isRing,
                  IBS->emitx0[islice], IBS->emity0[islice], IBS->sigmaDelta0[islice], bLength,
                  IBS->s, IBS->pCentral, IBS->betax[islice], IBS->alphax[islice], IBS->betay[islice], IBS->alphay[islice],
                  IBS->etax, IBS->etaxp, IBS->etay, IBS->etayp,
                  IBS->xRateVsS[islice], IBS->yRateVsS[islice], IBS->zRateVsS[islice],
                  &(IBS->xGrowthRate[islice]), &(IBS->yGrowthRate[islice]), &(IBS->zGrowthRate[islice]), 1, -1);
	  if (IBS->verbose > 3) {
	    printf("slice %ld, rate=%le, %le, %le\n", islice, IBS->xGrowthRate[islice], IBS->yGrowthRate[islice], IBS->zGrowthRate[islice]);
	    fflush(stdout);
	  }
        }
#else
        IBSRate(fabs(IBS->icharge[islice] / particleCharge),
                IBS->elements, 1, 0, IBS->isRing,
                IBS->emitx0[islice], IBS->emity0[islice], IBS->sigmaDelta0[islice], bLength,
                IBS->s, IBS->pCentral, IBS->betax[islice], IBS->alphax[islice], IBS->betay[islice], IBS->alphay[islice],
                IBS->etax, IBS->etaxp, IBS->etay, IBS->etayp,
                IBS->xRateVsS[islice], IBS->yRateVsS[islice], IBS->zRateVsS[islice],
                &(IBS->xGrowthRate[islice]), &(IBS->yGrowthRate[islice]), &(IBS->zGrowthRate[islice]), 1, -1);
	if (IBS->verbose > 3) {
	  printf("slice %ld, rate=%le, %le, %le\n", islice, IBS->xGrowthRate[islice], IBS->yGrowthRate[islice], IBS->zGrowthRate[islice]);
	  fflush(stdout);
	}
#endif
        IBS->xGrowthRate[islice] *= IBS->factor;
        IBS->yGrowthRate[islice] *= IBS->factor;
        IBS->zGrowthRate[islice] *= IBS->factor;
      }
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (IBS->verbose > 2 ) {
      printf("Finished computation of growth rates for slice %ld of %ld\n", islice, IBS->nslice);
      fflush(stdout);
    }
    }
    if (IBS->verbose > 2 ) {
      printf("Finished computation of growth rates for all slices\n");
      fflush(stdout);
    }
    
    iend = 0;
    for (islice = 0; islice < IBS->nslice; islice++) {
      if (IBS->verbose > 1) {
	printf("Starting application of IBS effect for slice %ld for bunch %ld\n", islice, iBucket);
	fflush(stdout);
      }
#if USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (!IBS->smooth) {
#if USE_MPI
        if (myid == 0) {
          double buffer[3];
          MPI_Status status;
          MPI_Recv(&buffer[0], 3, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
          IBS->xGrowthRate[islice] = buffer[0];
          IBS->yGrowthRate[islice] = buffer[1];
          IBS->zGrowthRate[islice] = buffer[2];
        } else if (myid == 1) {
          double buffer[3];
          buffer[0] = IBS->xGrowthRate[islice];
          buffer[1] = IBS->yGrowthRate[islice];
          buffer[2] = IBS->zGrowthRate[islice];
          MPI_Send(&buffer[0], 3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
#endif
        if (IBS->xGrowthRate[islice] < 0)
          bombElegant("IBSCATTER with SMOOTH=0 failed because horizontal growth rate is negative", NULL);
        if (IBS->yGrowthRate[islice] < 0)
          bombElegant("IBSCATTER with SMOOTH=0 failed because vertical growth rate is negative", NULL);
        if (IBS->zGrowthRate[islice] < 0)
          bombElegant("IBSCATTER with SMOOTH=0 failed because longitudinal growth rate is negative", NULL);
      }
      if (isSlave || !distributedBeam) {
        istart = iend;
        iend += count[islice];

        zRate[1] = 1. + IBS->dT * IBS->zGrowthRate[islice];
        if (islice == 0)
          zRate[0] = zRate[1];
        else
          zRate[0] = 1. + IBS->dT * IBS->zGrowthRate[islice - 1];
        if (islice == IBS->nslice - 1)
          zRate[2] = zRate[1];
        else
          zRate[2] = 1. + IBS->dT * IBS->zGrowthRate[islice + 1];

        RNSigma[0] = RNSigma[1] = RNSigma[2] = 0;
        if (!IBS->smooth) {
          double G;
          computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
          if (IBS->do_x) {
            G = IBS->dT * IBS->xGrowthRate[islice];
            RNSigma[0] = sqrt((2 * G + G * G) * S[1][1]);
          }
          if (IBS->do_y) {
            G = IBS->dT * IBS->yGrowthRate[islice];
            RNSigma[1] = sqrt((2 * G + G * G) * S[3][3]);
          }
          if (IBS->do_z) {
            G = IBS->dT * IBS->zGrowthRate[islice];
            RNSigma[2] = sqrt((2 * G + G * G) * S[5][5]);
          }
          if (index == NULL)
            bombElegant("index array is NULL in track_IBS. Seek professional help!", NULL);
          for (icoord = 1, ihcoord = 0; icoord < 6; icoord += 2, ihcoord++) {
            if (RNSigma[ihcoord]) {
              /* RNSigmaCheck[ihcoord] = 0; */
              for (ipart = istart; ipart < iend; ipart++) {
                randomNumber = gauss_rn_lim(0.0, RNSigma[ihcoord], 3.0, random_2);
                part[index[ipart]][icoord] += randomNumber;
                /*
                RNSigmaCheck[ihcoord] += sqr(randomNumber);
		*/
              }
              /*
              RNSigmaCheck[ihcoord] = sqrt(RNSigmaCheck[ihcoord]/(double)(iend-istart)/S[icoord][icoord]+1.);
	      */
            }
          }
          /*
            fprintf(stdout,"s=%g,islice=%ld,istart=%ld,iend=%ld,checkz=%g\n", 
            IBS->s[IBS->elements-1], islice, istart, iend, RNSigmaCheck[2]);
            */
        } else {
          /* inflate each emittance by the prescribed factor */
          inflateEmittance(part, Po, eta, 0, istart, iend, index, (1. + IBS->dT * IBS->xGrowthRate[islice]));
          inflateEmittance(part, Po, eta, 2, istart, iend, index, (1. + IBS->dT * IBS->yGrowthRate[islice]));
          inflateEmittanceZ(part, Po, IBS->isRing, tLength, istart, iend, index, zRate, IBS->dT);
        }
        /* update beam emittance information after IBS scatter for IBSCATTER */
        if (!IBS->isRing) {
          computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
          IBS->emitx[islice] = correctedEmittance(S, eta, 0, 1, 0, 0);
          IBS->emity[islice] = correctedEmittance(S, eta, 2, 3, 0, 0);
          IBS->emitl[islice] = SAFE_SQRT(S[4][4] * S[5][5] - sqr(S[4][5]));
          IBS->sigmaDelta[islice] = sqrt(S[5][5]);
          IBS->sigmaz[islice] = sqrt(S[4][4]);
        }
      }
    }

    if (IBS->verbose) {
#if USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
      if (myid == 0) {
#endif
        printf("Done applying IBS effects for bunch %ld\n", iBucket);
        fflush(stdout);
#if USE_MPI
      }
#endif
    }

    if (isSlave || !distributedBeam) {
      if (index) {
        free(index);
        index = NULL;
      }
      if (count) {
        free(count);
        count = NULL;
      }
      if (nBuckets != 1) {
        /* copy back to original buffer */
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }
  }

#if USE_MPI
  if (IBS->filename) {
    if (myid == 1) {
      /* processor 1 sends data back to master for output */
      if (MPI_Send(&IBS->icharge[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->emitx0[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->emity0[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->emitl0[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->sigmaDelta0[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->sigmaz0[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->xGrowthRate[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->xGrowthRate[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Send(&IBS->zGrowthRate[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("Error: MPI_Send returns error sending IBS data from processor 1\n");
        bombElegant("Communication error", NULL);
      }
      if (!IBS->isRing &&
          (MPI_Send(&IBS->emitx[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
           MPI_Send(&IBS->emity[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
           MPI_Send(&IBS->emitl[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
           MPI_Send(&IBS->sigmaDelta[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS ||
           MPI_Send(&IBS->sigmaz[0], IBS->nslice, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD) != MPI_SUCCESS)) {
        printf("Error: MPI_Send returns error sending IBS data from processor 1\n");
        bombElegant("Communication error", NULL);
      }
    } else if (myid == 0) {
      /* processor 0 (master) receives data from processor 1 */
      if (MPI_Recv(&IBS->icharge[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->emitx0[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->emity0[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->emitl0[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->sigmaDelta0[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->sigmaz0[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->xGrowthRate[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->xGrowthRate[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
          MPI_Recv(&IBS->zGrowthRate[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS) {
        printf("Error: MPI_Recv returns error retrieving IBS data from processor 1\n");
        bombElegant("Communication error", NULL);
      }
      if (!IBS->isRing &&
          (MPI_Recv(&IBS->emitx[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
           MPI_Recv(&IBS->emity[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
           MPI_Recv(&IBS->emitl[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
           MPI_Recv(&IBS->sigmaDelta[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS ||
           MPI_Recv(&IBS->sigmaz[0], IBS->nslice, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpiStatus) != MPI_SUCCESS)) {
        printf("Error: MPI_Recv returns error retrieving IBS data from processor 1\n");
        bombElegant("Communication error", NULL);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (IBS->filename && isMaster) {
    if (!(IBS->isInit)) {
      SDDS_IBScatterSetup(&(IBS->outPage), IBS->filename, SDDS_BINARY, 1, "IBS scatter growth rate output",
                          run, eptr, "ibs_tracking", IBS->isRing);
      IBS->isInit = 1;
    }

    if ((int)i_pass / IBS->interval > (IBS->doOut)) {
      (IBS->doOut)++;
      reset_IBS_output(element);
    }
    if (IBS->output) {
      dump_IBScatter(&(IBS->outPage), IBS, i_pass);
      IBS->output = 0;
    }
  }

  if (i_pass == n_passes - 1)
    free_IBS(IBS);

  if (time && time != time0)
    free(time);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

  if (IBS->verbose) {
#if USE_MPI
    if (myid == 0) {
#endif
      printf("Returning from IBSCATTER\n");
      fflush(stdout);
#if USE_MPI
    }
#endif
  }

  return;
}

void inflateEmittance(double **coord, double Po, double eta[4],
                      long offset, long istart, long iend, long *index, double factor) {
  long ipart, np;
  double factorSqrt, dpc = 0, c[2] = {0, 0}, cp[2] = {0, 0};

#if USE_MPI
  long npTotal;
  np = iend - istart;
  MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, workers);
  if (!npTotal)
    return;
#else
  np = iend - istart;
  if (!np)
    return;
#endif
  factorSqrt = sqrt(factor);

  for (ipart = istart; ipart < iend; ipart++) {
    c[0] += coord[index[ipart]][offset + 0];
    c[1] += coord[index[ipart]][offset + 1];
    dpc += coord[index[ipart]][5];
  }
#if USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &c[0], 2, MPI_DOUBLE, MPI_SUM, workers);
  MPI_Allreduce(MPI_IN_PLACE, &dpc, 1, MPI_DOUBLE, MPI_SUM, workers);
  c[0] /= npTotal;
  c[1] /= npTotal;
  dpc /= npTotal;
#else
  c[0] /= np;
  c[1] /= np;
  dpc /= np;
#endif
  for (ipart = istart; ipart < iend; ipart++) {
    cp[0] = eta[offset + 0] * dpc;
    cp[1] = eta[offset + 1] * dpc;
    coord[index[ipart]][offset + 0] = (coord[index[ipart]][offset + 0] - c[0] - cp[0]) * factorSqrt + c[0] + cp[0];
    coord[index[ipart]][offset + 1] = (coord[index[ipart]][offset + 1] - c[1] - cp[1]) * factorSqrt + c[1] + cp[1];
  }
}

void inflateEmittanceZ(double **coord, double Po, long isRing, double dt,
                       long istart, long iend, long *index, double zRate[3], double Duration) {
  long ipart, np;
  int64_t i;
  double c0, tc, dpc, *time, p, beta0, beta1;

#if USE_MPI
  long npTotal;
  np = iend - istart;
  MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, workers);
  if (!npTotal)
    return;
#else
  np = iend - istart;
  if (!np)
    return;
#endif

  time = tmalloc(sizeof(*time) * np);
  for (i = dpc = tc = 0; i < np; i++) {
    ipart = i + istart;
    dpc += coord[index[ipart]][5];
    p = Po * (1 + coord[index[ipart]][5]);
    beta0 = p / sqrt(p * p + 1);
    time[i] = coord[index[ipart]][4] / (beta0 * c_mks);
    tc += time[i];
  }
#if USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &tc, 1, MPI_DOUBLE, MPI_SUM, workers);
  MPI_Allreduce(MPI_IN_PLACE, &dpc, 1, MPI_DOUBLE, MPI_SUM, workers);
  tc /= npTotal;
  dpc /= npTotal;
#else
  tc /= np;
  dpc /= np;
#endif

  for (i = 0; i < np; i++) {
    ipart = i + istart;

    if (time[i] > tc)
      c0 = (time[i] - tc) / dt * (zRate[2] - zRate[1]) + zRate[1];
    else
      c0 = (time[i] - tc) / dt * (zRate[1] - zRate[0]) + zRate[1];

    p = Po * (1 + coord[index[ipart]][5]);
    beta0 = p / sqrt(p * p + 1);
    coord[index[ipart]][5] = (coord[index[ipart]][5] - dpc) * c0 + dpc;
    p = Po * (1 + coord[index[ipart]][5]);
    beta1 = p / sqrt(p * p + 1);
    coord[index[ipart]][4] += (beta1 - beta0) * Duration * c_mks / 2;
  }
  free(time);
  return;
}

/* Set twiss parameter arrays etc. */
void init_IBS(ELEMENT_LIST *element) {
  long count, nElements, i, j, init = 1;
  double startRingPos, finalPos = 0;
  double s0, s1, dt, delta_s, p0, gamma;
  ELEMENT_LIST *element0, *elementStartRing, *eptr = NULL;
  IBSCATTER *IBS = NULL;
  TWISS *tp;

  if (!element->twiss)
    bombElegant("Twiss parameters must be calculated befor IBS tracking.", NULL);

  /* Find out start point of ring */
  element0 = elementStartRing = element;
  startRingPos = 0;

  count = 0;
  while (element) {
    if (element->type == T_RECIRC) {
      startRingPos = element->end_pos;
      elementStartRing = element->succ;
      count++;
      break;
    }
    count++;
    element = element->succ;
  }

  if (elementStartRing != element0) {
    element = elementStartRing;
  } else {
    element = element0;
    count = 0;
  }

  nElements = 0;
  s1 = startRingPos;
  dt = delta_s = 0.;
  while (element) {
    s0 = s1;
    s1 = element->end_pos;
    if (s1 > s0) {
      if (element->pred)
        p0 = (element->Pref_output + element->pred->Pref_output) / 2.;
      else
        p0 = element->Pref_output;
      gamma = sqrt(p0 * p0 + 1);
      dt += (s1 - s0) * gamma / p0 / c_mks;
      delta_s += (s1 - s0);
    }

    if (element->type == T_IBSCATTER) {
      IBS = (IBSCATTER *)element->p_elem;
      IBS->revolutionLength = delta_s;
      if (nElements < 2)
        bombElegant("you need at least 2 other elements between IBSCATTERS, check twiss file for clue", NULL);
      IBS->dT = dt;
      dt = delta_s = 0.;
      IBS->elements = nElements;
      IBS->offset = count;
      IBS->output = 1;
      count = count + nElements + 1;
      if (!(IBS->name = SDDS_Calloc(nElements, sizeof(*(IBS->name)))) ||
          !(IBS->s = SDDS_Calloc(nElements, sizeof(*(IBS->s)))) ||
          !(IBS->pCentral = SDDS_Calloc(nElements, sizeof(*(IBS->pCentral)))))
        bombElegant("memory allocation failure in init_IBS", NULL);
      if (!(IBS->etax = SDDS_Realloc(IBS->etax, sizeof(*(IBS->etax)) * nElements)) ||
          !(IBS->etaxp = SDDS_Realloc(IBS->etaxp, sizeof(*(IBS->etaxp)) * nElements)) ||
          !(IBS->etay = SDDS_Realloc(IBS->etay, sizeof(*(IBS->etay)) * nElements)) ||
          !(IBS->etayp = SDDS_Realloc(IBS->etayp, sizeof(*(IBS->etayp)) * nElements)))
        bombElegant("memory allocation failure in init_IBS", NULL);
      if (!(IBS->icharge = SDDS_Realloc(IBS->icharge, sizeof(*(IBS->icharge)) * IBS->nslice)) ||
          !(IBS->emitx0 = SDDS_Realloc(IBS->emitx0, sizeof(*(IBS->emitx0)) * IBS->nslice)) ||
          !(IBS->emity0 = SDDS_Realloc(IBS->emity0, sizeof(*(IBS->emity0)) * IBS->nslice)) ||
          !(IBS->emitl0 = SDDS_Realloc(IBS->emitl0, sizeof(*(IBS->emitl0)) * IBS->nslice)) ||
          !(IBS->sigmaz0 = SDDS_Realloc(IBS->sigmaz0, sizeof(*(IBS->sigmaz0)) * IBS->nslice)) ||
          !(IBS->sigmaDelta0 = SDDS_Realloc(IBS->sigmaDelta0, sizeof(*(IBS->sigmaDelta0)) * IBS->nslice)) ||
          !(IBS->emitx = SDDS_Realloc(IBS->emitx, sizeof(*(IBS->emitx)) * IBS->nslice)) ||
          !(IBS->emity = SDDS_Realloc(IBS->emity, sizeof(*(IBS->emity)) * IBS->nslice)) ||
          !(IBS->emitl = SDDS_Realloc(IBS->emitl, sizeof(*(IBS->emitl)) * IBS->nslice)) ||
          !(IBS->sigmaz = SDDS_Realloc(IBS->sigmaz, sizeof(*(IBS->sigmaz)) * IBS->nslice)) ||
          !(IBS->sigmaDelta = SDDS_Realloc(IBS->sigmaDelta, sizeof(*(IBS->sigmaDelta)) * IBS->nslice)) ||
          !(IBS->xGrowthRate = SDDS_Realloc(IBS->xGrowthRate, sizeof(*(IBS->xGrowthRate)) * IBS->nslice)) ||
          !(IBS->yGrowthRate = SDDS_Realloc(IBS->yGrowthRate, sizeof(*(IBS->yGrowthRate)) * IBS->nslice)) ||
          !(IBS->zGrowthRate = SDDS_Realloc(IBS->zGrowthRate, sizeof(*(IBS->zGrowthRate)) * IBS->nslice)))
        bombElegant("memory allocation failure in init_IBS", NULL);
      if (!(IBS->betax = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->alphax = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->betay = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->alphay = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->xRateVsS = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->yRateVsS = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)) ||
          !(IBS->zRateVsS = (double **)czarray_2d(sizeof(double), IBS->nslice, nElements)))
        bombElegant("memory allocation failure in init_IBS", NULL);

      for (i = 0; i < IBS->elements; i++) {
        cp_str(&IBS->name[i], elementStartRing->name);
        IBS->s[i] = elementStartRing->end_pos;
        IBS->pCentral[i] = elementStartRing->Pref_output;
        IBS->etax[i] = elementStartRing->twiss->etax;
        IBS->etaxp[i] = elementStartRing->twiss->etapx;
        IBS->etay[i] = elementStartRing->twiss->etay;
        IBS->etayp[i] = elementStartRing->twiss->etapy;
        if (init)
          twiss0 = tmalloc(sizeof(*twiss0) * IBS->nslice);
        /*
          twiss0 = SDDS_Realloc(twiss0, sizeof(*twiss0)*IBS->nslice);
        */
        for (j = 0; j < IBS->nslice; j++) {
          if (init) {
            tp = &(twiss0[j]);
            copy_twiss(tp, elementStartRing->twiss);
          }
          IBS->betax[j][i] = elementStartRing->twiss->betax;
          IBS->alphax[j][i] = elementStartRing->twiss->alphax;
          IBS->betay[j][i] = elementStartRing->twiss->betay;
          IBS->alphay[j][i] = elementStartRing->twiss->alphay;
        }
        init = 0;
        elementStartRing = elementStartRing->succ;
      }
      IBS->elem = tmalloc(sizeof(*(IBS->elem)));
      IBS->elem->pred = IBS->elem->succ = NULL;
      elementStartRing = elementStartRing->pred;
      for (i = IBS->elements; i > 0; i--) {
        add_element(IBS->elem, elementStartRing);
        eptr = IBS->elem->succ;
        /* copy input energy to newly added element. Necessary if beamline contains RF cavity */
        eptr->Pref_input = elementStartRing->Pref_input;
        eptr->Pref_output = elementStartRing->Pref_output;
        elementStartRing = elementStartRing->pred;
      }
      IBS->elem = eptr;
      eptr->pred = eptr->pred->succ = NULL;
      nElements = -1;
      elementStartRing = element->succ;
    }
    nElements++;
    finalPos = element->end_pos;
    element = element->succ;
  }

  if (finalPos != IBS->s[IBS->elements - 1])
    bombElegant("You must have IBSCATTER at the end of the RING", NULL);
  element = element0;
  return;
}

void free_IBS(IBSCATTER *IBS) {
  free(IBS->name);
  free(IBS->s);
  free(IBS->pCentral);
  free(IBS->icharge);
  free(IBS->etax);
  free(IBS->etaxp);
  free(IBS->etay);
  free(IBS->etayp);
  free(IBS->emitx0);
  free(IBS->emity0);
  free(IBS->emitl0);
  free(IBS->emitx);
  free(IBS->emity);
  free(IBS->emitl);
  free(IBS->sigmaz0);
  free(IBS->sigmaDelta0);
  free(IBS->sigmaz);
  free(IBS->sigmaDelta);
  free(IBS->xGrowthRate);
  free(IBS->yGrowthRate);
  free(IBS->zGrowthRate);
  free_czarray_2d((void **)IBS->betax, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->betay, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->alphax, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->alphay, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->xRateVsS, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->yRateVsS, IBS->nslice, IBS->elements);
  free_czarray_2d((void **)IBS->zRateVsS, IBS->nslice, IBS->elements);
  free_elements1(IBS->elem);

  IBS->name = NULL;
  IBS->s = NULL;
  IBS->pCentral = NULL;
  IBS->icharge = NULL;
  IBS->etax = NULL;
  IBS->etaxp = NULL;
  IBS->etay = NULL;
  IBS->etayp = NULL;
  IBS->emitx0 = NULL;
  IBS->emity0 = NULL;
  IBS->emitl0 = NULL;
  IBS->emitx = NULL;
  IBS->emity = NULL;
  IBS->emitl = NULL;
  IBS->sigmaz0 = NULL;
  IBS->sigmaDelta0 = NULL;
  IBS->sigmaz = NULL;
  IBS->sigmaDelta = NULL;
  IBS->xGrowthRate = NULL;
  IBS->yGrowthRate = NULL;
  IBS->zGrowthRate = NULL;
  IBS->betax = IBS->betay = IBS->alphax = IBS->alphay = NULL;
  IBS->xRateVsS = IBS->yRateVsS = IBS->zRateVsS = NULL;
  IBS->elem = NULL;
  return;
}

void slicebeam(double **coord, int64_t np, double *time, double Po, long nslice, long *index, long *count, double *dt) {
  long j, islice, total;
  int64_t i;
  double tMaxAll, tMinAll;
  long *timeIndex;

  /* Count the number of particles in each slice and the keep the slice index for each particle */
  for (i = 0; i < np; i++)
    index[i] = -1;
  for (islice = 0; islice < nslice; islice++)
    count[islice] = 0;

  if (nslice > 1) {
    /* find limits of bunch longitudinal coordinates */
    find_min_max(&tMinAll, &tMaxAll, time, np);
#if USE_MPI
    if (isSlave || !distributedBeam)
      find_global_min_max(&tMinAll, &tMaxAll, np, workers);
      /* printf("tMinAll=%le, tMaxAll=%le\n", tMinAll, tMaxAll); */
#endif

    *dt = (tMaxAll - tMinAll) / (double)nslice;
    timeIndex = tmalloc(sizeof(*timeIndex) * np);
    for (i = 0; i < np; i++) {
      if ((timeIndex[i] = (time[i] - tMinAll) / (*dt)) < 0)
        timeIndex[i] = 0;
      else if (timeIndex[i] >= nslice)
        timeIndex[i] = nslice - 1;
    }
    j = total = 0;
    for (islice = 0; islice < nslice; islice++) {
      for (i = 0; i < np; i++) {
        if (timeIndex[i] == islice) {
          count[islice]++;
          index[j++] = i;
        }
      }
      total += count[islice];
    }
    if (total != np) {
      printf("IBSCATTER: slice-beam (nslice=%ld), total (%ld) is not equal to np (%" PRId64 "). Report it to code developer.\n", nslice, total, np);
      for (islice = 0; islice < nslice; islice++)
        printf("count[%ld] = %ld\n", islice, count[islice]);
      bombElegant(NULL, NULL);
    }
    free(timeIndex);
  } else {
    /* simplified code for single slice */
    *dt = 1; /* never actually used when only 1 slice present */
    count[0] = np;
    for (i = 0; i < np; i++)
      index[i] = i;
  }
  return;
}

void zeroslice(long islice, IBSCATTER *IBS) {
  long i;
  IBS->emitx0[islice] = IBS->emity0[islice] = IBS->emitl0[islice] = 0.;
  IBS->sigmaz0[islice] = IBS->sigmaDelta0[islice] = 0.;
  IBS->emitx[islice] = IBS->emity[islice] = IBS->emitl[islice] = 0.;
  IBS->sigmaz[islice] = IBS->sigmaDelta[islice] = 0.;
  IBS->xGrowthRate[islice] = IBS->yGrowthRate[islice] = IBS->zGrowthRate[islice] = 0.;
  for (i = 0; i < IBS->elements; i++) {
    IBS->xRateVsS[islice][i] = IBS->yRateVsS[islice][i] = IBS->zRateVsS[islice][i] = 0.;
  }
  return;
}

long computeSliceParameters(double C[6], double S[6][6], double **part, long *index, long start, long end, double Po) {
  long i, j, k, i1;
  double *time, dt, dp, beta, P;
#if USE_MPI
  long countTotal, count0;
#endif

  time = tmalloc(sizeof(*time) * (end - start));

  for (j = 0; j < 6; j++) {
    C[j] = 0;
    for (k = 0; k < 6; k++)
      S[j][k] = 0;
  }

  /* compute centroid slice parameters */
  for (i = start; i < end; i++) {
    for (j = 0; j < 4; j++) {
      C[j] += part[index[i]][j];
    }
    P = Po * (1 + part[index[i]][5]);
    beta = P / sqrt(P * P + 1);
    i1 = i - start;
    time[i1] = part[index[i]][4] / (beta * c_mks);
    C[4] += time[i1];
    C[5] += part[index[i]][5];
  }

#if USE_MPI
  count0 = end - start;
  MPI_Allreduce(&count0, &countTotal, 1, MPI_LONG, MPI_SUM, workers);
  MPI_Allreduce(MPI_IN_PLACE, &C[0], 6, MPI_DOUBLE, MPI_SUM, workers);
  for (j = 0; j < 6; j++)
    C[j] /= countTotal;
#else
  for (j = 0; j < 6; j++) {
    C[j] /= (double)(end - start);
    /* printf("C[%ld] = %le\n", j, C[j]); */
  }
#endif

  for (i = start; i < end; i++) {
    for (j = 0; j < 6; j++)
      for (k = 0; k <= j; k++)
        if (!(j >= 4 && k >= 4))
          S[j][k] += (part[index[i]][j] - C[j]) * (part[index[i]][k] - C[k]);
    i1 = i - start;
    dt = time[i1] - C[4];
    S[4][4] += sqr(dt);
    dp = (part[index[i]][5] - C[5]);
    S[5][5] += sqr(dp);
    S[5][4] += dt * dp;
  }
#if USE_MPI
  for (j = 0; j < 6; j++) {
    MPI_Allreduce(MPI_IN_PLACE, S[j], j + 1, MPI_DOUBLE, MPI_SUM, workers);
    for (k = 0; k <= j; k++)
      S[k][j] = S[j][k] = S[j][k] / countTotal;
  }
#else
  for (j = 0; j < 6; j++)
    for (k = 0; k <= j; k++) {
      S[j][k] /= (double)(end - start);
      S[k][j] = S[j][k];
    }
#endif

  free(time);
  return 0;
}

void forth_propagate_twiss(IBSCATTER *IBS, long islice, double betax0, double alphax0,
                           double betay0, double alphay0, RUN *run) {
  ELEMENT_LIST *elem;
  double tune[2];
  long i, waists[2];
  TWISS *tp;

  tp = &(twiss0[islice]);

  if (IBS->verbose>4) {
    printf("running forth_propagate_twiss with betax=%le, alphax=%le, betay=%le, alphay=%le\n",
	   tp->betax, tp->alphax, tp->betay, tp->alphay);
    fflush(stdout);
  }

  propagate_twiss_parameters(tp, tune, waists, NULL, IBS->elem, run, NULL, NULL, NULL);

  if (IBS->verbose>4) {
    printf("returned from propagate_twiss_parameters\n");
    fflush(stdout);
  }

  elem = IBS->elem;
  for (i = 0; i < IBS->elements; i++) {
    IBS->betax[islice][i] = elem->twiss->betax;
    IBS->alphax[islice][i] = elem->twiss->alphax;
    IBS->betay[islice][i] = elem->twiss->betay;
    IBS->alphay[islice][i] = elem->twiss->alphay;
    elem = elem->succ;
  }

  if (IBS->verbose>4) {
    printf("Updated twiss function array\n");
    fflush(stdout);
  }

  tp->betax = betax0;
  tp->alphax = alphax0;
  tp->phix = 0;
  tp->etax = IBS->etax[0];
  tp->etapx = IBS->etaxp[0];
  tp->betay = betay0;
  tp->alphay = alphay0;
  tp->phiy = 0;
  tp->etay = IBS->etay[0];
  tp->etapy = IBS->etayp[0];

  if (IBS->verbose>4) {
    printf("Restored twiss function values in beamline\n");
    fflush(stdout);
  }

  return;
}

void reset_IBS_output(ELEMENT_LIST *element) {
  ELEMENT_LIST *element0;
  IBSCATTER *IBS;

  element0 = element;
  while (element) {
    if (element->type == T_IBSCATTER) {
      IBS = (IBSCATTER *)element->p_elem;
      IBS->output = 1;
    }
    element = element->succ;
  }
  element = element0;
}

#define IBSCATTER_RING_PARAMETERS 20
#define IBSCATTER_LINAC_PARAMETERS 27
static SDDS_DEFINITION ibscatter_print_parameter[IBSCATTER_LINAC_PARAMETERS] = {
  {"TotalSlice", "&parameter name=TotalSlice, type=long, description=\"Total number of Slices\" &end"},
  {"NSlice", "&parameter name=NSlice, type=long, description=\"The ith slice of the beam\" &end"},
  {"Charge", "&parameter name=Charge, type=double, units=\"nC\", description=\"Slice charge in nC\" &end"},
  {"Particles", "&parameter name=Particles, type=double, description=\"Number of particles in slice\" &end"},
  {"s", "&parameter name=s, type=double, units=\"m\", description=\"IBScatter element location in beamline\" &end"},
  {"Pass", "&parameter name=Pass, type=long, description=\"Pass number\" &end"},
  {"StartPoint", "&parameter name=StartPoint, type=long, description=\"IStart point in the beamline\" &end"},
  {"Elements", "&parameter name=Elements, type=long, description=\"Number of elements to integrate\" &end"},
  {"ds", "&parameter name=ds, type=double, units=\"m\", description=\"Integrated length for IBS rate\" &end"},
  {"dt", "&parameter name=dt, type=double, units=\"m\", description=\"Integrated time for IBS scattering\" &end"},
  {"xGrowthRate", "&parameter name=xGrowthRate, symbol=\"g$bIBS,x$n\", units=\"1/s\", type=double, description=\"Accumulated IBS emittance growth rate in the horizontal plane\" &end"},
  {"yGrowthRate", "&parameter name=yGrowthRate, symbol=\"g$bIBS,y$n\", units=\"1/s\", type=double, description=\"Accumulated IBS emittance growth rate in the vertical plane\" &end"},
  {"zGrowthRate", "&parameter name=zGrowthRate, symbol=\"g$bIBS,z$n\", units=\"1/s\", type=double, description=\"Accumulated IBS emittance growth rate in the longitudinal plane\" &end"},
  {"enx0", "&parameter name=enx0, symbol=\"$gge$r$bx$n\", units=\"m$be$nc $gp$rm\", type=double, description=\"Normalized initial horizontal emittance\" &end"},
  {"eny0", "&parameter name=eny0, symbol=\"$gge$r$by$n\", units=\"m$be$nc $gp$rm\", type=double, description=\"Normalized initial vertical emittance\" &end"},
  {"emitx0", "&parameter name=emitx0, symbol=\"$ge$r$bx,Input$n\", units=\"$gp$rm\", type=double, description=\"Initial horizontal emittance\" &end"},
  {"emity0", "&parameter name=emity0, symbol=\"$ge$r$by,Input$n\", units=\"$gp$rm\", type=double, description=\"Initial vertical emittance\" &end"},
  {"emitl0", "&parameter name=emitl0, symbol=\"$ge$r$bl,Input$n\", units=\"s\", type=double, description=\"Initial longitudinal emittance\" &end"},
  {"sigmaDelta0", "&parameter name=sigmaDelta0, symbol=\"$gs$r$bd,Input$n\", type=double, description=\"Initial momentum spread\" &end"},
  {"sigmaz0", "&parameter name=sigmaz0, symbol=\"$gs$r$bz,Input$n\", units=m, type=double, description=\"Initial bunch length\" &end"},
  {"enx", "&parameter name=enx, symbol=\"$gge$r$bx$n\", units=\"m$be$nc $gp$rm\", type=double, description=\"Normalized horizontal emittance with IBS\" &end"},
  {"eny", "&parameter name=eny, symbol=\"$gge$r$by$n\", units=\"m$be$nc $gp$rm\", type=double, description=\"Normalized vertical emittance with IBS\" &end"},
  {"emitx", "&parameter name=emitx, symbol=\"$ge$r$bx$n\", units=\"$gp$rm\", type=double, description=\"Horizontal emittance with IBS\" &end"},
  {"emity", "&parameter name=emity, symbol=\"$ge$r$by$n\", units=\"$gp$rm\", type=double, description=\"Vertical emittance with IBS\" &end"},
  {"emitl", "&parameter name=emitl, symbol=\"$ge$r$bl$n\", units=\"s\", type=double, description=\"Longitudinal emittance with IBS\" &end"},
  {"sigmaDelta", "&parameter name=sigmaDelta, symbol=\"$gs$r$bd$n\", type=double, description=\"Momentum spread with IBS\" &end"},
  {"sigmaz", "&parameter name=sigmaz, symbol=\"$gs$r$bz$n\", units=m, type=double, description=\"Bunch length with IBS\" &end"},
};

#define IBSCATTER_COLUMNS 13
static SDDS_DEFINITION ibscatter_print_column[IBSCATTER_COLUMNS] = {
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"dIBSRatex", "&column name=dIBSRatex, units=\"1/(m s)\", type=double, description=\"Horizontal IBS Emittance Growth Rate\" &end"},
  {"dIBSRatey", "&column name=dIBSRatey, units=\"1/(m s)\", type=double, description=\"Vertical IBS Emittance Growth Rate\" &end"},
  {"dIBSRatez", "&column name=dIBSRatez, units=\"1/(m s)\", type=double, description=\"Longitudinal IBS Emittance Growth Rate\" &end"},
  {"betax", "&column name=betax, type=double, units=m, symbol=\"$gb$r$bx$n\", description=\"Horizontal beta-function\" &end"},
  {"alphax", "&column name=alphax, type=double, symbol=\"$ga$r$bx$n\", description=\"Horizontal alpha-function\" &end"},
  {"etax", "&column name=etax, type=double, units=m, symbol=\"$gc$r$bx$n\", description=\"Horizontal dispersion\" &end"},
  {"etaxp", "&column name=etaxp, type=double, symbol=\"$gc$r$bx$n$a'$n\", description=\"Slope of horizontal dispersion\" &end"},
  {"betay", "&column name=betay, type=double, units=m, symbol=\"$gb$r$by$n\", description=\"Vertical beta-function\" &end"},
  {"alphay", "&column name=alphay, type=double, symbol=\"$ga$r$by$n\", description=\"Vertical alpha-function\" &end"},
  {"etay", "&column name=etay, type=double, units=m, symbol=\"$gc$r$by$n\", description=\"Vertical dispersion\" &end"},
  {"etayp", "&column name=etayp, type=double, symbol=\"$gc$r$by$n$a'$n\", description=\"Slope of vertical dispersion\" &end"},
};

void SDDS_IBScatterSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                         RUN *run, ELEMENT_LIST *element, char *caller, long isRing) {
  log_entry("SDDS_IBScatterSetup");

  /*
    if (isRing)
      SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                            ibscatter_print_parameter, IBSCATTER_RING_PARAMETERS, ibscatter_print_column, IBSCATTER_COLUMNS,
                            caller, SDDS_EOS_NEWFILE|SDDS_EOS_COMPLETE);
    else
      SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                            ibscatter_print_parameter, IBSCATTER_LINAC_PARAMETERS, ibscatter_print_column, IBSCATTER_COLUMNS,
                            caller, SDDS_EOS_NEWFILE|SDDS_EOS_COMPLETE);
*/
  fflush(stdout);
  filename = compose_filename_occurence(filename, run->rootname, element->occurence);
  printf("Setting up IBS file %s for rootname %s and occurence %ld\n",
         filename, run->rootname, element->occurence);
  if (isRing)
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, run->runfile, run->lattice,
                            ibscatter_print_parameter, IBSCATTER_RING_PARAMETERS, ibscatter_print_column, IBSCATTER_COLUMNS,
                            caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
  else
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, run->runfile, run->lattice,
                            ibscatter_print_parameter, IBSCATTER_LINAC_PARAMETERS, ibscatter_print_column, IBSCATTER_COLUMNS,
                            caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
  log_exit("SDDS_IBScatterSetup");
}

void dump_IBScatter(SDDS_TABLE *SDDS_table, IBSCATTER *IBS, long pass) {
  long islice, i;
  double gamma;

  log_entry("dump_IBScatter");

  if (!IBS->elements)
    return;

  gamma = sqrt(ipow2(IBS->pCentral[IBS->elements - 1]) + 1.);
  for (islice = 0; islice < IBS->nslice; islice++) {
    if (!SDDS_StartTable(SDDS_table, IBS->elements)) {
      SDDS_SetError("Problem starting SDDS table (dump_IBScatter)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    for (i = 0; i < IBS->elements; i++) {
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, IBS->name[i], 1, IBS->s[i],
                             2, IBS->xRateVsS[islice][i], 3, IBS->yRateVsS[islice][i], 4, IBS->zRateVsS[islice][i],
                             5, IBS->betax[islice][i], 6, IBS->alphax[islice][i], 7, IBS->etax[i], 8, IBS->etaxp[i],
                             9, IBS->betay[islice][i], 10, IBS->alphay[islice][i], 11, IBS->etay[i], 12, IBS->etayp[i], -1)) {
        SDDS_SetError("Problem setting SDDS row values (dump_IBScatter)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }

    if ((!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Particles", fabs(IBS->icharge[islice] / particleCharge), NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Charge", IBS->icharge[islice] * 1e9, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "TotalSlice", IBS->nslice, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "NSlice", islice + 1, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "s", IBS->s[IBS->elements - 1], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Pass", pass, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "StartPoint", IBS->offset, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Elements", IBS->elements, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "ds", IBS->revolutionLength, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "dt", IBS->dT, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "xGrowthRate", IBS->xGrowthRate[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "yGrowthRate", IBS->yGrowthRate[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "zGrowthRate", IBS->zGrowthRate[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "enx0", IBS->emitx0[islice] * gamma, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "eny0", IBS->emity0[islice] * gamma, NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emitx0", IBS->emitx0[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emity0", IBS->emity0[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emitl0", IBS->emitl0[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "sigmaDelta0", IBS->sigmaDelta0[islice], NULL)) ||
        (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "sigmaz0", IBS->sigmaz0[islice], NULL))) {
      SDDS_SetError("Problem setting SDDS parameters (dump_IBScatter)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    if (!IBS->isRing) {
      if ((!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emitx", IBS->emitx[islice], NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emity", IBS->emity[islice], NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "emitl", IBS->emitl[islice], NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "enx", IBS->emitx[islice] * gamma, NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "eny", IBS->emity[islice] * gamma, NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "sigmaDelta", IBS->sigmaDelta[islice], NULL)) ||
          (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "sigmaz", IBS->sigmaz[islice] * c_mks, NULL))) {
        SDDS_SetError("Problem setting SDDS parameters (dump_IBScatter)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }

    if (!SDDS_WriteTable(SDDS_table)) {
      SDDS_SetError("Problem writing SDDS table (dump_IBScatter)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    SDDS_UpdatePage(SDDS_table, 0);
    if (!inhibitFileSync)
      SDDS_DoFSync(SDDS_table);
  }
  log_exit("dump_IBScatter");
}

void copy_twiss(TWISS *tp0, TWISS *tp1) {
  long i;
  tp0->betax = tp1->betax;
  tp0->alphax = tp1->alphax;
  tp0->phix = tp1->phix;
  tp0->etax = tp1->etax;
  tp0->etapx = tp1->etapx;
  tp0->apx = tp1->apx;
  tp0->betay = tp1->betay;
  tp0->alphay = tp1->alphay;
  tp0->phiy = tp1->phiy;
  tp0->etay = tp1->etay;
  tp0->etapy = tp1->etapy;
  tp0->apy = tp1->apy;
  tp0->Cx = tp1->Cx;
  tp0->Cy = tp1->Cy;
  tp0->periodic = tp1->periodic;
  for (i = 0; i < 6; i++)
    tp0->dI[i] = tp1->dI[i];
  return;
}
