/*************************************************************************\
* Copyright (c) 2024 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2024 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: particleTunes.c
 * purpose: Uses data collected from tracking to compute the per-particle tunes
 *          and output them to a file.
 *
 * Michael Borland, 2024
 */
#include "mdb.h"
#include "track.h"
#include "particleTunes.h"
#include "fftpackC.h"

char *tuneColumnName[4] = {"nux", "nuy", "nus", "nusp"};
char *JColumnName[3] = {"Jx", "Jy", "Js"};

double computeInvariant(double *q, double *qp, long turns, double tune)
{
  double sumq2, sumqp2, sumqqp, e2;
  double sumq, sumqp;
  long it;
  sumq2 = sumqp2 = sumqqp = sumq = sumqp = 0;
  if (tune<=0 || tune>=1)
    return DBL_MAX;
  /* Try to take an integer number of oscillations */
  if ((turns = ((long)(turns*tune))/tune)<=0)
    return DBL_MAX;
  for (it=0; it<turns; it++) {
    sumq  += q[it];
    sumqp += qp[it];
  }
  sumq  /= turns;
  sumqp /= turns;
  for (it=0; it<turns; it++) {
    sumq2 += sqr(q[it]-sumq);
    sumqp2 += sqr(qp[it]-sumqp);
    sumqqp += (q[it]-sumq)*(qp[it]-sumqp);
  }
  e2 = (sumq2*sumqp2 - sqr(sumqqp))/sqr(turns);
  if (e2>=0)
    return sqrt(e2)/2;
  else
    return DBL_MAX;
}

long setupParticleTunes
(
 NAMELIST_TEXT *nltext,
 RUN *run,
 VARY *control,
 PARTICLE_TUNES *ptunes
 )
{
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&particle_tunes, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &particle_tunes);

  if (!filename || !strlen(filename))
    bombElegant("supply filename", NULL);
  if (start_pid>end_pid)
    bombElegant("start_pid must not be greater than end_pid", NULL);
  if (pid_interval<1)
    bombElegant("pid_interval must not be less than 1", NULL);
  if (!include_x && !include_y && !include_s)
    bombElegant("At least one of include_x, include_y, or include_s must be non-zero", NULL);
  if (segment_length<0)
    bombElegant("segment_length must be non-negative", NULL);
  if (start_pass<0)
    bombElegant("start_pass must be non-negative", NULL);
  if (start_pass>control->n_passes)
    bombElegantVA("start_pass must be less than &run_control n_passes (%ld)", control->n_passes);

  ptunes->filename = compose_filename(filename, run->rootname);
  ptunes->startPID = start_pid;
  ptunes->endPID = end_pid;
  ptunes->PIDInterval = pid_interval;
  ptunes->include[0] = include_x;
  ptunes->include[1] = include_y;
  ptunes->include[2] = include_s;
  ptunes->include[3] = (spinCoordOffset?include_spin:0);
  ptunes->startPass = start_pass;
  ptunes->segmentLength = segment_length;

#if USE_MPI
  if (myid==0) {
#endif
  if (!SDDS_InitializeOutputElegant(&(ptunes->SDDSout), SDDS_BINARY, 1, NULL, NULL, ptunes->filename)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (!SDDS_DefineSimpleColumn(&(ptunes->SDDSout), (char*)"particleID", NULL, SDDS_ULONG64) ||
      (include_x &&
       (SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"Jx", (char*)"J$bx$n", (char*)"m", NULL, NULL, SDDS_DOUBLE, 0)==-1 ||
        SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"nux", (char*)"$gn$r$bx$n", NULL, NULL, NULL, SDDS_DOUBLE, 0)==-1)) ||
      (include_y &&
       (SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"Jy", (char*)"J$by$n", (char*)"m", NULL, NULL, SDDS_DOUBLE, 0)==-1 ||
        SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"nuy", (char*)"$gn$r$by$n", NULL, NULL, NULL, SDDS_DOUBLE, 0)==-1)) ||
       (include_s &&
       (SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"Js", (char*)"J$bs$n", (char*)"m", NULL, NULL, SDDS_DOUBLE, 0)==-1 ||
        SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"nus", (char*)"$gn$r$bs$n", NULL, NULL, NULL, SDDS_DOUBLE, 0)==-1)) ||
      (include_spin && spinCoordOffset &&
       SDDS_DefineColumn(&(ptunes->SDDSout), (char*)"nusp", (char*)"$gn$r$bsp$n", NULL, NULL, NULL, SDDS_DOUBLE, 0)==-1) ||
      !SDDS_DefineSimpleParameter(&(ptunes->SDDSout), "StartPass", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleParameter(&(ptunes->SDDSout), "EndPass", NULL, SDDS_LONG) ||
      !SDDS_WriteLayout(&(ptunes->SDDSout))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
#if USE_MPI
  }
#endif

  ptunes->initialized = 1;
  ptunes->dataPending = 0;
  ptunes->pass0 = -1;
  if (ptunes->segmentLength<=0)
    ptunes->maxBufferSize = control->n_passes;
  else    
    ptunes->maxBufferSize = ptunes->segmentLength;
  ptunes->indexHash = NULL;
  ptunes->data = (double***)calloc(8, sizeof(double**));
  ptunes->np = 0;
  
  return 1;
}

void accumulateParticleTuneData
(
 double **coord,
 int64_t np,
 long pass,
 PARTICLE_TUNES *ptunes
 )
{
  char pidText[32];
  long turnIndex, ic;
  int64_t ip;
  
  if (pass<ptunes->startPass)
    return;

  if (ptunes->pass0==-1)
    ptunes->pass0 = pass;

  turnIndex = pass - ptunes->pass0;

  if (ptunes->np==0) {
    long jp;
    int64_t pid;
    char *useParticle = NULL;
    if (ptunes->startPID>0 || ptunes->endPID>0 || ptunes->PIDInterval>1) {
      ptunes->np = 0;
      useParticle = calloc(np, sizeof(*useParticle));
      for (ip=0; ip<np; ip++) {
        pid = (int64_t)coord[ip][particleIDIndex];
        if ((ptunes->startPID<=0 || pid>=ptunes->startPID) &&
            (ptunes->endPID<=0 || pid<=ptunes->endPID) &&
            (ptunes->PIDInterval<=1 || (pid%ptunes->PIDInterval==0))) {
          ptunes->np += 1;
          useParticle[ip] = 1;
        }
      }
    } else
      ptunes->np = np;
    for (ic=0; ic<4; ic++) {
      if (ptunes->include[ic] &&
          (!(ptunes->data[ic*2+0] = (double**)czarray_2d(sizeof(double), ptunes->np, ptunes->maxBufferSize)) ||
           !(ptunes->data[ic*2+1] = (double**)czarray_2d(sizeof(double), ptunes->np, ptunes->maxBufferSize)))) {
        bombElegantVA("accumulateParticleTuneData: memory allocation failure for ic=%ld, np=%ld, length=%ld",
                      ic, np, ptunes->maxBufferSize);
      }
    }
    if (!(ptunes->particleID = calloc(ptunes->np, sizeof(long))))
      bombElegant("accumulateParticleTuneData: memory allocation failure for particleID.", NULL);
    if (!(ptunes->particleIndex = calloc(ptunes->np, sizeof(long))))
      bombElegant("accumulateParticleTuneData: memory allocation failure for particleIndex.", NULL);
    if (!(ptunes->turnIndexLimit = calloc(ptunes->np, sizeof(long))))
      bombElegant("accumulateParticleTuneData: memory allocation failure for turnIndexLimit.", NULL);
    ptunes->indexHash = hcreate(16);
    for (ip=jp=0; ip<np; ip++) {
      if (!useParticle || useParticle[ip]) {
        snprintf(pidText, 32, "%ld", (long)coord[ip][particleIDIndex]);
        ptunes->particleIndex[jp] = jp;
        if (!hadd(ptunes->indexHash, (unsigned char*)pidText, strlen(pidText), &(ptunes->particleIndex[jp])))
          bombElegantVA("Problem creating PID hash table for particle tunes: duplicate PID %ld\n", (long)coord[ip][particleIDIndex]);
        jp++;
      }
    }
    if (useParticle)
      free(useParticle);
    useParticle = NULL;
  }
  
  for (ip=0; ip<np; ip++) {
    long *pIndexPtr, pIndex;
    int64_t pid;
    pid = (int64_t)coord[ip][particleIDIndex];
    if ((ptunes->startPID<=0 || pid>=ptunes->startPID) &&
        (ptunes->endPID<=0 || pid<=ptunes->endPID) &&
        (ptunes->PIDInterval<=1 || (pid%ptunes->PIDInterval==0))) {
      snprintf(pidText, 32, "%" PRId64, pid);
      if (!hfind(ptunes->indexHash, (unsigned char *)pidText, strlen(pidText)))
        bombElegantVA("PID %ld of particle not found in hash table for particle tunes\n", pid);
      if (!(pIndexPtr = (long*)hstuff(ptunes->indexHash)))
        bombElegantVA("Problem retrieving particle index of PID %ld for particle tunes\n", pid);
      pIndex = *pIndexPtr;
      for (ic=0; ic<3; ic++) {
        if (ptunes->include[ic]) {
          ptunes->data[2*ic+0][pIndex][turnIndex] = coord[ip][2*ic+0];
          ptunes->data[2*ic+1][pIndex][turnIndex] = coord[ip][2*ic+1];
        }
      }
      if (spinCoordOffset && ptunes->include[ic]) {
	/* store x and y components */
        ptunes->data[2*ic+0][pIndex][turnIndex] = coord[ip][spinCoordOffset+0]; 
        ptunes->data[2*ic+1][pIndex][turnIndex] = coord[ip][spinCoordOffset+2];
      }
      ptunes->particleID[pIndex] = pid;
      ptunes->turnIndexLimit[pIndex] = turnIndex+1;
    }
  }

  turnIndex ++;
  ptunes->dataPending = 1;
  if (turnIndex==ptunes->segmentLength) {
    outputParticleTunes(ptunes, pass);
    ptunes->pass0 = -1;
  }
}

void outputParticleTunes(PARTICLE_TUNES *ptunes, long pass)
{
  int64_t ip;
  long it;
  int64_t pid;
  short ic, ic1;
  double frequency[2], phase[2], amplitude[2], dummy;
  double *tuneData = NULL, *J = NULL;
  double *dpass = NULL, pMean;

  if (!ptunes->initialized)
    return;
  if (!ptunes->dataPending)
    return;
  
#if USE_MPI
  long id;
  int64_t npTotal = 0, *np = NULL, iTotal, *particleID = NULL;
  short mpiAbortGlobal;
  MPI_Status mpiStatus;
  MPI_Barrier(MPI_COMM_WORLD);
  np = calloc(n_processors, sizeof(*np));
  MPI_Gather(&(ptunes->np), 1, MPI_INT64_T, np, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  for (id=0; id<n_processors; id++)
    npTotal += np[id];
  tuneData = calloc(myid==0?npTotal:ptunes->np, sizeof(double));
  J = calloc(myid==0?npTotal:ptunes->np, sizeof(double));
  if (myid==0) {
    if (!SDDS_StartPage(&(ptunes->SDDSout), npTotal) ||
	!SDDS_SetParameters(&(ptunes->SDDSout), SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
			    "StartPass", ptunes->pass0, "EndPass", pass, NULL)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
#else
  if (!SDDS_StartPage(&(ptunes->SDDSout), ptunes->np) ||
      !SDDS_SetParameters(&(ptunes->SDDSout), SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
			  "StartPass", ptunes->pass0, "EndPass", pass, NULL)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (ptunes->np) {
    tuneData = (double*)calloc(ptunes->np, sizeof(double));
    J = (double*)calloc(ptunes->np, sizeof(double));
  }
#endif

#if USE_MPI
  if (myid!=0) {
#endif
    /* need the pass in a double-precision array for fitting s(pass) to remove linear trend */
    dpass = (double*)calloc(pass-ptunes->pass0+1, sizeof(*dpass));
    for (ip=0; ip<pass-ptunes->pass0+1; ip++)
      dpass[ip] = ip;
#if USE_MPI
  }
  mpiAbort = 0;
#endif
  
  for (ic=0; ic<4; ic++) {
    if (!ptunes->include[ic])
      continue;
    for (ip=0; ip<ptunes->np; ip++) {
      
      pid = ptunes->particleID[ip];
      if ((ptunes->startPID<=0 || pid>=ptunes->startPID) &&
          (ptunes->endPID<=0 || pid<=ptunes->endPID) &&
          (ptunes->PIDInterval<=1 || (pid%ptunes->PIDInterval==0))) {
        if (ic==2) {
          /* s coordinate has linear increase that needs to be removed */
          double slope, intercept, variance;
          if (unweightedLinearFit(dpass, ptunes->data[2*ic+0][ip], ptunes->turnIndexLimit[ip],
                                   &slope, &intercept, &variance)) {
            for (it=0; it<ptunes->turnIndexLimit[ip]; it++)
              ptunes->data[2*ic+0][ip][it] -= intercept + slope*it;
          }
        }
	for (ic1=0; ic1<2; ic1++) {
	  /* Remove the average value */
	  compute_average(&pMean, ptunes->data[2*ic+ic1][ip], ptunes->turnIndexLimit[ip]);
	  for (it=0; it<ptunes->turnIndexLimit[ip]; it++)
	    ptunes->data[2*ic+ic1][ip][it] -= pMean;
	  /* Find the dominant frequency */
	  PerformNAFF(&frequency[ic1], &amplitude[ic1], &phase[ic1], &dummy, 0.0, 1, ptunes->data[2*ic+ic1][ip],
		      ptunes->turnIndexLimit[ip], NAFF_MAX_FREQUENCIES | NAFF_FREQ_CYCLE_LIMIT | NAFF_FREQ_ACCURACY_LIMIT,
		      0.0, 1, 200, 1e-12, 0, 0);
	}
        tuneData[ip] = adjustTuneHalfPlane(frequency[0], phase[0], phase[1]);
        J[ip] = computeInvariant(ptunes->data[2*ic+0][ip], ptunes->data[2*ic+1][ip], ptunes->turnIndexLimit[ip],
                                 tuneData[ip]);
      }
    }
#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    /* Collect data from each slave */
    if (myid==0) {
      long id;
      int64_t iTotal;
      MPI_Status mpiStatus;
      iTotal = 0;

      for (id=1; id<n_processors; id++) {
        if (np[id] &&
            (MPI_Recv(tuneData+iTotal, np[id], MPI_DOUBLE, id, 100, MPI_COMM_WORLD, &mpiStatus)!=MPI_SUCCESS ||
             MPI_Recv(J+iTotal, np[id], MPI_DOUBLE, id, 100, MPI_COMM_WORLD, &mpiStatus)!=MPI_SUCCESS ) ) {
          printf("Error: MPI_Recv returns error retrieving data from processor %ld\n", id);
          mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
        }
        iTotal += np[id]; 
      }
    } else {
      if (ptunes->np) {
        MPI_Send(tuneData, ptunes->np, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        MPI_Send(J, ptunes->np, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
      }
    }
    if (myid==0) {
      if (!SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, tuneData, npTotal,  tuneColumnName[ic]) ||
	  (ic<3 && 
	   !SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, J, npTotal,  JColumnName[ic])) ) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
      }
    }
    MPI_Allreduce(&mpiAbort, &mpiAbortGlobal, 1, MPI_SHORT, MPI_MAX, MPI_COMM_WORLD);
    if (mpiAbortGlobal)
      return;
#else
    if (!SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, tuneData, ptunes->np,  tuneColumnName[ic]) ||
	(ic<3 &&
	 !SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, J, ptunes->np,  JColumnName[ic])) ) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      bombElegant("IO error for particle tune output", NULL);
    }
#endif
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  /* Collect data from each slave */
  if (myid==0) {
    iTotal = 0;
    particleID = calloc(npTotal, sizeof(*particleID));
    for (id=1; id<n_processors; id++) {
      if (np[id] && MPI_Recv(particleID+iTotal, np[id], MPI_INT64_T, id, 100, MPI_COMM_WORLD, &mpiStatus)!=MPI_SUCCESS) {
        printf("Error: MPI_Recv returns error retrieving data from processor %ld\n", id);
        mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
      }
      iTotal += np[id]; 
    }
  } else {
    if (ptunes->np)
      MPI_Send(ptunes->particleID, ptunes->np, MPI_INT64_T, 0, 100, MPI_COMM_WORLD);
  }
  if (myid==0) {
    if (!SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, particleID, npTotal,  "particleID")) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
    }
  }
  free(particleID);
  MPI_Allreduce(&mpiAbort, &mpiAbortGlobal, 1, MPI_SHORT, MPI_MAX, MPI_COMM_WORLD);
  if (mpiAbortGlobal)
    return;
#else
  if (!SDDS_SetColumn(&(ptunes->SDDSout), SDDS_SET_BY_NAME, ptunes->particleID, ptunes->np, "particleID")) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    bombElegant("IO error for particle tune output", NULL);
  }
#endif
  
  if (tuneData)
    free(tuneData);
  if (dpass)
    free(dpass);
  if (J)
    free(J);
#if USE_MPI
  if (np)
    free(np);
  if (myid==0 && !SDDS_WritePage(&(ptunes->SDDSout))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
  }
#else
  if (!SDDS_WritePage(&(ptunes->SDDSout))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    bombElegant("IO error for particle tune output", NULL);
  }
#endif
  ptunes->dataPending = 0;
  ptunes->np = 0;
  if (ptunes->indexHash)
    hdestroy(ptunes->indexHash);
  ptunes->indexHash = NULL;
}

void finishParticleTunes(PARTICLE_TUNES *ptunes)
{
  int ic;
  if (!ptunes->initialized)
    return;
#if USE_MPI
  if (myid==0 && !SDDS_Terminate(&(ptunes->SDDSout))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    mpiAbort = MPI_ABORT_PARTICLE_TUNE_IO_ERROR;
  }
#else
  if (!SDDS_Terminate(&(ptunes->SDDSout))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    bombElegant("IO error for particle tune output", NULL);
  }
#endif
  ptunes->dataPending = 0;
  ptunes->np = 0;
  if (ptunes->indexHash)
    hdestroy(ptunes->indexHash);
  ptunes->indexHash = NULL;

  for (ic=0; ic<8; ic++) {
    if (ptunes->data[ic])
      free_czarray_2d((void**)ptunes->data[ic], ptunes->np, ptunes->maxBufferSize);
    ptunes->data[ic] = NULL;
  }
  if (ptunes->data)
    free(ptunes->data);
  ptunes->data = NULL;
  
  ptunes->initialized = 0;  
}
