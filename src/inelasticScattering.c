/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2017 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: inelasticScattering.c
 * purpose: do tracking to simulate inelastic scattering with recording of losses
 *
 * Michael Borland, 2017
 */

#include "mdb.h"
#include "track.h"
#if USE_MPI
#  include "inelasticScattering.h"

static SDDS_DATASET SDDSsa;
static SDDS_DATASET SDDSout;
static long fireOnPass = 1;
static FILE *fp_log = NULL;

static long nElements;
static ELEMENT_LIST **elementArray = NULL;
static long iElementName, iElementOccurence, iElementType, ideltaOrig, idkOrig, ikOrig, isOrig,
  ixLost, iyLost, ideltaLost, isLost;
static MPI_Win lastPassWin;
static long lastPassWorker = -1;

long nMomAp = 0;
double *sMomAp = NULL, *deltaNeg = NULL;

double computeScatteringDelta(long idelta, double s, double *dk) {
  long is;
  double kinv, kinv_max, kinv_min;

  if (nMomAp >= 2) {
    if (momentum_aperture_periodicity > 0)
      s = fmod(s, momentum_aperture_periodicity);
    for (is = 0; is < nMomAp; is++) {
      if (sMomAp[is] == s) {
        k_min = -deltaNeg[is];
        break;
      }
      if (sMomAp[is] > s) {
        if (is == 0)
          bombElegantVA("momentum aperture file doesn't cover the range of scattering locations, e.g., s=%le m", s);
        k_min = -(deltaNeg[is - 1] + (deltaNeg[is] - deltaNeg[is - 1]) / (sMomAp[is] - sMomAp[is - 1]) * (s - sMomAp[is - 1]));
        break;
      }
    }
    if (is == nMomAp) {
      if (momentum_aperture_periodicity) {
        k_min = -(deltaNeg[is - 1] + (deltaNeg[0] - deltaNeg[is - 1]) / (sMomAp[0] + momentum_aperture_periodicity - sMomAp[is - 1]) * (s - sMomAp[is - 1]));
      } else
        bombElegantVA("momentum aperture file doesn't cover the range of scattering locations, e.g., s=%le m", s);
    }
    k_min *= momentum_aperture_scale;
  }

  if (k_min >= 1) {
    bombElegantVA("|k_min| >= 1 for s=%le m", s);
  }

  kinv_max = 1 / k_min;
  kinv_min = 1;
  kinv = (kinv_max - kinv_min) / (n_k - 1) * idelta + kinv_min;
  if (dk) {
    double kinv1, kinv2;
    if (idelta == (n_k - 1)) {
      kinv1 = (kinv_max - kinv_min) / (n_k - 1) * (idelta - 1) + kinv_min;
      *dk = fabs((1 / kinv1 - 1 / kinv) / 2);
    } else if (idelta == 0) {
      kinv1 = (kinv_max - kinv_min) / (n_k - 1) * (idelta + 1) + kinv_min;
      *dk = fabs((1 / kinv1 - 1 / kinv) / 2);
    } else {
      kinv1 = (kinv_max - kinv_min) / (n_k - 1) * (idelta - 1) + kinv_min;
      kinv2 = (kinv_max - kinv_min) / (n_k - 1) * (idelta + 1) + kinv_min;
      *dk = (fabs(1 / kinv1 - 1 / kinv) + fabs(1 / kinv - 1 / kinv2)) / 2;
    }
  }

  return -1 / kinv;
}

void readMomentumAperture(char *momApFile) {
  SDDS_DATASET SDDSma;
  if (!fexists(momApFile))
    bombElegantVA("%s does not exist", momApFile);
  if (!SDDS_InitializeInputFromSearchPath(&SDDSma, momApFile) || !SDDS_ReadPage(&SDDSma)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSma, "s", "m", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSma, "deltaNegative", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
    SDDS_Bomb((char *)"invalid/missing columns in momentum aperture file: expect s (m) and deltaNegative");

  if ((nMomAp = SDDS_RowCount(&SDDSma)) < 2)
    bombElegantVA("Page 1 of %s has only %ld rows", momApFile, nMomAp);
  if (!(sMomAp = SDDS_GetColumnInDoubles(&SDDSma, "s")) ||
      !(deltaNeg = SDDS_GetColumnInDoubles(&SDDSma, "deltaNegative"))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }

  if (SDDS_ReadPage(&SDDSma) > 1)
    printWarning("inelastic_scattering: momentum acceptance file has more than one page.",
                 "Only the first page is used.");
}

static void deltaOffsetFunction(double **coord, long np, long pass, long i_elem, long n_elem, ELEMENT_LIST *eptr, double *pCentral) {
  long idelta, particleID, ie;
  long ip;
  long sharedData[2];
  long nKicksMade = 0;
  double ddelta;
#  if MPI_DEBUG
  static FILE *fpdeb = NULL;
  if (!fpdeb) {
    char s[1024];
    snprintf(s, 1024, "iscat-%03d.debug", myid);
    fpdeb = fopen(s, "w");
    fprintf(fpdeb, "SDDS1\n");
    fprintf(fpdeb, "&column name=ElementName type=string &end\n");
    fprintf(fpdeb, "&column name=ElementOccurence type=long &end\n");
    fprintf(fpdeb, "&column name=particleID type=long &end\n");
    fprintf(fpdeb, "&column name=idelta type=long &end\n");
    fprintf(fpdeb, "&column name=delta type=double &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#  endif
  if (i_elem == 0) {
#  if MPI_DEBUG
    printf("pass %ld, elem %ld call to deltaOffsetFunction\n", pass, i_elem);
    fflush(stdout);
#  endif
    sharedData[0] = pass;
    sharedData[1] = np;
    MPI_Put(&sharedData[0], 2, MPI_LONG, 0, 2 * myid, 2, MPI_LONG, lastPassWin);
    MPI_Win_fence(0, lastPassWin);
    lastPassWorker = pass;
  }

  if (pass == fireOnPass) {
    for (ie = 0; ie < nElements; ie++) {
      if (eptr == elementArray[ie])
        break;
    }
    if (ie == nElements) {
      return;
    }
    for (ip = nKicksMade = 0; ip < np; ip++) {
#  if MPI_DEBUG
      /*
      printf("checking particle %ld of %ld, particleID=%ld\n", ip, np, (long)coord[ip][6]);
      fflush(stdout);
      */
#  endif
      if ((particleID = coord[ip][6]) < 0) {
        /* We should never actually get here */
#  if MPI_DEBUG
        /*
        printf("buffer particle, skipping\n");
        fflush(stdout);
        */
#  endif
        continue;
      }
      if (particleID / n_k != ie) {
#  if MPI_DEBUG
        /*
        printf("not my problem, skipping\n");
        fflush(stdout);
        */
#  endif
        continue;
      }
      idelta = particleID % n_k;
      coord[ip][5] += (ddelta = computeScatteringDelta(idelta, eptr->end_pos, NULL));
#  if MPI_DEBUG
      fprintf(fpdeb, "%s %ld %ld %ld %le\n",
              eptr->name, eptr->occurence, particleID, idelta, ddelta);
#  endif
      nKicksMade++;
    }
#  if MPI_DEBUG
    printf("finished making %ld kicks\n", nKicksMade);
    fflush(stdout);
#  endif
  }
}
#endif

void setupInelasticScattering(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control) {
#if USE_MPI
  char description[200];
#endif

#if !USE_MPI
  bombElegant("inelastic_scattering command is not available in serial elegant.", NULL);
#endif

#if USE_MPI
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&inelastic_scattering, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &inelastic_scattering);
  if (momentum_aperture && strlen(momentum_aperture))
    readMomentumAperture(momentum_aperture);

  if (run->concat_order != 0)
    bombElegant("at present, elastic_scattering is incompatible with concatenation", NULL);

  /* check for data errors */
  if (!losses)
    bombElegant("no losses filename specified", NULL);
  if (!momentum_aperture) {
    if (k_min >= 1)
      bombElegant("k_min >= 1", NULL);
  } else {
    if (momentum_aperture_scale <= 0)
      bombElegant("momentum_aperture_scale<=0, which makes no sense", NULL);
  }
  if (s_start >= s_end)
    bombElegant("s_start >= s_end", NULL);
  if (include_name_pattern && has_wildcards(include_name_pattern) && strchr(include_name_pattern, '-'))
    include_name_pattern = expand_ranges(include_name_pattern);
  if (include_type_pattern && has_wildcards(include_type_pattern) && strchr(include_type_pattern, '-'))
    include_type_pattern = expand_ranges(include_type_pattern);

  nElements = 0;

  losses = compose_filename(losses, run->rootname);
  sprintf(description, "Inelastic scattering tracking");

  if (myid == 0) {
    if (!SDDS_InitializeOutputElegant(&SDDSsa, SDDS_BINARY, 1, description, "Inelastic scattering aperture", losses)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if ((iElementName = SDDS_DefineColumn(&SDDSsa, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementType = SDDS_DefineColumn(&SDDSsa, "ElementType", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementOccurence = SDDS_DefineColumn(&SDDSsa, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, 0)) < 0 ||
        (isOrig = SDDS_DefineColumn(&SDDSsa, "s", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ideltaOrig = SDDS_DefineColumn(&SDDSsa, "delta", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ikOrig = SDDS_DefineColumn(&SDDSsa, "ik", NULL, NULL, "k index (0 corresponds to k=kmin)", NULL, SDDS_LONG, 0)) < 0 ||
        (idkOrig = SDDS_DefineColumn(&SDDSsa, "dk", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ixLost = SDDS_DefineColumn(&SDDSsa, "xLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iyLost = SDDS_DefineColumn(&SDDSsa, "yLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ideltaLost = SDDS_DefineColumn(&SDDSsa, "deltaLost", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (isLost = SDDS_DefineColumn(&SDDSsa, "sLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        SDDS_DefineParameter(&SDDSsa, "Step", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSsa, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) < 0) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if (!SDDS_SaveLayout(&SDDSsa) || !SDDS_WriteLayout(&SDDSsa)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if (log_file) {
      log_file = compose_filename(log_file, run->rootname);
      fp_log = fopen(log_file, "w");
      fprintf(fp_log, "SDDS1\n&column name=Pass type=long &end\n");
      fprintf(fp_log, "&column name=Particles type=long &end\n");
      fprintf(fp_log, "&column name=MinParticles type=long &end\n");
      fprintf(fp_log, "&column name=MaxParticles type=long &end\n");
      fprintf(fp_log, "&column name=MeanParticles type=double &end\n");
      fprintf(fp_log, "&column name=ElapsedTime type=double units=s &end\n");
      fprintf(fp_log, "&column name=ElapsedCoreTime type=double units=s &end\n");
      fprintf(fp_log, "&data mode=ascii no_row_counts=1 &end\n");
    }
  }

  if (output) {
    long nsp;
    nsp = distributedBeam;
    distributedBeam = 1; /* trick SDDS_PhaseSpaceSetup() into opening file in parallel mode (slaves write) */
    output = compose_filename(output, run->rootname);
    SDDS_PhaseSpaceSetup(&SDDSout, output, SDDS_BINARY, 1, "output phase space", run->runfile, run->lattice,
                         "setupInelasticScattering");
    distributedBeam = nsp;
  }

#endif
}

void finishInelasticScattering() {
#if USE_MPI
  if (SDDS_IsActive(&SDDSsa) && !SDDS_Terminate(&SDDSsa)) {
    SDDS_SetError("Problem terminating SDDS output (finishInelasticScattering)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (SDDS_IsActive(&SDDSout) && !SDDS_Terminate(&SDDSout)) {
    SDDS_SetError("Problem terminating SDDS output (finishInelasticScattering)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#endif
}

long runInelasticScattering(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  double *startingCoord) {
#if USE_MPI
  double **coord;
  long nTotal, ie, idelta, nLost, nLeft, nElem, nEachProcessor, code, iRow;
  long ip;
  long nWorkingProcessors = n_processors - 1;
  double **lostParticles;
  double pCentral;
  ELEMENT_LIST *elem, *elem0;
  long *lastPass = NULL;
#  if MPI_DEBUG
  short mpiDebug = 1;
#  else
  short mpiDebug = 0;
#  endif
#endif

#if !USE_MPI
  bombElegant("Can't do inelastic scattering in serial elegant.", NULL);
#endif

#if USE_MPI
  if (myid == 0 || mpiDebug) {
    printf("Started multi-particle inelastic scattering algorithm\n");
    fflush(stdout);
  }

  elem = beamline->elem;

  /* determine how many elements will be tracked */
  elem0 = NULL;
  nElem = 0;
  elementArray = NULL;
  while (elem && elem->end_pos < s_end) {
    if (elem->end_pos >= s_start && elem->end_pos <= s_end &&
        (!include_name_pattern || wild_match(elem->name, include_name_pattern)) &&
        (!include_type_pattern || wild_match(entity_name[elem->type], include_type_pattern))) {
      if (!elem0)
        elem0 = elem;
      nElem++;
      elementArray = SDDS_Realloc(elementArray, sizeof(*elementArray) * (nElem + 10));
      elementArray[nElem - 1] = elem;
    }
    elem = elem->succ;
  }
  nElements = nElem;
  if (nElem == 0)
    SDDS_Bomb("no elements found for inelastic scattering computation");
  if (verbosity > 0 && (myid == 0 || mpiDebug)) {
    printf("%ld elements selected for tracking\n", nElements);
    fflush(stdout);
  }

  nElem = nElements;
  nTotal = nElem * n_k;
  if (nTotal % nWorkingProcessors != 0) {
    char warningText[1024];
    snprintf(warningText, 1024,
             "%ld working processors, %ld k values, %ld elements",
             nWorkingProcessors, n_k, nElem);
    printWarning("inelastic_scattering: The number of working processors does not evenly divide the number of particles.",
                 warningText);
    nEachProcessor = (nTotal / nWorkingProcessors) + 1;
  } else {
    nEachProcessor = nTotal / nWorkingProcessors;
  }

  if (myid == 0 || mpiDebug) {
    printf("nTotal = %ld, nWorkingProcessors = %ld, n_k = %ld, nElements = %ld, nEachProcessor = %ld\n",
           nTotal, nWorkingProcessors, n_k, nElements, nEachProcessor);
    fflush(stdout);
  }

  if (myid != 0)
    /* allocate and initialize array for tracking */
    coord = (double **)czarray_2d(sizeof(**coord), nEachProcessor, totalPropertiesPerParticle);
  else
    /* Used to track the fiducial particle */
    coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);

  setTrackingOmniWedgeFunction(NULL);
  if (startingCoord)
    memcpy(coord[0], startingCoord, sizeof(double) * 6);
  else
    memset(coord[0], 0, sizeof(**coord) * 6);
  coord[0][6] = 1;
  pCentral = run->p_central;
  if (verbosity > 1) {
    printf("Tracking fiducial particle\n");
    fflush(stdout);
  }
  delete_phase_references();
  reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
  fireOnPass = 1;
  code = do_tracking(NULL, coord, 1, NULL, beamline, &pCentral,
                     NULL, NULL, NULL, NULL, run, control->i_step,
                     FIRST_BEAM_IS_FIDUCIAL + (verbosity > 4 ? 0 : SILENT_RUNNING) + INHIBIT_FILE_OUTPUT, 1, 0, NULL, NULL, NULL, NULL, NULL);
  if (!code) {
    if (myid == 0)
      printf("Fiducial particle lost. Don't know what to do.\n");
    exitElegant(1);
  }
  if (myid == 0 || mpiDebug) {
    printf("Fiducial particle tracked.\n");
    fflush(stdout);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0 || mpiDebug) {
    printf("Fiducalization completed\n");
    fflush(stdout);
  }

  if (myid == 0) {
    lastPass = calloc(2 * n_processors, sizeof(*lastPass));
    MPI_Win_create(lastPass, 2 * n_processors * sizeof(long), sizeof(long), MPI_INFO_NULL, MPI_COMM_WORLD, &lastPassWin);
  } else
    MPI_Win_create(NULL, 0, sizeof(long), MPI_INFO_NULL, MPI_COMM_WORLD, &lastPassWin);
  MPI_Win_fence(0, lastPassWin);

  nLost = 0;
  nLeft = 0;
  if (myid != 0) {
    nLeft = nEachProcessor;
    for (ip = 0; ip < nLeft; ip++)
      coord[ip][6] = -2;
    for (ip = 0; ip < nLeft; ip++) {
      if (startingCoord)
        memcpy(coord[ip], startingCoord, sizeof(**coord) * 6);
      else
        memset(coord[ip], 0, sizeof(**coord) * 6);
      /* particle ID values for each processor are widely spaced, which leads
       * to better load balance given how the scattering sites and deltas are chosen
       */
      coord[ip][6] = myid - 1 + ip * nWorkingProcessors;
      if (coord[ip][6] >= nTotal) {
        /* Don't track more particles than needed */
        coord[ip][6] = -1;
        nLeft = ip;
      }
    }
    setTrackingOmniWedgeFunction(deltaOffsetFunction);
    if (verbosity > 1 && mpiDebug) {
      printf("Tracking\n");
      fflush(stdout);
    }
    nLost = nLeft;
    if (control->n_passes == 1)
      fireOnPass = 0;
    else
      fireOnPass = 1;
    nLeft = do_tracking(NULL, coord, nLeft, NULL, beamline, &pCentral,
                        NULL, NULL, NULL, NULL, run, control->i_step,
                        FIDUCIAL_BEAM_SEEN + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT,
                        control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    nLost -= nLeft;
    printf("Done tracking nLeft = %ld, nLost = %ld\n", nLeft, nLost);
    fflush(stdout);
    setTrackingOmniWedgeFunction(NULL);
  }

  if (myid == 0) {
    long iproc, nDone, nTotalLeft, nMinLeft, nMaxLeft, nMeanLeft, nSummed;
    nDone = 0;
    while (nDone != (n_processors - 1)) {
      MPI_Win_fence(0, lastPassWin);
      nDone = 0;
      nTotalLeft = 0;
      nMinLeft = LONG_MAX;
      nMaxLeft = -1;
      nSummed = 0;
      nMeanLeft = 0;
      for (iproc = 1; iproc < n_processors; iproc++) {
        if (lastPass[2 * iproc] == (control->n_passes - 1)) {
          nDone++;
        }
        nTotalLeft += lastPass[2 * iproc + 1];
        if (nMinLeft > lastPass[2 * iproc + 1])
          nMinLeft = lastPass[2 * iproc + 1];
        if (nMaxLeft < lastPass[2 * iproc + 1])
          nMaxLeft = lastPass[2 * iproc + 1];
        nMeanLeft += lastPass[2 * iproc + 1];
        nSummed++;
      }
      printMessageAndTime(stdout, "Pass ");
      printf(" %ld, %ld particles (min=%ld, max=%ld, ave=%g)\n", lastPass[2], nTotalLeft, nMinLeft, nMaxLeft,
             (nMeanLeft * 1.0) / nSummed);
      fflush(stdout);
      if (fp_log) {
        fprintf(fp_log, "%ld %ld %ld %ld %le %le %le\n", lastPass[2], nTotalLeft, nMinLeft, nMaxLeft,
                (nMeanLeft * 1.0) / nSummed, delapsed_time(), delapsed_time() * n_processors);
        fflush(fp_log);
      }
    }
  } else {
    long buffer[2];
    while (lastPassWorker != (control->n_passes - 1)) {
      lastPassWorker++;
      buffer[0] = lastPassWorker;
      buffer[1] = 0;
      MPI_Put(&buffer[0], 2, MPI_LONG, 0, 2 * myid, 2, MPI_LONG, lastPassWin);
      MPI_Win_fence(0, lastPassWin);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (fp_log) {
    fclose(fp_log);
    fp_log = NULL;
  }

  if (output) {
    long nsp;
    nsp = distributedBeam;
    distributedBeam = 1; /* trick dump_phase_space into working in paralle io mode */
    dump_phase_space(&SDDSout, coord, nLeft, control->i_step, run->p_central, 0.0, 0);
    distributedBeam = nsp;
  }

  gatherLostParticles(&lostParticles, &nLost, coord, nLeft, n_processors, myid);

  if (myid == 0 || mpiDebug) {
    printf("Lost-particle gather done, nLost = %ld\n", nLost);
    fflush(stdout);
  }

  if (myid == 0) {
    long badDeltaMin = 0, nDeltaMax = 0;
    double delta;
    if (!SDDS_StartPage(&SDDSsa, nLost) ||
        !SDDS_SetParameters(&SDDSsa, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Step",
                            control->i_step, NULL)) {
      SDDS_SetError("Problem writing SDDS table (doSlopeApertureSearch)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    for (ip = iRow = 0; ip < nLost; ip++) {
      double dk;
      if (verbosity > 5) {
        printf("Processing ip=%ld, particleID=%ld\n", ip, (long)lostParticles[ip][6]);
        fflush(stdout);
      }
      if (lostParticles[ip][6] < 0) {
        if (lostParticles[ip][6] == -2) {
          long j;
          printf("Problem with lost-particle accounting\n");
          for (j = 0; j < totalPropertiesPerParticle + 1; j++)
            printf("coord[%ld] = %le\n", j, lostParticles[ip][j]);
          bombElegant("problem with lost particle accounting!", NULL);
        }
        /* buffer particle, ignore */
        continue;
      }

      /* Figure out (iside, idelta, ie) */
      if ((particleID = lostParticles[ip][6]) >= 0) {
        ie = particleID / n_k;
        idelta = particleID % n_k;
        delta = computeScatteringDelta(idelta, elementArray[ie]->end_pos, &dk);
        if (idelta == (n_k - 1)) {
          printf("Problem with minimum delta limit: particle lost at this value\n");
          printf("particleID = %ld, ie = %ld, idelta = %ld, delta = %le\n",
                 particleID, ie, idelta, delta);
          badDeltaMin++;
        }
        if (idelta == 0)
          nDeltaMax++;
        if (!SDDS_SetRowValues(&SDDSsa, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iRow++,
                               iElementName, elementArray[ie]->name,
                               iElementType, entity_name[elementArray[ie]->type],
                               iElementOccurence, elementArray[ie]->occurence,
                               isOrig, elementArray[ie]->end_pos,
                               ideltaOrig, delta,
                               ikOrig, n_k - 1 - idelta,
                               idkOrig, dk,
                               ixLost, lostParticles[ip][0],
                               iyLost, lostParticles[ip][2],
                               ideltaLost, (lostParticles[ip][5] - pCentral) / pCentral,
                               isLost, lostParticles[ip][4],
                               -1)) {
          SDDS_SetError("Problem setting row values in SDDS table (doSlopeApertureSearch)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    if (verbosity > 5) {
      printf("About to write page\n");
      fflush(stdout);
    }
    if (!SDDS_WritePage(&SDDSsa)) {
      SDDS_SetError("Problem writing SDDS table (doSlopeApertureSearch)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDSsa);
    free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
    free_czarray_2d((void **)lostParticles, nLost, totalPropertiesPerParticle);
    if (badDeltaMin) {
      if (momentum_aperture)
        printWarning("inelastic_scattering: One or more particles at the minimum delta limit were lost, results suspect.",
                     "Reduce momentum_aperture_scale and re-rerun.");
      else
        printWarning("inelastic_scattering: One or more particles at the minimum delta limit were lost, results suspect.",
                     "Reduce k_min and re-rerun.");
    }
    if (nDeltaMax != nElements)
      printWarning("inelastic_scattering: One or more particles on the outer delta ring were not lost, results suspect.",
                   "Consider tracking more turns or adding physical apertures.");
  } else {
    free_czarray_2d((void **)coord, nEachProcessor, totalPropertiesPerParticle);
  }

  if (verbosity > 3 && (myid == 0 || mpiDebug)) {
    printf("Waiting on barrier after file operations\n");
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  return 1;
}
