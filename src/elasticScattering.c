/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2017 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: elasticScattering.c
 * purpose: Do tracking to find slope aperture starting from the end of each element.
 *
 * Michael Borland, 2017
 */

#include "mdb.h"
#include "track.h"
#if USE_MPI
#  include "elasticScattering.h"

static SDDS_DATASET SDDSsa;
static SDDS_DATASET SDDSout;
static long fireOnPass = 1;
static FILE *fp_log = NULL;

static double betax0, betay0;
static long nElements;
static ELEMENT_LIST **elementArray = NULL;
static long iElementName, iElementOccurence, iElementType, ixpOrig, iypOrig, isOrig,
  ixLost, iyLost, ideltaLost, isLost, ithetaOrig, iphiOrig;
static MPI_Win lastPassWin;
static long lastPassWorker = -1;

void computeScatteringAngles(long itheta, long iphi, double *xpReturn, double *ypReturn, double *thetaReturn, double *phiReturn,
                             TWISS *twiss) {
  double theta, phi, theta0;
  double xScale, yScale;

  phi = (iphi * PI) / (n_phi - 1);

  theta0 = theta_min;
  if (twiss_scaling) {
    xScale = sqrt(betax0 / twiss->betax);
    yScale = sqrt(betay0 / twiss->betay);
    theta0 = theta_min * (xScale < yScale ? xScale : yScale);
  }
  if (!quadratic_theta_spacing)
    theta = (theta_max - theta0) / (n_theta - 1.0) * itheta + theta0;
  else
    theta = (theta_max - theta0) * sqr(itheta/(n_theta-1.0)) + theta0;
  *xpReturn = theta * cos(phi);
  *ypReturn = theta * sin(phi);
  if (thetaReturn)
    *thetaReturn = theta;
  if (phiReturn)
    *phiReturn = phi;
}

static void slopeOffsetFunction(double **coord, int64_t np, long pass, long i_elem, long n_elem, ELEMENT_LIST *eptr, double *pCentral) {
  long itheta, iphi, id, ie, particleID;
  int64_t ip;
  long sharedData[2];
  MALIGN mal;
  long nKicksMade = 0;
  if (i_elem == 0) {
#  if MPI_DEBUG
    printf("pass %ld, elem %ld call to slopeOffsetFunction\n", pass, i_elem);
    fflush(stdout);
#  endif
    sharedData[0] = pass;
    sharedData[1] = np;
    MPI_Put(&sharedData[0], 2, MPI_LONG, 0, 2 * myid, 2, MPI_LONG, lastPassWin);
    MPI_Win_fence(0, lastPassWin);
    lastPassWorker = pass;
  }

  if (pass == fireOnPass) {
#  if MPI_DEBUG
    printf("firing on pass %ld, elem %ld\n", pass, i_elem);
    fflush(stdout);
#  endif
    for (ie = 0; ie < nElements; ie++) {
      if (eptr == elementArray[ie])
        break;
    }
    if (ie == nElements) {
#  if MPI_DEBUG
      printf("element not in array, returning\n");
      fflush(stdout);
#  endif
      return;
    }
#  if MPI_DEBUG
    printf("identified element %s as %ld in array (out of %ld)\n", eptr->name, ie, nElements);
    fflush(stdout);
    printf("elementArray[%ld] : name=%s, type=%s\n",
           ie, elementArray[ie]->name,
           entity_name[elementArray[ie]->type]);
    fflush(stdout);
#  endif
    memset(&mal, 0, sizeof(mal));
    mal.startPID = mal.endPID = -1;
    for (ip = nKicksMade = 0; ip < np; ip++) {
#  if MPI_DEBUG
      printf("checking particle %" PRId64 " of %" PRId64 ", particleID=%" PRId64 "\n",
             ip, np, (int64_t)coord[ip][6]);
      fflush(stdout);
#  endif
      if ((particleID = coord[ip][6]) < 0) {
        /* We should never actually get here */
#  if MPI_DEBUG
        printf("buffer particle, skipping\n");
        fflush(stdout);
#  endif
        continue;
      }
      id = particleID % (n_theta * n_phi);
      if ((particleID - id) / (n_theta * n_phi) != ie) {
#  if MPI_DEBUG
        printf("not my problem, skipping\n");
        fflush(stdout);
#  endif
        continue;
      }
      if (id > n_theta * n_phi)
        bombElegant("invalid id value (>n_theta*n_phi)", NULL);
      itheta = id % n_theta;
      iphi = id / n_theta;
      computeScatteringAngles(itheta, iphi, &mal.dxp, &mal.dyp, NULL, NULL, eptr->twiss);
#  if MPI_DEBUG
      printf("applying kicks\n");
      fflush(stdout);
#  endif
      offset_beam(coord + ip, 1, &mal, *pCentral);
      nKicksMade++;
    }
#  if MPI_DEBUG
    printf("finished making %ld kicks\n", nKicksMade);
    fflush(stdout);
#  endif
  }
}
#endif

void setupElasticScattering(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  long twissFlag) {
#if USE_MPI
  char description[200];
#endif

#if !USE_MPI
  bombElegant("elastic_scattering command is not available in serial elegant.", NULL);
#endif

#if USE_MPI
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&elastic_scattering, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &elastic_scattering);

  if (run->concat_order != 0)
    bombElegant("at present, elastic_scattering is incompatible with concatenation", NULL);

  /* check for data errors */
  if (!losses)
    bombElegant("no losses filename specified", NULL);
  if (theta_min >= theta_max)
    bombElegant("theta_min >= theta_max", NULL);
  if (s_start >= s_end)
    bombElegant("s_start >= s_end", NULL);
  if (include_name_pattern && has_wildcards(include_name_pattern) && strchr(include_name_pattern, '-'))
    include_name_pattern = expand_ranges(include_name_pattern);
  if (include_type_pattern && has_wildcards(include_type_pattern) && strchr(include_type_pattern, '-'))
    include_type_pattern = expand_ranges(include_type_pattern);

  nElements = 0;

  losses = compose_filename(losses, run->rootname);
  sprintf(description, "Elastic scattering tracking");

  if (myid == 0) {
    if (!SDDS_InitializeOutputElegant(&SDDSsa, SDDS_BINARY, 1, description, "Elastic scattering aperture", losses)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if ((iElementName = SDDS_DefineColumn(&SDDSsa, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementType = SDDS_DefineColumn(&SDDSsa, "ElementType", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementOccurence = SDDS_DefineColumn(&SDDSsa, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, 0)) < 0 ||
        (isOrig = SDDS_DefineColumn(&SDDSsa, "s", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ixpOrig = SDDS_DefineColumn(&SDDSsa, "xp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iypOrig = SDDS_DefineColumn(&SDDSsa, "yp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ithetaOrig = SDDS_DefineColumn(&SDDSsa, "theta", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iphiOrig = SDDS_DefineColumn(&SDDSsa, "phi", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ixLost = SDDS_DefineColumn(&SDDSsa, "xLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iyLost = SDDS_DefineColumn(&SDDSsa, "yLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (ideltaLost = SDDS_DefineColumn(&SDDSsa, "deltaLost", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (isLost = SDDS_DefineColumn(&SDDSsa, "sLost", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        SDDS_DefineParameter(&SDDSsa, "Step", NULL, NULL, NULL, NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSsa, "badThetaMin", NULL, NULL, "Number of particles on innermost theta ring that were lost. Should be zero for valid result.", NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDSsa, "badThetaMax", NULL, NULL, "Number of particles on outermost theta ring that were not lost. Should be zero for valid result.", NULL, SDDS_LONG, NULL) < 0 ||
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
                         "setupElasticScattering");
    distributedBeam = nsp;
  }

#endif
}

void finishElasticScattering() {
#if USE_MPI
  if (SDDS_IsActive(&SDDSsa) && !SDDS_Terminate(&SDDSsa)) {
    SDDS_SetError("Problem terminating SDDS output (finishElasticScattering)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (SDDS_IsActive(&SDDSout) && !SDDS_Terminate(&SDDSout)) {
    SDDS_SetError("Problem terminating SDDS output (finishElasticScattering)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#endif
}

long runElasticScattering(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  double *startingCoord) {
#if USE_MPI
  double **coord;
  long id, ie, itheta, iphi, nElem, code, iRow;
  int64_t nTotal, nLost, nLeft, nEachProcessor;
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
  bombElegant("Can't do elastic scattering in serial elegant.", NULL);
#endif

#if USE_MPI
  if (myid == 0 || mpiDebug) {
    printf("Started multi-particle elastic scattering algorithm\n");
    fflush(stdout);
  }

  elem = beamline->elem;
  if (twiss_scaling && !elem->twiss)
    bombElegant("twiss_scaling was invoked but twiss parameters were not computed", NULL);
  betax0 = elem->twiss->betax;
  betay0 = elem->twiss->betay;

  /* determine how man_phi elements will be tracked */
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
    SDDS_Bomb("no elements found for elastic scattering computation");
  if (verbosity > 0 && (myid == 0 || mpiDebug)) {
    printf("%ld elements selected for tracking\n", nElements);
    fflush(stdout);
  }

  nElem = nElements;
  nTotal = nElem * n_theta * n_phi;
  if (nTotal % nWorkingProcessors != 0) {
    char warningText[1024];
    snprintf(warningText, 1024,
             "%ld working processors with n_theta=%ld, n_phi=%ld, nElem=%ld\n",
             nWorkingProcessors, n_theta, n_phi, nElem);
    printWarning("elastic_scattering: The number of working processors does not evenly divide into the number of particles.",
                 warningText);
    nEachProcessor = (nTotal / nWorkingProcessors) + 1;
  } else {
    nEachProcessor = nTotal / nWorkingProcessors;
  }
  if (nWorkingProcessors % n_theta == 0) {
    char warningText[1024];
    snprintf(warningText, 1024,
             "%ld working processors with n_theta=%ld, which may give very poor load balancing and poor parallel performance.",
             nWorkingProcessors, n_theta);
    printWarning("elastic_scattering: The number of working processors is a multiple of n_theta.", warningText);
  }
  if (nWorkingProcessors % n_phi == 0) {
    char warningText[1024];
    snprintf(warningText, 1024,
             "%ld working processors with n_phi=%ld, which may give very poor load balancing and poor parallel performance",
             nWorkingProcessors, n_phi);
    printWarning("elastic_scattering: The number of working processors is a multiple of n_phi for elastic scattering.", warningText);
  }

  if (myid == 0 || mpiDebug) {
    printf("nTotal = %" PRId64 ", nWorkingProcessors = %ld, n_theta = %ld, n_phi = %ld, nElements = %ld, nEachProcessor = %" PRId64 "\n",
           nTotal, nWorkingProcessors, n_theta, n_phi, nElements, nEachProcessor);
    fflush(stdout);
  }

  if (myid != 0)
    /* allocate and initialize array for tracking */
    coord = (double **)czarray_2d(sizeof(**coord), nEachProcessor, totalPropertiesPerParticle);
  else
    /* Used to track the fiducial particle */
    coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);

  if (control->n_passes == 1)
    fireOnPass = 0;
  else
    fireOnPass = 1;

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
       * to better load balance given how the scattering sites and angles are chosen
       */
      coord[ip][6] = myid - 1 + ip * nWorkingProcessors;
      if (coord[ip][6] >= nTotal) {
        /* Don't track more particles than needed */
        coord[ip][6] = -1;
        nLeft = ip;
      }
    }
    setTrackingOmniWedgeFunction(slopeOffsetFunction);
    if (verbosity > 1 && mpiDebug) {
      printf("Tracking\n");
      fflush(stdout);
    }
    nLost = nLeft;
    fireOnPass = 0;
    nLeft = do_tracking(NULL, coord, nLeft, NULL, beamline, &pCentral,
                        NULL, NULL, NULL, NULL, run, control->i_step,
                        FIDUCIAL_BEAM_SEEN + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT,
                        control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    nLost -= nLeft;
    printf("Done tracking nLeft = %" PRId64 ", nLost = %" PRId64 "\n", nLeft, nLost);
    fflush(stdout);
    if (verbosity > 9) {
      /*
      for (ip = 0; ip < nLost; ip++)
        printf("ip = %ld, particleID = %ld\n", ip, (long)lostParticles[ip][6]);
      fflush(stdout);
      */
    }
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
    printf("Lost-particle gather done, nLost = %" PRId64 "\n", nLost);
    fflush(stdout);
  }

  if (myid == 0) {
    long badThetaMin = 0, nThetaMax = 0;
    if (!SDDS_StartPage(&SDDSsa, nLost) ||
        !SDDS_SetParameters(&SDDSsa, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Step",
                            control->i_step, NULL)) {
      SDDS_SetError("Problem writing SDDS table (doSlopeApertureSearch)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    for (ip = iRow = 0; ip < nLost; ip++) {
      double xpOrig, ypOrig, thetaOrig, phiOrig;
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

      /* Figure out (itheta, iphi, ie) */
      particleID = lostParticles[ip][6];
      id = particleID % (n_theta * n_phi);
      ie = (particleID - id) / (n_theta * n_phi);
      itheta = id % n_theta;
      iphi = id / n_theta;
      computeScatteringAngles(itheta, iphi, &xpOrig, &ypOrig, &thetaOrig, &phiOrig, elementArray[ie]->twiss);
      if (itheta == 0)
        badThetaMin++;
      if (itheta == (n_theta - 1))
        nThetaMax++;
      if (!SDDS_SetRowValues(&SDDSsa, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iRow++,
                             iElementName, elementArray[ie]->name,
                             iElementType, entity_name[elementArray[ie]->type],
                             iElementOccurence, elementArray[ie]->occurence,
                             isOrig, elementArray[ie]->end_pos,
                             ixpOrig, xpOrig,
                             iypOrig, ypOrig,
                             ithetaOrig, thetaOrig,
                             iphiOrig, phiOrig,
                             ixLost, lostParticles[ip][0],
                             iyLost, lostParticles[ip][2],
                             ideltaLost, (lostParticles[ip][5] - pCentral) / pCentral,
                             isLost, lostParticles[ip][4],
                             -1)) {
        SDDS_SetError("Problem setting row values in SDDS table (doSlopeApertureSearch)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (verbosity > 5) {
        printf("Set data for row %ld: %s, %s, %ld, %le, %le, %le, %le %le, %le, %le\n",
               iRow - 1,
               elementArray[ie]->name, entity_name[elementArray[ie]->type], elementArray[ie]->occurence,
               elementArray[ie]->end_pos, xpOrig, ypOrig,
               lostParticles[ip][0], lostParticles[ip][2], lostParticles[ip][5], lostParticles[ip][4]);
        fflush(stdout);
      }
    }
    if (!SDDS_SetParameters(&SDDSsa, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "badThetaMin",
                            badThetaMin, "badThetaMax", nElements * n_phi - nThetaMax, NULL)) {
      SDDS_SetError("Problem writing SDDS table (doSlopeApertureSearch)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
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
    if (badThetaMin) {
      char warningText[1024];
      snprintf(warningText, 1024,
               "Occurred in %ld cases. You should reduce theta_min and/or use twiss_scaling, and re-rerun.",
               badThetaMin);
      printWarning("elastic_scattering: Some inner-ring particles lost, results suspect.", warningText);
    }
    if (nThetaMax != nElements * n_phi) {
      char warningText[1024];
      snprintf(warningText, 1024,
               "Occurred in %ld cases. You should increase theta_max and re-rerun.",
               nElements * n_phi - nThetaMax);
      printWarning("elastic_scattering: Some outer-ring particles *not* lost, results suspect.", warningText);
    }
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
