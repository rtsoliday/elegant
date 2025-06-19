/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* routine: do_tracking()
 * purpose: track a collection of particles through a beamline
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"
#ifdef USE_GSL
#  include "gsl/gsl_poly.h"
#endif
#ifdef HAVE_GPU
#  include <gpu_base.h>
#  include <gpu_funcs.h>
#  include <gpu_limit_amplitudes.h>
#endif /* HAVE_GPU */

void flushTransverseFeedbackDriverFiles(TFBDRIVER *tfbd);
void set_up_frfmode(FRFMODE *rfmode, char *element_name, double element_z, long n_passes, RUN *run, long n_particles, double Po, double total_length);
void track_through_frfmode(double **part, long np, FRFMODE *rfmode, double Po, char *element_name, double element_z, long pass, long n_passes, CHARGE *charge);
void set_up_ftrfmode(FTRFMODE *rfmode, char *element_name, double element_z, long n_passes, RUN *run, long n_particles, double Po, double total_length);
void track_through_ftrfmode(double **part, long np, FTRFMODE *trfmode, double Po, char *element_name, double element_z, long pass, long n_passes, CHARGE *charge);
void transformEmittances(double **coord, long np, double pCentral, EMITTANCEELEMENT *ee);
void checkRFCAChangeTConflicts(LINE_LIST *beamline);

ELEMENT_LIST *findBeamlineMatrixElement(ELEMENT_LIST *eptr);
void trackLongitudinalOnlyRing(double **part, long np, VMATRIX *M, double *alpha);
void store_fitpoint_matrix_values(MARK *fpt, char *name, long occurence, VMATRIX *M);
void storeBPMReading(ELEMENT_LIST *eptr, double **coord, long np, double Po);
long trackWithIndividualizedLinearMatrix(double **particle, long particles,
                                         double **accepted, double Po, double z,
                                         ELEMENT_LIST *eptr,
                                         TWISS *twiss0, double *tune0,
                                         double *chrom, double *chrom2, double *chrom3,
                                         double *dbeta_dPoP, double *dalpha_dPoP,
                                         double *alphac, double *eta2,
                                         ILMATRIX *ilmat);
void matr_element_tracking(double **coord, VMATRIX *M, MATR *matr,
                           long np, double z);
void ematrix_element_tracking(double **coord, VMATRIX *M, EMATRIX *matr,
                              long np, double z, double *Pcentral);
void distributionScatter(double **part, long np, double Po, DSCATTER *scat, long iPass);
void storeMonitorOrbitValues(ELEMENT_LIST *eptr, double **part, long np);
void mhist_table(ELEMENT_LIST *eptr0, ELEMENT_LIST *eptr, long step, long pass, double **coord, long np,
                 double Po, double length, double charge, double z);
void set_up_mhist(MHISTOGRAM *mhist, RUN *run, long occurence);
void findMinMax(double **coord, long np, double *min, double *max, double *c0, double Po);

void interpolateFTable(double *B, double *xyz, FTABLE *ftable);
void ftable_frame_converter(double **coord, long np, FTABLE *ftable, long entrance_exit);
double choose_theta(double rho, double x0, double x1, double x2);
void track_through_space_harmonic_deflector(
                                            double **final,
                                            SHRFDF *rf_param,
                                            double **initial,
                                            long n_particles,
                                            double pc_central);

short determineP0ChangeBlocking(ELEMENT_LIST *eptr);

#if USE_MPI
static short mpiAbortGlobal;
typedef enum balanceMode { badBalance,
  startMode,
  goodBalance } balance;
void scatterParticles(double **coord, long *nToTrack, double **accepted,
                      long n_processors, int myid, balance balanceStatus,
                      double my_rate, double nParPerElements, double round,
                      int lostSinceSeqMod, int *distributed,
                      long *reAllocate, double *P_central);
void gatherParticles(double ***coord, long *nToTrack, long *nLost, double ***accepted, long n_processors,
                     int myid, double *round);
/* Avoid unnecessary communications by checking if an operation will be executed in advance*/
int usefulOperation(ELEMENT_LIST *eptr, unsigned long flags, long i_pass);
balance checkBalance(double my_wtime, int myid, long n_processors, int verbose);
#endif
static void checkBeamStructure(BEAM *beam);

#ifdef SORT
int comp_IDs(const void *coord1, const void *coord2);
#endif

void applyBeamBeamKicks(double **part, long np, BEAMBEAM *bb, double P0);

double beta_from_delta(double p, double delta) {
  p *= 1 + delta;
  return (p / sqrt(p * p + 1));
}

/* This is used if one needs to wedge a function into the lattice at a specific
 * location
 */

static void (*trackingWedgeFunction)(double **part, long np, long pass, double *pCentral) = NULL;
static ELEMENT_LIST *trackingWedgeElement = NULL;
void setTrackingWedgeFunction(void (*wedgeFunc)(double **part, long np, long pass, double *pCentral),
                              ELEMENT_LIST *eptr) {
  trackingWedgeFunction = wedgeFunc;
  trackingWedgeElement = eptr;
}

/* This is used if one needs to wedge a function after each element location
 */

static void (*trackingOmniWedgeFunction)(double **part, long np, long pass, long i_elem, long n_elem, ELEMENT_LIST *eptr, double *pCentral);
void setTrackingOmniWedgeFunction(void (*wedgeFunc)(double **part, long np, long pass, long i_elem, long n_elem, ELEMENT_LIST *eptr, double *pCentral)) {
  trackingOmniWedgeFunction = wedgeFunc;
}

static double timeCounter[N_TYPES], tStart;
static long runCounter[N_TYPES];
static long elementTimingActive = 0;

void resetElementTiming() {
  long i;
  printf("Resetting element timing data.\n");
  fflush(stdout);
  for (i = 0; i < N_TYPES; i++)
    timeCounter[i] = runCounter[i] = 0;
  elementTimingActive = 1;
}

void reportElementTiming() {
  if (elementTimingActive) {
    long i;
    printf("Time spent in different elements:\n");
    for (i = 0; i < N_TYPES; i++) {
      if (runCounter[i] != 0)
        printf("%16s: %10ld times, %10.3e s, %10.3e s/element\n", entity_name[i], runCounter[i], timeCounter[i], timeCounter[i] / runCounter[i]);
    }
    fflush(stdout);
  }
}

long do_tracking(
                 /* Either the beam pointer or the coord pointer must be supplied, but not both */
                 BEAM *beam,
                 double **coord,
                 long nOriginal, /* Used only if coord is supplied */
                 long *effort,
                 LINE_LIST *beamline,
                 double *P_central, /* beta*gamma for central particle */
                 double **accepted,
                 BEAM_SUMS **sums_vs_z,
                 long *n_z_points,
                 TRAJECTORY *traj_vs_z,
                 RUN *run,
                 long step,
                 unsigned long flags,
                 long n_passes,
                 long passOffset,
                 SASEFEL_OUTPUT *sasefel,
                 SLICE_OUTPUT *sliceAnalysis,
                 double *finalCharge,
                 double **lostParticles, /* Ignored */
                 ELEMENT_LIST *startElem) {
  OUTPUT_FILES *outputFiles;
  RFMODE *rfmode;
  TRFMODE *trfmode;
  FRFMODE *frfmode;
  FTRFMODE *ftrfmode;
  WATCH *watch = NULL;
  SLICE_POINT *slicePoint;
  STRAY *stray;
  HISTOGRAM *histogram;
  MHISTOGRAM *mhist;
  FTABLE *ftable;
  ENERGY *energy;
  MAXAMP *maxamp = NULL, maxampBuf;
  MALIGN *malign;
  TAPERAPC *taperapc = NULL;
  TAPERAPE *taperape = NULL;
  TAPERAPR *taperapr = NULL;
  APCONTOUR *apcontour = NULL;
  ELEMENT_LIST *eptr, *eptrPred, *eptrCLMatrix = NULL;
  double sMaxTransmittedMonitor; /* maximum s coordinate of transmitted particles at MONI, HMON, or VMON */
  static int sMaxTransmittedMonitorMemory = -1;
  long nToTrack;     /* number of particles being tracked */
  long nLeft;        /* number of those that are left after a tracking routine returns */
  long nLost = 0;    /* accumulated number lost */
  long nMaximum = 0; /* maximum number of particles seen */
  /* long show_dE; */
  long maxampOpenCode = 0, maxampExponent = 0, maxampYExponent = 0;
  double dgamma, dP[3], z, z_recirc, last_z, z_travel;
  long i, j, i_traj = 0, i_sums, i_pass, isConcat, i_elem;
  long i_sums_recirc, saveISR = 0;
  long watch_pt_seen, feedbackDriverSeen;
  double sum, x_max, y_max;
  long elliptical;
#if !SDDS_MPI_IO
  double et1, et2 = 0;
#endif
  long is_batch = 0, last_type;
  static long is_ansi_term = -1;
  char s[100];
#if defined(BEAM_SUMS_DEBUG)
  char *name;
#endif
  long check_nan, sums_allocated = 0;
  long elementsTracked, sliceAnDone = 0;
  CHARGE *charge;
  static long warnedAboutChargePosition = 0;
  unsigned long classFlags = 0;
  long nParticlesStartPass = 0;
  int myid = 0, active = 1;
  long memoryBefore = 0, memoryAfter = 0;
#if USE_MPI
#  ifdef SORT
  int nToTrackAtLastSort;
#  endif
  int needSort = 0;
  int lostSinceSeqMode = 0;
#  ifdef CHECKFLAGS
  long old_nToTrack = 0;
#  endif
  long nParElements = 0, nElements = 0;
  int checkFlags;
  double my_wtime = 0, start_wtime = 0, end_wtime, nParPerElements = 0, my_rate;
  double round = 0.5;
  balance balanceStatus;
#  if SDDS_MPI_IO
  int distributed = 1;
  long total_nOriginal;
  long total_nToTrack;
#  else
  int distributed = 0; /* indicate if the particles have been scattered */
#  endif
  long reAllocate = 0; /* indicate if new memory needs to be allocated */
#  ifdef USE_MPE       /* use the MPE library */
  int event1a, event1b, event2a, event2b;
  MPE_LOG_BYTES bytebuf;
  int bytebuf_pos = 0;
  event1a = MPE_Log_get_event_number();
  event1b = MPE_Log_get_event_number();
  event2a = MPE_Log_get_event_number();
  event2b = MPE_Log_get_event_number();
  if (isMaster) {
    MPE_Describe_state(event1a, event1b, "Watch", "red");
    MPE_Describe_info_state(event2a, event2b, "Tracking_element", "orange",
                            "Element: %s");
  }
#  endif
  balanceStatus = startMode;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid); /* get ID number for each processor */
  trackingContext.myid = myid;
  if (myid == 0)
    my_rate = 0.0;
  else
    my_rate = 1.0;
  if (notSinglePart && partOnMaster) /* This is a special case when the first beam is fiducial. We need scatter the beam in the second step. */
    distributed = 0;
#endif

  trajectoryTracking = (flags & TEST_PARTICLES);
  if (flags & TEST_PARTICLES)
    setObstructionsMode(0);
  else
    setObstructionsMode(1);

  outputFiles = &(run->outputFiles);

#if MPI_DEBUG && USE_MPI
  printf("do_tracking called with nOriginal=%ld, beam=%x\n", nOriginal, beam);
  fflush(stdout);
#endif
#ifdef DEBUG
  if (flags & FINAL_SUMS_ONLY)
    printf("FINAL_SUMS_ONLY set\n");
  if (flags & TEST_PARTICLES)
    printf("TEST_PARTICLES set\n");
  if (flags & BEGIN_AT_RECIRC)
    printf("BEGIN_AT_RECIRC set\n");
  if (flags & TEST_PARTICLE_LOSSES)
    printf("TEST_PARTICLE_LOSSES set\n");
  if (flags & SILENT_RUNNING)
    printf("SILENT_RUNNING set\n");
  if (flags & TIME_DEPENDENCE_OFF)
    printf("TIME_DEPENDENCE_OFF set\n");
  if (flags & INHIBIT_FILE_OUTPUT)
    printf("INHIBIT_FILE_OUTPUT set\n");
  if (flags & LINEAR_CHROMATIC_MATRIX)
    printf("LINEAR_CHROMATIC_MATRIX set\n");
  if (flags & LONGITUDINAL_RING_ONLY)
    printf("LONGITUDINAL_RING_ONLY set\n");
  if (flags & FIRST_BEAM_IS_FIDUCIAL)
    printf("FIRST_BEAM_IS_FIDUCIAL set\n");
  if (flags & FIDUCIAL_BEAM_SEEN)
    printf("FIDUCIAL_BEAM_SEEN set\n");
  if (flags & PRECORRECTION_BEAM)
    printf("PRECORRECTION_BEAM set\n");
  if (flags & RESTRICT_FIDUCIALIZATION)
    printf("RESTRICT_FIDUCIALIZATION set\n");
  if (flags & IBS_ONLY_TRACKING)
    printf("IBS_ONLY_TRACKING set\n");
  if (flags & CLOSED_ORBIT_TRACKING)
    printf("CLOSED_ORBIT_TRACKING set\n");
  if (flags & ALLOW_MPI_ABORT_TRACKING)
    printf("ALLOW_MPI_ABORT_TRACKING set\n");
  if (flags & RESET_RF_FOR_EACH_STEP)
    printf("RESET_RF_FOR_EACH_STEP set\n");
#endif

#ifdef DEBUG_CRASH
  printMessageAndTime(stdout, "do_tracking checkpoint 0\n");
#endif
#if TURBO_STRINGS
  trackingContext.rootname = run->rootname;
#else
  strncpy(trackingContext.rootname, run->rootname, CONTEXT_BUFSIZE);
#endif
  if (!coord && !beam)
    bombElegant("Null particle coordinate array and null beam pointer! (do_tracking)", NULL);
  if (coord && beam)
    bombElegant("Particle coordinate array and beam pointer both supplied!  (do_tracking)", NULL);
  if (beam) {
    coord = beam->particle;
    nOriginal = beam->n_to_track; /* used only for computing macroparticle charge */
    beam->n_lost = 0;
  }

#if SDDS_MPI_IO
  if (notSinglePart && !partOnMaster) {
    if (isMaster)
      nOriginal = 0;
    MPI_Allreduce(&nOriginal, &total_nOriginal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  } else /* single partticle case where all the processors track the same particle(s), or particles are on master when the first beam is fiducial */
    total_nOriginal = nOriginal;
#  if MPI_DEBUG
  printf("nOriginal = %ld, total_nOriginal = %ld\n", nOriginal, total_nOriginal);
#  endif
#endif

#ifdef WATCH_MEMORY
  printf("start do_tracking():  CPU: %6.2lf  PF: %6ld  MEM: %6ld\n",
         cpu_time() / 100.0, page_faults(), memoryUsage());
  fflush(stdout);
#endif
#ifdef DEBUG_CRASH
  printMessageAndTime(stdout, "do_tracking checkpoint 0.1\n");
#endif

  if (is_ansi_term == -1) {
    char *ptr;
    is_ansi_term = 1;
    if (!(ptr = getenv("TERM")))
      is_ansi_term = 0;
    else if (strcmp(ptr, "emacs") == 0)
      is_ansi_term = 0;
  }

#if SDDS_MPI_IO
  if (isSlave || (!notSinglePart))
#else
    if (isMaster)
#endif
      if (accepted)
        copy_particles(accepted, coord, nOriginal);

  eptr = beamline->elem;
  z = z_travel = z_recirc = last_z = beamline->elem->beg_pos;

  i_sums = i_sums_recirc = 0;
  x_max = y_max = 0;
  nToTrack = nLeft = nMaximum = nOriginal;

#if USE_MPI
  if (!partOnMaster && notSinglePart) {
    if (isMaster)
      nToTrack = 0;
    if (beam)
      MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  } else { /* singlePart tracking or partOnMaster */
    if (beam)
      beam->n_to_track_total = nToTrack;
  }
#endif
#if !SDDS_MPI_IO
  et1 = -2.0;
#endif
  elliptical = isConcat = 0;
  watch_pt_seen = feedbackDriverSeen = 0;

#ifdef SORT
  nToTrackAtLastSort = nToTrack;
#endif
#ifdef DEBUG_CRASH
  printMessageAndTime(stdout, "do_tracking checkpoint 0.2\n");
#endif

  check_nan = 1;
  eptr = beamline->elem;

#ifdef HAVE_GPU
  // gpuBaseInit(coord, nOriginal, accepted, lostBeam->particle, isMaster);
  gpuBaseInit(coord, nOriginal, accepted, NULL, isMaster);
#endif

  flags |= beamline->fiducial_flag;
  if ((flags & FIRST_BEAM_IS_FIDUCIAL && !(flags & FIDUCIAL_BEAM_SEEN)) || !(flags & FIRST_BEAM_IS_FIDUCIAL)) {
    /* this is required just in case rf elements etc. have previously
     * been activated by computation of correction matrices or trajectory
     * correction.
     */
    if (!(flags & SILENT_RUNNING) && !(flags & TEST_PARTICLES)) {
      printMessageAndTime(stdout, "This step establishes energy profile vs s (fiducial beam).\n");
    }
    if (run->n_passes_fiducial > 0)
      n_passes = run->n_passes_fiducial;
    flags &= ~FIDUCIAL_BEAM_SEEN;
    while (eptr) {
      eptr->Pref_output_fiducial = 0;
      eptr = eptr->succ;
    }
  }
  if (flags & RESET_RF_FOR_EACH_STEP) {
    delete_phase_references();
    reset_special_elements(beamline, RESET_INCLUDE_RF);
    if (!(flags & SILENT_RUNNING) && !(flags & TEST_PARTICLES)) {
      printMessageAndTime(stdout, "Rf phases/references reset.\n");
    }
  }
  reset_driftCSR();

  checkRFCAChangeTConflicts(beamline);

#ifdef DEBUG_CRASH
  printMessageAndTime(stdout, "do_tracking checkpoint 0.3\n");
#endif

  if (!(flags & FIDUCIAL_BEAM_SEEN) && flags & PRECORRECTION_BEAM)
    flags &= ~FIRST_BEAM_IS_FIDUCIAL;

  log_exit("do_tracking.1");
  log_entry("do_tracking.2");
#if defined(BEAM_SUMS_DEBUG)
  name = "_BEG_";
#endif
  last_type = sums_allocated = 0;
  charge = NULL;
  if (finalCharge)
    *finalCharge = 0;

  i_pass = passOffset;
#if SDDS_MPI_IO
  if (isSlave || (!notSinglePart) || partOnMaster) {
#else
    if (isMaster) { /* As the particles have not been distributed, only master needs to do these computation */
#endif
      if (run->concat_order && beamline->flags & BEAMLINE_CONCAT_DONE &&
          !(flags & TEST_PARTICLES)) {
        if (beamline->ecat_recirc && (i_pass || flags & BEGIN_AT_RECIRC))
          eptr = beamline->ecat_recirc;
        else
          eptr = beamline->ecat;
        isConcat = 1;
      } else if (beamline->elem_recirc && (i_pass || flags & BEGIN_AT_RECIRC))
        eptr = beamline->elem_recirc;
      else
        eptr = beamline->elem;
      if (check_nan) {
        nLeft = limit_amplitudes(coord, DBL_MAX, DBL_MAX, nToTrack, accepted, z, *P_central, 0,
                                 0, eptr);
        if (nLeft != nToTrack)
          recordLostParticles(beam, coord, nLeft, nToTrack, 0);
        nToTrack = nLeft;
      }
      if (run->apertureData.initialized) {
        nLeft = imposeApertureData(coord, nToTrack, accepted, 0.0, *P_central, &(run->apertureData), eptr);
        if (nLeft != nToTrack)
          recordLostParticles(beam, coord, nLeft, nToTrack, 0);
        nToTrack = nLeft;
      }
    }

    sMaxTransmittedMonitor = 0;
    if (sMaxTransmittedMonitorMemory == -1)
      sMaxTransmittedMonitorMemory = rpn_create_mem("sMaxTransmittedMonitor", 0);
    rpn_store(sMaxTransmittedMonitor, NULL, sMaxTransmittedMonitorMemory);

    for (i_pass = passOffset; i_pass < n_passes + passOffset; i_pass++) {
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.35, ");
      printf("pass = %ld\n", i_pass);
      fflush(stdout);
#endif
      if (run->trackingInterruptFile && strlen(run->trackingInterruptFile) &&
          fexists(run->trackingInterruptFile) && get_mtime(run->trackingInterruptFile) > run->trackingInterruptFileMtime) {
        n_passes = i_pass - passOffset;
        break;
      }
      log_entry("do_tracking.2.1");
      if (run->stopTrackingParticleLimit > 0) {
#if !USE_MPI
        if (nToTrack < run->stopTrackingParticleLimit)
#else
#  ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 0.36\n");
#  endif
        MPI_Allreduce(&nToTrack, &total_nToTrack, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 0.37\n");
#  endif
        if (total_nToTrack < run->stopTrackingParticleLimit)
#endif
          {
#ifdef HAVE_GPU
            coord = forceParticlesToCpu("stopTrackingParticleLimit reached");
#endif
            n_passes = i_pass - passOffset;
            break;
          }
      }

#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.4\n");
#endif

      ResetNoiseGroupValues();
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.41\n");
#endif
      if (applyElementModulations(&(run->modulationData), beamline, *P_central, coord, nToTrack, run, i_pass)) {
        beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
        beamline->flags &= ~BEAMLINE_TWISS_CURRENT;
      }
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.42\n");
#endif
      if (applyElementRamps(&(run->rampData), beamline, *P_central, run, i_pass)) {
        beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
        beamline->flags &= ~BEAMLINE_TWISS_CURRENT;
      }
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.43\n");
#endif

      if (beamline->links) {
        sprintf(s, "%.15e sto p_central  %ld sto turn", *P_central, i_pass);
        rpn(s);
        rpn_clear();
        if (rpn_check_error())
          exitElegant(1);
        if (assert_element_links(beamline->links, run, beamline, TURN_BY_TURN_LINK)) {
          beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
          beamline->flags &= ~BEAMLINE_TWISS_CURRENT;
        }
      }
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.44\n");
#endif
      if (beamline->flags & BEAMLINE_TWISS_WANTED && !(beamline->flags & BEAMLINE_TWISS_CURRENT) && !(flags & TEST_PARTICLES)) {
        update_twiss_parameters(run, beamline, NULL);
      }
#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.45\n");
#endif
      if (run->concat_order && !(flags & TEST_PARTICLES) &&
          !(beamline->flags & BEAMLINE_CONCAT_CURRENT)) {
        /* form concatenated beamline for tracking */
        if (getSCMULTSpecCount())
          bombElegant("space charge calculation can not work together with matrix concatenation tracking. \n Please remove concat_order from run_setup", NULL);
        concatenate_beamline(beamline, run);
      }

#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.5\n");
#endif

      if (run->concat_order && beamline->flags & BEAMLINE_CONCAT_DONE &&
          !(flags & TEST_PARTICLES)) {
        if (beamline->ecat_recirc && (i_pass || flags & BEGIN_AT_RECIRC))
          eptr = beamline->ecat_recirc;
        else
          eptr = beamline->ecat;
        isConcat = 1;
      } else if (beamline->elem_recirc && (i_pass || flags & BEGIN_AT_RECIRC))
        eptr = beamline->elem_recirc;
      else
        eptr = beamline->elem;

      if (i_pass == passOffset) {
        if (flags & LINEAR_CHROMATIC_MATRIX) {
          if (!isConcat) {
            printf("Error: in order to use the \"linear chromatic matrix\" for\n");
            fflush(stdout);
            printf("tracking, you must ask for matrix concatenation in the run_setup.\n");
            fflush(stdout);
            exitElegant(1);
          }
          eptrCLMatrix = findBeamlineMatrixElement(eptr);
        }
        if (flags & LONGITUDINAL_RING_ONLY) {
          if (!isConcat) {
            printf("Error: in order to use the \"longitudinal ring\" mode of\n");
            fflush(stdout);
            printf("tracking, you must ask for matrix concatenation in the run_setup.\n");
            fflush(stdout);
            exitElegant(1);
          }
          eptrCLMatrix = findBeamlineMatrixElement(eptr);
        }
      }

#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.6\n");
#endif

      if (sums_vs_z && n_z_points) {
        if (!sums_allocated && !*sums_vs_z) {
          /* allocate storage for beam sums */
          if (!isConcat)
            *n_z_points = beamline->n_elems + 1 +
              (run->wrap_around ? 0 : (n_passes - 1) * (beamline->n_elems - beamline->i_recirc));
          else
            *n_z_points = beamline->ncat_elems + 1 +
              (run->wrap_around ? 0 : (n_passes - 1) * (beamline->ncat_elems - beamline->i_recirc));
          if (flags & FINAL_SUMS_ONLY)
            *n_z_points = 0;
          *sums_vs_z = allocateBeamSums(flags, *n_z_points + 1);
          zero_beam_sums(*sums_vs_z, *n_z_points + 1);
          sums_allocated = 1;
        } else if (!run->combine_bunch_statistics && i_pass == passOffset)
          zero_beam_sums(*sums_vs_z, *n_z_points + 1);
      }

      if (run->wrap_around) {
        i_sums = i_sums_recirc;  /* ==0 for i_pass==0 */
        z = z_travel = z_recirc; /* ditto, for forward-tracking at least */
        last_z = z;
      }
      if (run->final_pass && sums_vs_z && n_z_points)
        zero_beam_sums(*sums_vs_z, *n_z_points + 1);

#ifdef DEBUG_CRASH
      printMessageAndTime(stdout, "do_tracking checkpoint 0.7\n");
#endif

      log_exit("do_tracking.2.1");
      log_entry("do_tracking.2.2");
      if (!(flags & SILENT_RUNNING) && !is_batch && n_passes != 1 && !(flags & TEST_PARTICLES) && !(run->tracking_updates == 0)) {

#if !SDDS_MPI_IO
        if ((et2 = delapsed_time()) - et1 > 2.0) {
          sprintf(s, "%ld particles present after pass %ld        ",
                  nToTrack, i_pass);
#else
          if (i_pass % 20 == 0) {
            if (!partOnMaster && notSinglePart) {
              sprintf(s, "%ld particles present after pass %ld        ",
                      beam ? beam->n_to_track_total : -1, i_pass);
            } else { /* singlePart tracking or partOnMaster */
              sprintf(s, "%ld particles present after pass %ld        ",
                      nToTrack, i_pass);
            }
#endif
            fputs(s, stdout);
            if (is_ansi_term)
              backspace(strlen(s));
            else
              fputc('\n', stdout);
            fflush(stdout);
#if !SDDS_MPI_IO
            et1 = et2;
#endif
          }
        }
        elementsTracked = -1;
        eptrPred = eptr;
#if USE_MPI
        if (notSinglePart) {
          my_wtime = 0.0;
          nParElements = 0;
          nElements = 0;
        }
#endif
        nParticlesStartPass = nToTrack;

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 0.8\n");
#endif

        if (getSCMULTSpecCount()) {
          /* prepare space charge effects calculation  */
#ifdef HAVE_GPU
          coord = forceParticlesToCpu("initializeSCMULT");
#endif
          if (!(flags & TEST_PARTICLES))
            initializeSCMULT(eptr, coord, nToTrack, *P_central, i_pass, beam? beam->id_slots_per_bunch : 0);
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 0.9\n");
#endif

        i_elem = 0;
        if (i_pass == passOffset && startElem) {
          /* start tracking from an interior point in the beamline */
          while (eptr && eptr != startElem) {
            if (eptr->type == T_MAXAMP) {
              maxamp = (MAXAMP *)eptr->p_elem;
              x_max = maxamp->x_max;
              y_max = maxamp->y_max;
              elliptical = maxamp->elliptical;
              maxampOpenCode = determineOpenSideCode(maxamp->openSide);
              maxampExponent = maxamp->exponent;
              maxampYExponent = maxamp->yExponent;
            } else if (eptr->type == T_TAPERAPC) {
              taperapc = (TAPERAPC *)eptr->p_elem;
              if (taperapc->sticky) {
                maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                maxamp->x_max = maxamp->y_max = x_max = y_max = taperapc->r[taperapc->e2Index];
                maxamp->elliptical = elliptical = 1;
                maxampOpenCode = 0;
                maxamp->openSide = NULL;
                maxamp->exponent = maxamp->yExponent = maxampExponent = maxampYExponent = 2;
              }
            } else if (eptr->type == T_TAPERAPE) {
              taperape = (TAPERAPE *)eptr->p_elem;
              if (taperape->sticky) {
                maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                maxamp->x_max = x_max = taperape->a[taperape->e2Index];
                maxamp->y_max = y_max = taperape->b[taperape->e2Index];
                maxamp->elliptical = elliptical = 1;
                maxampOpenCode = 0;
                maxamp->openSide = NULL;
                maxamp->exponent = maxampExponent = taperape->xExponent;
                maxamp->yExponent = maxampYExponent = taperape->yExponent;
              }
            } else if (eptr->type == T_TAPERAPR) {
              taperapr = (TAPERAPR *)eptr->p_elem;
              if (taperapr->sticky) {
                maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                maxamp->x_max = x_max = taperapr->xmax[taperapr->e2Index];
                maxamp->y_max = y_max = taperapr->ymax[taperapr->e2Index];
                maxamp->elliptical = elliptical = 0;
                maxampOpenCode = 0;
                maxamp->openSide = NULL;
                maxamp->exponent = maxampExponent = 0;
                maxamp->yExponent = maxampYExponent = 0;
              }
            } else if (eptr->type == T_CHARGE) {
              if (elementsTracked != 0 && !warnedAboutChargePosition) {
                warnedAboutChargePosition = 1;
                if (eptr->pred && eptr->pred->name) {
                  char buffer[16384];
                  snprintf(buffer, 16384, "Preceeded by %s.", eptr->pred->name);
                  printWarning("CHARGE element is not at the start of the beamline.", buffer);
                } else
                  printWarning("CHARGE element is not at the start of the beamline.", "Preceeded by ?.");
              }
              if (charge != NULL) {
                printf("Fatal error: multipole CHARGE elements in one beamline.\n");
                fflush(stdout);
                exitElegant(1);
              }
              charge = (CHARGE *)eptr->p_elem;
              charge->macroParticleCharge = 0;
#if !SDDS_MPI_IO
              if (nOriginal)
                charge->macroParticleCharge = charge->charge / (nOriginal);
#else
              if (notSinglePart) {
                if (total_nOriginal)
                  charge->macroParticleCharge = charge->charge / (total_nOriginal);
              } else {
                if (nOriginal)
                  charge->macroParticleCharge = charge->charge / (nOriginal);
              }
#endif
              if (charge->chargePerParticle)
                charge->macroParticleCharge = charge->chargePerParticle;
              if (charge->macroParticleCharge < 0)
                bombElegantVA("Error: CHARGE element should specify the quantity of charge (in Coulombs) without the sign. Specified value is %g\n", charge->charge);
            }
            eptr = eptr->succ;
            i_elem++;
          }
          z = z_travel = startElem->end_pos;
          startElem = NULL;
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 0.9.9\n");
#endif
#if TURBO_TRACKING
        bool skippable = !(flags & (TEST_PARTICLES + CLOSED_ORBIT_TRACKING + OPTIMIZING));
#endif
        while (eptr && (nToTrack || (USE_MPI && notSinglePart))) {
#if TURBO_TRACKING
          if (skippable) {
            if (run->checkBeamStructure && beam)
              checkBeamStructure(beam);
            while (eptr && eptr->ignore) {
              eptr = eptr->succ;
            }
          }
#else
          if (run->checkBeamStructure && beam && !(flags & (TEST_PARTICLES + CLOSED_ORBIT_TRACKING + OPTIMIZING)))
            checkBeamStructure(beam);
          if (eptr->ignore && !(flags & (TEST_PARTICLES + CLOSED_ORBIT_TRACKING + OPTIMIZING))) {
            eptr = eptr->succ;
            continue;
          }
#endif
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 1: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (run->showElementTiming) {
            elementTimingActive = 1;
            tStart = delapsed_time();
          }
          if (run->monitorMemoryUsage)
            memoryBefore = memoryUsage();
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 2: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (trackingOmniWedgeFunction) {
#ifdef HAVE_GPU
            coord = forceParticlesToCpu("trackingOmniWedgeFunction");
#endif
            (*trackingOmniWedgeFunction)(coord, nToTrack, i_pass, i_elem, beamline->n_elems, eptr, P_central);
          }
          if (trackingWedgeFunction && eptr == trackingWedgeElement) {
#ifdef HAVE_GPU
            coord = forceParticlesToCpu("trackingWedgeFunction");
#endif
            (*trackingWedgeFunction)(coord, nToTrack, i_pass, P_central);
          }
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif

          classFlags = entity_description[eptr->type].flags;
          elementsTracked++;
          log_entry("do_tracking.2.2.0");
          if (!eptr->name) {
            printf("error: element ending at %em has NULL name pointer\n", eptr->end_pos);
            fflush(stdout);
            if (eptr->pred && eptr->pred->name) {
              printf("previous element is %s\n", eptr->pred->name);
              fflush(stdout);
            } else if (eptr->succ && eptr->succ->name) {
              printf("next element is %s\n", eptr->succ->name);
              fflush(stdout);
            }
            abort();
          }
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.1: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (!eptr->p_elem && !run->concat_order) {
            printf("element %s has NULL p_elem pointer", eptr->name);
            fflush(stdout);
            exitElegant(1);
          }
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.2: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (eptr->type <= 0 || eptr->type >= N_TYPES) {
            printf("element %s has type %ld--not recognized/not allowed\n", eptr->name, eptr->type);
            fflush(stdout);
            exitElegant(1);
          }
          log_exit("do_tracking.2.2.0");
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.3: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif

          log_entry("do_tracking.2.2.1");

#ifdef HAVE_GPU
          setElementGpuData((void *)eptr);
#  ifndef GPU_VERIFY
          /* Ensure coord array is not used during GPU operations. */
          if (getGpuBase()->elementOnGpu)
            coord = NULL;
          else
            coord = getGpuBase()->coord;
#  endif
#endif

#ifdef SORT
          if (!USE_MPI || needSort)
            if (nToTrackAtLastSort > nToTrack) { /* indicates more particles are lost, need sort */
              if (beam && beam->bunchFrequency != 0)
                printWarning("particle ID sort not being performed because bunch frequency is nonzero.", NULL);
              else {
#  ifdef HAVE_GPU
                if (getElementOnGpu())
                  sortByPID(nToTrack);
                else {
#  endif
                  qsort(coord[0], nToTrack, totalPropertiesPerParticle * sizeof(double), comp_IDs);
                  if (accepted != NULL)
                    qsort(accepted[0], nToTrack, totalPropertiesPerParticle * sizeof(double), comp_IDs);
#  ifdef HAVE_GPU
                }
#  endif
                nToTrackAtLastSort = nToTrack;
                needSort = 0;
              }
            }
#endif
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.4: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (sums_vs_z && *sums_vs_z && (!run->final_pass || i_pass == n_passes - 1) && !(flags & FINAL_SUMS_ONLY) && !(flags & TEST_PARTICLES)) {
            if (i_sums < 0)
              bombElegant("attempt to accumulate beam sums with negative index!", NULL);
#if defined(BEAM_SUMS_DEBUG)
            printMessageAndTime(stdout, "Accumulating beam sums\n");
#endif
            accumulate_beam_sums(*sums_vs_z + i_sums, coord, nToTrack, *P_central,
                                 charge ? charge->macroParticleCharge : 0.0,
                                 NULL, 0.0, 0.0, -1, -1, 0);
            (*sums_vs_z)[i_sums].z = z;
            (*sums_vs_z)[i_sums].pass = i_pass;
            /*
              if (run->backtrack && i_sums==0)
              (*sums_vs_z)[i_sums].z = beamline->elem.end_pos;
            */
#if defined(BEAM_SUMS_DEBUG)
            printMessageAndTime(stdout, "Done accumulating beam sums\n");
            printf("beam sums accumulated in slot %ld for %s at z=%em, sx=%e\n",
                   i_sums, name, z, sqrt((*sums_vs_z)[i_sums].sum2[0] / nLeft));
            fflush(stdout);
#endif
            i_sums++;
          }
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.5: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif
#if USE_MPI
          if (notSinglePart) {
            active = 0;
            if (classFlags & UNIPROCESSOR) {
              /* This element cannot be done in parallel. Only the master CPU will work. */
              if (myid == 0)
                active = 1;
              else if (eptr->type == T_SCRIPT) /* The slave processors will be notified if they need allocate new memory */
                active = 1;
              else
                active = 0;
              if (parallelStatus == trueParallel) {
                if (!partOnMaster) {
                  if (usefulOperation(eptr, flags, i_pass)) {
                    /*
                      char buffer[16384];
                      snprintf(buffer, 16384, "%s (%s) is a serial element. It is not recommended for simulations with a large number of particles because of possible memory issues.", eptr->name, entity_name[eptr->type]);
                      printWarning(buffer, NULL);
                    */
                    gatherParticles(&coord, &nToTrack, &nLost, &accepted,
                                    n_processors, myid, &round);
                    if (isMaster)
                      nMaximum = nToTrack;
                    partOnMaster = 1;
                  }
                  /* update the nMaximum for recording the nLost on all the slave processors */
                  if (myid != 0)
                    nMaximum = nToTrack;
                }

                /* The element will change the state of particles. Scatter is required
                   for parallel computation */
                if (!(classFlags & (UNIDIAGNOSTIC & (~UNIPROCESSOR)))) {
                  parallelStatus = notParallel;
                }
              }
            } else {
              /* This element can be done in parallel. Only the slave CPUS will work. */
              if (myid != 0)
                active = 1;
              else { /* myid == 0 */
                if (!(classFlags & MPALGORITHM)) {
                  active = 0;
                } else /* The master CPU needs to participate communications */
                  active = 1;
              }
              if ((balanceStatus == badBalance) && (parallelStatus == trueParallel)) {
                gatherParticles(&coord, &nToTrack, &nLost, &accepted, n_processors, myid, &round);
                nMaximum = nToTrack;
                nLeft = nToTrack;
              }
              /* Particles will be scattered in startMode, bad balancing status or notParallel state */
              if ((balanceStatus == badBalance) || (parallelStatus == notParallel)) {
#  if USE_MPI && MPI_DEBUG
                printf("Scattering particles (1): nToTrack = %ld\n", nToTrack);
#  endif
                scatterParticles(coord, &nToTrack, accepted, n_processors, myid,
                                 balanceStatus, my_rate, nParPerElements, round, lostSinceSeqMode, &distributed, &reAllocate, P_central);
#  if USE_MPI && MPI_DEBUG
                printf("Scattering particles done: nToTrack = %ld\n", nToTrack);
#  endif
                nLeft = nToTrack;
                if (myid != 0) {
                  /* update the nMaximum for recording the nLost on all the slave processors */
                  nMaximum = nToTrack;
                }
                if (balanceStatus != startMode)
                  balanceStatus = goodBalance;
              } else if (balanceStatus == startMode) {
                /* For the first pass, scatter when it is not in parallel mode */
                if (parallelStatus != trueParallel) {
#  if USE_MPI && MPI_DEBUG
                  printf("Scattering particles (2): nToTrack = %ld\n", nToTrack);
#  endif
                  scatterParticles(coord, &nToTrack, accepted, n_processors, myid,
                                   balanceStatus, my_rate, nParPerElements, round, lostSinceSeqMode, &distributed, &reAllocate, P_central);
#  if USE_MPI && MPI_DEBUG
                  printf("Scattering particles done: nToTrack = %ld\n", nToTrack);
#  endif
                  nLeft = nToTrack;
                  if (myid != 0) {
                    /* update the nMaximum for recording the nLost on all the slave processors */
                    nMaximum = nToTrack;
                  }
                }
              }
              parallelStatus = trueParallel;
              lostSinceSeqMode = 0;
              if (myid != 0) {
                if (!(classFlags & MPALGORITHM)) {
                  /* We do not count time spent on those elements which need collective communications,
                     as it will be the time spent on synchronization instead of computation */
                  nElements++;
                  /* count the total number of particles tracked by all of the elements for each pass */
                  nParElements = nParElements + nToTrack;
                  start_wtime = MPI_Wtime();
                }
              }
              partOnMaster = 0;
            }
          }
#endif
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 3.6: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
          fflush(stdout);
#endif

#if defined(BEAM_SUMS_DEBUG)
          name = eptr->name;
#endif
          last_z = z;
          if (entity_description[eptr->type].flags & HAS_LENGTH && eptr->p_elem) {
            z += ((DRIFT *)eptr->p_elem)->length;
            z_travel += ((DRIFT *)eptr->p_elem)->length;
          } else {
            z += eptr->end_pos - eptr->beg_pos;
            z_travel += eptr->end_pos - eptr->beg_pos;
          }
#if TURBO_NOFLUSH
#else
          fflush(stdout);
#endif
          /* fill a structure that can be used to pass to other routines
           * information on the tracking context
           */
#if TURBO_STRINGS
          trackingContext.elementName = eptr->name;
#else
          strncpy(trackingContext.elementName, eptr->name, CONTEXT_BUFSIZE);
#endif
          trackingContext.element = eptr;
          trackingContext.elementOccurrence = eptr->occurence;
          trackingContext.sliceAnalysis = sliceAnalysis ? (sliceAnalysis->finalValuesOnly ? NULL : sliceAnalysis) : NULL;
          trackingContext.zStart = last_z;
          trackingContext.zEnd = z;
          trackingContext.step = step;
          trackingContext.elementType = eptr->type;
          trackingContext.flags = flags;

          log_exit("do_tracking.2.2.1");
          if (eptr->p_elem || eptr->matrix) {
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 4.0: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            if ((run->print_statistics > 0 || (run->print_statistics < 0 && (-run->print_statistics) <= (i_pass + 1))) && !(flags & TEST_PARTICLES)) {
              char buffer[16384];
#ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 4.1: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#endif
#if USE_MPI
              if (!partOnMaster && notSinglePart) {
                if (isMaster)
                  nToTrack = 0;
                if (beam)
                  MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
              }
#endif
#ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 4.2: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#endif
              snprintf(buffer, 16384, "Starting %s#%ld (%s) at s=%le to %le m, pass %ld, %ld particles, memory %ld kB\n",
                       eptr->name, eptr->occurence, entity_name[eptr->type],
                       last_z, eptr->end_pos, i_pass,
#if USE_MPI
                       myid == 0 ? (beam ? beam->n_to_track_total : -1) : nToTrack,
#else
                       nToTrack,
#endif
                       memoryUsage());
              printMessageAndTime(stdout, buffer);
              fflush(stdout);
            } /* print statistics */
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 4.3: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            /* show_dE = 0; */
            nLeft = nToTrack; /* in case it isn't set by the element tracking */
            if (eptr == eptrCLMatrix) {
              /* This element is the place-holder for the chromatic linear matrix or
               * the longitudinal-only matrix
               */
              if ((!USE_MPI || !notSinglePart) || (USE_MPI && (myid != 0))) {
                /* Only the slave CPUs will work on this part */
                if (flags & LINEAR_CHROMATIC_MATRIX) {
#ifdef HAVE_GPU
                  coord = forceParticlesToCpu("trackWithIndividualizedLinearMatrix");
#endif
                  nLeft = trackWithIndividualizedLinearMatrix(coord, nToTrack, accepted,
                                                              *P_central, z, eptrCLMatrix,
                                                              beamline->twiss0, beamline->tune,
                                                              beamline->chromaticity,
                                                              beamline->chrom2, beamline->chrom3,
                                                              beamline->dbeta_dPoP, beamline->dalpha_dPoP,
                                                              beamline->alpha, beamline->eta2, NULL);
                } else {
#ifdef HAVE_GPU
                  coord = forceParticlesToCpu("trackLongitudinalOnlyRing");
#endif
                  trackLongitudinalOnlyRing(coord, nToTrack,
                                            eptrCLMatrix->matrix,
                                            beamline->alpha);
                }
              }
            } /* linear-only matrix tracking */
            else if (eptr->type == T_BRANCH) {
              BRANCH *branch;
              long choice = 0;
              branch = (BRANCH *)(eptr->p_elem);
              if (i_pass == passOffset)
                branch->privateCounter = branch->counter;
              if (flags & TEST_PARTICLES) {
                choice = branch->defaultToElse;
              } else {
                if ((branch->interval > 0 && (i_pass - branch->offset) % branch->interval) ||
                    branch->privateCounter > 0)
                  choice = 1;
              }
              if (choice == 0) {
                if (!branch->beptr1)
                  bombElegant("No element pointer defined for BRANCH (1)---seek expert help!", NULL);
                eptr = branch->beptr1->pred;
                z = branch->beptr1->beg_pos;
                last_z = z;
                if (branch->verbosity && !(flags & TEST_PARTICLES)) {
                  printf("Branching to %s on pass %ld (z->%le)\n", branch->beptr1->name, i_pass, last_z);
                  fflush(stdout);
                }
              } else {
                if (!branch->beptr2)
                  bombElegant("No element pointer defined for BRANCH (2)---seek expert help!", NULL);
                eptr = branch->beptr2->pred;
                z = branch->beptr2->beg_pos;
                last_z = z;
                branch->privateCounter--;
                if (branch->verbosity && !(flags & TEST_PARTICLES)) {
                  printf("Branching to %s on pass %ld (z->%le)\n", branch->beptr2->name, i_pass, z);
                  fflush(stdout);
                }
              }
            } /* BRANCH element */
            else if (entity_description[eptr->type].flags & MATRIX_TRACKING &&
                     !(flags & IBS_ONLY_TRACKING)) {
#ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 5.0: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#endif
              if (!(entity_description[eptr->type].flags & HAS_MATRIX))
                bombElegant("attempt to matrix-multiply for element with no matrix!", NULL);
              if (!eptr->matrix) {
                if (!(eptr->matrix = compute_matrix(eptr, run, NULL)))
                  bombElegant("no matrix for element that must have matrix", NULL);
              }
              if (eptr->matrix->C[5] != 0) {
                char buffer[16384];
                snprintf(buffer, 16384, " Element in question is %s, C5=%le\n", eptr->name, eptr->matrix->C[5]);
                printWarning("Matrix with C5!=0 detected in matrix multiplier. All particles considered lost!", buffer);
                print_elem(stdout, eptr);
                nLeft = 0;
              } else {
                if (run->print_statistics > 1 && !(flags & TEST_PARTICLES)) {
                  printf("Tracking matrix for %s\n", eptr->name);
                  fflush(stdout);
                  if (run->print_statistics > 2) {
                    print_elem(stdout, eptr);
                    print_matrices(stdout, "", eptr->matrix);
                  }
                }
                if (flags & CLOSED_ORBIT_TRACKING) {
                  switch (eptr->type) {
                  case T_MONI:
                  case T_HMON:
                  case T_VMON:
                    storeMonitorOrbitValues(eptr, coord, nToTrack);
                    break;
                  default:
                    break;
                  }
                } else {
                  switch (eptr->type) {
                  case T_MONI:
                    if (((MONI *)eptr->p_elem)->storeTurnByTurn)
                      storeBPMReading(eptr, coord, nToTrack, *P_central);
                    break;
                  case T_HMON:
                    if (((HMON *)eptr->p_elem)->storeTurnByTurn)
                      storeBPMReading(eptr, coord, nToTrack, *P_central);
                    break;
                  case T_VMON:
                    if (((VMON *)eptr->p_elem)->storeTurnByTurn)
                      storeBPMReading(eptr, coord, nToTrack, *P_central);
                    break;
                  default:
                    break;
                  }
                  if (IS_MONITOR(eptr->type) && eptr->end_pos > sMaxTransmittedMonitor) {
#if USE_MPI
                    long nToTrackTotal = nToTrack;
                    if (beam) {
                      if (!partOnMaster && notSinglePart)
                        MPI_Allreduce(&nToTrack, &nToTrackTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
                    }
                    if (nToTrackTotal > 0)
                      rpn_store(sMaxTransmittedMonitor = eptr->end_pos, NULL, sMaxTransmittedMonitorMemory);
#else
                    if (nToTrack > 0)
                      rpn_store(sMaxTransmittedMonitor = eptr->end_pos, NULL, sMaxTransmittedMonitorMemory);
#endif
                  }
                }
                /* Only the slave CPUs will track */
                if ((!USE_MPI || !notSinglePart) || (USE_MPI && (myid != 0)))
                  track_particles(coord, eptr->matrix, coord, nToTrack);
              }
            }      /* matrix-based tracking */
            else { /* normal tracking ends up here */
              long type;
#ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 6.0: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#endif
              if (run->print_statistics > 1 && !(flags & TEST_PARTICLES)) {
                printf("Tracking element: ");
                fflush(stdout);
                if (run->print_statistics > 2)
                  print_elem(stdout, eptr);
              }
              type = eptr->type;
              if (flags & IBS_ONLY_TRACKING) {
                switch (type) {
                case T_IBSCATTER:
                case T_WATCH:
                case T_SLICE_POINT:
                case T_CLEAN:
                case T_RCOL:
                case T_CHARGE:
                  break;
                default:
                  type = -1;
                  break;
                }
              }
#ifdef USE_MPE
              bytebuf_pos = 0;
              MPE_Log_event(event2a, 0, NULL);
              MPE_Log_pack(bytebuf, &bytebuf_pos, 's', strlen(entity_name[eptr->type]), entity_name[eptr->type]);
#endif
              if (active && (((!USE_MPI || !notSinglePart) && nParticlesStartPass) ||
#if USE_MPI
                             (beam && beam->n_to_track_total) || (classFlags & RUN_ZERO_PARTICLES) ||
#endif
                             (!USE_MPI && nToTrack))) {
                switch (type) {
                case -1:
                  break;
                case T_CHARGE:
                  if ((i_pass == passOffset && !startElem) || ((CHARGE *)(eptr->p_elem))->allowChangeWhileRunning) {
                    if (elementsTracked != 0 && !warnedAboutChargePosition) {
                      warnedAboutChargePosition = 1;
                      if (eptr->pred && eptr->pred->name) {
                        char buffer[16384];
                        snprintf(buffer, 16384, "Preceeded by %s.", eptr->pred->name);
                        printWarning("CHARGE element is not at the start of the beamline.", buffer);
                      } else
                        printWarning("CHARGE element is not at the start of the beamline.", "Preceeded by ?.");
                    }
                    if (charge != NULL && !(((CHARGE *)(eptr->p_elem))->allowChangeWhileRunning && charge == ((CHARGE *)(eptr->p_elem)))) {
                      printf("Fatal error: multiple CHARGE elements in one beamline.\n");
                      fflush(stdout);
                      exitElegant(1);
                    }
                    charge = (CHARGE *)eptr->p_elem;
                    charge->idSlotsPerBunch = beam ? beam->id_slots_per_bunch : 0;
                    charge->macroParticleCharge = 0;
#if !SDDS_MPI_IO
                    if (nOriginal)
                      charge->macroParticleCharge = charge->charge / (nOriginal);
#else
                    if (notSinglePart) {
                      if (total_nOriginal)
                        charge->macroParticleCharge = charge->charge / (total_nOriginal);
                    } else {
                      if (nOriginal)
                        charge->macroParticleCharge = charge->charge / (nOriginal);
                    }
#endif
                  }
                  if (charge->chargePerParticle)
                    charge->macroParticleCharge = charge->chargePerParticle;
                  if (charge->macroParticleCharge < 0)
                    bombElegantVA("Error: CHARGE element should specify the quantity of charge (in Coulombs) without the sign. Specified value is %g\n", charge->charge);
                  break;
                case T_MARK:
                  if ((flags & (CLOSED_ORBIT_TRACKING | OPTIMIZING)) && ((MARK *)eptr->p_elem)->fitpoint && i_pass == n_passes - 1) {
                    /*
                      if (beamline->flags&BEAMLINE_TWISS_WANTED) {
                      if (!(beamline->flags&BEAMLINE_TWISS_DONE))
                      update_twiss_parameters(run, beamline, NULL);
                      store_fitpoint_twiss_parameters((MARK*)eptr->p_elem, eptr->name,
                      eptr->occurence, eptr->twiss);
                      }
                    */
                    if (!(flags & CLOSED_ORBIT_TRACKING)) {
                      if (isMaster || !notSinglePart)
                        store_fitpoint_matrix_values((MARK *)eptr->p_elem, eptr->name,
                                                     eptr->occurence, eptr->accumMatrix);
                      store_fitpoint_beam_parameters((MARK *)eptr->p_elem, eptr->name, eptr->occurence,
                                                     coord, nToTrack, *P_central);
                    }
                    if (flags & CLOSED_ORBIT_TRACKING)
                      storeMonitorOrbitValues(eptr, coord, nToTrack);
                  }
                  break;
                case T_RECIRC:
                  /* Recognize and record recirculation point.  */
                  if (i_pass == passOffset) {
                    i_sums_recirc = i_sums - 1;
                    z_recirc = last_z;
                  }
                  break;
                case T_RFDF:
                  if (!(flags & TIME_DEPENDENCE_OFF) || (flags & CLOSED_ORBIT_TRACKING))
                    track_through_rf_deflector(coord, (RFDF *)eptr->p_elem,
                                               coord, nToTrack, *P_central,
                                               beamline->revolution_length, z,
                                               i_pass);
                  else
                    exactDrift(coord, nToTrack, ((RFDF *)eptr->p_elem)->length);
                  break;
                case T_MRFDF:
                  if (!(flags & TIME_DEPENDENCE_OFF))
                    track_through_multipole_deflector(coord, (MRFDF *)eptr->p_elem,
                                                      coord, nToTrack, *P_central, i_pass);
                  break;
                case T_SHRFDF:
                  if (!(flags & TIME_DEPENDENCE_OFF))
                    track_through_space_harmonic_deflector(coord, (SHRFDF *)eptr->p_elem,
                                                           coord, nToTrack, *P_central);
                  break;
                case T_RFTM110:
                  if (!(flags & TIME_DEPENDENCE_OFF))
                    track_through_rftm110_deflector(coord, (RFTM110 *)eptr->p_elem,
                                                    coord, nToTrack, *P_central,
                                                    beamline->revolution_length, z,
                                                    i_pass);
                  break;
                case T_RMDF:
                  if (!(flags & TIME_DEPENDENCE_OFF))
                    track_through_ramped_deflector(coord, (RMDF *)eptr->p_elem,
                                                   coord, nToTrack, *P_central);
                  else
                    drift_beam(coord, nToTrack, ((RMDF *)eptr->p_elem)->length, run->default_order);
                  break;
                case T_RFTMEZ0:
                  nLeft = motion(coord, nToTrack, eptr->p_elem, eptr->type, P_central,
                                 &dgamma, dP, accepted, last_z);
                  /* show_dE = 1; */
                  break;
                case T_TMCF:
                case T_CEPL:
                case T_TWPL:
                  if (!(flags & TIME_DEPENDENCE_OFF)) {
                    nLeft = motion(coord, nToTrack, eptr->p_elem, eptr->type, P_central,
                                   &dgamma, dP, accepted, last_z);
                    /* show_dE = 1; */
                  } else
                    drift_beam(coord, nToTrack, ((TW_LINAC *)eptr->p_elem)->length, run->default_order);
                  break;
                case T_MAPSOLENOID:
                  nLeft = motion(coord, nToTrack, eptr->p_elem, eptr->type, P_central,
                                 &dgamma, dP, accepted, last_z);
                  break;
                case T_TWLA:
                case T_TWMTA:
                  nLeft = motion(coord, nToTrack, eptr->p_elem, eptr->type, P_central,
                                 &dgamma, dP, accepted, last_z);
                  /* show_dE = 1; */
                  break;
                case T_RCOL:
                  if (flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))
                    drift_beam(coord, nToTrack, ((RCOL *)eptr->p_elem)->length, run->default_order);
                  else {
                    nLeft = rectangular_collimator(coord, (RCOL *)eptr->p_elem, nToTrack, accepted, last_z, *P_central, eptr);
                  }
                  break;
                case T_ECOL:
                  if (flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))
                    drift_beam(coord, nToTrack, ((ECOL *)eptr->p_elem)->length, run->default_order);
                  else
                    nLeft = elliptical_collimator(coord, (ECOL *)eptr->p_elem, nToTrack, accepted, last_z, *P_central, eptr);
                  /* printf("After ECOL: nLeft = %ld, nToTrack = %ld\n", nLeft, nToTrack); */
                  break;
                case T_APCONTOUR:
                  apcontour = (APCONTOUR *)eptr->p_elem;
                  if (!(apcontour->initialized))
                    initializeApContour(apcontour);
                  if (apcontour->cancel || (apcontour->holdOff && !apcontour->sticky))
                    apcontour = NULL;
                  else {
                    if ((flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES)) || apcontour->holdOff)
                      drift_beam(coord, nToTrack, ((APCONTOUR *)eptr->p_elem)->length, run->default_order);
                    else
                      nLeft = trackThroughApContour(coord, apcontour, nToTrack, accepted, last_z, *P_central, eptr);
                    if (!apcontour->sticky)
                      apcontour = NULL;
                  }
                  break;
                case T_TAPERAPC:
                  taperapc = (TAPERAPC *)eptr->p_elem;
                  if (flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))
                    drift_beam(coord, nToTrack, ((TAPERAPC *)eptr->p_elem)->length, run->default_order);
                  else {
                    nLeft = trackThroughTaperApCirc(coord, (TAPERAPC *)eptr->p_elem, nToTrack, accepted, last_z, *P_central, eptr);
                    if (taperapc->sticky) {
                      maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                      maxamp->x_max = maxamp->y_max = x_max = y_max = taperapc->r[taperapc->e2Index];
                      maxamp->elliptical = elliptical = 1;
                      maxampOpenCode = 0;
                      maxamp->openSide = NULL;
                      maxamp->exponent = maxamp->yExponent = maxampExponent = maxampYExponent = 2;
                    }
                  }
                  break;
                case T_TAPERAPE:
                  taperape = (TAPERAPE *)eptr->p_elem;
                  if (flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))
                    drift_beam(coord, nToTrack, taperape->length, run->default_order);
                  else {
                    nLeft = trackThroughTaperApElliptical(coord, taperape, nToTrack, accepted, last_z, *P_central, eptr);
                    if (taperape->sticky) {
                      maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                      maxamp->x_max = x_max = taperape->a[taperape->e2Index];
                      maxamp->y_max = y_max = taperape->b[taperape->e2Index];
                      maxamp->elliptical = elliptical = 1;
                      maxampOpenCode = 0;
                      maxamp->openSide = NULL;
                      maxamp->exponent = maxampExponent = taperape->xExponent;
                      maxamp->yExponent = maxampYExponent = taperape->yExponent;
                    }
                  }
                  break;
                case T_TAPERAPR:
                  taperapr = (TAPERAPR *)eptr->p_elem;
                  if (flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))
                    drift_beam(coord, nToTrack, taperapr->length, run->default_order);
                  else {
                    nLeft = trackThroughTaperApRectangular(coord, taperapr, nToTrack, accepted, last_z, *P_central, eptr);
                    if (taperapr->sticky) {
                      maxamp = &maxampBuf; /* needed by KQUAD, CSBEND, etc */
                      maxamp->x_max = x_max = taperapr->xmax[taperapr->e2Index];
                      maxamp->y_max = y_max = taperapr->ymax[taperapr->e2Index];
                      maxamp->elliptical = elliptical = 0;
                      maxampOpenCode = 0;
                      maxamp->openSide = NULL;
                      maxamp->exponent = maxampExponent = 0;
                      maxamp->yExponent = maxampYExponent = 0;
                    }
                  }
                  break;
                case T_CLEAN:
                  if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES)))
                    nLeft = remove_outlier_particles(coord, (CLEAN *)eptr->p_elem,
                                                     nToTrack, accepted, z, *P_central);
                  break;
                case T_SCRAPER:
                  if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))) {

#ifdef HAVE_GPU
                    {
                      long dflag[2] = {0, 0};
                      if (((SCRAPER *)eptr->p_elem)->direction & DIRECTION_X) {
                        dflag[0] = ((SCRAPER *)eptr->p_elem)->direction & DIRECTION_PLUS_X ? 1 : 0;
                        dflag[1] = ((SCRAPER *)eptr->p_elem)->direction & DIRECTION_MINUS_X ? 1 : 0;
                      } else if (((SCRAPER *)eptr->p_elem)->direction & DIRECTION_Y) {
                        dflag[0] = ((SCRAPER *)eptr->p_elem)->direction & DIRECTION_PLUS_Y ? 1 : 0;
                        dflag[1] = ((SCRAPER *)eptr->p_elem)->direction & DIRECTION_MINUS_Y ? 1 : 0;
                      }
                      if (dflag[0] && dflag[1]) {
                        coord = forceParticlesToCpu("beam_scraper");
                      }
                    }
#endif

                    nLeft = beam_scraper(coord, (SCRAPER *)eptr->p_elem, nToTrack, accepted, last_z, *P_central, eptr);
                  } else {
                    exactDrift(coord, nToTrack, ((SCRAPER *)eptr->p_elem)->length);
                  }
                  break;
                case T_SPEEDBUMP:
                  if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))) {
                    nLeft = track_through_speedbump(coord, (SPEEDBUMP *)eptr->p_elem, nToTrack, accepted, last_z, *P_central, eptr);
                  } else {
                    exactDrift(coord, nToTrack, ((SPEEDBUMP *)eptr->p_elem)->length);
                  }
                  break;
                case T_PFILTER:
                  if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES)))
                    nLeft = track_through_pfilter(coord, (PFILTER *)eptr->p_elem, nToTrack,
                                                  accepted, z, *P_central);
                  break;
                case T_CENTER:
                  center_beam(coord, (CENTER *)eptr->p_elem, nToTrack, i_pass, *P_central);
                  break;
                case T_REMCOR:
                  remove_correlations(coord, (REMCOR *)eptr->p_elem, nToTrack);
                  break;
                case T_RFCA:
                  nLeft = simple_rf_cavity(coord, nToTrack, (RFCA *)eptr->p_elem, accepted, P_central, z_travel);
                  break;
                case T_RFCW:
                  nLeft = track_through_rfcw(coord, nToTrack, (RFCW *)eptr->p_elem, accepted, P_central, z_travel,
                                             run, i_pass, charge);
                  break;
                case T_MODRF:
                  modulated_rf_cavity(coord, nToTrack, (MODRF *)eptr->p_elem, *P_central, z_travel);
                  break;
                case T_WATCH:
#ifdef USE_MPE
                  MPE_Log_event(event1a, 0, "start watch"); /* record time spent on I/O operations */
#endif
#if USE_MPI
                  if (!notSinglePart) /* When each processor tracks the beam independently, the watch point will be disabled in Pelegant */
                    break;
                  if (!partOnMaster && notSinglePart) { /* Update the total particle number to get the correct charge */
                    if (isMaster)
                      nToTrack = 0;
                    if (beam)
                      MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                  } else { /* singlePart tracking or partOnMaster */
                    if (beam)
                      beam->n_to_track_total = nToTrack;
                  }
#endif
                  if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
                    watch = (WATCH *)eptr->p_elem;
                    if (!watch->disable) {
                      watch_pt_seen = 1;
                      if (!watch->initialized)
                        set_up_watch_point(watch, run, eptr->occurence, eptr->pred ? eptr->pred->name : NULL,
                                           eptr->pred ? eptr->pred->occurence : 0, i_pass, beamline->elem, beam ? beam->id_slots_per_bunch : 0);
                      if (i_pass == passOffset && (n_passes / watch->interval) == 0) {
                        char buffer[16384];
                        snprintf(buffer, 16384,
                                 "Settings n_passes=%ld and INTERVAL=%ld prevent WATCH output to file %s",
                                 n_passes, watch->interval, watch->filename);
                        printWarning(buffer, NULL);
                      }
#if SDDS_MPI_IO
                      if (watch_not_allowed) {
                        dup2(fdStdout, fileno(stdout));
                        printf("/****************************************************************************/\n");
                        printf("Watch point can not be used for dynamic aperture searching with Pelegant\n");
                        printf("/****************************************************************************/\n");
                        fflush(stdout);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                      }
#endif
                      fflush(stdout);
                      switch (watch->mode_code) {
                      case WATCH_COORDINATES:
                        dump_watch_particles(watch, step, i_pass, coord, nToTrack, *P_central,
                                             beamline->revolution_length,
                                             charge ? charge->macroParticleCharge : 0.0, z,
                                             beam ? beam->id_slots_per_bunch : 0);
                        break;
                      case WATCH_PARAMETERS:
                      case WATCH_CENTROIDS:
                        dump_watch_parameters(watch, step, i_pass, n_passes, coord, nToTrack,
#if SDDS_MPI_IO
                                              total_nOriginal,
#else
                                              nOriginal,
#endif
                                              *P_central,
                                              beamline->revolution_length, z,
                                              charge ? charge->macroParticleCharge : 0.0);

                        break;
                      case WATCH_FFT:
                        if (i_pass >= watch->start_pass && (i_pass - watch->start_pass) % watch->interval == 0 &&
                            (watch->end_pass < 0 || i_pass <= watch->end_pass)) {
#if SDDS_MPI_IO
                          /* This part will be done in serial for now. A parallel version of FFT could be used here */
                          if (!partOnMaster && notSinglePart) {
                            char buffer[16384];
                            snprintf(buffer, 16384, "%s (%s FFT) is a serial element. It is not recommended for simulations with a large number of particles because of possible memory issues.", eptr->name, entity_name[eptr->type]);
                            printWarning(buffer, NULL);
                          }
                          gatherParticles(&coord, &nToTrack, &nLost, &accepted, n_processors, myid, &round);
                          if (isMaster)
#endif
                            dump_watch_FFT(watch, step, i_pass, n_passes, coord, nToTrack, nOriginal, *P_central);
#if SDDS_MPI_IO
                          if (!partOnMaster && notSinglePart) {
#  if USE_MPI && MPI_DEBUG
                            printf("Scattering particles (3): nToTrack = %ld\n", nToTrack);
#  endif
                            scatterParticles(coord, &nToTrack, accepted, n_processors, myid,
                                             balanceStatus, my_rate, nParPerElements, round,
                                             lostSinceSeqMode, &distributed, &reAllocate, P_central);
#  if USE_MPI && MPI_DEBUG
                            printf("Scattering particles done: nToTrack = %ld\n", nToTrack);
#  endif
                            nLeft = nToTrack;
                          }
#endif
                          break;
                        }
                      }
                    }
                  }
#ifdef USE_MPE
                  MPE_Log_event(event1b, 0, "end watch");
#endif
                  break;
                case T_SLICE_POINT:
#if USE_MPI
                  if (!notSinglePart) /* When each processor tracks the beam independently, the slice point will be disabled in Pelegant */
                    break;
                  if (!partOnMaster && notSinglePart) { /* Update the total particle number to get the correct charge */
                    if (isMaster)
                      nToTrack = 0;
                    if (beam)
                      MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                  } else { /* singlePart tracking or partOnMaster */
                    if (beam)
                      beam->n_to_track_total = nToTrack;
                  }
#endif
                  if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
                    slicePoint = (SLICE_POINT *)eptr->p_elem;
                    if (!slicePoint->disable) {
                      watch_pt_seen = 1; /* sic */
                      if (!slicePoint->initialized)
                        set_up_slice_point(slicePoint, run, eptr->occurence, eptr->pred ? eptr->pred->name : NULL, beam ? beam->id_slots_per_bunch : 0);
                      if (i_pass == passOffset && (n_passes / slicePoint->interval) == 0) {
                        char buffer[16384];
                        snprintf(buffer, 16384,
                                 "Settings n_passes=%ld and INTERVAL=%ld prevent SLICE output to file %s for element %s",
                                 n_passes, slicePoint->interval, slicePoint->filename, eptr->name);
                        printWarning(buffer, NULL);
                      }
#if SDDS_MPI_IO
                      if (watch_not_allowed) {
                        dup2(fdStdout, fileno(stdout));
                        printf("/****************************************************************************/\n");
                        printf("Slice point can not be used for dynamic aperture searching with Pelegant\n");
                        printf("/****************************************************************************/\n");
                        fflush(stdout);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                      }
#endif
                      dump_slice_analysis(slicePoint, step, i_pass, n_passes, coord, nToTrack, *P_central,
                                          beamline->revolution_length, z, charge ? charge->macroParticleCharge : 0.0);
                    }
                  }
                  break;
                case T_HISTOGRAM:
                  if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
                    histogram = (HISTOGRAM *)eptr->p_elem;
                    if (!histogram->disable) {
                      watch_pt_seen = 1; /* yes, this should be here */
                      if (!histogram->initialized)
                        set_up_histogram(histogram, run, eptr->occurence, beam ? beam->id_slots_per_bunch : 0);
                      if (i_pass == passOffset && (n_passes / histogram->interval) == 0) {
                        char buffer[16384];
                        snprintf(buffer, 16384,
                                 "Settings n_passes=%ld and INTERVAL=%ld prevent SLICE output to file %s for element %s",
                                 n_passes, histogram->interval, histogram->filename, eptr->name);
                        printWarning(buffer, NULL);
                      }
                      fflush(stdout);
                      if (i_pass >= histogram->startPass && (i_pass - histogram->startPass) % histogram->interval == 0) {
#if !SDDS_MPI_IO
                        dump_particle_histogram(histogram, step, i_pass, coord, nToTrack, *P_central,
                                                beamline->revolution_length,
                                                charge ? charge->macroParticleCharge * nToTrack : 0.0, z);
#else
                        dump_particle_histogram(histogram, step, i_pass, coord, nToTrack, *P_central,
                                                beamline->revolution_length,
                                                charge ? charge->macroParticleCharge * beam->n_to_track_total : 0.0, z);

#endif
                      }
                    }
                  }
                  break;
                case T_MHISTOGRAM:
                  if (!eptr->pred)
                    bombElegant("MHISTOGRAM should not be the first element of the beamline.", NULL);
                  if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
                    mhist = (MHISTOGRAM *)eptr->p_elem;
                    if (!mhist->disable) {
                      watch_pt_seen = 1; /* yes, this should be here */
                      if (i_pass == passOffset && (n_passes / mhist->interval) == 0) {
                        char buffer[16384];
                        snprintf(buffer, 16384,
                                 "Settings n_passes=%ld and INTERVAL=%ld prevent MHISTOGRAM output for element %s",
                                 n_passes, mhist->interval, eptr->name);
                        printWarning(buffer, NULL);
                      }
                      if (i_pass >= mhist->startPass && (i_pass - mhist->startPass) % mhist->interval == 0) {

                        ELEMENT_LIST *eptr0;
                        if (mhist->lumped) {
                          if (!mhist->initialized && eptr->occurence == 1)
                            set_up_mhist(mhist, run, 0);
                          eptr0 = beamline->elem;
                          while (eptr0) {
                            if (eptr0->type == eptr->type && strcmp(eptr0->name, eptr->name) == 0)
                              break;
                            eptr0 = eptr0->succ;
                          }
                        } else {
                          if (!mhist->initialized)
                            set_up_mhist(mhist, run, eptr->occurence);
                          eptr0 = NULL;
                        }

                        mhist_table(eptr0, eptr, step, i_pass, coord, nToTrack, *P_central,
                                    beamline->revolution_length,
                                    charge ? charge->macroParticleCharge * nToTrack : 0.0, z);
                      }
                    }
                  }
                  break;
                case T_FTABLE:
                  ftable = (FTABLE *)eptr->p_elem;
                  field_table_tracking(coord, nToTrack, ftable, *P_central, run);
                  break;
                case T_BGGEXP:
                  trackBGGExpansion(coord, nToTrack, (BGGEXP *)eptr->p_elem, *P_central, accepted, NULL);
                  break;
                case T_BOFFAXE:
                  trackMagneticFieldOffAxisExpansion(coord, nToTrack, (BOFFAXE *)eptr->p_elem, *P_central, accepted, NULL);
                  break;
                case T_ROTATE:
                  if (((ROTATE *)eptr->p_elem)->on_pass < 0 || ((ROTATE *)eptr->p_elem)->on_pass == i_pass) {
                    if (((ROTATE *)eptr->p_elem)->excludeOptics) {
                      if (!(flags & TEST_PARTICLES)) {
                        VMATRIX *M;
                        M = rotation_matrix(((ROTATE *)eptr->p_elem)->tilt);
                        track_particles(coord, M, coord, nToTrack);
                        free_matrices(M);
                        free(M);
                      }
                    } else {
                      if (!(eptr->matrix))
                        eptr->matrix = rotation_matrix(((ROTATE *)eptr->p_elem)->tilt);
                      track_particles(coord, eptr->matrix, coord, nToTrack);
                    }
                  }
                  break;
                case T_MALIGN:
                  malign = (MALIGN *)eptr->p_elem;
                  if ((malign->on_pass == -1 || malign->on_pass == i_pass) && !(malign->excludeOrbit && (flags & TEST_PARTICLES)))
                    offset_beam(coord, nToTrack, (MALIGN *)eptr->p_elem, *P_central);
                  break;
                case T_PEPPOT:
                  nLeft = pepper_pot_plate(coord, (PEPPOT *)eptr->p_elem, nToTrack, accepted);
                  break;
                case T_ENERGY:
                  energy = (ENERGY *)eptr->p_elem;
                  if (energy->match_beamline) {
                    if ((flags & FIDUCIAL_BEAM_SEEN) && eptr->Pref_output_fiducial > 0)
                      /* Beamline momentum is defined.  Change particle reference momentum to match. */
                      set_central_momentum(coord, nToTrack, eptr->Pref_output_fiducial, P_central);
                    else
                      /* Compute new central momentum to match the average momentum of the particles. */
                      do_match_energy(coord, nToTrack, P_central, 0);
                    if (energy->match_particles)
                      bombElegant("can't match_beamline AND match_particles for ENERGY element", NULL);
                  } else if (energy->match_particles) {
                    /* change the particle momenta so that the centroid is the central momentum */
                    do_match_energy(coord, nToTrack, P_central, 1);
                  } else if (energy->central_energy)
                    /* Change particle reference momentum to match the given energy */
                    set_central_momentum(coord, nToTrack, sqrt(sqr(energy->central_energy + 1) - 1),
                                         P_central);
                  else if (energy->central_momentum)
                    /* Change particle reference momentum to match the given value */
                    set_central_momentum(coord, nToTrack, energy->central_momentum, P_central);
                  break;
                case T_MAXAMP:
                  if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES))) {
                    maxamp = (MAXAMP *)eptr->p_elem;
                    x_max = maxamp->x_max;
                    y_max = maxamp->y_max;
                    elliptical = maxamp->elliptical;
                    maxampOpenCode = determineOpenSideCode(maxamp->openSide);
                    maxampExponent = maxamp->exponent;
                    maxampYExponent = maxamp->yExponent;
                  }
                  break;
                case T_TRCOUNT:
                  /* >>>>> Needs to be updated */
                  /*
                   *n_original = nLeft;
                   if (accepted && i_pass==0)
                   copy_particles(accepted, coord, *n_original);
                  */
                  break;
                case T_ALPH:
                  if (!eptr->matrix && !(eptr->matrix = compute_matrix(eptr, run, NULL)))
                    bombElegant("no matrix for alpha magnet", NULL);
                  nLeft = alpha_magnet_tracking(coord, eptr->matrix, (ALPH *)eptr->p_elem, nToTrack,
                                                accepted, *P_central, z);
                  break;
                case T_MATR:
                  if (!eptr->matrix)
                    eptr->matrix = compute_matrix(eptr, run, NULL);
                  matr_element_tracking(coord, eptr->matrix, (MATR *)eptr->p_elem, nToTrack,
                                        z);
                  break;
                case T_EMATRIX:
                  if (!eptr->matrix)
                    eptr->matrix = compute_matrix(eptr, run, NULL);
                  ematrix_element_tracking(coord, eptr->matrix, (EMATRIX *)eptr->p_elem, nToTrack,
                                           z, P_central);
                  break;
                case T_ILMATRIX:
                  nLeft = trackWithIndividualizedLinearMatrix(coord, nToTrack, accepted, *P_central,
                                                              z, eptr,
                                                              NULL,
                                                              ((ILMATRIX *)eptr->p_elem)->tune,
                                                              ((ILMATRIX *)eptr->p_elem)->chrom,
                                                              ((ILMATRIX *)eptr->p_elem)->chrom2,
                                                              ((ILMATRIX *)eptr->p_elem)->chrom3,
                                                              ((ILMATRIX *)eptr->p_elem)->beta1,
                                                              ((ILMATRIX *)eptr->p_elem)->alpha1,
                                                              ((ILMATRIX *)eptr->p_elem)->alphac,
                                                              ((ILMATRIX *)eptr->p_elem)->eta1,
                                                              ((ILMATRIX *)eptr->p_elem));
                  break;
                case T_MULT:
                  nLeft = multipole_tracking(coord, nToTrack, (MULT *)eptr->p_elem, 0.0,
                                             *P_central, accepted, last_z);
                  break;
                case T_FMULT:
                  nLeft = fmultipole_tracking(coord, nToTrack, (FMULT *)eptr->p_elem, 0.0,
                                              *P_central, accepted, last_z);
                  break;
                case T_POLYNOMIALSERIES:
                  nLeft = polynomialSeries_tracking(coord, nToTrack, (POLYNOMIALSERIES *)eptr->p_elem, 0.0,
                                                    *P_central, accepted, z);
                  break;
                case T_KICKER:
                  if (flags & TIME_DEPENDENCE_OFF)
                    drift_beam(coord, nToTrack, ((KICKER *)eptr->p_elem)->length, run->default_order);
                  else
                    track_through_kicker(coord, nToTrack, (KICKER *)eptr->p_elem, *P_central, i_pass, run->default_order);
                  break;
                case T_MKICKER:
                  if (flags & TIME_DEPENDENCE_OFF)
                    drift_beam(coord, nToTrack, ((MKICKER *)eptr->p_elem)->length, run->default_order);
                  else
                    track_through_mkicker(coord, nToTrack, (MKICKER *)eptr->p_elem, *P_central, i_pass, run->default_order);
                  break;
                case T_KSBEND:
                  nLeft = track_through_kick_sbend(coord, nToTrack, (KSBEND *)eptr->p_elem, 0.0,
                                                   *P_central, accepted, z);
                  break;
                case T_CSBEND:
                  ((CSBEND *)eptr->p_elem)->edgeFlags = determine_bend_flags(eptr,
                                                                             ((CSBEND *)eptr->p_elem)->edge_effects[((CSBEND *)eptr->p_elem)->e1Index],
                                                                             ((CSBEND *)eptr->p_elem)->edge_effects[((CSBEND *)eptr->p_elem)->e2Index]);
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((CSBEND *)eptr->p_elem)->isr;
                    ((CSBEND *)eptr->p_elem)->isr = 0;
                  }

#ifdef HAVE_GPU
                  /* CSBEND with REFERENCE_CORRECTION currently doesn't work on the GPU */
                  if (((CSBEND *)eptr->p_elem)->referenceCorrection) {
                    coord = forceParticlesToCpu("track_through_csbend");
                  };
#endif

                  nLeft = track_through_csbend(coord, nToTrack, (CSBEND *)eptr->p_elem, 0.0,
                                               *P_central, accepted, last_z, NULL, run->rootname, maxamp, apcontour,
                                               &(run->apertureData), -1, eptr);
                  if (flags & TEST_PARTICLES)
                    ((CSBEND *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_CCBEND:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((CCBEND *)eptr->p_elem)->isr;
                    ((CCBEND *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = track_through_ccbend(coord, nToTrack, eptr, (CCBEND *)eptr->p_elem, *P_central, accepted,
                                               last_z, NULL, run->rootname, maxamp, apcontour, &(run->apertureData), -1, -1);
                  if (flags & TEST_PARTICLES)
                    ((CCBEND *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_LGBEND:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((LGBEND *)eptr->p_elem)->isr;
                    ((LGBEND *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = track_through_lgbend(coord, nToTrack, eptr, (LGBEND *)eptr->p_elem, *P_central, accepted,
                                               last_z, NULL, run->rootname, maxamp, apcontour, &(run->apertureData), -1, -1);
                  if (flags & TEST_PARTICLES)
                    ((LGBEND *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_CSRCSBEND:
                  ((CSRCSBEND *)eptr->p_elem)->edgeFlags = determine_bend_flags(eptr,
                                                                                ((CSRCSBEND *)eptr->p_elem)->edge_effects[((CSRCSBEND *)eptr->p_elem)->e1Index],
                                                                                ((CSRCSBEND *)eptr->p_elem)->edge_effects[((CSRCSBEND *)eptr->p_elem)->e2Index]);
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((CSRCSBEND *)eptr->p_elem)->isr;
                    ((CSRCSBEND *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = track_through_csbendCSR(coord, nToTrack, (CSRCSBEND *)eptr->p_elem, 0.0,
                                                  *P_central, accepted, last_z, z, charge, run->rootname,
                                                  maxamp, apcontour, &(run->apertureData), eptr);
                  if (flags & TEST_PARTICLES)
                    ((CSRCSBEND *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_CSRDRIFT:
                  nLeft = track_through_driftCSR(coord, nToTrack, (CSRDRIFT *)eptr->p_elem,
                                                 *P_central, accepted, last_z,
                                                 beamline->revolution_length, charge,
                                                 run->rootname);
                  break;
                case T_LSCDRIFT:
                  track_through_lscdrift(coord, nToTrack, (LSCDRIFT *)eptr->p_elem, *P_central, charge);
                  break;
                case T_SCMULT:
                  if (getSCMULTSpecCount() && !(flags & TEST_PARTICLES))
                    trackThroughSCMULT(coord, nToTrack, *P_central, i_pass, eptr, charge);
                  break;
                case T_EDRIFT:
                  exactDrift(coord, nToTrack, ((EDRIFT *)eptr->p_elem)->length);
                  break;
                case T_TUBEND:
                  nLeft = track_through_tubend(coord, nToTrack,
                                               (TUBEND *)eptr->p_elem, 0.0,
                                               *P_central, accepted, z);
                  break;
                case T_KQUAD:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((KQUAD *)eptr->p_elem)->isr;
                    ((KQUAD *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = multipole_tracking2(coord, nToTrack, eptr, 0.0,
                                              *P_central, accepted, last_z, maxamp, apcontour,
                                              &(run->apertureData), NULL, -1);
                  if (flags & TEST_PARTICLES)
                    ((KQUAD *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_KSEXT:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((KSEXT *)eptr->p_elem)->isr;
                    ((KSEXT *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = multipole_tracking2(coord, nToTrack, eptr, 0.0,
                                              *P_central, accepted, last_z, maxamp, apcontour,
                                              &(run->apertureData), NULL, -1);
                  if (flags & TEST_PARTICLES)
                    ((KSEXT *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_KOCT:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((KOCT *)eptr->p_elem)->isr;
                    ((KOCT *)eptr->p_elem)->isr = 0;
                  }
                  nLeft = multipole_tracking2(coord, nToTrack, eptr, 0.0,
                                              *P_central, accepted, last_z, maxamp, apcontour,
                                              &(run->apertureData), NULL, -1);
                  if (flags & TEST_PARTICLES)
                    ((KOCT *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_KQUSE:
                  if (((KQUSE *)eptr->p_elem)->matrixTracking) {
                    if (!eptr->matrix)
                      eptr->matrix = compute_matrix(eptr, run, NULL);
                    track_particles(coord, eptr->matrix, coord, nToTrack);
                  } else {
                    if (flags & TEST_PARTICLES) {
                      saveISR = ((KQUSE *)eptr->p_elem)->isr;
                      ((KQUSE *)eptr->p_elem)->isr = 0;
                    }
                    nLeft = multipole_tracking2(coord, nToTrack, eptr, 0.0,
                                                *P_central, accepted, last_z, maxamp, apcontour,
                                                &(run->apertureData), NULL, -1);
                    if (flags & TEST_PARTICLES)
                      ((KQUSE *)eptr->p_elem)->isr = saveISR;
                  }
                  break;
                case T_SAMPLE:
                  if (!(flags & TEST_PARTICLES))
                    nLeft = sample_particles(coord, (SAMPLE *)eptr->p_elem, nToTrack, accepted, z, *P_central);
                  break;
                case T_SCATTER:
                  if (!(flags & TEST_PARTICLES))
                    scatter_ele(coord, nToTrack, *P_central, (SCATTER *)eptr->p_elem, i_pass);
                  break;
                case T_DSCATTER:
                  if (!(flags & TEST_PARTICLES))
                    distributionScatter(coord, nToTrack, *P_central, (DSCATTER *)eptr->p_elem, i_pass);
                  break;
                case T_TSCATTER:
                  break;
                case T_NIBEND:
                  nLeft = lorentz(coord, nToTrack, (NIBEND *)eptr->p_elem, T_NIBEND, *P_central, accepted, NULL, NULL, NULL);
                  break;
                case T_NISEPT:
                  nLeft = lorentz(coord, nToTrack, (NISEPT *)eptr->p_elem, T_NISEPT, *P_central, accepted, NULL, NULL, NULL);
                  break;
                case T_BMAPXY:
                  nLeft = lorentz(coord, nToTrack, (BMAPXY *)eptr->p_elem, T_BMAPXY, *P_central, accepted, NULL, NULL, NULL);
                  break;
                case T_BMAPXYZ:
                  nLeft = lorentz(coord, nToTrack, (BMAPXYZ *)eptr->p_elem, T_BMAPXYZ, *P_central, accepted,
                                  maxamp, apcontour, &(run->apertureData));
                  break;
                case T_BRAT:
                  nLeft = trackBRAT(coord, nToTrack, (BRAT *)eptr->p_elem, *P_central, accepted);
                  /* printf("%ld particles left after BRAT %s\n", nLeft, eptr->name); */
                  break;
                case T_KPOLY:
                  nLeft = polynomial_kicks(coord, nToTrack, (KPOLY *)eptr->p_elem, 0.0,
                                           *P_central, accepted, z);
                  break;
                case T_HKPOLY:
                  nLeft = polynomial_hamiltonian(coord, nToTrack, (HKPOLY *)eptr->p_elem, 0.0, *P_central, accepted, z);
                  break;
                case T_RAMPRF:
                  ramped_rf_cavity(coord, nToTrack, (RAMPRF *)eptr->p_elem, *P_central, beamline->revolution_length,
                                   z_travel, i_pass);
                  break;
                case T_RAMPP:
                  ramp_momentum(coord, nToTrack, (RAMPP *)eptr->p_elem, P_central, i_pass);
                  break;
                case T_SOLE:
                  if (((SOLE *)eptr->p_elem)->B) {
                    SOLE *sptr;
                    double ks;
                    sptr = (SOLE *)eptr->p_elem;
                    if ((ks = -sptr->B / (*P_central * particleMass * c_mks / particleCharge)) != sptr->ks) {
                      sptr->ks = ks;
                      if (eptr->matrix) {
                        free_matrices(eptr->matrix);
                        free(eptr->matrix);
                        eptr->matrix = NULL;
                      }
                      if (!(eptr->matrix = compute_matrix(eptr, run, NULL)))
                        bombElegant("no matrix for element that must have matrix", NULL);
                    }
                    sptr->ks = 0; /* reset so it is clear that B is fundamental quantity */
                  }
                  if (!eptr->matrix) {
                    if (!(eptr->matrix = compute_matrix(eptr, run, NULL)))
                      bombElegant("no matrix for element that must have matrix", NULL);
                  }
                  track_particles(coord, eptr->matrix, coord, nToTrack);
                  break;
                case T_MATTER:
                  nLeft = track_through_matter(coord, nToTrack, i_pass, (MATTER *)eptr->p_elem, *P_central, accepted, z);
                  break;
                case T_RFMODE:
                  rfmode = (RFMODE *)eptr->p_elem;
                  if (!rfmode->initialized)
                    set_up_rfmode(rfmode, eptr->name, z, n_passes, run,
                                  nOriginal, *P_central,
                                  beamline->revolution_length);
                  if (!(flags & TEST_PARTICLES))
                    track_through_rfmode(coord, nToTrack, (RFMODE *)eptr->p_elem, *P_central,
                                         eptr->name, z_travel, i_pass, n_passes,
                                         charge);
                  break;
                case T_FRFMODE:
                  frfmode = (FRFMODE *)eptr->p_elem;
                  if (!frfmode->initialized)
                    set_up_frfmode(frfmode, eptr->name, z_travel, n_passes, run,
                                   nOriginal, *P_central,
                                   beamline->revolution_length);
                  if (!(flags & TEST_PARTICLES))
                    track_through_frfmode(coord, nToTrack, frfmode, *P_central,
                                          eptr->name, z, i_pass, n_passes,
                                          charge);
                  break;
                case T_TRFMODE:
                  trfmode = (TRFMODE *)eptr->p_elem;
                  if (!trfmode->initialized)
                    set_up_trfmode(trfmode, eptr->name, z_travel, n_passes, run, nOriginal);
                  if (!(flags & TEST_PARTICLES))
                    track_through_trfmode(coord, nToTrack, (TRFMODE *)eptr->p_elem, *P_central,
                                          eptr->name, z_travel, i_pass, n_passes,
                                          charge);
                  break;
                case T_FTRFMODE:
                  ftrfmode = (FTRFMODE *)eptr->p_elem;
                  if (!ftrfmode->initialized)
                    set_up_ftrfmode(ftrfmode, eptr->name, z, n_passes, run,
                                    nOriginal, *P_central,
                                    beamline->revolution_length);
                  if (!(flags & TEST_PARTICLES))
                    track_through_ftrfmode(coord, nToTrack, ftrfmode, *P_central,
                                           eptr->name, z, i_pass, n_passes,
                                           charge);
                  break;
                case T_ZLONGIT:
                  if (!(flags & TEST_PARTICLES))
                    track_through_zlongit(coord, nToTrack, (ZLONGIT *)eptr->p_elem, *P_central, run, i_pass,
                                          charge);
#ifdef MPI_DEBUG
                  printf("Returned from ZLONGIT\n");
                  fflush(stdout);
#endif
                  break;
                case T_ZTRANSVERSE:
                  if (!(flags & TEST_PARTICLES))
                    track_through_ztransverse(coord, nToTrack, (ZTRANSVERSE *)eptr->p_elem, *P_central, run, i_pass,
                                              charge);
                  break;
                case T_IONEFFECTS:
                  if (!(flags & TEST_PARTICLES))
                    trackWithIonEffects(coord, nToTrack, (IONEFFECTS *)eptr->p_elem, *P_central, i_pass, n_passes, charge);
                  nLeft = nToTrack;
                  break;
                case T_BEAMBEAM:
                  if (!(flags & TEST_PARTICLES))
                    applyBeamBeamKicks(coord, nToTrack, (BEAMBEAM *)eptr->p_elem, *P_central);
                  break;
                case T_CORGPIPE:
                  nLeft = elimit_amplitudes(coord, ((CORGPIPE *)eptr->p_elem)->radius, ((CORGPIPE *)eptr->p_elem)->radius,
                                            nToTrack, accepted, z - ((CORGPIPE *)eptr->p_elem)->length, *P_central, 0, 0, 2, 2,
                                            eptr);
                  track_through_corgpipe(coord, nLeft, (CORGPIPE *)eptr->p_elem, P_central, run, i_pass,
                                         charge);
                  nLeft = elimit_amplitudes(coord, ((CORGPIPE *)eptr->p_elem)->radius, ((CORGPIPE *)eptr->p_elem)->radius,
                                            nLeft, accepted, z, *P_central, 1, 0, 2, 2, eptr);
                  break;
                case T_CORGPLATES:
                  if (!(flags & TEST_PARTICLES))
                    track_through_corgplates(coord, nLeft, (CORGPLATES *)eptr->p_elem, P_central, run, i_pass, charge);
                  break;
                case T_LRWAKE:
                  if (!(flags & TEST_PARTICLES))
                    track_through_lrwake(coord, nToTrack, (LRWAKE *)eptr->p_elem, P_central, run, i_pass, charge);
                  break;
                case T_WAKE:
                  if (!(flags & TEST_PARTICLES))
                    track_through_wake(coord, nToTrack, (WAKE *)eptr->p_elem, P_central, run, i_pass, charge);
                  break;
                case T_TRWAKE:
                  if (!(flags & TEST_PARTICLES))
                    track_through_trwake(coord, nToTrack, (TRWAKE *)eptr->p_elem, *P_central, run, i_pass, charge);
                  break;
                case T_SREFFECTS:
                  track_SReffects(coord, nToTrack, (SREFFECTS *)eptr->p_elem, *P_central, eptr->twiss, &(beamline->radIntegrals),
                                  flags & TEST_PARTICLES);
                  break;
                case T_IBSCATTER:
#if USE_MPI && defined(MPI_DEBUG)
                  printf("Running IBSSCATTER\n");
#endif
                  if (!(flags & TEST_PARTICLES))
                    track_IBS(coord, nToTrack, eptr,
                              *P_central, beamline->elem, charge, i_pass, n_passes, run);
                  break;
                case T_SCRIPT:
#if !USE_MPI
                  if (nLeft < nMaximum && ((SCRIPT *)eptr->p_elem)->verbosity > 1)
                    printf("nLost=%ld, beam->n_particle=%ld, beam->n_to_track=%ld, nLeft=%ld, nToTrack=%ld, nMaximum=%ld\n",
                           nLost, beam ? beam->n_particle : -1, beam ? beam->n_to_track : -1, nLeft, nToTrack, nMaximum);
#endif
#if USE_MPI && MPI_DEBUG
                  printf("Preparing to call transformBeamWithScript, nToTrack=%ld\n", nToTrack);
                  fflush(stdout);
#endif
                  nLeft = transformBeamWithScript((SCRIPT *)eptr->p_elem, *P_central, charge,
                                                  beam, coord, nToTrack, run->rootname, i_pass, run->default_order, z, 0,
                                                  eptr->occurence, run->backtrack, beamline, run);
                  nLost = nToTrack - nLeft;
#if USE_MPI
                  nToTrack = nLeft;
                  nLost = 0;
#  if MPI_DEBUG
                  printf("Returned from script: nToTrack = %ld\n", nToTrack);
                  if (beam)
                    printf("beam->n_to_track_total = %ld\n", beam->n_to_track_total);
                  fflush(stdout);
#  endif
#endif
                  if (beam && coord != beam->particle) {
                    /* particles were created and so the particle array was changed */
                    coord = beam->particle;
                  }
                  if (beam && nMaximum < beam->n_to_track)
                    nMaximum = beam->n_to_track;
                  break;
                case T_FLOORELEMENT:
                  break;
                case T_TFBPICKUP:
                  if (!(flags & TEST_PARTICLES))
                    transverseFeedbackPickup((TFBPICKUP *)eptr->p_elem, coord, nToTrack, i_pass, *P_central, beam ? beam->id_slots_per_bunch : 0);
                  break;
                case T_CPICKUP:
                  if (!(flags & TEST_PARTICLES))
                    coolerPickup((CPICKUP *)eptr->p_elem, coord, nToTrack, i_pass, *P_central, beam ? beam->id_slots_per_bunch : 0);
                  break;
                case T_STRAY:
                  if (eptr->matrix) {
                    free_matrices(eptr->matrix);
                    free(eptr->matrix);
                    eptr->matrix = NULL;
                  }
                  stray = (STRAY *)eptr->p_elem;
                  eptr->matrix = stray_field_matrix(stray->length, &stray->lBx, &stray->gBx,
                                                    eptr->end_theta, stray->order ? stray->order : run->default_order,
                                                    *P_central,
                                                    stray->Wi);
                  track_particles(coord, eptr->matrix, coord, nToTrack);
                  break;
                case T_TFBDRIVER:
                  if (!(flags & TEST_PARTICLES))
                    transverseFeedbackDriver((TFBDRIVER *)eptr->p_elem, coord, nToTrack, beamline, i_pass, n_passes, run->rootname, *P_central, beam ? beam->id_slots_per_bunch : 0);
                  feedbackDriverSeen = 1;
#ifdef MPI_DEBUG
                  printf("Returned from TFBDRIVER\n");
                  fflush(stdout);
#endif
                  break;
                case T_CKICKER:
                  if (!(flags & TEST_PARTICLES))
                    coolerKicker((CKICKER *)eptr->p_elem, coord, nToTrack, beamline, i_pass, n_passes,
                                 run->rootname, *P_central, beam ? beam->id_slots_per_bunch : 0);
                  break;
                case T_LSRMDLTR:
                  nLeft = motion(coord, nToTrack, eptr->p_elem, eptr->type, P_central,
                                 &dgamma, dP, accepted, last_z);
                  /* show_dE = 1; */
                  break;
                case T_CWIGGLER:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((CWIGGLER *)eptr->p_elem)->isr;
                    ((CWIGGLER *)eptr->p_elem)->isr = 0;
                  }
                  GWigSymplecticPass(coord, nToTrack, *P_central, (CWIGGLER *)eptr->p_elem, NULL, 0, NULL);
                  if (flags & TEST_PARTICLES)
                    ((CWIGGLER *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_APPLE:
                  if (flags & TEST_PARTICLES) {
                    saveISR = ((APPLE *)eptr->p_elem)->isr;
                    ((APPLE *)eptr->p_elem)->isr = 0;
                  }
                  APPLE_Track(coord, nToTrack, *P_central, (APPLE *)eptr->p_elem);
                  if (flags & TEST_PARTICLES)
                    ((APPLE *)eptr->p_elem)->isr = saveISR;
                  break;
                case T_UKICKMAP:
                  nLeft = trackUndulatorKickMap(coord, accepted, nToTrack, *P_central, (UKICKMAP *)eptr->p_elem,
                                                last_z);
                  break;
                case T_KICKMAP:
                  nLeft = trackKickMap(coord, accepted, nToTrack, *P_central, (KICKMAP *)eptr->p_elem,
                                       last_z, NULL);
                  break;
                case T_TWISSELEMENT:
                  if (((TWISSELEMENT *)eptr->p_elem)->disable || flags & TEST_PARTICLES)
                    break;
                  if (((TWISSELEMENT *)eptr->p_elem)->applyOnce == 0 || i_pass == passOffset) {
                    /* If applying once, do so on the first pass through only */
                    if (((TWISSELEMENT *)eptr->p_elem)->fromBeam) {
                      /* Compute the transformation from the beam, rather than the lattice twiss parameters */
                      if (((TWISSELEMENT *)eptr->p_elem)->computeOnce == 0 ||
                          ((TWISSELEMENT *)eptr->p_elem)->transformComputed == 0) {
                        TWISS beamTwiss;
                        if (((TWISSELEMENT *)eptr->p_elem)->verbose)
                          printf("* Computing beam-based twiss transformation matrix for %s at z=%e m\n",
                                 eptr->name, z);
#if SDDS_MPI_IO
                        if (!partOnMaster && notSinglePart) {
                          if (isMaster)
                            nToTrack = 0;
                          if (beam)
                            MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                        } else { /* singlePart tracking or partOnMaster */
                          if (beam)
                            beam->n_to_track_total = nToTrack;
                        }
                        if (isMaster && (beam->n_to_track_total < 10)) {
#else
                          if (nToTrack < 10) {
#endif
                            printf("*** Error: too few particles (%ld) for computation of twiss parameters from beam\n", nToTrack);
                            exitElegant(1);
                          }
                          computeBeamTwissParameters(&beamTwiss, coord, nToTrack);
                          if (eptr->matrix) {
                            free_matrices(eptr->matrix);
                            free(eptr->matrix);
                            eptr->matrix = NULL;
                          }
                          eptr->matrix = twissTransformMatrix((TWISSELEMENT *)eptr->p_elem, &beamTwiss);
                          ((TWISSELEMENT *)eptr->p_elem)->transformComputed = 1;
                        }
                      }
                      if (((TWISSELEMENT *)eptr->p_elem)->transformComputed == 0) {
                        if (((TWISSELEMENT *)eptr->p_elem)->from0Values) {
                          if (eptr->matrix) {
                            free_matrices(eptr->matrix);
                            free(eptr->matrix);
                            eptr->matrix = NULL;
                          }
                          eptr->matrix = twissTransformMatrix1(&(((TWISSELEMENT *)eptr->p_elem)->twiss), &(((TWISSELEMENT *)eptr->p_elem)->twiss0));
                        } else {
                          printf("Error: The twiss parameter transformation matrix was not computed for element %s at z=%e m\n",
                                 eptr->name, z);
                          printf("This means you set FROM_BEAM=0 but didn't issue a twiss_output command.\n");
                          exitElegant(1);
                        }
                      }
                      if (eptr->matrix == NULL) {
                        printf("Error: twiss parameter transformation matrix was not computed for element %s at z=%e m\n",
                               eptr->name, z);
                        printf("and this wasn't properly detected.  Please send your input files to borland@aps.anl.gov.\n");
                        exitElegant(1);
                      }
                      if (((TWISSELEMENT *)eptr->p_elem)->verbose) {
                        TWISS beamTwiss;
                        printf("* Applying twiss parameter transformation matrix (%s at z=%e m) to beam.\n", eptr->name, z);
                        computeBeamTwissParameters(&beamTwiss, coord, nToTrack);
                        printf("  * Initial twiss parameters:\n");
                        printf("  betax = %le  alphax = %le  etax = %le, etaxp = %le\n",
                               beamTwiss.betax, beamTwiss.alphax, beamTwiss.etax, beamTwiss.etapx);
                        printf("  betay = %le  alphay = %le  etay = %le, etayp = %le\n",
                               beamTwiss.betay, beamTwiss.alphay, beamTwiss.etay, beamTwiss.etapy);
                        fflush(stdout);
                      }
                      track_particles(coord, eptr->matrix, coord, nToTrack);
                      if (((TWISSELEMENT *)eptr->p_elem)->verbose) {
                        TWISS beamTwiss;
                        computeBeamTwissParameters(&beamTwiss, coord, nToTrack);
                        printf("  * Final twiss parameters:\n");
                        printf("  betax = %le  alphax = %le  etax = %le, etaxp = %le\n",
                               beamTwiss.betax, beamTwiss.alphax, beamTwiss.etax, beamTwiss.etapx);
                        printf("  betay = %le  alphay = %le  etay = %le, etayp = %le\n",
                               beamTwiss.betay, beamTwiss.alphay, beamTwiss.etay, beamTwiss.etapy);
                        fflush(stdout);
                      }
                    }
                    break;
                  case T_EMITTANCE:
                    transformEmittances(coord, nToTrack, *P_central, (EMITTANCEELEMENT *)eptr->p_elem);
                    break;
                  case T_MRADINTEGRALS:
                    break;
                  case T_HCOR:
                  case T_VCOR:
                  case T_HVCOR:
                    if (!(entity_description[eptr->type].flags & HAS_MATRIX))
                      bombElegant("attempt to matrix-multiply for element with no matrix!", NULL);
                    if (!eptr->matrix) {
                      if (!(eptr->matrix = compute_matrix(eptr, run, NULL)))
                        bombElegant("no matrix for element that must have matrix", NULL);
                    }
                    /* Only the slave CPUs will track */
                    if ((!USE_MPI || !notSinglePart) || (USE_MPI && (myid != 0)))
                      track_particles(coord, eptr->matrix, coord, nToTrack);
                    switch (type) {
                    case T_HCOR:
                      if (((HCOR *)(eptr->p_elem))->synchRad)
                        addCorrectorRadiationKick(coord, nToTrack, eptr, type, *P_central, NULL,
                                                  flags & (CLOSED_ORBIT_TRACKING + TEST_PARTICLES));
                      break;
                    case T_VCOR:
                      if (((VCOR *)(eptr->p_elem))->synchRad)
                        addCorrectorRadiationKick(coord, nToTrack, eptr, type, *P_central, NULL,
                                                  flags & (CLOSED_ORBIT_TRACKING + TEST_PARTICLES));
                      break;
                    case T_HVCOR:
                      if (((HVCOR *)(eptr->p_elem))->synchRad)
                        addCorrectorRadiationKick(coord, nToTrack, eptr, type, *P_central, NULL,
                                                  flags & (CLOSED_ORBIT_TRACKING + TEST_PARTICLES));
                      break;
                    }
                    break;
                  case T_EHCOR:
                  case T_EVCOR:
                  case T_EHVCOR:
                    /* Only the slave CPUs will track */
                    if ((!USE_MPI || !notSinglePart) || (USE_MPI && (myid != 0)))
                      trackThroughExactCorrector(coord, nToTrack, eptr, *P_central, accepted, last_z, NULL);
                    break;
                    /* INSERT ENTRIES FOR NEW ELEMENTS ABOVE THIS LINE */
                  default:
                    printf("programming error: no tracking statements for element %s (type %s)\n",
                           eptr->name, entity_name[eptr->type]);
                    fflush(stdout);
                    exitElegant(1);
                    break;
                  }
                }
#ifdef USE_MPE
                MPE_Log_event(event2b, 0, bytebuf);
#endif
              }

#if USE_MPI
              if ((myid == 0) && notSinglePart && (!usefulOperation(eptr, flags, i_pass)))
                active = 0;
#endif
              if ((!USE_MPI || !notSinglePart) || (USE_MPI && active)) {
                if (!(flags & TEST_PARTICLES && !(flags & TEST_PARTICLE_LOSSES)) && !(classFlags & NO_APERTURE)) {
                  if (x_max || y_max) {
                    if (!elliptical)
                      nLeft = limit_amplitudes(coord, x_max, y_max, nLeft, accepted, z, *P_central,
                                               eptr->type == T_DRIF || eptr->type == T_STRAY,
                                               maxampOpenCode, eptr);
                    else
                      nLeft = elimit_amplitudes(coord, x_max, y_max, nLeft, accepted, z, *P_central,
                                                eptr->type == T_DRIF || eptr->type == T_STRAY,
                                                maxampOpenCode, maxampExponent, maxampYExponent, eptr);
                  }
                  if (run->apertureData.initialized)
                    nLeft = imposeApertureData(coord, nLeft, accepted, z, *P_central,
                                               &(run->apertureData), eptr);
                  if (apcontour && !(eptr->type == T_APCONTOUR && apcontour->holdOff))
                    nLeft = imposeApContour(coord, apcontour, nLeft, accepted, z, *P_central, eptr);
                }
              }
#ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 10: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#endif
              if (run->print_statistics > 0 && !(flags & TEST_PARTICLES)) {
                if (run->print_statistics > 1) {
                  report_stats(stdout, ": ");
                  printf("central momentum is %e    zstart = %em  zend = %em\n", *P_central, last_z, z);
                  fflush(stdout);
                }
                if (nLeft != nToTrack)
                  printf("%ld particles left\n", nLeft);
                fflush(stdout);
              }
            } else if (!(flags & TEST_PARTICLES)) {
              printf("element %s was ignored in tracking.\n",
                     eptr->name);
              fflush(stdout);
            }
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 11: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            if (flags & FIRST_BEAM_IS_FIDUCIAL) {
              if (!(flags & FIDUCIAL_BEAM_SEEN)) {
                short blockP0Change;
                /* Look at the change_p0 flag on the element for direction. Prevents, e.g., changing P_central after
                   RFCA elements that have change_p0=0
                */
                blockP0Change = determineP0ChangeBlocking(eptr);
                if (((run->always_change_p0 && !(flags & RESTRICT_FIDUCIALIZATION)) ||
                     ((entity_description[eptr->type].flags & MAY_CHANGE_ENERGY) && !blockP0Change)) &&
                    !(classFlags & (UNIDIAGNOSTIC & (~UNIPROCESSOR)))) {
                  do_match_energy(coord, nLeft, P_central, 0);
                }
                eptr->Pref_output_fiducial = *P_central;
              } else {
                if (*P_central != eptr->Pref_output_fiducial)
                  set_central_momentum(coord, nLeft, eptr->Pref_output_fiducial, P_central);
              }
            } else if (run->always_change_p0) {
              if (!(classFlags & (UNIDIAGNOSTIC & (~UNIPROCESSOR))))
                /* If it is a Diagnostic element, nothing needs to be done */
                do_match_energy(coord, nLeft, P_central, 0);
              eptr->Pref_output_fiducial = *P_central;
            } else if (!(flags & FIDUCIAL_BEAM_SEEN))
              eptr->Pref_output_fiducial = *P_central;
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 12: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            if (eptr->Pref_output_fiducial == 0)
              bombElegant("problem with fiducialization. Seek expert help!", NULL);
            if (i_pass == passOffset && traj_vs_z) {
              /* collect trajectory data--used mostly by trajectory correction routines */
              /* this is always false
                 if (!traj_vs_z[i_traj].centroid) {
                 printf("error: the trajectory centroid array for %s is NULL (do_tracking)",
                 eptr->name);
                 fflush(stdout);
                 exitElegant(1);
                 }
              */
              traj_vs_z[i_traj].elem = eptr;
              if (!(traj_vs_z[i_traj].n_part = nLeft)) {
                for (i = 0; i < 6; i++)
                  traj_vs_z[i_traj].centroid[i] = 0;
              } else {
#ifdef HAVE_GPU
                if (getElementOnGpu()) {
                  gpu_collect_trajectory_data(traj_vs_z[i_traj].centroid, nLeft);
                } else {
#endif
                  for (i = 0; i < 6; i++) {
                    for (j = sum = 0; j < nToTrack; j++)
                      sum += coord[j][i];
                    traj_vs_z[i_traj].centroid[i] = sum / nLeft;
                  }
#ifdef HAVE_GPU
                }
#endif
              }
              i_traj++;
            }
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 13: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            if (!(flags & TEST_PARTICLES) && sliceAnalysis && sliceAnalysis->active && !sliceAnalysis->finalValuesOnly) {
#if USE_MPI
              if (!(classFlags & UNIPROCESSOR)) { /* This function will be parallelized in the future */
                printf("performSliceAnalysisOutput is not supported in parallel mode currently.\n");
                MPI_Barrier(MPI_COMM_WORLD); /* Make sure the information can be printed before aborting */
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
#endif
#ifdef HAVE_GPU
              coord = forceParticlesToCpu("performSliceAnalysisOutput");
#endif
              performSliceAnalysisOutput(sliceAnalysis, coord, nToTrack,
                                         !sliceAnDone, step,
                                         *P_central,
                                         charge ? charge->macroParticleCharge * nToTrack : 0.0,
                                         eptr->name, z, 0);
              sliceAnDone = 1;
            }
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 14: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
#if USE_MPI
            if (notSinglePart) {
              if (!(classFlags & (UNIPROCESSOR | MPALGORITHM))) {
                end_wtime = MPI_Wtime();
                my_wtime = my_wtime + end_wtime - start_wtime;
              } else if (!(classFlags & ((UNIDIAGNOSTIC & (~UNIPROCESSOR)) | MPALGORITHM))) {
                /* a non-diagnostic uniprocessor element */
                if ((myid == 0) && (nMaximum != (nLeft + nLost)))
                  /* there are additional losses occurred */
                  lostSinceSeqMode = needSort = 1;
#  ifdef DEBUG_CRASH
                printMessageAndTime(stdout, "do_tracking checkpoint 14.1: ");
                printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
                printf("nMaximum = %ld nLeft = %ld nLost = %ld\n",
                       nMaximum, nLeft, nLost);
                fflush(stdout);
#  endif
                MPI_Bcast(&lostSinceSeqMode, 1, MPI_INT, 0, MPI_COMM_WORLD);
#  ifdef DEBUG_CRASH
                printMessageAndTime(stdout, "do_tracking checkpoint 14.2: ");
                printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
                fflush(stdout);
#  endif
              }
              if (classFlags & MPALGORITHM && isMaster) {
                /* Master does not need to do limit_amplitudes for MPALGORITHM elements */
                active = 0;
              }
            }
#endif
#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 15: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
            fflush(stdout);
#endif
            if ((!USE_MPI || !notSinglePart) || (USE_MPI && active)) {
              nLeft = limit_amplitudes(coord, DBL_MAX, DBL_MAX, nLeft, accepted, z, *P_central, 0, 0, eptr);
#ifndef HAVE_GPU /* Obstructions are not implemented in GPU code */
              nLeft = filterParticlesWithObstructions(coord, nLeft, accepted, z, *P_central);
#endif
              if (eptr->type != T_SCRIPT) { /* For the SCRIPT element, the lost particle coordinate will be recorded inside the element */
                if (nLeft != nToTrack)
                  recordLostParticles(beam, coord, nLeft, nToTrack, i_pass);
              }
            }

            if (getSCMULTSpecCount() && (entity_description[eptr->type].flags & HAS_LENGTH) && 
                !(flags & TEST_PARTICLES)) {
              /* calcaulate beam size at exit of element for use in space  charge calculation with SCMULT */
              /* need special care for element with 0 length but phase space rotation */
              if (((DRIFT *)eptr->p_elem)->length > 0.0) {
#ifdef HAVE_GPU
                coord = forceParticlesToCpu("accumulateSCMULT");
#endif
                accumulateSCMULT(coord, nToTrack, *P_central, eptr, beam ? beam->id_slots_per_bunch: 0);
              }
            }

#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 16: ");
            printf("element %s#%ld, %ld particles, %ld left, runInSinglePartMode=%ld\n", eptr->name, eptr->occurence, nToTrack, nLeft,
                   runInSinglePartMode);
            fflush(stdout);
#endif

#if USE_MPI
            if (flags & ALLOW_MPI_ABORT_TRACKING && !runInSinglePartMode) {
#  ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 16.1, mpiAbort=");
              printf("%d: ", mpiAbort);
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#  endif
              /* When performing regular parallel tracking, certain elements with MPALGORITHM=0 may need to abort, but master has no way
                 to know because it doesn't run the procedure. Here we check for setting of the mpiAbort variable on any processor */
              MPI_Barrier(MPI_COMM_WORLD);
#  ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 16.2: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#  endif
              MPI_Allreduce(&mpiAbort, &mpiAbortGlobal, 1, MPI_SHORT, MPI_MAX, MPI_COMM_WORLD);
#  ifdef DEBUG_CRASH
              printMessageAndTime(stdout, "do_tracking checkpoint 16.3: ");
              printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
              fflush(stdout);
#  endif
              if (mpiAbortGlobal) {
#  ifdef DEBUG_CRASH
                printMessageAndTime(stdout, "do_tracking checkpoint 16.4: ");
                printf("element %s#%ld, %ld particles, %ld left\n", eptr->name, eptr->occurence, nToTrack, nLeft);
                fflush(stdout);
#  endif
                if (mpiAbortGlobal < N_MPI_ABORT_TYPES)
                  printf("Run aborted by error in %s: %s\n", eptr->name, mpiAbortDescription[mpiAbortGlobal]);
                else
                  printf("Run aborted by error in %s: unknown code %ld\n", eptr->name, (long)mpiAbortGlobal);
                exitElegant(1);
              }
            }
#endif

#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 16.5: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
            fflush(stdout);
#endif
            last_type = eptr->type;
            eptrPred = eptr;
            eptr = eptr->succ;
            i_elem++;
            nToTrack = nLeft;
#if USE_MPI
            if (!partOnMaster && notSinglePart) {
              /* We have to collect information from all the processors to print correct info during tracking */
              long nToTrackC;
              nToTrackC = nToTrack;
              if (isMaster)
                nToTrackC = 0;
              if (beam)
                MPI_Reduce(&nToTrackC, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
              sprintf(s, "%ld particles present after pass %ld        ",
                      beam ? beam->n_to_track_total : -1, i_pass);
            } else {
              if (beam)
                beam->n_to_track_total = nToTrack;
            }
#endif

#ifdef DEBUG_CRASH
            printMessageAndTime(stdout, "do_tracking checkpoint 16.6: ");
            printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
            fflush(stdout);
#endif

            if (run->showElementTiming && last_type >= 0 && last_type <= N_TYPES) {
              timeCounter[last_type] += delapsed_time() - tStart;
              runCounter[last_type] += 1;
            }
            if (run->monitorMemoryUsage) {
              if ((memoryAfter = memoryUsage()) > memoryBefore) {
                printf("Memory usage increased by %ld kB in %s %s#%ld, pass %ld\n",
                       memoryAfter - memoryBefore, entity_name[last_type], eptrPred->name, eptrPred->occurence, i_pass);
                fflush(stdout);
              }
            }
          } /* end of the while loop */
#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 17: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
          fflush(stdout);
#endif
          if (!(flags & TEST_PARTICLES) && sliceAnalysis && sliceAnalysis->active && !sliceAnalysis->finalValuesOnly) {
#if USE_MPI
            if (notSinglePart) {
              if (!(classFlags & UNIPROCESSOR)) { /* This function will be parallelized in the future */
                printf("performSliceAnalysisOutput is not supported in parallel mode currently.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            }
#endif
#ifdef HAVE_GPU
            coord = forceParticlesToCpu("performSliceAnalysisOutput");
#endif
            performSliceAnalysisOutput(sliceAnalysis, coord, nToTrack,
                                       !sliceAnDone, step,
                                       *P_central,
                                       charge ? charge->macroParticleCharge * nToTrack : 0.0,
                                       eptrPred->name, eptrPred->end_pos, 0);

            sliceAnDone = 1;
          }

          if (effort) {
#if !SDDS_MPI_IO
            *effort += nLeft;
#else
            if ((isMaster && (partOnMaster || !notSinglePart)) || (isSlave && !partOnMaster))
              *effort += nLeft;
#endif
          }

#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 18: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
          fflush(stdout);
#endif

          log_exit("do_tracking.2.2");
#ifdef WATCH_MEMORY
          printf("main tracking loop done: CPU: %6.2lf  PF: %6ld  MEM: %6ld\n",
                 cpu_time() / 100.0, page_faults(), memoryUsage());
          fflush(stdout);
#endif
          if ((!USE_MPI || !notSinglePart) && (i_pass == passOffset || watch_pt_seen || feedbackDriverSeen)) {
            /* if eptr is not NULL, then all particles have been lost */
            /* some work still has to be done, however. */
            while (eptr) {
              if (sums_vs_z && *sums_vs_z && !(flags & FINAL_SUMS_ONLY) && !(flags & TEST_PARTICLES)) {
                if (i_sums < 0)
                  bombElegant("attempt to accumulate beam sums with negative index!", NULL);
#ifdef DEBUG_BEAM_SUMS
                printMessageAndTime(stdout, "Accumulating beam sums\n");
#endif
                accumulate_beam_sums(*sums_vs_z + i_sums, coord, nToTrack, *P_central,
                                     charge ? charge->macroParticleCharge : 0.0,
                                     NULL, 0.0, 0.0, -1, -1, 0);
#ifdef DEBUG_BEAM_SUMS
                printMessageAndTime(stdout, "Done accumulating beam sums\n");
#endif
                (*sums_vs_z)[i_sums].z = z;
                (*sums_vs_z)[i_sums].pass = i_pass;
                i_sums++;
              }
              if (entity_description[eptr->type].flags & HAS_LENGTH && eptr->p_elem)
                z += ((DRIFT *)eptr->p_elem)->length;
              else
                z += eptr->end_pos - eptr->beg_pos;
              switch (eptr->type) {
              case T_TFBDRIVER:
                flushTransverseFeedbackDriverFiles((TFBDRIVER *)(eptr->p_elem));
                break;
              case T_WATCH:
#if USE_MPI
                if (!notSinglePart) /* When each processor tracks the beam independently, the watch point will be disabled in Pelegant */
                  break;
#endif
                if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
                  watch = (WATCH *)eptr->p_elem;
                  if (!watch->initialized)
                    set_up_watch_point(watch, run, eptr->occurence, eptr->pred ? eptr->pred->name : NULL,
                                       eptr->pred ? eptr->pred->occurence : 0, i_pass, beamline->elem, beam ? beam->id_slots_per_bunch : 0);
                  if (!watch->disable) {
                    if (i_pass % watch->interval == 0) {
                      switch (watch->mode_code) {
                      case WATCH_COORDINATES:
                        break;
                      case WATCH_PARAMETERS:
                      case WATCH_CENTROIDS:
#ifdef HAVE_GPU
                        coord = forceParticlesToCpu("dump_watch_parameters");
#endif
                        dump_watch_parameters(watch, step, i_pass, n_passes, coord, nToTrack,
#if SDDS_MPI_IO
                                              total_nOriginal,
#else
                                              nOriginal,
#endif
                                              *P_central,
                                              beamline->revolution_length, z,
                                              charge ? charge->macroParticleCharge : 0.0);
                        break;
                      case WATCH_FFT:
#ifdef HAVE_GPU
                        coord = forceParticlesToCpu("dump_watch_FFT");
#endif
                        dump_watch_FFT(watch, step, i_pass, n_passes, coord, nToTrack, nOriginal, *P_central);
                        break;
                      }
                    }
                  }
                }
                break;
              default:
                break;
              }
              if (i_pass == passOffset && traj_vs_z) {
                /* collect trajectory data--used mostly by trajectory correction routines */
                /* This is always false
                   if (!traj_vs_z[i_traj].centroid) {
                   printf("error: the trajectory centroid array for %s is NULL (do_tracking)",
                   eptr->name);
                   fflush(stdout);
                   exitElegant(1);
                   }
                */
                traj_vs_z[i_traj].elem = eptr;
                traj_vs_z[i_traj].n_part = 0;
                for (i = 0; i < 6; i++)
                  traj_vs_z[i_traj].centroid[i] = 0;
                i_traj++;
              }
              eptr = eptr->succ;
            }
          }

#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 19: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
          fflush(stdout);
#endif

          if (sums_vs_z && (*sums_vs_z) && !(flags & FINAL_SUMS_ONLY) && !(flags & TEST_PARTICLES) &&
              (run->wrap_around || i_pass == n_passes - 1)) {
            if (i_sums < 0)
              bombElegant("attempt to accumulate beam sums with negative index!", NULL);
#ifdef DEBUG_BEAM_SUMS
            printMessageAndTime(stdout, "Accumulating beam sums\n");
#endif
            accumulate_beam_sums(*sums_vs_z + i_sums, coord, nToTrack, *P_central,
                                 charge ? charge->macroParticleCharge : 0.0,
                                 NULL, 0.0, 0.0, -1, -1, 0);
            (*sums_vs_z)[i_sums].z = z;
            (*sums_vs_z)[i_sums].pass = i_pass;
#if defined(BEAM_SUMS_DEBUG)
            printMessageAndTime(stdout, "Done accumulating beam sums\n");
            printf("beam sums accumulated in slot %ld for %s at z=%em, sx=%e\n",
                   i_sums, name, z, sqrt((*sums_vs_z)[i_sums].sum2[0] / nLeft));
            fflush(stdout);
#endif
            i_sums++;
          }

#ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 20: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
          fflush(stdout);
#endif

#if USE_MPI
          if (notSinglePart) {
            if (run->load_balancing_on == 1) { /* User can choose if load balancing needs to be done */
              if (balanceStatus == startMode) {
                balanceStatus = checkBalance(my_wtime, myid, n_processors, 1);
                /* calculate the rate for all of the slave processors */
                if (myid == 0) {
                  my_rate = 0.0;
                  nParPerElements = 0.0;
                } else {
                  nParPerElements = (double)nParElements / (double)nElements;
                  if (my_wtime != 0.0)
                    my_rate = nParPerElements / my_wtime;
                  else
                    my_rate = 1.;
                }
                /*  lostSinceSeqMode = 1; */ /* set flag to distribute jobs according to  the speed.
                    The default redistribution for the first turn is disabled
                    as it might cause some problems for random number generator */
              } else {                       /* The workload balancing will be checked for every pass by default.
                                                If user defined the CHECKFLAGS, the balance will be checked only
                                                when the nToTrack is changed. */
#  ifdef CHECKFLAGS
                if (myid == 0) {
                  if (old_nToTrack != nToTrack) {
                    checkFlags = 1;
                  } else
                    checkFlags = 0;
                }
                MPI_Bcast(&checkFlags, 1, MPI_INT, 0, MPI_COMM_WORLD);
#  else
                checkFlags = 1; /* the default option */
#  endif
                if (checkFlags)
                  balanceStatus = checkBalance(my_wtime, myid, n_processors, 1);
                if (balanceStatus == badBalance) {
                  if (myid == 0) {
                    my_rate = 0.0;
                    nParPerElements = 0.0;
                  } else {
                    nParPerElements = (double)nParElements / (double)nElements;
                    if (my_wtime != 0.0)
                      my_rate = nParPerElements / my_wtime;
                    else
                      /* set the speed to be euqal for the special case where all the elements are UNIPROCESSOR or MPALGORITHM */
                      my_rate = 1.;
                  }
                }
#  ifdef MPI_DEBUG
                printf("\n\nmyid=%d, nParPerElements=%e, my_time=%lf, my_rate=%lf\n",
                       myid, nParPerElements, my_wtime, nParPerElements / my_wtime);
                printf("nParElements=%ld, nElements=%ld\n", nParElements, nElements);
#  endif
              }
            } else {
              balanceStatus = goodBalance;
              if (run->load_balancing_on == -1)
                checkBalance(my_wtime, myid, n_processors, 1); /* Just to check and report, nothing done */
            }
#  ifdef CHECKFLAGS
            if (myid == 0)
              old_nToTrack = nToTrack;
#  endif
          }
#  ifdef DEBUG_CRASH
          printMessageAndTime(stdout, "do_tracking checkpoint 21: ");
          printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
          fflush(stdout);
#  endif
#endif
          if (outputFiles->particleTunes.initialized) {
#if USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            accumulateParticleTuneData(coord, nLeft, i_pass, &(outputFiles->particleTunes));
          }
        } /* end of the for loop for n_passes*/

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 22: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

        if (outputFiles->particleTunes.initialized)
          outputParticleTunes(&(outputFiles->particleTunes), i_pass - 1);

#ifdef SORT /* Sort the particles when the particles are lost at the very last element */
        if (!USE_MPI || needSort)
          if (nToTrackAtLastSort > nToTrack) { /* indicates more particles are lost, need sort */
            if (beam && beam->bunchFrequency != 0)
              printWarning("particle ID sort not being performed because bunch frequency is nonzero");
            else {
#  ifdef HAVE_GPU
              if (getElementOnGpu())
                sortByPID(nToTrack);
              else {
#  endif
                qsort(coord[0], nToTrack, totalPropertiesPerParticle * sizeof(double), comp_IDs);
                if (accepted != NULL)
                  qsort(accepted[0], nToTrack, totalPropertiesPerParticle * sizeof(double), comp_IDs);
#  ifdef HAVE_GPU
              }
#  endif
              nToTrackAtLastSort = nToTrack;
            }
          }
#endif

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 23: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

#if USE_MPI
#  if !SDDS_MPI_IO
        if (notSinglePart)
          /* change back to sequential mode before leaving the do_tracking function */
          if (parallelStatus == trueParallel && notSinglePart) {
            gatherParticles(&coord, &nToTrack, &nLost, &accepted, n_processors, myid, &round);
            MPI_Bcast(&nToTrack, 1, MPI_LONG, 0, MPI_COMM_WORLD);
            parallelStatus = notParallel;
            partOnMaster = 1;
          }
#  else
        /* Make sure that the particles are distributed to the slave processors for parallel IO */
        if (partOnMaster && notSinglePart) {
#    if USE_MPI && MPI_DEBUG
          printf("Scattering particles (4): nToTrack = %ld\n", nToTrack);
#    endif
          scatterParticles(coord, &nToTrack, accepted, n_processors, myid,
                           balanceStatus, my_rate, nParPerElements, round, lostSinceSeqMode, &distributed, &reAllocate, P_central);
#    if USE_MPI && MPI_DEBUG
          printf("Scattering particles done: nToTrack = %ld\n", nToTrack);
#    endif
          nLeft = nToTrack;
          parallelStatus = trueParallel;
        }
#  endif

#endif

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 24: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

        /* do this here to get report of CSR drift normalization */
        reset_driftCSR();

        log_exit("do_tracking.2");
        log_entry("do_tracking.3");

        if (((!USE_MPI && nLeft) || USE_MPI) && sums_vs_z && *sums_vs_z && !(flags & TEST_PARTICLES)) {
          if (flags & FINAL_SUMS_ONLY) {
            log_entry("do_tracking.3.1");
            i_sums = 0;
#ifdef DEBUG_BEAM_SUMS
            printMessageAndTime(stdout, "Accumulating beam sums\n");
#endif
            accumulate_beam_sums(*sums_vs_z + i_sums, coord, nToTrack, *P_central,
                                 charge ? charge->macroParticleCharge : 0.0,
                                 NULL, 0.0, 0.0, -1, -1, 0);
            (*sums_vs_z)[i_sums].z = z;
            (*sums_vs_z)[i_sums].pass = i_pass;
#if defined(BEAM_SUMS_DEBUG)
            printMessageAndTime(stdout, "Done accumulating beam sums\n");
            printf("beam sums accumulated in slot %ld for final sums at z=%em, sx=%e\n",
                   i_sums, z, sqrt((*sums_vs_z)[i_sums].sum2[0] / nLeft));
            fflush(stdout);
#endif
            log_exit("do_tracking.3.1");
          } else if (run->wrap_around) {
            log_entry("do_tracking.3.2");
            if (i_sums < 0)
              bombElegant("attempt to accumulate beam sums with negative index!", NULL);
            /* accumulate sums for final output */
#ifdef DEBUG_BEAM_SUMS
            printMessageAndTime(stdout, "Accumulating beam sums\n");
#endif
            accumulate_beam_sums(*sums_vs_z + i_sums, coord, nToTrack, *P_central,
                                 charge ? charge->macroParticleCharge : 0.0,
                                 NULL, 0.0, 0.0, -1, -1, 0);
            (*sums_vs_z)[i_sums].pass = i_pass;
#if defined(BEAM_SUMS_DEBUG)
            printMessageAndTime(stdout, "Done accumulating beam sums\n");
            printf("beam sums accumulated in slot %ld for final sums at z=%em, sx=%e\n",
                   i_sums, z, sqrt((*sums_vs_z)[i_sums].sum2[0] / nLeft));
            fflush(stdout);
#endif
            log_exit("do_tracking.3.2");
          } else {
            log_entry("do_tracking.3.3");
            if (i_sums < 0)
              bombElegant("attempt to accumulate beam sums with negative index!", NULL);
            copy_beam_sums(*sums_vs_z + i_sums, *sums_vs_z + i_sums - 1);
            (*sums_vs_z)[i_sums].pass = i_pass;
#if defined(BEAM_SUMS_DEBUG)
            printf("beam sums copied to slot %ld from slot %ld for final sums at z=%em, sx=%e\n",
                   i_sums, i_sums - 1, z, (*sums_vs_z)[i_sums].sum2[0]);
            fflush(stdout);
#endif
            log_exit("do_tracking.3.3");
          }
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 25: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

#ifdef HAVE_GPU
#  ifdef GPU_VERIFY
        displayTimings();
#  endif
        if (!coord)
          coord = getGpuBase()->coord;
        gpuBaseDealloc();
#endif

        if (sasefel && sasefel->active) {
          if (!charge) {
            printf("Can't compute SASE FEL---no CHARGE element seen");
            fflush(stdout);
            exitElegant(1);
          }
#if SDDS_MPI_IO
          if (!partOnMaster && notSinglePart) {
            if (isMaster)
              nToTrack = 0;
            if (beam)
              MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
          } else { /* singlePart tracking or partOnMaster */
            if (beam)
              beam->n_to_track_total = nToTrack;
          }
          /* The charge is correct on master only, but it should not affect the output */
          computeSASEFELAtEnd(sasefel, coord, nToTrack, *P_central, charge->macroParticleCharge * beam->n_to_track_total);
#else
          computeSASEFELAtEnd(sasefel, coord, nToTrack, *P_central, charge->macroParticleCharge * nToTrack);
#endif
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 26: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

        log_exit("do_tracking.3");
        log_entry("do_tracking.4");
        if (!(flags & SILENT_RUNNING) && !is_batch && n_passes != 1 && !(flags & TEST_PARTICLES)) {
#if !SDDS_MPI_IO
          printf("%ld particles present after pass %ld        \n",
                 nToTrack, i_pass);
#else
          if (!partOnMaster && notSinglePart) {
            /* We have to collect information from all the processors to print correct info during tracking */
            if (isMaster)
              nToTrack = 0;
            if (beam)
              MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            printf("%ld particles present after pass %ld        \n",
                   beam ? beam->n_to_track_total : -1, i_pass);
          } else {
            if (beam)
              beam->n_to_track_total = nToTrack;
            printf("%ld particles present after pass %ld        \n",
                   nToTrack, i_pass);
          }
#endif
          fflush(stdout);
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 27: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

#ifdef MPI_DEBUG
#  ifdef CHECKFLAGS
        printf("Balance is checked for the first pass and when particles are lost only.\n");
        fflush(stdout);
#  else
        if (run->load_balancing_on == 1) {
          printf("Balance is checked for every pass.\n");
          fflush(stdout);
        }
#  endif
#endif

        log_exit("do_tracking.4");

        log_exit("do_tracking");

        if (charge && finalCharge) {
#if !SDDS_MPI_IO
          *finalCharge = nToTrack * charge->macroParticleCharge;
#else
          if (!partOnMaster) {
            /* We have to collect information from all the processors to print correct info after tracking */
            if (isMaster)
              nToTrack = 0;
            if (beam)
              MPI_Reduce(&nToTrack, &(beam->n_to_track_total), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
          } else {
            if (beam)
              beam->n_to_track_total = nToTrack;
          }
          /* Only Master will have the correct information */
          *finalCharge = beam->n_to_track_total * charge->macroParticleCharge;
#endif
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 28: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
#endif

        if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
          eptr = beamline->elem;
          while (eptr) {
            if (eptr->type == T_WATCH) {
              watch = (WATCH *)eptr->p_elem;
              if (watch->initialized) {
                SDDS_UpdatePage(watch->SDDS_table, 0);
              }
            }
            eptr = eptr->succ;
          }
        }

#ifdef DEBUG_CRASH
        printMessageAndTime(stdout, "do_tracking checkpoint 29: ");
        printf("element %s#%ld, %ld particles, %ld left\n", eptr ? eptr->name : "?", eptr ? eptr->occurence : -1, nToTrack, nLeft);
        fflush(stdout);
        printf("returning with nToTrack = %ld\n", nToTrack);
        fflush(stdout);
#endif

        setObstructionsMode(1);

        if (beam)
          beam->n_accepted = nToTrack;

        return (nToTrack);
      }

      void offset_beam(
                       double **coord,
                       long nToTrack,
                       MALIGN *offset,
                       double P_central) {
        long i_part, allParticles;
        double *part, pc, beta, gamma, t;
        double ds;

#ifdef HAVE_GPU
        if (getElementOnGpu()) {
          startGpuTimer();
          gpu_offset_beam(nToTrack, offset, P_central);
#  ifdef GPU_VERIFY
          startCpuTimer();
          offset_beam(coord, nToTrack, offset, P_central);
          compareGpuCpu(nToTrack, "offset_beam");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */

        log_entry("offset_beam");

        if (offset->startPID >= 0 && offset->startPID > offset->endPID)
          bombElegantVA("Error: startPID (%ld) greater than endPID (%ld) for MALIGN element (offset_beam)\n", offset->startPID, offset->endPID);
        if ((offset->endPID >= 0 && offset->startPID < 0) || (offset->startPID >= 0 && offset->endPID < 0))
          bombElegantVA("Error: Invalid startPID (%ld) and endPID (%ld) in MALIGN element (offset_beam)\n", offset->startPID, offset->endPID);

        allParticles = (offset->startPID == -1) && (offset->endPID == -1);

        for (i_part = nToTrack - 1; i_part >= 0; i_part--) {
          part = coord[i_part];
          if (!allParticles && (part[6] < offset->startPID || part[6] > offset->endPID))
            continue;
          if (offset->dz)
            ds = offset->dz * sqrt(1 + sqr(part[1]) + sqr(part[3]));
          else
            ds = 0;
          part[0] += offset->dx + offset->dz * part[1];
          part[1] += offset->dxp;
          part[2] += offset->dy + offset->dz * part[3];
          part[3] += offset->dyp;
          part[4] += ds;
          if (offset->dt || offset->dp || offset->de) {
            pc = P_central * (1 + part[5]);
            beta = pc / (gamma = sqrt(1 + pc * pc));
            t = part[4] / (beta * c_mks) + offset->dt;
            if (offset->dp) {
              part[5] += offset->dp;
              pc = P_central * (1 + part[5]);
              beta = pc / sqrt(1 + pc * pc);
            }
            if (offset->de) {
              gamma += offset->de * gamma;
              pc = sqrt(gamma * gamma - 1);
              beta = pc / gamma;
              part[5] = (pc - P_central) / P_central;
            }
            part[4] = t * beta * c_mks;
          }
        }
        log_exit("offset_beam");
      }

      void do_match_energy(
                           double **coord,
                           long np,
                           double *P_central,
                           long change_beam) {
        long ip;
        double P_average, dP_centroid, P, t, dp, dr;
        long active = 1;
#ifdef USE_KAHAN
        double error = 0.0;
#endif
#if USE_MPI
        long np_total;
        double P_total = 0.0;
        if (notSinglePart) {
          if (((parallelStatus == trueParallel) && isSlave) || ((parallelStatus != trueParallel) && isMaster))
            active = 1;
          else
            active = 0;
        }
#endif

#ifdef HAVE_GPU
        if (getElementOnGpu()) {
#  ifdef GPU_VERIFY
          double P_central_init = *P_central;
#  endif /* GPU_VERIFY */
          startGpuTimer();
          gpu_do_match_energy(np, P_central, change_beam);
#  ifdef GPU_VERIFY
          startCpuTimer();
          do_match_energy(coord, np, &P_central_init, change_beam);
          compareGpuCpu(np, "do_match_energy");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */

        log_entry("do_match_energy");

#if (!USE_MPI)
        if (!np) {
          log_exit("do_match_energy");
          return;
        }
#else
        if (notSinglePart) {
          if (parallelStatus != trueParallel) {
            if (!np) {
              log_exit("do_match_energy");
              return;
            }
          } else {
            if (isMaster)
              np = 0; /* All the particles have been distributed to the slave processors */
            MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          }
        } else if (!np) {
          log_exit("do_match_energy");
          return;
        }
#endif

        if (!change_beam) {
          /* change the central momentum so that it matches the beam's centroid */
          P_average = 0;
          if (active) {
            for (ip = 0; ip < np; ip++) {
#ifndef USE_KAHAN
              P_average += (*P_central) * (1 + coord[ip][5]);
#else
              P_average = KahanPlus(P_average, (*P_central) * (1 + coord[ip][5]), &error);
#endif
            }
          }
#if (!USE_MPI)
          P_average /= np;
#else
          if (notSinglePart) {
            if (parallelStatus != trueParallel) {
              if (isMaster)
                P_average /= np;
            } else {
#  ifndef USE_KAHAN
              MPI_Allreduce(&P_average, &P_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
              P_total = KahanParallel(P_average, error, MPI_COMM_WORLD);

#  endif
              P_average = P_total / np_total;
            }
          } else /* Single particle case, all the processors will do the same as in serial version */
            P_average /= np;
#endif
#ifdef DEBUG_FIDUCIALIZATION
          printf("Changing reference momentum from %e to %e in %s at %e to match beam\n",
                 *P_central, P_average, trackingContext.elementName, trackingContext.zEnd);
#endif
          if (fabs(P_average - (*P_central)) / (*P_central) > 1e-14) {
            /* if (P_average!= *P_central) { */
            /* new dp/dr formula for update coord[5] improves accuracy */
            dp = (*P_central - P_average) / P_average;
            dr = (*P_central) / P_average;
            if (active) {
              for (ip = 0; ip < np; ip++)
                /*coord[ip][5] = ((1+coord[ip][5])*(*P_central) - P_average)/ P_average;*/
                coord[ip][5] = dp + coord[ip][5] * dr;
            }
            *P_central = P_average;
          }
        } else {
          /* change the particle momenta so that the centroid is the central momentum */
          /* the path length is adjusted so that the time-of-flight at the current
             velocity is fixed */
          P_average = 0;
          if (active) {
            for (ip = 0; ip < np; ip++) {
#ifndef USE_KAHAN
              P_average += (*P_central * (1 + coord[ip][5]));
#else
              P_average = KahanPlus(P_average, (*P_central * (1 + coord[ip][5])), &error);
#endif
            }
          }
#if (!USE_MPI)
          P_average /= np;
#else
          if (notSinglePart) {
            if (parallelStatus != trueParallel) {
              if (isMaster)
                P_average /= np;
            } else {
#  ifndef USE_KAHAN
              MPI_Allreduce(&P_average, &P_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
              P_total = KahanParallel(P_average, error, MPI_COMM_WORLD);
#  endif
              P_average = P_total / np_total;
            }
          } else
            P_average /= np;
#endif
          if (active) {
            dP_centroid = *P_central - P_average;
            for (ip = 0; ip < np; ip++) {
              P = (1 + coord[ip][5]) * (*P_central);
              t = coord[ip][4] / (P / sqrt(P * P + 1));
              P += dP_centroid;
              coord[ip][5] = (P - *P_central) / (*P_central);
              coord[ip][4] = t * (P / sqrt(P * P + 1));
#if defined(IEEE_MATH)
              if (isnan(coord[ip][4]) || isinf(coord[ip][4])) {
                long i;
                printf("error: bad time coordinate for particle %ld\n", ip);
                fflush(stdout);
                for (i = 0; i < 6; i++)
                  printf("%15.8e ", coord[ip][i]);
                fflush(stdout);
                fputc('\n', stdout);
                printf("P_average = %e  P_central = %e  t = %e  dP_centroid = %e\n",
                       P_average, *P_central, t, dP_centroid);
                fflush(stdout);
#  if (USE_MPI)
                if (active)
                  MPI_Abort(MPI_COMM_WORLD, 1);
#  endif
                abort();
              }
#endif
            }
          }
        }

        log_exit("do_match_energy");
      }

      void set_central_energy(
                              double **coord,
                              long np,
                              double new_energy, /* new central gamma - 1*/
                              double *P_central) {

        log_entry("set_central_energy");
        set_central_momentum(coord, np, sqrt(sqr(new_energy + 1) - 1), P_central);
        log_exit("set_central_energy");
      }

      void set_central_momentum(
                                double **coord,
                                long np,
                                double P_new, /* new central beta*gamma */
                                double *P_central) {
        long ip;

#ifdef HAVE_GPU
        if (getElementOnGpu()) {
#  ifdef GPU_VERIFY
          double P_central_init = *P_central;
#  endif /* GPU_VERIFY */
          startGpuTimer();
          gpu_set_central_momentum(np, P_new, P_central);
#  ifdef GPU_VERIFY
          startCpuTimer();
          set_central_momentum(coord, np, P_new, P_central);
          compareGpuCpu(np, "set_central_momentum");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */

#if (!USE_MPI)
        if (!np) {
          *P_central = P_new;
          return;
        }
        if (*P_central != P_new) {
          for (ip = 0; ip < np; ip++)
            coord[ip][5] = ((1 + coord[ip][5]) * (*P_central) - P_new) / P_new;
          *P_central = P_new;
        }
#else
        if (notSinglePart) {
          if (!np)
            *P_central = P_new;

          if (*P_central != P_new) {
#  ifdef DEBUG_FIDUCIALIZATION
            printf("Changing reference momentum from %e to %e in %s at %e to match beam\n",
                   *P_central, P_new, trackingContext.elementName, trackingContext.zEnd);
#  endif
            if (((parallelStatus == trueParallel) && isSlave) || ((parallelStatus != trueParallel) && isMaster)) {
              for (ip = 0; ip < np; ip++)
                coord[ip][5] = ((1 + coord[ip][5]) * (*P_central) - P_new) / P_new;
            }
            *P_central = P_new;
          }
        } else {
          if (!np) {
            *P_central = P_new;
            return;
          }
          if (*P_central != P_new) {
#  ifdef DEBUG_FIDUCIALIZATION
            printf("Changing reference momentum from %e to %e in %s\n",
                   *P_central, P_new, trackingContext.elementName);
#  endif
            for (ip = 0; ip < np; ip++)
              coord[ip][5] = ((1 + coord[ip][5]) * (*P_central) - P_new) / P_new;
            *P_central = P_new;
          }
        }
#endif
      }

      void remove_correlations(double **part, REMCOR *remcor, long np) {
        double sumxy, sumy2, ratio;
        long ip, ic, wc;
        long removeFrom[4];

        if (!np)
          return;

        removeFrom[0] = remcor->x;
        removeFrom[1] = remcor->xp;
        removeFrom[2] = remcor->y;
        removeFrom[3] = remcor->yp;
        wc = remcor->with - 1;

        for (ip = sumy2 = 0; ip < np; ip++)
          sumy2 += part[ip][wc] * part[ip][wc];
        if (!sumy2)
          return;

        for (ic = 0; ic < 4; ic++) {
          if (!removeFrom[ic] || ic == wc)
            continue;
          if (!remcor->ratioSet[ic]) {
            for (ip = sumxy = 0; ip < np; ip++)
              sumxy += part[ip][ic] * part[ip][wc];
            ratio = sumxy / sumy2;
            if (remcor->onceOnly) {
              remcor->ratio[ic] = ratio;
              remcor->ratioSet[ic] = 1;
            }
          } else
            ratio = remcor->ratio[ic];
          for (ip = 0; ip < np; ip++)
            part[ip][ic] -= part[ip][wc] * ratio;
        }
      }

      void center_beam(double **part, CENTER *center, long np, long iPass, double p0) {
        double sum, offset;
        long i, ic;
        /*
          printf("center_beam called with np=%ld\n", np);
          fflush(stdout);
        */
#ifdef HAVE_GPU
        if (getElementOnGpu()) {
          startGpuTimer();
          gpu_center_beam(center, np, iPass, p0);
#  ifdef GPU_VERIFY
          startCpuTimer();
          center_beam(part, center, np, iPass, p0);
          compareGpuCpu(np, "center_beam");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */
        /*
          if (!np) {
          return;
          }
        */

        if (center->onPass >= 0 && iPass != center->onPass)
          return;
        /*
          printf("center_beam starting loop, notSinglePart=%ld\n", notSinglePart);
          fflush(stdout);
        */
        for (ic = 0; ic < 6; ic++) {
          if (center->doCoord[ic]) {
            if (!center->deltaSet[ic]) {
              for (i = sum = 0; i < np; i++)
                sum += part[i][ic];
#if USE_MPI
              if (notSinglePart) {
                double sum_total;
                long np_total;

                MPI_Allreduce(&sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                sum = sum_total;
                MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
                if (np_total > 0)
                  center->delta[ic] = offset = sum / np_total;
                else
                  center->delta[ic] = offset = 0;
              } else
                center->delta[ic] = offset = (np > 0 ? sum / np : 0);
#else
              center->delta[ic] = offset = (np > 0 ? sum / np : 0);
#endif
              if (center->onceOnly)
                center->deltaSet[ic] = 1;
            } else
              offset = center->delta[ic];
            /*      printf("centering coordinate %ld by subtracting %le\n", ic, offset); */
            for (i = 0; i < np; i++)
              part[i][ic] -= offset;
          }
        }
        /*
          printf("center_beam step 1 done\n");
          fflush(stdout);
        */
        if (center->doCoord[ic = 6]) {
          /* Special treatment for time coordinate */
          double *timeCoord;
          timeCoord = tmalloc(sizeof(*timeCoord) * np);
          offset = computeTimeCoordinates(timeCoord, p0, part, np);
          if (center->deltaSet[ic])
            offset = center->delta[ic];
          for (i = 0; i < np; i++)
            timeCoord[i] -= offset;
          computeDistanceCoordinates(timeCoord, p0, part, np);
          if (center->onceOnly && !center->deltaSet[ic]) {
            center->delta[ic] = offset;
            center->deltaSet[ic] = 1;
          }
        }
        /*
          printf("center_beam step 2 done\n");
          fflush(stdout);
        */
      }

      void drift_beam(double **part, long np, double length, long order) {
        VMATRIX *M;

        log_entry("drift_beam");

        if (length) {
          M = drift_matrix(length, order);
          track_particles(part, M, part, np);
          free_matrices(M);
          tfree(M);
          M = NULL;
        }
        log_exit("drift_beam");
      }

      void scatter_ele(double **part, long np, double Po, SCATTER *scat, long iPass) {
        long i, ip;
        double t, P, beta;

        if (!np)
          return;
        if (iPass < scat->startOnPass || (scat->endOnPass >= 0 && iPass > scat->endOnPass))
          return;

        log_entry("scatter");
        if (strcasecmp(scat->distribution, "gaussian") == 0) {
          double sigma[4];
          sigma[0] = scat->x;
          sigma[1] = scat->xp;
          sigma[2] = scat->y;
          sigma[3] = scat->yp;
          for (ip = 0; ip < np; ip++) {
            if (scat->probability < 1 && random_2(1) > scat->probability)
              continue;
            for (i = 0; i < 4; i++) {
              if (!sigma[i])
                continue;
              part[ip][i] += gauss_rn(0, random_2) * sigma[i];
            }
            if (scat->dp) {
              P = (1 + part[ip][5]) * Po;
              beta = P / sqrt(sqr(P) + 1);
              t = part[ip][4] / beta;
              part[ip][5] += scat->dp * gauss_rn(0, random_2);
              P = (1 + part[ip][5]) * Po;
              beta = P / sqrt(sqr(P) + 1);
              part[ip][4] = t * beta;
            }
          }
        } else if (strcasecmp(scat->distribution, "uniform") == 0) {
          double u[4];
          u[0] = scat->x;
          u[1] = scat->xp;
          u[2] = scat->y;
          u[3] = scat->yp;
          for (ip = 0; ip < np; ip++) {
            if (scat->probability < 1 && random_2(1) > scat->probability)
              continue;
            for (i = 0; i < 4; i++) {
              if (!u[i])
                continue;
              part[ip][i] += 2 * u[i] * (random_2(1) - 0.5);
            }
            if (scat->dp) {
              P = (1 + part[ip][5]) * Po;
              beta = P / sqrt(sqr(P) + 1);
              t = part[ip][4] / beta;
              part[ip][5] += 2 * scat->dp * (random_2(1) - 0.5);
              P = (1 + part[ip][5]) * Po;
              beta = P / sqrt(sqr(P) + 1);
              part[ip][4] = t * beta;
            }
          }
        } else {
          bombElegantVA("Invalid distribution value \"%s\" given for SCATTER element: use \"gaussian\" or \"uniform\"",
                        scat->distribution);
        }

        log_exit("scatter");
      }

      void store_fitpoint_matrix_values(MARK *fpt, char *name, long occurence, VMATRIX *M) {
        char buffer[1000];
        long i, j, k, l, count;

        if (!M)
          return;
        if (!M->R)
          bombElegant("NULL R matrix passed to store_fitpoint_matrix_values", NULL);

        if (!(fpt->init_flags & 8)) {
          if (M->order == 1) {
            if (!(fpt->matrix_mem = malloc(sizeof(*(fpt->matrix_mem)) * (6 + 36))))
              bombElegant("memory allocation failure (store_fitpoint_matrix_values)", NULL);
          } else if (M->order == 2) {
            if (!(fpt->matrix_mem = malloc(sizeof(*(fpt->matrix_mem)) * (6 + 36 + 126))))
              bombElegant("memory allocation failure (store_fitpoint_matrix_values)", NULL);
          } else if (!(fpt->matrix_mem = malloc(sizeof(*(fpt->matrix_mem)) * (6 + 36 + 126 + 336))))
            bombElegant("memory allocation failure (store_fitpoint_matrix_values)", NULL);
          for (i = count = 0; i < 6; i++) {
            sprintf(buffer, "%s#%ld.C%ld", name, occurence, i + 1);
            fpt->matrix_mem[count++] = rpn_create_mem(buffer, 0);
          }
          for (i = 0; i < 6; i++) {
            for (j = 0; j < 6; j++) {
              sprintf(buffer, "%s#%ld.R%ld%ld", name, occurence, i + 1, j + 1);
              fpt->matrix_mem[count++] = rpn_create_mem(buffer, 0);
            }
          }
          if (M->order > 1) {
            for (i = 0; i < 6; i++) {
              for (j = 0; j < 6; j++) {
                for (k = 0; k <= j; k++) {
                  sprintf(buffer, "%s#%ld.T%ld%ld%ld", name, occurence, i + 1, j + 1, k + 1);
                  fpt->matrix_mem[count++] = rpn_create_mem(buffer, 0);
                }
              }
            }
          }
          if (M->order > 2) {
            for (i = 0; i < 6; i++) {
              for (j = 0; j < 6; j++) {
                for (k = 0; k <= j; k++) {
                  for (l = 0; l <= k; l++) {
                    sprintf(buffer, "%s#%ld.U%ld%ld%ld%ld", name, occurence, i + 1, j + 1, k + 1, l + 1);
                    fpt->matrix_mem[count++] = rpn_create_mem(buffer, 0);
                  }
                }
              }
            }
          }
          fpt->init_flags |= 8;
        }

        for (i = count = 0; i < 6; i++)
          rpn_store(M->C[i], NULL, fpt->matrix_mem[count++]);
        for (i = 0; i < 6; i++)
          for (j = 0; j < 6; j++)
            rpn_store(M->R[i][j], NULL, fpt->matrix_mem[count++]);
        if (M->order > 1)
          for (i = 0; i < 6; i++)
            for (j = 0; j < 6; j++)
              for (k = 0; k <= j; k++)
                rpn_store(M->T[i][j][k], NULL, fpt->matrix_mem[count++]);
        if (M->order > 2)
          for (i = 0; i < 6; i++)
            for (j = 0; j < 6; j++)
              for (k = 0; k <= j; k++)
                for (l = 0; l <= k; l++)
                  rpn_store(M->Q[i][j][k][l], NULL, fpt->matrix_mem[count++]);
      }

      void store_fitpoint_beam_parameters(MARK *fpt, char *name, long occurence, double **coord, long np, double Po) {
        long i, j, k;
        long npTotal;
        static double emit[3], sigma[6], centroid[6], beta[3], alpha[3], emitc[3], min[6], max[6];
        static BEAM_SUMS *sums;
        static char *centroid_name_suffix[8] = {
          "Cx", "Cxp", "Cy", "Cyp", "Cs", "Cdelta", "pCentral", "Particles"};
        static char *sigma_name_suffix[6] = {
          "Sx", "Sxp", "Sy", "Syp", "Ss", "Sdelta"};
        static char *emit_name_suffix[5] = {
          "ex", "ey", "es", "ecx", "ecy"};
        static char *beta_name_suffix[2] = {
          "betaxBeam",
          "betayBeam",
        };
        static char *alpha_name_suffix[2] = {
          "alphaxBeam",
          "alphayBeam",
        };
        static char *max_name_suffix[6] = {
          "max1", "max2", "max3", "max4", "max5", "max6"};
        static char *min_name_suffix[6] = {
          "min1", "min2", "min3", "min4", "min5", "min6"};
        static char s[1000];

        sums = allocateBeamSums(0, 1);
        zero_beam_sums(sums, 1);
        accumulate_beam_sums(sums, coord, np, Po, 0.0, NULL, 0.0, 0.0, -1, -1, 0);
#if USE_MPI
#  if MPI_DEBUG
        printf("myid=%d: np = %ld, parallelStatus = %d, partOnMaster = %d, notSinglePart = %d\n", myid, np,
               parallelStatus, partOnMaster, notSinglePart);
#  endif
        if (parallelStatus == trueParallel && !partOnMaster && notSinglePart)
          MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        else
          npTotal = np;
#  if MPI_DEBUG
        printf("myid=%d: npTotal = %ld\n", myid, npTotal);
#  endif
#else
        npTotal = np;
#endif
        /* if (isMaster || !notSinglePart) { */
        for (i = 0; i < 6; i++) {
          centroid[i] = sums->centroid[i];
          sigma[i] = sqrt(sums->beamSums2->sigma[i][i]);
          min[i] = sums->beamSums2->min[i];
          max[i] = sums->beamSums2->max[i];
          if (i % 2 == 0) {
            beta[i / 2] = alpha[i / 2] = emitc[i / 2] = emit[i / 2] = 0;
            computeEmitTwissFromSigmaMatrix(emit + i / 2, emitc + i / 2, beta + i / 2, alpha + i / 2, sums->beamSums2->sigma, i);
          }
        }

        if (!(fpt->init_flags & 2)) {
          fpt->centroid_mem = tmalloc(sizeof(*fpt->centroid_mem) * 8);
          fpt->sigma_mem = tmalloc(sizeof(*fpt->sigma_mem) * 6);
          fpt->min_mem = tmalloc(sizeof(*fpt->min_mem) * 6);
          fpt->max_mem = tmalloc(sizeof(*fpt->max_mem) * 6);
          fpt->emit_mem = tmalloc(sizeof(*fpt->emit_mem) * 5);
          fpt->betaBeam_mem = tmalloc(sizeof(*fpt->betaBeam_mem) * 2);
          fpt->alphaBeam_mem = tmalloc(sizeof(*fpt->alphaBeam_mem) * 2);
          fpt->sij_mem = tmalloc(sizeof(*fpt->sigma_mem) * 15);
          for (i = 0; i < 8; i++) {
            sprintf(s, "%s#%ld.%s", name, occurence, centroid_name_suffix[i]);
            fpt->centroid_mem[i] = rpn_create_mem(s, 0);
          }
          for (i = 0; i < 6; i++) {
            sprintf(s, "%s#%ld.%s", name, occurence, sigma_name_suffix[i]);
            fpt->sigma_mem[i] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.%s", name, occurence, min_name_suffix[i]);
            fpt->min_mem[i] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.%s", name, occurence, max_name_suffix[i]);
            fpt->max_mem[i] = rpn_create_mem(s, 0);
          }
          for (i = 0; i < 5; i++) {
            sprintf(s, "%s#%ld.%s", name, occurence, emit_name_suffix[i]);
            fpt->emit_mem[i] = rpn_create_mem(s, 0);
          }
          for (i = 0; i < 2; i++) {
            sprintf(s, "%s#%ld.%s", name, occurence, beta_name_suffix[i]);
            fpt->betaBeam_mem[i] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.%s", name, occurence, alpha_name_suffix[i]);
            fpt->alphaBeam_mem[i] = rpn_create_mem(s, 0);
          }
          for (i = k = 0; i < 6; i++) {
            for (j = i + 1; j < 6; j++, k++) {
              sprintf(s, "%s#%ld.s%ld%ld", name, occurence, i + 1, j + 1);
              fpt->sij_mem[k] = rpn_create_mem(s, 0);
            }
          }
          fpt->init_flags |= 2;
        }
        for (i = 0; i < 6; i++) {
          rpn_store(centroid[i], NULL, fpt->centroid_mem[i]);
          rpn_store(sigma[i], NULL, fpt->sigma_mem[i]);
          rpn_store(min[i], NULL, fpt->min_mem[i]);
          rpn_store(max[i], NULL, fpt->max_mem[i]);
        }
        for (i = 0; i < 3; i++)
          rpn_store(emit[i], NULL, fpt->emit_mem[i]);
        for (i = 0; i < 2; i++) {
          rpn_store(emitc[i], NULL, fpt->emit_mem[i + 3]);
          rpn_store(beta[i], NULL, fpt->betaBeam_mem[i]);
          /* printf("%s#%ld.%s = %e\n", name, occurence, beta_name_suffix[i], beta[i]); */
          rpn_store(alpha[i], NULL, fpt->alphaBeam_mem[i]);
        }
        for (i = k = 0; i < 6; i++)
          for (j = i + 1; j < 6; j++, k++)
            rpn_store(sums->beamSums2->sigma[i][j], NULL, fpt->sij_mem[k]);
        rpn_store(Po, NULL, fpt->centroid_mem[6]);
        rpn_store((double)npTotal, NULL, fpt->centroid_mem[7]);
        /* } */
        freeBeamSums(sums, 1);
      }

      void createBPMReadingMemories(ELEMENT_LIST *eptr, double **coord, long np, double Po) {
        /* run through the beamline and force creation of some variables for storing BPM readings */
        while (eptr) {
          switch (eptr->type) {
          case T_MONI:
            if (((MONI *)eptr->p_elem)->storeTurnByTurn)
              storeBPMReading(eptr, coord, np, Po);
            break;
          case T_HMON:
            if (((HMON *)eptr->p_elem)->storeTurnByTurn)
              storeBPMReading(eptr, coord, np, Po);
            break;
          case T_VMON:
            if (((VMON *)eptr->p_elem)->storeTurnByTurn)
              storeBPMReading(eptr, coord, np, Po);
            break;
          default:
            break;
          }
          eptr = eptr->succ;
        }
      }

      void storeBPMReading(ELEMENT_LIST *eptr, double **coord, long np, double Po) {
        BEAM_SUMS *sums;
        char s[1000];
        double x, y;
        MONI *moni;
        HMON *hmon;
        VMON *vmon;
        long n;
        /*
          long npTotal;
        */

        sums = allocateBeamSums(0, 1);
        zero_beam_sums(sums, 1);
        accumulate_beam_sums(sums, coord, np, Po, 0.0, NULL, 0.0, 0.0, -1, -1, 0);
        /*
          #if USE_MPI
          if (parallelStatus==trueParallel && partOnMaster && notSinglePart)
          MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
          else
          npTotal = np;
          #else
          npTotal = np;
          #endif
        */

        x = sums->centroid[0];
        y = sums->centroid[2];
        n = sums->n_part;
        freeBeamSums(sums, 1);

        switch (eptr->type) {
        case T_MONI:
          moni = (MONI *)(eptr->p_elem);
          if (!(moni->initialized & 2)) {
            snprintf(s, 1000, "%s#%ld.x", eptr->name, eptr->occurence);
            moni->tbtMemoryNumber[0] = rpn_create_mem(s, 0);
            snprintf(s, 1000, "%s#%ld.y", eptr->name, eptr->occurence);
            moni->tbtMemoryNumber[1] = rpn_create_mem(s, 0);
            snprintf(s, 1000, "%s#%ld.n", eptr->name, eptr->occurence);
            moni->tbtMemoryNumber[2] = rpn_create_mem(s, 0);
            moni->initialized |= 2;
          }
          rpn_store(computeMonitorReading(eptr, 0, x, y, 0), NULL, moni->tbtMemoryNumber[0]);
          rpn_store(computeMonitorReading(eptr, 2, x, y, 0), NULL, moni->tbtMemoryNumber[1]);
          rpn_store(n, NULL, moni->tbtMemoryNumber[2]);
          break;
        case T_HMON:
          hmon = (HMON *)(eptr->p_elem);
          if (!(hmon->initialized & 2)) {
            snprintf(s, 1000, "%s#%ld.x", eptr->name, eptr->occurence);
            hmon->tbtMemoryNumber[0] = rpn_create_mem(s, 0);
            snprintf(s, 1000, "%s#%ld.n", eptr->name, eptr->occurence);
            hmon->tbtMemoryNumber[1] = rpn_create_mem(s, 0);
            hmon->initialized |= 2;
          }
          rpn_store(computeMonitorReading(eptr, 0, x, y, 0), NULL, hmon->tbtMemoryNumber[0]);
          rpn_store(n, NULL, hmon->tbtMemoryNumber[1]);
          break;
        case T_VMON:
          vmon = (VMON *)(eptr->p_elem);
          if (!(vmon->initialized & 2)) {
            snprintf(s, 1000, "%s#%ld.x", eptr->name, eptr->occurence);
            vmon->tbtMemoryNumber[0] = rpn_create_mem(s, 0);
            snprintf(s, 1000, "%s#%ld.n", eptr->name, eptr->occurence);
            vmon->tbtMemoryNumber[1] = rpn_create_mem(s, 0);
            vmon->initialized |= 2;
          }
          rpn_store(computeMonitorReading(eptr, 2, x, y, 0), NULL, vmon->tbtMemoryNumber[0]);
          rpn_store(n, NULL, vmon->tbtMemoryNumber[1]);
          break;
        }
      }

      ELEMENT_LIST *findBeamlineMatrixElement(ELEMENT_LIST *eptr) {
        ELEMENT_LIST *eptr0 = NULL, *eptrPassed;
        long matrixSeen = 0;
        eptrPassed = eptr;
        while (eptr) {
          if (eptr->type == T_MATR) {
            eptr0 = eptr;
            matrixSeen = 1;
            eptr = eptr->succ;
            break;
          }
          eptr = eptr->succ;
        }
        if (!matrixSeen)
          bombElegant("Can't do \"linear chromatic\" or \"longitudinal-only\" matrix tracking---no matrices!", NULL);
        while (eptr) {
          if ((eptr->p_elem || eptr->matrix) && eptr->type == T_MATR) {
            printWarning("Possible matrix concatenation problem with \"linear chromatic\" or \"longitudinal-only\" matrix tracking",
                         "Concatenation resulted in more than one matrix.  Make sure the additional matrices do not affect the revolution matrix.\n");
            print_elem_list(stdout, eptrPassed);
            break;
          }
          eptr = eptr->succ;
        }
        return eptr0;
      }

      long trackWithIndividualizedLinearMatrix(double **particle, long particles, double **accepted,
                                               double Po, double z, ELEMENT_LIST *eptr,
                                               TWISS *twiss,
                                               double *tune0,
                                               double *chrom,  /* d   nu /ddelta   */
                                               double *chrom2, /* d^2 nu /ddelta^2 */
                                               double *chrom3, /* d^3 nu /ddelta^3 */
                                               double *dbeta_dPoP,
                                               double *dalpha_dPoP,
                                               double *alphac, /* Cs = Cs(0) + delta*alphac[0] + delta^2*alphac[1] */
                                               double *eta2,   /* x = x(0) + eta*delta + eta2*delta^2 */
                                               ILMATRIX *ilmat /* used only if twiss==NULL */
                                               ) {
        long ip, plane, offset, i, j, itop, is_lost;
        double *coord, deltaPoP, tune2pi, sin_phi, cos_phi;
        double alpha[2], beta[2], eta[4], beta1, alpha1, eta1, etap1, A[2];
        double R11, R22, R12;
        long allowResonanceCrossing = 0;
        static VMATRIX *M1 = NULL;
        double det;

        if (ilmat)
          allowResonanceCrossing = ilmat->allowResonanceCrossing;

        if (!M1) {
          M1 = tmalloc(sizeof(*M1));
          initialize_matrices(M1, 1);
        }
        if (twiss) {
          beta[0] = twiss->betax;
          beta[1] = twiss->betay;
          alpha[0] = twiss->alphax;
          alpha[1] = twiss->alphay;
          eta[0] = twiss->etax;
          eta[1] = twiss->etapx;
          eta[2] = twiss->etay;
          eta[3] = twiss->etapy;
        } else {
          for (i = 0; i < 2; i++) {
            beta[i] = ilmat->beta[i];
            alpha[i] = ilmat->alpha[i];
          }
          for (i = 0; i < 4; i++)
            eta[i] = ilmat->eta[i];
        }

        for (i = 0; i < 6; i++) {
          if (eptr->matrix && eptr->matrix->C)
            M1->C[i] = eptr->matrix->C[i];
          else
            M1->C[i] = 0;
          for (j = 0; j < 6; j++)
            M1->R[i][j] = i == j ? 1 : 0;
        }

        if (ilmat && ilmat->tilt)
          rotateBeamCoordinatesForMisalignment(particle, particles, ilmat->tilt);

        itop = particles - 1;
        for (ip = 0; ip < particles; ip++) {
          coord = particle[ip];
          deltaPoP = coord[5];
          /* remove the dispersive orbit from the particle coordinates */
          coord[5] -= deltaPoP;
          for (plane = 0; plane < 2; plane++) {
            coord[2 * plane] -= deltaPoP * (eta[2 * plane] + deltaPoP * eta2[2 * plane]);
            coord[2 * plane + 1] -= deltaPoP * (eta[2 * plane + 1] + deltaPoP * eta2[2 * plane + 1]);
            /* compute the betatron amplitude, if needed */
            A[plane] = 0;
            if (ilmat)
              A[plane] = (sqr(coord[2 * plane]) + ipow(alpha[plane] * coord[2 * plane] + beta[plane] * coord[2 * plane + 1], 2)) / beta[plane];
          }
          is_lost = 0;
          for (plane = 0; !is_lost && plane < 2; plane++) {
            tune2pi = PIx2 * (tune0[plane] +
                              deltaPoP * (chrom[plane] +
                                          deltaPoP / 2 * (chrom2[plane] + deltaPoP / 3 * chrom3[plane])));
            eta1 = eta[2 * plane] + deltaPoP * eta2[2 * plane];
            etap1 = eta[2 * plane + 1] + deltaPoP * eta2[2 * plane + 1];
            if (ilmat)
              tune2pi += PIx2 * (A[0] * (ilmat->tswax[plane] + A[0] * ilmat->tswax2[plane] / 2) +
                                 A[1] * (ilmat->tsway[plane] + A[1] * ilmat->tsway2[plane] / 2) +
                                 A[0] * A[1] * ilmat->tswaxay[plane]);
            offset = 2 * plane;
            if ((beta1 = beta[plane] + dbeta_dPoP[plane] * deltaPoP) <= 0) {
              if (ilmat->verbosity) {
                printf("nonpositive beta function for particle with delta=%le\n",
                       deltaPoP);
                printf("particle is lost\n");
              }
              is_lost = 1;
              continue;
            }
            if (!allowResonanceCrossing && labs(((long)(2 * tune2pi / PIx2)) - ((long)(2 * tune0[plane]))) != 0) {
              if (ilmat->verbosity) {
                printf("particle with delta=%le crossed integer or half-integer resonance\n",
                       deltaPoP);
                printf("particle is lost\n");
              }
              is_lost = 1;
              continue;
            }
            /* R11=R22 or R33=R44 */
            sin_phi = sin(tune2pi);
            cos_phi = cos(tune2pi);
            alpha1 = alpha[plane] + dalpha_dPoP[plane] * deltaPoP;
            /* R11 or R33 */
            R11 = M1->R[0 + offset][0 + offset] = cos_phi + alpha1 * sin_phi;
            /* R22 or R44 */
            R22 = M1->R[1 + offset][1 + offset] = cos_phi - alpha1 * sin_phi;
            /* R12 or R34 */
            if ((R12 = M1->R[0 + offset][1 + offset] = beta1 * sin_phi)) {
              /* R21 or R43 */
              M1->R[1 + offset][0 + offset] = (R11 * R22 - 1) / R12;
            } else {
              bombElegant("divided by zero in trackWithChromaticLinearMatrix", NULL);
            }
            /* Don't actually need these as we've set delta=0 */
            /* R16 or R36 */
            M1->R[0 + offset][5] = eta1 - eta1 * cos_phi - (alpha1 * eta1 + beta1 * etap1) * sin_phi;
            /* R26 or R46 */
            M1->R[1 + offset][5] = etap1 - etap1 * cos_phi + (eta1 + sqr(alpha1) * eta1 + alpha1 * beta1 * etap1) * sin_phi / beta1;
            /* We'll need these path-length terms later */
            /* R51 or R53 */
            M1->R[4][0 + offset] = -etap1 + etap1 * cos_phi + (eta1 + sqr(alpha1) * eta1 + alpha1 * beta1 * etap1) * sin_phi / beta1;
            /* R52 or R54 */
            M1->R[4][1 + offset] = eta1 - eta1 * cos_phi + (alpha1 * eta1 + beta1 * etap1) * sin_phi;

            det = M1->R[0 + offset][0 + offset] * M1->R[1 + offset][1 + offset] -
              M1->R[0 + offset][1 + offset] * M1->R[1 + offset][0 + offset];
            if (fabs(det - 1) > 1e-6) {
              if (ilmat->verbosity) {
                printf("Determinant is suspect for particle with delta=%e\n", deltaPoP);
                printf("particle is lost\n");
              }
              is_lost = 1;
              continue;
            }
          }
          if (is_lost) {
            swapParticles(particle[ip], particle[itop]);
            if (accepted)
              swapParticles(accepted[ip], accepted[itop]);
            particle[itop][4] = z;
            particle[itop][5] = Po * (1 + deltaPoP);
            --itop;
            --ip;
            --particles;
          } else {
            /* momentum-dependent pathlength */
            if (eptr->matrix && eptr->matrix->C)
              M1->C[4] = eptr->matrix->C[4] * (1 + deltaPoP * (alphac[0] + deltaPoP * (alphac[1] + deltaPoP * alphac[2])));
            else if (ilmat)
              M1->C[4] = ilmat->length * (1 + deltaPoP * (alphac[0] + deltaPoP * (alphac[1] + deltaPoP * alphac[2])));
            if (ilmat)
              /* amplitude-dependent path length */
              M1->C[4] += A[0] * (ilmat->dsdA[0] + A[0] * ilmat->dsdA2[0] / 2) +
                A[1] * (ilmat->dsdA[1] + A[1] * ilmat->dsdA2[1] / 2) +
                A[0] * A[1] * ilmat->dsdAxAy;
            track_particles(&coord, M1, &coord, 1);
            /* add back the dispersive orbit to the particle coordinates */
            coord[5] += deltaPoP;
            for (plane = 0; plane < 2; plane++) {
              coord[2 * plane] += deltaPoP * (eta[2 * plane] + deltaPoP * eta2[2 * plane]);
              coord[2 * plane + 1] += deltaPoP * (eta[2 * plane + 1] + deltaPoP * eta2[2 * plane + 1]);
            }
          }
        }

        if (ilmat && ilmat->tilt)
          rotateBeamCoordinatesForMisalignment(particle, particles, -ilmat->tilt);

        return particles;
      }

      void trackLongitudinalOnlyRing(double **part, long np, VMATRIX *M, double *alpha) {
        long ip;
        double *coord, length, alpha1, alpha2;

        alpha1 = alpha[0];
        alpha2 = alpha[1];
        length = M->C[4];
        for (ip = 0; ip < np; ip++) {
          coord = part[ip];
          coord[0] = coord[1] = coord[2] = coord[3] = 0;
          coord[4] += length * (1 + (alpha1 + alpha2 * coord[5]) * coord[5]);
        }
      }

      void matr_element_tracking(double **coord, VMATRIX *M, MATR *matr,
                                 long np, double z)
      /* subtract off <s> prior to using a user-supplied matrix to avoid possible
       * problems with R5? and T?5? elements
       */
      {
        long i;

#ifdef HAVE_GPU
        if (getElementOnGpu()) {
          startGpuTimer();
          gpu_matr_element_tracking(M, matr, np, z);
#  ifdef GPU_VERIFY
          startCpuTimer();
          matr_element_tracking(coord, M, matr, np, z);
          compareGpuCpu(np, "matr_element_tracking");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */

#if !USE_MPI
        if (!np)
          return;
#endif
        if (!matr) {
          track_particles(coord, M, coord, np);
        } else {
          if (!matr->fiducialSeen) {
            double sum = 0;
            for (i = 0; i < np; i++)
              sum += coord[i][4];
#if !USE_MPI
            matr->sReference = sum / np;
#else
            if (notSinglePart) {
              if (isSlave) {
                double sum_total;
                long np_total;

                MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
                MPI_Allreduce(&sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, workers);
                matr->sReference = sum_total / np_total;
              }
            } else
              matr->sReference = sum / np;
#endif
            matr->fiducialSeen = 1;
          }
          for (i = 0; i < np; i++)
            coord[i][4] -= matr->sReference;
          track_particles(coord, M, coord, np);
          for (i = 0; i < np; i++)
            coord[i][4] += matr->sReference;
        }
      }

      void ematrix_element_tracking(double **coord, VMATRIX *M, EMATRIX *matr,
                                    long np, double z, double *P_central)
      /* subtract off <s> prior to using a user-supplied matrix to avoid possible
       * problems with R5? and T?5? elements
       */
      {
        long i;

#ifdef HAVE_GPU
        if (getElementOnGpu()) {
          startGpuTimer();
          gpu_ematrix_element_tracking(M, matr, np, z, P_central);
#  ifdef GPU_VERIFY
          startCpuTimer();
          ematrix_element_tracking(coord, M, matr, np, z, P_central);
          compareGpuCpu(np, "ematr_element_tracking");
#  endif /* GPU_VERIFY */
          return;
        }
#endif /* HAVE_GPU */

#if !USE_MPI
        if (!np)
          return;
#endif
        if (!matr) {
          fprintf(stderr, "ematrix_element_tracking: matr=NULL, tracking with M (%ld order)\n",
                  M->order);
          track_particles(coord, M, coord, np);
        } else {
          if (!matr->fiducialSeen) {
            double sum = 0;
            for (i = 0; i < np; i++)
              sum += coord[i][4];
#if !USE_MPI
            matr->sReference = sum / np;
#else
            if (notSinglePart) {
              if (isSlave) {
                double sum_total;
                long np_total;

                MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
                MPI_Allreduce(&sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, workers);
                matr->sReference = sum_total / np_total;
              }
            } else
              matr->sReference = sum / np;
#endif
            matr->fiducialSeen = 1;
          }
          for (i = 0; i < np; i++)
            coord[i][4] -= matr->sReference;
          track_particles(coord, M, coord, np);
          for (i = 0; i < np; i++)
            coord[i][4] += matr->sReference;
        }
        if (matr->deltaP)
          *P_central += matr->deltaP;
      }

      void distributionScatter(double **part, long np, double Po, DSCATTER *scat, long iPass) {
        static DSCATTER_GROUP *dscatterGroup = NULL;
        static long dscatterGroups = 0;
        long i, ip, interpCode, nScattered, nLeftThisPass;
        double t, P, beta, amplitude, cdf;
        TRACKING_CONTEXT context;

        if (!np)
          return;
        getTrackingContext(&context);

        if (!scat->initialized) {
          SDDS_DATASET SDDSin;
          static char *planeName[5] = {"x", "xp", "y", "yp", "dp"};
          static short planeIndex[5] = {0, 1, 2, 3, 5};
          scat->initialized = 1;
          scat->indepData = scat->cdfData = NULL;
          scat->groupIndex = -1;
          if ((i = match_string(scat->plane, planeName, 5, MATCH_WHOLE_STRING)) < 0) {
            fprintf(stderr, "Error for %s: plane is not valid.  Give x, xp, y, yp, or dp\n",
                    context.elementName);
            exitElegant(1);
          }
          scat->iPlane = planeIndex[i];
          if (!SDDS_InitializeInputFromSearchPath(&SDDSin, scat->fileName) ||
              SDDS_ReadPage(&SDDSin) != 1) {
            fprintf(stderr, "Error for %s: file %s is not valid.\n", context.elementName, scat->fileName);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
            exitElegant(1);
          }
          if ((scat->nData = SDDS_RowCount(&SDDSin)) < 2) {
            fprintf(stderr, "Error for %s: file contains insufficient data.\n", context.elementName);
            exitElegant(1);
          }
          /* Get independent data */
          if (!(scat->indepData = SDDS_GetColumnInDoubles(&SDDSin, scat->valueName))) {
            fprintf(stderr, "Error for %s: independent variable data is invalid.\n",
                    context.elementName);
            exitElegant(1);
          }
          /* Check that independent data is monotonically increasing */
          for (i = 1; i < scat->nData; i++)
            if (scat->indepData[i] <= scat->indepData[i - 1]) {
              fprintf(stderr, "Error for %s: independent variable data is not monotonically increasing.\n",
                      context.elementName);
              exitElegant(1);
            }
          /* Get CDF or PDF data */
          if (!(scat->cdfData = SDDS_GetColumnInDoubles(&SDDSin,
                                                        scat->cdfName ? scat->cdfName : scat->pdfName))) {
            fprintf(stderr, "Error for %s: CDF/PDF data is invalid.\n",
                    context.elementName);
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
            exitElegant(1);
          }
          SDDS_Terminate(&SDDSin);
          if (!(scat->cdfName)) {
            /* must integrate to get the CDF */
            double *integral, *ptr;
            if (!(integral = malloc(sizeof(*integral) * scat->nData))) {
              fprintf(stderr, "Error for %s: memory allocation failure.\n", context.elementName);
              exitElegant(1);
            }
            trapazoidIntegration1(scat->indepData, scat->cdfData, scat->nData, integral);
            ptr = scat->cdfData;
            scat->cdfData = integral;
            free(ptr);
          }
          /* Check that CDF is monotonically increasing */
          for (i = 0; i < scat->nData; i++) {
            if (scat->cdfData[i] < 0 || (i && scat->cdfData[i] <= scat->cdfData[i - 1])) {
              long j;
              fprintf(stderr, "Error for %s: CDF not monotonically increasing at index %ld.\n", context.elementName,
                      i);
              for (j = 0; j <= i + 1 && j < scat->nData; j++)
                fprintf(stderr, "%ld %21.15e\n", j, scat->cdfData[i]);
              exitElegant(1);
            }
          }
          /* Normalize CDF to 1 */
          for (i = 0; i < scat->nData; i++)
            scat->cdfData[i] /= scat->cdfData[scat->nData - 1];

          if (scat->oncePerParticle) {
            /* Figure out scattering group assignment */
            if (scat->group >= 0) {
              for (i = 0; i < dscatterGroups; i++) {
                if (dscatterGroup[i].group == scat->group)
                  break;
              }
            } else
              i = dscatterGroups;
            if (i == dscatterGroups) {
              /* make a new group */
              fprintf(stderr, "Making new scattering group structure (#%ld, group %ld) for %s #%ld\n",
                      i, scat->group, context.elementName, context.elementOccurrence);
              if (!(dscatterGroup = SDDS_Realloc(dscatterGroup, sizeof(*dscatterGroup) * (dscatterGroups + 1)))) {
                fprintf(stderr, "Error for %s: memory allocation failure.\n", context.elementName);
                exitElegant(1);
              }
              dscatterGroup[i].particleIDScattered = NULL;
              dscatterGroup[i].nParticles = 0;
              dscatterGroup[i].group = scat->group;
              scat->firstInGroup = 1;
              dscatterGroups++;
            } else {
              fprintf(stderr, "Linking to scattering group structure (#%ld, group %ld) for %s #%ld\n",
                      i, scat->group, context.elementName, context.elementOccurrence);
              scat->firstInGroup = 0;
            }
            scat->groupIndex = i;
          }
          if (scat->startOnPass < 0)
            scat->startOnPass = 0;
        }

        if (iPass == 0 && scat->oncePerParticle && scat->firstInGroup) {
          /* Initialize data for remembering which particles have been scattered */
          dscatterGroup[scat->groupIndex].nParticles = np;
          if (dscatterGroup[scat->groupIndex].particleIDScattered)
            free(dscatterGroup[scat->groupIndex].particleIDScattered);
          if (!(dscatterGroup[scat->groupIndex].particleIDScattered = malloc(sizeof(*(dscatterGroup[scat->groupIndex].particleIDScattered)) * np))) {
            fprintf(stderr, "Error for %s: memory allocation failure.\n", context.elementName);
            exitElegant(1);
          }
          dscatterGroup[scat->groupIndex].nScattered = 0;
          dscatterGroup[scat->groupIndex].allScattered = 0;
        }

        if (iPass < scat->startOnPass)
          return;
        if (scat->endOnPass >= 0 && iPass > scat->endOnPass)
          return;
        if (scat->oncePerParticle && dscatterGroup[scat->groupIndex].allScattered)
          return;

        if (iPass == scat->startOnPass) {
          scat->nLeft = scat->limitTotal >= 0 ? scat->limitTotal : np;
          if (scat->nLeft > np)
            scat->nLeft = np;
        }
        nLeftThisPass = np;
        if (scat->limitTotal >= 0)
          nLeftThisPass = scat->nLeft;
        if (scat->limitPerPass >= 0) {
          if (scat->limitPerPass < nLeftThisPass)
            nLeftThisPass = scat->limitPerPass;
        }
        if (nLeftThisPass == 0)
          return;

        if (scat->oncePerParticle && dscatterGroup[scat->groupIndex].nParticles < np) {
          fprintf(stderr, "Error for %s: number of particles is greater than the size of the particle ID array.\n",
                  context.elementName);
          exitElegant(1);
        }

        nScattered = 0;
        for (ip = 0; ip < np; ip++) {
          if (scat->probability < 1 && random_2(1) > scat->probability)
            continue;
          if (nLeftThisPass == 0)
            break;
          if (scat->oncePerParticle) {
            short found = 0;
            if (dscatterGroup[scat->groupIndex].nScattered >= dscatterGroup[scat->groupIndex].nParticles) {
              fprintf(stderr, "All particles scattered for group %ld (nscattered=%ld, np0=%ld)\n",
                      scat->group, dscatterGroup[scat->groupIndex].nScattered,
                      dscatterGroup[scat->groupIndex].nParticles);
              dscatterGroup[scat->groupIndex].allScattered = 1;
              return;
            }
            for (i = 0; i < dscatterGroup[scat->groupIndex].nScattered; i++) {
              if (dscatterGroup[scat->groupIndex].particleIDScattered[i] == part[ip][6]) {
                found = 1;
                break;
              }
            }
            if (found)
              continue;
            dscatterGroup[scat->groupIndex].particleIDScattered[i] = part[ip][6];
            dscatterGroup[scat->groupIndex].nScattered += 1;
          }
          nScattered++;
          nLeftThisPass--;
          cdf = random_2(1);
          amplitude = scat->factor * interp(scat->indepData, scat->cdfData, scat->nData, cdf, 0, 1, &interpCode);
          if (scat->randomSign)
            amplitude *= random_2(1) > 0.5 ? 1 : -1;
          if (!interpCode) {
            char buffer[16384];
            snprintf(buffer, 16384, ". Element %s, CDF=%e",
                     context.elementName, cdf);
            printWarning("DSCATTER interpolation error", buffer);
          }
          if (scat->iPlane == 5) {
            /* momentum scattering */
            P = (1 + part[ip][5]) * Po;
            beta = P / sqrt(sqr(P) + 1);
            t = part[ip][4] / beta;
            part[ip][5] += amplitude;
            P = (1 + part[ip][5]) * Po;
            beta = P / sqrt(sqr(P) + 1);
            part[ip][4] = t * beta;
          } else
            part[ip][scat->iPlane] += amplitude;
        }
        scat->nLeft -= nScattered;
        /*
          fprintf(stderr, "%ld particles scattered by %s#%ld, group %ld, total=%ld\n",
          nScattered, context.elementName, context.elementOccurrence,
          scat->group, dscatterGroup[scat->groupIndex].nScattered);
        */
      }

      void recordLostParticles(
                               BEAM *beam,
                               double **coord, /* particle coordinates, with lost particles swapped to the top of the array */
                               long nLeft,     /* coord+nLeft is first lost particle */
                               long nToTrack,
                               long pass /* pass on which loss occurred */
                               ) {
        long i;

#if USE_MPI && MPI_DEBUG
        static FILE *fp = NULL;
#endif
        /*
          printf("Recording lost particles, nLeft=%ld, nToTrack=%ld, beam->n_lost=%ld\n",
          nLeft, nToTrack, beam?beam->n_lost:-1); fflush(stdout);
        */

        if (coord)
          for (i = nLeft; i < nToTrack; i++)
            coord[i][lossPassIndex] = pass;
        if (beam) {
          long newLost;
          newLost = (nToTrack - nLeft);
          beam->n_lost += newLost;
          /*
            printf("New lost = %ld, total lost = %ld\n",
            newLost, beam->n_lost);
          */
          for (i = nLeft; i < nToTrack; i++)
            beam->particle[i][lossPassIndex] = pass;
#if USE_MPI && MPI_DEBUG
          if (!fp) {
            char s[1024];
            snprintf(s, 1024, "losses-%04d.debug", myid);
            fp = fopen(s, "w");
            fprintf(fp, "SDDS1\n");
            fprintf(fp, "&column name=xLost type=double units=m &end\n");
            fprintf(fp, "&column name=xpLost type=double &end\n");
            fprintf(fp, "&column name=yLost type=double units=m &end\n");
            fprintf(fp, "&column name=ypLost type=double &end\n");
            fprintf(fp, "&column name=zLost type=double units=m &end\n");
            fprintf(fp, "&column name=pLost type=double &end\n");
            fprintf(fp, "&column name=XLost type=double units=m &end\n");
            fprintf(fp, "&column name=ZLost type=double units=m &end\n");
            fprintf(fp, "&column name=LossPass type=long &end\n");
            fprintf(fp, "&column name=particleID type=long &end\n");
            fprintf(fp, "&parameter name=Pass type=long &end\n");
            fprintf(fp, "&parameter name=ElementName type=string &end\n");
            fprintf(fp, "&data mode=ascii &end\n");
          }
          fprintf(fp, "%ld\n%s\n%ld\n", pass, trackingContext.elementName, newLost);
          for (i = nLeft; i < nToTrack; i++)
            fprintf(fp, "%le %le %le %le %le %le %le %le %ld %ld\n",
                    beam->particle[i][0], beam->particle[i][1],
                    beam->particle[i][2], beam->particle[i][3],
                    beam->particle[i][4], beam->particle[i][5],
                    globalLossCoordOffset > 0 ? beam->particle[i][globalLossCoordOffset] : -999,
                    globalLossCoordOffset > 0 ? beam->particle[i][globalLossCoordOffset + 2] : -999,
                    (long)beam->particle[i][lossPassIndex], (long)beam->particle[i][6]);
#endif
        }
      }

      void storeMonitorOrbitValues(ELEMENT_LIST *eptr, double **part, long np) {
        MONI *moni;
        HMON *hmon;
        VMON *vmon;
        MARK *mark;
        char s[1000];
        double centroid[6];
        long i;

        if (!np)
          return;

        switch (eptr->type) {
        case T_MONI:
          moni = (MONI *)eptr->p_elem;
          if (!moni->coFitpoint)
            return;
          compute_centroids(centroid, part, np);
          if (!(moni->initialized & 1)) {
            sprintf(s, "%s#%ld.xco", eptr->name, eptr->occurence);
            moni->coMemoryNumber[0] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.xpco", eptr->name, eptr->occurence);
            moni->coMemoryNumber[1] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.yco", eptr->name, eptr->occurence);
            moni->coMemoryNumber[2] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.ypco", eptr->name, eptr->occurence);
            moni->coMemoryNumber[3] = rpn_create_mem(s, 0);
            moni->initialized |= 1;
          }
          for (i = 0; i < 4; i++)
            rpn_store(centroid[i], NULL, moni->coMemoryNumber[i]);
          break;
        case T_MARK:
          mark = (MARK *)eptr->p_elem;
          if (!mark->fitpoint)
            return;
          compute_centroids(centroid, part, np);
          if (mark->co_mem == NULL) {
            mark->co_mem = tmalloc(4 * sizeof(*(mark->co_mem)));
            sprintf(s, "%s#%ld.xco", eptr->name, eptr->occurence);
            mark->co_mem[0] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.xpco", eptr->name, eptr->occurence);
            mark->co_mem[1] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.yco", eptr->name, eptr->occurence);
            mark->co_mem[2] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.ypco", eptr->name, eptr->occurence);
            mark->co_mem[3] = rpn_create_mem(s, 0);
          }
          for (i = 0; i < 4; i++)
            rpn_store(centroid[i], NULL, mark->co_mem[i]);
          break;
        case T_HMON:
          hmon = (HMON *)eptr->p_elem;
          if (!hmon->coFitpoint)
            return;
          compute_centroids(centroid, part, np);
          if (!(hmon->initialized & 1)) {
            sprintf(s, "%s#%ld.xco", eptr->name, eptr->occurence);
            hmon->coMemoryNumber[0] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.xpco", eptr->name, eptr->occurence);
            hmon->coMemoryNumber[1] = rpn_create_mem(s, 0);
            hmon->initialized |= 1;
          }
          for (i = 0; i < 2; i++)
            rpn_store(centroid[i], NULL, hmon->coMemoryNumber[i]);
          break;
        case T_VMON:
          vmon = (VMON *)eptr->p_elem;
          if (!vmon->coFitpoint)
            return;
          compute_centroids(centroid, part, np);
          if (!(vmon->initialized & 1)) {
            sprintf(s, "%s#%ld.yco", eptr->name, eptr->occurence);
            vmon->coMemoryNumber[0] = rpn_create_mem(s, 0);
            sprintf(s, "%s#%ld.ypco", eptr->name, eptr->occurence);
            vmon->coMemoryNumber[1] = rpn_create_mem(s, 0);
            vmon->initialized |= 1;
          }
          for (i = 0; i < 2; i++)
            rpn_store(centroid[i + 2], NULL, vmon->coMemoryNumber[i]);
          break;
        default:
          return;
        }
      }

#define DEBUG_SCATTER 0

#if USE_MPI
      void scatterParticles(double **coord, long *nToTrack, double **accepted,
                            long n_processors, int myid, balance balanceStatus,
                            double my_rate, double nParPerElements, double round,
                            int lostSinceSeqMode, int *distributed,
                            long *reAllocate, double *P_central) {
        long work_processors = n_processors - 1;
        int root = 0, i, j;
        int my_nToTrack, nItems, *nToTrackCounts;
        double total_rate, constTime, *rateCounts;
        MPI_Status status;
#  if DEBUG_SCATTER
        FILE *fpdeb;
        char s[1000];
#  endif

#  ifdef HAVE_GPU
        coord = forceParticlesToCpu("scatterParticles");
#  endif
        if (myid == 0) {
#  if MPI_DEBUG
          printf("Distributing %ld particles to %ld worker processors\n", *nToTrack, n_processors - 1);
          printf("Distributing particles, %ld particles now on processor %d\n", *nToTrack, myid);
#  endif
          fflush(stdout);
#  if DEBUG_SCATTER
          fpdeb = fopen("scatter.0", "w");
          fprintf(fpdeb, "SDDS1\n");
          fprintf(fpdeb, "&column name=x type=double units=m &end\n");
          fprintf(fpdeb, "&column name=xp type=double &end\n");
          fprintf(fpdeb, "&column name=y type=double units=m &end\n");
          fprintf(fpdeb, "&column name=yp type=double &end\n");
          fprintf(fpdeb, "&column name=t type=double units=s &end\n");
          fprintf(fpdeb, "&column name=p type=double &end\n");
          fprintf(fpdeb, "&column name=particleID type=long &end\n");
          fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
          for (i = 0; i < *nToTrack; i++) {
            for (j = 0; j < 6; j++)
              fprintf(fpdeb, "%le ", coord[i][j]);
            fprintf(fpdeb, "%.0lf\n", coord[i][j]);
          }
          fclose(fpdeb);
#  endif
        }

        nToTrackCounts = malloc(sizeof(int) * n_processors);
        rateCounts = malloc(sizeof(double) * n_processors);

        /* The particles will be distributed to slave processors evenly for the first pass */
        if (((balanceStatus == startMode) && (!*distributed))) {
          /* || lostSinceSeqMode || *reAllocate )  we don't do redistribution for load balancing, as it could cause memory problem */
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 1\n");
          fflush(stdout);
#  endif
          MPI_Bcast(nToTrack, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 1.1\n");
          fflush(stdout);
#  endif
          if (myid == 0)
            my_nToTrack = 0;
          else {
            my_nToTrack = *nToTrack / work_processors;
            if (myid <= (*nToTrack % work_processors))
              my_nToTrack++;
          }
          /* gather the number of particles to be sent to each processor */
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 1.2\n");
          fflush(stdout);
#  endif
          MPI_Gather(&my_nToTrack, 1, MPI_INT, nToTrackCounts, 1, MPI_INT, root, MPI_COMM_WORLD);
          *distributed = 1;
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 1.3, my_nToTrack=%d\n", my_nToTrack);
          fflush(stdout);
#  endif
        } else if (balanceStatus == badBalance) {
          double minRate;
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 2\n");
          fflush(stdout);
#  endif
          if (myid == 0)
            my_rate = 1.0; /* set it to nonzero */
          MPI_Allreduce(&my_rate, &minRate, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
          if (myid == 0)
            my_rate = 0.0;      /* set it back to zero */
          if (minRate == 0.0) { /* redistribute evenly when all particles are lost on any working processor */
            MPI_Bcast(nToTrack, 1, MPI_LONG, 0, MPI_COMM_WORLD);
            if (myid == 0)
              my_nToTrack = 0;
            else {
              my_nToTrack = *nToTrack / work_processors;
              if (myid <= (*nToTrack % work_processors))
                my_nToTrack++;
            }
            /* gather the number of particles to be sent to each processor */
            MPI_Gather(&my_nToTrack, 1, MPI_INT, nToTrackCounts, 1, MPI_INT, root, MPI_COMM_WORLD);
          } else {
#  if DEBUG_SCATTER
            printf("scatterParticles, branch 3\n");
            fflush(stdout);
#  endif
            /* calculating the number of jobs to be sent according to the speed of each processors */
            MPI_Gather(&my_rate, 1, MPI_DOUBLE, rateCounts, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
            MPI_Reduce(&my_rate, &total_rate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (myid == 0) {
              if (fabs(total_rate - 0.0) > 1e-12)
                constTime = *nToTrack / total_rate;
              else
                constTime = 0.0;
              my_nToTrack = 0;
              for (i = 1; i < work_processors; i++) {
                nToTrackCounts[i] = rateCounts[i] * constTime + round; /* round to the nearest integer */
                /* avoid sending more particles than the available ones */
                if ((my_nToTrack + nToTrackCounts[i]) >= *nToTrack) {
                  nToTrackCounts[i] = *nToTrack - my_nToTrack;
                  my_nToTrack = *nToTrack;
                  break;
                }
                /* count the total number of particles that have been assigned */
                my_nToTrack = my_nToTrack + nToTrackCounts[i];
              }
              if (i < work_processors) /* assign 0 particles to the remaining processors */
                for (j = i; j <= work_processors; j++)
                  nToTrackCounts[j] = 0;
              else /* The last processor will be responsible for all of the remaining particles */
                nToTrackCounts[work_processors] = *nToTrack - my_nToTrack;
#  ifdef MPI_DEBUG
              printf("total_rate=%lf, nToTrack=%ld, nToTrackForDistribution=%d\n",
                     total_rate, *nToTrack, my_nToTrack + nToTrackCounts[work_processors]);
#  endif
            }
            /* scatter the number of particles to be sent to each processor */
            MPI_Scatter(nToTrackCounts, 1, MPI_INT, &my_nToTrack, 1, MPI_INT, root, MPI_COMM_WORLD);
            *reAllocate = 0; /* set the flag back to 0 after scattering */
          }
        } else { /* keep the nToTrack unchanged */
#  if DEBUG_SCATTER
          printf("scatterParticles, branch 4\n");
          fflush(stdout);
#  endif
          if (myid == 0)
            my_nToTrack = 0;
          else
            my_nToTrack = *nToTrack;
          /* gather the number of particles to be sent to each processor */
          MPI_Gather(&my_nToTrack, 1, MPI_INT, nToTrackCounts, 1, MPI_INT, root, MPI_COMM_WORLD);
        }

        /* scatter particles to all of the slave processors */
        if (myid == 0) {
          my_nToTrack = 0;
          for (i = 1; i <= work_processors; i++) {
            /* calculate the number of elements that will be sent to each processor */
#  if DEBUG_SCATTER
            printf("Sending %d particles to processor %d\n", nToTrackCounts[i], i);
#  endif
            nItems = nToTrackCounts[i] * totalPropertiesPerParticle;
            MPI_Send(&coord[my_nToTrack][0], nItems, MPI_DOUBLE, i, 104, MPI_COMM_WORLD);
            if (accepted != NULL)
              MPI_Send(&accepted[my_nToTrack][0], nItems, MPI_DOUBLE, i, 105, MPI_COMM_WORLD);
            /* count the total number of particles that have been scattered */
            my_nToTrack = my_nToTrack + nToTrackCounts[i];
          }
          *nToTrack = 0;
        } else {
#  if DEBUG_SCATTER
          printf("receiving %d particles for processor %d\n", my_nToTrack, myid);
#  endif
          MPI_Recv(&coord[0][0], my_nToTrack * totalPropertiesPerParticle, MPI_DOUBLE, 0,
                   104, MPI_COMM_WORLD, &status);
          if (accepted != NULL)
            MPI_Recv(&accepted[0][0], my_nToTrack * totalPropertiesPerParticle,
                     MPI_DOUBLE, 0, 105, MPI_COMM_WORLD, &status);
          *nToTrack = my_nToTrack;
        }
        /* broadcast the P_central to all the slave processors */
        MPI_Bcast(P_central, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#  ifdef MPI_DEBUG
        if (MPI_DEBUG) {
          int namelen;
          char processor_name[MPI_MAX_PROCESSOR_NAME];
          if (myid != 0)
            /*      if ((balanceStatus == badBalance) || lostSinceSeqMode) */
            {
              MPI_Get_processor_name(processor_name, &namelen);
              fprintf(stderr, "%d will be computed on %d (%s)\n", my_nToTrack, myid, processor_name);
            }
        }
#  endif

        if (myid != 0) {
#  if DEBUG_SCATTER
          printf("%ld particles on processor %d\n", *nToTrack, myid);
          fflush(stdout);

          sprintf(s, "scatter.%d", myid);
          fpdeb = fopen(s, "w");
          fprintf(fpdeb, "SDDS1\n");
          fprintf(fpdeb, "&column name=x type=double units=m &end\n");
          fprintf(fpdeb, "&column name=xp type=double &end\n");
          fprintf(fpdeb, "&column name=y type=double units=m &end\n");
          fprintf(fpdeb, "&column name=yp type=double &end\n");
          fprintf(fpdeb, "&column name=t type=double units=s &end\n");
          fprintf(fpdeb, "&column name=p type=double &end\n");
          fprintf(fpdeb, "&column name=particleID type=long &end\n");
          fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
          for (i = 0; i < *nToTrack; i++) {
            for (j = 0; j < 6; j++)
              fprintf(fpdeb, "%le ", coord[i][j]);
            fprintf(fpdeb, "%.0lf\n", coord[i][j]);
          }
          fclose(fpdeb);
#  endif
        }

#  if MPI_DEBUG
        printf("%ld particles now on processor %d\n", *nToTrack, myid);
        report_stats(stdout, "Finished distributing particles: ");
#  endif

        free(nToTrackCounts);
        free(rateCounts);
      }

      void gatherParticles(double ***coord, long *nToTrack, long *nLost, double ***accepted, long n_processors, int myid, double *round) {
        long work_processors = n_processors - 1;
        int root = 0, nItems, displs;
        long i;
        long my_nToTrack, my_nLost, *nToTrackCounts,
          *nLostCounts, current_nLost = 0, nToTrack_total, nLost_total;

#  ifdef HAVE_GPU
        *coord = forceParticlesToCpu("gatherParticles");
#  endif

        MPI_Status status;
        /*
          printf("Gathering particles to master from %ld processors\n", work_processors);
          fflush(stdout);
        */

        nToTrackCounts = malloc(sizeof(long) * n_processors);
        nLostCounts = malloc(sizeof(long) * n_processors);

        if (myid == 0) {
          my_nToTrack = 0;
          my_nLost = 0;
        } else {
          my_nToTrack = *nToTrack;
          my_nLost = *nLost;
        }

        /* gather nToTrack and nLost from all of the slave processors to the master processors */
        MPI_Gather(&my_nToTrack, 1, MPI_LONG, nToTrackCounts, 1, MPI_LONG, root, MPI_COMM_WORLD);
        MPI_Gather(&my_nLost, 1, MPI_LONG, nLostCounts, 1, MPI_LONG, root, MPI_COMM_WORLD);
        MPI_Allreduce(&my_nToTrack, &nToTrack_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&my_nLost, &nLost_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

#  if MPI_DEBUG
        printf("gatherPaticles: nToTrack = %ld, nToTrack_total = %ld, nLost = %ld, nLost_total = %ld\n",
               *nToTrack, nToTrack_total, *nLost, nLost_total);
        if (myid == 0) {
          for (i = 0; i < n_processors; i++) {
            printf("nToTrackCounts[%ld] = %ld, nLostCounts[%ld] = %ld\n",
                   i, nToTrackCounts[i], i, nLostCounts[i]);
            fflush(stdout);
          }
        }
#  endif

        if (isMaster) {
          if (*coord == NULL)
            *coord = (double **)resize_czarray_2d((void **)(*coord), sizeof(double), nToTrack_total + nLost_total, totalPropertiesPerParticle);
          if ((dumpAcceptance && (*accepted == NULL)) || (accepted && *accepted)) {
            *accepted = (double **)resize_czarray_2d((void **)(*accepted), sizeof(double), nToTrack_total + nLost_total, totalPropertiesPerParticle);
          }
        }

        if (myid == 0) {
          for (i = 1; i <= work_processors; i++) {
            /* the number of elements that are received from each processor (for root only) */
            nItems = nToTrackCounts[i] * totalPropertiesPerParticle;
            /* collect information for the left particles */
            MPI_Recv(&(*coord)[my_nToTrack][0], nItems, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &status);

            /* count the total number of particles to track and the total number of lost after the most recent scattering */
            my_nToTrack = my_nToTrack + nToTrackCounts[i];
            current_nLost = current_nLost + nLostCounts[i];
          }
          *nLost = *nLost + current_nLost;
          *nToTrack = my_nToTrack;
        } else {
          MPI_Send(&(*coord)[0][0], my_nToTrack * totalPropertiesPerParticle, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        }

        /* collect information for the lost particles and gather the accepted array */

        MPI_Bcast(&current_nLost, 1, MPI_LONG, root, MPI_COMM_WORLD);

        if (myid == 0) {
          /* set up the displacement array and the number of elements that are received from each processor */
          nLostCounts[0] = 0;
          displs = my_nToTrack;
          my_nToTrack = 0;
          for (i = 1; i <= work_processors; i++) {
            /* gather information for lost particles */
            displs = displs + nLostCounts[i - 1];
            nItems = nLostCounts[i] * totalPropertiesPerParticle;
            MPI_Recv(&(*coord)[displs][0], nItems, MPI_DOUBLE, i, 102, MPI_COMM_WORLD, &status);
            if (accepted && *accepted != NULL) {
              MPI_Recv(&(*accepted)[my_nToTrack][0], nToTrackCounts[i] * totalPropertiesPerParticle, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, &status);
              MPI_Recv(&(*accepted)[displs][0], nItems, MPI_DOUBLE, i, 103, MPI_COMM_WORLD, &status);
              my_nToTrack = my_nToTrack + nToTrackCounts[i];
            }
          }
          /* update the round parameter to avoid more particles are distributed
             than the available particles */
          if ((my_nToTrack / work_processors) < work_processors)
            *round = 0.0;
        } else {
          /* send information for lost particles */
          MPI_Send(&(*coord)[my_nToTrack][0], my_nLost * totalPropertiesPerParticle, MPI_DOUBLE, root, 102, MPI_COMM_WORLD);
          if (accepted && *accepted != NULL) {
            MPI_Send(&(*accepted)[0][0], my_nToTrack * totalPropertiesPerParticle, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
            MPI_Send(&(*accepted)[my_nToTrack][0], my_nLost * totalPropertiesPerParticle, MPI_DOUBLE, root, 103, MPI_COMM_WORLD);
          }
        }
        MPI_Bcast(round, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        /*
          report_stats(stdout, "Finished gathering particles to master:");
        */
        free(nToTrackCounts);
        free(nLostCounts);
      }

      balance checkBalance(double my_wtime, int myid, long n_processors, int verbose) {
        double maxTime, minTime, *time;
        int i, balanceFlag = 1;
        int iFastest, iSlowest;
        static int imbalanceCounter = 2; /* counter for the number of continuously imbalanced passes,
                                            the 1st pass is treated specially */

        time = malloc(sizeof(double) * n_processors);

        MPI_Gather(&my_wtime, 1, MPI_DOUBLE, time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (myid == 0) {
          maxTime = minTime = time[1];
          iFastest = iSlowest = 1;
          for (i = 2; i < n_processors; i++) {
            if (maxTime < time[i]) {
              iFastest = i;
              maxTime = time[i];
            }
            if (minTime > time[i]) {
              iSlowest = i;
              minTime = time[i];
            }
          }
          if ((maxTime - minTime) / minTime > 0.10) {
            imbalanceCounter++;
            /* the workload balancing is treated as bad when there are 3 continuous imbalanced passes or
               it spends more than 5 minutes for a pass */
            if ((imbalanceCounter >= 3) || (maxTime > 300)) {
              balanceFlag = 0;
              imbalanceCounter = 0;
            }
          } else
            imbalanceCounter = 0;
        }
        MPI_Bcast(&balanceFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (verbose && myid == 0) {
          if ((maxTime - minTime) / minTime > 0.10) {
            printf("The balance is in bad status. ");
            printf("The fastest time (id=%d) is %e\n", iFastest, minTime);
            printf("The slowest time (id=%d) is %e\n", iSlowest, maxTime);
          } else
            printf("The balance is in good status. ");
          if (balanceFlag == 0)
            printf("We need redistribute the particles. ");
          printf("The difference is %4.2lf percent\n", (maxTime - minTime) / minTime * 100);
          fflush(stdout);
        }

        free(time);

        if (balanceFlag == 1)
          return goodBalance;
        else
          return badBalance;
      }

      int usefulOperation(ELEMENT_LIST *eptr, unsigned long flags, long i_pass) {
        WATCH *watch;
        HISTOGRAM *histogram;

        if (eptr->type == T_WATCH) {
          if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
            watch = (WATCH *)eptr->p_elem;
            if (!watch->disable) {
              if (!watch->initialized) {
                if (isSlave) /* Slave processors will not go through the WATCH element, so we set the flag here */
                  watch->initialized = 1;
              }
              if (i_pass >= watch->start_pass && (i_pass - watch->start_pass) % watch->interval == 0 &&
                  (watch->end_pass < 0 || i_pass <= watch->end_pass))
                return 1;
            }
          }
          return 0;
        } else if (eptr->type == T_HISTOGRAM) {
          if (!(flags & TEST_PARTICLES) && !(flags & INHIBIT_FILE_OUTPUT)) {
            histogram = (HISTOGRAM *)eptr->p_elem;
            if (!histogram->disable) {
              if (!histogram->initialized) {
                if (isSlave) /* Slave processors will not go through the HISTOGRAM element */
                  histogram->initialized = 1;
                return 1;
              }
              if (i_pass >= histogram->startPass && (i_pass - histogram->startPass) % histogram->interval == 0)
                return 1;
            }
          }
          return 0;
        }
        return 1; /* This is default for all the other UNIPROCESSOR elements */
      }
#endif

#ifdef SORT
      int comp_IDs(const void *coord1, const void *coord2) {
        if (((double *)coord1)[6] < ((double *)coord2)[6])
          return -1;
        else if (((double *)coord1)[6] > ((double *)coord2)[6])
          return 1;
        else
          return 0;
      }
#endif

      void transformEmittances(double **coord, long np, double pCentral, EMITTANCEELEMENT *ee) {
        double emit, emitc, factor, pAverage, eta, etap;
        long i, j;
        BEAM_SUMS *sums;

#if SDDS_MPI_IO
        long npTotal;
        MPI_Reduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (isMaster && npTotal < 10) {
          printf("*** Error: too few particles (<10) for emittance modification\n");
          exitElegant(1);
        }
#endif

        sums = allocateBeamSums(0, 1);
        zero_beam_sums(sums, 1);
        accumulate_beam_sums(sums, coord, np, pCentral, 0, NULL, 0.0, 0.0, -1, -1, 0);
        pAverage = pCentral * (1 + sums->centroid[5]);

        for (i = 0; i < 2; i++) {
          computeEmitTwissFromSigmaMatrix(&emit, &emitc, NULL, NULL, sums->beamSums2->sigma, 2 * i);
          eta = etap = 0;
          if (sums->beamSums2->sigma[5][5]) {
            eta = sums->beamSums2->sigma[2 * i + 0][5] / sums->beamSums2->sigma[5][5];
            etap = sums->beamSums2->sigma[2 * i + 1][5] / sums->beamSums2->sigma[5][5];
          }
          if (ee->emit[i] >= 0) {
            /* use geometric emittance */
            if (emitc > 0)
              factor = sqrt(ee->emit[i] / emitc);
            else
              continue;
          } else if (ee->emitn[i] >= 0) {
            /* use normalized emittance */
            emitc *= pAverage;
            if (emitc > 0)
              factor = sqrt(ee->emitn[i] / emitc);
            else
              continue;
          } else {
            /* do nothing */
            continue;
          }
          for (j = 0; j < np; j++) {
            coord[j][2 * i + 0] = factor * (coord[j][2 * i + 0] - eta * coord[j][5]) + eta * coord[j][5];
            coord[j][2 * i + 1] = factor * (coord[j][2 * i + 1] - eta * coord[j][5]) + eta * coord[j][5];
          }
        }
        freeBeamSums(sums, 1);
      }

      void set_up_mhist(MHISTOGRAM *mhist, RUN *run, long occurence) {
        SDDS_DATASET SDDSin;

        if (mhist->disable)
          return;
        if (mhist->interval <= 0)
          bombElegant("interval is non-positive for MHISTOGRAM element", NULL);
        if (!mhist->file1d && !mhist->file2dH && !mhist->file2dV &&
            !mhist->file2dL && !mhist->file4d && !mhist->file6d)
          bombElegant("No output request set to this mhistogram element. Use disable =1", NULL);

        if (!SDDS_InitializeInputFromSearchPath(&SDDSin, mhist->inputBinFile) ||
            SDDS_ReadPage(&SDDSin) <= 0 ||
            !(mhist->bins1d = (int32_t *)SDDS_GetColumnInLong(&SDDSin, "Bins_1D")) ||
            !(mhist->bins2d = (int32_t *)SDDS_GetColumnInLong(&SDDSin, "Bins_2D")) ||
            !(mhist->bins4d = (int32_t *)SDDS_GetColumnInLong(&SDDSin, "Bins_4D")) ||
            !(mhist->bins6d = (int32_t *)SDDS_GetColumnInLong(&SDDSin, "Bins_6D"))) {
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        if (!SDDS_Terminate(&SDDSin))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

        if (mhist->file1d) {
          if (mhist->bins1d[0] <= 2 && mhist->bins1d[1] <= 2 && mhist->bins1d[2] <= 2 &&
              mhist->bins1d[3] <= 2 && mhist->bins1d[4] <= 2 && mhist->bins1d[5] <= 2)
            bombElegant("All 1D bin's value less than 2", NULL);
          mhist->file1d = compose_filename_occurence(mhist->file1d, run->rootname, occurence);
        }
        if (mhist->file2dH) {
          if (mhist->bins2d[0] <= 2 || mhist->bins2d[1] <= 2)
            bombElegant("2D x-x' bin's value less than 2", NULL);
          mhist->file2dH = compose_filename_occurence(mhist->file2dH, run->rootname, occurence);
        }
        if (mhist->file2dV) {
          if (mhist->bins2d[2] <= 2 || mhist->bins2d[3] <= 2)
            bombElegant("2D y-y' bin's value less than 2", NULL);
          mhist->file2dV = compose_filename_occurence(mhist->file2dV, run->rootname, occurence);
        }
        if (mhist->file2dL) {
          if (mhist->bins2d[4] <= 2 || mhist->bins2d[5] <= 2)
            bombElegant("2D dt-dp bin's value less than 2", NULL);
          mhist->file2dL = compose_filename_occurence(mhist->file2dL, run->rootname, occurence);
        }
        if (mhist->file4d) {
          if (mhist->bins4d[0] <= 2 || mhist->bins4d[1] <= 2 ||
              mhist->bins4d[2] <= 2 || mhist->bins4d[3] <= 2)
            bombElegant("4D x-x'-y-y' bin's value less than 2", NULL);
          mhist->file4d = compose_filename_occurence(mhist->file4d, run->rootname, occurence);
        }
        if (mhist->file6d) {
          if (mhist->bins6d[0] <= 2 || mhist->bins6d[1] <= 2 || mhist->bins6d[2] <= 2 ||
              mhist->bins6d[3] <= 2 || mhist->bins6d[4] <= 2 || mhist->bins6d[5] <= 2)
            bombElegant("6D x-x'-y-y'-dt-dp bin's value less than 2", NULL);
          if (mhist->lumped)
            mhist->file6d = compose_filename_occurence(mhist->file6d, run->rootname, 0);
          else
            mhist->file6d = compose_filename_occurence(mhist->file6d, run->rootname, occurence);
        }

        mhist->initialized = 1;
        mhist->count = 0;

        return;
      }

#define MHISTOGRAM_TABLE_PARAMETERS 12
      static SDDS_DEFINITION mhist_table_para[MHISTOGRAM_TABLE_PARAMETERS] = {
        {"Step", "&parameter name=Step, symbol=\"m$be$nc\", description=\"Simulation step\", type=long &end"},
        {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", description=\"Reference beta*gamma\", type=double &end"},
        {"Charge", "&parameter name=Charge, units=\"C\", description=\"Beam charge\", type=double &end"},
        {"Particles", "&parameter name=Particles, description=\"Number of particles\",  type=long &end"},
        {"Pass", "&parameter name=Pass, type=long &end"},
        {"PassLength", "&parameter name=PassLength, units=\"m\", type=double &end"},
        {"PassCentralTime", "&parameter name=PassCentralTime, units=\"s\", type=double &end"},
        {"s", "&parameter name=s, units=\"m\", description=\"Location from beginning of beamline\", type=double &end"},
        {"ElementName", "&parameter name=ElementName, description=\"Previous element name\", type=string &end"},
        {"ElementOccurence", "&parameter name=ElementOccurence, description=\"Previous element's occurence\", format_string=%6ld, type=long, &end"},
        {"ElementType", "&parameter name=ElementType, description=\"Previous element-type name\", format_string=%10s, type=string, &end"},
        {"Normalize", "&parameter name=Normalize, description=\"If the table is normalized?\", type=long &end"},
      };
      static void *mhist_table_paraValue[MHISTOGRAM_TABLE_PARAMETERS];

      void mhist_table(ELEMENT_LIST *eptr0, ELEMENT_LIST *eptr, long step, long pass, double **coord, long np,
                       double Po, double length, double charge, double z) {
        char *Name[6] = {"x", "xp", "y", "yp", "dt", "delta"};
        char *Units[6] = {"m", "", "m", "", "s", ""};
        double Min[6], Max[6], c0[6], t0;
        long i, j;
        double part[6], P, beta;
        MHISTOGRAM *mhist;

        if (eptr0)
          mhist = (MHISTOGRAM *)eptr0->p_elem;
        else
          mhist = (MHISTOGRAM *)eptr->p_elem;

        t0 = pass * length * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
        mhist_table_paraValue[0] = (void *)(&step);
        mhist_table_paraValue[1] = (void *)(&Po);
        mhist_table_paraValue[2] = (void *)(&charge);
        mhist_table_paraValue[3] = (void *)(&np);
        mhist_table_paraValue[4] = (void *)(&pass);
        mhist_table_paraValue[5] = (void *)(&length);
        mhist_table_paraValue[6] = (void *)(&t0);
        mhist_table_paraValue[7] = (void *)(&z);
        mhist_table_paraValue[8] = (void *)(&eptr->pred->name);
        mhist_table_paraValue[9] = (void *)(&eptr->pred->occurence);
        mhist_table_paraValue[10] = (void *)(&entity_name[eptr->pred->type]);
        mhist_table_paraValue[11] = (void *)(&mhist->normalize);

        findMinMax(coord, np, Min, Max, c0, Po);
        Min[4] -= c0[4];
        Max[4] -= c0[4];

        if (mhist->file1d)
          mhist->x1d = chbook1m(Name, Units, Min, Max, mhist->bins1d, 6);

        if (mhist->file2dH)
          mhist->x2d = chbookn(Name, Units, 2, Min, Max, mhist->bins2d, 0);
        if (mhist->file2dV)
          mhist->y2d = chbookn(Name, Units, 2, Min, Max, mhist->bins2d, 2);
        if (mhist->file2dL)
          mhist->z2d = chbookn(Name, Units, 2, Min, Max, mhist->bins2d, 4);
        if (mhist->file4d)
          mhist->Tr4d = chbookn(Name, Units, 4, Min, Max, mhist->bins4d, 0);
        if (mhist->file6d)
          mhist->full6d = chbookn(Name, Units, 6, Min, Max, mhist->bins6d, 0);

        for (i = 0; i < np; i++) {
          for (j = 0; j < 6; j++) {
            if (j == 4) {
              P = Po * (1 + coord[i][5]);
              beta = P / sqrt(P * P + 1);
              part[4] = coord[i][4] / (beta * c_mks) - c0[4];
            } else
              part[j] = coord[i][j];
          }

          if (mhist->x1d)
            chfill1m(mhist->x1d, part, 1, mhist->bins1d, 6);
          if (mhist->x2d)
            chfilln(mhist->x2d, part, 1, 0);
          if (mhist->y2d)
            chfilln(mhist->y2d, part, 1, 2);
          if (mhist->z2d)
            chfilln(mhist->z2d, part, 1, 4);
          if (mhist->Tr4d)
            chfilln(mhist->Tr4d, part, 1, 0);
          if (mhist->full6d)
            chfilln(mhist->full6d, part, 1, 0);
        }

        if (mhist->x1d) {
          chprint1m(mhist->x1d, mhist->file1d, "One dimentional distribution", mhist_table_para,
                    mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbook1m(mhist->x1d);
        }
        if (mhist->x2d) {
          chprintn(mhist->x2d, mhist->file2dH, "x-xp distribution", mhist_table_para,
                   mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbookn(mhist->x2d);
        }
        if (mhist->y2d) {
          chprintn(mhist->y2d, mhist->file2dV, "y-yp distribution", mhist_table_para,
                   mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbookn(mhist->y2d);
        }
        if (mhist->z2d) {
          chprintn(mhist->z2d, mhist->file2dL, "dt-delta distribution", mhist_table_para,
                   mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbookn(mhist->z2d);
        }
        if (mhist->Tr4d) {
          chprintn(mhist->Tr4d, mhist->file4d, "x-xp-y-yp distribution", mhist_table_para,
                   mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbookn(mhist->Tr4d);
        }
        if (mhist->full6d) {
          chprintn(mhist->full6d, mhist->file6d, "full distribution", mhist_table_para,
                   mhist_table_paraValue, MHISTOGRAM_TABLE_PARAMETERS, mhist->normalize, 0, mhist->count);
          free_hbookn(mhist->full6d);
        }

        mhist->count++;

        return;
      }

      void findMinMax(double **coord, long np, double *min, double *max, double *c0, double Po) {
        long i, j;
        double P, beta, time;

        for (j = 0; j < 6; j++) {
          max[j] = -(min[j] = DBL_MAX);
          c0[j] = 0;
          for (i = 0; i < np; i++) {
            if (j == 4) {
              P = Po * (1 + coord[i][5]);
              beta = P / sqrt(P * P + 1);
              time = coord[i][4] / (beta * c_mks);
              c0[j] += time;
              if (min[j] > time)
                min[j] = time;
              if (max[j] < time)
                max[j] = time;
            } else {
              c0[j] += coord[i][j];
              if (min[j] > coord[i][j])
                min[j] = coord[i][j];
              if (max[j] < coord[i][j])
                max[j] = coord[i][j];
            }
          }
          c0[j] /= (double)np;
        }
        return;
      }

      void field_table_tracking(double **particle, long np, FTABLE *ftable, double Po, RUN *run) {
        long ip, ik, nKicks, debug = ftable->verbose;
        double *coord, p0, factor;
        double xyz[3], p[3], B[3], BA, pA, **A;
        /* double pz0; */
        double rho, theta0, theta1, theta2, theta, tm_a, tm_b, tm_c;
        double step, eomc, s_location;
        char *rootname;

        static SDDS_TABLE test_output;
        static long first_time = 1;
        static FILE *fpdebug = NULL;

        if (first_time && debug) {
          rootname = compose_filename("%s.phase", run->rootname);
          SDDS_PhaseSpaceSetup(&test_output, rootname, SDDS_BINARY, 1, "output phase space", run->runfile,
                               run->lattice, "setup_output");
          if (debug > 1) {
            fpdebug = fopen("ftable.sdds", "w");
            fprintf(fpdebug, "SDDS1\n&column name=ik type=long &end\n");
            fprintf(fpdebug, "&column name=after type=short &end\n");
            fprintf(fpdebug, "&column name=ip type=long &end\n");
            fprintf(fpdebug, "&column name=x type=double &end\n");
            fprintf(fpdebug, "&column name=y type=double &end\n");
            fprintf(fpdebug, "&column name=z type=double &end\n");
            fprintf(fpdebug, "&column name=Bx type=double &end\n");
            fprintf(fpdebug, "&column name=By type=double &end\n");
            fprintf(fpdebug, "&column name=Bz type=double &end\n");
            fprintf(fpdebug, "&data mode=ascii no_row_counts=1 &end\n");
          }
          first_time = 0;
        }

        if ((nKicks = ftable->nKicks) < 1)
          bombElegant("N_KICKS must be >=1 for FTABLE", NULL);
        if (!ftable->initialized)
          bombElegant("Initialize FTABLE: This shouldn't happen.", NULL);
        step = ftable->length / nKicks;

        /* convert coordinate frame from local to ftable element frame. Before misalignment.*/
        if (debug)
          dump_phase_space(&test_output, particle, np, -2, Po, 0, 0);
        if (ftable->e1 || ftable->l1)
          ftable_frame_converter(particle, np, ftable, 0);
        if (debug)
          dump_phase_space(&test_output, particle, np, -1, Po, 0, 0);

        if (ftable->dx || ftable->dy || ftable->dz)
          offsetBeamCoordinatesForMisalignment(particle, np, ftable->dx, ftable->dy, ftable->dz);
        if (ftable->tilt)
          rotateBeamCoordinatesForMisalignment(particle, np, ftable->tilt);
        if (debug)
          dump_phase_space(&test_output, particle, np, 0, Po, 0, 0);

        s_location = step / 2.;
        eomc = -particleCharge / particleMass / c_mks;
        A = (double **)czarray_2d(sizeof(double), 3, 3);
        for (ik = 0; ik < nKicks; ik++) {
          for (ip = 0; ip < np; ip++) {
            /* 1. get particle's coordinates */
            coord = particle[ip];
            factor = sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            p0 = (1. + coord[5]) * Po;
            p[2] = p0 / factor;
            p[0] = coord[1] * p[2];
            p[1] = coord[3] * p[2];

            /* 2. get field at the middle point */
            xyz[0] = coord[0] + coord[1] * step / 2.0;
            xyz[1] = coord[2] + coord[3] * step / 2.0;
            xyz[2] = s_location;
            interpolateFTable(B, xyz, ftable);
            if (fpdebug)
              fprintf(fpdebug, "%ld 0 %ld %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", ik, ip, xyz[0], xyz[1], xyz[2], B[0], B[1], B[2]);
            BA = sqrt(sqr(B[0]) + sqr(B[1]) + sqr(B[2]));
            /* 3. calculate the rotation matrix */
            A[0][0] = -(p[1] * B[2] - p[2] * B[1]);
            A[0][1] = -(p[2] * B[0] - p[0] * B[2]);
            A[0][2] = -(p[0] * B[1] - p[1] * B[0]);
            pA = sqrt(sqr(A[0][0]) + sqr(A[0][1]) + sqr(A[0][2]));
            /* When field not equal to zero or not parallel to the particles motion */
            if (BA > ftable->threshold && pA) {
              A[0][0] /= pA;
              A[0][1] /= pA;
              A[0][2] /= pA;
              A[1][0] = B[0] / BA;
              A[1][1] = B[1] / BA;
              A[1][2] = B[2] / BA;
              A[2][0] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
              A[2][1] = A[0][2] * A[1][0] - A[0][0] * A[1][2];
              A[2][2] = A[0][0] * A[1][1] - A[0][1] * A[1][0];

              /* 4. rotate coordinates from (x,y,z) to (u,v,w) with u point to BxP, v point to B */
              /* pz0 = p[2]; */
              rotate_coordinate(A, p, 0);
              if (p[2] < 0)
                bombElegant("Table function doesn't support particle going backward", NULL);
              rotate_coordinate(A, B, 0);

              /* 5. apply kick */
              rho = p[2] / (eomc * B[1]);
              theta0 = theta1 = theta2 = 0.;
              if (A[2][2]) {
                tm_a = 3.0 * A[0][2] / A[2][2];
                tm_b = -6.0 * A[1][2] * p[1] / p[2] / A[2][2] - 6.0;
                tm_c = 6.0 * step / rho / A[2][2];
#ifdef USE_GSL
                gsl_poly_solve_cubic(tm_a, tm_b, tm_c, &theta0, &theta1, &theta2);
#else
                bombElegant("gsl_poly_solve_cubic function is not available becuase this version of elegant was not built against the gsl library", NULL);
#endif
              } else if (A[0][2]) {
                tm_a = A[1][2] * p[1] / p[2] + A[2][2];
                theta0 = (tm_a - sqrt(sqr(tm_a) - 2. * A[0][2] * step / rho)) / A[0][2];
                theta1 = (tm_a + sqrt(sqr(tm_a) - 2. * A[0][2] * step / rho)) / A[0][2];
              } else {
                tm_a = A[1][2] * p[1] / p[2] + A[2][2];
                theta0 = step / rho / tm_a;
              }
              theta = choose_theta(rho, theta0, theta1, theta2);
              if (fpdebug)
                fprintf(fpdebug, "%ld 1 %ld %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n", ik, ip, xyz[0], xyz[1], xyz[2], B[0], B[1], B[2]);

              p[0] = -p[2] * sin(theta);
              p[2] *= cos(theta);
              xyz[0] = rho * (cos(theta) - 1);
              xyz[1] = (p[1] / p[2]) * rho * theta;
              xyz[2] = rho * sin(theta);

              /* 6. rotate back to (x,y,z) */
              rotate_coordinate(A, xyz, 1);
              rotate_coordinate(A, p, 1);
              coord[0] += xyz[0];
              coord[2] += xyz[1];
              coord[4] += sqrt(sqr(rho * theta) + sqr(xyz[1]));
              coord[1] = p[0] / p[2];
              coord[3] = p[1] / p[2];
            } else {
              coord[0] += coord[1] * step;
              coord[2] += coord[3] * step;
              coord[4] += step * factor;
            }
          }
          s_location += step;
          if (debug)
            dump_phase_space(&test_output, particle, np, ik + 1, Po, 0, 0);
        }

        free_czarray_2d((void **)A, 3, 3);

        if (ftable->tilt)
          rotateBeamCoordinatesForMisalignment(particle, np, -ftable->tilt);
        if (ftable->dx || ftable->dy || ftable->dz)
          offsetBeamCoordinatesForMisalignment(particle, np, -ftable->dx, -ftable->dy, -ftable->dz);
        if (debug)
          dump_phase_space(&test_output, particle, np, nKicks + 1, Po, 0, 0);

        /* convert coordinate frame from ftable element frame to local frame. After misalignment.*/
        if (ftable->e2 || ftable->l2)
          ftable_frame_converter(particle, np, ftable, 1);
        if (debug)
          dump_phase_space(&test_output, particle, np, nKicks + 2, Po, 0, 0);

        return;
      }

      /* 0: entrance; 1: exit */
      void ftable_frame_converter(double **particle, long np, FTABLE *ftable, long entrance_exit) {
        double *coord, x0, xp0, y0, yp0;
        double s, c, temp, dx, length;
        long ip, entrance = 0, exit = 1;

        if (entrance_exit == entrance) {
          if (!ftable->e1) {
            exactDrift(particle, np, -ftable->l1);
            return;
          }

          /* rotate to ftable element frame */
          s = sin(ftable->e1);
          c = cos(ftable->e1);
          for (ip = 0; ip < np; ip++) {
            coord = particle[ip];
            x0 = coord[0];
            xp0 = coord[1];
            y0 = coord[2];
            yp0 = coord[3];
            length = ftable->l1 - s * x0;
            temp = c - s * xp0;
            if (temp == 0)
              bombElegant("ftable_frame_converter: Particle will never get into Element.", NULL);

            coord[1] = (c * xp0 + s) / temp;
            coord[3] = yp0 / temp;
            coord[4] -= length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            coord[0] = c * x0 - coord[1] * length;
            coord[2] = y0 - coord[3] * length;
          }
        }

        if (entrance_exit == exit) {
          if (!ftable->e2) {
            exactDrift(particle, np, -ftable->l2);
            return;
          }

          /* rotate to normal local frame */
          s = sin(ftable->e2);
          c = cos(ftable->e2);
          dx = ftable->l2 * s;
          for (ip = 0; ip < np; ip++) {
            coord = particle[ip];
            x0 = coord[0];
            xp0 = coord[1];
            y0 = coord[2];
            yp0 = coord[3];
            length = c * ftable->l2 - s * x0;
            temp = c - s * xp0;
            if (temp == 0)
              bombElegant("ftable_frame_converter: Particle will never get into Element.", NULL);

            coord[1] = (c * xp0 + s) / temp;
            coord[3] = yp0 / temp;
            coord[4] -= length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            coord[0] = c * x0 + dx - coord[1] * length;
            coord[2] = y0 - coord[3] * length;
          }
        }

        return;
      }

      void interpolateFTable(double *B, double *xyz, FTABLE *ftable) {
        double dummy[3];
        /*
          B[0] = interpolate_bookn(ftable->Bx, dummy, xyz, 0, 0, 0, 1, ftable->verbose);
          B[1] = interpolate_bookn(ftable->By, dummy, xyz, 0, 0, 0, 1, ftable->verbose);
          B[2] = interpolate_bookn(ftable->Bz, dummy, xyz, 0, 0, 0, 1, ftable->verbose);
        */
        B[0] = ftable->factor * interpolate_bookn(ftable->Bx, dummy, xyz, 0, 0, 0, 1, 0);
        B[1] = ftable->factor * interpolate_bookn(ftable->By, dummy, xyz, 0, 0, 0, 1, 0);
        B[2] = ftable->factor * interpolate_bookn(ftable->Bz, dummy, xyz, 0, 0, 0, 1, 0);

        return;
      }

      short determineP0ChangeBlocking(ELEMENT_LIST *eptr) {
        if (!(entity_description[eptr->type].flags & MAY_CHANGE_ENERGY))
          return 1;
        switch (eptr->type) {
        case T_RFCA:
          return !((RFCA *)eptr->p_elem)->change_p0;
          break;
        case T_RFTMEZ0:
          return !((RFTMEZ0 *)eptr->p_elem)->change_p0;
          break;
        case T_TWLA:
          return !((TW_LINAC *)eptr->p_elem)->change_p0;
          break;
        case T_WAKE:
          return !((WAKE *)eptr->p_elem)->change_p0;
          break;
        case T_CORGPIPE:
          return !((CORGPIPE *)eptr->p_elem)->change_p0;
          break;
        case T_RFCW:
          return !((RFCW *)eptr->p_elem)->change_p0;
          break;
        }
        return 0;
      }

      void convertToCanonicalCoordinates(double **coord, long np, double p0, long includeTimeCoordinate) {
        long ip;
        double factor;
        /* double p, beta; */
        for (ip = 0; ip < np; ip++) {
          factor = (1 + coord[ip][5]) / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
          coord[ip][1] *= factor;
          coord[ip][3] *= factor;
          if (includeTimeCoordinate)
            coord[ip][4] *= -1;
        }
      }

      void convertFromCanonicalCoordinates(double **coord, long np, double p0, long includeTimeCoordinate) {
        long ip;
        double px, py, factor, delta;
        /* double p, beta; */
        for (ip = 0; ip < np; ip++) {
          px = coord[ip][1];
          py = coord[ip][3];
          delta = coord[ip][5];
          factor = 1 / sqrt(sqr(1 + delta) - sqr(px) - sqr(py));
          coord[ip][1] *= factor;
          coord[ip][3] *= factor;
          if (includeTimeCoordinate)
            coord[ip][4] *= -1;
        }
      }

      static void checkBeamStructure(BEAM *beam) {
        long i;
        char bad = 0;
        if (beam->original && beam->n_saved) {
          for (i = 1; i < beam->n_original; i++) {
            if (beam->original[i] < beam->original[i - 1]) {
              printf("Error in pointer order for i=%ld in beam->original\n", i);
              bad = 1;
            }
          }
        }
        if (beam->particle) {
          for (i = 1; i < beam->n_particle; i++) {
            if (beam->particle[i] < beam->particle[i - 1]) {
              printf("Error in pointer order for i=%ld in beam->particle\n", i);
              bad = 1;
            }
          }
        }
        if (beam->accepted) {
          for (i = 1; i < beam->n_particle; i++) {
            if (beam->accepted[i] < beam->accepted[i - 1]) {
              printf("Error in pointer order for i=%ld in beam->accepted\n", i);
              bad = 1;
            }
          }
        }
        if (bad) {
          fflush(stdout);
#if USE_MPI
          mpiAbort = MPI_ABORT_POINTER_ISSUE;
          return;
#else
          exit(1);
#endif
        }

        if (beam->particle) {
          for (i = 0; i < beam->n_particle; i++) {
            if (beam->particle[i][6] <= 0) {
              printf("Non-positive particleID %le for i=%ld in beam->particle\n", beam->particle[i][6], i);
              bad = 1;
            }
          }
        }
        if (beam->accepted) {
          for (i = 0; i < beam->n_particle; i++) {
            if (beam->accepted[i][6] <= 0) {
              printf("Non-positive particleID %le for i=%ld in beam->accepted\n", beam->particle[i][6], i);
              bad = 1;
            }
          }
        }
        if (bad) {
          fflush(stdout);
#if USE_MPI
          mpiAbort = MPI_ABORT_BAD_PARTICLE_ID;
          return;
#else
          exit(1);
#endif
        }
      }

      /* Put this here so routines in this file can see it. Putting it in tfeedback.cc
       * causes it to be invisible.
       */

      void flushTransverseFeedbackDriverFiles(TFBDRIVER *tfbd) {
#if USE_MPI
        if (myid != 0)
          return;
#endif
        if (tfbd->initialized && !(tfbd->dataWritten)) {
          if (!SDDS_WritePage(tfbd->SDDSout)) {
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb("problem writing data for TFBDRIVER output file (flushTransverseFeedbackDriverFiles)");
          }
          tfbd->dataWritten = 1;
        }
        tfbd->outputIndex = 0;
      }

      void applyBeamBeamKicks(double **part, long np, BEAMBEAM *bb, double P0) {
        long ip;
        short code;
        double kick[2], qx, qy, qz, delta, denom;
        if (bb->size[0] <= 0 || bb->size[1] <= 0)
          bombElegant("XSIZE and YSIZE must be positive for BEAMBEAM element", NULL);
        code = match_string(bb->distribution, beamBeamDistributionOption, N_BEAM_BEAM_DISTRIBUTIONS, 0);
        if (code == GAUSSIAN_BEAM_BEAM) {
          for (ip = 0; ip < np; ip++) {
            denom = sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3]));
            delta = part[ip][5];
            qx = part[ip][1] * (1 + delta) / denom;
            qy = part[ip][3] * (1 + delta) / denom;
            qz = sqrt(sqr(1 + delta) - sqr(qx) - sqr(qy));
            /* compute velocity changes in m/s */
            gaussianBeamKick(part[ip], bb->centroid, bb->size, 0, kick, bb->charge, particleMass, particleCharge * particleRelSign / e_mks);
            qx += kick[0] / (P0 * c_mks);
            qy += kick[1] / (P0 * c_mks);
            delta = sqrt(qx * qx + qy * qy + qz * qz) - 1;
            part[ip][1] = qx / qz;
            part[ip][3] = qy / qz;
            part[ip][5] = delta;
          }
        } else if (code == UNIFORM_ELLIPSOIDAL_BEAM_BEAM) {
          for (ip = 0; ip < np; ip++)
            ellipsoidalBeamKick(part[ip], P0, particleMass, -particleRelSign * particleCharge, bb->centroid, bb->size, bb->charge, 0);
        } else if (code == PARABOLIC_ELLIPSOIDAL_BEAM_BEAM) {
          for (ip = 0; ip < np; ip++)
            ellipsoidalBeamKick(part[ip], P0, particleMass, -particleRelSign * particleCharge, bb->centroid, bb->size, bb->charge, 1);
        } else
          bombElegantVA("BEAMBEAM distribution type %s not recognized", bb->distribution);
      }

      void checkRFCAChangeTConflicts(LINE_LIST *beamline) {
        ELEMENT_LIST *eptr;
        short hasChangeT = 0;
        long hasTimeDependence = 0;

        eptr = beamline->elem;

        while (eptr) {
          switch (eptr->type) {
          case T_RFCA:
            if (((RFCA *)(eptr->p_elem))->change_t)
              hasChangeT = 1;
            break;
          case T_RFCW:
            if (((RFCW *)(eptr->p_elem))->change_t)
              hasChangeT = 1;
            break;
          case T_CPICKUP:
          case T_CKICKER:
          case T_FRFMODE:
          case T_FTRFMODE:
          case T_IONEFFECTS:
          case T_KICKER:
          case T_LRWAKE:
          case T_MKICKER:
          case T_MRFDF:
          case T_RAMPP:
          case T_RAMPRF:
          case T_RFDF:
          case T_RFMODE:
          case T_RFTM110:
          case T_RFTMEZ0:
          case T_RMDF:
          case T_SHRFDF:
          case T_TFBDRIVER:
          case T_TFBPICKUP:
          case T_TRFMODE:
          case T_TMCF:
          case T_TWLA:
          case T_TWMTA:
          case T_TWPL:
            hasTimeDependence += 1;
            break;
          }
          eptr = eptr->succ;
        }
        if (hasChangeT && hasTimeDependence) {
          char buffer[16834];
          snprintf(buffer, 16384, "RFCA or RFCW elements with CHANGE_T=1 may give wrong results because of %ld time-dependent elements",
                   hasTimeDependence);
          printWarning("CHANGE_T=1 conflicts, potentially severely, with other time-dependent elements", buffer);
        }
      }
