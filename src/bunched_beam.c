/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: bunched_beam.c
 * purpose: Do tracking for bunched gaussian beams.
 *          See file bunched_beam.nl for input parameters.
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
//#include "mdbsun.h"
#include "track.h"
#include "bunched_beam.h"

char *beam_type[N_BEAM_TYPES] = {
  "gaussian",
  "hard-edge",
  "uniform-ellipse",
  "shell",
  "dynamic-aperture",
  "line",
  "halo(gaussian)",
};

static TRANSVERSE x_plane, y_plane;
static LONGITUDINAL longit;
static long x_beam_type, y_beam_type, longit_beam_type;

static SDDS_TABLE SDDS_bunch;
static long SDDS_bunch_initialized;
static long beamRepeatSeed, bunchGenerated, firstIsFiducial, beamCounter;
static long haltonID[6], haltonOpt;

void fill_transverse_structure(TRANSVERSE *trans, double emit, double beta,
                               double alpha, double eta, double etap, long xbeam_type, double cutoff,
                               double *xcentroid);
void fill_longitudinal_structure(LONGITUDINAL *xlongit, double xsigma_dp,
                                 double xsigma_s, double xcoupling_angle,
                                 double xemit, double xbeta, double xalpha, double xchirp,
                                 long xbeam_type, double xcutoff,
                                 double *xcentroid);
long generateBunchForMoments(double **particle, int64_t np, long symmetrize,
                             long *haltonID, long haltonOpt, double cutoff);
void finish_bunched_beam_setup(BEAM *beam, RUN *run, VARY *control, ERRORVAL *errcon,
                               OPTIM_VARIABLES *optim, OUTPUT_FILES *output, LINE_LIST *beamline,
                               long n_elements, long save_original);

void setup_bunched_beam(
  BEAM *beam,
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  OUTPUT_FILES *output,
  LINE_LIST *beamline,
  long n_elements,
  long save_original) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&bunched_beam, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (Po <= 0)
    Po = run->p_central;
  if (use_twiss_command_values && use_moments_output_values)
    bombElegant("Only one of use_twiss_command_values and use_moments_output_values may be nonzero", NULL);

  if (use_twiss_command_values) {
    TWISS twiss;
    long mode;

    if (!get_twiss_mode(&mode, &twiss) || mode != 0)
      bombElegant("use_twiss_command_values invalid unless twiss_output command given previously with matched=0 or in action mode with matched=1.",
                  NULL);
    beta_x = twiss.betax;
    alpha_x = twiss.alphax;
    eta_x = twiss.etax;
    etap_x = twiss.etapx;
    beta_y = twiss.betay;
    alpha_y = twiss.alphay;
    eta_y = twiss.etay;
    etap_y = twiss.etapy;
  }
  if (use_moments_output_values && save_original)
    bombElegant("&correct settings are incompatible with using results of moments_output for generating the beam. Set start_from_centroid and track_before_and_after to 0.", NULL);

  if (echoNamelists)
    print_namelist(stdout, &bunched_beam);
#if USE_MPI
  if (multiply_np_by_cores && n_processors > 1)
    n_particles_per_bunch *= (n_processors - 1);
#endif

  finish_bunched_beam_setup(beam, run, control, errcon, optim, output, beamline, n_elements, save_original);
}

void finish_bunched_beam_setup(
  BEAM *beam,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  OUTPUT_FILES *output,
  LINE_LIST *beamline,
  long n_elements,
  long save_original) {
  long i, offset;
  setFiducializationBunch(-1, -1);

  /* check for validity of namelist inputs */
  if (emit_nx && !emit_x)
    emit_x = emit_nx / Po;
  if (emit_ny && !emit_y)
    emit_y = emit_ny / Po;
  if (n_particles_per_bunch <= 0)
    bombElegant("n_particles_per_bunch is invalid", NULL);
#if SDDS_MPI_IO
  beam->n_original_total = beam->n_to_track_total = n_particles_per_bunch; /* record the total number of particles being tracked */
  if (!independentRunPerRank) {
    if (n_particles_per_bunch == 1) /* a special case for single particle tracking */
      distributedBeam = 0;
    else {
      distributedBeam = 1;
      if (isSlave) {
        long work_processors = n_processors - 1;
        long my_nToTrack = n_particles_per_bunch / work_processors;
        if (myid <= (n_particles_per_bunch % work_processors))
          my_nToTrack++;
        n_particles_per_bunch = my_nToTrack;
      } else {
        n_particles_per_bunch = 0;
      }
    }
  }
#endif
  if (!use_moments_output_values) {
    if (emit_x < 0 || beta_x <= 0)
      bombElegant("emit_x<=0 or beta_x<=0", NULL);
    if (emit_y < 0 || beta_y <= 0)
      bombElegant("emit_y<=0 or beta_y<=0", NULL);
    if (sigma_dp < 0 || sigma_s < 0)
      bombElegant("sigma_dp<0 or sigma_s<0", NULL);
    if (fabs(dp_s_coupling) > 1)
      bombElegant("|dp_s_coupling| > 1", NULL);
    if (emit_z < 0)
      bombElegant("emit_z < 0", NULL);
    if (beta_z < 0)
      bombElegant("beta_z < 0", NULL);
    if (emit_z && (sigma_dp || sigma_s))
      bombElegant("give emit_z or both sigma_dp and sigma_s", NULL);
    if (emit_z && beta_z <= 0)
      bombElegant("give beta_z with emit_z", NULL);
    if (((alpha_z ? 1 : 0) + (dp_s_coupling ? 1 : 0) + (momentum_chirp ? 1 : 0)) > 1)
      bombElegant("give only one of alpha_z, dp_s_coupling, or momentum_chirp", NULL);
    if (emit_z && dp_s_coupling)
      bombElegant("give alpha_z not dp_s_coupling with emit_z", NULL);
  }

  if (!distribution_type[0] ||
      (x_beam_type = match_string(distribution_type[0], beam_type,
                                  N_BEAM_TYPES, 0)) < 0)
    bombElegant("missing or unknown x distribution type", NULL);
  if (!distribution_type[1] ||
      (y_beam_type = match_string(distribution_type[1], beam_type,
                                  N_BEAM_TYPES, 0)) < 0)
    bombElegant("missing or unknown y distribution type", NULL);
  if (!distribution_type[2] ||
      (longit_beam_type = match_string(distribution_type[2], beam_type,
                                       N_BEAM_TYPES, 0)) < 0)
    bombElegant("missing or unknown longitudinal distribution type", NULL);
  if (use_moments_output_values) {
    if (x_beam_type != GAUSSIAN_BEAM || y_beam_type != GAUSSIAN_BEAM || longit_beam_type != GAUSSIAN_BEAM)
      printWarning("bunched_beam: setting distribution types to gaussian since use_moments_output_values is nonzero.", NULL);
    x_beam_type = y_beam_type = longit_beam_type = GAUSSIAN_BEAM;
  }

  if (distribution_cutoff[0] < 0)
    bombElegant("x distribution cutoff < 0", NULL);
  if (distribution_cutoff[1] < 0)
    bombElegant("y distribution cutoff < 0", NULL);
  if (distribution_cutoff[2] < 0)
    bombElegant("longitudinal distribution cutoff < 0", NULL);
  if (use_moments_output_values) {
    long i, warned = 0;
    double maxCutoff = 0;
    for (i = 0; i < 3; i++)
      if (distribution_cutoff[i] > maxCutoff)
        maxCutoff = distribution_cutoff[i];
    for (i = 0; i < 3; i++)
      if (distribution_cutoff[i] < maxCutoff) {
        distribution_cutoff[i] = maxCutoff;
        if (!warned) {
          char buffer[1024];
          snprintf(buffer, 1024,
                   "set bunched_beam distribution cutoff values set to %le since use_moments_output_values is nonzero", maxCutoff);
          printf(buffer, NULL);
        }
        warned = 1;
      }
  }

  if ((enforce_rms_values[0] || enforce_rms_values[1] || enforce_rms_values[2]) && n_particles_per_bunch % 4)
    fputs("Note: you will get better results for enforcing RMS values if n_particles_per_bunch is divisible by 4.\n", stdout);

  if (!use_moments_output_values && matched_to_cell) {
    unsigned long flags;
    flags = beamline->flags;
    if (!(control->cell = get_beamline(NULL, matched_to_cell, run->p_central, 0, 0, NULL, NULL)))
      bombElegant("unable to find beamline definition for cell", NULL);
    if (control->cell == beamline)
      beamline->flags = flags;
  } else
    control->cell = NULL;

  if (bunch) {
    /* prepare dump of input particles */
    bunch = compose_filename(bunch, run->rootname);
    SDDS_PhaseSpaceSetup(&SDDS_bunch, bunch, SDDS_BINARY, 1, "bunched-beam phase space", run->runfile,
                         run->lattice, "setup_bunched_beam");
    SDDS_bunch_initialized = 1;
  }

  /* set up structures for creating initial particle distributions */
  if (!use_moments_output_values) {
    if (!control->cell) {
      fill_transverse_structure(&x_plane, emit_x, beta_x, alpha_x, eta_x, etap_x,
                                x_beam_type, distribution_cutoff[0], centroid);
      fill_transverse_structure(&y_plane, emit_y, beta_y, alpha_y, eta_y, etap_y,
                                y_beam_type, distribution_cutoff[1], centroid + 2);
    }
    fill_longitudinal_structure(&longit,
                                sigma_dp, sigma_s, dp_s_coupling,
                                emit_z, beta_z, alpha_z, momentum_chirp,
                                longit_beam_type, distribution_cutoff[2], centroid + 4);
  }
  
  save_initial_coordinates = save_original || save_initial_coordinates;
  if (n_particles_per_bunch == 1)
    one_random_bunch = 1;
  if (!one_random_bunch)
    save_initial_coordinates = save_original;
  if (isSlave || !distributedBeam) {
    beam->particle = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
  }
#if SDDS_MPI_IO
  else {
    beam->particle = NULL;
  }
#endif
  if (isSlave) {
    if (!save_initial_coordinates)
      beam->original = beam->particle;
    else
      beam->original = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
  }
#if SDDS_MPI_IO
  else
    beam->original = NULL;
#endif
  if (isSlave) {
    if (run->acceptance)
      beam->accepted = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
    else
      beam->accepted = NULL;
  }
#if SDDS_MPI_IO
  else
    beam->accepted = NULL;
#endif
  beam->n_original = beam->n_to_track = beam->n_particle = n_particles_per_bunch;
  beam->n_accepted = beam->n_saved = 0;
  firstIsFiducial = first_is_fiducial;
  beamCounter = 0;
  beam->n_lost = 0;

  if (one_random_bunch) {
    /* make a seed for reinitializing the beam RN generator */
    beamRepeatSeed = 1e8 * random_4(1);
#if USE_MPI
    /* All processors will have same beamRepeatSeed after here. This will make
       it easy for the serial version to generate the same sequence */
    MPI_Bcast(&beamRepeatSeed, 1, MPI_LONG, 1, MPI_COMM_WORLD);
#endif
    random_4(-beamRepeatSeed);
  }
#if !USE_MPI
  else if ((run->random_sequence_No > 1) && (control->n_steps == 1)) { /* This part will take effect for regression test when random_sequence_No>1 */
    beamRepeatSeed = 1e8 * random_4(1);
    random_4(-beamRepeatSeed);
  }
#else
  else if (control->n_steps == 1) {
    beamRepeatSeed = 1e8 * random_4(1);
    MPI_Bcast(&beamRepeatSeed, 1, MPI_LONG, 1, MPI_COMM_WORLD);
  }
#endif
  bunchGenerated = 0;

  haltonOpt = optimized_halton;
  for (i = 0; i < 3; i++) {
    for (offset = 0; offset < 2; offset++) {
      haltonID[2 * i + offset] = 0;
      if (halton_sequence[i]) {
        if (halton_radix[2 * i + offset] < 0)
          bombElegant("Radix for Halton sequence can't be negative", NULL);
        if (haltonOpt) {
          if ((haltonID[2 * i + offset] = startModHaltonSequence(&halton_radix[2 * i + offset], 0.5)) < 0)
            bombElegant("problem starting Halton sequence", NULL);
        } else {
          if ((haltonID[2 * i + offset] = startHaltonSequence(&halton_radix[2 * i + offset], 0.5)) < 0)
            bombElegant("problem starting Halton sequence", NULL);
        }
        printf("Using radix %" PRId32 " for %s Halton sequence for coordinate %ld\n",
               halton_radix[2 * i + offset], (haltonOpt ? "optimized" : "original"), 2 * i + offset);
      }
    }
  }
  fflush(stdout);

  log_exit("setup_bunched_beam");
}

void setup_bunched_beam_moments(
  BEAM *beam,
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  OUTPUT_FILES *output,
  LINE_LIST *beamline,
  long n_elements,
  long save_original) {
  long i;
#include "bunched_beam2.h"

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&bunched_beam_moments, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (bunched_beam_moments_struct.Po <= 0)
    bunched_beam_moments_struct.Po = run->p_central;
  if (bunched_beam_moments_struct.use_moments_output_values && save_original)
    bombElegant("&correct settings are incompatible with using results of moments_output for generating the beam. Set start_from_centroid and track_before_and_after to 0.", NULL);

  if (echoNamelists)
    print_namelist(stdout, &bunched_beam_moments);
#if USE_MPI
  if (multiply_np_by_cores && n_processors > 1)
    n_particles_per_bunch *= (n_processors - 1);
#endif

  /* We'll be copying values to the variables for &bunched_beam, so we do a reset first */
  reset_namelist_values(&bunched_beam);

  if (spinCoordOffset) {
    long isp;
    double norm;
    for (isp=0; isp<3; isp++)
      spin[isp] = bunched_beam_moments_struct.spin[isp];
    norm = sqrt(sqr(spin[0])+sqr(spin[1])+sqr(spin[2]));
    for (isp=0; isp<3; isp++)
      spin[isp] /= norm;
  }

  if (bunched_beam_moments_struct.S1_beta < 0 || bunched_beam_moments_struct.S2_beta < 0 ||
      bunched_beam_moments_struct.S3_beta < 0 || bunched_beam_moments_struct.S4_beta < 0)
    bombElegant("S1_beta, S2_beta, S3_beta, and S4_beta must all be non-negative", NULL);
  if (bunched_beam_moments_struct.S5 < 0 || bunched_beam_moments_struct.S6 < 0)
    bombElegant("S5 and S6 must be non-negative", NULL);

  if ((emit_x = sqr(bunched_beam_moments_struct.S1_beta * bunched_beam_moments_struct.S2_beta) -
                sqr(bunched_beam_moments_struct.S12_beta)) < 0)
    bombElegant("implied horizontal emittance is invalid", NULL);
  if ((emit_x = sqrt(emit_x)) > 0) {
    beta_x = sqr(bunched_beam_moments_struct.S1_beta) / emit_x;
    alpha_x = -bunched_beam_moments_struct.S12_beta / emit_x;
  }

  if ((emit_y = sqr(bunched_beam_moments_struct.S3_beta * bunched_beam_moments_struct.S4_beta) -
                sqr(bunched_beam_moments_struct.S34_beta)) < 0)
    bombElegant("implied vertical emittance is invalid", NULL);
  if ((emit_y = sqrt(emit_y)) > 0) {
    beta_y = sqr(bunched_beam_moments_struct.S3_beta) / emit_y;
    alpha_y = -bunched_beam_moments_struct.S34_beta / emit_y;
  }

  if (bunched_beam_moments_struct.S6) {
    eta_x = bunched_beam_moments_struct.S16 / sqr(bunched_beam_moments_struct.S6);
    etap_x = bunched_beam_moments_struct.S26 / sqr(bunched_beam_moments_struct.S6);
    eta_y = bunched_beam_moments_struct.S36 / sqr(bunched_beam_moments_struct.S6);
    etap_y = bunched_beam_moments_struct.S46 / sqr(bunched_beam_moments_struct.S6);
    if (bunched_beam_moments_struct.S5)
      dp_s_coupling = bunched_beam_moments_struct.S56 / (bunched_beam_moments_struct.S5 * bunched_beam_moments_struct.S6);
  }
  sigma_s = bunched_beam_moments_struct.S5;
  sigma_dp = bunched_beam_moments_struct.S6;

  for (i = 0; i < 3; i++) {
    halton_sequence[i] = bunched_beam_moments_struct.halton_sequence[i];
    randomize_order[i] = bunched_beam_moments_struct.randomize_order[i];
    enforce_rms_values[i] = bunched_beam_moments_struct.enforce_rms_values[i];
    distribution_cutoff[i] = bunched_beam_moments_struct.distribution_cutoff[i];
    distribution_type[i] = bunched_beam_moments_struct.distribution_type[i];
  }
  for (i = 0; i < 6; i++) {
    halton_radix[i] = bunched_beam_moments_struct.halton_radix[i];
    centroid[i] = bunched_beam_moments_struct.centroid[i];
  }

  bunch = bunched_beam_moments_struct.bunch;
  n_particles_per_bunch = bunched_beam_moments_struct.n_particles_per_bunch;
  use_twiss_command_values = 0;
  use_moments_output_values = bunched_beam_moments_struct.use_moments_output_values;
  matched_to_cell = NULL;
  time_start = bunched_beam_moments_struct.time_start;
  Po = bunched_beam_moments_struct.Po;
  one_random_bunch = bunched_beam_moments_struct.one_random_bunch;
  save_initial_coordinates = bunched_beam_moments_struct.save_initial_coordinates;
  limit_invariants = bunched_beam_moments_struct.limit_invariants;
  symmetrize = bunched_beam_moments_struct.symmetrize;
  optimized_halton = bunched_beam_moments_struct.optimized_halton;
  limit_in_4d = bunched_beam_moments_struct.limit_in_4d;
  first_is_fiducial = bunched_beam_moments_struct.first_is_fiducial;

  finish_bunched_beam_setup(beam, run, control, errcon, optim, output, beamline, n_elements, save_original);
}

long new_bunched_beam(
  BEAM *beam,
  RUN *run,
  VARY *control,
  OUTPUT_FILES *output,
  long flags) {
  static long n_actual_particles;
  long i_coord;
  int64_t i_particle;
  unsigned long unstable;
  double s_offset, beta;
  double p_central, dummy, p, gamma;
  long i, offset;
#if USE_MPI
  double save_emit_x = 0, save_emit_y = 0,
         save_sigma_dp = 0, save_sigma_s = 0; /* initialize these to prevent spurious compiler warning */

  if ((firstIsFiducial && beamCounter == 0) || do_find_aperture || independentRunPerRank || (beam->n_original_total == 1)) {
    distributedBeam = 0;
    lessPartAllowed = 1;
  } else {
    distributedBeam = 1;
    partOnMaster = 0;
    lessPartAllowed = 0;
  }
#endif

  beamCounter++;
  if (firstIsFiducial && beamCounter == 1) {
    beam->n_original = beam->n_to_track = 1;
#if USE_MPI
    lessPartAllowed = 1; /* All the processors will do the same thing from now */
    if (isMaster) {      /* For parallel I/O version, memory will be allocated for one particle on master */
      beam->particle = (double **)czarray_2d(sizeof(double), 1, totalPropertiesPerParticle);
      if (!save_initial_coordinates)
        beam->original = beam->particle;
      else
        beam->original = (double **)czarray_2d(sizeof(double), 1, totalPropertiesPerParticle);
      if (run->acceptance)
        beam->accepted = (double **)czarray_2d(sizeof(double), 1, totalPropertiesPerParticle);
      beam->n_lost = 0;
    }
#endif
  } else {
    if (firstIsFiducial && beamCounter == 2) {
      bunchGenerated = 0;
      particleID = 1;
    }
    beam->n_original = beam->n_to_track = beam->n_particle = n_particles_per_bunch;
#if USE_MPI
    if (isMaster) {
      if (distributedBeam) {
        beam->particle = beam->original = beam->accepted = NULL;
        beam->n_original = beam->n_to_track = beam->n_particle = 0;
      } else {
        beam->particle = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
        if (!save_initial_coordinates)
          beam->original = beam->particle;
        else if (!bunchGenerated)
          beam->original = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
        if (run->acceptance)
          beam->accepted = (double **)czarray_2d(sizeof(double), n_particles_per_bunch, totalPropertiesPerParticle);
        beam->n_lost = 0;
      }
    }
#endif
  }

  beam->n_accepted = 0;

  p_central = beam->p0_original = run->p_central;

#if USE_MPI
  if (!distributedBeam || isSlave) /* Master has NULL pointer for both beam->original and beam->particle */
#endif
    if (flags & TRACK_PREVIOUS_BUNCH && beam->original == beam->particle)
      bombElegant("programming error: original beam coordinates needed but not saved", NULL);

  if (!bunchGenerated || !save_initial_coordinates) {
    if (one_random_bunch) {
      /* reseed the random number generator to get the same sequence again */
      /* means we don't have to store the original coordinates */
#if SDDS_MPI_IO
      /* There will be no same sequence for different processors */
      if (distributedBeam)
        random_4(-(beamRepeatSeed + 2 * (myid - 1)));
      else
        random_4(-beamRepeatSeed);
#else
      random_4(-beamRepeatSeed);
#endif
      /* For the Halton sequence, we need reset the start point */
      for (i = 0; i < 3; i++) {
        for (offset = 0; offset < 2; offset++) {
          if (halton_sequence[i]) {
            if (haltonOpt) {
              if (restartModHaltonSequence(haltonID[2 * i + offset], 0.5) < 0)
                bombElegant("problem restarting Halton sequence", NULL);
            } else {
              if (restartHaltonSequence(haltonID[2 * i + offset], 0.5) < 0)
                bombElegant("problem restarting Halton sequence", NULL);
            }
          }
        }
      }
    }
    /* This part will take effect for regression test */
#if SDDS_MPI_IO
    else if (control->n_steps == 1) {
      if (distributedBeam)
        random_4(-(beamRepeatSeed + 2 * (myid - 1)));
      else
        random_4(-beamRepeatSeed);
    }
#else
    else if ((run->random_sequence_No > 1) && (control->n_steps == 1)) {
      random_4(-beamRepeatSeed);
    }
#endif
    if (control->cell) {
      VMATRIX *M;
      unsigned long savedFlags;
      savedFlags = control->cell->flags;
      M = compute_periodic_twiss(&beta_x, &alpha_x, &eta_x, &etap_x, &dummy,
                                 &beta_y, &alpha_y, &eta_y, &etap_y, &dummy,
                                 (control->cell->elem_recirc ? control->cell->elem_recirc : control->cell->elem),
                                 NULL, run, &unstable, NULL, NULL);
      control->cell->flags = savedFlags;
      free_matrices(M);
      free(M);
      M = NULL;
      printf("matched Twiss parameters for beam generation:\nbetax = %13.6e m  alphax = %13.6e  etax = %13.6e m  etax' = %13.6e\n",
             beta_x, alpha_x, eta_x, etap_x);
      fflush(stdout);
      printf("betay = %13.6e m  alphay = %13.6e  etay = %13.6e m  etay' = %13.6e\n",
             beta_y, alpha_y, eta_y, etap_y);
      fflush(stdout);
      fill_transverse_structure(&x_plane, emit_x, beta_x, alpha_x, eta_x, etap_x,
                                x_beam_type, distribution_cutoff[0], centroid);
      fill_transverse_structure(&y_plane, emit_y, beta_y, alpha_y, eta_y, etap_y,
                                y_beam_type, distribution_cutoff[1], centroid + 2);
    }
    printf("generating bunch %ld\n", control->i_step);
    fflush(stdout);
#if USE_MPI
    if (firstIsFiducial && beamCounter == 1) {
      /* Set emittances to zero temporarily. This will make sure the coordniates are same 
	   across all the processors. */
      save_emit_x = x_plane.emit;
      save_emit_y = y_plane.emit;
      save_sigma_dp = longit.sigma_dp;
      save_sigma_s = longit.sigma_s;
      x_plane.emit = y_plane.emit = longit.sigma_dp = longit.sigma_s = 0;
    }
#endif
#if !SDDS_MPI_IO
    /* This part will take effect for regression test only where random_sequence_No >1 */
    if (run->random_sequence_No > 1 && (one_random_bunch || (!one_random_bunch && (control->n_steps == 1))) && (beam->n_to_track >= (run->random_sequence_No - 1))) {
      /* generate the same random number sequence as the parallel version */
      long work_processors = run->random_sequence_No - 1, my_n_actual_particles;
      long my_nToTrack, i;

      orig_sequence_No = remaining_sequence_No = run->random_sequence_No - 1;
      n_actual_particles = 0;
      for (i = 0; i < work_processors; i++) {
        /* This will make the serial version has the same start seed as each of the sequences 
	       in the parallel version */
        random_4(-(beamRepeatSeed + 2 * i));
        my_nToTrack = beam->n_to_track / work_processors;
        if (i < (beam->n_to_track % work_processors))
          my_nToTrack++;
        if (x_plane.beam_type == DYNAP_BEAM) {
          /* This is a special case, generate_bunch will be called one time only */
          remaining_sequence_No = 1;
          my_nToTrack = beam->n_to_track;
        }
        if (!use_moments_output_values) {
          my_n_actual_particles =
            generate_bunch(&beam->original[n_actual_particles], my_nToTrack, &x_plane,
                           &y_plane, &longit, enforce_rms_values, limit_invariants,
                           symmetrize, haltonID, haltonOpt, randomize_order, limit_in_4d, Po);
        } else {
          my_n_actual_particles =
            generateBunchForMoments(&beam->original[n_actual_particles], my_nToTrack, symmetrize,
                                    haltonID, haltonOpt, distribution_cutoff[0]);
        }
        n_actual_particles += my_n_actual_particles;
        if (x_plane.beam_type == DYNAP_BEAM)
          break;
      }
    } else {
#endif
      if (!use_moments_output_values) {
        n_actual_particles =
          generate_bunch(beam->original, beam->n_to_track, &x_plane,
                         &y_plane, &longit, enforce_rms_values, limit_invariants,
                         symmetrize, haltonID, haltonOpt, randomize_order, limit_in_4d, Po);
      } else {
        n_actual_particles =
          generateBunchForMoments(beam->original, beam->n_to_track, symmetrize, haltonID, haltonOpt, distribution_cutoff[0]);
      }
#if !SDDS_MPI_IO
    }
#endif

  if (spinCoordOffset) {
    long isp;
    double norm;
    norm = sqrt(sqr(spin[0])+sqr(spin[1])+sqr(spin[2]));
    for (isp=0; isp<3; isp++)
      spin[isp] /= norm;
    for (i_particle = 0; i_particle < beam->n_to_track; i_particle++)
      for (isp=0; isp<3; isp++)
	beam->original[i_particle][spinCoordOffset+isp] = spin[isp];
  }
    
#if USE_MPI
    if (firstIsFiducial && beamCounter == 1) {
      /* copy values back */
      x_plane.emit = save_emit_x;
      y_plane.emit = save_emit_y;
      longit.sigma_dp = save_sigma_dp;
      longit.sigma_s = save_sigma_s;
    }
#endif
    if (bunch) {
      if (!SDDS_bunch_initialized)
        bombElegant("'bunch' file is uninitialized (new_bunched_beam)", NULL);
      printf("dumping bunch\n");
      fflush(stdout);
      dump_phase_space(&SDDS_bunch, beam->original, n_actual_particles, control->i_step, Po,
                       sqrt(-1.0), n_actual_particles);
      if (one_random_bunch && !(firstIsFiducial && beamCounter == 1)) {
        if (!SDDS_Terminate(&SDDS_bunch)) {
          SDDS_SetError("Problem terminating 'bunch' file (new_bunched_beam)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        bunch = NULL;
      }
    }
  }

  beam->n_to_track = n_actual_particles;
#if USE_MPI
  if (distributedBeam)
    MPI_Allreduce(&beam->n_to_track, &beam->n_to_track_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif

  /* copy particles into tracking buffer, adding in the time offset 
     * and converting from deltap/p relative to bunch center (defined
     * by Po) to deltap/p relative to beamline energy (defined by
     * p_central).  Also offset particle ID for some cases.
     */
  beta = Po / sqrt(sqr(Po) + 1);
  s_offset = control->bunch_frequency ? (control->i_step - 1) * (beta * c_mks) / control->bunch_frequency : 0;
  for (i_particle = 0; i_particle < beam->n_to_track; i_particle++) {
    if (beam->particle != beam->original) {
      for (i_coord = 0; i_coord < totalPropertiesPerParticle; i_coord++)
        beam->particle[i_particle][i_coord] = beam->original[i_particle][i_coord];
#if USE_MPI
      beam->original[i_particle][6] += beam->n_to_track_total;
#else
      beam->original[i_particle][6] += beam->n_to_track;
#endif
    }
    p = Po * (1 + beam->particle[i_particle][5]);
    gamma = sqrt(sqr(p) + 1);
    beta = p / gamma;
    beam->particle[i_particle][5] = (p - p_central) / p_central;
    beam->particle[i_particle][4] += s_offset;
    beam->particle[i_particle][lossPassIndex] = -1;
    beam->particle[i_particle][bunchIndex] = 0;
  }

  bunchGenerated = 1;
#if USE_MPI
  beam->id_slots_per_bunch = beam->n_to_track_total;
#else
  beam->id_slots_per_bunch = beam->n_to_track;
#endif



  return (beam->n_to_track);
}

long track_beam(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  LINE_LIST *beamline,
  BEAM *beam,
  OUTPUT_FILES *output,
  unsigned long flags,
  long delayOutput,
  double *finalCharge) {
  double p_central;
  int64_t n_left;
  long effort;

  log_entry("track_beam");

  if (!run)
    bombElegant("RUN pointer is NULL (track_beam)", NULL);
  if (!control)
    bombElegant("VARY pointer is NULL (track_beam)", NULL);
  if (!errcon)
    bombElegant("ERROR pointer is NULL (track_beam)", NULL);
  if (!beamline)
    bombElegant("LINE_LIST pointer is NULL (track_beam)", NULL);
  if (!beam)
    bombElegant("BEAM pointer is NULL (track_beam)", NULL);
  if (!output)
    bombElegant("OUTPUT_FILES pointer is NULL (track_beam)", NULL);

  p_central = run->p_central;

#ifdef DEBUG
  flags &= !SILENT_RUNNING;
#  if USE_MPI
  printf("parallelStatus = %d, partOnMaster = %d, distributedBeam = %ld, independentRunPerRank = %ld\n",
         parallelStatus, partOnMaster, distributedBeam, independentRunPerRank);
#  endif
#endif

  /* now track particles */
  if (!(flags & SILENT_RUNNING)) {
#if !SDDS_MPI_IO
    printf("tracking %" PRId64 " particles\n", beam->n_to_track);
#else
    if (partOnMaster)
      printf("tracking %" PRId64 " particles\n", beam->n_to_track);
    else
      printf("tracking %" PRId64 " particles\n", beam->n_to_track_total);
#endif
    fflush(stdout);
  }
  effort = 0;

  /* Macro-scale per-step timing offset (seconds) that is deliberately kept out
     of the bunch coordinates part[][4]; time-dependent elements (RFMODE damping,
     BUMPER/MBUMPER, ion effects, ...) and time outputs pick it up through
     trackingClockOffset().  step_frequency==0 -> base 0 -> bit-identical to before.
     i_step starts at 1, so the first step has base 0.  Set on all ranks. */
  setTrackingClockBase(control->step_frequency ? (control->i_step - 1) / control->step_frequency : 0.0);

#if USE_MPI
  if (isSlave && beam->n_to_track < 1 && !lessPartAllowed)
    printWarning("The number of particles shouldn't be less than the number of processors", NULL);
  else {                /* do tracking in parallel */
    if (partOnMaster) { /* all processors will excute the same code */
      distributedBeam = 0;
    } else {
      distributedBeam = 1;
    }
  }

  if (isSlave || !distributedBeam)
#endif

#ifdef DEBUG
#  if USE_MPI
    printf("About to track: parallelStatus = %d, partOnMaster = %d, distributedBeam = %ld, independentRunPerRank = %ld\n",
           parallelStatus, partOnMaster, distributedBeam, independentRunPerRank);
#  endif
  printMessageAndTime(stdout, "Calling do_tracking\n");
#endif
  if (run->showElementTiming)
    resetElementTiming();
  n_left = do_tracking(beam, NULL, 0, &effort, beamline, &p_central,
                       beam->accepted, &output->sums_vs_z, &output->n_z_points,
                       NULL, run, control->i_step,
                       (!(run->centroid || run->bpmCentroid || run->sigma) ? FINAL_SUMS_ONLY : (!run->sigma ? CENTROID_SUMS_ONLY : 0)) |
                         ((control->fiducial_flag | flags) &
                          (LINEAR_CHROMATIC_MATRIX + LONGITUDINAL_RING_ONLY + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + FIDUCIAL_BEAM_SEEN + RESTRICT_FIDUCIALIZATION + PRECORRECTION_BEAM + IBS_ONLY_TRACKING + RESET_RF_FOR_EACH_STEP + OPTIMIZING)) |
                         ALLOW_MPI_ABORT_TRACKING,
                       firstIsFiducial && control->i_step == 1 ? 1 : control->n_passes,
                       0, &(output->sasefel), &(output->sliceAnalysis),
                       finalCharge, NULL, NULL);
#ifdef DEBUG
  printMessageAndTime(stdout, "Returned from do_tracking\n");
#endif
  if (run->showElementTiming)
    reportElementTiming();
  if (control->fiducial_flag & FIRST_BEAM_IS_FIDUCIAL && !(flags & PRECORRECTION_BEAM)) {
    control->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
    beamline->fiducial_flag |= FIDUCIAL_BEAM_SEEN; /* This is the one that matters */
  }

  if (!beam) {
    printf("error: beam pointer is null on return from do_tracking (track_beam)\n");
    fflush(stdout);
    abort();
  }
  beam->p0 = p_central;
  beam->n_accepted = n_left;

  /*
  if (!(flags&SILENT_RUNNING)) {
    extern unsigned long multipoleKicksDone;
#if SDDS_MPI_IO
    if (SDDS_MPI_IO) {
      long total_effort, total_multipoleKicksDone, total_left;
      
      if (distributedBeam) {
	if (isMaster) 
	  multipoleKicksDone = effort = n_left = 0;
	MPI_Reduce (&n_left, &total_left, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce (&effort, &total_effort, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce (&multipoleKicksDone, &total_multipoleKicksDone, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	printf("%ld particles transmitted, total effort of %ld particle-turns\n", total_left, total_effort);
	fflush(stdout);
	printf("%lu multipole kicks done\n\n", total_multipoleKicksDone);
      }
      else {
	printf("%ld particles transmitted, total effort of %ld particle-turns\n", n_left, effort);
	fflush(stdout);
	printf("%lu multipole kicks done\n\n", multipoleKicksDone);
      }
      fflush(stdout);
    }
#else
    printf("%ld particles transmitted, total effort of %ld particle-turns\n", n_left, effort);
    fflush(stdout);
    printf("%lu multipole kicks done\n\n", multipoleKicksDone);
    fflush(stdout);
#endif
  }
  */

#if USE_MPI
  /*  if (!independentRunPerRank) */
  /* Disable the beam output when all the processors track independently, especially for
       genetic optimization.  Using independentRunPerRank flag instead of distributedBeam as 
       we do need output when first is fiducial beam and it does not satisfy distributedBeam */
#endif
  if (!delayOutput)
    do_track_beam_output(run, control, errcon, optim,
                         beamline, beam, output, flags,
                         finalCharge ? *finalCharge : 0.0);

  if (!(flags & SILENT_RUNNING)) {
    report_stats(stdout, "Tracking step completed");
    fputs("\n\n", stdout);
  }

  log_exit("track_beam");
  return (1);
}

void do_track_beam_output(RUN *run, VARY *control,
                          ERRORVAL *errcon, OPTIM_VARIABLES *optim,
                          LINE_LIST *beamline, BEAM *beam,
                          OUTPUT_FILES *output, unsigned long flags,
                          double finalCharge) {
  VMATRIX *M;
  int64_t n_left;
  double p_central, p_central0;

  p_central = beam->p0;
  n_left = beam->n_accepted;
  p_central0 = beam->p0_original;

  if (run->output && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->output_initialized)
      bombElegant("'output' file is uninitialized (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping output beam data...");
      fflush(stdout);
    }
    dump_phase_space(&output->SDDS_output, beam->particle, n_left, control->i_step, p_central,
                     finalCharge, beam->id_slots_per_bunch);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->acceptance && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->accept_initialized)
      bombElegant("'acceptance' file is uninitialized (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping acceptance output...");
      fflush(stdout);
    }
    dump_phase_space(&output->SDDS_accept, beam->accepted, beam->n_accepted, control->i_step, p_central0,
                     finalCharge, beam->id_slots_per_bunch);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->losses && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->losses_initialized)
      bombElegant("'losses' file is uninitialized (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping lost-particle data...\n");
      fflush(stdout);
#if USE_MPI
      if (USE_MPI) {
        long total_lost, n_lost;
        if (distributedBeam) {
          if (isMaster)
            n_lost = 0;
          else
            n_lost = beam->n_lost;
          MPI_Allreduce(&n_lost, &total_lost, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        } else
          total_lost = beam->n_lost;
        printf("n_lost = %ld\n", total_lost);
      }
#else
      printf("n_lost = %" PRId64 "\n", beam->n_lost);
#endif
      fflush(stdout);
    }

    printf("Dumping lost particles: n_to_track = %" PRId64 ", n_lost = %" PRId64 ", n_particles = %" PRId64 "\n",
           beam->n_to_track, beam->n_lost, beam->n_particle);
    fflush(stdout);
    dump_lost_particles(&output->SDDS_losses, run->lossLimit, p_central0, beam->particle + beam->n_accepted,
                        beam->n_lost, control->i_step, beamline->revolution_length);
    beam->n_lost = 0;
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->centroid && !run->combine_bunch_statistics && !(flags & FINAL_SUMS_ONLY) && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->centroid_initialized)
      bombElegant("'centroid' file is uninitialized (track_beam)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping centroid data...");
      fflush(stdout);
    }
    dump_centroid(&output->SDDS_centroid, output->sums_vs_z, beamline, output->n_z_points, control->i_step, p_central, 0);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->bpmCentroid && !run->combine_bunch_statistics && !(flags & FINAL_SUMS_ONLY) && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->bpmCentroid_initialized)
      bombElegant("'bpm_centroid' file is uninitialized (track_beam)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping bpm_centroid data...");
      fflush(stdout);
    }
    dump_centroid(&output->SDDS_bpmCentroid, output->sums_vs_z, beamline, output->n_z_points, control->i_step, p_central, 1);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->sigma && !run->combine_bunch_statistics && !(flags & FINAL_SUMS_ONLY) && !(flags & INHIBIT_FILE_OUTPUT)) {
    if (!output->sigma_initialized)
      bombElegant("'sigma' file is uninitialized (track_beam)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (track_beam)", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping sigma data...");
      fflush(stdout);
    }
    if (flags & CENTROID_SUMS_ONLY) {
      bombElegant("sigma output requested but data wasn't accumulated. This is a bug!", NULL);
    }
    dump_sigma(&output->SDDS_sigma, output->sums_vs_z, beamline, output->n_z_points, control->i_step, p_central,
	       run->coupledSigmaOutput);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (run->final && !run->combine_bunch_statistics) {
    if (!(M = full_matrix(beamline->elem, run, 1)))
      bombElegant("computation of full matrix for final output failed (track_beam)", NULL);
    if (!output)
      bombElegant("output pointer is NULL on attempt to write 'final' file (track_beam)", NULL);
    if (!output->final_initialized)
      bombElegant("'final' file is uninitialized (track_beam)", NULL);
    if (!output->sums_vs_z)
      bombElegant("beam sums array for final output is NULL", NULL);
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping final properties data...");
      fflush(stdout);
    }
#if !USE_MPI
    dump_final_properties(&output->SDDS_final, output->sums_vs_z + output->n_z_points,
                          control->varied_quan_value, control->varied_quan_name ? *control->varied_quan_name : NULL,
                          control->n_elements_to_vary, control->n_steps * control->indexLimitProduct,
                          errcon->error_value, errcon->quan_final_index, errcon->quan_final_duplicates, errcon->n_items,
                          optim->varied_quan_value, optim->varied_quan_name ? *optim->varied_quan_name : NULL,
                          optim->n_variables ? optim->n_variables + 3 : 0,
                          optim->n_variables ? optim->lower_limit : NULL,
                          optim->n_variables ? optim->upper_limit : NULL,
                          control->i_step, beam->particle, beam->n_to_track, p_central, M,
                          finalCharge);
#else
#if MPI_DEBUG
    printf("Preparing to dump final properties data.\n");
    fflush(stdout);
#endif
    if (distributedBeam)
    MPI_Reduce(&(beam->n_to_track), &(beam->n_to_track_total), 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
#if MPI_DEBUG
    printf("Dumping final properties data.\n");
    fflush(stdout);
#endif
    dump_final_properties(&output->SDDS_final, output->sums_vs_z + output->n_z_points,
                          control->varied_quan_value, control->varied_quan_name ? *control->varied_quan_name : NULL,
                          control->n_elements_to_vary, control->n_steps * control->indexLimitProduct,
                          errcon->error_value, errcon->quan_final_index, errcon->quan_final_duplicates, errcon->n_items,
                          optim->varied_quan_value, optim->varied_quan_name ? *optim->varied_quan_name : NULL,
                          optim->n_variables ? optim->n_variables + 3 : 0,
                          optim->n_variables ? optim->lower_limit : NULL,
                          optim->n_variables ? optim->upper_limit : NULL,
                          control->i_step, beam->particle, beam->n_to_track_total, p_central, M,
                          finalCharge);
#if MPI_DEBUG
    printf("Finished dumping final properties data.\n");
    fflush(stdout);
#endif
#endif
    free_matrices(M);
    free(M);
    M = NULL;
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (output->sasefel.active && output->sasefel.filename) {
    if (!(flags & SILENT_RUNNING)) {
      printf("Dumping SASE FEL data...");
      fflush(stdout);
    }
    doSASEFELAtEndOutput(&(output->sasefel), control->i_step);
    if (!(flags & SILENT_RUNNING)) {
      printf("done.\n");
      fflush(stdout);
    }
  }

  if (!(flags & SILENT_RUNNING)) {
    printf("Post-tracking output completed.\n");
    fflush(stdout);
  }
}

void setup_output(
  OUTPUT_FILES *output,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  LINE_LIST *beamline) {
#if USE_MPI
  if (independentRunPerRank && (!enableOutput)) {
    if (run->acceptance || run->centroid || run->sigma || run->final || run->output || run->losses) {
      printf("\nN.B.: Tracking will be done independently on each processor for this simulation\n");
      printf("Pelegant does not provide intermediate output for optimization now.\n\n");
      run->acceptance = run->centroid = run->sigma = run->final = run->output = run->losses = NULL;
    }
    return;
  }
#endif

  if (run->acceptance) {
    /* prepare dump of accepted particles */
    SDDS_PhaseSpaceSetup(&output->SDDS_accept, run->acceptance, SDDS_BINARY, 1, "accepted phase space", run->runfile,
                         run->lattice, "setup_output");
    output->accept_initialized = 1;
  }

  if (run->centroid) {
    /* prepare dump of centroid vs z */
    SDDS_CentroidOutputSetup(&output->SDDS_centroid, run->centroid, SDDS_BINARY, 1, "centroid output", run->runfile,
                             run->lattice, "setup_output", 0);
    output->centroid_initialized = 1;
  }

  if (run->bpmCentroid) {
    /* prepare dump of centroid vs z at BPMs */
    SDDS_CentroidOutputSetup(&output->SDDS_bpmCentroid, run->bpmCentroid, SDDS_BINARY, 1, "output of centroid at BPMs", run->runfile,
                             run->lattice, "setup_output", 1);
    output->bpmCentroid_initialized = 1;
  }

  if (run->sigma) {
    /* prepare dump of sigma vs z */
    SDDS_SigmaOutputSetup(&output->SDDS_sigma, run->sigma, SDDS_BINARY, 1, run->runfile,
                          run->lattice, "setup_output", run->coupledSigmaOutput);
    output->sigma_initialized = 1;
  }

  if (run->output) {
    /* prepare output dump */
    SDDS_PhaseSpaceSetup(&output->SDDS_output, run->output, SDDS_BINARY, 1, "output phase space", run->runfile,
                         run->lattice, "setup_output");
    output->output_initialized = 1;
  }

  if (run->final) {
    /* prepare 'final' dump */
    SDDS_FinalOutputSetup(&output->SDDS_final, run->final, SDDS_BINARY, 1, "final properties", run->runfile, run->lattice,
                          control->varied_quan_name, control->varied_quan_unit, control->n_elements_to_vary,
                          errcon->quan_name, errcon->quan_unit, errcon->n_items,
                          errcon->quan_final_index, &errcon->quan_final_duplicates,
                          optim->varied_quan_name, optim->varied_quan_unit,
                          (optim->n_variables ? optim->n_variables + 3 : 0),
                          "setup_output");
    output->final_initialized = 1;
  }

  if (run->losses) {
    /* prepare dump of lost particles */
    SDDS_BeamLossSetup(&output->SDDS_losses, run->losses, SDDS_BINARY, 1,
                       "lost particle coordinates", run->runfile,
                       run->lattice, run->lossesIncludeGlobalCoordinates, "setup_output");
    output->losses_initialized = 1;
  }

  if (output->sums_vs_z)
    freeBeamSums(output->sums_vs_z, output->n_z_points + 1);
  output->sums_vs_z = NULL;
  output->n_z_points = 0;
}

void finish_output(
  OUTPUT_FILES *output,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  OPTIM_VARIABLES *optim,
  LINE_LIST *beamline,
  long n_elements,
  BEAM *beam,
  double finalCharge) {
  ELEMENT_LIST *eptr;

  log_entry("finish_output.1");

  if (!output)
    bombElegant("null OUTPUT_FILES pointer (finish_output)", NULL);
  if (!run)
    bombElegant("null RUN pointer (finish_output)", NULL);
  if (!control)
    bombElegant("null VARY pointer (finish_output)", NULL);
  if (!errcon)
    bombElegant("null ERROR pointer (finish_output)", NULL);
  if (!beam)
    bombElegant("null BEAM pointer (finish_output)", NULL);
  if (!beamline)
    bombElegant("null BEAMLINE pointer (finish_output)", NULL);

  eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_WATCH) {
      WATCH *watch;
      watch = (WATCH *)eptr->p_elem;
      if (watch->initialized) {
        if ((watch->useDisconnect && SDDS_IsDisconnected(watch->SDDS_table) &&
             !SDDS_ReconnectFile(watch->SDDS_table)) ||
            !SDDS_Terminate(watch->SDDS_table)) {
          char s[16384];
          snprintf(s, 16384, "Error terminating watch file %s\n", watch->filename);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          printWarning(s, NULL);
        }
      }
    }
    eptr = eptr->succ;
  }

  if (run->centroid && run->combine_bunch_statistics) {
    if (!output->centroid_initialized)
      bombElegant("'centroid' file is uninitialized (finish_output)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (finish_output)", NULL);
    dump_centroid(&output->SDDS_centroid, output->sums_vs_z, beamline, output->n_z_points, 0, beam->p0, 0);
  }
  if (run->bpmCentroid && run->combine_bunch_statistics) {
    if (!output->bpmCentroid_initialized)
      bombElegant("'bpm_centroid' file is uninitialized (finish_output)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (finish_output)", NULL);
    dump_centroid(&output->SDDS_bpmCentroid, output->sums_vs_z, beamline, output->n_z_points, 0, beam->p0, 1);
  }
  if (run->sigma && run->combine_bunch_statistics) {
    if (!output->sigma_initialized)
      bombElegant("'sigma' file is uninitialized (finish_output)", NULL);
    if (!output->sums_vs_z)
      bombElegant("missing beam sums pointer (finish_output)", NULL);
    dump_sigma(&output->SDDS_sigma, output->sums_vs_z, beamline, output->n_z_points, 0, beam->p0, run->coupledSigmaOutput);
  }
  if (run->final && run->combine_bunch_statistics) {
    VMATRIX *M;
    if (!output->final_initialized)
      bombElegant("'final' file is uninitialized (track_beam)", NULL);
#if !USE_MPI
    dump_final_properties(&output->SDDS_final, output->sums_vs_z + output->n_z_points,
                          control->varied_quan_value, control->varied_quan_name ? *control->varied_quan_name : NULL,
                          control->n_elements_to_vary, control->indexLimitProduct * control->n_steps,
                          errcon->error_value, errcon->quan_final_index, errcon->quan_final_duplicates,
                          errcon->n_items,
                          optim->varied_quan_value, optim->varied_quan_name ? *optim->varied_quan_name : NULL,
                          optim->n_variables ? optim->n_variables + 2 : 0,
                          optim->n_variables ? optim->lower_limit : NULL,
                          optim->n_variables ? optim->upper_limit : NULL,
                          0, beam->particle, beam->n_to_track, beam->p0, M = full_matrix(beamline->elem, run, 1),
                          finalCharge);
#else
    if (distributedBeam)
    MPI_Reduce(&(beam->n_to_track), &(beam->n_to_track_total), 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    dump_final_properties(&output->SDDS_final, output->sums_vs_z + output->n_z_points,
                          control->varied_quan_value, control->varied_quan_name ? *control->varied_quan_name : NULL,
                          control->n_elements_to_vary, control->indexLimitProduct * control->n_steps,
                          errcon->error_value, errcon->quan_final_index, errcon->quan_final_duplicates,
                          errcon->n_items,
                          optim->varied_quan_value, optim->varied_quan_name ? *optim->varied_quan_name : NULL,
                          optim->n_variables ? optim->n_variables + 2 : 0,
                          optim->n_variables ? optim->lower_limit : NULL,
                          optim->n_variables ? optim->upper_limit : NULL,
                          0, beam->particle, beam->n_to_track_total, beam->p0, M = full_matrix(beamline->elem, run, 1),
                          finalCharge);
#endif

    free_matrices(M);
    free(M);
    M = NULL;
  }
  log_exit("finish_output.1");

  log_entry("finish_output.2");
  if (run->output) {
    if (!output->output_initialized)
      bombElegant("'output' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_output)) {
      SDDS_SetError("Problem terminating 'output' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    }
    output->output_initialized = 0;
  }
  if (run->acceptance) {
    if (!output->accept_initialized)
      bombElegant("'acceptance' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_accept)) {
      SDDS_SetError("Problem terminating 'acceptance' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    output->accept_initialized = 0;
  }
  if (run->centroid) {
    if (!output->centroid_initialized)
      bombElegant("'centroid' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_centroid)) {
      SDDS_SetError("Problem terminating 'centroid' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    output->centroid_initialized = 0;
  }
  if (run->sigma) {
    if (!output->sigma_initialized)
      bombElegant("'sigma' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_sigma)) {
      SDDS_SetError("Problem terminating 'sigma' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    output->sigma_initialized = 0;
  }
  if (run->final) {
    if (!output->final_initialized)
      bombElegant("'final' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_final)) {
      SDDS_SetError("Problem terminating 'final' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    output->final_initialized = 0;
  }
  if (bunch) {
    if (!SDDS_bunch_initialized)
      bombElegant("'bunch' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&SDDS_bunch)) {
      SDDS_SetError("Problem terminating 'bunch' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_bunch_initialized = 0;
  }
  if (run->losses) {
    if (!output->losses_initialized)
      bombElegant("'losses' file is uninitialized (finish_output)", NULL);
    if (!SDDS_Terminate(&output->SDDS_losses)) {
      SDDS_SetError("Problem terminating 'losses' file (finish_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    output->losses_initialized = 0;
  }
  log_exit("finish_output.2");

  if (output->particleTunes.initialized)
    finishParticleTunes(&(output->particleTunes));
  
  log_entry("finish_output.3");
  if (output->sums_vs_z) {
    tfree(output->sums_vs_z);
    output->sums_vs_z = NULL;
  }
  log_exit("finish_output.3");
}

void fill_transverse_structure(
  TRANSVERSE *trans,
  double emit,
  double beta,
  double alpha,
  double eta,
  double etap,
  long xbeam_type,
  double cutoff,
  double *xcentroid) {
  log_entry("fill_transverse_structure");
  trans->emit = emit;
  trans->beta = beta;
  trans->alpha = alpha;
  trans->eta = eta;
  trans->etap = etap;
  trans->beam_type = xbeam_type;
  trans->cutoff = cutoff;
  trans->cent_posi = xcentroid[0];
  trans->cent_slope = xcentroid[1];
  log_exit("fill_transverse_structure");
}

void fill_longitudinal_structure(
  LONGITUDINAL *xlongit,
  double xsigma_dp,
  double xsigma_s,
  double xdp_s_coupling,
  double xemit_z,
  double xbeta_z,
  double xalpha_z,
  double xchirp,
  long xbeam_type,
  double xcutoff,
  double *xcentroid) {
  log_entry("fill_longitudinal_structure");
  xlongit->sigma_dp = xsigma_dp;
  xlongit->sigma_s = xsigma_s;
  xlongit->dp_s_coupling = xdp_s_coupling;
  xlongit->emit = xemit_z;
  xlongit->beta = xbeta_z;
  xlongit->alpha = xalpha_z;
  xlongit->beam_type = xbeam_type;
  xlongit->cutoff = xcutoff;
  xlongit->cent_s = xcentroid[0];
  xlongit->cent_dp = xcentroid[1];
  xlongit->chirp = xchirp;
  log_exit("fill_longitudinal_structure");
}

#if !defined(USE_GSL)
long generateBunchForMoments(double **particle, int64_t np, long symmetrize,
  long *haltonID, long haltonOpt, double cutoff) {
    bombElegant("Error: Cannot run generateBunchForMoments, elegant was not build with GSL support.", NULL);
    return(0);
  }
#else
#include "gsl/gsl_linalg.h"

long generateBunchForMoments(double **particle, int64_t np, long symmetrize,
                             long *haltonID, long haltonOpt, double cutoff) {
  int64_t ip;
  long i, j;
  double Mm[6][6], Cm[6], ptemp[6];
  gsl_matrix *M;

#if USE_MPI
  int64_t sum = 0, tmp, my_offset = 0, *offset = tmalloc(n_processors * sizeof(*offset)), total_particles = 0;
  if (isSlave && distributedBeam) {
  MPI_Allgather(&np, 1, MPI_INT64_T, offset, 1, MPI_INT64_T, workers);
    tmp = offset[0];
    for (i = 1; i < n_processors; i++) {
      sum += tmp;
      tmp = offset[i];
      offset[i] = sum;
    }
    offset[0] = 0;
    my_offset = offset[myid - 1];
    particleID += my_offset;
    total_particles = sum;
  }
#endif

  if (np <= 0)
    return np;

  /* Generate gaussian-distributed, uncoupled particle coordinates */
  for (i = 0; i < 3; i++) {
    gaussian_distribution(particle, np, 2 * i, 1, 1, symmetrize, haltonID, haltonOpt,
                          cutoff, 0, 1.0, 0);
    zero_centroid(particle, np, 2 * i);
    zero_centroid(particle, np, 2 * i + 1);
    enforce_sigma_values(particle, np, 2 * i, 1.0, 1.0);
  }

  /* Get moments matrix to which we want to match the distribution */
  if (!getMoments(Mm, Cm, 1, 1, 1))
    bombElegant("Error: use_moments_output_values is nonzero in bunched_beam command, but (appropriate) moments_output command was not given.\n", NULL);

  /* Perform Cholesky decomposition */
  if (!(M = gsl_matrix_alloc(6, 6)))
    bombElegant("Error: gsl_matrix_alloc() failed when making bunch.", NULL);
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      gsl_matrix_set(M, i, j, Mm[i][j]);
  if (gsl_linalg_cholesky_decomp(M))
    bombElegant("Error: gsl_linalg_cholesky_decomp1() failed when making bunch. Does sigma matrix have blocks of zeros?", NULL);

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      Mm[i][j] = gsl_matrix_get(M, i, j);
  gsl_matrix_free(M);

  /* Transform coordinates */
  for (ip = 0; ip < np; ip++) {
    memcpy(ptemp, particle[ip], sizeof(*ptemp) * 6);
    for (i = 0; i < 6; i++) {
      particle[ip][i] = 0;
      for (j = 0; j <= i; j++)
        particle[ip][i] += Mm[i][j] * ptemp[j];
      if (i != 4)
        particle[ip][i] += Cm[i];
    }
  }

  /* Provide particle IDs */
  for (ip = 0; ip < np; ip++)
    particle[ip][6] = particleID++;

#if USE_MPI
  /* prepare for the next bunch */
  particleID += (total_particles - np - my_offset);
#endif

  return np;
}
#endif
