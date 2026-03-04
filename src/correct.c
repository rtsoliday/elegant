/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: correct.c
 * purpose: trajectory/orbit correction for elegant
 * 
 * Michael Borland, 1991. 
 */
#include "mdb.h"
//#include "mdbsun.h"
#include "track.h"
#include "match_string.h"
#include "correctDefs.h"

#define TRAJECTORY_MODE 0
#define ORBIT_MODE 1
#define N_CORRECTION_MODES 2
char *correction_mode[N_CORRECTION_MODES] = {
  "trajectory", "orbit"};

char *correction_method[N_CORRECTION_METHODS] = {
  "global", "one-to-one", "thread", "one-to-best", "one-to-next", "coupled"};

/* For trajectory correction:
 *   C      :     NMON x NCOR matrix of dQ_mon/dkick_cor 
 *   Qo     :     NMON x 1    matrix of initial monitor positions, Q stands for either x or y 
 *   Q      :     NMON x 1    matrix of final monitor positions
 *   dK     :     NCOR x 1    matrix of changes to correctors 
 *
 *   Q = Qo + C*dK
 *   Want to minimize S = SUM((Qi*wi)^2), where wi is the weight for the ith monitor.
 *   Solution is dK = -Inverse(Trans(C) W C) Trans(C) W Qo, or dK = T Qo.
 *   W is NMON x NMON, and W(i,j) = delta(i,j)*wi.  A monitor with a zero or negative weight is
 *       ignored.
 *   T is NCOR x NMON, and is computed before any errors are added and before any parameters are varied
 */

long global_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations,
                          RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, ELEMENT_LIST **newly_pegged);
long global_coupled_trajcor(CORMON_DATA *CM, STEERING_LIST *SL, TRAJECTORY **traject, long n_iterations,
                            RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, ELEMENT_LIST **newly_pegged);
long one_to_one_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations, RUN *run,
                              LINE_LIST *beamline, double *starting_coord, BEAM *beam, long method);
long thread_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations, RUN *run,
                          LINE_LIST *beamline, double *starting_coord, BEAM *beam, long verbose);
long orbcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **orbit,
                  long n_iterations, double accuracy, double acc_req,
                  long clorb_iter, double clorb_iter_frac, double clorb_frac_mult,
                  long clorb_mult_interval, long clorb_track_for_orbit,
                  RUN *run, LINE_LIST *beamline, double *closed_orbit, double *Cdp, ELEMENT_LIST **newly_pegged);
ELEMENT_LIST *find_useable_moni_corr(int32_t *nmon, int32_t *ncor, long **mon_index,
                                     ELEMENT_LIST ***umoni, ELEMENT_LIST ***ucorr, double **kick_coef, long **sl_index,
                                     short **pegged, double **weight, long plane, STEERING_LIST *SL, RUN *run,
                                     LINE_LIST *beamline, long recircs);
void reorderCorrectorArray(ELEMENT_LIST **ucorr, long *sl_index, double *kick_coef, long ncor);
ELEMENT_LIST *next_element_of_type(ELEMENT_LIST *elem, long type);
ELEMENT_LIST *next_element_of_types(ELEMENT_LIST *elem, long *type, long n_types, long *index, char **corr_name,
                                    long *start_occurence, long *end_occurence, long *occurence_step, double *s_start, double *s_end);
long find_parameter_offset(char *param_name, long elem_type);
long zero_correctors_one_plane(CORMON_DATA *CM, RUN *run, STEERING_LIST *SL);
long zero_correctors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
long zero_hcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
long zero_vcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
#if USE_MPI
long sync_correctors_one_plane(CORMON_DATA *CM, RUN *run, STEERING_LIST *SL);
long sync_correctors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
long sync_hcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
long sync_vcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);
#endif
double rms_value(double *data, long n_data);
long steering_corrector(ELEMENT_LIST *eptr, long plane);
void zero_closed_orbit(TRAJECTORY *clorb, long n);
long find_index(long key, long *list, long n_listed);
long add_steer_elem_to_lists(STEERING_LIST *SL, long plane, char *name, char *item,
                             char *element_type, double tweek, double limit,
                             long start_occurence, long end_occurence, long occurence_step,
                             double s_start, double s_end,
                             LINE_LIST *beamline, RUN *run, long forceQuads, long verbose);
long add_steer_type_to_lists(STEERING_LIST *SL, long plane, long type, char *item, double tweek, double limit,
                             LINE_LIST *beamline, RUN *run, long forceQuadsBends);
double compute_kick_coefficient(ELEMENT_LIST *elem, long plane, long type, double corr_tweek, char *name, char *item, RUN *run);
double noise_value(double xamplitude, double xcutoff, long xerror_type);
void do_response_matrix_output(char *filename, char *type, RUN *run, char *beamline_name, CORMON_DATA *CM,
                               STEERING_LIST *SL, long plane);
void copy_steering_results(CORMON_DATA *CM, STEERING_LIST *SL, CORMON_DATA *CMA, long slot);
void prefillSteeringResultsArray(CORMON_DATA *CM, STEERING_LIST *SL, long iterations);
int remove_pegged_corrector(CORMON_DATA *CMA, CORMON_DATA *CM, STEERING_LIST *SL, ELEMENT_LIST *newly_pegged);
long preemptivelyFindPeggedCorrectors(CORMON_DATA *CM, STEERING_LIST *SL);
void copy_CM_structure(CORMON_DATA *CMA, CORMON_DATA *CM);
void clean_up_CM(CORMON_DATA *CM, short full);

static long rpn_x_mem = -1, rpn_y_mem = -1;
static long usePerturbedMatrix = 0, fixedLengthMatrix = 0;

double getMonitorWeight(ELEMENT_LIST *elem);
double getMonitorCalibration(ELEMENT_LIST *elem, long coord);
double getCorrectorCalibration(ELEMENT_LIST *elem, long coord);
void setup_bpm_output(char *filename, RUN *run);
void dump_bpm_data(TRAJECTORY *traj, long n_elems, char *description, long step);

#define UNIFORM_ERRORS 0
#define GAUSSIAN_ERRORS 1
#define PLUS_OR_MINUS_ERRORS 2
#define N_ERROR_TYPES 3
static char *known_error_type[N_ERROR_TYPES] = {
  "uniform", "gaussian", "plus_or_minus"};

void correction_setup(
  CORRECTION *_correct, /* the underscore is to avoid conflicts with the namelist "correct" */
  NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
#include "correct.h"
  char *item;

  log_entry("correction_setup");

  rpn_x_mem = rpn_create_mem("x", 0);
  rpn_y_mem = rpn_create_mem("y", 0);

  if (_correct->traj) {
    free_czarray_2d((void **)_correct->traj, 3, beamline->n_elems + 1);
    _correct->traj = NULL;
  }

  if (_correct->CMFx) {
    clean_up_CM(_correct->CMFx, 1);
    _correct->CMx = NULL;
    if (_correct->CMAx) {
      clean_up_CM(_correct->CMAx, 0);
      _correct->CMAx = NULL;
    }
  }

  if (_correct->CMFy) {
    clean_up_CM(_correct->CMFy, 1);
    _correct->CMFy = NULL;
    if (_correct->CMAy) {
      clean_up_CM(_correct->CMAy, 0);
      _correct->CMAy = NULL;
    }
  }

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&correct, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);

#if USE_MPI
  if (isSlave) {
    trajectory_output = NULL;
    corrector_output = NULL;
    bpm_output = NULL;
    statistics = NULL;
  }
#endif
  if (echoNamelists)
    print_namelist(stdout, &correct);

  if ((_correct->disable = disable))
    return;

  usePerturbedMatrix = use_perturbed_matrix;
  fixedLengthMatrix = fixed_length_matrix;
  if ((fixed_length_matrix || fixed_length) && checkChangeT(beamline))
    bombElegant("change_t is nonzero on one or more RF cavities. This is incompatible with fixed-length orbit computations.", NULL);

  /* check for valid input data */
  if ((_correct->mode = match_string(mode, correction_mode, N_CORRECTION_MODES, 0)) < 0)
    bombElegant("invalid correction mode", NULL);
  if ((_correct->method = match_string(method, correction_method, N_CORRECTION_METHODS, 0)) < 0)
    bombElegant("invalid correction method", NULL);
  if (_correct->method == COUPLED_CORRECTION && _correct->mode == ORBIT_MODE)
    bombElegant("coupled correction is only available for trajectories", NULL);
  if (corrector_tweek[0] == 0 || corrector_tweek[0] == 0)
    bombElegant("invalid corrector tweek(s)", NULL);
  if (corrector_limit[0] < 0 || corrector_limit[1] < 0 ||
      (corrector_limit[0] && corrector_limit[0] < corrector_tweek[0]) ||
      (corrector_limit[1] && corrector_limit[1] < corrector_tweek[1]))
    bombElegant("invalid corrector_limit(s)", NULL);
  if (correction_fraction[0] <= 0 || correction_fraction[1] <= 0)
    bombElegant("invalid correction_fraction(s)", NULL);
  if (correction_accuracy[0] < 0 || correction_accuracy[1] < 0)
    bombElegant("invalid correction_accuracy(s)", NULL);
  if (trajectory_output)
    setup_orb_traj_output(trajectory_output = compose_filename(trajectory_output, run->rootname),
                          correction_mode[_correct->mode], run);
  if (bpm_output)
    setup_bpm_output(bpm_output = compose_filename(bpm_output, run->rootname), run);
  if (statistics)
    setup_cormon_stats(statistics = compose_filename(statistics, run->rootname), run);
  if (corrector_output)
    setup_corrector_output(corrector_output = compose_filename(corrector_output, run->rootname), run);
  if (closed_orbit_accuracy <= 0)
    bombElegant("closed_orbit_accuracy must be > 0", NULL);
  if (closed_orbit_accuracy_requirement <= 0)
    bombElegant("closed_orbit_accuracy_requirement must be > 0", NULL);
  if (closed_orbit_iteration_fraction <= 0 ||
      closed_orbit_iteration_fraction > 1)
    bombElegant("closed_orbit_iteration_fraction must be on (0, 1]", NULL);
  _correct->clorb_accuracy = closed_orbit_accuracy;
  _correct->clorb_accuracy_requirement = closed_orbit_accuracy_requirement;
  _correct->clorb_iter_fraction = closed_orbit_iteration_fraction;
  _correct->clorb_fraction_multiplier = closed_orbit_fraction_multiplier;
  _correct->clorb_multiplier_interval = closed_orbit_multiplier_interval;
  _correct->clorb_track_for_orbit = closed_orbit_tracking_turns;
  _correct->verbose = verbose;
  _correct->track_before_and_after = track_before_and_after;
  _correct->prezero_correctors = prezero_correctors;
  _correct->use_response_from_computed_orbits = use_response_from_computed_orbits;
  _correct->start_from_centroid = start_from_centroid;
  _correct->use_actual_beam = use_actual_beam;
  _correct->rpn_store_response_matrix = rpn_store_response_matrix;
  if ((_correct->clorb_iterations = closed_orbit_iterations) <= 0)
    bombElegant("closed_orbit_iterations <= 0", NULL);
  if ((_correct->n_iterations = n_iterations) < 0)
    bombElegant("n_iterations < 0", NULL);
  if ((_correct->n_xy_cycles = n_xy_cycles) < 1)
    bombElegant("n_xy_cycles < 1", NULL);
  _correct->minimum_cycles = minimum_cycles;
  _correct->forceAlternation = force_alternation;
  if (threading_divisor[0] <= 1 || threading_divisor[1] <= 1)
    bombElegant("threading_divisors must be >1", NULL);
  _correct->xplane = do_correction[0];
  _correct->yplane = do_correction[1];

  _correct->CMFx = tmalloc(sizeof(*_correct->CMFx));
  _correct->CMFy = tmalloc(sizeof(*_correct->CMFy));
  memset(_correct->CMFx, 0, sizeof(sizeof(*_correct->CMFx)));
  memset(_correct->CMFy, 0, sizeof(sizeof(*_correct->CMFy)));

  _correct->CMFx->n_iterations = n_iterations;
  _correct->CMFy->n_iterations = n_iterations;
  _correct->CMFx->default_tweek = corrector_tweek[0];
  _correct->CMFy->default_tweek = corrector_tweek[1];
  _correct->CMFx->default_threading_divisor = threading_divisor[0];
  _correct->CMFy->default_threading_divisor = threading_divisor[1];
  _correct->CMFx->threading_correctors = threading_correctors[0];
  _correct->CMFy->threading_correctors = threading_correctors[1];
  _correct->CMFx->corr_limit = corrector_limit[0];
  _correct->CMFy->corr_limit = corrector_limit[1];
  _correct->CMFx->corr_fraction = correction_fraction[0];
  _correct->CMFy->corr_fraction = correction_fraction[1];
  _correct->CMFx->corr_accuracy = correction_accuracy[0];
  _correct->CMFy->corr_accuracy = correction_accuracy[1];
  _correct->CMFx->bpm_noise = bpm_noise[0];
  _correct->CMFy->bpm_noise = bpm_noise[1];
  if ((_correct->CMFx->bpm_noise_distribution = match_string(bpm_noise_distribution[0], known_error_type, N_ERROR_TYPES, 0)) < 0)
    bombElegant("unknown noise distribution type", NULL);
  if ((_correct->CMFy->bpm_noise_distribution = match_string(bpm_noise_distribution[1], known_error_type, N_ERROR_TYPES, 0)) < 0)
    bombElegant("unknown noise distribution type", NULL);
  _correct->CMFx->bpm_noise_cutoff = bpm_noise_cutoff[0];
  _correct->CMFy->bpm_noise_cutoff = bpm_noise_cutoff[1];
  _correct->CMFx->fixed_length = _correct->CMFy->fixed_length = fixed_length;
  _correct->response_only = n_iterations == 0;
  _correct->CMFx->T = _correct->CMFy->T = NULL;
  _correct->CMFx->C = _correct->CMFy->C = NULL;
  _correct->CMFx->Cp = _correct->CMFy->Cp = NULL;
  _correct->CMFx->bpmPlane = _correct->CMFx->corrPlane = 0;
  _correct->CMFy->bpmPlane = _correct->CMFy->corrPlane = 1;

  _correct->CMFx->remove_smallest_SVs = remove_smallest_SVs[0];
  _correct->CMFx->auto_limit_SVs = auto_limit_SVs[0];
  _correct->CMFx->Tikhonov_n = Tikhonov_n[0];
  _correct->CMFx->Tikhonov_relative_alpha = Tikhonov_relative_alpha[0];
  _correct->CMFx->keep_largest_SVs = keep_largest_SVs[0];
  if ((_correct->CMFx->minimum_SV_ratio = minimum_SV_ratio[0]) >= 1)
    bombElegant("minimum_SV_ratio should be less than 1 to be meaningful", NULL);

  _correct->CMFy->remove_smallest_SVs = remove_smallest_SVs[1];
  _correct->CMFy->auto_limit_SVs = auto_limit_SVs[1];
  _correct->CMFy->Tikhonov_n = Tikhonov_n[1];
  _correct->CMFy->Tikhonov_relative_alpha = Tikhonov_relative_alpha[1];
  _correct->CMFy->keep_largest_SVs = keep_largest_SVs[1];
  if ((_correct->CMFy->minimum_SV_ratio = minimum_SV_ratio[1]) >= 1)
    bombElegant("minimum_SV_ratio should be less than 1 to be meaningful", NULL);

  _correct->CMFx->remove_pegged = _correct->method==THREAD_CORRECTION?0:remove_pegged[0];
  _correct->CMFy->remove_pegged = _correct->method==THREAD_CORRECTION?0:remove_pegged[1];

  if (verbose)
    fputs("finding correctors/monitors and/or computing correction matrices\n", stdout);

  if (_correct->method != COUPLED_CORRECTION) {
    if (_correct->SLx.n_corr_types == 0) {
      long found = 0;
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_HCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_EHCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "HKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_HVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_EHVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "HKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_QUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_KQUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_DQCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_KSEXT, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);

      cp_str(&item, "XKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_CSBEND, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      if (!found)
        bombElegant("no horizontal steering elements found", NULL);
      if (verbose)
        printf("found %ld horizontal steering elements\n", found);
    }
    if (_correct->SLy.n_corr_types == 0) {
      long found = 0;
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_VCOR, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_EVCOR, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      cp_str(&item, "VKICK");
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_HVCOR, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_EHVCOR, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      cp_str(&item, "VKICK");
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_QUAD, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_KQUAD, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_DQCOR, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_KSEXT, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);

      cp_str(&item, "YKICK");
      found += add_steer_type_to_lists(&_correct->SLy, 2, T_CSBEND, item, _correct->CMFy->default_tweek,
                                       _correct->CMFy->corr_limit, beamline, run, 0);
      if (!found)
        bombElegant("no vertical steering elements found", NULL);
      if (verbose)
        printf("found %ld vertical steering elements\n", found);
    }

    if (_correct->mode == TRAJECTORY_CORRECTION) {
      compute_trajcor_matrices(_correct->CMFx, &_correct->SLx, 0, run, beamline,
                               (_correct->method == THREAD_CORRECTION ? COMPUTE_RESPONSE_FINDONLY : 0) |
                               ((!_correct->response_only && _correct->method == GLOBAL_CORRECTION) ? COMPUTE_RESPONSE_INVERT : 0),
                               _correct->rpn_store_response_matrix);
      compute_trajcor_matrices(_correct->CMFy, &_correct->SLy, 2, run, beamline,
                               (_correct->method == THREAD_CORRECTION ? COMPUTE_RESPONSE_FINDONLY : 0) |
                               ((!_correct->response_only && _correct->method == GLOBAL_CORRECTION) ? COMPUTE_RESPONSE_INVERT : 0),
                               _correct->rpn_store_response_matrix);
    } else if (_correct->mode == ORBIT_CORRECTION) {
      if (!_correct->use_response_from_computed_orbits) {
        compute_orbcor_matrices(_correct->CMFx, &_correct->SLx, 0, run, beamline,
                                (!_correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                (fixed_length_matrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                (verbose ? 0 : COMPUTE_RESPONSE_SILENT), _correct->rpn_store_response_matrix);
        compute_orbcor_matrices(_correct->CMFy, &_correct->SLy, 2, run, beamline,
                                (!_correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                (fixed_length_matrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                (verbose ? 0 : COMPUTE_RESPONSE_SILENT), _correct->rpn_store_response_matrix);
      } else {
#if USE_MPI
        compute_orbcor_matrices1p(_correct, run, beamline,
                                 (!_correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                   (fixed_length_matrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                  (verbose ? 0 : COMPUTE_RESPONSE_SILENT), 0, NULL, NULL);
#else
        compute_orbcor_matrices1(_correct, 0, run, beamline,
                                 (!_correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (fixed_length_matrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                 (verbose ? 0 : COMPUTE_RESPONSE_SILENT),
                                 0, NULL, NULL);
        compute_orbcor_matrices1(_correct, 2, run, beamline,
                                 (!_correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (fixed_length_matrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                 (verbose ? 0 : COMPUTE_RESPONSE_SILENT),
                                 0, NULL, NULL);
#endif
      }
    } else
      bombElegant("something impossible happened (correction_setup)", NULL);
  } else {
    /* Coupled global trajectory correction */
    if (_correct->SLy.n_corr_types != 0)
      bombElegant("For coupled correction, use plane='c' in all steering_elements commands", NULL);
    /* coupled correction uses the SLx and CMx structures only */
    if (_correct->SLx.n_corr_types == 0) {
      long found = 0;
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_HCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_EHCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "HKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_HVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_EHVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "HKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_QUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_KQUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 0, T_DQCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_VCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_EVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "VKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_HVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_EHVCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      cp_str(&item, "VKICK");
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_QUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_KQUAD, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      found += add_steer_type_to_lists(&_correct->SLx, 2, T_DQCOR, item, _correct->CMFx->default_tweek,
                                       _correct->CMFx->corr_limit, beamline, run, 0);
      if (!found)
        bombElegant("no steering elements found", NULL);
    }
    compute_coupled_trajcor_matrices(_correct->CMFx, &_correct->SLx, run, beamline, COMPUTE_RESPONSE_INVERT);
  }

  if (n_iterations != 0) {
    /* allocate space to store before/after data for correctors and monitors */
    _correct->CMFx->kick = (double **)czarray_2d(sizeof(**(_correct->CMFx->kick)), _correct->n_iterations + 1, _correct->CMFx->ncor);
    _correct->CMFx->posi = (double **)czarray_2d(sizeof(**(_correct->CMFx->posi)), _correct->n_iterations + 1, _correct->CMFx->nmon);
    if (_correct->method != COUPLED_CORRECTION) {
      _correct->CMFy->kick = (double **)czarray_2d(sizeof(**(_correct->CMFy->kick)), _correct->n_iterations + 1, _correct->CMFy->ncor);
      _correct->CMFy->posi = (double **)czarray_2d(sizeof(**(_correct->CMFy->posi)), _correct->n_iterations + 1, _correct->CMFy->nmon);
    }

    /* Allocate space to store before/after trajectories/closed orbits.
       * After each correction pass through x and y, the first trajectory is the initial one,
       * the second is after x correction, and the third is after x and y correction.
       */
    _correct->traj = (TRAJECTORY **)czarray_2d(sizeof(**_correct->traj), 3, beamline->n_elems + 1);
  }
  if (_correct->CMFx->remove_pegged) {
    _correct->CMAx = tmalloc(sizeof(*_correct->CMAx));
    memset(_correct->CMAx, 0, sizeof(*_correct->CMAx));
  }
  if (_correct->method != COUPLED_CORRECTION) {
    if (_correct->CMFy->remove_pegged) {
      _correct->CMAy = tmalloc(sizeof(*_correct->CMAy));
      memset(_correct->CMAy, 0, sizeof(*_correct->CMAy));
    }
  }

  if (verbose) {
    if (_correct->method != COUPLED_CORRECTION) {
      printf("there are %" PRId32 " useable horizontal monitors and %" PRId32 " useable horizontal correctors\n",
             _correct->CMFx->nmon, _correct->CMFx->ncor);
      printf("there are %" PRId32 " useable   vertical monitors and %" PRId32 " useable   vertical correctors\n",
             _correct->CMFy->nmon, _correct->CMFy->ncor);
      fflush(stdout);
    } else {
      printf("there are %" PRId32 " useable monitors and %" PRId32 " useable correctors\n",
             _correct->CMFx->nmon, _correct->CMFx->ncor);
    }
  }

  _correct->CMx = _correct->CMFx;
  _correct->CMy = _correct->CMFy;

  log_exit("correction_setup");
}

void clean_up_CM(CORMON_DATA *CM, short full) {
  if (CM) {
    if (CM->T)
      matrix_free(CM->T);
    if (CM->C)
      matrix_free(CM->C);
    if (CM->kick)
      free_czarray_2d((void **)CM->kick, CM->n_iterations + 1, CM->ncor);
    if (full) {
      /* these are shared between the primary and altered structures */
      if (CM->posi)
        free_czarray_2d((void **)CM->posi, CM->n_iterations + 1, CM->nmon);
      if (CM->mon_index)
        tfree(CM->mon_index);
      if (CM->umoni)
        tfree(CM->umoni);
      if (CM->weight)
        tfree(CM->weight);
    }
    if (CM->ucorr)
      tfree(CM->ucorr);
    if (CM->sl_index)
      tfree(CM->sl_index);
    if (CM->kick_coef)
      tfree(CM->kick_coef);
    if (CM->pegged)
      tfree(CM->pegged);
  }
}

void add_steering_element(CORRECTION *correct, LINE_LIST *beamline, RUN *run, NAMELIST_TEXT *nltext) {
#include "steer_elem.h"

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&steering_element, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);

  if (echoNamelists)
    print_namelist(stdout, &steering_element);

  if ((((s_start >= 0 && s_end >= 0) ? 1 : 0) +
       ((start_occurence != 0 && end_occurence != 0) ? 1 : 0) +
       ((after || before) ? 1 : 0)) > 1)
    bombElegant("can't combine start_occurence/end_occurence, s_start/s_end, and after/before---use one method only", NULL);
  if (start_occurence > end_occurence)
    bombElegant("start_occurence > end_occurence", NULL);
  if ((start_occurence != 0 && end_occurence != 0) && occurence_step <= 0)
    bombElegant("occurence_step<=0", NULL);

  if (after || before) {
    ELEMENT_LIST *context;
    context = NULL;
    s_start = -DBL_MAX;
    s_end = DBL_MAX;
    if (after && strlen(after)) {
      if (!(context = find_element(after, &context, beamline->elem))) {
        printf("Element %s not found in beamline.\n", after);
        exitElegant(1);
      }
      s_start = context->end_pos * (1 + 1e-15);
      if (find_element(after, &context, beamline->elem)) {
        printf("Element %s found in beamline more than once.\n", after);
        exitElegant(1);
      }
      if (verbose) {
        printf("'after' element %s found at s = %le m\n", after, s_start);
        fflush(stdout);
      }
    }
    context = NULL;
    if (before && strlen(before)) {
      if (!(context = find_element(before, &context, beamline->elem))) {
        printf("Element %s not found in beamline.\n", before);
        exitElegant(1);
      }
      s_end = context->end_pos * (1 - 1e-15);
      if (find_element(before, &context, beamline->elem)) {
        printf("Element %s found in beamline more than once.\n", before);
        exitElegant(1);
      }
      if (verbose) {
        printf("'before' element %s found at s = %le m\n", before, s_end);
        fflush(stdout);
      }
    }
    if (s_start > s_end)
      bombElegant("'after' element follows 'before' element!", NULL);
  }

  if (limit && (limit < tweek || limit < 0))
    bombElegant("invalid limit specified for steering element", NULL);

  if (plane[0] == 'h' || plane[0] == 'H') {
    if (!add_steer_elem_to_lists(&correct->SLx, 0, name, item, element_type, tweek, limit,
                                 start_occurence, end_occurence, occurence_step, s_start, s_end,
                                 beamline, run, 1, verbose))
      bombElegant("no match to given element name or type (1)", NULL);
  } else if (plane[0] == 'v' || plane[0] == 'V') {
    if (!add_steer_elem_to_lists(&correct->SLy, 2, name, item, element_type, tweek, limit,
                                 start_occurence, end_occurence, occurence_step, s_start, s_end,
                                 beamline, run, 1, verbose))
      bombElegant("no match to given element name or type (2)", NULL);
  } else if (plane[0] == 'c' || plane[0] == 'C') {
    if (!add_steer_elem_to_lists(&correct->SLx, 4, name, item, element_type, tweek, limit,
                                 start_occurence, end_occurence, occurence_step, s_start, s_end,
                                 beamline, run, 1, verbose + 10))
      bombElegant("no match to given element name or type (3)", NULL);
  } else
    bombElegantVA("invalid plane %c specified for steering element", plane[0]);
}

long add_steer_type_to_lists(STEERING_LIST *SL, long plane, long type, char *item, double tweek, double limit,
                             LINE_LIST *beamline, RUN *run, long forceQuads) {
  long found = 0;
  ELEMENT_LIST *context;
  context = beamline->elem;
  while (context && (context = next_element_of_type(context, type))) {
    found += add_steer_elem_to_lists(SL, plane, context->name, item, NULL, tweek, limit,
                                     0, 0, 1, -1, -1, beamline, run, forceQuads, 0);
    context = context->succ;
  }
  return found;
}

long add_steer_elem_to_lists(STEERING_LIST *SL, long plane, char *name, char *item,
                             char *element_type, double tweek, double limit,
                             long start_occurence, long end_occurence, long occurence_step,
                             double s_start, double s_end,
                             LINE_LIST *beamline, RUN *run, long forceQuadsBends, long verbose) {
  ELEMENT_LIST *context;
  long param_number, i, found, n_corr_types_start, notNeeded;

  if (SL->n_corr_types == 0) {
    if (SL->elem)
      tfree(SL->elem);
    SL->elem = NULL;
    if (SL->corr_param)
      tfree(SL->corr_param);
    SL->corr_param = NULL;
    if (SL->corr_tweek)
      tfree(SL->corr_tweek);
    SL->corr_tweek = NULL;
    if (SL->corr_limit)
      tfree(SL->corr_limit);
    SL->corr_limit = NULL;
    if (SL->param_offset)
      tfree(SL->param_offset);
    SL->param_offset = NULL;
    if (SL->param_index)
      tfree(SL->param_index);
    SL->param_index = NULL;
  }

  if (!name && !element_type)
    bombElegant("NULL name and element_type passed to add_steer_elem_to_list", NULL);
  if (name) {
    str_toupper(name);
    if (has_wildcards(name) && strchr(name, '-'))
      name = expand_ranges(name);
  } else
    cp_str(&name, "*");
  if (element_type) {
    str_toupper(element_type);
    if (has_wildcards(element_type) && strchr(element_type, '-'))
      element_type = expand_ranges(element_type);
  }

  if (!item)
    bombElegant("NULL item passed to add_steer_elem_to_list", NULL);
  str_toupper(item);

  context = NULL;
  found = 0;

  n_corr_types_start = SL->n_corr_types;

  while ((context = wfind_element(name, &context, beamline->elem))) {
    if (verbose > 1) {
      printf("Checking %s#%ld.%s at s=%le m for plane %c\n", context->name, context->occurence, item, context->end_pos,
             plane == 0 ? 'x' : (plane == 2 ? 'y' : 'c'));
      fflush(stdout);
    }
    if (element_type &&
        !wild_match(entity_name[context->type], element_type)) {
      if (verbose > 1) {
        printf("Type (%s) doesn't match required type %s for plane %c\n", entity_name[context->type], element_type,
               plane == 0 ? 'x' : (plane == 2 ? 'y' : 'c'));
        fflush(stdout);
      }
      continue;
    }
    if (start_occurence != 0 && end_occurence != 0) {
      if (context->occurence < start_occurence || context->occurence > end_occurence ||
          (context->occurence - start_occurence) % occurence_step != 0) {
        if (verbose > 1) {
          printf("Occurence %ld out of range [%ld, %ld]\n", context->occurence, start_occurence, end_occurence);
          fflush(stdout);
        }
        continue;
      }
    }
    if (s_start >= 0 && context->end_pos < s_start) {
      if (verbose > 1) {
        printf("Position %le before start position %le\n", context->end_pos, s_start);
        fflush(stdout);
      }
      continue;
    }
    if (s_end >= 0 && context->end_pos > s_end) {
      if (verbose > 1) {
        printf("Position %le before end position %le\n", context->end_pos, s_end);
        fflush(stdout);
      }
      continue;
    }

    notNeeded = 1;
    switch (context->type) {
    case T_QUAD:
      if (plane == 0) {
        if (!((QUAD *)(context->p_elem))->xSteering)
          ((QUAD *)(context->p_elem))->xSteering = forceQuadsBends;
        notNeeded = !((QUAD *)(context->p_elem))->xSteering;
      } else if (plane == 2) {
        if (!((QUAD *)(context->p_elem))->ySteering)
          ((QUAD *)(context->p_elem))->ySteering = forceQuadsBends;
        notNeeded = !((QUAD *)(context->p_elem))->ySteering;
      } else {
        if (strcmp(item, "HKICK") == 0 || strcmp(item, "DX") == 0) {
          if (!((QUAD *)(context->p_elem))->xSteering)
            ((QUAD *)(context->p_elem))->xSteering = forceQuadsBends;
          notNeeded = !((QUAD *)(context->p_elem))->xSteering;
        }
        if (strcmp(item, "VKICK") == 0 || strcmp(item, "DY") == 0) {
          if (!((QUAD *)(context->p_elem))->ySteering)
            ((QUAD *)(context->p_elem))->ySteering = forceQuadsBends;
          notNeeded = !((QUAD *)(context->p_elem))->ySteering;
        }
      }
      break;
    case T_KQUAD:
      if (plane == 0) {
        if (!((KQUAD *)(context->p_elem))->xSteering)
          ((KQUAD *)(context->p_elem))->xSteering = forceQuadsBends;
        notNeeded = !((KQUAD *)(context->p_elem))->xSteering;
      } else if (plane == 2) {
        if (!((KQUAD *)(context->p_elem))->ySteering)
          ((KQUAD *)(context->p_elem))->ySteering = forceQuadsBends;
        notNeeded = !((KQUAD *)(context->p_elem))->ySteering;
      } else {
        if (strcmp(item, "HKICK") == 0 || strcmp(item, "DX") == 0) {
          if (!((KQUAD *)(context->p_elem))->xSteering)
            ((KQUAD *)(context->p_elem))->xSteering = forceQuadsBends;
          notNeeded = !((KQUAD *)(context->p_elem))->xSteering;
        }
        if (strcmp(item, "VKICK") == 0 || strcmp(item, "DY") == 0) {
          if (!((KQUAD *)(context->p_elem))->ySteering)
            ((KQUAD *)(context->p_elem))->ySteering = forceQuadsBends;
          notNeeded = !((KQUAD *)(context->p_elem))->ySteering;
        }
      }
      break;
    case T_DQCOR:
      if (plane == 0) {
        if (!((DQCOR *)(context->p_elem))->xSteering)
          ((DQCOR *)(context->p_elem))->xSteering = forceQuadsBends;
        notNeeded = !((DQCOR *)(context->p_elem))->xSteering;
      } else if (plane == 2) {
        if (!((DQCOR *)(context->p_elem))->ySteering)
          ((DQCOR *)(context->p_elem))->ySteering = forceQuadsBends;
        notNeeded = !((DQCOR *)(context->p_elem))->ySteering;
      } else {
        if (strcmp(item, "HKICK") == 0) {
          if (!((DQCOR *)(context->p_elem))->xSteering)
            ((DQCOR *)(context->p_elem))->xSteering = forceQuadsBends;
          notNeeded = !((DQCOR *)(context->p_elem))->xSteering;
        }
        if (strcmp(item, "VKICK") == 0) {
          if (!((DQCOR *)(context->p_elem))->ySteering)
            ((DQCOR *)(context->p_elem))->ySteering = forceQuadsBends;
          notNeeded = !((DQCOR *)(context->p_elem))->ySteering;
        }
      }
      break;
    case T_CSBEND:
      if (plane == 0) {
        if (!((CSBEND *)(context->p_elem))->xSteering)
          ((CSBEND *)(context->p_elem))->xSteering = forceQuadsBends;
        notNeeded = !((CSBEND *)(context->p_elem))->xSteering;
      } else if (plane == 2) {
        if (!((CSBEND *)(context->p_elem))->ySteering)
          ((CSBEND *)(context->p_elem))->ySteering = forceQuadsBends;
        notNeeded = !((CSBEND *)(context->p_elem))->ySteering;
      } else {
        if (strcmp(item, "XKICK") == 0) {
          if (!((CSBEND *)(context->p_elem))->xSteering)
            ((CSBEND *)(context->p_elem))->xSteering = forceQuadsBends;
          notNeeded = !((CSBEND *)(context->p_elem))->xSteering;
        }
        if (strcmp(item, "YKICK") == 0) {
          if (!((CSBEND *)(context->p_elem))->ySteering)
            ((CSBEND *)(context->p_elem))->ySteering = forceQuadsBends;
          notNeeded = !((CSBEND *)(context->p_elem))->ySteering;
        }
      }
      break;
    default:
      notNeeded = 0;
      break;
    }

    if (notNeeded)
      continue;

    for (i = 0; i < n_corr_types_start; i++) {
      if (SL->elem[i] == context && strcmp(SL->corr_param[i], item) == 0)
        break;
    }
    if (i != n_corr_types_start)
      continue;

    if (verbose) {
      printf("Found matching element (%s #%ld at s=%le m).\n", context->name, context->occurence, context->end_pos);
      fflush(stdout);
    }

#ifdef DEBUG
    printf("Adding %s to %c plane steering list\n",
           context->name, plane ? (plane == 4 ? 'c' : 'y') : 'x');
#endif

    SL->elem = trealloc(SL->elem, (SL->n_corr_types + 1) * sizeof(*SL->elem));
    SL->corr_param = trealloc(SL->corr_param, (SL->n_corr_types + 1) * sizeof(*SL->corr_param));
    SL->corr_tweek = trealloc(SL->corr_tweek, (SL->n_corr_types + 1) * sizeof(*SL->corr_tweek));
    SL->corr_limit = trealloc(SL->corr_limit, (SL->n_corr_types + 1) * sizeof(*SL->corr_limit));
    SL->param_offset = trealloc(SL->param_offset, (SL->n_corr_types + 1) * sizeof(*SL->param_offset));
    SL->param_index = trealloc(SL->param_index, (SL->n_corr_types + 1) * sizeof(*SL->param_index));

    SL->elem[SL->n_corr_types] = context;
    cp_str(SL->corr_param + SL->n_corr_types, item);
    SL->corr_tweek[SL->n_corr_types] = tweek;
    SL->corr_limit[SL->n_corr_types] = limit;

    if ((SL->param_index[SL->n_corr_types] = param_number = confirm_parameter(item, context->type)) < 0 ||
        entity_description[context->type].parameter[param_number].type != IS_DOUBLE ||
        (SL->param_offset[SL->n_corr_types] = find_parameter_offset(item, SL->elem[SL->n_corr_types]->type)) < 0) {
      fprintf(stderr, "No such floating-point parameter (%s) for %s (add_steer_elem_to_lists)\n",
              item, context->name);
      exitElegant(1);
    }
    
    SL->n_corr_types += 1;
    found = 1;
  }
  return found;
}

double compute_kick_coefficient(ELEMENT_LIST *elem, long plane, long type, double corr_tweek, char *name, char *item, RUN *run) {
  double value, coef;
  long param_offset = 0, param_number;
  short try_matrix = 0;

  if ((param_number = confirm_parameter(item, type)) < 0 || (param_offset = find_parameter_offset(item, type)) < 0)
    bombElegant("invalid parameter or element type (compute_kick_coefficient)", NULL);

  coef = 0;

  switch (type) {
  case T_HCOR:
  case T_EHCOR:
    if (plane == 0 && param_offset == find_parameter_offset("KICK", type))
      coef = 1.0;
    break;
  case T_VCOR:
  case T_EVCOR:
    if (plane != 0 && param_offset == find_parameter_offset("KICK", type))
      coef = 1.0;
    break;
  case T_HVCOR:
  case T_EHVCOR:
    if (plane == 0 && param_offset == find_parameter_offset("HKICK", type))
      coef = 1.0;
    if (plane != 0 && param_offset == find_parameter_offset("VKICK", type))
      coef = 1.0;
    break;
  case T_CSBEND:
    if (plane == 0 && param_offset == find_parameter_offset("XKICK", type))
      coef = 1.0;
    else if (plane != 0 && param_offset == find_parameter_offset("YKICK", type))
      coef = 1.0;
    else
      try_matrix = 1;
    break;
  default:
    try_matrix = 1;
    break;
  }
  if (try_matrix) {
    if (entity_description[elem->type].flags & HAS_MATRIX) {
      VMATRIX *M1, *M2, *M0;
      M0 = elem->matrix;
      elem->matrix = NULL;

      value = *((double *)(elem->p_elem + param_offset));
      *((double *)(elem->p_elem + param_offset)) += corr_tweek;
      M1 = compute_matrix(elem, run, NULL);
      elem->matrix = NULL;

      *((double *)(elem->p_elem + param_offset)) -= 2 * corr_tweek;
      M2 = compute_matrix(elem, run, NULL);

      if (plane == 0)
        coef = (M1->C[1] - M2->C[1]) / (2 * corr_tweek);
      else
        coef = (M1->C[3] - M2->C[3]) / (2 * corr_tweek);
#ifdef DEBUG
      printf("computed kick coefficient for %s.%s: %g rad/%s\n",
             name, item, coef, entity_description[type].parameter[param_number].unit);
#endif
      free_matrices(M1);
      tfree(M1);
      M1 = NULL;
      free_matrices(M2);
      tfree(M2);
      M2 = NULL;
      *((double *)(elem->p_elem + param_offset)) = value;
      elem->matrix = M0;
    } else {
      printf("error: element %s does not have a matrix--can't be used for steering.\n",
             elem->name);
      fflush(stdout);
      exitElegant(1);
    }
  }

  return coef;
}

long do_correction(CORRECTION *correct, RUN *run, LINE_LIST *beamline, double *starting_coord,
                   BEAM *beam, long sim_step, unsigned long flags) {
  long i, i_cycle, x_failed, y_failed, n_x_iter_taken = 0, n_y_iter_taken = 0, bombed = 0, final_traj;
  double *closed_orbit, rms_before, rms_after, *Cdp;
  ELEMENT_LIST *newly_pegged;
#if USE_MPI
  /* do_correction may need differnt tracking mode, but it should not change the tracking mode */
  long notSinglePart_orig = notSinglePart;
  long partOnMaster_orig = partOnMaster;
#endif

  log_entry("do_correction");

  if (correct->response_only)
    return 1;

  if (correct->disable)
    return 1;

#if SDDS_MPI_IO
  if (correct->start_from_centroid && starting_coord && beam && beam->n_to_track_total && flags & INITIAL_CORRECTION) {
    compute_centroids(starting_coord, beam->particle, beam->n_to_track);
#else
  if (correct->start_from_centroid && starting_coord && beam && beam->n_to_track && flags & INITIAL_CORRECTION) {
    compute_centroids(starting_coord, beam->particle, beam->n_to_track);
#endif
  }

#if SDDS_MPI_IO
  /* We need choose different modes depending on different beams */
  if (!correct->use_actual_beam) {
    notSinglePart = 0;
    partOnMaster = 1;
  }
#endif

  closed_orbit = starting_coord; /* for return of closed orbit at starting point */

  if (correct->verbose && correct->n_iterations >= 1 && correct->n_xy_cycles > 0 && !(flags & NO_OUTPUT_CORRECTION)) {
    if (correct->CMFx->fixed_length && correct->mode == ORBIT_CORRECTION) {
      fputs(" plane   iter     rms kick     rms pos.    max kick     max pos.   mom.offset\n", stdout);
      fputs("                    mrad          mm         mrad          mm           %\n", stdout);
      fputs("-------  ----    ----------   ---------   ----------   ----------  ----------\n", stdout);
    } else {
      fputs(" plane   iter     rms kick     rms pos.    max kick     max pos.\n", stdout);
      fputs("                    mrad          mm         mrad          mm\n", stdout);
      fputs("-------  ----    ----------   ---------   ----------   ----------\n", stdout);
    }
  }

  if (correct->prezero_correctors && flags & INITIAL_CORRECTION) {
    long changed;
    if (beamline->elem_recirc)
      changed = zero_correctors(beamline->elem_recirc, run, correct);
    else
      changed = zero_correctors(beamline->elem, run, correct);
    if (changed && beamline->matrix) {
      free_matrices(beamline->matrix);
      tfree(beamline->matrix);
      beamline->matrix = NULL;
    }
  }

#if USE_MPI
  if (last_optimize_function_call) { /* Synchronize corrector's values after exiting optimization loop across all the processors to reproduce the optimal result */
    if (beamline->elem_recirc)
      sync_correctors(beamline->elem_recirc, run, correct);
    else
      sync_correctors(beamline->elem, run, correct);
  }
#endif
  correct->CMx = correct->CMFx;
  correct->CMy = correct->CMFy;
  memset(correct->CMFx->pegged, 0, correct->CMFx->ncor * sizeof(*correct->CMFx->pegged));
  memset(correct->CMFy->pegged, 0, correct->CMFy->ncor * sizeof(*correct->CMFy->pegged));

  correct->CMx->n_cycles_done = correct->CMy->n_cycles_done = 0;
  final_traj = 1;
  switch (correct->mode) {
  case TRAJECTORY_CORRECTION:
    x_failed = y_failed = 0;
    if (usePerturbedMatrix) {
      if (!(correct->CMx->nmon == 0 || correct->CMx->ncor == 0) && correct->xplane)
        compute_trajcor_matrices(correct->CMx, &correct->SLx, 0, run, beamline,
                                 (correct->method == THREAD_CORRECTION ? COMPUTE_RESPONSE_FINDONLY : 0) |
                                   (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (flags & NO_OUTPUT_CORRECTION ? COMPUTE_RESPONSE_SILENT : 0),
                                 correct->rpn_store_response_matrix);
      if (!(correct->CMy->nmon == 0 || correct->CMy->ncor == 0) && correct->yplane)
        compute_trajcor_matrices(correct->CMy, &correct->SLy, 2, run, beamline,
                                 (correct->method == THREAD_CORRECTION ? COMPUTE_RESPONSE_FINDONLY : 0) |
                                   (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (flags & NO_OUTPUT_CORRECTION ? COMPUTE_RESPONSE_SILENT : 0),
                                 correct->rpn_store_response_matrix);
    }
    for (i_cycle = 0; i_cycle < correct->n_xy_cycles; i_cycle++) {
      final_traj = 1;
      prefillSteeringResultsArray(correct->CMx, &correct->SLx, correct->n_iterations);
      if (!x_failed && correct->CMx->ncor && correct->CMx->nmon && correct->xplane) {
        if (preemptivelyFindPeggedCorrectors(correct->CMx, &correct->SLx)) {
          remove_pegged_corrector(correct->CMAx, correct->CMFx, &correct->SLx, NULL);
          correct->CMx = correct->CMAx;
        }
        if (correct->CMx && (correct->method == THREAD_CORRECTION || correct->CMx->C) && correct->CMx->ncor) {
          newly_pegged = NULL;
          switch (correct->method) {
          case COUPLED_CORRECTION:
            if ((n_x_iter_taken = global_coupled_trajcor(correct->CMx, &correct->SLx, correct->traj, correct->n_iterations,
                                                         run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL), &newly_pegged)) < 0)
              return 0;
            break;
          case GLOBAL_CORRECTION:
            if ((n_x_iter_taken = global_trajcor_plane(correct->CMx, &correct->SLx, 0, correct->traj, correct->n_iterations,
                                                       run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL), &newly_pegged)) < 0)
              return 0;
            break;
          case ONE_TO_ONE_CORRECTION:
          case ONE_TO_BEST_CORRECTION:
          case ONE_TO_NEXT_CORRECTION:
            n_x_iter_taken = one_to_one_trajcor_plane(correct->CMx, &correct->SLx, 0, correct->traj, correct->n_iterations,
                                                      run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL),
                                                      correct->method);
            break;
          case THREAD_CORRECTION:
            n_x_iter_taken = thread_trajcor_plane(correct->CMx, &correct->SLx, 0, correct->traj, correct->n_iterations,
                                                  run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL), correct->verbose);
            break;
          default:
            bombElegant("Invalid x trajectory correction mode---this should never happen!", NULL);
            break;
          }
          if (newly_pegged) {
            printf("One or more correctors newly pegged\n");
          }
          if (correct->CMx != correct->CMFx)
            copy_steering_results(correct->CMFx, &correct->SLx, correct->CMx, n_x_iter_taken);
          correct->CMFx->n_cycles_done = i_cycle + 1;
          if (correct->n_iterations < 1)
            break;
          rms_before = rms_value(correct->CMFx->posi[0], correct->CMFx->nmon);
          rms_after = rms_value(correct->CMFx->posi[correct->n_iterations], correct->CMFx->nmon);
#if defined(IEEE_MATH)
          if (isnan(rms_before) || isnan(rms_after) || isinf(rms_before) || isinf(rms_after)) {
            x_failed = 1;
            if (correct->verbose)
              fputs("horizontal trajectory diverged--setting correctors to zero\n", stdout);
            zero_hcorrectors(beamline->elem, run, correct);
          } else
#endif
            if (rms_before <= (rms_after + correct->CMFx->corr_accuracy) && i_cycle > correct->minimum_cycles &&
                correct->method != THREAD_CORRECTION) {
            x_failed = 1;
            if (correct->verbose && !correct->forceAlternation)
              fputs("trajectory not improved--discontinuing horizontal correction\n", stdout);
          }
          if (!(flags & NO_OUTPUT_CORRECTION)) {
            dump_cormon_stats(correct->verbose,
                              correct->method == COUPLED_CORRECTION ? 4 : 0, correct->CMFx->kick,
                              correct->CMFx->ncor, correct->CMFx->posi, correct->CMFx->nmon, NULL,
                              n_x_iter_taken, i_cycle, i_cycle == correct->n_xy_cycles - 1 || x_failed,
                              sim_step, !(flags & FINAL_CORRECTION));
            /*
              if ((flags&FINAL_CORRECTION) && (i_cycle==correct->n_xy_cycles-1 || x_failed))
              dump_corrector_data(correct->CMFx, &correct->SLx, correct->n_iterations, "horizontal", sim_step);
            */
          }
          if (newly_pegged && correct->CMx->remove_pegged && correct->CMx->n_cycles_done != correct->n_xy_cycles) {
            /* Compute new matrices for next iteratoin */
            fputs("Recomputing inverse response matrix for x plane to remove pegged corrector(s)\n", stdout);
            remove_pegged_corrector(correct->CMAx, correct->CMFx, &correct->SLx, newly_pegged);
            correct->CMx = correct->CMAx;
          }
          if (!(flags & NO_OUTPUT_CORRECTION) && (flags & INITIAL_CORRECTION) && i_cycle == 0 && ((correct->CMFx->ncor && correct->CMFx->nmon) || (correct->CMFy->ncor && correct->CMFy->nmon)))
            dump_orb_traj(correct->traj[0], beamline->n_elems, "uncorrected", sim_step);
        } else
          x_failed = 1;
      }
      prefillSteeringResultsArray(correct->CMy, &correct->SLy, correct->n_iterations);
      if (correct->method != COUPLED_CORRECTION) {
        if (!y_failed && correct->CMy->ncor && correct->CMy->nmon && correct->yplane) {
          final_traj = 2;
          if (preemptivelyFindPeggedCorrectors(correct->CMy, &correct->SLy)) {
            remove_pegged_corrector(correct->CMAy, correct->CMFy, &correct->SLy, NULL);
            correct->CMy = correct->CMAy;
          }
          if (correct->CMy && (correct->method == THREAD_CORRECTION || correct->CMy->C) && correct->CMy->ncor) {
            newly_pegged = NULL;
            switch (correct->method) {
            case GLOBAL_CORRECTION:
              if ((n_y_iter_taken = global_trajcor_plane(correct->CMy, &correct->SLy, 2, correct->traj + 1, correct->n_iterations,
                                                         run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL), &newly_pegged)) < 0)
                return 0;
              break;
            case ONE_TO_ONE_CORRECTION:
            case ONE_TO_BEST_CORRECTION:
            case ONE_TO_NEXT_CORRECTION:
              n_y_iter_taken = one_to_one_trajcor_plane(correct->CMy, &correct->SLy, 2, correct->traj + 1, correct->n_iterations,
                                                        run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL),
                                                        correct->method);
              break;
            case THREAD_CORRECTION:
              n_y_iter_taken = thread_trajcor_plane(correct->CMy, &correct->SLy, 2, correct->traj + 1, correct->n_iterations,
                                                    run, beamline, starting_coord, (correct->use_actual_beam ? beam : NULL),
                                                    correct->verbose);
              break;
            default:
              bombElegant("Invalid y trajectory correction mode---this should never happen!", NULL);
              break;
            }
            if (newly_pegged) {
              printf("One or more correctors newly pegged\n");
            }
            if (correct->CMy != correct->CMFy)
              copy_steering_results(correct->CMFy, &correct->SLy, correct->CMy, n_y_iter_taken);
            correct->CMFy->n_cycles_done = i_cycle + 1;
            rms_before = rms_value(correct->CMFy->posi[0], correct->CMFy->nmon);
            rms_after = rms_value(correct->CMFy->posi[correct->n_iterations], correct->CMFy->nmon);
#if defined(IEEE_MATH)
            if (isnan(rms_before) || isnan(rms_after) || isinf(rms_before) || isinf(rms_after)) {
              y_failed = 1;
              if (correct->verbose)
                fputs("vertical trajectory diverged--setting correctors to zero\n", stdout);
              zero_vcorrectors(beamline->elem, run, correct);
            } else
#endif
              if (rms_before <= (rms_after + correct->CMFy->corr_accuracy) && i_cycle > correct->minimum_cycles &&
                  correct->method != THREAD_CORRECTION) {
              y_failed = 1;
              if (correct->verbose && !correct->forceAlternation)
                fputs("trajectory not improved--discontinuing vertical correction\n", stdout);
            }
            if (!(flags & NO_OUTPUT_CORRECTION)) {
              dump_cormon_stats(correct->verbose, 2, correct->CMFy->kick,
                                correct->CMFy->ncor, correct->CMFy->posi, correct->CMFy->nmon, NULL,
                                n_y_iter_taken, i_cycle, i_cycle == correct->n_xy_cycles - 1 || y_failed,
                                sim_step, !(flags & FINAL_CORRECTION));
              /*
                if ((flags&FINAL_CORRECTION) && (i_cycle==correct->n_xy_cycles-1 || y_failed))
                dump_corrector_data(correct->CMFy, &correct->SLy, correct->n_iterations, "vertical", sim_step);
              */
            }
            if (newly_pegged && correct->CMy->remove_pegged && correct->CMy->n_cycles_done != correct->n_xy_cycles) {
              /* Compute new matrices for next iteration */
              fputs("Recomputing inverse response matrix for x plane to remove pegged corrector(s)\n", stdout);
              remove_pegged_corrector(correct->CMAy, correct->CMFy, &correct->SLy, newly_pegged);
              correct->CMy = correct->CMAy;
            }
          } else
            y_failed = 1;
        }
      }
      if ((x_failed && y_failed) || ((x_failed || y_failed) && correct->forceAlternation)) {
        if (correct->verbose && !(flags & NO_OUTPUT_CORRECTION))
          fputs("trajectory correction discontinued\n", stdout);
        break;
      }
      if (correct->forceAlternation)
        x_failed = y_failed = 0;
    }
    if (!(flags & NO_OUTPUT_CORRECTION) && (flags & FINAL_CORRECTION) &&
        ((correct->CMFx->ncor && correct->CMFx->nmon) || (correct->CMFy->ncor && correct->CMFy->nmon))) {
      dump_orb_traj(correct->traj[final_traj], beamline->n_elems, "corrected", sim_step);
      dump_bpm_data(correct->traj[final_traj], beamline->n_elems, "corrected", sim_step);
      if (correct->method != COUPLED_CORRECTION) {
        dump_corrector_data(correct->CMFx, &correct->SLx, n_x_iter_taken, "horizontal", sim_step);
        dump_corrector_data(correct->CMFy, &correct->SLy, n_y_iter_taken, "vertical", sim_step);
      } else
        dump_corrector_data(correct->CMFx, &correct->SLx, n_x_iter_taken, "coupled", sim_step);
    }
    if (starting_coord)
      for (i = 0; i < 6; i++)
        starting_coord[i] = 0; /* don't want to seem to be returning a closed orbit here */
    bombed = 0;
    break;
  case ORBIT_CORRECTION:
    if (correct->CMx->fixed_length)
      Cdp = tmalloc(sizeof(*Cdp) * (correct->n_iterations + 1));
    else
      Cdp = NULL;
    x_failed = y_failed = bombed = 0;
    if (usePerturbedMatrix) {
      if (correct->verbose && !(flags & NO_OUTPUT_CORRECTION))
        printf("Computing orbit correction matrices\n");
      if (!(correct->CMx->nmon == 0 || correct->CMx->ncor == 0) && correct->xplane) {
        if (!correct->use_response_from_computed_orbits)
          compute_orbcor_matrices(correct->CMx, &correct->SLx, 0, run, beamline,
                                  (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                  (fixedLengthMatrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                  (correct->verbose && !(flags & NO_OUTPUT_CORRECTION) ? 0 : COMPUTE_RESPONSE_SILENT),
                                  correct->rpn_store_response_matrix);
        else
          compute_orbcor_matrices1(correct, 0, run, beamline,
                                 (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (fixedLengthMatrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                 (correct->verbose && !(flags & NO_OUTPUT_CORRECTION) ? 0 : COMPUTE_RESPONSE_SILENT),
                                 0, NULL, NULL);
      }
      if (!(correct->CMy->nmon == 0 || correct->CMy->ncor == 0) && correct->yplane) {
        if (!correct->use_response_from_computed_orbits)
          compute_orbcor_matrices(correct->CMy, &correct->SLy, 2, run, beamline,
                                  (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                    (fixedLengthMatrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                  (correct->verbose && !(flags & NO_OUTPUT_CORRECTION) ? 0 : COMPUTE_RESPONSE_SILENT),
                                  correct->rpn_store_response_matrix);
        else
          compute_orbcor_matrices1(correct, 2, run, beamline,
                                 (!correct->response_only ? COMPUTE_RESPONSE_INVERT : 0) |
                                 (fixedLengthMatrix ? COMPUTE_RESPONSE_FIXEDLENGTH : 0) |
                                 (correct->verbose && !(flags & NO_OUTPUT_CORRECTION) ? 0 : COMPUTE_RESPONSE_SILENT),
                                 0, NULL, NULL);
      }
    }

    for (i_cycle = 0; i_cycle < correct->n_xy_cycles; i_cycle++) {
      final_traj = 1;
      prefillSteeringResultsArray(correct->CMx, &correct->SLx, correct->n_iterations);
      if (!x_failed && correct->CMx->ncor && correct->CMx->nmon && correct->xplane) {
        if (preemptivelyFindPeggedCorrectors(correct->CMx, &correct->SLx)) {
          remove_pegged_corrector(correct->CMAx, correct->CMFx, &correct->SLx, NULL);
          correct->CMx = correct->CMAx;
        }
        if (correct->CMx && correct->CMx->C && correct->CMx->ncor) {
          newly_pegged = NULL;
          if ((n_x_iter_taken = orbcor_plane(correct->CMx,
                                             &correct->SLx,
                                             0, correct->traj, correct->n_iterations, correct->clorb_accuracy,
                                             correct->clorb_accuracy_requirement,
                                             correct->clorb_iterations,
                                             correct->clorb_iter_fraction,
                                             correct->clorb_fraction_multiplier,
                                             correct->clorb_multiplier_interval,
                                             correct->clorb_track_for_orbit,
                                             run, beamline,
                                             closed_orbit, Cdp, &newly_pegged)) < 0) {
            printf("Horizontal correction has failed.\n");
            fflush(stdout);
            if (correct->CMx != correct->CMFx)
              copy_steering_results(correct->CMFx, &correct->SLx, correct->CMx, correct->n_iterations);
            bombed = 1;
            break;
          }
          if (newly_pegged) {
            printf("One or more correctors newly pegged\n");
          }
          if (correct->CMx != correct->CMFx)
            copy_steering_results(correct->CMFx, &correct->SLx, correct->CMx, n_x_iter_taken);
          correct->CMFx->n_cycles_done = i_cycle + 1;
          if (correct->n_iterations < 1)
            break;
          rms_before = rms_value(correct->CMFx->posi[0], correct->CMFx->nmon);
          rms_after = rms_value(correct->CMFx->posi[n_x_iter_taken], correct->CMFx->nmon);
#if defined(IEEE_MATH)
          if (isnan(rms_before) || isnan(rms_after) || isinf(rms_before) || isinf(rms_after)) {
            x_failed = 1;
            if (correct->verbose)
              fputs("horizontal orbit diverged--setting correctors to zero\n", stdout);
            zero_hcorrectors(beamline->elem, run, correct);
          } else
#endif
            if (rms_before <= (rms_after + correct->CMFx->corr_accuracy) && i_cycle >= correct->minimum_cycles) {
            x_failed = 1;
            if (correct->verbose && !correct->forceAlternation)
              fputs("orbit not improved--discontinuing horizontal correction\n", stdout);
          }
          if (!(flags & NO_OUTPUT_CORRECTION)) {
            dump_cormon_stats(correct->verbose, 0, correct->CMFx->kick,
                              correct->CMFx->ncor, correct->CMFx->posi, correct->CMFx->nmon, Cdp,
                              n_x_iter_taken, i_cycle, i_cycle == correct->n_xy_cycles - 1 || x_failed,
                              sim_step, !(flags & FINAL_CORRECTION));
            /*
              if ((flags&FINAL_CORRECTION) && (i_cycle==correct->n_xy_cycles-1 || x_failed)) 
              dump_corrector_data(correct->CMFx, &correct->SLx, correct->n_iterations, "horizontal", sim_step);
            */
          }
          if (!(flags & NO_OUTPUT_CORRECTION) && (flags & INITIAL_CORRECTION) && i_cycle == 0 &&
              ((correct->CMFx->ncor && correct->CMFx->nmon) || (correct->CMFy->ncor && correct->CMFy->nmon)))
            dump_orb_traj(correct->traj[0], beamline->n_elems, "uncorrected", sim_step);
          if (newly_pegged && correct->CMx->remove_pegged && correct->CMx->n_cycles_done != correct->n_xy_cycles) {
            /* Compute new matrices for next iteration */
            fputs("Recomputing inverse response matrix for x plane to remove pegged corrector(s)\n", stdout);
            remove_pegged_corrector(correct->CMAx, correct->CMFx, &correct->SLx, newly_pegged);
            correct->CMx = correct->CMAx;
          }
        } else
          x_failed = 1;
      }
      prefillSteeringResultsArray(correct->CMy, &correct->SLy, correct->n_iterations);
      if (!y_failed && correct->CMy->ncor && correct->CMy->nmon && correct->yplane) {
        final_traj = 2;
        if (preemptivelyFindPeggedCorrectors(correct->CMy, &correct->SLy)) {
          remove_pegged_corrector(correct->CMAy, correct->CMFy, &correct->SLy, NULL);
          correct->CMy = correct->CMAy;
        }
        if (correct->CMy && correct->CMy->C && correct->CMy->ncor) {
          newly_pegged = NULL;
          if ((n_y_iter_taken = orbcor_plane(correct->CMy,
                                             &correct->SLy,
                                             2, correct->traj + 1,
                                             correct->n_iterations, correct->clorb_accuracy,
                                             correct->clorb_accuracy_requirement,
                                             correct->clorb_iterations,
                                             correct->clorb_iter_fraction,
                                             correct->clorb_fraction_multiplier,
                                             correct->clorb_multiplier_interval,
                                             correct->clorb_track_for_orbit,
                                             run, beamline,
                                             closed_orbit, Cdp, &newly_pegged)) < 0) {
            bombed = 1;
            if (correct->CMy != correct->CMFy)
              copy_steering_results(correct->CMFy, &correct->SLy, correct->CMy, correct->n_iterations);
            printf("Vertical correction has failed.\n");
            fflush(stdout);
            break;
          }
          if (newly_pegged) {
            printf("One or more correctors newly pegged\n");
          }
          if (correct->CMFy != correct->CMy)
            copy_steering_results(correct->CMFy, &correct->SLy, correct->CMy, n_y_iter_taken);
          correct->CMFy->n_cycles_done = i_cycle + 1;
          rms_before = rms_value(correct->CMFy->posi[0], correct->CMFy->nmon);
          rms_after = rms_value(correct->CMFy->posi[n_y_iter_taken], correct->CMFy->nmon);
#if defined(IEEE_MATH)
          if (isnan(rms_before) || isnan(rms_after) || isinf(rms_before) || isinf(rms_after)) {
            y_failed = 1;
            if (correct->verbose)
              fputs("vertical orbit diverged--setting correctors to zero\n", stdout);
            zero_vcorrectors(beamline->elem, run, correct);
          } else
#endif
            if (rms_before <= (rms_after + correct->CMy->corr_accuracy) && i_cycle >= correct->minimum_cycles) {
            y_failed = 1;
            if (correct->verbose && !correct->forceAlternation)
              fputs("orbit not improved--discontinuing vertical correction\n", stdout);
          }
          if (!(flags & NO_OUTPUT_CORRECTION)) {
            dump_cormon_stats(correct->verbose, 2, correct->CMFy->kick,
                              correct->CMFy->ncor, correct->CMFy->posi, correct->CMFy->nmon, Cdp,
                              n_y_iter_taken, i_cycle, i_cycle == correct->n_xy_cycles - 1 || y_failed,
                              sim_step, !(flags & FINAL_CORRECTION));
            /*
              if ((flags&FINAL_CORRECTION) && (i_cycle==correct->n_xy_cycles-1 || y_failed))
              dump_corrector_data(correct->CMFy, &correct->SLy, correct->n_iterations, "vertical", sim_step);
            */
          }
          if (newly_pegged && correct->CMFy->remove_pegged && correct->CMFy->n_cycles_done != correct->n_xy_cycles) {
            /* Compute new matrices for next iteratoin */
            fputs("Recomputing inverse response matrix for y plane to remove pegged corrector(s)\n", stdout);
            remove_pegged_corrector(correct->CMAy, correct->CMFy, &correct->SLy, newly_pegged);
            correct->CMy = correct->CMAy;
          }
        } else
          y_failed = 1;
      }
      if ((x_failed && y_failed) || ((x_failed || y_failed) && correct->forceAlternation)) {
        if (correct->verbose && !(flags & NO_OUTPUT_CORRECTION))
          fputs("orbit correction discontinued\n", stdout);
        break;
      }
      if (correct->forceAlternation)
        x_failed = y_failed = 0;
    }
    if (!(flags & NO_OUTPUT_CORRECTION) && !bombed && (flags & FINAL_CORRECTION) &&
        ((correct->CMFx->ncor && correct->CMFx->nmon) || (correct->CMFy->ncor && correct->CMFy->nmon))) {
      dump_orb_traj(correct->traj[final_traj], beamline->n_elems, "corrected", sim_step);
      dump_bpm_data(correct->traj[final_traj], beamline->n_elems, "corrected", sim_step);
      dump_corrector_data(correct->CMFx, &correct->SLx, n_x_iter_taken, "horizontal", sim_step);
      dump_corrector_data(correct->CMFy, &correct->SLy, n_y_iter_taken, "vertical", sim_step);
    }
    break;
  }

  beamline->closed_orbit = correct->traj[final_traj];

#if USE_MPI
  notSinglePart = notSinglePart_orig;
  partOnMaster = partOnMaster_orig;
#endif

  log_exit("do_correction");
  return (!bombed);
}

void compute_trajcor_matrices(CORMON_DATA *CM, STEERING_LIST *SL, long coord, RUN *run, LINE_LIST *beamline, unsigned long flags,
                              short rpn_store_response_matrix) {
  ELEMENT_LIST *corr;
  TRAJECTORY *traj0, *traj1;
  long kick_offset, i_corr, i_moni, i;
  long n_part;
  double **one_part, p, p0, kick0, corr_tweek, corrCalibration, *moniCalibration, W0 = 0.0;
  double conditionNumber;
  VMATRIX *save;

  find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index,
                         &CM->umoni, &CM->ucorr, &CM->kick_coef, &CM->sl_index,
                         &CM->pegged, &CM->weight, coord, SL, run, beamline, 0);

  /* We don't need these coefficients since the response matrix includes them */
  for (i_corr = 0; i_corr < CM->ncor; i_corr++)
    CM->kick_coef[i_corr] = 1;

#ifdef DEBUG
  for (i = 0; i < CM->ncor; i++) {
    long sl_index;
    sl_index = CM->sl_index[i];
    printf("Corrector %ld: %s#%ld at s=%le, sl_index = %ld, kick_coef = %le, type = %s/%s, param_offset = %ld, param_name = %s\n",
           i, CM->ucorr[i]->name, CM->ucorr[i]->occurence, CM->ucorr[i]->end_pos,
           CM->sl_index[i], CM->kick_coef[i],
           entity_name[SL->elem[sl_index]->type], entity_name[CM->ucorr[i]->type],
           SL->param_offset[sl_index],
           entity_description[CM->ucorr[i]->type].parameter[SL->param_index[sl_index]].name);
  }
#endif

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0) {
    if (coord == 0)
      printWarning("Trajectory correction may be unstable (consider use of SV controls).", "More correctors than monitors for x plane.");
    else
      printWarning("Trajectory correction may be unstable (consider use of SV controls).", "More correctors than monitors for y plane.");
  }
  if (CM->ncor == 0) {
    if (coord == 0)
      printWarning("No trajectory correction done for x plane.", "No correctors for x plane.");
    else
      printWarning("No trajectory correction done for y plane.", "No correctors for y plane.");
    return;
  }
  if (CM->nmon == 0) {
    if (coord == 0)
      printWarning("No trajectory correction done for x plane.", "No monitors for x plane.");
    else
      printWarning("No trajectory correction done for y plane.", "No monitors for y plane.");
    CM->ncor = 0;
    return;
  }

  if (flags & COMPUTE_RESPONSE_FINDONLY)
    return;

  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    printf("computing response matrix...\n");
    fflush(stdout);
    report_stats(stdout, "start");
  }

  /* allocate matrices for this plane */
  if (CM->C)
    matrix_free(CM->C);
  CM->C = matrix_get(CM->nmon, CM->ncor); /* Response matrix */
  if (CM->Cp)
    matrix_free(CM->Cp);
  CM->Cp = matrix_get(CM->nmon, CM->ncor); /* Slope response matrix */
  if (CM->T)
    matrix_free(CM->T);
  CM->T = NULL;

  /* arrays for trajectory data */
  traj0 = tmalloc(sizeof(*traj0) * beamline->n_elems);
  traj1 = tmalloc(sizeof(*traj1) * beamline->n_elems);
  one_part = (double **)czarray_2d(sizeof(**one_part), 1, totalPropertiesPerParticle);

  /* find initial trajectory */
  p = p0 = sqrt(sqr(run->ideal_gamma) - 1);
  n_part = 1;
  fill_double_array(*one_part, totalPropertiesPerParticle, 0.0);
  if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   traj0, run, 0,
                   TEST_PARTICLES + TIME_DEPENDENCE_OFF, 1, 0, NULL, NULL, NULL, NULL, NULL))
    bombElegant("tracking failed for test particle (compute_trajcor_matrices())", NULL);

  /* set up weight matrix and monitor calibration array */
  moniCalibration = tmalloc(sizeof(*moniCalibration) * CM->nmon);
  CM->equalW = 1;
  for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
    CM->weight[i_moni] = getMonitorWeight(CM->umoni[i_moni]);
    if (!i_moni)
      W0 = CM->weight[i_moni];
    else if (W0 != CM->weight[i_moni])
      CM->equalW = 0;
    moniCalibration[i_moni] = getMonitorCalibration(CM->umoni[i_moni], coord);
  }

  for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
    corr = CM->ucorr[i_corr];
    kick_offset = SL->param_offset[CM->sl_index[i_corr]];
    corr_tweek = SL->corr_tweek[CM->sl_index[i_corr]];

    /* record value of corrector */
    kick0 = *((double *)(corr->p_elem + kick_offset));

    /* change the corrector by corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 + corr_tweek;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
#ifdef DEBUG
    printf("corrector %s tweeked to %e (type=%s, name=%s, offset=%ld)\n", corr->name, *((double *)(corr->p_elem + kick_offset)),
           entity_name[corr->type],
           entity_description[corr->type].parameter[SL->param_index[CM->sl_index[i_corr]]].name,
           kick_offset);
    fflush(stdout);
#endif

    if (corr->matrix) {
      save = corr->matrix;
      corr->matrix = NULL;
    } else
      save = NULL;
    compute_matrix(corr, run, NULL);
#ifdef DEBUG
    print_matrices(stdout, "*** corrector matrix:", corr->matrix);
#endif

    /* track with positively-tweeked corrector */
    p = p0;
    n_part = 1;
    fill_double_array(*one_part, totalPropertiesPerParticle, 0.0);
    if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                     traj1, run, 0, TEST_PARTICLES + TIME_DEPENDENCE_OFF, 1, 0, NULL, NULL, NULL, NULL, NULL))
      bombElegant("tracking failed for test particle (compute_trajcor_matrices())", NULL);

#if TWO_POINT_TRAJRESPONSE
    /* change the corrector by -corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 - corr_tweek;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
#  ifdef DEBUG
    printf("corrector %s tweeked to %e\n", corr->name, *((double *)(corr->p_elem + kick_offset)));
    fflush(stdout);
#  endif
    free_matrices(corr->matrix);
    free(corr->matrix);
    corr->matrix = NULL;
    compute_matrix(corr, run, NULL);
#  ifdef DEBUG
    print_matrices(stdout, "*** corrector matrix:", corr->matrix);
#  endif

    /* track with tweeked corrector */
    p = p0;
    n_part = 1;
    fill_double_array(*one_part, totalPropertiesPerParticle, 0.0);
    if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                     traj0, run, 0, TEST_PARTICLES + TIME_DEPENDENCE_OFF, 1, 0, NULL, NULL, NULL, NULL, NULL))
      bombElegant("tracking failed for test particle (compute_trajcor_matrices())", NULL);

    /* compute coefficients of array C that are driven by this corrector */
    corrCalibration = 1 / (2 * corr_tweek);
#else
    corrCalibration = 1 / corr_tweek;
#endif

    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      char memName[1024];
      i = CM->mon_index[i_moni];
      Mij(CM->C, i_moni, i_corr) = corrCalibration *
        (computeMonitorReading(CM->umoni[i_moni], coord, traj1[i].centroid, 0) - computeMonitorReading(CM->umoni[i_moni], coord, traj0[i].centroid, 0));
      Mij(CM->Cp, i_moni, i_corr) = corrCalibration*
	 (computeMonitorReading(CM->umoni[i_moni], coord+1, traj1[i].centroid, 0) - computeMonitorReading(CM->umoni[i_moni], coord+1, traj0[i].centroid, 0));
        
      if (rpn_store_response_matrix) {
        sprintf(memName, "%cR_%s#%ld_%s#%ld.%s", coord == 0 ? 'H' : 'V',
                CM->umoni[i_moni]->name, CM->umoni[i_moni]->occurence,
                CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                SL->corr_param[CM->sl_index[i_corr]]);
        rpn_store(Mij(CM->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
      }
    }

    /* change the corrector back */
    *((double *)(corr->p_elem + kick_offset)) = kick0;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (corr->matrix) {
      free_matrices(corr->matrix);
      free(corr->matrix);
      corr->matrix = NULL;
    }
    if (save) {
      corr->matrix = save;
      save = NULL;
    } else
      compute_matrix(corr, run, NULL);
  }
  free(moniCalibration);
  tfree(traj0);
  traj0 = NULL;
  tfree(traj1);
  traj1 = NULL;
  free_czarray_2d((void **)one_part, 1, totalPropertiesPerParticle);
  one_part = NULL;
#ifdef DEBUG
  matrix_show(CM->C, "%13.6le ", "influence matrix\n", stdout);
#endif
  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    report_stats(stdout, "done");
  }

  if (flags & COMPUTE_RESPONSE_INVERT) {
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "Computing correction matrix ");
      fflush(stdout);
    }

    /* compute correction matrix T */
    if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (CM->C->n - CM->C->m)) {
      CM->remove_smallest_SVs = CM->C->n - CM->C->m;
      printf("Removing %ld smallest singular values to prevent instability\n", (long)CM->remove_smallest_SVs);
    }

    CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight, (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                          CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n, 
                          0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
    matrix_scmul(CM->T, -1);

    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "\ndone.");
      printf("Condition number is %e\n", conditionNumber);
      fflush(stdout);
    }

#ifdef DEBUG
    matrix_show(CM->T, "%13.6le ", "correction matrix\n", stdout);
#endif
  }
}

long global_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations,
                          RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, ELEMENT_LIST **newly_pegged) {
  ELEMENT_LIST *corr, *eptr;
  TRAJECTORY *traj;
  long iteration, kick_offset;
  long i_moni, i_corr;
  long n_part, i, sl_index;
  unsigned long tracking_flags;
  double **particle;
  double p, reading, fraction, minFraction, param, change;
  MAT *Qo, *dK;
  long i_pegged;

  log_entry("global_trajcor_plane");

  if (!matrix_check(CM->C) || !matrix_check(CM->T))
    bombElegant("corrupted response matrix detected (global_trajcor_plane)", NULL);
  if (!CM->mon_index)
    bombElegant("monitor index array is NULL (global_trajcor_plane)", NULL);
  if (!CM->posi)
    bombElegant("monitor readout array is NULL (global_trajcor_plane)", NULL);
  if (!CM->kick)
    bombElegant("corrector value array is NULL (global_trajcor_plane)", NULL);
  if (!traject)
    bombElegant("no trajectory arrays supplied (global_trajcor_plane)", NULL);
  if (!CM->T)
    bombElegant("no inverse matrix computed (global_trajcor_plane)", NULL);

  Qo = matrix_get(CM->nmon, 1); /* Vector of BPM errors */

  if (!beam) {
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES;
  } else {
    if (beam->n_to_track == 0)
      bombElegant("no particles to track in global_trajcor_plane()", NULL);
    particle = (double **)czarray_2d(sizeof(**particle), beam->n_to_track, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES + TEST_PARTICLE_LOSSES;
  }

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0) {
    if (coord == 0)
      printWarning("More correctors than monitors for x plane.", "Correction may be unstable (consider use of SV controls).");
    else
      printWarning("More correctors than monitors for y plane.", "Correction may be unstable (consider use of SV controls).");
  }
  for (iteration = 0; iteration <= n_iterations; iteration++) {
    if (!CM->posi[iteration])
      bombElegant("monitor readout array for this iteration is NULL (global_trajcor_plane)", NULL);
    if (!CM->kick[iteration])
      bombElegant("corrector value array for this iteration is NULL (global_trajcor_plane)", NULL);

    /* find trajectory */
    p = sqrt(sqr(run->ideal_gamma) - 1);

    if (!beam) {
      if (!starting_coord)
        fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
      else {
        for (i = 0; i < 6; i++)
          particle[0][i] = starting_coord[i];
      }
      n_part = 1;
    } else
      copy_particles(particle, beam->particle, n_part = beam->n_to_track);

#ifdef DEBUG
    if (1) {
      double centroid;
      long i, j;
      printf("beam centroids before tracking (beam:%s):\n",
             beam ? "given" : "not given");
      for (i = 0; i < 4; i++) {
        centroid = 0;
        for (j = 0; j < n_part; j++) {
          centroid += particle[j][i];
        }
        centroid /= n_part;
        printf("<x%ld> = %e\n", i, centroid);
      }
    }
#endif

    n_part = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                         (BEAM_SUMS **)NULL, (long *)NULL,
                         traj = traject[iteration == 0 ? 0 : 1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
    if (beam) {
      printf("%ld particles survived tracking", n_part);
      fflush(stdout);
      if (n_part == 0) {
        for (i = 0; i < beamline->n_elems + 1; i++)
          if (traj[i].n_part == 0)
            break;
        if (i != 0 && i < beamline->n_elems + 1)
          printf("---all beam lost before z=%em (element %s)",
                 traj[i].elem->end_pos, traj[i].elem->name);
        fflush(stdout);
        fputc('\n', stdout);
      }
      fputc('\n', stdout);
    } else if (n_part == 0) {
      /* This actually should never happend given the tracking flags */
      printf("Beam lost before end of beamline during trajectory correction\n");
      fflush(stdout);
    }

    /* find readings at monitors and add in reading errors */
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      if (!(eptr = traj[CM->mon_index[i_moni]].elem))
        bombElegant("invalid element pointer in trajectory array (global_trajcor_plane)", NULL);
      reading = computeMonitorReading(eptr, coord, traj[CM->mon_index[i_moni]].centroid, 0);
      if (isnan(reading) || isinf(reading))
        return 0;
      CM->posi[iteration][i_moni] = reading;
      Mij(Qo, i_moni, 0) = reading + (CM->bpm_noise ? noise_value(CM->bpm_noise, CM->bpm_noise_cutoff, CM->bpm_noise_distribution) : 0);
    }

    if (iteration == n_iterations)
      break;

    /* solve for the corrector changes */
    dK = matrix_mult(CM->T, Qo);

#ifdef DEBUG
    matrix_show(Qo, "%13.6le ", "traj matrix\n", stdout);
    matrix_show(dK, "%13.6le ", "kick matrix\n", stdout);
#endif

    /* step through beamline find any kicks that are over their limits */
    minFraction = 1;
    i_pegged = -1;
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
      param = fabs(*((double *)(corr->p_elem + kick_offset)) +
                   (change = Mij(dK, i_corr, 0) / CM->kick_coef[i_corr] * CM->corr_fraction));
      if (SL->corr_limit[sl_index] && param > SL->corr_limit[sl_index]) {
        fraction = fabs((SL->corr_limit[sl_index] - fabs(*((double *)(corr->p_elem + kick_offset)))) / change);
        if (fraction < minFraction) {
          i_pegged = i_corr;
          minFraction = fraction;
        }
      }
    }
    if (i_pegged != -1) {
      *newly_pegged = CM->ucorr[i_pegged];
      CM->pegged[i_pegged] = 1;
    }
    fraction = minFraction * CM->corr_fraction;

#if defined(DEBUG)
    printf("Changing correctors:\n");
    fflush(stdout);
#endif
    /* step through beamline and change correctors */
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
#if defined(DEBUG)
      printf("name = %s#%ld, before = %e, ", corr->name, corr->occurence, *((double *)(corr->p_elem + kick_offset)));
      fflush(stdout);
#endif
      if (iteration == 0)
        CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      *((double *)(corr->p_elem + kick_offset)) += Mij(dK, i_corr, 0) * fraction / CM->kick_coef[i_corr];
      if (SL->corr_limit[sl_index] && fabs(*((double *)(corr->p_elem + kick_offset))) > (1 + 1e-6) * SL->corr_limit[sl_index]) {
        printf("**** Corrector %s#%ld went past limit (%e > %e) --- This shouldn't happen (1)\n",
               corr->name, corr->occurence, fabs(*((double *)(corr->p_elem + kick_offset))), SL->corr_limit[sl_index]);
        printf("fraction=%e -> kick = %e\n", fraction, *((double *)(corr->p_elem + kick_offset)));
      }
      CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
#if defined(DEBUG)
      printf("after = %e, limit = %le, sl_index = %ld\n", *((double *)(corr->p_elem + kick_offset)), SL->corr_limit[sl_index], sl_index);
      fflush(stdout);
#endif
      if (corr->matrix) {
        free_matrices(corr->matrix);
        tfree(corr->matrix);
        corr->matrix = NULL;
      }
      compute_matrix(corr, run, NULL);
    }
    matrix_free(dK);
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
  }

  /* indicate that beamline concatenation and Twiss parameter computation (if wanted) are not current */
  beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
  beamline->flags &= ~BEAMLINE_TWISS_CURRENT;

  if (!beam)
    free_czarray_2d((void **)particle, 1, totalPropertiesPerParticle);
  else
    free_czarray_2d((void **)particle, beam->n_to_track, totalPropertiesPerParticle);
  particle = NULL;
  log_exit("global_trajcor_plane");

  matrix_free(Qo);

  return iteration;
}

long one_to_one_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations, RUN *run,
                              LINE_LIST *beamline, double *starting_coord, BEAM *beam, long method) {
  ELEMENT_LIST *corr, *eptr;
  TRAJECTORY *traj;
  long iteration, kick_offset;
  long i_moni, i_corr, sl_index;
  long n_part, i;
  unsigned long tracking_flags;
  double **particle, param, fraction;
  double p, reading;
  double response;

  log_entry("one_to_one_trajcor_plane");
#ifdef DEBUG
  printf("Performing one-to-X correction of trajectory for plane %ld\n", coord);
#endif
  if (!matrix_check(CM->C))
    bombElegant("corrupted correction matrix detected (one_to_one_trajcor_plane)", NULL);
  if (!CM->mon_index)
    bombElegant("monitor index array is NULL (one_to_one_trajcor_plane)", NULL);
  if (!CM->posi)
    bombElegant("monitor readout array is NULL (one_to_one_trajcor_plane)", NULL);
  if (!CM->kick)
    bombElegant("corrector value array is NULL (one_to_one_trajcor_plane)", NULL);
  if (!traject)
    bombElegant("no trajectory arrays supplied (one_to_one_trajcor_plane)", NULL);

  if (!beam) {
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES;
  } else {
    if (beam->n_to_track == 0)
      bombElegant("no particles to track in one_to_one_trajcor_plane()", NULL);
    particle = (double **)czarray_2d(sizeof(**particle), beam->n_to_track, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES + TEST_PARTICLE_LOSSES;
  }

  for (iteration = 0; iteration <= n_iterations; iteration++) {
    if (!CM->posi[iteration])
      bombElegant("monitor readout array for this iteration is NULL (one_to_one_trajcor_plane)", NULL);
    if (!CM->kick[iteration])
      bombElegant("corrector value array for this iteration is NULL (one_to_one_trajcor_plane)", NULL);

    /* record the starting trajectory */
    p = sqrt(sqr(run->ideal_gamma) - 1);
    if (!beam) {
      if (!starting_coord)
        fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
      else {
        for (i = 0; i < totalPropertiesPerParticle; i++)
          particle[0][i] = starting_coord[i];
      }
      n_part = 1;
    } else
      copy_particles(particle, beam->particle, n_part = beam->n_to_track);
    n_part = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                         (BEAM_SUMS **)NULL, (long *)NULL,
                         traj = traject[iteration == 0 ? 0 : 1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      if (!(eptr = traj[CM->mon_index[i_moni]].elem))
        bombElegant("invalid element pointer in trajectory array (one_to_one_trajcor_plane)", NULL);
      CM->posi[iteration][i_moni] = computeMonitorReading(eptr, coord, traj[CM->mon_index[i_moni]].centroid, 0);
    }
    if (iteration == n_iterations)
      break;

    i_moni = 0;
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      switch (method) {
      case ONE_TO_NEXT_CORRECTION:
      case ONE_TO_ONE_CORRECTION:
        /* Find next BPM downstream of this corrector */
        for (; i_moni < CM->nmon; i_moni++)
          if (CM->ucorr[i_corr]->end_pos < CM->umoni[i_moni]->end_pos)
            break;
        break;
      case ONE_TO_BEST_CORRECTION:
        /* Find BPM with larger response than its neighbors */
        for (; i_moni < CM->nmon; i_moni++)
          if (CM->ucorr[i_corr]->end_pos < CM->umoni[i_moni]->end_pos)
            break;
        if (i_moni != CM->nmon) {
          response = fabs(Mij(CM->C, i_moni, i_corr));
          for (; i_moni < CM->nmon - 1; i_moni++) {
            if (response > fabs(Mij(CM->C, i_moni + 1, i_corr)))
              break;
            response = fabs(Mij(CM->C, i_moni + 1, i_corr));
          }
        }
        break;
      default:
        printf("Error: Unknown method in one_to_one_trajcor_plane: %ld---This shouldn't happen (2)!\n", method);
        exitElegant(1);
        break;
      }

      if (i_moni == CM->nmon)
        break;

      /* find trajectory */
      p = sqrt(sqr(run->ideal_gamma) - 1);

      if (!beam) {
        if (!starting_coord)
          fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
        else {
          for (i = 0; i < totalPropertiesPerParticle; i++)
            particle[0][i] = starting_coord[i];
        }
        n_part = 1;
      } else
        copy_particles(particle, beam->particle, n_part = beam->n_to_track);

      n_part = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                           (BEAM_SUMS **)NULL, (long *)NULL,
                           traj = traject[1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
      if (traj[CM->mon_index[i_moni]].n_part == 0) {
        printf("beam lost at position %e m\n", traj[CM->mon_index[i_moni]].elem->end_pos);
        break;
      }

      if (!(eptr = traj[CM->mon_index[i_moni]].elem))
        bombElegant("invalid element pointer in trajectory array (one_to_one_trajcor_plane)", NULL);
      reading = computeMonitorReading(eptr, coord, traj[CM->mon_index[i_moni]].centroid, 0);
      reading += (CM->bpm_noise ? noise_value(CM->bpm_noise, CM->bpm_noise_cutoff, CM->bpm_noise_distribution) : 0);

#if defined(DEBUG)
      printf("i_moni = %ld, i_corr = %ld, reading = %e", i_moni, i_corr, reading);
      fflush(stdout);
#endif
      if (iteration == n_iterations || (i_corr >= CM->ncor || Mij(CM->C, i_moni, i_corr) == 0)) {
#if defined(DEBUG)
        printf("\n");
        fflush(stdout);
#endif
        continue;
      }

      /* change corrector */
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];          /* steering list index of this corrector */
      kick_offset = SL->param_offset[sl_index]; /* offset of steering parameter in element structure */
      if (iteration == 0)
        /* Record initial value */
        CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      /* Compute new value of the parameter
       * NewValue = OldValue - BpmReading/ResponseCoefficient*CorrectionFraction
       */
      param = *((double *)(corr->p_elem + kick_offset)) - reading / Mij(CM->C, i_moni, i_corr) * CM->corr_fraction;
      /* Check that we haven't exceeded allowed strength */
      fraction = 1;
      if (param && SL->corr_limit[sl_index])
        if ((fraction = SL->corr_limit[sl_index] / fabs(param)) > 1)
          fraction = 1;
      *((double *)(corr->p_elem + kick_offset)) = param * fraction;
      /* Record the new kick strength */
      CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
#if defined(DEBUG)
      printf(", param = %e, fraction=%e -> kick = %e\n", param, fraction, *((double *)(corr->p_elem + kick_offset)));
      fflush(stdout);
#endif
      if (SL->corr_limit[sl_index] && fabs(*((double *)(corr->p_elem + kick_offset))) > (1 + 1e-6) * SL->corr_limit[sl_index]) {
        printf("**** Corrector %s#%ld went past limit (%e > %e) --- This shouldn't happen (3)\n",
               corr->name, corr->occurence, fabs(*((double *)(corr->p_elem + kick_offset))), SL->corr_limit[sl_index]);
        printf("param = %e, fraction=%e -> kick = %e\n", param, fraction, *((double *)(corr->p_elem + kick_offset)));
      }
      if (corr->matrix) {
        free_matrices(corr->matrix);
        tfree(corr->matrix);
        corr->matrix = NULL;
      }
      compute_matrix(corr, run, NULL);
      if (beamline->links)
        assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

      /* indicate that beamline concatenation and Twiss parameter computation (if wanted) are not current */
      beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
      beamline->flags &= ~BEAMLINE_TWISS_CURRENT;
    }
  }

  if (!beam)
    free_czarray_2d((void **)particle, 1, totalPropertiesPerParticle);
  else
    free_czarray_2d((void **)particle, beam->n_to_track, totalPropertiesPerParticle);
  particle = NULL;
  log_exit("one_to_one_trajcor_plane");
  return iteration;
}

long thread_trajcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **traject, long n_iterations, RUN *run,
                          LINE_LIST *beamline, double *starting_coord, BEAM *beam, long verbose) {
  ELEMENT_LIST *corr;
  TRAJECTORY *traj;
  long iteration, kick_offset;
  long i_corr, i_moni, sl_index, iElem, nElems, iBest;
  long n_part, n_left, i, done, direction, improved, worsened;
  double **particle, param, fraction, bestValue, origValue, lastValue;
  double p, scanStep;
  long iScan, nScan, nToTry, corrLeft;
  unsigned long tracking_flags;

  if (!CM->mon_index)
    bombElegant("monitor index array is NULL (thread__trajcor_plane)", NULL);
  if (!CM->posi)
    bombElegant("monitor readout array is NULL (thread__trajcor_plane)", NULL);
  if (!CM->kick)
    bombElegant("corrector value array is NULL (thread__trajcor_plane)", NULL);
  if (!traject)
    bombElegant("no trajectory arrays supplied (thread__trajcor_plane)", NULL);

  if (!beam) {
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
  } else {
    if (beam->n_to_track == 0)
      bombElegant("no particles to track in thread__trajcor_plane()", NULL);
    particle = (double **)czarray_2d(sizeof(**particle), beam->n_to_track, totalPropertiesPerParticle);
  }
  tracking_flags = TEST_PARTICLES + TEST_PARTICLE_LOSSES;
  nElems = beamline->n_elems;
  done = 0;

  nScan = CM->default_threading_divisor + 1.5;
  scanStep = 1.0 / (nScan - 1);
  nToTry = CM->threading_correctors;

  for (iteration = 0; iteration <= n_iterations; iteration++) {
    if (!CM->posi[iteration])
      bombElegant("monitor readout array for this iteration is NULL (thread__trajcor_plane)", NULL);
    if (!CM->kick[iteration])
      bombElegant("corrector value array for this iteration is NULL (thread__trajcor_plane)", NULL);

    /* Establish baseline distance traveled before loss */
    p = sqrt(sqr(run->ideal_gamma) - 1);
    if (!beam) {
      if (!starting_coord)
        fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
      else {
        for (i = 0; i < totalPropertiesPerParticle; i++)
          particle[0][i] = starting_coord[i];
      }
      n_part = 1;
    } else
      copy_particles(particle, beam->particle, n_part = beam->n_to_track);

    n_left = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                         (BEAM_SUMS **)NULL, (long *)NULL,
                         traj = traject[iteration == 0 ? 0 : 1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
    if (verbose > 1) {
      printf("%ld particles left after tracking through beamline\n", n_left);
      fflush(stdout);
    }
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      CM->posi[iteration][i_moni]
	= computeMonitorReading(traj[CM->mon_index[i_moni]].elem, coord, traj[CM->mon_index[i_moni]].centroid, 0);
    }
    if (iteration == n_iterations)
      break;

    if (n_left == 0) {
      for (iElem = iBest = 0; iElem < nElems; iElem++)
        if (traj[iElem].n_part == 0)
          break;
      if ((iBest = iElem)>=nElems)
        iBest = nElems-1;
      if (verbose > 1) {
        printf("Beam reaches element %s#%ld at s=%le m\n", traj[iBest].elem->name, traj[iBest].elem->occurence, 
               traj[iBest].elem->end_pos);
        fflush(stdout);
      }
      /* Find the first corrector upstream of the loss point */
      for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
        corr = CM->ucorr[i_corr];
        if (corr->end_pos > traj[iBest].elem->end_pos)
          break;
      }
      if (--i_corr < 0)
        i_corr = 0;
      if (nToTry > 0)
        corrLeft = nToTry;
      else
        corrLeft = -1;
      for (; i_corr >= 0 && corrLeft != 0; i_corr--, corrLeft--) {
        corr = CM->ucorr[i_corr];
        sl_index = CM->sl_index[i_corr];          /* steering list index of this corrector */
        kick_offset = SL->param_offset[sl_index]; /* offset of steering parameter in element structure */
        if (iteration == 0)
          /* Record initial value */
          CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];

        lastValue = bestValue = origValue = *((double *)(corr->p_elem + kick_offset));
        if (!done && corr->end_pos < traj[iBest].elem->end_pos) {
          improved = 0;
          for (direction = -1; !improved && !done && direction < 2; direction += 2) {
            worsened = 0;
            if (verbose > 1) {
              printf("Trying corrector %ld of %ld at s=%e m in direction %ld\n", i_corr, (long)CM->ncor, corr->end_pos, direction);
              fflush(stdout);
            }
            for (iScan = 0; !worsened && !done && iScan < nScan; iScan++) {
              /* -- Compute new value of the parameter
               *    NewValue = OldValue + CorrectorTweek 
               */
              param = origValue + SL->corr_limit[sl_index] / CM->kick_coef[i_corr] * iScan * scanStep * direction;
              /* Check that we haven't exceeded allowed strength */
              fraction = 1;
              if (param && SL->corr_limit[sl_index])
                if ((fraction = SL->corr_limit[sl_index] / fabs(param)) > 1)
                  fraction = 1;
              lastValue = *((double *)(corr->p_elem + kick_offset)) = param * fraction;
              if (corr->matrix) {
                free_matrices(corr->matrix);
                tfree(corr->matrix);
                corr->matrix = NULL;
              }
              compute_matrix(corr, run, NULL);
              if (beamline->links)
                assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

              /* indicate that beamline concatenation and Twiss parameter computation (if wanted) are not current */
              beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
              beamline->flags &= ~BEAMLINE_TWISS_CURRENT;

              /* -- Find trajectory */
              p = sqrt(sqr(run->ideal_gamma) - 1);
              if (!beam) {
                if (!starting_coord)
                  fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
                else {
                  for (i = 0; i < totalPropertiesPerParticle; i++)
                    particle[0][i] = starting_coord[i];
                }
                n_part = 1;
              } else
                copy_particles(particle, beam->particle, n_part = beam->n_to_track);
              n_left = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                                   (BEAM_SUMS **)NULL, (long *)NULL,
                                   traj = traject[1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
              /* Determine if this is better than the previous best */
              if (n_left == 0) {
                for (iElem = 0; iElem < nElems; iElem++) {
                  if (traj[iElem].n_part == 0)
                    break;
                }
                if (iElem > iBest) {
                  iBest = iElem;
                  bestValue = *((double *)(corr->p_elem + kick_offset));
                  improved = 1;
                } else if (iElem < iBest) {
                  /* getting worse, so quit this scan */
                  worsened = 1;
                }
              } else {
                /* Made it to the end, so quit */
                iBest = nElems;
                bestValue = *((double *)(corr->p_elem + kick_offset));
                done = 1;
                break;
              }
            }
          }

          if (lastValue != bestValue) {
            *((double *)(corr->p_elem + kick_offset)) = bestValue;
            if (corr->matrix) {
              free_matrices(corr->matrix);
              tfree(corr->matrix);
              corr->matrix = NULL;
            }
            compute_matrix(corr, run, NULL);
            if (beamline->links)
              assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
          }
          if (verbose && bestValue != origValue) {
            if (traj[iBest].elem) {
              printf("Beam now reaches element %ld at s=%le m (%ld left)\n", iBest, traj[iBest].elem->end_pos, n_left);
              fflush(stdout);
            }
          }

          /* indicate that beamline concatenation and Twiss parameter computation (if wanted) are not current */
          beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
          beamline->flags &= ~BEAMLINE_TWISS_CURRENT;
        }

        /* Record the kick strength */
        CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      }
    } else {
      /* beam reaches end */
      if (verbose) {
        printf("Beam reached end of beamline\n");
        fflush(stdout);
      }
      for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
        corr = CM->ucorr[i_corr];
        sl_index = CM->sl_index[i_corr];          /* steering list index of this corrector */
        kick_offset = SL->param_offset[sl_index]; /* offset of steering parameter in element structure */
        if (iteration == 0)
          /* Record initial value */
          CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];

        /* Record the kick strength */
        CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      }
    }
  }

  if (!beam)
    free_czarray_2d((void **)particle, 1, totalPropertiesPerParticle);
  else
    free_czarray_2d((void **)particle, beam->n_to_track, totalPropertiesPerParticle);
  particle = NULL;
  return iteration;
}

long findLongInArray(long key, long *array, long nElements) {
  long i;
  for (i = 0; i < nElements; i++)
    if (key == array[i])
      return i;
  return -1;
}

ELEMENT_LIST *find_useable_moni_corr(int32_t *nmon, int32_t *ncor, long **mon_index,
                                     ELEMENT_LIST ***umoni, ELEMENT_LIST ***ucorr,
                                     double **kick_coef, long **sl_index, short **pegged, double **weight,
                                     long plane, STEERING_LIST *SL,
                                     RUN *run, LINE_LIST *beamline, long recircs) {
  ELEMENT_LIST *corr, *moni, *start;
  long i_elem, index;
  long moni_type[2] = {0, 0};
  long iplane, planeToInclude[2] = {-1, -1};

  log_entry("find_useable_moni_corr");

  switch (plane) {
  case 0:
    moni_type[0] = T_HMON;
    moni_type[1] = T_MONI;
    planeToInclude[0] = 0;
    break;
  case 2:
    moni_type[0] = T_VMON;
    moni_type[1] = T_MONI;
    planeToInclude[0] = 2;
    break;
  case 4:
    /* kludge for both planes */
    moni_type[0] = T_HMON;
    moni_type[1] = T_VMON;
    planeToInclude[0] = 0;
    planeToInclude[1] = 2;
    break;
  default:
    printf("error: invalid coordinate for correction: %ld\n", plane);
    fflush(stdout);
    exitElegant(1);
    break;
  }

  start = beamline->elem;
  if (recircs) {
    /* advance to recirculation point, if there is one */
    while (start) {
      if (start->type == T_RECIRC)
        break;
      start = start->succ;
    }
    if (!start)
      start = beamline->elem;
  }
  if (*ncor && *nmon) {
    log_exit("find_useable_moni_corr");
    return start;
  }

  *ncor = *nmon = 0;
  *mon_index = NULL;
  *umoni = NULL;
  *ucorr = NULL;
  *kick_coef = NULL;
  *sl_index = NULL;
  *pegged = NULL;

  for (index = 0; index < SL->n_corr_types; index++) {
    /* first count correctors */
    corr = SL->elem[index];

    if (!recircs) {
      /* Since the beamline doesn't recirculate, advance through remainder of beamline
       * to make sure there are subsequent monitors */
      moni = corr->succ;
      while (moni) {
        if (findLongInArray(moni->type, moni_type, 2) != -1 && getMonitorWeight(moni) > 0)
          break;
        moni = moni->succ;
      }
      if (!moni) {
        char buffer[16384];
        snprintf(buffer, 16384, "No monitor found following %s#%ld at s=%le\n",
                 corr->name, corr->occurence, corr->end_pos);
        printWarning("Correctors without following monitors are ignored for trajectory correction.", buffer);
        continue; /* ignore correctors with no monitors following */
      }
    }

    for (iplane = 0; iplane < 2; iplane++) {
      if (planeToInclude[iplane] >= 0) {
        /* check that corrector is permitted to steer in this plane */
        if (steering_corrector(corr, planeToInclude[iplane])) {
          printf("Adding %s#%ld (type %s) at %e m to %c plane steering\n",
                 corr->name, corr->occurence, entity_name[corr->type],
                 corr->end_pos, planeToInclude[iplane] ? 'v' : 'h');
          printf("The steering parameter is name = %s, value = %e\n",
                 entity_description[corr->type].parameter[SL->param_index[index]].name,
                 *((double *)(corr->p_elem + SL->param_offset[index])));
          *ucorr = trealloc(*ucorr, sizeof(**ucorr) * (*ncor + 1));
          (*ucorr)[*ncor] = corr;
          *sl_index = trealloc(*sl_index, sizeof(**sl_index) * (*ncor + 1));
          (*sl_index)[*ncor] = index;
          *kick_coef = trealloc(*kick_coef, sizeof(**kick_coef) * (*ncor + 1));
          if (!((*kick_coef)[*ncor] = compute_kick_coefficient(corr, planeToInclude[iplane], corr->type,
                                                               SL->corr_tweek[index], corr->name,
                                                               SL->corr_param[index], run))) {
            char buffer[16834];
            if (plane != 4) {
              snprintf(buffer, 16384, "Changing %s.%s does not directly kick the beam.",
                       corr->name, SL->corr_param[index]);
              printWarning(buffer, "Use for steering may result in a crash.");
            } else {
              if (!((*kick_coef)[*ncor] = compute_kick_coefficient(corr, planeToInclude[iplane] == 0 ? 2 : 0, corr->type,
                                                                   SL->corr_tweek[index], corr->name,
                                                                   SL->corr_param[index], run))) {
                snprintf(buffer, 16384, "Changing %s.%s does not directly kick the beam.",
                         corr->name, SL->corr_param[index]);
                printWarning(buffer, "Use for steering may result in a crash.");
              }
            }
          }
          *ncor += 1;
          break;
        }
      }
    }
  }

  if (*ncor == 0 && !recircs)
    return start;

    /* put the correctors into s order */
#ifdef DEBUG
  printf("Corrector list before ordering\n");
  for (index = 0; index < *ncor; index++) {
    printf("Corrector %ld: %s#%ld at s=%le, sl_index = %ld, param_offset = %ld, param_index = %ld, kick_coef = %le\n",
           index, (*ucorr)[index]->name, (*ucorr)[index]->occurence, (*ucorr)[index]->end_pos,
           (*sl_index)[index], SL->param_offset[(*sl_index)[index]],
           SL->param_index[(*sl_index)[index]], (*kick_coef)[index]);
  }
#endif
  reorderCorrectorArray(*ucorr, *sl_index, *kick_coef, *ncor);

  if (!recircs) {
    /* count all monitors with one or more correctors upstream and non-negative weight */
    moni = start;
    i_elem = 0;
    while (moni) {
      if (moni->end_pos >= (*ucorr)[0]->end_pos &&
          findLongInArray(moni->type, moni_type, 2) != -1 && getMonitorWeight(moni) > 0) {
        printf("Found monitor %s#%ld (type %s) at s=%e\n",
               moni->name, moni->occurence, entity_name[moni->type], moni->end_pos);
        *nmon += 1;
        *umoni = trealloc(*umoni, *nmon * sizeof(**umoni));
        (*umoni)[*nmon - 1] = moni;
        *mon_index = trealloc(*mon_index, *nmon * sizeof(**mon_index));
        (*mon_index)[*nmon - 1] = i_elem;
      }
      moni = moni->succ;
      i_elem++;
    }
  } else {
    /* find all monitors */
    moni = start;
    i_elem = 0;
    while (moni) {
      if (findLongInArray(moni->type, moni_type, 2) != -1 && getMonitorWeight(moni) > 0) {
        *nmon += 1;
        *umoni = trealloc(*umoni, *nmon * sizeof(**umoni));
        (*umoni)[*nmon - 1] = moni;
        *mon_index = trealloc(*mon_index, *nmon * sizeof(**mon_index));
        (*mon_index)[*nmon - 1] = i_elem;
      }
      moni = moni->succ;
      i_elem++;
    }
  }
  *pegged = calloc(sizeof(**pegged), *ncor);
  *weight = calloc(sizeof(**weight), *nmon);
  log_exit("find_useable_moni_corr");
  return (start);
}

void compute_orbcor_matrices(CORMON_DATA *CM, STEERING_LIST *SL, long coord, RUN *run, LINE_LIST *beamline,
                             unsigned long flags, short rpn_store_response_matrix) {
  long i_corr, i_moni;
  double coef, htune, moniFactor, *corrFactor, *corrFactorFL, coefFL, W0 = 0.0;
  double conditionNumber;
  char memName[1024];

#ifdef DEBUG
  ELEMENT_LIST *start;
  start = find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index, &CM->umoni, &CM->ucorr,
                                 &CM->kick_coef, &CM->sl_index, &CM->pegged, &CM->weight, coord, SL, run, beamline, 1);
  printf("finding twiss parameters beginning at %s.\n", start->name);
  fflush(stdout);
#else
  find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index, &CM->umoni, &CM->ucorr,
                         &CM->kick_coef, &CM->sl_index, &CM->pegged, &CM->weight, coord, SL, run, beamline, 1);
#endif

  if (!(beamline->flags & BEAMLINE_TWISS_CURRENT)) {
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      printf("updating twiss parameters...");
      fflush(stdout);
    }
    update_twiss_parameters(run, beamline, NULL);

#ifdef DEBUG
    printf("Tunes: %e, %e\n", beamline->tune[0], beamline->tune[1]);
    printf("Initial eta: %e, %e\n",
           beamline->elem->twiss->etax,
           beamline->elem->twiss->etay);
    printf("Final eta: %e, %e\n",
           beamline->elast->twiss->etax,
           beamline->elast->twiss->etay);
    fflush(stdout);
#endif
    if (!(flags & COMPUTE_RESPONSE_SILENT))
      report_stats(stdout, "\ndone: ");
  }

#ifdef DEBUG
  printf("monitors: ");
  fflush(stdout);
  for (i_moni = 0; i_moni < CM->nmon; i_moni++)
    printf("%s ", CM->umoni[i_moni]->name);
  fflush(stdout);
  printf("\ncorrectors: ");
  fflush(stdout);
  for (i_corr = 0; i_corr < CM->ncor; i_corr++)
    printf("%s ", CM->ucorr[i_corr]->name);
  fflush(stdout);
  fputc('\n', stdout);
#endif

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0) {
    if (coord == 0)
      printWarning("Orbit correction may be unstable (consider use of SV controls).", "More correctors than monitors for x plane.");
    else
      printWarning("Orbit correction may be unstable (consider use of SV controls).", "More correctors than monitors for y plane.");
  }
  if (CM->ncor == 0) {
    if (coord == 0)
      printWarning("No orbit correction done for x plane.", "No correctors for x plane.");
    else
      printWarning("No orbit correction done for y plane.", "No correctors for y plane.");
  }
  if (CM->nmon == 0) {
    if (coord == 0)
      printWarning("No orbit correction done for x plane.", "No monitors for x plane.");
    else
      printWarning("No orbit correction done for y plane.", "No monitors for y plane.");
    CM->ncor = 0;
    return;
  }

  /* allocate matrices for this plane */
  if (CM->C)
    matrix_free(CM->C);
  CM->C = matrix_get(CM->nmon, CM->ncor); /* Response matrix */
  if (CM->T)
    matrix_free(CM->T);
  CM->T = NULL;

  if (flags & COMPUTE_RESPONSE_FINDONLY)
    return;

  /* set up weight matrix */
  CM->equalW = 1;
  for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
    CM->weight[i_moni] = getMonitorWeight(CM->umoni[i_moni]);
    if (!i_moni)
      W0 = CM->weight[i_moni];
    else if (CM->weight[i_moni] != W0)
      CM->equalW = 0;
  }

  /* find transfer matrix from correctors to monitors */
  if ((coef = 2 * sin(htune = PIx2 * beamline->tune[coord ? 1 : 0] / 2)) == 0)
    bombElegant("can't compute response matrix--beamline unstable", NULL);
  coefFL = beamline->matrix->R[4][5];

  corrFactor = tmalloc(sizeof(*corrFactor) * CM->ncor);
  corrFactorFL = tmalloc(sizeof(*corrFactorFL) * CM->ncor);
  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    printf("computing orbit response matrix...");
    fflush(stdout);
  }

  switch (coord) {
  case 0:
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      double betax, etax;
      betax = CM->ucorr[i_corr]->twiss->betax;
      etax = CM->ucorr[i_corr]->twiss->etax;
      if (CM->ucorr[i_corr]->pred) {
        betax = (betax + CM->ucorr[i_corr]->pred->twiss->betax) / 2;
        etax = (etax + CM->ucorr[i_corr]->pred->twiss->etax) / 2;
      }
      corrFactor[i_corr] =
        getCorrectorCalibration(CM->ucorr[i_corr], coord) * sqrt(betax) / coef;
      corrFactorFL[i_corr] =
        getCorrectorCalibration(CM->ucorr[i_corr], coord) * etax / coefFL;
    }
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      moniFactor =
        getMonitorCalibration(CM->umoni[i_moni], coord) * sqrt(CM->umoni[i_moni]->twiss->betax);
      for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
        double phi;
        phi = CM->ucorr[i_corr]->twiss->phix;
        if (CM->ucorr[i_corr]->pred)
          phi = (phi + CM->ucorr[i_corr]->pred->twiss->phix) / 2;
        Mij(CM->C, i_moni, i_corr) = moniFactor * corrFactor[i_corr] *
                                     cos(htune - fabs(CM->umoni[i_moni]->twiss->phix - phi));
        if (flags & COMPUTE_RESPONSE_FIXEDLENGTH)
          Mij(CM->C, i_moni, i_corr) -= CM->umoni[i_moni]->twiss->etax * corrFactorFL[i_corr];
        if (rpn_store_response_matrix) {
          sprintf(memName, "HR_%s#%ld_%s#%ld.%s",
                  CM->umoni[i_moni]->name, CM->umoni[i_moni]->occurence,
                  CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                  SL->corr_param[CM->sl_index[i_corr]]);
          rpn_store(Mij(CM->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
        }
      }
    }
    break;
  default:
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      double betay;
      betay = CM->ucorr[i_corr]->twiss->betay;
      if (CM->ucorr[i_corr]->pred)
        betay = (betay + CM->ucorr[i_corr]->pred->twiss->betay) / 2;
      corrFactor[i_corr] =
        getCorrectorCalibration(CM->ucorr[i_corr], coord) * sqrt(betay) / coef;
    }
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      moniFactor =
        getMonitorCalibration(CM->umoni[i_moni], coord) * sqrt(CM->umoni[i_moni]->twiss->betay);
      for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
        double phi;
        phi = CM->ucorr[i_corr]->twiss->phiy;
        if (CM->ucorr[i_corr]->pred)
          phi = (phi + CM->ucorr[i_corr]->pred->twiss->phiy) / 2;
        Mij(CM->C, i_moni, i_corr) = moniFactor * corrFactor[i_corr] *
                                     cos(htune - fabs(CM->umoni[i_moni]->twiss->phiy - phi));
        if (rpn_store_response_matrix) {
          sprintf(memName, "VR_%s#%ld_%s#%ld.%s",
                  CM->umoni[i_moni]->name, CM->umoni[i_moni]->occurence,
                  CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                  SL->corr_param[CM->sl_index[i_corr]]);
          rpn_store(Mij(CM->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
        }
      }
    }
  }
  free(corrFactor);
  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    report_stats(stdout, "\ndone");
    fflush(stdout);
  }
#ifdef DEBUG
  matrix_show(CM->C, "%13.6le ", "influence matrix\n", stdout);
#endif

  if (flags & COMPUTE_RESPONSE_INVERT) {
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      /* compute correction matrix T */
      printf("computing correction matrix...");
      fflush(stdout);
    }
    if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (CM->C->n - CM->C->m)) {
      CM->remove_smallest_SVs = CM->C->n - CM->C->m;
      printf("Removing %ld smallest singular values to prevent instability\n", (long)CM->remove_smallest_SVs);
    }
    CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight, (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                          CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                          0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
    matrix_scmul(CM->T, -1);

    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "\ndone.");
      printf("Condition number is %e\n", conditionNumber);
      fflush(stdout);
    }
#ifdef DEBUG
    matrix_show(CM->T, "%13.6le ", "correction matrix\n", stdout);
#endif
  }
}

/* Compute orbit response matrix from closed orbit, rather than using beta functions etc */
void compute_orbcor_matrices1(CORRECTION *_correct, long coord, RUN *run, LINE_LIST *beamline, unsigned long flags,
                               long coupled, CORMON_DATA *CMxy, CORMON_DATA *CMyx)
{
  ELEMENT_LIST *corr;
  TRAJECTORY *clorb0, *clorb1;
  long kick_offset, i_corr, i_moni, i, sl_index, otherCoord;
  double kick0, corr_tweek;
  VMATRIX *save, *M;
  char *matrixTypeName[2][2] = {{"H", "HV"}, {"VH", "V"}};
  char memName[1024];
  double conditionNumber, W0 = 0.0;
  CORMON_DATA *CM, *CMc;
  STEERING_LIST *SL;
  char messageBuffer[1024];

  if (flags & COMPUTE_RESPONSE_VERBOSE)
    report_stats(stdout, "Computing response matrix from closed orbits.");
  
  CM = CMc = NULL;
  SL = NULL;
  if (coord==0) {
    CM = _correct->CMFx;
    SL = &_correct->SLx;
    otherCoord = 2;
    if (coupled)
      CMc = CMyx; /* y plane response to x correctors */
  } else {
    CM = _correct->CMFy;
    SL = &_correct->SLy;
    otherCoord = 0;
    if (coupled)
      CMc = CMxy; /* x plane response to y correctors */
  }
    
  find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index, &CM->umoni, &CM->ucorr,
                         &CM->kick_coef, &CM->sl_index, &CM->pegged, &CM->weight, coord, SL, run, beamline, 1);
  if (CMc)
    find_useable_moni_corr(&CMc->nmon, &CMc->ncor, &CMc->mon_index, &CMc->umoni, &CMc->ucorr,
                           &CMc->kick_coef, &CMc->sl_index, &CMc->pegged, &CMc->weight, otherCoord, SL, run, beamline, 1);
  
  if (CM->ncor == 0) {
    if (coord == 0)
      printWarning("No orbit correction done for x plane.", "No correctors for x plane.");
    else
      printWarning("No orbit correction done for y plane.", "No correctors for y plane.");
  }
  
  /* We don't need these coefficients since the response matrix includes them */
  for (i_corr = 0; i_corr < CM->ncor; i_corr++)
    CM->kick_coef[i_corr] = 1;
  if (CMc)
    for (i_corr = 0; i_corr < CM->ncor; i_corr++)
      CMc->kick_coef[i_corr] = 1;
  
  if (CM->nmon == 0) {
    if (coord == 0)
      printWarning("No orbit correction done for x plane.", "No monitors for x plane.");
    else
      printWarning("No orbit correction done for y plane.", "No monitors for y plane.");
    CM->ncor = 0;
  }
  if (CMc && CMc->nmon == 0) {
    if (coord == 0)
      printWarning("No orbit correction done for x plane.", "No monitors for x plane.");
    else
      printWarning("No orbit correction done for y plane.", "No monitors for y plane.");
    CMc->ncor = 0;
  }
  
  /* allocate matrices for this plane */
  if (CM->C)
    matrix_free(CM->C);
  CM->C = matrix_get(CM->nmon, CM->ncor); /* Response matrix */
  if (CM->T)
    matrix_free(CM->T);
  CM->T = NULL;
  if (CMc) {
    if (CMc->C)
      matrix_free(CMc->C);
    CMc->C = matrix_get(CMc->nmon, CMc->ncor);
    CMc->T = NULL;
  }
  
  if (flags & COMPUTE_RESPONSE_VERBOSE) {
    snprintf(messageBuffer, 1024, "Found %ld correctors and %ld monitors for %c\n", (long)CM->ncor, (long)CM->nmon, coord?'x':'y');
    report_stats(stdout, messageBuffer);
  }
  if (flags & COMPUTE_RESPONSE_FINDONLY)
    return;
  
  clorb0 = tmalloc(sizeof(*clorb0) * (beamline->n_elems + 1));
  clorb1 = tmalloc(sizeof(*clorb1) * (beamline->n_elems + 1));

  if (!(M = beamline->matrix)) {
    if (beamline->elem_twiss)
      M = beamline->matrix = full_matrix(beamline->elem_twiss, run, 1);
    else
      M = beamline->matrix = full_matrix(beamline->elem, run, 1);
  }

  for (i_corr = 0; i_corr<CM->ncor; i_corr++) {
    sl_index = CM->sl_index[i_corr];
    corr = CM->ucorr[i_corr];
    kick_offset = SL->param_offset[sl_index];
    corr_tweek = SL->corr_tweek[sl_index];
    
    /* record value of corrector */
    kick0 = *((double *)(corr->p_elem + kick_offset));
    
    /* change the corrector by corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 + corr_tweek;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (flags & COMPUTE_RESPONSE_VERBOSE) {
      printf("Working on response for element %s#%ld, with values of %le +/- %le\n",
             corr->name, corr->occurence, kick0, corr_tweek);
      fflush(stdout);
    }

    if (corr->matrix) {
      save = corr->matrix;
      corr->matrix = NULL;
    } else
      save = NULL;

    compute_matrix(corr, run, NULL);
    
    /* find closed orbit with tweaked corrector */
    if (!find_closed_orbit(clorb1, _correct->clorb_accuracy, _correct->clorb_accuracy_requirement,
                           _correct->clorb_iterations, beamline, M, run, 0, 1, CM->fixed_length, NULL,
                           _correct->clorb_iter_fraction, _correct->clorb_fraction_multiplier, _correct->clorb_multiplier_interval, NULL,
                           _correct->clorb_track_for_orbit)) {
      printf("Failed to find perturbed closed orbit.\n");
      fflush(stdout);
      return;
    }
    
    /* change the corrector by -corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 - corr_tweek;
    compute_matrix(corr, run, NULL);
    /* find closed orbit with tweaked corrector */
    if (!find_closed_orbit(clorb0, _correct->clorb_accuracy, _correct->clorb_accuracy_requirement,
                           _correct->clorb_iterations, beamline, M, run, 0, 1, CM->fixed_length, NULL,
                           _correct->clorb_iter_fraction, _correct->clorb_fraction_multiplier, _correct->clorb_multiplier_interval, NULL,
                           _correct->clorb_track_for_orbit)) {
      printf("Failed to find perturbed closed orbit.\n");
      fflush(stdout);
      return;
    }
    /* compute coefficients of array C that are driven by this corrector */
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      i = CM->mon_index[i_moni];
      Mij(CM->C, i_moni, i_corr) = 
        (computeMonitorReading(CM->umoni[i_moni], coord, clorb1[i].centroid, 0)
         - computeMonitorReading(CM->umoni[i_moni], coord, clorb0[i].centroid, 0)) / (2 * corr_tweek);
      if (_correct->rpn_store_response_matrix) {
        /* store result in rpn memory */
        sprintf(memName, "%sR_%s#%ld_%s#%ld.%s",
                matrixTypeName[CM->bpmPlane][CM->corrPlane],
                CM->umoni[i_moni]->name, CM->umoni[i_moni]->occurence,
                CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                SL->corr_param[CM->sl_index[i_corr]]);
        rpn_store(Mij(CM->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
      }
    }
    if (coupled && CMc) {
      for (i_moni = 0; i_moni < CMc->nmon; i_moni++) {
        i = CMc->mon_index[i_moni];
        Mij(CMc->C, i_moni, i_corr) = 
          (computeMonitorReading(CMc->umoni[i_moni], otherCoord, clorb1[i].centroid, 0)
           - computeMonitorReading(CMc->umoni[i_moni], otherCoord, clorb0[i].centroid, 0)) / (2 * corr_tweek);
        if (_correct->rpn_store_response_matrix) {
          /* store result in rpn memory */
          sprintf(memName, "%sR_%s#%ld_%s#%ld.%s",
                  matrixTypeName[CMc->bpmPlane][CMc->corrPlane],
                  CMc->umoni[i_moni]->name, CMc->umoni[i_moni]->occurence,
                  CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                  SL->corr_param[CM->sl_index[i_corr]]);
          rpn_store(Mij(CMc->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
      }
      }
    }

    /* change the corrector back */
    *((double *)(corr->p_elem + kick_offset)) = kick0;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (corr->matrix) {
      free_matrices(corr->matrix);
      free(corr->matrix);
      corr->matrix = NULL;
    }
    if (save) {
      corr->matrix = save;
      save = NULL;
    } else
      compute_matrix(corr, run, NULL);
  }
  
    
  if (flags & COMPUTE_RESPONSE_INVERT) {
    /* set up weight matrix */
    CM->equalW = 1;
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      CM->weight[i_moni] = getMonitorWeight(CM->umoni[i_moni]);
      if (!i_moni)
        W0 = CM->weight[i_moni];
      else if (CM->weight[i_moni] != W0)
        CM->equalW = 0;
    }
    
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      /* compute correction matrix T */
      printf("computing correction matrix...");
      fflush(stdout);
    }
    if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (CM->C->n - CM->C->m)) {
      CM->remove_smallest_SVs = CM->C->n - CM->C->m;
      printf("Removing %ld smallest singular values to prevent instability\n", (long)CM->remove_smallest_SVs);
    }
    CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight, (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                          CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                          0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
    matrix_scmul(CM->T, -1);
    
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "\ndone.");
      printf("Condition number is %e\n", conditionNumber);
      fflush(stdout);
    }
  }

  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    report_stats(stdout, "\ndone");
    fflush(stdout);
  }
  
}

#if USE_MPI
/* Compute orbit response matrix from closed orbit, rather than using beta functions etc */
void compute_orbcor_matrices1p(CORRECTION *_correct, RUN *run, LINE_LIST *beamline, unsigned long flags,
                               long coupled, CORMON_DATA *CMxy, CORMON_DATA *CMyx)
{
  ELEMENT_LIST *corr;
  TRAJECTORY *clorb0, *clorb1;
  long kick_offset, i_corr, i_moni, i, sl_index, coord, otherCoord;
  long i_corrc;
  double kick0, corr_tweek;
  VMATRIX *save, *M;
  char *matrixTypeName[2][2] = {{"H", "HV"}, {"VH", "V"}};
  char memName[1024];
  double conditionNumber, W0 = 0.0;
  long index, indexc, bufOffset, bufSize;
  double *responseBufferSend = NULL, *responseBufferReceive = NULL;
  CORMON_DATA *CM, *CMx, *CMy, *CMc;
  STEERING_LIST *SL, *SLx, *SLy;
  char messageBuffer[1024];
  short parallelTrackingBasedMatricesSave;
#ifdef DEBUG
  static FILE *fpdeb = NULL;

  if (!fpdeb && myid==0) {
    fpdeb = fopen("com1p.sdds", "w");
    fprintf(fpdeb, "SDDS1\n&column name=x type=double units=m &end\n");
    fprintf(fpdeb, "&column name=y type=double units=m &end\n");
    fprintf(fpdeb, "&parameter name=Type type=string &end\n");
    fprintf(fpdeb, "&parameter name=Tweek type=double &end\n");
    fprintf(fpdeb, "&parameter name=delta type=double &end\n");
    fprintf(fpdeb, "&parameter name=FixedLength type=short &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif
  parallelTrackingBasedMatricesSave = parallelTrackingBasedMatrices;
  parallelTrackingBasedMatrices = 0;

  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stdout, "Computing response matrix from closed orbits.");
#ifdef MPI_DEBUG
  report_stats(stdout, "Computing response matrix from closed orbits.");
#endif
  
  CMx = CMy = CM = CMc = NULL;
  SLx = SLy = SL = NULL;
  for (coord=0; coord<3; coord+=2) {
    if (coord==0) {
      CMx = CM = _correct->CMFx;
      SLx = SL = &_correct->SLx;
      otherCoord = 2;
      if (coupled)
        CMc = CMyx; /* y plane response to x correctors */
    } else {
      CMy = CM = _correct->CMFy;
      SLy = SL = &_correct->SLy;
      otherCoord = 0;
      if (coupled)
        CMc = CMxy; /* x plane response to y correctors */
    }
    
    find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index, &CM->umoni, &CM->ucorr,
                           &CM->kick_coef, &CM->sl_index, &CM->pegged, &CM->weight, coord, SL, run, beamline, 1);
    if (CMc)
      find_useable_moni_corr(&CMc->nmon, &CMc->ncor, &CMc->mon_index, &CMc->umoni, &CMc->ucorr,
                           &CMc->kick_coef, &CMc->sl_index, &CMc->pegged, &CMc->weight, otherCoord, SL, run, beamline, 1);
    
    if (CM->ncor == 0) {
      if (coord == 0)
        printWarning("No orbit correction done for x plane.", "No correctors for x plane.");
      else
        printWarning("No orbit correction done for y plane.", "No correctors for y plane.");
    }

    /* We don't need these coefficients since the response matrix includes them */
    for (i_corr = 0; i_corr < CM->ncor; i_corr++)
      CM->kick_coef[i_corr] = 1;
    if (CMc)
      for (i_corr = 0; i_corr < CM->ncor; i_corr++)
        CMc->kick_coef[i_corr] = 1;
    
    if (CM->nmon == 0) {
      if (coord == 0)
        printWarning("No orbit correction done for x plane.", "No monitors for x plane.");
      else
        printWarning("No orbit correction done for y plane.", "No monitors for y plane.");
      CM->ncor = 0;
    }
    if (CMc && CMc->nmon == 0) {
      if (coord == 0)
        printWarning("No orbit correction done for x plane.", "No monitors for x plane.");
      else
        printWarning("No orbit correction done for y plane.", "No monitors for y plane.");
      CMc->ncor = 0;
    }
    
    /* allocate matrices for this plane */
    if (CM->C)
      matrix_free(CM->C);
    CM->C = matrix_get(CM->nmon, CM->ncor); /* Response matrix */
    if (CM->T)
      matrix_free(CM->T);
    CM->T = NULL;
    if (CMc) {
      if (CMc->C)
        matrix_free(CMc->C);
      CMc->C = matrix_get(CMc->nmon, CMc->ncor);
      CMc->T = NULL;
    }
    
    if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0) {
      snprintf(messageBuffer, 1024, "Found %ld correctors and %ld monitors for %c\n", (long)CM->ncor, (long)CM->nmon, coord?'x':'y');
      report_stats(stdout, messageBuffer);
    }
#ifdef MPI_DEBUG
    snprintf(messageBuffer, 1024, "Found %ld correctors and %ld monitors for %c\n", (long)CM->ncor, (long)CM->nmon, coord?'x':'y');
    report_stats(stdout, messageBuffer);
#endif
    if (flags & COMPUTE_RESPONSE_FINDONLY)
      return;
  }
  
  long nxx = CMx->ncor*CMx->nmon;
  long nyy = CMy->ncor*CMy->nmon;
  long nxy = CMy->ncor*CMx->nmon;
  long nyx = CMx->ncor*CMy->nmon;
  /* response buffer regions:
     [0, nxx-1] : x cor to x mon
     [nxx, nxx+nyy-1]: y cor to y mon
     [nxx+nyy, nxx+nyy+nyx-1]: x cor to y mon
     [nxx+nyy+nxy, nxx+nyy+nyx+nxy-1]: y cor to x mon
   */
  bufOffset = bufSize = nxx + nyy;
  if (coupled)
    bufSize += nxy + nyx;
  responseBufferSend = calloc(2*bufSize, sizeof(*responseBufferSend));
  responseBufferReceive = calloc(2*bufSize, sizeof(*responseBufferReceive));
  
  clorb0 = tmalloc(sizeof(*clorb0) * (beamline->n_elems + 1));
  clorb1 = tmalloc(sizeof(*clorb1) * (beamline->n_elems + 1));

  if (!(M = beamline->matrix)) {
    if (beamline->elem_twiss)
      M = beamline->matrix = full_matrix(beamline->elem_twiss, run, 1);
    else
      M = beamline->matrix = full_matrix(beamline->elem, run, 1);
  }

  index = 0;
  indexc = bufOffset;
  for (i_corrc = 0; i_corrc < CMx->ncor+CMy->ncor; i_corrc++) {
    if (i_corrc<CMx->ncor) {
      i_corr = i_corrc;
      CM = CMx;
      SL = SLx;
      coord = 0;
      otherCoord = 2;
      CMc = CMyx; /* y plane response to x correctors */
    } else {
      i_corr = i_corrc-CMx->ncor;
      CM = CMy;
      SL = SLy;
      coord = 2;
      otherCoord = 0;
      CMc = CMxy; /* x plane response to y correctors */
    }
#ifdef MPI_DEBUG
    fprintf(stdout, "Possibly working on i_corrc = %ld, i_corr=%ld, CM=%p, SL=%p, coord=%ld, otherCoord=%ld, CMc=%p\n",
            i_corrc, i_corr, CM, SL, coord, otherCoord, CMc);
    fflush(stdout);
#endif
    if (i_corrc%n_processors!=myid) {
      index += CM->nmon;
      if (coupled)
        indexc += CMc->nmon;
      continue;
    }
#ifdef MPI_DEBUG
    fprintf(stdout, "Actually working on i_corrc = %ld, i_corr=%ld, CM=%p, SL=%p, coord=%ld, otherCoord=%ld, CMc=%p\n",
            i_corrc, i_corr, CM, SL, coord, otherCoord, CMc);
    fflush(stdout);
#endif
    
    sl_index = CM->sl_index[i_corr];
    corr = CM->ucorr[i_corr];
    kick_offset = SL->param_offset[sl_index];
    corr_tweek = SL->corr_tweek[sl_index];
    
#ifdef MPI_DEBUG
    fprintf(stdout, "sl_index = %ld, corr=%p (%s), kick_offset=%ld, corr_tweek=%le\n", 
            sl_index, corr, corr->name, kick_offset, corr_tweek);
    fflush(stdout);
#endif
    /* record value of corrector */
    kick0 = *((double *)(corr->p_elem + kick_offset));
#ifdef MPI_DEBUG
    fprintf(stdout, "kick0 = %le\n", kick0);
    fflush(stdout);
#endif
    
    /* change the corrector by corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 + corr_tweek;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (flags & COMPUTE_RESPONSE_VERBOSE) {
      printf("Working on response for element %s#%ld, with values of %le +/- %le\n",
             corr->name, corr->occurence, kick0, corr_tweek);
      fflush(stdout);
    }
#ifdef MPI_DEBUG
    fprintf(stdout, "Working on response for element %s#%ld, with values of %le +/- %le\n",
            corr->name, corr->occurence, kick0, corr_tweek);
    fflush(stdout);
#endif

    if (corr->matrix) {
      save = corr->matrix;
      corr->matrix = NULL;
    } else
      save = NULL;
#ifdef MPI_DEBUG
    fprintf(stdout, "save = %p, computing new matrix\n", save);
    fflush(stdout);
#endif

    compute_matrix(corr, run, NULL);
    
#ifdef MPI_DEBUG
    fprintf(stdout, "Finding closed orbit for +tweek\n");
    fflush(stdout);
#endif
    /* find closed orbit with tweaked corrector */
    if (!find_closed_orbit(clorb1, _correct->clorb_accuracy, _correct->clorb_accuracy_requirement,
                           _correct->clorb_iterations, beamline, M, run, 0, 1, CM->fixed_length, NULL,
                           _correct->clorb_iter_fraction, _correct->clorb_fraction_multiplier, _correct->clorb_multiplier_interval, NULL,
                           _correct->clorb_track_for_orbit)) {
      printf("Failed to find perturbed closed orbit.\n");
      fflush(stdout);
      return;
    }
#ifdef DEBUG
    if (myid==0) {
      fprintf(fpdeb, "%s\n%21.15e\n%21.15e\n%ld\n", corr->name, corr_tweek, clorb1[0].centroid[5], CM->fixed_length);
      for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
        i = CM->mon_index[i_moni];
        fprintf(fpdeb, "%21.15e %21.15e\n", clorb1[i].centroid[0], clorb1[i].centroid[2]);
      }
      fprintf(fpdeb, "\n");
    }
#endif
    
#ifdef MPI_DEBUG
    fprintf(stdout, "Doing -tweek\n");
    fflush(stdout);
#endif
    /* change the corrector by -corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 - corr_tweek;
#ifdef MPI_DEBUG
    fprintf(stdout, "Computing matrix\n");
    fflush(stdout);
#endif
    compute_matrix(corr, run, NULL);
    /* find closed orbit with tweaked corrector */
#ifdef MPI_DEBUG
    fprintf(stdout, "Finding closed orbit\n");
    fflush(stdout);
#endif
    if (!find_closed_orbit(clorb0, _correct->clorb_accuracy, _correct->clorb_accuracy_requirement,
                           _correct->clorb_iterations, beamline, M, run, 0, 1, CM->fixed_length, NULL,
                           _correct->clorb_iter_fraction, _correct->clorb_fraction_multiplier, _correct->clorb_multiplier_interval, NULL,
                           _correct->clorb_track_for_orbit)) {
      printf("Failed to find perturbed closed orbit.\n");
      fflush(stdout);
      return;
    }
#ifdef DEBUG
    if (myid==0) {
      fprintf(fpdeb, "%s\n%21.15e\n%21.15e\n%ld\n", corr->name, -corr_tweek, clorb0[0].centroid[5], CM->fixed_length);
      for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
        i = CM->mon_index[i_moni];
        fprintf(fpdeb, "%21.15e %21.15e\n", clorb0[i].centroid[0], clorb0[i].centroid[2]);
      }
      fprintf(fpdeb, "\n");
    }
#endif
#ifdef MPI_DEBUG
    fprintf(stdout, "Computing in-plane response\n");
    fflush(stdout);
#endif
    /* compute coefficients of array C that are driven by this corrector */
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      i = CM->mon_index[i_moni];
      responseBufferSend[index++] = 
        (computeMonitorReading(CM->umoni[i_moni], coord, clorb1[i].centroid, 0)
         - computeMonitorReading(CM->umoni[i_moni], coord, clorb0[i].centroid, 0)) / (2 * corr_tweek);
    }
    if (coupled) {
#ifdef MPI_DEBUG
      fprintf(stdout, "Computing cross-plane response\n");
      fflush(stdout);
#endif
      for (i_moni = 0; i_moni < CMc->nmon; i_moni++) {
        i = CMc->mon_index[i_moni];
        responseBufferSend[indexc++] = 
          (computeMonitorReading(CMc->umoni[i_moni], otherCoord, clorb1[i].centroid, 0)
           - computeMonitorReading(CMc->umoni[i_moni], otherCoord, clorb0[i].centroid, 0)) / (2 * corr_tweek);
      }
    }

#ifdef MPI_DEBUG
    fprintf(stdout, "Setting corrector back\n");
    fflush(stdout);
#endif
    /* change the corrector back */
    *((double *)(corr->p_elem + kick_offset)) = kick0;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (corr->matrix) {
      free_matrices(corr->matrix);
      free(corr->matrix);
      corr->matrix = NULL;
    }
    if (save) {
      corr->matrix = save;
      save = NULL;
    } else
      compute_matrix(corr, run, NULL);
#ifdef MPI_DEBUG
    fprintf(stdout, "Done with corrector\n");
    fflush(stdout);
#endif
  }
  
  /* Share response matrix data */
  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stderr, "Waiting for workers\n");
#ifdef MPI_DEBUG
  fprintf(stdout, "Waiting on barrier (1)\n");
  fflush(stdout);
#endif
  MPI_Barrier(MPI_COMM_WORLD);

  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stderr, "Collecting data from workers\n");
#ifdef MPI_DEBUG
  fprintf(stdout, "Collecting data\n");
  fflush(stdout);
#endif
  MPI_Allreduce(responseBufferSend, responseBufferReceive, bufSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stderr, "Collating data\n");
  
  index = 0;
  indexc = bufOffset;
#ifdef MPI_DEBUG
  fprintf(stdout, "Collating data\n");
  fflush(stdout);
#endif
  for (i_corrc = 0; i_corrc < CMx->ncor+CMy->ncor; i_corrc++) {
    if (i_corrc<CMx->ncor) {
      i_corr = i_corrc;
      CMx = CM = _correct->CMFx;
      SL = SLx;
      CMc = CMyx; /* y plane response to x correctors */
    } else {
      i_corr = i_corrc-CMx->ncor;
      CMy = CM = _correct->CMFy;
      SL = SLy;
      CMc = CMxy; /* x plane response to y correctors */
    }
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      Mij(CM->C, i_moni, i_corr) = responseBufferReceive[index++];
      if (_correct->rpn_store_response_matrix) {
        /* store result in rpn memory */
        sprintf(memName, "%sR_%s#%ld_%s#%ld.%s",
                matrixTypeName[CM->bpmPlane][CM->corrPlane],
                CM->umoni[i_moni]->name, CM->umoni[i_moni]->occurence,
                CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence,
                SL->corr_param[CM->sl_index[i_corr]]);
        rpn_store(Mij(CM->C, i_moni, i_corr), NULL, rpn_create_mem(memName, 0));
      }
    }
    if (coupled) {
      for (i_moni = 0; i_moni < CMc->nmon; i_moni++)
        Mij(CMc->C, i_moni, i_corr) = responseBufferReceive[indexc++];
    }
    
    if (flags & COMPUTE_RESPONSE_INVERT) {
      /* set up weight matrix */
      CM->equalW = 1;
      for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
        CM->weight[i_moni] = getMonitorWeight(CM->umoni[i_moni]);
        if (!i_moni)
          W0 = CM->weight[i_moni];
        else if (CM->weight[i_moni] != W0)
          CM->equalW = 0;
      }
      
      if (!(flags & COMPUTE_RESPONSE_SILENT)) {
        /* compute correction matrix T */
        printf("computing correction matrix...");
        fflush(stdout);
      }
      if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (CM->C->n - CM->C->m)) {
        CM->remove_smallest_SVs = CM->C->n - CM->C->m;
        printf("Removing %ld smallest singular values to prevent instability\n", (long)CM->remove_smallest_SVs);
      }
      CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight, (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                            CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                            0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
      matrix_scmul(CM->T, -1);
      
      if (!(flags & COMPUTE_RESPONSE_SILENT)) {
        report_stats(stdout, "\ndone.");
        printf("Condition number is %e\n", conditionNumber);
        fflush(stdout);
      }
#ifdef MPI_DEBUG
      matrix_show(CM->T, "%13.6le ", "correction matrix\n", stdout);
#endif
    }
  }
#ifdef MPI_DEBUG
  fprintf(stdout, "Done collating data\n");
  fflush(stdout);
#endif
  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stderr, "Done collating data\n");
  
  free(responseBufferSend);
  free(responseBufferReceive);

  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stdout, "Waiting for workers.\n");
#ifdef MPI_DEBUG
  fprintf(stdout, "Waiting on barrier (2)\n");
  fflush(stdout);
#endif
  
  MPI_Barrier(MPI_COMM_WORLD);
  if ((flags & COMPUTE_RESPONSE_VERBOSE) && myid==0)
    report_stats(stdout, "Done computing response matrix.\n");
  
#ifdef MPI_DEBUG
  fprintf(stdout, "All done.\n");
  fflush(stdout);
#endif
  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    report_stats(stdout, "\ndone");
    fflush(stdout);
  }
  
  parallelTrackingBasedMatrices = parallelTrackingBasedMatricesSave;
}
#endif

long orbcor_plane(CORMON_DATA *CM, STEERING_LIST *SL, long coord, TRAJECTORY **orbit, long n_iterations,
                  double clorb_acc, double clorb_acc_req, long clorb_iter, double clorb_iter_frac, double clorb_frac_mult,
                  long clorb_mult_interval, long clorb_track_for_orbit,
                  RUN *run, LINE_LIST *beamline, double *closed_orbit, double *Cdp,
                  ELEMENT_LIST **newly_pegged) {
  ELEMENT_LIST *corr, *eptr;
  TRAJECTORY *clorb;
  VMATRIX *M;
  long iteration, kick_offset;
  long i_moni, i_corr, i, sl_index, i_pegged;
  double dp, reading;
  /* double last_rms_pos; */
  double best_rms_pos, rms_pos, corr_fraction;
  double fraction, param;
  MAT *Qo, *dK = NULL;

  log_entry("orbcor_plane");
  *newly_pegged = 0;

  if (!CM)
    bombElegant("NULL CORMON_DATA pointer passed to orbcor_plane", NULL);
  if (!orbit)
    bombElegant("NULL TRAJECTORY pointer passed to orbcor_plane", NULL);
  if (!run)
    bombElegant("NULL RUN pointer passed to orbcor_plane", NULL);
  if (!beamline)
    bombElegant("NULL LINE_LIST pointer passed to orbcor_plane", NULL);
  if (CM->ncor && CM->nmon && !CM->T)
    bombElegant("No inverse matrix computed prior to orbcor_plane", NULL);

  if (closed_orbit)
    dp = closed_orbit[5];
  else
    dp = 0;

  clorb = orbit[0];
  if (closed_orbit) {
    for (i = 0; i < 4; i++)
      clorb[0].centroid[i] = closed_orbit[i];
  } else {
    for (i = 0; i < 4; i++)
      clorb[0].centroid[i] = 0;
  }

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0) {
    if (coord == 0)
      printWarning("Orbit correction may be unstable (consider use of SV controls).", "More correctors than monitors for x plane.");
    else
      printWarning("Orbit correction may be unstable (consider use of SV controls).", "More correctors than monitors for y plane.");
  }

  Qo = matrix_get(CM->nmon, 1); /* Vector of BPM errors */

  best_rms_pos = rms_pos = DBL_MAX / 4;
  corr_fraction = CM->corr_fraction;
  for (iteration = 0; iteration <= n_iterations; iteration++) {
    clorb = orbit[iteration ? 1 : 0];
    delete_phase_references();
    if (!(M = beamline->matrix)) {
      if (beamline->elem_twiss)
        M = beamline->matrix = full_matrix(beamline->elem_twiss, run, 1);
      else
        M = beamline->matrix = full_matrix(beamline->elem, run, 1);
    }
    if (!M || !M->R)
      bombElegant("problem calculating full matrix (orbcor_plane)", NULL);
    if (iteration == 1)
      for (i = 0; i < 6; i++)
        orbit[1][0].centroid[i] = orbit[0][0].centroid[i];
    if (!find_closed_orbit(clorb, clorb_acc, clorb_acc_req, clorb_iter, beamline, M, run, dp, 1, CM->fixed_length, NULL,
                           clorb_iter_frac, clorb_frac_mult, clorb_mult_interval, NULL, clorb_track_for_orbit)) {
      printf("Failed to find closed orbit.\n");
      fflush(stdout);
      return (-1);
    }
#ifdef DEBUG
    printf("Closed orbit: %le, %le, %le, %le, %le, %le\n",
           clorb[0].centroid[0],
           clorb[0].centroid[1],
           clorb[0].centroid[2],
           clorb[0].centroid[3],
           clorb[0].centroid[4],
           clorb[0].centroid[5]);
#endif

    if (Cdp)
      Cdp[iteration] = clorb[0].centroid[5];
    if (closed_orbit) {
      for (i = 0; i < 6; i++)
        closed_orbit[i] = clorb[0].centroid[i];
    }

    /* find readings at monitors and add in reading errors */
    i = 1;
    /* last_rms_pos = rms_pos; */
    if (best_rms_pos > rms_pos)
      best_rms_pos = rms_pos;
    rms_pos = 0;
    for (i_moni = 0; i_moni < CM->nmon; i_moni++, i++) {
      while (CM->umoni[i_moni] != clorb[i].elem) {
        if (clorb[i].elem->succ)
          i++;
        else
          bombElegant("missing monitor in closed orbit", NULL);
      }

      eptr = clorb[i].elem;

      CM->posi[iteration][i_moni] = reading = computeMonitorReading(eptr, coord, clorb[i].centroid, 0);
      Mij(Qo, i_moni, 0) = reading + (CM->bpm_noise ? noise_value(CM->bpm_noise, CM->bpm_noise_cutoff, CM->bpm_noise_distribution) : 0.0);
      rms_pos += sqr(Mij(Qo, i_moni, 0));
      if (!clorb[i].elem->succ)
        break;
    }
    rms_pos = sqrt(rms_pos / CM->nmon);
    if (iteration == 0 && rms_pos > 1e9) {
      /* if the closed orbit has RMS > 1e9m, I assume correction won't work and routine bombs */
      printf("Orbit beyond 10^9 m.  Aborting correction.\n");
      fflush(stdout);
      return (-1);
    }
    if (rms_pos > clorb_acc && rms_pos > best_rms_pos * 1.01) {
      if (corr_fraction == 1 || corr_fraction == 0)
        break;
      printf("orbit diverging on iteration %ld: RMS=%e m (was %e m)--redoing with correction fraction of %e\n",
             iteration, rms_pos, best_rms_pos, corr_fraction = sqr(corr_fraction));
      best_rms_pos = rms_pos;
      fflush(stdout);
#if 0
      if (iteration>=1) {
        /* step through beamline and change correctors back to last values */
        for (i_corr=0; i_corr<CM->ncor; i_corr++) {
          corr = CM->ucorr[i_corr];
          sl_index = CM->sl_index[i_corr];
          kick_offset = SL->param_offset[sl_index];
          *((double*)(corr->p_elem+kick_offset)) = CM->kick[iteration-1][i_corr]/CM->kick_coef[i_corr];
          if (corr->matrix) {
            free_matrices(corr->matrix);
            tfree(corr->matrix);
            corr->matrix = NULL;
          }
          compute_matrix(corr, run, NULL);
        }
      }
      if (beamline->links)
        assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

      /* free concatenated matrix to force it to be recomputed with new corrector matrices */
      free_matrices(beamline->matrix);
      tfree(beamline->matrix);
      beamline->matrix = NULL;
      iteration -= 1;
      rms_pos = last_rms_pos;
      continue;
#endif
    }

    if (CM->fixed_length)
      dp = clorb[0].centroid[5];

    if (iteration == n_iterations)
      break;

    if (CM->nmon && CM->ncor)
      /* solve for the corrector kicks */
      dK = matrix_mult(CM->T, Qo);

#ifdef DEBUG
      /*
    matrix_show(Qo, "%13.6le ", "traj matrix\n", stdout);
    matrix_show(dK, "%13.6le ", "kick matrix\n", stdout);
    */
#endif

    /* see if any correctors are over their limit */
    fraction = corr_fraction;
    i_pegged = -1;
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      double coef;
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
      param = fabs(*((double *)(corr->p_elem + kick_offset)) +
                   (coef = Mij(dK, i_corr, 0) / CM->kick_coef[i_corr]) * corr_fraction);
      if (SL->corr_limit[sl_index] && param > SL->corr_limit[sl_index]) {
        double fraction1;
        fraction1 = fabs((fabs(SL->corr_limit[sl_index]) - fabs(*((double *)(corr->p_elem + kick_offset)))) / coef);
        if (fraction1 < fraction) {
          i_pegged = i_corr;
          fraction = fraction1;
        }
      }
    }
    if (i_pegged != -1) {
      *newly_pegged = CM->ucorr[i_pegged];
      CM->pegged[i_pegged] = 1;
      printf("correction fraction reduced to %le due to pegged corrector %s#%ld\n",
             fraction, (*newly_pegged)->name, (*newly_pegged)->occurence);
    }

    /* step through beamline and change correctors */
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
      if (iteration == 0)
        CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      *((double *)(corr->p_elem + kick_offset)) += Mij(dK, i_corr, 0) * fraction / CM->kick_coef[i_corr];
      CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
#ifdef DEBUG
      printf("corrector %s#%ld changing from %le to %le (kick_coef=%le, fraction=%le, param=%le)\n",
             corr->name, corr->occurence,
             CM->kick[iteration][i_corr],
             CM->kick[iteration + 1][i_corr],
             CM->kick_coef[i_corr], fraction,
             *((double *)(corr->p_elem + kick_offset)));
#endif

      if (corr->matrix) {
        free_matrices(corr->matrix);
        tfree(corr->matrix);
        corr->matrix = NULL;
      }
      compute_matrix(corr, run, NULL);
    }
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

    /* free concatenated matrix to force it to be recomputed with new corrector matrices */
    free_matrices(beamline->matrix);
    tfree(beamline->matrix);
    beamline->matrix = NULL;
    matrix_free(dK);
    /*
      messes up output !
    if (i_pegged!=-1) {
      iteration++;
      break;
    }
    */
  }

  if (rms_pos > 1e9) {
    /* if the final closed orbit has RMS > 1e9m, I assume correction didn't work and routine bombs */
    printf("Orbit beyond 1e9 m.  Aborting correction.\n");
    fflush(stdout);
    return (-1);
  }

  if (rms_pos > best_rms_pos + CM->corr_accuracy && (corr_fraction == 1 || corr_fraction == 0)) {
    printf("orbit not improving--iteration terminated at iteration %ld with last result of %e m RMS\n",
           iteration, rms_pos);
    fflush(stdout);
    /* step through beamline and change correctors back to last values */
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
      *((double *)(corr->p_elem + kick_offset)) = CM->kick[iteration][i_corr] / CM->kick_coef[i_corr];
      if (corr->matrix) {
        free_matrices(corr->matrix);
        tfree(corr->matrix);
        corr->matrix = NULL;
      }
      compute_matrix(corr, run, NULL);
    }
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

    /* free concatenated matrix to force it to be recomputed with new corrector matrices */
    free_matrices(beamline->matrix);
    tfree(beamline->matrix);
    beamline->matrix = NULL;
  }

  /* indicate that beamline concatenation and Twiss computation (if wanted) are not current */
  beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
  beamline->flags &= ~BEAMLINE_TWISS_CURRENT;

  matrix_free(Qo);

  log_exit("orbcor_plane");
  return (iteration);
}

ELEMENT_LIST *next_element_of_type(ELEMENT_LIST *elem, long type) {
  while (elem) {
    if (elem->type == type)
      return (elem);
    elem = elem->succ;
  }
  return (elem);
}

ELEMENT_LIST *next_element_of_types(ELEMENT_LIST *elem, long *type, long n_types, long *index, char **corr_name,
                                    long *start_occurence, long *end_occurence, long *occurence_step, double *s_start, double *s_end) {
  register long i;
  while (elem) {
    for (i = 0; i < n_types; i++) {
      if (corr_name[i] && strlen(corr_name[i]) && !wild_match(elem->name, corr_name[i]))
        continue;
      if (elem->type == type[i]) {
        if (start_occurence[i] != 0 && end_occurence[i] != 0 &&
            (elem->occurence < start_occurence[i] || elem->occurence > end_occurence[i] ||
             (elem->occurence - start_occurence[i]) % occurence_step[i]))
          continue;
        if (s_start[i] >= 0 && s_end[i] >= 0 &&
            (elem->end_pos < s_start[i] || elem->end_pos > s_end[i]))
          continue;
        *index = i;
        return (elem);
      }
    }
    elem = elem->succ;
  }
  return (elem);
}

long zero_correctors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  return zero_hcorrectors(elem, run, correct) +
         zero_vcorrectors(elem, run, correct);
}

long zero_correctors_one_plane(CORMON_DATA *CM, RUN *run, STEERING_LIST *SL) {
  long slIndex, i, paramOffset, n_zeroed;
  ELEMENT_LIST *elem;

  n_zeroed = 0;
  for (i = 0; i < CM->ncor; i++) {
    slIndex = CM->sl_index[i];
    paramOffset = SL->param_offset[slIndex];
    elem = CM->ucorr[i];
    *((double *)(elem->p_elem + paramOffset)) = 0;
    if (elem->matrix) {
      free_matrices(elem->matrix);
      tfree(elem->matrix);
      elem->matrix = NULL;
    }
    compute_matrix(elem, run, NULL);
    n_zeroed++;
  }
  return (n_zeroed);
}

long zero_hcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  long nz;
  nz = zero_correctors_one_plane(correct->CMx, run, &(correct->SLx));
  if (correct->verbose)
    printf("%ld H correctors set to zero\n", nz);
  return nz;
}

long zero_vcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  long nz;
  nz = zero_correctors_one_plane(correct->CMy, run, &(correct->SLy));
  if (correct->verbose)
    printf("%ld V correctors set to zero\n", nz);
  return nz;
}

#if USE_MPI
long sync_correctors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  return sync_hcorrectors(elem, run, correct) + sync_vcorrectors(elem, run, correct);
}

long sync_correctors_one_plane(CORMON_DATA *CM, RUN *run, STEERING_LIST *SL) {
  long slIndex, i, paramOffset;
  long n_synced = 0;
  ELEMENT_LIST *elem;
  for (i = 0; i < CM->ncor; i++) {
    slIndex = CM->sl_index[i];
    paramOffset = SL->param_offset[slIndex];
    elem = CM->ucorr[i];
    if (last_optimize_function_call)
      MPI_Bcast(elem->p_elem + paramOffset, 1, MPI_CHAR, min_value_location, MPI_COMM_WORLD);
    if (elem->matrix) {
      free_matrices(elem->matrix);
      tfree(elem->matrix);
      elem->matrix = NULL;
    }
    compute_matrix(elem, run, NULL);
    n_synced++;
  }

  return (n_synced);
}

long sync_hcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  long nz;

  nz = sync_correctors_one_plane(correct->CMx, run, &(correct->SLx));

  return nz;
}

long sync_vcorrectors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct) {
  long nz;

  nz = sync_correctors_one_plane(correct->CMy, run, &(correct->SLy));

  return nz;
}

#endif

void rotate_xy(double *x, double *y, double angle) {
  static double x0, y0, ca, sa;

  x0 = *x;
  y0 = *y;
  *x = x0 * (ca = cos(angle)) + y0 * (sa = sin(angle));
  *y = -x0 * sa + y0 * ca;
}

double rms_value(double *data, long n_data) {
  double sum2;
  long i;

  for (i = sum2 = 0; i < n_data; i++)
    sum2 += sqr(data[i]);
  if (n_data)
    return (sqrt(sum2 / n_data));
  else
    return (0.0);
}

long steering_corrector(ELEMENT_LIST *eptr, long plane) {
  switch (eptr->type) {
  case T_HVCOR:
    return ((HVCOR *)(eptr->p_elem))->steering;
  case T_HCOR:
    return ((HCOR *)(eptr->p_elem))->steering;
  case T_VCOR:
    return ((VCOR *)(eptr->p_elem))->steering;
  case T_EHVCOR:
    return ((EHVCOR *)(eptr->p_elem))->steering;
  case T_EHCOR:
    return ((EHCOR *)(eptr->p_elem))->steering;
  case T_EVCOR:
    return ((EVCOR *)(eptr->p_elem))->steering;
  case T_QUAD:
    if (plane == 0)
      return ((QUAD *)(eptr->p_elem))->xSteering;
    return ((QUAD *)(eptr->p_elem))->ySteering;
  case T_KQUAD:
    if (plane == 0)
      return ((KQUAD *)(eptr->p_elem))->xSteering;
    return ((KQUAD *)(eptr->p_elem))->ySteering;
  case T_DQCOR:
    if (plane == 0)
      return ((DQCOR *)(eptr->p_elem))->xSteering;
    return ((DQCOR *)(eptr->p_elem))->ySteering;
  case T_KSEXT:
    if (plane == 0)
      return ((KSEXT *)(eptr->p_elem))->xSteering;
    return ((KSEXT *)(eptr->p_elem))->ySteering;
    break;
  case T_CSBEND:
    if (plane == 0)
      return ((CSBEND *)(eptr->p_elem))->xSteering;
    return ((CSBEND *)(eptr->p_elem))->ySteering;
    break;
  default:
    return 1;
  }
}

long find_index(long key, long *list, long n_listed) {
  register long i;
  for (i = 0; i < n_listed; i++)
    if (key == list[i])
      return (i);
  return (-1);
}

double noise_value(double xamplitude, double xcutoff, long xerror_type) {
  switch (xerror_type) {
  case UNIFORM_ERRORS:
    return (2 * xamplitude * (random_3(0) - 0.5));
  case GAUSSIAN_ERRORS:
    return (gauss_rn_lim(0.0, xamplitude, xcutoff, random_3));
  case PLUS_OR_MINUS_ERRORS:
    /* return a number on [-x-1, x-1]), which is added to 1 in the calling routine
             * (since these are implemented as fractional errors)
             */
    return (xamplitude * (random_3(0) > 0.5 ? 1.0 : -1.0) - 1);
  default:
    bombElegant("unknown error type in perturbation()", NULL);
    exitElegant(1);
    break;
  }
  return (0.0);
}

double computeMonitorReading(ELEMENT_LIST *elem, long coord, double *centroid, unsigned long flags)
  /* coord = 0 is x, etc. */
{
  double calibration, tilt, reading;
  char *equation;
  double setpoint;
  double x, y, xp, yp;

  x = centroid[0];
  xp = centroid[1];
  y = centroid[2];
  yp = centroid[3];
  
  calibration = 1;
  setpoint = 0;
  equation = NULL;

  switch (elem->type) {
  case T_MONI:
    if (coord%2==0) {
      x -= ((MONI *)(elem->p_elem))->dx - ((MONI*)(elem->p_elem))->dz*xp;
      y -= ((MONI *)(elem->p_elem))->dy - ((MONI*)(elem->p_elem))->dz*yp;
    }
    switch (coord) {
    case 0:
      calibration = ((MONI *)(elem->p_elem))->xcalibration;
      setpoint = ((MONI *)(elem->p_elem))->xsetpoint;
      break;
    case 2:
      calibration = ((MONI *)(elem->p_elem))->ycalibration;
      setpoint = ((MONI *)(elem->p_elem))->ysetpoint;
      break;
    default:
      break;
    }
    tilt = ((MONI *)(elem->p_elem))->tilt;
    if (flags & COMPUTEMONITORREADING_TILT_0)
      tilt = 0;
    if (tilt) {
      rotate_xy(&x, &y, tilt);
      rotate_xy(&xp, &yp, tilt);
    }
    switch (coord) {
    case 0:
      equation = ((MONI *)(elem->p_elem))->x_readout;
      break;
    case 2:
      equation = ((MONI *)(elem->p_elem))->y_readout;
      break;
    default:
      break;
    }
    break;
  case T_HMON:
    if (coord%2==0) {
      x -= ((HMON *)(elem->p_elem))->dx - ((HMON*)(elem->p_elem))->dz*xp;
      y -= ((HMON *)(elem->p_elem))->dy - ((HMON*)(elem->p_elem))->dz*yp;
    }
    if (coord==0) {
      calibration = ((HMON *)(elem->p_elem))->calibration;
      setpoint = ((HMON *)(elem->p_elem))->setpoint;
    } 
    tilt = ((HMON *)(elem->p_elem))->tilt;
    if (flags & COMPUTEMONITORREADING_TILT_0)
      tilt = 0;
    if (tilt) {
      rotate_xy(&x, &y, tilt);
      rotate_xy(&xp, &yp, tilt);
    }
    if (coord==0) {
      equation = ((HMON *)(elem->p_elem))->readout;
    }
    if (coord != 0 && coord!=1)
      bombElegant("element in horizontal monitor list is not a Horizontal monitor--internal logic error", NULL);
    break;
  case T_VMON:
    if (coord%2==0) {
      x -= ((VMON *)(elem->p_elem))->dx - ((VMON*)(elem->p_elem))->dz*xp;
      y -= ((VMON *)(elem->p_elem))->dy - ((VMON*)(elem->p_elem))->dz*yp;
    }
    if (coord==2) {
      calibration = ((VMON *)(elem->p_elem))->calibration;
      setpoint = ((VMON *)(elem->p_elem))->setpoint;
    }
    tilt = ((VMON *)(elem->p_elem))->tilt;
    if (flags & COMPUTEMONITORREADING_TILT_0)
      tilt = 0;
    if (tilt) {
      rotate_xy(&x, &y, tilt);
      rotate_xy(&xp, &yp, tilt);
    }
    if (coord==2)
      equation = ((VMON *)(elem->p_elem))->readout;
    if (coord != 2 && coord!=3)
      bombElegant("element in horizontal monitor list is not a Horizontal monitor--internal logic error", NULL);
    break;
  default:
    printf("error: element %s found in monitor list--internal logic error\n",
           elem->name);
    fflush(stdout);
    abort();
    break;
  }

  if (flags & COMPUTEMONITORREADING_CAL_1)
    calibration = 1;

  if (equation) {
    rpn_clear();
    rpn_store(x, NULL, rpn_x_mem);
    rpn_store(y, NULL, rpn_y_mem);
    reading = rpn(equation) * calibration;
    if (rpn_check_error())
      exitElegant(1);
  } else {
    switch (coord) {
    case 0:
      reading = x * calibration;
      break;
    case 1:
      reading = xp * calibration;
      break;
    case 2:
      reading = y * calibration;
      break;
    case 3:
      reading = yp * calibration;
      break;
    default:
      bombElegantVA("invalid coordinate index %ld in computeMonitorReading", coord);
      reading = 0;
      break;
    }
  }

  return reading - setpoint;
}

double getMonitorWeight(ELEMENT_LIST *elem) {
  switch (elem->type) {
  case T_MONI:
    return ((MONI *)(elem->p_elem))->weight;
  case T_HMON:
    return ((HMON *)(elem->p_elem))->weight;
  case T_VMON:
    return ((VMON *)(elem->p_elem))->weight;
  }
  bombElegant("invalid element type in getMonitorWeight()", NULL);
  return (1);
}

double getMonitorCalibration(ELEMENT_LIST *elem, long coord) {
  switch (elem->type) {
  case T_HMON:
    return ((HMON *)(elem->p_elem))->calibration;
  case T_VMON:
    return ((VMON *)(elem->p_elem))->calibration;
  case T_MONI:
    if (coord)
      return ((MONI *)(elem->p_elem))->ycalibration;
    return ((MONI *)(elem->p_elem))->xcalibration;
  }
  bombElegant("invalid element type in getMonitorCalibration()", NULL);
  return (1);
}

void setMonitorCalibration(ELEMENT_LIST *elem, double calib, long coord) {
  switch (elem->type) {
  case T_HMON:
    ((HMON *)(elem->p_elem))->calibration = calib;
    break;
  case T_VMON:
    ((VMON *)(elem->p_elem))->calibration = calib;
    break;
  case T_MONI:
    if (coord)
      ((MONI *)(elem->p_elem))->ycalibration = calib;
    else
      ((MONI *)(elem->p_elem))->xcalibration = calib;
    break;
  default:
    bombElegant("invalid element type in setMonitorCalibration()", NULL);
    break;
  }
}

double getCorrectorCalibration(ELEMENT_LIST *elem, long coord) {
  double value = 0;
  switch (elem->type) {
  case T_HCOR:
    value = ((HCOR *)(elem->p_elem))->calibration;
    break;
  case T_VCOR:
    value = ((VCOR *)(elem->p_elem))->calibration;
    break;
  case T_HVCOR:
    if (coord)
      value = ((HVCOR *)(elem->p_elem))->ycalibration;
    else
      value = ((HVCOR *)(elem->p_elem))->xcalibration;
    break;
  case T_EHCOR:
    value = ((EHCOR *)(elem->p_elem))->calibration;
    break;
  case T_EVCOR:
    value = ((EVCOR *)(elem->p_elem))->calibration;
    break;
  case T_EHVCOR:
    if (coord)
      value = ((EHVCOR *)(elem->p_elem))->ycalibration;
    else
      value = ((EHVCOR *)(elem->p_elem))->xcalibration;
    break;
  case T_QUAD:
    if (coord)
      value = ((QUAD *)(elem->p_elem))->yKickCalibration;
    else
      value = ((QUAD *)(elem->p_elem))->xKickCalibration;
    break;
  case T_KQUAD:
    if (coord)
      value = ((KQUAD *)(elem->p_elem))->yKickCalibration;
    else
      value = ((KQUAD *)(elem->p_elem))->xKickCalibration;
    break;
  case T_DQCOR:
    if (coord)
      value = ((DQCOR *)(elem->p_elem))->yKickCalibration;
    else
      value = ((DQCOR *)(elem->p_elem))->xKickCalibration;
    break;
  default:
    value = 1;
    break;
  }
  return value;
}

void prefillSteeringResultsArray(CORMON_DATA *CM, STEERING_LIST *SL, long iterations)
/* This ensures that the kick values are valid even if the correction algorithm totally fails to run */
{
  long ic, it;
  for (it = 0; it <= iterations; it++) {
    for (ic = 0; ic < CM->ncor; ic++) {
      if (it == 0) {
        ELEMENT_LIST *corr;
        long sl_index, kick_offset;
        corr = CM->ucorr[ic];
        sl_index = CM->sl_index[ic];
        kick_offset = SL->param_offset[sl_index];
        CM->kick[it][ic] = (*((double *)(corr->p_elem + kick_offset))) * CM->kick_coef[ic];
      } else {
        CM->kick[it][ic] = CM->kick[0][ic];
      }
    }
  }
}

void copy_steering_results(CORMON_DATA *CM, STEERING_LIST *SL, CORMON_DATA *CMA, long iterations) {
  long ic0, ic1, it;
  for (it = 0; it <= iterations; it++) {
    for (ic0 = ic1 = 0; ic0 < CM->ncor; ic0++) {
      if (CM->pegged[ic0] || it == iterations) {
        /* compute value from the parameter */
        ELEMENT_LIST *corr;
        long sl_index, kick_offset;
        corr = CM->ucorr[ic0];
        sl_index = CM->sl_index[ic0];
        kick_offset = SL->param_offset[sl_index];
        CM->kick[it][ic0] = (*((double *)(corr->p_elem + kick_offset))) * CM->kick_coef[ic0];
      } else {
        /* copy new data */
        CM->kick[it][ic0] = CMA->kick[it][ic1];
        ic1++;
      }
    }
    memcpy(CM->posi[it], CMA->posi[it], CM->nmon * sizeof(CMA->posi[0][0]));
  }
}

int remove_pegged_corrector(CORMON_DATA *CMA, CORMON_DATA *CM, STEERING_LIST *SL, ELEMENT_LIST *newly_pegged) {
  long ic0, ic1, im;
  double conditionNumber;

  if (!CM->remove_pegged)
    return 0;

  if (newly_pegged) {
    for (ic0 = 0; ic0 < CM->ncor; ic0++) {
      if (CM->ucorr[ic0] == newly_pegged) {
        if (CM->pegged[ic0] == 2)
          bombElegant("Previously pegged corrector has pegged again---shouldn't happen. (4)", NULL);
        /* printf("Corrector %ld is pegged\n", ic0); */
        CM->pegged[ic0] = 2;
        break;
      }
    }
  }

  clean_up_CM(CMA, 0);
  copy_CM_structure(CMA, CM);

  CMA->ncor = 0;
  for (ic0 = 0; ic0 < CM->ncor; ic0++) {
    if (!CM->pegged[ic0])
      CMA->ncor += 1;
    else {
      ELEMENT_LIST *corr;
      long sl_index, kick_offset;
      double param;
      corr = CM->ucorr[ic0];
      sl_index = CM->sl_index[ic0];
      kick_offset = SL->param_offset[sl_index];
      param = *((double *)(corr->p_elem + kick_offset));
      printf("Corrector %ld, %s#%ld, is pegged at %le\n",
             ic0, corr->name, corr->occurence, param);
      CM->pegged[ic0] = 2;
    }
  }

  if (CMA->ncor == 0)
    return 0;

  CMA->ucorr = tmalloc(sizeof(*CMA->ucorr) * CMA->ncor);
  CMA->sl_index = tmalloc(sizeof(*CMA->sl_index) * CMA->ncor);
  CMA->kick = (double **)czarray_2d(sizeof(**(CMA->kick)), CMA->n_iterations + 1, CMA->ncor);
  CMA->C = matrix_get(CMA->nmon = CM->nmon, CMA->ncor);
  CMA->kick_coef = tmalloc(sizeof(*CMA->kick_coef) * CMA->ncor);
  CMA->pegged = tmalloc(sizeof(*CMA->pegged) * CMA->ncor);

  for (ic0 = ic1 = 0; ic0 < CM->ncor; ic0++) {
    if (!CM->pegged[ic0]) {
      CMA->ucorr[ic1] = CM->ucorr[ic0];
      CMA->sl_index[ic1] = CM->sl_index[ic0];
      CMA->kick_coef[ic1] = CM->kick_coef[ic0];
      CMA->pegged[ic1] = 0;
      for (im = 0; im < CM->nmon; im++)
        Mij(CMA->C, im, ic1) = Mij(CM->C, im, ic0);
      ic1++;
    }
  }

  if (CMA->auto_limit_SVs && (CMA->C->m < CMA->C->n) && CMA->remove_smallest_SVs < (CMA->C->n - CMA->C->m)) {
    CMA->remove_smallest_SVs = CMA->C->n - CMA->C->m;
    printf("Removing %ld smallest singular values to prevent instability\n", (long)CMA->remove_smallest_SVs);
  }
  CMA->T = matrix_invert(CMA->C, CMA->equalW ? NULL : CMA->weight, (int32_t)CMA->keep_largest_SVs, (int32_t)CMA->remove_smallest_SVs,
                         CMA->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                         0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
  printf("Inverted matrix with %ld (of %ld) correctors active, condition number = %le\n", (long)CMA->ncor, (long)CM->ncor, conditionNumber);
  matrix_scmul(CMA->T, -1);
  return 0;
}

void copy_CM_structure(CORMON_DATA *CMA, CORMON_DATA *CM) {
  memcpy(CMA, CM, sizeof(*CMA));
  CMA->C = NULL;
  CMA->T = NULL;

  CMA->ucorr = NULL;
  CMA->kick = NULL;
  CMA->kick_coef = NULL;
  CMA->sl_index = NULL;
  CMA->pegged = NULL;
  CMA->ncor = 0;
}

void finishCorrectionOutput() {
  finish_orb_traj_output();
  finish_bpm_output();
  finish_corrector_output();
  finish_cormon_stats();
}

long preemptivelyFindPeggedCorrectors(
  CORMON_DATA *CM,
  STEERING_LIST *SL) {
  long i_corr, sl_index, kick_offset, nNewPegged;
  double param;
  ELEMENT_LIST *corr;

  nNewPegged = 0;
  if (!CM->remove_pegged)
    return 0;

  for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
    corr = CM->ucorr[i_corr];
    sl_index = CM->sl_index[i_corr];
    kick_offset = SL->param_offset[sl_index];
    param = fabs(*((double *)(corr->p_elem + kick_offset)));
    if (SL->corr_limit[sl_index] && param >= SL->corr_limit[sl_index] && !CM->pegged[i_corr]) {
      /*
      printf("Marking corrector %ld, %s#%ld, as pegged\n",
	      i_corr, CM->ucorr[i_corr]->name, CM->ucorr[i_corr]->occurence);
      */
      CM->pegged[i_corr] = 1;
      nNewPegged++;
    }
  }
  return nNewPegged;
}

void compute_coupled_trajcor_matrices(
  CORMON_DATA *CM, STEERING_LIST *SL, RUN *run, LINE_LIST *beamline, unsigned long flags) {
  ELEMENT_LIST *corr;
  TRAJECTORY *traj0, *traj1;
  long kick_offset, i_corr, i_moni, i, corrPlane;
  long n_part;
  double **one_part, p, p0, kick0, corr_tweek, corrCalibration, *moniCalibration, W0 = 0.0;
  double conditionNumber;
  VMATRIX *save;

  find_useable_moni_corr(&CM->nmon, &CM->ncor, &CM->mon_index,
                         &CM->umoni, &CM->ucorr, &CM->kick_coef, &CM->sl_index,
                         &CM->pegged, &CM->weight, 4, SL, run, beamline, 0);

  /* We don't need these coefficients since the response matrix includes them */
  for (i_corr = 0; i_corr < CM->ncor; i_corr++)
    CM->kick_coef[i_corr] = 1;

#ifdef DEBUG
  for (i = 0; i < CM->ncor; i++) {
    long sl_index;
    sl_index = CM->sl_index[i];
    printf("Corrector %ld: %s#%ld at s=%le, sl_index = %ld, kick_coef = %le, type = %s/%s, param_offset = %ld, param_name = %s\n",
           i, CM->ucorr[i]->name, CM->ucorr[i]->occurence, CM->ucorr[i]->end_pos,
           CM->sl_index[i], CM->kick_coef[i],
           entity_name[SL->elem[sl_index]->type], entity_name[CM->ucorr[i]->type],
           SL->param_offset[sl_index],
           entity_description[CM->ucorr[i]->type].parameter[SL->param_index[sl_index]].name);
  }
#endif

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0)
    printWarning("Coupled trajectory correction may be unstable (consider use of SV controls).",
                 "More correctors than monitors.");
  if (CM->ncor == 0) {
    printWarning("Coupled trajectory correction not performed.",
                 "No correctors for coupled correction.");
    return;
  }
  if (CM->nmon == 0) {
    printWarning("Coupled trajectory correction not performed.",
                 "No monitors for coupled correction.");
    CM->ncor = 0;
    return;
  }

  if (flags & COMPUTE_RESPONSE_FINDONLY)
    return;

  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    printf("computing response matrix...\n");
    fflush(stdout);
    report_stats(stdout, "start");
  }

  /* allocate matrices for this plane */
  if (CM->C)
    matrix_free(CM->C);
  CM->C = matrix_get(CM->nmon, CM->ncor); /* Response matrix */
  if (CM->Cp)
    matrix_free(CM->Cp);
  CM->Cp = matrix_get(CM->nmon, CM->ncor); /* Slope response matrix */
  if (CM->T)
    matrix_free(CM->T);
  CM->T = NULL;

  /* arrays for trajectory data */
  traj0 = tmalloc(sizeof(*traj0) * beamline->n_elems);
  traj1 = tmalloc(sizeof(*traj1) * beamline->n_elems);
  one_part = (double **)czarray_2d(sizeof(**one_part), 1, totalPropertiesPerParticle);

  /* find initial trajectory */
  p = p0 = sqrt(sqr(run->ideal_gamma) - 1);
  n_part = 1;
  fill_double_array(*one_part, totalPropertiesPerParticle, 0.0);
  if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   traj0, run, 0,
                   TEST_PARTICLES + TIME_DEPENDENCE_OFF, 1, 0, NULL, NULL, NULL, NULL, NULL))
    bombElegant("tracking failed for test particle (compute_trajcor_matrices())", NULL);

  /* set up weight matrix and monitor calibration array */
  moniCalibration = tmalloc(sizeof(*moniCalibration) * CM->nmon);
  CM->equalW = 1;
  for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
    CM->weight[i_moni] = getMonitorWeight(CM->umoni[i_moni]);
    if (!i_moni)
      W0 = CM->weight[i_moni];
    else if (W0 != CM->weight[i_moni])
      CM->equalW = 0;
    moniCalibration[i_moni] = getMonitorCalibration(CM->umoni[i_moni],
                                                    CM->umoni[i_moni]->type == T_HMON ? 0 : 2);
  }

  for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
    corr = CM->ucorr[i_corr];
    kick_offset = SL->param_offset[CM->sl_index[i_corr]];
    corr_tweek = SL->corr_tweek[CM->sl_index[i_corr]];

    /* record value of corrector */
    kick0 = *((double *)(corr->p_elem + kick_offset));

    /* change the corrector by corr_tweek and compute the new matrix for the corrector */
    *((double *)(corr->p_elem + kick_offset)) = kick0 + corr_tweek;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);

    if (corr->matrix) {
      save = corr->matrix;
      corr->matrix = NULL;
    } else
      save = NULL;
    compute_matrix(corr, run, NULL);

    /* track with positively-tweeked corrector */
    p = p0;
    n_part = 1;
    fill_double_array(*one_part, totalPropertiesPerParticle, 0.0);
    if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                     traj1, run, 0, TEST_PARTICLES + TIME_DEPENDENCE_OFF, 1, 0, NULL, NULL, NULL, NULL, NULL))
      bombElegant("tracking failed for test particle (compute_trajcor_matrices())", NULL);

    corrPlane = CM->ucorr[i_corr]->type == T_HCOR || CM->ucorr[i_corr]->type == T_EHCOR ? 0 : 2;
    corrCalibration = getCorrectorCalibration(CM->ucorr[i_corr], corrPlane) / corr_tweek;

    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      long coord;
      i = CM->mon_index[i_moni];
      coord = 0;
      if (CM->umoni[i_moni]->type != T_HMON)
        coord = 2;
      Mij(CM->C, i_moni, i_corr) = corrCalibration *
	(computeMonitorReading(CM->umoni[i_moni], coord, traj1[i].centroid, 0) - computeMonitorReading(CM->umoni[i_moni], coord, traj0[i].centroid, 0));
      Mij(CM->Cp, i_moni, i_corr) = corrCalibration*
        (computeMonitorReading(CM->umoni[i_moni], coord+1, traj1[i].centroid, 0) - computeMonitorReading(CM->umoni[i_moni], coord+1, traj0[i].centroid, 0));
    }

    /* change the corrector back */
    *((double *)(corr->p_elem + kick_offset)) = kick0;
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
    if (corr->matrix) {
      free_matrices(corr->matrix);
      free(corr->matrix);
      corr->matrix = NULL;
    }
    if (save) {
      corr->matrix = save;
      save = NULL;
    } else
      compute_matrix(corr, run, NULL);
  }
  free(moniCalibration);
  tfree(traj0);
  traj0 = NULL;
  tfree(traj1);
  traj1 = NULL;
  free_czarray_2d((void **)one_part, 1, totalPropertiesPerParticle);
  one_part = NULL;
#ifdef DEBUG
  matrix_show(CM->C, "%13.6le ", "influence matrix\n", stdout);
#endif
  if (!(flags & COMPUTE_RESPONSE_SILENT)) {
    report_stats(stdout, "done");
  }

  if (flags & COMPUTE_RESPONSE_INVERT) {
    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "Computing correction matrix ");
      fflush(stdout);
    }

    /* compute correction matrix T */
    if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (CM->C->n - CM->C->m)) {
      CM->remove_smallest_SVs = CM->C->n - CM->C->m;
      printf("Removing %ld smallest singular values to prevent instability\n", (long)CM->remove_smallest_SVs);
    }

    CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight, (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                          CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                          0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
    matrix_scmul(CM->T, -1);

    if (!(flags & COMPUTE_RESPONSE_SILENT)) {
      report_stats(stdout, "\ndone.");
      printf("Condition number is %e\n", conditionNumber);
      fflush(stdout);
    }
  }

#ifdef DEBUG
  matrix_show(CM->C, "%13.6le ", "response matrix\n", stdout);
  matrix_show(CM->T, "%13.6le ", "correction matrix\n", stdout);
#endif
}

long global_coupled_trajcor(CORMON_DATA *CM, STEERING_LIST *SL, TRAJECTORY **traject, long n_iterations,
                            RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, ELEMENT_LIST **newly_pegged) {
  ELEMENT_LIST *corr, *eptr;
  TRAJECTORY *traj;
  long iteration, kick_offset;
  long i_moni, i_corr;
  long n_part, i, sl_index;
  double **particle;
  double p, reading, fraction, minFraction, param, change;
  MAT *Qo, *dK;
  long i_pegged;
  unsigned long tracking_flags;

  if (!matrix_check(CM->C) || !matrix_check(CM->T))
    bombElegant("corrupted response matrix detected (global_coupled_trajcor)", NULL);
  if (!CM->mon_index)
    bombElegant("monitor index array is NULL (global_coupled_trajcor)", NULL);
  if (!CM->posi)
    bombElegant("monitor readout array is NULL (global_coupled_trajcor)", NULL);
  if (!CM->kick)
    bombElegant("corrector value array is NULL (global_coupled_trajcor)", NULL);
  if (!traject)
    bombElegant("no trajectory arrays supplied (global_coupled_trajcor)", NULL);
  if (!CM->T)
    bombElegant("no inverse matrix computed (global_coupled_trajcor)", NULL);

  Qo = matrix_get(CM->nmon, 1); /* Vector to store trajectory error at BPMs */

  if (!beam) {
    particle = (double **)czarray_2d(sizeof(**particle), 1, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES;
  } else {
    if (beam->n_to_track == 0)
      bombElegant("no particles to track in global_coupled_trajcor()", NULL);
    particle = (double **)czarray_2d(sizeof(**particle), beam->n_to_track, totalPropertiesPerParticle);
    tracking_flags = TEST_PARTICLES + TEST_PARTICLE_LOSSES;
    printf("Including loss of test particles in trajectory correction\n");
    fflush(stdout);
  }

  if (CM->nmon < CM->ncor && CM->minimum_SV_ratio == 0 && CM->keep_largest_SVs == 0 && CM->remove_smallest_SVs == 0 &&
      CM->Tikhonov_relative_alpha<=0 && CM->Tikhonov_n<0) {
    printWarning("Coupled trajectory correction may be unstable (consider use of SV controls).",
                 "More correctors than monitors.");
  }
  for (iteration = 0; iteration <= n_iterations; iteration++) {
    if (!CM->posi[iteration])
      bombElegant("monitor readout array for this iteration is NULL (global_coupled_trajcor)", NULL);
    if (!CM->kick[iteration])
      bombElegant("corrector value array for this iteration is NULL (global_coupled_trajcor)", NULL);

    /* find trajectory */
    p = sqrt(sqr(run->ideal_gamma) - 1);

    if (!beam) {
      if (!starting_coord)
        fill_double_array(*particle, totalPropertiesPerParticle, 0.0);
      else {
        for (i = 0; i < 6; i++)
          particle[0][i] = starting_coord[i];
      }
      n_part = 1;
    } else
      copy_particles(particle, beam->particle, n_part = beam->n_to_track);

    n_part = do_tracking(NULL, particle, n_part, NULL, beamline, &p, (double **)NULL,
                         (BEAM_SUMS **)NULL, (long *)NULL,
                         traj = traject[iteration == 0 ? 0 : 1], run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL);
    if (beam) {
      printf("%ld particles survived tracking", n_part);
      fflush(stdout);
      if (n_part == 0) {
        for (i = 0; i < beamline->n_elems + 1; i++)
          if (traj[i].n_part == 0)
            break;
        if (i != 0 && i < beamline->n_elems + 1)
          printf("---all beam lost before z=%em (element %s)",
                 traj[i].elem->end_pos, traj[i].elem->name);
        fflush(stdout);
        fputc('\n', stdout);
      }
      fputc('\n', stdout);
    } else if (n_part == 0) {
      /* This actually should never happen given the tracking flags */
      printf("Beam lost before end of beamline during trajectory correction (flags=%lx)\n", tracking_flags);
      for (i = 0; i < beamline->n_elems + 1; i++)
        if (traj[i].n_part == 0)
          break;
      if (i != 0 && i < beamline->n_elems + 1) {
        printf("Beam lost before z=%em (element %s#%ld)\n",
               traj[i].elem->end_pos, traj[i].elem->name, traj[i].elem->occurence);
        if (i > 0)
          printf("Trajectory at end of previous element is x=%le, y=%le\n",
                 traj[i - 1].centroid[0], traj[i - 1].centroid[1]);
      }
      fflush(stdout);
    }

    /* find readings at monitors and add in reading errors */
    for (i_moni = 0; i_moni < CM->nmon; i_moni++) {
      long coord;
      if (!(eptr = traj[CM->mon_index[i_moni]].elem))
        bombElegant("invalid element pointer in trajectory array (global_coupled_trajcor)", NULL);
      coord = 0;
      if (CM->umoni[i_moni]->type != T_HMON)
        coord = 2;
      reading = computeMonitorReading(eptr, coord, traj[CM->mon_index[i_moni]].centroid, 0);
      if (isnan(reading) || isinf(reading))
        return 0;
      CM->posi[iteration][i_moni] = reading;
      Mij(Qo, i_moni, 0) = reading + (CM->bpm_noise ? noise_value(CM->bpm_noise, CM->bpm_noise_cutoff, CM->bpm_noise_distribution) : 0);
    }

    if (iteration == n_iterations)
      break;

    /* solve for the corrector changes */
    dK = matrix_mult(CM->T, Qo);
#ifdef DEBUG
    matrix_show(Qo, "%13.6le ", "traj matrix\n", stdout);
    matrix_show(dK, "%13.6le ", "kick matrix\n", stdout);
#endif

    /* step through beamline find any kicks that are over their limits */
    minFraction = 1;
    i_pegged = -1;
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
      param = fabs(*((double *)(corr->p_elem + kick_offset)) +
                   (change = Mij(dK, i_corr, 0) / CM->kick_coef[i_corr] * CM->corr_fraction));
      if (SL->corr_limit[sl_index] && param > SL->corr_limit[sl_index]) {
        fraction = fabs((SL->corr_limit[sl_index] - fabs(*((double *)(corr->p_elem + kick_offset)))) / change);
        if (fraction < minFraction) {
          i_pegged = i_corr;
          minFraction = fraction;
        }
      }
    }
    if (i_pegged != -1) {
      *newly_pegged = CM->ucorr[i_pegged];
      CM->pegged[i_pegged] = 1;
    }
    fraction = minFraction * CM->corr_fraction;

#if defined(DEBUG)
    printf("Changing correctors:\n");
    fflush(stdout);
#endif
    /* step through beamline and change correctors */
    for (i_corr = 0; i_corr < CM->ncor; i_corr++) {
      corr = CM->ucorr[i_corr];
      sl_index = CM->sl_index[i_corr];
      kick_offset = SL->param_offset[sl_index];
#if defined(DEBUG)
      printf("name = %s#%ld, before = %e, ", corr->name, corr->occurence, *((double *)(corr->p_elem + kick_offset)));
      fflush(stdout);
#endif
      if (iteration == 0)
        CM->kick[iteration][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
      *((double *)(corr->p_elem + kick_offset)) += Mij(dK, i_corr, 0) * fraction / CM->kick_coef[i_corr];
      if (SL->corr_limit[sl_index] && fabs(*((double *)(corr->p_elem + kick_offset))) > (1 + 1e-6) * SL->corr_limit[sl_index]) {
        printf("**** Corrector %s#%ld went past limit (%e > %e) --- This shouldn't happen (1)\n",
               corr->name, corr->occurence, fabs(*((double *)(corr->p_elem + kick_offset))), SL->corr_limit[sl_index]);
        printf("fraction=%e -> kick = %e\n", fraction, *((double *)(corr->p_elem + kick_offset)));
      }
      CM->kick[iteration + 1][i_corr] = *((double *)(corr->p_elem + kick_offset)) * CM->kick_coef[i_corr];
#if defined(DEBUG)
      printf("after = %e, limit = %le, sl_index = %ld (coef = %le)\n", *((double *)(corr->p_elem + kick_offset)), SL->corr_limit[sl_index], sl_index, CM->kick_coef[i_corr]);
      fflush(stdout);
#endif
      if (corr->matrix) {
        free_matrices(corr->matrix);
        tfree(corr->matrix);
        corr->matrix = NULL;
      }
      compute_matrix(corr, run, NULL);
    }
    matrix_free(dK);
    if (beamline->links)
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
  }

  /* indicate that beamline concatenation and Twiss parameter computation (if wanted) are not current */
  beamline->flags &= ~BEAMLINE_CONCAT_CURRENT;
  beamline->flags &= ~BEAMLINE_TWISS_CURRENT;

  if (!beam)
    free_czarray_2d((void **)particle, 1, totalPropertiesPerParticle);
  else
    free_czarray_2d((void **)particle, beam->n_to_track, totalPropertiesPerParticle);
  particle = NULL;
  log_exit("global_coupled_trajcor");

  matrix_free(Qo);

  return iteration;
}

void reorderCorrectorArray(ELEMENT_LIST **ucorr, long *sl_index, double *kick_coef, long ncor) {
  long *index, i;
  double *sData;
  ELEMENT_LIST **ucorr2;
  long *sl_index2;
  double *kick_coef2;

  sData = tmalloc(sizeof(*sData) * ncor);
  for (i = 0; i < ncor; i++)
    sData[i] = ucorr[i]->end_pos;

  index = sort_and_return_index(sData, SDDS_DOUBLE, ncor, 1);

  ucorr2 = tmalloc(sizeof(*ucorr) * ncor);
  sl_index2 = tmalloc(sizeof(*sl_index2) * ncor);
  kick_coef2 = tmalloc(sizeof(*kick_coef2) * ncor);

  memcpy(ucorr2, ucorr, sizeof(*ucorr2) * ncor);
  memcpy(sl_index2, sl_index, sizeof(*sl_index2) * ncor);
  memcpy(kick_coef2, kick_coef, sizeof(*kick_coef2) * ncor);

  for (i = 0; i < ncor; i++) {
    ucorr[i] = ucorr2[index[i]];
    sl_index[i] = sl_index2[index[i]];
    kick_coef[i] = kick_coef2[index[i]];
  }

  free(index);
  free(ucorr2);
  free(sl_index2);
  free(kick_coef2);
}
