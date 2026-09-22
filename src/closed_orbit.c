/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: closed_orbit.c
 * purpose: computation closed orbits
 *
 * Michael Borland, 1992
 */
#include "mdb.h"
#include "track.h"
#include "matlib.h"

long findFixedLengthClosedOrbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_req, long clorb_iter, LINE_LIST *beamline,
                                VMATRIX *M, RUN *run, double dp, long start_from_recirc, double *starting_point,
                                double change_fraction, double change_fraction_multiplier, long multiplier_interval, double *deviation, long n_turns);
long findSixDimClosedOrbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_req, long clorb_iter, LINE_LIST *beamline,
                           VMATRIX *M, RUN *run, long start_from_recirc, double *starting_point,
                           double change_fraction, double change_fraction_multiplier, long multiplier_interval, double *deviation, long n_turns);

static long SDDS_clorb_initialized = 0;
static SDDS_TABLE SDDS_clorb;
static long clorb_count = 0;

#define IC_S 0
#define IC_X 1
#define IC_XP 2
#define IC_Y 3
#define IC_YP 4
#define IC_ELEMENT 5
#define IC_OCCURENCE 6
#define IC_TYPE 7
#define N_COLUMNS 8
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"x", "&column name=x, units=m, type=double, description=\"Horizontal position\" &end"},
  {"xp", "&column name=xp, type=double, description=\"Horizontal slope\" &end"},
  {"y", "&column name=y, units=m, type=double, description=\"Vertical position\" &end"},
  {"yp", "&column name=yp, type=double, description=\"Vertical slope\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
};

#define IP_STEP 0
#define IP_XERROR 1
#define IP_YERROR 2
#define IP_DELTA 3
#define IP_SERROR 4
#define IP_FAILED 5
#define N_PARAMETERS 6
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"xError", "&parameter name=xError, type=double, units=m, description=\"Horizontal closed orbit convergence error\" &end"},
  {"yError", "&parameter name=yError, type=double, units=m, description=\"Vertical closed orbit convergence error\" &end"},
  {"delta", "&parameter name=delta, symbol=\"$gd$r\", type=double, description=\"Fractional energy offset of closed orbit\" &end"},
  {"lengthError", "&parameter name=lengthError, type=double, units=m, description=\"Deviation of orbit length from reference orbit length\" &end"},
  {"failed", "&parameter name=failed, type=short, description=\"Non-zero if orbit determination failed\" &end"},
};

#include "closed_orbit.h"

static TRAJECTORY *clorb = NULL;

long setup_closed_orbit(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {

  log_entry("setup_closed_orbit");

  if (clorb)
    free(clorb);
  clorb = tmalloc(sizeof(*clorb) * (beamline->n_elems + 1));

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&closed_orbit, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &closed_orbit);
  if (disable)
    return 0;

#if (USE_MPI)
  if (isSlave)
    output = NULL;
#endif
  if (output)
    output = compose_filename(output, run->rootname);
  if (closed_orbit_accuracy <= 0)
    bombElegant("closed_orbit_accuracy <= 0", NULL);
  if (closed_orbit_accuracy_requirement <= 0)
    bombElegant("closed_orbit_accuracy_requirement <= 0", NULL);
  if (closed_orbit_iterations < 1)
    bombElegant("closed_orbit_iterations < 1", NULL);
  if (iteration_fraction < 0)
    bombElegant("iteration_fraction must be >= 0", NULL);
  if (iteration_fraction > 1)
    printWarning("closed_orbit: iteration_fraction>1.", "This may be an error and lead to divergent orbits.");
  if (fraction_multiplier < 1)
    bombElegant("fraction_multiplier must not be less than 1", NULL);
  if (multiplier_interval < 1)
    bombElegant("multiplier_interval must not be less than 1", NULL);
  if (output) {
    SDDS_ElegantOutputSetup(&SDDS_clorb, output, SDDS_BINARY, 1, "closed orbit",
                            run->runfile, run->lattice, parameter_definition, N_PARAMETERS,
                            column_definition, N_COLUMNS, "setup_closed_orbit",
                            SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    SDDS_clorb_initialized = 1;
  }
  if (fixed_length < 0 || fixed_length > 2)
    bombElegant("fixed_length must be 0 (vary RF frequency), 1 (vary energy, momentum secant), or 2 (full 6-D closed orbit).", NULL);
  if (fixed_length == 1 && checkChangeT(beamline))
    bombElegant("change_t is nonzero on one or more RF cavities. This is incompatible with fixed_length=1 orbit computations.", NULL);

  log_exit("setup_closed_orbit");
  return 1 + immediate;
}

long checkChangeT(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  long change_t = 0;
  eptr = beamline->elem;
  while (eptr && change_t == 0) {
    switch (eptr->type) {
    case T_RFCA:
      change_t = ((RFCA *)eptr->p_elem)->change_t;
      break;
    case T_RFCW:
      change_t = ((RFCW *)eptr->p_elem)->change_t;
      break;
    default:
      break;
    }
    eptr = eptr->succ;
  }
  return change_t;
}

/* Save and zero change_t on every rf cavity for the duration of the fixed_length=2 6-D solve. With
 * change_t!=0 the cavity subtracts a whole number of rf fundamental periods from coord[4] each turn,
 * so coord[4] is carried as a reduced-time deviation; forcing change_t=0 keeps coord[4] as the
 * absolute path length (nominally the circumference), which is what the r[4]=(F[4]-X[4])-revolution
 * length constraint requires. The rf energy kick still depends on arrival time, so longitudinal
 * focusing (R[5][4]!=0) is preserved. Returns a malloc'd array of the saved values (caller passes it
 * to restoreRfChangeT, which frees it) and the count via *n. */
static long *saveAndZeroRfChangeT(LINE_LIST *beamline, long *n) {
  ELEMENT_LIST *eptr;
  long count = 0, *saved = NULL, i = 0;
  eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_RFCA || eptr->type == T_RFCW)
      count++;
    eptr = eptr->succ;
  }
  *n = count;
  if (count == 0)
    return NULL;
  saved = tmalloc(sizeof(*saved) * count);
  eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_RFCA) {
      saved[i++] = ((RFCA *)eptr->p_elem)->change_t;
      ((RFCA *)eptr->p_elem)->change_t = 0;
    } else if (eptr->type == T_RFCW) {
      saved[i++] = ((RFCW *)eptr->p_elem)->change_t;
      ((RFCW *)eptr->p_elem)->change_t = 0;
    }
    eptr = eptr->succ;
  }
  return saved;
}

static void restoreRfChangeT(LINE_LIST *beamline, long *saved, long n) {
  ELEMENT_LIST *eptr;
  long i = 0;
  if (!saved)
    return;
  eptr = beamline->elem;
  while (eptr && i < n) {
    if (eptr->type == T_RFCA)
      ((RFCA *)eptr->p_elem)->change_t = saved[i++];
    else if (eptr->type == T_RFCW)
      ((RFCW *)eptr->p_elem)->change_t = saved[i++];
    eptr = eptr->succ;
  }
  free(saved);
}

/* sixDimRfPathTarget: for fixed_length=2, compute the path length one turn must have so the beam
 * synchronizes to the rf, i.e. one revolution equals an integer number (the harmonic h) of rf
 * periods. This replaces the purely geometric revolution_length as the target in the r[4] residual,
 * so that the closed orbit responds to the rf frequency (an off-design frequency shifts the orbit
 * radially through a nonzero delta).
 *
 * elegant carries coord[4] = beta*c*t (simple_rfca.c), so the arrival time is t = coord[4]/(beta*c)
 * and the rf phase repeats when f_rf * t = h. With coord[4] the per-turn path length this gives
 *     target = h * beta0 * c / f_rf,
 * where beta0 is the reference velocity (from p_central). h is the fixed integer harmonic, inferred
 * once from the ACTUAL rf frequency and the design circumference:
 *     h = round(f_rf * revolution_length / (beta0 * c)).
 * At the matched frequency f0 = h*beta0*c/revolution_length this reduces to target=revolution_length,
 * recovering the previous behavior exactly.
 *
 * The base frequency is the lowest positive rf frequency among RFCA/RFCW cavities that actually focus
 * (freq>0 and volt!=0). Harmonic cavities are expected to be near-integer multiples of it; a warning
 * is issued for any that are not (no single closed orbit can synchronize incommensurate cavities).
 * If no qualifying cavity is found, revolution_length is returned and *hOut=0 (the caller's
 * R[5][4]!=0 requirement then bombs with the "requires longitudinal focusing" message). Using beta0
 * (not the solved beta(delta)) keeps target constant, so the Jacobian J=R-I stays exact; the error
 * is O(1-beta0), negligible for relativistic rings. */
static double sixDimRfPathTarget(LINE_LIST *beamline, double p_central, long *hOut) {
  ELEMENT_LIST *eptr;
  double fBase = 0, beta0, target, h;
  double volt, freq;

  /* lowest positive focusing frequency */
  eptr = beamline->elem;
  while (eptr) {
    freq = volt = 0;
    if (eptr->type == T_RFCA) {
      freq = ((RFCA *)eptr->p_elem)->freq;
      volt = ((RFCA *)eptr->p_elem)->volt;
    } else if (eptr->type == T_RFCW) {
      freq = ((RFCW *)eptr->p_elem)->freq;
      volt = ((RFCW *)eptr->p_elem)->volt;
    }
    if (freq > 0 && volt != 0 && (fBase == 0 || freq < fBase))
      fBase = freq;
    eptr = eptr->succ;
  }

  if (fBase == 0) {
    if (hOut)
      *hOut = 0;
    return beamline->revolution_length;
  }

  beta0 = beta_from_delta(p_central, 0.0);
  h = round(fBase * beamline->revolution_length / (beta0 * c_mks));
  if (h < 1)
    h = 1;
  target = h * beta0 * c_mks / fBase;

  /* warn if other cavities are not near-integer harmonics of the base */
  eptr = beamline->elem;
  while (eptr) {
    freq = volt = 0;
    if (eptr->type == T_RFCA) {
      freq = ((RFCA *)eptr->p_elem)->freq;
      volt = ((RFCA *)eptr->p_elem)->volt;
    } else if (eptr->type == T_RFCW) {
      freq = ((RFCW *)eptr->p_elem)->freq;
      volt = ((RFCW *)eptr->p_elem)->volt;
    }
    if (freq > 0 && volt != 0) {
      double ratio = freq / fBase;
      if (fabs(ratio - round(ratio)) > 1e-4) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer),
                 "cavity %s frequency %.6g is not a near-integer multiple of the base rf frequency "
                 "%.6g (ratio %.6g); the fixed_length=2 closed orbit synchronizes only to the base.",
                 eptr->name ? eptr->name : "?", freq, fBase, ratio);
        printWarning("closed_orbit: incommensurate rf frequencies in fixed_length=2 6-D closed orbit", buffer);
      }
    }
    eptr = eptr->succ;
  }

  if (hOut)
    *hOut = (long)h;
  return target;
}

long run_closed_orbit(RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, unsigned long flags) {
  double dp, deviation[6];
  long i, bad_orbit;
  VMATRIX *M;
  long do_output;

  do_output = flags & CLOSED_ORBIT_OUTPUT;
#if USE_MPI
  if (isSlave)
    do_output = 0;
#endif

  if (!starting_coord)
    bombElegant("starting_coord array is NULL (run_closed_orbit)", NULL);

  start_from_centroid = start_from_dp_centroid = 0;

  if ((start_from_centroid || start_from_dp_centroid) && !(flags & CLOSED_ORBIT_IGNORE_BEAM)) {
    double initial[6];
    if (!beam)
      bombElegant("no beam present for closed-orbit calculation starting from centroid", NULL);
    compute_centroids(initial, beam->particle, beam->n_to_track);
    if (start_from_centroid)
      memcpy(starting_coord, initial, 6 * sizeof(*starting_coord));
    dp = initial[5];
  } else
    dp = 0;

  if (verbosity && do_output) {
    printf("Starting point for closed orbit\n");
    for (i = 0; i < 6; i++)
      printf("%e%s", starting_coord[i], i == 5 ? "\n" : ", ");
  }

  if (!clorb)
    bombElegant("TRAJECTORY array for closed orbit not allocated (run_closed_orbit)", NULL);
  beamline->closed_orbit = clorb;

  if (beamline->elem_recirc)
    M = full_matrix(beamline->elem_recirc, run, 1);
  else
    M = full_matrix(beamline->elem, run, 1);

  bad_orbit = !find_closed_orbit(clorb, closed_orbit_accuracy, closed_orbit_accuracy_requirement,
                                 closed_orbit_iterations, beamline, M,
                                 run, dp, start_from_recirc, fixed_length,
                                 starting_coord, iteration_fraction,
                                 fraction_multiplier, multiplier_interval,
                                 deviation, tracking_turns);
  free_matrices(M);
  tfree(M);
  M = NULL;

  /* return closed orbit at the beginning of the ring */
  for (i = 0; i < 6; i++)
    starting_coord[i] = clorb[0].centroid[i];

  /* do output, if required */
  if (verbosity && !bad_orbit && do_output) {
    printf("closed orbit: \n");
    for (i = 0; i < 6; i++)
      printf("%.8e ", starting_coord[i]);
    fputc('\n', stdout);
    fflush(stdout);
  }

  if (do_output && SDDS_clorb_initialized)
    dump_closed_orbit(clorb, beamline->n_elems, clorb_count++, deviation, bad_orbit);

  return !bad_orbit;
}

void finish_clorb_output(void) {
  log_entry("finish_clorb_output");
  if (SDDS_IsActive(&SDDS_clorb) && !SDDS_Terminate(&SDDS_clorb)) {
    SDDS_SetError("Problem terminating SDDS output (finish_clorb_output)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDS_clorb_initialized = clorb_count = 0;
  log_exit("finish_clorb_output");
}

void dump_closed_orbit(TRAJECTORY *traj, long n_elems, long step, double *deviation, long bad_orbit) {
  long i, n, occurence, row;
  double position;
  char *name;

  log_entry("dump_closed_orbit");

  if (!SDDS_clorb_initialized)
    return;

  /* count number of trajectory elements actually used */
  for (i = 1; i < n_elems + 1; i++) {
    if (!traj[i].elem)
      break;
  }
  n = i;

  if (!SDDS_StartTable(&SDDS_clorb, n)) {
    SDDS_SetError("Unable to start SDDS table (dump_closed_orbit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_SetParameters(&SDDS_clorb, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IP_STEP, step,
                          IP_XERROR, deviation[0],
                          IP_YERROR, deviation[2],
                          IP_DELTA, traj[0].centroid[5],
                          IP_SERROR, deviation[4],
                          IP_FAILED, bad_orbit ? (short)1 : (short)0,
                          -1)) {
    SDDS_SetError("Unable to set SDDS parameters (dump_closed_orbit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  position = traj[1].elem->end_pos -
             (entity_description[traj[1].elem->type].flags & HAS_LENGTH ? *((double *)traj[1].elem->p_elem) : 0.0);
  name = "_BEG_";
  occurence = 1;

  for (i = row = 0; i < n; i++) {
    if (i) {
      position = traj[i].elem->end_pos;
      name = traj[i].elem->name;
      occurence = traj[i].elem->occurence;
    }
    if (output_monitors_only &&
        (i == 0 ||
         !(traj[i].elem->type == T_MONI || traj[i].elem->type == T_HMON || traj[i].elem->type == T_VMON)))
      continue;
    if (!SDDS_SetRowValues(&SDDS_clorb, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row++,
                           IC_S, position, IC_X, traj[i].centroid[0], IC_Y, traj[i].centroid[2],
                           IC_XP, traj[i].centroid[1], IC_YP, traj[i].centroid[3],
                           IC_ELEMENT, name, IC_OCCURENCE, occurence,
                           IC_TYPE, i == 0 ? "MARK" : entity_name[traj[i].elem->type], -1)) {
      printf("Unable to set row %ld values (dump_closed_orbit)\n", i);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

  if (!SDDS_WriteTable(&SDDS_clorb)) {
    SDDS_SetError("Unable to write closed orbit data (dump_closed_orbit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_clorb);
  if (!SDDS_EraseData(&SDDS_clorb)) {
    SDDS_SetError("Unable to erase closed orbit data (dump_closed_orbit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  log_exit("dump_closed_orbit");
}

long find_closed_orbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_requirement,
                       long clorb_iter, LINE_LIST *beamline, VMATRIX *M, RUN *run,
                       double dp, long start_from_recirc, long fixed_length, double *starting_point, double change_fraction,
                       double fraction_multiplier, long multiplier_interval,
                       double *deviation, long n_turns) {
  static MATRIX *R, *ImR, *INV_ImR, *INV_R, *C, *co, *diff, *change;
  static VMATRIX *Mco, *Mnew;
  static double **one_part;
  static long initialized = 0;
  long i, j, n_iter = 0, bad_orbit = 0, second_try;
  long n_part, method, goodCount, convergenceProblem = 0;
  double p, error, last_error, reference_error;
#ifdef CLORB_DEBUG
  static FILE *fpClorb = NULL;
  if (!fpClorb) {
    fpClorb = fopen("closed_orbit_debug.sdds", "w");
    fprintf(fpClorb, "SDDS1\n&column name=Iteration type=short &end\n");
    fprintf(fpClorb, "&column name=x0 type=double units=m &end\n");
    fprintf(fpClorb, "&column name=xp0 type=double &end\n");
    fprintf(fpClorb, "&column name=y0 type=double units=m &end\n");
    fprintf(fpClorb, "&column name=yp0 type=double &end\n");
    fprintf(fpClorb, "&column name=delta0 type=double &end\n");
    fprintf(fpClorb, "&column name=p type=double &end\n");
    fprintf(fpClorb, "&column name=fraction type=double &end\n");
    fprintf(fpClorb, "&column name=error type=double &end\n");
    fprintf(fpClorb, "&column name=method type=short &end\n");
    fprintf(fpClorb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif
  
  log_entry("find_closed_orbit");

  if (fixed_length == 1)
    return findFixedLengthClosedOrbit(clorb, clorb_acc, clorb_acc_requirement, clorb_iter, beamline, M, run, dp,
                                      start_from_recirc, starting_point, change_fraction, fraction_multiplier,
                                      multiplier_interval, deviation, n_turns);
  else if (fixed_length == 2)
    return findSixDimClosedOrbit(clorb, clorb_acc, clorb_acc_requirement, clorb_iter, beamline, M, run,
                                 start_from_recirc, starting_point, change_fraction, fraction_multiplier,
                                 multiplier_interval, deviation, n_turns);
  
#if SDDS_MPI_IO
  long distributedBeam_orig = distributedBeam; /* We need save the original value to switch it back */
  distributedBeam = 0;                       /* run as single particle mode, i.e., all processors will do the same thing */
#endif

#ifdef DEBUG
  printf("running find_closed_orbit: clorb_acc=%le, clorb_iter=%ld, dp=%le, start_from_recirc=%ld, fixed_length=%ld, change_fraction=%le, n_turns = %ld\n",
         clorb_acc, clorb_iter, dp, start_from_recirc, fixed_length, change_fraction, n_turns);
#endif

  /* method for finding closed orbit: 
   * 1. solve co[i] = C[i] + R[i][j]*co[j] for co[i]:
   *        co = INV(I-R)*C
   * 2. use Newton's method iteration starting with this solution:
   *        dco = INV(R)*(-co + F(co))
   *    where F(co) returns the coordinates at the end for starting
   *    coordinates co.
   */

  if (!M)
    bombElegant("no transport matrix passed to find_closed_orbit()", NULL);
  if (!M->R)
    bombElegant("faulty transport matrix passed to find_closed_orbit()", NULL);

  if (!initialized) {
    Mco = malloc(sizeof(*Mco));
    Mnew = malloc(sizeof(*Mnew));
    initialize_matrices(Mco, 1);
    initialize_matrices(Mnew, 2);
    m_alloc(&ImR, 4, 4);
    m_alloc(&R, 4, 4);
    m_alloc(&INV_ImR, 4, 4);
    m_alloc(&INV_R, 4, 4);
    m_alloc(&C, 4, 1);
    m_alloc(&co, 4, 1);
    m_alloc(&diff, 4, 1);
    m_alloc(&change, 4, 1);
    one_part = (double **)czarray_2d(sizeof(**one_part), 1, totalPropertiesPerParticle);
    initialized = 1;
  }

  for (i = 0; i < 4; i++) {
    C->a[i][0] = M->C[i];
    for (j = 0; j < 4; j++) {
      R->a[i][j] = M->R[i][j];
      ImR->a[i][j] = (i == j ? 1 : 0) - R->a[i][j];
    }
  }

  if (!m_invert(INV_ImR, ImR)) {
    printf("error: unable to invert matrix to find closed orbit (1)!\nThe transport matrix is:\n");
    fflush(stdout);
    for (i = 0; i < 4; i++)
      printf("C[%ld]: %le\n", i + 1, C->a[i][0]);
    for (i = 0; i < 4; i++) {
      printf("R[%ld]: ", i + 1);
      for (j = 0; j < 4; j++)
        printf("%14.6e ", R->a[i][j]);
      fputc('\n', stdout);
    }
    fflush(stdout);
    for (i = 0; i < 4; i++) {
      for (j = 0; j < 4; j++) {
        INV_ImR->a[i][j] = 0;
      }
    }
  }

  if (!starting_point) {
    if (!m_mult(co, INV_ImR, C))
      bombElegant("unable to solve for closed orbit--matrix multiplication error", NULL);
    for (i = 0; i < 4; i++)
      one_part[0][i] = co->a[i][0];
    one_part[0][4] = 0;
    one_part[0][5] = dp;
  } else {
    for (i = 0; i < 4; i++)
      one_part[0][i] = co->a[i][0] = starting_point[i];
    one_part[0][4] = 0;
    one_part[0][5] = dp;
  }

  p = run->p_central;
  if (deviation)
    deviation[4] = deviation[5] = 0;
  /* method=0: iterate using the R matrix; only invoked if n_turns>=0
   * method=1: track a specified number of turns, given by |n_turns|; only invoked if |n_turns|>0
   * method=2: iterate using the R matrix again, starting from tracking result (n_turns>0); or, fill trajectory buffer (n_turns<0)
   */
  for (method = 0; method < (n_turns != 0 ? 3 : 1); method++) {
    if ((method == 0 || method == 2)) {
      if (method == 0 && n_turns < 0)
        continue;
      n_iter = 0;
      reference_error = error = DBL_MAX / 4;
      bad_orbit = 0;
      second_try = 0;
      goodCount = 0;
      do {
        n_part = 1;
#ifdef DEBUG
        printf("n_iter=%ld,  trial point: %le, %le, %le, %le, %le, %le, p=%le\n",
               n_iter, one_part[0][0], one_part[0][1], one_part[0][2],
               one_part[0][3], one_part[0][4], one_part[0][5], p);
#endif
        if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                         clorb + 1, run, 0,
                         CLOSED_ORBIT_TRACKING + TEST_PARTICLES + TIME_DEPENDENCE_OFF + (start_from_recirc ? BEGIN_AT_RECIRC : 0),
                         1, 0,
                         NULL, NULL, NULL, NULL, NULL)) {
#ifdef CLORB_DEBUG
          printf("particle lost while tracking for closed orbit!\n");
#endif
          if (n_iter == 0 && !second_try) {
            /* Try again with zero coordinate to start */
            one_part[0][0] = one_part[0][1] = one_part[0][2] = one_part[0][3] = one_part[0][4] = 0;
            one_part[0][5] = dp;
            for (i = 0; i < 4; i++)
              co->a[i][0] = one_part[0][i];
            n_iter = -1;
            second_try = 1;
            p = run->p_central;
            continue;
          }
          n_iter = clorb_iter;
          break;
        }
        for (i = 0; i < 4; i++) {
          diff->a[i][0] = one_part[0][i] - co->a[i][0];
          if (deviation)
            deviation[i] = diff->a[i][0];
        }
        if (deviation)
          deviation[4] = one_part[0][4] - beamline->revolution_length;
        last_error = error;
        error = sqrt(sqr(diff->a[0][0]) + sqr(diff->a[1][0]) + sqr(diff->a[2][0]) + sqr(diff->a[3][0]));
        if (error < reference_error)
          reference_error = error;
#ifdef CLORB_DEBUG
        fprintf(fpClorb, "%ld %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %21.15le %ld\n",
                n_iter, one_part[0][0], one_part[0][1], one_part[0][2], one_part[0][3], one_part[0][5], p, change_fraction,
                error, method);
        fflush(fpClorb);
#endif
        if (error < clorb_acc)
          break;
        if (n_turns < 0)
          break;
        if (error > reference_error) {
          reference_error = error;
          change_fraction = change_fraction / fraction_divisor;
          goodCount = -10;
          if (change_fraction < 0.01) {
            char buffer[16384];
            snprintf(buffer, 16384,
                     "accuracy requirement: %e, previous error: %e, current error: %e",
                     clorb_acc, last_error, error);
            printWarning("closed_orbit: closed orbit diverging, iteration stopped", buffer);
            n_iter = clorb_iter;
            break;
          }
          printf("reduced closed orbit iteration fraction to %e\n", change_fraction);
          fflush(stdout);
        } else {
          goodCount++;
          if (goodCount > multiplier_interval) {
            change_fraction *= fraction_multiplier;
            if (change_fraction > 1)
              change_fraction = 1;
            goodCount = 0;
#ifdef DEBUG
            printf("increased iteration fraction to %e\n", change_fraction);
#endif
          }
        }
        if (change_fraction) {
          m_mult(change, INV_ImR, diff);
          if (change_fraction != 1)
            m_scmul(change, change, change_fraction);
          m_add(co, co, change);
          for (i = 0; i < 4; i++)
            one_part[0][i] = co->a[i][0];
        } else {
          for (i = 0; i < 4; i++) {
            co->a[i][0] = (co->a[i][0] + one_part[0][i]) / 2;
            one_part[0][i] = co->a[i][0];
          }
        }
        one_part[0][4] = 0;
        one_part[0][5] = dp;

        if (update_matrix) {
          /* Update the iteration matrix. Didn't find this helped. */
          printf("Updating matrix\n");
          long i, j;
          for (i=0; i<6; i++)
            Mco->C[i] = one_part[0][i];
          for (i=0; i<6; i++)
            for (j=0; j<6; j++)
              Mco->R[i][j] = (i == j ? 1 : 0);
          concat_matrices(Mnew, M, Mco, 0);
          for (i = 0; i < 4; i++) {
            C->a[i][0] = Mnew->C[i];
            for (j = 0; j < 4; j++) {
              R->a[i][j] = Mnew->R[i][j];
              ImR->a[i][j] = (i == j ? 1 : 0) - R->a[i][j];
            }
          }
          if (!m_invert(INV_ImR, ImR)) {
            printf("error: unable to invert matrix to find closed orbit (2)! R matrix is:\n");
	    for (i = 0; i < 4; i++) {
	      printf("R[%ld]: ", i + 1);
	      for (j = 0; j < 4; j++)
		printf("%14.6e ", R->a[i][j]);
	      fputc('\n', stdout);
	    }
	    fflush(stdout);
            for (i = 0; i < 4; i++) {
              for (j = 0; j < 4; j++) {
                INV_ImR->a[i][j] = 0;
              }
            }
          }
        }
      } while (++n_iter < clorb_iter);
      if (n_iter >= clorb_iter && error > clorb_acc_requirement) {
        printf("error: closed orbit did not converge to better than %e after %ld iterations (requirement is %e)\n",
               error, n_iter, clorb_acc_requirement);
        fflush(stdout);
        if (isnan(error) || isinf(error)) {
#if SDDS_MPI_IO
          distributedBeam = distributedBeam_orig;
#endif
          return 0;
        }
        bad_orbit = 1;
        convergenceProblem = 1;
      } else {
        bad_orbit = 0;
        break;
      }

    } else {
      /* try to find a good starting point by tracking several turns */
      long turn;
      double buffer[4];
      if (convergenceProblem) {
        if (n_turns > 0)
          printf("Trying secondary, tracking-based method for orbit determination (%ld turns).\n", labs(n_turns));
        else
          printf("Using tracking-based method for orbit determination (%ld turns).\n", labs(n_turns));
        fflush(stdout);
      }
      for (i = 0; i < 4; i++)
        one_part[0][i] = buffer[i] = 0;
      one_part[0][5] = dp;
      bad_orbit = 0;
      for (turn = 0; turn < labs(n_turns); turn++) {
        n_part = 1;
        if (do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                        (TRAJECTORY *)NULL, run, 0,
                        CLOSED_ORBIT_TRACKING + TEST_PARTICLES + TIME_DEPENDENCE_OFF + (start_from_recirc ? BEGIN_AT_RECIRC : 0),
                        1, 0, NULL, NULL, NULL, NULL, NULL)) {
          for (i = 0; i < 4; i++)
            buffer[i] += one_part[0][i];
          one_part[0][5] = dp;
        } else {
          bad_orbit = 1;
          break;
        }
      }
      if (!bad_orbit) {
        for (i = 0; i < 4; i++)
          one_part[0][i] = co->a[i][0] = buffer[i] / labs(n_turns);
        one_part[0][4] = 0;
        one_part[0][5] = dp;
        /*
        if (n_turns>0) {
          printf("New CO starting point (%ld turns): %e, %e, %e, %e, %e, %e\n",
                  turn, one_part[0][0], one_part[0][1], one_part[0][2], one_part[0][3], 
                  one_part[0][4], one_part[0][5]);
          fflush(stdout);
        }
        */
      } else {
        /* set up to use the previous answer and iterate more */
        for (i = 0; i < 4; i++)
          one_part[0][i] = co->a[i][0];
        one_part[0][4] = 0;
        one_part[0][5] = dp;
      }
    }
  }

#ifdef DEBUG
  printf("final closed-orbit after %ld iterations:\n%e %e %e %e %e %e\n",
         n_iter, one_part[0][0], one_part[0][1], one_part[0][2], one_part[0][3],
         one_part[0][4], one_part[0][5]);
  fflush(stdout);
#endif
  for (i = 0; i < 4; i++)
    clorb[0].centroid[i] = one_part[0][i];
  clorb[0].centroid[4] = 0;
  clorb[0].centroid[5] = dp;

#if SDDS_MPI_IO
  distributedBeam = distributedBeam_orig; /* Switch back to original parallel tracking mode */
#endif

  log_exit("find_closed_orbit");
  if (bad_orbit)
    return (0);
  return (1);
}

long findFixedLengthClosedOrbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_req,
                                long clorb_iter, LINE_LIST *beamline, VMATRIX *M, RUN *run,
                                double dp, long start_from_recirc, double *starting_point, double change_fraction,
                                double change_fraction_multiplier, long multiplier_interval,
                                double *deviation, long n_turns) {
  long nElems, iterationsLeft, i, iterationsDone;
  double error = 0, ds, last_dp;
  /* double lastError = 0, last_ds; */
  double orbit0[6], orbit1[6];
  double startingPoint[6];
  double iterationFactor = 1.0;
#if DEBUG
  static FILE *fpdeb = NULL;
  if (fpdeb == NULL) {
    fpdeb = fopen("flco.sdds", "w");
    fprintf(fpdeb, "SDDS1\n");
    fprintf(fpdeb, "&column name=Iteration type=long &end\n");
    fprintf(fpdeb, "&column name=x type=double &end\n");
    fprintf(fpdeb, "&column name=xp type=double &end\n");
    fprintf(fpdeb, "&column name=y type=double &end\n");
    fprintf(fpdeb, "&column name=yp type=double &end\n");
    fprintf(fpdeb, "&column name=ds type=double &end\n");
    fprintf(fpdeb, "&column name=delta type=double &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  } else {
    fprintf(fpdeb, "\n");
  }
#endif

  nElems = beamline->n_elems;
  iterationsLeft = clorb_iter / 10 + 10;
  /* last_ds = last_dp = sqrt(DBL_MAX/10); */
  last_dp = sqrt(DBL_MAX / 10);
  iterationsDone = 0;
  while (iterationsDone < iterationsLeft) {
#ifdef DEBUG
    printf("running find_closed_orbit for dp=%le, starting_point=%le, %le, %le, %le, %le, %le\n", dp,
           starting_point ? starting_point[0] : -1,
           starting_point ? starting_point[1] : -1,
           starting_point ? starting_point[2] : -1,
           starting_point ? starting_point[3] : -1,
           starting_point ? starting_point[4] : -1,
           starting_point ? starting_point[5] : -1);
#endif
    if (!find_closed_orbit(clorb, clorb_acc, clorb_acc_req, clorb_iter, beamline, M, run, dp, start_from_recirc,
                           0,
                           iterationsDone == 0 ? starting_point : startingPoint,
                           change_fraction, change_fraction_multiplier, multiplier_interval, deviation, n_turns)) {
      iterationFactor /= 3;
#ifdef DEBUG
      printf("find_closed_orbit() failed for iteration=%ld, reducing dp iteration factor to %le\n", iterationsDone, iterationFactor);
#endif
      dp = last_dp;
      if (iterationFactor < 1e-3) {
#ifdef DEBUG
        printf("find_closed_orbit() failed, iteration factor too small now (%le).\n", iterationFactor);
#endif
        return 0;
      }
      continue;
    }
    ds = clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[4] - beamline->revolution_length;
#ifdef DEBUG
    printf("ds = %le\n", ds);
    fprintf(fpdeb, "%ld %21.15e %21.15e %21.15e %21.15e %21.15e %21.15e\n",
            iterationsDone,
            clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[0],
            clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[1],
            clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[2],
            clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[3],
            ds, dp);
    fflush(fpdeb);
#endif
    /* lastError = error; */
    error = fabs(last_dp - dp);
#if DEBUG
    printf("orbit error for iterationsDone=%ld, dp=%le is %le, ds=%le:\n", iterationsDone, dp, error, ds);
    for (i = 0; i < 6; i++)
      printf("%10.3e ", clorb[0].centroid[i]);
    printf("\n");
    for (i = 0; i < 6; i++)
      printf("%10.3e ", clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[i] - (i == 4 ? beamline->revolution_length : 0));
    printf("\n");
#endif
    if (error < clorb_acc && iterationsDone > 1) {
#if DEBUG
      printf("exiting dp iteration loop with error=%le (< %le)\n", error, clorb_acc);
#endif
      break;
    }
    /* last_ds = ds; */
    last_dp = dp;
    if (iterationsDone == 0) {
      for (i = 0; i < 6; i++)
        orbit0[i] = clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[i];
      orbit0[4] = ds;
      if (deviation)
        deviation[4] = ds;
      dp -= change_fraction * ds / M->R[4][5] / 10;
      memcpy(startingPoint, orbit0, 6 * sizeof(startingPoint[0]));
    } else {
      if (iterationsDone > 1) {
        for (i = 0; i < 6; i++)
          orbit0[i] = orbit1[i];
      }
      for (i = 0; i < 6; i++)
        orbit1[i] = clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[i];
      orbit1[4] = ds;
      if (orbit1[4] != orbit0[4]) {
        dp += -(orbit1[5] - orbit0[5]) / (orbit1[4] - orbit0[4]) * orbit1[4] * iterationFactor;
        /* 
          for (i=0; i<4; i++) 
          startingPoint[i] = orbit1[i] - (orbit1[i]-orbit0[i])/(orbit1[4]-orbit0[4])*orbit1[4];
          */
        memcpy(startingPoint, orbit1, 6 * sizeof(startingPoint[0]));
      } else {
        memcpy(startingPoint, orbit1, 6 * sizeof(startingPoint[0]));
      }
    }
    iterationsDone++;
  }
#if DEBUG
  printf("%ld iterations done for delta in fixed-length orbit computation\ndelta convergence error was %le\ndelta=%le, length error was %le\n",
         iterationsDone, last_dp - dp, dp, ds);
#endif
  if (iterationsDone < iterationsLeft || error < clorb_acc_req)
    return 1;
  printWarning("closed_orbit: fixed length orbit iteration didn't converge", NULL);
  printf("error is %le, dp = %le, %le\n", error, dp, last_dp);
  for (i = 0; i < 6; i++)
    printf("%10.3e ", clorb[0].centroid[i]);
  printf("\n");
  for (i = 0; i < 6; i++)
    printf("%10.3e ", clorb[nElems - (beamline->elem_recirc && start_from_recirc ? beamline->i_recirc : 0)].centroid[i] - (i == 4 ? beamline->revolution_length : 0));
  printf("\n");

  return 0;
}

/* buildSixDimJacobian: form the 6x6 one-turn Jacobian R[i][k]=dF_i/dX_k by finite-differencing the
 * RF-on tracked map about `point`, then set ImR=I-R and INV_ImR=(I-R)^-1.
 *
 * This is the fallback used when the analytic one-turn matrix M from full_matrix() lacks the rf
 * phase-focusing term R[5][4] -- i.e. when the rf cavities are currently represented by drift
 * matrices (matched-twiss default; see modify_rfca_matrices()/cavities_are_drifts_if_matched in
 * twiss.cc), which makes I-R singular in the longitudinal block. It is also used for the
 * update_matrix refresh, since it re-forms the Jacobian about the current trial point (and captures
 * nonlinearities), which the fixed reference-point matrix M cannot.
 * Returns 1 on success, 0 if the probe was lost, -1 if (I-R) cannot be inverted. */
static long buildSixDimJacobian(double *point, LINE_LIST *beamline, RUN *run, unsigned long newtonPass,
                                double **one_part, MATRIX *R, MATRIX *ImR, MATRIX *INV_ImR) {
  double Fbase[6], p;
  /* Per-coordinate step: larger for the path-length coordinate (index 4) whose absolute value is
   * ~revolution_length, to limit cancellation in F4(x+h)-F4(x). */
  static const double hstep[6] = {1e-6, 1e-6, 1e-6, 1e-6, 1e-5, 1e-6};
  long i, k;

  for (i = 0; i < 6; i++)
    one_part[0][i] = point[i];
  one_part[0][6] = 1;
  p = run->p_central;
  if (!do_tracking(NULL, one_part, 1, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   (TRAJECTORY *)NULL, run, 0, newtonPass, 1, 0, NULL, NULL, NULL, NULL, NULL))
    return 0;
  for (i = 0; i < 6; i++)
    Fbase[i] = one_part[0][i];
  for (k = 0; k < 6; k++) {
    for (i = 0; i < 6; i++)
      one_part[0][i] = point[i];
    one_part[0][k] += hstep[k];
    one_part[0][6] = 1;
    p = run->p_central;
    if (!do_tracking(NULL, one_part, 1, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                     (TRAJECTORY *)NULL, run, 0, newtonPass, 1, 0, NULL, NULL, NULL, NULL, NULL))
      return 0;
    for (i = 0; i < 6; i++)
      R->a[i][k] = (one_part[0][i] - Fbase[i]) / hstep[k];
  }
  for (i = 0; i < 6; i++)
    for (k = 0; k < 6; k++)
      ImR->a[i][k] = (i == k ? 1 : 0) - R->a[i][k];
  if (!m_invert(INV_ImR, ImR))
    return -1;
  return 1;
}

/* jacobianFromMatrix: form the quasi-Newton Jacobian directly from the analytic one-turn matrix M,
 * i.e. R=M->R, ImR=I-R, INV_ImR=(I-R)^-1. Used when M already carries the rf phase-focusing term
 * (M->R[5][4]!=0), which is the usual case for the standalone &closed_orbit command: the real rf
 * cavity matrices (rf_cavity_matrix() populates R[5][4]) are in place, so no tracking is needed to
 * build the Jacobian. Returns 1 on success, -1 if (I-R) cannot be inverted. */
static long jacobianFromMatrix(VMATRIX *M, MATRIX *R, MATRIX *ImR, MATRIX *INV_ImR) {
  long i, k;
  for (i = 0; i < 6; i++)
    for (k = 0; k < 6; k++) {
      R->a[i][k] = M->R[i][k];
      ImR->a[i][k] = (i == k ? 1 : 0) - M->R[i][k];
    }
  if (!m_invert(INV_ImR, ImR))
    return -1;
  return 1;
}

/* findSixDimClosedOrbit: full 6-D closed-orbit solver for fixed_length=2.
 *
 * Unlike find_closed_orbit (4-D, RF time-dependence OFF) and findFixedLengthClosedOrbit
 * (an outer secant on delta wrapping the 4-D finder), this solves the transverse orbit, the
 * momentum offset delta, and the RF timing/synchronous phase together with a single quasi-Newton
 * iteration over all six coordinates.
 *
 * The probe is tracked with RF time-dependence ON (TIME_DEPENDENCE_OFF is NOT set) so that the RF
 * energy kick depends on arrival time (coordinate 4). change_t is forced to 0 on every rf cavity for
 * the duration of the solve (saveAndZeroRfChangeT) so coord[4] is carried as the absolute path
 * length (nominally the circumference) instead of being reduced by whole rf periods each turn. The
 * residual at trial fixed point X is then
 *     r[i] = F[i] - X[i]                  for i = 0,1,2,3,5   (transverse + delta closure)
 *     r[4] = (F[4] - X[4]) - pathTarget   (rf-synchronized path-length constraint)
 * where F is one-turn tracking and pathTarget = h*beta0*c/f_rf is the path length for which one
 * revolution equals h rf periods (sixDimRfPathTarget); at the matched frequency this is the design
 * circumference. Because pathTarget responds to the rf frequency, an off-design frequency shifts the
 * closed orbit radially through a nonzero delta. The constant drops under differentiation, so the
 * Jacobian is the full 6x6 J = R - I, which is non-singular only because RF phase focusing makes
 * R[5][4] != 0 (in the 4-D/no-RF case the longitudinal block of I-R is singular).
 * Newton step: dX = (I-R)^-1 . r.
 *
 * A stable RF fiducial is established once before the Newton loop and held across all iterates
 * (pattern mirrors momentumAperture.c): save beamline->fiducial_flag, zero it, delete phase
 * references, do one FIRST_BEAM_IS_FIDUCIAL pass at the design reference, then run the Newton
 * iterations with FIDUCIAL_BEAM_SEEN+FIRST_BEAM_IS_FIDUCIAL so they reuse (not re-establish) the
 * fiducial, and restore beamline->fiducial_flag on exit.
 */
long findSixDimClosedOrbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_req, long clorb_iter,
                           LINE_LIST *beamline, VMATRIX *M, RUN *run, long start_from_recirc,
                           double *starting_point, double change_fraction, double change_fraction_multiplier,
                           long multiplier_interval, double *deviation, long n_turns) {
  static MATRIX *R, *ImR, *INV_ImR, *X, *r, *change;
  static double **one_part;
  static long initialized = 0;
  long i, n_iter, goodCount, jstat;
  long n_part;
  double p, error, last_error, reference_error;
  double point[6];
  unsigned long fiducial_flag_save;
  unsigned long fidPass, newtonPass;
  long *rfChangeTSave;
  long nRfChangeT;
  long useAnalyticM;
  double pathTarget;
  long rfHarmonic;
#if SDDS_MPI_IO
  long distributedBeam_orig = distributedBeam; /* run single-particle mode: all processors do the same */
  distributedBeam = 0;
#endif

  log_entry("findSixDimClosedOrbit");
  printWarning("Using findSixDimClosedOrbit", "findSixDimClosedOrbit is not fully tested.");
  
  if (!initialized) {
    m_alloc(&R, 6, 6);
    m_alloc(&ImR, 6, 6);
    m_alloc(&INV_ImR, 6, 6);
    m_alloc(&X, 6, 1);
    m_alloc(&r, 6, 1);
    m_alloc(&change, 6, 1);
    one_part = (double **)czarray_2d(sizeof(**one_part), 1, totalPropertiesPerParticle);
    initialized = 1;
  }

  /* Seed X from a caller-supplied starting point, else from zero (the ring closed orbit is near the
   * reference; the Newton iteration refines it). */
  for (i = 0; i < 6; i++)
    X->a[i][0] = starting_point ? starting_point[i] : 0;

  p = run->p_central;
  if (deviation)
    for (i = 0; i < 6; i++)
      deviation[i] = 0;

  /* Force change_t=0 on all rf cavities so coord[4] stays absolute path length (see helper); restored
   * at every exit below. */
  rfChangeTSave = saveAndZeroRfChangeT(beamline, &nRfChangeT);

  /* Path length one turn must have to synchronize to the rf (h rf periods per revolution). Replaces
   * the geometric revolution_length so the orbit tracks the rf frequency. See sixDimRfPathTarget. */
  pathTarget = sixDimRfPathTarget(beamline, run->p_central, &rfHarmonic);
  if (verbosity > 1)
    printf("closed_orbit: fixed_length=2 rf harmonic h=%ld, target path length=%.15g m (design circumference=%.15g m)\n",
           rfHarmonic, pathTarget, beamline->revolution_length);

  /* --- Establish a stable RF fiducial and hold it across the Newton loop (see momentumAperture.c). --- */
  fiducial_flag_save = beamline->fiducial_flag;
  beamline->fiducial_flag = 0;
  fidPass = CLOSED_ORBIT_TRACKING + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT +
            (start_from_recirc ? BEGIN_AT_RECIRC : 0);
  newtonPass = CLOSED_ORBIT_TRACKING + TEST_PARTICLES + FIDUCIAL_BEAM_SEEN + FIRST_BEAM_IS_FIDUCIAL +
               SILENT_RUNNING + INHIBIT_FILE_OUTPUT + (start_from_recirc ? BEGIN_AT_RECIRC : 0);

  delete_phase_references();
  reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
  for (i = 0; i < 6; i++)
    one_part[0][i] = 0;
  one_part[0][6] = 1;
  p = run->p_central;
  if (!do_tracking(NULL, one_part, 1, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   (TRAJECTORY *)NULL, run, 0, fidPass, 1, 0, NULL, NULL, NULL, NULL, NULL)) {
    printWarning("closed_orbit: fiducial particle lost during fixed_length=2 6-D closed orbit setup", NULL);
    restoreRfChangeT(beamline, rfChangeTSave, nRfChangeT);
    beamline->fiducial_flag = fiducial_flag_save;
#if SDDS_MPI_IO
    distributedBeam = distributedBeam_orig;
#endif
    log_exit("findSixDimClosedOrbit");
    return 0;
  }

  /* --- Build the quasi-Newton Jacobian. --- */
  /* Prefer the analytic one-turn matrix M when it carries the rf phase-focusing term (M->R[5][4]!=0),
   * which is the usual case here: full_matrix() sees the real rf cavity matrices (they are only
   * swapped for drifts during matched-twiss computation, then restored -- modify_rfca_matrices/
   * reset_rfca_matrices in twiss.cc). This matches the finite-differenced Jacobian to working
   * precision while avoiding the extra one-turn tracks. When M lacks the term (cavities currently
   * treated as drifts) recover the longitudinal coupling by differencing the RF-on tracked map. */
  for (i = 0; i < 6; i++)
    point[i] = X->a[i][0];
  useAnalyticM = (M && M->R && (M->R[5][4] != 0.0));
  if (useAnalyticM) {
    jstat = jacobianFromMatrix(M, R, ImR, INV_ImR);
    if (jstat == -1) {
      /* analytic (I-R) singular despite nonzero R[5][4]; recover about the current point via tracking */
      jstat = buildSixDimJacobian(point, beamline, run, newtonPass, one_part, R, ImR, INV_ImR);
      useAnalyticM = 0;
    }
  } else
    jstat = buildSixDimJacobian(point, beamline, run, newtonPass, one_part, R, ImR, INV_ImR);
  if (jstat == 0) {
    printWarning("closed_orbit: probe lost while forming the fixed_length=2 6-D Jacobian", NULL);
    restoreRfChangeT(beamline, rfChangeTSave, nRfChangeT);
    beamline->fiducial_flag = fiducial_flag_save;
#if SDDS_MPI_IO
    distributedBeam = distributedBeam_orig;
#endif
    log_exit("findSixDimClosedOrbit");
    return 0;
  }
  if (jstat == -1) {
    printf("error: unable to invert (I-R) for the 6-D closed orbit (fixed_length=2).\n");
    printf("This mode requires an rf cavity that produces longitudinal focusing (R[5][4]!=0).\n");
    printf("The numerically-differenced one-turn map R is:\n");
    for (i = 0; i < 6; i++) {
      long jj;
      printf("R[%ld]: ", i + 1);
      for (jj = 0; jj < 6; jj++)
        printf("%14.6e ", R->a[i][jj]);
      fputc('\n', stdout);
    }
    fflush(stdout);
    restoreRfChangeT(beamline, rfChangeTSave, nRfChangeT);
    beamline->fiducial_flag = fiducial_flag_save;
#if SDDS_MPI_IO
    distributedBeam = distributedBeam_orig;
#endif
    bombElegant("cannot invert (I-R) for fixed_length=2 6-D closed orbit; ensure an rf cavity providing longitudinal focusing is configured", NULL);
  }

  /* --- 6-D quasi-Newton iteration. --- */
  n_iter = 0;
  reference_error = error = DBL_MAX / 4;
  goodCount = 0;
  do {
    n_part = 1;
    for (i = 0; i < 6; i++)
      one_part[0][i] = X->a[i][0];
    one_part[0][6] = 1;
    p = run->p_central;
    if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                     clorb + 1, run, 0, newtonPass, 1, 0, NULL, NULL, NULL, NULL, NULL)) {
      printWarning("closed_orbit: particle lost during fixed_length=2 6-D closed orbit iteration", NULL);
      n_iter = clorb_iter;
      break;
    }
    /* 6-D periodicity residual. Transverse and delta coordinates must repeat (r[i]=F[i]-X[i]); the
     * timing coordinate (index 4) is the absolute path length (change_t forced to 0 above), so the
     * fixed-length constraint is r[4]=(F[4]-X[4])-pathTarget, where pathTarget = h*beta0*c/f_rf is the
     * path length that synchronizes the beam to the rf (h rf periods per revolution, sixDimRfPathTarget).
     * This makes the orbit respond to the rf frequency; at the matched frequency pathTarget equals the
     * design circumference. The pathTarget constant drops out of the Jacobian. */
    for (i = 0; i < 6; i++)
      r->a[i][0] = one_part[0][i] - X->a[i][0];
    r->a[4][0] -= pathTarget;
    if (deviation)
      for (i = 0; i < 6; i++)
        deviation[i] = r->a[i][0];
    last_error = error;
    error = 0;
    for (i = 0; i < 6; i++)
      error += sqr(r->a[i][0]);
    error = sqrt(error);
    if (error < reference_error)
      reference_error = error;
    if (error < clorb_acc)
      break;
    if (error > reference_error) {
      if (error < clorb_acc_req)
        /* Already below the accuracy requirement and no longer improving: the residual is
         * fluctuating at the numerical floor (the r[4] path-length term cancels two
         * ~circumference-scale quantities). Accept the orbit rather than backing off the step
         * fraction into a spurious divergence report. */
        break;
      reference_error = error;
      change_fraction = change_fraction / fraction_divisor;
      goodCount = -10;
      if (change_fraction < 0.01) {
        char buffer[16384];
        snprintf(buffer, 16384, "accuracy requirement: %e, previous error: %e, current error: %e",
                 clorb_acc, last_error, error);
        printWarning("closed_orbit: 6-D (fixed_length=2) closed orbit diverging, iteration stopped", buffer);
        n_iter = clorb_iter;
        break;
      }
    } else {
      goodCount++;
      if (goodCount > multiplier_interval) {
        change_fraction *= change_fraction_multiplier;
        if (change_fraction > 1)
          change_fraction = 1;
        goodCount = 0;
      }
    }
    m_mult(change, INV_ImR, r);
    if (change_fraction != 1)
      m_scmul(change, change, change_fraction);
    m_add(X, X, change);

    if (update_matrix) {
      /* Re-form the Jacobian (I-R) numerically at the current trial point. */
      for (i = 0; i < 6; i++)
        point[i] = X->a[i][0];
      jstat = buildSixDimJacobian(point, beamline, run, newtonPass, one_part, R, ImR, INV_ImR);
      if (jstat <= 0) {
        printWarning("closed_orbit: failed to re-form the fixed_length=2 6-D Jacobian, iteration stopped", NULL);
        n_iter = clorb_iter;
        break;
      }
    }
  } while (++n_iter < clorb_iter);

  for (i = 0; i < 6; i++)
    clorb[0].centroid[i] = X->a[i][0];

  restoreRfChangeT(beamline, rfChangeTSave, nRfChangeT);
  beamline->fiducial_flag = fiducial_flag_save;
#if SDDS_MPI_IO
  distributedBeam = distributedBeam_orig;
#endif
  log_exit("findSixDimClosedOrbit");

  if (n_iter >= clorb_iter && error > clorb_acc_req) {
    printf("error: 6-D closed orbit did not converge to better than %e after %ld iterations (requirement is %e)\n",
           error, n_iter, clorb_acc_req);
    fflush(stdout);
    return 0;
  }
  return 1;
}

void zero_closed_orbit(TRAJECTORY *clorb, long n) {
  long i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < 6; j++)
      clorb[i].centroid[j] = 0;
}
