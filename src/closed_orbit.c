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
  if (fixed_length && checkChangeT(beamline))
    bombElegant("change_t is nonzero on one or more RF cavities. This is incompatible with fixed-length orbit computations.", NULL);

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

  if (fixed_length)
    return findFixedLengthClosedOrbit(clorb, clorb_acc, clorb_acc_requirement, clorb_iter, beamline, M, run, dp,
                                      start_from_recirc, starting_point, change_fraction, fraction_multiplier,
                                      multiplier_interval, deviation, n_turns);
  
#if SDDS_MPI_IO
  long notSinglePart_orig = notSinglePart; /* We need save the original value to switch it back */
  notSinglePart = 0;                       /* run as single particle mode, i.e., all processors will do the same thing */
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
          printf("reduced iteration fraction to %e\n", change_fraction);
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
          notSinglePart = notSinglePart_orig;
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
  notSinglePart = notSinglePart_orig; /* Switch back to original parallel tracking mode */
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

void zero_closed_orbit(TRAJECTORY *clorb, long n) {
  long i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < 6; j++)
      clorb[i].centroid[j] = 0;
}
