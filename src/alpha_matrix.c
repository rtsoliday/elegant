/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: alpha_matrix.c
 * contents: data and routines for calculating alpha magnet matrices.
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

extern double full_alpha_matrix_elems[506];
extern double first_half_alpha_matrix_elems[506];
extern double second_half_alpha_matrix_elems[506];

void initialize_precalculated_matrix(VMATRIX *matrix, double *gradient,
                                     double *gamma, double *matrix_element);

VMATRIX *alpha_magnet_matrix(
  double gradient, /* G/cm */
  double gamma,    /* central gamma */
  long maximum_order,
  long part) {
  long i, j, k, l, m, n, o, p;
  double sum1, sum2, uval1, uval2, uval3;
  double scale;            /* scaling factor */
  static double U[6][6];   /* scaling matrix */
  static double U_1[6][6]; /* its inverse */
  static long matrices_initialized = 0;
  static VMATRIX alpha_0;            /* pre-calculated matrix--full magnet */
  static VMATRIX alpha_1;            /* first half of alpha magnet */
  static VMATRIX alpha_2;            /* second half of alpha magnet */
  VMATRIX *alpha_pc;                 /* one of the above */
  static double gamma_0, gradient_0; /* gamma and gradient for same */
  static double gamma_1, gradient_1;
  static double gamma_2, gradient_2;
  double gamma_pc, gradient_pc;
  double *C_, **R_, ***T_, ****Q_; /* for individual matrices of a pre-calculated matrix */
  VMATRIX *M;                      /* scaled matrix */
  double *C, **R, ***T, ****Q;     /* individual matrices of same */

#ifdef DEBUG
  report_stats(stdout, "alpha_magnet_matrix called");
#endif

  log_entry("alpha_magnet_matrix");

  /* initialize the pre-calculated matrix */
  if (!matrices_initialized) {
    initialize_precalculated_matrix(&alpha_0, &gradient_0, &gamma_0,
                                    full_alpha_matrix_elems);

    initialize_precalculated_matrix(&alpha_1, &gradient_1, &gamma_1,
                                    first_half_alpha_matrix_elems);

    initialize_precalculated_matrix(&alpha_2, &gradient_2, &gamma_2,
                                    second_half_alpha_matrix_elems);
#ifdef DEBUG
    report_stats(stdout, "Precalculated alpha-magnet matrices have been initialized.\n");
#endif
    matrices_initialized = 1;
  }

  switch (part) {
  case 1: /* first half of alpha magnet */
    gamma_pc = gamma_1;
    gradient_pc = gradient_1;
    alpha_pc = &alpha_1;
    break;
  case 2: /* second half of magnet */
    gamma_pc = gamma_2;
    gradient_pc = gradient_2;
    alpha_pc = &alpha_2;
    break;
  case 3: /* scraper */
    return drift_matrix(0.0, 1);
    break;
  case 0:
  default: /* full alpha magnet */
    gamma_pc = gamma_0;
    gradient_pc = gradient_0;
    alpha_pc = &alpha_0;
    break;
  }

  /* compute the scaling matrix U */
  scale = sqrt((sqrt(sqr(gamma) - 1) / gradient) /
               (sqrt(sqr(gamma_pc) - 1) / gradient_pc));

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      U[i][j] = U_1[i][j] = 0;
  U[0][0] = U[2][2] = U[4][4] = scale;
  U_1[0][0] = U_1[2][2] = U_1[4][4] = 1. / scale;
  U[1][1] = U[3][3] = U[5][5] = 1;
  U_1[1][1] = U_1[3][3] = U_1[5][5] = 1;

#ifdef DEBUG
  report_stats(stdout, "preparing matrix memory");
#endif

  maximum_order = MIN(3, maximum_order);
  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = maximum_order);

#ifdef DEBUG
  report_stats(stdout, "matrix memory prepared");
#endif

  set_matrix_pointers(&C, &R, &T, &Q, M);
  set_matrix_pointers(&C_, &R_, &T_, &Q_, alpha_pc);

  /* Scale C: C[n] = U[n][i]*C_[i] */
  for (n = 0; n < 6; n++) {
    for (i = sum1 = 0; i < 6; i++)
      sum1 += U[n][i] * C_[i];
    C[n] = sum1;
  }

#ifdef DEBUG
  report_stats(stdout, "C matrix scaled");
#endif

  /* Scale R:  R[n][m] = U[n][i]*R_[i][j]*U_1[j][m] */
  for (n = 0; n < 6; n++) {
    for (m = 0; m < 6; m++) {
      for (i = sum1 = 0; i < 6; i++) {
        if ((uval1 = U[n][i])) {
          for (j = sum2 = 0; j < 6; j++)
            sum2 += R_[i][j] * U_1[j][m];
          sum1 += sum2 * uval1;
        }
      }
      R[n][m] = sum1;
    }
  }

#ifdef DEBUG
  report_stats(stdout, "R matrix scaled");
#endif

  if (M->order >= 2) {
    /* Scale T:  T[n][m][o] = U[n][i]*T_[i][j][k]*U_1[j][m]*U_1[k][o] */
    for (n = 0; n < 6; n++) {
      for (m = 0; m < 6; m++) {
        for (o = 0; o <= m; o++) {
          for (i = sum1 = 0; i < 6; i++) {
            if ((uval1 = U[n][i])) {
              for (j = sum2 = 0; j < 6; j++) {
                if ((uval2 = uval1 * U_1[j][m])) {
                  for (k = 0; k <= j; k++)
                    sum2 += T_[i][j][k] * U_1[k][o];
                  sum1 += uval2 * sum2;
                }
              }
            }
          }
          T[n][m][o] = sum1;
        }
      }
    }
  }

#ifdef DEBUG
  report_stats(stdout, "T matrix scaled");
#endif

  if (M->order >= 3) {
    /* Scale Q:  Q[n][m][o][p] = U[n][i]*Q_[i][j][k][l]*U_1[j][m]*U_1[k][o]*U_1[l][p] */
    for (n = 0; n < 6; n++) {
      for (m = 0; m < 6; m++) {
        for (o = 0; o <= m; o++) {
          for (p = 0; p <= o; p++) {
            for (i = sum1 = 0; i < 6; i++) {
              if ((uval1 = U[n][i])) {
                for (j = 0; j < 6; j++) {
                  if ((uval2 = uval1 * U_1[j][m])) {
                    for (k = 0; k <= j; k++) {
                      if ((uval3 = uval2 * U_1[k][o])) {
                        for (l = sum2 = 0; l <= k; l++)
                          sum2 += Q_[i][j][k][l] * U_1[l][p];
                        sum1 += uval3 * sum2;
                      }
                    }
                  }
                }
              }
            }
            Q[n][m][o][p] = sum1;
          }
        }
      }
    }
  }

#ifdef DEBUG
  report_stats(stdout, "Q matrix scaled");
#endif

  log_exit("alpha_magnet_matrix");

  return (M);
}

void initialize_precalculated_matrix(
  VMATRIX *matrix,
  double *gradient,
  double *gamma,
  double *matrix_element) {
  double *C, **R, ***T, ****Q;
  long i, j, k, l;

  log_entry("initialize_precalculated_matrix");

  initialize_matrices(matrix, matrix->order = 3);
  C = matrix->C;
  R = matrix->R;
  T = matrix->T;
  Q = matrix->Q;

  *gradient = *matrix_element++;
  *gamma = *matrix_element++;

  for (i = 0; i < 6; i++)
    C[i] = *matrix_element++;

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      R[i][j] = *matrix_element++;

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      for (k = 0; k <= j; k++)
        T[i][j][k] = *matrix_element++;

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      for (k = 0; k <= j; k++)
        for (l = 0; l <= k; l++)
          Q[i][j][k][l] = *matrix_element++;

  log_exit("initialize_precalculated_matrix");
}

long alpha_magnet_tracking(
  double **particle, VMATRIX *M, ALPH *alpha, int64_t n_part,
  double **accepted, double P_central, double z) {
  float xmax, xl = 0.0, xu = 0.0;
  long do_xl, do_xu;
  int64_t ip, itop;
  double *coord;

  log_entry("alpha_magnet_tracking");

  if (alpha->part == 0) {
    track_particles(particle, M, particle, n_part);
    log_exit("alpha_magnet_tracking");
    return (n_part);
  }

  if (alpha->part == 1)
    track_particles(particle, M, particle, n_part);

  if (alpha->xPuck > 0) {
    /* do filtering of particles with a "puck" scraper of length widthPuck 
       * ending at position xPuck */
    if ((xl = alpha->xPuck - alpha->widthPuck) < 0)
      xl = 0;
    if ((xu = alpha->xPuck) < 0)
      xu = 0;
    if (xu > xl) {
      xmax = ALPHA_CONST * sqrt(P_central / alpha->gradient);
      xl = xmax - xl;
      xu = xmax - xu;
      SWAP_DOUBLE(xl, xu);
      itop = n_part - 1;
      for (ip = 0; ip < n_part; ip++) {
        coord = particle[ip];
        if (coord[0] >= xl && coord[0] <= xu) {
          if (accepted)
            swapParticles(accepted[ip], accepted[itop]);
          /* record position and momentum of lost particle */
          swapParticles(particle[ip], particle[itop]);
          particle[itop][4] = z;
          particle[itop][5] = P_central * (1 + particle[itop][5]);
          itop--;
          ip--;
          n_part--;
        }
      }
    }
  }
  if (alpha->xs1 || alpha->xs2 || alpha->dp1 != -1 || alpha->dp2 != 1) {
    /* do filtering of particles with slits from the high and low momentum sides */
    do_xu = do_xl = 1;
    xmax = ALPHA_CONST * sqrt(P_central / alpha->gradient);
    if (alpha->dp1 != -1 || alpha->dp2 != 1) {
      /* If dp1 and/or dp2 are specified, then filter out all particles 
             * with dp<dp1 or dp>dp2.  Recall that coordinate system is such
             * that greater Xmax means smaller x.
             */
      if (alpha->dp1 == -1)
        do_xu = 0;
      else
        xu = xmax -
             ALPHA_CONST * sqrt((1 + alpha->dp1) * P_central / alpha->gradient);
      if (alpha->dp2 == 1)
        do_xl = 0;
      else
        xl = xmax -
             ALPHA_CONST * sqrt((1 + alpha->dp2) * P_central / alpha->gradient);
    } else {
      /* If dp1 and/or dp2 are specified, then filter out all particles 
             * with Xmax<xs1 or Xmax>xs2.  Recall that coordinate system is such
             * that greater Xmax means smaller x.
             */
      if (alpha->xs2)
        xl = xmax - alpha->xs2;
      else
        do_xl = 0;
      if (alpha->xs1)
        xu = xmax - alpha->xs1;
      else
        do_xu = 0;
    }
    itop = n_part - 1;
    for (ip = 0; ip < n_part; ip++) {
      coord = particle[ip];
      if ((do_xl && coord[0] <= xl) || (do_xu && coord[0] >= xu)) {
        if (accepted)
          swapParticles(accepted[ip], accepted[itop]);
        /* record position and momentum of lost particle */
        swapParticles(particle[ip], particle[itop]);
        particle[itop][4] = z;
        particle[itop][5] = P_central * (1 + particle[itop][5]);
        itop--;
        ip--;
        n_part--;
      }
    }
  }

  if (alpha->part == 2)
    track_particles(particle, M, particle, n_part);

  log_exit("alpha_magnet_tracking");
  return (n_part);
}
