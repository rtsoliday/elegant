/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* routine: motion()
 * purpose: integrate equations of motion for a particle in a
 *          static or time-dependent field.  This module is appropriate
 *          for elements with no coordinate system curvature.  lorentzX.c
 *          is better if there is coordinate system curvature.
 *
 * The equations of motion are
 *
 *      '         _                                                  _
 *     P  =  -eE (X, tau)/(m.c.omega) - e/(m.gamma.omega).eps   P B (X, tau)
 *      i       i                                            ijk j k
 *
 *      '
 *     X  =  P /gamma
 *      i     i
 *
 *       _   ____        _     _
 * where P = beta.gamma, X = k.x, and derivatives are with respect to
 * tau=omega.time.   E has units of V/m and B has units of T.
 *
 * For resonant fields, omega is the angular frequency and k=omega/c.
 * For ramped fields, omega is 1/ramp_time and k=omega/c.
 * For static fields, I arbitrarily take omega=2*PI c/L, where L is the total length.
 *
 * Michael Borland, 1989
 */
#if USE_MPI
#  include "mpi.h" /* Defines the interface to MPI allowing the use of all MPI functions. */
#  if USE_MPE
#    include "mpe.h" /* Defines the MPE library */
#  endif
#endif
#include <complex>
#if defined(__APPLE__)
#  include <cmath>
#  define isinf(x) std::isinf(x)
#  define isnan(x) std::isnan(x)
#endif
#include "mdb.h"
#include "match_string.h"
#include "track.h"
#include "matlib.h"

void (*set_up_derivatives(void *field, long field_type, double *kscale,
                          double z_start, double *Z_end, double *tau_start, double **part, int64_t n_part,
                          double Po, double *X_limit, double *Y_limit, double *accuracy,
                          long *n_steps))(double *, double *, double);
void (*set_up_stochastic_effects(long field_type))(double *, double, double);
void derivatives_tmcf_mode(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau);
void derivatives_rftmEz0(
                         double *qp, /* derivatives w.r.t. tau */
                         double *q,  /* X,Y,Z,Px,Py,Pz */
                         double tau);
void derivatives_ce_plates(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau);
void derivatives_tw_plates(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau);
void derivatives_tw_linac(
                          double *qp, /* derivatives w.r.t. tau */
                          double *q,  /* X,Y,Z,Px,Py,Pz */
                          double tau);
void derivatives_twmta(
                       double *qp, /* derivatives w.r.t. tau */
                       double *q,  /* X,Y,Z,Px,Py,Pz */
                       double tau);
void derivatives_mapSolenoid(
                             double *qp, /* derivatives w.r.t. tau */
                             double *q,  /* X,Y,Z,Px,Py,Pz */
                             double tau);
void derivatives_laserModulator(double *qp, double *q, double tau);
void stochastic_laserModulator(double *q, double tau, double h);

void computeFields_rftmEz0(double *E, double *BOverGamma, double *BOverGammaSol,
                           double q0, double q1, double q2, double tau, double gamma);
void makeRftmEz0FieldTestFile(RFTMEZ0 *rftmEz0);

void setupRftmEz0FromFile(RFTMEZ0 *rftmEz0, double frequency, double length);
void setupRftmEz0SolenoidFromFile(RFTMEZ0 *rftmEz0, double length, double k);
void setupMapSolenoidFromFile(MAP_SOLENOID *mapSol, double length);
double exit_function(double *qp, double *q, double phase);
double *select_fiducial(double **part, int64_t n_part, char *mode);
void select_integrator(char *desired_method);
void input_impulse_tw_linac(double *P, double *q);
void output_impulse_tw_linac(double *P, double *q);
void input_impulse_twmta(double *P, double *q);
void output_impulse_twmta(double *P, double *q);

static void (*derivatives)(double *xqp, double *xq, double xt);
static void (*stochasticEffects)(double *xq, double xt, double h);
static void (*input_impulse)(double *P, double *q);
static void (*output_impulse)(double *P, double *q);

static long integratorCode = -1;
#define RUNGE_KUTTA 0
#define BULIRSCH_STOER 1
#define NA_RUNGE_KUTTA 2
#define MODIFIED_MIDPOINT 3
#define N_METHODS 4
static char *method[N_METHODS] = {
  (char *)"runge-kutta", (char *)"bulirsch-stoer", (char *)"non-adaptive runge-kutta", (char *)"modified midpoint"};

long analyzeSpacing(double *z, long nz, double *dzReturn, FILE *fpError);

#define OUT_OF_APERTURE (-1)

/* end of element, in scaled coordinates */
double Z_end;

/* offsets to be added to particle position to get coordinates relative
 * to element transverse center.
 */
double X_offset, Y_offset;

/* position of centers of X and Y apertures relative to straight-ahead
 * centerline
 */
double X_aperture_center, Y_aperture_center;

/* maximum coordinates relative to aperture center
 * that a particle can have without being considered lost
 */
double X_limit, Y_limit;

/* coordinates reached by particle, relative to
 * aperture center of element, when lost.
 */
double X_out, Y_out, Z_out;
long limit_hit, radial_limit, derivCalls = 0;

/* flag to indicate that element changes the central momentum */
static long change_p0;

/* rotation matrices for tilt (roll), yaw, pitch */
static MATRIX *partRot, *fieldRot;
static MATRIX *partRot1, *fieldRot1;
/* geometrical center of the element used for rotation */
double X_center, Y_center, Z_center;
double X_center1, Y_center1, Z_center1;

#if defined(DEBUG)
static long starting_integration;
static double initial_phase = 0, final_phase = 0;
#endif

static double xMaxSeen = -1e300, xMinSeen = 1e300;
static long xMotionCenterVar = -1;

static double radCoef = 0;
static double isrCoef = 0;
static double P_central, gamma_central;

#define LSRMDLTR_IDEAL 0
#define LSRMDLTR_EXACT 1
#define LSRMDLTR_LEADING 2
#define N_LSRMDLTR_FIELD_EXPANSIONS 3
static char *lsrMdltrFieldExpansion[N_LSRMDLTR_FIELD_EXPANSIONS] = {
  (char *)"ideal", (char *)"exact", (char *)"leading terms"};
double HermitePolynomial(double x, long n);
double HermitePolynomialDeriv(double x, long n);
double HermitePolynomial2ndDeriv(double x, long n);

long motion(
            double **part,
            int64_t n_part,
            void *field,
            long field_type,
            double *pCentral,
            double *dgamma,
            double *dP,
            double **accepted,
            double z_start) {
  double tau, tau_limit;
  double tau_start;
  long n_eq = 6;
  static double q[6], dqdt[6]; /* X,Y,Z,Px,Py,Pz */
  static long misses[6], accmode[6];
  static double accuracy[6], tiny[6];
  double tolerance, hmax, hrec;
  double *coord;
  int64_t i_part, i_top;
  long rk_return = 0;
  double kscale, *P, Po, gamma;
  static double gamma0, P0[3];
  long n_steps;
  // double tol_factor;
  double end_factor;
  double X = 0.0, Y = 0.0;

#if !USE_MPI
  if (!n_part)
    return (0);
#endif
  change_p0 = 0;
  P_central = *pCentral;
  gamma_central = sqrt(sqr(P_central) + 1);

  radCoef = sqr(particleCharge) * pow3(P_central) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3.0)) * pow5(P_central) * 137.0359895);

  log_entry((char *)"motion");

  X_aperture_center = Y_aperture_center = 0;
  X_limit = Y_limit = 0;
  X_offset = Y_offset = 0;
  radial_limit = 0;
  input_impulse = output_impulse = NULL;

  derivatives = set_up_derivatives(field, field_type, &kscale, z_start, &Z_end,
                                   &tau_start, part, n_part, P_central,
                                   &X_limit, &Y_limit, &tolerance, &n_steps);
  stochasticEffects = set_up_stochastic_effects(field_type);

  if (n_steps < 2)
    bombElegant((char *)"too few steps for integration", NULL);

  P = q + 3;
  i_top = n_part - 1;
  *dgamma = dP[0] = dP[1] = dP[2] = 0;

  fill_long_array(accmode, 6, 3);
  fill_double_array(tiny, 6, 1e-16);
  hrec = 0;
  xMaxSeen = -1e300;
  xMinSeen = 1e300;
  if (xMotionCenterVar < 0)
    xMotionCenterVar = rpn_create_mem((char *)"xMotionCenter", 0);

  for (i_part = 0; i_part <= i_top; i_part++) {
    // tol_factor = 1;
    end_factor = 1.5;
    coord = part[i_part];
    q[0] = coord[0] * kscale;
    q[1] = coord[2] * kscale;
    if ((X_limit && fabs(q[0] - X_aperture_center) > X_limit) || isnan(q[0]) || isinf(q[0]) ||
        (Y_limit && fabs(q[1] - Y_aperture_center) > Y_limit) || isnan(q[1]) || isinf(q[1])) {
      swapParticles(part[i_part], part[i_top]);
      part[i_top][4] = z_start; /* record position of loss */
      part[i_top][5] = P_central * (1 + part[i_top][5]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      i_top--;
      i_part--;
    } else {
      fill_double_array(accuracy, 3, tolerance * kscale);
      fill_double_array(accuracy + 3, 3, tolerance);
      fill_long_array(misses, 6, 0);
      q[2] = 0.0;
      Po = P_central * (1 + coord[5]);
      gamma0 = gamma = sqrt(1 + sqr(Po));
      P0[2] = P[2] = Po / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      P0[0] = P[0] = coord[1] * P[2];
      P0[1] = P[1] = coord[3] * P[2];
      X_out = Y_out = Z_out = 0;
      tau_limit = end_factor * Z_end * gamma / P[2] +
        (tau = tau_start + coord[4] * kscale * gamma / Po);
      hmax = (tau_limit - tau) / n_steps;
      if (integratorCode == NA_RUNGE_KUTTA || integratorCode == MODIFIED_MIDPOINT)
        hrec = (tau_limit - tau) / n_steps;
      if (hrec <= 0)
        hrec = hmax;
      limit_hit = 0;
#if defined(DEBUG)
      starting_integration = 1;
      initial_phase = final_phase = 0;
#endif
      if (input_impulse)
        input_impulse(P, q);
      derivCalls = 0;
      if (integratorCode == NA_RUNGE_KUTTA)
        rk_return = rk_odeint3_na(q, derivatives, n_eq, accuracy, accmode, tiny, misses,
                                  &tau, tau_limit, sqr(tolerance) * (tau_limit - tau), hrec, hmax, &hrec, exit_function,
                                  sqr(tolerance) * kscale, stochasticEffects);
      else if (integratorCode == RUNGE_KUTTA)
        rk_return = rk_odeint3(q, derivatives, n_eq, accuracy, accmode, tiny, misses,
                               &tau, tau_limit, sqr(tolerance) * (tau_limit - tau), hrec, hmax, &hrec, exit_function,
                               sqr(tolerance) * kscale);
      else if (integratorCode == BULIRSCH_STOER)
        rk_return = bs_odeint3(q, derivatives, n_eq, accuracy, accmode, tiny, misses,
                               &tau, tau_limit, sqr(tolerance) * (tau_limit - tau), hrec, hmax, &hrec, exit_function,
                               sqr(tolerance) * kscale);
      else if (integratorCode == MODIFIED_MIDPOINT)
        rk_return = mmid_odeint3_na(q, derivatives, n_eq, accuracy, accmode, tiny, misses,
                                    &tau, tau_limit, sqr(tolerance) * (tau_limit - tau), hrec, hmax, &hrec, exit_function,
                                    sqr(tolerance) * kscale);

      switch (rk_return) {
      case DIFFEQ_ZERO_STEPSIZE:
      case DIFFEQ_CANT_TAKE_STEP:
      case DIFFEQ_OUTSIDE_INTERVAL:
      case DIFFEQ_XI_GT_XF:
      case DIFFEQ_EXIT_COND_FAILED:
        printf((char *)"Integration failure: %s\n", diffeq_result_description(rk_return));
        fflush(stdout);
        /*                    for (i=0; i<6; i++)
                              printf((char*)"%11.4e  ", coord[i]);
                              fflush(stdout);
                              printf((char*)"\ngamma = %.4e,  P=(%.4e, %.4e, %.4e)\n",
                              gamma0, P0[0], P0[1], P0[2]);
                              fflush(stdout);
                              printf((char*)"tolerance = %e     end_factor = %e\n",
                              tolerance, end_factor);
                              fflush(stdout);
        */
        swapParticles(part[i_part], part[i_top]);
        part[i_top][4] = z_start;
        part[i_top][5] = sqrt(sqr(P_central * (1 + part[i_top][5])) + 1);
        if (accepted)
          swapParticles(accepted[i_part], accepted[i_top]);
        i_top--;
        i_part--;
        break;
      default:
        (*derivatives)(dqdt, q, tau);
        Po = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]));
        gamma = sqrt(1 + sqr(Po));
        *dgamma += gamma - gamma0;
        dP[0] += P[0] - P0[0];
        dP[1] += P[1] - P0[1];
        dP[2] += P[2] - P0[2];
#if defined(DEBUG)
        printf((char *)"initial, final phase: %21.15e, %21.15e\n", initial_phase, final_phase);
        fflush(stdout);
        printf((char *)"deriv calls: %ld\n", derivCalls);
        fflush(stdout);
        printf((char *)"Exit: tau=%le, ef=%le\n", tau, exit_function(dqdt, q, tau));
        fflush(stdout);
        printf((char *)"Pout = %21.15e  Zout = %21.15le\n",
               Po, q[2] / kscale);
        fflush(stdout);
#endif
        if (!limit_hit) {
          if (isnan(q[0]) || isinf(q[0]) || isnan(q[1]) || isinf(q[1])) {
            X_out = Y_out = DBL_MAX;
            Z_out = q[2];
            limit_hit = 1;
          } else if (radial_limit) {
            if (X_limit) {
              X = fabs(q[0] - X_aperture_center);
              Y = fabs(q[1] - Y_aperture_center);
              if (sqr(X_limit) < (sqr(X) + sqr(Y))) {
                X_out = X;
                Y_out = Y;
                Z_out = q[2];
                limit_hit = 1;
              }
            }
          } else {
            if ((X_limit && (X = fabs(q[0] - X_aperture_center)) > X_limit) ||
                (Y_limit && (Y = fabs(q[1] - Y_aperture_center)) > Y_limit)) {
              X_out = X;
              Y_out = Y;
              Z_out = q[2];
              limit_hit = 1;
            }
          }
        }
        if (limit_hit || P[2] <= 0) {
          swapParticles(part[i_part], part[i_top]);
          /* record position of loss */
          part[i_top][0] = (X_out + X_aperture_center) / kscale;
          part[i_top][2] = (Y_out + Y_aperture_center) / kscale;
          part[i_top][4] = Z_out / kscale + z_start;
          part[i_top][5] = Po;
          if (accepted)
            swapParticles(accepted[i_part], accepted[i_top]);
          i_top--;
          i_part--;
          continue;
        }
        if (output_impulse)
          output_impulse(P, q);
        coord[0] = q[0] / kscale;
        coord[2] = q[1] / kscale;
        coord[1] = P[0] / P[2];
        coord[3] = P[1] / P[2];
        coord[5] = (Po - P_central) / P_central;
        coord[4] = (tau - tau_start) * Po / (kscale * gamma);
        break;
      }
    }
    if (i_part == 0)
      rpn_store((xMaxSeen + xMinSeen) / 2, NULL, xMotionCenterVar);
  }

#ifdef DEBUG
  printf("change_p0=%ld, n_part=%ld\n", change_p0, n_part);
#endif
  int64_t nTotal = n_part;
#if USE_MPI
  MPI_Allreduce(&n_part, &nTotal, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (change_p0 && nTotal!=0) {
#ifdef DEBUG
    printf("Running do_match_energy with p0=%le\n", *pCentral);
#endif  
    do_match_energy(part, n_part, pCentral, 0);
#ifdef DEBUG
    printf("Ran do_match_energy, got p0=%le\n", *pCentral);
#endif  
  }
  
  log_exit((char *)"motion");
  return (i_top + 1);
}

void *field_global;

void (*set_up_derivatives(
                          void *field,
                          long field_type,
                          double *kscale,
                          double z_start,
                          double *Z_end_inner_scope,
                          double *tau_start,
                          double **part,
                          int64_t n_part,
                          double P_central_inner_scope,
                          double *X_limit_inner_scope,
                          double *Y_limit_inner_scope,
                          double *accuracy,
                          long *n_steps))(double *, double *, double) {

  TMCF_MODE *tmcf;
  CE_PLATES *cep;
  TW_PLATES *twp;
  TW_LINAC *twla;
  TWMTA *twmta;
  RFTMEZ0 *rftmEz0;
  MAP_SOLENOID *mapSol;
  LSRMDLTR *lsrMdltr;

  double *fiducial, gamma, Po, Pz, gamma_w, omega;
  double Escale, Bscale, Ku, beta;

  field_global = field;

  Escale = (Bscale = particleCharge / particleMass) / c_mks;
  change_p0 = 0;

  switch (field_type) {
  case T_MAPSOLENOID:
    mapSol = (MAP_SOLENOID *)field;
    *kscale = 1;
    if (!mapSol->initialized)
      setupMapSolenoidFromFile(mapSol, mapSol->length);
    *tau_start = 0;
    *Z_end_inner_scope = mapSol->length;
    X_offset = -(X_center = X_aperture_center = mapSol->dx);
    Y_offset = -(Y_center = Y_aperture_center = mapSol->dy);
    Z_center = mapSol->length / 2;
    /* assume 1000m aperture for now */
    *X_limit_inner_scope = *Y_limit_inner_scope = 1e3;
    *accuracy = mapSol->accuracy;
    *n_steps = mapSol->n_steps;
    mapSol->zMap0 = mapSol->length / 2 - mapSol->lUniform / 2;
    mapSol->zMap1 = mapSol->length / 2 + mapSol->lUniform / 2;
    mapSol->BxUniformScaled = mapSol->BxUniform * particleCharge / (particleMass * c_mks);
    mapSol->ByUniformScaled = mapSol->ByUniform * particleCharge / (particleMass * c_mks);
    select_integrator(mapSol->method);
    setupRotate3Matrix((void **)&partRot, -mapSol->eTilt, -mapSol->eYaw, -mapSol->ePitch);
    setupRotate3Matrix((void **)&fieldRot, mapSol->eTilt, mapSol->eYaw, mapSol->ePitch);
    return (derivatives_mapSolenoid);
  case T_LSRMDLTR:
    lsrMdltr = (LSRMDLTR *)field;
    *tau_start = 0;
    select_integrator(lsrMdltr->method);
    *accuracy = lsrMdltr->accuracy;
    *n_steps = lsrMdltr->nSteps;
    lsrMdltr->ku = 2 * PI / (lsrMdltr->length / lsrMdltr->periods);
    gamma = sqrt(sqr(P_central_inner_scope) + 1);
    beta = P_central_inner_scope / gamma;
    Ku = lsrMdltr->Bu * particleCharge / (particleMass * c_mks) / (beta * lsrMdltr->ku);
    if ((lsrMdltr->fieldCode = match_string(lsrMdltr->fieldExpansion,
                                            lsrMdltrFieldExpansion, N_LSRMDLTR_FIELD_EXPANSIONS, 0)),
        0) {
      long i;
      fputs((char *)"Error: field expansion for LSRMDLTR elements must be one of:\n", stdout);
      for (i = 0; i < N_LSRMDLTR_FIELD_EXPANSIONS; i++)
        printf((char *)"    %s\n", lsrMdltrFieldExpansion[i]);
      exitElegant(1);
    }
    if (lsrMdltr->fieldCode != 1 && lsrMdltr->tguGradient != 0)
      bombElegant("non-zero TGU_GRADIENT is only supported for LSRMDLTR with \"exact\" field model", NULL);
    if (lsrMdltr->isr && integratorCode != NA_RUNGE_KUTTA)
      bombElegant((char *)"LSRMDLTR with ISR must use non-adaptive runge-kutta integration", NULL);
    if (lsrMdltr->usersLaserWavelength)
      lsrMdltr->laserWavelength = lsrMdltr->usersLaserWavelength;
    else
      lsrMdltr->laserWavelength = lsrMdltr->length / lsrMdltr->periods / (2 * sqr(gamma)) * (1 + sqr(Ku) / 2);
    lsrMdltr->k = *kscale = PIx2 / lsrMdltr->laserWavelength;
    lsrMdltr->ZRayleigh = lsrMdltr->k / 2 * sqr(lsrMdltr->laserW0);
    lsrMdltr->omega = omega = *kscale * c_mks;
    lsrMdltr->Escale = particleCharge / (particleMass * omega * c_mks);
    lsrMdltr->Bscale = particleCharge / (particleMass * omega);
    if (lsrMdltr->laserM < 0 || lsrMdltr->laserN < 0)
      bombElegant((char *)"M and N must be non-negative for LSRMDLTR", NULL);
    lsrMdltr->Ef0Laser = 2 / lsrMdltr->laserW0 *
      sqrt(lsrMdltr->laserPeakPower / (ipow(2, lsrMdltr->laserM + lsrMdltr->laserN) * PI * factorial(lsrMdltr->laserM) * factorial(lsrMdltr->laserN) * epsilon_o * c_mks));
    X_offset = 0;
    X_aperture_center = X_center = 0;
    Y_aperture_center = Y_center = 0;
    Y_offset = 0;
    *Z_end_inner_scope = lsrMdltr->length * (*kscale);
    Z_center = lsrMdltr->length / 2 * (*kscale);
    *X_limit_inner_scope = *Y_limit_inner_scope = 1e300;
    lsrMdltr->t0 = 0;
    if (lsrMdltr->timeProfileFile) {
      long i;
      TABLE data;
      if (!getTableFromSearchPath(&data, lsrMdltr->timeProfileFile, 1, 0))
        bombElegant((char *)"unable to read data for LSRMDLTR time profile", NULL);
      if (data.n_data <= 1)
        bombElegant((char *)"LSRMDLTR time profile contains less than 2 points", NULL);
      lsrMdltr->timeValue = data.c1;
      lsrMdltr->amplitudeValue = data.c2;
      lsrMdltr->tProfilePoints = data.n_data;
      for (i = 1; i < data.n_data; i++)
        if (data.c1[i - 1] > data.c1[i])
          bombElegant((char *)"time values not monotonically increasing in LSRMDLTR time profile", NULL);
      tfree(data.xlab);
      tfree(data.ylab);
      tfree(data.title);
      tfree(data.topline);
    }
    lsrMdltr->t0 = findFiducialTime(part, n_part, 0, lsrMdltr->length / 2, P_central_inner_scope, FID_MODE_TMEAN) + lsrMdltr->timeProfileOffset;
    return (derivatives_laserModulator);
  case T_RFTMEZ0:
    rftmEz0 = (RFTMEZ0 *)field;
    *kscale = (omega = PIx2 * rftmEz0->frequency) / c_mks;
    if (rftmEz0->change_p0)
      change_p0 = 1;
    if (!rftmEz0->Ez) {
      setupRftmEz0FromFile(rftmEz0, rftmEz0->frequency, rftmEz0->length);
      setupRftmEz0SolenoidFromFile(rftmEz0, rftmEz0->length, *kscale);
    }
    if (!rftmEz0->fiducial_part) {
      /* This is the fiducial particle--the phase offset is set so
       * that its phase when it reaches the cavity center will be
       * approximately rftmEz0->phase + omega*rftmEz0->time_offset.
       * Several other quantities are calculated here also.
       */
      rftmEz0->k = *kscale;
      rftmEz0->fiducial_part = fiducial = select_fiducial(part, n_part, rftmEz0->fiducial);
      if (fiducial) {
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        Pz = Po * sqrt(1 - sqr(fiducial[0]) - sqr(fiducial[2]));
        rftmEz0->phase0 =
          get_reference_phase((long)(rftmEz0->phase_reference + .5),
                              -rftmEz0->k * gamma * (rftmEz0->length / (2 * Pz) + fiducial[4] / Po));
      } else {
        /* phase to v=c */
        gamma = sqrt(1 + sqr(P_central_inner_scope));
        rftmEz0->phase0 = get_reference_phase((long)(rftmEz0->phase_reference + .5),
                                              -rftmEz0->k * rftmEz0->length / 2 + z_start);
        rftmEz0->fiducial_part = part[0];
      }
    }
    *tau_start = rftmEz0->phase + PIx2 * rftmEz0->frequency * rftmEz0->time_offset +
      rftmEz0->phase0;
    *Z_end_inner_scope = rftmEz0->length * (*kscale);
    X_center = X_aperture_center = rftmEz0->k * rftmEz0->dx;
    Y_center = Y_aperture_center = rftmEz0->k * rftmEz0->dy;
    Z_center = *Z_end_inner_scope / 2 + rftmEz0->k * rftmEz0->dzMA;
    X_center1 = rftmEz0->k * rftmEz0->dxSol;
    Y_center1 = rftmEz0->k * rftmEz0->dySol;
    Z_center1 = *Z_end_inner_scope / 2 + rftmEz0->k * rftmEz0->dzSolMA;
    /* assume 1000m aperture for now */
    *X_limit_inner_scope = rftmEz0->k * 1e3;
    *Y_limit_inner_scope = rftmEz0->k * 1e3;
    *accuracy = rftmEz0->accuracy;
    *n_steps = rftmEz0->n_steps;
    select_integrator(rftmEz0->method);
    setupRotate3Matrix((void **)&partRot, -rftmEz0->eTilt, -rftmEz0->eYaw, -rftmEz0->ePitch);
    setupRotate3Matrix((void **)&fieldRot, rftmEz0->eTilt, rftmEz0->eYaw, rftmEz0->ePitch);
    setupRotate3Matrix((void **)&partRot1, -rftmEz0->eTiltSol, -rftmEz0->eYawSol, -rftmEz0->ePitchSol);
    setupRotate3Matrix((void **)&fieldRot1, rftmEz0->eTiltSol, rftmEz0->eYawSol, rftmEz0->ePitchSol);
    rftmEz0->BxStrayScaled = rftmEz0->BxStray * particleCharge / (particleMass * rftmEz0->k * c_mks);
    rftmEz0->ByStrayScaled = rftmEz0->ByStray * particleCharge / (particleMass * rftmEz0->k * c_mks);
    if (!rftmEz0->initialized) {
      makeRftmEz0FieldTestFile(rftmEz0);
      rftmEz0->initialized = 1;
    }
    return (derivatives_rftmEz0);
  case T_TMCF:
    tmcf = (TMCF_MODE *)field;
    *kscale = (omega = PIx2 * tmcf->frequency) / c_mks;
    if (!tmcf->fiducial_part) {
      /* This is the fiducial particle--the phase offset is set so
       * that its phase when it reaches the cavity center will be
       * approximately tmcf->phase + omega*tmcf->time_offset.
       * Several other quantities are calculated here also.
       */
      tmcf->k = (omega = PIx2 * tmcf->frequency) / c_mks;
      *kscale = tmcf->k;
      tmcf->fiducial_part = fiducial = select_fiducial(part, n_part, tmcf->fiducial);
      if (fiducial) {
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        Pz = Po * sqrt(1 - sqr(fiducial[0]) - sqr(fiducial[2]));
        tmcf->phase0 =
          get_reference_phase((long)(tmcf->phase_reference + .5),
                              -tmcf->k * gamma * (tmcf->length / (2 * Pz) + fiducial[4] / Po));
      } else {
        /* phase to v=c */
        gamma = sqrt(1 + sqr(P_central_inner_scope));
        tmcf->phase0 = get_reference_phase((long)(tmcf->phase_reference + .5),
                                           -tmcf->k * tmcf->length / 2 + z_start);
        tmcf->fiducial_part = part[0];
      }
      tmcf->Er *= particleCharge / (particleMass * c_mks * omega);
      tmcf->Ez *= particleCharge / (particleMass * c_mks * omega);
      tmcf->Bphi *= particleCharge / (particleMass * omega);
    }
    *tau_start = tmcf->phase + PIx2 * tmcf->frequency * tmcf->time_offset +
      tmcf->phase0;
    *Z_end_inner_scope = tmcf->length * (*kscale);
    X_offset = tmcf->k * tmcf->radial_offset * cos(tmcf->tilt);
    Y_offset = tmcf->k * tmcf->radial_offset * sin(tmcf->tilt);
    *X_limit_inner_scope = tmcf->k * tmcf->x_max;
    *Y_limit_inner_scope = tmcf->k * tmcf->y_max;
    X_aperture_center = tmcf->k * tmcf->dx;
    Y_aperture_center = tmcf->k * tmcf->dy;
    *accuracy = tmcf->accuracy;
    *n_steps = tmcf->n_steps;
    select_integrator(tmcf->method);
    return (derivatives_tmcf_mode);
  case T_CEPL:
    cep = (CE_PLATES *)field;
    *kscale = 1. / (c_mks * cep->ramp_time);
    if (!cep->fiducial_part) {
      /* This is the fiducial particle--the tau offset tau0 is set
       * so that its tau when it reaches the cavity center will be
       * approximately 0. Several other quantities are calculated
       * here also.
       */
      *kscale = cep->k = 1. / (c_mks * cep->ramp_time);
      cep->fiducial_part = fiducial = select_fiducial(part, n_part, cep->fiducial);
      if (fiducial) {
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        Pz = Po * sqrt(1 - sqr(fiducial[0]) - sqr(fiducial[2]));
        cep->tau0 = get_reference_phase(cep->phase_reference,
                                        -cep->k * gamma * (cep->length / (2 * Pz) + fiducial[4] / Po));
      } else {
        /* phase to v=c */
        cep->tau0 = get_reference_phase((long)(cep->phase_reference + .5),
                                        -cep->k * cep->length / 2 + z_start);
        cep->fiducial_part = part[0];
      }
    }
    cep->E_scaled = particleCharge * cep->voltage * cep->ramp_time /
      (cep->gap * particleMass * c_mks);
    cep->E_static = particleCharge * cep->static_voltage * cep->ramp_time /
      (cep->gap * particleMass * c_mks);
    cep->cos_tilt = cos(cep->tilt);
    cep->sin_tilt = sin(cep->tilt);
    *tau_start = cep->time_offset / cep->ramp_time + cep->tau0;
    *Z_end_inner_scope = cep->length * (*kscale);
    *accuracy = cep->accuracy;
    *n_steps = cep->n_steps;
    X_offset = Y_offset = 0;
    *X_limit_inner_scope = cep->k * cep->x_max;
    *Y_limit_inner_scope = cep->k * cep->y_max;
    X_aperture_center = cep->k * cep->dx;
    Y_aperture_center = cep->k * cep->dy;
    select_integrator(cep->method);
    return (derivatives_ce_plates);
  case T_TWPL:
    twp = (TW_PLATES *)field;
    *kscale = 1. / (c_mks * twp->ramp_time);
    if (!twp->fiducial_part) {
      /* This is the fiducial particle--the tau offset tau0 is set
       * so that its tau when it reaches the cavity center will be
       * approximately omega*time_offset. Several other quantities are calculated
       * here also.
       */
      *kscale = twp->k = 1. / (c_mks * twp->ramp_time);
      twp->fiducial_part = fiducial = select_fiducial(part, n_part, twp->fiducial);
      if (fiducial) {
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        Pz = Po * sqrt(1 - sqr(fiducial[0]) - sqr(fiducial[2]));
        twp->tau0 = get_reference_phase(twp->phase_reference,
                                        -twp->k * gamma * (twp->length / (2 * Pz) + fiducial[4] / Po));
      } else {
        /* phase to v=c */
        twp->tau0 = get_reference_phase((long)(twp->phase_reference + .5),
                                        -twp->k * twp->length / 2 + z_start);
        twp->fiducial_part = part[0];
      }
    }
    twp->E_scaled = particleCharge * twp->voltage * twp->ramp_time /
      (twp->gap * particleMass * c_mks);
    twp->E_static = particleCharge * twp->static_voltage * twp->ramp_time /
      (twp->gap * particleMass * c_mks);
    twp->cos_tilt = cos(twp->tilt);
    twp->sin_tilt = sin(twp->tilt);
    *tau_start = twp->time_offset / twp->ramp_time + twp->tau0;
    *Z_end_inner_scope = twp->length * (*kscale);
    *accuracy = twp->accuracy;
    *n_steps = twp->n_steps;
    X_offset = Y_offset = 0;
    *X_limit_inner_scope = twp->k * twp->x_max;
    *Y_limit_inner_scope = twp->k * twp->y_max;
    X_aperture_center = twp->k * twp->dx;
    Y_aperture_center = twp->k * twp->dy;
#if defined(DEBUG)
    printf((char *)"TWPL parameters:\n");
    fflush(stdout);
    printf((char *)"l=%le acc=%le x_max=%le y_max=%le\n",
           twp->length, twp->accuracy, twp->x_max, twp->y_max);
    fflush(stdout);
    printf((char *)"dx=%le dy=%le method=%s ramp_time=%le phiref=%ld\n",
           twp->dx, twp->dy, twp->method, twp->ramp_time,
           twp->phase_reference);
    fflush(stdout);
#endif
    select_integrator(twp->method);
    return (derivatives_tw_plates);
  case T_TWLA:
    twla = (TW_LINAC *)field;
    radial_limit = 1;
    *kscale = (omega = twla->frequency * PIx2) / c_mks;
    twla->alphaS = twla->alpha / (*kscale);
    if (twla->change_p0)
      change_p0 = 1;
    gamma = sqrt(1 + sqr(P_central_inner_scope));
    if (!twla->fiducial_part) {
      /* This is the fiducial particle--the phase offset phase0 is
       * set so that its initial phase will be twla->phase +
       * omega*twla->time_offset. Several other quantities are
       * calculated here also.
       */
      twla->kz = *kscale / twla->beta_wave;
      twla->fiducial_part = fiducial = select_fiducial(part, n_part, twla->fiducial);
      if (fiducial) {
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        /* find the phase offset, phase0 = -omega*t_fiducial , or else equivalent value
         * for phase reference */
        twla->phase0 = get_reference_phase(twla->phase_reference,
                                           -omega / c_mks * gamma * fiducial[4] / Po);
      } else {
        /* use v=c */
        twla->phase0 = get_reference_phase(twla->phase_reference, -omega / c_mks * z_start);
        twla->fiducial_part = part[0];
      }
    }
    Escale /= omega;
    Bscale /= omega;
    twla->ErS = -twla->Ez * twla->kz / 2 * Escale * (twla->focussing ? 1 : 0) / (*kscale);
    twla->BphiS = -twla->Ez * omega / (2 * sqr(c_mks)) * Bscale * (twla->focussing ? 1 : 0) / (*kscale);
    twla->EzS = twla->Ez * Escale;
    twla->BsolS = twla->B_solenoid * Bscale;
    twla->FrP = 4 * sqr(particleCharge * twla->Ez / (omega * particleMass * c_mks)) / gamma * twla->sum_bn2;
    if (twla->sum_bn2 != 0 && fabs(particleCharge * twla->Ez * PI / (2 * twla->kz * me_mks * sqr(c_mks)) / P_central_inner_scope) > 0.1)
      printWarning((char *)"TWLA does not satisfy requirements for validity of Hartman-Rosenzweig ponderomotive transverse focusing treatment.", NULL);
    /* calculate initial tau value, less omega*t:
     *    tau_start = omega*(t_offset-t_fid)+phase
     *              = omega*t_offset + phase + phase0
     */
    *tau_start = twla->phase + omega * twla->time_offset + twla->phase0;
    *Z_end_inner_scope = twla->length * (*kscale);
    X_offset = X_aperture_center = (*kscale) * twla->dx;
    Y_offset = Y_aperture_center = (*kscale) * twla->dy;
    *X_limit_inner_scope = (*kscale) * twla->x_max;
    *Y_limit_inner_scope = (*kscale) * twla->y_max;
    *accuracy = twla->accuracy;
    *n_steps = twla->n_steps;
    select_integrator(twla->method);
    derivatives = derivatives_tw_linac;
    input_impulse = input_impulse_tw_linac;
    output_impulse = output_impulse_tw_linac;
    return (derivatives_tw_linac);
  case T_TWMTA:
    twmta = (TWMTA *)field;
    radial_limit = 0;
    *kscale = (omega = twmta->frequency * PIx2) / c_mks;
    twmta->alphaS = twmta->alpha / (*kscale);
    if (!twmta->fiducial_part) {
      /* This is the fiducial particle--the phase offset phase0 is
       * set so that its initial phase will be twmta->phase +
       * omega*twmta->time_offset. Several other quantities are
       * calculated here also.
       */
      twmta->kz = (*kscale) / twmta->beta_wave;
      gamma_w = 1 / sqrt(1 - sqr(twmta->beta_wave));
      twmta->ky = sqrt(sqr(twmta->kx) + sqr(twmta->kz / gamma_w));
      twmta->Kx = twmta->kx / (*kscale);
      twmta->Ky = twmta->ky / (*kscale);
      twmta->Kz = twmta->kz / (*kscale);
      /*
        printf((char*)"TWMTA kx, ky, kz = %e, %e, %e\n", twmta->kx, twmta->ky, twmta->kz);
        fflush(stdout);
        printf((char*)"      Ex, Ey, Ez = %e, %e, %e\n", twmta->ExS, twmta->EyS, twmta->EzS);
        fflush(stdout);
        printf((char*)"      Bx, By, Bz = %e, %e\n", twmta->BxS, twmta->ByS);
        fflush(stdout);
      */
      twmta->fiducial_part = fiducial = select_fiducial(part, n_part, twmta->fiducial);
      if (fiducial) {
        /* find the phase offset, phase0 = -omega*t_fiducial , or else equivalent value
         * for phase reference */
        Po = P_central_inner_scope * (1 + fiducial[5]);
        gamma = sqrt(1 + sqr(Po));
        twmta->phase0 = get_reference_phase(twmta->phase_reference,
                                            -omega / c_mks * gamma * fiducial[4] / Po);
      } else {
        /* use v=c */
        twmta->phase0 = get_reference_phase(twmta->phase_reference, -omega / c_mks * z_start);
        fiducial = part[0];
      }
    }
    Escale /= omega;
    Bscale /= omega;
    twmta->EzS = twmta->Ez * Escale;
    twmta->ExS = twmta->EzS * twmta->kx * twmta->kz / (sqr(twmta->ky) - sqr(twmta->kx));
    twmta->EyS = -twmta->ExS * twmta->ky / twmta->kx;
    twmta->BxS = twmta->Ez / omega * Bscale * twmta->ky *
      (sqr(twmta->kx) - sqr(twmta->ky) + sqr(twmta->kz)) / (sqr(twmta->ky) - sqr(twmta->kx));
    twmta->ByS = twmta->BxS * twmta->kx / twmta->ky;
    twmta->BsolS = twmta->Bsol * Bscale;
    /* calculate initial tau value, less omega*t:
     *    tau_start = omega*(-t_fid)+phase
     *              = phase + phase0
     */
    *tau_start = twmta->phase + twmta->phase0;
    *Z_end_inner_scope = twmta->length * (*kscale);
    X_offset = X_aperture_center = (*kscale) * twmta->dx;
    Y_offset = Y_aperture_center = (*kscale) * twmta->dy;
    *X_limit_inner_scope = (*kscale) * twmta->x_max;
    *Y_limit_inner_scope = (*kscale) * twmta->y_max;
    *accuracy = twmta->accuracy;
    *n_steps = twmta->n_steps;
    select_integrator(twmta->method);
    input_impulse = input_impulse_twmta;
    output_impulse = output_impulse_twmta;
    return (derivatives_twmta);
  default:
    bombElegant((char *)"invalid mode type (set_up_derivatives", NULL);
    break;
  }
  exitElegant(1);
  return NULL; /* suppresses spurious compiler warning */
}

void (*set_up_stochastic_effects(long field_type))(double *, double, double) {
  switch (field_type) {
  case T_LSRMDLTR:
    return (stochastic_laserModulator);
  default:
    return NULL;
    break;
  }
}

void derivatives_rftmEz0(
                         double *qp, /* derivatives w.r.t. phase */
                         double *q,  /* X,Y,Z,Px,Py,Pz */
                         double tau) {
  double gamma, *P, *Pp;
  double E[3], BOverGamma[3], BOverGammaDC[3];
  long i;

  derivCalls++;
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  computeFields_rftmEz0(E, BOverGamma, BOverGammaDC,
                        q[0], q[1], q[2], tau, gamma);
  for (i = 0; i < 3; i++)
    BOverGamma[i] += BOverGammaDC[i];

  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  Pp[0] = -(E[0] + (P[1] * BOverGamma[2] - P[2] * BOverGamma[1]));
  Pp[1] = -(E[1] + (P[2] * BOverGamma[0] - P[0] * BOverGamma[2]));
  Pp[2] = -(E[2] + (P[0] * BOverGamma[1] - P[1] * BOverGamma[0]));
}

void computeFields_rftmEz0(double *E, double *BOverGamma, double *BOverGammaDC,
                           double q0, double q1, double q2, double tau, double gamma) {
  double XYZ[3];
  RFTMEZ0 *rftmEz0;
  long iz, ir;
  double R, Zoffset, BphiOverRG, ErOverR, BrOverRG;
  double X, Y, Z, sinPhase, cosPhase;
  double B1, B2;

  rftmEz0 = (RFTMEZ0 *)field_global;

  /*** scaled RF fields ***/
  /* find coordinates relative to element center */
  XYZ[0] = q0 - X_center;
  XYZ[1] = q1 - Y_center;
  XYZ[2] = q2 - Z_center;
  /* perform coordinate rotation into the element frame */
  rotate3(XYZ, partRot);
  X = XYZ[0];
  Y = XYZ[1];
  Z = XYZ[2] + Z_center; /* Z is expressed relative to element start */
  R = sqrt(sqr(X) + sqr(Y));

  iz = Z / rftmEz0->dZ;
  if (iz < 0 || iz >= rftmEz0->nz) {
    E[0] = E[1] = E[2] = 0;
    BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
  } else {
    if (iz == (rftmEz0->nz - 1))
      iz -= 1;
    Zoffset = Z - iz * rftmEz0->dZ;
    sinPhase = sin(tau);
    cosPhase = cos(tau);
    E[2] = (rftmEz0->Ez[iz] + (rftmEz0->Ez[iz + 1] - rftmEz0->Ez[iz]) / rftmEz0->dZ * Zoffset) * rftmEz0->Ez_peak;
    if (rftmEz0->radial_order) {
      ErOverR = -(rftmEz0->dEzdZ[iz] +
                  (rftmEz0->dEzdZ[iz + 1] - rftmEz0->dEzdZ[iz]) / rftmEz0->dZ * Zoffset) /
        2 * sinPhase * rftmEz0->Ez_peak;
      E[0] = ErOverR * X;
      E[1] = ErOverR * Y;
      BphiOverRG = E[2] / 2 / gamma * cosPhase;
      BOverGamma[0] = -BphiOverRG * Y;
      BOverGamma[1] = BphiOverRG * X;
      BOverGamma[2] = 0;
    } else {
      E[0] = E[1] = 0;
      BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
    }
    E[2] *= sinPhase;
  }

  /* E and BOverGama contain field components in the element frame for
   * the present particle position (which was expressed in the element frame
   * also).  Rotate these components back to the lab frame.
   */
  rotate3(E, fieldRot);
  rotate3(BOverGamma, fieldRot);

  BOverGammaDC[0] = BOverGammaDC[1] = BOverGammaDC[2] = 0;
  if (rftmEz0->BzSol) {
    /** scaled solenoid fields **/
    /* find coordinates relative to element center */
    XYZ[0] = q0 - X_center1;
    XYZ[1] = q1 - Y_center1;
    XYZ[2] = q2 - Z_center1;
    /* perform coordinate rotation into the element frame */
    rotate3(XYZ, partRot1);
    X = XYZ[0];
    Y = XYZ[1];
    Z = XYZ[2] + Z_center1; /* Z is expressed relative to element start */
    R = sqrt(sqr(X) + sqr(Y));

    iz = (Z - rftmEz0->Z0Sol) / rftmEz0->dZSol;
    Zoffset = Z - (iz * rftmEz0->dZSol + rftmEz0->Z0Sol);
    if (rftmEz0->dRSol)
      ir = R / rftmEz0->dRSol;
    else
      ir = 0;
    if (iz >= 0 && iz < rftmEz0->nzSol && (rftmEz0->nrSol == 1 || ir < rftmEz0->nrSol)) {
      if (iz == rftmEz0->nzSol - 1)
        iz -= 1;
      if (ir == rftmEz0->nrSol - 1)
        ir -= 1;
      if (rftmEz0->nrSol == 1) {
        /* do off-axis expansion */
        BOverGammaDC[2] = (rftmEz0->BzSol[0][iz] +
                           (rftmEz0->BzSol[0][iz + 1] - rftmEz0->BzSol[0][iz]) / rftmEz0->dZSol * Zoffset) /
          gamma * rftmEz0->solenoidFactor;
        BrOverRG = -0.5 * (rftmEz0->dBzdZSol[iz] + (rftmEz0->dBzdZSol[iz + 1] - rftmEz0->dBzdZSol[iz]) / rftmEz0->dZSol * Zoffset) / gamma * rftmEz0->solenoidFactor;
        /* compute x and y components */
        BOverGammaDC[0] = X * BrOverRG;
        BOverGammaDC[1] = Y * BrOverRG;
      } else {
        BrOverRG = 0;
        if (R > 0) {
          /* compute Br/R */
          B1 = rftmEz0->BrSol[ir][iz] +
            (rftmEz0->BrSol[ir + 1][iz] - rftmEz0->BrSol[ir][iz]) / rftmEz0->dRSol * (R - ir * rftmEz0->dRSol);
          B2 = rftmEz0->BrSol[ir][iz + 1] +
            (rftmEz0->BrSol[ir + 1][iz + 1] - rftmEz0->BrSol[ir][iz + 1]) / rftmEz0->dRSol * (R - ir * rftmEz0->dRSol);
          BrOverRG = (B1 + (B2 - B1) * Zoffset / rftmEz0->dZSol) / gamma / R * rftmEz0->solenoidFactor;
          /* compute x and y components */
          BOverGammaDC[0] = X * BrOverRG;
          BOverGammaDC[1] = Y * BrOverRG;
        }
        /* compute Bz/Gamma */
        B1 = rftmEz0->BzSol[ir][iz] +
          (rftmEz0->BzSol[ir + 1][iz] - rftmEz0->BzSol[ir][iz]) / rftmEz0->dRSol * (R - ir * rftmEz0->dRSol);
        B2 = rftmEz0->BzSol[ir][iz + 1] +
          (rftmEz0->BzSol[ir + 1][iz + 1] - rftmEz0->BzSol[ir][iz + 1]) / rftmEz0->dRSol * (R - ir * rftmEz0->dRSol);
        BOverGammaDC[2] =
          (B1 + (B2 - B1) * Zoffset / rftmEz0->dZSol) / gamma * rftmEz0->solenoidFactor;
      }
    }
    rotate3(BOverGammaDC, fieldRot1);
  }
  BOverGammaDC[0] += rftmEz0->BxStrayScaled / gamma;
  BOverGammaDC[1] += rftmEz0->ByStrayScaled / gamma;
}

void derivatives_mapSolenoid(
                             double *qp, /* derivatives w.r.t. phase */
                             double *q,  /* X,Y,Z,Px,Py,Pz */
                             double tau) {
  double gamma, *P, *Pp;
  double BOverGamma[3];
  double XYZ[3];
  MAP_SOLENOID *mapSol;
  long iz, ir;
  double R, Zoffset, BrOverRG;
  double X, Y, Z;
  double B1, B2;

  derivCalls++;
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* find coordinates relative to element center */
  XYZ[0] = q[0] - X_center;
  XYZ[1] = q[1] - Y_center;
  XYZ[2] = q[2] - Z_center;
  /* perform coordinate rotation */
  rotate3(XYZ, partRot);
  X = XYZ[0];
  Y = XYZ[1];
  Z = XYZ[2] + Z_center;
  R = sqrt(sqr(X) + sqr(Y));

  mapSol = (MAP_SOLENOID *)field_global;
  iz = Z / mapSol->dz;
  Zoffset = Z - iz * mapSol->dz;
  ir = R / mapSol->dr;

  BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
  if (iz >= 0 && iz < mapSol->nz && ir < mapSol->nr) {
    if (iz == mapSol->nz - 1)
      iz -= 1;
    if (ir == mapSol->nr - 1)
      ir -= 1;
    BrOverRG = 0;
    if (R > 0) {
      /* compute Br/R */
      B1 = mapSol->Br[ir][iz] +
        (mapSol->Br[ir + 1][iz] - mapSol->Br[ir][iz]) / mapSol->dr * (R - ir * mapSol->dr);
      B2 = mapSol->Br[ir][iz + 1] +
        (mapSol->Br[ir + 1][iz + 1] - mapSol->Br[ir][iz + 1]) / mapSol->dr * (R - ir * mapSol->dr);
      BrOverRG = (B1 + (B2 - B1) * Zoffset / mapSol->dz) / gamma / R * mapSol->factor;
      BOverGamma[0] += X * BrOverRG;
      BOverGamma[1] += Y * BrOverRG;
    }
    /* compute Bz/Gamma */
    B1 = mapSol->Bz[ir][iz] +
      (mapSol->Bz[ir + 1][iz] - mapSol->Bz[ir][iz]) / mapSol->dr * (R - ir * mapSol->dr);
    B2 = mapSol->Bz[ir][iz + 1] +
      (mapSol->Bz[ir + 1][iz + 1] - mapSol->Bz[ir][iz + 1]) / mapSol->dr * (R - ir * mapSol->dr);
    BOverGamma[2] = (B1 + (B2 - B1) * Zoffset / mapSol->dz) / gamma * mapSol->factor;
  }

  if ((mapSol->BxUniform || mapSol->ByUniform) &&
      (Z > mapSol->zMap0 && Z < mapSol->zMap1)) {
    /* add components from uniform field, if any */
    BOverGamma[0] += mapSol->BxUniformScaled / gamma;
    BOverGamma[1] += mapSol->ByUniformScaled / gamma;
  }
  /* BOverGama contains field components in the element frame for
   * the present particle position (which was expressed in the element frame
   * also).  Rotate these components back to the lab frame.
   */
  rotate3(BOverGamma, fieldRot);

  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  Pp[0] = -(P[1] * BOverGamma[2] - P[2] * BOverGamma[1]);
  Pp[1] = -(P[2] * BOverGamma[0] - P[0] * BOverGamma[2]);
  Pp[2] = -(P[0] * BOverGamma[1] - P[1] * BOverGamma[0]);
  /* Since the field components and momenta are in the lab frame,
   * I don't need to perform a rotation of the forces.
   */
}

void derivatives_tmcf_mode(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau) {
  double gamma, *P, *Pp;
  double phase, sin_phase, cos_phase;
  double sin_tilt, cos_tilt;
  double X, Y, R, Z;
  static double E[3], B[3];
  TMCF_MODE *tmcf;

  derivCalls++;
  /* gamma = sqrt(P^2+1) */
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* Calculate scaled RF fields E and B, for which                */
  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma       */
  Pp = qp + 3;
  tmcf = (TMCF_MODE *)field_global;
  if ((Z = q[2]) <= Z_end && Z >= 0) {
    /* The electric fields are multiplied by sin(w.t) and the
     * magnetic field by cos(w.t).  This is consistent with the
     * data dumped by SF02 and SHY.
     */
    E[2] = tmcf->Ez * (sin_phase = sin(phase = (tau)));
    Y = q[1] + Y_offset;
    X = q[0] + X_offset;
    R = sqrt(sqr(X) + sqr(Y));
    E[0] = tmcf->Er * (cos_tilt = (R ? X / R : 0)) * sin_phase;
    E[1] = tmcf->Er * (sin_tilt = (R ? Y / R : 0)) * sin_phase;
    B[2] = 0;
    /* B is scaled by gamma here */
    B[1] = tmcf->Bphi * (cos_phase = cos(phase)) / gamma;
    B[0] = -B[1] * sin_tilt;
    B[1] *= cos_tilt;

    /* The sign of the electron is taken care of here. */
    Pp[0] = -(E[0] - P[2] * B[1]);
    Pp[1] = -(E[1] + P[2] * B[0]);
    Pp[2] = -(E[2] + (P[0] * B[1] - P[1] * B[0]));
  } else
    Pp[0] = Pp[1] = Pp[2] = 0;
}

void derivatives_ce_plates(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau) {
  double gamma, *P, *Pp, t_ramp;
  double E, z;
  CE_PLATES *cep;

  derivCalls++;
  /* gamma = sqrt(P^2+1) */
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* get scaled RF fields E and B, for which                */
  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  cep = (CE_PLATES *)field_global;
  if ((z = q[2]) <= Z_end && z >= 0) {
    /* Electrons are deflected to positive coordinates by a negative
     * electric field.
     */
    if ((t_ramp = tau) < 1 && t_ramp > 0)
      E = cep->E_scaled * t_ramp;
    else if (t_ramp <= 0)
      E = 0;
    else
      E = cep->E_scaled;
    E += cep->E_static;

    /* The sign of the electron charge is taken care of here. */
    Pp[0] = -E * cep->cos_tilt;
    Pp[1] = -E * cep->sin_tilt;
    Pp[2] = 0;
  } else
    Pp[0] = Pp[1] = Pp[2] = 0;
}

void derivatives_tw_plates(
                           double *qp, /* derivatives w.r.t. tau */
                           double *q,  /* X,Y,Z,Px,Py,Pz */
                           double tau) {
  double gamma, *P, *Pp, t_ramp;
  double E, z, B;
  TW_PLATES *twp;

  derivCalls++;
  /* gamma = sqrt(P^2+1) */
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* get scaled RF fields E and B, for which                */
  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  twp = (TW_PLATES *)field_global;
  if ((z = q[2]) <= Z_end && z >= 0) {
    /* Electrons are deflected to positive coordinates by a negative
     * electric field.  The magnetic field deflects in the same direction.
     */
    if ((t_ramp = tau + z - Z_end) < 1 && t_ramp > 0)
      E = twp->E_scaled * t_ramp;
    else if (t_ramp < 0)
      E = 0;
    else
      E = twp->E_scaled;
    E += twp->E_static;
    B = -E / gamma;
    Pp[0] = -(E - P[2] * B) * twp->cos_tilt;
    Pp[1] = -(E - P[2] * B) * twp->sin_tilt;
    Pp[2] = -B * (P[0] * twp->cos_tilt + P[1] * B * twp->sin_tilt);
  } else
    Pp[0] = Pp[1] = Pp[2] = 0;
}

void derivatives_tw_linac(
                          double *qp, /* derivatives w.r.t. tau */
                          double *q,  /* X,Y,Z,Px,Py,Pz */
                          double tau) {
  double gamma, *P, *Pp;
  static double E[3], B[3], FrP[2];
  double phase, X, Y, Z, cos_phase, sin_phase, droop;
  TW_LINAC *twla;

  derivCalls++;
  /* gamma = sqrt(P^2+1) */
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* get scaled RF fields E and B, for which                */
  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  twla = (TW_LINAC *)field_global;
  FrP[0] = FrP[1] = 0;
  droop = 1;
  if ((Z = q[2]) <= Z_end && Z >= 0) {
    if (twla->alphaS)
      droop = exp(-twla->alphaS * Z);
    E[2] = twla->EzS * (sin_phase = sin(phase = Z / twla->beta_wave - tau)) * droop;
#if defined(DEBUG)
    if (starting_integration) {
      starting_integration = 0;
      initial_phase = phase;
    } else
      final_phase = phase;
#endif
    Y = q[1] + Y_offset;
    X = q[0] + X_offset;
    E[0] = twla->ErS * (cos_phase = cos(phase)) * droop;
    E[1] = E[0] * Y;
    E[0] *= X;
    /* B is scaled by gamma here */
    B[2] = twla->BsolS / gamma;
    B[1] = twla->BphiS * cos_phase / gamma * droop;
    B[0] = -B[1] * Y;
    B[1] *= X;

    if (twla->FrP != 0) {
      /* Ponderomotive force from n!=0 space harmonics (Hartman et al, PRE 47, March 93 */
      FrP[0] = -X * twla->FrP * sqr(cos_phase);
      FrP[1] = -Y * twla->FrP * sqr(cos_phase);
    }

    /* The sign of the electron is taken care of here. */
    /* Also, the Ponderomotive force is always focusing */
    Pp[0] = -(E[0] + P[1] * B[2] - P[2] * B[1] - FrP[0]);
    Pp[1] = -(E[1] + P[2] * B[0] - P[0] * B[2] - FrP[1]);
    Pp[2] = -(E[2] + P[0] * B[1] - P[1] * B[0]);
  } else
    Pp[0] = Pp[1] = Pp[2] = 0;
}

void input_impulse_tw_linac(double *P, double *q) {
  double gamma;
  static double B[3];
  TW_LINAC *twla;

  /* gamma = sqrt(P^2+1) */
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  twla = (TW_LINAC *)field_global;
  if (twla->BsolS == 0)
    return;

  B[0] = -twla->BsolS * q[0] / (2 * gamma);
  B[1] = -twla->BsolS * q[1] / (2 * gamma);
  P[0] += P[2] * B[1];
  P[1] += -P[2] * B[0];
  P[2] += -P[0] * B[1] + P[1] * B[0];
}

void output_impulse_tw_linac(double *P, double *q) {
  double gamma;
  static double B[3];
  TW_LINAC *twla;

  /* gamma = sqrt(P^2+1) */
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  twla = (TW_LINAC *)field_global;
  if (twla->BsolS == 0)
    return;

  B[0] = twla->BsolS * q[0] / (2 * gamma);
  B[1] = twla->BsolS * q[1] / (2 * gamma);
  P[0] += P[2] * B[1];
  P[1] += -P[2] * B[0];
  P[2] += -P[0] * B[1] + P[1] * B[0];
}

void derivatives_twmta(
                       double *qp, /* derivatives w.r.t. tau */
                       double *q,  /* X,Y,Z,Px,Py,Pz */
                       double tau) {
  double gamma, *P, *Pp;
  static double E[3], B[3];
  double phase, X, Y, Z, droop;
  double sin_phase, cos_phase, cosh_KyY, sinh_KyY, cos_KxX, sin_KxX;
  TWMTA *twmta;

  derivCalls++;
  /* gamma = sqrt(P^2+1) */
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  /* get scaled RF fields E and B, for which                */
  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  twmta = (TWMTA *)field_global;
  droop = 1;
  if ((Z = q[2]) <= Z_end && Z >= 0) {
    Y = q[1] + Y_offset;
    X = q[0] + X_offset;
    sin_phase = sin(phase = Z / twmta->beta_wave - tau);
    cos_phase = cos(phase);
#if defined(DEBUG)
    if (starting_integration) {
      starting_integration = 0;
      initial_phase = phase;
    } else
      final_phase = phase;
#endif
    cosh_KyY = cosh(twmta->Ky * Y);
    sinh_KyY = sinh(twmta->Ky * Y);
    cos_KxX = cos(twmta->Kx * X);
    sin_KxX = sin(twmta->Kx * X);
    E[0] = twmta->ExS * sin_KxX * cosh_KyY * cos_phase;
    E[1] = twmta->EyS * cos_KxX * sinh_KyY * cos_phase;
    E[2] = twmta->EzS * cos_KxX * cosh_KyY * sin_phase;
    /* B is scaled by gamma here */
    B[0] = twmta->BxS / gamma * cos_KxX * sinh_KyY * cos_phase;
    B[1] = twmta->ByS / gamma * sin_KxX * cosh_KyY * cos_phase;
    B[2] = twmta->BsolS / gamma;

    if (twmta->alphaS)
      droop = exp(-twmta->alphaS * Z);

    /* The sign of the electron is taken care of here. */
    Pp[0] = -(E[0] + P[1] * B[2] - P[2] * B[1]) * droop;
    Pp[1] = -(E[1] + P[2] * B[0] - P[0] * B[2]) * droop;
    Pp[2] = -(E[2] + P[0] * B[1] - P[1] * B[0]) * droop;
  } else
    Pp[0] = Pp[1] = Pp[2] = 0;
}

void input_impulse_twmta(double *P, double *q) {
  double gamma;
  static double B[3];
  TWMTA *twmta;

  /* gamma = sqrt(P^2+1) */
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  twmta = (TWMTA *)field_global;
  if (twmta->BsolS == 0)
    return;

  B[0] = -twmta->BsolS * q[0] / (2 * gamma);
  B[1] = -twmta->BsolS * q[1] / (2 * gamma);
  P[0] += P[2] * B[1];
  P[1] += -P[2] * B[0];
  P[2] += -P[0] * B[1] + P[1] * B[0];
}

void output_impulse_twmta(double *P, double *q) {
  double gamma;
  static double B[3];
  TWMTA *twmta;

  /* gamma = sqrt(P^2+1) */
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  twmta = (TWMTA *)field_global;
  if (twmta->BsolS == 0)
    return;

  B[0] = twmta->BsolS * q[0] / (2 * gamma);
  B[1] = twmta->BsolS * q[1] / (2 * gamma);
  P[0] += P[2] * B[1];
  P[1] += -P[2] * B[0];
  P[2] += -P[0] * B[1] + P[1] * B[0];
}

double exit_function(
                     double *qp,
                     double *q,
                     double phase) {
  double X = 0.0, Y = 0.0, dZ;

  if ((dZ = Z_end - q[2]) >= 0 && !limit_hit) {
    if (radial_limit) {
      if (X_limit) {
        X = fabs(q[0] - X_aperture_center);
        Y = fabs(q[1] - Y_aperture_center);
        if (sqr(X_limit) < (sqr(X) + sqr(Y))) {
          X_out = X;
          Y_out = Y;
          Z_out = q[2];
          limit_hit = 1;
        }
      }
    } else {
      if ((X_limit && (X = fabs(q[0] - X_aperture_center)) > X_limit) ||
          (Y_limit && (Y = fabs(q[1] - Y_aperture_center)) > Y_limit)) {
        X_out = X;
        Y_out = Y;
        Z_out = q[2];
        limit_hit = 1;
      }
    }
  }
  if (q[5] <= 0) /* Pz */
    return (0.0);
  return (dZ);
}

#define FID_MEDIAN 0
#define FID_MINIMUM 1
#define FID_MAXIMUM 2
#define FID_AVERAGE 3
#define FID_FIRST 4
#define FID_LIGHT 5
#define N_KNOWN_MODES 6
static char *known_mode[N_KNOWN_MODES] = {
  (char *)"median", (char *)"minimum", (char *)"maximum", (char *)"average", (char *)"first", (char *)"light"};
#if USE_MPI
/* We need a separate array to hold the coordinates for the fiducial particle in the parallel version */
double best_particle[MAX_PROPERTIES_PER_PARTICLE];
#endif

double *select_fiducial(double **part, int64_t n_part, char *var_mode_in) {
  int64_t i;
  long i_best = 0, i_var, fid_mode;
  double value, best_value, sum;
  char *var, *mode, *var_mode, *ptr;
#if USE_MPI
  int64_t n_part_total;
#endif

  log_entry((char *)"select_fiducial");

  fid_mode = FID_AVERAGE;
  i_var = -1; /* t, p average */
  cp_str(&var_mode, var_mode_in == NULL ? (char *)(char *)"average" : var_mode_in);

#if !USE_MPI
  if (n_part <= 1) {
    log_exit((char *)"select_fiducial");
    return (part[0]);
  }
#else
  if (distributedBeam)
    MPI_Allreduce(&n_part, &n_part_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (strlen(var_mode) != 0) {
    if ((ptr = strchr(var_mode, ',')) && (var = get_token(var_mode))) {
      if (*var != 't' && *var != 'p')
        bombElegant((char *)"no known variable listed for fiducial--must be 't' or 'p'\n", NULL);
      if (*var == 't')
        i_var = 4;
      else
        i_var = 5;
    }
    if ((mode = get_token(var_mode))) {
      if ((fid_mode = match_string(mode, known_mode, N_KNOWN_MODES, 0)) < 0) {
        fputs((char *)"error: no known mode listed for fiducialization--must be one of:\n", stdout);
        for (i = 0; i < N_KNOWN_MODES; i++)
          printf((char *)"    %s\n", known_mode[i]);
        fflush(stdout);
        exitElegant(1);
      }
    }
  }
  if (i_var == -1 && fid_mode != FID_AVERAGE) {
    printf((char *)"unless you use average mode for fiducialization, you must specify t or p coordinate.\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (i_var == -1) {
    double deltaAve, tAve;
    static double coord[6];
    for (i = deltaAve = tAve = 0; i < n_part; i++) {
      deltaAve += part[i][5];
      tAve += part[i][4];
    }
#if !USE_MPI
    deltaAve /= n_part;
    tAve /= n_part;
#else
    if (distributedBeam) {
      double deltaAve_total, tAve_total;

      MPI_Allreduce(&deltaAve, &deltaAve_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&tAve, &tAve_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      deltaAve = deltaAve_total / n_part_total;
      tAve = tAve_total / n_part_total;
    } else {
      deltaAve /= n_part;
      tAve /= n_part;
    }
#endif
    coord[0] = coord[1] = coord[2] = coord[3] = 0;
    coord[4] = tAve;
    coord[5] = deltaAve;
    return coord;
  } else {
    switch (fid_mode) {
    case FID_MEDIAN:
#if !USE_MPI
      if ((i_best = find_median_of_row(&best_value, part, i_var, n_part)) < 0)
        bombElegant((char *)"error: computation of median failed (select_fiducial)", NULL);
#else
      printf((char *)"The median fiducial mode is not available in Pelegant.\n");
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 2);
#endif
      break;
    case FID_MINIMUM:
      i_best = 0;
      best_value = DBL_MAX;
      for (i = 0; i < n_part; i++) {
        if (best_value > (value = part[i][i_var])) {
          best_value = value;
          i_best = i;
        }
      }
#if USE_MPI
      if (distributedBeam) {
        struct {
          double val;
          int rank;
        } in, out;
        in.rank = myid;
        in.val = best_value;
        /* find the global best value and its location, i.e., on which processor */
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (in.rank == out.rank)
          memcpy(best_particle, part[i_best], sizeof(*best_particle) * totalPropertiesPerParticle);
        MPI_Bcast(best_particle, totalPropertiesPerParticle, MPI_DOUBLE, out.rank, MPI_COMM_WORLD);
        log_exit((char *)"select_fiducial");
        return (best_particle);
      }
#endif
      break;
    case FID_MAXIMUM:
      i_best = 0;
      best_value = -DBL_MAX;
      for (i = 0; i < n_part; i++) {
        if (best_value < (value = part[i][i_var])) {
          best_value = value;
          i_best = i;
        }
      }
#if USE_MPI
      if (distributedBeam) {
        struct {
          double val;
          int rank;
        } in, out;
        in.rank = myid;
        in.val = best_value;
        /* find the global best value and its location, i.e., on which processor */
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (in.rank == out.rank)
          memcpy(best_particle, part[i_best], sizeof(*best_particle) * totalPropertiesPerParticle);
        MPI_Bcast(best_particle, totalPropertiesPerParticle, MPI_DOUBLE, out.rank, MPI_COMM_WORLD);
        log_exit((char *)"select_fiducial");
        return (best_particle);
      }
#endif
      break;
    case FID_AVERAGE:
      for (i = sum = 0; i < n_part; i++)
        sum += part[i][i_var];
#if !USE_MPI
      sum /= n_part;
#else
      if (distributedBeam) {
        double sum_total;

        MPI_Allreduce(&sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sum = sum_total / n_part_total;
      } else
        sum /= n_part;
#endif
      best_value = DBL_MAX;
      for (i = 0; i < n_part; i++) {
        if (best_value > (value = fabs(sum - part[i][i_var]))) {
          best_value = value;
          i_best = i;
        }
      }
#if USE_MPI
      if (distributedBeam) {
        struct {
          double val;
          int rank;
        } in, out;
        in.rank = myid;
        in.val = best_value;
        /* find the global best value and its location, i.e., on which processor */
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (in.rank == out.rank)
          memcpy(best_particle, part[i_best], sizeof(*best_particle) * totalPropertiesPerParticle);
        MPI_Bcast(best_particle, totalPropertiesPerParticle, MPI_DOUBLE, out.rank, MPI_COMM_WORLD);
        log_exit((char *)"select_fiducial");
        return (best_particle);
      }
#endif
      break;
    case FID_FIRST:
#if !USE_MPI
      i_best = 0;
#else
      if (distributedBeam) {
        if (myid == 1)
          memcpy(best_particle, part[0], sizeof(*best_particle) * totalPropertiesPerParticle);
        MPI_Bcast(best_particle, totalPropertiesPerParticle, MPI_DOUBLE, 1, MPI_COMM_WORLD);
        log_exit((char *)"select_fiducial");
        return (best_particle);
      } else
        i_best = 0;
#endif
      break;
    case FID_LIGHT:
      /* special case--return NULL pointer to indicate that phasing is to v=c */
      log_exit((char *)"select_fiducial");
      return (NULL);
    default:
      bombElegant((char *)"apparent programming error in select_fiducial", NULL);
      break;
    }
    if (i_best < 0 || i_best > n_part)
      bombElegant((char *)"fiducial particle selection returned invalid value!", NULL);
  }

  log_exit((char *)"select_fiducial");
  return (part[i_best]);
}

void select_integrator(char *desired_method) {
  long i;

  log_entry((char *)"select_integrator");

#if defined(DEBUG)
  printf((char *)"select_integrator called with pointer %lx\n", (long)(desired_method));
  fflush(stdout);
  printf((char *)"this translates into string %s\n", desired_method);
  fflush(stdout);
#endif
  switch (integratorCode = match_string(desired_method, method, N_METHODS, 0)) {
  case RUNGE_KUTTA:
    printWarningForTracking((char *)"\"non-adaptive runge-kutta\" integrator is recommended.", (char *)"Adaptive Runge-Kutta was selected.");
    break;
  case BULIRSCH_STOER:
    printWarningForTracking((char *)"\"non-adaptive runge-kutta\" integrator is recommended.", (char *)"Bulirsch-Stoer was selected.");
    break;
  case NA_RUNGE_KUTTA:
    break;
  case MODIFIED_MIDPOINT:
    break;
  default:
    printf((char *)"error: unknown integration method %s requested.\n", desired_method);
    printf((char *)"Available methods are:\n");
    for (i = 0; i < N_METHODS; i++)
      printf((char *)"    %s\n", method[i]);
    fflush(stdout);
    exitElegant(1);
    break;
  }
  log_exit((char *)"select_integrator");
}

void setupRftmEz0FromFile(RFTMEZ0 *rftmEz0, double frequency, double length) {
  SDDS_DATASET SDDSin;
  long i;
  double *z;
  double k, Ez, EzMax;

  if (!rftmEz0)
    bombElegant((char *)"setupRftmEz0FromFile: NULL pointer passed.", NULL);

  if (rftmEz0->initialized)
    return;

  /* verify input parameters */
  if (!rftmEz0->inputFile || !rftmEz0->zColumn || !rftmEz0->EzColumn)
    bombElegant((char *)"RFTMEZ0 must have INPUTFILE, ZCOLUMN, and EZCOLUMN specified.", NULL);
  if (rftmEz0->radial_order > 1)
    bombElegant((char *)"RFTMEZ0 restricted to radial_order<=1 at present", NULL);

  /* read (z, Ez) data from the file  */
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, rftmEz0->inputFile) || SDDS_ReadPage(&SDDSin) != 1) {
    fprintf(stderr, (char *)"Error: unable to open or read RFTMEZ0 file %s\n",
            rftmEz0->inputFile);
    exitElegant(1);
  }
  if ((rftmEz0->nz = SDDS_RowCount(&SDDSin)) < 0 || rftmEz0->nz < 2) {
    fprintf(stderr, (char *)"Error: no data or insufficient data in RFTMEZ0 file %s\n",
            rftmEz0->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, rftmEz0->zColumn, (char *)"m", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 file %s.  Check existence, type, and units.\n",
            rftmEz0->zColumn, rftmEz0->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, rftmEz0->EzColumn, NULL, SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 file %s.  Check existence and type.\n",
            rftmEz0->EzColumn, rftmEz0->inputFile);
    exitElegant(1);
  }
  if (!(z = SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->zColumn)) ||
      !(rftmEz0->Ez = SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->EzColumn))) {
    fprintf(stderr, (char *)"Error: problem retrieving RFTMEZ0 data from SDDS structure for file %s\n",
            rftmEz0->inputFile);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  rftmEz0->z0 = z[0];

  if (!SDDS_Terminate(&SDDSin)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!(rftmEz0->dEzdZ = (double *)SDDS_Malloc(sizeof(*(rftmEz0->dEzdZ)) * rftmEz0->nz))) {

    fprintf(stderr, (char *)"Error: memory allocation failure setting up RFTMEZ0 file %s\n",
            rftmEz0->inputFile);
    exitElegant(1);
  }

  /* compute dEz/dz for use in computing Er(z) */
  if (!analyzeSpacing(z, rftmEz0->nz, &rftmEz0->dz, stderr)) {
    fprintf(stderr, (char *)"Error: problem with z points from RFTMEZ0 file %s\n",
            rftmEz0->inputFile);
    exitElegant(1);
  }

  if (fabs((length - (z[rftmEz0->nz - 1] - z[0])) / rftmEz0->dz) > 1e-4) {
    fprintf(stderr, (char *)"Error: declared length (%le) and length from fields (%le) from RFTMEZ0 file %s do not agree to 1/10^4 tolerance\n",
            length, z[rftmEz0->nz - 1] - z[0], rftmEz0->inputFile);
    exitElegant(1);
  }
  free(z);

  /* find maximum Ez */
  EzMax = 0;
  for (i = 0; i < rftmEz0->nz; i++) {
    if ((Ez = fabs(rftmEz0->Ez[i])) > EzMax)
      EzMax = Ez;
  }

  /* perform scaling and normalization (scaling by Ez_peak occurs when derivatives are computed) */
  k = (PIx2 * frequency) / c_mks;
  rftmEz0->dZ = rftmEz0->dz * k;
  fflush(stdout);
  if (EzMax)
    for (i = 0; i < rftmEz0->nz; i++)
      rftmEz0->Ez[i] *= particleCharge / (particleMass * c_mks * PIx2 * frequency) / EzMax;
  else
    for (i = 0; i < rftmEz0->nz; i++)
      rftmEz0->Ez[i] = 0;

  /* take derivative except at endpoints */
  for (i = 1; i < rftmEz0->nz - 1; i++)
    rftmEz0->dEzdZ[i] = (rftmEz0->Ez[i + 1] - rftmEz0->Ez[i - 1]) / (2 * rftmEz0->dZ);

  /* endpoints */
  rftmEz0->dEzdZ[0] = (rftmEz0->Ez[1] - rftmEz0->Ez[0]) / rftmEz0->dZ;
  rftmEz0->dEzdZ[rftmEz0->nz - 1] =
    (rftmEz0->Ez[rftmEz0->nz - 1] - rftmEz0->Ez[rftmEz0->nz - 2]) / rftmEz0->dZ;
}

void setupRftmEz0SolenoidFromFile(RFTMEZ0 *rftmEz0, double length, double k) {
  SDDS_DATASET SDDSin;
  double *z, *r, dz, dr, *rTemp, z0 = 0.0;
  long page, ir, iz;

  if (rftmEz0->initialized)
    return;

  rftmEz0->BrSol = rftmEz0->BzSol = NULL;

  /* check for solenoid input file */
  if (!rftmEz0->solenoidFile)
    return;

  if (!rftmEz0->solenoid_zColumn || !rftmEz0->solenoidBzColumn)
    SDDS_Bomb((char *)"missing column name for solenoid for RFTMEZ0 element");
  if (rftmEz0->solenoidBrColumn && !rftmEz0->solenoid_rColumn)
    SDDS_Bomb((char *)"missing column name for solenoid for RFTMEZ0 element---supply r column with Br column");

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, rftmEz0->solenoidFile)) {
    fprintf(stderr, (char *)"Error: unable to open or read RFTMEZ0 solenoid file %s\n",
            rftmEz0->solenoidFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, rftmEz0->solenoid_zColumn, (char *)"m", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 solenoid file %s.  Check existence, type, and units.\n",
            rftmEz0->solenoid_zColumn, rftmEz0->solenoidFile);
    exitElegant(1);
  }
  if (rftmEz0->solenoid_rColumn &&
      SDDS_CheckColumn(&SDDSin, rftmEz0->solenoid_rColumn, (char *)"m", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 solenoid file %s.  Check existence, type, and units.\n",
            rftmEz0->solenoid_rColumn, rftmEz0->solenoidFile);
    exitElegant(1);
  }

  if (SDDS_CheckColumn(&SDDSin, rftmEz0->solenoidBzColumn, (char *)"T", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 solenoid file %s.  Check existence, type, and units.\n",
            rftmEz0->solenoidBzColumn, rftmEz0->solenoidFile);
    exitElegant(1);
  }
  if (rftmEz0->solenoidBrColumn &&
      SDDS_CheckColumn(&SDDSin, rftmEz0->solenoidBrColumn, (char *)"T", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in RFTMEZ0 solenoid file %s.  Check existence, type, and units.\n",
            rftmEz0->solenoidBzColumn, rftmEz0->solenoidFile);
    exitElegant(1);
  }

  z = NULL;
  r = NULL;
  while ((page = SDDS_ReadPage(&SDDSin)) > 0) {
    if (page == 1) {
      if ((rftmEz0->nzSol = SDDS_RowCount(&SDDSin)) < 0 || rftmEz0->nzSol < 2) {
        fprintf(stderr, (char *)"Error: no data or insufficient data in RFTMEZ0 solenoid file %s\n",
                rftmEz0->solenoidFile);
        exitElegant(1);
      }
      if (!(z = SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->solenoid_zColumn))) {
        fprintf(stderr, (char *)"Error: problem getting z data from RFTMEZ0 solenoid file %s\n",
                rftmEz0->solenoidFile);
        exitElegant(1);
      }
      if (!analyzeSpacing(z, rftmEz0->nzSol, &dz, stderr)) {
        fprintf(stderr, (char *)"Problem with z spacing of solenoid data from RFTMEZ0 file %s (page %ld)\n",
                rftmEz0->solenoidFile, page);
        exitElegant(1);
      }
      z0 = z[0];
      if (fabs((length - (z[rftmEz0->nzSol - 1] - z[0])) / dz) > 1e-4) {
        fprintf(stderr, (char *)"Error: declared length and length from fields from RFTMEZ0 solenoid file %s do not agree to 1/10^4 tolerance\n",
                rftmEz0->solenoidFile);

        exitElegant(1);
      }
      free(z);
    } else {
      if (!rftmEz0->solenoidBrColumn) {
        fprintf(stderr, (char *)"Error: multi-page file for RFTMEZ0 solenoid, but no BrColumn or rColumn\n");
        exitElegant(1);
      }
      if (rftmEz0->nzSol != SDDS_RowCount(&SDDSin)) {
        fprintf(stderr, (char *)"Error: page %ld of RFTMEZ0 file %s has only %ld rows (%ld expected)\n",
                page, rftmEz0->solenoidFile, (long)SDDS_RowCount(&SDDSin), rftmEz0->nzSol);
        exitElegant(1);
      }
    }

    if (rftmEz0->solenoidBrColumn) {
      if (!(rTemp = SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->solenoid_rColumn))) {
        fprintf(stderr, (char *)"Error: problem getting r data from RFTMEZ0 solenoid file %s\n",
                rftmEz0->solenoidFile);
        exitElegant(1);
      }
      if (!(r = (double *)SDDS_Realloc(r, sizeof(*r) * page)))
        SDDS_Bomb((char *)"memory allocation failure (setupRftmEz0SolenoidFromFile)");
      r[page - 1] = rTemp[0];
      free(rTemp);
      if (!(rftmEz0->BrSol = (double **)SDDS_Realloc(rftmEz0->BrSol, sizeof(*rftmEz0->BrSol) * page)))
        SDDS_Bomb((char *)"memory allocation failure (setupRftmEz0SolenoidFromFile)");
      if (!(rftmEz0->BrSol[page - 1] =
            SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->solenoidBrColumn))) {
        fprintf(stderr, (char *)"Error: problem getting field data from RFTMEZ0 solenoid file %s\n",
                rftmEz0->solenoidFile);
        exitElegant(1);
      }
    }

    if (!(rftmEz0->BzSol = (double **)SDDS_Realloc(rftmEz0->BzSol, sizeof(*rftmEz0->BzSol) * page)))
      SDDS_Bomb((char *)"memory allocation failure (setupRftmEz0SolenoidFromFile)");
    if (!(rftmEz0->BzSol[page - 1] =
          SDDS_GetColumnInDoubles(&SDDSin, rftmEz0->solenoidBzColumn))) {
      fprintf(stderr, (char *)"Error: problem getting field data from RFTMEZ0 solenoid file %s\n",
              rftmEz0->solenoidFile);
      exitElegant(1);
    }
    rftmEz0->nrSol = page;
  }

  SDDS_Terminate(&SDDSin);

  if (rftmEz0->solenoid_rColumn) {
    if (!analyzeSpacing(r, rftmEz0->nrSol, &dr, stderr)) {
      fprintf(stderr, (char *)"Problem with r spacing of solenoid data from RFTMEZ0 file %s\nr values are:\n",
              rftmEz0->solenoidFile);
      for (ir = 0; ir < rftmEz0->nrSol; ir++)
        fprintf(stderr, (char *)"%le\n", r[ir]);
      exitElegant(1);
    }
    free(r);
  } else
    dr = 0;

  rftmEz0->dRSol = k * dr;
  rftmEz0->dZSol = k * dz;
  rftmEz0->Z0Sol = k * (z0 - rftmEz0->z0);

  /* perform scaling */
  for (ir = 0; ir < rftmEz0->nrSol; ir++)
    for (iz = 0; iz < rftmEz0->nzSol; iz++)
      rftmEz0->BzSol[ir][iz] *= particleCharge / (particleMass * k * c_mks);
  if (rftmEz0->BrSol)
    for (ir = 0; ir < rftmEz0->nrSol; ir++)
      for (iz = 0; iz < rftmEz0->nzSol; iz++)
        rftmEz0->BrSol[ir][iz] *= particleCharge / (particleMass * k * c_mks);

  rftmEz0->dBzdZSol = NULL;
  if (rftmEz0->nrSol == 1) {
    /* take derivative for off-axis expansion */

    if (!(rftmEz0->dBzdZSol = (double *)SDDS_Malloc(sizeof(*(rftmEz0->dBzdZSol)) * rftmEz0->nzSol))) {
      fprintf(stderr, (char *)"Error: memory allocation failure setting up RFTMEZ0 file %s\n",
              rftmEz0->solenoidFile);
      exitElegant(1);
    }
    /* take derivative except at endpoints */
    for (iz = 1; iz < rftmEz0->nzSol - 1; iz++)
      rftmEz0->dBzdZSol[iz] = (rftmEz0->BzSol[0][iz + 1] - rftmEz0->BzSol[0][iz - 1]) / (2 * rftmEz0->dZSol);

    /* endpoints */
    rftmEz0->dBzdZSol[0] =
      (rftmEz0->BzSol[0][1] - rftmEz0->BzSol[0][0]) / rftmEz0->dZSol;
    rftmEz0->dBzdZSol[rftmEz0->nzSol - 1] =
      (rftmEz0->BzSol[0][rftmEz0->nzSol - 1] - rftmEz0->BzSol[0][rftmEz0->nzSol - 2]) / rftmEz0->dZSol;
  }
}

long analyzeSpacing(double *z, long nz, double *dzReturn, FILE *fpError) {
  double dzMin, dzMax, dz;
  long i;

  if (nz < 3) {
    if (fpError)
      fprintf(fpError, (char *)"Problem with point spacing: less than 3 points\n");
    return 0;
  }

  dzMin = dzMax = z[1] - z[0];
  for (i = 2; i < nz; i++) {
    if ((dz = z[i] - z[i - 1]) < dzMin)
      dzMin = dz;
    if (dz > dzMax)
      dzMax = dz;
  }
  if (dzMin <= 0 || dzMax <= 0) {
    if (fpError)
      fprintf(fpError, (char *)"Error: points are not monotonically increasing\n");
    return 0;
  }
  if (fabs(1 - dzMin / dzMax) > 0.0001) {
    if (fpError)
      fprintf(fpError, (char *)"Error: spacing of points is not uniform (0.01%% tolerance)\n");
    return 0;
  }
  *dzReturn = (z[nz - 1] - z[0]) / (nz - 1.0);
  return 1;
}

void setupMapSolenoidFromFile(MAP_SOLENOID *mapSol, double length) {
  SDDS_DATASET SDDSin;
  double *z, *r, dz, dr, *rTemp;
  long page, ir, iz;

  if (mapSol->initialized)
    return;
  mapSol->initialized = 1;

  mapSol->Br = mapSol->Bz = NULL;

  /* check for solenoid input file */
  if (!mapSol->inputFile)
    SDDS_Bomb((char *)"no input file given for MAPSOLENOID element");

  if (!mapSol->zColumn || !mapSol->rColumn ||
      !mapSol->BzColumn || !mapSol->BrColumn)
    SDDS_Bomb((char *)"missing column name for solenoid for MAPSOLENOID element");

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, mapSol->inputFile)) {
    fprintf(stderr, (char *)"Error: unable to open or read MAPSOLENOID file %s\n",
            mapSol->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, mapSol->zColumn, (char *)"m", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in MAPSOLENOID file %s.  Check existence, type, and units.\n",
            mapSol->zColumn, mapSol->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, mapSol->rColumn, (char *)"m", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in MAPSOLENOID file %s.  Check existence, type, and units.\n",
            mapSol->rColumn, mapSol->inputFile);
    exitElegant(1);
  }

  if (SDDS_CheckColumn(&SDDSin, mapSol->BzColumn, (char *)"T", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in MAPSOLENOID file %s.  Check existence, type, and units.\n",
            mapSol->BzColumn, mapSol->inputFile);
    exitElegant(1);
  }
  if (SDDS_CheckColumn(&SDDSin, mapSol->BrColumn, (char *)"T", SDDS_ANY_FLOATING_TYPE,
                       stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, (char *)"Error: problem with column %s in MAPSOLENOID file %s.  Check existence, type, and units.\n",
            mapSol->BzColumn, mapSol->inputFile);
    exitElegant(1);
  }

  z = NULL;
  r = NULL;
  while ((page = SDDS_ReadPage(&SDDSin)) > 0) {
    if (page == 1) {
      if ((mapSol->nz = SDDS_RowCount(&SDDSin)) < 0 || mapSol->nz < 2) {
        fprintf(stderr, (char *)"Error: no data or insufficient data in MAPSOLENOID file %s\n",
                mapSol->inputFile);
        exitElegant(1);
      }
      if (!(z = SDDS_GetColumnInDoubles(&SDDSin, mapSol->zColumn))) {
        fprintf(stderr, (char *)"Error: problem getting z data from MAPSOLENOID file %s\n",
                mapSol->inputFile);
        exitElegant(1);
      }
      if (!analyzeSpacing(z, mapSol->nz, &dz, stderr)) {
        fprintf(stderr, (char *)"Problem with z spacing of solenoid data from MAPSOLENOID file %s (page %ld)\n",
                mapSol->inputFile, page);
        exitElegant(1);
      }
      if (fabs((length - (z[mapSol->nz - 1] - z[0])) / dz) > 1e-4) {
        fprintf(stderr, (char *)"Error: declared length (%le) and length from fields (%le) from MAP_SOLENOID file %s do not agree to 1/10^4 tolerance (in units of dz=%le)\nDiscrepancy is %le\n",
                length, z[mapSol->nz - 1] - z[0], mapSol->inputFile, dz,
                fabs((length - (z[mapSol->nz - 1] - z[0])) / dz));
        exitElegant(1);
      }
      free(z);
    } else {
      if (mapSol->nz != SDDS_RowCount(&SDDSin)) {
        fprintf(stderr, (char *)"Error: page %ld of MAPSOLENOID file %s has only %ld  rows (%ld expected)\n",
                page, mapSol->inputFile, (long)SDDS_RowCount(&SDDSin), mapSol->nz);
        exitElegant(1);
      }
    }

    if (!(rTemp = SDDS_GetColumnInDoubles(&SDDSin, mapSol->rColumn))) {
      fprintf(stderr, (char *)"Error: problem getting r data from MAPSOLENOID file %s\n",
              mapSol->inputFile);
      exitElegant(1);
    }
    if (!(r = (double *)SDDS_Realloc(r, sizeof(*r) * page)))
      SDDS_Bomb((char *)"memory allocation failure (setupmapSolSolenoidFromFile)");
    r[page - 1] = rTemp[0];
    free(rTemp);

    if (!(mapSol->Bz = (double **)SDDS_Realloc(mapSol->Bz, sizeof(*mapSol->Bz) * page)) ||
        !(mapSol->Br = (double **)SDDS_Realloc(mapSol->Br, sizeof(*mapSol->Br) * page)))
      SDDS_Bomb((char *)"memory allocation failure (setupmapSolSolenoidFromFile)");
    if (!(mapSol->Bz[page - 1] =
          SDDS_GetColumnInDoubles(&SDDSin, mapSol->BzColumn)) ||
        !(mapSol->Br[page - 1] =
          SDDS_GetColumnInDoubles(&SDDSin, mapSol->BrColumn))) {
      fprintf(stderr, (char *)"Error: problem getting field data from MAPSOLENOID file %s\n",
              mapSol->inputFile);
      exitElegant(1);
    }
    mapSol->nr = page;
  }

  SDDS_Terminate(&SDDSin);

  if (!analyzeSpacing(r, mapSol->nr, &dr, stderr)) {
    fprintf(stderr, (char *)"Problem with r spacing of solenoid data from MAPSOLENOID file %s\nr values are:\n",
            mapSol->inputFile);
    for (ir = 0; ir < mapSol->nr; ir++)
      fprintf(stderr, (char *)"%le\n", r[ir]);
    exitElegant(1);
  }
  free(r);

  /* perform scaling */
  for (ir = 0; ir < mapSol->nr; ir++)
    for (iz = 0; iz < mapSol->nz; iz++) {
      mapSol->Br[ir][iz] *= particleCharge / (particleMass * c_mks);
      mapSol->Bz[ir][iz] *= particleCharge / (particleMass * c_mks);
    }

  mapSol->dr = dr;
  mapSol->dz = dz;
}

void makeRftmEz0FieldTestFile(RFTMEZ0 *rftmEz0) {
  double dZ, dX, dY, scale, ERFscale, BRFscale, k;
  long iz, nz;
  double E[3], BOverGamma[3], BOverGammaSol[3];
  SDDS_DATASET SDDSout;
  TRACKING_CONTEXT context;
  char *filename;
  getTrackingContext(&context);

  field_global = rftmEz0;
  if (!rftmEz0->fieldTestFile)
    return;
  dZ = MIN(rftmEz0->dZ, rftmEz0->dZSol) / 2;
  nz = 2 * (rftmEz0->nz > rftmEz0->nzSol ? rftmEz0->nz : rftmEz0->nzSol);
  filename = compose_filename(rftmEz0->fieldTestFile, context.rootname);
  if (!SDDS_InitializeOutputElegant(&SDDSout, SDDS_BINARY, 0, NULL, NULL, filename) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"z", (char *)"m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"ExRFOverr", (char *)"V/m/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"EyRFOverr", (char *)"V/m/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"EzRF", (char *)"V/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"BxRFOverr", (char *)"T/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"ByRFOverr", (char *)"T/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"BzRF", (char *)"T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"BxSolOverr", (char *)"T/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"BySolOverr", (char *)"T/m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, (char *)"BzSol", (char *)"T", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, nz)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  free(filename);
  dX = dY = 1e-3 * rftmEz0->k / sqrt(2.0);
  ERFscale = 1 / (sin(PI / 2.0) * particleCharge / (particleMass * c_mks * PIx2 * rftmEz0->frequency));
  BRFscale = 1 / (cos(PI / 2.0) * particleCharge / (particleMass * PIx2 * rftmEz0->frequency));
  k = PIx2 * rftmEz0->frequency / c_mks;
  scale = particleMass * k * c_mks / particleCharge;
  for (iz = 0; iz < nz; iz++) {
    computeFields_rftmEz0(E, BOverGamma, BOverGammaSol,
                          dX, dY, iz * dZ,
                          PI / 2.0, 1.0);
    if (!SDDS_SetRowValues(&SDDSout,
                           SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iz,
                           (char *)"z", iz * dZ / rftmEz0->k,
                           (char *)"ExRFOverr", E[0] / 1e-3 * ERFscale,
                           (char *)"EyRFOverr", E[1] / 1e-3 * ERFscale,
                           (char *)"EzRF", E[2] * ERFscale,
                           (char *)"BxRFOverr", BOverGamma[0] / 1e-3 * BRFscale,
                           (char *)"ByRFOverr", BOverGamma[1] / 1e-3 * BRFscale,
                           (char *)"BzRF", BOverGamma[2] * BRFscale,
                           (char *)"BxSolOverr", BOverGammaSol[0] / 1e-3 * scale,
                           (char *)"BySolOverr", BOverGammaSol[1] / 1e-3 * scale,
                           (char *)"BzSol", BOverGammaSol[2] * scale, NULL)) {
      SDDS_SetError((char *)"Problem setting SDDS row values (makeRftmEz0FieldTestFile)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout)) {
    SDDS_SetError((char *)"Problem writing or terminating file (makeRftmEz0FieldTestFile)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
}

void laserModulatorUndulatorField(double *FieldB, double Bfactor, double kux, double kuy, double kuz,
                                  double a, double ku, double x, double tguCompFactor);
void computeLaserField(double *Ef, double *Bf, double phase, double Ef0, double ZR,
                       double k, double w0, double x, double y, double dz,
                       double *tValue, double *amplitudeValue, long profilePoints, double tForProfile,
                       long m, long n, double laserTilt);

#ifdef DEBUG
FILE *fppu = NULL;
#endif

void derivatives_laserModulator(double *qp, double *q, double tau) {
  LSRMDLTR *lsrMdltr;
  double gamma, *P, *Pp;
  double BOverGamma[3] = {0, 0, 0}, E[3] = {0, 0, 0}, Blaser[3] = {0, 0, 0};
  double Bscale;
  double x, y, z, factor_inner_scope, Bfactor, kuz, kuy, kux;
  long i, poleNumber;

#ifdef DEBUG
  if (!fppu) {
    fppu = fopen((char *)"lsrMdltr.sdds", (char *)"w");
    fprintf(fppu, (char *)"SDDS1\n");
    fprintf(fppu, (char *)"&column name=tau type=double &end\n");
    fprintf(fppu, (char *)"&column name=t type=double units=s &end\n");
    fprintf(fppu, (char *)"&column name=phi type=double &end\n");
    fprintf(fppu, (char *)"&column name=x type=double units=m &end\n");
    fprintf(fppu, (char *)"&column name=y type=double units=m &end\n");

    fprintf(fppu, (char *)"&column name=z type=double units=m &end\n");
    fprintf(fppu, (char *)"&column name=Px type=double &end\n");
    fprintf(fppu, (char *)"&column name=Py type=double &end\n");
    fprintf(fppu, (char *)"&column name=Pz type=double &end\n");
    fprintf(fppu, (char *)"&column name=Ex type=double &end\n");
    fprintf(fppu, (char *)"&column name=By type=double &end\n");
    fprintf(fppu, (char *)"&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  derivCalls++;
  P = q + 3;
  gamma = sqrt(sqr(P[0]) + sqr(P[1]) + sqr(P[2]) + 1);

  /* X' = Px/gamma, etc. */
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;

  lsrMdltr = (LSRMDLTR *)field_global;

  x = q[0] / lsrMdltr->k;
  y = q[1] / lsrMdltr->k;
  z = q[2] / lsrMdltr->k;
  if (x < xMinSeen)
    xMinSeen = x;
  if (x > xMaxSeen)
    xMaxSeen = x;

  poleNumber = z / (lsrMdltr->length / (2 * lsrMdltr->periods)) + 0.5;
  if (poleNumber > 2 && poleNumber < (2 * lsrMdltr->periods - 2))
    factor_inner_scope = 1;
  else if (poleNumber == 0 || poleNumber == 2 * lsrMdltr->periods)
    factor_inner_scope = lsrMdltr->poleFactor1;
  else if (poleNumber == 1 || poleNumber == (2 * lsrMdltr->periods - 1))
    factor_inner_scope = lsrMdltr->poleFactor2;
  else
    factor_inner_scope = lsrMdltr->poleFactor3;

  Bfactor = factor_inner_scope * lsrMdltr->Bu;
  kuz = lsrMdltr->ku * z;
  kuy = lsrMdltr->ku * y;
  kux = lsrMdltr->ku * x;
  /*
    if (lsrMdltr->helical==0) {
    if (lsrMdltr->fieldCode==LSRMDLTR_IDEAL) {
    BOverGamma[1] = Bfactor*cos(kuz);
    BOverGamma[2] = 0;
    } else if (lsrMdltr->fieldCode==LSRMDLTR_EXACT) {
    BOverGamma[1] = Bfactor*cos(kuz)*cosh(kuy);
    BOverGamma[2] = -Bfactor*sin(kuz)*sinh(kuy);
    } else {
    BOverGamma[1] = Bfactor*cos(kuz)*(1+sqr(kuy)/2);
    BOverGamma[2] = -Bfactor*sin(kuz)*kuy;
    }
    } else {
    if (lsrMdltr->fieldCode==LSRMDLTR_IDEAL) {
    BOverGamma[0] = -Bfactor*sin(kuz);
    BOverGamma[1] = Bfactor*cos(kuz);
    BOverGamma[2] = 0;
    } else if (lsrMdltr->fieldCode==LSRMDLTR_EXACT) {
    BOverGamma[0] = -Bfactor*sin(kuz)*cosh(kux);
    BOverGamma[1] = Bfactor*cos(kuz)*cosh(kuy);
    BOverGamma[2] = -Bfactor*sin(kuz)*sinh(kuy) -Bfactor*cos(kuz)*sinh(kux);
    } else {
    BOverGamma[0] = -Bfactor*sin(kuz)*(1+sqr(kux)/2);
    BOverGamma[1] = Bfactor*cos(kuz)*(1+sqr(kuy)/2);
    BOverGamma[2] = -Bfactor*sin(kuz)*kuy -Bfactor*cos(kuz)*kux;
    }
    }
  */
  laserModulatorUndulatorField(BOverGamma, Bfactor, kux, kuy, kuz, lsrMdltr->tguGradient, lsrMdltr->ku, x,
                               lsrMdltr->tguCompFactor);
  Bscale = lsrMdltr->Bscale / gamma;
  for (i = 0; i < 3; i++)
    BOverGamma[i] *= Bscale;

  if (lsrMdltr->Ef0Laser > 0 && lsrMdltr->laserW0 > 0) {
    computeLaserField(E, Blaser, -tau + q[2] - Z_center + lsrMdltr->laserPhase,
                      lsrMdltr->Ef0Laser, lsrMdltr->ZRayleigh, lsrMdltr->k, lsrMdltr->laserW0,
                      x - lsrMdltr->laserX0, y - lsrMdltr->laserY0, z - lsrMdltr->laserZ0 - Z_center / lsrMdltr->k,
                      lsrMdltr->timeValue, lsrMdltr->amplitudeValue, lsrMdltr->tProfilePoints,
                      (tau / lsrMdltr->omega - lsrMdltr->t0) - (q[2] - Z_center) / (lsrMdltr->k * c_mks),
                      lsrMdltr->laserM, lsrMdltr->laserN, lsrMdltr->laserTilt);
    for (i = 0; i < 3; i++) {
      E[i] *= lsrMdltr->Escale;
      BOverGamma[i] += Blaser[i] * Bscale;
    }
  }

  /* (Px,Py,Pz)' = (Ex,Ey,Ez) + (Px,Py,Pz)x(Bx,By,Bz)/gamma */
  Pp = qp + 3;
  Pp[0] = -(E[0] + P[1] * BOverGamma[2] - P[2] * BOverGamma[1]);
  Pp[1] = -(E[1] + P[2] * BOverGamma[0] - P[0] * BOverGamma[2]);
  Pp[2] = -(E[2] + P[0] * BOverGamma[1] - P[1] * BOverGamma[0]);

#ifdef DEBUG
  fprintf(fppu, (char *)"%e %e %e %e %e %e %e %e %e %e %e\n", tau, tau / lsrMdltr->omega,
          lsrMdltr->k * z - tau, x, y, z, P[0], P[1], P[2], E[0], BOverGamma[1] * gamma);
#endif
}

void stochastic_laserModulator(double *q, double tau, double h) {
  LSRMDLTR *lsrMdltr;
  double gamma, *P, p;
  double B[3] = {0, 0, 0};
  double x, y, z, Bfactor, kuz, kuy, kux;
  double B2, delta0, delta1, irho2, factor;
  long i, poleNumber;

  lsrMdltr = (LSRMDLTR *)field_global;

  if (!lsrMdltr->synchRad && !lsrMdltr->isr)
    return;

  P = q + 3;
  p = sqr(P[0]) + sqr(P[1]) + sqr(P[2]);
  gamma = sqrt(p + 1);
  p = sqrt(p);

  x = (q[0] - X_offset) / lsrMdltr->k;
  y = q[1] / lsrMdltr->k;
  z = q[2] / lsrMdltr->k;
  h = h / lsrMdltr->k;

  poleNumber = z / (lsrMdltr->length / (2 * lsrMdltr->periods)) + 0.5;
  if (poleNumber > 2 && poleNumber < (2 * lsrMdltr->periods - 2))
    factor = 1;
  else if (poleNumber == 0 || poleNumber == 2 * lsrMdltr->periods)
    factor = lsrMdltr->poleFactor1;
  else if (poleNumber == 1 || poleNumber == (2 * lsrMdltr->periods - 1))
    factor = lsrMdltr->poleFactor2;
  else
    factor = lsrMdltr->poleFactor3;

  Bfactor = factor * lsrMdltr->Bu;
  kuz = lsrMdltr->ku * z;
  kuy = lsrMdltr->ku * y;
  kux = lsrMdltr->ku * x;
  /*
    if (lsrMdltr->fieldCode==LSRMDLTR_IDEAL) {
    B[0] = Bfactor*cos(kuz);
    B[1] = 0;
    } else if (lsrMdltr->fieldCode==LSRMDLTR_EXACT) {
    B[0] = Bfactor*cos(kuz)*cosh(kuy);
    B[1] = Bfactor*sin(kuz)*sinh(kuy);
    } else {
    B[0] = Bfactor*cos(kuz)*(1+sqr(kuy)/2);
    B[1] = Bfactor*sin(kuz)*kuy;
    } */
  laserModulatorUndulatorField(B, Bfactor, kux, kuy, kuz, lsrMdltr->tguGradient, lsrMdltr->ku, x,
                               lsrMdltr->tguCompFactor);

  B2 = sqr(B[0]) + sqr(B[1]) + sqr(B[2]);
  /* 1/rho^2 for central momentum */
  irho2 = B2 / sqr(gamma * me_mks * c_mks / e_mks);
  delta1 = delta0 = (p - P_central) / P_central;
  if (lsrMdltr->synchRad)
    delta1 -= radCoef * (1 + delta0) * irho2 * h;
  if (lsrMdltr->isr)
    delta1 += isrCoef * sqr(1 + delta0) * pow(irho2, 0.75) * sqrt(h) * gauss_rn_lim(0.0, 1.0, 3.0, random_2);

  for (i = 0; i < 3; i++)
    P[i] *= (1 + delta1) / (1 + delta0);
}

void laserModulatorUndulatorField(double *FieldB, double Bfactor, double kux, double kuy, double kuz,
                                  double a, double ku, double x, double CF) {
  LSRMDLTR *lsrMdltr;
#if defined(DEBUG1) && !USE_MPI
  static FILE *fpdeb = NULL;
  if (fpdeb == NULL) {
    fpdeb = fopen("lsrmdltr.fld", "w");
    fprintf(fpdeb, "SDDS1\n");
    fprintf(fpdeb, "&column name=x type=double units=m &end\n");
    fprintf(fpdeb, "&column name=y type=double units=m &end\n");
    fprintf(fpdeb, "&column name=z type=double units=m &end\n");
    fprintf(fpdeb, "&column name=Bx type=double units=T &end\n");
    fprintf(fpdeb, "&column name=By type=double units=T &end\n");
    fprintf(fpdeb, "&column name=Bz type=double units=T &end\n");
    fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  lsrMdltr = (LSRMDLTR *)field_global;

  if (lsrMdltr->helical == 0) {
    if (lsrMdltr->fieldCode == LSRMDLTR_IDEAL) {
      FieldB[1] = Bfactor * cos(kuz);
      FieldB[2] = 0;
    } else if (lsrMdltr->fieldCode == LSRMDLTR_EXACT) {
      FieldB[1] = Bfactor * cos(kuz) * cosh(kuy);
      FieldB[2] = -Bfactor * sin(kuz) * sinh(kuy);
      if (a != 0) {
        FieldB[0] = Bfactor * a / ku * cos(kuz) * sinh(kuy);
        FieldB[1] *= (1 + a * x);
        FieldB[1] += CF * sqr(Bfactor) * a * e_mks / (2 * gamma_central * me_mks * c_mks * sqr(ku));
        FieldB[2] = -Bfactor * (1 + a * x) * sin(kuz) * sinh(kuy);
      }
#if defined(DEBUG1) && !USE_MPI
      fprintf(fpdeb, "%le %le %le %le %le %le\n", x, kuy / ku, kuz / ku, FieldB[0], FieldB[1], FieldB[2]);
#endif
    } else {
      /* leading terms only */
      FieldB[1] = Bfactor * cos(kuz) * (1 + sqr(kuy) / 2);
      FieldB[2] = -Bfactor * sin(kuz) * kuy;
    }
  } else {
    if (lsrMdltr->fieldCode == LSRMDLTR_IDEAL) {
      FieldB[0] = -Bfactor * sin(kuz);
      FieldB[1] = Bfactor * cos(kuz);
      FieldB[2] = 0;
    } else if (lsrMdltr->fieldCode == LSRMDLTR_EXACT) {
      FieldB[0] = -Bfactor * sin(kuz) * cosh(kux);
      FieldB[1] = Bfactor * cos(kuz) * cosh(kuy);
      FieldB[2] = -Bfactor * sin(kuz) * sinh(kuy) - Bfactor * cos(kuz) * sinh(kux);
    } else {
      /* leading terms only */
      FieldB[0] = -Bfactor * sin(kuz) * (1 + sqr(kux) / 2);
      FieldB[1] = Bfactor * cos(kuz) * (1 + sqr(kuy) / 2);
      FieldB[2] = -Bfactor * sin(kuz) * kuy - Bfactor * cos(kuz) * kux;
    }
  }
}

void computeLaserField(double *Ef, double *Bf, double phase, double Ef0, double ZR,
                       double k, double w0, double x, double y, double dz,
                       double *timeValue, double *amplitudeValue, long profilePoints,
                       double tForProfile, long m, long n, double tilt) {
  /* Based on Alex Chao's "Laser Acceleration --- Focused Laser" note (unpublished) */

#if DEBUG
  static FILE *fpdeb = NULL;
#endif
  double sin_tilt, cos_tilt;
  double Vx, Vy;
  std::complex<double> Ec;
  std::complex<double> Q;
  double w, Hm, Hn, Hmp, Hnp, Hmpp;
  static long lastIndex = 0;
  double amplitude = 1;
  long i;

  for (i = 0; i < 3; i++)
    Ef[i] = Bf[i] = 0;
  if (tilt) {
    sin_tilt = sin(tilt);
    cos_tilt = cos(tilt);
    Vx = x;
    Vy = y;
    x = Vx * cos_tilt + Vy * sin_tilt;
    y = -Vx * sin_tilt + Vy * cos_tilt;
  } else {
    cos_tilt = 1;
    sin_tilt = 0;
  }

  if (timeValue) {
    if (tForProfile < timeValue[0] || tForProfile > timeValue[profilePoints - 1])
      return;
    else {
      if ((i = lastIndex) >= (profilePoints - 1))
        i = profilePoints - 2;
      if (i < 0)
        i = 0;
      if (tForProfile < timeValue[i]) {
        do {
          i--;
        } while (i >= 0 && tForProfile < timeValue[i]);
      } else if (tForProfile > timeValue[i + 1]) {
        do {
          i++;
        } while (i < (profilePoints - 1) && tForProfile > timeValue[i + 1]);
      }
      if (i < 0 || i >= (profilePoints - 1))
        return;
      else
        amplitude = INTERPOLATE(amplitudeValue[i], amplitudeValue[i + 1],
                                timeValue[i], timeValue[i + 1],
                                tForProfile);
#if DEBUG
      if (!fpdeb) {
        fpdeb = fopen((char *)"lsrMdltr.sdds", (char *)"w");
        fprintf(fpdeb, (char *)"SDDS1\n&column name=t type=double units=s &end\n");
        fprintf(fpdeb, (char *)"&column name=A type=double &end\n");
        fprintf(fpdeb, (char *)"&column name=dz type=double units=m &end\n");
        fprintf(fpdeb, (char *)"&data mode=ascii no_row_counts=1 &end\n");
      }
      fprintf(fpdeb, (char *)"%e %e %e\n", tForProfile, amplitude, dz);
#endif
      lastIndex = i;
    }
  }
  Q = 1.0 / std::complex<double>(dz, -ZR);
  w = w0 * sqrt(1 + sqr(dz / ZR));
  Ec = std::complex<double>(0, phase) -
    std::complex<double>(0, (m + n + 1) * atan(dz / ZR)) +
    std::complex<double>(0, 1) * std::complex<double>(k * Q / 2.0 * (x * x + y * y));
  Ec = amplitude * Ef0 * w0 / w * exp(Ec);
  Hm = HermitePolynomial(sqrt(2.0) * x / w, m);
  Hn = HermitePolynomial(sqrt(2.0) * y / w, n);
  Hmp = HermitePolynomialDeriv(sqrt(2.0) * x / w, m);
  Hnp = HermitePolynomialDeriv(sqrt(2.0) * y / w, n);
  Hmpp = HermitePolynomial2ndDeriv(sqrt(2.0) * x / w, m);

  Ef[0] = Hm * Hn * Ec.real();
  Ef[1] = 0;
  Ef[2] = (Ec * (std::complex<double>(0, 1) * sqrt(2.0) / (k * w) * Hmp * Hn - Q * x * Hm * Hn)).real();
  Bf[0] = (Ec * (2 / sqr(k * w) * Hmp * Hnp + std::complex<double>(0, 1) * sqrt(2.0) * Q / (k * w) * (x * Hm * Hnp + y * Hmp * Hn) - x * y * Q * Q * Hm * Hn)).real() / c_mks;
  Bf[1] = (Ec * (-2 / sqr(k * w) * Hmpp * Hn - 2 * sqrt(2.0) * std::complex<double>(0, 1) * Q / (k * w) * x * Hmp * Hn + (Q * Q * sqr(x) - std::complex<double>(0, 1) * Q / k + 1.0) * Hm * Hn)).real() / c_mks;
  Bf[2] = (Ec * (std::complex<double>(0, 1) * sqrt(2.0) / (k * w) * Hm * Hnp - Q * y * Hm * Hn)).real() / c_mks;

  if (tilt) {
    Vx = Ef[0];
    Vy = Ef[1];
    Ef[0] = Vx * cos_tilt - Vy * sin_tilt;
    Ef[1] = Vx * sin_tilt + Vy * cos_tilt;
    Vx = Bf[0];
    Vy = Bf[1];
    Bf[0] = Vx * cos_tilt - Vy * sin_tilt;
    Bf[1] = Vx * sin_tilt + Vy * cos_tilt;
  }
}

double HermitePolynomial(double x, long n) {
  double result;
  /* Butkov, Mathematical Physics */
  switch (n) {
  case 0:
    result = 1;
    break;
  case 1:
    result = 2 * x;
    break;
  case 2:
    result = 4 * x * x - 2;
    break;
  case 3:
    result = 8 * ipow3(x) - 12 * x;
    break;
  case 4:
    result = ipow2(x);
    result = 16 * ipow2(result) - 48 * result + 12;
    break;
  default:
    fprintf(stderr, (char *)"Sorry, laser mode number too high---send an email to borland@aps.anl.gov if you really need this.\n");
    result = 0; /* suppresses spurious compiler warning */
    exitElegant(1);
    break;
  }
  return result;
}

double HermitePolynomialDeriv(double x, long n) {
  double result;
  /* Butkov, Mathematical Physics */
  switch (n) {
  case 0:
    result = 0;
    break;
  case 1:
    result = 2;
    break;
  case 2:
    result = 8 * x;
    break;
  case 3:
    result = 24 * ipow2(x) - 12;
    break;
  case 4:
    result = 64 * ipow3(x) - 96 * x;
    break;
  default:
    fprintf(stderr, (char *)"Sorry, laser mode number too high\n");
    result = 0; /* suppresses spurious compiler warning */
    exitElegant(1);
    break;
  }
  return result;
}

double HermitePolynomial2ndDeriv(double x, long n) {
  double result;
  /* Butkov, Mathematical Physics */
  switch (n) {
  case 0:
    result = 0;
    break;
  case 1:
    result = 0;
    break;
  case 2:
    result = 8;
    break;
  case 3:
    result = 48 * x;
    break;
  case 4:
    result = 3 * 64 * ipow2(x) - 96;
    break;
  default:
    fprintf(stderr, (char *)"Sorry, laser mode number too high\n");
    result = 0; /* suppresses spurious compiler warning */
    exitElegant(1);
    break;
  }
  return result;
}
