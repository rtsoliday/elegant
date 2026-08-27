/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: kick_sbend.c
 * contents:  track_through_kick_sbend()
 *
 *
 * Michael Borland, 1991, 1992.
 */
#include "mdb.h"
#include "track.h"

void kick_sbend_derivs(
  double *dQds, /* return: x', x'', y', y''  */
  double *Q,    /* input:  x , x' , y , y'   */
  double s      /* path length of reference trajectory--ignored */
);
void integrate_ord2(double *Qf, double *Qi, double s, long n);

static double Fy_0, Fy_x, Fy_x2, Fy_x3, Fy_x4;
static double Fy_y2, Fy_x_y2, Fy_x2_y2;
static double Fy_y4;

static double Fx_y, Fx_x_y, Fx_x2_y, Fx_x3_y;
static double Fx_y3, Fx_x_y3;

static double nh, betah2, gammah3, deltah4;
static double rho0, h, ha, h2, h3, rad_coef;

static long particle_lost, paraxial;
static double s_lost;

#define LEAP_FROG 0
#define MODIFIED_MIDPOINT 1
#define N_METHODS 2
static char *method[N_METHODS] = {"leap-frog", "modified-midpoint"};

long track_through_kick_sbend(double **part, int64_t n_part, KSBEND *ksbend, double p_error, double Po, double **accepted,
                              double z_start) {
  int64_t i_part, i_top;
  double rho, s;
  double x, xp, y, yp, dp;
  double n, beta, gamma, delta, fse;
  double tilt, etilt, cos_ttilt, sin_ttilt, ttilt;
  double *coord;
  double angle, e1, e2, Kg;
  double psi1, psi2;
  double Qi[6], Qf[6], dQds[6];
  double dcoord_etilt[6];
  long method_code;
  double dxi, dyi, dzi;
  double dxf, dyf, dzf;

  log_entry("track_through_kick_sbend");

  if (!ksbend)
    bombElegant("null KSBEND pointer (track_through_kick_sbend)", NULL);

  if (ksbend->angle == 0)
    bombElegant("angle = 0 for ksbend", NULL);

  if ((method_code = match_string(ksbend->method, method, N_METHODS, 0)) < 0)
    bombElegant("unknown method for ksbend integration", NULL);

  paraxial = ksbend->paraxial;

  if (ksbend->angle < 0) {
    angle = -ksbend->angle;
    e1 = -ksbend->e[ksbend->e1Index];
    e2 = -ksbend->e[ksbend->e2Index];
    etilt = ksbend->etilt;
    tilt = ksbend->tilt + PI;
    rho0 = -ksbend->length / angle;
    n = -sqr(rho0) * ksbend->k1;
    beta = 0.5 * ksbend->k2 * pow3(rho0);
    gamma = ksbend->k3 * pow4(rho0) / 6.;
    delta = ksbend->k4 * pow5(rho0) / 24.;
    rho0 = -rho0;
  } else {
    angle = ksbend->angle;
    e1 = ksbend->e[ksbend->e1Index];
    e2 = ksbend->e[ksbend->e2Index];
    etilt = ksbend->etilt;
    tilt = ksbend->tilt;
    rho0 = ksbend->length / angle;
    n = -sqr(rho0) * ksbend->k1;
    beta = 0.5 * ksbend->k2 * pow3(rho0);
    gamma = ksbend->k3 * pow4(rho0) / 6.;
    delta = ksbend->k4 * pow5(rho0) / 24.;
  }

  fse = ksbend->fse;
  h = 1. / rho0;
  h2 = sqr(h);
  h3 = h * h2;
  nh = n * h;
  betah2 = beta * h2;
  gammah3 = gamma * h3;
  deltah4 = delta * h2 * h2;
  ha = (1 + fse) * h;

  /* angles for fringe-field effects */
  Kg = 2 * ksbend->hgap * ksbend->fint;
  psi1 = Kg / rho0 / cos(e1) * (1 + sqr(sin(e1)));
  psi2 = Kg / rho0 / cos(e2) * (1 + sqr(sin(e2)));

  /* rad_coef is d((P-Po)/Po)/ds for the on-axis, on-momentum particle, where po is the momentum of
     * the central particle.
     */
  if (ksbend->synch_rad)
    rad_coef = sqr(particleCharge) * pow3(Po) * sqr(1 + fse) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho0));
  else
    rad_coef = 0;

  Fy_0 = 1;
  Fy_x = -nh;
  Fy_x2 = Fy_x3 = Fy_x4 = Fy_y2 = Fy_x_y2 = Fy_x2_y2 = Fy_y4 = 0;
  if (ksbend->nonlinear) {
    Fy_x2 = betah2;
    Fy_x3 = gammah3;
    Fy_y2 = (h * nh - 2 * betah2) / 2;
    Fy_x_y2 = -(2 * h * betah2 + nh * h2 + 6 * gammah3) / 2;
    Fy_x4 = deltah4;
    Fy_x2_y2 = -(3 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 12 * deltah4) / 2;
    Fy_y4 = (12 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 24 * deltah4) / 24;
  }

  Fx_y = -nh;
  Fx_x_y = Fx_x2_y = Fx_x3_y = Fx_y3 = Fx_x_y3 = 0;
  if (ksbend->nonlinear) {
    Fx_x_y = 2 * betah2;
    Fx_x2_y = 3 * gammah3;
    Fx_y3 = -(2 * h * betah2 + nh * h2 + 6 * gammah3) / 6;
    Fx_x3_y = 4 * deltah4;
    Fx_x_y3 = -(3 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 12 * deltah4) / 3;
  }

  ttilt = tilt + etilt;
  if (ttilt == 0) {
    cos_ttilt = 1;
    sin_ttilt = 0;
  } else if (fabs(fabs(ttilt) - PI) < 1e-12) {
    cos_ttilt = -1;
    sin_ttilt = 0;
  } else if (fabs(ttilt - PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = 1;
  } else if (fabs(ttilt + PIo2) < 1e-12) {
    cos_ttilt = 0;
    sin_ttilt = -1;
  } else {
    cos_ttilt = cos(ttilt);
    sin_ttilt = sin(ttilt);
  }

  log_entry("track_through_kick_sbend.1");

  if (etilt) {
    /* compute final offsets due to error-tilt of the magnet */
    /* see pages 90-93 of notebook 1 about this */
    double q1a, q2a, q3a;
    double q1b, q2b, q3b;
    double qp1, qp2, qp3;
    double dz, tan_alpha, k;

    q1a = (1 - cos(angle)) * rho0 * (cos(etilt) - 1);
    q2a = 0;
    q3a = (1 - cos(angle)) * rho0 * sin(etilt);
    qp1 = sin(angle) * cos(etilt);
    qp2 = cos(angle);
    k = sqrt(sqr(qp1) + sqr(qp2));
    qp1 /= k;
    qp2 /= k;
    qp3 = sin(angle) * sin(etilt) / k;
    tan_alpha = 1. / tan(angle) / cos(etilt);
    q1b = q1a * tan_alpha / (tan(angle) + tan_alpha);
    q2b = -q1b * tan(angle);
    dz = sqrt(sqr(q1b - q1a) + sqr(q2b - q2a));
    q3b = q3a + qp3 * dz;

    dcoord_etilt[0] = sqrt(sqr(q1b) + sqr(q2b));
    dcoord_etilt[1] = tan(atan(tan_alpha) - (PIo2 - angle));
    dcoord_etilt[2] = q3b;
    dcoord_etilt[3] = qp3;
    dcoord_etilt[4] = dz * sqrt(1 + sqr(qp3));
    dcoord_etilt[5] = 0;
#ifdef DEBUG
    printf("pre-tilt offsets due to ETILT=%le:  %le %le %le %le %le\n",
           etilt, dcoord_etilt[0], dcoord_etilt[1], dcoord_etilt[2],
           dcoord_etilt[3], dcoord_etilt[4]);
    fflush(stdout);
#endif

    /* rotate by tilt to get into same frame as bend equations. */
    rotateCoordinatesForMisalignment(dcoord_etilt, tilt);
#ifdef DEBUG
    printf("offsets due to ETILT=%le:  %le %le %le %le %le\n",
           etilt, dcoord_etilt[0], dcoord_etilt[1], dcoord_etilt[2],
           dcoord_etilt[3], dcoord_etilt[4]);
    fflush(stdout);
#endif
  } else
    fill_double_array(dcoord_etilt, 6L, 0.0);

  dxi = -ksbend->dx;
  dzi = ksbend->dz;
  dyi = -ksbend->dy;

  /* must use the original angle here because the translation is done after
     * the final rotation back
     */
  dxf = ksbend->dx * cos(ksbend->angle) + ksbend->dz * sin(ksbend->angle);
  dzf = ksbend->dx * sin(ksbend->angle) - ksbend->dz * cos(ksbend->angle);
  dyf = ksbend->dy;

  log_exit("track_through_kick_sbend.1");

  log_entry("track_through_kick_sbend.2");
  i_top = n_part - 1;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!part) {
      printf("error: null particle array found (working on particle %" PRId64 ") (track_through_kick_sbend)\n", i_part);
      fflush(stdout);
      abort();
    }
    if (!(coord = part[i_part])) {
      printf("error: null coordinate pointer for particle %" PRId64 " (track_through_kick_sbend)\n", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("error: null accepted particle pointer for particle %" PRId64 " (track_through_kick_sbend)\n", i_part);
      fflush(stdout);
      abort();
    }

    coord[4] += dzi * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    coord[0] = coord[0] + dxi + dzi * coord[1];
    coord[2] = coord[2] + dyi + dzi * coord[3];

    x = coord[0] * cos_ttilt + coord[2] * sin_ttilt;
    y = -coord[0] * sin_ttilt + coord[2] * cos_ttilt;
    xp = coord[1] * cos_ttilt + coord[3] * sin_ttilt;
    yp = -coord[1] * sin_ttilt + coord[3] * cos_ttilt;
    s = coord[4];
    dp = coord[5];

    rho = (1 + dp) * rho0;
    if (ksbend->edge_effects[ksbend->e1Index]) {
      /* apply edge focusing */
      if (ksbend->edge_order > 1)
        apply_edge_effects(&x, &xp, &y, &yp, rho, n, e1, ksbend->h[ksbend->e1Index], psi1 / (1 + dp), -1);
      else {
        xp += tan(e1) / rho * x;
        yp -= tan(e1 - psi1 / (1 + dp)) / rho * y;
      }
    }

    /* transform to curvilinear coordinates */
    xp *= (1 + x / rho0);
    yp *= (1 + x / rho0);

    /* load input coordinates into arrays */
    Qi[0] = x;
    Qi[1] = xp;
    Qi[2] = y;
    Qi[3] = yp;
    Qi[4] = 0;
    Qi[5] = dp;

    if (ksbend->edge_effects[ksbend->e1Index] && e1 != 0 && rad_coef) {
      /* pre-adjust dp/p to anticipate error made by integrating over entire sector */
      particle_lost = 0;
      kick_sbend_derivs(dQds, Qi, 0.0);
      Qi[5] -= dQds[5] * Qi[0] * tan(e1);
    }

    /* integrate using modified midpoint method */
    particle_lost = 0;
    kick_sbend_derivs(dQds, Qi, 0.0);
    if (!particle_lost) {
      if (method_code == MODIFIED_MIDPOINT)
        mmid(Qi, dQds, 6, 0.0, ksbend->length, ksbend->n_kicks, Qf, kick_sbend_derivs);
      else
        integrate_ord2(Qf, Qi, ksbend->length, ksbend->n_kicks);
    }

    if (particle_lost) {
      if (!part[i_top]) {
        printf("error: couldn't swap particles %" PRId64 " and %" PRId64 "--latter is null pointer (track_through_kick_sbend)\n",
               i_part, i_top);
        fflush(stdout);
        abort();
      }
      SWAP_PTR(part[i_part], part[i_top]);
      if (accepted) {
        if (!accepted[i_top]) {
          printf(
            "error: couldn't swap acceptance data for particles %" PRId64 " and %" PRId64 "--latter is null pointer (track_through_kick_sbend)\n",
            i_part, i_top);
          fflush(stdout);
          abort();
        }
        SWAP_PTR(accepted[i_part], accepted[i_top]);
      }
      part[i_top][4] = z_start + s_lost;
      part[i_top][5] = Po * (1 + part[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }

    if (ksbend->edge_effects[ksbend->e2Index] && e2 != 0 && rad_coef) {
      /* post-adjust dp/p to correct error made by integrating over entire sector */
      kick_sbend_derivs(dQds, Qf, 0.0);
      Qf[5] -= dQds[5] * Qf[0] * tan(e2);
    }

    if (rad_coef) {
      double p0, p1;
      double beta0, beta1;
      /* fix previous distance information to reflect new velocity--since distance
             * is really time-of-flight at the current velocity 
             */
      p0 = Po * (1 + dp);
      beta0 = p0 / sqrt(sqr(p0) + 1);
      p1 = Po * (1 + Qf[5]);
      beta1 = p1 / sqrt(sqr(p1) + 1);
      s = beta1 * (s / beta0 + 2 * Qf[4] / (beta0 + beta1));
    } else
      s += Qf[4];

    /* get final coordinates */
    x = Qf[0];
    xp = Qf[1];
    y = Qf[2];
    yp = Qf[3];
    dp = Qf[5];

    /* transform to cartesian coordinates */
    xp /= (1 + x / rho0);
    yp /= (1 + x / rho0);

    rho = (1 + dp) * rho0;
    if (ksbend->edge_effects[ksbend->e2Index]) {
      /* apply edge focusing */
      if (ksbend->edge_order > 1)
        apply_edge_effects(&x, &xp, &y, &yp, rho, n, e2, ksbend->h[ksbend->e2Index], psi2 / (1 + dp), 1);
      else {
        xp += tan(e2) / rho * x;
        yp -= tan(e2 - psi2 / (1 + dp)) / rho * y;
      }
    }

    coord[0] = x * cos_ttilt - y * sin_ttilt + dcoord_etilt[0];
    coord[2] = x * sin_ttilt + y * cos_ttilt + dcoord_etilt[2];
    coord[1] = xp * cos_ttilt - yp * sin_ttilt + dcoord_etilt[1];
    coord[3] = xp * sin_ttilt + yp * cos_ttilt + dcoord_etilt[3];
    coord[4] = s;
    coord[5] = dp;

    coord[0] += dxf + dzf * coord[1];
    coord[2] += dyf + dzf * coord[3];
    coord[4] += dzf * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  }
  log_exit("track_through_kick_sbend.2");

  log_exit("track_through_kick_sbend");
  return (i_top + 1);
}

void kick_sbend_derivs(
  double *dQds, /* return: x', x'', y', y'', s', d(dp/p)/ds  */
  double *Q,    /* input:  x , x' , y , y' , s , dp/p        */
  double s0     /* path length of reference trajectory--ignored */
) {
  double xp, yp, x, y, Fx, Fy, factor;
  double one_plus_hx, xp2, yp2, Tp2, Tp2_yp2, Tp, y2, one_plus_dp, xpyp;

  if (!dQds)
    bombElegant("null derivative pointer (kick_sbend_derivs)", NULL);
  if (!Q)
    bombElegant("null coordinate pointer (kick_sbend_derivs)", NULL);

  if (particle_lost) {
    dQds[0] = dQds[1] = dQds[2] = dQds[3] = dQds[4] = dQds[5] = 0;
    return;
  }

  x = Q[0];
  xp = dQds[0] = Q[1];
  y = Q[2];
  yp = dQds[2] = Q[3];

  if (FABS(xp) > slopeLimit || FABS(yp) > slopeLimit || FABS(x) > coordLimit || FABS(y) > coordLimit) {
    particle_lost = 1;
    s_lost = s0;
    dQds[0] = dQds[1] = dQds[2] = dQds[3] = dQds[4] = dQds[5] = 0;
    return;
  }
#ifdef IEEE_MATH
  if (isnan(xp) || isnan(yp) || isnan(x) || isnan(y)) {
    particle_lost = 1;
    s_lost = s0;
    dQds[0] = dQds[1] = dQds[2] = dQds[3] = dQds[4] = dQds[5] = 0;
    return;
  }
#endif

  /* compute quantities needed--h=1/rho0 is the curvature of the reference trajectory */
  one_plus_hx = 1 + h * x;
  if (paraxial) {
    /* guarantees symplectic behaviour with leap-frog integration */
    xp = yp = 0;
    xp2 = yp2 = xpyp = 0;
    Tp = one_plus_hx;
    Tp2 = sqr(one_plus_hx);
    Tp2_yp2 = Tp2;
  } else {
    xp2 = xp * xp;
    yp2 = yp * yp;
    xpyp = xp * yp;
    Tp2_yp2 = xp2 + sqr(one_plus_hx);
    Tp2 = Tp2_yp2 + yp2;
    Tp = sqrt(Tp2);
  }
  one_plus_dp = 1 + Q[5];
  y2 = y * y;

  /* compute scaled field values to using fourth-order expansion */
  Fx = (Fx_y + (Fx_x_y + (Fx_x2_y + Fx_x3_y * x) * x) * x + (Fx_y3 + Fx_x_y3 * x) * y2) * y;
  Fy = Fy_0 + (Fy_x + (Fy_x2 + (Fy_x3 + Fy_x4 * x) * x) * x) * x + (Fy_y2 + (Fy_x_y2 + Fy_x2_y2 * x) * x + Fy_y4 * y2) * y2;

  factor = Tp * ha / one_plus_dp;

  dQds[1] = ((Fx * xpyp - Fy * (Tp2_yp2)) * factor + h * (Tp2_yp2 + xp2)) / one_plus_hx;
  dQds[3] = ((Fx * (Tp2 - xp2) - Fy * xpyp) * factor + 2 * h * xpyp) / one_plus_hx;
  dQds[4] = Tp;
  if (rad_coef)
    dQds[5] = -rad_coef * (sqr(Fx) + sqr(Fy)) * sqr(one_plus_dp) * Tp;
  else
    dQds[5] = 0;
}

void leapstep(double *Q, void (*derivs)(double *dQds, double *Qi, double s), double ds) {
  static double F[6], dsh;

  dsh = ds / 2;
  Q[0] += dsh * Q[1];
  Q[2] += dsh * Q[3];
  derivs(F, Q, 0.0);
  Q[1] += ds * F[1];
  Q[3] += ds * F[3];
  Q[0] += dsh * Q[1];
  Q[2] += dsh * Q[3];
  Q[4] += F[4] * ds;
  Q[5] += F[5] * ds;
}

void integrate_ord2(double *Qf, double *Qi, double s, long n) {
  static long i;
  static double ds, dsh, dQds[6];

  for (i = 0; i < 6; i++)
    Qf[i] = Qi[i];

  ds = s / n;
  dsh = ds / 2;
  for (i = 0; i < n; i++) {
    Qf[0] += dsh * Qf[1];
    Qf[2] += dsh * Qf[3];
    kick_sbend_derivs(dQds, Qf, 0.0);
    Qf[1] += ds * dQds[1];
    Qf[3] += ds * dQds[3];
    Qf[0] += dsh * Qf[1];
    Qf[2] += dsh * Qf[3];
    Qf[4] += dQds[4] * ds;
    Qf[5] += dQds[5] * ds;
    /*        leapstep(Qf, kick_sbend_derivs, ds); */
  }
}
