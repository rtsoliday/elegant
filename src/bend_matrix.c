/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: bend_matrix()
 * purpose: calculate matrix for bending magnet 
 * 
 * Michael Borland, 1989, 1991
 */
#include "mdb.h"
#include "track.h"

void replaceWithNewMatrix(double *C, double **R, double ***T, double kx2, double ky2, double ha, double h,
                          double nh, double bh2, double gh3, double s);

VMATRIX *bend_matrix(
  double length, /* arc length */
  double angle,  /* bending angle */
  double ea1,    /* entrance and exit pole face angles */
  double ea2,
  double hPole1, /* entrance and exit pole curvatures */
  double hPole2,
  double k1,            /* quadrupole term */
  double k2,            /* sextupole term */
  double tilt,          /* tilt angle */
  double fint1,         /* FINT in MAD format, K in SLAC-75 notation */
  double fint2,         /* FINT in MAD format, K in SLAC-75 notation */
  double gap,           /* full gap */
  double fse,           /* Fractional Strength Error of all components */
  double fseDipole,     /* Fractional Strength Error of dipole component */
  double fseQuadrupole, /* Fractional Strength Error of quadrupole component */
  double etilt,         /* error tilt angle--changes y, y' for reference trajectory */
  long order,
  long edge_order,
  long flags,
  long TRANSPORT, /* if nonzero, uses TRANSPORT equation for T436 of edge, which I think is incorrect... */
  double hkick,   /* bend-plane steering kick */
  double vkick    /* non-bend-plane steering kick */
) {
  double n, beta, gamma;
  double h, ha;
  VMATRIX *M, *Medge, *Mtot, *tmp;

  log_entry("bend_matrix");

  if (FABS(angle) < 1e-10 || fabs(length/angle)>1e6) {
    if (k1==0 && k2==0) {
      printWarning("Bending magnet has |angle|<1e-10 or rho>1e6", "K1=K2=0 so using drift matrix");
      return drift_matrix(length, order);
    }
    if (k2==0) {
      printWarning("Bending magnet has |angle|<1e-10 or rho>1e6", "K1!=0 and K2=0 so using quadrupole matrix");
      return quadrupole_matrix(k1, length, order, fse+fseQuadrupole, hkick, vkick, 0, 0, NULL, 0.0, 0.0, NULL, NULL, 0);
    }
    if (k1==0) {
      printWarning("Bending magnet has |angle|<1e-10 or rho>1e6", "K1==0 and K2!=0 so using sextupole matrix");
      return sextupole_matrix(k2, 0.0, 0.0, length, order, fse, hkick, vkick, 0.0);
    }
    printWarning("Bending magnet has |angle|<1e-10 or rho>1e6", "K1!=0 and K2!=0 so using quse (quad/sext) matrix");
    return quse_matrix(k1, k2, length, order, fse+fseQuadrupole, fse);
  }
  if (length == 0)
    bombElegant("zero-length bend magnet not supported", NULL);

  if (angle < 0) {
    /* Note that k2 gets a minus sign here because beta has a rho^3 in it. */
    M = bend_matrix(length, -angle, -ea1, -ea2, hPole1, hPole2, k1, -k2, tilt,
                    fint1, fint2, gap, fse, fseDipole, fseQuadrupole, etilt, order, edge_order, flags, TRANSPORT,
                    -hkick, -vkick);
    tilt_matrices(M, PI);
    log_exit("bend_matrix");
    return (M);
  }

  /* calculate constants */
  h = angle / length;             /* coordinate system curvature */
  ha = h * (1 + fse + fseDipole); /* actual curvature due to bending field */
  /* field index --- this complex expression is needed to ensure that FSE works out right in sbend_matrix() */
  /* i.e., we get ky2 = -K1*(1+fse+fseQuadrupole) */
  n = -k1 * (1 + fse + fseQuadrupole) / (1 + fse + fseDipole) / sqr(h);
  /* sextupole index --- complex expression again ensures that fseDipole doesn't affect sextupole strength */
  /* in sbend_matrix() we compute beta*h^2*ha for use in expressions */
  beta = k2 * (1 + fse) / (1 + fse + fseDipole) / 2 / pow3(h);
  gamma = 0;

  if (hkick || vkick) {
    /* Split into 128 pieces with kicks in between */
    /* This approximates the effect of distributed steering kicks, but there are
       * missing terms in the field expansion so it isn't really correct. 
       */
    VMATRIX *Mc, *Mslice, *M1, *M2;
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, order);
    M1 = tmalloc(sizeof(*M1));
    initialize_matrices(M1, order);
    M2 = tmalloc(sizeof(*M2));
    initialize_matrices(M2, order);

    Mc = hvcorrector_matrix(0, hkick / 129, vkick / 129, 0, 0, 1, 1, 0, order);
    Mslice = sbend_matrix(length / 128, h, ha, n * h, beta * sqr(h), gamma * pow3(h), order);
    concat_matrices(M1, Mslice, Mc, 0); /* Mslice*Mc */
    concat_matrices(M2, M1, M1, 0);     /* (MSlice*Mc)^2 */
    concat_matrices(M1, M2, M2, 0);     /* (MSlice*Mc)^4 */
    concat_matrices(M2, M1, M1, 0);     /* (MSlice*Mc)^8 */
    concat_matrices(M1, M2, M2, 0);     /* (MSlice*Mc)^16 */
    concat_matrices(M2, M1, M1, 0);     /* (MSlice*Mc)^32 */
    concat_matrices(M1, M2, M2, 0);     /* (MSlice*Mc)^64 */
    concat_matrices(M2, M1, M1, 0);     /* (MSlice*Mc)^128 */
    concat_matrices(M, Mc, M2, 0);      /* Mc*[(MSlice*Mc)^128] */
    free_matrices(Mc);
    free(Mc);
    free_matrices(M1);
    free(M1);
    free_matrices(M2);
    free(M2);
    free_matrices(Mslice);
    free(Mslice);
  } else {
    M = sbend_matrix(length, h, ha, n * h, beta * sqr(h), gamma * pow3(h), order);
  }
#ifdef DEBUG
  print_matrices(stdout, "pure bend matrix", M, 0.0);
#endif

  if (edge_order == 0)
    edge_order = order;
  if (ea1 != 0 || ea2 != 0 || order > 1 || fint1 * gap != 0 || fint2 * gap != 0) {
    Mtot = tmalloc(sizeof(*Mtot));
    initialize_matrices(Mtot, M->order);

    if ((flags & BEND_EDGE1_EFFECTS) && !(flags & SAME_BEND_PRECEDES)) {
      Medge = edge_matrix(ea1, ha, hPole1, n, -1, fint1 * gap, order, edge_order >= 2, TRANSPORT, length);
      concat_matrices(Mtot, M, Medge, 0);
      tmp = Mtot;
      Mtot = M;
      M = tmp;
      free_matrices(Medge);
      tfree(Medge);
      Medge = NULL;
    }

    if ((flags & BEND_EDGE2_EFFECTS) && !(flags & SAME_BEND_FOLLOWS)) {
      Medge = edge_matrix(ea2, ha, hPole2, n, 1, fint2 * gap, order, edge_order >= 2, TRANSPORT, length);
      concat_matrices(Mtot, Medge, M, 0);
      tmp = Mtot;
      Mtot = M;
      M = tmp;
      free_matrices(Medge);
      tfree(Medge);
      Medge = NULL;
    }

    free_matrices(Mtot);
    tfree(Mtot);
    Mtot = NULL;
  }

#ifdef DEBUG
  print_matrices(stdout, "concatentated bend matrix", M, 0.0);
#endif

  tilt_matrices(M, tilt + etilt);
  if (etilt) {
    /* see pages 90-93 of notebook 1 about this */
    double q1a, q2a, q3a;
    double q1b, q2b, q3b;
    double qp1, qp2, qp3;
    double dz, alpha, dcoord[4];

    etilt *= -1;
    q1a = (1 - cos(angle)) / h * (cos(etilt) - 1);
    q2a = 0;
    q3a = (1 - cos(angle)) / h * sin(etilt);
    qp1 = sin(angle) * cos(etilt);
    qp2 = cos(angle);
    alpha = sqrt(sqr(qp1) + sqr(qp2));
    qp1 /= alpha;
    qp2 /= alpha;
    qp3 = sin(angle) * sin(etilt) / alpha;
    alpha = atan(1. / tan(angle) / cos(etilt));
    q1b = q1a * tan(alpha) / (tan(angle) + tan(alpha));
    q2b = -q1b * tan(angle);
    dz = sqrt(sqr(q1b - q1a) + sqr(q2b - q2a));
    q3b = q3a + qp3 * dz;
    dcoord[0] = sqrt(sqr(q1b) + sqr(q2b));
    dcoord[1] = tan(alpha - (PIo2 - angle));
    dcoord[2] = q3b;
    dcoord[3] = qp3;
    rotateCoordinatesForMisalignment(dcoord, tilt);
    M->C[0] += dcoord[0];
    M->C[1] += dcoord[1];
    M->C[2] += dcoord[2];
    M->C[3] += dcoord[3];
    M->C[4] += dz * sqrt(1 + sqr(qp3));
    etilt *= -1;
  }

  log_exit("bend_matrix");
  return (M);
}

VMATRIX *edge_matrix(
  double beta,     /* edge angle--beta>0->more rectangular */
  double h,        /* curvature of beam path */
  double hPole,    /* curvature of pole face */
  double n,        /* field index */
  long which_edge, /* -1=entrance, 1=exit */
  double gK,       /* gap*K (see SLAC 75, pg117) */
  long order,      /* order desired */
  long all_terms,  /* if zero, only chromatic second-order terms are included */
  long TRANSPORT,  /* if nonzero, uses TRANSPORT equation for T436, which I think is incorrect... */
  double length) {
  double tan_beta, tan2_beta, sec_beta, sec2_beta, h2;
  double psi = 0.0;
  VMATRIX *M;
  double **R, ***T;

  /*
  printf("edge_matrix(beta=%le, h=%le, hPole=%le, n=%le, which_edge=%ld, gK=%le, order=%ld, all_terms=%ld, TRANSPORT=%ld, length=%le)\n",
	 beta, h, hPole, n, which_edge, gK, order, all_terms, TRANSPORT, length);
  */
  
  log_entry("edge_matrix");

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(2, order));
  R = M->R;
  T = M->T;

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
  R[1][0] = h * (tan_beta = tan(beta));
  sec_beta = 1. / cos(beta);
  if (gK) {
    psi = (length < 0 ? -1 : 1) * gK * h * sec_beta * (1 + sqr(sin(beta)));
    R[3][2] = -h * tan(beta - psi);
  } else
    R[3][2] = -R[1][0];

  if (M->order > 1) {
    h2 = sqr(h);
    if (all_terms) {
      tan2_beta = sqr(tan_beta);
      T[0][0][0] = which_edge * h / 2 * tan2_beta;
      sec_beta = 1. / cos(beta);
      sec2_beta = sqr(sec_beta);
      T[0][2][2] = -which_edge * h / 2 * sec2_beta;
      T[1][0][0] = which_edge == -1 ? -n * h2 * tan_beta : -h2 * (n + tan2_beta / 2) * tan_beta;
      T[3][3][0] = -(T[2][2][0] = T[1][1][0] = -which_edge * h * tan2_beta);
      T[1][2][2] = which_edge == -1 ? h2 * (n + .5 + tan2_beta) * tan_beta : h2 * (n - tan2_beta / 2) * tan_beta;
      T[1][3][2] = which_edge * h * tan2_beta;
      T[3][2][0] = h2 * (2 * n + (which_edge == 1 ? sec2_beta : 0)) * tan_beta;
      T[3][2][1] = which_edge * h * sec2_beta;
      if (hPole != 0) {
        double term;
        term = h / 2 * hPole * sec_beta * sec2_beta;
        T[1][0][0] += term;
        T[1][2][2] -= term;
        T[3][2][0] -= 2 * term;
      }
    }
    T[1][5][0] = -R[1][0];
    if (!TRANSPORT) {
      T[3][5][2] = -R[3][2];
      if (gK)
        T[3][5][2] -= psi * h * sqr(1. / cos(beta - psi));
    } else
      T[3][5][2] = R[1][0];
  }

  log_exit("edge_matrix");
  return (M);
}

/* routine: hvcorrector_matrix()
 * purpose: return matrix for orbit corrector that kicks in both planes
 */

VMATRIX *hvcorrector_matrix(
  double length,              /* straight-through length--i.e., not arc length */
  double xkick, double ykick, /* these are angles, not slopes */
  double tilt, double b2,
  double xcal, double ycal,
  long do_edges, long max_order) {
  double kick;

  log_entry("hvcorrector_matrix");

  xkick *= xcal;
  ykick *= ycal;

  if (xkick == 0 && ykick == 0) {
    log_exit("hvcorrector_matrix");
    return (drift_matrix(length, max_order));
  }
  if (xkick == 0) {
    log_exit("hvcorrector_matrix");
    return (corrector_matrix(length, ykick, PIo2 + tilt, b2, 1.0, do_edges, max_order));
  }
  if (ykick == 0) {
    log_exit("hvcorrector_matrix");
    return (corrector_matrix(length, xkick, tilt, b2, 1.0, do_edges, max_order));
  }

  kick = atan(sqrt(sqr(tan(xkick)) + sqr(tan(ykick))));
  tilt += atan2(tan(ykick), tan(xkick));

  return corrector_matrix(length, kick, tilt, b2, 1.0, do_edges, max_order);
}

/* routine: corrector_matrix()
 * purpose: return matrix for orbit corrector.
 */

VMATRIX *corrector_matrix(
  double length, /* straight-through length--i.e., not arc length */
  double kick,   /* actually an angle */
  double tilt,
  double b2, /* By = Bo*(1+b2*x^2) */
  double calibration,
  long do_edges,
  long max_order) {
  VMATRIX *M, *Mtot, *Medge, *tmp;
  double h;
  double *C, **R, ***T;

  log_entry("corrector_matrix");

  kick *= calibration;

  if (kick == 0) {
    log_exit("corrector_matrix");
    return (drift_matrix(length, max_order));
  }

  if (length == 0) {
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, 1);
    M->C[0] = M->C[2] = M->C[3] = M->C[4] = M->C[5] = 0;
    M->C[1] = kick;
    M->R[0][0] = M->R[1][1] = M->R[2][2] = M->R[3][3] = M->R[4][4] = M->R[5][5] = 1;
    M->R[1][5] = -sin(kick);
  } else {
    double s1, K, e2;
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, max_order);
    s1 = length;
    K = sin(kick) / length;
    e2 = b2;
    C = M->C;
    R = M->R;
    R[4][4] = R[5][5] = 1;
    C[0] = pow(e2, 3) * pow(K, 7) * pow(s1, 14) / 2620800.0 + pow(e2, 2) * pow(K, 5) * pow(s1, 10) / 10800.0 + e2 * pow(K, 3) * pow(s1, 6) / 120.0 + K * pow(s1, 2) / 2.0;
    C[1] = pow(e2, 3) * pow(K, 7) * pow(s1, 13) / 187200.0 + pow(e2, 2) * pow(K, 5) * pow(s1, 9) / 1080.0 + e2 * pow(K, 3) * pow(s1, 5) / 20.0 + K * s1;
    C[2] = 0;
    C[3] = 0;
    C[4] = pow(e2, 6) * pow(K, 14) * pow(s1, 27) / 1.89236736E+12 + pow(e2, 5) * pow(K, 12) * pow(s1, 23) / 4.650048E+9 + 211.0 * pow(e2, 4) * pow(K, 10) * pow(s1, 19) / 5.762016E+9 + 29.0 * pow(e2, 3) * pow(K, 8) * pow(s1, 15) / 8424000.0 + 47.0 * pow(e2, 2) * pow(K, 6) * pow(s1, 11) / 237600.0 + e2 * pow(K, 4) * pow(s1, 7) / 140.0 + pow(K, 2) * pow(s1, 3) / 6.0 + s1;
    R[0][0] = pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 95040.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 560.0 + e2 * pow(K, 2) * pow(s1, 4) / 12.0 + 1;
    R[0][1] = pow(e2, 3) * pow(K, 6) * pow(s1, 13) / 187200.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 9) / 1080.0 + e2 * pow(K, 2) * pow(s1, 5) / 20.0 + s1;
    R[0][2] = 0;
    R[0][3] = 0;
    R[0][5] = -pow(e2, 3) * pow(K, 7) * pow(s1, 14) / 374400.0 - pow(e2, 2) * pow(K, 5) * pow(s1, 10) / 2160.0 - e2 * pow(K, 3) * pow(s1, 6) / 40.0 - K * pow(s1, 2) / 2.0;
    R[1][0] = pow(e2, 3) * pow(K, 6) * pow(s1, 11) / 7920.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 7) / 70.0 + e2 * pow(K, 2) * pow(s1, 3) / 3.0;
    R[1][1] = pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 14400.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 120.0 + e2 * pow(K, 2) * pow(s1, 4) / 4.0 + 1;
    R[1][2] = 0;
    R[1][3] = 0;
    R[1][5] = (-7.0) * pow(e2, 3) * pow(K, 7) * pow(s1, 13) / 187200.0 - pow(e2, 2) * pow(K, 5) * pow(s1, 9) / 216.0 + (-3.0) * e2 * pow(K, 3) * pow(s1, 5) / 20.0 - K * s1;
    R[2][0] = 0;
    R[2][1] = 0;
    R[2][2] = pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 95040.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 840.0 - e2 * pow(K, 2) * pow(s1, 4) / 12.0 + 1;
    R[2][3] = pow(e2, 3) * pow(K, 6) * pow(s1, 13) / 187200.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 9) / 2160.0 - e2 * pow(K, 2) * pow(s1, 5) / 20.0 + s1;
    R[2][5] = 0;
    R[3][0] = 0;
    R[3][1] = 0;
    R[3][2] = pow(e2, 3) * pow(K, 6) * pow(s1, 11) / 7920.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 7) / 105.0 - e2 * pow(K, 2) * pow(s1, 3) / 3.0;
    R[3][3] = pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 14400.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 240.0 - e2 * pow(K, 2) * pow(s1, 4) / 4.0 + 1;
    R[3][5] = 0;
    R[4][0] = pow(e2, 6) * pow(K, 13) * pow(s1, 25) / 3.70656E+10 + 47.0 * pow(e2, 5) * pow(K, 11) * pow(s1, 21) / 5.108102999999999E+9 + 461.0 * pow(e2, 4) * pow(K, 9) * pow(s1, 17) / 3.675672E+8 + 2867.0 * pow(e2, 3) * pow(K, 7) * pow(s1, 13) / 3.24324E+7 + 13.0 * pow(e2, 2) * pow(K, 5) * pow(s1, 9) / 3780.0 + e2 * pow(K, 3) * pow(s1, 5) / 15.0;
    R[4][1] = pow(e2, 6) * pow(K, 13) * pow(s1, 26) / 7.008768E+10 + pow(e2, 5) * pow(K, 11) * pow(s1, 22) / 2.02176E+8 + 211.0 * pow(e2, 4) * pow(K, 9) * pow(s1, 18) / 3.03264E+8 + 29.0 * pow(e2, 3) * pow(K, 7) * pow(s1, 14) / 561600.0 + 47.0 * pow(e2, 2) * pow(K, 5) * pow(s1, 10) / 21600.0 + e2 * pow(K, 3) * pow(s1, 6) / 20.0 + K * pow(s1, 2) / 2.0;
    R[4][2] = 0;
    R[4][3] = 0;
    R[4][5] = (-7.0) * pow(e2, 6) * pow(K, 14) * pow(s1, 27) / 9.4618368E+11 - pow(e2, 5) * pow(K, 12) * pow(s1, 23) / 3.87504E+8 + (-211.0) * pow(e2, 4) * pow(K, 10) * pow(s1, 19) / 5.762016E+8 + (-29.0) * pow(e2, 3) * pow(K, 8) * pow(s1, 15) / 1053000.0 + (-47.0) * pow(e2, 2) * pow(K, 6) * pow(s1, 11) / 39600.0 - e2 * pow(K, 4) * pow(s1, 7) / 35.0 - pow(K, 2) * pow(s1, 3) / 3.0;
    if (max_order > 1) {
      T = M->T;
      T[0][0][0] = 1.697530864197531E-4 * pow(e2, 3) * pow(K, 5) * pow(s1, 10) + .0222222222222222 * pow(e2, 2) * pow(K, 3) * pow(s1, 6) + 0.5 * e2 * K * pow(s1, 2);
      T[0][1][0] = pow(e2, 3) * pow(K, 5) * pow(s1, 11) / 7920.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 7) / 70.0 + e2 * K * pow(s1, 3) / 3.0;
      T[0][1][1] = 2.946127946127946E-5 * pow(e2, 3) * pow(K, 5) * pow(s1, 12) + .0032738095238095 * pow(e2, 2) * pow(K, 3) * pow(s1, 8) + .0833333333333333 * e2 * K * pow(s1, 4);
      T[0][2][0] = 0;
      T[0][2][1] = 0;
      T[0][2][2] = -1.697530864197531E-4 * pow(e2, 3) * pow(K, 5) *
                     pow(s1, 10) -
                   .0111111111111111 * pow(e2, 2) * pow(K, 3) * pow(s1, 6) - 0.5 * e2 * K * pow(s1, 2);
      T[0][3][0] = 0;
      T[0][3][1] = 0;
      T[0][3][2] = -pow(e2, 3) * pow(K, 5) * pow(s1, 11) / 7920.0 - pow(e2, 2) * pow(K, 3) * pow(s1, 7) / 630.0 - e2 * K * pow(s1, 3) / 3.0;
      T[0][3][3] = -2.946127946127946E-5 * pow(e2, 3) * pow(K, 5) *
                     pow(s1, 12) +
                   2.976190476190476E-4 * pow(e2, 2) * pow(K, 3) *
                     pow(s1, 8) -
                   .0833333333333333 * e2 * K * pow(s1, 4);
      T[0][5][0] = -pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 15840.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 140.0 - e2 * pow(K, 2) * pow(s1, 4) / 6.0;
      T[0][5][1] = -pow(e2, 3) * pow(K, 6) * pow(s1, 13) / 31200.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 9) / 270.0 - e2 * pow(K, 2) * pow(s1, 5) / 10.0;
      T[0][5][2] = 0;
      T[0][5][3] = 0;
      T[0][5][5] = 9.157509157509157E-6 * pow(e2, 3) * pow(K, 7) * pow(s1, 14) + .0012037037037037 * pow(e2, 2) * pow(K, 5) * pow(s1, 10) + 0.05 * e2 * pow(K, 3) * pow(s1, 6) + 0.5 * K * pow(s1, 2);
      T[1][0][0] = .0016975308641975 * pow(e2, 3) * pow(K, 5) * pow(s1, 9) + .1333333333333333 * pow(e2, 2) * pow(K, 3) * pow(s1, 5) + e2 * K * s1;
      T[1][1][0] = pow(e2, 3) * pow(K, 5) * pow(s1, 10) / 720.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 6) / 10.0 + e2 * K * pow(s1, 2);
      T[1][1][1] = 3.535353535353535E-4 * pow(e2, 3) * pow(K, 5) * pow(s1, 11) + .0261904761904762 * pow(e2, 2) * pow(K, 3) * pow(s1, 7) + .3333333333333333 * e2 * K * pow(s1, 3);
      T[1][2][0] = 0;
      T[1][2][1] = 0;
      T[1][2][2] = -.0016975308641975 * pow(e2, 3) * pow(K, 5) * pow(s1, 9) - .0666666666666667 * pow(e2, 2) * pow(K, 3) * pow(s1, 5) - 1.0 * e2 * K * s1;
      T[1][3][0] = 0;
      T[1][3][1] = 0;
      T[1][3][2] = -pow(e2, 3) * pow(K, 5) * pow(s1, 10) / 720.0 - pow(e2, 2) * pow(K, 3) * pow(s1, 6) / 90.0 - e2 * K * pow(s1, 2);
      T[1][3][3] = -3.535353535353535E-4 * pow(e2, 3) * pow(K, 5) *
                     pow(s1, 11) +
                   .0023809523809524 * pow(e2, 2) * pow(K, 3) *
                     pow(s1, 7) -
                   .3333333333333333 * e2 * K * pow(s1, 3);
      T[1][5][0] = -pow(e2, 3) * pow(K, 6) * pow(s1, 11) / 1320.0 + (-2.0) * pow(e2, 2) * pow(K, 4) * pow(s1, 7) / 35.0 + (-2.0) * e2 * pow(K, 2) * pow(s1, 3) / 3.0;
      T[1][5][1] = -pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 2400.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 30.0 - e2 * pow(K, 2) * pow(s1, 4) / 2.0;
      T[1][5][2] = 0;
      T[1][5][3] = 0;
      T[1][5][5] = 1.282051282051282E-4 * pow(e2, 3) * pow(K, 7) * pow(s1, 13) + 0.012037037037037 * pow(e2, 2) * pow(K, 5) * pow(s1, 9) + 0.3 * e2 * pow(K, 3) * pow(s1, 5) + K * s1;
      T[2][0][0] = 0;
      T[2][1][0] = 0;
      T[2][1][1] = 0;
      T[2][2][0] = 11.0 * pow(e2, 3) * pow(K, 5) * pow(s1, 10) / 32400.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 6) / 30.0 - e2 * K * pow(s1, 2);
      T[2][2][1] = pow(e2, 3) * pow(K, 5) * pow(s1, 11) / 7920.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 7) / 105.0 - e2 * K * pow(s1, 3) / 3.0;
      T[2][2][2] = 0;
      T[2][3][0] = pow(e2, 3) * pow(K, 5) * pow(s1, 11) / 7920.0 + 2.0 * pow(e2, 2) * pow(K, 3) * pow(s1, 7) / 315.0 - e2 * K * pow(s1, 3) / 3.0;
      T[2][3][1] = 7.0 * pow(e2, 3) * pow(K, 5) * pow(s1, 12) / 118800.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 8) / 336.0 - e2 * K * pow(s1, 4) / 6.0;
      T[2][3][2] = 0;
      T[2][3][3] = 0;
      T[2][5][0] = 0;
      T[2][5][1] = 0;
      T[2][5][2] = -pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 15840.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 210.0 + e2 * pow(K, 2) * pow(s1, 4) / 6.0;
      T[2][5][3] = -pow(e2, 3) * pow(K, 6) * pow(s1, 13) / 31200.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 9) / 540.0 + e2 * pow(K, 2) * pow(s1, 5) / 10.0;
      T[2][5][5] = 0;
      T[3][0][0] = 0;
      T[3][1][0] = 0;
      T[3][1][1] = 0;
      T[3][2][0] = 11.0 * pow(e2, 3) * pow(K, 5) * pow(s1, 9) / 3240.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 5) / 5.0 - 2 * e2 * K * s1;
      T[3][2][1] = pow(e2, 3) * pow(K, 5) * pow(s1, 10) / 720.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 6) / 15.0 - e2 * K * pow(s1, 2);
      T[3][2][2] = 0;
      T[3][3][0] = pow(e2, 3) * pow(K, 5) * pow(s1, 10) / 720.0 + 2.0 * pow(e2, 2) * pow(K, 3) * pow(s1, 6) / 45.0 - e2 * K * pow(s1, 2);
      T[3][3][1] = 7.0 * pow(e2, 3) * pow(K, 5) * pow(s1, 11) / 9900.0 + pow(e2, 2) * pow(K, 3) * pow(s1, 7) / 42.0 + (-2.0) * e2 * K * pow(s1, 3) / 3.0;
      T[3][3][2] = 0;
      T[3][3][3] = 0;
      T[3][5][0] = 0;
      T[3][5][1] = 0;
      T[3][5][2] = -pow(e2, 3) * pow(K, 6) * pow(s1, 11) / 1320.0 + (-4.0) * pow(e2, 2) * pow(K, 4) * pow(s1, 7) / 105.0 + 2.0 * e2 * pow(K, 2) * pow(s1, 3) / 3.0;
      T[3][5][3] = -pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 2400.0 - pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 60.0 + e2 * pow(K, 2) * pow(s1, 4) / 2.0;
      T[3][5][5] = 0;
      T[4][0][0] = 7.40831832546076E-10 * pow(e2, 6) * pow(K, 12) *
                     pow(s1, 23) +
                   2.151468606959185E-7 * pow(e2, 5) * pow(K, 10) *
                     pow(s1, 19) +
                   2.385357147261909E-5 * pow(e2, 4) * pow(K, 8) *
                     pow(s1, 15) +
                   .0012774571107904 * pow(e2, 3) * pow(K, 6) *
                     pow(s1, 11) +
                   .0341269841269841 * pow(e2, 2) * pow(K, 4) *
                     pow(s1, 7) +
                   .3333333333333333 * e2 * pow(K, 2) * pow(s1, 3);
      T[4][1][0] = pow(e2, 6) * pow(K, 12) * pow(s1, 24) / 1.482624E+9 + 47.0 * pow(e2, 5) * pow(K, 10) * pow(s1, 20) / 2.43243E+8 + 461.0 * pow(e2, 4) * pow(K, 8) * pow(s1, 16) / 2.16216E+7 + 2867.0 * pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 2494800.0 + 13.0 * pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 420.0 + e2 * pow(K, 2) * pow(s1, 4) / 3.0;
      T[4][1][1] = 1.71992359492359E-10 * pow(e2, 6) * pow(K, 12) *
                     pow(s1, 25) +
                   4.980750681808883E-8 * pow(e2, 5) * pow(K, 10) *
                     pow(s1, 21) +
                   5.634774629872669E-6 * pow(e2, 4) * pow(K, 8) *
                     pow(s1, 17) +
                   3.172676089342756E-4 * pow(e2, 3) * pow(K, 6) *
                     pow(s1, 13) +
                   .0091600529100529 * pow(e2, 2) * pow(K, 4) *
                     pow(s1, 9) +
                   .1166666666666667 * e2 * pow(K, 2) * pow(s1, 5) + 0.5 * s1;
      T[4][2][0] = 0;
      T[4][2][1] = 0;
      T[4][2][2] = -4.76904938184399E-11 * pow(e2, 6) * pow(K, 12) *
                     pow(s1, 23) -
                   3.81795784654784E-8 * pow(e2, 5) * pow(K, 10) *
                     pow(s1, 19) -
                   9.91219245187499E-6 * pow(e2, 4) * pow(K, 8) *
                     pow(s1, 15) -
                   8.301266634599969E-4 * pow(e2, 3) * pow(K, 6) *
                     pow(s1, 11) -
                   .0087301587301587 * pow(e2, 2) * pow(K, 4) *
                     pow(s1, 7) -
                   .3333333333333333 * e2 * pow(K, 2) * pow(s1, 3);
      T[4][3][0] = 0;
      T[4][3][1] = 0;
      T[4][3][2] = pow(e2, 6) * pow(K, 12) * pow(s1, 24) / 1.7791488E+10 + (-1229.0) * pow(e2, 5) * pow(K, 10) * pow(s1, 20) / 1.5567552E+11 + (-487.0) * pow(e2, 4) * pow(K, 8) * pow(s1, 16) / 7.783776E+7 + (-5417.0) * pow(e2, 3) * pow(K, 6) * pow(s1, 12) / 9979200.0 + pow(e2, 2) * pow(K, 4) * pow(s1, 8) / 252.0 - e2 * pow(K, 2) * pow(s1, 4) / 3.0;
      T[4][3][3] = 2.09088750755417E-11 * pow(e2, 6) * pow(K, 12) *
                     pow(s1, 25) -
                   1.203663571388439E-9 * pow(e2, 5) * pow(K, 10) *
                     pow(s1, 21) -
                   1.525492027943008E-6 * pow(e2, 4) * pow(K, 8) *
                     pow(s1, 17) -
                   1.165655332321999E-4 * pow(e2, 3) * pow(K, 6) *
                     pow(s1, 13) +
                   .0023478835978836 * pow(e2, 2) * pow(K, 4) *
                     pow(s1, 9) -
                   .1166666666666667 * e2 * pow(K, 2) * pow(s1, 5) + 0.5 * s1;
      T[4][5][0] = -pow(e2, 6) * pow(K, 13) * pow(s1, 25) / 2.8512E+9 + (-47.0) * pow(e2, 5) * pow(K, 11) * pow(s1, 21) / 4.64373E+8 + (-461.0) * pow(e2, 4) * pow(K, 9) * pow(s1, 17) / 4.08408E+7 + (-2867.0) * pow(e2, 3) * pow(K, 7) * pow(s1, 13) / 4633200.0 + (-13.0) * pow(e2, 2) * pow(K, 5) * pow(s1, 9) / 756.0 - e2 * pow(K, 3) * pow(s1, 5) / 5.0;
      T[4][5][1] = -pow(e2, 6) * pow(K, 13) * pow(s1, 26) / 5.39136E+9 + (-11.0) * pow(e2, 5) * pow(K, 11) * pow(s1, 22) / 2.02176E+8 + (-211.0) * pow(e2, 4) * pow(K, 9) * pow(s1, 18) / 3.3696E+7 + (-203.0) * pow(e2, 3) * pow(K, 7) * pow(s1, 14) / 561600.0 + (-47.0) * pow(e2, 2) * pow(K, 5) * pow(s1, 10) / 4320.0 + (-3.0) * e2 * pow(K, 3) * pow(s1, 6) / 20.0 - K * pow(s1, 2) / 2.0;
      T[4][5][2] = 0;
      T[4][5][3] = 0;
      T[4][5][5] = 5.12585463321456E-11 * pow(e2, 6) * pow(K, 14) *
                     pow(s1, 27) +
                   1.548371113588505E-8 * pow(e2, 5) * pow(K, 12) *
                     pow(s1, 23) +
                   1.867575515236334E-6 * pow(e2, 4) * pow(K, 10) *
                     pow(s1, 19) +
                   1.163342830009497E-4 * pow(e2, 3) * pow(K, 8) *
                     pow(s1, 15) +
                   .0039856902356902 * pow(e2, 2) * pow(K, 6) *
                     pow(s1, 11) +
                   .0714285714285714 * e2 * pow(K, 4) * pow(s1, 7) + 0.5 * pow(K, 2) * pow(s1, 3);
    }

    if (do_edges) {
      /* double arc; */
      /* arc = length*kick/sin(kick); */
      h = kick / length;
      Mtot = tmalloc(sizeof(*Mtot));
      initialize_matrices(Mtot, Mtot->order = max_order);
      Medge = edge_matrix(0.0, h, 0.0, 0.0, -1, 0.0, max_order, 1, 0, length);
      concat_matrices(Mtot, M, Medge, 0);
      tmp = Mtot;
      Mtot = M;
      M = tmp;
      free_matrices(Medge);
      tfree(Medge);
      Medge = NULL;

      Medge = edge_matrix(kick, h, 0.0, 0.0, 1, 0.0, max_order, 1, 0, length);
      concat_matrices(Mtot, Medge, M, 0);
      tmp = Mtot;
      Mtot = M;
      M = tmp;
      free_matrices(Medge);
      tfree(Medge);
      Medge = NULL;
      free_matrices(Mtot);
      tfree(Mtot);
      Mtot = NULL;
    }
  }

  if (tilt)
    tilt_matrices(M, tilt);

  log_exit("corrector_matrix");
  return (M);
}

/* routine: sbend_matrix
 * purpose: second-order sector bending magnet matrix calculation
 * 
 * Michael Borland, 1991. 
 */

VMATRIX *sbend_matrix(
  double t0, /* length of central trajectory with no errors */
  double h,  /* curvature of bend with no errors--a positive quantity! */
  double ha, /* actual curvature of bend */
  /* in midplane:  B = Bo*ha/h*(1 - n*h*x + beta*(h*x)^2 + gamma*(h*x)^3, h = e*Bo/Po */
  double nh,      /* (field index)*h       K1 = -n*h^2      */
  double betah2,  /* (setupole term)*h^2   K2 = 2*beta*h^3  */
  double gammah3, /* (octupole term)*h^3   K3 = 6*gamma*h^4 */
  long order) {
  double kx, cx, sx, kx2, kx4, kx6, cx2, cx3, cx4, sx2, sx3, sx4;
  double ky, cy, sy, ky2, ky4, cy2, cy3, sy2, sy3;
  /* double cy4, sy4; */
  double h2, ha2;
  double e111, e122, e133, e144, e161, e166, e331, e342, e363;
  double i11111, i21111, i12121, i22121, i11211, i21211, i12221, i22221, i11611, i21611, i12621, i22621, i16611,
    i26611, i11112, i21112, i12122, i22122, i11212, i21212, i12222, i22222, i11612, i21612, i12622, i22622,
    i16612, i26612, i13333, i23333, i14343, i24343, i13433, i23433, i14443, i24443, i13334, i23334, i14344,
    i24344, i13434, i23434, i14444, i24444, i11116, i21116, i12126, i22126, i16116, i26116, i11216, i21216,
    i12226, i22226, i16216, i26216, i11616, i21616, i12626, i22626, i16616, i26616, i16666, i26666, i33311,
    i43311, i34321, i44321, i33411, i43411, i34421, i44421, i33312, i43312, i34322, i44322, i33412, i43412,
    i34422, i44422, i33113, i43113, i34123, i44123, i33213, i43213, i34223, i44223, i33613, i43613, i34623,
    i44623, i36633, i46633, i33114, i43114, i34124, i44124, i33214, i43214, i34224, i44224, i33614, i43614,
    i34624, i44624, i36634, i46634, i33316, i43316, i34326, i44326, i36336, i46336, i33416, i43416, i34426,
    i44426, i36436, i46436;
  double small3;
  double t0_2, t0_3, t0_4, t0_5, t0_6;
  long ky2_is_zero, kx2_is_zero;
  double *C, **R, ***T;
  /* double ****U; */
  VMATRIX *M;

  log_entry("sbend_matrix");

  i11111 = i21111 = i12121 = i22121 = i11211 = i21211 = i12221 = i22221 = i11611 = i21611 = i12621 = i22621 = i16611 = 0;
  i26611 = i11112 = i21112 = i12122 = i22122 = i11212 = i21212 = i12222 = i22222 = i11612 = i21612 = i12622 = i22622 = 0;
  i16612 = i26612 = i13333 = i23333 = i14343 = i24343 = i13433 = i23433 = i14443 = i24443 = i13334 = i23334 = i14344 = 0;
  i24344 = i13434 = i23434 = i14444 = i24444 = i11116 = i21116 = i12126 = i22126 = i16116 = i26116 = i11216 = i21216 = 0;
  i12226 = i22226 = i16216 = i26216 = i11616 = i21616 = i12626 = i22626 = i16616 = i26616 = i16666 = i26666 = i33311 = 0;
  i43311 = i34321 = i44321 = i33411 = i43411 = i34421 = i44421 = i33312 = i43312 = i34322 = i44322 = i33412 = i43412 = 0;
  i34422 = i44422 = i33113 = i43113 = i34123 = i44123 = i33213 = i43213 = i34223 = i44223 = i33613 = i43613 = i34623 = 0;
  i44623 = i36633 = i46633 = i33114 = i43114 = i34124 = i44124 = i33214 = i43214 = i34224 = i44224 = i33614 = i43614 = 0;
  i34624 = i44624 = i36634 = i46634 = i33316 = i43316 = i34326 = i44326 = i36336 = i46336 = i33416 = i43416 = i34426 = 0;
  i44426 = i36436 = i46436 = 0;

#ifdef DEBUG
  printf("\n*** sbend_matrix called:\n  t0=%.16le, h=%.16le, ha=%.16le, n*h=%.16le, beta*h^2=%.16le, gamma*h^3=%.16le, order=%ld\n",
         t0, h, ha, nh, betah2, gammah3, order);
  fflush(stdout);
#endif

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(2, order));

  C = M->C;
  R = M->R;
  T = M->T;
  /* U = M->Q; */

  h2 = h * h;
  ha2 = ha * ha;

  kx2 = -(h2 - 2 * h * ha + nh * ha);
  ky2 = nh * ha;
  ky2_is_zero = (ky2 == 0);
  kx2_is_zero = (kx2 == 0);

  small3 = pow(1e-16, 1. / 3.);

  if (sqrt(fabs(kx2)) * fabs(t0) < small3) {
    kx2 = sqr(small3 / t0);
    if (kx2)
      kx2 = SIGN(kx2) * kx2;
  }

  if (kx2 > 0.0) {
    kx = sqrt(kx2);
    cx = cos(kx * t0);
    sx = sin(kx * t0) / kx;
  } else if (kx2 < 0.0) {
    kx = sqrt(-kx2);
    cx = cosh(kx * t0);
    sx = sinh(kx * t0) / kx;
  } else {
    /* This code should actual never be executed, but it keeps the compiler warnings at bay */
    kx = 0;
    cx = 1;
    sx = 1;
  }

  if (FABS(ky2) < small3) {
    ky2 = sqr(small3 / t0);
    if (ky2)
      ky2 = SIGN(ky2) * ky2;
  }
  if (ky2 > 0.0) {
    ky = sqrt(ky2);
    cy = cos(ky * t0);
    sy = sin(ky * t0) / ky;
  } else if (ky2 < 0.0) {
    ky = sqrt(-ky2);
    cy = cosh(ky * t0);
    sy = sinh(ky * t0) / ky;
  } else {
    /* This code should actual never be executed, but it keeps the compiler warnings at bay */
    ky = 0;
    cy = 1;
    sy = 1;
  }

#ifdef DEBUG
  printf("kx = %.16le, ky = %.16le\ncx = %.16le, cy = %.16le\nsx = %.16le, sy=%.16le\n",
         kx, ky, cx, cy, sx, sy);
  fflush(stdout);
  printf("kx2 = %.16le, ky2 = %.16le\n", kx2, ky2);
  fflush(stdout);
#endif

  C[0] = C[1] = C[2] = C[3] = C[5] = 0;
  C[4] = t0;
  if (h != 0) {
    C[0] = -(h - ha) * (cx - 1) / kx2;
    C[1] = (h - ha) * sx;
    R[0][0] = R[1][1] = cx;
    R[0][1] = sx;
    R[0][2] = R[0][3] = R[0][4] = 0;
    R[0][5] = -ha * (cx - 1) / kx2;
    R[1][0] = -sx * kx2;
    R[1][2] = R[1][3] = R[1][4] = 0;
    R[1][5] = sx * ha;
    R[2][2] = R[3][3] = cy;
    R[2][3] = sy;
    R[2][0] = R[2][1] = R[2][4] = R[2][5] = 0;
    R[3][2] = -sy * ky2;
    R[3][0] = R[3][1] = R[3][4] = R[3][5] = 0;
    R[4][0] = h * sx;
    R[4][1] = -(h * (cx - 1)) / kx2;
    R[4][2] = R[4][3] = 0;
    R[4][5] = (h * ha * (t0 - sx)) / kx2;
    C[4] += R[4][5] * (1 - ha / h);
  } else {
    /* steering corrector */
    C[1] = -tan(ha * t0);
    C[0] = -(1 - cos(ha * t0)) / ha;
    R[0][0] = R[1][1] = R[2][2] = R[3][3] = 1;
    R[0][1] = R[2][3] = t0;
    R[0][5] = C[0] + t0 * sin(ha * t0);
    R[1][5] = t0 * ha * cos(ha * t0);
  }
  R[5][0] = R[5][1] = R[5][2] = R[5][3] = R[5][4] = 0;
  R[4][4] = R[5][5] = 1;

  if (M->order > 1) {
    /* second-order matrix elements */

    kx4 = kx2 * kx2;
    kx6 = kx4 * kx2;
    cx2 = cx * cx;
    cx3 = cx2 * cx;
    cx4 = cx2 * cx2;
    sx2 = sx * sx;
    sx3 = sx2 * sx;
    sx4 = sx2 * sx2;

    ky4 = ky2 * ky2;
    cy2 = cy * cy;
    cy3 = cy2 * cy;
    /* cy4 = cy2*cy2; */
    sy2 = sy * sy;
    sy3 = sy2 * sy;
    /* sy4 = sy2*sy2; */

    t0_2 = sqr(t0);
    t0_3 = t0_2 * t0;
    t0_4 = t0_3 * t0;
    t0_5 = t0_4 * t0;
    t0_6 = t0_5 * t0;

    e111 = -ha * (h2 - 2 * h * nh + betah2);
    e122 = (4 * h - 3 * ha) / 2;
    e133 = -(ha * (h * nh - 2 * betah2)) / 2;
    e144 = -ha / 2;
    e161 = ha * (2 * h - nh);
    e166 = -ha;
    e331 = -2 * ha * (h * nh - betah2);
    e342 = 2 * h - ha;
    e363 = nh * ha;

    /* expressions for integrals that go into second-order matrix elements--see notebook 1, page 12.
         */
    if (kx2_is_zero && ky2_is_zero) {
      i11111 = t0_2 / 2;
      i21111 = t0;
      i11211 = t0_3 / 3;
      i21211 = t0_2;
      i11611 = (t0_4 * ha) / 12;
      i21611 = (t0_3 * ha) / 3;
      i16611 = t0_2 / 2;
      i26611 = t0;
      i11112 = t0_3 / 3;
      i21112 = t0_2;
      i11212 = t0_4 / 12;
      i21212 = t0_3 / 3;
      i12222 = t0_2 / 2;
      i22222 = t0;
      i11612 = (t0_5 * ha) / 20;
      i21612 = (t0_4 * ha) / 4;
      i12622 = (t0_3 * ha) / 3;
      i22622 = t0_2 * ha;
      i16612 = t0_3 / 6;
      i26612 = t0_2 / 2;
      i13333 = t0_2 / 2;
      i13433 = t0_3 / 3;
      i23433 = t0_2;
      i13334 = t0_3 / 3;
      i23334 = t0_2;
      i13434 = t0_4 / 12;
      i23434 = t0_3 / 3;
      i14444 = t0_2 / 2;
      i24444 = t0;
      i11116 = (t0_4 * ha) / 12;
      i21116 = (t0_3 * ha) / 3;
      i16116 = t0_2 / 2;
      i26116 = t0;
      i11216 = (t0_5 * ha) / 20;
      i21216 = (t0_4 * ha) / 4;
      i12226 = (t0_3 * ha) / 3;
      i22226 = t0_2 * ha;
      i16216 = t0_3 / 6;
      i26216 = t0_2 / 2;
      i11616 = (t0_6 * ha2) / 120;
      i21616 = (t0_5 * ha2) / 20;
      i12626 = (t0_4 * ha2) / 12;
      i22626 = (t0_3 * ha2) / 3;
      i16616 = (t0_4 * ha) / 24;
      i26616 = (t0_3 * ha) / 6;
      i16666 = t0_2 / 2;
      i26666 = t0;
      i33311 = t0_2 / 2;
      i43311 = t0;
      i33411 = t0_3 / 6;
      i43411 = t0_2 / 2;
      i33312 = t0_3 / 6;
      i43312 = t0_2 / 2;
      i33412 = t0_4 / 12;
      i43412 = t0_3 / 3;
      i34422 = t0_2 / 2;
      i44422 = t0;
      i33113 = t0_2 / 2;
      i43113 = t0;
      i33213 = t0_3 / 6;
      i43213 = t0_2 / 2;
      i33613 = (t0_4 * ha) / 24;
      i43613 = (t0_3 * ha) / 6;
      i36633 = t0_2 / 2;
      i46633 = t0;
      i33114 = t0_3 / 6;
      i43114 = t0_2 / 2;
      i33214 = t0_4 / 12;
      i43214 = t0_3 / 3;
      i34224 = t0_2 / 2;
      i44224 = t0;
      i33614 = (t0_5 * ha) / 40;
      i43614 = (t0_4 * ha) / 8;
      i34624 = (t0_3 * ha) / 6;
      i44624 = (t0_2 * ha) / 2;
      i36634 = t0_3 / 6;
      i46634 = t0_2 / 2;
      i33316 = (t0_4 * ha) / 24;
      i43316 = (t0_3 * ha) / 6;
      i36336 = t0_2 / 2;
      i46336 = t0;
      i33416 = (t0_5 * ha) / 40;
      i43416 = (t0_4 * ha) / 8;
      i34426 = (t0_3 * ha) / 6;
      i44426 = (t0_2 * ha) / 2;
      i36436 = t0_3 / 6;
      i46436 = t0_2 / 2;
    } else if (ky2_is_zero) {
      i11111 = -(cx - cx2 - 3 * kx2 * sx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx2);
      i21111 = -(sx * (-1 - 4 * cx + 2 * cx3 + 2 * cx * kx2 * sx2)) / 3;
      i12121 = (-2 * cx + 2 * cx2 + cx2 * kx2 * sx2 + kx4 * sx4) / 3;
      i22121 = 2 * kx2 * sx * (1 - 2 * cx + cx3 + cx * kx2 * sx2) / 3;
      i11211 = -2 * sx * (-1 + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i21211 = -2 * (-cx + cx4 - kx4 * sx4) / (3 * kx2);
      i12221 = 2 * sx * (-1 + cx3 + cx * kx2 * sx2) / 3;
      i22221 = 2 * (-cx + cx4 - kx4 * sx4) / 3;
      i11611 = ha * (2 * cx - 2 * cx2 - 6 * kx2 * sx2 + 2 * cx2 * kx2 * sx2 + 2 * kx4 * sx4 + 3 * kx2 * sx * t0) / (3 * kx4);
      i21611 = ha * (sx - 8 * cx * sx + 4 * cx3 * sx + 4 * cx * kx2 * sx3 + 3 * cx * t0) / (3 * kx2);
      i12621 = -2 * ha * (-2 * cx + 2 * cx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx2);
      i22621 = -4 * ha * sx * (1 - 2 * cx + cx3 + cx * kx2 * sx2) / 3;
      i16611 = (-cx + cx3 + cx * kx2 * sx2 + cx2 * kx2 * sx * t0 + kx4 * sx3 * t0) / (2 * kx2);
      i26611 = (sx + cx3 * t0 + cx * kx2 * sx2 * t0) / 2;
      i11112 = -2 * sx * (-1 + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i21112 = -2 * (-cx + cx4 - kx4 * sx4) / (3 * kx2);
      i12122 = 2 * sx * (-1 + cx3 + cx * kx2 * sx2) / 3;
      i22122 = 2 * (-cx + cx4 - kx4 * sx4) / 3;
      i11212 = (-2 * cx + 2 * cx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx4);
      i21212 = 2 * sx * (1 - 2 * cx + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i12222 = -(cx - cx2 - 3 * kx2 * sx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx2);
      i22222 = -(sx * (-1 - 4 * cx + 2 * cx3 + 2 * cx * kx2 * sx2)) / 3;
      i11612 = ha * (-2 * sx + 2 * cx * sx + 3 * cx2 * sx + 3 * kx2 * sx3 - 3 * cx * t0) / (3 * kx4);
      i21612 = ha * (-5 * cx + 2 * cx2 + 3 * cx3 - 2 * kx2 * sx2 + 3 * cx * kx2 * sx2 + 3 * kx2 * sx * t0) / (3 * kx4);
      i12622 = -2 * ha * sx * (-1 + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i22622 = -2 * ha * (-cx + cx4 - kx4 * sx4) / (3 * kx2);
      i16612 = (cx2 * sx + kx2 * sx3 - cx * t0) / (2 * kx2);
      i26612 = (-cx + cx3 + cx * kx2 * sx2 + kx2 * sx * t0) / (2 * kx2);
      i13333 = (-cx + cx2 + kx2 * sx2) / kx2;
      i23333 = sx;
      i14343 = 0;
      i24343 = 0;
      i13433 = 2 * (-sx + cx2 * t0 + kx2 * sx2 * t0) / kx2;
      i23433 = 2 * (-cx + cx2 + kx2 * sx2) / kx2;
      i14443 = 0;
      i24443 = 0;
      i13334 = 2 * (-sx + cx2 * t0 + kx2 * sx2 * t0) / kx2;
      i23334 = 2 * (-cx + cx2 + kx2 * sx2) / kx2;
      i14344 = 0;
      i24344 = 0;
      i13434 = (2 * cx - 2 * cx2 - 2 * kx2 * sx2 + cx2 * kx2 * t0_2 + kx4 * sx2 * t0_2) / kx4;
      i23434 = 2 * (-sx + cx2 * t0 + kx2 * sx2 * t0) / kx2;
      i14444 = (-cx + cx2 + kx2 * sx2) / kx2;
      i24444 = sx;
      i11116 = ha * (2 * cx - 2 * cx2 - 6 * kx2 * sx2 + 2 * cx2 * kx2 * sx2 + 2 * kx4 * sx4 + 3 * kx2 * sx * t0) / (3 * kx4);
      i21116 = ha * (sx - 8 * cx * sx + 4 * cx3 * sx + 4 * cx * kx2 * sx3 + 3 * cx * t0) / (3 * kx2);
      i12126 = -2 * ha * (-2 * cx + 2 * cx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx2);
      i22126 = -4 * ha * sx * (1 - 2 * cx + cx3 + cx * kx2 * sx2) / 3;
      i16116 = (-cx + cx3 + cx * kx2 * sx2 + cx2 * kx2 * sx * t0 + kx4 * sx3 * t0) / (2 * kx2);
      i26116 = (sx + cx3 * t0 + cx * kx2 * sx2 * t0) / 2;
      i11216 = ha * (-2 * sx + 2 * cx * sx + 3 * cx2 * sx + 3 * kx2 * sx3 - 3 * cx * t0) / (3 * kx4);
      i21216 = ha * (-5 * cx + 2 * cx2 + 3 * cx3 - 2 * kx2 * sx2 + 3 * cx * kx2 * sx2 + 3 * kx2 * sx * t0) / (3 * kx4);
      i12226 = -2 * ha * sx * (-1 + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i22226 = -2 * ha * (-cx + cx4 - kx4 * sx4) / (3 * kx2);
      i16216 = (cx2 * sx + kx2 * sx3 - cx * t0) / (2 * kx2);
      i26216 = (-cx + cx3 + cx * kx2 * sx2 + kx2 * sx * t0) / (2 * kx2);
      i11616 = -(ha2 * (4 * cx - 4 * cx2 - 6 * kx2 * sx2 + cx2 * kx2 * sx2 + kx4 * sx4 + 3 * kx2 * sx * t0)) / (3 * kx6);
      i21616 = -(ha2 * (-sx - 4 * cx * sx + 2 * cx3 * sx + 2 * cx * kx2 * sx3 + 3 * cx * t0)) / (3 * kx4);
      i12626 = ha2 * (-2 * cx + 2 * cx2 + cx2 * kx2 * sx2 + kx4 * sx4) / (3 * kx4);
      i22626 = 2 * ha2 * sx * (1 - 2 * cx + cx3 + cx * kx2 * sx2) / (3 * kx2);
      i16616 = -(ha * (cx - 2 * cx2 + cx3 - 2 * kx2 * sx2 + cx * kx2 * sx2 + cx2 * kx2 * sx * t0 + kx4 * sx3 * t0)) / (2 * kx4);
      i26616 = -(ha * (-sx + cx3 * t0 + cx * kx2 * sx2 * t0)) / (2 * kx2);
      i16666 = (-cx + cx2 + kx2 * sx2) / kx2;
      i26666 = sx;
      i33311 = (1 - cx) / kx2;
      i43311 = sx;
      i34321 = 0;
      i44321 = 0;
      i33411 = (2 * sx - t0 - cx * t0) / kx2;
      i43411 = (-1 + cx + kx2 * sx * t0) / kx2;
      i34421 = sx - t0;
      i44421 = -1 + cx;
      i33312 = (-sx + t0) / kx2;
      i43312 = (1 - cx) / kx2;
      i34322 = 0;
      i44322 = 0;
      i33412 = (2 - 2 * cx - kx2 * sx * t0) / kx4;
      i43412 = (sx - cx * t0) / kx2;
      i34422 = (1 - cx) / kx2;
      i44422 = sx;
      i33113 = (1 - cx) / kx2;
      i43113 = sx;
      i34123 = 0;
      i44123 = 0;
      i33213 = (-sx + t0) / kx2;
      i43213 = (1 - cx) / kx2;
      i34223 = 0;
      i44223 = 0;
      i33613 = ha * (-2 + 2 * cx + kx2 * t0_2) / (2 * kx4);
      i43613 = ha * (-sx + t0) / kx2;
      i34623 = 0;
      i44623 = 0;
      i36633 = t0_2 / 2;
      i46633 = t0;
      i33114 = (2 * sx - t0 - cx * t0) / kx2;
      i43114 = (-1 + cx + kx2 * sx * t0) / kx2;
      i34124 = sx - t0;
      i44124 = -1 + cx;
      i33214 = (2 - 2 * cx - kx2 * sx * t0) / kx4;
      i43214 = (sx - cx * t0) / kx2;
      i34224 = (1 - cx) / kx2;
      i44224 = sx;
      i33614 = ha * (-12 * sx + 6 * t0 + 6 * cx * t0 + kx2 * t0_3) / (6 * kx4);
      i43614 = ha * (2 - 2 * cx - 2 * kx2 * sx * t0 + kx2 * t0_2) / (2 * kx4);
      i34624 = ha * (-sx + t0) / kx2;
      i44624 = (1 - cx) * ha / kx2;
      i36634 = t0_3 / 6;
      i46634 = t0_2 / 2;
      i33316 = ha * (-2 + 2 * cx + kx2 * t0_2) / (2 * kx4);
      i43316 = ha * (-sx + t0) / kx2;
      i34326 = 0;
      i44326 = 0;
      i36336 = t0_2 / 2;
      i46336 = t0;
      i33416 = ha * (-12 * sx + 6 * t0 + 6 * cx * t0 + kx2 * t0_3) / (6 * kx4);
      i43416 = ha * (2 - 2 * cx - 2 * kx2 * sx * t0 + kx2 * t0_2) / (2 * kx4);
      i34426 = ha * (-sx + t0) / kx2;
      i44426 = (1 - cx) * ha / kx2;
      i36436 = t0_3 / 6;
      i46436 = t0_2 / 2;
    } else {
      i11111 = -(cx2 * sx2 * kx2 - cx2 + cx + sx4 * kx4 - 3 * sx2 * kx2) / (3 * kx2);
      i21111 = -(sx * (2 * cx3 + 2 * cx * sx2 * kx2 - 4 * cx - 1)) / 3;
      i12121 = (cx2 * sx2 * kx2 + 2 * cx2 - 2 * cx + sx4 * kx4) / 3;
      i22121 = (2 * sx * kx2 * (cx3 + cx * sx2 * kx2 - 2 * cx + 1)) / 3;
      i11211 = -(2 * sx * (cx3 + cx * sx2 * kx2 - 1)) / (3 * kx2);
      i21211 = -(2 * (cx4 - cx - sx4 * kx4)) / (3 * kx2);
      i12221 = (2 * sx * (cx3 + cx * sx2 * kx2 - 1)) / 3;
      i22221 = (2 * (cx4 - cx - sx4 * kx4)) / 3;
      i11611 = (ha * (2 * cx2 * sx2 * kx2 - 2 * cx2 + 2 * cx + 2 * sx4 * kx4 - 6 * sx2 * kx2 + 3 * sx * t0 * kx2)) / (3 * kx4);
      i21611 = (ha * (4 * cx3 * sx + 4 * cx * sx3 * kx2 - 8 * cx * sx + 3 * cx * t0 + sx)) / (3 * kx2);
      i12621 = -(2 * ha * (cx2 * sx2 * kx2 + 2 * cx2 - 2 * cx + sx4 * kx4)) / (3 * kx2);
      i22621 = -(4 * sx * ha * (cx3 + cx * sx2 * kx2 - 2 * cx + 1)) / 3;
      i16611 = (cx3 + cx2 * sx * t0 * kx2 + cx * sx2 * kx2 - cx + sx3 * t0 * kx4) / (2 * kx2);
      i26611 = (cx3 * t0 + cx * sx2 * t0 * kx2 + sx) / 2;
      i11112 = -(2 * sx * (cx3 + cx * sx2 * kx2 - 1)) / (3 * kx2);
      i21112 = -(2 * (cx4 - cx - sx4 * kx4)) / (3 * kx2);
      i12122 = (2 * sx * (cx3 + cx * sx2 * kx2 - 1)) / 3;
      i22122 = (2 * (cx4 - cx - sx4 * kx4)) / 3;
      i11212 = (cx2 * sx2 * kx2 + 2 * cx2 - 2 * cx + sx4 * kx4) / (3 * kx4);
      i21212 = (2 * sx * (cx3 + cx * sx2 * kx2 - 2 * cx + 1)) / (3 * kx2);
      i12222 = -(cx2 * sx2 * kx2 - cx2 + cx + sx4 * kx4 - 3 * sx2 * kx2) / (3 * kx2);
      i22222 = -(sx * (2 * cx3 + 2 * cx * sx2 * kx2 - 4 * cx - 1)) / 3;
      i11612 = (ha * (3 * cx2 * sx + 2 * cx * sx - 3 * cx * t0 + 3 * sx3 * kx2 - 2 * sx)) / (3 * kx4);
      i21612 = (ha * (3 * cx3 + 2 * cx2 + 3 * cx * sx2 * kx2 - 5 * cx - 2 * sx2 * kx2 + 3 * sx * t0 * kx2)) / (3 * kx4);
      i12622 = -(2 * sx * ha * (cx3 + cx * sx2 * kx2 - 1)) / (3 * kx2);
      i22622 = -(2 * ha * (cx4 - cx - sx4 * kx4)) / (3 * kx2);
      i16612 = (cx2 * sx - cx * t0 + sx3 * kx2) / (2 * kx2);
      i26612 = (cx3 + cx * sx2 * kx2 - cx + sx * t0 * kx2) / (2 * kx2);
      i13333 = -(cx2 * sy2 * kx2 * ky2 - cx2 * kx2 + 2 * cx2 * ky2 + cx * kx2 - 2 * cx * ky2 + sx2 * sy2 * kx4 * ky2 - sx2 * kx4 +
                 2 * sx2 * kx2 * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i23333 = -(2 * cx2 * cy * sy * ky2 + 2 * sx2 * cy * sy * kx2 * ky2 - sx * kx2 + 2 * sx * ky2) / (kx2 - 4 * ky2);
      i14343 = (ky4 * (cx2 * sy2 * kx2 - 2 * cx2 + 2 * cx + sx2 * sy2 * kx4 - 2 * sx2 * kx2)) / (kx2 * (kx2 - 4 * ky2));
      i24343 = (2 * ky4 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i13433 = (2 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i23433 = (2 * (cx2 * cy2 - cx2 * sy2 * ky2 - cx + sx2 * cy2 * kx2 - sx2 * sy2 * kx2 * ky2)) / (kx2 - 4 * ky2);
      i14443 = -(2 * ky2 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i24443 = -(2 * ky2 * (cx2 * cy2 - cx2 * sy2 * ky2 - cx + sx2 * cy2 * kx2 - sx2 * sy2 * kx2 * ky2)) / (kx2 - 4 * ky2);
      i13334 = (2 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i23334 = (2 * (cx2 * cy2 - cx2 * sy2 * ky2 - cx + sx2 * cy2 * kx2 - sx2 * sy2 * kx2 * ky2)) / (kx2 - 4 * ky2);
      i14344 = -(2 * ky2 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i24344 = -(2 * ky2 * (cx2 * cy2 - cx2 * sy2 * ky2 - cx + sx2 * cy2 * kx2 - sx2 * sy2 * kx2 * ky2)) / (kx2 - 4 * ky2);
      i13434 = (cx2 * sy2 * kx2 - 2 * cx2 + 2 * cx + sx2 * sy2 * kx4 - 2 * sx2 * kx2) / (kx2 * (kx2 - 4 * ky2));
      i23434 = (2 * (cx2 * cy * sy + sx2 * cy * sy * kx2 - sx)) / (kx2 - 4 * ky2);
      i14444 = -(cx2 * sy2 * kx2 * ky2 - cx2 * kx2 + 2 * cx2 * ky2 + cx * kx2 - 2 * cx * ky2 + sx2 * sy2 * kx4 * ky2 - sx2 * kx4 +
                 2 * sx2 * kx2 * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i24444 = -(2 * cx2 * cy * sy * ky2 + 2 * sx2 * cy * sy * kx2 * ky2 - sx * kx2 + 2 * sx * ky2) / (kx2 - 4 * ky2);
      i11116 = (ha * (2 * cx2 * sx2 * kx2 - 2 * cx2 + 2 * cx + 2 * sx4 * kx4 - 6 * sx2 * kx2 + 3 * sx * t0 * kx2)) / (3 * kx4);
      i21116 = (ha * (4 * cx3 * sx + 4 * cx * sx3 * kx2 - 8 * cx * sx + 3 * cx * t0 + sx)) / (3 * kx2);
      i12126 = -(2 * ha * (cx2 * sx2 * kx2 + 2 * cx2 - 2 * cx + sx4 * kx4)) / (3 * kx2);
      i22126 = -(4 * sx * ha * (cx3 + cx * sx2 * kx2 - 2 * cx + 1)) / 3;
      i16116 = (cx3 + cx2 * sx * t0 * kx2 + cx * sx2 * kx2 - cx + sx3 * t0 * kx4) / (2 * kx2);
      i26116 = (cx3 * t0 + cx * sx2 * t0 * kx2 + sx) / 2;
      i11216 = (ha * (3 * cx2 * sx + 2 * cx * sx - 3 * cx * t0 + 3 * sx3 * kx2 - 2 * sx)) / (3 * kx4);
      i21216 = (ha * (3 * cx3 + 2 * cx2 + 3 * cx * sx2 * kx2 - 5 * cx - 2 * sx2 * kx2 + 3 * sx * t0 * kx2)) / (3 * kx4);
      i12226 = -(2 * sx * ha * (cx3 + cx * sx2 * kx2 - 1)) / (3 * kx2);
      i22226 = -(2 * ha * (cx4 - cx - sx4 * kx4)) / (3 * kx2);
      i16216 = (cx2 * sx - cx * t0 + sx3 * kx2) / (2 * kx2);
      i26216 = (cx3 + cx * sx2 * kx2 - cx + sx * t0 * kx2) / (2 * kx2);
      i11616 = -(ha2 * (cx2 * sx2 * kx2 - 4 * cx2 + 4 * cx + sx4 * kx4 - 6 * sx2 * kx2 + 3 * sx * t0 * kx2)) / (3 * kx6);
      i21616 = -(ha2 * (2 * cx3 * sx + 2 * cx * sx3 * kx2 - 4 * cx * sx + 3 * cx * t0 - sx)) / (3 * kx4);
      i12626 = (ha2 * (cx2 * sx2 * kx2 + 2 * cx2 - 2 * cx + sx4 * kx4)) / (3 * kx4);
      i22626 = (2 * sx * ha2 * (cx3 + cx * sx2 * kx2 - 2 * cx + 1)) / (3 * kx2);
      i16616 = -(ha * (cx3 + cx2 * sx * t0 * kx2 - 2 * cx2 + cx * sx2 * kx2 + cx + sx3 * t0 * kx4 - 2 * sx2 * kx2)) / (2 * kx4);
      i26616 = -(ha * (cx3 * t0 + cx * sx2 * t0 * kx2 - sx)) / (2 * kx2);
      i16666 = (cx2 - cx + sx2 * kx2) / kx2;
      i26666 = sx;
      i33311 = -(cx * cy3 + cx * cy * sy2 * ky2 + 2 * sx * cy2 * sy * ky2 + 2 * sx * sy3 * ky4 - cy) / (kx2 - 4 * ky2);
      i43311 = -(cx * cy2 * sy * ky2 + cx * sy3 * ky4 - sx * cy3 * kx2 + 2 * sx * cy3 * ky2 - sx * cy * sy2 * kx2 * ky2 + 2 * sx * cy * sy2 * ky4 +
                 sy * ky2) /
               (kx2 - 4 * ky2);
      i34321 = -(ky2 * (2 * cx * cy3 + 2 * cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 + sx * sy3 * kx2 * ky2 - 2 * cy)) / (kx2 - 4 * ky2);
      i44321 = -(ky2 * (cx * cy2 * sy * kx2 - 2 * cx * cy2 * sy * ky2 + cx * sy3 * kx2 * ky2 - 2 * cx * sy3 * ky4 - sx * cy3 * kx2 -
                        sx * cy * sy2 * kx2 * ky2 + 2 * sy * ky2)) /
               (kx2 - 4 * ky2);
      i33411 = -(cx * cy2 * sy + cx * sy3 * ky2 - 2 * sx * cy3 - 2 * sx * cy * sy2 * ky2 + sy) / (kx2 - 4 * ky2);
      i43411 = (cx * cy3 + cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 - 2 * sx * cy2 * sy * ky2 + sx * sy3 * kx2 * ky2 - 2 * sx * sy3 * ky4 - cy) / (kx2 -
                                                                                                                                                    4 * ky2);
      i34421 = -(cx * cy2 * sy * kx2 + cx * sy3 * kx2 * ky2 - cx * sy * kx2 + 2 * cx * sy * ky2 - sx * cy * kx2 + sy * kx2 - 2 * sy * ky2) / (kx2 - 4 * ky2);
      i44421 = -(cx * cy3 * kx2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * kx2 + 2 * cx * cy * ky2 - sx * cy2 * sy * kx4 - sx * sy3 * kx4 * ky2 +
                 sx * sy * kx4 - sx * sy * kx2 * ky2 + cy * kx2 - 2 * cy * ky2) /
               (kx2 - 4 * ky2);
      i33312 = (2 * cx * cy2 * sy * ky2 + 2 * cx * sy3 * ky4 - sx * cy3 * kx2 - sx * cy * sy2 * kx2 * ky2 + sy * kx2 - 2 * sy * ky2) / (kx2 * (kx2 - 4 * ky2));
      i43312 = -(cx * cy3 * kx2 - 2 * cx * cy3 * ky2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * sy2 * ky4 + sx * cy2 * sy * kx2 * ky2 +
                 sx * sy3 * kx2 * ky4 - cy * kx2 + 2 * cy * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i34322 = (ky2 * (cx * cy2 * sy + cx * sy3 * ky2 - 2 * sx * cy3 - 2 * sx * cy * sy2 * ky2 + sy)) / (kx2 - 4 * ky2);
      i44322 = -(ky2 * (cx * cy3 + cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 - 2 * sx * cy2 * sy * ky2 + sx * sy3 * kx2 * ky2 - 2 * sx * sy3 * ky4 -
                        cy)) /
               (kx2 - 4 * ky2);
      i33412 = -(2 * cx * cy + 2 * sx * cy2 * sy * kx2 + 2 * sx * sy3 * kx2 * ky2 - sx * sy * kx2 - 2 * cy) / (kx2 * (kx2 - 4 * ky2));
      i43412 = -(2 * cx * cy2 * sy * kx2 + 2 * cx * sy3 * kx2 * ky2 - cx * sy * kx2 - 2 * cx * sy * ky2 + 2 * sx * cy3 * kx2 + 2 * sx * cy * sy2 * kx2 * ky2 -
                 3 * sx * cy * kx2 + 2 * sy * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i34422 = -(cx * cy3 + cx * cy * sy2 * ky2 + 2 * sx * cy2 * sy * ky2 + 2 * sx * sy3 * ky4 - cy) / (kx2 - 4 * ky2);
      i44422 = -(cx * cy2 * sy * ky2 + cx * sy3 * ky4 - sx * cy3 * kx2 + 2 * sx * cy3 * ky2 - sx * cy * sy2 * kx2 * ky2 + 2 * sx * cy * sy2 * ky4 +
                 sy * ky2) /
               (kx2 - 4 * ky2);
      i33113 = -(cx * cy3 + cx * cy * sy2 * ky2 + 2 * sx * cy2 * sy * ky2 + 2 * sx * sy3 * ky4 - cy) / (kx2 - 4 * ky2);
      i43113 = -(cx * cy2 * sy * ky2 + cx * sy3 * ky4 - sx * cy3 * kx2 + 2 * sx * cy3 * ky2 - sx * cy * sy2 * kx2 * ky2 + 2 * sx * cy * sy2 * ky4 +
                 sy * ky2) /
               (kx2 - 4 * ky2);
      i34123 = -(ky2 * (2 * cx * cy3 + 2 * cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 + sx * sy3 * kx2 * ky2 - 2 * cy)) / (kx2 - 4 * ky2);
      i44123 = -(ky2 * (cx * cy2 * sy * kx2 - 2 * cx * cy2 * sy * ky2 + cx * sy3 * kx2 * ky2 - 2 * cx * sy3 * ky4 - sx * cy3 * kx2 -
                        sx * cy * sy2 * kx2 * ky2 + 2 * sy * ky2)) /
               (kx2 - 4 * ky2);
      i33213 = (2 * cx * cy2 * sy * ky2 + 2 * cx * sy3 * ky4 - sx * cy3 * kx2 - sx * cy * sy2 * kx2 * ky2 + sy * kx2 - 2 * sy * ky2) / (kx2 * (kx2 - 4 * ky2));
      i43213 = -(cx * cy3 * kx2 - 2 * cx * cy3 * ky2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * sy2 * ky4 + sx * cy2 * sy * kx2 * ky2 +
                 sx * sy3 * kx2 * ky4 - cy * kx2 + 2 * cy * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i34223 = (ky2 * (cx * cy2 * sy + cx * sy3 * ky2 - 2 * sx * cy3 - 2 * sx * cy * sy2 * ky2 + sy)) / (kx2 - 4 * ky2);
      i44223 = -(ky2 * (cx * cy3 + cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 - 2 * sx * cy2 * sy * ky2 + sx * sy3 * kx2 * ky2 - 2 * sx * sy3 * ky4 -
                        cy)) /
               (kx2 - 4 * ky2);
      i33613 = (ha * (2 * cx * cy + 2 * sx * cy2 * sy * kx2 + 2 * sx * sy3 * kx2 * ky2 - 2 * sx * sy * kx2 + 4 * sx * sy * ky2 - 2 * cy + sy * t0 * kx2 -
                      4 * sy * t0 * ky2)) /
               (2 * kx2 * (kx2 - 4 * ky2));
      i43613 = (ha * (2 * cx * cy2 * sy * kx2 + 2 * cx * sy3 * kx2 * ky2 - 2 * cx * sy * kx2 + 2 * cx * sy * ky2 + 2 * sx * cy3 * kx2 + 2 * sx * cy * sy2 * kx2 * ky2 -
                      4 * sx * cy * kx2 + 4 * sx * cy * ky2 + cy * t0 * kx2 - 4 * cy * t0 * ky2 + sy * kx2 - 2 * sy * ky2)) /
               (2 * kx2 * (kx2 - 4 * ky2));
      i34623 = (ky2 * ha * (2 * cx * cy3 + 2 * cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 + sx * sy3 * kx2 * ky2 - 2 * cy)) / (kx2 * (kx2 - 4 * ky2));
      i44623 = (ky2 * ha * (cx * cy2 * sy * kx2 - 2 * cx * cy2 * sy * ky2 + cx * sy3 * kx2 * ky2 - 2 * cx * sy3 * ky4 - sx * cy3 * kx2 - sx * cy * sy2 * kx2 * ky2 + 2 * sy * ky2)) / (kx2 * (kx2 - 4 * ky2));
      i36633 = (cy3 + cy2 * sy * t0 * ky2 + cy * sy2 * ky2 - cy + sy3 * t0 * ky4) / (2 * ky2);
      i46633 = (cy3 * t0 + cy * sy2 * t0 * ky2 + sy) / 2;
      i33114 = -(cx * cy2 * sy + cx * sy3 * ky2 - 2 * sx * cy3 - 2 * sx * cy * sy2 * ky2 + sy) / (kx2 - 4 * ky2);
      i43114 = (cx * cy3 + cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 - 2 * sx * cy2 * sy * ky2 + sx * sy3 * kx2 * ky2 - 2 * sx * sy3 * ky4 - cy) / (kx2 -
                                                                                                                                                    4 * ky2);
      i34124 = -(cx * cy2 * sy * kx2 + cx * sy3 * kx2 * ky2 - cx * sy * kx2 + 2 * cx * sy * ky2 - sx * cy * kx2 + sy * kx2 - 2 * sy * ky2) / (kx2 - 4 * ky2);
      i44124 = -(cx * cy3 * kx2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * kx2 + 2 * cx * cy * ky2 - sx * cy2 * sy * kx4 - sx * sy3 * kx4 * ky2 +
                 sx * sy * kx4 - sx * sy * kx2 * ky2 + cy * kx2 - 2 * cy * ky2) /
               (kx2 - 4 * ky2);
      i33214 = -(2 * cx * cy + 2 * sx * cy2 * sy * kx2 + 2 * sx * sy3 * kx2 * ky2 - sx * sy * kx2 - 2 * cy) / (kx2 * (kx2 - 4 * ky2));
      i43214 = -(2 * cx * cy2 * sy * kx2 + 2 * cx * sy3 * kx2 * ky2 - cx * sy * kx2 - 2 * cx * sy * ky2 + 2 * sx * cy3 * kx2 + 2 * sx * cy * sy2 * kx2 * ky2 -
                 3 * sx * cy * kx2 + 2 * sy * ky2) /
               (kx2 * (kx2 - 4 * ky2));
      i34224 = -(cx * cy3 + cx * cy * sy2 * ky2 + 2 * sx * cy2 * sy * ky2 + 2 * sx * sy3 * ky4 - cy) / (kx2 - 4 * ky2);
      i44224 = -(cx * cy2 * sy * ky2 + cx * sy3 * ky4 - sx * cy3 * kx2 + 2 * sx * cy3 * ky2 - sx * cy * sy2 * kx2 * ky2 + 2 * sx * cy * sy2 * ky4 +
                 sy * ky2) /
               (kx2 - 4 * ky2);
      i33614 = (ha * (4 * cx * cy2 * sy * ky2 + 4 * cx * sy3 * ky4 - 2 * cx * sy * ky2 - 4 * sx * cy * ky2 + cy2 * sy * kx2 - 4 * cy2 * sy * ky2 - cy * t0 * kx2 +
                      4 * cy * t0 * ky2 + sy3 * kx2 * ky2 - 4 * sy3 * ky4 + 2 * sy * ky2)) /
               (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      i43614 = (ha * (4 * cx * cy3 * ky2 + 4 * cx * cy * sy2 * ky4 - 6 * cx * cy * ky2 - 4 * sx * cy2 * sy * kx2 * ky2 - 4 * sx * sy3 * kx2 * ky4 +
                      2 * sx * sy * kx2 * ky2 + 4 * sx * sy * ky4 + cy3 * kx2 - 4 * cy3 * ky2 + cy * sy2 * kx2 * ky2 - 4 * cy * sy2 * ky4 - cy * kx2 + 6 * cy * ky2 +
                      sy * t0 * kx2 * ky2 - 4 * sy * t0 * ky4)) /
               (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      i34624 = (ha * (cx * cy2 * sy * kx2 + cx * sy3 * kx2 * ky2 - cx * sy * kx2 + 2 * cx * sy * ky2 - sx * cy * kx2 + sy * kx2 - 2 * sy * ky2)) / (kx2 * (kx2 - 4 * ky2));
      i44624 = (ha * (cx * cy3 * kx2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * kx2 + 2 * cx * cy * ky2 - sx * cy2 * sy * kx4 - sx * sy3 * kx4 * ky2 +
                      sx * sy * kx4 - sx * sy * kx2 * ky2 + cy * kx2 - 2 * cy * ky2)) /
               (kx2 * (kx2 - 4 * ky2));
      i36634 = (cy2 * sy - cy * t0 + sy3 * ky2) / (2 * ky2);
      i46634 = (cy3 + cy * sy2 * ky2 - cy + sy * t0 * ky2) / (2 * ky2);
      i33316 = (ha * (2 * cx * cy + 2 * sx * cy2 * sy * kx2 + 2 * sx * sy3 * kx2 * ky2 - 2 * sx * sy * kx2 + 4 * sx * sy * ky2 - 2 * cy + sy * t0 * kx2 -
                      4 * sy * t0 * ky2)) /
               (2 * kx2 * (kx2 - 4 * ky2));
      i43316 = (ha * (2 * cx * cy2 * sy * kx2 + 2 * cx * sy3 * kx2 * ky2 - 2 * cx * sy * kx2 + 2 * cx * sy * ky2 + 2 * sx * cy3 * kx2 + 2 * sx * cy * sy2 * kx2 * ky2 -
                      4 * sx * cy * kx2 + 4 * sx * cy * ky2 + cy * t0 * kx2 - 4 * cy * t0 * ky2 + sy * kx2 - 2 * sy * ky2)) /
               (2 * kx2 * (kx2 - 4 * ky2));
      i34326 = (ky2 * ha * (2 * cx * cy3 + 2 * cx * cy * sy2 * ky2 + sx * cy2 * sy * kx2 + sx * sy3 * kx2 * ky2 - 2 * cy)) / (kx2 * (kx2 - 4 * ky2));
      i44326 = (ky2 * ha * (cx * cy2 * sy * kx2 - 2 * cx * cy2 * sy * ky2 + cx * sy3 * kx2 * ky2 - 2 * cx * sy3 * ky4 - sx * cy3 * kx2 - sx * cy * sy2 * kx2 * ky2 + 2 * sy * ky2)) / (kx2 * (kx2 - 4 * ky2));
      i36336 = (cy3 + cy2 * sy * t0 * ky2 + cy * sy2 * ky2 - cy + sy3 * t0 * ky4) / (2 * ky2);
      i46336 = (cy3 * t0 + cy * sy2 * t0 * ky2 + sy) / 2;
      i33416 = (ha * (4 * cx * cy2 * sy * ky2 + 4 * cx * sy3 * ky4 - 2 * cx * sy * ky2 - 4 * sx * cy * ky2 + cy2 * sy * kx2 - 4 * cy2 * sy * ky2 - cy * t0 * kx2 +
                      4 * cy * t0 * ky2 + sy3 * kx2 * ky2 - 4 * sy3 * ky4 + 2 * sy * ky2)) /
               (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      i43416 = (ha * (4 * cx * cy3 * ky2 + 4 * cx * cy * sy2 * ky4 - 6 * cx * cy * ky2 - 4 * sx * cy2 * sy * kx2 * ky2 - 4 * sx * sy3 * kx2 * ky4 +
                      2 * sx * sy * kx2 * ky2 + 4 * sx * sy * ky4 + cy3 * kx2 - 4 * cy3 * ky2 + cy * sy2 * kx2 * ky2 - 4 * cy * sy2 * ky4 - cy * kx2 + 6 * cy * ky2 +
                      sy * t0 * kx2 * ky2 - 4 * sy * t0 * ky4)) /
               (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      i34426 = (ha * (cx * cy2 * sy * kx2 + cx * sy3 * kx2 * ky2 - cx * sy * kx2 + 2 * cx * sy * ky2 - sx * cy * kx2 + sy * kx2 - 2 * sy * ky2)) / (kx2 * (kx2 - 4 * ky2));
      i44426 = (ha * (cx * cy3 * kx2 + cx * cy * sy2 * kx2 * ky2 - 2 * cx * cy * kx2 + 2 * cx * cy * ky2 - sx * cy2 * sy * kx4 - sx * sy3 * kx4 * ky2 +
                      sx * sy * kx4 - sx * sy * kx2 * ky2 + cy * kx2 - 2 * cy * ky2)) /
               (kx2 * (kx2 - 4 * ky2));
      i36436 = (cy2 * sy - cy * t0 + sy3 * ky2) / (2 * ky2);
      i46436 = (cy3 + cy * sy2 * ky2 - cy + sy * t0 * ky2) / (2 * ky2);
    }

    T[0][0][0] = e111 * i11111 + e122 * i12121;
    T[1][0][0] = e111 * i21111 + e122 * i22121;
    T[0][1][0] = e111 * i11211 + e122 * i12221;
    T[1][1][0] = e111 * i21211 + e122 * i22221;
    T[0][5][0] = e111 * i11611 + e122 * i12621 + e161 * i16611;
    T[1][5][0] = e111 * i21611 + e122 * i22621 + e161 * i26611;
    T[0][1][0] = e111 * i11112 + e122 * i12122;
    T[1][1][0] = e111 * i21112 + e122 * i22122;
    T[0][1][1] = e111 * i11212 + e122 * i12222;
    T[1][1][1] = e111 * i21212 + e122 * i22222;
    T[0][5][1] = e111 * i11612 + e122 * i12622 + e161 * i16612;
    T[1][5][1] = e111 * i21612 + e122 * i22622 + e161 * i26612;
    T[0][2][2] = e133 * i13333 + e144 * i14343;
    T[1][2][2] = e133 * i23333 + e144 * i24343;
    T[0][3][2] = e133 * i13433 + e144 * i14443;
    T[1][3][2] = e133 * i23433 + e144 * i24443;
    T[0][3][2] = e133 * i13334 + e144 * i14344;
    T[1][3][2] = e133 * i23334 + e144 * i24344;
    T[0][3][3] = e133 * i13434 + e144 * i14444;
    T[1][3][3] = e133 * i23434 + e144 * i24444;
    T[0][5][0] = e111 * i11116 + e122 * i12126 + e161 * i16116;
    T[1][5][0] = e111 * i21116 + e122 * i22126 + e161 * i26116;
    T[0][5][1] = e111 * i11216 + e122 * i12226 + e161 * i16216;
    T[1][5][1] = e111 * i21216 + e122 * i22226 + e161 * i26216;
    T[0][5][5] = e111 * i11616 + e122 * i12626 + e161 * i16616 + e166 * i16666;
    T[1][5][5] = e111 * i21616 + e122 * i22626 + e161 * i26616 + e166 * i26666;
    T[2][2][0] = e331 * i33311 + e342 * i34321;
    T[3][2][0] = e331 * i43311 + e342 * i44321;
    T[2][3][0] = e331 * i33411 + e342 * i34421;
    T[3][3][0] = e331 * i43411 + e342 * i44421;
    T[2][2][1] = e331 * i33312 + e342 * i34322;
    T[3][2][1] = e331 * i43312 + e342 * i44322;
    T[2][3][1] = e331 * i33412 + e342 * i34422;
    T[3][3][1] = e331 * i43412 + e342 * i44422;
    T[2][2][0] = e331 * i33113 + e342 * i34123;
    T[3][2][0] = e331 * i43113 + e342 * i44123;
    T[2][2][1] = e331 * i33213 + e342 * i34223;
    T[3][2][1] = e331 * i43213 + e342 * i44223;
    T[2][5][2] = e331 * i33613 + e342 * i34623 + e363 * i36633;
    T[3][5][2] = e331 * i43613 + e342 * i44623 + e363 * i46633;
    T[2][3][0] = e331 * i33114 + e342 * i34124;
    T[3][3][0] = e331 * i43114 + e342 * i44124;
    T[2][3][1] = e331 * i33214 + e342 * i34224;
    T[3][3][1] = e331 * i43214 + e342 * i44224;
    T[2][5][3] = e331 * i33614 + e342 * i34624 + e363 * i36634;
    T[3][5][3] = e331 * i43614 + e342 * i44624 + e363 * i46634;
    T[2][5][2] = e331 * i33316 + e342 * i34326 + e363 * i36336;
    T[3][5][2] = e331 * i43316 + e342 * i44326 + e363 * i46336;
    T[2][5][3] = e331 * i33416 + e342 * i34426 + e363 * i36436;
    T[3][5][3] = e331 * i43416 + e342 * i44426 + e363 * i46436;

    T[0][2][0] = T[1][2][0] = T[0][2][1] = T[1][2][1] = 0;
    T[0][3][0] = T[1][3][0] = T[0][3][1] = T[1][3][1] = 0;
    T[0][5][2] = T[1][5][2] = T[0][5][3] = T[1][5][3] = 0;
    T[2][0][0] = T[3][0][0] = T[2][1][0] = T[3][1][0] = T[2][1][1] = T[3][1][1] = 0;
    T[2][2][2] = T[3][2][2] = 0;
    T[2][3][2] = T[3][3][2] = T[2][3][3] = T[3][3][3] = T[2][5][0] = T[3][5][0] = T[2][5][1] = T[3][5][1] = 0;
    T[2][5][5] = T[3][5][5] = 0;

    if (kx2_is_zero && ky2_is_zero) {
      T[4][0][0] = T[4][1][0] = T[4][2][2] = T[4][3][2] = T[4][5][0] = 0;
      T[4][1][1] = T[4][3][3] = t0 / 2;
      T[4][5][5] = (t0_3 * ha2) / 6;
      T[4][5][1] = (t0_2 * ha) / 2;
    } else if (ky2_is_zero) {
      T[4][0][0] = (2 * h * cx * sx * kx2 * e122 - 2 * h * cx * sx * e111 - 8 * h * sx * kx2 * e122 - 4 * h * sx * e111 + 6 * h * t0 * kx2 * e122 + 6 * h * t0 * e111 - 3 * cx * sx * kx4 +
                    3 * t0 * kx4) /
                   (12 * kx2);

      T[4][1][0] = (2 * h * cx2 * sx2 * kx4 * e122 - 2 * h * cx2 * sx2 * kx2 * e111 + 4 * h * cx * kx2 * e122 - 4 * h * cx * e111 + 2 * h * sx4 * kx6 * e122 -
                    2 * h * sx4 * kx4 * e111 - 4 * h * kx2 * e122 + 4 * h * e111 - 3 * sx2 * kx6) /
                   (6 * kx4);

      T[4][1][1] = -(2 * h * cx * sx * kx2 * e122 - 2 * h * cx * sx * e111 + 4 * h * sx * kx2 * e122 + 8 * h * sx * e111 - 6 * h * t0 * kx2 * e122 - 6 * h * t0 * e111 - 3 * cx * sx * kx4 -
                     3 * t0 * kx4) /
                   (12 * kx4);

      T[4][2][2] = (h * e133 * (cx2 * t0 + sx2 * t0 * kx2 - sx)) / kx2;

      T[4][3][2] = (h * e133 * (cx2 * t0_2 * kx2 + 2 * cx + sx2 * t0_2 * kx4 - 2)) / kx4;

      T[4][3][3] = (2 * h * cx2 * t0_3 * kx2 * e133 + 6 * h * cx2 * t0 * kx2 * e144 - 12 * h * cx2 * t0 * e133 + 2 * h * sx2 * t0_3 * kx4 * e133 +
                    6 * h * sx2 * t0 * kx4 * e144 - 12 * h * sx2 * t0 * kx2 * e133 - 6 * h * sx * kx2 * e144 + 12 * h * sx * e133 + 3 * t0 * kx4) /
                   (6 * kx4);

      T[4][5][5] = (2 * h * cx * sx * kx2 * ha2 * e122 - 2 * h * cx * sx * ha2 * e111 + 6 * h * cx * t0 * kx2 * ha * e161 + 12 * h * cx * t0 * ha2 * e111 - 12 * h * sx * kx4 * e166 -
                    8 * h * sx * kx2 * ha2 * e122 - 18 * h * sx * kx2 * ha * e161 - 28 * h * sx * ha2 * e111 + 12 * h * t0 * kx4 * e166 + 6 * h * t0 * kx2 * ha2 * e122 +
                    12 * h * t0 * kx2 * ha * e161 + 18 * h * t0 * ha2 * e111 - 3 * cx * sx * kx4 * ha2 + 3 * t0 * kx4 * ha2) /
                   (12 * kx6);

      T[4][5][0] = -(2 * h * cx4 * t0 * kx2 * ha * e122 - 2 * h * cx4 * t0 * ha * e111 - 2 * h * cx3 * sx * kx2 * ha * e122 + 2 * h * cx3 * sx * ha * e111 +
                     3 * h * cx3 * t0 * kx2 * e161 + 4 * h * cx2 * sx2 * t0 * kx4 * ha * e122 - 4 * h * cx2 * sx2 * t0 * kx2 * ha * e111 - 6 * h * cx2 * sx * kx2 * e161 +
                     4 * h * cx2 * t0 * kx2 * ha * e122 + 8 * h * cx2 * t0 * ha * e111 - 2 * h * cx * sx3 * kx4 * ha * e122 + 2 * h * cx * sx3 * kx2 * ha * e111 +
                     3 * h * cx * sx2 * t0 * kx4 * e161 + 4 * h * cx * sx * kx2 * ha * e122 - 4 * h * cx * sx * ha * e111 + 6 * h * cx * t0 * ha * e111 + 2 * h * sx4 * t0 * kx6 * ha * e122 -
                     2 * h * sx4 * t0 * kx4 * ha * e111 - 6 * h * sx3 * kx4 * e161 + 4 * h * sx2 * t0 * kx4 * ha * e122 + 8 * h * sx2 * t0 * kx2 * ha * e111 -
                     8 * h * sx * kx2 * ha * e122 + 3 * h * sx * kx2 * e161 - 10 * h * sx * ha * e111 + 3 * cx2 * t0 * kx4 * ha - 3 * cx * sx * kx4 * ha +
                     3 * sx2 * t0 * kx6 * ha) /
                   (6 * kx4);

      T[4][5][1] = (h * cx4 * kx2 * ha * e122 - 3 * h * cx3 * kx2 * e161 - 6 * h * cx3 * ha * e111 - 2 * h * cx2 * ha * e111 - 3 * h * cx * sx2 * kx4 * e161 -
                    6 * h * cx * sx2 * kx2 * ha * e111 - 4 * h * cx * kx2 * ha * e122 - 3 * h * cx * kx2 * e161 - 2 * h * cx * ha * e111 - h * sx4 * kx6 * ha * e122 -
                    3 * h * sx * t0 * kx4 * e161 - 6 * h * sx * t0 * kx2 * ha * e111 + 3 * h * kx2 * ha * e122 + 6 * h * kx2 * e161 + 10 * h * ha * e111 - 3 * cx2 * kx4 * ha +
                    3 * kx4 * ha) /
                   (6 * kx6);
    } else {
      T[4][0][0] = (2 * h * cx * sx * kx2 * e122 - 2 * h * cx * sx * e111 - 8 * h * sx * kx2 * e122 - 4 * h * sx * e111 + 6 * h * t0 * kx2 * e122 + 6 * h * t0 * e111 -
                    3 * cx * sx * kx4 + 3 * t0 * kx4) /
                   (12 * kx2);
      T[4][1][0] = (2 * h * cx2 * sx2 * kx4 * e122 - 2 * h * cx2 * sx2 * kx2 * e111 + 4 * h * cx * kx2 * e122 - 4 * h * cx * e111 + 2 * h * sx4 * kx6 * e122 -
                    2 * h * sx4 * kx4 * e111 - 4 * h * kx2 * e122 + 4 * h * e111 - 3 * sx2 * kx6) /
                   (6 * kx4);
      T[4][1][1] = -(2 * h * cx * sx * kx2 * e122 - 2 * h * cx * sx * e111 + 4 * h * sx * kx2 * e122 + 8 * h * sx * e111 - 6 * h * t0 * kx2 * e122 - 6 * h * t0 * e111 -
                     3 * cx * sx * kx4 - 3 * t0 * kx4) /
                   (12 * kx4);
      T[4][2][2] = -(4 * h * sx * kx2 * e133 - 8 * h * sx * ky4 * e144 - 8 * h * sx * ky2 * e133 + 2 * h * cy * sy * kx2 * ky2 * e144 - 2 * h * cy * sy * kx2 * e133 -
                     2 * h * t0 * kx2 * ky2 * e144 - 2 * h * t0 * kx2 * e133 + 8 * h * t0 * ky4 * e144 + 8 * h * t0 * ky2 * e133 + cy * sy * kx4 * ky2 - 4 * cy * sy * kx2 * ky4 -
                     t0 * kx4 * ky2 + 4 * t0 * kx2 * ky4) /
                   (4 * kx2 * (kx2 - 4 * ky2));
      T[4][3][2] = (2 * h * cx2 * cy2 * kx2 * ky2 * e144 - 2 * h * cx2 * cy2 * kx2 * e133 - 4 * h * cx * ky4 * e144 + 4 * h * cx * ky2 * e133 +
                    2 * h * sx2 * cy2 * kx4 * ky2 * e144 - 2 * h * sx2 * cy2 * kx4 * e133 - 2 * h * kx2 * ky2 * e144 + 2 * h * kx2 * e133 + 4 * h * ky4 * e144 -
                    4 * h * ky2 * e133 + cy2 * kx4 * ky2 - 4 * cy2 * kx2 * ky4 - kx4 * ky2 + 4 * kx2 * ky4) /
                   (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      T[4][3][3] = -(4 * h * sx * kx2 * ky2 * e144 - 8 * h * sx * ky4 * e144 - 8 * h * sx * ky2 * e133 - 2 * h * cy * sy * kx2 * ky2 * e144 + 2 * h * cy * sy * kx2 * e133 -
                     2 * h * t0 * kx2 * ky2 * e144 - 2 * h * t0 * kx2 * e133 + 8 * h * t0 * ky4 * e144 + 8 * h * t0 * ky2 * e133 - cy * sy * kx4 * ky2 + 4 * cy * sy * kx2 * ky4 -
                     t0 * kx4 * ky2 + 4 * t0 * kx2 * ky4) /
                   (4 * kx2 * ky2 * (kx2 - 4 * ky2));
      T[4][3][2] = (h * ha * nh * h * cx2 * cy2 * kx2 - 2 * h * ha * nh * h * cx * ky2 + h * ha * nh * h * sx2 * cy2 * kx4 -
                    h * ha * nh * h * kx2 + 2 * h * ha * nh * h * ky2 - 2 * h * ha * betah2 * cx2 * cy2 * kx2 + 4 * h * ha * betah2 * cx * ky2 -
                    2 * h * ha * betah2 * sx2 * cy2 * kx4 + 2 * h * ha * betah2 * kx2 - 4 * h * ha * betah2 * ky2 - h * ha * cx2 * cy2 * kx2 * ky2 +
                    2 * h * ha * cx * ky4 - h * ha * sx2 * cy2 * kx4 * ky2 + h * ha * kx2 * ky2 - 2 * h * ha * ky4 + cy2 * kx4 * ky2 -
                    4 * cy2 * kx2 * ky4 - kx4 * ky2 + 4 * kx2 * ky4) /
                   (2 * kx2 * ky2 * (kx2 - 4 * ky2));
      T[4][3][3] = -(h * t0 * ha * nh * h * kx2 - 4 * h * t0 * ha * nh * h * ky2 - 2 * h * t0 * ha * betah2 * kx2 + 8 * h * t0 * ha * betah2 * ky2 +
                     h * t0 * ha * kx2 * ky2 - 4 * h * t0 * ha * ky4 + 4 * h * ha * nh * h * sx * ky2 - h * ha * nh * h * cy * sy * kx2 -
                     8 * h * ha * betah2 * sx * ky2 + 2 * h * ha * betah2 * cy * sy * kx2 - 2 * h * ha * sx * kx2 * ky2 + 4 * h * ha * sx * ky4 +
                     h * ha * cy * sy * kx2 * ky2 - t0 * kx4 * ky2 + 4 * t0 * kx2 * ky4 - cy * sy * kx4 * ky2 + 4 * cy * sy * kx2 * ky4) /
                   (4 * kx2 * ky2 * (kx2 - 4 * ky2));
      T[4][5][5] = (2 * h * cx * sx * kx2 * ha2 * e122 - 2 * h * cx * sx * ha2 * e111 + 6 * h * cx * t0 * kx2 * ha * e161 + 12 * h * cx * t0 * ha2 * e111 - 12 * h * sx * kx4 * e166 -
                    8 * h * sx * kx2 * ha2 * e122 - 18 * h * sx * kx2 * ha * e161 - 28 * h * sx * ha2 * e111 + 12 * h * t0 * kx4 * e166 + 6 * h * t0 * kx2 * ha2 * e122 +
                    12 * h * t0 * kx2 * ha * e161 + 18 * h * t0 * ha2 * e111 - 3 * cx * sx * kx4 * ha2 + 3 * t0 * kx4 * ha2) /
                   (12 * kx6);
      T[4][5][0] = -(2 * h * cx4 * t0 * kx2 * ha * e122 - 2 * h * cx4 * t0 * ha * e111 - 2 * h * cx3 * sx * kx2 * ha * e122 + 2 * h * cx3 * sx * ha * e111 +
                     3 * h * cx3 * t0 * kx2 * e161 + 4 * h * cx2 * sx2 * t0 * kx4 * ha * e122 - 4 * h * cx2 * sx2 * t0 * kx2 * ha * e111 - 6 * h * cx2 * sx * kx2 * e161 +
                     4 * h * cx2 * t0 * kx2 * ha * e122 + 8 * h * cx2 * t0 * ha * e111 - 2 * h * cx * sx3 * kx4 * ha * e122 + 2 * h * cx * sx3 * kx2 * ha * e111 +
                     3 * h * cx * sx2 * t0 * kx4 * e161 + 4 * h * cx * sx * kx2 * ha * e122 - 4 * h * cx * sx * ha * e111 + 6 * h * cx * t0 * ha * e111 + 2 * h * sx4 * t0 * kx6 * ha * e122 -
                     2 * h * sx4 * t0 * kx4 * ha * e111 - 6 * h * sx3 * kx4 * e161 + 4 * h * sx2 * t0 * kx4 * ha * e122 + 8 * h * sx2 * t0 * kx2 * ha * e111 -
                     8 * h * sx * kx2 * ha * e122 + 3 * h * sx * kx2 * e161 - 10 * h * sx * ha * e111 + 3 * cx2 * t0 * kx4 * ha - 3 * cx * sx * kx4 * ha +
                     3 * sx2 * t0 * kx6 * ha) /
                   (6 * kx4);
      T[4][5][1] = (h * cx4 * kx2 * ha * e122 - 3 * h * cx3 * kx2 * e161 - 6 * h * cx3 * ha * e111 - 2 * h * cx2 * ha * e111 - 3 * h * cx * sx2 * kx4 * e161 -
                    6 * h * cx * sx2 * kx2 * ha * e111 - 4 * h * cx * kx2 * ha * e122 - 3 * h * cx * kx2 * e161 - 2 * h * cx * ha * e111 - h * sx4 * kx6 * ha * e122 -
                    3 * h * sx * t0 * kx4 * e161 - 6 * h * sx * t0 * kx2 * ha * e111 + 3 * h * kx2 * ha * e122 + 6 * h * kx2 * e161 + 10 * h * ha * e111 - 3 * cx2 * kx4 * ha +
                    3 * kx4 * ha) /
                   (6 * kx6);
    }
#if defined(TEST_NEW_MATRIX)
    if (kx2 != 0 && ky2 != 0)
      replaceWithNewMatrix(C, R, T, kx2, ky2, ha, h, nh, betah2, 0.0, t0);
#endif

    T[0][1][0] += h * R[0][1];
    T[0][3][0] += h * R[0][3];
    T[1][0][0] -= h * R[0][0] * R[1][0];
    T[1][1][0] -= h * R[0][0] * R[1][1] + h * R[0][1] * R[1][0] - h * R[1][1];
    T[1][1][1] -= h * R[0][1] * R[1][1];
    T[1][2][0] -= h * R[0][0] * R[1][2] + h * R[0][2] * R[1][0];
    T[1][2][1] -= h * R[0][1] * R[1][2] + h * R[0][2] * R[1][1];
    T[1][2][2] -= h * R[0][2] * R[1][2];
    T[1][3][0] -= h * R[0][0] * R[1][3] + h * R[0][3] * R[1][0] - h * R[1][3];
    T[1][3][1] -= h * R[0][1] * R[1][3] + h * R[0][3] * R[1][1];
    T[1][3][2] -= h * R[0][2] * R[1][3] + h * R[0][3] * R[1][2];
    T[1][3][3] -= h * R[0][3] * R[1][3];
    T[1][4][0] -= h * R[0][0] * R[1][4] + h * R[0][4] * R[1][0];
    T[1][4][1] -= h * R[0][1] * R[1][4] + h * R[0][4] * R[1][1];
    T[1][4][2] -= h * R[0][2] * R[1][4] + h * R[0][4] * R[1][2];
    T[1][4][3] -= h * R[0][3] * R[1][4] + h * R[0][4] * R[1][3];
    T[1][4][4] -= h * R[0][4] * R[1][4];
    T[1][5][0] -= h * R[0][0] * R[1][5] + h * R[0][5] * R[1][0];
    T[1][5][1] -= h * R[0][1] * R[1][5] + h * R[0][5] * R[1][1];
    T[1][5][2] -= h * R[0][2] * R[1][5] + h * R[0][5] * R[1][2];
    T[1][5][3] -= h * R[0][3] * R[1][5] + h * R[0][5] * R[1][3];
    T[1][5][4] -= h * R[0][4] * R[1][5] + h * R[0][5] * R[1][4];
    T[1][5][5] -= h * R[0][5] * R[1][5];
    T[2][1][0] += h * R[2][1];
    T[2][3][0] += h * R[2][3];
    T[3][0][0] -= h * R[0][0] * R[3][0];
    T[3][1][0] -= h * R[0][0] * R[3][1] + h * R[0][1] * R[3][0] - h * R[3][1];
    T[3][1][1] -= h * R[0][1] * R[3][1];
    T[3][2][0] -= h * R[0][0] * R[3][2] + h * R[0][2] * R[3][0];
    T[3][2][1] -= h * R[0][1] * R[3][2] + h * R[0][2] * R[3][1];
    T[3][2][2] -= h * R[0][2] * R[3][2];
    T[3][3][0] -= h * R[0][0] * R[3][3] + h * R[0][3] * R[3][0] - h * R[3][3];
    T[3][3][1] -= h * R[0][1] * R[3][3] + h * R[0][3] * R[3][1];
    T[3][3][2] -= h * R[0][2] * R[3][3] + h * R[0][3] * R[3][2];
    T[3][3][3] -= h * R[0][3] * R[3][3];
    T[3][4][0] -= h * R[0][0] * R[3][4] + h * R[0][4] * R[3][0];
    T[3][4][1] -= h * R[0][1] * R[3][4] + h * R[0][4] * R[3][1];
    T[3][4][2] -= h * R[0][2] * R[3][4] + h * R[0][4] * R[3][2];
    T[3][4][3] -= h * R[0][3] * R[3][4] + h * R[0][4] * R[3][3];
    T[3][4][4] -= h * R[0][4] * R[3][4];
    T[3][5][0] -= h * R[0][0] * R[3][5] + h * R[0][5] * R[3][0];
    T[3][5][1] -= h * R[0][1] * R[3][5] + h * R[0][5] * R[3][1];
    T[3][5][2] -= h * R[0][2] * R[3][5] + h * R[0][5] * R[3][2];
    T[3][5][3] -= h * R[0][3] * R[3][5] + h * R[0][5] * R[3][3];
    T[3][5][4] -= h * R[0][4] * R[3][5] + h * R[0][5] * R[3][4];
    T[3][5][5] -= h * R[0][5] * R[3][5];
    T[4][1][0] += h * R[4][1];
    T[4][3][0] += h * R[4][3];
    if (h)
      C[4] += sqr((1 - ha / h)) * T[4][5][5];
  }

  log_exit("sbend_matrix");
  return (M);
}

long determine_bend_flags(ELEMENT_LIST *elem, long edge1_effects, long edge2_effects) {
  ELEMENT_LIST *other;
  long bend_flags;

  bend_flags = 0;
  other = elem->pred;
  while (other) {
    if (other->type == elem->type && strcmp(other->name, elem->name) == 0) {
      bend_flags |= SAME_BEND_PRECEDES;
      break;
    }
    if (other->type == T_MARK || other->type == T_WATCH)
      other = other->pred;
    else if (other->type == T_LSCDRIFT) {
      LSCDRIFT *lsc;
      lsc = (LSCDRIFT *)other->p_elem;
      if (lsc->length == 0)
        other = other->pred;
      else
        break;
    } else
      break;
  }
  other = elem->succ;
  while (other) {
    if (other->type == elem->type && strcmp(other->name, elem->name) == 0) {
      bend_flags |= SAME_BEND_FOLLOWS;
      break;
    }
    if (other->type == T_MARK || other->type == T_WATCH)
      other = other->succ;
    else if (other->type == T_LSCDRIFT) {
      LSCDRIFT *lsc;
      lsc = (LSCDRIFT *)other->p_elem;
      if (lsc->length == 0)
        other = other->succ;
      else
        break;
    } else
      break;
  }
  if (edge1_effects && !(bend_flags & SAME_BEND_PRECEDES))
    bend_flags |= BEND_EDGE1_EFFECTS;
  if (edge2_effects && !(bend_flags & SAME_BEND_FOLLOWS))
    bend_flags |= BEND_EDGE2_EFFECTS;
  bend_flags |= BEND_EDGE_DETERMINED;
  return (bend_flags);
}

void replaceWithNewMatrix(double *C, double **R, double ***T, double kx2, double ky2, double ha, double h,
                          double nh, double bh2, double gh3, double s) {
}
