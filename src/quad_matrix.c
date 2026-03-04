/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* contents: quadrupole_matrix(), qfringe_matrix(), quad_fringe(),
 *           qfringe_R_matrix(), qfringe_T_matrix()
 *
 * Michael Borland, 1989.
 */
#include "mdb.h"
#include "track.h"

static double swap_tmp;
#define swap_double(x, y) (swap_tmp = (x), (x) = (y), (y) = swap_tmp)

#define INSET_FRINGE 0
#define FIXED_STRENGTH_FRINGE 1
#define INTEGRALS_FRINGE 2
static char *fringeTypeOpt[3] = {"inset", "fixed-strength", "integrals"};

VMATRIX *quadrupole_matrix(double K1, double lHC, long maximum_order,
                           double fse,
                           double xkick, double ykick,
                           double edge1_effects, double edge2_effects,
                           char *fringeType, double ffringe, double lEffective,
                           double *fringeIntM, double *fringeIntP,
			   short radial) {
  VMATRIX *M;
  VMATRIX *Mfringe, *Mtot, *Md, *tmp;
  double *C, **R, ***T, ****U;
  double kl, k, sin_kl, cos_kl, cosh_kl, sinh_kl;
  double lNominal, lEdge = 0;
  long fringeCode = 0;

  K1 *= (1 + fse);

  fringeCode = -1;
  if (fringeType && (fringeCode = match_string(fringeType, fringeTypeOpt, 3, 0)) < 0)
    bombElegant("Unrecognized fringe type for QUAD or KQUAD", NULL);

  if (ffringe > 0 && lEffective > 0)
    bombElegant("FFRINGE>0 and LEFFECTIVE>0 for QUAD is not allowed.", NULL);
  if (lEffective > 0 && fringeCode != QFRINGE_INTEGRALS)
    bombElegant("LEFFECTIVE>0 requires FRINGE_TYPE=\"integrals\" for QUAD", NULL);

  if (K1 == 0 || lHC == 0) {
    M = drift_matrix(lHC, maximum_order);
  } else {
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, M->order = MIN(3, maximum_order));
    R = M->R;
    C = M->C;

    if (lEffective <= 0) {
      /* lHC is the "hard core" length (wherein K1 is constant)
         * lNominal is the effective length.
         * If fringe effects are off, these are the same.
         */
      lNominal = lHC;
      if (ffringe && fringeCode != INTEGRALS_FRINGE) {
        /* This is the old default behavior, triggered just by having FFRINGE non-zero */
        if (edge1_effects == 0 || edge2_effects == 0)
          bombElegant("EDGE1_EFFECTS and EDGE2_EFFECTS must both be non-zero for FFRINGE-based quadrupole fringe effects", NULL);

        /* If mode is fixedStrength, then the sloped area is symmetric about the nominal
           * entrance and exit points.  This means that the integrated strength is not
           * changed.   FFRINGE=(2*fringeLengthOnOneSide)/HardEdgeLength
           * If the mode is "inset", then the sloped areas end at the nominal ends of
           * the quad.  The integrated strength changes as the fringe fraction changes.
           */
        /* length of each edge */
        lEdge = lNominal * ffringe / 2;
        switch (fringeCode) {
        case 1:
          /* only half the total edge-field length is inside the nominal length */
          lHC = lNominal - lEdge;
          break;
        case 0:
          lHC = lNominal - 2 * lEdge;
          break;
        default:
          break;
        }
      }
    } else {
      /* Confusingly, lHC is the insertion length, lEffective the hard-edge length */
      lEdge = (lHC - lEffective) / 2;
      lHC = lEffective;
    }

    kl = (k = sqrt(fabs(K1))) * lHC;
    sin_kl = sin(kl);
    cos_kl = cos(kl);
    cosh_kl = cosh(kl);
    sinh_kl = sinh(kl);

    R[4][4] = R[5][5] = 1;
    C[4] = lHC;
    if (K1 > 0) {
      /* focussing in horizontal plane */
      R[0][0] = R[1][1] = cos_kl;
      R[0][1] = sin_kl / k;
      R[1][0] = -k * sin_kl;
      R[2][2] = R[3][3] = cosh_kl;
      R[2][3] = sinh_kl / k;
      R[3][2] = k * sinh_kl;

      if (M->order >= 2) {
        double k2, k3, k4, l2, cos_kl, cos_2kl, cos_3kl, sin_kl, sin_2kl, sin_3kl, cosh_kl, cosh_2kl, cosh_3kl, sinh_kl, sinh_2kl, sinh_3kl;
        /* double l3, l4; */
        k2 = ipow2(k);
        k3 = ipow3(k);
        k4 = ipow4(k);
        l2 = ipow2(lHC);
        /* l3 = ipow3(lHC) ; */
        /* l4 = ipow4(lHC) ; */
        cos_kl = cos(k * lHC);
        cos_2kl = cos(2 * k * lHC);
        cos_3kl = cos(3 * k * lHC);
        sin_kl = sin(k * lHC);
        sin_2kl = sin(2 * k * lHC);
        sin_3kl = sin(3 * k * lHC);
        cosh_kl = cosh(k * lHC);
        cosh_2kl = cosh(2 * k * lHC);
        cosh_3kl = cosh(3 * k * lHC);
        sinh_kl = sinh(k * lHC);
        sinh_2kl = sinh(2 * k * lHC);
        sinh_3kl = sinh(3 * k * lHC);

        T = M->T;
        T[0][5][0] = T[1][5][1] = kl * sin_kl / 2;
        T[0][5][1] = sin_kl / (2 * k) - lHC * cos_kl / 2;
        T[1][5][0] = k / 2 * (kl * cos_kl + sin_kl);
        T[2][5][2] = -kl / 2 * sinh_kl;
        T[2][5][3] = (sinh_kl / k - lHC * cosh_kl) / 2;
        T[3][5][2] = -k / 2 * (kl * cosh_kl + sinh_kl);
        T[3][5][3] = -kl / 2 * sinh_kl;
        T[4][0][0] = sqr(k) * (lHC - sin_kl / k * cos_kl) / 4;
        T[4][1][0] = -sqr(sin_kl) / 2;
        T[4][1][1] = (lHC + sin_kl / k * cos_kl) / 4;
        T[4][2][2] = -sqr(k) * (lHC - sinh_kl / k * cosh_kl) / 4;
        T[4][3][2] = sqr(sinh_kl) / 2;
        T[4][3][3] = (lHC + sinh_kl / k * cosh_kl) / 4;

        if (M->order >= 3) {
          U = M->Q;
          U[0][0][0][0] = (-3 * k3 * lHC * sin_kl) / 16. + (3 * k2 * sin_kl * sin_2kl) / 32.;
          U[0][1][0][0] = (3 * k2 * lHC * cos_kl) / 16. + (15 * k * sin_kl) / 64. - (9 * k * sin_3kl) / 64.;
          U[0][1][1][0] = (-3 * k * lHC * sin_kl) / 16. - (9 * sin_kl * sin_2kl) / 32.;
          U[0][1][1][1] = (3 * lHC * cos_kl) / 16. - (21 * sin_kl) / (64. * k) + (3 * sin_3kl) / (64. * k);
          U[0][2][2][0] = (k3 * lHC * sin_kl) / 8. + (k2 * cos_kl * pow(sinh_kl, 2)) / 16. - (3 * k2 * sin_kl * sinh_2kl) / 32.;
          U[0][2][2][1] = -(k2 * lHC * cos_kl) / 8. - (3 * k * sin_kl) / 32. + (k * cosh_2kl * sin_kl) / 32. + (3 * k * cos_kl * sinh_2kl) / 32.;
          U[0][3][2][0] = (k * sin_kl) / 16. - (3 * k * cosh_2kl * sin_kl) / 16. + (k * cos_kl * sinh_2kl) / 16.;
          U[0][3][2][1] = (cosh_kl * sin_kl * sinh_kl) / 8. + (3 * cos_kl * pow(sinh_kl, 2)) / 8.;
          U[0][3][3][0] = -(k * lHC * sin_kl) / 8. - (sin_kl * sin_2kl * pow(sinh_kl, 2)) / (16. * (-1 + cos_2kl)) - (3 * sin_kl * sinh_2kl) / 32.;
          U[0][3][3][1] = (lHC * cos_kl) / 8. - (11 * sin_kl) / (32. * k) + (cosh_2kl * sin_kl) / (32. * k) + (3 * cos_kl * sinh_2kl) / (32. * k);
          U[0][5][5][0] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * lHC * sin_kl) / 32.;
          U[0][5][5][1] = (7 * lHC * cos_kl) / 32. - (7 * sin_kl) / (32. * k) - (3 * k * l2 * sin_kl) / 32.;
          U[1][0][0][0] = (-3 * k4 * lHC * cos_kl) / 16. - (3 * k3 * sin_kl) / 16. + (3 * k3 * cos_2kl * sin_kl) / 16. + (3 * k3 * cos_kl * sin_2kl) / 32.;
          U[1][1][0][0] = (27 * k2 * cos_kl) / 64. - (27 * k2 * cos_3kl) / 64. - (3 * k3 * lHC * sin_kl) / 16.;
          U[1][1][1][0] = (-3 * k2 * lHC * cos_kl) / 16. - (3 * k * sin_kl) / 16. - (9 * k * cos_2kl * sin_kl) / 16. - (9 * k * cos_kl * sin_2kl) / 32.;
          U[1][1][1][1] = (-9 * cos_kl) / 64. + (9 * cos_3kl) / 64. - (3 * k * lHC * sin_kl) / 16.;
          U[1][2][2][0] = (k4 * lHC * cos_kl) / 8. + (k3 * sin_kl) / 8. - (3 * k3 * cosh_2kl * sin_kl) / 16. + (k3 * cos_kl * cosh_kl * sinh_kl) / 8. - (k3 * sin_kl * pow(sinh_kl, 2)) / 16. - (3 * k3 * cos_kl * sinh_2kl) / 32.;
          U[1][2][2][1] = (-7 * k2 * cos_kl) / 32. + (7 * k2 * cos_kl * cosh_2kl) / 32. + (k3 * lHC * sin_kl) / 8. - (k2 * sin_kl * sinh_2kl) / 32.;
          U[1][3][2][0] = (k2 * cos_kl) / 16. - (k2 * cos_kl * cosh_2kl) / 16. - (7 * k2 * sin_kl * sinh_2kl) / 16.;
          U[1][3][2][1] = (k * pow(cosh_kl, 2) * sin_kl) / 8. + (7 * k * cos_kl * cosh_kl * sinh_kl) / 8. - (k * sin_kl * pow(sinh_kl, 2)) / 4.;
          U[1][3][3][0] = -(k2 * lHC * cos_kl) / 8. - (k * sin_kl) / 8. - (3 * k * cosh_2kl * sin_kl) / 16. - (k * cosh_kl * sin_kl * sin_2kl * sinh_kl) / (8. * (-1 + cos_2kl)) - (k * pow(sin_kl, 3) * pow(sinh_kl, 2)) / (4. * pow(-1 + cos_2kl, 2)) - (k * cos_kl * sin_2kl * pow(sinh_kl, 2)) / (16. * (-1 + cos_2kl)) - (3 * k * cos_kl * sinh_2kl) / 32.;
          U[1][3][3][1] = (-7 * cos_kl) / 32. + (7 * cos_kl * cosh_2kl) / 32. - (k * lHC * sin_kl) / 8. - (sin_kl * sinh_2kl) / 32.;
          U[1][5][5][0] = (-19 * k2 * lHC * cos_kl) / 32. - (13 * k * sin_kl) / 32. + (3 * k3 * l2 * sin_kl) / 32.;
          U[1][5][5][1] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * lHC * sin_kl) / 32.;
          U[2][2][0][0] = (k2 * cosh_kl * pow(sin_kl, 2)) / 16. + (k3 * lHC * sinh_kl) / 8. - (3 * k2 * sin_2kl * sinh_kl) / 32.;
          U[2][2][1][0] = -(k * cosh_kl * sin_2kl) / 16. - (k * sinh_kl) / 16. + (3 * k * cos_2kl * sinh_kl) / 16.;
          U[2][2][1][1] = (k * lHC * sinh_kl) / 8. + (3 * sin_2kl * sinh_kl) / 32. + (pow(sin_kl, 2) * sinh_kl * sinh_2kl) / (16. * (1 - cosh_2kl));
          U[2][2][2][2] = (-3 * k3 * lHC * sinh_kl) / 16. + (3 * k2 * sinh_kl * sinh_2kl) / 32.;
          U[2][3][0][0] = (k2 * lHC * cosh_kl) / 8. - (3 * k * cosh_kl * sin_2kl) / 32. + (3 * k * sinh_kl) / 32. - (k * cos_2kl * sinh_kl) / 32.;
          U[2][3][1][0] = (-3 * cosh_kl * pow(sin_kl, 2)) / 8. - (cos_kl * sin_kl * sinh_kl) / 8.;
          U[2][3][1][1] = (lHC * cosh_kl) / 8. + (3 * cosh_kl * sin_2kl) / (32. * k) - (11 * sinh_kl) / (32. * k) + (cos_2kl * sinh_kl) / (32. * k);
          U[2][3][2][2] = (-3 * k2 * lHC * cosh_kl) / 16. - (15 * k * sinh_kl) / 64. + (9 * k * sinh_3kl) / 64.;
          U[2][3][3][2] = (3 * k * lHC * sinh_kl) / 16. + (9 * sinh_kl * sinh_2kl) / 32.;
          U[2][3][3][3] = (3 * lHC * cosh_kl) / 16. - (21 * sinh_kl) / (64. * k) + (3 * sinh_3kl) / (64. * k);
          U[2][5][5][2] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * lHC * sinh_kl) / 32.;
          U[2][5][5][3] = (7 * lHC * cosh_kl) / 32. - (7 * sinh_kl) / (32. * k) + (3 * k * l2 * sinh_kl) / 32.;
          U[3][2][0][0] = (k4 * lHC * cosh_kl) / 8. + (k3 * cos_kl * cosh_kl * sin_kl) / 8. - (3 * k3 * cosh_kl * sin_2kl) / 32. + (k3 * sinh_kl) / 8. - (3 * k3 * cos_2kl * sinh_kl) / 16. + (k3 * pow(sin_kl, 2) * sinh_kl) / 16.;
          U[3][2][1][0] = -(k2 * cosh_kl) / 16. + (k2 * cos_2kl * cosh_kl) / 16. - (7 * k2 * sin_2kl * sinh_kl) / 16.;
          U[3][2][1][1] = (k2 * lHC * cosh_kl) / 8. + (3 * k * cosh_kl * sin_2kl) / 32. + (k * sinh_kl) / 8. + (3 * k * cos_2kl * sinh_kl) / 16. + (k * pow(sin_kl, 2) * pow(sinh_kl, 3)) / (4. * pow(1 - cosh_2kl, 2)) + (k * cosh_kl * pow(sin_kl, 2) * sinh_2kl) / (16. * (1 - cosh_2kl)) + (k * cos_kl * sin_kl * sinh_kl * sinh_2kl) / (8. * (1 - cosh_2kl));
          U[3][2][2][2] = (-3 * k4 * lHC * cosh_kl) / 16. - (3 * k3 * sinh_kl) / 16. + (3 * k3 * cosh_2kl * sinh_kl) / 16. + (3 * k3 * cosh_kl * sinh_2kl) / 32.;
          U[3][3][0][0] = (7 * k2 * cosh_kl) / 32. - (7 * k2 * cos_2kl * cosh_kl) / 32. + (k3 * lHC * sinh_kl) / 8. - (k2 * sin_2kl * sinh_kl) / 32.;
          U[3][3][1][0] = (-7 * k * cos_kl * cosh_kl * sin_kl) / 8. - (k * pow(cos_kl, 2) * sinh_kl) / 8. - (k * pow(sin_kl, 2) * sinh_kl) / 4.;
          U[3][3][1][1] = (-7 * cosh_kl) / 32. + (7 * cos_2kl * cosh_kl) / 32. + (k * lHC * sinh_kl) / 8. + (sin_2kl * sinh_kl) / 32.;
          U[3][3][2][2] = (-27 * k2 * cosh_kl) / 64. + (27 * k2 * cosh_3kl) / 64. - (3 * k3 * lHC * sinh_kl) / 16.;
          U[3][3][3][2] = (3 * k2 * lHC * cosh_kl) / 16. + (3 * k * sinh_kl) / 16. + (9 * k * cosh_2kl * sinh_kl) / 16. + (9 * k * cosh_kl * sinh_2kl) / 32.;
          U[3][3][3][3] = (-9 * cosh_kl) / 64. + (9 * cosh_3kl) / 64. + (3 * k * lHC * sinh_kl) / 16.;
          U[3][5][5][2] = (19 * k2 * lHC * cosh_kl) / 32. + (13 * k * sinh_kl) / 32. + (3 * k3 * l2 * sinh_kl) / 32.;
          U[3][5][5][3] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * lHC * sinh_kl) / 32.;
        }
      }
    } else {
      /* defocussing in horizontal plane */
      R[2][2] = R[3][3] = cos_kl;
      R[2][3] = sin_kl / k;
      R[3][2] = -k * sin_kl;
      R[0][0] = R[1][1] = cosh_kl;
      R[0][1] = sinh_kl / k;
      R[1][0] = k * sinh_kl;

      if (M->order >= 2) {
        double k2, k3, k4, l2, cos_kl, cos_2kl, cos_3kl, sin_kl, sin_2kl, sin_3kl, cosh_kl, cosh_2kl, cosh_3kl, sinh_kl, sinh_2kl, sinh_3kl;
        /* double l3, l4; */
        k2 = ipow2(k);
        k3 = ipow3(k);
        k4 = ipow4(k);
        l2 = ipow2(lHC);
        /* l3 = ipow3(lHC) ; */
        /* l4 = ipow4(lHC) ; */
        cos_kl = cos(k * lHC);
        cos_2kl = cos(2 * k * lHC);
        cos_3kl = cos(3 * k * lHC);
        sin_kl = sin(k * lHC);
        sin_2kl = sin(2 * k * lHC);
        sin_3kl = sin(3 * k * lHC);
        cosh_kl = cosh(k * lHC);
        cosh_2kl = cosh(2 * k * lHC);
        cosh_3kl = cosh(3 * k * lHC);
        sinh_kl = sinh(k * lHC);
        sinh_2kl = sinh(2 * k * lHC);
        sinh_3kl = sinh(3 * k * lHC);

        T = M->T;
        T[2][5][2] = T[3][5][3] = kl * sin_kl / 2;
        T[2][5][3] = sin_kl / (2 * k) - lHC * cos_kl / 2;
        T[3][5][2] = k / 2 * (kl * cos_kl + sin_kl);
        T[0][5][0] = T[1][5][1] = -kl / 2 * sinh_kl;
        T[0][5][1] = (sinh_kl / k - lHC * cosh_kl) / 2;
        T[1][5][0] = -k / 2 * (kl * cosh_kl + sinh_kl);
        T[4][0][0] = -sqr(k) * (lHC - sinh_kl / k * cosh_kl) / 4;
        T[4][1][0] = sqr(sinh_kl) / 2;
        T[4][1][1] = (lHC + sinh_kl / k * cosh_kl) / 4;
        T[4][2][2] = sqr(k) * (lHC - sin_kl / k * cos_kl) / 4;
        T[4][3][2] = -sqr(sin_kl) / 2;
        T[4][3][3] = (lHC + sin_kl / k * cos_kl) / 4;

        if (M->order >= 3) {
          U = M->Q;
          U[0][0][0][0] = (-3 * k3 * lHC * sinh_kl) / 16. + (3 * k2 * sinh_kl * sinh_2kl) / 32.;
          U[0][1][0][0] = (-3 * k2 * lHC * cosh_kl) / 16. - (15 * k * sinh_kl) / 64. + (9 * k * sinh_3kl) / 64.;
          U[0][1][1][0] = (3 * k * lHC * sinh_kl) / 16. + (9 * sinh_kl * sinh_2kl) / 32.;
          U[0][1][1][1] = (3 * lHC * cosh_kl) / 16. - (21 * sinh_kl) / (64. * k) + (3 * sinh_3kl) / (64. * k);
          U[0][2][2][0] = (k2 * cosh_kl * pow(sin_kl, 2)) / 16. + (k3 * lHC * sinh_kl) / 8. - (3 * k2 * sin_2kl * sinh_kl) / 32.;
          U[0][2][2][1] = (k2 * lHC * cosh_kl) / 8. - (3 * k * cosh_kl * sin_2kl) / 32. + (3 * k * sinh_kl) / 32. - (k * cos_2kl * sinh_kl) / 32.;
          U[0][3][2][0] = -(k * cosh_kl * sin_2kl) / 16. - (k * sinh_kl) / 16. + (3 * k * cos_2kl * sinh_kl) / 16.;
          U[0][3][2][1] = (-3 * cosh_kl * pow(sin_kl, 2)) / 8. - (cos_kl * sin_kl * sinh_kl) / 8.;
          U[0][3][3][0] = (k * lHC * sinh_kl) / 8. + (3 * sin_2kl * sinh_kl) / 32. + (pow(sin_kl, 2) * sinh_kl * sinh_2kl) / (16. * (1 - cosh_2kl));
          U[0][3][3][1] = (lHC * cosh_kl) / 8. + (3 * cosh_kl * sin_2kl) / (32. * k) - (11 * sinh_kl) / (32. * k) + (cos_2kl * sinh_kl) / (32. * k);
          U[0][5][5][0] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * lHC * sinh_kl) / 32.;
          U[0][5][5][1] = (7 * lHC * cosh_kl) / 32. - (7 * sinh_kl) / (32. * k) + (3 * k * l2 * sinh_kl) / 32.;
          U[1][0][0][0] = (-3 * k4 * lHC * cosh_kl) / 16. - (3 * k3 * sinh_kl) / 16. + (3 * k3 * cosh_2kl * sinh_kl) / 16. + (3 * k3 * cosh_kl * sinh_2kl) / 32.;
          U[1][1][0][0] = (-27 * k2 * cosh_kl) / 64. + (27 * k2 * cosh_3kl) / 64. - (3 * k3 * lHC * sinh_kl) / 16.;
          U[1][1][1][0] = (3 * k2 * lHC * cosh_kl) / 16. + (3 * k * sinh_kl) / 16. + (9 * k * cosh_2kl * sinh_kl) / 16. + (9 * k * cosh_kl * sinh_2kl) / 32.;
          U[1][1][1][1] = (-9 * cosh_kl) / 64. + (9 * cosh_3kl) / 64. + (3 * k * lHC * sinh_kl) / 16.;
          U[1][2][2][0] = (k4 * lHC * cosh_kl) / 8. + (k3 * cos_kl * cosh_kl * sin_kl) / 8. - (3 * k3 * cosh_kl * sin_2kl) / 32. + (k3 * sinh_kl) / 8. - (3 * k3 * cos_2kl * sinh_kl) / 16. + (k3 * pow(sin_kl, 2) * sinh_kl) / 16.;
          U[1][2][2][1] = (7 * k2 * cosh_kl) / 32. - (7 * k2 * cos_2kl * cosh_kl) / 32. + (k3 * lHC * sinh_kl) / 8. - (k2 * sin_2kl * sinh_kl) / 32.;
          U[1][3][2][0] = -(k2 * cosh_kl) / 16. + (k2 * cos_2kl * cosh_kl) / 16. - (7 * k2 * sin_2kl * sinh_kl) / 16.;
          U[1][3][2][1] = (-7 * k * cos_kl * cosh_kl * sin_kl) / 8. - (k * pow(cos_kl, 2) * sinh_kl) / 8. - (k * pow(sin_kl, 2) * sinh_kl) / 4.;
          U[1][3][3][0] = (k2 * lHC * cosh_kl) / 8. + (3 * k * cosh_kl * sin_2kl) / 32. + (k * sinh_kl) / 8. + (3 * k * cos_2kl * sinh_kl) / 16. + (k * pow(sin_kl, 2) * pow(sinh_kl, 3)) / (4. * pow(1 - cosh_2kl, 2)) + (k * cosh_kl * pow(sin_kl, 2) * sinh_2kl) / (16. * (1 - cosh_2kl)) + (k * cos_kl * sin_kl * sinh_kl * sinh_2kl) / (8. * (1 - cosh_2kl));
          U[1][3][3][1] = (-7 * cosh_kl) / 32. + (7 * cos_2kl * cosh_kl) / 32. + (k * lHC * sinh_kl) / 8. + (sin_2kl * sinh_kl) / 32.;
          U[1][5][5][0] = (19 * k2 * lHC * cosh_kl) / 32. + (13 * k * sinh_kl) / 32. + (3 * k3 * l2 * sinh_kl) / 32.;
          U[1][5][5][1] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * lHC * sinh_kl) / 32.;
          U[2][2][0][0] = (k3 * lHC * sin_kl) / 8. + (k2 * cos_kl * pow(sinh_kl, 2)) / 16. - (3 * k2 * sin_kl * sinh_2kl) / 32.;
          U[2][2][1][0] = (k * sin_kl) / 16. - (3 * k * cosh_2kl * sin_kl) / 16. + (k * cos_kl * sinh_2kl) / 16.;
          U[2][2][1][1] = -(k * lHC * sin_kl) / 8. - (sin_kl * sin_2kl * pow(sinh_kl, 2)) / (16. * (-1 + cos_2kl)) - (3 * sin_kl * sinh_2kl) / 32.;
          U[2][2][2][2] = (-3 * k3 * lHC * sin_kl) / 16. + (3 * k2 * sin_kl * sin_2kl) / 32.;
          U[2][3][0][0] = -(k2 * lHC * cos_kl) / 8. - (3 * k * sin_kl) / 32. + (k * cosh_2kl * sin_kl) / 32. + (3 * k * cos_kl * sinh_2kl) / 32.;
          U[2][3][1][0] = (cosh_kl * sin_kl * sinh_kl) / 8. + (3 * cos_kl * pow(sinh_kl, 2)) / 8.;
          U[2][3][1][1] = (lHC * cos_kl) / 8. - (11 * sin_kl) / (32. * k) + (cosh_2kl * sin_kl) / (32. * k) + (3 * cos_kl * sinh_2kl) / (32. * k);
          U[2][3][2][2] = (3 * k2 * lHC * cos_kl) / 16. + (15 * k * sin_kl) / 64. - (9 * k * sin_3kl) / 64.;
          U[2][3][3][2] = (-3 * k * lHC * sin_kl) / 16. - (9 * sin_kl * sin_2kl) / 32.;
          U[2][3][3][3] = (3 * lHC * cos_kl) / 16. - (21 * sin_kl) / (64. * k) + (3 * sin_3kl) / (64. * k);
          U[2][5][5][2] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * lHC * sin_kl) / 32.;
          U[2][5][5][3] = (7 * lHC * cos_kl) / 32. - (7 * sin_kl) / (32. * k) - (3 * k * l2 * sin_kl) / 32.;
          U[3][2][0][0] = (k4 * lHC * cos_kl) / 8. + (k3 * sin_kl) / 8. - (3 * k3 * cosh_2kl * sin_kl) / 16. + (k3 * cos_kl * cosh_kl * sinh_kl) / 8. - (k3 * sin_kl * pow(sinh_kl, 2)) / 16. - (3 * k3 * cos_kl * sinh_2kl) / 32.;
          U[3][2][1][0] = (k2 * cos_kl) / 16. - (k2 * cos_kl * cosh_2kl) / 16. - (7 * k2 * sin_kl * sinh_2kl) / 16.;
          U[3][2][1][1] = -(k2 * lHC * cos_kl) / 8. - (k * sin_kl) / 8. - (3 * k * cosh_2kl * sin_kl) / 16. - (k * cosh_kl * sin_kl * sin_2kl * sinh_kl) / (8. * (-1 + cos_2kl)) - (k * pow(sin_kl, 3) * pow(sinh_kl, 2)) / (4. * pow(-1 + cos_2kl, 2)) - (k * cos_kl * sin_2kl * pow(sinh_kl, 2)) / (16. * (-1 + cos_2kl)) - (3 * k * cos_kl * sinh_2kl) / 32.;
          U[3][2][2][2] = (-3 * k4 * lHC * cos_kl) / 16. - (3 * k3 * sin_kl) / 16. + (3 * k3 * cos_2kl * sin_kl) / 16. + (3 * k3 * cos_kl * sin_2kl) / 32.;
          U[3][3][0][0] = (-7 * k2 * cos_kl) / 32. + (7 * k2 * cos_kl * cosh_2kl) / 32. + (k3 * lHC * sin_kl) / 8. - (k2 * sin_kl * sinh_2kl) / 32.;
          U[3][3][1][0] = (k * pow(cosh_kl, 2) * sin_kl) / 8. + (7 * k * cos_kl * cosh_kl * sinh_kl) / 8. - (k * sin_kl * pow(sinh_kl, 2)) / 4.;
          U[3][3][1][1] = (-7 * cos_kl) / 32. + (7 * cos_kl * cosh_2kl) / 32. - (k * lHC * sin_kl) / 8. - (sin_kl * sinh_2kl) / 32.;
          U[3][3][2][2] = (27 * k2 * cos_kl) / 64. - (27 * k2 * cos_3kl) / 64. - (3 * k3 * lHC * sin_kl) / 16.;
          U[3][3][3][2] = (-3 * k2 * lHC * cos_kl) / 16. - (3 * k * sin_kl) / 16. - (9 * k * cos_2kl * sin_kl) / 16. - (9 * k * cos_kl * sin_2kl) / 32.;
          U[3][3][3][3] = (-9 * cos_kl) / 64. + (9 * cos_3kl) / 64. - (3 * k * lHC * sin_kl) / 16.;
          U[3][5][5][2] = (-19 * k2 * lHC * cos_kl) / 32. - (13 * k * sin_kl) / 32. + (3 * k3 * l2 * sin_kl) / 32.;
          U[3][5][5][3] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * lHC * sin_kl) / 32.;
        }
      }
    }

    if (fringeCode == INSET_FRINGE || fringeCode == FIXED_STRENGTH_FRINGE) {
      if (lEdge && K1) {
        Md = NULL;
        Mtot = tmalloc(sizeof(*Mtot));
        initialize_matrices(Mtot, M->order);

        /* entrance fringe fields */
        Mfringe = quad_fringe(lEdge, K1, M->order, 0, 0.0);

        if (fringeCode == FIXED_STRENGTH_FRINGE) {
          /* drift back to fringe entrance */
          Md = drift_matrix(-lEdge / 2, M->order);
          concat_matrices(Mtot, Mfringe, Md, 0);
          tmp = Mfringe;
          Mfringe = Mtot;
          Mtot = tmp;
        }

        concat_matrices(Mtot, M, Mfringe, 0);
        tmp = Mtot;
        Mtot = M;
        M = tmp;
        free_matrices(Mfringe);
        tfree(Mfringe);
        Mfringe = NULL;

        /* exit fringe fields */
        Mfringe = quad_fringe(lEdge, K1, M->order, 1, 0.0);
        concat_matrices(Mtot, Mfringe, M, 0);
        free_matrices(Mfringe);
        tfree(Mfringe);
        Mfringe = NULL;
        tmp = Mtot;
        Mtot = M;
        M = tmp;

        if (fringeCode == FIXED_STRENGTH_FRINGE) {
          /* drift back to quad exit plane */
          concat_matrices(Mtot, Md, M, 0);
          tmp = M;
          M = Mtot;
          Mtot = tmp;
          free_matrices(Md);
          tfree(Md);
          Md = NULL;
        }
        free_matrices(Mtot);
        tfree(Mtot);
        Mtot = NULL;
      }
    } else if (lEdge != 0 || (fringeCode == INTEGRALS_FRINGE && (edge1_effects || edge2_effects))) {
      short hasFringeIntegrals = 0, i;
      Mtot = tmalloc(sizeof(*Mtot));
      initialize_matrices(Mtot, M->order);
      if (fringeCode == INTEGRALS_FRINGE && (edge1_effects || edge2_effects)) {
        for (i = 0; i < 5; i++)
          if (fringeIntM[i] || fringeIntP[i]) {
            hasFringeIntegrals = 1;
            break;
          }
        if (hasFringeIntegrals) {
          Mfringe = NULL;
          if (edge1_effects) {
            Mfringe = quadFringeMatrix(NULL, K1, lHC < 0, -1, fringeIntM, fringeIntP);
            concat_matrices(Mtot, M, Mfringe, 0);
            free_matrices(Mfringe);
            free(Mfringe);
            Mfringe = NULL;
            tmp = M;
            M = Mtot;
            Mtot = tmp;
          }
          if (edge2_effects) {
            Mfringe = quadFringeMatrix(Mfringe, K1, lHC < 0, 1, fringeIntM, fringeIntP);
            concat_matrices(Mtot, Mfringe, M, 0);
            free_matrices(Mfringe);
            free(Mfringe);
            Mfringe = NULL;
            tmp = M;
            M = Mtot;
            Mtot = tmp;
          }
        }
      }
      if (lEdge != 0) {
        /* drift from nominal entrance to beginning of field */
        Md = drift_matrix(lEdge, M->order);
        concat_matrices(Mtot, M, Md, 0);
        tmp = M;
        M = Mtot;
        Mtot = tmp;

        /* drift from nominal entrance to beginning of field */
        concat_matrices(Mtot, Md, M, 0);
        tmp = M;
        M = Mtot;
        Mtot = tmp;
        free_matrices(Md);
        free(Md);
        Md = NULL;
      }
      free_matrices(Mtot);
      free(Mtot);
      Mtot = NULL;
    }
  }

  if (radial) {
    long i, j, k, l;
    /* make a radially-focusing lens by copying the horizontal matrix to the vertical */
    for (i = 2; i < 4; i++)
      for (j = 2; j < 4; j++)
        M->R[i][j] = M->R[i - 2][j - 2];
    if (M->order > 1) {
      for (i = 0; i < 2; i++) {
        for (j = 0; j < 6; j++) {
          if (j == 2 || j == 3)
            continue;
          for (k = 0; k <= j; k++) {
            if (k == 2 || k == 3)
              continue;
            if (j < 4) {
              /* j=(0, 1) */
              if (k < 2) {
                /* k=(0, 1) */
                M->T[i + 2][j + 2][k + 2] = M->T[i][j][k];
              } else {
                /* k=(4, 5) */
                M->T[i + 2][j + 2][k] = M->T[i][j][k];
              }
            } else {
              /* j=(4, 5) */
              if (k < 2) {
                /* k=(0, 1) */
                if ((k + 2) <= j)
                  M->T[i + 2][j][k + 2] = M->T[i][j][k];
              } else {
                /* k=(4, 5) */
                M->T[i + 2][j][k] = M->T[i][j][k];
              }
            }
          }
        }
      }
      for (i = 4; i < 6; i++) {
        for (j = 0; j < 6; j++) {
          if (j == 2 || j == 3)
            continue;
          for (k = 0; k <= j; k++) {
            if (k == 2 || k == 3)
              continue;
            if (j < 4) {
              /* j=(0, 1) */
              if (k < 2) {
                /* k=(0, 1) */
                M->T[i][j + 2][k + 2] = M->T[i][j][k];
              } else {
                /* k=(4, 5) */
                M->T[i][j + 2][k] = M->T[i][j][k];
              }
            } else {
              /* j=(4, 5) */
              if (k < 2) {
                /* k=(0, 1) */
                if ((k + 2) <= j)
                  M->T[i][j][k + 2] = M->T[i][j][k];
              }
            }
          }
        }
      }
      if (M->order > 2) {
        for (i = 0; i < 2; i++) {
          for (j = 0; j < 6; j++) {
            if (j == 2 || j == 3)
              continue;
            for (k = 0; k <= j; k++) {
              if (k == 2 || k == 3)
                continue;
              for (l = 0; l <= k; l++) {
                if (l == 2 || l == 3)
                  continue;
                if (j < 4) {
                  /* j=(0, 1) */
                  if (k < 2) {
                    /* k=(0, 1) */
                    if (l < 2) {
                      M->Q[i + 2][j + 2][k + 2][l + 2] = M->Q[i][j][k][l];
                    } else {
                      /* l=(4, 5) */
                      M->Q[i + 2][j + 2][k + 2][l] = M->Q[i][j][k][l];
                    }
                  } else {
                    /* k=(4, 5) */
                    if ((l + 2) <= k)
                      M->Q[i + 2][j + 2][k][l + 2] = M->Q[i][j][k][l];
                  }
                } else {
                  /* j=(4, 5) */
                  if (k < 2) {
                    /* k=(0, 1) */
                    if ((k + 2) <= j) {
                      if (l < 2) {
                        M->Q[i + 2][j][k + 2][l + 2] = M->Q[i][j][k][l];
                      } else {
                        /* l=(4, 5) */
                        M->Q[i + 2][j][k + 2][l] = M->Q[i][j][k][l];
                      }
                    }
                  } else {
                    /* k=(4, 5) */
                    if ((l + 2) <= k)
                      M->Q[i + 2][j][k][l + 2] = M->Q[i][j][k][l];
                  }
                }
              }
            }
          }
        }
        for (i = 4; i < 6; i++) {
          for (j = 0; j < 6; j++) {
            if (j == 2 || j == 3)
              continue;
            for (k = 0; k <= j; k++) {
              if (k == 2 || k == 3)
                continue;
              for (l = 0; l <= k; l++) {
                if (l == 2 || l == 3)
                  continue;
                if (j < 4) {
                  /* j=(0, 1) */
                  if (k < 2) {
                    /* k=(0, 1) */
                    if (l < 2) {
                      M->Q[i][j + 2][k + 2][l + 2] = M->Q[i][j][k][l];
                    } else {
                      /* l=(4, 5) */
                      M->Q[i][j + 2][k + 2][l] = M->Q[i][j][k][l];
                    }
                  } else {
                    /* k=(4, 5) */
                    if ((l + 2) <= k)
                      M->Q[i][j + 2][k][l + 2] = M->Q[i][j][k][l];
                  }
                } else {
                  /* j=(4, 5) */
                  if (k < 2) {
                    /* k=(0, 1) */
                    if ((k + 2) <= j) {
                      if (l < 2) {
                        M->Q[i][j][k + 2][l + 2] = M->Q[i][j][k][l];
                      } else {
                        /* l=(4, 5) */
                        M->Q[i][j][k + 2][l] = M->Q[i][j][k][l];
                      }
                    }
                  } else {
                    /* k=(4, 5) */
                    if ((l + 2) <= k)
                      M->Q[i][j][k][l + 2] = M->Q[i][j][k][l];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (xkick || ykick) {
    /* put identical kicks at the entrance and exit */
    Mtot = tmalloc(sizeof(*Mtot));
    initialize_matrices(Mtot, M->order);
    Mfringe = hvcorrector_matrix(0, xkick / 2, ykick / 2, 0.0, 0.0, 1.0, 1.0, 0, M->order);
    concat_matrices(Mtot, Mfringe, M, 0);
    concat_matrices(M, Mtot, Mfringe, 0);
    free_matrices(Mfringe);
    tfree(Mfringe);
    Mfringe = NULL;
    free_matrices(Mtot);
    tfree(Mtot);
    Mtot = NULL;
  }

  return (M);
}

VMATRIX *quad_fringe(double l, double ko, long order, long reverse, double fse) {
  VMATRIX *M;

  log_entry("quad_fringe");

  ko *= (1 + fse);
  M = tmalloc(sizeof(*M));
  initialize_matrices(M, order);

  M->C[4] = l;
  M->R[5][5] = M->R[4][4] = 1;

  qfringe_R_matrix(
    &M->R[0][0], &M->R[0][1], &M->R[1][0], &M->R[1][1], ko / l, l);
  qfringe_R_matrix(
    &M->R[2][2], &M->R[2][3], &M->R[3][2], &M->R[3][3], -ko / l, l);
  if (reverse) {
    swap_double(M->R[0][0], M->R[1][1]);
    swap_double(M->R[2][2], M->R[3][3]);
  }

  if (order >= 2) {
    qfringe_T_matrix(
      &M->T[0][5][0], &M->T[0][5][1], &M->T[1][5][0], &M->T[1][5][1],
      &M->T[4][0][0], &M->T[4][1][0], &M->T[4][1][1],
      ko / l, l, reverse);
    qfringe_T_matrix(
      &M->T[2][5][2], &M->T[2][5][3], &M->T[3][5][2], &M->T[3][5][3],
      &M->T[4][2][2], &M->T[4][3][2], &M->T[4][3][3],
      -ko / l, l, reverse);
  }

  log_exit("quad_fringe");
  return (M);
}

void qfringe_R_matrix(
  double *R11, double *R12,
  double *R21, double *R22,
  double dk_dz, double l) {
  double term, l3;
  long n;

  log_entry("qfringe_R_matrix");

  if (!l) {
    *R11 = *R22 = 1;
    *R21 = *R12 = 0;
    log_exit("qfringe_R_matrix");
    return;
  }
  if (!dk_dz) {
    *R11 = *R22 = 1;
    *R21 = 0;
    *R12 = l;
    log_exit("qfringe_R_matrix");
    return;
  }

  l3 = pow3(l);

  /* compute R11, R21 */
  *R11 = *R21 = 0;
  term = 1;
  n = 0;
  do {
    *R11 += term;
    *R21 += n * term;
    term *= -l3 * dk_dz / ((n + 3) * (n + 2));
    n += 3;
  } while (FABS(term / (*R11)) > 1e-16);
  *R21 /= l;

  /* compute R12, R22 */
  *R12 = 0;
  *R22 = 0;
  term = l;
  n = 1;
  do {
    *R12 += term;
    *R22 += n * term;
    term *= -l3 * dk_dz / ((n + 3) * (n + 2));
    n += 3;
  } while (FABS(term / (*R12)) > 1e-16);
  *R22 /= l;
  log_exit("qfringe_R_matrix");
}

void qfringe_T_matrix(
  double *T116, double *T126, double *T216, double *T226,
  double *T511, double *T512, double *T522,
  double dk_dz, double l, long reverse) {
  double term, l3, ko, mult;
  long n;

  log_entry("qfringe_T_matrix");

  if (!l || !dk_dz) {
    *T116 = *T226 = *T216 = *T126 = *T511 = *T512 = *T522 = 0;
    log_exit("qfringe_T_matrix");
    return;
  }

  l3 = pow3(l);

  /* compute T116, T216 */
  *T116 = *T216 = 0;
  term = 1;
  n = 0;
  do {
    *T116 += -n * term / 3;
    *T216 += -sqr(n) * term / 3;
    term *= -l3 * dk_dz / ((n + 3) * (n + 2));
    n += 3;
  } while (FABS(term / (*T116 ? *T116 : 1)) > 1e-16);
  *T216 /= l;

  /* compute T126, T226 */
  *T126 = 0;
  *T226 = 0;
  term = l;
  n = 1;
  do {
    *T126 += -(n - 1) * term / 3;
    *T226 += -n * (n - 1) * term / 3;
    term *= -l3 * dk_dz / ((n + 3) * (n + 2));
    n += 3;
  } while (FABS(term / (*T126 ? *T126 : 1)) > 1e-16);
  *T226 /= l;

  if (reverse)
    swap_double(*T116, *T226);

  /* compute path-length terms using truncated series (truncated at z^12) */
  if (!reverse) {
    /* entrance fringe field */
    ko = dk_dz * l;
    /* T511 terms */
    term = sqr(ko) * pow3(l);
    if ((mult = ko * sqr(l)) > 1)
      printWarning("Path-length terms for QFRINGE may be inaccurate", "ko*lf^2 > 1");
    *T511 = 1. / 20. * term;
    term *= mult;
    *T511 += -1. / 240. * term;
    term *= mult;
    *T511 += 13. / 79200. * term;
    term *= mult;
    *T511 += -19. / 4989600. * term;
    term *= mult;
    *T511 += 19. / 325721088. * term;
    /* T522 terms */
    *T522 = term = l;
    term *= mult;
    *T522 += -1. / 6. * term;
    term *= mult;
    *T522 += 5. / 252. * term;
    term *= mult;
    *T522 += -11. / 11340. * term;
    term *= mult;
    *T522 += 187. / 7076160. * term;
    term *= mult;
    *T522 += -391. / 849139200. * term;
    *T522 /= 2;
    /* T512 terms */
    term = mult;
    *T512 = -1. / 6. * term;
    term *= mult;
    *T512 += 1. / 30. * term;
    term *= mult;
    *T512 += -1. / 480. * term;
    term *= mult;
    *T512 += 1. / 14784. * term;
    term *= mult;
    *T512 += -1. / 739200. * term;
    term *= mult;
    *T512 += 1. / 54454400. * term;
  } else {
    /* exit fringe field */
    ko = dk_dz * l;
    /* T511 terms */
    term = sqr(ko) * pow3(l);
    if ((mult = ko * sqr(l)) > 1)
      printWarning("Path-length terms for qfringe may be inaccurate.", "ko*lf^2 > 1.");
    *T511 = 2. / 15. * term;
    term *= mult;
    *T511 += -1. / 80. * term;
    term *= mult;
    *T511 += 1. / 1848. * term;
    term *= mult;
    *T511 += -1. / 73920 * term;
    term *= mult;
    *T511 += 3. / 13613600. * term;
    /* T522 terms */
    *T522 = term = l;
    term *= mult;
    *T522 += -1. / 6. * term;
    term *= mult;
    *T522 += 1. / 70. * term;
    term *= mult;
    *T522 += -1. / 1680. * term;
    term *= mult;
    *T522 += 1. / 68640. * term;
    term *= mult;
    *T522 += -3. / 12812800. * term;
    *T522 /= 2;
    /* T512 terms */
    term = mult;
    *T512 = -1. / 3. * term;
    term *= mult;
    *T512 += 1. / 20. * term;
    term *= mult;
    *T512 += -1. / 336. * term;
    term *= mult;
    *T512 += 1. / 10560. * term;
    term *= mult;
    *T512 += -3 / 1601600. * term;
    term *= mult;
    *T512 += 1. / 39603200. * term;
  }
  log_exit("qfringe_T_matrix");
}

/* This subroutine returns a stand-alone quadrpole fringe
 * matrix, in the event the user asks for a qfringe element
 * separate from a quadrupole
 */

VMATRIX *qfringe_matrix(
  double K1, double l, long direction, long order, double fse) {
  VMATRIX *M;

  log_entry("qfringe_matrix");

  M = quad_fringe(l, K1, order, (direction == -1 ? 1 : 0), fse);

  M->C[4] = l;
  M->R[5][5] = M->R[4][4] = 1;

  log_exit("qfringe_matrix");
  return (M);
}

VMATRIX *quse_matrix(double K1, double K2, double length, long maximum_order,
                     double fse1, double fse2) {
  VMATRIX *M;
  double *C, **R, ***T, ****U;
  double k, kl;

  if (K1 == 0 || length == 0) {
    if (K2 == 0)
      return drift_matrix(length, maximum_order);
    return sextupole_matrix(K2, 0.0, 0.0, length, maximum_order, fse2, 0.0, 0.0, 0);
  }
  if (fabs(K1 * (1 + fse1) * length) < 1e-6) {
    if (K2 != 0)
      return sextupole_matrix(K2, K1, 0.0, length, maximum_order, fse2, 0.0, 0.0, 0);
    return quadrupole_matrix(K1, length, maximum_order, fse1, 0.0, 0.0, 0, 0, NULL, 0.0, -1.0, NULL, NULL, 0);
  }

  K1 *= (1 + fse1);
  K2 *= (1 + fse2);

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(3, maximum_order));
  C = M->C;
  R = M->R;

  if (K1 > 0) {
    double k2, k3, k4, k5, k6, k7, l, l2;
    double cos_kl, cos_2kl, cos_3kl, sin_kl, sin_2kl, sin_3kl, cosh_kl, cosh_2kl, cosh_3kl, sinh_kl, sinh_2kl, sinh_3kl;
    /* double k8; */

    k = sqrt(K1);
    kl = k * length;
    l = length;

    k2 = ipow2(k);
    k3 = ipow3(k);
    k4 = ipow4(k);
    k5 = ipow5(k);
    k6 = ipow6(k);
    k7 = ipow7(k);
    /* k8 = ipow8(k) ; */
    l2 = ipow2(l);
    cos_kl = cos(kl);
    cos_2kl = cos(2 * kl);
    cos_3kl = cos(3 * kl);
    sin_kl = sin(kl);
    sin_2kl = sin(2 * kl);
    sin_3kl = sin(3 * kl);
    cosh_kl = cosh(kl);
    cosh_2kl = cosh(2 * kl);
    cosh_3kl = cosh(3 * kl);
    sinh_kl = sinh(kl);
    sinh_2kl = sinh(2 * kl);
    sinh_3kl = sinh(3 * kl);

    C[4] = length;

    R[0][0] = cos_kl;
    R[0][1] = sin_kl / k;
    R[1][0] = -(k * sin_kl);
    R[1][1] = cos_kl;
    R[2][2] = cosh_kl;
    R[2][3] = sinh_kl / k;
    R[3][2] = k * sinh_kl;
    R[3][3] = cosh_kl;
    R[4][4] = 1;
    R[5][5] = 1;

    if (M->order > 1) {
      T = M->T;
      T[0][0][0] = (-2 * K2 * ipow2(sin((k * l) / 2.))) / (3. * k2) - (K2 * cos_kl * ipow2(sin((k * l) / 2.))) / (3. * k2);
      T[0][1][0] = -(K2 * sin_kl) / (3. * k3) + (K2 * sin_2kl) / (6. * k3);
      T[0][1][1] = (-2 * K2 * ipow4(sin((k * l) / 2.))) / (3. * k4);
      T[0][2][2] = K2 / (4. * k2) - (3 * K2 * cos_kl) / (10. * k2) + (K2 * cosh_2kl) / (20. * k2);
      T[0][3][2] = -(K2 * sin_kl) / (5. * k3) + (K2 * sinh_2kl) / (10. * k3);
      T[0][3][3] = -K2 / (4. * k4) + (K2 * cos_kl) / (5. * k4) + (K2 * cosh_2kl) / (20. * k4);
      T[0][5][0] = (k * l * sin_kl) / 2.;
      T[0][5][1] = -(l * cos_kl) / 2. + sin_kl / (2. * k);
      T[1][0][0] = (-2 * K2 * cos((k * l) / 2.) * sin((k * l) / 2.)) / (3. * k) - (K2 * cos((k * l) / 2.) * cos_kl * sin((k * l) / 2.)) / (3. * k) + (K2 * ipow2(sin((k * l) / 2.)) * sin_kl) / (3. * k);
      T[1][1][0] = -(K2 * cos_kl) / (3. * k2) + (K2 * cos_2kl) / (3. * k2);
      T[1][1][1] = (-4 * K2 * cos((k * l) / 2.) * ipow3(sin((k * l) / 2.))) / (3. * k3);
      T[1][2][2] = (3 * K2 * sin_kl) / (10. * k) + (K2 * sinh_2kl) / (10. * k);
      T[1][3][2] = -(K2 * cos_kl) / (5. * k2) + (K2 * cosh_2kl) / (5. * k2);
      T[1][3][3] = -(K2 * sin_kl) / (5. * k3) + (K2 * sinh_2kl) / (10. * k3);
      T[1][5][0] = (k2 * l * cos_kl) / 2. + (k * sin_kl) / 2.;
      T[1][5][1] = (k * l * sin_kl) / 2.;
      T[2][2][0] = (K2 * cosh_kl) / (5. * k2) - (K2 * cos_kl * cosh_kl) / (5. * k2) + (2 * K2 * sin_kl * sinh_kl) / (5. * k2);
      T[2][2][1] = -(K2 * cosh_kl * sin_kl) / (5. * k3) + (3 * K2 * sinh_kl) / (5. * k3) - (2 * K2 * cos_kl * sinh_kl) / (5. * k3);
      T[2][3][0] = (2 * K2 * cosh_kl * sin_kl) / (5. * k3) - (K2 * sinh_kl) / (5. * k3) - (K2 * cos_kl * sinh_kl) / (5. * k3);
      T[2][3][1] = (2 * K2 * cosh_kl) / (5. * k4) - (2 * K2 * cos_kl * cosh_kl) / (5. * k4) - (K2 * sin_kl * sinh_kl) / (5. * k4);
      T[2][5][2] = -(k * l * sinh_kl) / 2.;
      T[2][5][3] = -(l * cosh_kl) / 2. + sinh_kl / (2. * k);
      T[3][2][0] = (3 * K2 * cosh_kl * sin_kl) / (5. * k) + (K2 * sinh_kl) / (5. * k) + (K2 * cos_kl * sinh_kl) / (5. * k);
      T[3][2][1] = (3 * K2 * cosh_kl) / (5. * k2) - (3 * K2 * cos_kl * cosh_kl) / (5. * k2) + (K2 * sin_kl * sinh_kl) / (5. * k2);
      T[3][3][0] = -(K2 * cosh_kl) / (5. * k2) + (K2 * cos_kl * cosh_kl) / (5. * k2) + (3 * K2 * sin_kl * sinh_kl) / (5. * k2);
      T[3][3][1] = (K2 * cosh_kl * sin_kl) / (5. * k3) + (2 * K2 * sinh_kl) / (5. * k3) - (3 * K2 * cos_kl * sinh_kl) / (5. * k3);
      T[3][5][2] = -(k2 * l * cosh_kl) / 2. - (k * sinh_kl) / 2.;
      T[3][5][3] = -(k * l * sinh_kl) / 2.;
      T[4][0][0] = (k2 * l) / 4. - (k * sin_2kl) / 8.;
      T[4][1][0] = -ipow2(sin_kl) / 2.;
      T[4][1][1] = l / 4. + sin_2kl / (8. * k);
      T[4][2][2] = -(k2 * l) / 4. + (k * sinh_2kl) / 8.;
      T[4][3][2] = ipow2(sinh_kl) / 2.;
      T[4][3][3] = l / 4. + sinh_2kl / (8. * k);
    }
    if (M->order > 2) {
      U = M->Q;
      U[0][0][0][0] = -ipow2(K2) / (12. * k4) + (3 * k2 * cos_kl) / 64. + (29 * ipow2(K2) * cos_kl) / (576. * k4) + (ipow2(K2) * cos_2kl) / (36. * k4) - (3 * k2 * cos_3kl) / 64. + (ipow2(K2) * cos_3kl) / (192. * k4) - (3 * k3 * l * sin_kl) / 16. + (5 * ipow2(K2) * l * sin_kl) / (48. * k3);
      U[0][1][0][0] = (3 * k2 * l * cos_kl) / 16. - (3 * ipow2(K2) * l * cos_kl) / (16. * k4) + (15 * k * sin_kl) / 64. + (9 * ipow2(K2) * sin_kl) / (64. * k5) - (9 * k * sin_3kl) / 64. + (ipow2(K2) * sin_3kl) / (64. * k5);
      U[0][1][1][0] = (-3 * k * l * sin_kl) / 16. + (3 * ipow2(K2) * l * sin_kl) / (16. * k5) - (ipow2(K2) * ipow2(sin_kl)) / (4. * k6) - (9 * sin_kl * sin_2kl) / 32. + (ipow2(K2) * sin_kl * sin_2kl) / (32. * k6);
      U[0][1][1][1] = (3 * l * cos_kl) / 16. - (5 * ipow2(K2) * l * cos_kl) / (48. * k6) - (21 * sin_kl) / (64. * k) + (5 * ipow2(K2) * sin_kl) / (576. * k7) + (ipow2(K2) * sin_2kl) / (18. * k7) + (3 * sin_3kl) / (64. * k) - (ipow2(K2) * sin_3kl) / (192. * k7);
      U[0][2][2][0] = (3 * ipow2(K2)) / (10. * k4) - (k2 * cos_kl) / 32. - (81 * ipow2(K2) * cos_kl) / (400. * k4) - (3 * ipow2(K2) * cos_2kl) / (40. * k4) + (3 * ipow2(K2) * cosh_2kl) / (200. * k4) + (k2 * cos_kl * cosh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (80. * k4) + (k3 * l * sin_kl) / 8. - (9 * ipow2(K2) * l * sin_kl) / (40. * k3) - (3 * k2 * sin_kl * sinh_2kl) / 32.;
      U[0][2][2][1] = -(k2 * l * cos_kl) / 8. + (9 * ipow2(K2) * l * cos_kl) / (40. * k4) - (3 * k * sin_kl) / 32. - (51 * ipow2(K2) * sin_kl) / (400. * k5) - (3 * ipow2(K2) * cos_kl * sin_kl) / (20. * k5) + (k * cosh_2kl * sin_kl) / 32. - (3 * ipow2(K2) * cosh_2kl * sin_kl) / (80. * k5) + (9 * ipow2(K2) * sinh_2kl) / (200. * k5) + (3 * k * cos_kl * sinh_2kl) / 32.;
      U[0][3][2][0] = (3 * ipow2(K2) * l * cos_kl) / (20. * k4) + (k * sin_kl) / 16. + (ipow2(K2) * sin_kl) / (25. * k5) - (3 * k * cosh_2kl * sin_kl) / 16. - (ipow2(K2) * sin_2kl) / (20. * k5) + (3 * ipow2(K2) * sinh_2kl) / (100. * k5) + (k * cos_kl * sinh_2kl) / 16. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (40. * k5);
      U[0][3][2][1] = (-3 * ipow2(K2)) / (10. * k6) - (3 * cos_kl) / 16. + (4 * ipow2(K2) * cos_kl) / (25. * k6) + (ipow2(K2) * cos_2kl) / (20. * k6) + (9 * ipow2(K2) * cosh_2kl) / (100. * k6) + (3 * cos_kl * cosh_2kl) / 16. + (3 * ipow2(K2) * l * sin_kl) / (20. * k5) + (sin_kl * sinh_2kl) / 16. - (3 * ipow2(K2) * sin_kl * sinh_2kl) / (40. * k6);
      U[0][3][3][0] = (-3 * ipow2(K2)) / (40. * k6) - cos_kl / 32. + (31 * ipow2(K2) * cos_kl) / (400. * k6) + (ipow2(K2) * cos_2kl) / (20. * k6) - (3 * ipow2(K2) * cosh_2kl) / (200. * k6) + (cos_kl * cosh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (80. * k6) - (k * l * sin_kl) / 8. + (9 * ipow2(K2) * l * sin_kl) / (40. * k5) - (3 * sin_kl * sinh_2kl) / 32.;
      U[0][3][3][1] = (l * cos_kl) / 8. - (9 * ipow2(K2) * l * cos_kl) / (40. * k6) - (11 * sin_kl) / (32. * k) + (41 * ipow2(K2) * sin_kl) / (400. * k7) + (cosh_2kl * sin_kl) / (32. * k) - (3 * ipow2(K2) * cosh_2kl * sin_kl) / (80. * k7) + (ipow2(K2) * sin_2kl) / (20. * k7) + (3 * ipow2(K2) * sinh_2kl) / (100. * k7) + (3 * cos_kl * sinh_2kl) / (32. * k);
      U[0][5][0][0] = -K2 / (8. * k2) + (K2 * cos_kl) / (6. * k2) - (K2 * cos_2kl) / (24. * k2) + (K2 * l * sin_kl) / (8. * k) + (K2 * l * sin_2kl) / (16. * k);
      U[0][5][1][0] = (3 * K2 * l) / (8. * k2) + (K2 * l * cos_kl) / (4. * k2) - (K2 * l * cos_2kl) / (8. * k2) - (7 * K2 * sin_kl) / (12. * k3) + (K2 * sin_2kl) / (24. * k3);
      U[0][5][1][1] = (-5 * K2) / (16. * k4) + (K2 * cos_kl) / (3. * k4) - (K2 * cos_2kl) / (48. * k4) + (K2 * l * sin_kl) / (4. * k3) - (K2 * l * sin_2kl) / (16. * k3);
      U[0][5][2][2] = K2 / (8. * k2) - (3 * K2 * cos_kl) / (25. * k2) - (K2 * cosh_2kl) / (200. * k2) - (9 * K2 * l * sin_kl) / (40. * k) - (3 * K2 * l * sinh_2kl) / (80. * k);
      U[0][5][3][2] = (-3 * K2 * l) / (8. * k2) + (3 * K2 * l * cos_kl) / (20. * k2) - (3 * K2 * l * cosh_2kl) / (40. * k2) + (17 * K2 * sin_kl) / (100. * k3) + (13 * K2 * sinh_2kl) / (200. * k3);
      U[0][5][3][3] = (-5 * K2) / (16. * k4) + (7 * K2 * cos_kl) / (25. * k4) + (13 * K2 * cosh_2kl) / (400. * k4) + (3 * K2 * l * sin_kl) / (20. * k3) - (3 * K2 * l * sinh_2kl) / (80. * k3);
      U[0][5][5][0] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * l * sin_kl) / 32.;
      U[0][5][5][1] = (7 * l * cos_kl) / 32. - (7 * sin_kl) / (32. * k) - (3 * k * l2 * sin_kl) / 32.;
      U[1][0][0][0] = (-3 * k4 * l * cos_kl) / 16. + (5 * ipow2(K2) * l * cos_kl) / (48. * k2) - (15 * k3 * sin_kl) / 64. + (31 * ipow2(K2) * sin_kl) / (576. * k3) - (ipow2(K2) * sin_2kl) / (18. * k3) + (9 * k3 * sin_3kl) / 64. - (ipow2(K2) * sin_3kl) / (64. * k3);
      U[1][1][0][0] = (27 * k2 * cos_kl) / 64. - (3 * ipow2(K2) * cos_kl) / (64. * k4) - (27 * k2 * cos_3kl) / 64. + (3 * ipow2(K2) * cos_3kl) / (64. * k4) - (3 * k3 * l * sin_kl) / 16. + (3 * ipow2(K2) * l * sin_kl) / (16. * k3);
      U[1][1][1][0] = (-3 * k2 * l * cos_kl) / 16. + (3 * ipow2(K2) * l * cos_kl) / (16. * k4) - (3 * k * sin_kl) / 16. + (3 * ipow2(K2) * sin_kl) / (16. * k5) - (ipow2(K2) * cos_kl * sin_kl) / (2. * k5) - (9 * k * cos_2kl * sin_kl) / 16. + (ipow2(K2) * cos_2kl * sin_kl) / (16. * k5) - (9 * k * cos_kl * sin_2kl) / 32. + (ipow2(K2) * cos_kl * sin_2kl) / (32. * k5);
      U[1][1][1][1] = (-9 * cos_kl) / 64. - (55 * ipow2(K2) * cos_kl) / (576. * k6) + (ipow2(K2) * cos_2kl) / (9. * k6) + (9 * cos_3kl) / 64. - (ipow2(K2) * cos_3kl) / (64. * k6) - (3 * k * l * sin_kl) / 16. + (5 * ipow2(K2) * l * sin_kl) / (48. * k5);
      U[1][2][2][0] = (k4 * l * cos_kl) / 8. - (9 * ipow2(K2) * l * cos_kl) / (40. * k2) + (5 * k3 * sin_kl) / 32. - (9 * ipow2(K2) * sin_kl) / (400. * k3) - (7 * k3 * cosh_2kl * sin_kl) / 32. + (3 * ipow2(K2) * cosh_2kl * sin_kl) / (80. * k3) + (3 * ipow2(K2) * sin_2kl) / (20. * k3) + (3 * ipow2(K2) * sinh_2kl) / (100. * k3) - (k3 * cos_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (40. * k3);
      U[1][2][2][1] = (-7 * k2 * cos_kl) / 32. + (39 * ipow2(K2) * cos_kl) / (400. * k4) - (3 * ipow2(K2) * ipow2(cos_kl)) / (20. * k4) + (9 * ipow2(K2) * cosh_2kl) / (100. * k4) + (7 * k2 * cos_kl * cosh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (80. * k4) + (k3 * l * sin_kl) / 8. - (9 * ipow2(K2) * l * sin_kl) / (40. * k3) + (3 * ipow2(K2) * ipow2(sin_kl)) / (20. * k4) - (k2 * sin_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * sin_kl * sinh_2kl) / (40. * k4);
      U[1][3][2][0] = (k2 * cos_kl) / 16. + (19 * ipow2(K2) * cos_kl) / (100. * k4) - (ipow2(K2) * cos_2kl) / (10. * k4) + (3 * ipow2(K2) * cosh_2kl) / (50. * k4) - (k2 * cos_kl * cosh_2kl) / 16. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (20. * k4) - (3 * ipow2(K2) * l * sin_kl) / (20. * k3) - (7 * k2 * sin_kl * sinh_2kl) / 16. + (3 * ipow2(K2) * sin_kl * sinh_2kl) / (40. * k4);
      U[1][3][2][1] = (3 * ipow2(K2) * l * cos_kl) / (20. * k4) + (3 * k * sin_kl) / 16. - (ipow2(K2) * sin_kl) / (100. * k5) - (k * cosh_2kl * sin_kl) / 16. - (3 * ipow2(K2) * cosh_2kl * sin_kl) / (20. * k5) - (ipow2(K2) * sin_2kl) / (10. * k5) + (9 * ipow2(K2) * sinh_2kl) / (50. * k5) + (7 * k * cos_kl * sinh_2kl) / 16. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (40. * k5);
      U[1][3][3][0] = -(k2 * l * cos_kl) / 8. + (9 * ipow2(K2) * l * cos_kl) / (40. * k4) - (3 * k * sin_kl) / 32. + (59 * ipow2(K2) * sin_kl) / (400. * k5) - (7 * k * cosh_2kl * sin_kl) / 32. + (3 * ipow2(K2) * cosh_2kl * sin_kl) / (80. * k5) - (ipow2(K2) * sin_2kl) / (10. * k5) - (3 * ipow2(K2) * sinh_2kl) / (100. * k5) - (k * cos_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (40. * k5);
      U[1][3][3][1] = (-7 * cos_kl) / 32. - (49 * ipow2(K2) * cos_kl) / (400. * k6) + (ipow2(K2) * cos_2kl) / (10. * k6) + (3 * ipow2(K2) * cosh_2kl) / (50. * k6) + (7 * cos_kl * cosh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (80. * k6) - (k * l * sin_kl) / 8. + (9 * ipow2(K2) * l * sin_kl) / (40. * k5) - (sin_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * sin_kl * sinh_2kl) / (40. * k6);
      U[1][5][0][0] = (K2 * l * cos_kl) / 8. + (K2 * l * cos_2kl) / 8. - (K2 * sin_kl) / (24. * k) + (7 * K2 * sin_2kl) / (48. * k);
      U[1][5][1][0] = (3 * K2) / (8. * k2) - (K2 * cos_kl) / (3. * k2) - (K2 * cos_2kl) / (24. * k2) - (K2 * l * sin_kl) / (4. * k) + (K2 * l * sin_2kl) / (4. * k);
      U[1][5][1][1] = (K2 * l * cos_kl) / (4. * k2) - (K2 * l * cos_2kl) / (8. * k2) - (K2 * sin_kl) / (12. * k3) - (K2 * sin_2kl) / (48. * k3);
      U[1][5][2][2] = (-9 * K2 * l * cos_kl) / 40. - (3 * K2 * l * cosh_2kl) / 40. - (21 * K2 * sin_kl) / (200. * k) - (19 * K2 * sinh_2kl) / (400. * k);
      U[1][5][3][2] = (-3 * K2) / (8. * k2) + (8 * K2 * cos_kl) / (25. * k2) + (11 * K2 * cosh_2kl) / (200. * k2) - (3 * K2 * l * sin_kl) / (20. * k) - (3 * K2 * l * sinh_2kl) / (20. * k);
      U[1][5][3][3] = (3 * K2 * l * cos_kl) / (20. * k2) - (3 * K2 * l * cosh_2kl) / (40. * k2) - (13 * K2 * sin_kl) / (100. * k3) + (11 * K2 * sinh_2kl) / (400. * k3);
      U[1][5][5][0] = (-19 * k2 * l * cos_kl) / 32. - (13 * k * sin_kl) / 32. + (3 * k3 * l2 * sin_kl) / 32.;
      U[1][5][5][1] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * l * sin_kl) / 32.;
      U[2][2][0][0] = (k2 * cosh_kl * ipow2(sin((k * l) / 2.))) / 8. + (13 * ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.))) / (50. * k4) + (k2 * cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / 8. + (ipow2(K2) * cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / (10. * k4) + (k3 * l * sinh_kl) / 8. - (9 * ipow2(K2) * l * sinh_kl) / (40. * k3) + (4 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4) - (3 * k2 * sin_2kl * sinh_kl) / 32. - (ipow2(K2) * sin_2kl * sinh_kl) / (80. * k4);
      U[2][2][1][0] = (-3 * ipow2(K2) * l * cosh_kl) / (20. * k4) + (23 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k5) - (k * cosh_kl * sin_2kl) / 16. - (ipow2(K2) * cosh_kl * sin_2kl) / (20. * k5) - (k * sinh_kl) / 16. - (51 * ipow2(K2) * sinh_kl) / (200. * k5) + (ipow2(K2) * cos_kl * sinh_kl) / (50. * k5) + (3 * k * cos_2kl * sinh_kl) / 16. + (ipow2(K2) * cos_2kl * sinh_kl) / (40. * k5);
      U[2][2][1][1] = -(cosh_kl * ipow2(sin((k * l) / 2.))) / 8. + (23 * ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.))) / (50. * k6) - (cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / 8. - (ipow2(K2) * cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / (10. * k6) + (k * l * sinh_kl) / 8. - (9 * ipow2(K2) * l * sinh_kl) / (40. * k5) + (11 * ipow2(K2) * sin_kl * sinh_kl) / (100. * k6) + (3 * sin_2kl * sinh_kl) / 32. + (ipow2(K2) * sin_2kl * sinh_kl) / (80. * k6);
      U[2][2][2][2] = (-3 * k2 * cosh_kl) / 64. - (101 * ipow2(K2) * cosh_kl) / (1600. * k4) + (3 * ipow2(K2) * cos_kl * cosh_kl) / (50. * k4) + (3 * k2 * cosh_3kl) / 64. + (ipow2(K2) * cosh_3kl) / (320. * k4) - (3 * k3 * l * sinh_kl) / 16. + (11 * ipow2(K2) * l * sinh_kl) / (80. * k3) - (3 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4);
      U[2][3][0][0] = (k2 * l * cosh_kl) / 8. - (9 * ipow2(K2) * l * cosh_kl) / (40. * k4) + (ipow2(K2) * cosh_kl * sin_kl) / (25. * k5) - (3 * k * cosh_kl * sin_2kl) / 32. - (ipow2(K2) * cosh_kl * sin_2kl) / (80. * k5) + (3 * k * sinh_kl) / 32. + (51 * ipow2(K2) * sinh_kl) / (200. * k5) - (ipow2(K2) * cos_kl * sinh_kl) / (50. * k5) - (k * cos_2kl * sinh_kl) / 32. - (ipow2(K2) * cos_2kl * sinh_kl) / (40. * k5);
      U[2][3][1][0] = (-3 * cosh_kl * ipow2(sin((k * l) / 2.))) / 4. - (13 * ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.))) / (50. * k6) - (3 * cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / 4. - (ipow2(K2) * cos_kl * cosh_kl * ipow2(sin((k * l) / 2.))) / (10. * k6) - (3 * ipow2(K2) * l * sinh_kl) / (20. * k5) + (17 * ipow2(K2) * sin_kl * sinh_kl) / (50. * k6) - (sin_2kl * sinh_kl) / 16. - (ipow2(K2) * sin_2kl * sinh_kl) / (20. * k6);
      U[2][3][1][1] = (l * cosh_kl) / 8. - (9 * ipow2(K2) * l * cosh_kl) / (40. * k6) + (7 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k7) + (3 * cosh_kl * sin_2kl) / (32. * k) + (ipow2(K2) * cosh_kl * sin_2kl) / (80. * k7) - (11 * sinh_kl) / (32. * k) + (51 * ipow2(K2) * sinh_kl) / (200. * k7) - (11 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k7) + (cos_2kl * sinh_kl) / (32. * k) + (ipow2(K2) * cos_2kl * sinh_kl) / (40. * k7);
      U[2][3][2][2] = (-3 * k2 * l * cosh_kl) / 16. + (3 * ipow2(K2) * l * cosh_kl) / (16. * k4) - (3 * ipow2(K2) * cosh_kl * sin_kl) / (20. * k5) - (15 * k * sinh_kl) / 64. - (69 * ipow2(K2) * sinh_kl) / (320. * k5) + (3 * ipow2(K2) * cos_kl * sinh_kl) / (20. * k5) + (9 * k * sinh_3kl) / 64. + (3 * ipow2(K2) * sinh_3kl) / (320. * k5);
      U[2][3][3][2] = (3 * k * l * sinh_kl) / 16. - (3 * ipow2(K2) * l * sinh_kl) / (16. * k5) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (20. * k6) + (9 * sinh_kl * sinh_2kl) / 32. + (3 * ipow2(K2) * sinh_kl * sinh_2kl) / (160. * k6);
      U[2][3][3][3] = (3 * l * cosh_kl) / 16. - (11 * ipow2(K2) * l * cosh_kl) / (80. * k6) + (2 * ipow2(K2) * cosh_kl * sin_kl) / (25. * k7) - (9 * sinh_kl) / (32. * k) + (73 * ipow2(K2) * sinh_kl) / (800. * k7) - (ipow2(K2) * cos_kl * sinh_kl) / (25. * k7) + (3 * cosh_2kl * sinh_kl) / (32. * k) + (ipow2(K2) * cosh_2kl * sinh_kl) / (160. * k7);
      U[2][5][2][0] = (K2 * cosh_kl) / (25. * k2) - (K2 * cos_kl * cosh_kl) / (25. * k2) - (3 * K2 * l * cosh_kl * sin_kl) / (10. * k) - (3 * K2 * l * sinh_kl) / (20. * k) + (3 * K2 * l * cos_kl * sinh_kl) / (20. * k) - (11 * K2 * sin_kl * sinh_kl) / (50. * k2);
      U[2][5][2][1] = (-9 * K2 * l * cosh_kl) / (20. * k2) + (3 * K2 * l * cos_kl * cosh_kl) / (10. * k2) - (K2 * cosh_kl * sin_kl) / (25. * k3) - (3 * K2 * sinh_kl) / (100. * k3) + (11 * K2 * cos_kl * sinh_kl) / (50. * k3) + (3 * K2 * l * sin_kl * sinh_kl) / (20. * k2);
      U[2][5][3][0] = (3 * K2 * l * cosh_kl) / (20. * k2) + (3 * K2 * l * cos_kl * cosh_kl) / (20. * k2) + (2 * K2 * cosh_kl * sin_kl) / (25. * k3) - (19 * K2 * sinh_kl) / (100. * k3) - (19 * K2 * cos_kl * sinh_kl) / (100. * k3) - (3 * K2 * l * sin_kl * sinh_kl) / (10. * k2);
      U[2][5][3][1] = (2 * K2 * cosh_kl) / (25. * k4) - (2 * K2 * cos_kl * cosh_kl) / (25. * k4) + (3 * K2 * l * cosh_kl * sin_kl) / (20. * k3) - (3 * K2 * l * sinh_kl) / (10. * k3) + (3 * K2 * l * cos_kl * sinh_kl) / (10. * k3) - (19 * K2 * sin_kl * sinh_kl) / (100. * k4);
      U[2][5][5][2] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * l * sinh_kl) / 32.;
      U[2][5][5][3] = (7 * l * cosh_kl) / 32. - (7 * sinh_kl) / (32. * k) + (3 * k * l2 * sinh_kl) / 32.;
      U[3][2][0][0] = (k4 * l * cosh_kl) / 8. - (9 * ipow2(K2) * l * cosh_kl) / (40. * k2) + (k3 * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / 8. + (13 * ipow2(K2) * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / (50. * k3) + (k3 * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / 8. + (ipow2(K2) * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / (10. * k3) + (4 * ipow2(K2) * cosh_kl * sin_kl) / (25. * k3) - (k3 * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / 8. - (ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / (10. * k3) - (3 * k3 * cosh_kl * sin_2kl) / 32. - (ipow2(K2) * cosh_kl * sin_2kl) / (80. * k3) + (k3 * sinh_kl) / 8. - (9 * ipow2(K2) * sinh_kl) / (40. * k3) + (4 * ipow2(K2) * cos_kl * sinh_kl) / (25. * k3) - (3 * k3 * cos_2kl * sinh_kl) / 16. - (ipow2(K2) * cos_2kl * sinh_kl) / (40. * k3) + (k3 * ipow2(sin((k * l) / 2.)) * sinh_kl) / 8. + (13 * ipow2(K2) * ipow2(sin((k * l) / 2.)) * sinh_kl) / (50. * k3) + (k3 * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / 8. + (ipow2(K2) * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / (10. * k3);
      U[3][2][1][0] = -(k2 * cosh_kl) / 16. - (81 * ipow2(K2) * cosh_kl) / (200. * k4) + (12 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k4) + (k2 * cos_2kl * cosh_kl) / 16. - (3 * ipow2(K2) * cos_2kl * cosh_kl) / (40. * k4) - (3 * ipow2(K2) * l * sinh_kl) / (20. * k3) + (11 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4) - (7 * k2 * sin_2kl * sinh_kl) / 16. - (ipow2(K2) * sin_2kl * sinh_kl) / (10. * k4);
      U[3][2][1][1] = (k2 * l * cosh_kl) / 8. - (9 * ipow2(K2) * l * cosh_kl) / (40. * k4) - (k * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / 8. + (23 * ipow2(K2) * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / (50. * k5) - (k * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / 8. - (ipow2(K2) * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / (10. * k5) + (11 * ipow2(K2) * cosh_kl * sin_kl) / (100. * k5) + (k * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / 8. + (ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / (10. * k5) + (3 * k * cosh_kl * sin_2kl) / 32. + (ipow2(K2) * cosh_kl * sin_2kl) / (80. * k5) + (k * sinh_kl) / 8. - (9 * ipow2(K2) * sinh_kl) / (40. * k5) + (11 * ipow2(K2) * cos_kl * sinh_kl) / (100. * k5) + (3 * k * cos_2kl * sinh_kl) / 16. + (ipow2(K2) * cos_2kl * sinh_kl) / (40. * k5) - (k * ipow2(sin((k * l) / 2.)) * sinh_kl) / 8. + (23 * ipow2(K2) * ipow2(sin((k * l) / 2.)) * sinh_kl) / (50. * k5) - (k * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / 8. - (ipow2(K2) * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / (10. * k5);
      U[3][2][2][2] = (-3 * k4 * l * cosh_kl) / 16. + (11 * ipow2(K2) * l * cosh_kl) / (80. * k2) - (9 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k3) - (15 * k3 * sinh_kl) / 64. + (119 * ipow2(K2) * sinh_kl) / (1600. * k3) - (3 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k3) + (9 * k3 * sinh_3kl) / 64. + (3 * ipow2(K2) * sinh_3kl) / (320. * k3);
      U[3][3][0][0] = (7 * k2 * cosh_kl) / 32. + (3 * ipow2(K2) * cosh_kl) / (100. * k4) + (ipow2(K2) * cos_kl * cosh_kl) / (50. * k4) - (7 * k2 * cos_2kl * cosh_kl) / 32. - (ipow2(K2) * cos_2kl * cosh_kl) / (20. * k4) + (k3 * l * sinh_kl) / 8. - (9 * ipow2(K2) * l * sinh_kl) / (40. * k3) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (50. * k4) - (k2 * sin_2kl * sinh_kl) / 32. + (3 * ipow2(K2) * sin_2kl * sinh_kl) / (80. * k4);
      U[3][3][1][0] = (-3 * ipow2(K2) * l * cosh_kl) / (20. * k4) - (3 * k * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / 4. - (13 * ipow2(K2) * cos((k * l) / 2.) * cosh_kl * sin((k * l) / 2.)) / (50. * k5) - (3 * k * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / 4. - (ipow2(K2) * cos((k * l) / 2.) * cos_kl * cosh_kl * sin((k * l) / 2.)) / (10. * k5) + (17 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k5) + (3 * k * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / 4. + (ipow2(K2) * cosh_kl * ipow2(sin((k * l) / 2.)) * sin_kl) / (10. * k5) - (k * cosh_kl * sin_2kl) / 16. - (ipow2(K2) * cosh_kl * sin_2kl) / (20. * k5) - (3 * ipow2(K2) * sinh_kl) / (20. * k5) + (17 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k5) - (k * cos_2kl * sinh_kl) / 8. - (ipow2(K2) * cos_2kl * sinh_kl) / (10. * k5) - (3 * k * ipow2(sin((k * l) / 2.)) * sinh_kl) / 4. - (13 * ipow2(K2) * ipow2(sin((k * l) / 2.)) * sinh_kl) / (50. * k5) - (3 * k * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / 4. - (ipow2(K2) * cos_kl * ipow2(sin((k * l) / 2.)) * sinh_kl) / (10. * k5);
      U[3][3][1][1] = (-7 * cosh_kl) / 32. + (3 * ipow2(K2) * cosh_kl) / (100. * k6) - (2 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k6) + (7 * cos_2kl * cosh_kl) / 32. + (ipow2(K2) * cos_2kl * cosh_kl) / (20. * k6) + (k * l * sinh_kl) / 8. - (9 * ipow2(K2) * l * sinh_kl) / (40. * k5) + (9 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k6) + (sin_2kl * sinh_kl) / 32. - (3 * ipow2(K2) * sin_2kl * sinh_kl) / (80. * k6);
      U[3][3][2][2] = (-27 * k2 * cosh_kl) / 64. - (9 * ipow2(K2) * cosh_kl) / (320. * k4) + (27 * k2 * cosh_3kl) / 64. + (9 * ipow2(K2) * cosh_3kl) / (320. * k4) - (3 * k3 * l * sinh_kl) / 16. + (3 * ipow2(K2) * l * sinh_kl) / (16. * k3) - (3 * ipow2(K2) * sin_kl * sinh_kl) / (10. * k4);
      U[3][3][3][2] = (3 * k2 * l * cosh_kl) / 16. - (3 * ipow2(K2) * l * cosh_kl) / (16. * k4) + (3 * ipow2(K2) * cosh_kl * sin_kl) / (20. * k5) + (3 * k * sinh_kl) / 16. - (3 * ipow2(K2) * sinh_kl) / (16. * k5) + (3 * ipow2(K2) * cos_kl * sinh_kl) / (20. * k5) + (9 * k * cosh_2kl * sinh_kl) / 16. + (3 * ipow2(K2) * cosh_2kl * sinh_kl) / (80. * k5) + (9 * k * cosh_kl * sinh_2kl) / 32. + (3 * ipow2(K2) * cosh_kl * sinh_2kl) / (160. * k5);
      U[3][3][3][3] = (-3 * cosh_kl) / 32. - (37 * ipow2(K2) * cosh_kl) / (800. * k6) + (ipow2(K2) * cos_kl * cosh_kl) / (25. * k6) + (3 * cosh_kl * cosh_2kl) / 32. + (ipow2(K2) * cosh_kl * cosh_2kl) / (160. * k6) + (3 * k * l * sinh_kl) / 16. - (11 * ipow2(K2) * l * sinh_kl) / (80. * k5) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k6) + (3 * sinh_kl * sinh_2kl) / 16. + (ipow2(K2) * sinh_kl * sinh_2kl) / (80. * k6);
      U[3][5][2][0] = (-3 * K2 * l * cosh_kl) / 20. - (3 * K2 * l * cos_kl * cosh_kl) / 20. - (12 * K2 * cosh_kl * sin_kl) / (25. * k) - (11 * K2 * sinh_kl) / (100. * k) - (11 * K2 * cos_kl * sinh_kl) / (100. * k) - (9 * K2 * l * sin_kl * sinh_kl) / 20.;
      U[3][5][2][1] = (-12 * K2 * cosh_kl) / (25. * k2) + (12 * K2 * cos_kl * cosh_kl) / (25. * k2) - (3 * K2 * l * cosh_kl * sin_kl) / (20. * k) - (9 * K2 * l * sinh_kl) / (20. * k) + (9 * K2 * l * cos_kl * sinh_kl) / (20. * k) - (11 * K2 * sin_kl * sinh_kl) / (100. * k2);
      U[3][5][3][0] = -(K2 * cosh_kl) / (25. * k2) + (K2 * cos_kl * cosh_kl) / (25. * k2) - (9 * K2 * l * cosh_kl * sin_kl) / (20. * k) + (3 * K2 * l * sinh_kl) / (20. * k) - (3 * K2 * l * cos_kl * sinh_kl) / (20. * k) - (3 * K2 * sin_kl * sinh_kl) / (100. * k2);
      U[3][5][3][1] = (-3 * K2 * l * cosh_kl) / (10. * k2) + (9 * K2 * l * cos_kl * cosh_kl) / (20. * k2) + (K2 * cosh_kl * sin_kl) / (25. * k3) - (11 * K2 * sinh_kl) / (50. * k3) + (3 * K2 * cos_kl * sinh_kl) / (100. * k3) - (3 * K2 * l * sin_kl * sinh_kl) / (20. * k2);
      U[3][5][5][2] = (19 * k2 * l * cosh_kl) / 32. + (13 * k * sinh_kl) / 32. + (3 * k3 * l2 * sinh_kl) / 32.;
      U[3][5][5][3] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * l * sinh_kl) / 32.;
    }
  } else {
    double k2, k3, k4, k5, k6, k7, l, l2;
    double cos_kl, cos_2kl, cos_3kl, sin_kl, sin_2kl, sin_3kl, cosh_kl, cosh_2kl, cosh_3kl, sinh_kl, sinh_2kl, sinh_3kl;
    /* double k8 */

    k = sqrt(-K1);
    kl = k * length;
    l = length;

    C[4] = length;

    k2 = ipow2(k);
    k3 = ipow3(k);
    k4 = ipow4(k);
    k5 = ipow5(k);
    k6 = ipow6(k);
    k7 = ipow7(k);
    /* k8 = ipow8(k) ; */
    l2 = ipow2(l);
    cos_kl = cos(kl);
    cos_2kl = cos(2 * kl);
    cos_3kl = cos(3 * kl);
    sin_kl = sin(kl);
    sin_2kl = sin(2 * kl);
    sin_3kl = sin(3 * kl);
    cosh_kl = cosh(kl);
    cosh_2kl = cosh(2 * kl);
    cosh_3kl = cosh(3 * kl);
    sinh_kl = sinh(kl);
    sinh_2kl = sinh(2 * kl);
    sinh_3kl = sinh(3 * kl);

    R[0][0] = cosh_kl;
    R[0][1] = sinh_kl / k;
    R[1][0] = k * sinh_kl;
    R[1][1] = cosh_kl;
    R[2][2] = cos_kl;
    R[2][3] = sin_kl / k;
    R[3][2] = -(k * sin_kl);
    R[3][3] = cos_kl;
    R[4][4] = 1;
    R[5][5] = 1;

    if (M->order > 1) {
      T = M->T;
      T[0][0][0] = (-2 * K2 * ipow2(sinh((k * l) / 2.))) / (3. * k2) - (K2 * cosh_kl * ipow2(sinh((k * l) / 2.))) / (3. * k2);
      T[0][1][0] = (K2 * sinh_kl) / (3. * k3) - (K2 * sinh_2kl) / (6. * k3);
      T[0][1][1] = (-2 * K2 * ipow4(sinh((k * l) / 2.))) / (3. * k4);
      T[0][2][2] = -K2 / (4. * k2) - (K2 * cos_2kl) / (20. * k2) + (3 * K2 * cosh_kl) / (10. * k2);
      T[0][3][2] = -(K2 * sin_2kl) / (10. * k3) + (K2 * sinh_kl) / (5. * k3);
      T[0][3][3] = -K2 / (4. * k4) + (K2 * cos_2kl) / (20. * k4) + (K2 * cosh_kl) / (5. * k4);
      T[0][5][0] = -(k * l * sinh_kl) / 2.;
      T[0][5][1] = -(l * cosh_kl) / 2. + sinh_kl / (2. * k);
      T[1][0][0] = (-2 * K2 * cosh((k * l) / 2.) * sinh((k * l) / 2.)) / (3. * k) - (K2 * cosh((k * l) / 2.) * cosh_kl * sinh((k * l) / 2.)) / (3. * k) - (K2 * ipow2(sinh((k * l) / 2.)) * sinh_kl) / (3. * k);
      T[1][1][0] = (K2 * cosh_kl) / (3. * k2) - (K2 * cosh_2kl) / (3. * k2);
      T[1][1][1] = (-4 * K2 * cosh((k * l) / 2.) * ipow3(sinh((k * l) / 2.))) / (3. * k3);
      T[1][2][2] = (K2 * sin_2kl) / (10. * k) + (3 * K2 * sinh_kl) / (10. * k);
      T[1][3][2] = -(K2 * cos_2kl) / (5. * k2) + (K2 * cosh_kl) / (5. * k2);
      T[1][3][3] = -(K2 * sin_2kl) / (10. * k3) + (K2 * sinh_kl) / (5. * k3);
      T[1][5][0] = -(k2 * l * cosh_kl) / 2. - (k * sinh_kl) / 2.;
      T[1][5][1] = -(k * l * sinh_kl) / 2.;
      T[2][2][0] = -(K2 * cos_kl) / (5. * k2) + (K2 * cos_kl * cosh_kl) / (5. * k2) + (2 * K2 * sin_kl * sinh_kl) / (5. * k2);
      T[2][2][1] = (-3 * K2 * sin_kl) / (5. * k3) + (2 * K2 * cosh_kl * sin_kl) / (5. * k3) + (K2 * cos_kl * sinh_kl) / (5. * k3);
      T[2][3][0] = (K2 * sin_kl) / (5. * k3) + (K2 * cosh_kl * sin_kl) / (5. * k3) - (2 * K2 * cos_kl * sinh_kl) / (5. * k3);
      T[2][3][1] = (2 * K2 * cos_kl) / (5. * k4) - (2 * K2 * cos_kl * cosh_kl) / (5. * k4) + (K2 * sin_kl * sinh_kl) / (5. * k4);
      T[2][5][2] = (k * l * sin_kl) / 2.;
      T[2][5][3] = -(l * cos_kl) / 2. + sin_kl / (2. * k);
      T[3][2][0] = (K2 * sin_kl) / (5. * k) + (K2 * cosh_kl * sin_kl) / (5. * k) + (3 * K2 * cos_kl * sinh_kl) / (5. * k);
      T[3][2][1] = (-3 * K2 * cos_kl) / (5. * k2) + (3 * K2 * cos_kl * cosh_kl) / (5. * k2) + (K2 * sin_kl * sinh_kl) / (5. * k2);
      T[3][3][0] = (K2 * cos_kl) / (5. * k2) - (K2 * cos_kl * cosh_kl) / (5. * k2) + (3 * K2 * sin_kl * sinh_kl) / (5. * k2);
      T[3][3][1] = (-2 * K2 * sin_kl) / (5. * k3) + (3 * K2 * cosh_kl * sin_kl) / (5. * k3) - (K2 * cos_kl * sinh_kl) / (5. * k3);
      T[3][5][2] = (k2 * l * cos_kl) / 2. + (k * sin_kl) / 2.;
      T[3][5][3] = (k * l * sin_kl) / 2.;
      T[4][0][0] = -(k2 * l) / 4. + (k * sinh_2kl) / 8.;
      T[4][1][0] = ipow2(sinh_kl) / 2.;
      T[4][1][1] = l / 4. + sinh_2kl / (8. * k);
      T[4][2][2] = (k2 * l) / 4. - (k * sin_2kl) / 8.;
      T[4][3][2] = -ipow2(sin_kl) / 2.;
      T[4][3][3] = l / 4. + sin_2kl / (8. * k);
    }

    if (M->order > 2) {
      U = M->Q;
      U[0][0][0][0] = -ipow2(K2) / (12. * k4) - (3 * k2 * cosh_kl) / 64. + (29 * ipow2(K2) * cosh_kl) / (576. * k4) + (ipow2(K2) * cosh_2kl) / (36. * k4) + (3 * k2 * cosh_3kl) / 64. + (ipow2(K2) * cosh_3kl) / (192. * k4) - (3 * k3 * l * sinh_kl) / 16. - (5 * ipow2(K2) * l * sinh_kl) / (48. * k3);
      U[0][1][0][0] = (-3 * k2 * l * cosh_kl) / 16. - (3 * ipow2(K2) * l * cosh_kl) / (16. * k4) - (3 * k * sinh_kl) / 32. + (5 * ipow2(K2) * sinh_kl) / (32. * k5) + (9 * k * cosh_2kl * sinh_kl) / 32. + (ipow2(K2) * cosh_2kl * sinh_kl) / (32. * k5);
      U[0][1][1][0] = (3 * k * l * sinh_kl) / 16. + (3 * ipow2(K2) * l * sinh_kl) / (16. * k5) - (ipow2(K2) * ipow2(sinh_kl)) / (4. * k6) + (9 * sinh_kl * sinh_2kl) / 32. + (ipow2(K2) * sinh_kl * sinh_2kl) / (32. * k6);
      U[0][1][1][1] = (3 * l * cosh_kl) / 16. + (5 * ipow2(K2) * l * cosh_kl) / (48. * k6) - (21 * sinh_kl) / (64. * k) - (5 * ipow2(K2) * sinh_kl) / (576. * k7) - (ipow2(K2) * sinh_2kl) / (18. * k7) + (3 * sinh_3kl) / (64. * k) + (ipow2(K2) * sinh_3kl) / (192. * k7);
      U[0][2][2][0] = (3 * ipow2(K2)) / (10. * k4) + (3 * ipow2(K2) * cos_2kl) / (200. * k4) + (k2 * cosh_kl) / 32. - (81 * ipow2(K2) * cosh_kl) / (400. * k4) - (k2 * cos_2kl * cosh_kl) / 32. - (3 * ipow2(K2) * cos_2kl * cosh_kl) / (80. * k4) - (3 * ipow2(K2) * cosh_2kl) / (40. * k4) + (k3 * l * sinh_kl) / 8. + (9 * ipow2(K2) * l * sinh_kl) / (40. * k3) - (3 * k2 * sin_2kl * sinh_kl) / 32.;
      U[0][2][2][1] = (k2 * l * cosh_kl) / 8. + (9 * ipow2(K2) * l * cosh_kl) / (40. * k4) + (9 * ipow2(K2) * sin_2kl) / (200. * k5) - (3 * k * cosh_kl * sin_2kl) / 32. + (3 * k * sinh_kl) / 32. - (51 * ipow2(K2) * sinh_kl) / (400. * k5) - (k * cos_2kl * sinh_kl) / 32. - (3 * ipow2(K2) * cos_2kl * sinh_kl) / (80. * k5) - (3 * ipow2(K2) * cosh_kl * sinh_kl) / (20. * k5);
      U[0][3][2][0] = (3 * ipow2(K2) * l * cosh_kl) / (20. * k4) + (3 * ipow2(K2) * sin_2kl) / (100. * k5) - (k * cosh_kl * sin_2kl) / 16. - (3 * ipow2(K2) * cosh_kl * sin_2kl) / (40. * k5) - (k * sinh_kl) / 16. + (ipow2(K2) * sinh_kl) / (25. * k5) + (3 * k * cos_2kl * sinh_kl) / 16. - (ipow2(K2) * cosh_kl * sinh_kl) / (10. * k5);
      U[0][3][2][1] = (3 * ipow2(K2)) / (10. * k6) - (9 * ipow2(K2) * cos_2kl) / (100. * k6) - (3 * cosh_kl) / 16. - (4 * ipow2(K2) * cosh_kl) / (25. * k6) + (3 * cos_2kl * cosh_kl) / 16. - (ipow2(K2) * cosh_2kl) / (20. * k6) + (3 * ipow2(K2) * l * sinh_kl) / (20. * k5) - (sin_2kl * sinh_kl) / 16. - (3 * ipow2(K2) * sin_2kl * sinh_kl) / (40. * k6);
      U[0][3][3][0] = (3 * ipow2(K2)) / (40. * k6) + (3 * ipow2(K2) * cos_2kl) / (200. * k6) - cosh_kl / 32. - (31 * ipow2(K2) * cosh_kl) / (400. * k6) + (cos_2kl * cosh_kl) / 32. + (3 * ipow2(K2) * cos_2kl * cosh_kl) / (80. * k6) - (ipow2(K2) * cosh_2kl) / (20. * k6) + (k * l * sinh_kl) / 8. + (9 * ipow2(K2) * l * sinh_kl) / (40. * k5) + (3 * sin_2kl * sinh_kl) / 32.;
      U[0][3][3][1] = (l * cosh_kl) / 8. + (9 * ipow2(K2) * l * cosh_kl) / (40. * k6) - (3 * ipow2(K2) * sin_2kl) / (100. * k7) + (3 * cosh_kl * sin_2kl) / (32. * k) - (11 * sinh_kl) / (32. * k) - (41 * ipow2(K2) * sinh_kl) / (400. * k7) + (cos_2kl * sinh_kl) / (32. * k) + (3 * ipow2(K2) * cos_2kl * sinh_kl) / (80. * k7) - (ipow2(K2) * cosh_kl * sinh_kl) / (10. * k7);
      U[0][5][0][0] = K2 / (8. * k2) - (K2 * cosh_kl) / (6. * k2) + (K2 * cosh_2kl) / (24. * k2) + (K2 * l * sinh_kl) / (8. * k) + (K2 * l * sinh_2kl) / (16. * k);
      U[0][5][1][0] = (-3 * K2 * l) / (8. * k2) - (K2 * l * cosh_kl) / (4. * k2) + (K2 * l * cosh_2kl) / (8. * k2) + (7 * K2 * sinh_kl) / (12. * k3) - (K2 * sinh_2kl) / (24. * k3);
      U[0][5][1][1] = (-5 * K2) / (16. * k4) + (K2 * cosh_kl) / (3. * k4) - (K2 * cosh_2kl) / (48. * k4) - (K2 * l * sinh_kl) / (4. * k3) + (K2 * l * sinh_2kl) / (16. * k3);
      U[0][5][2][2] = -K2 / (8. * k2) + (K2 * cos_2kl) / (200. * k2) + (3 * K2 * cosh_kl) / (25. * k2) - (3 * K2 * l * sin_2kl) / (80. * k) - (9 * K2 * l * sinh_kl) / (40. * k);
      U[0][5][3][2] = (3 * K2 * l) / (8. * k2) + (3 * K2 * l * cos_2kl) / (40. * k2) - (3 * K2 * l * cosh_kl) / (20. * k2) - (13 * K2 * sin_2kl) / (200. * k3) - (17 * K2 * sinh_kl) / (100. * k3);
      U[0][5][3][3] = (-5 * K2) / (16. * k4) + (13 * K2 * cos_2kl) / (400. * k4) + (7 * K2 * cosh_kl) / (25. * k4) + (3 * K2 * l * sin_2kl) / (80. * k3) - (3 * K2 * l * sinh_kl) / (20. * k3);
      U[0][5][5][0] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * l * sinh_kl) / 32.;
      U[0][5][5][1] = (7 * l * cosh_kl) / 32. - (7 * sinh_kl) / (32. * k) + (3 * k * l2 * sinh_kl) / 32.;
      U[1][0][0][0] = (-3 * k4 * l * cosh_kl) / 16. - (5 * ipow2(K2) * l * cosh_kl) / (48. * k2) - (15 * k3 * sinh_kl) / 64. - (31 * ipow2(K2) * sinh_kl) / (576. * k3) + (ipow2(K2) * sinh_2kl) / (18. * k3) + (9 * k3 * sinh_3kl) / 64. + (ipow2(K2) * sinh_3kl) / (64. * k3);
      U[1][1][0][0] = (-9 * k2 * cosh_kl) / 32. - (ipow2(K2) * cosh_kl) / (32. * k4) + (9 * k2 * cosh_kl * cosh_2kl) / 32. + (ipow2(K2) * cosh_kl * cosh_2kl) / (32. * k4) - (3 * k3 * l * sinh_kl) / 16. - (3 * ipow2(K2) * l * sinh_kl) / (16. * k3) + (9 * k2 * sinh_kl * sinh_2kl) / 16. + (ipow2(K2) * sinh_kl * sinh_2kl) / (16. * k4);
      U[1][1][1][0] = (3 * k2 * l * cosh_kl) / 16. + (3 * ipow2(K2) * l * cosh_kl) / (16. * k4) + (3 * k * sinh_kl) / 16. + (3 * ipow2(K2) * sinh_kl) / (16. * k5) - (ipow2(K2) * cosh_kl * sinh_kl) / (2. * k5) + (9 * k * cosh_2kl * sinh_kl) / 16. + (ipow2(K2) * cosh_2kl * sinh_kl) / (16. * k5) + (9 * k * cosh_kl * sinh_2kl) / 32. + (ipow2(K2) * cosh_kl * sinh_2kl) / (32. * k5);
      U[1][1][1][1] = (-9 * cosh_kl) / 64. + (55 * ipow2(K2) * cosh_kl) / (576. * k6) - (ipow2(K2) * cosh_2kl) / (9. * k6) + (9 * cosh_3kl) / 64. + (ipow2(K2) * cosh_3kl) / (64. * k6) + (3 * k * l * sinh_kl) / 16. + (5 * ipow2(K2) * l * sinh_kl) / (48. * k5);
      U[1][2][2][0] = (k4 * l * cosh_kl) / 8. + (9 * ipow2(K2) * l * cosh_kl) / (40. * k2) - (3 * ipow2(K2) * sin_2kl) / (100. * k3) - (k3 * cosh_kl * sin_2kl) / 32. + (3 * ipow2(K2) * cosh_kl * sin_2kl) / (40. * k3) + (5 * k3 * sinh_kl) / 32. + (9 * ipow2(K2) * sinh_kl) / (400. * k3) - (7 * k3 * cos_2kl * sinh_kl) / 32. - (3 * ipow2(K2) * cos_2kl * sinh_kl) / (80. * k3) - (3 * ipow2(K2) * sinh_2kl) / (20. * k3);
      U[1][2][2][1] = (9 * ipow2(K2) * cos_2kl) / (100. * k4) + (7 * k2 * cosh_kl) / 32. + (39 * ipow2(K2) * cosh_kl) / (400. * k4) - (7 * k2 * cos_2kl * cosh_kl) / 32. - (3 * ipow2(K2) * cos_2kl * cosh_kl) / (80. * k4) - (3 * ipow2(K2) * ipow2(cosh_kl)) / (20. * k4) + (k3 * l * sinh_kl) / 8. + (9 * ipow2(K2) * l * sinh_kl) / (40. * k3) - (k2 * sin_2kl * sinh_kl) / 32. + (3 * ipow2(K2) * sin_2kl * sinh_kl) / (40. * k4) - (3 * ipow2(K2) * ipow2(sinh_kl)) / (20. * k4);
      U[1][3][2][0] = (3 * ipow2(K2) * cos_2kl) / (50. * k4) - (k2 * cosh_kl) / 16. + (19 * ipow2(K2) * cosh_kl) / (100. * k4) + (k2 * cos_2kl * cosh_kl) / 16. - (3 * ipow2(K2) * cos_2kl * cosh_kl) / (20. * k4) - (ipow2(K2) * ipow2(cosh_kl)) / (10. * k4) + (3 * ipow2(K2) * l * sinh_kl) / (20. * k3) - (7 * k2 * sin_2kl * sinh_kl) / 16. - (3 * ipow2(K2) * sin_2kl * sinh_kl) / (40. * k4) - (ipow2(K2) * ipow2(sinh_kl)) / (10. * k4);
      U[1][3][2][1] = (3 * ipow2(K2) * l * cosh_kl) / (20. * k4) + (9 * ipow2(K2) * sin_2kl) / (50. * k5) - (7 * k * cosh_kl * sin_2kl) / 16. - (3 * ipow2(K2) * cosh_kl * sin_2kl) / (40. * k5) - (3 * k * sinh_kl) / 16. - (ipow2(K2) * sinh_kl) / (100. * k5) + (k * cos_2kl * sinh_kl) / 16. - (3 * ipow2(K2) * cos_2kl * sinh_kl) / (20. * k5) - (ipow2(K2) * sinh_2kl) / (10. * k5);
      U[1][3][3][0] = (k2 * l * cosh_kl) / 8. + (9 * ipow2(K2) * l * cosh_kl) / (40. * k4) - (3 * ipow2(K2) * sin_2kl) / (100. * k5) + (k * cosh_kl * sin_2kl) / 32. - (3 * ipow2(K2) * cosh_kl * sin_2kl) / (40. * k5) + (3 * k * sinh_kl) / 32. + (59 * ipow2(K2) * sinh_kl) / (400. * k5) + (7 * k * cos_2kl * sinh_kl) / 32. + (3 * ipow2(K2) * cos_2kl * sinh_kl) / (80. * k5) - (ipow2(K2) * sinh_2kl) / (10. * k5);
      U[1][3][3][1] = (-3 * ipow2(K2) * cos_2kl) / (50. * k6) - (7 * cosh_kl) / 32. + (49 * ipow2(K2) * cosh_kl) / (400. * k6) + (7 * cos_2kl * cosh_kl) / 32. + (3 * ipow2(K2) * cos_2kl * cosh_kl) / (80. * k6) - (ipow2(K2) * ipow2(cosh_kl)) / (10. * k6) + (k * l * sinh_kl) / 8. + (9 * ipow2(K2) * l * sinh_kl) / (40. * k5) + (sin_2kl * sinh_kl) / 32. - (3 * ipow2(K2) * sin_2kl * sinh_kl) / (40. * k6) - (ipow2(K2) * ipow2(sinh_kl)) / (10. * k6);
      U[1][5][0][0] = (K2 * l * cosh_kl) / 8. + (K2 * l * cosh_2kl) / 8. - (K2 * sinh_kl) / (24. * k) + (7 * K2 * sinh_2kl) / (48. * k);
      U[1][5][1][0] = (-3 * K2) / (8. * k2) + (K2 * cosh_kl) / (3. * k2) + (K2 * cosh_2kl) / (24. * k2) - (K2 * l * sinh_kl) / (4. * k) + (K2 * l * sinh_2kl) / (4. * k);
      U[1][5][1][1] = -(K2 * l * cosh_kl) / (4. * k2) + (K2 * l * cosh_2kl) / (8. * k2) + (K2 * sinh_kl) / (12. * k3) + (K2 * sinh_2kl) / (48. * k3);
      U[1][5][2][2] = (-3 * K2 * l * cos_2kl) / 40. - (9 * K2 * l * cosh_kl) / 40. - (19 * K2 * sin_2kl) / (400. * k) - (21 * K2 * sinh_kl) / (200. * k);
      U[1][5][3][2] = (3 * K2) / (8. * k2) - (11 * K2 * cos_2kl) / (200. * k2) - (8 * K2 * cosh_kl) / (25. * k2) - (3 * K2 * l * sin_2kl) / (20. * k) - (3 * K2 * l * sinh_kl) / (20. * k);
      U[1][5][3][3] = (3 * K2 * l * cos_2kl) / (40. * k2) - (3 * K2 * l * cosh_kl) / (20. * k2) - (11 * K2 * sin_2kl) / (400. * k3) + (13 * K2 * sinh_kl) / (100. * k3);
      U[1][5][5][0] = (19 * k2 * l * cosh_kl) / 32. + (13 * k * sinh_kl) / 32. + (3 * k3 * l2 * sinh_kl) / 32.;
      U[1][5][5][1] = (3 * k2 * l2 * cosh_kl) / 32. + (13 * k * l * sinh_kl) / 32.;
      U[2][2][0][0] = -(k2 * cos_kl) / 32. + (21 * ipow2(K2) * cos_kl) / (200. * k4) - (2 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k4) + (k2 * cos_kl * cosh_2kl) / 32. - (ipow2(K2) * cos_kl * cosh_2kl) / (40. * k4) + (k3 * l * sin_kl) / 8. + (9 * ipow2(K2) * l * sin_kl) / (40. * k3) - (4 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4) - (3 * k2 * sin_kl * sinh_2kl) / 32. + (ipow2(K2) * sin_kl * sinh_2kl) / (80. * k4);
      U[2][2][1][0] = (-3 * ipow2(K2) * l * cos_kl) / (20. * k4) + (k * sin_kl) / 16. - (51 * ipow2(K2) * sin_kl) / (200. * k5) + (ipow2(K2) * cosh_kl * sin_kl) / (50. * k5) - (3 * k * cosh_2kl * sin_kl) / 16. + (ipow2(K2) * cosh_2kl * sin_kl) / (40. * k5) + (23 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k5) + (k * cos_kl * sinh_2kl) / 16. - (ipow2(K2) * cos_kl * sinh_2kl) / (20. * k5);
      U[2][2][1][1] = -cos_kl / 32. - (51 * ipow2(K2) * cos_kl) / (200. * k6) + (7 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k6) + (cos_kl * cosh_2kl) / 32. - (ipow2(K2) * cos_kl * cosh_2kl) / (40. * k6) - (k * l * sin_kl) / 8. - (9 * ipow2(K2) * l * sin_kl) / (40. * k5) + (11 * ipow2(K2) * sin_kl * sinh_kl) / (100. * k6) - (3 * sin_kl * sinh_2kl) / 32. + (ipow2(K2) * sin_kl * sinh_2kl) / (80. * k6);
      U[2][2][2][2] = (3 * k2 * cos_kl) / 64. - (101 * ipow2(K2) * cos_kl) / (1600. * k4) - (3 * k2 * cos_3kl) / 64. + (ipow2(K2) * cos_3kl) / (320. * k4) + (3 * ipow2(K2) * cos_kl * cosh_kl) / (50. * k4) - (3 * k3 * l * sin_kl) / 16. - (11 * ipow2(K2) * l * sin_kl) / (80. * k3) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4);
      U[2][3][0][0] = -(k2 * l * cos_kl) / 8. - (9 * ipow2(K2) * l * cos_kl) / (40. * k4) - (3 * k * sin_kl) / 32. + (51 * ipow2(K2) * sin_kl) / (200. * k5) - (ipow2(K2) * cosh_kl * sin_kl) / (50. * k5) + (k * cosh_2kl * sin_kl) / 32. - (ipow2(K2) * cosh_2kl * sin_kl) / (40. * k5) + (ipow2(K2) * cos_kl * sinh_kl) / (25. * k5) + (3 * k * cos_kl * sinh_2kl) / 32. - (ipow2(K2) * cos_kl * sinh_2kl) / (80. * k5);
      U[2][3][1][0] = (-3 * ipow2(K2) * l * sin_kl) / (20. * k5) + (3 * cos_kl * ipow2(sinh((k * l) / 2.))) / 4. - (13 * ipow2(K2) * cos_kl * ipow2(sinh((k * l) / 2.))) / (50. * k6) + (3 * cos_kl * cosh_kl * ipow2(sinh((k * l) / 2.))) / 4. - (ipow2(K2) * cos_kl * cosh_kl * ipow2(sinh((k * l) / 2.))) / (10. * k6) + (17 * ipow2(K2) * sin_kl * sinh_kl) / (50. * k6) + (sin_kl * sinh_2kl) / 16. - (ipow2(K2) * sin_kl * sinh_2kl) / (20. * k6);
      U[2][3][1][1] = (l * cos_kl) / 8. + (9 * ipow2(K2) * l * cos_kl) / (40. * k6) - (11 * sin_kl) / (32. * k) - (51 * ipow2(K2) * sin_kl) / (200. * k7) + (11 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k7) + (cosh_2kl * sin_kl) / (32. * k) - (ipow2(K2) * cosh_2kl * sin_kl) / (40. * k7) - (7 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k7) + (3 * cos_kl * sinh_2kl) / (32. * k) - (ipow2(K2) * cos_kl * sinh_2kl) / (80. * k7);
      U[2][3][2][2] = (3 * k2 * l * cos_kl) / 16. + (3 * ipow2(K2) * l * cos_kl) / (16. * k4) + (15 * k * sin_kl) / 64. - (69 * ipow2(K2) * sin_kl) / (320. * k5) + (3 * ipow2(K2) * cosh_kl * sin_kl) / (20. * k5) - (9 * k * sin_3kl) / 64. + (3 * ipow2(K2) * sin_3kl) / (320. * k5) - (3 * ipow2(K2) * cos_kl * sinh_kl) / (20. * k5);
      U[2][3][3][2] = (-3 * k * l * sin_kl) / 16. - (3 * ipow2(K2) * l * sin_kl) / (16. * k5) - (9 * sin_kl * sin_2kl) / 32. + (3 * ipow2(K2) * sin_kl * sin_2kl) / (160. * k6) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (20. * k6);
      U[2][3][3][3] = (3 * l * cos_kl) / 16. + (11 * ipow2(K2) * l * cos_kl) / (80. * k6) - (21 * sin_kl) / (64. * k) - (141 * ipow2(K2) * sin_kl) / (1600. * k7) + (ipow2(K2) * cosh_kl * sin_kl) / (25. * k7) + (3 * sin_3kl) / (64. * k) - (ipow2(K2) * sin_3kl) / (320. * k7) - (2 * ipow2(K2) * cos_kl * sinh_kl) / (25. * k7);
      U[2][5][2][0] = -(K2 * cos_kl) / (25. * k2) + (K2 * cos_kl * cosh_kl) / (25. * k2) - (3 * K2 * l * sin_kl) / (20. * k) + (3 * K2 * l * cosh_kl * sin_kl) / (20. * k) - (3 * K2 * l * cos_kl * sinh_kl) / (10. * k) - (11 * K2 * sin_kl * sinh_kl) / (50. * k2);
      U[2][5][2][1] = (9 * K2 * l * cos_kl) / (20. * k2) - (3 * K2 * l * cos_kl * cosh_kl) / (10. * k2) + (3 * K2 * sin_kl) / (100. * k3) - (11 * K2 * cosh_kl * sin_kl) / (50. * k3) + (K2 * cos_kl * sinh_kl) / (25. * k3) + (3 * K2 * l * sin_kl * sinh_kl) / (20. * k2);
      U[2][5][3][0] = (-3 * K2 * l * cos_kl) / (20. * k2) - (3 * K2 * l * cos_kl * cosh_kl) / (20. * k2) + (19 * K2 * sin_kl) / (100. * k3) + (19 * K2 * cosh_kl * sin_kl) / (100. * k3) - (2 * K2 * cos_kl * sinh_kl) / (25. * k3) - (3 * K2 * l * sin_kl * sinh_kl) / (10. * k2);
      U[2][5][3][1] = (2 * K2 * cos_kl) / (25. * k4) - (2 * K2 * cos_kl * cosh_kl) / (25. * k4) + (3 * K2 * l * sin_kl) / (10. * k3) - (3 * K2 * l * cosh_kl * sin_kl) / (10. * k3) - (3 * K2 * l * cos_kl * sinh_kl) / (20. * k3) + (19 * K2 * sin_kl * sinh_kl) / (100. * k4);
      U[2][5][5][2] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * l * sin_kl) / 32.;
      U[2][5][5][3] = (7 * l * cos_kl) / 32. - (7 * sin_kl) / (32. * k) - (3 * k * l2 * sin_kl) / 32.;
      U[3][2][0][0] = (k4 * l * cos_kl) / 8. + (9 * ipow2(K2) * l * cos_kl) / (40. * k2) + (5 * k3 * sin_kl) / 32. + (3 * ipow2(K2) * sin_kl) / (25. * k3) - (2 * ipow2(K2) * cosh_kl * sin_kl) / (25. * k3) - (7 * k3 * cosh_2kl * sin_kl) / 32. + (ipow2(K2) * cosh_2kl * sin_kl) / (20. * k3) - (6 * ipow2(K2) * cos_kl * sinh_kl) / (25. * k3) - (k3 * cos_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (80. * k3);
      U[3][2][1][0] = (k2 * cos_kl) / 16. - (81 * ipow2(K2) * cos_kl) / (200. * k4) + (12 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k4) - (k2 * cos_kl * cosh_2kl) / 16. - (3 * ipow2(K2) * cos_kl * cosh_2kl) / (40. * k4) + (3 * ipow2(K2) * l * sin_kl) / (20. * k3) - (11 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k4) - (7 * k2 * sin_kl * sinh_2kl) / 16. + (ipow2(K2) * sin_kl * sinh_2kl) / (10. * k4);
      U[3][2][1][1] = -(k2 * l * cos_kl) / 8. - (9 * ipow2(K2) * l * cos_kl) / (40. * k4) - (3 * k * sin_kl) / 32. + (3 * ipow2(K2) * sin_kl) / (100. * k5) - (17 * ipow2(K2) * cosh_kl * sin_kl) / (100. * k5) - (7 * k * cosh_2kl * sin_kl) / 32. + (ipow2(K2) * cosh_2kl * sin_kl) / (20. * k5) + (39 * ipow2(K2) * cos_kl * sinh_kl) / (100. * k5) - (k * cos_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * cos_kl * sinh_2kl) / (80. * k5);
      U[3][2][2][2] = (-3 * k4 * l * cos_kl) / 16. - (11 * ipow2(K2) * l * cos_kl) / (80. * k2) - (15 * k3 * sin_kl) / 64. - (119 * ipow2(K2) * sin_kl) / (1600. * k3) + (3 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k3) + (9 * k3 * sin_3kl) / 64. - (3 * ipow2(K2) * sin_3kl) / (320. * k3) + (9 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k3);
      U[3][3][0][0] = (-7 * k2 * cos_kl) / 32. + (3 * ipow2(K2) * cos_kl) / (100. * k4) + (ipow2(K2) * cos_kl * cosh_kl) / (50. * k4) + (7 * k2 * cos_kl * cosh_2kl) / 32. - (ipow2(K2) * cos_kl * cosh_2kl) / (20. * k4) + (k3 * l * sin_kl) / 8. + (9 * ipow2(K2) * l * sin_kl) / (40. * k3) - (3 * ipow2(K2) * sin_kl * sinh_kl) / (50. * k4) - (k2 * sin_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * sin_kl * sinh_2kl) / (80. * k4);
      U[3][3][1][0] = (-3 * ipow2(K2) * l * cos_kl) / (20. * k4) - (3 * ipow2(K2) * sin_kl) / (20. * k5) + (17 * ipow2(K2) * cosh_kl * sin_kl) / (50. * k5) + (k * cosh_2kl * sin_kl) / 8. - (ipow2(K2) * cosh_2kl * sin_kl) / (10. * k5) + (3 * k * cos_kl * cosh((k * l) / 2.) * sinh((k * l) / 2.)) / 4. - (13 * ipow2(K2) * cos_kl * cosh((k * l) / 2.) * sinh((k * l) / 2.)) / (50. * k5) + (3 * k * cos_kl * cosh((k * l) / 2.) * cosh_kl * sinh((k * l) / 2.)) / 4. - (ipow2(K2) * cos_kl * cosh((k * l) / 2.) * cosh_kl * sinh((k * l) / 2.)) / (10. * k5) - (3 * k * sin_kl * ipow2(sinh((k * l) / 2.))) / 4. + (13 * ipow2(K2) * sin_kl * ipow2(sinh((k * l) / 2.))) / (50. * k5) - (3 * k * cosh_kl * sin_kl * ipow2(sinh((k * l) / 2.))) / 4. + (ipow2(K2) * cosh_kl * sin_kl * ipow2(sinh((k * l) / 2.))) / (10. * k5) + (17 * ipow2(K2) * cos_kl * sinh_kl) / (50. * k5) + (3 * k * cos_kl * ipow2(sinh((k * l) / 2.)) * sinh_kl) / 4. - (ipow2(K2) * cos_kl * ipow2(sinh((k * l) / 2.)) * sinh_kl) / (10. * k5) + (k * cos_kl * sinh_2kl) / 16. - (ipow2(K2) * cos_kl * sinh_2kl) / (20. * k5);
      U[3][3][1][1] = (-7 * cos_kl) / 32. - (3 * ipow2(K2) * cos_kl) / (100. * k6) + (2 * ipow2(K2) * cos_kl * cosh_kl) / (25. * k6) + (7 * cos_kl * cosh_2kl) / 32. - (ipow2(K2) * cos_kl * cosh_2kl) / (20. * k6) - (k * l * sin_kl) / 8. - (9 * ipow2(K2) * l * sin_kl) / (40. * k5) + (9 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k6) - (sin_kl * sinh_2kl) / 32. - (3 * ipow2(K2) * sin_kl * sinh_2kl) / (80. * k6);
      U[3][3][2][2] = (27 * k2 * cos_kl) / 64. - (9 * ipow2(K2) * cos_kl) / (320. * k4) - (27 * k2 * cos_3kl) / 64. + (9 * ipow2(K2) * cos_3kl) / (320. * k4) - (3 * k3 * l * sin_kl) / 16. - (3 * ipow2(K2) * l * sin_kl) / (16. * k3) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (10. * k4);
      U[3][3][3][2] = (-3 * k2 * l * cos_kl) / 16. - (3 * ipow2(K2) * l * cos_kl) / (16. * k4) - (3 * k * sin_kl) / 16. - (3 * ipow2(K2) * sin_kl) / (16. * k5) - (9 * k * cos_2kl * sin_kl) / 16. + (3 * ipow2(K2) * cos_2kl * sin_kl) / (80. * k5) + (3 * ipow2(K2) * cosh_kl * sin_kl) / (20. * k5) - (9 * k * cos_kl * sin_2kl) / 32. + (3 * ipow2(K2) * cos_kl * sin_2kl) / (160. * k5) + (3 * ipow2(K2) * cos_kl * sinh_kl) / (20. * k5);
      U[3][3][3][3] = (-9 * cos_kl) / 64. + (79 * ipow2(K2) * cos_kl) / (1600. * k6) + (9 * cos_3kl) / 64. - (3 * ipow2(K2) * cos_3kl) / (320. * k6) - (ipow2(K2) * cos_kl * cosh_kl) / (25. * k6) - (3 * k * l * sin_kl) / 16. - (11 * ipow2(K2) * l * sin_kl) / (80. * k5) + (3 * ipow2(K2) * sin_kl * sinh_kl) / (25. * k6);
      U[3][5][2][0] = (-3 * K2 * l * cos_kl) / 20. - (3 * K2 * l * cos_kl * cosh_kl) / 20. - (11 * K2 * sin_kl) / (100. * k) - (11 * K2 * cosh_kl * sin_kl) / (100. * k) - (12 * K2 * cos_kl * sinh_kl) / (25. * k) + (9 * K2 * l * sin_kl * sinh_kl) / 20.;
      U[3][5][2][1] = (12 * K2 * cos_kl) / (25. * k2) - (12 * K2 * cos_kl * cosh_kl) / (25. * k2) - (9 * K2 * l * sin_kl) / (20. * k) + (9 * K2 * l * cosh_kl * sin_kl) / (20. * k) - (3 * K2 * l * cos_kl * sinh_kl) / (20. * k) - (11 * K2 * sin_kl * sinh_kl) / (100. * k2);
      U[3][5][3][0] = (K2 * cos_kl) / (25. * k2) - (K2 * cos_kl * cosh_kl) / (25. * k2) + (3 * K2 * l * sin_kl) / (20. * k) - (3 * K2 * l * cosh_kl * sin_kl) / (20. * k) - (9 * K2 * l * cos_kl * sinh_kl) / (20. * k) - (3 * K2 * sin_kl * sinh_kl) / (100. * k2);
      U[3][5][3][1] = (3 * K2 * l * cos_kl) / (10. * k2) - (9 * K2 * l * cos_kl * cosh_kl) / (20. * k2) + (11 * K2 * sin_kl) / (50. * k3) - (3 * K2 * cosh_kl * sin_kl) / (100. * k3) - (K2 * cos_kl * sinh_kl) / (25. * k3) - (3 * K2 * l * sin_kl * sinh_kl) / (20. * k2);
      U[3][5][5][2] = (-19 * k2 * l * cos_kl) / 32. - (13 * k * sin_kl) / 32. + (3 * k3 * l2 * sin_kl) / 32.;
      U[3][5][5][3] = (-3 * k2 * l2 * cos_kl) / 32. - (13 * k * l * sin_kl) / 32.;
    }
  }

  return (M);
}

VMATRIX *dqcor_matrix(double K1, double L, long order, double fse, double xkick, double ykick)
{
  double *C, **R, ***T, K0, J0;
  VMATRIX *M;

  K1 *= 1+fse;

  if (K1==0)
    return hvcorrector_matrix(L, xkick, ykick, 0.0, 0.0, 1.0, 1.0, 0, order);
  if (L==0)
    bombElegant("DQCOR length can't be zero", NULL);
  
  computeQuadSteeringStrengths(&K0, &J0, xkick, ykick, L, K1);
  
  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(2, order));
  C = M->C;
  R = M->R;
  T = M->T;
  
  if (K1>0) {
    double k, cos_kL,kp2,sin_kL,cosh_kL,sinh_kL,J0p2,K0p2,sin_2kL,kp3,sinh_2kL,cos_2kL,cosh_2kL,kp4,K0p3,Lp2,J0p3,Lp3;
    k = sqrt(K1);
    cos_kL = cos(k*L) ;
    kp2 = pow(k,2) ;
    sin_kL = sin(k*L) ;
    cosh_kL = cosh(k*L) ;
    sinh_kL = sinh(k*L) ;
    J0p2 = pow(J0,2) ;
    K0p2 = pow(K0,2) ;
    sin_2kL = sin(2*k*L) ;
    kp3 = pow(k,3) ;
    sinh_2kL = sinh(2*k*L) ;
    cos_2kL = cos(2*k*L) ;
    cosh_2kL = cosh(2*k*L) ;
    kp4 = pow(k,4) ;
    K0p3 = pow(K0,3) ;
    Lp2 = pow(L,2) ;
    J0p3 = pow(J0,3) ;
    Lp3 = pow(L,3);
    
    C[0] = -(K0/kp2) + (K0*cos_kL)/kp2;
    C[1] = -((K0*sin_kL)/k);
    C[2] = J0/kp2 - (J0*cosh_kL)/kp2;
    C[3] = -((J0*sinh_kL)/k);
    C[4] = L - (0.25*J0p2*L)/kp2 + (0.25*K0p2*L)/kp2 -  (0.125*K0p2*sin_2kL)/kp3 +  (0.125*J0p2*sinh_2kL)/kp3 ;

    R[0][0] = cos_kL;
    R[0][1] = sin_kL/k;
    R[0][5] = (K0*L*sin_kL)/(2.*k);
    R[1][0] = -(k*sin_kL);
    R[1][1] = cos_kL;
    R[1][5] = (K0*L*cos_kL)/2. + (K0*sin_kL)/(2.*k);
    R[2][2] = cosh_kL;
    R[2][3] = sinh_kL/k;
    R[2][5] = (J0*L*sinh_kL)/(2.*k);
    R[3][2] = k*sinh_kL;
    R[3][3] = cosh_kL;
    R[3][5] = (J0*L*cosh_kL)/2. + (J0*sinh_kL)/(2.*k);
    R[4][0] =  + 0.5*K0*L - (0.25*K0*sin_2kL)/k;
    R[4][1] =  - (0.25*K0)/kp2 + (0.25*K0*cos_2kL)/kp2;
    R[4][2] =  + 0.5*J0*L - (0.25*J0*sinh_2kL)/k;
    R[4][3] =  + (0.25*J0)/kp2 - (0.25*J0*cosh_2kL)/kp2;
    R[4][4] = 1;
    R[4][5] =  + (0.25*J0p2*L)/kp2 - (0.25*K0p2*L)/kp2 +  (0.125*K0p2*L*cos_2kL)/kp2 -  (0.125*J0p2*L*cosh_2kL)/kp2 +  (0.0625*K0p2*sin_2kL)/kp3 -  (0.0625*J0p2*sinh_2kL)/kp3 ;
    R[5][5] = 1;

    if (M->order > 1) {
      T[0][0][0] = -2*K0*pow(sin((k*L)/2.),4);
      T[0][1][0] = (K0*pow(sin((k*L)/2.),2)*sin_kL)/k;
      T[0][1][1] = (-2*K0*pow(sin((k*L)/2.),2))/kp2 - (K0*cos_kL*pow(sin((k*L)/2.),2))/kp2 ;
      T[0][2][0] = (2*J0*cos_kL)/5. - (2*J0*cos_kL*cosh_kL)/5. + (J0*sin_kL*sinh_kL)/5.;
      T[0][2][1] = (3*J0*sin_kL)/(5.*k) - (2*J0*cosh_kL*sin_kL)/(5.*k) - (J0*cos_kL*sinh_kL)/(5.*k) ;
      T[0][2][2] = K0/4. - (K0*cos_kL)/5. - (K0*cosh_2kL)/20.;
      T[0][3][0] = (J0*sin_kL)/(5.*k) + (J0*cosh_kL*sin_kL)/(5.*k) - (2*J0*cos_kL*sinh_kL)/(5.*k) ;
      T[0][3][1] = (J0*cos_kL)/(5.*kp2) - (J0*cos_kL*cosh_kL)/(5.*kp2) - (2*J0*sin_kL*sinh_kL)/(5.*kp2) ;
      T[0][3][2] = (K0*sin_kL)/(10.*k) - (K0*sinh_2kL)/(20.*k);
      T[0][3][3] = -0.25*K0/kp2 + (3*K0*cos_kL)/(10.*kp2) - (K0*cosh_2kL)/(20.*kp2) ;
      T[0][5][0] = (3*K0p2)/(8.*kp2) - (2*J0p2*cos_kL)/(25.*kp2) - (K0p2*cos_kL)/(3.*kp2) -  (K0p2*cos_2kL)/(24.*kp2) +  (2*J0p2*cos_kL*cosh_kL)/(25.*kp2) + (k*L*sin_kL)/2. +  (J0p2*L*cosh_kL*sin_kL)/(10.*k) -  (K0p2*L*sin_2kL)/(8.*k) -  (J0p2*L*cos_kL*sinh_kL)/(5.*k) +  (3*J0p2*sin_kL*sinh_kL)/(50.*kp2) ;
      T[0][5][1] = (-3*K0p2*L)/(8.*kp2) - (L*cos_kL)/2. + (K0p2*L*cos_2kL)/(8.*kp2) -  (J0p2*L*cos_kL*cosh_kL)/(10.*kp2) +  (2*J0p2*sin_kL)/(25.*kp3) + sin_kL/(2.*k) +  (K0p2*sin_kL)/(3.*kp3) +  (2*J0p2*cosh_kL*sin_kL)/(25.*kp3) -  (K0p2*sin_2kL)/(24.*kp3) -  (3*J0p2*cos_kL*sinh_kL)/(50.*kp3) -  (J0p2*L*sin_kL*sinh_kL)/(5.*kp2) ;
      T[0][5][2] = (J0*K0)/(8.*kp2) - (3*J0*K0*cos_kL)/(25.*kp2) - (J0*K0*cosh_2kL)/(200.*kp2) - (J0*K0*L*sinh_2kL)/(40.*k) ;
      T[0][5][3] = -0.125*(J0*K0*L)/kp2 - (J0*K0*L*cosh_2kL)/(40.*kp2) + (4*J0*K0*sin_kL)/(25.*kp3) - (J0*K0*sinh_2kL)/(200.*kp3) ;
      T[0][5][5] = (3*J0p2*K0)/(16.*kp4) - K0/kp2 + (3*K0p3)/(16.*kp4) -  (J0p2*K0*Lp2)/(16.*kp2) -  (3*K0p3*Lp2)/(16.*kp2) -  (21*J0p2*K0*cos_kL)/(125.*kp4) + (K0*cos_kL)/kp2 -  (K0p3*cos_kL)/(9.*kp4) - (K0*Lp2*cos_kL)/8. -  (11*K0p3*cos_2kL)/(144.*kp4) +  (K0p3*Lp2*cos_2kL)/(16.*kp2) -  (2*J0p2*K0*cos_kL*cosh_kL)/(125.*kp4) -  (J0p2*K0*Lp2*cos_kL*cosh_kL)/(20.*kp2) -  (7*J0p2*K0*cosh_2kL)/(2000.*kp4) -  (J0p2*K0*Lp2*cosh_2kL)/(80.*kp2) +  (K0*L*sin_kL)/(8.*k) + (13*J0p2*K0*L*cosh_kL*sin_kL)/  (100.*kp3) - (K0p3*L*sin_2kL)/(24.*kp3) +  (9*J0p2*K0*L*cos_kL*sinh_kL)/(100.*kp3) -  (81*J0p2*K0*sin_kL*sinh_kL)/(500.*kp4) -  (J0p2*K0*Lp2*sin_kL*sinh_kL)/(10.*kp2) -  (J0p2*K0*L*sinh_2kL)/(200.*kp3) ;
      T[1][0][0] = -2*k*K0*pow(sin((k*L)/2.),2)*sin_kL;
      T[1][1][0] = K0*pow(sin((k*L)/2.),2) + 2*K0*cos_kL*pow(sin((k*L)/2.),2);
      T[1][1][1] = (-2*K0*cos((k*L)/2.)*sin((k*L)/2.))/k - (K0*cos((3*k*L)/2.)*sin((k*L)/2.))/k;
      T[1][2][0] = (-2*J0*k*sin_kL)/5. + (3*J0*k*cosh_kL*sin_kL)/5. - (J0*k*cos_kL*sinh_kL)/5. ;
      T[1][2][1] = (3*J0*cos_kL)/5. - (3*J0*cos_kL*cosh_kL)/5. - (J0*sin_kL*sinh_kL)/5.;
      T[1][2][2] = (k*K0*sin_kL)/5. - (k*K0*sinh_2kL)/10.;
      T[1][3][0] = (J0*cos_kL)/5. - (J0*cos_kL*cosh_kL)/5. + (3*J0*sin_kL*sinh_kL)/5.;
      T[1][3][1] = -0.2*(J0*sin_kL)/k - (J0*cosh_kL*sin_kL)/(5.*k) - (3*J0*cos_kL*sinh_kL)/(5.*k) ;
      T[1][3][2] = (K0*cos_kL)/10. - (K0*cosh_2kL)/10.;
      T[1][3][3] = (-3*K0*sin_kL)/(10.*k) - (K0*sinh_2kL)/(10.*k);
      T[1][5][0] = (kp2*L*cos_kL)/2. - (K0p2*L*cos_2kL)/4. - (J0p2*L*cos_kL*cosh_kL)/10. + (2*J0p2*sin_kL)/(25.*k) +  (k*sin_kL)/2. + (K0p2*sin_kL)/(3.*k) +  (2*J0p2*cosh_kL*sin_kL)/(25.*k) -  (K0p2*sin_2kL)/(24.*k) -  (3*J0p2*cos_kL*sinh_kL)/(50.*k) +  (3*J0p2*L*sin_kL*sinh_kL)/10. ;
      T[1][5][1] = (-3*K0p2)/(8.*kp2) + (2*J0p2*cos_kL)/(25.*kp2) +  (K0p2*cos_kL)/(3.*kp2) +  (K0p2*cos_2kL)/(24.*kp2) -  (2*J0p2*cos_kL*cosh_kL)/(25.*kp2) + (k*L*sin_kL)/2. -  (J0p2*L*cosh_kL*sin_kL)/(10.*k) -  (K0p2*L*sin_2kL)/(4.*k) -  (3*J0p2*L*cos_kL*sinh_kL)/(10.*k) -  (3*J0p2*sin_kL*sinh_kL)/(50.*kp2) ;
      T[1][5][2] = -0.05*(J0*K0*L*cosh_2kL) + (3*J0*K0*sin_kL)/(25.*k) - (7*J0*K0*sinh_2kL)/(200.*k) ;
      T[1][5][3] = -0.125*(J0*K0)/kp2 + (4*J0*K0*cos_kL)/(25.*kp2) - (7*J0*K0*cosh_2kL)/(200.*kp2) - (J0*K0*L*sinh_2kL)/(20.*k) ;
      T[1][5][5] = -0.125*(J0p2*K0*L)/kp2 - (3*K0p3*L)/(8.*kp2) - (K0*L*cos_kL)/8. + (K0p3*L*cos_2kL)/(24.*kp2) +  (3*J0p2*K0*L*cos_kL*cosh_kL)/(25.*kp2) -  (7*J0p2*K0*L*cosh_2kL)/(200.*kp2) +  (21*J0p2*K0*sin_kL)/(125.*kp3) - (7*K0*sin_kL)/(8.*k) +  (K0p3*sin_kL)/(9.*kp3) + (k*K0*Lp2*sin_kL)/8. -  (2*J0p2*K0*cosh_kL*sin_kL)/(125.*kp3) -  (J0p2*K0*Lp2*cosh_kL*sin_kL)/(20.*k) +  (K0p3*sin_2kL)/(9.*kp3) -  (K0p3*Lp2*sin_2kL)/(8.*k) -  (11*J0p2*K0*cos_kL*sinh_kL)/(125.*kp3) -  (3*J0p2*K0*Lp2*cos_kL*sinh_kL)/(20.*k) -  (4*J0p2*K0*L*sin_kL*sinh_kL)/(25.*kp2) -  (3*J0p2*K0*sinh_2kL)/(250.*kp3) -  (J0p2*K0*Lp2*sinh_2kL)/(40.*k) ;
      T[2][0][0] = J0/4. - (J0*cos_2kL)/20. - (J0*cosh_kL)/5.;
      T[2][1][0] = -0.05*(J0*sin_2kL)/k + (J0*sinh_kL)/(10.*k);
      T[2][1][1] = J0/(4.*kp2) + (J0*cos_2kL)/(20.*kp2) - (3*J0*cosh_kL)/(10.*kp2) ;
      T[2][2][0] = (2*K0*cosh_kL)/5. - (2*K0*cos_kL*cosh_kL)/5. - (K0*sin_kL*sinh_kL)/5. ;
      T[2][2][1] = (-2*K0*cosh_kL*sin_kL)/(5.*k) + (K0*sinh_kL)/(5.*k) + (K0*cos_kL*sinh_kL)/(5.*k) ;
      T[2][2][2] = -2*J0*pow(sinh((k*L)/2.),4);
      T[2][3][0] = -0.2*(K0*cosh_kL*sin_kL)/k + (3*K0*sinh_kL)/(5.*k) - (2*K0*cos_kL*sinh_kL)/(5.*k) ;
      T[2][3][1] = -0.2*(K0*cosh_kL)/kp2 + (K0*cos_kL*cosh_kL)/(5.*kp2) - (2*K0*sin_kL*sinh_kL)/(5.*kp2) ;
      T[2][3][2] = (J0*sinh_kL)/(2.*k) - (J0*sinh_2kL)/(4.*k);
      T[2][3][3] = (-2*J0*pow(sinh((k*L)/2.),2))/kp2 - (J0*cosh_kL*pow(sinh((k*L)/2.),2))/kp2 ;
      T[2][5][0] = -0.125*(J0*K0)/kp2 + (J0*K0*cos_2kL)/(200.*kp2) + (9*J0*K0*cosh_kL)/(25.*kp2) -  (6*J0*K0*cos_kL*cosh_kL)/(25.*kp2) -  (J0*K0*L*cosh_kL*sin_kL)/(10.*k) - (J0*K0*L*sin_2kL)/(40.*k) -  (J0*K0*L*cos_kL*sinh_kL)/(5.*k) +  (9*J0*K0*sin_kL*sinh_kL)/(50.*kp2) ;
      T[2][5][1] = (J0*K0*L)/(8.*kp2) + (J0*K0*L*cos_2kL)/(40.*kp2) + (J0*K0*L*cos_kL*cosh_kL)/(10.*kp2) -  (6*J0*K0*cosh_kL*sin_kL)/(25.*kp3) +  (J0*K0*sin_2kL)/(200.*kp3) +  (4*J0*K0*sinh_kL)/(25.*kp3) -  (9*J0*K0*cos_kL*sinh_kL)/(50.*kp3) -  (J0*K0*L*sin_kL*sinh_kL)/(5.*kp2) ;
      T[2][5][2] = (-3*J0p2)/(8.*kp2) + (J0p2*cosh_kL)/(3.*kp2) + (J0p2*cosh_2kL)/(24.*kp2) - (k*L*sinh_kL)/2. -  (J0p2*L*sinh_2kL)/(8.*k) ;
      T[2][5][3] = (3*J0p2*L)/(8.*kp2) - (L*cosh_kL)/2. - (J0p2*L*cosh_2kL)/(8.*kp2) -  (J0p2*sinh_kL)/(3.*kp3) + sinh_kL/(2.*k) +  (J0p2*cosh_kL*sinh_kL)/(12.*kp3) ;
      T[2][5][5] = (3*J0p3)/(16.*kp4) + J0/kp2 + (3*J0*K0p2)/(16.*kp4) +  (3*J0p3*Lp2)/(16.*kp2) +  (J0*K0p2*Lp2)/(16.*kp2) -  (7*J0*K0p2*cos_2kL)/(2000.*kp4) +  (J0*K0p2*Lp2*cos_2kL)/(80.*kp2) -  (J0p3*cosh_kL)/(9.*kp4) - (J0*cosh_kL)/kp2 -  (21*J0*K0p2*cosh_kL)/(125.*kp4) -  (J0*Lp2*cosh_kL)/8. -  (2*J0*K0p2*cos_kL*cosh_kL)/(125.*kp4) +  (J0*K0p2*Lp2*cos_kL*cosh_kL)/(20.*kp2) -  (11*J0p3*cosh_2kL)/(144.*kp4) -  (J0p3*Lp2*cosh_2kL)/(16.*kp2) -  (9*J0*K0p2*L*cosh_kL*sin_kL)/(100.*kp3) +  (J0*K0p2*L*sin_2kL)/(200.*kp3) + (J0*L*sinh_kL)/(8.*k) -  (13*J0*K0p2*L*cos_kL*sinh_kL)/(100.*kp3) +  (81*J0*K0p2*sin_kL*sinh_kL)/(500.*kp4) -  (J0*K0p2*Lp2*sin_kL*sinh_kL)/(10.*kp2) +  (J0p3*L*sinh_2kL)/(24.*kp3) ;
      T[3][0][0] = (J0*k*sin_2kL)/10. - (J0*k*sinh_kL)/5.;
      T[3][1][0] = -0.1*(J0*cos_2kL) + (J0*cosh_kL)/10.;
      T[3][1][1] = -0.1*(J0*sin_2kL)/k - (3*J0*sinh_kL)/(10.*k);
      T[3][2][0] = (k*K0*cosh_kL*sin_kL)/5. + (2*k*K0*sinh_kL)/5. - (3*k*K0*cos_kL*sinh_kL)/5. ;
      T[3][2][1] = (K0*cosh_kL)/5. - (K0*cos_kL*cosh_kL)/5. - (3*K0*sin_kL*sinh_kL)/5.;
      T[3][2][2] = -2*J0*k*pow(sinh((k*L)/2.),2)*sinh_kL;
      T[3][3][0] = (3*K0*cosh_kL)/5. - (3*K0*cos_kL*cosh_kL)/5. + (K0*sin_kL*sinh_kL)/5. ;
      T[3][3][1] = (-3*K0*cosh_kL*sin_kL)/(5.*k) - (K0*sinh_kL)/(5.*k) - (K0*cos_kL*sinh_kL)/(5.*k) ;
      T[3][3][2] = (J0*cosh_kL)/2. - (J0*cosh_2kL)/2.;
      T[3][3][3] = (-2*J0*cosh((k*L)/2.)*sinh((k*L)/2.))/k - (J0*cosh((3*k*L)/2.)*sinh((k*L)/2.))/k ;
      T[3][5][0] = -0.05*(J0*K0*L*cos_2kL) - (3*J0*K0*L*cos_kL*cosh_kL)/10. + (8*J0*K0*cosh_kL*sin_kL)/(25.*k) - (7*J0*K0*sin_2kL)/(200.*k) +  (9*J0*K0*sinh_kL)/(25.*k) - (13*J0*K0*cos_kL*sinh_kL)/(50.*k) +  (J0*K0*L*sin_kL*sinh_kL)/10. ;
      T[3][5][1] = (J0*K0)/(8.*kp2) + (7*J0*K0*cos_2kL)/(200.*kp2) + (4*J0*K0*cosh_kL)/(25.*kp2) -  (8*J0*K0*cos_kL*cosh_kL)/(25.*kp2) -  (3*J0*K0*L*cosh_kL*sin_kL)/(10.*k) - (J0*K0*L*sin_2kL)/(20.*k) -  (J0*K0*L*cos_kL*sinh_kL)/(10.*k) -  (13*J0*K0*sin_kL*sinh_kL)/(50.*kp2) ;
      T[3][5][2] = -0.5*(kp2*L*cosh_kL) - (J0p2*L*cosh_2kL)/4. + (J0p2*sinh_kL)/(3.*k) - (k*sinh_kL)/2. -  (J0p2*cosh_kL*sinh_kL)/(12.*k) ;
      T[3][5][3] = (3*J0p2)/(8.*kp2) - (J0p2*cosh_kL)/(3.*kp2) - (J0p2*pow(cosh_kL,2))/(24.*kp2) - (k*L*sinh_kL)/2. -  (J0p2*pow(sinh_kL,2))/(24.*kp2) -  (J0p2*L*sinh_2kL)/(4.*k) ;
      T[3][5][5] = (3*J0p3*L)/(8.*kp2) + (J0*K0p2*L)/(8.*kp2) + (7*J0*K0p2*L*cos_2kL)/(200.*kp2) - (J0*L*cosh_kL)/8. -  (3*J0*K0p2*L*cos_kL*cosh_kL)/(25.*kp2) -  (J0p3*L*cosh_2kL)/(24.*kp2) +  (11*J0*K0p2*cosh_kL*sin_kL)/(125.*kp3) -  (3*J0*K0p2*Lp2*cosh_kL*sin_kL)/(20.*k) +  (3*J0*K0p2*sin_2kL)/(250.*kp3) -  (J0*K0p2*Lp2*sin_2kL)/(40.*k) -  (J0p3*sinh_kL)/(9.*kp3) - (7*J0*sinh_kL)/(8.*k) -  (21*J0*K0p2*sinh_kL)/(125.*kp3) -  (J0*k*Lp2*sinh_kL)/8. +  (2*J0*K0p2*cos_kL*sinh_kL)/(125.*kp3) -  (J0*K0p2*Lp2*cos_kL*sinh_kL)/(20.*k) -  (4*J0*K0p2*L*sin_kL*sinh_kL)/(25.*kp2) -  (J0p3*sinh_2kL)/(9.*kp3) -  (J0p3*Lp2*sinh_2kL)/(8.*k) ;

      T[4][0][0] = + 0.25*kp2*L - 0.125*k*sin_2kL;
      T[4][1][0] = -0.25 + 0.25*cos_2kL;
      T[4][1][1] = + 0.25*L + (0.125*sin_2kL)/k;
      T[4][2][2] = - 0.25*kp2*L + 0.125*k*sinh_2kL;
      T[4][3][2] = -0.25 + 0.25*cosh_2kL;
      T[4][3][3] = + 0.25*L + (0.125*sinh_2kL)/k;
      T[4][5][0] = - 0.25*K0*L + 0.125*K0*L*cos_2kL + (0.0625*K0*sin_2kL)/k;
      T[4][5][1] = + (0.0625*K0)/kp2 + 0.125*K0*Lp2 - (0.0625*K0*cos_2kL)/kp2 + (0.125*K0*L*sin_2kL)/k ;
      T[4][5][2] = - 0.25*J0*L + 0.125*J0*L*cosh_2kL + (0.0625*J0*sinh_2kL)/k ;
      T[4][5][3] = - (0.0625*J0)/kp2 + 0.125*J0*Lp2 + (0.0625*J0*cosh_2kL)/kp2 + (0.125*J0*L*sinh_2kL)/k ;
      T[4][5][5] = - (0.0625*J0p2*L)/kp2 + (0.0625*K0p2*L)/kp2 +  0.020833333333333332*J0p2*Lp3 +  0.020833333333333332*K0p2*Lp3 -  (0.03125*K0p2*L*cos_2kL)/kp2 +  (0.03125*J0p2*L*cosh_2kL)/kp2 -  (0.015625*K0p2*sin_2kL)/kp3 +  (0.03125*K0p2*Lp2*sin_2kL)/k +  (0.015625*J0p2*sinh_2kL)/kp3 +  (0.03125*J0p2*Lp2*sinh_2kL)/k   ;
    }
  } else {
    double k, cosh_kL,kp2,sinh_kL,cos_kL,sin_kL,J0p2,K0p2,sin_2kL,kp3,sinh_2kL,cosh_2kL,cos_2kL,kp4,K0p3,Lp2,J0p3,Lp3;
    k = sqrt(-K1);
    cosh_kL = cosh(k*L) ;
    kp2 = pow(k,2) ;
    sinh_kL = sinh(k*L) ;
    cos_kL = cos(k*L) ;
    sin_kL = sin(k*L) ;
    J0p2 = pow(J0,2) ;
    K0p2 = pow(K0,2) ;
    sin_2kL = sin(2*k*L) ;
    kp3 = pow(k,3) ;
    sinh_2kL = sinh(2*k*L) ;
    cosh_2kL = cosh(2*k*L) ;
    cos_2kL = cos(2*k*L) ;
    kp4 = pow(k,4) ;
    K0p3 = pow(K0,3) ;
    Lp2 = pow(L,2) ;
    J0p3 = pow(J0,3) ;
    Lp3 = pow(L,3);
    
    C[0] = K0/kp2 - (K0*cosh_kL)/kp2;
    C[1] = -((K0*sinh_kL)/k);
    C[2] = -(J0/kp2) + (J0*cos_kL)/kp2;
    C[3] = -((J0*sin_kL)/k);
    C[4] =  + L + (0.25*J0p2*L)/kp2 - (0.25*K0p2*L)/kp2 -  (0.125*J0p2*sin_2kL)/kp3 +  (0.125*K0p2*sinh_2kL)/kp3 ;
      
    R[0][0] = cosh_kL;
    R[0][1] = sinh_kL/k;
    R[0][5] = (K0*L*sinh_kL)/(2.*k);
    R[1][0] = k*sinh_kL;
    R[1][1] = cosh_kL;
    R[1][5] = (K0*L*cosh_kL)/2. + (K0*sinh_kL)/(2.*k);
    R[2][2] = cos_kL;
    R[2][3] = sin_kL/k;
    R[2][5] = (J0*L*sin_kL)/(2.*k);
    R[3][2] = -(k*sin_kL);
    R[3][3] = cos_kL;
    R[3][5] = (J0*L*cos_kL)/2. + (J0*sin_kL)/(2.*k);
    R[4][0] =  + 0.5*K0*L - (0.25*K0*sinh_2kL)/k;
    R[4][1] =  + (0.25*K0)/kp2 - (0.25*K0*cosh_2kL)/kp2;
    R[4][2] =  + 0.5*J0*L - (0.25*J0*sin_2kL)/k;
    R[4][3] =  - (0.25*J0)/kp2 + (0.25*J0*cos_2kL)/kp2;
    R[4][4] = 1;
    R[4][5] =  - (0.25*J0p2*L)/kp2 + (0.25*K0p2*L)/kp2 +  (0.125*J0p2*L*cos_2kL)/kp2 -  (0.125*K0p2*L*cosh_2kL)/kp2 +  (0.0625*J0p2*sin_2kL)/kp3 -  (0.0625*K0p2*sinh_2kL)/kp3 ;
    R[5][5] = 1;

    if (M->order > 1) {
      T[0][0][0] = -2*K0*pow(sinh((k*L)/2.),4);
      T[0][1][0] = (K0*sinh_kL)/(2.*k) - (K0*sinh_2kL)/(4.*k);
      T[0][1][1] = (-2*K0*pow(sinh((k*L)/2.),2))/kp2 - (K0*cosh_kL*pow(sinh((k*L)/2.),2))/kp2 ;
      T[0][2][0] = (2*J0*cosh_kL)/5. - (2*J0*cos_kL*cosh_kL)/5. - (J0*sin_kL*sinh_kL)/5. ;
      T[0][2][1] = -0.2*(J0*cosh_kL*sin_kL)/k + (3*J0*sinh_kL)/(5.*k) - (2*J0*cos_kL*sinh_kL)/(5.*k) ;
      T[0][2][2] = K0/4. - (K0*cos_2kL)/20. - (K0*cosh_kL)/5.;
      T[0][3][0] = (-2*J0*cosh_kL*sin_kL)/(5.*k) + (J0*sinh_kL)/(5.*k) + (J0*cos_kL*sinh_kL)/(5.*k) ;
      T[0][3][1] = -0.2*(J0*cosh_kL)/kp2 + (J0*cos_kL*cosh_kL)/(5.*kp2) - (2*J0*sin_kL*sinh_kL)/(5.*kp2) ;
      T[0][3][2] = -0.05*(K0*sin_2kL)/k + (K0*sinh_kL)/(10.*k);
      T[0][3][3] = K0/(4.*kp2) + (K0*cos_2kL)/(20.*kp2) - (3*K0*cosh_kL)/(10.*kp2) ;
      T[0][5][0] = (-3*K0p2)/(8.*kp2) + (2*J0p2*cosh_kL)/(25.*kp2) +  (K0p2*cosh_kL)/(3.*kp2) -  (2*J0p2*cos_kL*cosh_kL)/(25.*kp2) +  (K0p2*cosh_2kL)/(24.*kp2) -  (J0p2*L*cosh_kL*sin_kL)/(5.*k) - (k*L*sinh_kL)/2. +  (J0p2*L*cos_kL*sinh_kL)/(10.*k) +  (3*J0p2*sin_kL*sinh_kL)/(50.*kp2) -  (K0p2*L*sinh_2kL)/(8.*k) ;
      T[0][5][1] = (3*K0p2*L)/(8.*kp2) - (L*cosh_kL)/2. + (J0p2*L*cos_kL*cosh_kL)/(10.*kp2) -  (K0p2*L*cosh_2kL)/(8.*kp2) +  (3*J0p2*cosh_kL*sin_kL)/(50.*kp3) -  (2*J0p2*sinh_kL)/(25.*kp3) + sinh_kL/(2.*k) -  (K0p2*sinh_kL)/(3.*kp3) -  (2*J0p2*cos_kL*sinh_kL)/(25.*kp3) -  (J0p2*L*sin_kL*sinh_kL)/(5.*kp2) +  (K0p2*sinh_2kL)/(24.*kp3) ;
      T[0][5][2] = -0.125*(J0*K0)/kp2 + (J0*K0*cos_2kL)/(200.*kp2) + (3*J0*K0*cosh_kL)/(25.*kp2) - (J0*K0*L*sin_2kL)/(40.*k) ;
      T[0][5][3] = (J0*K0*L)/(8.*kp2) + (J0*K0*L*cos_2kL)/(40.*kp2) + (J0*K0*sin_2kL)/(200.*kp3) - (4*J0*K0*sinh_kL)/(25.*kp3) ;
      T[0][5][5] = (3*J0p2*K0)/(16.*kp4) + K0/kp2 + (3*K0p3)/(16.*kp4) +  (J0p2*K0*Lp2)/(16.*kp2) +  (3*K0p3*Lp2)/(16.*kp2) -  (7*J0p2*K0*cos_2kL)/(2000.*kp4) +  (J0p2*K0*Lp2*cos_2kL)/(80.*kp2) -  (21*J0p2*K0*cosh_kL)/(125.*kp4) -  (K0*cosh_kL)/kp2 - (K0p3*cosh_kL)/(9.*kp4) -  (K0*Lp2*cosh_kL)/8. -  (2*J0p2*K0*cos_kL*cosh_kL)/(125.*kp4) +  (J0p2*K0*Lp2*cos_kL*cosh_kL)/(20.*kp2) -  (11*K0p3*cosh_2kL)/(144.*kp4) -  (K0p3*Lp2*cosh_2kL)/(16.*kp2) -  (9*J0p2*K0*L*cosh_kL*sin_kL)/(100.*kp3) +  (J0p2*K0*L*sin_2kL)/(200.*kp3) + (K0*L*sinh_kL)/(8.*k) -  (13*J0p2*K0*L*cos_kL*sinh_kL)/(100.*kp3) +  (81*J0p2*K0*sin_kL*sinh_kL)/(500.*kp4) -  (J0p2*K0*Lp2*sin_kL*sinh_kL)/(10.*kp2) +  (K0p3*L*sinh_2kL)/(24.*kp3) ;
      T[1][0][0] = -2*k*K0*pow(sinh((k*L)/2.),2)*sinh_kL;
      T[1][1][0] = (K0*cosh_kL)/2. - (K0*cosh_2kL)/2.;
      T[1][1][1] = (-2*K0*cosh((k*L)/2.)*sinh((k*L)/2.))/k - (K0*cosh((3*k*L)/2.)*sinh((k*L)/2.))/k ;
      T[1][2][0] = (J0*k*cosh_kL*sin_kL)/5. + (2*J0*k*sinh_kL)/5. - (3*J0*k*cos_kL*sinh_kL)/5. ;
      T[1][2][1] = (3*J0*cosh_kL)/5. - (3*J0*cos_kL*cosh_kL)/5. + (J0*sin_kL*sinh_kL)/5. ;
      T[1][2][2] = (k*K0*sin_2kL)/10. - (k*K0*sinh_kL)/5.;
      T[1][3][0] = (J0*cosh_kL)/5. - (J0*cos_kL*cosh_kL)/5. - (3*J0*sin_kL*sinh_kL)/5.;
      T[1][3][1] = (-3*J0*cosh_kL*sin_kL)/(5.*k) - (J0*sinh_kL)/(5.*k) - (J0*cos_kL*sinh_kL)/(5.*k) ;
      T[1][3][2] = -0.1*(K0*cos_2kL) + (K0*cosh_kL)/10.;
      T[1][3][3] = -0.1*(K0*sin_2kL)/k - (3*K0*sinh_kL)/(10.*k);
      T[1][5][0] = -0.5*(kp2*L*cosh_kL) - (J0p2*L*cos_kL*cosh_kL)/10. - (K0p2*L*cosh_2kL)/4. -  (3*J0p2*cosh_kL*sin_kL)/(50.*k) +  (2*J0p2*sinh_kL)/(25.*k) - (k*sinh_kL)/2. +  (K0p2*sinh_kL)/(3.*k) +  (2*J0p2*cos_kL*sinh_kL)/(25.*k) -  (3*J0p2*L*sin_kL*sinh_kL)/10. - (K0p2*sinh_2kL)/(24.*k) ;
      T[1][5][1] = (3*K0p2)/(8.*kp2) - (2*J0p2*cosh_kL)/(25.*kp2) -  (K0p2*cosh_kL)/(3.*kp2) +  (2*J0p2*cos_kL*cosh_kL)/(25.*kp2) -  (K0p2*cosh_2kL)/(24.*kp2) -  (3*J0p2*L*cosh_kL*sin_kL)/(10.*k) - (k*L*sinh_kL)/2. -  (J0p2*L*cos_kL*sinh_kL)/(10.*k) -  (3*J0p2*sin_kL*sinh_kL)/(50.*kp2) -  (K0p2*L*sinh_2kL)/(4.*k) ;
      T[1][5][2] = -0.05*(J0*K0*L*cos_2kL) - (7*J0*K0*sin_2kL)/(200.*k) + (3*J0*K0*sinh_kL)/(25.*k) ;
      T[1][5][3] = (J0*K0)/(8.*kp2) + (7*J0*K0*cos_2kL)/(200.*kp2) - (4*J0*K0*cosh_kL)/(25.*kp2) - (J0*K0*L*sin_2kL)/(20.*k) ;
      T[1][5][5] = (J0p2*K0*L)/(8.*kp2) + (3*K0p3*L)/(8.*kp2) + (7*J0p2*K0*L*cos_2kL)/(200.*kp2) - (K0*L*cosh_kL)/8. -  (3*J0p2*K0*L*cos_kL*cosh_kL)/(25.*kp2) -  (K0p3*L*cosh_2kL)/(24.*kp2) +  (11*J0p2*K0*cosh_kL*sin_kL)/(125.*kp3) -  (3*J0p2*K0*Lp2*cosh_kL*sin_kL)/(20.*k) +  (3*J0p2*K0*sin_2kL)/(250.*kp3) -  (J0p2*K0*Lp2*sin_2kL)/(40.*k) -  (21*J0p2*K0*sinh_kL)/(125.*kp3) - (7*K0*sinh_kL)/(8.*k) -  (K0p3*sinh_kL)/(9.*kp3) - (k*K0*Lp2*sinh_kL)/8. +  (2*J0p2*K0*cos_kL*sinh_kL)/(125.*kp3) -  (J0p2*K0*Lp2*cos_kL*sinh_kL)/(20.*k) -  (4*J0p2*K0*L*sin_kL*sinh_kL)/(25.*kp2) -  (K0p3*sinh_2kL)/(9.*kp3) -  (K0p3*Lp2*sinh_2kL)/(8.*k) ;
      T[2][0][0] = J0/4. - (J0*cos_kL)/5. - (J0*cosh_2kL)/20.;
      T[2][1][0] = (J0*sin_kL)/(10.*k) - (J0*sinh_2kL)/(20.*k);
      T[2][1][1] = -0.25*J0/kp2 + (3*J0*cos_kL)/(10.*kp2) - (J0*cosh_2kL)/(20.*kp2) ;
      T[2][2][0] = (2*K0*cos_kL)/5. - (2*K0*cos_kL*cosh_kL)/5. + (K0*sin_kL*sinh_kL)/5.;
      T[2][2][1] = (K0*sin_kL)/(5.*k) + (K0*cosh_kL*sin_kL)/(5.*k) - (2*K0*cos_kL*sinh_kL)/(5.*k) ;
      T[2][2][2] = -2*J0*pow(sin((k*L)/2.),4);
      T[2][3][0] = (3*K0*sin_kL)/(5.*k) - (2*K0*cosh_kL*sin_kL)/(5.*k) - (K0*cos_kL*sinh_kL)/(5.*k) ;
      T[2][3][1] = (K0*cos_kL)/(5.*kp2) - (K0*cos_kL*cosh_kL)/(5.*kp2) - (2*K0*sin_kL*sinh_kL)/(5.*kp2) ;
      T[2][3][2] = (J0*pow(sin((k*L)/2.),2)*sin_kL)/k;
      T[2][3][3] = (-2*J0*pow(sin((k*L)/2.),2))/kp2 - (J0*cos_kL*pow(sin((k*L)/2.),2))/kp2 ;
      T[2][5][0] = (J0*K0)/(8.*kp2) - (9*J0*K0*cos_kL)/(25.*kp2) + (6*J0*K0*cos_kL*cosh_kL)/(25.*kp2) -  (J0*K0*cosh_2kL)/(200.*kp2) -  (J0*K0*L*cosh_kL*sin_kL)/(5.*k) -  (J0*K0*L*cos_kL*sinh_kL)/(10.*k) +  (9*J0*K0*sin_kL*sinh_kL)/(50.*kp2) -  (J0*K0*L*sinh_2kL)/(40.*k) ;
      T[2][5][1] = -0.125*(J0*K0*L)/kp2 - (J0*K0*L*cos_kL*cosh_kL)/(10.*kp2) - (J0*K0*L*cosh_2kL)/(40.*kp2) -  (4*J0*K0*sin_kL)/(25.*kp3) +  (9*J0*K0*cosh_kL*sin_kL)/(50.*kp3) +  (6*J0*K0*cos_kL*sinh_kL)/(25.*kp3) -  (J0*K0*L*sin_kL*sinh_kL)/(5.*kp2) -  (J0*K0*sinh_2kL)/(200.*kp3) ;
      T[2][5][2] = (J0p2*cos_kL)/(6.*kp2) - (J0p2*cos_2kL)/(6.*kp2) +  (J0p2*pow(sin((k*L)/2.),4))/kp2 + (k*L*sin_kL)/2. -  (J0p2*L*sin_2kL)/(8.*k) ;
      T[2][5][3] = (-3*J0p2*L)/(8.*kp2) - (L*cos_kL)/2. + (J0p2*L*cos_2kL)/(8.*kp2) +  (J0p2*sin_kL)/(3.*kp3) + sin_kL/(2.*k) -  (J0p2*cos_kL*sin_kL)/(12.*kp3) ;
      T[2][5][5] = (3*J0p3)/(16.*kp4) - J0/kp2 + (3*J0*K0p2)/(16.*kp4) -  (3*J0p3*Lp2)/(16.*kp2) -  (J0*K0p2*Lp2)/(16.*kp2) -  (J0p3*cos_kL)/(9.*kp4) + (J0*cos_kL)/kp2 -  (21*J0*K0p2*cos_kL)/(125.*kp4) -  (J0*Lp2*cos_kL)/8. -  (11*J0p3*cos_2kL)/(144.*kp4) +  (J0p3*Lp2*cos_2kL)/(16.*kp2) -  (2*J0*K0p2*cos_kL*cosh_kL)/(125.*kp4) -  (J0*K0p2*Lp2*cos_kL*cosh_kL)/(20.*kp2) -  (7*J0*K0p2*cosh_2kL)/(2000.*kp4) -  (J0*K0p2*Lp2*cosh_2kL)/(80.*kp2) +  (J0*L*sin_kL)/(8.*k) + (13*J0*K0p2*L*cosh_kL*sin_kL)/  (100.*kp3) - (J0p3*L*sin_2kL)/(24.*kp3) +  (9*J0*K0p2*L*cos_kL*sinh_kL)/(100.*kp3) -  (81*J0*K0p2*sin_kL*sinh_kL)/(500.*kp4) -  (J0*K0p2*Lp2*sin_kL*sinh_kL)/(10.*kp2) -  (J0*K0p2*L*sinh_2kL)/(200.*kp3) ;
      T[3][0][0] = (J0*k*sin_kL)/5. - (J0*k*sinh_2kL)/10.;
      T[3][1][0] = (J0*cos_kL)/10. - (J0*cosh_2kL)/10.;
      T[3][1][1] = (-3*J0*sin_kL)/(10.*k) - (J0*sinh_2kL)/(10.*k);
      T[3][2][0] = (-2*k*K0*sin_kL)/5. + (3*k*K0*cosh_kL*sin_kL)/5. - (k*K0*cos_kL*sinh_kL)/5. ;
      T[3][2][1] = (K0*cos_kL)/5. - (K0*cos_kL*cosh_kL)/5. + (3*K0*sin_kL*sinh_kL)/5.;
      T[3][2][2] = -2*J0*k*pow(sin((k*L)/2.),2)*sin_kL;
      T[3][3][0] = (3*K0*cos_kL)/5. - (3*K0*cos_kL*cosh_kL)/5. - (K0*sin_kL*sinh_kL)/5.;
      T[3][3][1] = -0.2*(K0*sin_kL)/k - (K0*cosh_kL*sin_kL)/(5.*k) - (3*K0*cos_kL*sinh_kL)/(5.*k) ;
      T[3][3][2] = J0*pow(sin((k*L)/2.),2) + 2*J0*cos_kL*pow(sin((k*L)/2.),2);
      T[3][3][3] = (-2*J0*cos((k*L)/2.)*sin((k*L)/2.))/k - (J0*cos((3*k*L)/2.)*sin((k*L)/2.))/k;
      T[3][5][0] = (-3*J0*K0*L*cos_kL*cosh_kL)/10. - (J0*K0*L*cosh_2kL)/20. + (9*J0*K0*sin_kL)/(25.*k) - (13*J0*K0*cosh_kL*sin_kL)/(50.*k) +  (8*J0*K0*cos_kL*sinh_kL)/(25.*k) - (J0*K0*L*sin_kL*sinh_kL)/10. -  (7*J0*K0*sinh_2kL)/(200.*k) ;
      T[3][5][1] = -0.125*(J0*K0)/kp2 - (4*J0*K0*cos_kL)/(25.*kp2) + (8*J0*K0*cos_kL*cosh_kL)/(25.*kp2) -  (7*J0*K0*cosh_2kL)/(200.*kp2) -  (J0*K0*L*cosh_kL*sin_kL)/(10.*k) -  (3*J0*K0*L*cos_kL*sinh_kL)/(10.*k) -  (13*J0*K0*sin_kL*sinh_kL)/(50.*kp2) -  (J0*K0*L*sinh_2kL)/(20.*k) ;
      T[3][5][2] = (kp2*L*cos_kL)/2. - (J0p2*L*cos_2kL)/4. + (J0p2*sin_kL)/(3.*k) + (k*sin_kL)/2. -  (J0p2*cos_kL*sin_kL)/(12.*k) ;
      T[3][5][3] = (-3*J0p2)/(8.*kp2) + (J0p2*cos_kL)/(3.*kp2) + (J0p2*pow(cos_kL,2))/(24.*kp2) + (k*L*sin_kL)/2. -  (J0p2*pow(sin_kL,2))/(24.*kp2) -  (J0p2*L*sin_2kL)/(4.*k) ;
      T[3][5][5] = (-3*J0p3*L)/(8.*kp2) - (J0*K0p2*L)/(8.*kp2) - (J0*L*cos_kL)/8. + (J0p3*L*cos_2kL)/(24.*kp2) +  (3*J0*K0p2*L*cos_kL*cosh_kL)/(25.*kp2) -  (7*J0*K0p2*L*cosh_2kL)/(200.*kp2) +  (J0p3*sin_kL)/(9.*kp3) - (7*J0*sin_kL)/(8.*k) +  (21*J0*K0p2*sin_kL)/(125.*kp3) +  (J0*k*Lp2*sin_kL)/8. -  (2*J0*K0p2*cosh_kL*sin_kL)/(125.*kp3) -  (J0*K0p2*Lp2*cosh_kL*sin_kL)/(20.*k) +  (J0p3*sin_2kL)/(9.*kp3) -  (J0p3*Lp2*sin_2kL)/(8.*k) -  (11*J0*K0p2*cos_kL*sinh_kL)/(125.*kp3) -  (3*J0*K0p2*Lp2*cos_kL*sinh_kL)/(20.*k) -  (4*J0*K0p2*L*sin_kL*sinh_kL)/(25.*kp2) -  (3*J0*K0p2*sinh_2kL)/(250.*kp3) -  (J0*K0p2*Lp2*sinh_2kL)/(40.*k) ;
      T[4][0][0] = - 0.25*kp2*L + 0.125*k*sinh_2kL;
      T[4][1][0] = -0.25 + 0.25*cosh_2kL;
      T[4][1][1] = + 0.25*L + (0.125*sinh_2kL)/k;
      T[4][2][2] = + 0.25*kp2*L - 0.125*k*sin_2kL;
      T[4][3][2] = -0.25 + 0.25*cos_2kL;
      T[4][3][3] = + 0.25*L + (0.125*sin_2kL)/k;
      T[4][5][0] = - 0.25*K0*L + 0.125*K0*L*cosh_2kL + (0.0625*K0*sinh_2kL)/k ;
      T[4][5][1] = - (0.0625*K0)/kp2 + 0.125*K0*Lp2 + (0.0625*K0*cosh_2kL)/kp2 + (0.125*K0*L*sinh_2kL)/k ;
      T[4][5][2] = - 0.25*J0*L + 0.125*J0*L*cos_2kL + (0.0625*J0*sin_2kL)/k;
      T[4][5][3] = + (0.0625*J0)/kp2 + 0.125*J0*Lp2 - (0.0625*J0*cos_2kL)/kp2 + (0.125*J0*L*sin_2kL)/k ;
      T[4][5][5] = + (0.0625*J0p2*L)/kp2 - (0.0625*K0p2*L)/kp2 +  0.020833333333333332*J0p2*Lp3 +  0.020833333333333332*K0p2*Lp3 -  (0.03125*J0p2*L*cos_2kL)/kp2 +  (0.03125*K0p2*L*cosh_2kL)/kp2 -  (0.015625*J0p2*sin_2kL)/kp3 +  (0.03125*J0p2*Lp2*sin_2kL)/k +  (0.015625*K0p2*sinh_2kL)/kp3 +  (0.03125*K0p2*Lp2*sinh_2kL)/k   ;
    }
  }
  return M;
}

void computeQuadSteeringStrengths(double *K0, double *J0, double xkick0, double ykick0, double L, double K1)
{
    double xr, yr, alpha;
    double dxp, dyp;
    if (K1==0) {
      *K0 = -xkick0/L;
      *J0 = -ykick0/L;
    } else {
      xr = -xkick0/(K1*L);
      yr = ykick0/(K1*L);
      alpha = sqrt(fabs(K1));
      if (K1>=0) {
	dxp = alpha*xr*sin(alpha*L);
	dyp = -alpha*yr*sinh(alpha*L);
      } else {
	dxp = -alpha*xr*sinh(alpha*L);
	dyp = alpha*yr*sin(alpha*L);
      }
      if (xkick0 && dxp)
	*K0 = sqr(xkick0)/dxp/L;
      else
	*K0 = 0;
      if (ykick0 && dyp)
	*J0 = sqr(ykick0)/dyp/L;
      else
	*J0 = 0;
    }
}

