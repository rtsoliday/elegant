/*
  Two functions quadFringe and dipoleFringe are provided for tracking fringe field effects.
  The argument vec is the phase space vector to be tracked.
  The argument inFringe is a flag: -1 for going into the magnet and +1 for going out.
  There are magnet/fringe parameters need to be setup (indicated below).

  Physics approximations:

  assuming the fringes are symmetric
  quadFringe: soft linear fringe maps, leading order nonlinear generator
  dipoleFringe: a leading order nonlinear generator for hard edge

  If highOrder sets to 1, higher order terms are included for sufficient symplecticity
  (not necessarily more physics), but speed will sacrify.
 
* Original code by C.-X. Wang.
* Modified for use in elegant by M. Borland.
*/

#include "mdb.h"
#include "track.h"

static VMATRIX *quadPartialFringeMatrix(VMATRIX *M, double K1, long inFringe, double *fringeInt, long part);

void quadFringe(double **coord,
                const int64_t np,
                const double K1,
                double *fringeIntM0, /* I0m/K1, I1m/K1, I2m/K1, I3m/K1, Lambda2m/K1 */
                double *fringeIntP0, /* I0p/K1, I1p/K1, I2p/K1, I3p/K1, Lambda2p/K1 */
                const int backtrack,       /* 0 = forward, otherwise backward */
                const int inFringe,        /* -1 = entrance, +1 = exit */
                const int higherOrder,     /* +/-1=linear, +/-2=linear+3rd order, +/-3=linear+higher order */
                const int linearFlag,
                const double nonlinearFactor) {
  /* vec = {x, qx, y, qy, s, delta}
     */
  int64_t ip;
  double x, px, y, py, delta, xp, yp, denom;
  double *vec;
  double a, dx, dpx, dy, dpy, ds;
  static VMATRIX *M1;
  static VMATRIX *M2;

  if (linearFlag) {
    if (!M1) {
      M1 = tmalloc(sizeof(*M1));
      initialize_matrices(M1, 1);
      M2 = tmalloc(sizeof(*M2));
      initialize_matrices(M2, 1);
    }

    /* determine linear matrix for this delta */
    quadPartialFringeMatrix(M1, K1, 1, fringeIntM0, 1);
    quadPartialFringeMatrix(M2, K1, 1, fringeIntP0, 2);
    if (inFringe * (backtrack ? -1 : 1) == -1) {
      VMATRIX *Mtmp;
      SWAP_DOUBLE(M1->R[0][0], M1->R[1][1]);
      SWAP_DOUBLE(M1->R[2][2], M1->R[3][3]);
      SWAP_DOUBLE(M2->R[0][0], M2->R[1][1]);
      SWAP_DOUBLE(M2->R[2][2], M2->R[3][3]);
      Mtmp = M1;
      M1 = M2;
      M2 = Mtmp;
    }
    if (backtrack) {
      /* invert the matrices */
      SWAP_DOUBLE(M1->R[0][0], M1->R[1][1]);
      SWAP_DOUBLE(M1->R[2][2], M1->R[3][3]);
      M1->R[0][1] *= -1;
      M1->R[1][0] *= -1;
      M1->R[2][3] *= -1;
      M1->R[3][2] *= -1;
      SWAP_DOUBLE(M2->R[0][0], M2->R[1][1]);
      SWAP_DOUBLE(M2->R[2][2], M2->R[3][3]);
      M2->R[0][1] *= -1;
      M2->R[1][0] *= -1;
      M2->R[2][3] *= -1;
      M2->R[3][2] *= -1;
    }
  }

  for (ip = 0; ip < np; ip++) {
    vec = coord[ip];
    delta = vec[5];

    /* convert from (xp, yp) to (px, py) */
    xp = vec[1];
    yp = vec[3];
    denom = sqrt(1 + sqr(xp) + sqr(yp));
#if TURBO_RECIPROCALS
    vec[1] = ((1 + delta) / denom) * xp;
    vec[3] = ((1 + delta) / denom) * yp ;
#else
    vec[1] = (1 + delta) * xp / denom;
    vec[3] = (1 + delta) * yp / denom;
#endif

    if (linearFlag) {
      x = M1->R[0][0] * vec[0] + M1->R[0][1] * vec[1];
      px = M1->R[1][0] * vec[0] + M1->R[1][1] * vec[1];
      y = M1->R[2][2] * vec[2] + M1->R[2][3] * vec[3];
      py = M1->R[3][2] * vec[2] + M1->R[3][3] * vec[3];
    } else {
      x = vec[0];
      px = vec[1];
      y = vec[2];
      py = vec[3];
    }

    a = -inFringe * K1 / (12 * (1 + delta));

    dx = dpx = dy = dpy = ds = 0;

    if (abs(higherOrder) > 1) {
      double xpow[9], ypow[9], apow[4];
      //long i;
      xpow[0] = ypow[0] = apow[0] = 1;

      if (abs(higherOrder) > 2) {
        for (int i = 1; i < 9; i++) {
          xpow[i] = xpow[i - 1] * x;
          ypow[i] = ypow[i - 1] * y;
        }
        for (int i = 1; i < 4; i++)
          apow[i] = apow[i - 1] * a;
        dpx = (a * (12 * a * (-4 * py * x * y + px * (xpow[2] - 5 * ypow[2])) * (xpow[2] - ypow[2]) - 24 * (-2 * py * x * y + px * (xpow[2] + ypow[2])) + 4 * apow[2] * (-2 * py * x * y * (3 * xpow[4] + 50 * xpow[2] * ypow[2] - 21 * ypow[4]) + px * (xpow[6] + 75 * xpow[4] * ypow[2] - 105 * xpow[2] * ypow[4] - 35 * ypow[6])) + 3 * apow[3] * (-8 * py * x * y * (xpow[6] - 27 * xpow[4] * ypow[2] + 83 * xpow[2] * ypow[4] + 7 * ypow[6]) + px * (xpow[8] - 108 * xpow[6] * ypow[2] + 830 * xpow[4] * ypow[4] + 196 * xpow[2] * ypow[6] + 105 * ypow[8])))) / 8;
        dpy = (a * (-8 * px * x * y * (6 - 6 * a * (xpow[2] - ypow[2]) + apow[2] * (21 * xpow[4] - 50 * xpow[2] * ypow[2] - 3 * ypow[4]) + 3 * apow[3] * (7 * xpow[6] + 83 * xpow[4] * ypow[2] - 27 * xpow[2] * ypow[4] + ypow[6])) + py * (315 * apow[3] * xpow[8] + 28 * apow[2] * xpow[6] * (5 + 21 * a * ypow[2]) + 30 * a * xpow[4] * (2 + 14 * a * ypow[2] + 83 * apow[2] * ypow[4]) + ypow[2] * (24 + 12 * a * ypow[2] - 4 * apow[2] * ypow[4] + 3 * apow[3] * ypow[6]) - 12 * xpow[2] * (-2 + 6 * a * ypow[2] + 25 * apow[2] * ypow[4] + 27 * apow[3] * ypow[6])))) / 8;
        if (higherOrder > 0) {
          dx = (a * x * (8 * (xpow[2] + 3 * ypow[2]) + 4 * apow[2] * (5 * xpow[6] + 21 * xpow[4] * ypow[2] - 25 * xpow[2] * ypow[4] - ypow[6]) + apow[3] * (35 * xpow[8] + 84 * xpow[6] * ypow[2] + 498 * xpow[4] * ypow[4] - 108 * xpow[2] * ypow[6] + 3 * ypow[8]) + 12 * a * ipow2(xpow[2] - ypow[2]))) / 8;
          dy = (a * y * (-8 * (3 * xpow[2] + ypow[2]) + 4 * apow[2] * (xpow[6] + 25 * xpow[4] * ypow[2] - 21 * xpow[2] * ypow[4] - 5 * ypow[6]) + apow[3] * (3 * xpow[8] - 108 * xpow[6] * ypow[2] + 498 * xpow[4] * ypow[4] + 84 * xpow[2] * ypow[6] + 35 * ypow[8]) + 12 * a * ipow2(xpow[2] - ypow[2]))) / 8;
        }
      } else {
        for (int i = 1; i < 4; i++) {
          xpow[i] = xpow[i - 1] * x;
          ypow[i] = ypow[i - 1] * y;
        }
        dpx = 3 * a * (-px * xpow[2] + 2 * py * x * y - px * ypow[2]);
        dpy = 3 * a * (py * xpow[2] - 2 * px * x * y + py * ypow[2]);
        if (higherOrder > 0) {
          dx = a * (xpow[3] + 3 * x * ypow[2]);
          dy = a * (-3 * xpow[2] * y - ypow[3]);
        }
      }
      ds = (a / (1 + delta)) * (3 * py * y * xpow[2] - px * xpow[3] - 3 * px * x * ypow[2] + py * ypow[3]);
    }

    x += nonlinearFactor * dx;
    px += nonlinearFactor * dpx;
    y += nonlinearFactor * dy;
    py += nonlinearFactor * dpy;

    if (linearFlag) {
      /* determine and apply second linear matrix */
      vec[0] = M2->R[0][0] * x + M2->R[0][1] * px;
      vec[1] = M2->R[1][0] * x + M2->R[1][1] * px;
      vec[2] = M2->R[2][2] * y + M2->R[2][3] * py;
      vec[3] = M2->R[3][2] * y + M2->R[3][3] * py;
    } else {
      vec[0] = x;
      vec[1] = px;
      vec[2] = y;
      vec[3] = py;
    }
    vec[4] -= nonlinearFactor * ds;

    /* convert from (px, py) to (xp, yp) */
    px = vec[1];
    py = vec[3];
    denom = sqr(1 + delta) - sqr(px) - sqr(py);
    if (denom > 0) {
      denom = sqrt(denom);
#if TURBO_RECIPROCALS
      vec[1] = px * (1.0 / denom);
      vec[3] = py * (1.0 / denom);
#else
      vec[1] = px / denom;
      vec[3] = py / denom;
#endif
    } else
      vec[1] = vec[3] = DBL_MAX;
  }

//  if (M1) {
//    null_matrices(M1, 0);
//  }
//  if (M2) {
//    null_matrices(M2, 0);
//  }

}

void dipoleFringe(double *vec, double h, long inFringe, long higherOrder, double K1) {
  /* vec = {x, qx, y, qy, s, delta} */
  double x, px, y, py, delta;
  double a, dx, dpx, dy, dpy, ds;

  x = vec[0];
  px = vec[1];
  y = vec[2];
  py = vec[3];
  delta = vec[5];
  dx = dpx = dy = dpy = ds = 0;

  if (K1 != 0.0) {
    /* Modifications by Nicola Carmignani */
    a = -inFringe * K1 / (12 * (1 + delta));
    dx = a * (ipow3(x) + 3 * x * ipow2(y));
    dpx = 3 * a * (-px * ipow2(x) + 2 * py * x * y - px * ipow2(y));
    dy = a * (-3 * ipow2(x) * y - ipow3(y));
    dpy = 3 * a * (py * ipow2(x) - 2 * px * x * y + py * ipow2(y));
    ds = (a / (1 + delta)) * (3 * py * y * ipow2(x) - px * ipow3(x) - 3 * px * x * ipow2(y) + py * ipow3(y));
  } else {
    a = inFringe * h / (8 * (1 + delta));
    if (higherOrder) {
      double xr[11], yr[11], ar[11];
      long i, j;
      xr[0] = yr[0] = ar[0] = 1;
      for (j = 0, i = 1; i < 11; i++, j++) {
        xr[i] = xr[j] * x;
        yr[i] = yr[j] * y;
        ar[i] = ar[j] * a;
      }
      dx = (a * (7 * ar[7] * xr[9] + 7 * ar[8] * xr[10] + ar[5] * xr[7] * (7 + 27 * ar[2] * yr[2]) + ar[6] * xr[8] * (7 + 30 * ar[2] * yr[2]) + ar[4] * xr[6] * (7 + 24 * ar[2] * yr[2] + 30 * ar[4] * yr[4]) + ar[3] * xr[5] * (7 + 21 * ar[2] * yr[2] + 99 * ar[4] * yr[4]) + 3 * a * x * yr[2] * (-7 + 35 * ar[2] * yr[2] - 91 * ar[4] * yr[4] + 246 * ar[6] * yr[6]) + ar[2] * xr[4] * (7 + 21 * ar[2] * yr[2] - 90 * ar[4] * yr[4] + 1140 * ar[6] * yr[6]) + xr[3] * (7 * a + 189 * ar[5] * yr[4] - 975 * ar[7] * yr[6]) + 3 * yr[2] * (7 - 7 * ar[2] * yr[2] + 21 * ar[4] * yr[4] - 39 * ar[6] * yr[6] + 82 * ar[8] * yr[8]) - 7 * xr[2] * (-1 - 6 * ar[2] * yr[2] + 21 * ar[4] * yr[4] - 96 * ar[6] * yr[6] + 315 * ar[8] * yr[8]))) / 7;
      dpx = (a * (2 * py * y * (7 - 7 * a * x - 28 * ar[2] * yr[2] + 70 * x * ar[3] * yr[2] + 56 * x * ar[5] * (xr[2] - 6 * yr[2]) * yr[2] + 84 * ar[4] * yr[2] * (-xr[2] + yr[2]) + 3 * x * ar[7] * yr[2] * (xr[4] - 262 * xr[2] * yr[2] + 409 * yr[4]) - 4 * ar[6] * (5 * xr[4] * yr[2] - 162 * xr[2] * yr[4] + 57 * yr[6]) + 20 * ar[8] * (33 * xr[4] * yr[4] - 162 * xr[2] * yr[6] + 29 * yr[8])) + px * (-3 * ar[7] * xr[6] * yr[2] - 24 * ar[6] * xr[5] * yr[2] * (-1 + 55 * ar[2] * yr[2]) + 3 * ar[5] * xr[4] * yr[2] * (-28 + 655 * ar[2] * yr[2]) + 24 * ar[4] * xr[3] * yr[2] * (7 - 90 * ar[2] * yr[2] + 630 * ar[4] * yr[4]) + 3 * a * yr[2] * (-21 + 70 * ar[2] * yr[2] - 196 * ar[4] * yr[4] + 513 * ar[6] * yr[6]) - 7 * a * xr[2] * (-1 + 30 * ar[2] * yr[2] - 240 * ar[4] * yr[4] + 1227 * ar[6] * yr[6]) - 2 * x * (7 - 84 * ar[2] * yr[2] + 420 * ar[4] * yr[4] - 1596 * ar[6] * yr[6] + 5220 * ar[8] * yr[8])))) / 7;
      dy = -(a * y * (ar[7] * xr[6] * yr[2] + 8 * ar[6] * xr[5] * yr[2] * (-1 + 33 * ar[2] * yr[2]) - 8 * ar[4] * xr[3] * yr[2] * (7 - 54 * ar[2] * yr[2] + 270 * ar[4] * yr[4]) + xr[4] * (28 * ar[5] * yr[2] - 393 * ar[7] * yr[4]) + 3 * a * yr[2] * (7 - 14 * ar[2] * yr[2] + 28 * ar[4] * yr[4] - 57 * ar[6] * yr[6]) + a * xr[2] * (-7 + 70 * ar[2] * yr[2] - 336 * ar[4] * yr[4] + 1227 * ar[6] * yr[6]) + 2 * x * (7 - 28 * ar[2] * yr[2] + 84 * ar[4] * yr[4] - 228 * ar[6] * yr[6] + 580 * ar[8] * yr[8]))) / 7;
      dpy = (a * (-6 * px * y * (7 - 7 * a * x + 14 * ar[2] * (xr[2] - yr[2]) + 70 * x * ar[3] * yr[2] + 7 * ar[4] * (xr[4] - 14 * xr[2] * yr[2] + 9 * yr[4]) + 7 * ar[5] * (xr[5] + 18 * xr[3] * yr[2] - 39 * x * yr[4]) + 4 * ar[6] * (2 * xr[6] - 15 * xr[4] * yr[2] + 168 * xr[2] * yr[4] - 39 * yr[6]) + ar[7] * (9 * xr[7] + 66 * xr[5] * yr[2] - 975 * xr[3] * yr[4] + 984 * x * yr[6]) + 10 * ar[8] * (xr[8] + 2 * xr[6] * yr[2] + 114 * xr[4] * yr[4] - 294 * xr[2] * yr[6] + 41 * yr[8])) + py * (63 * ar[7] * xr[8] + 70 * ar[8] * xr[9] + 7 * ar[5] * xr[6] * (7 + 27 * ar[2] * yr[2]) + 8 * ar[6] * xr[7] * (7 + 30 * ar[2] * yr[2]) + 6 * ar[4] * xr[5] * (7 + 24 * ar[2] * yr[2] + 30 * ar[4] * yr[4]) + 5 * ar[3] * xr[4] * (7 + 21 * ar[2] * yr[2] + 99 * ar[4] * yr[4]) + 3 * a * xr[2] * (7 + 189 * ar[4] * yr[4] - 975 * ar[6] * yr[6]) + 3 * a * yr[2] * (-7 + 35 * ar[2] * yr[2] - 91 * ar[4] * yr[4] + 246 * ar[6] * yr[6]) + 4 * ar[2] * xr[3] * (7 + 21 * ar[2] * yr[2] - 90 * ar[4] * yr[4] + 1140 * ar[6] * yr[6]) - 14 * x * (-1 - 6 * ar[2] * yr[2] + 21 * ar[4] * yr[4] - 96 * ar[6] * yr[6] + 315 * ar[8] * yr[8])))) / 7;
    } else {
      dx = a * (ipow2(x) + 3 * ipow2(y));
      dpx = 2 * a * (-px * x + py * y);
      dy = -2 * a * x * y;
      dpy = 2 * a * (py * x - 3 * px * y);
    }

    ds = (a / (1 + delta)) * (2 * py * x * y - px * ipow2(x) - 3 * px * ipow2(y));
  }

  vec[0] += dx;
  vec[1] += dpx;
  vec[2] += dy;
  vec[3] += dpy;
  vec[4] += ds;
}

static VMATRIX *quadPartialFringeMatrix(VMATRIX *restrict M, const double K1, const long inFringe,
                                        double *restrict fringeInt, const long part) {
  double J1x, J2x, J3x, J1y, J2y, J3y;
  double K1sqr, expJ1x, expJ1y;

  if (!M) {
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, 1);
  }
  null_matrices(M, 0);
  M->R[4][4] = M->R[5][5] = 1;

  K1sqr = sqr(K1);

  if (part == 1) {
    J1x = inFringe * (K1 * fringeInt[1] - 2 * K1sqr * fringeInt[3] / 3. - K1sqr * fringeInt[0] * fringeInt[2] / 2);
    J2x = inFringe * K1 * fringeInt[2];
    J3x = inFringe * K1sqr * (fringeInt[2] + fringeInt[4] + fringeInt[0] * fringeInt[1]);

    //K1 = -K1;
    J1y = inFringe * (-K1 * fringeInt[1] - 2 * K1sqr * fringeInt[3] / 3. - K1sqr * fringeInt[0] * fringeInt[2] / 2);
    J2y = -J2x;
    J3y = J3x;
  } else {
    J1x = inFringe * (K1 * fringeInt[1] + K1sqr * fringeInt[0] * fringeInt[2] / 2);
    J2x = inFringe * K1 * fringeInt[2];
    J3x = inFringe * K1sqr * (fringeInt[4] - fringeInt[0] * fringeInt[1]);

    //K1 = -K1;
    J1y = inFringe * (-K1 * fringeInt[1] + K1sqr * fringeInt[0] * fringeInt[2] / 2);
    J2y = -J2x;
    J3y = J3x;
  }

  expJ1x = M->R[0][0] = exp(J1x);
  M->R[0][1] = J2x / expJ1x;
  M->R[1][0] = expJ1x * J3x;
  M->R[1][1] = (1 + J2x * J3x) / expJ1x;

  expJ1y = M->R[2][2] = exp(J1y);
  M->R[2][3] = J2y / expJ1y;
  M->R[3][2] = expJ1y * J3y;
  M->R[3][3] = (1 + J2y * J3y) / expJ1y;

  return M;
}

VMATRIX *quadFringeMatrix(VMATRIX *Mu, double K1, long backtrack, long inFringe, double *fringeIntM, double *fringeIntP) {
  VMATRIX *M1, *M2, *M;

  M1 = quadPartialFringeMatrix(NULL, K1, 1, fringeIntM, 1);
  M2 = quadPartialFringeMatrix(NULL, K1, 1, fringeIntP, 2);

  if (Mu) {
    M = Mu;
  } else {
    M = tmalloc(sizeof(*M));
    initialize_matrices(M, 1);
  }

  concat_matrices(M, M2, M1, 0);
  free_matrices(M1);
  free(M1);
  free_matrices(M2);
  free(M2);

  inFringe *= backtrack ? -1 : 1;
  if (inFringe == -1) {
    SWAP_DOUBLE(M->R[0][0], M->R[1][1]);
    SWAP_DOUBLE(M->R[2][2], M->R[3][3]);
  }

  if (backtrack) {
    SWAP_DOUBLE(M->R[0][0], M->R[1][1]);
    SWAP_DOUBLE(M->R[2][2], M->R[3][3]);
    M->R[0][1] *= -1;
    M->R[1][0] *= -1;
    M->R[2][3] *= -1;
    M->R[3][2] *= -1;
  }

  return M;
}
