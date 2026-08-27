/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: polynomial_kicks()
 * purpose: apply kicks due to a polynomial dependence on x or y
 * 
 * Michael Borland, 1992 
 */
#include "mdb.h"
#include "track.h"
#include "multipole.h"

long polynomial_kicks(
  double **particle, /* initial/final phase-space coordinates */
  int64_t n_part,       /* number of particles */
  KPOLY *kpoly,      /* kick-polynomial structure */
  double p_error,    /* p_nominal/p_central */
  double Po,
  double **accepted,
  double z_start) {
  double KL0; /* on-momentum integrated strength at x=1 or y=1 */
  double KL;
  double dx, dy, dz; /* offsets of the multipole center */
  long order;        /* order (n) */
  int64_t i_part, i_top;
  long yplane;
  double *coord;
  double cos_tilt, sin_tilt;
  double x, xp, y, yp;

  log_entry("polynomial_kicks");

  if (!particle)
    bombElegant("particle array is null (polynomial_kicks)", NULL);

  if (!kpoly)
    bombElegant("null KPOLY pointer (polynomial_kicks)", NULL);

  if ((order = kpoly->order) < 0)
    bombElegant("order < 0 for KPOLY element (polynomial_kicks)", NULL);

  if (kpoly->plane && (kpoly->plane[0] == 'y' || kpoly->plane[0] == 'Y'))
    yplane = 1;
  else {
    if (kpoly->plane && !(kpoly->plane[0] == 'x' || kpoly->plane[0] == 'X'))
      bombElegantVA("KPOLY plane value of %c not recognized", kpoly->plane[0]);
    yplane = 0;
  }

  KL0 = kpoly->coefficient * kpoly->factor;

  cos_tilt = cos(kpoly->tilt);
  sin_tilt = sin(kpoly->tilt);
  dx = kpoly->dx;
  dy = kpoly->dy;
  dz = kpoly->dz;

  i_top = n_part - 1;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!(coord = particle[i_part])) {
      printf("null coordinate pointer for particle %" PRId64 " (polynomial_kicks)", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("null accepted coordinates pointer for particle %" PRId64 " (polynomial_kicks)", i_part);
      fflush(stdout);
      abort();
    }

    /* adjust strength for momentum offset--good to all orders */
    KL = KL0 / (1 + coord[5]);

    /* calculate coordinates in rotated and offset frame */
    coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    coord[0] += -dx + dz * coord[1];
    coord[2] += -dy + dz * coord[3];

    x = cos_tilt * coord[0] + sin_tilt * coord[2];
    y = -sin_tilt * coord[0] + cos_tilt * coord[2];
    xp = cos_tilt * coord[1] + sin_tilt * coord[3];
    yp = -sin_tilt * coord[1] + cos_tilt * coord[3];

#if defined(IEEE_MATH)
    if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
#endif
    if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
        FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }

    if (yplane)
      yp += KL * ipow(y, order);
    else
      xp += KL * ipow(x, order);

    /* undo the rotation and store in place of initial coordinates */
    /* don't need to change coord[0] or coord[2] since x and y are unchanged */
    coord[1] = cos_tilt * xp - sin_tilt * yp;
    coord[3] = sin_tilt * xp + cos_tilt * yp;

    /* remove the coordinate offsets */
    coord[0] += dx - coord[1] * dz;
    coord[2] += dy - coord[3] * dz;
    coord[4] -= dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  }
  log_exit("polynomial_kicks");
  return (i_top + 1);
}

long polynomial_hamiltonian(
  double **particle, /* initial/final phase-space coordinates */
  int64_t n_part,       /* number of particles */
  HKPOLY *hkpoly,    /* kick-polynomial structure */
  double p_error,    /* p_nominal/p_central */
  double Po,
  double **accepted,
  double z_start) {
  double dx, dy, dz; /* offsets of the multipole center */
  int64_t i_part, i_top;
  long ix, iy, iqx, iqy, ir, idelta;
  long ixMax, iyMax, iqxMax, iqyMax, ideltaMax;
  double *coord;
  double cos_tilt, sin_tilt;
  double x, xp, y, yp, qx, qy, delta;
  double factor, dfactor;
  double xpow, ypow, qxpow, qypow, deltapow;
  double xpow1, ypow1, qxpow1, qypow1;

  if (!particle)
    bombElegant("particle array is null (polynomial_hamiltonian)", NULL);

  if (!hkpoly)
    bombElegant("null HKPOLY pointer (polynomial_hamiltonian)", NULL);
  if (hkpoly->length < 0)
    bombElegant("HKPOLY length (L) must be non-negative (polynomial_hamiltonian)", NULL);
  if (hkpoly->nRepeats < 0)
    bombElegant("HKPOLY N_REPEATS must be positive (polynomial_hamiltonian)", NULL);

  cos_tilt = cos(hkpoly->tilt);
  sin_tilt = sin(hkpoly->tilt);
  dx = hkpoly->dx;
  dy = hkpoly->dy;
  dz = hkpoly->dz;

  ixMax = iyMax = -1;
  for (ix = 0; ix < 7; ix++) {
    for (iy = 0; iy < 7; iy++) {
      if (hkpoly->K[ix][iy] != 0) {
        if (ix > ixMax)
          ixMax = ix;
        if (iy > iyMax)
          iyMax = iy;
      }
    }
  }

  iqxMax = iqyMax = ideltaMax = -1;
  if (hkpoly->driftType == 1) {
    for (iqx = 0; iqx < 7; iqx++) {
      for (iqy = 0; iqy < 7; iqy++) {
        for (idelta = 0; idelta < 7; idelta++) {
          if (hkpoly->E[iqx][iqy][idelta] != 0) {
            bombElegantVA("E[%ld][%ld][%ld] is nonzero but DRIFT_TYPE=1 in HKPOLY",
                          iqx, iqy, idelta);
          }
        }
      }
    }

    for (iqx = 0; iqx < 7; iqx++) {
      for (iqy = 0; iqy < 7; iqy++) {
        if (hkpoly->D[iqx][iqy] != 0) {
          if (iqx > iqxMax)
            iqxMax = iqx;
          if (iqy > iqyMax)
            iqyMax = iqy;
        }
      }
    }
  } else if (hkpoly->driftType == 2) {
    for (iqx = 0; iqx < 7; iqx++) {
      for (iqy = 0; iqy < 7; iqy++) {
        if (hkpoly->D[iqx][iqy] != 0) {
          bombElegantVA("D[%ld][%ld] is nonzero but DRIFT_TYPE=2 in HKPOLY",
                        iqx, iqy);
        }
      }
    }

    for (iqx = 0; iqx < 7; iqx++) {
      for (iqy = 0; iqy < 7; iqy++) {
        for (idelta = 0; idelta < 7; idelta++) {
          if (hkpoly->E[iqx][iqy][idelta] != 0) {
            if (iqx > iqxMax)
              iqxMax = iqx;
            if (iqy > iqyMax)
              iqyMax = iqy;
            if (idelta > ideltaMax)
              ideltaMax = idelta;
          }
        }
      }
    }
  } else {
    bombElegant("DRIFT_TYPE must be 1 or 2 for HKPOLY", NULL);
  }

  i_top = n_part - 1;
  for (i_part = 0; i_part <= i_top; i_part++) {
    if (!(coord = particle[i_part])) {
      printf("null coordinate pointer for particle %" PRId64 " (polynomial_hamiltonian)", i_part);
      fflush(stdout);
      abort();
    }
    if (accepted && !accepted[i_part]) {
      printf("null accepted coordinates pointer for particle %" PRId64 " (polynomial_hamiltonian)", i_part);
      fflush(stdout);
      abort();
    }

    /* calculate coordinates in rotated and offset frame */
    coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    coord[0] += -dx + dz * coord[1];
    coord[2] += -dy + dz * coord[3];

    x = cos_tilt * coord[0] + sin_tilt * coord[2];
    y = -sin_tilt * coord[0] + cos_tilt * coord[2];
    xp = cos_tilt * coord[1] + sin_tilt * coord[3];
    yp = -sin_tilt * coord[1] + cos_tilt * coord[3];

    if (FABS(x) > coordLimit || FABS(y) > coordLimit ||
        FABS(xp) > slopeLimit || FABS(yp) > slopeLimit) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }

#if defined(IEEE_MATH)
    if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
      swapParticles(particle[i_part], particle[i_top]);
      if (accepted)
        swapParticles(accepted[i_part], accepted[i_top]);
      particle[i_top][4] = z_start;
      particle[i_top][5] = Po * (1 + particle[i_top][5]);
      i_top--;
      i_part--;
      continue;
    }
#endif

    factor = hkpoly->factor / hkpoly->nRepeats;
    delta = coord[5];
    convertSlopesToMomenta(&qx, &qy, xp, yp, delta);
    for (ir = 0; ir <= hkpoly->nRepeats; ir++) {
      dfactor = 1;
      if (ir == 0 || ir == hkpoly->nRepeats)
        dfactor = 0.5;
      if (hkpoly->driftType == 1) {
        qx /= (1 + delta);
        qy /= (1 + delta);
        qxpow = 1;
        qxpow1 = 0;
        for (iqx = 0; iqx <= iqxMax; iqx++) {
          qypow = 1;
          qypow1 = 0;
          for (iqy = 0; iqy <= iqyMax; iqy++) {
            if (iqx)
              x += factor * hkpoly->D[iqx][iqy] * qxpow1 * qypow * dfactor;
            if (iqy)
              y += factor * hkpoly->D[iqx][iqy] * qxpow * qypow1 * dfactor;
            qypow1 = qypow * (iqy + 1);
            qypow *= qy;
          }
          qxpow1 = qxpow * (iqx + 1);
          qxpow *= qx;
        }
        qx *= (1 + delta);
        qy *= (1 + delta);
      } else {
        qxpow = 1;
        qxpow1 = 0;
        for (iqx = 0; iqx <= iqxMax; iqx++) {
          qypow = 1;
          qypow1 = 0;
          for (iqy = 0; iqy <= iqyMax; iqy++) {
            deltapow = 1;
            for (idelta = 0; idelta <= ideltaMax; idelta++) {
              if (iqx)
                x += factor * hkpoly->E[iqx][iqy][idelta] * qxpow1 * qypow * deltapow * dfactor;
              if (iqy)
                y += factor * hkpoly->E[iqx][iqy][idelta] * qxpow * qypow1 * deltapow * dfactor;
              deltapow *= delta;
            }
            qypow1 = qypow * (iqy + 1);
            qypow *= qy;
          }
          qxpow1 = qxpow * (iqx + 1);
          qxpow *= qx;
        }
      }

      if (ir == hkpoly->nRepeats)
        break;

      xpow = 1;
      xpow1 = 0;
      for (ix = 0; ix <= ixMax; ix++) {
        ypow = 1;
        ypow1 = 0;
        for (iy = 0; iy <= iyMax; iy++) {
          if (ix)
            qx -= factor * hkpoly->K[ix][iy] * xpow1 * ypow;
          if (iy)
            qy -= factor * hkpoly->K[ix][iy] * xpow * ypow1;
          ypow1 = ypow * (iy + 1);
          ypow *= y;
        }
        xpow1 = xpow * (ix + 1);
        xpow *= x;
      }

    } /* loop over repeats */
    convertMomentaToSlopes(&xp, &yp, qx, qy, delta);

    /* undo the rotation and store in place of initial coordinates */
    coord[0] = cos_tilt * x - sin_tilt * y;
    coord[2] = sin_tilt * x + cos_tilt * y;
    coord[1] = cos_tilt * xp - sin_tilt * yp;
    coord[3] = sin_tilt * xp + cos_tilt * yp;

    /* remove the coordinate offsets */
    coord[0] += dx - coord[1] * dz;
    coord[2] += dy - coord[3] * dz;
    coord[4] -= dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  }

  return (i_top + 1);
}
