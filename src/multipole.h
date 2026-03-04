#ifndef _MULTIPOLE_H
#define _MULTIPOLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "manual.h"


void computeTotalErrorMultipoleFields(MULTIPOLE_DATA *totalMult,
                                      MULTIPOLE_DATA *systematicMult,
				                              double systematicMultFactor,
                                      MULTIPOLE_DATA *edge1Mult,
                                      MULTIPOLE_DATA *edge2Mult,
                                      MULTIPOLE_DATA *randomMult,
				                              double randomMultFactor,
                                      MULTIPOLE_DATA *steeringMult,
				                              double steeringMultFactor,
                                      double KmL, long defaultOrder,
                                      long orderOverride, short *minOrder, short *maxOrder);

void randomizeErrorMultipoleFields(MULTIPOLE_DATA *randomMult);

//#define ODD(j) ((j)%2)
#define ODD(j) ((j)&1)

static inline double dblfactorial(long n) {
  double prod = 1;
  while (n > 1)
    prod *= n--;
  return prod;
}

#ifdef _MSC_VER
#define UNUSED /*empty*/
#else
#define UNUSED __attribute__((unused))
#endif

static inline void UNUSED fillPowerArray(const double x, double *xpow, const long order) {
  xpow[0] = 1;
  for (long i = 1; i <= order; i++) {
    xpow[i] = xpow[i - 1] * x;
  }
}

static void UNUSED fillPowerArrayReverse(const double x, double *xpow, const long order) {
  xpow[order] = 1;
  for (long i = order-1; i >= 0; i--) {
    xpow[i] = xpow[i + 1] * x;
  }
}

static inline void UNUSED fillPowerArrays(const double x, double *restrict xpow,
                                                    const double y, double *restrict ypow,
                                                    const long order) {
  xpow[0] = ypow[0] = 1;
  xpow[1] = x;
  ypow[1] = y;
  switch (order) {
    case 1:
      return;
    case 2:
      xpow[2] = x*x;
      ypow[2] = y*y;
      return;
    default:
      for (int i = 2; i <= order; i++) {
        xpow[i] = xpow[i - 1] * x;
        ypow[i] = ypow[i - 1] * y;
      }
  }
}

static inline double* expansion_coefficients(const long n) {
  static double **expansion_coef = NULL;
  static long *orderDone = NULL;
  static long maxOrder = -1;
  static double expansion_coef_0[] = {+1.};
  static double expansion_coef_1[] = {+1., +1.};
  static double expansion_coef_2[] = {+1./2, +1., -1./2};

  static double UNUSED expansion_coef_3[] = {+1./6, +1./2, -1./2, -1./6};
  static double UNUSED expansion_coef_4[] = {+1./24, +1./6, -1./(2*2), -1./6, +1./24};
  static double UNUSED expansion_coef_5[] = {+1./120, +1./24, -1./(2*6), -1./(6*2), +1./24, +1./120};
  static double UNUSED expansion_coef_6[] = {+1./720, +1./120, -1./(2*24), -1./(6*6),
                                      +1./(24*2), +1./120, -1./720};
  static double UNUSED expansion_coef_7[] = {+1./5040, +1./720, -1./(2*120), -1./(6*24),
                                      +1./(24*6), +1./(120*2), -1./720, -1./5040};

  switch (n) {
    case 0:
      return expansion_coef_0;
    case 1:
      return expansion_coef_1;
    case 2:
      return expansion_coef_2;
    default:
#define N_DEFAULT_ORDER 8
      if (maxOrder == -1) {
        expansion_coef = malloc(sizeof(*expansion_coef) * (N_DEFAULT_ORDER + 1));
        orderDone = calloc(N_DEFAULT_ORDER + 1, sizeof(*orderDone));
        maxOrder = N_DEFAULT_ORDER;
      }

      if (n <= maxOrder && orderDone[n])
        return expansion_coef[n];

      if (n > maxOrder) {
        //SDDS_Realloc
        expansion_coef = realloc(expansion_coef, sizeof(*expansion_coef) * (n + 1));
        orderDone = realloc(orderDone, sizeof(*orderDone) * (n + 1));
        for (long i = maxOrder + 1; i <= n; i++)
          orderDone[i] = 0;
        maxOrder = n;
      }

      expansion_coef[n] = malloc(sizeof(**expansion_coef) * (n + 1));

      /* calculate expansion coefficients with signs for (x+iy)^n/n! */
      for (long i = 0; i <= n; i++) {
        expansion_coef[n][i] = (ODD(i / 2) ? -1.0 : 1.0) / (dblfactorial(i) * dblfactorial(n - i));
      }
      orderDone[n] = 1;

      return expansion_coef[n];
  }
}




UNUSED
static void apply_all_kicks_noret(double *restrict qx,
                                  double *restrict qy,
                                  double x,
                                  double y,
                                  int *restrict KnLIdx,
                                  double *restrict KnL,
                                  double kickFrac,
                                  const int maxOrder,
                                  const int kidx,
                                  const int skew) {
  double sum_Fx, sum_Fy;
  double *coef;
  double qxt = 0, qyt = 0;
  double xpow[MAX_EXTRA_ORDER];
  double ypow[MAX_EXTRA_ORDER];

  ypow[0] = 1;
  for (int i = 1; i <= maxOrder; i++) {
    ypow[i] = ypow[i - 1] * y;
  }

  xpow[maxOrder] = 1;
  for (int i = maxOrder-1; i >= 0; i--) {
    xpow[i] = xpow[i + 1] * x;
  }

  for (int ki = 0; ki < kidx; ki++) {
    int order = KnLIdx[ki];
    coef = expansion_coefficients(order);
    sum_Fx = sum_Fy = 0;

    for (int i = 0; i <= order; i += 1) {
      double f = coef[i] * xpow[i+(maxOrder-order)] * ypow[i];
      if (i & 1) {
        sum_Fx += f;
      } else {
        sum_Fy += f;
      }
    }
    qxt -= KnL[ki] * kickFrac * sum_Fy;
    qyt += KnL[ki] * kickFrac * sum_Fx;
  }
  if (skew) {
    SWAP_DOUBLE(qxt, qyt);
    qyt = -qyt;
  }
  *qx += qxt;
  *qy += qyt;
}


static void UNUSED apply_canonical_multipole_kicks_ret(double *restrict qx, double *restrict qy,
                                            double *restrict delta_qx_return, double *restrict delta_qy_return,
                                            double *restrict xpow, double *restrict ypow,
                                            const long order, const double KnL, const long skew) {
  double *coef = expansion_coefficients(order);
  double sum_Fx = 0, sum_Fy = 0;

//  //even
//  for (i = 0; i <= order; i += 2)
//    sum_Fy += coef[i] * xpow[order - i] * ypow[i];
//  //odd
//  for (i = 1; i <= order; i += 2)
//    sum_Fx += coef[i] * xpow[order - i] * ypow[i];
  for (long i = 0; i <= order; i += 1) {
    double f = coef[i] * xpow[order-i] * ypow[i];
    if (i & 1) {
      sum_Fx += f;
    } else {
      sum_Fy += f;
    }
  }
  if (skew) {
    SWAP_DOUBLE(sum_Fx, sum_Fy);
    sum_Fx = -sum_Fx;
  }
  *qx -= KnL * sum_Fy;
  *qy += KnL * sum_Fx;
  *delta_qx_return -= KnL * sum_Fy;
  *delta_qy_return += KnL * sum_Fx;
}

static void UNUSED apply_canonical_multipole_kicks_noret(double *restrict qx, double *restrict qy,
                                            double *restrict xpow, double *restrict ypow,
                                            const int order, const double KnL, const int skew) {
  double *coef = expansion_coefficients(order);
  double sum_Fx = 0, sum_Fy = 0;

  for (int i = 0; i <= order; i += 1) {
    double f = coef[i] * xpow[order-i] * ypow[i];
    if (i & 1) {
      sum_Fx += f;
    } else {
      sum_Fy += f;
    }
  }
  if (skew) {
    SWAP_DOUBLE(sum_Fx, sum_Fy);
    sum_Fx = -sum_Fx;
  }
  *qx -= KnL * sum_Fy;
  *qy += KnL * sum_Fx;
}


extern unsigned short expandHamiltonian;

static inline int convertSlopesToMomenta(double *restrict qx, double *restrict qy, double xp, double yp, double delta) {
  if (expandHamiltonian) {
    *qx = (1 + delta) * xp;
    *qy = (1 + delta) * yp;
  } else {
    double denom;
    denom = sqrt(1 + sqr(xp) + sqr(yp));
#if TURBO_RECIPROCALS
    *qx = ((1 + delta) / denom) * xp ;
    *qy = ((1 + delta) / denom) * yp ;
#else
    *qx = (1 + delta) * xp / denom;
    *qy = (1 + delta) * yp / denom;
#endif
  }
  return 1;
}

static inline int convertMomentaToSlopes(double *restrict xp, double *restrict yp, double qx, double qy, double delta) {
  if (expandHamiltonian) {
    *xp = qx / (1 + delta);
    *yp = qy / (1 + delta);
  } else {
    double denom = sqr(1 + delta) - sqr(qx) - sqr(qy);
    if (denom <= 0) {
      printWarningForTracking("Particle acquired undefined slopes when integrating through kick multipole.", NULL);
      return 0;
    }
    denom = sqrt(denom);
#if TURBO_RECIPROCALS
    *xp = qx * (1.0 / denom);
    *yp = qy * (1.0 / denom);
#else
    *xp = qx / denom;
    *yp = qy / denom;
#endif
  }
  return 1;
}


#ifdef __cplusplus
}
#endif

#define EXSQRT(value, order) (order==0?sqrt(value):(1+0.5*((value)-1)))

#ifndef _GPU_MULTIPOLE_H_
extern unsigned long multipoleKicksDone;
#endif

#endif
