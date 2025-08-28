/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include <stdio.h>
#include "mdb.h"
#include "constants.h"
#include "zibs.h"
#include "track.h"

/* prototypes */
void IBSRate(double particles,
             long elements, long superperiods, long verbosity, long isRing,
             double emitx0, double emity0, double sigmaDelta0, double sigmaz0,
             double *s, double *pCentral, double *betax, double *alphax, double *betay,
             double *alphay, double *etax, double *etaxp, double *etay, double *etayp,
             double *xRateVsS, double *yRateVsS, double *zRateVsS,
             double *xGrowthRate, double *yGrowthRate, double *zGrowthRate, long isElegant,
             double sTotal) {
  long i, j, k;
  double test = 1e-5;
  double simpsonCoeff[2] = {2., 4.};
  double coulombLogReturn, constant, weight;
  double h, lambda0, lambda = 0, cof, term, func, polyx, polyy, polyz;
  double txi, tyi, tzi, sumx, sumy, sumz, zintx, zinty, zintz;

  double beta, gamma0, gamma1, gamma2;
  double emitx, emity, sigmaDelta, sigmaz;
  double phix, phiy, Cx, Cy;
  double x11, x12, x22, y22, y23, y33, z22;
  double A, c11[3], c22[3], c33[3], c12[3], c23[3];
  double a0, b0, c0, ax, bx, ay, by, az, bz;

#define STEPS 100
#define MAXDECADES 30
  long maxDecades = MAXDECADES;
  long steps = STEPS; /* number of integration steps per decade */

  if (elements < 2)
    bombElegant(NULL, "There are fewer than two elements in the twiss function arrays.\n");

  /* elegant give exit beam parameter. ibsEmit give input beam parameter */
  if (isElegant) {
    gamma0 = sqrt(pCentral[0] * pCentral[0] + 1);
    gamma1 = sqrt(pCentral[elements - 1] * pCentral[elements - 1] + 1);
    emitx0 = emitx0 * gamma1 / gamma0;
    emity0 = emity0 * gamma1 / gamma0;
    sigmaDelta0 = sigmaDelta0 * gamma1 / gamma0;
  } else {
    gamma0 = gamma1 = sqrt(pCentral[0] * pCentral[0] + 1);
  }

  if (verbosity > 2) {
    printf("IBS Rates:\n");
    printf("gamma=%le, emitx = %le, emity = %le, sigmaDelta = %le, sigmaz = %le\n",
           gamma0, emitx0, emity0, sigmaDelta0, sigmaz0);
    printf("superperiods = %ld, particles = %le, isRing=%ld\n",
           superperiods, particles, isRing);
  }

  *xGrowthRate = 0.0;
  *yGrowthRate = 0.0;
  *zGrowthRate = 0.0;

  steps = steps + steps % 2;

  gamma2 = gamma0 * gamma0;
  beta = pCentral[0] / gamma0;
  emitx = emitx0;
  emity = emity0;
  sigmaDelta = sigmaDelta0;
  sigmaz = sigmaz0;

  if (sTotal <= 0)
    sTotal = s[elements - 1] - s[0];

  for (i = 0; i < elements; i++) {
    if (i == 0) {
      weight = FABS(s[1] - s[0]) / 2.0 / sTotal;
    } else if (i == (elements - 1)) {
      weight = FABS(s[elements - 1] - s[elements - 2]) / 2.0 / sTotal;
    } else {
      weight = FABS(s[i + 1] - s[i - 1]) / 2.0 / sTotal;
    }

    if (!weight) {
      if (xRateVsS)
        xRateVsS[i] = 0;
      if (yRateVsS)
        yRateVsS[i] = 0;
      if (zRateVsS)
        zRateVsS[i] = 0;
      continue;
    }

    if (!isRing) {
      gamma2 = pCentral[i] * pCentral[i] + 1;
      gamma1 = sqrt(gamma2);
      beta = pCentral[i] / gamma1;
      emitx = emitx0 * gamma0 / gamma1;
      emity = emity0 * gamma0 / gamma1;
      sigmaDelta = sigmaDelta0 * gamma0 / gamma1;
    }

    coulombLogReturn = coulombLog(gamma1, emitx, emity, betax[i], betay[i], sigmaz, particles, 1);
    constant = particles * sqr(re_mks) * c_mks /
               (8 * PI * ipow3(beta) * ipow2(gamma2) * emitx * emity * sigmaDelta * sigmaz);
    constant *= coulombLogReturn;
    if (verbosity > 6)
      fprintf(stdout, "Coulomb log: %g.\n constant=: %g \n", coulombLogReturn, constant);

    phix = etaxp[i] + (alphax[i] * etax[i] / betax[i]);
    phiy = etayp[i] + (alphay[i] * etay[i] / betay[i]);

    x11 = betax[i] / emitx;
    x12 = -x11 * gamma1 * phix;
    Cx = x11 * ipow2(etax[i] / betax[i]) * gamma2;
    x22 = Cx + x12 * x12 / x11;

    y33 = betay[i] / emity;
    y23 = -y33 * gamma1 * phiy;
    Cy = y33 * ipow2(etay[i] / betay[i]) * gamma2;
    y22 = Cy + y23 * y23 / y33;

    z22 = gamma2 / sigmaDelta / sigmaDelta;

    A = x22 + y22 + z22;

    a0 = x11 + y33 + A;
    b0 = (x11 + y33) * A - x12 * x12 - y23 * y23 + x11 * y33;
    c0 = x11 * y33 * (Cx + Cy + z22);

    c11[0] = A * y33 - y23 * y23;
    c11[1] = A + y33;
    c11[2] = 1.;

    c22[0] = x11 * y33;
    c22[1] = x11 + y33;
    c22[2] = 1;

    c33[0] = A * x11 - x12 * x12;
    ;
    c33[1] = A + x11;
    c33[2] = 1;

    c12[0] = -x12 * y33;
    c12[1] = -x12;
    c12[2] = 0;

    c23[0] = -x11 * y23;
    c23[1] = -y23;
    c23[2] = 0;

    ax = (x11 + x22) * (c11[1] + c22[1] + c33[1]) - 3. * (x11 * c11[1] + 2. * x12 * c12[1] + x22 * c22[1]);
    bx = (x11 + x22) * (c11[0] + c22[0] + c33[0]) - 3. * (x11 * c11[0] + 2. * x12 * c12[0] + x22 * c22[0]);

    ay = (y22 + y33) * (c11[1] + c22[1] + c33[1]) - 3. * (y33 * c33[1] + 2. * y23 * c23[1] + y22 * c22[1]);
    by = (y22 + y33) * (c11[0] + c22[0] + c33[0]) - 3. * (y33 * c33[0] + 2. * y23 * c23[0] + y22 * c22[0]);

    az = z22 * (c11[1] + c22[1] + c33[1]) - 3. * z22 * c22[1];
    bz = z22 * (c11[0] + c22[0] + c33[0]) - 3. * z22 * c22[0];

    /* Using simpson's rule to do the integral. 
       split integral into decades with "steps" steps per decade.
       integration intervals be [0,10], [10,100], [100,1000], and so on. 
       j is the index over these intervals. k is the index over each step.
    */
    lambda0 = 0.0;
    zintx = zinty = zintz = 0.0;
    for (j = 0; j < maxDecades; j++) {
      if (j == 0) {
        h = 1. / steps;
      } else {
        h = lambda * 9. / steps;
      }

      sumx = sumy = sumz = 0.0;
      for (k = 0; k <= steps; k++) {
        lambda = lambda0 + k * h;
        /* split decade into "steps" steps (even number) and do simpsons rule.
           odd points have cof 4 and even points have cof 2. 
           the first and last point just count once. */
        cof = simpsonCoeff[k % 2];
        if (k == 0 || k == steps)
          cof = 1.;
        /*
        term = pow((ipow3(alam) + a0*ipow2(alam) + b0*alam + c0),0.5);
        */
        term = sqrt(ipow3(lambda) + a0 * ipow2(lambda) + b0 * lambda + c0);
        func = sqrt(lambda) / term / term / term;
        cof *= func;
        polyz = (az * lambda + bz);
        polyx = (ax * lambda + bx);
        polyy = (ay * lambda + by);
        sumz += cof * polyz;
        sumx += cof * polyx;
        sumy += cof * polyy;
      }
      lambda0 = lambda;
      sumz = (sumz / 3.0) * h;
      sumx = (sumx / 3.0) * h;
      sumy = (sumy / 3.0) * h;
      zintz += sumz;
      zintx += sumx;
      zinty += sumy;
      /* Test to see if integral has converged */
      if (FABS(sumz / zintz) < test && FABS(sumx / zintx) < test && FABS(sumy / zinty) < test)
        break;
      if (j == maxDecades)
        fprintf(stdout, "**Warning** Integral did not converge in %ld decades.\n", maxDecades);
    }

    txi = constant * zintx;
    tyi = constant * zinty;
    tzi = constant * zintz;
    if (verbosity > 5) {
      fprintf(stdout, "i=%ld, coulombLogReturn=%g, constant=%g, zintx=%g, zinty=%g, zintz=%g\n",
              i, coulombLogReturn, constant / coulombLogReturn, zintx, zinty, zintz);
    }
    if (!isRing)
      tzi *= 2.0;

    *xGrowthRate += txi * weight;
    *yGrowthRate += tyi * weight;
    *zGrowthRate += tzi * weight;

    /* these are the rate per meter */
    if (xRateVsS)
      xRateVsS[i] = txi;
    if (yRateVsS)
      yRateVsS[i] = tyi;
    if (zRateVsS)
      zRateVsS[i] = tzi;
  }

  if (verbosity > 1) {
    fprintf(stdout, "(Weighted) average rates (1/sec): longitudinal= %g"
                    "   horizontal= %g   vertical= %g\n",
            *zGrowthRate, *xGrowthRate, *yGrowthRate);
  }

  return;
}

double coulombLog(double gamma, double emitx, double emity,
                  double betaxAve, double betayAve, double sigz, double particles,
                  long noWarning) {
  double EMeV, transverseEnergy, tempeV, sigmaxcm, sigmaycm, sigmazcm;
  double volume, density, charge, debyeLength, rmax, rmin, rminClassical, rminQuantum;
  double value;
  long debug = 0;

  EMeV = sqrt(sqr(gamma) + 1) * me_mev;
  /*
    Calculate transverse temperature as 2*p*x'
    i.e., assume the transverse energy is temperature/2
    */
  transverseEnergy = 0.5e6 * (gamma * EMeV - me_mev) * (emitx / betaxAve);
  tempeV = 2 * transverseEnergy;
  /*
    calculate beam volume to get density (in cm**-3) 
    */
  sigmaxcm = 100. * sqrt(emitx * betaxAve);
  sigmaycm = 100. * sqrt(emity * betayAve);
  sigmazcm = 100. * sigz;
  volume = 8.0 * sqrt(pow(PI, 3)) * sigmaxcm * sigmaycm * sigmazcm;
  density = particles / volume;

  charge = 1;
  debyeLength = 743.4 * sqrt(tempeV / density) / charge;
  rmax = MIN(sigmaxcm, debyeLength);
  if (!noWarning && debyeLength < 2.0 * sigmaxcm)
    fprintf(stdout, "Warning: The beam density probably corresponds to an unreasonably high space charge tune shift in this case. (debyeLength < 2.0 * sigmaxcm)\n");

  /*
    Calculate rmin as larger of classical distance of closest approach
    or quantum mechanical diffraction limit from nuclear radius
    */
  rminClassical = 1.44e-7 * sqr(charge) / tempeV;
  rminQuantum = 1.9732858e-11 / (2.0 * sqrt(2.0e-6 * transverseEnergy * me_mev));
  rmin = MAX(rminClassical, rminQuantum);

  value = log(rmax / rmin);
  if (!noWarning && (value < 0.0))
    fprintf(stdout, "Warning: Colomb logarithm is less than zero.\n");
  if (debug) {
    fprintf(stdout, "Coulomb outputs (in cm):\nrminClassical = %10.3g\nrminQuantum = %10.3g\ndebyeLength = %10.3g\nsigmaxcm = %10.3g\n", rminClassical, rminQuantum, debyeLength, sigmaxcm);
  }
  return value;
}

#ifdef INCLUDE_COUPLING_ANGLE_FUNCTIONS
/* Unclear why this code is here, since it is never called. */

static double xRate, yRate;

double couplingAngleFn(double theta) {
  double st22;
  st22 = sqr(sin(theta) / 2);
  return (yRate + xRate) * st22 / (yRate + st22 * (xRate - 3 * yRate));
}

double computeCouplingAngle(double taux, double tauy, double emitx, double emity) {
  /* Returns solution of 
     emity/emitx = sin^2(theta)/4 * (2/tauy + 2/taux) / (2/tauy + sin^2(theta)/4*(2/taux-3*2/tauy)) 
     */
  double theta;

  xRate = 2 / taux;
  yRate = 2 / tauy;
  /* Use the interval-halving method since the derivatives go to zero for theta=0 and theta=pi/2 */
  theta = zeroInterp(couplingAngleFn, emity / emitx, 0.0, PI / 2, 0.01, 1e-10);
  /* printf("coupling angle merit function = %le (theta = %le)\n", couplingAngleFn(theta)-emity/emitx, theta); */
  return theta;
}
#endif
