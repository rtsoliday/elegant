/*
 *----------------------------------------------------------------------------
 * Modification Log:
 * -----------------
 * .05  2004-11-01      M. Borland, borland@aps.anl.gov
 *                      Removed prototypes and put them in gwig.h
 * 
 * .04  2003-04-29      YK Wu, Duke University, wu@fel.duke.edu 
 *			using scientific notation for constants. 
 *                      Checked with TRACY pascal code.
 *                      Computing differential pathlength only.
 *
 * .03  2003-04-28      YK Wu, Duke University, wu@fel.duke.edu 
 *			Convert to C code and cross-checked with the pascal version;
 *
 * .02  2001-12-xx      Y. K. Wu, Duke University, wu@fel.duke.edu
 *                      Implementing DA version of the wiggler integrator for Pascal.
 *                      Gauge is disabled !!! (Dec. 4, 2001)
 *  
 * .01  2001-02-12      Y. K. Wu, LBNL
 *                      Implementing a generic wiggler integrator
 *                      for paraxial-ray Hamiltonian approximation.
 *
 *                    
 *----------------------------------------------------------------------------
 *  Accelerator Physics Group, Duke FEL Lab, www.fel.duke.edu  
 */

#include "gwig.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void GWigRadiationKicks(struct gwig *pWig, double *X, double *Bxyz, double dl, double *sigmaDelta2);
void GWigB(struct gwig *pWig, double *Xvec, double *B, double poleFactor);
void GWigAddToFieldOutput(CWIGGLER *cwiggler, double Z, double *X, double *B, double poleFactor);

#define FIELD_OUTPUT 0

void GWigPass_2nd(struct gwig *pWig, double *X, double *sigmaDelta2, long singleStep) {
  int i, Nstep;
  double dl;

#if FIELD_OUTPUT
  static FILE *fpd = NULL;
  static long index = 0;
  if (!fpd) {
    fpd = fopen("gwig.sdds", "w");
    fprintf(fpd, "SDDS1\n");
    fprintf(fpd, "&column name=i type=long &end\n");
    fprintf(fpd, "&column name=z type=double &end\n");
    fprintf(fpd, "&column name=x type=double &end\n");
    fprintf(fpd, "&column name=y type=double &end\n");
    fprintf(fpd, "&column name=px type=double &end\n");
    fprintf(fpd, "&column name=py type=double &end\n");
    fprintf(fpd, "&column name=delta type=double &end\n");
    fprintf(fpd, "&column name=Bx type=double &end\n");
    fprintf(fpd, "&column name=By type=double &end\n");
    fprintf(fpd, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  Nstep = pWig->PN * (pWig->Nw);
  dl = pWig->Lw / (pWig->PN);

  if ((pWig->sr || pWig->isr || FIELD_OUTPUT) && pWig->Zw == 0) {
    double B[2];
    double ax, ay, axpy, aypx;
    GWigAx(pWig, X, &ax, &axpy, pWig->cwiggler->poleFactor[0]);
    GWigAy(pWig, X, &ay, &aypx, pWig->cwiggler->poleFactor[0]);
    GWigB(pWig, X, B, pWig->cwiggler->poleFactor[0]);
    X[1] -= ax;
    X[3] -= ay;
    if (pWig->sr || pWig->isr)
      GWigRadiationKicks(pWig, X, B, dl, sigmaDelta2);
#if FIELD_OUTPUT
    fprintf(fpd, "%ld %e %e %e %e %e %e %e %e\n",
            index, pWig->Zw, X[0], X[2], X[1], X[3], X[4], B[0], B[1]);
    index++;
#endif
    X[1] += ax;
    X[3] += ay;
  }

  for (i = 1; i <= Nstep; i++) {
    double pf, B[2];
    pf = GWigPoleFactor(pWig, ((i - 1) * pWig->cwiggler->length) / Nstep);
    if (i==1 && pWig->cwiggler->fieldOutputInitialized) {
      GWigB(pWig, X, B, pf);
      GWigAddToFieldOutput(pWig->cwiggler, pWig->Zw, X, B, pf);
    }
    GWigMap_2nd(pWig, X, dl, pf);
    if (pWig->sr || pWig->isr || pWig->cwiggler->fieldOutputInitialized) {
      double ax, ay, axpy, aypx;
      GWigAx(pWig, X, &ax, &axpy, pf);
      GWigAy(pWig, X, &ay, &aypx, pf);
      GWigB(pWig, X, B, pf);
      X[1] -= ax;
      X[3] -= ay;
      if (pWig->sr || pWig->isr)
        GWigRadiationKicks(pWig, X, B, dl, sigmaDelta2);
      if (pWig->cwiggler->fieldOutputInitialized)
        GWigAddToFieldOutput(pWig->cwiggler, pWig->Zw, X, B, pf);
#if FIELD_OUTPUT
      fprintf(fpd, "%ld %e %e %e %e %e %e %e %e\n",
              index, pWig->Zw, X[0], X[2], X[1], X[3], X[4], B[0], B[1]);
      index++;
#endif
      X[1] += ax;
      X[3] += ay;
    }
    if (singleStep)
      break;
  }
}

void GWigPass_4th(struct gwig *pWig, double *X, double *sigmaDelta2, long singleStep) {

  const double x1 = 1.3512071919596576340476878089715e0;
  const double x0 = -1.7024143839193152680953756179429e0;

  int i, Nstep;
  double dl, dl1, dl0;

  Nstep = pWig->PN * (pWig->Nw);

  dl = pWig->Lw / (pWig->PN);

  dl1 = x1 * dl;
  dl0 = x0 * dl;

  if ((pWig->sr || pWig->isr || FIELD_OUTPUT) && pWig->Zw == 0) {
    double B[2];
    double ax, ay, axpy, aypx;
    GWigAx(pWig, X, &ax, &axpy, pWig->cwiggler->poleFactor[0]);
    GWigAy(pWig, X, &ay, &aypx, pWig->cwiggler->poleFactor[0]);
    GWigB(pWig, X, B, pWig->cwiggler->poleFactor[0]);
    X[1] -= ax;
    X[3] -= ay;
    GWigRadiationKicks(pWig, X, B, dl, sigmaDelta2);
    X[1] += ax;
    X[3] += ay;
  }

  for (i = 1; i <= Nstep; i++) {
    double pf, B[2];
    pf = GWigPoleFactor(pWig, ((i - 1) * pWig->cwiggler->length) / Nstep);
    if (i==1 && pWig->cwiggler->fieldOutputInitialized) {
      GWigB(pWig, X, B, pf);
      GWigAddToFieldOutput(pWig->cwiggler, pWig->Zw, X, B, pf);
    }
    GWigMap_2nd(pWig, X, dl1, pf);
    GWigMap_2nd(pWig, X, dl0, pf);
    GWigMap_2nd(pWig, X, dl1, pf);
    if (pWig->sr || pWig->isr || pWig->cwiggler->fieldOutputInitialized) {
      double ax, ay, axpy, aypx;
      GWigAx(pWig, X, &ax, &axpy, pf);
      GWigAy(pWig, X, &ay, &aypx, pf);
      GWigB(pWig, X, B, pf);
      X[1] -= ax;
      X[3] -= ay;
      if (pWig->sr || pWig->isr)
        GWigRadiationKicks(pWig, X, B, dl, sigmaDelta2);
      if (pWig->cwiggler->fieldOutputInitialized)
        GWigAddToFieldOutput(pWig->cwiggler, pWig->Zw, X, B, pf);
      X[1] += ax;
      X[3] += ay;
    }
    if (singleStep)
      break;
  }
}

void GWigMap_2nd(struct gwig *pWig, double *X, double dl, double poleFactor) {

  double dld, dl2, dl2d;
  double ax, ay, axpy, aypx;

  dld = dl / (1.0e0 + X[4]);
  dl2 = 0.5e0 * dl;
  dl2d = dl2 / (1.0e0 + X[4]);

  /* Step1: increase a half step in z */
  pWig->Zw = pWig->Zw + dl2;

  /* Step2: a half drift in y */
  GWigAy(pWig, X, &ay, &aypx, poleFactor);
  X[1] = X[1] - aypx;
  X[3] = X[3] - ay;

  X[2] = X[2] + dl2d * X[3];
  X[5] = X[5] + 0.5e0 * dl2d * (X[3] * X[3]) / (1.0e0 + X[4]);

  GWigAy(pWig, X, &ay, &aypx, poleFactor);
  X[1] = X[1] + aypx;
  X[3] = X[3] + ay;

  /* Step3: a full drift in x */
  GWigAx(pWig, X, &ax, &axpy, poleFactor);
  X[1] = X[1] - ax;
  X[3] = X[3] - axpy;

  X[0] = X[0] + dld * X[1];
  /* Full path length */
  X[5] = X[5] + dl + 0.5e0 * dld * (X[1] * X[1]) / (1.0e0 + X[4]);
  /* Differential path length only 
  X[5] = X[5] + 0.5e0*dld*(X[1]*X[1])/(1.0e0+X[4]);
   */

  GWigAx(pWig, X, &ax, &axpy, poleFactor);
  X[1] = X[1] + ax;
  X[3] = X[3] + axpy;

  /* Step4: a half drift in y */
  GWigAy(pWig, X, &ay, &aypx, poleFactor);
  X[1] = X[1] - aypx;
  X[3] = X[3] - ay;

  X[2] = X[2] + dl2d * X[3];
  X[5] = X[5] + 0.5e0 * dl2d * (X[3] * X[3]) / (1.0e0 + X[4]);

  GWigAy(pWig, X, &ay, &aypx, poleFactor);
  X[1] = X[1] + aypx;
  X[3] = X[3] + ay;

  /* Step5: increase a half step in z */
  pWig->Zw = pWig->Zw + dl2;
}

void GWigAx(struct gwig *pWig, double *Xvec, double *pax, double *paxpy, double poleFactor) {

  int i;
  double x, y, z;
  double kx, ky, kz, tz, kw;
  double cx, sxkx, chx, shx, shxkx;
  double cy, sy, chy, shy, sz;
  double gamma0, beta0;
  double ax, axpy;

  x = Xvec[0];
  y = Xvec[2];
  z = pWig->Zw;

  gamma0 = pWig->E0 / XMC2;
  beta0 = sqrt(1e0 - 1e0 / (gamma0 * gamma0));

  kw = 2e0 * PI / (pWig->Lw);
  ax = -(q_e/m_e/clight)*pWig->BConstant[1]*z/(gamma0*beta0);
  axpy = 0e0;

  if (pWig->NHharm && z >= pWig->zStartH && z <= pWig->zEndH) {
    if (pWig->PB0H != 0)
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0H) * poleFactor;
    else
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0) * poleFactor;
    if (!pWig->HSplitPole) {
      /* Normal Horizontal Wiggler: note that one potentially could have: kx=0 */
      for (i = 0; i < pWig->NHharm; i++) {
        pWig->HCw[i] = pWig->HCw_raw[i] * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        cx = cos(kx * x);
        chy = cosh(ky * y);
        sz = sin(kz * z + tz);

        if (pWig->cwiggler->tguGradient && i == 0)
          /* Ax = (B0/ku)*((1 + a*x)*Cosh[ku*y]*Sin[ku*z] - (a*z)*(B0*e/(me*c*k))/(2*betaGamma) */
          /* We assume kw=kz, ky=kz, kx=0 */
          ax = ax + (pWig->HCw[i]) * ((1 + x * pWig->cwiggler->tguGradient) * chy * sz -
                                      pWig->cwiggler->tguCompFactor * pWig->cwiggler->tguGradient * z * pWig->Aw / (2 * beta0 * gamma0));
        else
          ax = ax + (pWig->HCw[i]) * (kw / kz) * cx * chy * sz;

        shy = sinh(ky * y);
        if (abs(kx / kw) > GWIG_EPS) {
          sxkx = sin(kx * x) / kx;
        } else {
          sxkx = x * sinc(kx * x);
        }

        if (pWig->cwiggler->tguGradient && i == 0)
          /* axpy = B0 (x + a*x^2/2) Sin[ku z] Sinh[ku y] */
          /* We assume kw=kz, ky=kz, kx=0 */
          axpy = axpy + pWig->HCw[i] * kw * x * (1 + pWig->cwiggler->tguGradient * x / 2) * shy * sz;
        else
          axpy = axpy + pWig->HCw[i] * (kw / kz) * ky * sxkx * shy * sz;
      }
    } else {
      /* Split-pole Horizontal Wiggler: note that one potentially could have: ky=0 (caught in main routine) */
      for (i = 0; i < pWig->NHharm; i++) {
        pWig->HCw[i] = pWig->HCw_raw[i] * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        cy = cos(ky * y);
        chx = cosh(kx * x);
        sz = sin(kz * z + tz);
        ax = ax + pWig->HCw[i] * (kw / kz) * cy * chx * sz;

        shxkx = sinh(kx * x) / kx;
        axpy = axpy - pWig->HCw[i] * (kw / kz) * ky * sin(ky * y) * shxkx * sz;
      }
    }
  }

  if (pWig->NVharm && z >= pWig->zStartV && z <= pWig->zEndV) {
    if (pWig->PB0V != 0)
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0V) * poleFactor;
    else
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0) * poleFactor;
    if (!pWig->VSplitPole) {
      /* Normal Vertical Wiggler: note that one potentially could have: ky=0 */
      for (i = 0; i < pWig->NVharm; i++) {
        pWig->VCw[i] = pWig->VCw_raw[i] * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        shx = sinh(kx * x);
        sy = sin(ky * y);
        sz = sin(kz * z + tz);
        ax = ax + pWig->VCw[i] * (kw / kz) * (ky / kx) * shx * sy * sz;

        chx = cosh(kx * x);
        cy = cos(ky * y);
        axpy = axpy + pWig->VCw[i] * (kw / kz) * ipow2(ky / kx) * chx * cy * sz;
      }
    } else {
      /* Split-pole Vertical Wiggler: note that one potentially could have: kx=0 (caught in main routine) */
      for (i = 0; i < pWig->NVharm; i++) {
        pWig->VCw[i] = pWig->VCw_raw[i] * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        sz = sin(kz * z + tz);
        ax = ax - pWig->VCw[i] * (kw / kz) * (ky / kx) * sinh(ky * y) * sin(kx * x) * sz;
        axpy = axpy + pWig->VCw[i] * (kw / kz) * ipow2(ky / kx) * cosh(ky * y) * cos(kx * x) * sz;
      }
    }
  }

  *pax = ax;
  *paxpy = axpy;
}

void GWigAy(struct gwig *pWig, double *Xvec, double *pay, double *paypx, double poleFactor) {
  int i;
  double x, y, z;
  double kx, ky, kz, tz, kw;
  double cx, sx, chx, shx;
  double cy, syky, chy, shy, sz, shyky;
  double gamma0, beta0;
  double ay, aypx;

  x = Xvec[0];
  y = Xvec[2];
  z = pWig->Zw;

  gamma0 = pWig->E0 / XMC2;
  beta0 = sqrt(1e0 - 1e0 / (gamma0 * gamma0));

  kw = 2e0 * PI / (pWig->Lw);
  ay = (q_e/m_e/clight)*pWig->BConstant[0]*z/(gamma0*beta0);
  aypx = 0e0;

  if (pWig->NHharm && z >= pWig->zStartH && z <= pWig->zEndH) {
    if (pWig->PB0H != 0)
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0H) * poleFactor;
    else
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0) * poleFactor;
    if (!pWig->HSplitPole) {
      /* Normal Horizontal Wiggler: note that one potentially could have: kx=0 */
      for (i = 0; i < pWig->NHharm; i++) {
        pWig->HCw[i] = (pWig->HCw_raw[i]) * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        sx = sin(kx * x);
        shy = sinh(ky * y);
        sz = sin(kz * z + tz);

        if (pWig->cwiggler->tguGradient && i == 0)
          /* Ay = a B0 Sin[ku z] Sinh[ku y]/k^2 */
          /* We assume kw=kz, ky=kz, kx=0 */
          ay = ay - (pWig->HCw[i]) * pWig->cwiggler->tguGradient * sz * shy / kw;
        else
          ay = ay + (pWig->HCw[i]) * (kw / kz) * (kx / ky) * sx * shy * sz;

        cx = cos(kx * x);
        chy = cosh(ky * y);

        aypx = aypx + (pWig->HCw[i]) * (kw / kz) * ipow2(kx / ky) * cx * chy * sz;
      }
    } else {
      /* Split-pole Horizontal Wiggler: note that one potentially could have: ky=0 (caught in main routine) */
      for (i = 0; i < pWig->NHharm; i++) {
        pWig->HCw[i] = (pWig->HCw_raw[i]) * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        sz = sin(kz * z + tz);
        ay = ay - pWig->HCw[i] * (kw / kz) * kx / ky * sin(ky * y) * sinh(kx * x) * sz;
        aypx = aypx + pWig->HCw[i] * (kw / kz) * ipow2(kx / ky) * cos(ky * y) * cosh(kx * x) * sz;
      }
    }
  }

  if (pWig->NVharm && z >= pWig->zStartV && z <= pWig->zEndV) {
    if (pWig->PB0V != 0)
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0V) * poleFactor;
    else
      pWig->Aw = (q_e / m_e / clight) / (2e0 * PI) * (pWig->Lw) * (pWig->PB0) * poleFactor;
    if (!pWig->VSplitPole) {
      /* Normal Vertical Wiggler, could have ky=0 */
      for (i = 0; i < pWig->NVharm; i++) {
        pWig->VCw[i] = (pWig->VCw_raw[i]) * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        chx = cosh(kx * x);
        cy = cos(ky * y);
        sz = sin(kz * z + tz);
        ay = ay + (pWig->VCw[i]) * (kw / kz) * chx * cy * sz;

        shx = sinh(kx * x);
        if (abs(ky / kw) > GWIG_EPS)
          syky = sin(ky * y) / ky;
        else
          syky = y * sinc(ky * y);
        aypx = aypx + (pWig->VCw[i]) * (kw / kz) * kx * shx * syky * sz;
      }
    } else {
      /* Split-pole Vertical Wiggler, could have kx=0 */
      for (i = 0; i < pWig->NVharm; i++) {
        pWig->VCw[i] = (pWig->VCw_raw[i]) * (pWig->Aw) / (gamma0 * beta0);
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        sz = sin(kz * z + tz);
        ay = ay + (pWig->VCw[i]) * (kw / kz) * cosh(ky * y) * cos(kx * x) * sz;
        shyky = sinh(ky * y) / ky;
        aypx = aypx - (pWig->VCw[i]) * (kw / kz) * kx * shyky * sin(kx * x) * sz;
      }
    }
  }

  *pay = ay;
  *paypx = aypx;
}

double sinc(double x) {
  double x2, result;
  /* Expand sinc(x) = sin(x)/x to x^8 */
  x2 = x * x;
  result = 1e0 - x2 / 6e0 * (1e0 - x2 / 20e0 * (1e0 - x2 / 42e0 * (1e0 - x2 / 72e0)));
  return result;
}

void GWigB(struct gwig *pWig, double *Xvec, double *B, double poleFactor)
/* Compute magnetic field at particle location.
 * Added by M. Borland, August 2007.
 */
{
  int i;
  double x, y, z;
  double kx, ky, kz, tz;
  /* double kw; */
  double cx, sx, chx, shx;
  double cy, sy, chy, shy;
  double cz;
  double B0;

  /* poleFactor = 1; */

  x = Xvec[0];
  y = Xvec[2];
  z = pWig->Zw;

  /* kw   = 2e0*PI/(pWig->Lw); */

  B[0] = pWig->BConstant[0];
  B[1] = pWig->BConstant[1];

  if (pWig->NHharm && z >= pWig->zStartH && z <= pWig->zEndH) {
    B0 = (pWig->PB0H ? pWig->PB0H : pWig->PB0) * poleFactor;
    if (!pWig->HSplitPole) {
      /* Normal Horizontal Wiggler: note that one potentially could have: kx=0 */
      for (i = 0; i < pWig->NHharm; i++) {
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        sx = sin(kx * x);
        cx = cos(kx * x);
        chy = cosh(ky * y);
        shy = sinh(ky * y);
        cz = cos(kz * z + tz);

        /* Accumulate field values in user-supplied array (Bx, By) */
        if (i == 0 && pWig->cwiggler->tguGradient) {
          /* Bx = (a*B0*Cos(ku*z)*Sinh(ku*y))/ku */
          /* By = (a*Power(B0,2)*CF*e)/(2.*betaGamma*c*Power(ku,2)*me) + B0*(1 + a*x)*Cos(ku*z)*Cosh(ku*y) */
          shy = sinh(ky * y);
          B[0] += B0 * pWig->HCw_raw[i] * pWig->cwiggler->tguGradient * cz * shy / kz;
          B[1] -= B0 * pWig->HCw_raw[i] * (chy * cz * (1 + pWig->cwiggler->tguGradient * x) + pWig->cwiggler->tguCompFactor * pWig->cwiggler->tguGradient * (B0 * pWig->HCw_raw[i] * q_e / (2 * m_e * clight * pWig->Po * sqr(kz))));

        } else {
          B[0] += B0 * pWig->HCw_raw[i] * kx / ky * sx * shy * cz;
          B[1] -= B0 * pWig->HCw_raw[i] * cx * chy * cz;
        }
      }
    } else {
      /* Split-pole Horizontal Wiggler: note that one potentially could have: ky=0 (caught in main routine) */
      for (i = 0; i < pWig->NHharm; i++) {
        kx = pWig->Hkx[i];
        ky = pWig->Hky[i];
        kz = pWig->Hkz[i];
        tz = pWig->Htz[i];

        shx = sinh(kx * x);
        chx = cosh(kx * x);
        cy = cos(ky * y);
        sy = sin(ky * y);
        cz = cos(kz * z + tz);

        B[0] -= B0 * pWig->HCw_raw[i] * kx / ky * shx * sy * cz;
        B[1] -= B0 * pWig->HCw_raw[i] * chx * cy * cz;
      }
    }
  }

  if (pWig->NVharm && z >= pWig->zStartV && z <= pWig->zEndV) {
    B0 = (pWig->PB0V ? pWig->PB0V : pWig->PB0) * poleFactor;
    if (!pWig->VSplitPole) {
      /* Normal Vertical Wiggler: note that one potentially could have: ky=0 */
      for (i = 0; i < pWig->NVharm; i++) {
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        shx = sinh(kx * x);
        chx = cosh(kx * x);
        sy = sin(ky * y);
        cy = cos(ky * y);
        cz = cos(kz * z + tz);

        /* Accumulate field values in user-supplied array (Bx, By) */
        B[0] += B0 * pWig->VCw_raw[i] * chx * cy * cz;
        B[1] -= B0 * pWig->VCw_raw[i] * ky / kx * shx * sy * cz;
      }
    } else {
      /* Split-pole Vertical Wiggler: note that one potentially could have: kx=0 (caught in main routine) */
      for (i = 0; i < pWig->NVharm; i++) {
        kx = pWig->Vkx[i];
        ky = pWig->Vky[i];
        kz = pWig->Vkz[i];
        tz = pWig->Vtz[i];

        sx = sin(kx * x);
        cx = cos(kx * x);
        shy = sinh(ky * y);
        chy = cosh(ky * y);
        cz = cos(kz * z + tz);

        /* Accumulate field values in user-supplied array (Bx, By) */
        B[0] += B0 * pWig->VCw_raw[i] * cx * chy * cz;
        B[1] += B0 * pWig->VCw_raw[i] * ky / kx * sx * shy * cz;
      }
    }
  }
}

void GWigRadiationKicks(struct gwig *pWig, double *X, double *Bxy, double dl, double *sigmaDelta2)
/* Apply kicks for synchrotron radiation.
 * Added by M. Borland, August 2007.
 */
{
  double Po, irho2, H, dFactor;
  double B2;
  double dDelta;

  /* B^2 in T^2 */
  if ((B2 = sqr(Bxy[0]) + sqr(Bxy[1])) == 0)
    return;

  /* Beam rigidity in T*m */
  H = (Po = pWig->Po) / 586.679074042074490;

  /* 1/rho^2 */
  irho2 = B2 / sqr(H);

  /* (1+delta)^2 */
  dFactor = sqr(1 + X[4]);

  if (pWig->sr) {
    /* Classical radiation loss */
    dDelta = -pWig->srCoef * dFactor * irho2 * dl;
    X[4] += dDelta;
    X[1] *= (1 + dDelta);
    X[3] *= (1 + dDelta);
  }

  if (pWig->isr) {
    /* Incoherent synchrotron radiation or quantum excitation */
    dDelta = pWig->isrCoef * dFactor * pow(irho2, 0.75) * sqrt(dl) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
    X[4] += dDelta;
    X[1] *= (1 + dDelta);
    X[3] *= (1 + dDelta);
  }

  if (sigmaDelta2) {
    if (dl < 0)
      bombElegant("dl<0 in GWigRadiationKicks", NULL);
    *sigmaDelta2 += sqr(pWig->isrCoef * dFactor) * pow(irho2, 1.5) * dl;
    /* printf("ZWig = %le, B = %le, sigmaDelta2 = %le\n", 
       pWig->Zw, sqrt(B2), *sigmaDelta2);
       */
  }
}

#ifdef HAVE_GPU
/* How many ISR random numbers a particle uses, for the GPU's table. GWigRadiationKicks skips those
 * where the field is exactly zero, which depends on z and the pole factors but not on x and y. */
long GWigCountRadiationKicks(struct gwig *pWig) {
  const double x1 = 1.3512071919596576340476878089715e0;
  const double x0 = -1.7024143839193152680953756179429e0;
  struct gwig wig = *pWig;
  double X[6] = {0, 0, 0, 0, 0, 0}, B[2], dl;
  long i, Nstep, count = 0;

  Nstep = wig.PN * wig.Nw;
  dl = wig.Lw / wig.PN;
  wig.Zw = 0;
  GWigB(&wig, X, B, wig.cwiggler->poleFactor[0]);
  if (sqr(B[0]) + sqr(B[1]) != 0)
    count++;
  for (i = 1; i <= Nstep; i++) {
    double pf = GWigPoleFactor(&wig, ((i - 1) * wig.cwiggler->length) / Nstep);
    if (wig.Pmethod == 2) {
      wig.Zw += 0.5 * dl;
      wig.Zw += 0.5 * dl;
    } else {
      wig.Zw += 0.5 * (x1 * dl);
      wig.Zw += 0.5 * (x1 * dl);
      wig.Zw += 0.5 * (x0 * dl);
      wig.Zw += 0.5 * (x0 * dl);
      wig.Zw += 0.5 * (x1 * dl);
      wig.Zw += 0.5 * (x1 * dl);
    }
    GWigB(&wig, X, B, pf);
    if (sqr(B[0]) + sqr(B[1]) != 0)
      count++;
  }
  return count;
}
#endif

void GWigAddToFieldOutput(CWIGGLER *cwiggler, double Z, double *X, double *B, double pf) {
  SDDS_DATASET *SDDSout;

  SDDSout = cwiggler->SDDSFieldOutput;

  if (cwiggler->fieldOutputRow == cwiggler->fieldOutputRows) {
    if (!SDDS_LengthenTable(SDDSout, 1000)) {
      printf("*** Error: problem lengthening table size for CWIGGLER field output.  Table was %ld rows\n",
             cwiggler->fieldOutputRows);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    cwiggler->fieldOutputRows += 1000;
  }
  if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, cwiggler->fieldOutputRow,
                         "x", (float)X[0], "y", (float)X[2], "z", (float)Z,
                         "px", (float)X[1], "py", (float)X[2],
                         "Bx", (float)B[0], "By", (float)B[1],
                         "PoleFactor", (float)pf,
                         NULL)) {
    printf("*** Error: problem setting row values for CWIGGLER field output.\n");
    SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  cwiggler->fieldOutputRow++;
  return;
}

double GWigPoleFactor(struct gwig *pWig, double z) {
  if (z < pWig->z3) {
    if (z < pWig->z1)
      return pWig->cwiggler->poleFactor[0];
    if (z < pWig->z2)
      return pWig->cwiggler->poleFactor[1];
    return pWig->cwiggler->poleFactor[2];
  }
  if (z < pWig->z4)
    return 1;
  if (z < pWig->z5)
    return pWig->cwiggler->poleFactor[2];
  if (z < pWig->z6)
    return pWig->cwiggler->poleFactor[1];
  return pWig->cwiggler->poleFactor[0];
}

