/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: apple.c
 *       This program is an application of BESSY method to second order.
 *
 *       "Symplectic tracking and compensation of dynamic field integrals in complex undulator structures"
 *       by Johannes Bahrdt and Godehard Wuestefeld, PRSTAB 14,040703 (2011)
 *
 * Aimin Xiao, 2011
 */

#include "mdb.h"
#include "track.h"

void f_ijk(double *X, long End_Pole, double step,
           double factor, APPLE *apple);

void APPLE_Track(double **coord, int64_t np, double pCentral, APPLE *apple) {
  long istep, nleft, iHarm;
  int64_t ipart;
  MALIGN malign;
  double X[6], denom, p, brho, eomc;
  double intBx, intBy, tmpx, tmpy, tmpg;

  if (!apple->initialized)
    InitializeAPPLE(apple->Input, apple);

  if (apple->tilt)
    rotateBeamCoordinatesForMisalignment(coord, np, apple->tilt);
  if (apple->dx || apple->dy || apple->dz) {
    memset(&malign, 0, sizeof(malign));
    malign.dx = -apple->dx;
    malign.dy = -apple->dy;
    malign.dz = apple->dz;
    malign.startPID = malign.endPID = -1;
    offset_beam(coord, np, &malign, pCentral);
  }

  eomc = -particleCharge / particleMass / c_mks;
  for (ipart = 0; ipart < np; ipart++) {
    p = (1. + coord[ipart][5]) * pCentral;
    brho = p / eomc;
    if (apple->shimOn) {
      intBx = intBy = 0.;
      for (iHarm = 0; iHarm < apple->ShimHarm; iHarm++) {
        tmpx = (double)iHarm * apple->kx_shim * coord[ipart][0];
        tmpy = (double)iHarm * apple->kx_shim * coord[ipart][2];
        tmpg = -(double)iHarm * apple->kx_shim * apple->dgap / 2.;
        intBx += (apple->Ci_shim[iHarm] * sin(tmpx) * cosh(tmpy) + apple->Si_shim[iHarm] * cos(tmpx) * sinh(tmpy)) * exp(tmpg);
        intBy += (-apple->Ci_shim[iHarm] * cos(tmpx) * sinh(tmpy) + apple->Si_shim[iHarm] * sin(tmpx) * cosh(tmpy)) * exp(tmpg);
      }
      coord[ipart][1] += intBy / brho * apple->shimScale;
      coord[ipart][3] -= intBx / brho * apple->shimScale;
    }
    /* convert from elegant coordinates (x, x', y, y', s, delta) 
     * to Canonical coordinates (x, px, y, py, s, delta) 
     * d =  sqrt(1+sqr(xp)+sqr(yp))
     * p = (1+delta)*pCentral
     * px = xp/d/p
     * py = yp/d/p
     */
    X[0] = coord[ipart][0];
    X[2] = coord[ipart][2];
    X[1] = coord[ipart][1] / (denom = sqrt(1 + sqr(coord[ipart][1]) + sqr(coord[ipart][3])));
    X[3] = coord[ipart][3] / denom;
    X[4] = coord[ipart][4];
    X[5] = coord[ipart][5];

    /* tracking through wiggler */
    if (apple->End_Pole) {
      if (apple->periods >= 2) {
        nleft = apple->periods - 2;
        f_ijk(X, 1, 0.5, 0.25 / brho, apple);
        f_ijk(X, 1, 0.5, -0.75 / brho, apple);
      } else {
        apple->End_Pole = 0;
        nleft = apple->periods;
        printWarningForTracking("PERIODS<2, setting END_POLE=0", NULL);
      }
    } else {
      nleft = apple->periods;
    }

    for (istep = 0; istep < apple->periods; istep++) {
      nleft -= apple->step;
      if (nleft < 0)
        break;
      f_ijk(X, 0, (double)apple->step, 1. / brho, apple); /* N full periods */
    }
    if (nleft) {
      f_ijk(X, 0, (double)(nleft + apple->step), 1. / brho, apple); /* All left over periods */
    }

    if (apple->End_Pole) {
      f_ijk(X, 1, 0.5, 0.75 / brho, apple);  /* half period */
      f_ijk(X, 1, 0.5, -0.25 / brho, apple); /* half period */
    }

    /* convert from Canonical coordinates (x, px, y, py, s, delta)
     * to elegant coordinates (x, x', y, y', s, delta)
     * d =  sqrt(1-sqr(px)-sqr(py))
     * xp = px/d
     * yp = py/d
     */
    coord[ipart][0] = X[0];
    coord[ipart][2] = X[2];
    coord[ipart][1] = X[1] / (denom = sqrt(1 - sqr(X[1]) - sqr(X[3])));
    coord[ipart][3] = X[3] / denom;
    coord[ipart][4] = X[4];
    coord[ipart][5] = X[5];

    if (apple->shimOn) {
      intBx = intBy = 0.;
      for (iHarm = 0; iHarm < apple->ShimHarm; iHarm++) {
        tmpx = (double)iHarm * apple->kx_shim * coord[ipart][0];
        tmpy = (double)iHarm * apple->kx_shim * coord[ipart][2];
        tmpg = -(double)iHarm * apple->kx_shim * apple->dgap / 2.;
        intBx += (apple->Ci_shim[iHarm] * sin(tmpx) * cosh(tmpy) + apple->Si_shim[iHarm] * cos(tmpx) * sinh(tmpy)) * exp(tmpg);
        intBy += (-apple->Ci_shim[iHarm] * cos(tmpx) * sinh(tmpy) + apple->Si_shim[iHarm] * sin(tmpx) * cosh(tmpy)) * exp(tmpg);
      }
      coord[ipart][1] += intBy / brho * apple->shimScale;
      coord[ipart][3] -= intBx / brho * apple->shimScale;
    }
    /*    fprintf(stdout,"ip=%ld,x=%g,xp=%g\n", ipart,coord[ipart][0],coord[ipart][1]); */
  }
  if (apple->dx || apple->dy || apple->dz) {
    memset(&malign, 0, sizeof(malign));
    malign.dx = apple->dx;
    malign.dy = apple->dy;
    malign.dz = -apple->dz;
    offset_beam(coord, np, &malign, pCentral);
  }
  if (apple->tilt)
    rotateBeamCoordinatesForMisalignment(coord, np, -apple->tilt);

  return;
}

void f_ijk(double *X, long End_Pole, double step,
           double factor, APPLE *apple) {
  double Ax, Ay;
  long j, i;
  double xpp, xmm, kx, ky;
  double Cm, Cp, Sm, Sp, Em, Ep;
  double Cx, Dx, Cy, Dy;
  double Cxx, Dxx, Cyx, Dyx;
  double Cxy, Dxy, Cyy, Dyy;
  double ccp, csp, ccm, csm;
  double scp, ssp, scm, ssm;
  double f101, f101x, f101y;
  double f011, f011x, f011y;
  double f002, f002x, f002y, pn;
  //double f003;
  double f003x, f003y;
  double Cxxx, Cxxy, Cxyx, Cxyy;
  double Dxxx, Dxxy, Dxyx, Dxyy;
  double Cyxx, Cyxy, Cyyx, Cyyy;
  double Dyxx, Dyxy, Dyyx, Dyyy;
  double Ax1, Ax2, Ax3;
  double Ay1, Ay2, Ay3;
  double Ax1x, Ax2x, Ax3x;
  double Ay1x, Ay2x, Ay3x;
  double Ax1y, Ax2y, Ax3y;
  double Ay1y, Ay2y, Ay3y;

  Dx = Dy = 0;

  xpp = X[0] + apple->x0;
  xmm = X[0] - apple->x0;

  f101 = f101x = f101y = 0;
  f011 = f011x = f011y = 0;
  f002 = f002x = f002y = 0;
  //f003 = 0;
  f003x = f003y = 0;
  pn = 1.;
  for (j = 0; j < apple->NzHarm; j++) {
    Cx = Dx = Cy = Dy = 0;
    Cxx = Dxx = Cyx = Dyx = 0;
    Cxy = Dxy = Cyy = Dyy = 0;
    Cxxx = Cxxy = Cxyx = Cxyy = 0;
    Dxxx = Dxxy = Dxyx = Dxyy = 0;
    Cyxx = Cyxy = Cyyx = Cyyy = 0;
    Dyxx = Dyxy = Dyyx = Dyyy = 0;
    for (i = 0; i < apple->NxHarm; i++) {
      kx = apple->kx[i][j];
      ky = apple->ky[i][j];
      Cm = cos(kx * xmm);
      Cp = cos(kx * xpp);
      Sm = sin(kx * xmm);
      Sp = sin(kx * xpp);
      Em = exp(-ky * X[2]);
      Ep = exp(ky * X[2]);
      ccp = Cm * Ep * apple->C1 + Cp * Ep * apple->C2 + Cp * Em * apple->C3 + Cm * Em * apple->C4;
      ccm = Cm * Ep * apple->C1 + Cp * Ep * apple->C2 - Cp * Em * apple->C3 - Cm * Em * apple->C4;
      csp = Cm * Ep * apple->S1 + Cp * Ep * apple->S2 + Cp * Em * apple->S3 + Cm * Em * apple->S4;
      csm = Cm * Ep * apple->S1 + Cp * Ep * apple->S2 - Cp * Em * apple->S3 - Cm * Em * apple->S4;
      scp = Sm * Ep * apple->C1 + Sp * Ep * apple->C2 + Sp * Em * apple->C3 + Sm * Em * apple->C4;
      scm = Sm * Ep * apple->C1 + Sp * Ep * apple->C2 - Sp * Em * apple->C3 - Sm * Em * apple->C4;
      ssp = Sm * Ep * apple->S1 + Sp * Ep * apple->S2 + Sp * Em * apple->S3 + Sm * Em * apple->S4;
      ssm = Sm * Ep * apple->S1 + Sp * Ep * apple->S2 - Sp * Em * apple->S3 - Sm * Em * apple->S4;

      Cx += apple->CoZ[i][j] * ccp;
      Dx += apple->CoZ[i][j] * csp;
      Cy += apple->CxXoYZ[i][j] * scm;
      Dy += apple->CxXoYZ[i][j] * ssm;

      Cxx += -apple->CxXoZ[i][j] * scp;
      Dxx += -apple->CxXoZ[i][j] * ssp;
      Cyx += apple->CxX2oYZ[i][j] * ccm;
      Dyx += apple->CxX2oYZ[i][j] * csm;

      Cxy += apple->CxYoZ[i][j] * ccm;
      Dxy += apple->CxYoZ[i][j] * csm;
      Cyy += apple->CxXoZ[i][j] * scp;
      Dyy += apple->CxXoZ[i][j] * ssp;
      if (End_Pole && apple->order == 3) {
        Cxxx += -apple->CxX2oZ[i][j] * ccp;
        Cxxy += -apple->CxXYoZ[i][j] * scm;
        Dxxx += -apple->CxX2oZ[i][j] * csp;
        Dxxy += -apple->CxXYoZ[i][j] * ssm;
        Cyxx += -apple->CxX3oYZ[i][j] * scm;
        Cyxy += apple->CxX2oZ[i][j] * ccp;
        Dyxx += -apple->CxX3oYZ[i][j] * ssm;
        Dyxy += apple->CxX2oZ[i][j] * csp;

        Cxyx += -apple->CxXYoZ[i][j] * scm;
        Cxyy += apple->CxY2oZ[i][j] * ccp;
        Dxyx += -apple->CxXYoZ[i][j] * ssm;
        Dxyy += apple->CxY2oZ[i][j] * csp;
        Cyyx += apple->CxX2oZ[i][j] * ccp;
        Cyyy += apple->CxXYoZ[i][j] * scm;
        Dyyx += apple->CxX2oZ[i][j] * csp;
        Dyyy += apple->CxXYoZ[i][j] * ssm;
      }
    }
    f002 += Cx * Cx + Cy * Cy + Dx * Dx + Dy * Dy;
    f002x += Cx * Cxx + Cy * Cyx + Dx * Dxx + Dy * Dyx;
    f002y += Cx * Cxy + Cy * Cyy + Dx * Dxy + Dy * Dyy;
    if (End_Pole) {
      f101 += Cx / apple->kz[j];
      f101x += Cxx / apple->kz[j];
      f101y += Cxy / apple->kz[j];
      f011 += Cy / apple->kz[j];
      f011x += Cyx / apple->kz[j];
      f011y += Cyy / apple->kz[j];
      if (apple->order == 3) {
        Ax1 = Cx * Cxx + Cy * Cyx + Dx * Dxx + Dy * Dyx;
        Ay1 = Cx * Cxy + Cy * Cyy + Dx * Dxy + Dy * Dyy;
        Ax2 = Dx * Dxx + Dy * Dyx - Cx * Cxx - Cy * Cyx;
        Ay2 = Dx * Dxy + Dy * Dyy - Cx * Cxy - Cy * Cyy;
        Ax3 = Cxx * Dx + Cx * Dxx + Cyx * Dy + Cy * Dyx;
        Ay3 = Cxy * Dx + Cx * Dxy + Cyy * Dy + Cy * Dyy;
        Ax1x = Cxx * Cxx + Cyx * Cyx + Dxx * Dxx + Dyx * Dyx + Cx * Cxxx + Cy * Cyxx + Dx * Dxxx + Dy * Dyxx;
        Ax1y = Cxy * Cxx + Cyy * Cyx + Dxy * Dxx + Dyy * Dyx + Cx * Cxxy + Cy * Cyxy + Dx * Dxxy + Dy * Dyxy;
        Ay1x = Cxx * Cxy + Cyx * Cyy + Dxx * Dxy + Dyx * Dyy + Cx * Cxyx + Cy * Cyyx + Dx * Dxyx + Dy * Dyyx;
        Ay1y = Cxy * Cxy + Cyy * Cyy + Dxy * Dxy + Dyy * Dyy + Cx * Cxyy + Cy * Cyyy + Dx * Dxyy + Dy * Dyyy;
        Ax2x = Dxx * Dxx + Dyx * Dyx - Cxx * Cxx - Cyx * Cyx + Dx * Dxxx + Dy * Dyxx - Cx * Cxxx - Cy * Cyxx;
        Ax2y = Dxy * Dxx + Dyy * Dyx - Cxy * Cxx - Cyy * Cyx + Dx * Dxxy + Dy * Dyxy - Cx * Cxxy - Cy * Cyxy;
        Ay2x = Dxx * Dxy + Dyx * Dyy - Cxx * Cxy - Cyx * Cyy + Dx * Dxyx + Dy * Dyyx - Cx * Cxyx - Cy * Cyyx;
        Ay2y = Dxy * Dxy + Dyy * Dyy - Cxy * Cxy - Cyy * Cyy + Dx * Dxyy + Dy * Dyyy - Cx * Cxyy - Cy * Cyyy;
        Ax3x = Cxxx * Dx + Cxx * Dxx + Cyxx * Dy + Cyx * Dyx + Cxx * Dxx + Cx * Dxxx + Cyx * Dyx + Cy * Dyxx;
        Ax3y = Cxxy * Dx + Cxy * Dxx + Cyxy * Dy + Cyy * Dyx + Cxx * Dxy + Cx * Dxxy + Cyx * Dyy + Cy * Dyxy;
        Ay3x = Cxyx * Dx + Cxx * Dxy + Cyyx * Dy + Cyx * Dyy + Cxy * Dxx + Cx * Dxyx + Cyy * Dyx + Cy * Dyyx;
        Ay3y = Cxyy * Dx + Cxy * Dxy + Cyyy * Dy + Cyy * Dyy + Cxy * Dxy + Cx * Dxyy + Cyy * Dyy + Cy * Dyyy;

        //f003 += 0.5 / sqr(apple->kz[j]) * (Cx * (4. * Ax3 / 3. + apple->lz * apple->kz[j] * Ax1) - 2. * Dx * (Ax1 - Ax2 / 3.));
        //f003 += 0.5 / sqr(apple->kz[j]) * (Cy * (4. * Ay3 / 3. + apple->lz * apple->kz[j] * Ay1) - 2. * Dy * (Ay1 - Ay2 / 3.));
        f003x += 0.5 / sqr(apple->kz[j]) * (Cxx * (4. * Ax3 / 3. + apple->lz * apple->kz[j] * Ax1) - 2. * Dxx * (Ax1 - Ax2 / 3.));
        f003x += 0.5 / sqr(apple->kz[j]) * (Cx * (4. * Ax3x / 3. + apple->lz * apple->kz[j] * Ax1x) - 2. * Dx * (Ax1x - Ax2x / 3.));
        f003x += 0.5 / sqr(apple->kz[j]) * (Cyx * (4. * Ay3 / 3. + apple->lz * apple->kz[j] * Ay1) - 2. * Dyx * (Ay1 - Ay2 / 3.));
        f003x += 0.5 / sqr(apple->kz[j]) * (Cy * (4. * Ay3x / 3. + apple->lz * apple->kz[j] * Ay1x) - 2. * Dy * (Ay1x - Ay2x / 3.));
        f003y += 0.5 / sqr(apple->kz[j]) * (Cxy * (4. * Ax3 / 3. + apple->lz * apple->kz[j] * Ax1) - 2. * Dxy * (Ax1 - Ax2 / 3.));
        f003y += 0.5 / sqr(apple->kz[j]) * (Cx * (4. * Ax3y / 3. + apple->lz * apple->kz[j] * Ax1y) - 2. * Dx * (Ax1y - Ax2y / 3.));
        f003y += 0.5 / sqr(apple->kz[j]) * (Cyy * (4. * Ay3 / 3. + apple->lz * apple->kz[j] * Ay1) - 2. * Dyy * (Ay1 - Ay2 / 3.));
        f003y += 0.5 / sqr(apple->kz[j]) * (Cy * (4. * Ay3y / 3. + apple->lz * apple->kz[j] * Ay1y) - 2. * Dy * (Ay1y - Ay2y / 3.));
      }
    }
    /*    printf("j=%ld, f002=%g , f002=%g, f002=%g\n", j, f002, f002x, f002y); */
  }
  f002 = -step * apple->lz / 4. * sqr(factor) * f002;
  f002x = -step * apple->lz / 2. * sqr(factor) * f002x;
  f002y = -step * apple->lz / 2. * sqr(factor) * f002y;
  if (End_Pole) {
    f101 = 2 * factor * f101;
    f101x = 2 * factor * f101x;
    f101y = 2 * factor * f101y;
    f011 = 2 * factor * f011;
    f011x = 2 * factor * f011x;
    f011y = 2 * factor * f011y;
    pn = (1. - f101x) * (1. - f011y) - f101y * f011x;
    //f003 *= pow(factor, 3.0);
    f003x *= pow(factor, 3.0);
    f003y *= pow(factor, 3.0);
  }

  /* Change momenta to the canonical momenta at the entrance */
  Ax = Dx * factor;
  Ay = Dy * factor;
  X[1] = X[1] + Ax;
  X[3] = X[3] + Ay;

  if (End_Pole) {
    X[1] = ((1 - f011y) * (X[1] + f002x + f003x) + f011x * (X[3] + f002y + f003y)) / pn;
    X[3] = ((1 - f101x) * (X[3] + f002y + f003y) + f101y * (X[1] + f002x + f003x)) / pn;
    X[0] = X[0] + X[1] * step * apple->lz - f101;
    X[2] = X[2] + X[3] * step * apple->lz - f011;
  } else {
    X[1] = X[1] + f002x;
    X[3] = X[3] + f002y;
    X[0] = X[0] + X[1] * step * apple->lz;
    X[2] = X[2] + X[3] * step * apple->lz;
  }
  X[4] += step * apple->lz;

  /* Change momenta from the canonical momenta back to the lab coordinate system at the exit */
  xpp = X[0] + apple->x0;
  xmm = X[0] - apple->x0;
  for (j = 0; j < apple->NzHarm; j++) {
    Dx = Dy = 0;
    for (i = 0; i < apple->NxHarm; i++) {
      kx = apple->kx[i][j];
      ky = apple->ky[i][j];
      Cm = cos(kx * xmm);
      Cp = cos(kx * xpp);
      Sm = sin(kx * xmm);
      Sp = sin(kx * xpp);
      Em = exp(-ky * X[2]);
      Ep = exp(ky * X[2]);
      csp = Cm * Ep * apple->S1 + Cp * Ep * apple->S2 + Cp * Em * apple->S3 + Cm * Em * apple->S4;
      ssm = Sm * Ep * apple->S1 + Sp * Ep * apple->S2 - Sp * Em * apple->S3 - Sm * Em * apple->S4;
      Dx += apple->CoZ[i][j] * csp;
      Dy += apple->CxXoYZ[i][j] * ssm;
    }
  }
  Ax = Dx * factor;
  Ay = Dy * factor;
  X[1] = X[1] - Ax;
  X[3] = X[3] - Ay;

  return;
}

void InitializeAPPLE(char *file, APPLE *apple) {
  SDDS_DATASET SDDSin, SDDSshim;
  double *Cmn, *kx, *ky, *kz, kw;
  double bo4, BxAmp, ByAmp;
  long row, rows, j;
  int32_t *xHarm, *zHarm, *sHarm;
  double kx_shim, *c_shim, *s_shim;

  if (apple->initialized)
    return;

  c_shim = s_shim = NULL;
  Cmn = kx = ky = kz = NULL;
  xHarm = zHarm = sHarm = NULL;

  if (apple->shimOn) {
    if (!apple->shimInput) {
      printf("Error: No shim input file provided.\n");
      exitElegant(1);
    }
    if (!SDDS_InitializeInput(&SDDSshim, apple->shimInput) || !SDDS_ReadPage(&SDDSshim)) {
      printf("Error: problem initializing file %s\n", apple->shimInput);
      exitElegant(1);
    }
    if (!(rows = SDDS_RowCount(&SDDSshim))) {
      printf("Error: no rows in file %s\n", apple->shimInput);
      exitElegant(1);
    }
    if (!SDDS_GetParameters(&SDDSshim, "kx0", &kx_shim, NULL)) {
      printf("Error: problem reading file kx0 %s\n", apple->shimInput);
      exitElegant(1);
    }
    if (!(sHarm = SDDS_GetColumnInLong(&SDDSshim, "i")) ||
        !(c_shim = SDDS_GetColumnInDoubles(&SDDSshim, "Ci")) ||
        !(s_shim = SDDS_GetColumnInDoubles(&SDDSshim, "Si"))) {
      printf("Error: problem reading file column data %s\n", apple->shimInput);
      exitElegant(1);
    }
    if (!SDDS_Terminate(&SDDSshim))
      printWarningForTracking("Problem terminating SHIM_INPUT file", NULL);

    apple->kx_shim = kx_shim;
    for (row = 0; row < rows; row++) {
      if (sHarm[row] > apple->ShimHarm)
        apple->ShimHarm = sHarm[row];
    }
    apple->ShimHarm = apple->ShimHarm + 1; /* include harm = 0 term */
    apple->Ci_shim = calloc(sizeof(*apple->Ci_shim), apple->ShimHarm);
    apple->Si_shim = calloc(sizeof(*apple->Si_shim), apple->ShimHarm);
    for (row = 0; row < rows; row++) {
      j = (long)(sHarm[row]);
      apple->Ci_shim[j] = c_shim[row];
      apple->Si_shim[j] = s_shim[row];
    }
  }

  if (!file) {
    apple->drift = 1;
    return;
  }

  if (!SDDS_InitializeInput(&SDDSin, file) || !SDDS_ReadPage(&SDDSin)) {
    printf("Error: problem initializing file %s\n", file);
    exitElegant(1);
  }
  rows = 0;
  if (!(rows = SDDS_RowCount(&SDDSin))) {
    printf("Error: no rows in file %s\n", file);
    exitElegant(1);
  }
  if (!(Cmn = SDDS_GetColumnInDoubles(&SDDSin, "Cmn")) ||
      !(kx = SDDS_GetColumnInDoubles(&SDDSin, "KxOverKw")) ||
      !(ky = SDDS_GetColumnInDoubles(&SDDSin, "KyOverKw")) ||
      !(kz = SDDS_GetColumnInDoubles(&SDDSin, "KzOverKw")) ||
      !(xHarm = SDDS_GetColumnInLong(&SDDSin, "xHarm")) ||
      !(zHarm = SDDS_GetColumnInLong(&SDDSin, "zHarm"))) {
    printf("Error: problem reading file %s\n", file);
    printf("Check for existence of Cmn, KxOverKw, KyOverKw, KzOverKw, xHarm and zHarm\n");
    exitElegant(1);
  }
  if (!SDDS_Terminate(&SDDSin))
    printWarningForTracking("Problem terminating INPUT_FILE file", NULL);

  /* Count total longitudinal and transverse harmonics */
  for (row = 0; row < rows; row++) {
    if (xHarm[row] > apple->NxHarm)
      apple->NxHarm = xHarm[row];
    if (zHarm[row] > apple->NzHarm)
      apple->NzHarm = zHarm[row];
  }
  apple->NzHarm = (long)((apple->NzHarm + 1.1) / 2.);
  apple->NxHarm = apple->NxHarm + 1; /* include xharm = 0 term */

  apple->Cij = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->kx = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->ky = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->kz = calloc(sizeof(*apple->kz), apple->NzHarm);
  apple->CoZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxXoZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxYoZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxXoYZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxX2oYZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxX2oZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxX3oYZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxXYoZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);
  apple->CxY2oZ = (double **)czarray_2d(sizeof(double), apple->NxHarm, apple->NzHarm);

  bo4 = apple->BMax / 4.;
  apple->lz = apple->length / (double)apple->periods;
  kw = PIx2 / apple->lz;
  for (row = 0; row < rows; row++) {
    if (kz[row] <= 0) {
      printf("*** Error: Problem with KzOverKw in %s: value %e is not positive\n", file, kz[row]);
      exitElegant(1);
    }
    if (ky[row] < kz[row]) {
      printf("*** Error: Problem with KyOverKw<KzOverKw in %s\n", file);
      exitElegant(1);
    }
    if (fabs(sqr(kx[row]) + sqr(kz[row]) - sqr(ky[row])) > 1e-6) {
      printf("*** Error: KyOverKw == sqrt(KxOverKw^2+KzOverKw^2) not satisfied to sufficient accuracy (1e-6) in %s\n", file);
      exitElegant(1);
    }
    j = (long)((zHarm[row] + 1.1) / 2.) - 1;
    kx[row] *= kw;
    ky[row] *= kw;
    kz[row] *= kw;
    apple->Cij[xHarm[row]][j] = -Cmn[row] * bo4 * exp(-ky[row] * apple->dgap / 2.);
    apple->BPeak[1] += apple->Cij[xHarm[row]][j];
    apple->Cij[xHarm[row]][j] = apple->Cij[xHarm[row]][j] / cos(kx[row] * apple->x0);
    apple->BPeak[0] += apple->Cij[xHarm[row]][j] * sin(kx[row] * apple->x0) * kx[row] / ky[row];
    apple->kx[xHarm[row]][j] = kx[row];
    apple->ky[xHarm[row]][j] = ky[row];
    apple->kz[j] = kz[row];
    apple->CoZ[xHarm[row]][j] = apple->Cij[xHarm[row]][j] / kz[row];
    apple->CxXoZ[xHarm[row]][j] = apple->CoZ[xHarm[row]][j] * kx[row];
    apple->CxYoZ[xHarm[row]][j] = apple->CoZ[xHarm[row]][j] * ky[row];
    apple->CxXoYZ[xHarm[row]][j] = apple->CxXoZ[xHarm[row]][j] / ky[row];
    apple->CxX2oYZ[xHarm[row]][j] = apple->CxXoYZ[xHarm[row]][j] * kx[row];
    apple->CxX2oZ[xHarm[row]][j] = apple->CxXoZ[xHarm[row]][j] * kx[row];
    apple->CxX3oYZ[xHarm[row]][j] = apple->CxX2oYZ[xHarm[row]][j] * kx[row];
    apple->CxXYoZ[xHarm[row]][j] = apple->CxXoZ[xHarm[row]][j] * ky[row];
    apple->CxY2oZ[xHarm[row]][j] = apple->CxYoZ[xHarm[row]][j] * ky[row];
    /*printf("i=%d, cmn=%g, kx=%g, ky=%g, kz=%g\n", row, apple->Cij[xHarm[row]][j], apple->kx[xHarm[row]][j], apple->ky[xHarm[row]][j], apple->kz[j]); */
  }
  apple->C1 = cos(apple->phi1);
  apple->C2 = cos(apple->phi2);
  apple->C3 = cos(apple->phi3);
  apple->C4 = cos(apple->phi4);
  apple->S1 = sin(apple->phi1);
  apple->S2 = sin(apple->phi2);
  apple->S3 = sin(apple->phi3);
  apple->S4 = sin(apple->phi4);
  BxAmp = sqrt(sqr(apple->C1 - apple->C2 + apple->C3 - apple->C4) + sqr(apple->S1 - apple->S2 + apple->S3 - apple->S4));
  ByAmp = sqrt(sqr(apple->C1 + apple->C2 + apple->C3 + apple->C4) + sqr(apple->S1 + apple->S2 + apple->S3 + apple->S4));
  apple->BPeak[0] = fabs(apple->BPeak[0] * BxAmp);
  apple->BPeak[1] = fabs(apple->BPeak[1] * ByAmp);

  apple->initialized = 1;

  if (c_shim)
    free(c_shim);
  if (s_shim)
    free(s_shim);
  if (Cmn)
    free(Cmn);
  if (kx)
    free(kx);
  if (ky)
    free(ky);
  if (kz)
    free(kz);
  if (xHarm)
    free(xHarm);
  if (zHarm)
    free(zHarm);
  if (sHarm)
    free(sHarm);

  return;
}
