/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: matter.c
 * contents: track_through_matter()
 *
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "track.h"

#define SQRT_3 (1.7320508075688772)
#define AMU (1.6605e-27)
#define SQR_PI (PI * PI)
#define ALPHA (1. / 137.036)
#define mp_mks 1.672648500000000e-27
#define kb_mks k_boltzmann_mks

#define BS_Y0 (1e-8)

#ifdef HAVE_GPU
#  include <gpu_base.h>
#  include <gpu_matter.h>
#endif

double radiationLength(long Z, double A, double rho);
double solveBremsstrahlungCDF(double F);

long track_through_matter(
  double **part, int64_t np, long iPass, MATTER *matter, double Po, double **accepted, double z0) {
  int64_t ip;
  double L, Nrad, *coord, theta_rms = 0, beta, beta4gamma2, P, gamma = 0.0;
  double z1, z2, dx, dy, ds = 0.0, t = 0.0, dGammaFactor;
  double K1, K2 = 0.0, sigmaTotal, probScatter = 0.0, dgamma;
  double Xo, probBSScatter = 0, probERScatter = 0, rho, nDensity;
  //long nScatters = 0;
  int64_t i_top;
  long isLost;
  /* long sections; */
  long sections0 = 1, impulseMode;
  double L1, prob, probBS, probER;
  long multipleScattering = 0;
  long hitsMatter;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    ip = gpu_track_through_matter(np, iPass, matter, Po, accepted, z0);
#  ifdef GPU_VERIFY
    startCpuTimer();
    track_through_matter(part, np, iPass, matter, Po, accepted, z0);
    compareGpuCpu(np, "track_through_matter");
#  endif /* GPU_VERIFY */
    return ip;
  }
#endif /* HAVE_GPU */

  log_entry("track_through_matter");

  if (particleIsElectron == 0)
    bombElegant("MATTER element doesn't work for particles other than electrons", NULL);

  if ((matter->startPass >= 0 && matter->startPass > iPass) || (matter->endPass >= 0 && matter->endPass < iPass)) {
    exactDrift(part, np, matter->length);
    return np;
  }

  if (matter->length != 0) {
    L = matter->length;
    impulseMode = 0;
  } else if (matter->lEffective != 0) {
    L = matter->lEffective;
    impulseMode = 1;
  } else
    return np;
  L1 = L; /* mostly to suppress compiler warning. */

  if (matter->energyDecay && (matter->nuclearBremsstrahlung || matter->electronRecoil))
    bombElegant("ENERGY_DECAY=1 and NUCLEAR_BREHMSSTRAHLUNG=1 or ELECTRON_RECOIL=1 options to MATTER/SCATTER element are mutually exclusive", NULL);

  beta = Po / sqrt(sqr(Po) + 1);
  beta4gamma2 = sqr(beta)*sqr(Po);
  rho = matter->rho;
  if (matter->pressure > 0 && matter->temperature > 0)
    rho = matter->multiplicity * matter->pressure/(kb_mks * matter->temperature) * mp_mks * matter->A;
  nDensity = rho/(mp_mks * matter->A);

  if (matter->Xo == 0) {
    if (matter->Z < 1 || matter->A < 1 || rho <= 0)
      bombElegant("XO=0 but Z, A, rho, pressure, temperature, multiplicity invalid for MATTER element", NULL);
    Xo = radiationLength(matter->Z, matter->A, rho);
    /*
    printf("Computed radiation length for Z=%ld, A=%le, rho=%le is %le m\n",
           matter->Z, matter->A, matter->rho, Xo);
    */
  } else {
    Xo = matter->Xo;
    if (matter->Z > 0 || matter->A > 0 || rho > 0) {
      if (matter->pressure && matter->temperature)
        printWarningForTracking("Redundant data supplied for MATTER element", "Xo, Z, A, temperature, pressure all specified");
      else
        printWarningForTracking("Redundant data supplied for MATTER element", "Xo, Z, A, rho all specified");
    }
  }

  Nrad = L / Xo;
  dGammaFactor = 1 - exp(-Nrad);
  prob = probBS = probER = 0;
  if ((Nrad < 1e-3 || matter->nuclearBremsstrahlung || matter->electronRecoil) && !matter->forceMCS) {
    if (rho == 0)
      bombElegant("MATTER element is too thin or requests special features---provide Z, A, and rho (or pressure and temperature) for single-scattering calculation.", NULL);
    K1 = 4 * matter->Z * (matter->Z + 1) * sqr(particleRadius / (beta * Po));
    K2 = sqr(pow(matter->Z, 1. / 3.) * ALPHA / Po);
    sigmaTotal = K1 * pow(PI, 3) / (sqr(K2) + K2 * SQR_PI);
    probScatter = rho / (AMU * matter->A) * L * sigmaTotal;
    /* printf("K1=%le, K2=%le, mean expected number of scatters is %le\n", K1, K2, probScatter); */
    probBSScatter = 0;
    if (matter->nuclearBremsstrahlung) {
      probBSScatter = 4 * L / (3 * Xo) * (-log(BS_Y0) - (1 - BS_Y0) + 3. / 8. * (1 - BS_Y0 * BS_Y0));
    }
    if (matter->electronRecoil) {
      probERScatter = L * rho / (AMU * matter->A) * PIx2 * matter->Z * sqr(re_mks) / Po * (1 / BS_Y0 - 1);
    }
    sections0 = probScatter / matter->pLimit + 1;
    Nrad /= sections0;
    multipleScattering = 0;
    L1 = L / sections0;
    prob = probScatter / sections0;
    probBS = probBSScatter / sections0;
    probER = probERScatter / sections0;
    /*
    printf("Sections=%ld, L1 = %le, probIS = %le, probBS = %le, probER = %le\n", sections0, L1, prob, probBS, probER);
    */
  } else {
    if (matter->forceMCS && Nrad<1e-3) {
        multipleScattering = 2;
	/* This is for the projected angles */
        theta_rms = sqrt(4*PI*nDensity*L*sqr(particleRadius)/beta4gamma2*matter->Z*(matter->Z+1)*log(184.15/pow(matter->Z,1./3.)));
    } else {
      multipleScattering = 1;
      theta_rms = 13.6 / particleMassMV / Po / sqr(beta) * sqrt(Nrad) * (1 + 0.038 * log(Nrad));
    }
  }

  i_top = np - 1;
  if (impulseMode)
    L = L1 = 0;
  for (ip = 0; ip <= i_top; ip++) {
    coord = part[ip];
    isLost = 0;
    hitsMatter = 1;
    if (matter->spacing > 0 && matter->width > 0) {
      double q, p, offset;
      if (matter->spacing <= matter->width)
        bombElegant("MATTER SPACING parameter is less than WIDTH parameter", NULL);
      if (matter->spacing <= 0)
        bombElegant("MATTER SPACING parameter is <= 0", NULL);
      offset = 0;
      if (matter->nSlots > 0 && matter->nSlots % 2 == 0)
        offset = matter->spacing / 2;
      q = coord[0] * cos(matter->tilt) + coord[2] * sin(matter->tilt) - (matter->center + offset);
      p = fabs(fmod(fabs(q), matter->spacing));
      if (p < matter->width / 2 || p > (matter->spacing - matter->width / 2)) {
        long slot;
        if (matter->nSlots > 0) {
          if (matter->nSlots % 2) {
            slot = fabs(q) / matter->spacing + 0.5;
            if (slot <= matter->nSlots / 2)
              hitsMatter = 0;
          } else {
            slot = fabs(q + matter->spacing / 2) / matter->spacing;
            if (slot < matter->nSlots / 2)
              hitsMatter = 0;
          }
        } else
          /* infinite slot array */
          hitsMatter = 0;
      }
    }
    if (hitsMatter && Nrad) {
      if (matter->energyDecay || matter->nuclearBremsstrahlung) {
        P = (1 + coord[5]) * Po;
        gamma = sqrt(sqr(P) + 1);
        beta = P / gamma;
        t = coord[4] / beta;
      }
      if (multipleScattering==1) {
        /* use the multiple scattering formula */
        z1 = gauss_rn(0, random_2);
        z2 = gauss_rn(0, random_2);
        coord[0] += (dx = (z1 / SQRT_3 + z2) * L * theta_rms / 2 + L * coord[1]);
        coord[1] += z2 * theta_rms;
        z1 = gauss_rn(0, random_2);
        z2 = gauss_rn(0, random_2);
        coord[2] += (dy = (z1 / SQRT_3 + z2) * L * theta_rms / 2 + L * coord[3]);
        coord[3] += z2 * theta_rms;
        ds = sqrt(sqr(L) + sqr(dx) + sqr(dy));
      } else if (multipleScattering==2) {
	coord[0] += L/2*coord[1];
	coord[2] += L/2*coord[3];
        z1 = gauss_rn(0, random_2);
        z2 = gauss_rn(0, random_2);
        coord[1] += theta_rms*z1;
        coord[3] += theta_rms*z2;
	coord[0] += L/2*coord[1];
	coord[2] += L/2*coord[3];
	ds = L;
      } else {
        /* model scattering using the cross section */
        double F, theta, phi, zs, dxp, dyp;
        long is;
        ds = dgamma = 0;
        /* sections = sections0; */
        for (is = 0; is < sections0 && !isLost; is++) {
          if (random_2(1) < prob) {
            //nScatters++;
            /* single-scattering computation */
            /* scatter occurs at location 0<=zs<=L */
            zs = L1 * random_2(1);
            /* pick a value for CDF and get corresponding angle */
            F = random_2(1);
            theta = sqrt((1 - F) * K2 * SQR_PI / (K2 + F * SQR_PI));
            phi = random_2(1) * PIx2;
            dxp = theta * sin(phi);
            dyp = theta * cos(phi);
            /* advance to location of scattering event */
            ds += zs * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            /* scatter */
            coord[1] += dxp;
            coord[3] += dyp;
            /* advance to end of slice */
            coord[0] += dxp * (L1 - zs);
            coord[2] += dyp * (L1 - zs);
            ds += (L1 - zs) * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
          } else {
            ds += L1 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            coord[0] += coord[1] * L1;
            coord[2] += coord[3] * L1;
          }
          if (matter->energyDecay || matter->nuclearBremsstrahlung) {
            if (probBS != 0 && random_2(1) < probBS)
              gamma -= gamma * solveBremsstrahlungCDF(random_2(1));
            if (probER != 0 && random_2(1) < probER)
              gamma -= BS_Y0 / (1 - random_2(1) * (1 - BS_Y0));
            if (gamma <= 1) {
              isLost = 1;
              break;
            }
          }
        }
      }
      if (!isLost) {
        if (matter->nuclearBremsstrahlung) {
          P = sqrt(sqr(gamma) - 1);
          coord[5] = (P - Po) / Po;
          beta = P / gamma;
          coord[4] = t * beta + ds;
        } else if (matter->energyDecay) {
          dgamma = gamma * dGammaFactor;
          if (matter->energyStraggle) {
            double dgamma1;
            /* very simple-minded estimate: StDev(dE) = Mean(dE)/2 */
            while ((dgamma1 = dgamma * (1 + 0.5 * gauss_rn(0, random_2))) < 0)
              ;
            dgamma = dgamma1;
          }
          gamma -= dgamma;
          if (gamma <= 1)
            isLost = 1;
          else {
            P = sqrt(sqr(gamma) - 1);
            coord[5] = (P - Po) / Po;
            beta = P / gamma;
            coord[4] = t * beta + ds;
          }
        } else
          coord[4] += ds;
      }
      if (isLost) {
        swapParticles(part[ip], part[i_top]);
        if (accepted)
          swapParticles(accepted[ip], accepted[i_top]);
        part[i_top][4] = z0 + ds;
        part[i_top][5] = 0;
        i_top--;
        ip--;
      }
    } else {
      coord[0] += L * coord[1];
      coord[2] += L * coord[3];
      coord[4] += L * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    }
  }

  log_exit("track_through_matter");
  return (i_top + 1);
}

double inelasticGasScattering(double Z, double gamma, double nL, double P) {
  double C1, C2, St;

  C1 = 16 * sqr(re_mks * Z) / (3 * 137) * log(183 / pow(Z, 1. / 3.)) * nL;
  C2 = 16 * sqr(re_mks) * Z / (3 * 137) * nL;
  St = P;

  return -(-40 * C1 + 81 * C2 - 40 * C2 * log((5 * gamma) / 2.) + sqrt(160 * C2 * (25 * C1 - 35 * C2 + 40 * St + 25 * C2 * log((5 * gamma) / 2.)) + ipow(40 * C1 - 81 * C2 + 40 * C2 * log((5 * gamma) / 2.), 2))) / (80. * C2);
}

double radiationLength(long Z, double A, double rho)
/* Returns radiation length for electrons in m for given Z and A (in AMUs). See PhysRevD.86.010001, page 329 */
{
  double fZ;
  double alpha, a2;
  double Lrad, Lradp;

  alpha = e_mks * e_mks / (4 * PI * epsilon_o * hbar_mks * c_mks);
  a2 = sqr(alpha * Z);
  fZ = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)));
  switch (Z) {
  case 1:
    Lrad = 5.31;
    Lradp = 6.144;
    break;
  case 2:
    Lrad = 4.79;
    Lradp = 5.621;
    break;
  case 3:
    Lrad = 4.74;
    Lradp = 5.805;
    break;
  case 4:
    Lrad = 4.71;
    Lradp = 5.924;
    break;
  default:
    Lrad = log(184.15 * pow(Z, -1. / 3.));
    Lradp = log(1194 * pow(Z, -2. / 3.));
    break;
  }
  /* factor is to convert to meters */
  return 1e-3 / (4 * alpha * sqr(re_mks) * NAvogadro / A * (Z * Z * (Lrad - fZ) + Z * Lradp)) / rho;
}

double *lnyTable = NULL, *FTable = NULL;

double solveBremsstrahlungCDF(double F)
/* Solve F == G(y)/G(1) where G(y)=(ln(y/y0) - (y-y0) + 3/8*(y^2-y0^2)
 */
{
  static double dy, y0 = BS_Y0;
  double y;
  long nf = 1000, i, code;

  if (!FTable) {
    /* make a table of F(y) with nf points */
    double y1;
    lnyTable = tmalloc(sizeof(*lnyTable) * nf);
    FTable = tmalloc(sizeof(*FTable) * nf);
    dy = (1 - y0) / (nf - 1.);
    for (i = 0; i < nf; i++) {
      y1 = y0 + i * dy;
      lnyTable[i] = log(y1 / y0);
      FTable[i] = log(y1 / y0) - (y1 - y0) + 3. / 8. * (sqr(y1) - sqr(y0));
    }
    for (i = 0; i < nf; i++)
      FTable[i] /= FTable[nf - 1];
  }

  y = interp(lnyTable, FTable, nf, F, 0, 1, &code);
  if (code == 1) {
    y = y0 * exp(y);
  } else {
    printWarningForTracking("Interpolation problem for bremsstrahlung.", NULL);
    y = y0;
  }
  return y;
}
