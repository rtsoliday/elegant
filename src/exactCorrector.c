/*************************************************************************\
* Copyright (c) 2015 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2015 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: exactCorrector.c
 * contents:  track_through_exact_corrector()
 *
 *
 * Michael Borland, 2015
 */
#include "mdb.h"
#include "track.h"
#include "multipole.h"

void computeTotalSteeringMultipoleErrors(MULTIPOLE_DATA *steeringMult, double sysFactor,
                                         MULTIPOLE_DATA *randomMult, double ranFactor,
                                         MULTIPOLE_DATA *totalMult,
                                         char *name);


#define FILLX fillPowerArray
#define FILLY fillPowerArray

#define DO_MKICKS_RET apply_canonical_multipole_kicks_ret
#define DO_MKICKS_NORET apply_canonical_multipole_kicks_noret

int applySteeringMultipoleKicks(
  double *coord,
  double xkick,
  double ykick,
  MULTIPOLE_DATA *multData) {
  long imult, maxOrder;
  double denom, qx, qy, delta;
  double *xpow, *ypow;
  
  if (!xkick && !ykick)
    return 1;

  denom = sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  delta = coord[5];
#if TURBO_RECIPROCALS
  qx = ((1 + delta) / denom) * coord[1] ;
  qy = ((1 + delta) / denom) * coord[3] ;
#else
  qx = (1 + delta) * coord[1] / denom;
  qy = (1 + delta) * coord[3] / denom;
#endif

  maxOrder = findMaximumOrder(-1, -1, multData, NULL, NULL);
  xpow = tmalloc(sizeof(*xpow) * (maxOrder + 1));
  ypow = tmalloc(sizeof(*ypow) * (maxOrder + 1));
  FILLX(coord[0], xpow, maxOrder);
  FILLY(coord[2], ypow, maxOrder);

  for (imult = 0; imult < multData->orders; imult++) {
    if (xkick)
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                      multData->order[imult],
                                      multData->KnL[imult] * xkick, 0);
    if (ykick)
      DO_MKICKS_NORET(&qx, &qy, xpow, ypow,
                                      multData->order[imult],
                                      multData->JnL[imult] * ykick, 1);
  }
  if ((denom = sqr(1 + delta) - sqr(qx) - sqr(qy)) <= 0)
    return 0;
  denom = sqrt(denom);
  coord[1] = qx / denom;
  coord[3] = qy / denom;

  free(xpow);
  free(ypow);

  return 1;
}

long trackThroughExactCorrector(double **part, long n_part, ELEMENT_LIST *eptr, double Po, double **accepted, double z_start, double *sigmaDelta2) {
  long i_part, i_top;
  double xkick, ykick, kick, tilt, length, *coord, rho0, theta0, rho, theta, alpha, arg;
  double dx = 0, dy = 0, dz = 0, rigidity0 = 0;
  MULTIPOLE_DATA *multData;
  long isr, sr;
  EHCOR *ehcor;
  EVCOR *evcor;
  EHVCOR *ehvcor;
  void *pePtr;
  short lost;

  pePtr = eptr->p_elem;

  xkick = ykick = kick = tilt = length = 0;
  isr = sr = 0;
  multData = NULL;

  switch (eptr->type) {
  case T_EHCOR:
    ehcor = (EHCOR *)pePtr;
    length = ehcor->length;
    xkick = ehcor->kick * ehcor->calibration;
    tilt = ehcor->tilt;
    isr = ehcor->isr;
    sr = ehcor->synchRad;
    dx = ehcor->dx;
    dy = ehcor->dy;
    dz = ehcor->dz;
    if (ehcor->steeringMultipoles && !ehcor->steeringMultipoleData.initialized) {
      readErrorMultipoleData(&(ehcor->steeringMultipoleData), ehcor->steeringMultipoles, 1);
    }
    multData = &(ehcor->steeringMultipoleData);
    if (ehcor->randomMultipoles) {
      if (!ehcor->randomMultipoleData.initialized) {
        if (!ehcor->steeringMultipoles)
          bombElegantVA("Error: must give STEERING_MULTIPOLES with RANDOM_MULTIPOLES for EHKICK %s\n",
                        eptr->name);
        readErrorMultipoleData(&(ehcor->randomMultipoleData), ehcor->randomMultipoles, 2);
      }
      if (!ehcor->multipolesRandomized) {
        /* generate instance of random multipoles, sum to make total */
        computeTotalSteeringMultipoleErrors(&(ehcor->steeringMultipoleData), ehcor->steeringMultipoleFactor,
                                            &(ehcor->randomMultipoleData), ehcor->randomMultipoleFactor,
                                            &(ehcor->totalMultipoleData), eptr->name);
        ehcor->multipolesRandomized = 1;
      }
      multData = &(ehcor->totalMultipoleData);
    }
    break;
  case T_EVCOR:
    evcor = (EVCOR *)pePtr;
    length = evcor->length;
    ykick = evcor->kick * evcor->calibration;
    tilt = evcor->tilt;
    isr = evcor->isr;
    sr = evcor->synchRad;
    dx = evcor->dx;
    dy = evcor->dy;
    dz = evcor->dz;
    if (evcor->steeringMultipoles && !evcor->steeringMultipoleData.initialized) {
      readErrorMultipoleData(&(evcor->steeringMultipoleData), evcor->steeringMultipoles, 1);
    }
    multData = &(evcor->steeringMultipoleData);
    if (evcor->randomMultipoles) {
      if (!evcor->randomMultipoleData.initialized) {
        if (!evcor->steeringMultipoles)
          bombElegantVA("Error: must give STEERING_MULTIPOLES with RANDOM_MULTIPOLES for EVKICK %s\n",
                        eptr->name);
        readErrorMultipoleData(&(evcor->randomMultipoleData), evcor->randomMultipoles, 2);
      }
      if (!evcor->multipolesRandomized) {
        /* generate instance of random multipoles, sum to make total */
        computeTotalSteeringMultipoleErrors(&(evcor->steeringMultipoleData), evcor->steeringMultipoleFactor,
                                            &(evcor->randomMultipoleData), evcor->randomMultipoleFactor,
                                            &(evcor->totalMultipoleData), eptr->name);
        evcor->multipolesRandomized = 1;
      }
      multData = &(evcor->totalMultipoleData);
    }
    break;
  case T_EHVCOR:
    ehvcor = (EHVCOR *)pePtr;
    length = ehvcor->length;
    xkick = ehvcor->xkick * ehvcor->xcalibration;
    ykick = ehvcor->ykick * ehvcor->ycalibration;
    tilt = ehvcor->tilt;
    isr = ehvcor->isr;
    sr = ehvcor->synchRad;
    dx = ehvcor->dx;
    dy = ehvcor->dy;
    dz = ehvcor->dz;
    if (ehvcor->steeringMultipoles && !ehvcor->steeringMultipoleData.initialized) {
      readErrorMultipoleData(&(ehvcor->steeringMultipoleData), ehvcor->steeringMultipoles, 1);
    }
    multData = &(ehvcor->steeringMultipoleData);
    if (ehvcor->randomMultipoles) {
      if (!ehvcor->randomMultipoleData.initialized) {
        if (!ehvcor->steeringMultipoles)
          bombElegantVA("Error: must give STEERING_MULTIPOLES with RANDOM_MULTIPOLES for EKICK %s\n",
                        eptr->name);
        readErrorMultipoleData(&(ehvcor->randomMultipoleData), ehvcor->randomMultipoles, 2);
      }
      if (!ehvcor->multipolesRandomized) {
        /* generate instance of random multipoles, sum to make total */
        computeTotalSteeringMultipoleErrors(&(ehvcor->steeringMultipoleData), ehvcor->steeringMultipoleFactor,
                                            &(ehvcor->randomMultipoleData), ehvcor->randomMultipoleFactor,
                                            &(ehvcor->totalMultipoleData), eptr->name);
        ehvcor->multipolesRandomized = 1;
      }
      multData = &(ehvcor->totalMultipoleData);
    }
    break;
  default:
    break;
  }

  /* xkick and ykick are actually angles, not kicks */
  theta0 = atan(sqrt(sqr(tan(xkick)) + sqr(tan(ykick))));
  if (theta0 == 0) {
    exactDrift(part, n_part, length);
    return n_part;
  }
  /* the tilt will account for the direction of bending */
  /* theta0 and rho0 are positive values */
  tilt += atan2(tan(ykick), tan(xkick));
  rho0 = fabs(length) / sin(theta0);
  if (length < 0) /* back tracking */
    tilt += PI;

  // The minus sign is because the sign of particleCharge*particleRelSign is positive for electrons
  // There is no factor of (1+delta) here. See AOP-TN-2010-029, Rev. 2
  rigidity0 = -Po*particleMass*c_mks/particleCharge*particleRelSign;

  if (sigmaDelta2)
    *sigmaDelta2 = 0;

  i_top = -1;
  if (isSlave || !distributedBeam) {
    i_top = n_part - 1;

    if (dx || dy || dz)
      offsetBeamCoordinatesForMisalignment(part, n_part, dx, dy, dz);
    if (tilt)
      rotateBeamCoordinatesForMisalignment(part, n_part, tilt);

    /* Apply steering kicks */
    for (i_part = 0; i_part <= i_top; i_part++) {
      lost = 0;
      if (!(coord = part[i_part])) {
        printf("error: null coordinate pointer for particle %ld (trackThroughExactCorrector)\n", i_part);
        fflush(stdout);
        abort();
      }

      if (multData && multData->orders)
        lost = !applySteeringMultipoleKicks(coord, theta0 / 2, 0.0, multData);

      if (!lost) {
        if (rho0 == 0 || length == 0) {
          coord[1] = tan(theta0 / (1 + coord[5]) + atan(coord[1]));
        } else {
          alpha = atan(coord[1]);
          rho = rho0 * (1 + coord[5]);
          if (rho > 0 && (arg = length / rho + sin(alpha)) <= 1 && arg >= -1) {
            double dyf, lprod;
            theta = asin(arg) - alpha;
            dyf = coord[3] / sqrt(1 + sqr(coord[1]) - sqr(coord[3]));
            coord[2] += dyf * theta * rho;
	    // Guard against underflow
	    lprod = theta * rho * sqrt(1 + sqr(dyf));
	    if (lprod<length)
	      lprod = length;
	    coord[4] += lprod;
            coord[0] += rho * (cos(alpha) - cos(theta + alpha));
            coord[1] = tan(theta + alpha);
	    if (spinCoordOffset) {
	      updateSpinQuaternionLocalFS(coord+spinCoordOffset,
					  0.0, -rigidity0/rho0, 0.0,
					  0.0, 0.0, 0.0, 0.0, /* effective length already computed in lprod */
					  lprod,
					  0.0, /* suppresses FS frame rotation term */
					  Po,
					  -particleCharge*particleRelSign, particleMass, particleAnomalousMagneticMoment);
	    }
          } else
            lost = 1;
        }

        if (!lost) {
          if (multData && multData->orders)
            lost = !applySteeringMultipoleKicks(coord, theta0 / 2, 0.0, multData);
        }
      }

      if (lost) {
        swapParticles(part[i_part], part[i_top]);
        if (accepted) {
          if (!accepted[i_top]) {
            printf(
              "error: couldn't swap acceptance data for particles %ld and %ld--latter is null pointer (trackThroughExactCorrector)\n",
              i_part, i_top);
            fflush(stdout);
            abort();
          }
          swapParticles(accepted[i_part], accepted[i_top]);
        }
        part[i_top][4] = z_start;
        part[i_top][5] = Po * (1 + part[i_top][5]);
        i_top--;
        i_part--;
      }
    }

    /* Note that we undo misalignments for all particles, including lost particles */
    if (tilt)
      rotateBeamCoordinatesForMisalignment(part, n_part, -tilt);
    if (dx || dy || dz)
      offsetBeamCoordinatesForMisalignment(part, n_part, -dx, -dy, -dz);
  }

  if (sr)
    addCorrectorRadiationKick(part, i_top, eptr, eptr->type, Po, sigmaDelta2, !isr);

  return i_top + 1;
}

void computeTotalSteeringMultipoleErrors(MULTIPOLE_DATA *systematicMult, double sysFactor,
                                         MULTIPOLE_DATA *randomMult, double ranFactor,
                                         MULTIPOLE_DATA *totalMult,
                                         char *name) {
  long i;
  if (!totalMult->initialized) {
    totalMult->orders = systematicMult->orders;
    totalMult->order = tmalloc(sizeof(*(totalMult->order)) * totalMult->orders);
    totalMult->KnL = tmalloc(sizeof(*(totalMult->KnL)) * totalMult->orders);
    totalMult->JnL = tmalloc(sizeof(*(totalMult->JnL)) * totalMult->orders);
    totalMult->initialized = 1;
  }
  for (i = 0; i < systematicMult->orders; i++) {
    if (systematicMult->order[i] != randomMult->order[i])
      bombElegantVA("Error: order mismatch for STEERING_MULTIPOLES and RANDOM_MULTIPOLES for EVKICK %s\n",
                    name);
    totalMult->order[i] = randomMult->order[i];
    totalMult->KnL[i] = systematicMult->KnL[i] * sysFactor + gauss_rn_lim(0.0, 1.0, 2.0, random_1_elegant) * randomMult->KnL[i] * ranFactor;
    totalMult->JnL[i] = systematicMult->JnL[i] * sysFactor + gauss_rn_lim(0.0, 1.0, 2.0, random_1_elegant) * randomMult->JnL[i] * ranFactor;
  }
}
