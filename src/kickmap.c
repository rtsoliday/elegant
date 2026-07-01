/*************************************************************************\
* Copyright (c) 2008 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2008 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: kickmap.c
 * purpose: track particles through a kick map (function of x and y)
 * 
 * Michael Borland, 2021
 */
#include "mdb.h"
#include "track.h"

void initializeKickMap(KICKMAP *map);
long interpolateKickMap(double *xpFactor, double *ypFactor, KICKMAP *map, double x, double y);

long trackKickMap(
  double **particle, /* array of particles */
  double **accepted, /* acceptance array */
  long nParticles,   /* number of particles */
  double pRef,       /* central momentum */
  KICKMAP *map,
  double zStart,
  double *sigmaDelta2) {
  long ip, iTop, ik, nKicks, kickSign = 1;
  double *coord;
  double dxp, dyp;
  double length;
  double dist, tan_yaw, yawOffset;
  double radCoef, isrCoef;

  length = map->length;
  if (map->flipSign)
    kickSign = -1;
  if (!map->initialized)
    initializeKickMap(map);

  if ((nKicks = map->nKicks) < 1)
    bombElegant("N_KICKS must be >=1 for UKICKMAP", NULL);

  length /= nKicks;

  if (map->dx || map->dy || map->dz)
    offsetBeamCoordinatesForMisalignment(particle, nParticles, map->dx, map->dy, map->dz);
  if (map->tilt)
    rotateBeamCoordinatesForMisalignment(particle, nParticles, map->tilt);

  iTop = nParticles - 1;
  dist = 0;
  tan_yaw = tan(map->yaw);
  if (abs(map->yawEnd) > 1)
    bombElegant("YAW_END parameter must be -1, 0, or 1", NULL);
  yawOffset = map->length / 2 * (map->yawEnd + 1.0) * tan_yaw;

  radCoef = sqr(particleCharge) * pow3(pRef) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(pRef) * 137.0359895);

  for (ik = 0; ik < nKicks; ik++) {
    if (isSlave || !distributedBeam) {
      for (ip = 0; ip <= iTop; ip++) {
        coord = particle[ip];

        /* 1. go through half length */
        coord[0] += coord[1] * length / 2.0;
        coord[2] += coord[3] * length / 2.0;
        coord[4] += length / 2.0 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));

        /* 2. apply the kicks 
         * use interpolation to get dxp and dyp 
         */
        if (!interpolateKickMap(&dxp, &dyp, map,
                                coord[0] - dist * tan_yaw + yawOffset, coord[2])) {
          /* particle is lost */
          swapParticles(particle[ip], particle[iTop]);
          if (accepted)
            swapParticles(accepted[ip], accepted[iTop]);
          particle[iTop][4] = zStart;
          particle[iTop][5] = pRef * (1 + particle[iTop][5]);
          iTop--;
          ip--;
        } else {
          dxp *= map->factor / ((1 + coord[5]) * kickSign * nKicks);
          dyp *= map->factor / ((1 + coord[5]) * kickSign * nKicks);
          coord[1] += dxp;
          coord[3] += dyp;

          if ((map->synchRad || map->isr) && length) {
            double dp, p, beta0, deltaFactor, F2, beta1;
            dp = coord[5];
            F2 = sqr(1 + dp) * (sqr(dxp) + sqr(dyp)) / sqr(length);
            p = pRef * (1 + dp);
            beta0 = p / sqrt(sqr(p) + 1);
            deltaFactor = 1 + dp;
            dp -= radCoef * sqr(deltaFactor) * F2 * length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
            if (map->isr)
              dp += isrCoef * deltaFactor * pow(F2, 0.75) * sqrt(length) * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
            if (sigmaDelta2)
              *sigmaDelta2 += sqr(isrCoef * deltaFactor) * pow(F2, 1.5) * length;
            p = pRef * (1 + dp);
            coord[1] *= (1 + coord[5]) / (1 + dp);
            coord[3] *= (1 + coord[5]) / (1 + dp);
            beta1 = p / sqrt(sqr(p) + 1);
            coord[5] = dp;
            coord[4] = beta1 * coord[4] / beta0;
          }

          /* 3. go through another half length */
          coord[0] += coord[1] * length / 2.0;
          coord[2] += coord[3] * length / 2.0;
          coord[4] += length / 2.0 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
        }
      }
    }
    dist += length;
  }

  if (map->tilt)
    rotateBeamCoordinatesForMisalignment(particle, nParticles, -map->tilt);
  if (map->dx || map->dy || map->dz)
    offsetBeamCoordinatesForMisalignment(particle, nParticles, -map->dx, -map->dy, -map->dz);

  return iTop + 1;
}

void initializeKickMap(KICKMAP *map) {
  SDDS_DATASET SDDSin;
  double *x = NULL, *y = NULL, *xpFactor = NULL, *ypFactor = NULL;
  long nx;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, map->inputFile) ||
      SDDS_ReadPage(&SDDSin) <= 0 ||
      !(x = SDDS_GetColumnInDoubles(&SDDSin, "x")) || !(y = SDDS_GetColumnInDoubles(&SDDSin, "y"))) {
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (!check_sdds_column(&SDDSin, "x", "m") ||
      !check_sdds_column(&SDDSin, "y", "m")) {
    fprintf(stderr, "UKICKMAP input file must have x and y in m (meters)\n");
    exitElegant(1);
  }

  if (!(xpFactor = SDDS_GetColumnInDoubles(&SDDSin, "xpFactor")) || !(ypFactor = SDDS_GetColumnInDoubles(&SDDSin, "ypFactor"))) {
    fprintf(stderr, "UKICKMAP input file must have both (xpFactor, ypFactor)\n");
    exitElegant(1);
  }
  if (!check_sdds_column(&SDDSin, "xpFactor", "") ||
      !check_sdds_column(&SDDSin, "ypFactor", "")) {
    fprintf(stderr, "KICKMAP input file must have xpFactor and ypFactor, which must be dimensionless\n");
    fprintf(stderr, "Do you mean to use UKICKMAP, for an undulator wiggler?\n");
    exitElegant(1);
  }

  if (!(map->points = SDDS_CountRowsOfInterest(&SDDSin)) || map->points < 2) {
    printf("file %s for UKICKMAP element has insufficient data\n", map->inputFile);
    fflush(stdout);
    exitElegant(1);
  }
  SDDS_Terminate(&SDDSin);

  if (map->xyFactor != 1) {
    long i;
    for (i = 0; i < map->points; i++) {
      x[i] *= map->xyFactor;
      y[i] *= map->xyFactor;
    }
  }

  /* It is assumed that the data is ordered so that x changes fastest.
   * This can be accomplished with sddssort -column=y,incr -column=x,incr
   * The points are assumed to be equipspaced.
   */
  nx = 1;
  map->xmin = x[0];
  while (nx < map->points) {
    if (x[nx - 1] > x[nx])
      break;
    nx++;
  }
  if ((nx >= map->points) || nx <= 1 || y[0] > y[nx] || (map->ny = map->points / nx) <= 1) {
    printf("file %s for UKICKMAP element doesn't have correct structure or amount of data\n",
           map->inputFile);
    fflush(stdout);
    printf("nx = %ld, ny=%ld\n", map->nx, map->ny);
    fflush(stdout);
    exitElegant(1);
  }
  map->nx = nx;
  map->xmax = x[nx - 1];
  map->dxg = (map->xmax - map->xmin) / (nx - 1);
  map->ymin = y[0];
  map->ymax = y[map->points - 1];
  map->dyg = (map->ymax - map->ymin) / (map->ny - 1);
  printf("UKICKMAP element from file %s: nx=%ld, ny=%ld, dxg=%e, dyg=%e, x:[%e, %e], y:[%e, %e]\n",
         map->inputFile, map->nx, map->ny, map->dxg, map->dyg,
         map->xmin, map->xmax,
         map->ymin, map->ymax);
  free(x);
  free(y);
  map->xpFactor = xpFactor;
  map->ypFactor = ypFactor;
  map->initialized = 1;
}

long interpolateKickMap(double *xpFactor, double *ypFactor, KICKMAP *map, double x, double y) {
  double Fa, Fb, fx, fy;
  long ix, iy;

  if (isnan(x) || isnan(y) || isinf(x) || isinf(y))
    return 0;

  ix = (x - map->xmin) / map->dxg;
  iy = (y - map->ymin) / map->dyg;
  if (ix < 0 || iy < 0 || ix > map->nx - 1 || iy > map->ny - 1)
    return 0;
  if (ix == (map->nx - 1))
    ix--;
  if (iy == (map->ny - 1))
    iy--;

  fx = (x - (ix * map->dxg + map->xmin)) / map->dxg;
  fy = (y - (iy * map->dyg + map->ymin)) / map->dyg;

  Fa = (1 - fy) * map->xpFactor[ix + iy * map->nx] + fy * map->xpFactor[ix + (iy + 1) * map->nx];
  Fb = (1 - fy) * map->xpFactor[ix + 1 + iy * map->nx] + fy * map->xpFactor[ix + 1 + (iy + 1) * map->nx];
  *xpFactor = (1 - fx) * Fa + fx * Fb;

  Fa = (1 - fy) * map->ypFactor[ix + iy * map->nx] + fy * map->ypFactor[ix + (iy + 1) * map->nx];
  Fb = (1 - fy) * map->ypFactor[ix + 1 + iy * map->nx] + fy * map->ypFactor[ix + 1 + (iy + 1) * map->nx];
  *ypFactor = (1 - fx) * Fa + fx * Fb;

  return 1;
}
