/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: kicker.c
 * contents: track_through_kicker()
 * 
 * Michael Borland, 1991
 */
#include "mdb.h"
#include "track.h"
#include "table.h"

void initializeDeflectionMap(KICKER *kicker);
long interpolateDeflectionMap(double *xpFactor, double *ypFactor, KICKER *kicker, double x, double y);
void set_up_kicker(KICKER *kicker);
void set_up_mkicker(MKICKER *kicker);

void track_through_kicker(
  double **part, long np, KICKER *kicker, double p_central, long pass, long default_order) {
  long i, n, ip;
  double time, time_offset, angle, t0, *coord, amplitude, ds;
  //double sum_amp;
  double x, xp, y, yp, dp, s, curv, dx;
  double xp0, yp0;
  double theta_i, alpha_i, alpha_f;
  double xpFactor, ypFactor;

  log_entry("track_through_kicker");

  if (!kicker->t_wf) {
    set_up_kicker(kicker);
    if (fabs(kicker->time_offset)>100*(kicker->tmax-kicker->tmin))
      printWarningForTracking("BUMPER time offset is large", "The TIME_OFFSET value for is much larger than the waveform length. Consider use of the FIRE_ON_PASS parameter to prevent problems with round-off error.");
  }
  
  if (np <= 0)
    return;

  if (kicker->fire_on_pass > pass) {
    drift_beam(part, np, kicker->length, default_order);
    log_exit("track_through_kicker");
    return;
  }

  if (!kicker->t_wf || !kicker->amp_wf || !kicker->n_wf ||
      kicker->tmin >= kicker->tmax)
    bombElegant("no (valid) waveform data for kicker", NULL);

  if (kicker->phase_reference == 0)
    kicker->phase_reference = unused_phase_reference();

  switch (get_phase_reference(&time_offset, kicker->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!kicker->fiducial_seen) {
      /* set reference phase so that the center of this bunch goes through
                 * at the desired phase.
                 */
      t0 = 0;
      for (ip = t0 = 0; ip < np; ip++)
        t0 += part[ip][4] / (c_mks * beta_from_delta(p_central, part[ip][5]));
#if (!USE_MPI)
      t0 /= np;
#else
      if (notSinglePart) {
        double t0_sum;
        long np_total;

        MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
        MPI_Allreduce(&t0, &t0_sum, 1, MPI_DOUBLE, MPI_SUM, workers);
        t0 = t0_sum / np_total;
      } else
        t0 /= np;
#endif
      kicker->t_fiducial = t0;
      kicker->fiducial_seen = 1;
    }
    set_phase_reference(kicker->phase_reference, time_offset = kicker->t_fiducial);
    break;
  default:
    bombElegant("unknown return value from get_phase_reference()", NULL);
    break;
  }

  if (kicker->length < 0) {
    log_exit("track_through_kicker");
    return;
  }

  time_offset += kicker->time_offset;

  if (kicker->dx || kicker->dy || kicker->dz)
    offsetBeamCoordinatesForMisalignment(part, np, kicker->dx, kicker->dy, kicker->dz);
  if (kicker->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, kicker->tilt);

  //sum_amp = 0;
  for (ip = 0; ip < np; ip++) {
    angle = kicker->angle;
    time = part[ip][4] / (c_mks * beta_from_delta(p_central, part[ip][5])) - time_offset;

    if (time > kicker->tmax || time < kicker->tmin) {
      if (kicker->periodic) {
        n = (time - kicker->tmin) / (kicker->tmax - kicker->tmin);
        time -= n * (kicker->tmax - kicker->tmin);
      }
    }

    amplitude = 0;
    if (time > kicker->tmax || time < kicker->tmin)
      angle *= 0;
    else {
      /* should to a binary search here ! */
      for (i = 1; i < kicker->n_wf; i++)
        if (kicker->t_wf[i] >= time)
          break;
      if (i == kicker->n_wf) {
        printf("error: waveform interpolation problem in track_through_kicker()\n");
        fflush(stdout);
        printf("particle time is %21.15e\nwaveform limits are %21.15e to %21.15e\n",
               time, kicker->tmin, kicker->tmax);
        fflush(stdout);
        exitElegant(1);
      }
      i--;
      angle *= (amplitude = INTERPOLATE(kicker->amp_wf[i], kicker->amp_wf[i + 1],
                                        kicker->t_wf[i], kicker->t_wf[i + 1], time));
    }
    //sum_amp += amplitude;

    coord = part[ip];

    if (!angle) {
      coord[0] += kicker->length * coord[1];
      coord[2] += kicker->length * coord[3];
      coord[4] += kicker->length * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
      continue;
    }

    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    s = coord[4];
    dp = coord[5];

    if (kicker->n_kicks <= 0) {
      xp0 = xp;
      yp0 = yp;
      if (kicker->length != 0) {
        curv = sin(-angle) / kicker->length / (1 + dp);
        theta_i = atan(xp);
        alpha_i = -theta_i;
        alpha_f = asin(kicker->length * curv + sin(alpha_i));
        if (fabs(curv * kicker->length) < 1e-12) {
          x += (dx = kicker->length * xp);
          s += (ds = fabs(kicker->length / cos(alpha_i)));
        } else {
          x += (dx = (cos(alpha_f) - cos(alpha_i)) / curv);
          s += (ds = fabs((alpha_f - alpha_i) / curv));
        }
        xp = tan(theta_i - (alpha_f - alpha_i));
        y += kicker->length * yp;
      } else {
        xp += angle / (1 + dp);
      }
      if (spinCoordOffset) {
        /* Use the kick values to infer the spin rotations. Note that FS frame rotation is not subtracted. */
	double P = p_central*(1+coord[5]);
	double gamma = sqrt(P*P+1);
	performQuaternionRotation(coord+spinCoordOffset,
				  -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				  (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				  0);
      }
    } else if (!kicker->deflectionMap) {
      double l1, x0, y0;
      l1 = kicker->length / kicker->n_kicks;
      if (l1 != 0) {
        for (i = 0; i < kicker->n_kicks; i++) {
	  xp0 = xp;
	  yp0 = yp;
          curv = sin(-angle) / kicker->length / (1 + dp) * (1 + kicker->b2 * (x * x - y * y));
          theta_i = atan(xp);
          alpha_i = -theta_i;
          alpha_f = asin(l1 * curv + sin(alpha_i));
          x0 = x;
          if (fabs(curv * l1) < 1e-12) {
            x += (dx = l1 * xp);
            s += (ds = fabs(l1 / cos(alpha_i)));
          } else {
            x += (dx = (cos(alpha_f) - cos(alpha_i)) / curv);
            s += (ds = fabs((alpha_f - alpha_i) / curv));
          }
          xp = tan(theta_i - (alpha_f - alpha_i));
          x0 = (x + x0) / 2;
          y0 = y;
          y += l1 * yp;
          y0 = (y + y0) / 2;
          yp += -2 * sin(angle) / kicker->length / (1 + dp) * ds * x0 * y0 * kicker->b2;
	  if (spinCoordOffset) {
            /* Use the kick values to infer the spin rotations. Note that FS frame rotation is not subtracted. */
	    double P = p_central*(1+coord[5]);
	    double gamma = sqrt(P*P+1);
	    performQuaternionRotation(coord+spinCoordOffset,
				  -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				  (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				  0);
	  }
        }
      } else {
        xp0 = xp;
	yp0 = yp;
        xp += angle / (1 + dp);
        if (spinCoordOffset) {
          /* Use the kick values to infer the spin rotations. Note that FS frame rotation is not subtracted. */
          double P = p_central*(1+coord[5]);
          double gamma = sqrt(P*P+1);
          performQuaternionRotation(coord+spinCoordOffset,
				  -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				  (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				  0);
	}
      }
    } else {
      double l1, l2;
      l1 = kicker->length / kicker->n_kicks;
      l2 = kicker->length / kicker->n_kicks / 2.;
      angle /= kicker->n_kicks;
      /* deflection map mode */
      for (i = 0; i < kicker->n_kicks; i++) {
	xp0 = xp;
	yp0 = yp;
        if (i == 0) {
          /* half drift */
          s += l2 * (1 + (sqr(xp) + sqr(yp)) / 2.);
          x += xp * l2;
          y += yp * l2;
        } else {
          /* full drift */
          s += l1 * (1 + (sqr(xp) + sqr(yp)) / 2.);
          x += xp * l1;
          y += yp * l1;
        }
        /* kick */
        interpolateDeflectionMap(&xpFactor, &ypFactor, kicker, x, y);
        xp += angle * xpFactor / (1 + dp);
        yp += angle * ypFactor / (1 + dp);
        if (spinCoordOffset) {
          /* Use the kick values to infer the spin rotations. Note that FS frame rotation is not subtracted. */
          double P = p_central*(1+coord[5]);
          double gamma = sqrt(P*P+1);
          performQuaternionRotation(coord+spinCoordOffset,
				  -(1+particleAnomalousMagneticMoment*gamma)*(yp-yp0),
				  (1+particleAnomalousMagneticMoment*gamma)*(xp-xp0),
				  0);
	}
      }
      /* half drift */
      s += l2 * (1 + (sqr(xp) + sqr(yp)) / 2.);
      x += xp * l2;
      y += yp * l2;
    }

    coord[0] = x;
    coord[1] = xp;
    coord[2] = y;
    coord[3] = yp;
    coord[4] = s;
  }

  /*
    if (np)
        printf("average kicker amplitude = %f\n", sum_amp/np);
        fflush(stdout);
 */

  if (kicker->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, -kicker->tilt);
  if (kicker->dx || kicker->dy || kicker->dz)
    offsetBeamCoordinatesForMisalignment(part, np, -kicker->dx, -kicker->dy, -kicker->dz);

  log_exit("track_through_kicker");
}

void set_up_kicker(KICKER *kicker) {
  TABLE data;
  long i;

  log_entry("set_up_kicker");

  if (!kicker->waveform)
    bombElegant("no waveform filename given for kicker", NULL);

  if (!getTableFromSearchPath(&data, kicker->waveform, 1, 0))
    bombElegant("unable to read waveform for kicker", NULL);

  if (data.n_data <= 1)
    bombElegant("kicker waveform contains less than 2 points", NULL);

  kicker->t_wf = data.c1;
  kicker->amp_wf = data.c2;
  kicker->n_wf = data.n_data;
  for (i = 0; i < kicker->n_wf - 1; i++)
    if (kicker->t_wf[i] > kicker->t_wf[i + 1])
      bombElegant("time values not monotonically increasing in kicker waveform", NULL);
  kicker->tmin = kicker->t_wf[0];
  kicker->tmax = kicker->t_wf[kicker->n_wf - 1];
  tfree(data.xlab);
  tfree(data.ylab);
  tfree(data.title);
  tfree(data.topline);
  data.xlab = data.ylab = data.title = data.topline = NULL;

  kicker->xpFactor = kicker->ypFactor = NULL;
  kicker->points = kicker->nx = kicker->ny = 0;
  kicker->xmin = kicker->xmax = kicker->dxg = 0;
  kicker->ymin = kicker->ymax = kicker->dyg = 0;
  if (kicker->deflectionMap) {
    if (kicker->n_kicks < 1)
      bombElegant("Need to have n_kicks>0 for deflection map in KICKER element", NULL);
    initializeDeflectionMap(kicker);
  }
  log_exit("set_up_kicker");
}

void track_through_mkicker(
  double **part, long np, MKICKER *kicker, double p_central, long pass, long default_order) {
  long i, n, ip;
  double time, time_offset, strength, t0, amplitude;
  double dummy;
  long n_parts;
  short skew[3] = {0, 0, 0};
  long multipoleOrder[3] = {0, 0, 0};
  double KnL[3] = {0, 0, 0};

  if (np <= 0)
    return;

  if (!kicker->t_wf)
    set_up_mkicker(kicker);

  if (kicker->order < 1)
    bombElegant("order<1 for MKICKER", NULL);

  if (kicker->fire_on_pass > pass) {
    drift_beam(part, np, kicker->length, default_order);
    return;
  }

  if (!kicker->t_wf || !kicker->amp_wf || !kicker->n_wf ||
      kicker->tmin >= kicker->tmax)
    bombElegant("no (valid) waveform data for MBUMPER", NULL);

  if (kicker->phase_reference == 0)
    kicker->phase_reference = unused_phase_reference();

  switch (get_phase_reference(&time_offset, kicker->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!kicker->fiducial_seen) {
      /* set reference phase so that the center of this bunch goes through
       * at the desired phase.
       */
      t0 = 0;
      for (ip = t0 = 0; ip < np; ip++)
        t0 += part[ip][4] / (c_mks * beta_from_delta(p_central, part[ip][5]));
#if (!USE_MPI)
      t0 /= np;
#else
      if (notSinglePart) {
        double t0_sum;
        long np_total;

        MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
        MPI_Allreduce(&t0, &t0_sum, 1, MPI_DOUBLE, MPI_SUM, workers);
        t0 = t0_sum / np_total;
      } else
        t0 /= np;
#endif
      kicker->t_fiducial = t0;
      kicker->fiducial_seen = 1;
    }
    set_phase_reference(kicker->phase_reference, time_offset = kicker->t_fiducial);
    break;
  default:
    bombElegant("unknown return value from get_phase_reference()", NULL);
    break;
  }

  if (kicker->length <= 0)
    return;

  time_offset += kicker->time_offset;

  if ((n_parts = ceil(kicker->n_kicks / 4.0)) < 1)
    n_parts = 1;

  if (kicker->dx || kicker->dy || kicker->dz)
    offsetBeamCoordinatesForMisalignment(part, np, kicker->dx, kicker->dy, kicker->dz);
  if (kicker->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, kicker->tilt);

  for (ip = 0; ip < np; ip++) {
    strength = kicker->strength;
    time = part[ip][4] / (c_mks * beta_from_delta(p_central, part[ip][5])) - time_offset;

    if (time > kicker->tmax || time < kicker->tmin) {
      if (kicker->periodic) {
        n = (time - kicker->tmin) / (kicker->tmax - kicker->tmin);
        time -= n * (kicker->tmax - kicker->tmin);
      }
    }

    amplitude = 0;
    if (time > kicker->tmax || time < kicker->tmin)
      strength *= 0;
    else {
      /* should to a binary search here ! */
      for (i = 1; i < kicker->n_wf; i++)
        if (kicker->t_wf[i] >= time)
          break;
      if (i == kicker->n_wf) {
        printf("error: waveform interpolation problem in track_through_mkicker()\n");
        fflush(stdout);
        printf("particle time is %21.15e\nwaveform limits are %21.15e to %21.15e\n",
               time, kicker->tmin, kicker->tmax);
        fflush(stdout);
        exitElegant(1);
      }
      i--;
      strength *= (amplitude = INTERPOLATE(kicker->amp_wf[i], kicker->amp_wf[i + 1],
                                           kicker->t_wf[i], kicker->t_wf[i + 1], time));
    }

    KnL[0] = strength * kicker->length;
    multipoleOrder[0] = kicker->order;

    integrate_kick_multipole_ordn(part[ip], kicker->dx, kicker->dy, 0.0, 0.0, p_central, 0.0, 0.0,
                                  multipoleOrder, KnL, skew, n_parts, -1,
                                  kicker->length, 4, NULL, NULL, NULL, NULL, &dummy, NULL, 0, 0.0);
  }

  if (kicker->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, -kicker->tilt);
  if (kicker->dx || kicker->dy || kicker->dz)
    offsetBeamCoordinatesForMisalignment(part, np, -kicker->dx, -kicker->dy, -kicker->dz);
}

void set_up_mkicker(MKICKER *kicker) {
  TABLE data;
  long i;

  if (!kicker->waveform)
    bombElegant("no waveform filename given for MBUMPER", NULL);

  if (!getTableFromSearchPath(&data, kicker->waveform, 1, 0))
    bombElegant("unable to read waveform for MBUMPER", NULL);

  if (data.n_data <= 1)
    bombElegant("kicker waveform contains less than 2 points", NULL);

  kicker->t_wf = data.c1;
  kicker->amp_wf = data.c2;
  kicker->n_wf = data.n_data;
  for (i = 0; i < kicker->n_wf - 1; i++)
    if (kicker->t_wf[i] > kicker->t_wf[i + 1])
      bombElegant("time values not monotonically increasing in MBUMPER waveform", NULL);
  kicker->tmin = kicker->t_wf[0];
  kicker->tmax = kicker->t_wf[kicker->n_wf - 1];
  tfree(data.xlab);
  tfree(data.ylab);
  tfree(data.title);
  tfree(data.topline);
  data.xlab = data.ylab = data.title = data.topline = NULL;
}

void initializeDeflectionMap(KICKER *kicker) {
  SDDS_DATASET SDDSin;
  double *x = NULL, *y = NULL, *xpFactor = NULL, *ypFactor = NULL;
  long nx;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, kicker->deflectionMap) ||
      SDDS_ReadPage(&SDDSin) <= 0 ||
      !(x = SDDS_GetColumnInDoubles(&SDDSin, "x")) || !(y = SDDS_GetColumnInDoubles(&SDDSin, "y"))) {
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (!check_sdds_column(&SDDSin, "x", "m") ||
      !check_sdds_column(&SDDSin, "y", "m")) {
    fprintf(stderr, "KICKER deflection map input file must have x and y in m (meters)\n");
    exitElegant(1);
  }

  if (!(xpFactor = SDDS_GetColumnInDoubles(&SDDSin, "xpFactor")) || !(ypFactor = SDDS_GetColumnInDoubles(&SDDSin, "ypFactor"))) {
    fprintf(stderr, "KICKER input file must have both (xpFactor, ypFactor)\n");
    exitElegant(1);
  }
  if (!check_sdds_column(&SDDSin, "xpFactor", "") ||
      !check_sdds_column(&SDDSin, "ypFactor", "")) {
    fprintf(stderr, "KICKER deflection map input file must have xpFactor and ypFactor with no units\n");
    exitElegant(1);
  }

  if (!(kicker->points = SDDS_CountRowsOfInterest(&SDDSin)) || kicker->points < 2) {
    printf("file %s for KICKER element has insufficient data\n", kicker->deflectionMap);
    fflush(stdout);
    exitElegant(1);
  }
  SDDS_Terminate(&SDDSin);

  /* It is assumed that the data is ordered so that x changes fastest.
   * This can be accomplished with sddssort -column=y,incr -column=x,incr
   * The points are assumed to be equipspaced.
   */
  nx = 1;
  kicker->xmin = x[0];
  while (nx < kicker->points) {
    if (x[nx - 1] > x[nx])
      break;
    nx++;
  }
  if ((nx >= kicker->points) || nx <= 1 || y[0] > y[nx] || (kicker->ny = kicker->points / nx) <= 1) {
    printf("file %s for KICKER element doesn't have correct structure or amount of data\n",
           kicker->deflectionMap);
    fflush(stdout);
    printf("nx = %ld, ny=%ld\n", kicker->nx, kicker->ny);
    fflush(stdout);
    exitElegant(1);
  }
  kicker->nx = nx;
  kicker->xmax = x[nx - 1];
  kicker->dxg = (kicker->xmax - kicker->xmin) / (nx - 1);
  kicker->ymin = y[0];
  kicker->ymax = y[kicker->points - 1];
  kicker->dyg = (kicker->ymax - kicker->ymin) / (kicker->ny - 1);
  printf("KICKER element from file %s: nx=%ld, ny=%ld, dxg=%e, dyg=%e, x:[%e, %e], y:[%e, %e]\n",
         kicker->deflectionMap, kicker->nx, kicker->ny, kicker->dxg, kicker->dyg,
         kicker->xmin, kicker->xmax,
         kicker->ymin, kicker->ymax);
  free(x);
  free(y);
  kicker->xpFactor = xpFactor;
  kicker->ypFactor = ypFactor;
  kicker->initialized = 1;
}

long interpolateDeflectionMap(double *xpFactor, double *ypFactor, KICKER *kicker, double x, double y) {
  double Fa, Fb, fx, fy;
  long ix, iy;

  *xpFactor = *ypFactor = 0;

  if (isnan(x) || isnan(y) || isinf(x) || isinf(y))
    return 0;

  ix = (x - kicker->xmin) / kicker->dxg;
  iy = (y - kicker->ymin) / kicker->dyg;
  if (ix < 0 || iy < 0 || ix > kicker->nx - 1 || iy > kicker->ny - 1)
    return 0;

  fx = (x - (ix * kicker->dxg + kicker->xmin)) / kicker->dxg;
  fy = (y - (iy * kicker->dyg + kicker->ymin)) / kicker->dyg;

  Fa = (1 - fy) * kicker->xpFactor[ix + iy * kicker->nx] + fy * kicker->xpFactor[ix + (iy + 1) * kicker->nx];
  Fb = (1 - fy) * kicker->xpFactor[ix + 1 + iy * kicker->nx] + fy * kicker->xpFactor[ix + 1 + (iy + 1) * kicker->nx];
  *xpFactor = (1 - fx) * Fa + fx * Fb;

  Fa = (1 - fy) * kicker->ypFactor[ix + iy * kicker->nx] + fy * kicker->ypFactor[ix + (iy + 1) * kicker->nx];
  Fb = (1 - fy) * kicker->ypFactor[ix + 1 + iy * kicker->nx] + fy * kicker->ypFactor[ix + 1 + (iy + 1) * kicker->nx];
  *ypFactor = (1 - fx) * Fa + fx * Fb;

  return 1;
}
