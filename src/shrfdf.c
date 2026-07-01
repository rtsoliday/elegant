/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: track_through_space_harmonic_deflector()
 * purpose: track particles through an RF deflector with space harmonics
 * 
 * Michael Borland, 1989
 * space harmonic by Yipeng Sun, 2019
 */
#include "mdb.h"
#include "track.h"

void set_up_shrfdf(SHRFDF *rf_param, double **initial, long n_particles, double pc_central);

void track_through_space_harmonic_deflector(
  double **final,
  SHRFDF *rf_param,
  double **initial,
  long n_particles,
  double pc_central) {
  double t_first; /* time when first particle reaches cavity */
  double x, xp, y, yp;
  double beta, px, py, pz, pc;
  double omega, k, t_part;
  double k_0, alpha_sq;
  double dpx, dpy, dpz, phase;
  long ip, mode;

  /*
  This is always false
  if (rf_param->frequency==0) 
    bombElegant("SHRFDF cannot have frequency=0", NULL);
  */
  if (rf_param->factor == 0)
    return;

  if (!rf_param->initialized || !rf_param->fiducial_seen)
    set_up_shrfdf(rf_param, initial, n_particles, pc_central);

  t_first = rf_param->t_first_particle;

  if (rf_param->tilt)
    rotateBeamCoordinatesForMisalignment(initial, n_particles, rf_param->tilt);

  for (ip = 0; ip < n_particles; ip++) {
    x = initial[ip][0];
    xp = initial[ip][1];
    y = initial[ip][2];
    yp = initial[ip][3];
    pc = pc_central * (1 + initial[ip][5]);
    pz = pc / sqrt(1 + sqr(xp) + sqr(yp));
    px = xp * pz;
    py = yp * pz;
    beta = pc / sqrt(1 + sqr(pc));
    t_part = initial[ip][4] / (c_mks * beta);
    dpx = dpy = dpz = 0;

    /* for fundamental mode, n=0 */
    k_0 = rf_param->period_phase / rf_param->period_length;
    omega = k_0 * c_mks;
    phase = omega * (t_part - t_first) + rf_param->phase[0];
    dpx += rf_param->v[0] * sin(phase);
    dpz += rf_param->v[0] * cos(phase) * k_0 * x;

    /* for mode space harmonics, n=1-9 */
    for (mode = 1; mode < 10; mode++) {
      if (rf_param->v[mode] == 0)
        continue;

      k = (rf_param->period_phase + 2 * PI * mode) / rf_param->period_length;
      alpha_sq = k * k - k_0 * k_0;
      omega = k * c_mks;
      phase = omega * (t_part - t_first) + rf_param->phase[mode];
      dpx += rf_param->v[mode] * sin(phase) * (0.5 + 0.0625 * alpha_sq * (3.0 * x * x + y * y));
      dpz += rf_param->v[mode] * k * x * cos(phase) * (0.5 + 0.0625 * alpha_sq * (x * x + y * y));
    }
    /* We assume the deflection is from magnetic fields, so p^2 doesn't change */
    px += dpx * rf_param->factor * particleCharge / (particleMass * sqr(c_mks));
    py += dpy * rf_param->factor * particleCharge / (particleMass * sqr(c_mks));
    pz = sqrt(sqr(pc) - sqr(px) - sqr(py));

    /* Now add the effect of longitudinal field */
    pz += dpz * rf_param->factor * particleCharge / (particleMass * sqr(c_mks));
    pc = sqrt(sqr(px) + sqr(py) + sqr(pz));

    xp = px / pz;
    yp = py / pz;

    beta = pc / sqrt(1 + sqr(pc));
    final[ip][0] = x;
    final[ip][1] = xp;
    final[ip][2] = y;
    final[ip][3] = yp;
    final[ip][4] = t_part * c_mks * beta;
    final[ip][5] = (pc - pc_central) / pc_central;
    final[ip][6] = initial[ip][6];
  }

  if (rf_param->tilt)
    rotateBeamCoordinatesForMisalignment(initial, n_particles, -rf_param->tilt);
}

void set_up_shrfdf(SHRFDF *rf_param, double **initial, long n_particles, double pc_central) {
  long ip;
  double pc, beta;
#ifdef USE_KAHAN
  double error = 0.0;
#endif

  if (!rf_param->fiducial_seen) {
    if (isSlave || !distributedBeam) {
      for (ip = rf_param->t_first_particle = 0; ip < n_particles; ip++) {
        pc = pc_central * (1 + initial[ip][5]);
        beta = pc / sqrt(1 + sqr(pc));
#ifndef USE_KAHAN
        rf_param->t_first_particle += initial[ip][4] / beta / c_mks;
#else
        rf_param->t_first_particle = KahanPlus(rf_param->t_first_particle, initial[ip][4] / beta / c_mks, &error);
#endif
      }
    }
#if USE_MPI
    if (USE_MPI && distributedBeam) {
      long n_total;
#  ifndef USE_KAHAN
      double tmp;
#  endif
      if (isMaster) {
        n_particles = 0;
        rf_param->t_first_particle = 0.0;
      }
#  ifndef USE_KAHAN
      MPI_Allreduce(&(rf_param->t_first_particle), &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      rf_param->t_first_particle = tmp;
#  else
      rf_param->t_first_particle = KahanParallel(rf_param->t_first_particle, error, MPI_COMM_WORLD);
#  endif

      MPI_Allreduce(&n_particles, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      n_particles = n_total;
    }
#endif
    if (n_particles)
      rf_param->t_first_particle /= n_particles;
    rf_param->fiducial_seen = 1;
  }
}
