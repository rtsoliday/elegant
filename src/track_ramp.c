/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: track_through_ramped_deflector()
 * purpose: track particles through a ramped electric field deflector
 * 
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

void track_through_ramped_deflector(
  double **final,
  RMDF *ramp_param,
  double **initial,
  int64_t n_particles,
  double pc_central) {
  double t_null;    /* time when field is zero, relative to first particle */
  double t_part;    /* time at which a particle enters plates */
  double Estrength; /* |e.V.ramp_time|/(gap.m.c^2) */
  double x, xp, y, yp, dxp, dyp;
  double beta, dbeta_r, dr, beta_z, gamma, pc;
  double cos_tilt, sin_tilt;
  double Estrengthp;
  double length, zstart, zend, zs2, ze2, f1, Zo;
  int64_t ip;
  long is;

  log_entry("track_through_ramped_deflector");

  if (!ramp_param->initialized) {
    ramp_param->initialized = 1;
    pc = pc_central * (1 + initial[0][5]);
    beta = pc / sqrt(1 + sqr(pc));
    beta_z = beta / (1 + sqr(initial[0][1]) + sqr(initial[0][3]));
    ramp_param->t_first_particle =
      (initial[0][4] / beta + ramp_param->length / (2 * beta_z)) / c_mks;
    if (ramp_param->n_sections < 1)
      ramp_param->n_sections = 10;
    ramp_param->initialized = 1;
    if (ramp_param->gap == 0 || ramp_param->ramp_time == 0)
      bombElegant("RMDF cannot have gap=0 or ramp_time=0", NULL);
  }

  t_null = ramp_param->t_first_particle + ramp_param->time_offset;
  cos_tilt = cos(ramp_param->tilt);
  sin_tilt = sin(ramp_param->tilt);
  Estrength = -particleCharge * ramp_param->voltage / (particleMass * pow3(c_mks) * ramp_param->ramp_time * ramp_param->gap);
  length = ramp_param->length / ramp_param->n_sections;

  for (ip = 0; ip < n_particles; ip++) {
    x = initial[ip][0] - ramp_param->dx;
    xp = initial[ip][1];
    y = initial[ip][2] - ramp_param->dy;
    yp = initial[ip][3];
    pc = pc_central * (1 + initial[ip][5]);
    beta = pc / (gamma = sqrt(1 + sqr(pc)));
    beta_z = beta / sqrt(1 + sqr(xp) + sqr(yp));
    t_part = initial[ip][4] / (c_mks * beta);
    Zo = (t_part - t_null) * beta_z * c_mks;
    zstart = 0;
    zend = length;
    for (is = 0; is < ramp_param->n_sections; is++, zstart = zend, zend += length) {
      dbeta_r = (Estrengthp = Estrength / (gamma * sqr(beta_z))) * (((ze2 = sqr(zend) / 2) + Zo * zend) -
                                                                    (f1 = (zs2 = sqr(zstart) / 2) + Zo * zstart));
      dr = Estrengthp / beta_z * ((ze2 * zend / 3 + Zo * ze2 - zend * f1) - (zs2 * zstart / 3 + Zo * zs2 - zstart * f1));
      dxp = dbeta_r * cos_tilt / beta_z;
      dyp = dbeta_r * sin_tilt / beta_z;
      t_part += length / (c_mks * beta_z);
      beta_z -=
        sqr(gamma) * pow3(beta_z) * (xp * dxp + yp * dyp) / (1 + sqr(beta_z * gamma));
      x += xp * length + dr * cos_tilt;
      y += yp * length + dr * sin_tilt;
      xp += dxp;
      yp += dyp;
      beta = beta_z * sqrt(1 + sqr(xp) + sqr(yp));
      pc = beta / sqrt(1 - sqr(beta));
      gamma = sqrt(1 + sqr(pc));
    }
    final[ip][0] = x + ramp_param->dx;
    final[ip][1] = xp;
    final[ip][2] = y + ramp_param->dy;
    final[ip][3] = yp;
    final[ip][4] = t_part * c_mks * beta;
    final[ip][5] = (pc - pc_central) / pc_central;
    final[ip][6] = initial[ip][6];
  }
  log_exit("track_through_ramped_deflector");
}
