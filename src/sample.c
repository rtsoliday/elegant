/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: sample_particles.c
 * purpose: sample the incoming particle distribution to produce the
 *          outgoing distribution
 *
 * Michael Borland, 1991
 */
#include "mdb.h"
#include "track.h"

long sample_particles(double **initial, SAMPLE *samp, int64_t np, double **accepted, double z, double p0) {
  long itop, ic;
  int64_t ip;

  log_entry("sample_particles");

  itop = np - 1;
  if (samp->interval != 1) {
    for (ip = ic = 0; ip < np; ip++, ic++) {
      if (ic % samp->interval != 0) {
        swapParticles(initial[ip], initial[itop]);
        if (accepted)
          swapParticles(accepted[ip], accepted[itop]);
        initial[itop][4] = z; /* record position of particle loss */
        initial[itop][5] = p0 * (1 + initial[itop][5]);
        --itop;
        --ip;
        --np;
      }
    }
  } else if (samp->fraction != 1) {
    for (ip = 0; ip < np; ip++) {
      if (random_2(1) > samp->fraction) {
        swapParticles(initial[ip], initial[itop]);
        if (accepted)
          swapParticles(accepted[ip], accepted[itop]);
        initial[itop][4] = z; /* record position of particle loss */
        initial[itop][5] = p0 * (1 + initial[itop][5]);
        --itop;
        --ip;
        --np;
      }
    }
  }
  log_exit("sample_particles");
  return (np);
}
