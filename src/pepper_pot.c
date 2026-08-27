/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: pepper_pot_plate()
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

#if !defined(SWAP_PTR)
double *temp_ptr;
#  define SWAP_PTR(x, y) (temp_ptr = (x), (x) = (y), (y) = temp_ptr)
#endif

long pepper_pot_plate(double **initial, PEPPOT *peppot, int64_t np, double **accepted) {
  double length, *ini;
  int64_t ip, itop;
  long i_hole, n_holes;
  double radius_2;
  double x_center, y_center;
  double transmission, scatter, rotate;

  log_entry("pepper_pot_plate");

  itop = np - 1;
  n_holes = peppot->n_holes;
  radius_2 = sqr(peppot->radii);

  if (peppot->length)
    transmission = sqrt(peppot->transmission);
  else
    transmission = peppot->transmission;

  for (ip = 0; ip < np; ip++) {
    ini = initial[ip];
    for (i_hole = 0; i_hole < n_holes; i_hole++) {
      x_center = peppot->x[i_hole];
      y_center = peppot->y[i_hole];
      if (radius_2 > sqr(ini[0] - x_center) + sqr(ini[2] - y_center))
        break;
    }
    if (i_hole == n_holes) {
      if (transmission == 0 || random_2(0) > transmission) {
        if (itop != ip) {
          swapParticles(initial[ip], initial[itop]);
          if (accepted)
            swapParticles(accepted[ip], accepted[itop]);
          --itop;
          --ip;
        }
        np--;
      } else if (transmission && peppot->theta_rms) {
        /* particle made it through material--scatter it */
        scatter = gauss_rn(0, random_2) * peppot->theta_rms;
        ini[1] += tan(scatter * cos(rotate = random_2(1) * PIx2));
        ini[3] += tan(scatter * sin(rotate));
      }
    }
  }
  if (np == 0 || (length = peppot->length) <= 0) {
    log_exit("pepper_pot_plate");
    return (np);
  }

  itop = np - 1;
  for (ip = 0; ip < np; ip++) {
    ini = initial[ip];
    ini[0] += length * ini[1];
    ini[2] += length * ini[3];
    for (i_hole = 0; i_hole < n_holes; i_hole++) {
      x_center = peppot->x[i_hole];
      y_center = peppot->y[i_hole];
      if (radius_2 > sqr(ini[0] - x_center) + sqr(ini[2] - y_center))
        break;
    }
    if (i_hole == n_holes) {
      if (transmission == 0 || random_2(1) > transmission) {
        if (itop != ip) {
          swapParticles(initial[ip], initial[itop]);
          if (accepted)
            swapParticles(accepted[ip], accepted[itop]);
          --itop;
          --ip;
        }
        np--;
      } else if (transmission && peppot->theta_rms) {
        /* particle made it through material--scatter it */
        scatter = gauss_rn(0, random_2) * peppot->theta_rms;
        ini[1] += tan(scatter * cos(rotate = random_2(1) * PIx2));
        ini[3] += tan(scatter * sin(rotate));
      }
    }
  }
  log_exit("pepper_pot_plate");
  return (np);
}
