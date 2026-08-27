/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: copy_particles()
 * purpose: copy particle data from one array to another
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

void copy_particles(
  double **copy,
  double **original,
  int64_t n_particles) {
  register int64_t ip;
  register long ic;
  double *cptr, *optr;
  char buffer[1024];
  
  log_entry("copy_particles");
  if (!copy)
    bombElegant("can't copy particles--target is NULL pointer (copy_particles)", NULL);
  if (!original)
    bombElegant("can't copy particles--source is NULL pointer (copy_particles)", NULL);

  for (ip = n_particles - 1; ip >= 0; ip--) {
    cptr = *copy++;
    optr = *original++;
    if (!cptr) {
      snprintf(buffer, 1024, "element %ld of target array is NULL pointer (copy_particles)", ip);
      bombElegant(buffer, NULL);
    }
    if (!optr) {
      snprintf(buffer, 1024, "element %ld of source array is NULL pointer (copy_particles)", ip);
      bombElegant(buffer, NULL);
    }
    for (ic = 6; ic >= 0; ic--)
      *cptr++ = *optr++;
  }
  log_exit("copy_particles");
}
