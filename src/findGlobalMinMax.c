/* file: findGlobalMinMax.c
 * The purpose of this file is to define the functions to find the 
 * global minimum and maximum values across all the processors after
 * the local minimum and maximum values are found on each processor.
 * Y.Wang, 2006
 */

#include "track.h"

void find_global_min_max(double *min, double *max, int64_t np, MPI_Comm comm) {
  double local_min, local_max;

  if (!np) { /* define MAX and MIN for the processors with 0 particles */
    local_min = DBL_MAX;
    local_max = -DBL_MAX;
  } else {
    local_min = *min;
    local_max = *max;
  }
  MPI_Allreduce(&local_min, min, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&local_max, max, 1, MPI_DOUBLE, MPI_MAX, comm);
}
