/* file: median_oag.c 
  * contents: compute_median(), compute_percentile(), compute_average(), compute_middle()
  * see also: find_XX() routines in rowmedian.c  These return the position of the
  *     median, etc. in parallel elegant
  * Yusong Wang, 2007
  */

#if USE_MPI

#  include "track.h"

long approximate_percentiles_p(double *position, double *percent, long positions, double *x, long n,
                               long bins) {
  double *hist, *cdf, xMin, xMax, xCenter, xRange;
  long i, j, k;

  hist = cdf = NULL;

  if (!distributedBeam)
    return approximate_percentiles(position, percent, positions, x, n, bins);

  if (bins < 2 || positions <= 0) /* In the parallel version, n=0 will not be a condition to return, as no particle on Master */
    return 0;

  if (!(hist = calloc(sizeof(*hist), bins)))
    return 0;
  if (isMaster && distributedBeam)
    n = 0;
  find_min_max(&xMin, &xMax, x, n);
  if (isMaster) {
    xMin = DBL_MAX;
    xMax = -DBL_MAX;
  }
  find_global_min_max(&xMin, &xMax, n, MPI_COMM_WORLD);

  xCenter = (xMax + xMin) / 2;
  xRange = (xMax - xMin) * (1 + 1. / bins) / 2;
  xMin = xCenter - xRange;
  xMax = xCenter + xRange;
  cdf = NULL;
  if (isSlave) {
    make_histogram(hist, bins, xMin, xMax, x, n, 1);
    cdf = NULL;
  } else {
    if (!(cdf = calloc(sizeof(*cdf), bins)))
      return 0;
  }

  MPI_Reduce(hist, cdf, bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (isMaster) {
    for (i = 1; i < bins; i++)
      cdf[i] += cdf[i - 1];
    for (i = 0; i < bins; i++)
      cdf[i] /= cdf[bins - 1];

    for (j = 0; j < positions; j++) {
      for (i = k = 0; i < bins; i++) {
        if (cdf[i] < percent[j] / 100.0)
          k = i;
        else
          break;
      }
      /* printf ("p%ld,cdf[%ld] = %lf %", j, k, cdf[k]*100); */
      position[j] = xMin + (k * (xMax - xMin)) / bins;
    }
    if (cdf)
      free(cdf);
  }
  MPI_Bcast(position, positions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (hist)
    free(hist);

  return 1;
}

#endif
