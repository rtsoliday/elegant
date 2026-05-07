#ifndef GPU_COMPUTE_CENTROIDS_H
#define GPU_COMPUTE_CENTROIDS_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_compute_centroids(double *centroid, long n_part);
void gpu_accumulate_beam_sums(void *sums, long n_part, double p_central, double mp_charge,
                              double *timeValue, double tMin, double tMax,
                              long startPID, long endPID, unsigned long flags);
long gpu_reductions_enabled(long nParticles);

#ifdef __cplusplus
}
#endif

#endif
