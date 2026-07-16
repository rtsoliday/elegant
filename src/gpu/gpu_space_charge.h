#ifndef GPU_SPACE_CHARGE_H
#define GPU_SPACE_CHARGE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_scmult_linear_supported(long nParticles, long nBuckets, long nonlinear,
                                  double sliceDuration, long horizontal,
                                  long vertical);
long gpu_scmult_nonlinear_supported(long nParticles, long nBuckets,
                                     long nonlinear, double sliceDuration,
                                     long horizontal, long vertical);
long gpu_scmult_single_bunch_supported(long nParticles, long idSlotsPerBunch,
                                        long nonlinear, double sliceDuration,
                                        long horizontal, long vertical);
long gpu_scmult_can_initialize_on_gpu(long nParticles);
long gpu_scmult_can_skip_cpu(long nParticles, long idSlotsPerBunch);
long gpu_scmult_count_bunches(long nParticles, long idSlotsPerBunch,
                              long *nBuckets);
long gpu_scmult_compute_centroid_sigma(long nParticles, double Po,
                                        double *center, double *sigma);
void gpu_track_through_scmult_linear(long nParticles, double charge, double c1,
                                     long horizontal, long vertical,
                                     long uniformDistribution,
                                     const double *center, const double *sigma,
                                     double dmux, double dmuy, double betax,
                                     double betay);
void gpu_track_through_scmult_nonlinear(long nParticles, double charge,
                                        double c1, long horizontal,
                                        long vertical,
                                        long uniformDistribution,
                                        const double *center,
                                        const double *sigma, double dmux,
                                        double dmuy, double betax,
                                        double betay);

#ifdef __cplusplus
}
#endif

#endif
