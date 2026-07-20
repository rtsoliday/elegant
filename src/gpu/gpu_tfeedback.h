#ifndef GPU_TFEEDBACK_H
#define GPU_TFEEDBACK_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

double gpu_tfeedback_pickup_average(long nParticles, long coordinate);
void gpu_tfeedback_apply_kick(long nParticles, long pickupCoordinate,
                              long longitudinal, double kick);

#ifdef __cplusplus
}
#endif

#endif
