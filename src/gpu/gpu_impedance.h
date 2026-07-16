#ifndef GPU_IMPEDANCE_H
#define GPU_IMPEDANCE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_impedance(long np, void *impedanceData, double Po,
                                 void *run, long iPass, void *charge);
long gpu_impedance_bunched_mode_action(long np, void *impedanceData,
                                       void *charge);

#ifdef __cplusplus
}
#endif

#endif
