#ifndef GPU_TRWAKE_H
#define GPU_TRWAKE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_trwake(long np, void *wakeData, double Po,
                              void *run, long i_pass, void *charge);
long gpu_trwake_bunched_mode_action(long np, void *wakeData, void *charge);
long gpu_trwake_bunched_mode_supported(long np, void *wakeData, void *charge);

#ifdef __cplusplus
}
#endif

#endif
