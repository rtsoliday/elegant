#ifndef GPU_WAKE_H
#define GPU_WAKE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_wake(long np, void *wakeData, double *PoInput,
                            void *run, long i_pass, void *charge);
long gpu_wake_bunched_mode_action(long np, void *wakeData, void *charge);
long gpu_wake_bunched_mode_supported(long np, void *wakeData, void *charge);

#ifdef __cplusplus
}
#endif

#endif
