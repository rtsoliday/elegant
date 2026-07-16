#ifndef GPU_CWAKE_H
#define GPU_CWAKE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_cwake(long np, void *cwakeData, double *PoInput,
                             void *run, long iPass, void *charge);
long gpu_cwake_bunched_mode_action(long np, void *cwakeData, void *charge);

#ifdef __cplusplus
}
#endif

#endif
