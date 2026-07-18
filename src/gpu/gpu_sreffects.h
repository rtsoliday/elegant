#ifndef GPU_SREFFECTS_H
#define GPU_SREFFECTS_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_sreffects(long nParticles,
                         const GPU_SREFFECTS_DATA *data);

#ifdef __cplusplus
}
#endif

#endif
