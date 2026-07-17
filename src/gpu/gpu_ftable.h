#ifndef GPU_FTABLE_H
#define GPU_FTABLE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_ftable(long nParticles, FTABLE *ftable, double pCentral);

#ifdef __cplusplus
}
#endif

#endif
