#ifndef GPU_RFDF_H
#define GPU_RFDF_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_apply_rfdf(long nParticles, const GPU_RFDF_DATA *data);

#ifdef __cplusplus
}
#endif

#endif
