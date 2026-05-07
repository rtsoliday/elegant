#ifndef GPU_LSC_H
#define GPU_LSC_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_lscdrift(long np, void *lscdrift, double Po, void *charge);

#ifdef __cplusplus
}
#endif

#endif
