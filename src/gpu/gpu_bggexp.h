#ifndef GPU_BGGEXP_H
#define GPU_BGGEXP_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_bggexp(long nParticles, const GPU_BGGEXP_DATA *data);

#ifdef __cplusplus
}
#endif

#endif
