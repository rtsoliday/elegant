#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_particles(void *M, long n_part);

#ifdef __cplusplus
}
#endif

#endif
