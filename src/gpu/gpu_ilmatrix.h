#ifndef GPU_ILMATRIX_H
#define GPU_ILMATRIX_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the number of particles left; the lost ones follow them in the CPU's order */
long gpu_track_ilmatrix(long nParticles, double **accepted, const GPU_ILMATRIX_DATA *data);

#ifdef __cplusplus
}
#endif

#endif
