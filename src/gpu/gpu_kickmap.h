#ifndef GPU_KICKMAP_H
#define GPU_KICKMAP_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_kickmap(double **particle, double **accepted, long nParticles,
                       double pRef, void *map, double zStart,
                       double *sigmaDelta2);
long gpu_track_undulator_kickmap(double **particle, double **accepted,
                                 long nParticles, double pRef, void *map,
                                 double zStart);

#ifdef __cplusplus
}
#endif

#endif
