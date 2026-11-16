#ifndef GPU_CWIGGLER_H
#define GPU_CWIGGLER_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* If fieldConsumer isn't NULL, it receives FIELD_OUTPUT records for the particles in each batch, in order */
typedef void (*GPU_CWIGGLER_FIELD_CONSUMER)(void *cookie, const double *record, long nParticles, long rowsPerParticle);
void gpu_track_cwiggler(long nParticles, const GPU_CWIGGLER_DATA *data,
                        GPU_CWIGGLER_FIELD_CONSUMER fieldConsumer, void *cookie);

#ifdef __cplusplus
}
#endif

#endif
