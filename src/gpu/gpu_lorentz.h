#ifndef GPU_LORENTZ_H
#define GPU_LORENTZ_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_lorentz(long nParticles, void *field, long fieldType,
                       double pCentral);

#ifdef __cplusplus
}
#endif

#endif
