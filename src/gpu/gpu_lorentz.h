#ifndef GPU_LORENTZ_H
#define GPU_LORENTZ_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_lorentz(long nParticles, void *field, long fieldType,
                       double pCentral);
/* LSRMDLTR (motion.cc) with the non-adaptive Runge-Kutta integrator; 0 when the CPU must track
 * (the device's coordinates are then those before the element) */
long gpu_track_lsrmdltr(long nParticles, void *lsrMdltr, double pCentral);

#ifdef __cplusplus
}
#endif

#endif
