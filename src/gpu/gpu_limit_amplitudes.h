#ifndef GPU_LIMIT_AMPLITUDES_H
#define GPU_LIMIT_AMPLITUDES_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_rectangular_collimator(void *rcol, long np, double **accepted,
                                double z, double Po, void *eptr);
long gpu_limit_amplitudes(double xmax, double ymax, long np, double **accepted,
                          double z, double Po, long extrapolate_z, long openCode,
                          void *eptr);
long gpu_removeInvalidParticles(long np, double **accepted, double z, double Po);
long gpu_elliptical_collimator(void *ecol, long np, double **accepted,
                               double z, double Po, void *eptr);
long gpu_elimit_amplitudes(double xmax, double ymax, long np, double **accepted,
                           double z, double Po, long extrapolate_z,
                           long openCode, long exponent, long yExponent, void *eptr);
long gpu_beam_scraper(void *scraper, long np, double **accepted,
                      double z, double Po, void *eptr);
long gpu_imposeApertureData(long np, double **accepted, double z, double Po,
                            void *apData, void *eptr);

#ifdef __cplusplus
}
#endif

#endif
