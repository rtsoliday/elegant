#ifndef GPU_RFMODE_H
#define GPU_RFMODE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_rfmode_single_bunch_supported(long nParticles,
                                        long bunchedBeamMode,
                                        void *charge);
double gpu_rfmode_time_mean(long nParticles, double pCentral);
long gpu_rfmode_histogram(long nParticles, double pCentral,
                          long bins, double tmin, double dt,
                          long *histogram, long *firstBin,
                          long *lastBin);
void gpu_rfmode_apply_kicks(long nParticles, double pCentral,
                            long bins, double tmin, double dt,
                            long firstBin, long lastBin,
                            long interpolate, long nCavities,
                            const double *voltage);
long gpu_trfmode_histogram(long nParticles, double pCentral,
                           long bins, double tmin, double dt,
                           double dx, double dy,
                           double *xsum, double *ysum,
                           unsigned long *histogram,
                           long *firstBin, long *lastBin);
void gpu_trfmode_apply_kicks(long nParticles, double pCentral,
                             long bins, double tmin, double dt,
                             long firstBin, long lastBin,
                             long interpolate, long nCavities,
                             const double *voltageX,
                             const double *voltageY,
                             const double *voltageZ);

#ifdef __cplusplus
}
#endif

#endif
