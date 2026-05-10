#ifndef GPU_MULTIPOLE_H
#define GPU_MULTIPOLE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_multipole_tracking2(long n_part, void *elem, double p_error,
                             double Po, double **accepted, double z_start,
                             void *maxamp, void *apcontour, void *apFileData,
                             double *sigmaDelta2, long iSlice);
long gpu_multipole_tracking(long n_part, void *multipole, double p_error,
                            double Po, double **accepted, double z_start);

#ifdef __cplusplus
}
#endif

#endif
