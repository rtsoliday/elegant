#ifndef GPU_CCBEND_H
#define GPU_CCBEND_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_through_ccbend(long n_part, void *eptr, void *ccbend,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp, void *apContour,
                              void *apFileData, long iPart,
                              long iFinalSlice);

#ifdef __cplusplus
}
#endif

#endif
