#ifndef GPU_LGBEND_H
#define GPU_LGBEND_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_through_lgbend(long n_part, void *eptr, void *lgbend,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp, void *apContour,
                              void *apFileData, long iPart,
                              long iFinalSlice);

#ifdef __cplusplus
}
#endif

#endif
