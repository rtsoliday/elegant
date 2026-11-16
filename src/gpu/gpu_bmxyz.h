#ifndef GPU_BMXYZ_H
#define GPU_BMXYZ_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_bmxyz(long nParticles, BMAPXYZ *bmxyz, double pCentral, MAXAMP *maxamp,
                     APCONTOUR *apcontour, APERTURE_DATA *apFileData, double zStart,
                     double **accepted, ELEMENT_LIST *eptr);

#ifdef __cplusplus
}
#endif

#endif
