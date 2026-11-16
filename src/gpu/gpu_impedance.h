#ifndef GPU_IMPEDANCE_H
#define GPU_IMPEDANCE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_impedance(long np, void *impedanceData, double Po,
                                 void *run, long iPass, void *charge);
long gpu_impedance_bunched_mode_action(long np, void *impedanceData,
                                       void *charge);
/* ZLONGIT and ZTRANSVERSE run on the same kernels (gpu_stub.c) */
void gpu_track_through_zlongit(long np, void *zlongit, double Po, void *run, long iPass,
                               void *charge);
long gpu_zlongit_bunched_mode_action(long np, void *zlongit, void *charge);
void gpu_track_through_ztransverse(long np, void *ztransverse, double Po, void *run, long iPass,
                                   void *charge);
long gpu_ztransverse_bunched_mode_action(long np, void *ztransverse, void *charge);
/* IBSCATTER (SMOOTH=1) shares their bunch plan */
void gpu_track_IBS(long np, void *ibs, double Po, void *run, void *charge);
long gpu_ibs_bunched_mode_action(long np, void *ibs, void *charge);
/* LRWAKE: the bunch sums on the host in particle order, elegant's history and wake sums, the kicks on the
 * device; 0 when the CPU must do the element */
long gpu_track_through_lrwake(long np, void *lrwake, double P0, void *run, long iPass, void *charge,
                              long *lastNBunches);

#ifdef __cplusplus
}
#endif

#endif
