#ifndef GPU_WAKE_H
#define GPU_WAKE_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_track_through_wake(long np, void *wakeData, double *PoInput,
                            void *run, long i_pass, void *charge);
long gpu_wake_bunched_mode_action(long np, void *wakeData, void *charge);
long gpu_wake_bunched_mode_supported(long np, void *wakeData, void *charge);
/* The smallest and largest s (coordinate 4) of the beam on the device; CORGPIPE and CORGPLATES size their
 * wake tables from them */
void gpu_beam_s_range(long np, double *sMin, double *sMax);

#ifdef __cplusplus
}
#endif

#endif
