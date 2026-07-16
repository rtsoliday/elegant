#ifndef GPU_TUNE_H
#define GPU_TUNE_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_tune_tracking_enabled(long particles);
long gpu_batched_tune_beamline_supported(void *beamline);

#ifdef __cplusplus
}
#endif

#endif
