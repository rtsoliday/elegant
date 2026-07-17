#ifndef GPU_SEARCH_H
#define GPU_SEARCH_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_search_tracking_enabled(long particles);
long gpu_batched_search_beamline_supported(void *beamline);
void gpuSetTrackingSuppressed(long suppressed);
void gpu_configure_batched_momentum_search(const double *deltaById,
                                           const long *targetById,
                                           long particles, long turns,
                                           long firePass,
                                           double *history,
                                           double *historyCount);
long gpu_apply_batched_momentum_search(long particles, long pass,
                                       long target, double dx, double dy,
                                       double pCentral);
void gpu_clear_batched_momentum_search(void);

#ifdef __cplusplus
}
#endif

#endif
