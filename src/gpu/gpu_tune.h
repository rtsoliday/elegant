#ifndef GPU_TUNE_H
#define GPU_TUNE_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_tune_tracking_enabled(long particles);
long gpu_batched_tune_beamline_supported(void *beamline);
long gpu_batched_frequency_map_beamline_supported(void *beamline);
long gpu_batched_frequency_map_cpu_tracking_required(void *beamline);
void gpu_batched_tune_tracking_set_cpu_only(long cpuOnly);
void gpu_batched_tune_tracking_report(const char *operation, long particles,
                                      long turns, long intervals);

#ifdef __cplusplus
}
#endif

#endif
