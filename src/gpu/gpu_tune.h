#ifndef GPU_TUNE_H
#define GPU_TUNE_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_tune_tracking_enabled(long particles);
long gpu_batched_tune_beamline_supported(void *beamline);
long gpu_batched_frequency_map_beamline_supported(void *beamline);
long gpu_batched_frequency_map_cpu_tracking_required(void *beamline);
/*
 * Returns 1 when a fused fixed-slot loss prepass completed, or 0 when the
 * caller must use the established CPU path.  Coordinates are particle-major
 * with `stride` values per row.  `survived` is indexed by the original row.
 */
long gpu_batched_tune_loss_prepass(
  void *beamline, double pCentral, const double *coordinate, long particles,
  long stride, long turns, long turnOffset, long *survived,
  long *survivorCount);
void gpu_batched_tune_tracking_set_cpu_only(long cpuOnly);
void gpu_batched_tune_tracking_report(const char *operation, long particles,
                                      long turns, long intervals);

#ifdef __cplusplus
}
#endif

#endif
