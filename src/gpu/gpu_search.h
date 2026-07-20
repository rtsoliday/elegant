#ifndef GPU_SEARCH_H
#define GPU_SEARCH_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_search_tracking_enabled(long particles);
long gpu_batched_search_beamline_supported(void *beamline);
long gpu_run_batched_aperture_search(
  void *beamline, double pCentral, long lines, long nx, long nSplits,
  double splitFraction, long nPasses, double xmax, double ymax,
  const double *orbit, const double *dxFactor, const double *dyFactor,
  double *xLimit, double *yLimit, double *xLost, double *yLost,
  double *sLost, long *lossPass, long *lossElement, long *originStable);
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
