#ifndef GPU_TUNE_H
#define GPU_TUNE_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_batched_tune_tracking_enabled(long particles);
long gpu_batched_tune_beamline_supported(void *beamline);
long gpu_batched_frequency_map_beamline_supported(void *beamline);
long gpu_batched_frequency_map_cpu_tracking_required(void *beamline);
/* A batch tracked for tune histories by the fused program, one kernel launch a turn: see gpu_stub.c.
 * Coordinates are particle-major with `stride` values per row; history has 4 blocks of
 * particles * turns.  Returns 0 when the caller must track on the CPU. */
long gpu_batched_tune_program_track(
  void *beamline, void *run, double pCentral, const double *coordinate,
  long particles, long stride, long turns, long turnOffset, double *history,
  double *ending, long *alive, long *exact);
/* Reports the caller's check of those results against the CPU; returns 1 if they may be used */
long gpu_batched_tune_program_checked(long particles, long checked, long exact, long lossMismatches,
                                      long keptButLost, long differing, double maxDifference);
double gpu_batched_tune_history_tolerance(void);
/* Whether a batch through this beamline can use the fused program's histories */
long gpu_batched_tune_program_beamline(void *beamline, void *run);
/* Threads for the tune analysis of a batch's histories: 1 unless the run set OMP_NUM_THREADS,
 * -ompThreads or ELEGANT_GPU_BATCHED_TUNE_THREADS; 16 at most */
long gpu_batched_tune_analysis_threads(void);
/* Keeps tracking calls on the CPU while the program's results are checked; returns the setting it
 * replaced, to be restored with a second call */
long gpu_batched_tune_cpu_scope(long cpuOnly);
void gpu_batched_tune_tracking_set_cpu_only(long cpuOnly);
/* While active, an RFCA takes each particle of a batch as a beam of its own and sets its reference again
 * for it at every call, as elegant's one-particle, one-turn do_tracking() calls do when
 * reset_rf_for_each_step is on.  Pass the beamline to turn it on if its fiducial flag asks for the
 * reset, NULL to turn it off. */
void gpu_batched_tune_own_rf_reference(void *beamline);
void gpu_batched_tune_tracking_report(const char *operation, long particles,
                                      long turns, long intervals);
/* A batched tune history recorded on the device: configure before tracking, record the
 * coordinates after each turn from a tracking wedge (0 when the beam isn't on the device),
 * fetch the recorded turns into a host history of 4 blocks of particles * turns, and clear */
void gpu_tune_history_configure(long particles, long turns);
long gpu_tune_history_record(long np, long turn);
void gpu_tune_history_fetch(double *history);
void gpu_tune_history_clear(void);

#ifdef __cplusplus
}
#endif

#endif
