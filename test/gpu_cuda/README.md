# CUDA GPU Benchmarks

This directory contains deterministic CPU benchmark inputs and comparison tools for the optional CUDA work.  The cases are intentionally small enough to smoke-test quickly, but the runner can scale the number of turns toward a one-minute CPU baseline before GPU work begins.

See [CUDA_GPU_SUPPORT.md](../../doc/CUDA_GPU_SUPPORT.md) for the CUDA build requirements, runtime controls, known CPU fallbacks, CI split, and release checklist.

## Cases

- `matrix`: ordinary matrix/drift tracking
- `exact_drift`: `EDRIFT` exact-drift tracking
- `phase2_helpers`: matrix-family elements plus `MALIGN`, `CENTER`, `ENERGY`, centroid, and beam-sum helper paths
- `phase2_time_center`: `CENTER,T=1` time-coordinate centering plus nearby matrix/helper paths
- `phase2_special_matrix`: special-cased matrix elements (`SOLE`, `ROTATE`, `HKICK`, and `VKICK`) that can stay GPU-resident when spin tracking and corrector radiation are inactive
- `phase2_matrix_extended`: additional matrix-tracking elements (`QUFRINGE`, `MAGNIFY`, `WIGGLER`, `LTHINLENS`, `LMIRROR`, `BEDGE`, `REFLECT`, and non-radiating combined `KICKER`)
- `phase2_residency`: matrix-tracking elements separated by coordinate-neutral `CHARGE`, `RECIRC`, `MARK`, `TRCOUNT`, and `FLOORELEMENT` elements
- `phase3_limit_amplitudes`: loose rectangular `MAXAMP` checks after GPU-resident matrix elements, exercising the Phase 3 no-loss aperture predicate
- `phase3_limit_loss`: tight rectangular `MAXAMP` checks after GPU-resident matrix elements, exercising Phase 3 CPU fallback for exact loss compaction by default and optional CUDA compaction when enabled
- `phase3_elimit_amplitudes`: loose elliptical `MAXAMP` checks after GPU-resident matrix elements, exercising the Phase 3 no-loss aperture predicate
- `phase3_elimit_loss`: tight elliptical `MAXAMP` checks after GPU-resident matrix elements, exercising Phase 3 CPU fallback for exact loss compaction by default and optional CUDA compaction when enabled
- `phase3_rcol`: loose rectangular collimators with GPU entrance/exit predicates and GPU drift
- `phase3_rcol_loss`: tight rectangular collimators exercising exact CPU fallback for loss compaction
- `phase3_ecol`: loose elliptical collimators with GPU entrance/exit predicates and GPU drift
- `phase3_ecol_loss`: tight elliptical collimators exercising exact CPU fallback for loss compaction
- `phase3_scraper`: loose one-sided `SCRAPER` checks with GPU entrance/exit predicates and GPU drift
- `phase3_scraper_loss`: tight one-sided `SCRAPER` checks exercising exact CPU fallback for loss compaction
- `phase15_aperture_data_loss`: interpolated physical aperture data with particle loss
- `phase15_remove_invalid_loss`: fixed SDDS-beam input with invalid coordinates passed through an identity `RFCA`
- `phase15_rcol_open_loss`: zero-length open-sided rectangular collimators with particle loss
- `phase15_ecol_mixed_loss`: mixed-exponent elliptical collimators with particle loss
- `phase15_ecol_open_global_loss`: open-sided mixed-exponent elliptical collimators with global loss-coordinate output
- `phase15_rcol_open_global_loss`: nonzero-length open-sided rectangular collimators with global loss-coordinate output
- `phase15_scraper_global_loss`: one-sided `SCRAPER` with global loss-coordinate output
- `phase15_scraper_two_sided_global_loss`: two-sided ideal `SCRAPER` with global loss-coordinate output
- `multipole`: deterministic Phase 4 `KQUAD`, `KSEXT`, and `KOCT`
- `phase17_multipole_misalignment`: deterministic Phase 17 original-mode `KQUAD`, `KSEXT`, and `KOCT` `DX`/`DY`/`DZ`/`TILT` misalignment coverage
- `phase4_dqcor`: deterministic Phase 4 `DQCOR` with normal/skew quad terms and steering kicks
- `phase17_dqcor_misalignment`: deterministic Phase 17 original-mode `DQCOR` `DX`/`DY`/`DZ`/`TILT` misalignment coverage
- `csbend`: deterministic Phase 4 non-CSR `CSBEND` with first-order edge focusing
- `phase17_csbend_misalignment`: deterministic Phase 17 original-mode non-CSR `CSBEND` `DX`/`DY`/`DZ`/`TILT`/`ETILT` misalignment coverage
- `phase4_csbend_expanded`: deterministic Phase 4 non-CSR `CSBEND` with first-order edge focusing and `EXPAND_HAMILTONIAN=1`
- `phase4_csbend_ho_edge`: deterministic Phase 4 non-CSR `CSBEND` with Brown-style higher-order `EDGE_EFFECTS=1` focusing
- `aperture_loss`: rectangular and elliptical collimators with particle loss
- `phase5_wake`: autoscaled single-bunch longitudinal `WAKE`
- `wake_trwake`: autoscaled `WAKE` and `TRWAKE`
- `wake_trwake_fixed_bins`: fixed-bin `WAKE` and `TRWAKE`
- `phase16_wake_smoothing`: fixed-bin smoothed longitudinal `WAKE`
- `phase16_wake_change_p0`: autoscaled longitudinal `WAKE` with `CHANGE_P0=1`
- `phase16_trwake_smoothing`: fixed-bin smoothed transverse `TRWAKE`
- `phase16_trwake_tilt`: fixed-bin tilted transverse `TRWAKE`
- `phase16_bunched_wake_single`: single-effective-bucket bunched `WAKE` and `TRWAKE`
- `phase16_bunched_wake_filter_skip`: bunched `WAKE` and `TRWAKE` filters that skip the only effective bucket
- `phase16_bunched_wake_multibucket_skip`: multi-bucket bunched `WAKE` and `TRWAKE` filters that skip every effective bucket
- `phase16_lsc_smoothing_filter`: fixed-bin `LSCDRIFT` with Savitzky-Golay smoothing and high-frequency cutoff filtering
- `phase16_lsc_kick_mode`: fixed-bin `LSCDRIFT` kick mode with `L=0,LEFFECTIVE>0`
- `phase16_lsc_auto_leffective`: fixed-bin `LSCDRIFT` with `AUTO_LEFFECTIVE=1`
- `phase16_lsc_backtrack`: fixed-bin backtracking `LSCDRIFT`
- `phase16_lsc_low_frequency_filter`: fixed-bin `LSCDRIFT` with low- and high-frequency cutoff filtering
- `lsc`: `LSCDRIFT`
- `csr`: `CSRCSBEND` and `CSRDRIFT`
- `phase6_csr_csbend`: isolated large-bin `CSRCSBEND`
- `phase6_csr_bins_512`: isolated `CSRCSBEND` with fewer bins
- `phase6_csr_bins_4096`: isolated `CSRCSBEND` with more bins
- `phase6_csr_short_bunch`: isolated `CSRCSBEND` with a shorter bunch
- `phase6_csr_long_bunch`: isolated `CSRCSBEND` with a longer bunch
- `phase14_csr_last_wake`: `CSRCSBEND` last-wake output plus `CSRDRIFT`
- `phase14_csr_filters`: `CSRCSBEND` high/low cutoff and wake-filter table plus `CSRDRIFT`
- `phase14_csr_saldin54`: `CSRCSBEND` last-wake handoff into `CSRDRIFT,USE_SALDIN54=1`
- `phase7_scmult_linear`: drift-only inserted linear `SCMULT`
- `phase18_rfca_thin`: zero-length thin `RFCA` after CPU fiducialization setup
- `phase19_matrix_load_balance`: matrix lattice with `load_balancing_on=1` for Pelegant fallback diagnostics
- `rfcw`: RF cavity wake path
- `lcls0`: production LCLS linac case from `/home/soliday/oag/apps/src/elegantTestSet/LCLS0`

Most included cases use fixed seeds, `tracking_updates=0`, `show_element_timing=1`, and `load_balancing_on=0`.  The Phase 19 `phase19_matrix_load_balance` case deliberately sets `load_balancing_on=1` to verify the Pelegant CPU fallback guard.  Stochastic radiation is disabled in the starter lattices.

## Running

Run a quick CPU smoke test:

```sh
./test/gpu_cuda/run_benchmarks.sh --quick
```

Run the CUDA binary after an opt-in CUDA build:

```sh
make -C src HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc
ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 \
  ./test/gpu_cuda/run_benchmarks.sh --quick \
  --elegant ./bin/Linux-x86_64-gpu/gpu-elegant \
  --label cuda-fallback-quick
```

Phase 2 enables the matrix, helper, and common reduction kernels at or above particle-count thresholds.  `ELEGANT_GPU_MIN_PARTICLES` remains the global fallback threshold, while `ELEGANT_GPU_MIN_MATRIX_PARTICLES`, `ELEGANT_GPU_MIN_HELPER_PARTICLES`, `ELEGANT_GPU_MIN_REDUCTION_PARTICLES`, `ELEGANT_GPU_MIN_APERTURE_PARTICLES`, and `ELEGANT_GPU_MIN_EXACT_DRIFT_PARTICLES` can tune individual paths; each path-specific value defaults to `ELEGANT_GPU_MIN_PARTICLES`.  Set `ELEGANT_GPU_MIN_PARTICLES=1` for small smoke tests; leave the default threshold in place for ordinary runs until the timing data is broader.  The first Phase 3 aperture slices keep loose rectangular and elliptical `MAXAMP`, `RCOL`, `ECOL`, and simple one-sided `SCRAPER` checks GPU-resident by counting losses on the device; if any particles would be lost, or if open-side/inverted/material-interaction apertures are used, they synchronize and run the exact CPU loss/compaction path by default.  Simple rectangular and elliptical `MAXAMP` loss compaction can be tested with `ELEGANT_GPU_ENABLE_APERTURE_COMPACTION=1`; it matched CPU output in smoke tests but is slower than the CPU fallback on current quick cases, so it remains opt-in.  The exact-drift kernel is correct but currently thresholded off by default because timing did not show a benefit; set `ELEGANT_GPU_ENABLE_EXACT_DRIFT=1` when explicitly testing it.  Helper scalar sums for `CENTER`, `MATR`/`EMATRIX` fiducials, and `ENERGY` matching use GPU reductions, so those helper elements require both the helper and reduction thresholds when the scalar sum is needed; ordinary-coordinate `CENTER` offsets are batched into one centroid reduction per element.  `MATR` and `EMATRIX` apply their fiducial `sReference` inside the CUDA matrix kernel, avoiding separate subtract/restore coordinate kernels.  When ordinary coordinate offsets and `CENTER,T=1` are both requested, a combined centroid/time reduction supplies both sets of sums; otherwise `CENTER,T=1` uses a dedicated GPU time-sum reduction.  `CENTER` coordinate updates and optional time-centering updates are fused into one helper kernel per element.  Generic `MATRIX_TRACKING` elements with ordinary first- through third-order matrices stay GPU-resident, including monitors and the extended matrix family above.  Unsupported matrix shapes or orders fall back to CPU before timing or kernel launch.  Coordinate-neutral `CHARGE`, `RECIRC`, `MARK`, `TRCOUNT`, and `FLOORELEMENT` elements preserve GPU residency only when a device particle buffer is already current; `MARK` fitpoint output still synchronizes to CPU before reading coordinates.  `SOLE`, `ROTATE`, `HKICK`, `VKICK`, and combined `KICKER` use the matrix GPU path when their follow-on CPU side effects are inactive; spin-tracking solenoids/rotations and radiating correctors still fall back to CPU.  Beam-sum reductions currently use the GPU only for the common no-filter/no-spin/no-exact-emittance path; specialized modes fall back to CPU synchronization.  Trajectory centroid collection uses the same GPU centroid reduction when tracking remains resident and at or above the reduction threshold.

Phase 2 is wrapped up for the current CUDA scope.  The final quick smoke sweep labels are `gpu-phase2-wrapup-matrix`, `gpu-phase2-wrapup-phase2_helpers`, `gpu-phase2-wrapup-phase2_time_center`, `gpu-phase2-wrapup-phase2_special_matrix`, `gpu-phase2-wrapup-phase2_matrix_extended`, `gpu-phase2-wrapup-phase2_residency`, and `gpu-phase2-wrapup-exact-drift-enabled`; each matched its CPU reference for all 4 SDDS files.  The next CUDA work should move to Phase 3 aperture/loss handling rather than continuing to tune Phase 2 helper launches.

Phase 3 is wrapped up for the current CUDA scope with `limit_amplitudes`, `elimit_amplitudes`, `rectangular_collimator`, `elliptical_collimator`, and simple `beam_scraper` support for aperture checks.  This is intentionally conservative: common loose-aperture checks avoid pulling otherwise resident particles back to the host, while default lossy paths preserve existing CPU-owned loss semantics.  Optional simple `MAXAMP` CUDA compaction preserves CPU swap order and then synchronizes back for lost-particle bookkeeping; use it only for targeted experiments until a faster parallel compaction path is available.  `removeInvalidParticles`, `imposeApertureData`, and faster parallel compaction are deferred Phase 3 follow-ups.  The `phase3_limit_loss`, `phase3_elimit_loss`, `phase3_rcol_loss`, `phase3_ecol_loss`, and `phase3_scraper_loss` cases cover the loss paths so the loss and final particle files are compared against CPU output.

Phase 15 starts the aperture/loss follow-up for simple rectangular and elliptical `MAXAMP` loss cases.  The exact opt-in compaction path behind `ELEGANT_GPU_ENABLE_APERTURE_COMPACTION=1` now reads back only lost tail rows in normal GPU builds.  A separate stable prefix-sum prototype is available with `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1`; it matched CPU output at `1e-11` for `phase3_limit_loss`, `phase3_elimit_loss`, `phase3_rcol_loss`, `phase3_ecol_loss`, `phase3_scraper_loss`, `phase15_aperture_data_loss`, `phase15_remove_invalid_loss`, `phase15_rcol_open_loss`, `phase15_ecol_mixed_loss`, `phase15_ecol_open_global_loss`, `phase15_rcol_open_global_loss`, `phase15_scraper_global_loss`, `phase15_scraper_two_sided_global_loss`, and production-style `maxamp1` output including `.acc` and `.los`.  It supports simple rectangular/elliptical `MAXAMP`, non-inverted `RCOL` including zero-length and nonzero-length open-sided `RCOL`, nonzero-length open-sided `RCOL` global loss-coordinate output, non-inverted even-exponent `ECOL` including open-sided and mixed-`YEXPONENT` `ECOL` with global loss-coordinate output, ideal `SCRAPER` including one-sided and two-sided global loss-coordinate output, interpolated rectangular `aperture_data` checks, and `removeInvalidParticles` for a narrow identity-`RFCA` trigger used to keep invalid-particle loss handling resident.  The stable scatter now promotes the scratch coordinate buffer to the resident device buffer instead of copying compacted rows back with a full device-to-device copy, and it scatters `accepted` rows on the device by default while parallel compaction is enabled; set `ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE=0` to disable the accepted-device path while debugging.  On the bounded `maxamp1` 3000-particle, 60-pass gate, CPU took 18.47 seconds, the fallback-heavy GPU path took 12.99 seconds, scratch-promotion stable compaction took 12.62 seconds, and accepted-device stable compaction took 12.55 seconds.  The focused open-`RCOL` 30,000-particle, 10-pass case took 4.14 seconds on CPU, 1.10 seconds with the fallback-heavy GPU path, and 0.96 seconds with stable open-`RCOL` compaction.  The focused open-`RCOL` global-loss-coordinate 30,000-particle, 10-pass case took 4.55 seconds on CPU and 1.05 seconds with stable CUDA compaction.  The focused one-sided `SCRAPER` global-loss-coordinate 30,000-particle, 10-pass case took 1.61 seconds on CPU and 0.53 seconds with stable CUDA compaction.  The focused two-sided `SCRAPER` global-loss-coordinate 30,000-particle, 10-pass case took 1.12 seconds on CPU and 0.45 seconds with stable CUDA compaction.  The focused mixed-exponent `ECOL` 30,000-particle, 10-pass case took 2.18 seconds on CPU and 0.61 seconds with stable CUDA compaction; the open-sided global-loss-coordinate ECOL variant took 2.25 seconds on CPU and 0.63 seconds with stable CUDA compaction.  The `collimate1`, `collimate2`, and `collimate3` production wrappers now cover open `RCOL`, open `ECOL`, nonzero-length open collimators, watch output, loss output, and acceptance output; they are correctness-clean but remain dominated by startup, watch output, and lost-tail synchronization.

Phase 4 is wrapped for the current CUDA scope with deterministic magnet support for simple `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, and non-CSR `CSBEND` elements.  It uses `ELEGANT_GPU_MIN_MAGNET_PARTICLES`, reports `magnets=` in CUDA timing, and keeps radiation/ISR, spin tracking, most magnet misalignments, KQUAD fringes/end drifts/radial mode, extra multipole files, aperture hooks, slice-by-slice tracking, CSBEND Hwang/Lindberg/curved fringe models, CSBEND reference correction, and detected losses on the CPU path.  CSBEND supports first-order `EDGE_EFFECTS=1` focusing, Brown-style higher-order `EDGE_EFFECTS=1` focusing including pole-face curvature terms, nonzero `E1`/`E2`, finite `HGAP`/`FINT` `psi` terms, edge kick limits for the first-order path, and deterministic expanded-Hamiltonian tracking.  The checked single-pass kernels preserve CPU fallback by backing up device particles before tracking and restoring them if invalid or lost particles are detected.  Current Phase 4 smoke labels include `gpu-phase4-multipole-checked`, `gpu-phase4-multipole-checked-verify`, `gpu-phase4-multipole-checked-baseline-60s`, `gpu-phase4-dqcor`, `gpu-phase4-dqcor-verify`, `gpu-phase4-csbend-edge`, `gpu-phase4-csbend-edge-verify`, `gpu-phase4-csbend-edge-baseline-60s`, `gpu-phase4-csbend-expanded`, `gpu-phase4-csbend-expanded-verify`, `gpu-phase4-csbend-expanded-baseline-60s`, `gpu-phase4-csbend-ho-edge`, `gpu-phase4-csbend-ho-edge-verify`, and `gpu-phase4-csbend-ho-edge-baseline-60s`; their CPU/GPU SDDS comparisons matched all 4 files.  Expanded-CSBEND baseline mode used 20000 particles, 99 CPU passes in 58.78 seconds and 787 CUDA passes in 35.28 seconds; high-order-edge baseline mode used 20000 particles, 66 CPU passes in 59.13 seconds and 676 CUDA passes in 39.28 seconds.  The common 8-pass samples matched all 4 files in both timing gates.  CSBEND Hwang/Lindberg/curved fringe models, reference/FSE correction, remaining magnet misalignments, radiation/stochastic effects, spin, sticky aperture hooks, slice-by-slice tracking, and corrector radiation kicks are deferred follow-ups.

Phase 17 starts the magnet-coverage expansion with narrow deterministic original-mode misalignment slices for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, and non-CSR `CSBEND`.  The CUDA path supports `DX`, `DY`, `DZ`, and `TILT` for multipoles and `DQCOR` when `MALIGN_METHOD=0`, `PITCH=0`, `YAW=0`, radiation/ISR and spin tracking are inactive, and no extra fringe/radial/multipole-file/aperture hooks are active.  Non-CSR `CSBEND` also supports simple original-mode `DX`, `DY`, `DZ`, `TILT`, and `ETILT`, while `EPITCH`, `EYAW`, nonzero `MALIGN_METHOD`, reference/FSE correction, advanced fringe models, radiation/spin, aperture hooks, and slice-by-slice tracking still fall back to CPU.  The `phase17_multipole_misalignment`, `phase17_dqcor_misalignment`, and `phase17_csbend_misalignment` cases matched CPU output for all 4 common SDDS files at `1e-11` in normal CUDA and `GPU_VERIFY` runs.  The multipole bounded timing gate used 30,000 particles and 23 passes: CPU took 59.97 seconds, while `gpu-elegant` took 7.23 seconds on the local RTX 3060, about 8.29x faster.  The DQCOR bounded timing gate used 30,000 particles and 12 passes: CPU took 58.59 seconds, while `gpu-elegant` took 4.46 seconds, about 13.14x faster.  The CSBEND misalignment gate used 30,000 particles and 11 passes: CPU took 57.36 seconds, while `gpu-elegant` took 4.53 seconds, about 12.66x faster.  Reports: `test/gpu_cuda/output/reports/phase17-multipole-misalignment-30k23.md`, `test/gpu_cuda/output/reports/phase17-dqcor-misalignment-30k12.md`, and `test/gpu_cuda/output/reports/phase17-csbend-misalignment-30k11.md`.

Phase 5 wrapped up with conservative `WAKE`, `TRWAKE`, and `LSCDRIFT` CUDA slices.  At that point, longitudinal `WAKE` supported `N_BINS=0` or fixed `N_BINS>=2`, `SMOOTHING=0`, `BUNCHED_BEAM_MODE=0`, and `CHANGE_P0=0`; transverse `TRWAKE` supported `N_BINS=0` or fixed `N_BINS>=2`, `SMOOTHING=0`, `BUNCHED_BEAM_MODE=0`, and `TILT=0`; `LSCDRIFT` supported fixed even `BINS>=2`, `LSC=1`, `SMOOTHING=0`, no frequency cutoffs, positive `L`, no backtracking, and no `AUTO_LEFFECTIVE`.  Phase 16 broadens the wake subset below.  The wake paths compute time-coordinate reductions on the device, perform bin assignment with explicit bin-edge handling, run wake convolution and kicks on CUDA, compare intermediate histograms and wake potentials in `GPU_VERIFY`, and report combined `wakes=` CUDA timing.  The first `LSCDRIFT` path keeps the small impedance FFT on the host for CPU-equivalent math while GPU kernels handle binning, histogram creation, kicks, and drift advancement; it reports `lsc=` timing and uses `ELEGANT_GPU_MIN_LSC_PARTICLES`.  Use `ELEGANT_GPU_MIN_WAKE_PARTICLES` to tune `WAKE`/`TRWAKE` independently.  Final Phase 5 timing labels include `gpu-phase5-wake-quick-device`, `gpu-phase5-wake-verify`, `gpu-phase5-wake-baseline-60s`, `gpu-phase5-wake-common-194`, `gpu-phase5-trwake-reduced-quick`, `gpu-phase5-trwake-reduced-verify`, `cpu-phase5-trwake-baseline-60s`, `gpu-phase5-trwake-reduced-baseline-60s`, `gpu-phase5-trwake-common-306`, `gpu-phase5-fixed-bins-1024-quick`, `gpu-phase5-fixed-bins-1024-verify`, `cpu-phase5-fixed-bins-1024-baseline-60s`, `gpu-phase5-fixed-bins-1024-baseline-60s`, `gpu-phase5-fixed-bins-1024-common-319`, `gpu-phase5-lsc-quick-device`, `gpu-phase5-lsc-verify`, `cpu-phase5-lsc-baseline-60s`, `gpu-phase5-lsc-baseline-60s`, and `gpu-phase5-lsc-common-303`.  The isolated 30000-particle `phase5_wake` CPU baseline used 194 passes in 54.96 seconds, while CUDA used 492 passes in 33.42 seconds; the common 194-pass CUDA run took 13.23 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.  The combined autoscaled `wake_trwake` CPU baseline used 306 passes in 48.15 seconds, while CUDA used 638 passes in 25.37 seconds; the common 306-pass CUDA run took 12.32 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.  The fixed-bin `wake_trwake_fixed_bins` CPU baseline used 319 passes in 49.87 seconds, while CUDA used 652 passes in 26.75 seconds; the common 319-pass CUDA run took 13.11 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.  The fixed-bin case preserves CPU behavior, including late-run TRWAKE warnings when the beam walks outside the fixed window.  The `lsc` CPU baseline used 303 passes in 44.67 seconds, while CUDA used 682 passes in 24.30 seconds; the common 303-pass CUDA run took 10.91 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.  At Phase 5 wrap, smoothed wakes, multi-bunch filtering, tilted `TRWAKE`, `LSCDRIFT` smoothing/frequency filters/backtracking/kick-mode/`AUTO_LEFFECTIVE`, `RFCW`, cuFFT-backed paths, and `gpu_findFiducialTime` semantics were deferred follow-ups.

Phase 16 starts the collective-effects follow-up by enabling smoothed single-bunch `WAKE` and `TRWAKE` for the existing supported binning subset.  CUDA still owns the particle-heavy binning, convolution, and kick stages; the first implementation applies the existing CPU Savitzky-Golay routine to the small histogram between CUDA stages to preserve smoothing semantics exactly.  `WAKE,CHANGE_P0=1` now uses the existing GPU central-momentum matching helper after the wake kick.  Tilted single-bunch `TRWAKE` is also supported when spin coordinates are inactive by rotating particles into the tilted wake frame for histogramming and kicks, then rotating back.  Single-effective-bucket `BUNCHED_BEAM_MODE=1` is now supported for serial/local `WAKE` and `TRWAKE`, including no-op filters that skip the only effective bucket; serial/local multi-bucket no-op filters are also supported when `START_BUNCH` is beyond the final effective bucket.  Multi-bunch filtering that selects any nonempty bucket, `END_BUNCH` filtering beyond the single effective bucket, and distributed Pelegant bunched wakes still force CPU fallback.  `LSCDRIFT` now supports Savitzky-Golay histogram smoothing, low- and high-frequency cutoff filtering, explicit kick mode with `L=0,LEFFECTIVE>0`, `AUTO_LEFFECTIVE` after elegant resolves the effective length, and backtracking; the small FFT/impedance calculation remains host-side for CPU-equivalent math.  `gpu_findFiducialTime` now supports `LIGHT` and serial/local full-beam `TMEAN` with `sOffset=0`, which keeps serial `modulate_elements` resident when the beam is already on the device.  Distributed Pelegant modulation still syncs to CPU for fiducial time, and RFCA/RFCW fiducialization remains deferred.  The new `phase16_wake_smoothing`, `phase16_wake_change_p0`, `phase16_trwake_smoothing`, `phase16_trwake_tilt`, `phase16_bunched_wake_single`, `phase16_bunched_wake_filter_skip`, `phase16_bunched_wake_multibucket_skip`, `phase16_lsc_smoothing_filter`, `phase16_lsc_kick_mode`, `phase16_lsc_auto_leffective`, `phase16_lsc_backtrack`, `phase16_lsc_low_frequency_filter`, and `phase16_fiducial_modulate` cases matched CPU at `1e-11` in normal CUDA and `GPU_VERIFY` builds, including intermediate histogram and wake-potential checks where applicable.  On bounded timings, smoothed `WAKE` took 1.08 seconds on CPU and 0.49 seconds on CUDA, `WAKE,CHANGE_P0=1` took 1.61 seconds on CPU and 0.63 seconds on CUDA, smoothed `TRWAKE` took 1.06 seconds on CPU and 0.49 seconds on CUDA, tilted `TRWAKE` took 1.11 seconds on CPU and 0.50 seconds on CUDA, single-effective-bucket bunched `WAKE`/`TRWAKE` used 268 CPU passes in 42.73 seconds and the same CUDA workload in 12.12 seconds, no-op filtered single-bucket bunched `WAKE`/`TRWAKE` used 40 CPU passes in 6.30 seconds and the same CUDA workload in 1.55 seconds, no-op filtered multi-bucket bunched `WAKE`/`TRWAKE` used 382 CPU passes in 58.62 seconds and the same CUDA workload in 11.88 seconds, smoothed/filtered `LSCDRIFT` took 0.89 seconds on CPU and 0.42 seconds on CUDA, kick-mode `LSCDRIFT` took 0.86 seconds on CPU and 0.43 seconds on CUDA, `AUTO_LEFFECTIVE` `LSCDRIFT` took 0.98 seconds on CPU and 0.49 seconds on CUDA, backtracking `LSCDRIFT` took 0.87 seconds on CPU and 0.47 seconds on CUDA, and low-frequency-filtered `LSCDRIFT` took 0.89 seconds on CPU and 0.44 seconds on CUDA.

Phase 6 is wrapped up for the current CUDA scope with a conservative `CSRCSBEND` wake-potential CUDA slice.  Histogramming, filtering, Savitzky-Golay smoothing, particle tracking, `CSRDRIFT`, Stupakov/Saldin drift state, and most `CSR_LAST_WAKE` bookkeeping remain CPU-owned, while CUDA computes the non-IGF CSRCSBEND `T1`, `T2`, and `dGamma` wake arrays.  Use `ELEGANT_GPU_MIN_CSR_PARTICLES` and `ELEGANT_GPU_MIN_CSR_BINS` to tune this path; the default CSR bin threshold is 1024, and CUDA timing reports `csr=`.  `GPU_VERIFY` compares the CSR arrays against a CPU shadow for each exercised wake calculation and, for the non-IGF path, compares the produced `CSR_LAST_WAKE` handoff state plus the CPU-shadow `dGamma`, including wake-filter-table postprocessing.

Current Phase 6 timing labels include `gpu-phase6-csr-quick-device`, `gpu-phase6-csr-verify`, `gpu-phase6-csr-csbend-quick`, `cpu-phase6-csr-csbend-baseline-60s`, `gpu-phase6-csr-csbend-baseline-60s`, `gpu-phase6-csr-csbend-common-273`, `gpu-phase6-csr-scratch-verify`, `gpu-phase6-csr-scratch-baseline-60s`, `gpu-phase6-csr-scratch-common-273`, `gpu-phase6-csr-hist-verify`, `gpu-phase6-csr-hist-baseline-60s`, `gpu-phase6-csr-hist-common-273`, `cpu-phase6-csr-regression-quick`, and `gpu-phase6-csr-regression-verify`.  The isolated 20000-particle `phase6_csr_csbend` CPU baseline used 273 passes in 37.01 seconds, while CUDA used 167 passes in 18.11 seconds; the common 273-pass CUDA run took 29.87 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.  Reusable CSR scratch buffers now remove the per-kick device allocation/free pairs and release at `gpuBaseDealloc`; correctness still matched CPU at `1e-11`, but timing was effectively flat, with the scratch baseline at 171 passes in 18.41 seconds and the common 273-pass scratch run at 29.98 seconds.  `ELEGANT_GPU_ENABLE_CSR_HISTOGRAM=1` enables an experimental fixed-bin raw histogram-fill path that keeps CPU range/bin sizing and SG preparation; it matched CPU at `1e-11`, but stayed opt-in because the 273-pass histogram run took 30.42 seconds.  Phase 14 improves this opt-in path by packing only the longitudinal coordinate before the CSR histogram host-to-device copy instead of staging full particle rows, packing CSR wake inputs into one upload, caching repeated CSR denominator prep/uploads, and returning CSR `T1`/`T2` arrays only for wake output or `GPU_VERIFY`.  `ELEGANT_GPU_ENABLE_CSR_KICK=1` enables a correct but currently slower host-packed CSR kick prototype; leave it off unless evaluating the next fully resident CSRCSBEND design.  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` enables the Phase 14 resident CSRCSBEND prototype, which keeps supported non-radiating, non-IGF CSRCSBEND body slices, coordinate-only ct range reduction, histogram fill, wake calculation, and CSR kicks on resident device coordinates while handing back to the CPU for final CSRDRIFT state and unsupported options.  On the 20,000-particle, 261-pass same-workload `phase6_csr_csbend` gate, the resident prototype improved from 16.24 seconds after body-data caching to 13.99 seconds after replacing the range beam-sums call with the coordinate-only min/max reduction, then to 13.45 seconds after preserving resident scratch across CSR wake-array growth and reusing CSRCSBEND body rollback/loss buffers.  A targeted resident-CSR short-island exception kept the simple matrix/drift island immediately before eligible CSRCSBEND sections on the GPU and brought the same gate to 9.82 seconds.  Skipping unnecessary CSRDRIFT handoff prep, using a checked CUDA simple-final-transform kernel, and keeping final coordinates GPU-resident when no immediate CSRDRIFT follows brought the gate to 8.46 seconds.  Adding a simple resident-entry path for no-offset/no-effective-entry-edge CSRCSBENDs avoids the remaining pre-CSRCSBEND CPU handoff when particles are already device-current; the current gate is 5.27 seconds, about 6.69x faster than the 35.24-second CPU reference, with all 4 common SDDS files matching at `1e-11`.

The `phase6_csr_bins_512`, `phase6_csr_bins_4096`, `phase6_csr_short_bunch`, and `phase6_csr_long_bunch` regression cases cover additional CSRCSBEND bin counts and bunch lengths; the CPU/GPU_VERIFY quick sweeps matched all 4 SDDS files per case at `1e-11`.  Phase 14 adds `phase14_csr_last_wake`, `phase14_csr_filters`, and `phase14_csr_saldin54` for last-wake output, CSRDRIFT, high/low cutoff filters, wake-filter-table postprocessing, Saldin54 drift-mode handoff, and aperture-state interaction around CSR sections.  Removing the remaining CPU final-transform and CSRDRIFT-state handoffs, broader resident CSR mode coverage, full `CSR_LAST_WAKE` coverage for IGF/Stupakov/Saldin modes, integrated Greens function, and broader production CSR regression coverage are deferred follow-ups.

Phase 7 is wrapped up for the current CUDA scope with an opt-in linear, unsliced, single-bucket `SCMULT` CUDA slice.  Enable it with `ELEGANT_GPU_ENABLE_SCMULT=1`; tune the threshold with `ELEGANT_GPU_MIN_SCMULT_PARTICLES`.  The current path uses CUDA beam-sum reductions for resident rms/centroid calculation, a CUDA bunch-index min/max reduction for first-pass single-bucket discovery, and resident `dmux`/`dmuy` accumulation; it also keeps `SCMULT` as a pass-through element when the device particle buffer is current.  CPU code still owns SCMULT insertion, nonlinear kicks, sliced kicks, and multi-bunch cases, so this path does not add the `GPU_SUPPORT` flag for `SCMULT` and is not automatic.  Current Phase 7 labels include `cpu-phase7-scmult-linear-drift-ref`, `gpu-phase7-scmult-linear-drift-verify`, `gpu-phase7-scmult-linear-drift-quick`, `cpu-phase7-scmult-linear-drift-baseline-60s`, `gpu-phase7-scmult-linear-drift-baseline-same`, `gpu-phase7-scmult-linear-init-verify`, `gpu-phase7-scmult-linear-init-quick`, and `gpu-phase7-scmult-linear-init-baseline-same`.  The verify run exercised 24 resident SCMULT CUDA kernels and matched all 5 common SDDS files at `1e-11`; the non-verify first-pass check showed no `initializeSCMULT` or `accumulateSCMULT` CPU synchronization, only final deallocation synchronization.  The 30000-particle CPU baseline scaled to 115 passes and ran in 53.47 seconds, while the current CUDA build ran the same 115-pass workload in 16.58 seconds with `SCMULT` at 6.887 seconds and matched CPU output at `1e-11`.  Sliced/nonlinear/multi-bunch SCMULT, broader production SCMULT profiling, cuFFT-backed Poisson solve work, field-map/wiggler profiling, and ion-effects profiling are deferred Phase 7 follow-ups; the next active CUDA work should move to Phase 8.

Phase 18 adds broader production profiling for SCMULT, field maps, wigglers, and ion/Poisson candidates.  The focused thin-RFCA gate is `phase18_rfca_thin`, which covers the zero-length no-wake RFCA path used to keep `scRing2` resident after its first CPU fiducialization setup pass.  Its 30,000-particle, 224-pass timing gate matched CPU output at `1e-11` and ran in 11.94 seconds on `gpu-elegant` versus 60.16 seconds on CPU.  Phase 18 also includes a narrow deterministic `KICKMAP`/`UKICKMAP` CUDA prototype, exercised by the `uKickMap1` production wrapper; after adding a device map-array cache, the 30,000-particle, 2,000-pass gate matched CPU output at `1e-11` and ran in 13.58 seconds on `gpu-elegant` versus 55.43 seconds on CPU.  General RFCA/RFCW fiducialization, nonzero-length or wake-bearing RF cavities, and `KICKMAP`/`UKICKMAP` radiation/ISR/misalignment/yaw paths remain CPU fallback.

Phase 8 is wrapped up for the current single-node, single-GPU CUDA scope with the first conservative Pelegant CUDA path.  Build it with `make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc`; it installs `bin/Linux-x86_64-gpu/gpu-Pelegant`.  Worker ranks select CUDA devices from the `CUDA_VISIBLE_DEVICES` list unless `ELEGANT_GPU_DEVICE` is set, while the no-particle master rank remains CPU-only.  MPI particle exchange still stages through host memory with `forceParticlesToCpu` in `scatterParticles` and `gatherParticles`; GPU-aware MPI is deferred.  GPU reductions are also forced through the CPU MPI reduction path for now, but read-only reduction fallback preserves the current element's GPU eligibility so matrix kernels can still run on worker ranks.  The benchmark runner accepts `--mpi-ranks N` for fixed-rank Pelegant runs.  Current smoke labels are `cpu-phase8-pelegant-matrix` and `gpu-phase8-pelegant-matrix`; both used 2 MPI ranks, 2000 particles, and 5 passes, matched all 4 common SDDS files at `1e-11`, and the CUDA run reported `matrix=320` on worker rank 1.  Deferred Phase 8 follow-ups require multi-GPU or cluster hardware: larger Pelegant timing cases, true MPI-aware GPU reductions, GPU-aware MPI particle exchange, deterministic load-balancing validation with multiple GPU worker ranks, one-GPU-per-worker timing, multi-GPU speedup validation, and multi-node measurements.  The next active CUDA work should move to Phase 9.

Phase 19 starts the Pelegant follow-up within the current one-GPU limit.  `test/gpu_cuda/pelegant_single_gpu.sh` runs paired CPU/GPU Pelegant cases, compares SDDS output, and records manifests.  The 2-rank `matrix` timing gate used 20,000 particles and 20 passes, matched all 4 common SDDS files at `1e-11`, and ran in 6.41 seconds on `gpu-Pelegant` versus 11.06 seconds on CPU Pelegant, about 1.73x faster.  A 3-rank `phase19_matrix_load_balance` diagnostic found that unguarded CUDA redistribution for `load_balancing_on=1` was unsafe on the single-GPU host, so CUDA Pelegant now forces that mode to CPU fallback in `ELEGANT_GPU_MODE=auto`; `ELEGANT_GPU_MODE=required` fails fast.  The fallback check matched CPU output at `1e-11` and is documented in `test/gpu_cuda/output/reports/phase19-pelegant-load-balance-fallback.md`.  True multi-GPU validation remains deferred until hardware is available.

Build the CUDA verification variant:

```sh
make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=/usr/local/cuda-12.4/bin/nvcc
```

This writes to `src/O.Linux-x86_64.gpu.verify` and `bin/Linux-x86_64-gpu-verify/gpu-elegant`.

Run CPU baselines aimed at about one minute per case:

```sh
./test/gpu_cuda/run_benchmarks.sh --baseline --target-seconds 60
```

The baseline mode first runs a small sample, scales the pass count toward the target, then runs the final timing.  `timeout` is used with a default guard of 180 seconds per run; override it with `TIMEOUT_SECONDS=<seconds>` when needed.

For `lcls0`, baseline mode scales `sample_fraction` instead of pass count, capped at the full 199999-particle input beam.

Use a different elegant binary:

```sh
./test/gpu_cuda/run_benchmarks.sh --elegant /path/to/elegant --quick
```

Run a fixed-rank Pelegant smoke:

```sh
./test/gpu_cuda/run_benchmarks.sh --quick --case matrix --mpi-ranks 2 \
  --elegant ./bin/Linux-x86_64/Pelegant --label cpu-phase8-pelegant-matrix
ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
  ./test/gpu_cuda/run_benchmarks.sh --quick --case matrix --mpi-ranks 2 \
  --elegant ./bin/Linux-x86_64-gpu/gpu-Pelegant --label gpu-phase8-pelegant-matrix
```

Run the Phase 19 paired Pelegant harness:

```sh
./test/gpu_cuda/pelegant_single_gpu.sh --ranks 2 --particles 20000 --passes 20
ELEGANT_GPU_MODE=auto ./test/gpu_cuda/pelegant_single_gpu.sh \
  --case phase19_matrix_load_balance --ranks 3 --particles 12000 --passes 10
```

The runner also locates `defns.rpn` from `RPN_DEFNS` or the adjacent `../SDDS/defns.rpn` checkout.  Use `--rpn-defns /path/to/defns.rpn` if your SDDS checkout lives somewhere else.

Each run writes files below `test/gpu_cuda/output/<run-label>/` and records a `manifest.tsv` with case, MPI rank count, particle count, pass count, extra macros such as `sample_fraction`, status, and timing.

## CI And Release Checks

`ci_smoke.sh` wraps the common Phase 9 build and smoke-test commands.  It defaults to a CPU build when no action is given.

```sh
./test/gpu_cuda/ci_smoke.sh --cpu-build
./test/gpu_cuda/ci_smoke.sh --cuda-build --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_smoke.sh --cuda-verify-build --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_smoke.sh --quick --case matrix
./test/gpu_cuda/ci_smoke.sh --mpi-smoke --case matrix --mpi-ranks 2
```

Use `--baseline --target-seconds 60` for timing gates so CPU baseline runs aim at one minute instead of accidentally expanding into hour-scale tests.  For same-workload CPU/GPU timing, use `--particles` and `--passes` to pin both runs to the same workload.  Use `--output DIR` to place benchmark outputs in a CI job artifact root, and use `--dry-run` to inspect the commands intended for a CI runner.

For Phase 10 release-invariance checks, use `release_invariance.sh` after building the ordinary CPU binary and, when available, `Pelegant`, `gpu-elegant`, and `gpu-Pelegant`.

```sh
./test/gpu_cuda/release_invariance.sh \
  --reference-elegant /path/to/clean/elegant \
  --candidate-elegant ./bin/Linux-x86_64/elegant \
  --reference-pelegant /path/to/clean/Pelegant \
  --candidate-pelegant ./bin/Linux-x86_64/Pelegant \
  --require-cuda-layout
```

The release-invariance helper runs the selected quick CPU tracking comparison, generates `&print_dictionary` SDDS and LaTeX output, checks known `GPUCapable` metadata values, compares dictionary files when a reference binary is provided, and verifies the expected binary layout.  Add `--allow-dictionary-diff` only after reviewing intentional metadata differences such as a new `GPUCapable` element flag.  Add `--clean-check` only when you intentionally want to remove local build object directories and verify that `make CUDA_AUTO=0 clean` and `make clean` clear the CPU, MPI, CUDA, and CUDA-verify variants.

Phase 20 adds `ci_release.sh` as a higher-level driver for local or hosted CI stages.  It can run split CPU build, CUDA compile, GPU smoke, timing, Pelegant smoke, release-invariance, and fallback-report stages, then collect logs, benchmark manifests, generated reports, fallback summaries, and `release_notes_template.md` under one artifact directory.

```sh
./test/gpu_cuda/ci_release.sh --cpu-build --fallback-report
./test/gpu_cuda/ci_release.sh --cuda-compile --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_release.sh --gpu-smoke --pelegant-smoke --case matrix
./test/gpu_cuda/ci_release.sh --all-available --label-prefix local-cuda-ci
```

The repository GitHub Actions workflow is `.github/workflows/gpu-cuda-ci.yml`.  It runs the CPU build/fallback-report job automatically for pull requests and pushes to `main` or `master`, and exposes manual `workflow_dispatch` inputs for the CUDA compile-only job, GPU runtime smoke, optional Pelegant smoke, and optional one-minute CPU-target timing.  The GPU runtime job expects a self-hosted runner labeled `self-hosted`, `linux`, `x64`, and `cuda`.

Use `summarize_fallbacks.py` when reviewing CUDA logs without rerunning benchmarks:

```sh
python3 test/gpu_cuda/summarize_fallbacks.py \
  --output-root test/gpu_cuda/output \
  --label-prefix local-cuda-ci \
  --output test/gpu_cuda/output/reports/local-cuda-ci-fallbacks.md
```

## Comparing CPU And GPU Outputs

After a CUDA build exists, run the same cases once with the CPU binary and once with the GPU-enabled binary, then compare the output directories:

```sh
python3 test/gpu_cuda/compare_sdds.py \
  test/gpu_cuda/output/cpu-quick \
  test/gpu_cuda/output/gpu-quick
```

The comparator uses `sddsdiff` when available.  For particle files with a `particleID` column it asks `sddsdiff` to use `particleID` as a row label, which keeps loss/compaction tests from failing only because rows appear in a different order.  Use looser tolerances for numerically sensitive collective-effect cases until the CUDA reductions and binning behavior are known to be stable.

## Production Cases

Production wrappers live under `production_cases/`.  They are small wrappers around existing test-set directories such as LCLS, CLIC, maxamp, collimate, csbend, and dqcor cases.  Some upstream lattices intentionally emit their original warnings; those are preserved so the production inputs remain recognizable.  Keep large generated reference output outside git, preferably under `test/gpu_cuda/output/`.

Phase 11 adds an opt-in production smoke subset:

```sh
./test/gpu_cuda/production_smoke.sh --label-prefix phase11
```

Use `--require-gpu` when validating on a GPU runner so the CUDA pass fails instead of silently falling back to CPU execution.

The curated subset and profiling candidates are documented in `production_cases/metadata.tsv`.  Some cases are intentionally fallback-heavy; keep them because they protect CPU synchronization and production-output compatibility.

## Benchmark Reports

Generate a Markdown report from existing CPU/GPU manifests with:

```sh
python3 test/gpu_cuda/report_benchmarks.py \
  --cpu-manifest test/gpu_cuda/output/phase11-final-cpu-production-smoke/manifest.tsv \
  --gpu-manifest test/gpu_cuda/output/phase11-final-gpu-production-smoke/manifest.tsv \
  --run-compare \
  --tolerance 1e-11 \
  --gpu-binary ./bin/Linux-x86_64-gpu/gpu-elegant \
  --metadata ELEGANT_GPU_MODE=required \
  --output test/gpu_cuda/output/reports/production-smoke.md
```

The report command reads existing outputs; it does not launch new benchmark workloads.  To rerun the curated production smoke and write a report in one bounded command, use:

```sh
./test/gpu_cuda/production_smoke.sh \
  --require-gpu \
  --label-prefix phase12-production \
  --report test/gpu_cuda/output/reports/phase12-production-smoke.md
```

For autoscaled baseline reports, pass the CPU and GPU baseline manifests without `--run-compare`.  The report shows same-workload speedup only when the CPU and GPU particle/pass counts match, and throughput speedup when baseline scaling chose different workloads.  Each report includes a short release-notes summary near the top.
