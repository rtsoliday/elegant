# CUDA GPU Support

This document describes the optional CUDA build and verification workflow for elegant.  The ordinary CPU executables are always built and should produce unchanged output.  When the top-level makefile finds a usable CUDA Toolkit, it also builds separate CUDA executables; CUDA runtime use remains off by default unless `ELEGANT_GPU_MODE` is set.  CUDA support is conservative and still expected to fall back to CPU code for unsupported elements, numerically sensitive modes, diagnostics, output, and MPI particle exchange.

The old dormant `HAVE_GPU` and `GPU_SUPPORT` hooks were useful signposts, but they predate the current CUDA implementation.  Treat them as legacy integration points that may need substantial modification before being used for new code paths.

`GPUCapable` dictionary metadata is intentionally still visible in ordinary CPU builds.  This preserves the historical dictionary shape, but it also means `GPU_SUPPORT` flag updates can change `&print_dictionary` output even when serial tracking behavior is unchanged.  Review dictionary output separately from tracking output when preparing a release.

## Known-Good Local Setup

The current implementation has been exercised on Linux x86_64 with:

- NVIDIA GeForce RTX 3060, compute capability `sm_86`
- NVIDIA driver `580.126.09`
- CUDA runtime reported by the driver as `13.0`
- CUDA Toolkit `12.4`, using `nvcc` `V12.4.131`

Other CUDA toolkit versions at release 12 or newer and other GPU architectures may work, but should be treated as new validation targets until they pass the build, correctness, and timing checks below.

## Build Requirements

- A working CPU elegant build and adjacent SDDS checkout.
- CUDA Toolkit with `nvcc` release 12 or newer.
- SDDS tools such as `sddsquery` and `sddsdiff` for output comparison.
- `python3` for `test/gpu_cuda/compare_sdds.py`.
- `mpirun` only for Pelegant smoke tests.

The makefiles honor `NVCC` and look for `nvcc` on `PATH`, `/usr/local/cuda-12.4/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, and other common CUDA install locations.  Auto builds ignore older CUDA toolkits; explicit CUDA builds fail early unless `nvcc --version` reports release 12 or newer.  Prefer the known local compiler explicitly when reproducing current results:

```sh
NVCC=/usr/local/cuda-12.4/bin/nvcc
```

Set `CUDA_ARCH` when building for a GPU other than the local RTX 3060 target.  The current local default is `sm_86`.

## Build Commands

Build everything available on the host:

```sh
make -j4
```

With MPI compiler wrappers available, this builds `elegant` and `Pelegant`.  With release-12-or-newer `nvcc` plus `libcudart` available, it also builds `gpu-elegant`; with both MPI and CUDA available, it also builds `gpu-Pelegant`.

Build only CPU binaries:

```sh
make CUDA_AUTO=0 -j4
```

Force the CUDA serial binary with a specific compiler:

```sh
make -C src HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4
```

This installs the CUDA serial executable as:

```text
bin/Linux-x86_64-gpu/gpu-elegant
```

Build the CUDA serial verification binary:

```sh
make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4
```

This installs:

```text
bin/Linux-x86_64-gpu-verify/gpu-elegant
```

Build CPU Pelegant:

```sh
make -C src -f Makefile.mpi -j4
```

Build CUDA Pelegant:

```sh
make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4
```

This installs:

```text
bin/Linux-x86_64-gpu/gpu-Pelegant
```

CUDA Pelegant is currently validated for fixed-rank, single-node runs with `load_balancing_on=0`.  When `load_balancing_on=1`, dynamic particle redistribution is deliberately forced to CPU fallback in `ELEGANT_GPU_MODE=auto`; `ELEGANT_GPU_MODE=required` fails fast instead of attempting an unvalidated CUDA redistribution path.

## Runtime Controls

The runtime defaults are intentionally cautious.  Use these controls for diagnosis, verification, and targeted performance work:

| Variable | Purpose |
| --- | --- |
| `ELEGANT_GPU_MODE=off|auto|required` | Disable CUDA, use CUDA when eligible, or fail if CUDA cannot initialize. |
| `ELEGANT_GPU_DEVICE=N` | Select a CUDA device explicitly.  Pelegant worker ranks otherwise map across `CUDA_VISIBLE_DEVICES`; rank 0 remains CPU-only when it owns no particles. |
| `ELEGANT_GPU_VERBOSE=1` | Print CUDA initialization, fallback, transfer, and kernel-count diagnostics. |
| `ELEGANT_GPU_MIN_PARTICLES=N` | Global particle-count threshold for CUDA work. |
| `ELEGANT_GPU_MIN_MATRIX_PARTICLES=N` | Matrix-family tracking threshold. |
| `ELEGANT_GPU_MIN_HELPER_PARTICLES=N` | Helper element threshold. |
| `ELEGANT_GPU_MIN_REDUCTION_PARTICLES=N` | Beam-sum, centroid, and related reduction threshold. |
| `ELEGANT_GPU_MIN_EXACT_DRIFT_PARTICLES=N` | Exact-drift threshold.  Exact drift is enabled by default for eligible GPU runs at or above this threshold. |
| `ELEGANT_GPU_MIN_APERTURE_PARTICLES=N` | Aperture predicate threshold. |
| `ELEGANT_GPU_MIN_MAGNET_PARTICLES=N` | Deterministic magnet threshold. |
| `ELEGANT_GPU_MIN_WAKE_PARTICLES=N` | `WAKE` and `TRWAKE` threshold. |
| `ELEGANT_GPU_MIN_LSC_PARTICLES=N` | `LSCDRIFT` threshold. |
| `ELEGANT_GPU_MIN_CSR_PARTICLES=N` | `CSRCSBEND` wake-potential threshold. |
| `ELEGANT_GPU_MIN_CSR_BINS=N` | Minimum CSR bin count; default is high enough to avoid small-case slowdown. |
| `ELEGANT_GPU_MIN_SCMULT_PARTICLES=N` | `SCMULT` threshold for the guarded Phase 7 slice. |
| `ELEGANT_GPU_ENABLE_EXACT_DRIFT=0|1` | Override exact-drift CUDA eligibility.  The default is enabled; set `0` to force exact drift back to CPU, or `1` to mark the choice explicitly in verbose diagnostics. |
| `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=0|1` | Override the Phase 15 stable prefix-sum aperture loss-compaction policy.  The default is enabled only for runs that do not write a `losses` file, where survivor tracking can remain resident without immediate lost-tail row copies.  Set `0` to force the older CPU loss fallback, or `1` to force the stable path even for `.los` output runs while benchmarking.  The path preserves the verified simple rectangular/elliptical `MAXAMP`, non-inverted `RCOL` including zero-length and nonzero-length open-sided `RCOL`, nonzero-length open-sided `RCOL` global loss-coordinate output, non-inverted even-exponent `ECOL` including open-sided and mixed-`YEXPONENT` cases with global loss-coordinate output, ideal `SCRAPER` including one-sided and two-sided global loss-coordinate output, interpolated rectangular `aperture_data`, and narrow identity-`RFCA` `removeInvalidParticles` loss bookkeeping, including accepted counts and loss values by `particleID` where supported.  It promotes the compacted scratch coordinate buffer to resident device storage and keeps `accepted` compaction on the device by default. |
| `ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE=0|1` | Controls device-side `accepted` array compaction when stable aperture or magnet loss compaction is active.  The default is enabled with those stable compaction paths; set it to `0` to force the older prefix readback and host-side accepted partition while debugging. |
| `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0|1` | Override the stable prefix-sum magnet loss-compaction policy for supported multipole, non-CSR `CSBEND`, and action-8 `KICKMAP`/`UKICKMAP` kernels.  The default is enabled only for runs that do not write a `losses` file, where survivor tracking can remain resident and no host loss rows are required.  Set `0` to force the older CPU loss fallback, or `1` to force the same guarded compaction attempt while benchmarking.  Accepted-array partitioning is supported through the same stable prefix map; `.los` or global-loss-coordinate output still forces CPU loss-row handling. |
| `ELEGANT_GPU_ENABLE_CSR_RESIDENT=0|1` | Override the Phase 14 resident `CSRCSBEND` policy.  The default is enabled for the guarded supported subset; set `0` to force the older CPU/non-resident CSR handoff path, or `1` to mark the choice explicitly in verbose diagnostics.  For supported non-IGF, non-radiating, non-backtracking CSRCSBEND sections, it keeps the body slice, coordinate-only device-side ct range reduction, histogram fill, wake calculation, and CSR kick on resident particle coordinates, preserves per-particle resident scratch across CSR wake-array growth, reuses CSRCSBEND body rollback/loss buffers, allows adjacent simple matrix/drift islands to remain GPU-resident when the upcoming CSRCSBEND is resident-eligible, performs checked CUDA simple initial/final transforms including first-order entrance/exit edge focusing when particles are already device-current, keeps no-CSR `CSRDRIFT` sections resident as exact or linearized drifts, avoids full CSRDRIFT final-prep for Stupakov-only drift consumers, keeps those final coordinates GPU-resident when safe, and hands off to the CPU for remaining CSRDRIFT state and unsupported options. |
| `ELEGANT_GPU_ENABLE_SCMULT=0|1` | Override the guarded linear, unsliced, single-bucket `SCMULT` CUDA slice.  The serial CUDA default is enabled for the existing guard set; set `0` to force the older CPU fallback, or `1` to enable the same guarded path explicitly, including MPI builds.  The May 9, 2026 action-8 `scRing2_no_watch` refresh matched CPU output at `1e-11` with 8 resident SCMULT kernels and only final deallocation synchronization; focused `phase65_scmult_nonlinear_fallback`, `phase66_scmult_sliced_fallback`, and `phase67_scmult_multibunch_fallback` gates confirm nonlinear, sliced, and multi-bunch SCMULT remain CPU-owned under the same policy. |
| `ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS=0|1` | Keep short simple-matrix runs on the CPU when the host copy is already current and a CPU-only element follows soon after.  Enabled by default to avoid host/device ping-pong in mixed lattices. |
| `ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS=N` | Maximum length for short-GPU-island CPU preservation; default is 4 simple matrix elements. |
| `ELEGANT_GPU_VERIFY=1` | Enable runtime comparison checks when the binary was built with `GPU_VERIFY=1`. |
| `ELEGANT_GPU_COMPARE_ABS`, `ELEGANT_GPU_COMPARE_REL` | General GPU verification tolerances. |
| `ELEGANT_GPU_REDUCTION_COMPARE_ABS`, `ELEGANT_GPU_REDUCTION_COMPARE_REL` | Reduction-specific verification tolerances. |
| `ELEGANT_GPU_WAKE_COMPARE_ABS`, `ELEGANT_GPU_WAKE_COMPARE_REL` | Wake-array verification tolerances. |

For small smoke tests, use `ELEGANT_GPU_MIN_PARTICLES=1` so the CUDA path is exercised.  For ordinary runs, keep the default thresholds unless timing data shows the CUDA path is beneficial.

## Supported CUDA Slices And CPU Fallbacks

Current CUDA support is deliberately incremental:

- Matrix-family tracking for ordinary first- through third-order matrix paths, including several helper and coordinate-neutral element cases.
- Common centroid, time-sum, and beam-sum reductions when specialized CPU-only modes are not active.
- Loose rectangular and elliptical aperture predicates for `MAXAMP`, `RCOL`, `ECOL`, and simple one-sided `SCRAPER`; exact lossy bookkeeping falls back to CPU when stable compaction is not selected.
- Default no-loss-output stable lossy aperture compaction for selected `MAXAMP`, `RCOL` including zero-length and nonzero-length open-sided `RCOL` with nonzero-length open global-loss-coordinate output, open-sided and mixed-`YEXPONENT` simple `ECOL` with global-loss-coordinate output, ideal `SCRAPER` including one-sided and two-sided global-loss-coordinate output, `aperture_data`, and narrow identity-`RFCA` `removeInvalidParticles` cases.  `.los` output runs still default to CPU loss-row handling unless `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` is set explicitly.
- Deterministic `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, simple `MULT`, and non-CSR `CSBEND` slices, including simple original-mode `DX`/`DY`/`DZ`/`TILT` misalignments for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, and `MULT`, plus simple original-mode `DX`/`DY`/`DZ`/`TILT`/`ETILT` for non-CSR `CSBEND`.  The `MULT` subset is limited to deterministic `ORDER=0..3`, `N_SLICES>0`, no synchrotron radiation, no spin tracking, and no aperture hook.  Detected magnet losses use the default no-loss-output stable compaction path for the supported subset, including accepted-array partitioning; `.los` and global-loss-coordinate output remain on the CPU loss-row path unless a future resident loss-row design is validated.
- Conservative `WAKE` and `TRWAKE` slices with fixed or autoscaled binning, including Phase 16 smoothed single-bunch modes that use CUDA for particle-heavy work and the existing CPU Savitzky-Golay routine for the small histogram smoothing step, `WAKE,CHANGE_P0=1` central-momentum matching, tilted single-bunch `TRWAKE` when spin coordinates are inactive, and serial/local `BUNCHED_BEAM_MODE=1` for single-effective-bucket `WAKE` and `TRWAKE`, no-op filtered cases that skip either the only effective bucket or all effective buckets in a detected multi-bucket beam, and Action 6 detected multi-bucket filters when `START_BUNCH`/`END_BUNCH` select one or more effective buckets.  Multi-bucket tracking is implemented as independent bucket-local CUDA wake passes to match CPU histogram and convolution semantics; `WAKE,CHANGE_P0=1` is supported for tracked buckets and for skip-only filters through a match-only CUDA action.
- Conservative `LSCDRIFT` slices with fixed even binning, including Phase 16 Savitzky-Golay histogram smoothing, low- and high-frequency cutoff filtering, explicit kick mode with `L=0,LEFFECTIVE>0`, `AUTO_LEFFECTIVE` after elegant resolves the effective length, and backtracking.
- Serial/local `gpu_findFiducialTime` support for `LIGHT`, full-beam and selected-bunch `TMEAN`, and full-beam and selected-bunch `PMAXIMUM`, including nonzero `sOffset` for RF cavity fiducialization, currently used to keep `modulate_elements`, zero-length thin `RFCA`, nonzero-length RF-only matrix-method and kick-method `RFCA`, and the guarded `RFCW` paths resident when possible.
- Conservative `CSRCSBEND` wake-potential array calculation after CPU-compatible histogram and derivative preparation.
- Guarded serial CUDA linear, unsliced, single-bucket `SCMULT`, including the production-shaped `scRing2_no_watch` action-8 quick refresh.  Nonlinear, sliced single-bunch, and multi-bunch modes are guarded by focused CPU-fallback gates rather than resident CUDA kernels.
- Zero-length thin `RFCA` kicks with GPU-side `T_REFERENCE`, `LIGHT`, serial/local full-beam or selected-bunch `TMEAN`, serial/local full-beam or selected-bunch `PMAXIMUM`, or serial/local `FIRST` phase setup for the restricted no-wake/no-`CHANGE_T`/no-focus shape used by the Phase 18 `scRing2` production wrapper; `CHANGE_P0=1`, deterministic `DX/DY` offsets, and `STANDING_WAVE=1` are supported through the existing GPU central-momentum matching helper and the CPU-equivalent thin-offset and single-kick standing-wave semantics.
- Nonzero-length RF-only matrix-method `RFCA` cavities with no wakes/LSC, no cavity `Q`, no `CHANGE_T`, no body-focus/linearized/locked/backtracking mode, optional `STANDING_WAVE=1`, and serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, or explicit `T_REFERENCE` fiducialization.  Entrance/exit focusing, `CHANGE_P0`, and deterministic `DX/DY` offsets are supported.  The focused `phase29_rfca_matrix_rf_only`, `phase30_rfca_matrix_fiducial_modes`, `phase40_rf_pmaximum_fiducial`, `phase47_rf_selected_tmean_fiducial`, `phase48_rf_selected_pmaximum_fiducial`, `phase50_rf_first_fiducial`, and `phase51_rf_standing_wave_single` regressions are the validation targets for this slice.
- Nonzero-length RF-only kick-method `RFCA,N_KICKS>=1` cavities with no wakes/LSC, no cavity `Q`, no `CHANGE_T`, no body-focus/linearized/locked/backtracking mode, and serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, or explicit `T_REFERENCE` fiducialization.  CUDA reuses the RF section loop with per-section length, voltage, and phase advance, using the CPU-equivalent section-center fiducial offset; entrance/exit focusing, `CHANGE_P0`, deterministic `DX/DY` offsets, and `STANDING_WAVE=1` for `N_KICKS=1`, for `N_KICKS>1` with explicit `T_REFERENCE`, or for `N_KICKS>1` with the same supported serial/local fiducial modes are supported.  The focused `phase39_rfca_kick_rf_only`, `phase40_rf_pmaximum_fiducial`, `phase45_rf_kick_treference`, `phase47_rf_selected_tmean_fiducial`, `phase48_rf_selected_pmaximum_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, `phase52_rf_standing_wave_multikick_treference`, and `phase53_rfca_standing_wave_multikick_fiducial` regressions are the validation targets for this slice.
- RF-only matrix-method `RFCW` cavities with no active wake columns, no LSC, no cavity `Q`, no linearized/backtracking mode, optional `STANDING_WAVE=1`, and serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, or explicit `T_REFERENCE` fiducialization; end focusing, `CHANGE_P0`, and deterministic `DX/DY` offsets are supported.  The CLIC `clic1` production wrapper plus focused offset, fiducial-mode, selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`, selected-bunch `FIRST`, and standing-wave regressions are the validation targets for this slice.
- RF-only kick-method `RFCW,N_KICKS>=1` cavities with no active wake columns, no LSC, no cavity `Q`, no `CHANGE_T`, no body-focus/linearized/backtracking mode, and serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, or explicit `T_REFERENCE` fiducialization.  CUDA uses the same per-section RF kick loop as the guarded collective path, without invoking wake or LSC kernels; deterministic `DX/DY` offsets, end focusing, `CHANGE_P0`, no-op `WAKES_AT_END`, and `STANDING_WAVE=1` for `N_KICKS=1`, for `N_KICKS>1` with explicit `T_REFERENCE`, or for `N_KICKS>1` with the same supported serial/local fiducial modes are supported inside the guarded slice.  The focused `phase38_rfcw_kick_rf_only`, `phase40_rf_pmaximum_fiducial`, `phase45_rf_kick_treference`, `phase47_rf_selected_tmean_fiducial`, `phase48_rf_selected_pmaximum_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, `phase52_rf_standing_wave_multikick_treference`, and `phase54_rfcw_standing_wave_multikick_fiducial` regressions are the validation targets for this slice.
- Narrow wake-bearing matrix-method `RFCW` cavities with serial/local longitudinal and/or transverse wake columns, single-wake-family wake-column subsets, or LSC-only kicks, optional fixed-even-bin `LSC=1` LSCKICK with interpolation and low/high-frequency cutoffs, no cavity `Q`, no `CHANGE_T`, no kick-method RF, no body-focus/linearized/backtracking mode, optional `STANDING_WAVE=1`, and serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, or explicit `T_REFERENCE` fiducialization.  This path composes the resident RF matrix kernel with the existing `WAKE`/`TRWAKE` and LSC bin/kick kernels and keeps deterministic `DX/DY` offsets local through the collective kicks; autoscaled/fixed wake bins, wake smoothing, interpolation, end focusing, `CHANGE_P0`, and positive-length matrix-method `WAKES_AT_END=1` CPU ordering are supported within the existing guards.  The focused `phase32_rfcw_matrix_wake`, `phase35_rfcw_matrix_wakes_at_end`, `phase36_rfcw_lsc`, `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, and bounded `lcls1` production wrapper are the validation targets for this slice.
- Narrow wake-bearing kick-method `RFCW,N_KICKS>=1` cavities with the same serial/local wake and LSCKICK guards, no cavity `Q`, no `CHANGE_T`, and no body-focus/linearized/backtracking mode.  CUDA loops over the CPU kick sections with per-section length and voltage, the traveling-wave per-section RF phase advance, entrance focusing only on the first section, exit focusing only on the final section, existing `WAKE`/`TRWAKE` and optional LSC kernels per section, and CPU-equivalent collective ordering for `WAKES_AT_END=0|1`; serial/local `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, serial/local `FIRST`, explicit `T_REFERENCE`, deterministic `DX/DY`, wake smoothing/interpolation, autoscaled/fixed wake bins, single-wake-family wake-column subsets, LSC-only kicks, fixed-even LSC bins with filters, `CHANGE_P0`, and `STANDING_WAVE=1` for `N_KICKS=1`, for `N_KICKS>1` with explicit `T_REFERENCE`, or for `N_KICKS>1` with the same supported serial/local fiducial modes are supported inside the guarded slice.  The focused `phase33_rfcw_kick_wake`, `phase34_rfcw_wakes_at_end`, `phase36_rfcw_lsc`, `phase37_rfcw_multikick`, `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, `phase52_rf_standing_wave_multikick_treference`, `phase54_rfcw_standing_wave_multikick_fiducial`, and bounded `lcls0` production wrapper are the validation targets for this slice.
- Narrow deterministic `KICKMAP`/`UKICKMAP` interpolation/kick tracking for serial/local no-radiation/no-ISR maps with no map-element offsets, tilt, or yaw, with device-resident map-array caching.  Checked CPU fallback remains available with `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0` if a particle leaves the map, but no-loss-output runs now use resident stable map-loss compaction by default for the supported subset, validated by `uKickMap1`, the high-count production-shaped `latticeErrors6` `UKICKMAP` wrapper, and the focused ordinary `phase62_kickmap_loss_compaction` `GKICKMAP` fixture.  The focused `phase63_kickmap_loss_output_fallback` and `phase64_kickmap_global_loss_fallback` gates verify that ordinary `KICKMAP` `.los`/`.acc` and global loss-coordinate map rows still take the explicit CPU fallback under the same compaction policy; `latticeErrors6_loss_output` and `latticeErrors6_global_loss` provide the corresponding high-count `UKICKMAP` fallback guards.
- Single-node Pelegant CUDA smoke coverage for fixed-rank `load_balancing_on=0` runs with host-staged MPI scatter/gather.

Known fallbacks and deferred areas include stochastic radiation/ISR, spin tracking, many advanced `CSBEND` models, remaining magnet-misalignment modes such as pitch/yaw, nonzero `MALIGN_METHOD`, remaining `CSBEND` misalignment modes, and radiation/spin combinations, high-order or file-backed `MULT`/`FMULT`, aperture hooks, full GPU-resident CSR tracking, remaining CSR drift state, CSR Derbenev/output diagnostics, distributed Pelegant bunched wakes, tilted `TRWAKE` with active spin coordinates, unsupported `LSCDRIFT` effective-length or unordered filter setups, RFCA/RFCW fiducialization outside the documented explicit-reference, `LIGHT`, full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`, and serial/local `FIRST` subsets, wake-bearing RFCA, RFCA with `CHANGE_T`, cavity `Q`, body-focus, linearized/locked phase, backtracking, or RFCA modes outside the zero-length thin, nonzero-length RF-only matrix-method, and nonzero-length RF-only kick-method subsets, LSC-in-RFCW outside the guarded matrix-method and kick-method collective slices, RFCW modes outside the documented matrix-method, single-kick, explicit-reference multi-kick, and supported non-explicit fiducial multi-kick `STANDING_WAVE=1` slices, linearized, cavity-`Q`, backtracking, distributed, or unsupported `RFCW` fiducial modes, Pelegant distributed-particle GPU fiducial-time reductions, Pelegant `load_balancing_on=1` dynamic CUDA redistribution, cuFFT-backed collective effects, nonlinear/sliced/multi-bunch `SCMULT`, Poisson solve work, `KICKMAP`/`UKICKMAP` radiation/ISR/misalignment/yaw and resident loss-output map-loss rows, field-map/wiggler CUDA kernels beyond the bounded Phase 18 CPU-fallback comparison wrappers and the narrow `KICKMAP`/`UKICKMAP` prototype, ion effects, GPU-aware MPI particle exchange, true MPI GPU reductions, and multi-GPU speedup validation.  The `phase61_csbend_advanced_fallback` gate covers representative advanced `CSBEND` fallbacks for nonzero `MALIGN_METHOD` with `EPITCH`/`EYAW` plus Hwang/Lindberg/curved fringe settings; `phase65_scmult_nonlinear_fallback`, `phase66_scmult_sliced_fallback`, and `phase67_scmult_multibunch_fallback` cover representative nonlinear, sliced single-bunch, and multi-bunch `SCMULT` fallbacks under the guarded SCMULT policy.

Unsupported cases should synchronize to CPU and preserve existing behavior.  A new CUDA path should not be enabled automatically until it has correctness evidence, timing evidence, and a threshold that avoids slowing down small cases.

Phase 17 extends deterministic magnet coverage to simple original-mode multipole, `DQCOR`, and non-CSR `CSBEND` misalignments.  Action 7 adds the first profile-driven follow-up: simple deterministic `MULT` with `ORDER=0..3`, plus default no-loss-output stable compaction for detected losses from supported magnet kernels.  The `phase17_multipole_misalignment` gate matched CPU output at `1e-11` and ran the 30,000-particle, 23-pass workload in 7.23 seconds on `gpu-elegant` versus 59.97 seconds on CPU.  The `phase17_dqcor_misalignment` gate matched CPU output at `1e-11` and ran the 30,000-particle, 12-pass workload in 4.46 seconds on `gpu-elegant` versus 58.59 seconds on CPU.  The `phase17_csbend_misalignment` gate matched CPU output at `1e-11`, passed the `GPU_VERIFY` quick check, and ran the 30,000-particle, 11-pass workload in 4.53 seconds on `gpu-elegant` versus 57.36 seconds on CPU.  The `phase55_mult_deterministic` quick gate matched CPU output at `1e-11`, passed `GPU_VERIFY`, reported 100 CUDA magnet kernels with no CPU-element fallback, and refreshed the production magnet profile to 131,004 simple CUDA candidates with 17,545 deferred/blocking definitions.  The `phase56_mult_loss_compaction` quick gate matched CPU output at `1e-11` with default magnet compaction, reporting no CPU-element fallback and only final deallocation synchronization.  The `phase57_mult_loss_accepted_compaction` quick gate also matched CPU output at `1e-11`, including `.acc`, with accepted-device compaction enabled and only final deallocation synchronization.  The `phase59_mult_loss_output_fallback` quick gate lost 2862 of 3000 particles and matched CPU output at `1e-11`, including `.los` and `.acc`, while proving the compaction policy still preserves loss-output semantics through the CPU loss-row fallback.  The `phase60_mult_global_loss_fallback` quick gate matched CPU output at `1e-11`, including the global loss-coordinate `.los` columns `X`, `Z`, and `thetaX`, through the same fallback guard.  The `phase58_csbend_loss_compaction` quick gate lost 47 of 3000 particles and matched CPU output at `1e-11`, including `.acc`, with no CPU-element fallback and only final deallocation synchronization.  The `phase61_csbend_advanced_fallback` quick gate matched CPU output at `1e-11` while confirming advanced `CSBEND` misalignment and fringe models remain explicit CPU fallbacks.  Reports are under `test/gpu_cuda/output/reports/`.

Phase 18 adds production profiling for `SCMULT`, field-map, wiggler, and ion/Poisson candidates.  The `scRing2` production wrapper exercises the guarded linear `SCMULT` path plus the narrow thin-`RFCA` CUDA kick, and the focused `phase18_rfca_thin` gate matched CPU output at `1e-11` with a 30,000-particle, 224-pass workload running in 11.94 seconds on `gpu-elegant` versus 60.16 seconds on CPU.  Action 8 refreshed the production-shaped `scRing2_no_watch` linear `SCMULT` wrapper; the quick 1000-particle, 8-pass gate matched all 5 common SDDS files at `1e-11`, reported 8 resident `SCMULT` kernels and no CPU-element fallback, and synchronized only at final `gpuBaseDealloc`.  The focused `phase65_scmult_nonlinear_fallback`, `phase66_scmult_sliced_fallback`, and `phase67_scmult_multibunch_fallback` gates matched CPU at `1e-11` under the same guarded policy: nonlinear/sliced matched 10 common files with 48 `trackThroughSCMULT fallback` synchronizations per case, and multi-bunch matched 7 common files with 2 `initializeSCMULT fallback` plus 48 `accumulateSCMULT fallback` synchronizations.  All three reported 0 resident `SCMULT` kernels.  Action 6 removed the focused thin-`RFCA` first-pass phase-setup synchronization and added `phase26_rfca_thin_change_p0`, `phase27_rfca_thin_fiducial_modes`, `phase28_rfca_thin_offset`, `phase29_rfca_matrix_rf_only`, `phase30_rfca_matrix_fiducial_modes`, `phase39_rfca_kick_rf_only`, `phase40_rf_pmaximum_fiducial`, `phase47_rf_selected_tmean_fiducial`, `phase48_rf_selected_pmaximum_fiducial`, `phase49_rfcw_wake_selected_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, `phase52_rf_standing_wave_multikick_treference`, `phase53_rfca_standing_wave_multikick_fiducial`, and `phase54_rfcw_standing_wave_multikick_fiducial` coverage for `CHANGE_P0`, `LIGHT`, serial/local full-beam and selected-bunch `TMEAN`, serial/local full-beam and selected-bunch `PMAXIMUM`, serial/local selected-bunch `FIRST`, deterministic `DX/DY` offsets, nonzero-length RF-only matrix-method RFCA with entrance/exit focusing, nonzero-length RF-only kick-method RFCA with per-section phase advance, guarded wake-bearing selected-bunch RFCW fiducialization, narrow matrix-method/single-kick `STANDING_WAVE=1` RFCA/RFCW residency, explicit-reference multi-kick standing-wave RFCA/RFCW residency, and non-explicit fiducial multi-kick standing-wave RFCA/RFCW residency.  The action-8 fallback refresh for `bmapxy1`, `bmxyz1`, `boffaxe1`, and `cwiggler10` matched all 19 common SDDS files at `1e-11`; CUDA reported expected CPU-owned `BMXYZ` and `CWIGGLER` handoffs, 10 read-only `WATCH parameter output` synchronizations from `cwiggler10`, and no resident field-map/wiggler kernels.  The narrow cached `uKickMap1` `UKICKMAP` prototype matched CPU output at `1e-11` and ran the 30,000-particle, 2,000-pass workload in 13.58 seconds on `gpu-elegant` versus 55.43 seconds on CPU.  Action 8 added resident map-loss compaction for the no-loss-output `KICKMAP`/`UKICKMAP` subset, now enabled by default: the 3,000-particle, 2,000-pass `uKickMap1` gate matched CPU output at `1e-11`, reduced the same workload's 101 `UKICKMAP particle loss fallback` syncs to final `gpuBaseDealloc` only, and ran in 1.82 seconds on `gpu-elegant` versus 5.54 seconds on CPU.  The high-count `latticeErrors6` `UKICKMAP` wrapper repeats 40 production septum maps per pass; its 30,000-particle, 2-pass compaction gate matched all 4 common SDDS files at `1e-11`, kept 564 survivors, removed 80 same-workload `UKICKMAP particle loss fallback` syncs, and ran in 0.41 seconds on `gpu-elegant` versus 0.77 seconds on CPU.  The focused ordinary `phase62_kickmap_loss_compaction` `GKICKMAP` fixture matched all 4 common SDDS files at `1e-11` and reduced 15 same-workload `KICKMAP particle loss fallback` syncs to final `gpuBaseDealloc` only under the same compaction policy.  The synthetic production-like `ionEffectsPoisson` wrapper now records expected CPU fallback coverage for `IONEFFECTS` plus a 16x16 Poisson grid; its 2,000-particle, 3-pass CPU/GPU auto-mode quick gate matched all 4 common SDDS files at `1e-11` while CUDA reported two `CPU element: IONEFFECTS` synchronizations plus final deallocation.  Phase 18 and action-8 reports are under `test/gpu_cuda/output/reports/`.

The focused `phase63_kickmap_loss_output_fallback` and `phase64_kickmap_global_loss_fallback` gates request `.los`/`.acc` and global loss-coordinate rows.  Both matched all 12 common SDDS files at `1e-11`, including `.los`, while reporting 15 explicit `KICKMAP particle loss fallback` synchronizations per case.  The production-shaped `latticeErrors6_loss_output` and `latticeErrors6_global_loss` wrappers provide the matching UKICKMAP loss-row guard; both matched all 12 common SDDS files at `1e-11`, including `.los`, while reporting 73 explicit `UKICKMAP particle loss fallback` synchronizations per case.

Phase 19 starts Pelegant validation within the available one-GPU hardware.  The `phase19-pelegant-single-gpu-matrix-after-guard` 2-rank matrix gate matched all 4 common SDDS files at `1e-11` and ran the 20,000-particle, 20-pass workload in 6.41 seconds on `gpu-Pelegant` versus 11.06 seconds on CPU Pelegant, about 1.73x faster.  A 3-rank `load_balancing_on=1` diagnostic exposed that unguarded CUDA redistribution was unsafe on this host, so CUDA Pelegant now forces that mode to CPU fallback unless CUDA is required; the fallback gate matched CPU output at `1e-11`.  True multi-GPU speedup validation remains deferred until matching hardware is available.

## Correctness And Timing Workflow

Quick CPU smoke:

```sh
./test/gpu_cuda/run_benchmarks.sh --quick --case matrix --label cpu-matrix-quick
```

Quick CUDA smoke:

```sh
ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
  ./test/gpu_cuda/run_benchmarks.sh --quick --case matrix \
  --elegant ./bin/Linux-x86_64-gpu/gpu-elegant \
  --label gpu-matrix-quick
```

Compare the outputs:

```sh
python3 test/gpu_cuda/compare_sdds.py \
  test/gpu_cuda/output/cpu-matrix-quick/matrix \
  test/gpu_cuda/output/gpu-matrix-quick/matrix \
  --tolerance 1e-11
```

Use `GPU_VERIFY=1` builds for in-process CPU shadow comparisons of intermediate arrays, reductions, element-local updates, and supported CSR last-wake handoff state:

```sh
ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_VERIFY=1 \
  ELEGANT_GPU_MIN_PARTICLES=1 \
  ./test/gpu_cuda/run_benchmarks.sh --quick --case matrix \
  --elegant ./bin/Linux-x86_64-gpu-verify/gpu-elegant \
  --label gpu-matrix-verify
```

Timing gates should aim CPU baseline cases at about one minute, not open-ended long runs:

```sh
./test/gpu_cuda/run_benchmarks.sh --baseline --target-seconds 60 --case matrix
```

The benchmark runner uses a timeout guard, defaulting to 180 seconds per elegant invocation, to avoid accidentally starting hour-long tests.  Increase `TIMEOUT_SECONDS` only for intentional long validation runs.

For Pelegant smoke tests:

```sh
./test/gpu_cuda/run_benchmarks.sh --quick --case matrix --mpi-ranks 2 \
  --elegant ./bin/Linux-x86_64/Pelegant --label cpu-pelegant-matrix
ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
  ./test/gpu_cuda/run_benchmarks.sh --quick --case matrix --mpi-ranks 2 \
  --elegant ./bin/Linux-x86_64-gpu/gpu-Pelegant --label gpu-pelegant-matrix
```

## CI And Release Controls

The local CI helper wraps the common build and smoke-test commands:

```sh
./test/gpu_cuda/ci_smoke.sh --cpu-build
./test/gpu_cuda/ci_smoke.sh --cuda-build --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_smoke.sh --cuda-verify-build --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_smoke.sh --quick --case matrix
```

The Phase 20 CI/release driver groups those primitives into named stages, writes stage logs, collects benchmark outputs, generates fallback summaries, and copies the release-notes template into one artifact directory:

```sh
./test/gpu_cuda/ci_release.sh --cpu-build --fallback-report
./test/gpu_cuda/ci_release.sh --cuda-compile --nvcc /usr/local/cuda-12.4/bin/nvcc
./test/gpu_cuda/ci_release.sh --gpu-smoke --pelegant-smoke --case matrix
./test/gpu_cuda/ci_release.sh --timing --case matrix --target-seconds 60
./test/gpu_cuda/ci_release.sh --all-available --label-prefix local-cuda-ci
```

Use `--dry-run` to inspect the commands before connecting a hosted CI runner.  If the CI service has separate CPU, CUDA compile-only, and GPU runners, call one or more stages per job and archive the directory reported as `CI/release artifacts`.
Use `--output-root DIR` on `ci_release.sh`, or `--output DIR` on `ci_smoke.sh`, when the CI runner requires benchmark output to live under a job-specific artifact directory.

The GitHub Actions workflow in `.github/workflows/gpu-cuda-ci.yml` wires these stages into CI:

- Pull requests and pushes to `main` or `master` run a CPU build/fallback-report job on `ubuntu-latest`.
- Manual `workflow_dispatch` can also run a CUDA compile-only job in an NVIDIA CUDA 12.4 development container, including the `GPU_VERIFY` build.
- Manual `workflow_dispatch` can run GPU correctness, optional timing, and optional Pelegant smoke stages on a self-hosted runner labeled `self-hosted`, `linux`, `x64`, and `cuda`.
- Each job checks out `rtsoliday/SDDS` next to `elegant`, uses `ci_release.sh`, and uploads the resulting `test/gpu_cuda/output/ci_artifacts/<label>/` directory.

The Phase 10 release-invariance helper adds checks for ordinary CPU behavior and CPU-visible dictionary metadata:

```sh
./test/gpu_cuda/release_invariance.sh \
  --reference-elegant /path/to/clean/elegant \
  --candidate-elegant ./bin/Linux-x86_64/elegant \
  --reference-pelegant /path/to/clean/Pelegant \
  --candidate-pelegant ./bin/Linux-x86_64/Pelegant \
  --require-cuda-layout
```

It runs bounded quick tracking comparisons against clean reference binaries, generates `&print_dictionary` SDDS and LaTeX output, checks selected `GPUCapable` metadata values, compares dictionary files when a reference binary is supplied, and checks the CPU/CUDA binary layout.  Use `--allow-dictionary-diff` only after reviewing intentional dictionary metadata differences; for example, the current CUDA work intentionally changes the CPU-visible `KOCT` `GPUCapable` value from `0` to `1`.  Use `--clean-check` only when intentionally removing local build products; it runs both `make CUDA_AUTO=0 clean` and `make clean`, then verifies that CPU, MPI, CUDA, and CUDA-verify object directories are gone.

Recommended CI split:

- CPU runner: run the normal CPU build and at least the quick CPU benchmark smoke.
- CUDA compile-only runner without a GPU: run the CUDA build and CUDA verify build.
- GPU runner: run CPU/GPU quick correctness comparisons, representative `GPU_VERIFY` cases, one-minute CPU-targeted timing baselines, and Pelegant smoke when MPI is available.
- Artifact collection: archive `test/gpu_cuda/output/ci_artifacts/<label>/`, including logs, generated reports, benchmark manifests, fallback TSV/Markdown, and `release_notes_template.md`.

Release checklist:

- CPU-only build succeeds and CPU output remains unchanged.
- `release_invariance.sh` passes against clean serial and MPI reference binaries, or any differences are explicitly reviewed.
- `&print_dictionary` SDDS and LaTeX output changes are intentional, including `GPUCapable` metadata changes from `GPU_SUPPORT` flags.
- CUDA serial and CUDA Pelegant builds succeed with the selected `NVCC` and `CUDA_ARCH`.
- Deterministic CPU/GPU benchmark outputs match within documented tolerances.
- `GPU_VERIFY` passes for each enabled CUDA feature and its intermediate arrays.
- Stochastic features, if enabled later, pass distribution checks rather than bitwise comparisons.  Use `test/gpu_cuda/compare_stochastic_sdds.py` across multiple fixed-seed CPU/GPU output pairs to compare row counts, means, sigmas, spin columns, emittance-like columns, energy spread, bunch length, finite numeric sample counts, and selected particle-column histograms.  The current passing action-10 guards are the two-seed `csbend1`, `spinTest2`, and `cwiggler10_radiation` distribution reports plus the five-seed `uKickMap4_radiation` distribution report.  The `uKickMap4_radiation` guard originally exposed non-finite `Ss`/`St` values from CUDA beam-sum covariance cancellation; negative diagonal covariance terms are now clamped before storage and sigma output uses safe square-root behavior.
- Timing baselines are aimed at one minute on CPU and have timeout guards.
- No CPU-only performance regression is observed.
- Each automatically enabled CUDA path has a threshold that avoids small-case slowdowns.
- GPU speedups are documented by benchmark label, hardware, driver, CUDA toolkit, `CUDA_ARCH`, particle count, pass count, and tolerance.
- Known CPU fallbacks and unsupported elements are documented for the release.

Generate a focused fallback report from any benchmark label prefix:

```sh
python3 test/gpu_cuda/summarize_fallbacks.py \
  --output-root test/gpu_cuda/output \
  --label-prefix local-cuda-ci \
  --output test/gpu_cuda/output/reports/local-cuda-ci-fallbacks.md \
  --tsv test/gpu_cuda/output/reports/local-cuda-ci-fallbacks.tsv
```
