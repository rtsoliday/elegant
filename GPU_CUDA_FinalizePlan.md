# CUDA GPU Finalize Plan

Generated: May 7, 2026

This plan consolidates the items listed as unfinished or deferred in
`GPU_CUDA_ImplementationPlan.md` and `GPU_CUDA_ImprovementPlan.md`, then
compares those notes against the current code. The initial comparison below was
based on static inspection of the repository; action-item sections include fresh
runtime validation where noted.

## Static Comparison Summary

| Area | Plan-file unfinished item | Current code status | Finalize action |
| --- | --- | --- | --- |
| Release and CI | Phase 10 still needs no-MPI/no-CUDA, MPI-only, and CUDA-only portability coverage; Phase 20 still needs hosted workflow execution and CUDA runner review. | `test/gpu_cuda/release_invariance.sh`, `test/gpu_cuda/ci_release.sh`, fallback reports, release notes template, and `.github/workflows/gpu-cuda-ci.yml` exist. The workflow is wired, but the plans do not record a successful fresh GitHub Actions run or a self-hosted CUDA runner review. | Run the workflow from a fresh checkout, capture artifacts, and add explicit portability coverage or documented release exceptions for no-MPI/no-CUDA, MPI-only, and CUDA-only hosts. |
| Benchmark reports | Phase 12 deferred CI artifact upload and a standardized release-note report filename. | Report generation and artifact collection now exist. The GitHub workflow uploads CI artifacts. A final standardized release-report filename still is not clearly enforced. | Standardize the release report names and require uploaded report/fallback artifacts in release review. |
| Basic kernels and residency | Phase 2 kept `exactDrift` opt-in, specialized beam-sum modes on CPU, and host synchronization for output and CPU-only consumers. | Code confirms `ELEGANT_GPU_ENABLE_EXACT_DRIFT` is still opt-in. Common reductions are implemented, while specialized reduction modes and MPI reductions still fall back. Output/diagnostics still synchronize when needed. | Keep exact drift opt-in unless new bounded timings show benefit. Track specialized reductions as low-risk cleanup only after higher-payoff fallbacks are addressed. |
| Apertures and losses | Phase 3 listed `gpu_removeInvalidParticles`, `gpu_imposeApertureData`, and faster compaction as deferred. Phase 15 later kept stable compaction opt-in and listed unsupported edge cases. | The old Phase 3 statement is partly refuted: current code has opt-in stable prefix-sum compaction for selected `MAXAMP`, `RCOL`, `ECOL`, `SCRAPER`, `aperture_data`, and narrow identity-`RFCA` `removeInvalidParticles`. The path remains opt-in, and unsupported inverted, all-loss, material-interaction, and broader invalid-particle cases still fall back. | Reduce lost-tail/global-loss overhead, decide whether any aperture compaction subset becomes recommended/default, and broaden only the edge cases with measured production value. |
| CSR | Phase 6 listed resident CSRCSBEND, range/binning residency, `CSR_LAST_WAKE`, `CSRDRIFT`, IGF, filters, aperture interaction, and production coverage as deferred. Phase 14 later wrapped an opt-in resident CSR subset. | Resident `CSRCSBEND` exists behind `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1`, but guards exclude IGF, wake-filter resident support, radiation/ISR, backtracking, bin-once, in-element output, and several handoff states. `gpu_track_through_driftCSR` now has a narrow no-CSR/no-LSC exact-or-linear drift subset, while state-consuming `CSRDRIFT` modes remain CPU-owned. Fresh action-5 validation confirms the guarded resident subset is production-correct as an opt-in and removes the validated resident final CPU handoff cases; production logs still show CPU-owned `CSRDRIFT` and unsupported `CSRCSBEND` transitions. | Keep resident CSR as a targeted opt-in, not a default. Reduce remaining CPU-owned `CSRDRIFT` and unsupported `CSRCSBEND` transitions next, and separately tackle IGF, wake-filter resident support, Saldin/LSC drift state, and broader CSR plus aperture/loss cases. |
| Wakes, LSC, and RF | Phase 5 deferred smoothed wakes, tilted `TRWAKE`, LSC filters/backtracking/kick/AUTO, `RFCW`, cuFFT paths, and fiducial-time semantics. Phase 16 resolved several of these. | The old Phase 5 deferrals are partly refuted: smoothed `WAKE`/`TRWAKE`, `WAKE,CHANGE_P0=1`, tilted single-bunch `TRWAKE` without spin, several `LSCDRIFT` modes, narrow serial/local fiducial time including selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`, and selected-bunch `FIRST`, selected-bucket/range and skipped-`CHANGE_P0` match-only bunched `WAKE`/`TRWAKE` filtering, zero-length thin `RFCA` with GPU phase setup, `LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST`, `CHANGE_P0`, and deterministic offsets, nonzero-length RF-only matrix-method `RFCA`, nonzero-length RF-only kick-method `RFCA,N_KICKS>=1` including explicit `T_REFERENCE` validated by `phase39_rfca_kick_rf_only`, `phase45_rf_kick_treference`, selected-bunch `TMEAN` validated by `phase47_rf_selected_tmean_fiducial`, selected-bunch `PMAXIMUM` validated by `phase48_rf_selected_pmaximum_fiducial`, and selected-bunch `FIRST` validated by `phase50_rf_first_fiducial`, the RF-only matrix-method `RFCW` subset including deterministic offsets and `LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST` fiducial modes, RF-only kick-method `RFCW,N_KICKS>=1` including explicit `T_REFERENCE` validated by `phase38_rfcw_kick_rf_only`, `phase45_rf_kick_treference`, selected-bunch `TMEAN` validated by `phase47_rf_selected_tmean_fiducial`, selected-bunch `PMAXIMUM` validated by `phase48_rf_selected_pmaximum_fiducial`, and selected-bunch `FIRST` validated by `phase50_rf_first_fiducial`, narrow serial/local wake-bearing matrix-method `RFCW` including autoscaled/fixed wake bins, single-wake-family wake columns, explicit nonzero `T_REFERENCE`, positive-length matrix `WAKES_AT_END=1`, selected-bunch fiducialization including `PMAXIMUM` and `FIRST`, and guarded LSCKICK support including LSC-only cavities validated by `phase36_rfcw_lsc`, `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`, and `phase50_rf_first_fiducial`, and guarded serial/local wake-bearing kick-method `RFCW,N_KICKS>=1` including `WAKES_AT_END=0\|1`, autoscaled/fixed wake bins, single-wake-family wake columns, explicit nonzero `T_REFERENCE`, selected-bunch fiducialization including `PMAXIMUM` and `FIRST`, and guarded LSCKICK support including LSC-only cavities validated by `phase36_rfcw_lsc`, `phase37_rfcw_multikick`, `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`, and `phase50_rf_first_fiducial` are implemented. Action 6 closes the current serial/local finalization scope; distributed Pelegant wake reductions, distributed RFCA/RFCW fiducialization, broader unsupported RFCW/LSC combinations, and cuFFT-backed paths are deferred. | Treat Action 6 as closed for this pass. Defer distributed bunched-wake reductions until MPI design exists, broader RFCA/RFCW modes until focused production evidence appears, and cuFFT until profiling shows FFT cost dominates transfers. |
| Magnets | Phase 4 deferred misalignments, radiation/ISR, spin, advanced `CSBEND`, slice-by-slice, aperture hooks, multipole data files, and corrector radiation kicks. Phase 17 added some deterministic misalignments. | The old "misalignment" deferral is partly refuted: original-mode `DX`/`DY`/`DZ`/`TILT` support exists for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, simple non-CSR `CSBEND` with `ETILT`, and action-7 simple `MULT` order 0 through 3. Code still rejects pitch/yaw, nonzero `MALIGN_METHOD`, radiation/ISR, spin, aperture hooks, slice tracking, reference/FSE correction, advanced fringe models, high-order/file-backed multipoles, and extra multipole files. `gpu_addCorrectorRadiationKick` remains unsupported. | Use the refreshed production magnet profile to choose the next deterministic slice. Keep stochastic radiation/ISR and spin blocked until distribution-level validation exists. |
| SCMULT, Poisson, ion effects | Phase 7 deferred sliced/nonlinear/multi-bunch SCMULT, Poisson/cuFFT work, field-map/wiggler profiling, and ion effects. Phase 18 added production profiling and `scRing2`; action 8 refreshed `scRing2_no_watch`. | Linear unsliced single-bucket `SCMULT` exists behind `ELEGANT_GPU_ENABLE_SCMULT=1`; `SCMULT` is not a general automatic `GPU_SUPPORT` element. The no-WATCH action-8 quick gate matched CPU at `1e-11` with only final deallocation sync. `phase65_scmult_nonlinear_fallback`, `phase66_scmult_sliced_fallback`, and `phase67_scmult_multibunch_fallback` now validate that nonlinear, sliced, and multi-bunch SCMULT stay on the CPU path under the same opt-in flag. No CUDA Poisson or ion-effects path is present, but the new `ionEffectsPoisson` wrapper now captures the expected CPU fallback for `IONEFFECTS` plus a 16x16 Poisson grid. | Keep linear SCMULT opt-in until a release decision or second source-family validation. Keep nonlinear, sliced, bunched, Poisson, and ion work CPU-owned unless profiling justifies a CUDA port; use `ionEffectsPoisson` as the future correctness gate. |
| Field maps and wigglers | Phase 18 deferred field-map/wiggler acceleration and high-count `UKICKMAP` decisions. | Bounded CPU-fallback wrappers exist for `BMAPXY`, `BMXYZ`, `BOFFAXE`, and `CWIGGLER`, and the action-8 refresh matched all 19 common SDDS files at `1e-11`. A narrow deterministic cached `KICKMAP`/`UKICKMAP` CUDA prototype exists, and action 8 adds opt-in resident map-loss compaction for the no-loss-output subset through `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`. The high-count production-shaped `latticeErrors6` `UKICKMAP` gate and focused ordinary `phase62_kickmap_loss_compaction` `GKICKMAP` gate now pass with resident compaction; `phase63_kickmap_loss_output_fallback`, `phase64_kickmap_global_loss_fallback`, `latticeErrors6_loss_output`, and `latticeErrors6_global_loss` confirm loss-output/global-loss map rows still use the CPU fallback under the same flag. Radiation/ISR, offsets, tilt, yaw, and loss-output map-loss rows still fall back. | Keep the map-loss compaction path opt-in pending release policy, keep `.los`/global-loss row semantics on CPU until designed, and treat `BMAPXY`/`BMXYZ`/`BOFFAXE`, `CWIGGLER`, and unwrapped `WIGGLER` as separate future ports. |
| Pelegant and multi-GPU | Phases 8 and 19 deferred larger Pelegant timing, dynamic load balancing, MPI-aware reductions, GPU-aware particle exchange, true multi-GPU, and multi-node validation. | Fixed-rank single-node Pelegant CUDA validation exists. `load_balancing_on=1` is guarded: CPU fallback in auto mode and fail-fast in required mode. MPI scatter/gather still forces host staging. | Resume only on suitable hardware: multi-GPU mapping/timing, GPU-resident dynamic redistribution, MPI-aware reductions, and GPU-aware exchange. |
| Stochastic validation | Multiple phases defer radiation/ISR/spin or distribution-based paths until stochastic validation exists. | Static profile tooling identifies stochastic blockers. Action 10 now adds a distribution-level SDDS comparator plus initial fixed-seed CPU/GPU guards for `csbend1`, `spinTest2`, `cwiggler10_radiation`, and `uKickMap4_radiation`. | Keep stochastic CUDA paths blocked until the distribution gates are broadened beyond the initial two-seed smoke size and tied to the corresponding feature guards. |

## Resolved Older Deferrals

These items were listed as unfinished in the older implementation plan but are
now covered in the current code or later improvement phases:

- Narrow `removeInvalidParticles` and `imposeApertureData` CUDA paths exist under `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1`.
- Smoothed `WAKE`/`TRWAKE`, `WAKE,CHANGE_P0=1`, tilted single-bunch
  `TRWAKE` without spin, several `LSCDRIFT` modes, narrow serial/local
  `gpu_findFiducialTime` including selected-bunch `TMEAN`, selected-bunch
  `PMAXIMUM`, and selected-bunch `FIRST`, selected-bucket, selected-range, and skipped-`CHANGE_P0`
  match-only bunched `WAKE`/`TRWAKE` filtering, zero-length thin `RFCA` with
  GPU phase setup, `LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST`, `CHANGE_P0`, and deterministic
  offsets, nonzero-length RF-only matrix-method `RFCA`, nonzero-length RF-only
  kick-method `RFCA,N_KICKS>=1` including explicit `T_REFERENCE`, selected-bunch
  `TMEAN`, selected-bunch `PMAXIMUM`, and selected-bunch `FIRST` validated by
  `phase39_rfca_kick_rf_only`, `phase45_rf_kick_treference`,
  `phase47_rf_selected_tmean_fiducial`,
  `phase48_rf_selected_pmaximum_fiducial`, and
  `phase50_rf_first_fiducial`, the RF-only matrix-method `RFCW`
  subset including deterministic offsets and `LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST`
  fiducial modes, the RF-only kick-method `RFCW,N_KICKS>=1` subset including
  explicit `T_REFERENCE`, selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`,
  and selected-bunch `FIRST`, validated by `phase38_rfcw_kick_rf_only`,
  `phase45_rf_kick_treference`, `phase47_rf_selected_tmean_fiducial`,
  `phase48_rf_selected_pmaximum_fiducial`, and
  `phase50_rf_first_fiducial`, the narrow serial/local wake-bearing
  matrix-method `RFCW` subset including autoscaled/fixed wake bins,
  single-wake-family wake columns, explicit nonzero `T_REFERENCE`, positive
  length matrix `WAKES_AT_END=1`, full-beam or selected-bunch `TMEAN`, full-beam
  or selected-bunch `PMAXIMUM`, selected-bunch `FIRST`, and optional guarded
  LSCKICK support, including
  LSC-only cavities, validated by `phase32_rfcw_matrix_wake`,
  `phase35_rfcw_matrix_wakes_at_end`, `phase36_rfcw_lsc`,
  `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`,
  `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`,
  `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`,
  `phase50_rf_first_fiducial`, and
  `lcls1`, and the guarded serial/local wake-bearing kick-method
  `RFCW,N_KICKS>=1` subset including `WAKES_AT_END=0|1`, autoscaled/fixed wake
  bins, single-wake-family wake columns, explicit nonzero `T_REFERENCE`,
  full-beam or selected-bunch `TMEAN`, full-beam or selected-bunch `PMAXIMUM`,
  selected-bunch `FIRST`,
  plus optional guarded LSCKICK support, including LSC-only cavities, validated
  by `phase33_rfcw_kick_wake`, `phase34_rfcw_wakes_at_end`,
  `phase36_rfcw_lsc`, `phase37_rfcw_multikick`,
  `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`,
  `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`,
  `phase46_rfcw_wake_treference`, `phase49_rfcw_wake_selected_fiducial`,
  `phase50_rf_first_fiducial`, and
  `lcls0` are implemented.
- Original-mode deterministic magnet misalignment support exists for the targeted multipole, `DQCOR`, and simple non-CSR `CSBEND` subsets.
- Opt-in resident `CSRCSBEND` exists for the verified simple non-IGF, non-radiating subset.
- Benchmark reporting, fallback summaries, release-invariance tooling, and the first GitHub Actions workflow exist.
- A narrow zero-length thin-`RFCA` path with GPU phase setup, `CHANGE_P0`, and deterministic offsets plus a narrow cached `KICKMAP`/`UKICKMAP` prototype exist.
- Fixed-rank single-node `gpu-Pelegant` validation exists, with an explicit dynamic-load-balancing guard.

## Final Action Items

### 1. Close Release Infrastructure

- Run `.github/workflows/gpu-cuda-ci.yml` in GitHub Actions from a fresh checkout.
- Confirm the automatic CPU build/fallback-report job passes.
- Run the manual CUDA compile and `GPU_VERIFY` compile jobs in the CUDA container.
- Register or relabel a self-hosted CUDA runner with `self-hosted`, `linux`, `x64`, and `cuda`, then run the GPU runtime smoke.
- Review and archive `test/gpu_cuda/output/ci_artifacts/<label>/` for the release record.
- Add or document coverage for no-MPI/no-CUDA, MPI-only, and CUDA-only host combinations.
- Standardize final release artifact names, including benchmark report, fallback report, and release notes draft.

### 2. Opt-In Promotion Policy

Status: complete as a static release policy. No opt-in path should become
automatic by default in this pass. The current evidence supports two targeted
production opt-ins, while the remaining flags should stay experimental or
diagnostic until they have broader timing evidence.

| Flag | Release decision | Evidence | Do not promote yet because | Next promotion gate |
| --- | --- | --- | --- | --- |
| `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` | Recommended targeted opt-in for CSR-heavy workloads that match the guarded simple `CSRCSBEND` subset. Do not enable by default yet. | The isolated 20,000-particle, 261-pass `phase6_csr_csbend` gate is recorded as 5.27s CUDA vs 35.24s CPU, about 6.69x faster, with all 4 common SDDS files matching at `1e-11`. The May 8 finalization focused quick run matched all 4 common files with `csr=48` and only final `gpuBaseDealloc` synchronization, and the focused `GPU_VERIFY` run passed resident `ctHist`, `T1`, `T2`, and `dGamma` checks. Focused no-CSR, Stupakov-only, first-order entry/exit-edge `CSRCSBEND`, and no-state linearized `CSRDRIFT` regressions now avoid resident final/entry/drift handoffs in normal runs. The latest production smoke with the flag matched all 70 common files; `lcls0` was 1.62x faster with `csr=320`, `lcls1` was 1.76x faster with `csr=300`, and the fallback report shows zero `CSRCSBEND resident final CPU handoff` requests. | The guard intentionally excludes IGF, wake-filter resident support, radiation/ISR, backtracking, bin-once, in-element output, Derbenev criterion evaluation, and Saldin/LSC/non-Stupakov `CSRDRIFT` state. The pre-action-6 production smoke was dominated by `RFCW`, while current targeted RF runs remove the validated `clic1`, `lcls1`, and `lcls0` RFCW handoffs. Aperture/loss fallbacks, 13 CPU-owned `CSRDRIFT` transitions, and 5 CPU-owned `CSRCSBEND` transitions remain. | Keep the exact guarded subset as a documented opt-in. Do not promote to default until the remaining state-consuming `CSRDRIFT` modes, unsupported `CSRCSBEND` transitions, and diagnostic/final-state cases have broader production-shaped validation. |
| `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` | Recommended targeted opt-in for verified loss-heavy `RCOL`, `ECOL`, and ideal `SCRAPER` subsets. Do not enable by default yet. | Focused same-workload gates show useful `RCOL`, `ECOL`, and `SCRAPER` speedups from about 2.49x to 4.33x, while preserving loss semantics at `1e-11`. The May 8 finalization production smoke matched all 70 common files with the flag enabled; `maxamp1` was 1.83x, while tiny watch-heavy collimator wrappers were still 0.18x-0.22x. | The no-loss-output path now avoids loss-tail row synchronization, but `.los` consumers still need row copies to preserve loss output. The production-style `MAXAMP` margin is modest, and watch-heavy or tiny collimator wrappers remain startup/synchronization dominated. Inverted, all-loss, material-interaction, and broader invalid-particle modes still fall back. | Keep as a targeted opt-in by documented workload shape. Do not make it default without a broader production-lattice timing win and a design for `.los` row synchronization that preserves loss pass, row ordering, global coordinates, and optional spin columns. |
| `ELEGANT_GPU_ENABLE_SCMULT=1` | Keep opt-in. Do not mark as production-recommended globally and do not enable automatically. | The linear unsliced `SCMULT` slice is correct and fast for `scRing2`, with production evidence around 2.56x-2.59x from the Phase 18 baseline and a May 9 action-8 no-WATCH quick refresh showing 1.57x, 8 resident SCMULT kernels, and only final deallocation synchronization. | The evidence still comes from one production source family, even though the no-WATCH variant isolates SCMULT/RF residency from diagnostic output. Nonlinear, sliced, and bunched SCMULT remain CPU-owned, and `SCMULT` intentionally lacks general automatic `GPU_SUPPORT` metadata. | Keep opt-in for now. Use the action-8 no-WATCH refresh as the additional production-shaped validation, but require either a second source family or a release decision before enabling any automatic SCMULT eligibility. |
| `ELEGANT_GPU_ENABLE_EXACT_DRIFT=1` | Keep experimental opt-in only. | The kernel is correctness-verified, but the recorded one-minute timing did not beat CPU: 30,000 particles, 140 CPU passes in 59.78s vs 126 CUDA passes in 59.76s. | Standalone exact drift is not a speed win and can add launch/transfer overhead. | Revisit only if a larger resident workflow shows exact drift as a measured bottleneck. |
| `ELEGANT_GPU_ENABLE_CSR_HISTOGRAM=1` | Keep diagnostic/experimental opt-in. Do not recommend as a speed path now that resident CSR exists. | The histogram path is correctness-verified and useful for CSR wake-array validation, but the old same-workload gate was slower than the scratch-only path: 30.42s vs 29.98s for the 273-pass run. | It still leaves range/bin sizing and other setup CPU-owned and is superseded for speed by `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1`. | Retain for verification and development. Promote no default behavior unless a non-resident CSR workload appears where it is consistently faster. |
| `ELEGANT_GPU_ENABLE_CSR_KICK=1` | Keep developer/stepping-stone opt-in. | The prototype is correctness-verified, including CSR wake-array and `CSR_LAST_WAKE` checks. | Recorded same-workload timing was not a robust speed win, and the host-packed implementation increased CSR kernel calls and transfer time. | Use only as part of future fully resident CSR work. Do not advertise as production tuning. |
| `ELEGANT_GPU_ENABLE_APERTURE_COMPACTION=1` | Keep experimental legacy opt-in; prefer the stable parallel compaction flag for new aperture timing. | It preserves simple `MAXAMP` loss semantics and now reads back only lost tail rows in normal builds. | Earlier quick timings were slower than CPU fallback, and the stable prefix-sum path is the active aperture direction. | Retain for compatibility/debugging unless later cleanup removes it deliberately. |
| `ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE=0|1` | Treat as a sub-control of parallel aperture compaction, not a separate promotion target. Default-on under `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` is acceptable. | Accepted-device compaction reduced sync requests in `maxamp1` and is part of the verified Phase 15 stable path. | It is meaningful only when parallel aperture compaction is enabled. | Keep documented as a debugging override. |

Promotion requirements for any future default enablement:

- Fresh CPU/GPU correctness comparison and `GPU_VERIFY` quick run for the focused feature.
- Bounded same-workload timing report on a representative case, preferably near the one-minute CPU target.
- Production smoke or production-shaped wrapper coverage when the feature affects real workloads.
- Fallback summary before and after the change, with new synchronization reasons reviewed.
- Documentation update for supported and unsupported modes.
- `GPUCapable` dictionary review whenever automatic eligibility or metadata changes.

### 3. Reduce High-Value Synchronization Hotspots

Status: ranked from existing GPU output. Generated
`test/gpu_cuda/output/reports/finalize-action3-sync-hotspots.md` and the TSV
companion from the current Phase 13, Phase 14, Phase 15, and Phase 18 GPU
stderr files. No fresh CUDA runtime was run for this item.

The aggregate report intentionally combines repeated phase snapshots. Use the
aggregate counts to rank recurring reasons, and use the latest production smoke
reports to judge the current serial-production state.

| Priority | Hotspot | Evidence | Action |
| --- | --- | --- | --- |
| 1 | Remaining `RFCW` CPU-element handoffs | The generated aggregate report counted `CPU element: RFCW` 702 times across the selected Phase 13/14/15 production-smoke snapshots. Action 6 removes the validated CLIC RF-only portion plus the bounded LCLS wake-bearing portions: `clic1`, `lcls1`, and `lcls0` each reduced their prior 78 `CPU element: RFCW` syncs to zero in targeted quick runs. `phase36_rfcw_lsc` covers guarded LSCKICK integration inside the matrix-method and `N_KICKS=1` RFCW collective slices, `phase37_rfcw_multikick` extends the guarded kick-method collective slice to `N_KICKS>1` with `WAKES_AT_END=0|1`, `phase38_rfcw_kick_rf_only` covers the RF-only kick-method slice, `phase45_rf_kick_treference` covers explicit `T_REFERENCE` in the RF-only kick-method RFCA/RFCW slice, `phase47_rf_selected_tmean_fiducial` covers selected-bunch `TMEAN` in supported RFCA/RF-only RFCW slices, `phase48_rf_selected_pmaximum_fiducial` covers selected-bunch `PMAXIMUM` in the same RF-only slices, `phase49_rfcw_wake_selected_fiducial` covers selected-bunch `TMEAN`/`PMAXIMUM` in guarded wake-bearing RFCW slices, `phase50_rf_first_fiducial` covers selected-bunch `FIRST` in supported RFCA/RF-only RFCW and guarded wake-bearing RFCW slices, `phase52_rf_standing_wave_multikick_treference` covers explicit-reference multi-kick standing-wave RFCA/RFCW, and `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, and `phase46_rfcw_wake_treference` close the fixed-bin, LSC-only, single-wake-family, and explicit wake-bearing `T_REFERENCE` focused guards. The aggregate report still predates this work, so rerun production smoke before ranking any remaining RF syncs. | Treat unsupported `RFCW` modes as deferred post-action-6 candidates: distributed particles, broader LSC/RF combinations outside the documented slices, distributed/MPI fiducial reductions, and broader kick/fiducial/RF modes. Do not reopen them without focused regression, production wrapper, fallback-summary delta, and `GPU_VERIFY` gates. |
| 2 | `UKICKMAP`/`KICKMAP` map-loss fallback | The Phase 18 cache run had 337 mutable `UKICKMAP particle loss fallback` syncs in `uKickMap1`, while still matching all 4 common SDDS files and showing about 4.08x speedup. Action 8 adds an opt-in resident stable compaction path for supported `KICKMAP`/`UKICKMAP` map losses when loss-output/global-loss rows are not needed. On a 3,000-particle, 2,000-pass `uKickMap1` slice, the old fallback path had 101 `UKICKMAP particle loss fallback` syncs; the compaction run matched CPU at `1e-11` and reduced synchronization to final `gpuBaseDealloc` only. The high-count production-shaped `latticeErrors6` wrapper now runs 40 production `UKICKMAP` elements per pass; the 30,000-particle, 2-pass compaction gate matched all 4 common files at `1e-11`, kept 564 survivors, removed 80 same-workload `UKICKMAP particle loss fallback` synchronizations, and ran in 0.41s versus 0.77s CPU. Focused `phase62_kickmap_loss_compaction` covers ordinary `GKICKMAP`: its 3,000-particle, 3-pass compaction gate matched all 4 common files at `1e-11` and removed 15 same-workload `KICKMAP particle loss fallback` synchronizations. Focused `phase63_kickmap_loss_output_fallback` and `phase64_kickmap_global_loss_fallback` keep the compaction flag enabled but request `.los`/`.acc` and global loss-coordinate rows; both quick gates matched all 12 common SDDS files at `1e-11`, including `.los`, and reported 15 explicit `KICKMAP particle loss fallback` synchronizations per case. Production-shaped `latticeErrors6_loss_output` and `latticeErrors6_global_loss` repeat the same high-count `UKICKMAP` maps with `.los`/`.acc` and global rows; both matched all 12 common files at `1e-11` and reported 73 explicit `UKICKMAP particle loss fallback` synchronizations per case. | Treat `uKickMap1`, `latticeErrors6`, and `phase62_kickmap_loss_compaction` as the resident no-loss-output map-loss gates, with `phase63`/`phase64` and `latticeErrors6_loss_output`/`latticeErrors6_global_loss` as the loss-row CPU-fallback guards. Keep the path opt-in pending release policy, and keep map loss with `.los`/global loss rows, radiation/ISR, offsets, tilt, and yaw on the CPU path. |
| 3 | `CSRDRIFT` and `CSRCSBEND` handoffs | The action-3 baseline had 11 `CSRCSBEND resident final CPU handoff` syncs. Action 5 removes that resident-final marker for the validated normal runs and reduces production CPU-owned `CSRCSBEND` transitions to 5, while 13 CPU-owned `CSRDRIFT` transitions remain. The pre-resident counter report had 64 `CSRDRIFT` targets. | Treat remaining CSR CPU-element transitions as the next CSR-local cleanup after `RFCW`, or as the smaller scoped target if `RFCW` is too broad. Preserve `CSR_LAST_WAKE`, CSR output files, `CSRDRIFT` state, and verification handoff semantics. |
| 4 | Aperture/loss synchronization | Phase 13/14 `maxamp1` still show 11 `elimit_amplitudes particle loss fallback` syncs. With stable aperture compaction enabled in Phase 15, production aperture smoke matches all 70 common files and leaves 30 visible loss-tail row-copy syncs: 14 `elliptical_collimator`, 11 `elimit_amplitudes`, and 5 `rectangular_collimator`. | Continue with lost-tail and global-loss bookkeeping cleanup, but do not broaden aperture semantics until same-workload timings justify it. Avoid spending release-critical effort on tiny watch-heavy wrappers unless they expose a correctness issue. |
| 5 | Read-only diagnostics and output | Phase 15 production smoke has 46 read-only sync requests: 30 loss-tail row-copy syncs plus 16 `WATCH coordinate output` reasons. The thin-`RFCA`/SCMULT no-watch runs show that output removal can dominate specific synthetic cases, but this is not a general compute port. | Preserve exact output semantics. If a diagnostic path changes from mutable synchronization to read-only synchronization or device-side summarization, add SDDS output comparison and fallback-counter regression coverage. |
| Out of scope here | Pelegant reduction synchronization | Phase 19 Pelegant reports show large read-only reduction sync counts, but those are MPI/rank coordination issues rather than serial `gpu-elegant` synchronization hotspots. | Keep Pelegant reductions under Action 9. Do not mix them into the serial production priority list. |

Short-GPU-island policy: keep the current default `maxElements=4`. The
Phase 13 counter-only smoke had 405 sync requests; the `maxElements=4`
production smoke matched all 59 common files, skipped 296 short simple-matrix
elements on CPU, and reduced sync requests to 278. Do not raise or remove this
default without a larger production-lattice timing that still improves wall
time and does not hide a future `RFCW`/CSR residency win.

Regression policy: every synchronization-reduction patch should archive a
before/after fallback summary, same-workload CPU/GPU comparison, and
`GPU_VERIFY` result when particle arrays or output timing are affected. Ignore
`gpuBaseDealloc` as a tuning target unless it masks a real lifetime bug.

### 4. Finish Aperture And Loss Work

Status: scoped implementation and release-correctness validation complete for
the currently supported aperture/loss subset. Generated
`test/gpu_cuda/output/reports/finalize-action4-aperture-loss.md` from the
Phase 15 production aperture smoke plus fresh CUDA runs for global-loss output
and no-loss-output aperture tracking. Generated
`test/gpu_cuda/output/reports/finalize-action4-fullphase15-aperture.md` from a
fresh full aperture quick sweep with parallel compaction enabled. Generated
`test/gpu_cuda/output/reports/finalize-action4-production-smoke.md` and
`test/gpu_cuda/output/reports/finalize-action4-production-fallbacks.md` from a
fresh production smoke with parallel compaction required on the GPU run.

Code change made for this slice:

- `gpuBaseInit` now records whether the current tracking call will write a
  `losses` file.
- Stable and legacy aperture compaction copy lost-tail rows back to the host
  only when loss output is needed. Runs that do not request `losses` output can
  keep survivor tracking resident without paying the immediate lost-tail
  device-to-host copy.
- CPU global-loss-coordinate bookkeeping is also gated behind the same
  loss-output need, preserving existing `.los` semantics while avoiding hidden
  host work when no loss file will consume those rows.
- The fallback summarizer now recognizes `CPU row synchronization` messages, so
  aperture loss-tail copies appear as named reasons instead of disappearing
  into aggregate read-only counts.
- The benchmark report generator now recognizes the same row-level
  synchronization messages, keeping production report release summaries aligned
  with fallback summaries.

Validation completed:

- `make -C src HAVE_CUDA=1 -j8` completed successfully.
- `phase15_scraper_two_sided_global_loss` passed CPU/GPU SDDS comparison at
  `1e-11` with `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1`, including
  `.los` global loss-coordinate output.
- New `phase15_elimit_loss_no_output` passed CPU/GPU SDDS comparison at
  `1e-11` with no `losses` file requested. Its CUDA stderr shows no aperture
  loss-tail row synchronization; only the final `gpuBaseDealloc` sync remains.
- A full focused aperture sweep passed CPU/GPU SDDS comparison at `1e-11` with
  `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1`: `phase3_limit_loss`,
  `phase3_elimit_loss`, `phase3_rcol_loss`, `phase3_ecol_loss`,
  `phase3_scraper_loss`, `phase15_aperture_data_loss`,
  `phase15_remove_invalid_loss`, `phase15_rcol_open_loss`,
  `phase15_ecol_mixed_loss`, `phase15_ecol_open_global_loss`,
  `phase15_rcol_open_global_loss`, `phase15_scraper_global_loss`,
  `phase15_scraper_two_sided_global_loss`, `phase15_elimit_loss_no_output`,
  and `aperture_loss`. This covers `.acc`, `.los`, global-loss-coordinate,
  `aperture_data`, `removeInvalidParticles`, and no-loss-output behavior.
- Fresh production smoke with `ELEGANT_GPU_MODE=required` and
  `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` passed CPU/GPU SDDS
  comparison at `1e-11` for all 70 common files. Same-workload speedups ranged
  from `collimate3` 0.18x to `dqcor1` 7.30x; the aperture-relevant cases were
  `maxamp1` 1.83x, `collimate1` 0.21x, `collimate2` 0.22x, and `collimate3`
  0.18x.

Deferred aperture/loss work after action 4:

- Loss-output row synchronization remains by design for `.los` consumers.
  Reduce those cases further only with a design that preserves loss pass, lost
  row ordering, global loss coordinates, and optional spin columns.
- Treat the verified aperture subset as a targeted production opt-in. The fresh
  production smoke supports documented workload-shape recommendations, not
  automatic default behavior.
- Add measured cases before broadening inverted `RCOL`/`ECOL`, all-loss,
  material-interaction, and remaining global-loss-coordinate modes.
- Broaden `removeInvalidParticles` beyond the identity-`RFCA` trigger only
  after the surrounding element semantics are verified.
- Add additional production aperture wrappers only after quick-mode runtime is
  bounded.

### 5. Finish CSR Work

Status: finalization validation and the first cleanup slices are complete.
Resident `CSRCSBEND` remains a targeted production opt-in, not a default. The
completed slices cover no-CSR `CSRDRIFT` exact/linear-drift residency,
Stupakov-only CSRDRIFT final-prep avoidance, and checked CUDA initial/final
transforms for first-order entrance/exit-edge `CSRCSBEND` sections. Generated
`test/gpu_cuda/output/reports/finalize-action5-csr-production-smoke.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-production-fallbacks.md`,
and `test/gpu_cuda/output/reports/finalize-action5-csr-focused-fallbacks.md`
from the first fresh resident-CSR runs. Generated
`test/gpu_cuda/output/reports/finalize-action5-csr-noop-drift-aperture-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-state-consuming-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-noop-production-smoke.md`,
and `test/gpu_cuda/output/reports/finalize-action5-csr-noop-production-fallbacks.md`
from the follow-up no-op CSRDRIFT guard slice. Generated
`test/gpu_cuda/output/reports/finalize-action5-csr-stupakov-prep-skip-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-edge-final-focused-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-edge-final-production-smoke.md`,
and
`test/gpu_cuda/output/reports/finalize-action5-csr-edge-final-production-fallbacks.md`
from the Stupakov-only and first-order edge-final cleanup slices. Generated
`test/gpu_cuda/output/reports/finalize-action5-csr-entry-edge-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action5-csr-entry-edge-production-smoke.md`,
and
`test/gpu_cuda/output/reports/finalize-action5-csr-entry-edge-production-fallbacks.md`
from the first-order entrance-edge cleanup slice. Generated
`test/gpu_cuda/output/reports/finalize-action5-csr-linear-drift-fallbacks.md`
and the TSV companion from the no-state linearized CSRDRIFT cleanup slice.

Code and harness changes made for these slices:

- `production_smoke.sh` now preserves and reports the CSR runtime controls
  `ELEGANT_GPU_ENABLE_CSR_HISTOGRAM`, `ELEGANT_GPU_ENABLE_CSR_KICK`,
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT`, `ELEGANT_GPU_MIN_CSR_PARTICLES`, and
  `ELEGANT_GPU_MIN_CSR_BINS`, so production smoke reports capture the CSR
  opt-in state instead of relying on external run notes.
- The production benchmark report parser now recognizes row-level CPU
  synchronization messages, keeping release summaries aligned with fallback
  summaries when CSR or aperture diagnostics request named row copies.
- Resident CSR final-state prep now distinguishes no-state `CSRDRIFT`,
  Stupakov-only state-consuming `CSRDRIFT`, and `CSRDRIFT` sections that still
  need full `CSRCSBEND` final-state preparation. A following
  `CSRDRIFT,CSR=0,LSC_BINS=0` no longer forces unused CSRDRIFT-state
  preparation or a resident `CSRCSBEND` final CPU handoff. A following
  Stupakov-only `CSRDRIFT` can use the existing CPU drift with minimal
  `csrWake` metadata instead of forcing the full final-prep handoff.
- With `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1`, the no-CSR `CSRDRIFT` subset now
  stays GPU-resident as an exact-drift kernel when `LINEARIZE=0` and as a
  linear drift kernel when `LINEARIZE=1`.
  State-consuming `CSRDRIFT` modes still force the existing CPU path.
- The checked resident `CSRCSBEND` simple-final CUDA kernel now supports
  first-order `EDGE2_EFFECTS=1` exit-edge focusing for `EDGE_ORDER<=1`. This
  covers the LCLS-style resident final transform that previously forced a
  `CSRCSBEND resident final CPU handoff`.
- The checked resident `CSRCSBEND` simple-entry CUDA kernel now supports
  first-order `EDGE1_EFFECTS=1` entrance-edge focusing for `EDGE_ORDER<=1`.
  This covers LCLS-style entrance edges with finite `HGAP/FINT` psi terms when
  other resident CSR guards pass.
- Added `phase14_csr_noop_drift_aperture`, which combines aperture loss before
  resident `CSRCSBEND` with a no-CSR `CSRDRIFT` section.
- Added `phase14_csr_entry_edge`, which covers resident `CSRCSBEND` first-order
  entrance and exit edge focusing without CSRDRIFT or output-side dependencies.
- Added `phase14_csr_linear_drift`, which covers no-state
  `CSRDRIFT,CSR=0,LSC_BINS=0,LINEARIZE=1` residency between resident
  `CSRCSBEND` sections.

Validation completed:

- `make -C src HAVE_CUDA=1 -j8` completed successfully.
- `make -C src HAVE_CUDA=1 GPU_VERIFY=1 -j8` completed successfully.
- `make -C src -j8` completed successfully.
- Latest production smoke with `ELEGANT_GPU_MODE=required` and
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed CPU/GPU SDDS comparison at
  `1e-11` for all 70 common files. Same-workload speedups ranged from
  `collimate3` 0.16x to `dqcor1` 7.16x; the CSR-heavy LCLS wrappers were
  `lcls0` 1.62x and `lcls1` 1.76x.
- The latest production run reported `csr=320` for `lcls0` and `csr=300` for
  `lcls1`. The fallback summary has zero `CSRCSBEND resident final CPU handoff`
  requests, while the remaining CSR synchronization is 5 CPU-owned
  `CSRCSBEND` transitions and 13 CPU-owned `CSRDRIFT` transitions. `RFCW`
  remains the dominant production synchronization target with 234 requests.
- Focused `phase6_csr_csbend` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 4 common SDDS files at
  `1e-11`. The normal CUDA stderr showed `csr=48` and only final
  `gpuBaseDealloc` synchronization.
- Focused `phase6_csr_csbend` `GPU_VERIFY` validation passed resident
  `ctHist`, `T1`, `T2`, and `dGamma` checks. The extra synchronization in that
  run is verification-only: 16 resident histogram verification requests plus
  4 resident final CPU handoffs.
- Focused `phase14_csr_noop_drift_aperture` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` and
  `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` passed all 5 common SDDS
  files at `1e-11`, including `.los`. Its normal CUDA stderr showed
  `csr=24`, `exactDrift=1`, one aperture loss-tail row copy, and no
  `CSRCSBEND resident final CPU handoff`.
- Focused `phase14_csr_noop_drift_aperture` `GPU_VERIFY` validation passed
  resident CSR histogram, `T1`, `T2`, `dGamma`, and `track_through_driftCSR`
  checks. Verification-only CSR final handoffs remain expected.
- Focused Stupakov-only `phase14_csr_last_wake` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 5 common SDDS files at
  `1e-11`. The final normal CUDA stderr showed `csr=48`, two CPU-owned
  `CSRDRIFT` transitions, and no `CSRCSBEND resident final CPU handoff`
  requests. The focused `GPU_VERIFY` run passed resident CSR histogram, `T1`,
  `T2`, `dGamma`, and drift checks; verification-only final handoffs remain
  expected.
- Focused `phase14_csr_saldin54` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 6 common SDDS files at
  `1e-11` and still reported 4 `CSRCSBEND resident final CPU handoff` requests,
  confirming that Saldin54 remains on the conservative final-prep path.
- Direct LCLS quick checks with `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` and
  `ELEGANT_GPU_MIN_CSR_BINS=1` matched CPU output at `1e-11`: `lcls0` matched
  all 15 common files with `csr=320`, and `lcls1` matched all 16 common files
  with `csr=300`. Both runs avoided the resident final handoff marker.
- Focused `phase14_csr_entry_edge` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 4 common SDDS files at
  `1e-11`. The normal CUDA stderr showed `csr=24`, no
  `CSRCSBEND resident initial CPU fallback`, and no
  `CSRCSBEND resident final CPU handoff`.
- Focused `phase14_csr_entry_edge` `GPU_VERIFY` validation passed resident CSR
  histogram, `T1`, `T2`, and `dGamma` checks. The entry kernel is exercised by
  the normal CPU/GPU comparison; `GPU_VERIFY` intentionally uses CPU entry
  transforms before resident body checks.
- Focused `phase14_csr_linear_drift` quick validation with
  `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 4 common SDDS files at
  `1e-11`. The normal CUDA stderr showed `csr=24`, `exactDrift=0`,
  `linearDrift=1`, no `CPU element: CSRDRIFT` fallback, and only the final
  `gpuBaseDealloc` synchronization.
- Focused `phase14_csr_linear_drift` `GPU_VERIFY` validation passed
  `track_through_driftCSR` comparison with zero max difference, plus the
  resident CSR histogram, `T1`, `T2`, and `dGamma` checks.
- Direct LCLS quick checks after the entrance-edge slice showed `lcls0`
  removing `BX01A` from the CPU-owned `CSRCSBEND` list. `lcls1` still keeps
  `BX01A` on CPU because that production element also enables Derbenev
  criterion evaluation and CSR output; `BW1A` and `BX31A` remain CPU-owned in
  both LCLS wrappers because they enable `ISR=1`.
  A source scan of the curated LCLS/CLIC production wrappers found no
  `CSRDRIFT,LINEARIZE=1` declarations, so the no-state linear drift slice does
  not change the latest production CSR fallback counts.

Deferred CSR work after action 5:

- Keep resident `CSRCSBEND` as a documented targeted opt-in. The fresh
  production smoke supports use on matching CSR-heavy workloads, but not
  automatic default enablement.
- Reduce CPU-owned `CSRDRIFT` execution for the remaining Saldin, LSC, and
  non-Stupakov state-consuming modes, CPU-owned unsupported `CSRCSBEND`
  transitions, Derbenev/output diagnostics, ISR/radiation CSR bends, and
  diagnostic/final CSR state. This must preserve `CSR_LAST_WAKE`, CSR
  output files, `CSRDRIFT` drift state, and `GPU_VERIFY` handoff semantics.
- Broaden CSR plus aperture/loss validation beyond the new upstream-`ECOL`
  aperture-loss regression only after a production-shaped case shows value.
- Revisit integrated Green's function only after the CPU-side IGF crash noted
  in the original plan is diagnosed.
- Extend CSR verification for Stupakov, Saldin, wake filters, and IGF state
  only as each mode moves toward CUDA residency.
- Add more LCLS-style production CSR wrappers after runtime is measured, and
  use their fallback summaries to decide whether CSR handoff reduction should
  outrank `RFCW` or `UKICKMAP` work.

### 6. Finish Collective And RF Work

Status: complete for the current serial/local finalization scope. The closed
scope includes the CLIC-shaped RF-only `RFCW` implementation slice, RF-only
`RFCW` deterministic-offset, `LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST` fiducial-mode,
and kick-method slices including explicit kick-method `T_REFERENCE`,
wake-bearing matrix-method and `N_KICKS>=1` kick-method `RFCW` slices
including fixed wake bins, explicit nonzero `T_REFERENCE`,
single-wake-family cavities, matrix-method `WAKES_AT_END=1`, kick-method
`WAKES_AT_END=0|1`, LSC-only cavities, and `PMAXIMUM` fiducial-mode,
zero-length thin-`RFCA` phase/setup plus `CHANGE_P0`,
`LIGHT`/`TMEAN`/`PMAXIMUM`/`FIRST` fiducial-mode, and deterministic-offset
slices, nonzero-length RF-only matrix-method and kick-method `RFCA` slices,
serial/local selected-bunch `TMEAN` and `PMAXIMUM` phase setup for supported
RFCA and RF-only RFCW slices, guarded wake-bearing selected-bunch
`TMEAN`/`PMAXIMUM` RFCW fiducialization, serial/local selected-bunch `FIRST`
phase setup for the supported RFCA, RF-only RFCW, and guarded wake-bearing
RFCW slices, narrow matrix-method, single-kick, explicit-reference multi-kick
`STANDING_WAVE=1` RFCA/RFCW slices, non-explicit fiducial multi-kick
`STANDING_WAVE=1` RFCA/RFCW slices, selected-bucket, selected-range, and
skipped-`CHANGE_P0` match-only bunched wake filter slices, plus guarded
LSCKICK support inside the wake-bearing matrix-method and kick-method
`RFCW,N_KICKS>=1` collective slices. Distributed particles, distributed/MPI
fiducial reductions, broader unsupported RF/RFCW/LSC combinations, and
cuFFT-backed collective work are explicitly deferred.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-mode-profile.md` and the
TSV companion from a static collective/RF source scan plus the latest
action-5 production stderr. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-baseline-fallbacks.md`
and TSV from the focused synthetic `rfcw` quick case. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-rf-only-focused.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-clic1.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-clic1-fallbacks.md`,
and the fallback TSV companions from the focused RF-only and CLIC production
validation runs. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-offset-focused.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-offset-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-offset-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-offset-regression-fallbacks.md`,
and TSV companions from the focused RF-only offset validation and RF-only
regression sweep. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fiducial-modes.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fiducial-modes-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fiducial-modes-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fiducial-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fiducial-regression-fallbacks.md`,
and TSV companions from the focused RF-only RFCW fiducial-mode validation and
expanded RF-only RFCW regression sweep. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-lcls1.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wake-lcls1-fallbacks.md`,
and TSV companions from the focused wake-bearing matrix-method RFCW validation,
expanded RFCW regression sweep, and bounded `lcls1` production validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-phase33.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-phase33-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-phase33-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-lcls0.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-lcls0-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-wake-regression-fallbacks.md`,
and TSV companions from the focused kick-method RFCW validation, bounded
`lcls0` production validation, and the earlier five-case RFCW regression sweep.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wakes-at-end-phase34.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wakes-at-end-phase34-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wakes-at-end-phase34-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wakes-at-end-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wakes-at-end-regression-fallbacks.md`,
and TSV companions from the focused `WAKES_AT_END` kick-method RFCW validation
and six-case RFCW regression sweep. Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wakes-at-end-phase35.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wakes-at-end-phase35-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wakes-at-end-phase35-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wakes-at-end-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-matrix-wakes-at-end-regression-fallbacks.md`,
and TSV companions from the focused matrix-method `WAKES_AT_END` RFCW
validation and seven-case RFCW regression sweep.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-phase36.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-phase36-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-phase36-verify-fallbacks.md`,
and TSV companions from the focused guarded RFCW LSCKICK validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-multikick-phase37.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-multikick-phase37-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-multikick-phase37-verify-fallbacks.md`,
and TSV companions from the focused guarded multi-kick RFCW collective
validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-rf-only-phase38.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-rf-only-phase38-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-kick-rf-only-phase38-verify-fallbacks.md`,
and TSV companions from the focused RF-only kick-method RFCW validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-change-p0.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-change-p0-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-change-p0-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-fiducial-modes.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-fiducial-modes-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-fiducial-modes-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-fiducial-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-fiducial-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-offset.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-offset-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-offset-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-offset-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-thin-offset-regression-fallbacks.md`,
and TSV companions from the focused zero-length thin-`RFCA,CHANGE_P0`,
thin-`RFCA` fiducial-mode and offset validation, and thin-`RFCA` regression
sweeps.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-rf-only.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-final-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-final-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-fiducial-modes.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-fiducial-modes-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-matrix-fiducial-modes-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-fiducial-regression.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-fiducial-regression-fallbacks.md`,
and TSV companions from the focused nonzero-length RF-only matrix-method
`RFCA` validation, matrix-`RFCA` fiducial-mode validation, and RFCA regression
sweeps.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfca-kick-rf-only-phase39.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-kick-rf-only-phase39-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-kick-rf-only-phase39-verify-fallbacks.md`,
and TSV companions from the focused RF-only kick-method RFCA validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-pmaximum-phase40.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-pmaximum-phase40-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-pmaximum-phase40-verify-fallbacks.md`,
and TSV companions from the focused serial/local RF-only `PMAXIMUM`
fiducialization validation across RFCA and RFCW slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-pmaximum-phase41.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-pmaximum-phase41-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-pmaximum-phase41-verify-fallbacks.md`,
and TSV companions from the focused wake-bearing RFCW `PMAXIMUM`
fiducialization validation across the guarded matrix-method and kick-method
collective slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fixed-wake-bins-phase42.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fixed-wake-bins-phase42-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-fixed-wake-bins-phase42-verify-fallbacks.md`,
and TSV companions from the focused wake-bearing RFCW fixed-wake-bin validation
across the guarded matrix-method and kick-method collective slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-only-phase43.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-only-phase43-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-lsc-only-phase43-verify-fallbacks.md`,
and TSV companions from the focused RFCW LSC-only validation across the guarded
matrix-method and kick-method collective slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-single-wake-planes-phase44.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-single-wake-planes-phase44-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-single-wake-planes-phase44-verify-fallbacks.md`,
and TSV companions from the focused RFCW single-wake-family validation across the
guarded matrix-method and kick-method collective slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-treference-phase46.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-treference-phase46-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-treference-phase46-verify-fallbacks.md`,
and TSV companions from the focused wake-bearing RFCW explicit `T_REFERENCE`
validation across the guarded matrix-method and kick-method collective slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-kick-treference-phase45.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-kick-treference-phase45-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-kick-treference-phase45-verify-fallbacks.md`,
and TSV companions from the focused RF-only kick-method RFCA/RFCW explicit
`T_REFERENCE` validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-tmean-phase47.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-tmean-phase47-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-tmean-phase47-verify-fallbacks.md`,
and TSV companions from the focused serial/local selected-bunch `TMEAN`
fiducialization validation across supported RFCA and RF-only RFCW slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-pmaximum-phase48.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-pmaximum-phase48-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-selected-pmaximum-phase48-verify-fallbacks.md`,
and TSV companions from the focused serial/local selected-bunch `PMAXIMUM`
fiducialization validation across supported RFCA and RF-only RFCW slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-selected-fiducial-phase49.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-selected-fiducial-phase49-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-wake-selected-fiducial-phase49-verify-fallbacks.md`,
and TSV companions from the focused guarded wake-bearing RFCW selected-bunch
`TMEAN`/`PMAXIMUM` validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-first-phase50.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-first-phase50-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-first-phase50-verify-fallbacks.md`,
and TSV companions from the focused serial/local selected-bunch `FIRST`
validation across supported RFCA, RF-only RFCW, and guarded wake-bearing RFCW
slices.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-phase51.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-phase51-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-phase51-verify-fallbacks.md`,
and TSV companions from the focused narrow matrix-method and single-kick
`STANDING_WAVE=1` RFCA/RFCW validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-multikick-phase52.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-multikick-phase52-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rf-standing-wave-multikick-phase52-verify-fallbacks.md`,
and TSV companions from the focused explicit-reference multi-kick
`STANDING_WAVE=1` RFCA/RFCW validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfca-standing-wave-multikick-phase53.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-standing-wave-multikick-phase53-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfca-standing-wave-multikick-phase53-verify-fallbacks.md`,
and TSV companions from the focused non-explicit fiducial multi-kick
`STANDING_WAVE=1` RFCA validation.
Generated
`test/gpu_cuda/output/reports/finalize-action6-rfcw-standing-wave-multikick-phase54.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-standing-wave-multikick-phase54-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-standing-wave-multikick-phase54-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-rfcw-standing-wave-multikick-phase54-regression-fallbacks.md`,
and TSV companions from the focused non-explicit fiducial multi-kick
`STANDING_WAVE=1` RFCW validation plus existing RFCW kick fiducial regression
sweep.
Generated
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-select.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-select-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-select-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-range.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-range-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-filter-range-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-range-regression-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-change-p0-skip.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-change-p0-skip-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-change-p0-skip-verify-fallbacks.md`,
`test/gpu_cuda/output/reports/finalize-action6-bunched-wake-match-regression-fallbacks.md`,
and TSV companions from the focused selected-bucket, selected-range, and
skipped-`CHANGE_P0` bunched wake validation plus the bunched-wake regression
sweeps.

Findings and action split:

- The pre-action-6 production smoke had 234 `RFCW` CPU-element syncs: 78 each
  in `lcls0`, `lcls1`, and `clic1`.
- CLIC's production `RFCW` syncs are the narrower first implementation
  candidate: `010_CAV`, `050_CAV`, and `110_CAV` are RF-only `RFCW` sections
  with end focusing, no active wake columns, and no LSC. `110_CAV` accounts for
  64 of the 78 CLIC syncs and does not use `CHANGE_P0`; `010_CAV` and
  `050_CAV` account for the remaining 14 and use `CHANGE_P0=1`.
- LCLS `RFCW` is a larger project, not a small cavity port. Both `lcls0` and
  `lcls1` use longitudinal plus transverse wake files, autoscaled bins,
  smoothing, interpolation, `CHANGE_P0`, and entrance/exit focusing. The
  matrix-method `lcls1` shape and the `N_KICKS=1` kick-method `lcls0` shape
  are now covered for serial/local wake-bearing `RFCW`, including
  deterministic `DX/DY` offsets in the focused regressions. The
  `phase34_rfcw_wakes_at_end` regression extends the guarded `N_KICKS=1` shape
  to CPU-equivalent `WAKES_AT_END=1` ordering, while
  `phase35_rfcw_matrix_wakes_at_end` validates the positive-length matrix-method
  behavior where CPU tracking applies wakes after the RF matrix independent of
  the flag. `phase36_rfcw_lsc` adds guarded fixed-bin LSCKICK coverage to the
  matrix-method and `N_KICKS=1` wake-bearing slices,
  `phase42_rfcw_fixed_wake_bins` confirms fixed wake-bin support across the
  guarded matrix-method and kick-method collective slices,
  `phase43_rfcw_lsc_only` confirms the same guarded LSCKICK path works without
  active wake columns, `phase44_rfcw_single_wake_planes` confirms
  longitudinal-only and transverse-only wake-family support across the same
  guarded collective slices, `phase46_rfcw_wake_treference` confirms explicit
  nonzero `T_REFERENCE` for wake-bearing matrix-method and kick-method
  collective slices, and
  `phase37_rfcw_multikick` extends the guarded kick-method collective path to
  `N_KICKS>1` for both `WAKES_AT_END=0` and `WAKES_AT_END=1`. Broader
  unsupported RF shapes remain outside this slice.
- RF-only kick-method `RFCW` is now covered by the same RF section loop without
  invoking wake or LSC kernels. `phase38_rfcw_kick_rf_only` validates
  `N_KICKS>0`, `LIGHT` and serial/local `TMEAN` fiducialization,
  deterministic offsets, entrance/exit focusing, `CHANGE_P0=1`, and no-op
  `WAKES_AT_END=1` handling.
- RF-only kick-method `RFCA` is now covered by the same RF section kernels
  without wake, LSC, or RFCW wrapper state. `phase39_rfca_kick_rf_only`
  validates `N_KICKS>0`, section-center `LIGHT` and serial/local `TMEAN`
  fiducialization, per-section RF phase advance, deterministic offsets,
  entrance/exit focusing, and `CHANGE_P0=1`.
- Explicit nonzero `T_REFERENCE` setup is now covered for the RF-only
  kick-method RFCA and RFCW section loops by `phase45_rf_kick_treference`, with
  deterministic offsets, focusing, `CHANGE_P0`, and no-op RFCW
  `WAKES_AT_END=1`.
- Serial/local `PMAXIMUM` fiducialization is now covered for the supported
  thin, matrix-method, and kick-method RFCA slices plus the RF-only and
  wake-bearing matrix-method and kick-method RFCW slices.
  `phase47_rf_selected_tmean_fiducial` confirms serial/local selected-bunch
  `TMEAN` for supported RFCA and RF-only RFCW slices,
  `phase48_rf_selected_pmaximum_fiducial` confirms serial/local selected-bunch
  `PMAXIMUM` for those same RF-only slices, and
  `phase49_rfcw_wake_selected_fiducial` confirms selected-bunch `TMEAN` and
  `PMAXIMUM` inside the guarded wake-bearing RFCW matrix-method and multi-kick
  collective slices. `phase50_rf_first_fiducial` confirms selected-bunch
  `FIRST` for supported RFCA, RF-only RFCW, and guarded wake-bearing RFCW
  slices. `phase51_rf_standing_wave_single` confirms the narrow
  `STANDING_WAVE=1` subset where CPU and CUDA section phase formulas are
  identical: matrix-method cavities and `N_KICKS=1` kick-method cavities.
  `phase52_rf_standing_wave_multikick_treference` confirms the next bounded
  subset: multi-kick standing-wave RFCA/RFCW cavities with explicit
  `T_REFERENCE`, including RF-only RFCA, RF-only RFCW, and guarded wake-bearing
  RFCW. `phase53_rfca_standing_wave_multikick_fiducial` confirms the RFCA-only
  non-explicit fiducial subset with serial/local `LIGHT`, selected-bunch
  `TMEAN`, selected-bunch `PMAXIMUM`, and selected-bunch `FIRST`.
  `phase54_rfcw_standing_wave_multikick_fiducial` confirms the analogous
  RFCW subset across RF-only and guarded wake-bearing kick-method sections.
  Distributed particles, distributed/MPI fiducial reductions, and broader
  bunch-restricted modes remain CPU-owned.
- The focused synthetic `rfcw` quick case passed CPU/GPU SDDS comparison at
  `1e-11`. Its normal CUDA stderr showed one matrix kernel, seven short-GPU-
  island CPU skips, and only final `gpuBaseDealloc` synchronization, so it is a
  correctness baseline but not representative of the production RFCW boundary
  cost.

Code and harness changes made for these slices:

- Added a narrow RF-only matrix-method `RFCW` CUDA path. It supports serial/local
  cavities with no active wake columns, no LSC, no cavity `Q`, no
  linearized/backtracking mode, `N_KICKS<=0`, optional `STANDING_WAVE=1`,
  entrance/exit
  focusing, deterministic `DX/DY` offsets, and `CHANGE_P0` through the existing
  GPU central-momentum matching helper.
- Added GPU-side first-use phase setup for this RFCW subset. The guard supports
  explicit `T_REFERENCE`, `LIGHT`, serial/local full-beam or selected-bunch
  `TMEAN`, serial/local full-beam or selected-bunch `PMAXIMUM`, and
  serial/local `FIRST`
  fiducialization, including nonzero `sOffset`; the shared selected-bunch
  `TMEAN`, `PMAXIMUM`, and `FIRST` helpers are validated for the RF-only RFCW
  subset by `phase47_rf_selected_tmean_fiducial`,
  `phase48_rf_selected_pmaximum_fiducial`, and
  `phase50_rf_first_fiducial`. Distributed Pelegant particles remain on CPU.
- Added a narrow serial/local wake-bearing matrix-method `RFCW` CUDA path. It
  composes the resident RF matrix kernel with the existing `WAKE` and `TRWAKE`
  CUDA kernels for cavities with longitudinal and/or transverse wake columns,
  optional guarded fixed-bin LSCKICK, no cavity `Q`, no `CHANGE_T`, no
  kick-method RF, no body-focus/linearized/backtracking mode, and optional
  `STANDING_WAVE=1`.
  The guard supports explicit `T_REFERENCE`, `LIGHT`, serial/local full-beam or
  selected-bunch `TMEAN`, serial/local full-beam or selected-bunch `PMAXIMUM`,
  and serial/local `FIRST` phase setup.
  Positive-length matrix method tracking applies collective kicks after the RF
  matrix for both `WAKES_AT_END=0` and `WAKES_AT_END=1`, matching CPU behavior.
  Deterministic `DX/DY` offsets stay subtracted through the collective kicks
  and are restored before invalid-particle removal and final `CHANGE_P0`
  matching.
- Added a narrow serial/local wake-bearing kick-method `RFCW,N_KICKS>=1` CUDA
  path. It loops over the CPU kick sections with per-section length and
  voltage, applies the traveling-wave `-ik*dtLight` RF phase advance, omits
  that section phase advance for standing-wave sections, keeps entrance
  focusing on the first section and exit focusing on the final section, applies
  the existing `WAKE` and `TRWAKE` CUDA kernels plus optional guarded fixed-bin
  LSCKICK per section, and preserves CPU-equivalent collective ordering for
  `WAKES_AT_END=0|1`. The path covers cavities with no cavity `Q`, no
  `CHANGE_T`, no body-focus/linearized/backtracking mode, and optional
  `STANDING_WAVE=1` for `N_KICKS=1`, for `N_KICKS>1` with explicit
  `T_REFERENCE`, or for `N_KICKS>1` with the same supported serial/local
  fiducial modes. Phase setup supports explicit `T_REFERENCE`, `LIGHT`,
  serial/local full-beam or selected-bunch `TMEAN`, serial/local full-beam or
  selected-bunch `PMAXIMUM`, or serial/local `FIRST`.
- Added a narrow serial/local RF-only kick-method `RFCW,N_KICKS>=1` CUDA
  eligibility path. It reuses the same RF section loop while requiring no
  active wake columns, no LSC, no cavity `Q`, no `CHANGE_T`, and no
  body-focus/linearized/backtracking mode. `STANDING_WAVE=1` is accepted for
  the single-kick slice, the explicit-reference multi-kick slice, and the
  supported non-explicit fiducial multi-kick slice where CUDA can use the
  CPU-equivalent section-center fiducial offset and skip the traveling-wave
  section phase advance to match CPU standing-wave semantics.
- Reused the existing LSC binning and kick kernels for RFCW LSCKICK support and
  added a CUDA reduction for the RF `dgamma/gamma` average used by the CPU LSC
  kick-spacing guard. The guarded LSCKICK subset requires fixed even
  `LSC_BINS`, supported interpolation, ordered low/high-frequency cutoff
  filters, and nonzero `LSC_RADIUS_FACTOR`.
- Added GPU-side first-use phase setup for the zero-length thin `RFCA` subset.
  The guard supports explicit `T_REFERENCE`, `LIGHT`, serial/local full-beam
  `TMEAN`, serial/local selected-bunch `TMEAN`, and serial/local full-beam or
  selected-bunch `PMAXIMUM`, plus serial/local `FIRST`, for no-wake,
  no-`CHANGE_T`, no-focus thin cavities with optional `STANDING_WAVE=1`,
  avoiding the prior first-pass CPU fiducialization setup fallback.
- Extended the zero-length thin `RFCA` CUDA path to support `CHANGE_P0=1`
  through the existing GPU central-momentum matching helper after the thin RF
  kick and invalid-particle check.
- Extended the zero-length thin `RFCA` CUDA path to accept deterministic
  `DX/DY` offsets for the same no-focus subset. CPU tracking subtracts and
  restores these offsets around an RF kick that does not depend on transverse
  position, so the CUDA path can preserve the same net semantics without a CPU
  handoff.
- Added a narrow nonzero-length RF-only matrix-method `RFCA` CUDA path by
  reusing the existing RF-only cavity matrix kernel. It supports serial/local
  no-wake/no-LSC `RFCA` cavities with `N_KICKS<=0`, no cavity `Q`, no
  `CHANGE_T`, no body-focus/linearized/locked/backtracking mode, optional
  `STANDING_WAVE=1`, explicit `T_REFERENCE`, `LIGHT`, serial/local full-beam or selected-bunch
  `TMEAN`, serial/local full-beam or selected-bunch `PMAXIMUM`, or serial/local
  `FIRST`
  fiducialization, entrance/exit focusing, `CHANGE_P0`, and deterministic
  `DX/DY` offsets.
- Added a narrow nonzero-length RF-only kick-method `RFCA,N_KICKS>=1` CUDA
  path by reusing the existing RF kick section kernels. It supports
  serial/local no-wake/no-LSC cavities with no cavity `Q`, no `CHANGE_T`, no
  body-focus/linearized/locked/backtracking mode, optional `STANDING_WAVE=1`
  for `N_KICKS=1`, for `N_KICKS>1` with explicit `T_REFERENCE`, or for
  `N_KICKS>1` with supported non-explicit RFCA fiducial setup, and phase setup
  from explicit `T_REFERENCE`, `LIGHT`, serial/local
  full-beam or selected-bunch `TMEAN`, or serial/local full-beam or
  selected-bunch `PMAXIMUM`, or serial/local `FIRST` fiducialization,
  per-section length, voltage, and phase advance, the CPU-equivalent
  section-center fiducial offset, entrance/exit focusing, `CHANGE_P0`, and
  deterministic `DX/DY` offsets.
- Added a CUDA `PMAXIMUM` fiducial reduction that mirrors the serial CPU
  pmaximum selection semantics for supported RF cavities and accepts the same
  nonzero RF-section `sOffset` used by matrix-method and kick-method phase
  setup. It now filters by particle-ID range for serial/local selected-bunch
  fiducialization while preserving the serial baseline and tie-breaking
  semantics.
- Added a CUDA `FIRST` fiducial reduction for supported RF cavities. It mirrors
  the serial CPU first-row semantics, accepts the same nonzero RF-section
  `sOffset`, and filters by particle-ID range for serial/local selected-bunch
  fiducialization. The CPU `FIRST` selected-particle branch now stops after the
  first matching particle instead of falling through to the no-match error.
- Relaxed RFCA/RFCW CUDA eligibility for the narrow `STANDING_WAVE=1` slice:
  matrix-method cavities, `N_KICKS=1` kick-method cavities, explicit
  `T_REFERENCE` multi-kick kick-method cavities, and multi-kick kick-method
  RFCA/RFCW cavities with supported non-explicit fiducial setup now remain
  resident using the CPU-equivalent section-center phase offset.
- Added `phase21_rfcw_rf_only`, a focused CLIC-style RF-only `RFCW` regression
  with end focusing plus both `CHANGE_P0=0` and `CHANGE_P0=1` cavities.
- Added `phase25_rfcw_rf_only_offset`, a focused RF-only `RFCW` regression
  that applies deterministic `DX/DY` offsets around the matrix-method cavity
  transform, including a `CHANGE_P0=1` cavity.
- Added `phase31_rfcw_rf_only_fiducial_modes`, a focused RF-only `RFCW`
  regression that exercises `FIDUCIAL="light"` and serial/local
  `FIDUCIAL="tmean"` phase setup with deterministic `DX/DY` offsets,
  entrance/exit focusing, and `CHANGE_P0=1`.
- Added `phase32_rfcw_matrix_wake`, a focused wake-bearing matrix-method
  `RFCW` regression with longitudinal/transverse wake columns, autoscaled
  bins, interpolation, smoothing, deterministic `DX/DY` offsets, entrance/exit
  focusing, and `CHANGE_P0=1`.
- Added `phase33_rfcw_kick_wake`, a focused wake-bearing `RFCW,N_KICKS=1`
  regression with longitudinal/transverse wake columns, autoscaled bins,
  interpolation, smoothing, deterministic `DX/DY` offsets, entrance/exit
  focusing, and `CHANGE_P0=1`.
- Added `phase34_rfcw_wakes_at_end`, a focused wake-bearing
  `RFCW,N_KICKS=1,WAKES_AT_END=1` regression with longitudinal/transverse wake
  columns, autoscaled bins, interpolation, smoothing, deterministic `DX/DY`
  offsets, entrance/exit focusing, and `CHANGE_P0=1`.
- Added `phase35_rfcw_matrix_wakes_at_end`, a focused wake-bearing
  matrix-method `RFCW,WAKES_AT_END=1` regression with longitudinal/transverse
  wake columns, autoscaled bins, interpolation, smoothing, deterministic
  `DX/DY` offsets, entrance/exit focusing, and `CHANGE_P0=1`.
- Added `phase36_rfcw_lsc`, a focused guarded RFCW LSCKICK regression that
  combines longitudinal/transverse wake columns with filtered fixed-bin LSC in
  one matrix-method RFCW, one `N_KICKS=1` kick-method RFCW, and one
  `N_KICKS=1,WAKES_AT_END=1` kick-method RFCW.
- Added `phase37_rfcw_multikick`, a focused guarded multi-kick RFCW regression
  with `N_KICKS=2,WAKES_AT_END=0` and `N_KICKS=3,WAKES_AT_END=1` cavities,
  longitudinal/transverse wake columns, smoothing/interpolation, deterministic
  `DX/DY` offsets, focusing, `CHANGE_P0=1`, and fixed-bin filtered LSC kicks.
- Added `phase38_rfcw_kick_rf_only`, a focused RF-only kick-method RFCW
  regression with `N_KICKS=2` and `N_KICKS=3` cavities, `LIGHT` and
  serial/local `TMEAN` fiducialization, deterministic `DX/DY` offsets,
  entrance/exit focusing, `CHANGE_P0=1`, and no active wake or LSC columns.
- Added `phase26_rfca_thin_change_p0`, a focused zero-length thin-`RFCA`
  regression with `CHANGE_P0=1` and a second no-`CHANGE_P0` thin cavity.
- Added `phase27_rfca_thin_fiducial_modes`, a focused zero-length thin-`RFCA`
  regression that exercises `FIDUCIAL="light"` and serial/local
  `FIDUCIAL="tmean"` phase setup, with `CHANGE_P0=1` on the `TMEAN` cavity.
- Added `phase28_rfca_thin_offset`, a focused zero-length thin-`RFCA`
  regression with deterministic `DX/DY` offsets, GPU phase setup, and
  `CHANGE_P0=1`.
- Added `phase29_rfca_matrix_rf_only`, a focused nonzero-length RF-only
  matrix-method `RFCA` regression with entrance/exit focusing, explicit and
  `LIGHT` fiducial phase setup, deterministic `DX/DY` offsets, and a
  `CHANGE_P0=1` cavity.
- Added `phase30_rfca_matrix_fiducial_modes`, a focused nonzero-length
  RF-only matrix-method `RFCA` regression that specifically exercises
  `FIDUCIAL="light"` and serial/local `FIDUCIAL="tmean"` phase setup with
  entrance/exit focusing, deterministic `DX/DY` offsets, and `CHANGE_P0=1`.
- Added `phase39_rfca_kick_rf_only`, a focused nonzero-length RF-only
  kick-method `RFCA` regression with `N_KICKS=2` and `N_KICKS=3` cavities,
  `LIGHT` and serial/local `TMEAN` fiducialization, deterministic `DX/DY`
  offsets, entrance/exit focusing, `CHANGE_P0=1`, and no wake or LSC
  arguments.
- Added `phase40_rf_pmaximum_fiducial`, a focused serial/local RF-only
  regression that exercises `FIDUCIAL="pmaximum"` across zero-length thin RFCA,
  nonzero-length matrix-method RFCA, kick-method RFCA, RF-only matrix-method
  RFCW, and RF-only kick-method RFCW, with deterministic offsets, focusing, and
  `CHANGE_P0` inside the guarded slices.
- Added `phase41_rfcw_wake_pmaximum_fiducial`, a focused wake-bearing RFCW
  regression that exercises `FIDUCIAL="pmaximum"` across the guarded
  matrix-method and multi-kick collective slices with longitudinal/transverse
  wakes, fixed-bin filtered LSC kicks, deterministic offsets, focusing,
  `CHANGE_P0`, and `WAKES_AT_END=0|1`.
- Added `phase42_rfcw_fixed_wake_bins`, a focused wake-bearing RFCW regression
  that exercises fixed longitudinal/transverse wake bins across the guarded
  matrix-method and kick-method collective slices with smoothing/interpolation,
  deterministic offsets, focusing, `CHANGE_P0`, and `WAKES_AT_END=0|1`.
- Added `phase43_rfcw_lsc_only`, a focused RFCW regression that exercises the
  guarded fixed-bin LSCKICK path without active longitudinal or transverse wake
  columns across the matrix-method and kick-method collective slices with
  deterministic offsets, focusing, `CHANGE_P0`, and `WAKES_AT_END=0|1`.
- Added `phase44_rfcw_single_wake_planes`, a focused RFCW regression that
  exercises longitudinal-only and transverse-only wake-family columns across the
  matrix-method and kick-method collective slices with smoothing/interpolation,
  deterministic offsets, focusing, `CHANGE_P0`, and `WAKES_AT_END=0|1`.
- Added `phase45_rf_kick_treference`, a focused RF-only kick-method RFCA/RFCW
  regression that exercises explicit nonzero `T_REFERENCE` with multi-kick
  section tracking, deterministic offsets, focusing, `CHANGE_P0`, and no-op
  RFCW `WAKES_AT_END=1`.
- Added `phase46_rfcw_wake_treference`, a focused wake-bearing RFCW regression
  that exercises explicit nonzero `T_REFERENCE` across the guarded matrix-method
  and multi-kick collective slices with longitudinal/transverse wake columns,
  fixed-bin filtered LSC kicks, smoothing/interpolation, deterministic offsets,
  focusing, `CHANGE_P0`, and `WAKES_AT_END=0|1`.
- Added `phase47_rf_selected_tmean_fiducial`, a focused two-bunch RF regression
  that exercises serial/local selected-bunch `TMEAN` fiducialization across
  zero-length thin, nonzero-length matrix-method, and kick-method RFCA plus
  RF-only matrix-method and kick-method RFCW.
- Added `phase48_rf_selected_pmaximum_fiducial`, a focused two-bunch RF
  regression that exercises serial/local selected-bunch `PMAXIMUM`
  fiducialization across zero-length thin, nonzero-length matrix-method, and
  kick-method RFCA plus RF-only matrix-method and kick-method RFCW.
- Added `phase49_rfcw_wake_selected_fiducial`, a focused guarded wake-bearing
  RFCW regression that exercises selected-bunch `TMEAN` and `PMAXIMUM`
  fiducialization across matrix-method and multi-kick collective slices with
  longitudinal/transverse wakes, fixed-bin filtered LSC kicks, deterministic
  offsets, focusing, `CHANGE_P0`, and `WAKES_AT_END=1`.
- Added `phase50_rf_first_fiducial`, a focused two-bunch RF regression that
  exercises selected-bunch `FIRST` fiducialization across zero-length thin,
  nonzero-length matrix-method, and kick-method RFCA plus RF-only matrix-method
  and kick-method RFCW and guarded wake-bearing matrix-method and multi-kick
  RFCW slices.
- Added `phase51_rf_standing_wave_single`, a focused RF regression that
  exercises `STANDING_WAVE=1` across zero-length thin RFCA, matrix-method RFCA,
  single-kick RFCA, RF-only matrix-method RFCW, RF-only single-kick RFCW, and
  guarded wake-bearing matrix-method/single-kick RFCW slices.
- Added `phase52_rf_standing_wave_multikick_treference`, a focused RF
  regression that exercises explicit-reference `STANDING_WAVE=1` multi-kick
  RFCA, RF-only RFCW, and guarded wake-bearing RFCW slices with `N_KICKS=2`
  and `N_KICKS=3`, deterministic offsets, focusing, `CHANGE_P0`, fixed-bin
  filtered LSC where applicable, and `WAKES_AT_END=1`.
- Added `phase53_rfca_standing_wave_multikick_fiducial`, a focused RFCA
  regression that exercises non-explicit fiducial setup for multi-kick
  `STANDING_WAVE=1` RFCA with `LIGHT`, selected-bunch `TMEAN`,
  selected-bunch `PMAXIMUM`, selected-bunch `FIRST`, deterministic offsets,
  focusing, and `CHANGE_P0`.
- Added `phase54_rfcw_standing_wave_multikick_fiducial`, a focused RFCW
  regression that exercises non-explicit fiducial setup for multi-kick
  `STANDING_WAVE=1` RFCW across RF-only and guarded wake-bearing kick-method
  sections with `LIGHT`, selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`,
  selected-bunch `FIRST`, deterministic offsets, focusing, `CHANGE_P0`,
  `WAKES_AT_END=0|1`, and fixed-bin filtered LSC kicks where applicable.
- Added the focused RF regressions to `run_benchmarks.sh`,
  `test/gpu_cuda/README.md`, `test/gpu_cuda/production_cases/README.md`, and
  `doc/CUDA_GPU_SUPPORT.md`.
- Added guarded selected-bucket and selected-range paths for serial/local
  bunched `WAKE` and `TRWAKE`. When a detected multi-bucket beam is narrowed by
  `START_BUNCH`/`END_BUNCH`, CUDA now loops over each selected effective bucket,
  filtering time reductions, binning, convolution, and kicks to that bucket
  while leaving other particles resident. This preserves CPU per-bucket wake
  semantics for single-bucket, multi-bucket range, and all-bucket filters.
  `WAKE,CHANGE_P0=1` is supported when at least one selected bucket is tracked.
- Added `GPU_BUNCHED_WAKE_MATCH_ONLY` for skipped `WAKE,CHANGE_P0=1` filters.
  CUDA now runs the existing central-momentum match helper without applying
  wake kicks when a bunched filter selects no buckets, while ordinary no-op
  skip filters still use the no-work fast path. Distributed Pelegant bunched
  wakes remain CPU-owned.
- Added `phase22_bunched_wake_filter_select`, a two-bucket regression that
  selects the later bucket for `WAKE` with `START_BUNCH=1,END_BUNCH=1` and the
  first bucket for `TRWAKE` with `END_BUNCH=0`.
- Added `phase23_bunched_wake_filter_range`, a three-bucket regression that
  selects multiple nonempty buckets for `WAKE` and `TRWAKE`, exercises
  all-bucket filtering, fixed-bin per-bucket `TRWAKE` centering, and
  multi-bucket `WAKE,CHANGE_P0=1`.
- Added `phase24_bunched_wake_change_p0_skip`, a two-bucket regression that
  selects no buckets for `WAKE,CHANGE_P0=1` while forcing nonzero mean `dp`,
  proving the CUDA match-only path updates central momentum without applying
  wake kicks.

Validation completed:

- `make -C src HAVE_CUDA=1 -j8` completed successfully.
- `make -C src HAVE_CUDA=1 GPU_VERIFY=1 -j8` completed successfully.
- Focused `phase21_rfcw_rf_only` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=48`, no `CPU element: RFCW` synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase21_rfcw_rf_only` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `1.550e-16` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase25_rfcw_rf_only_offset` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=48`, no `CPU element: RFCW` synchronization, and only final
  `gpuBaseDealloc`; this tiny focused case was overhead dominated at 0.43x
  same-workload speedup.
- Focused `phase25_rfcw_rf_only_offset` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `1.551e-16` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase31_rfcw_rf_only_fiducial_modes` quick validation passed
  CPU/GPU SDDS comparison at `1e-11` for all 4 common files. The normal CUDA
  run reported `helpers=48`, `reductions=124`, no `CPU element: RFCW`
  synchronization, and only final `gpuBaseDealloc`; this tiny focused case was
  overhead dominated at 0.52x same-workload speedup.
- Focused `phase31_rfcw_rf_only_fiducial_modes` `GPU_VERIFY` validation passed
  repeated `track_through_rfcw` CPU-shadow checks, with max absolute
  differences no larger than `1.550e-16` in the observed RFCW checks, and
  matched all 4 common SDDS files at `1e-11`.
- Focused `phase32_rfcw_matrix_wake` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=48`, `helpers=96`, `reductions=168`, `wakes=64`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`; this
  tiny focused case was overhead dominated at 0.36x same-workload speedup.
- Focused `phase32_rfcw_matrix_wake` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `1.550e-16` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase33_rfcw_kick_wake` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=48`, `helpers=112`, `reductions=168`, `wakes=64`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase33_rfcw_kick_wake` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `1.550e-16` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase34_rfcw_wakes_at_end` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=48`, `helpers=112`, `reductions=168`, `wakes=64`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase34_rfcw_wakes_at_end` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `1.550e-16` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase35_rfcw_matrix_wakes_at_end` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 4 common files. The normal CUDA run
  reported `matrix=48`, `helpers=96`, `reductions=168`, `wakes=64`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase35_rfcw_matrix_wakes_at_end` `GPU_VERIFY` validation passed
  repeated `track_through_rfcw` CPU-shadow checks, with max absolute
  differences no larger than `1.550e-16` in the observed RFCW checks, and
  matched all 4 common SDDS files at `1e-11`.
- Focused `phase36_rfcw_lsc` quick validation passed CPU/GPU SDDS comparison
  at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=30`, `helpers=138`, `reductions=198`, `wakes=72`, `lsc=36`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase36_rfcw_lsc` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `4.441e-15` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase37_rfcw_multikick` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=24`, `helpers=150`, `reductions=264`, `wakes=120`, `lsc=60`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`; this
  tiny focused case was overhead dominated at 0.35x same-workload speedup.
- Focused `phase37_rfcw_multikick` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `7.994e-15` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`. The VERIFY run reported `matrix=24`, `helpers=150`,
  `reductions=258`, `wakes=120`, `lsc=60`, no `CPU element: RFCW`
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase41_rfcw_wake_pmaximum_fiducial` quick validation passed
  CPU/GPU SDDS comparison at `1e-11` for all 4 common files. The normal CUDA
  run reported `matrix=30`, `helpers=192`, `reductions=330`, `wakes=144`,
  `lsc=72`, no `CPU element: RFCW` synchronization, no RFCW phase-reference
  setup synchronization, and only final `gpuBaseDealloc`.
- Focused `phase41_rfcw_wake_pmaximum_fiducial` `GPU_VERIFY` validation
  matched all 4 common SDDS files at `1e-11`. The VERIFY CUDA run reported
  `matrix=30`, `helpers=192`, `reductions=324`, `wakes=144`, `lsc=72`, no
  RFCW CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase42_rfcw_fixed_wake_bins` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=30`, `helpers=156`, `reductions=222`, `wakes=120`, `lsc=0`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase42_rfcw_fixed_wake_bins` `GPU_VERIFY` validation matched all 4
  common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=30`,
  `helpers=156`, `reductions=216`, `wakes=120`, `lsc=0`, no RFCW CPU-element
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase43_rfcw_lsc_only` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=30`, `helpers=156`, `reductions=144`, `wakes=0`, `lsc=72`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase43_rfcw_lsc_only` `GPU_VERIFY` validation matched all 4 common
  SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=30`,
  `helpers=156`, `reductions=138`, `wakes=0`, `lsc=72`, no RFCW CPU-element
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase44_rfcw_single_wake_planes` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 4 common files. The normal CUDA run
  reported `matrix=36`, `helpers=198`, `reductions=192`, `wakes=84`, `lsc=0`,
  no `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`.
- Focused `phase44_rfcw_single_wake_planes` `GPU_VERIFY` validation matched all
  4 common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=36`,
  `helpers=198`, `reductions=186`, `wakes=84`, `lsc=0`, no RFCW CPU-element
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase46_rfcw_wake_treference` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=30`, `helpers=192`, `reductions=324`, `wakes=144`, `lsc=72`, no
  RFCW CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase46_rfcw_wake_treference` `GPU_VERIFY` validation matched all 4
  common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=30`,
  `helpers=192`, `reductions=318`, `wakes=144`, `lsc=72`, no RFCW CPU-element
  or phase-reference setup synchronization, and only final `gpuBaseDealloc`.
- Focused `phase38_rfcw_kick_rf_only` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=64`, `helpers=304`, `reductions=124`, `wakes=0`, `lsc=0`, no
  `CPU element: RFCW` synchronization, and only final `gpuBaseDealloc`; this
  tiny focused case was overhead dominated at 0.64x same-workload speedup.
- Focused `phase38_rfcw_kick_rf_only` `GPU_VERIFY` validation passed repeated
  `track_through_rfcw` CPU-shadow checks, with max absolute differences no
  larger than `7.105e-15` in the observed RFCW checks, and matched all 4 common
  SDDS files at `1e-11`. The VERIFY run reported `matrix=64`, `helpers=304`,
  `reductions=116`, `wakes=0`, `lsc=0`, no `CPU element: RFCW`
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase45_rf_kick_treference` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `matrix=48`, `helpers=304`, `reductions=104`, `wakes=0`, `lsc=0`, no
  RFCA/RFCW CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase45_rf_kick_treference` `GPU_VERIFY` validation matched all 4
  common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=48`,
  `helpers=304`, `reductions=96`, `wakes=0`, `lsc=0`, no RFCA/RFCW CPU-element
  or phase-reference setup synchronization, and only final `gpuBaseDealloc`.
- Focused `phase47_rf_selected_tmean_fiducial` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 6 common files. The normal CUDA run
  reported `matrix=42`, `helpers=144`, `reductions=106`, `wakes=0`, `lsc=0`,
  no RFCA/RFCW CPU-element or phase-reference setup synchronization, and only
  final `gpuBaseDealloc`.
- Focused `phase47_rf_selected_tmean_fiducial` `GPU_VERIFY` validation matched
  all 6 common SDDS files at `1e-11`. The VERIFY CUDA run reported
  `matrix=42`, `helpers=144`, `reductions=100`, `wakes=0`, `lsc=0`, no
  RFCA/RFCW CPU-element or phase-reference setup synchronization, and only
  final `gpuBaseDealloc`.
- Focused `phase48_rf_selected_pmaximum_fiducial` quick validation passed
  CPU/GPU SDDS comparison at `1e-11` for all 6 common files. The normal CUDA run
  reported `matrix=42`, `helpers=144`, `reductions=106`, `wakes=0`, `lsc=0`, no
  RFCA/RFCW CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase48_rf_selected_pmaximum_fiducial` `GPU_VERIFY` validation
  matched all 6 common SDDS files at `1e-11`. The VERIFY CUDA run reported
  `matrix=42`, `helpers=144`, `reductions=100`, `wakes=0`, `lsc=0`, no
  RFCA/RFCW CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase49_rfcw_wake_selected_fiducial` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 6 common files. The normal CUDA run
  reported `matrix=30`, `helpers=174`, `reductions=474`, `wakes=432`,
  `lsc=72`, no RFCW CPU-element or phase-reference setup synchronization, and
  only final `gpuBaseDealloc`.
- Focused `phase49_rfcw_wake_selected_fiducial` `GPU_VERIFY` validation matched
  all 6 common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=30`,
  `helpers=174`, `reductions=468`, `wakes=432`, `lsc=72`, no RFCW CPU-element
  or phase-reference setup synchronization, and only final `gpuBaseDealloc`.
- Focused `phase50_rf_first_fiducial` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 6 common files. The normal CUDA run reported
  `matrix=54`, `helpers=240`, `reductions=344`, `wakes=216`, `lsc=36`, no
  RFCA/RFCW CPU-element or phase-reference setup synchronization, and only
  final `gpuBaseDealloc`.
- Focused `phase50_rf_first_fiducial` `GPU_VERIFY` validation matched all 6
  common SDDS files at `1e-11`. The VERIFY CUDA run reported `matrix=54`,
  `helpers=240`, `reductions=338`, `wakes=216`, `lsc=36`, no RFCA/RFCW
  CPU-element or phase-reference setup synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase51_rf_standing_wave_single` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 4 common files. The normal CUDA run
  reported `matrix=27`, `helpers=99`, `reductions=115`, `wakes=24`,
  `lsc=12`, no RFCA/RFCW CPU-element or phase-reference setup synchronization,
  and only final `gpuBaseDealloc`.
- Focused `phase51_rf_standing_wave_single` `GPU_VERIFY` validation matched all
  4 common SDDS files at `1e-11`. The VERIFY CUDA run reported no RFCA/RFCW
  CPU-element, phase-reference setup, or verification synchronization, and only
  final `gpuBaseDealloc`.
- Focused `phase52_rf_standing_wave_multikick_treference` quick validation
  passed CPU/GPU SDDS comparison at `1e-11` for all 4 common files. The normal
  CUDA run reported no RFCA/RFCW CPU-element fallback, no phase-reference setup
  fallback, and only final `gpuBaseDealloc`; this tiny focused case was
  overhead dominated at 0.24x same-workload speedup.
- Focused `phase52_rf_standing_wave_multikick_treference` `GPU_VERIFY`
  validation matched all 4 common SDDS files at `1e-11`. The VERIFY fallback
  summary reported no RFCA/RFCW CPU-element, phase-reference setup, or
  verification synchronization, and only final `gpuBaseDealloc`.
- Focused `phase53_rfca_standing_wave_multikick_fiducial` quick validation
  passed CPU/GPU SDDS comparison at `1e-11` for all 6 common files, including
  the two seed/prep outputs. The normal CUDA run reported no RFCA CPU-element
  fallback, no phase-reference setup fallback, and only final
  `gpuBaseDealloc`; this tiny focused case was overhead dominated at 0.70x
  same-workload speedup.
- Focused `phase53_rfca_standing_wave_multikick_fiducial` `GPU_VERIFY`
  validation matched all 6 common SDDS files at `1e-11`. The VERIFY fallback
  summary reported no RFCA CPU-element, phase-reference setup, or verification
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase54_rfcw_standing_wave_multikick_fiducial` quick validation
  passed CPU/GPU SDDS comparison at `1e-11` for all 6 common files, including
  the two seed/prep outputs. The normal CUDA run reported no RFCW CPU-element
  fallback, no phase-reference setup fallback, and only final
  `gpuBaseDealloc`; this tiny focused case was overhead dominated at 0.61x
  same-workload speedup.
- Focused `phase54_rfcw_standing_wave_multikick_fiducial` `GPU_VERIFY`
  validation matched all 6 common SDDS files at `1e-11`. The VERIFY fallback
  summary reported no RFCW CPU-element, phase-reference setup, or verification
  synchronization, and only final `gpuBaseDealloc`.
- A focused RFCW kick fiducial regression sweep after the phase-offset helper
  change matched all 16 common files at `1e-11` across
  `phase38_rfcw_kick_rf_only`, `phase49_rfcw_wake_selected_fiducial`, and
  `phase50_rf_first_fiducial`; the fallback summary reported no RFCW
  CPU-element or phase-reference setup fallback.
- The current RFCW regression sweep across `phase21_rfcw_rf_only`,
  `phase25_rfcw_rf_only_offset`, `phase31_rfcw_rf_only_fiducial_modes`,
  `phase32_rfcw_matrix_wake`, `phase33_rfcw_kick_wake`, and
  `phase34_rfcw_wakes_at_end`, and `phase35_rfcw_matrix_wakes_at_end` matched
  all 28 common files at `1e-11`; adding `phase36_rfcw_lsc` brings the focused
  covered RFCW outputs to 32 common files at `1e-11`, and adding
  `phase37_rfcw_multikick` brings the focused covered RFCW outputs to 36
  common files at `1e-11`; adding `phase38_rfcw_kick_rf_only` brings the
  focused covered RFCW outputs to 40 common files at `1e-11`; adding
  `phase41_rfcw_wake_pmaximum_fiducial` brings the focused covered RFCW
  outputs to 44 common files at `1e-11`; adding
  `phase42_rfcw_fixed_wake_bins` brings the focused covered RFCW outputs to 48
  common files at `1e-11`; adding `phase43_rfcw_lsc_only` brings the focused
  covered RFCW outputs to 52 common files at `1e-11`; adding
  `phase44_rfcw_single_wake_planes` brings the focused covered RFCW outputs to
  56 common files at `1e-11`; adding `phase46_rfcw_wake_treference` brings the
  focused covered RFCW outputs to 60 common files at `1e-11`; adding
  `phase48_rf_selected_pmaximum_fiducial` brings the RF-only selected-bunch
  RFCW coverage into the mixed RF gate, and adding
  `phase49_rfcw_wake_selected_fiducial` brings the focused covered wake-bearing
  RFCW outputs to 66 common files at `1e-11`. The fallback summaries showed no
  `RFCW` CPU-element fallback and only final `gpuBaseDealloc` synchronization
  per case.
- Bounded `lcls1` production-wrapper validation passed CPU/GPU SDDS comparison
  at `1e-11` for all 16 common files. It reduced the prior
  `CPU element: RFCW` synchronization count from 78 to zero for the
  matrix-method LCLS wake-bearing shape, reported `wakes=1186` with
  `cpuElement=0`, and ran 1.79x faster than the paired CPU quick run
  (`3.25s` GPU versus `5.83s` CPU).
- Bounded `lcls0` production-wrapper validation passed CPU/GPU SDDS comparison
  at `1e-11` for all 15 common files. It reduced the prior
  `CPU element: RFCW` synchronization count from 78 to zero for the
  `N_KICKS=1` LCLS wake-bearing shape, reported `wakes=1186` with
  `cpuElement=0`, and ran 1.41x faster than the paired CPU quick run
  (`5.83s` GPU versus `8.20s` CPU).
- Focused `phase26_rfca_thin_change_p0` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=96`, no `CPU element: RFCA` synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase26_rfca_thin_change_p0` `GPU_VERIFY` validation passed repeated
  `simple_rf_cavity` CPU-shadow checks, with max absolute differences no
  larger than `1.776e-15` in the observed RFCA checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase27_rfca_thin_fiducial_modes` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 4 common files. The normal CUDA run
  reported `helpers=72`, `reductions=182`, no `CPU element: RFCA`
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase27_rfca_thin_fiducial_modes` `GPU_VERIFY` validation passed
  repeated `simple_rf_cavity` CPU-shadow checks, with max absolute differences
  no larger than `1.776e-15` in the observed RFCA checks, and matched all 4
  common SDDS files at `1e-11`.
- Focused `phase28_rfca_thin_offset` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=72`, no `CPU element: RFCA` synchronization, and only final
  `gpuBaseDealloc`.
- Focused `phase28_rfca_thin_offset` `GPU_VERIFY` validation passed repeated
  `simple_rf_cavity` CPU-shadow checks, with max absolute differences no
  larger than `8.882e-16` in the observed RFCA checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase29_rfca_matrix_rf_only` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=48`, no `CPU element: RFCA` synchronization, and only final
  `gpuBaseDealloc`; this tiny focused case was overhead dominated at 0.52x
  same-workload speedup.
- Focused `phase29_rfca_matrix_rf_only` `GPU_VERIFY` validation passed repeated
  `simple_rf_cavity` CPU-shadow checks, with max absolute differences no larger
  than `1.551e-16` in the observed RFCA checks, and matched all 4 common SDDS
  files at `1e-11`.
- Focused `phase30_rfca_matrix_fiducial_modes` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 4 common files. The normal CUDA run
  reported `helpers=48`, `reductions=124`, no `CPU element: RFCA`
  synchronization, and only final `gpuBaseDealloc`; this tiny focused case was
  overhead dominated at 0.48x same-workload speedup.
- Focused `phase30_rfca_matrix_fiducial_modes` `GPU_VERIFY` validation passed
  repeated `simple_rf_cavity` CPU-shadow checks, with max absolute differences
  no larger than `1.550e-16` in the observed RFCA checks, and matched all 4
  common SDDS files at `1e-11`.
- Focused `phase39_rfca_kick_rf_only` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=304`, `reductions=124`, no `CPU element: RFCA` synchronization, and
  only final `gpuBaseDealloc`; this tiny focused case was overhead dominated
  at 0.50x same-workload speedup.
- Focused `phase39_rfca_kick_rf_only` `GPU_VERIFY` validation passed repeated
  `simple_rf_cavity` CPU-shadow checks, with max absolute differences no
  larger than `8.882e-15` in the observed RFCA checks, and matched all 4 common
  SDDS files at `1e-11`.
- Focused `phase40_rf_pmaximum_fiducial` quick validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 4 common files. The normal CUDA run reported
  `helpers=184`, `reductions=130`, no `CPU element: RFCA` or
  `CPU element: RFCW` synchronization, no RFCA/RFCW phase-reference setup
  synchronization, and only final `gpuBaseDealloc`.
- Focused `phase40_rf_pmaximum_fiducial` `GPU_VERIFY` validation passed
  CPU/GPU SDDS comparison at `1e-11` for all 4 common files. The VERIFY CUDA
  run reported `helpers=184`, `reductions=122`, no RFCA/RFCW CPU-element or
  phase-reference setup synchronization, and only final `gpuBaseDealloc`.
- The thin-RFCA regression sweep across `phase18_rfca_thin` and
  `phase26_rfca_thin_change_p0` matched all 8 common files at `1e-11`; the
  fallback summary showed no RFCA CPU-element fallback and only final
  `gpuBaseDealloc` synchronization per case.
- The expanded thin-RFCA regression sweep across `phase18_rfca_thin`,
  `phase26_rfca_thin_change_p0`, and
  `phase27_rfca_thin_fiducial_modes` matched all 12 common files at `1e-11`;
  the fallback summary showed no RFCA CPU-element fallback and only final
  `gpuBaseDealloc` synchronization per case.
- The current thin-RFCA regression sweep across `phase18_rfca_thin`,
  `phase26_rfca_thin_change_p0`, `phase27_rfca_thin_fiducial_modes`, and
  `phase28_rfca_thin_offset` matched all 16 common files at `1e-11`; the
  fallback summary showed no RFCA CPU-element fallback and only final
  `gpuBaseDealloc` synchronization per case.
- The final RFCA regression sweep across `phase18_rfca_thin`,
  `phase26_rfca_thin_change_p0`, `phase27_rfca_thin_fiducial_modes`,
  `phase28_rfca_thin_offset`, and `phase29_rfca_matrix_rf_only` matched all 20
  common files at `1e-11`; the fallback summary showed no RFCA CPU-element
  fallback and only final `gpuBaseDealloc` synchronization per case.
- The current RFCA regression sweep across `phase18_rfca_thin`,
  `phase26_rfca_thin_change_p0`, `phase27_rfca_thin_fiducial_modes`,
  `phase28_rfca_thin_offset`, `phase29_rfca_matrix_rf_only`, and
  `phase30_rfca_matrix_fiducial_modes` matched all 24 common files at `1e-11`;
  the fallback summary showed no RFCA CPU-element fallback and only final
  `gpuBaseDealloc` synchronization per case.
- Adding `phase39_rfca_kick_rf_only` brings focused RFCA coverage to 28 common
  files across the validated thin, matrix-method, and kick-method RF-only
  slices, with no RFCA CPU-element fallback in the new phase39 normal or
  `GPU_VERIFY` fallback summaries. The combined `phase40_rf_pmaximum_fiducial`
  case adds 4 common files covering serial/local RF-only `PMAXIMUM`
  fiducialization across RFCA and RFCW without RFCA/RFCW CPU-element or phase
  setup fallback. `phase45_rf_kick_treference` adds 4 mixed RFCA/RFCW
  kick-method common files covering explicit nonzero `T_REFERENCE` without
  RFCA/RFCW CPU-element or phase setup fallback.
  `phase47_rf_selected_tmean_fiducial` adds 6 mixed RFCA/RF-only RFCW common
  files covering selected-bunch `TMEAN` without RFCA/RFCW CPU-element or phase
  setup fallback, and `phase48_rf_selected_pmaximum_fiducial` adds 6 mixed
  RFCA/RF-only RFCW common files covering selected-bunch `PMAXIMUM` without
  RFCA/RFCW CPU-element or phase setup fallback. `phase49_rfcw_wake_selected_fiducial`
  adds 6 guarded wake-bearing RFCW common files covering selected-bunch
  `TMEAN`/`PMAXIMUM` without RFCW CPU-element or phase setup fallback.
  `phase50_rf_first_fiducial` adds 6 mixed RFCA/RF-only RFCW/guarded
  wake-bearing RFCW common files covering selected-bunch `FIRST` without
  RFCA/RFCW CPU-element or phase setup fallback. `phase51_rf_standing_wave_single`
  adds 4 mixed RFCA/RFCW common files covering the narrow matrix-method and
  single-kick `STANDING_WAVE=1` slice without RFCA/RFCW CPU-element or phase
  setup fallback. `phase52_rf_standing_wave_multikick_treference` adds 4
  mixed RFCA/RFCW common files covering explicit-reference multi-kick
  `STANDING_WAVE=1` RFCA/RFCW slices without RFCA/RFCW CPU-element or phase
  setup fallback. `phase53_rfca_standing_wave_multikick_fiducial` adds 6
  RFCA common files covering non-explicit fiducial multi-kick `STANDING_WAVE=1`
  RFCA without RFCA CPU-element or phase setup fallback.
  `phase54_rfcw_standing_wave_multikick_fiducial` adds 6 RFCW common files
  covering non-explicit fiducial multi-kick `STANDING_WAVE=1` RF-only and
  guarded wake-bearing RFCW without RFCW CPU-element or phase setup fallback.
- Targeted `clic1` required-mode production validation passed CPU/GPU SDDS
  comparison at `1e-11` for all 8 common files. It reduced the prior
  `CPU element: RFCW` synchronization count from 78 to zero, leaving only final
  `gpuBaseDealloc`, and ran 1.90x faster than the paired CPU quick run
  (`1.55s` GPU versus `2.94s` CPU).
- A broader `clic1` `GPU_VERIFY` run was attempted, but it failed before a
  useful RFCW gate on an existing `accumulate_beam_sums` reduction tolerance
  during the fiducial energy-profile setup. The focused `GPU_VERIFY` case is
  the verification gate for the RFCW particle update itself.
- Focused `phase22_bunched_wake_filter_select` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 6 common files. The normal CUDA run
  reported `wakes=40`, no CPU-element synchronization, and only final
  `gpuBaseDealloc`; this tiny filtered-bucket case was overhead dominated at
  0.56x same-workload speedup.
- Focused `phase22_bunched_wake_filter_select` `GPU_VERIFY` validation passed
  repeated `track_through_wake` and `track_through_trwake` CPU-shadow checks and
  matched all 6 common SDDS files at `1e-11`.
- Focused `phase23_bunched_wake_filter_range` quick validation passed CPU/GPU
  SDDS comparison at `1e-11` for all 6 common files. The normal CUDA run
  reported `wakes=96`, no CPU-element synchronization, and only final
  `gpuBaseDealloc`; this tiny range case was overhead dominated at 0.64x
  same-workload speedup.
- Focused `phase23_bunched_wake_filter_range` `GPU_VERIFY` validation passed
  repeated `track_through_wake` and `track_through_trwake` CPU-shadow checks and
  matched all 6 common SDDS files at `1e-11`.
- Focused `phase24_bunched_wake_change_p0_skip` quick validation passed
  CPU/GPU SDDS comparison at `1e-11` for all 6 common files. The normal CUDA
  run reported `helpers=1`, `wakes=0`, no CPU-element synchronization, and only
  final `gpuBaseDealloc`; this tiny match-only case was overhead dominated at
  0.57x same-workload speedup.
- Focused `phase24_bunched_wake_change_p0_skip` `GPU_VERIFY` validation passed
  repeated `track_through_wake` CPU-shadow checks with zero observed
  differences, and matched all 6 common SDDS files at `1e-11`.
- Existing bunched-wake quick regressions still matched CPU at `1e-11` for all
  14 common files across `phase16_bunched_wake_single`,
  `phase16_bunched_wake_filter_skip`, and
  `phase16_bunched_wake_multibucket_skip`.
- The expanded bunched-wake regression sweep across the three Phase 16 bunched
  cases plus `phase22_bunched_wake_filter_select` and
  `phase23_bunched_wake_filter_range` and
  `phase24_bunched_wake_change_p0_skip` matched all 32 common files at
  `1e-11`; the fallback summary showed no wake CPU-element fallback and only
  final `gpuBaseDealloc` synchronization per case.

Deferred follow-ups after action 6 closure:

- Keep the RF-only CLIC matrix-method slice, RF-only kick-method slice,
  wake-bearing matrix-method `lcls1` including positive-length matrix
  `WAKES_AT_END=1`, and guarded wake-bearing kick-method
  `N_KICKS>=1`/`WAKES_AT_END=0|1` slices, plus guarded fixed-bin LSCKICK and
  fixed wake bins plus single-wake-family cavities, LSC-only cavities, explicit
  nonzero `T_REFERENCE`, and serial/local full-beam or selected-bunch
  `TMEAN`/`PMAXIMUM`/`FIRST` inside those collective RFCW slices, plus explicit
  `T_REFERENCE` for the RF-only kick-method RFCA/RFCW section loops and
  selected-bunch `TMEAN`/`PMAXIMUM`/`FIRST` for supported RFCA and RF-only RFCW
  slices, explicit-reference multi-kick `STANDING_WAVE=1` RFCA/RFCW slices,
  and non-explicit fiducial multi-kick `STANDING_WAVE=1` RFCA/RFCW slices, as
  the current supported RF subsets for this finalization pass. Do not broaden
  them without a focused regression and production-shaped fallback delta.
- Defer distributed RFCW particles and remaining wake/RF/LSC combinations until
  their integration design is explicit and production evidence justifies the
  larger surface area.
- Defer distributed Pelegant bunched wakes until MPI-aware reductions and
  rank-local ownership semantics are designed.
- Defer broader RFCA/RFCW fiducialization beyond supported serial/local `LIGHT`,
  full-beam or selected-bunch `TMEAN`, and full-beam or selected-bunch
  `PMAXIMUM` modes, plus serial/local `FIRST`, in the zero-length thin `RFCA`,
  nonzero-length RF-only matrix-method and kick-method `RFCA`, RF-only `RFCW`,
  and wake-bearing matrix-method or guarded `N_KICKS>=1` `RFCW` paths only with
  dedicated regression cases. Distributed/MPI fiducial reductions and broader
  unsupported fiducial modes remain deferred until they have focused validation.
- Defer cuFFT-backed collective effects until profiling shows host
  FFT/convolution dominates over binning and transfer bridges.
- Action 6 should not block action 7. The next active implementation item is
  deterministic magnet coverage.

### 7. Finish Magnet Coverage

Status: wrapped for finalization. The release-sized deterministic magnet slice
is complete for simple `MULT`, and the detected-loss decision has an opt-in
prototype with accepted-array coverage for no-loss-output cases plus validated
`.los` and global-loss-coordinate fallback guards. Representative advanced
`CSBEND` fallbacks are covered by a focused regression. The remaining items
below are deferred follow-ups and should not block action 8.

Code change made for this slice:

- Added automatic CUDA eligibility for deterministic `MULT` elements with
  `ORDER=0..3`, `N_SLICES>0`, no synchrotron radiation, no spin tracking, and
  finite `L`/strength/offset parameters.
- Reused the checked CUDA multipole kernel for normal `KNL` and
  `BORE`/`BTIPL` strength forms, including `DX`/`DY`/`DZ`, `TILT`, `FACTOR`,
  and `EXPAND_HAMILTONIAN`.
- Kept high-order `MULT`, `FMULT`, radiation, spin, aperture hooks, and
  file/table-backed multipole data on the CPU path.
- Marked `MULT` as `GPUCapable=1` and updated the dictionary metadata check.
- Updated the production magnet profiler so matrix `KICKER`/`HVCOR` aliases are
  no longer counted as unsupported, and simple `MULT` is counted as covered.
- Added `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1` as a guarded stable
  prefix-sum compaction prototype for detected losses from supported multipole
  and non-CSR `CSBEND` kernels when no loss-output file or global
  loss-coordinate bookkeeping is active. Detected losses still fall back to CPU
  by default.
- Reused the stable accepted-array partition path for magnet loss compaction,
  including the device-side accepted scatter controlled by
  `ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE`.

Validation completed:

- Built CPU, CUDA, and `GPU_VERIFY` CUDA binaries.
- Added `phase55_mult_deterministic`, covering `MULT` orders 0 through 3,
  original-mode offsets/tilts, `EXPAND_HAMILTONIAN`, and `BORE`/`BTIPL`.
- Normal CUDA quick run matched CPU output for all 4 common SDDS files at
  `1e-11`; CUDA reported 100 magnet kernels and no CPU-element fallback, only
  final `gpuBaseDealloc` synchronization. Report:
  `test/gpu_cuda/output/reports/phase55-mult-deterministic-quick.md`.
- `GPU_VERIFY` quick run passed repeated `multipole_tracking` CPU-shadow
  checks and matched all 4 common SDDS files at `1e-11`.
- `&print_dictionary` plus `check_dictionary_gpu_support.py` passed with
  `MULT GPUCapable=1`.
- Refreshed `phase17-production-magnet-profile.md`: current simple CUDA
  candidates increased from 127,750 to 131,004, deferred/blocking definitions
  decreased from 20,799 to 17,545, and
  `unsupported_multipole_or_corrector_family` decreased from 3,278 to 24
  remaining `FMULT` definitions.
- Added `phase56_mult_loss_compaction`, a no-loss-output simple `MULT`
  detected-loss case. With `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, CUDA
  matched CPU output for all 4 common SDDS files at `1e-11`, reported no
  CPU-element fallback, and synchronized only at final `gpuBaseDealloc`. Report:
  `test/gpu_cuda/output/reports/phase56-mult-loss-compaction-quick.md`.
- Added `phase57_mult_loss_accepted_compaction`, the same simple `MULT`
  detected-loss shape with acceptance output enabled. With
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, CUDA matched CPU output for all
  5 common SDDS files at `1e-11`, including `.acc`, reported no CPU-element
  fallback, enabled accepted-device compaction, and synchronized only at final
  `gpuBaseDealloc`. Report:
  `test/gpu_cuda/output/reports/phase57-mult-loss-accepted-compaction-quick.md`.
- Added `phase59_mult_loss_output_fallback`, the same simple `MULT`
  detected-loss shape with `.los` and `.acc` output enabled. With
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, CUDA matched CPU output for all
  6 common SDDS files at `1e-11`, including `.los`, lost 2862 of 3000 particles
  (`Transmission=0.046`), and explicitly synchronized through
  `multipole_tracking particle loss fallback`. This validates that the opt-in
  compaction flag still preserves existing loss-output semantics by staying on
  the CPU loss-row path when `.los` is requested. Report:
  `test/gpu_cuda/output/reports/phase59-mult-loss-output-fallback-quick.md`.
- Added `phase60_mult_global_loss_fallback`, the same simple `MULT`
  detected-loss shape with `.los`, `losses_include_global_coordinates=1`, and
  `.acc` output enabled. With `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`,
  CUDA matched CPU output for all 6 common SDDS files at `1e-11`, including the
  `.los` `X`, `Z`, and `thetaX` global-coordinate columns, lost 2862 of 3000
  particles (`Transmission=0.046`), and explicitly synchronized through
  `multipole_tracking particle loss fallback`. This validates the current guard
  for global loss-coordinate consumers while resident global loss-row
  compaction remains deferred. Report:
  `test/gpu_cuda/output/reports/phase60-mult-global-loss-fallback-quick.md`.
- Added `phase58_csbend_loss_compaction`, a no-loss-output non-CSR `CSBEND`
  detected-loss case with first-order edges, original-mode misalignments, and
  acceptance output enabled. The quick gate lost 47 of 3000 particles
  (`Transmission=0.9843333`); with
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, CUDA matched CPU output for all
  5 common SDDS files at `1e-11`, including `.acc`, reported no CPU-element
  fallback, enabled accepted-device compaction, and synchronized only at final
  `gpuBaseDealloc`. Report:
  `test/gpu_cuda/output/reports/phase58-csbend-loss-compaction-quick.md`.
- Added `phase61_csbend_advanced_fallback`, an advanced non-CSR `CSBEND` case
  that exercises nonzero `MALIGN_METHOD` with `EPITCH`/`EYAW`, plus Hwang,
  Hwang/Lindberg, and curved fringe settings. CUDA matched CPU output for all 5
  common SDDS files at `1e-11`, reported `magnets=0`, and recorded 24
  `track_through_csbend unsupported option` synchronization requests. This
  validates the conservative fallback guard for representative advanced
  `CSBEND` shapes while resident support remains deferred. Report:
  `test/gpu_cuda/output/reports/phase61-csbend-advanced-fallback-quick.md`.

Deferred follow-ups after action 7 closure:

- Deferred: implement advanced `CSBEND` features only after a separate design
  and resident validation plan. Current fallback coverage includes nonzero
  `MALIGN_METHOD` with `EPITCH`/`EYAW` and Hwang/Lindberg/curved fringe models;
  reference/FSE correction, aperture hooks, and slice-by-slice tracking still
  need explicit fallback or implementation regressions before they are promoted.
- Deferred: broaden detected-loss compaction only after a resident loss-output
  row design is implemented for `.los` files and global loss-coordinate rows,
  plus separate validation for broader production `CSBEND` loss cases beyond
  the no-loss-output phase58 gate.
- Deferred: keep radiation/ISR, spin, and corrector radiation kicks blocked until
  stochastic/distribution validation is available.
- Deferred: leave `FMULT` and file/table-backed multipoles deferred unless
  production profiling justifies a separate data-loading design.

### 8. Finish SCMULT, Field Maps, Wigglers, Poisson, And Ion Scope

Status: complete for the current action-8 finalization scope. All remaining
items are deferred release-policy or future-port decisions. The action-8
validation refresh is complete for the production-shaped linear `SCMULT`
no-WATCH wrapper, the focused nonlinear, sliced, and multi-bunch `SCMULT`
fallback guards, the first resident
`UKICKMAP` map-loss compaction gate, the high-count production-shaped
`latticeErrors6` `UKICKMAP` compaction gate, the ordinary focused
`GKICKMAP` compaction fixture, the `KICKMAP`/`UKICKMAP` loss-row fallback
guards, and the refreshed field-map/wiggler fallback wrappers. The action-8
ion/Poisson fallback wrapper is also in place. `SCMULT`
remains opt-in pending a release decision or a second independent source-family
validation, and map-loss compaction remains opt-in under
`ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`.

Validation completed:

- Reran `scRing2_no_watch`, the production-shaped linear, unsliced `SCMULT`
  wrapper with WATCH diagnostics disabled, using
  `ELEGANT_GPU_ENABLE_SCMULT=1`.
- CPU/GPU quick runs matched all 5 common SDDS files at `1e-11`, including
  `.twi`. The fixed source beam tracked 1000 particles for 8 passes.
- CUDA reported 8 resident `SCMULT` kernels, 480 matrix kernels, 288 magnet
  kernels, 16 aperture kernels, no CPU-element fallback, and only final
  `gpuBaseDealloc` synchronization. Same-workload speedup was 1.57x on the
  quick gate. Report:
  `test/gpu_cuda/output/reports/action8-scmult-no-watch-quick.md`.
- Added `phase65_scmult_nonlinear_fallback` and
  `phase66_scmult_sliced_fallback` as focused deferred-mode guards under
  `ELEGANT_GPU_ENABLE_SCMULT=1`. CPU/GPU quick runs with 2,000 particles and
  2 passes matched all 10 common SDDS files at `1e-11`, including `.twi`.
  CUDA reported 48 `trackThroughSCMULT fallback` synchronizations per case,
  72 matrix kernels per case, and 0 resident `SCMULT` kernels, confirming
  nonlinear and sliced single-bunch kicks remain CPU-owned. Reports:
  `test/gpu_cuda/output/reports/action8-scmult-deferred-fallback.md` and
  `test/gpu_cuda/output/reports/action8-scmult-deferred-fallbacks.md`.
- Added `phase67_scmult_multibunch_fallback` as the matching multi-bunch
  deferred-mode guard under `ELEGANT_GPU_ENABLE_SCMULT=1`. The case generates
  a seed bunch, reloads it with `use_bunched_mode=1`, and duplicates it with a
  time stagger so SCMULT sees multiple buckets. CPU/GPU quick runs with 1,000
  seed particles per bunch and 2 passes matched all 7 common SDDS files at
  `1e-11`, including the seed `.out`/`.fin` files and tracked `.twi`. CUDA
  reported 2 `initializeSCMULT fallback` synchronizations, 48
  `accumulateSCMULT fallback` synchronizations, 72 matrix kernels, and 0
  resident `SCMULT` kernels,
  confirming multi-bunch SCMULT remains CPU-owned. Reports:
  `test/gpu_cuda/output/reports/action8-scmult-multibunch-fallback.md` and
  `test/gpu_cuda/output/reports/action8-scmult-multibunch-fallbacks.md`.
- Added opt-in stable resident map-loss compaction for the supported
  deterministic `KICKMAP`/`UKICKMAP` CUDA subset when no loss-output or global
  loss-coordinate rows are needed. The default checked path and CPU fallback
  remain unchanged when `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1` is not
  set, and unsupported radiation/ISR, offset, tilt, and yaw modes are still
  CPU-owned.
- Reran `uKickMap1` with 3,000 particles and 2,000 passes. The CPU run took
  5.54s. The opt-in resident compaction GPU run took 1.82s, matched all 4
  common SDDS files at `1e-11`, and reduced synchronization to only final
  `gpuBaseDealloc`. The same GPU workload without compaction matched CPU but
  had 101 `UKICKMAP particle loss fallback` synchronizations. Report:
  `test/gpu_cuda/output/reports/action8-ukickmap-map-loss-compaction.md`.
- Built `GPU_VERIFY=1` and reran the `uKickMap1` quick gate. It matched all 4
  common SDDS files at `1e-11`, with exact `trackUndulatorKickMap` CPU-shadow
  checks in the no-loss quick slice.
- Added the focused `latticeErrors6` high-count `UKICKMAP` wrapper with 40
  production septum maps per pass and no `.los` output. The 30,000-particle,
  2-pass CPU/GPU compaction gate matched all 4 common SDDS files at `1e-11`,
  kept 564 survivors, ran in 0.41s on `gpu-elegant` versus 0.77s on CPU, and
  synchronized only at final `gpuBaseDealloc`. The same GPU workload without
  compaction still matched CPU but requested 80 `UKICKMAP particle loss
  fallback` synchronizations and ran in 0.92s. Reports:
  `test/gpu_cuda/output/reports/action8-latticeErrors6-map-loss-compaction.md`
  and
  `test/gpu_cuda/output/reports/action8-latticeErrors6-map-loss-fallbacks.md`.
- Searched the local production test set for existing ion/Poisson candidates;
  no `IONEFFECTS` or `BEAMBEAM` deck was present, so `ionEffectsPoisson` is a
  synthetic production-like wrapper rather than a copied source deck.
- Added `phase62_kickmap_loss_compaction` as the ordinary non-undulator
  `GKICKMAP` fixture. CPU/GPU quick runs with 3,000 particles and 3 passes
  matched all 4 common SDDS files at `1e-11`. With compaction enabled, CUDA
  reported 30 magnet kernels for 15 `GKICKMAP` tracks plus 15 stable compaction
  passes and only final `gpuBaseDealloc` synchronization. The same GPU workload
  without compaction matched CPU but requested 15 `KICKMAP particle loss
  fallback` synchronizations. Reports:
  `test/gpu_cuda/output/reports/action8-kickmap-map-loss-compaction.md` and
  `test/gpu_cuda/output/reports/action8-kickmap-map-loss-fallbacks.md`.
- Added `phase63_kickmap_loss_output_fallback` and
  `phase64_kickmap_global_loss_fallback` as ordinary `GKICKMAP` loss-row
  fallback guards under `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`.
  CPU/GPU quick runs with 3,000 particles and 3 passes matched all 12 common
  SDDS files at `1e-11`, including `.los` and `.acc`; the global-loss case
  includes the `X`, `Z`, and `thetaX` columns. CUDA reported 15 explicit
  `KICKMAP particle loss fallback` synchronizations per case, confirming the
  opt-in resident compaction path still defers host loss-row semantics. Reports:
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallbacks.md`.
- Added `latticeErrors6_loss_output` and `latticeErrors6_global_loss` as
  high-count production-shaped `UKICKMAP` loss-row fallback guards under
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`. CPU/GPU quick runs with
  3,000 particles and 2 passes matched all 12 common SDDS files at `1e-11`,
  including `.los` and `.acc`; the global-loss case includes `X`, `Z`, and
  `thetaX`. CUDA reported 73 explicit `UKICKMAP particle loss fallback`
  synchronizations per case, confirming the opt-in resident compaction path
  also defers UKICKMAP host loss-row semantics. Reports:
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallbacks.md`.
- Added `ionEffectsPoisson`, a small production-like `IONEFFECTS` wrapper with
  a `CHARGE` element, pressure-profile SDDS data, ion-property SDDS data,
  positive ion spans, and a 16x16 Poisson grid. CPU/GPU auto-mode quick runs
  with 2,000 particles and 3 passes matched all 4 common SDDS files at
  `1e-11`. CUDA reported the expected CPU fallback for `IONEFFECTS` twice,
  one short-GPU-island skip, and final `gpuBaseDealloc` synchronization.
  Reports:
  `test/gpu_cuda/output/reports/action8-ion-effects-poisson-fallback.md` and
  `test/gpu_cuda/output/reports/action8-ion-effects-poisson-fallbacks.md`.
- Refreshed the bounded field-map/wiggler fallback wrappers `bmapxy1`,
  `bmxyz1`, `boffaxe1`, and `cwiggler10`. CPU/GPU required-mode quick runs
  matched all 19 common SDDS files at `1e-11`. CUDA reported expected
  CPU-owned behavior: `BMXYZ` and `CWIGGLER` CPU-element synchronizations,
  10 `WATCH parameter output` read-only synchronizations from `cwiggler10`,
  and no resident field-map/wiggler kernels. Reports:
  `test/gpu_cuda/output/reports/action8-fieldmap-wiggler-fallback-refresh.md`
  and
  `test/gpu_cuda/output/reports/action8-fieldmap-wiggler-fallbacks.md`.

Deferred action-8 tasks:

- Deferred: keep linear `SCMULT` behind `ELEGANT_GPU_ENABLE_SCMULT=1` unless
  release policy explicitly accepts the current `scRing2`/`scRing2_no_watch`
  evidence as sufficient for a narrow automatic eligibility window.
- Deferred: keep nonlinear, sliced, and bunched SCMULT deferred until profiling
  shows they matter; `phase65`/`phase66`/`phase67` now guard the nonlinear,
  sliced single-bunch, and multi-bunch fallback paths.
- Deferred: keep `KICKMAP`/`UKICKMAP` map-loss compaction opt-in pending
  release policy, now that `uKickMap1`, the high-count `latticeErrors6`
  `UKICKMAP` gate, and the ordinary `phase62_kickmap_loss_compaction`
  `GKICKMAP` gate pass.
- Deferred: keep map loss with `.los` output or global loss-coordinate rows on
  CPU; the `phase63`/`phase64` `KICKMAP` guards and
  `latticeErrors6_loss_output`/`latticeErrors6_global_loss` `UKICKMAP` guards
  validate this fallback under the compaction flag until resident loss-row
  semantics are designed and validated.
- Deferred: keep `BMAPXY`, `BMXYZ`, `BOFFAXE`, `CWIGGLER`, and unwrapped
  `WIGGLER` CPU-owned unless a separate measured port is justified.
- Deferred: keep `IONEFFECTS` and `poisson.cc` CPU-owned; use
  `ionEffectsPoisson` as the future correctness gate before considering any
  cuFFT-backed port.

### 9. Finish Pelegant And Multi-GPU Scope

Status: skipped/deferred for the current pass. The user explicitly chose to
skip action 9 before starting action 10. Keep these as future hardware-scope
items rather than blockers for the stochastic-validation work.

- On multi-GPU hardware, validate one GPU per worker rank and `CUDA_VISIBLE_DEVICES` mapping.
- Test multiple worker ranks sharing fewer GPUs and compare against CPU Pelegant and serial `gpu-elegant`.
- Design GPU-resident dynamic redistribution for `load_balancing_on=1`; keep the current guard until it passes correctness and timing gates.
- Prototype MPI-aware GPU reductions only after collective CPU fallback hotspots are quantified.
- Prototype GPU-aware MPI particle exchange only with an MPI stack known to support it.

### 10. Add Stochastic Validation Before Stochastic CUDA Paths

Status: initial harness, two-seed guards, and the first broadened field-map
guard are complete; broader seed coverage remains for the other stochastic
families. Action 10 now has a standalone
distribution-comparison tool, but no stochastic CUDA path is enabled. Action 9
is intentionally skipped for now per the current finalization order.

Validation/tooling completed:

- Added `test/gpu_cuda/compare_stochastic_sdds.py` as a separate comparator
  from deterministic `compare_sdds.py`. It accepts one positional
  reference/candidate output pair plus repeated `--pair REF=CAND` fixed-seed
  pairs, aggregates common SDDS outputs, and compares row counts, means,
  sigmas, and optional KS-style/histogram distances.
- The default `auto` column set covers common particle coordinates (`x`, `xp`,
  `y`, `yp`, `t`, `p`), centroid columns, sigma columns, emittance-like
  columns from `.sig`, spin particle/centroid/sigma columns where present,
  energy-spread and bunch-length columns where present, and row counts for
  loss/acceptance-style outputs. Explicit `--columns` and
  `--histogram-columns` keep stochastic feature gates tunable without changing
  deterministic comparisons. The comparator now also fails finite numeric
  sample-count mismatches, so `NaN`/`Inf` output is not silently dropped.
- Smoke-tested the tool on the existing CPU/GPU
  `phase67_scmult_multibunch_fallback` output. The distribution comparator
  passed 39 checks and wrote
  `test/gpu_cuda/output/reports/action10-stochastic-comparator-smoke.md` plus
  the matching TSV.
- Added the first representative radiation/ISR distribution gate using the
  existing production `csbend1` wrapper, whose source lattice has
  `CSBEND,ISR=1,SYNCH_RAD=1`. Two fixed-seed CPU/GPU quick pairs
  (`987654321` and `987654322`) passed 30 aggregate distribution checks and
  wrote `test/gpu_cuda/output/reports/action10-csbend1-stochastic-distribution.md`
  plus the matching TSV.
- The current `csbend1` CUDA run intentionally remains a fallback guard, not a
  stochastic CUDA port: GPU stderr reported `elements=0` and only the final
  `gpuBaseDealloc` synchronization. This preserves today's CPU-owned
  radiation/ISR behavior while establishing the distribution gate required
  before any future CUDA implementation is enabled.
- Added `spinTest2` as the first spin-tracking distribution guard. Two
  fixed-seed CPU/GPU quick pairs passed 43 aggregate checks, including `spx`,
  `spy`, `spz`, `Cspx`, `Cspy`, `Cspz`, and spin sigma-matrix terms. The GPU
  run remains a mixed/fallback guard with repeated spin-sensitive CPU
  synchronizations, not a spin CUDA port.
- Added `cwiggler10_radiation` as the first pure stochastic wiggler guard.
  Two fixed-seed CPU/GPU quick pairs passed 30 aggregate checks with
  `CWIGGLER,SYNCH_RAD=1,ISR=1`; the GPU run explicitly synchronized for the
  radiating `CWIGGLER` section and kept that tracking CPU-owned.
- Added `uKickMap4_radiation` as a bounded field-map radiation guard based on
  the source `uKickMap4` `UKICKMAP` ring. The initial two-seed CPU/GPU
  distribution report exposed non-finite `Ss` and `St` values in `.sig`.
  The failure was traced to CUDA beam-sum covariance cancellation in nearly
  zero-spread longitudinal coordinates: the device reduction used
  `sum(x*x)-sum(x)^2/n`, which can produce a small negative diagonal variance
  where the CPU centered-deviation path stays nonnegative.
- Fixed the longitudinal sigma blocker by clamping negative diagonal covariance
  terms in the CUDA beam-sum reduction and by using the existing safe-square-root
  behavior for WATCH-parameter and `.sig` standard-deviation output. A rebuilt
  CUDA binary passed the refreshed five-seed `uKickMap4_radiation` distribution
  report with all 31 checks passing:
  `test/gpu_cuda/output/reports/action10-uKickMap4-radiation-distribution-5seed.md`.
  Direct `Ss`/`St` scans of all five fixed-seed GPU `.sig` files showed zero
  `NaN`/`Inf` samples.
- Updated `test/gpu_cuda/README.md` with the fixed-seed `--pair` workflow and
  the rule that deterministic CUDA gates stay on `compare_sdds.py`, while
  stochastic radiation/ISR/spin/wiggler/field-map gates use distribution
  comparisons.

Remaining action-10 work:

- Broaden the remaining stochastic feature families beyond the initial two
  fixed seeds and tune tolerances from observed CPU/GPU distribution scatter
  rather than from quick two-seed smoke gates.
- Keep per-particle equality tests for deterministic modes and distribution
  tests for stochastic modes clearly separated in reports and release notes.

## Suggested Near-Term Order

1. Run the GitHub Actions workflow and self-hosted GPU runtime smoke, then archive artifacts.
2. Treat action 6 as closed for the current serial/local scope. Use `test/gpu_cuda/output/reports/finalize-action3-sync-hotspots.md` plus the action-6 CLIC, `lcls1`, `lcls0`, `phase36_rfcw_lsc`, `phase37_rfcw_multikick`, `phase38_rfcw_kick_rf_only`, `phase39_rfca_kick_rf_only`, `phase40_rf_pmaximum_fiducial`, `phase41_rfcw_wake_pmaximum_fiducial`, `phase42_rfcw_fixed_wake_bins`, `phase43_rfcw_lsc_only`, `phase44_rfcw_single_wake_planes`, `phase45_rf_kick_treference`, `phase46_rfcw_wake_treference`, `phase47_rf_selected_tmean_fiducial`, `phase48_rf_selected_pmaximum_fiducial`, `phase49_rfcw_wake_selected_fiducial`, `phase50_rf_first_fiducial`, `phase51_rf_standing_wave_single`, `phase52_rf_standing_wave_multikick_treference`, `phase53_rfca_standing_wave_multikick_fiducial`, and `phase54_rfcw_standing_wave_multikick_fiducial` deltas as the RF synchronization baseline. The CLIC RF-only plus RF-only kick-method, LCLS matrix-method, guarded `N_KICKS>=1` wake-bearing `RFCW`, and RF-only RFCA matrix/kick subsets, including the focused matrix-method and kick-method `WAKES_AT_END=1`, LSCKICK, multi-kick, fixed-wake-bin, LSC-only, single-wake-family, explicit `T_REFERENCE`, selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`, selected-bunch `FIRST`, narrow matrix/single-kick plus explicit-reference or non-explicit fiducial multi-kick `STANDING_WAVE=1` RFCA/RFCW gates, are now covered.
3. Apply the opt-in policy above: recommend targeted opt-in use for resident CSR and stable aperture compaction, keep linear SCMULT opt-in pending a release decision or second source-family validation, and keep exact drift plus CSR histogram/kick experimental.
4. Continue action 8 only if release policy wants resident map-loss `.los`/global-coordinate rows or a specific field-map/wiggler CUDA port. The high-count `latticeErrors6` `UKICKMAP`, ordinary `phase62_kickmap_loss_compaction` `GKICKMAP`, `KICKMAP` and `UKICKMAP` loss-row fallback guards, `ionEffectsPoisson` CPU-fallback, and refreshed `BMAPXY`/`BMXYZ`/`BOFFAXE`/`CWIGGLER` fallback gates now pass, and action 7 is wrapped for finalization; keep unsupported `RFCW`, distributed RF/MPI reductions, cuFFT, remaining CSR handoffs, and aperture overhead as deferred post-release candidates unless new production evidence changes the priority.
5. Continue action 10 by broadening each passing stochastic gate beyond the
   current two-seed smoke size before expanding any corresponding CUDA path.
