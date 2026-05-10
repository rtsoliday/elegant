# CUDA GPU Improvement Plan

This plan builds on `GPU_CUDA_ImplementationPlan.md` after the first CUDA implementation pass.  The initial work established optional CUDA builds, `gpu-elegant`, `gpu-Pelegant`, verification tooling, benchmark cases, conservative CUDA kernels, CPU fallback behavior, and user documentation.  The next work should focus on making the implementation more useful in production: fewer host/device synchronizations, broader coverage of common expensive physics, better release automation, and clearer guarantees for ordinary CPU serial and MPI builds.

## Current State

Implemented and verified CUDA areas:

- Optional CUDA build support for serial `gpu-elegant`.
- Optional CUDA build support for MPI `gpu-Pelegant`.
- Runtime controls for `off`, `auto`, `required`, thresholds, verbose diagnostics, and verification tolerances.
- Matrix-family tracking and helper reductions.
- Loose aperture predicates, with exact lossy bookkeeping still CPU-owned by default.
- Deterministic `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, and non-CSR `CSBEND` slices, including simple original-mode `KQUAD`/`KSEXT`/`KOCT`/`DQCOR`/`CSBEND` misalignment coverage.
- Conservative `WAKE`, `TRWAKE`, and `LSCDRIFT` slices.
- Conservative `CSRCSBEND` wake-potential slice.
- Guarded default-on serial linear, unsliced, single-bucket `SCMULT`, with `ELEGANT_GPU_ENABLE_SCMULT=0` as the fallback override.
- Benchmark and comparison tooling under `test/gpu_cuda/`.
- CUDA support documentation under `doc/CUDA_GPU_SUPPORT.md`.

Important limits still in place:

- CUDA runtime is conservative and still falls back to CPU for many advanced modes.
- The dormant `HAVE_GPU` and `GPU_SUPPORT` hooks are legacy integration points, not proof that a path is safe.  They may still need substantial modification before being used for additional features.
- Some CPU-visible metadata can change when `GPU_SUPPORT` flags are updated.  For example, the KOCT GPU-capability flag changes dictionary output even in ordinary serial builds.  Treat CPU-output preservation claims carefully and test dictionary/metadata paths separately from tracking output.
- Timing gates must continue to aim CPU baselines at about one minute, with timeout guards, so routine validation does not accidentally expand into hour-scale jobs.

## Improvement Priorities

| Priority | Area | Why It Matters | Main Risk |
| --- | --- | --- | --- |
| 1 | Release hardening and CPU invariance | Prevent CUDA changes from affecting ordinary `elegant` and `Pelegant` users | Metadata flags and build auto-detection can change visible behavior |
| 2 | Production test-set expansion | Synthetic cases are necessary but not enough for release confidence | Real tests can be large, stochastic, or hard to parameterize |
| 3 | Benchmark/report automation | Make speedups reproducible and reviewable | Timing noise and incomplete hardware metadata |
| 4 | Host/device residency | Most remaining slowdowns come from synchronization and transfers | More resident state can make fallback and diagnostics harder |
| 5 | CSR improvements | CSR remains only modestly faster and is important in production cases | Numerical sensitivity and stateful wake bookkeeping |
| 6 | Aperture/loss compaction | Loss-heavy cases currently fall back or use slower opt-in exact compaction | Preserving loss order, accepted arrays, and `particleID` semantics |
| 7 | Collective effects coverage | Wakes/LSC show useful speedups but many common modes are still CPU-only | Binning, filtering, smoothing, and multi-bunch semantics |
| 8 | Broader magnet support | Current deterministic magnets are fast, but many production features are excluded | Radiation, spin, fringe models, misalignments, and aperture hooks |
| 9 | SCMULT/Poisson and production profiling | Linear SCMULT is promising but narrow | Nonlinear/sliced/multi-bunch physics and FFT reproducibility |
| 10 | Pelegant/multi-GPU | Needed for cluster-scale benefit | Requires hardware with more than one GPU and MPI-aware validation |

## Standing Gates For Every Improvement

Every improvement must satisfy these gates before it is enabled by default:

1. CPU-only serial and MPI builds still succeed.
2. CPU tracking output is unchanged for representative serial and MPI cases.
3. CPU-visible metadata changes are intentional and documented.
4. CUDA build and CUDA verify build succeed.
5. Unsupported feature combinations fall back to CPU and preserve output.
6. `GPU_VERIFY` passes for any new kernel or intermediate array.
7. End-to-end SDDS comparisons pass with documented tolerances.
8. Timing shows benefit on a representative workload.
9. Timing runs use `--baseline --target-seconds 60` or another explicit bounded target, with timeout guards.
10. New automatic paths have thresholds that avoid small-case slowdowns.

## Phase 10: Release Hardening And CPU Invariance

Goal: make it safe to commit and release the current CUDA work without surprising serial or MPI users.

Tasks:

1. Add a CPU invariance checklist to the release workflow:
   - serial `elegant` output comparison against a clean baseline
   - MPI `Pelegant` output comparison against a clean baseline
   - `&print_dictionary` SDDS and LaTeX output comparison
   - binary-install layout check for CPU and CUDA variants
2. Keep `GPU_SUPPORT` metadata visible in CPU builds for now, because the dictionary has historically exposed these flags independent of the active build variant.  Review dictionary changes separately from tracking-output comparisons, and only change this policy if a release explicitly accepts a dictionary-format/metadata change.
3. Add a small dictionary regression case that checks GPU-capability fields for known elements.
4. Verify `make CUDA_AUTO=0 clean` and `make clean` both remove all CPU, MPI, CUDA, and CUDA-verify object variants.
5. Verify that simple `make` builds the expected variants on hosts with:
   - no MPI and no CUDA
   - MPI only
   - CUDA only
   - MPI plus CUDA
6. Document any intentional CPU-visible changes in `doc/CUDA_GPU_SUPPORT.md` and release notes.

Gate:

- A reviewer can distinguish tracking-output invariance from intentional metadata changes.

Completion status:

- `test/gpu_cuda/release_invariance.sh` provides a Phase 10 wrapper for CPU serial and MPI reference comparisons, dictionary SDDS/LaTeX comparisons, binary layout checks, and opt-in clean checks.
- `test/gpu_cuda/check_dictionary_gpu_support.py` checks selected `GPUCapable` metadata values so legacy `GPU_SUPPORT` flag changes are visible in review.
- `test/gpu_cuda/release_checks/print_dictionary.ele` is the small dictionary regression input used by the wrapper.
- The Phase 10 metadata policy is to keep `GPUCapable` dictionary values visible in ordinary CPU builds and make any changes explicit in review.
- Against the clean pre-GPU reference tree, the known intentional dictionary difference is `KOCT` changing `GPUCapable` from `0` to `1`; use `--allow-dictionary-diff` only after confirming that this is the only expected dictionary difference.
- Completed on the local MPI plus CUDA host on May 6, 2026:
  - plain `make -j4` rebuilt ordinary `elegant`, ordinary `Pelegant`, CUDA `gpu-elegant`, and CUDA `gpu-Pelegant` from a clean object state
  - `make CUDA_AUTO=0 clean` and `make clean` both removed the CPU, MPI, CUDA, and CUDA-verify object variants
  - clean pre-GPU serial `elegant` matrix outputs matched the candidate CPU binary at `1e-11`
  - clean pre-GPU MPI `Pelegant` matrix outputs matched the candidate MPI binary at `1e-11` with two ranks
  - rebuilt CPU/GPU quick matrix outputs matched at `1e-11`
- The no-MPI/no-CUDA, MPI-only, and CUDA-only host combinations remain release/CI portability coverage, because this local machine has both MPI and CUDA installed.

## Phase 11: Production Test-Set Expansion

Goal: pull in more real cases from `/home/soliday/oag/apps/src/elegantTestSet/` without turning the CUDA suite into an unbounded full regression run.

Recommended approach:

- Add curated wrappers under `test/gpu_cuda/production_cases/`, similar to the current `lcls0` wrapper.
- Prefer symlinks or thin wrapper inputs that reference the existing test-set data instead of copying large SDDS files into git.
- Keep quick-mode production wrappers small enough for routine CPU/GPU comparisons.
- Keep timing-mode production wrappers bounded with `--baseline --target-seconds 60` and the existing timeout guard.
- Treat stochastic production cases as a later distribution-test category unless deterministic settings can be forced cleanly.

Candidate test-set areas to evaluate first:

- `LCLS1` through `LCLS7`: variations on the current LCLS production case with heavy `RFCW`, `WAKE`, CSR, watch outputs, and some ISR/radiation variants.
- `CLIC1` and `CLIC2`: strong CSR/RFCW coverage, including many `CSRDRIFT` and `CSRCSBEND` sections.
- `csrSegment*`, `bunchComp*`, and `collectiveRing*`: additional CSR and collective-effect coverage beyond the synthetic CSR cases.
- `csbend*`, `csbendAp*`, `ccbend*`, and `ccbendSoftFringe*`: broader bend/fringe/aperture coverage for Phase 17 magnet work.
- `maxamp*`, `collimate*`, `apcontour*`, and `apContourNested*`: production-style aperture and loss semantics for Phase 15.
- `DQCOR*`, `mult*`, `uKickMap*`, and magnet-heavy tracking cases: broader deterministic and fallback magnet coverage.
- `cwiggler*`, `wigglerMoments*`, `bmapxy*`, and `bmxyz*`: field-map/wiggler profiling candidates for Phase 18.
- `track*`, `centroid*`, and `beam*`: lower-level tracking, centroid, and beam-generation regression coverage where they expose GPU residency or reduction issues.

Tasks:

1. Inventory the test-set directories by element families and output types.
2. Select a small smoke subset, roughly 5 to 10 wrappers, that covers the CUDA areas most likely to regress.
3. Select a larger profiling subset for intentional timing work.
4. Add wrapper metadata for each case:
   - source test-set path
   - dominant elements
   - deterministic or stochastic
   - quick-mode scaling rule
   - timing-mode scaling rule
   - expected CUDA path or expected CPU fallback
5. Add macros or wrapper edits only when needed to control particle count, pass count, sample fraction, seeds, and output root.
6. Compare CPU and GPU outputs with the existing SDDS comparator, using `particleID` row labels when applicable.
7. Record cases that are intentionally fallback-only so they still protect CPU synchronization and correctness behavior.
8. Do not enable large production wrappers in default quick CI until their runtime is proven bounded.

Gate:

- The expanded production smoke subset runs in bounded time, matches CPU output within documented tolerances, and identifies which CUDA/fallback paths were exercised.

Initial implementation:

- Added an opt-in `--production-smoke` mode to `test/gpu_cuda/run_benchmarks.sh`.
- Added `test/gpu_cuda/production_smoke.sh` to run the curated CPU production smoke, GPU production smoke, and SDDS comparison as one bounded check.
- Added smoke wrappers for `lcls1`, `clic1`, `csbend1`, `maxamp1`, `collimate1`, `collimate2`, `collimate3`, and `dqcor1`, building on the existing `lcls0` wrapper.
- Added `test/gpu_cuda/production_cases/metadata.tsv` with source paths, dominant element families, deterministic status, scaling rules, expected CUDA or fallback behavior, output coverage, and profiling-only candidates.
- Verified on May 6, 2026:
  - CPU production smoke completed successfully in bounded quick-mode time.
  - `gpu-elegant` production smoke completed successfully on the local RTX 3060 when run with device access.
  - `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase11-final` completed the CPU run, GPU run, and SDDS comparison as one check.
  - CPU/GPU SDDS comparison passed for all 59 common files at `1e-11`.
  - Device logs showed CUDA execution for `lcls0`, `lcls1`, `clic1`, `maxamp1`, and `collimate1`; `csbend1` and `dqcor1` behaved as fallback/compatibility wrappers in this production form.
- Phase 11 practical gate is complete for the initial smoke subset.  Deferred follow-ups: add more profiling wrappers only after measuring runtime, and decide later which production cases are safe enough for default CI.
- Phase 15 later added `collimate2` and `collimate3` to the production smoke subset for open-sided `ECOL` and nonzero-length open-sided collimator regression coverage.  The expanded subset matched all 70 common SDDS files at `1e-11`.

## Phase 12: Automated Benchmark Reports

Goal: make performance claims reproducible without manually mining README notes.

Tasks:

1. Add a benchmark report generator that reads `manifest.tsv`, CUDA verbose logs, compare results, hardware metadata, and git revision.
2. Report both:
   - same-workload speedup for identical particle/pass counts
   - throughput speedup for autoscaled baseline runs
3. Include:
   - GPU name, driver, CUDA runtime, toolkit, `CUDA_ARCH`
   - CPU model if easy to gather locally
   - build command and relevant environment variables
   - particle count, pass count, case label, tolerance, and comparison status
4. Add a `--report` mode or companion script under `test/gpu_cuda/`.
5. Keep timing runs bounded.  Default CPU targets should remain about one minute, and no report command should silently launch hour-scale jobs.
6. Generate a short Markdown summary suitable for release notes.

Gate:

- Re-running representative timing cases produces a single Markdown report with correctness and speedup data.

Initial implementation:

- Added `test/gpu_cuda/report_benchmarks.py`, a Markdown report generator that reads CPU/GPU `manifest.tsv` files, CUDA verbose logs, optional `compare_sdds.py` results, local hardware metadata, git revision, CUDA toolkit/runtime metadata, build commands, and relevant environment variables.
- The report includes same-workload speedup for identical CPU/GPU workloads and throughput speedup for autoscaled baseline manifests with different pass counts.
- Added `--report PATH` to `test/gpu_cuda/production_smoke.sh` so the curated production smoke can rerun CPU, rerun GPU, compare outputs, and emit a single Markdown report in one bounded command.
- Added explicit `--metadata KEY=VALUE` support and a release-notes summary section so reports preserve the GPU run mode, timeout, binary paths, correctness status, speedup range, and most frequent synchronization reasons.
- Verified using existing Phase 11 final production-smoke manifests:
  - report generation reran the SDDS comparison and recorded `all 59 common file(s) matched`
  - generated `test/gpu_cuda/output/reports/phase12-production-smoke.md`
  - generated `test/gpu_cuda/output/reports/phase12-matrix-baseline.md` from autoscaled 60-second CPU/GPU matrix baseline manifests, reporting throughput speedup separately from same-workload speedup
- Verified end-to-end on May 6, 2026 with `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase12-endtoend --report test/gpu_cuda/output/reports/phase12-endtoend-production-smoke.md`; the report includes GPU name, driver, CUDA driver capability, CUDA toolkit, CUDA runtime library, CPU model, git revision, correctness status, speedups, CUDA kernel/transfer counters, and top synchronization reasons.
- Rechecked after metadata/report-summary hardening with AST parsing, shell syntax checks, report regeneration from existing manifests, and `production_smoke.sh --dry-run --require-gpu --report`.
- Phase 12 practical gate is complete for the local production-smoke and matrix-baseline reports.  Deferred follow-ups: add CI artifact upload and standardize a release-note report filename once the CI layout is decided.

## Phase 13: Data Residency And Transfer Reduction

Goal: reduce CPU synchronization overhead across mixed GPU/CPU sections.

Tasks:

1. Add verbose transfer accounting by reason:
   - output/diagnostics
   - CPU-only element
   - aperture/loss fallback
   - MPI scatter/gather
   - verification
   - collective-effect host work
2. Add a benchmark summary of most frequent `forceParticlesToCpu` reasons for each case.
3. Preserve GPU residency through more coordinate-neutral elements where no output side effect reads particle coordinates.
4. Audit diagnostics and watch/output paths to identify which truly need full coordinate synchronization.
5. Add lightweight host snapshots for scalar diagnostics when a full coordinate copy is avoidable.
6. Revisit `GPU_VERIFY` paths that reset residency and accidentally hide downstream aperture or helper kernels.

Gate:

- At least one existing benchmark shows a measurable speedup from fewer transfers without weakening fallback correctness.

Status:

- Started on May 6, 2026 with runtime instrumentation rather than changing residency behavior immediately.
- Added CUDA-side synchronization accounting for `forceParticlesToCpu` requests, separating synchronization requests from actual device-to-host coordinate copy events.
- Categorized synchronization requests into output/diagnostics, CPU-only element, aperture/loss fallback, MPI scatter/gather, verification, collective host work, reductions, deallocation, and other.
- Kept the existing verbose per-reason log lines intact, and added aggregate `elegant CUDA: sync requests=... copies=...` summary lines so existing debugging remains useful while reports can consume compact counters.
- Updated `test/gpu_cuda/report_benchmarks.py` to include sync request/copy counts and top sync categories per case, plus total synchronization accounting in the release-notes summary.
- Verified with a GPU-required quick matrix run and with `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-sync --report test/gpu_cuda/output/reports/phase13-sync-production-smoke.md`; the production smoke passed with all 59 common SDDS files matching at `1e-11`.
- Current production-smoke accounting shows CPU-only element transitions dominate synchronization traffic, followed by aperture/loss fallback requests and deallocation.  This makes CPU-element residency preservation the first high-value reduction target.
- Added detailed CPU-element synchronization reasons with element type, name, and occurrence, plus report-side "top sync targets" so one-per-element lattices roll up to actionable groups such as `CPU element: RFCW` or `CPU element: CSRDRIFT`.
- Marked `MAXAMP` as a GPU-passive element because it only updates active aperture state; this preserves device residency through the element and lets the following `limit_amplitudes` or `elimit_amplitudes` call use the existing CUDA aperture path.
- Verified the focused `maxamp1` case with all 6 common SDDS files matching at `1e-11`; sync requests dropped from 181 in the detailed pre-change production smoke to 12 after the passive `MAXAMP` change.
- Verified the full production smoke with `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-maxamp-passive --report test/gpu_cuda/output/reports/phase13-maxamp-passive-production-smoke.md`; all 59 common SDDS files matched at `1e-11`, and total sync requests dropped from 574 to 405.
- Added `--particles` and `--passes` overrides to `test/gpu_cuda/run_benchmarks.sh` so same-workload CPU/GPU timing gates can be pinned without relying on larger autoscaled sample defaults.
- Ran a pinned `maxamp1` same-workload timing with 3000 particles and 245 passes, aimed at the bounded one-minute CPU timing scale.  Correctness still matched all 6 common files at `1e-11`, but the GPU run was slower (35.27s CPU vs 39.89s GPU, 0.88x) because 241 `elimit_amplitudes` loss-fallback copies dominate after the passive `MAXAMP` change.
- Marked `WATCH` as GPU-passive and moved synchronization to the actual `WATCH` coordinate/parameter/FFT output calls.  This keeps disabled or non-output watch points from forcing a generic CPU-element sync and makes active output appear as output/diagnostics in the logs.
- Verified the `WATCH` refinement with `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-watch-passive --report test/gpu_cuda/output/reports/phase13-watch-passive-production-smoke.md`; all 59 common SDDS files matched at `1e-11`.  The LCLS watch synchronizations now report as `WATCH coordinate output`.
- Added `copyParticlesToCpuReadOnly()` for diagnostic and reduction paths that need a host snapshot but do not modify particle coordinates.  Unlike `forceParticlesToCpu()`, this copies the device coordinates to the host without marking the device copy stale.
- Converted the verified read-only callers to the new API: `WATCH` coordinate/parameter/FFT output, final `dump_watch_parameters`/`dump_watch_FFT`, `MARK` fitpoint output, trajectory-centroid fallback below the CUDA reduction threshold, and below-threshold `compute_centroids`/`accumulate_beam_sums` fallbacks.
- Left `performSliceAnalysisOutput` on the mutable synchronization path because `performSliceAnalysis` temporarily swaps the longitudinal coordinate while computing time coordinates, even though it restores the value before returning.
- Verified the read-only synchronization split with serial, Pelegant, CUDA, and CUDA Pelegant rebuilds plus `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-readonly-sync --report test/gpu_cuda/output/reports/phase13-readonly-sync-production-smoke.md`; all 59 common SDDS files matched at `1e-11`.
- The read-only split did not materially reduce the aggregate production-smoke synchronization count by itself: the `phase13-readonly-sync` report still shows 405 synchronization request(s) and 396 device-to-host copy event(s), dominated by CPU-only `RFCW` and `CSRDRIFT`/`CSRCSBEND` transitions.  It does, however, remove unnecessary device invalidation from read-only diagnostics and reductions so future passive/GPU-capable elements can keep using resident coordinates.
- Extended the sync accounting and benchmark reports with read-only versus mutable request counts.  The report table now shows request/copy/read-only counts, while the release summary records total read-only and mutable requests.
- Changed the opt-in CUDA `limit_amplitudes`/`elimit_amplitudes` compaction path to synchronize its loss-output host snapshot with `copyParticlesToCpuReadOnly()` instead of invalidating the device copy after successful GPU compaction.
- Verified the legacy compaction-output change on `phase3_elimit_loss`; all 5 common SDDS files matched at `1e-11`.  The compaction run reported 143 sync request(s), 143 copy event(s), 142 read-only request(s), and 1 mutable deallocation request.
- Generated `test/gpu_cuda/output/reports/phase13-aperture-compact-readonly.md`; the quick same-workload timing remains slower than CPU (0.31x), so aperture compaction remains opt-in and is not a Phase 13 performance win yet.
- Re-ran the standard production smoke with the new counters using `test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-sync-readonly-counters --report test/gpu_cuda/output/reports/phase13-sync-readonly-counters-production-smoke.md`; all 59 common SDDS files matched at `1e-11`.  The standard smoke reports 405 sync request(s), 396 device-to-host copy event(s), 12 read-only request(s), and 393 mutable request(s).
- At this point the Phase 13 gate remained open for timing: transfer reduction and correctness were verified, but the longer same-workload timing showed that the next useful performance target was reducing lossy aperture fallback copies or moving to the larger remaining CPU-element targets (`RFCW`, `CSRDRIFT`/`CSRCSBEND`) rather than counting the passive `MAXAMP` change as a complete speedup win.
- Added a conservative short-GPU-island avoidance heuristic for mixed CPU/GPU lattices.  When the host coordinates are already current, the next eligible element is a simple matrix element, and a CPU-only element follows within a short configurable run, CUDA now leaves that short simple-matrix island on the CPU instead of copying host-to-device only to copy device-to-host again at the next CPU-only element.
- The short-island heuristic is enabled by default, can be disabled with `ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS=0`, and is bounded by `ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS` with a default of 4 simple matrix elements.  It does not force a device-to-host copy; it only preserves an already-current host copy.
- Verified the focused `rfcw` case with `ELEGANT_GPU_MODE=required ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1`; all 4 common SDDS files matched at `1e-11`, 7 short GPU islands were skipped, and only the final deallocation synchronization remained.
- Verified the focused `clic1` case with `ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS=4`; all 8 common SDDS files matched at `1e-11`, the `RFCW` CPU-element synchronization count dropped from the previous production-smoke pattern of 141 `RFCW` syncs plus deallocation to 78 `RFCW` syncs plus deallocation, and the focused quick timing was 1.93s GPU for the 2000-particle/1-pass case.  A max-8 exploratory run cut synchronization further but was slower, so the default was set to 4.
- Verified the full production smoke with `ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS=4 test/gpu_cuda/production_smoke.sh --require-gpu --label-prefix phase13-short-island-max4 --report test/gpu_cuda/output/reports/phase13-short-island-max4-production-smoke.md`; all 59 common SDDS files matched at `1e-11`.  The aggregate synchronization accounting dropped from the prior standard-smoke 405 request(s) and 396 device-to-host copy event(s) to 278 request(s) and 269 copy event(s), with 8 read-only and 270 mutable request(s).  The regenerated report includes short-island accounting and shows 296 simple-matrix element(s) kept on CPU with `maxElements=4`.
- The Phase 13 timing gate is now met: the max-4 production-smoke report shows measurable same-workload speedups from reduced transfers on existing benchmark cases without the max-8 slowdown, including `clic1` at 2.92s CPU / 1.89s GPU and `maxamp1` at 3.01s CPU / 1.50s GPU.
- Rebuilt `gpu-elegant` and CUDA Pelegant after the shared GPU bookkeeping/default-threshold changes.  A focused default-settings `clic1` run confirmed `maxElements=4`, matched all 8 common SDDS files at `1e-11`, skipped 90 simple-matrix element(s), and reported 79 total synchronization request(s).
- Deferred transfer-reduction follow-ups: tune the short-island default against larger production lattices, reduce lossy aperture fallback copies, add direct GPU support for the remaining high-frequency CPU-only sync targets where worthwhile, and continue CSR-specific work in Phase 14.

## Phase 14: CSR Improvement Pass

Goal: turn CSR from a modest speedup into a production-useful CUDA path.

Tasks:

1. Profile `phase6_csr_csbend`, `csr`, and `lcls0` with transfer-reason logging.
2. Keep CSR range determination, bin sizing, histogram, derivative preparation, wake calculation, and particle kick data resident where possible.
3. Replace the now-removed non-resident CSR histogram path with a resident path that avoids per-kick host coordinate copies.
4. Add `CSR_LAST_WAKE` verification beyond the current `T1`, `T2`, and `dGamma` arrays.
5. Add or extend cases for:
   - `CSRDRIFT`
   - Stupakov/Saldin modes
   - integrated Green's function
   - high/low frequency filters
   - CSR plus aperture checks
   - production LCLS0 CSR sections
6. Decide whether cuFFT helps for any CSR convolution path after profiling.

Gate:

- CSR-heavy same-workload speedup improves materially over the current roughly 1.2x result, or the new path remains opt-in.

Status:

- Started on May 6, 2026 from the now-removed non-resident CSR histogram prototype, leaving default CSR behavior unchanged.
- Implemented packed-coordinate staging for that non-resident CSR histogram slice: the GPU histogram copied one longitudinal coordinate double per particle instead of staging `totalPropertiesPerParticle` doubles per particle.  CPU-compatible range determination, bin sizing, SG preparation, wake bookkeeping, and particle kicks remained CPU-owned.
- Correctness checks passed for `phase6_csr_csbend` and `csr` quick CPU/GPU comparisons at `1e-11`; `GPU_VERIFY` also verified CSR `ctHist`, `T1`, `T2`, and `dGamma` for those cases.
- Production LCLS0 quick smoke passed at `1e-11` across 15 common SDDS files.  The bounded quick timing was CPU 8.72 seconds and GPU opt-in 6.05 seconds, for a 1.44x same-workload result.  CUDA logging showed 400 CSR calls, 0.831 seconds kernel time, 0.027 seconds H2D, 0.026 seconds D2H, and 90 synchronization requests dominated by CPU-only `RFCW`, `CSRCSBEND`, and `CSRDRIFT` transitions.
- The bounded isolated `phase6_csr_csbend` timing gate used the plan's one-minute CPU-target style: CPU selected 20,000 particles and 261 passes, then the GPU path ran the same workload.  Final outputs matched at `1e-11`; timing was CPU 35.23 seconds and GPU 32.69 seconds, or about 1.08x.  CUDA logging showed 8,352 CSR calls, 1.211 seconds kernel time, 0.266 seconds H2D, 0.160 seconds D2H, and 261 synchronization requests, so transfer volume is no longer the dominant isolated-case cost.
- Continued by avoiding ordinary CSR wake component copies: CUDA now copies `T1` and `T2` back only when CSR wake output or `GPU_VERIFY` needs them, while always returning `dGamma` for particle kicks and `CSRDRIFT` state.
- Added `phase14_csr_last_wake`, a focused `CSRCSBEND` last-wake-output plus `CSRDRIFT` regression case.  CPU/GPU quick comparison matched all 5 common files at `1e-11`, including the `.csr` wake output file, which verifies the conditional `T1`/`T2` return path.
- The same 20,000-particle, 261-pass isolated timing gate still matched at `1e-11`.  D2H time dropped from 0.160 seconds to 0.098 seconds, but wall time remained effectively flat at 32.72 seconds, so this is a correctness-preserving transfer cleanup rather than a Phase 14 speed-gate win.
- Added `GPU_VERIFY` coverage for the actual inline `CSRCSBEND` CUDA wake handoff into `CSR_LAST_WAKE`.  The CUDA helper now preserves a CPU-shadow `dGamma` for the last wake calculation, and `track_through_csbendCSR` compares that shadow plus the produced scalar handoff state before memory-minimization cleanup.  IGF, Stupakov, and post-CSRDRIFT Saldin `CSR_LAST_WAKE` state coverage remains a follow-up.
- Verified the new handoff check with `phase14_csr_last_wake` and `phase6_csr_csbend` GPU verification runs.  The `phase14_csr_last_wake` CPU/GPU comparison matched all 5 common files at `1e-11`, and the GPU verify log reported explicit `CSR_LAST_WAKE passed` checks.
- Extended the `CSR_LAST_WAKE` verifier to cover CSR wake-filter-table postprocessing by applying the same `WAKE_FILTER_FILE` table to the CPU-shadow `dGamma` before comparing the final handoff state.  Added `phase14_csr_filters`, which exercises high/low cutoff filters, `WAKE_FILTER_FILE`, `OUTPUT_LAST_WAKE_ONLY`, `CSRDRIFT`, and a surrounding aperture-state element.  CPU, normal GPU, and GPU_VERIFY quick runs matched all 5 common files at `1e-11`; GPU_VERIFY reported explicit `CSR_LAST_WAKE passed` checks for the filtered case.
- Added `phase14_csr_saldin54`, a focused regression for the CUDA-produced `CSR_LAST_WAKE` state feeding `CSRDRIFT,USE_SALDIN54=1`.  The case includes last-wake CSR output plus `SALDIN54_OUTPUT` so the Saldin normalization table is compared along with final particles, centroids, sigma, and CSR wake output.  CPU, normal GPU, and GPU_VERIFY quick runs matched all 6 common files at `1e-11`; GPU_VERIFY reported CSR wake-array checks and explicit `CSR_LAST_WAKE passed` checks.  Quick timing was CPU 0.18 seconds and normal GPU 0.53 seconds, dominated by startup and not a speed-gate result.
- Attempted a focused integrated-Greens-function regression with `CSRCSBEND,STEADY_STATE=1,IGF=1`, but the serial CPU binary crashed before any CUDA comparison.  IGF coverage remains deferred until the existing CPU-side IGF crash can be diagnosed separately.
- Added small CSR wake-transfer and prep cleanups: the CUDA CSR scratch state now caches the CPU-computed denominator array on the device when `nBins` and `dct` are unchanged, packs `ctHist` plus `ctHistDeriv` into one scratch upload for the wake kernel, and the GPU build skips recomputing the host-side `denom` table when the existing table still covers the current `nBins` and `dct`.  This avoids repeated `denom` uploads, replaces two wake-input H2D copies with one, and removes repeated `pow()` prep in stable-bin CUDA runs while preserving CPU-computed histogram, derivative, and denominator values.  `phase6_csr_csbend` and `phase14_csr_saldin54` CPU/GPU quick comparisons still matched at `1e-11`; `phase6_csr_csbend` GPU_VERIFY also passed with CSR wake-array and `CSR_LAST_WAKE` checks.  In the Saldin54 quick smoke, normal-GPU logged H2D time moved from 0.001803 seconds before these cleanups to 0.001555 seconds after the denominator device cache and 0.000699 seconds after packed wake inputs.  After the host prep cache, quick smoke H2D timings remained in the same noisy sub-millisecond to low-millisecond range, so this is useful per-kick cleanup rather than a speed-gate win.
- Prototyped a non-resident CUDA CSR particle-kick kernel.  The kernel applied the non-filtered CSR `dGamma` interpolation to packed `ct/x/dp` particle data and let intermediate wake calculations leave `dGamma` resident until host output or final `CSR_LAST_WAKE` state needed it.  Correctness held: `phase6_csr_csbend` and `phase14_csr_saldin54` CPU/GPU quick comparisons matched at `1e-11`, and `phase6_csr_csbend` GPU_VERIFY still passed CSR wake-array and `CSR_LAST_WAKE` checks.
- The first CSR kick prototype did not meet the speed gate, so it remains explicitly opt-in and is off by default.  On the 20,000-particle, 261-pass same-workload gate, CPU was 35.24 seconds, the default GPU histogram path with CSR kick disabled was 32.66 seconds, and the CSR kick prototype was 33.15 seconds.  CUDA logging showed the prototype raised CSR kernel calls from 8,352 to 12,528 and increased transfer time from H2D/D2H 0.244/0.098 seconds to 0.500/0.200 seconds.  The result confirms that a host-packed kick kernel is not enough; the next real speed target is keeping particle coordinates resident through the CSRCSBEND body, histogram, wake, and kick together.
- Found that the serial CUDA build inherited the base `-Og` host optimization flag after `-O3`, so `gpu-elegant` was running the CPU-owned parts of CSRCSBEND in debug-optimization mode.  The CUDA build block now appends `-O3` after the base flags for CUDA serial and GPU_VERIFY object directories, leaving the plain serial and MPI build flags unchanged.  With the same 20,000-particle, 261-pass gate, the final current-binary default GPU histogram run improved from 32.66 seconds to 31.66 seconds, with a best observed post-change run of 30.88 seconds; all 4 common SDDS files matched at `1e-11`, and GPU_VERIFY quick smoke also passed CSR wake-array and `CSR_LAST_WAKE` checks.
- Retested the non-resident CSR kick prototype after the CUDA-host `-O3` rebuild; it did not show a robust speed win at 31.54 seconds for the same 20,000-particle, 261-pass gate and still added CSR kernels plus transfer time.  The obsolete host-packed prototype was removed after resident CSR became the default path.
- Tested and rejected two transfer/CPU-side experiments.  A CSR-specific sync-island heuristic removed the repeated `CSRCSBEND BC#1` device-to-host synchronizations but moved too many simple matrix drifts back to the CPU, slowing the gate to 36.16 seconds; that code was removed.  An OpenMP version of the CPU fallback CSR kick loop matched output but slowed the gate to 37.67 seconds because the per-kick parallel-region overhead dominated; that code was also backed out.
- Added a resident CSRCSBEND path, now default-on for the guarded subset with `ELEGANT_GPU_ENABLE_CSR_RESIDENT=0` as the fallback override.  For the supported Phase 14 subset (non-backtracking, non-radiating, non-IGF, no wake-filter table, no in-element particle output, no active slice analysis, and no aperture data requiring CPU loss handling), the path copies the already-transformed CSRCSBEND coordinates and `beta0` state to the device, tracks each CSRCSBEND body slice on resident coordinates, computes ct min/max with a dedicated CUDA coordinate-range reduction, fills the histogram directly from device coordinate 4, computes the CSR wake on the device, and applies the CSR kick in place on resident device coordinates.  It still hands particles back to the CPU for final CSRDRIFT handoff state, final edge/coordinate transforms, and unsupported options.
- The resident prototype produced the first material isolated CSR speed-gate win.  On the 20,000-particle, 261-pass `phase6_csr_csbend` same-workload gate, CPU remained 35.24 seconds, the previous CUDA-host `-O3` non-resident run was 31.66 seconds, and the resident prototype ran in 16.31-16.32 seconds while matching all 4 common SDDS files at `1e-11`.  After caching the packed CSRCSBEND body data, the same gate ran in 16.24 seconds and still matched all 4 common SDDS files at `1e-11`.  Replacing the resident ct-range beam-sums call with a coordinate-only min/max reduction brought the same gate to 13.99 seconds.  Preserving per-particle resident scratch when CSR wake arrays grow, plus reusing the CSRCSBEND body rollback/loss buffers instead of allocating them every body slice, removed the one-time resident body fallback and brought the same gate to 13.45 seconds.  A targeted short-GPU-island exception keeps the simple matrix/drift island immediately before a resident-eligible `CSRCSBEND` on the GPU while leaving the default mixed-lattice short-island behavior unchanged elsewhere; this lowered the same gate to 9.82 seconds.  Skipping CSRDRIFT handoff prep when no immediate `CSRDRIFT` follows reduced the gate to 9.30 seconds.  A checked CUDA simple-final-transform kernel for no-offset, no-fringe-change resident CSRCSBENDs reduced it to 8.96 seconds.  Keeping those final coordinates GPU-resident when no immediate `CSRDRIFT` follows halved the remaining CSRCSBEND synchronization count and brought the final current-binary gate to 8.46 seconds, or about 4.17x faster than the CPU reference.  All 4 common SDDS files still matched at `1e-11`.  CUDA logging for the latest run showed 5,220 magnet/body/final kernels, 12,528 CSR kernels, 6,264 reductions, 3.36 seconds kernel time, 0.274 seconds H2D, and 0.249 seconds D2H; element timing showed `DRIF` at 1.40 seconds and `CSRCSBEND` at 6.45 seconds.  Fallback handling remains part of the prototype, and `GPU_VERIFY` builds still force the final resident handoff so CPU/GPU verification reductions compare fresh host state.
- Added a narrow resident-entry path for simple no-offset/no-effective-entry-edge CSRCSBENDs.  When particles are already device-current from the preceding simple matrix/drift island, CSRCSBEND now dispatches through the GPU wrapper, applies the initial CSR coordinate transform and `beta0` setup on the device, and avoids the per-CSRCSBEND device-to-host synchronization that previously happened before copying the same particles back to the GPU.  The 20,000-particle, 261-pass `phase6_csr_csbend` gate dropped from 8.46 seconds to 5.27 seconds while all 4 common SDDS files still matched CPU at `1e-11`; this is about 6.69x faster than the 35.24-second CPU reference.  CUDA logging for the new gate showed only the final deallocation synchronization, H2D/D2H time of 0.085/0.011 seconds, and element timing of `DRIF` 1.41 seconds and `CSRCSBEND` 3.24 seconds.  `GPU_VERIFY` keeps the previous CPU-handoff path for direct CPU/GPU reduction comparison, while normal GPU runs use the resident-entry path when its simple-geometry guard passes.
- `GPU_VERIFY` quick smoke passed with `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1`: resident `ctHist`, `T1`, `T2`, `dGamma`, and `CSR_LAST_WAKE` checks passed, and final SDDS output matched CPU at `1e-11`.  The now-removed non-resident histogram quick path also still matched CPU after the resident changes.
- Production smoke with `ELEGANT_GPU_ENABLE_CSR_RESIDENT=1` passed all 59 common SDDS files at `1e-11`; the latest report is `test/gpu_cuda/output/reports/phase14-csr-keep-resident-production.md`.  The final production quick run showed LCLS0 at 1.64x and LCLS1 at 1.84x same-workload speedup, with same-workload speedups ranging from the fallback-heavy `collimate1` case at 0.17x to `maxamp1` at 1.98x.  CPU-only `RFCW`, resident CSR final CPU handoffs, and aperture/loss fallbacks still dominate synchronization.  These small production quick timings remain useful correctness and profiling signals, while the isolated 20,000-particle CSR gate is the Phase 14 speed gate.
- Report: `test/gpu_cuda/output/reports/phase14-csr-keep-resident-production.md`.
- Production smoke after resident-entry dispatch passed all 59 common SDDS files at `1e-11`; the report is `test/gpu_cuda/output/reports/phase14-csr-entry-production.md`.  The bounded production smoke still shows CPU-only `RFCW`, remaining CSR final handoffs for CSRDRIFT/diagnostic state, and aperture/loss fallbacks as the leading synchronization targets.
- Wrapped Phase 14 on May 6, 2026.  The isolated CSR speed gate is met with the resident CSRCSBEND path: the 20,000-particle, 261-pass `phase6_csr_csbend` workload is 5.27 seconds with CUDA vs 35.24 seconds with CPU, about 6.69x faster, while all 4 common SDDS files match at `1e-11`.  Keep the verified simple CSR subset guarded, but enable it by default and use `ELEGANT_GPU_ENABLE_CSR_RESIDENT=0` for the CPU/non-resident fallback override.

Deferred Phase 14 follow-ups:

- Broaden default resident CSRCSBEND support beyond the current guarded subset only after focused correctness and production-shaped timing evidence.
- Remove or reduce remaining production CPU handoffs for `CSRDRIFT` state, diagnostic/final CSR state, and CSR-adjacent aperture/loss handling.
- Extend resident CSR support to currently unsupported modes: integrated Green's function, Stupakov modes, wake-filter-table resident support, radiating/backtracking CSR cases, active slice analysis, and in-element output cases.
- Revisit integrated Green's function coverage after the existing CPU-side IGF crash is diagnosed separately.
- Pull in more production CSR regression coverage from `/home/soliday/oag/apps/src/elegantTestSet/`, especially LCLS-style CSR sections.
- Reconsider cuFFT only if later profiling shows a real CSR convolution hotspot that the current resident path does not address.

## Phase 15: Aperture And Loss Handling

Goal: make lossy aperture cases faster without changing particle-loss semantics.

Tasks:

1. Implement parallel survivor/lost partitioning using prefix sums or a stable partition strategy.
2. Preserve CPU-compatible semantics for:
   - `accepted`
   - lost-particle output
   - `particleID`
   - `nLeft`
   - final charge and loss locations
3. Extend beyond simple rectangular/elliptical `MAXAMP` if timing justifies it:
   - `RCOL`
   - `ECOL`
   - `SCRAPER`
   - `imposeApertureData`
   - `removeInvalidParticles`
4. Compare loss sets by `particleID`, not only final coordinate files.
5. Keep exact CPU fallback for ambiguous boundary particles unless a documented tolerance rule is accepted.

Gate:

- A lossy aperture benchmark is faster than CPU fallback and matches loss semantics within the documented rule.

Status:

- Started on May 6, 2026 with simple rectangular and elliptical `MAXAMP` loss handling.  The legacy exact compaction path copied only the lost tail rows back to the host in normal GPU builds, while `GPU_VERIFY` still performed the full host synchronization needed for CPU shadow comparisons.  That legacy path was later removed after the stable prefix-sum compaction path superseded it.
- Added a separate stable parallel compaction prototype, originally behind `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` and later made automatic for runs that do not request `losses` output.  It uses CUDA survivor flags, prefix sums, and stable scatter into reusable aperture scratch storage, then copies only the lost tail rows needed for loss bookkeeping when `.los` output is present and the stable path is explicitly forced.  The stable scatter now promotes the scratch coordinate buffer to become the resident device coordinate buffer instead of performing a full device-to-device copy back into the original buffer; coordinate scratch capacity and prefix scratch capacity are tracked separately so the two coordinate buffers can safely ping-pong across compactions.  For `accepted` bookkeeping, the same survivor prefix now scatters the accepted array on the device by default under the parallel-compaction option, with `ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE=0` available as a fallback.  Mutable CPU synchronization and final teardown copy the accepted array back before CPU code or output can read it.  The path preserves `accepted`, loss sets and values by `particleID`, `nLeft`, final charge accounting, and simple global loss coordinates for the supported `MAXAMP`, `RCOL`, `ECOL`, and ideal `SCRAPER` subsets.  `RCOL` now includes zero-length and nonzero-length open-sided subsets, including the nonzero-length open-sided global-loss-coordinate subset; inverted `RCOL` still falls back.  `ECOL` now includes open-sided collimators, mixed `YEXPONENT` collimators, and global loss-coordinate output for the non-inverted even-exponent subset, matching the current CPU entrance/exit exponent semantics; inverted `ECOL` still falls back.  One-sided ideal `SCRAPER` loss compaction now supports global loss-coordinate output for entrance and exit losses by mirroring the CPU scraper `dz` convention before calling `convertLocalCoordinatesToGlobal`; two-sided ideal `SCRAPER` now reaches the existing CUDA `RCOL` stable-compaction path instead of being forced to CPU before dispatch.  `imposeApertureData` now uses the same stable compaction path for interpolated rectangular physical aperture data, and `removeInvalidParticles` uses it for a narrow identity-`RFCA` subset that exists to keep particles resident through invalid-particle loss handling without claiming general `RFCA` support.
- Correctness matched CPU output at `1e-11` for the focused `phase3_limit_loss`, `phase3_elimit_loss`, `phase3_rcol_loss`, `phase3_ecol_loss`, `phase3_scraper_loss`, `phase15_aperture_data_loss`, `phase15_remove_invalid_loss`, `phase15_rcol_open_loss`, `phase15_ecol_mixed_loss`, `phase15_ecol_open_global_loss`, `phase15_rcol_open_global_loss`, `phase15_scraper_global_loss`, and `phase15_scraper_two_sided_global_loss` cases, comparing all 5 common SDDS files in each case.  The production-style `maxamp1` quick case also matched all 6 common files, including `.acc` and `.los`, using `particleID`-aware SDDS comparison where available.
- Focused quick timing improved with stable parallel compaction, especially after removing the final device-to-device copy, but these tiny cases remain startup dominated and are still slower than CPU references: `phase3_limit_loss` was 0.25 seconds GPU vs 0.15 seconds CPU, `phase3_elimit_loss` 0.25 seconds GPU vs 0.15 seconds CPU, `phase3_rcol_loss` 0.24 seconds GPU vs 0.17 seconds CPU, `phase3_ecol_loss` 0.23 seconds GPU vs 0.16 seconds CPU, `phase3_scraper_loss` 0.24 seconds GPU vs 0.18 seconds CPU, `phase15_aperture_data_loss` 0.25 seconds GPU vs 0.19 seconds CPU, and `phase15_remove_invalid_loss` 0.17 seconds GPU vs 0.02 seconds CPU.
- The production-style `maxamp1` 3000-particle, 60-pass timing is now a small speed win for stable compaction: CPU took 18.47 seconds, the existing GPU fallback-heavy path took 12.99 seconds, the scratch-promotion stable path took 12.62 seconds, and the accepted-device stable path took 12.55 seconds, with all 6 common SDDS files matching at `1e-11`.  The accepted-device report is `test/gpu_cuda/output/reports/phase15-accepted-device-maxamp1.md`; it reduced synchronization requests from 273 to 137 and D2H time from about 0.0023 seconds to about 0.0011 seconds relative to the scratch-promotion-only stable path.  Keep the stable path opt-in until this small margin is confirmed on broader aperture production cases and the remaining lost-tail/global-loss bookkeeping overhead is reduced further.
- The zero-length open-sided `RCOL` extension was added for the `collimate1` family.  The focused `phase15_rcol_open_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`; CPU took 4.14 seconds, the fallback-heavy GPU path took 1.10 seconds, and the stable open-`RCOL` compaction path took 0.96 seconds.  The report is `test/gpu_cuda/output/reports/phase15-rcol-open-30k10.md`.
- The nonzero-length open-sided `RCOL` global-loss-coordinate extension was added for the stable compaction path.  It preserves CPU's special `dz=0` global-coordinate convention for finite open-side exit losses while still using the exit length for non-finite exit losses.  The focused `phase15_rcol_open_global_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`, including `.los` with global loss columns; CPU took 4.55 seconds and the stable CUDA compaction path took 1.05 seconds, for a 4.33x same-workload speedup.  The report is `test/gpu_cuda/output/reports/phase15-rcol-open-global-30k10.md`.
- The one-sided ideal `SCRAPER` global-loss-coordinate extension was added for the stable compaction path.  It preserves CPU's entrance-loss `dz=0` convention and its exit-loss scraper overshoot `dz` convention when filling `.los` global columns.  The focused `phase15_scraper_global_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`, including `.los` with `X`, `Z`, and `thetaX`; CPU took 1.61 seconds and the stable CUDA compaction path took 0.53 seconds, for a 3.04x same-workload speedup.  The report is `test/gpu_cuda/output/reports/phase15-scraper-global-30k10.md`.  The CUDA verify build also compiled and matched CPU output for the quick case; in `GPU_VERIFY` builds, this lossy scraper path still intentionally uses the CPU fallback because the normal stable-compaction path is disabled there.
- The two-sided ideal `SCRAPER` extension was added by mirroring CPU behavior: supported two-sided scrapers are translated to an in-memory `RCOL` and routed through the existing CUDA rectangular-collimator stable compaction path.  The old `do_tracking.c` pre-dispatch CPU synchronization is now limited to the still-unsupported all-loss and material-interaction two-sided cases.  The focused `phase15_scraper_two_sided_global_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`, including `.los` global columns; CPU took 1.12 seconds and the stable CUDA compaction path took 0.45 seconds, for a 2.49x same-workload speedup.  The report is `test/gpu_cuda/output/reports/phase15-scraper-two-sided-global-30k10.md`.  The CUDA verify build also compiled and matched CPU output for the quick case; as with other lossy stable-compaction paths, `GPU_VERIFY` disables the normal stable scatter and falls back for the lossy aperture step.
- The mixed-exponent `ECOL` extension was added for non-inverted even-exponent collimators.  The focused `phase15_ecol_mixed_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`; CPU took 2.18 seconds and the stable CUDA compaction path took 0.61 seconds, for a 3.57x same-workload speedup.  The report is `test/gpu_cuda/output/reports/phase15-ecol-mixed-30k10.md`.
- The open-sided mixed-exponent `ECOL` global-loss-coordinate regression was added for the same stable compaction subset.  The focused `phase15_ecol_open_global_loss` 30,000-particle, 10-pass same-workload timing matched all 5 common SDDS files at `1e-11`, including `.los` with `X`, `Z`, and `thetaX`; CPU took 2.25 seconds and the stable CUDA compaction path took 0.63 seconds, for a 3.57x same-workload speedup.  The report is `test/gpu_cuda/output/reports/phase15-ecol-open-global-30k10.md`.  The CUDA verify build also matched CPU output for the quick case, with the expected lossy-aperture fallback in `GPU_VERIFY`.
- Added production wrappers for `collimate2` and `collimate3`.  `collimate2` exercises open-sided `ECOL` with watch output; `collimate3` combines a nonzero-length open-sided `RCOL` with repeated nonzero-length open-sided `ECOL` elements and acceptance output.  Both matched CPU output at `1e-11`; `collimate3` matched all 6 common files including `.acc` and `.los`.  The 30,000-particle, 3-pass `collimate3` timing matched CPU but remained startup/synchronization dominated at 0.38 seconds GPU vs 0.09 seconds CPU, so this is correctness coverage rather than a speed gate.
- The curated production smoke with `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` and accepted-device compaction enabled matched all 70 common SDDS files at `1e-11`; the latest report is `test/gpu_cuda/output/reports/phase15-scraper-two-sided-production-smoke.md`.  In that quick smoke, `maxamp1` ran as the useful aperture speed signal, while `collimate1`, `collimate2`, and `collimate3` remained tiny watch/lost-tail-heavy correctness checks.  They are kept in the smoke subset as aperture loss-semantics regressions, not performance claims.
- Added `phase15_remove_invalid_loss`, a fixed SDDS-beam regression that drives four invalid particles through the narrow identity-`RFCA` CUDA entry path.  Default CUDA fallback and normal CUDA with `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` both matched all 5 common SDDS files at `1e-11`; the parallel-compaction log showed three CUDA RFCA entries, four aperture kernels, and a read-only lost-tail synchronization for 4 rows.  `GPU_VERIFY` also matched CPU output after the reduction verifier was made tolerant of matching CPU/GPU `NaN` values.  Quick timing was still startup dominated: CPU 0.01 seconds, normal CUDA 0.19 seconds, and CUDA verify 0.20 seconds.
- Wrapped Phase 15 on May 7, 2026.  The practical aperture/loss gate is met for the verified subset: focused same-workload `RCOL`, `ECOL`, and ideal `SCRAPER` cases now show useful speedups from about 2.49x to 4.33x while preserving CPU loss semantics at `1e-11`, and the expanded production aperture smoke remains clean across all 70 common SDDS files.  The default policy now enables stable compaction only when no `losses` file is requested; `.los` output still requires an explicit `ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=1` because the remaining work is lost-tail/global-loss overhead reduction, broader production timing, and unsupported edge-case expansion rather than core correctness.

Deferred Phase 15 follow-ups:

- Further reduce stable-compaction overhead, especially repeated small lost-tail transfers and global-loss bookkeeping.
- Turn stable compaction into the default `.los` `MAXAMP` loss mode only after it beats the current GPU fallback path by a more comfortable margin across broader same-workload timings and preserves loss-row bookkeeping without repeated host row copies.
- Extend the same loss-semantics checks to inverted, remaining global-loss-coordinate, all-loss, and material-interaction `RCOL`, `ECOL`, and `SCRAPER` cases only when timing shows they are worth moving beyond the CPU fallback.
- Broaden `removeInvalidParticles` beyond the narrow identity-`RFCA` trigger only after broader `RFCA` CUDA semantics or profiling justify it.
- Broaden production aperture regression coverage with additional cases from `/home/soliday/oag/apps/src/elegantTestSet/`.

## Phase 16: Collective Effects Coverage

Goal: broaden the useful `WAKE`, `TRWAKE`, and `LSCDRIFT` CUDA coverage.

Tasks:

1. Add smoothed `WAKE` and `TRWAKE` modes.
2. Add multi-bunch filtering and start/end bunch behavior.
3. Add tilted `TRWAKE`.
4. Extend `LSCDRIFT` to smoothing, frequency filters, backtracking, kick mode, and `AUTO_LEFFECTIVE` where profiling supports it.
5. Reconcile `gpu_findFiducialTime` semantics before enabling RF wake paths that depend on it.
6. Revisit `RFCW`, especially because the LCLS0 production wrapper reports it as a dominant timed element.
7. Evaluate cuFFT only after the data layout and binning are resident enough for FFT cost to dominate transfer cost.

Gate:

- Each new mode has intermediate-array `GPU_VERIFY` checks for histograms and wake potentials, plus end-to-end SDDS comparisons.

Status:

- Started on May 7, 2026 with smoothed `WAKE` and `TRWAKE` coverage for the existing single-bunch, fixed/autoscaled binning subset.  The CUDA path now permits `SMOOTHING=1` for supported `WAKE` and `TRWAKE` elements.  It keeps the particle-heavy binning, convolution, and kick work on CUDA, while applying the existing CPU Savitzky-Golay smoothing routine to the small per-bin histogram between CUDA stages so the first Phase 16 slice preserves existing smoothing semantics exactly.
- Added focused fixed-bin regression cases `phase16_wake_smoothing` and `phase16_trwake_smoothing`.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files in both cases.  `GPU_VERIFY` also compared the smoothed longitudinal `Itime`, transverse `posItimeX/Y`, and wake-potential `Vtime` arrays against CPU intermediates with zero observed mismatch in the quick runs.
- Bounded 30,000-particle, 5-pass same-workload timings were a useful speed win despite the host-side smoothing bridge: smoothed `WAKE` took 1.08 seconds on CPU and 0.49 seconds on CUDA, while smoothed `TRWAKE` took 1.06 seconds on CPU and 0.49 seconds on CUDA.  Reports: `test/gpu_cuda/output/reports/phase16-wake-smoothing-30k5.md` and `test/gpu_cuda/output/reports/phase16-trwake-smoothing-30k5.md`.
- Added `WAKE,CHANGE_P0=1` support by letting the CUDA wake path call the existing GPU central-momentum matching helper after the wake kick and propagating the updated `P_central` back through the `track_through_wake` wrapper.  `GPU_VERIFY` now compares the CPU and GPU post-wake central momentum for this mode in addition to the coordinate and wake-array checks.
- Added the focused `phase16_wake_change_p0` regression case.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files, with zero observed `Itime` and `Vtime` mismatch in the quick `GPU_VERIFY` run.
- The bounded 30,000-particle, 5-pass `WAKE,CHANGE_P0=1` timing took 1.61 seconds on CPU and 0.63 seconds on CUDA, or about 2.56x faster.  Report: `test/gpu_cuda/output/reports/phase16-wake-change-p0-30k5.md`.
- Continued on May 7, 2026 with tilted single-bunch `TRWAKE` support when spin coordinates are inactive.  CUDA now rotates transverse coordinates and slopes into the tilted wake frame for histogram drive terms and wake kicks, then rotates the updated coordinates back to the global frame using the same CPU-style trigonometric special cases used by the existing GPU bend code.
- Added the focused `phase16_trwake_tilt` regression case.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  `GPU_VERIFY` also compared the tilted `posItimeX/Y` and `VtimeX/Y` arrays against CPU intermediates with zero observed wake-array mismatch in the quick run.
- The bounded 30,000-particle, 5-pass tilted `TRWAKE` timing took 1.11 seconds on CPU and 0.50 seconds on CUDA, or about 2.22x faster.  Report: `test/gpu_cuda/output/reports/phase16-trwake-tilt-30k5.md`.
- Continued with `LSCDRIFT` smoothing, high-frequency cutoff filtering, and explicit kick mode via `L=0,LEFFECTIVE>0`.  CUDA still uses the existing host FFT and impedance calculation for CPU-equivalent math, but now applies the CPU Savitzky-Golay histogram smoothing and high-frequency filter between the CUDA binning and CUDA kick/drift stages.
- Added focused `phase16_lsc_smoothing_filter` and `phase16_lsc_kick_mode` regression cases.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files in both cases.  `GPU_VERIFY` also compared the smoothed/filtered `Itime` and `Vtime` arrays against CPU intermediates with zero observed wake-array mismatch in the quick runs.
- Bounded 30,000-particle, 5-pass same-workload timings showed useful speedups for the new `LSCDRIFT` modes: smoothing plus high-frequency filtering took 0.89 seconds on CPU and 0.42 seconds on CUDA, or about 2.12x faster; explicit kick mode took 0.86 seconds on CPU and 0.43 seconds on CUDA, or about 2.00x faster.  Reports: `test/gpu_cuda/output/reports/phase16-lsc-smoothing-filter-30k5.md` and `test/gpu_cuda/output/reports/phase16-lsc-kick-mode-30k5.md`.
- Added `LSCDRIFT AUTO_LEFFECTIVE` support for cases where elegant resolves the effective length from the preceding element before entering the CUDA branch.  The support guard now allows unresolved `AUTO_LEFFECTIVE=1` during eligibility checking, while still requiring a positive effective length by the time the GPU path executes.
- Added the focused `phase16_lsc_auto_leffective` regression case.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  `GPU_VERIFY` compared the `Itime` and `Vtime` arrays against CPU intermediates with zero observed wake-array mismatch in the quick run.
- The bounded 30,000-particle, 5-pass `AUTO_LEFFECTIVE` timing took 0.98 seconds on CPU and 0.49 seconds on CUDA, or about 2.00x faster.  Report: `test/gpu_cuda/output/reports/phase16-lsc-auto-leffective-30k5.md`.
- Added conservative `LSCDRIFT` backtracking support.  The CUDA support guard now accepts negative internal `LSCDRIFT` lengths only when elegant has set the element's `backtrack` flag, and the CUDA path uses the existing backtrack sign convention in the voltage and kick/drift stages.
- Added the focused `phase16_lsc_backtrack` regression case.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  `GPU_VERIFY` compared the backtracked `Itime` and `Vtime` arrays against CPU intermediates with zero observed wake-array mismatch in the quick run.
- The bounded 30,000-particle, 5-pass backtracking timing took 0.87 seconds on CPU and 0.47 seconds on CUDA, or about 1.85x faster.  Report: `test/gpu_cuda/output/reports/phase16-lsc-backtrack-30k5.md`.
- Added `LSCDRIFT` low-frequency cutoff support for the ordered cutoff subset.  Enabling this also required fixing a legacy CPU `LSCDRIFT` high-pass loop that wrote before the packed FFT buffer at the DC bin; the fixed CPU code now zeros DC explicitly and clamps the cutoff ramp to the valid frequency range in both `track_through_lscdrift` and `addLSCKick`.
- Added the focused `phase16_lsc_low_frequency_filter` regression case with both low- and high-frequency filters.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  `GPU_VERIFY` compared the filtered `Itime` and `Vtime` arrays against CPU intermediates with zero observed wake-array mismatch in the quick run.
- The bounded 30,000-particle, 5-pass low-frequency-filter timing took 0.89 seconds on CPU and 0.44 seconds on CUDA, or about 2.02x faster.  Report: `test/gpu_cuda/output/reports/phase16-lsc-low-frequency-filter-30k5.md`.
- Added a narrow `gpu_findFiducialTime` implementation for `LIGHT` mode and serial/local full-beam `TMEAN` with `sOffset=0`, reusing the existing CUDA time-coordinate reduction.  This is enough to keep `modulate_elements` resident in serial `gpu-elegant` when the beam is already on the device, but it deliberately does not enable RFCA/RFCW fiducialization, fiducial-bunch filtering, `FIRST`, `PMAX`, nonzero `sOffset`, or distributed Pelegant GPU reductions.
- Added the focused `phase16_fiducial_modulate` regression case.  Serial CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  The final quick timing was 0.36 seconds on CPU and 0.25 seconds on CUDA for the 3,000-particle, 5-pass smoke.  The normal CUDA log showed 240 matrix kernels, 250 reductions, and no mid-run CPU synchronization.
- Pelegant distributed-particle modulation now forces a CPU fiducial-time fallback with an explicit synchronization instead of trying the unfinished MPI GPU reduction branch.  A 2-rank CPU/GPU Pelegant smoke for `phase16_fiducial_modulate` matched all 4 common SDDS files at `1e-11`; the CUDA log showed the expected `findFiducialTime MPI fallback` synchronization events.
- Added conservative single-effective-bucket `BUNCHED_BEAM_MODE=1` support for `WAKE` and `TRWAKE`.  The CUDA wrappers now preflight the bunch-index range on the device and keep the existing CUDA wake path only when all particles are in one effective bucket, `START_BUNCH<=0`, and the run is serial/local.  Unsupported multi-bunch layouts that select nonempty buckets and distributed Pelegant layouts force an explicit CPU fallback before the wake is applied.
- Added the focused `phase16_bunched_wake_single` regression case with fixed-bin `WAKE` and autoscaled `TRWAKE`.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files.  The `GPU_VERIFY` run compared the wake histograms and wake-potential arrays for 4 `WAKE` and 4 `TRWAKE` applications, and the normal CUDA log reported `wakes=8` with no bunched-wake CPU fallback.  The CUDA Pelegant build also compiles with the new MPI fallback guard.
- Restored CUDA `TRWAKE` fixed-bin centering to match the existing CPU path for single-bucket fixed-bin `TRWAKE`; the older `wake_trwake_fixed_bins` case now passes both normal CUDA comparison and `GPU_VERIFY` after this correction, preserving the known fixed-window warnings.
- On the bounded 30,000-particle timing gate, `phase16_bunched_wake_single` used 268 CPU passes in 42.73 seconds and the same 268-pass CUDA workload took 12.12 seconds, or about 3.53x faster, with all 4 common SDDS files matching at `1e-11`.
- Continued with filtered no-op support for single-effective-bucket bunched `WAKE` and `TRWAKE`.  When the CPU semantics would skip the only effective bucket, such as `START_BUNCH=1` on a one-bunch beam, the CUDA wrapper now returns without forcing particles back to the CPU.  Multi-bunch layouts that actually require per-bucket subsetting still fall back.
- Added the focused `phase16_bunched_wake_filter_skip` regression case.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 4 common SDDS files, and the CUDA logs showed `wakes=0` with no `WAKE bunched beam CPU fallback` or `TRWAKE bunched beam CPU fallback`.  The 30,000-particle, 40-pass timing gate took 6.30 seconds on CPU and 1.55 seconds on CUDA, or about 4.06x faster, with all 4 common SDDS files matching at `1e-11`.
- Extended the no-op filter path to serial/local multi-bucket bunched `WAKE` and `TRWAKE` when `START_BUNCH` is beyond the final effective bucket, so CPU semantics would skip every bucket.  The helper still falls back for multi-bucket cases that select any nonempty bucket or need per-bucket subsetting.
- Added the focused `phase16_bunched_wake_multibucket_skip` regression case.  It creates three effective buckets with `&sdds_beam,use_bunched_mode=1,n_duplicates=2` and sets `START_BUNCH=3`, so both `WAKE` and `TRWAKE` are all-skipped no-ops.  Normal CUDA and `GPU_VERIFY` builds matched CPU output at `1e-11` for all 6 common SDDS files, and the CUDA logs showed `wakes=0` with no bunched-wake CPU fallback.  The one-minute CPU timing gate used 10,000 seed particles duplicated to 30,000 tracked particles and 382 passes: CPU took 58.62 seconds, CUDA took the same workload in 11.88 seconds, or about 4.93x faster.  Report: `test/gpu_cuda/output/reports/phase16-bunched-wake-multibucket-skip-30k382.md`.
- Phase 16 is wrapped for the current CUDA scope as of May 7, 2026.  The implemented slices cover the high-value `WAKE`, `TRWAKE`, `LSCDRIFT`, and narrow fiducial-time items that had bounded correctness and timing evidence.  The remaining collective-effects items below are deferred follow-ups rather than Phase 16 blockers.  The next active CUDA work should move to Phase 17.

Deferred Phase 16 follow-ups:

- General multi-bunch `WAKE`/`TRWAKE` filtering, including positive `START_BUNCH` cases that select a later nonempty bucket, `END_BUNCH` filtering beyond the single effective bucket, per-bucket subsetting, and distributed Pelegant reductions.
- `RFCW` and RFCA/RFCW fiducialization paths.
- cuFFT-backed collective effects after the remaining host-side binning/FFT bridges are no longer the dominant cost.

## Phase 17: Magnet Coverage Expansion

Goal: expand from deterministic magnet slices toward production magnet usage.

Tasks:

1. Profile production magnet-heavy cases to decide which excluded features matter most.
2. Add support selectively for:
   - magnet misalignments
   - selected aperture hooks
   - slice-by-slice tracking
   - `CSBEND` reference/FSE correction
   - Hwang/Lindberg/curved `CSBEND` fringe models
3. Keep stochastic radiation/ISR and spin tracking separate from deterministic magnet work.
4. Add stochastic distribution-test tooling before enabling stochastic CUDA magnet paths.
5. Re-run `GPU_SUPPORT` metadata review whenever a new element becomes automatically eligible.

Gate:

- New magnet modes pass deterministic comparisons first; stochastic modes require distribution-level validation over multiple fixed seeds.

Initial implementation:

- Started Phase 17 on May 7, 2026 with a narrow deterministic misalignment slice for original-mode `KQUAD`, `KSEXT`, and `KOCT` tracking.  The CUDA path now supports nonzero `DX`, `DY`, `DZ`, and `TILT` for these multipoles when `MALIGN_METHOD=0`, `PITCH=0`, `YAW=0`, radiation/ISR are inactive, spin tracking is inactive, no extra multipole/fringe/radial/aperture hooks are active, and the existing checked magnet fallback guard accepts the element.
- Continued Phase 17 with the same original-mode `DX`, `DY`, `DZ`, and `TILT` support for `DQCOR`, while preserving CPU fallback for pitch/yaw, nonzero `MALIGN_METHOD`, radiation/ISR, spin, and extra systematic/random multipole files.
- Added simple original-mode `CSBEND` `DX`, `DY`, `DZ`, `TILT`, and `ETILT` support for the deterministic non-CSR CSBEND path, while preserving CPU fallback for `EPITCH`, `EYAW`, nonzero `MALIGN_METHOD`, radiation/ISR, spin, aperture hooks, reference/FSE correction, slice-by-slice tracking, and advanced fringe/curved models.
- The CUDA kernels apply CPU-equivalent entrance and exit coordinate transforms around the existing multipole and CSBEND integrators.  Pitch/yaw, nonzero `MALIGN_METHOD`, aperture hooks, radiation/stochastic effects, spin tracking, and more advanced misalignment modes still fall back to CPU.
- Added `test/gpu_cuda/cases/phase17_multipole_misalignment`, a bounded deterministic regression that exercises misaligned `KQUAD`, `KSEXT`, and `KOCT` elements without stochastic physics.
- Added `test/gpu_cuda/cases/phase17_dqcor_misalignment`, a bounded deterministic regression that exercises misaligned `DQCOR` elements with normal/skew quadrupole terms and small steering kicks.
- Added `test/gpu_cuda/cases/phase17_csbend_misalignment`, a bounded deterministic regression that exercises non-CSR `CSBEND` `DX`, `DY`, `DZ`, `TILT`, and `ETILT` with first-order edge focusing and no stochastic physics.
- Correctness evidence: CPU quick, normal CUDA quick, and `GPU_VERIFY` quick runs matched all 4 common SDDS files at `1e-11` for the Phase 17 misalignment cases.  The verify logs reported checked multipole and CSBEND agreement down to roundoff scale; the CSBEND verify smoke showed checked `track_through_csbend` max absolute differences no larger than `1.421e-14`.
- Timing evidence: the one-minute CPU baseline used 30,000 particles and 23 passes, taking 59.97 seconds on the CPU.  The same workload on `gpu-elegant` took 7.23 seconds on the local RTX 3060, about an 8.29x throughput speedup, and all 4 common SDDS files matched at `1e-11`.  The report is `test/gpu_cuda/output/reports/phase17-multipole-misalignment-30k23.md`.
- DQCOR timing evidence: the one-minute CPU baseline used 30,000 particles and 12 passes, taking 58.59 seconds on the CPU.  The same workload on `gpu-elegant` took 4.46 seconds on the local RTX 3060, about a 13.14x throughput speedup, and all 4 common SDDS files matched at `1e-11`.  The report is `test/gpu_cuda/output/reports/phase17-dqcor-misalignment-30k12.md`.
- CSBEND misalignment timing evidence: the one-minute CPU baseline used 30,000 particles and 11 passes, taking 57.36 seconds on the CPU.  The same workload on `gpu-elegant` took 4.53 seconds on the local RTX 3060, about a 12.66x throughput speedup, and all 4 common SDDS files matched at `1e-11`.  The report is `test/gpu_cuda/output/reports/phase17-csbend-misalignment-30k11.md`.
- Build evidence: `make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=/usr/local/cuda-12.4/bin/nvcc` and `make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc` both completed locally.
- Continued Phase 17 with `test/gpu_cuda/profile_magnet_features.py`, a static profiler for `.ele` and `.lte` files in `/home/soliday/oag/apps/src/elegantTestSet/`.  The profiler writes Markdown and TSV reports under `test/gpu_cuda/output/reports/` and is intended to guide the next CUDA magnet slice before adding more kernels.
- Production magnet-profile evidence from May 7, 2026: the profiler scanned 1,576 files, 477 cases, and 148,549 magnet definitions.  It classified 127,750 definitions as current simple CUDA candidates and 20,799 definitions as having deferred/blocking features.  The largest remaining blockers were radiation/ISR (10,568 definitions), advanced misalignment (5,760), unsupported multipole/corrector families such as `MULT`/`KICKER` (3,278), advanced bend families such as `CCBEND`/`LGBEND` (1,506), field-map/wiggler families (256), and `CSRCSBEND` collective bends (168).  The report is `test/gpu_cuda/output/reports/phase17-production-magnet-profile.md`.
- The profiling result supports keeping stochastic radiation/ISR, `CSRCSBEND`, advanced bend/fringe models, and field maps as separately measured follow-ups rather than expanding the deterministic Phase 17 CUDA path casually.  It also shows that additional deterministic production wrappers should come from the simple-candidate list, especially high-count cases such as `latticeErrors6`, `maxamp4`, and `errorSampling*`, before widening support beyond original-mode misalignments.
- Phase 17 is wrapped for the current CUDA scope as of May 7, 2026.  The implemented slices cover original-mode deterministic misalignments for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, and simple non-CSR `CSBEND`, with correctness, timing, build, and production-profile evidence recorded above.  The remaining magnet items below are deferred follow-ups rather than Phase 17 blockers.  The next active CUDA work should move to Phase 18.

Deferred Phase 17 follow-ups:

- Extend misalignment coverage beyond original-mode `KQUAD`/`KSEXT`/`KOCT`/`DQCOR`/simple `CSBEND`, starting with the production cases that profiling shows are actually common.
- Keep the default no-loss-output stable compaction subset for supported multipole, non-CSR `CSBEND`, and `KICKMAP`/`UKICKMAP` losses; defer resident `.los` and global loss-coordinate row semantics until they have a separate design and validation gate.
- Evaluate remaining `CSBEND` misalignment modes, reference/FSE correction, selected aperture hooks, slice-by-slice tracking, and the Hwang/Lindberg/curved `CSBEND` fringe models as separate measured slices.  The static production profile suggests reference/FSE correction and advanced `CSBEND` fringe models are present but much less common than radiation/ISR and advanced misalignment in the current test set.
- Evaluate `MULT`/`KICKER` and advanced bend families separately from `KQUAD`/`KSEXT`/`KOCT`/`DQCOR`/simple `CSBEND`; the production profile shows they are common enough to track but not part of the current deterministic CUDA kernels.
- Add broader deterministic production wrappers from the simple-candidate profile list, such as `maxamp4` and `errorSampling*`, before broadening automatic magnet eligibility beyond the current validated loss-compaction subset.
- Re-run `GPU_SUPPORT` metadata and dictionary review whenever a new magnet element or mode becomes automatically CUDA-eligible.
- Keep radiation/ISR and spin-tracking magnet paths deferred until stochastic/distribution-level validation tooling exists.

## Phase 18: SCMULT, Poisson, Field Maps, And Wigglers

Goal: choose the next expensive physics path based on production profiling.

Tasks:

1. Add broader production profiling cases for SCMULT, field maps, wigglers, and ion effects.
2. Extend SCMULT only where measured benefit exists:
   - sliced linear
   - nonlinear
   - multi-bunch
3. Evaluate cuFFT-backed Poisson solve work in `poisson.cc`.
4. Compare field-solver outputs at both field-grid and particle-coordinate levels.
5. Keep any automatic SCMULT use limited to the current guarded serial linear single-bucket window until production profiling supports broader enablement.

Gate:

- A production-like case shows meaningful speedup, not just an isolated synthetic win.

Initial implementation:

- Started Phase 18 on May 7, 2026 with `test/gpu_cuda/profile_phase18_features.py`, a static profiler for `.ele` and `.lte` files in `/home/soliday/oag/apps/src/elegantTestSet/`.  The profiler scans for `&insert_sceffects`, explicit `SCMULT`, field-map families (`UKICKMAP`, `KICKMAP`, `BMAPXY`, `BMXYZ`, `BGGEXP`, `BRAT`, and related map elements), `CWIGGLER`/`WIGGLER`, direct `IONEFFECTS`, adjacent `BEAMBEAM`, and Poisson-related settings.
- Production Phase 18 profile evidence from May 7, 2026: the profiler scanned 1,576 files and found 276 Phase 18 feature occurrences in 74 cases.  Static feature counts were 227 field-map occurrences, 44 wiggler occurrences, 4 SCMULT insertion occurrences, and 1 adjacent collective occurrence.  The dominant field-map type was `UKICKMAP` with 184 occurrences; the dominant wiggler type was `CWIGGLER` with 34 occurrences.  The report is `test/gpu_cuda/output/reports/phase18-production-feature-profile.md`.
- SCMULT profile evidence: `scRing2` is the only production test-set case matching the current linear/unsliced static shape.  `scRing1` and `scRing3` are nonlinear unsliced cases, and `scRing-nonlinearBunched` is nonlinear with bunched-mode input.  This supports measuring `scRing2` before widening the guarded SCMULT path, while keeping nonlinear and bunched SCMULT deferred.
- Field-map and wiggler profile evidence: high-count field-map candidates include `latticeErrors6`, `brat*`, `errorSampling*`, `ccbendSoftFringe*`, `lgbend2`, `orbitCor12`, and `tuneCorrection1`.  Smaller comparison-oriented candidates include `bmapxy1`, `bmxyz1`, `boffaxe1`, `cwiggler10`, and `wig3`, because they give field-grid, particle-output, external-field, or radiation-integral coverage without immediately starting from the largest lattices.
- Ion/Poisson profile evidence: the current scan found no direct `IONEFFECTS` occurrences and no Poisson-grid settings in the production test set.  Before CUDA work in `poisson.cc`, add a small production-like ion/Poisson wrapper with both field-grid and particle-coordinate comparison outputs.
- Added Phase 18 profile rows to `test/gpu_cuda/production_cases/metadata.tsv` for `scRing2`, `scRing1`, `scRing-nonlinearBunched`, `latticeErrors6`, `bmapxy1`, `bmxyz1`, `boffaxe1`, `cwiggler10`, `wig3`, `uKickMap4`, and `uKickMap5`.
- Added `test/gpu_cuda/production_cases/scRing2`, a production wrapper for the current linear SCMULT CUDA path.  The wrapper uses the source fixed SDDS beam from `/home/soliday/oag/apps/src/elegantTestSet/scRing2`, so bounded benchmarks scale by pass count rather than particle count.
- Added `scRing2` benchmark defaults to `test/gpu_cuda/run_benchmarks.sh`: 8 passes for quick runs and 64 passes for baseline sampling.  The CPU one-minute baseline autoscaled from the 64-pass sample to a 601-pass run, keeping Phase 18 timing bounded by the existing benchmark policy.
- SCMULT correctness evidence from May 7, 2026: CPU quick, GPU quick, and GPU-VERIFY quick runs for `scRing2` matched all 5 common SDDS files (`.cen`, `.fin`, `.out`, `.sig`, `.twi`) at tolerance `1e-11`.  The GPU-VERIFY run also reported `trackThroughSCMULT linear resident` agreement with maximum absolute error `1.355e-20` and maximum relative error `7.983e-16`.
- SCMULT timing evidence from May 7, 2026 before the thin-RFCA update: the 601-pass CPU baseline took 58.53s, and the same 601-pass `gpu-elegant` workload with `ELEGANT_GPU_ENABLE_SCMULT=1`, `ELEGANT_GPU_MODE=required`, and `ELEGANT_GPU_MIN_PARTICLES=1` took 23.08s, for a 2.54x throughput speedup on the RTX 3060 host.  The report is `test/gpu_cuda/output/reports/phase18-scmult-scRing2-601pass.md`.
- The first `scRing2` CUDA log showed SCMULT was productive but sync-heavy: 601 `RFCA` CPU-element synchronizations and 601 `WATCH` read-only output synchronizations in the 601-pass run.  This motivated the no-WATCH wrapper and the thin-RFCA follow-up below.
- Added `test/gpu_cuda/production_cases/scRing2_no_watch`, a profiling variant that disables the production `WATCH` element with `&alter_elements type=WATCH,item=DISABLE,value=1` while preserving the common physics outputs.  This isolates diagnostic-output synchronization cost without editing the source production lattice.
- `scRing2_no_watch` correctness and timing evidence from May 7, 2026 before the thin-RFCA update: the 601-pass CPU and GPU runs matched all 5 common SDDS files at tolerance `1e-11`; CPU took 58.73s and GPU took 22.69s, for a 2.59x speedup.  Compared with the standard GPU `scRing2` 601-pass result, disabling `WATCH` removed 601 read-only/output sync requests but improved wall time only from 23.08s to 22.69s, about 1.7%.  The remaining 601 mutable syncs were all `RFCA` CPU-element synchronizations.  The report is `test/gpu_cuda/output/reports/phase18-scmult-scRing2-nowatch-601pass.md`.
- Added a narrow thin-RFCA CUDA path in `src/gpu/gpu_stub.c` and `src/gpu/gpu_cuda_runtime.cu`.  It supports zero-length `RFCA` elements after CPU fiducialization has established the phase reference, with no wakes, no `CHANGE_P0`, no `CHANGE_T`, no cavity `Q`, no body/end focusing, no offsets, no linearization/lock-phase mode, and no backtracking.  The first pass still falls back to CPU for fiducialization setup; subsequent passes remain resident and apply the RF kick on the GPU before resident invalid-particle removal.
- Thin-RFCA correctness evidence from May 7, 2026: the `GPU_VERIFY` quick run for `scRing2_no_watch` reported `simple_rf_cavity` agreement with maximum absolute and relative error `0.000e+00`, and the normal 601-pass CPU/GPU runs matched common SDDS outputs at tolerance `1e-11`.  The standard `scRing2` 601-pass comparison, including the `.w1` WATCH file, matched all 6 common files at tolerance `1e-11`.
- Thin-RFCA timing evidence from May 7, 2026: standard `scRing2` now takes 22.86s on `gpu-elegant` for the 601-pass workload, a 2.56x throughput speedup versus the 58.53s CPU baseline and a small improvement over the previous 23.08s GPU run.  The CUDA log no longer has `RFCA` CPU-element synchronizations; the remaining standard-wrapper syncs are 601 WATCH read-only/output requests, plus one RFCA fiducialization setup request and deallocation.  The no-WATCH thin-RFCA run took 22.69s, matched CPU at `1e-11`, and reduced normal CUDA sync accounting to the one RFCA setup request plus deallocation.  Reports are `test/gpu_cuda/output/reports/phase18-scmult-scRing2-rfca-601pass.md` and `test/gpu_cuda/output/reports/phase18-scmult-scRing2-nowatch-rfca-601pass.md`.
- Added `test/gpu_cuda/cases/phase18_rfca_thin`, a focused synthetic regression gate for the narrow thin-RFCA path.  The case uses 8 zero-length RFCA occurrences per pass, so the first pass exercises CPU fiducialization setup for each occurrence and later passes exercise resident CUDA RF kicks.
- `phase18_rfca_thin` correctness and timing evidence from May 7, 2026: CPU quick, normal CUDA quick, and `GPU_VERIFY` quick runs matched all 4 common SDDS files at tolerance `1e-11`.  The `GPU_VERIFY` run reported repeated `simple_rf_cavity` passes with maximum absolute error `0.000e+00` except one final check at `1.549e-16`, with no mismatches.  The one-minute CPU baseline autoscaled to 30,000 particles and 224 passes, taking 60.16s; the same GPU workload took 11.94s, a 5.04x throughput speedup.  CUDA sync accounting showed 8 RFCA fiducialization setup requests, one for each first-pass RFCA occurrence, plus deallocation; the remaining 1,784 RFCA traversals stayed resident.  The report is `test/gpu_cuda/output/reports/phase18-rfca-thin.md`.
- Phase 18 sync conclusion: RFCA/WATCH synchronization is now understood for `scRing2`.  Removing per-pass RFCA sync and disabling WATCH output dramatically reduces sync counts but does not materially move wall time beyond the existing 2.56x-2.59x speedup, so remaining effort should shift to broader SCMULT eligibility decisions or field-map/wiggler prototypes rather than more `scRing2` synchronization work.
- Added bounded Phase 18 field-map and wiggler wrappers for `bmapxy1`, `bmxyz1`, `boffaxe1`, and `cwiggler10`.  These wrappers reference the production map/harmonic data by absolute path and are intentionally excluded from the default production smoke set until field-map CUDA work is chosen.
- Field-map/wiggler correctness evidence from May 7, 2026: CPU and `gpu-elegant` quick runs matched 19 common files at tolerance `1e-11`, including `BMXYZ` particle-output diagnostics (`.pout`), `BOFFAXE` particle and field-grid diagnostics (`.pout`, `.field`), `CWIGGLER` field output (`.cwigOut`), and `WATCH` output (`.w1`).  The report is `test/gpu_cuda/output/reports/phase18-fieldmap-wiggler-quick.md`.
- Field-map/wiggler timing evidence from May 7, 2026: the four-case quick set remains bounded on CPU (`bmapxy1` 0.58s, `bmxyz1` 4.19s, `boffaxe1` 1.66s, `cwiggler10` 2.82s).  The GPU binary is slightly slower for these wrappers (`bmapxy1` 0.74s, `bmxyz1` 4.41s, `boffaxe1` 2.03s, `cwiggler10` 2.97s), as expected for CPU-fallback-heavy field-map and wiggler paths.
- Bounded-test adjustment: the `bmxyz1` wrapper omits final-properties output because that forced tracking-based matrix determination through the 1,923,201-point field map, and the `boffaxe1` wrapper omits source-style matrix output for the same timing-budget reason.  Heavier matrix-specific field-map checks should be separate follow-up wrappers, not part of quick Phase 18 comparison.
- Added a narrow deterministic `KICKMAP`/`UKICKMAP` CUDA interpolation/kick prototype in `src/gpu/gpu_stub.c` and `src/gpu/gpu_cuda_runtime.cu`, with `src/gpu/gpu_kickmap.h` wiring through `do_tracking.c`.  The supported subset is serial/local tracking with initialized map arrays, `N_KICKS>=1`, nonzero length, no radiation/ISR, no map-element `TILT`/`DX`/`DY`/`DZ`/`YAW`, and CPU fallback if the checked CUDA kernel detects any particle leaving the map.  `GPU_VERIFY` replays the same no-loss step on the CPU shadow and compares immediately.
- Added `test/gpu_cuda/production_cases/uKickMap1`, a bounded generated-bunch wrapper around the production `uKickMap1/kickmap.sdds` file.  The wrapper uses `FIELD_FACTOR=0.01` so the long timing gate stays within the map for most particles and continues exercising 2,000 `UKICKMAP` traversals rather than terminating early from wholesale particle loss.
- Added a one-entry `KICKMAP`/`UKICKMAP` device map cache keyed by the host `xpFactor`/`ypFactor` arrays and point count.  The CUDA kernel now consumes device-resident map arrays instead of allocating, uploading, and freeing the map arrays on every traversal; the cache is released with `gpuBaseDealloc`.
- `uKickMap1` correctness and timing evidence from May 7, 2026 after map caching: CPU quick, normal CUDA quick, and `GPU_VERIFY` quick runs matched all 4 common SDDS files at tolerance `1e-11`; `GPU_VERIFY` reported `trackUndulatorKickMap` agreement with maximum absolute and relative error `0.000e+00`.  The 30,000-particle, 2,000-pass timing workload matched all 4 common files at `1e-11` and ran in 13.58s on `gpu-elegant` versus 55.43s on CPU, a 4.08x throughput speedup.  The CUDA log recorded 2,000 magnet kernels and 337 particle-loss fallback synchronizations late in the run; handling map loss without CPU fallback remains deferred.  The report is `test/gpu_cuda/output/reports/phase18-ukickmap1.md`.
- Phase 18 is wrapped for the current CUDA scope as of May 7, 2026.  The implemented slices cover production profiling, the guarded linear `SCMULT` production case, the narrow zero-length thin-`RFCA` path, bounded field-map/wiggler comparison wrappers, and the cached deterministic `KICKMAP`/`UKICKMAP` prototype, with correctness and timing evidence recorded above.  The remaining Phase 18 items below are deferred follow-ups rather than Phase 18 blockers.  The next active CUDA work should move to Phase 19.

Deferred Phase 18 follow-ups:

- Keep linear SCMULT automatic only inside the guarded serial single-bucket window.  The current production evidence is correct and 2.56x-2.59x faster, and the thin-RFCA subpath now has a focused regression gate, but broadening SCMULT enablement should still wait for at least one more production-shaped validation case.
- Keep additional RFCA/RFCW work deferred beyond the narrow zero-length thin-RFCA path.  Nonzero length, wakes, `CHANGE_P0`, `CHANGE_T`, cavity `Q`, body/end focusing, lock/linearized phase modes, offsets, backtracking, and RFCW still require CPU fallback.
- Add heavier matrix-specific field-map comparison wrappers for `bmxyz1` and `boffaxe1` only after they can be kept within the one-minute CPU timing policy.
- Profile high-count `UKICKMAP` cases such as `latticeErrors6` only after the bounded `uKickMap1` CUDA prototype has a better loss strategy, or after the high-count case is shown to stay inside the checked no-loss subset.
- Keep nonlinear, sliced, and bunched SCMULT deferred until the linear production candidate has measured benefit and correctness.
- Keep radiation/ISR wiggler and `KICKMAP`/`UKICKMAP` paths deferred until distribution-level stochastic validation exists.
- Replace `KICKMAP`/`UKICKMAP` CPU fallback on map loss with a stable resident compaction or a prechecked no-loss eligibility guard.
- Add a small ion/Poisson production-like case before evaluating cuFFT-backed `poisson.cc` work.

## Phase 19: Pelegant And Multi-GPU Validation

Goal: validate the Pelegant CUDA path as far as possible on the current one-GPU host, while keeping true multi-GPU work deferred until matching hardware is available.

Tasks:

1. Run larger Pelegant CPU/GPU timing cases on the current single-GPU host.
2. Validate deterministic load balancing with fixed rank counts.
3. On multi-GPU hardware:
   - test one GPU per worker rank
   - test `CUDA_VISIBLE_DEVICES` mapping
   - test multiple worker ranks sharing fewer GPUs
   - compare multi-GPU speedup against CPU-only Pelegant
4. Prototype MPI-aware GPU reductions only after staged CPU reductions are fully profiled.
5. Prototype GPU-aware MPI particle exchange only on hardware and MPI builds that support it.
6. Keep host-staged scatter/gather as the default fallback until GPU-aware MPI is proven.

Gate:

- Multi-GPU timing improves over CPU Pelegant and serial `gpu-elegant` for large enough workloads.

Initial implementation:

- Started Phase 19 on May 7, 2026 with an explicit one-GPU scope.  No true multi-GPU logic was added; the implementation is limited to fixed-rank Pelegant validation and conservative fallback guards that can be verified on the available hardware.
- Added `test/gpu_cuda/pelegant_single_gpu.sh`, which runs paired CPU/GPU Pelegant cases through `run_benchmarks.sh`, keeps MPI rank count, particles, passes, and labels explicit, and compares SDDS output at the selected tolerance.
- Added `test/gpu_cuda/cases/phase19_matrix_load_balance/benchmark.ele`, a matrix-lattice diagnostic with `load_balancing_on=1`.  This keeps the ordinary `matrix` case at `load_balancing_on=0` for GPU-active timing while giving Phase 19 a focused dynamic-load-balancing probe.
- Rebuilt CPU Pelegant with `make -C src -f Makefile.mpi -j4` and CUDA Pelegant with `make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4`.
- GPU-active fixed-rank evidence after the fallback guard: `./test/gpu_cuda/pelegant_single_gpu.sh --ranks 2 --particles 20000 --passes 20 --label-prefix phase19-pelegant-matrix-after-guard` matched all 4 common SDDS files at `1e-11`.  CPU Pelegant took 11.06 seconds, `gpu-Pelegant` took 6.41 seconds, for about 1.73x speedup.  The CUDA log reported `matrix=1280` on worker rank 1.  Report: `test/gpu_cuda/output/reports/phase19-pelegant-single-gpu-matrix-after-guard.md`.
- Dynamic load-balancing finding: the unguarded 3-rank, one-GPU `phase19_matrix_load_balance` diagnostic reached Pelegant redistribution (`We need redistribute the particles`) and then produced a SIGSEGV/timeout in the CUDA run.  This is treated as a Phase 19 blocker for GPU-resident dynamic redistribution rather than something to broaden without hardware coverage.
- Added a conservative CUDA Pelegant guard for `load_balancing_on=1`: in `ELEGANT_GPU_MODE=auto`, CUDA Pelegant disables CUDA for that run and uses CPU fallback; in `ELEGANT_GPU_MODE=required`, it fails fast instead of attempting unvalidated redistribution.  The fallback run `ELEGANT_GPU_MODE=auto ./test/gpu_cuda/pelegant_single_gpu.sh --case phase19_matrix_load_balance --ranks 3 --particles 12000 --passes 10 --label-prefix phase19-pelegant-load-balance-fallback` matched all 4 common SDDS files at `1e-11`; CPU took 2.16 seconds and the guarded fallback run took 2.41 seconds.  Report: `test/gpu_cuda/output/reports/phase19-pelegant-load-balance-fallback.md`.
- The required-mode guard was checked with `phase19-load-balance-required-expected-fail`: a tiny 3-rank `phase19_matrix_load_balance` run exited in 0.63 seconds with status 1 and printed `Pelegant load_balancing_on=1 CUDA redistribution is deferred`, confirming it fails quickly instead of reaching the earlier redistribution crash.
- Phase 19 is wrapped for the current CUDA scope as of May 7, 2026.  The implemented slice covers fixed-rank single-node Pelegant CUDA validation on the available one-GPU host, plus a conservative guard for dynamic Pelegant load balancing.  The remaining Pelegant and multi-GPU items below are deferred follow-ups rather than Phase 19 blockers.  The next active CUDA work should move to Phase 20.

Deferred Phase 19 follow-ups:

- True multi-GPU timing and mapping checks, including one GPU per worker rank and multiple workers sharing fewer GPUs.
- GPU-resident dynamic Pelegant redistribution for `load_balancing_on=1`.
- MPI-aware GPU reductions and GPU-aware MPI particle exchange.
- Multi-node timing and correctness checks against CPU Pelegant and serial `gpu-elegant`.

## Phase 20: CI And Release Infrastructure

Goal: make CUDA support maintainable over time.

Tasks:

1. Wire `test/gpu_cuda/ci_smoke.sh` into the repository's chosen CI system.
2. Split CI into:
   - CPU-only build/test
   - CUDA compile-only build
   - CUDA verify build
   - GPU hardware correctness smoke
   - bounded timing/report job
   - Pelegant MPI smoke
3. Archive benchmark reports as CI artifacts.
4. Add a "known fallback" report so new CPU synchronizations are visible in review.
5. Add a release-note template with speedups, hardware, tolerances, known fallbacks, and deferred work.

Gate:

- CI can catch CPU build regressions, CUDA compile regressions, and at least one GPU runtime correctness regression before release.

Initial implementation:

- Started Phase 20 on May 7, 2026 by adding `test/gpu_cuda/ci_release.sh`, a local CI/release driver that composes the existing `ci_smoke.sh`, `release_invariance.sh`, `report_benchmarks.py`, and fallback-summary tooling into split stages.  It supports CPU build, CUDA compile, GPU smoke, timing, Pelegant smoke, release-invariance, fallback-report, and `--all-available` modes, with per-stage logs and a single artifact directory.
- Extended `test/gpu_cuda/ci_smoke.sh` with `--output DIR` so hosted CI jobs can put benchmark output under a job-specific artifact root instead of always writing into `test/gpu_cuda/output`.
- Added `test/gpu_cuda/summarize_fallbacks.py`, which scans `elegant.stderr` files by output-root, label prefix, or explicit run directory and generates Markdown plus optional TSV summaries of CPU synchronizations, fallback messages, short-GPU-island skips, and per-case sync accounting.
- Added `test/gpu_cuda/release_notes_template.md` with sections for build matrix, hardware/runtime, correctness, timing, known CPU fallbacks, `GPU_VERIFY` coverage, CPU invariance, deferred work, and release decision.
- Generated an initial fallback report from existing Phase 19 labels with `python3 test/gpu_cuda/summarize_fallbacks.py --output-root test/gpu_cuda/output --label-prefix phase19-pelegant --output test/gpu_cuda/output/reports/phase20-fallback-summary-phase19.md --tsv test/gpu_cuda/output/reports/phase20-fallback-summary-phase19.tsv --max-rows 25`.  It scanned 8 stderr files and showed `accumulate_beam_sums below CUDA reduction threshold` as the dominant Pelegant synchronization reason, plus the expected `Pelegant load_balancing_on=1 CUDA redistribution is deferred; using CPU fallback` message.
- Verified the CI/release driver stage wiring with a full dry-run over CPU build, CUDA compile, GPU smoke, timing, Pelegant smoke, release-invariance, and fallback-report stages using label `phase20-dry`.  Also dry-ran `ci_smoke.sh --quick --output /tmp/elegant-phase20-output` and `ci_release.sh --gpu-smoke --output-root /tmp/elegant-phase20-output` to confirm the output-root plumbing is visible in the generated commands.
- Ran a real lightweight artifact-generation pass with `./test/gpu_cuda/ci_release.sh --fallback-report --label-prefix phase19-pelegant --artifact-dir test/gpu_cuda/output/ci_artifacts/phase20-phase19-fallback-artifacts`.  The artifact directory contains stage logs, copied Phase 19 benchmark manifests, fallback Markdown/TSV, an artifact manifest, and the release-notes template.
- Ran a real GPU/MPI runtime smoke through the Phase 20 driver with `./test/gpu_cuda/ci_release.sh --gpu-smoke --pelegant-smoke --fallback-report --case matrix --label-prefix phase20-runtime-smoke --artifact-dir test/gpu_cuda/output/ci_artifacts/phase20-runtime-smoke`.  Serial CPU/GPU `matrix` output matched all 4 common SDDS files at `1e-11`, `GPU_VERIFY` ran successfully, and 2-rank CPU/GPU Pelegant `matrix` output matched all 4 common SDDS files at `1e-11`.  The fallback report scanned 5 stderr files, found no fallback messages, and showed only expected synchronization reasons (`accumulate_beam_sums below CUDA reduction threshold` and `gpuBaseDealloc`).  Artifacts are under `test/gpu_cuda/output/ci_artifacts/phase20-runtime-smoke/`.
- Added `.github/workflows/gpu-cuda-ci.yml` as the first hosted-CI wiring.  Pull requests and pushes to `main` or `master` run the CPU build/fallback-report job on `ubuntu-latest`; manual `workflow_dispatch` can run the CUDA compile-only and `GPU_VERIFY` build in an NVIDIA CUDA 12.4 development container; manual GPU runtime, optional timing, and optional Pelegant smoke run on a self-hosted runner labeled `self-hosted`, `linux`, `x64`, and `cuda`.  Each job checks out `rtsoliday/SDDS` next to `elegant` and uploads the `ci_release.sh` artifact directory.

## Suggested Near-Term Order

1. Continue Phase 20: run `.github/workflows/gpu-cuda-ci.yml` in GitHub Actions, confirm the CPU and CUDA compile jobs pass from a fresh checkout, then register or relabel a self-hosted CUDA runner for the GPU runtime job and review the uploaded `test/gpu_cuda/output/ci_artifacts/<label>/` artifacts.
2. Deferred Phase 19 follow-up: resume true multi-GPU Pelegant validation only when hardware with more than one GPU is available.
3. Deferred Phase 18 follow-up: decide whether the now-verified linear `SCMULT` production path should expand beyond the guarded serial single-bucket automatic window, using `scRing2` plus at least one additional validation shape.
4. Deferred Phase 18 follow-up: improve the narrow `KICKMAP`/`UKICKMAP` prototype by removing the late-run CPU loss fallback, then decide whether to broaden into high-count `UKICKMAP` production cases or switch to a different field-map/wiggler family.

## Notes For Future Contributors

- Do not assume an old `GPU_SUPPORT` flag means the CUDA path is complete or safe.
- Do not enable a CUDA path by default just because it is correct; it also needs timing evidence and a threshold.
- Do not compare GPU and CPU output only at the final SDDS file when intermediate arrays can diverge and later cancel out.
- Keep CPU baseline timing targets bounded to about one minute unless a human explicitly chooses a longer production validation.
- Prefer adding focused production wrappers under `test/gpu_cuda/production_cases/` rather than copying large generated data into git.
