# Phased Plan For Optional CUDA GPU Support

## Goals

- Add optional CUDA acceleration without changing the default CPU-only build or results.
- Keep particle data resident on the GPU across consecutive GPU-capable elements, copying back only for CPU-only elements, diagnostics, output, MPI scatter/gather, or verification.
- Focus first on routines with large particle-parallel work and existing GPU hook points.
- Require correctness checks and timing evidence before enabling each GPU path by default.

## Current Codebase Starting Point

The source already has dormant GPU integration points:

- `HAVE_GPU` guards appear in tracking and element routines.
- `GPU_SUPPORT` flags already exist in `src/track_data.c` for several element types.
- `src/do_tracking.c` already calls GPU runtime concepts such as `gpuBaseInit`, `gpuBaseDealloc`, `setElementGpuData`, `getElementOnGpu`, and `forceParticlesToCpu`.
- Existing GPU hook calls are present for:
  - matrix tracking: `src/matrix.c`, `matr_element_tracking`, `ematrix_element_tracking`
  - apertures and particle validity: `src/limit_amplitudes.c`
  - centroids and beam sums: `src/compute_centroids.c`
  - multipoles: `src/multipole.c`
  - bends and CSR: `src/csbend.c`
  - longitudinal and transverse wakes: `src/wake.c`, `src/trwake.c`
  - LSC drift: `src/lsc.c`
  - RF cavity wake paths: `src/simple_rfca.c`
  - matter scattering: `src/matter.c`

The CUDA backend headers and implementations referenced by these hooks are not present in this checkout. These dormant hooks also appear to be out of date relative to the current CPU code, so they should be treated as signposts for likely integration points, not as a current or complete API contract. Some hooks, flags, data-flow assumptions, and verification paths may need substantial modification or replacement before they are safe to use. The recommended approach is to start by auditing each existing `gpu_*` call against the current CPU routine, preserve only the parts that still match current behavior, and implement a repo-owned CUDA backend around the verified interfaces.

## Candidate Priority

| Priority | Code paths | Why first | Main risks |
| --- | --- | --- | --- |
| 1 | `track_particles`, `matr_element_tracking`, `ematrix_element_tracking`, `exactDrift`, simple offsets/centering/energy matching | Ubiquitous, embarrassingly particle-parallel, low physics risk | Transfer overhead can dominate unless data stays resident |
| 2 | `limit_amplitudes`, rectangular/elliptical collimators, scraper, aperture data, invalid-particle removal | Many simple per-particle decisions, often run after many elements | Particle loss compaction must preserve `particleID` and accepted/lost data |
| 3 | `multipole_tracking2` for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`; non-CSR `CSBEND` | Expensive per-particle maps, commonly used in large tracking jobs | Radiation, aperture checks, misalignments, and edge cases increase branch divergence |
| 4 | `compute_centroids`, `accumulate_beam_sums`, slice/trajectory reductions | Reductions over large particle arrays and frequent diagnostics | Deterministic reductions and MPI interaction |
| 5 | `WAKE`, `TRWAKE`, `LSCDRIFT`, `RFCW` | Histograms, reductions, convolution/FFT style work can benefit strongly | Floating-point order, binning boundary behavior, and multi-bunch logic |
| 6 | `CSRCSBEND`, `CSRDRIFT` | High cost in realistic collective-effect simulations | Most numerically sensitive; CSR wake state must match CPU behavior closely |
| 7 | `SCMULT`/Poisson, field maps, wigglers, ion effects | Potentially high payoff after baseline backend is stable | Fewer existing hooks; needs more invasive integration |

## Correctness Strategy

Use three levels of verification.

1. Kernel/unit verification
   - Add small deterministic tests for every CUDA kernel using synthetic particle arrays.
   - Cover normal particles, lost particles, boundary coordinates, zero particles, one particle, and large particle counts.
   - Compare CPU and GPU outputs using absolute plus relative tolerances, not only absolute tolerance.

2. In-process `GPU_VERIFY`
   - Implement the existing `GPU_VERIFY` hooks so a GPU-capable element can run the GPU path, then run the CPU path from the same saved input state, then compare.
   - Compare all active particle properties, accepted/lost arrays, `P_central`, accumulated beam sums, and element-specific state such as CSR wake data.
   - Report max absolute error, max relative error, RMS error, coordinate and `particleID` of worst error, and whether loss classification changed.
   - Keep `GPU_VERIFY` separate from timing runs because it intentionally doubles work and adds transfers.

3. End-to-end SDDS regression tests
   - Run CPU and GPU builds from identical input files with fixed `random_number_seed`, `tracking_updates=0`, `show_element_timing=1`, and `load_balancing_on=0`.
   - Use distinct rootnames, for example `cpu/%s` and `gpu/%s`.
   - Compare generated SDDS outputs:
     - `*.out`, `*.fin`, `*.cen`, `*.sig`, `*.lost`, `*.acc`, and relevant wake/CSR/watch outputs.
     - Sort or label particle output by `particleID` before comparing when particle loss/compaction can change row order.
     - Use `sddsdiff -compareCommon=column -tolerance=<value> -rowlabel=particleID,nocomparison` for particle files where applicable.
   - For richer checks, add a helper that reads SDDS columns and applies `abs(a-b) <= abs_tol + rel_tol * max(scale, abs(a), abs(b))`.

Suggested deterministic tolerances:

- Matrix/drift/simple aperture paths: `abs_tol=1e-13`, `rel_tol=1e-12`.
- Multipoles and non-CSR bends: `abs_tol=1e-12`, `rel_tol=1e-10`.
- Wake/LSC/RF collective paths: `abs_tol=1e-10`, `rel_tol=1e-8`, tightened once reductions are stable.
- CSR paths: start with `abs_tol=1e-9`, `rel_tol=1e-7`, then tighten using representative regression cases.

For stochastic physics:

- First verify with stochastic effects disabled, for example ISR/scattering/random errors off where possible.
- Prefer counter-based per-particle RNG for CUDA so CPU/GPU can produce reproducible random streams independent of thread scheduling.
- When identical random streams are not practical, compare distribution-level quantities instead of per-particle coordinates: means, sigmas, emittances, loss counts, energy spread, bunch length, and relevant histograms.
- Require changes to be statistically consistent over multiple fixed seeds, not just one run.

Loss handling rules:

- Deterministic tests should have identical surviving `particleID` sets.
- If a loss decision differs, fail the test unless the particle is within a documented numerical tolerance of an aperture or boundary and aggregate beam statistics remain within tolerance.
- Always compare `nLeft`, `finalCharge`, and loss locations.

## Timing Strategy

Every phase must include timing before it is considered complete.

- Add or use a release build mode first. Current `src/Makefile` includes `-Og`, which is useful for debugging but can make CPU baselines misleading. Compare GPU against an optimized CPU build.
- Collect CPU timing using existing `run_setup.show_element_timing=1`.
- Add CUDA event timing around each GPU element path and include transfer time separately from kernel time.
- Size each timing case so the CPU-only run is aimed at about 1 minute. Adjust particle count, turns, or lattice repetitions to stay near that target, and split oversized cases rather than allowing a single timing test to run for an hour.
- Run at least 5 repeats after one warm-up run; report median and min/max.
- Sweep particle counts, for example `1e3`, `1e4`, `1e5`, `1e6`, because small bunches may be faster on CPU.
- Establish per-element GPU thresholds such as `min_particles_for_gpu` so small jobs stay on CPU automatically.
- Require either:
  - at least 10 percent end-to-end speedup for a representative input, or
  - at least 1.5x speedup for the targeted element with no more than 3 percent end-to-end slowdown.
- If a CUDA path is correct but not faster, keep it available only behind an explicit opt-in until later phases reduce transfer overhead.

## Phase 0: Baseline, Profiling, And Test Corpus

Deliverables:

- A repeatable benchmark suite under a new directory such as `test/gpu_cuda/`.
- CPU baseline timing reports and output files for each benchmark.
- A correctness comparator script for SDDS outputs.

Tasks:

1. Create benchmark cases covering:
   - matrix-only tracking with many turns and many particles
   - `KQUAD`/`KSEXT`/`KOCT` heavy lattices
   - non-CSR `CSBEND`
   - aperture and particle loss
   - `WAKE`/`TRWAKE`
   - `LSCDRIFT`
   - `CSRCSBEND`/`CSRDRIFT`
   - RF cavity wake paths
   - at least one user-provided production input
2. Add fixed seeds and deterministic settings to all benchmark inputs.
3. Capture current CPU output as the reference artifact.
4. Record baseline timing with `show_element_timing=1`.
5. Identify the top element types by wall time before each porting phase.

Gate:

- The benchmark suite must run cleanly on CPU and produce stable results over repeated runs.

Implementation status:

- Initial benchmark cases are in `test/gpu_cuda/cases/` for matrix tracking, `KQUAD`/`KSEXT`/`KOCT`, non-CSR `CSBEND`, aperture/loss handling, `WAKE`/`TRWAKE`, `LSCDRIFT`, `CSRCSBEND`/`CSRDRIFT`, and `RFCW`.
- `test/gpu_cuda/run_benchmarks.sh` runs quick smoke tests by default and has a `--baseline --target-seconds 60` mode for CPU baseline timing. The runner also has a per-run timeout guard so oversized tests do not accidentally run for an hour.
- `test/gpu_cuda/compare_sdds.py` compares SDDS output trees with `sddsdiff`, using `particleID` row labels when available.
- `test/gpu_cuda/production_cases/lcls0/` wraps `/home/soliday/oag/apps/src/elegantTestSet/LCLS0` as the initial real production input. It exercises a production LCLS lattice with real SDDS beam and wake files, including `WAKE`, `RFCW`, `CSRCSBEND`, `CSRDRIFT`, and collimator/aperture elements.
- The LCLS0 quick production run tracks about 20k sampled particles and currently takes about 8 seconds on the local CPU build. The dominant timed elements are `RFCW` and `CSRCSBEND`, which makes it a good end-to-end production check for later CUDA phases.

## Phase 1: Optional CUDA Build And Runtime Skeleton

Deliverables:

- CPU-only build remains unchanged.
- CUDA build compiles only when explicitly requested.
- Runtime can select CPU or GPU per element and fall back safely.

Tasks:

1. Audit the existing `HAVE_GPU`, `GPU_SUPPORT`, and `gpu_*` hook points before implementing new CUDA code:
   - confirm each hook still wraps the complete current CPU behavior
   - identify stale assumptions about particle layout, element state, random numbers, CSR state, and output side effects
   - remove or redesign hooks that would force an unsafe or inefficient backend shape
   - update `GPU_SUPPORT` flags only after the corresponding CUDA path is verified
2. Decide which existing hook names can remain as compatibility wrappers and which need new interfaces.
3. Add Makefile support:
   - `HAVE_CUDA=1` or `HAVE_GPU=1` opt-in.
   - `NVCC`, `CUDA_HOME`, `CUDA_ARCH`, CUDA include path, CUDA library path.
   - Use `NVCC=/usr/local/cuda-12.4/bin/nvcc` as the known-good local compiler setting for initial development on this machine.
   - When implementing Makefile discovery, search for `nvcc` on `PATH`, `/usr/local/cuda/bin/nvcc`, `/usr/local/cuda-12.4/bin/nvcc`, other `/usr/local/cuda-*/bin/nvcc` installations, and allow users to override with `NVCC=<path>`.
   - `.cu` compilation rules in `Makefile.build`.
   - link `cudart`; add `cublas`/`cufft` only when first used.
4. Add `src/gpu/` headers matching the retained or redesigned interfaces:
   - `gpu_base.h`
   - `gpu_funcs.h`
   - `gpu_matrix.h`
   - `gpu_limit_amplitudes.h`
   - feature headers for later phases.
5. Implement the runtime base:
   - device discovery and selection
   - CUDA error checking
   - allocation and deallocation
   - host-to-device and device-to-host synchronization
   - `getElementOnGpu`
   - `setElementGpuData`
   - `forceParticlesToCpu`
   - `startGpuTimer`, `startCpuTimer`, `displayTimings`
   - `compareGpuCpu`
6. Use a structure-of-arrays device layout for the six coordinates plus optional properties, while preserving the existing host `double **` representation.
7. Add runtime controls:
   - `off`, `auto`, and `required` modes
   - optional element allow/deny list
   - minimum particle count threshold
   - verification mode
8. Make unsupported features force CPU rather than fail, except in `required` mode.

Original hook reuse is not a gate by itself. It is acceptable, and likely necessary, for this phase to heavily modify the old GPU hooks if that produces a clearer, safer interface.

Gate:

- CPU build output is unchanged.
- CUDA build runs a CPU fallback path successfully on a machine without a usable GPU.
- `GPU_VERIFY` can compare a no-op GPU element without corrupting particle data.

Implementation status:

- Initial Phase 1 build support is in place. The default CPU build remains unchanged; CUDA support is enabled only with `HAVE_CUDA=1` or `HAVE_GPU=1`.
- The CUDA build searches for `nvcc` on `PATH`, `/usr/local/cuda-12.4/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, and other `/usr/local/cuda-*/bin/nvcc` installations. The known-good local command is:

  ```sh
  make -C src HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc
  ```

- CUDA-enabled objects are built under `src/O.Linux-x86_64.gpu`, with the executable copied to `bin/Linux-x86_64-gpu/gpu-elegant`, so the normal CPU object and binary directories are not reused.
- `Makefile.build` now supports `.cu` sources and creates nested object directories as needed.
- `src/gpu/` now contains the retained compatibility headers plus a Phase 1 runtime skeleton. It links `cudart`, queries the CUDA runtime, supports `ELEGANT_GPU_MODE=off|auto|required`, `ELEGANT_GPU_DEVICE`, `ELEGANT_GPU_MIN_PARTICLES`, `ELEGANT_GPU_VERIFY`, and `ELEGANT_GPU_VERBOSE`.
- `GPU_VERIFY=1` can be added to the CUDA make command; it builds into `src/O.Linux-x86_64.gpu.verify` and installs `bin/Linux-x86_64-gpu-verify/gpu-elegant` so verification objects are not mixed with the normal CUDA fallback build.
- No element kernels are enabled yet. In `off` and `auto`, every element falls back to the CPU path. In `required`, the program fails clearly if no CUDA device is available or if tracking reaches an element before a verified CUDA kernel exists.
- The dormant `HAVE_GPU` and `GPU_SUPPORT` hooks remain treated as stale compatibility signposts. Phase 1 deliberately does not trust the existing `GPU_SUPPORT` flags for runtime enablement; each flag should be revalidated and possibly revised in the later kernel phases.
- Several verification-only stale hooks were repaired during Phase 1: current `eptr` arguments are now passed in `CSBEND` and aperture verification calls, and the obsolete `compareCSR_LAST_WAKE` dependency was removed pending the first CSR kernel.

Verification performed:

- `make -C src O.Linux-x86_64/elegant -j4` reported the CPU target up to date.
- `make -C src HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4` completed successfully.
- `make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j4` completed successfully.
- `src/O.Linux-x86_64.gpu/elegant` starts as `gpu-elegant` and links against `/usr/local/cuda-12.4/lib64/libcudart.so.12`.
- Phase 0 quick CPU and CUDA-fallback runs completed successfully:

  ```sh
  ./test/gpu_cuda/run_benchmarks.sh --quick --elegant /home/soliday/github/elegant/src/O.Linux-x86_64/elegant --label cpu-phase1-smoke
  ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ./test/gpu_cuda/run_benchmarks.sh --quick --elegant /home/soliday/github/elegant/src/O.Linux-x86_64.gpu/elegant --label cuda-fallback-phase1-smoke
  ```

- `python3 test/gpu_cuda/compare_sdds.py test/gpu_cuda/output/cpu-phase1-smoke test/gpu_cuda/output/cuda-fallback-phase1-smoke-final` matched all 49 common SDDS files, including the LCLS0 production wrapper.
- A `GPU_VERIFY=1` matrix fallback smoke run matched the CPU matrix reference.
- A short device-discovery smoke run outside the sandbox selected `NVIDIA GeForce RTX 3060` and still used the Phase 1 CPU fallback path.
- A required-mode smoke run outside the sandbox failed as expected with: `no CUDA element kernels are enabled in the Phase 1 skeleton`.
- `test/gpu_cuda/run_benchmarks.sh` now canonicalizes relative `--elegant` and `--rpn-defns` paths before changing into case directories.

## Phase 2: Particle-Resident Basic Kernels

Deliverables:

- Particle data can remain on the GPU over simple consecutive elements.
- Basic kernels are correct and timed.

Targets:

- `gpu_track_particles`
- `gpu_matr_element_tracking`
- `gpu_ematrix_element_tracking`
- `gpu_exactDrift`
- `gpu_offset_beam`
- `gpu_center_beam`
- `gpu_do_match_energy`
- `gpu_set_central_momentum`
- `gpu_compute_centroids`
- `gpu_accumulate_beam_sums`

Tasks:

1. Implement first-order, second-order, and third-order matrix tracking.
2. Implement exact drift.
3. Implement offset, center, match-energy, and central-momentum update helpers.
4. Implement centroid and beam-sum reductions using deterministic reduction order where practical.
5. Ensure output paths and CPU-only elements call `forceParticlesToCpu`.
6. Determine the particle-count threshold at which each basic kernel helps.

Gate:

- Deterministic end-to-end tests match the CPU reference.
- Matrix-heavy benchmark shows an end-to-end improvement or the GPU path remains thresholded/off by default.

Wrap-up status:

- Phase 2 is functionally complete for the current scope. The retained CUDA paths meet the correctness gate, have bounded quick and target-60-second timing data, and are protected by particle-count thresholds so small jobs can stay on CPU.
- Move to Phase 3 rather than continuing to polish Phase 2. Recent Phase 2 work mostly reduced launch counts without meaningful end-to-end timing gains, and the tested fused `ENERGY,MATCH_PARTICLES=1` reduction/update kernel was rejected because it was slower.
- Carry these known Phase 2 limits forward:
  - `exactDrift` remains correct but opt-in only with `ELEGANT_GPU_ENABLE_EXACT_DRIFT=1` because timing did not justify enabling it by default.
  - specialized `accumulate_beam_sums` modes still synchronize to CPU.
  - CPU-only elements, file output, and host-side coordinate consumers still synchronize through `forceParticlesToCpu`.
  - `MARK` fitpoint output still synchronizes before reading coordinates.
  - unsupported matrix orders/shapes and radiation/spin side-effect paths fall back to CPU.

Implementation status:

- First Phase 2 slice is implemented for `track_particles` on generic `MATRIX_TRACKING` elements with first-, second-, and third-order matrix support. This includes the original simple set (`DRIF`, `QUAD`, `SBEN`, `RBEN`, `SEXT`, `OCT`, `MONI`, `HMON`, `VMON`) plus `QUFRINGE`, `MAGNIFY`, `WIGGLER`, `LTHINLENS`, `LMIRROR`, `BEDGE`, and `REFLECT`.
- The next Phase 2 slice adds CUDA coverage for `MATR`, `EMATRIX`, `MALIGN`, `ENERGY`, and `CENTER` helper paths. `CENTER` now supports both ordinary coordinate centering (`X`, `XP`, `Y`, `YP`, `S`, `DELTA`) and time-coordinate centering (`T=1`).
- A CUDA particle buffer now stays allocated and current across consecutive supported matrix elements. CPU-only elements, output paths, and CPU reductions call `forceParticlesToCpu` before reading host coordinates.
- CUDA Phase 2 paths are enabled in `ELEGANT_GPU_MODE=auto|required` only when the active particle count is at or above the configured threshold. `ELEGANT_GPU_MIN_PARTICLES` remains the global fallback threshold, while `ELEGANT_GPU_MIN_MATRIX_PARTICLES`, `ELEGANT_GPU_MIN_HELPER_PARTICLES`, `ELEGANT_GPU_MIN_REDUCTION_PARTICLES`, and `ELEGANT_GPU_MIN_EXACT_DRIFT_PARTICLES` can tune individual paths. Each path-specific value defaults to `ELEGANT_GPU_MIN_PARTICLES`, whose default remains 10000 particles. Use `ELEGANT_GPU_MIN_PARTICLES=1` only for small smoke tests.
- `exactDrift` has a CUDA kernel and a dedicated `exact_drift` benchmark case, but timing did not show a benefit in the current benchmark. It is therefore off by default and must be explicitly enabled with `ELEGANT_GPU_ENABLE_EXACT_DRIFT=1`.
- `GPU_VERIFY=1` now compares matrix GPU output against the CPU path in-process with default absolute/relative tolerances of `1e-12`. Tolerances can be overridden with `ELEGANT_GPU_COMPARE_ABS` and `ELEGANT_GPU_COMPARE_REL`.
- CUDA timing counters report element count, passive-residency count, matrix/exact-drift/helper/reduction kernel count, kernel time, host-to-device transfer time, and device-to-host transfer time when `ELEGANT_GPU_VERBOSE=1`.
- Matrix paths now probe the matrix object before starting the GPU timer or launching a kernel. `track_particles`, `MATR`, and `EMATRIX` fall back to CPU cleanly when the matrix is absent, has an unsupported order, or lacks the arrays required by the CUDA first- through third-order kernel.
- `compute_centroids` now uses a CUDA reduction when the particle buffer is resident on the GPU.
- `accumulate_beam_sums` now uses a CUDA reduction for the common no-filter, no-spin, no-exact-normalized-emittance path. The GPU path accumulates centroid sums, upper-triangular products, min/max, and maxabs values with a deterministic fixed-block reduction order, then updates the existing `BEAM_SUMS`/`BEAM_SUMS2` structures on the host.
- `MATR` and `EMATRIX` now apply their fiducial `sReference` inside the CUDA matrix kernel instead of launching separate subtract/restore coordinate kernels around matrix tracking.
- Helper scalar sums also now reuse CUDA reductions: `CENTER` offsets, `MATR`/`EMATRIX` fiducial `s` references, and `ENERGY` matching no longer force a full device-to-host particle synchronization merely to compute those averages. `CENTER` batches all requested ordinary coordinate sums into one centroid reduction per element, and when ordinary coordinate centering and `T=1` are both requested it uses a combined centroid/time reduction. The follow-on `CENTER` coordinate and time-coordinate updates are also fused into one helper kernel per element. Helper elements that need scalar sums require both the helper and reduction thresholds. `MALIGN` remains governed by the helper threshold alone.
- `CENTER,T=1` now uses a CUDA time-centering kernel that updates `s` as `s -= c * beta * t_offset`, matching the CPU `computeTimeCoordinates`/`computeDistanceCoordinates` behavior without materializing a host-side time array. It uses the combined centroid/time reduction when ordinary coordinate offsets are also needed, otherwise it uses a dedicated time-sum reduction.
- `MONI`, `HMON`, and `VMON` now remain GPU-resident as matrix-tracking monitor elements. Their optional turn-by-turn and closed-orbit readout paths use the existing GPU centroid/beam-sum reductions before the monitor matrix is applied, avoiding the previous CPU synchronization at every passive BPM in Phase 2 benchmark lattices.
- Special-cased matrix elements `SOLE`, `ROTATE`, `HKICK`/`HCOR`, `VKICK`/`VCOR`, and combined `KICKER`/`HVCOR` now also use the generic matrix GPU path when no immediate CPU side effect is required. Solenoid and rotation elements still fall back when spin coordinates are active, and correctors fall back when `SYNCH_RAD` or `ISR` is enabled.
- Coordinate-neutral `CHARGE`, `RECIRC`, `MARK`, `TRCOUNT`, and `FLOORELEMENT` elements now preserve GPU residency when they are reached with a current device particle buffer. They do not start GPU residency by themselves, and `MARK` fitpoint output still synchronizes before reading coordinates. Trajectory centroid collection now uses the GPU centroid reduction instead of the old unsupported `gpu_collect_trajectory_data` stub.
- Specialized `accumulate_beam_sums` modes still fall back to CPU synchronization: explicit `timeValue`, active time or PID filters, exact normalized emittance, `BEAM_SUMS_EXACTEMIT`, spin sums, and spin-coordinate offsets.
- `GPU_VERIFY=1` now compares reduction outputs in-process. The reduction comparison uses `ELEGANT_GPU_REDUCTION_COMPARE_ABS` and `ELEGANT_GPU_REDUCTION_COMPARE_REL`, falling back to the general `ELEGANT_GPU_COMPARE_ABS`/`ELEGANT_GPU_COMPARE_REL`; the default reduction tolerance is `1e-10`. Large relative differences can appear for near-zero covariance values, so the absolute tolerance remains the primary guard there.

Verification notes:

- CPU matrix quick and CUDA matrix quick outputs matched for all 4 SDDS files. The CUDA run exercised 280 matrix kernels.
- `GPU_VERIFY=1` matrix quick passed in-process comparisons for order-2 and order-3 matrices with `maxAbs=0` and `maxRel=0`, then matched the CPU SDDS files.
- CPU matrix 60-second baseline: 20000 particles, 116 passes, 59.66 seconds.
- CUDA matrix 60-second baseline: 20000 particles, 215 passes, 57.69 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files, indicating a useful matrix-kernel throughput speedup at this size.
- CPU exact-drift 60-second baseline: 30000 particles, 140 passes, 59.78 seconds.
- CUDA exact-drift 60-second baseline with `ELEGANT_GPU_ENABLE_EXACT_DRIFT=1`: 30000 particles, 126 passes, 59.76 seconds. The common 40-pass sample outputs matched all 4 CPU SDDS files, but timing still does not justify enabling it by default.
- CPU `phase2_helpers` quick and CUDA `phase2_helpers` quick outputs matched for all 4 SDDS files. The CUDA run exercised 120 matrix kernels and 201 helper kernels.
- `GPU_VERIFY=1` `phase2_helpers` quick passed in-process comparisons for `offset_beam`, `center_beam`, `do_match_energy`, `matr_element_tracking`, `ematr_element_tracking`, and ordinary matrix tracking, then matched the CPU SDDS files.
- CPU `phase2_helpers` 60-second baseline: 20000 particles, 233 passes, 57.66 seconds.
- CUDA `phase2_helpers` 60-second baseline: 20000 particles, 310 passes, 54.85 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files, indicating the helper slice is beneficial at this size despite CPU-ordered reduction transfers.
- After adding the common reduction kernels, CUDA `phase2_helpers` quick matched the CPU quick reference for all 4 SDDS files and reported `matrix=120`, `helpers=201`, `reductions=200`, `wall=0.062670s`, `kernel=0.031904s`, `h2d=0.000497s`, and `d2h=0.003767s`.
- `GPU_VERIFY=1` `phase2_helpers` quick also matched all 4 CPU SDDS files. In-process reduction checks for `accumulate_beam_sums` passed with maximum absolute differences on the order of `1e-14`; some relative errors were large only for near-zero values and remained within the configured absolute tolerance.
- CUDA `phase2_helpers` 60-second baseline after common reduction kernels: 20000 particles, 702 passes, 49.37 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. Compared with the CPU 60-second baseline, this is about 3.5x more passes per second; compared with the earlier GPU helper-only baseline, it is about 2.5x more passes per second.
- After moving helper scalar sums to GPU reductions, CUDA `phase2_helpers` quick matched all 4 CPU SDDS files and reported `matrix=120`, `helpers=201`, `reductions=308`, `wall=0.065347s`, `kernel=0.033797s`, `h2d=0.000521s`, and `d2h=0.000648s`. `GPU_VERIFY=1` also matched all 4 CPU SDDS files, with helper comparisons passing and reduction absolute differences remaining on the order of `1e-14`.
- CUDA `phase2_helpers` 60-second baseline after helper scalar-sum reductions: 20000 particles, 690 passes, 45.84 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. This is about 3.7x more passes per second than the CPU baseline, about 2.7x more passes per second than the helper-only GPU baseline, and about 6 percent faster than the prior common-reduction GPU baseline.
- Added `test/gpu_cuda/cases/phase2_time_center` to exercise `CENTER,T=1` directly. The first version of the case was too unstable for a meaningful 60-second run, so the lattice was softened; the stabilized case keeps all 20000 particles through the timing run.
- CPU `phase2_time_center` quick and CUDA `phase2_time_center` quick outputs matched for all 4 SDDS files. The CUDA quick run reported `matrix=80`, `helpers=140`, `reductions=260`, `wall=0.054777s`, `kernel=0.025881s`, `h2d=0.000649s`, and `d2h=0.000748s`.
- `GPU_VERIFY=1` `phase2_time_center` quick matched all 4 CPU SDDS files and passed in-process comparisons for `CENTER,T=1`, with `center_beam` maximum absolute coordinate differences around `2e-16`.
- CPU `phase2_time_center` 60-second baseline: 20000 particles, 288 passes, 59.88 seconds.
- CUDA `phase2_time_center` 60-second baseline: 20000 particles, 1019 passes, 59.15 seconds. The common 200-pass sample outputs matched all 4 CPU SDDS files, for about 3.6x more passes per second than CPU on this case.
- A post-change regression run of CUDA `phase2_helpers` quick still matched all 4 CPU SDDS files and reported `matrix=120`, `helpers=201`, `reductions=308`, `wall=0.060831s`, `kernel=0.031779s`, `h2d=0.000504s`, and `d2h=0.000615s`.
- After making `MONI`/`HMON`/`VMON` GPU-resident, CUDA `phase2_helpers` quick matched all 4 CPU SDDS files and reported `matrix=140`, `helpers=201`, `reductions=334`, `wall=0.049219s`, `kernel=0.035949s`, `h2d=0.000032s`, and `d2h=0.000000s`. The previous per-monitor `CPU element after CUDA element` synchronizations disappeared from this quick run.
- `GPU_VERIFY=1` `phase2_helpers` quick also matched all 4 CPU SDDS files after monitor residency; in-process monitor matrix comparisons passed with `maxAbs=0` and `maxRel=0`.
- CUDA `phase2_helpers` 60-second baseline after monitor residency: 20000 particles, 1062 passes, 45.52 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. This is about 5.8x more passes per second than the CPU baseline and about 54 percent faster than the prior helper-scalar-reduction GPU baseline.
- CUDA `phase2_time_center` quick after monitor residency matched all 4 CPU SDDS files and reported `matrix=100`, `helpers=140`, `reductions=286`, `wall=0.043324s`, `kernel=0.029795s`, `h2d=0.000036s`, and `d2h=0.000000s`. `GPU_VERIFY=1` also matched all 4 CPU SDDS files and passed in-process comparisons.
- CUDA `phase2_time_center` 60-second baseline after monitor residency: 20000 particles, 1693 passes, 58.29 seconds. The common 200-pass sample outputs matched all 4 CPU SDDS files. This is about 6.0x more passes per second than the CPU baseline and about 68 percent faster than the prior `CENTER,T=1` GPU baseline.
- Added `test/gpu_cuda/cases/phase2_special_matrix` for `SOLE`, both ordinary and `EXCLUDE_OPTICS` `ROTATE`, and non-radiating `HKICK`/`VKICK` elements.
- CPU `phase2_special_matrix` quick and CUDA `phase2_special_matrix` quick outputs matched for all 4 SDDS files. The CUDA quick run reported `matrix=300`, `helpers=0`, `reductions=306`, `wall=0.070084s`, `kernel=0.053579s`, `h2d=0.000044s`, and `d2h=0.000000s`.
- `GPU_VERIFY=1` `phase2_special_matrix` quick matched all 4 CPU SDDS files and passed in-process `track_particles_M1`, `track_particles_M2`, and `track_particles_M3` comparisons with `maxAbs=0` and `maxRel=0`.
- CPU `phase2_special_matrix` 60-second baseline: 20000 particles, 127 passes, 59.44 seconds.
- CUDA `phase2_special_matrix` 60-second baseline: 20000 particles, 822 passes, 50.03 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. This is about 7.7x more passes per second than the CPU baseline on the new special-matrix case.
- Post-change CUDA quick regressions for `phase2_helpers` and `phase2_time_center` still matched their CPU references for all 4 SDDS files. The runs reported `matrix=140`, `helpers=201`, `reductions=334` for `phase2_helpers` and `matrix=100`, `helpers=140`, `reductions=286` for `phase2_time_center`.
- Added `test/gpu_cuda/cases/phase2_matrix_extended` for the remaining ordinary `MATRIX_TRACKING` family and non-radiating combined `KICKER`: `QUFRINGE`, `MAGNIFY`, `WIGGLER`, `LTHINLENS`, `LMIRROR`, `BEDGE`, `REFLECT`, and `HVCOR`.
- CPU `phase2_matrix_extended` quick and CUDA `phase2_matrix_extended` quick outputs matched for all 4 SDDS files. The CUDA quick run reported `matrix=300`, `helpers=0`, `reductions=306`, `wall=0.066299s`, `kernel=0.050337s`, `h2d=0.000038s`, and `d2h=0.000000s`.
- `GPU_VERIFY=1` `phase2_matrix_extended` quick matched all 4 CPU SDDS files and passed in-process `track_particles_M1`, `track_particles_M2`, and `track_particles_M3` comparisons with `maxAbs=0` and `maxRel=0`.
- CPU `phase2_matrix_extended` baseline, aimed at a 60-second CPU run: 20000 particles, 140 passes, 44.76 seconds.
- CUDA `phase2_matrix_extended` 60-second baseline: 20000 particles, 839 passes, 56.71 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. This is about 4.7x more passes per second than the CPU baseline on this extended matrix case.
- Post-change CUDA quick regressions for the original `matrix` and `phase2_special_matrix` cases still matched their CPU references for all 4 SDDS files. The runs reported `matrix=320`, `reductions=326` for `matrix` and `matrix=300`, `reductions=306` for `phase2_special_matrix`, with no mid-run device-to-host synchronization.
- Added a matrix support probe before CUDA launch so newly eligible matrix elements and explicit `MATR`/`EMATRIX` inputs fall back instead of failing if a future input produces an unsupported matrix. Normal CUDA quick regressions for `phase2_matrix_extended`, `phase2_helpers`, and `phase2_special_matrix` matched their CPU references for all 4 SDDS files.
- `GPU_VERIFY=1` quick regressions after the matrix support probe matched CPU SDDS output for `phase2_matrix_extended` and `phase2_helpers`; in-process comparisons continued to pass for `track_particles_M1`, `track_particles_M2`, `track_particles_M3`, `matr_element_tracking`, and `ematr_element_tracking`.
- CUDA timing after the support probe showed no meaningful regression: `phase2_matrix_extended` kept the same 839-pass scaled run and completed in 57.09 seconds, while `phase2_helpers` completed 1071 passes in 45.53 seconds. The common 20-pass samples matched all 4 CPU SDDS files for both cases.
- Added `test/gpu_cuda/cases/phase2_residency` for matrix elements separated by coordinate-neutral `CHARGE`, `RECIRC`, `MARK`, `TRCOUNT`, and `FLOORELEMENT` elements.
- CPU `phase2_residency` quick and CUDA `phase2_residency` quick outputs matched for all 4 SDDS files. The CUDA quick run reported `elements=320`, `passive=124`, `matrix=320`, `reductions=450`, `h2d=0.000028s`, and `d2h=0.000000s`.
- `GPU_VERIFY=1` `phase2_residency` quick matched CPU SDDS output for all 4 files; in-process matrix comparisons continued to pass with `maxAbs=0` and `maxRel=0`.
- CPU `phase2_residency` 60-second baseline: 20000 particles, 100 passes, 59.37 seconds.
- CUDA `phase2_residency` 60-second baseline: 20000 particles, 615 passes, 51.86 seconds. The CUDA run reported `passive=15374`, `matrix=39360`, `reductions=55350`, and no mid-run device-to-host synchronization; the common 20-pass sample outputs matched all 4 CPU SDDS files. This is about 7.0x more passes per second than CPU on this residency-focused case.
- A post-change CUDA quick regression for `phase2_helpers` still matched the CPU reference for all 4 SDDS files and reported `matrix=140`, `helpers=201`, `reductions=334`, and `d2h=0.000000s`.
- Added per-path particle thresholds for Phase 2 timing experiments: `ELEGANT_GPU_MIN_MATRIX_PARTICLES`, `ELEGANT_GPU_MIN_HELPER_PARTICLES`, `ELEGANT_GPU_MIN_REDUCTION_PARTICLES`, and `ELEGANT_GPU_MIN_EXACT_DRIFT_PARTICLES`. They default to `ELEGANT_GPU_MIN_PARTICLES`, so existing runs keep the prior behavior unless a path-specific threshold is set.
- With the default path-specific thresholds inherited from `ELEGANT_GPU_MIN_PARTICLES=1`, CUDA `phase2_helpers` quick matched the CPU reference for all 4 SDDS files and reported `matrix=140`, `helpers=201`, `reductions=334`, `wall=0.049348s`, `h2d=0.000036s`, and `d2h=0.000000s`.
- Raising `ELEGANT_GPU_MIN_HELPER_PARTICLES` and `ELEGANT_GPU_MIN_REDUCTION_PARTICLES` to 100000 for the 2000-particle `phase2_helpers` quick run forced those paths back to CPU, reported `helpers=0` and `reductions=0`, and still matched the CPU reference for all 4 SDDS files.
- Raising `ELEGANT_GPU_MIN_MATRIX_PARTICLES` to 100000 for the 2000-particle `matrix` quick run disabled matrix launches, reported `matrix=0`, and still matched the CPU reference for all 4 SDDS files.
- `GPU_VERIFY=1` `phase2_helpers` quick passed after the threshold changes, with in-process checks still passing for matrix tracking, helper kernels, and reduction comparisons; the SDDS files matched the CPU reference for all 4 files.
- CUDA `phase2_helpers` target-60-second baseline after adding the threshold gates: 20000 particles, 1062 passes, 45.45 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files, indicating no meaningful regression from the threshold branching compared with the prior 45.52-second helper baseline.
- Batched `CENTER` ordinary-coordinate sums so one centroid reduction supplies all requested `X`, `XP`, `Y`, `YP`, `S`, and `DELTA` offsets for a `CENTER` element. CUDA `phase2_helpers` quick matched all 4 CPU SDDS files and reduced the quick-run reduction count from 334 to 274.
- CUDA `phase2_time_center` quick after the `CENTER` batching change matched all 4 CPU SDDS files and reduced the quick-run reduction count from 286 to 226 while preserving the separate time-sum reduction for `T=1`.
- `GPU_VERIFY=1` `phase2_time_center` quick passed after the `CENTER` batching change; in-process `center_beam` comparisons had maximum absolute differences around `2e-16`, and all 4 SDDS files matched CPU.
- CUDA `phase2_time_center` target-60-second baseline after `CENTER` batching: 20000 particles, 1732 passes, 57.91 seconds. The common 200-pass sample outputs matched all 4 CPU SDDS files, and throughput was slightly better than the prior 1693-pass, 58.29-second run.
- Added a combined centroid/time reduction for `CENTER` elements that need both ordinary coordinate offsets and `T=1`, avoiding a second reduction on those elements.
- CUDA `phase2_time_center` quick after the combined `CENTER` reduction matched all 4 CPU SDDS files and reduced the quick-run reduction count from 226 to 206.
- `GPU_VERIFY=1` `phase2_time_center` quick passed after the combined `CENTER` reduction; in-process `center_beam` comparisons again had maximum absolute differences around `2e-16`, and all 4 SDDS files matched CPU.
- CUDA `phase2_time_center` target-60-second baseline after the combined `CENTER` reduction: 20000 particles, 1760 passes, 58.38 seconds. The common 200-pass sample outputs matched all 4 CPU SDDS files. This is slightly faster than the prior 1732-pass, 57.91-second run and about 6.3x more passes per second than the CPU baseline.
- Fused the `MATR`/`EMATRIX` fiducial `sReference` adjustment into the CUDA matrix kernel. CUDA `phase2_helpers` quick matched all 4 CPU SDDS files and reduced helper kernel launches from 201 to 121, with `matrix=140`, `helpers=121`, `reductions=274`, `h2d=0.000037s`, and `d2h=0.000000s`.
- `GPU_VERIFY=1` `phase2_helpers` quick passed after the fused fiducial change; in-process `matr_element_tracking` and `ematr_element_tracking` comparisons had `maxAbs=0` and `maxRel=0`, and all 4 SDDS files matched CPU.
- A plain CUDA `matrix` quick regression also matched all 4 CPU SDDS files after adding the optional fiducial fields to the packed matrix data, confirming the normal non-fiducial matrix path was unchanged. It reported `wall=0.075778s` and `kernel=0.057241s`, comparable to the prior post-residency quick run.
- CUDA `phase2_helpers` target-60-second baseline after the fused fiducial change: 20000 particles, 1111 passes, 46.02 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files. This is a modest throughput improvement over the prior 1062-pass, 45.45-second helper baseline.
- Fused ordinary `CENTER` coordinate updates and optional `CENTER,T=1` time updates into one CUDA helper kernel per `CENTER` element. CUDA `phase2_helpers` quick matched all 4 CPU SDDS files and reduced helper kernel launches from 121 to 61, with `matrix=140`, `helpers=61`, `reductions=274`, `h2d=0.000036s`, and `d2h=0.000000s`.
- CUDA `phase2_time_center` quick after the fused `CENTER` update matched all 4 CPU SDDS files and reduced helper kernel launches from 140 before the `MATR`/`CENTER` fusions to 60, with `matrix=100`, `helpers=60`, `reductions=206`, `h2d=0.000036s`, and `d2h=0.000000s`.
- `GPU_VERIFY=1` quick runs for `phase2_helpers` and `phase2_time_center` passed after the fused `CENTER` update; in-process `center_beam` comparisons had maximum absolute differences around `2e-16`, and all SDDS files matched CPU.
- CUDA `phase2_helpers` target-60-second baseline after the fused `CENTER` update: 20000 particles, 1111 passes, 46.04 seconds. The common 20-pass sample outputs matched all 4 CPU SDDS files; throughput was essentially unchanged from the 1111-pass, 46.02-second fused-fiducial baseline despite halving helper launch count.
- CUDA `phase2_time_center` target-60-second baseline after the fused `CENTER` update: 20000 particles, 1727 passes, 57.15 seconds. The common 200-pass sample outputs matched all 4 CPU SDDS files; passes per second were essentially flat compared with the prior 1760-pass, 58.38-second combined-reduction baseline.
- Tested a fused `ENERGY,MATCH_PARTICLES=1` reduction/update kernel, but did not retain it. It matched all 4 CPU SDDS files and passed `GPU_VERIFY=1` with `do_match_energy` maximum absolute differences around `7e-15`, but the target-60-second `phase2_helpers` timing regressed to 1026 passes in 44.81 seconds versus the retained 1111-pass, 46.04-second baseline.
- Final Phase 2 wrap-up quick smoke sweep matched CPU references for all 4 SDDS files in each case: `matrix`, `phase2_helpers`, `phase2_time_center`, `phase2_special_matrix`, `phase2_matrix_extended`, `phase2_residency`, and opt-in `exact_drift`.
- Fresh wrap-up CUDA counters: `matrix` reported `matrix=320`, `reductions=326`; `phase2_helpers` reported `matrix=140`, `helpers=61`, `reductions=274`; `phase2_time_center` reported `matrix=100`, `helpers=60`, `reductions=206`; `phase2_special_matrix` reported `matrix=300`, `reductions=306`; `phase2_matrix_extended` reported `matrix=300`, `reductions=306`; `phase2_residency` reported `passive=124`, `matrix=320`, `reductions=450`; opt-in `exact_drift` reported `matrix=40`, `exactDrift=800`, `reductions=851`.

## Phase 3: Apertures, Losses, And Particle Compaction

Wrap-up status:

- Phase 3 is wrapped up for the current CUDA scope. The implemented aperture predicates cover the highest-value simple aperture/loss paths identified so far, and the remaining aperture work below is deferred so Phase 4 can start.
- Added conservative rectangular `limit_amplitudes`, elliptical `elimit_amplitudes`, `rectangular_collimator`, `elliptical_collimator`, and simple one-sided `beam_scraper` CUDA predicates for `MAXAMP`, `RCOL`, `ECOL`, and `SCRAPER` checks that run after GPU-resident elements.
- The Phase 3 slice counts would-be losses on the device and keeps particles resident only when the count is zero.
- By default, if any particle would be lost, if open-side aperture logic is requested, or if the particle count is below `ELEGANT_GPU_MIN_APERTURE_PARTICLES`, the code synchronizes and runs the exact CPU aperture loss/compaction path.
- Added opt-in exact CUDA compaction kernels for simple rectangular and elliptical `MAXAMP` losses. Enable with `ELEGANT_GPU_ENABLE_APERTURE_COMPACTION=1`. These kernels mirror the CPU swap-with-top compaction order and then synchronize the full particle array back to the host so existing lost-particle bookkeeping remains unchanged. They are disabled by default because current timing is slower than the CPU fallback.
- Added `ELEGANT_GPU_MIN_APERTURE_PARTICLES`, an `apertures=` timing counter, and smoke cases `phase3_limit_amplitudes`, `phase3_limit_loss`, `phase3_elimit_amplitudes`, `phase3_elimit_loss`, `phase3_rcol`, `phase3_rcol_loss`, `phase3_ecol`, `phase3_ecol_loss`, `phase3_scraper`, and `phase3_scraper_loss`.
- Build checks passed for `make -C src`, `make -C src HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc`, and `make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=/usr/local/cuda-12.4/bin/nvcc`.
- Quick SDDS comparisons matched all 5 common files for the rectangular no-loss case (`cpu-phase3-limit-amplitudes-matrix` vs `gpu-phase3-limit-amplitudes-matrix`), rectangular lossy fallback case (`cpu-phase3-limit-loss` vs `gpu-phase3-limit-loss`), elliptical no-loss case (`cpu-phase3-elimit-amplitudes-final` vs `gpu-phase3-elimit-amplitudes-final`), elliptical lossy fallback case (`cpu-phase3-elimit-loss` vs `gpu-phase3-elimit-loss`), rectangular collimator no-loss/loss cases (`cpu-phase3-rcol`/`cpu-phase3-rcol-loss` vs GPU), and elliptical collimator no-loss/loss cases (`cpu-phase3-ecol`/`cpu-phase3-ecol-loss` vs GPU).
- Fresh post-compaction quick SDDS comparisons matched all 5 common files for `phase3_limit_amplitudes`, `phase3_elimit_amplitudes`, default lossy `phase3_limit_loss`, default lossy `phase3_elimit_loss`, and opt-in compacting `phase3_limit_loss`/`phase3_elimit_loss`.
- Rectangular no-loss quick timing was CPU 0.50 seconds and GPU 0.31 seconds with CUDA counters `matrix=320`, `reductions=326`, and `apertures=280`.
- Rectangular lossy default-fallback quick timing was CPU 0.15 seconds and GPU 0.26 seconds with CUDA counters `matrix=192`, `reductions=196`, and `apertures=168`; the slowdown is expected because this path still synchronizes and runs CPU compaction whenever losses occur.
- Rectangular lossy opt-in CUDA compaction quick timing was CPU 0.15 seconds and GPU 0.30 seconds with CUDA counters `matrix=192`, `reductions=196`, and `apertures=286`; this confirmed correctness but not speed, so the path remains opt-in.
- Elliptical no-loss quick timing was CPU 0.49 seconds and GPU 0.33 seconds with CUDA counters `matrix=320`, `reductions=326`, and `apertures=280`.
- Elliptical lossy default-fallback quick timing was CPU 0.15 seconds and GPU 0.28 seconds with CUDA counters `matrix=192`, `reductions=196`, and `apertures=168`.
- Elliptical lossy opt-in CUDA compaction quick timing was CPU 0.15 seconds and GPU 0.46 seconds with CUDA counters `matrix=192`, `reductions=196`, and `apertures=310`; this path is correct but clearly not yet beneficial.
- Rectangular collimator no-loss quick timing was CPU 0.60 seconds and GPU 0.32 seconds with CUDA counters `matrix=360`, `exactDrift=40`, and `apertures=80`.
- Rectangular collimator lossy fallback quick timing was CPU 0.18 seconds and GPU 0.24 seconds with CUDA counters `matrix=216`, `reductions=244`, and `apertures=24`.
- Elliptical collimator no-loss quick timing was CPU 0.61 seconds and GPU 0.31 seconds with CUDA counters `matrix=360`, `exactDrift=40`, and `apertures=80`.
- Elliptical collimator lossy fallback quick timing was CPU 0.16 seconds and GPU 0.24 seconds with CUDA counters `matrix=216`, `reductions=244`, and `apertures=24`.
- One-sided scraper no-loss quick timing was CPU 0.50 seconds and GPU 0.31 seconds with CUDA counters `matrix=360`, `exactDrift=40`, `reductions=406`, and `apertures=80`.
- One-sided scraper lossy fallback quick timing was CPU 0.18 seconds and GPU 0.26 seconds with CUDA counters `matrix=216`, `reductions=244`, and `apertures=24`.
- `GPU_VERIFY=1` passed the no-loss `rectangular_collimator` and `elliptical_collimator` in-process checks with zero coordinate differences.
- `GPU_VERIFY=1` passed the no-loss `beam_scraper` in-process checks with zero coordinate differences.
- The `GPU_VERIFY` `MAXAMP` no-loss smoke run matched CPU SDDS output, but its aperture counter was zero because matrix verification currently resets GPU residency before the post-element aperture check. Use non-verify CPU/GPU paired runs for `MAXAMP` aperture-residency timing until in-process verification can preserve a saved pre-aperture state.

Deferred Phase 3 follow-ups:

- `gpu_removeInvalidParticles`: still uses the CPU path. This is expected to be lower payoff than Phase 4 multipole/bend work and should be revisited when a broader GPU loss-compaction mechanism exists.
- `gpu_imposeApertureData`: still uses the CPU path. Defer until the aperture-data interpolation and loss-coordinate semantics can be mapped cleanly to CUDA and compared against production aperture-data cases.
- Faster parallel loss compaction: the current opt-in `MAXAMP` compaction kernels are exact but serial and slower than CPU fallback in quick tests. A future version should use a parallel partition/scan strategy, preserve accepted/lost particle semantics, and avoid unnecessary host synchronization when downstream tracking can stay GPU-resident.

Deliverables:

- GPU aperture checks preserve accepted/lost particle semantics.

Targets:

- `gpu_limit_amplitudes`
- `gpu_elimit_amplitudes`
- `gpu_rectangular_collimator`
- `gpu_elliptical_collimator`
- `gpu_beam_scraper`

Tasks:

1. Implement aperture predicates with the same boundary conventions as CPU code.
2. Implement stable particle compaction for surviving particles where timing justifies keeping it enabled by default.
3. Preserve and compare `particleID`, `lossPassIndex`, accepted arrays, and loss coordinates.
4. Verify mixed runs where particles are lost before a CPU-only element.
5. Time with low-loss and high-loss cases because compaction cost changes with loss fraction.

Gate:

- Surviving `particleID` sets are identical in deterministic tests.
- Loss output files compare within tolerance.
- GPU path is enabled only above the measured break-even particle count.

Gate status:

- Passed for the implemented no-loss aperture predicates and CPU-fallback lossy paths.
- Opt-in rectangular/elliptical `MAXAMP` CUDA compaction passed correctness checks but did not pass the timing gate, so it remains disabled by default.
- Deferred follow-ups remain outside the Phase 3 wrap-up gate and should not block Phase 4.

## Phase 4: Symplectic Multipoles And Non-CSR Bends

Deliverables:

- Main magnet tracking kernels are accelerated for large bunches.

Implementation status:

- Phase 4 is wrapped for the current CUDA scope. The implemented deterministic magnet kernels meet the Phase 4 correctness and timing gates, and the remaining magnet variants below are deferred follow-ups that should not block Phase 5.
- First deterministic multipole slice is implemented for `gpu_multipole_tracking2` on simple `KQUAD`, `KSEXT`, `KOCT`, and `DQCOR` elements. The CUDA path uses double precision, avoids fast math, and keeps `--fmad=false` in the CUDA build flags.
- First deterministic non-CSR `CSBEND` slice is implemented for bends without radiation, reference correction, Hwang/Lindberg/curved fringe models, apertures, spin, or misalignment/error-tilt corrections. It supports first-order `EDGE_EFFECTS=1` edge focusing, Brown-style higher-order `EDGE_EFFECTS=1` focusing including pole-face curvature terms, `E1`/`E2`, finite `HGAP`/`FINT` `psi` terms, edge kick limits for the first-order path, and deterministic expanded-Hamiltonian tracking. It reuses the CPU-generated CSBEND field coefficient tables and applies the same checked single-pass device backup/restore pattern as the multipole path.
- `KOCT` now has current `GPU_SUPPORT` metadata so it can enter the runtime GPU eligibility checks.
- The multipole path has a path-specific threshold, `ELEGANT_GPU_MIN_MAGNET_PARTICLES`, which defaults to `ELEGANT_GPU_MIN_PARTICLES`.
- CUDA timing output now includes a `magnets=` counter for magnet kernels.
- The implemented magnet kernels use a checked single-pass device path: particle coordinates are backed up on the device, tracked once, and restored before CPU fallback if a particle becomes invalid or lost. This avoids duplicate predicate passes while preserving exact CPU fallback semantics.

Deferred Phase 4 follow-ups:

- Radiation, ISR, distribution-based radiation, and photon output paths.
- Spin coordinates.
- Multipole misalignments, tilt, pitch, yaw, and nonzero `MALIGN_METHOD`.
- KQUAD edge/fringe effects, radial mode, and `LEFFECTIVE` end drifts.
- Error/edge/steering multipole data files or initialized extra multipole data.
- CSBEND `REFERENCE_CORRECTION`, `FSE_CORRECTION`, Hwang/Lindberg/curved fringe models (`EDGE_EFFECTS` 2, 4, or 5), and misalignment/error-tilt/pitch/yaw corrections.
- Sticky aperture hooks (`MAXAMP`, `APCONTOUR`, `aperture_data`) and slice-by-slice tracking.
- GPU-resident magnet loss handling beyond the current checked backup/restore fallback for particle loss or invalid slopes.
- Corrector radiation kicks.

Completed Phase 4 targets:

- `gpu_multipole_tracking2` for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`
- `gpu_track_through_csbend` for non-CSR `CSBEND`

Deferred Phase 4 target:

- `gpu_addCorrectorRadiationKick`

Tasks:

1. Completed: start with radiation and stochastic effects disabled.
2. Completed: implement common multipole kernel code for orders used by `KQUAD`, `KSEXT`, `KOCT`, and `DQCOR`.
3. Completed: implement non-CSR `CSBEND` without `REFERENCE_CORRECTION`, then broaden it through first-order edges, expanded Hamiltonian, and Brown-style higher-order `EDGE_EFFECTS=1`.
4. Completed: avoid CUDA fast-math by default. Use double precision and keep `--fmad=false` in the CUDA build flags.
5. Deferred: support misalignments, tilt/pitch/yaw, slices, end drifts, aperture hooks, stochastic/radiation effects, Hwang/Lindberg/curved fringe models, and corrector radiation kicks.

Gate:

- Per-particle deterministic output matches the CPU reference within the Phase 4 tolerance.
- Magnet-heavy benchmark shows meaningful element-level and end-to-end speedup.

Gate status:

- `multipole` quick CPU and CUDA outputs matched all 4 SDDS files. CUDA quick used `magnets=80` and completed in 0.24 seconds versus 0.29 seconds for CPU.
- `GPU_VERIFY=1` `multipole` quick passed in-process `multipole_tracking2` comparisons with maximum absolute differences around `1e-19`, then matched all 4 CPU SDDS files.
- CPU `multipole` one-minute baseline: 20000 particles, 54 passes, 59.28 seconds.
- CUDA `multipole` one-minute-style baseline after checked single-pass optimization: 20000 particles, 472 passes, 47.60 seconds. The common 10-pass sample matched all 4 CPU SDDS files. This is about 10.9x more passes per second than the CPU baseline for this magnet-heavy case.
- Added `phase4_dqcor` for DQCOR-specific coverage. CPU/GPU quick outputs matched all 4 SDDS files, CUDA quick completed in 0.27 seconds versus 0.61 seconds for CPU, and `GPU_VERIFY=1` passed DQCOR in-process comparisons with maximum absolute differences around `1e-16`.
- `csbend` quick CPU and CUDA outputs matched all 4 SDDS files for the original no-active-edge case. CUDA quick used `magnets=32` and completed in 0.25 seconds.
- `GPU_VERIFY=1` `csbend` quick passed in-process `track_through_csbend` comparisons with maximum absolute differences around `4e-15`, then matched all 4 CPU SDDS files for the original no-active-edge case.
- CPU `csbend` one-minute baseline: 20000 particles, 68 passes, 59.13 seconds.
- CUDA `csbend` one-minute-style baseline: 20000 particles, 676 passes, 38.28 seconds. The common 8-pass sample matched all 4 CPU SDDS files. This is about 15.4x more passes per second than the CPU baseline for this CSBEND-heavy case.
- The updated `csbend` case now includes first-order edge focusing with nonzero `E1`/`E2`, finite `HGAP`, and `FINT`. CPU/GPU quick outputs matched all 4 SDDS files, CUDA quick used `magnets=32`, and `GPU_VERIFY=1` passed in-process `track_through_csbend` comparisons with maximum absolute differences around `4e-15`.
- Updated first-order-edge `csbend` timing gate: CPU one-minute baseline used 20000 particles, 67 passes, 59.20 seconds. CUDA one-minute-style baseline used 20000 particles, 686 passes, 39.39 seconds. The common 8-pass sample matched all 4 CPU SDDS files. This is about 15.4x more passes per second than the CPU baseline for this CSBEND-heavy case.
- Added `phase4_csbend_expanded` for deterministic `CSBEND,EXPAND_HAMILTONIAN=1` coverage with first-order edge focusing. CPU/GPU quick outputs matched all 4 SDDS files, CUDA quick used `magnets=32`, and `GPU_VERIFY=1` passed in-process `track_through_csbend` comparisons with maximum absolute differences around `1e-18`.
- Expanded-Hamiltonian `CSBEND` timing gate: CPU one-minute baseline used 20000 particles, 99 passes, 58.78 seconds. CUDA one-minute-style baseline used 20000 particles, 787 passes, 35.28 seconds. The common 8-pass sample matched all 4 CPU SDDS files. This is about 13.3x more passes per second than the CPU baseline for this expanded-CSBEND-heavy case.
- Added `phase4_csbend_ho_edge` for deterministic Brown-style higher-order `EDGE_EFFECTS=1` coverage with nonzero `H1`/`H2` pole-face curvature terms. CPU/GPU quick outputs matched all 4 SDDS files, CUDA quick used `magnets=32`, and `GPU_VERIFY=1` passed in-process `track_through_csbend` comparisons with maximum absolute differences around `4e-15`.
- Higher-order-edge `CSBEND` timing gate: CPU one-minute baseline used 20000 particles, 66 passes, 59.13 seconds. CUDA one-minute-style baseline used 20000 particles, 676 passes, 39.28 seconds. The common 8-pass sample matched all 4 CPU SDDS files. This is about 15.4x more passes per second than the CPU baseline for this high-order-edge CSBEND-heavy case.
- Phase 4 is wrapped for the current CUDA scope. Hwang/Lindberg/curved `CSBEND` fringe models, reference/FSE correction, magnet misalignments, radiation/stochastic effects, spin, sticky aperture hooks, slice-by-slice tracking, and corrector radiation kicks remain deferred Phase 4 follow-ups.

## Phase 5: Wakes, LSC, And RF Wake Paths

Deliverables:

- Collective effects that use binning, histograms, and convolution are accelerated.

Targets:

- `gpu_track_through_wake`
- `gpu_track_through_trwake`
- `gpu_track_through_lscdrift`
- `gpu_trackRfCavityWithWakes`
- `gpu_track_through_rfcw`
- `gpu_findFiducialTime` only after the CPU/GPU semantics are reconciled

Tasks:

1. Implement GPU time-coordinate computation and bunch indexing.
2. Implement deterministic histogram/bin assignment, with explicit handling for bin-edge particles.
3. Use CUB or custom reductions for charge/current profiles.
4. Use cuFFT only where convolution/FFT cost dominates and data layout supports it.
5. Keep multi-bunch logic and start/end bunch filtering identical to CPU behavior.
6. Compare intermediate histograms and wake potentials in `GPU_VERIFY`, not just final coordinates.
7. Revisit the disabled/simple RFCA comments in `src/simple_rfca.c` and either update semantics or leave forced CPU with a clear reason.

Gate:

- Wake and LSC benchmark outputs match aggregate and per-particle tolerances.
- Timing includes binning, reductions, convolution, kicks, and all transfers.

Phase 5 wrap-up status:

- Phase 5 is wrapped up for the current CUDA scope. The active CUDA implementation covers the conservative `WAKE`, `TRWAKE`, fixed-bin wake, and `LSCDRIFT` slices below; the next active CUDA work should move to Phase 6 CSR bend/drift support.
- First CUDA slice implemented for longitudinal `WAKE` with `N_BINS=0` or fixed `N_BINS>=2`, `SMOOTHING=0`, `BUNCHED_BEAM_MODE=0`, and `CHANGE_P0=0`.
- This slice computes time-coordinate min/max reductions on the GPU, performs deterministic bin assignment, runs the longitudinal wake convolution and energy kicks on CUDA, and compares `Itime` and `Vtime` in `GPU_VERIFY`.
- Added `ELEGANT_GPU_MIN_WAKE_PARTICLES`, `wakes=` timing counters, and the isolated `phase5_wake` benchmark case.
- Verification labels: `gpu-phase5-wake-quick-device`, `gpu-phase5-wake-verify`, `gpu-phase5-wake-trwake-quick`, `cpu-phase5-wake-baseline-60s`, `gpu-phase5-wake-baseline-60s`, and `gpu-phase5-wake-common-194`.
- Timing gate: CPU one-minute baseline used 30000 particles, 194 passes, and 54.96 seconds. CUDA baseline used 30000 particles, 492 passes, and 33.42 seconds. The apples-to-apples 194-pass CUDA run took 13.23 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Second CUDA slice implemented for transverse `TRWAKE` with `N_BINS=0` or fixed `N_BINS>=2`, `SMOOTHING=0`, `BUNCHED_BEAM_MODE=0`, and `TILT=0`. It computes time-coordinate reductions on the GPU, assigns bins on the GPU, computes `posItimeX/Y` with fixed-order per-bin block reductions, convolves both transverse wake planes, and applies transverse kicks without synchronizing after longitudinal `WAKE`.
- `GPU_VERIFY` now compares `posItimeX`, `posItimeY`, `VtimeX`, and `VtimeY` before the final `track_through_trwake` coordinate comparison. Current maximum intermediate differences for the optimized reduction are on the order of `1e-16` or below for the `wake_trwake` quick case.
- `wake_trwake` timing gate: CPU one-minute baseline used 30000 particles, 306 passes, and 48.15 seconds. CUDA baseline used 30000 particles, 638 passes, and 25.37 seconds. The apples-to-apples 306-pass CUDA run took 12.32 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Added `wake_trwake_fixed_bins` for fixed-bin `WAKE`/`TRWAKE` coverage with `N_BINS=1024`. `GPU_VERIFY` compares fixed-bin `Itime`, `posItimeX/Y`, and `VtimeX/Y`; the quick verify case matched intermediate arrays exactly for the exercised bins. The fixed-window benchmark preserves CPU behavior, including late-run TRWAKE warnings if the beam walks outside the fixed bin range.
- Fixed-bin timing gate: CPU one-minute-style baseline used 30000 particles, 319 passes, and 49.87 seconds. CUDA baseline used 30000 particles, 652 passes, and 26.75 seconds. The apples-to-apples 319-pass CUDA run took 13.11 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Third CUDA slice implemented for `LSCDRIFT` with fixed even `BINS>=2`, `LSC=1`, `SMOOTHING=0`, interpolation enabled or disabled, positive `L`, no `AUTO_LEFFECTIVE`, no backtracking, and no frequency cutoffs. This first slice keeps the small impedance FFT on the host for CPU-equivalent math, while GPU kernels handle time binning, current histogram creation, longitudinal kicks, and drift advancement.
- Added `ELEGANT_GPU_MIN_LSC_PARTICLES` and `lsc=` CUDA timing counters. `GPU_VERIFY` compares `Itime` and `Vtime` for each LSC step before the final `track_through_lscdrift` coordinate comparison; the quick verify case matched intermediate arrays exactly for the exercised bins and final coordinates within the existing GPU compare tolerances.
- `lsc` timing gate: CPU one-minute-style baseline used 30000 particles, 303 passes, and 44.67 seconds. CUDA baseline used 30000 particles, 682 passes, and 24.30 seconds. The apples-to-apples 303-pass CUDA run took 10.91 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Deferred Phase 5 follow-ups: smoothed wakes, multi-bunch filtering, tilted `TRWAKE`, `LSCDRIFT` smoothing/frequency filters/backtracking/kick-mode/`AUTO_LEFFECTIVE`, `RFCW`, cuFFT-backed paths where worthwhile, and `gpu_findFiducialTime` semantics.

## Phase 6: CSR Bends And CSR Drifts

Deliverables:

- GPU CSR is correct enough for production opt-in and faster on CSR-heavy workloads.

Targets:

- `gpu_track_through_csbendCSR`
- `gpu_track_through_driftCSR`

Tasks:

1. Port CSR histogramming and wake convolution in small, independently verifiable pieces.
2. Compare `CSR_LAST_WAKE` state in `GPU_VERIFY`, including wake arrays and Stupakov/Saldin mode state.
3. Verify CSR with:
   - small energy spread
   - short and long bunches
   - multiple bin counts
   - drift-after-bend cases
   - CSR plus apertures
4. Keep CPU fallback for modes not yet implemented.
5. Require extensive timing because CSR kernels may be memory-bandwidth bound or FFT-bound depending on settings.

Gate:

- CSR regression cases meet documented tolerances.
- CSR-heavy benchmark demonstrates speedup after including transfer and synchronization overhead.

Current Phase 6 status:

- Phase 6 is wrapped up for the current CUDA scope. The active CUDA implementation covers the conservative `CSRCSBEND` wake-potential slice, CSR timing/threshold controls, reusable CSR scratch buffers, an opt-in fixed-bin histogram experiment, and regression coverage for additional bin counts and bunch lengths; the next active CUDA work should move to Phase 7.
- First CUDA slice implemented for `CSRCSBEND` wake-potential accumulation after CPU-compatible histogramming, filtering, and Savitzky-Golay derivative preparation. This first slice supports the ordinary non-IGF CSR wake equations with steady-state or transient mode and trapezoid or simple integration; particle tracking, histogram setup, filters, `CSRDRIFT`, Stupakov/Saldin drift state, and `CSR_LAST_WAKE` bookkeeping remain CPU-owned.
- Added `ELEGANT_GPU_MIN_CSR_PARTICLES`, `ELEGANT_GPU_MIN_CSR_BINS`, and `csr=` CUDA timing counters. The default bin threshold is 1024 so small-bin CSR cases do not pay transfer and launch overhead unless explicitly requested.
- `GPU_VERIFY` compares the CUDA-computed `T1`, `T2`, and `dGamma` arrays against a CPU shadow for each exercised CSRCSBEND wake calculation. The `gpu-phase6-csr-verify` quick case exercised 32 CSR kernels and matched the final CPU reference for all 4 SDDS files at `1e-11`.
- Added `phase6_csr_csbend`, an isolated large-bin CSRCSBEND benchmark with `BINS=2048`, so timing is not hidden by `CSRDRIFT`/Stupakov CPU work. Quick CPU/GPU SDDS comparisons matched all 4 files at `1e-11`.
- Timing gate: CPU one-minute-style baseline used 20000 particles, 273 passes, and 37.01 seconds. CUDA baseline used 20000 particles, 167 passes, and 18.11 seconds. The apples-to-apples 273-pass CUDA run took 29.87 seconds, ran 4368 CSR kernels, and matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Reusable CSR CUDA scratch buffers now persist across wake calculations and are released from `gpuBaseDealloc`, removing six per-kick `cudaMalloc`/`cudaFree` pairs from the first CSRCSBEND CUDA slice. Correctness remained unchanged: `gpu-phase6-csr-scratch-verify` passed the `GPU_VERIFY` wake-array checks and matched the CPU quick case for all 4 SDDS files at `1e-11`, while the scratch-buffer 273-pass CUDA run also matched the CPU baseline for all 4 SDDS files at `1e-11`.
- Scratch-buffer timing was effectively flat at the full-run level. The saved scratch-buffer baseline used 20000 particles, 171 passes, and 18.41 seconds compared with the previous 167 passes in 18.11 seconds, while the apples-to-apples 273-pass scratch-buffer CUDA run took 29.98 seconds compared with 29.87 seconds before. Treat this as allocator cleanup rather than a meaningful end-to-end speedup; the next CSR speedups need to reduce CPU synchronization and host/device transfer work.
- Added an opt-in experimental fixed-bin CSRCSBEND histogram fill path controlled by `ELEGANT_GPU_ENABLE_CSR_HISTOGRAM=1`. CPU code still owns range determination, bin sizing, filtering, Savitzky-Golay smoothing, and derivative preparation, while CUDA fills the raw coordinate histogram and `GPU_VERIFY` compares it before the existing wake-array checks. The `gpu-phase6-csr-hist-verify` quick case matched the CPU quick case for all 4 SDDS files at `1e-11`.
- CSR histogram timing did not pass the automatic-enable gate because copying the CPU-tracked coordinates to the GPU for each wake setup costs more than the raw histogram fill saves. The histogram-enabled saved baseline used 20000 particles, 167 passes, and 18.35 seconds; the apples-to-apples 273-pass histogram run took 30.42 seconds and matched the CPU baseline for all 4 SDDS files at `1e-11`, compared with 29.98 seconds for the scratch-only CUDA run. Keep this path opt-in until CSRCSBEND tracking or histogram range/binning can stay more GPU-resident.
- Added four CSRCSBEND regression cases for bin-count and bunch-length coverage: `phase6_csr_bins_512`, `phase6_csr_bins_4096`, `phase6_csr_short_bunch`, and `phase6_csr_long_bunch`. The `gpu-phase6-csr-regression-verify` quick sweep matched `cpu-phase6-csr-regression-quick` for all 4 SDDS files per case at `1e-11`, and `GPU_VERIFY` exercised 16 CSR kernels per case with wake-array comparisons enabled.
- Deferred Phase 6 follow-ups: full GPU-resident CSRCSBEND tracking, GPU-resident CSR range/binning that avoids per-kick coordinate transfers, detailed `CSR_LAST_WAKE` state comparison beyond the first wake arrays, `CSRDRIFT` and Stupakov/Saldin modes, CSR plus aperture cases, integrated Greens function, wake-filter GPU paths, and broader production CSR regression coverage.

## Phase 7: Space Charge, Field Maps, And Other High-Cost Physics

Deliverables:

- Additional GPU work is driven by profiling rather than guesswork.

Candidates:

- `SCMULT` and `src/poisson.cc` using cuFFT for Poisson solves.
- Field-map and wiggler tracking if user workloads show they dominate.
- Ion effects if distribution fitting and field calculations dominate real cases.

Tasks:

1. Re-run profiling after Phases 1-6.
2. Port only the next largest measured bottleneck.
3. Prefer library implementations for FFT and dense linear algebra.
4. Add new `GPU_SUPPORT` flags only after the implementation passes correctness and timing gates.

Gate:

- New work must beat the existing CPU path on a representative workload before becoming automatic.

Current Phase 7 status:

- Phase 7 is wrapped up for the current CUDA scope. The implemented work covers the measured high-value `SCMULT` slice that passed correctness and timing gates; the remaining space-charge, Poisson, field-map, wiggler, and ion-effects work is deferred until profiling shows it is the next dominant bottleneck. The next active CUDA work should move to Phase 8.
- Added the first opt-in Phase 7 CUDA slice for linear, unsliced, single-bucket `SCMULT` kicks. The current slice now keeps the common single-bunch `SCMULT` path GPU-resident by using CUDA beam-sum reductions for rms/centroid calculation, a CUDA bunch-index min/max reduction for first-pass single-bucket discovery, and resident `dmux`/`dmuy` accumulation. CPU code still owns SCMULT insertion, nonlinear kicks, sliced kicks, and multi-bunch cases. Enable this experimental path with `ELEGANT_GPU_ENABLE_SCMULT=1`; tune it with `ELEGANT_GPU_MIN_SCMULT_PARTICLES`.
- Added `phase7_scmult_linear`, a stable drift-only benchmark with inserted linear `SCMULT` elements. The `gpu-phase7-scmult-linear-drift-verify` quick run exercised 24 resident SCMULT CUDA kernels, `GPU_VERIFY` matched each linear-kick CPU shadow, and the final GPU output matched `cpu-phase7-scmult-linear-drift-ref` for all 5 common SDDS files at `1e-11`. A follow-up first-pass initialization check with `gpu-phase7-scmult-linear-init-verify` and `gpu-phase7-scmult-linear-init-quick` also matched the CPU reference for all 5 common SDDS files at `1e-11`; the non-verify quick run showed no `initializeSCMULT` or `accumulateSCMULT` CPU synchronization, only final deallocation synchronization.
- Timing gate is met for the isolated drift-only Phase 7 benchmark, but the path remains opt-in until broader production cases are profiled. The 30000-particle CPU baseline scaled to 115 passes and ran in 53.47 seconds with 15.62 seconds reported in `SCMULT`; the current CUDA build ran the same 30000-particle/115-pass case in 16.58 seconds with 6.887 seconds reported in `SCMULT`, `scmult=2760`, `reductions=12881`, and no device-to-host transfer during tracking. CPU/GPU SDDS output for that common baseline matched at `1e-11`.
- Deferred Phase 7 follow-ups: sliced and nonlinear SCMULT, true multi-bunch SCMULT handling, broader production SCMULT profiling, cuFFT-backed `poisson.cc`/Poisson solve work, field-map/wiggler profiling, and ion-effects profiling.

## Phase 8: Pelegant And Multi-GPU

Deliverables:

- MPI execution can use one GPU per worker rank where available.

Tasks:

1. Start with serial `elegant`; enable `Pelegant` only after the serial runtime is stable.
2. Map one GPU per MPI worker rank, honoring `CUDA_VISIBLE_DEVICES`.
3. Ensure `scatterParticles` and `gatherParticles` call `forceParticlesToCpu` until GPU-aware MPI support is explicitly added.
4. Avoid GPU work on ranks with zero particles.
5. Keep load balancing deterministic for regression tests.
6. Add GPU-aware MPI experiments only after CPU staging is correct and profiled.

Gate:

- `Pelegant` CPU and GPU runs compare within tolerance for fixed rank counts.
- Multi-GPU timing improves over CPU-only `Pelegant` for large particle counts.

Current Phase 8 status:

- Phase 8 is wrapped up for the current single-node, single-GPU CUDA scope. The implemented work covers CUDA `Pelegant` build/install support, conservative worker-rank device selection, host-staged MPI particle exchange, no-GPU master-rank behavior, and a fixed-rank Pelegant CPU/GPU correctness smoke. The remaining Phase 8 work requires multi-GPU or cluster hardware to validate meaningfully, so the next active CUDA work should move to Phase 9.
- Added an opt-in CUDA `Pelegant` build path: `make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc` builds in `src/O.Linux-x86_64.gpu.mpi` and installs `bin/Linux-x86_64-gpu/gpu-Pelegant`. `GPU_VERIFY=1` uses the matching `gpu.mpi.verify` object directory and `bin/Linux-x86_64-gpu-verify`.
- CUDA device selection now maps MPI worker ranks onto the CUDA-visible device list when `ELEGANT_GPU_DEVICE` is not set. For true-parallel Pelegant tracking, worker rank `myid=1` uses visible device 0, `myid=2` uses visible device 1 when present, and so on modulo the visible device count; an explicit `ELEGANT_GPU_DEVICE` still overrides this. The no-particle master rank stays in CPU staging mode and does not open a CUDA device.
- The existing `scatterParticles` and `gatherParticles` CPU-staging hooks were kept in place: both call `forceParticlesToCpu` before MPI transfers. GPU-aware MPI remains deferred until the staged path is correct and profiled.
- GPU reductions are disabled under true-parallel Pelegant for now because the CPU reduction paths contain MPI collectives that all ranks must enter together. Read-only CPU reduction fallbacks refresh host coordinates but preserve the current element's GPU eligibility, allowing worker-rank matrix kernels to run between staged reductions without deadlocking.
- `test/gpu_cuda/run_benchmarks.sh` now has `--mpi-ranks N`, which prepends `mpirun -np N` while keeping the existing timeout guard, labels, manifests, and CPU/GPU comparison workflow.
- Fixed-rank Phase 8 smoke status: `cpu-phase8-pelegant-matrix` and `gpu-phase8-pelegant-matrix` used 2 MPI ranks, 2000 particles, and 5 passes. CPU real time was 0.64 seconds; CUDA Pelegant real time was 0.73 seconds and rank 1 reported `matrix=320` with rank 0 in CPU staging. The 4 common SDDS files matched at `1e-11`.
- Deferred Phase 8 follow-ups: larger Pelegant timing cases, true MPI-aware GPU reductions, GPU-aware MPI particle exchange, deterministic load-balancing validation with multiple GPU worker ranks, one-GPU-per-worker timing, multi-GPU speedup validation, and multi-node/cluster measurements on hardware with more than one CUDA device.

## Phase 9: Documentation, CI, And Release Controls

Deliverables:

- Users can build, run, verify, and diagnose CUDA support.

Tasks:

1. Document build requirements:
   - supported CUDA versions
   - supported GPU architectures
   - build variables
   - runtime controls
2. Document known CPU fallbacks and unsupported elements.
3. Add CI jobs:
   - CPU-only build
   - CUDA compile-only job if GPU runners are unavailable
   - CUDA correctness/timing job on GPU runners
4. Add a release checklist:
   - all deterministic GPU tests pass
   - stochastic tests pass distribution checks
   - no CPU-only performance regression
   - GPU speedups are documented by benchmark

Current Phase 9 status:

- Added `doc/CUDA_GPU_SUPPORT.md` as the primary CUDA user and release-control guide. It documents the known-good local CUDA setup, `NVCC=/usr/local/cuda-12.4/bin/nvcc`, `CUDA_ARCH`, build commands for `gpu-elegant` and `gpu-Pelegant`, runtime controls, known CPU fallbacks, correctness checks, one-minute CPU-targeted timing gates, and the release checklist.
- Added `test/gpu_cuda/ci_smoke.sh`, a CI-friendly wrapper for CPU builds, CUDA compile-only builds, `GPU_VERIFY` builds, quick CPU/GPU comparisons, one-minute-style baselines, and fixed-rank Pelegant smoke tests. It supports `--dry-run` so CI commands can be reviewed before running expensive jobs.
- Linked the CUDA support guide from the root `README.md` and `test/gpu_cuda/README.md`, and added short CI/release command examples to the benchmark README.
- Updated the top-level `make` flow so a simple `make` always builds the CPU executables, automatically adds `Pelegant` when MPI C and C++ compiler wrappers are available, automatically adds `gpu-elegant` when `nvcc` and `libcudart` are found, and automatically adds `gpu-Pelegant` when both MPI and CUDA build requirements are available. Use `CUDA_AUTO=0` to skip automatic CUDA builds.
- Phase 9 follow-ups: wire these commands into the repository's chosen CI service, add a real GPU-runner job once runner hardware is available, add stochastic distribution-test tooling before enabling stochastic CUDA paths, and automate benchmark report generation for release notes.

## Definition Of Done For Each GPU Feature

A feature is complete only when all of the following are true:

- CPU-only build and CPU output are unchanged.
- GPU build succeeds with CUDA enabled.
- Runtime fallback works when the feature is disabled or unsupported.
- `GPU_VERIFY` passes for the feature.
- End-to-end SDDS tests pass with documented tolerances.
- Timing shows benefit on at least one representative case.
- The automatic GPU threshold avoids slowdowns on small cases.
- Any numerical differences are documented in the benchmark report.
