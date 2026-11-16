# GPU parity validation

For output validation without a performance claim, add `--correctness-only`
to each `baseline` command. This explicitly selects zero warm-ups and one run
per test, permits `--jobs 8`, and records that timing gates are unavailable.
`compare-existing` retains the numerical and CUDA-activity checks and preserves
a failed coverage report even if a case ran entirely on CPU. Do not combine
these artifacts with `--minimum-speedup` or `--pre-change-gpu` timing gates.
The default serial timing workflow in `test-set/README.md` is unchanged.

The current `frequency-map-batched-heavy` implementation explicitly selects its
CPU history backend. It can validate outputs but shows no CUDA activity, so
`test-set/suite.json` exempts it from the CUDA-activity gate and `comparison.txt`
names it on every run. Keep this limitation visible until native histories are
certified; a successful process exit alone is not GPU coverage.

## Tracking fixes (September 2026)

CPU source is unchanged. The reference implementation was inspected in
`matrix.c`, `multipole.h`, `csbend.c`, `limit_amplitudes.c`, `track_rf.c`
and the LSC routines;
the changes are in `gpu_cuda_runtime.cu` and `gpu_stub.c`.

- **SPEEDBUMP:** an unsigned direction selects both sides. The CUDA circle-line
  intersection now uses the CPU square-root/square evaluation order as well.
- **Third-order matrices:** compute the coordinate product before multiplying
  by the coefficient, including the CPU zero-coordinate skips. Algebraic
  reassociation was causing differences in sensitive downstream statistics.
- **Multipoles:** honor `TURBO_RECIPROCALS` from `manual.h`. The default CPU path
  divides after multiplying the slope by momentum. Multiplication by a rounded
  reciprocal is different. Dipole conversions have separate helpers because
  their CPU implementation deliberately uses reciprocal multiplication.
- **Finite collimators:** reproduce entrance loss removal and exit loss removal
  as two separate swap-with-tail permutations. A single union loss mask had the
  right survivor count but reordered loss and accepted-particle rows. The map
  is constructed on the device, then particle scattering runs in parallel.
- **RFCA:** reference-sensitive output uses the CPU fiducial-time calculation
  and host sine evaluation. CUDA constructs each phase and applies each kick.
  Reference-kick buffers are reused and released with the existing RF scratch;
  the phase/kick transfers are timed and usage counters identify this hybrid
  path. This does not move the complete cavity tracking to CPU.
- **LSC:** reference-sensitive output uses the CPU summation order for beam
  size. Beam radius determines impedance and adaptive step size, so a change
  in the reduction can change every subsequent kick. Binning, drifts and
  voltage kicks remain on CUDA. Existing CPU FFT/voltage preparation is retained.

### Remaining fallback and explicit controls

Non-expanded CSBEND sector drifts multiply a small difference by the bend radius, so one
floating-point step in the drift's angle becomes a position difference. Ordinary
reference-sensitive output used to keep these bends on the CPU for that reason. That rule was
removed on November 5, 2026, after a biased CUDA function in these drifts was replaced; see "GPU
correctness items" below. These bends now run on the GPU in default mode, and they match the CPU
to rounding size instead of exactly. `ELEGANT_GPU_ENABLE_CSBEND_DRIFT=0` keeps them on the CPU.
Distributed MPI runs keep them on the CPU unless that setting is explicit.

`ELEGANT_GPU_RFCA_REFERENCE_MATH=0` and
`ELEGANT_GPU_LSC_REFERENCE_VARIANCE=0` opt out of the corresponding reference
preparation; 1 explicitly enables it. Unset means automatic selection for
reference-sensitive output. Neither setting is needed by the 23 legacy cases.
Distributed MPI LSCDRIFT uses its existing CPU collective implementation. The
CUDA local histogram path does not implement the corresponding MPI reductions;
allowing workers into that path while the master enters CPU collectives hangs.
The eligibility guard is compiled only for MPI. Single-process LSC is unchanged.

Particle thresholds are unchanged. No momentum-aperture batching changes from
the earlier archived proposal are included.

## Numerical acceptance

Keep the exact comparisons and the existing legacy screen
`abs(a-b) <= max(1e-15, 1e-12*max(abs(a),abs(b)))`.
The companion suite retains its existing relative screen of 1e-9.
Do not globally increase the absolute tolerance: a tolerance suitable for a
position in metres can be inappropriate for a time in seconds.

The remaining `trajCorrect2` changes are reviewed separately. This fixture
repeatedly corrects a trajectory using 1e-6 corrector perturbations and changes
reference momentum at every element. Near-zero corrected centroids are poor
candidates for a relative-error requirement. A case-specific review accepts
absolute differences up to 1e-14 for its affected positions/slopes, correction
kicks and dimensionless moment statistics; 1e-12 for fractional-momentum extrema
and percentile widths; 2e-13 m for affected longitudinal extrema; and 1e-13 for
affected response-matrix entries. These are a review of this fixture, not a
general error bound for tracking or search decisions. Time fields retain the
original tolerance, as do every unlisted field, all discrete data, schemas,
file sets and row ordering. No output is ignored. The raw strict comparison
continues to flag this case; the separate review records each exception and
its measured maximum, without changing the harness defaults.

### Approval, evidence and limits of the conclusion

The user explicitly approved this reasoning on September 16, 2026. The exact
34-field policy is preserved in
[trajCorrect2-review-policy.json](trajCorrect2-review-policy.json), alongside
this documentation so it remains available without the local benchmark disk.

The observed maxima were 3.67e-15 m for horizontal centroid position, 2.94e-15
for horizontal slope, 1.54e-15 rad for corrector kick, 1.07e-13 m for affected
longitudinal extrema, 6.69e-13 for fractional-momentum extrema/widths, and
4.54e-14 in native units for the affected transfer-matrix entries. Candidate
`trajCorrect2` outputs were identical to the original GPU baseline: these
differences predated the GPU fixes.

These limits were selected after inspecting the results. They express an
engineering judgment about this fixture's physical sensitivity, not an
independently derived numerical error bound. Not every discrepancy was traced
conclusively to its originating arithmetic operation. Do not describe this as
passing the original strict tolerance or as proof that every discrepancy is
roundoff. The raw strict comparison still fails; the separate review accepts
the explicitly named differences. Future changes outside the named limits
require a new investigation, not automatic tolerance expansion.

Local evidence was in
`GPU-Testing/momentum-nonfiducial-20260915/parity-final/`: the raw
`cpu-comparison/`, `original-gpu-comparison/`, `reviewed-acceptance.json`, and
`review_trajcorrect.py`. No additional outputs were ignored. That directory was
lost in November 2026; the reviewed policy remains in
[trajCorrect2-review-policy.json](trajCorrect2-review-policy.json).

## Reproduction

Build with `make -C src -j8 HAVE_CUDA=1`. Also check CPU with `make -C src -j8`,
verification with `make -C src -j8 HAVE_CUDA=1 GPU_VERIFY=1`, and GPU/MPI with
`make -C src -f Makefile.mpi -j8 HAVE_CUDA=1`.

For the 23 legacy cases use the preserved CPU manifest's case list, freeze the
candidate executable, and run the existing regression harness `baseline`
command with `--test-set elegantTestSet --jobs 8 --timeout 3600`. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and leave
`ELEGANT_GPU_*` unset. Compare with `compare-existing --baseline CPU_ARTIFACT
--candidate GPU_ARTIFACT --output COMPARISON_ARTIFACT`.

For companion output coverage use `baseline --test-set src/gpu/test-set
--jobs 4 --correctness-only --timeout 3600` for the GPU and a fresh preserved CPU
baseline. Suite GPU settings are recorded in the manifests. Parallel run times
are not evidence of speedup or non-regression.

Full logs, frozen binaries, comparisons and manifests were removed in November
2026; the notes and source diffs are kept in
`GPU-Testing/history/2026-09-15_momentum-search-nonfiducial/`.

## Subsequent full-suite check (September 16, 2026)

The fresh 668-case legacy and 39-case companion CPU/GPU rerun found a new
`twissElem5` regression in the recent GPU changes. `run.cen` and `run.sig`
differ (Cdelta maximum absolute difference 1.55e-6), while the final particle
file matches. Isolated reruns reproduce this with the current GPU binary; the
preserved pre-fix GPU binary matches CPU. This discrepancy is **not approved**
under the trajCorrect2 policy. It was subsequently fixed as described below.
The earlier 23-case results alone did not establish full parity.

Legacy results: 657 identical, 8 within existing roundoff, trajCorrect2 accepted
under its existing policy, twissElem5 unresolved, sddsBeam2 unavailable on both.
All 39 companion outputs meet existing tolerances; the known frequency-map
CUDA-activity gap remains. See
`GPU-Testing/history/2026-09-16_full-parity/REPORT.md`.

### twissElem5 RF synchronization fix

RF fiducial preparation called `forceParticlesToCpu`, which marks host data
mutable and clears `elementOnGpu`. The RF kick then updated device coordinates
without restoring that dispatch flag. End-of-element centroid and sigma output
therefore read the pre-kick host coordinates. Final particle output synchronized
independently, explaining why it was already correct.

`gpu_findFiducialTime` now uses `copyParticlesToCpuReadOnly`: the CPU reference
calculation only reads coordinates, so this preserves device validity and the
GPU dispatch flag. Subsequent output synchronizes the post-kick coordinates.
No CPU code, physics formulas, particle thresholds or tolerances changed. RF
phase preparation retains CPU arithmetic; phase construction and particle kicks
remain CUDA operations. `twissElem5` is the existing regression fixture for this
end-of-line RF/output interaction; include all five output files when comparing.

The isolated corrected run matches all five CPU output files exactly under the
existing metadata exclusions. The report and before/after patches are in
`GPU-Testing/history/2026-09-16_twissElem5-rf-sync-fix/`; the detailed validation
outputs and frozen binary were removed in November 2026. The historical full-suite report
above is retained unchanged; this focused correction is not another full-suite
rerun.

The nine focused legacy cases (twissElem1–5, modulate4, twoCavityMoments1,
acSextupole1 and branch4) all match fresh CPU outputs exactly. RFCA-heavy
passes its unchanged companion tolerance. GPU_VERIFY twissElem5 and CPU,
GPU, GPU_VERIFY and GPU/MPI builds pass; MPI runtime was not rerun.

## RFCW LSC energy fix (November 3, 2026)

With `ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT=1` and
`ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT=1`, an RFCW with LSC received about twice
its energy change under `first_is_fiducial` or `always_change_p0`. NLSJitter1 gave
`pCentral` 8561 instead of 4405. Default settings were not affected in any case
tested.

The LSC reference beam-size step used a mutable CPU synchronization, which cleared
`elementOnGpu` while tracking continued on the device. `do_tracking` then ran
`do_match_energy` on stale host coordinates with the updated `P_central`.
`gpuLscCenteredVariances` now uses `copyParticlesToCpuReadOnly`. CPU source is
unchanged.

After the fix, NLSJitter1 `pCentral` matches CPU to 5e-12 relative with all three
wake and RFCW opt-ins. The 425-test routine list is identical to the pre-fix GPU
reference, and of 43 opt-in tests only NLSJitter1 changed. CPU, GPU, GPU_VERIFY and
GPU/MPI builds pass; MPI runtime was not rerun.

GPU_VERIFY couldn't check an RFCW with LSC while this synchronization was active: the
mid-element copy overwrote the host input that its CPU check re-tracks, so the check
reported a false mismatch. Fixed on November 5, 2026: in that build the variance is taken
from a separate copy.

Notes and the test decks' location are in
`GPU-Testing/history/2026-11-03_wake-rfcw-optin-testing/REPORT.md` and
`GPU-Testing/cases/rfcw-lsc/`.

## Feature suite (November 4, 2026)

`feature-set/` holds 138 small cases, each guarding one GPU feature or fallback,
with the WAKE, RFCW and RFCA change-p0 opt-ins on and CSR forced on. See
`feature-set/README.md`. It uses the same 1e-15 absolute and 1e-9 relative screen
as the companion suite; no tolerance was raised to admit a result.

Compared with CPU on `sm_120`: 148 within the screen, 9 flagged, listed in that
README. Open, except as noted:

- `csr-saldin54` differs by `|dp|` 4.2e-3 at `p` of 11742 with CSR forced on. This
  isn't a GPU bug: on the CPU alone, a 1e-13 change in `sigma_s` moves `p` by 2.3e-3,
  because `CSRDRIFT,USE_SALDIN54=1` amplifies roundoff. The first Saldin table's inputs
  and results match exactly between CPU and GPU.
- Some global loss errors are the same on CPU and GPU, so this comparison can't see
  them. For example, `SCRAPER` with a length reports `Z` mirrored within the element,
  and `KQUAD` and `KSEXT` with `TILT` or `DX` report `X` in the element's frame.

A CPU bug it found is fixed: `ukickmap.c` applied its radiation block outside the
`else` of the loss test, so each loss gave the particle swapped into that slot an
extra kick, and the GPU differed from CPU by whole numbers of kicks. With the block
inside the `else`, CPU and GPU agree exactly. No elegantTestSet result changed.

Two CPU global loss errors in `limit_amplitudes.c` are also fixed, and the CPU now
agrees with the GPU and with positions known by construction. `SPEEDBUMP` computed `X`
before moving the particle to the hit point. `TAPERAPR` with `TILT` or `DX` computed
`X` and `thetaX` in the element's rotated and shifted frame. `TAPERAPE` and `APCONTOUR`
have the same frame error and run CPU code in `gpu-elegant`.

`&obstruction_data` isn't implemented on the GPU. A run with obstruction data loaded
stays on the CPU, including batched searches, and `ELEGANT_GPU_MODE=required` stops with
an error. Outputs match the CPU exactly for tracking, `&find_aperture` and
`&momentum_aperture`. `&frequency_map` and `&tune_footprint` also stay on the CPU, even
though they track with `TEST_PARTICLES`, which turns obstructions off on the CPU.

`KQUAD`, `KSEXT` and `KOCT` multipole files run on the GPU. The CPU and GPU share
`setupKquadErrorMultipoles` and the matching functions in `multipole.c`, so files are read
and random errors drawn in the same order. Kicks use the CPU's expansion coefficients,
loaded into a device table. `GPU_VERIFY` shows differences from the CPU of exactly 0 per
element, or 1.8e-15 with misalignment and radiation. A magnet uses the CPU if a file term
has order above 30 or the magnet has more than 96 nonzero terms.

`DQCOR` files run on the GPU the same way, through `setupDqcorErrorMultipoles`. `CCBEND`
files run on the GPU through `setupCcbendErrorMultipoles`, with edge and body kicks where
`integrate_kick_KnL` applies them. `GPU_VERIFY` shows differences from the CPU of at most
1.7e-18 per `CCBEND` and 8.9e-16 per `DQCOR`, the second with `DZ`. A `CCBEND` with more
than 96 nonzero terms uses the CPU.

`CCBEND` runs on the GPU with `FRINGEMODEL=0`, the default, as well as -1 and 1. The host
computes the three coefficients of the `y'` kick at each edge with the arithmetic of
`verticalRbendFringe`, and the kernel applies them. `GPU_VERIFY` shows differences from the
CPU of at most 1.4e-17 per bend.

The kernel leaves `s` and `p` unconverted in the row of a particle lost to undefined
slopes, so `GPU_VERIFY` compares only survivors after a loss. The normal build doesn't read that row:
with loss output, the magnet falls back to the CPU, and `run.los` matches exactly in
`magnet-multipole-files-undefined-slopes`.

A CPU bug found this way is fixed: the multipole file store in `multipole.c` matched on
file name only, although steering files are normalized and allocate strengths when read.
Using one file for both steering and another role crashed or silently gave wrong results,
depending on the order. Elements sharing a steering file also shared its strength arrays,
so the last factor computed applied to all of them, and now each has its own arrays. CPU
and GPU share this code. Order limits were checked against the body file's orders, for
edge and steering files too, and correctors with no random file ignored
`STEERING_MULTIPOLE_FACTOR`. Correctors with multipole files stay on the CPU.
`tests/test_multipole_files.py` guards all of these.

A particle already past the slope or coordinate limits at a magnet's entrance now reports
its loss at the magnet's start. `integrate_kick_KnL` and `integrate_kick_multipole_ordn`
used to return before clearing the loss offset, so it reported what the previous particle
left, on the CPU and in the batched GPU search alike. `tests/test_loss_position.py` and
`search-aperture-entrance-loss` guard this.

`MALIGN_METHOD` 1 and 2 run on the GPU for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR` and `CSBEND`.
The kernels apply the matrices from `getLinearizedMisalignmentMatrix` in `malign_mat.c`
with the same arithmetic as the CPU's particle loop. `GPU_VERIFY` shows differences from
the CPU of at most 1.8e-15 per magnet and 1.6e-14 per bend. Methods 3 and 4 are below,
under "CCBEND options and exact misalignments". Lost rows aren't untransformed or reordered like the CPU's, so `GPU_VERIFY`
compares only survivors after a loss. The normal build doesn't read those rows, and with
loss output the magnet falls back to the CPU.

`CWIGGLER` with `BY_FILE` and `BX_FILE` runs on the GPU, in normal and split-pole form.
The kernel keeps the order of operations in `gwig.c`, and reads the harmonics from a
device table that is uploaded only when its contents change, because the wavenumbers
depend on the period length. `GPU_VERIFY` shows differences from the CPU of at most
1.0e-17 per wiggler in `feature-set`. `TILT` also runs on the GPU, with files or
sinusoidal, with or without spin. So do constant fields, pole factors, `DX`, `DY` and `DZ`.
`ISR`, `FIELD_OUTPUT`, `GAP` and `TGU` have run on the GPU since November 6, 2026. The CPU field
code tests `TGU_GRADIENT` rather than the `TGU` flag, and so does the kernel. Before that move, the
GPU ignored the gradient: with `TGU_GRADIENT=2`, `Cx` was 6.3e-11 instead of the CPU's -1.27e-7.

The exact correctors can show a larger difference from the CPU for a rare particle:
`x += rho * (cos(alpha) - cos(theta + alpha))` turns one floating-point step in a cosine
into 1.1e-16 times `rho`, which is 3.7e-14 m for the 330 m radius in
`gputest1-resident-magnets-focused`. Other particles there agree to 7e-18.

GPU bugs fixed on November 4, 2026, each with a case in `feature-set/` except the
last, and all present in the previous binaries:

- `&particle_tunes` read stale host coordinates when nothing else synced the beam,
  which gave every particle `nux=2.0` and `Jx=1.8e308`.
- Batched `&frequency_map` and `&tune_footprint` ignored `always_change_p0=1`, which
  gave wrong tunes off momentum. These runs now use the unbatched path.
- GPU CSBEND and CSRCSBEND applied `coordLimit` and `slopeLimit`, which `csbend.c`
  doesn't have, and dropped particles with a slope above 1.0.
- SCMULT moments summed raw squares, which lost the bunch length to roundoff far
  along a machine. A 1 mm bunch at `s` of 1e6 m made the run exit. The sums are now
  relative to the first particle, and a beam size the kicks can't use falls back to
  the CPU.
- An exact corrector recorded a slope above 1 as lost at its entrance, with zero
  global loss coordinates. It now matches the CPU, at the exit.
- A binary with no kernels for the card now falls back to the CPU in `auto` mode and
  exits with a clear message in `required` mode. No suite covers this. To check, run
  a `gpu-elegant-sm_*` built for a different card.

`output` or `final` in `&run_setup` keeps LSCDRIFT on the CPU, with no override,
and keeps CSR there unless a CSR setting is explicit. Check the `lsc` and `csr`
counters before treating a deck as coverage of those paths. GPU activity counts any
element, so also check the counter for the element a case is named for:
`lgbend-deterministic-heavy` passed the activity check for months with LGBEND on
the CPU.

The extra `test-set` decks (`--include-extra-decks`) compared with CPU: of 102
tests including the 39 main cases, 42 identical, 57 within the screen, 3 flagged.
The 3 are `gputest1-resident-magnets-focused` CSBEND decks, by 9.2e-14 m or less.
`mode2.ele` is excluded and still needs a CPU `Pelegant` comparison; on `gpu-elegant`
it is identical with the GPU on, with `ELEGANT_GPU_MODE=off`, and with
`-ompThreads=4`.

## Coverage (November 4, 2026)

`make -C src HAVE_CUDA=1 GPU_COVERAGE=1` builds a gcov binary in
`bin/<arch>-gpu-coverage`. It sees host code only, not inside device kernels. Across
the feature set, the suite and its extra decks, the elegantTestSet routine and opt-in
lists, default `auto` mode and the operating modes, it reaches 83.4% of the lines and
366 of 408 functions in `gpu_stub.c`. Most of what remains is failure handling,
`GPU_VERIFY`-only code and code that is disabled, such as CENTER on the device.

To cover what gcov can't see, a differential sweep set each option of each GPU element
in turn and compared CPU with GPU: 683 decks over 55 element types, 612 of them
compared, 604 matching. Only 303 of the 612 options changed the CPU result, so the
others prove little, and it tests one option at a time with one value.

## CSRDRIFT wake and WATCH el (November 5, 2026)

Two GPU results differed from the CPU and are fixed:

- A resident `CSRCSBEND` converted final coordinates on the device before the preparation
  for a following `CSRDRIFT`, which bins `c*t`, so the wake was placed in the wrong position
  for any drift without `USE_STUPAKOV=1`. The device now converts only when no preparation
  follows. `bunchComp1` in default mode had ended with `Sdelta` 6.36e-4 against 1.02e-3.
- A parameter-mode `WATCH` using GPU sums computed `el` from `sigma[5][6]`, which sparse
  sums leave 0, so it reported `St * Sdelta`. The GPU now computes that term separately.
  `sigma[5][6]` stays 0, matching the CPU's sparse sums.

`GPU_VERIFY` caught neither at the time. It had no check for `el`, and `compareCSR_LAST_WAKE`
compares `s0` and the other scalars with themselves. It now has both checks; see "GPU correctness
items" below. The guards are `csr-saldin54`,
`csr-drift-plain-wake`, `csr-drift-overtaking-wake` and `watch-parameter-el` in
`feature-set`.

`GPU_VERIFY` needs `ELEGANT_GPU_VERIFY=1` at run time as well as the build flag. Without it,
the checks under `gpuBase.verifyMode` in `gpu_stub.c` don't run: CSR resident, `RFMODE`
family, `TFBDRIVER` and histogram.

A CPU nudge as large as a GPU difference doesn't show the difference is noise. `csr-saldin54`
was first explained that way: its GPU difference was the size of the CPU's, but `EDGE_ORDER=2`
dropped the GPU difference from 4.2e-3 to 1.1e-10 while the CPU's stayed 2.3e-3. Back such a
verdict with element comparisons.

## GPU_VERIFY after a loss (November 5, 2026)

After a loss, the `multipole_tracking`, `multipole_tracking2` and `track_through_csbend`
checks compare only survivors. Lost rows can't match, and the kernels reproduce the CPU's
survivor order, so rows are compared as they stand.

A caller must not run its CPU shadow when `getElementOnGpu()` has been cleared. That means
a fallback or the loss path has already tracked the particles on the CPU, so a second run
would track them twice, and `compareGpuCpu` would return early without reporting it. On
`search-aperture-entrance-loss` this gave a wrong `Area`. Callers of the magnets, `RFCW`,
`LSCDRIFT` and `POLYNOMIALSERIES` are guarded.

To check a `GPU_VERIFY` run for this, compare its outputs with the CPU reference. They
should be within the usual envelope.

## GPU correctness items (November 5, 2026)

**`CSBEND` drift.** CUDA's `asin` and `atan` are biased for small arguments; `sin` and `cos`
aren't. Against glibc on 4 million arguments:

| Function, range | Results that differ | Mean difference |
| --- | --- | --- |
| CUDA `asin`, 0.01 to 0.1 | 35.8% | +0.335 ulp |
| CUDA `asin`, 0.1 to 0.125 | 60.1% | -0.601 ulp |
| CUDA `atan`, 0.01 to 0.1 | 11.7% | +0.117 ulp |
| CUDA `sin`, `cos`, 0.01 to 0.1 | 0.05%, 0.08% | about 0 |
| `gpuAccurateAsin`, 0.01 to 0.1 | 0.29% | +0.003 ulp |
| `gpuAccurateAsin`, 0.125 to 0.4 | 1.96% | +0.018 ulp |
| `gpuAccurateAtan`, 0.01 to 0.1 | 0.28% | +0.0003 ulp |

An unbiased rounding difference averages out over turns, but a biased one builds up.
`gpu_accurate_trig.h` carries the leading term in extra precision. The `CSBEND` drifts and the
exact correctors use it, in the normal build as well. The test-set cases are at most 6% slower.

On `track1`, forced, `Cx` at 8 cm after 1,024 turns:

| Page | GPU before | GPU now | CPU's own libm path |
| --- | --- | --- | --- |
| 0 | 2.2e-13 | 2.2e-13 | 2.4e-13 |
| 5 | 2.2e-12 | 1.5e-13 | 3.4e-13 |
| 10 | 1.8e-11 | 9.5e-13 | 3.9e-13 |
| 13 | 4.1e-11 | 1.7e-12 | 1.8e-12 |
| 15 | 5.5e-11 | 6.7e-13 | 8.4e-13 |

The last column is what the CPU's answer moves when glibc takes another path
(`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-FMA4,-AVX`). Exact parity with one libm isn't
meaningful below that size.

The difference still grows with turns, as the CPU's own libm difference does. Fitted over
turns 64 to 1,024, on the largest-amplitude pages:

| Page | GPU before | GPU now | CPU's own libm path |
| --- | --- | --- | --- |
| 13 | turns^1.74 | turns^1.26 | turns^1.35 |
| 15 | turns^1.80 | turns^1.58 | turns^1.47 |

Page 10 is the exception: turns^1.40 now, against turns^0.81 for the CPU's libm path.

**New `GPU_VERIFY` checks.** All had the old bug put back to confirm that they fire:

- `CSRDRIFT preparation input` compares the `c*t` the wake uses with the CPU's. It fires on
  `csr-saldin54`, `csr-drift-plain-wake` and `bunchComp3`.
- `CSRCSBEND resident kick` compares the kick on `delta`. It is new, and nothing had checked it.
- `WATCH el` compares `el` with `rms_longitudinal_emittance`. It fires on `watch-parameter-el`.

`WATCH el` didn't run at first. `compareGpuCpu` leaves only the host current, so a `WATCH` never
counted as on the GPU in this build, and `compareReductionArrays` reset the flag for passive
elements. In `GPU_VERIFY` builds a `WATCH` now refreshes the device and takes the GPU path. A
check isn't proven until it has been seen to fail.

Its limit is `ELEGANT_GPU_WATCH_EL_COMPARE` (1e-12) plus `DBL_EPSILON * |t| / St`, relative to
`St^2 * Sdelta^2`. Each time carries rounding of about `DBL_EPSILON * |t|`, so far along a
machine the CPU's and GPU's sums can't agree better than that. The second term came from that
reasoning and was then tested: across 245,277 checks on 23 decks the largest difference is
1.2e-7, on `ramprf1`, where that term is 9.0e-7. None exceeds the limit. The old bug measures
0.64, so a real error stays visible unless `t` and `delta` are almost uncorrelated.

**Other `GPU_VERIFY` stops.** `RFCW` no longer overwrites its own input, as above. The exact
corrector check now pairs rows by particle ID, because the CPU leaves `NaN` rows for
`do_tracking` to remove. `POLYNOMIALSERIES` no longer tracks twice after its fallback.

**`LSCDRIFT` and `RFCW` LSC with no time spread.** `LSCDRIFT` used to exit in required mode with
a misleading message. It now finishes the element on the CPU, in the normal build too. `RFCW` LSC
exits with a clear message, because the CPU also exits for identical particles.

**Not exercised:** the `POLYNOMIALSERIES` fallback, because below the threshold the element stays on the
CPU and never reaches it. The `do_match_energy` guard is for MPI only and isn't compiled in
these builds. `gpu-Pelegant` wasn't built. Only `sm_120` was run.

**Removing the `CSBEND` rule.** Without the rule, 8 of the 660 default decks move bends onto the
GPU: `ERL1`, `LCLSII-1`, `LCLSII-2`, `branch4`, `bunchComp6`, `csbend7`, `csbendAp1` and
`csbendAp2`. All 660 pass. Against the CPU: 644 identical, 10 roundoff, 6 flagged, which are the
first 5 of those decks and `trajCorrect2`. Those 5 flags aren't proof of noise on their own, but
259 of 260 flagged columns are within 3 times a CPU nudge of `p_central`, and `GPU_VERIFY` passes
800 bend comparisons, largest 4.5e-13. The exception is `x` in `LCLSII-2` `run.w3`, 1.3e-15 m, which
bend `BCXH2` shows directly at 1.25e-15. `ERL1` is 2.7 times faster, and `bunchComp6` 1.75 times.
Details are in `history/2026-11-05_csbend-rule-off/REPORT.md` in GPU-Testing.

## CCBEND options and exact misalignments (November 5, 2026)

`CCBEND` now runs on the GPU on its first pass, and with `K3` to `K8`, `TILT`, `YAW`, `DX`, `DY`,
`DZ`, `ETILT`, `EPITCH`, `EYAW` and `MALIGN_METHOD` 0 to 4. The CPU still finds the FSE and `x`
offset, on its own reference particle: `gpuPackCcbendTracking` calls `optimizeCcbendForGpu` with
the on-GPU flag cleared. A one-pass line of 20 `CCBEND`s and 200,000 particles takes 0.52 s,
against 6.86 s on the CPU.

`MALIGN_METHOD` 3 and 4 run on the GPU for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, `CSBEND` and `CCBEND`.
`getExactMisalignmentFrame` in `malign_mat.c` gives each face's axes, wedge angle and drift length
from the CPU's own code, and `gpuApplyExactMisalignment` runs the per-particle loop.

`GPU_VERIFY` shows differences from the CPU of at most 1.7e-16 per `CCBEND` and 1.0e-17 per
`CWIGGLER` in `feature-set`. Deliberate mistakes in each feature were caught by its cases, except
`search-frequency-map-ccbend-options`, `search-aperture-ccbend-options` and
`magnet-ccbend-tilt-loss`. Those can't show a kernel mistake: the batched frequency map tracks on
the CPU, `find_aperture` is below the `CCBEND` threshold, and loss output falls back to the CPU.
The 660 and the routine list stay identical because their `CCBEND` decks have too few particles to
reach the GPU, so check this work with `feature-set` and the forced `ccbendSoftFringe1`, `2`,
`errorSampling1` to `3` and `lgbend2`.

The CPU applies `ETILT` alone at the exit but not at the entrance, because the entrance test leaves
it out. The kernel does the same.

`CWIGGLER` `DX`, `DY`, `DZ` also run on the GPU. They first needed a CPU fix: `GWigSymplecticPass.c` and
`apple.c` didn't set `malign.startPID` and `endPID` to -1 for the exit call to `offset_beam`, so the exit
offset applied only to particle ID 0 and particles left the element still displaced. Both elements now match
an explicit `MALIGN` before and after them exactly. See `history/2026-11-05_offset-exit-check/REPORT.md`.

See `history/2026-11-05_ccbend-options/REPORT.md` in GPU-Testing.

## More options on the GPU (November 6, 2026)

These now run on the GPU, in plain tracking only. Batched programs refuse the first four, through
`gpuElementNeedsPlainTracking`.

| Element | Options |
| --- | --- |
| `CCBEND` | `ISR`, `CENTROID_OUTPUT_FILE`, spin, `&aperture_data` files, `APCONTOUR` |
| `CWIGGLER` | `ISR`, `FIELD_OUTPUT`, `GAP`, `TGU` and `TGU_GRADIENT` |
| `LGBEND` | `DX`, `DY`, `DZ`, `ETILT`, `EPITCH`, `EYAW`, `TILT`. The CPU always uses the exact body-centered method for these |

`GWigRadiationKicks` skips a random number where the field is exactly zero. That depends on `z` and the
pole factors but not on `x` and `y`, so `GWigCountRadiationKicks` counts them once per call.

`ISR` first used a table of random numbers from a host copy of the generator. "ISR random numbers on the
device" below replaces it.

**Centroid and field records.** Random numbers and records go through the device in batches of about
256 MB. The host sums centroids in the CPU's order, and `FIELD_OUTPUT` rows go through the CPU's own
`GWigAddToFieldOutput`. `FIELD_OUTPUT` gains little speed, since writing each row dominates. Under
`GPU_VERIFY`, the CPU's second pass writes these files.

**A pointer swap gave wrong GPU results.** `POLYSERIES` and `KSBEND` removed lost particles by swapping row
pointers. Host and device copies treat the rows as one block in order, so lost particles came back:
`KSBEND` kept 239 of 500 particles in the release `gpu-elegant` where the CPU kept 152. Both now use
`swapParticles`, with no change to CPU results. See bug 10 in `elegant-CPU-bugs-and-fixes.md`.

**Which cases can't show a kernel mistake.** `magnet-ccbend-isr-loss` and `magnet-ccbend-centroids-loss`
always fall back to the CPU, so they test only the fallback. `search-frequency-map-prepass-losses` tests the
fused prepass only under `GPU_VERIFY`, because the normal build returns before running it.

`gpu_copy_csbend_csr_dgamma` is the one function in the paths named for coverage that no test reaches. By
reading, the last slice always copies `dGamma` to the host, so it runs only if the resident kick fails after
the wake was computed.

**Default 660.** `cwigglerMoments2` now runs its `ISR` wigglers on the GPU and flags against the CPU, with
`alphaxBeam` at 1.3e-9 relative, within the CPU's own nudge. It takes 44 s against 61 s on the CPU. Against
the CPU: 643 identical, 10 roundoff, 7 flagged, which are the 6 from before and this one.

**CSR check.** The routine list keeps CSR on the CPU, because `output` or `final` does unless CSR is
explicitly enabled. `test-lists/elegantTestSet-gpu-csr.txt` runs 33 decks with it enabled, and 26 reach GPU
CSR or LSC. They repeat exactly, but 28 flag against the CPU, since a roundoff change in `t` can move a
particle to the next histogram bin. On 5 of them, none of 441 flagged columns is more than 3 times a CPU
nudge, and `GPU_VERIFY` has no stops on those 26. Compare these with their GPU reference, not the CPU.

See `history/2026-11-06_more-options/REPORT.md` in GPU-Testing.

## ISR random numbers on the device (November 6, 2026)

These now run on the GPU, in plain tracking only. Batched programs refuse them, through
`gpuElementNeedsPlainTracking`.

| Element | Options |
| --- | --- |
| `KQUAD`, `KSEXT`, `KOCT`, `DQCOR` | `ISR` |
| `CSBEND` | `ISR`. `USE_RAD_DIST` and photon output stay on the CPU, see "Options that stay on the CPU" below |
| `LGBEND` | `ISR`, spin, `CENTROID_OUTPUT_FILE`, `&aperture_data` files, `APCONTOUR` |
| `CCBEND` | `USE_RAD_DIST`, `ADD_OPENING_ANGLE`, `SR_IN_ORDINARY_MATRIX`, which tracking never reads on the CPU |
| `CWIGGLER` | `TILT` with spin |

**The device makes the same random numbers as the CPU, in the same order,** so results stay comparable.
`random_2` is `dlaran`, a linear congruential generator modulo 2^48 whose value is its state over 2^48, so
one draw gives the state. `gpuCudaIsrFillDraws` jumps to each pair of uniforms by powers of the multiplier,
makes the same Box-Muller number as `gauss_rn_lim`, rejects past `SR_gaussian_limit`, and closes the gaps
with an exclusive scan. The result can't differ from the CPU's, apart from the rounding of `log`, `sin` and
`sqrt`.

**`random_2` is caught up only when something on the CPU next needs it.** Its state can't be set, and
stepping it costs 6.5 ns per number, which would be most of the run time with `ISR` on every magnet. So in
GPU builds, a macro in `track.h` sends elegant's own `random_2` calls to `gpuSyncedRandom2`, which steps it
by what the device has used. A negative seed restarts the device's place too.

**A loss in an `ISR` element redoes the element on the CPU with the same numbers.** `gpuIsrAbandon` returns the
device to where the element started and, if `random_2` has fallen behind, catches it up first. The draw taken
to read the state is handed back to the CPU's first call. That closes the gap the host table had, when every
particle was lost before its first draw.

**`LGBEND` draws by segment, then particle,** because the CPU tracks all particles through a segment before
the next. So one table for the magnet can't be split by particle. Above 2^27 numbers, set by
`ELEGANT_GPU_LGBEND_ISR_TABLE_DRAWS`, the magnet runs a segment at a time, which gives the same numbers in the same
order. A segment's table can then split by particle, with centroid records too. The others split a large beam into batches of about 256 MB, set by `ELEGANT_GPU_RECORDED_BATCH_BYTES`.

Measured with `SYNCH_RAD=1`, `ISR=1` and 2 passes, on one CPU thread against the GPU, and differing from the CPU
by at most 1e-17 in `Cx` and `Cdelta`:

| Line | CPU | Host table, as first built | Now |
| --- | --- | --- | --- |
| 20 `CCBEND`s, 200,000 particles | 6.44 s | 3.25 s | 1.45 s |
| 4 `CWIGGLER`s of 60 periods, 20,000 particles | 57.49 s | 5.30 s | 0.92 s |
| 40 `KQUAD`s, 200,000 particles | 28.03 s | Not run | 1.90 s |
| 40 `CSBEND`s, 200,000 particles | 32.17 s | Not run | 2.12 s |

A CPU element that draws a random number between each magnet brings the catch-up cost back, in the host's
time rather than the device's.

**RPN random numbers.** `grnd`, `grndl` and `srnd` in `SDDS/rpns/code/math.c` call `random_2` directly, past the macro,
`srnd` through the library's `random_1`. So a second macro in `track.h` catches `random_2` up before every `rpn()`, and
the device reads the state again afterward. That brings back the catch-up cost for decks with `&link_elements` or
`&modulate_elements` expressions, once per pass. On the CPU, `grnd` with no `srnd` first reseeds from the time of day, so
only decks that call `srnd` can match at all.

**Known limit.** The number taken to read the state can't be returned to the generator. If every particle is lost
before a magnet's first draw, and RPN then draws before the next `ISR` magnet, the sequence is off by one. Results stay
statistically right, but stop matching the CPU exactly.

**Replaced on November 7, 2026.** `random_2`'s state can now be read and set, so there is no catch-up, neither macro, and
no off-by-one. See "`random_2`'s state, block sizes for small beams, and multipole spin" below.

**Each piece was broken on purpose.** Three cases didn't catch their mistake at first and were changed.
`magnet-cwiggler-tilt-spin` needed a bend first, since a roll does nothing to a spin along `z`.
`isr-small-batches-loss` lost a particle in every first batch, which never needs the restore. The state restore
after a loss only matters when the CPU's redo draws nothing, so `isr-loss-before-first-draw` is new. The
`GPU_VERIFY` check of the random numbers stops when the rejection is removed.

**Results.** Feature set, 260: 26 identical, 228 roundoff, the same 6 flagged. `GPU_VERIFY`: no stops on 260, 39,
42 and 20 decks, with random numbers at most 8.9e-16 from the CPU's. CPU 660: identical to the previous reference.

**Default 660 against the CPU:** 638 identical, 12 roundoff, 10 flagged, which are the earlier 7 and `ERL2`,
`twoCavityMoments1` and `uKickMap5`, which now run `ISR` on the GPU. Those and the other 13 decks that moved or
flag in forced runs have 215 flagged columns, none more than 3 times a CPU nudge. The largest is `lgbend2`
`xError`, at 1.88. Compare a moved deck with its GPU reference. `ERL2` takes 15.5 s against 47.1 s on the CPU.

See `history/2026-11-06_more-isr/REPORT.md` in GPU-Testing.

## LGBEND on by default (November 6, 2026)

`LGBEND` now runs on the GPU by default at 512 particles or more. `ELEGANT_GPU_ENABLE_LGBEND=0` keeps it on the
CPU, and `ELEGANT_GPU_MIN_LGBEND_PARTICLES` sets the threshold. Measured on an RTX PRO 4000 Blackwell only, with
the `lgbend2` ring of 160 `LGBEND`s, 11,698 elements and `ISR` on, 3 passes:

| Particles, output | CPU | `LGBEND` on the CPU | `LGBEND` on the GPU |
| --- | --- | --- | --- |
| 256, `centroid` and `sigma` | 8.8 s | 9.0 s | 9.5 s |
| 512, `centroid` and `sigma` | Not run | 13.1 s | 11.9 s |
| 16,384, `centroid` and `sigma` | 266.6 s | 204.1 s | 61.7 s |
| 65,536, `centroid` and `sigma` | Not run | 806.9 s | 238.0 s |
| 16,384, no output | 225.4 s | 158.7 s | 16.1 s |

An older test on an RTX 3060 showed no gain, and slower cards haven't been measured since.

At 16,384 particles the ring differs from the CPU by at most 1.1e-14 m in `Cx` and 1.6e-14 m in `Sx` at every
element, and `GPU_VERIFY` has no stops.

See `history/2026-11-06_lgbend-default/REPORT.md` in GPU-Testing.

## Beam sums for output and the LGBEND first pass (November 6, 2026)

**Beam sums per element now use the device.** Since May 2026, a `centroid` or `sigma` file kept every beam sum on
the CPU, so results stayed closer to the CPU's, and each sum copied the whole beam back. On the `lgbend2` ring at 16,384
particles, that was 42 s of a 60.7 s run. Now the sums per element for those files use the device, above the usual
reduction threshold of 10,000 particles by default, and the run takes 18.3 s.
`ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS=0` restores the CPU.

Only those sums moved, through `BEAM_SUMS_ELEMENT_OUTPUT`. Sums for the final point, fit points and optimization terms
stay on the CPU, because results steer on them: moving them all sent `emittanceExchange`'s optimizer down a different
path, with a final beam differing by 1e-4 m. `compute_centroids` keeps its old rule too.

**What this costs.** The device adds in a different order, so the sums differ by rounding. On the default 660 against
the CPU, the flagged list goes from 10 to 54 decks. Every new flag is in a `.sig` file, and no `.out` or `.param`
file moved:

| Columns | Largest difference | Why |
| --- | --- | --- |
| `minimum5`, `maximum5`, `ma5` | 9.1e-13 m | One or two rounding steps of `s`, which sets the centroid they subtract |
| `alphaxBeam`, `betaxBeam`, `ecnx` | 2e-6 relative | Near-zero emittance, where `s11 s22 - s12^2` cancels. CPU nudges reach the same size |

So compare a large-beam deck with its GPU reference. Of 142 flagged columns on 7 decks, 139 are within 3 times a CPU
nudge. The other 3 are `twissElem1` `s` columns at 1.4e-14 m, one rounding step of `s`, which a nudge of `p_central`
doesn't move.

**`LGBEND` runs on the GPU from its first pass.** The optimization is now `optimizeLgbendIfNeeded` in `lgbend.c`, and
`gpuPackLgbendTracking` calls it with the on-GPU flag cleared, as `CCBEND` does. The gate only refuses a magnet in the
middle of being optimized, so `OPTIMIZE_FSE=0` runs on the GPU too. Twenty `LGBEND`s, 16,384 particles and one pass with no
`twiss_output` went from 4.6 s to 0.33 s.

See `history/2026-11-06_speed/REPORT.md` in GPU-Testing.

## Options that stay on the CPU (November 6, 2026)

**`CSBEND` `USE_RAD_DIST` and photon output can't run in parallel and still match the CPU.** `addRadiationKick` draws a
Poisson number of photons at each step, then a number for each photon, so how many random numbers a particle uses depends
on its own path. Particles draw in order, so a particle's place in the sequence depends on every particle before it, all
the way through the magnet. The device can only know that by tracking them one at a time. `ISR` works because every
particle uses a known count. A separate stream per particle would be right statistically, but it wouldn't match the
CPU, so it couldn't be checked against it. Photon output also needs the `USE_RAD_DIST` path, and writes a row per
photon.

**`APCONTOUR` `Logic` with anything but `&&`, `||` and `!`.** The CPU runs each contour's expression through `rpn()` for
each particle. `gpuCompileContourLogic` turns those three operators into stack operations for the kernel. Any other
expression, or a stack deeper than 62, stays on the CPU.

Above 2^27 random numbers, `LGBEND` `ISR` runs a segment at a time. At 2,000,000 particles through 20 magnets it takes
15.3 s against 734.6 s on the CPU, with `Cx` at most 1.1e-19 m from the CPU's.

See `history/2026-11-06_cpu-options/REPORT.md` in GPU-Testing.

## APCONTOUR on the GPU (November 6, 2026)

A sticky `APCONTOUR` used to copy the whole beam to the CPU after every element, so one sticky contour put a ring back at
about CPU speed. `gpu_track_through_apcontour` now runs that check and the element itself on the device. With nothing lost,
nothing is copied. On the `lgbend2` ring, 16,384 particles and 3 passes, it takes 17.1 s against 253.6 s on the CPU, and
16.2 s with no contour.

A loss at `L=0` uses the same compaction as the other apertures. A loss at `L>0` falls back to the CPU, which searches for
where the particle hit. Spin, `DZ` and `HOLD_OFF` stay on the CPU. The multipole and `CSBEND` kernels now check the
contour at each slice, as `CCBEND` and `LGBEND` already did. `&aperture_data` files still keep those magnets on the CPU.

See `history/2026-11-06_contours/REPORT.md` in GPU-Testing.

## Open-sided MAXAMP, SREFFECTS and large beams (November 7, 2026)

An audit by real copy counts found two more elements that copied the beam to the CPU every time. `MAXAMP` with
`OPEN_SIDE` now runs on the device, rectangular and elliptical, with the open side passed to the count and the
compaction. `SREFFECTS` with excitation now runs on the device, using `gpuTrackInBatches` and the same random number
table as `ISR`. At 262,144 particles, 2,000 drifts behind an open-sided `MAXAMP` go from 10.74 s to 0.50 s, against
5.46 s on the CPU, and 400 `SREFFECTS` calls go from 13.12 s to 0.54 s, against 22.69 s.

`QEXCITATION=0` had a parity bug. The CPU still draws three numbers per particle, and the device drew none, so the
sequence differed for whatever drew next. The device now draws them too. So `SREFFECTS` with `QEXCITATION=0` below the
magnet threshold now stays on the CPU, where it used to skip the draws on the device: `srEffects4` by default goes from
6.86 s to 11.39 s, against 10.80 s on the CPU. Above the threshold the device still wins: on the same ring at 65,536
particles and 30 passes, 5.68 s against 12.61 s.

Loss counts now use a grid of blocks with an atomic counter, because counts are integers and a grid can't change a result.
`centroid` output alone used a sums kernel with one block, and now uses the kernel `sigma` uses, skipping the products. At
2,097,152 particles and 2,000 drifts, `centroid` goes from 176.53 s to 4.31 s and an elliptical `MAXAMP` from 92.44 s to 3.14 s. The other
single-block kernels stay, because the rest are sums whose rounding would change, or are ordered.

**`GPU_VERIFY` doesn't reach a `MAXAMP` check after an element.** The check after each element leaves the beam on the
host, so the check that follows runs on the CPU. So `GPU_VERIFY` has never tested these, open-sided or not. The evidence for
open sides is the feature cases and `maxamp2`, `maxamp3` and `maxamp4` against the CPU.

See `history/2026-11-06_copies/REPORT.md` in GPU-Testing.

## `pmaximum` fiducialization (November 7, 2026)

Serial `findFiducialTime` started its `pmaximum` search at particle 0's `delta` and only accepted a larger value, so it
stopped with "No available particle for RF cavity fiducialization" whenever particle 0 had the highest momentum. That
includes every one-particle beam, equal momenta, and a bunch file whose first row has more momentum than the fiducial
bunch. Pelegant started at `-DBL_MAX` and didn't stop. `gpuFiducialPmaximumKernel` copied the serial start: it
compared every particle with `coord[5]` and skipped particle 0. Both now start at `-DBL_MAX` and include particle 0. The
kernel's reduction already picks the lowest index among ties, as the CPU does.

When the old code didn't stop, the largest `delta` was above particle 0's, so the old and new code pick the same
particle. All 294 older feature cases are identical to the previous references on the CPU and on the GPU. Two new cases,
`rf-pmaximum-fiducial-first-highest` and `rf-selected-pmaximum-fiducial-other-bunch-first`, stop on both of the old
binaries and reach the device kernel by default. The first ties particle 0 with particle 701, which arrives 2 ps later,
so picking the wrong one would move `Cdelta` by 5.4e-7, where the GPU and CPU differ by 6.7e-17.

`GPU_VERIFY` doesn't check the fiducial time. Only the comparison with the CPU covers it.

See `history/2026-11-07_pmaximum/REPORT.md` in GPU-Testing.

## A shrinking beam stays on the device, WATCH sums, and prepass losses (November 7, 2026)

**A beam on the device stays there below the general thresholds,** down to 1,000 particles. When losses took a beam
below 10,000 particles, `LGBEND` and `CCBEND`, with their lower thresholds, kept it on the device, while the `MAXAMP`
check after each of them went back to the CPU. So the beam was copied around every bend. Now matrix, helper, exact
drift, reduction, aperture and magnet checks run on the device while the beam is there and has at least
`ELEGANT_GPU_MIN_RESIDENT_PARTICLES` (default 1,000) particles. `ELEGANT_GPU_ENABLE_KEEP_RESIDENT=0` restores the old
rule. Collective effects, `RFMODE` and other element thresholds are unchanged, and so is the rule that keeps exact drifts
on the CPU below their threshold when there's an order-sensitive output file.

| Ring of 11,698 elements, 3 passes, no output | CPU | Before | Now |
| --- | --- | --- | --- |
| 8,000 particles | 112.69 s | 55.77 s, 960 copies | 10.39 s, 1 copy |
| 2,000 particles | Not run | 19.48 s | 9.66 s |
| 1,000 particles | Not run | 13.50 s | 9.48 s |
| 400 particles, below the floor | Not run | 10.07 s | 10.05 s |

Setup takes about 4.5 s of each. Below the floor, `CCBEND` still takes the beam to the device and the check after it
copies it back, as before. The cost shows on `losses1`, whose 10,000 particles all get lost on a short lattice with
output at every element: it now stays on the device down to 1,000 particles, and its median of 5 runs goes from 6.51 s to
6.67 s. Lowering every threshold to 64 made it about 8% slower in an earlier test, which is why there's a floor.

**`WATCH` parameter and centroid sums use the device when a `centroid` or `sigma` file is requested,** as the sums per
element already did. They used to copy the beam once per `WATCH` call. With 50 `WATCH` points over 20 passes, 65,536
particles take 2.24 s, against 6.16 s before and 80.54 s on the CPU. These files now differ from the CPU's by rounding.
Trajectory centroids and `compute_centroids` stay on the CPU in those runs: they steer correction, fit points and beam
centering. Slice analysis selects by time, which the device sums don't support.

**The fused tune prepass** runs only in `GPU_VERIFY` builds. `search-frequency-map-prepass-magnet-losses` loses particles in
its `CSBEND`, `CCBEND` and `SCRAPER`, which no case did before. The CPU used to repair a loss the prepass missed without a
trace, so `GPU_VERIFY` now stops if a particle the prepass kept is lost on the CPU. Dropping the losses in all three
elements stops the new case, and losing extra particles in each flags it against the CPU. A missing check inside the
`CSBEND` alone isn't caught, because the `MAXAMP` check at its exit loses the same particles. Only a particle that leaves
the aperture inside the bend and returns by its exit would show it, and I couldn't make a case for that.

**Default 660:** the same 56 decks are flagged against the CPU, with the same largest differences. Eight decks moved by
roundoff against the previous GPU run: `LCLSII-2`, `TTF2`, `collimate3`, `losses1`, `rcol3`, `rfdf4`, `script2`, `uKickMap5`.

See `history/2026-11-07_residency/REPORT.md` in GPU-Testing.

## `random_2`'s state, block sizes for small beams, and multipole spin (November 7, 2026)

**`random_2`'s state can be read and set.** `SDDS/mdbmth/drand.c` now has `random_2_get_state` and
`random_2_set_state`. Each `ISR` element reads the state when it starts and moves `random_2` past the numbers it used
when it commits, so `random_2` is always current. The lazy catch-up, the borrowed first number and both macros in
`track.h` are gone. On the ring at 16,384 particles and 3 passes, one `&modulate_elements` expression used to take the
run from 16.0 s to 31.3 s; it now takes 10.15 s, against 10.05 s without it. The off-by-one after a step that lost every
particle before the first draw is gone too, and `isr-loss-before-first-draw-rpn` checks it: the old binary's `Cx`
differed from the CPU's by 2e-3 relative there. `gpu-elegant` now needs an SDDS library with these two functions.

**Small beams use smaller blocks.** Each thread tracks its own particle, so block size can't change a result.
`gpuParticleThreads` halves the block from 256 down to one warp until there are at least two blocks per
multiprocessor, for matrices, multipoles, `CSBEND`, `CCBEND`, `LGBEND`, kick maps and `POLYNOMIALSERIES`, checked and
stable-compaction kernels alike. `ELEGANT_GPU_PARTICLE_THREADS` fixes the size. This card has 70 multiprocessors and
slow double precision, so a few thousand particles in 256-thread blocks left most of it idle.

**Multipoles and `CSBEND` track spin on the device.** `KQUAD`, `KSEXT`, `KOCT` and `DQCOR` follow
`multipole_tracking2()`, and `CSBEND` follows `track_through_csbend()` and `integrate_csbend_ordn()`, through the same
`gpuSpinUpdateLocalFS`, which is `updateSpinQuaternionLocalFS()` with `Bz` zero. Spin stayed on the CPU with edge
multipole files, misalignment methods other than 0, zero length, `MULT`, `KQUSE`, and `CSBEND` with the expanded
Hamiltonian; since November 8 none does, zero length last, after CPU bug 13 (see below). **For a tilted multipole the CPU used to turn the spin twice:** `rotateBeamCoordinatesForMisalignment()` turns it
with the coordinates, and `multipole_tracking2()` turned it again. Fixed later the same day on the CPU and the device, as
CPU bug 12; both now turn it once.
`WATCH` parameter output and beam sums still refuse spin, so a spin run copies once per `WATCH` call or element sum.

**Results.** Feature set, now 305: 29 identical, 270 roundoff, the same 6 flagged; the 299 older cases are identical on
the CPU and the GPU. `GPU_VERIFY` has no stops on 305, and on `spinTest1` forced it checked 12,972 multipole and 17,296
`CSBEND` calls with spin, at most 3.6e-15. Default 660: identical to the previous GPU run, the same 56 flagged against
the CPU, and the CPU 660 is identical too. Routine list: only `spinTest1` and `spinTest2` moved, now flagged against the
CPU, but within 1.54 and 2.96 times a CPU nudge. `spinTest1` is a simplex optimization on spin centroids, and a 1e-14
nudge moves its result by as much as the GPU does.

See `history/2026-11-07_rng-threads/REPORT.md` in GPU-Testing.

## `RAMPP`, `FMULT`, feedback, and jobs sharing the GPU (November 7, 2026)

**`RAMPP` runs on the device.** A kernel makes each particle's time with `beta_from_delta()`'s formula, and the host
adds them in particle order, so the mean time is the CPU's to the last bit and `RAMPP` doesn't need the CPU for output
in CPU order. A second kernel rescales `delta`. `GPU_VERIFY` compares every time exactly, and the rescale differed from
the CPU's by 0 on every pass of `rampp-waveform`. In verify
builds a device helper must also apply the CPU's change to the host copy, or the next comparison stops: the first verify
run of `rampp-waveform` did, by 1.09e-4 in `delta`.

**`FMULT` uses the multipole kernel:** order 4, one slice per kick, the file's terms scaled by `(1+FSE)*FACTOR`, offsets
and `TILT` as misalignment method 0, no aperture, and spin with the same limits as the other multipoles.
`expandHamiltonian` is whatever the last magnet left, as on the CPU. In required mode a single particle now goes to the
device too, where each 50-kick `FMULT` costs 0.75 ms on one thread, so `fmult5` passes the routine list's 300 s limit.

**Feedback.** `TFBPICKUP` `RMS_NOISE` no longer keeps it on the CPU: the noise is drawn on the host after the device
average, as on the CPU. `BUNCHED_BEAM_MODE`, whose default is 1, did keep every default `TFBPICKUP` and `TFBDRIVER` on the
CPU. With one bunch the CPU's bunched path is the plain one, so `do_tracking()` now notes at the start of a run whether
every particle has the same bunch index, the rule `index_bunch_assignments()` uses, and bunched-mode feedback uses the
device when it does. A beam with two bunches stays on the CPU.

**`NIBEND` stays on the CPU.** The device has only non-adaptive Runge-Kutta with a hard edge; the defaults are adaptive
Runge-Kutta, a linear fringe, `ADJUST_BOUNDARY` and `FUDGE_PATH_LENGTH`. Adaptive steps chosen on the device could differ
from the CPU's by rounding, which moves results by up to the integration accuracy, not by rounding.

**Gaussian draws.** The fill used for `SREFFECTS` and `ISR` no longer calls `thrust::reduce` and
`thrust::exclusive_scan`, which allocate and free device memory each call and wait three times. It does one CUB scan into
scratch kept between calls and one copy back of two numbers. The draws are unchanged. Four 100,000-particle `SREFFECTS`
jobs sharing the GPU took 133 s each against 24.7 s alone; now 96.8 s against 23.3 s.

**Found:** `WATCH` parameter sums used the device only when a `centroid` or `sigma` file was also written (changed
the same day, below), and helpers such as concatenated `MATR` stayed on the CPU when a `WATCH` writes output (changed the same day, below).
On the 100,000-particle job at 2,000 passes, these took the run from 2.7 s to 44.4 s.

**Results.** Feature set, now 311: 29 identical, 276 roundoff, the same 6 flagged; the 305 older cases are identical on
the CPU and the GPU. `GPU_VERIFY` has no stops on 311. CPU 660: only `photonEmission1`'s `GPUCapable` flags moved. Default
GPU 660: `feedback2`, `feedback4` and `feedback5` now flagged against the CPU, as their feedback runs on the device; each is
within 2.53 times a CPU nudge, the time-plane pickup in `feedback4` and `feedback5` magnifying a last-bit change in the
average. Routine list, now 424 without `fmult5`: the `FMULT` and feedback decks moved, and only `feedback4` and `feedback5`
are new flags against the CPU.

See `history/2026-11-07_rampp-fmult/REPORT.md` in GPU-Testing.

## `WATCH` sums without output files (November 7, 2026)

`WATCH` parameter and centroid sums now use the device whether or not the run writes a `centroid` or `sigma` file.
`dump_watch_parameters()` passes a new `BEAM_SUMS_WATCH_OUTPUT`, which the CPU's sums ignore, because these sums feed only
the `WATCH` file. Other sums keep the rule from item 10. A `WATCH` file in such a run now differs from the CPU's by
rounding. `ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS=0` restores the CPU. On the 100,000-particle `SREFFECTS` job, the default
run goes from 44.4 s to 24.1 s, with the same final `WATCH` row; the remaining copies are at the concatenated `MATR`
helpers.

**Results.** Feature set, now 312: the 311 older cases identical on the CPU and the GPU, the same 6 flagged; `GPU_VERIFY`
has no stops. CPU 660 identical. Default GPU 660: `branch4`, `feedback2` and `frfmode2` moved by rounding, and the same 59
flagged against the CPU. Routine list identical.

See `history/2026-11-07_watch-sums/REPORT.md` in GPU-Testing.

## Helpers that match the CPU, with output in CPU order (November 7, 2026)

With `output`, `final` or a `WATCH` file, helper elements stayed on the CPU unless `ELEGANT_GPU_MIN_HELPER_PARTICLES` was
set, a rule from May 2026 meant to keep those files closer to the CPU's. `GPU_VERIFY` with verbose output showed
`MALIGN`, `MATR`, `EMATRIX` and `ENERGY` with a given momentum matching the CPU to 0 on every call, so the rule bought
nothing for them. They now skip it. One gap: `MATR` and `EMATRIX` set a reference time from the beam's mean `s` on their
first call, which the device summed in its own order. Verify couldn't see this, since the CPU check reuses the reference
the device set. That sum now uses the CPU's order, from one read-only copy. `ENERGY` with `MATCH_PARTICLES` or
`MATCH_BEAMLINE` averages the beam's momentum on the device, differs by up to 8.9e-16, and keeps the rule, as do
`RFCW`'s RF-only paths. On the 100,000-particle `SREFFECTS` job with concatenated `MATR`, the default run now takes
2.55 s, against 24.1 s, with the same final `WATCH` row.

**Results.** Feature set, now 313: the older cases identical apart from `matrix-helpers` by rounding; `GPU_VERIFY` has no
stops, 249 identical to the CPU. Routine list: only `nonlocalDispersion1` moved, now within rounding of the CPU. **Default
GPU 660:** the rule also kept elements next to the helpers on the CPU, and they now use the device. 25 decks moved, and
`losses1`, `losses3`, `wake2` and `wake4` are newly flagged against the CPU, 63 in all. `GPU_VERIFY` at default
settings on 15 of the moved decks has no stops; every element matches exactly except `CSBEND`, at most 3.6e-15, and
`SREFFECTS`, 1.1e-19. The new
differences are in coordinate files downstream of `CSBEND` and in per-element `sigma` sums. `trajCorrect2` stays within
its approved review, at 0.83 of its worst limit as before. The user chose to keep this on November 7, 2026.

See `history/2026-11-07_helpers/REPORT.md` in GPU-Testing.

## Exact drifts, bends below the floor, spin sums and tune histories (November 8, 2026)

**Exact drifts with output in CPU order.** `gpuExactDriftParticleCountAllowed` dropped the resident rule when a run
wrote `output`, `final` or a `WATCH` file, so collimator lengths and zero-strength `MULT`s sent a beam on the device back
below their threshold. Device exact drifts match the CPU to 0 (800 calls, verbose `GPU_VERIFY`), so they keep the
resident rule now. A lossy line settling at 1,362 particles went from 600 copies to 1, and from 1.48 s to 0.81 s.

**Bends below the floor.** The `MAXAMP` check after each element sent the beam back below 1,000 particles, so `CCBEND`
and `LGBEND`, with lower thresholds of their own, made a round trip each. The short-island policy couldn't see it: the
check isn't an element, and these bends lack `IS_MAGNET`. `do_tracking()` now reports whether the check follows, and with
the beam on the host such an element stays on the CPU when the check couldn't use the device. Required mode is unchanged.
The ring at 400 particles: 480 copies to 0, at the same speed.

**Spin sums on the device.** `WATCH` parameter output and beam sums refused spin. The device now sums the spin
components and their centered products, merged as `accumulate_beam_sums()` merges them, and `GPU_VERIFY` compares them.
A spin ring with `WATCH`es and `centroid` and `sigma` files: 193 copies to 1, 0.58 s to 0.33 s.

**One tracking call for a tune history.** The batched tune footprint and frequency map called `do_tracking()` once per
turn. With one period per turn and no other wedge, they now make one call, and a tracking wedge records each turn at
the start of each pass, on the device while the beam is there, numbering passes as the per-turn calls did. The
`tune-footprint-batched-heavy` outputs are identical to the old path's on the GPU and with it off.

**Results.** Feature set, now 316: the spin cases with output files and `resident-below-threshold` moved by rounding, and
the frequency map and tune footprint cases are identical; the same 6 flagged against the CPU, and `GPU_VERIFY` has no
stops. CPU 660 identical. Default GPU 660: only `collimate3` moved, by 1e-21, with the same 63 flagged against the CPU.
Routine list: only `spinTest2` moved, by rounding.

See `history/2026-11-08_items-23-27/REPORT.md` in GPU-Testing.

## Momentum matching in CPU order, reused device buffers, and RFCW's RF-only paths (November 8, 2026)

**Momentum matching.** `do_match_energy()` averages `P_central*(1+delta)` with `KahanPlus()` in particle order. The device
summed in its own order and differed by up to 8.9e-16, so `ENERGY` with `MATCH_PARTICLES` or `MATCH_BEAMLINE` kept the
helper output rule, and the matching `always_change_p0` and RF `CHANGE_P0` do on the device drifted from the CPU's.
With `output`, `final` or a `WATCH` file, the device now copies the one momentum coordinate to the host (or reads the
host copy when it is current) and makes the CPU's Kahan sum there; the particles change on the device. Matching `ENERGY`
then skips the rule. `ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER=1` or `0` sets this for every run. Verbose `GPU_VERIFY` shows
0 difference on every `do_match_energy` call in the feature set (63 calls). New case
`matrix-energy-match-ordered-output`: 41 copies to 1, with `run.out` and the `WATCH` file now identical to the CPU.
Scaled to 100,000 particles and 20 passes: 6.83 s to 1.96 s (CPU 12.09 s). The CPU's sum costs about 0.33 ms a call
there; the device's own sum without the rule takes 0.48 s.

**Reused device buffers.** 33 runtime functions (loss counts, reductions, the checked magnet trackers' loss counts and
beam backups, wakes and LSC) called `cudaMalloc` and `cudaFree` on every call. They now reuse buffers that grow as
needed and are released with the other scratch; a call's n-th buffer takes slot n. The checked trackers' error paths
also no longer destroy the cached timing events. No value changes: every case without momentum matching is bitwise
unchanged. A 50,000-particle `KQUAD`/`KSEXT`/`KOCT` ring, 150 passes: 4.19 s to 3.61 s. Four jobs sharing the GPU
didn't change (16.7 s each, 6.3 s at 2,000 particles), so the sharing cost isn't the allocations.

**RFCW's RF-only paths.** They don't match the CPU exactly, so they keep the output rule. They use the same device code
as `RFCA`'s RF-only paths, which never followed the rule, and verify gives the same per-call differences for both: up to
8.9e-15 in kick mode and 1.5e-16 in matrix mode. The kick kernel adds `-ik*omega*dtLight` and the time offset to the
phase rather than to `t` as `simple_rfca.c` does, and uses CUDA's `sin`. Without the rule, a 20,000-particle RFCW line
with an `output` file runs in 1.85 s rather than 5.50 s (CPU 6.18 s); that is a parity decision left to the user.

**Results.** Feature set, now 317: 35 older cases moved by rounding, all of them doing momentum matching; the rest are
bitwise unchanged. Against the CPU: 28 identical, 283 roundoff, the same 6 flagged. `GPU_VERIFY` has no stops; against
the CPU 286 identical (251 before), and `wake-bunched-change-p0-skip` is no longer flagged. CPU feature set identical.
Default GPU 660: 654 identical; `firstIsFid1` to `firstIsFid4`, `ERL2` and `trajCorrect2` moved, all doing momentum
matching, and `trajCorrect2` now has no field outside the strict envelope, so 62 are flagged against the CPU. Routine
list: the 5 that moved all came closer to the CPU, `firstIsFid2` from 802,758 values outside the envelope to 1,520;
`trajCorrect2`'s 9 are within its approved review, at most 0.04 of a limit.

See `history/2026-11-08_items-31-33/REPORT.md` in GPU-Testing.

## `RAMPRF` on the device and exact RF kicks with output (November 8, 2026)

**`RAMPRF`.** `ramped_rf_cavity()` had no device code, so every pass copied the beam to the host and back. It now
finds the waveform values, phase and fiducial on the host as before and runs its particle loop on the device: the
half drifts, `add_to_particle_energy()`'s kick, and the CPU's stop with its message on a coordinate that isn't finite.
The no-voltage drift (`s` grows by `L`, not the path length) is the CPU's too. The fixed-frequency fiducial uses
`gpu_findFiducialTime()`, which with output in CPU order is the CPU's `findFiducialTime()`. `RAMPRF` now carries
`GPU_SUPPORT`, which changes `dictionary.sdds` (`photonEmission1`) on the CPU as well. It follows the helper threshold.
`rampBooster1` in forced mode: 20,000 copies to 1. A 20,000-particle booster ramp, 1,000 passes: 1.84 s to 0.86 s
(CPU 4.55 s). At 100 particles it is 8% slower than before (3.37 s against 3.11 s; CPU 0.89 s), because each pass now
makes three small transfers rather than two.

**RF kicks in the CPU's order.** For a traveling wave, `simple_rfca.c` takes `ik*dtLight` from the time before it
multiplies by `omega`; the device folded `-ik*omega*dtLight` into the phase. `gpuRfcwKickInitialKernel` and the LSC
energy-change sum now take that shift as the CPU does, and `PHASE` and `DELTA_PHASE` are added one at a time, as the CPU
adds them. A thin `RFCA` with `DX` or `DY` now applies and removes the offsets as the CPU does; the kick doesn't use
x or y, but the CPU's round trip rounds them.

**The CPU's `sin` and `log` with output.** With `output`, `final` or a `WATCH` file (`ELEGANT_GPU_RFCA_REFERENCE_MATH`,
as for the thin kick), kick-method and matrix-method `RFCA` and `RFCW` and `RAMPRF` now make their phases on the device,
copy them to the host, and use the CPU's `sin`; the matrix method also takes its `R12` from the CPU's `log`. The
particles change on the device. Verbose `GPU_VERIFY` shows 0 difference on every RF-only `RFCA`, `RFCW` and `RAMPRF`
call in cases with output; RFCW kick mode was up to 8.9e-15 before. The calls that still differ are in cases without
output files, or are cavities with wakes or LSC, whose sums are in device order. So RFCW's RF-only paths no longer
follow the helper output rule. A 20,000-particle RFCW line with output, 200 passes: 5.48 s to 2.07 s (CPU 5.92 s; 1.89 s
with the device's `sin`). The same line with `RFCA`, which already ran on the device: 1.86 s to 2.04 s for exact output.

**Mistakes built on purpose.** One build had two: `RAMPRF`'s first half drift without the slope factor flagged all
three `RAMPRF` cases, and the old phase order took `run.out` off the CPU's exact values in the four traveling-wave
multi-kick cases with output (`rfca-delta-phase`, `rfca-kick-rf-only`, `rfcw-kick-rf-only`, `rfcw-multikick`); the
standing-wave, matrix and thin cases have no shift and stayed identical, as they should.

**Results.** Feature set, now 320 with three `RAMPRF` cases: against the previous GPU references 34 older cases moved,
all of them RF cases; three of those are single-ulp changes the harness can't measure and so marks for review. Against
the CPU: 28 identical, 286 roundoff, the same 6 flagged. `GPU_VERIFY` has no stops; against the CPU 289 identical (286
before). CPU feature set identical. Default GPU 660: 657 identical. `CLIC1` and `CLIC2` moved: their RF-only RFCW
cavities now run on the device, with `run.out` and `run.fin` identical to the CPU, and only the per-element centroid
and sigma sums in device order. `photonEmission1` moved by the `GPUCapable` flag, and matches the new CPU binary.
Routine list: 422 identical; `fiducial1`'s four particle files and `rfMatrix2`'s `run.out` are now identical to the CPU.
GPU test set as before.

See `history/2026-11-08_items-30-34/REPORT.md` in GPU-Testing.

## `SREFFECTS` draws, `GPU_VERIFY` gaps, and new coverage (November 8, 2026)

**`SREFFECTS` random numbers with output.** The device made the Gaussian numbers for `ISR` and `SREFFECTS` from
`random_2`'s exact uniform sequence, but with CUDA's `log` and `sin`, which differ from glibc's by an ulp now and then
(up to 4.4e-16 in a number). In a ring that grows: forced `ramprf1` ended up to 0.56% off the CPU in momentum after
200,000 passes, although every `RAMPRF` call was exact. With order-sensitive output, `SREFFECTS` now makes its numbers on
the host as `gauss_rn_lim()` does and copies them to the device; its kernel was already the CPU's arithmetic, so its
particles now match the CPU exactly. `ELEGANT_GPU_ISR_REFERENCE_MATH=0` or `1` sets this for every element and run.
`gauss_rn()` makes each number from two uniforms with `log` and `sincos`, so a number depends only on where its pair
starts: OpenMP threads can make stretches of pairs from jumped `random_2` states, from 2,048 numbers up, and the
numbers beyond the cutoff are then dropped in order. The fill uses one thread, as elegant does, unless the run asks for
more with `-ompThreads`, `OMP_NUM_THREADS` or `ELEGANT_GPU_ISR_REFERENCE_THREADS`: idle OpenMP threads spin between
fills and take cores other jobs may need (forced `srEffects2` on 8 threads: 16 s of CPU time in 2.4 s). The verify
build compares each number with `gauss_rn_lim()` exactly (201 million in `sreffects-deterministic-heavy`), and the
output is the same at every thread count tried (1, 3, 4 and 8). Cost: `sreffects-deterministic-heavy` 0.38 s to 4.45 s
on one thread, 1.04 s on 8 (CPU 7.4 s), so on one thread it misses the suite's 2x gate (1.67x) and `test-set/suite.json`
sets `ELEGANT_GPU_ISR_REFERENCE_THREADS=8` (7.3x); a 20,000-particle ring
of 80 matrix elements, 1,000 passes, 0.91 s to 1.09 s on 8 threads (CPU 27.6 s); none measurable on the APS ring.
Forced `srEffects2`, with 3,000 numbers per fill, takes 5.4 s on one thread and 2.4 s on 8, against 4.98 s with the
device's numbers.
The magnets keep the device's numbers by default: their kernels differ from the CPU anyway, and host numbers cost about
twice as much even on 8 threads (a ring with `ISR` on every `KQUAD` and `CSBEND`, 20,000 particles, 20 passes: 0.90 s
against 1.93 s, CPU 23.4 s).

**`GPU_VERIFY` gaps closed.**
- The `MAXAMP`, aperture-data and `APCONTOUR` checks after an element never ran on the device in this build:
  `startCpuTimer()` cleared the element's flag before its CPU shadow, and aperture loss compaction was compiled out,
  so every loss went to the CPU unchecked. `do_tracking` now puts the element back on the device for those checks
  (`gpuVerifyResumeElementOnGpu()`), and the compaction runs in the verify build too, leaving the host's rows,
  accepted coordinates and global loss coordinates to the CPU shadow. Each aperture's check then compares survivor
  counts and survivors' rows (`compareGpuCpuApertureResult()`), row by row; `SCRAPER` was paired by particle ID
  until its survivors came out in the CPU's order (below). Lost rows aren't compared, because their global loss
  coordinates exist only on the host.
- Fiducial times: `gpu_findFiducialTime()` now compares its time with the CPU's `findFiducialTime()`
  (`ELEGANT_GPU_FIDUCIAL_COMPARE_REL`, 1e-12).
- First-call references: `MATR` and `EMATRIX` found their reference time on the device and the CPU shadow reused it.
  On the first call the shadow now finds its own and compares (exact).

Proof that each check fails: one verify build with four mistakes stopped at the new checks. `elimit_amplitudes` (an
elliptical compaction that keeps particles just outside) stopped in three cases, off by one survivor;
`limit_amplitudes` in two; the `MATR` reference time in both `MATR` cases; and the fiducial time on
`rf-pmaximum-fiducial-first-highest`, 2 ps off, the old `pmaximum` bug's choice of the later tied particle.

**New cases.** `RAMPRF` with `pmaximum` (2), `CSBEND` losses inside the bend (2), and a ring that loses its beam
through the 1,000 floor (3). A normal build whose `CSBEND` checked the aperture only before its first step failed both
`CSBEND` cases and none of the other 14 `CSBEND` cases. A build that didn't copy the beam back to the CPU after the
floor failed `resident-floor-ring-final`.

**`track1` past 1,024 turns.** Pages 0, 5, 10, 13 and 15, each alone, to 8,192 turns. The largest `Cx` difference from
the CPU at 8,192 turns is 1.1e-12, 3.2e-12, 3.1e-11, 5.7e-11 and 1.5e-11 m, against 4.4e-13, 2.8e-12, 1.1e-11, 3.0e-11
and 2.3e-11 m for the CPU on glibc's other path (`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-FMA4,-AVX`). Page 10 stays
about 2.5 times the CPU's own figure, as at 1,024 turns, so it isn't pulling away.

**Results.** Feature set (327): against the CPU 28 identical, 293 within the envelope, the same 6 flagged; the verify
build runs all 327 with no stops. elegantTestSet in default mode: 61 flagged against the CPU, down from 63, none new
(`acSextupole1` and `feedback6`, both `SREFFECTS`, dropped off). Routine list (424, forced): the 7 tests that moved
all have `SREFFECTS`, and against the CPU they went from 7 flagged to 1 (`feedback2`, TFB pickup sums in device order,
1e-9 relative). GPU test set: 5 identical and 34 within against the CPU.

**Forced slow tests.** Required mode ignores the particle thresholds, so a one-particle deck runs every element on the
GPU, one thread each. Of the 173 slow tests that finished forced on November 5, 172 finish at a 900 s limit; 32 are
flagged against the CPU, 23 as before and 9 new, all small (values near zero, `WATCH` sums and device `ISR` numbers
since November 6 and 7, and `photonEmission1`'s expected `GPUCapable` change). `track4` no longer finishes: on
November 5 a CPU element left its beam on the host for the rest of each step, and now it stays on the device. The
other 63 can't finish in required mode; in auto mode with thresholds of 64 (`test-lists/gpu-auto-64.env` in
GPU-Testing) all 63 match the CPU to rounding, but only 9 of them use the GPU.

See `history/2026-11-08_coverage/REPORT.md` in GPU-Testing.

## Aperture files in magnets, contour options and spin options (November 8, 2026)

**`&aperture_data` inside magnets.** `KQUAD`, `KSEXT`, `KOCT`, `DQCOR`, `KQUSE` and `CSBEND` now check an aperture
file on the device. `setupMultApertureData()` takes the file once, at `z_start` plus half the effective length for
`multipole_tracking2()` and half `L` for `CSBEND`, and its limits and center offsets go into the kernel's aperture with
the reverse tilt from `MAXAMP`, as `CCBEND` already did. `magnet-kquad-aperture-data` and `magnet-csbend-aperture-data`
use half a betatron period in each magnet and a file that narrows at each center, so only the check inside the magnet
loses particles. A build that took the file at the entrance failed both.

**Contours.** `HOLD_OFF` only stops the element itself from checking, which `do_tracking()` does, so held-off
contours are checked after later elements on the device, and the element's own drift runs there too. Spin no longer
keeps a contour on the CPU: the CPU's roll of coordinates and spins by `TILT` and back changes survivors only by
rounding, and compaction moves whole rows. `DZ` moves the check point along the slope, `x - DX + DZ*xp` in the CPU's
order; magnets ignore it, since `checkApContour()` checks a particle with zero slope. A `DZ` contour that loses particles
goes to the CPU, which also moves each lost particle's `s` back along its slope. `Logic` now takes `popl` as well as
`&&`, `||` and `!`; other tokens need the CPU's `rpn()`.

**Spin.** Edge multipole files turn the spin after each edge kick over a hundredth of a slice, with the new slopes at
the entrance and the old ones at the exit, as `integrate_kick_multipole_ordn()` does. Misalignment methods 1 to 4 roll
the spin with the design tilt through `rotateBeamCoordinatesForMisalignment()`, as method 0 does, and leave it alone
for `ETILT`, `PITCH` and `YAW`; the kernels already rolled it so. `MULT` only rolls the spin with `TILT`
(`spinRollOnly`). `CSBEND` with the expanded Hamiltonian turns the spin at its edges and tilt but not in its body, since
`integrate_csbend_ordn_expanded()` doesn't. `KQUSE` now runs on the device at all, as a quadrupole with the sextupole as a
second order, without `MATRIX_TRACKING` or `SYNCH_RAD`.

**Checks.** 13 new feature cases, all within the envelope against the CPU and passing the verify build. One build
with eleven mistakes (file aperture at the entrance for multipoles and `CSBEND`, no entrance edge turn, `MULT` turning
the spin in its body, the expanded `CSBEND` turning it in its body, no roll for misaligned multipoles or `CSBEND`, `DZ`
ignored, `popl` doing nothing, compaction leaving spins behind, `HOLD_OFF` contours never checked) flagged each of its
11 cases and neither `KQUSE` case; a build packing `KQUSE`'s sextupole as an octupole flagged exactly the three cases
that use `KQUSE`. The 327 older cases are unchanged from the previous GPU run.

**Left on the CPU.** `CSRCSBEND` with `MAXAMP`, a contour or an aperture file, since its resident CSR path has no loss
handling; `BMXYZ` with any of them; contours behind batched searches, which have no contour step yet; `Logic` tokens
other than these four. `quse2` (one particle, 8,192 passes) no longer finishes in required mode, like `track4`.

See `history/2026-11-08_cpu-features/REPORT.md` in GPU-Testing.

## `CSBEND` edge models 2 to 4 and slice analysis sums (November 8, 2026)

**Edge models.** `EDGE1_EFFECTS` and `EDGE2_EFFECTS` 2 (`dipoleFringeKHwang()`), 3 (`applySimpleDipoleEdgeKick()`) and 4
(`dipoleFringeKHwangRLindberg()`) now run on the device, model 2 at edge orders 1 and 2. The fringe bodies are the
CPU's term for term, with `ipow2`/`ipow3` as plain products, and the edge angle's `1/cos`, `tan`, `sin` and `cos` come
from the host, so only model 4's `exp()` and model 3's `tan(e - psi/(1+delta))` are the device's. With `HGAP=0` the CPU
stops for model 2 at `EDGE_ORDER>1`, and model 4 divides by the gap, so those stay on the CPU.

**A particle that turns non-finite inside a magnet** stays in the CPU's beam until the `limit_amplitudes()` with
`DBL_MAX` limits that `do_tracking()` calls after every element drops it; the device's magnet kernels drop it
themselves, since `gpu_limit_amplitudes()` skips that call. Outputs are the same, and a losses file sends the magnet to
the CPU. Model 4's `exp()` overflows for runaway particles (`y` about 1.6 m in `magnet-csbend-spin`), so the verify
build's `CSBEND` loss check now drops non-finite particles from the CPU shadow's result the same way before comparing.

**Slice analysis.** `performSliceAnalysis()` takes its numbers from the device when only the device has the beam:
`computeTimeCoordinatesOnly()` into column 4's place, the bunch's span by `thrust::minmax_element`, the slice bounds
from the CPU's expressions on the host, then each count, mean sum and second-moment sum run by one device thread over the
particles in order, so they are the CPU's exactly; the host divides and derives the rest as before. `do_tracking()`
then skips the read-only copy. With the same binary, `run.slan` is byte-identical with and without
`ELEGANT_GPU_ENABLE_SLICE_SUMS=0`. The verify build keeps the host current after every element, so it doesn't reach
this path; its slice check (`sliceMoments()` in `sliceAnalysis.c`) compares exactly if it does. Later sums now also run
on the device in `bunchComp1`, `bunchComp1m`, `bunchComp2` and `bunchComp10`, which moved by rounding and were flagged
against the CPU before.

**Checks.** 4 new feature cases, within the envelope against the CPU and passing the verify build. Three builds, each
with an edge-model mistake and in two a slice mistake (exit formulas at the entrance with the wrong moment pairs; model
4 as model 2 with twice `c`; model 3 ignoring its kick limit and using the entrance angle at the exit), flagged exactly
the cases aimed at. The other 340 feature cases are unchanged but `magnet-csbend-spin` and
`magnet-csbend-advanced-fallback`, whose model 2 to 4 bends moved to the device (rounding).

**Not done.** Spin in zero-length multipoles and `CSRCSBEND` with apertures were held back by CPU bugs, since fixed
(below). `BMXYZ` apertures and contours behind batched searches have no
use in elegantTestSet and need larger changes (stage-point checks in the integrator; more than one contour on the
device at once).

See `history/2026-11-08_edges-slices/REPORT.md` in GPU-Testing.

### CPU bugs 13 to 15

Three CPU bugs found while matching the GPU are fixed on November 8, as bugs 13 to 15 in
`elegant-CPU-bugs-and-fixes.md`.

**Spin in zero-length multipoles (13).** The CPU passed the kick as the field over a step of zero, so every spin became
NaN, even with no kick. `turnSpinForKick()` now passes it over a unit step, which turns the spin by the same amount.
`gpuMultipoleSpin` does the same, and `gpuPackMultipoleTracking` no longer keeps these magnets on the CPU. The kernel's
body step now starts at 0 for each substep as the CPU's does: left unset with no drift, it made the device's spin NaN
where the CPU's no longer was. A `KQUAD` with `L=0` and no strength now leaves spins as no magnet does on both.

**`&slice_analysis` with `s_start` (14).** The page now starts at the first element in range, so an `s_start` past the
first element no longer stops the run. CPU and GPU share this code.

**`CSRCSBEND` `beta0` after a loss (15).** The CPU now swaps `beta0[]` with the lost particle. The device's `CSRCSBEND`
body hands any slice that loses a particle back to the CPU with `beta0[]`, so GPU runs get the fix too. `MAXAMP`,
contours and aperture files there now run on the device too (below).

**Checks.** `magnet-kquad-spin-zero-length` is new: a zero-length `KQUAD` with kicks and a tilt, one with no strength
and a thin `FMULT`, with spin. `slice-analysis-whole-beam` now starts at `s_start=0.5`. Both agree with the CPU within
the envelope and pass the verify build, which compares `K0`, `KZ` and `FT` exactly. The new case first flagged the unset
body step, so it reaches the device's zero-length spin. The 660 CPU tests are unchanged but `photonEmission1`'s
`dictionary.sdds`.

See `history/2026-11-08_cpu-bug-fixes/REPORT.md` in GPU-Testing.

### `CSRCSBEND` apertures

`MAXAMP`, an `&aperture_data` file and a sticky `APCONTOUR` no longer keep a resident `CSRCSBEND` on the CPU. Each
body slice takes the aperture checks of `integrate_csbend_ordn()`: at the slice's entrance and exit, with the
`MULT_APERTURE_DATA` that `setupMultApertureData()` set up (file evaluated at the bend's center, reverse tilt), through
`gpuInsideApertureLimit`/`gpuInsideApertureContour`. A drift step that the CPU's integrator gives up on (`f <= 0`,
`|qx/f| > 1`, `cos = 0`) is a loss too, as on the CPU.

**Order.** The CPU swaps a lost particle with the last one not yet tracked, then tracks that one in its place. The
device tracks every particle in its own row, which gives the same results since particles don't interact within a
slice, then rebuilds the CPU's order: particles kept below the final count never move, so one thread visits only the
holes there and the places above, about twice the number lost, and a parallel gather moves the rows and `beta0[]`.
Survivors come out in the CPU's rows exactly, so outputs with particle rows match and later sums run in the same order.
Lost rows get what the CPU writes for them (slopes, the rotation of `convertFromCSBendCoords()`, the slice's start plus
`dz_lost`, `p`).

**When the CPU still does it.** At first lost particles stayed on the device only when nothing read their rows; since
the same day a losses file and acceptance output are handled too (below). Global loss coordinates still send the slice
with the first loss back to the CPU. So does a particle with a coordinate that isn't finite, which the device
can't follow the CPU through. After device losses, a later fallback copies `beta0[]` back to the host, since it has
moved.

**Checks.** 4 new feature cases (`csr-maxamp-loss`, `csr-apcontour-loss`, `csr-aperture-data-loss`,
`csr-maxamp-losses-file`), within the envelope against the CPU, with every survivor in the same row. The verify build
tracks a CPU shadow through the same swaps and compares every column of every survivor and `beta0[]` exactly, in slices
that lose up to 121 particles. Mistake builds that kept the survivors in another order, left `beta0[]` behind, ignored
the reverse tilt or ignored the file's center each failed the cases aimed at them. `csr-resident-body-loss` now keeps
its losses on the device.

The CPU used to report every `CSRCSBEND` body loss as if it were in the first slice. That is fixed as CPU bug 16: a
body loss is at the slice's start plus `dz_lost`, and the device's lost rows take the slice's start the same way. Only
a losses file reads them, and then the bend runs on the CPU.

See `history/2026-11-08_csr-apertures/REPORT.md` in GPU-Testing.

### Losses in the CPU's order, round by round

The device already rebuilt the CPU's swap order after losses (`gpuBuildCpuLossOrderDestination()`), but as one round.
Four CPU loops remove losses in rounds, each round running its swaps over the particles still there: `GKICKMAP` and
`UKICKMAP` kick by kick, `LGBEND` segment by segment, and `SCRAPER` with length at its entrance and then its exit.
Particles lost in more than one round then came out in another order, which was the five row-order flags of the
feature set (`run.acc`, `run.los` or `run.out` by `particleID`). The collimators already ran their two rounds on the
device (`gpuCollimatorLossOrderKernel`); every other element with device compaction loses particles in one round.

**The builder.** Each flag kernel now records the round a particle is lost in (`lossRound`: the kick, the segment, or
entrance and exit). `gpuBuildCpuRoundLossOrderDestination()` lists the lost particles by round and runs one thread
over them: in each round, particles kept below that round's final count don't move, so it visits the round's holes
there, in order, and the places above, about twice the number lost in all. `CSRCSBEND` body slices use it too, as one
round. For `LGBEND`, the CPU loses particles only in `integrate_kick_KnL()`; where the device's tracker gives up
elsewhere (a plane switch, a value that isn't finite) the CPU would carry a NaN, and the nearest segment stands in.

`ELEGANT_GPU_CHECK_LOSS_ORDER=1` also runs the CPU's loops themselves, round by round over every place, and stops if
any destination differs.

**OpenMP paths.** `CSBEND`, `CCBEND`, `LGBEND` and the multipoles, tracked on host threads in searches, compacted their
losses keeping survivors in their original order. `gpuCompactParticlesInCpuOrder()` now makes the serial loops'
swaps instead, so those paths leave the CPU's order too.

See `history/2026-11-08_cpu-order-losses/REPORT.md` in GPU-Testing.

### `BMXYZ` apertures

`BMXYZ` with `MAXAMP`, an `&aperture_data` file or a sticky `APCONTOUR` now runs on the device. `lorentz()` checks
the aperture in `bmapxyz_deriv_function()`, at every derivative of the non-adaptive Runge-Kutta, trial points
included, and keeps the first point outside as the lost particle's coordinates, at `z_start` plus that point's `z`.
The device checks at the same derivatives (`gpuBmxyzDerivatives()`), stops there, and writes the same row; losses go
to the end in the CPU's order through `gpuBuildCpuRoundLossOrderDestination()` (one round). The aperture data comes from
`setupMultApertureData()` at the element's start, as in `lorentz()`; global loss coordinates stay on the CPU, which
fills them only for obstructions. A losses file gets the lost rows from the device, and acceptance output moves with
the survivors, through the apertures' helpers. The verify build compares survivor counts and rows. Results are bit
for bit the CPU's on the direct decks. A mistake build that checked only at step ends, and one that left the element's
start out of the loss position, each failed `bmxyz-maxamp-losses-accepted`.

### Contours in the resident aperture search

The device-resident aperture search (drifts, `KQUAD`, `KSEXT`, `KOCT`, zero-length `RCOL`) now takes `APCONTOUR`
elements with no length or `DZ`, as do_tracking() treats them: `CANCEL`, or `HOLD_OFF` without `STICKY`, ends a sticky
contour; a sticky contour is checked inside the multipoles and after every element without `NO_APERTURE`
(`imposeApContour()`), a loss at that element's end. do_tracking() keeps a sticky contour into the next pass, where
elements before the one that set it check it too, so a program whose sticky contour would reach them is left to the
ordinary search. Contours are uploaded into a cache of 32, by their points, so several stay on the device at once;
`gpuPinApertureContours()` keeps a program's while it is packed. The multipole items needed their aperture data
packed for the contour check to undo a tilt: without it the check moved every point to the origin, which the new cases
caught. The verify build runs the resident program next to the ordinary search and compares every line.

For a convex contour the check after each element can't change which trial is lost, since a point inside a drift lies
between two points other checks test; a mistake build without it passed, and only non-convex contours could show it.

The fused tune-program prepass (frequency maps and tune footprints) runs only in the verify build and still leaves
contours to ordinary tracking. Tune footprints run that tracking on the device; batched frequency maps track on the CPU,
with `-ompThreads` when given, by design (`gpu_batched_frequency_map_cpu_tracking_required()`).

### `CSRCSBEND` losses with loss output or acceptance

With a losses file or acceptance output, `CSRCSBEND` body losses now stay on the device too: each slice's lost rows are
copied to the host, and the accepted rows move with the survivors, through the destinations the aperture helpers use.
Global loss coordinates still go to the CPU, which stops at the first loss in a `CSRCSBEND`, and so do the verify
build's, whose CPU shadow here keeps neither. A mistake build that copied no lost rows and moved no accepted rows
failed `csr-maxamp-losses-file` and `csr-maxamp-accepted`.

See `history/2026-11-08_three-features/REPORT.md` in GPU-Testing.

## Monitors and matrix magnets in the momentum search, and CPU bug 17 (November 9, 2026)

The independent momentum search in `gpu/momentum_search.h` runs non-fiducial searches, and fiducial ones whose lattice
`gpu_batched_search_beamline_supported()` refuses (a `CSBEND` with `K3` to `K8`, for example; since the radiation
change below, it takes every search it accepts). Unlike that generic
check, `gpu_momentum_search_beamline_supported()` keeps its own list of element types, and it refused `QUAD`, `SBEN`,
`RBEN`, `SEXT`, `KICKER`, `EKICKER` and the monitors, though all run on the GPU in ordinary tracking. It now takes
them. Monitors are refused if they store turn-by-turn readings, which average over the beam, so a batch would write
different readings than one particle at a time. `SREFFECTS` stays refused: even with `QEXCITATION=0` it draws three
random numbers per particle, so a batch would leave the random-number state different from the scalar search.
Every boundary trial is still tracked again on the CPU with the GPU off, and a lane that disagrees reruns the scalar
search, so the published apertures stay the CPU's.

CPU bug 17: the scalar search returned with its tracking wedge (the momentum offset at the element under test)
still installed. The independent search then refused to run ("external tracking callbacks are already installed"),
so later steps of the same `run_setup` stayed on the CPU; and the wedge fired in any later multi-turn tracking at that
element, on CPU and GPU alike. `doMomentumApertureSearch()` now clears it. See bug 17 in
`elegant-CPU-bugs-and-fixes.md`.

**Checks.** Three new feature cases run in auto mode, so they have GPU activity only if the batch runs:
`search-momentum-matrix-elements` (non-fiducial), `search-momentum-matrix-elements-fiducial` (with a `K3` bend) and
`search-momentum-after-scalar` (a scalar search, then a batched one in a second `run_setup`). All three match the CPU
exactly, also with `-ompThreads=4`, and pass the verify build (17,664 element checks). A build that refused monitors
again failed all three for missing GPU activity, and one without the wedge fix failed `search-momentum-after-scalar`.
The other 358 cases are identical to the previous CPU and GPU references. The forced routine list (424) is identical to
its reference. The 12 elegantTestSet `momentum_aperture` tests in default mode match the CPU exactly; `momAp4`,
`touschek4` and `touschek7` now batch, `momAp5` now stops at its `SREFFECTS`, and the second steps of `momAp7` and
`momAp8` no longer see a leftover wedge. The CPU 660 are identical to the previous CPU reference.
The slow-test copies `momAp1` (a chicane with `SBEN`s, 4 passes) and `momAp8` (`MONI`, two steps) now batch and match
the CPU exactly; `momAp1` sends one lane back to the CPU, where a trial lands on `AP1`'s edge to the last bit.

See `history/2026-11-09_momentum-search/REPORT.md` in GPU-Testing.

### Monitors in the fused tune prepass

The fused tune-program prepass, which runs only in the verify build, skipped `MONI`, `HMON` and `VMON` like markers,
so a monitor's length was left out of the ring. In a frequency map with 3.5 m of monitors, it kept 6 particles the CPU
lost, and the verify build stopped. Monitors with length are now packed as their drift matrices; the
`NO_APERTURE` flag already leaves out the aperture check after them, as on the CPU. Production builds never run
the prepass. `search-frequency-map-prepass-monitors` is the deck; the 14 frequency map and tune footprint cases
match the CPU exactly under the verify build, and the other 361 feature cases are identical to before.

See `history/2026-11-09_monitor-prepass/REPORT.md` in GPU-Testing.

## Batched searches through bends with `K3` to `K8` (November 9, 2026)

`gpu_batched_search_beamline_supported()` refused any `CSBEND` with `K3` to `K8`, waiting for search baselines for
high-order bend maps. That kept the batched `&find_aperture` line searches and the generic batched fiducial
`&momentum_aperture` off such lattices (`multipoles2` in elegantTestSet, for one), and sent fiducial momentum searches on
them to the independent search instead. Bends with radiation stay refused; only `K3` to `K8` are allowed now. The
kernel already carried these terms, and ordinary tracking already used it.

The baselines found one difference that had nothing to do with the bends: the generic momentum search computed its
trial momenta as `start + i*interval` and rebuilt each split's start from the reduced interval, while the scalar
search adds the interval step by step. The reported momenta then differed from the CPU's in the last bit, and the
tunes at loss by up to 1.6e-9. It now steps as the scalar search does, so those columns match the CPU exactly; only the
coordinates of lost particles differ, by rounding (up to 1e-11 relative, as large without `K3` to `K8`).

**Checks.** New cases: `magnet-csbend-high-order` (the first to cover these terms at all), `search-aperture-csbend-high-order`
and `search-momentum-csbend-high-order`, all within rounding of the CPU and passing the verify build.
`search-momentum-matrix-elements-fiducial` now uses a radiating bend to stay on the independent search. Mistake builds:
the old rule failed both search cases (no GPU activity); dropping `K5` to `K8` on the GPU failed all three; the old
trial arithmetic failed `search-momentum-csbend-high-order`. The other 361 feature cases are identical to before.
With OpenMP, 30 of the 34 search cases match the CPU exactly and 4 are within rounding. The forced routine list is
identical to its reference. The 18 elegantTestSet decks with such bends or momentum searches match the CPU exactly in
default mode, where `multipoles2`'s search now batches (on the CPU at default thresholds) and `momAp8`'s takes the
generic path. With 64-particle thresholds, the slow-test copies of both run on the GPU with the CPU's apertures; the
positions of particles lost while blowing up differ, as much as a 1e-15 nudge to the CPU's lattice moves them.

See `history/2026-11-09_high-order-search/REPORT.md` in GPU-Testing.

## Radiation in batched searches, and which momentum search runs (November 9, 2026)

`gpu_batched_search_beamline_supported()` no longer refuses `CSBEND` and `KQUAD` with `SYNCH_RAD`, the last of its
special cases; it now refuses only elements the GPU can't track in a batch. Radiation was already allowed there in
`KSEXT`, `CCBEND` and `LGBEND`, and the independent momentum search already took it.

That rule also decided between the two batched momentum searches: a fiducial search on an accepted lattice went to the
generic one (`doMomentumApertureSearchBatched()`), which publishes the GPU's results, instead of the independent one,
which tracks every boundary trial again on the CPU and publishes those. So lifting it would have moved radiating decks
such as `momAp4` to the less exact path, as the `K3` to `K8` change had already done to `momAp8`'s copy (flagged for
the positions of lost particles). Two changes keep every result the CPU's where it was:

- The independent search now takes every search it accepts, fiducial or not; the generic one takes only what it
  refuses. This is also faster: the GPU test set's `momentum-aperture-batched-heavy` went from 4.1 s to 1.9 s (15.5 s
  on the CPU) and now matches the CPU exactly.
- The generic search now needs as many lanes (two per location) as the independent one. With fewer, it had batched
  every candidate of a split for every turn, slower than the scalar search: `momAp7` took 18.3 s against 12.6 s,
  and `momAp8` 8.4 s against 6.6 s, since radiation and `K3` to `K8` let them in. They run the scalar search again.

**Checks.** New cases `search-aperture-radiation` (within rounding: one lost position, 2e-12),
`search-momentum-radiation` (identical to the CPU) and `search-momentum-generic` (the generic path, now that the
independent one takes the others: a `WATCH`, which only the generic path accepts; momenta and tunes identical, lost
positions within rounding). `search-momentum-csbend-high-order` moved to 32 locations, on the independent search,
and is identical to the CPU. Radiation changes the results (aperture area 1.021e-4 against 1.065e-4 without it).
Mistake builds: refusing radiation again failed `search-aperture-radiation` (no GPU activity); the old routing made
`search-momentum-radiation` differ from the CPU and the GPU reference; the old trial arithmetic made
`search-momentum-generic` differ from the GPU reference. The other 364 feature cases are identical to before.
The new and moved cases pass the verify build. With OpenMP, 32 of the 37 search cases match the CPU exactly and 5 are
within rounding. The forced routine list is identical to its reference; the 18 elegantTestSet decks with momentum
searches or high-order bends match the CPU exactly in default mode; the slow-test copies `momAp1`, `4`, `5`, `7`, `8`
and `9` match the CPU exactly. In the GPU test set only `momentum-aperture-batched-heavy` changed, to identical with the
CPU.

See `history/2026-11-09_radiation-search/REPORT.md` in GPU-Testing.

## A CPU re-check in the batched aperture line search (November 9, 2026)

The batched `n-line` `&find_aperture` (`do_aperture_search_line_batched()`, host-batched or device-resident) published
the GPU's results. Its boundaries matched the CPU in every test, but the positions of lost particles were the GPU's,
within rounding, and far from the axis, where particles blow up, by much more (`multipoles2`'s copy, flagged for 3.9e-2).
It now does what the independent momentum search does: after the batch, it tracks each line's last surviving trial
and the first loss it reports again on the CPU, with the GPU off, as the scalar search tracks them. When they agree,
the loss columns take the CPU's values. When either disagrees, it prints a notice and the scalar search reruns the
whole command. That costs at most two particles per line. Both paths now return where the reported lost trial started
(`x0 + firstLost*dx`, as the boundary itself is computed); the resident kernel passes it back with its other line
results, and the verify build compares the two paths' values.

**Checks.** All 17 `search-aperture-*` feature cases match the CPU exactly; four were within rounding before. A build
without the re-check put those four back within rounding, which the GPU reference catches. A build that recorded the
wrong starting point on the host path made 7 searches disagree and rerun as scalar searches, with output identical
to the CPU, which shows the fallback works. The other 351 feature cases are identical to before. The verify build passes all 17 (648,494 checks) with output identical to the CPU. With OpenMP, 36 of the 37 search
cases match the CPU exactly; only `search-momentum-generic` is within rounding. The 9 elegantTestSet decks with line
searches (`daLine1` to `5`, `dynAp6`, `dynAp7`, `latticeErrors3`, `multipoles2`) match the CPU exactly in default
mode, the slow-test copies `multipoles2` (no longer flagged) and `daLine4` on the GPU, the GPU test set its GPU
reference, and the forced routine list its reference.

See `history/2026-11-09_aperture-recheck/REPORT.md` in GPU-Testing.

## A CPU re-check of losses in the generic batched momentum search (November 9, 2026)

The generic batched momentum search (`doMomentumApertureSearchBatched()`), now used only for searches the independent
one refuses, already tracked each task's final survivor again on the CPU to recompute its tunes, but ignored whether
it survived, and kept the GPU's loss columns. It now also tracks the first loss again, the same way (one particle,
the scalar search's wedge, GPU off). The survivor must survive and the loss must be lost; the loss columns then take
the CPU's values. Otherwise it prints a notice and the scalar search reruns the command. With the aperture line
search and the independent momentum search, every batched aperture and momentum search now publishes the CPU's
results. (Batched tune footprints still publish tunes tracked on the GPU.)

**Checks.** `search-momentum-generic` matches the CPU exactly (it was within rounding) and is the only feature case that
changed. A build without the loss re-check put it back within rounding, which the GPU reference catches; one that
treated every confirmed loss as a disagreement fell back to the scalar search, with output identical to the CPU.
The verify build passes the six momentum cases with output identical to the CPU. With OpenMP, all 37 search cases
match the CPU exactly. The 18 elegantTestSet decks with momentum searches or high-order bends, the GPU test set and
the forced routine list are unchanged, with no fallbacks.

See `history/2026-11-09_generic-recheck/REPORT.md` in GPU-Testing.

## Wakes and `TRFMODE` in the CPU's order, and skipped particles in batched tune footprints (November 9, 2026)

**Wakes with output files.** With order-sensitive output, the device now follows `wake.c`, `trwake.c` and `rfcw.c`
step for step. `TRWAKE` bin sums are added one thread per bin, in index order, instead of by a block reduction. With
`N_BINS`, `WAKE`, and `TRWAKE` with more than one bunch, center the bins on the beam's mean time, which is now summed
in index order (`gpuCudaOrderedTimeSum()`). A tilted `TRWAKE` rotates the particles outside the bins into the element
and back, as the CPU does, and takes `pz` from the rotated slopes. Drive and probe powers group their products as
`ipow()` does (`gpuCpuIpow()`). `ELEGANT_GPU_WAKE_CPU_ORDER=0` or `1` overrides the default, which is on only with
order-sensitive output.

**`TRFMODE` and `FTRFMODE`.** Each bin's x and y sums were atomic adds. One thread per bin now adds them in particle
order, as `trfmode.c` and `ftrfmode.c` do.

**Batched tune footprints.** These published tunes that differed from the scalar path's in 2 of 441 points of a
`POLYSERIES` footprint. A CPU bug in `polynomialSeries_tracking()` caused it: when a particle's transverse momentum
exceeds its total, it leaves the particle loop with `break`. That particle is never lost, and every particle after it
skips the element on that pass and on every later one. The GPU kernel already falls back to the CPU function then, so
the GPU matched the CPU batch exactly; the scalar path tracks each particle alone, with nothing after it.
`taylorSeries.c` now reports the particles a stopped pass skips to a registered function
(`setPolynomialSeriesStopFunction()`), and `computeTunesFromTrackingBatch()` reruns each of them alone with
`computeTunesFromTracking()`. The batched frequency map gets the same fix. The CPU's physics is unchanged; the bug is
listed in Part 3 of `elegant-CPU-bugs-and-fixes.md`. (Later the same day the CPU bug was fixed as bug 21, so the
particle is lost, and this rerun was removed; see the next section but one.)

**Checks.** The 21 feature cases with `RFCW`, `WAKE` or `TRWAKE` that changed now have `run.out` identical to the
CPU's (it differed by rounding); their centroid and sigma files still differ by rounding, because the feature suite
forces beam sums onto the device. The other 347 are unchanged: 41 identical to the CPU, 1 flagged (`scmult-zero-length`),
as before. The verify build passes the 29 wake, 3 `POLYSERIES` and 2 tune-footprint cases with no difference in any
of 2,270 element checks. A 441-point `POLYSERIES` footprint matches the CPU exactly (2 points differed), as do maps
built to overflow. The two `TRFMODE` test set cases repeat exactly on a quiet or a busy GPU, where the binary before
this change differed on every run. The routine list is unchanged, and 25 elegantTestSet decks with wakes, `TRFMODE`,
footprints and frequency maps in default mode are unchanged or match the CPU, apart from flags they already had.
Mistake builds: without the stop function the 2 points differ again; with the CPU-order wake sums off, 17 wake cases'
`run.out` differ; without the rotation outside the bins, `wake-trwake-tilt` differs; the old atomic adds vary again.
`pz` from rotated slopes and `ipow()` grouping change nothing in any case that can be built (1 + x'² + y'² rounds the
same at ordinary slopes); they follow the CPU's code. Found on the way: `TRWAKE` with `N_BINS` and one bunch applied no
kick on the CPU when the beam wasn't near t = 0, and the GPU did the same; fixed in both as CPU bug 18, below.

See `history/2026-11-09_exactness/REPORT.md` in GPU-Testing.

## `TRWAKE` bins for a single bunch, CPU bug 18 (November 9, 2026)

With `N_BINS` set, `track_through_trwake()` centered a single bunch's bins on t = 0, because only bunched beam mode
computed `tmean`; a bunch more than a few bin widths away got no transverse kick, with an "Only 0 of N particles were
binned" warning. `WAKE` lost the same line in 2014 and got it back six days later; `TRWAKE` didn't. The CPU now takes
the bunch's mean time (over all workers for Pelegant's distributed beam), and `gpuTrackTrwakeBucket()` drops the copy
it had made of the old bins: in index order with output files, as for `WAKE`, otherwise from the device's sums.

**Checks.** Only the five feature cases with fixed bins and one bunch changed, on the CPU and the GPU, and they now
bin every particle; their `run.out` matches the CPU exactly. The other 363 are unchanged, and against the new CPU
reference the GPU is as before: 41 identical, 1 flagged (`scmult-zero-length`). The verify build passes the 29 wake
cases with no difference in 2,118 element checks. The binary from before the fix is flagged in all five against the
fixed CPU. The 34 elegantTestSet decks with `TRWAKE` or `RFCW` are identical to the CPU baseline, as none sets
`N_BINS`; neither does the GPU test set.

See `history/2026-11-09_trwake-tmean/REPORT.md` in GPU-Testing.

## Pelegant wake bins, exact corrector losses and `POLYSERIES` losses, CPU bugs 19 to 21 (November 9, 2026)

**Pelegant's wake bins (bug 19).** With `N_BINS`, each worker centered `WAKE` and `TRWAKE` bins on its own particles'
mean time, then the workers' bins were added together out of line. `computeBunchTimeMeans_p()` now gives every worker
each bunch's mean over all of them, in one `MPI_Allreduce` before the bunch loop. Serial elegant is unchanged, and
`gpu-elegant` doesn't use MPI.

**Exact corrector losses (bug 20).** `do_tracking()` ignored `trackThroughExactCorrector()`'s survivor count, so a lost
particle kept going with βγ in its momentum slot. The GPU branch already used its count; the CPU now does too.

**`POLYSERIES` losses (bug 21).** A particle with more transverse momentum than total stopped the pass for itself and
every particle after it; it is now lost. The GPU kernel already handed such passes to the CPU function, and the batched
tune footprint's rerun of skipped particles is removed, since nothing is skipped any more.

**Checks.** `Pelegant -np 4` against serial elegant, from the same beam file: `WAKE` with `N_BINS`, one bunch, from
7.7e-6 in `p` to 0; the bunched cases `wake-bunched-select-range` and `-select-one` from 1.4e-4 and 3.5e-4 to a few
ulp. A corrector deck that loses 85 of 3,000 particles: the CPU now writes the 85 loss rows `gpu-elegant` writes, and
the survivors agree to 6.4e-15. The 441-particle `POLYSERIES` track loses its six unstable particles at the passes where
they used to freeze, on CPU and GPU alike, and the batched footprint matches the scalar CPU exactly with no rerun.
The 368 feature cases, the GPU test set, its four CPU cases with these elements, and 31 elegantTestSet decks with exact
correctors or `POLYSERIES` are all unchanged; the verify build passes the nine affected feature cases. The feature-set
references stay those of bug 18 (`2026-11-09_trwake-tmean`), since no case changed.

See `history/2026-11-09_three-fixes/REPORT.md` in GPU-Testing.

## Pelegant bunched wakes with a bunch missing from a worker, CPU bug 22 (November 9, 2026)

In bunched mode, a Pelegant worker with none of a bunch's particles skipped that bunch in `WAKE`, `TRWAKE` and `CWAKE`,
and with it the bunch's collective calls, so later calls were paired with other bunches': with a four-bunch beam read
in bunch order, `WAKE` hung and `CWAKE` aborted on a false beam-length error. A distributed beam now skips a bunch only
when it is empty on every worker, using the counts from bug 19's `computeBunchTimeMeans_p()`, and `CWAKE` also centers
`N_BINS` bins on the mean over all workers (bug 19). With `-np 4` and `-np 7` the runs finish and match serial elegant to
a few ulp, as does GPU Pelegant. `gpu-elegant` doesn't use MPI; serial elegant, the 368 feature cases on CPU and GPU, and
35 elegantTestSet decks with these elements are unchanged.

See `history/2026-11-09_empty-bunch/REPORT.md` in GPU-Testing.

## Two crashes, CPU bugs 23 and 24, and the full 660 (November 9, 2026)

**Bug 23.** `&moments_output` on a one-element beamline read the second element's Coulomb log for its `_BEG_` row and
crashed; it now uses the first element when there's no second. No GPU path.

**Bug 24.** With `n_passes` above 1, the scalar momentum search's wedge stored a turn on every pass but reset its count
only on the offset pass, so trials lost before that pass wrote past the turn record (`momAp6` with `n_passes = 4`).
It now stores from the offset pass on, as the batched wedge counts. `gpu-elegant` reaches this search when it falls
back and crashed the same way; both now finish with identical output.

**Full runs after these.** CPU 660: all identical to the 11-08 CPU baseline, so CPU bugs 17 to 24 change no
elegantTestSet output. GPU 660 in default mode: against the new CPU run 574 identical, 26 within rounding, 60 flagged,
every one already flagged on 11-08 (61 then; `photonEmission1`'s `dictionary.sdds` now matches); against the 11-08 GPU
run only six moved: the `bunchComp` CSR decks' sigma files (same columns, same largest absolute differences), `momAp6`
(with the CPU, from bug 16) and `photonEmission1`. The 368 feature cases are unchanged on CPU and GPU. The 54 slow-test copies give the 15 known flags against the CPU, at the sizes listed on 2026-11-09, and nothing new.

See `history/2026-11-09_crashes/REPORT.md` and `history/2026-11-09_full-660/REPORT.md` in GPU-Testing.

## gpu-Pelegant on the 340 Pelegant tests (November 9, 2026)

The test harness takes `--launcher` (here `mpiexec -np 4`) and `--reference-directory preference`. On one GPU, the
first gpu-Pelegant run hung in five tests and crashed in five.

**Hangs.** The master holds no particles and stays on the CPU, so a GPU path that replaces a CPU routine making a
collective call leaves the ranks pairing different calls. With a distributed beam these now stay on the CPU: `RFCA`
without length (the CPU first sums the particle count over all ranks; `feedback3`, `4`, `7`), `MATR` and `EMATRIX`
first passes and `ENERGY` (averages over the workers), `CSRDRIFT` with nothing to do (a count sum), `SCMULT` (beam
sizes over the workers; the device's were local) and the opt-in `WAKE`/`TRWAKE` drift paths (histograms over the
workers). `gpu_do_match_energy()` had made an `MPI_Allreduce` of its own before `do_match_energy()`'s
(`accelBeta3`, `full457MeV`, both `always_change_p0`); it now goes straight to the CPU routine. The `CSRCSBEND`
device entry is off in MPI builds, since `csbend.c` builds its CUDA CSR steps only without MPI and the entry ran the
CPU's steps on a possibly stale host copy. The other GPU element paths make no collective call on the CPU side, keep
the CPU's (`RAMPP`, `CCBEND`/`LGBEND` centroid output), or were already CPU-only under MPI.

**Crashes.** After a serial element gathered the beam (`WATCH` FFT, `REMCOR`; `losses1`, `remcor1`, `ringRF1` to `3`),
`scatterParticles()` took the master's host array from the CUDA layer, which still held the one from the start of
tracking. `gatherParticles()` now passes its array on with `gpuSetCpuParticleArray()`.

**Acceptance rows.** The device partitions all rows of `accepted` at an aperture, lost rows after the survivors as on
the CPU, but counted only the survivors' rows as valid, so a copy back left the lost particles' rows on the host in
their old order. Serial elegant never reads them, but Pelegant's `SCRIPT` hands a worker extra particles that pick up
those rows (`trackingInterrupt1`, forced). `gpuStablePartitionAcceptedOnDevice()` now keeps every row held since the
upload and `gpuCopyAcceptedDeviceToHost()` copies them all; serial outputs are unchanged (feature set, GPU test set,
the 37 elegantTestSet decks with acceptance output in both modes). CPU Pelegant's `SCRIPT` itself moves particles
between workers without their `accepted` rows (68 of 5,039 rows wrong in that test; none in serial), a CPU bug.

**Results.** In default mode gpu-Pelegant matches a CPU Pelegant built with the same flags (`-Og`) on 333 of the 335
tests both finish: `expandHamiltonian6` is within rounding, as in serial, and `geneticOpt1` differs between two CPU
runs. The committed CPU Pelegant build has `-O3` while gpu-Pelegant adds `-Og` (`Makefile.mpi`), which alone changes
`bunchComp1`, `bunchComp1m` and `bunchComp3` (from the first `CSRDRIFT` after a CSR bend, relative 6e-12, growing
downstream); those decks put no element on the device. The five failing everywhere are `elasticScattering1` and
`inelasticScattering1` to `3` (a CPU bug: `MPI_Win lastPassWin` is never freed, and UCX aborts in `MPI_Finalize`) and
`backwardThenForward1` (its `runScript` names missing decks; run directly, CPU and GPU match). In forced mode, on the
310 tests without aperture, momentum and frequency map searches, 297 of 304 match, `expandHamiltonian6` and `mult4`
are within rounding, `ERL2` (`ISR`), `radDamping6` and `radDamping7` (radiation) are rounding as in the serial forced
review, and `trackingInterrupt1` matches after the acceptance fix. The searches weren't run forced: with four ranks
sharing one GPU they run hundreds of times slower and time out while busy.

See `history/2026-11-09_pelegant/REPORT.md` in GPU-Testing.

## Three slowdowns from the timing report (November 10, 2026)

**`TRFMODE` and `FTRFMODE` bin sums.** Since the repeatability change of November 9, one thread per bin scanned
the whole beam for its particles. A stable radix sort of particle indices by bin (CUB) now keeps each bin's particles
in particle order, and each bin's thread adds only its own, with the same arithmetic, so the sums are bit for bit what
they were. Beams beyond CUB's `int` item count keep the old kernel. The two GPU test set cases go from 16.4 s to
1.8 s (the CPU takes 3.9 s), and their outputs are identical to the previous GPU reference.

**CUDA context at startup.** In auto mode device 0 is now selected at its first use, through
`gpuSelectDevice()` (set the device and probe for usable kernels, as `gpuBaseInit()` did), called where an
element, reduction, beam sum, `SCMULT`, CSR wake, `RFCW` LSC kick, tune prepass or batched search first puts work on
the device. Required mode, verbose output and any other device still select at once. `cudaFree()` on a null pointer
creates a context, so the release functions now skip null pointers (`gpuFreeDevice()`). A run that never uses the
GPU starts in 0.05 s instead of 0.21 s (elegant: 0.02 s), and in 0.02 s with `ELEGANT_GPU_MODE=off`. A path that
reaches the device without passing one of these still gets device 0's primary context from the runtime, so a single
GPU sees no difference; only the unsupported-card fallback would be skipped on that path.

**Searches.** `setElementGpuData()` skips the element checks when the beam is on the host and below every threshold
the element's type can use (`gpuElementHasOwnThreshold()` lists the types with thresholds of their own); not in
required mode or verify builds. The batched line aperture search now runs only when it can pay
(`gpu_batched_aperture_search_worthwhile()`): when the device-resident search takes the whole beamline, when its
batches reach the element threshold, in required mode, or with more than one OpenMP thread. Otherwise the scalar
search runs, as in elegant; on SPEAR with 231 to 1,681 trials the batched one took 13% to 34% longer with the same
apertures. The resident search also checks its beamline before selecting the device. `fmap3` goes from 46.9 s to
34.4 s (elegant 32.5 s), `daLine4` 32.8 s to 25.9 s (24.5 s), `multipoles2` 18.4 s to 15.9 s (15.3 s).

**Environment reads.** `gpuBaseInit()` runs at every tracking call and read about 115 `ELEGANT_GPU_*` variables
with `getenv()`, which scans the whole environment. Decks that track one particle tens of thousands of times paid
for it in every mode, `off` included: `crossPlane1` (60,542 tracking calls) took 1.30 s against elegant's 1.00 s.
`gpuGetenv()` now looks each name up once and keeps the result, keyed by the name's address, since every name is a
string constant and elegant never changes its environment. Run alone, `crossPlane1` now takes 1.04 s to 1.09 s with
the GPU off and 1.16 s in auto mode. Found with gcov line counts from the `GPU_COVERAGE` build.

**Per-element calls.** When a tracking call starts without a usable device (`off` mode, no device, suppressed
replay tracking, a Pelegant master without particles), `gpuElementHooksNeeded()` returns 0 and `do_tracking()` skips
`gpuSetAfterElementAperture()`, `setElementGpuData()` and the `coord` refresh at each element. A device never comes
back during a tracking call, and those calls would only mark the host current again. Verify builds always make them.
`rfmode2` (one particle, 1,000,000 passes) with the GPU off: 2.54 s to 2.45 s.

**Checks.** Feature set (368), forced routine list (424), GPU test set (39) and the 47 Pelegant tests that use the
GPU: all identical to their references. All 660 elegantTestSet decks in default mode match the November 9 GPU
reference. Their total time went from 4,499 s to 4,257 s (elegant: 5,000 s), and in the 490 decks that don't use the
GPU the median extra time per run went from 0.17 s to 0.02 s. These timings predate the `getenv()` and per-element
changes. The final build, with the OpenMP change below, gave identical output again on the feature set, the routine
list, the GPU test set, the 47 Pelegant tests and all 660 decks; 205 decks with RF-mode, wake, `LGBEND`, `CSRCSBEND`,
`EDRIFT` or tilted elements also match with `-ompThreads=4`. The CPU-only build compiles the changed files cleanly.

**OpenMP `if` clauses.** Particle loops that the GPU build runs on OpenMP tracking threads (`-ompThreads`) carried
`#pragma omp parallel for ... if(gpuOmpTrackingRequested(np))` or `#pragma omp taskloop if(...)`. With gcc a false
`if` still enters the OpenMP runtime, about 0.15 us a call, at every such element and in `exactDrift()` and
`rotateBeamCoordinates()`. `GPU_OMP_PARALLEL_FOR(condition, loop)` and `GPU_OMP_TASKLOOP(condition, loop)` in
`track.h` now test the condition in C and use the pragma without `if` on the threaded branch; other builds run the
loop as written. 27 sites in `rfmode.cc`, `frfmode.c`, `trfmode.c`, `ftrfmode.c`, `trwake.c`, `wake.c`, `cwake.c`,
`csbend.c`, `tilt_matrices.c`, `lgbend.c` and `insertSCeffects.cc`. Left as they were: the `GPU_VERIFY` checks in
`insertSCeffects.cc`, regions entered once per search (`frequencyMap.c`, `momentumAperture.c`), and the wiggler and
`BGGEXP` integrators (`GWigSymplecticPass.c`, `magneticFieldFromGeneralizedGradient.c`), whose per-particle work
dwarfs the call and whose loops carry long `private` lists. No loop-written variable is read after a converted
taskloop, so the serial branch matches the old one-thread behavior. `rfmode2` with the GPU off: 2.45 s to 2.12 s
(elegant 2.10 s); in auto mode 2.69 s to 2.33 s, the rest being the GPU layer's per-element checks with a device
present.

See `history/2026-11-10_timing/REPORT.md` in GPU-Testing.

## RF cavity and wake paths on by default (November 10, 2026)

**What changed.** In `gpu-elegant`, `ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT`, `ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT`
and `ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT` are now on unless set to 0, so `RFCA` and `RFCW` with `CHANGE_P0` or
`always_change_p0`, `WAKE`, and `RFCW` with a longitudinal wake or LSC run on the device at the usual thresholds.
`gpu-Pelegant` keeps all three off unless set to 1; it wasn't validated. The change is in `gpuBaseInit()` in
`gpu_stub.c`; no CPU source and no kernel changed.

Three wake cases still need `ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT=1` set explicitly, because on the device they
were slower than the CPU (`gpuSlowWakePathsRequested()`):

| Case | Why it is slow | Measured |
| --- | --- | --- |
| `WAKE`, `TRWAKE` or `RFCW` with a fixed `N_BINS`, when output needs CPU order | The mean time is added by one device thread | 5.9 ms per wake for 10,000 particles; `bunchComp10` 4.7 s to 20.5 s with the slice sums below |
| `TRWAKE`, alone or in an `RFCW`, when output needs CPU order | Each bin's sums are added by one thread that scans the whole beam | `multibunchPartialShortRangeWakes` 0.57 s to 2.3 s |
| `RFCW` with `N_KICKS` above 0 and a wake | The wake kernels and their 64-thread sums run at every kick | `LCLS5` (200,000 particles) 6.4 s to 14.7 s; 42 ms a cavity against 15 ms on the CPU |

`ELEGANT_GPU_ENABLE_SLICE_SUMS` is now off unless set to 1. With one device thread per sum, slice analysis took
5.6 ms per element for 10,000 particles and 63 ms for 100,000; the read-only copy with the CPU's sums takes a tenth
of that, and `bunchComp10`'s outputs are identical either way. The feature suite sets the variable to 1 so its two
slice cases still run the device sums.

**Why the differences are rounding.** The verify build, with the three paths on and thresholds of 64, checked every
element against the CPU from the same input in the 56 elegantTestSet decks the paths reach or that have these
elements (`results/2026-11-10_cavity-defaults/verify-all3`, and `final-verify` for the final defaults):

| Check | Checks | Result |
| --- | ---: | --- |
| `track_through_rfcw` | 9,104 | Identical |
| `track_through_rfcw` with LSC (`NLSJitter1`) | 2,592 | Identical |
| `do_match_energy` | 7,871 | Identical |
| `track_through_wake`, `track_through_trwake` | 3,103 and 55 | Identical |
| `set_central_momentum` | 1,314 | Identical |
| `simple_rf_cavity` (`RFCA`) | 88,871 | Identical in 30 of 36 decks |

The counts are from `verify-all3`, where the three held-back cases ran as well. The six decks with `RFCA` differences are `accelBeta`, `accelBeta1`, `accelBeta2`, `accelBeta3`, `accelBeta5` and
`trajCorrect2`. None writes output that needs CPU order, so the device uses its own `sin` and `log` and a parallel sum
for the mean momentum. The largest difference per cavity is 2.8e-14 m in `s`, one unit in the last place. With
`ELEGANT_GPU_RFCA_REFERENCE_MATH=1` and `ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER=1`, every `RFCA` and `do_match_energy`
check in those six decks is identical too (`verify-all3-cpumath`), so those are the only sources.

`accelBeta2`, `accelBeta3` and `accelBeta5` show the largest output differences (`alphayBeam` 6.3e-10 in
`accelBeta5`, where alpha passes through zero). CPU elegant with the cavity voltage scaled by 1 + 1e-15 moves the same
columns by the same amounts, within a factor of ten and usually two (`alphayBeam` 6.1e-10): these decks amplify a
last-place change through 16 to 120 cavities. `trajCorrect2`'s 25 fields outside the default envelope under forced
thresholds are all inside the limits of its approved review policy (largest: `Cx` 7.2e-15 m against 1e-14); in
default mode it is within the default envelope.

**Outputs.** Final build, compared with the previous GPU run and with the CPU:

| Run | Against the previous GPU run | Against the CPU, before | Against the CPU, after |
| --- | --- | --- | --- |
| 660 elegantTestSet decks, default mode | 626 identical, 34 changed | 574 identical, 26 rounding, 60 flagged | 572, 26, 62 |
| 424 routine tests, thresholds of 64 | 385 identical, 39 changed | 240, 69, 115 | 237, 70, 117 |
| Feature set (368), GPU test set (39), 47 Pelegant tests | All identical | | Feature set: 1 flagged, as before |

The two new flags in each elegantTestSet run are `ibsAccel1` and `ibsAccel2`: 1.8e-15 m in the extremes of `s`, one
unit in the last place and 2.1e-12 relative. In 22 of the 34 changed default-mode decks the largest difference from
the CPU is the same field and size as before. The verify build stops nowhere in the 56 decks with the final defaults.

**Speed.** Each deck run alone on one P-core, previous defaults (set by the four variables) against the new ones, same
binary. The 40 decks that changed or were suspected: 217.7 s to 147.9 s. `LCLSII-1` 9.0 s to 4.7 s, `LCLSII-2` 12.0 s
to 5.6 s, `trajCorrect2` 36.6 s to 14.0 s, `bunchComp1` 17.7 s to 8.4 s, `bunchComp1m` 14.4 s to 5.2 s, `accelBeta3`
3.0 s to 0.7 s, `full457MeV` 2.8 s to 0.7 s. The decks that looked slower in single runs (`firstIsFid2`, `firstIsFid3`, `LCLS3`,
`LCLS7`, `scraper1`, `rfcw1`, `multibunchPartialShortRangeWakes`) use the same CPU time in alternating runs; their
wall times scatter by half a second. The 200,000-particle LCLS-II case of the timing report goes from 60.6 s to 21.2 s (the CPU
takes 228 s).

**What it leaves.** The three held-back wake cases and the device slice sums need sums that use more than one thread
or 64 threads. `gpuWakeTimeSums()` also runs a full beam-sums kernel on one 64-thread block to get the time span.
The three wake cases were released later the same day; see the next section.

The feature suite's `suite.json` gained `ELEGANT_GPU_ENABLE_SLICE_SUMS`, so results from before this change can't be
compared with later ones by the harness. Its new references are `final-feature` (GPU) and `feature-cpu-ref` (CPU) in
`results/2026-11-10_cavity-defaults`; the same build with the old suite file and the variable set in the environment
matched the previous GPU reference in all 368 cases. The new routine reference is `final-routine` there.

## Wake sums and cavity reference times off one thread (November 10, 2026)

**What changed.** Five steps in the wake and cavity paths did work for the whole beam on one device thread, one
64-thread block, or the host after a copy of the beam. Each now gives the same numbers sooner, and the three wake
cases held back in the section above are on by default (`gpuSlowWakePathsRequested()` is gone).

| Step | Before | Now | Numbers |
| --- | --- | --- | --- |
| Mean time for a fixed `N_BINS`, CPU order (`gpuCudaOrderedTimeSum`) | One device thread added the times: 5.9 ms for 10,000 particles | The device makes each particle's time; the host adds them in particle order | Identical |
| `TRWAKE` bin sums, CPU order (`gpuTrwakeSortedBinSumsKernel`) | One thread per bin scanned the whole beam | A stable sort of particle indices by bin (CUB, as for `TRFMODE`); one thread per bin adds its own particles in order | Identical |
| Time span for the wake bins (`gpuCudaTimeSpan`) | `gpuBeamSumsKernel`: every sum and product of the beam on one 64-thread block | Count, least and greatest time only, one particle per thread | Identical: comparisons only |
| Cavity reference time, CPU math (`gpuFiducialTimeFromDeviceValues`) | The whole beam copied to the host for `findFiducialTime()`: 3.1 ms a cavity for 200,000 particles | The device makes `findFiducialTime()`'s value for each particle; the host takes the first, the one of greatest delta, or adds them with the CPU's Kahan sum | Identical |
| Cavity reference time without CPU math | Sums on one 64-thread block: 2.8 ms a cavity for 200,000 particles | The same path as above, or the CPU routine when the host has the beam | Now the CPU's time exactly; they differed by up to 5e-23 s before |

The unordered mean time for a fixed `N_BINS` (no output that needs CPU order) still comes from the old kernels, so
those numbers don't move. `gpuRfcaReferenceKicks()`'s host loop for the CPU's `sin` and `log` now uses the host
threads the ISR numbers use (`gpuIsrHostThreads()`: one unless the run asks for more) from 4,096 particles; each
particle's numbers depend on that particle alone. With `OMP_NUM_THREADS=8`, the 200,000-particle LCLS-II run gives
identical output with this loop on one thread or eight.

**Proof that the tests cover it.** A build with six deliberate errors, one switched on per run by an environment
variable, each changed the feature set: the ordered mean added in reverse (11 cases), the sorted bin sums swapped
between planes (23), the time span shifted (19), the reference-time Kahan sum reversed (1), the wrong particle for
`FIRST` (6) and for `PMAX` (13). The errors were removed and the sources compared with the saved copies before the
final builds.

**Outputs.** Final build against the references of the section above:

| Run | Against the previous GPU reference | Against the CPU, before | Against the CPU, now |
| --- | --- | --- | --- |
| Feature set (368) | 367 identical, `rfca-thin-fiducial-modes-no-output` within rounding | 41 identical, 326 rounding, 1 flagged | The same |
| GPU test set (39), 47 Pelegant tests | All identical | | |
| 424 routine tests, thresholds of 64 | 402 identical, 22 changed | 237, 70, 117 | 237, 70, 117 |
| 660 elegantTestSet decks, default mode | 634 identical, 26 changed | 572, 26, 62 | 569, 29, 62 |

The 62 flagged decks are the same ones. `rfcw1`, `rfcw2` and `rfcw3` (cavities with `N_KICKS` above 0, on the device
for the first time by default) went from identical to within rounding. Where the largest difference from the CPU
changed, it shrank: `accelBeta` 1.4e-14 to 7.1e-15 m, `accelBeta2` 1.1e-13 to 5.4e-14, `full457MeV` 1.1e-13 to
5.7e-14. The verify build stops nowhere in the 56 cavity and wake decks; all 17,329 reference times, 9,104 `RFCW`,
3,103 `WAKE`, 55 `TRWAKE` and 7,871 momentum-matching checks are identical, and 625 of 88,871 `RFCA` checks differ
by one unit in the last place of `s`, as before.

**Speed.** Each deck alone on one P-core.

| Deck | This morning | After the default change | Now |
| --- | ---: | ---: | ---: |
| LCLS-II timing case, 200,000 particles | 60.6 s | 21.2 s | 14.0 s |
| The same without `output`, `final` and `WATCH` files | | 8.9 s | 8.0 s |
| `bunchComp10` (fixed bins, slice analysis) | 4.7 s | 3.9 s (20.5 s unguarded) | 2.2 s |
| `LCLS5` (`RFCW` wakes at every kick) | 6.5 s | 6.9 s (14.7 s unguarded) | 4.4 s |
| `firstIsFid2` (`RFCW` with `TRWAKE`) | 5.2 s | 5.7 s | 3.8 s |
| `multibunchPartialShortRangeWakes` | 0.67 s | 0.83 s (2.3 s unguarded) | 0.55 s |
| The 40 affected decks together | 217.7 s | 147.9 s | 118.5 s |

In the LCLS-II case an `RFCW` went from 12.3 ms to 4.5 ms and the time in sums from 2.7 s to 0.45 s a step; the CPU
takes 228 s for the run. With eight host threads it takes 12.5 s. `rfcw1` to `rfcw3` take 0.21 s against 0.15 s: they
now start the device for a 0.1 s run.

**What it leaves.** In the LCLS-II case the per-element `centroid` and `sigma` sums are now the largest sum cost
(0.2 ms each, 2,300 a step), and the CPU's `sin` and `log` on one host thread the largest part of an `RFCW`. Still on
one 64-thread block: the LSC statistics and `RFCW` energy-gain sums, the unordered mean for a fixed `N_BINS`, and the
generic beam, centered and min/max sums. The device slice sums (off by default) still use one thread per sum.

New references in `results/2026-11-10_narrow-sums`: `final-ets`, `final-routine`, `final-feature`, `final-suite`,
`final-pel`, `final-verify`. The feature set's CPU reference is still `2026-11-10_cavity-defaults/feature-cpu-ref`.

## Fewer square roots and divisions in the `LGBEND` kernel (November 10, 2026)

**What changed.** `gpuLgbendBody()` takes a `fewerRoots` argument, 1 for `LGBEND` and 0 for `CCBEND`. With it, each
integration substep uses one square root and one reciprocal where the CPU's form uses two square roots and two
divisions: the slopes are `q/pz` with `pz^2 = (1+dp)^2 - qx^2 - qy^2`, so `sqrt(1 + xp^2 + yp^2)` is `(1+dp)/pz` and
needs no second root. The radiation step uses one division instead of six and `sqrt(F2 sqrt(F2))` for
`pow(F2, 0.75)`. The kick strengths `KnL/nSlices` are made once per segment for both elements (the same value as
before). No CPU source changed.

**Why only `LGBEND`.** The verify build shows how each kernel stood against the CPU before this change, on the APS-U
ring with 1,000 particles: `LGBEND` already differed in all 160 checks (largest 2.8e-14, 1.1e-13 with radiation),
`CSBEND` in 199 of 200 (6.8e-15 to 4.8e-14), while `CCBEND` and the multipoles differ only in the last place of
small numbers (1e-19, 2e-20; over the feature set 452 of 263,215 multipole checks differ at all). `CCBEND` therefore
keeps the CPU's form: with the new one it differed by a unit in the last place of `s`, for 1% of the ring's time.
The multipole kernels weren't touched.

**Agreement with the CPU after the change.** `LGBEND` on the ring: largest difference 2.3e-13 m, one unit in the last
place of `s` beyond 1,024 m; largest relative difference 5.2e-8 without radiation, as before, and 1.7e-6 with it
(5.2e-7 before). Over the feature set the verify build stops nowhere; its 91 `LGBEND` checks differ by at most
7.1e-15 and 3.6e-10 relative.

**Outputs.** Final build against the references of the section above: the 660 elegantTestSet decks in default mode,
the 424 routine tests and the 47 Pelegant tests are all identical (no elegantTestSet deck puts enough particles
through an `LGBEND` to reach the device). The feature set changes in 17 cases, all within rounding: the 16
`magnet-lgbend-*` cases and `isr-small-batches`; against the CPU it is 41 identical, 326 rounding and 1 flagged, as
before. The GPU test set changes in `lgbend-deterministic-heavy`, within rounding; against the CPU 6 identical and
33 rounding, as before.

**Speed.** APS-U ring (160 `LGBEND`, radiation in bends and `K` magnets), one P-core, time in tracking per pass:

| Particles | Before | Now |
| ---: | ---: | ---: |
| 1,000 | 0.66 s | 0.51 s |
| 16,000 | 1.81 s | 1.38 s |
| 64,000 | 6.89 s | 5.18 s |
| 64,000, no radiation | 4.71 s | 3.43 s |
| 256,000 | 25.6 s | 19.2 s |

One `LGBEND` for 64,000 particles: 24.5 ms to 13.9 ms, and 18.4 ms to 10.5 ms without radiation. Of the 5.2 s pass,
`LGBEND` is now 2.2 s, `CSBEND` 1.4 s, `KQUAD` 0.76 s and `KSEXT` 0.36 s.

**A `CSBEND` change that was tried and not kept.** The drift through a sector turns the momentum by the same angle
for every particle. With that angle's sine, cosine and versine made once per substep, the angle-sum formulas and one
square root replace `asin`, `sin` and three `cos` calls per particle and substep, and `tan(a + phi) - tan(phi)`
becomes `sin(a)/(cos(a + phi) cos(phi))`. A `CSBEND` went from 4.06 ms to 2.75 ms (7.2 ms to 5.9 ms with radiation).
Its difference from the CPU grew from 4.8e-14 to 2.9e-13 m, and the cause is the CPU's formula: in 20,000 random
substeps against 60-digit arithmetic, `x = rho0 (f - 1) + f x` is wrong by up to 6.7e-15 m (`rho0` times a unit in
the last place of `f`), and the new form by 4.1e-19 m; `qx` and the path are equally good in both. Adopting it would
make the device more accurate than the CPU and further from it, unless `csbend.c` took the same form. It is saved as
`results/2026-11-10_magnet-kernels/csbend-drift-prototype.patch` for that decision.

New references in `results/2026-11-10_magnet-kernels`: `final-feature`, `final-suite`, `final-verify-feature`. The
660, routine and Pelegant references stay in `results/2026-11-10_narrow-sums`.

## `IMPEDANCE`, `CWAKE` and `LSCDRIFT` on the device with output in CPU order (November 11, 2026)

A run with `output`, `final` or a coordinate `WATCH` asks for results in the CPU's order. Until now that sent
`IMPEDANCE`, `CWAKE` and `LSCDRIFT` to the CPU, `LSCDRIFT` with no override. `impedance1` (500,000 particles) took
36.7 s on gpu-elegant against 47.5 s on elegant, `cwake1` (5,000,000) 13.4 s against 14.1 s, and `lsc4` spent 95 of
its 103 s in `LSCDRIFT` on the CPU. They now run on the device with such output too, and give the CPU's numbers to
the last bit:

- **Histograms.** The count in each bin is exact in any order. The sums of x and y in each bin are added in particle
  order: `gpuCombinedWakeSortedHistogramKernel`, after the stable sort by bin that `TRWAKE` already used.
- **`CWAKE`.** The direct convolution kernel adds each output bin's terms in `convolveArrays()`'s order, so from exact
  histograms its voltages are exact. cuFFT isn't used in CPU order. Fixed bins (`N_BINS`) sit on the mean time, which
  comes from `gpuWakeOrderedTimeMean()`, as for `WAKE`.
- **`IMPEDANCE`.** The three histograms come to the host, where `gpuImpedanceHostVoltage()` runs
  `track_through_impedance()`'s own `realFFT`, impedance product and inverse, and the voltages go back for the kicks.
  The arrays have the impedance file's length (2,048 bins in `impedance1`), so this costs little beside the binning and
  the kicks.
- **Kicks.** The transverse kick is now `V / (1e6 * mass * sign) / pz`, in `applyTransverseWakeKicks()`'s order; it
  was one division by the product. `IMPEDANCE`'s longitudinal kick now follows `track_through_impedance()`'s own
  interpolation rule (none in the first bin, no kick when the upper point lies past the last bin), which differs from
  the wake routines' rule at those two bins. Both hold with and without CPU-order output, so `IMPEDANCE` and `CWAKE`
  results without such output moved by rounding.
- **`LSCDRIFT`.** The device path already made its histogram as a count and its voltage on the host with the CPU's FFT,
  and with CPU-order output it took the beam size from `rms_emittance()`. Only the rule kept it on the CPU. The beam
  size now comes from the x and y columns alone, with `rms_emittance()`'s sums written out, instead of from a copy of
  the whole beam at every step.

`ELEGANT_GPU_WAKE_CPU_ORDER=0` selects the unordered sums and the device's FFT for `IMPEDANCE` and `CWAKE`, as it does
for `WAKE` and `TRWAKE`. `ELEGANT_GPU_LSC_REFERENCE_VARIANCE=0` selects the device's beam size for `LSCDRIFT`. CSR is
unchanged: with CPU-order output `CSRCSBEND` and `CSRDRIFT` stay on the CPU unless `ELEGANT_GPU_ENABLE_CSR_DRIFT` is
set (see "Not changed" below).

### Momentum matching skips a beam it has already found matched

With `always_change_p0`, `do_match_energy()` runs after every element, and with CPU-order output the device version
copies the momenta to the host and adds them there in particle order: one thread, 0.55 ms for 200,000 particles. Most
elements leave the momenta alone. When a call finds the beam already matched (no change needed), a copy of the
momentum column stays on the device (`gpuCudaColumnSnapshotTake()`), with the particle count and central momentum.
The next call compares the column with that copy bit for bit on the device, and if nothing differs and the count and
central momentum are the same, it returns: the same numbers in the same order would give the same sum and the same
"no change". A call that changes the beam drops the copy, so the call after it adds the beam up again.
`ELEGANT_GPU_MATCH_ENERGY_SNAPSHOT=0` turns this off. In the first (fiducial) step of the 200,000-particle LCLS-II
deck, 2,591 of 3,180 sums are skipped.

### Host threads that sleep

The numbers the device takes from this host's libm (an RF cavity's `sin` and `log` with CPU-order output, and
`SREFFECTS`'s Gaussian numbers) were made on OpenMP threads, one unless the run asked for more. OpenMP threads spin
after each job: with 8, the LCLS-II deck used 31 s of CPU time for a 12.6 s run. They are now made on a small pool of
POSIX threads that sleep between jobs (`gpuHostParallelFor()`).

- The count is what it was: one, unless the run asks for more with `-ompThreads` or `OMP_NUM_THREADS`
  (`ELEGANT_GPU_ISR_REFERENCE_THREADS`, which the GPU test set uses for its `SREFFECTS` case, still sets it too).
  gpu-elegant starts no thread of its own accord. A first version took up to 8 cores by default and had a variable of
  its own, `ELEGANT_GPU_HOST_THREADS`; both were removed the same day at the user's direction. Each number depends on
  its own index alone, so any count gives the same numbers.
- Waking a sleeping thread costs 0.5 to 1 ms on this host, so a job goes to the pool only when one thread would need
  about 2 ms for it: from 100,000 particles for the cavity numbers and 50,000 numbers for a Gaussian fill. With lower
  limits, 8 threads made `modulate4` (10,000 particles) take 31 s instead of 20.4 s.

### Checks

- **Verify build.** `impedance1` (five channels, five bunches, 20 passes), `cwake1` (seven channels, 5,000,000
  particles) and three `LSCDRIFT`s of `lsc4` with its `WATCH` files (4,000,000 particles): every histogram, voltage
  and coordinate comparison shows zero difference. The 20,000-particle LCLS-II deck passes all 16,659 element checks,
  the 2,104 of `do_match_energy` with zero difference. The feature set and the GPU test set complete with no stops.
- **Feature set.** Nine new cases, all identical to the CPU with their elements on the device:
  `impedance-output-cpu-order`, `impedance-output-bunches`, `cwake-output-cpu-order`, `cwake-output-autoscale`,
  `lsc-output-cpu-order`, `lsc-output-kick-mode`, `match-energy-skip-output`, `sreffects-host-threads` and
  `rfca-host-threads`. They write `output` only: `centroid`, `sigma` and `final` files hold sums the device makes in
  its own order, which would hide a lost last bit behind expected rounding. The suite has 377 cases: against the CPU
  50 identical, 326 within rounding and 1 flagged (`scmult-zero-length`, as before). Of the 368 older cases 364 are
  identical to the previous GPU reference; `impedance-ramp-passes` and `cwake-ramp-change-p0` moved by rounding (the
  kick order), and `impedance-bunched-skip` and `cwake-bunched-skip` in their centroid and sigma sums only, which the
  device now makes because the beam stays there.
- **GPU test set.** 34 of 39 identical to the previous GPU reference. The five `IMPEDANCE` and `CWAKE` cases moved by
  rounding, `impedance-heavy` by one unit in the last place of a centroid. Against the CPU still 6 identical and 33
  within rounding.
- **elegantTestSet, default mode.** 659 of 660 identical to the GPU reference. `lsc4`'s centroid and sigma files moved
  by rounding: its beam now stays on the device, where those sums are made, and its `WATCH` and `HISTOGRAM` files are
  unchanged. Against the CPU 569 identical, 29 within rounding and 62 flagged, as before. `impedance1`, `cwake1`,
  `lsc1`, `lsc2`, `csrSegment2` and `csrSegment3` now run their collective elements on the device and stay identical
  to elegant.
- **Forced routine list.** All 424 identical to the reference.
- **gpu-Pelegant.** All 47 identical to the reference; under MPI these paths stay off unless asked for.
- **CSR list.** 12 of 33 identical to the November 8 reference, among them every deck with `LSCDRIFT`. The other 21
  are RF linac decks (`LCLS0` to `LCLS7`, `firstIsFid1` to `4`, five `bunchComp` decks, `CLIC1`, `CLIC2`,
  `forwardThenBack1`, `scraper1`): the November 10 cavity changes moved the same decks in the routine list, and this
  list wasn't rerun that day. With the momentum skip off, the 33 are identical to the run with it on. `final-csr` is
  the new reference.
- **Thread rule.** On the test set's `SREFFECTS` deck with nothing set, gpu-elegant takes 4.55 s with 4.30 s of CPU
  time: one thread. With `OMP_NUM_THREADS=8` or `-ompThreads=8` it takes 1.3 s. `sreffects-host-threads` and
  `rfca-host-threads` get their four threads from `ELEGANT_GPU_ISR_REFERENCE_THREADS=4` in `suite.json`, since the
  harness takes only `ELEGANT_GPU_*` settings per test.
- **Switches.** With `ELEGANT_GPU_WAKE_CPU_ORDER=0` the four `IMPEDANCE` and `CWAKE` output cases stop being
  identical; with `ELEGANT_GPU_LSC_REFERENCE_VARIANCE=0` the two LSC ones do.
- **Mistake build.** Seven errors behind a switch, in a build that was replaced afterwards (the source was restored
  and compared with saved copies; the rebuilt binary repeats the test set and eight affected decks exactly). Each
  bin's particles added in reverse order, and the old kick order, each failed the four `IMPEDANCE` and `CWAKE`
  cases. The device's FFT in CPU order failed the two `IMPEDANCE` cases. The y column read from x failed the two LSC
  cases. The unordered mean time failed `cwake-output-cpu-order`, but only once a 37.5 m drift stood before its
  `CWAKE`: with times near zero the mean's last bits never reach the bins. A pool range one element short failed
  `sreffects-host-threads` and `rfca-host-threads`, which run with four threads; the test set's `SREFFECTS` case
  can't see it, since its excitation is off. A momentum skip that doesn't compare the beam failed `match-energy-skip-output`, and `LCLSII-1`
  stopped.

### Timing

Alone on one P-core with `OMP_NUM_THREADS=1`:

| Deck | elegant | gpu-elegant before | gpu-elegant now |
| --- | ---: | ---: | ---: |
| `impedance1` | 47.5 s | 36.7 s | 8.9 s |
| `cwake1` | 14.1 s | 13.4 s | 2.5 s |
| `lsc4` | 171 s | 93.1 s | 23.6 s |
| LCLS-II, 200,000 particles | 227 s | 14.1 s | 12.6 s |
| `LCLSII-1` | 24.3 s | 3.59 s | 3.17 s |

The LCLS-II run's CPU time went from 13.4 s to 11.5 s; its wall time varies by a second from run to run. With
`OMP_NUM_THREADS=8` it takes 11.5 s. `lsc1` and `lsc2` (1.0 s) take 0.05 s longer, for the CUDA context they now create.
`impedance1` in CPU order is slower than with the device's FFT (8.9 s against 5.3 s): a sort and a trip to the host
for every bunch and pass. The GPU test set takes 40.6 s against 40.1 s, inside its run-to-run spread; no deck was
found slower in CPU time.

See `results/2026-11-11_cpu-order` in GPU-Testing: `final-ets`, `final-routine`, `final-feature`, `feature-cpu-ref`,
`final-suite`, `final-pel`, `final-csr`, `final-verify-feature`, `final-verify-suite`, `mutation/` and
`binaries-final.sha256` (gpu-elegant `e0a5ad8f`). The 660, routine and CSR runs used an earlier build of the day
(`a5546aa2`), whose source differs from the final one only in the thread-count rule, and ran with one thread; the
final build repeats the test set, eight affected decks and the 47 Pelegant tests exactly (`after-thread-rule/`).

### Not changed: CSR

`CSRCSBEND` and `CSRDRIFT` still go to the CPU when output is in CPU order. With them forced onto the device,
`bunchComp1` takes 4.2 s instead of 8.2 s (elegant: 12.0 s), but its results move far more than rounding: `Sdelta` by
9e-6 relative, `ecnx` by 9e-7, `Cx` by 2e-10 m, and 528,755 values leave the 1e-12 screening envelope where 519 do
now. The histograms that drive the CSR wake are counts, and a particle that lands in a neighboring bin because the
bend kernel differs from the CPU in the last bits changes them by whole particles. The CSR bend's kernel would have to
match the CPU bit for bit first, and it uses `sin`, `cos` and `asin`, which the device's libm doesn't reproduce. That
is a decision for the user, not made here.

## Beams under 10,000 particles go on the device when the lattice pays (November 11, 2026)

`ELEGANT_GPU_MIN_PARTICLES` (10,000) kept a smaller beam on the CPU unless an `LGBEND` or `CCBEND`, which have lower
thresholds of their own, took it. A beam that losses took under 10,000 stayed on the device only until some element
sent it to the host; it never came back. Yet with 1,000 particles a `CSBEND` took 3.2 ms on the CPU and 0.39 ms on the
device, and a 32-kick `KQUAD` 1.6 ms and 0.22 ms (`rfRef1`).

Lowering the thresholds to 1,000 for every run was tried first. It changes 117 of the 660 elegantTestSet decks. Most
of the gain was in six ring decks, and many small decks paid 0.15 s for a CUDA context they had no use for:
`fmult1`, which runs elegant eight times, went from 1.05 s to 2.02 s. `trackingInterrupt1`, whose 10,000 particles
shrink while a `SCRIPT` and a `SCATTER` send the beam to the host every pass, went from 6.3 s to 24.5 s.

So each tracking call now makes its own choice, in `gpuChooseSmallBeamThresholds()` (`gpu_stub.c`), called from
`do_tracking()` at the first pass:

1. It walks the elements tracking will use (with `concat_order`, the concatenated ones) as tracking would with the
   general thresholds at the 1,000 resident floor: which elements the device takes, where the beam is copied to the
   device and back, and which short runs the short-run rule keeps on the CPU. A ring's second pass is the one
   counted, since the beam starts it where the first left it.
2. Each element the device would take has a cost on each side: so many ns a particle on the CPU, so many us on the
   device whatever the beam size. A copy of the beam costs 20 us and 0.01 us a particle, a tracking call 250 us.
3. From the sums comes the particle count at which the CPU's total is 1.4 times the device's. If the beam has that
   many, the general thresholds (matrix, helper, exact drift, reduction, aperture, magnet, wake, LSC, CSR, SCMULT;
   not one the run set by name) drop to that count, or to 1,000 if it is lower, for this call.
4. A run with no CUDA context yet waits until the calls so far would have saved 0.3 s. One with a context doesn't.
5. A beam at or above 10,000 is walked as if it had 9,999, so that it returns to the device after losses.

| Element | CPU, ns a particle | Device, us |
| --- | ---: | ---: |
| Matrix, drift, corrector, monitor, collimator and any other the device takes | 15 | 10 |
| The same with `centroid` or `sigma` output | 60 | 40 |
| `CSBEND`, added for each slice | 500 | 100 |
| `KQUAD`, `KSEXT`, `KOCT`, `KQUSE`, added for each kick | 25 | 5, and 20 once |
| `DQCOR`, added for each slice | 300 | 50 |
| `CCBEND`, added | 300 | 100 |
| `LGBEND`, added | 10,000 | 850 |
| `BGGEXP`, added | 1,500,000 | 200,000 |
| `RFCA`, `RFCW`, added | 10 | 55 |
| `SREFFECTS`, added | 95 | 55 |
| `TFBPICKUP`, `TFBDRIVER` | 7 | 21 |

The CPU figures are the low end of what `show_element_timing` gave on `rfRef1`, `ibScatter3`, `scRing1`,
`multipoles3`, `centroid2`, `feedback2`, `srEffects2`, `rcol1`, `bggexp1`, `DQCOR2` and the APS-U ring, so the saving
is not overstated. They are this host's (one P-core, RTX PRO 4000 Blackwell). The estimate reads the lattice, the
particle count and the passes, never a clock, so a run takes the same path every time.

`ELEGANT_GPU_ENABLE_SMALL_BEAMS=0` turns the choice off. A run that sets `ELEGANT_GPU_MIN_PARTICLES` has chosen for
itself and is left alone, as is required mode. gpu-Pelegant keeps the general thresholds: the costs are the serial
build's. `ELEGANT_GPU_SMALL_BEAM_REPORT=1` prints each call's sums and choice, and 2 adds the element types, without
the CUDA context that `ELEGANT_GPU_VERBOSE` starts (which changes both the timing and the choice).

### Timing

The 117 decks alone on one P-core with `OMP_NUM_THREADS=1`, CPU time (user plus system):

| | Rule off | Rule on |
| --- | ---: | ---: |
| All 117 | 845.4 s | 470.6 s |
| `ibScatter2` | 89.2 s | 12.3 s |
| `ibScatter3` | 96.4 s | 19.2 s |
| `rfRef1` | 57.0 s | 8.0 s |
| `scRing1`, `scRing2`, `scRing3` | 46.7 s | 8.5 s |
| `losses3` (10,000 particles, shrinking) | 13.3 s | 7.9 s |
| `srEffects2` | 10.2 s | 5.3 s |
| `centroid2` | 4.9 s | 2.4 s |
| `multipoles3` | 3.6 s | 1.0 s |
| `bggexp1` | 2.85 s | 0.35 s |
| `ringRF1` | 2.30 s | 1.02 s |
| LCLS-II, 5,000 particles (not in the test set) | 7.9 s | 1.9 s |
| LCLS-II, 2,000 particles | 4.7 s | 1.7 s |

Wall time went from 900 s to 533 s. 47 decks lower the thresholds in some call. Four of them use more CPU time than
with the rule off: `dynAp4`, `dynAp5` and `losses2` by about 0.12 s, the CUDA context, because nearly all of their
beam is lost at once and the estimate can't know that; `NLSJitter1` by 0.06 s in 13.8 s. The 70 decks that never
lower them are within 0.09 s either way, which is their spread.

Wall time on one core is a poor measure here. `ringRF1` took 2.6 to 5.0 s of wall time at a steady 2.3 s of CPU
time, from its file output, and several decks first counted as slower with the thresholds at 1,000 (`ria2`,
`ringRF1`) were not. The times above were taken with `--launcher "/usr/bin/time -a -o gpu-time.txt -f %U,%S,%e
taskset -c 2"`, which leaves each elegant run's times in the deck's output directory.

Not gained: matrix lattices near their break-even count stay on the CPU (`feedback2` at 1,000 particles; `ringRF2`
to `ringRF4`, where a `MODRF` the device doesn't take costs two copies a pass).

### What it changes in results

A beam the device now takes differs from elegant's by rounding where it was identical before. Of the 660 default
decks, 538 are identical to elegant, 35 inside the screening envelope and 87 flagged, where there were 569, 29 and
62. 32 decks changed class:

- 24 from identical to flagged: `bunchComp3`, `bunchComp5`, `dynAp4`, `dynAp5`, `ecol1`, `ecol2`, `fseCSBEND1`,
  `ibScatter2`, `ibScatter3`, `ignoreElements1`, `losses2`, `maxamp1`, `maxamp2`, `maxamp3`, `multipoles3`,
  `radDamping6`, `ramprf1`, `rcol1`, `rfRef1`, `scRing1`, `scRing2`, `scRing3`, `srEffects4`, `uKickMap4`.
- 1 from inside the envelope to flagged: `rfdf4` (one value, 1.1e-15 in `xpMinimum`).
- 7 from identical to inside the envelope: `bggexp1`, `bggexp3`, `centroid2`, `ringRF1`, `srEffects2`,
  `srEffects3`, `srEffects5`.

None is more than rounding. Beam sizes and emittances agree to 4e-10 relative or better (`ibScatter3` `Sdelta`
3.6e-10, `ibScatter2` `Ss` 2.8e-10, `rfRef1` `St` 2.1e-10, `uKickMap4` `betaxBeam` 3.7e-10, the `scRing` decks 2e-11),
except where the quantity is near zero. `ecol1`, `ecol2`, `maxamp1` to `maxamp3`, `rcol1` and `uKickMap4` have no
bunch length, so `Ss` is rounding on both sides: it differs by 1.8e-12 m, one unit in the last place of the path
length after their passes. `srEffects4`'s bunch is 1.4 um long and its `Ss` differs by 2.3e-14 m. Particle
coordinates differ by 3e-12 m or less, except in `fseCSBEND1`: 6.9e-11 m in `x`, from `CSBEND`s with a 0.1 mrad
angle and 1,000 kicks. 22 of the 32 are in the forced routine list, where they already ran on the device and
their differences had been reviewed. The other ten are `ibScatter2`, `ibScatter3`, `radDamping6`, `ramprf1`,
`rfRef1`, `scRing1` to `scRing3`, `srEffects3` and `srEffects5`.

### Checks

- **660 default against the GPU reference.** 625 identical; the 35 that differ are among the 47 decks that lower
  the thresholds. A second run of the 117 affected decks, one at a time with the report on, gave the same files.
- **Forced routine list (424), CSR list (33), GPU test set (39), 47 Pelegant tests.** Identical to the references of
  the section before. These set `ELEGANT_GPU_MIN_PARTICLES`, run in required mode or use MPI, so the choice isn't
  made.
- **Feature set.** 385 cases: 384 within the envelope (55 identical) and `scmult-zero-length` flagged as before.
  The 377 older cases differ from the CPU exactly as they did, line for line of the report. The eight
  `small-beam-*` cases are described in the feature set's README; three go on the device and five stay.
- **Mistake build.** Nine errors behind a switch, in a build that was replaced afterwards (source restored and
  compared with saved copies). Thresholds never lowered: `small-beam-kick-ring` and `small-beam-ring-no-copies` have
  no GPU activity. The opt-out ignored: `small-beam-opt-out` stops being identical. The context's worth ignored:
  `small-beam-context-not-worth-it`. Copies free: `small-beam-copies-cost-more` and
  `small-beam-concatenated-ring`. The lattice walked in place of the concatenated elements:
  `small-beam-concatenated-ring`. The thresholds lowered for the walk left lowered: four of the five cases that
  should stay. A beam at or above 10,000 left alone: `small-beam-returns-after-losses` no longer matches the GPU
  reference (its class is the same, since the first pass is on the device either way). The short-run rule ignored
  in the walk: `small-beam-short-runs-stay`. A ring's first pass counted: `small-beam-ring-no-copies`.
- **Verify build.** Feature set (385) and test set (39) complete with no stops. Of the 45 decks that lowered the
  thresholds in an earlier build, 42 complete; `csbendAp1`, `csbendAp2` and `fseCSBEND1` stop in a `CSBEND` with a
  0.1 mrad angle and 1,000 kicks (2e-10 m in `x` against the 1e-12 tolerance). They stop the same way with
  `ELEGANT_GPU_ENABLE_SMALL_BEAMS=0`: `csbendAp1` has 10,000 particles and was on the device before, and
  `fseCSBEND1` stops with the thresholds forced to 64. That rounding is the bend kernel's, not this change's.

See `results/2026-11-11_threshold` in GPU-Testing: `final-ets`, `final-routine`, `final-csr`, `final-suite`,
`final-pel`, `final-feature`, `feature-cpu-ref`, `final-verify-feature`, `final-verify-suite`,
`final-verify-lowered`, `time-off4`, `time-final`, `now-differ.txt`, `now-differ-sizes.txt`, `mutation/` and
`binaries-final.sha256` (gpu-elegant `03730cc9`).

## Frequency maps and tune footprints tracked by the fused program (November 11, 2026)

A batched `&frequency_map` tracked on the CPU by design: `gpu_batched_frequency_map_cpu_tracking_required()` returns
1, because the device's turn-by-turn histories had never been shown to be the CPU's. The fused tune program, which
tracks a whole grid through the beamline with one kernel launch a turn, ran only in verify builds, and only to
predict losses. The batched path also needed `full_grid_output=1`, which is not the default, and `&tune_footprint`
batched nothing but `POLYNOMIALSERIES` maps. So `fmap3`, a 21 by 21 map through an APS lattice, took 34.6 s in
gpu-elegant against 33.2 s in elegant.

### What changed

- **The program returns histories.** `gpu_batched_tune_program_track()` (`gpu_stub.c`) gives each particle's x, x',
  y and y' at the start of every turn, its final coordinates and whether it survived. It replaces
  `gpu_batched_tune_loss_prepass()`.
- **The CPU checks a sample before the histories are used.** `computeTunesFromTrackingBatch()` (`twiss.cc`) has the
  CPU track eight of the survivors and eight of the lost, evenly spaced through the grid, and the survivors that
  started with the largest x, y and momentum offsets either way. Every turn of their four histories, their six final
  coordinates and their survival must be the device's, bit for bit. Then the device's histories serve for the rest
  of the grid. Otherwise the CPU tracks the rest as well, as it did before, so the output is the CPU's either way.
- **The program runs only where its arithmetic is the CPU's.** Each element has to be in a configuration the verify
  build has shown to be exact (below). A beamline with any other, a bend among them, is tracked on the CPU as
  before.
- **The verify build checks every particle.** The CPU tracks the whole grid, as it always has there, and the run
  stops if a program that claims the CPU's bits differs from the CPU in any particle's history, final coordinate or
  survival, or if any program kept a particle the CPU lost.
- **More beamlines fit the program.** It now takes elements the CPU tracks by matrix alone (`QUAD`, `SBEN`, `SEXT`
  and the like), `MALIGN` with its `ON_PASS` (a new operation, `GPU_TUNE_OP_MALIGN`, in the kernel), and `WATCH`,
  `HISTOGRAM` and `CHARGE`, which do nothing in this tracking.
- **The default grid output is batched.** Without `full_grid_output` a point has a row only if both of its tunes
  were found, as in the point-by-point loop.
- **`&tune_footprint` batches any beamline the program takes**, with the device's element-by-element tracking kept
  out of the batch, as in a frequency map.
- **The program picks its own device.** `gpuBaseInit()` leaves no active device after a tracking call with the
  device suppressed, which is how the checks run, so a second interval used to find none. The batched-tune settings
  are also read before the first tracking call, which a tune footprint's first batch can precede.
- **The tune analysis uses one thread unless the run asks for more.** After tracking, NAFF ran on up to 16 threads
  by default. It now follows the rule set for the host thread pool: `OMP_NUM_THREADS`, `-ompThreads`, or an explicit
  `ELEGANT_GPU_BATCHED_TUNE_THREADS`. The results don't depend on the thread count.

`ELEGANT_GPU_ENABLE_TUNE_PROGRAM=0` turns the program off. gpu-Pelegant is unchanged.

### Which operations are the CPU's, bit for bit

Each was tested alone in a ring of two `KQUAD`s, 81 particles for two intervals of 16 turns, under the verify build:

| Exact | Not exact |
| --- | --- |
| Every matrix element; `EDRIFT`; `HKICK`, `VKICK`; `EHKICK`, `EVKICK` with `TILT` and `CALIBRATION` | `CSBEND`: 3% of the particles with one bend a turn, 18% to 42% with eight, up to 5e-10 of the history's range after 32 turns. The rest have the CPU's bits |
| `KQUAD`, `KSEXT`, `KOCT` and `DQCOR` with integration order 2, 4 or 6, `DX`, `DY`, `TILT`, `PITCH`, `YAW`, `FSE`, steering, edge effects, `EXPAND_HAMILTONIAN` | `CCBEND`: 13% of the particles with four a turn, up to 1e-12. `LGBEND`: not tested alone |
| Thin `MULT` | A kick magnet with `MALIGN_METHOD` 1 or 2 and no misalignment: 5e-15. The CPU transforms at the entrance whatever the offsets, and at the exit only if one is set |
| `RCOL`, `ECOL` with offsets; `SCRAPER`; `SPEEDBUMP`; `TAPERAPC`; `TAPERAPR` without `TILT`; `MAXAMP` | A kick magnet with `DZ`: 8e-16. A `MULT` with length: 3e-14. A `TAPERAPR` with `TILT`: one particle in 81, 2e-16 |
| `MALIGN` with any offsets, on one pass or every pass; `RFCA` with no voltage | |

`gpuTuneProgramExactElement()` holds the second column, and a program with any of it is not run by default. A kick
magnet with a `MALIGN_METHOD` and a misalignment was exact where tested (`DX`), but isn't claimed. These
differences are in the kernels the ordinary element-by-element tracking uses too.

The table is as it stood when the program first ran. The same day the `MALIGN_METHOD`, `DZ` and `CCBEND` rows were
corrected in the kernels: see the next section, which also says why `CSBEND` and `LGBEND` differ.

### Bends: a decision for the user

`ELEGANT_GPU_TUNE_HISTORY_TOLERANCE` lets a run use the device's histories when they aren't the CPU's: a positive
value is the largest difference allowed in the sampled histories, as a fraction of each history's largest value. It
is unset by default, so nothing here changes a result. What it would buy and cost, measured with the tolerance at 1:

| Deck | CPU tracking | Device histories | Tunes against elegant | Other |
| --- | ---: | ---: | --- | --- |
| `ccbend6`: the APS-U style ring (200 `CSBEND`, 160 `CCBEND`, 160 `LGBEND`, 10,600 operations), 51 by 51, two intervals of 64 turns | 622 s (elegant: 865 s) | 55 s | median 2.7e-10, 99% under 1.8e-9, largest 1.8e-8 | The same 1,345 points have tunes. Diffusion rate: median difference 1e-6, largest 8e-4 |
| `fmap1`: 8 `CSBEND` of 45 degrees, 21 by 21, amplitudes to 0.2 m, 128 turns | 4.2 s | 2.7 s | median 1.2e-10, largest 4.1e-10 | The same 203 points have tunes |

The sampled histories differed by up to 9e-6 of their range in `ccbend6` (64 turns, with particles next to the
dynamic aperture in the sample) and 5e-8 in `fmap1`.

### Timing

Alone on one P-core with `OMP_NUM_THREADS=1`, wall time:

| Deck | elegant | gpu-elegant before | gpu-elegant now |
| --- | ---: | ---: | ---: |
| `fmap3`: an APS lattice of kick quadrupoles, kick sextupoles and matrix bends, 3,718 operations a turn; 21 by 21, 256 turns | 33.2 s | 34.6 s | 13.5 s |
| The same, 51 by 51 | 190.9 s | | 18.4 s |
| The same, 101 by 101 | about 750 s (not run) | | 39.1 s; 19.8 s with `OMP_NUM_THREADS=8` |
| Test set's FODO map, 33 by 33, two intervals of 32 turns | 2.10 s | 1.99 s | 1.55 s |
| `fmap1`, `fmap2`, `fmap5`: rings with `CSBEND`, tracked on the CPU | 5.7, 1.6, 20.0 s | 4.2, 1.7, 20.3 s | 4.1, 1.6, 19.8 s |

"Before" is the new binary with batched tune tracking off, which is the point-by-point path these decks took (for
the FODO map, with the program off). Every file in the table is byte-identical to elegant's, and the 101 by 101
file is the same with one thread and eight.

The program is bound by how fast one device thread runs, not by how many there are: `fmap3`'s 255 launches take
11.3 s whether the grid has 441 points or 2,601, and 15.3 s for 10,201. A device thread runs this code 50 to 100
times slower than a CPU core, so a small grid gains little and the gain grows with the grid. What is left is the
CPU's own tracking of the 18 sampled particles (1.6 s for `fmap3`) and the tune analysis: about 4 ms for each
surviving point at 256 turns, 22 s of the 39.1 s at 101 by 101, which is why threads help there.

### Checks

- **660 default decks.** All 660 identical to the GPU reference of the section before; against elegant, 538
  identical, 35 inside the envelope and 87 flagged, as before. `fmap3` is now tracked by the program;
  `fmap1` and `fmap2` take the batched CPU path where they went point by point, as do `fmap5` and `ccbend6`,
  which are not among the 660. All five files are byte-identical to elegant's.
- **Forced routine list (424), CSR list (33), GPU test set (39), 47 Pelegant tests.** Identical to the references
  of the section before.
- **Feature set.** 393 cases: 392 within the envelope (63 identical) and `scmult-zero-length` flagged as before.
  The 385 older cases differ from the CPU exactly as they did, line for line of the report. The eight new ones,
  `search-frequency-map-program*` and `search-tune-footprint-program`, are identical to the CPU and are described in
  the feature set's README.
- **Verify build.** Feature set (393), test set (39) and eight of the 660 (`fmap1`, `fmap2`, `fmap3`, `fmap5`,
  `brat4`, `polySeries1`, `polySeries2`, `momentumFootprint1`) complete with no stops. In `fmap3` all 441 particles
  have the CPU's histories for all 256 turns. In the bend decks the program runs as a diagnostic: `fmap1` and
  `fmap2` differ by up to 8e-8 of a history's range, `fmap5` by 9e-6, with no particle lost on one side only.
- **Mistake build.** Seven errors behind a switch, in a build that was replaced afterwards (source restored and
  compared with saved copies). A `MALIGN` packed for the pass after its own, and every particle reported alive:
  the sample check sends the grid back to the CPU, so the output doesn't move, but fewer logs say "its histories are
  used" and the verify build stops in five and six of the eight cases. The check passing whatever it finds:
  `search-frequency-map-program-fallback` stops matching the CPU. Final coordinates not taken, and histories copied
  a turn late: five cases stop matching. A beamline with bends claimed exact: `search-frequency-map-program-bend-stays`
  stops matching (its sample happens to be clean, which is why a claim has to be right) and the verify build stops
  in four cases. A row written for a point whose second tune failed: `search-frequency-map-program`, once a `MALIGN`
  on pass 40 had been added to put points with a first tune onto the collimator in the second interval.

### Not done

- Bends, above. Besides a tolerance, the other way is to make the bend kernels exact, which for `CSBEND` needs the
  drift formula change that is still undecided.
- The kernel differences in the table's second column, which the element-by-element tracking shares. The
  `MALIGN_METHOD` one matters most: APS-U style lattices set it on every magnet.
- The tune analysis is still on the host.
- `LGBEND` alone was not measured; `ccbend6` has all three bends.

See `results/2026-11-11_fmap` in GPU-Testing: `final-ets`, `final-routine`, `final-csr`, `final-suite`, `final-pel`,
`final-feature`, `feature-cpu-ref`, `final-verify-feature`, `final-verify-suite`, `final-verify-fmap`, `time-cpu`,
`time-before`, `time-now`, `time-now-omp8`, `mutation/`, `source-before/`, `source-final/` and
`binaries-final.sha256` (gpu-elegant `aa2911bd`). The bend measurements are in `tmp/2026-11-11_fmap`.

## Kick magnets and `CCBEND` to the CPU's last bit, and why the other bends differ (November 11, 2026)

The section before lists kernels that differ from the CPU in the last digits, and those differences kept every
lattice with a bend out of the fused tune program. Each was looked into. Seven were differences in the order of
operations, or a step one side took and the other didn't, and are corrected, so kick magnets and `CCBEND` now give
the CPU's bits. `CSBEND` and `LGBEND` differ for reasons the device can't remove on its own, set out below. Two
claims the program made were also wrong: for `LEFFECTIVE`, which is now true, and for exact correctors with a
kick, which is withdrawn.

### Kick magnets (`KQUAD`, `KSEXT`, `KOCT`, `DQCOR`)

- **A linear `MALIGN_METHOD` with no misalignment.** With `MALIGN_METHOD` 1 or 2 the CPU applies the entrance
  transformation whatever the offsets, and the exit one only if an offset, pitch, yaw or tilt is set. The device
  applied both or neither. It differed by 5e-15, and APS-U style lattices set the method on every magnet.
- **`DZ`.** The CPU adds `DZ`'s path to the particle's coordinate before the magnet and takes it off after. The
  device added it to the path through the body, which rounds differently: 8e-16.
- **`LEFFECTIVE`.** A `KQUAD` with `LEFFECTIVE` has a drift at each end. The CPU adds each end drift's path, and
  the edge fringe's, to the particle's coordinate as it goes, and the body's path in between. The device gathered
  all of it in one sum and added once, which came out a unit off in `s` in 40% of the particles. The program claimed
  the CPU's bits for it, wrongly: the sample check sent such a grid back to the CPU, and a verify build stopped.
- **A tilt of a full turn.** `rotateBeamCoordinatesForMisalignment()` doesn't rotate for a tilt within 1e-12 of 2 pi.
  The device's helper rotated by the sine of 2 pi in double precision, -2.4e-16. This is what a `CCBEND` with a
  negative angle and `TILT=pi` has, since the CPU adds pi for the sign. Every tilted element but `CSBEND` goes
  through that routine on the CPU, and now through `gpuRotationTiltSinCos()` on the device. `CSBEND` has its own
  code with no such case, so there the CPU does rotate by -2.4e-16, and the device still does too
  (`gpuCsbendTiltSinCos()`). The first version of this change gave `CSBEND` the full-turn case as well, and
  `csbendRotation1`, which had been identical to the CPU, came out 1e-29 off in `y`: that is how the difference
  between the two CPU routines was noticed.
- **Radiation in the program.** The program's multipole operation has no radiation, so a `KQUAD` or `KSEXT` with
  `SYNCH_RAD` was tracked without it. The sample check caught that and sent the grid to the CPU, so no file was
  wrong, but the device's pass was wasted. The program now isn't built for such a beamline.

37 configurations of `KQUAD`, `KSEXT`, `KOCT`, `DQCOR` and `MULT` were then run for 20,000 particles through one
magnet with zero tolerance in the verify build. 34 give the CPU's bits: integration order 2, 4 and 6, `LEFFECTIVE`
shorter and longer than `L` with edges and each `MALIGN_METHOD`, `EDGE1_EFFECTS` and `EDGE2_EFFECTS` 1 to 3 with
fringe integrals, `BORE` and `B`, steering with calibrations, `FSE`, `TILT` (also pi/2, pi and 2 pi), `PITCH`, `YAW`,
body, random, edge and steering multipole files, `EXPAND_HAMILTONIAN`, a zero length, and a thin `MULT`. The three
that don't:

- A `KQUAD` with `SYNCH_RAD`: 5 particles of 20,000 differ in `delta`, by a unit in the last place. Not looked into.
  The program doesn't take radiation.
- A `MULT` with length: every particle, 1e-16. As before, it keeps a beamline out of the program, as does a
  `TAPERAPR` with `TILT` (2e-16).
- `RADIAL=1` isn't on the device.

### Exact correctors: a claim withdrawn

The program claimed the CPU's bits for `EHKICK`, `EVKICK` and `EKICKER`, because 81 particles for 32 turns came out
the same. With a kick, the CPU takes `atan()`, `sin()`, `asin()`, `cos()` and `tan()` for each particle, from the
host's library, and the device takes its own. At a kick of 0.2 mrad 20,000 particles came out the same; at 20 mrad
124 values differed in the last place. So an exact corrector with a kick now keeps a beamline out of the program
unless a tolerance is set. With no kick the CPU makes it a drift, and that is still claimed.

### `CCBEND`

- **A division and a multiplication the CPU does after every slice.** `track_through_ccbend()` works out its
  radiation coefficients whatever the flags. With `ISR` off it negates the ISR coefficient and doesn't zero it, and
  `integrate_kick_KnL()` runs its radiation block whenever either coefficient isn't zero. So for an ordinary
  `CCBEND` with no radiation, each slice ends with `qx /= (1+dp); qy /= (1+dp);` and then `qx *= (1+dp);
  qy *= (1+dp);` with `dp` unchanged. That isn't a no-op in floating point: the chance that `q/(1+dp)*(1+dp)` isn't
  `q` is about `|dp|`, and when it isn't, the momentum moves by one unit in the last place. With a momentum spread of
  1e-3, 0.1% to 0.3% of the particles came out of one `CCBEND` a unit off in a slope. The device now does the same
  round trip (`momentumRoundTrip` in `GPU_CCBEND_DATA`), in the general kernel and in the order-2 kernel with no
  fringe, which also takes its exit slopes from the momenta as that block left them.
- **A magnet shorter than 1e-6 m with `ISR` or `SYNCH_RAD`.** The CPU zeroes both coefficients there. The device
  kept its classical radiation; it now drops it too.
- **The last product of the `FRINGEMODEL=1` fringe.** The CPU has `px2 = px2 * temp * temp`, which is
  `(px2 * temp) * temp`; the device had `px2 *= temp * temp`. `temp` is 1 unless the `K5` fringe integral is set and
  `EDGE_ORDER` is 2 or more. `LGBEND` shares this function.

How the first was found: a replay of the device's arithmetic in strict double precision (Python, with exact
fractions where an operation could be fused) gave the device's bits, so the device was doing what its source says.
Printing the CPU's values inside `switchRbendPlane()` and the integrator, in a build that was replaced afterwards,
showed where the CPU's left the replay. With the round trip added, the replay gave the CPU's file for all 20,000
particles.

62 configurations were then tried, each for 20,000 particles through one magnet with zero tolerance, or for 81
particles and 32 turns through the program, in the verify build. 59 give the CPU's bits:

| The CPU's bits | Not |
| --- | --- |
| Integration order 2, 4 and 6; 1 to 16 slices; either sign of the angle; `K1` to `K8`; `FSE`, `FSE_DIPOLE`, `FSE_QUADRUPOLE`, `XKICK`; `OPTIMIZE_FSE` and `OPTIMIZE_DX` on and off; `REFERENCE_CORRECTION`; `COMPENSATE_KN` | `ISR`, alone, with `ISR1PART=0` and with `SYNCH_RAD`: 15% of the particles differ in `delta`, by up to 1e-16. The CPU's `pow(F2, 0.75)` comes from the host's library and the device's from CUDA's |
| `FRINGEMODEL` -1, 0 with `EDGE_ORDER` 0 to 3, and 1 with `EDGE_ORDER` 1 and 2; `HGAP` and `FINT`; every fringe integral, alone and together | |
| `TILT`, `YAW`, `DX`, `DY`, `DZ`, `ETILT`, `EPITCH`, `EYAW` with each `MALIGN_METHOD` from 0 to 4; `DX_DY_SIGN`; a reversed magnet | |
| Body, edge and random multipole files with order limits and a factor; `SYNCH_RAD`; a `MAXAMP` before it; two different `CCBEND`s in a line | |

The program now claims the CPU's bits for a `CCBEND`, with one exception. The `FRINGEMODEL=1` fringe calls `exp()`
for each particle when the `I1` integral is set, or `K5` with `EDGE_ORDER` 2 or more. The device's `exp()` and the
host's gave the same bits for every particle tried, but nothing promises that, so a beamline with such a `CCBEND`
stays on the CPU unless a tolerance is set. (`gpuTuneProgramExactElement()` holds the rule.)

### `CSBEND`: the trigonometric functions

A `CSBEND`'s drift calls `asin()`, `sin()` and `cos()` for every particle at every substep. The device's come from
`gpuAccurateAsin()` and CUDA, the CPU's from the host's C library. Both are within about half a unit in the last
place of the true value, and they pick different neighbors for a few arguments in ten thousand. Each such result is
then multiplied by the bend radius. That is where the 3% of particles a turn in the table of the section before
comes from, and no change on the device alone removes it: the device would have to repeat one version of one C
library's functions step for step, and elegant's own results already depend on that version.

Two things would make `CSBEND` exact, and both change elegant's results in the last digits, so they are the user's
decision:

- The drift formula that needs no trigonometric function per particle (prototyped on the device on November 10:
  `csbend-drift-prototype.patch`), in elegant and gpu-elegant both. It is also faster on each side, and closer to the
  true value than the present formula. Whether anything else in `CSBEND` differs can't be seen until this is
  removed.
- One implementation of the three functions compiled into both.

### `LGBEND`: by design

The `LGBEND` kernel was rearranged on November 10 to take one square root and one reciprocal a substep where the
CPU takes two square roots and two divisions, for 1.75 times the speed. It differs from the CPU in every particle,
by a unit or two in the last place of `s`. The CPU's `LGBEND` also has the round trip above. Making it exact means
going back to the CPU's form and giving up that speed, in the program only or everywhere. Not done: an APS-U
style ring has `CSBEND`s too, so it would not yet put such a ring on the program.

### What it changes

- A frequency map or tune footprint through a lattice whose only bends are `CCBEND`s (and matrix bends) is now
  tracked by the program by default, with the CPU's output.
- Element-by-element tracking of a `CCBEND`, or of a kick magnet with `MALIGN_METHOD`, `DZ` or `LEFFECTIVE`, now
  gives the CPU's bits where it gave them to within rounding. Nothing in a result moves by more than rounding.
- A lattice with an exact corrector that has a kick, or a `KQUAD` with `LEFFECTIVE`, was taken by the program
  before; the first now stays on the CPU, and the second is tracked by the program with the CPU's bits.

### How the last three were found

After the `CCBEND` change, every feature case whose particle coordinates still differed from the CPU's was listed:
92 of 396. Most have an element known not to be exact (`CSBEND`, `LGBEND`, `CWIGGLER`, CSR, space charge, ISR).
The others were taken apart magnet by magnet at zero tolerance, which gave `LEFFECTIVE`, the full-turn tilt and
the exact correctors. Still on that list and not looked into: `FMULT` (37 particles of 2,000), `APCONTOUR` with
`TILT` (1e-18; the CPU rotates the coordinates there and back, which isn't exact), and the `KQUAD` radiation above.

### Checks

- **Feature set.** 396 cases: 395 within the envelope (66 identical) and `scmult-zero-length` flagged as before.
  Against the comparison of the section before, 49 cases moved and every one moved toward the CPU: fewer of its
  values differ, and what is left is in the sums of the `centroid` and `sigma` files or comes from another element.
  None moved away. `magnet-malign-method1` went from 6,384 differing values to 453, `magnet-ccbend-misalignment-method0`
  from 326 to 141. Three cases are new or changed, described in the feature set's README:
  `search-frequency-map-program-ccbend` (new), `-opcodes` (exact correctors with no kick, a `KQUAD` with
  `LEFFECTIVE`) and `-tolerance` (the elements that call a library function).
- **GPU test set (39).** 37 identical to the reference of the section before. `gputest1-resident-magnets-focused`
  went from 20,158 values that differ from the CPU to 7, and `ccbend-deterministic-heavy` from 636 to 521.
- **660 default decks.** 658 identical to the GPU reference of the section before. `fseCCBEND1` and `kquadEdgeBody1`
  moved, and both are now identical to the CPU (1,684 and 1,040 values differed). Against elegant: 540 identical, 33
  inside the envelope and 87 flagged (538, 35 and 87 before).
- **Forced routine list (424).** 418 identical to the reference of the section before. Six moved, all toward the
  CPU: `fseCCBEND1` and `kquadEdgeBody1` are now identical to it, `fmult2` has no value left that differs (2,606
  before), and `mult4`, `kickElemMisalign1` and `kickElemMisalign3` have fewer. Against the CPU: 239 identical, 68
  inside the envelope, 117 flagged (237, 70 and 117 before).
- **CSR list (33).** Identical to the reference of the section before.
- **Pelegant (47).** Identical to the reference of the section before.
- **Verify build.** Feature set (396) and test set (39) complete with no stops. 19 feature cases run the program,
  over 35 intervals. In 27 of them every particle has the CPU's history and final coordinates, among them the four
  cases with `CCBEND`s in the program (the new one, with 675 particles, and `search-frequency-map-ccbend-options`,
  `-fringe-model-0` and `-dqcor-files`, which were tracked on the CPU before and had up to 13% of their particles
  off). The other eight intervals are the `CSBEND` and `MULT` cases that aren't claimed.
- **Mistake build.** The round trip switched off by a variable, in a verify build that was replaced afterwards
  (source restored and compared with the saved copy). The verify build stops in four of the five `CCBEND` program
  cases: 40 of 675 particles differ in the new one, 7 to 15 of 75 in the three older ones. With the variable unset
  all five pass. The new case first had one momentum offset of 1e-4, and the mistake moved 1 particle of 225, which
  is why it has offsets of 5e-3 now.
- **Along the way.** The `MALIGN_METHOD` and `DZ` changes were checked alone first (`chain-ets`, `chain-routine`):
  all 660 default decks unchanged, and `fmult2`, `mult4`, `kickElemMisalign1` and `kickElemMisalign3` moved toward
  the CPU in the forced list. The CPU binary was rebuilt from the restored `ccbend.c` after the diagnostic prints,
  and its 393 older feature results are the earlier CPU reference's, apart from time stamps.

### Timing

The round trip adds two divisions and two multiplications a slice. In the APS-U style ring with no radiation,
64,000 particles and one pass, with each kernel timed (`show_element_timing`, `ELEGANT_GPU_PROFILE_TIMING=1`), a
`CCBEND` takes 0.397 ms where it took 0.390 ms on November 10: 63.6 ms for the ring's 160 against 62.4 ms, in a
pass of 8.0 s. `KQUAD` (0.924 ms), `KSEXT` (0.674 ms) and `LGBEND` (10.5 ms) are no slower than they were.

### Not done

- `CSBEND` and `LGBEND`, above.
- `CCBEND` with `ISR`, the `exp()` in its fringe, and exact correctors with a kick, would need the host library's
  `pow()`, `exp()` and trigonometric functions on the device.
- A `MULT` with length, a `TAPERAPR` with `TILT`, `FMULT`, `APCONTOUR` with `TILT`, and radiation in a kick magnet.

See `results/2026-11-11_exact-kernels` in GPU-Testing: `final-feature`, `final-feature-cpu`, `final-suite`,
`final-ets`, `final-routine`, `final-csr`, `final-pel`, `final-verify-feature`, `final-verify-suite`, `chain-ets` and
`chain-routine` (the `MALIGN_METHOD` and `DZ` changes alone), `ck-*` and `fin-*` (earlier stages of the same day),
`mutation/`, `source-before/`, `source-final-4/` and `binaries-final-4.sha256`. The probes are
in `tmp/2026-11-11_fmap/ccb` (`probes.txt` to `probes7.txt`, `count.py`) and `tmp/2026-11-11_fmap/malign`
(`probe.py`).

## A test of the `CSBEND` drift with no trigonometric function, in both builds (November 11, 2026)

The section before says a `CSBEND` can't give the CPU's bits while its drift calls `asin()`, `sin()` and `cos()` for
every particle, and that the drift formula prototyped on November 10 would remove those calls if elegant and
gpu-elegant both used it. This is a test of that, in a scratch build. **Nothing here is in the source tree:** the
four files were restored from saved copies and compared, and the verify binary was rebuilt from them and rerun on the
feature set (396) and the test set (39), with no stops and every output identical to the run before the test. Its
checksum is new (`30205621`), since a link carries its date; gpu-elegant, gpu-Pelegant and elegant were not rebuilt.

### What was built

`integrate_csbend_ordn()` (`csbend.c`) and the two device kernels that repeat it got the same drift. The drift turns
the momentum by one angle for every particle, so the sine, cosine and versine of that angle are made once per
substep, on the host in both cases, and the angle-sum formulas with one square root for `cos(phi)` replace the
per-particle `asin()`, `sin()` and three `cos()`. Only the verify binary was built, which runs both sides and
compares them.

Two more differences showed up on the way, both on the device's side and both of the kind the section before
corrected for `CCBEND`:

- **The same division and multiplication after every kick.** `track_through_csbend()` negates its ISR coefficient
  with `ISR` off and doesn't zero it, so the CPU's radiation block runs after every kick of an ordinary `CSBEND`:
  `QX /= (1+delta); QY /= (1+delta);` and back. With the drift alone changed, a `CSBEND` with `SYNCH_RAD` was already
  identical and one without it still had 0.5% of its particles off, now by 1e-18 where it had been 1e-15.
  (`CSRCSBEND` zeroes the coefficient, so it has no such step.)
- **A product in the second-order edge.** `apply_edge_effects()` has `T133 * sqr(y0)`; the device had
  `T133 * y0 * y0`, which is `(T133 * y0) * y0`.

### Result

51 configurations of `CSBEND`, each for 20,000 particles through one magnet with zero tolerance:

| | The CPU's bits |
| --- | --- |
| Before | none of 51. 40 to 3,705 of each magnet's 120,000 values differ, by up to 3e-14 |
| With the drift in both, the round trip and the edge product on the device | 42 of 51 |

The 42: integration order 2, 4 and 6; 1 to 16 slices; angles to 0.79 rad of either sign; `K1` to `K8`; `NONLINEAR=0`;
`USE_BN`; `XREFERENCE` with `F` and `G` terms; `FSE` and its dipole and quadrupole parts; `XKICK`, `YKICK`; `TILT` (also
pi/2 and 2 pi); `ETILT`, `EPITCH`, `EYAW`; `DX`, `DY`, `DZ` with `MALIGN_METHOD` 0, 2 and 4; `EXPAND_HAMILTONIAN`;
`SYNCH_RAD`; edges of first order (with `H1`, `H2`, kick limits, angles to 0.25 rad) and of second order (also with
`FINT`); edge models 2, 3 and 4 at a small edge angle and 2 at a large one; edge model 5 with no fringe integrals;
and the APS-U style `M3` bend with its edges off. Nine `CSBEND`s through the tune program, 81 particles for 32
turns, were identical too.

The nine that still differ all have the CPU calling a library function for each particle:

| Configuration | Values of 120,000 that differ | The function |
| --- | --- | --- |
| `ISR` | 197, in `delta`, up to 4e-19 | `pow()` |
| First-order edges with `HGAP` and `FINT` | 1 (edge angle 0.01) to 161 (0.25), up to 9e-19 | `tan(e - psi/(1+delta))` |
| Edge models 3 and 4 with `FINT` at an edge angle of 0.25 | 158 and 7, up to 9e-19 | the same kind |
| The APS-U style `M3` and `M4` as the ring has them: edge model 5 with fringe integrals | 192 to 222, up to 2e-16 | `exp()`, `sinh()`, `cosh()` |

So the drift is what keeps the body of a `CSBEND` from the CPU's bits, and the test removes it. What is left is in
the edges, and only where they take a library function per particle. For the APS-U style ring that is every
`CSBEND`: 1% of the particles out of each one would still be a unit off.

### What the formula does to elegant's own results

The 424 decks of the forced list, on the CPU alone, with the scratch build against the restored one (which gives
plain elegant's results exactly, all 424):

- 331 unchanged, 93 changed. Seven of the 93 are inside the rounding envelope of the comparisons here; 86 are
  outside it.
- The largest change in a position: under 1e-13 m in 46 decks, 1e-13 to 1e-10 m in 23, 1e-10 to 1e-7 m in 15, and
  more in 9. The large ones are decks that magnify any change in the last digit (CSR, wakes, an optimizer, an
  aperture at the beam's edge) and one that shows the present formula's own error: `csbendAp2`, whose bend has a
  radius of 1e6 m, moves by up to 7e-5 m in the positions of its lost particles.
- Against arithmetic with 60 digits, for 200 particles through one bend: the present formula is off by up to 2e-8 m
  in `x` for that bend (radius 1e6 m, 100 slices) and the trig-free one by 1e-16 m. For the APS-U style `M3` (radius
  30 m) it is 2e-14 m against 4e-18 m, and for a bend of 0.5 rad in 1 m, 1.6e-15 m against 7e-17 m. The present
  form, `x = rho0*(f - 1) + f*x`, takes a difference of nearly equal numbers and multiplies it by the radius.

So adopting it would move elegant's results in 93 of these decks, toward the exact values, and the release baseline
of `elegantTestSet` would need to be regenerated.

### Speed

- **CPU.** The APS-U style ring with 2,000 particles, one pass, one core: a `CSBEND` takes 4.52 ms where it takes
  5.91 ms now (1.31 times faster). The scratch code makes the three constants on every call; cached per magnet it
  would do a little better.
- **GPU.** Not rebuilt for this test. The November 10 prototype took a `CSBEND` from 4.06 ms to 2.75 ms at 64,000
  particles.

### What it would take to adopt

- The drift in `csbend.c` and in the two kernels, with the three constants per substep in `GPU_CSBEND_DATA` (packed
  by `gpuPackCsbendTracking()` and `gpuPackCsrCsbendBodyTracking()`), as in `csbend-drift-both-sides.patch`. This
  changes elegant.
- The round trip and the edge product on the device. These two don't touch elegant and could go in alone, but alone
  they make no `CSBEND` exact.
- `CSBEND` could then join the tune program's exact list where its edges call no library function per particle.
  A ring like APS-U's would still need the same `exp()`, `sinh()`, `cosh()` and `tan()` on both sides (and
  `LGBEND` in the CPU's form, and exact correctors' trigonometric functions).

See `results/2026-11-11_csbend-drift-test` in GPU-Testing: `csbend-drift-both-sides.patch`, `before/` and `scratch/`
(the four files), `probes*.txt` (the lists and their counts before and with the scratch build),
`scratch-cpu-routine`, `restored-cpu-routine` and `scratch-vs-restored-cpu-routine` (elegant's results),
`restored-verify-feature` and `restored-verify-suite` (the verify binary after the restore). The 60-digit check and
the timing decks are in `tmp/2026-11-11_csbend-test`.

## The `CSBEND` drift with no trigonometric function, adopted in both builds (November 11, 2026)

The user took the result of the test above and asked for the change. This one **changes elegant**: `csbend.c`'s
`integrate_csbend_ordn()`, which `CSBEND` and `CSRCSBEND` share, drifts by a different formula, and its results
move in the last digits and, where a deck magnifies them, by more. gpu-elegant has the same formula, so the two
still agree, and now agree to the last bit in the body of a `CSBEND`.

### What changed

- **`csbend.c`.** The drift turns the momentum by one angle for every particle, the substep's length over the
  bending radius. `computeCsbendDriftSteps()` makes the sine, cosine and versine of that angle for each substep,
  and the integrator gets `sin(phi + a)` and `cos(phi + a)` from the sums of angles with one square root for
  `cos(phi)`, where it called `asin()`, `sin()` and `cos()` three times for each particle. The new position is
  `(rho0 (cos(phi) vers(a) + sin(a) sin(phi)) + cos(phi) x)/cos(phi + a)`, which has no difference of nearly equal
  numbers to multiply by the radius. `setCsbendDriftSteps()` makes the constants once for a magnet, before its
  particles; the integrator uses them if they are for its slice length, radius and number of substeps and makes its
  own otherwise, so its result never depends on that call, and it only reads them, which matters where it runs on
  several threads. A particle whose momentum is exactly across the reference direction (`cos(phi)` zero) is now
  lost there. The expanded-Hamiltonian integrator has no such drift and is unchanged.
- **The device's two kernels** (`gpuCsbendTrackParticleData()` and the `CSRCSBEND` body slice) drift by the same
  lines, with the constants made on the host by the same arithmetic (`gpuPackCsbendDriftSteps()`, in
  `GPU_CSBEND_DATA`).
- **The CPU's division and multiplication by 1+delta after every kick**, with radiation off, as in `CCBEND`
  (`momentumRoundTrip`, set as `track_through_csbend()` would have its coefficients).
- **The second-order edge's products** in the CPU's order.
- **Edge constants from the host.** The default edge, the second-order edge and the simple symplectic kick took
  `tan(e)` and `cos(e)` of the edge angle from the device's library for every particle, and the radiation at an
  edge took `tan(e)` too. Those are one number for a magnet, and the host already sent them (`edgeTrig1`,
  `edgeTrig2`), so the device uses the host's. With no fringe angle the vertical kick's `tan(e - psi/(1+delta))`
  is that number as well. The device's values had been the host's for every angle tried, but only by the two
  libraries agreeing.
- **The tune program takes `CSBEND`s.** `gpuTuneProgramExactElement()` claims the CPU's bits for a `CSBEND` with
  no `ISR` whose edges take no library function for a particle: no edge effects, the default edge (1) and the
  simple symplectic kick (3) with `HGAP*FINT` zero, and K. Hwang's fringe (2). With a fringe angle the first two
  take a tangent for each particle, and models 4 and 5 take `exp()` and hyperbolic functions, so those stay on the
  CPU unless a tolerance is set.

### Probes

The lists of the test, on the verify build of the final source: 42 of 51 `CSBEND` configurations give the CPU's
bits and the same nine don't, value for value as in the scratch build. Ten `CSBEND`s through the tune program are
identical. The `CCBEND` list (43 of 44, `ISR` apart) and the kick-magnet list (34 of 37) are as they were.

### elegant's own results

All 660 decks on the CPU, new against the reference of November 9:

- 490 unchanged and 170 changed: 7 inside the rounding envelope and 163 outside.
- The largest change in a position: none in 28 of the 170, under 1e-13 m in 47, 1e-13 to 1e-10 m in 60, 1e-10 to
  1e-7 m in 23, and more in 12. The large ones magnify a change in the last digit: Monte Carlo scattering
  (`touschek1`, `touschek3`: lost particles' rows, by millimeters), CSR and wakes, radiation damping over 110,000
  turns, an optimizer, or an aperture at the beam's edge. `csbendAp2` is the present formula's own error for a
  radius of 1e6 m, as in the test.
- The 424 of the forced list are, deck for deck, the scratch build's results of the test: making the constants
  once per magnet changed no number.
- The GPU test set's 39 on the CPU: 35 unchanged, 4 changed.

The CPU references here are new: `cpu-660`, `cpu-routine-view`, `cpu-csr-view`, `cpu-suite`, `cpu-feature` and
`cpu-pel-og` in `results/2026-11-11_csbend-drift`. The baseline from the 2026.4.0 release (`elegantTestSet`
r36002) still describes that release; 170 of its decks are now expected to differ from the development tree.

### gpu-elegant against the new elegant

| Set | Identical | Inside the envelope | Flagged | Before (old formula on both) |
| --- | ---: | ---: | ---: | --- |
| 660 default decks | 549 | 30 | 81 | 540, 33, 87 |
| Forced routine list (424) | 250 | 79 | 95 | 239, 68, 117 |
| CSR list (33) | 5 | 0 | 28 | 5, 0, 28 |
| GPU test set (39) | 6 | 33 | 0 | 6, 33, 0 |
| Feature set (397) | 69 | 327 | 1 | 66, 329, 1 of 396 |
| Pelegant, 47 tests, against the `-Og` CPU Pelegant | 46 | 1 | 0 | |

- **No deck that was identical to the CPU differs now**, in the 660 or the forced list. In the 660, 9 more are
  identical and the class improved in 11; it got worse in one, `TTF2`, where two rows of 511 are a unit off in the
  last place of `s` and the beam-size minimum built from them leaves the envelope. In the forced list the class
  improved in 27 and got worse in none.
- **Feature set.** 46 cases moved toward the CPU, among them every `magnet-csbend-*` case
  (`magnet-csbend-loss-compaction` from 8,857 values that differ to 246, `magnet-csbend-high-order` from 676 to 32),
  and `slice-analysis` and `slice-analysis-whole-beam` are now identical. `isr-small-batches` and its `-loss`
  variant have a few more values that differ (664 to 703, 121 to 128): they have `ISR` in five kinds of element,
  each with a `pow()` from two libraries. `scmult-zero-length` is flagged as before.
- **Test set.** `csbend-heavy` went from 697 values that differ to 502, `csr-heavy` from 289 to 184,
  `magnet-synchrad-heavy` from 12 to 8.
- **Against the GPU references of the section before**, as expected with a new formula: 488 of the 660 unchanged,
  329 of the forced 424, 6 of the 33 CSR decks, 35 of the test set's 39, 41 of the 47 Pelegant tests. The decks that
  moved are the ones that moved on the CPU.

### Verify build and the deliberate mistakes

- Feature set (397), test set (39) and eight frequency-map and footprint decks of the 660 complete with no stops.
  20 feature cases run the program, over 36 intervals; in 32 every particle has the CPU's history and final
  coordinates, and the other four are the two `MULT` cases that aren't claimed. The new case,
  `search-frequency-map-program-csbend`, has eight kinds of `CSBEND` and momentum offsets of 5e-3.
- `fmap1` and `fmap2` (eight `CSBEND`s of 45 degrees): all 441 particles identical, where they differed by up to
  8e-8 of a history's range. `fmap5`: 1 particle of 1,001 differs, by 1e-16. Their bends have `HGAP` and `FINT`, so
  the program doesn't claim them and the CPU tracks these maps as before; that one particle is why.
- Two mistakes behind a variable, in a verify build that was replaced afterwards (source restored and compared):
  the round trip left out, and the versine made as `1 - cos(a)`. The first stops the new case (14 particles
  differ); the second stops it and `search-frequency-map-prepass-magnet-losses`. With the variable unset all pass.

### Speed

| | Before | Now |
| --- | ---: | ---: |
| A `CSBEND` on the CPU, APS-U style ring, 2,000 particles | 5.91 ms | 4.30 ms |
| A `CSBEND` on the device, the same ring, 64,000 particles | 4.06 ms | 2.82 ms |
| `fmap1`, `fmap2`, `fmap5` in elegant | 5.7, 1.6, 20.0 s | 3.4, 1.3, 15.1 s |

A pass of the ring at 64,000 particles on the device takes 7.8 s where it took 8.05 s.

### Not done

- A `CSBEND` whose edges have a fringe angle (`HGAP` and `FINT`), or models 4 and 5, or `ISR`: the CPU takes a
  tangent, an exponential or a power for each particle there. These would need the same functions compiled into
  both builds. The APS-U style ring's `CSBEND`s are of this kind, and it has `LGBEND`s and exact correctors with
  kicks besides.
- The slow tests with larger beams (`cases/slow-wide`, 54 copies) were not rerun; their references are from the old
  formula.
- No note of the change was added to the manual.

See `results/2026-11-11_csbend-drift` in GPU-Testing: `cpu-660`, `cpu-routine-view`, `cpu-csr-view`, `cpu-suite`,
`cpu-feature`, `cpu-pel-og`, `gpu-660`, `gpu-routine`, `gpu-csr`, `gpu-suite`, `gpu-feature`, `gpu-pel`,
`verify-feature`, `verify-suite`, `verify-fmap`, `mutation/`, `source-before/`, `source-final/`,
`csbend-drift-adopted.patch` and `binaries-final.sha256`.

## `CSBEND` on host threads in CPU-tracked frequency maps (November 12, 2026)

A frequency map or tune footprint that the device program doesn't take is tracked on the CPU, and with
`-ompThreads` the magnets share their particle loops among host threads. `CSBEND` did so only with
`MALIGN_METHOD` not 0, edge model 5 or no edges, and no radiation, so a default `CSBEND` stayed on one thread.
In `fmap5` the bends are 11.3 of 14.7 s and that time was the same with eight threads.

### What changed

`trackCsbendParticleOmp()` in `csbend.c` is now the body of the particle loop of `track_through_csbend()` for a
whole element, statement for statement: the `MALIGN_METHOD=0` offsets and tilt at entrance and exit, edge model 1
(first and second order), 2, 3, 4 and 5, classical radiation with its edge corrections, and the path-length
scaling the serial loop always does. The test in `track_through_csbend()` that sends a bend to the threads no
longer asks for a misalignment method, an edge model or no `SYNCH_RAD`. It still refuses `ISR` (random numbers
in particle order), distribution-based radiation, photon output, spin, `sigmaDelta2`, acceptance arrays, global
loss coordinates, reference correction and single slices, which all stay in the serial loop.

The old routine added the body's path length directly where the serial loop scales the path by the ratio of the
velocities first; with equal velocities that can differ in the last bit. The new one does what the serial loop
does.

Everything changed is inside `#ifdef HAVE_GPU`, so elegant is not affected, and with one thread gpu-elegant
doesn't reach it.

### Checks

- **One thread against four, 55 bend settings.** A small ring with one `CSBEND` kind and a `MAXAMP`, 128 particles,
  32 turns, the map tracked on the CPU (`ELEGANT_GPU_ENABLE_TUNE_PROGRAM=0`): the `.fma` files are the same bytes
  for all 55 (the 51 settings of the November 11 drift test and four more), with particles lost in most of them.
  Five needed a wide aperture for the reference particle to survive.
- **Feature set.** New case `search-frequency-map-csbend-openmp-kinds` (eight bend kinds; 50 of 128 particles
  survive). The suite has 398 cases: against the CPU 70 identical, 327 within rounding, 1 flagged
  (`scmult-zero-length`, as before). The 50 `search-*` cases with `--elegant-argument=-ompThreads=4` are all
  identical to their one-thread results.
- **elegantTestSet search decks with `-ompThreads=4`.** 38 decks (dynamic aperture, frequency maps, momentum
  aperture, Touschek): all identical to one thread, 29 against the November 11 default reference and 9 against
  one-thread runs of this build.
- **Mistakes.** Eight errors behind a variable in the thread routine, in a build that was replaced afterwards
  (source restored and compared): an entrance offset term, the gap term of the default edge, of edge model 3 and of
  model 4, the radiation correction at the entrance edge, the path length, the exit pole-face curvature, and the
  `ETILT` offset. The new case differs from its one-thread result under each, and is the same with the variable
  unset.
- Not rerun: the forced routine list, the CSR list, the GPU test set, gpu-Pelegant and the verify build's sets. They
  run with one thread, which doesn't reach the changed code.

### Speed

Alone, P-cores, wall time:

| Threads (`-ompThreads`) | `fmap5` | `fmap1` |
| ---: | ---: | ---: |
| 1 | 14.7 s | 3.07 s |
| 2 | 7.5 s | 1.62 s |
| 4 | 4.1 s | 0.88 s |
| 8 | 2.4 s | 0.56 s |
| 16 (8 P-cores and 8 E-cores) | 3.0 s | 0.64 s |

Before, eight threads gave 12.2 s and 2.65 s.

### Found on the way, not fixed

- **A frequency map with an RF cavity differs from elegant.** With an `RFCA` in the ring, gpu-elegant's batched
  frequency map gives tunes that differ from elegant's by about 1e-4, with one thread too; with
  `ELEGANT_GPU_ENABLE_BATCHED_TUNE_TRACKING=0` it matches. Elegant tracks each particle one turn per call; the
  batch tracks all turns in one call. The cause was not looked for. No frequency-map deck in elegantTestSet has a
  cavity, and the one feature case with one (`search-frequency-map-program-opcodes`) runs on the device program
  and matches.
- `OMP_NUM_THREADS` alone still doesn't start the tracking threads; only `-ompThreads` does.
- `fmap4` (`CWIGGLER`s) gains little: 544 s to 490 s on eight threads.
- Idle threads still spin: `fmap5` on eight threads uses 18.5 s of CPU time for 14.7 s of work.

See `results/2026-11-12_omp-bends` in GPU-Testing: `final-cpu-feature`, `final-gpu-feature`,
`final-gpu-feature-search-omp4`, `gpu-ets-search-omp4`, `final-gpu-ets-nine-1`, `mutation/`, `source-before/`,
`source-final/`, `omp-bends.patch`; probes and timings in `tmp/2026-11-12_omp`.

## Frequency maps of a ring with an RF cavity (November 12, 2026)

### The bug

With an `RFCA` that has a voltage, gpu-elegant's batched frequency map gave tunes that differ from elegant's: by
1.08e-3 for every particle in a FODO ring with a 5 MV cavity at 170 degrees and 32 turns. It did so with one
thread, with the device program on or off, and whether or not particles were lost.

elegant's `computeTunesFromTracking()` calls `do_tracking()` once a turn with one particle. Each call begins by
deleting the phase references and resetting the RF elements (`RESET_RF_FOR_EACH_STEP`, set unless `run_control`
has `reset_rf_for_each_step = 0`), so the cavity takes its reference time from that particle on that turn: the
particle sees the cavity's nominal phase on every turn, and there is no synchrotron motion in the map. The batch
tracks all turns of all particles in one call, so the reference was set once, on the first turn, from all the
particles, and they then moved in phase. A print of the cavity's reference time in a temporary build showed it:
constant in the batch, a new value each turn in elegant. With `reset_rf_for_each_step = 0` the reference is the
fiducial particle's in both and the two programs already agreed to the last bit.

### The fix

`gpu_batched_frequency_map_beamline_supported()` in `gpu_stub.c` now refuses a beamline holding an element whose
reference elegant sets again at each call (`gpuElementReferenceResetEachCall()`): with the reset on, an `RFCA`
that has a voltage, a `Q` or `CHANGE_T`, and `RFCW`, `RFDF`, `RFTM110`, `MODRF`, `RAMPRF`, `TMCF`, `RFTMEZ0`,
`CEPL`, `TWPL`, `TWLA`, `TWMTA`, and the mode elements with `RESET_FOR_EACH_STEP`; a `KICKER` always. Such a map
then takes elegant's own path, one particle and one turn at a time, on one thread. With
`reset_rf_for_each_step = 0`, or a cavity with no voltage, the map is batched as before. The tune footprint was
not affected: it batches only what the device program takes, which has no cavity with a voltage.

No CPU source changed.

### Checks

- Seven small rings (cavity alone, with a second interval, with losses, with bends, bends without a cavity, all
  together, and the reset off): the five that differed from elegant are now the same bytes; the two that agreed
  still do and are still batched.
- Two feature cases, both identical to the CPU: `search-frequency-map-rfca-reference` (bends, losses, cavity,
  second interval; the build before the fix differed from elegant on this ring) and
  `search-frequency-map-rfca-no-reset` (the same with the reset off, which stays batched). The suite has 400
  cases: against the CPU 72 identical, 327 within rounding, 1 flagged (`scmult-zero-length`, as before). The 52
  `search-*` cases with `-ompThreads=4` are identical to their one-thread results.
- `fmap1`, `fmap2`, `fmap3` of elegantTestSet are identical to the reference; no frequency-map deck there has a
  cavity.
- `RFMODE` and `TRFMODE` with a `CHARGE` were already refused by the existing rule on collective elements, and
  match. `SREFFECTS` with `QEXCITATION` is batched and matches.

### Not done

- Such maps got no speedup with this fix alone; the next section puts a plain `RFCA` back in the batch.
- The batched dynamic-aperture and momentum-aperture searches were not examined for the same effect. They confirm
  their boundary trials on the CPU.
- The forced routine list, CSR list, GPU test set and gpu-Pelegant were not rerun: only the frequency map's choice
  of path changed.

See `results/2026-11-12_rf-fmap` in GPU-Testing (`cpu-feature`, `gpu-feature`, `gpu-feature-search-omp4`,
`gpu-ets-fmap`, `source-before/`, `source-final/`, `rf-fmap.patch`) and the rings in `tmp/2026-11-12_rffmap`.

## A plain `RFCA` in the frequency-map batch, each particle its own reference (November 12, 2026)

The fix above sent every map with a powered cavity down elegant's one-particle path. A plain `RFCA` is now tracked
in the batch again, doing what elegant does.

### What changed

- `simple_rf_cavity()` (`simple_rfca.c`, inside `HAVE_GPU`): while a frequency-map batch is tracked on a beamline
  whose RF is reset at each call (`gpuTuneOwnRfReference()`), a cavity with a voltage takes the particles one at a
  time. Before each it deletes the phase references and clears `fiducial_seen`, as the start of elegant's
  `do_tracking()` call would, then calls `trackRfCavityWithWakes()` for that one particle. The reference time,
  `LINEARIZE` or `LOCK_PHASE` averages and everything else are then that particle's own, from the same code
  elegant runs. A particle the cavity removes is swapped to the end.
- `doFrequencyMapBatched()` (`frequencyMap.c`) turns this on for its tracking and off after
  (`gpu_batched_tune_own_rf_reference()`); it is on only if the beamline's fiducial flag has
  `RESET_RF_FOR_EACH_STEP`.
- `gpuElementReferenceResetEachCall()` (`gpu_stub.c`) no longer refuses such an `RFCA`, unless it has `CHANGE_T`
  (it moves the tracking clock of the whole call) or `CHANGE_P0` (it rematches the central momentum to one
  particle), or another `RFCA` has the same `PHASE_REFERENCE` number (elegant lets the first set the reference the
  others use on that turn). `RFCW`, `RFDF`, `MODRF` and the others stay refused.
  `ELEGANT_GPU_ENABLE_TUNE_RFCA_REFERENCE=0` refuses every powered `RFCA` again.

elegant itself is not changed: every edit is inside `HAVE_GPU`.

### Checks

- **27 cavity settings** in a ring with bends, a `MAXAMP` and a second interval, elegant against gpu-elegant with
  one and with four threads, `.fma` files compared byte for byte: all the same. Batched: phases 0, 90, 170 and
  180 degrees, a long cavity with 5, 1 and 0 kicks, end focusing, offsets, `FIDUCIAL` `light` and `pmaximum`,
  `T_REFERENCE`, zero frequency, two cavities, one cavity twice in the ring, two cavities with their own
  reference numbers, the reset off, and a voltage that loses every particle. Not batched, and the same:
  `BODY_FOCUS_MODEL="SRS"`, `LINEARIZE`, `LOCK_PHASE`, `Q`, a shared reference number, `CHANGE_T`, `CHANGE_P0`,
  `RFCW`, `RFDF` (the first four because other rules keep those cavities off the batch). A 28th setting had a
  `FIDUCIAL` string elegant rejects.
- **The seven rings of the section above**: all the same bytes as elegant with one and four threads, all batched.
- **Feature set.** `search-frequency-map-rfca-reference` is now batched; new `-rfca-two-cavities` (batched),
  `-rfca-shared-reference` and `-rfca-change-t` (not batched). The suite has 403 cases: against the CPU 75
  identical, 327 within rounding, 1 flagged (`scmult-zero-length`, as before). The 55 `search-*` cases with
  `-ompThreads=4` are identical to their one-thread results. The verify build runs the five cavity cases with no
  stop and the CPU's output.
- **elegantTestSet** `fmap1`, `fmap2`, `fmap3`: identical to the reference.
- **Mistakes** behind a variable, in a build replaced afterwards (sources restored and compared): `fiducial_seen`
  not cleared, the references not deleted, and the per-particle mode never turned on each make
  `-rfca-reference` and `-rfca-two-cavities` differ from elegant; ignoring a shared reference number makes
  `-rfca-shared-reference` differ. Allowing `CHANGE_T` in the batch changed nothing in `-rfca-change-t`: that
  rule is a precaution this case doesn't prove necessary.

### Speed

A 41 by 41 map of 256 turns of the small ring with a cavity, alone: elegant 4.5 s; gpu-elegant 4.3 s with one
thread and 1.05 s with `-ompThreads=8`, the same file each time. Kept off the batch it takes 4.75 s.

### Not done

- `RFCW`, `RFDF`, `MODRF` and the other RF elements, cavities with `CHANGE_T` or `CHANGE_P0`, and cavities that
  share a reference number are still tracked one particle at a time.
- No case has a particle removed inside the cavity while others go on, so the swap there has not been seen to
  work.
- The forced routine list, CSR list, GPU test set and gpu-Pelegant were not rerun: the new code runs only inside
  a frequency-map batch.

See `results/2026-11-12_rf-batch` in GPU-Testing (`cpu-feature`, `gpu-feature`, `gpu-feature-search-omp4`,
`verify-rfca`, `gpu-ets-fmap`, `mutation/`, `source-before/`, `source-final/`, `rf-batch.patch`) and
`tmp/2026-11-12_rffmap` (`rf_probe.py`, `rf-probes.txt`, `rf-probes-result.txt`).

## Sums over the beam, measured again (November 12, 2026)

The November 10 ranking had the sums over the beam as the main device cost of the LCLS-II deck (2.7 s of 3.1 s of
kernel time a step). They were timed again one kernel at a time, with a new counter, after the work of November
10 and 11.

### A timer for each kernel call site

With `ELEGANT_GPU_PROFILE_TIMING=1` and `ELEGANT_GPU_VERBOSE=1`, the activity summary now lists the helper and
reduction kernels by the line of `gpu_stub.c` that ran them, with calls and seconds
(`elegant CUDA: reduction kernel site gpu_stub.c:10742 calls=2604 seconds=0.437729`). Nothing else changed: the
403 feature cases and the 39 GPU test-set cases are identical to their references.

### What the LCLS-II deck spends now

200,000 particles, three steps of about 3.7 s, alone on one P-core, 12.0 s in all:

| Part of a step | Time |
| --- | ---: |
| All reduction kernels | 0.44 s |
| of which the centroid and sigma sums after each element (2,604 calls, 0.17 ms each) | 0.437 s |
| of which every other sum (294 calls) | 0.008 s |
| Host `sin` and `log` for cavity kicks that match elegant (`ELEGANT_GPU_RFCA_REFERENCE_MATH`) | 0.73 s |
| Host sum in particle order for momentum matching (`ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER`) | 0.40 s |
| Five zero-angle `CSBEND`s tracked on the CPU, 64 calls | 0.68 s |
| `WATCH` output, 8 files | 0.44 s |

The 64-thread kernels and the 289 copies to the host of the old ranking are gone from this deck: 16 copies a
step remain, none for a sum. With `OMP_NUM_THREADS=8` the run takes 10.4 s.

### A change tried and not kept

The centroid and sigma sums use at most 64 blocks of 64 threads, so each thread adds 49 particles at 200,000.
Giving them more blocks above 65,536 particles (up to 4,096, 16 particles a thread) took the sums from 0.440 s to
0.361 s a step, 2% of the run: a call costs 0.04 ms fixed and the rest follows the two reads of the beam, not the
thread count (0.053 ms at 20,000 particles, 0.040 ms at 2,000). It also changes the rounding of every `.cen` and
`.sig` value above 65,536 particles. It was taken out again; the patch is
`results/2026-11-12_sums/tried-wide-blocks/wide-blocks.patch`.

### Left

- The host arithmetic that keeps cavity kicks and momentum sums the same as elegant's is 1.1 s of a 3.7 s step.
  Only host threads shorten it without changing results, and those are used only when the run asks for them.
- One read of the beam instead of two for the centroid and sigma sums (products about an approximate center)
  would save about 0.15 s a step and change the rounding of every `.sig` file. Not done.
- The zero-angle `CSBEND`s on the CPU.

See `results/2026-11-12_sums` in GPU-Testing and the runs in `tmp/2026-11-12_sums`.

## The ring's magnet kernels, measured again (November 12, 2026)

The third item of the November 10 ranking: in the ring nearly all of a pass is the arithmetic of four kernels.
That is still so. APS-U style ring, 64,000 particles, radiation and ISR on, alone on one P-core; a pass is 4.9 s
(6.9 s on November 10):

| Element | One element | A pass | Share | Added by `SYNCH_RAD` | Added by ISR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LGBEND` (160) | 13.9 ms | 2.22 s | 45% | 0.7 ms | 2.7 ms |
| `CSBEND` (200) | 5.75 ms | 1.15 s | 24% | 0.7 ms | 2.25 ms |
| `KQUAD` (400) | 1.86 ms | 0.75 s | 15% | 0.23 ms | 0.71 ms |
| `KSEXT` (240) | 1.45 ms | 0.35 s | 7% | | |
| `EDRIFT` (5,250) | 0.035 ms | 0.18 s | 4% | | |
| `CCBEND` (160) | 0.54 ms | 0.09 s | 2% | 0.04 ms | 0.10 ms |

ISR is about 1.17 s of the pass. Of that, 0.68 s is making the random numbers (3,480 fills in three passes, 2.07 s),
now timed by itself in the profile summary (`ISR numbers kernel site`).

### Where a fill's time goes

Timed in a throwaway build with parts left out: `sqrt(-2 log(u1))` 44%, `sin` 23%, the prefix sum and scatter that
drop the numbers beyond the Gaussian limit 23%, the rest (the generator's jump, memory) 9%.

### Kept: 32-bit flags and ranks in the fill

The accepted flags and their ranks were 64-bit; they are 32-bit now, which halves what the prefix sum moves. The
fill went from 2.07 s to 1.89 s for three passes (1.2% of the tracking), with the same numbers: the verify build
compares every draw with `gauss_rn_lim()` and the count of uniforms used, 631 fills over the 26 ISR feature
cases, all passing. A fill of more than 1e9 pairs is refused; it wouldn't fit the device before either. The 403
feature cases and 39 GPU test-set cases are identical to their references.

### Tried and not kept

- **Two quotients from one division.** Where a kernel divides two numbers by the same value (the slope
  conversion, the momentum round trip), one reciprocal and two products corrected with their exact remainders
  give the same correctly rounded quotients. It was slower: `KSEXT` 1.45 to 1.52 ms, `KQUAD` 1.86 to 1.95 ms.
  This device divides quickly. Saved as `trial-divide-pair.cu`.
- **Kernels compiled without the ISR step** for elements with no ISR: nothing for `CSBEND` and `CCBEND`, 2.7% for
  a `KQUAD` with no radiation. Not worth eight kernel variants. Saved as `trial-isr-free-kernels.cu`.

### Would help, but changes results

- **The ISR kick's 3/4 power.** With `sqrt(F2 * sqrt(F2))` for `pow(F2, 0.75)` in the multipole and `CSBEND`
  kernels, as `LGBEND` already has, the tracking took 13.64 s where it took 14.65 s (7%): `KQUAD` 1.86 to
  1.60 ms, `KSEXT` 1.45 to 1.23 ms, `CSBEND` 5.75 to 4.89 ms. It moves the result of every element with ISR by
  rounding. Done in elegant too it would make this step the same arithmetic in both programs, as the `CSBEND`
  drift was. Not done.
- **The fill's `log` and `sin`**, 0.46 s a pass, are the device library's. A faster pair would move every ISR
  number by rounding.

### Left

`LGBEND`, 45% of the pass, was rewritten on November 10 and is one square root and one division for each of its
800 substeps a particle. I found nothing more there that keeps its results.

See `results/2026-11-12_ring` in GPU-Testing (`ring.patch`, the trial sources, `gpu-feature`, `gpu-suite`,
`verify-isr`) and the runs in `tmp/2026-11-12_ring`.

## `-ompThreads` and `OMP_NUM_THREADS` made the same, and cheap loops kept off the threads (November 12, 2026)

### The discrepancy

The two ways of asking for host threads reached different code. `OMP_NUM_THREADS` set the host thread pool, the
tune analysis, MKL and the switch that sends a small aperture search to the device batch, but none of the
particle loops. `-ompThreads` set the particle loops, the pool and the tune analysis, but not MKL or that switch.
So `dynAp7` got faster only with the variable (4.2 s to 1.4 s) and `fmap5` and `bggexp2` only with the option.

### One count

`main()` now settles one count of host threads once the command line is read: `-ompThreads=N` if given, else
`OMP_NUM_THREADS` if set, else one. `gpuSetHostThreads()` makes it the count of the tracking loops, and with the
option `omp_set_num_threads()` and, unless `MKL_NUM_THREADS` is set, `mkl_set_num_threads()` are called too, so
the pool, the tune analysis, the search switch and MKL all see the same number either way. The command line wins:
`-ompThreads=1` with `OMP_NUM_THREADS=8` runs on one thread. When the count is more than one, a line on stderr
says so and where it came from (`gpu-elegant: using 4 host threads (OMP_NUM_THREADS).`); it replaces the
"experimental ... cores not shared" notice of `-ompThreads`. gpu-Pelegant is unchanged.

### Cheap loops

With the variable now turning the particle loops on, their cost mattered more. Loops that do a few operations a
particle (the kick and copy loops of `WAKE`, `TRWAKE`, `CWAKE`, `RFMODE`, `TRFMODE`, `FRFMODE`, `FTRFMODE`, the
CSR kick of `CSRCSBEND`, `computeTimeCoordinatesOnly()`, and the space-charge kicks) started the OpenMP team for
any beam of 64 particles, and the other threads then spun: `feedback6` (100-particle buckets through `FRFMODE`
and `FTRFMODE`) used 489 s of CPU time for a 62 s run on eight threads, and `ibScatter3` 141 s for 20 s, from
`computeTimeCoordinatesOnly()` called by `index_bunch_assignments()` on 1,000 particles every pass (found with a
temporary trace of the callers that ask for threads). These loops now ask `gpuOmpLightLoopRequested()`, which
says yes from 100,000 particles; `ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES` changes that. Magnets, `BGGEXP`,
`CWIGGLER` and the search and frequency-map tracking keep the 64-particle rule: a particle there is microseconds
of work.

### Timings

Alone, eight P-cores for the threaded runs; wall time, and CPU time in brackets:

| Deck | One thread | `OMP_NUM_THREADS=8`, before | `-ompThreads=8`, before | Either, now |
| --- | ---: | ---: | ---: | ---: |
| `feedback6` | 61.5 s (61) | 59.0 s (59) | 62.5 s (489) | 59.1 s (59) |
| `ibScatter3` | 18.2 s (17) | 18.2 s (17) | 19.8 s (141) | 20 to 22 s (17) |
| `bunchComp1` | 7.4 s (7.1) | 7.0 s (6.8) | 7.7 s (34.5) | 7.1 s (6.9) |
| `dynAp7` | 4.2 s (4.2) | 1.41 s (1.4) | 4.2 s (4.2) | 1.41 s (1.4) |
| `fmap5` | 14.7 s (14.7) | 12.7 s (14.5) | 12.2 s (38.2) | 2.4 s (18.5) |
| `bggexp2` | 10.0 s (10.0) | 9.6 s (9.5) | 1.7 s (10.4) | 1.4 s (10.4) |
| `modulate4` | 20.8 s (18.7) | 20.6 s (18.2) | 23.4 s (18.1) | 19.9 s (18.3) |
| LCLS-II, 200,000 particles | 11.7 to 13.5 s (11.2) | 10.7 s (11.4) | 11.7 s (11.8) | 11.9 s (11.4) |

The `fmap5` gain is from the `CSBEND` work of the same day. The two switches now give the same CPU time on every
deck. `ibScatter3`'s and LCLS-II's wall times vary from run to run with file writing; their CPU times do not.

### Checks

- **elegantTestSet with four threads.** All 660 default decks with `OMP_NUM_THREADS=4` are identical to the
  one-thread GPU reference, and again with `ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES=64` so that the cheap loops
  run on the tests' small beams. This is the first time the whole set has been run with threads.
- **Feature set.** The 403 cases are identical to the reference with one thread, with `OMP_NUM_THREADS=4` and
  with `-ompThreads=4` (both with the cheap-loop limit at 64).
- **GPU test set.** The 39 cases are identical to the reference.
- **Precedence**, on `fmap5`: the variable at 4 and the option at 4 both take 4.1 s with 15.8 s of CPU time and
  print their line; `-ompThreads=2` with `OMP_NUM_THREADS=8` takes 7.4 s; `-ompThreads=1` with
  `OMP_NUM_THREADS=8`, and `OMP_NUM_THREADS=1`, take 14.8 s like no setting at all.
- The CPU build wasn't rebuilt: `GPU_OMP_PARALLEL_FOR` drops its condition there, so the new function never
  appears in it, and the `elegant.c` change is inside `HAVE_GPU`.

### Not done

- Threaded MKL was kept at one thread in these runs (`MKL_NUM_THREADS=1`), as in every reference run. With MKL
  following the thread count, its own rounding with several threads was not examined.
- Idle threads still spin inside a batched search or frequency map while the host does serial work between
  elements (`fmap5`: 18.5 s of CPU time for 14.7 s of work).
- Single-particle searches gain nothing from threads.

See `results/2026-11-12_threads` in GPU-Testing (`gpu-660-omp4`, `gpu-660-omp4-light64`, `feature-1`,
`feature-omp4`, `feature-opt4`, `suite-1`, `threads.patch`, `source-before/`, `source-final/`) and the runs in
`tmp/2026-11-12_threads`.

## `ILMATRIX` on the device and the `SLICE` element in one pass (November 12, 2026)

A ring collective-effects run was timed element by element for the first time (`collectiveRing1` from
elegantTestSet: 480,000 particles in 48 bunches, 100 turns). The plan had been to put `ILMATRIX` on the device,
on the guess that it held such runs back. The measurement said otherwise:

| | elegant | gpu-elegant | gpu-elegant, 8 threads |
| --- | ---: | ---: | ---: |
| Whole run, before | 172.8 s | 156.8 s | 154.7 s |
| `SLICE` (6 a turn) | 99.4 s | 90.1 s | 88.2 s |
| `WATCH` (18 a turn) | 24.6 s | 23.9 s | 24.4 s |
| `ILMATRIX` | 2.8 s | 2.8 s | 2.8 s |

gpu-elegant gained almost nothing: the beam is on the host for nearly the whole turn, and 58% of the run is the
`SLICE` diagnostic. Both were changed, `SLICE` first.

### `SLICE` in one pass (gpu-elegant only)

`dump_slice_analysis()` calls `accumulate_beam_sums()` once per slice, and each call goes through the whole
beam about forty times to find the slice's few particles: 50 slices of a 480,000-particle beam took 0.17 s a
call. In gpu-elegant's build (`HAVE_GPU`, not `USE_MPI`, element on the host) `slicePoint.c` now finds each
slice's particles once and gives `accumulate_beam_sums()` only those, with the same limits:

- A particle is in slice `k` when `t0[k] <= t <= t0[k] + dt0`, both ends included, with `t0[k]` found by
  repeated addition as the loop finds it. Both limits never decrease with `k`, so two binary searches give the
  first and last slice a particle is in. A particle on a limit is in two slices, and one above the last
  rounded limit is in none, as before.
- `accumulate_beam_sums()` tests the particles again with the same comparisons and adds them in particle
  order with the same Kahan sums, so every sum is the same to the last bit.
- An empty slice still goes through the sums, with one particle that isn't in it, because they leave 0/0 in
  the sigma matrix.
- When the slices can't be told apart (`dt0` not above zero, or a limit that doesn't move when `dt0` is added)
  the loop over the whole beam is used.

`ELEGANT_GPU_SLICE_SINGLE_PASS=0` uses the loop over the whole beam. elegant's own build is unchanged; the same
code would work there.

### `ILMATRIX` on the device

`trackWithIndividualizedLinearMatrix()` in `do_tracking.c` hands an `ILMATRIX` element to
`gpu_track_ilmatrix()` (`gpu_stub.c`), whose kernel (`gpuIlmatrixTrackKernel` in `gpu_cuda_runtime.cu`) follows
the CPU's code statement for statement: the rotation for `TILT`, removal of the dispersive orbit, the betatron
amplitudes, the two phase advances with chromaticity to third order and tune shifts with amplitude, the three
loss tests, the 6 by 6 first-order product as `track_particles()` does it, and the way back.

- **Sines and cosines.** CUDA's differ from glibc's by a unit in the last place now and then. With output in
  CPU order (`gpuBase.orderSensitiveOutputNeeded`, as for an RF cavity) a first kernel makes the phase
  advances, the host takes `sin()` and `cos()` of them (on the sleeping thread pool from 50,000 particles, if
  the run asked for threads) and the tracking kernel uses those. The result is then the CPU's to the last
  bit. Without such output the kernel takes its own. `ELEGANT_GPU_ILMATRIX_REFERENCE_MATH=1` or `0` sets it
  for every run.
- **Losses.** Lost particles are left as the CPU leaves them (in the element's frame, without their
  dispersive orbit, with `z` and the momentum) and moved behind the survivors in the order the CPU's swaps
  give (`gpuBuildCpuLossOrderDestination`), `accepted` rows too.
- **The rotation's cosine and sine** come from `rotateBeamCoordinatesForMisalignment()` itself, applied to two
  unit vectors, so its special angles are the CPU's.
- **`(long)` of the tune** for the resonance test follows the CPU for values a `long` can't hold.
- **Stays on the CPU:** spin tracking, global loss coordinates, backtracking, gpu-Pelegant, and the
  `LINEAR_CHROMATIC_MATRIX` mode, which uses the same function for other elements.
- `ELEGANT_GPU_ENABLE_ILMATRIX=0` keeps it on the CPU; `ELEGANT_GPU_MIN_ILMATRIX_PARTICLES` (64) is its own
  threshold, under the magnets' rule like `SREFFECTS`. `ILMATRIX` has `GPU_SUPPORT` in `track_data.c` now, so
  the dictionary says it is GPU capable (elegant was rebuilt for that flag).

### Result

One run each, alone:

| | elegant | gpu-elegant | gpu-elegant, 8 threads |
| --- | ---: | ---: | ---: |
| Whole run, before | 172.8 s | 156.8 s | 154.7 s |
| Whole run, now | 170.8 s | 76.6 s (71 to 77 in three runs) | 68.4 s |
| `SLICE` | 93.5 s | 5.1 s | 4.9 s |
| `ILMATRIX` | 2.8 s | 1.0 s | 0.4 s |
| CPU time | 135 s | 51 s | 77 s |

`SLICE` accounts for 85 s of the 80 to 86 s gained and `ILMATRIX` for 1.8 s. With the host's sines and cosines
an `ILMATRIX` call takes 2.0 ms per 100,000 particles against 5.7 ms on the CPU; with the device's own (no
output in CPU order) it takes 0.21 ms, and the feature-timing deck of nothing but `ILMATRIX` and matrix elements
went from 5.0 s to 0.32 s. What is left in the ring deck is host work: `WATCH` 26 s (pages of 10,000 particles
from a 480,000-particle beam, 18 a turn), `ZTRANSVERSE` 9 s, `FTRFMODE` 7 s, `HISTOGRAM` 7 s, `ZLONGIT` 4 s.

### Checks

- **The ring deck's 66 output files.** The slice files are byte-identical with and without the single pass,
  and every file's data is elegant's (the files differ in the version string and, in the watch files, the
  elapsed-time and memory parameters). The threaded run's files are the one-thread run's.
- **Feature set, 414 cases** (403 before): 413 within rounding of elegant, 85 identical, 1 flagged
  (`scmult-zero-length`, as before). The 403 older cases give the same data as the references in both
  builds. Nine `ILMATRIX` cases and two `SLICE` cases are new; ten of them must be identical to elegant and
  are, and `ilmatrix-device-math` (the device's own sines) is within rounding by design.
- **`GPU_VERIFY`.** 168 `ILMATRIX` calls in the new cases compared with the CPU's code in the same run: zero
  difference in 163, 90 of them in the two cases that lose particles; the 5 with the device's sines differ by
  1e-12 relative.
- **Deliberate mistakes, 18 in a trial build, all caught** by a case that must be identical. `SLICE`: an
  upper limit that excludes the limit itself, a lower limit that does, an empty slice skipped, members in
  reverse order. `ILMATRIX`: the device's sines used with output, a lost particle left with its incoming
  coordinates, no rotation back, the amplitude term of the path length dropped, its cross term dropped, the
  third-order momentum compaction dropped, the dispersion's slope with momentum dropped, `chrom3` dropped, the
  cross term of the tune shift dropped, `ALPHAX1M` dropped, two lost rows swapped, two surviving rows swapped,
  the resonance test without its factor of two, and `accepted` left unsorted. Three first tries didn't test
  what they meant to (two were applied after the phases had gone to the host, and the row swap undid itself
  on the next loss); they were corrected and then caught.
- **Routine list, 424 tests, forced settings:** 423 unchanged against the GPU reference. `ilmatrix4` differs in
  `run.fin` by 2e-15 relative: the beam now ends on the device, which makes the final sums in its own order;
  its coordinate `WATCH` file is unchanged. `ilmatrix4`, `rfmode5` and `modulate4` ran 10,000, 5,999 and
  18,600 `ILMATRIX` calls on the device.
- **Default settings:** the 16 decks of the 660 with an `ILMATRIX` or `SLICE` element are unchanged against
  the GPU reference, and elegant's against the CPU reference.
- **GPU test set:** 39 of 39 unchanged.

### Not done, and what the cases can't prove

- `ilmatrix-with-matrix` can't tell whether the constant terms come from the element's matrix or the element:
  they are the same numbers.
- No deliberate mistake was made in the host thread pool's use, and none in the three loss tests' order
  (a particle that fails two of them is lost either way).
- The lost particles' global coordinates and spin weren't put on the device; those runs use the CPU.
- `SLICE`'s 5 s that remain are three passes over the whole beam for each element (times, then two passes
  to sort), which could be one pass shared by the `SLICE` elements at the same place.
- `WATCH` in coordinate mode with a particle-ID range starts a table for the whole beam each time and then
  writes a small part of it; it is now the largest item in this deck and wasn't looked at.
- `ZLONGIT`, `ZTRANSVERSE` and `LRWAKE` have no device path, and `RFMODE`, `FRFMODE` and `FTRFMODE` ran on the
  host here, so the beam still returns to the host every turn.

See `results/2026-11-12_ring-collective` in GPU-Testing (`final-timing/`, `after-slice/`, `after-ilmatrix/`,
`cpu-feature`, `gpu-feature`, `gpu-routine`, `gpu-suite`, `verify-new`, `mut-*` and `mut2-*`, `mutation-trial/`,
`ring-collective.patch`, `source-final/`) and `results/2026-11-12_collective-ring-before`.

## Particle losses: the loss order in parallel, and magnets kept on the device with a losses file (November 13, 2026)

Losses used to cost gpu-elegant most of its gain. A line of 400 elements with 1,000,000 particles took 2.1 s
with no aperture, 10.7 s with five `MAXAMP`s that removed 10% of the beam, and 30.9 s with those and a losses
file (`results/2026-11-12_loss-cost` in GPU-Testing). The cost was per element call that lost anything, and it
grew with the size of the beam, not with the number lost. There were two causes, and both are fixed.

### The loss order, one particle at a time

elegant removes a lost particle by swapping the last active row into its slot and testing that row next. To
leave the particles in that order the device ran elegant's loop on one thread over the whole beam
(`gpuBuildCpuLossOrderDestinationKernel`): 80 ms for a million particles, where the element itself takes 0.3
to 0.45 ms. Where the loop leaves each particle follows from the count of survivors below it alone. With K
survivors of N:

- a survivor below K stays where it is;
- a survivor at or above K fills a hole: the j-th one counted from the top goes to the j-th lost particle's
  place;
- a lost particle above K moves down one place;
- a lost particle at or below K, the r-th lost one counted from 0, goes to the top (r = 0) or to the place just
  under the r-th survivor counted from the top.

The kernel now does every particle on its own, with a binary search in the survivor prefix for the two "r-th
one" lookups. The old loop is kept as `gpuCpuLossOrderLoopKernel`: `ELEGANT_GPU_CHECK_LOSS_ORDER=1` runs it
too and stops if any place differs, and `ELEGANT_GPU_ENABLE_PARALLEL_LOSS_ORDER=0` uses it alone. About 15
places use this builder (apertures, matrix elements, multipoles, `CSBEND`, `CCBEND`, exact correctors,
`ILMATRIX`, `BMXYZ`). The builder for losses in several rounds (`LGBEND`, kick maps, `SCRAPER`) already
visited only the lost particles and is unchanged.

### Magnets with a losses file

With a losses file or global loss coordinates, a magnet in which the device found even one lost particle was
tracked again on the CPU, the whole beam: 0.64 s for a `KQUAD` with a million particles against 0.012 s. The
reason was the lost rows. elegant writes more into them than the place and momentum of the loss: every
routine goes on to apply its exit fringe, end drift and misalignment to the lost rows too, with the momentum
where the momentum deviation was, and `LGBEND` applies the entrance fringe of each later segment. The device
kernels don't do that.

Now the device tracks and compacts as it does without a losses file, and elegant's own routine tracks only
the lost particles (`gpuMagnetLossRowsFromCpu()` in `gpu_stub.c`):

1. After the compaction kernel the old device buffer still holds every particle as it was before the magnet.
   The lost particles' rows are gathered from it in the order of their final places and copied to the host.
2. The CPU routine (`multipole_tracking2`, `multipole_tracking`, `fmultipole_tracking`,
   `track_through_csbend`, `track_through_ccbend`, `track_through_lgbend`, `trackKickMap`,
   `trackUndulatorKickMap`) runs on those rows alone. It loses them all and leaves each row as elegant writes
   it, global loss coordinates included.
3. The routine's swaps move the rows. Each row carries its place in the `lossPassIndex` slot, which nothing
   reads until a loss is recorded, and is put back there afterwards. The rows then go to the host's tail and
   the device's tail.
4. If the routine keeps a particle the device lost (a particle within rounding of an aperture), nothing has
   been changed and the whole element goes to the CPU as before. This did not happen in any test, so
   `ELEGANT_GPU_TEST_MAGNET_LOSS_DISAGREEMENT=1` makes the code act as if it had, for the two cases that
   guard this path.

`ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION` is on by default with a losses file too; `=0` gives the old
behavior. ISR with a loss, slice-by-slice tracking and `sigmaDelta2` still go to the CPU, as before.

One consequence: in those magnet calls the survivors used to come from the CPU's code and now come from the
device's, as they do in every call without a loss. Where the device's kernel rounds differently from elegant
(`LGBEND`, misaligned magnets, `MULT`), results that were identical to elegant's are now within rounding of
them, and the centroid and sigma sums at those elements are made on the device.

### Result

1,000,000 particles, one pass through 400 elements (100 `KQUAD`, 100 matrix quadrupoles, 200 drifts), one run
each, alone, with `ELEGANT_GPU_PROFILE_TIMING=1`:

| Deck | Lost | Before | Now |
| --- | ---: | ---: | ---: |
| No aperture | 0 | 2.1 s | 2.3 s |
| `MAXAMP`, no losses file | 100,666 | 10.7 s | 3.2 s |
| `MAXAMP`, losses file | 100,666 | 30.9 s | 3.3 s |
| The same with the apertures widened | 8,569 | 20.6 s | 3.0 s |
| Sticky `APCONTOUR`, losses file | 100,666 | 36.4 s | 3.7 s |

elegant takes 79.5 s on the `MAXAMP` deck with a file. In the feature cost table the three loss rows went from
6.5, 2.2 and 2.2 times elegant's speed to 24, 24 and 23 at 100,000 particles
(`history/2026-11-13_feature-timing-4` in GPU-Testing, in which every gpu-elegant row was measured again). `apcontour2` of the test set (1,000,000 particles, 65%
lost along 56 magnets, losses file with global coordinates) went from 198 s to 19.4 s; elegant takes 195 s.

With `ELEGANT_GPU_ENABLE_PARALLEL_LOSS_ORDER=0` the same binary takes 11.0 s on the `MAXAMP` deck with a
file (matrix quadrupoles 1.62 s, drifts 3.29 s, `KQUAD` 5.05 s, against 0.06, 0.09 and 2.18 s), so the
one-thread walk itself was the 80 ms.

### Checks

- **Losses files against elegant.** Byte-identical: the `MAXAMP` deck with a file (100,666 rows),
  `apcontour2` (651,988 rows with global coordinates; its acceptance and output files too), and 13 feature
  cases with a losses file that reach the new code: misaligned `KQUAD`, `KSEXT` and `CSBEND` with linear and
  exact misalignment, multipole files, `MULT` with global coordinates, four `CCBEND` cases, `CSBEND` losses
  inside the bend, `GKICKMAP` with global coordinates, `KQUAD`, `KSEXT`, `KOCT` and `KQUSE` with every
  particle lost in one magnet, `FMULT`, `UKICKMAP`. The CPU routine never kept a particle the device lost. Over the whole feature set, 43
  of the 49 losses files are byte-identical to elegant's. Three of the other six are the `LGBEND` and
  global-coordinate cases below, and three are older cases that this change left as they were
  (`aperture-apcontour-losses`, `magnet-exact-corrector-global-loss`, `resident-floor-ring-output`).
- **`LGBEND`** (`magnet-lgbend-losses-file`, three passes, losses in all nine bend calls): elegant's
  particles, passes and row order in all 414 rows. The 10 rows lost before or in the first bend are
  identical; later rows differ by 1e-15 relative after one bend, growing to 1e-13 after eight, because the
  survivors now go through the device's `LGBEND`. With the old rule every one of these calls ran on the CPU.
- **Global coordinates** (`magnet-malign-losses-file-global`): all 259 rows lost inside magnets are identical
  to elegant's, global coordinates included. Global `X` is one unit in the last place off in 97 of the 241
  rows lost inside drifts. Those rows are written by the aperture code, and the old rule gives the same
  file, so this is not from this change. Not looked into.
- **Feature set, 422 cases** (414 before, eight new, see "Magnet losses with a losses file" in
  `feature-set/README.md`): 421 within rounding of elegant, 85 identical, 1 flagged (`scmult-zero-length`, as
  before). No older case changed its class. Against the GPU reference of November 12, four older cases with a
  losses file moved, by 1e-16 relative in their centroid, sigma and final files (`magnet-malign-loss`,
  `magnet-malign-exact-loss`, `magnet-mult-global-loss-fallback`, `magnet-multipole-files-loss`): their
  survivors now come from the device. No losses file changed.
- **Routine list, 424 tests, forced settings:** 421 unchanged against the GPU reference. `acceptance1`, `2`
  and `3` have a losses file and losses in `KQUAD`s; their centroid and sigma files move by 1e-16 relative
  at those elements (the comparison flags `run.cen` because the differences are too small for its measure),
  and their losses and acceptance files are unchanged.
- **CSR list, 33 tests:** unchanged. **GPU test set, 39 tests:** unchanged.
- **The whole-element fallback** (`magnet-malign-loss-cpu-keeps`, `magnet-lgbend-losses-file-cpu-keeps`,
  with the test switch): 29 and 9 magnet calls go to the CPU after the device has compacted. Every output
  file has the data of a run with `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0`, and the losses, acceptance
  and output files are byte-identical to elegant's. Under gpu-Pelegant with 4 ranks
  `magnet-malign-loss-cpu-keeps` gives Pelegant's losses data.
- **`GPU_VERIFY`:** the verify binary completes all 422 feature cases with no stops.
- **gpu-Pelegant, 4 ranks, 11 loss cases** against Pelegant built with `-Og`: the losses data are identical in
  nine. `magnet-malign-losses-file-global` has the same last-place difference in global `X` and `Z`, and
  `aperture-apcontour-losses`, which doesn't reach the new code, differs by one unit in the last place in
  rows lost on the second pass.
- **The loss order.** A Python model of the rule against elegant's loop: every pattern of up to 10 particles
  and 4,000 random ones up to 400 (`loss_order_model.py`, kept in `results/2026-11-13_loss-fix`).
  `ELEGANT_GPU_CHECK_LOSS_ORDER=1` passes on the `MAXAMP` deck with a file, on the feature cases and on the
  424 tests of the routine list.
- **Deliberate mistakes, five in a trial build, on nine loss cases** (`results/2026-11-13_loss-fix/mutation`):
  - a survivor from the tail put in the wrong hole: the order check stops 7 cases, and without the check
    the files of the same 7 differ (the other two, `LGBEND` and `UKICKMAP`, use the builder for rounds);
  - lost particles above the final count not moved down: the order check stops 7 cases, and 5 losses files
    differ;
  - lost rows left in the order the CPU routine's swaps give: the losses files of all 6 magnet cases with a
    file differ;
  - lost rows not copied to the host: the same 6 differ;
  - the kick counter left as the CPU routine changed it: no file differs. `multipoleKicksDone` is only
    printed, and these runs don't print it, so restoring it is not covered by a test.
  The final binaries were rebuilt from the restored sources and give the first build's outputs.

### Not done

- **A magnet call that loses a particle is still tracked three times on the device**: once to find the
  losses (and the beam is put back), once to mark the survivors, once to pack the beam. Such a call takes
  3.3 times a call with no loss (41.6 ms against 12.7 ms for a `KQUAD` and a million particles). This is
  why the loss decks are 25 times faster than elegant and not 50, and it is 17.8 s of `apcontour2`'s 19.4 s.
- The `APCONTOUR` element itself still goes to the CPU when it loses particles and global loss coordinates
  are wanted (26 calls in `apcontour2`).
- ISR with a loss, slice-by-slice tracking and `sigmaDelta2` go to the CPU as before.
- No real deck here has a particle the device loses and the CPU keeps; that fallback runs only under the
  test switch.

Results are in `results/2026-11-13_loss-fix` in GPU-Testing: `cpu-feature-422` and `gpu-feature-422` (the
feature references), `gpu-routine-final` (the routine reference), `gpu-csr`, `gpu-suite`,
`verify-feature-422`, `pel-cpu` and `pel-gpu-final`, `apcontour2`, the five loss decks, `mutation/`,
`source-final/` and `binaries-final.sha256`. The mistakes, the timings and the CSR, suite and order-check runs
were made with the build before the test switch was added; the feature set, the routine list, the verify run
and the gpu-Pelegant cases were repeated with the final binaries.

## Wakes with a set number of bins: two narrow sums widened (November 13, 2026)

The feature cost table showed `WAKE` and `TRWAKE` at 0.6 to 0.7 times elegant's speed, and `RFCW` at 1.0. The
wake itself was not the cost. Two beam-wide reductions were, each on one block of 64 threads:

- **The mean time.** A wake with `N_BINS` set centers its bins on the bunch's mean time. With output in the
  CPU's order (`output`, `final` or a `WATCH`) the mean is added in particle order
  (`gpuWakeOrderedTimeMean()`), which was already fast. Without such output it came from
  `gpuCudaBeamSums`, which adds every sum and product of the beam to get one of them: 41 ms for a million
  particles at every wake call. It now comes from `gpuCudaTimeSpan`, the kernel that already found the least
  and greatest time with one particle per thread, which adds the times pairwise inside each block of 256 and
  then over the blocks: 0.1 ms. `gpuWakeTimeSums()` in `gpu_stub.c` is the one place changed.
  `ELEGANT_GPU_ENABLE_WIDE_WAKE_TIME_SUM=0` uses the old kernels. `WAKE`, `TRWAKE`, `CWAKE` and both wakes
  of an `RFCW` share it.
- **The range of bunch numbers.** In bunched beam mode every wake call asks for the least and greatest bunch
  number in the beam, twice (`gpuCudaLongMinMax`, from `gpuBunchedWakePlan()`): 2 ms each for a million
  particles, which was most of what a wake with automatic bins cost. It now runs one particle per thread:
  0.06 ms. These are integer comparisons and a count, so the result is the same. `TRFMODE`, `FTRFMODE` and
  `SCMULT` use the same kernel.

Which cases were slow: only a wake with `N_BINS` set and no output in the CPU's order, which is what the
benchmark deck was. With an output file, or with automatic bins, a `WAKE` call already took 0.42 to 0.67 ms
for 100,000 particles against elegant's 2.9 ms.

### Result

Per call, from `show_element_timing` with `ELEGANT_GPU_PROFILE_TIMING=1`, in the feature-timing decks
(`N_BINS=1024`, no output file):

| | elegant | Before, 100,000 | Now | Before, 1,000,000 | Now |
| --- | ---: | ---: | ---: | ---: | ---: |
| `WAKE` | 29 ns a particle | 4.35 ms | 0.29 ms | 45.8 ms | 1.07 ms |
| `TRWAKE` | 36 ns a particle | 4.77 ms | 0.69 ms | 50 ms | 4.2 ms |
| `RFCW` with both wakes | 98 ns a particle | 9.24 ms | 1.22 ms | 97 ms | 9.6 ms |

That is 10 and 27 times elegant's speed for `WAKE`, 5 and 9 for `TRWAKE`, 8 and 10 for `RFCW`. In the GPU test
set `cwake-heavy` went from 0.74 s to 0.40 s and `cwake-multibunch-heavy` from 0.81 s to 0.43 s.

### Checks

- **Old sum against new, and against elegant.** Three decks of 100,000 particles with centroid and sigma
  files and no output in the CPU's order (`WAKE`, `TRWAKE`, `RFCW`, 200 wake calls each): the new sum and
  the old one give the same files to the last bit, and both differ from elegant's as before (beam sizes and
  emittances by 6e-12 to 1e-11 relative).
- **Feature set, 426 cases** (four new): 425 within rounding of elegant, 85 identical, 1 flagged
  (`scmult-zero-length`, as before). The 422 older cases give the reference's files. No older case had a wake
  with `N_BINS` set and no output in the CPU's order, so nothing tested this sum before; the four new cases
  do (`wake-trwake-fixed-bins-no-output`, `rfcw-fixed-wake-bins-no-output`,
  `wake-bunched-fixed-bins-no-output`, `cwake-bunched-fixed-bins-no-output`), for one bunch, a bunch picked by
  its number, and bunches stored in order.
- **Routine list, 424 tests, forced settings:** all unchanged against the GPU reference. **GPU test set, 39
  tests:** 38 unchanged; `cwake-multibunch-heavy` has a `CWAKE` with `N_BINS` set and no such output, and
  moves by 1e-12 relative and less in its centroid and sigma files.
- **`GPU_VERIFY`:** the verify binary completes all 426 feature cases with no stops.
- **gpu-Pelegant, 4 ranks, six wake cases:** identical to Pelegant built with `-Og` (wakes run on the CPU
  there).
- **Deliberate mistakes, four in a trial build** (`results/2026-11-13_wake-sums/mutation`), on ten wake
  cases: other bunches' times added to the sum (caught by the bunch-number case); the first 256 particles
  left out of the sum, and no sum returned (each caught by all four new cases, and by none of the six older
  ones, which use the ordered mean); the largest bunch number found without the last step of the reduction
  (two bunched cases stop with "beam longer than the wake", two more change). The binaries were rebuilt from
  the restored sources and give the first build's files.

### Not done

- `TRWAKE` is now 4.2 ms for a million particles and `RFCW` 9.6 ms, where `WAKE` is 1.07 ms. What they
  spend it on was not looked into.
- With output in the CPU's order a `WAKE` with `N_BINS` set takes 0.67 ms for 100,000 particles against 0.42
  ms with automatic bins: the ordered mean and the ordered bin sums. Not changed.

Results are in `results/2026-11-13_wake-sums` in GPU-Testing: `cpu-feature-426` and `gpu-feature-426` (the
feature references), `gpu-suite` (the GPU test set reference), `gpu-routine`, `verify-feature`, `pel-cpu`
and `pel-gpu`, the timing and comparison decks, and `mutation/`. The routine reference is still
`results/2026-11-13_loss-fix/gpu-routine-final`.

## Magnets that lose particles tracked once, and `TRFMODE` bin sums (November 13, 2026)

Two more changes the same day, checked with the lighter set agreed with the user for the rest of this list:
the feature set, each change's own decks against elegant, and the routine list once over both. The
deliberate-mistake build, the CSR list, the verify and gpu-Pelegant runs and the full feature table are
left for the end of the list.

### One pass for a magnet call that loses particles

A magnet call in which the device found a lost particle was tracked three times: once by the checked
kernel, which then put the beam back from its backup, once to mark the survivors, and once to pack the
beam. Such a call took 3.3 times a call with no loss. Now the checked kernels of the kick multipoles
(`KQUAD`, `KSEXT`, `KOCT`, `KQUSE`, `DQCOR`), `MULT`, `FMULT`, `CSBEND` and `CCBEND` also note which
particles they kept (`gpuMagnetSurvivorFlags()`), and on a loss the rows are packed as tracked
(`gpuCompactTrackedMagnet()`, `gpuCudaCompactTrackedRows`). The kernel and the order are the same, so the
results are the same. With a losses file the rows from before the magnet, which elegant's routine needs for
the lost particles, come from the checked kernel's backup. If that routine keeps a particle the device
lost, the beam is put back from the backup and the element goes to the CPU as before.
`ELEGANT_GPU_ENABLE_ONE_PASS_MAGNET_LOSS=0` gives the three passes back. ISR with a loss is unchanged.
`LGBEND` and the kick maps, which remove losses in rounds, still take three passes.

| 1,000,000 particles | Before | Now |
| --- | ---: | ---: |
| `MAXAMP` loss deck, no losses file | 3.2 s | 2.5 s |
| The same with a losses file | 3.3 s | 2.4 s |
| Sticky `APCONTOUR`, losses file | 3.7 s | 2.7 s |
| The same line with no aperture | 2.3 s | 2.3 s |
| `apcontour2` (elegant: 195 s) | 19.4 s | 8.9 s |

A `KQUAD` call that loses particles now takes about 1.3 times a call that doesn't.

Checks: the losses files of the two loss decks and of `apcontour2` (with its acceptance and output files)
are byte-identical to elegant's; the feature set's 426 cases give the reference's files, and 45 of its 51
losses files are byte-identical to elegant's, as before; the two `magnet-*-cpu-keeps` cases run the new
put-back from the backup.

### `TRFMODE` and `FTRFMODE` bin sums

`TRFMODE` adds each bin's `x` and `y` in particle order, so its result is the CPU's. One thread per bin
did that, reading each particle's row through the sorted index: 0.5 us a particle, so the cost followed the
fullest bin. A million particles in a bunch 20 bins long took 28 ms a call (elegant: 20 ms); a bunch 600
bins long took 4.9 ms. Now each bin has a block of 256 threads: they read a stretch of the bin's terms into
shared memory together, and one thread adds them in the same order (`gpuTrfmodeSortedBinSumsKernel`). The
sums are the same to the last bit.

| Per call | elegant | Before, 100,000 | Now | Before, 1,000,000 | Now |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TRFMODE` | 2.0 ms and 20 ms | 1.89 ms | 0.78 ms | 28.0 ms | 6.8 ms |
| `FTRFMODE` (168 modes) | 2.4 ms and 24 ms | 2.2 ms | 1.11 ms | 31 ms | 7.1 ms |

Of the 6.8 ms about 3.6 ms is kernels and 3 ms the mean time, which the host adds in particle order.
Checks: the feature set and all 39 tests of the GPU test set give the references' files;
`trfmode-resident-heavy` went from 1.79 s to 0.86 s and `ftrfmode-resident-heavy` from 1.77 s to 0.89 s.
Not done: in bunched mode with many bunches these elements still run on the host, which is where
`feedback6` and the ring deck spend their time.

### `IBSCATTER`: measured, not changed

The feature table's 0.4 times elegant's speed was the noise of a deck with one call. With gpu-elegant's
timers, one call for a million particles is 87 ms in elegant's routine plus 11 ms to bring the beam to the
host and back (elegant: 75 ms). So there is nothing to gain from the copies, and a gain needs the element
on the device: time slices, each slice's sums in the CPU's order, the growth rates on the host, and kicks
with `random_2`'s numbers in the CPU's order. Not started.

### Routine list over both changes

All 424 tests give the GPU reference's files (`results/2026-11-13_one-pass-loss/gpu-routine`), and the verify
binary and gpu-Pelegant build without warnings. Sources after each change are kept in
`results/2026-11-13_one-pass-loss/source-after` and `results/2026-11-13_trfmode/source-after`. The
references are unchanged: `results/2026-11-13_wake-sums` for the feature set and the GPU test set,
`results/2026-11-13_loss-fix/gpu-routine-final` for the routine list.

## CSR with output: `CSRCSBEND` on the device gives elegant's particles (November 13, 2026)

Until now a CSR bend went to the CPU whenever the run wrote `output`, `final` or a `WATCH` file, because
the device's bend did not give elegant's particles to the last bit, and CSR's binning turns a last-bit
difference in one particle's time into a visible one in the whole beam. The user chose to make the device
exact first and only then let it run with output. It now does, by default, and the floor of 1,024 bins is
gone. Checked with the lighter set; the end-of-list checks are still owed.

### What differed, and the fixes

| Cause | Where | Fix |
| --- | --- | --- |
| The wake kernel grouped two products its own way: `sqr(denom[1])` is a function call on the CPU, so the square is formed first, and the two factors `CSRConstant*length/nSlices` and `-CSRConstant*length/nSlices/slippageLength13` are formed before they multiply the sums | `gpuCsrCsbendWakeKernel` | The square is formed first, and `csbend.c` passes the two factors as it computes them (`t1Factor`, `t2Factor`) |
| The edge kicks took the device's `tan()` of the edge angle, which is not glibc's in the last bit (it differs at 0.08 rad) | `gpuCsrCsbendEnterSimpleCheckedKernel`, `gpuCsrCsbendFinalizeSimpleCheckedKernel` | The host passes `tan(e1)` and `tan(e2)` |
| With `HGAP*FINT` not zero the edge takes `tan()` of an angle that depends on each particle's momentum, which the host can't pass | The same kernels | With output in the CPU's order that edge runs on the CPU and the body stays on the device (`gpuCsrCsbendDeviceEntrySupported`, `gpu_track_csbend_csr_enter_simple`, `gpu_track_csbend_csr_finalize_simple`) |
| With `&slice_analysis`, elegant converts the beam out of bend coordinates and back around each output inside the bend (`convertFromCSBendCoords`, `convertToCSBendCoords`). Time is divided by c and multiplied by it again, which does not return the same number, so elegant's beam changes by rounding. The device wrote the output from a copy and kept its own rows | `csbend.c`, after `performSliceAnalysisOutput` | The device repeats the conversion pair on its rows (`gpu_csr_csbend_output_round_trip`, `gpuCsrCsbendOutputRoundTripKernel`), tilt included |

The rule that sent CSR to the CPU with ordered output is off (`gpuCsrCpuForOrderedOutput()`);
`ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT=1` brings it back.

The floor of 1,024 bins (`ELEGANT_GPU_MIN_CSR_BINS`) dated from when only the wake was on the device. With
the whole bend there the number of bins says nothing about whether the device pays, so the default is now
no floor; `ELEGANT_GPU_MIN_CSR_BINS=1024` brings it back. The floor never applied in required mode.

### Result

A four-bend chicane at 150 MeV, 100 kicks a bend, with an output file, one thread, `sm_120`:

| Deck | elegant | gpu-elegant before | Now |
| --- | ---: | ---: | ---: |
| 1,000,000 particles, 3,072 bins, whole tracking step | 47.4 s | 47.6 s | 3.5 s |
| The four bends alone | 46.1 s | 46.3 s | 2.18 s |
| 1,000,000 particles, 512 bins, the four bends | not run | 46.2 s | 2.19 s |
| 20,000 particles, 512 bins, the four bends | 0.87 s | 0.87 s | 0.076 s |
| 10,000 particles, 512 bins, the four bends | 0.43 s | not run | 0.053 s |

The particle files are byte-identical to elegant's in every row. The four `CSRDRIFT` elements of the deck
still run on the CPU (0.34 s of the 3.5 s).

On the 33 CSR tests of `elegantTestSet`, by default: 24 now run their CSR bends on the device (3 with the
floor), and the sum of the tests' run times, four at a time, went from 139 s to 115 s (elegant: 247 s). `csrSegment1`
3.8 s to 0.6 s, `rfdf3` 12.8 s to 6.1 s, `bunchComp1` 7.6 s to 5.0 s. Six that looked slower in that run
are not when timed alone.

### Checks

- **Probes** (`results/2026-11-13_csr-exact/csr_probe.py`): 78 short lines at 150 MeV with 20,000 particles
  and an output file, each run in elegant and in gpu-elegant: bend options one at a time (edge angles, edge
  order, `K1`, `K2`, `USE_BN`, order 4, `N_KICKS=100`, steady state, smoothing, cutoffs, `FSE`, `TILT`,
  `ETILT`, `DX`, `SYNCH_RAD`, `CSR=0`, 4 to 3,072 bins), chicanes with each `CSRDRIFT` model, and
  `&slice_analysis` with and without tilt. By default all 78 particle files are byte-identical to
  elegant's, and so are the slice analysis files. With the CSR opt-ins set, 69 of 69 (run before the last
  nine were added).
- **The 33 CSR tests, by default**: every output file but the per-element statistics matches elegant. With
  the statistics sums on the host (`ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS=0`) 32 tests match elegant in every
  file; the other is `rfdf3`'s `.fin`, which differed the same way before and has nothing to do with CSR.
  With the default sums, `.sig` and `.cen` differ from elegant in the same 22 tests as before.
- **The 33 CSR tests, forced** (`gpu-forced-csr.env`): before, 4 to 22 particle, wake and `WATCH` files a
  test differed from elegant in 19 tests, by up to order 1 in small quantities. Now none does in 18 of
  them; `TTF2` differs only in parameter-mode `WATCH` files and statistics, which are device sums, and it
  runs `LSC` on the device. This run is the new CSR list reference.
- **Feature set**: 428 cases, two new: `csr-edge-angles-output` and `csr-slice-analysis-tilt`. 427 within
  the envelope, 85 identical, `scmult-zero-length` flagged as before. Of the 426 older cases, 15 CSR cases
  moved, all toward elegant: four differed from elegant in their particle file and none does now. The other
  411 give the reference's files.
- `bunchComp1` was the deck that showed the slice analysis difference: 11 of its files differed from the
  stored run, by up to 0.76 of `x`. Now only its `.sig` and `.cen` do, at 1e-16, from sums that moved to the
  device with the beam.

### Not done

- `CSRDRIFT` with CSR on runs on the CPU. The device has only the drift with CSR off. (Done later the
  same day: next section.)
- A bend with `HGAP*FINT` not zero has its two edges on the CPU when output is in the CPU's order: two
  copies of the beam a bend.
- Not run for this change yet: the deliberate-mistake build, the verify build and gpu-Pelegant (both need
  rebuilding), the routine list, the GPU test set, and the feature table.

References: feature set `results/2026-11-13_csr-exact/cpu-feature-428` and `gpu-feature-428`; CSR list
`results/2026-11-13_csr-exact/gpu-csr-forced`. Sources before and after are in `source-before` and
`source-after-bends` there.

## `CSRDRIFT` with CSR on the device (November 13, 2026)

With the bends on the device, the CSR drifts between them were what was left on the CPU: the beam came
to the host for each one. Now a `CSRDRIFT` with `CSR=1` stays on the device, in every drift model, and
gives elegant's particles to the last bit. Checked with the lighter set.

### How

`track_through_driftCSR()` and `track_through_driftCSR_Stupakov()` in `csbend.c` run as before and keep
everything that works on the wake's bins: propagating and attenuating the stored wake, the spread and
Saldin factors, and for Stupakov's model the filters, the smoothing, the derivative and the wake itself.
Their loops over particles go to the device, written in the same arithmetic:

| Step | Device | Notes |
| --- | --- | --- |
| Drift each kick and turn `s` into `c*t` | `gpu_csr_drift_advance`, `gpuCsrDriftAdvanceKernel` | Also the half drift at the end and, for Stupakov, the drift back to where the last one ended |
| Histogram of `c*t` (Stupakov) | The bend's `gpu_prepare_csbend_csr_histogram_device` and `gpu_compute_csbend_csr_histogram_device` | Counts, so exact; the bins come to the host |
| Kick from the wake, then `c*t` back to `s` | `gpu_csr_drift_kick`, `gpuCsrDriftKickKernel` | The host's `dGamma` goes to the device each kick |
| Particles inside the wake's bins | `gpu_csr_drift_count_in_range` | elegant prints a line for a particle outside them (Stupakov), so if any is outside the beam goes to the host and elegant's loop runs |

`gpu_track_through_driftCSR` calls `track_through_driftCSR_cuda_resident_entry` when there is a wake to
apply. A drift with `LSC_BINS` stays on the CPU. `ELEGANT_GPU_ENABLE_CSR_DRIFT_ELEMENT=0` sends CSR drifts
back to the CPU. In a verify build each of the steps is checked against the CPU's arithmetic where it is
made; the whole element is not run a second time, because that would change the stored wake twice.

Not built with MPI: gpu-Pelegant keeps CSR drifts with CSR on the CPU, as before.

### Result

The four-bend chicane at 1,000,000 particles, 3,072 bins, with an output file, one thread, `sm_120`:

| Deck | elegant | Drifts on the CPU | Now |
| --- | ---: | ---: | ---: |
| Four `CSRDRIFT`, 4 kicks each: the drifts | 0.33 s | 0.34 s | 0.078 s |
| The whole tracking step | 47.4 s | 3.5 s | 3.17 s |
| Four `CSRDRIFT`, 50 kicks each (`DZ=0.01`): the drifts | 3.85 s | 3.82 s | 0.98 s |
| The whole tracking step | 51.0 s | 7.0 s | 4.1 s |

The particle files are byte-identical to elegant's. No copy of the beam is left in the deck except for
the output. What remains of a drift kick, about 4.9 ms, is Stupakov's wake on the host, which costs the
same for any number of particles.

For the other drift models the bend before the drift still brings the beam to the host once, because
elegant prepares bunch lengths and beam sizes for them from the particles at the bend's end.

### Checks

- **Probes**: 100 lines now, 21 of them new for the drift's options (Stupakov with `DZ`, one kick,
  `LINEARIZE`, an output file, slice analysis, `LSC_BINS`, the bend's filters, a quadrupole or another
  drift before it, a drift before any bend; the plain model with and without `LINEARIZE`, attenuation,
  overtaking length with its multiplier and the 68-percentile bunch length, three spread modes, Saldin's
  model with more points and after a quadrupole). By default 98 of 98 give elegant's particle file byte
  for byte, and the slice analysis and Stupakov output files too. The other two use
  `WAVELENGTH_MODE="peak-to-peak"`, which has no defined answer in elegant (below).
- **Feature set**: 430 cases, two new: `csr-drift-stupakov-output` and
  `csr-drift-stupakov-linearize-slice`. 429 within the envelope, 85 identical, `scmult-zero-length`
  flagged. Of the 428 older cases 423 give the reference's files and five CSR cases changed in `.cen`,
  `.sig` and `.fin` only, by rounding: sums that are now made on the device because the beam is there.
- **The 33 CSR tests, by default with host sums**: 32 match elegant in every file and `rfdf3` differs in
  `.fin` as before. No CSR drift brings the beam to the host.
- **The 33 CSR tests, forced**: against the reference of the section above, 20 tests changed, in `.sig`,
  `.cen` and `TTF2`'s parameter-mode `WATCH` files only. Against elegant, no particle or wake file differs
  in any test but `TTF2`, as before. This run is the new CSR list reference.

### A CPU bug found on the way, not fixed

`CSRDRIFT` with `SPREAD=1` and `WAVELENGTH_MODE="peak-to-peak"` uses a number that is never set. At the
end of each bend `csbend.c` takes the peak-to-peak wavelength from `csrWake.dGamma` and `csrWake.bins`,
but `reset_driftCSR()` cleared both at the start of that bend and they are filled in a few lines later,
so `index_min_max()` returns without setting `imin` and `imax`. elegant and gpu-elegant with
`ELEGANT_GPU_MODE=off` give different particles for such a deck. The fix would be to pass `dGamma` and
`nBins`. Set aside as a CPU-only bug.

### Not done

- Stupakov's wake itself is computed on the host each kick.
- A drift with `LSC_BINS`, and CSR drifts in gpu-Pelegant, stay on the CPU.
- Not run yet for the CSR changes: the deliberate-mistake build, the verify build and gpu-Pelegant, the
  routine list, the GPU test set, and the feature table.

References: feature set `results/2026-11-13_csr-exact/cpu-feature-430` and `gpu-feature-430`; CSR list
`results/2026-11-13_csr-exact/gpu-csr-forced-drift`. Sources: `source-after-drift` there.

## End-of-list checks for November 13, 2026

The checks left for the end under the lighter plan, run once over the day's changes after the loss
fixes: one pass for a magnet that loses particles, the `TRFMODE` bin sums, CSR bends with output, and CSR
drifts on the device. All on `sm_120`, one thread.

| Check | Result |
| --- | --- |
| Verify build, feature set (430 cases, `ELEGANT_GPU_VERIFY=1`) | The first run stopped in six CSR drift cases: after a CSR drift on the device the host rows that a verify build compares against were half a kick behind, because that element is not run a second time on the host. Fixed in the verify build only (`csbend.c`, a read-only copy at the drift's end). Then all 430 complete with no stop; 92 advance, 28 drift and 92 kick checks of the CSR drift pass. Outputs against elegant: 398 identical, 31 within the envelope, `scmult-zero-length` flagged |
| gpu-Pelegant, 47 tests on 4 ranks | All 47 give the reference's files (`results/2026-11-11_csbend-drift/gpu-pel`); 46 match CPU Pelegant and `expandHamiltonian6` differs by rounding, as before |
| Routine list, 424 tests, forced | 404 give the reference's files. The other 20 are CSR decks, which now keep their CSR elements on the device: they changed in `.sig` and `.cen`, and `TTF2` in seven parameter-mode `WATCH` files, all sums that moved to the device with the beam. Against elegant no particle file differs in those 20 that didn't before. New routine reference |
| GPU test set, 39 tests | 38 give the reference's files; `csr-heavy` changed in `.cen` and `.sig` by rounding, and went from 0.41 s to 0.26 s. New reference |
| Deliberate mistakes | Below |
| Feature set with the restored build | 429 of 430 give the files from before the mistakes; the other is `csr-edge-angles-output`, changed on purpose (below), which matches elegant's particle file |
| Feature cost table, every gpu-elegant row redone (544 runs, 25 minutes) | `history/2026-11-13_feature-timing-6`, since replaced by `-7`: the same runs, with the cells that were within the start-up scatter measured again with more passes (see `GPU-Testing/TIMING-PROCEDURE.md`, section 10), so the rows named here are unchanged except `csrcsbend`, now 8.8x, 22x and 20x. Table `-8` takes gpu-elegant's times from wall-clock slopes instead of tracking times, which miss queued work; there `csrcsbend` is 8.8x, 22x and 21x, `csrdrift` 1.4x, 4.3x and 3.8x, `trfmode` 3.0x and `ftrfmode` 3.4x at 1,000,000. Table `-9` is `-8` with the five rows that the short-run rule touches measured again (last section). In table 6: `csrcsbend` 1.0x to 11x, 24x and 21x at 10,000, 100,000 and 1,000,000 particles; `csrdrift` 1.0x to 1.5x, 4.7x and 4.1x; `trfmode` 0.7x to 2.6x and `ftrfmode` 0.8x to 2.9x at 1,000,000; the three loss rows a third faster. No other row moved by more than the method's noise. `watch-centroid` reads 3 times slower, which is start-up scatter in a deck with two calls: run by hand it takes 0.25 to 0.30 s from run to run, and the element timer gives 6.5 ms per call |

### Deliberate mistakes

Two builds of gpu-elegant with mistakes put in on purpose, each in a path changed today. Mistake 2 and
mistake 9 are in kernels and always on; the others were switched on one at a time by an environment
variable that exists only in those builds. The diffs are in `results/2026-11-13_csr-exact/mutation`.

| # | Mistake | Caught by |
| --- | --- | --- |
| 1 | The rows packed after a magnet lost particles are not made the beam (`gpuCompactTrackedMagnet`) | `magnet-ccbend-tilt-loss`, `magnet-csbend-internal-loss`, `magnet-fmult-losses-file`, `magnet-kquad-losses-file-all-lost` |
| 2 | `TRFMODE` adds each bin's terms in the other order | `trfmode-resident-heavy` and `ftrfmode-resident-heavy` of the GPU test set. The feature set has no `TRFMODE` case and did not change |
| 3 | `csbend.c` groups the wake's factor as `CSRConstant*(length/nSlices)` | Nothing at first: with 16 slices, and with 10, both groupings give the same number. `csr-edge-angles-output` now has bends of 12 and 11 slices and catches it |
| 4 | The host's `tan(e1)` is one unit in the last place off | `csr-edge-angles-output`, `csr-drift-stupakov-output` |
| 5 | The conversion pair around slice analysis is not repeated on the device | `csr-slice-analysis-tilt` |
| 6 | Each CSR drift step is one unit in the last place long | `csr-drift-plain-wake`, `csr-drift-stupakov-output`, `csr-saldin54` |
| 7 | The wake's first bin is one unit in the last place off in the CSR drift kick | The same three |
| 8 | One particle too few counted inside the wake's bins | `csr-drift-stupakov-output`: the beam goes to the host for the kick, which shows in the statistics files. In the other drift models the count only feeds a printed message, which appears 16 times in their logs and never in the reference's |
| 9 | The wake kernel multiplies `0.3*denom[1]*denom[1]` left to right, as before today's fix | `csr-bins-512`, `csr-short-bunch`, `csr-maxamp-loss`, `csr-slice-analysis-tilt`, `csr-drift-stupakov-output` |

With no switch set the first mistake build gave the reference's files in all 430 feature cases, so the
switched mistakes did nothing until switched on.

References after these checks: feature set `results/2026-11-13_csr-exact/cpu-feature-final` and
`gpu-feature-final`; routine list `results/2026-11-13_csr-exact/gpu-routine`; GPU test set
`results/2026-11-13_csr-exact/gpu-suite`; CSR list `results/2026-11-13_csr-exact/gpu-csr-forced-drift`;
gpu-Pelegant unchanged. Final sources are in `results/2026-11-13_csr-exact/source-final`, and the binaries'
checksums in `binaries-final.sha256` there.

## Short runs between CPU-only elements judged by cost (November 13, 2026)

Asked for by the user after the question why `ZLONGIT` stays on the CPU with a large beam. `ZLONGIT` has
no device code (no `GPU_SUPPORT` in `track_data.c`; the same for `ZTRANSVERSE`), and this section does
not change that. What it changes is what happens around such an element.

### What was wrong

When the host holds the beam, `gpuShortGpuIslandAhead()` in `gpu_stub.c` looks at the elements the device
could take before the next one it can't. If they count 4 or less (a matrix element 1, a magnet 2) the run
stayed on the CPU, to save the beam's trip to the device and back. The count took no account of the
beam. In the feature-table deck with a `ZLONGIT` at every third element, each run is drift, quadrupole,
drift, so with 100,000 particles the whole deck ran at elegant's speed: 4.85 s, where the matrix
elements take 6.4 ms each on the CPU and the trip about 1 ms.

### What was measured

`results/2026-11-13_island-rule/island_probe.py` and `island_probe2.py` track a line of (`CENTER`, run)
cells, `CENTER` being an element the device doesn't take, with the rule as built and with
`ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS=0`, and take the cost of one run from the wall-clock slope between
two numbers of passes, less the same for a line of `CENTER`s alone. In ns a particle, CPU / device with
its two copies:

| Run | 10,000 particles | 100,000 | 1,000,000 |
| --- | ---: | ---: | ---: |
| One drift, first order | 16.6 / 12.5 | 16.2 / 9.0 | 18.4 / 10.1 |
| Drift, quadrupole, drift, first order | 49.5 / 13.4 | 50.5 / 9.6 | 45.7 / 10.7 |
| One drift, second order | 62.2 / 13.0 | 60.4 / 9.6 | 59.9 / 7.3 |
| Drift, quadrupole, drift, third order | 301 / 15.6 | 304 / 11.5 | 301 / 12.1 |
| One `KQUAD`, 8 slices | 653 / 27.4 | 655 / 21.2 | 630 / 23.4 |
| Drift, `KQUAD`, drift | 776 / 30.2 | 691 / 22.0 | 757 / 26.0 |
| One `CSBEND`, 8 slices | 891 / 30.1 | 891 / 23.1 | 857 / 23.3 |

From 10,000 particles every run the rule kept on the CPU was cheaper on the device, by 1.3 times for a
lone first-order drift and 37 times for a lone `CSBEND`. The trip costs 7 to 13 ns a particle.

### What changed

A run that is short by its count now goes to the device all the same when its estimated cost on the CPU
is `GPU_SHORT_ISLAND_MARGIN` (1.4) times its cost on the device or more. The estimate for each element is
that of `gpuSmallBeamElementCost()`, which the small-beam choice already uses (15 ns a particle and 10 us
for a matrix element, so many ns and us a kick or slice for the magnets, more of both when the run writes
centroid or sigma output), plus 45 ns a particle for a matrix of second order or higher, plus two copies
of the beam at 20 us and 10 ns a particle each. The copies are priced at about twice what was measured
and the CPU numbers are the low ends, so the estimate errs toward the CPU: a lone first-order matrix
stays there. The estimate uses the lattice and the particle count, never a clock.

- `gpuChooseSmallBeamThresholds()` walks the lattice with the same function, so the small-beam estimate
  counts the runs tracking will take.
- An element's matrix is made when it is first tracked, and its order isn't known before. Until then it
  is priced as first order, so a lone second-order matrix stays on the CPU on the first pass and goes
  from the second.
- `ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0` goes back to the count alone.
  `ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS=0` still sends every run to the device.
- gpu-Pelegant keeps the count alone: the timings are the serial build's.
- `ELEGANT_GPU_VERBOSE=1` prints how many runs went by cost (`sentToDeviceByCost`).

The rule covers what it covered: runs that start with a matrix element, a magnet, `TAPERAPC`, `TAPERAPR`
or `SPEEDBUMP`. A run that starts with another element, such as an `EDRIFT`, always went to the device
and still does; for a lone `EDRIFT` that loses (9 to 12 ns a particle for the trip, about 3 ns on the
CPU). Not changed.

### Result

| Deck | Particles | Count rule | Cost rule |
| --- | ---: | ---: | ---: |
| Feature-table `zlongit` deck (a `ZLONGIT` every third element) | 10,000 | 1.68 s | 0.93 s |
| | 100,000 | 4.79 s | 1.22 s |
| | 1,000,000 | 23.3 s | 5.09 s |
| Feature-table `ioneffects` deck | 10,000 | 1.38 s | 0.73 s |
| | 100,000 | 12.4 s | 5.88 s |
| `ibsDrift1`, `ibsDrift2` (test set, default mode) | | 2.11 s, 2.29 s | 1.07 s, 1.21 s |
| `LCLSII-2` | | 4.20 s | 3.32 s |
| `forwardThenBack1` | | 10.9 s | 9.08 s |
| `bunchComp1m`, `twla1` | | 4.33 s, 7.46 s | 4.34 s, 7.48 s |
| `uKickMap2` | | 0.22 s | 0.26 s |

The test-set rows are the seven decks of the 424-test routine list whose results move in default mode,
run alone three times each. What is left of the `zlongit` deck is `ZLONGIT` itself, 4.7 ms a call at
100,000 particles.

### Checks

| Check | Result |
| --- | --- |
| Feature set, 430 cases, new build against the GPU reference | 429 unchanged; `small-beam-short-runs-stay` moved, as it should: its lone `KQUAD`s now go to the device |
| Feature set, 436 cases (6 new, 1 adjusted, below), against elegant | 435 within the envelope, 87 of them identical, `scmult-zero-length` flagged as before |
| Routine list, 424 tests, GPU forced on, against the GPU reference | 414 unchanged. In `LCLSII-2`, `TTF2`, `bunchComp1m`, `forwardThenBack1`, `ibsDrift1`, `ibsDrift2`, `ria1`, `ria2`, `twla1` and `uKickMap2` the `.cen` and `.sig` files moved and no other file |
| The same list in default mode, cost rule against count rule in one build | 417 unchanged; the same decks less `TTF2`, `ria1` and `ria2`, the same two files |
| CSR list, 33 tests | 30 unchanged; `TTF2`, `bunchComp1m` and `forwardThenBack1`, the same two files |
| GPU test set, 39 tests | 39 unchanged |
| Verify build (`GPU_VERIFY=1`, `ELEGANT_GPU_VERIFY=1`) on the 436 cases | 436 pass |
| gpu-Pelegant | Built; not run: its rule is unchanged |

The files that moved hold sums over the beam at each element. Where a run now goes to the device those
sums are the device's, added in its order. The largest moves are in `minimum5`, `maximum5` and `ma5` of
the `.sig` files, 2e-15 to 5e-13 in absolute terms: they are taken about the centroid of a coordinate
whose own value is a million and more times larger, so a last-bit move of the centroid shows as 2e-12
to 1e-8 of them, which the comparison flags (6 decks forced, 5 in default mode, 3 in the CSR list;
`forwardThenBack1` also has `betaxBeam` and `alphaxBeam` at 2e-12). No particle file, final file or
`WATCH` file moved in any test.

### Feature cases

Six new and one adjusted, described in "Short runs between CPU-only elements" and "Small beams" of
`feature-set/README.md`: `short-island-goes`, `short-island-first-order-stays`,
`short-island-second-order-goes`, `short-run-threshold-5000-stays`, `short-run-threshold-6500-goes`,
`small-beam-short-runs-pay`, and `small-beam-short-runs-stay`, which now sets
`ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0` and so still guards the count rule and the estimate's use of it.

### Deliberate mistakes

One build with the mistakes behind an environment variable that exists only in it; with the variable
unset it gave the good build's files in all seven cases. Mistake 1 needs no build.

| # | Mistake | Caught by |
| --- | --- | --- |
| 1 | The count alone decides (`ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0` everywhere) | `short-island-goes`, `short-island-second-order-goes`, `short-run-threshold-6500-goes`, `small-beam-short-runs-pay` |
| 2 | Every short run goes to the device | `short-island-first-order-stays`, `short-island-second-order-goes`, `short-run-threshold-5000-stays` |
| 3 | The copies cost nothing | The same three |
| 4 | A margin of 1 | `short-run-threshold-5000-stays` |
| 5 | No extra cost for a second-order matrix | `short-island-second-order-goes` |
| 6 | Every matrix priced as second order | `short-island-first-order-stays`, `short-island-second-order-goes`, `short-run-threshold-5000-stays` |
| 7 | The copies cost nothing per particle | The same three |
| 8 | Only the run's first element is priced | `short-island-goes`, `short-run-threshold-6500-goes` |
| 10 | The small-beam estimate leaves short runs out | `small-beam-short-runs-pay` |

The source was restored by copy from `results/2026-11-13_island-rule/source-after` and compared; the
rebuilt binaries give the seven cases' files again.

### References after this change

Feature set `results/2026-11-13_island-rule/cpu-feature` and `gpu-feature`; routine list
`results/2026-11-13_island-rule/gpu-routine`; CSR list `results/2026-11-13_island-rule/gpu-csr`; GPU test
set unchanged (`results/2026-11-13_csr-exact/gpu-suite`); gpu-Pelegant unchanged. Sources before and
after, the diff of `gpu_stub.c` and the binaries' checksums are in `results/2026-11-13_island-rule`.

### Not done

- `ZLONGIT` and `ZTRANSVERSE` on the device.
- Runs that start with an element the rule doesn't cover (the lone `EDRIFT` above).
- The matrix order before the first pass: a line tracked once keeps a lone second-order matrix on the CPU.
- The 236 slow tests and gpu-Pelegant were not run.

### The feature cost table after the change

Table 8 (`history/2026-11-13_feature-timing-8` in GPU-Testing) was made before the change. To find the rows
the rule touches, each of the 287 decks that table 8 ran with gpu-elegant at 1,000 particles or more was run
with `ELEGANT_GPU_VERBOSE=1` as built and with `ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0`, and the counters
printed at the end (elements on the device by kind, short-run decisions, copies) were compared
(`results/2026-11-13_feature-timing-9/scan`). 13 decks of five rows differ, none at 1,000 particles; the
other 274 do the same under both rules. The 13 were measured again, with 1 and with 8 host threads, and
table 9 (`history/2026-11-13_feature-timing-9`) is table 8 with those numbers:

| Row | Particles | Table 8 | Table 9 | The deck |
| --- | ---: | ---: | ---: | --- |
| `zlongit` | 10,000 | 1.95 ms (CPU) | 1.73 ms | 1.70 s to 0.98 s |
| | 100,000 | 4.24 ms (CPU) | 4.58 ms | 4.81 s to 1.24 s |
| | 1,000,000 | 29.1 ms (CPU) | 38.6 ms | 23.3 s to 5.05 s |
| `ioneffects` | 10,000 | 1.25 ms (CPU) | 1.38 ms | 1.38 s to 0.74 s |
| | 100,000 | 12.8 ms (CPU) | 13.9 ms | 12.4 s to 5.83 s |
| `ftable`, `lsrmdltr`, `bggexp-symplectic` | 10,000 and up | | within 3% | calls of 0.2 to 27 s; a trip adds 10 ms at most |

Table 8's numbers for `zlongit` and `ioneffects` are the element's own time in a deck that stayed on the
CPU. Table 9's are measured like the other rows (wall-clock time against the line without the element), so
they hold the beam's trip to the host and back: about 0.1 ms at 10,000 particles, 1 ms at 100,000 and 10 ms
at 1,000,000. The call itself is still the CPU's. That is why `ioneffects` reads 0.9 times elegant's speed
per call while its deck takes half the time. Each `ZLONGIT` also costs about 0.6 ms once in a run.

## `ZLONGIT` and `ZTRANSVERSE` on the device (November 14, 2026)

Asked for by the user ("Start with zlongit/ztransverse") after the evaluation of the 53 element types that
had no device code (`history/2026-11-14_cpu-only-candidates/REPORT.md` in GPU-Testing). Work folder
`/local/GPU-Testing/results/2026-11-14_zimpedance` (sources before and after, every run below).

### What changed

- `track_data.c`: `GPU_SUPPORT` on both, so elegant was rebuilt (`&print_dictionary` shows them as GPU
  capable). `zlongit.c` and `ztransverse.c`: a dispatch block under `HAVE_GPU`, the same as
  `impedance.c`'s (bunch-plan action, device call, under `GPU_VERIFY` the CPU routine and
  `compareGpuCpu()`). No other CPU code changed.
- `gpu_stub.c`: `gpu_track_through_zlongit()`, `gpu_track_through_ztransverse()` and their bunch-plan
  actions, on the combined wake's kernels in impedance mode. Each bunch: the time span on the device,
  `set_up_zlongit()`/`set_up_ztransverse()` with the CPU's arguments (weak references), the span check and
  messages of the CPU routine, then `gpuCudaCombinedWakeTrack()`.
- `ZLONGIT`'s own grid (`BIN_CENTERED=0`, the default): bins on the mean time (`compute_average()`, from the
  host's ordered sum when output needs the CPU's order), `ib = (t - tmin)/dt` truncated, and bin centers at
  `tmin + dt/2.0 + dt*ib` for the interpolation. `GPU_COMBINED_WAKE_DATA` has `legacyGrid` and
  `tminCenter` (formed on the host as `tmin + dt / 2.0`, so the device's `tminCenter + dt*ib` is the CPU's
  sum). Zero keeps the other users' grid.
- The histogram is always the ordered one (stable sort by bin, sums in particle order). With output in the
  CPU's order, smoothing or `WAKES`, the host makes the voltages with the CPU's operations
  (`SavitzkyGolaySmooth`, `realFFT`, the products in `zlongit.c`'s and `ztransverse.c`'s grouping, the
  inverse `realFFT`) and writes the `WAKES` tables statement for statement (not in a verify build, where the
  CPU rerun writes them); otherwise cuFFT makes the voltages, as for `IMPEDANCE`.
- `ZTRANSVERSE`: channel 1 kicks x' and channel 2 y'; a drive exponent of 1 bins (x - `DX`) or (y - `DY`),
  one below 1 bins 1; the probe exponent multiplies by x or y to that power; a plane with zero
  `FACTOR`*`XFACTOR` (or `YFACTOR`)*ramp is not kicked and writes zeros to `WAKES`, as on the CPU.
- On the CPU still: `AREA_WEIGHT`, `REVERSE_TIME_ORDER`, drive exponents above 1 (an `ipow` histogram),
  probe exponents outside 0 to 3 (`gpuIntegerPower()` is `ipow()` to the last bit only there),
  `INTERPOLATE` other than 0 or 1, gpu-Pelegant with a distributed beam. `ELEGANT_GPU_ENABLE_ZIMPEDANCE=0`
  keeps both on the CPU.
- Bunched beams whose rows aren't in bunch order (after losses: elegant moves lost rows to the end; after
  an injection by `SCRIPT`) went to the CPU, as they still do for `IMPEDANCE` and `CWAKE`. For `ZLONGIT`
  and `ZTRANSVERSE` the device now gathers the rows by bunch with a stable radix sort (each bunch keeps
  elegant's particle order, as `index_bunch_assignments()`'s bucket lists do), tracks the gathered copy and
  writes the rows back (`gpuCudaGatherByBunch()`, `gpuCudaScatterByBunch()`).
  `ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER=0` sends such beams to the CPU. In `collectiveRing1` (48 bunches,
  losses from turn 40) 60% of the impedance calls had gone to the CPU; in forced `rfmode7` 9,998 of 11,998.
  Now none do.
- Shared with `IMPEDANCE` and `CWAKE` (their results unchanged, see Checks):
  `gpuCombinedWakeSortedHistogramKernel` now runs one block per bin, the block reading each stretch of the
  bin's particles into shared memory and one thread adding them in order (as `TRFMODE`'s since November
  13); one thread per bin reading through the index took about 0.5 us a particle of the fullest bin. The
  scratch and its cuFFT plans are no longer rebuilt when the number of partial histograms shrinks (bunches
  of different sizes after losses rebuilt them every call). Only enabled channels' voltages are copied, and
  the partial histograms are cleared only when used.
- Cost model (`gpuSmallBeamElementCost()`): 50 ns a particle on the CPU, 600 us a call on the device.

### Result

| | 10,000 particles | 100,000 | 1,000,000 |
| --- | ---: | ---: | ---: |
| Feature-table `zlongit` row, table 9 (CPU call plus the trip) | 1.73 ms | 4.58 ms | 38.6 ms |
| The same row now | 670 us | 805 us | 2.53 ms |
| Speedup over elegant | 4.1x | 6.1x | 19.3x |

The row's broad-band resonator auto-scales to 131,072 bins. With that many bins the partial histograms
fitted 21 partial sums in their 64 MB, so 21 threads walked 48,000 particles each (69 ms a call); and the
host's FFTs of that grid took 2 ms a call. Hence the ordered histogram always, and cuFFT unless the host's
FFT is needed.

The collective-ring benchmark (480,000 particles in 48 bunches, 100 turns, output in the CPU's order):
76.6 s to 73.0 s; `ZTRANSVERSE` 9.05 s to 6.22 s, `ZLONGIT` 4.23 s to 3.46 s. Its bunches have 10,000
particles, and `LRWAKE`, the RF modes (refused for `RECORD`, `DRIVE_FREQUENCY`, `OUTPUT_FILE`) and the
feedback (bunched mode with 48 bunches) still take the beam to the host every turn.

### Checks

- Feature set, 448 cases (12 new): 97 identical to elegant (the 10 new exact cases among them), 350 within
  the rounding envelope (the two new `*-no-output` cases among them), `scmult-zero-length` flagged as
  before. The 436 old cases are identical to the November 13 GPU reference.
- The 23 elegantTestSet decks with `ZLONGIT` or `ZTRANSVERSE`, default and forced: all identical to elegant,
  `collectiveRing1` included.
- Forced routine list (424) and GPU test set (39): identical to their references
  (`2026-11-13_island-rule/gpu-routine`, `2026-11-13_csr-exact/gpu-suite`).
- Verify build: the 12 new cases and the three CPU-order `IMPEDANCE`/`CWAKE` cases, no mismatch.

### Deliberate mistakes

One build (installed as the `sm_120` variant, which was then rebuilt from the good sources) with the
mistakes behind an environment variable that existed only in it; with it unset no exact case changed.

| # | Mistake | Caught by |
| --- | --- | --- |
| 1 | `ZLONGIT`'s bin centers at `tmin + dt*ib` | the four `zlongit-*` cases with output |
| 2 | `ZLONGIT` on the other users' grid | the same four |
| 3 | The mean time from the device's sum with ordered output | the same four |
| 4 | The voltage factor's products grouped otherwise | `zlongit-output-cpu-order`, after its `FACTOR`s were changed to 0.77 and 0.7: with 0.5 and 1 both groupings gave the same bits |
| 5 | Drive exponent 0 binned as 1 | `ztransverse-output-cpu-order`, `ztransverse-bunches-unsorted` |
| 6 | Probe exponents ignored | the same two |
| 7 | Gathered rows not written back | both `*-bunches-unsorted` |
| 8 | Each bunch gathered in reverse order | both `*-bunches-unsorted` (`ZLONGIT`'s through its mean time) |
| 9 | The ordered histogram added in reverse within each stretch | 7 cases: the `ztransverse-*` cases with output, `impedance-output-cpu-order`, `impedance-output-bunches`, `cwake-output-cpu-order` |
| 10 | `WAKES` `LinearDensity` without its factor | the three `zlongit-*` cases with `WAKES` |

### Not done

- `IMPEDANCE` and `CWAKE` with bunches out of order still go to the CPU; the gather would serve them too.
- gpu-Pelegant keeps the combined wake off unless asked; it was rebuilt but not run.
- The feature cost table was not reissued; the `zlongit` row above is from `feature_timing.py --only zlongit`
  (`results/2026-11-14_zimpedance/feature-zlongit-3`).

## `IBSCATTER` on the device (November 14, 2026)

The second item of the ranked list in `history/2026-11-14_cpu-only-candidates/REPORT.md` (GPU-Testing), done
after `ZLONGIT` and `ZTRANSVERSE` under the user's "Continue to work through the items on the ranked list
hitting the most important". Work folder `/local/GPU-Testing/results/2026-11-14_ibscatter` (the changed
sources as they stood at the end, and every run below).

### What changed

- `track_data.c`: `GPU_SUPPORT` on `IBSCATTER`, so elegant was rebuilt. `ibs_tracking.c`: a dispatch block
  under `HAVE_GPU` before the bucket loop (bunch-plan action, device call, under `GPU_VERIFY` the CPU routine
  and `compareGpuCpu()`), and a label before the IBS file output so the device path writes the same file. No
  other CPU code changed.
- `gpu_stub.c`: `gpu_track_IBS()` and `gpu_ibs_bunched_mode_action()`. The bunch plan is the combined wake's
  (`gpuBunchedWakePlan()`, with the gather of rows out of bunch order from the `ZLONGIT` work). For each bucket
  the rows are copied to the host and `track_IBS()`'s bucket body runs on them with elegant's own functions
  (weak references): `computeTimeCoordinatesOnly()`, `slicebeam()`, `computeSliceParameters()`,
  `correctedEmittance()`, `forth_propagate_twiss()`, `IBSRate()`, `zeroslice()`, with the CPU's messages. So
  every sum, growth rate and IBS-file value is the CPU's to the bit. The device then applies each slice's 11
  numbers (centroids, the x and y inflation factors, the three momentum factors) to its rows,
  `gpuIbsInflateKernel` in `gpu_cuda_runtime.cu`, with the CPU's expressions in the CPU's order. With
  `ISRING=0` the rows are copied again after the inflation for the slice emittances that the file records.
- Why the sums are on the host: a sum in particle order is one chain of dependent adds. On this card an FP64
  add in a chain takes about 28 ns, so one block adding 10,000 particles' second moments took 1.3 ms a call
  however it was arranged (registers, one block per chain, unrolling), against 1.4 ms for the whole CPU
  routine. The copy and the host's sums cost 0.3 ms at 10,000 particles.
- On the CPU still: `SMOOTH=0` (random kicks drawn in particle order), `NSLICE` below 1, bunched mode
  outside a ring, and gpu-Pelegant. `ELEGANT_GPU_ENABLE_IBSCATTER=0` keeps `IBSCATTER` on the CPU.
- Cost model (`gpuSmallBeamElementCost()`): 60 ns a particle on the CPU, 400 us a call on the device.
- Seen in passing, not fixed (a CPU bug, set aside under the user's rule of November 4): in `track_IBS()`'s
  MPI branch `xGrowthRate` is sent twice and `yGrowthRate` never.

### Result

| | 10,000 particles | 100,000 | 1,000,000 |
| --- | ---: | ---: | ---: |
| Feature-table `ibscatter` row, table 9 (CPU call plus the trip) | 3.24 ms | 8.32 ms | 72.7 ms |
| The same row now | 2.96 ms | 5.31 ms | 25 ms |
| Speedup over elegant | 1.1x | 1.4x | 3.0x |

The host's sums set the floor, so the gain is the trip it saves and the inflation. In the elegantTestSet decks,
run one at a time with the same binary and `IBSCATTER` on and off: `ibsAccel1` 5.76 s to 3.54 s, `ibsAccel2`
5.78 to 3.79, `ibsDrift1` 1.15 to 0.68, `ibsDrift2` 1.22 to 0.79 (elegant 9.1, 9.4, 2.1, 2.3 s);
`ibScatter1` to `3` (one `IBSCATTER` a turn in a ring of small beams) are unchanged within the scatter.

### Checks

- Feature set, 456 cases (8 new, see the `IBSCATTER` section of `feature-set/README.md`): 104 identical to
  elegant (the 7 new cases that write only `output` among them, IBS files included), 351 within the rounding
  envelope (`ibscatter-no-output` among them), `scmult-zero-length` flagged as before. The 448 older cases are
  identical to the `ZLONGIT` work's GPU reference (`2026-11-14_zimpedance/gpu-feature-final`).
- `ibscatter-bunches-unsorted` with `ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER=0`: all five calls fall back, so the
  case reaches the gather.
- The 9 elegantTestSet decks with `IBSCATTER`, default thresholds, against elegant: every particle file
  identical; `.sig` (6 decks) and the `WATCH` centroid files of `ibScatter2` and `ibScatter3` differ by the
  device's order of sums. The same binary with `IBSCATTER` on against off
  (`ELEGANT_GPU_ENABLE_IBSCATTER=0`): particle and `WATCH` files identical, `.sig` different in the same 6
  decks (with `IBSCATTER` on the CPU the host has the beam and makes those sums); with forced thresholds
  `ibScatter1` to `3` are identical on and off.
- Deliberate mistakes and the verify build: with the other items, at the end of the list.

## `LSRMDLTR` on the device (November 14, 2026)

The third item of the ranked list, after `IBSCATTER`. Work folder
`/local/GPU-Testing/results/2026-11-14_lsrmdltr` (`source-before`, every run below).

### What changed

- `motion.cc`: the `LSRMDLTR` case of `set_up_derivatives()` now calls `setupLaserModulator()` (declared in
  `track.h`) for the derived quantities and the time profile, the same statements in the same order, and then
  sets `t0` from the beam as before. elegant's outputs for the 7 decks with `LSRMDLTR` are unchanged against
  the November 11 CPU reference. `track_data.c`: `GPU_SUPPORT`. `do_tracking.c`: a dispatch block like the
  Lorentz family's (device call, CPU fallback, under `GPU_VERIFY` the CPU routine and `compareGpuCpu()`).
- `gpu_cuda_runtime.cu`: one thread a particle runs `motion()`'s loop body: `rk_odeint3_na()` with its exit
  search, `rk4_step()`, `derivatives_laserModulator()` (`laserModulatorUndulatorField()` in all three
  expansions, planar and helical, the transverse gradient, pole factors; `computeLaserField()` with the
  Hermite-Gaussian modes to TEM44, tilt, time profile), and `stochastic_laserModulator()`. The operations are
  the CPU's in the CPU's order (the build has `--fmad=false`), complex arithmetic as libstdc++ does it,
  `1/(dz - i ZR)` by Smith's method as libgcc's `__divdc3`. Launches of at most 32,768 particles.
- `gpu_stub.c`: `gpu_track_lsrmdltr()` calls `setupLaserModulator()` (weak reference) and makes `t0` with
  `gpu_findFiducialTime()`, which sums the device's per-particle times on the host in elegant's order, so `t0`
  is the CPU's to the bit. `xMotionCenter` (the RPN variable `motion()` sets from the first particle's x
  excursion) is set from the device's first particle.
- A particle the CPU would lose (a coordinate not finite at the entrance, an integration failure, `Pz <= 0`)
  makes the call restore the beam and send it to the CPU, which loses it with its own message. So does a
  particle whose step loop passes 4 `N_STEPS` + 1,000 (the CPU's has no limit).
- `ISR`: `motion()` draws `gauss_rn_lim(0, 1, 3, random_2)` once a step, particle after particle, and a
  particle's steps depend on its path, so where its numbers start depends on all the particles before it. The
  counts hardly depend on the numbers, so the host guesses a count (the last call's, or `N_STEPS`/1.5 + 1),
  the device tracks with offsets from it and reports each particle's count, the host forms the CPU's offsets
  and tracks again from the first particle whose offset was wrong, until they agree. The numbers come from the
  table the magnets' ISR uses (`gpuIsrFillDraws()`); afterwards `random_2` moves by exactly the numbers used.
  The table holds up to 1 GB at a time (`ELEGANT_GPU_LSRMDLTR_DRAW_BYTES`), batch after batch. In the test
  decks one pass is the rule; two when particles' counts differ.
- On the CPU still: the other integrators (adaptive Runge-Kutta, Bulirsch-Stoer, modified midpoint), laser
  modes above 4 and anything else `motion()` rejects (so that the CPU gives the message), gpu-Pelegant.
  `ELEGANT_GPU_ENABLE_LSRMDLTR=0` keeps it on the CPU; its own threshold is 64 particles
  (`ELEGANT_GPU_MIN_LSRMDLTR_PARTICLES`). Cost model: 190 ns a particle a step on the CPU, 49 us a step a call
  on the device.
- The device's `sin`, `cos`, `cosh`, `sinh` and `exp` are CUDA's (the Gouy phase's `atan` is
  `gpuAccurateAtan()`), so the particles differ from elegant's by rounding.

### Result

| | 10,000 particles | 100,000 | 1,000,000 |
| --- | ---: | ---: | ---: |
| Feature-table `lsrmdltr` row, table 9 (on the CPU) | 190 ms | 1.92 s | 19 s |
| The same row now | 9.74 ms | 62.7 ms | 584 ms |
| Speedup over elegant | 19.6x | 30.5x | 32.7x |

elegantTestSet decks, each run alone: `lsrMdltr1` to `6` 4.5 to 5.2 s to 0.35 to 0.43 s (12x to 14x),
`lsrMdltr3` (`ISR=1`, 10,000 particles, 4,000 steps) 10.3 s to 0.71 s, `NLSJitter1` 13.3 s to 4.7 s.

### Checks

- Feature set, 467 cases (11 new, see the `LSRMDLTR` section of `feature-set/README.md`): 106 identical to
  elegant (`lsrmdltr-fallback` and `lsrmdltr-opt-out` among them), 360 within the rounding envelope (the other
  9 new cases), `scmult-zero-length` flagged as before. The 456 older cases are identical to the `IBSCATTER`
  run's, CPU and GPU.
- The 7 decks with `LSRMDLTR`, default and forced thresholds: 6 within the envelope, `lsrMdltr3` with `ISR`
  among them, so every particle took elegant's random numbers. `NLSJitter1`'s final `WATCH` is flagged (x
  2.6e-9 m, p 5.6e-5 against rms x 2.9e-5 m): the linac amplifies rounding. With a `WATCH` added right
  after the laser heater the device differs from elegant by 4e-16 of p and t there; elegant with `BU` one ulp
  larger moves the final `WATCH` by x 2.6e-9 m and p 7.6e-5, as much as the device. With `LSRMDLTR` on
  the CPU, gpu-elegant's `NLSJitter1` is identical to elegant's.
- `lsrmdltr-isr-mixed-counts` with a table of about 10 particles' numbers: 446 passes over some 220
  batches, particles identical to the run in one batch.
- Deliberate mistakes and the verify build: with the other items, at the end of the list.

## `LRWAKE` on the device (November 14, 2026)

The fourth item of the ranked list. Work folder `/local/GPU-Testing/results/2026-11-14_lrwake`.

### What changed

- `lrwake.c`: the block of `track_through_lrwake()` that turns the bunch sums into bunch means, moves the history
  and sums the wakes of earlier bunches is now `lrwakeBunchVoltages()` (declared in `track.h`, with
  `set_up_lrwake()`), the same statements; the routine calls it. A dispatch under `HAVE_GPU` at the start of the
  routine, where its static `lastNBunches` is in reach. `track_data.c`: `GPU_SUPPORT`. elegant's `lrwake1` and
  `collectiveRing1` are unchanged against the November 11 and November 14 CPU runs.
- `gpu_stub.c`, `gpu_track_through_lrwake()`: the bunch range from the device's bunch column (as
  `index_bunch_assignments()` finds it), the same check that the number of bunches didn't grow (else the CPU
  stops the run with its message), `set_up_lrwake()`, then `gpuCudaLrwakeValues()`: each particle's time with
  `computeTimeCoordinatesOnly()`'s expression, kept on the device, and t, x, y and the bunch column copied to the
  host, which adds them per bunch in particle order. `lrwakeBunchVoltages()` does the rest on the host, and
  `gpuCudaLrwakeKick()` applies `add_to_particle_energy()` and the transverse kicks with the CPU's operations. In
  a verify build the history is worked on a copy, so that the CPU's rerun finds it as it was.
- Why the sums are on the host: as for `IBSCATTER`, an ordered sum on the device is a chain of dependent adds;
  the copy is 32 bytes a particle.
- On the CPU still: gpu-Pelegant, trajectory tracking. `ELEGANT_GPU_ENABLE_LRWAKE=0` keeps it on the CPU. Cost
  model: 40 ns a particle on the CPU, 150 us a call on the device.

### Result

The collective-ring benchmark (480,000 particles in 48 bunches, 100 turns), `element_breakdown.py`, `LRWAKE` off
and on alternately, twice: `LRWAKE` 1.60 s to 0.33 s; `TFBPICKUP`, next in the ring and still on the CPU for
48 bunches, takes over the beam's trip to the host (+0.28 s). The whole run moved by -2.2 s and +2.6 s in the
two pairs (75.2, 73.0, 78.5, 81.1 s): `WATCH` and `HISTOGRAM` write files and vary by several seconds from
run to run. A first single pair (75.8 s off, 92.0 s on) was that scatter; the number of copies to the host is
the same either way (`ELEGANT_GPU_SYNC_REASONS=1`, 10 in 5 turns). Where the element after `LRWAKE` is on the
device, the trip goes away.

### Checks

- Feature set, 472 cases (5 new): 110 identical to elegant (`lrwake-bunches-output`, `lrwake-bunches-unsorted`,
  `lrwake-single-bunch`, `lrwake-opt-out` among them), 361 within the envelope (`lrwake-no-output`, its `.cen`
  and `.sig`), `scmult-zero-length` flagged as before. The 467 older cases are identical to the `LSRMDLTR` run's,
  CPU and GPU. With `FACTOR=0` the particles of `lrwake-bunches-output` move by 1e-4 to 1e-2 of their range, so
  the wakes act.
- `lrwake1` and `collectiveRing1`, default and forced thresholds: identical to elegant.
- Deliberate mistakes and the verify build: with the other items, at the end of the list.

## `CORGPIPE` and `CORGPLATES` on the device (November 14, 2026)

Item 5 of the ranked list (the cheap group). Work folder `/local/GPU-Testing/results/2026-11-14_corrugated`.

- Both elements are `exactDrift()`, a `WAKE` built from a formula, and `exactDrift()`; each of the three already
  went to the device when `getElementOnGpu()` was set. What was missing: `GPU_SUPPORT` in `track_data.c`,
  eligibility (in `gpuWakeElementSupported()`, with the limits the device `WAKE` has: `INTERPOLATE` 0 or 1,
  `N_BINS` 0 or at least 2), and, when `TMAX=0` and `N_BINS=0`, the beam's smallest and largest s, which
  `wake.c` now takes from `gpu_beam_s_range()` (a device reduction; the extremes are the same in any order; the
  verify build checks them against the CPU's loop). `ELEGANT_GPU_ENABLE_CORRUGATED=0` keeps them on the CPU.
- Per call at 1,000,000 particles (five of each, three passes): `CORGPIPE` 54 ms to 11.6 ms, `CORGPLATES` 40 ms
  to 4.2 ms; the deck 2.21 s to 1.17 s. `dechirper1` and `dechirper2` (100,000 particles, one call) take 0.1 s on
  the CPU and are start-up on the GPU.
- Checks: `dechirper1` and `dechirper2` identical to elegant at default and forced thresholds and with the opt-out;
  the four feature cases as listed in `feature-set/README.md` (three identical, the centroid-only one within
  rounding).

## `TAPERAPE` on the device, and `TAPERAPR` with `TILT` made exact (November 14, 2026)

Item 6. Work folder `/local/GPU-Testing/results/2026-11-14_taperape`.

- `TAPERAPE` joins `TAPERAPC` and `TAPERAPR` in `gpuTaperApertureTrackParticle()`
  (`GPU_TAPER_APERTURE_ELLIPTICAL`): the offset and tilt, `insideTaperedEllipse()` at the entrance and the exit
  and the bisection to `RESOLUTION`, the loss position, `exactDrift()` for survivors in the tilted frame, then the
  tilt and offset undone; the CPU's loss order comes from the existing builder. `do_tracking.c` dispatches it
  like the other two; `track_data.c` has `GPU_SUPPORT`; `ELEGANT_GPU_ENABLE_TAPERAPE=0` keeps it on the CPU.
- The test of a point against the ellipse takes `atan2`, `cos`, `sin` and `pow`. The device's can differ from
  glibc's in the last place, so a test whose two radii agree to 1e-12 is counted, and a call with any such test
  goes to the CPU with the coordinates as they were (the device's results are in the scratch copy). So every
  decision is elegant's. `aperture-taperape-boundary` places two particles exactly on the ellipse and reaches
  that fallback.
- Global loss coordinates: `trackThroughTaperApElliptical()` converts a loss while the particle is in the
  element's offset and tilted frame, so the device puts x, x' and y of that frame into the three global columns
  and the host converts them with `convertLocalCoordinatesToGlobal()`. The first build converted the lab-frame
  values and `aperture-taperape`'s losses file differed in `X` for the tilted element.
- `TAPERAPR` with `TILT`: survivors' path length came from the slopes rotated back to the lab frame, where the
  CPU's `exactDrift()` runs in the tilted frame; the verify build had found the fused program 2e-16 off on
  November 11, and the program had refused it since. The device now takes the tilted frame's slopes for both,
  and `gpuTuneProgramExactElement()` no longer refuses it: `search-frequency-map-taperapr-tilt` is identical,
  and the program's own check against the CPU found no difference.
- Not done: the batched momentum search's list (`gpu_momentum_search_beamline_supported()`) still lacks
  `TAPERAPE` (and `UKICKMAP`), which the APS-U lattice in the test set has; the fused program would need the
  uncertain-test fallback first.
- Checks: the six cases in `feature-set/README.md` and the older taper cases: particles and losses identical, the
  centroid, sigma and final files within rounding (device order of sums), as before.

## `CENTER` to the bit, and elements idle on a pass (November 14, 2026)

Items 7 and 8. Work folder `/local/GPU-Testing/results/2026-11-14_center-idle`.

### What changed

- `gpu_center_beam()` rewritten to follow `center_beam()` step by step: each coordinate's mean from that column
  copied to the host (`gpuCudaCoordinateColumnToHost()`) and added in particle order with a plain sum, as the
  CPU adds it; the offsets subtracted on the device; then, for `T`, each particle's time
  (`gpuCudaTimeColumnToHost()`, after the other offsets, so an `S` offset moves the time mean as on the CPU)
  added with the CPU's Kahan sum, and the round trip through time on the device
  (`gpuCenterTimeRoundTripKernel`: `computeTimeCoordinates()`, the offset, `computeDistanceCoordinates()`, even
  for a zero offset). The old code took all its sums before any offset and moved s by c*beta*offset, so it was
  never the CPU's to the bit, and `gpuHelperElementSupported()` refused `CENTER` every time. Now
  `gpuHelperMatchesCpu()` accepts it. `ELEGANT_GPU_ENABLE_CENTER=0` keeps it on the CPU. Not in gpu-Pelegant:
  there `center_beam()`'s sums, and `computeTimeCoordinates()` even with a stored offset, are collective and
  the master stays on the CPU, so a device path on the workers would hang.
- Idle passes: `do_tracking()` gives the GPU code the pass number (`gpuSetTrackingPass()`), and
  `gpuIdlePassElement()` lets `BRANCH`, `TWISS` with `APPLY_ONCE=1` after its first pass (or `DISABLE`),
  `POLAR` and `CENTER` off their `ON_PASS`, and `SCATTER` outside `STARTONPASS`..`ENDONPASS` leave the beam on
  the device, in plain tracking only (the batched programs neither branch nor know the pass). `RIMULT`
  (`T_MRADINTEGRALS`, which tracking ignores) is passive everywhere. Each element's CPU case still runs on those
  passes, on the host's stale copy, and does nothing to it. `ELEGANT_GPU_ENABLE_IDLE_PASS=0` restores the trips.
  Not done: `DSCATTER` (it initializes before its pass check), `BUMPER`/`MBUMPER` before `FIRE_ON_PASS` (a drift).

### Checks

- The 53 elegantTestSet decks with `CENTER`, `RIMULT`, `BRANCH`, `TWISS`, `POLAR` or `SCATTER`, on the same
  binary with the new paths on and off (`ELEGANT_GPU_ENABLE_CENTER=0`, `ELEGANT_GPU_ENABLE_IDLE_PASS=0`): no
  particle file changed; `.cen` and `.sig` changed in 17 (the `LCLS`, `firstIsFid` and `twissElem` decks,
  `scraper1`), where the beam now stays on the device and those sums are the device's. `branch2`, `branch4`
  and `LCLSII-1`/`-2` differ from elegant by rounding with the new paths off too. `backwardThenForward1` and
  `optimization12` fail on the CPU as well. `LCLS0` 1.60 s to 1.37 s, `LCLS3` 2.46 to 1.85.
- The six feature cases in `feature-set/README.md`: identical to elegant. With `ELEGANT_GPU_SYNC_REASONS=1`,
  `idle-pass-elements` copies the beam once for `TWISS` (its first pass) and twice for `SCATTER` (its two
  passes), and 24 element calls stay on the device.

## Checks at the end of the list (November 14, 2026)

Work folders `/local/GPU-Testing/results/2026-11-14_mistakes` and `/local/GPU-Testing/results/2026-11-14_final`
(final sources in `source-final`, checksums in `binaries-final.sha256`).

### Verify build

The 40 feature cases added today, with `ELEGANT_GPU_VERIFY=1`: all complete, 11,348 checks passed (50 of
`track_IBS`, 16 of `LSRMDLTR`, 24 of `track_through_lrwake`, 12 of the corrugated s range, 8 of `center_beam`).
The first run stopped in all seven `ibscatter-*` cases: `forth_propagate_twiss()` starts from `twiss0`, the
last call's beam, and stores this call's there, and the slice loop reads the next slice's growth rate from the
last call, so the CPU's redo found the device path's values. `ibs_tracking.c` now saves and restores them
around the device call in verify builds only; ordinary builds are unchanged.

### Deliberate mistakes

One build (the `sm_120` variant, rebuilt from the clean sources afterwards and checked free of the mistake
code) with 15 mistakes behind `ELEGANT_GPU_PLANTED_MISTAKE`; with it unset, the 40 new cases gave the clean
results. A case that must be identical to elegant catches a mistake if any output changes; the `LSRMDLTR` cases,
which differ by rounding, catch one if the comparison flags them.

| # | Mistake | Caught by |
| --- | --- | --- |
| 1 | `IBSCATTER` momentum factors of the neighbouring slices swapped | `ibscatter-slices-output`, `-linac-output`, `-no-output` |
| 2 | `ISRING=0` slice emittances not taken again after the inflation | `ibscatter-linac-output` |
| 3 | `random_2` committed one ISR number short | `lsrmdltr-isr` (through the `KQUAD` after), `lsrmdltr-isr-batches` |
| 4 | `t0` without `TIME_OFFSET` | `lsrmdltr-time-profile` |
| 5 | The first ISR offsets accepted without the check | `lsrmdltr-isr-mixed-counts`, `lsrmdltr-isr-batches` |
| 6 | `LRWAKE` bunch sums in reverse particle order | the four `lrwake-*` cases with `LRWAKE` on the device |
| 7 | `LRWAKE` kicks without `RAMP_PASSES` | the same four |
| 8 | The beam's largest s taken as its smallest | `corrugated-pipe-output`, `corrugated-plates-output` |
| 9 | `TAPERAPE` global loss coordinates from the lab frame | `aperture-taperape` |
| 10 | `CENTER`'s time mean taken before the other offsets | `center-once-only` |
| 11 | `CENTER`'s coordinate means with a Kahan sum | `center-output`, `center-once-only`, `center-on-pass` |
| 12 | `TWISS` idle on its first pass too | nothing, rightly: its matrix goes through `track_particles()`, which goes to the device by itself, so the result is the same (with `FROM_BEAM=1` it would read the host's stale beam, which is why its acting pass is never idle) |
| 13 | `SCATTER` idle on its first active pass | `idle-pass-elements` |
| 14 | `TAPERAPR` survivors' path length from the slopes rotated back (the old code) | `aperture-taperapr-tilt` |
| 15 | `LSRMDLTR` second and third pole factors swapped on the device | all nine `lsrmdltr-*` cases on the device |

### gpu-Pelegant

Built after each change. The first build warned of two unused tables in MPI builds (moved into the non-MPI
branch). Every new device path is refused under `USE_MPI` or with a distributed beam except `TAPERAPE` and the
idle passes, which make no collective; `CENTER` was the exception found while checking this (see above).

### Feature set

488 cases (40 added today after the `ZLONGIT` work): 121 identical to elegant, 366 within the envelope,
`scmult-zero-length` flagged as before. elegant's results for the 472 cases of the `LRWAKE` run are unchanged;
gpu-elegant's changed in four: `matrix-center-time` and `matrix-helpers` (`.cen` and `.sig`: `CENTER` now keeps
the beam on the device, so the next sums are the device's) and `aperture-taper-accepted` and
`aperture-taper-global-loss`, whose particle files, a last bit off elegant's before (inside the envelope), are
now identical to it (the tilted `TAPERAPR`'s path length).

### Routine list, GPU suite and gpu-Pelegant

- Forced routine list (424 tests, `gpu-forced-64.env`) against the `ZLONGIT` work's `final-routine`: 393 unchanged,
  31 changed, each accounted for: `.cen`/`.sig` of the `LCLS`, `firstIsFid` and `twissElem` decks and
  `scraper1` (`CENTER` and `TWISS` keep the beam on the device), `.sig`/`.fin` of the `ibsAccel` and `ibsDrift`
  decks (`IBSCATTER` on the device), `run.out` of `lsrMdltr1` to `6` (rounding), `NLSJitter1`'s `WATCH`
  (amplified rounding, see the `LSRMDLTR` section), and `fmult2`, whose error logs differ only in the
  `&associate` header that records the temporary run directory.
- GPU suite (39 tests) against `final-suite`: 38 unchanged; `gputest1-resident-magnets-focused` moved 3 of 8,192
  times by an ulp, from its `TAPERAPR` at `TILT=0.04`. Against elegant itself that file now differs in 1 row
  (a `CSBEND` edge model 5 with radiation, as before) where it differed in 4.
- gpu-Pelegant, rebuilt last, on 18 decks with today's element types (`lrwake1`, `dechirper1`/`2`, `ibsDrift1`,
  `ibsAccel1`, `ibScatter2`, `lsrMdltr1`/`3`, `NLSJitter1`, `LCLS0`/`3`, `twissElem1`, `scatter1`, `branch1`,
  `polar1`, `mRadInt1`, `firstIsFid1`, `errorSampling1`), `mpiexec -np 4`, against CPU Pelegant built with `-Og`
  the same day: all complete, no hang, and every compared output identical.

### Feature table 10

Every gpu-elegant cell of the feature cost table measured again with the final binary, 1 and 8 host threads
(`/local/GPU-Testing/history/2026-11-14_feature-timing-10`: `FEATURES.md`, `FEATURES.pdf`; runs in
`results/2026-11-14_feature-timing-10`). 256 of 272 cells within 15% of table 9; the others are `lsrmdltr`
(19x to 33x elegant now), `zlongit` (4.1x to 20x), `ibscatter` (1.4x and 2.9x at 100,000 and 1,000,000) and
seven `WATCH` cells. `CENTER` now has `GPU_SUPPORT` in `track_data.c` (it had GPU code that was never used), so
the table's list of CPU-only types, read from that flag, has 44 types where table 9 had 53.

## `BEAMBEAM`, `RFTMEZ0`, `TWLA`, `MAPSOLENOID` and `MODRF` on the device (evening of November 14, 2026)

Items 1, 2 and 4 of the second evaluation of the CPU-only types
(`/local/GPU-Testing/history/2026-11-14_cpu-only-eval/REPORT.md`). Work folder
`/local/GPU-Testing/results/2026-11-14_items-1-2-4` (clean sources in `source-clean`, every run below).

### What changed

- `gpu_faddeeva.h` (new): `Faddeeva::w(z)` for the device, as elegant calls it (default `relerr`, the
  precomputed table). The package's two tables of Chebyshev polynomials, `erfcx_y100()` and `w_im_y100()`,
  are `Faddeeva.cc`'s lines unchanged (checked with `diff`), so they give the CPU's values to the bit with
  `--fmad=false`; `erfcx`, `w_im` and `w` follow `Faddeeva.cc`'s statements in order. Its `exp`, `sin` and
  `cos` are CUDA's.
- `BEAMBEAM`: `gpuBeamBeamKernel()` is `applyBeamBeamKicks()` with `gaussianBeamKick()`,
  `roundGaussianBeamKick()` and `ellipsoidalBeamKick()` (glibc's `csqrt` and libgcc's `__divdc3` written out);
  the products of constants are formed on the host as the CPU forms them (`gpu_track_beambeam()`). No sums,
  random numbers or losses, so it is also allowed in gpu-Pelegant. `ELEGANT_GPU_ENABLE_BEAMBEAM=0` opts out;
  magnet threshold.
- `RFTMEZ0`, `TWLA`, `MAPSOLENOID` with `METHOD="non-adaptive runge-kutta"`: `motionGpuSetup()` (new, in
  `motion.cc`) runs `motion()`'s set-up for the call (the same globals, fiducial selection from a read-only
  host copy of the beam when it is due) and returns what the particle loop needs; the device runs that loop
  (`gpuMotionKernel()`, a copy of `LSRMDLTR`'s `rk_odeint3_na()` with these fields; `rotate3()` as
  `mat_mult()`'s sums). Entrance, integration and exit losses are marked on the device with `motion()`'s
  loss records in the rows; the host forms `motion()`'s swap order and the device gathers the rows
  (`gpuCudaMotionPermute()`). With `CHANGE_P0` the central momentum is matched on the device to all the
  particles the call was given, lost ones included, before their rows go to the host for the losses file, as
  `motion()` does. A rectangular-aperture loss after the integration (whose recorded y can be the previous
  particle's in `motion()`), a step loop past 4 `N_STEPS` + 1,000, or lost particles with an accepted beam send
  the call to the CPU, with the element put back as it was before the set-up. `TWMTA`, the adaptive
  integrators and gpu-Pelegant stay on the CPU. `ELEGANT_GPU_ENABLE_RFTMEZ0`, `_TWLA`, `_MAPSOLENOID`;
  threshold `ELEGANT_GPU_MIN_MOTION_PARTICLES` (64).
- `MODRF`: `mod_rfca.c` keeps its set-up, phase reference, record file and fill factor and dispatches like
  `ramped_rfca.c`: the fiducial time from `gpu_findFiducialTime()`, the mean time from the device's particle
  times added on the host with `KahanPlus` in particle order (`gpu_rf_particle_times()`), and the kick is
  `RAMPRF`'s kernel, which now takes a time offset (0 for `RAMPRF`, so its arithmetic is unchanged; the CPU's
  `sin` on the host with output in CPU order). Not with a distributed beam. `ELEGANT_GPU_ENABLE_MODRF=0`.
- `track_data.c`: `GPU_SUPPORT` for the five types. `do_tracking.c`: dispatch blocks; in verify builds the
  motion elements' state is restored before the CPU's `motion()` reruns them (below).

### Result

- Collider ring (`ILMATRIX`, `RFCA`, `SREFFECTS`, gaussian `BEAMBEAM`), 1,000,000 particles: 150 ms a turn
  before, 8.5 ms now (elegant 360 ms).
- elegantTestSet, default thresholds: `ria1` (87 `RFTMEZ0`s, 6,000 particles) 4.79 s to 0.62 s; `twla1` 7.49 s
  to 0.53 s. PAR ring with `MODRF`, 100,000 particles, 64 passes: 3.70 s to 0.40 s. Per-call numbers are in
  feature table 11.

### Checks

- Exactness: `MAPSOLENOID` and `MODRF` give elegant's particles to the bit; `BEAMBEAM`, `RFTMEZ0` and `TWLA`
  differ by rounding (one ulp after a few turns on the collider ring; `twla1` 2e-12). `ria1`'s low-energy
  heavy-ion beam amplifies rounding until 919 of 6,000 particles differ by more than 1e-3 in p at the end;
  elegant with `p_central` one ulp larger moves 962 of them that far.
- Feature set, 514 cases (26 new, see its README): the 488 earlier ones are unchanged on the CPU and on the
  GPU against `results/2026-11-14_final`; against elegant 131 are identical, 382 within the envelope, and
  `scmult-zero-length` is flagged as before (`cpu-feature-514`, `gpu-feature-514`: the current references).
- Forced routine list (424): 420 unchanged; `mapSolenoid1` (device sums), `ria1`, `ria2` (amplified rounding
  as above) and `twla1` (`betayBeam` and `alphayBeam` of two rows 3e-12 apart, from the sigma matrix) changed.
  GPU suite (39): unchanged. `beamBeam1`, `twla2` and `ringRF2`, default and forced: within the envelope.
- Verify build: the 26 new cases pass (28 `BEAMBEAM`, 18 `motion()` and 72 `MODRF` checks at 1e-12). The first
  run failed `twla-solenoid`: `TWLA`'s ponderomotive factor uses the fiducial particle's energy on the call
  that selects the fiducial and the central energy on later calls, so the CPU's rerun, a later call, differed;
  the verify build now restores the element before the rerun, and `gpu_track_motion()` restores it before any
  fallback to the CPU.
- Deliberate mistakes: 20 planted behind `ELEGANT_GPU_PLANTED_MISTAKE` in the `sm_120` variant (rebuilt from the
  clean sources afterwards), all caught by the new cases: `Faddeeva::w`'s continued fraction, sums and real-axis
  branch; the gaussian kick's vertical sign; the round kick; both ellipsoidal kicks; the on-axis and mapped
  solenoid fields; `rotate3()`; `TWLA`'s impulse and droop; the loss records, the loss order and the momentum
  match over the lost particles; `MAPSOLENOID`'s uniform field and grid; `MODRF`'s time offset and its Kahan
  sum. The mapped-solenoid mistake was caught only after `rftmez0-solenoid`'s map was made to vary with r:
  elegantTestSet's map has the same Bz on every radial page.
- gpu-Pelegant (rebuilt from the clean sources), `mpiexec -np 4`, on `beamBeam1`, `ringRF2`, `twla1`, `twla2` and
  `mapSolenoid1`, against the `-Og` CPU Pelegant: all complete, no hang, every compared output identical.
- CPU findings, logged and not fixed: `motion()` with `CHANGE_P0` includes the particles it lost in that call
  in `do_match_energy()` (their p column holds the absolute momentum then); the `TWLA` ponderomotive factor
  above; `MHISTOGRAM` without `INPUT_BINS` stops elegant with a segmentation fault (`set_up_mhist()` reads the
  bin counts from that file without checking that it was given); `TUBEND` loses every particle.

### Feature table 11

The feature cost table measured again after these five types moved, 1 and 8 host threads, with elegant now run
at 1,000,000 particles as well (`/local/GPU-Testing/history/2026-11-14_feature-timing-11`: `FEATURES.md`,
`FEATURES.pdf`; runs and `note.txt` in `results/2026-11-14_feature-timing-11`). 152 rows, 83 of them new: 136
of the 139 element types have a row (not `TUBEND`, and `TFBDRIVER` and `CKICKER` only within their pairs).
264 of the 272 cells shared with table 10 are within 15%; the eight others are five `WATCH` cells and three that
are scatter. At 10,000, 100,000 and 1,000,000 particles: `rftmez0` 26x, 40x and 38x elegant, `twla` 28x, 42x and
36x, `mapsolenoid` 23x, 35x and 35x, `modrf` 3.7x, 5.9x and 7.2x, `beambeam` 6.3x, 12x and 18x. Fixed in the
report on the way: elegant's time of a paired row (`tfeedback`, `cooler`) now holds both elements, so
`tfeedback`'s speedups are about twice table 10's; `BUMPER` and `MBUMPER` use a flat waveform, since with the
test set's pulses every call after the first only drifted.

## 26 CPU-only types on the device (November 15, 2026)

The 26 items of the sub-1x evaluation that a port could lift (`/local/GPU-Testing/history/2026-11-15_sub1x-eval/REPORT.md`,
"Which of these a GPU port would fix"): every CPU-only row of the feature tables but `SCRIPT`. Work folder
`/local/GPU-Testing/results/2026-11-15_ports` (sources at the start in `source-start`, every run below).

### What changed

- Three new files hold the ports: `gpu_ports.h` (data and prototypes, included at the end of `gpu_base.h`),
  `gpu_ports_device.h` (kernels, included at the end of `gpu_cuda_runtime.cu`) and `gpu_ports_host.h` (host
  functions, included at the end of `gpu_stub.c`). One table, `gpuPortTypes[]`, gives each type its switch
  (`ELEGANT_GPU_ENABLE_<TYPE>`, on by default; `_COOLER` for `CPICKUP` and `CKICKER`, `_GFWIGGLER` for `APPLE`),
  its support check and its small-beam cost. They follow the magnet threshold and are off in gpu-Pelegant,
  which was checked only on the CPU paths.
- Shared machinery:
  - Losses: kernels flag lost particles; the host rebuilds elegant's swap order from the flags (or follows the
    CPU's loop itself where random numbers depend on the order) and the device gathers the rows. Lost rows go to
    the host for the losses file. An accepted beam or global loss coordinates send the call to the CPU.
  - Sums: made on the host from columns copied off the device, in particle order, with `KahanPlus` where the CPU
    uses it.
  - Host copies: CPU routines given a read-only copy run between `gpu_port_host_begin()` and
    `gpu_port_host_end()` (`GPU_HOST_BEGIN`/`END` in `do_tracking.c`), so their own helpers
    (`compute_centroids()`, `accumulate_beam_sums()`) stay on the host.
  - Invalid coordinates: the kernels count survivors with a non-finite x or y, and `gpu_limit_amplitudes()` then
    removes them as elegant's `limit_amplitudes()` does after the element.
- By type:
  - `KPOLY`, `HKPOLY`, `MBUMPER` (own fourth-order integrator, waveform cached on the device; the first call's
    set-up and fiducial time from a host copy).
  - `RMDF`, `MRFDF`, `SHRFDF`, `RFTM110`: first-call times from host copies; `RFTM110`'s noise from `random_3` on
    the host in the CPU's order.
  - `STRAY` and `ALPH`: matrices through the device's `track_particles()`, `ALPH`'s puck and slits as kernels.
    `KSBEND`: the integrator, the host's tangents. `BOFFAXE`: the field expansion with the file's table uploaded
    once (`boffaxeFieldTable()` in `magneticFieldOffAxisExpansion.c`); the set-up, which reads the file, runs on the
    host with no particles (`trackMagneticFieldOffAxisExpansion()` with none), so the first call is the device's too.
    At first it ran the whole first call on the CPU: with one pass (a line) no call reached the device, and one pass
    of 1,000,000 particles through two of them took 73 s against elegant's 55 s. `GFWIGGLER`: `APPLE_Track()`.
  - `PFILTER`, `SAMPLE`, `CLEAN`: compaction in the CPU's order, sums on the host. `EMITTANCE`, `TWISS`, `REMCOR`:
    transformations from host sums, applied on the device. `MHISTOGRAM`: minima, maxima and integer counts on the
    device. `CPICKUP` reads a host copy; `CKICKER` changes delta on it and that column goes back.
  - `SCATTER`, `DSCATTER`, `MATTER`, `PEPPOT`: `random_2` and `gauss_rn` drawn on the host in elegant's order,
    applied on the device. `MATTER` takes the multiple-scattering modes and the single-scattering sections (the
    host draws each section's events, the device replays the sections); slots, bremsstrahlung and electron recoil
    stay on the CPU. `DSCATTER`'s first call reads its file without tracking; `ONCEPERPARTICLE` stays on the CPU.
    `PEPPOT`: the device tests the holes, the host draws and orders, twice. `CBSCAT`: tracked on a current host
    copy, x', y', s and delta sent back.
  - `IONEFFECTS`: the ions on a current host copy, as elegant tracks them; the Gaussian field's kicks to each
    bunch's electrons on the device with `BEAMBEAM`'s `gaussianBeamKick()` (`ionEffects.cc` calls
    `gpu_ioneffects_kicks()`); other fields and `conserve_momentum` kick the copy and x', y' go back.
- CPU files, all inside `HAVE_GPU` or behavior-neutral: `do_tracking.c` (dispatch, host-copy macros, `CBSCAT` and
  `IONEFFECTS` copies only on passes they act on), `matter.c` (calls `gpu_track_matter()`; `gpu_matter.h` and its
  stub are gone), `ionEffects.cc` (`ionEffectsActiveOnPass()`, `ionEffectsUseDeviceKicks()`, the bunch's rows),
  `track.h`, `track_data.c` (`GPU_SUPPORT` on the 27 types), `magneticFieldOffAxisExpansion.c`.
- Verify build: the per-particle comparison counts NaN against NaN (or the same infinity) as a match; `SAMPLE`'s
  fraction mode reruns the CPU with the same random numbers; `elegantto` links (weak `swapParticles`,
  `transformEmittances`).
- Slice analysis after each element: elegant makes it before it drops the particles an element just lost, so its
  row at a loss element counts them, with their recorded z and momentum. The lost rows now go to the host whenever
  that analysis is on, as they do for a losses file (`gpuBaseInit()`'s `lossOutputNeeded` in `do_tracking.c`).
  Found with `PFILTER` on the device in `bunchComp1` and `bunchComp1m`, whose rows at `L3SC1PFIL` read stale rows
  (6,607 particles in slice 3 against elegant's 0); now identical. Any device loss element had the same gap.

### Checks

- Exactness: every new type gives elegant's particles to the bit in its feature cases except `RFTM110` and
  `GFWIGGLER` (CUDA's `sin`, `cos` and `exp`), which differ by rounding. `MRFDF`, `SHRFDF`, `KSBEND` with a
  fringe integral and the `IONEFFECTS` kicks use CUDA's functions too and are identical in their cases; on
  elegantTestSet's `mrfdf1` to `mrfdf3` `MRFDF` differs by rounding.
- Feature set, 592 cases (78 new, see its README): against elegant 207 identical (76 of the new ones), 384 within
  the envelope, `scmult-zero-length` flagged as before. The 514 earlier ones: unchanged on the CPU; on the GPU
  against `results/2026-11-14_items-1-2-4/gpu-feature-514`, 511 unchanged and three changed by rounding in their
  centroid and sigma sums only (`magnet-ksbend-loss-with-gpu`, `matrix-octu-rben-alph`, `matter-fallback`, whose
  `KSBEND`, `ALPH` and `MATTER` now run on the device). References: `cpu-feature-592`, `gpu-feature-592`.
- Twelve earlier cases used `PFILTER`, `SCATTER` or `TWISS` as the CPU element that splits a line
  (`small-beam-*`, `short-island-*`, `short-run-*`, `idle-pass-*`). With those types on the device they stopped
  testing their rules (six changed); `suite.json` now keeps the types on the CPU for them, and all twelve match the
  previous GPU reference exactly.
- Weak cases found and fixed on the way: `rmdf-ramp` and `rmdf-opt-out` lost every particle by the third pass (the
  ramp's field grows with time; `VOLTAGE` is now 1e3, and all 500 survive with a visible kick); `cooler-opt-out`
  was an older deck that lost every particle (now `cooler-pair`'s); `alph-filters` had its puck and slits off the
  beam, so the slits took everything (they are measured back from `XMAX`; now the puck and momentum slits take 153
  particles and the position slits 142, and 205 of 500 survive two passes); `pfilter-fixed-centered`'s delta
  window never cut inside the fixed fraction limits (`DELTALIMIT` 2.5e-3 to 1.2e-3); `cbscat-*` scattered 3 particles of 500 (`FACTOR` 2e5 and a wider laser: 82).
  `rmdf-invalid` still ends with no particles, by design: it tests the removal of invalid coordinates.
- Deliberate mistakes: 32 planted behind `ELEGANT_GPU_PLANTED_MISTAKE` in the `sm_120` variant (patch
  `mistakes/plant.py`; the variant rebuilt from the clean sources afterwards), one or more for each type and for
  the shared loss order, host-copy guard and column upload. All 32 are caught by the new cases: 27 as significant
  changes, five only at rounding size in cases that must be identical (the `SHRFDF` fundamental's sine, `KSBEND`'s
  path length per step, the host-copy guard under `TWISS`, `SCATTER`'s time through a delta kick, `CBSCAT`'s s
  column), which a comparison with the GPU reference shows. Three needed a second round: the `GFWIGGLER` mistake
  was planted in a value recomputed before use, the `MHISTOGRAM` fill never read the switch, and the
  `PFILTER` delta window it changed never cut (the case is fixed, above).
- elegantTestSet decks with the new types (31, eight of them also in the routine list: `ionEffects1`, `matter1`-`5`,
  `dscatter1`-`4`, `scatter1`, `2`, `4`, `hkpoly1`-`5`, `mbumper1`, `mrfdf1`-`3`, `stray1`-`3`, `boffaxe1`,
  `emitTransform1`, `2`, `remcor1`, `chargeTest1`, `rfdf5`), against elegant: at the default thresholds 27
  identical; forced, 16 identical and 11 within the envelope (sums in `.fin` files, `MRFDF`'s functions); the
  same four flagged in both modes:
  - `ionEffects1` (324 bunches, 1,000 turns): the centroids differ from pass 3 by a last bit and grow to 2e-8 m.
    elegant itself with `p_central_mev` one ulp larger moves them 1.2e-8 m (sizes 2.2e-8 m against 2.3e-8 m), so
    this is rounding (the device's `exp` in the kick) amplified by the ion instability. With
    `ELEGANT_GPU_ENABLE_IONEFFECTS=0` the deck is identical to elegant.
  - `rfdf5`: `RFTM110`'s CUDA `sin` and `cos`, up to 8e-9 relative in the particle files.
  - `emitTransform1` and `2`: the minimum and maximum of s in the sigma file move by 7e-15. The same with
    `ELEGANT_GPU_ENABLE_EMITTANCE=0`: other elements of the deck on the device, not this port.
- Verify build (`ELEGANT_GPU_VERIFY=1`), on the 78 new cases, the three earlier ones whose sums changed
  (`matter-fallback` among them) and the twelve whose `suite.json` settings changed: all 93 pass, 3,135 element checks at 1e-12. The first run
  failed four, all of them the test's or the comparison's fault: `rmdf-ramp`, `rmdf-invalid` and `cooler-opt-out`
  compared NaN with NaN, and `sample-fraction`'s CPU rerun drew new random numbers.
- Forced routine list (424): all pass; against `results/2026-11-14_items-1-2-4/routine`, 387 unchanged and 37
  changed, all decks with `PFILTER` (`FFTB*`, `LCLS*`, `bunchComp*`, `firstIsFid*`, `scraper1`, `ria*`,
  `trajCorrect2`, `reflectTest`), `SCATTER` (`LCLSII-2`, `losses3`), `DSCATTER` (`ERL2`), `TWISS` (`twissElem*`),
  `EMITTANCE` or `RFTM110` (`maxamp4`), where the beam now stays on the device. Against elegant (a CPU run of the
  37) each test's largest relative difference is the old reference's, except new rounding-size columns (minima
  and maxima of s and t in `.fin` files near 1e-8, `Cdelta` and `s16` of 1e-20 and 1e-23 in absolute terms) and
  the `bunchComp1` and `bunchComp1m` slice-analysis row fixed above; after that fix the four decks with slice
  analysis differ from the old reference only in device sums.
- GPU suite (39): unchanged.
- gpu-Pelegant (rebuilt), `mpiexec -np 4`, on `beamBeam1`, `ringRF2`, `twla1`, `twla2`, `mapSolenoid1` and 13 decks
  with the new types (`ionEffects1`, `matter1`, `2`, `dscatter1`, `scatter1`, `2`, `hkpoly1`, `mbumper1`,
  `boffaxe1`, `emitTransform1`, `remcor1`, `chargeTest1`, `losses3`), against the `-Og` CPU Pelegant: all 18
  identical. The first run stopped `ionEffects1` with a segmentation fault on the master, which holds no
  particles and no bunch index: `trackWithIonEffects()` read the bunch's rows from it in a new `HAVE_GPU` line
  (fixed; the rows are taken only for the device kicks, which gpu-Pelegant never makes).
- CPU findings, logged and not fixed (the device copies elegant): `track_ramp.c` line 41 takes the first
  particle's `beta_z` without the `sqrt` that line 64 has; `matter.c`'s single-scattering sections don't move x
  and y by the particle's slope in a section where it scatters (only by the kick times the rest of the section).

### Speed (feature table 12)

Table 11 with the gpu-elegant cells of the 26 rows measured again with the final binaries (both thread counts;
every elegant cell and every other row is table 11's): `/local/GPU-Testing/history/2026-11-15_feature-timing-12`
(`FEATURES.md`, `FEATURES.pdf`, `speedups-26.md`; runs in `results/2026-11-15_ports-timing`). Speed over elegant at
10,000 / 100,000 / 1,000,000 particles, 1 host thread:

| Row | Table 11 | Now |
| --- | --- | --- |
| `kpoly` | 0.4x / 0.37x / 0.54x | 3.1x / 15x / 34x |
| `hkpoly` | 0.73x / 0.7x / 0.75x | 6.6x / 21x / 31x |
| `mbumper` | 0.89x / 0.89x / 0.91x | 24x / 66x / 79x |
| `rmdf` | 0.9x / 0.93x / 0.95x | 17x / 33x / 37x |
| `mrfdf` | 0.75x / 0.7x / 0.76x | 8.1x / 21x / 27x |
| `shrfdf` | 0.69x / 0.71x / 0.81x | 6.1x / 19x / 29x |
| `rftm110` | 0.74x / 0.67x / 0.73x | 8.7x / 24x / 33x |
| `stray` | 0.91x / 0.9x / 0.92x | 103x / 179x / 181x |
| `alph` | 0.91x / 0.88x / 0.91x | 76x / 131x / 140x |
| `ksbend` | 0.92x / 0.93x / 0.96x | 22x / 44x / 50x |
| `boffaxe` | 0.75x / 0.75x / 0.75x | 28x / 34x / 41x |
| `gfwiggler` | 0.95x / 0.96x / 0.96x | 23x / 44x / 52x |
| `pfilter` | 0.14x / 0.21x / 0.41x | 1.2x / 16x / 108x |
| `sample` | 0.12x / 0.15x / 0.33x | 158x / 186x / 498x |
| `clean` | 0.5x / 0.5x / 0.72x | 1.1x / 1.7x / 3.4x |
| `emittance` | 0.84x / 0.87x / 0.93x | 0.88x / 0.95x / 1x |
| `twiss` | 0.78x / 0.84x / 0.94x | 1.2x / 1.1x / 1.1x |
| `remcor` | 0.36x / 0.62x / 0.73x | 0.94x / 2.4x / 4.1x |
| `mhistogram` | 0.87x / 0.76x / 0.87x | 1.1x / 4.4x / 11x |
| `cooler` | 0.96x / 1x / 0.72x | 0.99x / 1x / 0.72x |
| `scatter` | 0.86x / 0.86x / 0.89x | 0.95x / 1x / 1x |
| `dscatter` | 0.73x / 0.73x / 0.73x | 0.93x / 0.99x / 1x |
| `matter` | 0.83x / 0.82x / 0.84x | 0.9x / 0.97x / 1x |
| `peppot` | 0.23x / 0.28x / 0.58x | 0.62x / 1.4x / 2.4x |
| `cbscat` | 0.9x / 0.94x / 0.96x | 0.93x / 0.94x / 0.93x |
| `ioneffects` | 0.89x / 0.92x / - | 4.2x / 4.6x / - |

- The kicks, deflectors, matrices, field integrators, `SAMPLE`, `PFILTER` from 100,000 particles and `IONEFFECTS` are
  now well above elegant.
- `CLEAN`, `REMCOR`, `MHISTOGRAM` and `PEPPOT` gain with the beam's size but stay modest: each call still adds or
  draws on the host.
- `EMITTANCE`, `TWISS`, `SCATTER`, `DSCATTER`, `MATTER`, `CBSCAT` and the cooler pair are near 1x (they were 0.7x to
  0.96x). Their cost is work elegant does too and that stays on the host for exactness: elegant's random numbers
  drawn in its order, and sums over the beam in particle order. The port saves the beam's trip to the host and
  back, which is what put them below 1x. Going further needs the random numbers or the ordered sums on the device.
- `gfwiggler` at 10,000 first read 16x from the element's own time: the report took the one-cell deck for one on
  the CPU because its first matrix element is charged the CUDA start-up. It is now a wall-clock slope (`gpu_from`
  in `feature_timing.py`), 23x. The report's notes for table 12 are in its `FEATURES.md`.
- Two decks had to be made again: table 11 used `cases/timing-features`, whose `MHISTOGRAM` deck has `INPUT_BINS`
  and whose `MBUMPER` deck a flat waveform; the first runs here used the older `cases/timing-features-11`, where
  `MHISTOGRAM` stops elegant with a segmentation fault and `MBUMPER` idles between pulses.

## Ten more rows on the device, rounding only (afternoon of November 15, 2026)

The user's limit: rounding differences are acceptable, different random numbers are not. So every particle keeps
the uniforms elegant's `random_2` gives it, in elegant's order, and every decision those numbers make (whether a
particle scatters, in which section, whether it is lost, and so the order of the rows) is elegant's; what is
computed from the numbers may differ by rounding. `CBSCAT` was left out: its decision to scatter compares a
computed probability with a uniform, and a rounding difference there could flip it. Work folder
`/local/GPU-Testing/results/2026-11-15_ports2` (sources before the planted mistakes in `source-preplant`).

### What changed

- `TMCF`, `CEPL`, `TWPL`, `TWMTA` on the device's `motion()` integrator (non-adaptive Runge-Kutta, as `RFTMEZ0`
  and `TWLA` since November 14): `gpuMotionDerivatives()` in `gpu_cuda_runtime.cu` takes their four field
  functions (`TWMTA` with its entrance and exit impulses), `motionGpuSetup()` (`motion.cc`) accepts the types,
  `do_tracking.c` sends them through the same case (a drift with `TIME_DEPENDENCE_OFF`, as before), switches
  `ELEGANT_GPU_ENABLE_TMCF`, `_CEPL`, `_TWPL`, `_TWMTA`. CUDA's `sin`, `cos`, `cosh`, `sinh`, `exp`: rounding.
- `FTABLE`: any `N_KICKS`, `L1`, `L2`, `E1`, `E2`, `DX`, `DY`, `DZ`, `TILT` (before: one kick, no frame changes);
  `ftable_frame_converter()`, `offsetBeamCoordinatesForMisalignment()` and `rotateBeamCoordinatesForMisalignment()`
  in the kernel, their sines and cosines from the host. Each kick's sine and cosine of the bend angle are now
  `gpuAccurateSin` and `gpuAccurateCos`.
- `BUMPER` (`KICKER`) as a port (`gpu_ports*.h`, switch `ELEGANT_GPU_ENABLE_BUMPER`): waveform time and amplitude,
  `PERIODIC`, the exact arc, kicks with `B2`, the deflection map, tilt and offsets; `FIRE_ON_PASS` drifts and the
  first call's set-up and fiducial on the host. `gpuAccurateSin`, `Atan`, `Asin`, `Cos`, CUDA's `tan`. Not with spin.
- `BGGEXP` with `SYMPLECTIC=1`: `gpuBggexpSymplecticKernel()` and `gpuBggexpVectorPotential()`
  (`computeGGEVectorPotential()` in the CPU's order of operations, its divisions by the same host-made
  denominators), every `Z_INTERVAL` planes, in a bend's frame with `IS_BEND` (the frame's sines and cosines from the
  host). Normal data, no radiation, particle output or misalignment, as for the other integrator. Switch
  `ELEGANT_GPU_ENABLE_BGGEXP_SYMPLECTIC`. Both integrators share `gpuBggexpIntegerPower()`, unchanged: its exponents
  are never negative, since the device takes normal-field tables, whose m starts at 1. (A reciprocal for negative
  exponents, as `ipow()` has, was tried and made the non-symplectic kernel 18% slower; see the speed section.)
- `SCATTER`, `DSCATTER`, `MATTER`: the device makes the random numbers. dlaran is a multiplicative generator mod
  2^48, so the host walks elegant's sequence with integer steps (`GPU_PORT_NEXT_STATE`, `gpuDlaranJump`) for every
  decision and records where each particle's numbers start; the device makes them from there
  (`gpuPortUniform`, `gpuPortGauss` as `gauss_rn()`, `gpuPortInterp` as `interp()`). Without decision draws (a
  probability of 1, `MATTER`'s multiple scattering without energy decay) particle i's numbers start at a jump of
  i times its count, and the host walks nothing. `MATTER`'s straggling, whose loop ends on the sign of a Gaussian
  number and decides the losses, stays `gauss_rn()` itself on the host. The uniforms are elegant's to the bit,
  so `DSCATTER` and uniform `SCATTER` stay identical; the Gaussian numbers use CUDA's `log` and `sin`.
  `ELEGANT_GPU_PORT_DEVICE_DRAWS=0` makes the numbers on the host again. `DSCATTER`'s `interp()` warnings are
  printed by the host, in elegant's order, when the device counts a failure.
- Files: `gpu_base.h`, `gpu_cuda_runtime.cu`, `gpu_stub.c`, `gpu_ports.h`, `gpu_ports_device.h`, `gpu_ports_host.h`;
  CPU side `do_tracking.c` (case order for the motion types, `BUMPER` dispatch), `motion.cc`,
  `magneticFieldFromGeneralizedGradient.c` (inside `HAVE_GPU`), `track_data.c` (`GPU_SUPPORT` on `TMCF`, `CEPL`,
  `TWPL`, `TWMTA`, `KICKER`). All of it is off in gpu-Pelegant.

### Checks

- Random numbers: in every case and deck below the particles that scatter, the sections with an event, the lost
  particles and the row order are elegant's, and the verify build checks `random_2`'s state after each `SCATTER`,
  `DSCATTER` and `MATTER` call against elegant's (exact in all 43 verify cases). `DSCATTER` and uniform `SCATTER`
  are bit-identical; Gaussian `SCATTER` and `MATTER` differ from elegant by rounding (CUDA's `log` and `sin` in
  `gauss_rn()`'s formula). The nine earlier `MATTER` and Gaussian `SCATTER` feature cases moved from identical to
  rounding for that reason; with `ELEGANT_GPU_PORT_DEVICE_DRAWS=0` (`scatter-host-draws`, `matter-host-draws`)
  they are identical again.
- Feature set, 621 cases (29 new, see its README): against elegant 213 identical, 405 within the envelope, three
  flagged: `scmult-zero-length` as before, and `ftable-quad-kicks` and `ftable-misaligned` (below). The 592 earlier
  cases: identical to `results/2026-11-15_ports/cpu-feature-592` on the CPU (the CPU elegant was rebuilt); on the
  GPU 583 identical to `gpu-feature-592` and the nine above changed by rounding (compared through a view of the new
  run carrying the old run's test-set record, `make_relabeled_view.py`, since new data files and `suite.json`
  entries change the fingerprint). References: `cpu-feature-621`, `gpu-feature-621`.
- `FTABLE` with a weak field is ill-conditioned in elegant itself: the kick's bend angle is the small root of
  GSL's cubic, computed as a difference of numbers of order one, so an ulp of CUDA's `acos` or `cos` against glibc's
  moves it by up to 1e-7 of itself. On `ftable-quad-kicks` the particles differ from elegant by up to 4.6e-13 m;
  elegant with `p_central` one ulp larger moves them 1.8e-13 m (medians 1.4e-18 against 6.1e-18 m), and the same
  holds for `ftable-misaligned` (1.0e-14 against 2.3e-14 m), elegantTestSet's `ftable1` (x 1.7e-15 against
  9.2e-14 m) and `ftable2` (8.8e-11 m median against 8.5e-11 m). The accurate sine and cosine of the angle didn't
  change that (the cubic dominates) and were kept. The strong-field dipole case is within 2e-12.
- Verify build (`ELEGANT_GPU_VERIFY=1`) on the 29 new cases and the 14 earlier `SCATTER`, `DSCATTER` and `MATTER`
  ones: 41 pass at the default 1e-10, 348 element checks; the two `FTABLE` quadrupole cases fail it by the amount
  above and pass with `ELEGANT_GPU_COMPARE_REL=1e-6 ELEGANT_GPU_COMPARE_ABS=1e-8` (largest 1.5e-7 relative in s).
  `BUMPER` checks show no difference at all in `bumper-arc` and `bumper-fire-on-pass`.
- Weak cases found and fixed: every `TWMTA` case lost its whole beam (`BETA_WAVE=1` makes elegant divide by zero
  in the transverse-mode amplitudes; now 0.99 with `KX=2`); a `TMCF` case without `X_MAX` lost everything, and
  elegant's adaptive integrators can't track that cavity (step-size errors), so there is no `TMCF` CPU-options
  case; `bumper-arc`'s first kicker never fired (`TIME_OFFSET` is subtracted from the particle's time), and its
  periodic one never folded (elegant folds only times past the waveform's end).
- Deliberate mistakes: 33 planted behind `ELEGANT_GPU_PLANTED_MISTAKE` in the `sm_120` variant
  (`mistakes/plant2.py`; the switch is defined at the top of `gpu_cuda_runtime.cu`, whose kernels come before
  `gpu_ports_device.h`; sources restored from `source-preplant` and checked by sha256), two to five for each item:
  the four motion types' fields, `FTABLE`'s entrance, offset, untilt and exit, `BUMPER`'s arc, `B2`, map,
  `PERIODIC` and `FIRE_ON_PASS`, the symplectic integrator's potential, iteration, bend frame and path length, and
  the random numbers (the Gaussian formula, the uniform's offset, `interp()`'s bracket, the sign, `MATTER`'s order
  of numbers, the host walks' strides and first states). All 33 move the results far above the unplanted variant's
  rounding (the smallest by 6e-16 in t, 580 times it), and the five smallest were checked to be flagged against
  elegant too. One needed a second round: the `PERIODIC` mistake changed nothing until `bumper-arc` had particles
  past the waveform's end.
- The device's Gaussian numbers: from 200,000 `SCATTER` draws (`XP=1` on a beam at rest, so x' is the number),
  87.5% are elegant's to the bit and the rest differ by 1 or 2 ulps (four by 3), both ways (55% toward the
  smaller magnitude).
- elegantTestSet decks with these types (`bggexp2`, `bumper1`-`3`, `ftable1`, `2`, `scatter1`-`4`,
  `dscatter1`-`4`, `matter1`-`5`), against elegant: at the default thresholds 11 identical, six within the envelope
  (`bggexp2`, `matter1`, `2`, `scatter1`, `2`, `4`) and `ftable1` and `ftable2` flagged (above); forced, 6
  identical, 10 within the envelope, and `ftable1`, `ftable2` and `bumper3` flagged. `bumper3`'s one value is x at
  the kicker's exit, 5.6e-15 m off: elegant's arc takes cos(alpha_f) - cos(alpha_i) of two angles near zero and
  divides it by a curvature of 0.02, so an ulp of one cosine is 5.5e-15. `bggexp2` (`SYMPLECTIC=1`, 1,000
  particles) runs in 2.0 s against elegant's 9.9 s and differs by 1e-15 relative.
- Forced routine list (424): all pass; against `results/2026-11-15_ports/routine`, 414 unchanged and 10 changed:
  `bggexp2`, `bumper2`, `bumper3`, `losses3`, `physAp2`, `trackingInterrupt1`, `fiducial1`, `LCLSII-2`,
  `bunchComp1`, `bunchComp1m`. Against elegant (a CPU run of the 10) each one's differences are the old
  reference's or new ones at rounding size, except `LCLSII-2`, whose centroids moved from 3e-13 to 3e-12 m from
  elegant: its `SCATTER` (`HTRUND`, `DP=1.2e-4`) now has the device's Gaussian numbers. With
  `ELEGANT_GPU_PORT_DEVICE_DRAWS=0` the deck is back at the old reference's distance exactly, and elegant itself
  with `DP` one ulp larger moves the same columns by the same amounts (the GPU run and that nudged elegant run
  agree to 1e-17 m in the centroids): rounding, amplified by the linac.
- GPU suite (39): 38 unchanged, `ftable-fixed-step-heavy` changed by rounding (the accurate sine and cosine; its
  distance from elegant is the same as before).
- gpu-Pelegant (rebuilt; none of the new device paths is on under MPI), `mpiexec -np 4`, on `bumper2`, `bumper3`,
  `ftable1`, `bggexp2`, `scatter1`, `2`, `dscatter1`, `matter1`, `2` and `twla1`, against the `-Og` CPU Pelegant:
  all 10 identical. `bumper1` stops both Pelegants alike (its output file is locked with four ranks).
- Builds: CPU elegant (rebuilt, the non-CUDA build compiles), gpu-elegant, the verify build, the `sm_120` variant
  (rebuilt from the clean sources after the mistakes: restoring the files with their old times had left the
  planted objects in place, so the restored sources were touched and rebuilt; no binary holds the switch) and
  gpu-Pelegant.

### Speed (feature table 13)

Table 12 with the gpu-elegant cells of the 14 rows these changes reach measured again (both thread counts; every
elegant cell and every other row is table 12's): the ten rows above, and `rftmez0`, `twla`, `mapsolenoid` (same
integrator kernel, whose field function took the four new types) and `bggexp` (the power function the symplectic
integrator shares). `/local/GPU-Testing/history/2026-11-15_feature-timing-13` (`FEATURES.md`, `FEATURES.pdf`,
`speedups-14.md`; runs in `results/2026-11-15_ports2-timing-b`; a first version with only the ten rows is kept in
its `first-version`). Speed over elegant at 10,000 / 100,000 / 1,000,000 particles, 1 host thread:

| Row | Table 12 | Now |
| --- | --- | --- |
| `twmta` | 1x / 1x / 1x | 23x / 34x / 34x |
| `tmcf` | 0.99x / 0.99x / 1x | 21x / 33x / 36x |
| `cepl` | 1x / 1x / 0.96x | 30x / 46x / 51x |
| `twpl` | 1x / 1x / 1x | 29x / 45x / 49x |
| `rftmez0` | 26x / 40x / 38x | 26x / 40x / 44x |
| `twla` | 28x / 42x / 36x | 28x / 41x / 40x |
| `mapsolenoid` | 23x / 35x / 35x | 23x / 35x / 36x |
| `bggexp` | 76x / - / - | 76x / - / - |
| `bggexp-symplectic` | 1x / 1x / - | 41x / 70x / - |
| `ftable` | 1x / 1x / 1x | 36x / 95x / 116x |
| `bumper` | 0.96x / 0.95x / 0.98x | 20x / 53x / 65x |
| `scatter` | 0.95x / 1x / 1x | 27x / 102x / 163x |
| `dscatter` | 0.93x / 0.99x / 1x | 8.7x / 59x / 124x |
| `matter` | 0.9x / 0.97x / 1x | 2.8x / 3.6x / 3.4x |

- `MATTER` stays near 3x because its deck has energy decay: the host still reads each particle's momentum and
  decides the losses in elegant's order (the straggling, when on, with `gauss_rn()` itself). Without energy decay
  its multiple scattering needs nothing from the host.
- `bggexp` first measured 18% slower (65x): the power function had a reciprocal for negative exponents, which the
  device's tables never give. Without it the row is table 12's again (0.369 s a call at 10,000). The other three
  rows that share code are within 2% of table 12 up to 100,000 particles and 2% to 15% faster at 1,000,000.
- At 1 and 1,000 particles the rows are within 3% of table 12 or faster; `ftable`, `scatter`, `dscatter` and
  `matter` stay on the CPU there by the small-beam estimate.
- The 1- and 8-thread runs of the 67 new cells are 0.9% apart in the median, never more than 5%.
- The elements that always run on the CPU are down from 12 types to 7 (`BRAT`, `TUBEND`, `POLAR`, `SLICE`,
  `RIMULT`, `BRANCH`, `SCRIPT`), and no row of the table runs on the CPU only because of its settings
  (`CPU_BY_SETTING` in `feature_timing.py` is empty).
- After the power function was restored, the BGGEXP feature cases, elegantTestSet's `bggexp1`-`4` and `bggMoments1`
  (forced) and the GPU suite's `bggexp-deterministic-heavy` are bit-identical to the runs above, and all four GPU
  binaries were rebuilt from the final sources.

## `SLICE` on the device (evening of November 15, 2026)

Of the seven element types still CPU-only after table 13 (`BRAT`, `TUBEND`, `POLAR`, `SLICE`, `RIMULT`, `BRANCH`,
`SCRIPT`), `SLICE` was the one worth moving: `RIMULT` and `BRANCH` do no particle work and already leave the beam on
the device, `POLAR` acts once a run, `SCRIPT` runs an outside program, `TUBEND` loses every particle, and `BRAT`
(adaptive Bulirsch-Stoer through a field map) is a large port whose step choices could make it more than rounding.
In table 13 `SLICE` took 173 ms a call at 1,000,000 particles against elegant's 376 ms, nearly all of it the beam's
copy to the host and the host's sums. Its output had to stay elegant's to the bit, which the device's parallel beam
sums (for `WATCH` and the centroid files) don't give.

### What changed

- `gpu_ports_device.h`: the particles' times and the bunch's earliest and latest (`gpuSlicePointTimeKernel()`: the
  CPU's formula; the first particle with either extreme wins a tie, as the CPU's loop with strict comparisons has
  it; particles outside the PID range take no part); each particle's slices with `sliceRange()`'s binary search
  (`gpuSlicePointMemberKernel()`), a (slice, particle) pair per membership, a stable CUB radix sort by slice, so each
  slice keeps particle order, and the members' seven columns gathered slice after slice into pinned host memory
  (`gpuCudaSlicePointMembers()`).
- `gpu_ports_host.h`: `gpu_slice_point_times()` and `gpu_slice_point_sums()`. The host adds every sum
  `accumulate_beam_sums()` makes for a slice (the seven column sums, then the 15 `BEAM_SUMS_SPARSE` products about the
  means it divides out) in particle order with `KahanPlus()`'s steps, all of a slice's sums in one pass over its
  members, and slices on the host threads when the run asks for threads. `SLICE` is in `gpuPortTypes[]` (switch
  `ELEGANT_GPU_ENABLE_SLICE`).
- `slicePoint.c` (in `HAVE_GPU`, not MPI): with the element on the device, `dump_slice_analysis()` takes the times
  and extremes from the device, makes the slices' lower limits by the CPU's repeated addition, and fills each slice's
  `BEAM_SUMS` from those sums with `accumulate_beam_sums()`'s own final formulas (`sliceSumsFromDevice()`; an empty
  slice gets 0/0 in its sigmas, as on the CPU). A call the device can't slice (no particles, a non-positive or
  vanishing width, which `accumulate_beam_sums()` treats as the whole beam, more memberships than the sort takes)
  gets a read-only host copy and the host's code. The verify build compares the extremes and every slice's count,
  charge, centroids and sigma matrix with the CPU's, exactly.
- `track_data.c`: `GPU_SUPPORT` on `SLICE`. gpu-Pelegant leaves it on the CPU.
- The sums first ran on the device too, one warp a sum (lanes loading 32 values for lane 0), then one thread a
  sum: 133 ms and then 102 ms a call at 1,000,000 particles. A sum in order with Kahan's correction is a chain of
  four dependent double additions a value, and on this device (`sm_120`, double precision at 1/64 rate) that took
  300 to 600 ns a value; ten slices of 100,000 members each are 100,000 links long whatever the parallelism. The
  host runs a slice's 7 and then 15 chains side by side, a few ns a value, and the beam's columns cost a 56 MB copy:
  27 ms a call.

### Checks

- Feature set, 627 cases (6 new `SLICE` cases, see its README): against elegant 219 identical (the new six and the
  two older `slice-element-*` cases, now on the device, among them), 405 within the envelope, the same three flagged
  as before (`scmult-zero-length`, `ftable-quad-kicks`, `ftable-misaligned`). The 621 earlier cases are unchanged on
  the CPU and on the GPU against `cpu-feature-621` and `gpu-feature-621`. The final binary repeats the first
  `gpu-feature-627` run exactly (`gpu-feature-627b`). References: `results/2026-11-15_ports2/cpu-feature-627`,
  `gpu-feature-627b`.
- The device path is taken where it should be: no host copy for `SLICE` in `slice-device-passes`; one for the pass
  of `slice-device-one-time` with a single time, for the one-particle range of `slice-element-ranges`, and for the
  negative-ID `SLICE` of `slice-device-pid-reversed`.
- Verify build: the 10 `SLICE` cases pass, every slice's numbers compared exactly with the CPU's (10 device calls
  in `slice-device-passes` alone).
- Deliberate mistakes: 11 planted behind `ELEGANT_GPU_PLANTED_MISTAKE` in the `sm_120` variant
  (`mistakes/plant4.py`), in the time formula, the extremes, the slice limits' test, the PID range, the members'
  order, the slice width, the means' division, a sigma pair, the time mean and each pass's Kahan correction; all 11
  move the `SLICE` cases off elegant's bits, nine as significant changes and the two dropped Kahan corrections at
  rounding size, in cases that must be identical. The first version (sums on the device) caught all 10 of its own
  (`plant3.py`). The sources were restored and touched, and the variant rebuilt; no binary holds the switch.
- elegantTestSet: `collectiveRing1` (ten `SLICE` elements over PID ranges, 100 turns) identical to elegant, 73 s
  against 181 s; `scRing-nonlinearBunched` (`SLICE` with `BUNCH_SERIES`) has every slice file identical and differs
  only in its centroid and sigma files, which the device sums by design.
- GPU suite (39): unchanged. gpu-Pelegant (rebuilt) on its 10 decks: identical to the `-Og` CPU Pelegant.

### Speed (feature table 14)

Table 13 with the gpu-elegant cells of the `slice` row measured three times and the median taken; everything else is
table 13's: `/local/GPU-Testing/history/2026-11-15_feature-timing-14` (runs in `results/2026-11-15_slice-timing`).
Speed over elegant at 10,000 / 100,000 / 1,000,000 particles: table 13 2.8x / 2.5x / 2.2x; now 4.0x / 7.8x / 11.4x
with 1 host thread (2.0 ms, 3.6 ms and 33 ms a call), 4.3x / 9.8x / 27x with 8. At 1 and 1,000 particles the deck
stays on the CPU and the cells move by up to 2x between runs: the element writes its file every call. Six element
types now always run on the CPU: `BRAT`, `TUBEND`, `POLAR`, `RIMULT`, `BRANCH`, `SCRIPT`.

## Upstream pull `7acd2d7..526689f` (November 16, 2026)

Four upstream commits (Borland, from Kuklev's AI scan and the `n_periods` work) were merged into the uncommitted
working tree with `git merge-file`; nothing was stashed, reset or committed. Backups, the three-way copies and the
runs below are in `/local/GPU-Testing/results/2026-11-16_pull-526689f`.

### What upstream changed and what the GPU needed

- `chrom.c`, `chromDefs.h`, `twiss.cc`: second- and third-order chromaticity and the `CCBEND`/`LGBEND` radiation
  integrals now take `n_periods` into account. Host-only: the device never makes these numbers (`ILMATRIX` takes its
  chromaticities from its own parameters). No GPU change. No elegantTestSet or feature deck sets `n_periods`.
- `csbend.c`: `integrate_csbend_ordn()` now turns the spin with the slopes, `QX` and `QY` times
  `(1 + X/rho0)/sqrt((1 + DPoP)^2 - QX^2 - QY^2)`, instead of `QX` and `QY`. The device body step
  (`gpu_cuda_runtime.cu`, the `gpuSpinUpdateLocalFS()` call in the `CSBEND` kernel, its only call site) makes the same
  factor, operation for operation. `CCBEND`, `LGBEND` and the multipoles already pass slopes; the expanded
  Hamiltonian leaves the spin alone on both sides.
- `lsc.c`: `LSCDRIFT` and `LSCKICK` (RFCW) take the impedance and `kSC` at the beam's momentum,
  `Po*(1 + <delta>)` over the binned particles (`lscBeamMomentum()`), not at `Po`. Upstream's commit also edited
  `gpu_stub.c` and `gpu_cuda_runtime.cu`: one `atomicAdd` of `delta` per particle into a single device double in the
  binning kernel. That adds in a different order on every run, so results would no longer repeat, and it would
  break the bit-for-bit `LSCDRIFT` match with output that this branch keeps. It was replaced:
  - `gpuLscStatisticsKernel` also sums column 5, in its fixed order, in the same pass as x and y
    (`centroidSum[5]`); the binning kernel and `gpuCudaLscBin()` are back to their earlier form.
  - `gpuLscBeamMomentum()` (host): with output in the CPU's order (`gpuLscCpuOrderSums()`, the rule
    `gpuLscCenteredVariances()` already used), column 5 comes to the host and the CPU's sum runs on it in particle
    order; otherwise the fixed-order sum is used. If the binning counted fewer particles than there are (the CPU warns
    "this shouldn't happen"), the time column comes too and the host leaves out what `binTimeDistribution()` leaves
    out. Both device LSC paths already refuse a distributed beam, so `lscBeamMomentum()`'s MPI reductions don't
    arise.
  - The verify shadows use upstream's `gpuLscCpuBeamMomentum()` unchanged.

### Checks

- Merge: the local diff before and after is the same except `gpuCudaLscBin()` (84 files either way).
- Feature set, 627 cases: the new CPU elegant moves 16 cases against `cpu-feature-627` (4 `CSBEND` spin cases and 2
  RFCW LSC cases significantly, 10 LSC cases at rounding size); the new gpu-elegant moves the same 16 against
  `gpu-feature-627b`, with the same classes. GPU against the new CPU: 219 identical, 405 within the envelope, the same
  three flagged as before (`scmult-zero-length`, `ftable-quad-kicks`, `ftable-misaligned`), and the same 219 cases
  identical as before; `lsc-output-cpu-order` and `lsc-output-kick-mode` still give elegant's bits. The old GPU
  numbers in the six significant cases are outside the envelope of the new CPU numbers, so the cases catch a device
  path that misses either change. References: `cpu-feature`, `gpu-feature`.
- Verify build, 20 LSC, RFCW-LSC and `CSBEND` spin cases: all pass; `LSCDRIFT` (particles, `Itime`, `Vtime`) and the
  RFCW LSC kick at `maxAbs=0`, `CSBEND` at rounding size as before.
- GPU suite (39): against `2026-11-15_ports2/suite-slice` only `lsc-heavy` moved, at rounding size; against a new CPU
  suite run 6 identical, 33 rounding, none significant (`cpu-suite`, `gpu-suite`).
- elegantTestSet, the 17 decks with LSC or spin (`CLIC1`, `CLIC2`, `lsc1`-`lsc4`, `ecol3`, `rcol2`, `NLSJitter1`,
  `csrSegment2`, `csrSegment3`, `spinTest1`, `spinTest2`, `polar1`-`polar4`; `spinTest3` has no runnable input),
  default settings, against the new CPU elegant: 12 identical, `lsc4` rounding, 4 flagged. `CLIC1`/`CLIC2` (since
  November 11) and `NLSJitter1` (final `WATCH`, since November 14) are the known amplified-rounding flags. `lsc3` differs
  only in `run.sig`/`run.cen` (device-order sums; fitted beta and extremes) and every particle file is elegant's; with
  `ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS=0` all of `lsc3` is identical to elegant (`cmp-lsc3-hostsums`). `lsc3` 542 s
  on the CPU, 307 s here.
- Not tested separately: whether the fixed-order device sum alone would have broken the exact cases (only the
  host-order path runs there). gpu-Pelegant was not rebuilt; its LSC stays on the CPU with a distributed beam.

## Committed without the CPU fixes (November 16, 2026)

The GPU work was committed without the CPU changes that alter elegant's output, which wait for approval:
fixes 1 to 9 and 11 to 24 and the `CSBEND` drift formula (Part 4) of `elegant-CPU-bugs-and-fixes.md` in the
elegant checkout. The commit has the GPU code, the CPU-file changes that don't change elegant's results
(moved code, helpers, `HAVE_GPU` blocks, loop macros, `GPU_SUPPORT` flags), and fix 10, which changes no output
and keeps rows in order for the device. A preprocessor comparison of every changed CPU file against `HEAD`,
serial and MPI, shows nothing else; the scripts are in
`/local/GPU-Testing/results/2026-11-16_pull-526689f/cpu-audit`, the commit's files and logs in
`/local/GPU-Testing/results/2026-11-16_gpu-commit`.

The GPU code had followed those fixes. So that gpu-elegant gives the committed elegant's results, the commit
holds these temporary rules. Each is marked in the code with the fix it waits for; remove them when the CPU
changes are committed (the working copy doesn't have them):

| Waits for | Rule | Where |
| --- | --- | --- |
| Fix 5 | A `UKICKMAP` with `SYNCH_RAD` stays on the CPU | `gpuKickMapElementSupported()` |
| Fix 6 | An `LGBEND` stays on the CPU when loss rows or global loss coordinates are wanted | `gpuLgbendElementSupported()` |
| Fixes 7, 8 | `SPEEDBUMP` and `TAPERAPR` stay on the CPU when global loss coordinates are wanted | the aperture elements' switch in `gpu_stub.c` |
| Fix 9 | A `CWIGGLER` or `GFWIGGLER` with `DX`, `DY` or `DZ` stays on the CPU | `GWigSymplecticPass()` in `GWigSymplecticPass.c`, `gpuAppleSupported()` in `gpu_ports_host.h` |
| Fixes 12, 13 | A tilted `KQUAD`, `KSEXT`, `KOCT`, `KQUSE` or `DQCOR`, or a zero-length multipole, with spin tracking stays on the CPU | `gpuPackMultipoleTracking()` |
| Fixes 15, 16 | A `CSRCSBEND` body slice that loses a particle is restored and handed to the CPU, which does the rest of the bend | `gpu_track_csbend_csr_body_slice()`: `keepLosses = 0` |
| Fix 18 | `TRWAKE` with `N_BINS`, and an `RFCW` whose transverse wake has `N_BINS`, stay on the CPU (for the `RFCW`, decided before any device setup) | `gpuTrwakeDataSupported()`, `gpuRfcwMatrixWakeElementSupported()`, `gpuRfcwKickWakeElementSupported()` |
| Fix 21 | No batched tune footprint or frequency map for a beamline with `POLYNOMIALSERIES` | `gpu_batched_tune_beamline_supported()`, `gpu_batched_frequency_map_beamline_supported()` |
| Part 4 | `CSBEND` isn't exact: the fused tune program doesn't take it, and CSR with output goes to the CPU by default (`ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT=0` keeps it on the device) | `gpuTuneProgramExactElement()`, `gpuCsrCpuForOrderedOutput()` |

No rule was needed for the others: fixes 1 to 4, 14 and 23 are in code both sides share or that has no
device path; 16's exit losses are handed to the CPU already; 11 differs only where the CPU stops; 17's GPU
search still refuses to run while the leftover wedge is installed; 19 and 22 are Pelegant's, and gpu-Pelegant
runs those elements on the CPU; 24's batched searches keep their own turn record; and 20's device behavior is
unchanged since `HEAD`.

One difference is left on purpose. Fix 6 also changes the `sLost` that `&find_aperture` reports for a particle
already past a limit when it reaches a magnet (a lattice without apertures): the CPU reports the distance the
particle before it in the same call left, which depends on how the CPU groups particles, while the batched
search on the device reports the magnet's entrance. Matching it would mean keeping the batched aperture search
off for nearly every lattice, so `search-aperture-entrance-loss` stays flagged until fix 6 is committed.

Without the drift formula, `CSBEND` on the device agrees with elegant to rounding, as before November 11, 2026,
and outputs that amplify rounding (CSR histograms and their derivatives, fitted twiss parameters, tunes from
NAFF) move by more than the comparison's envelope.

Also found: when an `RFCW` falls back to the CPU after its device setup has run, elegant's own setup runs again
and stops with "You can't give wakeFile along with trWakeFile or zWakeFile". The fix-18 rule is decided before
setup, so no deck reaches this now; the fallback is in the working copy too and isn't fixed.
### Checks (`/local/GPU-Testing/results/2026-11-16_gpu-commit`)

- The commit's CPU elegant against `HEAD`'s (built in a clean worktree): elegantTestSet, 660 tests, all identical
  except `photonEmission1`'s `dictionary.sdds`, where `GPUCapable` now reads 1 for the device elements. Feature set:
  the 620 cases both complete are identical. Three more cases complete only with the commit: the `KSBEND` and
  `POLYSERIES` loss cases, whose output files are identical but `HEAD` aborts at exit (fix 10). Four cases stop
  on both, as fixes 11 and 14 describe.
- gpu-elegant against the commit's CPU elegant. Feature set (623 cases that complete on both): 215 identical,
  368 within the envelope, 40 flagged: the three known ones (`ftable-misaligned`, `ftable-quad-kicks`,
  `scmult-zero-length`), `search-aperture-entrance-loss` (above), and 36 that contain a `CSBEND`, `CSRCSBEND` or
  `CSRDRIFT`. With `CSBEND` and CSR kept on the CPU (`ELEGANT_GPU_ENABLE_CSBEND_DRIFT=0`,
  `ELEGANT_GPU_ENABLE_CSR_DRIFT=0`, `ELEGANT_GPU_ENABLE_CSR_RESIDENT_DRIFT=0`) none of the 36 is flagged. GPU suite
  (39): 6 identical, 32 within the envelope, `csbend-heavy` flagged at rounding size (4e-15 in `Cx`).
  elegantTestSet, default settings: 514 identical, 49 within the envelope, 97 flagged. 87 of the 97 are also
  flagged by the working copy's gpu-elegant against its own CPU elegant; the other ten (`csbend1`, `csbend7`,
  `csbendAp1`, `csbendAp2`, `dynAp4`, `dynAp5`, `fseCSBEND1`, `multipoles3`, `rfdf4`, `twoCavityMoments1`) are
  not flagged with `CSBEND` on the CPU.
- All builds (elegant, Pelegant, gpu-elegant, gpu-Pelegant) compile without warnings. The verify build and
  gpu-Pelegant runs weren't repeated for the commit.
