# GPU parity validation

For output validation without a performance claim, add `--correctness-only`
to each `baseline` command. This explicitly selects zero warm-ups and one run
per test, permits `--jobs 8`, and records that timing gates are unavailable.
`compare-existing` retains the numerical and CUDA-activity checks and preserves
a failed coverage report even if a case ran entirely on CPU. Do not combine
these artifacts with `--minimum-speedup` or `--pre-change-gpu` timing gates.
The default serial timing workflow in `test-set/README.md` is unchanged.

The current `frequency-map-batched-heavy` implementation explicitly selects its
CPU history backend. It can validate outputs, but currently fails the suite's
CUDA-activity gate. Keep this limitation visible until native histories are
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

Non-expanded CSBEND sector drifts use differences of nearly equal cosines.
A host/device libm difference of about an ulp is multiplied by the bend radius;
GPU_VERIFY localized early position differences of order 1e-15 to 1e-14 m.
These can grow in collective tracking. Until the transcendental implementation
is reconciled, ordinary reference-sensitive output uses the CPU element path
for these bends. Expanded-Hamiltonian CSBEND remains GPU eligible.
`ELEGANT_GPU_ENABLE_CSBEND_DRIFT=1`, required GPU mode, and GPU_VERIFY retain the
CUDA implementation so it remains available for validated looser envelopes
and verification. This restriction is not a claim of pure-GPU parity.

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

Local evidence is in
`GPU-Testing/momentum-nonfiducial-20260915/parity-final/`: the raw
`cpu-comparison/`, `original-gpu-comparison/`, `reviewed-acceptance.json`, and
`review_trajcorrect.py`. No additional outputs were ignored.

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

Full logs, frozen binaries, source patches, comparisons and manifests are in
`GPU-Testing/momentum-nonfiducial-20260915/`. The final report is
`proposedChanges/gpu-parity-fixes-20260916.md`.

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
[full-suite report](../../proposedChanges/full-parity-20260916.md) and
`GPU-Testing/full-parity-20260916-161533/REPORT.md`.

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
existing metadata exclusions. Detailed validation and frozen binary are in
`GPU-Testing/twissElem5-fix-20260916-174447/`. The historical full-suite report
above is retained unchanged; this focused correction is not another full-suite
rerun.

The nine focused legacy cases (twissElem1–5, modulate4, twoCavityMoments1,
acSextupole1 and branch4) all match fresh CPU outputs exactly. RFCA-heavy
passes its unchanged companion tolerance. GPU_VERIFY twissElem5 and CPU,
GPU, GPU_VERIFY and GPU/MPI builds pass; MPI runtime was not rerun.
