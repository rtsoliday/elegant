# GPU test set

This standalone suite contains fixed, GPU-heavy elegant workloads.  The cases
cover matrix tracking, high-order multipoles, CSBEND tracking, longitudinal
space charge, coherent synchrotron radiation, nonlinear transverse space
charge, combined impedance and wake tracking, and RF cavities.  They are based
on the repository's earlier CUDA
benchmark fixtures, with larger workloads and without order-sensitive particle
dumps that would force CPU fallback.

Run CPU and GPU artifact sets serially so tests do not contend for cores or the
GPU:

```sh
python3 src/gpu/scripts/elegant_test_regression.py baseline \
  --test-set src/gpu/test-set --jobs 1 \
  --elegant bin/Linux-x86_64/elegant \
  --output GPU-Testing/gpu-test-set-cpu

python3 src/gpu/scripts/elegant_test_regression.py baseline \
  --test-set src/gpu/test-set --jobs 1 \
  --elegant bin/Linux-x86_64-gpu/gpu-elegant \
  --output GPU-Testing/gpu-test-set-gpu-pre-change

# Build the candidate, then create a third completed artifact set.
python3 src/gpu/scripts/elegant_test_regression.py baseline \
  --test-set src/gpu/test-set --jobs 1 \
  --elegant bin/Linux-x86_64-gpu/gpu-elegant \
  --output GPU-Testing/gpu-test-set-gpu-candidate

python3 src/gpu/scripts/elegant_test_regression.py compare-existing \
  --baseline GPU-Testing/gpu-test-set-cpu \
  --pre-change-gpu GPU-Testing/gpu-test-set-gpu-pre-change \
  --candidate GPU-Testing/gpu-test-set-gpu-candidate \
  --target-test scmult-nonlinear-heavy \
  --output GPU-Testing/gpu-test-set-comparison
```

The GPU suite automatically performs one warm-up and five measured runs per
test.  It records every sample plus the median and median absolute deviation in
`manifest.json`, and writes a compact `baseline-summary.json` with executable
hashes, the test-set fingerprint, hardware/driver information, GPU activity,
and synchronization/fallback evidence.  If the initial relative MAD exceeds
5%, the harness collects five additional samples.

The comparison requires every candidate case to report GPU activity and checks
output significance.  Targeted cases must be at least 2x faster than CPU and,
when `--pre-change-gpu` is supplied, at least 2x faster than the immediately
preceding GPU build.  Tests not named with `--target-test` are performance
guards: they may not regress by more than 5%, and total suite time may not
regress by more than 2%.  Omitting `--target-test` targets every case.  The
threshold can be overridden with `--minimum-speedup`.

The suite configuration supplies the required `ELEGANT_GPU_*` settings
automatically and records them in each run manifest, so a GPU run cannot
silently measure CPU fallback.  For centroid and sigma outputs, the suite uses
a `1e-15` absolute floor and `1e-9` relative envelope to screen expected GPU
numerical variation.  Ordinary `elegantTestSet` comparisons retain the stricter
global defaults.

`scmult-nonlinear-heavy` is the targeted case for the deterministic, unsliced,
single-bunch nonlinear SCMULT path.  It deliberately uses unequal horizontal
and vertical beam sizes so the benchmark exercises the complex-error-function
field calculation rather than only the round-beam shortcut.  Sliced and
multi-bunch nonlinear SCMULT remain CPU fallbacks.

`impedance-heavy` and `cwake-heavy` are deterministic single-bunch workloads
for the combined IMPEDANCE and CWAKE elements.  Both exercise longitudinal,
dipole, and quadrupole channels with compact centroid and sigma output.  Their
analytical impedance and wake tables are local to the suite, so the cases do
not depend on or modify `elegantTestSet`.

`impedance-multibunch-heavy` and `cwake-multibunch-heavy` repeat this coverage
for five input bunches and exercise both all-bunch and partial-bunch selection.
They share the deterministic, down-sampled five-page SDDS beam in
`multibunch-data`; machine-specific timing artifacts remain outside the suite.

`cwake-fft-heavy` uses 5000 beam bins and a 5000-point wake table so the
resident, zero-padded cuFFT convolution path is exercised independently of the
smaller direct-convolution CWAKE case.

`frequency-map-batched-heavy` tracks a 65-by-65 frequency-map grid for 128
turns through the high-order POLYNOMIALSERIES map.  The grid points are
independent particles with persistent particle IDs, allowing the complete grid
to be tracked together while the original per-particle coordinate histories
and output row order are retained.  `run-aps-fallback.ele` verifies that an APS
beamline outside the currently validated subset retains point-by-point CPU
tracking rather than accepting amplified tune differences.

`dynamic-aperture-batched-heavy` exercises deterministic n-line dynamic
aperture refinement through GPU-supported multipoles and a compact aperture.
All line/step trials for one refinement level are tracked together with stable
search IDs; loss coordinates and loss-pass metadata are restored by ID after
particle compaction. `run-small-fallback.ele` covers the below-threshold CPU
path.

`momentum-aperture-batched-heavy` searches both momentum directions at 24
lattice locations through deterministic CSBEND and multipole cells. Trials
share one tracking call per refinement level and receive their momentum offset
at the requested lattice location using stable search IDs. The resulting loss
coordinates and pass numbers are mapped back to the original search rows.
Tune-history/resonance-crossing mode remains on the CPU and is covered by
`run-resonance-fallback.ele`. Searches containing synchrotron-radiation
damping or third- and higher-order CSBEND terms also retain point-by-point CPU
tracking until dedicated search baselines validate their amplified boundary
and tune sensitivity.

`tune-footprint-batched-heavy` exercises batched chromatic and transverse tune
footprints through a high-order POLYNOMIALSERIES map.  The directory also
contains `run-linear.ele` for the lower-order map and
`run-small-fallback.ele` for below-threshold CPU-fallback verification.  Only
`run.ele` is included in the timed suite.  Batched history analysis uses up to
16 independent CPU workers after the resident GPU tracking pass; the NAFF
implementation is thread-local and produces deterministic per-grid results.
The suite uses a 32-particle magnet threshold so point-at-a-time and small-grid
POLYNOMIALSERIES tracking remains on the CPU, while each main batched grid is
large enough to exercise the CUDA path.

`rfdf-deterministic-heavy` tracks 32768 particles for 256 passes through a
21-kick RFDF element with a voltage waveform.  Fiducial and waveform
calculations stay on the host and the per-particle drift/kick integration is
performed by CUDA.  `run-noise-fallback.ele` verifies that voltage- and
phase-noise modes remain on the CPU, and `run-small-fallback.ele` covers the
particle-count threshold.

`bggexp-deterministic-heavy` tracks 4096 particles through the 2001-point Q98
generalized-gradient field table using the non-symplectic predictor-corrector
integrator.  The supported CUDA subset is a straight, aligned, normal-field
BGGEXP with `SYNCH_RAD=ISR=0`, no particle-output file, and `Z_INTERVAL=1`.
The immutable generalized-gradient coefficients are uploaded once and reused.
Companion inputs cover symplectic, ISR, particle-output, and below-threshold
CPU fallbacks.

`magnet-synchrad-heavy` tracks 16384 particles through repeated `KQUAD`,
`CSBEND`, `KICKMAP`, and `UKICKMAP` elements with deterministic classical
synchrotron-radiation damping.  The supported CUDA subset requires
`SYNCH_RAD=1, ISR=0`; stochastic ISR remains a CPU path.  The paired kick-map
elements use local analytical SDDS maps and the main workload writes only a
centroid-mode `WATCH` file.  Companion inputs cover ISR, below-threshold, and
map-loss behavior.

`sreffects-deterministic-heavy` tracks 65536 particles through repeated
deterministic `SREFFECTS` elements.  The main workload covers damping and
average loss with and without dispersion offsets, damping-only behavior, and
loss-only behavior while retaining the particle array on the GPU.  Companion
inputs keep `QEXCITATION=1` and below-threshold workloads on the CPU.  The
target seeds nonzero horizontal and vertical dispersion so the optional orbit
offset branch is exercised with nonzero corrections.

`bmxyz-fixed-step-heavy` tracks 1024 particles through six fixed-step `BMXYZ`
elements using the shared double-precision Q98 field grid.  The supported CUDA
subset uses non-adaptive Runge--Kutta integration, first-order trilinear field
interpolation, `SYNCH_RAD=0`, no particle diagnostic or aperture-contour file,
and a resident immutable field map.  Companion inputs cover synchrotron
radiation, particle diagnostics, and the below-threshold CPU fallback.

`ccbend-deterministic-heavy` tracks 32768 particles through repeated,
pre-optimized positive-angle `CCBEND` elements.  The supported CUDA subset is
order-2, hard/no-fringe tracking with normal dipole, quadrupole, and sextupole
terms; it excludes radiation, spin, misalignment, multipole-error files,
centroid files, partial-element tracking, and soft fringe models.  Reference
trajectory and path-length corrections computed during host optimization are
applied on the GPU.  Companion inputs cover the below-threshold and soft-fringe
CPU fallbacks.

`bmapxy-fixed-step-heavy`, `nibend-fixed-step-heavy`, and
`nisept-fixed-step-heavy` exercise the fixed-step Lorentz-family integrator.
The CUDA subsets are respectively file-backed bilinear `BMAPXY`, hard-edge
positive-angle `NIBEND` without radiation or trajectory adjustment, and
positive-angle linear-fringe `NISEPT`.  Each target uses deterministic compact
centroid and sigma output.  The `run-small-fallback.ele` companions cover the
particle-count threshold; adaptive BMAPXY/NISEPT and radiation-enabled NIBEND
companions verify unsupported-option CPU fallback.

`rfmode-resident-heavy` and `frfmode-resident-heavy` track 65536 particles for
2048 passes through deterministic single-bunch longitudinal resonator modes.
The evolving cavity phasors remain on the host, while CUDA performs the time
coordinate evaluation, bucket validation, histogram construction, and
longitudinal energy kicks using persistent scratch buffers.  The host sums the
resident time coordinates in CPU particle order before updating the mode state.
Companion inputs cover binless or mode-output diagnostics and the
below-threshold CPU fallback.  The measured production crossover threshold is
8192 particles.  Generator feedback, noise, multiple physical bunches, and MPI
remain CPU paths.

`tfeedback-deterministic-heavy` tracks 262144 particles for 500 passes through
a deterministic single-bunch transverse `TFBPICKUP`/`TFBDRIVER` pair.  CUDA
performs the pickup coordinate reduction and the independent driver kicks,
while the small filter and delay histories remain persistent on the host.
Companion inputs cover pickup noise, frequency-dependent driver behavior, and
the below-threshold CPU fallback.  Set `ELEGANT_GPU_ENABLE_TFEEDBACK=0` to
disable the path explicitly; `ELEGANT_GPU_MIN_TFEEDBACK_PARTICLES` controls
the crossover threshold.

`touschek-resident-heavy` keeps Touschek random scattering and particle
selection on the CPU, then tracks the selected batch through a deterministic
ring containing `KQUAD`, `SREFFECTS`, and `RFMODE` elements.  Two hundred and
fifty-six downstream `TSCATTER` markers reproduce dense production sampling:
after the selected distribution has been generated, later markers are tracking
no-ops and retain the current particle array on the GPU instead of breaking
residency.  The main
case processes only the first scattering location and writes the initial
scattered bunch, loss distribution, and longitudinal loss histogram.  The
companion `run-loss.ele` applies a tight final aperture to exercise lost-row and
particle-ID handling.
