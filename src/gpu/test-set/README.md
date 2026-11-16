# GPU test set

This standalone suite contains fixed, GPU-heavy elegant workloads.  The cases
cover matrix tracking, high-order multipoles, CSBEND tracking, longitudinal
space charge, coherent synchrotron radiation, nonlinear transverse space
charge, combined impedance and wake tracking, and RF cavities.  They are based
on the repository's earlier CUDA
benchmark fixtures, with larger workloads and without order-sensitive particle
dumps that would force CPU fallback.

The Tkinter benchmark launcher is available with
`python3 src/gpu/scripts/elegant_test_regression.py gui`. Quick Check screens
output and GPU activity with one run per case. Timed Guard retains the serial
repeated-sample protocol below and can use a prior GPU artifact for slowdown
guards. The manual commands remain available in the GUI's Manual launcher tab.

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

Each case's `run.ele` is its workload.  Other decks in a case folder check
fallbacks, loss handling and smaller variants.  The harness runs them only when
asked, each as a test named `CASE__DECK`: add `--include-extra-decks
--correctness-only` to a `baseline` command, or name one after its options.
They are not timing workloads.  `extra_deck_exclusions` in `suite.json` leaves
out `mode2.ele`, which serial CPU `elegant` cannot run.

`gpu_activity_exempt_tests` in `suite.json` lists tests that need not show GPU
activity: `frequency-map-batched-heavy` and the `*-fallback` extra decks.
`comparison.txt` still names each exempt test that showed none.

Small feature and fallback checks are in `../feature-set`.

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
and output row order are retained.  `run-aps-fallback.ele` is a 3-by-3 map
through an APS lattice of kick quadrupoles, kick sextupoles and matrix bends:
nine points are below the batch threshold, so it stays on point-by-point CPU
tracking.  Since November 11, 2026 a larger grid through that lattice is
tracked on the device by the fused program, with output identical to the
CPU's (elegantTestSet `fmap3`).

`frequency-map-resident-magnets-heavy` extends the batched frequency-map path
to deterministic GPU-supported magnet lattices and enables
`include_changes=1`, which tracks two consecutive tune intervals without
returning to point-by-point tracking.  Only particles with a valid first tune
enter the second interval, with results scattered back into stable grid order.
For non-MPI `gpu-elegant`, `-ompThreads=N` explicitly enables experimental
OpenMP particle tracking in the loss-sensitive batched CPU fallback used by
frequency-map and momentum-aperture searches.  The default is one tracking
thread.  Since November 12, 2026 the environment variable `OMP_NUM_THREADS`
means exactly the same as `-ompThreads`: there is one count of host threads
for the tracking loops, the host thread pool, the tune analysis, the
batched-search switch, OpenMP's own default and MKL (unless `MKL_NUM_THREADS`
is set).  The command line wins over the environment, so `-ompThreads=1`
forces one thread; with neither, gpu-elegant uses one.  A start-up line on
stderr reports the count and where it came from when it is more than one.
Loops that do only a few operations a particle (wake and cavity-mode kicks,
time coordinates) use threads from 100,000 particles;
`ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES=64` lowers that so the tests' small
beams run them.  Batched momentum searches use the checked CUDA LGBEND path even with
`ELEGANT_GPU_ENABLE_LGBEND=0`; this keeps the search
ensemble resident and avoids repeated CPU-island synchronization.  OpenMP is
used only when an element or option still requires CPU tracking.  The OpenMP
path keeps element setup and sticky aperture state serial,
keeps one worker team alive across the complete batched tracking call, submits
deterministic magnet integration and exact drifts as particle taskloops, and
performs stable ordered loss compaction after each taskloop.  Compaction
preserves the fixed host row addresses required by CUDA synchronization while
moving complete particle records into stable survivor/loss order.  Stochastic
ISR remains on the serial fallback.  Momentum-aperture boundary confirmation
is also grouped into one CPU ensemble after the CUDA search, allowing the
requested OpenMP threads to process CPU-only magnet formulas without
restarting the tracker for every boundary point.  Numerically sensitive
boundary mismatches still rerun the complete scalar lane.  Set
`ELEGANT_GPU_BATCH_MOMENTUM_BOUNDARY_REPLAY=0` to restore point-by-point
boundary confirmation for diagnosis.  Set
`ELEGANT_GPU_PROFILE_MOMENTUM_BOUNDARY_REPLAY=1` to emit element timing for
the grouped CPU boundary confirmation.  The regression harness
accepts repeatable executable options such as
`--elegant-argument=-ompThreads=4`.  Users must use OpenMP only on a GPU node
whose requested CPU cores are not shared with other jobs.
`run-loss.ele` checks the loss-sensitive batched CPU policy, stable grid-row
mapping after entrance losses, and the defined `s=0` output for lost particles.
`run-isr-fallback.ele` confirms that ISR and other unsupported magnet options
retain the scalar point-by-point fallback.
`run-fused-opcodes.ele` is a compact GPU_VERIFY companion for the internal
frequency-map program. It covers exact and matrix drifts, thin and thick
multipoles, exact correctors, CSBEND/CCBEND/LGBEND, circular and rectangular
sticky apertures, collimators, a scraper, SPEEDBUMP, zero-voltage RFCA, passive
elements, and fixed-slot loss bookkeeping. `run-fused-isr-fallback.ele`
verifies that stochastic radiation rejects the program before tracking.

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
`mode2.ele` is an explicit parity case for the fixed-grid mode-2 algorithm in non-MPI
`gpu-elegant`. CUDA is preferred for eligible elements; `-ompThreads=N`
supplies the accepted loss-sensitive CPU fallback. Set
`ELEGANT_GPU_MOMENTUM_BATCH_PARTICLES` below the full grid size to test stable
ID accounting across batch boundaries. It intentionally does not use the
`run*.ele` naming convention because the suite's serial CPU baseline does not
support mode 2; compare this case separately with CPU `Pelegant`. For example,
use `ELEGANT_GPU_MODE=off ELEGANT_GPU_MOMENTUM_BATCH_PARTICLES=65 gpu-elegant
-ompThreads=1 mode2.ele` and repeat with `-ompThreads=4`.
Set `ELEGANT_GPU_MOMENTUM_BOUNDARY_VERIFY=1` to confirm each reported
fixed-grid boundary with CPU/OpenMP arithmetic and rescan only directions whose
boundary classifications disagree. This additional correctness check is
disabled by default because it can substantially increase runtime.
Setting `ELEGANT_GPU_MOMENTUM_BATCH_PARTICLES=1` exercises the singleton CPU
fallback while retaining fixed-grid mode-2 semantics. In
`ELEGANT_GPU_MODE=required`, singleton batches continue to use CUDA instead.
Tune-history/resonance-crossing mode remains on the CPU and is covered by
`run-resonance-fallback.ele`. Searches containing synchrotron-radiation
damping or third- and higher-order CSBEND terms also retain point-by-point CPU
tracking until dedicated search baselines validate their amplified boundary
and tune sensitivity.

`tune-footprint-batched-heavy` exercises batched chromatic and transverse tune
footprints through a high-order POLYNOMIALSERIES map.  The directory also
contains `run-linear.ele` for the lower-order map and
`run-small-fallback.ele` for below-threshold CPU-fallback verification.  Only
`run.ele` is included in the timed suite.  Batched history analysis uses the
16 CPU workers the suite asks for with `ELEGANT_GPU_BATCHED_TUNE_THREADS`.
Without that setting it uses one thread unless the run sets `OMP_NUM_THREADS`
or `-ompThreads` (before November 11, 2026 it started up to 16 by itself).  The
NAFF implementation is thread-local and produces deterministic per-grid results.
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
Since November 15, 2026 `SYMPLECTIC=1` also runs on the device, with any
`Z_INTERVAL` and in a bend's frame (feature-set `bggexp-symplectic*` cases).
The immutable generalized-gradient coefficients are uploaded once and reused.
Companion inputs cover ISR, particle-output, and below-threshold CPU
fallbacks; the 4-particle symplectic companion now stays on the CPU by the
particle threshold.

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
loss-only behavior while retaining the particle array on the GPU.  A companion
input runs `QEXCITATION=1`, which also runs on the GPU above the threshold, and
another keeps a below-threshold workload on the CPU.  The
target seeds nonzero horizontal and vertical dispersion so the optional orbit
offset branch is exercised with nonzero corrections.  Its `SREFFECTS` numbers
are made on the host, as `gauss_rn_lim()` makes them, which takes one thread
unless the run asks for more, so `suite.json` sets
`ELEGANT_GPU_ISR_REFERENCE_THREADS=8`.  On one thread the case was 1.67x faster
than the CPU, under the 2x target; on 8 it was 7.3x.

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

`lgbend-deterministic-heavy` covers the validated order-2 LGBEND CUDA subset,
with a threshold of 64 particles rather than the default of 512.  The CUDA path
runs the CPU's optimization on its reference particle before the first pass, so
every pass runs on the device.  LGBEND has been on by default since November 6, 2026, when this deck
ran 2.8 times faster than the CPU on an RTX PRO 4000 Blackwell.  An older test on
an RTX 3060 showed 0.98--1.02x at 32768 to 262144 particles, so
`ELEGANT_GPU_ENABLE_LGBEND=0` keeps LGBEND on the CPU.

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

The `run-clock.ele` companion combines a serial `RFCA,CHANGE_T=1` producer
with the supported single-bunch CUDA `RFMODE` histogram and kick path.  Its RF
frequencies are deliberately non-harmonic so the resonator phase depends on
the removed whole-period offsets accumulated by the tracking clock.

`tfeedback-deterministic-heavy` tracks 262144 particles for 500 passes through
a deterministic single-bunch transverse `TFBPICKUP`/`TFBDRIVER` pair.  CUDA
performs the pickup coordinate reduction and the independent driver kicks,
while the small filter and delay histories remain persistent on the host.
Companion inputs cover pickup noise, frequency-dependent driver behavior, and
the below-threshold CPU fallback.  Since 2026-11-07 the noise companion
(`run-noise-fallback.ele`) uses the device too: the noise is drawn on the host
after the device average, as on the CPU, so the name is historical.  Set `ELEGANT_GPU_ENABLE_TFEEDBACK=0` to
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
