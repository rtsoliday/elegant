# GPU feature set

This suite holds 627 small CPU/GPU parity cases, each guarding one GPU feature or fallback. They
take about 90 s on the GPU and under 10 s on the CPU. The GPU is slower here because the cases are too
small to benefit from it, so this suite has no timing role. Timed workloads are in `../test-set`.

93 cases come from `test/gpu_cuda/cases`, deleted in commit `112e160`. They keep that tree's
`--quick` particle and pass counts. The other 421 were hand-written, 31 of them on 2026-11-04 to
reach GPU code that coverage showed no test reached, 7 to guard bug fixes the same day, 2 for the `&obstruction_data` fallback, 18 for multipole files, 8 for `MALIGN_METHOD`, 11 for `CWIGGLER` files, 1 for loss positions, 5 for `CCBEND` `FRINGEMODEL=0`. Files shared by several cases are in `data/`, 3 for the CSR wake and `el` fixes, 2 for `LSCDRIFT` with no time spread and `POLYNOMIALSERIES`, 27 for `CCBEND` options, `MALIGN_METHOD` 3 and 4, and `CWIGGLER` pole factors, constant fields and offsets, and 25 on 2026-11-06 for `LGBEND` misalignments, `CCBEND` ISR, centroid output, spin and apertures, `CWIGGLER` ISR, `FIELD_OUTPUT`, `GAP` and `TGU`, and paths no test reached, and 20 the same day for `ISR` on `KQUAD`, `KSEXT` and `CSBEND`, the device's random numbers, more `LGBEND` options, and `CWIGGLER` `TILT` with spin, 2 for `LGBEND` with no optimization done before its first pass, 5 for `APCONTOUR` `Logic`, large `LGBEND` `ISR` tables and RPN random numbers, 13 for `APCONTOUR` on the device and centroids with large tables, and 14 for open-sided `MAXAMP`, `SREFFECTS` with excitation, losses past the first 256 particles and centroid-only sums, and 2 on 2026-11-07 for `pmaximum` fiducialization when the first particle has the most momentum, and 3 the same day for the fused prepass's losses in bends and a scraper, a beam kept on the device below the thresholds, and `WATCH` sums with output files, and 6 more for RPN after a step that lost every particle, and spin in multipoles and `CSBEND`, and 6 more for `RAMPP`, `FMULT` with and without spin, pickup noise, and feedback in bunched beam mode with one and two bunches, and 1 for `WATCH` sums with no output files, and 1 for helpers that match the CPU exactly with output in CPU order, and 3 on 2026-11-08 for exact drifts with an output file, spin `WATCH` sums, and the batched tune footprint's history on the device, and 1 for momentum matching with output in CPU order, and 3 for `RAMPRF` on the device, and 2 for `RAMPRF` with `pmaximum`, 2 for `CSBEND` losses inside the bend, and 3 for a ring that loses its beam through the 1,000 floor, and 18 more on 2026-11-08 for options that ran only on the CPU, `CSBEND` edge models 2 to 4, slice analysis and spin in zero-length multipoles, and 4 for `CSRCSBEND` apertures, and 9 for `BMXYZ` apertures, contours in the resident aperture search and `CSRCSBEND` losses with output, and 9 on 2026-11-11 for `IMPEDANCE`, `CWAKE` and `LSCDRIFT` on the device with output in CPU order, momentum matching that skips a beam it has already matched, and the host thread pool, and 8 the same day for the choice each tracking call makes for a beam under the 10,000 thresholds, and 12 more for tune histories tracked by the fused program, and 8 on 2026-11-13 for magnet losses with a losses file, and 4 the same day for wakes with a set number of bins and no output in the CPU's order, and 2 for `CSRCSBEND` on the device with output in the CPU's order, and 2 for `CSRDRIFT` with CSR on the device, and 12 on 2026-11-14 for `ZLONGIT` and `ZTRANSVERSE` on the device, and 8 the same day for `IBSCATTER`, and 11 for `LSRMDLTR`, and 5 for `LRWAKE`, and 4 for `CORGPIPE` and `CORGPLATES`, 6 for `TAPERAPE` and the tilted `TAPERAPR`, and 6 for `CENTER` and idle passes, and 26 on the evening of 2026-11-14 for `BEAMBEAM`, `RFTMEZ0`, `TWLA`, `MAPSOLENOID` and `MODRF`, and 78 on 2026-11-15 for the 27 types (26 items, the cooler's two counted as one) that moved from the CPU to the device, `KPOLY` to `IONEFFECTS`, and 29 later that day for `TMCF`, `CEPL`, `TWPL`, `TWMTA`, `FTABLE`'s kicks and frames, `BUMPER`, `BGGEXP` with `SYMPLECTIC=1`, and the random numbers of `SCATTER`, `DSCATTER` and `MATTER` made on the device, and 6 for the `SLICE` element on the device.

## Running

From the repository root:

```sh
python3 src/gpu/scripts/elegant_test_regression.py baseline --test-set src/gpu/feature-set \
  --elegant bin/Linux-x86_64/elegant --output CPU_ARTIFACT --jobs 8 --correctness-only
python3 src/gpu/scripts/elegant_test_regression.py baseline --test-set src/gpu/feature-set \
  --elegant bin/Linux-x86_64-gpu/gpu-elegant --output GPU_ARTIFACT --jobs 1 --correctness-only
python3 src/gpu/scripts/elegant_test_regression.py compare-existing \
  --baseline CPU_ARTIFACT --candidate GPU_ARTIFACT --output COMPARISON_ARTIFACT
```

Pass case names after the `baseline` options to run only those cases.

`suite.json` supplies the GPU settings, and they are recorded in each run's manifest:

- Thresholds of 64 (RFMODE 8192) and `ELEGANT_GPU_MODE=required`. The `obstruction-data` cases use
  `auto` through `gpu_environment_by_test`, because `required` mode stops on them by design. A
  setting given with `--environment` takes priority over this.
- The WAKE, RFCW and RFCA change-p0 opt-ins, because these cases exist to test those paths. `LGBEND` is on by default, and the suite still sets it explicitly with a threshold of 64.
- CSR, CSR resident, LSC and CSBEND set explicitly, and short GPU islands allowed. Without these,
  order-sensitive output and short beamlines keep several cases on the CPU.

Screening uses the same 1e-15 absolute and 1e-9 relative envelope as `../test-set`.
`../PARITY-VALIDATION.md` explains why the absolute tolerance must not be raised globally.

Since 2026-11-11 `IMPEDANCE`, `CWAKE` and `LSCDRIFT` run on the device with order-sensitive output
too, and must then match the CPU exactly: `impedance-output-cpu-order`, `impedance-output-bunches`,
`cwake-output-cpu-order`, `cwake-output-autoscale`, `lsc-output-cpu-order` and `lsc-output-kick-mode`
request `output` for that, and nothing else: `centroid`, `sigma` and `final` files hold sums the device
makes in its own order, which would hide a lost last bit in the particles behind expected rounding.
`cwake-output-cpu-order` puts a 37.5 m drift before its `CWAKE`, because the order of the mean-time sum
only shows when the times are far from zero. `match-energy-skip-output` (`always_change_p0` with an
`RFCA` that leaves the central momentum alone), `sreffects-host-threads` and `rfca-host-threads` (beams
large enough for the host thread pool, with `ELEGANT_GPU_ISR_REFERENCE_THREADS=4` from `suite.json`,
the test-only setting the test set's `SREFFECTS` case uses; outside the tests the pool gets threads
only from `OMP_NUM_THREADS` or `-ompThreads`, and the harness takes only `ELEGANT_GPU_*` settings
per test) are built
the same way. All nine must stay identical to the CPU; each failed under a deliberate mistake (see
"`IMPEDANCE`, `CWAKE` and `LSCDRIFT` on the device with output in CPU order" in
`../PARITY-VALIDATION.md`). The older LSC cases don't request `output`. `lsc-backtrack-fallback`
stays on the CPU anyway, by rule.

Since 2026-11-12 `ILMATRIX` runs on the device, with the sines and cosines of its phase advances from the
host's libm when output is in CPU order, and must then match the CPU exactly: `ilmatrix-basic`,
`ilmatrix-amplitude`, `ilmatrix-tilt`, `ilmatrix-losses`, `ilmatrix-with-matrix`, `ilmatrix-host-threads` and
`ilmatrix-ring-turn` request `output` (and `ilmatrix-losses` the `losses` and `acceptance` files, for the order
of the lost particles) and nothing else. `ilmatrix-losses-global-coordinates` stays on the CPU by rule.
`ilmatrix-device-math` sets `ELEGANT_GPU_ILMATRIX_REFERENCE_MATH=0` in `suite.json` and is expected within
rounding, not identical. The `SLICE` element, which gpu-elegant filled in one pass over the beam on the host
(and since 2026-11-15 fills on the device, see "SLICE on the device" below), is in
`slice-element-ranges` and `slice-element-quantized-times`; both must stay identical. The ten exact cases each
failed under deliberate mistakes, 18 in all (see "`ILMATRIX` on the device and the `SLICE` element in one pass"
in `../PARITY-VALIDATION.md`). `ilmatrix-with-matrix` can't prove where the constant terms come from: the
element's matrix holds the same numbers as the element.

## Current result

On 2026-11-15, with `sm_120`: 591 of the 592 cases are within the roundoff envelope, 207 of them identical to the CPU, and 1 is flagged, so
`compare-existing` exits 1 (references `results/2026-11-15_ports/cpu-feature-592` and `gpu-feature-592`). Of the 78
cases added that day, 76 are identical and `rftm110-cavity` and `gfwiggler-harmonics` differ by rounding. Treat a new flag, or a larger difference in one of these, as a regression. A repeat GPU
run with the same binary gave identical results.

| Case | Difference | Status |
| --- | --- | --- |
| `scmult-zero-length` | `Cx` differs by 4e-14 m | Not a bug. With `sigma_s=0` the kick depends on `(s - mean)/sigma_s` and `sigma_s` is about 1e-9 m, so a 1-ulp change in the mean matters. Reversing the particle order on the CPU moves `Cx` by 6.6e-15. This case guards against the exit that a zero bunch length used to cause |

`aperture-scraper-global-loss`, `aperture-scraper-accepted-offsets`, `magnet-kickmap-accepted-compaction`,
`magnet-lgbend-loss-compaction` and `magnet-ukickmap-loss-compaction` were flagged for row order until the device
rebuilt the CPU's loss order round by round (kick, segment, or entrance then exit) on 2026-11-08. Run any loss case
with `ELEGANT_GPU_CHECK_LOSS_ORDER=1` to check that builder against the CPU's loops. Since 2026-11-13 it also checks
the builder for losses in one round, which no longer walks the beam on one thread, against the walk; the 422 cases
of that day pass with it on.

`magnet-csbend-misalignment` and `csr-short-bunch` were flagged until `gpu_accurate_trig.h` was added.

`aperture-taper-accepted` and `aperture-taper-global-loss` (a `TAPERAPR` with `TILT=0.04`) had particle files a
last bit off the CPU's, inside the envelope, until survivors' path length was taken in the tilted frame on
2026-11-14; they are now identical.

ISR, centroid records and `FIELD_OUTPUT` run on the device, but only plain tracking uses them: batched programs refuse these options. To test the OpenMP paths, run the cases with `--elegant-argument=-ompThreads=4` or with `OMP_NUM_THREADS=4` (the two mean the same), and with `ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES=64` so the cheap loops, which otherwise wait for 100,000 particles, run too; every case must give its one-thread output. All 37 match the CPU exactly: every batched aperture and momentum search now re-checks its boundary trials on the CPU.

For `GPU_VERIFY` runs of this suite, set `ELEGANT_GPU_VERIFY=1` as well as using the verify binary. Without it, the
checks under `gpuBase.verifyMode` in `gpu_stub.c` don't run.

The 78 cases added on 2026-11-15 pass with the verify binary (with `ELEGANT_GPU_VERIFY=1`), as do the earlier cases
whose outputs or settings changed that day. The verify comparison counts NaN against NaN as a match, since elegant
removes such particles after the element on both sides.

With the verify binary, all 327 complete with no stops, and so do the 17 added later on 2026-11-08; the 18th, `magnet-kquad-spin-zero-length`, and `slice-analysis-whole-beam` with its new `s_start` complete too, as do the three `search-momentum-*` cases added on 2026-11-09, which match the CPU exactly. All 14 `search-frequency-map-*` and `search-tune-footprint-*` cases, `search-frequency-map-prepass-monitors` among them, match the CPU exactly under the verify build, and the three `*-csbend-high-order` cases, the two `*-radiation` searches and `search-momentum-generic` pass it. All 17 `search-aperture-*` cases do too, identical to the CPU. Their outputs against the CPU: 294 identical, 32 within the
envelope, and `scmult-zero-length` flagged. Since 2026-11-08 the verify binary also checks the `MAXAMP`,
aperture-data and `APCONTOUR` checks after each element, aperture losses compacted on the device (survivor counts and
rows, paired by `particleID` for `SCRAPER`), fiducial times, and `MATR` and `EMATRIX` reference times.

## Cases

### Matrix and helpers (16)

| Case | Guards |
| --- | --- |
| `matrix-exact-drift` | `EDRIFT` exact-drift tracking |
| `matrix-helpers` | Matrix-family elements plus `MALIGN`, `CENTER`, `ENERGY`, centroid, and beam-sum helper paths |
| `matrix-center-time` | `CENTER,T=1` time-coordinate centering plus nearby matrix/helper paths |
| `matrix-special-elements` | Special-cased matrix elements (`SOLE`, `ROTATE`, `HKICK`, and `VKICK`) that can stay GPU-resident when spin tracking and corrector radiation are inactive |
| `matrix-extended-elements` | Additional matrix-tracking elements (`QUFRINGE`, `MAGNIFY`, `WIGGLER`, `LTHINLENS`, `LMIRROR`, `BEDGE`, `REFLECT`, and non-radiating combined `KICKER`) |
| `matrix-residency` | Matrix-tracking elements separated by coordinate-neutral `CHARGE`, `RECIRC`, `MARK`, `TRCOUNT`, and `FLOORELEMENT` elements |
| `matrix-fiducial-modulate` | `&modulate_elements` on a quad `K1` with `refresh_matrix=1` |
| `matrix-octu-rben-alph` | `OCTU`, `RBEN` and `ALPH`, which no other GPU test uses |
| `matrix-energy-element` | `ENERGY` with `CENTRAL_ENERGY`, `CENTRAL_MOMENTUM`, `MATCH_PARTICLES` and `MATCH_BEAMLINE` |
| `matrix-corrector-radiation-fallback` | `HKICK` and `VKICK` with `SYNCH_RAD=1`, which stay on the CPU. Guards the `gpu_addCorrectorRadiationKick` stub, which exits if reached |
| `reduction-centroid-only` | A `centroid` file with no `sigma` file, and a parameter `WATCH` that reuses the same sums |
| `watch-parameter-sums-output-files` | Parameter and centroid `WATCH` sums on the device with `centroid` and `sigma` files, as in default runs. The suite's explicit output-reduction threshold is unset for this case, so the default rule applies |
| `watch-parameter-sums-no-output-files` | Parameter and centroid `WATCH` sums on the device with no `centroid` or `sigma` file, by the default rule. Before 2026-11-07 they copied the beam to the CPU each pass |
| `matrix-helpers-ordered-output` | `MALIGN` with `DP` and `DE`, `EMATRIX` with `DELTAP`, `MATR` with a chirp, so each reference time matters, and `ENERGY` with a given momentum, with `output`, `final` and a coordinate `WATCH`, by the default rule. These use the device, and `run.out` and the `WATCH` file match the CPU exactly. `run.fin`'s statistics are device sums, so they differ by rounding |
| `matrix-energy-match-ordered-output` | `ENERGY` with `MATCH_PARTICLES` and `MATCH_BEAMLINE`, and `always_change_p0` matching after every element, with `output`, `final` and a coordinate `WATCH`, by the default rule. With output in CPU order the average momentum is the CPU's Kahan sum in particle order, so these use the device, and `run.out` and the `WATCH` file match the CPU exactly |
| `rampp-waveform` | `RAMPP` with a momentum waveform: the device makes each particle's time and rescales the momenta, between RF and `KQUAD`s |

### Apertures and losses (48)

| Case | Guards |
| --- | --- |
| `aperture-apcontour` | The `APCONTOUR` element with `L=0`, offset, tilt and scale factors, stopping part of the beam on the device |
| `aperture-apcontour-invert` | `INVERT=1`, so the shape is an obstruction |
| `aperture-apcontour-logic` | `Logic` expressions on the element |
| `aperture-apcontour-length` | `L>0`: a beam that passes the entrance and leaves the contour by the exit, which falls back to the CPU to find where, then a contour that stops nothing, so survivors drift on the device. A beam also lost at the entrance can't show a mistake in the exit check. Two contours in one lattice also catch a stale upload |
| `aperture-apcontour-output` | `output`, so particle order must match the CPU |
| `aperture-apcontour-accepted` | An acceptance file, so accepted coordinates must follow the survivors |
| `aperture-apcontour-losses` | A losses file, so lost rows must reach the host with their position and momentum |
| `aperture-apcontour-sticky` | A sticky contour, checked after each drift and matrix element on the device, then cancelled |
| `aperture-apcontour-dz` | A sticky contour with `DZ`, `DX` and `TILT`. The beam diverges and is cut at twice its size, so `x` alone never leaves the contour but `x + DZ*xp` sometimes does: a check that ignored `DZ` would find no loss. A loss sends the check to the CPU, which also moves the lost particles' `s` |
| `aperture-apcontour-hold-off` | A sticky contour with `HOLD_OFF` and `L>0`: the element only drifts, on the device, and the elements after it check the contour on the device |
| `aperture-apcontour-spin` | A sticky, tilted contour with spin, checked inside `KQUAD`s and after each element, so survivors and their spins are compacted together |
| `aperture-apcontour-popl` | `Logic` with `popl`, which drops the second contour's result, so the beam is kept in the large box, not the small one |
| `aperture-mixed-acceptance` | `KQUAD`s with rectangular and elliptical collimators, particle loss and acceptance output |
| `aperture-maxamp-rect-loss` | Tight rectangular `MAXAMP` after GPU-resident matrix elements, with a `losses` file |
| `aperture-maxamp-ellipse-loss` | Tight elliptical `MAXAMP` after GPU-resident matrix elements, with a `losses` file |
| `aperture-maxamp-ellipse-loss-no-output` | Tight elliptical `MAXAMP` loss case without a `losses` file, exercising loss-tail synchronization gating |
| `aperture-rcol-noloss` | Loose rectangular collimators with GPU entrance/exit predicates and GPU drift |
| `aperture-rcol-loss` | Tight rectangular collimators with particle loss |
| `aperture-ecol-noloss` | Loose elliptical collimators with GPU entrance/exit predicates and GPU drift |
| `aperture-ecol-loss` | Tight elliptical collimators with particle loss |
| `aperture-scraper-noloss` | Loose one-sided `SCRAPER` checks with GPU entrance/exit predicates and GPU drift |
| `aperture-scraper-global-loss` | One-sided `SCRAPER` with global loss-coordinate output |
| `aperture-scraper-two-sided-global-loss` | Two-sided ideal `SCRAPER` with global loss-coordinate output |
| `aperture-rcol-open-zero-length-loss` | Zero-length open-sided rectangular collimators with particle loss |
| `aperture-rcol-open-global-loss` | Nonzero-length open-sided rectangular collimators with global loss-coordinate output |
| `aperture-ecol-mixed-exponent-loss` | Mixed-exponent elliptical collimators with particle loss |
| `aperture-ecol-open-global-loss` | Open-sided mixed-exponent elliptical collimators with global loss-coordinate output |
| `aperture-data-loss` | Interpolated physical aperture data with particle loss |
| `aperture-remove-invalid` | Fixed SDDS-beam input with invalid coordinates passed through an identity `RFCA` |
| `aperture-taper-global-loss` | `TAPERAPC` and `TAPERAPR`, tilted and offset, with global loss coordinates |
| `aperture-taper-accepted` | The same tapers with acceptance output |
| `aperture-speedbump-global-loss` | `SPEEDBUMP` from `-x` and `y`, with global loss coordinates |
| `aperture-speedbump-accepted` | The same speed bumps with acceptance output |
| `aperture-scraper-accepted-offsets` | `SCRAPER` with offsets and acceptance output |
| `aperture-maxamp-open` | Rectangular `MAXAMP` with each `OPEN_SIDE` on the device, stopping part of the beam on the other sides |
| `aperture-maxamp-open-ellipse` | The same with an elliptical `MAXAMP` |
| `aperture-maxamp-open-losses` | Both shapes with a losses file. The CPU doesn't extrapolate `z` when a side is open, so the lost rows must agree |
| `aperture-maxamp-open-output` | Both shapes with `output`, so particle order must match the CPU |
| `aperture-maxamp-open-accepted` | Both shapes with an acceptance file, so accepted coordinates must follow the survivors |
| `aperture-maxamp-open-global-loss` | Both shapes with global loss coordinates |
| `aperture-apcontour-tail`, `aperture-scraper-tail`, `aperture-remove-invalid-tail` | Losses only past the first 256 particles, so the loss count has to look at the whole beam. A beam that also loses an early particle can't show a count that stops too soon |
| `resident-threshold-crossing` | 12,000 particles on the device, lost at collimators that tighten along the line, through the 10,000 threshold and the 1,000 floor. Runs in auto mode with the default thresholds, so elements below their thresholds stay on the device while the beam is there |
| `resident-threshold-crossing-output` | `resident-threshold-crossing` with an `output` file. Collimator lengths are exact drifts, which match the CPU on the device, so they keep the resident rule with output in CPU order. Below the 1,000 floor, `CCBEND` stays on the CPU when the `MAXAMP` check after it would send the beam back |
| `resident-floor-ring` | A ring that loses its beam a little each pass: `SREFFECTS` takes the same energy from every particle and an `ECOL` where there is dispersion scrapes it. 12,000 particles, below the 10,000 thresholds after the first pass and below the 1,000 floor after about 22, with `KQUAD`, `KSEXT`, `CSBEND`, `MULT` and a `WATCH` file on the device while the beam is there. Auto mode with the default thresholds |
| `resident-floor-ring-output` | `resident-floor-ring` with output and a losses file, so the survivors' order and the lost rows must match the CPU through the floor |
| `resident-floor-ring-final` | `resident-floor-ring` with only a `final` file. No beam sums read the beam back first, so the element after the floor brings it to the CPU itself. A build that skipped that copy fails here |

### Misalignment methods (15)

`MALIGN_METHOD` 1 to 4 run on the GPU for `KQUAD`, `KSEXT`, `KOCT`, `DQCOR` and `CSBEND`, using the CPU's own matrices and
frames. `CCBEND` has its own cases below. `LGBEND` always uses the CPU's exact method for misalignments; see its cases under Magnets.

| Case | Guards |
| --- | --- |
| `magnet-malign-method1`, `magnet-malign-method2` | `KQUAD`, `KSEXT`, `KOCT` with offsets, pitch, yaw and tilt, alone and together, and with no errors |
| `magnet-malign-csbend` | `CSBEND` with both methods, positive and negative angles and every error, next to the original method |
| `magnet-malign-options` | With multipole files, radiation, fringes, `LEFFECTIVE`, `BORE` and kicks |
| `magnet-malign-dqcor` | `DQCOR` with both methods |
| `magnet-malign-loss` | Losses at a sticky aperture inside misaligned magnets, with loss and acceptance output |
| `magnet-malign-exact-method3`, `magnet-malign-exact-method4` | The same as the first two, with the exact methods |
| `magnet-malign-exact-csbend` | `CSBEND` with both exact methods, including `TILT` |
| `magnet-malign-exact-dqcor` | `DQCOR` with both exact methods |
| `magnet-malign-exact-options` | The exact methods with multipole files, radiation, fringes, `LEFFECTIVE`, `BORE` and kicks |
| `magnet-malign-exact-loss` | Losses at a sticky aperture inside magnets with the exact methods |
| `magnet-malign-exact-mixed` | Methods 2, 3 and 4 side by side |
| `search-aperture-malign-exact` | The exact methods in the batched aperture search |
| `search-aperture-malign` | The batched `&find_aperture` search |

### Multipole files (20)

`KQUAD`, `KSEXT` and `KOCT` with `SYSTEMATIC_MULTIPOLES` and `RANDOM_MULTIPOLES`, and `KQUAD` and `KSEXT` with
`EDGE_MULTIPOLES` and `STEERING_MULTIPOLES`, run on the GPU, as do all `DQCOR` and `CCBEND` files and `FMULT`. The files are in
`data/`, except `FMULT`'s, which are in its cases.

| Case | Guards |
| --- | --- |
| `magnet-multipole-files-systematic` | Systematic files with orders 0 to 13, integration orders 2, 4 and 6, and a multipole factor |
| `magnet-multipole-files-random` | Random files, drawing the same numbers in the same order as the CPU over three steps |
| `magnet-multipole-files-edge-steering` | Edge and steering files, alone and with body files and kicks |
| `magnet-multipole-files-misalignment-radiation` | Files with `DX`, `DY`, `DZ`, `TILT`, `FSE`, fringes, `BORE`, `LEFFECTIVE` and `SYNCH_RAD` |
| `magnet-multipole-files-loss` | Losses at a sticky aperture inside magnets, with loss and acceptance output |
| `magnet-multipole-files-undefined-slopes` | Losses from undefined slopes inside a magnet, with loss output |
| `magnet-multipole-files-order-limit-fallback` | An order above 30 uses the CPU for that magnet, and other magnets stay on the GPU |
| `search-aperture-multipole-files` | The batched `&find_aperture` search, with random errors |
| `search-frequency-map-multipole-files` | The batched `&frequency_map`, with random errors |
| `magnet-multipole-files-dqcor` | `DQCOR` quadrupole systematic and random files and dipole steering files, with kicks, factors, `TILT` and misalignment |
| `magnet-ccbend-multipole-files` | `CCBEND` body and edge files, with `FRINGEMODEL` -1 and 1, both bend directions, integration orders 2, 4 and 6, `EDGE1` and `EDGE2`, reversed elements and order limits |
| `magnet-ccbend-multipole-files-reference` | `REFERENCE_ORDER` 0, 1 and 2 in both bend directions |
| `magnet-ccbend-multipole-files-random` | Random files, drawing the same numbers in the same order as the CPU over three steps, beside a `KQUAD` that also draws |
| `magnet-ccbend-multipole-files-radiation` | `SYNCH_RAD`, so the radiation sees the file kicks |
| `magnet-ccbend-multipole-files-loss` | Losses at a `MAXAMP` aperture, with loss and acceptance output |
| `magnet-ccbend-multipole-files-term-limit-fallback` | More than 96 nonzero terms uses the CPU for that bend, and other bends stay on the GPU |
| `search-frequency-map-ccbend-dqcor-files` | The batched `&frequency_map` |
| `search-aperture-ccbend-dqcor-files` | The batched `&find_aperture` search |
| `magnet-fmult` | `FMULT` normal and skew terms with `TILT`, offsets, `FSE`, `FACTOR`, `SYNCH_RAD`, `N_KICKS` and a thin `FMULT`, between `KQUAD`s |
| `magnet-fmult-spin` | `FMULT` with spin, `TILT`, offsets and radiation |

### CCBEND options (28)

`CCBEND` runs on the GPU from its first pass, with `K3` to `K8`, `TILT`, `YAW`, misalignments, `ISR`, spin, centroid output,
aperture files and `APCONTOUR`. The CPU still finds the FSE and `x` offset, on its own reference particle. Each case but the
two `-loss` fallbacks below and the last three catches a deliberate mistake in the kernel. In two of the last three
the kernel isn't reached: `find_aperture` is below the `CCBEND` threshold, and loss output falls back to the CPU.
Those only show that the path still works with these options. The third, the batched frequency map, was tracked on
the CPU until 2026-11-11. The fused program tracks it now, with the CPU's bits, and a verify build stops if a
particle differs (see "Tune histories from the fused program").

| Case | Guards |
| --- | --- |
| `magnet-ccbend-first-pass` | One pass, so each bend is optimized in its only pass |
| `magnet-ccbend-no-optimize` | `OPTIMIZE_FSE=0`, `OPTIMIZE_DX=0`, so the offsets stay 0 |
| `magnet-ccbend-reoptimize` | `K1` and `ANGLE` changing between steps |
| `magnet-ccbend-high-orders` | `K3` to `K8` in both bend directions, with integration orders, fringe models, `COMPENSATE_KN`, `FSE` and files |
| `magnet-ccbend-high-orders-radiation` | `SYNCH_RAD`, so the radiation sees those kicks |
| `magnet-ccbend-tilt` | `TILT` at the angles the CPU special-cases and general angles, both bend directions |
| `magnet-ccbend-yaw` | `YAW`, which makes the two planes differ, both directions and reversed |
| `magnet-ccbend-misalignment-method0` to `magnet-ccbend-misalignment-method4` | `DX`, `DY`, `DZ`, `ETILT`, `EPITCH`, `EYAW`, alone and together, with `TILT` and `YAW`. `ETILT` alone applies only at the exit, as on the CPU |
| `magnet-ccbend-tilt-loss-compaction` | Losses at a `MAXAMP` aperture with `TILT` and `DX`, without loss output, so the kernel decides who is lost |
| `magnet-ccbend-misalignment-loss` | `DX` and `DY` with loss and acceptance output |
| `magnet-ccbend-aperture-data` | An `&aperture_data` file with off-center limits, taken at each magnet's center, with `DX` and `TILT` |
| `magnet-ccbend-apcontour` | A sticky `APCONTOUR` with its own offset, tilt and scale factors, and an offset, tilted magnet |
| `magnet-ccbend-apcontour-invert` | `INVERT=1`, so the inside of the shape is the obstruction |
| `magnet-ccbend-apcontour-multiple` | Two contours, with particles required to be inside both |
| `magnet-ccbend-apcontour-logic` | `Logic` expressions, keeping particles inside either of two rectangles and outside either of two holes. The kernel runs them as stack operations |
| `magnet-ccbend-isr` | `ISR`, with and without radiation, both signs of `ANGLE`, `TILT` and misalignments. A CPU element with `ISR` follows, so the random numbers must stay in step |
| `magnet-ccbend-isr-loss` | `ISR` with losses, which redo the bend on the CPU with the same random numbers. Can't show a kernel mistake, since the bend always falls back |
| `magnet-ccbend-centroids` | `CENTROID_OUTPUT_FILE` with radiation, `ISR`, `TILT`, both signs of `ANGLE`, misalignments and high orders |
| `magnet-ccbend-centroids-loss` | `CENTROID_OUTPUT_FILE` with losses, which use the CPU for that bend. Can't show a kernel mistake in the records |
| `magnet-ccbend-spin` | Spin with fringes, radiation, order 6, `TILT`, both signs of `ANGLE`, `YAW`, misalignment methods 0 and 4, high orders and `FRINGEMODEL=0` |
| `magnet-ccbend-spin-loss` | Spin with losses, so survivors and their spins are compacted on the device |
| `magnet-ccbend-tilt-loss` | `TILT` with loss and acceptance output, which falls back to the CPU |
| `search-frequency-map-ccbend-options` | The batched `&frequency_map` with these options |
| `search-aperture-ccbend-options` | `&find_aperture` with these options |

### CCBEND FRINGEMODEL=0 (5)

The default `CCBEND` fringe model, a `y'` kick at each edge from `HGAP`, `FINT1`, `FINT2`, `K1` and `K2`. The bends have a large
`ANGLE` and `K2`, so a wrong coefficient at any `EDGE_ORDER` would show far above roundoff.

| Case | Guards |
| --- | --- |
| `magnet-ccbend-fringe-model-0` | Both bend directions, `HGAP` with `FINT1` and `FINT2`, `EDGE_ORDER` 0 to 3, integration orders 2, 4 and 6, `COMPENSATE_KN`, files, and reversed elements, which swap `FINT1` and `FINT2` |
| `magnet-ccbend-fringe-model-0-radiation` | `SYNCH_RAD`, so the exit kick sees the energy after radiation |
| `magnet-ccbend-fringe-model-0-loss` | Losses at a `MAXAMP` aperture, with loss and acceptance output |
| `search-frequency-map-ccbend-fringe-model-0` | The batched `&frequency_map` |
| `search-aperture-ccbend-fringe-model-0` | The batched `&find_aperture` search |

### CWIGGLER files and options (18)

`CWIGGLER` with `BY_FILE` and `BX_FILE`, normal or split-pole, runs on the GPU, as does `TILT`. These
cases don't request `final`, because it makes the CPU find each wiggler's matrix by tracking 11,249
particles, which would dominate their time. The files are in `data/`.

| Case | Guards |
| --- | --- |
| `magnet-cwiggler-files-normal` | Normal `By` and `Bx` files, `kx` and `ky` of zero, below `GWIG_EPS`, below and above `kw`, both integration orders, 64 harmonics, beside a sinusoidal wiggler |
| `magnet-cwiggler-files-split-pole` | Split-pole `By` and `Bx` files, and one of each form together |
| `magnet-cwiggler-files-phase` | First-harmonic phases with and without `FORCE_MATCHED`, and adjacent wigglers with their end flags off |
| `magnet-cwiggler-files-radiation` | `SYNCH_RAD` with every field form, which uses the harmonic field calculation |
| `magnet-cwiggler-tilt` | `TILT` at the angles the CPU special-cases and a general angle, with files and sinusoidal |
| `magnet-cwiggler-files-vary` | Length and field changing between steps. Length changes the wavenumbers, so a stale harmonic table would show |
| `magnet-cwiggler-files-divided` | `&divide_elements`, which changes periods, length and end flags |
| `magnet-cwiggler-pole-factors` | `POLE1_FACTOR` to `POLE3_FACTOR` with sinusoidal fields and files, both integration orders, radiation and `TILT` |
| `magnet-cwiggler-constant-field` | `BX_CONSTANT` and `BY_CONSTANT` with sinusoidal fields and files, both integration orders, radiation and `TILT` |
| `magnet-cwiggler-offsets` | `DX`, `DY` and `DZ`, alone and together, with sinusoidal fields and files, both integration orders, radiation, `TILT`, pole factors and constant fields |
| `magnet-cwiggler-output-isr-mixed` | `FIELD_OUTPUT` and `ISR`, each on its own wiggler, beside a wiggler with neither |
| `magnet-cwiggler-tgu-gradient` | `TGU_GRADIENT` changes the field even with `TGU=0`, on sinusoidal and file-based wigglers |
| `magnet-cwiggler-tgu` | `TGU=1` with gradient and compensation, radiation, `TILT`, offsets and pole factors |
| `magnet-cwiggler-gap` | Field from `GAP` and an undulator model instead of `B_MAX` |
| `magnet-cwiggler-isr` | `ISR` with sinusoidal fields and files, both integration orders, radiation, `TILT`, offsets and a pole factor of zero, which skips random numbers. A CPU element with `ISR` follows, so the random numbers must stay in step |
| `magnet-cwiggler-field-output` | `FIELD_OUTPUT` with sinusoidal fields and files, both integration orders, radiation, `ISR` and `TILT` |
| `search-frequency-map-cwiggler-files` | The batched `&frequency_map`, with `TILT` and split poles |
| `search-aperture-cwiggler-files` | The batched `&find_aperture` search, with `TILT` and split poles |

### ISR random numbers and more options (30)

The device makes `ISR` random numbers itself, the same sequence `random_2` would have made, starting from
`random_2`'s state, and moves `random_2` past the numbers each element used when the element finishes. Most `ISR` cases follow the magnet with `R: HKICK,...,ISR=1`, which
stays on the CPU, so the sequence must be in the right place afterward.

| Case | Guards |
| --- | --- |
| `magnet-kquad-isr` | `KQUAD`, `KSEXT`, `KOCT` and `DQCOR` `ISR`, with and without `SYNCH_RAD`, integration orders 2, 4 and 6, misalignments and `TILT` |
| `magnet-kquad-isr-loss` | The same with `MAXAMP` stopping part of the beam. The magnet falls back to the CPU, so this tests only the fallback's numbers |
| `magnet-csbend-isr` | `CSBEND` `ISR`, with and without `SYNCH_RAD`, integration orders 2, 4 and 6, both signs of `ANGLE`, `K1`, `K2`, edges, `EXPAND_HAMILTONIAN`, `TILT` and misalignments. Without `SYNCH_RAD`, the kernel must still take the path that uses velocity |
| `magnet-csbend-isr-loss` | The same with losses. Tests only the fallback's numbers |
| `magnet-lgbend-isr` | `LGBEND` `ISR`, with and without `SYNCH_RAD`, misalignments and `TILT`. The draws go by segment, then particle |
| `magnet-lgbend-isr-loss` | The same with losses. Tests only the fallback's numbers |
| `magnet-lgbend-centroids` | `CENTROID_OUTPUT_FILE` with radiation, `ISR`, misalignments and `TILT` |
| `magnet-lgbend-spin`, `magnet-lgbend-spin-loss` | Spin through each segment and its fringes. In the `-loss` case the aperture stops part of the beam, so survivors and their spins are compacted on the device |
| `magnet-lgbend-aperture-file` | An `&aperture_data` file with off-center limits that change along the line, so each segment has its own aperture |
| `magnet-lgbend-apcontour` | A sticky `APCONTOUR` with no `Logic` expression, with the contour and the magnet offset and tilted |
| `magnet-lgbend-apcontour-logic` | The same with `Logic` expressions, using `\|\|`, `!` and `&&` |
| `magnet-lgbend-isr-segments` | `magnet-lgbend-isr` under `ELEGANT_GPU_LGBEND_ISR_TABLE_DRAWS=1024`, so the bend runs a segment at a time in batches. Its results must be identical to that case's |
| `magnet-lgbend-isr-segments-centroids` | `magnet-lgbend-centroids` a segment at a time. Every output file must be identical to that case's, including the records |
| `magnet-lgbend-isr-segments-loss` | The same with a loss in a later segment, and a drift on the GPU before the bend, so the beam must be restored from before the magnet. A loss in the first segment, or a CPU element just before the bend, doesn't need the restore, so this can't catch a mistake with a wider beam or without the drift |
| `magnet-lgbend-first-pass` | One pass with no `twiss_output`, so the GPU path must run the CPU's optimization itself. Catches skipping it, and `FSE` left out |
| `magnet-lgbend-no-optimize` | `OPTIMIZE_FSE=0`, which used to keep `LGBEND` on the CPU in every pass |
| `magnet-ccbend-radiation-options` | `USE_RAD_DIST`, `ADD_OPENING_ANGLE` and `SR_IN_ORDINARY_MATRIX`, which tracking ignores |
| `magnet-cwiggler-tilt-spin` | `TILT` with spin, after a `CCBEND` that turns the spin out of `z`. A tilt does nothing to a spin along `z` |
| `isr-small-batches` | Every element with `ISR`, with centroid and field output, under `ELEGANT_GPU_RECORDED_BATCH_BYTES=4096`, so the results must not depend on the batch size |
| `isr-small-batches-sreffects` | `SREFFECTS` with excitation under the same small batches, so each batch must track its own particles |
| `isr-small-batches-loss` | The same with a few tail particles lost, so a later batch has a loss and the earlier batches must be restored before the CPU redoes the magnet. A lost particle in the first batch doesn't reach this, so a wider beam tests less |
| `isr-gaussian-limit`, `isr-gaussian-no-limit` | `SR_gaussian_limit=1`, which rejects a third of the numbers, and `0`, which rejects none |
| `isr-loss-before-first-draw` | A first step that stops every particle at the entrance of the first `ISR` magnet, so the CPU's redo draws nothing and the second step must start where the first did |
| `isr-reset-scattering-seed` | `reset_scattering_seed=1`, so the device's place in the sequence must restart at each step |
| `isr-rpn-draws` | `grnd` in a `&modulate_elements` expression each pass, which calls the library's generator directly, so `random_2` must already be current when RPN runs. `srnd` first makes the CPU repeatable |
| `isr-rpn-reseed` | `srnd` in that expression, which restarts `random_2` through the library's `random_1`, so the device must start again from the new seed |
| `isr-loss-before-first-draw-rpn` | A first step that stops every particle at the first `ISR` magnet, then a `grnd` that draws from `random_2` before the next one. The device used to take a number to read `random_2`'s state, so RPN got the wrong one and the second step was one number off |
| `search-frequency-map-isr` | The batched `&frequency_map` must refuse `ISR` and keep the magnet out of its program. Runs in `auto` mode with a threshold, since at one particle in `required` mode the magnet itself would run `ISR` on the GPU |

The older `magnet-ccbend-isr`, `magnet-ccbend-isr-loss` and `magnet-cwiggler-isr` now use the same `HKICK` as the CPU element
after the magnet.

### SREFFECTS (3)

`SREFFECTS` draws three numbers per particle from the same sequence as `ISR`. An `ISR` element on the CPU follows it in each case, so a wrong count shows up there. With output in CPU order the numbers are made on the host as `gauss_rn_lim()` makes them (one thread unless the run asks for more), so `run.out` matches the CPU exactly; the verify build compares every number with `gauss_rn_lim()` exactly.

| Case | Guards |
| --- | --- |
| `sreffects-excite` | Excitation on the device, with `FRACTION`, `INCLUDE_OFFSETS`, `COUPLING`, and with `DAMPING` and `LOSSES` off |
| `sreffects-no-excite` | `QEXCITATION=0`, where the CPU still draws three numbers per particle, so the device has to use them up too |
| `sreffects-cutoff` | `CUTOFF`, which rejects numbers at its own limit and not the one for `ISR` |

### Magnets (58)

| Case | Guards |
| --- | --- |
| `magnet-kquad-apcontour` | `KQUAD` and `KSEXT` behind a sticky contour, with `TILT` and `DX`. Particles pass outside the contour mid-magnet and return by the exit, so only the kernel can stop them |
| `magnet-kquad-apcontour-logic` | `KQUAD` with `Logic` expressions |
| `magnet-kquad-apcontour-isr` | `KQUAD` `ISR`, where a loss redoes the magnet on the CPU with the same numbers |
| `magnet-csbend-apcontour` | `CSBEND` behind a sticky contour, with both signs of `ANGLE`, `TILT` and `DX`, and the same mid-magnet loss |
| `magnet-multipole-misalignment` | `KQUAD`, `KSEXT` and `KOCT` with `DX`, `DY`, `DZ` and `TILT` |
| `magnet-mult` | Simple `MULT`, orders 0 through 3 |
| `magnet-mult-loss-compaction` | Simple `MULT` loss compaction with no loss output |
| `magnet-mult-loss-compaction-acceptance` | Simple `MULT` loss compaction with acceptance output |
| `magnet-mult-global-loss-fallback` | Simple `MULT` with a losses file and global loss coordinates. Despite the name, the magnet stays on the GPU since 2026-11-13 and the CPU routine writes only the lost rows |
| `magnet-dqcor-misalignment` | `DQCOR` with normal and skew terms, steering kicks, `DX`, `DY`, `DZ` and `TILT` |
| `magnet-csbend-misalignment` | Non-CSR `CSBEND` with `DX`, `DY`, `DZ`, `TILT` and `ETILT` |
| `magnet-csbend-expanded` | Non-CSR `CSBEND` with first-order edge focusing and `EXPAND_HAMILTONIAN=1` |
| `magnet-csbend-high-order-edge` | Non-CSR `CSBEND` with Brown-style higher-order `EDGE_EFFECTS=1` focusing |
| `magnet-csbend-high-order` | `CSBEND` with `K3` to `K8`, at amplitudes where every term matters: integration orders 2 and 4, both bend directions, and one bend with `TILT`, `DX`, `DY` and `DZ`. No case covered these terms before; a build that dropped `K5` to `K8` on the GPU failed it |
| `magnet-csbend-loss-compaction` | Non-CSR `CSBEND` loss compaction with acceptance output |
| `magnet-csbend-internal-loss` | A `CSBEND` with half a betatron period, so particles leave the `MAXAMP` inside the bend and are back inside by its exit. Only the check at each step loses them, 1,339 of 4,000. With output and a losses file: until 2026-11-13 the kernel's loss count sent the bend to the CPU, and now the device keeps the survivors and the CPU routine writes the lost rows |
| `magnet-csbend-internal-loss-no-output` | The same with no particle output, so the device compacts the survivors. A build that checked only before the first step failed both, and none of the other 14 `CSBEND` cases |
| `magnet-csbend-advanced-fallback` | `CSBEND` with pitch and yaw misalignment and Hwang, Lindberg and curved fringe models. Despite the name, its magnets run on the GPU now that those misalignments are supported |
| `magnet-kickmap-loss-compaction` | `GKICKMAP` map-loss compaction with no loss output |
| `magnet-kickmap-global-loss-fallback` | `GKICKMAP` with a losses file and global loss coordinates. Despite the name, the map stays on the GPU since 2026-11-13 |
| `magnet-ccbend-loss-compaction` | `CCBEND` loss compaction with acceptance output |
| `magnet-lgbend-loss-compaction` | `LGBEND` loss compaction with acceptance output |
| `magnet-lgbend-aperture-data` | `LGBEND` with `APERTURE_DATA`, using the APS `AM1` configuration |
| `magnet-lgbend-misalignment` | `DX`, `DY`, `DZ`, `ETILT`, `EPITCH`, `EYAW` and `TILT`, alone and together, and radiation |
| `magnet-lgbend-misalignment-loss` | `TILT`, `DX`, `DY`, `ETILT` with `MAXAMP` stopping part of the beam: the aperture sees the offset |
| `magnet-lgbend-misalignment-aperture` | `APERTURE_DATA` with `DX`, `DY`, `TILT`, `ETILT`: the local aperture is checked after the offset and reverse tilt |
| `magnet-ukickmap-loss-compaction` | `UKICKMAP` with `SYNCH_RAD=1` and losses. Regression test for the extra radiation kicks fixed in `ukickmap.c` on 2026-11-04: CPU and GPU differ in `p` if that returns |
| `magnet-kickmap-accepted-compaction` | `GKICKMAP` loss compaction with acceptance output |
| `magnet-polyseries` | `POLYNOMIALSERIES` kernel, with its coefficients in `poly.sdds` |
| `magnet-polyseries-few-particles` | Fewer particles than the threshold in `required` mode, so it falls back to the CPU inside the GPU path |
| `magnet-polyseries-loss` | Losses at the slope limit, including the first row, which fall back to the CPU. The survivors must reach the next GPU element in the right rows |
| `magnet-ksbend-loss-with-gpu` | `KSBEND`, on the CPU, loses particles, including the first row, and the next GPU element must see the survivors in the right rows |
| `magnet-exact-corrector-loss` | `EHKICK`, `EVKICK` and `EKICKER` with losses |
| `magnet-exact-corrector-global-loss` | The same with `losses_include_global_coordinates=1`, including a slope above 1, which is lost at the element exit |
| `magnet-multipole-field-options` | `KQUAD`, `KSEXT` and `KOCT` with `BORE` and `B`, and `KSEXT` with `K1` and `J1` |
| `magnet-csbend-kick-limit` | `CSBEND` with `FSE`, edge kick limits and `KICK_LIMIT_SCALING` |
| `magnet-kquad-spin` | Spin through `KQUAD`, `KSEXT`, `KOCT` and `DQCOR` with `TILT`, offsets, steering kicks, fringes, radiation, and integration orders 2, 4 and 6 |
| `magnet-kquad-spin-isr` | Spin with `ISR`, so the spin turns after each kick and before the random kick, in the CPU's order |
| `magnet-kquad-spin-loss` | Spin with losses inside the magnets, so survivors and their spins are compacted on the device |
| `magnet-kquad-spin-watch` | Spin with parameter and centroid `WATCH`es and `centroid` and `sigma` files, so the spin centroids and sigmas are summed on the device |
| `magnet-csbend-spin` | `CSBEND` spin with edge models 1 to 5, edge order 2, `TILT`, offsets, gradients, radiation and integration orders 2, 4 and 6 |
| `magnet-csbend-spin-loss` | `CSBEND` spin with losses inside the bends, so survivors and their spins are compacted on the device. Without `ISR`, since a loss in an `ISR` magnet redoes it on the CPU |
| `magnet-kquad-spin-edge-files` | Spin with edge multipole files: the spin turns after each edge kick, with the new slopes at the entrance and the old ones at the exit |
| `magnet-kquad-spin-malign` | `KQUAD`, `KSEXT` and `DQCOR` spin with misalignment methods 1 to 4, `TILT`, offsets, `PITCH` and `YAW`. Every method rolls the spin with the design tilt |
| `magnet-csbend-spin-malign` | `CSBEND` spin with misalignment methods 1 to 4, `TILT` and `ETILT`. The spin rolls with the design tilt only |
| `magnet-csbend-spin-expanded` | `CSBEND` spin with the expanded Hamiltonian: no turn in the body, while the edges and the tilt still turn it |
| `magnet-mult-spin` | `MULT` with spin: thick, thin and zero-strength, where `multipole_tracking()` only rolls the spin with `TILT` |
| `magnet-kquad-spin-zero-length` | Spin through a zero-length `KQUAD` with kicks and `TILT`, one with no strength, and a thin `FMULT`. The kick turns the spin over a unit step; dividing by the step of zero made every spin NaN on the CPU (bug 13), and the device's unset body step did the same |
| `magnet-kquse` | `KQUSE` on the device: `TILT`, offsets, `FSE1`, `FSE2`, integration orders 2, 4 and 6, the expanded Hamiltonian, `ISR` and losses. `MATRIX_TRACKING=1` stays on the CPU |
| `magnet-kquse-spin` | `KQUSE` with spin |
| `magnet-kquad-aperture-data` | `KQUAD`, `KSEXT`, `KQUSE` and `DQCOR` in an `&aperture_data` file that narrows and moves off center at each magnet's center. Each magnet is half a betatron period, so only the check inside it, at its center, loses particles; no losses file, so the device compacts them. A build that took the file at the entrance failed it |
| `magnet-csbend-aperture-data` | The same for `CSBEND`, one bending each way |
| `magnet-csbend-edge-models` | `CSBEND` edge models 2 (K. Hwang), 3 (symplectic Brown kick) and 4 (Hwang as symplectified by R. Lindberg) at edge orders 1 and 2, with gaps, `FINT`, `FINT1`/`FINT2`, `H1`/`H2`, mixed models, a kick limit that binds, a gradient, `TILT`, the expanded Hamiltonian and a negative angle |
| `magnet-csbend-edge-models-spin` | The same with spin, which turns after each edge from its kick |
| `bmxyz-maxamp-loss` | `BMXYZ` with `MAXAMP`, rectangular then elliptical: `lorentz()` checks the aperture at every derivative, trial points included, and the device loses particles at the same point and in the CPU's order. Shares `data/bmxyz_field.sdds` |
| `bmxyz-apcontour-loss` | The same inside a sticky `APCONTOUR` |
| `bmxyz-aperture-data-loss` | An `&aperture_data` file, evaluated at each element's start as `lorentz()` does, off center |
| `bmxyz-maxamp-losses-accepted` | With a losses file and acceptance output: the lost rows are the first point outside, at its `z`. Builds that checked only at step ends, or left the element's start out of the loss position, failed it |

### Magnet losses with a losses file (8)

Added 2026-11-13. With a losses file (or global loss coordinates) a magnet that lost a particle used to be
tracked again on the CPU, the whole beam. Now the device keeps the survivors and elegant's own routine tracks
only the lost particles to write their rows (`gpuMagnetLossRowsFromCpu()` in `gpu_stub.c`). The losses file
must have elegant's particles, passes and row order in every case. Its values are elegant's to the last bit
where the device's kernels are (the `KQUAD` family without misalignment, `CCBEND`, kick maps, `FMULT` here)
and within rounding after a particle has been through a kernel that rounds differently (`LGBEND`, misaligned
magnets). Ten older cases with a losses file reach the same code: `magnet-malign-loss`,
`magnet-malign-exact-loss`, `magnet-multipole-files-loss`, `magnet-mult-global-loss-fallback`, the four
`magnet-ccbend-*-loss` cases with a file, `magnet-csbend-internal-loss` and
`magnet-kickmap-global-loss-fallback`; their losses files are byte-identical to elegant's.
`ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0` gives the old behavior.

| Case | What it covers |
| --- | --- |
| `magnet-kquad-losses-file-all-lost` | `KQUAD` (tilt, offset, edge effects), `KSEXT`, `KOCT` and `KQUSE` behind a sticky aperture, some lost in each, then every particle left lost in one magnet. Losses file byte-identical to elegant's |
| `magnet-fmult-losses-file` | `FMULT` strong enough to lose all 2,000 particles inside it. Byte-identical |
| `magnet-ukickmap-losses-file` | `UKICKMAP` losing 1,976 of 3,000. Byte-identical |
| `magnet-lgbend-losses-file` | `LGBEND` with acceptance output, three passes, losses in all nine bend calls. Rows lost in the first bend are identical; later rows differ by up to 1e-13 relative because the survivors now go through the device's `LGBEND`, where the old rule sent every one of these calls to the CPU |
| `magnet-lgbend-losses-file-global` | The same with global loss coordinates |
| `magnet-malign-loss-cpu-keeps` | `magnet-malign-loss` with `ELEGANT_GPU_TEST_MAGNET_LOSS_DISAGREEMENT=1` from `suite.json`: after the device has compacted, the code acts as if elegant's routine kept a particle the device lost, and the whole magnet goes to the CPU. That fallback can't be reached by a real deck on purpose. Outputs must be those of `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0` |
| `magnet-lgbend-losses-file-cpu-keeps` | `magnet-lgbend-losses-file` with the same switch. Every `LGBEND` call here loses a particle, so every one goes to the CPU and the losses file is byte-identical to elegant's |
| `magnet-malign-losses-file-global` | Misaligned `KQUAD`, `KSEXT` and `CSBEND` with global loss coordinates. All 259 rows lost inside magnets are identical to elegant's, global coordinates included. Global `X` is one unit in the last place off in 97 of the 241 rows lost inside drifts, which the aperture code writes, as it was before this change (the old rule gives the same file) |

### WAKE and TRWAKE (14)

| Case | Guards |
| --- | --- |
| `wake-trwake-autoscale` | Autoscaled `WAKE` and `TRWAKE` |
| `wake-trwake-fixed-bins` | Fixed-bin `WAKE` and `TRWAKE` |
| `wake-smoothing` | Fixed-bin smoothed longitudinal `WAKE` |
| `wake-change-p0` | Autoscaled longitudinal `WAKE` with `CHANGE_P0=1` |
| `wake-trwake-smoothing` | Fixed-bin smoothed transverse `TRWAKE` |
| `wake-trwake-tilt` | Fixed-bin tilted transverse `TRWAKE` |
| `wake-bunched-single` | Single-effective-bucket bunched `WAKE` and `TRWAKE` |
| `wake-bunched-skip` | Multi-bucket bunched `WAKE` and `TRWAKE` filters that skip every effective bucket |
| `wake-bunched-select-one` | Multi-bucket bunched `WAKE` and `TRWAKE` filters that select exactly one effective bucket |
| `wake-bunched-select-range` | Multi-bucket bunched `WAKE` and `TRWAKE` filters that select multiple effective buckets |
| `wake-trwake-fixed-bins-no-output` | `wake-trwake-fixed-bins` with only centroid and sigma output. With no `output`, `final` or `WATCH` file, a wake with `N_BINS` set centers its bins on a mean time that the device adds in its own order. Since 2026-11-13 that sum comes from the time-span kernel (`gpuCudaTimeSpan`); before, from the beam-sum kernels, 400 times slower, and no case reached it. A build that left the first 256 particles out of the sum, and one that returned no sum, each failed this case and the three below |
| `wake-bunched-fixed-bins-no-output` | `wake-bunched-select-range` the same way: the mean time of one bunch, picked by its bunch number. A build that added the other bunches' times failed it alone, and one that found the wrong largest bunch number (`gpuCudaLongMinMax`, also widened that day) failed it and three other bunched cases |
| `wake-bunched-change-p0-skip` | Skipped bunched `WAKE,CHANGE_P0=1` match-only path |
| `wake-ramp-passes` | `WAKE` and `TRWAKE` with `RAMP_PASSES` |

With output files these wake cases, and the `rfcw-*` cases with wakes, now give `run.out` identical to the CPU's;
their centroid and sigma files differ by rounding because the suite puts beam sums on the device. Until CPU bug 18
was fixed on 2026-11-09, `TRWAKE` with `N_BINS` and one bunch centered its bins on t = 0, so `wake-trwake-tilt`,
`wake-trwake-fixed-bins`, `wake-trwake-smoothing`, `wake-ramp-passes` and `rfcw-fixed-wake-bins` binned no particles
and gave no transverse kick (see `elegant-CPU-bugs-and-fixes.md`). They now bin every particle and kick.

### LSCDRIFT (6)

| Case | Guards |
| --- | --- |
| `lsc-auto-leffective` | Fixed-bin `LSCDRIFT` with `AUTO_LEFFECTIVE=1` |
| `lsc-backtrack-fallback` | Backtracking `LSCDRIFT`, which stays on the CPU by rule |
| `lsc-kick-mode` | Fixed-bin `LSCDRIFT` kick mode with `L=0,LEFFECTIVE>0` |
| `lsc-low-frequency-filter` | Fixed-bin `LSCDRIFT` with low- and high-frequency cutoff filtering |
| `lsc-smoothing-filter` | Fixed-bin `LSCDRIFT` with Savitzky-Golay smoothing and high-frequency cutoff filtering |
| `lsc-zero-time-spread` | `LSCDRIFT` with `sigma_s=0`, which finishes the element on the CPU |

### CSR (22)

| Case | Guards |
| --- | --- |
| `csr-bins-512` | Isolated `CSRCSBEND` with fewer bins |
| `csr-short-bunch` | Isolated `CSRCSBEND` with a shorter bunch |
| `csr-entry-edge` | Resident `CSRCSBEND` first-order entrance and exit edge focusing |
| `csr-resident-host-entry` | `DX`, `ETILT` or `EDGE_ORDER=2`: the entrance is on the CPU, then the body on the device |
| `csr-resident-body-loss` | A radius of 0.1 m and a beam wider than that, so particles are lost inside the body. With no losses file they stay lost on the device, in the CPU's order |
| `csr-maxamp-loss` | `MAXAMP`, rectangular then elliptical, losing particles in most body slices on the device, which moves them and their `beta0` into the CPU's order. `B2` bends the other way with `TILT`, so its aperture is checked after the reverse tilt and its entrance runs on the host. A build that took the survivors in another order, or left `beta0` behind, or ignored the reverse tilt, failed it |
| `csr-apcontour-loss` | The same inside a sticky `APCONTOUR` |
| `csr-aperture-data-loss` | An `&aperture_data` file that narrows and moves off center at each bend's center, where the file is evaluated. A build that ignored the center failed it |
| `csr-maxamp-losses-file` | `csr-maxamp-loss` with a losses file: the lost rows come back from the device after each body slice. A build that didn't copy them failed it |
| `csr-maxamp-accepted` | The same with acceptance output: the accepted rows move with the survivors on the device. A build that didn't move them failed it |
| `csr-filters` | `CSRCSBEND` high/low cutoff and wake-filter table plus `CSRDRIFT` |
| `csr-last-wake` | `CSRCSBEND` last-wake output plus `CSRDRIFT` |
| `csr-linear-drift` | No-state `CSRDRIFT,LINEARIZE=1` residency between resident `CSRCSBEND` sections |
| `csr-noop-drift-aperture` | Aperture loss before resident `CSRCSBEND` plus a no-CSR `CSRDRIFT` drift |
| `csr-saldin54` | `CSRCSBEND` last-wake handoff into `CSRDRIFT,USE_SALDIN54=1` |
| `csr-drift-plain-wake` | Resident `CSRCSBEND` wake into a plain `CSRDRIFT` at 150 MeV, where the wake's position matters |
| `csr-drift-overtaking-wake` | The same with `USE_OVERTAKING_LENGTH=1` |
| `csr-edge-angles-output` | Four bends at 150 MeV with edge angles of 0.02 to 0.08 rad and an output file, where the device must give elegant's particles to the last bit. Two of the bends have 12 and 11 slices: with 16 slices, or 10, the wake's factor `CSRConstant*length/nSlices` comes out the same however it is grouped, and a build that grouped it differently passed. The edge kicks take `tan()` of the edge angle from the host, because the device's `tan()` differs from glibc's in the last bit at 0.08 rad, and CSR binning amplifies that |
| `csr-drift-stupakov-output` | `CSRDRIFT,USE_STUPAKOV=1` on the device after each bend of a chicane, with `DZ`, a `STUPAKOV_OUTPUT` file, and two more drifts after a quadrupole, which makes the drift first take the beam back to where the last one ended. The host computes the wake from the device's histogram, and the device advances and kicks the particles in the CPU's arithmetic: the particle file and the Stupakov file must be elegant's to the last bit |
| `csr-drift-stupakov-linearize-slice` | The same drift with `&slice_analysis` output every second kick, and one with `LINEARIZE=1`, which needs the bend's end on the host |
| `csr-slice-analysis-tilt` | `&slice_analysis` with a tilted bend that writes slices inside the body every 4 kicks. elegant converts the beam out of bend coordinates and back around that output, which changes the times by rounding; the device repeats the conversion. A build that skipped it differed from elegant in nearly every particle |
| `watch-parameter-el` | `el` from a parameter-mode `WATCH` on a chirped beam, with GPU sums |

### SCMULT (7)

| Case | Guards |
| --- | --- |
| `scmult-linear` | Drift-only inserted linear `SCMULT` |
| `scmult-sliced-fallback` | Sliced linear `SCMULT`, which stays on the CPU |
| `scmult-multibunch-fallback` | Multi-bunch linear `SCMULT`, which stays on the CPU |
| `scmult-nonlinear-round` | Nonlinear `SCMULT` starting from a round beam |
| `scmult-far-along-linear` | Linear `SCMULT` with a 1 mm bunch at `s` of 1e6 m, about 1000 turns of a 1 km ring. Raw sums of squares used to lose the bunch length to roundoff here |
| `scmult-far-along-nonlinear` | The same for nonlinear `SCMULT` |
| `scmult-zero-length` | `sigma_s=0` must fall back to the CPU, not exit |

### RFCA (9)

| Case | Guards |
| --- | --- |
| `rfca-thin-change-p0` | Zero-length thin `RFCA,CHANGE_P0=1` with GPU phase setup and central-momentum matching |
| `rfca-thin-fiducial-modes` | Zero-length thin `RFCA` with `LIGHT` and serial/local full-beam `TMEAN` GPU phase setup |
| `rfca-thin-offset` | Zero-length thin `RFCA` with deterministic `DX/DY` offsets |
| `rfca-matrix-rf-only` | Nonzero-length RF-only matrix-method `RFCA` with end focusing, `CHANGE_P0`, fiducial phase setup, and deterministic `DX/DY` offsets |
| `rfca-matrix-fiducial-modes` | Nonzero-length RF-only matrix-method `RFCA` with `LIGHT` and serial/local full-beam `TMEAN` GPU phase setup |
| `rfca-kick-rf-only` | Nonzero-length RF-only kick-method `RFCA,N_KICKS>0` with `LIGHT`/`TMEAN` fiducialization, offsets, focusing, and `CHANGE_P0` |
| `rfca-standing-wave-multikick` | Multi-kick `STANDING_WAVE=1` `RFCA` with fiducialization |
| `rfca-delta-phase` | `RFCA` with nonzero `DELTA_PHASE` and 4 kicks. elegantTestSet has no `DELTA_PHASE` test |
| `rfca-thin-fiducial-modes-no-output` | `rfca-thin-fiducial-modes` without `output` or `final`, so fiducialization runs on the device |

### RFCW (18)

| Case | Guards |
| --- | --- |
| `rfcw-rf-only-offset` | RF-only `RFCW` matrix-method cavities with deterministic `DX/DY` offsets |
| `rfcw-rf-only-fiducial-modes` | RF-only `RFCW` matrix-method cavities with `LIGHT` and serial/local full-beam `TMEAN` GPU phase setup |
| `rfcw-kick-rf-only` | RF-only kick-method `RFCW,N_KICKS>0` with `LIGHT`/`TMEAN` fiducialization, offsets, focusing, and `CHANGE_P0` |
| `rfcw-matrix-wake` | Matrix-method `RFCW` with longitudinal/transverse wakes, smoothing/interpolation, offsets, focusing, and `CHANGE_P0` |
| `rfcw-kick-wake` | Kick-method `RFCW,N_KICKS=1` with longitudinal/transverse wakes, smoothing/interpolation, offsets, focusing, and `CHANGE_P0` |
| `rfcw-matrix-wakes-at-end` | Matrix-method `RFCW,WAKES_AT_END=1` with longitudinal/transverse wakes, smoothing/interpolation, offsets, focusing, and `CHANGE_P0` |
| `rfcw-wake-lsc` | Matrix-method and kick-method `RFCW` with longitudinal/transverse wakes plus fixed-bin filtered LSC kicks |
| `rfcw-multikick` | Kick-method `RFCW,N_KICKS>1` with longitudinal/transverse wakes, fixed-bin filtered LSC kicks, and `WAKES_AT_END=0|1` |
| `rfcw-wake-pmaximum-fiducial` | Wake-bearing matrix-method and kick-method `RFCW` with `FIDUCIAL="pmaximum"` and fixed-bin filtered LSC kicks |
| `rfcw-fixed-wake-bins` | Wake-bearing matrix-method and kick-method `RFCW` with fixed longitudinal/transverse wake bins |
| `rfcw-fixed-wake-bins-no-output` | The same with only centroid and sigma output, so both wakes of each `RFCW` use the device's own sum for their mean time (see `wake-trwake-fixed-bins-no-output`) |
| `rfcw-lsc-only` | Matrix-method and kick-method `RFCW` with fixed-bin filtered LSC kicks and no active wake columns |
| `rfcw-single-wake-planes` | Matrix-method and kick-method `RFCW` with longitudinal-only or transverse-only wake columns |
| `rfcw-wake-treference` | Wake-bearing matrix-method and kick-method `RFCW` with explicit nonzero `T_REFERENCE` |
| `rfcw-wake-selected-fiducial` | Wake-bearing matrix-method and kick-method `RFCW` with selected-bunch `TMEAN` and `PMAXIMUM` |
| `rfcw-standing-wave-multikick` | Multi-kick `STANDING_WAVE=1` `RFCW` with fiducialization |
| `rfcw-lsc-plain` | Ten `RFCW` cells with LSC and `CHANGE_P0` |
| `rfcw-lsc-first-is-fiducial` | The same cells with `first_is_fiducial=1`, the regression test for the doubled energy gain fixed on 2026-11-03 |

### RAMPRF (5)

| Case | Guards |
| --- | --- |
| `ramprf-waveforms` | `RAMPRF` with voltage, phase and frequency waveforms and a length, with `RAMPP`, `KQUAD`s, a coordinate `WATCH` and output in CPU order, so the kicks use the CPU's `sin` |
| `ramprf-fixed-frequency` | `RAMPRF` at a fixed frequency with `tmean`, `first` and `light` fiducials, and one with no voltage, which drifts as the CPU does |
| `ramprf-waveforms-no-output` | `ramprf-waveforms` without output files, so the device makes its own `sin` |
| `ramprf-pmaximum-first-highest` | `pmaximum` fiducials on thin and long `RAMPRF`s when the first particle has the most momentum, tied with particle 701, with output in CPU order |
| `ramprf-pmaximum-no-output` | `pmaximum` fiducials on `RAMPRF` with no particle output, so the kicks use the device's `sin` |

### RFCA and RFCW together (11)

| Case | Guards |
| --- | --- |
| `rf-pmaximum-fiducial` | Serial/local RF-only `RFCA` and `RFCW` cavities with `FIDUCIAL="pmaximum"` across thin, matrix-method, and kick-method slices |
| `rf-kick-treference` | RF-only kick-method `RFCA` and `RFCW` with explicit nonzero `T_REFERENCE` |
| `rf-selected-tmean-fiducial` | Serial/local selected-bunch `TMEAN` fiducialization across RFCA and RF-only RFCW slices |
| `rf-selected-pmaximum-fiducial` | Serial/local selected-bunch `PMAXIMUM` fiducialization across RFCA and RF-only RFCW slices |
| `rf-first-fiducial` | Selected-bunch `FIRST` fiducialization across supported RFCA, RF-only RFCW, and guarded wake-bearing RFCW slices |
| `rf-standing-wave-single` | `STANDING_WAVE=1` `RFCA` and `RFCW`, matrix-method and single-kick |
| `rf-standing-wave-multikick-treference` | Multi-kick `STANDING_WAVE=1` `RFCA` and `RFCW` with explicit `T_REFERENCE` |
| `rf-pmaximum-fiducial-no-output` | `rf-pmaximum-fiducial` without `output` or `final` |
| `rf-kick-treference-no-output` | `rf-kick-treference` without `output` or `final` |
| `rf-pmaximum-fiducial-first-highest` | `pmaximum` when the first particle has the most momentum, tied with particle 701, which arrives 2 ps later. The CPU and the device kernel used to stop here, and now both must pick the first particle |
| `rf-selected-pmaximum-fiducial-other-bunch-first` | `pmaximum` on bunch 0 of a two-bunch file whose first row is from bunch 1 and has the most momentum. The CPU and the device kernel used to stop here |

### Particle tunes (1)

| Case | Guards |
| --- | --- |
| `diagnostic-particle-tunes` | `&particle_tunes` with no other output, so nothing else syncs the beam to the host each pass |

### Fast orbit feedback (1)

| Case | Guards |
| --- | --- |
| `fofb-control` | `&fast_orbit_feedback` with RF control, limits, anti-windup and BPM noise. elegantTestSet has no such test |

### Element modulation (3)

| Case | Guards |
| --- | --- |
| `modulate-columns` | `&modulate_elements` with `element_amplitude_column` from a table |
| `modulate-noise` | `&modulate_elements` noise with `correlated=1` |
| `modulate-rng` | `&modulate_elements` random numbers with both `correlated=1` and `correlated=0` |

### IMPEDANCE (2)

| Case | Guards |
| --- | --- |
| `impedance-ramp-passes` | `IMPEDANCE` with `RAMP_PASSES` and `FACTOR` |
| `impedance-bunched-skip` | `IMPEDANCE` with a bunch filter that skips every bucket |

### CWAKE (3)

| Case | Guards |
| --- | --- |
| `cwake-ramp-change-p0` | `CWAKE` with `RAMP_PASSES` and `CHANGE_P0` |
| `cwake-bunched-skip` | `CWAKE,CHANGE_P0=1` with a bunch filter that skips every bucket |
| `cwake-bunched-fixed-bins-no-output` | Bunched-mode `CWAKE` with `N_BINS` set acting on every bunch, with only centroid and sigma output: the mean time of each bunch from the device's own sum (see `wake-trwake-fixed-bins-no-output`) |

### ZLONGIT and ZTRANSVERSE (12)

Since 2026-11-14 both run on the device: the device bins (always in particle order) and kicks. With
output in the CPU's order, smoothing or a `WAKES` file the host makes the voltages from the histograms with
the CPU's own operations (and writes the `WAKES` tables); otherwise cuFFT makes them, as for `IMPEDANCE`.
The cases that write only `output` (and `WAKES`) must be identical to the CPU, `WAKES` tables included; the
two `*-no-output` cases take the cuFFT path.

| Case | Guards |
| --- | --- |
| `zlongit-output-cpu-order` | A broad-band resonator on `ZLONGIT`'s own grid (bins centered on the mean time, interpolated, `RAMP_PASSES`, `FACTOR`) and a file impedance with `BIN_CENTERED=1`, no interpolation and `START_ON_PASS`; a 37.5 m drift first, so the order of the mean-time sum shows |
| `zlongit-wakes-smoothing` | `WAKES` files: a file impedance with Savitzky-Golay smoothing writing every second pass from pass 1, and a broad-band resonator (`Ra`, `fo`, `Deltaf`) |
| `zlongit-bunches-output` | Three bunches, `START_BUNCH=1`, `END_BUNCH=2`, a `WAKES` file whose `Bunch` is the CPU's bucket number |
| `zlongit-bunches-unsorted` | Three bunches after an `RCOL` removes particles, so the rows are out of bunch order; the device gathers them by bunch with a stable sort and writes them back. Before 2026-11-14 such a beam went to the CPU |
| `zlongit-no-output` | Only centroid and sigma output: voltages from cuFFT, and bins centered on a mean time the device adds in its own order (the histograms are in particle order either way). Rounding expected |
| `ztransverse-output-cpu-order` | An off-axis beam: a broad-band resonator driven by (x - `DX`) and (y - `DY`) with `XFACTOR` and `RAMP_PASSES`, a quadrupole file impedance (drive exponents 0, probe exponents 1), and a dipole one with probe exponents 2 and 3 |
| `ztransverse-wakes-smoothing` | `WAKES` files: a dipole impedance with smoothing, `YFACTOR=0` (zero y voltage, unsmoothed y histogram in the table), `START_ON_PASS`, and a broad-band resonator (`Rs`, `fo`, `Deltaf`) |
| `ztransverse-bunches-output` | Three off-axis bunches, `START_BUNCH=0`, `END_BUNCH=1`, a `WAKES` file |
| `ztransverse-bunches-unsorted` | `zlongit-bunches-unsorted` for `ZTRANSVERSE`: a dipole impedance with a `WAKES` file and a quadrupole one with `END_BUNCH=1` |
| `ztransverse-no-output` | Only centroid and sigma output: voltages from cuFFT. Rounding expected |
| `zimpedance-cpu-options` | The options that stay on the CPU: `AREA_WEIGHT`, `REVERSE_TIME_ORDER`, a drive exponent of 2 and a probe exponent of 4 |
| `zimpedance-opt-out` | `ELEGANT_GPU_ENABLE_ZIMPEDANCE=0` (from `suite.json`) keeps both on the CPU |

### IBSCATTER (8)

Since 2026-11-14 `IBSCATTER` with `SMOOTH=1` runs on the device. The sums that set the growth rates stay
exact: the host copies the rows down and runs elegant's own `slicebeam()`, `computeSliceParameters()` and
`IBSRate()`, and writes the IBS file; the device applies each slice's inflation to its rows. A sum in particle
order is one long chain of dependent adds, which takes about 28 ns per add on the device, so a device sum was
slower than the CPU. `SMOOTH=0`, which draws random kicks in particle order, stays on the CPU, as does an MPI
build. The cases that write only `output` must be identical to the CPU, the IBS files included.

| Case | Guards |
| --- | --- |
| `ibscatter-ring-output` | A ring with `ISRING=1`, `NSLICE=1`: one bucket, growth rates from the copied rows, the device inflation, and the IBS file. `FACTOR=3e4` makes the growth show in 5 passes |
| `ibscatter-slices-output` | `NSLICE=5`: each row is inflated with its own slice's factors, and the momentum factor is interpolated from the neighbouring slices by the row's time |
| `ibscatter-linac-output` | `ISRING=0`, `NSLICE=3`, `BUNCHED_BEAM_MODE=0`, at the end of each cell of a line tracked once: after the inflation the host copies the rows again for the slice emittances in the IBS file |
| `ibscatter-bunches-output` | `BUNCHED_BEAM_MODE=1` with three bunches: each bunch is a bucket with its own charge and growth rates |
| `ibscatter-bunches-unsorted` | Three bunches after an `RCOL` removes particles, so the rows are out of bunch order; the device gathers them by bunch and writes them back. With `ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER=0` all five calls go to the CPU, so the case reaches the gather |
| `ibscatter-cpu-options` | A `SMOOTH=0` element, which stays on the CPU, in the same ring as a `SMOOTH=1` element on the device |
| `ibscatter-no-output` | `NSLICE=5` with only centroid and sigma output, whose sums the device makes in its own order. Rounding expected |
| `ibscatter-opt-out` | `ELEGANT_GPU_ENABLE_IBSCATTER=0` (from `suite.json`) keeps `IBSCATTER` on the CPU |

### LSRMDLTR (11)

Since 2026-11-14 `LSRMDLTR` with the non-adaptive Runge-Kutta integrator runs on the device, one thread a
particle, with `motion.cc`'s field expressions and `rk_odeint3_na()`'s steps in the CPU's order. Its sines,
cosines, `cosh`, `sinh` and `exp` are CUDA's, so these cases differ from the CPU by rounding; the laser's
reference time `t0` is elegant's exactly. With `ISR` each particle takes one number a step from `random_2`'s
sequence in the CPU's order: the device guesses where each particle's numbers start, the counts it reports
give the CPU's starts, and particles from the first wrong guess on are tracked again (see
`gpu_track_lsrmdltr()`). A particle the CPU would lose sends the whole call back to the CPU.

| Case | Guards |
| --- | --- |
| `lsrmdltr-laser-heater` | A planar undulator, the default "leading terms" field, a TEM00 laser at the resonant wavelength with `LASER_PHASE`, `LASER_X0`, `LASER_Y0` and `LASER_Z0` |
| `lsrmdltr-field-models` | All three field expansions on planar and helical undulators, `TGU_GRADIENT` with `TGU_COMP_FACTOR`, other `POLE_FACTOR`s, an off-axis beam |
| `lsrmdltr-laser-modes` | TEM31 with `LASER_TILT` and `LASER_WAVELENGTH`, TEM42 off axis: the Hermite polynomials and their derivatives in each field component |
| `lsrmdltr-time-profile` | `TIME_PROFILE` and `TIME_OFFSET`, a bunch longer than the profile, and `t0` from the device's values summed on the host in elegant's order |
| `lsrmdltr-synch-rad` | `SYNCH_RAD=1` at 2 GeV, the energy loss after each step |
| `lsrmdltr-isr` | `ISR=1` on two elements with different `N_STEPS`, then a `KQUAD` with `ISR`, which matches only if `random_2` moved by exactly the numbers used |
| `lsrmdltr-isr-mixed-counts` | 5 MeV and a 10% energy spread, so particles take 2,001 or 2,002 steps: the first guess is wrong and a second pass settles it |
| `lsrmdltr-isr-batches` | The same with 200 particles and a table of about 50 particles' numbers (`ELEGANT_GPU_LSRMDLTR_DRAW_BYTES` from `suite.json`): four batches, 8 passes |
| `lsrmdltr-cpu-options` | `METHOD="modified midpoint"`, which stays on the CPU, next to `METHOD="non-adaptive"` (an abbreviation, as elegantTestSet writes it) on the device |
| `lsrmdltr-fallback` | `ACCURACY=0`: elegant loses every particle ("Integration failure"); the device puts the beam back and the CPU tracks. Identical |
| `lsrmdltr-opt-out` | `ELEGANT_GPU_ENABLE_LSRMDLTR=0` (from `suite.json`) keeps it on the CPU. Identical |

### LRWAKE (5)

Since 2026-11-14 `LRWAKE` runs on the device in elegant (not gpu-Pelegant). The device makes each particle's
time and copies t, x, y and the bunch column to the host, which adds them per bunch in particle order as
`track_through_lrwake()` does; `lrwakeBunchVoltages()`, split out of that routine, moves the history and sums the
wakes on the host; the device applies the kicks. The particles are elegant's to the bit. The wake table is
`data/lrwake.sdds` (decaying resonances, 0 to 4 us).

| Case | Guards |
| --- | --- |
| `lrwake-bunches-output` | Four bunches 2 ns apart, all five wake columns with their factors, `RAMP_PASSES`, `TURNS_TO_KEEP`, six passes, off axis. Identical |
| `lrwake-bunches-unsorted` | An off-axis beam through an `RCOL` that removes particles every pass (8,000 in, 1,880 out), so the rows are out of bunch order and the bunches differ in size; each particle's bunch comes from its own column. Identical |
| `lrwake-single-bunch` | A beam not in bunched mode: one bunch, kicked by its own wakes from earlier passes. Identical |
| `lrwake-no-output` | `lrwake-bunches-output` with only centroid and sigma output, whose sums the device makes in its own order. Rounding expected |
| `lrwake-opt-out` | `ELEGANT_GPU_ENABLE_LRWAKE=0` (from `suite.json`) keeps it on the CPU. Identical |

### CORGPIPE and CORGPLATES (4)

Since 2026-11-14 both run on the device. They are `exactDrift()`, a `WAKE` built from a formula, and `exactDrift()`,
each of which already had a device path; with `TMAX=0` (and `N_BINS=0` for `CORGPIPE`) the table is sized from
the beam's s range, which the device now finds (the extremes are the same in any order).

| Case | Guards |
| --- | --- |
| `corrugated-pipe-output` | `CORGPIPE` with `TMAX=0`, `INTERPOLATE=1`, `RAMP_PASSES`, `CHANGE_P0`. Identical |
| `corrugated-plates-output` | Two `CORGPLATES` with `TMAX=0` and Savitzky-Golay `SMOOTHING`. Identical |
| `corrugated-fixed-bins-no-output` | `CORGPIPE` with `N_BINS`, `CORGPLATES` with `TMAX`, only centroid and sigma output. Rounding expected |
| `corrugated-opt-out` | `ELEGANT_GPU_ENABLE_CORRUGATED=0` (from `suite.json`) keeps both on the CPU. Identical |

### TAPERAPE (6)

Since 2026-11-14 `TAPERAPE` runs on the device with the circular and rectangular tapers. Its test of a point
against the ellipse takes `atan2`, `cos`, `sin` and `pow`, where the device's library can round differently from
glibc's; a test that falls within 1e-12 of the boundary is counted, and the whole call then goes to the CPU, so
the decisions are always elegant's. A loss's global coordinates come from the element's offset and tilted frame,
as `trackThroughTaperApElliptical()` makes them. The same day the tilted `TAPERAPR` was made exact (survivors'
path length from the tilted frame's slopes), and the fused tune program now takes it.

| Case | Guards |
| --- | --- |
| `aperture-taperape` | Tapered semi-axes, `DX`, `DY`, `TILT`, losses at the entrance and part-way along (the bisection), `STICKY`, a losses file with global coordinates. Particles and losses identical |
| `aperture-taperape-exponents` | `XEXPONENT=4` (2/4 is 0 in elegant's integer division), `YEXPONENT=1`, and a zero-length `TAPERAPE`. Particles identical |
| `aperture-taperape-boundary` | Two particles exactly on the entrance ellipse: the tests are uncertain and the call goes to the CPU. Identical |
| `aperture-taperapr-tilt` | `TAPERAPR` at `TILT=0.7` and `-1.1` with `DX`. Particles identical |
| `search-frequency-map-taperapr-tilt` | `search-frequency-map-program-opcodes` with its `TAPERAPR` tilted: the fused program takes it and its check finds no difference. Identical |
| `aperture-taperape-opt-out` | `ELEGANT_GPU_ENABLE_TAPERAPE=0` (from `suite.json`) keeps it on the CPU. Particles identical |

### CENTER and idle passes (6)

Since 2026-11-14 `CENTER` runs on the device to the bit: each mean from one column copied to the host and added
in particle order, the time mean from each particle's time with the CPU's Kahan sum after the other offsets, and
the round trip through time on the device. Elements that leave the beam alone on a pass keep it on the device on
that pass (`gpuIdlePassElement()`, plain tracking only): `BRANCH`, `TWISS` applied once after its first pass,
`POLAR` and `CENTER` off their `ON_PASS`, `SCATTER` outside its passes; `RIMULT` always.
Since `SCATTER` and `TWISS` run on the device (2026-11-15), the two `idle-pass-*` cases keep them on the CPU
(`ELEGANT_GPU_ENABLE_SCATTER=0`, `ELEGANT_GPU_ENABLE_TWISS=0` from `suite.json`), so their active passes still take
the beam to the host.

| Case | Guards |
| --- | --- |
| `center-output` | `X`, `XP`, `Y`, `YP`, `DELTA` and `T` on every pass. Identical |
| `center-once-only` | `ONCE_ONLY=1` on `S` and `T` together (`S` moves before the time mean, as on the CPU), and `X`. Identical |
| `center-on-pass` | `ON_PASS=2`: idle on the other passes. Identical |
| `center-opt-out` | `ELEGANT_GPU_ENABLE_CENTER=0` (from `suite.json`) keeps it on the CPU. Identical |
| `idle-pass-elements` | `TWISS` with `APPLY_ONCE=1`, `RIMULT`, `SCATTER` on passes 2 and 3 only (random numbers in the CPU's order), and a `BRANCH` that jumps once its counter runs out. Identical |
| `idle-pass-opt-out` | `ELEGANT_GPU_ENABLE_IDLE_PASS=0` (from `suite.json`): the same elements take the beam to the CPU every pass. Identical |

### BEAMBEAM (9)

Since the evening of 2026-11-14 `BEAMBEAM` runs on the device, one thread a particle, with
`applyBeamBeamKicks()`'s statements in the CPU's order. The gaussian kick evaluates `Faddeeva::w` twice
(`gpu_faddeeva.h`: the package's two tables of Chebyshev polynomials copied unchanged, so they give the CPU's
values to the bit, and the rest of its statements in order); the uniform and parabolic kicks use glibc's
`csqrt` and libgcc's complex division written out. The device's `exp`, `sin`, `cos`, `atan2` and `hypot` are
CUDA's, so the cases with the element on the device differ from the CPU by rounding.

| Case | Guards |
| --- | --- |
| `beambeam-gaussian` | A flat gaussian opposing beam off axis, with centroid and sigma output |
| `beambeam-round` | `XSIZE` = `YSIZE`: `roundGaussianBeamKick()` |
| `beambeam-tall` | `YSIZE` > `XSIZE`: the planes swapped, a negative charge, both offsets |
| `beambeam-uniform` | `DISTRIBUTION="uniform"`, particles inside and outside the ellipse |
| `beambeam-parabolic` | `DISTRIBUTION="parabolic"`, inside and outside |
| `beambeam-far` | The opposing beam 1 mm away: `Faddeeva::w`'s continued fraction, which uses no function. Identical |
| `beambeam-plane` | A beam with no vertical size, so y = 0 exactly: `Faddeeva::w`'s real-axis branch (`w_im`). The vertical kick is a rounding residue of about 1e-20, different on the device |
| `beambeam-ring` | A collider-like ring, `ILMATRIX`, `RFCA` and `BEAMBEAM` each turn, 20 turns, a centroid `WATCH` |
| `beambeam-opt-out` | `ELEGANT_GPU_ENABLE_BEAMBEAM=0` (from `suite.json`) keeps it on the CPU. Identical |

### RFTMEZ0, TWLA and MAPSOLENOID (13)

Since the evening of 2026-11-14 these three run on the device with the non-adaptive Runge-Kutta
integrator, one thread a particle, as `LSRMDLTR` does: `motionGpuSetup()` (`motion.cc`) makes `motion()`'s
set-up, fiducial selection included, and the device runs its particle loop (`rk_odeint3_na()`, the fields
of `derivatives_rftmEz0()`, `derivatives_tw_linac()` with its impulses and `derivatives_mapSolenoid()`,
`rotate3()` as `mat_mult()` sums). A lost particle's row holds what `motion()` records, `motion()`'s swaps
give the rows' order, and with `CHANGE_P0` the central momentum is matched to all the particles the call
was given, as `motion()` does. `MAPSOLENOID` uses no function but `sqrt` and matches elegant to the bit;
`RFTMEZ0`'s `sin` and `cos` and `TWLA`'s `sin`, `cos` and `exp` are CUDA's, so those differ by rounding.

| Case | Guards |
| --- | --- |
| `rftmez0-cavity` | Two cavities from an on-axis Ez table, `RADIAL_ORDER` 1 and 0, `CHANGE_P0`, median and average time fiducials |
| `rftmez0-solenoid` | An on-axis Bz table (Br from its derivative) and a map of Br and Bz in r and z, offset and tilted, stray fields |
| `rftmez0-misaligned` | `DX`, `DY`, `DZ`, `ETILT`, `EPITCH`, `EYAW`: `rotate3()` of the particle and the fields, with `CHANGE_P0` |
| `rftmez0-cpu-options` | `METHOD="modified midpoint"`, which stays on the CPU, next to the non-adaptive integrator on the device |
| `rftmez0-opt-out` | `ELEGANT_GPU_ENABLE_RFTMEZ0=0` (from `suite.json`) keeps it on the CPU. Identical |
| `twla-linac` | Four cells as elegantTestSet's `twla1` has them: focussing, `CHANGE_P0`, average-time fiducial |
| `twla-losses` | A radial aperture that takes most of the beam, `CHANGE_P0` and a losses file: the loss records, their order, and the momentum matched to the lost particles too, as `motion()` does |
| `twla-entrance-losses` | An offset aperture smaller than the beam: losses at the entrance test and inside the integration, without `CHANGE_P0` |
| `twla-solenoid` | `B_SOLENOID` (the entrance and exit impulses), `ALPHA`, `SUM_BN2`, `BETA_WAVE` below 1, offsets, `FOCUSSING=0`, a `p,maximum` fiducial |
| `twla-opt-out` | `ELEGANT_GPU_ENABLE_TWLA=0` (from `suite.json`) keeps it on the CPU. Identical |
| `mapsolenoid-map` | elegantTestSet's `mapSolenoid1` map, with centroid and sigma output. The particles are elegant's to the bit; the centroid and sigma sums, made on the device, differ by rounding |
| `mapsolenoid-uniform-misaligned` | `BXUNIFORM`, `BYUNIFORM` over `LUNIFORM`, offsets, tilt, yaw and pitch. Identical |
| `mapsolenoid-opt-out` | `ELEGANT_GPU_ENABLE_MAPSOLENOID=0` (from `suite.json`) keeps it on the CPU. Identical |

### TMCF, CEPL, TWPL and TWMTA (9)

Since the afternoon of 2026-11-15 these four `motion()` types run on the device as the three above do: the
non-adaptive Runge-Kutta integrator, one thread a particle, with `derivatives_tmcf_mode()`,
`derivatives_ce_plates()`, `derivatives_tw_plates()` and `derivatives_twmta()` (with its entrance and exit
impulses). Their `sin`, `cos`, `cosh`, `sinh` and `exp` are CUDA's, so they differ by rounding. The adaptive
integrators stay on the CPU; on these decks elegant's adaptive Runge-Kutta and Bulirsch-Stoer stop with step
errors, so no case uses them.

| Case | Guards |
| --- | --- |
| `tmcf-cavity` | TMCF twice with radial and longitudinal fields, a tilt, offsets and an aperture that keeps the beam |
| `tmcf-opt-out` | `ELEGANT_GPU_ENABLE_TMCF=0` (from `suite.json`) keeps it on the CPU. Identical |
| `cepl-ramp` | CEPL twice, its field ramping across a 3 mm bunch, with a static voltage and a tilt |
| `cepl-opt-out` | `ELEGANT_GPU_ENABLE_CEPL=0` (from `suite.json`). Identical |
| `twpl-wave` | TWPL twice, ramping across the bunch, with a static voltage and a tilt (its magnetic term in the longitudinal kick) |
| `twpl-opt-out` | `ELEGANT_GPU_ENABLE_TWPL=0` (from `suite.json`). Identical |
| `twmta-linac` | Three TWMTA cells with transverse modes (`KX`, `BETA_WAVE` below 1), a solenoid field and attenuation |
| `twmta-losses` | TWMTA with an aperture that loses 107 of 300 particles and a losses file: the device marks them and `motion()`'s swaps give the order |
| `twmta-opt-out` | `ELEGANT_GPU_ENABLE_TWMTA=0` (from `suite.json`). Identical |

### MODRF (4)

Since the evening of 2026-11-14 `MODRF` runs on the device in elegant (not with a distributed beam in
gpu-Pelegant). `mod_rfca.c` keeps its own set-up, phase reference, record file and fill factor; the device
gives each particle's time, which the host adds in particle order for the mean, and the kick is `RAMPRF`'s
kernel with the phase taken from t minus that mean (the CPU's `sin` on the host when output is in CPU
order). The cases are elegantTestSet's PAR ring (`ringRF2`) with 200 particles.

| Case | Guards |
| --- | --- |
| `modrf-phase-modulation` | Decaying phase modulation, 32 turns. Identical |
| `modrf-amplitude-record` | Amplitude modulation with `AMPHASE` and `AMOFFSET`, the cavity's fill (`Q`), `START_TIME` and a `RECORD` file. Identical |
| `modrf-zero-voltage` | `VOLT=0` with a length: a drift. Identical |
| `modrf-opt-out` | `ELEGANT_GPU_ENABLE_MODRF=0` (from `suite.json`) keeps it on the CPU. Identical |

### Polynomial kicks and MBUMPER (10)

Since 2026-11-15 `KPOLY`, `HKPOLY` and `MBUMPER` run on the device, one thread a particle, with
`polynomial_kicks()`, `polynomial_hamiltonian()` and `track_through_mkicker()`'s statements in the CPU's order. A
lost particle is flagged on the device, the host rebuilds elegant's swap order from the flags, and the device
gathers the rows. `MBUMPER`'s first call (`set_up_mkicker()` and the fiducial time) reads a host copy. An accepted
beam or global loss coordinates send the call to the CPU. All identical to elegant.

| Case | Guards |
| --- | --- |
| `kpoly-kicks` | KPOLY in both planes with tilts, offsets, DZ, FACTOR and orders 3, 2 and 0: polynomial_kicks() on the device, which must give elegant's particles to the bit (only the output file is asked for) |
| `kpoly-losses` | KPOLY removing the particles past coord_limit: the CPU's swap order of the lost rows, their z and momentum in the losses file, rebuilt on the host from the device's flags. Identical |
| `kpoly-opt-out` | ELEGANT_GPU_ENABLE_KPOLY=0 (suite.json) keeps KPOLY on the CPU. Identical |
| `hkpoly-drift1` | HKPOLY with kick terms and a DRIFT_TYPE=1 drift polynomial, N_REPEATS, FACTOR, tilt and offsets: polynomial_hamiltonian() on the device. Identical |
| `hkpoly-drift2` | HKPOLY with a DRIFT_TYPE=2 drift polynomial whose terms depend on delta: polynomial_hamiltonian() on the device. Identical |
| `hkpoly-losses` | HKPOLY removing the particles past coord_limit, in the CPU's swap order. Identical |
| `hkpoly-opt-out` | ELEGANT_GPU_ENABLE_HKPOLY=0 (suite.json) keeps HKPOLY on the CPU. Identical |
| `mbumper-waveform` | MBUMPER, a quadrupole kicker whose waveform changes across the bunch (3 ps rms against a 20 ps waveform; the tail past it gets no kick), with tilt, offsets, DZ, TIME_OFFSET and N_KICKS=8 (two parts of the fourth-order integrator), and a sextupole one. The first call sets the fiducial on the CPU; with one pass, the second element is the device's. Identical |
| `mbumper-periodic` | MBUMPER with a periodic waveform, firing from the second pass (drifts before it, kept on the device by track_particles()), sextupole and octupole orders, and a tilt. Identical |
| `mbumper-opt-out` | ELEGANT_GPU_ENABLE_MBUMPER=0 (suite.json) keeps MBUMPER on the CPU. Identical |

### BUMPER (5)

Since the afternoon of 2026-11-15 `BUMPER` (`KICKER`) runs on the device: the waveform's time and amplitude, the
exact arc (`N_KICKS=0`), the kicks with `B2`, the deflection map, `PERIODIC`, tilt and offsets, with
`gpuAccurateSin`, `Atan`, `Asin` and `Cos` and CUDA's `tan`, so it differs by rounding. The first call's set-up and
fiducial time read a host copy; passes before `FIRE_ON_PASS` are drifts kept on the device. Not with spin.

| Case | Guards |
| --- | --- |
| `bumper-arc` | An exact arc with a sine waveform across a 5 cm bunch, and a zero-length kicker whose waveform repeats (`PERIODIC`) for the particles before it starts |
| `bumper-kicks-b2` | 20 kicks with `B2`, a periodic flat waveform, a tilt and offsets |
| `bumper-map` | 10 kicks scaled by a bilinear deflection map across the beam |
| `bumper-fire-on-pass` | `FIRE_ON_PASS=2` over four passes: drifts on passes 0 and 1 |
| `bumper-opt-out` | `ELEGANT_GPU_ENABLE_BUMPER=0` (from `suite.json`). Identical |

### RF deflectors (9)

Since 2026-11-15 `RMDF`, `MRFDF`, `SHRFDF` and `RFTM110` run on the device. Their first-call times come from read-only
host copies, and `RFTM110`'s noise is drawn from `random_3` on the host in the CPU's order. `RFTM110` differs by
rounding (CUDA's `sin` and `cos`); the others are identical here.

| Case | Guards |
| --- | --- |
| `rmdf-ramp` | RMDF with a tilt, offsets, sections and a time offset: track_through_ramped_deflector() on the device, its first particle's time from a read-only copy on the first call, at 5 MeV, over three passes. Identical |
| `rmdf-invalid` | RMDF at 6 GeV, far above the low energies its approximation is for: beta passes 1 and every particle's coordinates become NaN, which do_tracking()'s limit_amplitudes() removes after the element. The device counts them and lets that call remove them, in the CPU's order. Identical |
| `rmdf-opt-out` | rmdf-ramp with ELEGANT_GPU_ENABLE_RMDF=0 (suite.json): RMDF on the CPU. Identical |
| `mrfdf-modes` | MRFDF with all five modes, normal and skew, its own frequency and phase each, a tilt, and a range of particle IDs (the others are still turned with the tilt and back): the device's particle loop with CUDA's sin and cos (identical here; they could differ by rounding). The fiducial time is set_up_mrfdf() on a read-only copy |
| `mrfdf-opt-out` | ELEGANT_GPU_ENABLE_MRFDF=0 (suite.json) keeps MRFDF on the CPU. Identical |
| `shrfdf-harmonics` | SHRFDF with the fundamental and three space harmonics, phases and a tilt: the device's particle loop with CUDA's sin and cos. Identical here; they could differ by rounding |
| `shrfdf-opt-out` | ELEGANT_GPU_ENABLE_SHRFDF=0 (suite.json) keeps SHRFDF on the CPU. Identical |
| `rftm110-cavity` | RFTM110 with a tilt, a periodic voltage waveform, voltage and phase noise (drawn from random_3 on the host, in the CPU's order) and a pass window on a second one: the device's particle loop with CUDA's sin and cos, so it differs by rounding |
| `rftm110-opt-out` | ELEGANT_GPU_ENABLE_RFTM110=0 (suite.json) keeps RFTM110 on the CPU. Identical |

### STRAY, ALPH, KSBEND, BOFFAXE and GFWIGGLER (13)

Since 2026-11-15: `STRAY`'s and `ALPH`'s matrices go through the device's `track_particles()`, and `ALPH`'s puck and slits are kernels with the CPU's limits and swap order. `KSBEND`'s integrator runs on the device with the host's tangents (CUDA's `tan` when there is a fringe integral). `BOFFAXE`'s field expansion uses the file's table, uploaded once; its set-up, which reads the file, runs on the host with no particles, so even a one-pass line's calls are the device's. `GFWIGGLER` (`APPLE_Track()`) uses CUDA's `sin`, `cos` and `exp`, so it differs by rounding. The other cases are identical.

| Case | Guards |
| --- | --- |
| `stray-field` | STRAY with local and global components at second order: stray_field_matrix() on the host, the matrix on the device through track_particles() (the global components need the floor coordinates). Identical |
| `stray-opt-out` | ELEGANT_GPU_ENABLE_STRAY=0 (suite.json) keeps STRAY on the CPU. Identical |
| `alph-filters` | An alpha magnet in two halves with a puck scraper and momentum slits on the first and position slits on another: the matrices through track_particles() on the device, the filters as kernels with the CPU's float limits and swap order. Identical |
| `alph-opt-out` | ELEGANT_GPU_ENABLE_ALPH=0 (suite.json) keeps ALPH on the CPU. Identical |
| `ksbend-exact` | KSBEND, nonlinear, with edges but no fringe integral, offsets, tilt and FSE, modified-midpoint integration: track_through_kick_sbend() on the device with the host's tangents. Identical |
| `ksbend-fringe` | KSBEND bending the other way, with a fringe integral (tan(e - psi/(1+dp)) per particle, CUDA's), second-order edges with curvature, radiation, ETILT and leap-frog integration. Identical here; CUDA's tan could differ by rounding |
| `ksbend-losses` | KSBEND losing particles inside, past coord_limit: their z from the integrator's position, in the CPU's swap order. Identical |
| `ksbend-opt-out` | ELEGANT_GPU_ENABLE_KSBEND=0 (suite.json) keeps KSBEND on the CPU. Identical |
| `boffaxe-field` | BOFFAXE, a quadrupole from its on-axis gradient and seven derivatives, with z subdivisions, tilt, offsets and classical radiation: the set-up (which reads the file) on the host with no particles, every call on the device. Identical |
| `boffaxe-sextupole` | BOFFAXE read as a sextupole, every derivative, every third z point: the device. Identical |
| `boffaxe-opt-out` | ELEGANT_GPU_ENABLE_BOFFAXE=0 (suite.json) keeps BOFFAXE on the CPU. Identical |
| `gfwiggler-harmonics` | GFWIGGLER with three field harmonics, third-order end poles, phases, tilt and offsets: APPLE_Track() on the device with CUDA's sin, cos and exp, so it differs by rounding |
| `gfwiggler-opt-out` | ELEGANT_GPU_ENABLE_GFWIGGLER=0 (suite.json) keeps GFWIGGLER on the CPU. Identical |

### FTABLE (5)

Since the afternoon of 2026-11-15 `FTABLE` takes any `N_KICKS`, `L1` and `L2`, edge angles, offsets and a tilt on
the device (before, one kick and no frame changes). The kick's cubic for the bend angle is elegant's (GSL's
formula), and where the field is weak its small root is a difference of numbers near 1, so one ulp of CUDA's `acos`
or `cos` moves the angle by up to 1e-7 of itself: the quadrupole cases differ from elegant by up to 5e-13 m, as
much as elegant moves when `p_central` changes by one ulp. The comparison flags `ftable-quad-kicks` and
`ftable-misaligned` for it (see `../PARITY-VALIDATION.md`).

| Case | Guards |
| --- | --- |
| `ftable-quad-kicks` | A quadrupole-and-sextupole table (`data/ftable_quad.sdds`) in 100 kicks. Flagged: amplified rounding |
| `ftable-misaligned` | 40 kicks with `DX`, `DY`, `DZ`, `TILT` and `FACTOR`. Flagged: amplified rounding |
| `ftable-bend-edges` | A 0.1 rad dipole table (`data/ftable_dipole.sdds`) with `L1`, `L2`, `E1`, `E2` in 200 kicks |
| `ftable-cpu-options` | `VERBOSE=1`, which writes elegant's phase-space file: on the CPU. Identical |
| `ftable-opt-out` | `ELEGANT_GPU_ENABLE_FTABLE=0` (from `suite.json`). Identical |

### BGGEXP with SYMPLECTIC=1 (4)

Since the afternoon of 2026-11-15 the symplectic integrator runs on the device too: the implicit midpoint rule on
the vector potential (`computeGGEVectorPotential()` in the CPU's order of operations) every `Z_INTERVAL` planes,
in a bend's frame when `IS_BEND` is set, with `BX`, `BY` and the expansion offset. Normal data only, no radiation
or particle output, no misalignment. CUDA's `sin`, `cos` and `atan2`, so rounding. The table is
`data/bggexp_small.ggrad` (quadrupole and sextupole, three gradients each, 241 planes).

| Case | Guards |
| --- | --- |
| `bggexp-symplectic` | Every fourth plane, with a uniform `BX` |
| `bggexp-symplectic-bend` | `IS_BEND` with vertex, entry and exit points, every second plane, `DXEXPANSION` |
| `bggexp-symplectic-cpu-options` | `SYNCH_RAD=1`, which stays on the CPU. Identical |
| `bggexp-symplectic-opt-out` | `ELEGANT_GPU_ENABLE_BGGEXP_SYMPLECTIC=0` (from `suite.json`). Identical |

### Filters, beam adjustments and the cooler (19)

Since 2026-11-15: `PFILTER`, `SAMPLE` and `CLEAN` compact the beam on the device in the CPU's swap order, with any
sums made on the host from copied columns in particle order. `EMITTANCE`, `TWISS` and `REMCOR` form their
transformations from host sums and apply them on the device. `MHISTOGRAM`'s minima, maxima and counts are made on
the device (integer counts, exact). `CPICKUP` reads a host copy; `CKICKER` changes delta on that copy, and only
that column goes back. Host routines given a copy run between `gpu_port_host_begin()` and `gpu_port_host_end()`,
so their own helpers stay on the host. All identical.

| Case | Guards |
| --- | --- |
| `pfilter-percentiles` | PFILTER removing the lowest 5% and highest 8% in delta, by the percentiles of a copied delta column, in two passes on the device (the CPU's swap order each). Identical |
| `pfilter-fixed-centered` | PFILTER fixing its momentum limits on the first pass, then a centered delta window from the Kahan sum of the survivors' delta, in particle order. Identical |
| `pfilter-opt-out` | ELEGANT_GPU_ENABLE_PFILTER=0 (suite.json) keeps PFILTER on the CPU. Identical |
| `sample-interval` | SAMPLE keeping every third particle looked at: the host runs sample_particles()' loop on an index array, the device gathers the rows. Identical |
| `sample-fraction` | SAMPLE keeping a random 70%: random_2 drawn on the host in the CPU's order. Identical |
| `sample-opt-out` | ELEGANT_GPU_ENABLE_SAMPLE=0 (suite.json) keeps SAMPLE on the CPU. Identical |
| `clean-stdeviation` | CLEAN removing particles past 2 standard deviations in x, t and delta: the host's centroids and deviations of the copied columns, the test on the device. Identical |
| `clean-absolute` | CLEAN by absolute deviation, then by absolute value. Identical |
| `clean-opt-out` | ELEGANT_GPU_ENABLE_CLEAN=0 (suite.json) keeps CLEAN on the CPU. Identical |
| `emittance-change` | EMITTANCE setting the horizontal geometric and the vertical normalized emittance of a beam with dispersion: the beam sums of a read-only copy (the CPU's), the transformation on the device. Identical |
| `emittance-opt-out` | ELEGANT_GPU_ENABLE_EMITTANCE=0 (suite.json) keeps EMITTANCE on the CPU. Identical |
| `twiss-from-beam` | TWISS computing its transformation from the beam on every pass: the beam's Twiss parameters from a read-only copy, the matrix on the device. Identical |
| `twiss-opt-out` | ELEGANT_GPU_ENABLE_TWISS=0 (suite.json) keeps TWISS on the CPU. Identical |
| `remcor-delta` | REMCOR removing the correlation of x, x' and y' with delta, and of x with t once only: the host's sums of copied columns, the subtraction on the device. Identical |
| `remcor-opt-out` | ELEGANT_GPU_ENABLE_REMCOR=0 (suite.json) keeps REMCOR on the CPU. Identical |
| `mhistogram-files` | MHISTOGRAM writing every kind of histogram: minima, maxima and counts on the device (integer counts, exact), the time mean of a copied column on the host. Identical |
| `mhistogram-opt-out` | ELEGANT_GPU_ENABLE_MHISTOGRAM=0 (suite.json) keeps MHISTOGRAM on the CPU. Identical |
| `cooler-pair` | An optical stochastic cooling pickup and kicker with a drift and an RF cavity between them: the pickup reads a copy of the beam, the kicker changes delta on the host's current copy and only that column goes back. Identical |
| `cooler-opt-out` | cooler-pair with ELEGANT_GPU_ENABLE_COOLER=0 (suite.json): CPICKUP and CKICKER on the CPU. Identical |

### Random elements (29)

Since 2026-11-15 `SCATTER`, `DSCATTER`, `MATTER` and `PEPPOT` run on the device with elegant's random numbers in
elegant's order. Since that afternoon the device makes those of `SCATTER`, `DSCATTER` and `MATTER` itself: the host
walks `random_2`'s sequence with dlaran's integer steps for every decision the numbers make (whether a particle
scatters, `DSCATTER`'s limits, `MATTER`'s sections with an event, its straggling with `gauss_rn()` itself, and so
which particles are lost and the order), and the device makes each particle's numbers from where they start. The
uniforms are elegant's to the bit, so `DSCATTER` and uniform `SCATTER` stay identical; the Gaussian numbers use
CUDA's `log` and `sin`, so Gaussian `SCATTER` and `MATTER` differ by rounding. `ELEGANT_GPU_PORT_DEVICE_DRAWS=0`
makes them on the host again. `PEPPOT` still draws on the host. Where a lost particle trades places with the last
one and changes which particle is drawn for next, the host follows elegant's loop. `DSCATTER`'s first call reads its file
without tracking; `ONCEPERPARTICLE` stays on the CPU. `MATTER` takes the multiple-scattering modes and elegant's
single-scattering sections, but not slots, bremsstrahlung or electron recoil. `CBSCAT` tracks on a current host
copy and sends x', y', s and delta back. The cases that use `PFILTER`, `SCATTER` or `TWISS` as CPU
elements (`small-beam-*`, `short-island-*`, `short-run-*`, `idle-pass-*`) keep them on the CPU through `suite.json`.

| Case | Guards |
| --- | --- |
| `scatter-gaussian` | SCATTER with Gaussian kicks to x, x', y, y' and delta at a probability of 0.6 from the second pass: the host walks the sequence for the decisions, the device makes the numbers and adds them, keeping the time through the delta kick. Rounding |
| `scatter-gaussian-every` | SCATTER with Gaussian kicks to x, y' and delta on every particle: no decision draws, so particle i's numbers start 6i into the sequence. Rounding |
| `scatter-uniform-probability` | SCATTER with uniform kicks at a probability of 0.3: the host's decisions, the device's uniforms. Identical |
| `scatter-host-draws` | scatter-gaussian-every with `ELEGANT_GPU_PORT_DEVICE_DRAWS=0` (from `suite.json`): the host makes the numbers. Identical |
| `scatter-uniform` | SCATTER with uniform kicks to x' and delta on every particle on passes 0 and 1, idle on pass 2. Identical |
| `scatter-opt-out` | scatter-gaussian with ELEGANT_GPU_ENABLE_SCATTER=0: SCATTER on the CPU. Identical |
| `dscatter-dp-pdf` | DSCATTER in delta from an integrated PDF with random signs, a probability and limits per pass and in total over four passes: the first call reads the file without tracking, then the host draws the numbers and the device adds them. Identical |
| `dscatter-xp-cdf` | DSCATTER in x' from a CDF on passes 1 and 2 of four. Identical |
| `dscatter-limit-total` | DSCATTER in y' with random signs and a total limit of 300 over four passes: no decision draws, the first particles up to the limit scatter. Identical |
| `dscatter-cdf-gap` | DSCATTER from a CDF covering 0.2 to 0.9 only (`data/dscatter_cdf_gap.sdds`) at a probability of 0.5: `interp()` fails for the other draws on the device too, and the host prints elegant's warnings in its order. Identical |
| `dscatter-once` | DSCATTER with ONCEPERPARTICLE=1, which the device doesn't take: the CPU tracks it. Identical |
| `dscatter-opt-out` | dscatter-dp-pdf with ELEGANT_GPU_ENABLE_DSCATTER=0: DSCATTER on the CPU. Identical |
| `matter-multiple` | MATTER thick enough for the multiple-scattering formula (1e-3 m of aluminum), with energy decay and straggling: four Gaussian numbers per particle made on the device, the straggle with `gauss_rn()` on the host. Rounding |
| `matter-losses` | MATTER at 0.6 MeV, 0.0178 m of aluminum with energy decay and straggling: about a quarter of the particles fall to gamma 1, so the host decides the losses in elegant's order and the device puts the rows there, with z and delta of the lost rows set as elegant does. Rounding |
| `matter-mcs-forced` | MATTER 1e-6 m thick with FORCE_MCS=1: the thin-scatterer multiple-scattering kick, two Gaussian numbers per particle. Rounding |
| `matter-impulse` | MATTER with L=0 and LEFFECTIVE=1e-3: the kicks without the length. Rounding |
| `matter-single` | MATTER 5e-5 m thick: elegant's single-scattering sections (PLIMIT=0.02), with energy decay and straggling; the host walks each section's draw and the device makes an event's three numbers and replays the sections. Rounding |
| `matter-gas` | MATTER as 1 m of nitrogen at 10 Pa: density from the pressure, temperature and multiplicity, radiation length computed, single scattering in sections. Rounding |
| `matter-pass-window` | MATTER on pass 1 only (START_PASS=1, END_PASS=1): an exact drift on the device on passes 0 and 2. Rounding |
| `matter-slots` | MATTER with slots, which the device doesn't take: the CPU tracks it. Identical |
| `matter-opt-out` | matter-multiple with ELEGANT_GPU_ENABLE_MATTER=0: MATTER on the CPU. Identical |
| `matter-fallback` | MATTER 1e-6 m thick with energy decay, single scattering (`PLIMIT=0.1`) in a line of drifts. Rounding |
| `matter-host-draws` | matter-multiple with `ELEGANT_GPU_PORT_DEVICE_DRAWS=0` (from `suite.json`): the host makes the numbers. Identical |
| `peppot-holes` | PEPPOT with five holes, partial transmission and scattering in the plate: the device tests the holes, the host draws the numbers and decides the losses in elegant's order, twice (before and after the plate's length). Identical |
| `peppot-opaque` | PEPPOT of zero length with three holes and no transmission. Identical |
| `peppot-opt-out` | peppot-holes with ELEGANT_GPU_ENABLE_PEPPOT=0: PEPPOT on the CPU. Identical |
| `cbscat-compton` | CBSCAT (Klein-Nishina, exact recoil) on passes 0 and 2: tracked on a current host copy, then x', y', s and delta go to the device. Identical |
| `cbscat-thomson` | CBSCAT with the Thomson cross section and the approximate recoil, at a crossing angle. Identical |
| `cbscat-opt-out` | cbscat-compton with ELEGANT_GPU_ENABLE_CBSCAT=0: CBSCAT on the CPU. Identical |

### IONEFFECTS (5)

Since 2026-11-15 the ions are tracked on a current host copy, as elegant tracks them. The Gaussian field's kicks to
the electrons go to each bunch's rows of the device's beam, with `BEAMBEAM`'s `gaussianBeamKick()`. The other
fields, and `conserve_momentum` (which needs the kicks' sum), kick the host copy, and x' and y' go back. All
identical here.

| Case | Guards |
| --- | --- |
| `ioneffects-gaussian` | IONEFFECTS with the Gaussian field on one bunch over twelve passes, off for the first two: the ions on a current host copy as elegant tracks them, the kicks to the electrons on the device with BEAMBEAM's gaussianBeamKick(). Identical here; the device's exp and Faddeeva function could differ by rounding |
| `ioneffects-bunches` | IONEFFECTS with the Gaussian field on three bunches: each bunch's kicks go to its own rows of the device's beam. Identical here; the device's exp and Faddeeva function could differ by rounding |
| `ioneffects-bigaussian` | IONEFFECTS with the bi-Gaussian fit: the kicks on the host copy, x' and y' sent back. Identical |
| `ioneffects-conserve` | IONEFFECTS with the Gaussian field and conserve_momentum=1, which needs the kicks' sum: on the host copy, x' and y' sent back. Identical |
| `ioneffects-opt-out` | ioneffects-gaussian with ELEGANT_GPU_ENABLE_IONEFFECTS=0: IONEFFECTS on the CPU. Identical |

### Transverse feedback (4)

| Case | Guards |
| --- | --- |
| `tfeedback-planes` | `TFBPICKUP` and `TFBDRIVER` on planes `y`, `delta` and `time` |
| `tfeedback-noise` | `RMS_NOISE` on the `y` and `time` pickups, drawn on the host after the device average |
| `tfeedback-bunched-one-bunch` | The default `BUNCHED_BEAM_MODE=1` with one bunch, which uses the device, and `y` pickup noise |
| `tfeedback-bunched-two-bunches` | `BUNCHED_BEAM_MODE=1` with two bunches from `&sdds_beam` `use_bunched_mode=1`. The pickups and drivers stay on the CPU and feed back on each bunch |

### Obstructions (2)

`&obstruction_data` isn't implemented on the GPU, so these cases must match the CPU with no GPU activity.

| Case | Guards |
| --- | --- |
| `obstruction-data-tracking` | Tracking stays on the CPU, with losses at obstructions in drifts and magnets and global loss coordinates |
| `obstruction-data-find-aperture` | `&find_aperture`, which would otherwise use the batched GPU search, stays on the CPU |

### Batched searches (24)

| Case | Guards |
| --- | --- |
| `search-aperture-entrance-loss` | No collimator, so particles are lost at magnet entrances. `sLost` used to differ from the CPU search by a magnet's length, because that position came from the previous particle |
| `search-aperture-one-line` | Batched `&find_aperture` in `one-line` mode |
| `search-aperture-three-line` | Batched `&find_aperture` in `three-line` mode |
| `search-aperture-eleven-line` | Batched `&find_aperture` in `eleven-line` mode |
| `search-frequency-map-delta` | Batched `&frequency_map` with `ndelta=3` |
| `search-frequency-map-always-change-p0` | `always_change_p0=1` must not use batched `&frequency_map`, because a batch shares one central momentum |
| `search-tune-footprint-always-change-p0` | The same for batched `&tune_footprint` |
| `search-tune-footprint-device-history` | Batched `&tune_footprint` without `always_change_p0`, so all turns are one tracking call and each turn's history is recorded on the device |
| `search-frequency-map-prepass-losses` | Each aperture type stops part of the grid. Under `GPU_VERIFY` the fused prepass decides who survives, so comparing with the CPU tests its losses. In the normal build the CPU tracked this until 2026-11-11; the fused program does now, as it does the two cases below, since `CCBEND` and `CSBEND` give the CPU's bits |
| `search-frequency-map-prepass-magnet-losses` | A `FODO` ring whose focusing quadrupoles are a `CSBEND` and a `CCBEND`, with a `SCRAPER` tighter than the `MAXAMP` on `-x`, so the prepass loses particles in all three. Under `GPU_VERIFY`, the run also stops if a particle the prepass kept is lost on the CPU |
| `search-frequency-map-prepass-monitors` | `MONI`, `HMON` and `VMON` with length, 3.5 m in all. The fused prepass skipped monitors, so under `GPU_VERIFY` it kept 6 particles the CPU lost and the run stopped. It now tracks them as their drift matrices |
| `search-frequency-map-csbend-openmp` | Bends that qualify for the OpenMP `CSBEND` path. Reach it with `--elegant-argument=-ompThreads=4` |
| `search-frequency-map-csbend-openmp-kinds` | Eight `CSBEND` kinds the host threads take since 2026-11-12: default edges with gap terms and `MALIGN_METHOD=0` offsets, second-order edges, edge models 2 to 4, classical radiation, expanded Hamiltonian. Tracked on the CPU; with `--elegant-argument=-ompThreads=4` the file must be the one-thread file |
| `search-frequency-map-rfca-reference` | A ring with an `RFCA` at 170 degrees. elegant sets the cavity's reference again for each particle on each turn; in the batch the cavity takes each particle as a beam of its own. One reference for the batch gave tunes off by 1e-3 |
| `search-frequency-map-rfca-two-cavities` | The same with a thin and a long three-kick cavity, each with its own reference. Batched |
| `search-frequency-map-rfca-shared-reference` | Two cavities with one `PHASE_REFERENCE` number: must not be batched |
| `search-frequency-map-rfca-change-t` | A cavity with `CHANGE_T=1`: not batched |
| `search-frequency-map-rfca-no-reset` | The same with `reset_rf_for_each_step = 0`, where the reference is the fiducial particle's: the map is batched and must still be identical |
| `search-aperture-apcontour-sticky` | The device-resident `n-line` search with a sticky `APCONTOUR` first in the ring: the multipoles check it inside, every element after it checks it at its end, and losses get the CPU's `s`. Its first run caught the multipoles' contour check moving every point to the origin, with no aperture data packed to undo their tilt |
| `search-aperture-apcontour-sticky-midring` | The same contour halfway around: it would carry into the next pass ahead of where it's set, so the search leaves the resident program. A build that kept it there failed this case |
| `search-aperture-apcontour` | A non-sticky contour with `XFACTOR` and `YFACTOR`, checked only where it is |
| `search-aperture-apcontour-two` | Two sticky contours, the second with `TILT` and `DX`, then `CANCEL`, with several contours on the device at once |
| `search-momentum-matrix-elements` | The independent (non-fiducial) batched `&momentum_aperture` through `QUAD`, `SBEN`, `RBEN`, `SEXT`, `MONI`, `HMON`, `VMON`, `KICKER` and `EKICKER`, which it used to refuse. In auto mode, so the case has GPU activity only when the batch runs |
| `search-momentum-matrix-elements-fiducial` | The same, fiducial, with a `CSBEND` with `SYNCH_RAD`. Fiducial searches go to the independent search first since 2026-11-09 |
| `search-aperture-csbend-high-order` | The batched `n-line` `&find_aperture` through bends with `K3` to `K8`, which the batched search refused until 2026-11-09. In auto mode, so the case has GPU activity only when the batch runs |
| `search-momentum-csbend-high-order` | A fiducial `&momentum_aperture` through the same ring, 32 locations, on the independent search |
| `search-momentum-generic` | The generic batched fiducial `&momentum_aperture`, which takes searches the independent one refuses (here for a `WATCH`), on the `K3` to `K8` ring, 32 locations. Its trial momenta step as the scalar search's do, and since 2026-11-09 it tracks each task's last survivor and first loss again on the CPU, so it matches the CPU exactly. A build without the loss re-check put it back within rounding; one with the old `start + i*interval` made 14 of 32 momenta differ in the last bit. Both stay inside the envelope, so the comparison with the GPU reference catches them |
| `search-aperture-radiation` | The batched `n-line` `&find_aperture` through `CSBEND`s and `KQUAD`s with `SYNCH_RAD`, which the batched search refused until 2026-11-09. In auto mode, so the case has GPU activity only when the batch runs |
| `search-momentum-radiation` | A fiducial `&momentum_aperture` through the same ring, 32 locations. Fiducial searches now go to the independent search first, which confirms every boundary on the CPU, so it matches the CPU exactly; on the generic path, which used to take it, lost positions differed by up to 7.6e-9 |
| `search-momentum-after-scalar` | A single-particle search below the batch threshold, then a batched one in a second `run_setup`. The first used to leave its tracking wedge installed, so the second stayed on the CPU and the case had no GPU activity |

### Tune histories from the fused program (12)

`&frequency_map` and `&tune_footprint` track a batch of grid points for their turn-by-turn histories.
`gpu_batched_tune_program_track()` in `gpu_stub.c` tracks the whole batch on the device, one kernel launch a turn,
and `computeTunesFromTrackingBatch()` in `twiss.cc` uses its histories only if the CPU, tracking a sample of the
particles itself, gets the same bits. The program runs only where every element is in a configuration whose device
arithmetic is the CPU's; otherwise the CPU tracks, as before. Under `GPU_VERIFY` the CPU tracks every particle, and
a difference in a program that claims the CPU's bits stops the run. With verbose output, "its histories are used"
says the program's results went into the file.

| Case | Guards |
| --- | --- |
| `search-frequency-map-program` | A 15 by 15 map through matrices, an exact drift, `KQUAD`s, `KSEXT`s, `MALIGN`s on the first pass, the second and every pass, a collimator that takes the outer points, a `MAXAMP`, a `WATCH` and a `CHARGE`. Two intervals, default grid output (a row only for a point with both tunes). Identical to the CPU |
| `search-frequency-map-program-full-grid` | The same with `full_grid_output=1` and three momentum offsets |
| `search-frequency-map-program-opcodes` | One of each operation the program claims the CPU's bits for: correctors, exact correctors with no kick (the CPU makes those a drift), a thin `MULT`, `KQUAD` and `KSEXT` of order 4 and 6 with offsets, tilts, pitch and yaw, one with `LEFFECTIVE`, a `DQCOR`, sticky tapered apertures, collimators, a scraper, a `SPEEDBUMP`, a `MAXAMP`, an `RFCA` with no voltage, a monitor with length. The verify build stops here if one of them differs. `LEFFECTIVE` was not the CPU's bits until 2026-11-11: its end drifts go on the path coordinate itself, before and after the body's |
| `search-frequency-map-program-bend-stays` | The first ring with one `CSBEND` whose edges have a fringe angle (`HGAP` and `FINT`). Such an edge takes `tan(e - psi/(1+delta))` for each particle, from the device's library where the CPU calls the host's, so the program doesn't claim the CPU's bits for it: the program isn't run, and the file is the CPU's. (A plain `CSBEND` until 2026-11-11, when it became exact) |
| `search-frequency-map-program-inexact-stays` | A `MULT` with length, where the device differs from the CPU by 3e-14: the program isn't run |
| `search-frequency-map-program-malign-methods` | Kick magnets with each `MALIGN_METHOD`, with and without offsets, and with `DZ`. Two of these were not the CPU's bits until the kernel was corrected on 2026-11-11: a linear method with no misalignment (the CPU transforms at the entrance and not at the exit) and `DZ` (its path is added before the element's own). The program tracks the map; the verify build stops if any particle differs |
| `search-frequency-map-program-radiation-stays` | `KQUAD`s with `SYNCH_RAD=1`. The program's multipole operation has no radiation, so the program isn't built and the CPU tracks. A program that tracked them would leave the radiation out |
| `search-frequency-map-program-ccbend` | The first ring with a `CCBEND` in every cell, five kinds: the defaults; order 6 with the `FRINGEMODEL=1` fringe and `K1` to `K3`; order 2 with no fringe, which has its own kernel; a negative angle with `TILT`, `YAW` and a linear `MALIGN_METHOD`; `SYNCH_RAD` with an exact `MALIGN_METHOD`. The device's `CCBEND` was not the CPU's bits until 2026-11-11: the CPU divides `qx` and `qy` by 1+delta and multiplies them back after every slice, radiation or not, and the device now does the same. The program tracks the map with no tolerance; the verify build stops if any particle differs |
| `search-frequency-map-program-csbend` | The first ring with a `CSBEND` in every cell, eight kinds: the defaults; order 6 with `K1` to `K3` and the default edges; order 2 with second-order edges and `H1`, `H2`; K. Hwang's fringe with `HGAP` and `FINT`; a negative angle with `TILT`, a linear `MALIGN_METHOD` and the simple symplectic edge kick; `SYNCH_RAD` with an exact `MALIGN_METHOD`; `EXPAND_HAMILTONIAN`; a weak bend of radius 3e4 m. Three momentum offsets of 5e-3. A `CSBEND` was not the CPU's bits until 2026-11-11: elegant and gpu-elegant now both drift with constants made once on the host and no trigonometric function per particle, and the device repeats the CPU's division and multiplication by 1+delta after every kick. The program tracks the map with no tolerance; the verify build stops if any particle differs |
| `search-frequency-map-program-tolerance` | A ring with elements that take a library function for each particle, from the device's library where the CPU calls the host's: a `CCBEND` whose `FRINGEMODEL=1` fringe has `I1` and `K5` integrals (`exp()`), and exact correctors with kicks (trigonometric functions). The program doesn't claim the CPU's bits for those, so it runs because the suite sets `ELEGANT_GPU_TUNE_HISTORY_TOLERANCE=1e-6`, and its histories are used |
| `search-frequency-map-program-fallback` | Eight `MULT`s with length, which differ from the CPU in every particle, and a tolerance of 1e-30: the program runs, the sampled histories differ by more than that, and the CPU tracks the rest of the grid. Identical to the CPU. (`CSBEND`s until they became exact) |
| `search-tune-footprint-program` | A batched `&tune_footprint` through a FODO ring of kick magnets. Before, only `POLYNOMIALSERIES` maps were batched. Its momentum scan and x-y grid, two intervals each, are tracked by the program; both files are the CPU's |

### SLICE on the device (6)

Since the evening of 2026-11-15 the `SLICE` element runs with the beam on the device, its numbers the CPU's to
the bit: the device makes the particles' times and the bunch's earliest and latest (the first particle with either
wins a tie, as the CPU's loop has it) and, from the slices' lower limits the host makes by the CPU's repeated
addition, each slice's members in particle order (a stable sort by slice) with their columns side by side; those
come to the host, which adds every sum `accumulate_beam_sums()` makes for a slice in that order with
`KahanPlus()`'s steps, all of a slice's sums in one pass. A call the device can't slice (no particles, slices of
no width, which take the whole beam) gets a read-only host copy and the host's code. `slice-element-ranges` and
`slice-element-quantized-times` above run on the device too. All identical.

| Case | Guards |
| --- | --- |
| `slice-device-passes` | 50 slices of 20,000 particles every second pass from pass 1 to 5 of seven (`INTERVAL`, `START_PASS`, `END_PASS`), and one with `REFERENCE_FREQUENCY` |
| `slice-device-pid-reversed` | `START_PID` above `END_PID`: every particle timed, none summed, every slice empty (0/0 sigmas); and only negative IDs, which time no particle and go to the host |
| `slice-device-one-time` | First in the line on a beam with no spread: one time on pass 0 (slices of no width, on the host), a little spread from the drifts on pass 1 (the device) |
| `slice-device-all-lost` | After a collimator that stops every particle: elegant's empty slices with no particles (`RUN_ZERO_PARTICLES`) |
| `slice-device-spin` | Spin tracking: rows of more than seven columns |
| `slice-opt-out` | `ELEGANT_GPU_ENABLE_SLICE=0` (from `suite.json`): the host's single pass over the beam |

### Slice analysis (2)

| Case | Guards |
| --- | --- |
| `slice-analysis` | `&slice_analysis` with 8 slices after every element, the beam on the device and no `centroid` or `sigma` file, so the time coordinates, the bunch's span and every slice's sums come from the device, each sum over the particles in order. Its `run.slan` is byte-identical to the same binary's with `ELEGANT_GPU_ENABLE_SLICE_SUMS=0`, which brings the beam to the host instead |
| `slice-analysis-whole-beam` | The whole beam only (`n_slices=0`) from `s_start=0.5` to `s_end`. The first element ends before `s_start`, so the page must start at the first element in range; the CPU used to write a row before it started a page, and stop (bug 14) |

### Small beams (9)

A beam of 1,000 to 9,999 particles goes on the device when `gpuChooseSmallBeamThresholds()` in `gpu_stub.c`
estimates that the lattice pays for it. These run in auto mode with the default thresholds. The kick quadrupole,
kick sextupole and matrix kernels give the CPU's bits, so each lattice has a parameter-mode `WATCH` where the beam
would be on the device: the device adds its sums in its own order, and they differ from the CPU's in the last digits.
A case that should stay on the CPU is identical to the CPU, and one that should go to the device is not.
Since `PFILTER` runs on the device (2026-11-15), the cases that use it to split the lattice keep it on the CPU with
`ELEGANT_GPU_ENABLE_PFILTER=0` from `suite.json`.
`ELEGANT_GPU_SMALL_BEAM_REPORT=1` prints each call's estimate.

| Case | Guards |
| --- | --- |
| `small-beam-kick-ring` | 2,000 particles for 10 passes in a ring of `KQUAD`s and `KSEXT`s: an estimated 17 ms a pass on the CPU and 2.6 ms on the device, so the thresholds come down and the beam goes on the device. A build that never lowers them has no GPU activity here |
| `small-beam-opt-out` | The same with `ELEGANT_GPU_ENABLE_SMALL_BEAMS=0`: identical to the CPU |
| `small-beam-context-not-worth-it` | One pass of 1,000 particles through that ring with no CUDA context yet. The suite runs it without `ELEGANT_GPU_VERBOSE`, which starts one. The 0.006 s the device would save is under the 0.3 s a context is worth, so it is identical to the CPU |
| `small-beam-copies-cost-more` | 2,000 particles through runs of six matrix elements between `PFILTER`s, which the device doesn't take. The 20 copies each way cost more than the runs save: identical to the CPU. A build that counts copies as free sends the beam to the device |
| `small-beam-concatenated-ring` | 1,500 particles for 40 passes with `concat_order=1`. The estimate has to walk the concatenated elements, six for the device and a `PFILTER`, which don't pay: identical to the CPU. Walking the lattice's 123 elements sends the beam to the device |
| `small-beam-short-runs-stay` | Ten lone `KQUAD`s between `PFILTER`s, then six matrix elements, 1,500 particles, with short runs judged by their element count alone (`ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0`). Tracking then keeps a run as short as one magnet on the CPU, so the estimate must leave the ten out, and the six alone don't pay: identical to the CPU |
| `small-beam-short-runs-pay` | The same lattice with short runs judged by cost, the default. A lone `KQUAD` of 16 slices is estimated at 0.62 ms on the CPU and 0.18 ms on the device with its two copies, so tracking takes each, the estimate counts them, and the thresholds come down. A build whose estimate leaves the short runs out keeps the beam on the CPU and the case has no GPU activity |
| `small-beam-ring-no-copies` | 1,500 particles for 30 passes in a ring of six matrix elements. From the second pass on the beam is never copied, so the estimate (135 us a pass against 68 us) lowers the thresholds to its break-even count, 1,063. A build that charges every pass the first pass's copies keeps the beam on the CPU and the case has no GPU activity |
| `small-beam-returns-after-losses` | 12,000 particles cut to about 8,900 by a collimator on the first pass, in the kick ring with a `PFILTER` that sends the beam to the host each pass. The thresholds come down for the call, so the beam returns to the device after each `PFILTER`. The class doesn't show a build that leaves it on the CPU after the first pass (the first pass is on the device either way); the comparison with the GPU reference does |

### Short runs between CPU-only elements (5)

When the host holds the beam and only a few elements the device can take lie before the next one it can't,
`gpuShortGpuIslandAhead()` in `gpu_stub.c` decides whether the run is worth the beam's trip to the device and back.
Until 2026-11-13 it went by the number of elements alone (a matrix 1, a magnet 2, the device from more than 4), whatever the
beam. Now a run that is short by that count still goes when its estimated cost on the CPU is 1.4 times its cost on
the device or more: 15 ns a particle for a first-order matrix, 60 ns for a higher order, the kick and slice numbers
of the small-beam estimate, 20 us and 10 ns a particle for each copy. These cases use the parameter-mode `WATCH` of
the small-beam cases to show where the beam was. The first three run in auto mode with the default thresholds, the
last two with the suite's thresholds of 64 so that only this rule decides.
The `PFILTER`s that end the runs are kept on the CPU with `ELEGANT_GPU_ENABLE_PFILTER=0` from `suite.json`, since
`PFILTER` itself runs on the device from 2026-11-15.

| Case | Guards |
| --- | --- |
| `short-island-goes` | 12,000 particles through runs of three matrix elements between `PFILTER`s. 0.54 ms on the CPU against 0.31 ms on the device: every run goes. A build that goes by the count alone, or that prices the run by its first element, never uses the device |
| `short-island-first-order-stays` | 12,000 particles, 2 passes, first-order quadrupoles each alone between `PFILTER`s. 0.18 ms on the CPU against 0.29 ms for the trip: they stay, and the output is identical to the CPU. A build that sends every short run, counts copies as free or per particle as free, or prices every matrix as second order changes the `WATCH` sums |
| `short-island-second-order-goes` | The same with second-order quadrupoles: 0.72 ms on the CPU. The order is known once the matrix exists, so the first pass stays and the second goes. A build without the higher-order price never uses the device |
| `short-run-threshold-5000-stays` | 5,000 particles through runs of three first-order matrix elements: 225 us on the CPU, under 1.4 times the device's 170 us. Identical to the CPU. A margin of 1, or copies priced too low, sends the runs to the device |
| `short-run-threshold-6500-goes` | The same lattice at 6,500 particles, past the 5,765 where the rule changes sides: 292 us against 1.4 times 200 us. A build that goes by the count alone never uses the device |
