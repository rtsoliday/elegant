# Production Cases

These wrappers reference production inputs from `/home/soliday/oag/apps/src/elegantTestSet/` without copying large SDDS fixtures into this repository.

The bounded smoke subset can be run end-to-end with:

```sh
./test/gpu_cuda/production_smoke.sh --label-prefix phase11
```

Use `--require-gpu` when the command should fail instead of falling back to CPU execution if CUDA device access is unavailable.  The helper runs the subset once with the CPU binary, once with `gpu-elegant`, then compares the run directories with `compare_sdds.py`.  The wrappers use fixed seeds, disable progress chatter with `tracking_updates=0`, enable `show_element_timing=1`, and keep scaling explicit through `metadata.tsv`.

The current smoke set is `lcls0`, `lcls1`, `clic1`, `csbend1`, `maxamp1`, `collimate1`, `collimate2`, `collimate3`, and `dqcor1`.  The `metadata.tsv` file records each source path, dominant element families, deterministic status, quick/baseline scaling rule, expected CUDA path or fallback behavior, and output coverage.

Phase 17 magnet-profile work can be refreshed with:

```sh
python3 test/gpu_cuda/profile_magnet_features.py /home/soliday/oag/apps/src/elegantTestSet
```

Phase 18 SCMULT, Poisson, field-map, wiggler, and ion-effect profiling can be refreshed with:

```sh
python3 test/gpu_cuda/profile_phase18_features.py /home/soliday/oag/apps/src/elegantTestSet
```

The profilers write Markdown and TSV summaries under `test/gpu_cuda/output/reports/`.  They are static scanners for choosing follow-up CUDA targets; runtime timing and SDDS comparisons are still required before enabling any new path.

Phase 18 profile wrappers include the SCMULT pair `scRing2` and `scRing2_no_watch`, plus the field-map and wiggler wrappers `bmapxy1`, `bmxyz1`, `boffaxe1`, `cwiggler10`, and `uKickMap1`.  These are intentionally not part of the default smoke subset; run them explicitly with `run_benchmarks.sh --case NAME` until their timing and output tolerances are recorded.

Use `scRing2_no_watch` to quantify WATCH synchronization cost without changing the source production lattice.  It disables WATCH diagnostics through `&alter_elements`, so `.w1` output is intentionally absent while the common physics outputs should remain comparable.  The May 7, 2026 Phase 18 thin-RFCA measurement reduced normal no-WATCH CUDA sync accounting to one RFCA setup request plus deallocation and ran the 601-pass case in 22.69s, versus 22.86s for standard `scRing2` with WATCH output enabled.

Use `uKickMap1` to exercise the narrow deterministic `UKICKMAP` CUDA prototype against the production kick-map SDDS file.  The May 7, 2026 timing gate matched all 4 common SDDS files at `1e-11` and, after device map-array caching, ran the 30,000-particle, 2,000-pass workload in 13.58s on `gpu-elegant`, versus 55.43s on CPU.

The Phase 18 diagnostic wrappers use nonstandard SDDS suffixes for field-map internals.  Include them in comparisons with:

```sh
python3 test/gpu_cuda/compare_sdds.py CPU_DIR GPU_DIR --tolerance 1e-11 --extensions cen,fin,out,sig,pout,field,cwigOut,w1
```

Keep generated timing and reference output under `test/gpu_cuda/output/` or another ignored directory.  Do not add large production wrappers to default quick CI until their runtime and comparison tolerance are proven bounded.
