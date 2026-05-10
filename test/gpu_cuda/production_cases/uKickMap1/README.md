# uKickMap1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/uKickMap1`

This Phase 18 wrapper exercises the deterministic `UKICKMAP` interpolation/kick path with the production kick-map SDDS file referenced by absolute path.  It generates a bounded multi-particle bunch instead of using the source deck's two-dimensional one-particle scan, so timing scales through particle count and pass count under the shared benchmark harness.

The CUDA path is intentionally narrow: no radiation, no ISR, no map-element misalignment, and no yaw.  Map losses fall back to CPU by default; `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1` enables the opt-in resident stable compaction path when no loss-output or global loss-coordinate rows are needed.

May 7, 2026 evidence: CPU quick, normal CUDA quick, and `GPU_VERIFY` quick runs matched all 4 common SDDS files at tolerance `1e-11`, with `GPU_VERIFY` reporting exact `trackUndulatorKickMap` agreement for the exercised steps.  After adding a device map-array cache, the 30,000-particle, 2,000-pass timing gate matched CPU output at `1e-11` and ran in 13.58s on `gpu-elegant` versus 55.43s on CPU, a 4.08x speedup.  That long run still had 337 late particle-loss fallback synchronizations.

May 9, 2026 action-8 evidence: the 3,000-particle, 2,000-pass resident-compaction gate matched all 4 common SDDS files at `1e-11` with `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`.  The CPU run took 5.54s, the compaction GPU run took 1.82s, and CUDA synchronized only at final `gpuBaseDealloc`.  The same GPU workload without compaction had 101 `UKICKMAP particle loss fallback` synchronizations.  The quick `GPU_VERIFY` run also passed exact `trackUndulatorKickMap` CPU-shadow checks for the no-loss slice.  Report: `test/gpu_cuda/output/reports/action8-ukickmap-map-loss-compaction.md`.
