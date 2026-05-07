# uKickMap1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/uKickMap1`

This Phase 18 wrapper exercises the deterministic `UKICKMAP` interpolation/kick path with the production kick-map SDDS file referenced by absolute path.  It generates a bounded multi-particle bunch instead of using the source deck's two-dimensional one-particle scan, so timing scales through particle count and pass count under the shared benchmark harness.

The CUDA path is intentionally narrow: no radiation, no ISR, no map-element misalignment, no yaw, and CPU fallback if any particle leaves the map.

May 7, 2026 evidence: CPU quick, normal CUDA quick, and `GPU_VERIFY` quick runs matched all 4 common SDDS files at tolerance `1e-11`, with `GPU_VERIFY` reporting exact `trackUndulatorKickMap` agreement for the exercised steps.  After adding a device map-array cache, the 30,000-particle, 2,000-pass timing gate matched CPU output at `1e-11` and ran in 13.58s on `gpu-elegant` versus 55.43s on CPU, a 4.08x speedup.  That long run still had 337 late particle-loss fallback synchronizations, so resident loss handling remains a follow-up.
