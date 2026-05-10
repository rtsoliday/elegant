# Phase 67 SCMULT Multi-Bunch Fallback

Focused action-8 guard for multi-bunch linear `SCMULT` under the guarded
SCMULT policy.  The tracking lattice mirrors the
linear `phase7_scmult_linear` drift ring, but the case first generates a
single seed bunch and then reloads it with `use_bunched_mode=1` plus a
duplicate time stagger so `SCMULT` sees multiple buckets.  That keeps the
multi-bunch space-charge kick CPU-owned while surrounding supported elements
may stay on the GPU.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 1,000 seed particles per bunch and 2 passes matched
  all 7 common SDDS files at `1e-11`, including the seed `.out`/`.fin` files
  and tracked `.twi`.
- With the guarded SCMULT path enabled, CUDA reported 72 matrix kernels,
  0 resident `SCMULT` kernels, 2 `initializeSCMULT fallback`
  synchronizations, and 48 `accumulateSCMULT fallback` synchronizations.
- Reports:
  `test/gpu_cuda/output/reports/action8-scmult-multibunch-fallback.md` and
  `test/gpu_cuda/output/reports/action8-scmult-multibunch-fallbacks.md`.
