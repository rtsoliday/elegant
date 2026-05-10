# Phase 66 SCMULT Sliced Fallback

Focused action-8 guard for sliced linear `SCMULT` when
`ELEGANT_GPU_ENABLE_SCMULT=1` is set.  The lattice mirrors the linear
`phase7_scmult_linear` drift ring, but requests a nonzero `slice_duration` so
slice-based space-charge kicks remain CPU-owned while surrounding supported
elements may stay on the GPU.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 2,000 particles and 2 passes matched all 5 common
  SDDS files at `1e-11`, including `.twi`.
- With `ELEGANT_GPU_ENABLE_SCMULT=1`, CUDA reported 72 matrix kernels,
  0 resident `SCMULT` kernels, and 48 `trackThroughSCMULT fallback`
  synchronizations.
- Reports:
  `test/gpu_cuda/output/reports/action8-scmult-deferred-fallback.md` and
  `test/gpu_cuda/output/reports/action8-scmult-deferred-fallbacks.md`.
