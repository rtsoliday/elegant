# Phase 62 KICKMAP Loss Compaction

Focused action-8 fixture for the ordinary non-undulator `KICKMAP` path using
elegant's `GKICKMAP` element name.  The
local SDDS map uses dimensionless `xpFactor`/`ypFactor` columns, unlike
`UKICKMAP`, and the generated bunch is intentionally broad enough to lose
particles at the map boundary without requesting `.los` output.

Run this case to validate the default shared stable map-loss compaction path for
ordinary `KICKMAP`.  With `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=0`, the
checked CUDA path should still match CPU by falling back on detected map loss.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 3,000 particles and 3 passes matched all 4 common
  SDDS files at `1e-11`.
- With default magnet loss compaction, CUDA reported 30 magnet kernels for 15
  `GKICKMAP` tracks plus 15 stable compaction passes and only final
  `gpuBaseDealloc` synchronization.
- The same GPU workload without compaction matched CPU but requested 15
  `KICKMAP particle loss fallback` synchronizations.
- Reports:
  `test/gpu_cuda/output/reports/action8-kickmap-map-loss-compaction.md` and
  `test/gpu_cuda/output/reports/action8-kickmap-map-loss-fallbacks.md`.
