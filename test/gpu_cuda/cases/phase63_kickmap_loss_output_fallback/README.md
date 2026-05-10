# Phase 63 KICKMAP Loss-Output Fallback

Focused action-8 fixture for ordinary `GKICKMAP` map losses when `.los` and
`.acc` output rows are requested.  The particle distribution and map are shared
with `phase62_kickmap_loss_compaction`, but loss output makes host loss-row
ordering authoritative.

Run this case with `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1` to verify the
resident KICKMAP compaction flag still preserves `.los` semantics by using the
explicit `KICKMAP particle loss fallback` path.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 3,000 particles and 3 passes matched all 6 common
  SDDS files at `1e-11`, including `.los` and `.acc`.
- CUDA reported 15 `KICKMAP particle loss fallback` synchronizations under
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, confirming loss-output rows
  remain CPU-owned.
- Reports:
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallbacks.md`.
