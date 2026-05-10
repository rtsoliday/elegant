# Phase 64 KICKMAP Global-Loss Fallback

Focused action-8 fixture for ordinary `GKICKMAP` map losses when `.los` output
requests global loss-coordinate columns.  It reuses the deterministic
`phase62_kickmap_loss_compaction` map and broad bunch, but adds
`losses_include_global_coordinates=1`.

Run this case to verify the default resident KICKMAP compaction policy still
defers global loss-row semantics to the explicit `KICKMAP particle loss
fallback` path.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 3,000 particles and 3 passes matched all 6 common
  SDDS files at `1e-11`, including `.los` and `.acc`.
- The `.los` file includes the global loss-coordinate columns `X`, `Z`, and
  `thetaX`.
- CUDA reported 15 `KICKMAP particle loss fallback` synchronizations,
  confirming global loss rows remain CPU-owned.
- Reports:
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-kickmap-loss-output-fallbacks.md`.
