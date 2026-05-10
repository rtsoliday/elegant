# latticeErrors6 UKICKMAP Global-Loss Wrapper

Source: `/home/soliday/oag/apps/src/elegantTestSet/latticeErrors6`

Action-8 production-profile wrapper for the same high-count septum-map
`UKICKMAP` slice as `latticeErrors6`, but with `.los`, `.acc`, and
`losses_include_global_coordinates=1` enabled.

Run with `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1` to confirm that the
opt-in resident map-loss compaction path still defers UKICKMAP global loss-row
semantics to the explicit `UKICKMAP particle loss fallback` path.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 3,000 particles and 2 passes matched all 6 common
  SDDS files at `1e-11`, including `.los` and `.acc`.
- The `.los` file includes the global loss-coordinate columns `X`, `Z`, and
  `thetaX`.
- CUDA reported 73 `UKICKMAP particle loss fallback` synchronizations under
  `ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`, confirming global loss rows
  remain CPU-owned.
- Reports:
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallbacks.md`.
