# latticeErrors6 UKICKMAP Loss-Output Wrapper

Source: `/home/soliday/oag/apps/src/elegantTestSet/latticeErrors6`

Action-8 production-profile wrapper for the same high-count septum-map
`UKICKMAP` slice as `latticeErrors6`, but with `.los` and `.acc` output
enabled.  It repeats ten production kick maps four times per pass and keeps the
same intentionally broad beam used for the no-loss-output compaction gate.

Run this case to confirm that the default resident map-loss compaction policy
still defers UKICKMAP loss-row output to the explicit `UKICKMAP particle loss
fallback` path.

May 9, 2026 action-8 validation:

- CPU/GPU quick runs with 3,000 particles and 2 passes matched all 6 common
  SDDS files at `1e-11`, including `.los` and `.acc`.
- CUDA reported 73 `UKICKMAP particle loss fallback` synchronizations,
  confirming loss-output rows remain CPU-owned.
- Reports:
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallback.md` and
  `test/gpu_cuda/output/reports/action8-ukickmap-loss-output-fallbacks.md`.
