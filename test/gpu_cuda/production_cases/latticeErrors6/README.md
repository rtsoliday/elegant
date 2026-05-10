# latticeErrors6 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/latticeErrors6`

This action-8 wrapper is a bounded high-count `UKICKMAP` gate extracted from
the `latticeErrors6` septum-map section.  It uses the ten production
`skick-*.sdds` map files by absolute path and repeats that sequence four times
per pass, giving 40 `UKICKMAP` elements per pass without pulling in the full
ring's unrelated optics, multipole-error, closed-orbit, and diagnostic work.

The generated bunch is intentionally offset and broad enough to exercise map
losses without requesting a `.los` file.  That makes this case suitable for
checking the opt-in resident stable map-loss compaction path under
`ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION=1`.

May 9, 2026 action-8 validation:

- The 30,000-particle, 2-pass CPU/GPU compaction gate matched all 4 common
  SDDS files at `1e-11` and kept 564 survivors.
- CPU time was 0.77s.  The opt-in compaction GPU run took 0.41s and
  synchronized only at final `gpuBaseDealloc`.
- The same GPU workload without compaction still matched CPU, but requested 80
  `UKICKMAP particle loss fallback` synchronizations and ran in 0.92s.
- Reports:
  `test/gpu_cuda/output/reports/action8-latticeErrors6-map-loss-compaction.md`
  and
  `test/gpu_cuda/output/reports/action8-latticeErrors6-map-loss-fallbacks.md`.
