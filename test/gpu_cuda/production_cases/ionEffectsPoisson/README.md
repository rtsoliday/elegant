# ionEffectsPoisson Production-Like Case

This action-8 wrapper is a small synthetic `IONEFFECTS`/Poisson gate.  The
local production test set does not include an `IONEFFECTS` or `BEAMBEAM` deck,
so this case supplies the minimum production-like inputs needed to exercise the
CPU-owned ion-effects setup before any CUDA work in `poisson.cc` is considered:
a `CHARGE` element, pressure-profile SDDS data, ion-property SDDS data,
positive ion spans, and a 16x16 Poisson grid.

The current CUDA expectation is explicit CPU fallback for `IONEFFECTS` and the
Poisson field solve.  Use this wrapper to keep that fallback visible in reports
and to provide a stable future correctness gate if a CUDA Poisson path is
prototyped.

May 9, 2026 action-8 validation:

- CPU/GPU auto-mode quick runs with 2,000 particles and 3 passes matched all 4
  common SDDS files at `1e-11`.
- CUDA reported two `CPU element: IONEFFECTS` synchronizations, one
  short-GPU-island skip, and final `gpuBaseDealloc` synchronization.
- The run emitted the expected warning that the Poisson solver for
  `IONEFFECTS` is still under testing/debugging.
- Reports:
  `test/gpu_cuda/output/reports/action8-ion-effects-poisson-fallback.md` and
  `test/gpu_cuda/output/reports/action8-ion-effects-poisson-fallbacks.md`.
