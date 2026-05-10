# LCLS0 Production Case

This Phase 0 production benchmark wraps the existing elegant test-set case at:

`/home/soliday/oag/apps/src/elegantTestSet/LCLS0`

It was selected because it is a realistic LCLS linac tracking input with a 199999-row SDDS beam, production wake tables, many `RFCW` elements, `WAKE`, `CSRCSBEND`, `CSRDRIFT`, and aperture/collimator elements in the lattice.  The quick benchmark uses a 10 percent sample of the input beam, which keeps runtime bounded while still exercising particle-heavy collective paths.

The wrapper intentionally references the source test-set files instead of copying large SDDS fixtures into this repository.

Action 6 uses this wrapper as the production validation target for the
wake-bearing `RFCW,N_KICKS=1` CUDA slice.  The May 8, 2026 quick run matched
all 15 common SDDS files at `1e-11`, reported `wakes=1186` with no RFCW
CPU-element synchronization, and ran 1.41x faster than the paired CPU quick
run.

Known note: the source lattice uses the deprecated `N_KICKS` alias on `CSRCSBEND`, so elegant prints deprecation warnings during parsing.  The wrapper preserves that source behavior instead of rewriting the production lattice.
