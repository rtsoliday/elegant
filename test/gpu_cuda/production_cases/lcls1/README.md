# LCLS1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/LCLS1`

This wrapper keeps the production LCLS lattice and beam files external to the repository.  It is similar to `lcls0`, but keeps the deterministic alignment-error setup from the source input so the smoke set covers production error tables as well as `RFCW`, `WAKE`, `CSRCSBEND`, `CSRDRIFT`, and aperture elements.

Quick mode uses `sample_fraction=0.05` to keep the 199999-row source beam bounded.  Baseline mode scales only the sample fraction toward the selected target time.

Action 6 uses this wrapper as the production validation target for the
wake-bearing matrix-method `RFCW` CUDA slice.  The May 8, 2026 quick run
matched all 16 common SDDS files at `1e-11`, reported `wakes=1186` with no
RFCW CPU-element synchronization, and ran 1.79x faster than the paired CPU
quick run.
