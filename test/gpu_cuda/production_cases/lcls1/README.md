# LCLS1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/LCLS1`

This wrapper keeps the production LCLS lattice and beam files external to the repository.  It is similar to `lcls0`, but keeps the deterministic alignment-error setup from the source input so the smoke set covers production error tables as well as `RFCW`, `WAKE`, `CSRCSBEND`, `CSRDRIFT`, and aperture elements.

Quick mode uses `sample_fraction=0.05` to keep the 199999-row source beam bounded.  Baseline mode scales only the sample fraction toward the selected target time.
