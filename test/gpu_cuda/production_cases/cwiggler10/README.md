# cwiggler10 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/cwiggler10`

This Phase 18 profile wrapper exercises a deterministic `CWIGGLER` case with external harmonic field files and field-output diagnostics.  It keeps the source data files external and referenced by absolute path so the repository does not grow large binary fixtures.

The current CUDA implementation is expected to use CPU fallback for `CWIGGLER`.  This wrapper is for bounded CPU/GPU comparison and for choosing whether a future wiggler kernel is worth implementing.
