# bmapxy1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/bmapxy1`

This Phase 18 profile wrapper exercises a small deterministic `BMAPXY` field-map case.  The source field map is referenced by absolute path so the wrapper can run from the shared GPU benchmark harness without copying the SDDS-like map fixture into this repository.

The current CUDA implementation is expected to use CPU fallback for the field-map element.  This wrapper exists to bound runtime, preserve CPU/GPU output comparison coverage, and provide a small target before any future `BMAPXY` CUDA work.
