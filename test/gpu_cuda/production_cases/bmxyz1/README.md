# bmxyz1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/bmxyz1`

This Phase 18 profile wrapper exercises a deterministic `BMXYZ` field-map case with the same map file used by the source input deck.  It also enables `PARTICLE_OUTPUT_FILE` so CPU/GPU comparison can include particle-coordinate and sampled-field output from inside the map.

The current CUDA implementation is expected to use CPU fallback for `BMXYZ`.  This wrapper is intended to establish correctness and timing evidence before attempting any field-map kernels.
