# boffaxe1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/boffaxe1`

This Phase 18 profile wrapper exercises the source `BOFFAXE` off-axis expansion case with the production gradient table referenced by absolute path.  It enables element particle/field diagnostic output so future CUDA work can compare particle-level and field-grid behavior.

The current CUDA implementation is expected to use CPU fallback for `BOFFAXE`.  This wrapper keeps the timing bounded and makes the diagnostic outputs reproducible before any field-map acceleration is attempted.

The source case's matrix-output path is intentionally left out of this quick wrapper because it forces an expensive tracking-based matrix determination through the field map.  Add it as a separate heavier follow-up case after quick particle/field comparisons are stable.
