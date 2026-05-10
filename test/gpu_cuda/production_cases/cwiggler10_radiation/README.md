# cwiggler10 Radiation Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/cwiggler10`

This action-10 wrapper turns on `SYNCH_RAD` and `ISR` for the bounded
`CWIGGLER` fixture used by the field-map/wiggler profiling suite. It keeps the
same external harmonic files and output shape as `cwiggler10`, but separates
stochastic wiggler validation from the deterministic fallback refresh.

The current CUDA implementation is expected to keep radiating `CWIGGLER`
tracking on the CPU path. Use this as the pure stochastic wiggler distribution
guard before enabling any CUDA wiggler radiation path.
