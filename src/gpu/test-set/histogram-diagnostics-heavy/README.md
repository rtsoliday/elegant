# GPU histogram diagnostics target

This target repeatedly produces all-coordinate `HISTOGRAM` output for a
500,000-particle deterministic bunch.  It exercises dynamic bin ranges,
normalization, persistent CUDA scratch, and a fixed-bin-size particle-ID
selection.  The final parameter-mode `WATCH` is a compact aggregate-output
guard and is not included in the histogram timing loop.

`run-small-fallback.ele` verifies the below-threshold CPU fallback when
`ELEGANT_GPU_MIN_HISTOGRAM_PARTICLES=64`.

GPU histogram diagnostics are enabled by default and can be disabled with
`ELEGANT_GPU_ENABLE_HISTOGRAM_DIAGNOSTICS=0`.  The suite explicitly sets
`ELEGANT_GPU_MIN_OUTPUT_DRIFT_REDUCTION_PARTICLES=1` to exercise the resident
`WATCH` aggregate path.  Ordinary runs continue to use CPU output reductions
unless that threshold is explicitly set, since enabling them globally changed
strict diagnostic output in existing tests.
