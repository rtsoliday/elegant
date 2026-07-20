# Fixed-step FTABLE GPU target

`run.ele` tracks 262,144 particles through 32 one-step `FTABLE`
occurrences.  This isolates the initially supported deterministic subset while
exercising persistent reuse of the read-only three-dimensional field grid.

The CUDA eligibility guard requires no entrance/exit frame conversion,
misalignment, tilt, backtracking, or verbose particle diagnostics.  It also
requires `N_KICKS=1`; multi-kick maps remain on the CPU because the CPU GSL
cubic-root trajectory accumulates platform-specific transcendental differences
beyond the suite's roundoff envelope.

`run-small-fallback.ele` and `run-verbose-fallback.ele` are companion inputs for
manual eligibility checks.  They exercise the below-threshold and diagnostic
output fallbacks, respectively.
