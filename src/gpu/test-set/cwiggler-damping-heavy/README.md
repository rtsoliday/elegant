# Deterministic CWIGGLER damping target

`run.ele` benchmarks ideal sinusoidal horizontal, vertical, and helical
CWIGGLER tracking with classical synchrotron-radiation damping enabled and
ISR disabled.  It covers order-4 tracking for all three orientations and an
additional order-2 horizontal element.  It uses compact centroid and sigma
outputs rather than a full particle dump.

`run-isr-fallback.ele` keeps stochastic ISR on the CPU.  The separate
`run-small-fallback.ele` case verifies the 64-particle GPU threshold.
