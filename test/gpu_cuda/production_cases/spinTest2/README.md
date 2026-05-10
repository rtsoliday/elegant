# spinTest2 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/spinTest2`

This action-10 wrapper is a bounded spin-tracking gate based on the source
`spinTest2` storage-ring deck. It keeps the original lattice and spin-enabled
beam shape, but scales particle and pass counts through the shared benchmark
harness so fixed-seed CPU/GPU distribution checks stay quick.

The current CUDA implementation is expected to keep spin-tracking elements on
the CPU path. Use this wrapper as a fallback guard before enabling any
spin-aware CUDA tracking or spin-aware stochastic validation path.
