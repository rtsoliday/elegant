# uKickMap4 Radiation Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/uKickMap4`

This action-10 wrapper is a bounded stochastic radiation gate for the source
`uKickMap4` lattice. The source file defines both `UKICKMAP` and `CWIGGLER`
elements, but the selected `RING` line exercises the production `UKICKMAP`
sections plus radiating bend/quadrupole/sextupole sections. Particle and pass
counts are scaled through the shared benchmark harness.

The current CUDA implementation is expected to use CPU fallback for
radiating/ISR `UKICKMAP` and related magnet sections. Use this wrapper as the
stochastic field-map distribution guard before enabling any corresponding CUDA
path.
