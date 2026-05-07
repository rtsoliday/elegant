# CLIC1 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/CLIC1`

This wrapper targets a CLIC RTML case with strong collective-effect coverage.  It keeps the source lattice external and scales the generated beam size with the usual `n_particles` macro.

Quick mode uses 2000 particles and one pass.  Baseline mode uses a larger particle count and keeps pass count fixed, which avoids accidentally turning the RTML case into a long-running job.
