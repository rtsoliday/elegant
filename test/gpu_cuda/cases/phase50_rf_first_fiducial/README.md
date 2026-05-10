# phase50_rf_first_fiducial

Focused Action 6 regression for serial/local selected-bunch
`FIDUCIAL="first"` phase setup across the supported RFCA, RF-only RFCW, and
guarded wake-bearing RFCW slices.

The lattice exercises zero-length thin RFCA, nonzero-length matrix-method RFCA,
kick-method RFCA, RF-only matrix-method RFCW, RF-only kick-method RFCW, and
guarded wake-bearing matrix/kick RFCW with longitudinal/transverse wakes,
fixed-bin filtered LSC kicks, deterministic offsets, focusing, `CHANGE_P0`, and
`WAKES_AT_END=1`.  The beam is duplicated into two bunches and
`fiducialization_bunch=1` selects the second bunch, so the `FIRST` fiducial
path must honor the particle-ID range filter instead of simply using row zero.
