# phase31_rfcw_rf_only_fiducial_modes

Focused regression for RF-only matrix-method `RFCW` fiducialization on CUDA.

The lattice exercises `FIDUCIAL="light"` and serial/local full-beam
`FIDUCIAL="tmean"` with `N_KICKS=0`, no active wake columns, no LSC, entrance
and exit focusing, deterministic `DX`/`DY` offsets, and `CHANGE_P0=1` on the
`TMEAN` cavity. It intentionally avoids wake files/columns, cavity `Q`,
standing-wave mode, body-focus models, linearization, backtracking, and
unsupported fiducial modes.
