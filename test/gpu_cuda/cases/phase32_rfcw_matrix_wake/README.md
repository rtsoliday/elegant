# phase32_rfcw_matrix_wake

Focused Action 6 regression for the first wake-bearing `RFCW` CUDA subset.

This case exercises a nonzero-length matrix-method `RFCW` with longitudinal and
transverse wake columns, autoscaled bins, interpolation, Savitzky-Golay
smoothing, entrance/exit focusing, deterministic `DX/DY` offsets, and
`CHANGE_P0=1`.  The wake table is shared with the older synthetic `rfcw` case,
but `N_KICKS=0` selects the matrix-method ordering used by the new resident
CUDA path.

It intentionally avoids LSC, kick-method RF, `CHANGE_T`, standing-wave/body
focus models, cavity `Q`, backtracking, distributed Pelegant particles, and
`WAKES_AT_END`.
