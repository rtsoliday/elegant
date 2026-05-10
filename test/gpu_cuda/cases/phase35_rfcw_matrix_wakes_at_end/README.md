# phase35_rfcw_matrix_wakes_at_end

Focused Action 6 regression for wake-bearing matrix-method `RFCW` with
`WAKES_AT_END=1`.

This case mirrors `phase32_rfcw_matrix_wake`, but sets `WAKES_AT_END=1`.
For positive-length matrix-method RFCW, CPU tracking still applies the
longitudinal and transverse wake kicks after the RF matrix, so this case
validates the guarded CUDA eligibility and ordering for autoscaled wake bins,
interpolation, Savitzky-Golay smoothing, deterministic `DX/DY` offsets,
entrance/exit focusing, and `CHANGE_P0=1`.

It intentionally avoids LSC, kick-method RF, `CHANGE_T`, standing-wave/body
focus models, cavity `Q`, backtracking, and distributed Pelegant particles.
