# phase34_rfcw_wakes_at_end

Focused Action 6 regression for wake-bearing `RFCW,N_KICKS=1` with
`WAKES_AT_END=1`.

This case mirrors `phase33_rfcw_kick_wake`, but moves the longitudinal and
transverse wake kicks after the RF exit half-step.  It exercises the guarded
CUDA ordering for autoscaled wake bins, interpolation, Savitzky-Golay
smoothing, deterministic `DX/DY` offsets, entrance/exit focusing, and
`CHANGE_P0=1`.

It intentionally avoids LSC, `CHANGE_T`, standing-wave/body focus models,
cavity `Q`, backtracking, distributed Pelegant particles, and multi-kick RF
sections.
