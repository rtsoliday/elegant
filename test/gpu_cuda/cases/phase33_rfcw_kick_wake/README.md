# phase33_rfcw_kick_wake

Focused Action 6 regression for the first wake-bearing kick-method `RFCW`
CUDA subset.

This case mirrors `phase32_rfcw_matrix_wake`, but sets `N_KICKS=1` to exercise
the LCLS0-style kick ordering: RF entrance focusing and half-drift, the RF
energy kick, longitudinal and transverse wake kicks, final half-drift and exit
focusing, then `CHANGE_P0=1`.  It also keeps autoscaled bins, interpolation,
Savitzky-Golay smoothing, and deterministic `DX/DY` offsets.

It intentionally avoids LSC, `CHANGE_T`, standing-wave/body focus models,
cavity `Q`, backtracking, distributed Pelegant particles, and `WAKES_AT_END`.
