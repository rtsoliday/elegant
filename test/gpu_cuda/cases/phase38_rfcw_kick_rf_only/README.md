# phase38_rfcw_kick_rf_only

Focused RFCW regression for the CUDA RF-only kick-method path when
`N_KICKS>0` and no wake or LSC columns are active.  The lattice exercises
`LIGHT` and serial/local `TMEAN` fiducialization, deterministic offsets,
entrance/exit focusing, `CHANGE_P0=1`, and a no-op `WAKES_AT_END=1` flag.
