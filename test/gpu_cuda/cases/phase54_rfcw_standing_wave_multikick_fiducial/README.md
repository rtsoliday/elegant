# phase54_rfcw_standing_wave_multikick_fiducial

Focused Action 6 regression for non-explicit fiducial setup in multi-kick
`STANDING_WAVE=1` RFCW CUDA tracking.

The case covers RF-only and guarded wake-bearing RFCW kick-method cavities with
`N_KICKS>1`, deterministic offsets, entrance/exit focusing, `CHANGE_P0`,
`WAKES_AT_END=0|1`, fixed-bin filtered LSC kicks in the wake-bearing sections,
and serial/local `LIGHT`, selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`,
and selected-bunch `FIRST` fiducialization.
