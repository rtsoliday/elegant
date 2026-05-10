# phase53_rfca_standing_wave_multikick_fiducial

Focused Action 6 regression for non-explicit fiducial setup in multi-kick
`STANDING_WAVE=1` RFCA CUDA tracking.

The case covers RF-only RFCA kick-method cavities with `N_KICKS>1`, deterministic
offsets, entrance/exit focusing, `CHANGE_P0`, and serial/local `LIGHT`,
selected-bunch `TMEAN`, selected-bunch `PMAXIMUM`, and selected-bunch `FIRST`
fiducialization.  The analogous RFCW non-explicit multi-kick standing-wave
subset remains guarded until it gets its own regression.
