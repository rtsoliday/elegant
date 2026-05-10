# phase52_rf_standing_wave_multikick_treference

Focused Action 6 regression for explicit-reference multi-kick
`STANDING_WAVE=1` RFCA/RFCW CUDA tracking.

The case covers RF-only RFCA, RF-only RFCW, and guarded wake-bearing RFCW
kick-method cavities with `N_KICKS>1` and explicit `T_REFERENCE`.  Other
multi-kick standing-wave fiducial modes remain CPU-owned until they get a
separate offset/fiducial regression.
