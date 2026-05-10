# phase51_rf_standing_wave_single

Focused Action 6 regression for the narrow `STANDING_WAVE=1` RFCA/RFCW CUDA
slice.

The lattice exercises zero-length thin RFCA, nonzero-length matrix-method RFCA,
single-kick RFCA, RF-only matrix-method RFCW, RF-only single-kick RFCW, and
guarded wake-bearing matrix/single-kick RFCW.  Multi-kick standing-wave
cavities intentionally remain CPU-owned because the CPU omits traveling-wave
section phase advance for that mode only when `N_KICKS>1`.
