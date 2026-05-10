# Phase 18 Thin RFCA

This case exercises the narrow Phase 18 zero-length `RFCA` CUDA path.  Action 6 added GPU-side phase-reference setup for supported `T_REFERENCE`, `LIGHT`, and serial/local full-beam `TMEAN` fiducialization, so this case now stays resident from the first RFCA occurrence.

The lattice intentionally avoids wakes, `CHANGE_T`, nonzero cavity length, cavity `Q`, body/end focusing, offsets, linearized/locked phase mode, and backtracking.  Companion cases cover the same thin-RFCA subset with `CHANGE_P0=1`, `LIGHT`/`TMEAN` fiducialization, and deterministic `DX/DY` offsets; broader RFCA/RFCW shapes remain CPU fallback.

Measured on May 7, 2026: CPU/GPU/GPU-VERIFY quick runs matched all 4 common SDDS files at tolerance `1e-11`.  The one-minute CPU baseline autoscaled to 30,000 particles and 224 passes, taking 60.16s; the same `gpu-elegant` workload took 11.94s, a 5.04x throughput speedup.  After the May 8 Action 6 phase-setup cleanup, the focused thin-RFCA regression matched CPU output at `1e-11` and showed only final `gpuBaseDealloc` synchronization.
