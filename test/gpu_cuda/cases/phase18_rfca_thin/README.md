# Phase 18 Thin RFCA

This case exercises the narrow Phase 18 zero-length `RFCA` CUDA path.  The first pass still uses the CPU path to establish fiducialization and phase-reference state for each RFCA occurrence; later passes should stay resident and apply the thin RF kick on the GPU.

The lattice intentionally avoids wakes, `CHANGE_P0`, `CHANGE_T`, nonzero cavity length, cavity `Q`, body/end focusing, offsets, linearized/locked phase mode, and backtracking.  Those RFCA/RFCW shapes remain CPU fallback.

Measured on May 7, 2026: CPU/GPU/GPU-VERIFY quick runs matched all 4 common SDDS files at tolerance `1e-11`.  The one-minute CPU baseline autoscaled to 30,000 particles and 224 passes, taking 60.16s; the same `gpu-elegant` workload took 11.94s, a 5.04x throughput speedup.  CUDA sync accounting showed 8 RFCA setup requests, one for each first-pass RFCA occurrence, plus deallocation.
