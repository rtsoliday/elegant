# Phase 26 Thin RFCA CHANGE_P0

This case exercises the Action 6 extension of the zero-length thin `RFCA` CUDA path.  It uses GPU-side phase setup plus the existing GPU central-momentum matching helper for `RFCA,CHANGE_P0=1`.

The lattice intentionally avoids wakes, `CHANGE_T`, nonzero cavity length, cavity `Q`, body/end focusing, offsets, linearized/locked phase mode, and backtracking.  The companion `phase28_rfca_thin_offset` case covers deterministic `DX/DY` offsets for the same zero-length no-focus subset; broader RFCA/RFCW shapes remain CPU fallback.
