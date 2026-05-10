# Phase 27 Thin RFCA Fiducial Modes

This case validates Action 6 GPU-side phase setup for zero-length thin `RFCA` elements using `FIDUCIAL="light"` and `FIDUCIAL="tmean"` instead of explicit `T_REFERENCE`.

The `tmean` cavity also uses `CHANGE_P0=1` so the case covers both GPU fiducial-time reduction and central-momentum matching in the thin-RFCA subset.
