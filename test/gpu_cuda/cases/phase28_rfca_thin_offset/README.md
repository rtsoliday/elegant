# Phase 28 Thin RFCA Offsets

This case validates Action 6 support for deterministic `DX/DY` offsets on zero-length thin `RFCA` elements in the CUDA path.

For this restricted no-focus thin-RFCA subset, CPU tracking subtracts and restores the offset around an RF kick that does not depend on transverse position.  The case combines nonzero offsets with GPU phase setup and `CHANGE_P0=1` to prove the shape remains resident without RFCA CPU-element synchronization.
