# phase29_rfca_matrix_rf_only

Focused regression for the CUDA RF-only matrix-method `RFCA` subset.

The lattice uses nonzero-length cavities with `N_KICKS=0`, entrance and exit
focusing, deterministic `DX`/`DY` offsets, one explicit `T_REFERENCE`, one
`FIDUCIAL="light"` cavity, and one `CHANGE_P0=1` cavity. It intentionally avoids
wakes, LSC, body-focus models, standing-wave mode, `CHANGE_T`, linearization,
locked phase, and backtracking.
