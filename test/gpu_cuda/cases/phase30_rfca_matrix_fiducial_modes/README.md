# phase30_rfca_matrix_fiducial_modes

Focused regression for nonzero-length RF-only matrix-method `RFCA`
fiducialization on CUDA.

The lattice exercises `FIDUCIAL="light"` and serial/local full-beam
`FIDUCIAL="tmean"` phase setup with `N_KICKS=0`, entrance and exit focusing,
deterministic `DX`/`DY` offsets, and `CHANGE_P0=1` on the `TMEAN` cavity. It
intentionally avoids wakes, LSC, body-focus models, standing-wave mode,
`CHANGE_T`, linearization, locked phase, and backtracking.
