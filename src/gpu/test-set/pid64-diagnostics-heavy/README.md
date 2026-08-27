# 64-bit particle-ID GPU regression

This case duplicates a one-particle SDDS beam whose first `particleID` is
`2147483648`, then exercises three GPU-supported PID filters:

- `MALIGN` affects the lower half of the duplicated ID range.
- `RFDF` affects the upper half over 256 passes.
- `HISTOGRAM` selects the lower half for its final diagnostic page.

The coordinate watch and histogram make a CPU/GPU baseline comparison expose
any truncation of the 64-bit PID bounds or device-side particle IDs.
