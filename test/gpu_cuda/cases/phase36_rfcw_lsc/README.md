# phase36_rfcw_lsc

Focused RFCW regression that exercises the CUDA LSCKICK integration inside the
existing serial/local RFCW collective guards.  The lattice includes one
matrix-method RFCW, one kick-method `N_KICKS=1` RFCW, and one kick-method
`N_KICKS=1,WAKES_AT_END=1` RFCW.  All three combine longitudinal and
transverse wake columns with fixed-bin LSC filtering.
