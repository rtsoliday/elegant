# RFMODE tracking-clock GPU regression

`run-clock.ele` combines a serial `RFCA,CHANGE_T=1` producer with the
supported single-bunch CUDA `RFMODE` histogram and kick path.  The 1 GHz RFCA
removes whole periods from the particle time coordinate while the non-harmonic
1.05 GHz resonator must retain the removed interval in its damping and phase.
The compact lattice avoids chaotic magnet amplification, so CPU/GPU output
differences test the tracking-clock handoff directly.
