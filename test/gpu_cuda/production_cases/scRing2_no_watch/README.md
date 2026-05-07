# scRing2 No-WATCH Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/scRing2`

This Phase 18 profiling variant uses the same linear `SCMULT` production case as `scRing2`, but disables all `WATCH` elements with `&alter_elements type=WATCH,item=DISABLE,value=1`.  It is intended to isolate how much GPU time is spent synchronizing device particles back to the CPU for diagnostic WATCH output.

Compare it against `scRing2` with the same pass count and `ELEGANT_GPU_ENABLE_SCMULT=1`.  The common physics outputs should continue to match CPU output at tolerance `1e-11`; the `.w1` diagnostic file is intentionally absent.

Measured on May 7, 2026 after the Phase 18 thin-RFCA path: the 601-pass CPU/GPU no-WATCH runs matched all 5 common SDDS files at tolerance `1e-11`.  CPU took 58.73s and GPU took 22.69s, a 2.59x speedup.  The normal CUDA sync accounting dropped to one CPU fiducialization setup request and one deallocation request, which shows the repeated RFCA and WATCH syncs are no longer the main limiter for this case.
