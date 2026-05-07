# scRing2 Production Case

Source: `/home/soliday/oag/apps/src/elegantTestSet/scRing2`

This Phase 18 profile wrapper exercises the only production test-set `SCMULT` case that statically matches the current opt-in CUDA shape: linear, unsliced, and single-bucket-like.  It uses the source fixed SDDS beam, so benchmark scaling changes the pass count rather than the particle count.

Quick CPU/GPU/GPU-VERIFY runs matched the common SDDS outputs at tolerance `1e-11`.  The one-minute CPU baseline autoscaled to 601 passes: CPU `elegant` took 58.53s, while the current `gpu-elegant` with `ELEGANT_GPU_ENABLE_SCMULT=1` and the Phase 18 thin-RFCA path took 22.86s for the same pass count, a 2.56x throughput speedup.

The Phase 18 thin-RFCA path keeps zero-length RFCA kicks resident after the first CPU fiducialization setup pass.  The 601-pass CUDA log no longer has RFCA CPU-element synchronization; the remaining repeated syncs are the expected WATCH read-only output requests.  The companion `scRing2_no_watch` wrapper disables WATCH diagnostics and matched common CPU/GPU outputs at `1e-11`, but only ran at 22.69s, so this case is now dominated by tracking kernel time rather than RFCA/WATCH transfer overhead.
