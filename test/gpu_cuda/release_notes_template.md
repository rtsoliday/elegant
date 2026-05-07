# CUDA GPU Release Notes Template

## Summary

- Release candidate:
- Git revision:
- Worktree state:
- Release date:
- Prepared by:

## Build Matrix

| Variant | Command | Result | Notes |
| --- | --- | --- | --- |
| CPU `elegant` | `make -C src -j<N>` |  |  |
| CPU `Pelegant` | `make -C src -f Makefile.mpi -j<N>` |  |  |
| CUDA `gpu-elegant` | `make -C src HAVE_CUDA=1 NVCC=<path> CUDA_ARCH=<arch> -j<N>` |  |  |
| CUDA verify | `make -C src HAVE_CUDA=1 GPU_VERIFY=1 NVCC=<path> CUDA_ARCH=<arch> -j<N>` |  |  |
| CUDA `gpu-Pelegant` | `make -C src -f Makefile.mpi HAVE_CUDA=1 NVCC=<path> CUDA_ARCH=<arch> -j<N>` |  |  |

## Hardware And Runtime

| Item | Value |
| --- | --- |
| CPU |  |
| GPU |  |
| Driver |  |
| CUDA toolkit |  |
| CUDA runtime |  |
| CUDA_ARCH |  |
| MPI |  |
| Key environment variables |  |

## Correctness

| Case | Binary | Particles | Passes | MPI ranks | Tolerance | Result | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

## Timing

| Case | CPU seconds | GPU seconds | Speedup | Particles | Passes | Notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

CPU timing tests should target about one minute per case unless intentionally overridden.  Record any timeout changes.

## Known CPU Fallbacks

Link the generated fallback report, then summarize any notable synchronization changes:

- Fallback report:
- New or changed fallback reasons:
- Expected CPU-owned paths:

## GPU Verify Coverage

| Feature | Case | Intermediate arrays checked | Result |
| --- | --- | --- | --- |
|  |  |  |  |

## CPU Invariance

- Clean reference serial binary:
- Clean reference MPI binary:
- `release_invariance.sh` result:
- Dictionary metadata differences reviewed:
- CPU tracking differences reviewed:

## Deferred Work

- 

## Release Decision

- Decision:
- Residual risk:
- Follow-up owner/date:
