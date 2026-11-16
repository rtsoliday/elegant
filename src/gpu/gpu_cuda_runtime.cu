#include "gpu_base.h"
#include "gpu_accurate_trig.h"
#include "gpu_faddeeva.h"
#include "../manual.h"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <limits.h>

#define GPU_MATRIX_CACHE_SLOTS 6
__constant__ GPU_MATRIX_DATA gpuMatrixData[GPU_MATRIX_CACHE_SLOTS];
__constant__ GPU_MULTIPOLE_DATA gpuMultipoleData;
/* Expansion coefficients from the CPU code, so that multipole-file kicks use identical values */
__device__ double gpuMultipoleCoefficientTable[GPU_MULTIPOLE_MAX_TERM_ORDER + 1][GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
__constant__ GPU_CSBEND_DATA gpuCsbendData;

#define GPU_REDUCTION_THREADS 64
#define GPU_BEAM_OUTPUT_THREADS 64
#define GPU_BEAM_SUM_BLOCKS 64
#define GPU_HISTOGRAM_THREADS 128
#define GPU_HISTOGRAM_BLOCKS 64
#define GPU_MATCH_ENERGY_BLOCKS 64
#define GPU_OPEN_PLUS_X 1
#define GPU_OPEN_PLUS_Y 2
#define GPU_OPEN_MINUS_X 3
#define GPU_OPEN_MINUS_Y 4

/* cudaFree(NULL) creates the CUDA context, which takes about 0.15 s, so releasing buffers a run never
 * allocated mustn't call it */
static inline cudaError_t gpuFreeDevice(void *pointer) {
  return pointer ? cudaFree(pointer) : cudaSuccess;
}

/*
 * Elegant removes a lost particle by swapping the last active row into its
 * slot, then immediately testing that replacement.  This is that loop itself,
 * on one device thread, from the exclusive survivor prefix.  It walks the whole
 * beam, about 80 ms for a million particles, so it's kept only as the check
 * (ELEGANT_GPU_CHECK_LOSS_ORDER=1) and the fallback
 * (ELEGANT_GPU_ENABLE_PARALLEL_LOSS_ORDER=0) of the kernel below.
 */
__global__ void gpuCpuLossOrderLoopKernel(
  const long *survivorPrefix, long *destination, long nParticles,
  long survivors) {
  long activeIndex = 0;
  long activeTop = nParticles - 1;
  long source = 0;
  long rightPrefix = survivors;
  int survives;

  if (blockIdx.x || threadIdx.x || nParticles <= 0)
    return;
  survives = ((nParticles == 1) ? survivors : survivorPrefix[1]) -
             survivorPrefix[0];
  while (activeIndex <= activeTop) {
    if (survives) {
      destination[source] = activeIndex++;
      if (activeIndex > activeTop)
        break;
      source = activeIndex;
      survives =
        ((activeIndex == activeTop) ? rightPrefix :
                                      survivorPrefix[activeIndex + 1]) -
        survivorPrefix[activeIndex];
    } else {
      long tailSource;
      long tailPrefix;

      destination[source] = activeTop;
      if (source == activeIndex && activeIndex == activeTop)
        break;
      tailSource = activeTop--;
      tailPrefix = survivorPrefix[tailSource];
      source = tailSource;
      survives = rightPrefix - tailPrefix;
      rightPrefix = tailPrefix;
    }
  }
}

/*
 * Where that loop leaves each particle, from the survivor prefix alone, so every particle is done
 * on its own.  With K survivors of N:
 *   a survivor below K stays where it is;
 *   a survivor at or above K fills a hole: the j-th one counted from the top goes to the j-th lost
 *     particle's place;
 *   a lost particle above K moves down one place;
 *   a lost particle at or below K, the r-th lost one counted from 0, goes to the top (r = 0) or to
 *     the place just under the r-th survivor counted from the top.
 * The two "r-th one" lookups are binary searches in the prefix.
 */
__global__ void gpuBuildCpuLossOrderDestinationKernel(
  const long *survivorPrefix, long *destination, long nParticles,
  long survivors) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long below, lo = 0, hi = nParticles - 1;

  if (i >= nParticles)
    return;
  below = survivorPrefix[i];
  if (((i == nParticles - 1) ? survivors : survivorPrefix[i + 1]) - below) {
    long fromTop;

    if (i < survivors) {
      destination[i] = i;
      return;
    }
    /* the largest q with fewer than fromTop lost particles below it is the fromTop-th lost one */
    fromTop = survivors - below;
    while (lo < hi) {
      long mid = (lo + hi + 1) / 2;

      if (mid - survivorPrefix[mid] <= fromTop - 1)
        lo = mid;
      else
        hi = mid - 1;
    }
    destination[i] = lo;
  } else if (i > survivors) {
    destination[i] = i - 1;
  } else {
    long rank = i - below;

    if (!rank) {
      destination[i] = nParticles - 1;
      return;
    }
    /* the largest q with at least rank survivors at or above it is the rank-th survivor from the top */
    while (lo < hi) {
      long mid = (lo + hi + 1) / 2;

      if (survivors - survivorPrefix[mid] >= rank)
        lo = mid;
      else
        hi = mid - 1;
    }
    destination[i] = lo - 1;
  }
}

__global__ void gpuCompareLongsKernel(const long *a, const long *b, long n, int *failed) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n && a[i] != b[i])
    *failed = 1;
}

static cudaError_t gpuRoundLossRounds(long nParticles, int **rounds);
static long *gpuRoundLossCheckBuffer(int **failed);

static int gpuLossOrderCheckRequested(void) {
  static int checkOrder = -1;

  if (checkOrder < 0) {
    const char *value = getenv("ELEGANT_GPU_CHECK_LOSS_ORDER");
    checkOrder = value && atoi(value) ? 1 : 0;
  }
  return checkOrder;
}

static cudaError_t gpuBuildCpuLossOrderDestination(long *survivorPrefix,
                                                    long nParticles,
                                                    long survivors) {
  static int parallel = -1;
  const int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  cudaError_t status;

  if (parallel < 0) {
    const char *value = getenv("ELEGANT_GPU_ENABLE_PARALLEL_LOSS_ORDER");
    parallel = value && !atoi(value) ? 0 : 1;
  }
  if (!parallel) {
    gpuCpuLossOrderLoopKernel<<<1, 1>>>(
      survivorPrefix, survivorPrefix + nParticles, nParticles, survivors);
    return cudaGetLastError();
  }
  gpuBuildCpuLossOrderDestinationKernel<<<blocks, threads>>>(
    survivorPrefix, survivorPrefix + nParticles, nParticles, survivors);
  if ((status = cudaGetLastError()) != cudaSuccess)
    return status;
  if (gpuLossOrderCheckRequested()) {
    int *rounds = NULL, *deviceFailed = NULL, failed = 0;
    long *loop;

    if ((status = gpuRoundLossRounds(nParticles, &rounds)) != cudaSuccess)
      return status;
    loop = gpuRoundLossCheckBuffer(&deviceFailed);
    if ((status = cudaMemset(deviceFailed, 0, sizeof(int))) != cudaSuccess)
      return status;
    gpuCpuLossOrderLoopKernel<<<1, 1>>>(survivorPrefix, loop, nParticles, survivors);
    gpuCompareLongsKernel<<<blocks, threads>>>(loop, survivorPrefix + nParticles, nParticles,
                                               deviceFailed);
    if ((status = cudaGetLastError()) != cudaSuccess ||
        (status = cudaMemcpy(&failed, deviceFailed, sizeof(int), cudaMemcpyDeviceToHost)) !=
          cudaSuccess)
      return status;
    if (failed) {
      fprintf(stderr, "elegant CUDA: the CPU's loss order didn't come out for %ld lost of %ld\n",
              nParticles - survivors, nParticles);
      return cudaErrorUnknown;
    }
  }
  return cudaSuccess;
}

static cudaError_t gpuPublishCpuLossOrderDestination(long *survivorPrefix,
                                                     long nParticles) {
  return cudaMemcpy(survivorPrefix, survivorPrefix + nParticles,
                    static_cast<size_t>(nParticles) * sizeof(*survivorPrefix),
                    cudaMemcpyDeviceToDevice);
}

/* Loops that remove losses in rounds (kick by kick, segment by segment, or entrance then exit) run
 * those swaps once per round over the particles still there.  lossRound[i] is the round particle i
 * is lost in, or -1 if it's kept, and lost[] lists the lost particles by round.  In each round the
 * particles kept below the round's final count don't move, so one thread visits only the round's
 * holes there, in order, and the places above, about twice the number lost in all.  cur[] and pos[]
 * start as the identity and end as each place's particle and each particle's place */
__device__ void gpuSortLongs(long *a, long n) {
  for (long start = n / 2 - 1, end = n - 1; end > 0;) {
    long root, child, t;

    if (start >= 0)
      root = start--;
    else {
      t = a[0];
      a[0] = a[end];
      a[end] = t;
      root = 0;
      end--;
    }
    while ((child = 2 * root + 1) <= end) {
      if (child < end && a[child] < a[child + 1])
        child++;
      if (a[root] >= a[child])
        break;
      t = a[root];
      a[root] = a[child];
      a[child] = t;
      root = child;
    }
  }
}

__global__ void gpuBuildCpuRoundLossOrderKernel(const int *lossRound, const long *lost, long nLost,
                                                long nParticles, long *cur, long *pos, long *holes,
                                                int *failed) {
  long top = nParticles - 1;

  if (blockIdx.x || threadIdx.x)
    return;
  for (long k = 0; k < nLost;) {
    int round = lossRound[lost[k]];
    long end = k, nKept, nHoles = 0;

    while (end < nLost && lossRound[lost[end]] == round)
      end++;
    nKept = top + 1 - (end - k);
    for (long j = k; j < end; j++)
      if (pos[lost[j]] < nKept)
        holes[nHoles++] = pos[lost[j]];
    gpuSortLongs(holes, nHoles);
    for (long j = 0; j < nHoles; j++) {
      long h = holes[j], x = cur[h];

      for (;;) {
        long t, y;

        if (top < nKept) {
          *failed = 1;
          return;
        }
        t = top--;
        y = cur[t];
        cur[t] = x;
        pos[x] = t;
        if (lossRound[y] < 0 || lossRound[y] > round) {
          cur[h] = y;
          pos[y] = h;
          break;
        }
        x = y;
      }
    }
    /* the rest are all lost this round, each swapped from nKept to the top */
    for (; top >= nKept; top--) {
      long a = cur[nKept], b = cur[top];

      cur[nKept] = b;
      pos[b] = nKept;
      cur[top] = a;
      pos[a] = top;
    }
    k = end;
  }
}

/* ELEGANT_GPU_CHECK_LOSS_ORDER=1: the CPU's loops themselves, round by round over every place,
 * to check the builder above */
__global__ void gpuCheckCpuRoundLossOrderKernel(const int *lossRound, long nParticles, int maxRound,
                                                long *order, const long *destination, int *failed) {
  long active = nParticles;

  if (blockIdx.x || threadIdx.x)
    return;
  for (long i = 0; i < nParticles; i++)
    order[i] = i;
  for (int round = 0; round <= maxRound; round++)
    for (long i = 0; i < active;) {
      if (lossRound[order[i]] == round) {
        long t = order[i];

        order[i] = order[--active];
        order[active] = t;
      } else
        i++;
    }
  for (long i = 0; i < nParticles; i++)
    if (destination[order[i]] != i)
      *failed = 1;
}

__global__ void gpuIdentityLongsKernel(long *a, long *b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    a[i] = i;
    if (b)
      b[i] = i;
  }
}

struct GpuLossRoundLost {
  __host__ __device__ bool operator()(int round) const {
    return round >= 0;
  }
};

static int *gpuRoundLossRound = NULL;
static int *gpuRoundLossKeys = NULL;
static int *gpuRoundLossFailed = NULL;
static long *gpuRoundLossLost = NULL;
static long *gpuRoundLossCur = NULL;
static long *gpuRoundLossPos = NULL;
static long *gpuRoundLossHoles = NULL;
static long gpuRoundLossCapacity = 0;

/* The round array a flag kernel fills for gpuBuildCpuRoundLossOrderDestination() */
static cudaError_t gpuRoundLossRounds(long nParticles, int **rounds) {
  cudaError_t status = cudaSuccess;

  if (gpuRoundLossCapacity < nParticles) {
    gpuFreeDevice(gpuRoundLossRound);
    gpuFreeDevice(gpuRoundLossKeys);
    gpuFreeDevice(gpuRoundLossFailed);
    gpuFreeDevice(gpuRoundLossLost);
    gpuFreeDevice(gpuRoundLossCur);
    gpuFreeDevice(gpuRoundLossPos);
    gpuFreeDevice(gpuRoundLossHoles);
    gpuRoundLossRound = gpuRoundLossKeys = gpuRoundLossFailed = NULL;
    gpuRoundLossLost = gpuRoundLossCur = gpuRoundLossPos = gpuRoundLossHoles = NULL;
    gpuRoundLossCapacity = 0;
    if ((status = cudaMalloc(&gpuRoundLossRound, nParticles * sizeof(int))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossKeys, nParticles * sizeof(int))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossFailed, sizeof(int))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossLost, nParticles * sizeof(long))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossCur, nParticles * sizeof(long))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossPos, nParticles * sizeof(long))) != cudaSuccess ||
        (status = cudaMalloc(&gpuRoundLossHoles, nParticles * sizeof(long))) != cudaSuccess)
      return status;
    gpuRoundLossCapacity = nParticles;
  }
  *rounds = gpuRoundLossRound;
  return cudaSuccess;
}

/* A beam-sized array and a flag for gpuBuildCpuLossOrderDestination()'s check, valid after
 * gpuRoundLossRounds() */
static long *gpuRoundLossCheckBuffer(int **failed) {
  *failed = gpuRoundLossFailed;
  return gpuRoundLossCur;
}

/* destination[i], the place the CPU's loops leave particle i in, from the rounds a flag kernel left
 * in gpuRoundLossRounds(); survivors is checked against them */
static cudaError_t gpuBuildCpuRoundLossOrderDestination(long *destination, long nParticles,
                                                        long survivors) {
  const int checkOrder = gpuLossOrderCheckRequested();
  const int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int failed = 0;
  long nLost;
  cudaError_t status;

  nLost = static_cast<long>(
    thrust::copy_if(thrust::device, thrust::counting_iterator<long>(0),
                    thrust::counting_iterator<long>(nParticles), gpuRoundLossRound,
                    gpuRoundLossLost, GpuLossRoundLost()) - gpuRoundLossLost);
  if (nLost != nParticles - survivors)
    return cudaErrorInvalidValue;
  if (!nLost) {
    gpuIdentityLongsKernel<<<blocks, threads>>>(destination, NULL, nParticles);
    return cudaGetLastError();
  }
  thrust::gather(thrust::device, gpuRoundLossLost, gpuRoundLossLost + nLost, gpuRoundLossRound,
                 gpuRoundLossKeys);
  thrust::stable_sort_by_key(thrust::device, gpuRoundLossKeys, gpuRoundLossKeys + nLost,
                             gpuRoundLossLost);
  gpuIdentityLongsKernel<<<blocks, threads>>>(gpuRoundLossCur, gpuRoundLossPos, nParticles);
  if ((status = cudaMemset(gpuRoundLossFailed, 0, sizeof(int))) != cudaSuccess)
    return status;
  gpuBuildCpuRoundLossOrderKernel<<<1, 1>>>(gpuRoundLossRound, gpuRoundLossLost, nLost, nParticles,
                                            gpuRoundLossCur, gpuRoundLossPos, gpuRoundLossHoles,
                                            gpuRoundLossFailed);
  if ((status = cudaMemcpy(destination, gpuRoundLossPos, nParticles * sizeof(long),
                           cudaMemcpyDeviceToDevice)) != cudaSuccess)
    return status;
  if (checkOrder) {
    int maxRound = 0;

    if ((status = cudaMemcpy(&maxRound, gpuRoundLossKeys + nLost - 1, sizeof(int),
                             cudaMemcpyDeviceToHost)) != cudaSuccess)
      return status;
    gpuCheckCpuRoundLossOrderKernel<<<1, 1>>>(gpuRoundLossRound, nParticles, maxRound,
                                              gpuRoundLossCur, destination, gpuRoundLossFailed);
  }
  if ((status = cudaMemcpy(&failed, gpuRoundLossFailed, sizeof(int), cudaMemcpyDeviceToHost)) !=
      cudaSuccess)
    return status;
  if (failed) {
    fprintf(stderr, "elegant CUDA: the CPU's loss order didn't come out for %ld lost of %ld\n",
            nLost, nParticles);
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

__global__ void gpuBeamStatisticsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  GPU_BEAM_SUM_DATA *partial, double *timeValue);

typedef struct GPU_GENERIC_REDUCTION_SCRATCH {
  GPU_BEAM_SUM_DATA *partial;
  GPU_BEAM_SUM_DATA *result;
} GPU_GENERIC_REDUCTION_SCRATCH;

static GPU_GENERIC_REDUCTION_SCRATCH gpuGenericReductionScratch;

static void releaseGenericReductionScratch(void) {
  gpuFreeDevice(gpuGenericReductionScratch.partial);
  gpuFreeDevice(gpuGenericReductionScratch.result);
  std::memset(&gpuGenericReductionScratch, 0,
              sizeof(gpuGenericReductionScratch));
}

/* Device buffers that reductions, loss counts, loss checks and wakes reuse from call to call.
 * cudaMalloc and cudaFree each wait for the device, which costs at every element, most of all
 * when jobs share the GPU.  A call's n-th buffer takes slot n; calls never overlap, since all
 * run on the default stream and none calls another. */
#define GPU_REUSED_DEVICE_BUFFERS 10

typedef struct GPU_REUSED_DEVICE_BUFFER {
  void *data;
  size_t bytes;
} GPU_REUSED_DEVICE_BUFFER;

static GPU_REUSED_DEVICE_BUFFER gpuReusedDeviceBuffer[GPU_REUSED_DEVICE_BUFFERS];

template <typename T>
static cudaError_t gpuReuseDeviceBuffer(int slot, T **data, size_t bytes) {
  GPU_REUSED_DEVICE_BUFFER *buffer = gpuReusedDeviceBuffer + slot;
  cudaError_t status;

  if (buffer->bytes < bytes) {
    gpuFreeDevice(buffer->data);
    buffer->data = NULL;
    buffer->bytes = 0;
    status = cudaMalloc(&buffer->data, bytes);
    if (status != cudaSuccess)
      return status;
    buffer->bytes = bytes;
  }
  *data = static_cast<T *>(buffer->data);
  return cudaSuccess;
}

static void releaseReusedDeviceBuffers(void) {
  int slot;

  for (slot = 0; slot < GPU_REUSED_DEVICE_BUFFERS; slot++)
    gpuFreeDevice(gpuReusedDeviceBuffer[slot].data);
  std::memset(gpuReusedDeviceBuffer, 0, sizeof(gpuReusedDeviceBuffer));
}

extern "C" void gpuCudaReductionRelease(void) {
  releaseGenericReductionScratch();
  releaseReusedDeviceBuffers();
}

static int ensureGenericReductionScratch(void) {
  cudaError_t status;

  if (gpuGenericReductionScratch.partial && gpuGenericReductionScratch.result)
    return static_cast<int>(cudaSuccess);
  releaseGenericReductionScratch();
  status = cudaMalloc(&gpuGenericReductionScratch.partial,
                      GPU_BEAM_SUM_BLOCKS *
                        sizeof(*gpuGenericReductionScratch.partial));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaMalloc(&gpuGenericReductionScratch.result,
                      sizeof(*gpuGenericReductionScratch.result));
  if (status != cudaSuccess) {
    releaseGenericReductionScratch();
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaSuccess);
}

typedef struct GPU_MATCH_ENERGY_PARTIAL {
  int64_t count;
  double sum;
  double error;
} GPU_MATCH_ENERGY_PARTIAL;

typedef struct GPU_COMBINED_WAKE_SCRATCH {
  double *time;
  double *pz;
  long *pbin;
  double *histogram;
  double *histogramPartial;
  double *voltage;
  double *table;
  double *fftReal;
  cufftDoubleComplex *driverFrequency;
  cufftDoubleComplex *tableFrequency;
  cufftDoubleComplex *channelFrequency;
  unsigned long long *binnedCount;
  int *driverFirstBin;
  long particleCapacity;
  long binCapacity;
  long tableCapacity;
  long histogramPartials;
  /* histogramPartial holds this many; a call uses histogramPartials of them, chosen from its own
   * particle count, so bunches of different sizes don't rebuild the scratch and its cuFFT plans */
  long histogramPartialCapacity;
  long planBins;
  cufftHandle forwardPlan;
  cufftHandle inversePlan;
  const double *hostTable[GPU_COMBINED_WAKE_CHANNELS];
  long hostTablePoints[GPU_COMBINED_WAKE_CHANNELS];
} GPU_COMBINED_WAKE_SCRATCH;

static GPU_COMBINED_WAKE_SCRATCH gpuCombinedWakeScratch;

typedef struct GPU_POLYNOMIAL_SERIES_SCRATCH {
  double *coefficient;
  int32_t *exponent;
  double *backup;
  unsigned long long *invalidCount;
  long termCapacity;
  long particleCapacity;
  long stride;
  const void *owner;
  long ownerTerms;
} GPU_POLYNOMIAL_SERIES_SCRATCH;

static GPU_POLYNOMIAL_SERIES_SCRATCH gpuPolynomialSeriesScratch;

typedef struct GPU_BGGEXP_SCRATCH {
  double *Cmn;
  double *dCmnDz;
  double *coefficient;
  double *multipoleFactor;
  double *potential;
  int *m;
  int *gradient;
  int *radialPower;
  long tableCapacity;
  long termCapacity;
  long tableNz;
  long tableTerms;
  const void *tableOwner;
  unsigned long long tableSignature;
} GPU_BGGEXP_SCRATCH;

static GPU_BGGEXP_SCRATCH gpuBggexpScratch;

typedef struct GPU_FTABLE_SCRATCH {
  double *field[3];
  long capacity;
  long dimensions[3];
  const void *tableOwner;
} GPU_FTABLE_SCRATCH;

static GPU_FTABLE_SCRATCH gpuFtableScratch;

typedef struct GPU_CWIGGLER_SCRATCH {
  GPU_CWIGGLER_HARMONIC *device;
  GPU_CWIGGLER_HARMONIC *uploaded;
  long capacity;
  long count;
} GPU_CWIGGLER_SCRATCH;

static GPU_CWIGGLER_SCRATCH gpuCwigglerScratch;

typedef struct GPU_BMXYZ_SCRATCH {
  double *coordBackup;
  long coordCapacity;
  unsigned long long *failedCount;
} GPU_BMXYZ_SCRATCH;

static GPU_BMXYZ_SCRATCH gpuBmxyzScratch;

typedef struct GPU_LORENTZ_SCRATCH {
  double *field[2];
  long fieldCapacity;
  long nx, ny;
  const void *tableOwner;
  double *coordBackup;
  long coordCapacity;
  unsigned long long *failedCount;
} GPU_LORENTZ_SCRATCH;

static GPU_LORENTZ_SCRATCH gpuLorentzScratch;

typedef struct GPU_RFMODE_SCRATCH {
  double *time;
  long *pbin;
  unsigned long long *histogram;
  double *xsum;
  double *ysum;
  double *voltage;
  unsigned long long *binnedCount;
  long particleCapacity;
  long binCapacity;
} GPU_RFMODE_SCRATCH;

static GPU_RFMODE_SCRATCH gpuRfmodeScratch;

/* Particle indices sorted by TRFMODE bin, stably, so each bin's particles stay in particle order */
typedef struct GPU_TRFMODE_SORT_SCRATCH {
  unsigned int *keysIn;
  unsigned int *keysOut;
  long *indexIn;
  long *indexOut;
  unsigned long long *binStart;
  void *temp;
  size_t tempBytes;
  long particleCapacity;
  long binCapacity;
} GPU_TRFMODE_SORT_SCRATCH;

static GPU_TRFMODE_SORT_SCRATCH gpuTrfmodeSortScratch;

typedef struct GPU_HISTOGRAM_SCRATCH {
  GPU_HISTOGRAM_RANGE_DATA *rangePartial;
  GPU_HISTOGRAM_RANGE_DATA *rangeResult;
  unsigned long long *histogramPartial;
  unsigned long long *histogram;
  long binCapacity;
} GPU_HISTOGRAM_SCRATCH;

typedef struct GPU_HISTOGRAM_BIN_DATA {
  long bins;
  int64_t startPID;
  int64_t endPID;
  unsigned int coordinateMask;
  double pCentral;
  double cMks;
  double timeOffset;
  double lower[7];
  double upper[7];
} GPU_HISTOGRAM_BIN_DATA;

static GPU_HISTOGRAM_SCRATCH gpuHistogramScratch;

static GPU_SCMULT_MOMENT_DATA *gpuScmultMomentResult;

static void releaseScmultScratch(void) {
  gpuFreeDevice(gpuScmultMomentResult);
  gpuScmultMomentResult = NULL;
}

extern "C" void gpuCudaScmultRelease(void) {
  releaseScmultScratch();
}

static int ensureScmultScratch(void) {
  cudaError_t status;

  if (gpuScmultMomentResult)
    return static_cast<int>(cudaSuccess);
  status = cudaMalloc(&gpuScmultMomentResult,
                      (GPU_BEAM_SUM_BLOCKS + 1) *
                        sizeof(*gpuScmultMomentResult));
  if (status != cudaSuccess) {
    releaseScmultScratch();
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaSuccess);
}

static void releaseHistogramScratch(void) {
  gpuFreeDevice(gpuHistogramScratch.rangePartial);
  gpuFreeDevice(gpuHistogramScratch.rangeResult);
  gpuFreeDevice(gpuHistogramScratch.histogramPartial);
  gpuFreeDevice(gpuHistogramScratch.histogram);
  std::memset(&gpuHistogramScratch, 0, sizeof(gpuHistogramScratch));
}

extern "C" void gpuCudaHistogramRelease(void) {
  releaseHistogramScratch();
}

static int ensureHistogramScratch(long bins) {
  cudaError_t status;

  if (bins < 2)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuHistogramScratch.rangePartial) {
    status = cudaMalloc(&gpuHistogramScratch.rangePartial,
                        GPU_HISTOGRAM_BLOCKS *
                          sizeof(*gpuHistogramScratch.rangePartial));
    if (status != cudaSuccess) {
      releaseHistogramScratch();
      return static_cast<int>(status);
    }
  }
  if (!gpuHistogramScratch.rangeResult) {
    status = cudaMalloc(&gpuHistogramScratch.rangeResult,
                        sizeof(*gpuHistogramScratch.rangeResult));
    if (status != cudaSuccess) {
      releaseHistogramScratch();
      return static_cast<int>(status);
    }
  }
  if (!gpuHistogramScratch.histogramPartial ||
      !gpuHistogramScratch.histogram ||
      gpuHistogramScratch.binCapacity < bins) {
    gpuFreeDevice(gpuHistogramScratch.histogramPartial);
    gpuFreeDevice(gpuHistogramScratch.histogram);
    gpuHistogramScratch.histogramPartial = NULL;
    gpuHistogramScratch.histogram = NULL;
    status = cudaMalloc(
      &gpuHistogramScratch.histogramPartial,
      7 * bins * GPU_HISTOGRAM_BLOCKS *
        sizeof(*gpuHistogramScratch.histogramPartial));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuHistogramScratch.histogram,
                          7 * bins *
                            sizeof(*gpuHistogramScratch.histogram));
    if (status != cudaSuccess) {
      releaseHistogramScratch();
      return static_cast<int>(status);
    }
    gpuHistogramScratch.binCapacity = bins;
  }
  return static_cast<int>(cudaSuccess);
}

static void releaseRfmodeScratch(void) {
  gpuFreeDevice(gpuRfmodeScratch.time);
  gpuFreeDevice(gpuRfmodeScratch.pbin);
  gpuFreeDevice(gpuRfmodeScratch.histogram);
  gpuFreeDevice(gpuRfmodeScratch.xsum);
  gpuFreeDevice(gpuRfmodeScratch.ysum);
  gpuFreeDevice(gpuRfmodeScratch.voltage);
  gpuFreeDevice(gpuRfmodeScratch.binnedCount);
  std::memset(&gpuRfmodeScratch, 0, sizeof(gpuRfmodeScratch));
}

static void releaseTrfmodeSortScratch(void) {
  gpuFreeDevice(gpuTrfmodeSortScratch.keysIn);
  gpuFreeDevice(gpuTrfmodeSortScratch.keysOut);
  gpuFreeDevice(gpuTrfmodeSortScratch.indexIn);
  gpuFreeDevice(gpuTrfmodeSortScratch.indexOut);
  gpuFreeDevice(gpuTrfmodeSortScratch.binStart);
  gpuFreeDevice(gpuTrfmodeSortScratch.temp);
  std::memset(&gpuTrfmodeSortScratch, 0, sizeof(gpuTrfmodeSortScratch));
}

extern "C" void gpuCudaRfmodeRelease(void) {
  releaseRfmodeScratch();
  releaseTrfmodeSortScratch();
}

/* CUB counts items in an int */
static int ensureTrfmodeSortScratch(long particles, long bins) {
  cudaError_t status = cudaSuccess;
  size_t sortBytes = 0, scanBytes = 0, bytes;

  if (particles <= 0 || particles > INT_MAX || bins < 2 || bins >= INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuTrfmodeSortScratch.keysIn || gpuTrfmodeSortScratch.particleCapacity < particles) {
    gpuFreeDevice(gpuTrfmodeSortScratch.keysIn);
    gpuFreeDevice(gpuTrfmodeSortScratch.keysOut);
    gpuFreeDevice(gpuTrfmodeSortScratch.indexIn);
    gpuFreeDevice(gpuTrfmodeSortScratch.indexOut);
    gpuTrfmodeSortScratch.keysIn = gpuTrfmodeSortScratch.keysOut = NULL;
    gpuTrfmodeSortScratch.indexIn = gpuTrfmodeSortScratch.indexOut = NULL;
    gpuTrfmodeSortScratch.particleCapacity = 0;
    status = cudaMalloc(&gpuTrfmodeSortScratch.keysIn, particles * sizeof(unsigned int));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuTrfmodeSortScratch.keysOut, particles * sizeof(unsigned int));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuTrfmodeSortScratch.indexIn, particles * sizeof(long));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuTrfmodeSortScratch.indexOut, particles * sizeof(long));
    if (status != cudaSuccess) {
      releaseTrfmodeSortScratch();
      return static_cast<int>(status);
    }
    gpuTrfmodeSortScratch.particleCapacity = particles;
  }
  if (!gpuTrfmodeSortScratch.binStart || gpuTrfmodeSortScratch.binCapacity < bins) {
    gpuFreeDevice(gpuTrfmodeSortScratch.binStart);
    gpuTrfmodeSortScratch.binStart = NULL;
    status = cudaMalloc(&gpuTrfmodeSortScratch.binStart, bins * sizeof(unsigned long long));
    if (status != cudaSuccess) {
      releaseTrfmodeSortScratch();
      return static_cast<int>(status);
    }
    gpuTrfmodeSortScratch.binCapacity = bins;
  }
  status = cub::DeviceRadixSort::SortPairs(
    NULL, sortBytes, gpuTrfmodeSortScratch.keysIn, gpuTrfmodeSortScratch.keysOut,
    gpuTrfmodeSortScratch.indexIn, gpuTrfmodeSortScratch.indexOut, static_cast<int>(particles));
  if (status == cudaSuccess)
    status = cub::DeviceScan::ExclusiveSum(
      NULL, scanBytes, gpuRfmodeScratch.histogram, gpuTrfmodeSortScratch.binStart,
      static_cast<int>(bins));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  bytes = sortBytes > scanBytes ? sortBytes : scanBytes;
  if (!gpuTrfmodeSortScratch.temp || gpuTrfmodeSortScratch.tempBytes < bytes) {
    gpuFreeDevice(gpuTrfmodeSortScratch.temp);
    gpuTrfmodeSortScratch.temp = NULL;
    gpuTrfmodeSortScratch.tempBytes = 0;
    status = cudaMalloc(&gpuTrfmodeSortScratch.temp, bytes);
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuTrfmodeSortScratch.tempBytes = bytes;
  }
  return static_cast<int>(cudaSuccess);
}

static int ensureRfmodeParticleScratch(long particles) {
  cudaError_t status;

  if (particles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuRfmodeScratch.time || !gpuRfmodeScratch.pbin ||
      gpuRfmodeScratch.particleCapacity < particles) {
    gpuFreeDevice(gpuRfmodeScratch.time);
    gpuFreeDevice(gpuRfmodeScratch.pbin);
    gpuRfmodeScratch.time = NULL;
    gpuRfmodeScratch.pbin = NULL;
    status = cudaMalloc(&gpuRfmodeScratch.time,
                        particles * sizeof(*gpuRfmodeScratch.time));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuRfmodeScratch.pbin,
                          particles * sizeof(*gpuRfmodeScratch.pbin));
    if (status != cudaSuccess) {
      releaseRfmodeScratch();
      return static_cast<int>(status);
    }
    gpuRfmodeScratch.particleCapacity = particles;
  }
  return static_cast<int>(cudaSuccess);
}

static int ensureRfmodeScratch(long particles, long bins) {
  cudaError_t status;
  int particleStatus;

  if (bins < 2)
    return static_cast<int>(cudaErrorInvalidValue);
  particleStatus = ensureRfmodeParticleScratch(particles);
  if (particleStatus != static_cast<int>(cudaSuccess))
    return particleStatus;
  if (!gpuRfmodeScratch.binnedCount) {
    status = cudaMalloc(&gpuRfmodeScratch.binnedCount,
                        sizeof(*gpuRfmodeScratch.binnedCount));
    if (status != cudaSuccess) {
      releaseRfmodeScratch();
      return static_cast<int>(status);
    }
  }
  if (!gpuRfmodeScratch.histogram || !gpuRfmodeScratch.xsum ||
      !gpuRfmodeScratch.ysum || !gpuRfmodeScratch.voltage ||
      gpuRfmodeScratch.binCapacity < bins) {
    gpuFreeDevice(gpuRfmodeScratch.histogram);
    gpuFreeDevice(gpuRfmodeScratch.xsum);
    gpuFreeDevice(gpuRfmodeScratch.ysum);
    gpuFreeDevice(gpuRfmodeScratch.voltage);
    gpuRfmodeScratch.histogram = NULL;
    gpuRfmodeScratch.xsum = NULL;
    gpuRfmodeScratch.ysum = NULL;
    gpuRfmodeScratch.voltage = NULL;
    status = cudaMalloc(&gpuRfmodeScratch.histogram,
                        bins * sizeof(*gpuRfmodeScratch.histogram));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuRfmodeScratch.xsum,
                          bins * sizeof(*gpuRfmodeScratch.xsum));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuRfmodeScratch.ysum,
                          bins * sizeof(*gpuRfmodeScratch.ysum));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuRfmodeScratch.voltage,
                          3 * bins * sizeof(*gpuRfmodeScratch.voltage));
    if (status != cudaSuccess) {
      releaseRfmodeScratch();
      return static_cast<int>(status);
    }
    gpuRfmodeScratch.binCapacity = bins;
  }
  return static_cast<int>(cudaSuccess);
}

static void releaseFtableScratch(void) {
  for (long field = 0; field < 3; field++)
    gpuFreeDevice(gpuFtableScratch.field[field]);
  std::memset(&gpuFtableScratch, 0, sizeof(gpuFtableScratch));
}

extern "C" void gpuCudaFtableRelease(void) {
  releaseFtableScratch();
}

static void releaseCwigglerScratch(void) {
  gpuFreeDevice(gpuCwigglerScratch.device);
  std::free(gpuCwigglerScratch.uploaded);
  std::memset(&gpuCwigglerScratch, 0, sizeof(gpuCwigglerScratch));
}

extern "C" void gpuCudaCwigglerRelease(void) {
  releaseCwigglerScratch();
}

static void releaseBmxyzScratch(void) {
  gpuFreeDevice(gpuBmxyzScratch.coordBackup);
  gpuFreeDevice(gpuBmxyzScratch.failedCount);
  std::memset(&gpuBmxyzScratch, 0, sizeof(gpuBmxyzScratch));
}

extern "C" void gpuCudaBmxyzRelease(void) {
  releaseBmxyzScratch();
}

static void releaseLorentzScratch(void) {
  gpuFreeDevice(gpuLorentzScratch.field[0]);
  gpuFreeDevice(gpuLorentzScratch.field[1]);
  gpuFreeDevice(gpuLorentzScratch.coordBackup);
  gpuFreeDevice(gpuLorentzScratch.failedCount);
  std::memset(&gpuLorentzScratch, 0, sizeof(gpuLorentzScratch));
}

extern "C" void gpuCudaLorentzRelease(void) {
  releaseLorentzScratch();
}

static int ensureLorentzScratch(long coordinateValues,
                                const GPU_LORENTZ_DATA *data) {
  cudaError_t status;

  if (coordinateValues <= 0 || !data)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuLorentzScratch.failedCount) {
    status = cudaMalloc(&gpuLorentzScratch.failedCount,
                        sizeof(*gpuLorentzScratch.failedCount));
    if (status != cudaSuccess) {
      releaseLorentzScratch();
      return static_cast<int>(status);
    }
  }
  if (!gpuLorentzScratch.coordBackup ||
      gpuLorentzScratch.coordCapacity < coordinateValues) {
    gpuFreeDevice(gpuLorentzScratch.coordBackup);
    gpuLorentzScratch.coordBackup = NULL;
    status = cudaMalloc(&gpuLorentzScratch.coordBackup,
                        coordinateValues * sizeof(*gpuLorentzScratch.coordBackup));
    if (status != cudaSuccess) {
      releaseLorentzScratch();
      return static_cast<int>(status);
    }
    gpuLorentzScratch.coordCapacity = coordinateValues;
  }
  if (data->type == GPU_LORENTZ_BMAPXY) {
    long fieldValues;
    int upload = 0;
    if (!data->tableOwner || !data->field[0] || !data->field[1] ||
        data->nx < 2 || data->ny < 2 || data->nx > LONG_MAX / data->ny)
      return static_cast<int>(cudaErrorInvalidValue);
    fieldValues = data->nx * data->ny;
    if (!gpuLorentzScratch.field[0] || !gpuLorentzScratch.field[1] ||
        gpuLorentzScratch.fieldCapacity < fieldValues) {
      gpuFreeDevice(gpuLorentzScratch.field[0]);
      gpuFreeDevice(gpuLorentzScratch.field[1]);
      gpuLorentzScratch.field[0] = gpuLorentzScratch.field[1] = NULL;
      status = cudaMalloc(&gpuLorentzScratch.field[0],
                          fieldValues * sizeof(*gpuLorentzScratch.field[0]));
      if (status == cudaSuccess)
        status = cudaMalloc(&gpuLorentzScratch.field[1],
                            fieldValues * sizeof(*gpuLorentzScratch.field[1]));
      if (status != cudaSuccess) {
        releaseLorentzScratch();
        return static_cast<int>(status);
      }
      gpuLorentzScratch.fieldCapacity = fieldValues;
      upload = 1;
    }
    if (gpuLorentzScratch.tableOwner != data->tableOwner ||
        gpuLorentzScratch.nx != data->nx || gpuLorentzScratch.ny != data->ny)
      upload = 1;
    if (upload) {
      for (long field = 0; field < 2; field++) {
        status = cudaMemcpy(gpuLorentzScratch.field[field], data->field[field],
                            fieldValues * sizeof(*data->field[field]),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
          return static_cast<int>(status);
      }
      gpuLorentzScratch.tableOwner = data->tableOwner;
      gpuLorentzScratch.nx = data->nx;
      gpuLorentzScratch.ny = data->ny;
    }
  }
  return static_cast<int>(cudaSuccess);
}

static int ensureBmxyzScratch(long coordinateValues) {
  cudaError_t status;

  if (coordinateValues <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuBmxyzScratch.failedCount) {
    status = cudaMalloc(&gpuBmxyzScratch.failedCount,
                        sizeof(*gpuBmxyzScratch.failedCount));
    if (status != cudaSuccess) {
      releaseBmxyzScratch();
      return static_cast<int>(status);
    }
  }
  if (!gpuBmxyzScratch.coordBackup ||
      gpuBmxyzScratch.coordCapacity < coordinateValues) {
    gpuFreeDevice(gpuBmxyzScratch.coordBackup);
    gpuBmxyzScratch.coordBackup = NULL;
    status = cudaMalloc(&gpuBmxyzScratch.coordBackup,
                        coordinateValues * sizeof(*gpuBmxyzScratch.coordBackup));
    if (status != cudaSuccess) {
      releaseBmxyzScratch();
      return static_cast<int>(status);
    }
    gpuBmxyzScratch.coordCapacity = coordinateValues;
  }
  return static_cast<int>(cudaSuccess);
}

static int ensureFtableScratch(const GPU_FTABLE_DATA *data) {
  cudaError_t status;
  long tableValues = 1;
  int upload = 0;

  if (!data || !data->tableOwner)
    return static_cast<int>(cudaErrorInvalidValue);
  for (long dimension = 0; dimension < 3; dimension++) {
    if (data->dimensions[dimension] < 2 ||
        tableValues > LONG_MAX / data->dimensions[dimension])
      return static_cast<int>(cudaErrorInvalidValue);
    tableValues *= data->dimensions[dimension];
  }
  for (long field = 0; field < 3; field++) {
    if (!data->field[field])
      return static_cast<int>(cudaErrorInvalidValue);
  }
  if (gpuFtableScratch.capacity < tableValues ||
      !gpuFtableScratch.field[0] || !gpuFtableScratch.field[1] ||
      !gpuFtableScratch.field[2]) {
    long capacity = tableValues;
    releaseFtableScratch();
    gpuFtableScratch.capacity = capacity;
    for (long field = 0; field < 3; field++) {
      status = cudaMalloc(&gpuFtableScratch.field[field],
                          capacity * sizeof(*gpuFtableScratch.field[field]));
      if (status != cudaSuccess) {
        releaseFtableScratch();
        return static_cast<int>(status);
      }
    }
    upload = 1;
  }
  if (gpuFtableScratch.tableOwner != data->tableOwner)
    upload = 1;
  for (long dimension = 0; dimension < 3; dimension++) {
    if (gpuFtableScratch.dimensions[dimension] !=
        data->dimensions[dimension])
      upload = 1;
  }
  if (upload) {
    for (long field = 0; field < 3; field++) {
      status = cudaMemcpy(gpuFtableScratch.field[field], data->field[field],
                          tableValues * sizeof(*data->field[field]),
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess)
        return static_cast<int>(status);
    }
    gpuFtableScratch.tableOwner = data->tableOwner;
    std::memcpy(gpuFtableScratch.dimensions, data->dimensions,
                sizeof(gpuFtableScratch.dimensions));
  }
  return static_cast<int>(cudaSuccess);
}

/* Elements often share one harmonic file, so compare contents rather than owners.
 * The wavenumbers depend on the period length, which can change between calls.
 */
static int ensureCwigglerScratch(const GPU_CWIGGLER_DATA *data) {
  cudaError_t status;
  long count = data->horizontalHarmonics + data->verticalHarmonics;
  size_t bytes;

  if (count <= 0 || !data->harmonics)
    return static_cast<int>(cudaErrorInvalidValue);
  bytes = count * sizeof(*data->harmonics);
  if (gpuCwigglerScratch.capacity < count) {
    releaseCwigglerScratch();
    status = cudaMalloc(&gpuCwigglerScratch.device, bytes);
    if (status != cudaSuccess) {
      releaseCwigglerScratch();
      return static_cast<int>(status);
    }
    gpuCwigglerScratch.uploaded =
      static_cast<GPU_CWIGGLER_HARMONIC *>(std::malloc(bytes));
    if (!gpuCwigglerScratch.uploaded) {
      releaseCwigglerScratch();
      return static_cast<int>(cudaErrorMemoryAllocation);
    }
    gpuCwigglerScratch.capacity = count;
  }
  if (gpuCwigglerScratch.count == count &&
      !std::memcmp(gpuCwigglerScratch.uploaded, data->harmonics, bytes))
    return static_cast<int>(cudaSuccess);
  status = cudaMemcpy(gpuCwigglerScratch.device, data->harmonics, bytes,
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  std::memcpy(gpuCwigglerScratch.uploaded, data->harmonics, bytes);
  gpuCwigglerScratch.count = count;
  return static_cast<int>(cudaSuccess);
}

static void releaseBggexpScratch(void) {
  gpuFreeDevice(gpuBggexpScratch.Cmn);
  gpuFreeDevice(gpuBggexpScratch.dCmnDz);
  gpuFreeDevice(gpuBggexpScratch.coefficient);
  gpuFreeDevice(gpuBggexpScratch.multipoleFactor);
  gpuFreeDevice(gpuBggexpScratch.potential);
  gpuFreeDevice(gpuBggexpScratch.m);
  gpuFreeDevice(gpuBggexpScratch.gradient);
  gpuFreeDevice(gpuBggexpScratch.radialPower);
  std::memset(&gpuBggexpScratch, 0, sizeof(gpuBggexpScratch));
}

extern "C" void gpuCudaBggexpRelease(void) {
  releaseBggexpScratch();
}

static int ensureBggexpScratch(const GPU_BGGEXP_DATA *data) {
  cudaError_t status;
  long tableValues;

  if (!data || !data->tableOwner || data->nz <= 1 ||
      data->termCount <= 0 || !data->m || !data->gradient ||
      !data->radialPower || !data->coefficient ||
      !data->multipoleFactor || !data->Cmn || !data->dCmnDz)
    return static_cast<int>(cudaErrorInvalidValue);
  tableValues = data->nz * data->termCount;
  if (tableValues <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  if (gpuBggexpScratch.tableCapacity < tableValues ||
      gpuBggexpScratch.termCapacity < data->termCount ||
      !gpuBggexpScratch.Cmn || !gpuBggexpScratch.dCmnDz ||
      !gpuBggexpScratch.coefficient || !gpuBggexpScratch.multipoleFactor ||
      !gpuBggexpScratch.potential ||
      !gpuBggexpScratch.m || !gpuBggexpScratch.gradient ||
      !gpuBggexpScratch.radialPower) {
    long tableCapacity = gpuBggexpScratch.tableCapacity;
    long termCapacity = gpuBggexpScratch.termCapacity;
    if (tableCapacity < tableValues)
      tableCapacity = tableValues;
    if (termCapacity < data->termCount)
      termCapacity = data->termCount;
    releaseBggexpScratch();
    gpuBggexpScratch.tableCapacity = tableCapacity;
    gpuBggexpScratch.termCapacity = termCapacity;
    status = cudaMalloc(&gpuBggexpScratch.Cmn,
                        tableCapacity * sizeof(*gpuBggexpScratch.Cmn));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.dCmnDz,
                        tableCapacity * sizeof(*gpuBggexpScratch.dCmnDz));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.coefficient,
                        termCapacity * sizeof(*gpuBggexpScratch.coefficient));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.multipoleFactor,
                        termCapacity * sizeof(*gpuBggexpScratch.multipoleFactor));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.potential,
                        3 * termCapacity * sizeof(*gpuBggexpScratch.potential));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.m,
                        termCapacity * sizeof(*gpuBggexpScratch.m));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.gradient,
                        termCapacity * sizeof(*gpuBggexpScratch.gradient));
    if (status != cudaSuccess)
      goto fail;
    status = cudaMalloc(&gpuBggexpScratch.radialPower,
                        termCapacity * sizeof(*gpuBggexpScratch.radialPower));
    if (status != cudaSuccess)
      goto fail;
  }

  status = cudaMemcpy(gpuBggexpScratch.m, data->m,
                      data->termCount * sizeof(*data->m),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaMemcpy(gpuBggexpScratch.gradient, data->gradient,
                      data->termCount * sizeof(*data->gradient),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaMemcpy(gpuBggexpScratch.radialPower, data->radialPower,
                      data->termCount * sizeof(*data->radialPower),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaMemcpy(gpuBggexpScratch.coefficient, data->coefficient,
                      data->termCount * sizeof(*data->coefficient),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaMemcpy(gpuBggexpScratch.multipoleFactor,
                      data->multipoleFactor,
                      data->termCount * sizeof(*data->multipoleFactor),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  if (data->symplectic) {
    if (!data->potential)
      return static_cast<int>(cudaErrorInvalidValue);
    status = cudaMemcpy(gpuBggexpScratch.potential, data->potential,
                        3 * data->termCount * sizeof(*data->potential),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }

  if (gpuBggexpScratch.tableOwner != data->tableOwner ||
      gpuBggexpScratch.tableSignature != data->tableSignature ||
      gpuBggexpScratch.tableNz != data->nz ||
      gpuBggexpScratch.tableTerms != data->termCount) {
    for (long term = 0; term < data->termCount; term++) {
      status = cudaMemcpy(gpuBggexpScratch.Cmn + term * data->nz,
                          data->Cmn[term],
                          data->nz * sizeof(*gpuBggexpScratch.Cmn),
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess)
        return static_cast<int>(status);
      status = cudaMemcpy(gpuBggexpScratch.dCmnDz + term * data->nz,
                          data->dCmnDz[term],
                          data->nz * sizeof(*gpuBggexpScratch.dCmnDz),
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess)
        return static_cast<int>(status);
    }
    gpuBggexpScratch.tableOwner = data->tableOwner;
    gpuBggexpScratch.tableSignature = data->tableSignature;
    gpuBggexpScratch.tableNz = data->nz;
    gpuBggexpScratch.tableTerms = data->termCount;
  }
  return static_cast<int>(cudaSuccess);

fail:
  releaseBggexpScratch();
  return static_cast<int>(status);
}

static void releasePolynomialSeriesScratch(void) {
  if (gpuPolynomialSeriesScratch.coefficient)
    gpuFreeDevice(gpuPolynomialSeriesScratch.coefficient);
  if (gpuPolynomialSeriesScratch.exponent)
    gpuFreeDevice(gpuPolynomialSeriesScratch.exponent);
  if (gpuPolynomialSeriesScratch.backup)
    gpuFreeDevice(gpuPolynomialSeriesScratch.backup);
  if (gpuPolynomialSeriesScratch.invalidCount)
    gpuFreeDevice(gpuPolynomialSeriesScratch.invalidCount);
  std::memset(&gpuPolynomialSeriesScratch, 0,
              sizeof(gpuPolynomialSeriesScratch));
}

extern "C" void gpuCudaPolynomialSeriesRelease(void) {
  releasePolynomialSeriesScratch();
}

static int ensurePolynomialSeriesScratch(long nParticles, int stride,
                                         long totalTerms) {
  cudaError_t status;
  long particleCapacity = gpuPolynomialSeriesScratch.particleCapacity;
  long termCapacity = gpuPolynomialSeriesScratch.termCapacity;

  if (nParticles <= 0 || stride < 6 || totalTerms <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (particleCapacity >= nParticles && termCapacity >= totalTerms &&
      gpuPolynomialSeriesScratch.stride == stride &&
      gpuPolynomialSeriesScratch.coefficient &&
      gpuPolynomialSeriesScratch.exponent &&
      gpuPolynomialSeriesScratch.backup &&
      gpuPolynomialSeriesScratch.invalidCount)
    return static_cast<int>(cudaSuccess);
  if (particleCapacity < nParticles)
    particleCapacity = nParticles;
  if (termCapacity < totalTerms)
    termCapacity = totalTerms;
  releasePolynomialSeriesScratch();
  gpuPolynomialSeriesScratch.particleCapacity = particleCapacity;
  gpuPolynomialSeriesScratch.termCapacity = termCapacity;
  gpuPolynomialSeriesScratch.stride = stride;
  status = cudaMalloc(
    &gpuPolynomialSeriesScratch.coefficient,
    termCapacity * sizeof(*gpuPolynomialSeriesScratch.coefficient));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(
    &gpuPolynomialSeriesScratch.exponent,
    6 * termCapacity * sizeof(*gpuPolynomialSeriesScratch.exponent));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(
    &gpuPolynomialSeriesScratch.backup,
    particleCapacity * stride * sizeof(*gpuPolynomialSeriesScratch.backup));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(
    &gpuPolynomialSeriesScratch.invalidCount,
    sizeof(*gpuPolynomialSeriesScratch.invalidCount));
  if (status != cudaSuccess)
    goto fail;
  return static_cast<int>(cudaSuccess);

fail:
  releasePolynomialSeriesScratch();
  return static_cast<int>(status);
}

static void releaseCombinedWakeScratch(void) {
  gpuFreeDevice(gpuCombinedWakeScratch.time);
  gpuFreeDevice(gpuCombinedWakeScratch.pz);
  gpuFreeDevice(gpuCombinedWakeScratch.pbin);
  gpuFreeDevice(gpuCombinedWakeScratch.histogram);
  gpuFreeDevice(gpuCombinedWakeScratch.histogramPartial);
  gpuFreeDevice(gpuCombinedWakeScratch.voltage);
  gpuFreeDevice(gpuCombinedWakeScratch.table);
  gpuFreeDevice(gpuCombinedWakeScratch.fftReal);
  gpuFreeDevice(gpuCombinedWakeScratch.driverFrequency);
  gpuFreeDevice(gpuCombinedWakeScratch.tableFrequency);
  gpuFreeDevice(gpuCombinedWakeScratch.channelFrequency);
  gpuFreeDevice(gpuCombinedWakeScratch.binnedCount);
  gpuFreeDevice(gpuCombinedWakeScratch.driverFirstBin);
  if (gpuCombinedWakeScratch.forwardPlan)
    cufftDestroy(gpuCombinedWakeScratch.forwardPlan);
  if (gpuCombinedWakeScratch.inversePlan)
    cufftDestroy(gpuCombinedWakeScratch.inversePlan);
  std::memset(&gpuCombinedWakeScratch, 0, sizeof(gpuCombinedWakeScratch));
}

extern "C" void gpuCudaCombinedWakeRelease(void) {
  releaseCombinedWakeScratch();
}

static int ensureCombinedWakeScratch(long nParticles, long bins,
                                     long tablePoints, long planBins) {
  cudaError_t status;
  cufftResult fftStatus;
  long particleCapacity = gpuCombinedWakeScratch.particleCapacity;
  long binCapacity = gpuCombinedWakeScratch.binCapacity;
  long tableCapacity = gpuCombinedWakeScratch.tableCapacity;
  long frequencyPoints, histogramCapacityBins;
  long desiredHistogramPartials, histogramPartials, maxHistogramPartials;

  if (nParticles <= 0 || bins < 2 || tablePoints <= 0 || planBins < bins)
    return static_cast<int>(cudaErrorInvalidValue);
  histogramCapacityBins = binCapacity > bins ? binCapacity : bins;
  maxHistogramPartials =
    (64L * 1024 * 1024) /
    (3 * histogramCapacityBins * static_cast<long>(sizeof(double)));
  if (maxHistogramPartials < 1)
    maxHistogramPartials = 1;
  if (maxHistogramPartials > 1024)
    maxHistogramPartials = 1024;
  if (maxHistogramPartials >= 256)
    maxHistogramPartials = maxHistogramPartials / 256 * 256;
  desiredHistogramPartials = (nParticles + 255) / 256;
  if (desiredHistogramPartials >= 256)
    desiredHistogramPartials =
      (desiredHistogramPartials + 255) / 256 * 256;
  if (desiredHistogramPartials < 1)
    desiredHistogramPartials = 1;
  histogramPartials = desiredHistogramPartials < maxHistogramPartials ?
                      desiredHistogramPartials : maxHistogramPartials;
  if (particleCapacity >= nParticles && binCapacity >= bins &&
      tableCapacity >= tablePoints &&
      gpuCombinedWakeScratch.planBins == planBins &&
      gpuCombinedWakeScratch.histogramPartialCapacity >= histogramPartials) {
    gpuCombinedWakeScratch.histogramPartials = histogramPartials;
    return static_cast<int>(cudaSuccess);
  }

  if (particleCapacity < nParticles)
    particleCapacity = nParticles;
  if (binCapacity < bins)
    binCapacity = bins;
  if (tableCapacity < tablePoints)
    tableCapacity = tablePoints;
  releaseCombinedWakeScratch();
  gpuCombinedWakeScratch.particleCapacity = particleCapacity;
  gpuCombinedWakeScratch.binCapacity = binCapacity;
  gpuCombinedWakeScratch.tableCapacity = tableCapacity;
  gpuCombinedWakeScratch.histogramPartials = histogramPartials;
  gpuCombinedWakeScratch.histogramPartialCapacity = histogramPartials;
  gpuCombinedWakeScratch.planBins = planBins;
  frequencyPoints = planBins / 2 + 1;

  status = cudaMalloc(&gpuCombinedWakeScratch.time,
                      particleCapacity * sizeof(*gpuCombinedWakeScratch.time));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.pz,
                      particleCapacity * sizeof(*gpuCombinedWakeScratch.pz));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.pbin,
                      particleCapacity * sizeof(*gpuCombinedWakeScratch.pbin));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.histogram,
                      3 * binCapacity * sizeof(*gpuCombinedWakeScratch.histogram));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.histogramPartial,
                      3 * binCapacity * histogramPartials *
                      sizeof(*gpuCombinedWakeScratch.histogramPartial));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.voltage,
                      GPU_COMBINED_WAKE_CHANNELS * binCapacity *
                      sizeof(*gpuCombinedWakeScratch.voltage));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.table,
                      GPU_COMBINED_WAKE_CHANNELS * tableCapacity *
                      sizeof(*gpuCombinedWakeScratch.table));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.fftReal,
                      planBins * sizeof(*gpuCombinedWakeScratch.fftReal));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.driverFrequency,
                      3 * frequencyPoints *
                      sizeof(*gpuCombinedWakeScratch.driverFrequency));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.tableFrequency,
                      GPU_COMBINED_WAKE_CHANNELS * frequencyPoints *
                      sizeof(*gpuCombinedWakeScratch.tableFrequency));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.channelFrequency,
                      GPU_COMBINED_WAKE_CHANNELS * frequencyPoints *
                      sizeof(*gpuCombinedWakeScratch.channelFrequency));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.binnedCount,
                      sizeof(*gpuCombinedWakeScratch.binnedCount));
  if (status != cudaSuccess)
    goto fail;
  status = cudaMalloc(&gpuCombinedWakeScratch.driverFirstBin,
                      3 * sizeof(*gpuCombinedWakeScratch.driverFirstBin));
  if (status != cudaSuccess)
    goto fail;

  fftStatus = cufftPlan1d(&gpuCombinedWakeScratch.forwardPlan,
                         static_cast<int>(planBins), CUFFT_D2Z, 1);
  if (fftStatus != CUFFT_SUCCESS) {
    status = cudaErrorUnknown;
    goto fail;
  }
  fftStatus = cufftPlan1d(&gpuCombinedWakeScratch.inversePlan,
                         static_cast<int>(planBins), CUFFT_Z2D, 1);
  if (fftStatus != CUFFT_SUCCESS) {
    status = cudaErrorUnknown;
    goto fail;
  }
  return static_cast<int>(cudaSuccess);

fail:
  releaseCombinedWakeScratch();
  return static_cast<int>(status);
}

static int uploadCsbendDataIfNeeded(const GPU_CSBEND_DATA *csbend) {
  static GPU_CSBEND_DATA cachedCsbendData;
  static int cachedCsbendDataValid = 0;
  cudaError_t status;

  if (!csbend)
    return static_cast<int>(cudaErrorInvalidValue);
  if (cachedCsbendDataValid &&
      std::memcmp(&cachedCsbendData, csbend, sizeof(*csbend)) == 0)
    return static_cast<int>(cudaSuccess);
  status = cudaMemcpyToSymbol(gpuCsbendData, csbend, sizeof(*csbend));
  if (status != cudaSuccess) {
    cachedCsbendDataValid = 0;
    return static_cast<int>(status);
  }
  std::memcpy(&cachedCsbendData, csbend, sizeof(cachedCsbendData));
  cachedCsbendDataValid = 1;
  return static_cast<int>(cudaSuccess);
}

static int uploadMatrixDataIfNeeded(const GPU_MATRIX_DATA *matrix,
                                    int *matrixSlot) {
  static GPU_MATRIX_DATA cached[GPU_MATRIX_CACHE_SLOTS];
  static int valid[GPU_MATRIX_CACHE_SLOTS] = {0};
  static int replacementSlot = 0;
  cudaError_t status;
  int slot;

  if (!matrix || !matrixSlot)
    return static_cast<int>(cudaErrorInvalidValue);
  for (slot = 0; slot < GPU_MATRIX_CACHE_SLOTS; slot++) {
    if (valid[slot] &&
        std::memcmp(cached + slot, matrix, sizeof(*matrix)) == 0) {
      *matrixSlot = slot;
      return static_cast<int>(cudaSuccess);
    }
  }
  for (slot = 0; slot < GPU_MATRIX_CACHE_SLOTS; slot++)
    if (!valid[slot])
      break;
  if (slot == GPU_MATRIX_CACHE_SLOTS) {
    slot = replacementSlot;
    replacementSlot = (replacementSlot + 1) % GPU_MATRIX_CACHE_SLOTS;
  }
  status = cudaMemcpyToSymbol(gpuMatrixData, matrix, sizeof(*matrix),
                              (unsigned long)slot * sizeof(*matrix),
                              cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    valid[slot] = 0;
    return static_cast<int>(status);
  }
  cached[slot] = *matrix;
  valid[slot] = 1;
  *matrixSlot = slot;
  return static_cast<int>(cudaSuccess);
}

static int timeCopy(void *dst, const void *src, unsigned long bytes,
                    cudaMemcpyKind kind, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t status;

  if (milliseconds)
    *milliseconds = 0;
  status = cudaEventCreate(&start);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(status);
  }
  cudaEventRecord(start, 0);
  status = cudaMemcpy(dst, src, bytes, kind);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  if (milliseconds && status == cudaSuccess)
    cudaEventElapsedTime(milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<int>(status);
}

static int getCachedTimingEvents(cudaEvent_t *start, cudaEvent_t *stop) {
  cudaError_t status;

  if (!start || !stop)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!*start) {
    status = cudaEventCreate(start);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  if (!*stop) {
    status = cudaEventCreate(stop);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  return static_cast<int>(cudaSuccess);
}

static int gpuDetailedKernelTimingEnabled(void) {
  static int initialized = 0, enabled = 0;
  const char *value;

  if (initialized)
    return enabled;
  initialized = 1;
  value = std::getenv("ELEGANT_GPU_PROFILE_TIMING");
  if (value && (std::strcmp(value, "0") && std::strcmp(value, "false") &&
                std::strcmp(value, "FALSE") && std::strcmp(value, "off") &&
                std::strcmp(value, "OFF")))
    enabled = 1;
  return enabled;
}

static int finishTimedKernel(cudaEvent_t start, cudaEvent_t stop,
                             float *milliseconds) {
  cudaError_t status;

  status = cudaEventRecord(stop, 0);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaEventSynchronize(stop);
  if (milliseconds && status == cudaSuccess)
    cudaEventElapsedTime(milliseconds, start, stop);
  return static_cast<int>(status);
}

__device__ __forceinline__ int gpuTOffset(int i, int j, int k) {
  return i * 21 + j * (j + 1) / 2 + k;
}

__device__ __forceinline__ int gpuQOffset(int i, int j, int k, int l) {
  return i * 56 + j * (j + 1) * (j + 2) / 6 + k * (k + 1) / 2 + l;
}

__device__ __forceinline__ void gpuApplyPackedMatrixData(
  double *part, const GPU_MATRIX_DATA *matrix) {
  double ini[6], temp[6];
  int i, j, k;

  for (i = 0; i < 6; i++)
    ini[i] = part[i];
  if (matrix->useSReference)
    ini[4] -= matrix->sReference;

  if (matrix->order == 3 || matrix->order == 1) {
    for (i = 5; i >= 0; i--) {
      double sum = matrix->C[i];
      for (j = matrix->termOffset[i]; j < matrix->termOffset[i + 1]; j++) {
        const GPU_MATRIX_TERM *term = matrix->term + j;
        /* Match track_particles: form coord_jk before multiplying by T/Q.
         * Reassociation here changes sensitive derived beam statistics. */
        if (matrix->order == 3 && ini[term->j] == 0)
          continue;
        double value;
        if (term->degree >= 2) {
          if (ini[term->k] == 0)
            continue;
          double coord_jk = ini[term->j] * ini[term->k];
          value = term->coefficient * coord_jk;
        } else
          value = term->coefficient * ini[term->j];
        if (term->degree >= 3)
          value *= ini[term->l];
        sum += value;
      }
      temp[i] = sum;
    }
  } else if (matrix->order == 2) {
    for (i = 5; i >= 0; i--) {
      double sum = matrix->C[i];
      for (j = 5; j >= 0; j--) {
        double sum1 = matrix->R[i * 6 + j];
        int termIndex = i * 6 + j;
        for (k = matrix->secondOrderOffset[termIndex];
             k < matrix->secondOrderOffset[termIndex + 1]; k++)
          sum1 += matrix->term[k].coefficient * ini[matrix->term[k].k];
        sum += sum1 * ini[j];
      }
      temp[i] = sum;
    }
  }

  if (matrix->useSReference)
    temp[4] += matrix->sReference;

  for (i = 5; i >= 0; i--)
    part[i] = temp[i];
}

__device__ __forceinline__ void gpuApplyPackedMatrix(double *part,
                                                     int matrixSlot) {
  gpuApplyPackedMatrixData(part, gpuMatrixData + matrixSlot);
}

__global__ void gpuTrackParticlesKernel(double *coord, long nParticles,
                                        int stride, int matrixSlot) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  gpuApplyPackedMatrix(coord + ip * stride, matrixSlot);
}

__device__ __forceinline__ double gpuAddRn(double a, double b) {
  return __dadd_rn(a, b);
}

__device__ __forceinline__ double gpuSubRn(double a, double b) {
  return __dadd_rn(a, -b);
}

__device__ __forceinline__ double gpuMulRn(double a, double b) {
  return __dmul_rn(a, b);
}

__device__ __forceinline__ double gpuDivRn(double a, double b) {
  return __ddiv_rn(a, b);
}

__device__ __forceinline__ double gpuExactPathLengthFactor(double xp,
                                                          double yp) {
  return sqrt(gpuAddRn(gpuAddRn(1.0, gpuMulRn(xp, xp)),
                       gpuMulRn(yp, yp)));
}

__device__ __forceinline__ void gpuExactDriftParticle(double *part,
                                                      double length) {
  double xp = part[1];
  double yp = part[3];

  part[0] = gpuAddRn(part[0], gpuMulRn(xp, length));
  part[2] = gpuAddRn(part[2], gpuMulRn(yp, length));
  part[4] = gpuAddRn(part[4],
                     gpuMulRn(length, gpuExactPathLengthFactor(xp, yp)));
}

__global__ void gpuExactDriftKernel(double *coord, long nParticles, int stride, double length) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  gpuExactDriftParticle(coord + ip * stride, length);
}

__global__ void gpuLinearDriftKernel(double *coord, long nParticles, int stride, double length) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  part[0] += part[1] * length;
  part[2] += part[3] * length;
  part[4] += length;
}

__device__ __forceinline__ int gpuKickMapInterpolate(double *xpFactor,
                                                     double *ypFactor,
                                                     const double *xpMap,
                                                     const double *ypMap,
                                                     const GPU_KICKMAP_DATA *map,
                                                     double x, double y) {
  double fx, fy, fa, fb;
  long ix, iy;

  if (!isfinite(x) || !isfinite(y))
    return 0;
  ix = static_cast<long>(gpuDivRn(gpuSubRn(x, map->xmin), map->dxg));
  iy = static_cast<long>(gpuDivRn(gpuSubRn(y, map->ymin), map->dyg));
  if (ix < 0 || iy < 0 || ix > map->nx - 1 || iy > map->ny - 1)
    return 0;
  if (ix == map->nx - 1)
    ix--;
  if (iy == map->ny - 1)
    iy--;

  fx = gpuDivRn(gpuSubRn(x, gpuAddRn(gpuMulRn((double)ix, map->dxg),
                                      map->xmin)),
                map->dxg);
  fy = gpuDivRn(gpuSubRn(y, gpuAddRn(gpuMulRn((double)iy, map->dyg),
                                      map->ymin)),
                map->dyg);

  fa = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fy), xpMap[ix + iy * map->nx]),
                gpuMulRn(fy, xpMap[ix + (iy + 1) * map->nx]));
  fb = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fy), xpMap[ix + 1 + iy * map->nx]),
                gpuMulRn(fy, xpMap[ix + 1 + (iy + 1) * map->nx]));
  *xpFactor = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fx), fa),
                       gpuMulRn(fx, fb));

  fa = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fy), ypMap[ix + iy * map->nx]),
                gpuMulRn(fy, ypMap[ix + (iy + 1) * map->nx]));
  fb = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fy), ypMap[ix + 1 + iy * map->nx]),
                gpuMulRn(fy, ypMap[ix + 1 + (iy + 1) * map->nx]));
  *ypFactor = gpuAddRn(gpuMulRn(gpuSubRn(1.0, fx), fa),
                       gpuMulRn(fx, fb));
  return 1;
}

__device__ __forceinline__ int gpuKickMapTrackParticle(
  double *part, const GPU_KICKMAP_DATA *map, const double *xpMap,
  const double *ypMap, int writeOutput, int *lostKick = NULL) {
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double s = part[4];
  double dp = part[5];
  double dxp, dyp, kickXp, kickYp;

  for (long ik = 0; ik < map->nKicks; ik++) {
    x = gpuAddRn(x, gpuDivRn(gpuMulRn(xp, map->length), 2.0));
    y = gpuAddRn(y, gpuDivRn(gpuMulRn(yp, map->length), 2.0));
    s = gpuAddRn(s, gpuMulRn(map->halfLength,
                             gpuExactPathLengthFactor(xp, yp)));

    if (!gpuKickMapInterpolate(&dxp, &dyp, xpMap, ypMap, map, x, y)) {
      if (lostKick)
        *lostKick = (int)ik;
      return 0;
    }
    if (map->undulator) {
      double dp1 = gpuAddRn(1.0, dp);
      double scale = gpuDivRn(map->kickScale, gpuMulRn(dp1, dp1));
      kickXp = gpuMulRn(dxp, scale);
      kickYp = gpuMulRn(dyp, scale);
      xp = gpuAddRn(xp, kickXp);
      yp = gpuAddRn(yp, kickYp);
    } else {
      double scale = gpuDivRn(map->kickScale, gpuAddRn(1.0, dp));
      kickXp = gpuMulRn(dxp, scale);
      kickYp = gpuMulRn(dyp, scale);
      xp = gpuAddRn(xp, kickXp);
      yp = gpuAddRn(yp, kickYp);
    }

    if (map->undulator) {
      if (map->radiationKick) {
        double deltaFactor = gpuAddRn(1.0, dp);
        dp = gpuSubRn(
          dp, gpuMulRn(map->radiationKick,
                       gpuMulRn(deltaFactor, deltaFactor)));
      }
    } else if (map->radCoef && map->length) {
      double oldDp = dp;
      double deltaFactor = 1 + oldDp;
      double kick2 = kickXp * kickXp + kickYp * kickYp;
      double F2 =
        deltaFactor * deltaFactor * kick2 / (map->length * map->length);
      double p = map->pRef * deltaFactor;
      double beta0 = p / sqrt(p * p + 1);
      double pathFactor = sqrt(1 + xp * xp + yp * yp);

      dp -= map->radCoef * deltaFactor * deltaFactor * F2 *
            map->length * pathFactor;
      p = map->pRef * (1 + dp);
      xp *= deltaFactor / (1 + dp);
      yp *= deltaFactor / (1 + dp);
      {
        double beta1 = p / sqrt(p * p + 1);
        s = beta1 * s / beta0;
      }
    }

    x = gpuAddRn(x, gpuDivRn(gpuMulRn(xp, map->length), 2.0));
    y = gpuAddRn(y, gpuDivRn(gpuMulRn(yp, map->length), 2.0));
    s = gpuAddRn(s, gpuMulRn(map->halfLength,
                             gpuExactPathLengthFactor(xp, yp)));
  }

  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    part[4] = s;
    part[5] = dp;
  }
  return 1;
}

__global__ void gpuKickMapTrackCheckedKernel(double *coord, long nParticles,
                                             int stride,
                                             GPU_KICKMAP_DATA map,
                                             const double *xpMap,
                                             const double *ypMap,
                                             unsigned long long *lostCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (!gpuKickMapTrackParticle(part, &map, xpMap, ypMap, 1))
    atomicAdd(lostCount, 1ULL);
}

__global__ void gpuKickMapSurvivorFlagKernel(double *coord, long nParticles,
                                             int stride,
                                             GPU_KICKMAP_DATA map,
                                             const double *xpMap,
                                             const double *ypMap,
                                             long *survivorPrefix,
                                             int *lossRound) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int kick = -1;

  if (ip >= nParticles)
    return;
  /* trackKickMap() and trackUndulatorKickMap() remove losses kick by kick */
  survivorPrefix[ip] =
    gpuKickMapTrackParticle(coord + ip * stride, &map, xpMap, ypMap, 0, &kick) ? 1 : 0;
  lossRound[ip] = survivorPrefix[ip] ? -1 : kick;
}

__global__ void gpuKickMapStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors, GPU_KICKMAP_DATA map,
  const double *xpMap, const double *ypMap, double zStart, double pRef) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination;
  int survives;

  if (ip >= nParticles)
    return;
  destination = survivorPrefix[ip];
  survives = destination < survivors;
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (survives) {
    gpuKickMapTrackParticle(target, &map, xpMap, ypMap, 1);
  } else {
    target[4] = zStart;
    target[5] = pRef * (1 + target[5]);
  }
}

__device__ __forceinline__ double gpuMultipoleCoefficient(int order, int i) {
  switch (order) {
  case 0:
    return 1.0;
  case 1:
    return 1.0;
  case 2:
    return i == 0 ? 0.5 : (i == 1 ? 1.0 : -0.5);
  case 3:
    if (i == 0)
      return 1.0 / 6.0;
    if (i == 1)
      return 0.5;
    if (i == 2)
      return -0.5;
    return -1.0 / 6.0;
  default:
    return 0.0;
  }
}

__device__ __forceinline__ void gpuMultipoleFillPowerArray(double value,
                                                           double *power,
                                                           int order) {
  power[0] = 1.0;
  for (int i = 1; i <= order; i++)
    power[i] = power[i - 1] * value;
}

__device__ __forceinline__ int gpuMultipoleConvertSlopesToMomenta(double *qx,
                                                                  double *qy,
                                                                  double xp,
                                                                  double yp,
                                                                  double delta,
                                                                  int expandHamiltonian) {
  if (expandHamiltonian) {
    *qx = (1 + delta) * xp;
    *qy = (1 + delta) * yp;
  } else {
    double denom = sqrt(1 + xp * xp + yp * yp);
#if TURBO_RECIPROCALS
    double factor = (1 + delta) / denom;
    *qx = factor * xp;
    *qy = factor * yp;
#else
    /* Match multipole.h: division is not reciprocal multiplication. */
    *qx = __ddiv_rn((1 + delta) * xp, denom);
    *qy = __ddiv_rn((1 + delta) * yp, denom);
#endif
  }
  return 1;
}

__device__ __forceinline__ int gpuMultipoleConvertMomentaToSlopes(double *xp,
                                                                  double *yp,
                                                                  double qx,
                                                                  double qy,
                                                                  double delta,
                                                                  int expandHamiltonian) {
  if (expandHamiltonian) {
    *xp = qx / (1 + delta);
    *yp = qy / (1 + delta);
  } else {
    double factor = (1 + delta) * (1 + delta) - qx * qx - qy * qy;
    if (factor <= 0)
      return 0;
    factor = sqrt(factor);
#if TURBO_RECIPROCALS
    factor = 1 / factor;
    *xp = qx * factor;
    *yp = qy * factor;
#else
    *xp = __ddiv_rn(qx, factor);
    *yp = __ddiv_rn(qy, factor);
#endif
  }
  return 1;
}

/* Dipole conversions deliberately use reciprocal multiplication on CPU,
 * independently of the multipole TURBO_RECIPROCALS setting. */
__device__ __forceinline__ int gpuDipoleConvertSlopesToMomenta(double *qx,
                                                                  double *qy,
                                                                  double xp,
                                                                  double yp,
                                                                  double delta,
                                                                  int expandHamiltonian) {
  if (expandHamiltonian) {
    *qx = (1 + delta) * xp;
    *qy = (1 + delta) * yp;
  } else {
    double factor = (1 + delta) / sqrt(1 + xp * xp + yp * yp);
    *qx = xp * factor;
    *qy = yp * factor;
  }
  return 1;
}

__device__ __forceinline__ int gpuDipoleConvertMomentaToSlopes(double *xp,
                                                                  double *yp,
                                                                  double qx,
                                                                  double qy,
                                                                  double delta,
                                                                  int expandHamiltonian) {
  if (expandHamiltonian) {
    double factor = 1 / (1 + delta);
    *xp = qx * factor;
    *yp = qy * factor;
  } else {
    double factor = (1 + delta) * (1 + delta) - qx * qx - qy * qy;
    if (factor <= 0)
      return 0;
    factor = 1 / sqrt(factor);
    *xp = qx * factor;
    *yp = qy * factor;
  }
  return 1;
}

__device__ __forceinline__ void gpuMultipoleApplyKick(double *qx, double *qy,
                                                      double *deltaQx,
                                                      double *deltaQy,
                                                      const double *xpow,
                                                      const double *ypow,
                                                      int order,
                                                      double KnL,
                                                      int skew) {
  double sumFx = 0;
  double sumFy = 0;

  for (int i = 0; i <= order; i++) {
    double f = gpuMultipoleCoefficient(order, i) * xpow[order - i] * ypow[i];
    if (i & 1)
      sumFx += f;
    else
      sumFy += f;
  }
  if (skew) {
    double temp = sumFx;
    sumFx = -sumFy;
    sumFy = temp;
  }
  *qx -= KnL * sumFy;
  *qy += KnL * sumFx;
  *deltaQx -= KnL * sumFy;
  *deltaQy += KnL * sumFx;
}

/* Same arithmetic, in the same order, as the particle loop of the CPU's linearized misalignment methods */
__device__ __forceinline__ void gpuApplyLinearMisalignment(
  double *x, double *xp, double *y, double *yp, double *s, double dp,
  const GPU_LINEAR_MISALIGNMENT *misalignment, int face) {
  double factor = (1 + dp) / sqrt(1 + *xp * *xp + *yp * *yp);
  double initial[6] = {*x, *xp * factor, *y, *yp * factor, *s, dp};
  double final[6];

  for (int i = 5; i >= 0; i--) {
    double sum = misalignment->C[face][i];
    for (int j = 5; j >= 0; j--)
      sum += misalignment->R[face][i][j] * initial[j];
    final[i] = sum;
  }
  factor = 1 / sqrt((1 + final[5]) * (1 + final[5]) - final[1] * final[1] -
                    final[3] * final[3]);
  *x = final[0];
  *xp = final[1] * factor;
  *y = final[2];
  *yp = final[3] * factor;
  *s = final[4];
}

/* Same arithmetic as the particle loop of the CPU's exact methods */
__device__ __forceinline__ void gpuApplyExactMisalignment(
  double *x, double *xp, double *y, double *yp, double *s, double dp,
  const GPU_EXACT_MISALIGNMENT *misalignment, int whichFace) {
  const GPU_EXACT_MISALIGNMENT_FACE &face = misalignment->face[whichFace];
  double factor, qx, qy, tB1, tB2, tB3, tB4, pz, factor1, factor2;
  double tC1, tC2, tC3, tC5, pzP, tD1, tD3, rD[3], pD[3];

  if (!misalignment->active[whichFace])
    return;
  factor = (1 + dp) / sqrt(1 + *xp * *xp + *yp * *yp);
  qx = *xp * factor;
  qy = *yp * factor;
  tB1 = face.csi[0] * *x + face.csi[1] * *y;
  tB2 = face.csi[0] * qx + face.csi[1] * qy;
  tB3 = face.eta[0] * *x + face.eta[1] * *y;
  tB4 = face.eta[0] * qx + face.eta[1] * qy;
  /* wedge drift */
  pz = sqrt((1 + dp) * (1 + dp) - tB2 * tB2 - tB4 * tB4);
  factor1 = tB1 / (1 - tB2 * face.tanThetaW / pz);
  factor2 = factor1 * face.tanThetaW / pz;
  tC1 = factor1 / face.cosThetaW;
  tC2 = tB2 * face.cosThetaW + pz * face.sinThetaW;
  tC3 = tB3 + tB4 * factor2;
  tC5 = *s + (1 + dp) * factor2;
  /* conventional drift */
  pzP = sqrt((1 + dp) * (1 + dp) - tC2 * tC2 - tB4 * tB4);
  tD1 = tC1 + face.LD * tC2 / pzP;
  tD3 = tC3 + face.LD * tB4 / pzP;
  *s = tC5 + face.LD * (1 + dp) / pzP;
  for (int i = 0; i < 3; i++) {
    rD[i] = face.csiFinal[i] * tD1 + face.etaFinal[i] * tD3;
    pD[i] = face.csiFinal[i] * tC2 + face.etaFinal[i] * tB4;
  }
  *x = (rD[0] * face.xAxis[0] + rD[1] * face.xAxis[1] + rD[2] * face.xAxis[2]) -
       (face.offset[0] * face.xFrame[0] + face.offset[1] * face.xFrame[1] + face.offset[2] * face.xFrame[2]);
  *y = (rD[0] * face.yAxis[0] + rD[1] * face.yAxis[1] + rD[2] * face.yAxis[2]) -
       (face.offset[0] * face.yFrame[0] + face.offset[1] * face.yFrame[1] + face.offset[2] * face.yFrame[2]);
  qx = pD[0] * face.xAxis[0] + pD[1] * face.xAxis[1] + pD[2] * face.xAxis[2];
  qy = pD[0] * face.yAxis[0] + pD[1] * face.yAxis[1] + pD[2] * face.yAxis[2];
  factor = 1 / sqrt((1 + dp) * (1 + dp) - qx * qx - qy * qy);
  *xp = qx * factor;
  *yp = qy * factor;
}

__device__ __forceinline__ void gpuMultipoleApplyTerm(double *qx, double *qy,
                                                      double *deltaQx,
                                                      double *deltaQy,
                                                      const double *xpow,
                                                      const double *ypow,
                                                      int order,
                                                      double KnL,
                                                      int skew) {
  const double *coefficient = gpuMultipoleCoefficientTable[order];
  double sumFx = 0;
  double sumFy = 0;

  for (int i = 0; i <= order; i++) {
    double f = coefficient[i] * xpow[order - i] * ypow[i];
    if (i & 1)
      sumFx += f;
    else
      sumFy += f;
  }
  if (skew) {
    double temp = sumFx;
    sumFx = -sumFy;
    sumFy = temp;
  }
  *qx -= KnL * sumFy;
  *qy += KnL * sumFx;
  *deltaQx -= KnL * sumFy;
  *deltaQy += KnL * sumFx;
}

__device__ __forceinline__ void gpuMultipoleApplyEdgeTerms(
  double *qx, double *qy, double *deltaQx, double *deltaQy, double x, double y,
  double *xpow, double *ypow, const GPU_MULTIPOLE_DATA *data) {
  *deltaQx = 0;
  *deltaQy = 0;
  gpuMultipoleFillPowerArray(x, xpow, data->maxTermOrder);
  gpuMultipoleFillPowerArray(y, ypow, data->maxTermOrder);
  for (int iTerm = 0; iTerm < data->nEdgeTerms; iTerm++)
    gpuMultipoleApplyTerm(qx, qy, deltaQx, deltaQy, xpow, ypow,
                          data->term[iTerm].order,
                          data->term[iTerm].strength,
                          data->term[iTerm].skew);
}

__device__ __forceinline__ double gpuIntegerPower(double value,
                                                   int exponent) {
  double result = 1;
  for (int i = 0; i < exponent; i++)
    result *= value;
  return result;
}

__device__ __forceinline__ int gpuApertureOpenSideLoses(
  int openSide, double x, double y, double xMax, double yMax) {
  switch (openSide) {
  case GPU_OPEN_PLUS_X:
    if (x > 0 && (fabs(y) < yMax || yMax <= 0))
      return 0;
    break;
  case GPU_OPEN_MINUS_X:
    if (x < 0 && (fabs(y) < yMax || yMax <= 0))
      return 0;
    break;
  case GPU_OPEN_PLUS_Y:
    if (y > 0 && (fabs(x) < xMax || xMax <= 0))
      return 0;
    break;
  case GPU_OPEN_MINUS_Y:
    if (y < 0 && (fabs(x) < xMax || xMax <= 0))
      return 0;
    break;
  default:
    break;
  }
  return 1;
}

/* Kernels that track each particle on its own use smaller blocks for a small beam, so that every
 * multiprocessor gets work, which matters most where double precision is slow.  A particle's result
 * doesn't depend on the block size.  ELEGANT_GPU_PARTICLE_THREADS fixes the size. */
static int gpuParticleThreads(long nParticles) {
  static int multiprocessors = -1, fixed = 0;
  int threads = 256;

  if (multiprocessors < 0) {
    const char *value = getenv("ELEGANT_GPU_PARTICLE_THREADS");
    int device = 0;

    fixed = value && *value ? atoi(value) : 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
      multiprocessors = 0;
  }
  if (fixed >= 32 && fixed <= 1024 && !(fixed & (fixed - 1)))
    return fixed;
  while (threads > 32 && (nParticles + threads - 1) / threads < 2L * multiprocessors)
    threads /= 2;
  return threads;
}

#define GPU_LOSS_COUNT_THREADS 256

/* Loss counts are integers, so a grid launch can't change a result */
__device__ __forceinline__ void gpuLossCountAdd(long *lostCount) {
  atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), 1ULL);
}

static inline int gpuLossCountBlocks(long nParticles) {
  return static_cast<int>((nParticles + GPU_LOSS_COUNT_THREADS - 1) / GPU_LOSS_COUNT_THREADS);
}

__device__ __forceinline__ int gpuInsideApertureLimit(
  double xInput, double yInput, const GPU_APERTURE_LIMIT_DATA &aperture) {
  double x = xInput - aperture.xCen, y = yInput - aperture.yCen;
  int outside = 0;

  if (!aperture.present)
    return 1;
  if (aperture.sinReverseTilt ||
      aperture.cosReverseTilt != 1) {
    double x0 = x, y0 = y;
    x = x0 * aperture.cosReverseTilt +
        y0 * aperture.sinReverseTilt;
    y = -x0 * aperture.sinReverseTilt +
        y0 * aperture.cosReverseTilt;
  }
  if (!isfinite(x) || !isfinite(y))
    return 0;
  if (!aperture.elliptical ||
      aperture.xMax <= 0 || aperture.yMax <= 0) {
    outside = (aperture.xMax > 0 && fabs(x) > aperture.xMax) ||
              (aperture.yMax > 0 && fabs(y) > aperture.yMax);
  } else {
    outside =
      gpuIntegerPower(x / aperture.xMax, aperture.xExponent) +
      gpuIntegerPower(y / aperture.yMax, aperture.yExponent) >= 1;
  }
  if (!outside)
    return 1;
  return aperture.openSide &&
         !gpuApertureOpenSideLoses(aperture.openSide, x, y,
                                   aperture.xMax, aperture.yMax);
}

__device__ __noinline__ int gpuInsideApertureContour(
  double xInput, double yInput, const GPU_APERTURE_LIMIT_DATA &aperture,
  const GPU_APERTURE_CONTOUR_DATA &contour);

__device__ __forceinline__ void gpuExactDriftLocal(
  double &x, double xp, double &y, double yp, double &path,
  double length) {
  x += xp * length;
  y += yp * length;
  path += length * sqrt(1 + xp * xp + yp * yp);
}

__device__ __forceinline__ void gpuApplyQuadFringeMatrix(
  double &x, double &px, double &y, double &py,
  const double matrix[8]) {
  double x0 = x, px0 = px, y0 = y, py0 = py;
  x = matrix[0] * x0 + matrix[1] * px0;
  px = matrix[2] * x0 + matrix[3] * px0;
  y = matrix[4] * y0 + matrix[5] * py0;
  py = matrix[6] * y0 + matrix[7] * py0;
}

__device__ int gpuQuadFringeParticle(
  double &x, double &xp, double &y, double &yp, double &path,
  double delta, double K1, int inFringe, int higherOrder,
  int linearFlag, double nonlinearFactor,
  const double matrix1[8], const double matrix2[8]) {
  double px, py, denom;
  double a, dx = 0, dpx = 0, dy = 0, dpy = 0, ds = 0;

  denom = sqrt(1 + xp * xp + yp * yp);
  px = (1 + delta) * xp / denom;
  py = (1 + delta) * yp / denom;
  if (linearFlag)
    gpuApplyQuadFringeMatrix(x, px, y, py, matrix1);

  a = -inFringe * K1 / (12 * (1 + delta));
  if (abs(higherOrder) > 1) {
    double xpow[9], ypow[9], apow[4];
    xpow[0] = ypow[0] = apow[0] = 1;
    if (abs(higherOrder) > 2) {
      for (int i = 1; i < 9; i++) {
        xpow[i] = xpow[i - 1] * x;
        ypow[i] = ypow[i - 1] * y;
      }
      for (int i = 1; i < 4; i++)
        apow[i] = apow[i - 1] * a;
      dpx =
        (a *
         (12 * a *
              (-4 * py * x * y + px * (xpow[2] - 5 * ypow[2])) *
              (xpow[2] - ypow[2]) -
          24 * (-2 * py * x * y + px * (xpow[2] + ypow[2])) +
          4 * apow[2] *
            (-2 * py * x * y *
               (3 * xpow[4] + 50 * xpow[2] * ypow[2] - 21 * ypow[4]) +
             px *
               (xpow[6] + 75 * xpow[4] * ypow[2] -
                105 * xpow[2] * ypow[4] - 35 * ypow[6])) +
          3 * apow[3] *
            (-8 * py * x * y *
               (xpow[6] - 27 * xpow[4] * ypow[2] +
                83 * xpow[2] * ypow[4] + 7 * ypow[6]) +
             px *
               (xpow[8] - 108 * xpow[6] * ypow[2] +
                830 * xpow[4] * ypow[4] + 196 * xpow[2] * ypow[6] +
                105 * ypow[8])))) /
        8;
      dpy =
        (a *
         (-8 * px * x * y *
            (6 - 6 * a * (xpow[2] - ypow[2]) +
             apow[2] *
               (21 * xpow[4] - 50 * xpow[2] * ypow[2] - 3 * ypow[4]) +
             3 * apow[3] *
               (7 * xpow[6] + 83 * xpow[4] * ypow[2] -
                27 * xpow[2] * ypow[4] + ypow[6])) +
          py *
            (315 * apow[3] * xpow[8] +
             28 * apow[2] * xpow[6] * (5 + 21 * a * ypow[2]) +
             30 * a * xpow[4] *
               (2 + 14 * a * ypow[2] + 83 * apow[2] * ypow[4]) +
             ypow[2] *
               (24 + 12 * a * ypow[2] - 4 * apow[2] * ypow[4] +
                3 * apow[3] * ypow[6]) -
             12 * xpow[2] *
               (-2 + 6 * a * ypow[2] + 25 * apow[2] * ypow[4] +
                27 * apow[3] * ypow[6])))) /
        8;
      if (higherOrder > 0) {
        double squareDifference =
          (xpow[2] - ypow[2]) * (xpow[2] - ypow[2]);
        dx =
          (a * x *
           (8 * (xpow[2] + 3 * ypow[2]) +
            4 * apow[2] *
              (5 * xpow[6] + 21 * xpow[4] * ypow[2] -
               25 * xpow[2] * ypow[4] - ypow[6]) +
            apow[3] *
              (35 * xpow[8] + 84 * xpow[6] * ypow[2] +
               498 * xpow[4] * ypow[4] - 108 * xpow[2] * ypow[6] +
               3 * ypow[8]) +
            12 * a * squareDifference)) /
          8;
        dy =
          (a * y *
           (-8 * (3 * xpow[2] + ypow[2]) +
            4 * apow[2] *
              (xpow[6] + 25 * xpow[4] * ypow[2] -
               21 * xpow[2] * ypow[4] - 5 * ypow[6]) +
            apow[3] *
              (3 * xpow[8] - 108 * xpow[6] * ypow[2] +
               498 * xpow[4] * ypow[4] + 84 * xpow[2] * ypow[6] +
               35 * ypow[8]) +
            12 * a * squareDifference)) /
          8;
      }
    } else {
      for (int i = 1; i < 4; i++) {
        xpow[i] = xpow[i - 1] * x;
        ypow[i] = ypow[i - 1] * y;
      }
      dpx = 3 * a * (-px * xpow[2] + 2 * py * x * y - px * ypow[2]);
      dpy = 3 * a * (py * xpow[2] - 2 * px * x * y + py * ypow[2]);
      if (higherOrder > 0) {
        dx = a * (xpow[3] + 3 * x * ypow[2]);
        dy = a * (-3 * xpow[2] * y - ypow[3]);
      }
    }
    ds = (a / (1 + delta)) *
         (3 * py * y * x * x - px * x * x * x -
          3 * px * x * y * y + py * y * y * y);
  }

  x += nonlinearFactor * dx;
  px += nonlinearFactor * dpx;
  y += nonlinearFactor * dy;
  py += nonlinearFactor * dpy;
  if (linearFlag)
    gpuApplyQuadFringeMatrix(x, px, y, py, matrix2);
  path -= nonlinearFactor * ds;

  denom = (1 + delta) * (1 + delta) - px * px - py * py;
  if (denom <= 0)
    return 0;
  denom = sqrt(denom);
  xp = px / denom;
  yp = py / denom;
  return isfinite(x) && isfinite(xp) && isfinite(y) && isfinite(yp) &&
         isfinite(path);
}

__device__ __noinline__ void gpuSpinRotate(double *S, double rx, double ry, double rz);

/* rotateSpinsCoordinateSystem() for one particle */
__device__ __forceinline__ void gpuMultipoleSpinRoll(double *S, double cosTilt, double sinTilt) {
  double S0 = S[0], S1 = S[1];

  S[0] = cosTilt * S0 + sinTilt * S1;
  S[1] = -sinTilt * S0 + cosTilt * S1;
}

/* updateSpinQuaternionLocalFS() with Bz zero */
__device__ __noinline__ void gpuSpinUpdateLocalFS(double *S, double Bx, double By, double x, double xp,
                                                  double yp, double ds, double h, double p, double charge,
                                                  double massC, double a) {
  const double tiny = 1e-300;
  double B[3], betav[3], Omega[3];
  double gamma = sqrt(p * p + 1);
  double beta = p / gamma;
  double onePlusHx = 1.0 + h * x;
  double dlds = sqrt(onePlusHx * onePlusHx + xp * xp + yp * yp);
  double betaDotB, coeff, kpar;

  B[0] = Bx;
  B[1] = By;
  B[2] = 0;
  if (dlds < tiny)
    return;
  betav[0] = xp / dlds * beta;
  betav[1] = yp / dlds * beta;
  betav[2] = onePlusHx / dlds * beta;
  betaDotB = betav[0] * B[0] + betav[1] * B[1] + betav[2] * B[2];
  coeff = -(charge / (massC * beta)) * dlds;
  kpar = a * gamma / (gamma + 1.0);
  for (int i = 0; i < 3; i++)
    Omega[i] = coeff * ((a + 1.0 / gamma) * B[i] - kpar * betaDotB * betav[i]);
  gpuSpinRotate(S, Omega[0] * ds, (Omega[1] + h) * ds, Omega[2] * ds);
}

/* The spin update multipole_tracking2() makes after each kick */
__device__ __forceinline__ void gpuMultipoleSpin(double *S, double deltaQx, double deltaQy, double x,
                                                 double xp, double yp, double ds, double dp,
                                                 const GPU_MULTIPOLE_DATA *data) {
  /* turnSpinForKick(): a zero-length magnet passes the kick as the field over a unit step */
  if (ds == 0)
    ds = 1;
  gpuSpinUpdateLocalFS(S, data->spinRigidity * deltaQy / ds, -data->spinRigidity * deltaQx / ds, x, xp, yp, ds,
                       0.0, data->Po * (1 + dp), data->spinCharge, data->spinMassC, data->spinAnomaly);
}

template <bool Radiation, bool Aperture>
__device__ int gpuMultipoleTrackParticleData(
  double *part, int stride, int writeOutput,
  const GPU_MULTIPOLE_DATA *data, long ip = 0) {
  const double *isrDraws = data->isrCoef ? data->isrDraws + ip * data->isrDrawsPerParticle : NULL;
  double driftFrac[8];
  double kickFrac[8];
  double KnLp[3];
  double xpow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
  double ypow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double dp = part[5];
  double initialDp = dp;
  double qx, qy, s = 0;
  double lossDistance = 0;
  double timeCoordinate = part[4];
  double beta0 = 0;
  double drift, xkick, ykick;
  double spin[3] = {0, 0, 0};
  int nSubsteps = 0;
  int maxOrder = 0;

  (void)stride;
  if (Radiation) {
    double p = data->Po * (1 + dp);
    beta0 = p / sqrt(p * p + 1);
  }
  if (data->spinOffset) {
    for (int i = 0; i < 3; i++)
      spin[i] = part[data->spinOffset + i];
    /* rotateBeamCoordinatesForMisalignment() turns spins with the coordinates */
    if (data->sinTilt || data->cosTilt != 1)
      gpuMultipoleSpinRoll(spin, data->cosTilt, data->sinTilt);
  }
  if (data->hasExactMisalignment) {
    gpuApplyExactMisalignment(&x, &xp, &y, &yp, &timeCoordinate, dp,
                              &data->exactMisalignment, 0);
  } else if (data->hasLinearMisalignment) {
    gpuApplyLinearMisalignment(&x, &xp, &y, &yp, &timeCoordinate, dp,
                               &data->misalignment, 0);
  } else if (data->dx || data->dy || data->dz) {
    /* offsetBeamCoordinatesForMisalignment() adds DZ's path to the coordinate itself, before the
     * element's own path is added to it */
    timeCoordinate += data->dz * sqrt(1 + xp * xp + yp * yp);
    x = x - data->dx + data->dz * xp;
    y = y - data->dy + data->dz * yp;
  }
  if (data->sinTilt || data->cosTilt != 1) {
    double x0 = x;
    double y0 = y;
    double xp0 = xp;
    double yp0 = yp;
    x = x0 * data->cosTilt + y0 * data->sinTilt;
    y = -x0 * data->sinTilt + y0 * data->cosTilt;
    xp = xp0 * data->cosTilt + yp0 * data->sinTilt;
    yp = -xp0 * data->sinTilt + yp0 * data->cosTilt;
  }
  /* exactDrift() and quadFringe() act on the particle's path coordinate itself, at both ends, and
   * the integrator adds the body's path to it in between.  Gathering all of it in s and adding once
   * rounds differently: a KQUAD with LEFFECTIVE came out a unit off in s in 40% of the particles. */
  if (data->endDrift)
    gpuExactDriftLocal(x, xp, y, yp, timeCoordinate, data->endDrift);
  if (data->edge1Effects &&
      !gpuQuadFringeParticle(x, xp, y, yp, timeCoordinate,
                             dp, data->k1, -1, data->edge1Effects,
                             data->edge1Linear,
                             data->edge1NonlinearFactor,
                             data->edge1M1, data->edge1M2))
    return 0;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
    return 0;

  if (!gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return 0;
  if (data->nEdgeTerms) {
    double edgeDeltaQx, edgeDeltaQy;

    gpuMultipoleApplyEdgeTerms(&qx, &qy, &edgeDeltaQx, &edgeDeltaQy, x, y, xpow, ypow, data);
    /* At the entrance the CPU turns the spin with the slopes after the edge kick, over a hundredth
     * of a slice */
    if (data->spinOffset && !data->spinRollOnly) {
      if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                              data->expandHamiltonian))
        return 0;
      gpuMultipoleSpin(spin, edgeDeltaQx, edgeDeltaQy, x, xp, yp,
                       data->drift / data->nSlices / 100.0, dp, data);
    }
  }
  if (data->initialSlopeRoundTrip) {
    if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                            data->expandHamiltonian))
      return 0;
  }

  switch (data->integrationOrder) {
  case 2:
    nSubsteps = 2;
    driftFrac[0] = 0.5;
    driftFrac[1] = 0.5;
    kickFrac[0] = 1.0;
    kickFrac[1] = 0.0;
    break;
  case 4: {
    const double beta = 1.25992104989487316477;
    nSubsteps = 4;
    driftFrac[0] = 0.5 / (2 - beta);
    driftFrac[1] = (1 - beta) / (2 - beta) / 2;
    driftFrac[2] = (1 - beta) / (2 - beta) / 2;
    driftFrac[3] = 0.5 / (2 - beta);
    kickFrac[0] = 1.0 / (2 - beta);
    kickFrac[1] = -beta / (2 - beta);
    kickFrac[2] = 1.0 / (2 - beta);
    kickFrac[3] = 0.0;
    break;
  }
  case 6:
    nSubsteps = 8;
    driftFrac[0] = 0.39225680523878;
    driftFrac[1] = 0.5100434119184585;
    driftFrac[2] = -0.47105338540975655;
    driftFrac[3] = 0.0687531682525181;
    driftFrac[4] = 0.0687531682525181;
    driftFrac[5] = -0.47105338540975655;
    driftFrac[6] = 0.5100434119184585;
    driftFrac[7] = 0.39225680523878;
    kickFrac[0] = 0.784513610477560;
    kickFrac[1] = 0.235573213359357;
    kickFrac[2] = -1.17767998417887;
    kickFrac[3] = 1.3151863206839063;
    kickFrac[4] = -1.17767998417887;
    kickFrac[5] = 0.235573213359357;
    kickFrac[6] = 0.784513610477560;
    kickFrac[7] = 0.0;
    break;
  default:
    return 0;
  }

  drift = data->drift / data->nSlices;
  xkick = data->xkick / data->nSlices;
  ykick = data->ykick / data->nSlices;
  for (int i = 0; i < 3; i++) {
    KnLp[i] = data->KnL[i] / data->nSlices;
    if (data->KnL[i] && data->order[i] > maxOrder)
      maxOrder = (int)data->order[i];
  }
  if ((xkick || ykick) && maxOrder < 0)
    maxOrder = 0;
  if (maxOrder > 3)
    return 0;
  if (data->maxTermOrder > maxOrder)
    maxOrder = data->maxTermOrder;

  for (long iKick = 0; iKick < data->nSlices; iKick++) {
    double deltaQx = 0;
    double deltaQy = 0;

    if (Aperture &&
        (!gpuInsideApertureLimit(x + data->dx, y + data->dy, data->aperture) ||
         (data->contour.nContours &&
          !gpuInsideApertureContour(x + data->dx, y + data->dy, data->aperture, data->contour)))) {
      if (writeOutput) {
        part[0] = x;
        part[2] = y;
        part[4] = data->lossZStart + lossDistance;
        part[5] = data->Po * (1 + initialDp);
        if (data->spinOffset)
          for (int i = 0; i < 3; i++)
            part[data->spinOffset + i] = spin[i];
      }
      return 0;
    }
    for (int step = 0; step < nSubsteps; step++) {
      /* the CPU's step of zero in a zero-length magnet, which the spin turn sees */
      double dsh = 0;

      if (drift) {
        dsh = drift * driftFrac[step];
        x += xp * dsh;
        y += yp * dsh;
        if (data->expandHamiltonian)
          s += dsh * (1 + (xp * xp + yp * yp) / 2);
        else
          s += dsh * sqrt(1 + xp * xp + yp * yp);
        if (Aperture)
          lossDistance += dsh;
      }

      if (!kickFrac[step])
        break;

      gpuMultipoleFillPowerArray(x, xpow, maxOrder);
      gpuMultipoleFillPowerArray(y, ypow, maxOrder);
      deltaQx = deltaQy = 0;

      for (int iOrder = 0; iOrder < 3; iOrder++) {
        if (data->KnL[iOrder])
          gpuMultipoleApplyKick(&qx, &qy, &deltaQx, &deltaQy,
                                xpow, ypow, (int)data->order[iOrder],
                                KnLp[iOrder] * kickFrac[step],
                                data->skew[iOrder]);
      }
      if (xkick)
        gpuMultipoleApplyKick(&qx, &qy, &deltaQx, &deltaQy,
                              xpow, ypow, 0, -xkick * kickFrac[step], 0);
      if (ykick)
        gpuMultipoleApplyKick(&qx, &qy, &deltaQx, &deltaQy,
                              xpow, ypow, 0, -ykick * kickFrac[step], 1);
      for (int iTerm = data->nEdgeTerms;
           iTerm < data->nEdgeTerms + data->nSteeringTerms; iTerm++)
        gpuMultipoleApplyTerm(&qx, &qy, &deltaQx, &deltaQy, xpow, ypow,
                              data->term[iTerm].order,
                              data->term[iTerm].strength *
                                (data->term[iTerm].skew ? ykick : xkick) *
                                kickFrac[step],
                              data->term[iTerm].skew);
      for (int iTerm = data->nEdgeTerms + data->nSteeringTerms;
           iTerm < data->nEdgeTerms + data->nSteeringTerms + data->nBodyTerms;
           iTerm++)
        gpuMultipoleApplyTerm(&qx, &qy, &deltaQx, &deltaQy, xpow, ypow,
                              data->term[iTerm].order,
                              data->term[iTerm].strength * kickFrac[step] /
                                data->nSlices,
                              data->term[iTerm].skew);

      if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                              data->expandHamiltonian))
        return 0;
      if (data->spinOffset && !data->spinRollOnly)
        gpuMultipoleSpin(spin, deltaQx, deltaQy, x, xp, yp, 2 * dsh, dp, data);

      if (data->radiationBlock && drift) {
        double onePlusDp = 1 + dp;

        qx /= (1 + dp);
        qy /= (1 + dp);
        if (Radiation || isrDraws) {
          double deltaFactor = onePlusDp * onePlusDp;
          double normalizedDeltaQx = deltaQx / kickFrac[step];
          double normalizedDeltaQy = deltaQy / kickFrac[step];
          double F2 =
            (normalizedDeltaQx / drift - xkick / drift) *
              (normalizedDeltaQx / drift - xkick / drift) +
            (normalizedDeltaQy / drift + ykick / drift) *
              (normalizedDeltaQy / drift + ykick / drift);
          double dsFactor = sqrt(1 + xp * xp + yp * yp);

          if (Radiation)
            dp -= data->radCoef * deltaFactor * F2 * (dsFactor * drift * kickFrac[step]);
          if (isrDraws) {
            dp -= data->isrCoef * deltaFactor * pow(F2, 0.75) *
                  sqrt(dsFactor * drift / (nSubsteps - 1)) * *isrDraws;
            isrDraws++;
          }
        }
        qx *= (1 + dp);
        qy *= (1 + dp);
        if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                                data->expandHamiltonian))
          return 0;
      }
    }
  }

  if (Aperture &&
      (!gpuInsideApertureLimit(x + data->dx, y + data->dy, data->aperture) ||
       (data->contour.nContours &&
        !gpuInsideApertureContour(x + data->dx, y + data->dy, data->aperture, data->contour)))) {
    if (writeOutput) {
      part[0] = x;
      part[2] = y;
      part[4] = data->lossZStart + lossDistance;
      part[5] = data->Po * (1 + initialDp);
      if (data->spinOffset)
        for (int i = 0; i < 3; i++)
          part[data->spinOffset + i] = spin[i];
    }
    return 0;
  }
  if (data->nEdgeTerms) {
    double edgeDeltaQx, edgeDeltaQy;

    gpuMultipoleApplyEdgeTerms(&qx, &qy, &edgeDeltaQx, &edgeDeltaQy, x, y, xpow, ypow, data);
    /* At the exit the CPU turns the spin with the slopes from before the edge kick */
    if (data->spinOffset && !data->spinRollOnly)
      gpuMultipoleSpin(spin, edgeDeltaQx, edgeDeltaQy, x, xp, yp,
                       data->drift / data->nSlices / 100.0, dp, data);
  }
  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;

  if (Radiation) {
    double p = data->Po * (1 + dp);
    double beta1 = p / sqrt(p * p + 1);

    timeCoordinate =
      beta1 * (timeCoordinate / beta0 + 2 * s / (beta0 + beta1));
  } else {
    /* the integrator's coord[4] += s, before the exit fringe and end drift */
    timeCoordinate += s;
    s = 0;
  }

  if (data->edge2Effects &&
      !gpuQuadFringeParticle(x, xp, y, yp, timeCoordinate,
                             dp, data->k1, 1, data->edge2Effects,
                             data->edge2Linear,
                             data->edge2NonlinearFactor,
                             data->edge2M1, data->edge2M2))
    return 0;
  if (data->endDrift)
    gpuExactDriftLocal(x, xp, y, yp, timeCoordinate, data->endDrift);
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
    return 0;

  if (data->sinTilt || data->cosTilt != 1) {
    double x0 = x;
    double y0 = y;
    double xp0 = xp;
    double yp0 = yp;
    x = x0 * data->cosTilt - y0 * data->sinTilt;
    y = x0 * data->sinTilt + y0 * data->cosTilt;
    xp = xp0 * data->cosTilt - yp0 * data->sinTilt;
    yp = xp0 * data->sinTilt + yp0 * data->cosTilt;
  }
  if (data->hasExactMisalignment) {
    if (!Radiation) {
      timeCoordinate += s;
      s = 0;
    }
    gpuApplyExactMisalignment(&x, &xp, &y, &yp, &timeCoordinate, dp,
                              &data->exactMisalignment, 1);
  } else if (data->hasLinearMisalignment == 1) {
    /* 2 is a linear method with no misalignment and no tilt: multipole_tracking2() transforms at the
     * entrance whatever the offsets, and at the exit only if one of them is set */
    if (!Radiation) {
      timeCoordinate += s;
      s = 0;
    }
    gpuApplyLinearMisalignment(&x, &xp, &y, &yp, &timeCoordinate, dp,
                               &data->misalignment, 1);
  } else if (!data->hasLinearMisalignment && (data->dx || data->dy || data->dz)) {
    if (!Radiation) {
      timeCoordinate += s;
      s = 0;
    }
    timeCoordinate -= data->dz * sqrt(1 + xp * xp + yp * yp);
    x = x + data->dx - data->dz * xp;
    y = y + data->dy - data->dz * yp;
  }

  if (data->spinOffset && (data->sinTilt || data->cosTilt != 1))
    gpuMultipoleSpinRoll(spin, data->cosTilt, -data->sinTilt);

  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    if (Radiation)
      part[4] = timeCoordinate;
    else
      part[4] = timeCoordinate + s;
    part[5] = dp;
    if (data->spinOffset)
      for (int i = 0; i < 3; i++)
        part[data->spinOffset + i] = spin[i];
  }
  return 1;
}

template <bool Radiation, bool Aperture>
__device__ int gpuMultipoleTrackParticle(double *part, int stride,
                                         int writeOutput, long ip = 0) {
  return gpuMultipoleTrackParticleData<Radiation, Aperture>(
    part, stride, writeOutput, &gpuMultipoleData, ip);
}

template <bool Radiation, bool Aperture>
__global__ void gpuMultipolePredicateKernel(double *coord, long nParticles,
                                            int stride, long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;

  for (long ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    if (!gpuMultipoleTrackParticle<Radiation, Aperture>(part, stride, 0))
      localCount++;
  }

  partial[thread] = localCount;
  __syncthreads();
  for (long offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (thread < offset)
      partial[thread] += partial[thread + offset];
    __syncthreads();
  }
  if (thread == 0)
    *lostCount = partial[0];
}

template <bool Radiation, bool Aperture>
__global__ void gpuMultipoleTrackKernel(double *coord, long nParticles,
                                        int stride) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  gpuMultipoleTrackParticle<Radiation, Aperture>(
    coord + ip * stride, stride, 1);
}

template <bool Radiation, bool Aperture>
__global__ void gpuMultipoleTrackCheckedKernel(
  double *coord, long nParticles, int stride,
  unsigned long long *lostCount, long *survivorFlag) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuMultipoleTrackParticle<Radiation, Aperture>(
          coord + ip * stride, stride, 1, ip))
      localCount = 1;
    if (survivorFlag)
      survivorFlag[ip] = localCount ? 0 : 1;
  }
  partial[threadIdx.x] = localCount;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

template <bool Radiation, bool Aperture>
__global__ void gpuMultipoleSurvivorFlagKernel(
  double *coord, long nParticles, int stride, long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuMultipoleTrackParticle<Radiation, Aperture>(
      coord + ip * stride, stride, 0) ? 1 : 0;
}

template <bool Radiation, bool Aperture>
__global__ void gpuMultipoleStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination;
  int survives;

  if (ip >= nParticles)
    return;
  destination = survivorPrefix[ip];
  survives = destination < survivors;
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuMultipoleTrackParticle<Radiation, Aperture>(target, stride, 1);
}

__device__ __forceinline__ void gpuExactCorrectorRotate(
  double *x, double *xp, double *y, double *yp, double cosTilt,
  double sinTilt) {
  double x0 = *x, xp0 = *xp, y0 = *y, yp0 = *yp;
  *x = x0 * cosTilt + y0 * sinTilt;
  *y = -x0 * sinTilt + y0 * cosTilt;
  *xp = xp0 * cosTilt + yp0 * sinTilt;
  *yp = -xp0 * sinTilt + yp0 * cosTilt;
}

__device__ int gpuExactCorrectorTrackParticle(
  double *part, int writeOutput, const GPU_EXACT_CORRECTOR_DATA &data) {
  double theta0 = data.theta0;
  double x = part[0], xp = part[1], y = part[2], yp = part[3];
  double s = part[4], dp = part[5];
  int survives = 1;
  int lostAtExit = 0;

  if (theta0 == 0) {
    if (writeOutput)
      gpuExactDriftParticle(part, data.length);
    return 1;
  }

  if (data.dx || data.dy || data.dz) {
    s += data.dz * sqrt(1 + xp * xp + yp * yp);
    x = x - data.dx + data.dz * xp;
    y = y - data.dy + data.dz * yp;
  }
  if (data.sinTilt || data.cosTilt != 1)
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, data.sinTilt);

  {
    double rho0 = data.rho0;
    if (rho0 == 0 || data.length == 0) {
      xp = tan(theta0 / (1 + dp) + gpuAccurateAtan(xp));
    } else {
      double alpha = gpuAccurateAtan(xp);
      double rho = rho0 * (1 + dp);
      double arg = data.length / rho + sin(alpha);
      if (rho > 0 && arg <= 1 && arg >= -1) {
        double theta = gpuAccurateAsin(arg) - alpha;
        double denominator = sqrt(1 + xp * xp - yp * yp);
        double dyf = yp / denominator;
        double lprod = theta * rho * sqrt(1 + dyf * dyf);
        if (lprod < data.length)
          lprod = data.length;
        y += dyf * theta * rho;
        s += lprod;
        x += rho * (cos(alpha) - cos(theta + alpha));
        xp = tan(theta + alpha);
        /* On the CPU a slope above 1 reaches the element exit with y = NaN, and limit_amplitudes
           removes it there. The device skips that check, so record the same loss here. */
        if (!(denominator > 0)) {
          survives = 0;
          lostAtExit = 1;
        }
      } else {
        survives = 0;
      }
    }
  }

  if (data.sinTilt || data.cosTilt != 1)
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, -data.sinTilt);
  if (data.dx || data.dy || data.dz) {
    s -= data.dz * sqrt(1 + xp * xp + yp * yp);
    x = x + data.dx - data.dz * xp;
    y = y + data.dy - data.dz * yp;
  }

  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    if (survives) {
      part[4] = s;
      part[5] = dp;
    } else {
      part[4] = lostAtExit ? data.zStart + data.length : data.zStart;
      part[5] = data.pCentral * (1 + dp);
    }
  }
  return survives;
}

__global__ void gpuExactCorrectorSurvivorFlagKernel(
  double *coord, long nParticles, int stride,
  GPU_EXACT_CORRECTOR_DATA data, long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip < nParticles)
    survivorPrefix[ip] = gpuExactCorrectorTrackParticle(
      coord + ip * stride, 1, data) ? 1 : 0;
}

__global__ void gpuExactCorrectorStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip >= nParticles)
    return;
  long destination = survivorPrefix[ip];
  double *part = coord + ip * stride;
  double *target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
}

__device__ __forceinline__ void gpuTaperRectangularPlaneHit(
  double coordinate, double slope, double start, double end,
  double length, double sign, int *lost, double *zLost) {
  if (start <= 0 || end <= 0)
    return;
  if (sign * coordinate > start) {
    *lost = 1;
    *zLost = 0;
    return;
  }
  double denominator = sign * end - sign * start - slope * length;
  double candidate =
    (coordinate - sign * start) * length / denominator;
  if (candidate >= 0 && candidate <= length) {
    *lost = 1;
    if (candidate < *zLost)
      *zLost = candidate;
  }
}

/* limit_amplitudes.c's insideTaperedEllipse().  atan2, cos, sin and pow are the device's, so a point within a
 * hair of the boundary may fall on the other side than on the CPU; such a test is counted in *uncertain. */
__device__ int gpuTaperInsideEllipse(double x0, double xp, double y0, double yp, double z,
                                     const GPU_TAPER_APERTURE_DATA &data) {
  double x, y, r2, theta, ct, st, a, b, xe, ye, re2;

  x = x0 + xp * z;
  y = y0 + yp * z;
  r2 = x * x + y * y;
  theta = atan2(y, x);
  a = data.dadz * z + data.xStart;
  b = data.dbdz * z + data.yStart;
  ct = cos(theta);
  st = sin(theta);
  xe = a * pow(fabs(ct), (double)(2 / data.xExponent)) * (ct < 0 ? -1 : (ct > 0 ? 1 : 0));
  ye = b * pow(fabs(st), (double)(2 / data.yExponent)) * (st < 0 ? -1 : (st > 0 ? 1 : 0));
  re2 = xe * xe + ye * ye;
  if (data.uncertain && fabs(re2 - r2) <= 1e-12 * (re2 > r2 ? re2 : r2))
    atomicAdd(data.uncertain, 1ULL);
  if (re2 <= r2)
    return 0;
  return 1;
}

__device__ int gpuTaperApertureTrackParticle(
  double *part, int writeOutput, const GPU_TAPER_APERTURE_DATA &data) {
  double x = part[0], xp = part[1], y = part[2], yp = part[3];
  double zLost = 0, sDrifted = 0, xLocal = 0, xpLocal = 0, yLocal = 0;
  int lost = 0, sReady = 0;

  if (data.type == GPU_TAPER_APERTURE_CIRCULAR) {
    double xLocal = x - data.dx;
    double yLocal = y - data.dy;
    double rho = 0, rho2 = 0, rStart = 0, rStart2 = 0;
    double r = data.xStart < data.xEnd ? data.xStart : data.xEnd;
    double r2Limit = r * r;
    if (data.length > 0) {
      rho = (data.xEnd - data.xStart) / data.length;
      rho2 = rho * rho;
      rStart = data.xStart;
      rStart2 = rStart * rStart;
    }
    if (xLocal * xLocal + yLocal * yLocal >= rStart2) {
      lost = 1;
    } else if (rho) {
      double a = xp * xp + yp * yp - rho2;
      double b = 2 * xLocal * xp + 2 * yLocal * yp -
                 2 * rStart * rho;
      double c = xLocal * xLocal + yLocal * yLocal - rStart2;
      double determinant = b * b - 4 * a * c;
      if (determinant >= 0) {
        zLost = (-b + sqrt(determinant)) / (2 * a);
        if (zLost >= 0 && zLost <= data.length)
          lost = 1;
      }
    } else if (xLocal * xLocal + yLocal * yLocal >= r2Limit) {
      lost = 1;
    }
  } else if (data.type == GPU_TAPER_APERTURE_RECTANGULAR) {
    x -= data.dx;
    y -= data.dy;
    if (data.sinTilt || data.cosTilt != 1)
      gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                              data.cosTilt, data.sinTilt);
    zLost = DBL_MAX;
    gpuTaperRectangularPlaneHit(x, xp, data.xStart, data.xEnd,
                                data.length, 1, &lost, &zLost);
    gpuTaperRectangularPlaneHit(x, xp, data.xStart, data.xEnd,
                                data.length, -1, &lost, &zLost);
    gpuTaperRectangularPlaneHit(y, yp, data.yStart, data.yEnd,
                                data.length, 1, &lost, &zLost);
    gpuTaperRectangularPlaneHit(y, yp, data.yStart, data.yEnd,
                                data.length, -1, &lost, &zLost);
    if (!lost) {
      zLost = data.length;
      /* exactDrift() in the tilted frame, as on the CPU */
      sDrifted = part[4] + data.length * sqrt(1 + xp * xp + yp * yp);
      sReady = 1;
    }
    x += zLost * xp;
    y += zLost * yp;
    if (data.sinTilt || data.cosTilt != 1)
      gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                              data.cosTilt, -data.sinTilt);
    x += data.dx;
    y += data.dy;
  } else if (data.type == GPU_TAPER_APERTURE_ELLIPTICAL) {
    /* trackThroughTaperApElliptical() */
    int lost0 = 0, lost1 = 0, lost2;
    double z0 = 0, z1 = data.length;

    if (data.dx || data.dy) {
      x -= data.dx;
      y -= data.dy;
    }
    if (data.sinTilt || data.cosTilt != 1)
      gpuExactCorrectorRotate(&x, &xp, &y, &yp, data.cosTilt, data.sinTilt);
    if (!gpuTaperInsideEllipse(x, xp, y, yp, z0, data)) {
      lost0 = 1;
      zLost = z0;
    } else if (data.length > 0) {
      if (!gpuTaperInsideEllipse(x, xp, y, yp, z1, data)) {
        lost1 = 1;
        while ((z1 - z0) > data.resolution) {
          zLost = (z0 + z1) / 2;
          lost2 = !gpuTaperInsideEllipse(x, xp, y, yp, zLost, data);
          if (lost2 == lost1)
            z1 = zLost;
          else if (lost2 == lost0)
            z0 = zLost;
        }
        zLost = (z0 + z1) / 2;
      }
    }
    lost = lost0 + lost1;
    if (lost) {
      x += zLost * xp;
      y += zLost * yp;
      xLocal = x;
      xpLocal = xp;
      yLocal = y;
    } else {
      x += xp * data.length;
      y += yp * data.length;
      sDrifted = part[4] + data.length * sqrt(1 + xp * xp + yp * yp);
      sReady = 1;
    }
    if (data.sinTilt || data.cosTilt != 1)
      gpuExactCorrectorRotate(&x, &xp, &y, &yp, data.cosTilt, -data.sinTilt);
    if (data.dx || data.dy) {
      x += data.dx;
      y += data.dy;
    }
  } else {
    return 0;
  }

  if (writeOutput) {
    if (data.type == GPU_TAPER_APERTURE_CIRCULAR) {
      double length = lost ? zLost : data.length;
      x += length * xp;
      y += length * yp;
    }
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    if (lost) {
      part[4] = data.zStart + zLost;
      part[5] = data.pCentral * (1 + part[5]);
      if (data.type == GPU_TAPER_APERTURE_ELLIPTICAL && data.localLossColumn >= 0) {
        part[data.localLossColumn] = xLocal;
        part[data.localLossColumn + 1] = xpLocal;
        part[data.localLossColumn + 2] = yLocal;
      }
    } else if (sReady) {
      part[4] = sDrifted;
    } else {
      part[4] += data.length * sqrt(1 + xp * xp + yp * yp);
    }
  }
  return !lost;
}

__global__ void gpuTaperApertureSurvivorFlagKernel(
  double *coord, long nParticles, int stride,
  GPU_TAPER_APERTURE_DATA data, long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip < nParticles)
    survivorPrefix[ip] = gpuTaperApertureTrackParticle(
      coord + ip * stride, 0, data) ? 1 : 0;
}

__global__ void gpuTaperApertureStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors,
  GPU_TAPER_APERTURE_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip >= nParticles)
    return;
  long destination = survivorPrefix[ip];
  double *part = coord + ip * stride;
  double *target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuTaperApertureTrackParticle(target, 1, data);
}

__device__ int gpuSpeedbumpTrackParticle(
  double *part, int writeOutput, const GPU_SPEEDBUMP_DATA &data) {
  double xiHit = DBL_MAX, yiHit = DBL_MAX;
  int phit = 0;

  for (int idir = 0; idir < 2; idir++) {
    int directionEnabled = idir ? data.minusDirection : data.plusDirection;
    int dsign = idir ? -1 : 1;
    double position, xi = 0, yi = 0;
    double dxPlane = 2 * data.length;
    int hit = 0;

    if (!directionEnabled)
      continue;
    position = data.scraperConvention ?
      dsign * data.position : data.position;
    if (dsign * part[data.plane] >=
        position + data.height + data.offset * dsign) {
      dxPlane = 0;
      xi = 0;
      yi = part[data.plane];
      hit = 1;
    } else if (part[data.plane + 1]) {
      dxPlane =
        (dsign * (position + data.height + data.offset * dsign) -
         part[data.plane]) /
        part[data.plane + 1];
      if (dxPlane >= 0 &&
          dxPlane <=
            data.length / 2 + data.dzCenter - data.chord / 2) {
        hit = 1;
        xi = dxPlane;
        yi = (position + data.height) * dsign + data.offset;
      }
    }
    if (!hit && data.radius) {
      double x1 = -(data.length / 2 + data.dzCenter);
      double dx = data.length;
      double x2 = x1 + dx;
      double y1 =
        position + data.radius + data.offset * dsign -
        dsign * part[data.plane];
      double dy = -data.length * dsign * part[data.plane + 1];
      double y2 = y1 + dy;
      double D = x1 * y2 - x2 * y1;
      /* Preserve the CPU circle-line intersection's rounding sequence. */
      double dr = sqrt(dx * dx + dy * dy);
      double dr2 = dr * dr;
      double radiusDr = data.radius * dr;
      double disc = radiusDr * radiusDr - D * D;

      if (disc >= 0) {
        if (disc == 0) {
          xi = D * dy / dr2;
          yi = -D * dx / dr2;
        } else {
          double root = sqrt(disc);
          double dySign = dy < 0 ? -1 : (dy > 0 ? 1 : 0);
          double xi1 = (D * dy + dySign * dx * root) / dr2;
          double yi1 = (-D * dx + fabs(dy) * root) / dr2;
          double xi2 = (D * dy - dySign * dx * root) / dr2;
          double yi2 = (-D * dx - fabs(dy) * root) / dr2;
          if (xi1 < xi2) {
            xi = xi1;
            yi = yi1;
          } else {
            xi = xi2;
            yi = yi2;
          }
        }
        hit = 1;
        xi += data.length / 2 + data.dzCenter;
        yi = (yi - (position + data.radius + data.offset * dsign)) /
             (-dsign);
      }
      if (!hit &&
          dxPlane >= data.length / 2 + data.dzCenter &&
          dxPlane <= data.length) {
        xi = dxPlane;
        yi = (position + data.height) * dsign + data.offset;
        hit = 1;
      }
    }
    if (hit && xi < xiHit) {
      xiHit = xi;
      yiHit = yi;
      phit = 1;
    }
  }

  if (writeOutput) {
    if (!phit) {
      gpuExactDriftParticle(part, data.length);
    } else {
      int otherPlane = data.plane == 0 ? 2 : 0;
      part[data.plane] = yiHit;
      part[otherPlane] += xiHit * part[otherPlane + 1];
      part[4] = data.zStart + xiHit;
      part[5] = data.pCentral * (1 + part[5]);
    }
  }
  return !phit;
}

__global__ void gpuSpeedbumpSurvivorFlagKernel(
  double *coord, long nParticles, int stride,
  GPU_SPEEDBUMP_DATA data, long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip < nParticles)
    survivorPrefix[ip] = gpuSpeedbumpTrackParticle(
      coord + ip * stride, 0, data) ? 1 : 0;
}

__global__ void gpuSpeedbumpStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors,
  GPU_SPEEDBUMP_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip >= nParticles)
    return;
  long destination = survivorPrefix[ip];
  double *part = coord + ip * stride;
  double *target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuSpeedbumpTrackParticle(target, 1, data);
}

__device__ __forceinline__ int gpuCcbendSwitchPlane(
  double *x, double *xp, double *y, double *yp, double *s,
  double dp, double tanAngle, double cosAngle, double sinAngle) {
  double denominator = 1 - *xp * tanAngle;
  double ds, norm, qx0, qy0, qz0, qx, qy, qz;

  if (denominator == 0 || cosAngle == 0)
    return 0;
  ds = *x * tanAngle / denominator;
  *x = (*x + *xp * ds) / cosAngle;
  *y = *y + *yp * ds;
  *s += ds;
  norm = sqrt(1 + *xp * *xp + *yp * *yp);
  qx0 = *xp * (1 + dp) / norm;
  qy0 = *yp * (1 + dp) / norm;
  qz0 = (1 + dp) / norm;
  qx = qx0 * cosAngle + qz0 * sinAngle;
  qy = qy0;
  qz = -qx0 * sinAngle + qz0 * cosAngle;
  if (qz == 0)
    return 0;
  *xp = qx / qz;
  *yp = qy / qz;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) &&
         isfinite(*yp) && isfinite(*s);
}

/* Same as checkApContour(): the particle is at zero slope and the contour has no length */
__device__ __noinline__ int gpuInsideApertureContour(
  double xInput, double yInput, const GPU_APERTURE_LIMIT_DATA &aperture,
  const GPU_APERTURE_CONTOUR_DATA &contour) {
  double x0 = xInput - aperture.xCen, y0 = yInput - aperture.yCen, x, y;
  int keep = contour.invert ? 0 : 1;
  /* A stack of truth values, top bit first, for a contour file's Logic expressions */
  unsigned long long stack = keep;

  if (aperture.sinReverseTilt || aperture.cosReverseTilt != 1) {
    x = x0 * aperture.cosReverseTilt + y0 * aperture.sinReverseTilt;
    y = -x0 * aperture.sinReverseTilt + y0 * aperture.cosReverseTilt;
    x0 = x;
    y0 = y;
  }
  x0 -= contour.dx;
  y0 -= contour.dy;
  x = (x0 * contour.cosTilt + y0 * contour.sinTilt) / contour.xFactor;
  y = (-x0 * contour.sinTilt + y0 * contour.cosTilt) / contour.yFactor;
  for (long ic = 0; ic < contour.nContours; ic++) {
    const double *px = contour.x + contour.start[ic];
    const double *py = contour.y + contour.start[ic];
    long n = contour.start[ic + 1] - contour.start[ic];
    long winding = 0;
    int inside, keepThis;

    for (long i1 = 0; i1 < n; i1++) {
      long i2 = i1 == n - 1 ? 0 : i1 + 1;
      double side = (px[i2] - px[i1]) * (y - py[i1]) - (x - px[i1]) * (py[i2] - py[i1]);

      if (py[i1] <= y) {
        if (py[i2] > y && side > 0)
          ++winding;
      } else if (py[i2] <= y && side < 0) {
        --winding;
      }
    }
    inside = winding != 0;
    keepThis = inside != contour.invert;
    if (contour.logic) {
      stack = (stack << 1) | (keepThis ? 1 : 0);
      for (long op = contour.logicStart[ic]; op < contour.logicStart[ic + 1]; op++) {
        if (contour.logic[op] == GPU_CONTOUR_LOGIC_NOT) {
          stack ^= 1;
        } else if (contour.logic[op] == GPU_CONTOUR_LOGIC_POP) {
          stack >>= 1;
        } else {
          unsigned long long a = stack & 1, b = (stack >> 1) & 1;

          stack >>= 1;
          stack = (stack & ~1ULL) | (contour.logic[op] == GPU_CONTOUR_LOGIC_AND ? (a & b) : (a | b));
        }
      }
    } else {
      keep = contour.invert ? (keep || keepThis) : (keep && keepThis);
    }
  }
  return contour.logic ? (int)(stack & 1) : keep;
}

__device__ int gpuLgbendFringe(
  double *x, double *xp, double *y, double *yp, double *path, double dp,
  double tant, double sint, double sect,
  double invRhoPlus, double K1plus,
  double invRhoMinus, double K1minus,
  const GPU_LGBEND_SEGMENT_DATA &segment, int isExit, int edgeOrder);
__device__ __forceinline__ int gpuLgbendBody(
  double *x, double *xp, double *y, double *yp, double *path, double *dp,
  const GPU_LGBEND_SEGMENT_DATA &segment, long nSlices,
  int integrationOrder, double Po, double radCoef,
  const GPU_APERTURE_LIMIT_DATA &aperture,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long segmentIndex,
  double coordLimit, double slopeLimit,
  const GPU_CCBEND_DATA *fileTerms, double apertureDx, double apertureDy,
  const GPU_BEND_OPTIONS &options, double *spin, const double *isrDraws,
  double *sliceRecord, double zOffset, int fewerRoots);

/* Same as performQuaternionRotation() */
__device__ __noinline__ void gpuSpinRotate(double *S, double rx, double ry, double rz) {
  const double tiny = 1e-300;
  double theta = sqrt(rx * rx + ry * ry + rz * rz);
  double half, s, qw, qx, qy, qz, vx, vy, vz, tx, ty, tz, snorm;

  if (theta < tiny)
    return;
  half = 0.5 * theta;
  s = sin(half) / theta;
  qw = cos(half);
  qx = rx * s;
  qy = ry * s;
  qz = rz * s;
  vx = S[0];
  vy = S[1];
  vz = S[2];
  tx = 2.0 * (qy * vz - qz * vy);
  ty = 2.0 * (qz * vx - qx * vz);
  tz = 2.0 * (qx * vy - qy * vx);
  S[0] = vx + qw * tx + (qy * tz - qz * ty);
  S[1] = vy + qw * ty + (qz * tx - qx * tz);
  S[2] = vz + qw * tz + (qx * ty - qy * tx);
  snorm = sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
  if (snorm > tiny) {
    S[0] /= snorm;
    S[1] /= snorm;
    S[2] /= snorm;
  }
}

/* The spin part of rotateBeamCoordinatesForMisalignment() */
__device__ __forceinline__ void gpuSpinRoll(double *S, double cosAngle, double sinAngle) {
  double Sx = S[0], Sy = S[1];

  S[0] = Sx * cosAngle + Sy * sinAngle;
  S[1] = -Sx * sinAngle + Sy * cosAngle;
}

/* applyFringeSpinKick() */
__device__ __noinline__ void gpuBendFringeSpin(double *S, double xp, double xp0, double yp, double yp0,
                                               double y, double dp, double Po, double rho,
                                               const GPU_BEND_OPTIONS &options) {
  double p = Po * (1 + dp);
  double gamma = sqrt(p * p + 1);

  gpuSpinRotate(S, -(1 + options.spinAnomaly * gamma) * (yp - yp0),
                +(1 + options.spinAnomaly * gamma) * (xp - xp0),
                (rho != 0) ? -(1 + options.spinAnomaly) * y / rho : 0.0);
}

/* The body step of integrate_kick_KnL() and updateSpinQuaternionLocalFS() with h=0 and Bz=0 */
__device__ __noinline__ void gpuBendBodySpin(double *S, double deltaQx, double deltaQy,
                                             double xp, double yp, double drift, long nSlices,
                                             double kickFraction, double dp, double Po,
                                             const GPU_BEND_OPTIONS &options) {
  const double tiny = 1e-300;
  double ds = (drift / nSlices) * kickFraction;
  double scale, B[3], betav[3], Omega[3], p, gamma, beta, dlds, betaDotB, coeff, kpar;

  if (ds == 0)
    return;
  scale = options.spinBrho / ds;
  B[0] = +deltaQy * scale;
  B[1] = -deltaQx * scale;
  B[2] = 0.0;
  p = Po * (1 + dp);
  gamma = sqrt(p * p + 1);
  beta = p / gamma;
  dlds = sqrt(1.0 + xp * xp + yp * yp);
  if (dlds < tiny)
    return;
  betav[0] = xp / dlds * beta;
  betav[1] = yp / dlds * beta;
  betav[2] = 1.0 / dlds * beta;
  betaDotB = betav[0] * B[0] + betav[1] * B[1] + betav[2] * B[2];
  coeff = -(options.spinCharge / (options.spinMassC * beta)) * dlds;
  kpar = options.spinAnomaly * gamma / (gamma + 1.0);
  for (int i = 0; i < 3; i++)
    Omega[i] = coeff * ((options.spinAnomaly + 1.0 / gamma) * B[i] - kpar * betaDotB * betav[i]);
  gpuSpinRotate(S, Omega[0] * ds, Omega[1] * ds, Omega[2] * ds);
}

/* One row of the centroid output: X, theta, Y, phi, Z, delta */
__device__ __noinline__ void gpuBendRecordSlice(double *record, double x, double xp, double y, double yp,
                                                double dp, double dz, const GPU_BEND_OPTIONS &options) {
  double dX = x - options.globalDxOffset;

  record[0] = options.globalX1 + dX * options.globalCos - dz * options.globalSin;
  record[1] = -options.globalTheta1 + options.globalCosExtraTilt * gpuAccurateAtan(xp);
  record[2] = y;
  record[3] = gpuAccurateAtan(yp);
  record[4] = options.globalZ1 + dX * options.globalSin + dz * options.globalCos;
  record[5] = dp;
}

/* Not inlined, because the fused interpreter inlines the bend body and shouldn't carry these arrays */
__device__ __noinline__ void gpuCcbendApplyHighKn(
  double *qx, double *qy, double *deltaQx, double *deltaQy, double x, double y,
  const GPU_CCBEND_DATA &data, double kickFraction, long nSlices);
__device__ __noinline__ void gpuCcbendApplyFileTerms(
  double *qx, double *qy, double *deltaQx, double *deltaQy, double x, double y,
  const GPU_MULTIPOLE_TERM *term, int nTerms, int maxOrder,
  double kickFraction, long nSlices) {
  double xpow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
  double ypow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];

  gpuMultipoleFillPowerArray(x, xpow, maxOrder);
  gpuMultipoleFillPowerArray(y, ypow, maxOrder);
  for (int iTerm = 0; iTerm < nTerms; iTerm++)
    gpuMultipoleApplyTerm(qx, qy, deltaQx, deltaQy, xpow, ypow, term[iTerm].order,
                          term[iTerm].strength * kickFraction / nSlices, term[iTerm].skew);
}

__device__ int gpuCcbendTrackNoFringeOrder2(
  double *part, const GPU_CCBEND_DATA &data, int writeOutput) {
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double path = part[4];
  double dp = part[5];
  double qx, qy, denominator, bodyPath = 0;
  double xpow[3], ypow[3];
  double halfDrift = data.chordLength / data.nSlices * 0.5;

  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path) || !isfinite(dp))
    return 0;
  if (data.sinTilt || data.cosTilt != 1)
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, data.sinTilt);
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.planeTan, data.planeCos, data.planeSin))
    return 0;
  x -= data.dxOffset;
  if (fabs(x) > data.coordLimit || fabs(y) > data.coordLimit ||
      fabs(xp) > data.slopeLimit || fabs(yp) > data.slopeLimit)
    return 0;

  denominator = sqrt(1 + xp * xp + yp * yp);
  qx = (1 + dp) * xp / denominator;
  qy = (1 + dp) * yp / denominator;
  denominator = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
  if (denominator <= 0)
    return 0;
  denominator = sqrt(denominator);
  xp = qx / denominator;
  yp = qy / denominator;

  for (long slice = 0; slice < data.nSlices; slice++) {
    double deltaQx = 0, deltaQy = 0;

    bodyPath += halfDrift * sqrt(1 + xp * xp + yp * yp);
    x += xp * halfDrift;
    y += yp * halfDrift;
    gpuMultipoleFillPowerArray(x, xpow, 2);
    gpuMultipoleFillPowerArray(y, ypow, 2);
    for (int order = 0; order < 3; order++) {
      if (data.KnL[order])
        gpuMultipoleApplyKick(&qx, &qy, &deltaQx, &deltaQy,
                              xpow, ypow, order,
                              data.KnL[order] / data.nSlices, 0);
    }
    denominator = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
    if (denominator <= 0)
      return 0;
    denominator = sqrt(denominator);
    xp = qx / denominator;
    yp = qy / denominator;
    bodyPath += halfDrift * sqrt(1 + xp * xp + yp * yp);
    x += xp * halfDrift;
    y += yp * halfDrift;
    if (data.momentumRoundTrip) {
      /* integrate_kick_KnL()'s radiation block, which runs with radiation off too (see the packer) */
      qx /= (1 + dp);
      qy /= (1 + dp);
      qx *= (1 + dp);
      qy *= (1 + dp);
    }
  }
  if (data.momentumRoundTrip) {
    /* the CPU takes the slopes out of the body from the momenta as that block left them */
    denominator = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
    if (denominator <= 0)
      return 0;
    denominator = sqrt(denominator);
    xp = qx / denominator;
    yp = qy / denominator;
  }

  path += bodyPath;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp) ||
      fabs(x) > data.coordLimit || fabs(y) > data.coordLimit ||
      fabs(xp) > data.slopeLimit || fabs(yp) > data.slopeLimit)
    return 0;

  x -= data.xAdjust;
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.planeTan, data.planeCos, data.planeSin))
    return 0;
  if (data.sinTilt || data.cosTilt != 1)
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, -data.sinTilt);
  if (data.referenceCorrection & 2) {
    x -= data.referenceTrajectory[0];
    xp -= data.referenceTrajectory[1];
    y -= data.referenceTrajectory[2];
    yp -= data.referenceTrajectory[3];
  }
  if (data.referenceCorrection & 1)
    path -= data.referenceTrajectory[4];
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path))
    return 0;
  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    part[4] = path;
    part[5] = dp;
  }
  return 1;
}

__device__ __noinline__ void gpuCcbendApplyHighKn(
  double *qx, double *qy, double *deltaQx, double *deltaQy, double x, double y,
  const GPU_CCBEND_DATA &data, double kickFraction, long nSlices) {
  double xpow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
  double ypow[GPU_MULTIPOLE_MAX_TERM_ORDER + 1];

  gpuMultipoleFillPowerArray(x, xpow, data.maxTermOrder);
  gpuMultipoleFillPowerArray(y, ypow, data.maxTermOrder);
  for (int i = 0; i < data.nHighKn; i++)
    gpuMultipoleApplyTerm(qx, qy, deltaQx, deltaQy, xpow, ypow, data.highKnOrder[i],
                          data.highKnL[i] / nSlices * kickFraction, 0);
}

/* Same arithmetic as offsetParticlesForMisalignment() for a CCBEND, whose design tilt is applied separately */
__device__ void gpuCcbendMisalign(double *x, double *xp, double *y, double *yp, double *path,
                                  double dp, const GPU_CCBEND_DATA &data, int face) {
  int offset = data.dx || data.dy || data.dz;

  if (data.malignMethod == 1 || data.malignMethod == 2) {
    gpuApplyLinearMisalignment(x, xp, y, yp, path, dp, &data.misalignment, face);
    return;
  }
  if (data.malignMethod >= 3) {
    gpuApplyExactMisalignment(x, xp, y, yp, path, dp, &data.exactMisalignment, face);
    return;
  }
  if (face == 0) {
    if (offset) {
      *path += data.dz * sqrt(1 + *xp * *xp + *yp * *yp);
      *x = *x - data.dx + data.dz * *xp;
      *y = *y - data.dy + data.dz * *yp;
    }
    if (data.sinEtilt || data.cosEtilt != 1)
      gpuExactCorrectorRotate(x, xp, y, yp, data.cosEtilt, data.sinEtilt);
  } else {
    if (data.sinEtilt || data.cosEtilt != 1)
      gpuExactCorrectorRotate(x, xp, y, yp, data.cosEtilt, -data.sinEtilt);
    if (offset) {
      *path += -data.dz * sqrt(1 + *xp * *xp + *yp * *yp);
      *x = *x + data.dx + -data.dz * *xp;
      *y = *y + data.dy + -data.dz * *yp;
    }
  }
}

__device__ int gpuCcbendTrackParticle(double *part, int stride,
                                      const GPU_CCBEND_DATA &data,
                                      int writeOutput, long ip) {
  double spinValues[3], *spin = NULL;
  double xp0, yp0;
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double path = part[4];
  double dp = part[5];
  GPU_LGBEND_SEGMENT_DATA segment = {};

  (void)stride;
  if (data.fringeModel == -1 && data.integrationOrder == 2 &&
      data.radCoef == 0 && !data.aperture.present && !data.options.contour.nContours &&
      !data.nEdge1Terms && !data.nEdge2Terms && !data.nBodyTerms &&
      !data.nHighKn && !data.hasYaw && !data.exitMisaligned &&
      !data.options.isrCoef && !data.options.sliceRecord && !data.options.spinOffset)
    return gpuCcbendTrackNoFringeOrder2(part, data, writeOutput);
  if (data.options.spinOffset) {
    spin = spinValues;
    for (int i = 0; i < 3; i++)
      spin[i] = part[data.options.spinOffset + i];
  }
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path) || !isfinite(dp))
    return 0;
  if (data.sinTilt || data.cosTilt != 1) {
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, data.sinTilt);
    if (spin)
      gpuSpinRoll(spin, data.cosTilt, data.sinTilt);
  }
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.planeTan, data.planeCos, data.planeSin))
    return 0;
  if (spin && data.spinAngle1)
    gpuSpinRotate(spin, 0.0, data.spinAngle1, 0.0);
  if (data.entryMisaligned) {
    gpuCcbendMisalign(&x, &xp, &y, &yp, &path, dp, data, 0);
    if (spin && data.malignMethod == 0 && (data.sinEtilt || data.cosEtilt != 1))
      gpuSpinRoll(spin, data.cosEtilt, data.sinEtilt);
  }
  x -= data.dxOffset;
  xp0 = xp;
  yp0 = yp;

  segment.length = data.chordLength;
  segment.invRho = 1 / data.rho0;
  segment.K1 = data.KnL[1] / data.chordLength;
  segment.angleSign = data.angleSign;
  for (int i = 0; i < 3; i++)
    segment.KnL[i] = data.KnL[i];
  for (int i = 0; i < 8; i++) {
    segment.fringe1[i] = data.fringeInt1[i];
    segment.fringe2[i] = data.fringeInt2[i];
  }
  if (data.fringeModel == 0 && data.edgeOrder >= 1)
    yp -= y * (data.edge1Kick[0] + data.edge1Kick[1] * x +
               data.edge1Kick[2] * (x * x - y * y)) / (1 + dp);
  if (data.fringeModel == 1 &&
      !gpuLgbendFringe(
        &x, &xp, &y, &yp, &path, dp,
        data.fringe1Tan, data.fringe1Sin, data.fringe1Sec,
        segment.invRho, segment.K1, 0, 0,
        segment, 0, data.edgeOrder))
      return 0;
  if (spin && data.fringeModel != -1 && !(data.fringeModel == 0 && data.edgeOrder < 1))
    gpuBendFringeSpin(spin, xp, xp0, yp, yp0, y, dp, data.Po, data.rho0 * data.angleSign, data.options);
  if (!gpuLgbendBody(
        &x, &xp, &y, &yp, &path, &dp, segment, data.nSlices,
        data.integrationOrder, data.Po, data.radCoef, data.aperture,
        NULL, -1,
        data.coordLimit, data.slopeLimit, &data, data.dx, data.dy, data.options, spin,
        data.options.isrCoef ? data.options.isrDraws + ip * data.nSlices : NULL,
        data.options.sliceRecord ?
          data.options.sliceRecord + ip * (data.nSlices + 1) * GPU_BEND_RECORD_SIZE : NULL,
        0.0, 0))
    return 0;
  xp0 = xp;
  yp0 = yp;
  if (data.fringeModel == 0 && data.edgeOrder >= 1)
    yp -= y * (data.edge2Kick[0] + data.edge2Kick[1] * x +
               data.edge2Kick[2] * (x * x - y * y)) / (1 + dp);
  if (data.fringeModel == 1 &&
      !gpuLgbendFringe(
        &x, &xp, &y, &yp, &path, dp,
        data.fringe2Tan, data.fringe2Sin, data.fringe2Sec,
        0, 0, segment.invRho, segment.K1,
        segment, 1, data.edgeOrder))
      return 0;
  if (spin && data.fringeModel != -1 && !(data.fringeModel == 0 && data.edgeOrder < 1))
    gpuBendFringeSpin(spin, xp, xp0, yp, yp0, y, dp, data.Po, data.rho0 * data.angleSign, data.options);

  x -= data.xAdjust;
  if (data.exitMisaligned) {
    if (spin && data.malignMethod == 0 && (data.sinEtilt || data.cosEtilt != 1))
      gpuSpinRoll(spin, data.cosEtilt, -data.sinEtilt);
    gpuCcbendMisalign(&x, &xp, &y, &yp, &path, dp, data, 1);
  }
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.exitPlaneTan, data.exitPlaneCos, data.exitPlaneSin))
    return 0;
  if (spin && data.spinAngle2)
    gpuSpinRotate(spin, 0.0, data.spinAngle2, 0.0);
  if (data.sinTilt || data.cosTilt != 1) {
    gpuExactCorrectorRotate(&x, &xp, &y, &yp,
                            data.cosTilt, -data.sinTilt);
    if (spin)
      gpuSpinRoll(spin, data.cosTilt, -data.sinTilt);
  }
  if (data.referenceCorrection & 2) {
    x -= data.referenceTrajectory[0];
    xp -= data.referenceTrajectory[1];
    y -= data.referenceTrajectory[2];
    yp -= data.referenceTrajectory[3];
  }
  if (data.referenceCorrection & 1)
    path -= data.referenceTrajectory[4];
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path))
    return 0;
  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    part[4] = path;
    part[5] = dp;
    if (spin)
      for (int i = 0; i < 3; i++)
        part[data.options.spinOffset + i] = spin[i];
  }
  return 1;
}

__global__ void gpuCcbendTrackCheckedKernel(
  double *coord, long nParticles, int stride, GPU_CCBEND_DATA data,
  unsigned long long *lostCount, long *survivorFlag) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles &&
      !gpuCcbendTrackParticle(coord + ip * stride, stride, data, 1, ip))
    localCount = 1;
  if (survivorFlag && ip < nParticles)
    survivorFlag[ip] = localCount ? 0 : 1;
  partial[threadIdx.x] = localCount;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

__global__ void gpuCcbendSurvivorFlagKernel(
  double *coord, long nParticles, int stride, GPU_CCBEND_DATA data,
  long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuCcbendTrackParticle(coord + ip * stride, stride, data, 0, ip) ? 1 : 0;
}

__global__ void gpuCcbendStableTrackScatterKernel(
  double *coord, double *scratch, const long *destination,
  long nParticles, int stride, GPU_CCBEND_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  target = scratch + destination[ip] * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuCcbendTrackParticle(target, stride, data, 1, ip);
}

__device__ __forceinline__ int gpuLgbendSwitchPlane(
  double *x, double *xp, double *y, double *yp, double *path,
  double dp, double dx, double tanAngle, double cosAngle, double sinAngle,
  int exitPlane) {
  double denominator, ds, norm, qx0, qy0, qz0, qx, qy, qz;

  if (exitPlane)
    *x += dx;
  denominator = 1 - *xp * tanAngle;
  if (denominator == 0 || cosAngle == 0)
    return 0;
  ds = *x * tanAngle / denominator;
  *x = (*x + *xp * ds) / cosAngle;
  *y = *y + *yp * ds;
  *path += ds;
  norm = sqrt(1 + *xp * *xp + *yp * *yp);
  qx0 = *xp * (1 + dp) / norm;
  qy0 = *yp * (1 + dp) / norm;
  qz0 = (1 + dp) / norm;
  qx = qx0 * cosAngle + qz0 * sinAngle;
  qy = qy0;
  qz = -qx0 * sinAngle + qz0 * cosAngle;
  if (qz == 0)
    return 0;
  *xp = qx / qz;
  *yp = qy / qz;
  if (!exitPlane)
    *x += dx;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) &&
         isfinite(*yp) && isfinite(*path);
}

__device__ int gpuLgbendFringe(
  double *x, double *xp, double *y, double *yp, double *path, double dp,
  double tant, double sint, double sect,
  double invRhoPlus, double K1plus,
  double invRhoMinus, double K1minus,
  const GPU_LGBEND_SEGMENT_DATA &segment, int isExit, int edgeOrder) {
  const double *fringe = isExit ? segment.fringe2 : segment.fringe1;
  double intK0 = fringe[0] * segment.angleSign;
  double intI0 = fringe[1];
  double intK2 = fringe[2];
  double intI1 = fringe[3];
  double intK4 = fringe[4] * segment.angleSign;
  double intK5 = fringe[5] * segment.angleSign;
  double intK6 = fringe[6] * segment.angleSign;
  double intK7 = fringe[7];
  double x1 = *x, y1 = *y, px1, py1, tau1;
  double x2, y2, px2, py2, tau2;
  double sect3 = sect * sect * sect;
  double invP = 1.0 / (1.0 + dp);
  double focX0 =
    -tant * intK5 - 0.5 * intI0 * (2.0 - tant * tant);
  double focY0 =
    -tant * (invRhoPlus - invRhoMinus) +
    tant * sect * sect * intK5 +
    0.5 * intI0 * (2.0 + tant * tant);
  double focYd =
    sect3 * ((1.0 + sint * sint) * intK2 - intK7);
  double temp = sqrt(1.0 + *xp * *xp + *yp * *yp);
  double dispX, kickPx, expT;

  px1 = (1.0 + dp) * *xp / temp;
  py1 = (1.0 + dp) * *yp / temp;
  tau1 = -*path + sint * x1;
  px1 -= (1.0 + dp) * sint;

  temp = sect * intI1 * invP;
  if (fabs(temp) < 1.0e-20)
    temp = 1.0e-20;
  dispX = -sect * intK0 * invP;
  kickPx = -tant * (intI1 + 0.5 * intK4);
  expT = exp(temp);

  x2 = expT * x1 + dispX * (expT - 1.0) / temp;
  y2 = y1 / expT;
  px2 =
    px1 / expT +
    kickPx * (1.0 - 1.0 / expT) / temp +
    focX0 *
      (x1 * (expT - 1.0 / expT) / (2.0 * temp) +
       dispX * (expT + 1.0 / expT - 2.0) /
         (2.0 * temp * temp));
  py2 =
    expT * py1 +
    y1 * (focY0 + focYd * invP) *
      (expT - 1.0 / expT) / (2.0 * temp);
  tau2 =
    tau1 +
    invP *
      (kickPx * (temp * x1 + dispX) * (1.0 + temp - expT) /
         (temp * temp) -
       px1 * dispX - (px1 * x1 - py1 * y1) * temp +
       (0.5 * focYd * invP +
        focY0 * (1.0 / (expT * expT) - 1.0 + 2.0 * temp) /
          (4.0 * temp)) *
         y1 * y1 -
       x1 * x1 * focX0 *
         (expT * expT - 1.0 - 2.0 * temp) / (4.0 * temp) -
       x1 * dispX * focX0 *
         (expT * expT - 2.0 * expT + 1.0) /
         (2.0 * temp * temp) -
       dispX * dispX * focX0 *
         (expT * expT - 4.0 * expT + 2.0 * temp + 3.0) /
         (4.0 * temp * temp * temp));

  if (edgeOrder >= 2) {
    x1 = x2;
    y1 = y2 * exp(-sect * intK5 * x2 * invP);
    px1 =
      px2 -
      (0.5 * intK6 + 0.25 * tant * (K1plus - K1minus)) *
        x2 * x2 +
      sect * intK5 * invP * py2 * y2;
    py1 = py2 * exp(sect * intK5 * x2 * invP);
    tau1 = tau2 + sect * intK5 * invP * invP * py2 * x2 * y2;

    x2 =
      x1 -
      0.5 * sect3 *
        (intK5 - (invRhoPlus - invRhoMinus)) *
        invP * y1 * y1;
    y2 = y1;
    px2 =
      px1 +
      (0.5 * sect * sect * intK6 -
       0.25 * tant * (K1plus - K1minus)) *
        y1 * y1;
    py2 =
      py1 +
      sect3 * (intK5 - (invRhoPlus - invRhoMinus)) *
        invP * px1 * y1 +
      (sect * sect * intK6 -
       0.5 * tant * (K1plus - K1minus)) *
        x1 * y1;
    tau2 =
      tau1 +
      0.5 * sect3 *
        (intK5 - (invRhoPlus - invRhoMinus)) *
        invP * invP * y1 * y1 *
        (px1 +
         (0.25 * sect * sect * intK6 -
          0.125 * tant * (K1plus - K1minus)) *
           y1 * y1);

    temp = -0.5 * sect3 * intK5 * invP * x2;
    tau2 += temp * invP * px2 * x2;
    temp = 1.0 + temp;
    if (temp == 0)
      return 0;
    x2 /= temp;
    /* in the CPU's order: (px2 * temp) * temp */
    px2 = px2 * temp * temp;
  }

  px2 += (1.0 + dp) * sint;
  temp = (1.0 + dp) * (1.0 + dp) - px2 * px2 - py2 * py2;
  if (temp <= 0)
    return 0;
  temp = sqrt(temp);
  *x = x2;
  *xp = px2 / temp;
  *y = y2;
  *yp = py2 / temp;
  *path = -tau2 + sint * x2;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) &&
         isfinite(*yp) && isfinite(*path);
}

__device__ __forceinline__ int gpuLoadSymplecticFractions(
  int integrationOrder, double *driftFrac, double *kickFrac,
  int *nSubsteps);

/* checkMultAperture() takes the local aperture after the offset and the reverse tilt */
__device__ __forceinline__ int gpuInsideLgbendLocalAperture(
  double xInput, double yInput, const GPU_APERTURE_LIMIT_DATA &aperture,
  const GPU_LGBEND_LOCAL_APERTURE_POINT &point) {
  double x0 = xInput - aperture.xCen, y0 = yInput - aperture.yCen;
  double x = x0, y = y0;

  if (aperture.sinReverseTilt || aperture.cosReverseTilt != 1) {
    x = x0 * aperture.cosReverseTilt + y0 * aperture.sinReverseTilt;
    y = -x0 * aperture.sinReverseTilt + y0 * aperture.cosReverseTilt;
  }
  return !(fabs(x - point.xCenter) >= point.xMax ||
           fabs(y - point.yCenter) >= point.yMax);
}

__device__ __forceinline__ int gpuLgbendBody(
  double *x, double *xp, double *y, double *yp, double *path, double *dp,
  const GPU_LGBEND_SEGMENT_DATA &segment, long nSlices,
  int integrationOrder, double Po, double radCoef,
  const GPU_APERTURE_LIMIT_DATA &aperture,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long segmentIndex,
  double coordLimit, double slopeLimit,
  const GPU_CCBEND_DATA *fileTerms, double apertureDx, double apertureDy,
  const GPU_BEND_OPTIONS &options, double *spin, const double *isrDraws,
  double *sliceRecord, double zOffset, int fewerRoots) {
  double driftFrac[8], kickFrac[8];
  double isrCoef = options.isrCoef;
  double inputDp = *dp;
  double qx, qy, denominator, bodyPath = 0;
  double inputPath = *path;
  double beta0 = 1;
  double drift = segment.length / nSlices;
  /* the kick strengths for one slice, which the loops below scale by each substep's fraction */
  double strength0 = segment.KnL[0] / nSlices;
  double strength1 = segment.KnL[1] / nSlices;
  double strength2 = segment.KnL[2] / nSlices;
  /* With fewerRoots (LGBEND), pathFactor is sqrt(1 + xp^2 + yp^2) for the current slopes.  The slopes are
   * q/pz with pz^2 = (1+dp)^2 - qx^2 - qy^2, so this is (1+dp)/pz, and one square root and one
   * reciprocal serve each substep where the CPU's form, kept for CCBEND, takes two square roots and two
   * divisions.  An LGBEND for 64,000 particles went from 18.4 ms to 10.6 ms, and its difference from
   * the CPU stays a unit or two in the last place of s. */
  double pathFactor = 0, inverse;
  double radiationScale = 0;
  int nSubsteps = 0;

  if (!gpuLoadSymplecticFractions(integrationOrder, driftFrac,
                                  kickFrac, &nSubsteps))
    return 0;
  if (fabs(*x) > coordLimit || fabs(*y) > coordLimit ||
      fabs(*xp) > slopeLimit || fabs(*yp) > slopeLimit)
    return 0;
  if (radCoef) {
    double p = Po * (1 + *dp);
    beta0 = p / sqrt(p * p + 1);
    if (beta0 == 0)
      return 0;
  }
  denominator = sqrt(1 + *xp * *xp + *yp * *yp);
  qx = (1 + *dp) * *xp / denominator;
  qy = (1 + *dp) * *yp / denominator;
  if (fileTerms && fileTerms->nEdge1Terms) {
    double unusedQx = 0, unusedQy = 0;
    gpuCcbendApplyFileTerms(&qx, &qy, &unusedQx, &unusedQy, *x, *y,
                            fileTerms->term, fileTerms->nEdge1Terms,
                            fileTerms->maxTermOrder, 1.0, 1);
  }
  denominator = (1 + *dp) * (1 + *dp) - qx * qx - qy * qy;
  if (denominator <= 0)
    return 0;
  if (fewerRoots) {
    inverse = 1 / sqrt(denominator);
    *xp = qx * inverse;
    *yp = qy * inverse;
    pathFactor = (1 + *dp) * inverse;
    if ((radCoef || isrCoef) && drift && nSubsteps >= 2)
      radiationScale = 1 / (kickFrac[nSubsteps - 2] * drift);
  } else {
    denominator = sqrt(denominator);
    *xp = qx / denominator;
    *yp = qy / denominator;
  }

  for (long slice = 0; slice < nSlices; slice++) {
    double deltaQx = 0, deltaQy = 0;

    if (sliceRecord)
      gpuBendRecordSlice(sliceRecord + slice * GPU_BEND_RECORD_SIZE, *x, *xp, *y, *yp, inputDp,
                         zOffset + slice * drift, options);
    if (localAperture && localAperture->present &&
        !gpuInsideLgbendLocalAperture(*x + apertureDx, *y + apertureDy, aperture,
                                      localAperture->point[segmentIndex][slice]))
      return 0;
    if (!gpuInsideApertureLimit(*x + apertureDx, *y + apertureDy, aperture))
      return 0;
    if (options.contour.nContours &&
        !gpuInsideApertureContour(*x + apertureDx, *y + apertureDy, aperture, options.contour))
      return 0;
    for (int step = 0; step < nSubsteps; step++) {
      double dsh = drift * driftFrac[step];

      bodyPath += dsh * (fewerRoots ? pathFactor : sqrt(1 + *xp * *xp + *yp * *yp));
      *x += *xp * dsh;
      *y += *yp * dsh;
      if (step == nSubsteps - 1)
        break;

      /* LGBEND is limited to normal orders 0--2.  Expanding these kicks
       * directly avoids the generic power arrays in the fused interpreter. */
      deltaQx = deltaQy = 0;
      if (segment.KnL[0]) {
        double strength = strength0 * kickFrac[step];
        qx -= strength;
        deltaQx -= strength;
      }
      if (segment.KnL[1]) {
        double strength = strength1 * kickFrac[step];
        double sumFy = *x;
        double sumFx = *y;
        qx -= strength * sumFy;
        qy += strength * sumFx;
        deltaQx -= strength * sumFy;
        deltaQy += strength * sumFx;
      }
      if (segment.KnL[2]) {
        double strength = strength2 * kickFrac[step];
        double x2 = *x * *x;
        double y2 = *y * *y;
        double sumFy = 0.5 * x2;
        double sumFx = *x * *y;
        sumFy += -0.5 * y2;
        qx -= strength * sumFy;
        qy += strength * sumFx;
        deltaQx -= strength * sumFy;
        deltaQy += strength * sumFx;
      }
      if (fileTerms && fileTerms->nHighKn)
        gpuCcbendApplyHighKn(&qx, &qy, &deltaQx, &deltaQy, *x, *y, *fileTerms,
                             kickFrac[step], nSlices);
      if (fileTerms && fileTerms->nBodyTerms)
        gpuCcbendApplyFileTerms(&qx, &qy, &deltaQx, &deltaQy, *x, *y,
                                fileTerms->term + fileTerms->nEdge1Terms + fileTerms->nEdge2Terms,
                                fileTerms->nBodyTerms, fileTerms->maxTermOrder,
                                kickFrac[step], nSlices);
      denominator =
        (1 + *dp) * (1 + *dp) - qx * qx - qy * qy;
      if (denominator <= 0)
        return 0;
      if (fewerRoots) {
        inverse = 1 / sqrt(denominator);
        *xp = qx * inverse;
        *yp = qy * inverse;
        pathFactor = (1 + *dp) * inverse;
      } else {
        denominator = sqrt(denominator);
        *xp = qx / denominator;
        *yp = qy / denominator;
      }
      if (spin)
        gpuBendBodySpin(spin, deltaQx, deltaQy, *xp, *yp, drift, nSlices, kickFrac[step], *dp, Po, options);
    }

    if ((radCoef || isrCoef) && drift && fewerRoots) {
      /* the same radiation step with one division, and two square roots for the 3/4 power */
      double onePlusDp = 1 + *dp;
      double deltaFactor = onePlusDp * onePlusDp;
      double fx = deltaQx * radiationScale, fy = deltaQy * radiationScale;
      double F2 = fx * fx + fy * fy;
      double dsFactor = pathFactor * drift;

      if (radCoef)
        *dp -= radCoef * deltaFactor * F2 * dsFactor;
      if (isrCoef)
        *dp -= isrCoef * deltaFactor * sqrt(F2 * sqrt(F2)) * sqrt(dsFactor) * isrDraws[slice];
      onePlusDp = (1 + *dp) / onePlusDp;
      qx *= onePlusDp;
      qy *= onePlusDp;
    } else if ((radCoef || isrCoef) && drift) {
      double deltaFactor = (1 + *dp) * (1 + *dp);
      double normalizedDeltaQx =
        deltaQx / kickFrac[nSubsteps - 2];
      double normalizedDeltaQy =
        deltaQy / kickFrac[nSubsteps - 2];
      double F2 =
        (normalizedDeltaQx / drift) * (normalizedDeltaQx / drift) +
        (normalizedDeltaQy / drift) * (normalizedDeltaQy / drift);
      double dsFactor =
        sqrt(1 + *xp * *xp + *yp * *yp) * drift;

      qx /= (1 + *dp);
      qy /= (1 + *dp);
      if (radCoef)
        *dp -= radCoef * deltaFactor * F2 * dsFactor;
      if (isrCoef)
        *dp -= isrCoef * deltaFactor * pow(F2, 0.75) * sqrt(dsFactor) * isrDraws[slice];
      qx *= (1 + *dp);
      qy *= (1 + *dp);
    } else if (!fewerRoots && fileTerms && fileTerms->momentumRoundTrip && drift) {
      /* CCBEND without radiation: the CPU's radiation block still runs (see the packer) */
      qx /= (1 + *dp);
      qy /= (1 + *dp);
      qx *= (1 + *dp);
      qy *= (1 + *dp);
    }
  }

  if (localAperture && localAperture->present &&
      !gpuInsideLgbendLocalAperture(*x + apertureDx, *y + apertureDy, aperture,
                                    localAperture->point[segmentIndex][nSlices]))
    return 0;
  if (!gpuInsideApertureLimit(*x + apertureDx, *y + apertureDy, aperture))
    return 0;
  if (options.contour.nContours &&
      !gpuInsideApertureContour(*x + apertureDx, *y + apertureDy, aperture, options.contour))
    return 0;
  if (fileTerms && fileTerms->nEdge2Terms) {
    double unusedQx = 0, unusedQy = 0;
    gpuCcbendApplyFileTerms(&qx, &qy, &unusedQx, &unusedQy, *x, *y,
                            fileTerms->term + fileTerms->nEdge1Terms, fileTerms->nEdge2Terms,
                            fileTerms->maxTermOrder, 1.0, 1);
  }
  denominator = (1 + *dp) * (1 + *dp) - qx * qx - qy * qy;
  if (denominator <= 0)
    return 0;
  denominator = sqrt(denominator);
  *xp = qx / denominator;
  *yp = qy / denominator;
  if (sliceRecord)
    gpuBendRecordSlice(sliceRecord + nSlices * GPU_BEND_RECORD_SIZE, *x, *xp, *y, *yp, *dp,
                       zOffset + nSlices * drift, options);
  if (radCoef) {
    double p = Po * (1 + *dp);
    double beta1 = p / sqrt(p * p + 1);
    *path = beta1 *
      (inputPath / beta0 + 2 * bodyPath / (beta0 + beta1));
  } else {
    *path = inputPath + bodyPath;
  }
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) &&
         isfinite(*yp) && isfinite(*path) &&
         isfinite(*dp) &&
         fabs(*x) <= coordLimit && fabs(*y) <= coordLimit &&
         fabs(*xp) <= slopeLimit && fabs(*yp) <= slopeLimit;
}

/* Where a particle is lost, as the round of track_through_lgbend_impl(), which removes losses
 * segment by segment.  The CPU loses them only in integrate_kick_KnL(); elsewhere it would carry a
 * coordinate that isn't finite, so the nearest segment stands in */
__device__ __forceinline__ int gpuLgbendLost(int *lossRound, long round) {
  if (lossRound)
    *lossRound = (int)round;
  return 0;
}

__device__ int gpuLgbendTrackParticle(
  double *part, int stride, const GPU_LGBEND_DATA &data,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  int writeOutput, long ip = 0, int *lossRound = NULL) {
  double x = part[0], xp = part[1], y = part[2], yp = part[3];
  double path = part[4], dp = part[5];
  double spinValues[3], *spin = NULL;

  (void)stride;
  if (data.options.spinOffset) {
    spin = spinValues;
    for (int i = 0; i < 3; i++)
      spin[i] = part[data.options.spinOffset + i];
  }
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path) || !isfinite(dp))
    return gpuLgbendLost(lossRound, 0);
  if (data.firstSegment == 0) {
    if (data.predrift) {
      x += xp * data.predrift;
      y += yp * data.predrift;
      path += data.predrift * sqrt(1 + xp * xp + yp * yp);
    }
    if (data.sinTilt || data.cosTilt != 1) {
      gpuExactCorrectorRotate(&x, &xp, &y, &yp, data.cosTilt, data.sinTilt);
      if (spin)
        gpuSpinRoll(spin, data.cosTilt, data.sinTilt);
    }
    if (!gpuLgbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                               data.entryPosition,
                               data.entryPlaneTan, data.entryPlaneCos,
                               data.entryPlaneSin, 0))
      return gpuLgbendLost(lossRound, 0);
    if (spin && data.entryAngle)
      gpuSpinRotate(spin, 0.0, data.entryAngle, 0.0);
    if (data.misaligned)
      gpuApplyExactMisalignment(&x, &xp, &y, &yp, &path, dp, &data.exactMisalignment, 0);
  }
  for (long i = data.firstSegment; i < data.lastSegment; i++) {
    double invRhoMinus = i ? data.segment[i - 1].invRho : 0.0;
    double invRhoPlus = i + 1 < data.nSegments ?
                          data.segment[i + 1].invRho : 0.0;
    double K1minus = i ? data.segment[i - 1].K1 : 0.0;
    double K1plus = i + 1 < data.nSegments ?
                      data.segment[i + 1].K1 : 0.0;
    double spinRho = data.segment[i].invRho != 0 ?
                       data.segment[i].angleSign / data.segment[i].invRho : 0.0;
    double xp0 = xp, yp0 = yp;
    long particleSegment = ip * (data.lastSegment - data.firstSegment) + i - data.firstSegment;

    if (data.segment[i].has1) {
      if (!gpuLgbendFringe(
            &x, &xp, &y, &yp, &path, dp,
            data.segment[i].fringe1Tan,
            data.segment[i].fringe1Sin,
            data.segment[i].fringe1Sec,
            data.segment[i].invRho, data.segment[i].K1,
            invRhoMinus, K1minus, data.segment[i], 0, data.edgeOrder))
        return gpuLgbendLost(lossRound, i - data.firstSegment);
      if (spin)
        gpuBendFringeSpin(spin, xp, xp0, yp, yp0, y, dp, data.Po, spinRho, data.options);
    }
    if (!gpuLgbendBody(
          &x, &xp, &y, &yp, &path, &dp, data.segment[i],
          data.nSlices, data.integrationOrder, data.Po, data.radCoef,
          localAperture && localAperture->hasFileAperture ?
            localAperture->fileAperture[i] : data.aperture,
          localAperture, i,
          data.coordLimit, data.slopeLimit, NULL, data.apertureDx, data.apertureDy,
          data.options, spin,
          data.options.isrCoef ?
            data.options.isrDraws +
              (i * data.options.isrParticles + data.options.isrFirstParticle + ip) * data.nSlices : NULL,
          data.options.sliceRecord ?
            data.options.sliceRecord + particleSegment * (data.nSlices + 1) * GPU_BEND_RECORD_SIZE : NULL,
          data.zOffset[i], 1))
      return gpuLgbendLost(lossRound, i - data.firstSegment);
    xp0 = xp;
    yp0 = yp;
    if (data.segment[i].has2) {
      if (!gpuLgbendFringe(
            &x, &xp, &y, &yp, &path, dp,
            data.segment[i].fringe2Tan,
            data.segment[i].fringe2Sin,
            data.segment[i].fringe2Sec,
            invRhoPlus, K1plus,
            data.segment[i].invRho, data.segment[i].K1,
            data.segment[i], 1, data.edgeOrder))
        return gpuLgbendLost(lossRound, i - data.firstSegment);
      if (spin)
        gpuBendFringeSpin(spin, xp, xp0, yp, yp0, y, dp, data.Po, spinRho, data.options);
    }
  }
  if (data.lastSegment == data.nSegments) {
    if (data.misaligned)
      gpuApplyExactMisalignment(&x, &xp, &y, &yp, &path, dp, &data.exactMisalignment, 1);
    if (!gpuLgbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                               -data.exitPosition,
                               data.exitPlaneTan, data.exitPlaneCos,
                               data.exitPlaneSin, 1))
      return gpuLgbendLost(lossRound, data.lastSegment - data.firstSegment);
    if (spin && data.exitAngle)
      gpuSpinRotate(spin, 0.0, -data.exitAngle, 0.0);
    if (data.postdrift) {
      x += xp * data.postdrift;
      y += yp * data.postdrift;
      path += data.postdrift * sqrt(1 + xp * xp + yp * yp);
    }
    if (data.sinTilt || data.cosTilt != 1) {
      gpuExactCorrectorRotate(&x, &xp, &y, &yp, data.cosTilt, -data.sinTilt);
      if (spin)
        gpuSpinRoll(spin, data.cosTilt, -data.sinTilt);
    }
  }
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path))
    return gpuLgbendLost(lossRound, data.lastSegment - data.firstSegment);
  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    part[4] = path;
    part[5] = dp;
    if (spin)
      for (int i = 0; i < 3; i++)
        part[data.options.spinOffset + i] = spin[i];
  }
  return 1;
}

__global__ void gpuLgbendTrackCheckedKernel(
  double *coord, long nParticles, int stride, GPU_LGBEND_DATA data,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles &&
      !gpuLgbendTrackParticle(coord + ip * stride, stride, data,
                              localAperture, 1, ip))
    localCount = 1;
  partial[threadIdx.x] = localCount;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

__global__ void gpuLgbendSurvivorFlagKernel(
  double *coord, long nParticles, int stride, GPU_LGBEND_DATA data,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long *survivorPrefix, int *lossRound) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int round = -1;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuLgbendTrackParticle(coord + ip * stride, stride, data,
                           localAperture, 0, 0, &round) ? 1 : 0;
  lossRound[ip] = survivorPrefix[ip] ? -1 : round;
}

__global__ void gpuLgbendStableTrackScatterKernel(
  double *coord, double *scratch, const long *destination,
  long nParticles, int stride, GPU_LGBEND_DATA data,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  target = scratch + destination[ip] * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuLgbendTrackParticle(target, stride, data, localAperture, 1);
}

__global__ void gpuAddCoordinateKernel(double *coord, long nParticles, int stride,
                                       int index, double value) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  coord[ip * stride + index] += value;
}

__global__ void gpuOffsetBeamKernel(double *coord, long nParticles, int stride,
                                    double dx, double dxp, double dy, double dyp,
                                    double dz, double dt, double dp, double de,
                                    double pCentral, int64_t startPID, int64_t endPID,
                                    int allParticles, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, pc, beta, gamma, t, ds;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (!allParticles && (part[6] < startPID || part[6] > endPID))
    return;

  ds = dz ? dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]) : 0;
  part[0] += dx + dz * part[1];
  part[1] += dxp;
  part[2] += dy + dz * part[3];
  part[3] += dyp;
  part[4] += ds;
  if (dt || dp || de) {
    pc = pCentral * (1 + part[5]);
    gamma = sqrt(1 + pc * pc);
    beta = pc / gamma;
    t = part[4] / (beta * cMks) + dt;
    if (dp) {
      part[5] += dp;
      pc = pCentral * (1 + part[5]);
      beta = pc / sqrt(1 + pc * pc);
    }
    if (de) {
      gamma += de * gamma;
      pc = sqrt(gamma * gamma - 1);
      beta = pc / gamma;
      part[5] = (pc - pCentral) / pCentral;
    }
    part[4] = t * beta * cMks;
  }
}

__global__ void gpuBatchedMomentumSearchKernel(
  double *coord, long nParticles, int stride, const double *searchData,
  long searchParticles, long target, long pass, long firePass,
  double *history, double *historyCount, long turns,
  double dx, double dy, double pCentral, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long id, turn;
  double *part, pc, beta, t;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  id = (long)part[6] - 1;
  if (id < 0 || id >= searchParticles ||
      (long)searchData[2 * id] != target)
    return;
  if (pass == firePass) {
    part[0] += dx;
    part[2] += dy;
    pc = pCentral * (1 + part[5]);
    beta = pc / sqrt(1 + pc * pc);
    t = part[4] / (beta * cMks);
    part[5] += searchData[2 * id + 1];
    pc = pCentral * (1 + part[5]);
    beta = pc / sqrt(1 + pc * pc);
    part[4] = t * beta * cMks;
  }
  turn = pass - firePass;
  if (!history || !historyCount || turn < 0 || turn >= turns)
    return;
  history[(id * 5 + 0) * turns + turn] = part[0];
  history[(id * 5 + 1) * turns + turn] = part[1];
  history[(id * 5 + 2) * turns + turn] = part[2];
  history[(id * 5 + 3) * turns + turn] = part[3];
  history[(id * 5 + 4) * turns + turn] = part[5];
  historyCount[id] = turn + 1;
}

__global__ void gpuSetCentralMomentumKernel(double *coord, long nParticles, int stride,
                                            double oldP, double newP) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  part[5] = ((1 + part[5]) * oldP - newP) / newP;
}

/* ramp_momentum()'s time of each particle, with beta_from_delta() written out */
__global__ void gpuParticleTimesKernel(const double *coord, long nParticles, int stride,
                                       double pCentral, double cMks, double *times) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double p;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  p = pCentral;
  p *= 1 + part[5];
  times[ip] = part[4] / (cMks * (p / sqrt(p * p + 1)));
}

/* ramp_momentum()'s change of the momentum offsets */
__global__ void gpuRamppRescaleKernel(double *coord, long nParticles, int stride,
                                      double oldP, double newP) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  part[5] = (1 + part[5]) * oldP / newP - 1;
}

__global__ void gpuMatchEnergyKernel(double *coord, long nParticles, int stride,
                                     double oldP, double averageP, int changeBeam) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, dp, dr, dPCentroid, p, t, beta;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (!changeBeam) {
    dp = (oldP - averageP) / averageP;
    dr = oldP / averageP;
    part[5] = dp + part[5] * dr;
  } else {
    dPCentroid = oldP - averageP;
    p = (1 + part[5]) * oldP;
    beta = p / sqrt(p * p + 1);
    t = part[4] / beta;
    p += dPCentroid;
    part[5] = (p - oldP) / oldP;
    part[4] = t * (p / sqrt(p * p + 1));
  }
}

__global__ void gpuMatchEnergyAndAverageKernel(double *coord, long nParticles,
                                               int stride, double oldP,
                                               int changeBeam,
                                               GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  __shared__ double error[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  double averageP;

  count[tid] = 0;
  sum[tid] = 0;
  error[tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double value = oldP * (1 + part[5]);
    double y = value - error[tid];
    double tmp = sum[tid] + y;
    error[tid] = (tmp - sum[tid]) - y;
    sum[tid] = tmp;
    count[tid]++;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      double y, tmp;
      count[tid] += count[tid + offset];
      y = sum[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
      y = error[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
    }
    __syncthreads();
  }

  if (count[0] <= 0)
    return;
  averageP = sum[0] / count[0];
  if (tid == 0) {
    result->count = count[0];
    result->centroidSum[5] = averageP;
  }
  __syncthreads();

  if (!changeBeam) {
    double dp, dr;
    if (oldP != 0 && fabs(averageP - oldP) / fabs(oldP) <= 1e-14)
      return;
    dp = (oldP - averageP) / averageP;
    dr = oldP / averageP;
    for (ip = tid; ip < nParticles; ip += blockDim.x) {
      double *part = coord + ip * stride;
      part[5] = dp + part[5] * dr;
    }
  } else {
    double dPCentroid = oldP - averageP;
    for (ip = tid; ip < nParticles; ip += blockDim.x) {
      double *part = coord + ip * stride;
      double p = (1 + part[5]) * oldP;
      double beta = p / sqrt(p * p + 1);
      double t = part[4] / beta;
      p += dPCentroid;
      part[5] = (p - oldP) / oldP;
      part[4] = t * (p / sqrt(p * p + 1));
    }
  }
}

__global__ void gpuMatchEnergyPartialKernel(
  double *coord, long nParticles, int stride, double oldP,
  GPU_MATCH_ENERGY_PARTIAL *partial) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  __shared__ double error[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  sum[tid] = 0;
  error[tid] = 0;
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    double *part = coord + ip * stride;
    double value = oldP * (1 + part[5]);
    double y = value - error[tid];
    double tmp = sum[tid] + y;
    error[tid] = (tmp - sum[tid]) - y;
    sum[tid] = tmp;
    count[tid]++;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      double y, tmp;
      count[tid] += count[tid + offset];
      y = sum[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
      y = error[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x].count = count[0];
    partial[blockIdx.x].sum = sum[0];
    partial[blockIdx.x].error = error[0];
  }
}

__global__ void gpuMatchEnergyFinalizeKernel(
  const GPU_MATCH_ENERGY_PARTIAL *partial, int blocks,
  GPU_BEAM_SUM_DATA *result, double *averageP) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  __shared__ double error[GPU_REDUCTION_THREADS];
  int tid = threadIdx.x;

  count[tid] = 0;
  sum[tid] = 0;
  error[tid] = 0;
  if (tid < blocks) {
    count[tid] = partial[tid].count;
    sum[tid] = partial[tid].sum;
    error[tid] = partial[tid].error;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      double y, tmp;
      count[tid] += count[tid + offset];
      y = sum[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
      y = error[tid + offset] - error[tid];
      tmp = sum[tid] + y;
      error[tid] = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
    }
    __syncthreads();
  }

  if (tid == 0) {
    double average = count[0] ? sum[0] / count[0] : 0;
    result->count = count[0];
    result->centroidSum[5] = average;
    *averageP = average;
  }
}

__global__ void gpuMatchEnergyApplyKernel(double *coord, long nParticles,
                                          int stride, double oldP,
                                          const double *averageP,
                                          int changeBeam) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double average;

  if (ip >= nParticles)
    return;
  average = *averageP;
  if (!changeBeam) {
    double dp, dr;
    if (oldP != 0 && fabs(average - oldP) / fabs(oldP) <= 1e-14)
      return;
    dp = (oldP - average) / average;
    dr = oldP / average;
    coord[ip * stride + 5] = dp + coord[ip * stride + 5] * dr;
  } else {
    double *part = coord + ip * stride;
    double dPCentroid = oldP - average;
    double p = (1 + part[5]) * oldP;
    double beta = p / sqrt(p * p + 1);
    double t = part[4] / beta;
    p += dPCentroid;
    part[5] = (p - oldP) / oldP;
    part[4] = t * (p / sqrt(p * p + 1));
  }
}

__global__ void gpuCoordinateColumnKernel(const double *coord, long nParticles,
                                          int stride, int column, double *values) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    values[ip] = coord[ip * stride + column];
}

/* Keep phase construction and tracking on CUDA; reference-sensitive output
 * may evaluate the sine with the same host math library as CPU tracking.
 * The phase is simple_rfca.c's, at the middle of a section of the given
 * length, and the matrix method also needs each particle's momentum. */
__global__ void gpuRfcaPhaseKernel(const double *coord, double *phaseValue,
  double *momentum, long nParticles, int stride, double pCentral,
  double length, double omega, double phase, double timeShift, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip >= nParticles)
    return;
  const double *part = coord + ip * stride;
  double p = pCentral * (1 + part[5]);
  double beta = p / sqrt(p * p + 1);
  double ds = length ? length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]) : 0;
  double t = (part[4] + ds) / (cMks * beta);
  phaseValue[ip] = part[5] == -1 ? 0 : omega * (t - timeShift) + phase;
  if (momentum)
    momentum[ip] = p;
}

__global__ void gpuRfcaThinKickKernel(double *coord, long nParticles, int stride,
                                      double pCentral, double volt,
                                      double omega, double phase,
                                      double cMks, const double *referenceKick) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, gamma, beta, t, dgamma, gamma1Raw, gamma1;
  double p1, pz, pz1, pRatio;

  if (ip >= nParticles)
    return;

  part = coord + ip * stride;
  if (part[5] == -1)
    return;

  p = pCentral * (1 + part[5]);
  gamma = sqrt(p * p + 1);
  beta = p / gamma;
  t = part[4] / (cMks * beta);
  dgamma = referenceKick ? referenceKick[ip] : volt * sin(omega * t + phase);
  gamma1Raw = gamma + dgamma;
  gamma1 = gamma1Raw;
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - pCentral) / pCentral;
  part[4] = t * cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
  if (gamma1Raw <= 1)
    part[5] = -1;
}

/* ramped_rf_cavity(): a half drift, the kick add_to_particle_energy() makes and a half drift, for
 * every particle as the CPU does.  Stage 1 makes the first drift and the phases, and stage 2 the
 * kick from referenceKick and the second drift; stage 0 does both with the device's sin, and stage 3
 * is the CPU's drift with no voltage.  badCount counts particles left with a coordinate that isn't
 * finite, which the CPU stops on. */
/* RAMPRF's particle loop (tOffset 0, so t - tOffset is t) and MODRF's (tOffset its tAve) */
__global__ void gpuRamprfKernel(double *coord, long nParticles, int stride,
                                double pCentral, double length, double volt,
                                double omega, double phase, double tOffset, double cMks,
                                int stage, double *phaseValue,
                                const double *referenceKick,
                                unsigned long long *badCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, gamma, beta, t, dgamma, gamma1, p1, pz, pz1, pRatio;
  int ic;

  if (ip >= nParticles)
    return;

  part = coord + ip * stride;
  if (stage == 3) {
    part[0] += part[1] * length;
    part[2] += part[3] * length;
    part[4] += length;
    return;
  }
  if (stage != 2) {
    part[0] += part[1] * length / 2;
    part[2] += part[3] * length / 2;
    part[4] += length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  }

  p = pCentral * (1 + part[5]);
  gamma = sqrt(p * p + 1);
  beta = p / gamma;
  t = part[4] / (cMks * beta);
  if (stage == 1) {
    phaseValue[ip] = omega * (t - tOffset) + phase;
    return;
  }
  dgamma = stage == 2 ? referenceKick[ip] : volt * sin(omega * (t - tOffset) + phase);

  gamma1 = gamma + dgamma;
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - pCentral) / pCentral;
  part[4] = t * cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;

  part[0] += part[1] * length / 2;
  part[2] += part[3] * length / 2;
  part[4] += length / 2.0 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);

  for (ic = 0; ic < 6; ic++)
    if (isnan(part[ic]) || isinf(part[ic]))
      break;
  if (ic != 6)
    atomicAdd(badCount, 1ULL);
}

__global__ void gpuTfeedbackKickKernel(double *coord, long nParticles,
                                       int stride, int pickupCoordinate,
                                       int longitudinal, double kick) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (longitudinal)
    part[5] += kick;
  else
    part[pickupCoordinate + 1] += kick / (1 + part[5]);
}

__global__ void gpuRfdfKernel(double *coord, long nParticles, int stride,
                              GPU_RFDF_DATA data, int particleIdIndex) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double tPart, tLight = 0;
  double x, xp, y, yp, beta, px, py, pz, pc;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  x = part[0];
  xp = part[1];
  y = part[2];
  yp = part[3];
  pc = data.pCentral * (1 + part[5]);
  pz = pc / sqrt(1 + xp * xp + yp * yp);
  px = xp * pz;
  py = yp * pz;
  beta = pc / sqrt(1 + pc * pc);
  tPart = part[4] / (data.cMks * beta);

  for (long is = 0; is <= data.nKicks; is++) {
    double cosPhase;
    if (is == 0 || is == data.nKicks) {
      tPart += data.length * sqrt(1 + xp * xp + yp * yp) /
               (2 * data.cMks * beta);
      tLight = data.dtLight;
      x += xp * data.length / 2;
      y += yp * data.length / 2;
      if (is == data.nKicks)
        break;
    } else {
      tPart += data.length * sqrt(1 + xp * xp + yp * yp) /
               (data.cMks * beta);
      tLight += 2 * data.dtLight;
      x += xp * data.length;
      y += yp * data.length;
    }
    if (!((data.startPID >= 0 &&
           part[particleIdIndex] < data.startPID) ||
          (data.endPID >= 0 &&
           part[particleIdIndex] > data.endPID))) {
      double phase = (tPart - tLight) * data.omega + data.ePhase;
      cosPhase = cos(phase);
      px += data.eStrength * cosPhase *
            (1 + data.b2 * (x * x - y * y) / 2.0);
      if (data.b2)
        py -= data.eStrength * cosPhase * data.b2 * x * y;
      if (data.magneticDeflection)
        pz = sqrt(pc * pc - px * px - py * py);
      pz += data.eStrength * data.k * x *
            (1 + data.b2 * (x * x - 3 * y * y) / 6) * sin(phase);
      xp = px / pz;
      yp = py / pz;
      pc = sqrt(px * px + py * py + pz * pz);
    }
  }

  beta = pc / sqrt(1 + pc * pc);
  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = tPart * data.cMks * beta;
  part[5] = (pc - data.pCentral) / data.pCentral;
}

__global__ void gpuSreffectsKernel(double *coord, long nParticles, int stride,
                                   GPU_SREFFECTS_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double P, beta, t;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;

  P = (1 + part[5]) * data.pCentral;
  beta = P / sqrt(P * P + 1);
  t = part[4] / beta;

  if (data.lossOnly) {
    part[5] += data.Ddelta;
  } else {
    double deltaChange;
    double xpEta = part[5] * data.etapx;
    double ypEta;

    const double *draws = data.draws + 3 * ip;

    part[1] = (part[1] - xpEta) * data.Fx + data.Srxp * draws[0] + xpEta;
    ypEta = part[5] * data.etapy;
    part[3] = (part[3] - ypEta) * data.Fy + data.Sryp * draws[1] + ypEta;
    deltaChange = -part[5];
    part[5] = data.Ddelta + part[5] * data.Fdelta + data.Srdelta * draws[2];
    deltaChange += part[5];
    if (data.includeOffsets) {
      part[0] += data.etax * deltaChange;
      part[1] += data.etapx * deltaChange;
      part[2] += data.etay * deltaChange;
      part[3] += data.etapy * deltaChange;
    }
  }

  P = (1 + part[5]) * data.pCentral;
  beta = P / sqrt(P * P + 1);
  part[4] = t * beta;
}

typedef struct GPU_BGGEXP_DEVICE_DATA {
  long nz;
  long termCount;
  const int *m;
  const int *gradient;
  const int *radialPower;
  const double *coefficient;
  const double *multipoleFactor;
  const double *Cmn;
  const double *dCmnDz;
  double dz;
  double zMin;
  double zMax;
  double xCenter;
  double yCenter;
  double length;
  double dxExpansion;
  double pCentral;
  double strength;
  double Bx;
  double By;
  double BFactor[3];
  double particleCharge;
  double particleRelSign;
  double particleMass;
  double cMks;
  const double *potential;
  int isBend;
  long zInterval;
  double step;
  double scaleA;
  double xEntry, zEntry, xExit, zExit;
  double sinEntry, cosEntry, sinExit, cosExit;
} GPU_BGGEXP_DEVICE_DATA;

__device__ double gpuBggexpIntegerPower(double value, int exponent) {
  double power;
  int exponentStack[32];
  int depth = 0;

  /* The exponents are never negative: the device takes normal-field tables, whose m starts at 1 (addBGGExpData()),
   * so 2n+m-1 >= 0 in the field and the vector potential.  (A reciprocal for negative exponents, as ipow() has, made
   * the non-symplectic kernel 18% slower.) */
  if (value == 0)
    return exponent == 0 ? 1.0 : 0.0;
  while (exponent > 8) {
    exponentStack[depth++] = exponent;
    exponent /= 2;
  }
  switch (exponent) {
  case 0:
    power = 1.0;
    break;
  case 1:
    power = value;
    break;
  case 2:
    power = value * value;
    break;
  case 3:
    power = value * value;
    power *= value;
    break;
  case 4:
    power = value * value;
    power *= power;
    break;
  case 5:
    power = value * value;
    power = power * power * value;
    break;
  case 6:
    power = value * value;
    power = power * power * power;
    break;
  case 7:
    power = value * value * value;
    power = power * power * value;
    break;
  case 8:
    power = value * value;
    power = power * power;
    power *= power;
    break;
  default:
    power = 0;
    break;
  }
  while (depth) {
    exponent = exponentStack[--depth];
    power = exponent % 2 ? power * power * value : power * power;
  }
  return power;
}

__device__ __forceinline__ void gpuBggexpFields(
  double *Bx, double *By, double *Bz, double x, double y, long iz,
  const GPU_BGGEXP_DEVICE_DATA &data) {
  double Br = 0, Bphi = 0;
  double r = sqrt(x * x + y * y);
  double phi = atan2(y, x);
  double sinPhi = sin(phi);
  double cosPhi = cos(phi);
  double sinMphi = 0, cosMphi = 0;
  int previousM = INT_MIN;

  *Bz = 0;
  for (long termIndex = 0; termIndex < data.termCount; termIndex++) {
    int m = data.m[termIndex];
    int ig = data.gradient[termIndex];
    double term;
    long tableIndex = termIndex * data.nz + iz;

    if (m != previousM) {
      sinMphi = sin(m * phi);
      cosMphi = cos(m * phi);
      previousM = m;
    }
    term = data.coefficient[termIndex] *
           gpuBggexpIntegerPower(r, data.radialPower[termIndex]);
    *Bz += term * data.dCmnDz[tableIndex] * r * sinMphi *
           data.multipoleFactor[termIndex];
    term *= data.Cmn[tableIndex];
    Br += term * (2 * ig + m) * sinMphi *
          data.multipoleFactor[termIndex];
    Bphi += m * term * cosMphi * data.multipoleFactor[termIndex];
  }
  *Bx = ((data.Bx + (Br * cosPhi - Bphi * sinPhi)) * data.strength) *
        data.BFactor[0];
  *By = ((data.By + (Br * sinPhi + Bphi * cosPhi)) * data.strength) *
        data.BFactor[1];
  *Bz *= data.strength * data.BFactor[2];
}

__global__ void gpuBggexpKernel(double *coord, long nParticles, int stride,
                                GPU_BGGEXP_DEVICE_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double Bx, By, Bz;
  double x, xp, y, yp, s, delta;
  double xTemp, yTemp, xpTemp, ypTemp;
  double xNew, yNew, xpNew, ypNew;
  double ds, denom, preFactorDz;
  double zVertex, zEntry, zExit;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  x = part[0];
  xp = part[1];
  y = part[2];
  yp = part[3];
  s = part[4];
  delta = part[5];

  zVertex = (data.zMax + data.zMin) / 2;
  zEntry = zVertex - data.length / 2;
  zExit = zVertex + data.length / 2;
  x -= data.dxExpansion;
  x -= data.xCenter;
  y -= data.yCenter;
  x -= (zEntry - data.zMin) * xp;
  y -= (zEntry - data.zMin) * yp;
  s -= (zEntry - data.zMin) * sqrt(1.0 + xp * xp + yp * yp);

  for (long iz = 0; iz < data.nz - 1; iz++) {
    gpuBggexpFields(&Bx, &By, &Bz, x, y, iz, data);
    denom = sqrt(1.0 + xp * xp + yp * yp);
    preFactorDz =
      -data.dz * data.particleCharge * data.particleRelSign /
      (data.pCentral * data.particleMass * data.cMks * (1.0 + delta));
    preFactorDz *= denom;
    xTemp = x + data.dz * xp;
    yTemp = y + data.dz * yp;
    xpTemp =
      xp + preFactorDz *
             ((yp * Bz - (1.0 + xp * xp) * By) + xp * yp * Bx);
    ypTemp =
      yp + preFactorDz *
             (((1.0 + yp * yp) * Bx - xp * Bz) - xp * yp * By);
    ds = data.dz * denom;

    gpuBggexpFields(&Bx, &By, &Bz, xTemp, yTemp, iz + 1, data);
    preFactorDz =
      -data.dz * data.particleCharge * data.particleRelSign /
      (data.pCentral * data.particleMass * data.cMks * (1.0 + delta));
    preFactorDz *= sqrt(1.0 + xpTemp * xpTemp + ypTemp * ypTemp);
    xNew = 0.5 * (x + xTemp + data.dz * xpTemp);
    yNew = 0.5 * (y + yTemp + data.dz * ypTemp);
    xpNew =
      0.5 * (xp + xpTemp +
             preFactorDz *
               ((ypTemp * Bz - (1.0 + xpTemp * xpTemp) * By) +
                xpTemp * ypTemp * Bx));
    ypNew =
      0.5 * (yp + ypTemp +
             preFactorDz *
               (((1.0 + ypTemp * ypTemp) * Bx - xpTemp * Bz) -
                xpTemp * ypTemp * By));
    ds = 0.5 *
         (ds + data.dz *
                 sqrt(1.0 + xpTemp * xpTemp + ypTemp * ypTemp));
    x = xNew;
    y = yNew;
    xp = xpNew;
    yp = ypNew;
    s += ds;
  }

  x -= (data.zMax - zExit) * xp;
  y -= (data.zMax - zExit) * yp;
  s -= (data.zMax - zExit) * sqrt(1.0 + xp * xp + yp * yp);
  x += data.xCenter;
  y += data.yCenter;
  x += data.dxExpansion;

  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = s;
}

/* computeGGEVectorPotential() for normal data, in the CPU's order of operations */
__device__ void gpuBggexpVectorPotential(
  double *Ax, double *dAx_dx, double *dAx_dy,
  double *Ay, double *dAy_dx, double *dAy_dy,
  double *dAz_dx, double *dAz_dy, double x, double y, long iz,
  const GPU_BGGEXP_DEVICE_DATA &data) {
  double r = sqrt(x * x + y * y);
  double phi = atan2(y, x);
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);
  double sin_m1phi = 0, cos_m1phi = 0, sin_mphi = 0, cos_mphi = 0;
  int previousM = INT_MIN;

  *Ax = *dAx_dx = *dAx_dy = *Ay = *dAy_dx = *dAy_dy = *dAz_dx = *dAz_dy = 0;
  for (long termIndex = 0; termIndex < data.termCount; termIndex++) {
    int m = data.m[termIndex];
    int ig = data.gradient[termIndex];
    double mfactor = data.multipoleFactor[termIndex];
    const double *potential = data.potential + 3 * termIndex;
    long tableIndex = termIndex * data.nz + iz;
    double dC = data.dCmnDz[tableIndex];
    double C = data.Cmn[tableIndex];
    double term, rDeriv, phiDeriv;

    if (m != previousM) {
      sin_m1phi = sin((m + 1) * phi);
      cos_m1phi = cos((m + 1) * phi);
      sin_mphi = sin(m * phi);
      cos_mphi = cos(m * phi);
      previousM = m;
    }
    term = potential[0] * gpuBggexpIntegerPower(r, 2 * ig + m) / potential[1];

    *Ax += term * cos_m1phi * dC * mfactor;
    *Ay += term * sin_m1phi * dC * mfactor;

    rDeriv = (2 * ig + m + 1) * cos_m1phi * dC;
    phiDeriv = -(m + 1) * sin_m1phi * dC;
    *dAx_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
    *dAx_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

    rDeriv = (2 * ig + m + 1) * sin_m1phi * dC;
    phiDeriv = (m + 1) * cos_m1phi * dC;
    *dAy_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
    *dAy_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

    term = potential[0] * gpuBggexpIntegerPower(r, 2 * ig + m - 1) / potential[2];
    rDeriv = -(2 * ig + m) * cos_mphi * C;
    phiDeriv = m * sin_mphi * C;
    *dAz_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
    *dAz_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;
  }
  *Ax *= r;
  *Ay *= r;
  *dAz_dx = *dAz_dx - data.By;
  *dAz_dy = *dAz_dy + data.Bx;
}

/* trackBGGExpansion() with SYMPLECTIC=1, no radiation: canonical coordinates, implicit midpoint rule
 * every Z_INTERVAL planes, the CPU's bend frame changes and drifts */
__global__ void gpuBggexpSymplecticKernel(double *coord, long nParticles,
                                          int stride,
                                          GPU_BGGEXP_DEVICE_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double x, y, px, py, s, delta, delta_s, denom, magnet_s, sin_phi, cos_phi;
  double xMid, yMid, xNext, yNext, xLoop, yLoop, pxNext, pyNext, pxLoop, pyLoop;
  double ux = 0, uy = 0;
  double Ax, dAx_dx, dAx_dy, Ay, dAy_dx, dAy_dy, dAz_dx, dAz_dy;
  const double epsImplConverge = 1.e-14 * 1.0e-3;
  long iz, iImpLoop;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;

  /* Transform to canonical coordinates */
  delta = part[5];
  denom = 1.0 / sqrt(1.0 + part[1] * part[1] + part[3] * part[3]);
  x = part[0];
  px = part[1] * (1.0 + delta) * denom;
  y = part[2];
  py = part[3] * (1.0 + delta) * denom;
  s = part[4];

  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  if (data.isBend) {
    sin_phi = data.sinEntry;
    cos_phi = data.cosEntry;
    magnet_s = x * sin_phi / (cos_phi - sin_phi * px * denom);
  } else {
    sin_phi = magnet_s = 0;
    cos_phi = 1;
  }
  x = (x + px * magnet_s * denom * denom) / cos_phi;
  y = (y + py * magnet_s * denom * denom);
  px = px * cos_phi + sin_phi / denom;
  delta_s = (1 + delta) * magnet_s * denom;

  x += data.xEntry - data.dxExpansion;
  if (data.xCenter || data.yCenter) {
    x -= data.xCenter;
    y -= data.yCenter;
  }

  /* Drift backward from entrance plane to beginning of field map */
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  x -= (data.zEntry - data.zMin) * px * denom;
  y -= (data.zEntry - data.zMin) * py * denom;
  s -= (data.zEntry - data.zMin) * (1.0 + delta) * denom;

  /* Back 1/2 step for the implicit midpoint rule */
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  x -= 0.5 * data.step * data.zInterval * px * denom;
  y -= 0.5 * data.step * data.zInterval * py * denom;
  s -= 0.5 * data.step * data.zInterval * (1.0 + delta) * denom;

  for (iz = 0; iz < data.nz; iz += data.zInterval) {
    gpuBggexpVectorPotential(&Ax, &dAx_dx, &dAx_dy, &Ay, &dAy_dx, &dAy_dy,
                             &dAz_dx, &dAz_dy, x, y, iz, data);
    ux = px - data.scaleA * Ax;
    uy = py - data.scaleA * Ay;
    denom = (1.0 + delta) * (1.0 + delta) - ux * ux - uy * uy;
    denom = 1.0 / sqrt(denom);
    xNext = x + data.step * data.zInterval * ux * denom;
    yNext = y + data.step * data.zInterval * uy * denom;
    pxNext = px + data.step * data.zInterval * data.scaleA *
                    ((ux * dAx_dx + uy * dAy_dx) * denom + dAz_dx);
    pyNext = py + data.step * data.zInterval * data.scaleA *
                    ((ux * dAx_dy + uy * dAy_dy) * denom + dAz_dy);
    iImpLoop = 0;
    do {
      xLoop = xNext;
      yLoop = yNext;
      pxLoop = pxNext;
      pyLoop = pyNext;
      xMid = 0.5 * (x + xLoop);
      yMid = 0.5 * (y + yLoop);
      gpuBggexpVectorPotential(&Ax, &dAx_dx, &dAx_dy, &Ay, &dAy_dx, &dAy_dy,
                               &dAz_dx, &dAz_dy, xMid, yMid, iz, data);
      ux = 0.5 * (px + pxLoop) - data.scaleA * Ax;
      uy = 0.5 * (py + pyLoop) - data.scaleA * Ay;
      denom = (1.0 + delta) * (1.0 + delta) - ux * ux - uy * uy;
      denom = 1.0 / sqrt(denom);
      xNext = x + data.step * data.zInterval * ux * denom;
      yNext = y + data.step * data.zInterval * uy * denom;
      pxNext = px + data.step * data.zInterval * data.scaleA *
                      ((ux * dAx_dx + uy * dAy_dx) * denom + dAz_dx);
      pyNext = py + data.step * data.zInterval * data.scaleA *
                      ((ux * dAx_dy + uy * dAy_dy) * denom + dAz_dy);
      iImpLoop++;
      if (iImpLoop > 10)
        break;
    } while ((fabs(xNext - xLoop) > epsImplConverge) ||
             (fabs(yNext - yLoop) > epsImplConverge) ||
             (fabs(pxNext - pxLoop) > epsImplConverge) ||
             (fabs(pyNext - pyLoop) > epsImplConverge));
    x = xNext;
    y = yNext;
    px = pxNext;
    py = pyNext;
    delta_s += data.step * data.zInterval * ((1.0 + delta) * denom - 1.0);
    s += data.step * data.zInterval;
  }

  /* Back 1/2 step to the last plane used; the CPU's forward drift after it never runs (iz >= nz) */
  px = ux;
  py = uy;
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  x -= 0.5 * data.step * data.zInterval * px * denom;
  y -= 0.5 * data.step * data.zInterval * py * denom;
  s -= 0.5 * data.step * data.zInterval * (1.0 + delta) * denom;

  /* Drift backward from end of field map to exit plane */
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  x -= (data.zMax - data.zExit) * px * denom;
  y -= (data.zMax - data.zExit) * py * denom;
  s -= (data.zMax - data.zExit) * (1.0 + delta) * denom;

  if (data.xCenter || data.yCenter) {
    x += data.xCenter;
    y += data.yCenter;
  }
  x -= data.xExit - data.dxExpansion;

  sin_phi = data.sinExit;
  cos_phi = data.cosExit;
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  magnet_s = x * sin_phi / (cos_phi - sin_phi * px * denom);
  x = (x + px * magnet_s * denom * denom) / cos_phi;
  y = (y + py * magnet_s * denom * denom);
  px = px * cos_phi + sin_phi / denom;
  delta_s += (1 + delta) * magnet_s * denom;

  part[0] = x;
  denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
  denom = 1.0 / sqrt(denom);
  part[1] = px * denom;
  part[2] = y;
  part[3] = py * denom;
  part[4] = s + delta_s;
  part[5] = delta;
}

__device__ __forceinline__ double gpuCwigglerPoleFactor(
  double z, const GPU_CWIGGLER_DATA &data) {
  if (z < data.z3) {
    if (z < data.z1)
      return data.poleFactor[0];
    if (z < data.z2)
      return data.poleFactor[1];
    return data.poleFactor[2];
  }
  if (z < data.z4)
    return 1.0;
  if (z < data.z5)
    return data.poleFactor[2];
  if (z < data.z6)
    return data.poleFactor[1];
  return data.poleFactor[0];
}

__device__ __forceinline__ double gpuCwigglerSinc(double x) {
  double x2 = x * x;
  return 1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0)));
}

/* The harmonic functions keep the CPU's order of operations (gwig.c) */
__device__ __noinline__ void gpuCwigglerHarmonicAx(
  double *pax, double *paxpy, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  double ax = data.axConstant * z / data.betaGamma, axpy = 0.0;

  if (data.horizontalHarmonics && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    double aw = data.horizontalAw * poleFactor;
    for (long i = 0; i < data.horizontalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[i];
      double cw = h.strength * aw / data.betaGamma;
      double sz = sin(h.kz * z + h.phase);
      if (!data.horizontalSplitPole) {
        double sxkx = h.useSine ? sin(h.kx * x) / h.kx : x * gpuCwigglerSinc(h.kx * x);
        if (data.tguGradient && i == 0)
          ax = ax + cw * ((1 + x * data.tguGradient) * cosh(h.ky * y) * sz -
                          data.tguCompFactor * data.tguGradient * z * aw / (2 * data.betaGamma));
        else
          ax = ax + cw * h.kwOverKz * cos(h.kx * x) * cosh(h.ky * y) * sz;
        if (data.tguGradient && i == 0)
          axpy = axpy + cw * data.kw * x * (1 + data.tguGradient * x / 2) * sinh(h.ky * y) * sz;
        else
          axpy = axpy + cw * h.kwOverKz * h.ky * sxkx * sinh(h.ky * y) * sz;
      } else {
        double shxkx = sinh(h.kx * x) / h.kx;
        ax = ax + cw * h.kwOverKz * cos(h.ky * y) * cosh(h.kx * x) * sz;
        axpy = axpy - cw * h.kwOverKz * h.ky * sin(h.ky * y) * shxkx * sz;
      }
    }
  }
  if (data.verticalHarmonics && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    double aw = data.verticalAw * poleFactor;
    for (long i = 0; i < data.verticalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[data.horizontalHarmonics + i];
      double cw = h.strength * aw / data.betaGamma;
      double sz = sin(h.kz * z + h.phase);
      if (!data.verticalSplitPole) {
        ax = ax + cw * h.kwOverKz * h.ratio * sinh(h.kx * x) * sin(h.ky * y) * sz;
        axpy = axpy + cw * h.kwOverKz * h.ratioSquared * cosh(h.kx * x) * cos(h.ky * y) * sz;
      } else {
        ax = ax - cw * h.kwOverKz * h.ratio * sinh(h.ky * y) * sin(h.kx * x) * sz;
        axpy = axpy + cw * h.kwOverKz * h.ratioSquared * cosh(h.ky * y) * cos(h.kx * x) * sz;
      }
    }
  }
  *pax = ax;
  *paxpy = axpy;
}

__device__ __noinline__ void gpuCwigglerHarmonicAy(
  double *pay, double *paypx, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  double ay = data.ayConstant * z / data.betaGamma, aypx = 0.0;

  if (data.horizontalHarmonics && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    double aw = data.horizontalAw * poleFactor;
    for (long i = 0; i < data.horizontalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[i];
      double cw = h.strength * aw / data.betaGamma;
      double sz = sin(h.kz * z + h.phase);
      if (!data.horizontalSplitPole) {
        if (data.tguGradient && i == 0)
          ay = ay - cw * data.tguGradient * sz * sinh(h.ky * y) / data.kw;
        else
          ay = ay + cw * h.kwOverKz * h.ratio * sin(h.kx * x) * sinh(h.ky * y) * sz;
        aypx = aypx + cw * h.kwOverKz * h.ratioSquared * cos(h.kx * x) * cosh(h.ky * y) * sz;
      } else {
        ay = ay - cw * h.kwOverKz * h.kx / h.ky * sin(h.ky * y) * sinh(h.kx * x) * sz;
        aypx = aypx + cw * h.kwOverKz * h.ratioSquared * cos(h.ky * y) * cosh(h.kx * x) * sz;
      }
    }
  }
  if (data.verticalHarmonics && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    double aw = data.verticalAw * poleFactor;
    for (long i = 0; i < data.verticalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[data.horizontalHarmonics + i];
      double cw = h.strength * aw / data.betaGamma;
      double sz = sin(h.kz * z + h.phase);
      if (!data.verticalSplitPole) {
        double syky = h.useSine ? sin(h.ky * y) / h.ky : y * gpuCwigglerSinc(h.ky * y);
        ay = ay + cw * h.kwOverKz * cosh(h.kx * x) * cos(h.ky * y) * sz;
        aypx = aypx + cw * h.kwOverKz * h.kx * sinh(h.kx * x) * syky * sz;
      } else {
        double shyky = sinh(h.ky * y) / h.ky;
        ay = ay + cw * h.kwOverKz * cosh(h.ky * y) * cos(h.kx * x) * sz;
        aypx = aypx - cw * h.kwOverKz * h.kx * shyky * sin(h.kx * x) * sz;
      }
    }
  }
  *pay = ay;
  *paypx = aypx;
}

__device__ __noinline__ void gpuCwigglerHarmonicField(
  double *pbx, double *pby, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  double bx = data.constantBx, by = data.constantBy;

  if (data.horizontalHarmonics && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    double b0 = data.horizontalField * poleFactor;
    for (long i = 0; i < data.horizontalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[i];
      double cz = cos(h.kz * z + h.phase);
      if (!data.horizontalSplitPole) {
        if (i == 0 && data.tguGradient) {
          bx += b0 * h.strength * data.tguGradient * cz * sinh(h.ky * y) / h.kz;
          by -= b0 * h.strength *
                (cosh(h.ky * y) * cz * (1 + data.tguGradient * x) +
                 data.tguCompFactor * data.tguGradient *
                   (b0 * h.strength * data.tguCharge / (data.tguFieldScale * (h.kz * h.kz))));
        } else {
          bx += b0 * h.strength * h.kx / h.ky * sin(h.kx * x) * sinh(h.ky * y) * cz;
          by -= b0 * h.strength * cos(h.kx * x) * cosh(h.ky * y) * cz;
        }
      } else {
        bx -= b0 * h.strength * h.kx / h.ky * sinh(h.kx * x) * sin(h.ky * y) * cz;
        by -= b0 * h.strength * cosh(h.kx * x) * cos(h.ky * y) * cz;
      }
    }
  }
  if (data.verticalHarmonics && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    double b0 = data.verticalField * poleFactor;
    for (long i = 0; i < data.verticalHarmonics; i++) {
      const GPU_CWIGGLER_HARMONIC &h = data.harmonics[data.horizontalHarmonics + i];
      double cz = cos(h.kz * z + h.phase);
      if (!data.verticalSplitPole) {
        bx += b0 * h.strength * cosh(h.kx * x) * cos(h.ky * y) * cz;
        by -= b0 * h.strength * h.ky / h.kx * sinh(h.kx * x) * sin(h.ky * y) * cz;
      } else {
        bx += b0 * h.strength * cos(h.kx * x) * cosh(h.ky * y) * cz;
        by += b0 * h.strength * h.ky / h.kx * sin(h.kx * x) * sinh(h.ky * y) * cz;
      }
    }
  }
  *pbx = bx;
  *pby = by;
}

__device__ __forceinline__ void gpuCwigglerAx(
  double *ax, double *axpy, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  if (data.harmonics) {
    gpuCwigglerHarmonicAx(ax, axpy, x, y, z, poleFactor, data);
    return;
  }
  *ax = data.axConstant * z / data.betaGamma;
  *axpy = 0.0;
  if (data.hasHorizontal && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    double coefficient = data.horizontalAw * poleFactor / data.betaGamma;
    double sinz = sin(data.kw * z + data.horizontalPhase);
    *ax += coefficient * cosh(data.kw * y) * sinz;
    *axpy = coefficient * data.kw * x * sinh(data.kw * y) * sinz;
  }
}

__device__ __forceinline__ void gpuCwigglerAy(
  double *ay, double *aypx, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  if (data.harmonics) {
    gpuCwigglerHarmonicAy(ay, aypx, x, y, z, poleFactor, data);
    return;
  }
  *ay = data.ayConstant * z / data.betaGamma;
  *aypx = 0.0;
  if (data.hasVertical && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    double coefficient = data.verticalAw * poleFactor / data.betaGamma;
    double sinz = sin(data.kw * z + data.verticalPhase);
    *ay += coefficient * cosh(data.kw * x) * sinz;
    *aypx = coefficient * data.kw * sinh(data.kw * x) * y * sinz;
  }
}

__device__ __forceinline__ void gpuCwigglerField(
  double *bx, double *by, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  if (data.harmonics) {
    gpuCwigglerHarmonicField(bx, by, x, y, z, poleFactor, data);
    return;
  }
  *bx = data.constantBx;
  *by = data.constantBy;
  if (data.hasHorizontal && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    *by -= data.horizontalField * poleFactor * cosh(data.kw * y) *
           cos(data.kw * z + data.horizontalPhase);
  }
  if (data.hasVertical && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    *bx += data.verticalField * poleFactor * cosh(data.kw * x) *
           cos(data.kw * z + data.verticalPhase);
  }
}

/* The values GWigAddToFieldOutput() takes */
__device__ __noinline__ void gpuCwigglerRecordField(double *row, double x, double qx, double y, double z,
                                                    double bx, double by, double poleFactor) {
  row[0] = x;
  row[1] = qx;
  row[2] = y;
  row[3] = z;
  row[4] = bx;
  row[5] = by;
  row[6] = poleFactor;
}

__device__ __forceinline__ void gpuCwigglerRadiationKick(
  double *qx, double *qy, double *delta, double x, double y, double z,
  double dl, double poleFactor, const GPU_CWIGGLER_DATA &data,
  const double *isrDraws, long *isrIndex, double *fieldRow) {
  double ax, ay, axpy, aypx, bx, by, b2, rigidity, irho2;
  double dFactor, dDelta;

  gpuCwigglerAx(&ax, &axpy, x, y, z, poleFactor, data);
  gpuCwigglerAy(&ay, &aypx, x, y, z, poleFactor, data);
  gpuCwigglerField(&bx, &by, x, y, z, poleFactor, data);
  *qx -= ax;
  *qy -= ay;
  b2 = bx * bx + by * by;
  if (b2 != 0.0 && (data.synchRad || isrDraws)) {
    rigidity = data.pCentral / 586.679074042074490;
    irho2 = b2 / (rigidity * rigidity);
    dFactor = (1.0 + *delta) * (1.0 + *delta);
    if (data.synchRad) {
      dDelta = -data.srCoef * dFactor * irho2 * dl;
      *delta += dDelta;
      *qx *= 1.0 + dDelta;
      *qy *= 1.0 + dDelta;
    }
    if (isrDraws) {
      double draw = *isrIndex < data.isrDrawsPerParticle ? isrDraws[*isrIndex] : 0.0;

      (*isrIndex)++;
      dDelta = data.isrCoef * dFactor * pow(irho2, 0.75) * sqrt(dl) * draw;
      *delta += dDelta;
      *qx *= 1.0 + dDelta;
      *qy *= 1.0 + dDelta;
    }
  }
  if (fieldRow)
    gpuCwigglerRecordField(fieldRow, x, *qx, y, z, bx, by, poleFactor);
  *qx += ax;
  *qy += ay;
}

__device__ __forceinline__ void gpuCwigglerMapSecondOrder(
  double *x, double *qx, double *y, double *qy, double *s,
  double delta, double *z, double dl, double poleFactor,
  const GPU_CWIGGLER_DATA &data) {
  double dld = dl / (1.0 + delta);
  double dl2 = 0.5 * dl;
  double dl2d = dl2 / (1.0 + delta);
  double ax, ay, axpy, aypx;

  *z += dl2;

  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx -= aypx;
  *qy -= ay;
  *y += dl2d * *qy;
  *s = *s + 0.5 * dl2d * (*qy * *qy) / (1.0 + delta);
  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx += aypx;
  *qy += ay;

  gpuCwigglerAx(&ax, &axpy, *x, *y, *z, poleFactor, data);
  *qx -= ax;
  *qy -= axpy;
  *x += dld * *qx;
  *s = *s + dl + 0.5 * dld * (*qx * *qx) / (1.0 + delta);
  gpuCwigglerAx(&ax, &axpy, *x, *y, *z, poleFactor, data);
  *qx += ax;
  *qy += axpy;

  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx -= aypx;
  *qy -= ay;
  *y += dl2d * *qy;
  *s = *s + 0.5 * dl2d * (*qy * *qy) / (1.0 + delta);
  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx += aypx;
  *qy += ay;

  *z += dl2;
}

__global__ void gpuCwigglerKernel(double *coord, long nParticles, int stride,
                                  GPU_CWIGGLER_DATA data) {
  const double fourthOrderX1 =
    1.3512071919596576340476878089715;
  const double fourthOrderX0 =
    -1.7024143839193152680953756179429;
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double x, xp, qx, y, yp, qy, s, delta, denom, z = 0.0;
  double periodLength, dl, dl1, dl0;
  long totalSteps;
  const double *isrDraws;
  double *fieldRecord;
  long isrIndex = 0;

  if (ip >= nParticles)
    return;
  isrDraws = data.isrCoef ? data.isrDraws + ip * data.isrDrawsPerParticle : NULL;
  fieldRecord = data.fieldRecord ?
    data.fieldRecord + ip * (data.periods * data.stepsPerPeriod + 1) * GPU_CWIGGLER_FIELD_RECORD_SIZE : NULL;
  part = coord + ip * stride;
  x = part[0];
  xp = part[1];
  y = part[2];
  yp = part[3];
  s = part[4];
  delta = part[5];
  if (data.hasTilt) {
    double x0 = x, xp0 = xp, y0 = y, yp0 = yp;
    x = x0 * data.cosTilt + y0 * data.sinTilt;
    y = -x0 * data.sinTilt + y0 * data.cosTilt;
    xp = xp0 * data.cosTilt + yp0 * data.sinTilt;
    yp = -xp0 * data.sinTilt + yp0 * data.cosTilt;
    if (data.spinOffset)
      gpuSpinRoll(part + data.spinOffset, data.cosTilt, data.sinTilt);
  }
  if (data.hasOffset) {
    double ds = data.dz * sqrt(1.0 + xp * xp + yp * yp);
    x += -data.dx + data.dz * xp;
    y += -data.dy + data.dz * yp;
    s += ds;
  }
  denom = sqrt(1.0 + xp * xp + yp * yp);
  qx = (1.0 + delta) * xp / denom;
  qy = (1.0 + delta) * yp / denom;

  periodLength = data.periodLength;
  dl = periodLength / data.stepsPerPeriod;
  dl1 = fourthOrderX1 * dl;
  dl0 = fourthOrderX0 * dl;
  totalSteps = data.periods * data.stepsPerPeriod;
  if (data.synchRad || isrDraws)
    gpuCwigglerRadiationKick(&qx, &qy, &delta, x, y, z, dl,
                              data.poleFactor[0], data, isrDraws, &isrIndex, NULL);
  for (long step = 1; step <= totalSteps; step++) {
    double poleFactor = gpuCwigglerPoleFactor(
      ((step - 1) * data.length) / totalSteps, data);
    if (step == 1 && fieldRecord) {
      double bx, by;

      gpuCwigglerField(&bx, &by, x, y, z, poleFactor, data);
      gpuCwigglerRecordField(fieldRecord, x, qx, y, z, bx, by, poleFactor);
    }
    if (data.integrationOrder == 2) {
      gpuCwigglerMapSecondOrder(&x, &qx, &y, &qy, &s, delta, &z,
                                dl, poleFactor, data);
    } else {
      gpuCwigglerMapSecondOrder(&x, &qx, &y, &qy, &s, delta, &z,
                                dl1, poleFactor, data);
      gpuCwigglerMapSecondOrder(&x, &qx, &y, &qy, &s, delta, &z,
                                dl0, poleFactor, data);
      gpuCwigglerMapSecondOrder(&x, &qx, &y, &qy, &s, delta, &z,
                                dl1, poleFactor, data);
    }
    if (data.synchRad || isrDraws || fieldRecord)
      gpuCwigglerRadiationKick(&qx, &qy, &delta, x, y, z, dl, poleFactor, data, isrDraws, &isrIndex,
                                fieldRecord ? fieldRecord + step * GPU_CWIGGLER_FIELD_RECORD_SIZE : NULL);
  }

  denom = sqrt((1.0 + delta) * (1.0 + delta) - qx * qx - qy * qy);
  xp = (qx - data.exitAx) / denom;
  yp = (qy - data.exitAy) / denom;
  if (data.hasOffset) {
    double ds = -data.dz * sqrt(1.0 + xp * xp + yp * yp);
    x += data.dx + -data.dz * xp;
    y += data.dy + -data.dz * yp;
    s += ds;
  }
  if (data.hasTilt) {
    double x0 = x, xp0 = xp, y0 = y, yp0 = yp;
    x = x0 * data.cosTilt - y0 * data.sinTilt;
    y = x0 * data.sinTilt + y0 * data.cosTilt;
    xp = xp0 * data.cosTilt - yp0 * data.sinTilt;
    yp = xp0 * data.sinTilt + yp0 * data.cosTilt;
    if (data.spinOffset)
      gpuSpinRoll(part + data.spinOffset, data.cosTilt, -data.sinTilt);
  }
  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = s;
  part[5] = delta;
}

__device__ __forceinline__ void gpuFtableInterpolate(
  double fieldValue[3], double x, double y, double z,
  const GPU_FTABLE_DATA &data) {
  double position[3] = {x, y, z};
  double coefficient[3][2];
  long grid[3][2];

  fieldValue[0] = fieldValue[1] = fieldValue[2] = 0.0;
  for (long dimension = 0; dimension < 3; dimension++) {
    double normalized;
    if (position[dimension] > data.maximum[dimension] ||
        position[dimension] < data.minimum[dimension])
      return;
    normalized = (position[dimension] - data.minimum[dimension]) /
                 (data.maximum[dimension] - data.minimum[dimension]);
    grid[dimension][0] =
      static_cast<long>(normalized * data.dimensions[dimension] - 0.5);
    coefficient[dimension][1] =
      (position[dimension] - data.minimum[dimension]) /
        data.spacing[dimension] -
      grid[dimension][0] - 0.5;
    if (coefficient[dimension][1] < 0) {
      grid[dimension][0]--;
      coefficient[dimension][1]++;
    }
    coefficient[dimension][0] = 1.0 - coefficient[dimension][1];
    grid[dimension][1] = grid[dimension][0] + 1;
    if (grid[dimension][0] < 0)
      grid[dimension][0] = 0;
    if (grid[dimension][1] == data.dimensions[dimension])
      grid[dimension][1]--;
  }

  for (long corner = 0; corner < 8; corner++) {
    long bitX = (corner >> 2) & 1;
    long bitY = (corner >> 1) & 1;
    long bitZ = corner & 1;
    long index =
      grid[2][bitZ] +
      data.dimensions[2] *
        (grid[1][bitY] + data.dimensions[1] * grid[0][bitX]);
    double weight = 1.0;
    weight *= coefficient[2][bitZ];
    weight *= coefficient[1][bitY];
    weight *= coefficient[0][bitX];
    for (long field = 0; field < 3; field++)
      fieldValue[field] += weight * data.field[field][index];
  }
  for (long field = 0; field < 3; field++)
    fieldValue[field] *= data.factor;
}

__device__ __forceinline__ void gpuFtableRotate(
  const double matrix[9], double vector[3], int inverse) {
  double result[3] = {0.0, 0.0, 0.0};
  for (long row = 0; row < 3; row++) {
    for (long column = 0; column < 3; column++) {
      if (!inverse)
        result[row] += matrix[row * 3 + column] * vector[column];
      else
        result[row] += matrix[column * 3 + row] * vector[column];
    }
  }
  vector[0] = result[0];
  vector[1] = result[1];
  vector[2] = result[2];
}

__device__ __forceinline__ int gpuFtableSolveCubic(
  double a, double b, double c, double *x0, double *x1, double *x2) {
  double q = a * a - 3.0 * b;
  double r = 2.0 * a * a * a - 9.0 * a * b + 27.0 * c;
  double Q = q / 9.0;
  double R = r / 54.0;
  double Q3 = Q * Q * Q;
  double R2 = R * R;
  double CR2 = 729.0 * r * r;
  double CQ3 = 2916.0 * q * q * q;
  *x0 = *x1 = *x2 = 0.0;
  if (R == 0.0 && Q == 0.0) {
    *x0 = *x1 = *x2 = -a / 3.0;
    return 3;
  }
  if (CR2 == CQ3) {
    double sqrtQ = sqrt(Q);
    if (R > 0.0) {
      *x0 = -2.0 * sqrtQ - a / 3.0;
      *x1 = *x2 = sqrtQ - a / 3.0;
    } else {
      *x0 = *x1 = -sqrtQ - a / 3.0;
      *x2 = 2.0 * sqrtQ - a / 3.0;
    }
    return 3;
  }
  if (R2 < Q3) {
    double sgnR = R >= 0.0 ? 1.0 : -1.0;
    double ratio = sgnR * sqrt(R2 / Q3);
    double theta = acos(ratio);
    double norm = -2.0 * sqrt(Q);
    double root0 = norm * cos(theta / 3.0) - a / 3.0;
    double root1 =
      norm * cos((theta + 2.0 * M_PI) / 3.0) - a / 3.0;
    double root2 =
      norm * cos((theta - 2.0 * M_PI) / 3.0) - a / 3.0;
    if (root0 > root1) {
      double swap = root0;
      root0 = root1;
      root1 = swap;
    }
    if (root1 > root2) {
      double swap = root1;
      root1 = root2;
      root2 = swap;
    }
    if (root0 > root1) {
      double swap = root0;
      root0 = root1;
      root1 = swap;
    }
    *x0 = root0;
    *x1 = root1;
    *x2 = root2;
    return 3;
  }
  {
    double sgnR = R >= 0.0 ? 1.0 : -1.0;
    double A =
      -sgnR * pow(fabs(R) + sqrt(R2 - Q3), 1.0 / 3.0);
    double B = Q / A;
    *x0 = A + B - a / 3.0;
  }
  return 1;
}

__device__ __forceinline__ double gpuFtableChooseTheta(
  double rho, double x0, double x1, double x2) {
  double theta;
  if (rho < 0.0) {
    theta = -DBL_MAX;
    if (x0 < 0.0 && x0 > theta)
      theta = x0;
    if (x1 < 0.0 && x1 > theta)
      theta = x1;
    if (x2 < 0.0 && x2 > theta)
      theta = x2;
  } else {
    theta = DBL_MAX;
    if (x0 > 0.0 && x0 < theta)
      theta = x0;
    if (x1 > 0.0 && x1 < theta)
      theta = x1;
    if (x2 > 0.0 && x2 < theta)
      theta = x2;
  }
  return theta;
}

__global__ void gpuFtableKernel(double *coord, long nParticles, int stride,
                                GPU_FTABLE_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double step, sLocation;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  step = data.length / data.nKicks;
  sLocation = step / 2.0;

  /* ftable_frame_converter(entrance), offsetBeamCoordinatesForMisalignment(),
   * rotateBeamCoordinatesForMisalignment() */
  if (data.entrance == 1) {
    part[0] += part[1] * -data.l1;
    part[2] += part[3] * -data.l1;
    part[4] += -data.l1 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  } else if (data.entrance == 2) {
    double x0 = part[0], xp0 = part[1], y0 = part[2], yp0 = part[3], length, temp;
    length = data.l1 - data.sinE1 * x0;
    temp = data.cosE1 - data.sinE1 * xp0;
    part[1] = (data.cosE1 * xp0 + data.sinE1) / temp;
    part[3] = yp0 / temp;
    part[4] -= length * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = data.cosE1 * x0 - part[1] * length;
    part[2] = y0 - part[3] * length;
  }
  if (data.doOffset) {
    part[4] += data.dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - data.dx + data.dz * part[1];
    part[2] = part[2] - data.dy + data.dz * part[3];
  }
  if (data.doTilt) {
    double x = part[0], xp = part[1], y = part[2], yp = part[3];
    part[0] = x * data.cosTilt + y * data.sinTilt;
    part[2] = -x * data.sinTilt + y * data.cosTilt;
    part[1] = xp * data.cosTilt + yp * data.sinTilt;
    part[3] = -xp * data.sinTilt + yp * data.cosTilt;
  }

  for (long kick = 0; kick < data.nKicks; kick++, sLocation += step) {
    double factor = sqrt(1.0 + part[1] * part[1] + part[3] * part[3]);
    double p0 = (1.0 + part[5]) * data.pCentral;
    double p[3], B[3], matrix[9], pA, BA;
    double xyz[3];

    p[2] = p0 / factor;
    p[0] = part[1] * p[2];
    p[1] = part[3] * p[2];
    gpuFtableInterpolate(B, part[0] + part[1] * step / 2.0,
                         part[2] + part[3] * step / 2.0,
                         sLocation, data);
    BA = sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
    matrix[0] = -(p[1] * B[2] - p[2] * B[1]);
    matrix[1] = -(p[2] * B[0] - p[0] * B[2]);
    matrix[2] = -(p[0] * B[1] - p[1] * B[0]);
    pA = sqrt(matrix[0] * matrix[0] +
              matrix[1] * matrix[1] + matrix[2] * matrix[2]);
    if (BA > data.threshold && pA != 0.0) {
      double rho, theta0 = 0.0, theta1 = 0.0, theta2 = 0.0, theta;
      double tmA, tmB, tmC;
      matrix[0] /= pA;
      matrix[1] /= pA;
      matrix[2] /= pA;
      matrix[3] = B[0] / BA;
      matrix[4] = B[1] / BA;
      matrix[5] = B[2] / BA;
      matrix[6] = matrix[1] * matrix[5] - matrix[2] * matrix[4];
      matrix[7] = matrix[2] * matrix[3] - matrix[0] * matrix[5];
      matrix[8] = matrix[0] * matrix[4] - matrix[1] * matrix[3];
      gpuFtableRotate(matrix, p, 0);
      gpuFtableRotate(matrix, B, 0);
      rho = p[2] / (data.eomc * B[1]);
      if (matrix[8] != 0.0) {
        tmA = 3.0 * matrix[2] / matrix[8];
        tmB = -6.0 * matrix[5] * p[1] / p[2] / matrix[8] - 6.0;
        tmC = 6.0 * step / rho / matrix[8];
        gpuFtableSolveCubic(tmA, tmB, tmC,
                            &theta0, &theta1, &theta2);
      } else if (matrix[2] != 0.0) {
        tmA = matrix[5] * p[1] / p[2] + matrix[8];
        theta0 =
          (tmA - sqrt(tmA * tmA - 2.0 * matrix[2] * step / rho)) /
          matrix[2];
        theta1 =
          (tmA + sqrt(tmA * tmA - 2.0 * matrix[2] * step / rho)) /
          matrix[2];
      } else {
        tmA = matrix[5] * p[1] / p[2] + matrix[8];
        theta0 = step / rho / tmA;
      }
      theta = gpuFtableChooseTheta(rho, theta0, theta1, theta2);
      /* theta is small and rho large where the field is weak, so an ulp of cos(theta) - 1 is
       * multiplied by rho: near-correctly rounded sine and cosine keep that next to glibc's */
      p[0] = -p[2] * gpuAccurateSin(theta);
      p[2] *= gpuAccurateCos(theta);
      xyz[0] = rho * (gpuAccurateCos(theta) - 1.0);
      xyz[1] = (p[1] / p[2]) * rho * theta;
      xyz[2] = rho * gpuAccurateSin(theta);
      gpuFtableRotate(matrix, xyz, 1);
      gpuFtableRotate(matrix, p, 1);
      part[0] += xyz[0];
      part[2] += xyz[1];
      part[4] += sqrt((rho * theta) * (rho * theta) + xyz[1] * xyz[1]);
      part[1] = p[0] / p[2];
      part[3] = p[1] / p[2];
    } else {
      part[0] += part[1] * step;
      part[2] += part[3] * step;
      part[4] += step * factor;
    }
  }

  if (data.doTilt) {
    double x = part[0], xp = part[1], y = part[2], yp = part[3];
    part[0] = x * data.cosUntilt + y * data.sinUntilt;
    part[2] = -x * data.sinUntilt + y * data.cosUntilt;
    part[1] = xp * data.cosUntilt + yp * data.sinUntilt;
    part[3] = -xp * data.sinUntilt + yp * data.cosUntilt;
  }
  if (data.doOffset) {
    part[4] += -data.dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - -data.dx + -data.dz * part[1];
    part[2] = part[2] - -data.dy + -data.dz * part[3];
  }
  /* ftable_frame_converter(exit) */
  if (data.exit == 1) {
    part[0] += part[1] * -data.l2;
    part[2] += part[3] * -data.l2;
    part[4] += -data.l2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  } else if (data.exit == 2) {
    double x0 = part[0], xp0 = part[1], y0 = part[2], yp0 = part[3], length, temp;
    length = data.cosE2 * data.l2 - data.sinE2 * x0;
    temp = data.cosE2 - data.sinE2 * xp0;
    part[1] = (data.cosE2 * xp0 + data.sinE2) / temp;
    part[3] = yp0 / temp;
    part[4] -= length * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = data.cosE2 * x0 + data.dxExit - part[1] * length;
    part[2] = y0 - part[3] * length;
  }
}

__device__ __forceinline__ int gpuBmxyzInterpolate(
  double *F0, double *F1, double *F2, double x, double y, double z,
  const GPU_BMXYZ_DATA &data) {
  long ix, iy, iz;
  double fx, fy, fz;
  double result[3];
  double position[3] = {x, y, z};
  long grid[3];

  for (long dimension = 0; dimension < 3; dimension++)
    grid[dimension] = static_cast<long>(
      (position[dimension] - data.minimum[dimension]) /
      data.spacing[dimension]);
  ix = grid[0];
  iy = grid[1];
  iz = grid[2];
  *F0 = *F1 = *F2 = 0.0;
  if (ix < 0 || iy < 0 || iz < 0 ||
      ix >= data.dimensions[0] - 1 ||
      iy >= data.dimensions[1] - 1 ||
      iz >= data.dimensions[2] - 1)
    return 0;

  fx = (x - (ix * data.spacing[0] + data.minimum[0])) /
       data.spacing[0];
  fy = (y - (iy * data.spacing[1] + data.minimum[1])) /
       data.spacing[1];
  fz = (z - (iz * data.spacing[2] + data.minimum[2])) /
       data.spacing[2];
  for (long field = 0; field < 3; field++) {
    double interp1[2][2], interp2[2];
    const double *values = data.field[field];
    long nx = data.dimensions[0];
    long nxy = nx * data.dimensions[1];

    interp1[0][0] =
      (1.0 - fz) * values[ix + iy * nx + iz * nxy] +
      fz * values[ix + iy * nx + (iz + 1) * nxy];
    interp1[1][0] =
      (1.0 - fz) * values[ix + 1 + iy * nx + iz * nxy] +
      fz * values[ix + 1 + iy * nx + (iz + 1) * nxy];
    interp1[0][1] =
      (1.0 - fz) * values[ix + (iy + 1) * nx + iz * nxy] +
      fz * values[ix + (iy + 1) * nx + (iz + 1) * nxy];
    interp1[1][1] =
      (1.0 - fz) * values[ix + 1 + (iy + 1) * nx + iz * nxy] +
      fz * values[ix + 1 + (iy + 1) * nx + (iz + 1) * nxy];
    interp2[0] = (1.0 - fy) * interp1[0][0] + fy * interp1[0][1];
    interp2[1] = (1.0 - fy) * interp1[1][0] + fy * interp1[1][1];
    result[field] =
      data.strengthFactor *
      ((1.0 - fx) * interp2[0] + fx * interp2[1]);
  }
  *F0 = result[0];
  *F1 = result[1];
  *F2 = result[2];
  return 1;
}

/* bmapxyz_deriv_function() checks the aperture at every derivative, trial points included, and
 * keeps the first point outside as the lost particle's coordinates */
typedef struct GPU_BMXYZ_LOSS {
  int lost;
  double state[8];
} GPU_BMXYZ_LOSS;

/* 1 for a derivative, 0 for a coordinate that isn't finite, 2 for a point outside the aperture */
__device__ __forceinline__ int gpuBmxyzDerivatives(
  double derivative[8], const double q[8],
  const GPU_BMXYZ_DATA &data, GPU_BMXYZ_LOSS *loss = NULL) {
  double F0, F1, F2;

  if (!isfinite(q[1]) || !isfinite(q[2]))
    return 0;
  if (loss && data.apertures &&
      (!gpuInsideApertureLimit(q[1], q[2], data.aperture) ||
       (data.contour.nContours &&
        !gpuInsideApertureContour(q[1], q[2], data.aperture, data.contour)))) {
    loss->lost = 1;
    for (int i = 0; i < 8; i++)
      loss->state[i] = q[i];
    return 2;
  }
  derivative[0] = q[3];
  derivative[1] = q[4];
  derivative[2] = q[5];
  derivative[6] = 1.0;
  derivative[7] = 0.0;
  if (!gpuBmxyzInterpolate(&F0, &F1, &F2, q[1], q[2], q[0], data)) {
    derivative[3] = derivative[4] = derivative[5] = 0.0;
    return 1;
  }
  derivative[3] = (q[4] * F2 - q[5] * F1) / (1.0 + q[7]);
  derivative[4] = (q[5] * F0 - q[3] * F2) / (1.0 + q[7]);
  derivative[5] = (q[3] * F1 - q[4] * F0) / (1.0 + q[7]);
  if (data.fieldIsMagnetic) {
    derivative[3] *= data.fieldScale;
    derivative[4] *= data.fieldScale;
    derivative[5] *= data.fieldScale;
  }
  return 1;
}

__device__ __forceinline__ int gpuBmxyzRk4Step(
  double finalState[8], const double initialState[8],
  const double initialDerivative[8], double h,
  const GPU_BMXYZ_DATA &data, GPU_BMXYZ_LOSS *loss = NULL) {
  double k1[8], k2[8], k3[8], temporary[8], derivative[8];
  int code;

  for (long equation = 0; equation < 8; equation++) {
    k1[equation] = h * initialDerivative[equation];
    temporary[equation] = initialState[equation] + k1[equation] / 2.0;
  }
  if ((code = gpuBmxyzDerivatives(derivative, temporary, data, loss)) != 1)
    return code;
  for (long equation = 0; equation < 8; equation++) {
    k2[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k2[equation] / 2.0;
  }
  if ((code = gpuBmxyzDerivatives(derivative, temporary, data, loss)) != 1)
    return code;
  for (long equation = 0; equation < 8; equation++) {
    k3[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k3[equation];
  }
  if ((code = gpuBmxyzDerivatives(derivative, temporary, data, loss)) != 1)
    return code;
  for (long equation = 0; equation < 8; equation++)
    finalState[equation] = initialState[equation] +
      (k1[equation] / 2.0 + k2[equation] + k3[equation] +
       h * derivative[equation] / 2.0) / 3.0;
  return 1;
}

__device__ __forceinline__ int gpuBmxyzSign(double value) {
  return value < 0.0 ? -1 : (value > 0.0 ? 1 : 0);
}

/* 1 at the exit, 0 for what only the CPU follows, 2 if lost at an aperture (loss) */
__device__ int gpuBmxyzIntegrate(double state[8],
                                 const GPU_BMXYZ_DATA &data, GPU_BMXYZ_LOSS *loss = NULL) {
  double y0[8], y1[8], derivative0[8], derivative1[8];
  int code;
  double x0 = 0.0, x1, ex0, ex1;
  double xf = 2.0 * data.fieldLength;
  double hStep = data.fieldLength * data.integrationAccuracy;
  double exitAccuracy = data.integrationAccuracy *
                        data.integrationAccuracy * xf;
  long maxSteps;

  if (exitAccuracy < data.fieldLength * 1e-14)
    exitAccuracy = data.fieldLength * 1e-14;
  maxSteps = static_cast<long>(ceil(xf / hStep)) + 2;
  for (long equation = 0; equation < 8; equation++)
    y0[equation] = state[equation];
  if ((code = gpuBmxyzDerivatives(derivative0, y0, data, loss)) != 1)
    return code;
  ex0 = data.fieldLength - y0[0];

  for (long step = 0; step < maxSteps; step++) {
    double xdiff;
    if (fabs(ex0) < exitAccuracy) {
      for (long equation = 0; equation < 8; equation++)
        state[equation] = y0[equation];
      return 1;
    }
    xdiff = xf - x0;
    if (xdiff < hStep)
      hStep = xdiff;
    x1 = x0;
    if ((code = gpuBmxyzRk4Step(y1, y0, derivative0, hStep, data, loss)) != 1)
      return code;
    x1 += hStep;
    if ((code = gpuBmxyzDerivatives(derivative1, y1, data, loss)) != 1)
      return code;
    ex1 = data.fieldLength - y1[0];
    if (gpuBmxyzSign(ex0) != gpuBmxyzSign(ex1))
      break;
    if (fabs(xf - x1) < exitAccuracy)
      return 0;
    for (long equation = 0; equation < 8; equation++) {
      y0[equation] = y1[equation];
      derivative0[equation] = derivative1[equation];
    }
    ex0 = ex1;
    x0 = x1;
  }

  if (fabs(ex1) < exitAccuracy) {
    for (long equation = 0; equation < 8; equation++)
      state[equation] = y1[equation];
    return 1;
  }
  for (long iteration = 0; iteration <= 400; iteration++) {
    double y2[8], derivative2[8];
    double h = -ex0 * (x1 - x0) / (ex1 - ex0) * 0.995;
    double x2 = x0;
    double ex2;
    if ((code = gpuBmxyzRk4Step(y2, y0, derivative0, h, data, loss)) != 1)
      return code;
    x2 += h;
    if ((code = gpuBmxyzDerivatives(derivative2, y2, data, loss)) != 1)
      return code;
    ex2 = data.fieldLength - y2[0];
    if (fabs(ex2) < exitAccuracy) {
      for (long equation = 0; equation < 8; equation++)
        state[equation] = y2[equation];
      return 1;
    }
    if (gpuBmxyzSign(ex1) == gpuBmxyzSign(ex2)) {
      for (long equation = 0; equation < 8; equation++) {
        y1[equation] = y2[equation];
        derivative1[equation] = derivative2[equation];
      }
      x1 = x2;
      ex1 = ex2;
    } else {
      for (long equation = 0; equation < 8; equation++) {
        y0[equation] = y2[equation];
        derivative0[equation] = derivative2[equation];
      }
      x0 = x2;
      ex0 = ex2;
    }
  }
  return 0;
}

/* With apertures, lossRound[] gets each particle's round for gpuBuildCpuRoundLossOrderDestination(),
 * lorentz() losing them in one, and lostCount the number lost */
__global__ void gpuBmxyzKernel(double *coord, long nParticles, int stride,
                               GPU_BMXYZ_DATA data,
                               unsigned long long *failedCount,
                               int *lossRound, unsigned long long *lostCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double state[8];
  double *part;
  double dzds;
  GPU_BMXYZ_LOSS loss;
  int code;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  dzds = 1.0 / sqrt(1.0 + part[1] * part[1] + part[3] * part[3]);
  state[0] = 0.0;
  state[1] = part[0];
  state[2] = part[2];
  state[3] = dzds;
  state[4] = part[1] * dzds;
  state[5] = part[3] * dzds;
  state[6] = part[4];
  state[7] = part[5];
  loss.lost = 0;
  code = gpuBmxyzIntegrate(state, data, lossRound ? &loss : NULL);
  if (lossRound)
    lossRound[ip] = code == 2 ? 0 : -1;
  if (code == 0) {
    atomicAdd(failedCount, 1ULL);
    return;
  }
  if (code == 2) {
    /* bmapxyz_coord_transform() of the first point outside; z where it was; p as lorentz() leaves it */
    part[0] = loss.state[1];
    part[1] = loss.state[4] / loss.state[3];
    part[2] = loss.state[2];
    part[3] = loss.state[5] / loss.state[3];
    part[4] = data.lossZStart + loss.state[0];
    part[5] = data.pCentral * (1 + loss.state[7]);
    atomicAdd(lostCount, 1ULL);
    return;
  }
  part[0] = state[1];
  part[1] = state[4] / state[3];
  part[2] = state[2];
  part[3] = state[5] / state[3];
  part[4] = state[6];
  part[5] = state[7];
}

__device__ __forceinline__ double gpuLorentzExitValue(
  const double state[8], const GPU_LORENTZ_DATA &data) {
  if (data.type == GPU_LORENTZ_BMAPXY)
    return data.length - state[0];
  if (data.type == GPU_LORENTZ_NIBEND)
    return state[1] * data.exitSlope + data.exitIntercept - state[0];
  return data.exitIntercept - state[0];
}

__device__ __forceinline__ int gpuLorentzDerivatives(
  double derivative[8], double state[8], const GPU_LORENTZ_DATA &data) {
  const double *velocity = state + 3;
  double F0 = 0.0, F1 = 0.0, F2 = 0.0;

  if (!isfinite(state[1]) || !isfinite(state[2]))
    return 0;
  derivative[0] = velocity[0];
  derivative[1] = velocity[1];
  derivative[2] = velocity[2];
  derivative[6] = 1.0;
  derivative[7] = 0.0;

  if (data.type == GPU_LORENTZ_BMAPXY) {
    long ix = static_cast<long>((state[1] - data.xmin) / data.dx);
    long iy = static_cast<long>((state[2] - data.ymin) / data.dy);
    if (ix >= 0 && iy >= 0 && ix < data.nx - 1 && iy < data.ny - 1) {
      double fx = (state[1] - (ix * data.dx + data.xmin)) / data.dx;
      double fy = (state[2] - (iy * data.dy + data.ymin)) / data.dy;
      double Fa = (1.0 - fy) * data.field[0][ix + iy * data.nx] +
                  fy * data.field[0][ix + (iy + 1) * data.nx];
      double Fb = (1.0 - fy) * data.field[0][ix + 1 + iy * data.nx] +
                  fy * data.field[0][ix + 1 + (iy + 1) * data.nx];
      F1 = data.strengthFactor * ((1.0 - fx) * Fa + fx * Fb);
      Fa = (1.0 - fy) * data.field[1][ix + iy * data.nx] +
           fy * data.field[1][ix + (iy + 1) * data.nx];
      Fb = (1.0 - fy) * data.field[1][ix + 1 + iy * data.nx] +
           fy * data.field[1][ix + 1 + (iy + 1) * data.nx];
      F2 = data.strengthFactor * ((1.0 - fx) * Fa + fx * Fb);
      if (data.fieldIsMagnetic) {
        F1 *= data.fieldScale;
        F2 *= data.fieldScale;
      }
    }
  } else if (data.type == GPU_LORENTZ_NIBEND) {
    double S = data.strengthFactor / (1.0 + state[7]);
    double dq0 = state[0] -
      (state[1] * data.entranceSlope + data.fringeEntranceIntercept);
    double z = dq0 * data.cosAlpha1;
    if (z >= 0.0) {
      dq0 = state[0] -
        (state[1] * data.exitSlope + data.fringeExitIntercept);
      z = dq0 * data.cosAlpha2;
      if (z <= 0.0)
        F2 = S;
    }
  } else if (data.type == GPU_LORENTZ_NISEPT) {
    double S = data.strengthFactor / (1.0 + state[7]);
    double dq0 = state[0] - data.entranceIntercept;
    double dq1 = state[1] - (data.q1Reference + data.q1Offset);
    if (dq0 > 0.0) {
      if (data.fringeLength && dq0 <= data.fringeLength) {
        S /= data.fringeLength;
        F0 = S * state[2] * (1.0 + data.b1 * dq1);
        F1 = S * state[2] * dq0 * data.b1;
        F2 = S * dq0 * (1.0 + data.b1 * dq1);
      } else {
        dq0 = state[0] - data.fringeExitIntercept;
        if (dq0 < data.fringeLength) {
          if (data.fringeLength && dq0 > 0.0) {
            S /= data.fringeLength;
            F0 = -S * state[2] * (1.0 + data.b1 * dq1);
            F1 = S * state[2] * (data.fringeLength - dq0) * data.b1;
            F2 = S * (data.fringeLength - dq0) *
                 (1.0 + data.b1 * dq1);
          } else {
            F1 = S * data.b1 * state[2];
            F2 = S * (1.0 + data.b1 * dq1);
          }
        }
      }
    }
  } else {
    return 0;
  }

  derivative[3] = velocity[1] * F2 - velocity[2] * F1;
  derivative[4] = velocity[2] * F0 - velocity[0] * F2;
  derivative[5] = velocity[0] * F1 - velocity[1] * F0;
  if (data.type == GPU_LORENTZ_BMAPXY) {
    double denominator = 1.0 + state[7];
    derivative[3] /= denominator;
    derivative[4] /= denominator;
    derivative[5] /= denominator;
  }
  return isfinite(derivative[3]) && isfinite(derivative[4]) &&
         isfinite(derivative[5]);
}

__device__ __forceinline__ int gpuLorentzRk4Step(
  double finalState[8], const double initialState[8],
  const double initialDerivative[8], double h,
  const GPU_LORENTZ_DATA &data) {
  double k1[8], k2[8], k3[8], temporary[8], derivative[8];

  for (long equation = 0; equation < 8; equation++) {
    k1[equation] = h * initialDerivative[equation];
    temporary[equation] = initialState[equation] + k1[equation] / 2.0;
  }
  if (!gpuLorentzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++) {
    k2[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k2[equation] / 2.0;
  }
  if (!gpuLorentzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++) {
    k3[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k3[equation];
  }
  if (!gpuLorentzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++)
    finalState[equation] = initialState[equation] +
      (k1[equation] / 2.0 + k2[equation] + k3[equation] +
       h * derivative[equation] / 2.0) / 3.0;
  return 1;
}

__device__ int gpuLorentzIntegrate(double state[8],
                                   const GPU_LORENTZ_DATA &data) {
  double y0[8], y1[8], derivative0[8], derivative1[8];
  double x0 = 0.0, x1 = 0.0, ex0, ex1 = 0.0;
  double xf = 2.0 * data.length;
  double hStep = data.length * data.integrationAccuracy;
  double exitAccuracy = data.integrationAccuracy *
                        data.integrationAccuracy * xf;
  long maxSteps;

  if (exitAccuracy < data.length * 1e-14)
    exitAccuracy = data.length * 1e-14;
  maxSteps = static_cast<long>(ceil(xf / hStep)) + 2;
  for (long equation = 0; equation < 8; equation++)
    y0[equation] = state[equation];
  if (!gpuLorentzDerivatives(derivative0, y0, data))
    return 0;
  ex0 = gpuLorentzExitValue(y0, data);

  for (long step = 0; step < maxSteps; step++) {
    double xdiff;
    if (fabs(ex0) < exitAccuracy) {
      for (long equation = 0; equation < 8; equation++)
        state[equation] = y0[equation];
      return 1;
    }
    xdiff = xf - x0;
    if (xdiff < hStep)
      hStep = xdiff;
    x1 = x0;
    if (!gpuLorentzRk4Step(y1, y0, derivative0, hStep, data))
      return 0;
    x1 += hStep;
    if (!gpuLorentzDerivatives(derivative1, y1, data))
      return 0;
    ex1 = gpuLorentzExitValue(y1, data);
    if (gpuBmxyzSign(ex0) != gpuBmxyzSign(ex1))
      break;
    if (fabs(xf - x1) < exitAccuracy)
      return 0;
    for (long equation = 0; equation < 8; equation++) {
      y0[equation] = y1[equation];
      derivative0[equation] = derivative1[equation];
    }
    ex0 = ex1;
    x0 = x1;
  }

  if (fabs(ex1) < exitAccuracy) {
    for (long equation = 0; equation < 8; equation++)
      state[equation] = y1[equation];
    return 1;
  }
  for (long iteration = 0; iteration <= 400; iteration++) {
    double y2[8], derivative2[8];
    double h = -ex0 * (x1 - x0) / (ex1 - ex0) * 0.995;
    double x2 = x0;
    double ex2;
    if (!gpuLorentzRk4Step(y2, y0, derivative0, h, data))
      return 0;
    x2 += h;
    if (!gpuLorentzDerivatives(derivative2, y2, data))
      return 0;
    ex2 = gpuLorentzExitValue(y2, data);
    if (fabs(ex2) < exitAccuracy) {
      for (long equation = 0; equation < 8; equation++)
        state[equation] = y2[equation];
      return 1;
    }
    if (gpuBmxyzSign(ex1) == gpuBmxyzSign(ex2)) {
      for (long equation = 0; equation < 8; equation++) {
        y1[equation] = y2[equation];
        derivative1[equation] = derivative2[equation];
      }
      x1 = x2;
      ex1 = ex2;
    } else {
      for (long equation = 0; equation < 8; equation++) {
        y0[equation] = y2[equation];
        derivative0[equation] = derivative2[equation];
      }
      x0 = x2;
      ex0 = ex2;
    }
  }
  return 0;
}

__device__ __forceinline__ int gpuLorentzTransformIn(
  double state[8], const double *part, const GPU_LORENTZ_DATA &data) {
  double dzds = 1.0 / sqrt(1.0 + part[1] * part[1] + part[3] * part[3]);
  double dxds = part[1] * dzds;
  double dyds = part[3] * dzds;

  if (data.type == GPU_LORENTZ_BMAPXY) {
    state[0] = 0.0;
    state[1] = part[0];
    state[2] = part[2];
    state[3] = dzds;
    state[4] = dxds;
    state[5] = dyds;
    state[6] = part[4];
    state[7] = part[5];
    return 1;
  }
  if (data.type == GPU_LORENTZ_NIBEND) {
    double halfAngle = data.angle / 2.0;
    double sinHalf = sin(halfAngle), cosHalf = cos(halfAngle);
    double alpha = halfAngle - data.e1;
    double q0[3], velocity[3], q0Intersection, ds;
    q0[0] = -(data.rho + part[0]) * sinHalf;
    q0[1] = (data.rho + part[0]) * cosHalf;
    q0[2] = part[2];
    velocity[0] = dzds * cosHalf - dxds * sinHalf;
    velocity[1] = dzds * sinHalf + dxds * cosHalf;
    velocity[2] = dyds;
    q0Intersection = data.rho * (cosHalf * tan(alpha) - sinHalf);
    ds = (-q0[1] * tan(alpha) + q0Intersection - q0[0]) /
         (velocity[0] + velocity[1] * tan(alpha));
    state[0] = q0[0] + velocity[0] * ds;
    state[1] = q0[1] + velocity[1] * ds;
    state[2] = q0[2] + velocity[2] * ds;
    state[3] = velocity[0];
    state[4] = velocity[1];
    state[5] = velocity[2];
    state[6] = part[4] + ds;
    state[7] = part[5];
    return isfinite(ds);
  }
  if (data.type == GPU_LORENTZ_NISEPT) {
    double sinE1 = sin(data.e1), cosE1 = cos(data.e1);
    double q0[3], velocity[3], ds;
    q0[0] = -(data.rho + part[0]) * sinE1;
    q0[1] = (data.rho + part[0]) * cosE1;
    q0[2] = part[2];
    velocity[0] = dzds * cosE1 - dxds * sinE1;
    velocity[1] = dzds * sinE1 + dxds * cosE1;
    velocity[2] = dyds;
    ds = (data.entranceIntercept - q0[0]) / velocity[0];
    state[0] = q0[0] + velocity[0] * ds;
    state[1] = q0[1] + velocity[1] * ds;
    state[2] = q0[2] + velocity[2] * ds;
    state[3] = velocity[0];
    state[4] = velocity[1];
    state[5] = velocity[2];
    state[6] = part[4] + ds;
    state[7] = part[5];
    return isfinite(ds);
  }
  return 0;
}

__device__ __forceinline__ int gpuLorentzTransformOut(
  double *part, const double state[8], const GPU_LORENTZ_DATA &data) {
  if (data.type == GPU_LORENTZ_BMAPXY) {
    part[0] = state[1];
    part[1] = state[4] / state[3];
    part[2] = state[2];
    part[3] = state[5] / state[3];
    part[4] = state[6];
    part[5] = state[7];
    return isfinite(part[1]) && isfinite(part[3]);
  }
  if (data.type == GPU_LORENTZ_NIBEND) {
    double halfAngle = data.angle / 2.0;
    double tanHalf = tan(halfAngle), sinHalf = sin(halfAngle),
           cosHalf = cos(halfAngle);
    double ds = -(state[0] - state[1] * tanHalf) /
                (state[3] - state[4] * tanHalf);
    double q0 = state[0] + ds * state[3];
    double q1 = state[1] + ds * state[4];
    double q2 = state[2] + ds * state[5];
    double dzds = state[3] * cosHalf - state[4] * sinHalf;
    double dxds = state[3] * sinHalf + state[4] * cosHalf;
    part[0] = sqrt(q0 * q0 + q1 * q1) - data.rho;
    part[1] = dxds / dzds;
    part[2] = q2;
    part[3] = state[5] / dzds;
    part[4] = state[6] + ds;
    part[5] = state[7];
    return isfinite(ds) && isfinite(part[1]) && isfinite(part[3]);
  }
  if (data.type == GPU_LORENTZ_NISEPT) {
    double tanE2 = tan(data.e2), sinE2 = sin(data.e2),
           cosE2 = cos(data.e2);
    double ds = -(state[0] - state[1] * tanE2) /
                (state[3] - state[4] * tanE2);
    double q0 = state[0] + ds * state[3];
    double q1 = state[1] + ds * state[4];
    double q2 = state[2] + ds * state[5];
    double dzds = state[3] * cosE2 - state[4] * sinE2;
    double dxds = state[3] * sinE2 + state[4] * cosE2;
    part[0] = sqrt(q0 * q0 + q1 * q1) - data.rho;
    part[1] = dxds / dzds;
    part[2] = q2;
    part[3] = state[5] / dzds;
    part[4] = state[6] + ds;
    part[5] = state[7];
    return isfinite(ds) && isfinite(part[1]) && isfinite(part[3]);
  }
  return 0;
}

__global__ void gpuLorentzKernel(double *coord, long nParticles, int stride,
                                 GPU_LORENTZ_DATA data,
                                 unsigned long long *failedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double state[8];
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (!gpuLorentzTransformIn(state, part, data) ||
      !gpuLorentzIntegrate(state, data) ||
      !gpuLorentzTransformOut(part, state, data))
    atomicAdd(failedCount, 1ULL);
}

__global__ void gpuRfcwRfOnlyMatrixKernel(double *coord, long nParticles, int stride,
                                          double pCentral, double length,
                                          double volt, double omega,
                                          double phase, int end1Focus,
                                          int end2Focus, double dx, double dy,
                                          double cMks, long *lostCount,
                                          const double *referenceKick,
                                          const double *referenceR12) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, gamma, beta0, ds1, t, dgamma, gamma1;
  double dP, R12, R22, x, xp, y, yp, inverseF;

  if (ip >= nParticles)
    return;

  part = coord + ip * stride;
  if (part[5] == -1) {
    if (lostCount)
      atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), 1ULL);
    return;
  }

  part[0] -= dx;
  part[2] -= dy;

  p = pCentral * (1 + part[5]);
  gamma = sqrt(p * p + 1);
  beta0 = p / gamma;
  ds1 = length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  t = (part[4] + ds1) / (cMks * beta0);
  dgamma = referenceKick ? referenceKick[ip] : volt * sin(omega * t + phase);

  if (end1Focus && length) {
    inverseF = dgamma / (2 * gamma * length);
    part[1] -= part[0] * inverseF;
    part[3] -= part[2] * inverseF;
  }

  gamma1 = gamma + dgamma;
  dP = sqrt(gamma1 * gamma1 - 1) - p;
  R22 = 1 / (1 + dP / p);
  if (referenceR12)
    R12 = referenceR12[ip];
  else if (fabs(dP / p) > 1e-14)
    R12 = length * (p / dP * log(1 + dP / p));
  else
    R12 = length;

  part[4] += ds1;
  x = part[0];
  xp = part[1];
  part[0] = x + xp * R12;
  part[1] = xp * R22;
  y = part[2];
  yp = part[3];
  part[2] = y + yp * R12;
  part[3] = yp * R22;
  part[4] += length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  part[5] = (p + dP - pCentral) / pCentral;

  if (gamma1 <= 1)
    part[5] = -1;
  if (end2Focus && length) {
    inverseF = -dgamma / (2 * gamma1 * length);
    part[1] -= part[0] * inverseF;
    part[3] -= part[2] * inverseF;
  }
  part[4] = (p + dP) / gamma1 * part[4] / beta0;
  part[0] += dx;
  part[2] += dy;
  if (lostCount) {
    int lost = part[5] <= -1;
    for (int ic = 0; ic < 6 && !lost; ic++)
      lost = isnan(part[ic]);
    if (lost)
      atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), 1ULL);
  }
}

/* timeShift is simple_rfca.c's ik*dtLight for a traveling wave, taken from
 * the time before the phase is made, as the CPU does */
__global__ void gpuRfcwKickInitialKernel(double *coord, double *inverseF,
                                         long nParticles, int stride,
                                         double pCentral, double length,
                                         double volt, double omega,
                                         double phase, double timeShift,
                                         int end1Focus, double cMks,
                                         const double *referenceKick) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, gamma, beta, dc4, t, dgamma, gamma1Raw, gamma1;
  double p1, pz, pz1, pRatio, invF;

  if (ip >= nParticles)
    return;

  inverseF[ip] = 0;
  part = coord + ip * stride;
  if (part[5] == -1)
    return;

  p = pCentral * (1 + part[5]);
  gamma = sqrt(p * p + 1);
  beta = p / gamma;
  dc4 = length ? length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]) : 0;
  t = (part[4] + dc4) / (cMks * beta);
  dgamma = referenceKick ? referenceKick[ip] : volt * sin(omega * (t - timeShift) + phase);

  if (length) {
    if (end1Focus) {
      invF = dgamma / (2 * gamma * length);
      part[1] -= part[0] * invF;
      part[3] -= part[2] * invF;
    }
    part[0] += part[1] * length / 2;
    part[2] += part[3] * length / 2;
    part[4] += length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  }

  gamma1Raw = gamma + dgamma;
  gamma1 = gamma1Raw;
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - pCentral) / pCentral;
  part[4] = t * cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
  if (gamma1Raw <= 1)
    part[5] = -1;
  else if (length)
    inverseF[ip] = -dgamma / (2 * gamma1Raw * length);
}

__global__ void gpuRfcwKickFinalKernel(double *coord, const double *inverseF,
                                       long nParticles, int stride,
                                       double length, int end2Focus) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, invF;

  if (ip >= nParticles)
    return;

  part = coord + ip * stride;
  if (!length)
    return;
  part[0] += part[1] * length / 2;
  part[2] += part[3] * length / 2;
  part[4] += length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  if (end2Focus) {
    invF = inverseF[ip];
    part[1] -= part[0] * invF;
    part[3] -= part[2] * invF;
  }
}

__global__ void gpuRfcwDgammaOverGammaSumsKernel(double *coord, long nParticles,
                                                 int stride, double pCentral,
                                                 double length, double volt,
                                                 double omega, double phase,
                                                 double timeShift, double cMks,
                                                 GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  sum[tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double p, gamma, beta, ds, t, dgamma;

    if (part[5] == -1)
      continue;
    p = pCentral * (1 + part[5]);
    gamma = sqrt(p * p + 1);
    if (gamma == 0)
      continue;
    beta = p / gamma;
    ds = length / 2 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    t = (part[4] + ds) / (cMks * beta);
    dgamma = volt * sin(omega * (t - timeShift) + phase);
    count[tid]++;
    sum[tid] += dgamma / gamma;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      sum[tid] += sum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->centroidSum[6] = sum[0];
  }
}

__global__ void gpuSubtractCoordinateKernel(double *coord, long nParticles, int stride,
                                            int index, double value) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  coord[ip * stride + index] -= value;
}

__global__ void gpuCenterBeamKernel(double *coord, long nParticles, int stride,
                                    unsigned int coordinateMask,
                                    double offset0, double offset1, double offset2,
                                    double offset3, double offset4, double offset5,
                                    int doTime, double pCentral, double timeOffset,
                                    double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, beta;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (coordinateMask & 0x01u)
    part[0] -= offset0;
  if (coordinateMask & 0x02u)
    part[1] -= offset1;
  if (coordinateMask & 0x04u)
    part[2] -= offset2;
  if (coordinateMask & 0x08u)
    part[3] -= offset3;
  if (coordinateMask & 0x10u)
    part[4] -= offset4;
  if (coordinateMask & 0x20u)
    part[5] -= offset5;
  if (doTime) {
    p = pCentral * (1 + part[5]);
    beta = p / sqrt(p * p + 1);
    part[4] -= cMks * beta * timeOffset;
  }
}

__device__ __forceinline__ int gpuLimitAmplitudeLost(double *part, double xmax,
                                                     double ymax, int openCode) {
  int lost = 0;

  if (xmax > 0 && fabs(part[0]) > xmax)
    lost = 1;
  if (ymax > 0 && fabs(part[2]) > ymax)
    lost = 1;
  if (lost && openCode && !gpuApertureOpenSideLoses(openCode, part[0], part[2], xmax, ymax))
    lost = 0;
  if (!isfinite(part[0]) || !isfinite(part[2]))
    lost = 1;
  return lost;
}

__global__ void gpuLimitAmplitudeLossCountKernel(double *coord, long nParticles, int stride,
                                                 double xmax, double ymax, int openCode,
                                                 long *lostCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles && gpuLimitAmplitudeLost(coord + ip * stride, xmax, ymax, openCode))
    gpuLossCountAdd(lostCount);
}

__device__ __forceinline__ double gpuCoordinateSign(double value) {
  return value < 0 ? -1.0 : 1.0;
}

__global__ void gpuLimitAmplitudeSurvivorFlagKernel(double *coord,
                                                    long nParticles,
                                                    int stride,
                                                    double xmax,
                                                    double ymax,
                                                    int openCode,
                                                    long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] = gpuLimitAmplitudeLost(coord + ip * stride, xmax, ymax, openCode) ? 0 : 1;
}

__global__ void gpuLimitAmplitudesStableScatterKernel(double *coord,
                                                      double *scratch,
                                                      const long *survivorPrefix,
                                                      long nParticles,
                                                      int stride,
                                                      double xmax,
                                                      double ymax,
                                                      int openCode,
                                                      double z,
                                                      double pCentral,
                                                      long extrapolateZ,
                                                      long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost;
  double dz = 0;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  lost = gpuLimitAmplitudeLost(part, xmax, ymax, openCode);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (!lost)
    return;

  if (extrapolateZ && isfinite(part[0]) && isfinite(part[2])) {
    double dzx = -DBL_MAX;
    double dzy = -DBL_MAX;

    if ((xmax > 0 && fabs(part[0]) > xmax) && part[1] != 0)
      dzx = (part[0] - gpuCoordinateSign(part[1]) * xmax) / part[1];
    if ((ymax > 0 && fabs(part[2]) > ymax) && part[3] != 0)
      dzy = (part[2] - gpuCoordinateSign(part[3]) * ymax) / part[3];
    dz = dzx > dzy ? -dzx : -dzy;
    if (dz == -DBL_MAX)
      dz = 0;
  }
  target[0] += dz * target[1];
  target[2] += dz * target[3];
  target[4] = z + dz;
  target[5] = pCentral * (1 + target[5]);
}

__device__ __forceinline__ double gpuIntegerPower(double value, long exponent) {
  double result = 1;

  for (long i = 0; i < exponent; i++)
    result *= value;
  return result;
}

__device__ __forceinline__ int gpuELimitAmplitudeLost(double *part,
                                                      double xmax,
                                                      double ymax,
                                                      long exponent,
                                                      long yExponent,
                                                      int openCode) {
  double a2, b2, normalized;

  if (!isfinite(part[0]) || !isfinite(part[2]))
    return 1;
  a2 = gpuIntegerPower(xmax, exponent);
  b2 = gpuIntegerPower(ymax, yExponent);
  normalized =
    gpuIntegerPower(part[0], exponent) / a2 +
    gpuIntegerPower(part[2], yExponent) / b2;
  if (!(normalized > 1))
    return 0;
  return !openCode || gpuApertureOpenSideLoses(openCode, part[0], part[2], xmax, ymax);
}

__global__ void gpuELimitAmplitudeLossCountKernel(double *coord, long nParticles, int stride,
                                                  double xmax, double ymax, long exponent,
                                                  long yExponent, int openCode, long *lostCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles &&
      gpuELimitAmplitudeLost(coord + ip * stride, xmax, ymax, exponent, yExponent, openCode))
    gpuLossCountAdd(lostCount);
}

__global__ void gpuELimitAmplitudeSurvivorFlagKernel(double *coord,
                                                     long nParticles,
                                                     int stride,
                                                     double xmax,
                                                     double ymax,
                                                     long exponent,
                                                     long yExponent,
                                                     int openCode,
                                                     long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuELimitAmplitudeLost(coord + ip * stride, xmax, ymax,
                           exponent, yExponent, openCode) ? 0 : 1;
}

__global__ void gpuELimitAmplitudesStableScatterKernel(double *coord,
                                                       double *scratch,
                                                       const long *survivorPrefix,
                                                       long nParticles,
                                                       int stride,
                                                       double xmax,
                                                       double ymax,
                                                       long exponent,
                                                       long yExponent,
                                                       int openCode,
                                                       double z,
                                                       double pCentral,
                                                       long extrapolateZ,
                                                       long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost;
  double dz = 0;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  lost = gpuELimitAmplitudeLost(part, xmax, ymax, exponent, yExponent, openCode);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (!lost)
    return;

  if (isfinite(part[0]) && isfinite(part[2]) &&
      extrapolateZ && exponent == 2 && yExponent == 2) {
    double a2 = xmax * xmax;
    double b2 = ymax * ymax;
    double c0 = part[0] * part[0] / a2 + part[2] * part[2] / b2 - 1;
    double c1 = 2 * (part[0] * part[1] / a2 + part[2] * part[3] / b2);
    double c2 = part[1] * part[1] / a2 + part[3] * part[3] / b2;
    double det = c1 * c1 - 4 * c0 * c2;

    if (z > 0 && c2 != 0 && det >= 0) {
      double root = sqrt(det);
      dz = (-c1 + root) / (2 * c2);
      if (dz > 0)
        dz = (-c1 - root) / (2 * c2);
      if (z + dz < 0)
        dz = -z;
      target[0] += dz * target[1];
      target[2] += dz * target[3];
    }
  }
  target[4] = z + dz;
  target[5] = pCentral * (1 + target[5]);
}

__device__ __forceinline__ int gpuInvalidParticleLost(double *part) {
  for (int ic = 0; ic < 6; ic++) {
    if (isnan(part[ic]))
      return 1;
  }
  return part[5] <= -1;
}

__global__ void gpuRemoveInvalidParticlesLossCountKernel(double *coord,
                                                         long nParticles,
                                                         int stride,
                                                         long *lostCount) {
  long localCount = 0;

  for (long ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles; ip = nParticles) {
    double *part = coord + ip * stride;

    localCount += gpuInvalidParticleLost(part);
  }

  if (localCount)
    atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), (unsigned long long)localCount);
}

__global__ void gpuRemoveInvalidParticlesSurvivorFlagKernel(double *coord,
                                                            long nParticles,
                                                            int stride,
                                                            long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] = gpuInvalidParticleLost(coord + ip * stride) ? 0 : 1;
}

__global__ void gpuRemoveInvalidParticlesStableScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, double z, double pCentral, long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  lost = gpuInvalidParticleLost(part);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (!lost)
    return;
  target[4] = z;
  target[5] = pCentral * (1 + target[5]);
}

__global__ void gpuStableScatterRowsKernel(double *coord, double *scratch,
                                           const long *survivorPrefix,
                                           long nParticles, int stride,
                                           long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination;

  if (ip >= nParticles)
    return;
  destination = survivorPrefix[ip];
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
}

__device__ __forceinline__ int gpuEvaluateLostWithOpenSides(long code,
                                                            double dx,
                                                            double dy,
                                                            double xsize,
                                                            double ysize) {
  switch (code) {
  case GPU_OPEN_PLUS_X:
    return !(dx > 0 && (fabs(dy) < ysize || ysize <= 0));
  case GPU_OPEN_MINUS_X:
    return !(dx < 0 && (fabs(dy) < ysize || ysize <= 0));
  case GPU_OPEN_PLUS_Y:
    return !(dy > 0 && (fabs(dx) < xsize || xsize <= 0));
  case GPU_OPEN_MINUS_Y:
    return !(dy < 0 && (fabs(dx) < xsize || xsize <= 0));
  default:
    return 1;
  }
}

__device__ __forceinline__ int gpuRectangularCollimatorLostAtPosition(double x,
                                                                      double y,
                                                                      double xmax,
                                                                      double ymax,
                                                                      double xCenter,
                                                                      double yCenter,
                                                                      long openCode,
                                                                      int exitPlane) {
  double dx = x - xCenter;
  double dy = y - yCenter;

  if (!isfinite(x) || !isfinite(y))
    return 1;
  if (openCode && exitPlane) {
    int lost = 0;

    if (xmax > 0 && fabs(dx) > xmax)
      lost += gpuEvaluateLostWithOpenSides(openCode, dx, 0, xmax, ymax);
    if (ymax > 0 && fabs(dy) > ymax)
      lost += gpuEvaluateLostWithOpenSides(openCode, 0, dy, xmax, ymax);
    return lost != 0;
  }
  if ((xmax > 0 && fabs(dx) > xmax) ||
      (ymax > 0 && fabs(dy) > ymax))
    return openCode ? gpuEvaluateLostWithOpenSides(openCode, dx, dy, xmax, ymax) : 1;
  return 0;
}

__global__ void gpuRectangularCollimatorLossCountKernel(double *coord, long nParticles, int stride,
                                                        double xmax, double ymax,
                                                        double xCenter, double yCenter,
                                                        double length, long openCode,
                                                        long *lostCount) {
  const double boundaryGuard = 1e-12;
  long localCount = 0;

  for (long ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles; ip = nParticles) {
    double *part = coord + ip * stride;
    double x = part[0] + length * part[1];
    double y = part[2] + length * part[3];
    int lost = 0;

    if (!isfinite(x) || !isfinite(y)) {
      lost = 1;
    } else {
      double dx = x - xCenter;
      double dy = y - yCenter;

      if (openCode && length > 0) {
        if (xmax > 0 && fabs(dx) > xmax * (1 - boundaryGuard))
          lost += gpuEvaluateLostWithOpenSides(openCode, dx, 0, xmax, ymax);
        if (ymax > 0 && fabs(dy) > ymax * (1 - boundaryGuard))
          lost += gpuEvaluateLostWithOpenSides(openCode, 0, dy, xmax, ymax);
      } else if ((xmax > 0 && fabs(dx) > xmax * (1 - boundaryGuard)) ||
                 (ymax > 0 && fabs(dy) > ymax * (1 - boundaryGuard))) {
        lost = openCode ? gpuEvaluateLostWithOpenSides(openCode, dx, dy, xmax, ymax) : 1;
      }
    }
    localCount += lost;
  }

  if (localCount)
    atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), (unsigned long long)localCount);
}

__global__ void gpuEllipticalCollimatorLossCountKernel(double *coord, long nParticles, int stride,
                                                       double xmax, double ymax,
                                                       double xCenter, double yCenter,
                                                       long exponent, long yExponent,
                                                       double length,
                                                       long openCode,
                                                       long *lostCount) {
  const double boundaryGuard = 1e-12;
  long localCount = 0;

  for (long ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles; ip = nParticles) {
    double *part = coord + ip * stride;
    double x = part[0] + length * part[1];
    double y = part[2] + length * part[3];
    int lost = 0;

    if (!isfinite(x) || !isfinite(y)) {
      lost = 1;
    } else {
      double normalizedX = (x - xCenter) / xmax;
      double normalizedY = (y - yCenter) / ymax;
      long effectiveYExponent = length == 0 ? yExponent : exponent;
      double normalized =
        gpuIntegerPower(normalizedX, exponent) +
        gpuIntegerPower(normalizedY, effectiveYExponent);

      if (normalized > 1 - boundaryGuard)
        lost = openCode ? gpuEvaluateLostWithOpenSides(openCode, normalizedX, normalizedY, 1, 1) :
                          1;
    }
    localCount += lost ? 1 : 0;
  }

  if (localCount)
    atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), (unsigned long long)localCount);
}

__device__ __forceinline__ int gpuRectangularCollimatorLostAt(double *part,
                                                              double xmax,
                                                              double ymax,
                                                              double xCenter,
                                                              double yCenter,
                                                              double length,
                                                              long openCode) {
  /* Entrance checks do not advance the particle (in particular, avoid
   * turning a finite position into NaN via 0 * an infinite slope). */
  double x = length == 0 ? part[0] : part[0] + length * part[1];
  double y = length == 0 ? part[2] : part[2] + length * part[3];
  return gpuRectangularCollimatorLostAtPosition(x, y, xmax, ymax, xCenter, yCenter,
                                                openCode, length > 0);
}

__global__ void gpuBatchedApertureInitializeKernel(
  long lines, long nx, double xmax, double ymax,
  const double *dxFactor, const double *dyFactor,
  double *dx, double *dy, double *x0, double *y0,
  double *xLimit, double *yLimit,
  double *xLost, double *yLost, double *sLost,
  long *lossPass, long *lossElement, long *originStable) {
  long line = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (line >= lines)
    return;
  dx[line] = xmax / (nx - 1) * dxFactor[line];
  dy[line] = ymax / (nx - 1) * dyFactor[line];
  x0[line] = y0[line] = 0;
  xLimit[line] = yLimit[line] = 0;
  xLost[line] = yLost[line] = sLost[line] = DBL_MAX;
  lossPass[line] = lossElement[line] = -1;
  if (line == 0)
    *originStable = 1;
}

__global__ void gpuBatchedApertureRefineLinesKernel(
  long lines, double splitFraction, double *dx, double *dy,
  double *x0, double *y0, const double *xLimit, const double *yLimit) {
  long line = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (line >= lines)
    return;
  x0[line] = xLimit[line] + dx[line] * splitFraction;
  y0[line] = yLimit[line] + dy[line] * splitFraction;
  dx[line] *= splitFraction;
  dy[line] *= splitFraction;
}

__global__ void gpuBatchedAperturePrepareKernel(
  long lines, long nSteps, const double *orbit,
  const double *dx, const double *dy, const double *x0, const double *y0,
  double *trialCoord, long trialStride) {
  long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long trials = lines * nSteps;
  long line, step;
  double *part;

  if (id >= trials)
    return;
  line = id / nSteps;
  step = id - line * nSteps;
  part = trialCoord + id * trialStride;
  for (int ic = 0; ic < 6; ic++)
    part[ic] = orbit[ic];
  part[0] = x0[line] + step * dx[line] + orbit[0];
  part[2] = y0[line] + step * dy[line] + orbit[2];
}

__device__ __forceinline__ int gpuBatchedApertureMatrixDrift(
  double *part, const GPU_BATCHED_APERTURE_DRIFT_DATA *drift) {
  double xp = part[1], yp = part[3];
  double x = drift->length * xp + part[0];
  double y = drift->length * yp + part[2];
  double s = drift->length + part[4];

  if (drift->order > 1) {
    s += drift->length / 2 * yp * yp;
    s += drift->length / 2 * xp * xp;
  }
  part[0] = x;
  part[2] = y;
  part[4] = s;
  return isfinite(x) && isfinite(y);
}

__device__ __forceinline__ int gpuBatchedApertureMultipoleDrift(
  double *part, const GPU_BATCHED_APERTURE_DRIFT_DATA *drift) {
  double x = part[0], xp = part[1], y = part[2], yp = part[3];
  double pathFactor;

  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp) ||
      fabs(x) > drift->coordLimit || fabs(y) > drift->coordLimit ||
      fabs(xp) > drift->slopeLimit || fabs(yp) > drift->slopeLimit)
    return 0;
  x += xp * drift->length;
  y += yp * drift->length;
  pathFactor = drift->expandHamiltonian ?
    1 + (xp * xp + yp * yp) / 2 : sqrt(1 + xp * xp + yp * yp);
  part[0] = x;
  part[2] = y;
  part[4] += drift->length * pathFactor;
  return isfinite(x) && isfinite(y) &&
    fabs(x) <= drift->coordLimit && fabs(y) <= drift->coordLimit;
}

__device__ __forceinline__ int gpuApContourLost(
  const double *part, const GPU_APERTURE_CONTOUR_DATA &contour, double length);

__global__ void gpuBatchedApertureTrackKernel(
  double *trialCoord, long trials, int stride,
  const GPU_BATCHED_APERTURE_ELEMENT *element, long elements,
  long passes, double pCentral, unsigned char *survived,
  double *trialXLost, double *trialYLost, double *trialSLost,
  long *trialLossPass, long *trialLossElement) {
  long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int alive = 1;

  if (id >= trials)
    return;
  part = trialCoord + id * stride;
  trialLossPass[id] = trialLossElement[id] = -1;
  for (long pass = 0; alive && pass < passes; pass++) {
    for (long ie = 0; ie < elements; ie++) {
      const GPU_BATCHED_APERTURE_ELEMENT *item = element + ie;

      switch (item->type) {
      case GPU_BATCHED_APERTURE_MATRIX_DRIFT:
        if (!gpuBatchedApertureMatrixDrift(part, &item->data.drift))
          alive = 0;
        break;
      case GPU_BATCHED_APERTURE_MULTIPOLE_DRIFT:
        if (!gpuBatchedApertureMultipoleDrift(part, &item->data.drift))
          alive = 0;
        break;
      case GPU_BATCHED_APERTURE_MULTIPOLE:
        if (!gpuMultipoleTrackParticleData<false, true>(
              part, stride, 1, &item->data.multipole))
          alive = 0;
        break;
      case GPU_BATCHED_APERTURE_RCOL:
        if (gpuRectangularCollimatorLostAtPosition(
              part[0], part[2], item->data.rcol.xmax,
              item->data.rcol.ymax, item->data.rcol.xCenter,
              item->data.rcol.yCenter, 0, 0)) {
          part[4] = item->data.rcol.sStart;
          part[5] = pCentral * (1 + part[5]);
          alive = 0;
        }
        break;
      case GPU_BATCHED_APERTURE_APCONTOUR:
        if (item->data.apContour.check &&
            gpuApContourLost(part, item->data.apContour.contour, 0.0)) {
          part[4] = item->data.apContour.sStart;
          part[5] = pCentral * (1 + part[5]);
          alive = 0;
        }
        break;
      default:
        alive = 0;
        break;
      }
      /* imposeApContour() after the element, for a sticky contour */
      if (alive && item->postContourActive && gpuApContourLost(part, item->postContour, 0.0)) {
        part[4] = item->sEnd;
        part[5] = pCentral * (1 + part[5]);
        alive = 0;
      }
      if (!alive) {
        trialXLost[id] = part[0];
        trialYLost[id] = part[2];
        trialSLost[id] = part[4];
        trialLossPass[id] = pass;
        trialLossElement[id] = item->elementIndex;
        break;
      }
    }
  }
  survived[id] = alive ? 1 : 0;
}

__global__ void gpuBatchedApertureReduceKernel(
  long lines, long nSteps, long split,
  const unsigned char *survived,
  const double *trialXLost, const double *trialYLost,
  const double *trialSLost, const long *trialLossPass,
  const long *trialLossElement, const double *dx, const double *dy,
  const double *x0, const double *y0, double *xLimit, double *yLimit,
  double *xLost, double *yLost, double *sLost,
  long *lossPass, long *lossElement, double *lossX0, double *lossY0,
  long *originStable) {
  long line = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long firstLost = nSteps;
  long lastSurvived;

  if (line >= lines)
    return;
  if (split == 0 && line == 0 && !survived[0])
    *originStable = 0;
  for (long step = 0; step < nSteps; step++) {
    if (!survived[line * nSteps + step]) {
      firstLost = step;
      break;
    }
  }
  lastSurvived = firstLost - 1;
  if (lastSurvived >= 0) {
    double candidateX = x0[line] + lastSurvived * dx[line];
    double candidateY = y0[line] + lastSurvived * dy[line];

    if (split == 0 ||
        (dx[line] > 0 && xLimit[line] < candidateX) ||
        (dx[line] == 0 && yLimit[line] < candidateY) ||
        (dx[line] < 0 && xLimit[line] > candidateX)) {
      xLimit[line] = candidateX;
      yLimit[line] = candidateY;
    }
  }
  if (firstLost < nSteps) {
    long id = line * nSteps + firstLost;
    xLost[line] = trialXLost[id];
    yLost[line] = trialYLost[id];
    sLost[line] = trialSLost[id];
    lossPass[line] = trialLossPass[id];
    lossElement[line] = trialLossElement[id];
    /* where that trial started, so the host can track it again on the CPU */
    lossX0[line] = x0[line] + firstLost * dx[line];
    lossY0[line] = y0[line] + firstLost * dy[line];
  }
}

__global__ void gpuRectangularCollimatorSurvivorFlagKernel(double *coord,
                                                           long nParticles,
                                                           int stride,
                                                           double xmax,
                                                           double ymax,
                                                           double xCenter,
                                                           double yCenter,
                                                           double length,
                                                           long openCode,
                                                           long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  lost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, 0, openCode);
  if (!lost && length > 0)
    lost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, length, openCode);
  survivorPrefix[ip] = lost ? 0 : 1;
}

__global__ void gpuRectangularCollimatorStableScatterKernel(double *coord,
                                                            double *scratch,
                                                            const long *survivorPrefix,
                                                            long nParticles,
                                                            int stride,
                                                            double xmax,
                                                            double ymax,
                                                            double xCenter,
                                                            double yCenter,
                                                            double length,
                                                            long openCode,
                                                            double z,
                                                            double pCentral,
                                                            long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost, entranceLost;
  double x1, y1;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  entranceLost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, 0, openCode);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, length, openCode);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];

  if (entranceLost) {
    target[4] = z;
    target[5] = pCentral * (1 + target[5]);
    return;
  }
  if (length <= 0)
    return;

  x1 = part[0] + length * part[1];
  y1 = part[2] + length * part[3];
  if (lost) {
    if (!isfinite(x1) || !isfinite(y1)) {
      target[0] = x1;
      target[2] = y1;
      target[4] = z + length;
    } else {
      int isOut = 0;
      double zx = DBL_MAX;
      double zy = DBL_MAX;
      double dz;

      if (!openCode) {
        if (xmax > 0 && fabs(x1 - xCenter) > xmax)
          isOut += 1;
        if (ymax > 0 && fabs(y1 - yCenter) > ymax)
          isOut += 2;
        if ((isOut & 1) && part[1] != 0)
          zx = (gpuCoordinateSign(part[1]) * xmax - (part[0] - xCenter)) / part[1];
        if ((isOut & 2) && part[3] != 0)
          zy = (gpuCoordinateSign(part[3]) * ymax - (part[2] - yCenter)) / part[3];
        dz = zx < zy ? zx : zy;
        target[0] = part[0] + part[1] * dz;
        target[2] = part[2] + part[3] * dz;
        target[4] = z + dz;
      }
    }
    target[5] = pCentral * (1 + target[5]);
    return;
  }

  target[0] = x1;
  target[2] = y1;
  target[4] += length * sqrt(1 + target[1] * target[1] + target[3] * target[3]);
}

__device__ __forceinline__ int gpuEllipticalCollimatorLostAt(double *part,
                                                             double xmax,
                                                             double ymax,
                                                             double xCenter,
                                                             double yCenter,
                                                             long exponent,
                                                             long yExponent,
                                                             double length,
                                                             long openCode) {
  /* Entrance checks do not advance the particle (in particular, avoid
   * turning a finite position into NaN via 0 * an infinite slope). */
  double x = length == 0 ? part[0] : part[0] + length * part[1];
  double y = length == 0 ? part[2] : part[2] + length * part[3];
  double normalizedX, normalizedY, normalized;
  long effectiveYExponent = length == 0 ? yExponent : exponent;

  if (!isfinite(x) || !isfinite(y))
    return 1;
  normalizedX = (x - xCenter) / xmax;
  normalizedY = (y - yCenter) / ymax;
  normalized =
    gpuIntegerPower(normalizedX, exponent) +
    gpuIntegerPower(normalizedY, effectiveYExponent);
  if (normalized > 1)
    return openCode ? gpuEvaluateLostWithOpenSides(openCode, normalizedX, normalizedY, 1, 1) : 1;
  return 0;
}

/* Finite collimators remove entrance and exit losses in separate passes.
 * A union of their loss masks gives the same survivors but a different
 * swap-with-tail permutation. Keep the CPU's two passes on the device. */
template <bool elliptical>
__global__ void gpuCollimatorLossOrderKernel(
  double *coord, long *order, long nParticles, int stride,
  double xmax, double ymax, double xCenter, double yCenter,
  double length, long openCode, long exponent, long yExponent) {
  if (blockIdx.x || threadIdx.x)
    return;
  long *destination = order + nParticles;
  for (long i = 0; i < nParticles; i++)
    order[i] = i;
  long active = nParticles;
  for (int pass = 0; pass < (length > 0 ? 2 : 1); pass++) {
    double distance = pass ? length : 0;
    for (long i = 0; i < active; ) {
      double *part = coord + order[i] * stride;
      int lost = elliptical ?
        gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                     exponent, yExponent, distance, openCode) :
        gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                       distance, openCode);
      if (lost) {
        long source = order[i];
        order[i] = order[--active];
        order[active] = source;
      } else
        i++;
    }
  }
  for (long i = 0; i < nParticles; i++)
    destination[order[i]] = i;
}

__global__ void gpuEllipticalCollimatorSurvivorFlagKernel(double *coord,
                                                          long nParticles,
                                                          int stride,
                                                          double xmax,
                                                          double ymax,
                                                          double xCenter,
                                                          double yCenter,
                                                          long exponent,
                                                          long yExponent,
                                                          double length,
                                                          long openCode,
                                                          long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  lost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                       exponent, yExponent, 0, openCode);
  if (!lost && length > 0)
    lost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                         exponent, yExponent, length, openCode);
  survivorPrefix[ip] = lost ? 0 : 1;
}

__global__ void gpuEllipticalCollimatorStableScatterKernel(double *coord,
                                                           double *scratch,
                                                           const long *survivorPrefix,
                                                           long nParticles,
                                                           int stride,
                                                           double xmax,
                                                           double ymax,
                                                           double xCenter,
                                                           double yCenter,
                                                           long exponent,
                                                           long yExponent,
                                                           double length,
                                                           long openCode,
                                                           double z,
                                                           double pCentral,
                                                           long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost, entranceLost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  entranceLost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                               exponent, yExponent, 0, openCode);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                         exponent, yExponent, length, openCode);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];

  if (entranceLost) {
    target[4] = z;
    target[5] = pCentral * (1 + target[5]);
    return;
  }
  if (length <= 0)
    return;

  target[0] = part[0] + length * part[1];
  target[2] = part[2] + length * part[3];
  if (lost) {
    target[4] = z + length;
    target[5] = pCentral * (1 + target[5]);
  } else {
    target[4] += length * sqrt(1 + target[1] * target[1] + target[3] * target[3]);
  }
}

__global__ void gpuScraperLossCountKernel(double *coord, long nParticles, int stride,
                                          int plane, double center, double position,
                                          int sideSign, double length,
                                          long *lostCount) {
  long localCount = 0;

  for (long ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles; ip = nParticles) {
    double *part = coord + ip * stride;
    double coordinate = part[plane] + length * part[plane + 1] - center;

    if (sideSign * coordinate > sideSign * position)
      localCount++;
  }

  if (localCount)
    atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), (unsigned long long)localCount);
}

__device__ __forceinline__ int gpuScraperLostAt(double *part, int plane,
                                                double center,
                                                double position,
                                                int sideSign,
                                                double length) {
  double coordinate = part[plane] + length * part[plane + 1] - center;

  return sideSign * coordinate > sideSign * position;
}

__global__ void gpuScraperSurvivorFlagKernel(double *coord,
                                             long nParticles,
                                             int stride,
                                             int plane,
                                             double center,
                                             double position,
                                             int sideSign,
                                             double length,
                                             long *survivorPrefix,
                                             int *lossRound) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  /* beam_scraper() removes the entrance's losses, then the exit's after the drift */
  lost = gpuScraperLostAt(part, plane, center, position, sideSign, 0);
  lossRound[ip] = lost ? 0 : -1;
  if (!lost && length > 0 &&
      (lost = gpuScraperLostAt(part, plane, center, position, sideSign, length)))
    lossRound[ip] = 1;
  survivorPrefix[ip] = lost ? 0 : 1;
}

__global__ void gpuScraperStableScatterKernel(double *coord,
                                              double *scratch,
                                              const long *survivorPrefix,
                                              long nParticles,
                                              int stride,
                                              int plane,
                                              double center,
                                              double position,
                                              int sideSign,
                                              double length,
                                              double z,
                                              double pCentral,
                                              long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost, entranceLost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  entranceLost = gpuScraperLostAt(part, plane, center, position, sideSign, 0);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuScraperLostAt(part, plane, center, position, sideSign, length);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];

  if (entranceLost) {
    target[4] = z;
    target[5] = pCentral * (1 + target[5]);
    return;
  }
  if (length <= 0)
    return;

  target[0] = part[0] + length * part[1];
  target[2] = part[2] + length * part[3];
  if (lost) {
    double dz = (target[plane] - center - position) / target[plane + 1];

    target[4] = z + length - dz;
    target[0] -= target[1] * dz;
    target[2] -= target[3] * dz;
    target[5] = pCentral * (1 + target[5]);
  } else {
    target[4] += length * sqrt(1 + target[1] * target[1] + target[3] * target[3]);
  }
}

__device__ __forceinline__ int gpuApertureDataLost(double *part,
                                                   double xCenter,
                                                   double yCenter,
                                                   double xSize,
                                                   double ySize) {
  double dx = part[0] - xCenter;
  double dy = part[2] - yCenter;

  if ((xSize && fabs(dx) > xSize) ||
      (ySize && fabs(dy) > ySize))
    return 1;
  return 0;
}

__global__ void gpuApertureDataLossCountKernel(double *coord,
                                               long nParticles,
                                               int stride,
                                               double xCenter,
                                               double yCenter,
                                               double xSize,
                                               double ySize,
                                               long *lostCount) {
  long localCount = 0;

  for (long ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles; ip = nParticles) {
    double *part = coord + ip * stride;

    localCount += gpuApertureDataLost(part, xCenter, yCenter, xSize, ySize);
  }

  if (localCount)
    atomicAdd(reinterpret_cast<unsigned long long *>(lostCount), (unsigned long long)localCount);
}

/* An APCONTOUR element checks each end of its length, on a straight line */
__device__ __forceinline__ int gpuApContourLost(
  const double *part, const GPU_APERTURE_CONTOUR_DATA &contour, double length) {
  GPU_APERTURE_LIMIT_DATA none = {};

  none.cosReverseTilt = 1;
  if (contour.dz) {
    /* offsetBeamCoordinatesForMisalignment() moves the particle along its slope, in this order */
    GPU_APERTURE_CONTOUR_DATA offset = contour;
    double x = part[0] - contour.dx + contour.dz * part[1];
    double y = part[2] - contour.dy + contour.dz * part[3];

    offset.dx = offset.dy = 0;
    if (!gpuInsideApertureContour(x, y, none, offset))
      return 1;
    return length > 0 &&
           !gpuInsideApertureContour(x + part[1] * length, y + part[3] * length, none, offset);
  }
  if (!gpuInsideApertureContour(part[0], part[2], none, contour))
    return 1;
  return length > 0 &&
         !gpuInsideApertureContour(part[0] + part[1] * length, part[2] + part[3] * length, none, contour);
}

__global__ void gpuApContourLossCountKernel(double *coord, long nParticles, int stride,
                                            GPU_APERTURE_CONTOUR_DATA contour, double length,
                                            unsigned long long *lostCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles && gpuApContourLost(coord + ip * stride, contour, length))
    atomicAdd(lostCount, 1ULL);
}

__global__ void gpuApContourSurvivorFlagKernel(double *coord, long nParticles, int stride,
                                               GPU_APERTURE_CONTOUR_DATA contour,
                                               long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] = gpuApContourLost(coord + ip * stride, contour, 0.0) ? 0 : 1;
}

__global__ void gpuApContourStableScatterKernel(double *coord, double *scratch,
                                                const long *survivorPrefix, long nParticles,
                                                int stride, GPU_APERTURE_CONTOUR_DATA contour,
                                                double z, double pCentral) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  target = scratch + survivorPrefix[ip] * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (!gpuApContourLost(part, contour, 0.0))
    return;
  target[4] = z;
  target[5] = pCentral * (1 + target[5]);
}

__global__ void gpuApertureDataSurvivorFlagKernel(double *coord,
                                                  long nParticles,
                                                  int stride,
                                                  double xCenter,
                                                  double yCenter,
                                                  double xSize,
                                                  double ySize,
                                                  long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuApertureDataLost(coord + ip * stride, xCenter, yCenter,
                        xSize, ySize) ? 0 : 1;
}

__global__ void gpuApertureDataStableScatterKernel(double *coord,
                                                   double *scratch,
                                                   const long *survivorPrefix,
                                                   long nParticles,
                                                   int stride,
                                                   double xCenter,
                                                   double yCenter,
                                                   double xSize,
                                                   double ySize,
                                                   double z,
                                                   double pCentral,
                                                   long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination, lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  destination = survivorPrefix[ip];
  lost = gpuApertureDataLost(part, xCenter, yCenter, xSize, ySize);
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (!lost)
    return;
  target[4] = z;
  target[5] = pCentral * (1 + target[5]);
}

__device__ __forceinline__ int gpuUpperTriangularIndex(int i, int j) {
  return i * 7 - i * (i - 1) / 2 + (j - i);
}

__device__ __forceinline__ double gpuParticleTime(double *part, double pCentral, double cMks) {
  double p = pCentral * (1 + part[5]);
  return part[4] * sqrt(p * p + 1) / (cMks * p);
}

__device__ __forceinline__ int gpuHistogramParticleSelected(
  const double *part, int64_t startPID, int64_t endPID) {
  return (startPID < 0 && endPID < 0) ||
         (part[6] >= startPID && part[6] <= endPID);
}

__device__ __forceinline__ double gpuHistogramParticleTime(
  const double *part, double pCentral, double cMks) {
  double p = pCentral * (1 + part[5]);
  return part[4] / (cMks * p / sqrt(p * p + 1));
}

__global__ void gpuHistogramRangePartialKernel(
  const double *coord, long nParticles, int stride, double pCentral,
  double cMks, int64_t startPID, int64_t endPID,
  unsigned int coordinateMask, GPU_HISTOGRAM_RANGE_DATA *partial) {
  __shared__ int64_t count[GPU_HISTOGRAM_THREADS];
  __shared__ double minimum[7][GPU_HISTOGRAM_THREADS];
  __shared__ double maximum[7][GPU_HISTOGRAM_THREADS];
  long tid = threadIdx.x;
  long ip;
  int icoord;

  count[tid] = 0;
  for (icoord = 0; icoord < 7; icoord++) {
    minimum[icoord][tid] = DBL_MAX;
    maximum[icoord][tid] = -DBL_MAX;
  }
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    const double *part = coord + ip * stride;
    double time = 0;

    if (!gpuHistogramParticleSelected(part, startPID, endPID))
      continue;
    count[tid]++;
    if (coordinateMask & ((1U << 4) | (1U << 6)))
      time = gpuHistogramParticleTime(part, pCentral, cMks);
    for (icoord = 0; icoord < 7; icoord++) {
      double value;

      if (!(coordinateMask & (1U << icoord)))
        continue;
      value = (icoord == 4 || icoord == 6) ? time : part[icoord];
      if (value < minimum[icoord][tid])
        minimum[icoord][tid] = value;
      if (value > maximum[icoord][tid])
        maximum[icoord][tid] = value;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (icoord = 0; icoord < 7; icoord++) {
        if (minimum[icoord][tid + offset] < minimum[icoord][tid])
          minimum[icoord][tid] = minimum[icoord][tid + offset];
        if (maximum[icoord][tid + offset] > maximum[icoord][tid])
          maximum[icoord][tid] = maximum[icoord][tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial[blockIdx.x].count = count[0];
    for (icoord = 0; icoord < 7; icoord++) {
      partial[blockIdx.x].minimum[icoord] = minimum[icoord][0];
      partial[blockIdx.x].maximum[icoord] = maximum[icoord][0];
    }
  }
}

__global__ void gpuHistogramRangeFinalizeKernel(
  const GPU_HISTOGRAM_RANGE_DATA *partial,
  GPU_HISTOGRAM_RANGE_DATA *result) {
  __shared__ int64_t count[GPU_HISTOGRAM_THREADS];
  __shared__ double minimum[7][GPU_HISTOGRAM_THREADS];
  __shared__ double maximum[7][GPU_HISTOGRAM_THREADS];
  long tid = threadIdx.x;
  int icoord;

  count[tid] = tid < GPU_HISTOGRAM_BLOCKS ? partial[tid].count : 0;
  for (icoord = 0; icoord < 7; icoord++) {
    minimum[icoord][tid] = tid < GPU_HISTOGRAM_BLOCKS ?
                             partial[tid].minimum[icoord] : DBL_MAX;
    maximum[icoord][tid] = tid < GPU_HISTOGRAM_BLOCKS ?
                             partial[tid].maximum[icoord] : -DBL_MAX;
  }
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (icoord = 0; icoord < 7; icoord++) {
        if (minimum[icoord][tid + offset] < minimum[icoord][tid])
          minimum[icoord][tid] = minimum[icoord][tid + offset];
        if (maximum[icoord][tid + offset] > maximum[icoord][tid])
          maximum[icoord][tid] = maximum[icoord][tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    result->count = count[0];
    for (icoord = 0; icoord < 7; icoord++) {
      result->minimum[icoord] = minimum[icoord][0];
      result->maximum[icoord] = maximum[icoord][0];
    }
  }
}

__global__ void gpuHistogramBinPartialKernel(
  const double *coord, long nParticles, int stride,
  GPU_HISTOGRAM_BIN_DATA data, unsigned long long *partial) {
  long ip;

  for (ip = (long)blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    const double *part = coord + ip * stride;
    double time = 0;

    if (!gpuHistogramParticleSelected(part, data.startPID, data.endPID))
      continue;
    if (data.coordinateMask & ((1U << 4) | (1U << 6)))
      time = gpuHistogramParticleTime(part, data.pCentral, data.cMks);
    for (int icoord = 0; icoord < 7; icoord++) {
      double binSize, dbin, value;
      long bin;

      if (!(data.coordinateMask & (1U << icoord)))
        continue;
      value = (icoord == 4 || icoord == 6) ? time : part[icoord];
      if (icoord == 6)
        value -= data.timeOffset;
      binSize = (data.upper[icoord] - data.lower[icoord]) / data.bins;
      dbin = (value - data.lower[icoord]) / binSize;
      bin = static_cast<long>(dbin);
      if (dbin < 0 || bin < 0 || bin >= data.bins)
        continue;
      atomicAdd(partial +
                  (icoord * data.bins + bin) * GPU_HISTOGRAM_BLOCKS +
                  blockIdx.x,
                1ULL);
    }
  }
}

__global__ void gpuHistogramBinFinalizeKernel(
  const unsigned long long *partial, long bins,
  unsigned long long *histogram) {
  long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long sum = 0;

  if (index >= 7 * bins)
    return;
  for (long block = 0; block < GPU_HISTOGRAM_BLOCKS; block++)
    sum += partial[index * GPU_HISTOGRAM_BLOCKS + block];
  histogram[index] = sum;
}

__device__ __forceinline__ void gpuWakeAddToParticleEnergy(double *part,
                                                           double timeOfFlight,
                                                           const GPU_WAKE_LONGITUDINAL_DATA *wake,
                                                           double dgamma) {
  double p = wake->pCentral * (1 + part[5]);
  double gamma = sqrt(p * p + 1);
  double gamma1 = gamma + dgamma;
  double p1, pz, pz1, pRatio;

  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - wake->pCentral) / wake->pCentral;
  part[4] = timeOfFlight * wake->cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
}

__global__ void gpuRfmodeTimeKernel(
  double *coord, long nParticles, int stride, double pCentral,
  double cMks, double *time) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    time[ip] = gpuParticleTime(coord + ip * stride, pCentral, cMks);
}

__global__ void gpuRfmodeHistogramKernel(
  long nParticles, GPU_RFMODE_DATA data, const double *time,
  long *pbin, unsigned long long *histogram,
  unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double t;
  long ib;

  if (ip >= nParticles)
    return;
  t = time[ip];
  pbin[ip] = -1;
  ib = static_cast<long>((t - data.tmin) / data.dt);
  if (ib < 0 || ib >= data.bins)
    return;
  pbin[ip] = ib;
  atomicAdd(histogram + ib, 1ULL);
  atomicAdd(binnedCount, 1ULL);
}

__global__ void gpuTrfmodeHistogramKernel(
  double *coord, long nParticles, int stride, GPU_RFMODE_DATA data,
  const double *time, long *pbin, unsigned long long *histogram,
  double *xsum, double *ysum, unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long ib;

  if (ip >= nParticles)
    return;
  pbin[ip] = -1;
  ib = static_cast<long>((time[ip] - data.tmin) / data.dt);
  if (ib < 0 || ib >= data.bins)
    return;
  pbin[ip] = ib;
  atomicAdd(histogram + ib, 1ULL);
  atomicAdd(binnedCount, 1ULL);
}

/* Each bin's x and y sums, added in particle order as trfmode.c and
 * ftrfmode.c add them.  Atomic adds made these depend on thread timing, so
 * TRFMODE and FTRFMODE differed from run to run. */
__global__ void gpuTrfmodeBinSumsKernel(
  const double *coord, long nParticles, int stride, GPU_RFMODE_DATA data,
  const long *pbin, double *xsum, double *ysum) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double sx = 0, sy = 0;

  if (ib >= data.bins)
    return;
  for (long ip = 0; ip < nParticles; ip++) {
    if (pbin[ip] != ib)
      continue;
    sx += coord[ip * stride] - data.dx;
    sy += coord[ip * stride + 2] - data.dy;
  }
  xsum[ib] = sx;
  ysum[ib] = sy;
}

/* Keys for a stable sort of particles by bin; particles outside the bins sort last */
__global__ void gpuTrfmodeSortKeysKernel(const long *pbin, long nParticles, long bins,
                                         unsigned int *keys, long *index) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long ib;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  keys[ip] = ib >= 0 && ib < bins ? static_cast<unsigned int>(ib) : static_cast<unsigned int>(bins);
  index[ip] = ip;
}

/* The same sums as gpuTrfmodeBinSumsKernel, in the same particle order, over each bin's own particles
 * only: the stable sort keeps them in particle order.  One block per bin.  The sums must be added in
 * order, so one thread adds them, but the block's threads read each stretch of the bin's particles
 * into shared memory first.  One thread reading them itself, through the index, took 0.5 us a
 * particle: 28 ms for a million particles in a bunch 20 bins long. */
#define GPU_TRFMODE_SUM_THREADS 256

__global__ void gpuTrfmodeSortedBinSumsKernel(
  const double *coord, int stride, GPU_RFMODE_DATA data, const unsigned long long *binStart,
  const unsigned long long *histogram, const long *index, double *xsum, double *ysum) {
  __shared__ double x[GPU_TRFMODE_SUM_THREADS];
  __shared__ double y[GPU_TRFMODE_SUM_THREADS];
  long ib = blockIdx.x;
  int tid = threadIdx.x;
  double sx = 0, sy = 0;
  unsigned long long start, count, base;

  if (ib >= data.bins)
    return;
  start = binStart[ib];
  count = histogram[ib];
  for (base = 0; base < count; base += blockDim.x) {
    if (base + tid < count) {
      long ip = index[start + base + tid];

      x[tid] = coord[ip * stride] - data.dx;
      y[tid] = coord[ip * stride + 2] - data.dy;
    }
    __syncthreads();
    if (tid == 0) {
      unsigned long long n = count - base < blockDim.x ? count - base : blockDim.x;

      for (unsigned long long j = 0; j < n; j++) {
        sx += x[j];
        sy += y[j];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    xsum[ib] = sx;
    ysum[ib] = sy;
  }
}

__device__ __forceinline__ void gpuRfmodeAddToParticleEnergy(
  double *part, double timeOfFlight, const GPU_RFMODE_DATA *data,
  double dgamma) {
  double p = data->pCentral * (1 + part[5]);
  double gamma = sqrt(p * p + 1);
  double gamma1 = gamma + dgamma;
  double p1, pz, pz1, pRatio;

  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - data->pCentral) / data->pCentral;
  part[4] = timeOfFlight * data->cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
}

__global__ void gpuRfmodeApplyKicksKernel(
  double *coord, long nParticles, int stride, GPU_RFMODE_DATA data,
  const double *time, const long *pbin, const double *voltage) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, value;
  long ib;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  if (ib < 0 || ib >= data.bins)
    return;
  part = coord + ip * stride;
  if (data.interpolate) {
    double dt1 = time[ip] - (data.tmin + data.dt * (ib + 0.5));
    long ib1, ib2;
    if (dt1 < 0) {
      ib1 = ib - 1;
      ib2 = ib;
    } else {
      ib1 = ib;
      ib2 = ib + 1;
    }
    if (ib2 > data.lastBin) {
      ib2--;
      ib1--;
    }
    if (ib1 < data.firstBin) {
      ib1++;
      ib2++;
    }
    dt1 = time[ip] - (data.tmin + data.dt * (ib1 + 0.5));
    value = voltage[ib1] +
      (voltage[ib2] - voltage[ib1]) / data.dt * dt1;
  } else {
    value = voltage[ib];
  }
  gpuRfmodeAddToParticleEnergy(
    part, time[ip], &data,
    data.nCavities * value /
      (1e6 * data.particleMassMV * data.particleRelSign));
}

__device__ __forceinline__ double gpuTrfmodeVoltage(
  const double *voltage, long ib, double time,
  const GPU_RFMODE_DATA *data) {
  if (data->interpolate) {
    double dt1 = time - (data->tmin + data->dt * (ib + 0.5));
    long ib1, ib2;
    if (dt1 < 0) {
      ib1 = ib - 1;
      ib2 = ib;
    } else {
      ib1 = ib;
      ib2 = ib + 1;
    }
    if (ib2 > data->lastBin) {
      ib2--;
      ib1--;
    }
    if (ib1 < data->firstBin) {
      ib1++;
      ib2++;
    }
    dt1 = time - (data->tmin + data->dt * (ib1 + 0.5));
    return voltage[ib1] +
      (voltage[ib2] - voltage[ib1]) / data->dt * dt1;
  }
  return voltage[ib];
}

__global__ void gpuTrfmodeApplyKicksKernel(
  double *coord, long nParticles, int stride, GPU_RFMODE_DATA data,
  const double *time, const long *pbin, const double *voltage) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, P, Px, Py, Pz, factor, Vx, Vy, Vz;
  long ib;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  if (ib < 0 || ib >= data.bins)
    return;
  part = coord + ip * stride;
  Vx = gpuTrfmodeVoltage(voltage, ib, time[ip], &data);
  Vy = gpuTrfmodeVoltage(voltage + data.bins, ib, time[ip], &data);
  Vz = gpuTrfmodeVoltage(voltage + 2 * data.bins, ib, time[ip], &data);
  factor = data.nCavities /
    (1e6 * data.particleMassMV * data.particleRelSign);
  P = data.pCentral * (1 + part[5]);
  Pz = P / sqrt(1 + part[1] * part[1] + part[3] * part[3]) +
    factor * Vz;
  Px = part[1] * Pz + factor * Vx;
  Py = part[3] * Pz + factor * Vy;
  P = sqrt(Pz * Pz + Px * Px + Py * Py);
  part[1] = Px / Pz;
  part[3] = Py / Pz;
  part[5] = (P - data.pCentral) / data.pCentral;
  part[4] = time[ip] * data.cMks * P / sqrt(P * P + 1);
}

__global__ void gpuWakeBinKernel(double *coord, long nParticles, int stride,
                                 GPU_WAKE_LONGITUDINAL_DATA wake,
                                 double *time, long *pbin, double *itime,
                                 unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double t;
  long ib;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  pbin[ip] = -1;
  if (wake.useBunchFilter &&
      static_cast<long>(part[wake.bunchIndexColumn]) != wake.selectedBunch)
    return;
  t = gpuParticleTime(part, wake.pCentral, wake.cMks);
  time[ip] = t;
  ib = static_cast<long>((t - wake.tmin) / wake.dt + 0.5);
  if (ib < 0 || ib > wake.bins - 1)
    return;
  atomicAdd(itime + ib, 1.0);
  pbin[ip] = ib;
  atomicAdd(binnedCount, 1ULL);
}

__global__ void gpuWakeConvolveKernel(double *vtime, const double *itime,
                                      const double *wakeTable,
                                      GPU_WAKE_LONGITUDINAL_DATA wake) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long ib1, ib2, di;
  double sum = 0;

  if (ib >= wake.bins)
    return;
  ib2 = ib + wake.i0;
  ib1 = di = 0;
  if (ib2 >= wake.wakePoints) {
    di = ib2 - wake.wakePoints + 1;
    ib1 += di;
    ib2 -= di;
  }
  for (; ib1 < wake.bins && ib2 >= 0; ib1++, ib2--)
    sum += itime[ib1] * wakeTable[ib2];
  vtime[ib] = sum * wake.factor;
  if (ib == 0)
    vtime[wake.bins] = 0;
}

__global__ void gpuWakeApplyKicksKernel(double *coord, long nParticles,
                                        int stride,
                                        GPU_WAKE_LONGITUDINAL_DATA wake,
                                        const double *time, const long *pbin,
                                        const double *vtime) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  long ib;
  double dt1, dgam;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  if (ib < 0 || ib > wake.bins - 1)
    return;
  part = coord + ip * stride;
  if (wake.interpolate) {
    dt1 = time[ip] - (wake.tmin + wake.dt * ib);
    if ((dt1 < 0 && ib) || ib == wake.bins - 1) {
      ib--;
      dt1 += wake.dt;
    }
    dgam = (vtime[ib] + (vtime[ib + 1] - vtime[ib]) / wake.dt * dt1) /
           (1e6 * wake.particleMassMV * wake.particleRelSign);
  } else {
    dgam = vtime[ib] / (1e6 * wake.particleMassMV * wake.particleRelSign);
  }
  if (dgam)
    gpuWakeAddToParticleEnergy(part, time[ip], &wake, -dgam);
}

__device__ __forceinline__ void gpuLscAddToParticleEnergy(double *part,
                                                          double timeOfFlight,
                                                          const GPU_LSC_DATA *lsc,
                                                          double dgamma) {
  double p = lsc->pCentral * (1 + part[5]);
  double gamma = sqrt(p * p + 1);
  double gamma1 = gamma + dgamma;
  double p1, pz, pz1, pRatio;

  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - lsc->pCentral) / lsc->pCentral;
  part[4] = timeOfFlight * lsc->cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
}

__global__ void gpuLscBinKernel(double *coord, long nParticles, int stride,
                                GPU_LSC_DATA lsc, double *itime,
                                unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, t;
  long ib;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  t = gpuParticleTime(part, lsc.pCentral, lsc.cMks);
  ib = static_cast<long>((t - lsc.tmin) / lsc.dt + 0.5);
  if (ib < 0 || ib > lsc.bins - 1)
    return;
  atomicAdd(itime + ib, 1.0);
  atomicAdd(binnedCount, 1ULL);
}

__global__ void gpuLscApplyKickAndDriftKernel(double *coord, long nParticles,
                                              int stride, GPU_LSC_DATA lsc,
                                              const double *vtime) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, t, dt1, dgam, sign;
  long ib;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  t = gpuParticleTime(part, lsc.pCentral, lsc.cMks);
  ib = static_cast<long>((t - lsc.tmin) / lsc.dt + 0.5);
  if (ib >= 0 && ib <= lsc.bins - 1) {
    if (lsc.interpolate) {
      dt1 = t - (lsc.tmin + lsc.dt * ib);
      if ((dt1 < 0 && ib) || ib == lsc.bins - 1) {
        ib--;
        dt1 += lsc.dt;
      }
      dgam = (vtime[ib] + (vtime[ib + 1] - vtime[ib]) / lsc.dt * dt1) /
             (1e6 * lsc.particleMassMV * lsc.particleRelSign);
    } else {
      dgam = vtime[ib] / (1e6 * lsc.particleMassMV * lsc.particleRelSign);
    }
    if (dgam)
      gpuLscAddToParticleEnergy(part, t, &lsc, -dgam);
  }
  if (lsc.doDrift) {
    sign = lsc.backtrack ? -1 : 1;
    part[4] += lsc.length * sqrt(1 + part[1] * part[1] + part[3] * part[3]) * sign;
    part[0] += lsc.length * part[1] * sign;
    part[2] += lsc.length * part[3] * sign;
  }
}

__global__ void gpuScmultMomentPartialKernel(
  double *coord, long nParticles, int stride,
  GPU_SCMULT_MOMENT_DATA *partial) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[3][GPU_REDUCTION_THREADS];
  __shared__ double squareSum[3][GPU_REDUCTION_THREADS];
  /* Sum relative to the first particle. The mean of s is the distance traveled, so raw sums of
     squares would leave the bunch length to roundoff far along a machine. */
  double reference[3] = {coord[0], coord[2], coord[4]};
  long tid = threadIdx.x;
  long ip;
  int i;

  count[tid] = 0;
  for (i = 0; i < 3; i++) {
    sum[i][tid] = 0;
    squareSum[i][tid] = 0;
  }
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)blockDim.x * gridDim.x) {
    double *part = coord + ip * stride;
    double value[3] = {part[0] - reference[0], part[2] - reference[1],
                       part[4] - reference[2]};

    count[tid]++;
    for (i = 0; i < 3; i++) {
      sum[i][tid] += value[i];
      squareSum[i][tid] += value[i] * value[i];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 3; i++) {
        sum[i][tid] += sum[i][tid + offset];
        squareSum[i][tid] += squareSum[i][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x].count = count[0];
    for (i = 0; i < 3; i++) {
      partial[blockIdx.x].reference[i] = reference[i];
      partial[blockIdx.x].sum[i] = sum[i][0];
      partial[blockIdx.x].squareSum[i] = squareSum[i][0];
    }
  }
}

__global__ void gpuScmultMomentFinalizeKernel(
  const GPU_SCMULT_MOMENT_DATA *partial,
  GPU_SCMULT_MOMENT_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[3][GPU_REDUCTION_THREADS];
  __shared__ double squareSum[3][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  int i;

  count[tid] = partial[tid].count;
  for (i = 0; i < 3; i++) {
    sum[i][tid] = partial[tid].sum[i];
    squareSum[i][tid] = partial[tid].squareSum[i];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 3; i++) {
        sum[i][tid] += sum[i][tid + offset];
        squareSum[i][tid] += squareSum[i][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    for (i = 0; i < 3; i++) {
      result->reference[i] = partial[0].reference[i];
      result->sum[i] = sum[i][0];
      result->squareSum[i] = squareSum[i][0];
    }
  }
}

__global__ void gpuScmultLinearKickKernel(double *coord, long nParticles,
                                          int stride,
                                          GPU_SCMULT_LINEAR_DATA data) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double k0, dz;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (data.uniformDistribution)
    k0 = data.longitudinalScale;
  else {
    dz = part[4] - data.center[2];
    k0 = data.longitudinalScale *
         exp(-0.5 * dz * dz / (data.sigma[2] * data.sigma[2]));
  }
  if (data.horizontal)
    part[1] += k0 * data.dmux / data.betax * (part[0] - data.center[0]);
  if (data.vertical)
    part[3] += k0 * data.dmuy / data.betay * (part[2] - data.center[1]);
}

/* Algorithm 680 (Poppe and Wijers), matching the CPU wofz implementation
 * used by nonlinearSCKick.  Nonlinear SCMULT only calls this with yi >= 0,
 * but the quadrant handling is retained so the implementation stays complete.
 */
__device__ __forceinline__ int gpuScmultNearestInteger(double value) {
  return static_cast<int>(value >= 0 ? floor(value + 0.5) :
                                     -floor(0.5 - value));
}

__device__ __forceinline__ bool gpuScmultWofz(double xi, double yi,
                                               double *u, double *v) {
  int kapn = 0, n, nu, np1;
  double xabs = fabs(xi), yabs = fabs(yi);
  double x, y, qrho, xabsq, xquad, yquad;
  double xsum, ysum, xaux, u1, v1, u2 = 0, v2 = 0, daux;
  double h = 0, h2 = 0, qlambda = 0;
  double rx = 0, ry = 0, sx = 0, sy = 0, tx, ty, c;
  bool usePowerSeries, useTaylor;

  if (!u || !v || xabs > 5e153 || yabs > 5e153)
    return false;

  x = xabs / (float)6.3;
  y = yabs / (float)4.4;
  qrho = x * x + y * y;
  xabsq = xabs * xabs;
  xquad = xabsq - yabs * yabs;
  yquad = 2 * xabs * yabs;
  usePowerSeries = qrho < 0.085264;

  if (usePowerSeries) {
    int j;

    qrho = (1 - y * (float)0.85) * sqrt(qrho);
    n = gpuScmultNearestInteger(qrho * 72 + 6);
    j = 2 * n + 1;
    xsum = (float)1.0 / j;
    ysum = 0;
    for (int i = n; i >= 1; --i) {
      j -= 2;
      xaux = (xsum * xquad - ysum * yquad) / i;
      ysum = (xsum * yquad + ysum * xquad) / i;
      xsum = xaux + (float)1.0 / j;
    }
    u1 = (xsum * yabs + ysum * xabs) * -1.12837916709551257388 +
         (float)1.0;
    v1 = (xsum * xabs - ysum * yabs) * 1.12837916709551257388;
    daux = exp(-xquad);
    u2 = daux * cos(yquad);
    v2 = -daux * sin(yquad);
    *u = u1 * u2 - v1 * v2;
    *v = u1 * v2 + v1 * u2;
  } else {
    if (qrho > (float)1.0) {
      qrho = sqrt(qrho);
      nu = static_cast<int>(1442 / (qrho * 26 + 77) + 3);
    } else {
      qrho = (1 - y) * sqrt(1 - qrho);
      h = qrho * (float)1.88;
      h2 = 2 * h;
      kapn = gpuScmultNearestInteger(qrho * 34 + 7);
      nu = gpuScmultNearestInteger(qrho * 26 + 16);
    }
    useTaylor = h > (float)0.0;
    if (useTaylor)
      qlambda = gpuIntegerPower(h2, kapn);

    for (n = nu; n >= 0; --n) {
      np1 = n + 1;
      tx = yabs + h + np1 * rx;
      ty = xabs - np1 * ry;
      c = (float)0.5 / (tx * tx + ty * ty);
      rx = c * tx;
      ry = c * ty;
      if (useTaylor && n <= kapn) {
        tx = qlambda + sx;
        sx = rx * tx - ry * sy;
        sy = ry * tx + rx * sy;
        qlambda /= h2;
      }
    }

    if (h == (float)0.0) {
      *u = rx * 1.12837916709551257388;
      *v = ry * 1.12837916709551257388;
    } else {
      *u = sx * 1.12837916709551257388;
      *v = sy * 1.12837916709551257388;
    }
    if (yabs == (float)0.0)
      *u = exp(-xabs * xabs);
  }

  if (yi < (float)0.0) {
    double w1;

    if (usePowerSeries) {
      u2 *= 2;
      v2 *= 2;
    } else {
      xquad = -xquad;
      if (yquad > 3537118876014220.0 || xquad > 708.503061461606)
        return false;
      w1 = exp(xquad) * 2;
      u2 = w1 * cos(yquad);
      v2 = -w1 * sin(yquad);
    }
    *u = u2 - *u;
    *v = v2 - *v;
    if (xi > (float)0.0)
      *v = -*v;
  } else if (xi < (float)0.0) {
    *v = -*v;
  }
  return true;
}

__device__ __forceinline__ void gpuScmultLinearFallback(
  double *part, const GPU_SCMULT_LINEAR_DATA *data) {
  double k0;

  if (data->uniformDistribution) {
    k0 = data->c1 / data->sigma[2] * data->charge *
         sqrt(3.141592653589793238462643383279502884 / 6.0);
  } else {
    double dz = part[4] - data->center[2];
    k0 = data->c1 / data->sigma[2] * data->charge *
         exp(-0.5 * dz * dz / (data->sigma[2] * data->sigma[2]));
  }
  if (data->horizontal)
    part[1] += k0 * data->dmux / data->betax *
               (part[0] - data->center[0]);
  if (data->vertical)
    part[3] += k0 * data->dmuy / data->betay *
               (part[2] - data->center[1]);
}

__global__ void gpuScmultNonlinearKickKernel(double *coord, long nParticles,
                                             int stride,
                                             GPU_SCMULT_LINEAR_DATA data) {
  const double pi = 3.141592653589793238462643383279502884;
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double x, y, z, k0, kickX, kickY;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  x = part[0] - data.center[0];
  y = part[2] - data.center[1];
  z = part[4] - data.center[2];

  if (data.uniformDistribution)
    k0 = data.longitudinalScale;
  else {
    double normalizedZ = z / data.sigma[2];
    k0 = data.longitudinalScale * exp(-0.5 * normalizedZ * normalizedZ);
  }

  if (data.roundBeam) {
    double r = sqrt(x * x + y * y);
    double dp = 0, theta = 0;

    if (r > 0) {
      double normalizedR = r / data.roundSigma;
      dp = (1 - exp(-0.5 * normalizedR * normalizedR)) / r;
      theta = atan2(y, x);
    }
    k0 *= sqrt(2.0 / pi);
    part[1] += dp * cos(theta) * k0;
    part[3] += dp * sin(theta) * k0;
    return;
  }

  {
    double ay, w1Real, w1Imag, w2Real, w2Imag;
    double waReal, waImag, wbReal, wbImag, c3, wReal, wImag;
    double kx, ky;

    if (data.swapXY) {
      double tmp = x;
      x = y;
      y = -tmp;
    }

    ay = fabs(y);
    w1Real = x * data.inverseSd;
    w1Imag = ay * data.inverseSd;
    w2Real = w1Real * data.minorMajorRatio;
    w2Imag = w1Imag * data.majorMinorRatio;
    if (!gpuScmultWofz(w1Real, w1Imag, &waReal, &waImag) ||
        !gpuScmultWofz(w2Real, w2Imag, &wbReal, &wbImag)) {
      gpuScmultLinearFallback(part, &data);
      return;
    }

    c3 = exp(-x * x * data.inverseTwoMajorSigma2 -
             y * y * data.inverseTwoMinorSigma2);
    wReal = waReal - c3 * wbReal;
    wImag = waImag - c3 * wbImag;
    kx = k0 * data.kxScale;
    ky = k0 * data.kyScale;

    if (data.swapXY) {
      kickX = -kx * wReal * (y > 0 ? 1 : -1);
      kickY = ky * wImag;
    } else {
      kickX = kx * wImag;
      kickY = ky * wReal * (y > 0 ? 1 : -1);
    }
    part[1] += kickX;
    part[3] += kickY;
  }
}

/* ipow() from mdbmth, grouping the products as it does */
__device__ double gpuCpuIpow(double x, long p) {
  double hp;

  if (x == 0)
    return p == 0 ? 1. : 0.;
  if (p < 0)
    return 1. / gpuCpuIpow(x, -p);
  switch (p) {
  case 0:
    return 1.;
  case 1:
    return x;
  case 2:
    return x * x;
  case 3:
    hp = x * x;
    return hp * x;
  case 4:
    hp = x * x;
    return hp * hp;
  case 5:
    hp = x * x;
    return hp * hp * x;
  case 6:
    hp = x * x;
    return hp * hp * hp;
  case 7:
    hp = x * x * x;
    return hp * hp * x;
  case 8:
    hp = x * x;
    hp = hp * hp;
    return hp * hp;
  default:
    hp = gpuCpuIpow(x, p / 2);
    return p % 2 ? hp * hp * x : hp * hp;
  }
}

__device__ __forceinline__ double gpuTrwakeDriveValue(double coordinate,
                                                      double offset,
                                                      long exponent) {
  double value = coordinate - offset;

  if (exponent == 1)
    return value;
  if (exponent < 1)
    return 1;
  return gpuCpuIpow(value, exponent);
}

__device__ __forceinline__ void gpuTrwakeLocalCoordinates(
  const double *part, GPU_TRWAKE_DATA wake, double *x, double *xp,
  double *y, double *yp) {
  if (wake.hasTilt) {
    *x = part[0] * wake.cosTilt + part[2] * wake.sinTilt;
    *y = -part[0] * wake.sinTilt + part[2] * wake.cosTilt;
    *xp = part[1] * wake.cosTilt + part[3] * wake.sinTilt;
    *yp = -part[1] * wake.sinTilt + part[3] * wake.cosTilt;
  } else {
    *x = part[0];
    *xp = part[1];
    *y = part[2];
    *yp = part[3];
  }
}

__device__ __forceinline__ void gpuTrwakeStoreLocalCoordinates(
  double *part, GPU_TRWAKE_DATA wake, double x, double xp,
  double y, double yp) {
  if (wake.hasTilt) {
    part[0] = x * wake.cosTilt - y * wake.sinTilt;
    part[2] = x * wake.sinTilt + y * wake.cosTilt;
    part[1] = xp * wake.cosTilt - yp * wake.sinTilt;
    part[3] = xp * wake.sinTilt + yp * wake.cosTilt;
  } else {
    part[1] = xp;
    part[3] = yp;
  }
}

__global__ void gpuTrwakePrepareKernel(double *coord, long nParticles,
                                       int stride, GPU_TRWAKE_DATA wake,
                                       double *time, double *pz, long *pbin,
                                       unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double t;
  long ib;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  pbin[ip] = -1;
  if (wake.useBunchFilter &&
      static_cast<long>(part[wake.bunchIndexColumn]) != wake.selectedBunch)
    return;
  t = gpuParticleTime(part, wake.pCentral, wake.cMks);
  time[ip] = t;
  ib = static_cast<long>((t - wake.tmin) / wake.dt + 0.5);
  if (ib < 0 || ib > wake.bins - 1)
    return;
  pbin[ip] = ib;
  {
    /* from the tilted slopes, as the CPU computes pz after rotating the beam */
    double x, xp, y, yp;
    gpuTrwakeLocalCoordinates(part, wake, &x, &xp, &y, &yp);
    pz[ip] = wake.pCentral * (1 + part[5]) / sqrt(1 + xp * xp + yp * yp);
  }
  atomicAdd(binnedCount, 1ULL);
}

__global__ void gpuTrwakeBinSumsKernel(double *coord, long nParticles,
                                       int stride, GPU_TRWAKE_DATA wake,
                                       const long *pbin,
                                       double *posItimeX,
                                       double *posItimeY) {
  extern __shared__ double trwakePartial[];
  double *partialX = trwakePartial;
  double *partialY = trwakePartial + blockDim.x;
  long ib = blockIdx.x;
  long tid = threadIdx.x;
  long ip;
  double sumX = 0, sumY = 0;

  if (ib >= wake.bins)
    return;
  if (wake.cpuOrderSums) {
    /* one thread adds the bin's particles in their order, as the CPU does */
    if (tid == 0) {
      for (ip = 0; ip < nParticles; ip++) {
        double x, xp, y, yp;
        if (pbin[ip] != ib)
          continue;
        gpuTrwakeLocalCoordinates(coord + ip * stride, wake, &x, &xp, &y, &yp);
        sumX += gpuTrwakeDriveValue(x, wake.offset[0], wake.driveExponent[0]);
        sumY += gpuTrwakeDriveValue(y, wake.offset[1], wake.driveExponent[1]);
      }
      posItimeX[ib] = sumX;
      posItimeY[ib] = sumY;
    }
    return;
  }
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double x, xp, y, yp;
    if (pbin[ip] != ib)
      continue;
    double *part = coord + ip * stride;
    gpuTrwakeLocalCoordinates(part, wake, &x, &xp, &y, &yp);
    sumX += gpuTrwakeDriveValue(x, wake.offset[0],
                                wake.driveExponent[0]);
    sumY += gpuTrwakeDriveValue(y, wake.offset[1],
                                wake.driveExponent[1]);
  }
  partialX[tid] = sumX;
  partialY[tid] = sumY;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partialX[tid] += partialX[tid + offset];
      partialY[tid] += partialY[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    posItimeX[ib] = partialX[0];
    posItimeY[ib] = partialY[0];
  }
}

/* The same sums as gpuTrwakeBinSumsKernel's one thread per bin, in the same particle order, over each
 * bin's own particles only: a stable sort by bin keeps them in particle order.  The scan of the whole
 * beam for every bin made multibunchPartialShortRangeWakes four times slower than the CPU. */
__global__ void gpuTrwakeSortedBinSumsKernel(const double *coord, long nParticles,
                                             int stride, GPU_TRWAKE_DATA wake,
                                             const unsigned int *sortedBin,
                                             const long *sortedIndex,
                                             double *posItimeX,
                                             double *posItimeY) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long low = 0, high = nParticles, k;
  double sumX = 0, sumY = 0;

  if (ib >= wake.bins)
    return;
  while (low < high) {
    long middle = low + (high - low) / 2;
    if (sortedBin[middle] < static_cast<unsigned int>(ib))
      low = middle + 1;
    else
      high = middle;
  }
  for (k = low; k < nParticles && sortedBin[k] == static_cast<unsigned int>(ib); k++) {
    double x, xp, y, yp;
    gpuTrwakeLocalCoordinates(coord + sortedIndex[k] * stride, wake, &x, &xp, &y, &yp);
    sumX += gpuTrwakeDriveValue(x, wake.offset[0], wake.driveExponent[0]);
    sumY += gpuTrwakeDriveValue(y, wake.offset[1], wake.driveExponent[1]);
  }
  posItimeX[ib] = sumX;
  posItimeY[ib] = sumY;
}

__global__ void gpuTrwakeConvolveKernel(double *vtime,
                                        const double *posItime,
                                        const double *wakeTable,
                                        GPU_TRWAKE_DATA wake,
                                        int plane) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long ib1, ib2, di;
  double sum = 0;

  if (ib >= wake.bins)
    return;
  ib2 = ib + wake.i0;
  ib1 = di = 0;
  if (ib2 >= wake.wakePoints) {
    di = ib2 - wake.wakePoints + 1;
    ib1 += di;
    ib2 -= di;
  }
  for (; ib1 < wake.bins && ib2 >= 0; ib1++, ib2--)
    sum += posItime[ib1] * wakeTable[ib2];
  vtime[ib] = sum * wake.factor[plane];
  if (ib == 0)
    vtime[wake.bins] = 0;
}

__device__ __forceinline__ double gpuTrwakeInterpolatedVoltage(
  const double *vtime, const double *time, long ip, long ib,
  GPU_TRWAKE_DATA wake) {
  double dt1, value;

  if (wake.interpolate) {
    dt1 = time[ip] - (wake.tmin + wake.dt * ib);
    if ((dt1 < 0 && ib) || ib == wake.bins - 1) {
      ib--;
      dt1 += wake.dt;
    }
    value = vtime[ib] + (vtime[ib + 1] - vtime[ib]) / wake.dt * dt1;
  } else {
    value = vtime[ib];
  }
  return value;
}

__global__ void gpuTrwakeApplyKicksKernel(double *coord, long nParticles,
                                          int stride, GPU_TRWAKE_DATA wake,
                                          const double *time,
                                          const double *pz,
                                          const long *pbin,
                                          const double *vtimeX,
                                          const double *vtimeY) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  long ib;
  double voltage;
  double x, xp, y, yp;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  part = coord + ip * stride;
  if (ib < 0 || ib > wake.bins - 1) {
    /* The CPU rotates every particle of the bunch there and back */
    if (wake.hasTilt &&
        !(wake.useBunchFilter &&
          static_cast<long>(part[wake.bunchIndexColumn]) != wake.selectedBunch)) {
      gpuTrwakeLocalCoordinates(part, wake, &x, &xp, &y, &yp);
      gpuTrwakeStoreLocalCoordinates(part, wake, x, xp, y, yp);
    }
    return;
  }
  gpuTrwakeLocalCoordinates(part, wake, &x, &xp, &y, &yp);
  if (wake.hasWake[0] && wake.factor[0]) {
    voltage = gpuTrwakeInterpolatedVoltage(vtimeX, time, ip, ib, wake);
    if (wake.probeExponent[0])
      voltage *= gpuCpuIpow(x, wake.probeExponent[0]);
    if (voltage)
      xp += voltage / (1e6 * wake.particleMassMV *
                       wake.particleRelSign) / pz[ip];
  }
  if (wake.hasWake[1] && wake.factor[1]) {
    voltage = gpuTrwakeInterpolatedVoltage(vtimeY, time, ip, ib, wake);
    if (wake.probeExponent[1])
      voltage *= gpuCpuIpow(y, wake.probeExponent[1]);
    if (voltage)
      yp += voltage / (1e6 * wake.particleMassMV *
                       wake.particleRelSign) / pz[ip];
  }
  gpuTrwakeStoreLocalCoordinates(part, wake, x, xp, y, yp);
}

__global__ void gpuCombinedWakePrepareKernel(
  double *coord, long nParticles, int stride, GPU_COMBINED_WAKE_DATA wake,
  double *time, double *pz, long *pbin,
  unsigned long long *binnedCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double t;
  long ib;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  pbin[ip] = -1;
  if (wake.useBunchFilter &&
      static_cast<long>(part[wake.bunchIndexColumn]) != wake.selectedBunch)
    return;
  t = gpuParticleTime(part, wake.pCentral, wake.cMks);
  time[ip] = t;
  /* zlongit.c's own grid truncates: ib = (time - tmin) / dt */
  if (wake.legacyGrid)
    ib = static_cast<long>((t - wake.tmin) / wake.dt);
  else
    ib = static_cast<long>((t - wake.tmin) / wake.dt + 0.5);
  if (ib < 0 || ib >= wake.bins)
    return;
  pbin[ip] = ib;
  pz[ip] = wake.pCentral * (1 + part[5]) /
           sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  atomicAdd(binnedCount, 1ULL);
}

__device__ __forceinline__ double gpuPolynomialIntegerPower(double value,
                                                            int32_t exponent) {
  double result = 1;
  for (int32_t i = 0; i < exponent; i++)
    result *= value;
  return result;
}

__global__ void gpuPolynomialSeriesKernel(
  double *coord, long nParticles, int stride,
  GPU_POLYNOMIAL_SERIES_DATA data, const double *coefficient,
  const int32_t *exponent, unsigned long long *invalidCount) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double input[6], output[6], power;
  double *part;
  double cosTilt, sinTilt, x, xp, y, yp, qx, qy;
  double denom, dp, p, beta0;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  cosTilt = cos(data.tilt);
  sinTilt = sin(data.tilt);

  part[4] += data.dz * sqrt(1 + part[1] * part[1] +
                            part[3] * part[3]);
  part[0] += -data.dx + data.dz * part[1];
  part[2] += -data.dy + data.dz * part[3];
  x = cosTilt * part[0] + sinTilt * part[2];
  y = -sinTilt * part[0] + cosTilt * part[2];
  xp = cosTilt * part[1] + sinTilt * part[3];
  yp = -sinTilt * part[1] + cosTilt * part[3];
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp) ||
      fabs(x) > data.coordinateLimit || fabs(y) > data.coordinateLimit ||
      fabs(xp) > data.slopeLimit || fabs(yp) > data.slopeLimit) {
    atomicAdd(invalidCount, 1ULL);
    return;
  }

  dp = part[5];
  p = data.pCentral * (1 + dp);
  denom = sqrt(1 + xp * xp + yp * yp);
  qx = (1 + dp) * xp / denom;
  qy = (1 + dp) * yp / denom;
  beta0 = p / sqrt(p * p + 1);
  input[0] = x;
  input[1] = qx;
  input[2] = y;
  input[3] = qy;
  input[4] = 0;
  input[5] = dp;

  for (long coordinate = 0; coordinate < 6; coordinate++) {
    output[coordinate] = 0;
    for (long term = data.coordinateOffset[coordinate];
         term < data.coordinateOffset[coordinate + 1]; term++) {
      power = coefficient[term];
      power *= gpuPolynomialIntegerPower(input[0], exponent[6 * term + 0]);
      power *= gpuPolynomialIntegerPower(input[1], exponent[6 * term + 1]);
      power *= gpuPolynomialIntegerPower(input[2], exponent[6 * term + 2]);
      power *= gpuPolynomialIntegerPower(input[3], exponent[6 * term + 3]);
      power *= gpuPolynomialIntegerPower(input[4], exponent[6 * term + 4]);
      power *= gpuPolynomialIntegerPower(input[5], exponent[6 * term + 5]);
      output[coordinate] += power;
    }
  }
  x = output[0];
  qx = output[1];
  y = output[2];
  qy = output[3];
  dp = output[5];
  denom = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
  if (!(denom > 0) || !isfinite(denom)) {
    atomicAdd(invalidCount, 1ULL);
    return;
  }
  denom = sqrt(denom);
  xp = qx / denom;
  yp = qy / denom;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp) ||
      !isfinite(output[4]) || !isfinite(dp)) {
    atomicAdd(invalidCount, 1ULL);
    return;
  }

  part[0] = cosTilt * x - sinTilt * y;
  part[1] = cosTilt * xp - sinTilt * yp;
  part[2] = sinTilt * x + cosTilt * y;
  part[3] = sinTilt * xp + cosTilt * yp;
  part[4] += output[4] * beta0;
  part[5] = dp;
  part[0] += data.dx - part[1] * data.dz;
  part[2] += data.dy - part[3] * data.dz;
  part[4] -= data.dz * sqrt(1 + part[1] * part[1] +
                            part[3] * part[3]);
}

__global__ void gpuCombinedWakeHistogramPartialKernel(
  const double *coord, long nParticles, int stride,
  GPU_COMBINED_WAKE_DATA wake, const long *pbin, double *partial,
  long partialCount) {
  long partialIndex = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long first, last;
  double *local;

  if (partialIndex >= partialCount)
    return;
  first = nParticles * partialIndex / partialCount;
  last = nParticles * (partialIndex + 1) / partialCount;
  local = partial + partialIndex * 3 * wake.bins;
  for (long ip = first; ip < last; ip++) {
    long ib = pbin[ip];
    const double *part;
    if (ib < 0 || ib >= wake.bins)
      continue;
    part = coord + ip * stride;
    local[ib] += 1;
    local[wake.bins + ib] += part[0] - wake.offset[0];
    local[2 * wake.bins + ib] += part[2] - wake.offset[1];
  }
}

__global__ void gpuCombinedWakeHistogramReduceKernel(
  const double *partial, long bins, long partialCount, double *histogram) {
  extern __shared__ double combinedWakePartial[];
  long histogramIndex = blockIdx.x;
  long tid = threadIdx.x;
  double sum = 0;

  if (histogramIndex >= 3 * bins)
    return;
  for (long ipartial = tid; ipartial < partialCount;
       ipartial += blockDim.x)
    sum += partial[ipartial * 3 * bins + histogramIndex];
  combinedWakePartial[tid] = sum;
  __syncthreads();
  for (long offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset)
      combinedWakePartial[tid] += combinedWakePartial[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    histogram[histogramIndex] = combinedWakePartial[0];
}

/* The three histograms as the CPU's binning loop makes them, each bin's particles added in particle
 * order: a stable sort by bin keeps them in that order.  One block per bin: the sums must be added in
 * order, so one thread adds them, but the block's threads read each stretch of the bin's particles into
 * shared memory first, as gpuTrfmodeSortedBinSumsKernel does.  One thread per bin reading them itself
 * through the index took about 0.5 us a particle in the fullest bin (until November 14, 2026). */
#define GPU_COMBINED_WAKE_SUM_THREADS 256

__global__ void gpuCombinedWakeSortedHistogramKernel(
  const double *coord, long nParticles, int stride, GPU_COMBINED_WAKE_DATA wake,
  const unsigned int *sortedBin, const long *sortedIndex, double *histogram) {
  __shared__ double x[GPU_COMBINED_WAKE_SUM_THREADS];
  __shared__ double y[GPU_COMBINED_WAKE_SUM_THREADS];
  __shared__ long binFirst, binCount;
  long ib = blockIdx.x;
  int tid = threadIdx.x;
  double count = 0, sumX = 0, sumY = 0;

  if (ib >= wake.bins)
    return;
  if (tid == 0) {
    long low = 0, high = nParticles, first;
    while (low < high) {
      long middle = low + (high - low) / 2;
      if (sortedBin[middle] < static_cast<unsigned int>(ib))
        low = middle + 1;
      else
        high = middle;
    }
    first = low;
    high = nParticles;
    while (low < high) {
      long middle = low + (high - low) / 2;
      if (sortedBin[middle] <= static_cast<unsigned int>(ib))
        low = middle + 1;
      else
        high = middle;
    }
    binFirst = first;
    binCount = low - first;
  }
  __syncthreads();
  for (long base = 0; base < binCount; base += blockDim.x) {
    if (base + tid < binCount) {
      const double *part = coord + sortedIndex[binFirst + base + tid] * stride;
      x[tid] = part[0] - wake.offset[0];
      y[tid] = part[2] - wake.offset[1];
    }
    __syncthreads();
    if (tid == 0) {
      long n = binCount - base < blockDim.x ? binCount - base : blockDim.x;
      for (long j = 0; j < n; j++) {
        count += 1;
        sumX += x[j];
        sumY += y[j];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    histogram[ib] = count;
    histogram[wake.bins + ib] = sumX;
    histogram[2 * wake.bins + ib] = sumY;
  }
}

__global__ void gpuCombinedWakeConvolutionKernel(
  double *voltage, const double *histogram, const double *table,
  GPU_COMBINED_WAKE_DATA wake, long tableStride) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long channel = blockIdx.y;
  long ib1, ib2, di;
  double sum = 0;

  if (channel >= GPU_COMBINED_WAKE_CHANNELS || ib >= wake.bins ||
      !wake.enabled[channel])
    return;
  ib2 = ib + wake.i0;
  ib1 = di = 0;
  if (ib2 >= wake.tablePoints) {
    di = ib2 - wake.tablePoints + 1;
    ib1 += di;
    ib2 -= di;
  }
  for (; ib1 < wake.bins && ib2 >= 0; ib1++, ib2--)
    sum += histogram[wake.driver[channel] * wake.bins + ib1] *
           table[channel * tableStride + ib2];
  voltage[channel * wake.bins + ib] = sum * wake.factor[channel];
}

__global__ void gpuCombinedWakeFrequencyMultiplyKernel(
  cufftDoubleComplex *output, const cufftDoubleComplex *driver,
  const cufftDoubleComplex *table, long frequencyPoints) {
  long frequency = (long)blockIdx.x * blockDim.x + threadIdx.x;
  cufftDoubleComplex result;

  if (frequency >= frequencyPoints)
    return;
  result.x = driver[frequency].x * table[frequency].x -
             driver[frequency].y * table[frequency].y;
  result.y = driver[frequency].x * table[frequency].y +
             driver[frequency].y * table[frequency].x;
  output[frequency] = result;
}

__global__ void gpuCombinedWakeDriverFirstBinKernel(
  const double *histogram, long bins, int *driverFirstBin) {
  long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long driver, bin;

  if (index >= 3 * bins || histogram[index] == 0)
    return;
  driver = index / bins;
  bin = index - driver * bins;
  atomicMin(driverFirstBin + driver, static_cast<int>(bin));
}

__global__ void gpuCombinedWakeExtractConvolutionKernel(
  double *voltage, const double *convolution,
  const int *driverFirstBin, GPU_COMBINED_WAKE_DATA wake,
  long channel, long fftBins) {
  long ib = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long convolutionIndex;
  int firstBin;

  if (ib >= wake.bins || channel >= GPU_COMBINED_WAKE_CHANNELS)
    return;
  convolutionIndex = ib + wake.i0;
  firstBin = driverFirstBin[wake.driver[channel]];
  voltage[channel * wake.bins + ib] =
    firstBin < wake.bins && convolutionIndex >= firstBin &&
    convolutionIndex < fftBins ?
    convolution[convolutionIndex] * wake.factor[channel] / fftBins : 0;
}

__global__ void gpuCombinedImpedanceMultiplyKernel(
  cufftDoubleComplex *channelFrequency,
  const cufftDoubleComplex *driverFrequency, const double *table,
  GPU_COMBINED_WAKE_DATA wake, long tableStride) {
  long frequency = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long channel = blockIdx.y;
  long frequencyPoints = wake.bins / 2 + 1;
  cufftDoubleComplex result = {0, 0};

  if (channel >= GPU_COMBINED_WAKE_CHANNELS || frequency >= frequencyPoints ||
      !wake.enabled[channel])
    return;
  const cufftDoubleComplex value =
    driverFrequency[wake.driver[channel] * frequencyPoints + frequency];
  double zr, zi = 0;
  if (frequency == 0) {
    zr = table[channel * tableStride];
  } else if (frequency == wake.bins / 2 && !(wake.bins % 2)) {
    zr = table[channel * tableStride + wake.bins - 1];
  } else {
    zr = table[channel * tableStride + 2 * frequency - 1];
    zi = table[channel * tableStride + 2 * frequency];
  }
  double factor = wake.factor[channel] / wake.bins;
  result.x = (value.x * zr - value.y * zi) * factor;
  result.y = (value.x * zi + value.y * zr) * factor;
  channelFrequency[channel * frequencyPoints + frequency] = result;
}

__device__ __forceinline__ double gpuCombinedInterpolatedVoltage(
  const double *voltage, double time, long ib,
  const GPU_COMBINED_WAKE_DATA *wake) {
  double dt1;

  if (!wake->interpolate)
    return voltage[ib];
  dt1 = time - (wake->tmin + wake->dt * ib);
  if ((dt1 < 0 && ib) || ib == wake->bins - 1) {
    ib--;
    dt1 += wake->dt;
  }
  return voltage[ib] + (voltage[ib + 1] - voltage[ib]) /
                       wake->dt * dt1;
}

/* track_through_impedance()'s own interpolation for the longitudinal kick: none in the first bin, and
 * no kick (returns 0) when the upper point would lie past the last bin */
__device__ __forceinline__ int gpuCombinedImpedanceLongitudinalVoltage(
  const double *voltage, double time, long ib,
  const GPU_COMBINED_WAKE_DATA *wake, double *value) {
  double dt1;

  if (!wake->interpolate || ib <= 0) {
    *value = voltage[ib];
    return 1;
  }
  /* ZLONGIT's own grid: time - (tmin + dt / 2.0 + dt * ib), with tmin + dt / 2.0 formed on the host */
  if (wake->legacyGrid)
    dt1 = time - (wake->tminCenter + wake->dt * ib);
  else
    dt1 = time - (wake->tmin + wake->dt * ib);
  if (dt1 < 0)
    dt1 += wake->dt;
  else
    ib += 1;
  if (ib >= wake->bins)
    return 0;
  *value = voltage[ib - 1] + (voltage[ib] - voltage[ib - 1]) / wake->dt * dt1;
  return 1;
}

__device__ __forceinline__ void gpuCombinedAddToParticleEnergy(
  double *part, double timeOfFlight, const GPU_COMBINED_WAKE_DATA *wake,
  double dgamma) {
  double p = wake->pCentral * (1 + part[5]);
  double gamma = sqrt(p * p + 1);
  double gamma1 = gamma + dgamma;
  double p1, pz, pz1, pRatio;

  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - wake->pCentral) / wake->pCentral;
  part[4] = timeOfFlight * wake->cMks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
}

__global__ void gpuCombinedWakeApplyKicksKernel(
  double *coord, long nParticles, int stride, GPU_COMBINED_WAKE_DATA wake,
  const double *time, const double *pz, const long *pbin,
  const double *voltage) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  long ib;
  double transversePz;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  if (ib < 0 || ib >= wake.bins)
    return;
  part = coord + ip * stride;
  if (wake.enabled[0]) {
    double value = 0;
    int kick = 1;
    if (wake.mode == GPU_COMBINED_WAKE_MODE_IMPEDANCE)
      kick = gpuCombinedImpedanceLongitudinalVoltage(voltage, time[ip], ib, &wake, &value);
    else
      value = gpuCombinedInterpolatedVoltage(voltage, time[ip], ib, &wake);
    if (kick) {
      double dgamma = value /
        (1e6 * wake.particleMassMV * wake.particleRelSign);
      if (dgamma)
        gpuCombinedAddToParticleEnergy(part, time[ip], &wake, -dgamma);
    }
  }
  if (wake.mode == GPU_COMBINED_WAKE_MODE_TIME)
    transversePz = wake.pCentral * (1 + part[5]) /
      sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  else
    transversePz = pz[ip];

  for (long channel = 1; channel < GPU_COMBINED_WAKE_CHANNELS; channel++) {
    const double *channelVoltage;
    double value;
    int plane;
    if (!wake.enabled[channel])
      continue;
    channelVoltage = voltage + channel * wake.bins;
    value = gpuCombinedInterpolatedVoltage(channelVoltage, time[ip], ib, &wake);
    plane = wake.kickPlane[channel];
    if (wake.probeExponent[channel] > 0) {
      double coordinate = plane == 1 ? part[0] : part[2];
      value *= gpuIntegerPower(coordinate,
                               wake.probeExponent[channel]);
    }
    if (value && transversePz) {
      /* the order of applyTransverseWakeKicks(), so the kick is the CPU's to the last bit */
      value = value / (1e6 * wake.particleMassMV * wake.particleRelSign) /
              transversePz;
      if (plane == 1)
        part[1] += value;
      else if (plane == 2)
        part[3] += value;
    }
  }
}

__global__ void gpuCsrCsbendWakeKernel(const double *ctHist,
                                       const double *ctHistDeriv,
                                       const double *denom,
                                       double *T1,
                                       double *T2,
                                       double *dGamma,
                                       long nBins,
                                       double CSRConstant,
                                       double dsSlice,
                                       double slippageLength13,
                                       double dct,
                                       long steadyState,
                                       long trapazoidIntegration,
                                       long diSlippage,
                                       long diSlippage4,
                                       double t1Factor,
                                       double t2Factor) {
  long iBin = (long)blockIdx.x * blockDim.x + threadIdx.x;
  /* csbend.c writes 0.3 * sqr(denom[1]) * ..., and sqr() is a function: the square is formed first */
  double denom1Squared = denom[1] * denom[1];
  double term1 = 0;
  double term2 = 0;
  double t1 = 0;
  double t2 = 0;
  long count = 0;

  if (iBin >= nBins)
    return;

  if (CSRConstant) {
    if (steadyState) {
      if (!trapazoidIntegration) {
        for (long iBinBehind = iBin + 1; iBinBehind < nBins; iBinBehind++)
          t1 += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
      } else {
        if ((iBin + 1) < nBins)
          term1 = ctHistDeriv[iBin + 1] / denom[1];
        for (long iBinBehind = iBin + 1; iBinBehind < nBins;
             iBinBehind++, count++)
          t1 += (term2 = ctHistDeriv[iBinBehind] /
                         denom[iBinBehind - iBin]);
        if ((iBin + 1) < nBins)
          t1 += 0.3 * denom1Squared *
                (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) /
                dct;
        if (count > 1)
          t1 -= (term1 + term2) / 2;
      }
    } else {
      if (!trapazoidIntegration) {
        for (long iBinBehind = iBin + 1;
             iBinBehind <= iBin + diSlippage && iBinBehind < nBins;
             iBinBehind++)
          t1 += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
      } else {
        if ((iBin + 1) < nBins && (iBin + 1) <= iBin + diSlippage)
          term1 = ctHistDeriv[iBin + 1] / denom[1] / 2;
        for (long iBinBehind = iBin + 1;
             iBinBehind <= iBin + diSlippage && iBinBehind < nBins;
             iBinBehind++, count++)
          t1 += (term2 = ctHistDeriv[iBinBehind] /
                         denom[iBinBehind - iBin]);
        if (diSlippage > 0 && (iBin + 1) < nBins)
          t1 += 0.3 * denom1Squared *
                (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) /
                dct;
        if (count > 1)
          t1 -= (term1 + term2) / 2;
      }
      if ((iBin + diSlippage) < nBins)
        t2 += ctHist[iBin + diSlippage];
      if ((iBin + diSlippage4) < nBins)
        t2 -= ctHist[iBin + diSlippage4];
    }
    /* CSRConstant * length / nSlices and its T2 counterpart, formed by csbend.c in its own order:
     * CSRConstant * (length / nSlices) is the same only when nSlices is a power of two */
    t1 *= t1Factor;
    t2 *= t2Factor;
  }

  T1[iBin] = t1;
  T2[iBin] = t2;
  dGamma[iBin] = t1 + t2;
}

__global__ void gpuCsrCsbendKickInPlaceKernel(double *coord,
                                              long nParticles,
                                              int stride,
                                              const double *dGamma,
                                              long nBins,
                                              double ctLower,
                                              double dct,
                                              double Po,
                                              double rho0) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double ct, x, dp, f, binPosition;
  long iBin;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  ct = part[4];
  x = part[0];
  dp = part[5];
  binPosition = (ct - ctLower) / dct;
  iBin = static_cast<long>(binPosition);
  f = binPosition;
  f -= iBin;
  if (iBin >= 0 && iBin < nBins - 1) {
    dp += ((1 - f) * dGamma[iBin] + f * dGamma[iBin + 1]) /
          Po * (1 + x / rho0);
  }
  part[5] = dp;
}

/* convertFromCSBendCoords() then convertToCSBendCoords() with ctMode 1, as CSRCSBEND runs them
 * around output inside the bend. The pair does not return the starting numbers to the last bit
 * (time is divided by c, then multiplied by it), so the device repeats it to stay equal to the CPU. */
__global__ void gpuCsrCsbendOutputRoundTripKernel(double *coord,
                                                  long nParticles,
                                                  int stride,
                                                  double cosTilt,
                                                  double sinTilt,
                                                  double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double X, XP, Y, YP, x, xp, y, yp, t;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  X = part[0];
  XP = part[1];
  Y = part[2];
  YP = part[3];
  x = X * cosTilt - Y * sinTilt;
  y = X * sinTilt + Y * cosTilt;
  xp = XP * cosTilt - YP * sinTilt;
  yp = XP * sinTilt + YP * cosTilt;
  t = part[4] / cMks;
  X = x;
  Y = y;
  XP = xp;
  YP = yp;
  x = X * cosTilt + Y * sinTilt;
  y = -X * sinTilt + Y * cosTilt;
  xp = XP * cosTilt + YP * sinTilt;
  yp = -XP * sinTilt + YP * cosTilt;
  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = t * cMks;
}

/* The per-particle loops of track_through_driftCSR() and track_through_driftCSR_Stupakov() in csbend.c,
 * in their arithmetic, so a CSRDRIFT on the device gives the CPU's particles to the last bit.
 *
 * Advance: the drift of each kick, which also turns s into c*t (toTime), and the plain drifts before
 * the first kick and after the last one (toTime 0; a negative dz is the CPU's "coord -= dzFirst*..."). */
__global__ void gpuCsrDriftAdvanceKernel(double *coord,
                                         long nParticles,
                                         int stride,
                                         double dz,
                                         double Po,
                                         long linearOptics,
                                         long toTime) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double xp, yp, s, p, beta;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  xp = part[1];
  yp = part[3];
  part[0] += xp * dz;
  part[2] += yp * dz;
  s = part[4];
  if (toTime) {
    p = Po * (1 + part[5]);
    beta = p / sqrt(p * p + 1);
    if (linearOptics)
      part[4] = (s + dz) / beta;
    else
      part[4] = (s + dz * sqrt(1 + xp * xp + yp * yp)) / beta;
  } else {
    if (linearOptics)
      part[4] = s + dz;
    else
      part[4] = s + dz * sqrt(1 + xp * xp + yp * yp);
  }
}

/* The kick from the wake by linear interpolation, then c*t back to s.  dGamma is NULL when the CPU has
 * no wake to apply (its "if (csrWake.dGamma)"); applyKick 0 is the Stupakov loop with CSR off. */
__global__ void gpuCsrDriftKickKernel(double *coord,
                                      long nParticles,
                                      int stride,
                                      const double *dGamma,
                                      long nBins,
                                      long applyKick,
                                      double ct0,
                                      double dct,
                                      double Po,
                                      long useFactor,
                                      double factor) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double ct, dp, f, p, beta;
  long iBin;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  ct = part[4];
  dp = part[5];
  if (dGamma && applyKick) {
    f = (ct - ct0) / dct;
    iBin = static_cast<long>(f);
    f -= iBin;
    if (iBin >= 0 && iBin < nBins - 1) {
      double kick = ((1 - f) * dGamma[iBin] + f * dGamma[iBin + 1]) / Po;
      if (useFactor)
        kick = kick * factor;
      dp += kick;
    }
  }
  p = (1 + dp) * Po;
  beta = p / sqrt(p * p + 1);
  part[5] = dp;
  part[4] = beta * ct;
}

/* How many particles the kick loop finds inside the wake's bins */
struct GpuCsrDriftInRange {
  const double *coord;
  int stride;
  double ct0;
  double dct;
  long nBins1;
  __host__ __device__ long operator()(long ip) const {
    double f = (coord[ip * stride + 4] - ct0) / dct;
    long iBin = static_cast<long>(f);
    return iBin >= 0 && iBin < nBins1 ? 1 : 0;
  }
};

__global__ void gpuCsrHistogramKernel(double *coord,
                                      long nParticles,
                                      int stride,
                                      long coordinateIndex,
                                      double lower,
                                      double binSize,
                                      long bins,
                                      double *hist) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double value;
  long iBin;

  if (ip >= nParticles)
    return;
  value = coord[ip * stride + coordinateIndex];
  if (isnan(value) || isinf(value))
    return;
  iBin = static_cast<long>((value - lower) / binSize);
  if (iBin < 0 || iBin > bins - 1)
    return;
  atomicAdd(hist + iBin, 1.0);
}

__global__ void gpuCenterTimeKernel(double *coord, long nParticles, int stride,
                                    double pCentral, double timeOffset, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, beta;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  p = pCentral * (1 + part[5]);
  beta = p / sqrt(p * p + 1);
  part[4] -= cMks * beta * timeOffset;
}

__global__ void gpuCentroidSumsKernel(double *coord, long nParticles, int stride,
                                      GPU_BEAM_SUM_DATA *result) {
  __shared__ double sum[6][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i;

  for (i = 0; i < 6; i++)
    sum[i][tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    for (i = 0; i < 6; i++)
      sum[i][tid] += part[i];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (i = 0; i < 6; i++)
        sum[i][tid] += sum[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = nParticles;
    for (i = 0; i < 6; i++)
      result->centroidSum[i] = sum[i][0];
    result->centroidSum[6] = 0;
  }
}

__global__ void gpuTimeSumsKernel(double *coord, long nParticles, int stride,
                                  double pCentral, double cMks,
                                  GPU_BEAM_SUM_DATA *result) {
  __shared__ double sum[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  sum[tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x)
    sum[tid] += gpuParticleTime(coord + ip * stride, pCentral, cMks);
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset)
      sum[tid] += sum[tid + offset];
    __syncthreads();
  }

  if (tid == 0) {
    result->count = nParticles;
    result->centroidSum[6] = sum[0];
  }
}

__global__ void gpuSelectedTimeSumsKernel(double *coord, long nParticles,
                                          int stride, double pCentral,
                                          double cMks, int bunchColumn,
                                          long selectedBunch,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  __shared__ double minValue[GPU_REDUCTION_THREADS];
  __shared__ double maxValue[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  sum[tid] = 0;
  minValue[tid] = DBL_MAX;
  maxValue[tid] = -DBL_MAX;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double t;
    if (static_cast<long>(part[bunchColumn]) != selectedBunch)
      continue;
    t = gpuParticleTime(part, pCentral, cMks);
    count[tid]++;
    sum[tid] += t;
    if (t < minValue[tid])
      minValue[tid] = t;
    if (t > maxValue[tid])
      maxValue[tid] = t;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      sum[tid] += sum[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->centroidSum[6] = sum[0];
    result->min[6] = count[0] ? minValue[0] : 0;
    result->max[6] = count[0] ? maxValue[0] : 0;
  }
}

__global__ void gpuFiducialTimeSumsKernel(double *coord, long nParticles, int stride,
                                          double pCentral, double sOffset,
                                          double cMks, int particleIdColumn,
                                          int64_t startPID, int64_t endPID,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  double error;

  count[tid] = 0;
  sum[tid] = 0;
  error = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double p = pCentral * (1 + part[5]);
    double beta = p / sqrt(p * p + 1);
    double value, y, tmp;

    if ((startPID >= 0 || endPID >= 0) &&
        (particleIdColumn < 0 || particleIdColumn >= stride ||
         static_cast<int64_t>(part[particleIdColumn]) < startPID ||
         static_cast<int64_t>(part[particleIdColumn]) > endPID))
      continue;
    value = (part[4] + sOffset) / (cMks * beta);
    y = value - error;
    tmp = sum[tid] + y;
    error = (tmp - sum[tid]) - y;
    sum[tid] = tmp;
    count[tid]++;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      double y = sum[tid + offset] - error;
      double tmp = sum[tid] + y;
      error = (tmp - sum[tid]) - y;
      sum[tid] = tmp;
      count[tid] += count[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    for (int i = 0; i < 7; i++)
      result->centroidSum[i] = 0;
    result->centroidSum[6] = sum[0];
  }
}

__global__ void gpuFiducialFirstKernel(double *coord, long nParticles,
                                       int stride, double pCentral,
                                       double sOffset, double cMks,
                                       int particleIdColumn,
                                       int64_t startPID, int64_t endPID,
                                       GPU_BEAM_SUM_DATA *result) {
  __shared__ double bestTime[GPU_REDUCTION_THREADS];
  __shared__ long bestIndex[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  bestTime[tid] = 0;
  bestIndex[tid] = LONG_MAX;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;

    if ((startPID >= 0 || endPID >= 0) &&
        (particleIdColumn < 0 || particleIdColumn >= stride ||
         static_cast<int64_t>(part[particleIdColumn]) < startPID ||
         static_cast<int64_t>(part[particleIdColumn]) > endPID))
      continue;
    if (ip < bestIndex[tid]) {
      double p = pCentral * (1 + part[5]);
      double beta = p / sqrt(p * p + 1);

      bestIndex[tid] = ip;
      bestTime[tid] = (part[4] + sOffset) / (cMks * beta);
    }
    break;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset && bestIndex[tid + offset] < bestIndex[tid]) {
      bestIndex[tid] = bestIndex[tid + offset];
      bestTime[tid] = bestTime[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0 && bestIndex[0] != LONG_MAX) {
    result->count = 1;
    result->centroidSum[6] = bestTime[0];
  }
}

__global__ void gpuFiducialPmaximumKernel(double *coord, long nParticles,
                                          int stride, double pCentral,
                                          double sOffset, double cMks,
                                          int particleIdColumn,
                                          int64_t startPID, int64_t endPID,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ double bestDelta[GPU_REDUCTION_THREADS];
  __shared__ double bestTime[GPU_REDUCTION_THREADS];
  __shared__ long bestIndex[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  bestDelta[tid] = -DBL_MAX;
  bestTime[tid] = 0;
  bestIndex[tid] = LONG_MAX;
  if (nParticles <= 0)
    return;

  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double delta = part[5];

    if ((startPID >= 0 || endPID >= 0) &&
        (particleIdColumn < 0 || particleIdColumn >= stride ||
         static_cast<int64_t>(part[particleIdColumn]) < startPID ||
         static_cast<int64_t>(part[particleIdColumn]) > endPID))
      continue;
    if (delta > bestDelta[tid] ||
        (delta == bestDelta[tid] && ip < bestIndex[tid])) {
      double p = pCentral * (1 + delta);
      double beta = p / sqrt(p * p + 1);

      bestDelta[tid] = delta;
      bestTime[tid] = (part[4] + sOffset) / (cMks * beta);
      bestIndex[tid] = ip;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset && bestIndex[tid + offset] != LONG_MAX) {
      if (bestIndex[tid] == LONG_MAX ||
          bestDelta[tid + offset] > bestDelta[tid] ||
          (bestDelta[tid + offset] == bestDelta[tid] &&
           bestIndex[tid + offset] < bestIndex[tid])) {
        bestDelta[tid] = bestDelta[tid + offset];
        bestTime[tid] = bestTime[tid + offset];
        bestIndex[tid] = bestIndex[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0 && bestIndex[0] != LONG_MAX) {
    result->count = 1;
    result->max[5] = bestDelta[0];
    result->centroidSum[6] = bestTime[0];
  }
}

__global__ void gpuCentroidTimeSumsKernel(double *coord, long nParticles, int stride,
                                          double pCentral, double cMks,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i;

  for (i = 0; i < 7; i++)
    sum[i][tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    for (i = 0; i < 6; i++)
      sum[i][tid] += part[i];
    sum[6][tid] += gpuParticleTime(part, pCentral, cMks);
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (i = 0; i < 7; i++)
        sum[i][tid] += sum[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = nParticles;
    for (i = 0; i < 7; i++)
      result->centroidSum[i] = sum[i][0];
  }
}

/*
 * Small aggregate reductions are used by several host-side tracking helpers.
 * Keep a fixed number of deterministic partials so large beams can use the
 * whole device without changing the launch-dependent reduction order between
 * runs.  mode 0 sums the six phase-space coordinates, mode -1 sums time, and
 * mode -2 sums both.
 */
__global__ void gpuSimpleSumsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  int mode, GPU_BEAM_SUM_DATA *partial) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  double localSum[7];
  int64_t localCount = 0;
  long tid = threadIdx.x;
  long ip;
  int i;

  for (i = 0; i < 7; i++)
    localSum[i] = 0;
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    double *part = coord + ip * stride;
    localCount++;
    if (mode != -1)
      for (i = 0; i < 6; i++)
        localSum[i] += part[i];
    if (mode != 0)
      localSum[6] += gpuParticleTime(part, pCentral, cMks);
  }
  count[tid] = localCount;
  for (i = 0; i < 7; i++)
    sum[i][tid] = localSum[i];
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 7; i++)
        sum[i][tid] += sum[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    GPU_BEAM_SUM_DATA *result = partial + blockIdx.x;
    result->count = count[0];
    for (i = 0; i < 7; i++)
      result->centroidSum[i] = sum[i][0];
  }
}

__global__ void gpuSimpleSumsFinalizeKernel(
  const GPU_BEAM_SUM_DATA *partial, int blocks,
  GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  int tid = threadIdx.x;
  int i;

  count[tid] = 0;
  for (i = 0; i < 7; i++)
    sum[i][tid] = 0;
  if (tid < blocks) {
    count[tid] = partial[tid].count;
    for (i = 0; i < 7; i++)
      sum[i][tid] = partial[tid].centroidSum[i];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 7; i++)
        sum[i][tid] += sum[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    for (i = 0; i < 7; i++)
      result->centroidSum[i] = sum[i][0];
  }
}

__global__ void gpuBeamSumsKernel(double *coord, long nParticles, int stride,
                                  double pCentral, double cMks,
                                  GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double pSum[GPU_REDUCTION_THREADS];
  __shared__ double gammaSum[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  __shared__ double product[28][GPU_REDUCTION_THREADS];
  __shared__ double maxabs[7][GPU_REDUCTION_THREADS];
  __shared__ double minValue[7][GPU_REDUCTION_THREADS];
  __shared__ double maxValue[7][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i, j;

  count[tid] = 0;
  pSum[tid] = gammaSum[tid] = 0;
  for (i = 0; i < 7; i++) {
    sum[i][tid] = 0;
    maxabs[i][tid] = 0;
    minValue[i][tid] = DBL_MAX;
    maxValue[i][tid] = -DBL_MAX;
  }
  for (i = 0; i < 28; i++)
    product[i][tid] = 0;

  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double value[7];
    double p = pCentral * (1 + part[5]);
    value[0] = part[0];
    value[1] = part[1];
    value[2] = part[2];
    value[3] = part[3];
    value[4] = part[4];
    value[5] = part[5];
    value[6] = gpuParticleTime(part, pCentral, cMks);
    count[tid]++;
    pSum[tid] += p;
    gammaSum[tid] += sqrt(p * p + 1);
    for (i = 0; i < 7; i++) {
      double absValue = fabs(value[i]);
      sum[i][tid] += value[i];
      if (absValue > maxabs[i][tid])
        maxabs[i][tid] = absValue;
      if (value[i] < minValue[i][tid])
        minValue[i][tid] = value[i];
      if (value[i] > maxValue[i][tid])
        maxValue[i][tid] = value[i];
    }
    for (i = 0; i < 7; i++)
      for (j = i; j < 7; j++)
        product[gpuUpperTriangularIndex(i, j)][tid] += value[i] * value[j];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      pSum[tid] += pSum[tid + offset];
      gammaSum[tid] += gammaSum[tid + offset];
      for (i = 0; i < 7; i++) {
        sum[i][tid] += sum[i][tid + offset];
        if (maxabs[i][tid + offset] > maxabs[i][tid])
          maxabs[i][tid] = maxabs[i][tid + offset];
        if (minValue[i][tid + offset] < minValue[i][tid])
          minValue[i][tid] = minValue[i][tid + offset];
        if (maxValue[i][tid + offset] > maxValue[i][tid])
          maxValue[i][tid] = maxValue[i][tid + offset];
      }
      for (i = 0; i < 28; i++)
        product[i][tid] += product[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->pSum = pSum[0];
    result->gammaSum = gammaSum[0];
    for (i = 0; i < 7; i++) {
      result->centroidSum[i] = sum[i][0];
      result->maxabs[i] = maxabs[i][0];
      result->min[i] = minValue[i][0];
      result->max[i] = maxValue[i][0];
    }
    for (i = 0; i < 28; i++)
      result->productSum[i] = product[i][0];
  }
}

__global__ void gpuCenteredBeamSumsKernel(double *coord, long nParticles,
                                          int stride, double pCentral,
                                          double cMks,
                                          const double *centroid,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double product[28][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i, j;

  count[tid] = 0;
  for (i = 0; i < 28; i++)
    product[i][tid] = 0;

  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double value[7];
    value[0] = part[0] - centroid[0];
    value[1] = part[1] - centroid[1];
    value[2] = part[2] - centroid[2];
    value[3] = part[3] - centroid[3];
    value[4] = part[4] - centroid[4];
    value[5] = part[5] - centroid[5];
    value[6] = gpuParticleTime(part, pCentral, cMks) - centroid[6];
    count[tid]++;
    for (i = 0; i < 7; i++)
      for (j = i; j < 7; j++)
        product[gpuUpperTriangularIndex(i, j)][tid] += value[i] * value[j];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 28; i++)
        product[i][tid] += product[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    for (i = 0; i < 28; i++)
      result->productSum[i] = product[i][0];
  }
}

/*
 * The ordinary beam-output path needs extrema and centroids from the first
 * pass, followed by products about that centroid.  The legacy implementation
 * also formed unused raw products and ran both passes in a single CUDA block.
 * These kernels retain the numerically safer two-pass formula while spreading
 * each pass across the device.  The finalizers operate on one partial per
 * block, so the reduction order remains deterministic for a fixed launch.
 */
__global__ void gpuBeamStatisticsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  GPU_BEAM_SUM_DATA *partial, double *timeValue) {
  __shared__ int64_t count[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double pSum[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double gammaSum[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double sum[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double maxabs[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double minValue[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double maxValue[7][GPU_BEAM_OUTPUT_THREADS];
  int64_t localCount;
  double localPSum, localGammaSum;
  double localSum[7], localMaxabs[7], localMin[7], localMax[7];
  long tid = threadIdx.x;
  long ip;
  int i;

  localCount = 0;
  localPSum = localGammaSum = 0;
  for (i = 0; i < 7; i++) {
    localSum[i] = 0;
    localMaxabs[i] = 0;
    localMin[i] = DBL_MAX;
    localMax[i] = -DBL_MAX;
  }
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    double *part = coord + ip * stride;
    double value[7];
    double p = pCentral * (1 + part[5]);
    value[0] = part[0];
    value[1] = part[1];
    value[2] = part[2];
    value[3] = part[3];
    value[4] = part[4];
    value[5] = part[5];
    value[6] = gpuParticleTime(part, pCentral, cMks);
    timeValue[ip] = value[6];
    localCount++;
    localPSum += p;
    localGammaSum += sqrt(p * p + 1);
    for (i = 0; i < 7; i++) {
      double absValue = fabs(value[i]);
      localSum[i] += value[i];
      if (absValue > localMaxabs[i])
        localMaxabs[i] = absValue;
      if (value[i] < localMin[i])
        localMin[i] = value[i];
      if (value[i] > localMax[i])
        localMax[i] = value[i];
    }
  }
  count[tid] = localCount;
  pSum[tid] = localPSum;
  gammaSum[tid] = localGammaSum;
  for (i = 0; i < 7; i++) {
    sum[i][tid] = localSum[i];
    maxabs[i][tid] = localMaxabs[i];
    minValue[i][tid] = localMin[i];
    maxValue[i][tid] = localMax[i];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      pSum[tid] += pSum[tid + offset];
      gammaSum[tid] += gammaSum[tid + offset];
      for (i = 0; i < 7; i++) {
        sum[i][tid] += sum[i][tid + offset];
        if (maxabs[i][tid + offset] > maxabs[i][tid])
          maxabs[i][tid] = maxabs[i][tid + offset];
        if (minValue[i][tid + offset] < minValue[i][tid])
          minValue[i][tid] = minValue[i][tid + offset];
        if (maxValue[i][tid + offset] > maxValue[i][tid])
          maxValue[i][tid] = maxValue[i][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    GPU_BEAM_SUM_DATA *result = partial + blockIdx.x;
    result->count = count[0];
    result->pSum = pSum[0];
    result->gammaSum = gammaSum[0];
    for (i = 0; i < 7; i++) {
      result->centroidSum[i] = sum[i][0];
      result->maxabs[i] = maxabs[i][0];
      result->min[i] = minValue[i][0];
      result->max[i] = maxValue[i][0];
    }
  }
}

__global__ void gpuBeamStatisticsFinalizeKernel(
  const GPU_BEAM_SUM_DATA *partial, int blocks, GPU_BEAM_SUM_DATA *result,
  double *centroid) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double pSum[GPU_REDUCTION_THREADS];
  __shared__ double gammaSum[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  __shared__ double maxabs[7][GPU_REDUCTION_THREADS];
  __shared__ double minValue[7][GPU_REDUCTION_THREADS];
  __shared__ double maxValue[7][GPU_REDUCTION_THREADS];
  int tid = threadIdx.x;
  int i;

  count[tid] = 0;
  pSum[tid] = gammaSum[tid] = 0;
  for (i = 0; i < 7; i++) {
    sum[i][tid] = 0;
    maxabs[i][tid] = 0;
    minValue[i][tid] = DBL_MAX;
    maxValue[i][tid] = -DBL_MAX;
  }
  if (tid < blocks) {
    count[tid] = partial[tid].count;
    pSum[tid] = partial[tid].pSum;
    gammaSum[tid] = partial[tid].gammaSum;
    for (i = 0; i < 7; i++) {
      sum[i][tid] = partial[tid].centroidSum[i];
      maxabs[i][tid] = partial[tid].maxabs[i];
      minValue[i][tid] = partial[tid].min[i];
      maxValue[i][tid] = partial[tid].max[i];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      pSum[tid] += pSum[tid + offset];
      gammaSum[tid] += gammaSum[tid + offset];
      for (i = 0; i < 7; i++) {
        sum[i][tid] += sum[i][tid + offset];
        if (maxabs[i][tid + offset] > maxabs[i][tid])
          maxabs[i][tid] = maxabs[i][tid + offset];
        if (minValue[i][tid + offset] < minValue[i][tid])
          minValue[i][tid] = minValue[i][tid + offset];
        if (maxValue[i][tid + offset] > maxValue[i][tid])
          maxValue[i][tid] = maxValue[i][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->pSum = pSum[0];
    result->gammaSum = gammaSum[0];
    for (i = 0; i < 7; i++) {
      result->centroidSum[i] = sum[i][0];
      result->maxabs[i] = maxabs[i][0];
      result->min[i] = minValue[i][0];
      result->max[i] = maxValue[i][0];
      centroid[i] = count[0] ? sum[i][0] / count[0] : 0;
    }
  }
}

/* Spin sums for beam sums with spin tracking: the three spin components' sums, then their products
 * about the centroid, in the order of sigma[i][j] for i <= j.  Blocks are summed in a fixed order. */
__global__ void gpuSpinSumsPartialKernel(const double *coord, long nParticles, int stride, int spinOffset,
                                         const double *spinCentroid, double *partial) {
  __shared__ double sum[6][GPU_BEAM_OUTPUT_THREADS];
  long tid = threadIdx.x;
  double local[6] = {0, 0, 0, 0, 0, 0};
  long ip;
  int i;

  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    const double *spin = coord + ip * stride + spinOffset;
    if (!spinCentroid) {
      local[0] += spin[0];
      local[1] += spin[1];
      local[2] += spin[2];
    } else {
      double v0 = spin[0] - spinCentroid[0];
      double v1 = spin[1] - spinCentroid[1];
      double v2 = spin[2] - spinCentroid[2];
      local[0] += v0 * v0;
      local[1] += v0 * v1;
      local[2] += v0 * v2;
      local[3] += v1 * v1;
      local[4] += v1 * v2;
      local[5] += v2 * v2;
    }
  }
  for (i = 0; i < 6; i++)
    sum[i][tid] = local[i];
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset)
      for (i = 0; i < 6; i++)
        sum[i][tid] += sum[i][tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    for (i = 0; i < 6; i++)
      partial[blockIdx.x * 6 + i] = sum[i][0];
}

__global__ void gpuSpinSumsFinalizeKernel(const double *partial, int blocks, int values,
                                          long nParticles, double *result, double *spinCentroid) {
  int k = threadIdx.x;
  double total = 0;

  if (k >= values)
    return;
  for (int block = 0; block < blocks; block++)
    total += partial[block * 6 + k];
  result[k] = total;
  if (spinCentroid)
    spinCentroid[k] = total / nParticles;
}

__global__ void gpuCenteredBeamSumsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  const double *centroid, const double *timeValue, unsigned int productMask,
  GPU_BEAM_SUM_DATA *partial) {
  __shared__ int64_t count[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double product[28][GPU_BEAM_OUTPUT_THREADS];
  int64_t localCount;
  double localProduct[28];
  long tid = threadIdx.x;
  long ip;
  int i, j;

  localCount = 0;
  for (i = 0; i < 28; i++)
    localProduct[i] = 0;
  for (ip = (long)blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += (long)gridDim.x * blockDim.x) {
    double *part = coord + ip * stride;
    double value[7];
    value[0] = part[0] - centroid[0];
    value[1] = part[1] - centroid[1];
    value[2] = part[2] - centroid[2];
    value[3] = part[3] - centroid[3];
    value[4] = part[4] - centroid[4];
    value[5] = part[5] - centroid[5];
    value[6] = timeValue[ip] - centroid[6];
    localCount++;
    for (i = 0; i < 7; i++)
      for (j = i; j < 7; j++) {
        int index = gpuUpperTriangularIndex(i, j);
        if (productMask & (1U << index))
          localProduct[index] += value[i] * value[j];
      }
  }
  count[tid] = localCount;
  for (i = 0; i < 28; i++)
    product[i][tid] = localProduct[i];
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 28; i++)
        product[i][tid] += product[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    GPU_BEAM_SUM_DATA *result = partial + blockIdx.x;
    result->count = count[0];
    for (i = 0; i < 28; i++)
      result->productSum[i] = product[i][0];
  }
}

__global__ void gpuCenteredBeamSumsFinalizeKernel(
  const GPU_BEAM_SUM_DATA *partial, int blocks,
  GPU_BEAM_SUM_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double product[28][GPU_REDUCTION_THREADS];
  int tid = threadIdx.x;
  int i;

  count[tid] = 0;
  for (i = 0; i < 28; i++)
    product[i][tid] = 0;
  if (tid < blocks) {
    count[tid] = partial[tid].count;
    for (i = 0; i < 28; i++)
      product[i][tid] = partial[tid].productSum[i];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      for (i = 0; i < 28; i++)
        product[i][tid] += product[i][tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    for (i = 0; i < 28; i++)
      result->productSum[i] = product[i][0];
  }
}

__global__ void gpuLscStatisticsKernel(double *coord, long nParticles,
                                        int stride, double pCentral,
                                        double cMks,
                                        GPU_BEAM_SUM_DATA *result) {
  __shared__ double xSum[GPU_REDUCTION_THREADS];
  __shared__ double ySum[GPU_REDUCTION_THREADS];
  __shared__ double deltaSum[GPU_REDUCTION_THREADS];
  __shared__ double minTime[GPU_REDUCTION_THREADS];
  __shared__ double maxTime[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  xSum[tid] = 0;
  ySum[tid] = 0;
  deltaSum[tid] = 0;
  minTime[tid] = DBL_MAX;
  maxTime[tid] = -DBL_MAX;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double time = gpuParticleTime(part, pCentral, cMks);

    xSum[tid] += part[0];
    ySum[tid] += part[2];
    deltaSum[tid] += part[5];
    if (time < minTime[tid])
      minTime[tid] = time;
    if (time > maxTime[tid])
      maxTime[tid] = time;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      xSum[tid] += xSum[tid + offset];
      ySum[tid] += ySum[tid + offset];
      deltaSum[tid] += deltaSum[tid + offset];
      if (minTime[tid + offset] < minTime[tid])
        minTime[tid] = minTime[tid + offset];
      if (maxTime[tid + offset] > maxTime[tid])
        maxTime[tid] = maxTime[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = nParticles;
    result->centroidSum[0] = xSum[0];
    result->centroidSum[2] = ySum[0];
    /* lscBeamMomentum()'s sum over the binned particles, which are all of them unless the CPU warns */
    result->centroidSum[5] = deltaSum[0];
    result->min[6] = minTime[0];
    result->max[6] = maxTime[0];
  }
}

__global__ void gpuLscTransverseSumsKernel(double *coord, long nParticles,
                                            int stride, double xCentroid,
                                            double yCentroid,
                                            GPU_BEAM_SUM_DATA *result) {
  __shared__ double S11[GPU_REDUCTION_THREADS];
  __shared__ double S33[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  S11[tid] = 0;
  S33[tid] = 0;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double x = part[0] - xCentroid;
    double y = part[2] - yCentroid;

    S11[tid] += x * x;
    S33[tid] += y * y;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      S11[tid] += S11[tid + offset];
      S33[tid] += S33[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = nParticles;
    result->centroidSum[0] = xCentroid * nParticles;
    result->centroidSum[2] = yCentroid * nParticles;
    result->productSum[gpuUpperTriangularIndex(0, 0)] = S11[0];
    result->productSum[gpuUpperTriangularIndex(2, 2)] = S33[0];
  }
}

/* Count, least and greatest value of an integer property (the bunch number), one particle per thread.
 * It ran on one block of 64 threads, 2 ms for a million particles, and the wakes ask for it at every
 * call.  Comparisons and a count only, so the result is the same. */
#define GPU_LONG_MIN_MAX_THREADS 256

__global__ void gpuLongMinMaxPartialKernel(const double *coord, long nParticles, int stride,
                                           int coordinateIndex,
                                           GPU_LONG_MIN_MAX_DATA *partial) {
  __shared__ int64_t count[GPU_LONG_MIN_MAX_THREADS];
  __shared__ long minValue[GPU_LONG_MIN_MAX_THREADS];
  __shared__ long maxValue[GPU_LONG_MIN_MAX_THREADS];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  count[tid] = 0;
  minValue[tid] = LONG_MAX;
  maxValue[tid] = LONG_MIN;
  if (ip < nParticles) {
    long value = static_cast<long>(coord[ip * stride + coordinateIndex]);

    count[tid] = 1;
    minValue[tid] = maxValue[tid] = value;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x].count = count[0];
    partial[blockIdx.x].min = minValue[0];
    partial[blockIdx.x].max = maxValue[0];
  }
}

__global__ void gpuLongMinMaxFinalizeKernel(const GPU_LONG_MIN_MAX_DATA *partial, int blocks,
                                            GPU_LONG_MIN_MAX_DATA *result) {
  __shared__ int64_t count[GPU_LONG_MIN_MAX_THREADS];
  __shared__ long minValue[GPU_LONG_MIN_MAX_THREADS];
  __shared__ long maxValue[GPU_LONG_MIN_MAX_THREADS];
  int tid = threadIdx.x;

  count[tid] = 0;
  minValue[tid] = LONG_MAX;
  maxValue[tid] = LONG_MIN;
  for (int ib = tid; ib < blocks; ib += blockDim.x) {
    count[tid] += partial[ib].count;
    if (partial[ib].min < minValue[tid])
      minValue[tid] = partial[ib].min;
    if (partial[ib].max > maxValue[tid])
      maxValue[tid] = partial[ib].max;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->min = minValue[0];
    result->max = maxValue[0];
  }
}

__global__ void gpuSortedBunchValidateKernel(
  const double *coord, long nParticles, int stride, int coordinateIndex,
  int *sorted) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ip + 1 >= nParticles)
    return;
  if (static_cast<long>(coord[ip * stride + coordinateIndex]) >
      static_cast<long>(coord[(ip + 1) * stride + coordinateIndex]))
    atomicExch(sorted, 0);
}

__global__ void gpuSortedBunchRangesKernel(
  const double *coord, long nParticles, int stride, int coordinateIndex,
  long minBunch, long nBuckets, long *start, long *count) {
  long bucket = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long low, high, middle, first, last, target;

  if (bucket >= nBuckets)
    return;
  target = minBunch + bucket;
  low = 0;
  high = nParticles;
  while (low < high) {
    middle = low + (high - low) / 2;
    if (static_cast<long>(coord[middle * stride + coordinateIndex]) < target)
      low = middle + 1;
    else
      high = middle;
  }
  first = low;
  low = first;
  high = nParticles;
  while (low < high) {
    middle = low + (high - low) / 2;
    if (static_cast<long>(coord[middle * stride + coordinateIndex]) <= target)
      low = middle + 1;
    else
      high = middle;
  }
  last = low;
  start[bucket] = first;
  count[bucket] = last - first;
}

__global__ void gpuDoubleMinMaxKernel(double *coord, long nParticles, int stride,
                                      int coordinateIndex,
                                      GPU_DOUBLE_MIN_MAX_DATA *result) {
  __shared__ int64_t count[GPU_REDUCTION_THREADS];
  __shared__ double minValue[GPU_REDUCTION_THREADS];
  __shared__ double maxValue[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  minValue[tid] = DBL_MAX;
  maxValue[tid] = -DBL_MAX;

  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double value = part[coordinateIndex];
    if (isinf(value) || isnan(value))
      continue;
    count[tid]++;
    if (value < minValue[tid])
      minValue[tid] = value;
    if (value > maxValue[tid])
      maxValue[tid] = value;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->min = minValue[0];
    result->max = maxValue[0];
  }
}

static int launchTimedKernel(cudaError_t launchStatus, cudaEvent_t start, cudaEvent_t stop,
                             float *milliseconds) {
  cudaError_t status = launchStatus;

  if (status == cudaSuccess)
    status = cudaGetLastError();
  if (!start || !stop)
    return static_cast<int>(status);
  if (status == cudaSuccess)
    status = cudaEventRecord(stop, 0);
  if (status == cudaSuccess)
    status = cudaEventSynchronize(stop);
  if (milliseconds && status == cudaSuccess)
    cudaEventElapsedTime(milliseconds, start, stop);
  return static_cast<int>(status);
}

static int prepareTimedLaunch(cudaEvent_t *start, cudaEvent_t *stop, float *milliseconds) {
  static cudaEvent_t cachedStart = NULL, cachedStop = NULL;
  cudaError_t status;

  if (!start || !stop)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  if (!gpuDetailedKernelTimingEnabled()) {
    *start = NULL;
    *stop = NULL;
    return static_cast<int>(cudaSuccess);
  }
  status = static_cast<cudaError_t>(getCachedTimingEvents(&cachedStart,
                                                          &cachedStop));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  *start = cachedStart;
  *stop = cachedStop;
  status = cudaEventRecord(*start, 0);
  return static_cast<int>(status);
}

extern "C" int gpuCudaRuntimeGetDeviceCount(int *count) {
  return static_cast<int>(cudaGetDeviceCount(count));
}

extern "C" int gpuCudaRuntimeSetDevice(int device) {
  return static_cast<int>(cudaSetDevice(device));
}

__global__ void gpuDeviceCodeProbeKernel(int *value) {
  *value = 1;
}

extern "C" int gpuCudaRuntimeProbeDeviceCode(void) {
  int *value = NULL;
  cudaError_t status = cudaMalloc(&value, sizeof(*value));

  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuDeviceCodeProbeKernel<<<1, 1>>>(value);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  gpuFreeDevice(value);
  return static_cast<int>(status);
}

extern "C" const char *gpuCudaRuntimeGetErrorString(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int gpuCudaRuntimeGetDeviceName(int device, char *name, unsigned long nameSize) {
  cudaDeviceProp prop;
  cudaError_t status;

  if (!name || !nameSize)
    return static_cast<int>(cudaErrorInvalidValue);
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  std::strncpy(name, prop.name, nameSize - 1);
  name[nameSize - 1] = '\0';
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaMallocDouble(void **ptr, unsigned long count) {
  return static_cast<int>(cudaMalloc(ptr, count * sizeof(double)));
}

extern "C" int gpuCudaMallocBytes(void **ptr, unsigned long bytes) {
  return static_cast<int>(cudaMalloc(ptr, bytes));
}

extern "C" int gpuCudaFree(void *ptr) {
  return static_cast<int>(gpuFreeDevice(ptr));
}

extern "C" int gpuCudaCopyHostToDevice(void *dst, const void *src, unsigned long count, float *milliseconds) {
  return timeCopy(dst, src, count * sizeof(double), cudaMemcpyHostToDevice, milliseconds);
}

extern "C" int gpuCudaMemsetZeroDouble(void *data, unsigned long count) {
  return static_cast<int>(cudaMemset(data, 0, count * sizeof(double)));
}

extern "C" int gpuCudaCopyDeviceToHost(void *dst, const void *src, unsigned long count, float *milliseconds) {
  return timeCopy(dst, src, count * sizeof(double), cudaMemcpyDeviceToHost, milliseconds);
}

/* The ISR random numbers: the same sequence as gauss_rn_lim(0, 1, limit, random_2), from the state
 * of its generator.  dlaran is a multiplicative generator mod 2^48, so a pair of uniforms can be
 * computed directly from its position in the sequence, and rejected values are removed afterward. */
#define GPU_DLARAN_MULTIPLIER 33952834046453ULL
#define GPU_DLARAN_MASK ((1ULL << 48) - 1)

typedef struct GPU_ISR_JUMP {
  unsigned long long power[48]; /* the multiplier to the power 2^i */
} GPU_ISR_JUMP;

__global__ void gpuIsrPairKernel(double *value, int *accepted, long nPairs,
                                 unsigned long long firstState, double limit, GPU_ISR_JUMP jump) {
  long pair = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long state = firstState, steps;
  double u1, u2, z;

  if (pair >= nPairs)
    return;
  steps = 2 * (unsigned long long)pair;
  for (int bit = 0; steps; bit++, steps >>= 1)
    if (steps & 1)
      state = (state * jump.power[bit]) & GPU_DLARAN_MASK;
  u1 = state * 3.552713678800501e-15;
  state = (state * GPU_DLARAN_MULTIPLIER) & GPU_DLARAN_MASK;
  u2 = state * 3.552713678800501e-15;
  z = sqrt(-2 * log(u1)) * sin(6.28318530717958647692528676655900576 * u2);
  value[pair] = z;
  accepted[pair] = limit <= 0 || fabs(z) <= limit;
}

/* counts[0] is the number of accepted pairs and counts[1] the pair that made the last draw */
__global__ void gpuIsrScatterKernel(const double *value, const int *accepted, const int *rank, long nPairs,
                                    long count, double *draws, long *counts) {
  long pair = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (pair >= nPairs)
    return;
  if (pair == nPairs - 1)
    counts[0] = rank[pair] + accepted[pair];
  if (accepted[pair] && rank[pair] < count) {
    draws[rank[pair]] = value[pair];
    if (rank[pair] == count - 1)
      counts[1] = pair;
  }
}

/* Kept between calls, so a fill allocates nothing once the sizes stop growing */
static struct {
  double *draws;
  long drawCapacity;
  double *value;
  int *accepted; /* 32-bit flags and ranks: half the memory the prefix sum has to move */
  int *rank;
  long *counts;
  long pairCapacity;
  void *scanStorage;
  size_t scanStorageBytes;
} gpuIsrScratch;

extern "C" int gpuCudaIsrFillDraws(long count, unsigned long long firstState, double limit,
                                   const void **draws, long *uniformsUsed, float *milliseconds) {
  GPU_ISR_JUMP jump;
  const int threads = 256;
  double acceptance = limit > 0 ? std::erf(limit / std::sqrt(2.0)) : 1.0;
  long nPairs = (long)(count / acceptance * 1.01) + 1024;
  long counts[2] = {0, 0};
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int status;

  if (count <= 0 || !draws || !uniformsUsed || acceptance <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  /* the ranks are 32-bit; a fill this large would not fit the device in any case */
  if (nPairs > 1000000000L)
    return static_cast<int>(cudaErrorMemoryAllocation);
  jump.power[0] = GPU_DLARAN_MULTIPLIER;
  for (int i = 1; i < 48; i++)
    jump.power[i] = (jump.power[i - 1] * jump.power[i - 1]) & GPU_DLARAN_MASK;
  if (gpuIsrScratch.drawCapacity < count) {
    gpuFreeDevice(gpuIsrScratch.draws);
    gpuIsrScratch.draws = NULL;
    gpuIsrScratch.drawCapacity = 0;
    cudaStatus = cudaMalloc(&gpuIsrScratch.draws, count * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuIsrScratch.drawCapacity = count;
  }
  if (!gpuIsrScratch.counts) {
    cudaStatus = cudaMalloc(&gpuIsrScratch.counts, 2 * sizeof(long));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaSuccess;
  /* One wait per fill: the counts come back with the draws already in place.  Too few accepted pairs
   * (rare) repeats it with more, and the first pairs are the same. */
  do {
    int blocks;
    size_t scanBytes = 0;

    if (gpuIsrScratch.pairCapacity < nPairs) {
      gpuFreeDevice(gpuIsrScratch.value);
      gpuFreeDevice(gpuIsrScratch.accepted);
      gpuFreeDevice(gpuIsrScratch.rank);
      gpuIsrScratch.value = NULL;
      gpuIsrScratch.accepted = NULL;
      gpuIsrScratch.rank = NULL;
      gpuIsrScratch.pairCapacity = 0;
      cudaStatus = cudaMalloc(&gpuIsrScratch.value, nPairs * sizeof(double));
      if (cudaStatus == cudaSuccess)
        cudaStatus = cudaMalloc(&gpuIsrScratch.accepted, nPairs * sizeof(int));
      if (cudaStatus == cudaSuccess)
        cudaStatus = cudaMalloc(&gpuIsrScratch.rank, nPairs * sizeof(int));
      if (cudaStatus != cudaSuccess)
        break;
      gpuIsrScratch.pairCapacity = nPairs;
    }
    cudaStatus = cub::DeviceScan::ExclusiveSum(NULL, scanBytes, gpuIsrScratch.accepted,
                                               gpuIsrScratch.rank, nPairs);
    if (cudaStatus != cudaSuccess)
      break;
    if (gpuIsrScratch.scanStorageBytes < scanBytes) {
      gpuFreeDevice(gpuIsrScratch.scanStorage);
      gpuIsrScratch.scanStorage = NULL;
      gpuIsrScratch.scanStorageBytes = 0;
      cudaStatus = cudaMalloc(&gpuIsrScratch.scanStorage, scanBytes);
      if (cudaStatus != cudaSuccess)
        break;
      gpuIsrScratch.scanStorageBytes = scanBytes;
    }
    scanBytes = gpuIsrScratch.scanStorageBytes;
    blocks = static_cast<int>((nPairs + threads - 1) / threads);
    gpuIsrPairKernel<<<blocks, threads>>>(gpuIsrScratch.value, gpuIsrScratch.accepted, nPairs,
                                          firstState, limit, jump);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = cub::DeviceScan::ExclusiveSum(gpuIsrScratch.scanStorage, scanBytes,
                                                 gpuIsrScratch.accepted, gpuIsrScratch.rank, nPairs);
    if (cudaStatus != cudaSuccess)
      break;
    gpuIsrScatterKernel<<<blocks, threads>>>(gpuIsrScratch.value, gpuIsrScratch.accepted,
                                             gpuIsrScratch.rank, nPairs, count, gpuIsrScratch.draws,
                                             gpuIsrScratch.counts);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMemcpy(counts, gpuIsrScratch.counts, sizeof(counts), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      break;
    if (counts[0] < count) {
      nPairs += nPairs / 2;
      if (nPairs > 1000000000L) {
        cudaStatus = cudaErrorMemoryAllocation;
        break;
      }
    }
  } while (counts[0] < count);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  *draws = gpuIsrScratch.draws;
  *uniformsUsed = 2 * (counts[1] + 1);
  return static_cast<int>(cudaSuccess);
}

/* The same numbers made on the host, copied into the buffer the kernels read.
 * To look into: this copy blocks the host on every fill. On a shared GPU each wait can sit
 * behind other processes' kernels (forced srEffects2 took 172 s in a 4-job routine run,
 * 103 s with device numbers). A pinned, double-buffered cudaMemcpyAsync on the kernel's
 * stream would let the host make the next fill meanwhile; measure first that this wait
 * is the cause, since the device path also waits once per fill to read back its count. */
extern "C" int gpuCudaIsrUploadDraws(long count, const double *hostDraws, const void **draws,
                                     float *milliseconds) {
  cudaError_t cudaStatus;

  if (count <= 0 || !hostDraws || !draws)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuIsrScratch.drawCapacity < count) {
    gpuFreeDevice(gpuIsrScratch.draws);
    gpuIsrScratch.draws = NULL;
    gpuIsrScratch.drawCapacity = 0;
    cudaStatus = cudaMalloc(&gpuIsrScratch.draws, count * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuIsrScratch.drawCapacity = count;
  }
  *draws = gpuIsrScratch.draws;
  return timeCopy(gpuIsrScratch.draws, hostDraws, count * sizeof(double), cudaMemcpyHostToDevice,
                  milliseconds);
}

extern "C" int gpuCudaCopyDeviceToDevice(void *dst, const void *src, unsigned long count) {
  return static_cast<int>(cudaMemcpy(dst, src, count * sizeof(double), cudaMemcpyDeviceToDevice));
}

extern "C" int gpuCudaCopyHostBytesToDevice(void *dst, const void *src, unsigned long bytes) {
  return static_cast<int>(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

extern "C" int gpuCudaCopyDeviceBytesToHost(void *dst, const void *src, unsigned long bytes, float *milliseconds) {
  return timeCopy(dst, src, bytes, cudaMemcpyDeviceToHost, milliseconds);
}

extern "C" int gpuCudaStableScatterRows(void *coord, void *scratchCoord,
                                        const void *prefix,
                                        long nParticles, int stride,
                                        long survivors,
                                        float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  const int blockSize = 256;
  int gridSize;
  int status;

  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0 ||
      survivors < 0 || survivors > nParticles)
    return static_cast<int>(cudaErrorInvalidValue);
  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuStableScatterRowsKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), static_cast<double *>(scratchCoord),
    static_cast<const long *>(prefix), nParticles, stride, survivors);
  cudaStatus = cudaGetLastError();
  return launchTimedKernel(cudaStatus, start, stop, milliseconds);
}

__global__ void gpuGatherLostRowsKernel(const double *coord, double *scratch,
                                        const long *destination, long nParticles, int stride,
                                        long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long place;

  if (ip >= nParticles || (place = destination[ip]) < survivors)
    return;
  for (int ic = 0; ic < stride; ic++)
    scratch[place * stride + ic] = coord[ip * stride + ic];
}

/* The copy of the beam that a magnet's checked kernel made before tracking in place (reused buffer
 * 1), if it still holds nParticles rows */
static const double *gpuMagnetBackupRows(long nParticles, int stride) {
  const GPU_REUSED_DEVICE_BUFFER *buffer = gpuReusedDeviceBuffer + 1;

  if (!buffer->data ||
      buffer->bytes < static_cast<size_t>(nParticles) * stride * sizeof(double))
    return NULL;
  return static_cast<const double *>(buffer->data);
}

/* A magnet's checked kernel tracked in place and left prefix[i] = 1 for the particles it kept.  This
 * puts the rows, as tracked, into scratchCoord in the order the CPU's loss loop leaves them, with each
 * particle's place in prefix.  No second or third pass through the magnet: the flag and scatter kernels
 * of gpuCuda*TrackStableCompact() each tracked every particle again. */
extern "C" int gpuCudaCompactTrackedRows(void *coord, void *scratchCoord, void *prefix,
                                         long nParticles, int stride, long *remaining,
                                         float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
  thrust::exclusive_scan(flags, flags + nParticles, flags);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuBuildCpuLossOrderDestination(devicePrefix, nParticles, survivors);
  if (cudaStatus == cudaSuccess) {
    gpuStableScatterRowsKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

/* Puts the beam back as it was before a magnet's checked kernel tracked it in place */
extern "C" int gpuCudaRestoreMagnetBackup(void *coord, long nParticles, int stride) {
  const double *backup = gpuMagnetBackupRows(nParticles, stride);

  if (!coord || !backup || nParticles <= 0 || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaMemcpy(coord, backup,
                                     static_cast<size_t>(nParticles) * stride * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
}

/* After a compaction into scratchCoord with each particle's place in destination: the rows of the
 * lost particles from before the element, put at their places in scratchCoord's tail and copied to
 * hostRows in that order.  coord holds those rows when the compaction kernels tracked from it into
 * scratchCoord; with coord NULL they come from the magnet's backup, for gpuCudaCompactTrackedRows(). */
extern "C" int gpuCudaLostRowsBeforeElement(const void *coord, void *scratchCoord,
                                            const void *destination, long nParticles,
                                            int stride, long survivors, double *hostRows,
                                            float *milliseconds) {
  const int blockSize = 256;
  int gridSize;
  cudaError_t cudaStatus;

  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= survivors)
    return static_cast<int>(cudaSuccess);
  if (!coord && stride > 0)
    coord = gpuMagnetBackupRows(nParticles, stride);
  if (!coord || !scratchCoord || !destination || !hostRows || stride <= 0 || survivors < 0)
    return static_cast<int>(cudaErrorInvalidValue);
  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  gpuGatherLostRowsKernel<<<gridSize, blockSize>>>(
    static_cast<const double *>(coord), static_cast<double *>(scratchCoord),
    static_cast<const long *>(destination), nParticles, stride, survivors);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  return timeCopy(hostRows, static_cast<double *>(scratchCoord) + survivors * stride,
                  static_cast<size_t>(nParticles - survivors) * stride * sizeof(double),
                  cudaMemcpyDeviceToHost, milliseconds);
}

extern "C" int gpuCudaTrackParticles(void *coord, long nParticles, int stride,
                                     const GPU_MATRIX_DATA *matrix, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t status;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int matrixSlot;

  if (milliseconds)
    *milliseconds = 0;
  status = static_cast<cudaError_t>(uploadMatrixDataIfNeeded(matrix,
                                                              &matrixSlot));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  if (!gpuDetailedKernelTimingEnabled()) {
    gpuTrackParticlesKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, matrixSlot);
    return static_cast<int>(cudaGetLastError());
  }
  status = static_cast<cudaError_t>(getCachedTimingEvents(&start, &stop));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaEventRecord(start, 0);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuTrackParticlesKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                               nParticles, stride,
                                               matrixSlot);
  return finishTimedKernel(start, stop, milliseconds);
}

extern "C" int gpuCudaExactDrift(void *coord, long nParticles, int stride,
                                 double length, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t status;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);

  if (milliseconds)
    *milliseconds = 0;
  status = static_cast<cudaError_t>(getCachedTimingEvents(&start, &stop));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  cudaEventRecord(start, 0);
  gpuExactDriftKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride, length);
  return finishTimedKernel(start, stop, milliseconds);
}

extern "C" int gpuCudaLinearDrift(void *coord, long nParticles, int stride,
                                  double length, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t status;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);

  if (milliseconds)
    *milliseconds = 0;
  status = static_cast<cudaError_t>(getCachedTimingEvents(&start, &stop));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  cudaEventRecord(start, 0);
  gpuLinearDriftKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                            nParticles, stride, length);
  return finishTimedKernel(start, stop, milliseconds);
}

extern "C" int gpuCudaKickMapTrackChecked(void *coord, long nParticles,
                                          int stride,
                                          const GPU_KICKMAP_DATA *data,
                                          const double *xpFactor,
                                          const double *ypFactor,
                                          long points,
                                          long *lostCount,
                                          float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *backup = NULL;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  unsigned long long coordCount;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || !xpFactor || !ypFactor || !lostCount ||
      stride < 6 || nParticles < 0 || points <= 0 ||
      data->nKicks < 1 || data->nx <= 1 || data->ny <= 1 ||
      data->dxg == 0 || data->dyg == 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  coordCount = static_cast<unsigned long long>(nParticles) *
               static_cast<unsigned long long>(stride);
  cudaStatus = gpuReuseDeviceBuffer(1, &backup, coordCount * sizeof(*backup));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backup, coord, coordCount * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuKickMapTrackCheckedKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, *data,
    xpFactor, ypFactor, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount,
                          sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backup, coordCount * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaKickMapTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_KICKMAP_DATA *data, const double *xpFactor,
  const double *ypFactor, long points, double zStart, double pRef,
  long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  int *lossRound = NULL;
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !data || !xpFactor || !ypFactor ||
      stride < 6 || nParticles < 0 || points <= 0 ||
      data->nKicks < 1 || data->nx <= 1 || data->ny <= 1 ||
      data->dxg == 0 || data->dyg == 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  cudaStatus = gpuRoundLossRounds(nParticles, &lossRound);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuKickMapSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *data,
    xpFactor, ypFactor, devicePrefix, lossRound);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuRoundLossOrderDestination(
        devicePrefix + nParticles, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuKickMapStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, survivors, *data,
      xpFactor, ypFactor,
      zStart, pRef);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaMultipoleLoadCoefficients(const double *table) {
  return static_cast<int>(cudaMemcpyToSymbol(
    gpuMultipoleCoefficientTable, table, sizeof(gpuMultipoleCoefficientTable)));
}

extern "C" int gpuCudaMultipoleTrack(void *coord, long nParticles, int stride,
                                     const GPU_MULTIPOLE_DATA *multipole,
                                     int writeOutput, long *lostCount,
                                     float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int threads = writeOutput ? gpuParticleThreads(nParticles) : GPU_REDUCTION_THREADS;
  int blocks = writeOutput ? static_cast<int>((nParticles + threads - 1) / threads) : 1;
  int status;

  if (!coord || !multipole)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  if (lostCount)
    *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  cudaStatus = cudaMemcpyToSymbol(gpuMultipoleData, multipole, sizeof(*multipole));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  if (!writeOutput) {
    if (!lostCount)
      return static_cast<int>(cudaErrorInvalidValue);
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  if (writeOutput) {
    if (multipole->radCoef) {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipoleTrackKernel<true, true><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride);
      else
        gpuMultipoleTrackKernel<true, false><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride);
    } else {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipoleTrackKernel<false, true><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride);
      else
        gpuMultipoleTrackKernel<false, false><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride);
    }
  } else {
    if (multipole->radCoef) {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipolePredicateKernel<true, true><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount);
      else
        gpuMultipolePredicateKernel<true, false><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount);
    } else {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipolePredicateKernel<false, true><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount);
      else
        gpuMultipolePredicateKernel<false, false><<<blocks, threads>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount);
    }
  }
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  if (!writeOutput) {
    cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                            cudaMemcpyDeviceToHost);
    return static_cast<int>(cudaStatus);
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaMultipoleTrackChecked(void *coord, long nParticles,
                                            int stride,
                                            const GPU_MULTIPOLE_DATA *multipole,
                                            void *survivorFlags,
                                            long *lostCount,
                                            float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !multipole || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = cudaMemcpyToSymbol(gpuMultipoleData, multipole, sizeof(*multipole));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = gpuReuseDeviceBuffer(1, &backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  if (multipole->radCoef) {
    if (multipole->aperture.present || multipole->contour.nContours)
      gpuMultipoleTrackCheckedKernel<true, true>
        <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount,
          static_cast<long *>(survivorFlags));
    else
      gpuMultipoleTrackCheckedKernel<true, false>
        <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount,
          static_cast<long *>(survivorFlags));
  } else {
    if (multipole->aperture.present || multipole->contour.nContours)
      gpuMultipoleTrackCheckedKernel<false, true>
        <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount,
          static_cast<long *>(survivorFlags));
    else
      gpuMultipoleTrackCheckedKernel<false, false>
        <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
          static_cast<double *>(coord), nParticles, stride, deviceLostCount,
          static_cast<long *>(survivorFlags));
  }
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  /* With survivor flags the caller compacts the rows as tracked (gpuCudaCompactTrackedRows) and
   * takes what it needs of the rows from before the element out of the backup */
  if (cudaStatus == cudaSuccess && hostLostCount && !survivorFlags) {
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCcbendTrackChecked(void *coord, long nParticles,
                                          int stride,
                                          const GPU_CCBEND_DATA *ccbend,
                                          void *survivorFlags,
                                          long *lostCount,
                                          float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !ccbend || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = gpuReuseDeviceBuffer(1, &backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuCcbendTrackCheckedKernel<<<blocks, threads,
                                threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, *ccbend,
    deviceLostCount, static_cast<long *>(survivorFlags));
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount,
                          sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount && !survivorFlags)
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaLgbendTrackChecked(void *coord, long nParticles,
                                           int stride,
                                           const GPU_LGBEND_DATA *lgbend,
                                           const GPU_LGBEND_LOCAL_APERTURE_DATA
                                             *localAperture,
                                           long *lostCount,
                                           float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  GPU_LGBEND_LOCAL_APERTURE_DATA *deviceLocalAperture = NULL;
  unsigned long long count;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !lgbend || !lostCount || stride <= 0 ||
      lgbend->nSegments <= 0 ||
      lgbend->nSegments > GPU_LGBEND_MAX_SEGMENTS || lgbend->nSlices <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (localAperture && (localAperture->present || localAperture->hasFileAperture)) {
    if (localAperture->nSegments != lgbend->nSegments ||
        localAperture->nSlices != lgbend->nSlices ||
        localAperture->nSlices > GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES)
      return static_cast<int>(cudaErrorInvalidValue);
    cudaStatus =
      gpuReuseDeviceBuffer(0, &deviceLocalAperture, sizeof(*deviceLocalAperture));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    cudaStatus = cudaMemcpy(deviceLocalAperture, localAperture,
                            sizeof(*deviceLocalAperture),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = gpuReuseDeviceBuffer(1, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = gpuReuseDeviceBuffer(2, &backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuLgbendTrackCheckedKernel<<<blocks, threads,
                                threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, *lgbend,
    deviceLocalAperture,
    deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount,
                          sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount)
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCcbendTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_CCBEND_DATA *ccbend, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors = nParticles;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !ccbend || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCcbendSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *ccbend, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  }
  if (cudaStatus == cudaSuccess) {
    gpuCcbendStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, *ccbend);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaLgbendTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_LGBEND_DATA *lgbend,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_LGBEND_LOCAL_APERTURE_DATA *deviceLocalAperture = NULL;
  long survivors = nParticles;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  int *lossRound = NULL;
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !lgbend || stride <= 0 ||
      lgbend->nSegments <= 0 ||
      lgbend->nSegments > GPU_LGBEND_MAX_SEGMENTS ||
      lgbend->nSlices <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (localAperture && (localAperture->present || localAperture->hasFileAperture)) {
    if (localAperture->nSegments != lgbend->nSegments ||
        localAperture->nSlices != lgbend->nSlices ||
        localAperture->nSlices > GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES)
      return static_cast<int>(cudaErrorInvalidValue);
    cudaStatus =
      gpuReuseDeviceBuffer(0, &deviceLocalAperture, sizeof(*deviceLocalAperture));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    cudaStatus = cudaMemcpy(deviceLocalAperture, localAperture,
                            sizeof(*deviceLocalAperture),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  cudaStatus = gpuRoundLossRounds(nParticles, &lossRound);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuLgbendSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *lgbend,
    deviceLocalAperture, devicePrefix, lossRound);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuRoundLossOrderDestination(
        devicePrefix + nParticles, nParticles, survivors);
  }
  if (cudaStatus == cudaSuccess) {
    gpuLgbendStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, *lgbend,
      deviceLocalAperture);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaExactCorrectorTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_EXACT_CORRECTOR_DATA *corrector, long *remaining,
  long *usedScratch, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining || !usedScratch)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  *usedScratch = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !corrector || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuExactCorrectorSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *corrector,
    devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    if (survivors != nParticles) {
      thrust::exclusive_scan(flags, flags + nParticles, flags);
      cudaStatus = cudaGetLastError();
      if (cudaStatus == cudaSuccess)
        cudaStatus = gpuBuildCpuLossOrderDestination(
          devicePrefix, nParticles, survivors);
    }
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess && survivors != nParticles) {
    gpuExactCorrectorStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
    if (cudaStatus == cudaSuccess)
      *usedScratch = 1;
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaTaperApertureTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_TAPER_APERTURE_DATA *taper0, long *remaining,
  float *milliseconds, long *uncertainCount) {
  static unsigned long long *uncertain = NULL;
  GPU_TAPER_APERTURE_DATA taperCopy, *taper = &taperCopy;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !taper0 || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  taperCopy = *taper0;
  taperCopy.uncertain = NULL;
  if (uncertainCount) {
    *uncertainCount = 0;
    if (!uncertain) {
      cudaStatus = cudaMalloc(&uncertain, sizeof(*uncertain));
      if (cudaStatus != cudaSuccess)
        return static_cast<int>(cudaStatus);
    }
    cudaStatus = cudaMemset(uncertain, 0, sizeof(*uncertain));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    taperCopy.uncertain = uncertain;
  }

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuTaperApertureSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *taper,
    devicePrefix);
  /* the scatter kernel tests again; only the first pass counts uncertain tests */
  taperCopy.uncertain = NULL;
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuTaperApertureStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, survivors, *taper);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  if (status == static_cast<int>(cudaSuccess) && uncertainCount) {
    unsigned long long count = 0;
    cudaStatus = cudaMemcpy(&count, uncertain, sizeof(count), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    *uncertainCount = static_cast<long>(count);
  }
  return status;
}

extern "C" int gpuCudaSpeedbumpTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_SPEEDBUMP_DATA *speedbump, long *remaining,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !speedbump || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSpeedbumpSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *speedbump,
    devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuSpeedbumpStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, survivors, *speedbump);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaMultipoleTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_MULTIPOLE_DATA *multipole, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !multipole || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  cudaStatus = cudaMemcpyToSymbol(gpuMultipoleData, multipole,
                                  sizeof(*multipole));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  if (multipole->radCoef) {
    if (multipole->aperture.present || multipole->contour.nContours)
      gpuMultipoleSurvivorFlagKernel<true, true><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), nParticles, stride, devicePrefix);
    else
      gpuMultipoleSurvivorFlagKernel<true, false><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), nParticles, stride, devicePrefix);
  } else {
    if (multipole->aperture.present || multipole->contour.nContours)
      gpuMultipoleSurvivorFlagKernel<false, true><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), nParticles, stride, devicePrefix);
    else
      gpuMultipoleSurvivorFlagKernel<false, false><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), nParticles, stride, devicePrefix);
  }
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    if (multipole->radCoef) {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipoleStableTrackScatterKernel<true, true>
          <<<gridSize, blockSize>>>(
            static_cast<double *>(coord), static_cast<double *>(scratchCoord),
            devicePrefix + nParticles, nParticles, stride, survivors);
      else
        gpuMultipoleStableTrackScatterKernel<true, false>
          <<<gridSize, blockSize>>>(
            static_cast<double *>(coord), static_cast<double *>(scratchCoord),
            devicePrefix + nParticles, nParticles, stride, survivors);
    } else {
      if (multipole->aperture.present || multipole->contour.nContours)
        gpuMultipoleStableTrackScatterKernel<false, true>
          <<<gridSize, blockSize>>>(
            static_cast<double *>(coord), static_cast<double *>(scratchCoord),
            devicePrefix + nParticles, nParticles, stride, survivors);
      else
        gpuMultipoleStableTrackScatterKernel<false, false>
          <<<gridSize, blockSize>>>(
            static_cast<double *>(coord), static_cast<double *>(scratchCoord),
            devicePrefix + nParticles, nParticles, stride, survivors);
    }
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

__device__ __forceinline__ int gpuLoadSymplecticFractions(int integrationOrder,
                                                          double *driftFrac,
                                                          double *kickFrac,
                                                          int *nSubsteps) {
  switch (integrationOrder) {
  case 2:
    *nSubsteps = 2;
    driftFrac[0] = 0.5;
    driftFrac[1] = 0.5;
    kickFrac[0] = 1.0;
    kickFrac[1] = 0.0;
    return 1;
  case 4: {
    const double beta = 1.25992104989487316477;
    *nSubsteps = 4;
    driftFrac[0] = 0.5 / (2 - beta);
    driftFrac[1] = (1 - beta) / (2 - beta) / 2;
    driftFrac[2] = (1 - beta) / (2 - beta) / 2;
    driftFrac[3] = 0.5 / (2 - beta);
    kickFrac[0] = 1.0 / (2 - beta);
    kickFrac[1] = -beta / (2 - beta);
    kickFrac[2] = 1.0 / (2 - beta);
    kickFrac[3] = 0.0;
    return 1;
  }
  case 6:
    *nSubsteps = 8;
    driftFrac[0] = 0.39225680523878;
    driftFrac[1] = 0.5100434119184585;
    driftFrac[2] = -0.47105338540975655;
    driftFrac[3] = 0.0687531682525181;
    driftFrac[4] = 0.0687531682525181;
    driftFrac[5] = -0.47105338540975655;
    driftFrac[6] = 0.5100434119184585;
    driftFrac[7] = 0.39225680523878;
    kickFrac[0] = 0.784513610477560;
    kickFrac[1] = 0.235573213359357;
    kickFrac[2] = -1.17767998417887;
    kickFrac[3] = 1.3151863206839063;
    kickFrac[4] = -1.17767998417887;
    kickFrac[5] = 0.235573213359357;
    kickFrac[6] = 0.784513610477560;
    kickFrac[7] = 0.0;
    return 1;
  default:
    break;
  }
  return 0;
}

__device__ __forceinline__ void gpuCsbendFields(double *Fx, double *Fy,
                                                double x, double y,
                                                const GPU_CSBEND_DATA *data) {
  double yp[11];
  double sumFx = 0;
  double sumFy = 0;
  double xt = 1;

  if (!data->hasSkew && !data->hasNormal) {
    *Fx = 0;
    *Fy = 1;
    return;
  }

  yp[0] = 1;
  for (int i = 1; i < data->expansionOrder1; i++)
    yp[i] = yp[i - 1] * y;

  if (data->hasSkew) {
    for (int i = 0; i < data->expansionOrder1; i++) {
      for (int j = 0; j < data->expansionOrder1 - i; j++) {
        int ij = i * 11 + j;
        sumFx += data->Fx[ij] * xt * yp[j];
        sumFy += data->Fy[ij] * xt * yp[j];
      }
      xt *= x;
    }
  } else {
    for (int i = 0; i < data->expansionOrder1; i++) {
      for (int j = 0; j < data->expansionOrder1 - i; j++) {
        int ij = i * 11 + j;
        if (j & 1)
          sumFx += data->Fx[ij] * xt * yp[j];
        else
          sumFy += data->Fy[ij] * xt * yp[j];
      }
      xt *= x;
    }
  }
  *Fx = sumFx;
  *Fy = sumFy;
}

__device__ __forceinline__ void gpuCsbendFields(double *Fx, double *Fy,
                                                double x, double y) {
  gpuCsbendFields(Fx, Fy, x, y, &gpuCsbendData);
}

/* tanEdge is tan(e) from the host, the number the CPU's tan(e1) gives.  With no fringe angle (psi zero)
 * the vertical kick takes it too, so that this edge has no function from the device's library, which can
 * round differently from the host's.  With one, tan(e - psi/(1+delta)) is different for every particle. */
__device__ __forceinline__ int gpuCsbendApplyEdge(double *xp, double *yp,
                                                  double x, double y,
                                                  double dp, double e,
                                                  double tanEdge,
                                                  double psi,
                                                  double kickLimit,
                                                  const GPU_CSBEND_DATA *data) {
  double onePlusDp = 1 + dp;
  double rho;
  double deltaXp;

  if (onePlusDp == 0)
    return 0;
  rho = onePlusDp * data->rhoActual;
  if (rho == 0)
    return 0;
  deltaXp = tanEdge / rho * x;
  if (kickLimit > 0 && fabs(deltaXp) > kickLimit)
    deltaXp = copysign(kickLimit, deltaXp);
  *xp += deltaXp;
  *yp -= (psi ? tan(e - psi / onePlusDp) : tanEdge) / rho * y;
  return isfinite(*xp) && isfinite(*yp);
}

__device__ __forceinline__ int gpuCsbendApplyHigherOrderEdge(double *x,
                                                             double *xp,
                                                             double *y,
                                                             double *yp,
                                                             double dp,
                                                             double beta,
                                                             const double *edgeTrig,
                                                             double he,
                                                             double psi,
                                                             int whichEdge,
                                                             const GPU_CSBEND_DATA *data) {
  double onePlusDp = 1 + dp;
  double rho, h, h2, tanBeta, tan2Beta, cosBeta, secBeta, sec2Beta;
  double R21, R43;
  double T111, T133, T211, T441, T331, T221, T233, T243, T431, T432;
  double x0 = *x;
  double xp0 = *xp;
  double y0 = *y;
  double yp0 = *yp;

  if (onePlusDp == 0)
    return 0;
  rho = onePlusDp * data->rhoActual;
  if (rho == 0)
    return 0;
  /* the host's cos(beta) and tan(beta), and its tangent again where there is no fringe angle: see
   * gpuCsbendApplyEdge() */
  cosBeta = edgeTrig[3];
  if (cosBeta == 0)
    return 0;

  h = 1 / rho;
  tanBeta = edgeTrig[1];
  R21 = h * tanBeta;
  R43 = -h * (psi ? tan(beta - psi * onePlusDp) : tanBeta);

  h2 = h * h;
  tan2Beta = tanBeta * tanBeta;
  T111 = whichEdge * h / 2 * tan2Beta;
  secBeta = 1 / cosBeta;
  sec2Beta = secBeta * secBeta;
  T133 = -whichEdge * h / 2 * sec2Beta;
  T211 = whichEdge == -1 ?
           -data->fieldIndex * h2 * tanBeta :
           -h2 * (data->fieldIndex + tan2Beta / 2) * tanBeta;
  T441 = -(T331 = T221 = -whichEdge * h * tan2Beta);
  T233 = whichEdge == -1 ?
           h2 * (data->fieldIndex + 0.5 + tan2Beta) * tanBeta :
           h2 * (data->fieldIndex - tan2Beta / 2) * tanBeta;
  T243 = whichEdge * h * tan2Beta;
  T431 = h2 * (2 * data->fieldIndex +
               (whichEdge == 1 ? sec2Beta : 0)) * tanBeta;
  T432 = whichEdge * h * sec2Beta;
  if (he != 0) {
    double term = h / 2 * he * sec2Beta * secBeta;
    T211 += term;
    T233 -= term;
    T431 -= 2 * term;
  }

  /* apply_edge_effects() squares first: T111 * sqr(x0) */
  *x = x0 + T111 * (x0 * x0) + T133 * (y0 * y0);
  *xp = xp0 + R21 * x0 + T211 * (x0 * x0) + T221 * x0 * xp0 +
        T233 * (y0 * y0) + T243 * y0 * yp0;
  *y = y0 + T331 * x0 * y0;
  *yp = yp0 + R43 * y0 + T441 * yp0 * x0 + T431 * x0 * y0 +
        T432 * xp0 * y0;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) && isfinite(*yp);
}

__device__ int gpuCsbendApplyCurvedDipoleFringe(
  double *x, double *xp, double *y, double *yp, double *path, double dp,
  double edge, int inFringe, const double *integrals,
  const GPU_CSBEND_DATA *data) {
  double x0 = *x, y0 = *y, px0, py0;
  double x1, px1, y1, py1, x2, px2, y2, py2;
  double dtau, mx, my, delX, focX, focY;
  double secEdge = 1 / cos(edge);
  double tanEdge = tan(edge);
  double onePlusDp = 1 + dp;
  double intK0 = integrals[0] * (data->edgeFlip ? -1 : 1);
  double intK1 = integrals[1];
  double intK2 = integrals[2];
  double intK3 = integrals[3] * (data->edgeFlip ? -1 : 1);
  double intK4 = integrals[4] * (data->edgeFlip ? -1 : 1);
  double intK5 = integrals[5];
  double intK6 = integrals[6];

  if (onePlusDp == 0 ||
      !gpuDipoleConvertSlopesToMomenta(
        &px0, &py0, *xp, *yp, dp, data->expandHamiltonian))
    return 0;

  mx = (intK3 + intK4) / onePlusDp;
  my = intK3 / onePlusDp;
  delX = intK0 / onePlusDp;
  focX = intK1 + tanEdge / data->rhoActual;
  focY = intK2 / onePlusDp;

  if (fabs(mx) < 1e-5) {
    x1 = x0 * exp(-mx) +
         dp * delX * (1 - 0.5 * mx * (1 - mx / 3 + mx * mx / 12));
    px1 =
      px0 * exp(mx) +
      focX * (x0 * (1 + mx * mx / 6) +
              dp * delX * (0.5 + mx * mx / 24));
    dtau =
      px0 * (mx * x0 + delX) / onePlusDp +
      delX * px0 * 0.5 * mx * (1 + mx / 3 + mx * mx / 12) +
      focX * x0 * x0 * mx * (0.5 - (mx - 0.5 * mx * mx) / 3) /
        onePlusDp +
      dp * focX * delX * (delX + mx * x0 / onePlusDp) *
        (1 + 0.05 * mx * mx) / 6 +
      focX * delX * x0 * 0.5 * (1 + mx * mx / 12) / onePlusDp +
      dp * focX * delX * (2 * mx * x0 - dp * delX) *
        (1 - mx * (0.75 - mx * (0.35 - 0.125 * mx))) /
        (6 * onePlusDp);
  } else {
    x1 = x0 * exp(-mx) + dp * delX * (1 - exp(-mx)) / mx;
    px1 = px0 * exp(mx) +
          focX * (x0 * sinh(mx) / mx +
                  dp * delX * (cosh(mx) - 1) / (mx * mx));
    dtau =
      px0 * (mx * x0 + delX) / onePlusDp +
      delX * px0 * (exp(mx) - 1 - mx) / mx +
      focX * x0 * x0 * 0.25 *
        ((exp(-2 * mx) - 1) / mx + 2) / onePlusDp +
      dp * focX * delX * (delX + mx * x0 / onePlusDp) *
        (sinh(mx) - mx) / (mx * mx * mx) +
      focX * delX * x0 * (cosh(mx) - 1) /
        (mx * mx * onePlusDp) +
      dp * focX * delX * (2 * mx * x0 - dp * delX) * 0.25 *
        (2 - (3 + exp(-2 * mx) - 4 * exp(-mx)) / mx) /
        (mx * mx * onePlusDp);
  }
  if (fabs(my) < 1e-5) {
    y1 = y0 * exp(my);
    py1 = py0 * exp(-my) +
          y0 * (focY - focX) * (1 + my * my / 6);
    dtau +=
      (-my * py0 * y0 +
       0.5 * (focY + focX * (my + my * my * (2 + my) / 3) *
                         y0 * y0)) /
      onePlusDp;
  } else {
    y1 = y0 * exp(my);
    py1 = py0 * exp(-my) +
          y0 * (focY - focX) * sinh(my) / my;
    dtau +=
      (-my * py0 * y0 +
       0.5 * (focY +
              0.5 * focX * ((exp(2 * my) - 1) / my - 2)) *
         y0 * y0) /
      onePlusDp;
  }

  if (data->edgeOrder > 1) {
    double sec2Rho =
      secEdge * secEdge / (data->rhoActual * onePlusDp);
    double tanRho = tanEdge / (data->rhoActual * onePlusDp);

    if (inFringe == -1) {
      x2 = x1;
      y2 = y1 * exp(tanEdge * tanRho * x1);
      px2 =
        px1 - tanEdge * tanRho * y1 * py1 -
        (tanEdge * (tanEdge * tanRho / data->rhoActual - data->K1) -
         0.5 * intK6) *
          x1 * x1;
      py2 = py1 * exp(-tanEdge * tanRho * x1);
      dtau +=
        -tanEdge * tanRho * tanRho * x1 * x1 * x1 / 3 -
        tanRho * tanRho * data->rhoActual * x1 * y1 * py1;

      x1 = x2 + 0.5 * sec2Rho * y2 * y2;
      y1 = y2;
      px1 =
        px2 +
        (0.5 * tanRho / data->rhoActual - 0.5 * intK5 -
         data->K1 * tanEdge + 0.25 * intK1 * sec2Rho) *
          y2 * y2;
      py1 =
        py2 +
        (tanRho / data->rhoActual - intK5 -
         2 * data->K1 * tanEdge + 0.5 * intK1 * sec2Rho) *
          x2 * y2 -
        sec2Rho * y2 * px2;
      dtau +=
        (0.5 * sec2Rho *
           (0.25 * (2 * tanEdge * data->K1 + intK5) * y2 * y2 -
            px2) +
         (0.5 * tanRho / data->rhoActual +
          0.25 * intK1 * sec2Rho) *
           x2) *
        y2 * y2 / onePlusDp;

      {
        double factor = 1 + 0.5 * tanEdge * tanRho * x1;
        if (factor == 0)
          return 0;
        x2 = x1 / factor;
        y2 = y1;
        px2 = px1 * factor * factor;
        py2 = py1;
      }
      dtau += 0.5 * tanRho * tanRho * data->rhoActual *
               x1 * x1 * px1;
    } else {
      x2 = x1;
      y2 = y1 * exp(-tanEdge * tanRho * x1);
      px2 =
        px1 + tanEdge * tanRho * y1 * py1 +
        (tanEdge *
           (0.5 * tanEdge * tanRho / data->rhoActual + data->K1) +
         0.5 * intK6) *
          x1 * x1;
      py2 = py1 * exp(tanEdge * tanRho * x1);
      dtau +=
        tanEdge * tanRho * tanRho * x1 * x1 * x1 / 6 +
        tanRho * tanRho * data->rhoActual * x1 * y1 * py1;

      x1 = x2 - 0.5 * sec2Rho * y2 * y2;
      y1 = y2;
      px1 =
        px2 +
        (0.5 * tanEdge * tanEdge * tanRho / data->rhoActual -
         0.5 * intK5 - data->K1 * tanEdge -
         0.25 * intK1 * sec2Rho) *
          y2 * y2;
      py1 =
        py2 +
        (tanEdge * tanEdge * tanRho / data->rhoActual - intK5 -
         2 * data->K1 * tanEdge - 0.5 * intK1 * sec2Rho) *
          x2 * y2 +
        sec2Rho * y2 * px2;
      dtau +=
        (-0.5 * sec2Rho *
           (0.25 * (2 * tanEdge * data->K1 + intK5) * y2 * y2 -
            px2) +
         (0.5 * tanEdge * tanEdge * tanRho / data->rhoActual -
          0.25 * intK1 * sec2Rho) *
           x2) *
        y2 * y2 / onePlusDp;

      {
        double factor = 1 - 0.5 * tanEdge * tanRho * x1;
        if (factor == 0)
          return 0;
        x2 = x1 / factor;
        y2 = y1;
        px2 = px1 * factor * (1 + 0.5 * tanEdge * tanRho * x1);
        py2 = py1;
      }
      dtau -= 0.5 * tanRho * tanRho * data->rhoActual *
               x1 * x1 * px1;
    }
  } else {
    x2 = x1;
    px2 = px1;
    y2 = y1;
    py2 = py1;
  }

  /*
   * curvedDipoleFringe returns each canonical coordinate as q0+(q2-q0).
   * Preserve that subtraction/addition sequence instead of assigning q2
   * directly: it is observable at one ulp for near-zero exit slopes.
   */
  x2 = x0 + (x2 - x0);
  px2 = px0 + (px2 - px0);
  y2 = y0 + (y2 - y0);
  py2 = py0 + (py2 - py0);
  if (!gpuDipoleConvertMomentaToSlopes(
        xp, yp, px2, py2, dp, data->expandHamiltonian))
    return 0;
  *x = x2;
  *y = y2;
  *path -= dtau;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) &&
         isfinite(*yp) && isfinite(*path);
}

#define GPU_IPOW1(x) (x)
#define GPU_IPOW2(x) ((x) * (x))
#define GPU_IPOW3(x) ((x) * (x) * (x))

/* dipoleFringeKHwang() (EDGE_EFFECTS=2) and dipoleFringeKHwangRLindberg() (4), term for term, in the
 * canonical coordinates track_through_csbend() gives them.  The edge angle's trig comes from the
 * host.  Neither changes s or delta */
__device__ __noinline__ int gpuCsbendApplyHwangFringe(
  double *x, double *xp, double *y, double *yp, double dp, double inFringe, int symplectic,
  const double *trig, double fint, double Rhe, const GPU_CSBEND_DATA *data) {
  const double rho = data->rhoActual, gap = data->fringeGap;
  const double sec_edge = trig[0], tan_edge = trig[1], sin_edge = trig[2], cos_edge = trig[3];
  const double k0 = GPU_IPOW2(M_PI) / 6., k2 = fint, k3 = 1.0 * 1. / 6.;
  const double k4 = -1.0 * GPU_IPOW2(M_PI) / 3., k5 = 0.0, k6 = -1.0;
  const long edgeOrder = data->edgeOrder;
  double x0 = *x, px0, y0 = *y, py0, dp0 = dp;

  if (!gpuDipoleConvertSlopesToMomenta(&px0, &py0, *xp, *yp, dp, data->expandHamiltonian))
    return 0;
  if (!symplectic) {
    double dx, dpx, dy, dpy;

    dx = dpx = dy = dpy = 0;
    if (edgeOrder > 1) {

      /* entrance */
      if (inFringe == -1.) {
        dx = inFringe * GPU_IPOW2(sec_edge) * GPU_IPOW2(gap) * k0 / rho / (1 + dp0) + inFringe * GPU_IPOW2(x0) * GPU_IPOW2(tan_edge) / 2 / rho / (1 + dp0) - inFringe * GPU_IPOW2(y0) * GPU_IPOW2(sec_edge) / 2 / rho / (1 + dp0);
        dy = -inFringe * x0 * y0 * GPU_IPOW2(tan_edge) / rho / (1 + dp0);
        dpx = -1. * GPU_IPOW3(sec_edge) * sin_edge * GPU_IPOW2(gap) * k0 / rho / rho / (1 + dp0) + tan_edge * x0 / rho + GPU_IPOW2(y0) / 2 * (2 * GPU_IPOW3(tan_edge)) / GPU_IPOW2(rho) / (1 + dp0) + GPU_IPOW2(y0) / 2 * (GPU_IPOW1(tan_edge)) / GPU_IPOW2(rho) / (1 + dp0) - inFringe * (x0 * px0 - y0 * py0) * GPU_IPOW2(tan_edge) / rho / (1 + dp0) + k4 * GPU_IPOW2(sin_edge) * GPU_IPOW2(gap) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe - k5 * x0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe + k6 * (y0 * y0 - x0 * x0) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe;
        dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + GPU_IPOW2(sin_edge)) * gap / (1 + dp0) / GPU_IPOW2(rho) / GPU_IPOW3(cos_edge) + inFringe * (x0 * py0 + y0 * px0) * GPU_IPOW2(tan_edge) / rho / (1 + dp0) + inFringe * y0 * px0 / rho / (1 + dp0) + k3 * GPU_IPOW3(y0) * (2. / 3. / cos_edge - 4. / 3. / GPU_IPOW3(cos_edge)) / (1 + dp0) / rho / rho / gap + k6 * x0 * y0 / GPU_IPOW3(cos_edge) / rho * Rhe;
      }
      /* exit */
      if (inFringe == 1.) {
        dx = inFringe * GPU_IPOW2(sec_edge) * GPU_IPOW2(gap) * k0 / rho / (1 + dp0) + inFringe * GPU_IPOW2(x0) * GPU_IPOW2(tan_edge) / 2 / rho / (1 + dp0) - inFringe * GPU_IPOW2(y0) * GPU_IPOW2(sec_edge) / 2 / rho / (1 + dp0);
        dy = -inFringe * x0 * y0 * GPU_IPOW2(tan_edge) / rho / (1 + dp0);
        dpx = tan_edge * x0 / rho - GPU_IPOW2(y0) / 2 * (1 * GPU_IPOW3(tan_edge)) / GPU_IPOW2(rho) / (1 + dp0) - GPU_IPOW2(x0) / 2 * (1 * GPU_IPOW3(tan_edge)) / GPU_IPOW2(rho) / (1 + dp0) - inFringe * (x0 * px0 - y0 * py0) * GPU_IPOW2(tan_edge) / rho / (1 + dp0) + k4 * GPU_IPOW2(sin_edge) * GPU_IPOW2(gap) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe - k5 * x0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe + k6 * (y0 * y0 - x0 * x0) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe;
        dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + GPU_IPOW2(sin_edge)) * gap / (1 + dp0) / GPU_IPOW2(rho) / GPU_IPOW3(cos_edge) + inFringe * (x0 * py0 + y0 * px0) * GPU_IPOW2(tan_edge) / rho / (1 + dp0) + inFringe * y0 * px0 / rho / (1 + dp0) + x0 * y0 * GPU_IPOW2(sec_edge) * tan_edge / GPU_IPOW2(rho) / (1 + dp0) + k3 * GPU_IPOW3(y0) * (2. / 3. / cos_edge - 4. / 3. / GPU_IPOW3(cos_edge)) / (1 + dp0) / rho / rho / gap - k5 * y0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe + k6 * x0 * y0 / GPU_IPOW3(cos_edge) / rho * Rhe;
      }

    } else {
      /* linear terms in transverse coordinates only */

      /* entrance */
      if (inFringe == -1.) {
        dx = inFringe * GPU_IPOW2(sec_edge) * GPU_IPOW2(gap) * k0 / rho / (1 + dp0);
        dy = 0;
        dpx = -1. * GPU_IPOW3(sec_edge) * sin_edge * GPU_IPOW2(gap) * k0 / rho / rho / (1 + dp0) + tan_edge * x0 / rho + k4 * GPU_IPOW2(sin_edge) * GPU_IPOW2(gap) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe - k5 * x0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe;
        dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + GPU_IPOW2(sin_edge)) * gap / (1 + dp0) / GPU_IPOW2(rho) / GPU_IPOW3(cos_edge);
      }

      /* exit */
      if (inFringe == 1.) {
        dx = inFringe * GPU_IPOW2(sec_edge) * GPU_IPOW2(gap) * k0 / rho / (1 + dp0);
        dy = 0;
        dpx = tan_edge * x0 / rho + k4 * GPU_IPOW2(sin_edge) * GPU_IPOW2(gap) / 2 / GPU_IPOW3(cos_edge) / rho * Rhe - k5 * x0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe;
        dpy = -1. * tan_edge * y0 / rho + k2 * y0 * (1 + GPU_IPOW2(sin_edge)) * gap / (1 + dp0) / GPU_IPOW2(rho) / GPU_IPOW3(cos_edge) - k5 * y0 * GPU_IPOW1(sin_edge) * GPU_IPOW1(gap) / GPU_IPOW3(cos_edge) / rho * Rhe;
      }
    }

    *x = x0 + dx;
    *y = y0 + dy;
    px0 += dpx;
    py0 += dpy;
  } else {
    const double rho2 = GPU_IPOW2(rho);
    const double sec2_edge = GPU_IPOW2(sec_edge), cos3_edge = GPU_IPOW3(cos_edge);
    const double tan2_edge = GPU_IPOW2(tan_edge), tan3_edge = GPU_IPOW3(tan_edge);
    double x1, px1, y1, py1, x2, px2, y2, py2, x3, px3, y3, py3;
    double x4, px4, y4, py4, x5, px5, y5, py5, t1, t2;

    if (inFringe == -1) {
      /* entrance */

      x1 = x0;
      px1 = px0 + tan_edge / rho * x0 + tan_edge / (2 * rho2 * (1 + dp0)) * GPU_IPOW2(y0) - tan3_edge / (rho2 * (1 + dp0)) * GPU_IPOW2(x0) - gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * x0 + k6 * sec2_edge * Rhe / (2 * rho) * (GPU_IPOW2(y0) - GPU_IPOW2(x0));
      y1 = y0;
      py1 = py0 - tan_edge / rho * y0 + tan_edge / (rho2 * (1 + dp0)) * x0 * y0 + gap * k2 * (1 + GPU_IPOW2(sin_edge)) / (rho2 * (1 + dp0) * cos3_edge) * y0 + 2 * k3 * (GPU_IPOW2(cos_edge) - 2) / (3 * gap * rho2 * cos3_edge) * GPU_IPOW3(y0) + gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * y0 + k6 * GPU_IPOW3(sec_edge) * Rhe / rho * x0 * y0;

      t1 = (1 + tan2_edge / (2 * rho * (1 + dp0)) * x1);
      x2 = x1 / t1;
      px2 = px1 * GPU_IPOW2(t1);
      y2 = y1;
      py2 = py1;

      t1 = sec2_edge / (rho * (1 + dp0));
      x3 = x2 + t1 / 2 * GPU_IPOW2(y2);
      px3 = px2;
      y3 = y2;
      py3 = py2 - t1 * y2 * px2;

      t1 = tan2_edge / (rho * (1 + dp0));
      t2 = exp(t1 * x3);
      x4 = x3;
      px4 = px3 - t1 * y3 * py3;
      y4 = y3 * t2;
      py4 = py3 / t2;

      t1 = GPU_IPOW2(gap) * k0 * sec2_edge / (rho * (1 + dp0));
      x5 = x4 - t1;
      px5 = px4 - t1 * tan_edge / rho + GPU_IPOW2(gap) * k4 * GPU_IPOW2(sin_edge) * Rhe / (2 * rho * cos3_edge);
      y5 = y4;
      py5 = py4;
    } else {
      /* exit */

      x1 = x0;
      px1 = px0 + tan_edge / rho * x0 + tan3_edge / (2 * rho2 * (1 + dp0)) * GPU_IPOW2(y0) + tan3_edge / (2 * rho2 * (1 + dp0)) * GPU_IPOW2(x0) - gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * x0 + k6 * sec2_edge * Rhe / (2 * rho) * (GPU_IPOW2(y0) - GPU_IPOW2(x0));
      y1 = y0;
      py1 = py0 - tan_edge / rho * y0 + tan3_edge / (rho2 * (1 + dp0)) * x0 * y0 + gap * k2 * (1 + GPU_IPOW2(sin_edge)) / (rho2 * (1 + dp0) * cos3_edge) * y0 + 2 * k3 * (GPU_IPOW2(cos_edge) - 2) / (3 * gap * rho2 * cos3_edge) * GPU_IPOW3(y0) + gap * k5 * sin_edge * Rhe / (rho * cos3_edge) * y0 + k6 * GPU_IPOW3(sec_edge) * Rhe / rho * x0 * y0;

      t1 = (1 - tan2_edge / (2 * rho * (1 + dp0)) * x1);
      x2 = x1 / t1;
      px2 = px1 * GPU_IPOW2(t1);
      y2 = y1;
      py2 = py1;

      t1 = sec2_edge / (rho * (1 + dp0));
      x3 = x2 - t1 / 2 * GPU_IPOW2(y2);
      px3 = px2;
      y3 = y2;
      py3 = py2 + t1 * y2 * px2;

      t1 = -tan2_edge / (rho * (1 + dp0));
      t2 = exp(t1 * x3);
      x4 = x3;
      px4 = px3 - t1 * y3 * py3;
      y4 = y3 * t2;
      py4 = py3 / t2;

      x5 = x4 + GPU_IPOW2(gap) * k0 * sec2_edge / (rho * (1 + dp0));
      px5 = px4 + GPU_IPOW2(gap) * k4 * GPU_IPOW2(sin_edge) * Rhe / (2 * rho * cos3_edge);
      y5 = y4;
      py5 = py4;
    }

    *x = x5;
    *y = y5;
    px0 = px5;
    py0 = py5;
  }
  return gpuDipoleConvertMomentaToSlopes(xp, yp, px0, py0, dp, data->expandHamiltonian);
}

/* applySimpleDipoleEdgeKick() (EDGE_EFFECTS=3): the linear Brown kicks, given to the canonical momenta */
__device__ __forceinline__ int gpuCsbendApplySimpleEdgeKick(
  double *xp, double *yp, double x, double y, double dp, double tanEdge, double e, double psi,
  double kickLimit, const GPU_CSBEND_DATA *data) {
  double qx, qy, dqx, dqy;

  if (!gpuDipoleConvertSlopesToMomenta(&qx, &qy, *xp, *yp, dp, data->expandHamiltonian))
    return 0;
  dqx = tanEdge / data->rhoActual * x;
  if (kickLimit > 0 && fabs(dqx) > kickLimit)
    dqx = (dqx < 0 ? -1 : 1) * kickLimit;
  dqy = -(psi ? tan(e - psi / (1 + dp)) : tanEdge) / data->rhoActual * y;
  qx += dqx;
  qy += dqy;
  return gpuDipoleConvertMomentaToSlopes(xp, yp, qx, qy, dp, data->expandHamiltonian);
}

__device__ __forceinline__ int gpuCsbendApplyConfiguredEdge(double *x,
                                                            double *xp,
                                                            double *y,
                                                            double *yp,
                                                            double *path,
                                                            double dp,
                                                            double e,
                                                            double he,
                                                            double psi,
                                                            double kickLimit,
                                                            int whichEdge,
                                                            const GPU_CSBEND_DATA *data) {
  int effect = whichEdge < 0 ?
    data->edgeEffect1 : data->edgeEffect2;
  if (effect == 2 || effect == 4)
    return gpuCsbendApplyHwangFringe(
      x, xp, y, yp, dp, whichEdge < 0 ? -1. : 1., effect == 4,
      whichEdge < 0 ? data->edgeTrig1 : data->edgeTrig2,
      whichEdge < 0 ? data->fint1 : data->fint2, he, data);
  if (effect == 3)
    return gpuCsbendApplySimpleEdgeKick(
      xp, yp, *x, *y, dp, whichEdge < 0 ? data->edgeTrig1[1] : data->edgeTrig2[1], e, psi,
      kickLimit, data);
  if (effect == 5)
    return gpuCsbendApplyCurvedDipoleFringe(
      x, xp, y, yp, path, dp, e, whichEdge,
      whichEdge < 0 ? data->fringeInt1 : data->fringeInt2, data);
  if (data->edgeOrder > 1)
    return gpuCsbendApplyHigherOrderEdge(x, xp, y, yp, dp, e,
                                         whichEdge < 0 ? data->edgeTrig1 : data->edgeTrig2, he, psi,
                                         whichEdge, data);
  return gpuCsbendApplyEdge(xp, yp, *x, *y, dp, e, whichEdge < 0 ? data->edgeTrig1[1] : data->edgeTrig2[1],
                            psi, kickLimit, data);
}

__device__ __forceinline__ void gpuCsbendWriteLoss(
  double *part, double x, double xp, double y, double yp, double dp,
  double lossDistance, const GPU_CSBEND_DATA *data, const double *spin) {

  part[0] = x * data->cosTilt - y * data->sinTilt;
  part[2] = x * data->sinTilt + y * data->cosTilt;
  part[1] = xp * data->cosTilt - yp * data->sinTilt;
  part[3] = xp * data->sinTilt + yp * data->cosTilt;
  part[4] = data->lossZStart + lossDistance;
  part[5] = data->Po * (1 + dp);
  if (data->spinOffset)
    for (int i = 0; i < 3; i++)
      part[data->spinOffset + i] = spin[i];
}

/* The spin turn track_through_csbend() infers from an edge's kick */
__device__ __forceinline__ void gpuCsbendEdgeSpin(double *S, double xp, double xpBefore, double yp,
                                                  double ypBefore, double y, double rho, double dp0,
                                                  const GPU_CSBEND_DATA *data) {
  double P = data->Po * (1 + dp0);
  double gamma = sqrt(P * P + 1);

  gpuSpinRotate(S, -(1 + data->spinAnomaly * gamma) * (yp - ypBefore),
                (1 + data->spinAnomaly * gamma) * (xp - xpBefore),
                -(1 + data->spinAnomaly) * y / rho);
}

__device__ int gpuCsbendTrackParticleData(
  double *part, int stride, int writeOutput, const GPU_CSBEND_DATA *data, long ip = 0) {
  const double *isrDraws = data->isrCoef ? data->isrDraws + ip * data->isrDrawsPerParticle : NULL;
  double driftFrac[8];
  double kickFrac[8];
  double x0 = part[0];
  double xp0 = part[1];
  double y0 = part[2];
  double yp0 = part[3];
  double s0 = part[4];
  double dp = part[5];
  double dp0 = dp;
  double x, xp, y, yp, qx, qy, dist;
  double trackedS;
  double lossDistance = 0;
  double onePlusDp = 1 + dp;
  double dsSlice;
  double spin[3] = {0, 0, 0};
  int nSubsteps = 0;

  (void)stride;
  if (!isfinite(x0) || !isfinite(xp0) || !isfinite(y0) ||
      !isfinite(yp0) || !isfinite(dp))
    return 0;
  if (onePlusDp == 0)
    return 0;
  if (data->hasExactMisalignment) {
    gpuApplyExactMisalignment(&x0, &xp0, &y0, &yp0, &s0, dp,
                              &data->exactMisalignment, 0);
  } else if (data->hasLinearMisalignment) {
    gpuApplyLinearMisalignment(&x0, &xp0, &y0, &yp0, &s0, dp,
                               &data->misalignment, 0);
  } else if (data->hasMisalignment) {
    s0 += data->dzi * sqrt(1 + xp0 * xp0 + yp0 * yp0);
    x0 = x0 + data->dxi + data->dzi * xp0;
    y0 = y0 + data->dyi + data->dzi * yp0;
  }
  if (!gpuLoadSymplecticFractions(data->integrationOrder, driftFrac,
                                  kickFrac, &nSubsteps))
    return 0;

  x = x0 * data->cosTilt + y0 * data->sinTilt;
  y = -x0 * data->sinTilt + y0 * data->cosTilt;
  xp = xp0 * data->cosTilt + yp0 * data->sinTilt;
  yp = -xp0 * data->sinTilt + yp0 * data->cosTilt;
  if (data->spinOffset) {
    for (int i = 0; i < 3; i++)
      spin[i] = part[data->spinOffset + i];
    gpuMultipoleSpinRoll(spin, data->cosTilt, data->sinTilt);
  }

  if (!gpuInsideApertureLimit(x, y, data->aperture) ||
        (data->contour.nContours &&
         !gpuInsideApertureContour(x, y, data->aperture, data->contour))) {
    if (writeOutput)
      gpuCsbendWriteLoss(part, x, xp, y, yp, dp, lossDistance, data, spin);
    return 0;
  }
  if (data->edge1) {
    double xpBefore = xp, ypBefore = yp, rho = (1 + dp) * data->rhoActual;

    if (!gpuCsbendApplyConfiguredEdge(&x, &xp, &y, &yp, &s0, dp, data->e1,
                                      data->he1, data->psi1,
                                      data->edgeKickLimit1, -1, data))
      return 0;
    if (data->spinOffset)
      gpuCsbendEdgeSpin(spin, xp, xpBefore, yp, ypBefore, y, rho, dp0, data);
  }

  if (data->radCoef && data->edge1 && data->e1 != 0) {
    double Fx, Fy, dpPrime;
    double onePlusXh = 1 + x / data->rho0;

    gpuCsbendFields(&Fx, &Fy, x, y, data);
    dpPrime =
      -data->radCoef * (Fx * Fx + Fy * Fy) * (1 + dp) * (1 + dp) *
      sqrt(onePlusXh * onePlusXh + xp * xp + yp * yp);
    dp -= dpPrime * x * data->edgeTrig1[1];
    onePlusDp = 1 + dp;
    if (onePlusDp == 0)
      return 0;
  }

  if (!gpuDipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return 0;

  dist = 0;
  dsSlice = __ddiv_rn(data->length, (double)data->nSlices);
  for (long i = 0; i < data->nSlices; i++) {
    if (!gpuInsideApertureLimit(x, y, data->aperture) ||
        (data->contour.nContours &&
         !gpuInsideApertureContour(x, y, data->aperture, data->contour))) {
      if (writeOutput &&
        gpuDipoleConvertMomentaToSlopes(
            &xp, &yp, qx, qy, dp, data->expandHamiltonian))
        gpuCsbendWriteLoss(part, x, xp, y, yp, dp, lossDistance, data, spin);
      return 0;
    }
    for (int j = 0; j < nSubsteps; j++) {
      double dsh = dsSlice * driftFrac[j];

      if (data->expandHamiltonian) {
        qx += dsh * onePlusDp / (2 * data->rho0);
        dist += dsh * (1 + (qx * qx + qy * qy) / 2);
        x += qx * dsh / onePlusDp;
        y += qy * dsh / onePlusDp;
        qx += dsh * onePlusDp / (2 * data->rho0);
      } else {
        /* integrate_csbend_ordn()'s drift: the momentum turns by the same angle for every particle,
         * whose sine, cosine and versine come from the host (gpuPackCsbendDriftSteps()), and the sums
         * of angles give sin(phi + a) and cos(phi + a) with no trigonometric function here.  Each line
         * is the CPU's, operation for operation. */
        double f = (1 + dp) * (1 + dp) - qy * qy;
        double sinPhi, sine, cosi, cosPhi, factor, inverse;

        if (f <= 0)
          return 0;
        f = sqrt(f);
        if (fabs(qx / f) > 1)
          return 0;
        sinPhi = qx / f;
        cosPhi = sqrt(1 - sinPhi * sinPhi);
        if (cosPhi == 0)
          return 0;
        sine = data->driftSin[j] * cosPhi + data->driftCos[j] * sinPhi;
        cosi = data->driftCos[j] * cosPhi - data->driftSin[j] * sinPhi;
        if (cosi == 0)
          return 0;
        inverse = 1 / (f * cosi);
        qx = f * sine;
        factor = (data->rho0 + x) * data->driftSin[j] * inverse;
        y += qy * factor;
        dist += factor * (1 + dp);
        x = (data->rho0 * (cosPhi * data->driftVers[j] + data->driftSin[j] * sinPhi) + cosPhi * x) * (f * inverse);
      }
      lossDistance += dsh;
      if (!isfinite(x) || !isfinite(y) || !isfinite(qx) ||
          !isfinite(qy) || !isfinite(dist))
        return 0;

      if (kickFrac[j] == 0)
        break;
      double ds = dsSlice * kickFrac[j];
      double Fx, Fy;
      gpuCsbendFields(&Fx, &Fy, x, y, data);
      qx += __ddiv_rn(-ds * (1 + __ddiv_rn(x, data->rho0)) * Fy,
                      data->rhoActual);
      qy += __ddiv_rn(ds * (1 + __ddiv_rn(x, data->rho0)) * Fx,
                      data->rhoActual);
      /* integrate_csbend_ordn_expanded() leaves the spin alone; the edges and the tilt still turn it.
       * integrate_csbend_ordn() turns it with the slopes, QX and QY times (1 + X/rho0)/pz. */
      if (data->spinOffset && !data->expandHamiltonian) {
        double f = __ddiv_rn(1 + __ddiv_rn(x, data->rho0), sqrt((1 + dp) * (1 + dp) - qx * qx - qy * qy));

        gpuSpinUpdateLocalFS(spin, Fx * data->spinFieldConversion, Fy * data->spinFieldConversion, x, f * qx,
                             f * qy, ds, data->spinH, data->Po * (1 + dp), data->spinCharge, data->spinMassC,
                             data->spinAnomaly);
      }
      if (data->radCoef || isrDraws) {
        double f =
          __ddiv_rn(1 + x * data->invRho0,
                    sqrt((1 + dp) * (1 + dp) - qx * qx - qy * qy));
        double radXp = qx * f;
        double radYp = qy * f;
        double dsFactor =
          sqrt((1 + x * data->invRho0) *
                 (1 + x * data->invRho0) +
               radXp * radXp + radYp * radYp);
        double F2 = Fx * Fx + Fy * Fy;
        double deltaFactor = (1 + dp) * (1 + dp);

        if (!isfinite(f))
          return 0;
        qx = __ddiv_rn(qx, 1 + dp);
        qy = __ddiv_rn(qy, 1 + dp);
        if (data->radCoef)
          dp -= data->radCoef * deltaFactor * F2 * ds * dsFactor;
        if (isrDraws) {
          double dsIsr = data->expandHamiltonian ? dsSlice / 3 : dsSlice / (nSubsteps - 1);

          dp -= data->isrCoef * deltaFactor * pow(F2, 0.75) * sqrt(dsIsr * dsFactor) * *isrDraws;
          isrDraws++;
        }
        onePlusDp = 1 + dp;
        if (onePlusDp == 0)
          return 0;
        qx *= onePlusDp;
        qy *= onePlusDp;
      } else if (data->momentumRoundTrip) {
        /* the CPU's radiation block with no radiation to apply: see gpuPackCsbendTracking() */
        qx = qx / (1 + dp);
        qy = qy / (1 + dp);
        qx *= (1 + dp);
        qy *= (1 + dp);
      }
    }
  }

  if (!gpuDipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;
  if (!gpuInsideApertureLimit(x, y, data->aperture) ||
        (data->contour.nContours &&
         !gpuInsideApertureContour(x, y, data->aperture, data->contour))) {
    if (writeOutput)
      gpuCsbendWriteLoss(part, x, xp, y, yp, dp, lossDistance, data, spin);
    return 0;
  }

  if (data->radCoef && data->edge2 && data->e2 != 0) {
    double Fx, Fy, dpPrime;
    double onePlusXh = 1 + x / data->rho0;

    gpuCsbendFields(&Fx, &Fy, x, y, data);
    dpPrime =
      -data->radCoef * (Fx * Fx + Fy * Fy) * (1 + dp) * (1 + dp) *
      sqrt(onePlusXh * onePlusXh + xp * xp + yp * yp);
    dp -= dpPrime * x * data->edgeTrig2[1];
  }

  if (data->radCoef || data->isrCoef) {
    double p0 = data->Po * (1 + dp0);
    double p1 = data->Po * (1 + dp);
    double beta0 = p0 / sqrt(p0 * p0 + 1);
    double beta1 = p1 / sqrt(p1 * p1 + 1);

    trackedS = beta1 * s0 / beta0 + dist;
  } else {
    trackedS = s0 + dist;
  }

  if (data->edge2) {
    double xpBefore = xp, ypBefore = yp, rho = (1 + dp) * data->rhoActual;

    if (!gpuCsbendApplyConfiguredEdge(&x, &xp, &y, &yp, &trackedS, dp, data->e2,
                                      data->he2, data->psi2,
                                      data->edgeKickLimit2, 1, data))
      return 0;
    if (data->spinOffset)
      gpuCsbendEdgeSpin(spin, xp, xpBefore, yp, ypBefore, y, rho, dp0, data);
  }

  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;

  if (writeOutput) {
    double xOut = x * data->cosTilt - y * data->sinTilt +
                  data->dcoordEtilt[0];
    double yOut = x * data->sinTilt + y * data->cosTilt +
                  data->dcoordEtilt[2];
    double xpOut = xp * data->cosTilt - yp * data->sinTilt +
                   data->dcoordEtilt[1];
    double ypOut = xp * data->sinTilt + yp * data->cosTilt +
                   data->dcoordEtilt[3];
    double sOut = trackedS + data->dcoordEtilt[4];

    if (data->hasMisalignment) {
      xOut += data->dxf + data->dzf * xpOut;
      yOut += data->dyf + data->dzf * ypOut;
      sOut += data->dzf * sqrt(1 + xpOut * xpOut + ypOut * ypOut);
    } else if (data->hasExactMisalignment) {
      gpuApplyExactMisalignment(&xOut, &xpOut, &yOut, &ypOut, &sOut, dp,
                                &data->exactMisalignment, 1);
    } else if (data->hasLinearMisalignment) {
      gpuApplyLinearMisalignment(&xOut, &xpOut, &yOut, &ypOut, &sOut, dp,
                                 &data->misalignment, 1);
    }
    part[0] = xOut;
    part[2] = yOut;
    part[1] = xpOut;
    part[3] = ypOut;
    part[4] = sOut;
    part[5] = dp;
    if (data->spinOffset) {
      gpuMultipoleSpinRoll(spin, data->cosTilt, -data->sinTilt);
      for (int i = 0; i < 3; i++)
        part[data->spinOffset + i] = spin[i];
    }
  }
  return 1;
}

__device__ int gpuCsbendTrackParticle(double *part, int stride,
                                      int writeOutput, long ip = 0) {
  return gpuCsbendTrackParticleData(
    part, stride, writeOutput, &gpuCsbendData, ip);
}

__device__ __forceinline__ int gpuCsrCsbendInsideApertures(double x, double y,
                                                           const GPU_CSBEND_DATA *data) {
  return gpuInsideApertureLimit(x, y, data->aperture) &&
         (!data->contour.nContours ||
          gpuInsideApertureContour(x, y, data->aperture, data->contour));
}

/* The row track_through_csbendCSR() writes for a particle integrate_csbend_ordn() loses: the slopes
 * from convertFromDipoleCanonicalCoordinates(), turned back by convertFromCSBendCoords(), the loss
 * position and the momentum */
__device__ int gpuCsrCsbendLoseBodySliceParticle(double *part, double x, double qx, double y,
                                                 double qy, double dp, double dzLost,
                                                 const GPU_CSBEND_DATA *data) {
  double f = 1 / sqrt((1 + dp) * (1 + dp) - qx * qx - qy * qy);
  double xp = qx * f, yp = qy * f;

  part[0] = x * data->lossCosTtilt - y * data->lossSinTtilt;
  part[2] = x * data->lossSinTtilt + y * data->lossCosTtilt;
  part[1] = xp * data->lossCosTtilt - yp * data->lossSinTtilt;
  part[3] = xp * data->lossSinTtilt + yp * data->lossCosTtilt;
  part[4] = data->lossZStart + dzLost;
  part[5] = data->Po * (1 + dp);
  return 0;
}

/* One body slice of track_through_csbendCSR().  Returns 1 if the particle is kept, 0 if
 * integrate_csbend_ordn() loses it, at an aperture check or in a drift, leaving the CPU's row for a
 * lost particle, or -1 for what only the CPU follows, such as a coordinate that isn't finite */
__device__ int gpuCsrCsbendTrackBodySliceParticle(double *part, int stride,
                                                  double beta0) {
  const GPU_CSBEND_DATA *data = &gpuCsbendData;
  double driftFrac[8];
  double kickFrac[8];
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double s0 = part[4];
  double dp = part[5];
  double qx, qy, dist;
  double onePlusDp = 1 + dp;
  double dsSlice, dzLost = 0;
  int nSubsteps = 0;
  int apertures = data->aperture.present || data->contour.nContours;

  (void)stride;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(dp) || !isfinite(beta0) || beta0 == 0)
    return -1;
  if (onePlusDp == 0)
    return -1;
  if (!gpuLoadSymplecticFractions(data->integrationOrder, driftFrac,
                                  kickFrac, &nSubsteps))
    return -1;
  if (!gpuDipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return -1;

  dist = 0;
  dsSlice = data->length / data->nSlices;
  for (long i = 0; i < data->nSlices; i++) {
    if (apertures && !gpuCsrCsbendInsideApertures(x, y, data))
      return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);
    for (int j = 0; j < nSubsteps; j++) {
      double dsh = dsSlice * driftFrac[j];

      if (data->expandHamiltonian) {
        qx += dsh * onePlusDp / (2 * data->rho0);
        dist += dsh * (1 + (qx * qx + qy * qy) / 2);
        x += qx * dsh / onePlusDp;
        y += qy * dsh / onePlusDp;
        qx += dsh * onePlusDp / (2 * data->rho0);
      } else {
        /* the same drift as in gpuCsbendTrackParticleData(), which is integrate_csbend_ordn()'s */
        double f = onePlusDp * onePlusDp - qy * qy;
        double sinPhi, sine, cosi, cosPhi, factor, inverse;

        if (f <= 0)
          return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);
        f = sqrt(f);
        if (fabs(qx / f) > 1)
          return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);
        sinPhi = qx / f;
        cosPhi = sqrt(1 - sinPhi * sinPhi);
        if (cosPhi == 0)
          return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);
        sine = data->driftSin[j] * cosPhi + data->driftCos[j] * sinPhi;
        cosi = data->driftCos[j] * cosPhi - data->driftSin[j] * sinPhi;
        if (cosi == 0)
          return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);
        inverse = 1 / (f * cosi);
        qx = f * sine;
        factor = (data->rho0 + x) * data->driftSin[j] * inverse;
        y += qy * factor;
        dist += factor * onePlusDp;
        x = (data->rho0 * (cosPhi * data->driftVers[j] + data->driftSin[j] * sinPhi) + cosPhi * x) * (f * inverse);
      }
      dzLost += dsh;
      if (!isfinite(x) || !isfinite(y) || !isfinite(qx) ||
          !isfinite(qy) || !isfinite(dist))
        return -1;

      if (kickFrac[j] == 0)
        break;
      double ds = dsSlice * kickFrac[j];
      double Fx, Fy;
      gpuCsbendFields(&Fx, &Fy, x, y);
      qx += -ds * (1 + x / data->rho0) * Fy / data->rhoActual;
      qy += ds * (1 + x / data->rho0) * Fx / data->rhoActual;
    }
  }
  dzLost = data->nSlices * dsSlice;
  if (apertures && !gpuCsrCsbendInsideApertures(x, y, data))
    return gpuCsrCsbendLoseBodySliceParticle(part, x, qx, y, qy, dp, dzLost, data);

  if (!gpuDipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return -1;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return -1;

  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = s0 + dist / beta0;
  part[5] = dp;
  return 1;
}

__global__ void gpuCsbendTrackCheckedKernel(double *coord, long nParticles,
                                            int stride,
                                            unsigned long long *lostCount, long *survivorFlag) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuCsbendTrackParticle(coord + ip * stride, stride, 1, ip))
      localCount = 1;
    if (survivorFlag)
      survivorFlag[ip] = localCount ? 0 : 1;
  }
  partial[threadIdx.x] = localCount;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

__global__ void gpuCsbendSurvivorFlagKernel(double *coord, long nParticles,
                                            int stride,
                                            long *survivorPrefix) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuCsbendTrackParticle(coord + ip * stride, stride, 0) ? 1 : 0;
}

__global__ void gpuCsbendStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long destination;

  if (ip >= nParticles)
    return;
  destination = survivorPrefix[ip];
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  gpuCsbendTrackParticle(target, stride, 1);
}

__global__ void gpuTuneProgramInitializeKernel(
  double *coordinate, long particles, int stride,
  GPU_TUNE_PARTICLE_STATUS *status, double *history, long turns) {
  long id = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= particles)
    return;
  status[id].alive = 1;
  status[id].lossTurn = -1;
  status[id].lossElement = -1;
  if (history) {
    double *part = coordinate + id * stride;
    long block = particles * turns;
    history[id * turns] = part[0];
    history[block + id * turns] = part[1];
    history[2 * block + id * turns] = part[2];
    history[3 * block + id * turns] = part[3];
  }
}

__device__ __forceinline__ int gpuTuneProgramCollimatorTrack(
  double *part, const GPU_TUNE_COLLIMATOR_DATA &data, int elliptical) {
  int lost;

  if (elliptical) {
    lost = gpuEllipticalCollimatorLostAt(
      part, data.xMax, data.yMax, data.xCenter, data.yCenter,
      data.xExponent, data.yExponent, 0, data.openCode);
    if (!lost && data.length > 0)
      lost = gpuEllipticalCollimatorLostAt(
        part, data.xMax, data.yMax, data.xCenter, data.yCenter,
        data.xExponent, data.yExponent, data.length, data.openCode);
  } else {
    lost = gpuRectangularCollimatorLostAt(
      part, data.xMax, data.yMax, data.xCenter, data.yCenter, 0,
      data.openCode);
    if (!lost && data.length > 0)
      lost = gpuRectangularCollimatorLostAt(
        part, data.xMax, data.yMax, data.xCenter, data.yCenter,
        data.length, data.openCode);
  }
  if (!lost && data.length)
    gpuExactDriftParticle(part, data.length);
  return !lost;
}

__device__ __forceinline__ int gpuTuneProgramScraperTrack(
  double *part, const GPU_TUNE_SCRAPER_DATA &data) {
  int lost = 0;

  if (data.sideSign)
    lost = gpuScraperLostAt(
      part, data.plane, data.center, data.position, data.sideSign, 0);
  if (!lost && data.secondSideSign)
    lost = gpuScraperLostAt(
      part, data.plane, data.center, data.position,
      data.secondSideSign, 0);
  if (!lost && data.length > 0 && data.sideSign)
    lost = gpuScraperLostAt(
      part, data.plane, data.center, data.position, data.sideSign,
      data.length);
  if (!lost && data.length > 0 && data.secondSideSign)
    lost = gpuScraperLostAt(
      part, data.plane, data.center, data.position,
      data.secondSideSign, data.length);
  if (!lost && data.length)
    gpuExactDriftParticle(part, data.length);
  return !lost;
}

/* offset_beam() for one particle, in its order of operations (gpuOffsetBeamKernel is the same) */
__device__ __forceinline__ void gpuTuneProgramMalign(
  double *part, const GPU_TUNE_MALIGN_DATA &data) {
  double pc, beta, gamma, t, ds;

  ds = data.dz ? data.dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]) : 0;
  part[0] += data.dx + data.dz * part[1];
  part[1] += data.dxp;
  part[2] += data.dy + data.dz * part[3];
  part[3] += data.dyp;
  part[4] += ds;
  if (data.dt || data.dp || data.de) {
    pc = data.pCentral * (1 + part[5]);
    gamma = sqrt(1 + pc * pc);
    beta = pc / gamma;
    t = part[4] / (beta * data.cMks) + data.dt;
    if (data.dp) {
      part[5] += data.dp;
      pc = data.pCentral * (1 + part[5]);
      beta = pc / sqrt(1 + pc * pc);
    }
    if (data.de) {
      gamma += data.de * gamma;
      pc = sqrt(gamma * gamma - 1);
      beta = pc / gamma;
      part[5] = (pc - data.pCentral) / data.pCentral;
    }
    part[4] = t * beta * data.cMks;
  }
}

__global__ void gpuTuneProgramTurnKernel(
  double *coordinate, long particles, int stride, GPU_TUNE_PROGRAM program,
  GPU_TUNE_PARTICLE_STATUS *status, double *history, long turns, long turn,
  long turnOffset) {
  long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int alive;

  if (id >= particles || !status[id].alive)
    return;
  part = coordinate + id * stride;
  alive = 1;
  for (long ie = 0; ie < program.opCount && alive; ie++) {
    const GPU_TUNE_PROGRAM_OP &op = program.op[ie];

    switch (op.opcode) {
    case GPU_TUNE_OP_NOP:
      break;
    case GPU_TUNE_OP_EXACT_DRIFT:
      gpuExactDriftParticle(part, program.drift[op.dataIndex].length);
      break;
    case GPU_TUNE_OP_MATRIX:
      gpuApplyPackedMatrixData(part, program.matrix + op.dataIndex);
      break;
    case GPU_TUNE_OP_MULTIPOLE:
      alive = gpuMultipoleTrackParticleData<false, true>(
        part, stride, 1, program.multipole + op.dataIndex);
      break;
    case GPU_TUNE_OP_EXACT_CORRECTOR:
      alive = gpuExactCorrectorTrackParticle(
        part, 1, program.exactCorrector[op.dataIndex]);
      break;
    case GPU_TUNE_OP_CSBEND:
      alive = gpuCsbendTrackParticleData(
        part, stride, 1, program.csbend + op.dataIndex);
      break;
    case GPU_TUNE_OP_CCBEND:
      alive = gpuCcbendTrackParticle(
        part, stride, program.ccbend[op.dataIndex], 1, 0);
      break;
    case GPU_TUNE_OP_LGBEND: {
      const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture =
        op.auxiliaryIndex >= 0 ?
          program.lgbendLocalAperture + op.auxiliaryIndex : NULL;
      alive = gpuLgbendTrackParticle(
        part, stride, program.lgbend[op.dataIndex], localAperture, 1);
      break;
    }
    case GPU_TUNE_OP_RCOL:
      alive = gpuTuneProgramCollimatorTrack(
        part, program.collimator[op.dataIndex], 0);
      break;
    case GPU_TUNE_OP_ECOL:
      alive = gpuTuneProgramCollimatorTrack(
        part, program.collimator[op.dataIndex], 1);
      break;
    case GPU_TUNE_OP_SCRAPER:
      alive = gpuTuneProgramScraperTrack(
        part, program.scraper[op.dataIndex]);
      break;
    case GPU_TUNE_OP_TAPER_APERTURE:
      alive = gpuTaperApertureTrackParticle(
        part, 1, program.taperAperture[op.dataIndex]);
      break;
    case GPU_TUNE_OP_SPEEDBUMP:
      alive = gpuSpeedbumpTrackParticle(
        part, 1, program.speedbump[op.dataIndex]);
      break;
    case GPU_TUNE_OP_MALIGN:
      if (program.malign[op.dataIndex].onPass == -1 ||
          program.malign[op.dataIndex].onPass == turn - 1 + turnOffset)
        gpuTuneProgramMalign(part, program.malign[op.dataIndex]);
      break;
    default:
      alive = 0;
      break;
    }
    if (alive && op.postApertureIndex >= 0)
      alive = gpuInsideApertureLimit(
        part[0], part[2], program.postAperture[op.postApertureIndex]);
    if (!alive) {
      status[id].alive = 0;
      status[id].lossTurn = turn - 1 + turnOffset;
      status[id].lossElement = op.elementIndex;
    }
  }
  if (alive && history) {
    long block = particles * turns;
    history[id * turns + turn] = part[0];
    history[block + id * turns + turn] = part[1];
    history[2 * block + id * turns + turn] = part[2];
    history[3 * block + id * turns + turn] = part[3];
  }
}

/* Each particle's outcome from gpuCsrCsbendTrackBodySliceParticle(), with counts[0] the particles
 * lost and counts[1] those only the CPU can follow */
__global__ void gpuCsrCsbendBodySliceCheckedKernel(double *coord,
                                                   long nParticles,
                                                   int stride,
                                                   const double *beta0,
                                                   signed char *outcome,
                                                   unsigned long long *counts) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long lost = 0, cpuOnly = 0;

  if (ip < nParticles) {
    int code = gpuCsrCsbendTrackBodySliceParticle(coord + ip * stride, stride,
                                                  beta0[ip]);
    outcome[ip] = (signed char)code;
    lost = code == 0;
    cpuOnly = code < 0;
  }
  partial[threadIdx.x] = lost;
  partial[blockDim.x + threadIdx.x] = cpuOnly;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
      partial[blockDim.x + threadIdx.x] += partial[blockDim.x + threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (partial[0])
      atomicAdd(counts, partial[0]);
    if (partial[blockDim.x])
      atomicAdd(counts + 1, partial[blockDim.x]);
  }
}

/* A body slice's outcomes as the rounds of gpuBuildCpuRoundLossOrderDestination(): one round */
__global__ void gpuCsrCsbendBodyLossRoundKernel(const signed char *outcome, int *lossRound,
                                                long nParticles) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    lossRound[ip] = outcome[ip] == 0 ? 0 : -1;
}

__global__ void gpuCsrCsbendBodyScatterKernel(const double *coord, double *out,
                                              const double *beta0, double *beta0Out,
                                              const long *destination, long nParticles, int stride) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles) {
    long to = destination[ip];

    for (int ic = 0; ic < stride; ic++)
      out[to * stride + ic] = coord[ip * stride + ic];
    beta0Out[to] = beta0[ip];
  }
}

__global__ void gpuCsrCsbendEnterSimpleCheckedKernel(
  double *coord, long nParticles, int stride, double pCentral,
  double coordinateSign, int edge1Effect, double e1, double psi1, double tanE1,
  double rhoActual, double *beta0, unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    double *part = coord + ip * stride;
    double p0 = pCentral * (1 + part[5]);
    double beta = 0;

    if (!isfinite(p0)) {
      localCount = 1;
    } else {
      beta = p0 / sqrt(p0 * p0 + 1);
      if (!isfinite(beta) || beta == 0) {
        localCount = 1;
      } else {
        if (coordinateSign == -1) {
          part[0] = -part[0];
          part[1] = -part[1];
          part[2] = -part[2];
          part[3] = -part[3];
        }
        part[4] /= beta;
        beta0[ip] = beta;
        if (edge1Effect) {
          double rho = (1 + part[5]) * rhoActual;
          if (!isfinite(rho) || rho == 0) {
            localCount = 1;
          } else {
            /* tanE1 is the host's tan(e1): the device's tan() isn't glibc's to the last bit (at
             * e1 = 0.08 they differ, and 8% of the particles came out one unit off in yp).  With no
             * fringe term the CPU's tan(e1 - psi1 / (1 + dp)) is tan(e1) too. */
            double deltaXp = tanE1 / rho * part[0];
            part[1] += deltaXp;
            part[3] -= (psi1 ? tan(e1 - psi1 / (1 + part[5])) : tanE1) / rho * part[2];
          }
        }
      }
    }
  }
  partial[threadIdx.x] = localCount;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

__global__ void gpuCsrCsbendFinalizeSimpleCheckedKernel(
  double *coord, long nParticles, int stride, double pCentral,
  double coordinateSign, int edge2Effect, double e2, double psi2, double tanE2,
  double rhoActual, unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    double *part = coord + ip * stride;
    double p1 = pCentral * (1 + part[5]);
    if (!isfinite(p1) || p1 <= 0) {
      localCount = 1;
    } else {
      double beta1 = p1 / sqrt(p1 * p1 + 1);
      part[4] = part[4] * beta1;
      if (edge2Effect) {
        double rho = (1 + part[5]) * rhoActual;
        if (!isfinite(rho) || rho == 0) {
          localCount = 1;
        } else {
          double deltaXp = tanE2 / rho * part[0];
          part[1] += deltaXp;
          part[3] -= (psi2 ? tan(e2 - psi2 / (1 + part[5])) : tanE2) / rho * part[2];
        }
      }
      if (coordinateSign == -1) {
        part[0] = -part[0];
        part[1] = -part[1];
        part[2] = -part[2];
        part[3] = -part[3];
      }
	    }
	  }
  partial[threadIdx.x] = localCount;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0])
    atomicAdd(lostCount, partial[0]);
}

extern "C" int gpuCudaCsbendTrackChecked(void *coord, long nParticles,
                                         int stride,
                                         const GPU_CSBEND_DATA *csbend,
                                         void *survivorFlags,
                                         long *lostCount,
                                         float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !csbend || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  status = uploadCsbendDataIfNeeded(csbend);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = gpuReuseDeviceBuffer(1, &backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuCsbendTrackCheckedKernel<<<blocks, threads,
                                threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, deviceLostCount,
    static_cast<long *>(survivorFlags));
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  /* With survivor flags the caller compacts the rows as tracked (gpuCudaCompactTrackedRows) and
   * takes what it needs of the rows from before the element out of the backup */
  if (cudaStatus == cudaSuccess && hostLostCount && !survivorFlags) {
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCsbendTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_CSBEND_DATA *csbend, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = gpuParticleThreads(nParticles);
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !csbend || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  status = uploadCsbendDataIfNeeded(csbend);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuCsbendSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuCsbendStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

static signed char *gpuCsrBodyOutcome = NULL;
static double *gpuCsrBodyBeta0 = NULL;
static unsigned long long *gpuCsrBodyCounts = NULL;
static long gpuCsrBodyCapacity = 0;

static int gpuEnsureCsrBodyLossScratch(long nParticles) {
  cudaError_t status = cudaSuccess;

  if (gpuCsrBodyCounts && gpuCsrBodyCapacity >= nParticles)
    return static_cast<int>(cudaSuccess);
  gpuFreeDevice(gpuCsrBodyOutcome);
  gpuFreeDevice(gpuCsrBodyBeta0);
  gpuFreeDevice(gpuCsrBodyCounts);
  gpuCsrBodyOutcome = NULL;
  gpuCsrBodyBeta0 = NULL;
  gpuCsrBodyCounts = NULL;
  gpuCsrBodyCapacity = 0;
  if ((status = cudaMalloc(&gpuCsrBodyOutcome, nParticles * sizeof(*gpuCsrBodyOutcome))) != cudaSuccess ||
      (status = cudaMalloc(&gpuCsrBodyBeta0, nParticles * sizeof(*gpuCsrBodyBeta0))) != cudaSuccess ||
      (status = cudaMalloc(&gpuCsrBodyCounts, 2 * sizeof(*gpuCsrBodyCounts))) != cudaSuccess)
    return static_cast<int>(status);
  gpuCsrBodyCapacity = nParticles;
  return static_cast<int>(cudaSuccess);
}

/* One body slice on the device.  With keepLosses, particles lost as the CPU loses them are moved
 * into the CPU's order, with beta0, each particle's place left in destination, and *remaining is the
 * number kept.  Otherwise, or if any particle needs the CPU, the coordinates are restored and
 * *lostCount is nonzero */
extern "C" int gpuCudaCsrCsbendBodySliceChecked(void *coord,
                                                long nParticles,
                                                int stride,
                                                const GPU_CSBEND_DATA *csbend,
                                                double *beta0,
                                                void *backup,
                                                int keepLosses,
                                                long *destination,
                                                long *remaining,
                                                long *lostCount,
                                                float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long hostCounts[2] = {0, 0};
  double *coordinates = static_cast<double *>(coord);
  double *backupCoord = static_cast<double *>(backup);
  unsigned long long count;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  long nKept;
  int *lossRound = NULL;
  int status;

  if (!coord || !csbend || !beta0 || !backupCoord || !remaining || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  *remaining = nParticles;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  status = uploadCsbendDataIfNeeded(csbend);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = gpuEnsureCsrBodyLossScratch(nParticles);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuCsrBodyCounts, 0, 2 * sizeof(*gpuCsrBodyCounts));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backupCoord, coordinates, count * sizeof(*backupCoord),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(cudaStatus);
  }
  gpuCsrCsbendBodySliceCheckedKernel<<<blocks, threads,
                                       2 * threads * sizeof(unsigned long long)>>>(
    coordinates, nParticles, stride, beta0, gpuCsrBodyOutcome, gpuCsrBodyCounts);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(hostCounts, gpuCsrBodyCounts, sizeof(hostCounts),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  if (!hostCounts[0] && !hostCounts[1])
    return static_cast<int>(cudaSuccess);
  nKept = nParticles - static_cast<long>(hostCounts[0]);
  if (!hostCounts[1] && keepLosses && nKept > 1) {
    /* track_through_csbendCSR()'s order, beta0 moving with the particles */
    if ((cudaStatus = gpuRoundLossRounds(nParticles, &lossRound)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuCsrCsbendBodyLossRoundKernel<<<blocks, threads>>>(gpuCsrBodyOutcome, lossRound, nParticles);
    if (!destination)
      return static_cast<int>(cudaErrorInvalidValue);
    if ((cudaStatus = gpuBuildCpuRoundLossOrderDestination(destination, nParticles,
                                                           nKept)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuCsrCsbendBodyScatterKernel<<<blocks, threads>>>(coordinates, backupCoord, beta0,
                                                       gpuCsrBodyBeta0, destination,
                                                       nParticles, stride);
    if ((cudaStatus = cudaGetLastError()) != cudaSuccess ||
        (cudaStatus = cudaMemcpy(coordinates, backupCoord, count * sizeof(*backupCoord),
                                 cudaMemcpyDeviceToDevice)) != cudaSuccess ||
        (cudaStatus = cudaMemcpy(beta0, gpuCsrBodyBeta0, nParticles * sizeof(*beta0),
                                 cudaMemcpyDeviceToDevice)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    *remaining = nKept;
    return static_cast<int>(cudaSuccess);
  }
  cudaStatus = cudaMemcpy(coordinates, backupCoord, count * sizeof(*backupCoord),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostCounts[0] + hostCounts[1]);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCsrCsbendEnterSimpleChecked(void *coord,
                                                  long nParticles,
                                                  int stride,
                                                  double pCentral,
                                                  double coordinateSign,
                                                  int edge1Effect,
                                                  double e1,
                                                  double psi1,
                                                  double rhoActual,
                                                  double *beta0,
                                                  void *backup,
                                                  void *deviceLostCount,
                                                  long *lostCount,
                                                  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *lostCountDevice = static_cast<unsigned long long *>(deviceLostCount);
  unsigned long long hostLostCount = 0;
  double *backupCoord = static_cast<double *>(backup);
  unsigned long long count;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !beta0 || !backupCoord || !lostCountDevice || !lostCount ||
      stride < 6 || pCentral == 0 ||
      (coordinateSign != 1 && coordinateSign != -1) ||
      (edge1Effect && rhoActual == 0))
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = cudaMemset(lostCountDevice, 0, sizeof(*lostCountDevice));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backupCoord, coord, count * sizeof(*backupCoord),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(cudaStatus);
  }
  gpuCsrCsbendEnterSimpleCheckedKernel<<<blocks, threads,
                                         threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral,
    coordinateSign, edge1Effect, e1, psi1, tan(e1), rhoActual, beta0, lostCountDevice);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(&hostLostCount, lostCountDevice, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backupCoord, count * sizeof(*backupCoord),
                            cudaMemcpyDeviceToDevice);
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCsrCsbendFinalizeSimpleChecked(void *coord,
                                                     long nParticles,
                                                     int stride,
                                                     double pCentral,
                                                     double coordinateSign,
                                                     int edge2Effect,
                                                     double e2,
                                                     double psi2,
                                                     double rhoActual,
                                                     void *backup,
                                                     void *deviceLostCount,
                                                     long *lostCount,
                                                     float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *lostCountDevice = static_cast<unsigned long long *>(deviceLostCount);
  unsigned long long hostLostCount = 0;
  double *backupCoord = static_cast<double *>(backup);
  unsigned long long count;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !backupCoord || !lostCountDevice || !lostCount ||
      stride < 6 || pCentral == 0 ||
      (coordinateSign != 1 && coordinateSign != -1) ||
      (edge2Effect && rhoActual == 0))
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = cudaMemset(lostCountDevice, 0, sizeof(*lostCountDevice));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(backupCoord, coord, count * sizeof(*backupCoord),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(cudaStatus);
  }
  gpuCsrCsbendFinalizeSimpleCheckedKernel<<<blocks, threads,
                                            threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral,
    coordinateSign, edge2Effect, e2, psi2, tan(e2), rhoActual, lostCountDevice);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemcpy(&hostLostCount, lostCountDevice, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backupCoord, count * sizeof(*backupCoord),
                            cudaMemcpyDeviceToDevice);
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

/* The histograms in the CPU's order: keys, a stable sort by bin, then one thread per bin */
static cudaError_t gpuCombinedWakeLaunchSortedHistogram(
  const double *coord, long nParticles, int stride,
  const GPU_COMBINED_WAKE_DATA *wake, int threads) {
  size_t bytes = gpuTrfmodeSortScratch.tempBytes;
  int particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  int endBit = 0;
  cudaError_t cudaStatus;

  while (endBit < 32 && (1UL << endBit) <= static_cast<unsigned long>(wake->bins))
    endBit++;
  gpuTrfmodeSortKeysKernel<<<particleBlocks, threads>>>(
    gpuCombinedWakeScratch.pbin, nParticles, wake->bins, gpuTrfmodeSortScratch.keysIn,
    gpuTrfmodeSortScratch.indexIn);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cub::DeviceRadixSort::SortPairs(
      gpuTrfmodeSortScratch.temp, bytes, gpuTrfmodeSortScratch.keysIn,
      gpuTrfmodeSortScratch.keysOut, gpuTrfmodeSortScratch.indexIn,
      gpuTrfmodeSortScratch.indexOut, static_cast<int>(nParticles), 0, endBit);
  if (cudaStatus == cudaSuccess) {
    gpuCombinedWakeSortedHistogramKernel<<<static_cast<int>(wake->bins),
                                           GPU_COMBINED_WAKE_SUM_THREADS>>>(
        coord, nParticles, stride, *wake, gpuTrfmodeSortScratch.keysOut,
        gpuTrfmodeSortScratch.indexOut, gpuCombinedWakeScratch.histogram);
    cudaStatus = cudaGetLastError();
  }
  return cudaStatus;
}

/* With wake->cpuOrder the histograms are added in particle order and nothing goes through cuFFT: a
 * time-domain wake takes the direct convolution, whose sums run in convolveArrays()'s order, and an
 * impedance takes its voltages from hostVoltage, which runs the CPU's own FFTs on the histograms. */
extern "C" int gpuCudaCombinedWakeTrack(
  void *coord, long nParticles, int stride,
  const GPU_COMBINED_WAKE_DATA *wake,
  const double *const *tables, long *binnedCount,
  double *histogramReturn, double *voltageReturn,
  GPU_COMBINED_WAKE_HOST_VOLTAGE hostVoltage, void *hostVoltageContext,
  float *milliseconds) {
  static double *hostHistogram = NULL, *hostVoltageValues = NULL;
  static long hostBins = 0;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  cufftResult fftStatus;
  unsigned long long hostBinnedCount = 0;
  long frequencyPoints, planBins, requiredConvolutionBins;
  int useTimeFft = 0;
  int threads = 256;
  int particleBlocks, binBlocks, histogramPartialBlocks, status;
  dim3 channelGrid;

  if (!coord || !wake || !tables || !binnedCount || stride < 6 ||
      wake->bins < 2 || wake->tablePoints <= 0 || wake->dt <= 0 ||
      (wake->mode != GPU_COMBINED_WAKE_MODE_TIME &&
       wake->mode != GPU_COMBINED_WAKE_MODE_IMPEDANCE))
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  for (long channel = 0; channel < GPU_COMBINED_WAKE_CHANNELS; channel++) {
    if (wake->enabled[channel] && (!tables[channel] ||
        wake->driver[channel] < 0 || wake->driver[channel] > 2))
      return static_cast<int>(cudaErrorInvalidValue);
  }

  /* cpuOrder chooses the ordered histogram; the voltages come from hostVoltage when one is given and
   * from cuFFT otherwise.  IMPEDANCE gives one whenever it asks for CPU order; ZLONGIT and ZTRANSVERSE
   * always ask for the ordered histogram and give one only when output, smoothing or WAKES need the
   * host's FFT. */
  if (wake->cpuOrder) {
    status = ensureTrfmodeSortScratch(nParticles, wake->bins);
    if (status != static_cast<int>(cudaSuccess))
      return status;
  }
  if (hostVoltage) {
    if (wake->mode == GPU_COMBINED_WAKE_MODE_IMPEDANCE && hostBins < wake->bins) {
      std::free(hostHistogram);
      std::free(hostVoltageValues);
      hostHistogram = static_cast<double *>(std::malloc(3 * wake->bins * sizeof(*hostHistogram)));
      hostVoltageValues = static_cast<double *>(
        std::malloc(GPU_COMBINED_WAKE_CHANNELS * wake->bins * sizeof(*hostVoltageValues)));
      hostBins = hostHistogram && hostVoltageValues ? wake->bins : 0;
      if (!hostBins) {
        std::free(hostHistogram);
        std::free(hostVoltageValues);
        hostHistogram = hostVoltageValues = NULL;
        return static_cast<int>(cudaErrorMemoryAllocation);
      }
    }
  }

  planBins = wake->bins;
  useTimeFft = wake->mode == GPU_COMBINED_WAKE_MODE_TIME &&
               wake->allowTimeFft && !wake->cpuOrder &&
               wake->bins > 0 &&
               wake->tablePoints > 262144 / wake->bins;
  if (useTimeFft) {
    requiredConvolutionBins = wake->bins + wake->tablePoints - 1;
    planBins = 1;
    while (planBins < requiredConvolutionBins) {
      if (planBins > LONG_MAX / 2)
        return static_cast<int>(cudaErrorInvalidValue);
      planBins <<= 1;
    }
  }

  status = ensureCombinedWakeScratch(nParticles, wake->bins,
                                     wake->tablePoints, planBins);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  binBlocks = static_cast<int>((wake->bins + threads - 1) / threads);
  histogramPartialBlocks = static_cast<int>(
    (gpuCombinedWakeScratch.histogramPartials + threads - 1) / threads);
  frequencyPoints = planBins / 2 + 1;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuCombinedWakeScratch.histogram, 0,
                          3 * wake->bins *
                          sizeof(*gpuCombinedWakeScratch.histogram));
  /* the partial histograms are used only without CPU order; with many bins they are tens of MB */
  if (cudaStatus == cudaSuccess && !wake->cpuOrder)
    cudaStatus = cudaMemset(
      gpuCombinedWakeScratch.histogramPartial, 0,
      3 * wake->bins * gpuCombinedWakeScratch.histogramPartials *
      sizeof(*gpuCombinedWakeScratch.histogramPartial));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuCombinedWakeScratch.voltage, 0,
                            GPU_COMBINED_WAKE_CHANNELS * wake->bins *
                            sizeof(*gpuCombinedWakeScratch.voltage));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuCombinedWakeScratch.binnedCount, 0,
                            sizeof(*gpuCombinedWakeScratch.binnedCount));
  if (cudaStatus == cudaSuccess && useTimeFft)
    cudaStatus = cudaMemset(gpuCombinedWakeScratch.driverFirstBin, 0x7f,
                            3 * sizeof(*gpuCombinedWakeScratch.driverFirstBin));

  for (long channel = 0;
       cudaStatus == cudaSuccess && channel < GPU_COMBINED_WAKE_CHANNELS;
       channel++) {
    if (!wake->enabled[channel])
      continue;
    if (gpuCombinedWakeScratch.hostTable[channel] == tables[channel] &&
        gpuCombinedWakeScratch.hostTablePoints[channel] == wake->tablePoints)
      continue;
    cudaStatus = cudaMemcpy(
      gpuCombinedWakeScratch.table +
        channel * gpuCombinedWakeScratch.tableCapacity,
      tables[channel], wake->tablePoints * sizeof(**tables),
      cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess && useTimeFft)
      cudaStatus = cudaMemset(gpuCombinedWakeScratch.fftReal, 0,
                              planBins * sizeof(*gpuCombinedWakeScratch.fftReal));
    if (cudaStatus == cudaSuccess && useTimeFft)
      cudaStatus = cudaMemcpy(
        gpuCombinedWakeScratch.fftReal,
        gpuCombinedWakeScratch.table +
          channel * gpuCombinedWakeScratch.tableCapacity,
        wake->tablePoints * sizeof(*gpuCombinedWakeScratch.fftReal),
        cudaMemcpyDeviceToDevice);
    if (cudaStatus == cudaSuccess && useTimeFft) {
      fftStatus = cufftExecD2Z(
        gpuCombinedWakeScratch.forwardPlan,
        gpuCombinedWakeScratch.fftReal,
        gpuCombinedWakeScratch.tableFrequency + channel * frequencyPoints);
      if (fftStatus != CUFFT_SUCCESS)
        cudaStatus = cudaErrorUnknown;
    }
    if (cudaStatus == cudaSuccess) {
      gpuCombinedWakeScratch.hostTable[channel] = tables[channel];
      gpuCombinedWakeScratch.hostTablePoints[channel] = wake->tablePoints;
    }
  }

  if (cudaStatus == cudaSuccess)
    gpuCombinedWakePrepareKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      gpuCombinedWakeScratch.time, gpuCombinedWakeScratch.pz,
      gpuCombinedWakeScratch.pbin, gpuCombinedWakeScratch.binnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess && wake->cpuOrder) {
    cudaStatus = gpuCombinedWakeLaunchSortedHistogram(
      static_cast<const double *>(coord), nParticles, stride, wake, threads);
  } else {
    if (cudaStatus == cudaSuccess)
      gpuCombinedWakeHistogramPartialKernel<<<histogramPartialBlocks, threads>>>(
        static_cast<const double *>(coord), nParticles, stride, *wake,
        gpuCombinedWakeScratch.pbin, gpuCombinedWakeScratch.histogramPartial,
        gpuCombinedWakeScratch.histogramPartials);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      gpuCombinedWakeHistogramReduceKernel<<<static_cast<int>(3 * wake->bins),
                                             threads,
                                             threads * sizeof(double)>>>(
        gpuCombinedWakeScratch.histogramPartial, wake->bins,
        gpuCombinedWakeScratch.histogramPartials,
        gpuCombinedWakeScratch.histogram);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaGetLastError();
  }
  if (cudaStatus == cudaSuccess && useTimeFft) {
    gpuCombinedWakeDriverFirstBinKernel<<<
      static_cast<int>((3 * wake->bins + threads - 1) / threads), threads>>>(
        gpuCombinedWakeScratch.histogram, wake->bins,
        gpuCombinedWakeScratch.driverFirstBin);
    cudaStatus = cudaGetLastError();
  }
  if (cudaStatus == cudaSuccess &&
      wake->mode == GPU_COMBINED_WAKE_MODE_TIME && !useTimeFft) {
    channelGrid = dim3(binBlocks, GPU_COMBINED_WAKE_CHANNELS, 1);
    gpuCombinedWakeConvolutionKernel<<<channelGrid, threads>>>(
      gpuCombinedWakeScratch.voltage,
      gpuCombinedWakeScratch.histogram,
      gpuCombinedWakeScratch.table, *wake,
      gpuCombinedWakeScratch.tableCapacity);
    cudaStatus = cudaGetLastError();
  } else if (cudaStatus == cudaSuccess &&
             wake->mode == GPU_COMBINED_WAKE_MODE_TIME) {
    int driverNeeded[3] = {0, 0, 0};
    int frequencyBlocks = static_cast<int>(
      (frequencyPoints + threads - 1) / threads);
    for (long channel = 0; channel < GPU_COMBINED_WAKE_CHANNELS; channel++)
      if (wake->enabled[channel])
        driverNeeded[wake->driver[channel]] = 1;
    for (int driver = 0; driver < 3 && cudaStatus == cudaSuccess; driver++) {
      if (!driverNeeded[driver])
        continue;
      cudaStatus = cudaMemset(gpuCombinedWakeScratch.fftReal, 0,
                              planBins * sizeof(*gpuCombinedWakeScratch.fftReal));
      if (cudaStatus == cudaSuccess)
        cudaStatus = cudaMemcpy(
          gpuCombinedWakeScratch.fftReal,
          gpuCombinedWakeScratch.histogram + driver * wake->bins,
          wake->bins * sizeof(*gpuCombinedWakeScratch.fftReal),
          cudaMemcpyDeviceToDevice);
      if (cudaStatus == cudaSuccess) {
        fftStatus = cufftExecD2Z(
          gpuCombinedWakeScratch.forwardPlan,
          gpuCombinedWakeScratch.fftReal,
          gpuCombinedWakeScratch.driverFrequency + driver * frequencyPoints);
        if (fftStatus != CUFFT_SUCCESS)
          cudaStatus = cudaErrorUnknown;
      }
    }
    for (long channel = 0;
         channel < GPU_COMBINED_WAKE_CHANNELS && cudaStatus == cudaSuccess;
         channel++) {
      if (!wake->enabled[channel])
        continue;
      gpuCombinedWakeFrequencyMultiplyKernel<<<frequencyBlocks, threads>>>(
        gpuCombinedWakeScratch.channelFrequency + channel * frequencyPoints,
        gpuCombinedWakeScratch.driverFrequency +
          wake->driver[channel] * frequencyPoints,
        gpuCombinedWakeScratch.tableFrequency + channel * frequencyPoints,
        frequencyPoints);
      cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess)
        continue;
      fftStatus = cufftExecZ2D(
        gpuCombinedWakeScratch.inversePlan,
        gpuCombinedWakeScratch.channelFrequency + channel * frequencyPoints,
        gpuCombinedWakeScratch.fftReal);
      if (fftStatus != CUFFT_SUCCESS) {
        cudaStatus = cudaErrorUnknown;
        continue;
      }
      gpuCombinedWakeExtractConvolutionKernel<<<binBlocks, threads>>>(
        gpuCombinedWakeScratch.voltage, gpuCombinedWakeScratch.fftReal,
        gpuCombinedWakeScratch.driverFirstBin, *wake, channel, planBins);
      cudaStatus = cudaGetLastError();
    }
  } else if (cudaStatus == cudaSuccess && hostVoltage) {
    /* impedance mode with the host's voltages */
    cudaStatus = cudaMemcpy(hostHistogram, gpuCombinedWakeScratch.histogram,
                            3 * wake->bins * sizeof(*hostHistogram),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus == cudaSuccess) {
      /* only the enabled channels: the kicks read no others, and the device's are already zero */
      for (long channel = 0; channel < GPU_COMBINED_WAKE_CHANNELS; channel++)
        if (wake->enabled[channel])
          std::memset(hostVoltageValues + channel * wake->bins, 0,
                      wake->bins * sizeof(*hostVoltageValues));
      hostVoltage(hostHistogram, hostVoltageValues, hostVoltageContext);
      for (long channel = 0;
           cudaStatus == cudaSuccess && channel < GPU_COMBINED_WAKE_CHANNELS; channel++)
        if (wake->enabled[channel])
          cudaStatus = cudaMemcpy(gpuCombinedWakeScratch.voltage + channel * wake->bins,
                                  hostVoltageValues + channel * wake->bins,
                                  wake->bins * sizeof(*hostVoltageValues),
                                  cudaMemcpyHostToDevice);
    }
  } else if (cudaStatus == cudaSuccess) {
    int driverNeeded[3] = {0, 0, 0};
    for (long channel = 0; channel < GPU_COMBINED_WAKE_CHANNELS; channel++)
      if (wake->enabled[channel])
        driverNeeded[wake->driver[channel]] = 1;
    for (int driver = 0; driver < 3 && cudaStatus == cudaSuccess; driver++) {
      if (!driverNeeded[driver])
        continue;
      fftStatus = cufftExecD2Z(
        gpuCombinedWakeScratch.forwardPlan,
        gpuCombinedWakeScratch.histogram + driver * wake->bins,
        gpuCombinedWakeScratch.driverFrequency + driver * frequencyPoints);
      if (fftStatus != CUFFT_SUCCESS) {
        std::fprintf(stderr, "elegant CUDA: cuFFT D2Z failed with status %d\n",
                     static_cast<int>(fftStatus));
        cudaStatus = cudaErrorUnknown;
      }
    }
    if (cudaStatus == cudaSuccess) {
      channelGrid = dim3(
        static_cast<unsigned int>((frequencyPoints + threads - 1) / threads),
        GPU_COMBINED_WAKE_CHANNELS, 1);
      gpuCombinedImpedanceMultiplyKernel<<<channelGrid, threads>>>(
        gpuCombinedWakeScratch.channelFrequency,
        gpuCombinedWakeScratch.driverFrequency,
        gpuCombinedWakeScratch.table, *wake,
        gpuCombinedWakeScratch.tableCapacity);
      cudaStatus = cudaGetLastError();
    }
    for (long channel = 0;
         channel < GPU_COMBINED_WAKE_CHANNELS && cudaStatus == cudaSuccess;
         channel++) {
      if (!wake->enabled[channel])
        continue;
      fftStatus = cufftExecZ2D(
        gpuCombinedWakeScratch.inversePlan,
        gpuCombinedWakeScratch.channelFrequency + channel * frequencyPoints,
        gpuCombinedWakeScratch.voltage + channel * wake->bins);
      if (fftStatus != CUFFT_SUCCESS) {
        std::fprintf(stderr, "elegant CUDA: cuFFT Z2D failed with status %d\n",
                     static_cast<int>(fftStatus));
        cudaStatus = cudaErrorUnknown;
      }
    }
  }

  if (cudaStatus == cudaSuccess)
    gpuCombinedWakeApplyKicksKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      gpuCombinedWakeScratch.time, gpuCombinedWakeScratch.pz,
      gpuCombinedWakeScratch.pbin, gpuCombinedWakeScratch.voltage);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount,
                            gpuCombinedWakeScratch.binnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && histogramReturn)
    cudaStatus = cudaMemcpy(histogramReturn,
                            gpuCombinedWakeScratch.histogram,
                            3 * wake->bins * sizeof(*histogramReturn),
                            cudaMemcpyDeviceToHost);
  if (voltageReturn) {
    for (long channel = 0;
         cudaStatus == cudaSuccess && channel < GPU_COMBINED_WAKE_CHANNELS;
         channel++)
      cudaStatus = cudaMemcpy(
        voltageReturn + channel * wake->bins,
        gpuCombinedWakeScratch.voltage + channel * wake->bins,
        wake->bins * sizeof(*voltageReturn), cudaMemcpyDeviceToHost);
  }

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);
  return status;
}

extern "C" int gpuCudaRfmodeHistogram(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  unsigned long long *histogramReturn, long *binnedCount,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long hostBinnedCount = 0;
  int status, threads = 256;
  int blocks;

  if (!coord || !data || !histogramReturn || !binnedCount ||
      nParticles <= 0 || stride < 6 || data->bins < 2 || data->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  status = ensureRfmodeScratch(nParticles, data->bins);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuRfmodeScratch.histogram, 0,
                          data->bins * sizeof(*gpuRfmodeScratch.histogram));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuRfmodeScratch.binnedCount, 0,
                            sizeof(*gpuRfmodeScratch.binnedCount));
  if (cudaStatus == cudaSuccess)
    gpuRfmodeHistogramKernel<<<blocks, threads>>>(
      nParticles, *data, gpuRfmodeScratch.time, gpuRfmodeScratch.pbin,
      gpuRfmodeScratch.histogram, gpuRfmodeScratch.binnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(histogramReturn, gpuRfmodeScratch.histogram,
                            data->bins * sizeof(*histogramReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, gpuRfmodeScratch.binnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);
  return status;
}

extern "C" int gpuCudaTrfmodeHistogram(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  unsigned long long *histogramReturn, double *xsumReturn,
  double *ysumReturn, long *binnedCount, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long hostBinnedCount = 0;
  int status, threads = 256;
  int blocks, sorted, endBit = 0;

  if (!coord || !data || !histogramReturn || !xsumReturn || !ysumReturn ||
      !binnedCount || nParticles <= 0 || stride < 6 || data->bins < 2 ||
      data->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  status = ensureRfmodeScratch(nParticles, data->bins);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* Without the sort, the one-thread-per-bin sums each scan the whole beam */
  sorted = ensureTrfmodeSortScratch(nParticles, data->bins) == static_cast<int>(cudaSuccess);
  if (!sorted)
    (void)cudaGetLastError();
  while (endBit < 32 && (1UL << endBit) <= static_cast<unsigned long>(data->bins))
    endBit++;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuRfmodeScratch.histogram, 0,
                          data->bins * sizeof(*gpuRfmodeScratch.histogram));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuRfmodeScratch.xsum, 0,
                            data->bins * sizeof(*gpuRfmodeScratch.xsum));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuRfmodeScratch.ysum, 0,
                            data->bins * sizeof(*gpuRfmodeScratch.ysum));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuRfmodeScratch.binnedCount, 0,
                            sizeof(*gpuRfmodeScratch.binnedCount));
  if (cudaStatus == cudaSuccess)
    gpuTrfmodeHistogramKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data,
      gpuRfmodeScratch.time, gpuRfmodeScratch.pbin,
      gpuRfmodeScratch.histogram, gpuRfmodeScratch.xsum,
      gpuRfmodeScratch.ysum, gpuRfmodeScratch.binnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess && sorted) {
    size_t bytes = gpuTrfmodeSortScratch.tempBytes;

    gpuTrfmodeSortKeysKernel<<<blocks, threads>>>(
      gpuRfmodeScratch.pbin, nParticles, data->bins, gpuTrfmodeSortScratch.keysIn,
      gpuTrfmodeSortScratch.indexIn);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = cub::DeviceRadixSort::SortPairs(
        gpuTrfmodeSortScratch.temp, bytes, gpuTrfmodeSortScratch.keysIn,
        gpuTrfmodeSortScratch.keysOut, gpuTrfmodeSortScratch.indexIn,
        gpuTrfmodeSortScratch.indexOut, static_cast<int>(nParticles), 0, endBit);
    bytes = gpuTrfmodeSortScratch.tempBytes;
    if (cudaStatus == cudaSuccess)
      cudaStatus = cub::DeviceScan::ExclusiveSum(
        gpuTrfmodeSortScratch.temp, bytes, gpuRfmodeScratch.histogram,
        gpuTrfmodeSortScratch.binStart, static_cast<int>(data->bins));
    if (cudaStatus == cudaSuccess)
      gpuTrfmodeSortedBinSumsKernel<<<static_cast<int>(data->bins), GPU_TRFMODE_SUM_THREADS>>>(
        static_cast<double *>(coord), stride, *data, gpuTrfmodeSortScratch.binStart,
        gpuRfmodeScratch.histogram, gpuTrfmodeSortScratch.indexOut, gpuRfmodeScratch.xsum,
        gpuRfmodeScratch.ysum);
  } else if (cudaStatus == cudaSuccess)
    gpuTrfmodeBinSumsKernel<<<static_cast<int>((data->bins + threads - 1) / threads),
                              threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data,
      gpuRfmodeScratch.pbin, gpuRfmodeScratch.xsum, gpuRfmodeScratch.ysum);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(histogramReturn, gpuRfmodeScratch.histogram,
                            data->bins * sizeof(*histogramReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(xsumReturn, gpuRfmodeScratch.xsum,
                            data->bins * sizeof(*xsumReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(ysumReturn, gpuRfmodeScratch.ysum,
                            data->bins * sizeof(*ysumReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, gpuRfmodeScratch.binnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);
  return status;
}

extern "C" int gpuCudaRfmodeTimeCoordinates(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  double *timeReturn, float *kernelMilliseconds,
  float *transferMilliseconds) {
  cudaEvent_t start, stop;
  int status, threads = 256;
  int blocks;

  if (!coord || !timeReturn || nParticles <= 0 || stride < 6 ||
      pCentral == 0 || cMks <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (kernelMilliseconds)
    *kernelMilliseconds = 0;
  if (transferMilliseconds)
    *transferMilliseconds = 0;
  status = ensureRfmodeParticleScratch(nParticles);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, kernelMilliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfmodeTimeKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride,
    pCentral, cMks, gpuRfmodeScratch.time);
  status = launchTimedKernel(cudaGetLastError(), start, stop,
                             kernelMilliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return timeCopy(timeReturn, gpuRfmodeScratch.time,
                  nParticles * sizeof(*timeReturn), cudaMemcpyDeviceToHost,
                  transferMilliseconds);
}

extern "C" int gpuCudaRfmodeApplyKicks(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  const double *voltage, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int status, threads = 256;
  int blocks;

  if (!coord || !data || !voltage || nParticles <= 0 || stride < 6 ||
      data->bins < 2 || data->dt <= 0 || data->firstBin < 0 ||
      data->lastBin < data->firstBin || data->lastBin >= data->bins)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuRfmodeScratch.time || !gpuRfmodeScratch.pbin ||
      gpuRfmodeScratch.particleCapacity < nParticles ||
      gpuRfmodeScratch.binCapacity < data->bins)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(gpuRfmodeScratch.voltage, voltage,
                          data->bins * sizeof(*voltage),
                          cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuRfmodeApplyKicksKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data,
      gpuRfmodeScratch.time, gpuRfmodeScratch.pbin,
      gpuRfmodeScratch.voltage);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  return launchTimedKernel(cudaStatus, start, stop, milliseconds);
}

extern "C" int gpuCudaTrfmodeApplyKicks(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  const double *voltageX, const double *voltageY,
  const double *voltageZ, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int status, threads = 256;
  int blocks;

  if (!coord || !data || !voltageX || !voltageY || !voltageZ ||
      nParticles <= 0 || stride < 6 || data->bins < 2 || data->dt <= 0 ||
      data->firstBin < 0 || data->lastBin < data->firstBin ||
      data->lastBin >= data->bins)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuRfmodeScratch.time || !gpuRfmodeScratch.pbin ||
      gpuRfmodeScratch.particleCapacity < nParticles ||
      gpuRfmodeScratch.binCapacity < data->bins)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(gpuRfmodeScratch.voltage, voltageX,
                          data->bins * sizeof(*voltageX),
                          cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(gpuRfmodeScratch.voltage + data->bins, voltageY,
                            data->bins * sizeof(*voltageY),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(gpuRfmodeScratch.voltage + 2 * data->bins,
                            voltageZ, data->bins * sizeof(*voltageZ),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuTrfmodeApplyKicksKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data,
      gpuRfmodeScratch.time, gpuRfmodeScratch.pbin,
      gpuRfmodeScratch.voltage);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  return launchTimedKernel(cudaStatus, start, stop, milliseconds);
}

extern "C" int gpuCudaPolynomialSeriesTrack(
  void *coord, long nParticles, int stride,
  const GPU_POLYNOMIAL_SERIES_DATA *data, const double *coefficient,
  const int32_t *exponent, const void *owner, long *invalidCount,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long hostInvalidCount = 0;
  unsigned long count;
  int threads = gpuParticleThreads(nParticles), blocks, status;

  if (!coord || !data || !coefficient || !exponent || !owner ||
      !invalidCount || nParticles <= 0 || stride < 6 ||
      data->totalTerms <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *invalidCount = 0;
  status = ensurePolynomialSeriesScratch(nParticles, stride,
                                         data->totalTerms);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  if (gpuPolynomialSeriesScratch.owner != owner ||
      gpuPolynomialSeriesScratch.ownerTerms != data->totalTerms) {
    cudaStatus = cudaMemcpy(
      gpuPolynomialSeriesScratch.coefficient, coefficient,
      data->totalTerms * sizeof(*coefficient), cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMemcpy(
        gpuPolynomialSeriesScratch.exponent, exponent,
        6 * data->totalTerms * sizeof(*exponent), cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess) {
      gpuPolynomialSeriesScratch.owner = owner;
      gpuPolynomialSeriesScratch.ownerTerms = data->totalTerms;
    }
  } else {
    cudaStatus = cudaSuccess;
  }
  count = (unsigned long)nParticles * (unsigned long)stride;
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(
      gpuPolynomialSeriesScratch.backup, coord,
      count * sizeof(*gpuPolynomialSeriesScratch.backup),
      cudaMemcpyDeviceToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(gpuPolynomialSeriesScratch.invalidCount, 0,
                            sizeof(*gpuPolynomialSeriesScratch.invalidCount));
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  if (cudaStatus == cudaSuccess) {
    gpuPolynomialSeriesKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data,
      gpuPolynomialSeriesScratch.coefficient,
      gpuPolynomialSeriesScratch.exponent,
      gpuPolynomialSeriesScratch.invalidCount);
    cudaStatus = cudaGetLastError();
  }
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(
      &hostInvalidCount, gpuPolynomialSeriesScratch.invalidCount,
      sizeof(hostInvalidCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostInvalidCount)
    cudaStatus = cudaMemcpy(
      coord, gpuPolynomialSeriesScratch.backup,
      count * sizeof(*gpuPolynomialSeriesScratch.backup),
      cudaMemcpyDeviceToDevice);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *invalidCount = static_cast<long>(hostInvalidCount);
  return status;
}

extern "C" int gpuCudaWakeLongitudinalTrack(void *coord, long nParticles,
                                            int stride,
                                            const GPU_WAKE_LONGITUDINAL_DATA *wake,
                                            const double *wakeTable,
                                            long *binnedCount,
                                            double *itimeReturn,
                                            double *vtimeReturn,
                                            float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  long *devicePbin = NULL;
  double *deviceItime = NULL;
  double *deviceVtime = NULL;
  double *deviceWake = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int binBlocks;
  int status;

  if (!coord || !wake || !wakeTable || !binnedCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->wakePoints <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  binBlocks = static_cast<int>((wake->bins + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &deviceVtime, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(4, &deviceWake, wake->wakePoints * sizeof(*deviceWake));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(5, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto cleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceItime, 0, wake->bins * sizeof(*deviceItime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceVtime, 0, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceWake, wakeTable,
                            wake->wakePoints * sizeof(*deviceWake),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuWakeBinKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePbin, deviceItime, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    gpuWakeConvolveKernel<<<binBlocks, threads>>>(deviceVtime, deviceItime,
                                                  deviceWake, *wake);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    gpuWakeApplyKicksKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePbin, deviceVtime);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && itimeReturn)
    cudaStatus = cudaMemcpy(itimeReturn, deviceItime,
                            wake->bins * sizeof(*itimeReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeReturn)
    cudaStatus = cudaMemcpy(vtimeReturn, deviceVtime,
                            wake->bins * sizeof(*vtimeReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

cleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaWakeLongitudinalHistogram(
  void *coord, long nParticles, int stride,
  const GPU_WAKE_LONGITUDINAL_DATA *wake, long *binnedCount,
  double *itimeReturn, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  long *devicePbin = NULL;
  double *deviceItime = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int status;

  if (!coord || !wake || !binnedCount || !itimeReturn)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto wakeHistogramCleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceItime, 0, wake->bins * sizeof(*deviceItime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess)
    gpuWakeBinKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePbin, deviceItime, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(itimeReturn, deviceItime,
                            wake->bins * sizeof(*itimeReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

wakeHistogramCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaWakeLongitudinalTrackFromHistogram(
  void *coord, long nParticles, int stride,
  const GPU_WAKE_LONGITUDINAL_DATA *wake, const double *wakeTable,
  const double *itimeInput, long *binnedCount,
  double *vtimeReturn, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  long *devicePbin = NULL;
  double *deviceItime = NULL;
  double *deviceVtime = NULL;
  double *deviceWake = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int binBlocks;
  int status;

  if (!coord || !wake || !wakeTable || !itimeInput || !binnedCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->wakePoints <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  binBlocks = static_cast<int>((wake->bins + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &deviceVtime, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(4, &deviceWake, wake->wakePoints * sizeof(*deviceWake));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(5, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto wakeTrackHistogramCleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceItime, 0, wake->bins * sizeof(*deviceItime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceVtime, 0, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceWake, wakeTable,
                            wake->wakePoints * sizeof(*deviceWake),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuWakeBinKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePbin, deviceItime, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceItime, itimeInput,
                            wake->bins * sizeof(*deviceItime),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuWakeConvolveKernel<<<binBlocks, threads>>>(deviceVtime, deviceItime,
                                                  deviceWake, *wake);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    gpuWakeApplyKicksKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePbin, deviceVtime);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeReturn)
    cudaStatus = cudaMemcpy(vtimeReturn, deviceVtime,
                            wake->bins * sizeof(*vtimeReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

wakeTrackHistogramCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

/* TRWAKE's bin sums.  In CPU order they go through the sort when the beam fits CUB's int count;
 * otherwise, and for the unordered sums, the kernel that scans the beam for each bin runs. */
static cudaError_t gpuTrwakeLaunchBinSums(double *coord, long nParticles, int stride,
                                          const GPU_TRWAKE_DATA *wake,
                                          const long *devicePbin,
                                          double *devicePosItimeX,
                                          double *devicePosItimeY, int threads) {
  cudaError_t cudaStatus = cudaSuccess;

  if (wake->cpuOrderSums &&
      ensureTrfmodeSortScratch(nParticles, wake->bins) == static_cast<int>(cudaSuccess)) {
    size_t bytes = gpuTrfmodeSortScratch.tempBytes;
    int particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
    int endBit = 0;

    while (endBit < 32 && (1UL << endBit) <= static_cast<unsigned long>(wake->bins))
      endBit++;
    gpuTrfmodeSortKeysKernel<<<particleBlocks, threads>>>(
      devicePbin, nParticles, wake->bins, gpuTrfmodeSortScratch.keysIn,
      gpuTrfmodeSortScratch.indexIn);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = cub::DeviceRadixSort::SortPairs(
        gpuTrfmodeSortScratch.temp, bytes, gpuTrfmodeSortScratch.keysIn,
        gpuTrfmodeSortScratch.keysOut, gpuTrfmodeSortScratch.indexIn,
        gpuTrfmodeSortScratch.indexOut, static_cast<int>(nParticles), 0, endBit);
    if (cudaStatus == cudaSuccess) {
      gpuTrwakeSortedBinSumsKernel<<<static_cast<int>((wake->bins + threads - 1) / threads),
                                     threads>>>(
        coord, nParticles, stride, *wake, gpuTrfmodeSortScratch.keysOut,
        gpuTrfmodeSortScratch.indexOut, devicePosItimeX, devicePosItimeY);
      cudaStatus = cudaGetLastError();
    }
    return cudaStatus;
  }
  (void)cudaGetLastError();
  gpuTrwakeBinSumsKernel<<<static_cast<int>(wake->bins), threads,
                           2 * threads * sizeof(double)>>>(
    coord, nParticles, stride, *wake, devicePbin, devicePosItimeX,
    devicePosItimeY);
  return cudaGetLastError();
}

extern "C" int gpuCudaTrwakeTrack(void *coord, long nParticles, int stride,
                                  const GPU_TRWAKE_DATA *wake,
                                  const double *wakeTableX,
                                  const double *wakeTableY,
                                  long *binnedCount,
                                  double *posItimeXReturn,
                                  double *posItimeYReturn,
                                  double *vtimeXReturn,
                                  double *vtimeYReturn,
                                  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  double *devicePz = NULL;
  long *devicePbin = NULL;
  double *devicePosItimeX = NULL;
  double *devicePosItimeY = NULL;
  double *deviceVtimeX = NULL;
  double *deviceVtimeY = NULL;
  double *deviceWakeX = NULL;
  double *deviceWakeY = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int binBlocks;
  int status;

  if (!coord || !wake || !binnedCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->wakePoints <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((wake->hasWake[0] && !wakeTableX) || (wake->hasWake[1] && !wakeTableY))
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  binBlocks = static_cast<int>((wake->bins + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(4, &devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(5, &deviceVtimeX, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(6, &deviceVtimeY, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  if (wake->hasWake[0]) {
    cudaStatus = gpuReuseDeviceBuffer(7, &deviceWakeX, wake->wakePoints * sizeof(*deviceWakeX));
    if (cudaStatus != cudaSuccess)
      goto trwakeCleanupWithoutEvents;
  }
  if (wake->hasWake[1]) {
    cudaStatus = gpuReuseDeviceBuffer(8, &deviceWakeY, wake->wakePoints * sizeof(*deviceWakeY));
    if (cudaStatus != cudaSuccess)
      goto trwakeCleanupWithoutEvents;
  }
  cudaStatus = gpuReuseDeviceBuffer(9, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto trwakeCleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceVtimeX, 0, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceVtimeY, 0, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess && wake->hasWake[0])
    cudaStatus = cudaMemcpy(deviceWakeX, wakeTableX,
                            wake->wakePoints * sizeof(*deviceWakeX),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess && wake->hasWake[1])
    cudaStatus = cudaMemcpy(deviceWakeY, wakeTableY,
                            wake->wakePoints * sizeof(*deviceWakeY),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuTrwakePrepareKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePz, devicePbin, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuTrwakeLaunchBinSums(static_cast<double *>(coord), nParticles, stride, wake,
                                        devicePbin, devicePosItimeX, devicePosItimeY, threads);
  if (cudaStatus == cudaSuccess && wake->hasWake[0])
    gpuTrwakeConvolveKernel<<<binBlocks, threads>>>(
      deviceVtimeX, devicePosItimeX, deviceWakeX, *wake, 0);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess && wake->hasWake[1])
    gpuTrwakeConvolveKernel<<<binBlocks, threads>>>(
      deviceVtimeY, devicePosItimeY, deviceWakeY, *wake, 1);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    gpuTrwakeApplyKicksKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePz, devicePbin, deviceVtimeX, deviceVtimeY);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && posItimeXReturn)
    cudaStatus = cudaMemcpy(posItimeXReturn, devicePosItimeX,
                            wake->bins * sizeof(*posItimeXReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && posItimeYReturn)
    cudaStatus = cudaMemcpy(posItimeYReturn, devicePosItimeY,
                            wake->bins * sizeof(*posItimeYReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeXReturn)
    cudaStatus = cudaMemcpy(vtimeXReturn, deviceVtimeX,
                            wake->bins * sizeof(*vtimeXReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeYReturn)
    cudaStatus = cudaMemcpy(vtimeYReturn, deviceVtimeY,
                            wake->bins * sizeof(*vtimeYReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

trwakeCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaTrwakeHistogram(void *coord, long nParticles,
                                      int stride,
                                      const GPU_TRWAKE_DATA *wake,
                                      long *binnedCount,
                                      double *posItimeXReturn,
                                      double *posItimeYReturn,
                                      float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  double *devicePz = NULL;
  long *devicePbin = NULL;
  double *devicePosItimeX = NULL;
  double *devicePosItimeY = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int status;

  if (!coord || !wake || !binnedCount || !posItimeXReturn ||
      !posItimeYReturn)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(4, &devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(5, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto trwakeHistogramCleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess)
    gpuTrwakePrepareKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePz, devicePbin, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuTrwakeLaunchBinSums(static_cast<double *>(coord), nParticles, stride, wake,
                                        devicePbin, devicePosItimeX, devicePosItimeY, threads);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(posItimeXReturn, devicePosItimeX,
                            wake->bins * sizeof(*posItimeXReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(posItimeYReturn, devicePosItimeY,
                            wake->bins * sizeof(*posItimeYReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

trwakeHistogramCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaTrwakeTrackFromHistogram(
  void *coord, long nParticles, int stride, const GPU_TRWAKE_DATA *wake,
  const double *wakeTableX, const double *wakeTableY,
  const double *posItimeXInput, const double *posItimeYInput,
  long *binnedCount, double *vtimeXReturn, double *vtimeYReturn,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  double *devicePz = NULL;
  long *devicePbin = NULL;
  double *devicePosItimeX = NULL;
  double *devicePosItimeY = NULL;
  double *deviceVtimeX = NULL;
  double *deviceVtimeY = NULL;
  double *deviceWakeX = NULL;
  double *deviceWakeY = NULL;
  unsigned long long *deviceBinnedCount = NULL;
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int particleBlocks;
  int binBlocks;
  int status;

  if (!coord || !wake || !binnedCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (wake->bins <= 0 || wake->wakePoints <= 0 || wake->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((wake->hasWake[0] && (!wakeTableX || !posItimeXInput)) ||
      (wake->hasWake[1] && (!wakeTableY || !posItimeYInput)))
    return static_cast<int>(cudaErrorInvalidValue);

  particleBlocks = static_cast<int>((nParticles + threads - 1) / threads);
  binBlocks = static_cast<int>((wake->bins + threads - 1) / threads);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(1, &devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(2, &devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(3, &devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(4, &devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(5, &deviceVtimeX, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = gpuReuseDeviceBuffer(6, &deviceVtimeY, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  if (wake->hasWake[0]) {
    cudaStatus = gpuReuseDeviceBuffer(7, &deviceWakeX, wake->wakePoints * sizeof(*deviceWakeX));
    if (cudaStatus != cudaSuccess)
      goto trwakeTrackHistogramCleanupWithoutEvents;
  }
  if (wake->hasWake[1]) {
    cudaStatus = gpuReuseDeviceBuffer(8, &deviceWakeY, wake->wakePoints * sizeof(*deviceWakeY));
    if (cudaStatus != cudaSuccess)
      goto trwakeTrackHistogramCleanupWithoutEvents;
  }
  cudaStatus = gpuReuseDeviceBuffer(9, &deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto trwakeTrackHistogramCleanupWithoutEvents;
  }

  cudaStatus = cudaMemset(deviceVtimeX, 0, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceVtimeY, 0, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess && wake->hasWake[0])
    cudaStatus = cudaMemcpy(deviceWakeX, wakeTableX,
                            wake->wakePoints * sizeof(*deviceWakeX),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess && wake->hasWake[1])
    cudaStatus = cudaMemcpy(deviceWakeY, wakeTableY,
                            wake->wakePoints * sizeof(*deviceWakeY),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuTrwakePrepareKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePz, devicePbin, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(devicePosItimeX, posItimeXInput,
                            wake->bins * sizeof(*devicePosItimeX),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(devicePosItimeY, posItimeYInput,
                            wake->bins * sizeof(*devicePosItimeY),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess && wake->hasWake[0])
    gpuTrwakeConvolveKernel<<<binBlocks, threads>>>(
      deviceVtimeX, devicePosItimeX, deviceWakeX, *wake, 0);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess && wake->hasWake[1])
    gpuTrwakeConvolveKernel<<<binBlocks, threads>>>(
      deviceVtimeY, devicePosItimeY, deviceWakeY, *wake, 1);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    gpuTrwakeApplyKicksKernel<<<particleBlocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      deviceTime, devicePz, devicePbin, deviceVtimeX, deviceVtimeY);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeXReturn)
    cudaStatus = cudaMemcpy(vtimeXReturn, deviceVtimeX,
                            wake->bins * sizeof(*vtimeXReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && vtimeYReturn)
    cudaStatus = cudaMemcpy(vtimeYReturn, deviceVtimeY,
                            wake->bins * sizeof(*vtimeYReturn),
                            cudaMemcpyDeviceToHost);

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

trwakeTrackHistogramCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLscBin(void *coord, long nParticles, int stride,
                             const GPU_LSC_DATA *lsc, long *binnedCount,
                             double *itimeReturn, void *deviceItimeScratch,
                             void *deviceBinnedCountScratch,
                             float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  double *deviceItime = static_cast<double *>(deviceItimeScratch);
  unsigned long long *deviceBinnedCount =
    static_cast<unsigned long long *>(deviceBinnedCountScratch);
  unsigned long long hostBinnedCount = 0;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || !lsc || !binnedCount || !itimeReturn)
    return static_cast<int>(cudaErrorInvalidValue);
  *binnedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (lsc->bins <= 0 || lsc->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  if (!deviceItime) {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceItime, lsc->bins * sizeof(*deviceItime));
    if (cudaStatus != cudaSuccess)
      goto lscBinCleanupWithoutEvents;
  }
  if (!deviceBinnedCount) {
    cudaStatus = gpuReuseDeviceBuffer(1, &deviceBinnedCount, sizeof(*deviceBinnedCount));
    if (cudaStatus != cudaSuccess)
      goto lscBinCleanupWithoutEvents;
  }

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto lscBinCleanupWithoutEvents;
  }

  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceItime, 0, lsc->bins * sizeof(*deviceItime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemset(deviceBinnedCount, 0, sizeof(*deviceBinnedCount));
  if (cudaStatus == cudaSuccess)
    gpuLscBinKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *lsc,
      deviceItime, deviceBinnedCount);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(&hostBinnedCount, deviceBinnedCount,
                            sizeof(hostBinnedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(itimeReturn, deviceItime,
                            lsc->bins * sizeof(*itimeReturn),
                            cudaMemcpyDeviceToHost);

  status = cudaStatus == cudaSuccess ?
             finishTimedKernel(start, stop, milliseconds) :
             static_cast<int>(cudaStatus);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  return status;

lscBinCleanupWithoutEvents:
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLscApplyKickAndDrift(void *coord, long nParticles,
                                           int stride,
                                           const GPU_LSC_DATA *lsc,
                                           const double *vtime,
                                           void *deviceVtimeScratch,
                                           float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  double *deviceVtime = static_cast<double *>(deviceVtimeScratch);
  int threads = 256;
  int blocks;
  int status;

  if (!coord || !lsc || !vtime)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (lsc->bins <= 0 || lsc->dt <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  if (!deviceVtime) {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceVtime, (lsc->bins + 1) * sizeof(*deviceVtime));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceVtime, vtime,
                            (lsc->bins + 1) * sizeof(*deviceVtime),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuLscApplyKickAndDriftKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *lsc, deviceVtime);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  status = cudaStatus == cudaSuccess ?
             finishTimedKernel(start, stop, milliseconds) :
             static_cast<int>(cudaStatus);

  return status;
}

extern "C" int gpuCudaScmultMoments(void *coord, long nParticles, int stride,
                                      GPU_SCMULT_MOMENT_DATA *result,
                                      float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int status;

  if (!coord || !result || nParticles <= 0 || stride < 5)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  std::memset(result, 0, sizeof(*result));

  status = ensureScmultScratch();
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess) {
    gpuScmultMomentPartialKernel<<<GPU_BEAM_SUM_BLOCKS,
                                    GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride,
      gpuScmultMomentResult);
    gpuScmultMomentFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
      gpuScmultMomentResult,
      gpuScmultMomentResult + GPU_BEAM_SUM_BLOCKS);
  }
  status = cudaStatus == cudaSuccess ?
             finishTimedKernel(start, stop, milliseconds) :
             static_cast<int>(cudaStatus);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result,
                          gpuScmultMomentResult + GPU_BEAM_SUM_BLOCKS,
                          sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaScmultLinearKick(void *coord, long nParticles, int stride,
                                       const GPU_SCMULT_LINEAR_DATA *data,
                                       float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || !data || stride < 6 || data->sigma[2] == 0 ||
      (data->horizontal && data->betax == 0) ||
      (data->vertical && data->betay == 0))
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuScmultLinearKickKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data);
  return cudaStatus == cudaSuccess ?
           finishTimedKernel(start, stop, milliseconds) :
           static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaScmultNonlinearKick(
  void *coord, long nParticles, int stride,
  const GPU_SCMULT_LINEAR_DATA *data, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int threads = 64;
  int blocks;
  int status;

  if (!coord || !data || stride < 6 || data->sigma[0] <= 0 ||
      data->sigma[1] <= 0 || data->sigma[2] == 0 ||
      (data->horizontal && data->betax == 0) ||
      (data->vertical && data->betay == 0))
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuScmultNonlinearKickKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *data);
  return cudaStatus == cudaSuccess ?
           finishTimedKernel(start, stop, milliseconds) :
           static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaCsrCsbendWake(const double *ctHist,
                                    const double *ctHistDeriv,
                                    const double *denom,
                                    double *T1,
                                    double *T2,
                                    double *dGamma,
                                    long nBins,
                                    double CSRConstant,
                                    double dsSlice,
                                    double slippageLength13,
                                    double dct,
                                    long steadyState,
                                    long trapazoidIntegration,
                                    long diSlippage,
                                    long diSlippage4,
                                    double t1Factor,
                                    double t2Factor,
                                    float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks;
  int status;

  if (!ctHist || !ctHistDeriv || !denom || !T1 || !T2 || !dGamma ||
      nBins <= 0 || dct <= 0 || slippageLength13 == 0)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nBins + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCsrCsbendWakeKernel<<<blocks, threads>>>(
    ctHist, ctHistDeriv, denom, T1, T2, dGamma, nBins, CSRConstant,
    dsSlice, slippageLength13, dct, steadyState, trapazoidIntegration,
    diSlippage, diSlippage4, t1Factor, t2Factor);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrCsbendKickInPlace(void *coord,
                                           long nParticles,
                                           int stride,
                                           const double *dGamma,
                                           long nBins,
                                           double ctLower,
                                           double dct,
                                           double Po,
                                           double rho0,
                                           float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || !dGamma || nParticles <= 0 || stride < 6 ||
      nBins < 2 || dct <= 0 || Po == 0 || rho0 == 0)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCsrCsbendKickInPlaceKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, dGamma,
    nBins, ctLower, dct, Po, rho0);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrCsbendOutputRoundTrip(void *coord,
                                               long nParticles,
                                               int stride,
                                               double cosTilt,
                                               double sinTilt,
                                               double cMks,
                                               float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || nParticles <= 0 || stride < 6 || cMks == 0)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCsrCsbendOutputRoundTripKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, cosTilt, sinTilt, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrDriftAdvance(void *coord,
                                      long nParticles,
                                      int stride,
                                      double dz,
                                      double Po,
                                      long linearOptics,
                                      long toTime,
                                      float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCsrDriftAdvanceKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, dz, Po, linearOptics, toTime);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrDriftKick(void *coord,
                                   long nParticles,
                                   int stride,
                                   const double *dGamma,
                                   long nBins,
                                   long applyKick,
                                   double ct0,
                                   double dct,
                                   double Po,
                                   long useFactor,
                                   double factor,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || nParticles <= 0 || stride < 6 || Po == 0 ||
      (dGamma && applyKick && (nBins < 2 || dct <= 0)))
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCsrDriftKickKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, dGamma, nBins, applyKick,
    ct0, dct, Po, useFactor, factor);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrDriftCountInRange(void *coord,
                                           long nParticles,
                                           int stride,
                                           double ct0,
                                           double dct,
                                           long nBins,
                                           long *count,
                                           float *milliseconds) {
  cudaEvent_t start, stop;
  GpuCsrDriftInRange inRange;
  int status;

  if (!coord || !count || nParticles <= 0 || stride < 6 || nBins < 2 || dct <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  inRange.coord = static_cast<const double *>(coord);
  inRange.stride = stride;
  inRange.ct0 = ct0;
  inRange.dct = dct;
  inRange.nBins1 = nBins - 1;
  try {
    *count = thrust::transform_reduce(thrust::device,
                                      thrust::counting_iterator<long>(0),
                                      thrust::counting_iterator<long>(nParticles),
                                      inRange, 0L, thrust::plus<long>());
  } catch (...) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCsrHistogram(void *coord,
                                   long nParticles,
                                   int stride,
                                   long coordinateIndex,
                                   double lower,
                                   double binSize,
                                   long bins,
                                   double *deviceHist,
                                   double *histReturn,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int threads = 256;
  int blocks;
  int status;

  if (!coord || !deviceHist || !histReturn || nParticles <= 0 ||
      stride < 1 || coordinateIndex < 0 || coordinateIndex >= stride ||
      bins <= 0 || binSize <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  cudaStatus = cudaMemset(deviceHist, 0, bins * sizeof(*deviceHist));
  if (cudaStatus == cudaSuccess)
    gpuCsrHistogramKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, coordinateIndex,
      lower, binSize, bins, deviceHist);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(histReturn, deviceHist,
                            bins * sizeof(*histReturn),
                            cudaMemcpyDeviceToHost);
  return launchTimedKernel(cudaStatus, start, stop, milliseconds);
}

extern "C" int gpuCudaAddCoordinate(void *coord, long nParticles, int stride,
                                    int index, double value, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuAddCoordinateKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                              index, value);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaOffsetBeam(void *coord, long nParticles, int stride,
                                 double dx, double dxp, double dy, double dyp,
                                 double dz, double dt, double dp, double de,
                                 double pCentral, int64_t startPID, int64_t endPID,
                                 int allParticles, double cMks, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuOffsetBeamKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                           dx, dxp, dy, dyp, dz, dt, dp, de,
                                           pCentral, startPID, endPID,
                                           allParticles, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* Records x, x', y and y' of each particle for one turn of a batched tune history, at row
 * (particle ID - 1) * turns + turn of each coordinate's block */
__global__ void gpuTuneHistoryRecordKernel(const double *coord, long nParticles, int stride, int idColumn,
                                           long particles, long turns, long turn, double *history) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  long id, block = particles * turns;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  id = (long)part[idColumn] - 1;
  if (id < 0 || id >= particles)
    return;
  history[id * turns + turn] = part[0];
  history[block + id * turns + turn] = part[1];
  history[2 * block + id * turns + turn] = part[2];
  history[3 * block + id * turns + turn] = part[3];
}

extern "C" int gpuCudaTuneHistoryRecord(const void *coord, long nParticles, int stride, int idColumn,
                                        long particles, long turns, long turn, void *history) {
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);

  if (!coord || !history || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  gpuTuneHistoryRecordKernel<<<blocks, threads>>>(static_cast<const double *>(coord), nParticles, stride,
                                                  idColumn, particles, turns, turn,
                                                  static_cast<double *>(history));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gpuCudaClearBatchedSearchHistory(
  void *history, unsigned long historyCount, void *turnCount,
  unsigned long particles) {
  cudaError_t status;
  status = cudaMemset(history, 0, historyCount * sizeof(double));
  if (status == cudaSuccess)
    status = cudaMemset(turnCount, 0, particles * sizeof(double));
  return static_cast<int>(status);
}

extern "C" int gpuCudaTuneProgramTrack(
  const GPU_TUNE_PROGRAM *program, const double *coordinate, long particles,
  int stride, long turns, long turnOffset, double *endingCoordinate,
  GPU_TUNE_PARTICLE_STATUS *particleStatus, double *history,
  float *milliseconds, long *launches, long *transfers) {
  GPU_TUNE_PROGRAM deviceProgram;
  double *deviceCoordinate = NULL, *deviceHistory = NULL;
  GPU_TUNE_PARTICLE_STATUS *deviceStatus = NULL;
  cudaEvent_t start = NULL, stop = NULL;
  cudaDeviceProp deviceProperties;
  cudaError_t cudaStatus = cudaSuccess;
  int currentDevice = 0, threads = 32, blocks, status;
  size_t coordinateValues, historyValues = 0;

#define GPU_TUNE_DEVICE_POINTERS(action)               \
  action(op);                                          \
  action(drift);                                       \
  action(matrix);                                      \
  action(multipole);                                   \
  action(exactCorrector);                              \
  action(csbend);                                      \
  action(ccbend);                                      \
  action(lgbend);                                      \
  action(lgbendLocalAperture);                         \
  action(collimator);                                  \
  action(scraper);                                     \
  action(taperAperture);                               \
  action(speedbump);                                   \
  action(malign);                                      \
  action(postAperture)
#define GPU_TUNE_NULL(member) deviceProgram.member = NULL
#define GPU_TUNE_FREE(member) gpuFreeDevice(deviceProgram.member)
#define GPU_TUNE_UPLOAD(member, count)                                  \
  do {                                                                   \
    if ((count) > 0) {                                                   \
      cudaStatus = cudaMalloc(                                           \
        (void **)&deviceProgram.member,                                  \
        (size_t)(count) * sizeof(*deviceProgram.member));                \
      if (cudaStatus != cudaSuccess)                                     \
        goto cleanup;                                                    \
      cudaStatus = cudaMemcpy(                                           \
        deviceProgram.member, program->member,                           \
        (size_t)(count) * sizeof(*deviceProgram.member),                 \
        cudaMemcpyHostToDevice);                                         \
      if (cudaStatus != cudaSuccess)                                     \
        goto cleanup;                                                    \
    }                                                                    \
  } while (0)

  if (milliseconds)
    *milliseconds = 0;
  if (launches)
    *launches = 0;
  if (transfers)
    *transfers = 0;
  if (!program || program->opCount <= 0 || !program->op ||
      !coordinate || particles <= 0 || stride < 6 || turns <= 1 ||
      !endingCoordinate || !particleStatus)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((size_t)particles > SIZE_MAX / (size_t)stride)
    return static_cast<int>(cudaErrorInvalidValue);
  coordinateValues = (size_t)particles * (size_t)stride;
  if (history) {
    if ((size_t)particles > SIZE_MAX / (size_t)turns / 4)
      return static_cast<int>(cudaErrorInvalidValue);
    historyValues = 4 * (size_t)particles * (size_t)turns;
  }
  /*
   * The fused interpreter has a deliberately large per-thread register
   * footprint.  Spread a small fixed-slot grid across the available SMs,
   * even when this means partially filled warps.  Full-size grids retain
   * one-warp blocks and therefore do not sacrifice steady-state throughput.
   */
  if (cudaGetDevice(&currentDevice) == cudaSuccess &&
      cudaGetDeviceProperties(&deviceProperties, currentDevice) ==
        cudaSuccess &&
      deviceProperties.multiProcessorCount > 0) {
    long slotsPerSm =
      (particles + deviceProperties.multiProcessorCount - 1) /
      deviceProperties.multiProcessorCount;
    if (slotsPerSm < threads)
      threads = slotsPerSm > 0 ? (int)slotsPerSm : 1;
  }
  blocks = (int)((particles + threads - 1) / threads);

  deviceProgram = *program;
  GPU_TUNE_DEVICE_POINTERS(GPU_TUNE_NULL);
  GPU_TUNE_UPLOAD(op, program->opCount);
  GPU_TUNE_UPLOAD(drift, program->driftCount);
  GPU_TUNE_UPLOAD(matrix, program->matrixCount);
  GPU_TUNE_UPLOAD(multipole, program->multipoleCount);
  GPU_TUNE_UPLOAD(exactCorrector, program->exactCorrectorCount);
  GPU_TUNE_UPLOAD(csbend, program->csbendCount);
  GPU_TUNE_UPLOAD(ccbend, program->ccbendCount);
  GPU_TUNE_UPLOAD(lgbend, program->lgbendCount);
  GPU_TUNE_UPLOAD(lgbendLocalAperture,
                  program->lgbendLocalApertureCount);
  GPU_TUNE_UPLOAD(collimator, program->collimatorCount);
  GPU_TUNE_UPLOAD(scraper, program->scraperCount);
  GPU_TUNE_UPLOAD(taperAperture, program->taperApertureCount);
  GPU_TUNE_UPLOAD(speedbump, program->speedbumpCount);
  GPU_TUNE_UPLOAD(malign, program->malignCount);
  GPU_TUNE_UPLOAD(postAperture, program->postApertureCount);

  cudaStatus = cudaMalloc(
    &deviceCoordinate, coordinateValues * sizeof(*deviceCoordinate));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(
    &deviceStatus, (size_t)particles * sizeof(*deviceStatus));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  if (historyValues) {
    cudaStatus = cudaMalloc(
      &deviceHistory, historyValues * sizeof(*deviceHistory));
    if (cudaStatus != cudaSuccess)
      goto cleanup;
    cudaStatus = cudaMemset(
      deviceHistory, 0, historyValues * sizeof(*deviceHistory));
    if (cudaStatus != cudaSuccess)
      goto cleanup;
  }
  cudaStatus = cudaMemcpy(
    deviceCoordinate, coordinate,
    coordinateValues * sizeof(*deviceCoordinate), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  if (transfers)
    (*transfers)++;

  cudaStatus = cudaEventCreate(&start);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaEventCreate(&stop);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  gpuTuneProgramInitializeKernel<<<blocks, threads>>>(
    deviceCoordinate, particles, stride, deviceStatus, deviceHistory, turns);
  cudaStatus = cudaGetLastError();
  if (launches)
    (*launches)++;
  for (long turn = 1; cudaStatus == cudaSuccess && turn < turns; turn++) {
    gpuTuneProgramTurnKernel<<<blocks, threads>>>(
      deviceCoordinate, particles, stride, deviceProgram, deviceStatus,
      deviceHistory, turns, turn, turnOffset);
    cudaStatus = cudaGetLastError();
    if (launches)
      (*launches)++;
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  start = stop = NULL;
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto cleanup;
  }

  cudaStatus = cudaMemcpy(
    endingCoordinate, deviceCoordinate,
    coordinateValues * sizeof(*deviceCoordinate), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  if (transfers)
    (*transfers)++;
  cudaStatus = cudaMemcpy(
    particleStatus, deviceStatus,
    (size_t)particles * sizeof(*deviceStatus), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  if (transfers)
    (*transfers)++;
  if (historyValues) {
    cudaStatus = cudaMemcpy(
      history, deviceHistory, historyValues * sizeof(*deviceHistory),
      cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      goto cleanup;
    if (transfers)
      (*transfers)++;
  }

cleanup:
  if (start)
    cudaEventDestroy(start);
  if (stop)
    cudaEventDestroy(stop);
  gpuFreeDevice(deviceHistory);
  gpuFreeDevice(deviceStatus);
  gpuFreeDevice(deviceCoordinate);
  GPU_TUNE_DEVICE_POINTERS(GPU_TUNE_FREE);
#undef GPU_TUNE_UPLOAD
#undef GPU_TUNE_FREE
#undef GPU_TUNE_NULL
#undef GPU_TUNE_DEVICE_POINTERS
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaBatchedApertureSearch(
  const GPU_BATCHED_APERTURE_ELEMENT *element, long elements,
  long lines, long nx, long nSplits, double splitFraction, long passes,
  double pCentral, double xmax, double ymax, const double *orbit,
  const double *dxFactor, const double *dyFactor,
  double *xLimitReturn, double *yLimitReturn,
  double *xLostReturn, double *yLostReturn, double *sLostReturn,
  long *lossPassReturn, long *lossElementReturn,
  double *lossX0Return, double *lossY0Return, long *originStableReturn,
  float *milliseconds) {
  GPU_BATCHED_APERTURE_ELEMENT *deviceElement = NULL;
  double *deviceLine = NULL, *deviceTrial = NULL, *deviceTrialLoss = NULL;
  double *deviceFactor = NULL, *deviceOrbit = NULL;
  long *deviceLineLoss = NULL, *deviceTrialLossIndex = NULL;
  unsigned char *deviceSurvived = NULL;
  long *deviceOriginStable = NULL;
  double *dx, *dy, *x0, *y0, *xLimit, *yLimit, *xLost, *yLost, *sLost;
  double *lossX0, *lossY0;
  long *lossPass, *lossElement;
  long maxSteps, maxTrials, lineBlocks, trialBlocks;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus = cudaSuccess;
  int threads = 128;
  int status = static_cast<int>(cudaSuccess);

  if (!element || elements <= 0 || lines <= 0 || nx < 2 || nSplits < 0 ||
      splitFraction <= 0 || splitFraction >= 1 || passes <= 0 ||
      !orbit || !dxFactor || !dyFactor || !xLimitReturn || !yLimitReturn ||
      !xLostReturn || !yLostReturn || !sLostReturn || !lossPassReturn ||
      !lossElementReturn || !lossX0Return || !lossY0Return ||
      !originStableReturn)
    return static_cast<int>(cudaErrorInvalidValue);
  maxSteps = static_cast<long>(1 / splitFraction - 0.5);
  if (maxSteps < nx)
    maxSteps = nx;
  if (maxSteps < 1 || lines > LONG_MAX / maxSteps)
    return static_cast<int>(cudaErrorInvalidValue);
  maxTrials = lines * maxSteps;
  lineBlocks = (lines + threads - 1) / threads;
  trialBlocks = (maxTrials + threads - 1) / threads;

  cudaStatus = cudaMalloc(&deviceElement, elements * sizeof(*deviceElement));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceFactor, 2 * lines * sizeof(*deviceFactor));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceOrbit, 6 * sizeof(*deviceOrbit));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceLine, 11 * lines * sizeof(*deviceLine));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceLineLoss, 2 * lines * sizeof(*deviceLineLoss));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceOriginStable, sizeof(*deviceOriginStable));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceTrial, 6 * maxTrials * sizeof(*deviceTrial));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceSurvived,
                          maxTrials * sizeof(*deviceSurvived));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceTrialLoss,
                          3 * maxTrials * sizeof(*deviceTrialLoss));
  if (cudaStatus != cudaSuccess)
    goto cleanup;
  cudaStatus = cudaMalloc(&deviceTrialLossIndex,
                          2 * maxTrials * sizeof(*deviceTrialLossIndex));
  if (cudaStatus != cudaSuccess)
    goto cleanup;

  dx = deviceLine;
  dy = dx + lines;
  x0 = dy + lines;
  y0 = x0 + lines;
  xLimit = y0 + lines;
  yLimit = xLimit + lines;
  xLost = yLimit + lines;
  yLost = xLost + lines;
  sLost = yLost + lines;
  lossX0 = sLost + lines;
  lossY0 = lossX0 + lines;
  lossPass = deviceLineLoss;
  lossElement = lossPass + lines;

  cudaStatus = cudaMemcpy(deviceElement, element,
                          elements * sizeof(*deviceElement),
                          cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceFactor, dxFactor,
                            lines * sizeof(*deviceFactor),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceFactor + lines, dyFactor,
                            lines * sizeof(*deviceFactor),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceOrbit, orbit, 6 * sizeof(*deviceOrbit),
                            cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    goto cleanup;
  gpuBatchedApertureInitializeKernel<<<lineBlocks, threads>>>(
    lines, nx, xmax, ymax, deviceFactor, deviceFactor + lines,
    dx, dy, x0, y0, xLimit, yLimit, xLost, yLost, sLost,
    lossPass, lossElement, deviceOriginStable);
  cudaStatus = cudaGetLastError();
  for (long split = 0; cudaStatus == cudaSuccess && split <= nSplits; split++) {
    long nSteps = split == 0 ? nx :
      static_cast<long>(1 / splitFraction - 0.5);
    long trials;

    if (nSteps < 1)
      nSteps = 1;
    trials = lines * nSteps;
    if (split > 0) {
      gpuBatchedApertureRefineLinesKernel<<<lineBlocks, threads>>>(
        lines, splitFraction, dx, dy, x0, y0, xLimit, yLimit);
      cudaStatus = cudaGetLastError();
    }
    if (cudaStatus == cudaSuccess) {
      trialBlocks = (trials + threads - 1) / threads;
      gpuBatchedAperturePrepareKernel<<<trialBlocks, threads>>>(
        lines, nSteps, deviceOrbit, dx, dy, x0, y0, deviceTrial, 6);
      cudaStatus = cudaGetLastError();
    }
    if (cudaStatus == cudaSuccess) {
      gpuBatchedApertureTrackKernel<<<trialBlocks, threads>>>(
        deviceTrial, trials, 6, deviceElement, elements, passes, pCentral,
        deviceSurvived, deviceTrialLoss,
        deviceTrialLoss + maxTrials, deviceTrialLoss + 2 * maxTrials,
        deviceTrialLossIndex, deviceTrialLossIndex + maxTrials);
      cudaStatus = cudaGetLastError();
    }
    if (cudaStatus == cudaSuccess) {
      gpuBatchedApertureReduceKernel<<<lineBlocks, threads>>>(
        lines, nSteps, split, deviceSurvived, deviceTrialLoss,
        deviceTrialLoss + maxTrials, deviceTrialLoss + 2 * maxTrials,
        deviceTrialLossIndex, deviceTrialLossIndex + maxTrials,
        dx, dy, x0, y0, xLimit, yLimit, xLost, yLost, sLost,
        lossPass, lossElement, lossX0, lossY0, deviceOriginStable);
      cudaStatus = cudaGetLastError();
    }
  }
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(xLimitReturn, xLimit,
                            lines * sizeof(*xLimitReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(yLimitReturn, yLimit,
                            lines * sizeof(*yLimitReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(xLostReturn, xLost,
                            lines * sizeof(*xLostReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(yLostReturn, yLost,
                            lines * sizeof(*yLostReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(sLostReturn, sLost,
                            lines * sizeof(*sLostReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(lossPassReturn, lossPass,
                            lines * sizeof(*lossPassReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(lossElementReturn, lossElement,
                            lines * sizeof(*lossElementReturn),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(lossX0Return, lossX0,
                            lines * sizeof(*lossX0Return),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(lossY0Return, lossY0,
                            lines * sizeof(*lossY0Return),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(originStableReturn, deviceOriginStable,
                            sizeof(*originStableReturn),
                            cudaMemcpyDeviceToHost);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);

cleanup:
  gpuFreeDevice(deviceElement);
  gpuFreeDevice(deviceFactor);
  gpuFreeDevice(deviceOrbit);
  gpuFreeDevice(deviceLine);
  gpuFreeDevice(deviceLineLoss);
  gpuFreeDevice(deviceOriginStable);
  gpuFreeDevice(deviceTrial);
  gpuFreeDevice(deviceSurvived);
  gpuFreeDevice(deviceTrialLoss);
  gpuFreeDevice(deviceTrialLossIndex);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaApplyBatchedMomentumSearch(
  void *coord, long nParticles, int stride, const void *searchData,
  long searchParticles, long target, long pass, long firePass,
  void *history, void *historyCount, long turns,
  double dx, double dy, double pCentral, double cMks,
  float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuBatchedMomentumSearchKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride,
    static_cast<const double *>(searchData), searchParticles,
    target, pass, firePass, static_cast<double *>(history),
    static_cast<double *>(historyCount), turns,
    dx, dy, pCentral, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaSetCentralMomentum(void *coord, long nParticles, int stride,
                                         double oldP, double newP, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSetCentralMomentumKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                                   stride, oldP, newP);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaParticleTimes(void *coord, long nParticles, int stride, double pCentral,
                                    double cMks, void *deviceTimes, double *hostTimes,
                                    float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuParticleTimesKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                              pCentral, cMks, static_cast<double *>(deviceTimes));
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(hostTimes, deviceTimes, nParticles * sizeof(double),
                                     cudaMemcpyDeviceToHost));
}

extern "C" int gpuCudaRamppRescale(void *coord, long nParticles, int stride, double oldP,
                                   double newP, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRamppRescaleKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                             oldP, newP);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaMatchEnergy(void *coord, long nParticles, int stride,
                                  double oldP, double averageP, int changeBeam,
                                  float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuMatchEnergyKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                            stride, oldP, averageP, changeBeam);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaMatchEnergyAndAverage(void *coord, long nParticles,
                                            int stride, double oldP,
                                            int changeBeam,
                                            GPU_BEAM_SUM_DATA *result,
                                            void *deviceScratch,
                                            float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  GPU_MATCH_ENERGY_PARTIAL *partial;
  GPU_BEAM_SUM_DATA *deviceResult;
  double *deviceAverage;
  char *scratch = static_cast<char *>(deviceScratch);
  int blocks, status;

  if (!coord || !result || !deviceScratch || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  partial = reinterpret_cast<GPU_MATCH_ENERGY_PARTIAL *>(scratch);
  deviceResult = reinterpret_cast<GPU_BEAM_SUM_DATA *>(
    partial + GPU_MATCH_ENERGY_BLOCKS);
  deviceAverage = reinterpret_cast<double *>(deviceResult + 1);
  blocks = static_cast<int>((nParticles + GPU_BEAM_OUTPUT_THREADS - 1) /
                            GPU_BEAM_OUTPUT_THREADS);
  if (blocks > GPU_MATCH_ENERGY_BLOCKS)
    blocks = GPU_MATCH_ENERGY_BLOCKS;
  if (blocks < 1)
    blocks = 1;

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuMatchEnergyPartialKernel<<<blocks, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, oldP, partial);
  gpuMatchEnergyFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
    partial, blocks, deviceResult, deviceAverage);
  gpuMatchEnergyApplyKernel<<<static_cast<int>((nParticles + 255) / 256), 256>>>(
    static_cast<double *>(coord), nParticles, stride, oldP, deviceAverage,
    changeBeam);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

/* One coordinate of every particle, in particle order, copied to the host */
extern "C" int gpuCudaCoordinateColumnToHost(const void *coord, long nParticles, int stride,
                                             int column, double *values, float *milliseconds) {
  double *deviceValues = NULL;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int status;

  if (milliseconds)
    *milliseconds = 0;
  if (!coord || !values || nParticles <= 0 || column < 0 || column >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceValues, nParticles * sizeof(*deviceValues));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCoordinateColumnKernel<<<static_cast<int>((nParticles + 255) / 256), 256>>>(
    static_cast<const double *>(coord), nParticles, stride, column, deviceValues);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(values, deviceValues, nParticles * sizeof(*values),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

/* A copy of one coordinate of every particle kept on the device, so a later call can tell at little
 * cost whether any particle's value has changed since: momentum matching in CPU order uses it to skip
 * its sum over a beam that it has already found matched. */
static double *gpuColumnSnapshot = NULL;
static long gpuColumnSnapshotCapacity = 0;
static unsigned int *gpuColumnSnapshotDifferent = NULL;

__global__ void gpuColumnSnapshotTakeKernel(const double *coord, long nParticles, int stride,
                                            int column, double *snapshot) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    snapshot[ip] = coord[ip * stride + column];
}

/* Bits, not values: -0 for 0 or one NaN for another is a change too */
__global__ void gpuColumnSnapshotCompareKernel(const double *coord, long nParticles, int stride,
                                               int column, const double *snapshot,
                                               unsigned int *different) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles &&
      __double_as_longlong(coord[ip * stride + column]) != __double_as_longlong(snapshot[ip]))
    *different = 1;
}

extern "C" void gpuCudaColumnSnapshotRelease(void) {
  gpuFreeDevice(gpuColumnSnapshot);
  gpuFreeDevice(gpuColumnSnapshotDifferent);
  gpuColumnSnapshot = NULL;
  gpuColumnSnapshotDifferent = NULL;
  gpuColumnSnapshotCapacity = 0;
}

extern "C" int gpuCudaColumnSnapshotTake(const void *coord, long nParticles, int stride,
                                         int column, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus = cudaSuccess;
  int status;

  if (milliseconds)
    *milliseconds = 0;
  if (!coord || nParticles <= 0 || column < 0 || column >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuColumnSnapshotCapacity < nParticles) {
    gpuFreeDevice(gpuColumnSnapshot);
    gpuColumnSnapshot = NULL;
    gpuColumnSnapshotCapacity = 0;
    cudaStatus = cudaMalloc(&gpuColumnSnapshot, nParticles * sizeof(*gpuColumnSnapshot));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuColumnSnapshotCapacity = nParticles;
  }
  if (!gpuColumnSnapshotDifferent) {
    cudaStatus = cudaMalloc(&gpuColumnSnapshotDifferent, sizeof(*gpuColumnSnapshotDifferent));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuColumnSnapshotTakeKernel<<<static_cast<int>((nParticles + 255) / 256), 256>>>(
    static_cast<const double *>(coord), nParticles, stride, column, gpuColumnSnapshot);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* *same is 1 when every particle's value has the bits the snapshot holds */
extern "C" int gpuCudaColumnSnapshotCompare(const void *coord, long nParticles, int stride,
                                            int column, int *same, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned int different = 1;
  int status;

  if (milliseconds)
    *milliseconds = 0;
  if (!same)
    return static_cast<int>(cudaErrorInvalidValue);
  *same = 0;
  if (!coord || nParticles <= 0 || column < 0 || column >= stride ||
      !gpuColumnSnapshot || !gpuColumnSnapshotDifferent || gpuColumnSnapshotCapacity < nParticles)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuColumnSnapshotDifferent, 0, sizeof(*gpuColumnSnapshotDifferent));
  if (cudaStatus == cudaSuccess)
    gpuColumnSnapshotCompareKernel<<<static_cast<int>((nParticles + 255) / 256), 256>>>(
      static_cast<const double *>(coord), nParticles, stride, column, gpuColumnSnapshot,
      gpuColumnSnapshotDifferent);
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&different, gpuColumnSnapshotDifferent, sizeof(different),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *same = different ? 0 : 1;
  return static_cast<int>(cudaSuccess);
}

extern "C" unsigned long gpuCudaMatchEnergyScratchBytes(void) {
  return GPU_MATCH_ENERGY_BLOCKS * sizeof(GPU_MATCH_ENERGY_PARTIAL) +
         sizeof(GPU_BEAM_SUM_DATA) + sizeof(double);
}

extern "C" int gpuCudaRfcaPhaseValues(void *coord, void *phaseValues,
  void *momentum, long nParticles, int stride, double pCentral,
  double length, double omega, double phase, double timeShift, double cMks,
  float *milliseconds) {
  cudaEvent_t start, stop;
  int status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcaPhaseKernel<<<(nParticles + 255) / 256, 256>>>(
    static_cast<double *>(coord), static_cast<double *>(phaseValues),
    static_cast<double *>(momentum), nParticles, stride, pCentral, length,
    omega, phase, timeShift, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfcaThinKick(void *coord, long nParticles, int stride,
                                   double pCentral, double volt, double omega,
                                   double phase, double cMks,
                                   const void *referenceKick,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcaThinKickKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
    stride, pCentral, volt, omega, phase, cMks,
    static_cast<const double *>(referenceKick));
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* badCount is read back after stages 0 and 2 only */
extern "C" int gpuCudaRamprf(void *coord, long nParticles, int stride,
                             double pCentral, double length, double volt,
                             double omega, double phase, double tOffset, double cMks,
                             int stage, void *phaseValues,
                             const void *referenceKick, long *badCount,
                             float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceBadCount = NULL, count = 0;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status, checked = stage == 0 || stage == 2;

  if (badCount)
    *badCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || (stage == 1 && !phaseValues) || (stage == 2 && !referenceKick) ||
      (checked && !badCount))
    return static_cast<int>(cudaErrorInvalidValue);
  if (checked) {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceBadCount, sizeof(*deviceBadCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    cudaStatus = cudaMemset(deviceBadCount, 0, sizeof(*deviceBadCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRamprfKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
    stride, pCentral, length, volt, omega, phase, tOffset, cMks, stage,
    static_cast<double *>(phaseValues), static_cast<const double *>(referenceKick),
    deviceBadCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess) || !checked)
    return status;
  cudaStatus = cudaMemcpy(&count, deviceBadCount, sizeof(count), cudaMemcpyDeviceToHost);
  *badCount = static_cast<long>(count);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaTfeedbackKick(void *coord, long nParticles, int stride,
                                     int pickupCoordinate, int longitudinal,
                                     double kick, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || nParticles <= 0 || stride < 6 ||
      (!longitudinal && pickupCoordinate != 0 && pickupCoordinate != 2))
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuTfeedbackKickKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, pickupCoordinate,
    longitudinal, kick);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfdfTrack(void *coord, long nParticles, int stride,
                                const GPU_RFDF_DATA *data,
                                int particleIdIndex, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7 ||
      particleIdIndex < 0 || particleIdIndex >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfdfKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                     nParticles, stride, *data,
                                     particleIdIndex);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* ILMATRIX, statement for statement as trackWithIndividualizedLinearMatrix() in do_tracking.c, so
 * that with the CPU's sines and cosines the result is the CPU's to the last bit.  (--fmad=false: no
 * multiply and add is fused, as in the CPU's code.) */
#define GPU_ILMATRIX_PIx2 6.283185307179586476925287

__device__ __forceinline__ void gpuIlmatrixRotate(double *c, double cosA, double sinA) {
  double x = c[0], xp = c[1], y = c[2], yp = c[3];

  c[0] = x * cosA + y * sinA;
  c[2] = -x * sinA + y * cosA;
  c[1] = xp * cosA + yp * sinA;
  c[3] = -xp * sinA + yp * cosA;
}

/* The CPU's (long) of a double: the smallest long for a value a long can't hold, NaN included */
__device__ __forceinline__ long long gpuIlmatrixLong(double value) {
  if (!(value > -9223372036854775808.0 && value < 9223372036854775808.0))
    return LLONG_MIN;
  return static_cast<long long>(value);
}

/* The particle in the element's frame with its dispersive orbit removed, its betatron amplitudes and
 * its two phase advances */
__device__ __forceinline__ void gpuIlmatrixPrepare(const double *part, const GPU_ILMATRIX_DATA &d,
                                                   double *c, double *deltaPoP, double *A,
                                                   double *tune2pi) {
  double delta;
  int plane;

  for (int i = 0; i < 6; i++)
    c[i] = part[i];
  if (d.rotateIn)
    gpuIlmatrixRotate(c, d.cosIn, d.sinIn);
  delta = c[5];
  c[5] -= delta;
  for (plane = 0; plane < 2; plane++) {
    double t;

    c[2 * plane] -= delta * (d.eta[2 * plane] + delta * d.eta1[2 * plane]);
    c[2 * plane + 1] -= delta * (d.eta[2 * plane + 1] + delta * d.eta1[2 * plane + 1]);
    t = d.alpha[plane] * c[2 * plane] + d.beta[plane] * c[2 * plane + 1];
    A[plane] = (c[2 * plane] * c[2 * plane] + t * t) / d.beta[plane];
  }
  for (plane = 0; plane < 2; plane++) {
    tune2pi[plane] = GPU_ILMATRIX_PIx2 * (d.tune[plane] +
                                          delta * (d.chrom[plane] +
                                                   delta / 2 * (d.chrom2[plane] + delta / 3 * d.chrom3[plane])));
    tune2pi[plane] += GPU_ILMATRIX_PIx2 * (A[0] * (d.tswax[plane] + A[0] * d.tswax2[plane] / 2) +
                                           A[1] * (d.tsway[plane] + A[1] * d.tsway2[plane] / 2) +
                                           A[0] * A[1] * d.tswaxay[plane]);
  }
  *deltaPoP = delta;
}

/* The phase advances alone, for the host to take the sines and cosines of */
__global__ void gpuIlmatrixPhaseKernel(const double *coord, long nParticles, int stride,
                                       GPU_ILMATRIX_DATA d, double *phase) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double c[6], delta, A[2], tune2pi[2];

  if (ip >= nParticles)
    return;
  gpuIlmatrixPrepare(coord + ip * stride, d, c, &delta, A, tune2pi);
  phase[2 * ip] = tune2pi[0];
  phase[2 * ip + 1] = tune2pi[1];
}

/* sinCos: sin and cos of each particle's two phase advances from the host's libm (four numbers a
 * particle), or NULL for the device's own.  survivor[ip] is 1 or 0.  counters[0] counts the lost,
 * counters[1] those the CPU stops the run for (R12 or R34 zero). */
__global__ void gpuIlmatrixTrackKernel(double *coord, long nParticles, int stride,
                                       GPU_ILMATRIX_DATA d, const double *sinCos,
                                       long *survivor, long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double c[6], temp[6], delta, A[2], tune2pi[2], R[6][6], C[6];
  int plane, lost = 0, i, j;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  gpuIlmatrixPrepare(part, d, c, &delta, A, tune2pi);
  for (i = 0; i < 6; i++) {
    C[i] = d.C[i];
    for (j = 0; j < 6; j++)
      R[i][j] = i == j ? 1 : 0;
  }
  for (plane = 0; !lost && plane < 2; plane++) {
    int offset = 2 * plane;
    double eta1, etap1, beta1, alpha1, sin_phi, cos_phi, R11, R22, R12, det;

    eta1 = d.eta[2 * plane] + delta * d.eta1[2 * plane];
    etap1 = d.eta[2 * plane + 1] + delta * d.eta1[2 * plane + 1];
    if ((beta1 = d.beta[plane] + d.beta1[plane] * delta) <= 0) {
      lost = 1;
      continue;
    }
    if (!d.allowResonanceCrossing &&
        gpuIlmatrixLong(2 * tune2pi[plane] / GPU_ILMATRIX_PIx2) != d.tuneTwice[plane]) {
      lost = 1;
      continue;
    }
    if (sinCos) {
      sin_phi = sinCos[4 * ip + 2 * plane];
      cos_phi = sinCos[4 * ip + 2 * plane + 1];
    } else {
      sin_phi = sin(tune2pi[plane]);
      cos_phi = cos(tune2pi[plane]);
    }
    alpha1 = d.alpha[plane] + d.alpha1[plane] * delta;
    R11 = R[0 + offset][0 + offset] = cos_phi + alpha1 * sin_phi;
    R22 = R[1 + offset][1 + offset] = cos_phi - alpha1 * sin_phi;
    R12 = R[0 + offset][1 + offset] = beta1 * sin_phi;
    if (R12 != 0) {
      R[1 + offset][0 + offset] = (R11 * R22 - 1) / R12;
    } else {
      atomicAdd(reinterpret_cast<unsigned long long *>(counters + 1), 1ULL);
      survivor[ip] = 1;
      return;
    }
    R[0 + offset][5] = eta1 - eta1 * cos_phi - (alpha1 * eta1 + beta1 * etap1) * sin_phi;
    R[1 + offset][5] = etap1 - etap1 * cos_phi + (eta1 + alpha1 * alpha1 * eta1 + alpha1 * beta1 * etap1) * sin_phi / beta1;
    R[4][0 + offset] = -etap1 + etap1 * cos_phi + (eta1 + alpha1 * alpha1 * eta1 + alpha1 * beta1 * etap1) * sin_phi / beta1;
    R[4][1 + offset] = eta1 - eta1 * cos_phi + (alpha1 * eta1 + beta1 * etap1) * sin_phi;
    det = R[0 + offset][0 + offset] * R[1 + offset][1 + offset] -
          R[0 + offset][1 + offset] * R[1 + offset][0 + offset];
    if (fabs(det - 1) > 1e-6) {
      lost = 1;
      continue;
    }
  }
  if (lost) {
    /* as the CPU leaves it: in the element's frame, without its dispersive orbit */
    for (i = 0; i < 4; i++)
      part[i] = c[i];
    part[4] = d.z;
    part[5] = d.pCentral * (1 + delta);
    survivor[ip] = 0;
    atomicAdd(reinterpret_cast<unsigned long long *>(counters), 1ULL);
    return;
  }
  C[4] = d.length * (1 + delta * (d.alphac[0] + delta * (d.alphac[1] + delta * d.alphac[2])));
  C[4] += A[0] * (d.dsdA[0] + A[0] * d.dsdA2[0] / 2) +
          A[1] * (d.dsdA[1] + A[1] * d.dsdA2[1] / 2) +
          A[0] * A[1] * d.dsdAxAy;
  /* track_particles() for a first-order matrix */
  for (i = 5; i >= 0; i--) {
    double sum = C[i];

    for (j = 5; j >= 0; j--)
      sum += R[i][j] * c[j];
    temp[i] = sum;
  }
  for (i = 5; i >= 0; i--)
    c[i] = temp[i];
  c[5] += delta;
  for (plane = 0; plane < 2; plane++) {
    c[2 * plane] += delta * (d.eta[2 * plane] + delta * d.eta1[2 * plane]);
    c[2 * plane + 1] += delta * (d.eta[2 * plane + 1] + delta * d.eta1[2 * plane + 1]);
  }
  if (d.rotateOut)
    gpuIlmatrixRotate(c, d.cosOut, d.sinOut);
  for (i = 0; i < 6; i++)
    part[i] = c[i];
  survivor[ip] = 1;
}

extern "C" int gpuCudaIlmatrixPhases(void *coord, void *phase, long nParticles, int stride,
                                     const GPU_ILMATRIX_DATA *data, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !phase || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuIlmatrixPhaseKernel<<<blocks, threads>>>(static_cast<const double *>(coord), nParticles, stride,
                                              *data, static_cast<double *>(phase));
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* Tracks in place and, if any particle is lost, moves the rows to scratchCoord in the CPU's order
 * (the caller then takes scratchCoord as the beam).  prefix: 2*nParticles longs. */
extern "C" int gpuCudaIlmatrixTrack(void *coord, void *scratchCoord, void *prefix, long nParticles,
                                    int stride, const GPU_ILMATRIX_DATA *data, const void *sinCos,
                                    long *remaining, long *divideByZero, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  long *counters = NULL, hostCounters[2] = {0, 0};
  long survivors;
  const int blockSize = 256;
  int gridSize, status;

  if (!remaining || !divideByZero)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  *divideByZero = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || !data || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStatus = gpuReuseDeviceBuffer(0, &counters, 2 * sizeof(*counters));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(counters, 0, 2 * sizeof(*counters));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuIlmatrixTrackKernel<<<gridSize, blockSize>>>(static_cast<double *>(coord), nParticles, stride, *data,
                                                  static_cast<const double *>(sinCos), devicePrefix, counters);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(hostCounters, counters, sizeof(hostCounters), cudaMemcpyDeviceToHost);
  survivors = nParticles - hostCounters[0];
  if (cudaStatus == cudaSuccess && !hostCounters[1] && hostCounters[0] > 0) {
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(devicePrefix, nParticles, survivors);
    if (cudaStatus == cudaSuccess) {
      gpuStableScatterRowsKernel<<<gridSize, blockSize>>>(
        static_cast<double *>(coord), static_cast<double *>(scratchCoord),
        devicePrefix + nParticles, nParticles, stride, survivors);
      cudaStatus = cudaGetLastError();
    }
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess)) {
    *remaining = survivors;
    *divideByZero = hostCounters[1];
  }
  return status;
}

extern "C" int gpuCudaSreffectsTrack(void *coord, long nParticles, int stride,
                                      const GPU_SREFFECTS_DATA *data,
                                      float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSreffectsKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                          nParticles, stride, *data);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaBggexpTrack(void *coord, long nParticles, int stride,
                                  const GPU_BGGEXP_DATA *data,
                                  float *milliseconds) {
  GPU_BGGEXP_DEVICE_DATA deviceData;
  cudaEvent_t start, stop;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = ensureBggexpScratch(data);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  std::memset(&deviceData, 0, sizeof(deviceData));
  deviceData.nz = data->nz;
  deviceData.termCount = data->termCount;
  deviceData.m = gpuBggexpScratch.m;
  deviceData.gradient = gpuBggexpScratch.gradient;
  deviceData.radialPower = gpuBggexpScratch.radialPower;
  deviceData.coefficient = gpuBggexpScratch.coefficient;
  deviceData.multipoleFactor = gpuBggexpScratch.multipoleFactor;
  deviceData.Cmn = gpuBggexpScratch.Cmn;
  deviceData.dCmnDz = gpuBggexpScratch.dCmnDz;
  deviceData.dz = data->dz;
  deviceData.zMin = data->zMin;
  deviceData.zMax = data->zMax;
  deviceData.xCenter = data->xCenter;
  deviceData.yCenter = data->yCenter;
  deviceData.length = data->length;
  deviceData.dxExpansion = data->dxExpansion;
  deviceData.pCentral = data->pCentral;
  deviceData.strength = data->strength;
  deviceData.Bx = data->Bx;
  deviceData.By = data->By;
  std::memcpy(deviceData.BFactor, data->BFactor,
              sizeof(deviceData.BFactor));
  deviceData.particleCharge = data->particleCharge;
  deviceData.particleRelSign = data->particleRelSign;
  deviceData.particleMass = data->particleMass;
  deviceData.cMks = data->cMks;
  if (data->symplectic) {
    if (data->zInterval <= 0 || !(data->step > 0))
      return static_cast<int>(cudaErrorInvalidValue);
    deviceData.potential = gpuBggexpScratch.potential;
    deviceData.isBend = data->isBend;
    deviceData.zInterval = data->zInterval;
    deviceData.step = data->step;
    deviceData.scaleA = data->scaleA;
    deviceData.xEntry = data->xEntry;
    deviceData.zEntry = data->zEntry;
    deviceData.xExit = data->xExit;
    deviceData.zExit = data->zExit;
    deviceData.sinEntry = data->sinEntry;
    deviceData.cosEntry = data->cosEntry;
    deviceData.sinExit = data->sinExit;
    deviceData.cosExit = data->cosExit;
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  if (data->symplectic)
    gpuBggexpSymplecticKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, deviceData);
  else
    gpuBggexpKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                        nParticles, stride, deviceData);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCwigglerTrack(void *coord, long nParticles, int stride,
                                    const GPU_CWIGGLER_DATA *data,
                                    float *milliseconds) {
  GPU_CWIGGLER_DATA deviceData;
  cudaEvent_t start, stop;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7 ||
      data->periods <= 0 || data->stepsPerPeriod <= 0 ||
      data->periodLength <= 0 ||
      (data->synchRad && (data->pCentral <= 0 || data->srCoef <= 0)) ||
      (data->integrationOrder != 2 && data->integrationOrder != 4))
    return static_cast<int>(cudaErrorInvalidValue);
  deviceData = *data;
  if (data->harmonics) {
    status = ensureCwigglerScratch(data);
    if (status != static_cast<int>(cudaSuccess))
      return status;
    deviceData.harmonics = gpuCwigglerScratch.device;
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCwigglerKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                        nParticles, stride, deviceData);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaFtableTrack(void *coord, long nParticles, int stride,
                                  const GPU_FTABLE_DATA *data,
                                  float *milliseconds) {
  GPU_FTABLE_DATA deviceData;
  cudaEvent_t start, stop;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7 ||
      data->nKicks < 1 || data->length <= 0 || data->pCentral <= 0 ||
      data->eomc == 0)
    return static_cast<int>(cudaErrorInvalidValue);
  status = ensureFtableScratch(data);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  deviceData = *data;
  for (long field = 0; field < 3; field++)
    deviceData.field[field] = gpuFtableScratch.field[field];
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuFtableKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                      nParticles, stride, deviceData);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* With data->apertures, particles lost as lorentz() loses them go to the end in its order: rows move
 * into scratchCoord, with each row's place in prefix, and *remaining is the number kept */
extern "C" int gpuCudaBmxyzTrack(void *coord, long nParticles, int stride,
                                  const GPU_BMXYZ_DATA *data,
                                  void *scratchCoord, void *prefix, long *remaining,
                                  long *failedCount, float *milliseconds) {
  static unsigned long long *lostCount = NULL;
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_BMXYZ_DATA deviceData;
  GPU_FTABLE_DATA uploadData;
  cudaError_t cudaStatus;
  int status;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  long coordinateValues;
  unsigned long long hostFailedCount = 0, hostLostCount = 0;
  int *lossRound = NULL;

  if (!coord || !data || !failedCount || nParticles <= 0 || stride < 7 ||
      nParticles > LONG_MAX / stride)
    return static_cast<int>(cudaErrorInvalidValue);
  *failedCount = 0;
  if (remaining)
    *remaining = nParticles;
  if (data->apertures && (!scratchCoord || !prefix || !remaining))
    return static_cast<int>(cudaErrorInvalidValue);
  coordinateValues = nParticles * stride;
  status = ensureBmxyzScratch(coordinateValues);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  std::memset(&uploadData, 0, sizeof(uploadData));
  uploadData.tableOwner = data->tableOwner;
  for (long dimension = 0; dimension < 3; dimension++)
    uploadData.dimensions[dimension] = data->dimensions[dimension];
  for (long field = 0; field < 3; field++)
    uploadData.field[field] = data->field[field];
  status = ensureFtableScratch(&uploadData);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  deviceData = *data;
  for (long field = 0; field < 3; field++)
    deviceData.field[field] = gpuFtableScratch.field[field];
  cudaStatus = cudaMemcpy(gpuBmxyzScratch.coordBackup, coord,
                          coordinateValues * sizeof(double),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(gpuBmxyzScratch.failedCount, 0,
                          sizeof(*gpuBmxyzScratch.failedCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  if (data->apertures) {
    if (!lostCount && (cudaStatus = cudaMalloc(&lostCount, sizeof(*lostCount))) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    if ((cudaStatus = cudaMemset(lostCount, 0, sizeof(*lostCount))) != cudaSuccess ||
        (cudaStatus = gpuRoundLossRounds(nParticles, &lossRound)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  if (milliseconds)
    *milliseconds = 0;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuBmxyzKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                      nParticles, stride, deviceData,
                                      gpuBmxyzScratch.failedCount, lossRound, lostCount);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostFailedCount, gpuBmxyzScratch.failedCount,
                          sizeof(hostFailedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *failedCount = static_cast<long>(hostFailedCount);
  if (hostFailedCount) {
    cudaStatus = cudaMemcpy(coord, gpuBmxyzScratch.coordBackup,
                            coordinateValues * sizeof(double),
                            cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    return static_cast<int>(cudaSuccess);
  }
  if (!data->apertures)
    return static_cast<int>(cudaSuccess);
  if ((cudaStatus = cudaMemcpy(&hostLostCount, lostCount, sizeof(hostLostCount),
                               cudaMemcpyDeviceToHost)) != cudaSuccess)
    return static_cast<int>(cudaStatus);
  if (!hostLostCount)
    return static_cast<int>(cudaSuccess);
  {
    long *devicePrefix = static_cast<long *>(prefix);
    long kept = nParticles - static_cast<long>(hostLostCount);

    if ((cudaStatus = gpuBuildCpuRoundLossOrderDestination(devicePrefix + nParticles, nParticles,
                                                           kept)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuStableScatterRowsKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                                     static_cast<double *>(scratchCoord),
                                                     devicePrefix + nParticles, nParticles,
                                                     stride, kept);
    if ((cudaStatus = cudaGetLastError()) != cudaSuccess ||
        (cudaStatus = gpuPublishCpuLossOrderDestination(devicePrefix, nParticles)) != cudaSuccess)
      return static_cast<int>(cudaStatus);
    *remaining = kept;
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaLorentzTrack(void *coord, long nParticles, int stride,
                                    const GPU_LORENTZ_DATA *data,
                                    long *failedCount, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_LORENTZ_DATA deviceData;
  cudaError_t cudaStatus;
  int status;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  long coordinateValues;
  unsigned long long hostFailedCount = 0;

  if (!coord || !data || !failedCount || nParticles <= 0 || stride < 7 ||
      nParticles > LONG_MAX / stride || data->length <= 0 ||
      data->integrationAccuracy <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *failedCount = 0;
  coordinateValues = nParticles * stride;
  status = ensureLorentzScratch(coordinateValues, data);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  deviceData = *data;
  if (data->type == GPU_LORENTZ_BMAPXY) {
    deviceData.field[0] = gpuLorentzScratch.field[0];
    deviceData.field[1] = gpuLorentzScratch.field[1];
  }
  cudaStatus = cudaMemcpy(gpuLorentzScratch.coordBackup, coord,
                          coordinateValues * sizeof(double),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(gpuLorentzScratch.failedCount, 0,
                          sizeof(*gpuLorentzScratch.failedCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  if (milliseconds)
    *milliseconds = 0;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuLorentzKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                        nParticles, stride, deviceData,
                                        gpuLorentzScratch.failedCount);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostFailedCount, gpuLorentzScratch.failedCount,
                          sizeof(hostFailedCount), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *failedCount = static_cast<long>(hostFailedCount);
  if (hostFailedCount) {
    cudaStatus = cudaMemcpy(coord, gpuLorentzScratch.coordBackup,
                            coordinateValues * sizeof(double),
                            cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  return static_cast<int>(cudaSuccess);
}

/* LSRMDLTR: motion.cc's laser modulator with rk_odeint3_na() (SDDS mdbmth/rkODE.c), one thread a
 * particle. The operations follow the CPU's in order (the build has --fmad=false); sin, cos, cosh,
 * sinh and exp are CUDA's (atan gpuAccurateAtan()'s), so results differ from elegant's by rounding. A particle the CPU would
 * lose (a coordinate not finite, an integration failure, Pz <= 0) makes the whole call fall back.
 * With ISR each particle takes one number a step from its offset in the table and reports how many it
 * took, so that the host can check the offsets against the CPU's order (gpu_track_lsrmdltr()). */
typedef struct GPU_LSRMDLTR_SCRATCH {
  double *coordBackup;
  long coordCapacity;
  double *xRange;
  long xRangeCapacity;
  double *profile;
  long profileCapacity;
  long long *offsets;
  int *counts;
  long offsetCapacity;
  unsigned long long *failedCount;
} GPU_LSRMDLTR_SCRATCH;

static GPU_LSRMDLTR_SCRATCH gpuLsrmdltrScratch;

extern "C" void gpuCudaLsrmdltrRelease(void) {
  gpuFreeDevice(gpuLsrmdltrScratch.coordBackup);
  gpuFreeDevice(gpuLsrmdltrScratch.xRange);
  gpuFreeDevice(gpuLsrmdltrScratch.profile);
  gpuFreeDevice(gpuLsrmdltrScratch.offsets);
  gpuFreeDevice(gpuLsrmdltrScratch.counts);
  gpuFreeDevice(gpuLsrmdltrScratch.failedCount);
  memset(&gpuLsrmdltrScratch, 0, sizeof(gpuLsrmdltrScratch));
}

struct GpuLsrComplex {
  double re, im;
};

__device__ __forceinline__ GpuLsrComplex gpuLsrComplex(double re, double im) {
  GpuLsrComplex c;
  c.re = re;
  c.im = im;
  return c;
}

/* std::complex<double> operations as libstdc++ does them for finite values */
__device__ __forceinline__ GpuLsrComplex gpuLsrMul(GpuLsrComplex a, GpuLsrComplex b) {
  return gpuLsrComplex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ GpuLsrComplex gpuLsrScale(GpuLsrComplex a, double s) {
  return gpuLsrComplex(a.re * s, a.im * s);
}

__device__ __forceinline__ GpuLsrComplex gpuLsrDivide(GpuLsrComplex a, double s) {
  return gpuLsrComplex(a.re / s, a.im / s);
}

__device__ __forceinline__ GpuLsrComplex gpuLsrAdd(GpuLsrComplex a, GpuLsrComplex b) {
  return gpuLsrComplex(a.re + b.re, a.im + b.im);
}

__device__ __forceinline__ GpuLsrComplex gpuLsrSubtract(GpuLsrComplex a, GpuLsrComplex b) {
  return gpuLsrComplex(a.re - b.re, a.im - b.im);
}

/* 1/(c + i d): libgcc's __divdc3 (Smith's method) for finite values */
__device__ __forceinline__ GpuLsrComplex gpuLsrReciprocal(double c, double d) {
  double a = 1, b = 0, ratio, denom;

  if (fabs(c) < fabs(d)) {
    ratio = c / d;
    denom = (c * ratio) + d;
    return gpuLsrComplex(((a * ratio) + b) / denom, ((b * ratio) - a) / denom);
  }
  ratio = d / c;
  denom = (d * ratio) + c;
  return gpuLsrComplex(((b * ratio) + a) / denom, (b - (a * ratio)) / denom);
}

/* glibc's cexp for finite values */
__device__ __forceinline__ GpuLsrComplex gpuLsrExp(GpuLsrComplex z) {
  double s, c, e;

  sincos(z.im, &s, &c);
  e = exp(z.re);
  return gpuLsrComplex(e * c, e * s);
}

__device__ __forceinline__ double gpuLsrHermite(double x, int n) {
  double r;

  switch (n) {
  case 0:
    return 1;
  case 1:
    return 2 * x;
  case 2:
    return 4 * x * x - 2;
  case 3:
    return 8 * (x * x * x) - 12 * x;
  default:
    r = x * x;
    return 16 * (r * r) - 48 * r + 12;
  }
}

__device__ __forceinline__ double gpuLsrHermiteDeriv(double x, int n) {
  switch (n) {
  case 0:
    return 0;
  case 1:
    return 2;
  case 2:
    return 8 * x;
  case 3:
    return 24 * (x * x) - 12;
  default:
    return 64 * (x * x * x) - 96 * x;
  }
}

__device__ __forceinline__ double gpuLsrHermite2ndDeriv(double x, int n) {
  switch (n) {
  case 0:
  case 1:
    return 0;
  case 2:
    return 8;
  case 3:
    return 48 * x;
  default:
    return 3 * 64 * (x * x) - 96;
  }
}

__device__ __forceinline__ double gpuLsrPoleFactor(const GPU_LSRMDLTR_DATA &d, double z) {
  long poleNumber = (long)(z / (d.length / (2 * d.periods)) + 0.5);

  if (poleNumber > 2 && poleNumber < (2 * d.periods - 2))
    return 1;
  if (poleNumber == 0 || poleNumber == 2 * d.periods)
    return d.poleFactor1;
  if (poleNumber == 1 || poleNumber == (2 * d.periods - 1))
    return d.poleFactor2;
  return d.poleFactor3;
}

/* laserModulatorUndulatorField() */
__device__ __forceinline__ void gpuLsrUndulatorField(double *B, const GPU_LSRMDLTR_DATA &d,
                                                     double Bfactor, double kux, double kuy,
                                                     double kuz, double x) {
  double a = d.tguGradient, ku = d.ku, CF = d.tguCompFactor;

  if (d.helical == 0) {
    if (d.fieldCode == 0) {
      B[1] = Bfactor * cos(kuz);
      B[2] = 0;
    } else if (d.fieldCode == 1) {
      B[1] = Bfactor * cos(kuz) * cosh(kuy);
      B[2] = -Bfactor * sin(kuz) * sinh(kuy);
      if (a != 0) {
        B[0] = Bfactor * a / ku * cos(kuz) * sinh(kuy);
        B[1] *= (1 + a * x);
        B[1] += CF * (Bfactor * Bfactor) * a * d.eMks /
                (2 * d.gammaCentral * d.meMks * d.cMks * (ku * ku));
        B[2] = -Bfactor * (1 + a * x) * sin(kuz) * sinh(kuy);
      }
    } else {
      B[1] = Bfactor * cos(kuz) * (1 + (kuy * kuy) / 2);
      B[2] = -Bfactor * sin(kuz) * kuy;
    }
  } else {
    if (d.fieldCode == 0) {
      B[0] = -Bfactor * sin(kuz);
      B[1] = Bfactor * cos(kuz);
      B[2] = 0;
    } else if (d.fieldCode == 1) {
      B[0] = -Bfactor * sin(kuz) * cosh(kux);
      B[1] = Bfactor * cos(kuz) * cosh(kuy);
      B[2] = -Bfactor * sin(kuz) * sinh(kuy) - Bfactor * cos(kuz) * sinh(kux);
    } else {
      B[0] = -Bfactor * sin(kuz) * (1 + (kux * kux) / 2);
      B[1] = Bfactor * cos(kuz) * (1 + (kuy * kuy) / 2);
      B[2] = -Bfactor * sin(kuz) * kuy - Bfactor * cos(kuz) * kux;
    }
  }
}

/* computeLaserField(); lastIndex is the thread's own, where the CPU keeps one for all particles */
__device__ void gpuLsrLaserField(double *Ef, double *Bf, const GPU_LSRMDLTR_DATA &d, double phase,
                                 double x, double y, double dz, double tForProfile,
                                 long *lastIndex) {
  const double sqrt2 = sqrt(2.0);
  double sinTilt, cosTilt, Vx, Vy, w, Hm, Hn, Hmp, Hnp, Hmpp, amplitude = 1, kw, scale;
  double k = d.k, ZR = d.ZRayleigh, w0 = d.laserW0;
  int m = d.laserM, n = d.laserN;
  GpuLsrComplex Q, Ec, I = gpuLsrComplex(0, 1), term, sum;
  long i;

  for (i = 0; i < 3; i++)
    Ef[i] = Bf[i] = 0;
  if (d.laserTilt) {
    sinTilt = sin(d.laserTilt);
    cosTilt = cos(d.laserTilt);
    Vx = x;
    Vy = y;
    x = Vx * cosTilt + Vy * sinTilt;
    y = -Vx * sinTilt + Vy * cosTilt;
  } else {
    cosTilt = 1;
    sinTilt = 0;
  }
  if (d.timeValue) {
    const double *tv = d.timeValue;
    long np = d.tProfilePoints;
    if (tForProfile < tv[0] || tForProfile > tv[np - 1])
      return;
    if ((i = *lastIndex) >= (np - 1))
      i = np - 2;
    if (i < 0)
      i = 0;
    if (tForProfile < tv[i]) {
      do {
        i--;
      } while (i >= 0 && tForProfile < tv[i]);
    } else if (tForProfile > tv[i + 1]) {
      do {
        i++;
      } while (i < (np - 1) && tForProfile > tv[i + 1]);
    }
    if (i < 0 || i >= (np - 1))
      return;
    amplitude = (d.amplitudeValue[i + 1] - d.amplitudeValue[i]) / (tv[i + 1] - tv[i]) *
                  (tForProfile - tv[i]) + d.amplitudeValue[i];
    *lastIndex = i;
  }
  Q = gpuLsrReciprocal(dz, -ZR);
  w = w0 * sqrt(1 + (dz / ZR) * (dz / ZR));
  /* complex(0, phase) - complex(0, (m+n+1)*atan(dz/ZR)) + I*complex(k*Q/2.0*(x*x+y*y)) */
  Ec = gpuLsrSubtract(gpuLsrComplex(0, phase), gpuLsrComplex(0, (m + n + 1) * gpuAccurateAtan(dz / ZR)));
  Ec = gpuLsrAdd(Ec, gpuLsrMul(I, gpuLsrScale(gpuLsrDivide(gpuLsrScale(Q, k), 2.0), x * x + y * y)));
  Ec = gpuLsrScale(gpuLsrExp(Ec), amplitude * d.Ef0Laser * w0 / w);
  Hm = gpuLsrHermite(sqrt2 * x / w, m);
  Hn = gpuLsrHermite(sqrt2 * y / w, n);
  Hmp = gpuLsrHermiteDeriv(sqrt2 * x / w, m);
  Hnp = gpuLsrHermiteDeriv(sqrt2 * y / w, n);
  Hmpp = gpuLsrHermite2ndDeriv(sqrt2 * x / w, m);
  kw = k * w;

  Ef[0] = Hm * Hn * Ec.re;
  Ef[1] = 0;
  /* Ec * (I*sqrt(2)/(k*w)*Hmp*Hn - Q*x*Hm*Hn) */
  term = gpuLsrScale(gpuLsrScale(gpuLsrDivide(gpuLsrScale(I, sqrt2), kw), Hmp), Hn);
  sum = gpuLsrSubtract(term, gpuLsrScale(gpuLsrScale(gpuLsrScale(Q, x), Hm), Hn));
  Ef[2] = gpuLsrMul(Ec, sum).re;
  /* Ec * (2/sqr(k*w)*Hmp*Hnp + I*sqrt(2)*Q/(k*w)*(x*Hm*Hnp + y*Hmp*Hn) - x*y*Q*Q*Hm*Hn) */
  scale = 2 / (kw * kw) * Hmp * Hnp;
  term = gpuLsrScale(gpuLsrDivide(gpuLsrMul(gpuLsrScale(I, sqrt2), Q), kw),
                     x * Hm * Hnp + y * Hmp * Hn);
  sum = gpuLsrComplex(term.re + scale, term.im);
  term = gpuLsrScale(gpuLsrScale(gpuLsrMul(gpuLsrScale(Q, x * y), Q), Hm), Hn);
  sum = gpuLsrSubtract(sum, term);
  Bf[0] = gpuLsrMul(Ec, sum).re / d.cMks;
  /* Ec * (-2/sqr(k*w)*Hmpp*Hn - 2*sqrt(2)*I*Q/(k*w)*x*Hmp*Hn + (Q*Q*sqr(x) - I*Q/k + 1.0)*Hm*Hn) */
  scale = -2 / (kw * kw) * Hmpp * Hn;
  term = gpuLsrScale(gpuLsrScale(gpuLsrScale(gpuLsrDivide(gpuLsrMul(gpuLsrScale(I, 2 * sqrt2), Q), kw),
                                             x), Hmp), Hn);
  sum = gpuLsrComplex(scale - term.re, -term.im);
  term = gpuLsrSubtract(gpuLsrScale(gpuLsrMul(Q, Q), x * x), gpuLsrDivide(gpuLsrMul(I, Q), k));
  term = gpuLsrComplex(term.re + 1.0, term.im);
  sum = gpuLsrAdd(sum, gpuLsrScale(gpuLsrScale(term, Hm), Hn));
  Bf[1] = gpuLsrMul(Ec, sum).re / d.cMks;
  /* Ec * (I*sqrt(2)/(k*w)*Hm*Hnp - Q*y*Hm*Hn) */
  term = gpuLsrScale(gpuLsrScale(gpuLsrDivide(gpuLsrScale(I, sqrt2), kw), Hm), Hnp);
  sum = gpuLsrSubtract(term, gpuLsrScale(gpuLsrScale(gpuLsrScale(Q, y), Hm), Hn));
  Bf[2] = gpuLsrMul(Ec, sum).re / d.cMks;

  if (d.laserTilt) {
    Vx = Ef[0];
    Vy = Ef[1];
    Ef[0] = Vx * cosTilt - Vy * sinTilt;
    Ef[1] = Vx * sinTilt + Vy * cosTilt;
    Vx = Bf[0];
    Vy = Bf[1];
    Bf[0] = Vx * cosTilt - Vy * sinTilt;
    Bf[1] = Vx * sinTilt + Vy * cosTilt;
  }
}

typedef struct GPU_LSR_STATE {
  double xMin, xMax;
  long lastIndex;
  /* ISR numbers: the next one and how many were taken; short when the table ran out */
  long long nextDraw;
  int drawsTaken, drawsShort;
} GPU_LSR_STATE;

/* derivatives_laserModulator() */
__device__ void gpuLsrDerivatives(double *qp, const double *q, double tau,
                                  const GPU_LSRMDLTR_DATA &d, GPU_LSR_STATE *state) {
  const double *P = q + 3;
  double *Pp = qp + 3;
  double gamma, x, y, z, Bfactor, Bscale;
  double B[3] = {0, 0, 0}, E[3] = {0, 0, 0}, Blaser[3];
  int i;

  gamma = sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2] + 1);
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;
  x = q[0] / d.k;
  y = q[1] / d.k;
  z = q[2] / d.k;
  if (x < state->xMin)
    state->xMin = x;
  if (x > state->xMax)
    state->xMax = x;
  Bfactor = gpuLsrPoleFactor(d, z) * d.Bu;
  gpuLsrUndulatorField(B, d, Bfactor, d.ku * x, d.ku * y, d.ku * z, x);
  Bscale = d.Bscale / gamma;
  for (i = 0; i < 3; i++)
    B[i] *= Bscale;
  if (d.laserOn) {
    gpuLsrLaserField(E, Blaser, d, -tau + q[2] - d.Zcenter + d.laserPhase,
                     x - d.laserX0, y - d.laserY0, z - d.laserZ0 - d.Zcenter / d.k,
                     (tau / d.omega - d.t0) - (q[2] - d.Zcenter) / (d.k * d.cMks),
                     &state->lastIndex);
    for (i = 0; i < 3; i++) {
      E[i] *= d.Escale;
      B[i] += Blaser[i] * Bscale;
    }
  }
  Pp[0] = -(E[0] + P[1] * B[2] - P[2] * B[1]);
  Pp[1] = -(E[1] + P[2] * B[0] - P[0] * B[2]);
  Pp[2] = -(E[2] + P[0] * B[1] - P[1] * B[0]);
}

/* stochastic_laserModulator() */
__device__ void gpuLsrSynchrotronRadiation(double *q, double h, const GPU_LSRMDLTR_DATA &d,
                                           GPU_LSR_STATE *state) {
  double *P = q + 3;
  double p, gamma, x, y, z, B2, irho2, delta0, delta1, factor, inner;
  double B[3] = {0, 0, 0};
  int i;

  p = P[0] * P[0] + P[1] * P[1] + P[2] * P[2];
  gamma = sqrt(p + 1);
  p = sqrt(p);
  x = (q[0] - 0.0) / d.k;
  y = q[1] / d.k;
  z = q[2] / d.k;
  h = h / d.k;
  gpuLsrUndulatorField(B, d, gpuLsrPoleFactor(d, z) * d.Bu, d.ku * x, d.ku * y, d.ku * z, x);
  B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
  inner = gamma * d.meMks * d.cMks / d.eMks;
  irho2 = B2 / (inner * inner);
  delta1 = delta0 = (p - d.pCentral) / d.pCentral;
  if (d.synchRad)
    delta1 -= d.radCoef * (1 + delta0) * irho2 * h;
  if (d.isr) {
    double gauss = 0;
    if (state->nextDraw < d.drawCount)
      gauss = d.draws[state->nextDraw];
    else
      state->drawsShort = 1;
    state->nextDraw++;
    state->drawsTaken++;
    delta1 += d.isrCoef * ((1 + delta0) * (1 + delta0)) * pow(irho2, 0.75) * sqrt(h) * gauss;
  }
  factor = (1 + delta1) / (1 + delta0);
  for (i = 0; i < 3; i++)
    P[i] *= factor;
}

/* exit_function(): dZ, and whether a coordinate passed the CPU's 1e300 limits */
__device__ __forceinline__ double gpuLsrExit(const double *q, const GPU_LSRMDLTR_DATA &d,
                                             int *limitHit) {
  double dZ = d.Zend - q[2];

  if (dZ >= 0 && !*limitHit && (fabs(q[0]) > 1e300 || fabs(q[1]) > 1e300))
    *limitHit = 1;
  if (q[5] <= 0)
    return 0.0;
  return dZ;
}

__device__ __forceinline__ int gpuLsrSign(double x) {
  return x < 0 ? -1 : (x > 0 ? 1 : 0);
}

/* rk4_step() */
__device__ void gpuLsrRk4Step(double *yf, double x, const double *yi, const double *dydx, double h,
                              const GPU_LSRMDLTR_DATA &d, GPU_LSR_STATE *state) {
  double k1[6], k2[6], k3[6], yTemp[6], dydxTemp[6], x1;
  int i;

  for (i = 0; i < 6; i++) {
    k1[i] = h * dydx[i];
    yTemp[i] = yi[i] + k1[i] / 2;
  }
  x1 = x + h / 2;
  gpuLsrDerivatives(dydxTemp, yTemp, x1, d, state);
  for (i = 0; i < 6; i++) {
    k2[i] = h * dydxTemp[i];
    yTemp[i] = yi[i] + k2[i] / 2;
  }
  gpuLsrDerivatives(dydxTemp, yTemp, x1, d, state);
  for (i = 0; i < 6; i++) {
    k3[i] = h * dydxTemp[i];
    yTemp[i] = yi[i] + k3[i];
  }
  x1 = x + h;
  gpuLsrDerivatives(dydxTemp, yTemp, x1, d, state);
  for (i = 0; i < 6; i++)
    yf[i] = yi[i] + (k1[i] / 2 + k2[i] + k3[i] + h * dydxTemp[i] / 2) / 3;
}

/* rk_odeint3_na(); returns its DIFFEQ_* code, or -100 when the loop runs past maxLoops (the CPU would
 * not stop) */
__device__ int gpuLsrIntegrate(double *yif, double *x0, double xf, double xAccuracy, double hStep,
                               double exitAccuracy, const GPU_LSRMDLTR_DATA &d,
                               GPU_LSR_STATE *state, int *limitHit) {
  double y0[6], y1[6], y2[6], dydx0[6], dydx1[6], dydx2[6];
  double ex0, ex1, ex2, x1, x2, xdiff;
  long loops;
  int i;

  if (*x0 > xf)
    return 0;
  if (fabs(*x0 - xf) < xAccuracy)
    return 1;
  for (i = 0; i < 6; i++)
    y0[i] = yif[i];
  gpuLsrDerivatives(dydx0, y0, *x0, d, state);
  ex0 = gpuLsrExit(y0, d, limitHit);
  for (loops = 0;; loops++) {
    if (loops > d.maxLoops)
      return -100;
    if (fabs(ex0) < exitAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y0[i];
      return 2;
    }
    if ((xdiff = xf - *x0) < hStep)
      hStep = xdiff;
    x1 = *x0;
    gpuLsrRk4Step(y1, x1, y0, dydx0, hStep, d, state);
    if (d.synchRad || d.isr)
      gpuLsrSynchrotronRadiation(y1, hStep, d, state);
    x1 += hStep;
    gpuLsrDerivatives(dydx1, y1, x1, d, state);
    ex1 = gpuLsrExit(y1, d, limitHit);
    if (gpuLsrSign(ex0) != gpuLsrSign(ex1))
      break;
    if (fabs(xdiff = xf - x1) < xAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y1[i];
      *x0 = x1;
      return 3;
    }
    for (i = 0; i < 6; i++) {
      dydx0[i] = dydx1[i];
      y0[i] = y1[i];
    }
    ex0 = ex1;
    *x0 = x1;
  }
  if (fabs(ex1) < exitAccuracy) {
    for (i = 0; i < 6; i++)
      yif[i] = y1[i];
    *x0 = x1;
    return 2;
  }
  for (loops = 400; loops >= 0; loops--) {
    hStep = -ex0 * (x1 - *x0) / (ex1 - ex0) * 0.995;
    x2 = *x0;
    gpuLsrRk4Step(y2, x2, y0, dydx0, hStep, d, state);
    x2 += hStep;
    gpuLsrDerivatives(dydx2, y2, x2, d, state);
    ex2 = gpuLsrExit(y2, d, limitHit);
    if (fabs(ex2) < exitAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y2[i];
      *x0 = x2;
      return 2;
    }
    if (gpuLsrSign(ex1) == gpuLsrSign(ex2)) {
      for (i = 0; i < 6; i++) {
        y1[i] = y2[i];
        dydx1[i] = dydx2[i];
      }
      x1 = x2;
      ex1 = ex2;
    } else {
      for (i = 0; i < 6; i++) {
        y0[i] = y2[i];
        dydx0[i] = dydx2[i];
      }
      *x0 = x2;
      ex0 = ex2;
    }
  }
  return -4;
}

/* motion()'s loop body for one particle */
__global__ void gpuLsrmdltrKernel(double *coord, long nParticles, int stride, GPU_LSRMDLTR_DATA d,
                                  double *xRange, unsigned long long *failedCount,
                                  const long long *offsets, int *counts) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, q[6], dqdt[6], *P = q + 3;
  double kscale = d.k, Po, gamma, tau, tauLimit, hrec, tolerance = d.tolerance;
  GPU_LSR_STATE state;
  int limitHit = 0, status;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  state.xMin = 1e300;
  state.xMax = -1e300;
  state.lastIndex = 0;
  state.nextDraw = offsets ? offsets[ip] : 0;
  state.drawsTaken = state.drawsShort = 0;
  q[0] = part[0] * kscale;
  q[1] = part[2] * kscale;
  if (fabs(q[0]) > 1e300 || isnan(q[0]) || isinf(q[0]) ||
      fabs(q[1]) > 1e300 || isnan(q[1]) || isinf(q[1])) {
    atomicAdd(failedCount, 1ULL);
    return;
  }
  q[2] = 0.0;
  Po = d.pCentral * (1 + part[5]);
  gamma = sqrt(1 + Po * Po);
  P[2] = Po / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  P[0] = part[1] * P[2];
  P[1] = part[3] * P[2];
  tau = 0 + part[4] * kscale * gamma / Po;
  tauLimit = 1.5 * d.Zend * gamma / P[2] + tau;
  hrec = (tauLimit - tau) / d.nSteps;
  status = gpuLsrIntegrate(q, &tau, tauLimit, (tolerance * tolerance) * (tauLimit - tau), hrec,
                           (tolerance * tolerance) * kscale, d, &state, &limitHit);
  if (counts)
    counts[ip] = state.drawsShort ? -1 : state.drawsTaken;
  if (status < 1) {
    atomicAdd(failedCount, 1ULL);
    return;
  }
  gpuLsrDerivatives(dqdt, q, tau, d, &state);
  Po = sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]);
  gamma = sqrt(1 + Po * Po);
  if (limitHit || isnan(q[0]) || isinf(q[0]) || isnan(q[1]) || isinf(q[1]) ||
      fabs(q[0]) > 1e300 || fabs(q[1]) > 1e300 || P[2] <= 0) {
    atomicAdd(failedCount, 1ULL);
    return;
  }
  part[0] = q[0] / kscale;
  part[2] = q[1] / kscale;
  part[1] = P[0] / P[2];
  part[3] = P[1] / P[2];
  part[5] = (Po - d.pCentral) / d.pCentral;
  part[4] = (tau - 0) * Po / (kscale * gamma);
  if (xRange && ip == 0) {
    xRange[0] = state.xMin;
    xRange[1] = state.xMax;
  }
}

extern "C" int gpuCudaLsrmdltrBackup(const void *coord, long values) {
  cudaError_t cudaStatus;

  if (!coord || values <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuLsrmdltrScratch.coordCapacity < values) {
    gpuFreeDevice(gpuLsrmdltrScratch.coordBackup);
    gpuLsrmdltrScratch.coordBackup = NULL;
    gpuLsrmdltrScratch.coordCapacity = 0;
    cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.coordBackup, values * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuLsrmdltrScratch.coordCapacity = values;
  }
  return static_cast<int>(cudaMemcpy(gpuLsrmdltrScratch.coordBackup, coord, values * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
}

/* values from firstValue on, back from the copy gpuCudaLsrmdltrBackup() made */
extern "C" int gpuCudaLsrmdltrRestore(void *coord, long firstValue, long values) {
  if (!coord || firstValue < 0 || values < 0 || !gpuLsrmdltrScratch.coordBackup ||
      firstValue + values > gpuLsrmdltrScratch.coordCapacity)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!values)
    return static_cast<int>(cudaSuccess);
  return static_cast<int>(cudaMemcpy(static_cast<double *>(coord) + firstValue,
                                     gpuLsrmdltrScratch.coordBackup + firstValue,
                                     values * sizeof(double), cudaMemcpyDeviceToDevice));
}

/* One pass over nParticles rows from coord.  xRange0, if given, gets the first row's smallest and
 * largest x (motion()'s xMotionCenter); offsets (host) and counts (host) are for ISR. */
extern "C" int gpuCudaLsrmdltrPass(void *coord, long nParticles, int stride,
                                   const GPU_LSRMDLTR_DATA *data, const long long *offsets,
                                   int *counts, double *xRange0, long *failedCount,
                                   float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_LSRMDLTR_DATA deviceData;
  cudaError_t cudaStatus;
  unsigned long long hostFailedCount = 0;
  long chunk, first;
  int status, threads = 128;
  float chunkMs;

  if (!coord || !data || !failedCount || nParticles <= 0 || stride < 7 || data->nSteps < 2 ||
      (data->isr && (!offsets || !counts || !data->draws)))
    return static_cast<int>(cudaErrorInvalidValue);
  *failedCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (!gpuLsrmdltrScratch.xRange) {
    cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.xRange, 2 * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  if (!gpuLsrmdltrScratch.failedCount) {
    cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.failedCount, sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  if (offsets && gpuLsrmdltrScratch.offsetCapacity < nParticles) {
    gpuFreeDevice(gpuLsrmdltrScratch.offsets);
    gpuFreeDevice(gpuLsrmdltrScratch.counts);
    gpuLsrmdltrScratch.offsets = NULL;
    gpuLsrmdltrScratch.counts = NULL;
    gpuLsrmdltrScratch.offsetCapacity = 0;
    cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.offsets, nParticles * sizeof(long long));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.counts, nParticles * sizeof(int));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuLsrmdltrScratch.offsetCapacity = nParticles;
  }
  deviceData = *data;
  if (data->timeValue) {
    long points = data->tProfilePoints;
    if (points < 2 || !data->amplitudeValue)
      return static_cast<int>(cudaErrorInvalidValue);
    if (gpuLsrmdltrScratch.profileCapacity < 2 * points) {
      gpuFreeDevice(gpuLsrmdltrScratch.profile);
      gpuLsrmdltrScratch.profile = NULL;
      gpuLsrmdltrScratch.profileCapacity = 0;
      cudaStatus = cudaMalloc(&gpuLsrmdltrScratch.profile, 2 * points * sizeof(double));
      if (cudaStatus != cudaSuccess)
        return static_cast<int>(cudaStatus);
      gpuLsrmdltrScratch.profileCapacity = 2 * points;
    }
    cudaStatus = cudaMemcpy(gpuLsrmdltrScratch.profile, data->timeValue, points * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMemcpy(gpuLsrmdltrScratch.profile + points, data->amplitudeValue,
                              points * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    deviceData.timeValue = gpuLsrmdltrScratch.profile;
    deviceData.amplitudeValue = gpuLsrmdltrScratch.profile + points;
  }
  if (offsets) {
    cudaStatus = cudaMemcpy(gpuLsrmdltrScratch.offsets, offsets, nParticles * sizeof(long long),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemset(gpuLsrmdltrScratch.failedCount, 0, sizeof(unsigned long long));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* launches of at most 32,768 particles, so that one launch stays short at 10,000 steps */
  for (first = 0; first < nParticles; first += chunk) {
    chunk = nParticles - first;
    if (chunk > 32768)
      chunk = 32768;
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuLsrmdltrKernel<<<static_cast<int>((chunk + threads - 1) / threads), threads>>>(
      static_cast<double *>(coord) + first * stride, chunk, stride, deviceData,
      first == 0 && xRange0 ? gpuLsrmdltrScratch.xRange : NULL, gpuLsrmdltrScratch.failedCount,
      offsets ? gpuLsrmdltrScratch.offsets + first : NULL,
      offsets ? gpuLsrmdltrScratch.counts + first : NULL);
    chunkMs = 0;
    status = finishTimedKernel(start, stop, &chunkMs);
    if (status != static_cast<int>(cudaSuccess))
      return status;
    if (milliseconds)
      *milliseconds += chunkMs;
  }
  cudaStatus = cudaMemcpy(&hostFailedCount, gpuLsrmdltrScratch.failedCount, sizeof(hostFailedCount),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *failedCount = static_cast<long>(hostFailedCount);
  if (!hostFailedCount && offsets) {
    cudaStatus = cudaMemcpy(counts, gpuLsrmdltrScratch.counts, nParticles * sizeof(int),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  if (!hostFailedCount && xRange0) {
    cudaStatus = cudaMemcpy(xRange0, gpuLsrmdltrScratch.xRange, 2 * sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  return static_cast<int>(cudaSuccess);
}

/* Each particle's time as modulated_rf_cavity() forms it for its mean: s/(c beta_from_delta(p0, delta)) */
__global__ void gpuRfParticleTimeKernel(const double *coord, long nParticles, int stride, double pCentral,
                                        double cMks, double *value) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double p;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  p = pCentral * (1 + part[5]);
  value[ip] = part[4] / (cMks * (p / sqrt(p * p + 1)));
}

static double *gpuRfParticleTimes = NULL;
static long gpuRfParticleTimeCapacity = 0;

extern "C" int gpuCudaRfParticleTimes(const void *coord, long nParticles, int stride, double pCentral,
                                      double cMks, double *hostValues, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int threads = 256, status;

  if (!coord || !hostValues || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuRfParticleTimeCapacity < nParticles) {
    gpuFreeDevice(gpuRfParticleTimes);
    gpuRfParticleTimes = NULL;
    gpuRfParticleTimeCapacity = 0;
    cudaStatus = cudaMalloc(&gpuRfParticleTimes, nParticles * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuRfParticleTimeCapacity = nParticles;
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfParticleTimeKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<const double *>(coord), nParticles, stride, pCentral, cMks, gpuRfParticleTimes);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(hostValues, gpuRfParticleTimes, nParticles * sizeof(double),
                                     cudaMemcpyDeviceToHost));
}

extern "C" void gpuCudaRfParticleTimesRelease(void) {
  gpuFreeDevice(gpuRfParticleTimes);
  gpuRfParticleTimes = NULL;
  gpuRfParticleTimeCapacity = 0;
}

/* RFTMEZ0, TWLA and MAPSOLENOID: motion()'s particle loop with rk_odeint3_na() (SDDS mdbmth/rkODE.c), one
 * thread a particle, in the CPU's order of operations (the build has --fmad=false). The fields are
 * derivatives_rftmEz0() with computeFields_rftmEz0(), derivatives_tw_linac() with its impulses, and
 * derivatives_mapSolenoid(); rotate3() is mat_mult()'s sum. MAPSOLENOID has no transcendental function;
 * RFTMEZ0's sin and cos and TWLA's sin, cos and exp are CUDA's, so those differ from elegant's by rounding.
 * A particle the CPU loses is marked and its row holds what motion() records for it; the host puts the
 * rows in motion()'s order. A particle that would need the CPU's state from another particle (a loss in the
 * rectangular check after the integration, whose recorded y can be the last particle's), or whose step
 * loop runs on past maxLoops, makes the call go to the CPU. */
typedef struct GPU_MOTION_STATE {
  int limitHit;
  double Xout, Yout, Zout;
} GPU_MOTION_STATE;

static double *gpuMotionTables = NULL;
static long gpuMotionTableCapacity = 0;
static unsigned char *gpuMotionLost = NULL;
static long gpuMotionLostCapacity = 0;
static unsigned long long *gpuMotionCounts = NULL;

static long long *gpuMotionPerm = NULL;
static long gpuMotionPermCapacity = 0;

extern "C" void gpuCudaMotionRelease(void) {
  gpuFreeDevice(gpuMotionTables);
  gpuFreeDevice(gpuMotionPerm);
  gpuMotionPerm = NULL;
  gpuMotionPermCapacity = 0;
  gpuFreeDevice(gpuMotionLost);
  gpuFreeDevice(gpuMotionCounts);
  gpuMotionTables = NULL;
  gpuMotionLost = NULL;
  gpuMotionCounts = NULL;
  gpuMotionTableCapacity = gpuMotionLostCapacity = 0;
}

/* rotate3(): (z, x, y) times the matrix, mat_mult()'s sums */
__device__ __forceinline__ void gpuMotionRotate3(double *data, const double *R) {
  double Q0[3], Q1[3];
  int i, k;

  Q0[0] = data[2];
  Q0[1] = data[0];
  Q0[2] = data[1];
  for (i = 0; i < 3; i++) {
    double c = 0;
    for (k = 0; k < 3; k++)
      c += R[3 * i + k] * Q0[k];
    Q1[i] = c;
  }
  data[2] = Q1[0];
  data[0] = Q1[1];
  data[1] = Q1[2];
}

/* computeFields_rftmEz0() */
__device__ static void gpuMotionRftmEz0Fields(double *E, double *BOverGamma, double *BOverGammaDC,
                                              double q0, double q1, double q2, double tau, double gamma,
                                              const GPU_MOTION_DATA &d) {
  double XYZ[3], R, Zoffset, BphiOverRG, ErOverR, BrOverRG, X, Y, Z, sinPhase, cosPhase, B1, B2;
  long iz, ir;

  XYZ[0] = q0 - d.Xcenter;
  XYZ[1] = q1 - d.Ycenter;
  XYZ[2] = q2 - d.Zcenter;
  gpuMotionRotate3(XYZ, d.partRot);
  X = XYZ[0];
  Y = XYZ[1];
  Z = XYZ[2] + d.Zcenter;
  iz = (long)(Z / d.dZ);
  if (iz < 0 || iz >= d.nz) {
    E[0] = E[1] = E[2] = 0;
    BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
  } else {
    if (iz == (d.nz - 1))
      iz -= 1;
    Zoffset = Z - iz * d.dZ;
    sinPhase = sin(tau);
    cosPhase = cos(tau);
    E[2] = (d.Ez[iz] + (d.Ez[iz + 1] - d.Ez[iz]) / d.dZ * Zoffset) * d.EzPeak;
    if (d.radialOrder) {
      ErOverR = -(d.dEzdZ[iz] + (d.dEzdZ[iz + 1] - d.dEzdZ[iz]) / d.dZ * Zoffset) / 2 * sinPhase * d.EzPeak;
      E[0] = ErOverR * X;
      E[1] = ErOverR * Y;
      BphiOverRG = E[2] / 2 / gamma * cosPhase;
      BOverGamma[0] = -BphiOverRG * Y;
      BOverGamma[1] = BphiOverRG * X;
      BOverGamma[2] = 0;
    } else {
      E[0] = E[1] = 0;
      BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
    }
    E[2] *= sinPhase;
  }
  gpuMotionRotate3(E, d.fieldRot);
  gpuMotionRotate3(BOverGamma, d.fieldRot);

  BOverGammaDC[0] = BOverGammaDC[1] = BOverGammaDC[2] = 0;
  if (d.hasSolenoid) {
    XYZ[0] = q0 - d.Xcenter1;
    XYZ[1] = q1 - d.Ycenter1;
    XYZ[2] = q2 - d.Zcenter1;
    gpuMotionRotate3(XYZ, d.partRot1);
    X = XYZ[0];
    Y = XYZ[1];
    Z = XYZ[2] + d.Zcenter1;
    R = sqrt(X * X + Y * Y);
    iz = (long)((Z - d.Z0Sol) / d.dZSol);
    Zoffset = Z - (iz * d.dZSol + d.Z0Sol);
    if (d.dRSol)
      ir = (long)(R / d.dRSol);
    else
      ir = 0;
    if (iz >= 0 && iz < d.nzSol && (d.nrSol == 1 || ir < d.nrSol)) {
      if (iz == d.nzSol - 1)
        iz -= 1;
      if (ir == d.nrSol - 1)
        ir -= 1;
      if (d.nrSol == 1) {
        BOverGammaDC[2] = (d.BzSol[iz] + (d.BzSol[iz + 1] - d.BzSol[iz]) / d.dZSol * Zoffset) /
                          gamma * d.solenoidFactor;
        BrOverRG = -0.5 * (d.dBzdZSol[iz] + (d.dBzdZSol[iz + 1] - d.dBzdZSol[iz]) / d.dZSol * Zoffset) /
                   gamma * d.solenoidFactor;
        BOverGammaDC[0] = X * BrOverRG;
        BOverGammaDC[1] = Y * BrOverRG;
      } else {
        const double *Br0 = d.BrSol + ir * d.nzSol, *Br1 = Br0 + d.nzSol;
        const double *Bz0 = d.BzSol + ir * d.nzSol, *Bz1 = Bz0 + d.nzSol;
        BrOverRG = 0;
        if (R > 0) {
          B1 = Br0[iz] + (Br1[iz] - Br0[iz]) / d.dRSol * (R - ir * d.dRSol);
          B2 = Br0[iz + 1] + (Br1[iz + 1] - Br0[iz + 1]) / d.dRSol * (R - ir * d.dRSol);
          BrOverRG = (B1 + (B2 - B1) * Zoffset / d.dZSol) / gamma / R * d.solenoidFactor;
          BOverGammaDC[0] = X * BrOverRG;
          BOverGammaDC[1] = Y * BrOverRG;
        }
        B1 = Bz0[iz] + (Bz1[iz] - Bz0[iz]) / d.dRSol * (R - ir * d.dRSol);
        B2 = Bz0[iz + 1] + (Bz1[iz + 1] - Bz0[iz + 1]) / d.dRSol * (R - ir * d.dRSol);
        BOverGammaDC[2] = (B1 + (B2 - B1) * Zoffset / d.dZSol) / gamma * d.solenoidFactor;
      }
    }
    gpuMotionRotate3(BOverGammaDC, d.fieldRot1);
  }
  BOverGammaDC[0] += d.BxStrayScaled / gamma;
  BOverGammaDC[1] += d.ByStrayScaled / gamma;
}

__device__ static void gpuMotionDerivatives(double *qp, const double *q, double tau, const GPU_MOTION_DATA &d) {
  const double *P = q + 3;
  double *Pp = qp + 3, gamma;

  gamma = sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2] + 1);
  qp[0] = P[0] / gamma;
  qp[1] = P[1] / gamma;
  qp[2] = P[2] / gamma;
  if (d.type == GPU_MOTION_RFTMEZ0) {
    /* derivatives_rftmEz0() */
    double E[3], BOverGamma[3], BOverGammaDC[3];
    int i;
    gpuMotionRftmEz0Fields(E, BOverGamma, BOverGammaDC, q[0], q[1], q[2], tau, gamma, d);
    for (i = 0; i < 3; i++)
      BOverGamma[i] += BOverGammaDC[i];
    Pp[0] = -(E[0] + (P[1] * BOverGamma[2] - P[2] * BOverGamma[1]));
    Pp[1] = -(E[1] + (P[2] * BOverGamma[0] - P[0] * BOverGamma[2]));
    Pp[2] = -(E[2] + (P[0] * BOverGamma[1] - P[1] * BOverGamma[0]));
  } else if (d.type == GPU_MOTION_TWLA) {
    /* derivatives_tw_linac() */
    double E[3], B[3], FrP[2], phase, X, Y, Z, cos_phase, droop;
    FrP[0] = FrP[1] = 0;
    droop = 1;
    if ((Z = q[2]) <= d.Zend && Z >= 0) {
      if (d.alphaS)
        droop = exp(-d.alphaS * Z);
      E[2] = d.EzS * sin(phase = Z / d.betaWave - tau) * droop;
      Y = q[1] + d.Yoffset;
      X = q[0] + d.Xoffset;
      E[0] = d.ErS * (cos_phase = cos(phase)) * droop;
      E[1] = E[0] * Y;
      E[0] *= X;
      B[2] = d.BsolS / gamma;
      B[1] = d.BphiS * cos_phase / gamma * droop;
      B[0] = -B[1] * Y;
      B[1] *= X;
      if (d.FrP != 0) {
        FrP[0] = -X * d.FrP * (cos_phase * cos_phase);
        FrP[1] = -Y * d.FrP * (cos_phase * cos_phase);
      }
      Pp[0] = -(E[0] + P[1] * B[2] - P[2] * B[1] - FrP[0]);
      Pp[1] = -(E[1] + P[2] * B[0] - P[0] * B[2] - FrP[1]);
      Pp[2] = -(E[2] + P[0] * B[1] - P[1] * B[0]);
    } else
      Pp[0] = Pp[1] = Pp[2] = 0;
  } else if (d.type == GPU_MOTION_TMCF) {
    /* derivatives_tmcf_mode() */
    double E[3], B[3], phase, sin_phase, cos_phase, sin_tilt, cos_tilt, X, Y, R, Z;
    if ((Z = q[2]) <= d.Zend && Z >= 0) {
      E[2] = d.tmcfEz * (sin_phase = sin(phase = (tau)));
      Y = q[1] + d.Yoffset;
      X = q[0] + d.Xoffset;
      R = sqrt(X * X + Y * Y);
      E[0] = d.tmcfEr * (cos_tilt = (R ? X / R : 0)) * sin_phase;
      E[1] = d.tmcfEr * (sin_tilt = (R ? Y / R : 0)) * sin_phase;
      B[2] = 0;
      B[1] = d.tmcfBphi * (cos_phase = cos(phase)) / gamma;
      B[0] = -B[1] * sin_tilt;
      B[1] *= cos_tilt;
      Pp[0] = -(E[0] - P[2] * B[1]);
      Pp[1] = -(E[1] + P[2] * B[0]);
      Pp[2] = -(E[2] + (P[0] * B[1] - P[1] * B[0]));
    } else
      Pp[0] = Pp[1] = Pp[2] = 0;
  } else if (d.type == GPU_MOTION_CEPL) {
    /* derivatives_ce_plates() */
    double E, z, t_ramp;
    if ((z = q[2]) <= d.Zend && z >= 0) {
      if ((t_ramp = tau) < 1 && t_ramp > 0)
        E = d.Escaled * t_ramp;
      else if (t_ramp <= 0)
        E = 0;
      else
        E = d.Escaled;
      E += d.Estatic;
      Pp[0] = -E * d.cosTilt;
      Pp[1] = -E * d.sinTilt;
      Pp[2] = 0;
    } else
      Pp[0] = Pp[1] = Pp[2] = 0;
  } else if (d.type == GPU_MOTION_TWPL) {
    /* derivatives_tw_plates() */
    double E, z, B, t_ramp;
    if ((z = q[2]) <= d.Zend && z >= 0) {
      if ((t_ramp = tau + z - d.Zend) < 1 && t_ramp > 0)
        E = d.Escaled * t_ramp;
      else if (t_ramp < 0)
        E = 0;
      else
        E = d.Escaled;
      E += d.Estatic;
      B = -E / gamma;
      Pp[0] = -(E - P[2] * B) * d.cosTilt;
      Pp[1] = -(E - P[2] * B) * d.sinTilt;
      Pp[2] = -B * (P[0] * d.cosTilt + P[1] * B * d.sinTilt);
    } else
      Pp[0] = Pp[1] = Pp[2] = 0;
  } else if (d.type == GPU_MOTION_TWMTA) {
    /* derivatives_twmta() */
    double E[3], B[3], phase, X, Y, Z, droop;
    double sin_phase, cos_phase, cosh_KyY, sinh_KyY, cos_KxX, sin_KxX;
    droop = 1;
    if ((Z = q[2]) <= d.Zend && Z >= 0) {
      Y = q[1] + d.Yoffset;
      X = q[0] + d.Xoffset;
      sin_phase = sin(phase = Z / d.betaWave - tau);
      cos_phase = cos(phase);
      cosh_KyY = cosh(d.Ky * Y);
      sinh_KyY = sinh(d.Ky * Y);
      cos_KxX = cos(d.Kx * X);
      sin_KxX = sin(d.Kx * X);
      E[0] = d.ExS * sin_KxX * cosh_KyY * cos_phase;
      E[1] = d.EyS * cos_KxX * sinh_KyY * cos_phase;
      E[2] = d.EzS * cos_KxX * cosh_KyY * sin_phase;
      B[0] = d.BxS / gamma * cos_KxX * sinh_KyY * cos_phase;
      B[1] = d.ByS / gamma * sin_KxX * cosh_KyY * cos_phase;
      B[2] = d.BsolS / gamma;
      if (d.alphaS)
        droop = exp(-d.alphaS * Z);
      Pp[0] = -(E[0] + P[1] * B[2] - P[2] * B[1]) * droop;
      Pp[1] = -(E[1] + P[2] * B[0] - P[0] * B[2]) * droop;
      Pp[2] = -(E[2] + P[0] * B[1] - P[1] * B[0]) * droop;
    } else
      Pp[0] = Pp[1] = Pp[2] = 0;
  } else {
    /* derivatives_mapSolenoid() */
    double BOverGamma[3], XYZ[3], R, Zoffset, BrOverRG, X, Y, Z, B1, B2;
    long iz, ir;
    XYZ[0] = q[0] - d.Xcenter;
    XYZ[1] = q[1] - d.Ycenter;
    XYZ[2] = q[2] - d.Zcenter;
    gpuMotionRotate3(XYZ, d.partRot);
    X = XYZ[0];
    Y = XYZ[1];
    Z = XYZ[2] + d.Zcenter;
    R = sqrt(X * X + Y * Y);
    iz = (long)(Z / d.dzMap);
    Zoffset = Z - iz * d.dzMap;
    ir = (long)(R / d.drMap);
    BOverGamma[0] = BOverGamma[1] = BOverGamma[2] = 0;
    if (iz >= 0 && iz < d.nzMap && ir < d.nrMap) {
      const double *Br0, *Br1, *Bz0, *Bz1;
      if (iz == d.nzMap - 1)
        iz -= 1;
      if (ir == d.nrMap - 1)
        ir -= 1;
      Br0 = d.BrMap + ir * d.nzMap;
      Br1 = Br0 + d.nzMap;
      Bz0 = d.BzMap + ir * d.nzMap;
      Bz1 = Bz0 + d.nzMap;
      BrOverRG = 0;
      if (R > 0) {
        B1 = Br0[iz] + (Br1[iz] - Br0[iz]) / d.drMap * (R - ir * d.drMap);
        B2 = Br0[iz + 1] + (Br1[iz + 1] - Br0[iz + 1]) / d.drMap * (R - ir * d.drMap);
        BrOverRG = (B1 + (B2 - B1) * Zoffset / d.dzMap) / gamma / R * d.factor;
        BOverGamma[0] += X * BrOverRG;
        BOverGamma[1] += Y * BrOverRG;
      }
      B1 = Bz0[iz] + (Bz1[iz] - Bz0[iz]) / d.drMap * (R - ir * d.drMap);
      B2 = Bz0[iz + 1] + (Bz1[iz + 1] - Bz0[iz + 1]) / d.drMap * (R - ir * d.drMap);
      BOverGamma[2] = (B1 + (B2 - B1) * Zoffset / d.dzMap) / gamma * d.factor;
    }
    if (d.hasUniform && (Z > d.zMap0 && Z < d.zMap1)) {
      BOverGamma[0] += d.BxUniformScaled / gamma;
      BOverGamma[1] += d.ByUniformScaled / gamma;
    }
    gpuMotionRotate3(BOverGamma, d.fieldRot);
    Pp[0] = -(P[1] * BOverGamma[2] - P[2] * BOverGamma[1]);
    Pp[1] = -(P[2] * BOverGamma[0] - P[0] * BOverGamma[2]);
    Pp[2] = -(P[0] * BOverGamma[1] - P[1] * BOverGamma[0]);
  }
}

/* input_impulse_tw_linac() (sign -1) and output_impulse_tw_linac() (sign 1); input_impulse_twmta() and
 * output_impulse_twmta() are the same */
__device__ __forceinline__ void gpuMotionTwlaImpulse(double *P, const double *q, double BsolS, int output) {
  double gamma, B[2];

  gamma = sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2] + 1);
  if (BsolS == 0)
    return;
  if (output) {
    B[0] = BsolS * q[0] / (2 * gamma);
    B[1] = BsolS * q[1] / (2 * gamma);
  } else {
    B[0] = -BsolS * q[0] / (2 * gamma);
    B[1] = -BsolS * q[1] / (2 * gamma);
  }
  P[0] += P[2] * B[1];
  P[1] += -P[2] * B[0];
  P[2] += -P[0] * B[1] + P[1] * B[0];
}

/* exit_function() */
__device__ __forceinline__ double gpuMotionExit(const double *q, const GPU_MOTION_DATA &d, GPU_MOTION_STATE *s) {
  double X = 0.0, Y = 0.0, dZ;

  if ((dZ = d.Zend - q[2]) >= 0 && !s->limitHit) {
    if (d.radialLimit) {
      if (d.Xlimit) {
        X = fabs(q[0] - d.XapertureCenter);
        Y = fabs(q[1] - d.YapertureCenter);
        if (d.Xlimit * d.Xlimit < (X * X + Y * Y)) {
          s->Xout = X;
          s->Yout = Y;
          s->Zout = q[2];
          s->limitHit = 1;
        }
      }
    } else {
      if ((d.Xlimit && (X = fabs(q[0] - d.XapertureCenter)) > d.Xlimit) ||
          (d.Ylimit && (Y = fabs(q[1] - d.YapertureCenter)) > d.Ylimit)) {
        s->Xout = X;
        s->Yout = Y;
        s->Zout = q[2];
        s->limitHit = 1;
      }
    }
  }
  if (q[5] <= 0)
    return 0.0;
  return dZ;
}

__device__ static void gpuMotionRk4Step(double *yf, double x, const double *yi, const double *dydx, double h,
                                        const GPU_MOTION_DATA &d) {
  double k1[6], k2[6], k3[6], yTemp[6], dydxTemp[6], x1;
  int i;

  for (i = 0; i < 6; i++) {
    k1[i] = h * dydx[i];
    yTemp[i] = yi[i] + k1[i] / 2;
  }
  x1 = x + h / 2;
  gpuMotionDerivatives(dydxTemp, yTemp, x1, d);
  for (i = 0; i < 6; i++) {
    k2[i] = h * dydxTemp[i];
    yTemp[i] = yi[i] + k2[i] / 2;
  }
  gpuMotionDerivatives(dydxTemp, yTemp, x1, d);
  for (i = 0; i < 6; i++) {
    k3[i] = h * dydxTemp[i];
    yTemp[i] = yi[i] + k3[i];
  }
  x1 = x + h;
  gpuMotionDerivatives(dydxTemp, yTemp, x1, d);
  for (i = 0; i < 6; i++)
    yf[i] = yi[i] + (k1[i] / 2 + k2[i] + k3[i] + h * dydxTemp[i] / 2) / 3;
}

/* rk_odeint3_na() with no stochastic function; its DIFFEQ_* code, or -100 past maxLoops */
__device__ static int gpuMotionIntegrate(double *yif, double *x0, double xf, double xAccuracy, double hStep,
                                         double exitAccuracy, const GPU_MOTION_DATA &d, GPU_MOTION_STATE *s) {
  double y0[6], y1[6], y2[6], dydx0[6], dydx1[6], dydx2[6];
  double ex0, ex1, ex2, x1, x2, xdiff;
  long loops;
  int i;

  if (*x0 > xf)
    return 0;
  if (fabs(*x0 - xf) < xAccuracy)
    return 1;
  for (i = 0; i < 6; i++)
    y0[i] = yif[i];
  gpuMotionDerivatives(dydx0, y0, *x0, d);
  ex0 = gpuMotionExit(y0, d, s);
  for (loops = 0;; loops++) {
    if (loops > d.maxLoops)
      return -100;
    if (fabs(ex0) < exitAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y0[i];
      return 2;
    }
    if ((xdiff = xf - *x0) < hStep)
      hStep = xdiff;
    x1 = *x0;
    gpuMotionRk4Step(y1, x1, y0, dydx0, hStep, d);
    x1 += hStep;
    gpuMotionDerivatives(dydx1, y1, x1, d);
    ex1 = gpuMotionExit(y1, d, s);
    if (gpuLsrSign(ex0) != gpuLsrSign(ex1))
      break;
    if (fabs(xdiff = xf - x1) < xAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y1[i];
      *x0 = x1;
      return 3;
    }
    for (i = 0; i < 6; i++) {
      dydx0[i] = dydx1[i];
      y0[i] = y1[i];
    }
    ex0 = ex1;
    *x0 = x1;
  }
  if (fabs(ex1) < exitAccuracy) {
    for (i = 0; i < 6; i++)
      yif[i] = y1[i];
    *x0 = x1;
    return 2;
  }
  for (loops = 400; loops >= 0; loops--) {
    hStep = -ex0 * (x1 - *x0) / (ex1 - ex0) * 0.995;
    x2 = *x0;
    gpuMotionRk4Step(y2, x2, y0, dydx0, hStep, d);
    x2 += hStep;
    gpuMotionDerivatives(dydx2, y2, x2, d);
    ex2 = gpuMotionExit(y2, d, s);
    if (fabs(ex2) < exitAccuracy) {
      for (i = 0; i < 6; i++)
        yif[i] = y2[i];
      *x0 = x2;
      return 2;
    }
    if (gpuLsrSign(ex1) == gpuLsrSign(ex2)) {
      for (i = 0; i < 6; i++) {
        y1[i] = y2[i];
        dydx1[i] = dydx2[i];
      }
      x1 = x2;
      ex1 = ex2;
    } else {
      for (i = 0; i < 6; i++) {
        y0[i] = y2[i];
        dydx0[i] = dydx2[i];
      }
      *x0 = x2;
      ex0 = ex2;
    }
  }
  return -4;
}

/* motion()'s loop body for one particle; counts[0] the particles lost, counts[1] those that need the CPU */
__global__ void gpuMotionKernel(double *coord, long nParticles, int stride, GPU_MOTION_DATA d,
                                unsigned char *lost, unsigned long long *counts) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, q[6], *P = q + 3, kscale = d.kscale, Po, gamma, tau, tauLimit, hrec;
  double tolerance = d.tolerance;
  GPU_MOTION_STATE s;
  int status;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  lost[ip] = 0;
  q[0] = part[0] * kscale;
  q[1] = part[2] * kscale;
  if ((d.Xlimit && fabs(q[0] - d.XapertureCenter) > d.Xlimit) || isnan(q[0]) || isinf(q[0]) ||
      (d.Ylimit && fabs(q[1] - d.YapertureCenter) > d.Ylimit) || isnan(q[1]) || isinf(q[1])) {
    part[4] = d.zStart;
    part[5] = d.pCentral * (1 + part[5]);
    lost[ip] = 1;
    atomicAdd(counts, 1ULL);
    return;
  }
  q[2] = 0.0;
  Po = d.pCentral * (1 + part[5]);
  gamma = sqrt(1 + Po * Po);
  P[2] = Po / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  P[0] = part[1] * P[2];
  P[1] = part[3] * P[2];
  s.Xout = s.Yout = s.Zout = 0;
  tauLimit = 1.5 * d.Zend * gamma / P[2] + (tau = d.tauStart + part[4] * kscale * gamma / Po);
  hrec = (tauLimit - tau) / d.nSteps;
  s.limitHit = 0;
  if (d.type == GPU_MOTION_TWLA || d.type == GPU_MOTION_TWMTA)
    gpuMotionTwlaImpulse(P, q, d.BsolS, 0);
  status = gpuMotionIntegrate(q, &tau, tauLimit, (tolerance * tolerance) * (tauLimit - tau), hrec,
                              (tolerance * tolerance) * kscale, d, &s);
  if (status == -100) {
    atomicAdd(counts + 1, 1ULL);
    return;
  }
  if (status < 1) {
    /* integration failure */
    part[4] = d.zStart;
    part[5] = sqrt((d.pCentral * (1 + part[5])) * (d.pCentral * (1 + part[5])) + 1);
    lost[ip] = 1;
    atomicAdd(counts, 1ULL);
    return;
  }
  Po = sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]);
  gamma = sqrt(1 + Po * Po);
  if (!s.limitHit) {
    if (isnan(q[0]) || isinf(q[0]) || isnan(q[1]) || isinf(q[1])) {
      s.Xout = s.Yout = 1.7976931348623157e308;
      s.Zout = q[2];
      s.limitHit = 1;
    } else if (d.radialLimit) {
      if (d.Xlimit) {
        double X = fabs(q[0] - d.XapertureCenter), Y = fabs(q[1] - d.YapertureCenter);
        if (d.Xlimit * d.Xlimit < (X * X + Y * Y)) {
          s.Xout = X;
          s.Yout = Y;
          s.Zout = q[2];
          s.limitHit = 1;
        }
      }
    } else if ((d.Xlimit && fabs(q[0] - d.XapertureCenter) > d.Xlimit) ||
               (d.Ylimit && fabs(q[1] - d.YapertureCenter) > d.Ylimit)) {
      /* motion() records the last particle's Y here when only X is outside */
      atomicAdd(counts + 1, 1ULL);
      return;
    }
  }
  if (s.limitHit || P[2] <= 0) {
    part[0] = (s.Xout + d.XapertureCenter) / kscale;
    part[2] = (s.Yout + d.YapertureCenter) / kscale;
    part[4] = s.Zout / kscale + d.zStart;
    part[5] = Po;
    lost[ip] = 1;
    atomicAdd(counts, 1ULL);
    return;
  }
  if (d.type == GPU_MOTION_TWLA || d.type == GPU_MOTION_TWMTA)
    gpuMotionTwlaImpulse(P, q, d.BsolS, 1);
  part[0] = q[0] / kscale;
  part[2] = q[1] / kscale;
  part[1] = P[0] / P[2];
  part[3] = P[1] / P[2];
  part[5] = (Po - d.pCentral) / d.pCentral;
  part[4] = (tau - d.tauStart) * Po / (kscale * gamma);
}

/* One call: the tables to the device (tables[] holds every array the type uses, back to back, and data's
 * pointers are offsets into it, cast), then the particles. lostHost gets the flags when any is lost. */
extern "C" int gpuCudaMotionPass(void *coord, long nParticles, int stride, const GPU_MOTION_DATA *data,
                                 const double *tables, long tableValues, unsigned char *lostHost,
                                 long *lostCount, long *cpuCount, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_MOTION_DATA deviceData;
  cudaError_t cudaStatus;
  unsigned long long hostCounts[2] = {0, 0};
  long chunk, first;
  int status, threads = 128;
  float chunkMs;

  if (!coord || !data || !lostCount || !cpuCount || nParticles <= 0 || stride < 7 || data->nSteps < 2 ||
      tableValues < 0 || (tableValues && !tables))
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = *cpuCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (gpuMotionTableCapacity < tableValues) {
    gpuFreeDevice(gpuMotionTables);
    gpuMotionTables = NULL;
    gpuMotionTableCapacity = 0;
    cudaStatus = cudaMalloc(&gpuMotionTables, tableValues * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMotionTableCapacity = tableValues;
  }
  if (gpuMotionLostCapacity < nParticles) {
    gpuFreeDevice(gpuMotionLost);
    gpuMotionLost = NULL;
    gpuMotionLostCapacity = 0;
    cudaStatus = cudaMalloc(&gpuMotionLost, nParticles);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMotionLostCapacity = nParticles;
  }
  if (!gpuMotionCounts) {
    cudaStatus = cudaMalloc(&gpuMotionCounts, 2 * sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  if (tableValues) {
    cudaStatus = cudaMemcpy(gpuMotionTables, tables, tableValues * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  deviceData = *data;
#define GPU_MOTION_TABLE(p) ((p) ? gpuMotionTables + (long)(size_t)(p) - 1 : NULL)
  deviceData.Ez = GPU_MOTION_TABLE(data->Ez);
  deviceData.dEzdZ = GPU_MOTION_TABLE(data->dEzdZ);
  deviceData.BzSol = GPU_MOTION_TABLE(data->BzSol);
  deviceData.BrSol = GPU_MOTION_TABLE(data->BrSol);
  deviceData.dBzdZSol = GPU_MOTION_TABLE(data->dBzdZSol);
  deviceData.BzMap = GPU_MOTION_TABLE(data->BzMap);
  deviceData.BrMap = GPU_MOTION_TABLE(data->BrMap);
#undef GPU_MOTION_TABLE
  cudaStatus = cudaMemset(gpuMotionCounts, 0, 2 * sizeof(unsigned long long));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* launches of at most 32,768 particles, so that one launch stays short at many steps */
  for (first = 0; first < nParticles; first += chunk) {
    chunk = nParticles - first;
    if (chunk > 32768)
      chunk = 32768;
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMotionKernel<<<static_cast<int>((chunk + threads - 1) / threads), threads>>>(
      static_cast<double *>(coord) + first * stride, chunk, stride, deviceData, gpuMotionLost + first,
      gpuMotionCounts);
    chunkMs = 0;
    status = finishTimedKernel(start, stop, &chunkMs);
    if (status != static_cast<int>(cudaSuccess))
      return status;
    if (milliseconds)
      *milliseconds += chunkMs;
  }
  cudaStatus = cudaMemcpy(hostCounts, gpuMotionCounts, sizeof(hostCounts), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostCounts[0]);
  *cpuCount = static_cast<long>(hostCounts[1]);
  if (hostCounts[0] && !hostCounts[1] && lostHost) {
    cudaStatus = cudaMemcpy(lostHost, gpuMotionLost, nParticles, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  return static_cast<int>(cudaSuccess);
}

__global__ void gpuMotionGatherKernel(double *coord, const double *source, const long long *perm,
                                      long nParticles, int stride) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *from;
  double *to;
  int ic;

  if (ip >= nParticles)
    return;
  from = source + perm[ip] * stride;
  to = coord + ip * stride;
  for (ic = 0; ic < stride; ic++)
    to[ic] = from[ic];
}

/* Rows in motion()'s order after losses: row i becomes the tracked row perm[i] (host).  Uses the copy that
 * gpuCudaLsrmdltrBackup() made, which is no longer needed. */
extern "C" int gpuCudaMotionPermute(void *coord, long nParticles, int stride, const long long *permHost) {
  cudaError_t cudaStatus;
  int threads = 256;

  if (!coord || !permHost || nParticles <= 0 || !gpuLsrmdltrScratch.coordBackup ||
      gpuLsrmdltrScratch.coordCapacity < nParticles * stride)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuMotionPermCapacity < nParticles) {
    gpuFreeDevice(gpuMotionPerm);
    gpuMotionPerm = NULL;
    gpuMotionPermCapacity = 0;
    cudaStatus = cudaMalloc(&gpuMotionPerm, nParticles * sizeof(long long));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMotionPermCapacity = nParticles;
  }
  cudaStatus = cudaMemcpy(gpuMotionPerm, permHost, nParticles * sizeof(long long), cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(gpuLsrmdltrScratch.coordBackup, coord, nParticles * stride * sizeof(double),
                            cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuMotionGatherKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), gpuLsrmdltrScratch.coordBackup, gpuMotionPerm, nParticles, stride);
  return static_cast<int>(cudaGetLastError());
}

/* BEAMBEAM: applyBeamBeamKicks() (do_tracking.c), one thread a particle, in the CPU's order of
 * operations; gaussianBeamKick() and roundGaussianBeamKick(), or ellipsoidalBeamKick() (ionEffects.cc).
 * Faddeeva::w() is gpuFaddeevaW(), and exp, sin, cos, atan2 and hypot are CUDA's, so results differ from
 * elegant's by rounding. std::complex and libgcc's complex division are written out for finite values. */
__device__ __forceinline__ GpuFaddeevaComplex gpuBbMul(GpuFaddeevaComplex a, GpuFaddeevaComplex b) {
  return gpuFaddeevaC(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

/* __divdc3 (Smith's method) for finite values */
__device__ __forceinline__ GpuFaddeevaComplex gpuBbDivide(GpuFaddeevaComplex num, GpuFaddeevaComplex den) {
  double a = num.re, b = num.im, c = den.re, d = den.im, ratio, denom;

  if (fabs(c) < fabs(d)) {
    ratio = c / d;
    denom = (c * ratio) + d;
    return gpuFaddeevaC(((a * ratio) + b) / denom, ((b * ratio) - a) / denom);
  }
  ratio = d / c;
  denom = (d * ratio) + c;
  return gpuFaddeevaC(((b * ratio) + a) / denom, (b - (a * ratio)) / denom);
}

/* glibc's csqrt for finite values */
__device__ static GpuFaddeevaComplex gpuBbSqrt(GpuFaddeevaComplex x) {
  const double dblMax = 1.7976931348623157e308, dblMin = 2.2250738585072014e-308;
  double re = x.re, im = x.im, d, r, s;
  int scale = 0;

  if (im == 0) {
    if (re < 0)
      return gpuFaddeevaC(0, copysign(sqrt(-re), im));
    return gpuFaddeevaC(fabs(sqrt(re)), copysign(0.0, im));
  }
  if (re == 0) {
    if (fabs(im) >= 2 * dblMin)
      r = sqrt(0.5 * fabs(im));
    else
      r = 0.5 * sqrt(2 * fabs(im));
    return gpuFaddeevaC(r, copysign(r, im));
  }
  if (fabs(re) > dblMax / 4) {
    scale = 1;
    re = scalbn(re, -2);
    im = scalbn(im, -2);
  } else if (fabs(im) > dblMax / 4) {
    scale = 1;
    if (fabs(re) >= 4 * dblMin)
      re = scalbn(re, -2);
    else
      re = 0;
    im = scalbn(im, -2);
  } else if (fabs(re) < 2 * dblMin && fabs(im) < 2 * dblMin) {
    scale = -((53 + 1) / 2);
    re = scalbn(re, -2 * scale);
    im = scalbn(im, -2 * scale);
  }
  d = hypot(re, im);
  if (re > 0) {
    r = sqrt(0.5 * (d + re));
    if (scale == 1 && fabs(im) < 1) {
      s = im / r;
      r = scalbn(r, scale);
      scale = 0;
    } else
      s = 0.5 * (im / r);
  } else {
    s = sqrt(0.5 * (d - re));
    if (scale == 1 && fabs(im) < 1) {
      r = fabs(im / s);
      s = scalbn(s, scale);
      scale = 0;
    } else
      r = fabs(0.5 * (im / s));
  }
  if (scale) {
    r = scalbn(r, scale);
    s = scalbn(s, scale);
  }
  return gpuFaddeevaC(r, copysign(s, im));
}

/* gaussianBeamKick(coord, centroid, size, 0, kick, charge, particleMass, ionCharge) */
__device__ static void gpuBbGaussianKick(const double *coord, const GPU_BEAMBEAM_DATA &d, double *kick) {
  double sx, sy, x, y, sd, Fx, Fy, C2, C3, ay, C12;
  GpuFaddeevaComplex w1, w2, erf1, erf2, Fc;
  int swapXY;

  kick[0] = 0;
  kick[1] = 0;
  sx = d.size[0];
  sy = d.size[1];
  if (fabs(sx - sy) / sx < 1e-9) {
    /* roundGaussianBeamKick() */
    double sig, r, dp, theta;
    sig = (sx + sy) / 2;
    x = coord[0] - d.centroid[0];
    y = coord[2] - d.centroid[1];
    r = sqrt(x * x + y * y);
    dp = d.C1round / r * (1 - exp(-(r * r) / (2 * (sig * sig))));
    theta = atan2(y, x);
    kick[0] = -dp * cos(theta) / d.mass;
    kick[1] = -dp * sin(theta) / d.mass;
    return;
  }
  x = coord[0] - d.centroid[0];
  y = coord[2] - d.centroid[1];
  swapXY = 0;
  if (sx < sy) {
    double tmp;
    swapXY = 1;
    tmp = sx;
    sx = sy;
    sy = tmp;
    tmp = x;
    x = y;
    y = -tmp;
  }
  ay = fabs(y);
  sd = sqrt(2.0 * (sx * sx - sy * sy));
  w1 = gpuFaddeevaC(x / sd, ay / sd);
  w2 = gpuFaddeevaC(x / sd * sy / sx, ay / sd * sx / sy);
  C2 = sqrt(d.twoPi / (sx * sx - sy * sy));
  C3 = exp(-(x * x) / (2 * (sx * sx)) - (y * y) / (2 * (sy * sy)));
  erf1 = gpuFaddeevaW(w1);
  erf2 = gpuFaddeevaW(w2);
  /* C1 * C2 * (erf1 - C3 * erf2) */
  C12 = d.C1 * C2;
  Fc = gpuFaddeevaC(C12 * (erf1.re - C3 * erf2.re), C12 * (erf1.im - C3 * erf2.im));
  Fx = Fc.im;
  if (y > 0)
    Fy = Fc.re;
  else
    Fy = -Fc.re;
  if (swapXY) {
    double tmp = Fx;
    Fx = -Fy;
    Fy = tmp;
  }
  kick[0] = -Fx * d.inverseMass;
  kick[1] = -Fy * d.inverseMass;
}

/* ellipsoidalBeamKick(coord, P0, particleMass, pCharge, centroid, size, charge, parabolic) */
__device__ static void gpuBbEllipsoidalKick(double *coord, const GPU_BEAMBEAM_DATA &dd, int parabolic) {
  double x, y, a, b, ELx, ELy, xsign = 1, ysign = 1;
  double qx, qy, qz, delta, denom, p, beta, betaz, dpx, dpy;
  GpuFaddeevaComplex EL, zbar, c1, c2, xi, wbar, t;

  x = coord[0] - dd.centroid[0];
  y = coord[2] - dd.centroid[1];
  a = dd.size[0];
  b = dd.size[1];
  if (((x / a) * (x / a) + (y / b) * (y / b)) > 1) {
    if (x < 0)
      x *= (xsign = -1);
    if (y < 0)
      y *= (ysign = -1);
    zbar = gpuFaddeevaC(x, -y);
    /* sqrt(zbar * zbar - a * a + b * b) */
    t = gpuBbMul(zbar, zbar);
    t = gpuFaddeevaC((t.re - a * a) + b * b, t.im);
    c1 = gpuBbSqrt(t);
    if (!parabolic) {
      /* 4 * charge / (zbar + c1) / (4 * PI * epsilon_o) */
      EL = gpuBbDivide(gpuFaddeevaC(dd.fourCharge, 0.0), gpuFaddeevaC(zbar.re + c1.re, zbar.im + c1.im));
      EL = gpuFaddeevaC(EL.re / dd.fourPiEpsilon, EL.im / dd.fourPiEpsilon);
    } else {
      /* (8 * charge / 3) * (zbar + 2.0 * c1) / (c2 * c2) / (4 * PI * epsilon_o) */
      c2 = gpuFaddeevaC(zbar.re + c1.re, zbar.im + c1.im);
      t = gpuFaddeevaC(zbar.re + c1.re * 2.0, zbar.im + c1.im * 2.0);
      t = gpuFaddeevaC(t.re * dd.eightChargeThirds, t.im * dd.eightChargeThirds);
      EL = gpuBbDivide(t, gpuBbMul(c2, c2));
      EL = gpuFaddeevaC(EL.re / dd.fourPiEpsilon, EL.im / dd.fourPiEpsilon);
    }
    ELx = xsign * EL.re;
    ELy = ysign * EL.im;
  } else {
    if (!parabolic) {
      ELx = dd.fourCharge / (a + b) * (x / a) / dd.fourPiEpsilon;
      ELy = dd.fourCharge / (a + b) * (y / b) / dd.fourPiEpsilon;
    } else {
      if (x < 0)
        x *= (xsign = -1);
      if (y < 0)
        y *= (ysign = -1);
      zbar = gpuFaddeevaC(x, -y);
      wbar = gpuFaddeevaC(b * x / a, -a * y / b);
      xi = gpuFaddeevaC(x / a, y / b);
      /* 8 * charge * xi / (a + b) * (1.0 - ((2.0 * zbar + wbar) * xi) / (3.0 * (a + b))) / (4 * PI * epsilon_o) */
      t = gpuFaddeevaC(zbar.re * 2.0 + wbar.re, zbar.im * 2.0 + wbar.im);
      t = gpuBbMul(t, xi);
      t = gpuFaddeevaC(t.re / (3.0 * (a + b)), t.im / (3.0 * (a + b)));
      t = gpuFaddeevaC(1.0 - t.re, 0.0 - t.im);
      EL = gpuFaddeevaC(xi.re * dd.eightCharge, xi.im * dd.eightCharge);
      EL = gpuFaddeevaC(EL.re / (a + b), EL.im / (a + b));
      EL = gpuBbMul(EL, t);
      EL = gpuFaddeevaC(EL.re / dd.fourPiEpsilon, EL.im / dd.fourPiEpsilon);
      ELx = xsign * EL.re;
      ELy = ysign * EL.im;
    }
  }
  delta = coord[5];
  p = dd.P0 * (1 + delta);
  beta = p / sqrt(p * p + 1);
  betaz = beta / sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  dpx = ELx * dd.pCharge / (betaz * dd.cMks);
  dpy = ELy * dd.pCharge / (betaz * dd.cMks);
  denom = sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  delta = coord[5];
  qx = coord[1] * (1 + delta) / denom;
  qy = coord[3] * (1 + delta) / denom;
  qz = sqrt((1 + delta) * (1 + delta) - qx * qx - qy * qy);
  qx += dpx / dd.pMassCP0;
  qy += dpy / dd.pMassCP0;
  delta = sqrt(qx * qx + qy * qy + qz * qz) - 1;
  coord[1] = qx / qz;
  coord[3] = qy / qz;
  coord[5] = delta;
}

__global__ void gpuBeamBeamKernel(double *coord, long nParticles, int stride, GPU_BEAMBEAM_DATA d) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, kick[2], qx, qy, qz, delta, denom;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (d.distribution != 0) {
    gpuBbEllipsoidalKick(part, d, d.distribution == 2);
    return;
  }
  denom = sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  delta = part[5];
  qx = part[1] * (1 + delta) / denom;
  qy = part[3] * (1 + delta) / denom;
  qz = sqrt((1 + delta) * (1 + delta) - qx * qx - qy * qy);
  gpuBbGaussianKick(part, d, kick);
  qx += kick[0] / d.P0c;
  qy += kick[1] / d.P0c;
  delta = sqrt(qx * qx + qy * qy + qz * qz) - 1;
  part[1] = qx / qz;
  part[3] = qy / qz;
  part[5] = delta;
}

extern "C" int gpuCudaBeamBeamKick(void *coord, long nParticles, int stride, const GPU_BEAMBEAM_DATA *data,
                                   float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  int status, threads = 128;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = static_cast<int>(cudaEventRecord(start, 0));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuBeamBeamKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, *data);
  return finishTimedKernel(start, stop, milliseconds);
}

/* LRWAKE: each particle's time (computeTimeCoordinatesOnly()'s expression), kept for the kick, and t, x, y and
 * the bunch column packed for the host, which adds them per bunch in particle order; then the kicks of
 * track_through_lrwake()'s last loop, add_to_particle_energy() included, in the CPU's order of operations. */
static double *gpuLrwakeTime = NULL, *gpuLrwakePacked = NULL, *gpuLrwakeVoltage = NULL;
static long gpuLrwakeCapacity = 0, gpuLrwakeVoltageCapacity = 0;

extern "C" void gpuCudaLrwakeRelease(void) {
  gpuFreeDevice(gpuLrwakeTime);
  gpuFreeDevice(gpuLrwakePacked);
  gpuFreeDevice(gpuLrwakeVoltage);
  gpuLrwakeTime = gpuLrwakePacked = gpuLrwakeVoltage = NULL;
  gpuLrwakeCapacity = gpuLrwakeVoltageCapacity = 0;
}

__global__ void gpuLrwakeValuesKernel(const double *coord, long nParticles, int stride, double pCentral,
                                      double cMks, int bunchColumn, double *time, double *packed) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double t, p;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  p = pCentral * (part[5] + 1);
  t = part[4] * sqrt(p * p + 1) / (cMks * p);
  time[ip] = t;
  packed[4 * ip] = t;
  packed[4 * ip + 1] = part[0];
  packed[4 * ip + 2] = part[2];
  packed[4 * ip + 3] = bunchColumn >= 0 ? part[bunchColumn] : 0;
}

extern "C" int gpuCudaLrwakeValues(const void *coord, long nParticles, int stride, double pCentral,
                                   double cMks, int bunchColumn, double *hostPacked,
                                   float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int status, threads = 256;

  if (!coord || nParticles <= 0 || stride < 7 || !hostPacked || bunchColumn >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuLrwakeCapacity < nParticles) {
    gpuFreeDevice(gpuLrwakeTime);
    gpuFreeDevice(gpuLrwakePacked);
    gpuLrwakeTime = gpuLrwakePacked = NULL;
    gpuLrwakeCapacity = 0;
    cudaStatus = cudaMalloc(&gpuLrwakeTime, nParticles * sizeof(double));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&gpuLrwakePacked, 4 * nParticles * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuLrwakeCapacity = nParticles;
  }
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuLrwakeValuesKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<const double *>(coord), nParticles, stride, pCentral, cMks, bunchColumn, gpuLrwakeTime,
    gpuLrwakePacked);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(hostPacked, gpuLrwakePacked, 4 * nParticles * sizeof(double),
                                     cudaMemcpyDeviceToHost));
}

__global__ void gpuLrwakeKickKernel(double *coord, long nParticles, int stride, const double *time,
                                    const double *voltage, long nBunches, long ibMin, int bunchColumn,
                                    double factor, double pCentral, double massMV, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x, ib;
  double *part, dgam, P, gamma, gamma1, P1, Pz, Pz1, PRatio, pz;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  ib = bunchColumn >= 0 && nBunches > 1 ? (long)(part[bunchColumn] - ibMin) : 0;
  if (ib < 0 || ib >= nBunches)
    return;
  dgam = voltage[2 * nBunches + ib] / (1e6 * massMV) * factor;
  /* add_to_particle_energy(part, time, P0, -dgam) */
  P = pCentral * (1 + part[5]);
  gamma1 = (gamma = sqrt(P * P + 1)) + -dgam;
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  P1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (P1 - pCentral) / pCentral;
  part[4] = time[ip] * cMks * P1 / gamma1;
  Pz = P / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  Pz1 = sqrt(Pz * Pz + gamma1 * gamma1 - gamma * gamma);
  PRatio = Pz / Pz1;
  part[1] *= PRatio;
  part[3] *= PRatio;
  pz = pCentral * (1 + part[5]) / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  part[1] += (voltage[ib] + part[0] * voltage[3 * nBunches + ib]) * factor / (1e6 * massMV) / pz;
  part[3] += (voltage[nBunches + ib] + part[2] * voltage[4 * nBunches + ib]) * factor / (1e6 * massMV) / pz;
}

/* hostVoltage: Vx, Vy, Vz, Qx, Qy for each bunch, nBunches values each */
extern "C" int gpuCudaLrwakeKick(void *coord, long nParticles, int stride, const double *hostVoltage,
                                 long nBunches, long ibMin, int bunchColumn, double factor,
                                 double pCentral, double massMV, double cMks, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int status, threads = 256;

  if (!coord || nParticles <= 0 || stride < 7 || !hostVoltage || nBunches < 1 ||
      gpuLrwakeCapacity < nParticles || !gpuLrwakeTime)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuLrwakeVoltageCapacity < 5 * nBunches) {
    gpuFreeDevice(gpuLrwakeVoltage);
    gpuLrwakeVoltage = NULL;
    gpuLrwakeVoltageCapacity = 0;
    cudaStatus = cudaMalloc(&gpuLrwakeVoltage, 5 * nBunches * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuLrwakeVoltageCapacity = 5 * nBunches;
  }
  cudaStatus = cudaMemcpy(gpuLrwakeVoltage, hostVoltage, 5 * nBunches * sizeof(double),
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuLrwakeKickKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, gpuLrwakeTime, gpuLrwakeVoltage, nBunches, ibMin,
    bunchColumn, factor, pCentral, massMV, cMks);
  return finishTimedKernel(start, stop, milliseconds);
}

extern "C" int gpuCudaRfcwRfOnlyMatrix(void *coord, long nParticles, int stride,
                                       double pCentral, double length,
                                       double volt, double omega,
                                       double phase, int end1Focus,
                                       int end2Focus, double dx, double dy,
                                       double cMks, const void *referenceKick,
                                       const void *referenceR12,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcwRfOnlyMatrixKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                                 stride, pCentral, length,
                                                 volt, omega, phase,
                                                 end1Focus, end2Focus, dx, dy,
                                                 cMks, NULL,
                                                 static_cast<const double *>(referenceKick),
                                                 static_cast<const double *>(referenceR12));
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfcwRfOnlyMatrixChecked(
  void *coord, long nParticles, int stride, double pCentral, double length,
  double volt, double omega, double phase, int end1Focus, int end2Focus,
  double dx, double dy, double cMks, const void *referenceKick,
  const void *referenceR12, void *deviceLostCount,
  long *lostCount, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !deviceLostCount || !lostCount || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*lostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuRfcwRfOnlyMatrixKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, length, volt,
    omega, phase, end1Focus, end2Focus, dx, dy, cMks,
    static_cast<long *>(deviceLostCount),
    static_cast<const double *>(referenceKick),
    static_cast<const double *>(referenceR12));
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(lostCount, deviceLostCount,
                                     sizeof(*lostCount),
                                     cudaMemcpyDeviceToHost));
}

extern "C" int gpuCudaRfcwKickInitial(void *coord, void *inverseF,
                                      long nParticles, int stride,
                                      double pCentral, double length,
                                      double volt, double omega, double phase,
                                      double timeShift, int end1Focus,
                                      double cMks, const void *referenceKick,
                                      float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !inverseF)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcwKickInitialKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), static_cast<double *>(inverseF),
    nParticles, stride, pCentral, length, volt, omega, phase, timeShift,
    end1Focus, cMks, static_cast<const double *>(referenceKick));
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfcwKickFinal(void *coord, const void *inverseF,
                                    long nParticles, int stride,
                                    double length, int end2Focus,
                                    float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !inverseF)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcwKickFinalKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), static_cast<const double *>(inverseF),
    nParticles, stride, length, end2Focus);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfcwDgammaOverGammaSums(void *coord, long nParticles,
                                              int stride, double pCentral,
                                              double length, double volt,
                                              double omega, double phase,
                                              double timeShift, double cMks,
                                              GPU_BEAM_SUM_DATA *result,
                                              float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcwDgammaOverGammaSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral,
    length, volt, omega, phase, timeShift, cMks, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaSubtractCoordinate(void *coord, long nParticles, int stride,
                                         int index, double value, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSubtractCoordinateKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                                   stride, index, value);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCenterTime(void *coord, long nParticles, int stride,
                                 double pCentral, double timeOffset, double cMks,
                                 float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCenterTimeKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                           stride, pCentral, timeOffset, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* CENTER's time coordinate as center_beam() does it: computeTimeCoordinates(), the offset subtracted, and
 * computeDistanceCoordinates(), even for a zero offset (the round trip itself rounds) */
__global__ void gpuCenterTimeRoundTripKernel(double *coord, long nParticles, int stride, double pCentral,
                                             double timeOffset, double cMks) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, P, t, beta;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  P = pCentral * (part[5] + 1);
  t = part[4] * sqrt(P * P + 1) / (cMks * P);
  t -= timeOffset;
  beta = P / sqrt(P * P + 1);
  part[4] = cMks * beta * t;
}

extern "C" int gpuCudaCenterTimeRoundTrip(void *coord, long nParticles, int stride, double pCentral,
                                          double timeOffset, double cMks, float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCenterTimeRoundTripKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                                    pCentral, timeOffset, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

/* Each particle's time (computeTimeCoordinates()'s expression) copied to the host */
__global__ void gpuTimeColumnKernel(const double *coord, long nParticles, int stride, double pCentral,
                                    double cMks, double *time) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double P;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  P = pCentral * (part[5] + 1);
  time[ip] = part[4] * sqrt(P * P + 1) / (cMks * P);
}

extern "C" int gpuCudaTimeColumnToHost(const void *coord, long nParticles, int stride, double pCentral,
                                       double cMks, double *hostTimes, float *milliseconds) {
  static double *deviceTimes = NULL;
  static long capacity = 0;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || nParticles <= 0 || !hostTimes)
    return static_cast<int>(cudaErrorInvalidValue);
  if (capacity < nParticles) {
    gpuFreeDevice(deviceTimes);
    deviceTimes = NULL;
    capacity = 0;
    cudaStatus = cudaMalloc(&deviceTimes, nParticles * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    capacity = nParticles;
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuTimeColumnKernel<<<blocks, threads>>>(static_cast<const double *>(coord), nParticles, stride, pCentral,
                                           cMks, deviceTimes);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(hostTimes, deviceTimes, nParticles * sizeof(double),
                                     cudaMemcpyDeviceToHost));
}

extern "C" int gpuCudaCenterBeam(void *coord, long nParticles, int stride,
                                 unsigned int coordinateMask, const double *offset,
                                 int doTime, double pCentral, double timeOffset,
                                 double cMks, float *milliseconds) {
  cudaEvent_t start, stop;
  double offsetValues[6] = {0, 0, 0, 0, 0, 0};
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (offset)
    std::memcpy(offsetValues, offset, sizeof(offsetValues));
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCenterBeamKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride,
                                           coordinateMask, offsetValues[0], offsetValues[1],
                                           offsetValues[2], offsetValues[3], offsetValues[4],
                                           offsetValues[5], doTime, pCentral, timeOffset, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaHistogramRanges(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  int64_t startPID, int64_t endPID, unsigned int coordinateMask,
  GPU_HISTOGRAM_RANGE_DATA *result, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  int status;

  if (!coord || !result || nParticles <= 0 || stride < 7 ||
      !(coordinateMask & 0x7fU))
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  status = ensureHistogramScratch(2);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuHistogramRangePartialKernel<<<GPU_HISTOGRAM_BLOCKS,
                                   GPU_HISTOGRAM_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, pCentral, cMks,
    startPID, endPID, coordinateMask, gpuHistogramScratch.rangePartial);
  gpuHistogramRangeFinalizeKernel<<<1, GPU_HISTOGRAM_THREADS>>>(
    gpuHistogramScratch.rangePartial, gpuHistogramScratch.rangeResult);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(result, gpuHistogramScratch.rangeResult,
                                     sizeof(*result),
                                     cudaMemcpyDeviceToHost));
}

extern "C" int gpuCudaHistogramBins(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  int64_t startPID, int64_t endPID, long bins, unsigned int coordinateMask,
  double timeOffset, const double *lower, const double *upper,
  unsigned long long *histogramReturn, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_HISTOGRAM_BIN_DATA data;
  cudaError_t cudaStatus;
  int blocks, status;

  if (!coord || !lower || !upper || !histogramReturn || nParticles <= 0 ||
      stride < 7 || bins < 2 || !(coordinateMask & 0x7fU))
    return static_cast<int>(cudaErrorInvalidValue);
  status = ensureHistogramScratch(bins);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  std::memset(&data, 0, sizeof(data));
  data.bins = bins;
  data.startPID = startPID;
  data.endPID = endPID;
  data.coordinateMask = coordinateMask;
  data.pCentral = pCentral;
  data.cMks = cMks;
  data.timeOffset = timeOffset;
  std::memcpy(data.lower, lower, sizeof(data.lower));
  std::memcpy(data.upper, upper, sizeof(data.upper));
  cudaStatus = cudaMemset(
    gpuHistogramScratch.histogramPartial, 0,
    7 * bins * GPU_HISTOGRAM_BLOCKS *
      sizeof(*gpuHistogramScratch.histogramPartial));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  gpuHistogramBinPartialKernel<<<GPU_HISTOGRAM_BLOCKS,
                                 GPU_HISTOGRAM_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, data,
    gpuHistogramScratch.histogramPartial);
  blocks = static_cast<int>((7 * bins + 255) / 256);
  gpuHistogramBinFinalizeKernel<<<blocks, 256>>>(
    gpuHistogramScratch.histogramPartial, bins,
    gpuHistogramScratch.histogram);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(
    histogramReturn, gpuHistogramScratch.histogram,
    7 * bins * sizeof(*histogramReturn), cudaMemcpyDeviceToHost));
}

static int gpuCudaRunReduction(void *coord, long nParticles, int stride,
                               double pCentral, double cMks, int beamSums,
                               GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  int blocks, status;

  if (!coord || !result || nParticles < 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  status = ensureGenericReductionScratch();
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemset(gpuGenericReductionScratch.result, 0,
                          sizeof(*gpuGenericReductionScratch.result));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaSuccess;
  if (beamSums > 0) {
    gpuBeamSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord), nParticles,
                                                    stride, pCentral, cMks,
                                                    gpuGenericReductionScratch.result);
  } else {
    blocks = static_cast<int>((nParticles + GPU_REDUCTION_THREADS - 1) /
                              GPU_REDUCTION_THREADS);
    if (blocks < 1)
      blocks = 1;
    if (blocks > GPU_BEAM_SUM_BLOCKS)
      blocks = GPU_BEAM_SUM_BLOCKS;
    gpuSimpleSumsPartialKernel<<<blocks, GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
      beamSums, gpuGenericReductionScratch.partial);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      gpuSimpleSumsFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
        gpuGenericReductionScratch.partial, blocks,
        gpuGenericReductionScratch.result);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaMemcpy(
    result, gpuGenericReductionScratch.result, sizeof(*result),
    cudaMemcpyDeviceToHost));
}

/* What findFiducialTime() computes for each particle, whether the particle is in the fiducial bunch,
 * and its delta, for the host to pick from or add in particle order */
__global__ void gpuFiducialValuesKernel(const double *coord, long nParticles,
                                        int stride, double pCentral,
                                        double sOffset, double cMks,
                                        int particleIdColumn, int64_t startPID,
                                        int64_t endPID, double *value,
                                        unsigned char *selected, double *delta) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double p;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  /* beta_from_delta(), then findFiducialTime()'s expression */
  p = pCentral;
  p *= 1 + part[5];
  value[ip] = (part[4] + sOffset) / (cMks * (p / sqrt(p * p + 1)));
  selected[ip] = (startPID < 0 && endPID < 0) ||
                 (particleIdColumn >= 0 && particleIdColumn < stride &&
                  part[particleIdColumn] >= startPID &&
                  part[particleIdColumn] <= endPID);
  if (delta)
    delta[ip] = part[5];
}

/* A cavity's reference time needs one number per particle, not the beam: the copy of the whole beam
 * to the host took 3 ms a cavity for 200,000 particles */
extern "C" int gpuCudaFiducialValues(void *coord, long nParticles, int stride,
                                     double pCentral, double sOffset,
                                     double cMks, int particleIdColumn,
                                     int64_t startPID, int64_t endPID,
                                     double *hostValue,
                                     unsigned char *hostSelected,
                                     double *hostDelta, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceValue = NULL, *deviceDelta = NULL;
  unsigned char *deviceSelected = NULL;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (milliseconds)
    *milliseconds = 0;
  if (!coord || nParticles <= 0 || !hostValue || !hostSelected)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceValue, nParticles * sizeof(*deviceValue));
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuReuseDeviceBuffer(1, &deviceSelected, nParticles * sizeof(*deviceSelected));
  if (cudaStatus == cudaSuccess && hostDelta)
    cudaStatus = gpuReuseDeviceBuffer(2, &deviceDelta, nParticles * sizeof(*deviceDelta));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuFiducialValuesKernel<<<blocks, threads>>>(
    static_cast<const double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceValue, deviceSelected,
    deviceDelta);
  status = launchTimedKernel(cudaGetLastError(), start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(hostValue, deviceValue, nParticles * sizeof(*hostValue),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(hostSelected, deviceSelected,
                            nParticles * sizeof(*hostSelected),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostDelta)
    cudaStatus = cudaMemcpy(hostDelta, deviceDelta, nParticles * sizeof(*hostDelta),
                            cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

/* Each particle's time as the wakes take it, and whether it belongs to the selected bunch */
__global__ void gpuOrderedTimeValuesKernel(double *coord, long nParticles,
                                           int stride, double pCentral,
                                           double cMks, int bunchColumn,
                                           long selectedBunch, double *time,
                                           unsigned char *selected) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  selected[ip] = bunchColumn < 0 ||
                 static_cast<long>(part[bunchColumn]) == selectedBunch;
  time[ip] = gpuParticleTime(part, pCentral, cMks);
}

/* The sum of particle times in index order, as compute_average() and the
 * bunched loop in track_through_wake() make it.  The device makes each
 * particle's time and the host adds them in particle order, so the CPU's
 * rounding is reproduced.  One device thread used to add them: 5.9 ms for
 * 10,000 particles. */
extern "C" int gpuCudaOrderedTimeSum(void *coord, long nParticles, int stride,
                                     double pCentral, double cMks,
                                     int bunchColumn, long selectedBunch,
                                     double *sum, long *count,
                                     float *milliseconds) {
  static double *hostTime = NULL;
  static unsigned char *hostSelected = NULL;
  static long hostCapacity = 0;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceTime = NULL;
  unsigned char *deviceSelected = NULL;
  double total = 0;
  long n = 0, ip;
  int threads = gpuParticleThreads(nParticles);
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || nParticles <= 0 || !sum || !count)
    return static_cast<int>(cudaErrorInvalidValue);
  if (hostCapacity < nParticles) {
    std::free(hostTime);
    std::free(hostSelected);
    hostTime = static_cast<double *>(std::malloc(nParticles * sizeof(*hostTime)));
    hostSelected = static_cast<unsigned char *>(std::malloc(nParticles * sizeof(*hostSelected)));
    hostCapacity = hostTime && hostSelected ? nParticles : 0;
    if (!hostCapacity) {
      std::free(hostTime);
      std::free(hostSelected);
      hostTime = NULL;
      hostSelected = NULL;
      return static_cast<int>(cudaErrorMemoryAllocation);
    }
  }
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuReuseDeviceBuffer(1, &deviceSelected, nParticles * sizeof(*deviceSelected));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuOrderedTimeValuesKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    bunchColumn, selectedBunch, deviceTime, deviceSelected);
  status = launchTimedKernel(cudaGetLastError(), start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(hostTime, deviceTime, nParticles * sizeof(*hostTime),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(hostSelected, deviceSelected,
                            nParticles * sizeof(*hostSelected),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  for (ip = 0; ip < nParticles; ip++) {
    if (!hostSelected[ip])
      continue;
    total += hostTime[ip];
    n++;
  }
  *sum = total;
  *count = n;
  return static_cast<int>(cudaSuccess);
}

/* Count, least and greatest time of the particles (of one bunch when bunchColumn is not negative),
 * one particle per thread.  Those are comparisons only, so they don't depend on the order.  The sum
 * of the times does: it is added pairwise inside each block of 256 and then over the blocks, the same
 * way at every call for a given number of particles, but not in the order the CPU adds them. */
typedef struct GPU_TIME_SPAN_PARTIAL {
  int64_t count;
  double minValue;
  double maxValue;
  double sum;
} GPU_TIME_SPAN_PARTIAL;

#define GPU_TIME_SPAN_THREADS 256

__global__ void gpuTimeSpanPartialKernel(double *coord, long nParticles,
                                         int stride, double pCentral,
                                         double cMks, int bunchColumn,
                                         long selectedBunch,
                                         GPU_TIME_SPAN_PARTIAL *partial) {
  __shared__ int64_t count[GPU_TIME_SPAN_THREADS];
  __shared__ double minValue[GPU_TIME_SPAN_THREADS];
  __shared__ double maxValue[GPU_TIME_SPAN_THREADS];
  __shared__ double sum[GPU_TIME_SPAN_THREADS];
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  count[tid] = 0;
  minValue[tid] = DBL_MAX;
  maxValue[tid] = -DBL_MAX;
  sum[tid] = 0;
  if (ip < nParticles) {
    double *part = coord + ip * stride;
    if (bunchColumn < 0 ||
        static_cast<long>(part[bunchColumn]) == selectedBunch) {
      double t = gpuParticleTime(part, pCentral, cMks);
      count[tid] = 1;
      sum[tid] = t;
      if (t < minValue[tid])
        minValue[tid] = t;
      if (t > maxValue[tid])
        maxValue[tid] = t;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      sum[tid] += sum[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x].count = count[0];
    partial[blockIdx.x].minValue = minValue[0];
    partial[blockIdx.x].maxValue = maxValue[0];
    partial[blockIdx.x].sum = sum[0];
  }
}

__global__ void gpuTimeSpanFinalizeKernel(const GPU_TIME_SPAN_PARTIAL *partial,
                                          int blocks,
                                          GPU_TIME_SPAN_PARTIAL *result) {
  __shared__ int64_t count[GPU_TIME_SPAN_THREADS];
  __shared__ double minValue[GPU_TIME_SPAN_THREADS];
  __shared__ double maxValue[GPU_TIME_SPAN_THREADS];
  __shared__ double sum[GPU_TIME_SPAN_THREADS];
  int tid = threadIdx.x;

  count[tid] = 0;
  minValue[tid] = DBL_MAX;
  maxValue[tid] = -DBL_MAX;
  sum[tid] = 0;
  for (int ib = tid; ib < blocks; ib += blockDim.x) {
    count[tid] += partial[ib].count;
    sum[tid] += partial[ib].sum;
    if (partial[ib].minValue < minValue[tid])
      minValue[tid] = partial[ib].minValue;
    if (partial[ib].maxValue > maxValue[tid])
      maxValue[tid] = partial[ib].maxValue;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      count[tid] += count[tid + offset];
      sum[tid] += sum[tid + offset];
      if (minValue[tid + offset] < minValue[tid])
        minValue[tid] = minValue[tid + offset];
      if (maxValue[tid + offset] > maxValue[tid])
        maxValue[tid] = maxValue[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result->count = count[0];
    result->minValue = minValue[0];
    result->maxValue = maxValue[0];
    result->sum = sum[0];
  }
}

/* The wakes need only this of the beam's times to place their bins.  They used to take it from
 * gpuBeamSumsKernel, which adds every sum and product of the beam on one block of 64 threads. */
extern "C" int gpuCudaTimeSpan(void *coord, long nParticles, int stride,
                               double pCentral, double cMks, int bunchColumn,
                               long selectedBunch, long *count, double *tmin,
                               double *tmax, double *tsum, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_TIME_SPAN_PARTIAL *partial = NULL;
  GPU_TIME_SPAN_PARTIAL hostResult;
  int blocks, status;

  if (!coord || nParticles <= 0 || !count || !tmin || !tmax)
    return static_cast<int>(cudaErrorInvalidValue);
  *count = 0;
  *tmin = *tmax = 0;
  if (tsum)
    *tsum = 0;
  if (milliseconds)
    *milliseconds = 0;
  if ((nParticles + GPU_TIME_SPAN_THREADS - 1) / GPU_TIME_SPAN_THREADS >= INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + GPU_TIME_SPAN_THREADS - 1) / GPU_TIME_SPAN_THREADS);
  cudaStatus = gpuReuseDeviceBuffer(0, &partial, (static_cast<size_t>(blocks) + 1) * sizeof(*partial));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuTimeSpanPartialKernel<<<blocks, GPU_TIME_SPAN_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    bunchColumn, selectedBunch, partial);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    gpuTimeSpanFinalizeKernel<<<1, GPU_TIME_SPAN_THREADS>>>(partial, blocks, partial + blocks);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostResult, partial + blocks, sizeof(hostResult),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *count = static_cast<long>(hostResult.count);
  if (hostResult.count > 0) {
    *tmin = hostResult.minValue;
    *tmax = hostResult.maxValue;
    if (tsum)
      *tsum = hostResult.sum;
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCentroidSums(void *coord, long nParticles, int stride,
                                   GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  return gpuCudaRunReduction(coord, nParticles, stride, 0, 1, 0, result, milliseconds);
}

extern "C" int gpuCudaTimeSums(void *coord, long nParticles, int stride,
                               double pCentral, double cMks,
                               GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  return gpuCudaRunReduction(coord, nParticles, stride, pCentral, cMks, -1, result, milliseconds);
}

extern "C" int gpuCudaSelectedTimeSums(void *coord, long nParticles, int stride,
                                       double pCentral, double cMks,
                                       int bunchColumn, long selectedBunch,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  if (!coord || nParticles <= 0 || bunchColumn < 0 || bunchColumn >= stride)
    return static_cast<int>(cudaErrorInvalidValue);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSelectedTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    bunchColumn, selectedBunch, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialTimeSums(void *coord, long nParticles, int stride,
                                       double pCentral, double sOffset,
                                       double cMks, int particleIdColumn,
                                       int64_t startPID, int64_t endPID,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuFiducialTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialPmaximum(void *coord, long nParticles,
                                       int stride, double pCentral,
                                       double sOffset, double cMks,
                                       int particleIdColumn, int64_t startPID,
                                       int64_t endPID,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuFiducialPmaximumKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialFirst(void *coord, long nParticles,
                                    int stride, double pCentral,
                                    double sOffset, double cMks,
                                    int particleIdColumn, int64_t startPID,
                                    int64_t endPID, GPU_BEAM_SUM_DATA *result,
                                    float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuFiducialFirstKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaCentroidTimeSums(void *coord, long nParticles, int stride,
                                       double pCentral, double cMks,
                                       GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  return gpuCudaRunReduction(coord, nParticles, stride, pCentral, cMks, -2, result, milliseconds);
}

extern "C" int gpuCudaBeamSums(void *coord, long nParticles, int stride,
                               double pCentral, double cMks,
                               GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  return gpuCudaRunReduction(coord, nParticles, stride, pCentral, cMks, 1, result, milliseconds);
}

extern "C" int gpuCudaCenteredBeamSums(void *coord, long nParticles,
                                       int stride, double pCentral,
                                       double cMks, const double *centroid,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  double *deviceCentroid = NULL;
  int status;

  if (!coord || !centroid || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = gpuReuseDeviceBuffer(1, &deviceCentroid, sizeof(double) * 7);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemcpy(deviceCentroid, centroid, sizeof(double) * 7,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCenteredBeamSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    deviceCentroid, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" unsigned long gpuCudaBeamSums2ScratchBytes(long nParticles) {
  return (2 + 2 * GPU_BEAM_SUM_BLOCKS) * sizeof(GPU_BEAM_SUM_DATA) +
         (7 + (nParticles > 0 ? nParticles : 0)) * sizeof(double);
}

static struct {
  double *partial;
  double *result;
  double *centroid;
} gpuSpinSumsScratch;

extern "C" int gpuCudaSpinSums(void *coord, long nParticles, int stride, int spinOffset,
                               double centroidSum[3], double productSum[6], float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double host[9];
  int blocks, status;

  if (!coord || nParticles <= 0 || spinOffset < 0 || spinOffset + 3 > stride)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuSpinSumsScratch.partial) {
    cudaStatus = cudaMalloc(&gpuSpinSumsScratch.partial, GPU_BEAM_SUM_BLOCKS * 6 * sizeof(double));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&gpuSpinSumsScratch.result, 9 * sizeof(double));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&gpuSpinSumsScratch.centroid, 3 * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  blocks = static_cast<int>((nParticles + GPU_BEAM_OUTPUT_THREADS - 1) / GPU_BEAM_OUTPUT_THREADS);
  if (blocks > GPU_BEAM_SUM_BLOCKS)
    blocks = GPU_BEAM_SUM_BLOCKS;
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSpinSumsPartialKernel<<<blocks, GPU_BEAM_OUTPUT_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, spinOffset, NULL, gpuSpinSumsScratch.partial);
  gpuSpinSumsFinalizeKernel<<<1, 32>>>(gpuSpinSumsScratch.partial, blocks, 3, nParticles,
                                        gpuSpinSumsScratch.result, gpuSpinSumsScratch.centroid);
  gpuSpinSumsPartialKernel<<<blocks, GPU_BEAM_OUTPUT_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, spinOffset, gpuSpinSumsScratch.centroid,
    gpuSpinSumsScratch.partial);
  gpuSpinSumsFinalizeKernel<<<1, 32>>>(gpuSpinSumsScratch.partial, blocks, 6, nParticles,
                                        gpuSpinSumsScratch.result + 3, NULL);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(host, gpuSpinSumsScratch.result, sizeof(host), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  std::memcpy(centroidSum, host, 3 * sizeof(double));
  std::memcpy(productSum, host + 3, 6 * sizeof(double));
  return static_cast<int>(cudaSuccess);
}

/* performSliceAnalysis(): computeTimeCoordinatesOnly() into column 4's place, then computeSliceMoments()
 * for each slice, every sum run by one thread over the particles in order, so they are the CPU's */
static struct {
  double *time;
  long timeCapacity;
  double *bounds; /* each slice's lower and upper time */
  double *mean;
  double *sum;
  long *count;
  long sliceCapacity;
} gpuSliceScratch;

__global__ void gpuSliceTimeKernel(const double *coord, long nParticles, int stride, double Po,
                                   double cMks, double *time) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *part;
  double P;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  P = Po * (part[5] + 1);
  time[ip] = part[4] * sqrt(P * P + 1) / (cMks * P);
}

__device__ __forceinline__ double gpuSliceCoordinate(const double *coord, const double *time, long ip,
                                                     int stride, int j) {
  return j == 4 ? time[ip] : coord[ip * stride + j];
}

/* One thread per slice and coordinate.  Bounds that are equal take every particle */
__global__ void gpuSliceSumsKernel(const double *coord, const double *time, long nParticles, int stride,
                                   const double *bounds, long slices, double *sum, long *count) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long slice, used = 0;
  int j;
  double lower, upper, total = 0;

  if (t >= slices * 6)
    return;
  slice = t / 6;
  j = (int)(t % 6);
  lower = bounds[2 * slice];
  upper = bounds[2 * slice + 1];
  for (long ip = 0; ip < nParticles; ip++) {
    if (lower != upper && (time[ip] < lower || time[ip] > upper))
      continue;
    used++;
    total += gpuSliceCoordinate(coord, time, ip, stride, j);
  }
  sum[t] = total;
  if (j == 0)
    count[slice] = used;
}

/* One thread per slice and lower-triangle pair, about the means the host divided out */
__global__ void gpuSliceSecondMomentsKernel(const double *coord, const double *time, long nParticles,
                                            int stride, const double *bounds, const double *mean,
                                            long slices, double *sum) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long slice;
  int pair, j = 0, k;
  double lower, upper, total = 0;

  if (t >= slices * 21)
    return;
  slice = t / 21;
  pair = (int)(t % 21);
  while (pair > j) {
    pair -= j + 1;
    j++;
  }
  k = pair;
  lower = bounds[2 * slice];
  upper = bounds[2 * slice + 1];
  for (long ip = 0; ip < nParticles; ip++) {
    if (lower != upper && (time[ip] < lower || time[ip] > upper))
      continue;
    total += (gpuSliceCoordinate(coord, time, ip, stride, j) - mean[slice * 6 + j]) *
             (gpuSliceCoordinate(coord, time, ip, stride, k) - mean[slice * 6 + k]);
  }
  sum[t] = total;
}

static int gpuEnsureSliceScratch(long nParticles, long slices) {
  cudaError_t status = cudaSuccess;

  if (gpuSliceScratch.timeCapacity < nParticles) {
    gpuFreeDevice(gpuSliceScratch.time);
    gpuSliceScratch.time = NULL;
    gpuSliceScratch.timeCapacity = 0;
    status = cudaMalloc(&gpuSliceScratch.time, nParticles * sizeof(double));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuSliceScratch.timeCapacity = nParticles;
  }
  if (gpuSliceScratch.sliceCapacity < slices) {
    gpuFreeDevice(gpuSliceScratch.bounds);
    gpuFreeDevice(gpuSliceScratch.mean);
    gpuFreeDevice(gpuSliceScratch.sum);
    gpuFreeDevice(gpuSliceScratch.count);
    gpuSliceScratch.bounds = gpuSliceScratch.mean = gpuSliceScratch.sum = NULL;
    gpuSliceScratch.count = NULL;
    gpuSliceScratch.sliceCapacity = 0;
    status = cudaMalloc(&gpuSliceScratch.bounds, 2 * slices * sizeof(double));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuSliceScratch.mean, 6 * slices * sizeof(double));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuSliceScratch.sum, 21 * slices * sizeof(double));
    if (status == cudaSuccess)
      status = cudaMalloc(&gpuSliceScratch.count, slices * sizeof(long));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuSliceScratch.sliceCapacity = slices;
  }
  return static_cast<int>(cudaSuccess);
}

/* The time coordinates, and the bunch's earliest and latest */
extern "C" int gpuCudaSliceTimes(void *coord, long nParticles, int stride, double Po, double cMks,
                                 double *tMin, double *tMax, float *milliseconds) {
  cudaEvent_t start, stop;
  int status;

  if (!coord || nParticles <= 0 || !tMin || !tMax)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuEnsureSliceScratch(nParticles, 1);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSliceTimeKernel<<<static_cast<int>((nParticles + 255) / 256), 256>>>(
    static_cast<const double *>(coord), nParticles, stride, Po, cMks, gpuSliceScratch.time);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  {
    thrust::device_ptr<double> time(gpuSliceScratch.time);
    auto extremes = thrust::minmax_element(time, time + nParticles);
    *tMin = *extremes.first;
    *tMax = *extremes.second;
  }
  return static_cast<int>(cudaGetLastError());
}

/* For each of slices bounds (lower, upper): the particle count, the coordinate means as
 * computeSliceMoments() divides them, and the 21 second-moment sums of the lower triangle */
extern "C" int gpuCudaSliceMoments(void *coord, long nParticles, int stride, const double *bounds,
                                   long slices, long *count, double *mean, double *secondSum,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *sum;
  int status;

  if (!coord || nParticles <= 0 || slices <= 0 || !bounds || !count || !mean || !secondSum)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuEnsureSliceScratch(nParticles, slices);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  sum = static_cast<double *>(std::malloc(6 * slices * sizeof(double)));
  if (!sum)
    return static_cast<int>(cudaErrorMemoryAllocation);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  cudaStatus = cudaMemcpy(gpuSliceScratch.bounds, bounds, 2 * slices * sizeof(double), cudaMemcpyHostToDevice);
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess) {
    gpuSliceSumsKernel<<<static_cast<int>((slices * 6 + 63) / 64), 64>>>(
      static_cast<const double *>(coord), gpuSliceScratch.time, nParticles, stride,
      gpuSliceScratch.bounds, slices, gpuSliceScratch.sum, gpuSliceScratch.count);
    cudaStatus = cudaGetLastError();
  }
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(sum, gpuSliceScratch.sum, 6 * slices * sizeof(double), cudaMemcpyDeviceToHost);
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(count, gpuSliceScratch.count, slices * sizeof(long), cudaMemcpyDeviceToHost);
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess) {
    /* computeSliceMoments() divides on the host's side of the loop */
    for (long slice = 0; slice < slices; slice++)
      for (int j = 0; j < 6; j++)
        mean[slice * 6 + j] = count[slice] ? sum[slice * 6 + j] / count[slice] : 0;
    cudaStatus = cudaMemcpy(gpuSliceScratch.mean, mean, 6 * slices * sizeof(double), cudaMemcpyHostToDevice);
  }
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess) {
    gpuSliceSecondMomentsKernel<<<static_cast<int>((slices * 21 + 63) / 64), 64>>>(
      static_cast<const double *>(coord), gpuSliceScratch.time, nParticles, stride,
      gpuSliceScratch.bounds, gpuSliceScratch.mean, slices, gpuSliceScratch.sum);
    cudaStatus = cudaGetLastError();
  }
  if (status == static_cast<int>(cudaSuccess) && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(secondSum, gpuSliceScratch.sum, 21 * slices * sizeof(double), cudaMemcpyDeviceToHost);
  std::free(sum);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return launchTimedKernel(cudaStatus, start, stop, milliseconds);
}

extern "C" int gpuCudaBeamSums2(void *coord, long nParticles, int stride,
                                double pCentral, double cMks,
                                GPU_BEAM_SUM_DATA *result,
                                GPU_BEAM_SUM_DATA *centeredResult,
                                void *deviceScratch,
                                unsigned int productMask,
                                float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_BEAM_SUM_DATA *deviceResult;
  GPU_BEAM_SUM_DATA *deviceCenteredResult;
  GPU_BEAM_SUM_DATA *partial;
  GPU_BEAM_SUM_DATA *centeredPartial;
  double *deviceCentroid;
  double *deviceTime;
  GPU_BEAM_SUM_DATA hostResult[2];
  cudaError_t cudaStatus;
  char *scratch = static_cast<char *>(deviceScratch);
  int blocks, status, timed;

  if (!coord || !result || !centeredResult || !deviceScratch || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  std::memset(centeredResult, 0, sizeof(*centeredResult));

  blocks = static_cast<int>((nParticles + GPU_REDUCTION_THREADS - 1) /
                            GPU_REDUCTION_THREADS);
  if (blocks > GPU_BEAM_SUM_BLOCKS)
    blocks = GPU_BEAM_SUM_BLOCKS;
  if (blocks < 1)
    blocks = 1;

  deviceResult = reinterpret_cast<GPU_BEAM_SUM_DATA *>(scratch);
  deviceCenteredResult = deviceResult + 1;
  partial = deviceCenteredResult + 1;
  centeredPartial = partial + GPU_BEAM_SUM_BLOCKS;
  deviceCentroid = reinterpret_cast<double *>(centeredPartial +
                                               GPU_BEAM_SUM_BLOCKS);
  deviceTime = deviceCentroid + 7;

  timed = gpuDetailedKernelTimingEnabled();
  if (milliseconds)
    *milliseconds = 0;
  if (timed) {
    status = getCachedTimingEvents(&start, &stop);
    if (status != static_cast<int>(cudaSuccess))
      return status;
    cudaStatus = cudaEventRecord(start, 0);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }

  gpuBeamStatisticsPartialKernel<<<blocks, GPU_BEAM_OUTPUT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    partial, deviceTime);
  gpuBeamStatisticsFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
    partial, blocks, deviceResult, deviceCentroid);
  if (productMask) {
    gpuCenteredBeamSumsPartialKernel<<<blocks, GPU_BEAM_OUTPUT_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
      deviceCentroid, deviceTime, productMask, centeredPartial);
    gpuCenteredBeamSumsFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
      centeredPartial, blocks, deviceCenteredResult);
  }

  status = timed ? finishTimedKernel(start, stop, milliseconds) :
                   static_cast<int>(cudaGetLastError());
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(hostResult, deviceResult, sizeof(hostResult),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *result = hostResult[0];
  *centeredResult = hostResult[1];
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaLscStatistics(void *coord, long nParticles, int stride,
                                     double pCentral, double cMks,
                                     GPU_BEAM_SUM_DATA *result,
                                     void *deviceResultScratch,
                                     float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult =
    static_cast<GPU_BEAM_SUM_DATA *>(deviceResultScratch);
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  if (!deviceResult) {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuLscStatisticsKernel<<<1, GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
      deviceResult);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLscTransverseSums(void *coord, long nParticles,
                                         int stride, double xCentroid,
                                         double yCentroid,
                                         GPU_BEAM_SUM_DATA *result,
                                         void *deviceResultScratch,
                                         float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult =
    static_cast<GPU_BEAM_SUM_DATA *>(deviceResultScratch);
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  if (!deviceResult) {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuLscTransverseSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, xCentroid,
      yCentroid, deviceResult);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLongMinMax(void *coord, long nParticles, int stride,
                                 int coordinateIndex,
                                 GPU_LONG_MIN_MAX_DATA *result,
                                 float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_LONG_MIN_MAX_DATA *partial = NULL;
  int blocks, status;

  if (!result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  if (!coord || nParticles <= 0 || coordinateIndex < 0 || coordinateIndex >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((nParticles + GPU_LONG_MIN_MAX_THREADS - 1) / GPU_LONG_MIN_MAX_THREADS >= INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  blocks = static_cast<int>((nParticles + GPU_LONG_MIN_MAX_THREADS - 1) / GPU_LONG_MIN_MAX_THREADS);

  cudaStatus = gpuReuseDeviceBuffer(0, &partial, (static_cast<size_t>(blocks) + 1) * sizeof(*partial));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuLongMinMaxPartialKernel<<<blocks, GPU_LONG_MIN_MAX_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, coordinateIndex, partial);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    gpuLongMinMaxFinalizeKernel<<<1, GPU_LONG_MIN_MAX_THREADS>>>(partial, blocks, partial + blocks);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, partial + blocks, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaSortedBunchRanges(
  void *coord, long nParticles, int stride, int coordinateIndex,
  long minBunch, long nBuckets, long *start, long *count, long *sorted,
  float *milliseconds) {
  cudaEvent_t startEvent, stopEvent;
  cudaError_t cudaStatus;
  long *deviceStart = NULL, *deviceCount = NULL;
  int *deviceSorted = NULL, hostSorted = 1;
  int threads = 256, blocks, bucketBlocks, status;

  if (!coord || nParticles <= 0 || stride <= 0 || coordinateIndex < 0 ||
      coordinateIndex >= stride || nBuckets <= 0 || !start || !count ||
      !sorted)
    return static_cast<int>(cudaErrorInvalidValue);
  *sorted = 0;
  cudaStatus = gpuReuseDeviceBuffer(0, &deviceStart, nBuckets * sizeof(*deviceStart));
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuReuseDeviceBuffer(1, &deviceCount, nBuckets * sizeof(*deviceCount));
  if (cudaStatus == cudaSuccess)
    cudaStatus = gpuReuseDeviceBuffer(2, &deviceSorted, sizeof(*deviceSorted));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceSorted, &hostSorted, sizeof(hostSorted),
                            cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    goto cleanup;

  status = prepareTimedLaunch(&startEvent, &stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto cleanup;
  }
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  bucketBlocks = static_cast<int>((nBuckets + threads - 1) / threads);
  gpuSortedBunchValidateKernel<<<blocks, threads>>>(
    static_cast<const double *>(coord), nParticles, stride, coordinateIndex,
    deviceSorted);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    gpuSortedBunchRangesKernel<<<bucketBlocks, threads>>>(
      static_cast<const double *>(coord), nParticles, stride, coordinateIndex,
      minBunch, nBuckets, deviceStart, deviceCount);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, startEvent, stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto cleanup;
  }
  cudaStatus = cudaMemcpy(&hostSorted, deviceSorted, sizeof(hostSorted),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(start, deviceStart,
                            nBuckets * sizeof(*deviceStart),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(count, deviceCount,
                            nBuckets * sizeof(*deviceCount),
                            cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    *sorted = hostSorted ? 1 : 0;

cleanup:
  return static_cast<int>(cudaStatus);
}

/* A bunched beam whose rows aren't in bunch order (after losses or an injection) for ZLONGIT and
 * ZTRANSVERSE: a stable sort by bunch number, which keeps particle order inside each bunch as
 * index_bunch_assignments()'s bucket lists do, the rows gathered in that order, each bunch's range,
 * and later the rows written back where they came from */
typedef struct {
  unsigned int *keysIn, *keysOut;
  long *indexIn, *indexOut;
  void *temp;
  size_t tempBytes;
  double *gathered;
  long *start, *count;
  long particleCapacity, gatheredCapacity, bucketCapacity;
  long particles;
  int stride;
} GPU_BUNCH_GATHER_SCRATCH;

static GPU_BUNCH_GATHER_SCRATCH gpuBunchGatherScratch;

extern "C" void gpuCudaBunchGatherRelease(void) {
  gpuFreeDevice(gpuBunchGatherScratch.keysIn);
  gpuFreeDevice(gpuBunchGatherScratch.keysOut);
  gpuFreeDevice(gpuBunchGatherScratch.indexIn);
  gpuFreeDevice(gpuBunchGatherScratch.indexOut);
  gpuFreeDevice(gpuBunchGatherScratch.temp);
  gpuFreeDevice(gpuBunchGatherScratch.gathered);
  gpuFreeDevice(gpuBunchGatherScratch.start);
  gpuFreeDevice(gpuBunchGatherScratch.count);
  std::memset(&gpuBunchGatherScratch, 0, sizeof(gpuBunchGatherScratch));
}

__global__ void gpuBunchGatherKeysKernel(const double *coord, long nParticles, int stride,
                                         int coordinateIndex, long minBunch,
                                         unsigned int *keys, long *index) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  keys[ip] = static_cast<unsigned int>(
    static_cast<long>(coord[ip * stride + coordinateIndex]) - minBunch);
  index[ip] = ip;
}

__global__ void gpuBunchGatherRowsKernel(const double *coord, double *gathered, const long *index,
                                         long nParticles, int stride) {
  long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *source;
  double *target;

  if (row >= nParticles)
    return;
  source = coord + index[row] * stride;
  target = gathered + row * stride;
  for (int j = 0; j < stride; j++)
    target[j] = source[j];
}

__global__ void gpuBunchScatterRowsKernel(double *coord, const double *gathered, const long *index,
                                          long nParticles, int stride) {
  long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *source;
  double *target;

  if (row >= nParticles)
    return;
  source = gathered + row * stride;
  target = coord + index[row] * stride;
  for (int j = 0; j < stride; j++)
    target[j] = source[j];
}

/* Every bunch number must lie in [minBunch, minBunch + nBuckets), as gpuCudaLongMinMax found it */
extern "C" int gpuCudaGatherByBunch(void *coord, long nParticles, int stride, int coordinateIndex,
                                    long minBunch, long nBuckets, void **gatheredReturn,
                                    long *start, long *count, float *milliseconds) {
  GPU_BUNCH_GATHER_SCRATCH *s = &gpuBunchGatherScratch;
  cudaEvent_t startEvent, stopEvent;
  cudaError_t cudaStatus = cudaSuccess;
  size_t sortBytes = 0;
  int threads = 256, blocks, bucketBlocks, endBit = 0, status;

  if (!coord || nParticles <= 0 || nParticles > INT_MAX || stride <= 0 || coordinateIndex < 0 ||
      coordinateIndex >= stride || nBuckets <= 0 || nBuckets > (long)UINT_MAX || !gatheredReturn ||
      !start || !count)
    return static_cast<int>(cudaErrorInvalidValue);
  if (s->particleCapacity < nParticles) {
    gpuFreeDevice(s->keysIn);
    gpuFreeDevice(s->keysOut);
    gpuFreeDevice(s->indexIn);
    gpuFreeDevice(s->indexOut);
    s->keysIn = s->keysOut = NULL;
    s->indexIn = s->indexOut = NULL;
    s->particleCapacity = 0;
    cudaStatus = cudaMalloc(&s->keysIn, nParticles * sizeof(*s->keysIn));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&s->keysOut, nParticles * sizeof(*s->keysOut));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&s->indexIn, nParticles * sizeof(*s->indexIn));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&s->indexOut, nParticles * sizeof(*s->indexOut));
    if (cudaStatus != cudaSuccess) {
      gpuCudaBunchGatherRelease();
      return static_cast<int>(cudaStatus);
    }
    s->particleCapacity = nParticles;
  }
  if (s->gatheredCapacity < nParticles * stride) {
    gpuFreeDevice(s->gathered);
    s->gathered = NULL;
    s->gatheredCapacity = 0;
    cudaStatus = cudaMalloc(&s->gathered, nParticles * stride * sizeof(*s->gathered));
    if (cudaStatus != cudaSuccess) {
      gpuCudaBunchGatherRelease();
      return static_cast<int>(cudaStatus);
    }
    s->gatheredCapacity = nParticles * stride;
  }
  if (s->bucketCapacity < nBuckets) {
    gpuFreeDevice(s->start);
    gpuFreeDevice(s->count);
    s->start = s->count = NULL;
    s->bucketCapacity = 0;
    cudaStatus = cudaMalloc(&s->start, nBuckets * sizeof(*s->start));
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMalloc(&s->count, nBuckets * sizeof(*s->count));
    if (cudaStatus != cudaSuccess) {
      gpuCudaBunchGatherRelease();
      return static_cast<int>(cudaStatus);
    }
    s->bucketCapacity = nBuckets;
  }
  cudaStatus = cub::DeviceRadixSort::SortPairs(NULL, sortBytes, s->keysIn, s->keysOut, s->indexIn,
                                               s->indexOut, static_cast<int>(nParticles));
  if (cudaStatus == cudaSuccess && s->tempBytes < sortBytes) {
    gpuFreeDevice(s->temp);
    s->temp = NULL;
    s->tempBytes = 0;
    cudaStatus = cudaMalloc(&s->temp, sortBytes);
    if (cudaStatus == cudaSuccess)
      s->tempBytes = sortBytes;
  }
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  while (endBit < 32 && (1UL << endBit) < static_cast<unsigned long>(nBuckets))
    endBit++;
  if (endBit == 0)
    endBit = 1;

  status = prepareTimedLaunch(&startEvent, &stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  bucketBlocks = static_cast<int>((nBuckets + threads - 1) / threads);
  gpuBunchGatherKeysKernel<<<blocks, threads>>>(static_cast<const double *>(coord), nParticles,
                                                stride, coordinateIndex, minBunch, s->keysIn,
                                                s->indexIn);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    size_t bytes = s->tempBytes;
    /* LSD radix sort: stable, so rows of one bunch keep their order */
    cudaStatus = cub::DeviceRadixSort::SortPairs(s->temp, bytes, s->keysIn, s->keysOut,
                                                 s->indexIn, s->indexOut,
                                                 static_cast<int>(nParticles), 0, endBit);
  }
  if (cudaStatus == cudaSuccess) {
    gpuBunchGatherRowsKernel<<<blocks, threads>>>(static_cast<const double *>(coord), s->gathered,
                                                  s->indexOut, nParticles, stride);
    cudaStatus = cudaGetLastError();
  }
  if (cudaStatus == cudaSuccess) {
    gpuSortedBunchRangesKernel<<<bucketBlocks, threads>>>(s->gathered, nParticles, stride,
                                                          coordinateIndex, minBunch, nBuckets,
                                                          s->start, s->count);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, startEvent, stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(start, s->start, nBuckets * sizeof(*start), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(count, s->count, nBuckets * sizeof(*count), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  s->particles = nParticles;
  s->stride = stride;
  *gatheredReturn = s->gathered;
  return static_cast<int>(cudaSuccess);
}

/* The gathered rows back to the places gpuCudaGatherByBunch() took them from */
extern "C" int gpuCudaScatterByBunch(void *coord, long nParticles, int stride, float *milliseconds) {
  GPU_BUNCH_GATHER_SCRATCH *s = &gpuBunchGatherScratch;
  cudaEvent_t startEvent, stopEvent;
  cudaError_t cudaStatus;
  int threads = 256, blocks, status;

  if (!coord || nParticles <= 0 || nParticles != s->particles || stride != s->stride ||
      !s->gathered || !s->indexOut)
    return static_cast<int>(cudaErrorInvalidValue);
  status = prepareTimedLaunch(&startEvent, &stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  gpuBunchScatterRowsKernel<<<blocks, threads>>>(static_cast<double *>(coord), s->gathered,
                                                 s->indexOut, nParticles, stride);
  cudaStatus = cudaGetLastError();
  s->particles = 0;
  return launchTimedKernel(cudaStatus, startEvent, stopEvent, milliseconds);
}

/* IBSCATTER (November 14, 2026).  The host makes slicebeam()'s slices and computeSliceParameters()'s
 * sums from a copy of the rows (an ordered sum is one chain of dependent additions, about 28 ns each on
 * this device and 1 ns on the host); the device does the inflation of inflateEmittance() and
 * inflateEmittanceZ(), each row with its slice's parameters. */
#define GPU_IBS_SLICE_PARAMETERS 11

typedef struct {
  unsigned int *slice;
  double *parameters;
  long particleCapacity, sliceCapacity;
} GPU_IBS_SCRATCH;

static GPU_IBS_SCRATCH gpuIbsScratch;

extern "C" void gpuCudaIbsRelease(void) {
  gpuFreeDevice(gpuIbsScratch.slice);
  gpuFreeDevice(gpuIbsScratch.parameters);
  std::memset(&gpuIbsScratch, 0, sizeof(gpuIbsScratch));
}

/* computeSliceParameters() and inflateEmittanceZ(): P = Po*(1+delta), beta = P/sqrt(P*P+1), t = s/(beta*c) */
__device__ __forceinline__ double gpuIbsBetaTime(const double *part, double pCentral, double cMks) {
  double P = pCentral * (1 + part[5]);
  double beta = P / sqrt(P * P + 1);
  return part[4] / (beta * cMks);
}

/* inflateEmittance() in x and in y, then inflateEmittanceZ(), for one row.  Per slice: the means of x,
 * x', y, y', delta and t, the square roots of the two factors, and zRate[3] */
__global__ void gpuIbsInflateKernel(double *coord, long nParticles, int stride,
                                    const unsigned int *sliceOfRow, const double *parameters,
                                    double eta0, double eta1, double eta2, double eta3,
                                    double pCentral, double cMks, double dtSlice, double duration) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *a;
  double *part, cp0, cp1, time, c0, p, beta0, beta1;

  if (ip >= nParticles)
    return;
  a = parameters + (sliceOfRow ? static_cast<long>(sliceOfRow[ip]) : 0) * GPU_IBS_SLICE_PARAMETERS;
  part = coord + ip * stride;
  /* a: 0 x, 1 x', 2 y, 3 y', 4 delta, 5 t (means), 6 sqrt(x factor), 7 sqrt(y factor), 8-10 zRate */
  cp0 = eta0 * a[4];
  cp1 = eta1 * a[4];
  part[0] = (part[0] - a[0] - cp0) * a[6] + a[0] + cp0;
  part[1] = (part[1] - a[1] - cp1) * a[6] + a[1] + cp1;
  cp0 = eta2 * a[4];
  cp1 = eta3 * a[4];
  part[2] = (part[2] - a[2] - cp0) * a[7] + a[2] + cp0;
  part[3] = (part[3] - a[3] - cp1) * a[7] + a[3] + cp1;
  time = gpuIbsBetaTime(part, pCentral, cMks);
  if (time > a[5])
    c0 = (time - a[5]) / dtSlice * (a[10] - a[9]) + a[9];
  else
    c0 = (time - a[5]) / dtSlice * (a[9] - a[8]) + a[9];
  p = pCentral * (1 + part[5]);
  beta0 = p / sqrt(p * p + 1);
  part[5] = (part[5] - a[4]) * c0 + a[4];
  p = pCentral * (1 + part[5]);
  beta1 = p / sqrt(p * p + 1);
  part[4] += (beta1 - beta0) * duration * cMks / 2;
}

/* sliceOfRow (host, NULL with one slice): each row's slice; parameters (host): GPU_IBS_SLICE_PARAMETERS
 * per slice */
extern "C" int gpuCudaIbsInflate(void *coord, long nParticles, int stride, long nslice,
                                 const unsigned int *sliceOfRow, const double *parameters,
                                 const double *eta, double pCentral, double cMks, double dtSlice,
                                 double duration, float *milliseconds) {
  GPU_IBS_SCRATCH *s = &gpuIbsScratch;
  cudaEvent_t startEvent, stopEvent;
  cudaError_t cudaStatus = cudaSuccess;
  int threads = 256, status;

  if (!coord || nParticles <= 0 || stride < 6 || nslice < 1 || !parameters || !eta)
    return static_cast<int>(cudaErrorInvalidValue);
  if (sliceOfRow && s->particleCapacity < nParticles) {
    gpuFreeDevice(s->slice);
    s->slice = NULL;
    s->particleCapacity = 0;
    cudaStatus = cudaMalloc(&s->slice, nParticles * sizeof(*s->slice));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    s->particleCapacity = nParticles;
  }
  if (s->sliceCapacity < nslice) {
    gpuFreeDevice(s->parameters);
    s->parameters = NULL;
    s->sliceCapacity = 0;
    cudaStatus = cudaMalloc(&s->parameters, GPU_IBS_SLICE_PARAMETERS * nslice * sizeof(*s->parameters));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    s->sliceCapacity = nslice;
  }
  if (sliceOfRow)
    cudaStatus = cudaMemcpy(s->slice, sliceOfRow, nParticles * sizeof(*sliceOfRow),
                            cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(s->parameters, parameters,
                            GPU_IBS_SLICE_PARAMETERS * nslice * sizeof(*parameters),
                            cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&startEvent, &stopEvent, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuIbsInflateKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, sliceOfRow ? s->slice : NULL, s->parameters,
    eta[0], eta[1], eta[2], eta[3], pCentral, cMks, dtSlice, duration);
  cudaStatus = cudaGetLastError();
  return launchTimedKernel(cudaStatus, startEvent, stopEvent, milliseconds);
}

extern "C" int gpuCudaDoubleMinMax(void *coord, long nParticles, int stride,
                                   int coordinateIndex,
                                   GPU_DOUBLE_MIN_MAX_DATA *result,
                                   void *deviceResultScratch,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_DOUBLE_MIN_MAX_DATA *deviceResult = NULL;
  int status;

  if (!result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  if (!coord || nParticles <= 0 || coordinateIndex < 0 || coordinateIndex >= stride)
    return static_cast<int>(cudaErrorInvalidValue);

  if (deviceResultScratch) {
    deviceResult = static_cast<GPU_DOUBLE_MIN_MAX_DATA *>(deviceResultScratch);
  } else {
    cudaStatus = gpuReuseDeviceBuffer(0, &deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuDoubleMinMaxKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord),
                                                      nParticles, stride,
                                                      coordinateIndex,
                                                      deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                              double xmax, double ymax, int openCode,
                                              long *lostCount, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuLimitAmplitudeLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLimitAmplitudesStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, int openCode, double z, double pCentral, long extrapolateZ,
  long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuLimitAmplitudeSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax, openCode,
    devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuLimitAmplitudesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, xmax, ymax, openCode, z, pCentral,
      extrapolateZ, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaELimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                               double xmax, double ymax,
                                               long exponent, long yExponent, int openCode,
                                               long *lostCount, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuELimitAmplitudeLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    exponent, yExponent, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaELimitAmplitudesStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, long exponent, long yExponent, int openCode, double z,
  double pCentral, long extrapolateZ, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuELimitAmplitudeSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    exponent, yExponent, openCode, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuELimitAmplitudesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, xmax, ymax,
      exponent, yExponent, openCode, z, pCentral, extrapolateZ, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaRemoveInvalidParticlesLossCount(void *coord,
                                                      long nParticles,
                                                      int stride,
                                                      long *lostCount,
                                                      float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRemoveInvalidParticlesLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaRemoveInvalidParticlesStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles,
  int stride, double z, double pCentral, long *remaining,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuRemoveInvalidParticlesSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuRemoveInvalidParticlesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaRectangularCollimatorLossCount(void *coord, long nParticles, int stride,
                                                     double xmax, double ymax,
                                                     double xCenter, double yCenter,
                                                     double length, long openCode,
                                                     long *lostCount,
                                                     float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRectangularCollimatorLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, length, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaRectangularCollimatorStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, double xCenter, double yCenter, double length,
  long openCode, double z, double pCentral, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuRectangularCollimatorSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, length, openCode, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess) {
      gpuCollimatorLossOrderKernel<false><<<1, 1>>>(
        static_cast<double *>(coord), devicePrefix, nParticles, stride,
        xmax, ymax, xCenter, yCenter, length, openCode, 0, 0);
      cudaStatus = cudaGetLastError();
    }
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuRectangularCollimatorStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, xmax, ymax,
      xCenter, yCenter, length, openCode, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaEllipticalCollimatorLossCount(void *coord, long nParticles, int stride,
                                                    double xmax, double ymax,
                                                    double xCenter, double yCenter,
                                                    long exponent, long yExponent,
                                                    double length,
                                                    long openCode,
                                                    long *lostCount, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuEllipticalCollimatorLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, exponent, yExponent, length, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaEllipticalCollimatorStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, double xCenter, double yCenter, long exponent,
  long yExponent, double length, long openCode, double z, double pCentral, long *remaining,
  float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuEllipticalCollimatorSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, exponent, yExponent, length, openCode, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess) {
      gpuCollimatorLossOrderKernel<true><<<1, 1>>>(
        static_cast<double *>(coord), devicePrefix, nParticles, stride,
        xmax, ymax, xCenter, yCenter, length, openCode, exponent, yExponent);
      cudaStatus = cudaGetLastError();
    }
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuEllipticalCollimatorStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, xmax, ymax,
      xCenter, yCenter, exponent, yExponent, length, openCode,
      z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaScraperLossCount(void *coord, long nParticles, int stride,
                                       int plane, double center, double position,
                                       int sideSign, double length, long *lostCount,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount || (plane != 0 && plane != 2) || (sideSign != 1 && sideSign != -1))
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuScraperLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, plane, center, position,
    sideSign, length, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaScraperStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  int plane, double center, double position, int sideSign, double length,
  double z, double pCentral, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  int *lossRound = NULL;
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0 ||
      (plane != 0 && plane != 2) || (sideSign != 1 && sideSign != -1))
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  cudaStatus = gpuRoundLossRounds(nParticles, &lossRound);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuScraperSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, plane, center,
    position, sideSign, length, devicePrefix, lossRound);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuRoundLossOrderDestination(
        devicePrefix + nParticles, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuScraperStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, plane, center, position,
      sideSign, length, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaApContourLossCount(void *coord, long nParticles, int stride,
                                         const GPU_APERTURE_CONTOUR_DATA *contour, double length,
                                         long *lostCount, float *milliseconds) {
  static unsigned long long *deviceLostCount = NULL;
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long hostLostCount = 0;
  const int blockSize = 256;
  int status;

  if (!lostCount || !contour)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!deviceLostCount) {
    cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuApContourLossCountKernel<<<static_cast<int>((nParticles + blockSize - 1) / blockSize), blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *contour, length, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount, sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  *lostCount = (long)hostLostCount;
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaApertureDataLossCount(void *coord, long nParticles,
                                            int stride, double xCenter,
                                            double yCenter, double xSize,
                                            double ySize, long *lostCount,
                                            float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int status;

  if (!lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  cudaStatus = gpuReuseDeviceBuffer(0, &deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuApertureDataLossCountKernel<<<gpuLossCountBlocks(nParticles), GPU_LOSS_COUNT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xCenter, yCenter,
    xSize, ySize, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                          cudaMemcpyDeviceToHost);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaApertureDataStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xCenter, double yCenter, double xSize, double ySize, double z,
  double pCentral, long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuApertureDataSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, xCenter, yCenter,
    xSize, ySize, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(
        devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuApertureDataStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride,
      xCenter, yCenter, xSize, ySize,
      z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(
        devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaApContourStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_APERTURE_CONTOUR_DATA *contour, double z, double pCentral,
  long *remaining, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long survivors;
  long *devicePrefix = static_cast<long *>(prefix);
  thrust::device_ptr<long> flags(devicePrefix);
  const int blockSize = 256;
  int gridSize;
  int status;

  if (!remaining || !contour)
    return static_cast<int>(cudaErrorInvalidValue);
  *remaining = nParticles;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  if (!coord || !scratchCoord || !prefix || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuApContourSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *contour, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuBuildCpuLossOrderDestination(devicePrefix, nParticles, survivors);
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuApContourStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix + nParticles, nParticles, stride, *contour, z, pCentral);
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess)
      cudaStatus = gpuPublishCpuLossOrderDestination(devicePrefix, nParticles);
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaMemoryAvailable(unsigned long long *bytes) {
  size_t available = 0, total = 0;
  cudaError_t status = cudaMemGetInfo(&available, &total);
  *bytes = available;
  return (int)status;
}

/* Element types moved to the device on November 15, 2026 */
#include "gpu_ports_device.h"
