#include "gpu_base.h"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <limits.h>

#define GPU_MATRIX_CACHE_SLOTS 6
__constant__ GPU_MATRIX_DATA gpuMatrixData[GPU_MATRIX_CACHE_SLOTS];
__constant__ GPU_MULTIPOLE_DATA gpuMultipoleData;
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

__global__ void gpuBeamStatisticsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  GPU_BEAM_SUM_DATA *partial, double *timeValue);

typedef struct GPU_GENERIC_REDUCTION_SCRATCH {
  GPU_BEAM_SUM_DATA *partial;
  GPU_BEAM_SUM_DATA *result;
} GPU_GENERIC_REDUCTION_SCRATCH;

static GPU_GENERIC_REDUCTION_SCRATCH gpuGenericReductionScratch;

static void releaseGenericReductionScratch(void) {
  cudaFree(gpuGenericReductionScratch.partial);
  cudaFree(gpuGenericReductionScratch.result);
  std::memset(&gpuGenericReductionScratch, 0,
              sizeof(gpuGenericReductionScratch));
}

extern "C" void gpuCudaReductionRelease(void) {
  releaseGenericReductionScratch();
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
  long count;
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

typedef struct GPU_BMXYZ_SCRATCH {
  double *coordBackup;
  long coordCapacity;
  unsigned long long *failedCount;
} GPU_BMXYZ_SCRATCH;

static GPU_BMXYZ_SCRATCH gpuBmxyzScratch;

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

typedef struct GPU_HISTOGRAM_SCRATCH {
  GPU_HISTOGRAM_RANGE_DATA *rangePartial;
  GPU_HISTOGRAM_RANGE_DATA *rangeResult;
  unsigned long long *histogramPartial;
  unsigned long long *histogram;
  long binCapacity;
} GPU_HISTOGRAM_SCRATCH;

typedef struct GPU_HISTOGRAM_BIN_DATA {
  long bins;
  long startPID;
  long endPID;
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
  cudaFree(gpuScmultMomentResult);
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
  cudaFree(gpuHistogramScratch.rangePartial);
  cudaFree(gpuHistogramScratch.rangeResult);
  cudaFree(gpuHistogramScratch.histogramPartial);
  cudaFree(gpuHistogramScratch.histogram);
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
    cudaFree(gpuHistogramScratch.histogramPartial);
    cudaFree(gpuHistogramScratch.histogram);
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
  cudaFree(gpuRfmodeScratch.time);
  cudaFree(gpuRfmodeScratch.pbin);
  cudaFree(gpuRfmodeScratch.histogram);
  cudaFree(gpuRfmodeScratch.xsum);
  cudaFree(gpuRfmodeScratch.ysum);
  cudaFree(gpuRfmodeScratch.voltage);
  cudaFree(gpuRfmodeScratch.binnedCount);
  std::memset(&gpuRfmodeScratch, 0, sizeof(gpuRfmodeScratch));
}

extern "C" void gpuCudaRfmodeRelease(void) {
  releaseRfmodeScratch();
}

static int ensureRfmodeParticleScratch(long particles) {
  cudaError_t status;

  if (particles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!gpuRfmodeScratch.time || !gpuRfmodeScratch.pbin ||
      gpuRfmodeScratch.particleCapacity < particles) {
    cudaFree(gpuRfmodeScratch.time);
    cudaFree(gpuRfmodeScratch.pbin);
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
    cudaFree(gpuRfmodeScratch.histogram);
    cudaFree(gpuRfmodeScratch.xsum);
    cudaFree(gpuRfmodeScratch.ysum);
    cudaFree(gpuRfmodeScratch.voltage);
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
    cudaFree(gpuFtableScratch.field[field]);
  std::memset(&gpuFtableScratch, 0, sizeof(gpuFtableScratch));
}

extern "C" void gpuCudaFtableRelease(void) {
  releaseFtableScratch();
}

static void releaseBmxyzScratch(void) {
  cudaFree(gpuBmxyzScratch.coordBackup);
  cudaFree(gpuBmxyzScratch.failedCount);
  std::memset(&gpuBmxyzScratch, 0, sizeof(gpuBmxyzScratch));
}

extern "C" void gpuCudaBmxyzRelease(void) {
  releaseBmxyzScratch();
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
    cudaFree(gpuBmxyzScratch.coordBackup);
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

static void releaseBggexpScratch(void) {
  cudaFree(gpuBggexpScratch.Cmn);
  cudaFree(gpuBggexpScratch.dCmnDz);
  cudaFree(gpuBggexpScratch.coefficient);
  cudaFree(gpuBggexpScratch.multipoleFactor);
  cudaFree(gpuBggexpScratch.m);
  cudaFree(gpuBggexpScratch.gradient);
  cudaFree(gpuBggexpScratch.radialPower);
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
    cudaFree(gpuPolynomialSeriesScratch.coefficient);
  if (gpuPolynomialSeriesScratch.exponent)
    cudaFree(gpuPolynomialSeriesScratch.exponent);
  if (gpuPolynomialSeriesScratch.backup)
    cudaFree(gpuPolynomialSeriesScratch.backup);
  if (gpuPolynomialSeriesScratch.invalidCount)
    cudaFree(gpuPolynomialSeriesScratch.invalidCount);
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
  cudaFree(gpuCombinedWakeScratch.time);
  cudaFree(gpuCombinedWakeScratch.pz);
  cudaFree(gpuCombinedWakeScratch.pbin);
  cudaFree(gpuCombinedWakeScratch.histogram);
  cudaFree(gpuCombinedWakeScratch.histogramPartial);
  cudaFree(gpuCombinedWakeScratch.voltage);
  cudaFree(gpuCombinedWakeScratch.table);
  cudaFree(gpuCombinedWakeScratch.fftReal);
  cudaFree(gpuCombinedWakeScratch.driverFrequency);
  cudaFree(gpuCombinedWakeScratch.tableFrequency);
  cudaFree(gpuCombinedWakeScratch.channelFrequency);
  cudaFree(gpuCombinedWakeScratch.binnedCount);
  cudaFree(gpuCombinedWakeScratch.driverFirstBin);
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
      gpuCombinedWakeScratch.histogramPartials == histogramPartials)
    return static_cast<int>(cudaSuccess);

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

__device__ __forceinline__ void gpuApplyPackedMatrix(double *part,
                                                     int matrixSlot) {
  const GPU_MATRIX_DATA *matrix = gpuMatrixData + matrixSlot;
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
        double value = term->coefficient * ini[term->j];
        if (term->degree >= 2)
          value *= ini[term->k];
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

__global__ void gpuTrackParticlesKernel(double *coord, long nParticles,
                                        int stride, int matrixSlot) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void gpuExactDriftKernel(double *coord, long nParticles, int stride, double length) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double xp, yp;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  xp = part[1];
  yp = part[3];
  part[0] = gpuAddRn(part[0], gpuMulRn(xp, length));
  part[2] = gpuAddRn(part[2], gpuMulRn(yp, length));
  part[4] = gpuAddRn(part[4],
                     gpuMulRn(length, gpuExactPathLengthFactor(xp, yp)));
}

__global__ void gpuLinearDriftKernel(double *coord, long nParticles, int stride, double length) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  const double *ypMap, int writeOutput) {
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

    if (!gpuKickMapInterpolate(&dxp, &dyp, xpMap, ypMap, map, x, y))
      return 0;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
                                             long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuKickMapTrackParticle(coord + ip * stride, &map, xpMap, ypMap, 0) ? 1 : 0;
}

__global__ void gpuKickMapStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors, GPU_KICKMAP_DATA map,
  const double *xpMap, const double *ypMap, double zStart, double pRef) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, destination;
  int survives;

  if (ip >= nParticles)
    return;
  prefix = survivorPrefix[ip];
  survives = (ip + 1 < nParticles) ? survivorPrefix[ip + 1] > prefix :
                                     survivors > prefix;
  destination = survives ? prefix : survivors + (ip - prefix);
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
    double factor = (1 + delta) / sqrt(1 + xp * xp + yp * yp);
    *qx = xp * factor;
    *qy = yp * factor;
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

template <bool Radiation>
__device__ int gpuMultipoleTrackParticle(double *part, int stride,
                                         int writeOutput) {
  const GPU_MULTIPOLE_DATA *data = &gpuMultipoleData;
  double driftFrac[8];
  double kickFrac[8];
  double KnLp[3];
  double xpow[4], ypow[4];
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double dp = part[5];
  double qx, qy, s = 0;
  double timeCoordinate = part[4];
  double beta0 = 0;
  double drift, xkick, ykick;
  int nSubsteps = 0;
  int maxOrder = 0;

  (void)stride;
  if (Radiation) {
    double p = data->Po * (1 + dp);
    beta0 = p / sqrt(p * p + 1);
  }
  if (data->dx || data->dy || data->dz) {
    if (Radiation)
      timeCoordinate += data->dz * sqrt(1 + xp * xp + yp * yp);
    else
      s += data->dz * sqrt(1 + xp * xp + yp * yp);
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
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
    return 0;

  if (!gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return 0;
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

  for (long iKick = 0; iKick < data->nSlices; iKick++) {
    double deltaQx = 0;
    double deltaQy = 0;

    for (int step = 0; step < nSubsteps; step++) {
      double dsh;

      if (drift) {
        dsh = drift * driftFrac[step];
        x += xp * dsh;
        y += yp * dsh;
        if (data->expandHamiltonian)
          s += dsh * (1 + (xp * xp + yp * yp) / 2);
        else
          s += dsh * sqrt(1 + xp * xp + yp * yp);
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

      if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                              data->expandHamiltonian))
        return 0;

      if (data->radiationBlock && drift) {
        double onePlusDp = 1 + dp;

        qx /= (1 + dp);
        qy /= (1 + dp);
        if (Radiation) {
          double deltaFactor = onePlusDp * onePlusDp;
          double normalizedDeltaQx = deltaQx / kickFrac[step];
          double normalizedDeltaQy = deltaQy / kickFrac[step];
          double F2 =
            (normalizedDeltaQx / drift - xkick / drift) *
              (normalizedDeltaQx / drift - xkick / drift) +
            (normalizedDeltaQy / drift + ykick / drift) *
              (normalizedDeltaQy / drift + ykick / drift);
          double dsFactor =
            sqrt(1 + xp * xp + yp * yp) * drift * kickFrac[step];

          dp -= data->radCoef * deltaFactor * F2 * dsFactor;
        }
        qx *= (1 + dp);
        qy *= (1 + dp);
        if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                                data->expandHamiltonian))
          return 0;
      }
    }
  }

  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;

  if (Radiation) {
    double p = data->Po * (1 + dp);
    double beta1 = p / sqrt(p * p + 1);

    timeCoordinate =
      beta1 * (timeCoordinate / beta0 + 2 * s / (beta0 + beta1));
  }

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
  if (data->dx || data->dy || data->dz) {
    if (Radiation)
      timeCoordinate -= data->dz * sqrt(1 + xp * xp + yp * yp);
    else
      s -= data->dz * sqrt(1 + xp * xp + yp * yp);
    x = x + data->dx - data->dz * xp;
    y = y + data->dy - data->dz * yp;
  }

  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    if (Radiation)
      part[4] = timeCoordinate;
    else
      part[4] += s;
    part[5] = dp;
  }
  return 1;
}

template <bool Radiation>
__global__ void gpuMultipolePredicateKernel(double *coord, long nParticles,
                                            int stride, long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;

  for (long ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    if (!gpuMultipoleTrackParticle<Radiation>(part, stride, 0))
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

template <bool Radiation>
__global__ void gpuMultipoleTrackKernel(double *coord, long nParticles,
                                        int stride) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  gpuMultipoleTrackParticle<Radiation>(coord + ip * stride, stride, 1);
}

template <bool Radiation>
__global__ void gpuMultipoleTrackCheckedKernel(
  double *coord, long nParticles, int stride,
  unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuMultipoleTrackParticle<Radiation>(
          coord + ip * stride, stride, 1))
      localCount = 1;
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

template <bool Radiation>
__global__ void gpuMultipoleSurvivorFlagKernel(
  double *coord, long nParticles, int stride, long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuMultipoleTrackParticle<Radiation>(
      coord + ip * stride, stride, 0) ? 1 : 0;
}

template <bool Radiation>
__global__ void gpuMultipoleStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, destination;
  int survives;

  if (ip >= nParticles)
    return;
  prefix = survivorPrefix[ip];
  survives = (ip + 1 < nParticles) ? survivorPrefix[ip + 1] > prefix :
                                     survivors > prefix;
  destination = survives ? prefix : survivors + (ip - prefix);
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (survives)
    gpuMultipoleTrackParticle<Radiation>(target, stride, 1);
}

__device__ __forceinline__ int gpuCcbendSwitchPlane(
  double *x, double *xp, double *y, double *yp, double *s,
  double dp, double angle) {
  double tanAngle = tan(angle);
  double cosAngle = cos(angle);
  double sinAngle = sin(angle);
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

__device__ int gpuCcbendTrackParticle(double *part, int stride,
                                      GPU_CCBEND_DATA data,
                                      int writeOutput) {
  double x = part[0];
  double xp = part[1];
  double y = part[2];
  double yp = part[3];
  double path = part[4];
  double dp = part[5];
  double qx, qy, denominator, bodyPath = 0;
  double xpow[3], ypow[3];
  double halfDrift = data.chordLength / data.nSlices * 0.5;

  (void)stride;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(path) || !isfinite(dp))
    return 0;
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.angleHalf))
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
  }

  denominator = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
  if (denominator <= 0)
    return 0;
  denominator = sqrt(denominator);
  xp = qx / denominator;
  yp = qy / denominator;
  path += bodyPath;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp) ||
      fabs(x) > data.coordLimit || fabs(y) > data.coordLimit ||
      fabs(xp) > data.slopeLimit || fabs(yp) > data.slopeLimit)
    return 0;

  x -= data.xAdjust;
  if (!gpuCcbendSwitchPlane(&x, &xp, &y, &yp, &path, dp,
                             data.angleHalf))
    return 0;
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

__global__ void gpuCcbendTrackCheckedKernel(
  double *coord, long nParticles, int stride, GPU_CCBEND_DATA data,
  unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles &&
      !gpuCcbendTrackParticle(coord + ip * stride, stride, data, 1))
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

__global__ void gpuAddCoordinateKernel(double *coord, long nParticles, int stride,
                                       int index, double value) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  coord[ip * stride + index] += value;
}

__global__ void gpuOffsetBeamKernel(double *coord, long nParticles, int stride,
                                    double dx, double dxp, double dy, double dyp,
                                    double dz, double dt, double dp, double de,
                                    double pCentral, long startPID, long endPID,
                                    int allParticles, double cMks) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  if (turn < 0 || turn >= turns)
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  part[5] = ((1 + part[5]) * oldP - newP) / newP;
}

__global__ void gpuMatchEnergyKernel(double *coord, long nParticles, int stride,
                                     double oldP, double averageP, int changeBeam) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  __shared__ long count[GPU_REDUCTION_THREADS];
  __shared__ double sum[GPU_REDUCTION_THREADS];
  __shared__ double error[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  sum[tid] = 0;
  error[tid] = 0;
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gpuRfcaThinKickKernel(double *coord, long nParticles, int stride,
                                      double pCentral, double volt,
                                      double omega, double phase,
                                      double cMks) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  dgamma = volt * sin(omega * t + phase);
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

__global__ void gpuTfeedbackKickKernel(double *coord, long nParticles,
                                       int stride, int pickupCoordinate,
                                       int longitudinal, double kick) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

    part[1] = (part[1] - xpEta) * data.Fx + xpEta;
    ypEta = part[5] * data.etapy;
    part[3] = (part[3] - ypEta) * data.Fy + ypEta;
    deltaChange = -part[5];
    part[5] = data.Ddelta + part[5] * data.Fdelta;
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
} GPU_BGGEXP_DEVICE_DATA;

__device__ double gpuBggexpIntegerPower(double value, int exponent) {
  double power;
  int exponentStack[32];
  int depth = 0;

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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

__device__ __forceinline__ void gpuCwigglerAx(
  double *ax, double *axpy, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  *ax = 0.0;
  *axpy = 0.0;
  if (data.hasHorizontal && z >= data.zStartHorizontal &&
      z <= data.zEndHorizontal) {
    double coefficient = data.horizontalCoefficient * poleFactor;
    double sinz = sin(data.kw * z + data.horizontalPhase);
    *ax = coefficient * cosh(data.kw * y) * sinz;
    *axpy = coefficient * data.kw * x * sinh(data.kw * y) * sinz;
  }
}

__device__ __forceinline__ void gpuCwigglerAy(
  double *ay, double *aypx, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  *ay = 0.0;
  *aypx = 0.0;
  if (data.hasVertical && z >= data.zStartVertical &&
      z <= data.zEndVertical) {
    double coefficient = data.verticalCoefficient * poleFactor;
    double sinz = sin(data.kw * z + data.verticalPhase);
    *ay = coefficient * cosh(data.kw * x) * sinz;
    *aypx = coefficient * data.kw * sinh(data.kw * x) * y * sinz;
  }
}

__device__ __forceinline__ void gpuCwigglerField(
  double *bx, double *by, double x, double y, double z,
  double poleFactor, const GPU_CWIGGLER_DATA &data) {
  *bx = 0.0;
  *by = 0.0;
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

__device__ __forceinline__ void gpuCwigglerRadiationKick(
  double *qx, double *qy, double *delta, double x, double y, double z,
  double dl, double poleFactor, const GPU_CWIGGLER_DATA &data) {
  double ax, ay, axpy, aypx, bx, by, b2, rigidity, irho2;
  double dFactor, dDelta;

  gpuCwigglerAx(&ax, &axpy, x, y, z, poleFactor, data);
  gpuCwigglerAy(&ay, &aypx, x, y, z, poleFactor, data);
  gpuCwigglerField(&bx, &by, x, y, z, poleFactor, data);
  *qx -= ax;
  *qy -= ay;
  b2 = bx * bx + by * by;
  if (b2 != 0.0) {
    rigidity = data.pCentral / 586.679074042074490;
    irho2 = b2 / (rigidity * rigidity);
    dFactor = (1.0 + *delta) * (1.0 + *delta);
    dDelta = -data.srCoef * dFactor * irho2 * dl;
    *delta += dDelta;
    *qx *= 1.0 + dDelta;
    *qy *= 1.0 + dDelta;
  }
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
  *s += 0.5 * dl2d * *qy * *qy / (1.0 + delta);
  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx += aypx;
  *qy += ay;

  gpuCwigglerAx(&ax, &axpy, *x, *y, *z, poleFactor, data);
  *qx -= ax;
  *qy -= axpy;
  *x += dld * *qx;
  *s += dl + 0.5 * dld * *qx * *qx / (1.0 + delta);
  gpuCwigglerAx(&ax, &axpy, *x, *y, *z, poleFactor, data);
  *qx += ax;
  *qy += axpy;

  gpuCwigglerAy(&ay, &aypx, *x, *y, *z, poleFactor, data);
  *qx -= aypx;
  *qy -= ay;
  *y += dl2d * *qy;
  *s += 0.5 * dl2d * *qy * *qy / (1.0 + delta);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double x, xp, qx, y, yp, qy, s, delta, denom, z = 0.0;
  double periodLength, dl, dl1, dl0;
  long totalSteps;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  x = part[0];
  xp = part[1];
  y = part[2];
  yp = part[3];
  s = part[4];
  delta = part[5];
  denom = sqrt(1.0 + xp * xp + yp * yp);
  qx = (1.0 + delta) * xp / denom;
  qy = (1.0 + delta) * yp / denom;

  periodLength = data.periodLength;
  dl = periodLength / data.stepsPerPeriod;
  dl1 = fourthOrderX1 * dl;
  dl0 = fourthOrderX0 * dl;
  totalSteps = data.periods * data.stepsPerPeriod;
  if (data.synchRad)
    gpuCwigglerRadiationKick(&qx, &qy, &delta, x, y, z, dl,
                              data.poleFactor[0], data);
  for (long step = 1; step <= totalSteps; step++) {
    double poleFactor = gpuCwigglerPoleFactor(
      ((step - 1) * data.length) / totalSteps, data);
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
    if (data.synchRad)
      gpuCwigglerRadiationKick(&qx, &qy, &delta, x, y, z, dl,
                                poleFactor, data);
  }

  denom = sqrt((1.0 + delta) * (1.0 + delta) - qx * qx - qy * qy);
  part[0] = x;
  part[1] = qx / denom;
  part[2] = y;
  part[3] = qy / denom;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double step, sLocation;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  step = data.length / data.nKicks;
  sLocation = step / 2.0;

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
      p[0] = -p[2] * sin(theta);
      p[2] *= cos(theta);
      xyz[0] = rho * (cos(theta) - 1.0);
      xyz[1] = (p[1] / p[2]) * rho * theta;
      xyz[2] = rho * sin(theta);
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

__device__ __forceinline__ int gpuBmxyzDerivatives(
  double derivative[8], const double q[8],
  const GPU_BMXYZ_DATA &data) {
  double F0, F1, F2;

  if (!isfinite(q[1]) || !isfinite(q[2]))
    return 0;
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
  const GPU_BMXYZ_DATA &data) {
  double k1[8], k2[8], k3[8], temporary[8], derivative[8];

  for (long equation = 0; equation < 8; equation++) {
    k1[equation] = h * initialDerivative[equation];
    temporary[equation] = initialState[equation] + k1[equation] / 2.0;
  }
  if (!gpuBmxyzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++) {
    k2[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k2[equation] / 2.0;
  }
  if (!gpuBmxyzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++) {
    k3[equation] = h * derivative[equation];
    temporary[equation] = initialState[equation] + k3[equation];
  }
  if (!gpuBmxyzDerivatives(derivative, temporary, data))
    return 0;
  for (long equation = 0; equation < 8; equation++)
    finalState[equation] = initialState[equation] +
      (k1[equation] / 2.0 + k2[equation] + k3[equation] +
       h * derivative[equation] / 2.0) / 3.0;
  return 1;
}

__device__ __forceinline__ int gpuBmxyzSign(double value) {
  return value < 0.0 ? -1 : (value > 0.0 ? 1 : 0);
}

__device__ int gpuBmxyzIntegrate(double state[8],
                                 const GPU_BMXYZ_DATA &data) {
  double y0[8], y1[8], derivative0[8], derivative1[8];
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
  if (!gpuBmxyzDerivatives(derivative0, y0, data))
    return 0;
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
    if (!gpuBmxyzRk4Step(y1, y0, derivative0, hStep, data))
      return 0;
    x1 += hStep;
    if (!gpuBmxyzDerivatives(derivative1, y1, data))
      return 0;
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
    if (!gpuBmxyzRk4Step(y2, y0, derivative0, h, data))
      return 0;
    x2 += h;
    if (!gpuBmxyzDerivatives(derivative2, y2, data))
      return 0;
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

__global__ void gpuBmxyzKernel(double *coord, long nParticles, int stride,
                               GPU_BMXYZ_DATA data,
                               unsigned long long *failedCount) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double state[8];
  double *part;
  double dzds;

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
  if (!gpuBmxyzIntegrate(state, data)) {
    atomicAdd(failedCount, 1ULL);
    return;
  }
  part[0] = state[1];
  part[1] = state[4] / state[3];
  part[2] = state[2];
  part[3] = state[5] / state[3];
  part[4] = state[6];
  part[5] = state[7];
}

__global__ void gpuRfcwRfOnlyMatrixKernel(double *coord, long nParticles, int stride,
                                          double pCentral, double length,
                                          double volt, double omega,
                                          double phase, int end1Focus,
                                          int end2Focus, double dx, double dy,
                                          double cMks, long *lostCount) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  dgamma = volt * sin(omega * t + phase);

  if (end1Focus && length) {
    inverseF = dgamma / (2 * gamma * length);
    part[1] -= part[0] * inverseF;
    part[3] -= part[2] * inverseF;
  }

  gamma1 = gamma + dgamma;
  dP = sqrt(gamma1 * gamma1 - 1) - p;
  R22 = 1 / (1 + dP / p);
  if (fabs(dP / p) > 1e-14)
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

__global__ void gpuRfcwKickInitialKernel(double *coord, double *inverseF,
                                         long nParticles, int stride,
                                         double pCentral, double length,
                                         double volt, double omega,
                                         double phase, int end1Focus,
                                         double cMks) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  dgamma = volt * sin(omega * t + phase);

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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
                                                 double cMks,
                                                 GPU_BEAM_SUM_DATA *result) {
  __shared__ long count[GPU_REDUCTION_THREADS];
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
    dgamma = volt * sin(omega * t + phase);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gpuLimitAmplitudeLossCountKernel(double *coord, long nParticles, int stride,
                                                 double xmax, double ymax,
                                                 long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    int lost = 0;

    if (xmax > 0 && fabs(part[0]) > xmax)
      lost = 1;
    if (ymax > 0 && fabs(part[2]) > ymax)
      lost = 1;
    if (!isfinite(part[0]) || !isfinite(part[2]))
      lost = 1;
    localCount += lost ? 1 : 0;
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

__device__ __forceinline__ double gpuCoordinateSign(double value) {
  return value < 0 ? -1.0 : 1.0;
}

__device__ __forceinline__ int gpuLimitAmplitudeLost(double *part, double xmax,
                                                     double ymax) {
  int lost = 0;

  if (xmax > 0 && fabs(part[0]) > xmax)
    lost = 1;
  if (ymax > 0 && fabs(part[2]) > ymax)
    lost = 1;
  if (!isfinite(part[0]) || !isfinite(part[2]))
    lost = 1;
  return lost;
}

__global__ void gpuLimitAmplitudeSurvivorFlagKernel(double *coord,
                                                    long nParticles,
                                                    int stride,
                                                    double xmax,
                                                    double ymax,
                                                    long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] = gpuLimitAmplitudeLost(coord + ip * stride, xmax, ymax) ? 0 : 1;
}

__global__ void gpuLimitAmplitudesStableScatterKernel(double *coord,
                                                      double *scratch,
                                                      const long *survivorPrefix,
                                                      long nParticles,
                                                      int stride,
                                                      double xmax,
                                                      double ymax,
                                                      double z,
                                                      double pCentral,
                                                      long extrapolateZ,
                                                      long survivors) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost;
  double dz = 0;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  lost = gpuLimitAmplitudeLost(part, xmax, ymax);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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

__global__ void gpuELimitAmplitudeLossCountKernel(double *coord, long nParticles, int stride,
                                                  double xmax, double ymax,
                                                  long exponent, long yExponent,
                                                  long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  const double boundaryGuard = 1e-12;
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    int lost = 0;

    if (!isfinite(part[0]) || !isfinite(part[2])) {
      lost = 1;
    } else {
      double normalized =
        gpuIntegerPower(part[0] / xmax, exponent) +
        gpuIntegerPower(part[2] / ymax, yExponent);
      if (normalized > 1 - boundaryGuard)
        lost = 1;
    }
    localCount += lost ? 1 : 0;
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

__device__ __forceinline__ int gpuELimitAmplitudeLost(double *part,
                                                      double xmax,
                                                      double ymax,
                                                      long exponent,
                                                      long yExponent) {
  double a2, b2, normalized;

  if (!isfinite(part[0]) || !isfinite(part[2]))
    return 1;
  a2 = gpuIntegerPower(xmax, exponent);
  b2 = gpuIntegerPower(ymax, yExponent);
  normalized =
    gpuIntegerPower(part[0], exponent) / a2 +
    gpuIntegerPower(part[2], yExponent) / b2;
  return normalized > 1;
}

__global__ void gpuELimitAmplitudeSurvivorFlagKernel(double *coord,
                                                     long nParticles,
                                                     int stride,
                                                     double xmax,
                                                     double ymax,
                                                     long exponent,
                                                     long yExponent,
                                                     long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuELimitAmplitudeLost(coord + ip * stride, xmax, ymax,
                           exponent, yExponent) ? 0 : 1;
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
                                                       double z,
                                                       double pCentral,
                                                       long extrapolateZ,
                                                       long survivors) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost;
  double dz = 0;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  lost = gpuELimitAmplitudeLost(part, xmax, ymax, exponent, yExponent);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;

    localCount += gpuInvalidParticleLost(part);
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

__global__ void gpuRemoveInvalidParticlesSurvivorFlagKernel(double *coord,
                                                            long nParticles,
                                                            int stride,
                                                            long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] = gpuInvalidParticleLost(coord + ip * stride) ? 0 : 1;
}

__global__ void gpuRemoveInvalidParticlesStableScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, double z, double pCentral, long survivors) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  lost = gpuInvalidParticleLost(part);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, destination;
  int survives;

  if (ip >= nParticles)
    return;
  prefix = survivorPrefix[ip];
  survives = (ip + 1 < nParticles) ? survivorPrefix[ip + 1] > prefix :
                                     survivors > prefix;
  destination = survives ? prefix : survivors + (ip - prefix);
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
  __shared__ long partial[GPU_REDUCTION_THREADS];
  const double boundaryGuard = 1e-12;
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
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

__global__ void gpuEllipticalCollimatorLossCountKernel(double *coord, long nParticles, int stride,
                                                       double xmax, double ymax,
                                                       double xCenter, double yCenter,
                                                       long exponent, long yExponent,
                                                       double length,
                                                       long openCode,
                                                       long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  const double boundaryGuard = 1e-12;
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
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

__device__ __forceinline__ int gpuRectangularCollimatorLostAt(double *part,
                                                              double xmax,
                                                              double ymax,
                                                              double xCenter,
                                                              double yCenter,
                                                              double length,
                                                              long openCode) {
  double x = part[0] + length * part[1];
  double y = part[2] + length * part[3];
  return gpuRectangularCollimatorLostAtPosition(x, y, xmax, ymax, xCenter, yCenter,
                                                openCode, length > 0);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost, entranceLost;
  double x1, y1;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  entranceLost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, 0, openCode);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuRectangularCollimatorLostAt(part, xmax, ymax, xCenter, yCenter, length, openCode);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  double x = part[0] + length * part[1];
  double y = part[2] + length * part[3];
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost, entranceLost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  entranceLost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                               exponent, yExponent, 0, openCode);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuEllipticalCollimatorLostAt(part, xmax, ymax, xCenter, yCenter,
                                         exponent, yExponent, length, openCode);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double coordinate = part[plane] + length * part[plane + 1] - center;

    if (sideSign * coordinate > sideSign * position)
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
                                             long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  lost = gpuScraperLostAt(part, plane, center, position, sideSign, 0);
  if (!lost && length > 0)
    lost = gpuScraperLostAt(part, plane, center, position, sideSign, length);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost, entranceLost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  entranceLost = gpuScraperLostAt(part, plane, center, position, sideSign, 0);
  lost = entranceLost;
  if (!lost && length > 0)
    lost = gpuScraperLostAt(part, plane, center, position, sideSign, length);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;
  long ip;

  for (ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;

    localCount += gpuApertureDataLost(part, xCenter, yCenter, xSize, ySize);
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

__global__ void gpuApertureDataSurvivorFlagKernel(double *coord,
                                                  long nParticles,
                                                  int stride,
                                                  double xCenter,
                                                  double yCenter,
                                                  double xSize,
                                                  double ySize,
                                                  long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, lost;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  prefix = survivorPrefix[ip];
  lost = gpuApertureDataLost(part, xCenter, yCenter, xSize, ySize);
  target = scratch + (lost ? survivors + (ip - prefix) : prefix) * stride;
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
  const double *part, long startPID, long endPID) {
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
  double cMks, long startPID, long endPID,
  unsigned int coordinateMask, GPU_HISTOGRAM_RANGE_DATA *partial) {
  __shared__ long count[GPU_HISTOGRAM_THREADS];
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
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  __shared__ long count[GPU_HISTOGRAM_THREADS];
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

  for (ip = blockIdx.x * blockDim.x + threadIdx.x; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  long index = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    time[ip] = gpuParticleTime(coord + ip * stride, pCentral, cMks);
}

__global__ void gpuRfmodeHistogramKernel(
  long nParticles, GPU_RFMODE_DATA data, const double *time,
  long *pbin, unsigned long long *histogram,
  unsigned long long *binnedCount) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  long ib;

  if (ip >= nParticles)
    return;
  pbin[ip] = -1;
  ib = static_cast<long>((time[ip] - data.tmin) / data.dt);
  if (ib < 0 || ib >= data.bins)
    return;
  pbin[ip] = ib;
  atomicAdd(histogram + ib, 1ULL);
  atomicAdd(xsum + ib, coord[ip * stride] - data.dx);
  atomicAdd(ysum + ib, coord[ip * stride + 2] - data.dy);
  atomicAdd(binnedCount, 1ULL);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ib = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  __shared__ long count[GPU_REDUCTION_THREADS];
  __shared__ double sum[3][GPU_REDUCTION_THREADS];
  __shared__ double squareSum[3][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i;

  count[tid] = 0;
  for (i = 0; i < 3; i++) {
    sum[i][tid] = 0;
    squareSum[i][tid] = 0;
  }
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += blockDim.x * gridDim.x) {
    double *part = coord + ip * stride;
    double value[3] = {part[0], part[2], part[4]};

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
      partial[blockIdx.x].sum[i] = sum[i][0];
      partial[blockIdx.x].squareSum[i] = squareSum[i][0];
    }
  }
}

__global__ void gpuScmultMomentFinalizeKernel(
  const GPU_SCMULT_MOMENT_DATA *partial,
  GPU_SCMULT_MOMENT_DATA *result) {
  __shared__ long count[GPU_REDUCTION_THREADS];
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
      result->sum[i] = sum[i][0];
      result->squareSum[i] = squareSum[i][0];
    }
  }
}

__global__ void gpuScmultLinearKickKernel(double *coord, long nParticles,
                                          int stride,
                                          GPU_SCMULT_LINEAR_DATA data) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

__device__ __forceinline__ double gpuTrwakeDriveValue(double coordinate,
                                                      double offset,
                                                      long exponent) {
  double value = coordinate - offset;

  if (exponent == 1)
    return value;
  if (exponent < 1)
    return 1;
  return gpuIntegerPower(value, exponent);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  pz[ip] = wake.pCentral * (1 + part[5]) /
           sqrt(1 + part[1] * part[1] + part[3] * part[3]);
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

__global__ void gpuTrwakeConvolveKernel(double *vtime,
                                        const double *posItime,
                                        const double *wakeTable,
                                        GPU_TRWAKE_DATA wake,
                                        int plane) {
  long ib = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  long ib;
  double voltage;
  double x, xp, y, yp;

  if (ip >= nParticles)
    return;
  ib = pbin[ip];
  if (ib < 0 || ib > wake.bins - 1)
    return;
  part = coord + ip * stride;
  gpuTrwakeLocalCoordinates(part, wake, &x, &xp, &y, &yp);
  if (wake.hasWake[0] && wake.factor[0]) {
    voltage = gpuTrwakeInterpolatedVoltage(vtimeX, time, ip, ib, wake);
    if (wake.probeExponent[0])
      voltage *= gpuIntegerPower(x, wake.probeExponent[0]);
    if (voltage)
      xp += voltage / (1e6 * wake.particleMassMV *
                       wake.particleRelSign) / pz[ip];
  }
  if (wake.hasWake[1] && wake.factor[1]) {
    voltage = gpuTrwakeInterpolatedVoltage(vtimeY, time, ip, ib, wake);
    if (wake.probeExponent[1])
      voltage *= gpuIntegerPower(y, wake.probeExponent[1]);
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long partialIndex = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gpuCombinedWakeConvolutionKernel(
  double *voltage, const double *histogram, const double *table,
  GPU_COMBINED_WAKE_DATA wake, long tableStride) {
  long ib = blockIdx.x * blockDim.x + threadIdx.x;
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
  long frequency = blockIdx.x * blockDim.x + threadIdx.x;
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
  long index = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ib = blockIdx.x * blockDim.x + threadIdx.x;
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
  long frequency = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
    double value = gpuCombinedInterpolatedVoltage(
      voltage, time[ip], ib, &wake);
    double dgamma = value /
      (1e6 * wake.particleMassMV * wake.particleRelSign);
    if (dgamma)
      gpuCombinedAddToParticleEnergy(part, time[ip], &wake, -dgamma);
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
      value /= 1e6 * wake.particleMassMV * wake.particleRelSign *
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
                                       long diSlippage4) {
  long iBin = blockIdx.x * blockDim.x + threadIdx.x;
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
          t1 += 0.3 * denom[1] * denom[1] *
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
          t1 += 0.3 * denom[1] * denom[1] *
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
    t1 *= CSRConstant * dsSlice;
    t2 *= -CSRConstant * dsSlice / slippageLength13;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void gpuCsrHistogramKernel(double *coord,
                                      long nParticles,
                                      int stride,
                                      long coordinateIndex,
                                      double lower,
                                      double binSize,
                                      long bins,
                                      double *hist) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
                                          long startPID, long endPID,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ long count[GPU_REDUCTION_THREADS];
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
         static_cast<long>(part[particleIdColumn]) < startPID ||
         static_cast<long>(part[particleIdColumn]) > endPID))
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
                                       long startPID, long endPID,
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
         static_cast<long>(part[particleIdColumn]) < startPID ||
         static_cast<long>(part[particleIdColumn]) > endPID))
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
                                          long startPID, long endPID,
                                          GPU_BEAM_SUM_DATA *result) {
  __shared__ double bestDelta[GPU_REDUCTION_THREADS];
  __shared__ double bestTime[GPU_REDUCTION_THREADS];
  __shared__ long bestIndex[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  double baselineDelta;

  bestDelta[tid] = -DBL_MAX;
  bestTime[tid] = 0;
  bestIndex[tid] = LONG_MAX;
  if (nParticles <= 0)
    return;

  baselineDelta = coord[5];
  for (ip = tid + 1; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double delta = part[5];

    if ((startPID >= 0 || endPID >= 0) &&
        (particleIdColumn < 0 || particleIdColumn >= stride ||
         static_cast<long>(part[particleIdColumn]) < startPID ||
         static_cast<long>(part[particleIdColumn]) > endPID))
      continue;
    if (delta > baselineDelta &&
        (delta > bestDelta[tid] ||
         (delta == bestDelta[tid] && ip < bestIndex[tid]))) {
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
  __shared__ long count[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  double localSum[7];
  long localCount = 0;
  long tid = threadIdx.x;
  long ip;
  int i;

  for (i = 0; i < 7; i++)
    localSum[i] = 0;
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  __shared__ long count[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double pSum[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double gammaSum[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double sum[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double maxabs[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double minValue[7][GPU_BEAM_OUTPUT_THREADS];
  __shared__ double maxValue[7][GPU_BEAM_OUTPUT_THREADS];
  long localCount;
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
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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

__global__ void gpuCenteredBeamSumsPartialKernel(
  double *coord, long nParticles, int stride, double pCentral, double cMks,
  const double *centroid, const double *timeValue, unsigned int productMask,
  GPU_BEAM_SUM_DATA *partial) {
  __shared__ long count[GPU_BEAM_OUTPUT_THREADS];
  __shared__ double product[28][GPU_BEAM_OUTPUT_THREADS];
  long localCount;
  double localProduct[28];
  long tid = threadIdx.x;
  long ip;
  int i, j;

  localCount = 0;
  for (i = 0; i < 28; i++)
    localProduct[i] = 0;
  for (ip = blockIdx.x * blockDim.x + tid; ip < nParticles;
       ip += gridDim.x * blockDim.x) {
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  __shared__ double minTime[GPU_REDUCTION_THREADS];
  __shared__ double maxTime[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  xSum[tid] = 0;
  ySum[tid] = 0;
  minTime[tid] = DBL_MAX;
  maxTime[tid] = -DBL_MAX;
  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    double time = gpuParticleTime(part, pCentral, cMks);

    xSum[tid] += part[0];
    ySum[tid] += part[2];
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

__global__ void gpuLongMinMaxKernel(double *coord, long nParticles, int stride,
                                    int coordinateIndex,
                                    GPU_LONG_MIN_MAX_DATA *result) {
  __shared__ long count[GPU_REDUCTION_THREADS];
  __shared__ long minValue[GPU_REDUCTION_THREADS];
  __shared__ long maxValue[GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;

  count[tid] = 0;
  minValue[tid] = LONG_MAX;
  maxValue[tid] = LONG_MIN;

  for (ip = tid; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    long value = static_cast<long>(part[coordinateIndex]);
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

__global__ void gpuSortedBunchValidateKernel(
  const double *coord, long nParticles, int stride, int coordinateIndex,
  int *sorted) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  if (ip + 1 >= nParticles)
    return;
  if (static_cast<long>(coord[ip * stride + coordinateIndex]) >
      static_cast<long>(coord[(ip + 1) * stride + coordinateIndex]))
    atomicExch(sorted, 0);
}

__global__ void gpuSortedBunchRangesKernel(
  const double *coord, long nParticles, int stride, int coordinateIndex,
  long minBunch, long nBuckets, long *start, long *count) {
  long bucket = blockIdx.x * blockDim.x + threadIdx.x;
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
  __shared__ long count[GPU_REDUCTION_THREADS];
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
  return static_cast<int>(cudaFree(ptr));
}

extern "C" int gpuCudaCopyHostToDevice(void *dst, const void *src, unsigned long count, float *milliseconds) {
  return timeCopy(dst, src, count * sizeof(double), cudaMemcpyHostToDevice, milliseconds);
}

extern "C" int gpuCudaCopyDeviceToHost(void *dst, const void *src, unsigned long count, float *milliseconds) {
  return timeCopy(dst, src, count * sizeof(double), cudaMemcpyDeviceToHost, milliseconds);
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

extern "C" int gpuCudaTrackParticles(void *coord, long nParticles, int stride,
                                     const GPU_MATRIX_DATA *matrix, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  cudaError_t status;
  int threads = 256;
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
  int threads = 256;
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  coordCount = static_cast<unsigned long long>(nParticles) *
               static_cast<unsigned long long>(stride);
  cudaStatus = cudaMalloc(&backup, coordCount * sizeof(*backup));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(backup, coord, coordCount * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  gpuKickMapTrackCheckedKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, *data,
    xpFactor, ypFactor, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount,
                          sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backup, coordCount * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  cudaFree(backup);
  cudaFree(deviceLostCount);
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
  if (!coord || !scratchCoord || !prefix || !data || !xpFactor || !ypFactor ||
      stride < 6 || nParticles < 0 || points <= 0 ||
      data->nKicks < 1 || data->nx <= 1 || data->ny <= 1 ||
      data->dxg == 0 || data->dyg == 0)
    return static_cast<int>(cudaErrorInvalidValue);

  gridSize = static_cast<int>((nParticles + blockSize - 1) / blockSize);
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuKickMapSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, *data,
    xpFactor, ypFactor, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuKickMapStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, survivors, *data, xpFactor, ypFactor,
      zStart, pRef);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaMultipoleTrack(void *coord, long nParticles, int stride,
                                     const GPU_MULTIPOLE_DATA *multipole,
                                     int writeOutput, long *lostCount,
                                     float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  long *deviceLostCount = NULL;
  int threads = writeOutput ? 256 : GPU_REDUCTION_THREADS;
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
    cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
    if (cudaStatus != cudaSuccess) {
      cudaFree(deviceLostCount);
      return static_cast<int>(cudaStatus);
    }
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (deviceLostCount)
      cudaFree(deviceLostCount);
    return status;
  }
  if (writeOutput) {
    if (multipole->radCoef)
      gpuMultipoleTrackKernel<true><<<blocks, threads>>>(
        static_cast<double *>(coord), nParticles, stride);
    else
      gpuMultipoleTrackKernel<false><<<blocks, threads>>>(
        static_cast<double *>(coord), nParticles, stride);
  } else {
    if (multipole->radCoef)
      gpuMultipolePredicateKernel<true><<<blocks, threads>>>(
        static_cast<double *>(coord), nParticles, stride, deviceLostCount);
    else
      gpuMultipolePredicateKernel<false><<<blocks, threads>>>(
        static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  }
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (deviceLostCount)
      cudaFree(deviceLostCount);
    return status;
  }
  if (!writeOutput) {
    cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                            cudaMemcpyDeviceToHost);
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaMultipoleTrackChecked(void *coord, long nParticles,
                                            int stride,
                                            const GPU_MULTIPOLE_DATA *multipole,
                                            long *lostCount,
                                            float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = 256;
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
  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = cudaMalloc(&backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  if (multipole->radCoef)
    gpuMultipoleTrackCheckedKernel<true>
      <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
        static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  else
    gpuMultipoleTrackCheckedKernel<false>
      <<<blocks, threads, threads * sizeof(unsigned long long)>>>(
        static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  cudaFree(backup);
  cudaFree(deviceLostCount);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaCcbendTrackChecked(void *coord, long nParticles,
                                          int stride,
                                          const GPU_CCBEND_DATA *ccbend,
                                          long *lostCount,
                                          float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  if (!coord || !ccbend || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);
  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = cudaMalloc(&backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  gpuCcbendTrackCheckedKernel<<<blocks, threads,
                                threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, *ccbend,
    deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount,
                          sizeof(hostLostCount), cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount)
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  cudaFree(backup);
  cudaFree(deviceLostCount);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  *lostCount = static_cast<long>(hostLostCount);
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaMultipoleTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_MULTIPOLE_DATA *multipole, long *remaining, float *milliseconds) {
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

  if (multipole->radCoef)
    gpuMultipoleSurvivorFlagKernel<true><<<gridSize, blockSize>>>(
      static_cast<double *>(coord), nParticles, stride, devicePrefix);
  else
    gpuMultipoleSurvivorFlagKernel<false><<<gridSize, blockSize>>>(
      static_cast<double *>(coord), nParticles, stride, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L,
                               thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    if (multipole->radCoef)
      gpuMultipoleStableTrackScatterKernel<true><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), static_cast<double *>(scratchCoord),
        devicePrefix, nParticles, stride, survivors);
    else
      gpuMultipoleStableTrackScatterKernel<false><<<gridSize, blockSize>>>(
        static_cast<double *>(coord), static_cast<double *>(scratchCoord),
        devicePrefix, nParticles, stride, survivors);
    cudaStatus = cudaGetLastError();
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
                                                double x, double y) {
  const GPU_CSBEND_DATA *data = &gpuCsbendData;
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

__device__ __forceinline__ int gpuCsbendApplyEdge(double *xp, double *yp,
                                                  double x, double y,
                                                  double dp, double e,
                                                  double psi,
                                                  double kickLimit) {
  const GPU_CSBEND_DATA *data = &gpuCsbendData;
  double onePlusDp = 1 + dp;
  double rho;
  double deltaXp;

  if (onePlusDp == 0)
    return 0;
  rho = onePlusDp * data->rhoActual;
  if (rho == 0)
    return 0;
  deltaXp = tan(e) / rho * x;
  if (kickLimit > 0 && fabs(deltaXp) > kickLimit)
    deltaXp = copysign(kickLimit, deltaXp);
  *xp += deltaXp;
  *yp -= tan(e - psi / onePlusDp) / rho * y;
  return isfinite(*xp) && isfinite(*yp);
}

__device__ __forceinline__ int gpuCsbendApplyHigherOrderEdge(double *x,
                                                             double *xp,
                                                             double *y,
                                                             double *yp,
                                                             double dp,
                                                             double beta,
                                                             double he,
                                                             double psi,
                                                             int whichEdge) {
  const GPU_CSBEND_DATA *data = &gpuCsbendData;
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
  cosBeta = cos(beta);
  if (cosBeta == 0)
    return 0;

  h = 1 / rho;
  tanBeta = tan(beta);
  R21 = h * tanBeta;
  R43 = -h * tan(beta - psi * onePlusDp);

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

  *x = x0 + T111 * x0 * x0 + T133 * y0 * y0;
  *xp = xp0 + R21 * x0 + T211 * x0 * x0 + T221 * x0 * xp0 +
        T233 * y0 * y0 + T243 * y0 * yp0;
  *y = y0 + T331 * x0 * y0;
  *yp = yp0 + R43 * y0 + T441 * yp0 * x0 + T431 * x0 * y0 +
        T432 * xp0 * y0;
  return isfinite(*x) && isfinite(*xp) && isfinite(*y) && isfinite(*yp);
}

__device__ __forceinline__ int gpuCsbendApplyConfiguredEdge(double *x,
                                                            double *xp,
                                                            double *y,
                                                            double *yp,
                                                            double dp,
                                                            double e,
                                                            double he,
                                                            double psi,
                                                            double kickLimit,
                                                            int whichEdge) {
  if (gpuCsbendData.edgeOrder > 1)
    return gpuCsbendApplyHigherOrderEdge(x, xp, y, yp, dp, e, he, psi,
                                         whichEdge);
  return gpuCsbendApplyEdge(xp, yp, *x, *y, dp, e, psi, kickLimit);
}

__device__ int gpuCsbendTrackParticle(double *part, int stride, int writeOutput) {
  const GPU_CSBEND_DATA *data = &gpuCsbendData;
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
  double onePlusDp = 1 + dp;
  double dsSlice;
  int nSubsteps = 0;

  (void)stride;
  if (!isfinite(x0) || !isfinite(xp0) || !isfinite(y0) ||
      !isfinite(yp0) || !isfinite(dp))
    return 0;
  if (onePlusDp == 0)
    return 0;
  if (data->hasMisalignment) {
    s0 += data->dzi * sqrt(1 + xp0 * xp0 + yp0 * yp0);
    x0 = x0 + data->dxi + data->dzi * xp0;
    y0 = y0 + data->dyi + data->dzi * yp0;
  }
  if (fabs(x0) > data->coordLimit || fabs(y0) > data->coordLimit ||
      fabs(xp0) > data->slopeLimit || fabs(yp0) > data->slopeLimit)
    return 0;
  if (!gpuLoadSymplecticFractions(data->integrationOrder, driftFrac,
                                  kickFrac, &nSubsteps))
    return 0;

  x = x0 * data->cosTilt + y0 * data->sinTilt;
  y = -x0 * data->sinTilt + y0 * data->cosTilt;
  xp = xp0 * data->cosTilt + yp0 * data->sinTilt;
  yp = -xp0 * data->sinTilt + yp0 * data->cosTilt;

  if (data->edge1 &&
      !gpuCsbendApplyConfiguredEdge(&x, &xp, &y, &yp, dp, data->e1,
                                    data->he1, data->psi1,
                                    data->edgeKickLimit1, -1))
    return 0;

  if (data->radCoef && data->edge1 && data->e1 != 0) {
    double Fx, Fy, dpPrime;
    double onePlusXh = 1 + x / data->rho0;

    gpuCsbendFields(&Fx, &Fy, x, y);
    dpPrime =
      -data->radCoef * (Fx * Fx + Fy * Fy) * (1 + dp) * (1 + dp) *
      sqrt(onePlusXh * onePlusXh + xp * xp + yp * yp);
    dp -= dpPrime * x * tan(data->e1);
    onePlusDp = 1 + dp;
    if (onePlusDp == 0)
      return 0;
  }

  if (!gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return 0;

  dist = 0;
  dsSlice = data->length / data->nSlices;
  for (long i = 0; i < data->nSlices; i++) {
    for (int j = 0; j < nSubsteps; j++) {
      double dsh = dsSlice * driftFrac[j];

      if (data->expandHamiltonian) {
        qx += dsh * onePlusDp / (2 * data->rho0);
        dist += dsh * (1 + (qx * qx + qy * qy) / 2);
        x += qx * dsh / onePlusDp;
        y += qy * dsh / onePlusDp;
        qx += dsh * onePlusDp / (2 * data->rho0);
      } else {
        double f = (1 + dp) * (1 + dp) - qy * qy;
        double sinPhi, phi, sine, cosi, tang, cosPhi, factor;

        if (f <= 0)
          return 0;
        f = sqrt(f);
        if (fabs(qx / f) > 1)
          return 0;
        sinPhi = qx / f;
        phi = asin(sinPhi);
        sine = sin(dsh / data->rho0 + phi);
        cosi = cos(dsh / data->rho0 + phi);
        if (cosi == 0)
          return 0;
        tang = sine / cosi;
        cosPhi = cos(phi);
        qx = f * sine;
        factor = (data->rho0 + x) * cosPhi / f *
                 (tang - sinPhi / cosPhi);
        y += qy * factor;
        dist += factor * (1 + dp);
        f = cosPhi / cosi;
        x = data->rho0 * (f - 1) + f * x;
      }
      if (!isfinite(x) || !isfinite(y) || !isfinite(qx) ||
          !isfinite(qy) || !isfinite(dist))
        return 0;

      if (kickFrac[j] == 0)
        break;
      double ds = dsSlice * kickFrac[j];
      double Fx, Fy;
      gpuCsbendFields(&Fx, &Fy, x, y);
      qx += -ds * (1 + x / data->rho0) * Fy / data->rhoActual;
      qy += ds * (1 + x / data->rho0) * Fx / data->rhoActual;
      if (data->radCoef) {
        double momentum2 = (1 + dp) * (1 + dp) - qx * qx - qy * qy;
        double factor, radXp, radYp, dsFactor, F2, deltaFactor;

        if (momentum2 <= 0)
          return 0;
        factor = (1 + x / data->rho0) / sqrt(momentum2);
        radXp = qx * factor;
        radYp = qy * factor;
        dsFactor =
          sqrt((1 + x / data->rho0) * (1 + x / data->rho0) +
               radXp * radXp + radYp * radYp);
        F2 = Fx * Fx + Fy * Fy;
        deltaFactor = (1 + dp) * (1 + dp);
        qx /= (1 + dp);
        qy /= (1 + dp);
        dp -= data->radCoef * deltaFactor * F2 * ds * dsFactor;
        onePlusDp = 1 + dp;
        if (onePlusDp == 0)
          return 0;
        qx *= onePlusDp;
        qy *= onePlusDp;
      }
    }
  }

  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;

  if (data->radCoef && data->edge2 && data->e2 != 0) {
    double Fx, Fy, dpPrime;
    double onePlusXh = 1 + x / data->rho0;

    gpuCsbendFields(&Fx, &Fy, x, y);
    dpPrime =
      -data->radCoef * (Fx * Fx + Fy * Fy) * (1 + dp) * (1 + dp) *
      sqrt(onePlusXh * onePlusXh + xp * xp + yp * yp);
    dp -= dpPrime * x * tan(data->e2);
  }

  if (data->radCoef) {
    double p0 = data->Po * (1 + dp0);
    double p1 = data->Po * (1 + dp);
    double beta0 = p0 / sqrt(p0 * p0 + 1);
    double beta1 = p1 / sqrt(p1 * p1 + 1);

    trackedS = beta1 * s0 / beta0 + dist;
  } else {
    trackedS = s0 + dist;
  }

  if (data->edge2 &&
      !gpuCsbendApplyConfiguredEdge(&x, &xp, &y, &yp, dp, data->e2,
                                    data->he2, data->psi2,
                                    data->edgeKickLimit2, 1))
    return 0;

  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
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
    }
    part[0] = xOut;
    part[2] = yOut;
    part[1] = xpOut;
    part[3] = ypOut;
    part[4] = sOut;
    part[5] = dp;
  }
  return 1;
}

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
  double dsSlice;
  int nSubsteps = 0;

  (void)stride;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) ||
      !isfinite(yp) || !isfinite(dp) || !isfinite(beta0) || beta0 == 0)
    return 0;
  if (onePlusDp == 0)
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
    return 0;
  if (!gpuLoadSymplecticFractions(data->integrationOrder, driftFrac,
                                  kickFrac, &nSubsteps))
    return 0;
  if (!gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp,
                                          data->expandHamiltonian))
    return 0;

  dist = 0;
  dsSlice = data->length / data->nSlices;
  for (long i = 0; i < data->nSlices; i++) {
    for (int j = 0; j < nSubsteps; j++) {
      double dsh = dsSlice * driftFrac[j];

      if (data->expandHamiltonian) {
        qx += dsh * onePlusDp / (2 * data->rho0);
        dist += dsh * (1 + (qx * qx + qy * qy) / 2);
        x += qx * dsh / onePlusDp;
        y += qy * dsh / onePlusDp;
        qx += dsh * onePlusDp / (2 * data->rho0);
      } else {
        double f = onePlusDp * onePlusDp - qy * qy;
        double sinPhi, phi, sine, cosi, tang, cosPhi, factor;

        if (f <= 0)
          return 0;
        f = sqrt(f);
        if (fabs(qx / f) > 1)
          return 0;
        sinPhi = qx / f;
        phi = asin(sinPhi);
        sine = sin(dsh / data->rho0 + phi);
        cosi = cos(dsh / data->rho0 + phi);
        if (cosi == 0)
          return 0;
        tang = sine / cosi;
        cosPhi = cos(phi);
        qx = f * sine;
        factor = (data->rho0 + x) * cosPhi / f *
                 (tang - sinPhi / cosPhi);
        y += qy * factor;
        dist += factor * onePlusDp;
        f = cosPhi / cosi;
        x = data->rho0 * (f - 1) + f * x;
      }
      if (!isfinite(x) || !isfinite(y) || !isfinite(qx) ||
          !isfinite(qy) || !isfinite(dist))
        return 0;

      if (kickFrac[j] == 0)
        break;
      double ds = dsSlice * kickFrac[j];
      double Fx, Fy;
      gpuCsbendFields(&Fx, &Fy, x, y);
      qx += -ds * (1 + x / data->rho0) * Fy / data->rhoActual;
      qy += ds * (1 + x / data->rho0) * Fx / data->rhoActual;
    }
  }

  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;
  if (!isfinite(x) || !isfinite(xp) || !isfinite(y) || !isfinite(yp))
    return 0;
  if (fabs(x) > data->coordLimit || fabs(y) > data->coordLimit ||
      fabs(xp) > data->slopeLimit || fabs(yp) > data->slopeLimit)
    return 0;

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
                                            unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuCsbendTrackParticle(coord + ip * stride, stride, 1))
      localCount = 1;
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
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuCsbendTrackParticle(coord + ip * stride, stride, 0) ? 1 : 0;
}

__global__ void gpuCsbendStableTrackScatterKernel(
  double *coord, double *scratch, const long *survivorPrefix,
  long nParticles, int stride, long survivors) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, *target;
  long prefix, destination;
  int survives;

  if (ip >= nParticles)
    return;
  prefix = survivorPrefix[ip];
  survives = (ip + 1 < nParticles) ? survivorPrefix[ip + 1] > prefix :
                                     survivors > prefix;
  destination = survives ? prefix : survivors + (ip - prefix);
  part = coord + ip * stride;
  target = scratch + destination * stride;
  for (int ic = 0; ic < stride; ic++)
    target[ic] = part[ic];
  if (survives)
    gpuCsbendTrackParticle(target, stride, 1);
}

__global__ void gpuCsrCsbendBodySliceCheckedKernel(double *coord,
                                                   long nParticles,
                                                   int stride,
                                                   const double *beta0,
                                                   unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuCsrCsbendTrackBodySliceParticle(coord + ip * stride, stride,
                                            beta0[ip]))
      localCount = 1;
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

__global__ void gpuCsrCsbendEnterSimpleCheckedKernel(
  double *coord, long nParticles, int stride, double pCentral,
  double coordinateSign, int edge1Effect, double e1, double psi1,
  double rhoActual, double *beta0, unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
            double deltaXp = tan(e1) / rho * part[0];
            part[1] += deltaXp;
            part[3] -= tan(e1 - psi1 / (1 + part[5])) / rho * part[2];
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
  double coordinateSign, int edge2Effect, double e2, double psi2,
  double rhoActual, unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
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
          double deltaXp = tan(e2) / rho * part[0];
          part[1] += deltaXp;
          part[3] -= tan(e2 - psi2 / (1 + part[5])) / rho * part[2];
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
                                         long *lostCount,
                                         float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  unsigned long long *deviceLostCount = NULL;
  unsigned long long hostLostCount = 0;
  double *backup = NULL;
  unsigned long long count;
  int threads = 256;
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
  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  count = static_cast<unsigned long long>(nParticles) *
          static_cast<unsigned long long>(stride);
  cudaStatus = cudaMalloc(&backup, count * sizeof(*backup));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(backup, coord, count * sizeof(*backup),
                          cudaMemcpyDeviceToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }
  gpuCsbendTrackCheckedKernel<<<blocks, threads,
                                threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(backup);
    cudaFree(deviceLostCount);
    return status;
  }

  cudaStatus = cudaMemcpy(&hostLostCount, deviceLostCount, sizeof(hostLostCount),
                          cudaMemcpyDeviceToHost);
  if (cudaStatus == cudaSuccess && hostLostCount) {
    cudaStatus = cudaMemcpy(coord, backup, count * sizeof(*backup),
                            cudaMemcpyDeviceToDevice);
  }
  cudaFree(backup);
  cudaFree(deviceLostCount);
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
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuCsbendStableTrackScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, survivors);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaCsrCsbendBodySliceChecked(void *coord,
                                                long nParticles,
                                                int stride,
                                                const GPU_CSBEND_DATA *csbend,
                                                const double *beta0,
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

  if (!coord || !csbend || !beta0 || !backupCoord || !lostCountDevice || !lostCount)
    return static_cast<int>(cudaErrorInvalidValue);
  *lostCount = 0;
  if (milliseconds)
    *milliseconds = 0;
  if (nParticles <= 0)
    return static_cast<int>(cudaSuccess);

  status = uploadCsbendDataIfNeeded(csbend);
  if (status != static_cast<int>(cudaSuccess))
    return status;
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
  gpuCsrCsbendBodySliceCheckedKernel<<<blocks, threads,
                                       threads * sizeof(unsigned long long)>>>(
    static_cast<double *>(coord), nParticles, stride, beta0,
    lostCountDevice);
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
    coordinateSign, edge1Effect, e1, psi1, rhoActual, beta0, lostCountDevice);
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
    coordinateSign, edge2Effect, e2, psi2, rhoActual, lostCountDevice);
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

extern "C" int gpuCudaCombinedWakeTrack(
  void *coord, long nParticles, int stride,
  const GPU_COMBINED_WAKE_DATA *wake,
  const double *const *tables, long *binnedCount,
  double *histogramReturn, double *voltageReturn,
  float *milliseconds) {
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

  planBins = wake->bins;
  useTimeFft = wake->mode == GPU_COMBINED_WAKE_MODE_TIME &&
               wake->allowTimeFft &&
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
  if (cudaStatus == cudaSuccess)
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
  int blocks;

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
  int threads = 256, blocks, status;

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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtime, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceWake, wake->wakePoints * sizeof(*deviceWake));
  if (cudaStatus != cudaSuccess)
    goto cleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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

  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceVtime);
  cudaFree(deviceWake);
  cudaFree(deviceBinnedCount);
  return status;

cleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceVtime);
  cudaFree(deviceWake);
  cudaFree(deviceBinnedCount);
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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto wakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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

  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceBinnedCount);
  return status;

wakeHistogramCleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceBinnedCount);
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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceItime, wake->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtime, (wake->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceWake, wake->wakePoints * sizeof(*deviceWake));
  if (cudaStatus != cudaSuccess)
    goto wakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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

  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceVtime);
  cudaFree(deviceWake);
  cudaFree(deviceBinnedCount);
  return status;

wakeTrackHistogramCleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePbin);
  cudaFree(deviceItime);
  cudaFree(deviceVtime);
  cudaFree(deviceWake);
  cudaFree(deviceBinnedCount);
  return static_cast<int>(cudaStatus);
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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtimeX, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtimeY, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeCleanupWithoutEvents;
  if (wake->hasWake[0]) {
    cudaStatus = cudaMalloc(&deviceWakeX, wake->wakePoints * sizeof(*deviceWakeX));
    if (cudaStatus != cudaSuccess)
      goto trwakeCleanupWithoutEvents;
  }
  if (wake->hasWake[1]) {
    cudaStatus = cudaMalloc(&deviceWakeY, wake->wakePoints * sizeof(*deviceWakeY));
    if (cudaStatus != cudaSuccess)
      goto trwakeCleanupWithoutEvents;
  }
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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
    gpuTrwakeBinSumsKernel<<<static_cast<int>(wake->bins), threads,
                             2 * threads * sizeof(double)>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      devicePbin, devicePosItimeX, devicePosItimeY);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
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

  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceVtimeX);
  cudaFree(deviceVtimeY);
  cudaFree(deviceWakeX);
  cudaFree(deviceWakeY);
  cudaFree(deviceBinnedCount);
  return status;

trwakeCleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceVtimeX);
  cudaFree(deviceVtimeY);
  cudaFree(deviceWakeX);
  cudaFree(deviceWakeY);
  cudaFree(deviceBinnedCount);
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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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
    gpuTrwakeBinSumsKernel<<<static_cast<int>(wake->bins), threads,
                             2 * threads * sizeof(double)>>>(
      static_cast<double *>(coord), nParticles, stride, *wake,
      devicePbin, devicePosItimeX, devicePosItimeY);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
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

  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceBinnedCount);
  return status;

trwakeHistogramCleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceBinnedCount);
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

  cudaStatus = cudaMalloc(&deviceTime, nParticles * sizeof(*deviceTime));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePz, nParticles * sizeof(*devicePz));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePbin, nParticles * sizeof(*devicePbin));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeX, wake->bins * sizeof(*devicePosItimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&devicePosItimeY, wake->bins * sizeof(*devicePosItimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtimeX, (wake->bins + 1) * sizeof(*deviceVtimeX));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceVtimeY, (wake->bins + 1) * sizeof(*deviceVtimeY));
  if (cudaStatus != cudaSuccess)
    goto trwakeTrackHistogramCleanupWithoutEvents;
  if (wake->hasWake[0]) {
    cudaStatus = cudaMalloc(&deviceWakeX, wake->wakePoints * sizeof(*deviceWakeX));
    if (cudaStatus != cudaSuccess)
      goto trwakeTrackHistogramCleanupWithoutEvents;
  }
  if (wake->hasWake[1]) {
    cudaStatus = cudaMalloc(&deviceWakeY, wake->wakePoints * sizeof(*deviceWakeY));
    if (cudaStatus != cudaSuccess)
      goto trwakeTrackHistogramCleanupWithoutEvents;
  }
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
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

  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceVtimeX);
  cudaFree(deviceVtimeY);
  cudaFree(deviceWakeX);
  cudaFree(deviceWakeY);
  cudaFree(deviceBinnedCount);
  return status;

trwakeTrackHistogramCleanupWithoutEvents:
  cudaFree(deviceTime);
  cudaFree(devicePz);
  cudaFree(devicePbin);
  cudaFree(devicePosItimeX);
  cudaFree(devicePosItimeY);
  cudaFree(deviceVtimeX);
  cudaFree(deviceVtimeY);
  cudaFree(deviceWakeX);
  cudaFree(deviceWakeY);
  cudaFree(deviceBinnedCount);
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
  int freeDeviceItime = 0;
  int freeDeviceBinnedCount = 0;
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
    cudaStatus = cudaMalloc(&deviceItime, lsc->bins * sizeof(*deviceItime));
    if (cudaStatus != cudaSuccess)
      goto lscBinCleanupWithoutEvents;
    freeDeviceItime = 1;
  }
  if (!deviceBinnedCount) {
    cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
    if (cudaStatus != cudaSuccess)
      goto lscBinCleanupWithoutEvents;
    freeDeviceBinnedCount = 1;
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

  if (freeDeviceItime)
    cudaFree(deviceItime);
  if (freeDeviceBinnedCount)
    cudaFree(deviceBinnedCount);
  return status;

lscBinCleanupWithoutEvents:
  if (freeDeviceItime)
    cudaFree(deviceItime);
  if (freeDeviceBinnedCount)
    cudaFree(deviceBinnedCount);
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
  int freeDeviceVtime = 0;
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
    cudaStatus = cudaMalloc(&deviceVtime, (lsc->bins + 1) * sizeof(*deviceVtime));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    freeDeviceVtime = 1;
  }

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceVtime)
      cudaFree(deviceVtime);
    return status;
  }

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

  if (freeDeviceVtime)
    cudaFree(deviceVtime);
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
    diSlippage, diSlippage4);
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
                                 double pCentral, long startPID, long endPID,
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

extern "C" int gpuCudaClearBatchedSearchHistory(
  void *history, unsigned long historyCount, void *turnCount,
  unsigned long particles) {
  cudaError_t status;
  status = cudaMemset(history, 0, historyCount * sizeof(double));
  if (status == cudaSuccess)
    status = cudaMemset(turnCount, 0, particles * sizeof(double));
  return static_cast<int>(status);
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

extern "C" unsigned long gpuCudaMatchEnergyScratchBytes(void) {
  return GPU_MATCH_ENERGY_BLOCKS * sizeof(GPU_MATCH_ENERGY_PARTIAL) +
         sizeof(GPU_BEAM_SUM_DATA) + sizeof(double);
}

extern "C" int gpuCudaRfcaThinKick(void *coord, long nParticles, int stride,
                                   double pCentral, double volt, double omega,
                                   double phase, double cMks,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  int status;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRfcaThinKickKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles,
                                             stride, pCentral, volt, omega,
                                             phase, cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
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

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuBggexpKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                      nParticles, stride, deviceData);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaCwigglerTrack(void *coord, long nParticles, int stride,
                                    const GPU_CWIGGLER_DATA *data,
                                    float *milliseconds) {
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
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCwigglerKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                        nParticles, stride, *data);
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

extern "C" int gpuCudaBmxyzTrack(void *coord, long nParticles, int stride,
                                  const GPU_BMXYZ_DATA *data,
                                  long *failedCount, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_BMXYZ_DATA deviceData;
  GPU_FTABLE_DATA uploadData;
  cudaError_t cudaStatus;
  int status;
  int threads = 128;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);
  long coordinateValues;
  unsigned long long hostFailedCount = 0;

  if (!coord || !data || !failedCount || nParticles <= 0 || stride < 7 ||
      nParticles > LONG_MAX / stride)
    return static_cast<int>(cudaErrorInvalidValue);
  *failedCount = 0;
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
                                      gpuBmxyzScratch.failedCount);
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
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaRfcwRfOnlyMatrix(void *coord, long nParticles, int stride,
                                       double pCentral, double length,
                                       double volt, double omega,
                                       double phase, int end1Focus,
                                       int end2Focus, double dx, double dy,
                                       double cMks,
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
                                                 cMks, NULL);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
}

extern "C" int gpuCudaRfcwRfOnlyMatrixChecked(
  void *coord, long nParticles, int stride, double pCentral, double length,
  double volt, double omega, double phase, int end1Focus, int end2Focus,
  double dx, double dy, double cMks, void *deviceLostCount,
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
    static_cast<long *>(deviceLostCount));
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
                                      int end1Focus, double cMks,
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
    nParticles, stride, pCentral, length, volt, omega, phase,
    end1Focus, cMks);
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
                                              double cMks,
                                              GPU_BEAM_SUM_DATA *result,
                                              float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuRfcwDgammaOverGammaSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral,
    length, volt, omega, phase, cMks, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
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
  long startPID, long endPID, unsigned int coordinateMask,
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
  long startPID, long endPID, long bins, unsigned int coordinateMask,
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

  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuSelectedTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    bunchColumn, selectedBunch, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialTimeSums(void *coord, long nParticles, int stride,
                                       double pCentral, double sOffset,
                                       double cMks, int particleIdColumn,
                                       long startPID, long endPID,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuFiducialTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialPmaximum(void *coord, long nParticles,
                                       int stride, double pCentral,
                                       double sOffset, double cMks,
                                       int particleIdColumn, long startPID,
                                       long endPID,
                                       GPU_BEAM_SUM_DATA *result,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuFiducialPmaximumKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaFiducialFirst(void *coord, long nParticles,
                                    int stride, double pCentral,
                                    double sOffset, double cMks,
                                    int particleIdColumn, long startPID,
                                    long endPID, GPU_BEAM_SUM_DATA *result,
                                    float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuFiducialFirstKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, sOffset,
    cMks, particleIdColumn, startPID, endPID, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
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

  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMalloc(&deviceCentroid, sizeof(double) * 7);
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceCentroid);
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemcpy(deviceCentroid, centroid, sizeof(double) * 7,
                          cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceCentroid);
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceCentroid);
    cudaFree(deviceResult);
    return status;
  }
  gpuCenteredBeamSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    deviceCentroid, deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceCentroid);
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  cudaFree(deviceCentroid);
  cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" unsigned long gpuCudaBeamSums2ScratchBytes(long nParticles) {
  return (2 + 2 * GPU_BEAM_SUM_BLOCKS) * sizeof(GPU_BEAM_SUM_DATA) +
         (7 + (nParticles > 0 ? nParticles : 0)) * sizeof(double);
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
  gpuCenteredBeamSumsPartialKernel<<<blocks, GPU_BEAM_OUTPUT_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
    deviceCentroid, deviceTime, productMask, centeredPartial);
  gpuCenteredBeamSumsFinalizeKernel<<<1, GPU_REDUCTION_THREADS>>>(
    centeredPartial, blocks, deviceCenteredResult);

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
  int freeDeviceResult = 0;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  if (!deviceResult) {
    cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    freeDeviceResult = 1;
  }
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuLscStatisticsKernel<<<1, GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, pCentral, cMks,
      deviceResult);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  if (freeDeviceResult)
    cudaFree(deviceResult);
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
  int freeDeviceResult = 0;
  int status;

  if (!coord || !result || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));

  if (!deviceResult) {
    cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    freeDeviceResult = 1;
  }
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus == cudaSuccess)
    gpuLscTransverseSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(
      static_cast<double *>(coord), nParticles, stride, xCentroid,
      yCentroid, deviceResult);
  status = finishTimedKernel(start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result),
                          cudaMemcpyDeviceToHost);
  if (freeDeviceResult)
    cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLongMinMax(void *coord, long nParticles, int stride,
                                 int coordinateIndex,
                                 GPU_LONG_MIN_MAX_DATA *result,
                                 float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_LONG_MIN_MAX_DATA *deviceResult = NULL;
  int status;

  if (!result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  if (!coord || nParticles <= 0 || coordinateIndex < 0 || coordinateIndex >= stride)
    return static_cast<int>(cudaErrorInvalidValue);

  cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceResult, 0, sizeof(*deviceResult));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceResult);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  gpuLongMinMaxKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord),
                                                    nParticles, stride,
                                                    coordinateIndex,
                                                    deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
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
  cudaStatus = cudaMalloc(&deviceStart, nBuckets * sizeof(*deviceStart));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMalloc(&deviceCount, nBuckets * sizeof(*deviceCount));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMalloc(&deviceSorted, sizeof(*deviceSorted));
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
  cudaFree(deviceStart);
  cudaFree(deviceCount);
  cudaFree(deviceSorted);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaDoubleMinMax(void *coord, long nParticles, int stride,
                                   int coordinateIndex,
                                   GPU_DOUBLE_MIN_MAX_DATA *result,
                                   void *deviceResultScratch,
                                   float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_DOUBLE_MIN_MAX_DATA *deviceResult = NULL;
  int freeDeviceResult = 0;
  int status;

  if (!result)
    return static_cast<int>(cudaErrorInvalidValue);
  std::memset(result, 0, sizeof(*result));
  if (!coord || nParticles <= 0 || coordinateIndex < 0 || coordinateIndex >= stride)
    return static_cast<int>(cudaErrorInvalidValue);

  if (deviceResultScratch) {
    deviceResult = static_cast<GPU_DOUBLE_MIN_MAX_DATA *>(deviceResultScratch);
  } else {
    cudaStatus = cudaMalloc(&deviceResult, sizeof(*deviceResult));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    freeDeviceResult = 1;
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  gpuDoubleMinMaxKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord),
                                                      nParticles, stride,
                                                      coordinateIndex,
                                                      deviceResult);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    if (freeDeviceResult)
      cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  if (freeDeviceResult)
    cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                              double xmax, double ymax,
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuLimitAmplitudeLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLimitAmplitudesStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, double z, double pCentral, long extrapolateZ,
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
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuLimitAmplitudesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, xmax, ymax, z, pCentral, extrapolateZ,
      survivors);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}

extern "C" int gpuCudaELimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                               double xmax, double ymax,
                                               long exponent, long yExponent,
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuELimitAmplitudeLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    exponent, yExponent, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaELimitAmplitudesStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  double xmax, double ymax, long exponent, long yExponent, double z,
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
    exponent, yExponent, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuELimitAmplitudesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, xmax, ymax, exponent, yExponent,
      z, pCentral, extrapolateZ, survivors);
    cudaStatus = cudaGetLastError();
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuRemoveInvalidParticlesLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                          cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
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
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuRemoveInvalidParticlesStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuRectangularCollimatorLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, length, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
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
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuRectangularCollimatorStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, xmax, ymax, xCenter, yCenter,
      length, openCode, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuEllipticalCollimatorLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xmax, ymax,
    xCenter, yCenter, exponent, yExponent, length, openCode, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
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
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuEllipticalCollimatorStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, xmax, ymax, xCenter, yCenter,
      exponent, yExponent, length, openCode, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuScraperLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, plane, center, position,
    sideSign, length, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount), cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
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
  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;

  gpuScraperSurvivorFlagKernel<<<gridSize, blockSize>>>(
    static_cast<double *>(coord), nParticles, stride, plane, center,
    position, sideSign, length, devicePrefix);
  cudaStatus = cudaGetLastError();
  if (cudaStatus == cudaSuccess) {
    survivors = thrust::reduce(flags, flags + nParticles, 0L, thrust::plus<long>());
    thrust::exclusive_scan(flags, flags + nParticles, flags);
    cudaStatus = cudaGetLastError();
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuScraperStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, plane, center, position,
      sideSign, length, z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
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

  cudaStatus = cudaMalloc(&deviceLostCount, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  cudaStatus = cudaMemset(deviceLostCount, 0, sizeof(*deviceLostCount));
  if (cudaStatus != cudaSuccess) {
    cudaFree(deviceLostCount);
    return static_cast<int>(cudaStatus);
  }

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  gpuApertureDataLossCountKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, xCenter, yCenter,
    xSize, ySize, deviceLostCount);
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceLostCount);
    return status;
  }
  cudaStatus = cudaMemcpy(lostCount, deviceLostCount, sizeof(*lostCount),
                          cudaMemcpyDeviceToHost);
  cudaFree(deviceLostCount);
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
  } else {
    survivors = nParticles;
  }
  if (cudaStatus == cudaSuccess) {
    gpuApertureDataStableScatterKernel<<<gridSize, blockSize>>>(
      static_cast<double *>(coord), static_cast<double *>(scratchCoord),
      devicePrefix, nParticles, stride, xCenter, yCenter, xSize, ySize,
      z, pCentral, survivors);
    cudaStatus = cudaGetLastError();
  }
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *remaining = survivors;
  return status;
}
