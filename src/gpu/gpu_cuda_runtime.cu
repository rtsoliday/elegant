#include "gpu_base.h"

#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cstring>
#include <float.h>
#include <limits.h>

__constant__ GPU_MATRIX_DATA gpuMatrixData;
__constant__ GPU_MULTIPOLE_DATA gpuMultipoleData;
__constant__ GPU_CSBEND_DATA gpuCsbendData;

#define GPU_REDUCTION_THREADS 64
#define GPU_OPEN_PLUS_X 1
#define GPU_OPEN_PLUS_Y 2
#define GPU_OPEN_MINUS_X 3
#define GPU_OPEN_MINUS_Y 4

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

static int timeKernel(cudaError_t launchStatus, float *milliseconds) {
  cudaError_t status;

  status = launchStatus;
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    return static_cast<int>(status);
  if (!milliseconds)
    return static_cast<int>(cudaSuccess);
  status = cudaGetLastError();
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

__global__ void gpuTrackParticlesKernel(double *coord, long nParticles, int stride) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double ini[6], temp[6];
  double *part;
  int i, j, k, l;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  for (i = 0; i < 6; i++)
    ini[i] = part[i];
  if (gpuMatrixData.useSReference)
    ini[4] -= gpuMatrixData.sReference;

  if (gpuMatrixData.order == 3) {
    for (i = 5; i >= 0; i--) {
      double sum = gpuMatrixData.C[i];
      for (j = 5; j >= 0; j--) {
        double coord_j = ini[j];
        if (coord_j != 0) {
          sum += gpuMatrixData.R[i * 6 + j] * coord_j;
          for (k = j; k >= 0; k--) {
            double coord_k = ini[k];
            if (coord_k != 0) {
              double coord_jk = coord_j * coord_k;
              sum += gpuMatrixData.T[gpuTOffset(i, j, k)] * coord_jk;
              for (l = k; l >= 0; l--)
                sum += gpuMatrixData.Q[gpuQOffset(i, j, k, l)] * coord_jk * ini[l];
            }
          }
        }
      }
      temp[i] = sum;
    }
  } else if (gpuMatrixData.order == 2) {
    for (i = 5; i >= 0; i--) {
      double sum = gpuMatrixData.C[i];
      for (j = 5; j >= 0; j--) {
        double sum1 = gpuMatrixData.R[i * 6 + j];
        for (k = j; k >= 0; k--)
          sum1 += gpuMatrixData.T[gpuTOffset(i, j, k)] * ini[k];
        sum += sum1 * ini[j];
      }
      temp[i] = sum;
    }
  } else {
    for (i = 5; i >= 0; i--) {
      double sum = gpuMatrixData.C[i];
      for (j = 5; j >= 0; j--)
        sum += gpuMatrixData.R[i * 6 + j] * ini[j];
      temp[i] = sum;
    }
  }

  if (gpuMatrixData.useSReference)
    temp[4] += gpuMatrixData.sReference;

  for (i = 5; i >= 0; i--)
    part[i] = temp[i];
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
  double dxp, dyp;

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
      xp = gpuAddRn(xp, gpuMulRn(dxp, scale));
      yp = gpuAddRn(yp, gpuMulRn(dyp, scale));
    } else {
      double scale = gpuDivRn(map->kickScale, gpuAddRn(1.0, dp));
      xp = gpuAddRn(xp, gpuMulRn(dxp, scale));
      yp = gpuAddRn(yp, gpuMulRn(dyp, scale));
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

__device__ int gpuMultipoleTrackParticle(double *part, int stride, int writeOutput) {
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
  double drift, xkick, ykick;
  int nSubsteps = 0;
  int maxOrder = 0;

  (void)stride;
  if (data->dx || data->dy || data->dz) {
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
        qx /= (1 + dp);
        qy /= (1 + dp);
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
    s -= data->dz * sqrt(1 + xp * xp + yp * yp);
    x = x + data->dx - data->dz * xp;
    y = y + data->dy - data->dz * yp;
  }

  if (writeOutput) {
    part[0] = x;
    part[1] = xp;
    part[2] = y;
    part[3] = yp;
    part[4] += s;
    part[5] = dp;
  }
  return 1;
}

__global__ void gpuMultipolePredicateKernel(double *coord, long nParticles,
                                            int stride, long *lostCount) {
  __shared__ long partial[GPU_REDUCTION_THREADS];
  long localCount = 0;
  long thread = threadIdx.x;

  for (long ip = thread; ip < nParticles; ip += blockDim.x) {
    double *part = coord + ip * stride;
    if (!gpuMultipoleTrackParticle(part, stride, 0))
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

__global__ void gpuMultipoleTrackKernel(double *coord, long nParticles,
                                        int stride) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  gpuMultipoleTrackParticle(coord + ip * stride, stride, 1);
}

__global__ void gpuMultipoleTrackCheckedKernel(double *coord, long nParticles,
                                               int stride,
                                               unsigned long long *lostCount) {
  extern __shared__ unsigned long long partial[];
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long localCount = 0;

  if (ip < nParticles) {
    if (!gpuMultipoleTrackParticle(coord + ip * stride, stride, 1))
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

__global__ void gpuMultipoleSurvivorFlagKernel(double *coord, long nParticles,
                                               int stride,
                                               long *survivorPrefix) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  survivorPrefix[ip] =
    gpuMultipoleTrackParticle(coord + ip * stride, stride, 0) ? 1 : 0;
}

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
    gpuMultipoleTrackParticle(target, stride, 1);
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

__global__ void gpuRfcwRfOnlyMatrixKernel(double *coord, long nParticles, int stride,
                                          double pCentral, double length,
                                          double volt, double omega,
                                          double phase, int end1Focus,
                                          int end2Focus, double dx, double dy,
                                          double cMks) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part, p, gamma, beta0, ds1, t, dgamma, gamma1;
  double dP, R12, R22, x, xp, y, yp, inverseF;

  if (ip >= nParticles)
    return;

  part = coord + ip * stride;
  if (part[5] == -1)
    return;

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

__global__ void gpuScmultLinearKickKernel(double *coord, long nParticles,
                                          int stride,
                                          GPU_SCMULT_LINEAR_DATA data) {
  long ip = blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  double k0, dz;

  if (ip >= nParticles)
    return;
  part = coord + ip * stride;
  if (data.uniformDistribution) {
    k0 = data.c1 / data.sigma[2] * data.charge *
         sqrt(3.141592653589793238462643383279502884 / 6.0);
  } else {
    dz = part[4] - data.center[2];
    k0 = data.c1 / data.sigma[2] * data.charge *
         exp(-0.5 * dz * dz / (data.sigma[2] * data.sigma[2]));
  }
  if (data.horizontal)
    part[1] += k0 * data.dmux / data.betax * (part[0] - data.center[0]);
  if (data.vertical)
    part[3] += k0 * data.dmuy / data.betay * (part[2] - data.center[1]);
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

__global__ void gpuBeamSumsKernel(double *coord, long nParticles, int stride,
                                  double pCentral, double cMks,
                                  GPU_BEAM_SUM_DATA *result) {
  __shared__ long count[GPU_REDUCTION_THREADS];
  __shared__ double sum[7][GPU_REDUCTION_THREADS];
  __shared__ double product[28][GPU_REDUCTION_THREADS];
  __shared__ double maxabs[7][GPU_REDUCTION_THREADS];
  __shared__ double minValue[7][GPU_REDUCTION_THREADS];
  __shared__ double maxValue[7][GPU_REDUCTION_THREADS];
  long tid = threadIdx.x;
  long ip;
  int i, j;

  count[tid] = 0;
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
    value[0] = part[0];
    value[1] = part[1];
    value[2] = part[2];
    value[3] = part[3];
    value[4] = part[4];
    value[5] = part[5];
    value[6] = gpuParticleTime(part, pCentral, cMks);
    count[tid]++;
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
  if (status == cudaSuccess)
    status = cudaEventRecord(stop, 0);
  if (status == cudaSuccess)
    status = cudaEventSynchronize(stop);
  if (milliseconds && status == cudaSuccess)
    cudaEventElapsedTime(milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  return timeKernel(cudaSuccess, NULL);
}

static int prepareTimedLaunch(cudaEvent_t *start, cudaEvent_t *stop, float *milliseconds) {
  cudaError_t status;

  if (milliseconds)
    *milliseconds = 0;
  status = cudaEventCreate(start);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaEventCreate(stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(*start);
    return static_cast<int>(status);
  }
  status = cudaEventRecord(*start, 0);
  if (status != cudaSuccess) {
    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaSuccess);
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
  cudaEvent_t start, stop;
  cudaError_t status;
  int threads = 256;
  int blocks = static_cast<int>((nParticles + threads - 1) / threads);

  if (milliseconds)
    *milliseconds = 0;
  status = cudaMemcpyToSymbol(gpuMatrixData, matrix, sizeof(*matrix));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaEventCreate(&start);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(status);
  }
  cudaEventRecord(start, 0);
  gpuTrackParticlesKernel<<<blocks, threads>>>(static_cast<double *>(coord), nParticles, stride);
  cudaEventRecord(stop, 0);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaEventSynchronize(stop);
  if (milliseconds && status == cudaSuccess)
    cudaEventElapsedTime(milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  return timeKernel(cudaSuccess, NULL);
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
  if (writeOutput)
    gpuMultipoleTrackKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                                 nParticles, stride);
  else
    gpuMultipolePredicateKernel<<<blocks, threads>>>(static_cast<double *>(coord),
                                                     nParticles, stride,
                                                     deviceLostCount);
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
  gpuMultipoleTrackCheckedKernel<<<blocks, threads,
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

  gpuMultipoleSurvivorFlagKernel<<<gridSize, blockSize>>>(
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
    gpuMultipoleStableTrackScatterKernel<<<gridSize, blockSize>>>(
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
  double x, xp, y, yp, qx, qy, dist;
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
    }
  }

  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp,
                                          data->expandHamiltonian))
    return 0;

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
    double sOut = s0 + dist + data->dcoordEtilt[4];

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
                             double *itimeReturn, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceItime = NULL;
  unsigned long long *deviceBinnedCount = NULL;
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
  cudaStatus = cudaMalloc(&deviceItime, lsc->bins * sizeof(*deviceItime));
  if (cudaStatus != cudaSuccess)
    goto lscBinCleanupWithoutEvents;
  cudaStatus = cudaMalloc(&deviceBinnedCount, sizeof(*deviceBinnedCount));
  if (cudaStatus != cudaSuccess)
    goto lscBinCleanupWithoutEvents;

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaStatus = static_cast<cudaError_t>(status);
    goto lscBinCleanupWithoutEvents;
  }

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

  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    *binnedCount = static_cast<long>(hostBinnedCount);

  cudaFree(deviceItime);
  cudaFree(deviceBinnedCount);
  return status;

lscBinCleanupWithoutEvents:
  cudaFree(deviceItime);
  cudaFree(deviceBinnedCount);
  return static_cast<int>(cudaStatus);
}

extern "C" int gpuCudaLscApplyKickAndDrift(void *coord, long nParticles,
                                           int stride,
                                           const GPU_LSC_DATA *lsc,
                                           const double *vtime,
                                           float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  double *deviceVtime = NULL;
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
  cudaStatus = cudaMalloc(&deviceVtime, (lsc->bins + 1) * sizeof(*deviceVtime));
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceVtime);
    return status;
  }

  cudaStatus = cudaMemcpy(deviceVtime, vtime,
                          (lsc->bins + 1) * sizeof(*deviceVtime),
                          cudaMemcpyHostToDevice);
  if (cudaStatus == cudaSuccess)
    gpuLscApplyKickAndDriftKernel<<<blocks, threads>>>(
      static_cast<double *>(coord), nParticles, stride, *lsc, deviceVtime);
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaGetLastError();
  status = launchTimedKernel(cudaStatus, start, stop, milliseconds);

  cudaFree(deviceVtime);
  return status;
}

extern "C" int gpuCudaScmultLinearKick(void *coord, long nParticles, int stride,
                                       const GPU_SCMULT_LINEAR_DATA *data,
                                       float *milliseconds) {
  cudaEvent_t start, stop;
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

  status = prepareTimedLaunch(&start, &stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  blocks = static_cast<int>((nParticles + threads - 1) / threads);
  gpuScmultLinearKickKernel<<<blocks, threads>>>(
    static_cast<double *>(coord), nParticles, stride, *data);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
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
  gpuMatchEnergyAndAverageKernel<<<1, GPU_REDUCTION_THREADS>>>(
    static_cast<double *>(coord), nParticles, stride, oldP, changeBeam,
    deviceResult);
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
                                                 cMks);
  return launchTimedKernel(cudaSuccess, start, stop, milliseconds);
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

static int gpuCudaRunReduction(void *coord, long nParticles, int stride,
                               double pCentral, double cMks, int beamSums,
                               GPU_BEAM_SUM_DATA *result, float *milliseconds) {
  cudaEvent_t start, stop;
  cudaError_t cudaStatus;
  GPU_BEAM_SUM_DATA *deviceResult = NULL;
  int status;

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
  if (beamSums > 0) {
    gpuBeamSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord), nParticles,
                                                    stride, pCentral, cMks, deviceResult);
  } else if (beamSums < -1) {
    gpuCentroidTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord), nParticles,
                                                            stride, pCentral, cMks, deviceResult);
  } else if (beamSums < 0) {
    gpuTimeSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord), nParticles,
                                                    stride, pCentral, cMks, deviceResult);
  } else {
    gpuCentroidSumsKernel<<<1, GPU_REDUCTION_THREADS>>>(static_cast<double *>(coord), nParticles,
                                                        stride, deviceResult);
  }
  status = launchTimedKernel(cudaSuccess, start, stop, milliseconds);
  if (status != static_cast<int>(cudaSuccess)) {
    cudaFree(deviceResult);
    return status;
  }
  cudaStatus = cudaMemcpy(result, deviceResult, sizeof(*result), cudaMemcpyDeviceToHost);
  cudaFree(deviceResult);
  return static_cast<int>(cudaStatus);
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
