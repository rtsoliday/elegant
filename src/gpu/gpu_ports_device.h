/* Device code for the element types moved on November 15, 2026 (gpu_ports.h).  Included at the end of
 * gpu_cuda_runtime.cu, whose helpers it uses.  Every kernel follows its CPU routine's statements in order
 * (with --fmad=false, IEEE division and square root), so the particles come out as the CPU's. */

#include <cfloat>

/* Buffers shared by these kernels: a flag a particle (lost, or needing attention), counters, the rows
 * kept while the CPU's loss order is gathered, and the loss order itself. */
static unsigned char *gpuPortFlags = NULL;
static long gpuPortFlagsCapacity = 0;
static unsigned long long *gpuPortCounters = NULL;
static double *gpuPortRows = NULL;
static long gpuPortRowsCapacity = 0;
static long long *gpuPortPerm = NULL;
static long gpuPortPermCapacity = 0;
static double *gpuPortValues = NULL;
static long gpuPortValuesCapacity = 0;

#define GPU_PORT_COUNTERS 4
/* counters[0]: particles flagged lost; [1]: warnings; [2]: smallest index of a particle in error
 * (ULLONG_MAX when none); [3]: spare */

static int gpuPortPrepare(long nParticles, int needFlags) {
  cudaError_t status;
  unsigned long long initial[GPU_PORT_COUNTERS] = {0, 0, ~0ULL, 0};

  if (!gpuPortCounters) {
    status = cudaMalloc(&gpuPortCounters, GPU_PORT_COUNTERS * sizeof(*gpuPortCounters));
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  if (needFlags && gpuPortFlagsCapacity < nParticles) {
    gpuFreeDevice(gpuPortFlags);
    gpuPortFlags = NULL;
    gpuPortFlagsCapacity = 0;
    status = cudaMalloc(&gpuPortFlags, nParticles * sizeof(*gpuPortFlags));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuPortFlagsCapacity = nParticles;
  }
  status = cudaMemcpy(gpuPortCounters, initial, sizeof(initial), cudaMemcpyHostToDevice);
  return static_cast<int>(status);
}

static int gpuPortEnsureValues(long count) {
  cudaError_t status;

  if (gpuPortValuesCapacity >= count)
    return static_cast<int>(cudaSuccess);
  gpuFreeDevice(gpuPortValues);
  gpuPortValues = NULL;
  gpuPortValuesCapacity = 0;
  status = cudaMalloc(&gpuPortValues, count * sizeof(*gpuPortValues));
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuPortValuesCapacity = count;
  return static_cast<int>(cudaSuccess);
}

extern "C" int gpuCudaPortCounters(unsigned long long *counters) {
  if (!counters || !gpuPortCounters)
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaMemcpy(counters, gpuPortCounters, GPU_PORT_COUNTERS * sizeof(*counters),
                                     cudaMemcpyDeviceToHost));
}

extern "C" int gpuCudaPortFlags(unsigned char *flags, long nParticles) {
  if (!flags || !gpuPortFlags || gpuPortFlagsCapacity < nParticles)
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaMemcpy(flags, gpuPortFlags, nParticles * sizeof(*flags), cudaMemcpyDeviceToHost));
}

/* A value a kernel left in the scratch array, such as the time of the particle in error */
extern "C" int gpuCudaPortValue(long index, double *value) {
  if (!value || !gpuPortValues || index < 0 || index >= gpuPortValuesCapacity)
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaMemcpy(value, gpuPortValues + index, sizeof(*value), cudaMemcpyDeviceToHost));
}

__global__ void gpuPortGatherKernel(double *coord, const double *rows, const long long *perm, long nParticles,
                                    int stride) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  for (int ic = 0; ic < stride; ic++)
    coord[ip * stride + ic] = rows[perm[ip] * stride + ic];
}

/* Rows in the given order: row i of the result is row perm[i] of the beam */
extern "C" int gpuCudaPortPermute(void *coord, long nParticles, int stride, const long long *permHost) {
  cudaError_t status;
  int threads = 256;

  if (!coord || !permHost || nParticles <= 0 || stride <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuPortPermCapacity < nParticles) {
    gpuFreeDevice(gpuPortPerm);
    gpuPortPerm = NULL;
    gpuPortPermCapacity = 0;
    status = cudaMalloc(&gpuPortPerm, nParticles * sizeof(*gpuPortPerm));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuPortPermCapacity = nParticles;
  }
  if (gpuPortRowsCapacity < nParticles * stride) {
    gpuFreeDevice(gpuPortRows);
    gpuPortRows = NULL;
    gpuPortRowsCapacity = 0;
    status = cudaMalloc(&gpuPortRows, nParticles * stride * sizeof(*gpuPortRows));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuPortRowsCapacity = nParticles * stride;
  }
  status = cudaMemcpy(gpuPortPerm, permHost, nParticles * sizeof(*gpuPortPerm), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(gpuPortRows, coord, nParticles * stride * sizeof(*gpuPortRows), cudaMemcpyDeviceToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuPortGatherKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), gpuPortRows, gpuPortPerm, nParticles, stride);
  return static_cast<int>(cudaGetLastError());
}

static int gpuPortLaunchFinish(cudaEvent_t start, cudaEvent_t stop, float *milliseconds) {
  return finishTimedKernel(start, stop, milliseconds);
}

/* A particle polynomial_kicks() or polynomial_hamiltonian() removes: the offsets already applied stay, as
 * on the CPU, then its z and absolute momentum are stored */
__device__ __forceinline__ void gpuPortMarkLost(double *coord, double zStart, double Po, unsigned char *flags,
                                                long ip, unsigned long long *counters) {
  coord[4] = zStart;
  coord[5] = Po * (1 + coord[5]);
  flags[ip] = 1;
  atomicAdd(&counters[0], 1ULL);
}

/* A survivor whose x or y is no longer finite: do_tracking()'s limit_amplitudes() removes it after the
 * element, which the device does only when told to (gpuPortInvalidPending) */
__device__ __forceinline__ void gpuPortCountInvalid(const double *coord, unsigned long long *counters) {
  if (!isfinite(coord[0]) || !isfinite(coord[2]))
    atomicAdd(&counters[3], 1ULL);
}

/* ---------------------------------------------------------------------------------------------------- */
/* KPOLY: polynomial_kicks() */

__global__ void gpuKpolyKernel(double *particles, long nParticles, int stride, GPU_KPOLY_DATA d,
                               unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, KL, x, y, xp, yp;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  flags[ip] = 0;
  KL = d.KL0 / (1 + coord[5]);
  coord[4] += d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  coord[0] += -d.dx + d.dz * coord[1];
  coord[2] += -d.dy + d.dz * coord[3];

  x = d.cosTilt * coord[0] + d.sinTilt * coord[2];
  y = -d.sinTilt * coord[0] + d.cosTilt * coord[2];
  xp = d.cosTilt * coord[1] + d.sinTilt * coord[3];
  yp = -d.sinTilt * coord[1] + d.cosTilt * coord[3];

  if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp) ||
      fabs(x) > d.coordLimit || fabs(y) > d.coordLimit ||
      fabs(xp) > d.slopeLimit || fabs(yp) > d.slopeLimit) {
    gpuPortMarkLost(coord, d.zStart, d.Po, flags, ip, counters);
    return;
  }

  if (d.yplane)
    yp += KL * gpuCpuIpow(y, d.order);
  else
    xp += KL * gpuCpuIpow(x, d.order);

  coord[1] = d.cosTilt * xp - d.sinTilt * yp;
  coord[3] = d.sinTilt * xp + d.cosTilt * yp;

  coord[0] += d.dx - coord[1] * d.dz;
  coord[2] += d.dy - coord[3] * d.dz;
  coord[4] -= d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaKpoly(void *coord, long nParticles, int stride, const GPU_KPOLY_DATA *data,
                            float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  int status, threads = 128;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  status = gpuPortPrepare(nParticles, 1);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = static_cast<int>(cudaEventRecord(start, 0));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuKpolyKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, *data, gpuPortFlags, gpuPortCounters);
  return gpuPortLaunchFinish(start, stop, milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* HKPOLY: polynomial_hamiltonian() */

__global__ void gpuHkpolyKernel(double *particles, long nParticles, int stride, const GPU_HKPOLY_DATA *dp,
                                unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const GPU_HKPOLY_DATA &d = *dp;
  double *coord, x, y, xp, yp, qx, qy, delta, factor, dfactor;
  double xpow, ypow, qxpow, qypow, deltapow;
  double xpow1, ypow1, qxpow1, qypow1;
  long ir;
  int ix, iy, iqx, iqy, idelta;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  flags[ip] = 0;

  coord[4] += d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  coord[0] += -d.dx + d.dz * coord[1];
  coord[2] += -d.dy + d.dz * coord[3];

  x = d.cosTilt * coord[0] + d.sinTilt * coord[2];
  y = -d.sinTilt * coord[0] + d.cosTilt * coord[2];
  xp = d.cosTilt * coord[1] + d.sinTilt * coord[3];
  yp = -d.sinTilt * coord[1] + d.cosTilt * coord[3];

  if (fabs(x) > d.coordLimit || fabs(y) > d.coordLimit ||
      fabs(xp) > d.slopeLimit || fabs(yp) > d.slopeLimit ||
      isnan(x) || isnan(xp) || isnan(y) || isnan(yp)) {
    gpuPortMarkLost(coord, d.zStart, d.Po, flags, ip, counters);
    return;
  }

  factor = d.factor;
  delta = coord[5];
  gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, delta, d.expandHamiltonian);
  for (ir = 0; ir <= d.nRepeats; ir++) {
    dfactor = 1;
    if (ir == 0 || ir == d.nRepeats)
      dfactor = 0.5;
    if (d.driftType == 1) {
      qx /= (1 + delta);
      qy /= (1 + delta);
      qxpow = 1;
      qxpow1 = 0;
      for (iqx = 0; iqx <= d.iqxMax; iqx++) {
        qypow = 1;
        qypow1 = 0;
        for (iqy = 0; iqy <= d.iqyMax; iqy++) {
          if (iqx)
            x += factor * d.D[iqx][iqy] * qxpow1 * qypow * dfactor;
          if (iqy)
            y += factor * d.D[iqx][iqy] * qxpow * qypow1 * dfactor;
          qypow1 = qypow * (iqy + 1);
          qypow *= qy;
        }
        qxpow1 = qxpow * (iqx + 1);
        qxpow *= qx;
      }
      qx *= (1 + delta);
      qy *= (1 + delta);
    } else {
      qxpow = 1;
      qxpow1 = 0;
      for (iqx = 0; iqx <= d.iqxMax; iqx++) {
        qypow = 1;
        qypow1 = 0;
        for (iqy = 0; iqy <= d.iqyMax; iqy++) {
          deltapow = 1;
          for (idelta = 0; idelta <= d.ideltaMax; idelta++) {
            if (iqx)
              x += factor * d.E[iqx][iqy][idelta] * qxpow1 * qypow * deltapow * dfactor;
            if (iqy)
              y += factor * d.E[iqx][iqy][idelta] * qxpow * qypow1 * deltapow * dfactor;
            deltapow *= delta;
          }
          qypow1 = qypow * (iqy + 1);
          qypow *= qy;
        }
        qxpow1 = qxpow * (iqx + 1);
        qxpow *= qx;
      }
    }

    if (ir == d.nRepeats)
      break;

    xpow = 1;
    xpow1 = 0;
    for (ix = 0; ix <= d.ixMax; ix++) {
      ypow = 1;
      ypow1 = 0;
      for (iy = 0; iy <= d.iyMax; iy++) {
        if (ix)
          qx -= factor * d.K[ix][iy] * xpow1 * ypow;
        if (iy)
          qy -= factor * d.K[ix][iy] * xpow * ypow1;
        ypow1 = ypow * (iy + 1);
        ypow *= y;
      }
      xpow1 = xpow * (ix + 1);
      xpow *= x;
    }
  }
  /* convertMomentaToSlopes() leaves the slopes alone and warns when they are undefined */
  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, delta, d.expandHamiltonian))
    atomicAdd(&counters[1], 1ULL);

  coord[0] = d.cosTilt * x - d.sinTilt * y;
  coord[2] = d.sinTilt * x + d.cosTilt * y;
  coord[1] = d.cosTilt * xp - d.sinTilt * yp;
  coord[3] = d.sinTilt * xp + d.cosTilt * yp;

  coord[0] += d.dx - coord[1] * d.dz;
  coord[2] += d.dy - coord[3] * d.dz;
  coord[4] -= d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  gpuPortCountInvalid(coord, counters);
}

static GPU_HKPOLY_DATA *gpuHkpolyDeviceData = NULL;

extern "C" int gpuCudaHkpoly(void *coord, long nParticles, int stride, const GPU_HKPOLY_DATA *data,
                             float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  int status, threads = 128;
  cudaError_t cudaStatus;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  status = gpuPortPrepare(nParticles, 1);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  if (!gpuHkpolyDeviceData) {
    cudaStatus = cudaMalloc(&gpuHkpolyDeviceData, sizeof(*gpuHkpolyDeviceData));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
  }
  cudaStatus = cudaMemcpy(gpuHkpolyDeviceData, data, sizeof(*data), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
    return static_cast<int>(cudaStatus);
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = static_cast<int>(cudaEventRecord(start, 0));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuHkpolyKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, gpuHkpolyDeviceData, gpuPortFlags, gpuPortCounters);
  return gpuPortLaunchFinish(start, stop, milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* MBUMPER: track_through_mkicker()'s particle loop with its offsets and tilt around it, and
 * integrate_kick_multipole_ordn() (multipole.c) for one multipole, no radiation, apertures or edges, at
 * fourth order, as track_through_mkicker() calls it.  That routine's result is ignored by the caller, so a
 * particle it gives up on keeps the coordinates it had. */

__device__ __forceinline__ double gpuPortBetaFromDelta(double p, double delta) {
  p *= 1 + delta;
  return p / sqrt(p * p + 1);
}

/* integrate_kick_multipole_ordn(coord, dx, dy, 0, 0, Po, 0, 0, order, KnL, skew, nParts, -1, length, 4,
 * NULL, NULL, NULL, NULL, &dzLoss, NULL, 0, 0); returns 0 where the CPU returns 0 before storing, and
 * counts the warnings convertMomentaToSlopes() prints */
__device__ int gpuMbumperIntegrate(double *coord, const GPU_MBUMPER_DATA &d, double KnL,
                                   unsigned long long *counters) {
  const double beta = 1.25992104989487316477;
  double driftFrac[4], kickFrac[4];
  double xpow[GPU_MBUMPER_MAX_ORDER + 1], ypow[GPU_MBUMPER_MAX_ORDER + 1];
  double x, y, xp, yp, dp, qx, qy, s, drift, KnLp, dsh, deltaQx, deltaQy;
  int maxOrder;

  driftFrac[0] = 0.5 / (2 - beta);
  driftFrac[1] = (1 - beta) / (2 - beta) / 2;
  driftFrac[2] = (1 - beta) / (2 - beta) / 2;
  driftFrac[3] = 0.5 / (2 - beta);
  kickFrac[0] = 1.0 / (2 - beta);
  kickFrac[1] = -beta / (2 - beta);
  kickFrac[2] = 1.0 / (2 - beta);
  kickFrac[3] = 0.0;

  drift = d.length / d.nParts;
  x = coord[0];
  xp = coord[1];
  y = coord[2];
  yp = coord[3];
  s = 0;
  dp = coord[5];
  if (isnan(x) || isnan(xp) || isnan(y) || isnan(yp))
    return 0;
  if (fabs(x) > d.coordLimit || fabs(y) > d.coordLimit ||
      fabs(xp) > d.slopeLimit || fabs(yp) > d.slopeLimit)
    return 0;
  gpuMultipoleConvertSlopesToMomenta(&qx, &qy, xp, yp, dp, d.expandHamiltonian);
  maxOrder = d.order > 0 ? (int)d.order : 0;
  KnLp = KnL / d.nParts;
  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp, d.expandHamiltonian)) {
    atomicAdd(&counters[1], 1ULL);
    return 0;
  }
  for (long iKick = 0; iKick < d.nParts; iKick++) {
    dsh = 0;
    for (int step = 0; step < 4; step++) {
      if (drift) {
        dsh = drift * driftFrac[step];
        x += xp * dsh;
        y += yp * dsh;
        if (d.expandHamiltonian)
          s += dsh * (1 + (xp * xp + yp * yp) / 2);
        else
          s += dsh * sqrt(1 + xp * xp + yp * yp);
      }
      if (!kickFrac[step])
        break;
      gpuMultipoleFillPowerArray(x, xpow, maxOrder);
      gpuMultipoleFillPowerArray(y, ypow, maxOrder);
      deltaQx = deltaQy = 0;
      if (KnL)
        gpuMultipoleApplyKick(&qx, &qy, &deltaQx, &deltaQy, xpow, ypow, (int)d.order, KnLp * kickFrac[step], 0);
      if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp, d.expandHamiltonian)) {
        atomicAdd(&counters[1], 1ULL);
        return 0;
      }
    }
  }
  if (!gpuMultipoleConvertMomentaToSlopes(&xp, &yp, qx, qy, dp, d.expandHamiltonian)) {
    atomicAdd(&counters[1], 1ULL);
    return 0;
  }
  coord[0] = x;
  coord[1] = xp;
  coord[2] = y;
  coord[3] = yp;
  coord[4] += s;
  coord[5] = dp;
  return 1;
}

__global__ void gpuMbumperKernel(double *particles, long nParticles, int stride, GPU_MBUMPER_DATA d,
                                 unsigned long long *counters, double *errorTime) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, strength, time, amplitude, KnL, x, xp, y, yp;
  long i, n;

  if (ip >= nParticles)
    return;
  part = particles + ip * stride;
  /* offsetBeamCoordinatesForMisalignment(dx, dy, dz) */
  if (d.doOffset) {
    part[4] += d.dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - d.dx + d.dz * part[1];
    part[2] = part[2] - d.dy + d.dz * part[3];
  }
  /* rotateBeamCoordinatesForMisalignment(tilt) */
  if (d.doTilt) {
    x = part[0];
    xp = part[1];
    y = part[2];
    yp = part[3];
    part[0] = x * d.cosTilt + y * d.sinTilt;
    part[2] = -x * d.sinTilt + y * d.cosTilt;
    part[1] = xp * d.cosTilt + yp * d.sinTilt;
    part[3] = -xp * d.sinTilt + yp * d.cosTilt;
  }

  strength = d.strength;
  time = part[4] / (d.cMks * gpuPortBetaFromDelta(d.pCentral, part[5])) + d.clockOffset - d.timeOffset;
  if (time > d.tmax || time < d.tmin) {
    if (d.periodic) {
      n = (long)((time - d.tmin) / (d.tmax - d.tmin));
      time -= n * (d.tmax - d.tmin);
    }
  }
  amplitude = 0;
  if (time > d.tmax || time < d.tmin)
    strength *= 0;
  else {
    for (i = 1; i < d.nWaveform; i++)
      if (d.tWaveform[i] >= time)
        break;
    if (i == d.nWaveform) {
      /* the CPU stops with an error at the first such particle */
      atomicMin(&counters[2], (unsigned long long)ip);
      errorTime[ip] = time;
      return;
    }
    i--;
    amplitude = (d.ampWaveform[i + 1] - d.ampWaveform[i]) / (d.tWaveform[i + 1] - d.tWaveform[i]) *
                  (time - d.tWaveform[i]) + d.ampWaveform[i];
    strength *= amplitude;
  }
  KnL = strength * d.length;
  gpuMbumperIntegrate(part, d, KnL, counters);

  /* rotateBeamCoordinatesForMisalignment(-tilt) */
  if (d.doTilt) {
    x = part[0];
    xp = part[1];
    y = part[2];
    yp = part[3];
    part[0] = x * d.cosUntilt + y * d.sinUntilt;
    part[2] = -x * d.sinUntilt + y * d.cosUntilt;
    part[1] = xp * d.cosUntilt + yp * d.sinUntilt;
    part[3] = -xp * d.sinUntilt + yp * d.cosUntilt;
  }
  /* offsetBeamCoordinatesForMisalignment(-dx, -dy, -dz) */
  if (d.doOffset) {
    part[4] += -d.dz * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - -d.dx + -d.dz * part[1];
    part[2] = part[2] - -d.dy + -d.dz * part[3];
  }
  gpuPortCountInvalid(part, counters);
}

static double *gpuMbumperWaveform = NULL;
static long gpuMbumperWaveformCapacity = 0;
static const double *gpuMbumperWaveformHostT = NULL, *gpuMbumperWaveformHostA = NULL;
static long gpuMbumperWaveformN = 0;

extern "C" int gpuCudaMbumper(void *coord, long nParticles, int stride, const GPU_MBUMPER_DATA *data,
                              const double *tWaveform, const double *ampWaveform, float *milliseconds) {
  static cudaEvent_t start = NULL, stop = NULL;
  GPU_MBUMPER_DATA d;
  cudaError_t cudaStatus;
  int status, threads = 128;

  if (!coord || !data || !tWaveform || !ampWaveform || nParticles <= 0 || stride < 6 || data->nWaveform < 2)
    return static_cast<int>(cudaErrorInvalidValue);
  if (milliseconds)
    *milliseconds = 0;
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortEnsureValues(nParticles);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* the waveform, uploaded again only when the element's table changes */
  if (gpuMbumperWaveformCapacity < 2 * data->nWaveform) {
    gpuFreeDevice(gpuMbumperWaveform);
    gpuMbumperWaveform = NULL;
    gpuMbumperWaveformCapacity = 0;
    gpuMbumperWaveformHostT = gpuMbumperWaveformHostA = NULL;
    cudaStatus = cudaMalloc(&gpuMbumperWaveform, 2 * data->nWaveform * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMbumperWaveformCapacity = 2 * data->nWaveform;
  }
  if (gpuMbumperWaveformHostT != tWaveform || gpuMbumperWaveformHostA != ampWaveform ||
      gpuMbumperWaveformN != data->nWaveform) {
    cudaStatus = cudaMemcpy(gpuMbumperWaveform, tWaveform, data->nWaveform * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMemcpy(gpuMbumperWaveform + data->nWaveform, ampWaveform, data->nWaveform * sizeof(double),
                              cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuMbumperWaveformHostT = tWaveform;
    gpuMbumperWaveformHostA = ampWaveform;
    gpuMbumperWaveformN = data->nWaveform;
  }
  d = *data;
  d.tWaveform = gpuMbumperWaveform;
  d.ampWaveform = gpuMbumperWaveform + data->nWaveform;
  status = getCachedTimingEvents(&start, &stop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  status = static_cast<int>(cudaEventRecord(start, 0));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuMbumperKernel<<<static_cast<int>((nParticles + threads - 1) / threads), threads>>>(
    static_cast<double *>(coord), nParticles, stride, d, gpuPortCounters, gpuPortValues);
  return gpuPortLaunchFinish(start, stop, milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* Shared by the kernels below: one pair of timing events, and the launch */

static cudaEvent_t gpuPortStart = NULL, gpuPortStop = NULL;

static int gpuPortBegin(float *milliseconds) {
  int status;

  if (milliseconds)
    *milliseconds = 0;
  status = getCachedTimingEvents(&gpuPortStart, &gpuPortStop);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  return static_cast<int>(cudaEventRecord(gpuPortStart, 0));
}

static int gpuPortEnd(float *milliseconds) {
  return finishTimedKernel(gpuPortStart, gpuPortStop, milliseconds);
}

static int gpuPortBlocks(long nParticles, int threads) {
  return static_cast<int>((nParticles + threads - 1) / threads);
}

/* rotateBeamCoordinatesForMisalignment() for one particle (no spin: these types refuse spin tracking) */
__device__ __forceinline__ void gpuPortRotate(double *coord, double cosA, double sinA) {
  double x = coord[0], xp = coord[1], y = coord[2], yp = coord[3];

  coord[0] = x * cosA + y * sinA;
  coord[2] = -x * sinA + y * cosA;
  coord[1] = xp * cosA + yp * sinA;
  coord[3] = -xp * sinA + yp * cosA;
}

/* ---------------------------------------------------------------------------------------------------- */
/* RMDF: track_through_ramped_deflector()'s particle loop */

__global__ void gpuRmdfKernel(double *particles, long nParticles, int stride, GPU_RMDF_DATA d,
                                unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, x, xp, y, yp, dxp, dyp, beta, dbeta_r, dr, beta_z, gamma, pc, t_part;
  double Estrengthp, zstart, zend, zs2, ze2, f1, Zo;
  long is;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  x = coord[0] - d.dx;
  xp = coord[1];
  y = coord[2] - d.dy;
  yp = coord[3];
  pc = d.pcCentral * (1 + coord[5]);
  beta = pc / (gamma = sqrt(1 + pc * pc));
  beta_z = beta / sqrt(1 + xp * xp + yp * yp);
  t_part = coord[4] / (d.cMks * beta);
  Zo = (t_part - d.tNull) * beta_z * d.cMks;
  zstart = 0;
  zend = d.length;
  for (is = 0; is < d.nSections; is++, zstart = zend, zend += d.length) {
    Estrengthp = d.Estrength / (gamma * (beta_z * beta_z));
    ze2 = (zend * zend) / 2;
    zs2 = (zstart * zstart) / 2;
    f1 = zs2 + Zo * zstart;
    dbeta_r = Estrengthp * ((ze2 + Zo * zend) - f1);
    dr = Estrengthp / beta_z * ((ze2 * zend / 3 + Zo * ze2 - zend * f1) - (zs2 * zstart / 3 + Zo * zs2 - zstart * f1));
    dxp = dbeta_r * d.cosTilt / beta_z;
    dyp = dbeta_r * d.sinTilt / beta_z;
    t_part += d.length / (d.cMks * beta_z);
    beta_z -= (gamma * gamma) * (beta_z * beta_z * beta_z) * (xp * dxp + yp * dyp) /
              (1 + (beta_z * gamma) * (beta_z * gamma));
    x += xp * d.length + dr * d.cosTilt;
    y += yp * d.length + dr * d.sinTilt;
    xp += dxp;
    yp += dyp;
    beta = beta_z * sqrt(1 + xp * xp + yp * yp);
    pc = beta / sqrt(1 - beta * beta);
    gamma = sqrt(1 + pc * pc);
  }
  coord[0] = x + d.dx;
  coord[1] = xp;
  coord[2] = y + d.dy;
  coord[3] = yp;
  coord[4] = t_part * d.cMks * beta;
  coord[5] = (pc - d.pcCentral) / d.pcCentral;
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaRmdf(void *coord, long nParticles, int stride, const GPU_RMDF_DATA *data,
                           float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRmdfKernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride,
                                                        *data, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* MRFDF: track_through_multipole_deflector()'s particle loop, with the tilt around it.  CUDA's sin and cos,
 * so the particles differ from elegant's by rounding. */

__global__ void gpuMrfdfKernel(double *particles, long nParticles, int stride, GPU_MRFDF_DATA d,
                                unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, x, xp, y, yp, beta, px, py, pz, pc, omega, k, t_part;
  double ptPhaseFactor, pzPhaseFactor, dpx, dpy, dpz, phase;
  int mode;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosTilt, d.sinTilt);
  if (!((d.startPID >= 0 && coord[6] < d.startPID) || (d.endPID >= 0 && coord[6] > d.endPID))) {
    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    pc = d.pcCentral * (1 + coord[5]);
    pz = pc / sqrt(1 + xp * xp + yp * yp);
    px = xp * pz;
    py = yp * pz;
    beta = pc / sqrt(1 + pc * pc);
    t_part = coord[4] / (d.cMks * beta);
    dpx = dpy = dpz = 0;
    for (mode = 0; mode < 5; mode++) {
      if (d.frequency[mode] == 0 || (d.a[mode] == 0 && d.b[mode] == 0))
        continue;
      omega = 2 * d.pi * d.frequency[mode];
      k = omega / d.cMks;
      phase = d.phase[mode] * d.pi / 180.0 + omega * (t_part - d.tFirst);
      ptPhaseFactor = cos(phase) / k;
      pzPhaseFactor = sin(phase);
      switch (mode) {
      case 0:
        dpx += d.b[0] * ptPhaseFactor;
        dpy -= d.a[0] * ptPhaseFactor;
        dpz += (d.b[0] * x - d.a[0] * y) * pzPhaseFactor;
        break;
      case 1:
        dpx += 2 * (d.b[1] * x - d.a[1] * y) * ptPhaseFactor;
        dpy -= 2 * (d.b[1] * y + d.a[1] * x) * ptPhaseFactor;
        dpz += (d.b[1] * (x * x - y * y) - 2 * d.a[1] * x * y) * pzPhaseFactor;
        break;
      case 2:
        dpx += (d.b[2] * (3 * x * x - 3 * y * y) + d.a[2] * (-6 * x * y)) * ptPhaseFactor;
        dpy -= (d.b[2] * 6 * x * y + d.a[2] * (3 * x * x - 3 * y * y)) * ptPhaseFactor;
        dpz += (d.b[2] * ((x * x * x) - 3 * x * y * y) - d.a[2] * (3 * x * x * y - (y * y * y))) * pzPhaseFactor;
        break;
      case 3:
        dpx += (d.b[3] * (4 * (x * x * x) - 12 * x * y * y) + d.a[3] * (-12 * x * x * y + 4 * (y * y * y))) *
               ptPhaseFactor;
        dpy -= (d.b[3] * (12 * x * x * y - 4 * (y * y * y)) + d.a[3] * (4 * (x * x * x) - 12 * x * y * y)) *
               ptPhaseFactor;
        dpz += (d.b[3] * (((x * x) * (x * x)) - 6 * ((x * y) * (x * y)) + ((y * y) * (y * y))) -
                4 * d.a[3] * (x * x - y * y) * x * y) * pzPhaseFactor;
        break;
      case 4: {
        double x2 = x * x, y2 = y * y;
        double x5 = x2 * x2 * x, y5 = y2 * y2 * y;
        dpx += (d.b[4] * (5 * ((y * y) * (y * y)) + 5 * ((x * x) * (x * x)) - 30 * ((x * y) * (x * y))) +
                d.a[4] * (-20 * (x * x * x) * y + 20 * x * (y * y * y))) * ptPhaseFactor;
        dpy -= (d.b[4] * (20 * (x * x * x) * y - 20 * x * (y * y * y)) +
                d.a[4] * (5 * ((x * x) * (x * x)) - 30 * ((x * y) * (x * y)) + 5 * ((y * y) * (y * y)))) *
               ptPhaseFactor;
        dpz += (d.b[4] * (x5 - 10 * (x * x * x) * y * y + 5 * x * ((y * y) * (y * y))) -
                d.a[4] * (y5 + 5 * ((x * x) * (x * x)) * y - 10 * y * ((x * y) * (x * y)))) * pzPhaseFactor;
        break;
      }
      }
    }
    px += dpx * d.factor * d.charge / d.massC2;
    py += dpy * d.factor * d.charge / d.massC2;
    pz = sqrt(pc * pc - px * px - py * py);
    pz += dpz * d.factor * d.charge / d.massC2;
    pc = sqrt(px * px + py * py + pz * pz);
    xp = px / pz;
    yp = py / pz;
    beta = pc / sqrt(1 + pc * pc);
    coord[0] = x;
    coord[1] = xp;
    coord[2] = y;
    coord[3] = yp;
    coord[4] = t_part * d.cMks * beta;
    coord[5] = (pc - d.pcCentral) / d.pcCentral;
  }
  if (d.doTilt)
    gpuPortRotate(coord, d.cosUntilt, d.sinUntilt);
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaMrfdf(void *coord, long nParticles, int stride, const GPU_MRFDF_DATA *data,
                            float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuMrfdfKernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride,
                                                        *data, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* SHRFDF: track_through_space_harmonic_deflector()'s particle loop, with the tilt around it.  CUDA's sin and
 * cos: rounding. */

__global__ void gpuShrfdfKernel(double *particles, long nParticles, int stride, GPU_SHRFDF_DATA d,
                                unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, x, xp, y, yp, beta, px, py, pz, pc, omega, k, t_part, k_0, alpha_sq, dpx, dpy, dpz, phase;
  int mode;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosTilt, d.sinTilt);
  x = coord[0];
  xp = coord[1];
  y = coord[2];
  yp = coord[3];
  pc = d.pcCentral * (1 + coord[5]);
  pz = pc / sqrt(1 + xp * xp + yp * yp);
  px = xp * pz;
  py = yp * pz;
  beta = pc / sqrt(1 + pc * pc);
  t_part = coord[4] / (d.cMks * beta);
  dpx = dpy = dpz = 0;
  k_0 = d.periodPhase / d.periodLength;
  omega = k_0 * d.cMks;
  phase = omega * (t_part - d.tFirst) + d.phase[0];
  dpx += d.v[0] * sin(phase);
  dpz += d.v[0] * cos(phase) * k_0 * x;
  for (mode = 1; mode < 10; mode++) {
    if (d.v[mode] == 0)
      continue;
    k = (d.periodPhase + 2 * d.pi * mode) / d.periodLength;
    alpha_sq = k * k - k_0 * k_0;
    omega = k * d.cMks;
    phase = omega * (t_part - d.tFirst) + d.phase[mode];
    dpx += d.v[mode] * sin(phase) * (0.5 + 0.0625 * alpha_sq * (3.0 * x * x + y * y));
    dpz += d.v[mode] * k * x * cos(phase) * (0.5 + 0.0625 * alpha_sq * (x * x + y * y));
  }
  px += dpx * d.factor * d.charge / d.massC2;
  py += dpy * d.factor * d.charge / d.massC2;
  pz = sqrt(pc * pc - px * px - py * py);
  pz += dpz * d.factor * d.charge / d.massC2;
  pc = sqrt(px * px + py * py + pz * pz);
  xp = px / pz;
  yp = py / pz;
  beta = pc / sqrt(1 + pc * pc);
  coord[0] = x;
  coord[1] = xp;
  coord[2] = y;
  coord[3] = yp;
  coord[4] = t_part * d.cMks * beta;
  coord[5] = (pc - d.pcCentral) / d.pcCentral;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosUntilt, d.sinUntilt);
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaShrfdf(void *coord, long nParticles, int stride, const GPU_SHRFDF_DATA *data,
                             float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuShrfdfKernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride,
                                                        *data, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* RFTM110: track_through_rftm110_deflector()'s particle loop, with the tilt around it.  CUDA's sin and cos:
 * rounding. */

__global__ void gpuRftm110Kernel(double *particles, long nParticles, int stride, GPU_RFTM110_DATA d,
                                unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, x, xp, y, yp, rho, beta, px, py, pz, beta_x, beta_y, beta_z, pc, phase, Ez, cBx, cBy;
  double cos_phi, sin_phi, krho2, krho4, krho6, cos_2phi, t_part;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosTilt, d.sinTilt);
  x = coord[0];
  y = coord[2];
  if ((rho = sqrt(x * x + y * y)) > 0) {
    cos_phi = x / rho;
    sin_phi = y / rho;
  } else {
    cos_phi = 1;
    sin_phi = 0;
  }
  xp = coord[1];
  yp = coord[3];
  pc = d.pcCentral * (1 + coord[5]);
  pz = pc / sqrt(1 + xp * xp + yp * yp);
  px = xp * pz;
  py = yp * pz;
  beta = pc / sqrt(1 + pc * pc);
  t_part = coord[4] / (d.cMks * beta);
  phase = d.omega * t_part + d.phase0;
  krho2 = (d.k * rho) * (d.k * rho);
  krho4 = krho2 * krho2;
  krho6 = krho2 * krho4;
  cos_2phi = 2 * (cos_phi * cos_phi) - 1;
  Ez = -d.voltTimes2 * d.k * rho * (192 - 24 * krho2 + krho4) * cos_phi * sin(phase) / 384.;
  cBx = d.voltTimes2 * krho2 * (384 - 32 * krho2 + krho4) * cos_phi * cos(phase) * sin_phi / 3072.;
  cBy = -d.voltTimes2 *
        (-9216 + 2304 * krho2 - 144 * krho4 + 4 * krho6 + 1152 * krho2 * cos_2phi - 96 * krho4 * cos_2phi +
         3 * krho6 * cos_2phi) *
        cos(phase) / 18432.;
  beta_x = px / pc;
  beta_y = py / pc;
  beta_z = pz / pc;
  px += beta_z * cBy;
  py += -beta_z * cBx;
  pz += -(Ez + beta_x * cBy - beta_y * cBx);
  xp = px / pz;
  yp = py / pz;
  pc = sqrt(px * px + py * py + pz * pz);
  beta = pc / sqrt(1 + pc * pc);
  coord[0] = x;
  coord[1] = xp;
  coord[2] = y;
  coord[3] = yp;
  coord[4] = t_part * d.cMks * beta;
  coord[5] = (pc - d.pcCentral) / d.pcCentral;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosUntilt, d.sinUntilt);
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaRftm110(void *coord, long nParticles, int stride, const GPU_RFTM110_DATA *data,
                              float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 7)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRftm110Kernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride,
                                                        *data, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* ALPH: alpha_magnet_tracking()'s puck and slit filters.  Its matrix goes through track_particles(). */

__global__ void gpuAlphFilterKernel(double *particles, long nParticles, int stride, GPU_ALPH_FILTER_DATA d,
                                    unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord;
  int lost;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.slits)
    lost = (d.doXl && coord[0] <= d.xl) || (d.doXu && coord[0] >= d.xu);
  else
    lost = coord[0] >= d.xl && coord[0] <= d.xu;
  flags[ip] = lost ? 1 : 0;
  if (lost) {
    coord[4] = d.z;
    coord[5] = d.Po * (1 + coord[5]);
    atomicAdd(&counters[0], 1ULL);
  }
}

extern "C" int gpuCudaAlphFilter(void *coord, long nParticles, int stride, const GPU_ALPH_FILTER_DATA *data,
                                 float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 1);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuAlphFilterKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride,
                                                                *data, gpuPortFlags, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* KSBEND: track_through_kick_sbend(), kick_sbend_derivs(), integrate_ord2(), mmid() (mdbmth) and
 * apply_edge_effects() (csbend.c).  The edge's tan(e - psi/(1+dp)) is CUDA's when psi is not zero, so with
 * a fringe integral the particles differ from elegant's by rounding. */

struct GpuKsbendState {
  int lost;
  double sLost;
};

__device__ void gpuKsbendDerivs(double *dQds, const double *Q, double s0, const GPU_KSBEND_DATA &d,
                                GpuKsbendState &st) {
  double xp, yp, x, y, Fx, Fy, factor;
  double one_plus_hx, xp2, yp2, Tp2, Tp2_yp2, Tp, y2, one_plus_dp, xpyp;

  if (st.lost) {
    dQds[0] = dQds[1] = dQds[2] = dQds[3] = dQds[4] = dQds[5] = 0;
    return;
  }
  x = Q[0];
  xp = dQds[0] = Q[1];
  y = Q[2];
  yp = dQds[2] = Q[3];
  if (fabs(xp) > d.slopeLimit || fabs(yp) > d.slopeLimit || fabs(x) > d.coordLimit || fabs(y) > d.coordLimit ||
      isnan(xp) || isnan(yp) || isnan(x) || isnan(y)) {
    st.lost = 1;
    st.sLost = s0;
    dQds[0] = dQds[1] = dQds[2] = dQds[3] = dQds[4] = dQds[5] = 0;
    return;
  }
  one_plus_hx = 1 + d.h * x;
  if (d.paraxial) {
    xp = yp = 0;
    xp2 = yp2 = xpyp = 0;
    Tp = one_plus_hx;
    Tp2 = one_plus_hx * one_plus_hx;
    Tp2_yp2 = Tp2;
  } else {
    xp2 = xp * xp;
    yp2 = yp * yp;
    xpyp = xp * yp;
    Tp2_yp2 = xp2 + one_plus_hx * one_plus_hx;
    Tp2 = Tp2_yp2 + yp2;
    Tp = sqrt(Tp2);
  }
  one_plus_dp = 1 + Q[5];
  y2 = y * y;
  Fx = (d.Fx_y + (d.Fx_x_y + (d.Fx_x2_y + d.Fx_x3_y * x) * x) * x + (d.Fx_y3 + d.Fx_x_y3 * x) * y2) * y;
  Fy = d.Fy_0 + (d.Fy_x + (d.Fy_x2 + (d.Fy_x3 + d.Fy_x4 * x) * x) * x) * x +
       (d.Fy_y2 + (d.Fy_x_y2 + d.Fy_x2_y2 * x) * x + d.Fy_y4 * y2) * y2;
  factor = Tp * d.ha / one_plus_dp;
  dQds[1] = ((Fx * xpyp - Fy * (Tp2_yp2)) * factor + d.h * (Tp2_yp2 + xp2)) / one_plus_hx;
  dQds[3] = ((Fx * (Tp2 - xp2) - Fy * xpyp) * factor + 2 * d.h * xpyp) / one_plus_hx;
  dQds[4] = Tp;
  if (d.radCoef)
    dQds[5] = -d.radCoef * (Fx * Fx + Fy * Fy) * (one_plus_dp * one_plus_dp) * Tp;
  else
    dQds[5] = 0;
}

/* apply_edge_effects(); tanBeta and secBeta are the host's tan(beta) and 1/cos(beta) */
__device__ void gpuKsbendEdge(double *x, double *xp, double *y, double *yp, double rho, double n, double beta,
                              double tanBeta, double secBeta, double he, double psi, long which_edge) {
  double h, tan_beta, tan2_beta, sec_beta, sec2_beta, h2;
  double R21, R43;
  double T111, T133, T211, T441, T331, T221, T233, T243, T431, T432;
  double x0, xp0, y0, yp0;

  h = 1 / rho;
  R21 = h * (tan_beta = tanBeta);
  R43 = -h * (psi == 0 ? tanBeta : tan(beta - psi));
  h2 = h * h;
  T111 = which_edge * h / 2 * (tan2_beta = tan_beta * tan_beta);
  sec_beta = secBeta;
  sec2_beta = sec_beta * sec_beta;
  T133 = -which_edge * h / 2 * sec2_beta;
  T211 = which_edge == -1 ? -n * h2 * tan_beta : -h2 * (n + tan2_beta / 2) * tan_beta;
  T441 = -(T331 = T221 = -which_edge * h * tan2_beta);
  T233 = which_edge == -1 ? h2 * (n + .5 + tan2_beta) * tan_beta : h2 * (n - tan2_beta / 2) * tan_beta;
  T243 = which_edge * h * tan2_beta;
  T431 = h2 * (2 * n + (which_edge == 1 ? sec2_beta : 0)) * tan_beta;
  T432 = which_edge * h * sec2_beta;
  if (he != 0) {
    double term;
    term = h / 2 * he * sec2_beta * sec_beta;
    T211 += term;
    T233 -= term;
    T431 -= 2 * term;
  }
  x0 = *x;
  xp0 = *xp;
  y0 = *y;
  yp0 = *yp;
  *x = x0 + T111 * (x0 * x0) + T133 * (y0 * y0);
  *xp = xp0 + R21 * x0 + T211 * (x0 * x0) + T221 * x0 * xp0 + T233 * (y0 * y0) + T243 * y0 * yp0;
  *y = y0 + T331 * x0 * y0;
  *yp = yp0 + R43 * y0 + T441 * yp0 * x0 + T431 * x0 * y0 + T432 * xp0 * y0;
}

/* tan(e - psi/(1 + dp)): the host's tan(e) when psi is zero, as the argument is then e itself */
__device__ __forceinline__ double gpuKsbendEdgeTan(double e, double tanE, double psi, double dp) {
  return psi == 0 ? tanE : tan(e - psi / (1 + dp));
}

__global__ void gpuKsbendKernel(double *particles, long nParticles, int stride, GPU_KSBEND_DATA d,
                                unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, x, xp, y, yp, dp, s, rho, Qi[6], Qf[6], dQds[6];
  GpuKsbendState st;
  long i;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  flags[ip] = 0;
  coord[4] += d.dzi * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  coord[0] = coord[0] + d.dxi + d.dzi * coord[1];
  coord[2] = coord[2] + d.dyi + d.dzi * coord[3];

  x = coord[0] * d.cosTtilt + coord[2] * d.sinTtilt;
  y = -coord[0] * d.sinTtilt + coord[2] * d.cosTtilt;
  xp = coord[1] * d.cosTtilt + coord[3] * d.sinTtilt;
  yp = -coord[1] * d.sinTtilt + coord[3] * d.cosTtilt;
  s = coord[4];
  dp = coord[5];

  rho = (1 + dp) * d.rho0;
  if (d.edge1) {
    if (d.edgeOrder > 1)
      gpuKsbendEdge(&x, &xp, &y, &yp, rho, d.n, d.e1, d.tanE1, d.secE1, d.he1, d.psi1 / (1 + dp), -1);
    else {
      xp += d.tanE1 / rho * x;
      yp -= gpuKsbendEdgeTan(d.e1, d.tanE1, d.psi1, dp) / rho * y;
    }
  }
  xp *= (1 + x / d.rho0);
  yp *= (1 + x / d.rho0);
  Qi[0] = x;
  Qi[1] = xp;
  Qi[2] = y;
  Qi[3] = yp;
  Qi[4] = 0;
  Qi[5] = dp;

  if (d.edge1 && d.e1 != 0 && d.radCoef) {
    st.lost = 0;
    st.sLost = 0;
    gpuKsbendDerivs(dQds, Qi, 0.0, d, st);
    Qi[5] -= dQds[5] * Qi[0] * d.tanE1;
  }
  st.lost = 0;
  st.sLost = 0;
  gpuKsbendDerivs(dQds, Qi, 0.0, d, st);
  if (!st.lost) {
    if (d.modifiedMidpoint) {
      /* mmid(Qi, dQds, 6, 0.0, length, n_kicks, Qf, kick_sbend_derivs) */
      double ym[6], yn[6], hstep, hTimes2, xval = 0, ynSave;
      long j;
      hTimes2 = (hstep = d.length / d.nKicks) * 2;
      for (i = 0; i < 6; i++) {
        ym[i] = Qi[i];
        yn[i] = Qi[i] + hstep * dQds[i];
      }
      for (j = 1; j < d.nKicks; j++) {
        xval = 0.0 + hstep * j;
        gpuKsbendDerivs(Qf, yn, xval, d, st);
        for (i = 0; i < 6; i++) {
          ynSave = yn[i];
          yn[i] = ym[i] + hTimes2 * Qf[i];
          ym[i] = ynSave;
        }
      }
      gpuKsbendDerivs(Qf, yn, xval + d.length, d, st);
      for (i = 0; i < 6; i++)
        Qf[i] = (ym[i] + yn[i] + hstep * Qf[i]) / 2;
    } else {
      /* integrate_ord2(Qf, Qi, length, n_kicks) */
      double ds, dsh, F[6];
      for (i = 0; i < 6; i++)
        Qf[i] = Qi[i];
      ds = d.length / d.nKicks;
      dsh = ds / 2;
      for (i = 0; i < d.nKicks; i++) {
        Qf[0] += dsh * Qf[1];
        Qf[2] += dsh * Qf[3];
        gpuKsbendDerivs(F, Qf, 0.0, d, st);
        Qf[1] += ds * F[1];
        Qf[3] += ds * F[3];
        Qf[0] += dsh * Qf[1];
        Qf[2] += dsh * Qf[3];
        Qf[4] += F[4] * ds;
        Qf[5] += F[5] * ds;
      }
    }
  }
  if (st.lost) {
    coord[4] = d.zStart + st.sLost;
    coord[5] = d.Po * (1 + coord[5]);
    flags[ip] = 1;
    atomicAdd(&counters[0], 1ULL);
    return;
  }

  if (d.edge2 && d.e2 != 0 && d.radCoef) {
    gpuKsbendDerivs(dQds, Qf, 0.0, d, st);
    Qf[5] -= dQds[5] * Qf[0] * d.tanE2;
  }
  if (d.radCoef) {
    double p0, p1, beta0, beta1;
    p0 = d.Po * (1 + dp);
    beta0 = p0 / sqrt(p0 * p0 + 1);
    p1 = d.Po * (1 + Qf[5]);
    beta1 = p1 / sqrt(p1 * p1 + 1);
    s = beta1 * (s / beta0 + 2 * Qf[4] / (beta0 + beta1));
  } else
    s += Qf[4];

  x = Qf[0];
  xp = Qf[1];
  y = Qf[2];
  yp = Qf[3];
  dp = Qf[5];
  xp /= (1 + x / d.rho0);
  yp /= (1 + x / d.rho0);

  rho = (1 + dp) * d.rho0;
  if (d.edge2) {
    if (d.edgeOrder > 1)
      gpuKsbendEdge(&x, &xp, &y, &yp, rho, d.n, d.e2, d.tanE2, d.secE2, d.he2, d.psi2 / (1 + dp), 1);
    else {
      xp += d.tanE2 / rho * x;
      yp -= gpuKsbendEdgeTan(d.e2, d.tanE2, d.psi2, dp) / rho * y;
    }
  }
  coord[0] = x * d.cosTtilt - y * d.sinTtilt + d.dcoordEtilt[0];
  coord[2] = x * d.sinTtilt + y * d.cosTtilt + d.dcoordEtilt[2];
  coord[1] = xp * d.cosTtilt - yp * d.sinTtilt + d.dcoordEtilt[1];
  coord[3] = xp * d.sinTtilt + yp * d.cosTtilt + d.dcoordEtilt[3];
  coord[4] = s;
  coord[5] = dp;
  coord[0] += d.dxf + d.dzf * coord[1];
  coord[2] += d.dyf + d.dzf * coord[3];
  coord[4] += d.dzf * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaKsbend(void *coord, long nParticles, int stride, const GPU_KSBEND_DATA *data,
                             float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6 || data->nKicks < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 1);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuKsbendKernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride,
                                                            *data, gpuPortFlags, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* BOFFAXE: trackMagneticFieldOffAxisExpansion()'s particle loop with the misalignment around it, and
 * computeMagneticFieldFromOffAxisExpansion() in its order (ipow5 to ipow9 as mdb.h's inline functions,
 * ipow() and dfactorial() from mdbmth) */

__device__ __forceinline__ double gpuPortIpow5(double x) { double y = x * x; return y * y * x; }
__device__ __forceinline__ double gpuPortIpow6(double x) { double y = x * x; return y * y * y; }
__device__ __forceinline__ double gpuPortIpow7(double x) { double y = x * x; return y * y * y * x; }
__device__ __forceinline__ double gpuPortIpow8(double x) { double y = x * x; y = y * y; return y * y; }
__device__ __forceinline__ double gpuPortIpow9(double x) { double y = x * x * x; return y * y * y; }
#define GPU_PORT_IPOW3(x) ((x) * (x) * (x))
#define GPU_PORT_IPOW4(x) (((x) * (x)) * ((x) * (x)))

__device__ __forceinline__ double gpuPortDfactorial(long n) {
  double prod = 1;
  while (n > 0)
    prod *= n--;
  return prod;
}

#define GPU_BOFFAXE_MAX_DZ1 16

__device__ void gpuBoffaxeField(double *B, double x, double y, long iz, long is, const GPU_BOFFAXE_DATA &d) {
  double fieldData[GPU_BOFFAXE_MAX_DZ1];
  double delz = 0;
  long i, j;
  int eo = d.expansionOrder;

  if (iz < 0 || iz >= d.nz)
    B[0] = B[1] = B[2] = 0;
  else {
    if (is && d.zSubdivisions > 1)
      delz = (1.0 * is) / d.zSubdivisions * d.dataDz;
    for (i = 0; i < d.nDz1; i++) {
      fieldData[i] = d.field[i * d.nz + iz];
      if (delz)
        for (j = i + 1; j < d.nDz1; j++)
          fieldData[i] += d.field[j * d.nz + iz] * gpuCpuIpow(delz, j - i) / gpuPortDfactorial(j - i);
    }
    if (d.order == 1) {
      B[0] = fieldData[0] * y;
      B[1] = fieldData[0] * x;
      B[2] = 0;
      if (d.nDz1 > 1 && (eo == 0 || eo > 1)) {
        B[2] = fieldData[1] * x * y;
        if (d.nDz1 > 2 && (eo == 0 || eo > 2)) {
          B[0] -= fieldData[2] * (x * x * y / 4 + GPU_PORT_IPOW3(y) / 12);
          B[1] -= fieldData[2] * (GPU_PORT_IPOW3(x) / 12 + x * y * y / 4);
          if (d.nDz1 > 3 && (eo == 0 || eo > 3)) {
            B[2] -= fieldData[3] * (GPU_PORT_IPOW3(x) * y + x * GPU_PORT_IPOW3(y)) / 12;
            if (d.nDz1 > 4 && (eo == 0 || eo > 4)) {
              B[0] += fieldData[4] * ((x * x) * GPU_PORT_IPOW3(y) / 36 + (5 * GPU_PORT_IPOW4(x) * y + gpuPortIpow5(y)) / 720);
              B[1] += fieldData[4] * (GPU_PORT_IPOW3(x) * (y * y) / 36 + (gpuPortIpow5(x) + 5 * x * GPU_PORT_IPOW4(y)) / 720);
              if (d.nDz1 > 5 && (eo == 0 || eo > 5)) {
                B[2] += fieldData[5] * (GPU_PORT_IPOW3(x * y) / 108 + (gpuPortIpow5(x) * y + x * gpuPortIpow5(y)) / 720);
                if (d.nDz1 > 6 && (eo == 0 || eo > 6)) {
                  B[0] -= fieldData[6] * (5 * GPU_PORT_IPOW4(x) * GPU_PORT_IPOW3(y) + 3 * (x * x) * gpuPortIpow5(y)) / 4320;
                  B[1] -= fieldData[6] * (3 * gpuPortIpow5(x) * (y * y) + 5 * GPU_PORT_IPOW3(x) * GPU_PORT_IPOW4(y)) / 4320;
                  if (d.nDz1 > 7 && (eo == 0 || eo > 7)) {
                    B[2] -= fieldData[7] * (gpuPortIpow5(x) * GPU_PORT_IPOW3(y) + GPU_PORT_IPOW3(x) * gpuPortIpow5(y)) / 4320;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      B[0] = fieldData[0] * x * y;
      B[1] = 0.5 * fieldData[0] * (x * x - y * y);
      B[2] = 0;
      if (d.nDz1 > 1 && (eo == 0 || eo > 1)) {
        B[2] = fieldData[1] * (0.5 * (x * x) * y - GPU_PORT_IPOW3(y) / 6);
        if (d.nDz1 > 2 && (eo == 0 || eo > 2)) {
          B[0] -= fieldData[2] * (x * GPU_PORT_IPOW3(y) / 24 + GPU_PORT_IPOW3(x) * y / 8);
          B[1] += fieldData[2] * (5 * GPU_PORT_IPOW4(y) / 96 - ((x * y) * (x * y)) / 16 - GPU_PORT_IPOW4(x) / 32);
          if (d.nDz1 > 3 && (eo == 0 || eo > 3)) {
            B[2] += fieldData[3] * (gpuPortIpow5(y) / 96 - (x * x) * GPU_PORT_IPOW3(y) / 48 - GPU_PORT_IPOW4(x) * y / 32);
            if (d.nDz1 > 4 && (eo == 0 || eo > 4)) {
              B[0] += fieldData[4] * (x * gpuPortIpow5(y) / 1920 + GPU_PORT_IPOW3(x * y) / 192 + 3 * gpuPortIpow5(x) * y / 640);
              B[1] += fieldData[4] * (-7 * gpuPortIpow6(y) / 3840 + ((x * (y * y)) * (x * (y * y))) / 768 +
                                      (((x * x) * y) * ((x * x) * y)) / 256 + gpuPortIpow6(x) / 1280);
              if (d.nDz1 > 5 && (eo == 0 || eo > 5)) {
                B[2] += fieldData[5] * (-gpuPortIpow7(y) / 3840 + (x * x) * gpuPortIpow5(y) / 3840 +
                                        GPU_PORT_IPOW4(x) * GPU_PORT_IPOW3(y) / 768 + gpuPortIpow6(x) * y / 1280);
                if (d.nDz1 > 6 && (eo == 0 || eo > 6)) {
                  B[0] += fieldData[6] * (-GPU_PORT_IPOW3(x) * gpuPortIpow5(y) / 11520 - gpuPortIpow5(x) * GPU_PORT_IPOW3(y) / 5760 -
                                          gpuPortIpow7(x) * y / 11520);
                  B[1] += fieldData[6] * (gpuPortIpow8(y) / 30720 - gpuPortIpow6(x) * (y * y) / 11520 -
                                          GPU_PORT_IPOW4(x * y) / 9216 - gpuPortIpow8(x) / 92160);
                  if (d.nDz1 > 7 && (eo == 0 || eo > 7)) {
                    B[2] += fieldData[7] * (gpuPortIpow9(y) / 276480 - GPU_PORT_IPOW4(x) * gpuPortIpow5(y) / 46080 -
                                            gpuPortIpow6(x) * GPU_PORT_IPOW3(y) / 34560 - gpuPortIpow8(x) * y / 92160);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for (i = 0; i < 3; i++)
    B[i] *= d.strength;
}

__global__ void gpuBoffaxeKernel(double *particles, long nParticles, int stride, GPU_BOFFAXE_DATA d,
                                 unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part, ds, x, y, xp, yp, delta, s, denom, B[3], p[3], pOrig;
  double xpTemp, ypTemp, xpNew, ypNew, xTemp, yTemp, xNew, yNew, preFactorDz;
  long iz, is;

  if (ip >= nParticles)
    return;
  part = particles + ip * stride;
  if (d.doOffset) {
    part[4] += d.dz0 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - d.dx + d.dz0 * part[1];
    part[2] = part[2] - d.dy + d.dz0 * part[3];
  }
  if (d.doTilt)
    gpuPortRotate(part, d.cosTilt, d.sinTilt);

  x = part[0];
  xp = part[1];
  y = part[2];
  yp = part[3];
  s = part[4];
  delta = part[5];
  x -= (d.zEntry - d.zMin) * xp;
  y -= (d.zEntry - d.zMin) * yp;
  s -= (d.zEntry - d.zMin) * sqrt(1.0 + xp * xp + yp * yp);
  denom = sqrt(1 + xp * xp + yp * yp);
  pOrig = d.pCentral * (1 + delta);
  p[2] = pOrig / denom;
  p[0] = xp * p[2];
  p[1] = yp * p[2];
  B[0] = B[1] = B[2] = 0;
  for (iz = 0; iz < d.nz - 1; iz += d.zInterval) {
    for (is = 0; is < d.zSubdivisions; is++) {
      denom = sqrt(1 + xp * xp + yp * yp);
      p[2] = d.pCentral * (1 + delta) / denom;
      p[0] = xp * p[2];
      p[1] = yp * p[2];
      gpuBoffaxeField(B, x, y, iz, is, d);
      preFactorDz = -d.dz * d.q * d.relSign / (d.pCentral * d.mass * d.cMks * (1.0 + delta));
      preFactorDz = preFactorDz * sqrt(1.0 + xp * xp + yp * yp);
      xTemp = x + d.dz * xp;
      yTemp = y + d.dz * yp;
      xpTemp = xp + preFactorDz * ((yp * B[2] - (1.0 + xp * xp) * B[1]) + xp * yp * B[0]);
      ypTemp = yp + preFactorDz * (((1.0 + yp * yp) * B[0] - xp * B[2]) - xp * yp * B[1]);
      ds = d.dz * sqrt(1 + xp * xp + yp * yp);
      if (is == (d.zSubdivisions - 1))
        gpuBoffaxeField(B, x, y, iz + d.zInterval, 0, d);
      else
        gpuBoffaxeField(B, x, y, iz, is + 1, d);
      preFactorDz = -d.dz * d.q * d.relSign / (d.pCentral * d.mass * d.cMks * (1.0 + delta));
      preFactorDz = preFactorDz * sqrt(1.0 + xpTemp * xpTemp + ypTemp * ypTemp);
      xNew = 0.5 * (x + xTemp + d.dz * xpTemp);
      yNew = 0.5 * (y + yTemp + d.dz * ypTemp);
      xpNew = 0.5 * (xp + xpTemp + preFactorDz * ((ypTemp * B[2] - (1.0 + xpTemp * xpTemp) * B[1]) + xpTemp * ypTemp * B[0]));
      ypNew = 0.5 * (yp + ypTemp + preFactorDz * (((1.0 + ypTemp * ypTemp) * B[0] - xpTemp * B[2]) - xpTemp * ypTemp * B[1]));
      ds = 0.5 * (ds + d.dz * sqrt(1 + xpTemp * xpTemp + ypTemp * ypTemp));
      x = xNew;
      y = yNew;
      xp = xpNew;
      yp = ypNew;
      s += ds;
      denom = sqrt(1 + xp * xp + yp * yp);
      p[2] = d.pCentral * (1 + delta) / denom;
      p[0] = xp * p[2];
      p[1] = yp * p[2];
      if (d.synchRad) {
        double B2 = B[0] * B[0] + B[1] * B[1];
        delta = delta - d.radCoef * d.pCentral * (1.0 + delta) * B2 * ds;
      }
    }
  }
  if (iz < (d.nz - 1)) {
    x += xp * d.dataDz * (d.nz - (iz - 1));
    y += yp * d.dataDz * (d.nz - (iz - 1));
    s += d.dataDz * (d.nz - (iz - 1)) * sqrt(1 + xp * xp + yp * yp);
  }
  x -= (d.zMax - d.zExit) * xp;
  y -= (d.zMax - d.zExit) * yp;
  s -= (d.zMax - d.zExit) * sqrt(1.0 + xp * xp + yp * yp);
  part[0] = x;
  part[1] = xp;
  part[2] = y;
  part[3] = yp;
  part[4] = s;
  if (d.synchRad) {
    double gamma0, beta0, gamma1, p1, beta1;
    gamma0 = sqrt((d.pCentral * (1 + part[5])) * (d.pCentral * (1 + part[5])) + 1);
    beta0 = d.pCentral * (1 + part[5]) / gamma0;
    p1 = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    gamma1 = sqrt(p1 * p1 + 1);
    beta1 = p1 / gamma1;
    part[4] *= beta1 / beta0;
    part[5] = p1 / d.pCentral - 1;
  }
  if (d.doTilt)
    gpuPortRotate(part, d.cosUntilt, d.sinUntilt);
  if (d.doOffset) {
    part[4] += -d.dz0 * sqrt(1 + part[1] * part[1] + part[3] * part[3]);
    part[0] = part[0] - -d.dx + -d.dz0 * part[1];
    part[2] = part[2] - -d.dy + -d.dz0 * part[3];
  }
  gpuPortCountInvalid(part, counters);
}

static double *gpuBoffaxeTable = NULL;
static long gpuBoffaxeTableCapacity = 0;
static const void *gpuBoffaxeTableKey = NULL;

extern "C" int gpuCudaBoffaxe(void *coord, long nParticles, int stride, const GPU_BOFFAXE_DATA *data,
                              double **fieldData, const void *tableKey, float *milliseconds) {
  GPU_BOFFAXE_DATA d;
  cudaError_t cudaStatus;
  long i, count;
  int status;

  if (!coord || !data || !fieldData || nParticles <= 0 || stride < 6 || data->nDz1 < 1 ||
      data->nDz1 > GPU_BOFFAXE_MAX_DZ1 || data->nz < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  count = data->nDz1 * data->nz;
  if (gpuBoffaxeTableCapacity < count) {
    gpuFreeDevice(gpuBoffaxeTable);
    gpuBoffaxeTable = NULL;
    gpuBoffaxeTableCapacity = 0;
    gpuBoffaxeTableKey = NULL;
    cudaStatus = cudaMalloc(&gpuBoffaxeTable, count * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuBoffaxeTableCapacity = count;
  }
  if (gpuBoffaxeTableKey != tableKey) {
    for (i = 0; i < data->nDz1; i++) {
      cudaStatus = cudaMemcpy(gpuBoffaxeTable + i * data->nz, fieldData[i], data->nz * sizeof(double),
                              cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess)
        return static_cast<int>(cudaStatus);
    }
    gpuBoffaxeTableKey = tableKey;
  }
  d = *data;
  d.field = gpuBoffaxeTable;
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuBoffaxeKernel<<<gpuPortBlocks(nParticles, 64), 64>>>(static_cast<double *>(coord), nParticles, stride, d,
                                                           gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* GFWIGGLER: APPLE_Track() and f_ijk().  CUDA's sin, cos, exp, cosh, sinh and pow: rounding. */

#define GPU_APPLE_AT(name, i, j) (d.table[d.name + (i) * d.NzHarm + (j)])

__device__ void gpuAppleFijk(double *X, int End_Pole, double step, double factor, const GPU_APPLE_DATA &d) {
  double Ax, Ay, xpp, xmm, kx, ky, Cm, Cp, Sm, Sp, Em, Ep;
  double Cx, Dx, Cy, Dy, Cxx, Dxx, Cyx, Dyx, Cxy, Dxy, Cyy, Dyy;
  double ccp, csp, ccm, csm, scp, ssp, scm, ssm;
  double f101, f101x, f101y, f011, f011x, f011y, f002, f002x, f002y, pn;
  double f003x, f003y;
  double Cxxx, Cxxy, Cxyx, Cxyy, Dxxx, Dxxy, Dxyx, Dxyy;
  double Cyxx, Cyxy, Cyyx, Cyyy, Dyxx, Dyxy, Dyyx, Dyyy;
  double Ax1, Ax2, Ax3, Ay1, Ay2, Ay3;
  double Ax1x, Ax2x, Ax3x, Ay1x, Ay2x, Ay3x;
  double Ax1y, Ax2y, Ax3y, Ay1y, Ay2y, Ay3y;
  long i, j;

  Dx = Dy = 0;
  xpp = X[0] + d.x0;
  xmm = X[0] - d.x0;
  f101 = f101x = f101y = 0;
  f011 = f011x = f011y = 0;
  f002 = f002x = f002y = 0;
  f003x = f003y = 0;
  pn = 1.;
  for (j = 0; j < d.NzHarm; j++) {
    Cx = Dx = Cy = Dy = 0;
    Cxx = Dxx = Cyx = Dyx = 0;
    Cxy = Dxy = Cyy = Dyy = 0;
    Cxxx = Cxxy = Cxyx = Cxyy = 0;
    Dxxx = Dxxy = Dxyx = Dxyy = 0;
    Cyxx = Cyxy = Cyyx = Cyyy = 0;
    Dyxx = Dyxy = Dyyx = Dyyy = 0;
    for (i = 0; i < d.NxHarm; i++) {
      kx = GPU_APPLE_AT(kx, i, j);
      ky = GPU_APPLE_AT(ky, i, j);
      Cm = cos(kx * xmm);
      Cp = cos(kx * xpp);
      Sm = sin(kx * xmm);
      Sp = sin(kx * xpp);
      Em = exp(-ky * X[2]);
      Ep = exp(ky * X[2]);
      ccp = Cm * Ep * d.C1 + Cp * Ep * d.C2 + Cp * Em * d.C3 + Cm * Em * d.C4;
      ccm = Cm * Ep * d.C1 + Cp * Ep * d.C2 - Cp * Em * d.C3 - Cm * Em * d.C4;
      csp = Cm * Ep * d.S1 + Cp * Ep * d.S2 + Cp * Em * d.S3 + Cm * Em * d.S4;
      csm = Cm * Ep * d.S1 + Cp * Ep * d.S2 - Cp * Em * d.S3 - Cm * Em * d.S4;
      scp = Sm * Ep * d.C1 + Sp * Ep * d.C2 + Sp * Em * d.C3 + Sm * Em * d.C4;
      scm = Sm * Ep * d.C1 + Sp * Ep * d.C2 - Sp * Em * d.C3 - Sm * Em * d.C4;
      ssp = Sm * Ep * d.S1 + Sp * Ep * d.S2 + Sp * Em * d.S3 + Sm * Em * d.S4;
      ssm = Sm * Ep * d.S1 + Sp * Ep * d.S2 - Sp * Em * d.S3 - Sm * Em * d.S4;
      Cx += GPU_APPLE_AT(CoZ, i, j) * ccp;
      Dx += GPU_APPLE_AT(CoZ, i, j) * csp;
      Cy += GPU_APPLE_AT(CxXoYZ, i, j) * scm;
      Dy += GPU_APPLE_AT(CxXoYZ, i, j) * ssm;
      Cxx += -GPU_APPLE_AT(CxXoZ, i, j) * scp;
      Dxx += -GPU_APPLE_AT(CxXoZ, i, j) * ssp;
      Cyx += GPU_APPLE_AT(CxX2oYZ, i, j) * ccm;
      Dyx += GPU_APPLE_AT(CxX2oYZ, i, j) * csm;
      Cxy += GPU_APPLE_AT(CxYoZ, i, j) * ccm;
      Dxy += GPU_APPLE_AT(CxYoZ, i, j) * csm;
      Cyy += GPU_APPLE_AT(CxXoZ, i, j) * scp;
      Dyy += GPU_APPLE_AT(CxXoZ, i, j) * ssp;
      if (End_Pole && d.order == 3) {
        Cxxx += -GPU_APPLE_AT(CxX2oZ, i, j) * ccp;
        Cxxy += -GPU_APPLE_AT(CxXYoZ, i, j) * scm;
        Dxxx += -GPU_APPLE_AT(CxX2oZ, i, j) * csp;
        Dxxy += -GPU_APPLE_AT(CxXYoZ, i, j) * ssm;
        Cyxx += -GPU_APPLE_AT(CxX3oYZ, i, j) * scm;
        Cyxy += GPU_APPLE_AT(CxX2oZ, i, j) * ccp;
        Dyxx += -GPU_APPLE_AT(CxX3oYZ, i, j) * ssm;
        Dyxy += GPU_APPLE_AT(CxX2oZ, i, j) * csp;
        Cxyx += -GPU_APPLE_AT(CxXYoZ, i, j) * scm;
        Cxyy += GPU_APPLE_AT(CxY2oZ, i, j) * ccp;
        Dxyx += -GPU_APPLE_AT(CxXYoZ, i, j) * ssm;
        Dxyy += GPU_APPLE_AT(CxY2oZ, i, j) * csp;
        Cyyx += GPU_APPLE_AT(CxX2oZ, i, j) * ccp;
        Cyyy += GPU_APPLE_AT(CxXYoZ, i, j) * scm;
        Dyyx += GPU_APPLE_AT(CxX2oZ, i, j) * csp;
        Dyyy += GPU_APPLE_AT(CxXYoZ, i, j) * ssm;
      }
    }
    f002 += Cx * Cx + Cy * Cy + Dx * Dx + Dy * Dy;
    f002x += Cx * Cxx + Cy * Cyx + Dx * Dxx + Dy * Dyx;
    f002y += Cx * Cxy + Cy * Cyy + Dx * Dxy + Dy * Dyy;
    if (End_Pole) {
      double kz = d.table[d.kz + j], lz = d.lz;
      f101 += Cx / kz;
      f101x += Cxx / kz;
      f101y += Cxy / kz;
      f011 += Cy / kz;
      f011x += Cyx / kz;
      f011y += Cyy / kz;
      if (d.order == 3) {
        Ax1 = Cx * Cxx + Cy * Cyx + Dx * Dxx + Dy * Dyx;
        Ay1 = Cx * Cxy + Cy * Cyy + Dx * Dxy + Dy * Dyy;
        Ax2 = Dx * Dxx + Dy * Dyx - Cx * Cxx - Cy * Cyx;
        Ay2 = Dx * Dxy + Dy * Dyy - Cx * Cxy - Cy * Cyy;
        Ax3 = Cxx * Dx + Cx * Dxx + Cyx * Dy + Cy * Dyx;
        Ay3 = Cxy * Dx + Cx * Dxy + Cyy * Dy + Cy * Dyy;
        Ax1x = Cxx * Cxx + Cyx * Cyx + Dxx * Dxx + Dyx * Dyx + Cx * Cxxx + Cy * Cyxx + Dx * Dxxx + Dy * Dyxx;
        Ax1y = Cxy * Cxx + Cyy * Cyx + Dxy * Dxx + Dyy * Dyx + Cx * Cxxy + Cy * Cyxy + Dx * Dxxy + Dy * Dyxy;
        Ay1x = Cxx * Cxy + Cyx * Cyy + Dxx * Dxy + Dyx * Dyy + Cx * Cxyx + Cy * Cyyx + Dx * Dxyx + Dy * Dyyx;
        Ay1y = Cxy * Cxy + Cyy * Cyy + Dxy * Dxy + Dyy * Dyy + Cx * Cxyy + Cy * Cyyy + Dx * Dxyy + Dy * Dyyy;
        Ax2x = Dxx * Dxx + Dyx * Dyx - Cxx * Cxx - Cyx * Cyx + Dx * Dxxx + Dy * Dyxx - Cx * Cxxx - Cy * Cyxx;
        Ax2y = Dxy * Dxx + Dyy * Dyx - Cxy * Cxx - Cyy * Cyx + Dx * Dxxy + Dy * Dyxy - Cx * Cxxy - Cy * Cyxy;
        Ay2x = Dxx * Dxy + Dyx * Dyy - Cxx * Cxy - Cyx * Cyy + Dx * Dxyx + Dy * Dyyx - Cx * Cxyx - Cy * Cyyx;
        Ay2y = Dxy * Dxy + Dyy * Dyy - Cxy * Cxy - Cyy * Cyy + Dx * Dxyy + Dy * Dyyy - Cx * Cxyy - Cy * Cyyy;
        Ax3x = Cxxx * Dx + Cxx * Dxx + Cyxx * Dy + Cyx * Dyx + Cxx * Dxx + Cx * Dxxx + Cyx * Dyx + Cy * Dyxx;
        Ax3y = Cxxy * Dx + Cxy * Dxx + Cyxy * Dy + Cyy * Dyx + Cxx * Dxy + Cx * Dxxy + Cyx * Dyy + Cy * Dyxy;
        Ay3x = Cxyx * Dx + Cxx * Dxy + Cyyx * Dy + Cyx * Dyy + Cxy * Dxx + Cx * Dxyx + Cyy * Dyx + Cy * Dyyx;
        Ay3y = Cxyy * Dx + Cxy * Dxy + Cyyy * Dy + Cyy * Dyy + Cxy * Dxy + Cx * Dxyy + Cyy * Dyy + Cy * Dyyy;
        f003x += 0.5 / (kz * kz) * (Cxx * (4. * Ax3 / 3. + lz * kz * Ax1) - 2. * Dxx * (Ax1 - Ax2 / 3.));
        f003x += 0.5 / (kz * kz) * (Cx * (4. * Ax3x / 3. + lz * kz * Ax1x) - 2. * Dx * (Ax1x - Ax2x / 3.));
        f003x += 0.5 / (kz * kz) * (Cyx * (4. * Ay3 / 3. + lz * kz * Ay1) - 2. * Dyx * (Ay1 - Ay2 / 3.));
        f003x += 0.5 / (kz * kz) * (Cy * (4. * Ay3x / 3. + lz * kz * Ay1x) - 2. * Dy * (Ay1x - Ay2x / 3.));
        f003y += 0.5 / (kz * kz) * (Cxy * (4. * Ax3 / 3. + lz * kz * Ax1) - 2. * Dxy * (Ax1 - Ax2 / 3.));
        f003y += 0.5 / (kz * kz) * (Cx * (4. * Ax3y / 3. + lz * kz * Ax1y) - 2. * Dx * (Ax1y - Ax2y / 3.));
        f003y += 0.5 / (kz * kz) * (Cyy * (4. * Ay3 / 3. + lz * kz * Ay1) - 2. * Dyy * (Ay1 - Ay2 / 3.));
        f003y += 0.5 / (kz * kz) * (Cy * (4. * Ay3y / 3. + lz * kz * Ay1y) - 2. * Dy * (Ay1y - Ay2y / 3.));
      }
    }
  }
  f002 = -step * d.lz / 4. * (factor * factor) * f002;
  f002x = -step * d.lz / 2. * (factor * factor) * f002x;
  f002y = -step * d.lz / 2. * (factor * factor) * f002y;
  if (End_Pole) {
    f101 = 2 * factor * f101;
    f101x = 2 * factor * f101x;
    f101y = 2 * factor * f101y;
    f011 = 2 * factor * f011;
    f011x = 2 * factor * f011x;
    f011y = 2 * factor * f011y;
    pn = (1. - f101x) * (1. - f011y) - f101y * f011x;
    f003x *= pow(factor, 3.0);
    f003y *= pow(factor, 3.0);
  }
  Ax = Dx * factor;
  Ay = Dy * factor;
  X[1] = X[1] + Ax;
  X[3] = X[3] + Ay;
  if (End_Pole) {
    X[1] = ((1 - f011y) * (X[1] + f002x + f003x) + f011x * (X[3] + f002y + f003y)) / pn;
    X[3] = ((1 - f101x) * (X[3] + f002y + f003y) + f101y * (X[1] + f002x + f003x)) / pn;
    X[0] = X[0] + X[1] * step * d.lz - f101;
    X[2] = X[2] + X[3] * step * d.lz - f011;
  } else {
    X[1] = X[1] + f002x;
    X[3] = X[3] + f002y;
    X[0] = X[0] + X[1] * step * d.lz;
    X[2] = X[2] + X[3] * step * d.lz;
  }
  X[4] += step * d.lz;
  xpp = X[0] + d.x0;
  xmm = X[0] - d.x0;
  for (j = 0; j < d.NzHarm; j++) {
    Dx = Dy = 0;
    for (i = 0; i < d.NxHarm; i++) {
      kx = GPU_APPLE_AT(kx, i, j);
      ky = GPU_APPLE_AT(ky, i, j);
      Cm = cos(kx * xmm);
      Cp = cos(kx * xpp);
      Sm = sin(kx * xmm);
      Sp = sin(kx * xpp);
      Em = exp(-ky * X[2]);
      Ep = exp(ky * X[2]);
      csp = Cm * Ep * d.S1 + Cp * Ep * d.S2 + Cp * Em * d.S3 + Cm * Em * d.S4;
      ssm = Sm * Ep * d.S1 + Sp * Ep * d.S2 - Sp * Em * d.S3 - Sm * Em * d.S4;
      Dx += GPU_APPLE_AT(CoZ, i, j) * csp;
      Dy += GPU_APPLE_AT(CxXoYZ, i, j) * ssm;
    }
  }
  Ax = Dx * factor;
  Ay = Dy * factor;
  X[1] = X[1] - Ax;
  X[3] = X[3] - Ay;
}

__device__ void gpuAppleShim(double *coord, double brho, const GPU_APPLE_DATA &d) {
  double intBx = 0., intBy = 0., tmpx, tmpy, tmpg;
  long iHarm;

  for (iHarm = 0; iHarm < d.ShimHarm; iHarm++) {
    tmpx = (double)iHarm * d.kx_shim * coord[0];
    tmpy = (double)iHarm * d.kx_shim * coord[2];
    tmpg = -(double)iHarm * d.kx_shim * d.dgap / 2.;
    intBx += (d.table[d.Ci_shim + iHarm] * sin(tmpx) * cosh(tmpy) + d.table[d.Si_shim + iHarm] * cos(tmpx) * sinh(tmpy)) * exp(tmpg);
    intBy += (-d.table[d.Ci_shim + iHarm] * cos(tmpx) * sinh(tmpy) + d.table[d.Si_shim + iHarm] * sin(tmpx) * cosh(tmpy)) * exp(tmpg);
  }
  coord[1] += intBy / brho * d.shimScale;
  coord[3] -= intBx / brho * d.shimScale;
}

__global__ void gpuAppleKernel(double *particles, long nParticles, int stride, GPU_APPLE_DATA d,
                               unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, X[6], denom, p, brho, ds;
  long istep, nleft;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.doTilt)
    gpuPortRotate(coord, d.cosTilt, d.sinTilt);
  /* offset_beam() with MALIGN DX=-dx, DY=-dy, DZ=dz */
  if (d.doOffset) {
    double ox = -d.dx, oy = -d.dy, oz = d.dz;
    ds = oz ? oz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]) : 0;
    coord[0] += ox + oz * coord[1];
    coord[1] += 0.0;
    coord[2] += oy + oz * coord[3];
    coord[3] += 0.0;
    coord[4] += ds;
  }
  p = (1. + coord[5]) * d.pCentral;
  brho = p / d.eomc;
  if (d.shimOn)
    gpuAppleShim(coord, brho, d);
  X[0] = coord[0];
  X[2] = coord[2];
  X[1] = coord[1] / (denom = sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]));
  X[3] = coord[3] / denom;
  X[4] = coord[4];
  X[5] = coord[5];
  if (d.endPole) {
    nleft = d.periods - 2;
    gpuAppleFijk(X, 1, 0.5, 0.25 / brho, d);
    gpuAppleFijk(X, 1, 0.5, -0.75 / brho, d);
  } else
    nleft = d.periods;
  for (istep = 0; istep < d.periods; istep++) {
    nleft -= d.step;
    if (nleft < 0)
      break;
    gpuAppleFijk(X, 0, (double)d.step, 1. / brho, d);
  }
  if (nleft)
    gpuAppleFijk(X, 0, (double)(nleft + d.step), 1. / brho, d);
  if (d.endPole) {
    gpuAppleFijk(X, 1, 0.5, 0.75 / brho, d);
    gpuAppleFijk(X, 1, 0.5, -0.25 / brho, d);
  }
  coord[0] = X[0];
  coord[2] = X[2];
  coord[1] = X[1] / (denom = sqrt(1 - X[1] * X[1] - X[3] * X[3]));
  coord[3] = X[3] / denom;
  coord[4] = X[4];
  coord[5] = X[5];
  if (d.shimOn)
    gpuAppleShim(coord, brho, d);
  /* offset_beam() with MALIGN DX=dx, DY=dy, DZ=-dz */
  if (d.doOffset) {
    double ox = d.dx, oy = d.dy, oz = -d.dz;
    ds = oz ? oz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]) : 0;
    coord[0] += ox + oz * coord[1];
    coord[1] += 0.0;
    coord[2] += oy + oz * coord[3];
    coord[3] += 0.0;
    coord[4] += ds;
  }
  if (d.doTilt)
    gpuPortRotate(coord, d.cosUntilt, d.sinUntilt);
  gpuPortCountInvalid(coord, counters);
}

static double *gpuAppleTable = NULL;
static long gpuAppleTableCapacity = 0;
static const void *gpuAppleTableKey = NULL;

extern "C" int gpuCudaApple(void *coord, long nParticles, int stride, const GPU_APPLE_DATA *data,
                            const double *hostTable, long tableCount, const void *tableKey, float *milliseconds) {
  GPU_APPLE_DATA d;
  cudaError_t cudaStatus;
  int status;

  if (!coord || !data || !hostTable || nParticles <= 0 || stride < 6 || tableCount < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuAppleTableCapacity < tableCount) {
    gpuFreeDevice(gpuAppleTable);
    gpuAppleTable = NULL;
    gpuAppleTableCapacity = 0;
    gpuAppleTableKey = NULL;
    cudaStatus = cudaMalloc(&gpuAppleTable, tableCount * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuAppleTableCapacity = tableCount;
  }
  if (gpuAppleTableKey != tableKey) {
    cudaStatus = cudaMemcpy(gpuAppleTable, hostTable, tableCount * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuAppleTableKey = tableKey;
  }
  d = *data;
  d.table = gpuAppleTable;
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuAppleKernel<<<gpuPortBlocks(nParticles, 64), 64>>>(static_cast<double *>(coord), nParticles, stride, d,
                                                         gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* Columns for the host's sums, packed column by column in particle order */

__device__ __forceinline__ double gpuPortColumnValue(const double *part, int code, double Po, double cMks) {
  double p, beta;
  int j;

  switch (code) {
  case GPU_PORT_COLUMN_TIME:
    p = Po * (1 + part[5]);
    beta = p / sqrt(p * p + 1);
    return part[4] / (beta * cMks);
  case GPU_PORT_COLUMN_CLEAN_S:
  case GPU_PORT_COLUMN_FINITE:
    for (j = 0; j < 6; j++)
      if (isnan(part[j]) || isinf(part[j]))
        break;
    if (code == GPU_PORT_COLUMN_FINITE)
      return j == 6 ? 1 : 0;
    if (j != 6)
      return part[4];
    p = Po * (1 + part[5]);
    beta = p / sqrt(p * p + 1);
    return part[4] / (beta * cMks);
  default:
    return part[code];
  }
}

__global__ void gpuPortPackKernel(const double *coord, long nParticles, int stride, int nColumns, const int *codes,
                                  double Po, double cMks, double *packed) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  for (int k = 0; k < nColumns; k++)
    packed[k * nParticles + ip] = gpuPortColumnValue(coord + ip * stride, codes[k], Po, cMks);
}

#define GPU_PORT_MAX_COLUMNS 8
static int *gpuPortCodes = NULL;

/* The columns codes[] of the first nParticles particles, column after column, to host */
extern "C" int gpuCudaPortColumns(const void *coord, long nParticles, int stride, int nColumns, const int *codes,
                                  double Po, double cMks, double *host) {
  cudaError_t status;
  int code;

  if (!coord || !codes || !host || nParticles <= 0 || nColumns < 1 || nColumns > GPU_PORT_MAX_COLUMNS)
    return static_cast<int>(cudaErrorInvalidValue);
  code = gpuPortEnsureValues(nParticles * nColumns);
  if (code != static_cast<int>(cudaSuccess))
    return code;
  if (!gpuPortCodes) {
    status = cudaMalloc(&gpuPortCodes, GPU_PORT_MAX_COLUMNS * sizeof(int));
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  status = cudaMemcpy(gpuPortCodes, codes, nColumns * sizeof(int), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuPortPackKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<const double *>(coord), nParticles, stride,
                                                              nColumns, gpuPortCodes, Po, cMks, gpuPortValues);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(host, gpuPortValues, nParticles * nColumns * sizeof(double), cudaMemcpyDeviceToHost);
  return static_cast<int>(status);
}

__global__ void gpuPortSetColumnKernel(double *coord, long nParticles, int stride, int column, const double *values) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip < nParticles)
    coord[ip * stride + column] = values[ip];
}

/* One coordinate of every particle, from host values */
extern "C" int gpuCudaPortSetColumn(void *coord, long nParticles, int stride, int column, const double *host) {
  cudaError_t status;
  int code;

  if (!coord || !host || nParticles <= 0 || column < 0 || column >= stride)
    return static_cast<int>(cudaErrorInvalidValue);
  code = gpuPortEnsureValues(nParticles);
  if (code != static_cast<int>(cudaSuccess))
    return code;
  status = cudaMemcpy(gpuPortValues, host, nParticles * sizeof(double), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuPortSetColumnKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride,
                                                                   column, gpuPortValues);
  return static_cast<int>(cudaGetLastError());
}

/* Rows first..nParticles-1 marked lost at z with their absolute momentum (SAMPLE, CLEAN with no finite particle) */
__global__ void gpuPortMarkTailKernel(double *coord, long first, long nParticles, int stride, double z, double Po) {
  long ip = first + (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (ip >= nParticles)
    return;
  coord[ip * stride + 4] = z;
  coord[ip * stride + 5] = Po * (1 + coord[ip * stride + 5]);
}

extern "C" int gpuCudaPortMarkTail(void *coord, long first, long nParticles, int stride, double z, double Po) {
  if (!coord || first < 0 || nParticles <= first)
    return static_cast<int>(cudaSuccess);
  gpuPortMarkTailKernel<<<gpuPortBlocks(nParticles - first, 256), 256>>>(static_cast<double *>(coord), first,
                                                                          nParticles, stride, z, Po);
  return static_cast<int>(cudaGetLastError());
}

/* ---------------------------------------------------------------------------------------------------- */
/* PFILTER's passes */

__global__ void gpuPfilterKernel(double *particles, long nParticles, int stride, GPU_PFILTER_DATA d,
                                 unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, p;
  int lost;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  switch (d.mode) {
  case GPU_PFILTER_PERCENTILE:
    lost = (d.upper && coord[5] > d.limit) || (!d.upper && coord[5] < d.limit);
    break;
  case GPU_PFILTER_FIXED:
    p = (1 + coord[5]) * d.Po;
    lost = (d.hasUpper && p > d.pUpper) || (d.hasLower && p < d.pLower);
    break;
  default:
    lost = !(fabs(coord[5] - d.reference) < d.deltaLimit);
    break;
  }
  flags[ip] = lost ? 1 : 0;
  if (lost) {
    coord[4] = d.z;
    if (d.mode == GPU_PFILTER_FIXED)
      coord[5] = (1 + coord[5]) * d.Po;
    else
      coord[5] = d.Po * (1 + coord[5]);
    atomicAdd(&counters[0], 1ULL);
  }
}

extern "C" int gpuCudaPfilter(void *coord, long nParticles, int stride, const GPU_PFILTER_DATA *data,
                              float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 1);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuPfilterKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data,
                                                             gpuPortFlags, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* CLEAN: remove_outlier_particles()'s test */

__global__ void gpuCleanKernel(double *particles, long nParticles, int stride, GPU_CLEAN_DATA d,
                               unsigned char *flags, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, ini[6];
  int j, is_out;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  for (j = 0; j < 6; j++)
    ini[j] = coord[j];
  ini[4] = gpuPortColumnValue(coord, GPU_PORT_COLUMN_CLEAN_S, d.Po, d.cMks);
  for (j = is_out = 0; j < 6; j++) {
    if (d.limit[j] <= 0)
      continue;
    switch (d.mode) {
    case 0:
      if (fabs(ini[j] - d.centroid[j]) / d.stDev[j] > d.limit[j])
        is_out = 1;
      break;
    case 1:
      if (fabs(ini[j] - d.centroid[j]) > d.limit[j])
        is_out = 2;
      break;
    default:
      if (fabs(ini[j]) > d.limit[j])
        is_out = 3;
      break;
    }
    if (isnan(ini[j]) || isinf(ini[j]))
      is_out = 1;
    if (is_out)
      break;
  }
  flags[ip] = is_out ? 1 : 0;
  if (is_out) {
    coord[4] = d.z;
    coord[5] = d.Po * (1 + coord[5]);
    atomicAdd(&counters[0], 1ULL);
  }
}

extern "C" int gpuCudaClean(void *coord, long nParticles, int stride, const GPU_CLEAN_DATA *data, float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 1);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuCleanKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data,
                                                           gpuPortFlags, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* EMITTANCE and REMCOR transformations */

__global__ void gpuEmittanceKernel(double *particles, long nParticles, int stride, GPU_EMITTANCE_DATA d) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord;
  int i;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  for (i = 0; i < 2; i++) {
    if (!d.doPlane[i])
      continue;
    coord[2 * i + 0] = d.factor[i] * (coord[2 * i + 0] - d.eta[i] * coord[5]) + d.eta[i] * coord[5];
    coord[2 * i + 1] = d.factor[i] * (coord[2 * i + 1] - d.eta[i] * coord[5]) + d.eta[i] * coord[5];
  }
}

extern "C" int gpuCudaEmittance(void *coord, long nParticles, int stride, const GPU_EMITTANCE_DATA *data,
                                float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuEmittanceKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data);
  return gpuPortEnd(milliseconds);
}

__global__ void gpuRemcorKernel(double *particles, long nParticles, int stride, GPU_REMCOR_DATA d) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *part;
  int ic;

  if (ip >= nParticles)
    return;
  part = particles + ip * stride;
  for (ic = 0; ic < 4; ic++)
    if (d.active[ic])
      part[ic] -= part[d.wc] * d.ratio[ic];
}

extern "C" int gpuCudaRemcor(void *coord, long nParticles, int stride, const GPU_REMCOR_DATA *data, float *milliseconds) {
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuRemcorKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* MHISTOGRAM: findMinMax()'s minima and maxima (exact in any order), and mhist_table()'s fill loop with
 * chfill1m() and chfilln()'s bin indices, counted with integer atomics */

struct GpuMhistMin {
  __device__ double operator()(double a, double b) const { return a > b ? b : a; }
};
struct GpuMhistMax {
  __device__ double operator()(double a, double b) const { return a < b ? b : a; }
};

/* min and max of the six columns (time for the fifth), as findMinMax() makes them */
extern "C" int gpuCudaMhistMinMax(const void *coord, long nParticles, int stride, double Po, double cMks,
                                  double *min, double *max, double *time) {
  int codes[6] = {0, 1, 2, 3, GPU_PORT_COLUMN_TIME, 5};
  cudaError_t status;
  int code, j;

  if (!coord || !min || !max || !time || nParticles <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  code = gpuPortEnsureValues(nParticles * 6);
  if (code != static_cast<int>(cudaSuccess))
    return code;
  if (!gpuPortCodes) {
    status = cudaMalloc(&gpuPortCodes, GPU_PORT_MAX_COLUMNS * sizeof(int));
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  status = cudaMemcpy(gpuPortCodes, codes, 6 * sizeof(int), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuPortPackKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<const double *>(coord), nParticles, stride,
                                                              6, gpuPortCodes, Po, cMks, gpuPortValues);
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return static_cast<int>(status);
  for (j = 0; j < 6; j++) {
    thrust::device_ptr<double> column(gpuPortValues + j * nParticles);
    min[j] = thrust::reduce(column, column + nParticles, DBL_MAX, GpuMhistMin());
    max[j] = thrust::reduce(column, column + nParticles, -DBL_MAX, GpuMhistMax());
  }
  /* the times, for the host's mean in particle order */
  status = cudaMemcpy(time, gpuPortValues + 4 * nParticles, nParticles * sizeof(double), cudaMemcpyDeviceToHost);
  return static_cast<int>(status);
}

typedef GPU_MHIST_BOOK GpuMhistBook;

__global__ void gpuMhistFillKernel(const double *particles, long nParticles, int stride, double Po, double cMks,
                                   double t0, int nBooks, const GpuMhistBook *books) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *coord;
  double part[6], P, beta;
  long index, temp;
  int b, i, j;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  for (j = 0; j < 6; j++) {
    if (j == 4) {
      P = Po * (1 + coord[5]);
      beta = P / sqrt(P * P + 1);
      part[4] = coord[4] / (beta * cMks) - t0;
    } else
      part[j] = coord[j];
  }
  for (b = 0; b < nBooks; b++) {
    const GpuMhistBook &bk = books[b];
    if (bk.kind == 0) {
      for (j = 0; j < bk.nD; j++) {
        index = (long)((part[bk.column[j]] - bk.xmin[j]) / bk.dx[j]);
        if (index < 0)
          index = 0;
        if (index > bk.bins - 1)
          index = bk.bins - 1;
        atomicAdd(&bk.count[index * bk.nD + j], 1ULL);
      }
    } else {
      index = 0;
      for (i = 0; i < bk.nD; i++) {
        temp = (long)((part[i + bk.offset] - bk.xmin[i]) / bk.dx[i]);
        if (temp < 0)
          temp = 0;
        if (temp > bk.xbins[i] - 1)
          temp = bk.xbins[i] - 1;
        if (i > 0)
          index = index * bk.xbins[i - 1] + temp;
        if (i == 0)
          index = temp;
      }
      atomicAdd(&bk.count[index], 1ULL);
    }
  }
}

static GpuMhistBook *gpuMhistBooks = NULL;
static unsigned long long *gpuMhistCounts = NULL;
static long gpuMhistCountsCapacity = 0;

/* books[]'s count pointers are offsets into one array of total counts, returned in counts */
extern "C" int gpuCudaMhistFill(const void *coord, long nParticles, int stride, double Po, double cMks, double t0,
                                int nBooks, const void *booksHost, long total, unsigned long long *counts) {
  GpuMhistBook books[GPU_MHIST_BOOKS];
  cudaError_t status;
  long offset;
  int b;

  if (!coord || !booksHost || !counts || nParticles <= 0 || nBooks < 1 || nBooks > GPU_MHIST_BOOKS || total < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuMhistCountsCapacity < total) {
    gpuFreeDevice(gpuMhistCounts);
    gpuMhistCounts = NULL;
    gpuMhistCountsCapacity = 0;
    status = cudaMalloc(&gpuMhistCounts, total * sizeof(*gpuMhistCounts));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuMhistCountsCapacity = total;
  }
  if (!gpuMhistBooks) {
    status = cudaMalloc(&gpuMhistBooks, GPU_MHIST_BOOKS * sizeof(*gpuMhistBooks));
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  memcpy(books, booksHost, nBooks * sizeof(*books));
  for (b = 0; b < nBooks; b++) {
    offset = books[b].countOffset;
    books[b].count = gpuMhistCounts + offset;
  }
  status = cudaMemset(gpuMhistCounts, 0, total * sizeof(*gpuMhistCounts));
  if (status == cudaSuccess)
    status = cudaMemcpy(gpuMhistBooks, books, nBooks * sizeof(*books), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return static_cast<int>(status);
  gpuMhistFillKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<const double *>(coord), nParticles, stride, Po,
                                                               cMks, t0, nBooks, gpuMhistBooks);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(counts, gpuMhistCounts, total * sizeof(*counts), cudaMemcpyDeviceToHost);
  return static_cast<int>(status);
}

/* ---------------------------------------------------------------------------------------------------- */
/* Per-particle values the host made (the CPU's random numbers), to the device */

static double *gpuPortUpload = NULL;
static long gpuPortUploadCapacity = 0;

static int gpuPortUploadValues(const double *host, long count) {
  cudaError_t status;

  if (gpuPortUploadCapacity < count) {
    gpuFreeDevice(gpuPortUpload);
    gpuPortUpload = NULL;
    gpuPortUploadCapacity = 0;
    status = cudaMalloc(&gpuPortUpload, count * sizeof(double));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuPortUploadCapacity = count;
  }
  return static_cast<int>(cudaMemcpy(gpuPortUpload, host, count * sizeof(double), cudaMemcpyHostToDevice));
}

/* SCATTER, DSCATTER, PEPPOT: part[i] += increment for the flagged particles; delta with the time kept */
__global__ void gpuIncrementKernel(double *particles, long nParticles, int stride, GPU_INCREMENT_DATA d,
                                   const double *values, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const double *v;
  double *part, P, beta, t;
  int i;

  if (ip >= nParticles)
    return;
  v = values + ip * GPU_INCREMENT_VALUES;
  if (!v[0])
    return;
  part = particles + ip * stride;
  for (i = 0; i < 4; i++)
    if (d.active[i])
      part[i] += v[1 + i];
  if (d.dpActive) {
    P = (1 + part[5]) * d.Po;
    beta = P / sqrt(P * P + 1);
    t = part[4] / beta;
    part[5] += v[5];
    P = (1 + part[5]) * d.Po;
    beta = P / sqrt(P * P + 1);
    part[4] = t * beta;
  }
  gpuPortCountInvalid(part, counters);
}

extern "C" int gpuCudaIncrement(void *coord, long nParticles, int stride, const GPU_INCREMENT_DATA *data,
                                const double *hostValues, float *milliseconds) {
  int status;

  if (!coord || !data || !hostValues || nParticles <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortUploadValues(hostValues, nParticles * GPU_INCREMENT_VALUES);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuIncrementKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data,
                                                               gpuPortUpload, gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* random_2's numbers on the device.  dlaran multiplies its 48-bit state by GPU_PORT_DLARAN_MULTIPLIER and
 * returns the new state over 2^48, so every uniform is the CPU's exactly; gauss_rn()'s Gaussian numbers
 * differ from glibc's only by the rounding of CUDA's log and sin. */
#define GPU_PORT_DLARAN_MULTIPLIER 33952834046453ULL
#define GPU_PORT_DLARAN_MASK ((1ULL << 48) - 1)
#define GPU_PORT_DLARAN_SCALE 3.552713678800501e-15
#define GPU_PORT_PIX2 6.283185307179586476925287

/* The uniform whose state is *state, which then moves to the next */
__device__ __forceinline__ double gpuPortUniform(unsigned long long *state) {
  double u = *state * GPU_PORT_DLARAN_SCALE;

  *state = (*state * GPU_PORT_DLARAN_MULTIPLIER) & GPU_PORT_DLARAN_MASK;
  return u;
}

/* gauss_rn(0, random_2) */
__device__ __forceinline__ double gpuPortGauss(unsigned long long *state) {
  double urn1 = gpuPortUniform(state), urn2 = gpuPortUniform(state);
  double factor = sqrt(-2 * log(urn1));

  return factor * sin(GPU_PORT_PIX2 * urn2);
}

/* state * power^n modulo 2^48 */
__device__ __forceinline__ unsigned long long gpuPortJump(unsigned long long state, unsigned long long power,
                                                          unsigned long long n) {
  for (; n; n >>= 1) {
    if (n & 1)
      state = (state * power) & GPU_PORT_DLARAN_MASK;
    power = (power * power) & GPU_PORT_DLARAN_MASK;
  }
  return state;
}

/* A particle's first state: from the host's list, or first * stride^ip for the first count particles */
__device__ __forceinline__ unsigned long long gpuPortDrawState(const double *states, long ip, long count,
                                                               unsigned long long first, unsigned long long stride) {
  if (states)
    return (unsigned long long)states[ip];
  return ip < count ? gpuPortJump(first, stride, (unsigned long long)ip) : 0;
}

/* interp(f, x, n, xo, 0, 1, &code): the bracketing search and two-point Lagrange formula of mdbmth */
__device__ double gpuPortInterp(const double *f, const double *x, long n, double xo, int *returnCode) {
  long hi, lo, mid, offset, i, j;
  double denom, numer, sum;

  lo = 0;
  hi = n - 1;
  if (lo == hi) {
    *returnCode = 0;
    return f[0];
  }
  if (x[lo] < x[hi]) {
    if (xo < x[lo = 0]) {
      *returnCode = 0;
      return f[lo];
    }
    if (xo > x[hi = n - 1]) {
      *returnCode = 0;
      return f[hi];
    }
    while ((hi - lo) > 1) {
      mid = (lo + hi) / 2;
      if (xo < x[mid])
        hi = mid;
      else
        lo = mid;
    }
  } else {
    if (xo > x[lo = 0]) {
      *returnCode = 0;
      return f[lo];
    }
    if (xo < x[hi = n - 1]) {
      *returnCode = 0;
      return f[hi];
    }
    while ((hi - lo) > 1) {
      mid = (lo + hi) / 2;
      if (xo > x[mid])
        hi = mid;
      else
        lo = mid;
    }
  }
  offset = lo;
  offset = offset > 0 ? offset : 0;
  offset = offset < n - 2 ? offset : n - 2;
  offset = offset > 0 ? offset : 0;
  x += offset;
  f += offset;
  sum = 0;
  for (i = 0; i < 2; i++) {
    denom = 1;
    numer = 1;
    for (j = 0; j < 2; j++) {
      if (i != j) {
        denom *= (x[i] - x[j]);
        numer *= (xo - x[j]);
        if (numer == 0) {
          *returnCode = 1;
          return f[j];
        }
      }
    }
    if (denom == 0) {
      *returnCode = 0;
      return 0.0;
    }
    sum += f[i] * numer / denom;
  }
  *returnCode = 1;
  return sum;
}

/* SCATTER (scatter_ele()) and DSCATTER (distributionScatter()) for a particle the host found scatters,
 * with its numbers made here; then gpuIncrementKernel()'s update */
__global__ void gpuDrawIncrementKernel(double *particles, long nParticles, int stride, GPU_DRAW_DATA d,
                                       const double *states, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long state;
  double v[6], *part, P, beta, t;
  int i;

  if (ip >= nParticles)
    return;
  if (!(state = gpuPortDrawState(states, ip, d.count, d.first, d.stride)))
    return;
  for (i = 1; i < 6; i++)
    v[i] = 0;
  if (d.dscatter) {
    int code;
    double cdf = gpuPortUniform(&state);
    double amplitude = d.factor * gpuPortInterp(d.indep, d.cdf, d.nData, cdf, &code);
    if (d.randomSign)
      amplitude *= gpuPortUniform(&state) > 0.5 ? 1 : -1;
    if (!code)
      atomicAdd(&counters[1], 1ULL);
    v[d.column] = amplitude;
  } else {
    for (i = 0; i < 4; i++) {
      if (!d.sigma[i])
        continue;
      v[1 + i] = d.gaussian ? gpuPortGauss(&state) * d.sigma[i] : 2 * d.sigma[i] * (gpuPortUniform(&state) - 0.5);
    }
    if (d.dp)
      v[5] = d.gaussian ? d.dp * gpuPortGauss(&state) : 2 * d.dp * (gpuPortUniform(&state) - 0.5);
  }
  part = particles + ip * stride;
  for (i = 0; i < 4; i++)
    if (d.inc.active[i])
      part[i] += v[1 + i];
  if (d.inc.dpActive) {
    P = (1 + part[5]) * d.inc.Po;
    beta = P / sqrt(P * P + 1);
    t = part[4] / beta;
    part[5] += v[5];
    P = (1 + part[5]) * d.inc.Po;
    beta = P / sqrt(P * P + 1);
    part[4] = t * beta;
  }
  gpuPortCountInvalid(part, counters);
}

/* DSCATTER's table, kept while the host's arrays stay the same */
static double *gpuDscatterTable = NULL;
static long gpuDscatterTableCapacity = 0;
static const double *gpuDscatterHostIndep = NULL, *gpuDscatterHostCdf = NULL;
static long gpuDscatterTableN = 0;

/* hostStates: one a particle, or NULL for data->count particles in order (see GPU_DRAW_DATA) */
extern "C" int gpuCudaDrawIncrement(void *coord, long nParticles, int stride, const GPU_DRAW_DATA *data,
                                    const double *hostStates, unsigned long long *counters, float *milliseconds) {
  GPU_DRAW_DATA d;
  cudaError_t cudaStatus;
  int status;

  if (!coord || !data || nParticles <= 0 || stride < 6 || (data->dscatter && (!data->indep || !data->cdf ||
                                                                             data->nData < 1)))
    return static_cast<int>(cudaErrorInvalidValue);
  d = *data;
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess) && hostStates)
    status = gpuPortUploadValues(hostStates, nParticles);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  if (data->dscatter) {
    if (gpuDscatterTableCapacity < 2 * data->nData) {
      gpuFreeDevice(gpuDscatterTable);
      gpuDscatterTable = NULL;
      gpuDscatterTableCapacity = 0;
      gpuDscatterHostIndep = gpuDscatterHostCdf = NULL;
      cudaStatus = cudaMalloc(&gpuDscatterTable, 2 * data->nData * sizeof(double));
      if (cudaStatus != cudaSuccess)
        return static_cast<int>(cudaStatus);
      gpuDscatterTableCapacity = 2 * data->nData;
    }
    if (gpuDscatterHostIndep != data->indep || gpuDscatterHostCdf != data->cdf || gpuDscatterTableN != data->nData) {
      cudaStatus = cudaMemcpy(gpuDscatterTable, data->indep, data->nData * sizeof(double), cudaMemcpyHostToDevice);
      if (cudaStatus == cudaSuccess)
        cudaStatus = cudaMemcpy(gpuDscatterTable + data->nData, data->cdf, data->nData * sizeof(double),
                                cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess)
        return static_cast<int>(cudaStatus);
      gpuDscatterHostIndep = data->indep;
      gpuDscatterHostCdf = data->cdf;
      gpuDscatterTableN = data->nData;
    }
    d.indep = gpuDscatterTable;
    d.cdf = gpuDscatterTable + data->nData;
  }
  status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuDrawIncrementKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, d,
                                                                   hostStates ? gpuPortUpload : NULL, gpuPortCounters);
  status = gpuPortEnd(milliseconds);
  if (status == static_cast<int>(cudaSuccess) && counters)
    status = static_cast<int>(cudaMemcpy(counters, gpuPortCounters, GPU_PORT_COUNTERS * sizeof(*counters),
                                         cudaMemcpyDeviceToHost));
  return status;
}

/* ---------------------------------------------------------------------------------------------------- */
/* MATTER: track_through_matter() for one particle, with the host's random numbers and energy loss (which
 * decided the losses and so the order) */

static double *gpuMatterEvents = NULL;
static long gpuMatterEventsCapacity = 0;

__global__ void gpuMatterKernel(double *particles, long nParticles, int stride, GPU_MATTER_DATA d,
                                const double *values, const double *events, unsigned long long *counters) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double v[GPU_MATTER_VALUES] = {0, 0, 0, 0, 0, 0};
  double *coord, P, gamma = 0, beta, t = 0, dx, dy, ds, z1, z2, z[4];
  const double sqrt3 = 1.7320508075688772;
  unsigned long long state = 0;
  int i;

  if (ip >= nParticles)
    return;
  /* without energy decay the device's numbers need no values: particle ip's start at first * stride^ip */
  if (values)
    for (i = 0; i < GPU_MATTER_VALUES; i++)
      v[i] = values[ip * GPU_MATTER_VALUES + i];
  coord = particles + ip * stride;
  if (d.deviceDraws && d.multipleScattering) {
    state = values ? (unsigned long long)v[0] : gpuPortJump(d.first, d.stride, (unsigned long long)ip);
    /* mode 1: four numbers, mode 2: two */
    for (i = 0; i < (d.multipleScattering == 1 ? 4 : 2); i++)
      z[i] = gpuPortGauss(&state);
  } else
    for (i = 0; i < 4; i++)
      z[i] = v[i];
  if (d.energyDecay) {
    P = (1 + coord[5]) * d.Po;
    gamma = sqrt(P * P + 1);
    beta = P / gamma;
    t = coord[4] / beta;
  }
  if (d.multipleScattering == 1) {
    z1 = z[0];
    z2 = z[1];
    coord[0] += (dx = (z1 / sqrt3 + z2) * d.L * d.thetaRms / 2 + d.L * coord[1]);
    coord[1] += z2 * d.thetaRms;
    z1 = z[2];
    z2 = z[3];
    coord[2] += (dy = (z1 / sqrt3 + z2) * d.L * d.thetaRms / 2 + d.L * coord[3]);
    coord[3] += z2 * d.thetaRms;
    ds = sqrt(d.L * d.L + dx * dx + dy * dy);
  } else if (d.multipleScattering == 2) {
    coord[0] += d.L / 2 * coord[1];
    coord[2] += d.L / 2 * coord[3];
    z1 = z[0];
    z2 = z[1];
    coord[1] += d.thetaRms * z1;
    coord[3] += d.thetaRms * z2;
    coord[0] += d.L / 2 * coord[1];
    coord[2] += d.L / 2 * coord[3];
    ds = d.L;
  } else {
    /* single scattering: each section a drift, or the host's scattering event in it */
    const double *e = events + (long)v[0] * GPU_MATTER_EVENT_VALUES;
    long nEvents = (long)v[1], next = 0, is;
    double zs, dxp, dyp;

    ds = 0;
    for (is = 0; is < d.sections; is++) {
      if (next < nEvents && (long)e[next * GPU_MATTER_EVENT_VALUES] == is) {
        if (d.deviceDraws) {
          /* the host's loop: zs, F, then phi, from the event's numbers */
          const double sqrPi = 3.141592653589793 * 3.141592653589793;
          double F, theta, phi;
          state = (unsigned long long)e[next * GPU_MATTER_EVENT_VALUES + 1];
          zs = d.L1 * gpuPortUniform(&state);
          F = gpuPortUniform(&state);
          theta = sqrt((1 - F) * d.K2 * sqrPi / (d.K2 + F * sqrPi));
          phi = gpuPortUniform(&state) * GPU_PORT_PIX2;
          dxp = theta * sin(phi);
          dyp = theta * cos(phi);
        } else {
          zs = e[next * GPU_MATTER_EVENT_VALUES + 1];
          dxp = e[next * GPU_MATTER_EVENT_VALUES + 2];
          dyp = e[next * GPU_MATTER_EVENT_VALUES + 3];
        }
        next++;
        ds += zs * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
        coord[1] += dxp;
        coord[3] += dyp;
        coord[0] += dxp * (d.L1 - zs);
        coord[2] += dyp * (d.L1 - zs);
        ds += (d.L1 - zs) * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
      } else {
        ds += d.L1 * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
        coord[0] += coord[1] * d.L1;
        coord[2] += coord[3] * d.L1;
      }
    }
  }
  if (v[5]) {
    /* lost: gamma fell to 1 or below */
    coord[4] = d.z0 + ds;
    coord[5] = 0;
    return;
  }
  if (d.energyDecay) {
    gamma -= v[4];
    P = sqrt(gamma * gamma - 1);
    coord[5] = (P - d.Po) / d.Po;
    beta = P / gamma;
    coord[4] = t * beta + ds;
  } else
    coord[4] += ds;
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaMatter(void *coord, long nParticles, int stride, const GPU_MATTER_DATA *data,
                             const double *hostValues, const double *hostEvents, long nEvents,
                             float *milliseconds) {
  cudaError_t copyStatus;
  int status;

  if (!coord || !data || (!hostValues && !(data->deviceDraws && data->multipleScattering && !data->energyDecay)) ||
      nParticles <= 0 || stride < 6 || nEvents < 0 || (nEvents && !hostEvents))
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess) && hostValues)
    status = gpuPortUploadValues(hostValues, nParticles * GPU_MATTER_VALUES);
  if (status == static_cast<int>(cudaSuccess) && nEvents) {
    if (gpuMatterEventsCapacity < nEvents * GPU_MATTER_EVENT_VALUES) {
      gpuFreeDevice(gpuMatterEvents);
      gpuMatterEvents = NULL;
      gpuMatterEventsCapacity = 0;
      copyStatus = cudaMalloc(&gpuMatterEvents, nEvents * GPU_MATTER_EVENT_VALUES * sizeof(double));
      if (copyStatus != cudaSuccess)
        return static_cast<int>(copyStatus);
      gpuMatterEventsCapacity = nEvents * GPU_MATTER_EVENT_VALUES;
    }
    status = static_cast<int>(cudaMemcpy(gpuMatterEvents, hostEvents, nEvents * GPU_MATTER_EVENT_VALUES * sizeof(double),
                                         cudaMemcpyHostToDevice));
  }
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuMatterKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, *data,
                                                            hostValues ? gpuPortUpload : NULL, gpuMatterEvents,
                                                            gpuPortCounters);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* PEPPOT: the optional drift, then which particles miss every hole (flag 1) */

__global__ void gpuPeppotKernel(double *particles, long nParticles, int stride, GPU_PEPPOT_DATA d,
                                unsigned char *flags) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *ini, dx, dy;
  long i;

  if (ip >= nParticles)
    return;
  ini = particles + ip * stride;
  if (d.length) {
    ini[0] += d.length * ini[1];
    ini[2] += d.length * ini[3];
  }
  for (i = 0; i < d.nHoles; i++) {
    dx = ini[0] - d.x[i];
    dy = ini[2] - d.y[i];
    if (d.radius2 > dx * dx + dy * dy)
      break;
  }
  flags[ip] = i == d.nHoles ? 1 : 0;
}

static double *gpuPeppotHoles = NULL;
static long gpuPeppotHolesCapacity = 0;

extern "C" int gpuCudaPeppot(void *coord, long nParticles, int stride, const GPU_PEPPOT_DATA *data, const double *hostX,
                             const double *hostY, unsigned char *hostFlags) {
  GPU_PEPPOT_DATA d;
  cudaError_t status;
  int code;

  if (!coord || !data || nParticles <= 0 || stride < 6 || data->nHoles < 0 || !hostFlags)
    return static_cast<int>(cudaErrorInvalidValue);
  code = gpuPortPrepare(nParticles, 1);
  if (code != static_cast<int>(cudaSuccess))
    return code;
  if (gpuPeppotHolesCapacity < 2 * data->nHoles + 2) {
    gpuFreeDevice(gpuPeppotHoles);
    gpuPeppotHoles = NULL;
    gpuPeppotHolesCapacity = 0;
    status = cudaMalloc(&gpuPeppotHoles, (2 * data->nHoles + 2) * sizeof(double));
    if (status != cudaSuccess)
      return static_cast<int>(status);
    gpuPeppotHolesCapacity = 2 * data->nHoles + 2;
  }
  if (data->nHoles > 0) {
    status = cudaMemcpy(gpuPeppotHoles, hostX, data->nHoles * sizeof(double), cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
      status = cudaMemcpy(gpuPeppotHoles + data->nHoles, hostY, data->nHoles * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  d = *data;
  d.x = gpuPeppotHoles;
  d.y = gpuPeppotHoles + data->nHoles;
  gpuPeppotKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(static_cast<double *>(coord), nParticles, stride, d,
                                                            gpuPortFlags);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(hostFlags, gpuPortFlags, nParticles * sizeof(*hostFlags), cudaMemcpyDeviceToHost);
  return static_cast<int>(status);
}

/* ---------------------------------------------------------------------------------------------------- */
/* IONEFFECTS: applyIonKicksToElectronBunch()'s Gaussian-field kicks (ionEffects.cc) to one bunch's rows,
 * with BEAMBEAM's gaussianBeamKick() */

static long long *gpuIonRows = NULL;
static long gpuIonRowsCapacity = 0;

__global__ void gpuIonKickKernel(double *particles, long nRows, int stride, const long long *rows,
                                 GPU_BEAMBEAM_DATA d) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, kick[2];

  if (i >= nRows)
    return;
  coord = particles + (rows ? rows[i] : i) * stride;
  gpuBbGaussianKick(coord, d, kick);
  coord[1] += kick[0] / d.cMks / d.P0;
  coord[3] += kick[1] / d.cMks / d.P0;
}

extern "C" int gpuCudaIonKicks(void *coord, long nRows, int stride, const long long *hostRows,
                               const GPU_BEAMBEAM_DATA *data, float *milliseconds) {
  cudaError_t status;
  int code;

  if (!coord || !data || nRows <= 0 || stride < 6)
    return static_cast<int>(cudaErrorInvalidValue);
  if (hostRows) {
    if (gpuIonRowsCapacity < nRows) {
      gpuFreeDevice(gpuIonRows);
      gpuIonRows = NULL;
      gpuIonRowsCapacity = 0;
      status = cudaMalloc(&gpuIonRows, nRows * sizeof(*gpuIonRows));
      if (status != cudaSuccess)
        return static_cast<int>(status);
      gpuIonRowsCapacity = nRows;
    }
    status = cudaMemcpy(gpuIonRows, hostRows, nRows * sizeof(*gpuIonRows), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  code = gpuPortBegin(milliseconds);
  if (code != static_cast<int>(cudaSuccess))
    return code;
  gpuIonKickKernel<<<gpuPortBlocks(nRows, 128), 128>>>(static_cast<double *>(coord), nRows, stride,
                                                        hostRows ? gpuIonRows : NULL, *data);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* BUMPER (KICKER): track_through_kicker()'s particle loop with its misalignments.  sin, cos, asin and atan
 * are gpu_accurate_trig.h's where the arguments are small (they round as glibc does there), tan is CUDA's. */

static double *gpuKickerTables = NULL;
static long gpuKickerTablesCapacity = 0;
static const double *gpuKickerHostT = NULL, *gpuKickerHostA = NULL, *gpuKickerHostXp = NULL, *gpuKickerHostYp = NULL;
static long gpuKickerHostN = 0, gpuKickerHostMap = 0;

/* interpolateDeflectionMap(), with elegant's flat indexing (a particle in the cell past the last column reads
 * the next row, as on the CPU); values past the table read 0 */
__device__ __forceinline__ double gpuKickerMapValue(const double *table, long index, long values) {
  return index < values ? table[index] : 0.0;
}

__device__ void gpuKickerMap(double *xpFactor, double *ypFactor, const GPU_KICKER_DATA &d, double x, double y) {
  double Fa, Fb, fx, fy;
  long ix, iy;

  *xpFactor = *ypFactor = 0;
  if (isnan(x) || isnan(y) || isinf(x) || isinf(y))
    return;
  ix = (long)((x - d.xmin) / d.dxg);
  iy = (long)((y - d.ymin) / d.dyg);
  if (ix < 0 || iy < 0 || ix > d.nx - 1 || iy > d.ny - 1)
    return;
  fx = (x - (ix * d.dxg + d.xmin)) / d.dxg;
  fy = (y - (iy * d.dyg + d.ymin)) / d.dyg;
  Fa = (1 - fy) * gpuKickerMapValue(d.xpFactor, ix + iy * d.nx, d.mapValues) +
       fy * gpuKickerMapValue(d.xpFactor, ix + (iy + 1) * d.nx, d.mapValues);
  Fb = (1 - fy) * gpuKickerMapValue(d.xpFactor, ix + 1 + iy * d.nx, d.mapValues) +
       fy * gpuKickerMapValue(d.xpFactor, ix + 1 + (iy + 1) * d.nx, d.mapValues);
  *xpFactor = (1 - fx) * Fa + fx * Fb;
  Fa = (1 - fy) * gpuKickerMapValue(d.ypFactor, ix + iy * d.nx, d.mapValues) +
       fy * gpuKickerMapValue(d.ypFactor, ix + (iy + 1) * d.nx, d.mapValues);
  Fb = (1 - fy) * gpuKickerMapValue(d.ypFactor, ix + 1 + iy * d.nx, d.mapValues) +
       fy * gpuKickerMapValue(d.ypFactor, ix + 1 + (iy + 1) * d.nx, d.mapValues);
  *ypFactor = (1 - fx) * Fa + fx * Fb;
}

__global__ void gpuKickerKernel(double *particles, long nParticles, int stride, GPU_KICKER_DATA d,
                                unsigned long long *counters, double *errorTime) {
  long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
  double *coord, angle, time, x, xp, y, yp, s, dp, curv, dx, ds, theta_i, alpha_i, alpha_f;
  long i, n;

  if (ip >= nParticles)
    return;
  coord = particles + ip * stride;
  if (d.doOffset) {
    coord[4] += d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
    coord[0] = coord[0] - d.dx + d.dz * coord[1];
    coord[2] = coord[2] - d.dy + d.dz * coord[3];
  }
  if (d.doTilt) {
    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    coord[0] = x * d.cosTilt + y * d.sinTilt;
    coord[2] = -x * d.sinTilt + y * d.cosTilt;
    coord[1] = xp * d.cosTilt + yp * d.sinTilt;
    coord[3] = -xp * d.sinTilt + yp * d.cosTilt;
  }

  angle = d.angle;
  time = coord[4] / (d.cMks * gpuPortBetaFromDelta(d.pCentral, coord[5])) + d.clockOffset - d.timeOffset;
  if (time > d.tmax || time < d.tmin) {
    if (d.periodic) {
      n = (long)((time - d.tmin) / (d.tmax - d.tmin));
      time -= n * (d.tmax - d.tmin);
    }
  }
  if (time > d.tmax || time < d.tmin)
    angle *= 0;
  else {
    for (i = 1; i < d.nWaveform; i++)
      if (d.tWaveform[i] >= time)
        break;
    if (i == d.nWaveform) {
      /* the CPU stops with an error at the first such particle */
      atomicMin(&counters[2], (unsigned long long)ip);
      errorTime[ip] = time;
      return;
    }
    i--;
    angle *= (d.ampWaveform[i + 1] - d.ampWaveform[i]) / (d.tWaveform[i + 1] - d.tWaveform[i]) *
               (time - d.tWaveform[i]) + d.ampWaveform[i];
  }

  if (!angle) {
    coord[0] += d.length * coord[1];
    coord[2] += d.length * coord[3];
    coord[4] += d.length * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
  } else {
    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    s = coord[4];
    dp = coord[5];
    if (d.nKicks <= 0) {
      if (d.length != 0) {
        curv = gpuAccurateSin(-angle) / d.length / (1 + dp);
        theta_i = gpuAccurateAtan(xp);
        alpha_i = -theta_i;
        alpha_f = gpuAccurateAsin(d.length * curv + gpuAccurateSin(alpha_i));
        if (fabs(curv * d.length) < 1e-12) {
          x += (dx = d.length * xp);
          s += (ds = fabs(d.length / gpuAccurateCos(alpha_i)));
        } else {
          x += (dx = (gpuAccurateCos(alpha_f) - gpuAccurateCos(alpha_i)) / curv);
          s += (ds = fabs((alpha_f - alpha_i) / curv));
        }
        xp = tan(theta_i - (alpha_f - alpha_i));
        y += d.length * yp;
      } else
        xp += angle / (1 + dp);
    } else if (!d.deflectionMap) {
      double l1 = d.length / d.nKicks, x0, y0;
      if (l1 != 0) {
        for (i = 0; i < d.nKicks; i++) {
          curv = gpuAccurateSin(-angle) / d.length / (1 + dp) * (1 + d.b2 * (x * x - y * y));
          theta_i = gpuAccurateAtan(xp);
          alpha_i = -theta_i;
          alpha_f = gpuAccurateAsin(l1 * curv + gpuAccurateSin(alpha_i));
          x0 = x;
          if (fabs(curv * l1) < 1e-12) {
            x += (dx = l1 * xp);
            s += (ds = fabs(l1 / gpuAccurateCos(alpha_i)));
          } else {
            x += (dx = (gpuAccurateCos(alpha_f) - gpuAccurateCos(alpha_i)) / curv);
            s += (ds = fabs((alpha_f - alpha_i) / curv));
          }
          xp = tan(theta_i - (alpha_f - alpha_i));
          x0 = (x + x0) / 2;
          y0 = y;
          y += l1 * yp;
          y0 = (y + y0) / 2;
          yp += -2 * gpuAccurateSin(angle) / d.length / (1 + dp) * ds * x0 * y0 * d.b2;
        }
      } else
        xp += angle / (1 + dp);
    } else {
      double l1 = d.length / d.nKicks, l2 = d.length / d.nKicks / 2., xpFactor, ypFactor;
      angle /= d.nKicks;
      for (i = 0; i < d.nKicks; i++) {
        if (i == 0) {
          s += l2 * (1 + (xp * xp + yp * yp) / 2.);
          x += xp * l2;
          y += yp * l2;
        } else {
          s += l1 * (1 + (xp * xp + yp * yp) / 2.);
          x += xp * l1;
          y += yp * l1;
        }
        gpuKickerMap(&xpFactor, &ypFactor, d, x, y);
        xp += angle * xpFactor / (1 + dp);
        yp += angle * ypFactor / (1 + dp);
      }
      s += l2 * (1 + (xp * xp + yp * yp) / 2.);
      x += xp * l2;
      y += yp * l2;
    }
    coord[0] = x;
    coord[1] = xp;
    coord[2] = y;
    coord[3] = yp;
    coord[4] = s;
  }

  if (d.doTilt) {
    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    coord[0] = x * d.cosUntilt + y * d.sinUntilt;
    coord[2] = -x * d.sinUntilt + y * d.cosUntilt;
    coord[1] = xp * d.cosUntilt + yp * d.sinUntilt;
    coord[3] = -xp * d.sinUntilt + yp * d.cosUntilt;
  }
  if (d.doOffset) {
    coord[4] += -d.dz * sqrt(1 + coord[1] * coord[1] + coord[3] * coord[3]);
    coord[0] = coord[0] - -d.dx + -d.dz * coord[1];
    coord[2] = coord[2] - -d.dy + -d.dz * coord[3];
  }
  gpuPortCountInvalid(coord, counters);
}

extern "C" int gpuCudaKicker(void *coord, long nParticles, int stride, const GPU_KICKER_DATA *data,
                             const double *tWaveform, const double *ampWaveform, const double *xpFactor,
                             const double *ypFactor, float *milliseconds) {
  GPU_KICKER_DATA d;
  cudaError_t cudaStatus;
  long need;
  int status;

  if (!coord || !data || !tWaveform || !ampWaveform || nParticles <= 0 || stride < 6 || data->nWaveform < 2 ||
      (data->deflectionMap && (!xpFactor || !ypFactor || data->mapValues < 1)))
    return static_cast<int>(cudaErrorInvalidValue);
  status = gpuPortPrepare(nParticles, 0);
  if (status == static_cast<int>(cudaSuccess))
    status = gpuPortEnsureValues(nParticles);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* the waveform and map, uploaded again only when the element's tables change */
  need = 2 * data->nWaveform + (data->deflectionMap ? 2 * data->mapValues : 0);
  if (gpuKickerTablesCapacity < need) {
    gpuFreeDevice(gpuKickerTables);
    gpuKickerTables = NULL;
    gpuKickerTablesCapacity = 0;
    gpuKickerHostT = gpuKickerHostA = gpuKickerHostXp = gpuKickerHostYp = NULL;
    cudaStatus = cudaMalloc(&gpuKickerTables, need * sizeof(double));
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuKickerTablesCapacity = need;
  }
  if (gpuKickerHostT != tWaveform || gpuKickerHostA != ampWaveform || gpuKickerHostN != data->nWaveform ||
      gpuKickerHostXp != xpFactor || gpuKickerHostYp != ypFactor || gpuKickerHostMap != data->mapValues) {
    cudaStatus = cudaMemcpy(gpuKickerTables, tWaveform, data->nWaveform * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess)
      cudaStatus = cudaMemcpy(gpuKickerTables + data->nWaveform, ampWaveform, data->nWaveform * sizeof(double),
                              cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess && data->deflectionMap)
      cudaStatus = cudaMemcpy(gpuKickerTables + 2 * data->nWaveform, xpFactor, data->mapValues * sizeof(double),
                              cudaMemcpyHostToDevice);
    if (cudaStatus == cudaSuccess && data->deflectionMap)
      cudaStatus = cudaMemcpy(gpuKickerTables + 2 * data->nWaveform + data->mapValues, ypFactor,
                              data->mapValues * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
      return static_cast<int>(cudaStatus);
    gpuKickerHostT = tWaveform;
    gpuKickerHostA = ampWaveform;
    gpuKickerHostN = data->nWaveform;
    gpuKickerHostXp = xpFactor;
    gpuKickerHostYp = ypFactor;
    gpuKickerHostMap = data->mapValues;
  }
  d = *data;
  d.tWaveform = gpuKickerTables;
  d.ampWaveform = gpuKickerTables + data->nWaveform;
  if (data->deflectionMap) {
    d.xpFactor = gpuKickerTables + 2 * data->nWaveform;
    d.ypFactor = gpuKickerTables + 2 * data->nWaveform + data->mapValues;
  }
  status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuKickerKernel<<<gpuPortBlocks(nParticles, 128), 128>>>(static_cast<double *>(coord), nParticles, stride, d,
                                                           gpuPortCounters, gpuPortValues);
  return gpuPortEnd(milliseconds);
}

/* ---------------------------------------------------------------------------------------------------- */
/* SLICE: dump_slice_analysis()'s numbers (slicePoint.c), the CPU's to the bit.  The device makes the times and
 * the bunch's earliest and latest (the first particle with either wins a tie, as the CPU's loop has it) and, from
 * the host's slice limits, each slice's members in particle order (a stable sort by slice), their columns side by
 * side; the host adds them (gpu_slice_point_sums()).  Sums in order with KahanPlus() are chains of dependent
 * double additions, which on this device took 300 to 600 ns a value each (100 ms a call at 1,000,000 particles);
 * the host runs a slice's chains together, a few ns a value. */

static struct {
  double *time;
  int *first, *count;
  long long *offset;
  long particleCapacity;
  int *keys[2], *values[2];
  double *compact; /* the members' x, x', y, y', s, delta and time, column after column */
  long pairCapacity;
  double *t0;
  int *sliceCount;
  long sliceCapacity;
  double *hostMembers; /* pinned: the members' columns for the host's sums */
  long long hostMembersCapacity;
  double *blockValue;   /* per block: min, max */
  long long *blockIndex;
  void *cubTemp;
  size_t cubBytes;
} gpuSlicePoint;

#define GPU_SLICE_TIME_BLOCKS 256
#define GPU_SLICE_TIME_THREADS 256

/* The first of equal values wins, as the CPU's strict comparisons in particle order have it */
__device__ __forceinline__ void gpuSliceBetterMax(double *value, long long *index, double v, long long i) {
  if (v > *value || (v == *value && i < *index)) {
    *value = v;
    *index = i;
  }
}

__device__ __forceinline__ void gpuSliceBetterMin(double *value, long long *index, double v, long long i) {
  if (v < *value || (v == *value && i < *index)) {
    *value = v;
    *index = i;
  }
}

/* dump_slice_analysis()'s time loop; particles the PID range leaves out get -DBL_MAX and take no part in the
 * extremes */
__global__ void gpuSlicePointTimeKernel(const double *coord, long nParticles, int stride, double Po, double cMks,
                                        double clockOffset, int allPIDs, double startPID, double endPID,
                                        double *time, double *blockValue, long long *blockIndex) {
  __shared__ double minValue[GPU_SLICE_TIME_THREADS], maxValue[GPU_SLICE_TIME_THREADS];
  __shared__ long long minIndex[GPU_SLICE_TIME_THREADS], maxIndex[GPU_SLICE_TIME_THREADS];
  double tMin = DBL_MAX, tMax = -DBL_MAX;
  long long iMin = LLONG_MAX, iMax = LLONG_MAX;
  int tid = threadIdx.x;

  for (long i = (long)blockIdx.x * blockDim.x + tid; i < nParticles; i += (long)gridDim.x * blockDim.x) {
    const double *part = coord + i * stride;
    if (allPIDs || (part[6] >= startPID && part[6] <= endPID)) {
      double p = Po * (1 + part[5]), gamma = sqrt(p * p + 1), beta = p / gamma;
      double t = part[4] / (cMks * beta) + clockOffset;
      time[i] = t;
      gpuSliceBetterMax(&tMax, &iMax, t, i);
      gpuSliceBetterMin(&tMin, &iMin, t, i);
    } else
      time[i] = -DBL_MAX;
  }
  minValue[tid] = tMin;
  minIndex[tid] = iMin;
  maxValue[tid] = tMax;
  maxIndex[tid] = iMax;
  __syncthreads();
  for (int half = blockDim.x / 2; half > 0; half /= 2) {
    if (tid < half) {
      gpuSliceBetterMin(&minValue[tid], &minIndex[tid], minValue[tid + half], minIndex[tid + half]);
      gpuSliceBetterMax(&maxValue[tid], &maxIndex[tid], maxValue[tid + half], maxIndex[tid + half]);
    }
    __syncthreads();
  }
  if (!tid) {
    blockValue[2 * blockIdx.x] = minValue[0];
    blockValue[2 * blockIdx.x + 1] = maxValue[0];
    blockIndex[2 * blockIdx.x] = minIndex[0];
    blockIndex[2 * blockIdx.x + 1] = maxIndex[0];
  }
}

/* sliceRange() (slicePoint.c): the slices with t0[k] <= t <= t0[k] + dt0 are first..last */
__device__ __forceinline__ void gpuSliceRange(double t, const double *t0, double dt0, long nSlices, long *first,
                                              long *last) {
  long low = 0, high = nSlices;

  while (low < high) {
    long middle = low + (high - low) / 2;
    if (t0[middle] + dt0 >= t)
      high = middle;
    else
      low = middle + 1;
  }
  *first = low;
  low = -1;
  high = nSlices - 1;
  while (low < high) {
    long middle = low + (high - low + 1) / 2;
    if (t0[middle] <= t)
      low = middle;
    else
      high = middle - 1;
  }
  *last = low;
}

/* findSliceMembers()'s test: accumulate_beam_sums()'s PID range, a time that is a number, then the slices */
__global__ void gpuSlicePointMemberKernel(const double *coord, long nParticles, int stride, int allPIDs,
                                          double startPID, double endPID, const double *time, const double *t0,
                                          double dt0, long nSlices, int *first, int *count, int *sliceCount) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long f, l, k;
  double t;

  if (i >= nParticles)
    return;
  first[i] = 0;
  count[i] = 0;
  if (!(allPIDs || (coord[i * stride + 6] >= startPID && coord[i * stride + 6] <= endPID)))
    return;
  t = time[i];
  if (!(t == t))
    return;
  gpuSliceRange(t, t0, dt0, nSlices, &f, &l);
  if (l < f)
    return;
  first[i] = (int)f;
  count[i] = (int)(l - f + 1);
  for (k = f; k <= l; k++)
    atomicAdd(&sliceCount[k], 1);
}

/* A (slice, particle) pair for each membership, in particle order */
__global__ void gpuSlicePointPairKernel(long nParticles, const int *first, const int *count, const long long *offset,
                                        int *keys, int *values) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nParticles)
    return;
  for (int k = 0; k < count[i]; k++) {
    keys[offset[i] + k] = first[i] + k;
    values[offset[i] + k] = (int)i;
  }
}

/* The members' coordinates and times, slice after slice, each slice in particle order */
__global__ void gpuSlicePointGatherKernel(const double *coord, int stride, const double *time, const int *member,
                                          long long nPairs, double *compact) {
  long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long ip;

  if (j >= nPairs)
    return;
  ip = member[j];
  for (int c = 0; c < 6; c++)
    compact[c * nPairs + j] = coord[ip * stride + c];
  compact[6 * nPairs + j] = time[ip];
}

static int gpuSlicePointGrow(void **pointer, long count, size_t size) {
  gpuFreeDevice(*pointer);
  *pointer = NULL;
  return static_cast<int>(cudaMalloc(pointer, count * size));
}

/* The times into gpuSlicePoint.time and the bunch's extremes */
extern "C" int gpuCudaSlicePointTimes(const void *coord, long nParticles, int stride, double Po, double cMks,
                                      double clockOffset, int allPIDs, double startPID, double endPID,
                                      double *tMin, double *tMax, float *milliseconds) {
  double value[2 * GPU_SLICE_TIME_BLOCKS];
  long long index[2 * GPU_SLICE_TIME_BLOCKS];
  long long iMin = LLONG_MAX, iMax = LLONG_MAX;
  int status;

  if (!coord || nParticles <= 0 || stride < 7 || !tMin || !tMax)
    return static_cast<int>(cudaErrorInvalidValue);
  if (gpuSlicePoint.particleCapacity < nParticles) {
    gpuSlicePoint.particleCapacity = 0;
    if ((status = gpuSlicePointGrow((void **)&gpuSlicePoint.time, nParticles, sizeof(double))) ||
        (status = gpuSlicePointGrow((void **)&gpuSlicePoint.first, nParticles, sizeof(int))) ||
        (status = gpuSlicePointGrow((void **)&gpuSlicePoint.count, nParticles, sizeof(int))) ||
        (status = gpuSlicePointGrow((void **)&gpuSlicePoint.offset, nParticles, sizeof(long long))))
      return status;
    gpuSlicePoint.particleCapacity = nParticles;
  }
  if (!gpuSlicePoint.blockValue &&
      ((status = gpuSlicePointGrow((void **)&gpuSlicePoint.blockValue, 2 * GPU_SLICE_TIME_BLOCKS, sizeof(double))) ||
       (status = gpuSlicePointGrow((void **)&gpuSlicePoint.blockIndex, 2 * GPU_SLICE_TIME_BLOCKS,
                                   sizeof(long long)))))
    return status;
  status = gpuPortBegin(milliseconds);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSlicePointTimeKernel<<<GPU_SLICE_TIME_BLOCKS, GPU_SLICE_TIME_THREADS>>>(
    static_cast<const double *>(coord), nParticles, stride, Po, cMks, clockOffset, allPIDs, startPID, endPID,
    gpuSlicePoint.time, gpuSlicePoint.blockValue, gpuSlicePoint.blockIndex);
  status = gpuPortEnd(milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemcpy(value, gpuSlicePoint.blockValue, sizeof(value), cudaMemcpyDeviceToHost));
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemcpy(index, gpuSlicePoint.blockIndex, sizeof(index), cudaMemcpyDeviceToHost));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  /* the CPU's starting values, then the blocks' winners with the same rule */
  *tMin = DBL_MAX;
  *tMax = -DBL_MAX;
  for (int b = 0; b < GPU_SLICE_TIME_BLOCKS; b++) {
    if (value[2 * b] < *tMin || (value[2 * b] == *tMin && index[2 * b] < iMin)) {
      *tMin = value[2 * b];
      iMin = index[2 * b];
    }
    if (value[2 * b + 1] > *tMax || (value[2 * b + 1] == *tMax && index[2 * b + 1] < iMax)) {
      *tMax = value[2 * b + 1];
      iMax = index[2 * b + 1];
    }
  }
  return static_cast<int>(cudaSuccess);
}

/* Each slice's count and first member, and its members' seven columns (column c of slice k's member m at
 * members[c * pairs + start[k] + m]) in pinned host memory kept until the next call.  Call after
 * gpuCudaSlicePointTimes(). */
extern "C" int gpuCudaSlicePointMembers(const void *coord, long nParticles, int stride, int allPIDs,
                                        double startPID, double endPID, const double *t0, double dt0, long nSlices,
                                        long *count, long long *start, const double **members, long long *pairs,
                                        float *milliseconds) {
  long long last, nPairs, total = 0;
  int lastCount, status, bits = 1, *hostCount;
  size_t bytes = 0, sortBytes = 0;

  if (!coord || nParticles <= 0 || nSlices <= 0 || nSlices > INT_MAX || !t0 || !count || !start || !members ||
      !pairs || gpuSlicePoint.particleCapacity < nParticles)
    return static_cast<int>(cudaErrorInvalidValue);
  *pairs = 0;
  *members = NULL;
  if (gpuSlicePoint.sliceCapacity < nSlices) {
    gpuSlicePoint.sliceCapacity = 0;
    if ((status = gpuSlicePointGrow((void **)&gpuSlicePoint.t0, nSlices, sizeof(double))) ||
        (status = gpuSlicePointGrow((void **)&gpuSlicePoint.sliceCount, nSlices, sizeof(int))))
      return status;
    gpuSlicePoint.sliceCapacity = nSlices;
  }
  status = gpuPortBegin(milliseconds);
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemcpy(gpuSlicePoint.t0, t0, nSlices * sizeof(double), cudaMemcpyHostToDevice));
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemset(gpuSlicePoint.sliceCount, 0, nSlices * sizeof(int)));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSlicePointMemberKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(
    static_cast<const double *>(coord), nParticles, stride, allPIDs, startPID, endPID, gpuSlicePoint.time,
    gpuSlicePoint.t0, dt0, nSlices, gpuSlicePoint.first, gpuSlicePoint.count, gpuSlicePoint.sliceCount);
  /* where each particle's pairs start */
  status = static_cast<int>(cub::DeviceScan::ExclusiveSum(NULL, bytes, gpuSlicePoint.count, gpuSlicePoint.offset,
                                                          (int)nParticles));
  if (status == static_cast<int>(cudaSuccess) && bytes > gpuSlicePoint.cubBytes) {
    gpuFreeDevice(gpuSlicePoint.cubTemp);
    gpuSlicePoint.cubTemp = NULL;
    gpuSlicePoint.cubBytes = 0;
    if ((status = static_cast<int>(cudaMalloc(&gpuSlicePoint.cubTemp, bytes))) == static_cast<int>(cudaSuccess))
      gpuSlicePoint.cubBytes = bytes;
  }
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cub::DeviceScan::ExclusiveSum(gpuSlicePoint.cubTemp, bytes, gpuSlicePoint.count,
                                                            gpuSlicePoint.offset, (int)nParticles));
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemcpy(&last, gpuSlicePoint.offset + nParticles - 1, sizeof(long long),
                                         cudaMemcpyDeviceToHost));
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cudaMemcpy(&lastCount, gpuSlicePoint.count + nParticles - 1, sizeof(int),
                                         cudaMemcpyDeviceToHost));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  nPairs = last + lastCount;
  if (nPairs > INT_MAX)
    return static_cast<int>(cudaErrorInvalidValue);
  /* the slices' counts and starts */
  if (!(hostCount = static_cast<int *>(std::malloc(nSlices * sizeof(int)))))
    return static_cast<int>(cudaErrorMemoryAllocation);
  status = static_cast<int>(cudaMemcpy(hostCount, gpuSlicePoint.sliceCount, nSlices * sizeof(int),
                                       cudaMemcpyDeviceToHost));
  for (long s = 0; s < nSlices; s++) {
    start[s] = total;
    total += hostCount[s];
    count[s] = hostCount[s];
  }
  std::free(hostCount);
  if (status == static_cast<int>(cudaSuccess) && total != nPairs)
    status = static_cast<int>(cudaErrorUnknown);
  if (status != static_cast<int>(cudaSuccess))
    return status;
  *pairs = nPairs;
  if (!nPairs)
    return gpuPortEnd(milliseconds);
  if (gpuSlicePoint.pairCapacity < nPairs) {
    gpuSlicePoint.pairCapacity = 0;
    for (int b = 0; b < 2; b++)
      if ((status = gpuSlicePointGrow((void **)&gpuSlicePoint.keys[b], nPairs, sizeof(int))) ||
          (status = gpuSlicePointGrow((void **)&gpuSlicePoint.values[b], nPairs, sizeof(int))))
        return status;
    if ((status = gpuSlicePointGrow((void **)&gpuSlicePoint.compact, GPU_SLICE_COLUMNS * nPairs, sizeof(double))))
      return status;
    gpuSlicePoint.pairCapacity = nPairs;
  }
  if (gpuSlicePoint.hostMembersCapacity < nPairs) {
    if (gpuSlicePoint.hostMembers)
      cudaFreeHost(gpuSlicePoint.hostMembers);
    gpuSlicePoint.hostMembers = NULL;
    gpuSlicePoint.hostMembersCapacity = 0;
    status = static_cast<int>(cudaMallocHost(&gpuSlicePoint.hostMembers, GPU_SLICE_COLUMNS * nPairs * sizeof(double)));
    if (status != static_cast<int>(cudaSuccess))
      return status;
    gpuSlicePoint.hostMembersCapacity = nPairs;
  }
  gpuSlicePointPairKernel<<<gpuPortBlocks(nParticles, 256), 256>>>(nParticles, gpuSlicePoint.first,
                                                                   gpuSlicePoint.count, gpuSlicePoint.offset,
                                                                   gpuSlicePoint.keys[0], gpuSlicePoint.values[0]);
  /* stable, so each slice keeps particle order */
  while ((1L << bits) < nSlices)
    bits++;
  status = static_cast<int>(cub::DeviceRadixSort::SortPairs(NULL, sortBytes, gpuSlicePoint.keys[0],
                                                            gpuSlicePoint.keys[1], gpuSlicePoint.values[0],
                                                            gpuSlicePoint.values[1], (int)nPairs, 0, bits));
  if (status == static_cast<int>(cudaSuccess) && sortBytes > gpuSlicePoint.cubBytes) {
    gpuFreeDevice(gpuSlicePoint.cubTemp);
    gpuSlicePoint.cubTemp = NULL;
    gpuSlicePoint.cubBytes = 0;
    if ((status = static_cast<int>(cudaMalloc(&gpuSlicePoint.cubTemp, sortBytes))) == static_cast<int>(cudaSuccess))
      gpuSlicePoint.cubBytes = sortBytes;
  }
  if (status == static_cast<int>(cudaSuccess))
    status = static_cast<int>(cub::DeviceRadixSort::SortPairs(gpuSlicePoint.cubTemp, sortBytes, gpuSlicePoint.keys[0],
                                                              gpuSlicePoint.keys[1], gpuSlicePoint.values[0],
                                                              gpuSlicePoint.values[1], (int)nPairs, 0, bits));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  gpuSlicePointGatherKernel<<<static_cast<int>((nPairs + 255) / 256), 256>>>(
    static_cast<const double *>(coord), stride, gpuSlicePoint.time, gpuSlicePoint.values[1], nPairs,
    gpuSlicePoint.compact);
  status = static_cast<int>(cudaMemcpy(gpuSlicePoint.hostMembers, gpuSlicePoint.compact,
                                       GPU_SLICE_COLUMNS * nPairs * sizeof(double), cudaMemcpyDeviceToHost));
  if (status != static_cast<int>(cudaSuccess))
    return status;
  *members = gpuSlicePoint.hostMembers;
  return gpuPortEnd(milliseconds);
}
