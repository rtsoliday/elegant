/* Host side of the element types moved to the device on November 15, 2026 (gpu_ports.h).  Included at
 * the end of gpu_stub.c, whose helpers it uses.  Each gpu_track_*() takes what its CPU routine takes, does
 * the CPU's set-up on the host, and either tracks the call on the device or hands it to the CPU routine
 * (an option the device lacks, an error the CPU reports, or settings the CPU rejects). */

/* elegantto links gpu_stub.c without the tracking routines */
#pragma weak polynomial_kicks
#pragma weak polynomial_hamiltonian
#pragma weak track_through_mkicker
#pragma weak drift_beam
#pragma weak trackingClockOffset
/* kicker.c doesn't declare this in track.h */
void set_up_mkicker(MKICKER *kicker);
#pragma weak set_up_mkicker
/* track_rf.c and shrfdf.c don't declare these in track.h */
void set_up_mrfdf(MRFDF *rf_param, double **initial, int64_t n_particles, double pc_central);
void set_up_shrfdf(SHRFDF *rf_param, double **initial, int64_t n_particles, double pc_central);
void set_up_rftm110(RFTM110 *rf_param, double **initial, int64_t n_particles, double pc_central);
void track_through_space_harmonic_deflector(double **final, SHRFDF *rf_param, double **initial,
                                            int64_t n_particles, double pc_central);
#pragma weak set_up_mrfdf
#pragma weak set_up_shrfdf
#pragma weak set_up_rftm110
#pragma weak track_through_ramped_deflector
#pragma weak track_through_multipole_deflector
#pragma weak track_through_space_harmonic_deflector
#pragma weak track_through_rftm110_deflector
#pragma weak GetNoiseGroupValue
#pragma weak random_3
#pragma weak find_nearby_array_entry
#pragma weak linear_interpolation
/* the BOFFAXE field table (magneticFieldOffAxisExpansion.c) */
long boffaxeFieldTable(BOFFAXE *boa, long *nz, long *nDz1, double *dz, double *zMin, double *zMax,
                       double ***fieldData);
#pragma weak boffaxeFieldTable
#pragma weak alpha_magnet_tracking
#pragma weak track_through_kick_sbend
#pragma weak trackMagneticFieldOffAxisExpansion
#pragma weak InitializeAPPLE
#pragma weak APPLE_Track
#pragma weak rotateCoordinatesForMisalignment
#pragma weak track_particles
#pragma weak track_through_pfilter
#pragma weak sample_particles
#pragma weak remove_outlier_particles
void transformEmittances(double **coord, int64_t np, double pCentral, EMITTANCEELEMENT *ee);
#pragma weak transformEmittances
#pragma weak swapParticles
#pragma weak remove_correlations
#pragma weak compute_percentiles
#pragma weak accumulate_beam_sums
#pragma weak allocateBeamSums
#pragma weak zero_beam_sums
#pragma weak freeBeamSums
#pragma weak computeEmitTwissFromSigmaMatrix
#pragma weak KahanPlus
#pragma weak random_2
/* do_tracking.c doesn't declare this in track.h */
void findMinMax(double **coord, int64_t np, double *min, double *max, double *c0, double Po);
#pragma weak findMinMax

extern int gpuCudaPortCounters(unsigned long long *counters);
extern int gpuCudaPortFlags(unsigned char *flags, long nParticles);
extern int gpuCudaPortValue(long index, double *value);
extern int gpuCudaPortPermute(void *coord, long nParticles, int stride, const long long *permHost);
extern int gpuCudaKpoly(void *coord, long nParticles, int stride, const GPU_KPOLY_DATA *data, float *milliseconds);
extern int gpuCudaHkpoly(void *coord, long nParticles, int stride, const GPU_HKPOLY_DATA *data,
                         float *milliseconds);
extern int gpuCudaMbumper(void *coord, long nParticles, int stride, const GPU_MBUMPER_DATA *data,
                          const double *tWaveform, const double *ampWaveform, float *milliseconds);

#define GPU_PORT_NO_ERROR (~0ULL)

/* Settings each type's CPU routine rejects stay on the CPU, for its message */
static long gpuKpolySupported(ELEMENT_LIST *eptr) {
  KPOLY *kpoly = (KPOLY *)eptr->p_elem;

  if (kpoly->order < 0)
    return 0;
  if (kpoly->plane && !(kpoly->plane[0] == 'x' || kpoly->plane[0] == 'X' ||
                        kpoly->plane[0] == 'y' || kpoly->plane[0] == 'Y'))
    return 0;
  return 1;
}

static long gpuHkpolySupported(ELEMENT_LIST *eptr) {
  HKPOLY *hkpoly = (HKPOLY *)eptr->p_elem;
  long i, j, k;

  if (hkpoly->length < 0 || hkpoly->nRepeats < 0 || (hkpoly->driftType != 1 && hkpoly->driftType != 2))
    return 0;
  /* the CPU stops when a coefficient of the other drift type is set */
  for (i = 0; i < 7; i++)
    for (j = 0; j < 7; j++) {
      if (hkpoly->driftType == 2 && hkpoly->D[i][j] != 0)
        return 0;
      for (k = 0; k < 7; k++)
        if (hkpoly->driftType == 1 && hkpoly->E[i][j][k] != 0)
          return 0;
    }
  return 1;
}

static long gpuMbumperSupported(ELEMENT_LIST *eptr) {
  MKICKER *kicker = (MKICKER *)eptr->p_elem;

  return kicker->order >= 1 && kicker->order <= GPU_MBUMPER_MAX_ORDER && !spinCoordOffset &&
         !(obstructionDataPresent && obstructionDataPresent());
}

static long gpuRmdfSupported(ELEMENT_LIST *eptr) {
  RMDF *rmdf = (RMDF *)eptr->p_elem;

  return rmdf->gap != 0 && rmdf->ramp_time != 0;
}

static long gpuMrfdfSupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  return !spinCoordOffset;
}

static long gpuShrfdfSupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  return !spinCoordOffset;
}

static long gpuRftm110Supported(ELEMENT_LIST *eptr) {
  RFTM110 *rf = (RFTM110 *)eptr->p_elem;

  if (rf->frequency == 0 || spinCoordOffset)
    return 0;
#ifdef GPU_VERIFY
  /* the CPU's rerun draws the noise again from random_3, whose state can't be put back */
  if (rf->voltageNoise || rf->phaseNoise || rf->voltageNoiseGroup || rf->phaseNoiseGroup)
    return 0;
#endif
  return 1;
}

static long gpuStraySupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  return 1;
}

static long gpuAlphSupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  return 1;
}

static long gpuKsbendSupported(ELEMENT_LIST *eptr) {
  KSBEND *ksbend = (KSBEND *)eptr->p_elem;
  static char *methods[2] = {"leap-frog", "modified-midpoint"};

  return ksbend->angle != 0 && ksbend->n_kicks >= 1 && ksbend->method &&
         match_string(ksbend->method, methods, 2, 0) >= 0;
}

static long gpuBoffaxeSupported(ELEMENT_LIST *eptr) {
  BOFFAXE *boa = (BOFFAXE *)eptr->p_elem;

  /* ISR draws random_2 at every step; the particle output file is written particle by particle */
  if ((boa->synchRad && boa->isr) || boa->particleOutputFile || spinCoordOffset)
    return 0;
  return boa->order == 1 || boa->order == 2;
}

static long gpuAppleSupported(ELEMENT_LIST *eptr) {
  APPLE *apple = (APPLE *)eptr->p_elem;

  /* Until CPU fix 9 is committed (elegant-CPU-bugs-and-fixes.md), APPLE_Track()'s exit offset moves only the
   * particle with ID 0, where the device moves every particle back; with DX, DY or DZ the CPU tracks.  Remove
   * the offset test with that fix. */
  if (apple && (apple->dx || apple->dy || apple->dz))
    return 0;
  return !spinCoordOffset;
}

static long gpuAlwaysSupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  return 1;
}

static long gpuCleanSupported(ELEMENT_LIST *eptr) {
  CLEAN *clean = (CLEAN *)eptr->p_elem;
  static char *modes[3] = {"stdeviation", "absdeviation", "absvalue"};

  return clean->mode && match_string(clean->mode, modes, 3, 0) >= 0;
}

static long gpuRemcorSupported(ELEMENT_LIST *eptr) {
  REMCOR *remcor = (REMCOR *)eptr->p_elem;

  return remcor->with >= 1 && remcor->with <= 6;
}

static long gpuScatterSupported(ELEMENT_LIST *eptr);
static long gpuKickerSupported(ELEMENT_LIST *eptr);
static long gpuDscatterSupported(ELEMENT_LIST *eptr);
static long gpuMatterSupported(ELEMENT_LIST *eptr);

/* The types, what elegant spends a particle on one (feature table 11 at 10,000 particles, for
 * gpuSmallBeamElementCost()) and what a call costs the device with its counters read back */
typedef struct {
  long type;
  const char *setting;
  long (*supported)(ELEMENT_LIST *eptr);
  double cpuNs, deviceUs;
  long enabled;
} GPU_PORT_TYPE;

static GPU_PORT_TYPE gpuPortTypes[] = {
  {T_KPOLY, "ELEGANT_GPU_ENABLE_KPOLY", gpuKpolySupported, 7, 15, 0},
  {T_HKPOLY, "ELEGANT_GPU_ENABLE_HKPOLY", gpuHkpolySupported, 26, 20, 0},
  /* about 1,600 ns a particle with a 1,000-point waveform: the linear search and four kicks */
  {T_MKICKER, "ELEGANT_GPU_ENABLE_MBUMPER", gpuMbumperSupported, 230, 25, 0},
  {T_RMDF, "ELEGANT_GPU_ENABLE_RMDF", gpuRmdfSupported, 208, 15, 0},
  {T_MRFDF, "ELEGANT_GPU_ENABLE_MRFDF", gpuMrfdfSupported, 33, 15, 0},
  {T_SHRFDF, "ELEGANT_GPU_ENABLE_SHRFDF", gpuShrfdfSupported, 21, 15, 0},
  {T_RFTM110, "ELEGANT_GPU_ENABLE_RFTM110", gpuRftm110Supported, 30, 15, 0},
  /* STRAY and ALPH are matrices, which track_particles() puts on the device */
  {T_STRAY, "ELEGANT_GPU_ENABLE_STRAY", gpuStraySupported, 63, 12, 0},
  {T_ALPH, "ELEGANT_GPU_ENABLE_ALPH", gpuAlphSupported, 63, 15, 0},
  {T_KSBEND, "ELEGANT_GPU_ENABLE_KSBEND", gpuKsbendSupported, 198, 20, 0},
  /* about 27 us a particle with the test set's 100,000 steps */
  {T_BOFFAXE, "ELEGANT_GPU_ENABLE_BOFFAXE", gpuBoffaxeSupported, 27000, 200, 0},
  {T_APPLE, "ELEGANT_GPU_ENABLE_GFWIGGLER", gpuAppleSupported, 143, 20, 0},
  /* filters and beam adjustments: the host's sums of copied columns, in particle order */
  {T_PFILTER, "ELEGANT_GPU_ENABLE_PFILTER", gpuAlwaysSupported, 2, 20, 0},
  {T_SAMPLE, "ELEGANT_GPU_ENABLE_SAMPLE", gpuAlwaysSupported, 2, 20, 0},
  {T_CLEAN, "ELEGANT_GPU_ENABLE_CLEAN", gpuCleanSupported, 11, 40, 0},
  {T_EMITTANCE, "ELEGANT_GPU_ENABLE_EMITTANCE", gpuAlwaysSupported, 75, 40, 0},
  {T_TWISSELEMENT, "ELEGANT_GPU_ENABLE_TWISS", gpuAlwaysSupported, 40, 40, 0},
  {T_REMCOR, "ELEGANT_GPU_ENABLE_REMCOR", gpuRemcorSupported, 7, 30, 0},
  {T_MHISTOGRAM, "ELEGANT_GPU_ENABLE_MHISTOGRAM", gpuAlwaysSupported, 46, 60, 0},
  /* the optical stochastic cooler: the pickup reads a copy, the kicker changes delta on the host */
  {T_CPICKUP, "ELEGANT_GPU_ENABLE_COOLER", gpuAlwaysSupported, 100, 40, 0},
  {T_CKICKER, "ELEGANT_GPU_ENABLE_COOLER", gpuAlwaysSupported, 100, 40, 0},
  /* random elements: the host still draws the numbers, so the device saves the coordinate work and the
   * beam's trip to the host and back */
  {T_SCATTER, "ELEGANT_GPU_ENABLE_SCATTER", gpuScatterSupported, 5, 30, 0},
  {T_DSCATTER, "ELEGANT_GPU_ENABLE_DSCATTER", gpuDscatterSupported, 5, 30, 0},
  {T_MATTER, "ELEGANT_GPU_ENABLE_MATTER", gpuMatterSupported, 20, 40, 0},
  {T_PEPPOT, "ELEGANT_GPU_ENABLE_PEPPOT", gpuAlwaysSupported, 10, 40, 0},
  /* CBSCAT tracks on a host copy and sends x', y', s and delta back */
  {T_CBSCAT, "ELEGANT_GPU_ENABLE_CBSCAT", gpuAlwaysSupported, 0, 40, 0},
  /* BUMPER: a waveform search and an arc a particle (feature table 12: 175 ns a particle at 100,000) */
  {T_KICKER, "ELEGANT_GPU_ENABLE_BUMPER", gpuKickerSupported, 175, 20, 0},
  /* IONEFFECTS: the ions on a host copy, the Gaussian field's kicks to the electrons on the device */
  {T_IONEFFECTS, "ELEGANT_GPU_ENABLE_IONEFFECTS", gpuAlwaysSupported, 150, 60, 0},
  /* SLICE: the times, the members of each slice and their sums in particle order (feature table 13: elegant
   * 376 ns a particle); a call that can't use them makes a read-only host copy (dump_slice_analysis()) */
  {T_SLICE_POINT, "ELEGANT_GPU_ENABLE_SLICE", gpuAlwaysSupported, 376, 200, 0},
};
#define GPU_PORT_TYPES ((long)(sizeof(gpuPortTypes) / sizeof(gpuPortTypes[0])))

static GPU_PORT_TYPE *gpuPortType(long type) {
  long i;

  for (i = 0; i < GPU_PORT_TYPES; i++)
    if (gpuPortTypes[i].type == type)
      return gpuPortTypes + i;
  return NULL;
}

/* Each type is on by default; ELEGANT_GPU_ENABLE_<TYPE>=0 keeps it on the CPU.  gpu-Pelegant keeps them
 * off: they were checked with the serial build. */
static void gpuPortReadSettings(void) {
  long i;

  for (i = 0; i < GPU_PORT_TYPES; i++) {
#if USE_MPI
    gpuPortTypes[i].enabled = 0;
#else
    gpuPortTypes[i].enabled = !gpuEnvSet(gpuPortTypes[i].setting) || gpuEnvFlag(gpuPortTypes[i].setting);
#endif
  }
}

static void gpuPortDescribeSettings(void) {
  long i, any = 0;

  for (i = 0; i < GPU_PORT_TYPES; i++)
    if (gpuPortTypes[i].enabled) {
      if (!any)
        fprintf(stderr, "elegant CUDA:");
      fprintf(stderr, " %s", entity_name[gpuPortTypes[i].type]);
      any = 1;
    }
  if (any)
    fprintf(stderr, " enabled with the magnet threshold.\n");
}

/* gpuElementEligible() for these types: 1 with *eligible set when eptr is one of them */
static long gpuPortElementEligible(ELEMENT_LIST *eptr, long nParticles, long *eligible) {
  GPU_PORT_TYPE *port = eptr ? gpuPortType(eptr->type) : NULL;

  *eligible = 0;
  if (!port)
    return 0;
  *eligible = port->enabled && eptr->p_elem && port->supported(eptr) && gpuMagnetParticleCountAllowed(nParticles);
  return 1;
}

#if !USE_MPI
static void gpuPortSmallBeamCost(ELEMENT_LIST *eptr, double *cpuNs, double *deviceUs) {
  GPU_PORT_TYPE *port = gpuPortType(eptr->type);

  if (port) {
    *cpuNs += port->cpuNs;
    *deviceUs += port->deviceUs;
  }
}
#endif

/* The CPU's swap order: a particle lost at i trades rows with the last one not yet looked at, which is
 * then looked at in its place.  Gathers the rows on the device and returns the particles left. */
static long gpuPortSwapOrder(long nParticles, const char *label) {
  static unsigned char *flags = NULL;
  static long long *perm = NULL;
  static long capacity = 0;
  long i, top = nParticles - 1;
  int status;

  if (capacity < nParticles) {
    free(flags);
    free(perm);
    flags = (unsigned char *)malloc(nParticles * sizeof(*flags));
    perm = (long long *)malloc(nParticles * sizeof(*perm));
    if (!flags || !perm)
      gpuRequiredFailure("unable to allocate the loss order");
    capacity = nParticles;
  }
  status = gpuCudaPortFlags(flags, nParticles);
  if (status != 0)
    gpuFatalStatus(label, status);
  for (i = 0; i < nParticles; i++)
    perm[i] = i;
  i = 0;
  while (i <= top) {
    if (!flags[i])
      i++;
    else {
      long long row = perm[i];
      unsigned char flag = flags[i];
      perm[i] = perm[top];
      perm[top] = row;
      flags[i] = flags[top];
      flags[top] = flag;
      top--;
    }
  }
  status = gpuCudaPortPermute(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, perm);
  if (status != 0)
    gpuFatalStatus(label, status);
  return top + 1;
}

static void gpuPortCounters(unsigned long long *counters, const char *label) {
  int status = gpuCudaPortCounters(counters);

  if (status != 0)
    gpuFatalStatus(label, status);
}

/* The survivors the last of these kernels left with an x or y that is not finite */
static long gpuPortInvalidSurvivors(void) {
  unsigned long long counters[4];

  gpuPortCounters(counters, "invalid coordinates count");
  return counters[3] != 0;
}

/* After a kernel that may have flagged losses: the CPU's row order, the lost rows to the host when the
 * run records them, and the particles left */
static long gpuPortFinishLosses(long nParticles, unsigned long long lost, const char *label) {
  long nLeft = nParticles;

  if (lost)
    nLeft = gpuPortSwapOrder(nParticles, label);
  gpuMarkDeviceChanged(nLeft);
  if (nLeft < nParticles)
    gpuCopyApertureLossRowsToHost(nLeft, nParticles, label);
  return nLeft;
}

#ifdef GPU_VERIFY
static void gpuPortVerifyLosses(long gpuLeft, long cpuLeft, long nParticles, const char *label) {
  compareGpuCpuApertureResult(gpuLeft, cpuLeft, nParticles, 0, label);
  /* the order of the lost rows, and their z and momentum, too */
  if (gpuLeft < nParticles)
    compareGpuCpu(nParticles, label);
}
#endif

/* ---------------------------------------------------------------------------------------------------- */
/* KPOLY */

static long gpuKpolyOnCpu(long nParticles, KPOLY *kpoly, double Po, double **accepted, double zStart,
                          const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long nLeft;

  gpuBase.elementOnGpu = 0;
  nLeft = polynomial_kicks(coord, nParticles, kpoly, 0.0, Po, accepted, zStart);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return nLeft;
}

long gpu_track_kpoly(double **particle, long nParticles, void *element, double Po, double **accepted,
                     double zStart) {
  KPOLY *kpoly = (KPOLY *)element;
  GPU_KPOLY_DATA data;
  unsigned long long counters[4];
  float milliseconds = 0;
  long nLeft;
  int status;

  (void)particle;
  if (nParticles <= 0)
    return nParticles;
  if (!kpoly)
    gpuRequiredFailure("NULL KPOLY CUDA tracking data");
  if (accepted || globalLossCoordOffset > 0)
    return gpuKpolyOnCpu(nParticles, kpoly, Po, accepted, zStart, "KPOLY with an accepted beam or global loss coordinates");
  memset(&data, 0, sizeof(data));
  data.yplane = kpoly->plane && (kpoly->plane[0] == 'y' || kpoly->plane[0] == 'Y');
  data.KL0 = kpoly->coefficient * kpoly->factor;
  data.cosTilt = cos(kpoly->tilt);
  data.sinTilt = sin(kpoly->tilt);
  data.dx = kpoly->dx;
  data.dy = kpoly->dy;
  data.dz = kpoly->dz;
  data.order = kpoly->order;
  data.coordLimit = coordLimit;
  data.slopeLimit = slopeLimit;
  data.zStart = zStart;
  data.Po = Po;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaKpoly(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("KPOLY CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPortCounters(counters, "KPOLY counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  nLeft = gpuPortFinishLosses(nParticles, counters[0], "KPOLY loss order");
#ifdef GPU_VERIFY
  startCpuTimer();
  gpuPortVerifyLosses(nLeft, polynomial_kicks(gpuBase.coord, nParticles, kpoly, 0.0, Po, accepted, zStart),
                      nParticles, "polynomial_kicks");
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* ---------------------------------------------------------------------------------------------------- */
/* HKPOLY */

static long gpuHkpolyOnCpu(long nParticles, HKPOLY *hkpoly, double Po, double **accepted, double zStart,
                           const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long nLeft;

  gpuBase.elementOnGpu = 0;
  nLeft = polynomial_hamiltonian(coord, nParticles, hkpoly, 0.0, Po, accepted, zStart);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return nLeft;
}

#define GPU_UNDEFINED_SLOPES_WARNING "Particle acquired undefined slopes when integrating through kick multipole."

long gpu_track_hkpoly(double **particle, long nParticles, void *element, double Po, double **accepted,
                      double zStart) {
  HKPOLY *hkpoly = (HKPOLY *)element;
  static GPU_HKPOLY_DATA data;
  unsigned long long counters[4], i;
  float milliseconds = 0;
  long nLeft, ix, iy, iqx, iqy, idelta;
  int status;

  (void)particle;
  if (!hkpoly)
    gpuRequiredFailure("NULL HKPOLY CUDA tracking data");
  if (nParticles <= 0)
    return nParticles;
  if (accepted || globalLossCoordOffset > 0)
    return gpuHkpolyOnCpu(nParticles, hkpoly, Po, accepted, zStart,
                          "HKPOLY with an accepted beam or global loss coordinates");
  memset(&data, 0, sizeof(data));
  memcpy(data.K, hkpoly->K, sizeof(data.K));
  memcpy(data.D, hkpoly->D, sizeof(data.D));
  memcpy(data.E, hkpoly->E, sizeof(data.E));
  data.cosTilt = cos(hkpoly->tilt);
  data.sinTilt = sin(hkpoly->tilt);
  data.dx = hkpoly->dx;
  data.dy = hkpoly->dy;
  data.dz = hkpoly->dz;
  data.factor = hkpoly->factor / hkpoly->nRepeats;
  data.nRepeats = hkpoly->nRepeats;
  data.driftType = hkpoly->driftType;
  data.coordLimit = coordLimit;
  data.slopeLimit = slopeLimit;
  data.zStart = zStart;
  data.Po = Po;
  data.expandHamiltonian = expandHamiltonian;
  /* polynomial_hamiltonian()'s highest powers */
  data.ixMax = data.iyMax = -1;
  for (ix = 0; ix < 7; ix++)
    for (iy = 0; iy < 7; iy++)
      if (hkpoly->K[ix][iy] != 0) {
        if (ix > data.ixMax)
          data.ixMax = ix;
        if (iy > data.iyMax)
          data.iyMax = iy;
      }
  data.iqxMax = data.iqyMax = data.ideltaMax = -1;
  for (iqx = 0; iqx < 7; iqx++)
    for (iqy = 0; iqy < 7; iqy++) {
      if (hkpoly->driftType == 1) {
        if (hkpoly->D[iqx][iqy] != 0) {
          if (iqx > data.iqxMax)
            data.iqxMax = iqx;
          if (iqy > data.iqyMax)
            data.iqyMax = iqy;
        }
      } else
        for (idelta = 0; idelta < 7; idelta++)
          if (hkpoly->E[iqx][iqy][idelta] != 0) {
            if (iqx > data.iqxMax)
              data.iqxMax = iqx;
            if (iqy > data.iqyMax)
              data.iqyMax = iqy;
            if (idelta > data.ideltaMax)
              data.ideltaMax = idelta;
          }
    }

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaHkpoly(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("HKPOLY CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPortCounters(counters, "HKPOLY counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  nLeft = gpuPortFinishLosses(nParticles, counters[0], "HKPOLY loss order");
#ifdef GPU_VERIFY
  startCpuTimer();
  gpuPortVerifyLosses(nLeft, polynomial_hamiltonian(gpuBase.coord, nParticles, hkpoly, 0.0, Po, accepted, zStart),
                      nParticles, "polynomial_hamiltonian");
#else
  for (i = 0; i < counters[1]; i++)
    printWarningForTracking(GPU_UNDEFINED_SLOPES_WARNING, NULL);
#endif
  (void)i;
  gpuRecordWallSeconds();
  return nLeft;
}

/* ---------------------------------------------------------------------------------------------------- */
/* MBUMPER */

/* rotateBeamCoordinatesForMisalignment()'s cosine and sine; 0 when it leaves the beam alone */
static long gpuPortRotation(double angle, double *cosA, double *sinA) {
  if (!angle || fabs(fabs(angle) - PIx2) < 1e-12)
    return 0;
  if (fabs(fabs(angle) - PI) < 1e-12) {
    *cosA = -1;
    *sinA = 0;
  } else if (fabs(angle - PIo2) < 1e-12) {
    *cosA = 0;
    *sinA = 1;
  } else if (fabs(angle + PIo2) < 1e-12) {
    *cosA = 0;
    *sinA = -1;
  } else {
    *cosA = cos(angle);
    *sinA = sin(angle);
  }
  return 1;
}

/* 1 when the device tracked the call; 0 when track_through_mkicker() must (settings it stops on).  Its
 * set-up reads the waveform file on the host, and the fiducial time is its sum, in particle order, over a
 * read-only copy of the beam. */
long gpu_track_mbumper(double **particle, long nParticles, void *element, double pCentral, long pass,
                       long defaultOrder) {
  MKICKER *kicker = (MKICKER *)element;
  GPU_MBUMPER_DATA data;
  unsigned long long counters[4], i;
  double timeOffset;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return 1;
  if (!kicker)
    gpuRequiredFailure("NULL MBUMPER CUDA tracking data");
  if (!kicker->t_wf)
    set_up_mkicker(kicker);
  if (kicker->order < 1)
    return 0;
  if (kicker->fire_on_pass > pass) {
    /* track_particles() keeps the drift on the device */
    drift_beam(particle, nParticles, kicker->length, defaultOrder);
    return 1;
  }
  if (!kicker->t_wf || !kicker->amp_wf || !kicker->n_wf || kicker->tmin >= kicker->tmax)
    return 0;
  if (kicker->phase_reference == 0)
    kicker->phase_reference = unused_phase_reference();
  switch (get_phase_reference(&timeOffset, kicker->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!kicker->fiducial_seen) {
      /* the center of this bunch goes through at the desired phase: track_through_mkicker()'s sum */
      double **part = copyParticlesToCpuReadOnly("MBUMPER fiducial time");
      double t0 = 0;
      long ip;

      for (ip = 0; ip < nParticles; ip++)
        t0 += part[ip][4] / (c_mks * beta_from_delta(pCentral, part[ip][5]));
      t0 /= nParticles;
      t0 += trackingClockOffset();
      kicker->t_fiducial = t0;
      kicker->fiducial_seen = 1;
    }
    set_phase_reference(kicker->phase_reference, timeOffset = kicker->t_fiducial);
    break;
  default:
    return 0;
  }
  if (kicker->length <= 0)
    return 1;
  timeOffset += kicker->time_offset;

  memset(&data, 0, sizeof(data));
  data.nWaveform = kicker->n_wf;
  data.tmin = kicker->tmin;
  data.tmax = kicker->tmax;
  data.strength = kicker->strength;
  data.length = kicker->length;
  data.pCentral = pCentral;
  data.cMks = c_mks;
  data.clockOffset = trackingClockOffset();
  data.timeOffset = timeOffset;
  data.doOffset = kicker->dx || kicker->dy || kicker->dz;
  data.dx = kicker->dx;
  data.dy = kicker->dy;
  data.dz = kicker->dz;
  data.doTilt = kicker->tilt != 0 && gpuPortRotation(kicker->tilt, &data.cosTilt, &data.sinTilt);
  if (data.doTilt)
    gpuPortRotation(-kicker->tilt, &data.cosUntilt, &data.sinUntilt);
  data.coordLimit = coordLimit;
  data.slopeLimit = slopeLimit;
  data.order = kicker->order;
  if ((data.nParts = ceil(kicker->n_kicks / 4.0)) < 1)
    data.nParts = 1;
  data.periodic = kicker->periodic ? 1 : 0;
  data.expandHamiltonian = expandHamiltonian;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaMbumper(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, kicker->t_wf,
                          kicker->amp_wf, &milliseconds);
  if (status != 0)
    gpuFatalStatus("MBUMPER CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPortCounters(counters, "MBUMPER counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  if (counters[2] != GPU_PORT_NO_ERROR) {
    double time = 0;
    status = gpuCudaPortValue((long)counters[2], &time);
    if (status != 0)
      gpuFatalStatus("MBUMPER error time", status);
    printf("error: waveform interpolation problem in track_through_mkicker()\n");
    fflush(stdout);
    printf("particle time is %21.15e\nwaveform limits are %21.15e to %21.15e\n", time, kicker->tmin, kicker->tmax);
    fflush(stdout);
    exitElegant(1);
  }
  gpuMarkDeviceChanged(nParticles);
#ifdef GPU_VERIFY
  startCpuTimer();
  track_through_mkicker(gpuBase.coord, nParticles, kicker, pCentral, pass, defaultOrder);
  compareGpuCpu(nParticles, "track_through_mkicker");
#else
  for (i = 0; i < counters[1]; i++)
    printWarningForTracking(GPU_UNDEFINED_SLOPES_WARNING, NULL);
#endif
  (void)i;
  gpuRecordWallSeconds();
  return 1;
}

/* ---------------------------------------------------------------------------------------------------- */
/* The RF deflectors.  Their set-up and fiducial time are the CPU's own routines, given a read-only copy of
 * the beam on the call that needs it; the particle loops run on the device. */

extern int gpuCudaRmdf(void *coord, long nParticles, int stride, const GPU_RMDF_DATA *data, float *milliseconds);
extern int gpuCudaMrfdf(void *coord, long nParticles, int stride, const GPU_MRFDF_DATA *data, float *milliseconds);
extern int gpuCudaShrfdf(void *coord, long nParticles, int stride, const GPU_SHRFDF_DATA *data,
                         float *milliseconds);
extern int gpuCudaRftm110(void *coord, long nParticles, int stride, const GPU_RFTM110_DATA *data,
                          float *milliseconds);

/* The beam for a set-up routine that reads it only when the fiducial is not yet seen */
static double **gpuPortSetupBeam(double **particle, long fiducialSeen, const char *reason) {
  return fiducialSeen ? particle : copyParticlesToCpuReadOnly(reason);
}

static void gpuPortTilt(double tilt, int *doTilt, double *cosTilt, double *sinTilt, double *cosUntilt,
                        double *sinUntilt) {
  *doTilt = tilt != 0 && gpuPortRotation(tilt, cosTilt, sinTilt);
  if (*doTilt)
    gpuPortRotation(-tilt, cosUntilt, sinUntilt);
}

/* RMDF; 1 when tracked */
long gpu_track_rmdf(double **particle, long nParticles, void *element, double pCentral) {
  RMDF *rmdf = (RMDF *)element;
  GPU_RMDF_DATA data;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return 1;
  if (!rmdf)
    gpuRequiredFailure("NULL RMDF CUDA tracking data");
  if (!rmdf->initialized) {
    /* track_through_ramped_deflector()'s first call: the first particle's time */
    double **part = copyParticlesToCpuReadOnly("RMDF first particle");
    double pc, beta, beta_z;

    rmdf->initialized = 1;
    pc = pCentral * (1 + part[0][5]);
    beta = pc / sqrt(1 + sqr(pc));
    beta_z = beta / (1 + sqr(part[0][1]) + sqr(part[0][3]));
    rmdf->t_first_particle = (part[0][4] / beta + rmdf->length / (2 * beta_z)) / c_mks;
    if (rmdf->n_sections < 1)
      rmdf->n_sections = 10;
    rmdf->initialized = 1;
  }
  memset(&data, 0, sizeof(data));
  data.dx = rmdf->dx;
  data.dy = rmdf->dy;
  data.pcCentral = pCentral;
  data.cMks = c_mks;
  data.tNull = rmdf->t_first_particle + rmdf->time_offset;
  data.cosTilt = cos(rmdf->tilt);
  data.sinTilt = sin(rmdf->tilt);
  data.Estrength = -particleCharge * rmdf->voltage / (particleMass * pow3(c_mks) * rmdf->ramp_time * rmdf->gap);
  data.length = rmdf->length / rmdf->n_sections;
  data.nSections = rmdf->n_sections;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaRmdf(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RMDF CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  startCpuTimer();
  track_through_ramped_deflector(gpuBase.coord, rmdf, gpuBase.coord, nParticles, pCentral);
  compareGpuCpu(nParticles, "track_through_ramped_deflector");
#endif
  (void)particle;
  gpuRecordWallSeconds();
  return 1;
}

/* MRFDF; 1 when tracked */
long gpu_track_mrfdf(double **particle, long nParticles, void *element, double pCentral, long pass) {
  MRFDF *rf = (MRFDF *)element;
  GPU_MRFDF_DATA data;
  float milliseconds = 0;
  int status;

  if (!rf)
    gpuRequiredFailure("NULL MRFDF CUDA tracking data");
  if (nParticles <= 0 || rf->factor == 0)
    return 1;
  if (!rf->initialized || !rf->fiducial_seen)
    set_up_mrfdf(rf, gpuPortSetupBeam(particle, rf->fiducial_seen, "MRFDF fiducial time"), nParticles, pCentral);
  if ((rf->startPass >= 0 && pass < rf->startPass) || (rf->endPass >= 0 && pass > rf->endPass))
    return 1;
  memset(&data, 0, sizeof(data));
  memcpy(data.a, rf->a, sizeof(data.a));
  memcpy(data.b, rf->b, sizeof(data.b));
  memcpy(data.frequency, rf->frequency, sizeof(data.frequency));
  memcpy(data.phase, rf->phase, sizeof(data.phase));
  data.factor = rf->factor;
  data.charge = particleCharge;
  data.massC2 = particleMass * sqr(c_mks);
  data.pcCentral = pCentral;
  data.cMks = c_mks;
  data.tFirst = rf->t_first_particle;
  data.pi = PI;
  data.startPID = rf->startPID;
  data.endPID = rf->endPID;
  gpuPortTilt(rf->tilt, &data.doTilt, &data.cosTilt, &data.sinTilt, &data.cosUntilt, &data.sinUntilt);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaMrfdf(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("MRFDF CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  startCpuTimer();
  track_through_multipole_deflector(gpuBase.coord, rf, gpuBase.coord, nParticles, pCentral, pass);
  compareGpuCpu(nParticles, "track_through_multipole_deflector");
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* SHRFDF; 1 when tracked */
long gpu_track_shrfdf(double **particle, long nParticles, void *element, double pCentral) {
  SHRFDF *rf = (SHRFDF *)element;
  GPU_SHRFDF_DATA data;
  float milliseconds = 0;
  int status;

  if (!rf)
    gpuRequiredFailure("NULL SHRFDF CUDA tracking data");
  if (nParticles <= 0 || rf->factor == 0)
    return 1;
  if (!rf->initialized || !rf->fiducial_seen)
    set_up_shrfdf(rf, gpuPortSetupBeam(particle, rf->fiducial_seen, "SHRFDF fiducial time"), nParticles, pCentral);
  memset(&data, 0, sizeof(data));
  memcpy(data.v, rf->v, sizeof(data.v));
  memcpy(data.phase, rf->phase, sizeof(data.phase));
  data.periodPhase = rf->period_phase;
  data.periodLength = rf->period_length;
  data.factor = rf->factor;
  data.charge = particleCharge;
  data.massC2 = particleMass * sqr(c_mks);
  data.pcCentral = pCentral;
  data.cMks = c_mks;
  data.tFirst = rf->t_first_particle;
  data.pi = PI;
  gpuPortTilt(rf->tilt, &data.doTilt, &data.cosTilt, &data.sinTilt, &data.cosUntilt, &data.sinUntilt);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaShrfdf(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("SHRFDF CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  startCpuTimer();
  track_through_space_harmonic_deflector(gpuBase.coord, rf, gpuBase.coord, nParticles, pCentral);
  compareGpuCpu(nParticles, "track_through_space_harmonic_deflector");
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* RFTM110; 1 when tracked.  The central particle's time, the voltage waveform and the noise (random_3) are
 * track_through_rftm110_deflector()'s statements on the host. */
long gpu_track_rftm110(double **particle, long nParticles, void *element, double pCentral, double lCentral,
                       double zEnd, long pass) {
  RFTM110 *rf = (RFTM110 *)element;
  GPU_RFTM110_DATA data;
  double gamma, beta, t0, voltTimes2, omega, k, t_first, phase0;
  float milliseconds = 0;
  long i_volt;
  int status;
#ifdef GPU_VERIFY
  RFTM110 before;
#endif

  (void)zEnd;
  if (!rf)
    gpuRequiredFailure("NULL RFTM110 CUDA tracking data");
  if (rf->voltage == 0 || (rf->startPass >= 0 && pass < rf->startPass) || (rf->endPass >= 0 && pass > rf->endPass))
    return 1;
  if (nParticles <= 0)
    return 0;
  if (!rf->initialized || !rf->fiducial_seen)
    set_up_rftm110(rf, gpuPortSetupBeam(particle, rf->fiducial_seen, "RFTM110 fiducial time"), nParticles,
                   pCentral);
#ifdef GPU_VERIFY
  before = *rf;
#endif
  gamma = sqrt(sqr(pCentral) + 1);
  beta = pCentral / gamma;
  if (pass == 0) {
    rf->Ts = rf->t_first_particle;
    if (rf->alignWaveforms)
      rf->Ts = 0;
  } else
    rf->Ts += lCentral / (beta * c_mks);
  t0 = rf->Ts;
  if (rf->n_Vpts) {
    if (rf->voltageIsPeriodic && t0 > rf->V_tFinal)
      t0 = fmod(t0 - rf->V_tInitial, rf->V_tFinal - rf->V_tInitial) + rf->V_tInitial;
    i_volt = find_nearby_array_entry(rf->t_Vf, rf->n_Vpts, t0);
    voltTimes2 = linear_interpolation(rf->Vfactor, rf->t_Vf, rf->n_Vpts, t0, i_volt);
    if (voltTimes2 == 0)
      return 1;
  } else
    voltTimes2 = 1;
  voltTimes2 *= 2 * rf->voltage / (1e6 * particleMassMV * particleRelSign) *
                (gauss_rn_lim(1.0, rf->voltageNoise, 2, random_3) +
                 (rf->voltageNoiseGroup ? rf->groupVoltageNoise * GetNoiseGroupValue(rf->voltageNoiseGroup) : 0));
  omega = 2 * PI * rf->frequency;
  k = omega / c_mks;
  t_first = rf->t_first_particle;
  phase0 = (rf->phase + gauss_rn_lim(0.0, rf->phaseNoise, 2, random_3) +
            (rf->phaseNoiseGroup ? rf->groupPhaseNoise * GetNoiseGroupValue(rf->phaseNoiseGroup) : 0)) *
             PI / 180.0 -
           omega * t_first;

  memset(&data, 0, sizeof(data));
  data.voltTimes2 = voltTimes2;
  data.omega = omega;
  data.k = k;
  data.phase0 = phase0;
  data.pcCentral = pCentral;
  data.cMks = c_mks;
  gpuPortTilt(rf->tilt, &data.doTilt, &data.cosTilt, &data.sinTilt, &data.cosUntilt, &data.sinUntilt);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaRftm110(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFTM110 CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  /* the CPU's rerun advances the central time from where it was; no noise here (gpuRftm110Supported()) */
  *rf = before;
  startCpuTimer();
  track_through_rftm110_deflector(gpuBase.coord, rf, gpuBase.coord, nParticles, pCentral, lCentral, zEnd, pass);
  compareGpuCpu(nParticles, "track_through_rftm110_deflector");
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* ---------------------------------------------------------------------------------------------------- */
/* ALPH: alpha_magnet_tracking().  Its matrix goes through track_particles(), which puts it on the device;
 * the puck and slit filters are a kernel each, with the CPU's swap order. */

extern int gpuCudaAlphFilter(void *coord, long nParticles, int stride, const GPU_ALPH_FILTER_DATA *data,
                             float *milliseconds);

#ifdef GPU_VERIFY
/* alpha_magnet_tracking()'s filter loop, for the verify build's comparison */
static long gpuAlphFilterCpu(double **particle, long n_part, double **accepted, const GPU_ALPH_FILTER_DATA *d) {
  long ip, itop = n_part - 1;
  double *coord;

  for (ip = 0; ip < n_part; ip++) {
    coord = particle[ip];
    if (d->slits ? ((d->doXl && coord[0] <= d->xl) || (d->doXu && coord[0] >= d->xu))
                 : (coord[0] >= d->xl && coord[0] <= d->xu)) {
      if (accepted)
        swapParticles(accepted[ip], accepted[itop]);
      swapParticles(particle[ip], particle[itop]);
      particle[itop][4] = d->z;
      particle[itop][5] = d->Po * (1 + particle[itop][5]);
      itop--;
      ip--;
      n_part--;
    }
  }
  return n_part;
}
#endif

static long gpuAlphFilter(long nParticles, GPU_ALPH_FILTER_DATA *data) {
  unsigned long long counters[4];
  float milliseconds = 0;
  long nLeft;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaAlphFilter(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("ALPH filter CUDA kernel", status);
  gpuRecordApertureKernel(milliseconds);
  gpuPortCounters(counters, "ALPH filter counters");
  nLeft = counters[0] ? gpuPortSwapOrder(nParticles, "ALPH loss order") : nParticles;
  gpuMarkDeviceChanged(nLeft);
#ifdef GPU_VERIFY
  gpuPortVerifyLosses(nLeft, gpuAlphFilterCpu(gpuBase.coord, nParticles, NULL, data), nParticles,
                      "alpha_magnet_tracking filter");
#endif
  return nLeft;
}

long gpu_track_alph(double **particle, void *matrix, void *element, long n_part, double **accepted,
                    double P_central, double z) {
  ALPH *alpha = (ALPH *)element;
  VMATRIX *M = (VMATRIX *)matrix;
  GPU_ALPH_FILTER_DATA data;
  float xmax, xl = 0.0, xu = 0.0;
  long do_xl, do_xu, nInitial = n_part;

  if (!alpha || !M)
    gpuRequiredFailure("NULL ALPH CUDA tracking data");
  if (accepted || globalLossCoordOffset > 0) {
    double **coord = forceParticlesToCpu("ALPH with an accepted beam or global loss coordinates");
    long nLeft;

    gpuBase.elementOnGpu = 0;
    nLeft = alpha_magnet_tracking(coord, M, alpha, n_part, accepted, P_central, z);
    gpuMarkHostWillChange();
    gpuRecordWallSeconds();
    return nLeft;
  }
  if (alpha->part == 0) {
    track_particles(particle, M, particle, n_part);
    return n_part;
  }
  if (alpha->part == 1)
    track_particles(particle, M, particle, n_part);
  memset(&data, 0, sizeof(data));
  data.z = z;
  data.Po = P_central;
  if (alpha->xPuck > 0) {
    if ((xl = alpha->xPuck - alpha->widthPuck) < 0)
      xl = 0;
    if ((xu = alpha->xPuck) < 0)
      xu = 0;
    if (xu > xl) {
      xmax = ALPHA_CONST * sqrt(P_central / alpha->gradient);
      xl = xmax - xl;
      xu = xmax - xu;
      SWAP_DOUBLE(xl, xu);
      data.slits = 0;
      data.xl = xl;
      data.xu = xu;
      startGpuTimer();
      n_part = gpuAlphFilter(n_part, &data);
    }
  }
  if (alpha->xs1 || alpha->xs2 || alpha->dp1 != -1 || alpha->dp2 != 1) {
    do_xu = do_xl = 1;
    xmax = ALPHA_CONST * sqrt(P_central / alpha->gradient);
    if (alpha->dp1 != -1 || alpha->dp2 != 1) {
      if (alpha->dp1 == -1)
        do_xu = 0;
      else
        xu = xmax - ALPHA_CONST * sqrt((1 + alpha->dp1) * P_central / alpha->gradient);
      if (alpha->dp2 == 1)
        do_xl = 0;
      else
        xl = xmax - ALPHA_CONST * sqrt((1 + alpha->dp2) * P_central / alpha->gradient);
    } else {
      if (alpha->xs2)
        xl = xmax - alpha->xs2;
      else
        do_xl = 0;
      if (alpha->xs1)
        xu = xmax - alpha->xs1;
      else
        do_xu = 0;
    }
    data.slits = 1;
    data.doXl = do_xl;
    data.doXu = do_xu;
    data.xl = xl;
    data.xu = xu;
    startGpuTimer();
    n_part = gpuAlphFilter(n_part, &data);
  }
  if (alpha->part == 2)
    track_particles(particle, M, particle, n_part);
  if (n_part < nInitial)
    gpuCopyApertureLossRowsToHost(n_part, nInitial, "ALPH lost particles");
  gpuRecordWallSeconds();
  return n_part;
}

/* ---------------------------------------------------------------------------------------------------- */
/* KSBEND: track_through_kick_sbend()'s set-up on the host, statement for statement */

extern int gpuCudaKsbend(void *coord, long nParticles, int stride, const GPU_KSBEND_DATA *data, float *milliseconds);

static long gpuKsbendOnCpu(long nParticles, KSBEND *ksbend, double Po, double **accepted, double zStart,
                           const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long nLeft;

  gpuBase.elementOnGpu = 0;
  nLeft = track_through_kick_sbend(coord, nParticles, ksbend, 0.0, Po, accepted, zStart);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return nLeft;
}

long gpu_track_ksbend(double **particle, long nParticles, void *element, double Po, double **accepted,
                      double zStart) {
  KSBEND *ksbend = (KSBEND *)element;
  static char *methods[2] = {"leap-frog", "modified-midpoint"};
  GPU_KSBEND_DATA d;
  unsigned long long counters[4];
  double n, beta, gamma, delta, fse, tilt, etilt, ttilt, angle, e1, e2, Kg, rho0, h, h2, h3, nh, betah2,
    gammah3, deltah4;
  float milliseconds = 0;
  long nLeft;
  int status;

  (void)particle;
  if (!ksbend)
    gpuRequiredFailure("NULL KSBEND CUDA tracking data");
  if (nParticles <= 0)
    return nParticles;
  if (accepted || globalLossCoordOffset > 0)
    return gpuKsbendOnCpu(nParticles, ksbend, Po, accepted, zStart,
                          "KSBEND with an accepted beam or global loss coordinates");
  memset(&d, 0, sizeof(d));
  d.modifiedMidpoint = match_string(ksbend->method, methods, 2, 0) == 1;
  d.paraxial = ksbend->paraxial ? 1 : 0;
  if (ksbend->angle < 0) {
    angle = -ksbend->angle;
    e1 = -ksbend->e[ksbend->e1Index];
    e2 = -ksbend->e[ksbend->e2Index];
    etilt = ksbend->etilt;
    tilt = ksbend->tilt + PI;
    rho0 = -ksbend->length / angle;
    n = -sqr(rho0) * ksbend->k1;
    beta = 0.5 * ksbend->k2 * pow3(rho0);
    gamma = ksbend->k3 * pow4(rho0) / 6.;
    delta = ksbend->k4 * pow5(rho0) / 24.;
    rho0 = -rho0;
  } else {
    angle = ksbend->angle;
    e1 = ksbend->e[ksbend->e1Index];
    e2 = ksbend->e[ksbend->e2Index];
    etilt = ksbend->etilt;
    tilt = ksbend->tilt;
    rho0 = ksbend->length / angle;
    n = -sqr(rho0) * ksbend->k1;
    beta = 0.5 * ksbend->k2 * pow3(rho0);
    gamma = ksbend->k3 * pow4(rho0) / 6.;
    delta = ksbend->k4 * pow5(rho0) / 24.;
  }
  fse = ksbend->fse;
  h = 1. / rho0;
  h2 = sqr(h);
  h3 = h * h2;
  nh = n * h;
  betah2 = beta * h2;
  gammah3 = gamma * h3;
  deltah4 = delta * h2 * h2;
  d.h = h;
  d.ha = (1 + fse) * h;
  d.rho0 = rho0;
  d.n = n;
  d.length = ksbend->length;
  Kg = 2 * ksbend->hgap * ksbend->fint;
  d.psi1 = Kg / rho0 / cos(e1) * (1 + sqr(sin(e1)));
  d.psi2 = Kg / rho0 / cos(e2) * (1 + sqr(sin(e2)));
  if (ksbend->synch_rad)
    d.radCoef = sqr(particleCharge) * pow3(Po) * sqr(1 + fse) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho0));
  else
    d.radCoef = 0;
  d.Fy_0 = 1;
  d.Fy_x = -nh;
  if (ksbend->nonlinear) {
    d.Fy_x2 = betah2;
    d.Fy_x3 = gammah3;
    d.Fy_y2 = (h * nh - 2 * betah2) / 2;
    d.Fy_x_y2 = -(2 * h * betah2 + nh * h2 + 6 * gammah3) / 2;
    d.Fy_x4 = deltah4;
    d.Fy_x2_y2 = -(3 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 12 * deltah4) / 2;
    d.Fy_y4 = (12 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 24 * deltah4) / 24;
  }
  d.Fx_y = -nh;
  if (ksbend->nonlinear) {
    d.Fx_x_y = 2 * betah2;
    d.Fx_x2_y = 3 * gammah3;
    d.Fx_y3 = -(2 * h * betah2 + nh * h2 + 6 * gammah3) / 6;
    d.Fx_x3_y = 4 * deltah4;
    d.Fx_x_y3 = -(3 * h * gammah3 - h3 * nh - 2 * h2 * betah2 + 12 * deltah4) / 3;
  }
  ttilt = tilt + etilt;
  if (ttilt == 0) {
    d.cosTtilt = 1;
    d.sinTtilt = 0;
  } else if (fabs(fabs(ttilt) - PI) < 1e-12) {
    d.cosTtilt = -1;
    d.sinTtilt = 0;
  } else if (fabs(ttilt - PIo2) < 1e-12) {
    d.cosTtilt = 0;
    d.sinTtilt = 1;
  } else if (fabs(ttilt + PIo2) < 1e-12) {
    d.cosTtilt = 0;
    d.sinTtilt = -1;
  } else {
    d.cosTtilt = cos(ttilt);
    d.sinTtilt = sin(ttilt);
  }
  if (etilt) {
    double q1a, q2a, q3a, q1b, q2b, q3b, qp1, qp2, qp3, dz, tan_alpha, k;

    q1a = (1 - cos(angle)) * rho0 * (cos(etilt) - 1);
    q2a = 0;
    q3a = (1 - cos(angle)) * rho0 * sin(etilt);
    qp1 = sin(angle) * cos(etilt);
    qp2 = cos(angle);
    k = sqrt(sqr(qp1) + sqr(qp2));
    qp1 /= k;
    qp2 /= k;
    qp3 = sin(angle) * sin(etilt) / k;
    tan_alpha = 1. / tan(angle) / cos(etilt);
    q1b = q1a * tan_alpha / (tan(angle) + tan_alpha);
    q2b = -q1b * tan(angle);
    dz = sqrt(sqr(q1b - q1a) + sqr(q2b - q2a));
    q3b = q3a + qp3 * dz;
    d.dcoordEtilt[0] = sqrt(sqr(q1b) + sqr(q2b));
    d.dcoordEtilt[1] = tan(atan(tan_alpha) - (PIo2 - angle));
    d.dcoordEtilt[2] = q3b;
    d.dcoordEtilt[3] = qp3;
    d.dcoordEtilt[4] = dz * sqrt(1 + sqr(qp3));
    d.dcoordEtilt[5] = 0;
    rotateCoordinatesForMisalignment(d.dcoordEtilt, tilt);
  }
  d.dxi = -ksbend->dx;
  d.dzi = ksbend->dz;
  d.dyi = -ksbend->dy;
  d.dxf = ksbend->dx * cos(ksbend->angle) + ksbend->dz * sin(ksbend->angle);
  d.dzf = ksbend->dx * sin(ksbend->angle) - ksbend->dz * cos(ksbend->angle);
  d.dyf = ksbend->dy;
  d.e1 = e1;
  d.e2 = e2;
  d.tanE1 = tan(e1);
  d.tanE2 = tan(e2);
  d.secE1 = 1. / cos(e1);
  d.secE2 = 1. / cos(e2);
  d.he1 = ksbend->h[ksbend->e1Index];
  d.he2 = ksbend->h[ksbend->e2Index];
  d.edge1 = ksbend->edge_effects[ksbend->e1Index] ? 1 : 0;
  d.edge2 = ksbend->edge_effects[ksbend->e2Index] ? 1 : 0;
  d.edgeOrder = (int)ksbend->edge_order;
  d.coordLimit = coordLimit;
  d.slopeLimit = slopeLimit;
  d.zStart = zStart;
  d.Po = Po;
  d.nKicks = ksbend->n_kicks;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaKsbend(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &d, &milliseconds);
  if (status != 0)
    gpuFatalStatus("KSBEND CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPortCounters(counters, "KSBEND counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  nLeft = gpuPortFinishLosses(nParticles, counters[0], "KSBEND loss order");
#ifdef GPU_VERIFY
  startCpuTimer();
  gpuPortVerifyLosses(nLeft, track_through_kick_sbend(gpuBase.coord, nParticles, ksbend, 0.0, Po, accepted, zStart),
                      nParticles, "track_through_kick_sbend");
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* ---------------------------------------------------------------------------------------------------- */
/* BOFFAXE: trackMagneticFieldOffAxisExpansion().  Its set-up (the file, the settings' checks) runs on the host
 * with no particles on the first call; the field table goes to the device once. */

extern int gpuCudaBoffaxe(void *coord, long nParticles, int stride, const GPU_BOFFAXE_DATA *data, double **fieldData,
                          const void *tableKey, float *milliseconds);

long gpu_track_boffaxe(double **particle, long nParticles, void *element, double pCentral) {
  BOFFAXE *boa = (BOFFAXE *)element;
  GPU_BOFFAXE_DATA d;
  double **fieldData, dataDz, tableZMin, tableZMax;
  long nz, nDz1;
  float milliseconds = 0;
  int status;

  (void)particle;
  if (!boa)
    gpuRequiredFailure("NULL BOFFAXE CUDA tracking data");
  if (nParticles <= 0)
    return 1;
  if (!boa->initialized)
    /* the first call's set-up (the file, its checks, the field output file) without particles, so that call
     * runs on the device too: with one pass every call is a first call */
    trackMagneticFieldOffAxisExpansion(particle, 0, boa, pCentral, NULL, NULL);
  if (!boffaxeFieldTable(boa, &nz, &nDz1, &dataDz, &tableZMin, &tableZMax, &fieldData))
    return 0;
  /* the settings the CPU stops on */
  if (boa->zInterval < 1 || boa->zSubdivisions < 1 || nz % boa->zInterval != 0 || boa->length < 0 ||
      (boa->fieldLength > 0 && fabs(boa->fieldLength - (tableZMax - tableZMin)) > 1e-6 * dataDz) || nDz1 > 16)
    return 0;
  memset(&d, 0, sizeof(d));
  d.nz = nz;
  d.nDz1 = nDz1;
  d.dataDz = dataDz;
  d.dz = dataDz * boa->zInterval / boa->zSubdivisions;
  d.zMin = -(tableZMax - tableZMin) / 2;
  d.zMax = +(tableZMax - tableZMin) / 2;
  d.zEntry = -boa->length / 2;
  d.zExit = boa->length / 2;
  d.pCentral = pCentral;
  d.q = particleCharge;
  d.relSign = particleRelSign;
  d.mass = particleMass;
  d.cMks = c_mks;
  if (boa->synchRad)
    d.radCoef = ipow4(particleCharge) / (6 * PI * epsilon_o * ipow4(c_mks) * ipow3(particleMass));
  d.strength = boa->strength;
  d.dx = boa->dx;
  d.dy = boa->dy;
  d.dz0 = boa->dz;
  d.zInterval = boa->zInterval;
  d.zSubdivisions = boa->zSubdivisions;
  d.order = boa->order;
  d.expansionOrder = boa->expansionOrder;
  d.synchRad = boa->synchRad ? 1 : 0;
  d.doOffset = boa->dx || boa->dy || boa->dz;
  gpuPortTilt(boa->tilt, &d.doTilt, &d.cosTilt, &d.sinTilt, &d.cosUntilt, &d.sinUntilt);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaBoffaxe(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &d, fieldData, fieldData,
                          &milliseconds);
  if (status != 0)
    gpuFatalStatus("BOFFAXE CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  startCpuTimer();
  trackMagneticFieldOffAxisExpansion(gpuBase.coord, nParticles, boa, pCentral, NULL, NULL);
  compareGpuCpu(nParticles, "trackMagneticFieldOffAxisExpansion");
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* ---------------------------------------------------------------------------------------------------- */
/* GFWIGGLER (APPLE): APPLE_Track(), with InitializeAPPLE() on the host and its harmonic tables on the
 * device */

extern int gpuCudaApple(void *coord, long nParticles, int stride, const GPU_APPLE_DATA *data, const double *hostTable,
                        long tableCount, const void *tableKey, float *milliseconds);

long gpu_track_apple(double **particle, long nParticles, void *element, double pCentral) {
  APPLE *apple = (APPLE *)element;
  GPU_APPLE_DATA d;
  static double *table = NULL;
  static long tableCapacity = 0;
  double **arrays[11];
  long nArrays = 11, count, i, j, k, offset, *offsets[11];
  float milliseconds = 0;
  int status;

  (void)particle;
  if (!apple)
    gpuRequiredFailure("NULL GFWIGGLER CUDA tracking data");
  if (nParticles <= 0)
    return 1;
  if (!apple->initialized)
    InitializeAPPLE(apple->Input, apple);
  if (apple->End_Pole && apple->periods < 2) {
    /* APPLE_Track() does this at its first particle */
    apple->End_Pole = 0;
    printWarningForTracking("PERIODS<2, setting END_POLE=0", NULL);
  }
  memset(&d, 0, sizeof(d));
  d.NxHarm = apple->NxHarm;
  d.NzHarm = apple->NzHarm;
  d.ShimHarm = apple->shimOn ? apple->ShimHarm : 0;
  arrays[0] = apple->kx;
  arrays[1] = apple->ky;
  arrays[2] = apple->CoZ;
  arrays[3] = apple->CxXoZ;
  arrays[4] = apple->CxYoZ;
  arrays[5] = apple->CxXoYZ;
  arrays[6] = apple->CxX2oYZ;
  arrays[7] = apple->CxX2oZ;
  arrays[8] = apple->CxXYoZ;
  arrays[9] = apple->CxX3oYZ;
  arrays[10] = apple->CxY2oZ;
  offsets[0] = &d.kx;
  offsets[1] = &d.ky;
  offsets[2] = &d.CoZ;
  offsets[3] = &d.CxXoZ;
  offsets[4] = &d.CxYoZ;
  offsets[5] = &d.CxXoYZ;
  offsets[6] = &d.CxX2oYZ;
  offsets[7] = &d.CxX2oZ;
  offsets[8] = &d.CxXYoZ;
  offsets[9] = &d.CxX3oYZ;
  offsets[10] = &d.CxY2oZ;
  count = nArrays * d.NxHarm * d.NzHarm + d.NzHarm + 2 * d.ShimHarm;
  if (count < 1)
    count = 1;
  if (tableCapacity < count) {
    free(table);
    if (!(table = (double *)calloc(count, sizeof(*table))))
      gpuRequiredFailure("unable to allocate the GFWIGGLER tables");
    tableCapacity = count;
  }
  offset = 0;
  for (k = 0; k < nArrays; k++) {
    *offsets[k] = offset;
    for (i = 0; i < d.NxHarm; i++)
      for (j = 0; j < d.NzHarm; j++)
        table[offset + i * d.NzHarm + j] = arrays[k] && arrays[k][i] ? arrays[k][i][j] : 0;
    offset += d.NxHarm * d.NzHarm;
  }
  d.kz = offset;
  for (j = 0; j < d.NzHarm; j++)
    table[offset + j] = apple->kz ? apple->kz[j] : 0;
  offset += d.NzHarm;
  d.Ci_shim = offset;
  d.Si_shim = offset + d.ShimHarm;
  for (i = 0; i < d.ShimHarm; i++) {
    table[d.Ci_shim + i] = apple->Ci_shim[i];
    table[d.Si_shim + i] = apple->Si_shim[i];
  }
  d.C1 = apple->C1;
  d.C2 = apple->C2;
  d.C3 = apple->C3;
  d.C4 = apple->C4;
  d.S1 = apple->S1;
  d.S2 = apple->S2;
  d.S3 = apple->S3;
  d.S4 = apple->S4;
  d.x0 = apple->x0;
  d.lz = apple->lz;
  d.kx_shim = apple->kx_shim;
  d.dgap = apple->dgap;
  d.shimScale = apple->shimScale;
  d.pCentral = pCentral;
  d.eomc = -particleCharge / particleMass / c_mks;
  d.dx = apple->dx;
  d.dy = apple->dy;
  d.dz = apple->dz;
  d.periods = apple->periods;
  d.step = apple->step;
  d.order = apple->order;
  d.endPole = apple->End_Pole ? 1 : 0;
  d.shimOn = apple->shimOn ? 1 : 0;
  d.doOffset = apple->dx || apple->dy || apple->dz;
  gpuPortTilt(apple->tilt, &d.doTilt, &d.cosTilt, &d.sinTilt, &d.cosUntilt, &d.sinUntilt);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  /* the tables go up again when another element's, or newly made arrays, are tracked */
  status = gpuCudaApple(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &d, table, count,
                        apple->kx ? (const void *)apple->kx : (const void *)apple, &milliseconds);
  if (status != 0)
    gpuFatalStatus("GFWIGGLER CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuPortInvalidPending = 1;
#ifdef GPU_VERIFY
  startCpuTimer();
  APPLE_Track(gpuBase.coord, nParticles, pCentral, apple);
  compareGpuCpu(nParticles, "APPLE_Track");
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* ---------------------------------------------------------------------------------------------------- */
/* Filters and beam adjustments.  Their sums are the CPU's, of columns copied to the host in particle
 * order; the device applies the result and removes particles in the CPU's swap order. */

extern int gpuCudaPortColumns(const void *coord, long nParticles, int stride, int nColumns, const int *codes, double Po,
                              double cMks, double *host);
extern int gpuCudaPortSetColumn(void *coord, long nParticles, int stride, int column, const double *host);
extern int gpuCudaPortMarkTail(void *coord, long first, long nParticles, int stride, double z, double Po);
extern int gpuCudaPfilter(void *coord, long nParticles, int stride, const GPU_PFILTER_DATA *data, float *milliseconds);
extern int gpuCudaClean(void *coord, long nParticles, int stride, const GPU_CLEAN_DATA *data, float *milliseconds);
extern int gpuCudaEmittance(void *coord, long nParticles, int stride, const GPU_EMITTANCE_DATA *data,
                            float *milliseconds);
extern int gpuCudaRemcor(void *coord, long nParticles, int stride, const GPU_REMCOR_DATA *data, float *milliseconds);
extern int gpuCudaMhistMinMax(const void *coord, long nParticles, int stride, double Po, double cMks, double *min,
                              double *max, double *time);
extern int gpuCudaMhistFill(const void *coord, long nParticles, int stride, double Po, double cMks, double t0,
                            int nBooks, const void *booksHost, long total, unsigned long long *counts);

/* nColumns columns of the first nParticles particles, column after column, in a buffer kept here */
static double *gpuPortHostColumns(long nParticles, int nColumns, const int *codes, double Po, const char *label) {
  static double *buffer = NULL;
  static long capacity = 0;
  int status;

  if (capacity < nParticles * nColumns) {
    free(buffer);
    if (!(buffer = (double *)malloc(nParticles * nColumns * sizeof(*buffer))))
      gpuRequiredFailure("unable to allocate the host columns");
    capacity = nParticles * nColumns;
  }
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaPortColumns(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, nColumns, codes, Po, c_mks,
                              buffer);
  if (status != 0)
    gpuFatalStatus(label, status);
  return buffer;
}

/* The CPU routine on the host when an accepted beam or global loss coordinates must move with the losses */
static double **gpuPortCpuCoord(const char *reason) {
  double **coord = forceParticlesToCpu(reason);

  gpuBase.elementOnGpu = 0;
  return coord;
}

static void gpuPortCpuDone(void) {
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
}

/* One filter pass on the first nParticles particles; the particles left */
static long gpuPortPfilterPass(long nParticles, GPU_PFILTER_DATA *data) {
  unsigned long long counters[4];
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaPfilter(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("PFILTER CUDA kernel", status);
  gpuRecordApertureKernel(milliseconds);
  gpuPortCounters(counters, "PFILTER counters");
  return counters[0] ? gpuPortSwapOrder(nParticles, "PFILTER loss order") : nParticles;
}

long gpu_track_pfilter(double **initial, void *element, long np, double **accepted, double z, double Po) {
  PFILTER *pfilter = (PFILTER *)element;
  GPU_PFILTER_DATA d;
  long nLeft = np, n0 = np;
  double reference;
#ifdef GPU_VERIFY
  PFILTER before;
  long cpuLeft;
#endif

  (void)initial;
  if (np <= 0)
    return np;
  if (accepted || globalLossCoordOffset > 0) {
    double **coord = gpuPortCpuCoord("PFILTER with an accepted beam or global loss coordinates");
    nLeft = track_through_pfilter(coord, pfilter, np, accepted, z, Po);
    gpuPortCpuDone();
    return nLeft;
  }
#ifdef GPU_VERIFY
  before = *pfilter;
#endif
  startGpuTimer();
  memset(&d, 0, sizeof(d));
  d.z = z;
  d.Po = Po;
  if ((pfilter->lowerFraction || pfilter->upperFraction) && !pfilter->limitsFixed) {
    double level[2] = {-1, -1}, limit[2], upper[2];
    int code5 = 5;
    long count = 0, i;
    double *deltaBuffer = gpuPortHostColumns(np, 1, &code5, Po, "PFILTER delta column");

    if (pfilter->lowerFraction > 0 && pfilter->lowerFraction < 1) {
      upper[count] = 0;
      level[count++] = pfilter->lowerFraction * 100;
    }
    if (pfilter->upperFraction > 0 && pfilter->upperFraction < 1) {
      upper[count] = 1;
      level[count++] = 100 - pfilter->upperFraction * 100;
    }
    compute_percentiles(limit, level, count, deltaBuffer, np);
    for (i = 0; i < 2; i++) {
      if (level[i] < 0)
        break;
      if (upper[i]) {
        pfilter->pUpper = (1 + limit[i]) * Po;
        pfilter->hasUpper = 1;
      } else {
        pfilter->pLower = (1 + limit[i]) * Po;
        pfilter->hasLower = 1;
      }
      if (!pfilter->fixPLimits) {
        d.mode = GPU_PFILTER_PERCENTILE;
        d.upper = upper[i] ? 1 : 0;
        d.limit = limit[i];
        nLeft = gpuPortPfilterPass(nLeft, &d);
      }
    }
    if (pfilter->fixPLimits)
      pfilter->limitsFixed = 1;
  }
  if (pfilter->limitsFixed) {
    d.mode = GPU_PFILTER_FIXED;
    d.hasUpper = pfilter->hasUpper ? 1 : 0;
    d.hasLower = pfilter->hasLower ? 1 : 0;
    d.pUpper = pfilter->pUpper;
    d.pLower = pfilter->pLower;
    nLeft = gpuPortPfilterPass(nLeft, &d);
  }
  if (pfilter->deltaLimit >= 0) {
    reference = 0.0;
    if (pfilter->beamCentered && nLeft > 0) {
      int code5 = 5;
      double error = 0.0, *delta = gpuPortHostColumns(nLeft, 1, &code5, Po, "PFILTER delta column");
      long ip;
      for (ip = 0; ip < nLeft; ip++)
        reference = KahanPlus(reference, delta[ip], &error);
      reference /= nLeft;
    }
    d.mode = GPU_PFILTER_DELTA;
    d.reference = reference;
    d.deltaLimit = pfilter->deltaLimit;
    nLeft = gpuPortPfilterPass(nLeft, &d);
  }
  gpuMarkDeviceChanged(nLeft);
#ifdef GPU_VERIFY
  *pfilter = before;
  startCpuTimer();
  cpuLeft = track_through_pfilter(gpuBase.coord, pfilter, n0, NULL, z, Po);
  gpuPortVerifyLosses(nLeft, cpuLeft, n0, "track_through_pfilter");
#else
  if (nLeft < n0)
    gpuCopyApertureLossRowsToHost(nLeft, n0, "PFILTER lost particles");
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* SAMPLE: which particles sample_particles() removes depends only on how many it looks at and on random_2,
 * so the host runs its loop on an index array and the device gathers the rows */
long gpu_track_sample(double **initial, void *element, long np, double **accepted, double z, double p0) {
  SAMPLE *samp = (SAMPLE *)element;
  static long long *perm = NULL;
  static long capacity = 0;
  long ip, ic, itop, n0 = np;
  int status;
#ifdef GPU_VERIFY
  unsigned long long state = 0;
#endif

  (void)initial;
  if (np <= 0 || (samp->interval == 1 && samp->fraction == 1))
    return np;
  if (accepted || globalLossCoordOffset > 0) {
    double **coord = gpuPortCpuCoord("SAMPLE with an accepted beam or global loss coordinates");
    np = sample_particles(coord, samp, np, accepted, z, p0);
    gpuPortCpuDone();
    return np;
  }
  if (capacity < np) {
    free(perm);
    if (!(perm = (long long *)malloc(np * sizeof(*perm))))
      gpuRequiredFailure("unable to allocate the SAMPLE order");
    capacity = np;
  }
  for (ip = 0; ip < np; ip++)
    perm[ip] = ip;
  itop = np - 1;
  if (samp->interval != 1) {
    for (ip = ic = 0; ip < np; ip++, ic++)
      if (ic % samp->interval != 0) {
        long long row = perm[ip];
        perm[ip] = perm[itop];
        perm[itop] = row;
        --itop;
        --ip;
        --np;
      }
  } else {
#ifdef GPU_VERIFY
    state = random_2_get_state();
#endif
    for (ip = 0; ip < np; ip++)
      if (random_2(1) > samp->fraction) {
        long long row = perm[ip];
        perm[ip] = perm[itop];
        perm[itop] = row;
        --itop;
        --ip;
        --np;
      }
  }
  startGpuTimer();
  gpuCopyHostToDevice(n0);
  if (np < n0) {
    status = gpuCudaPortPermute(gpuBase.deviceCoord, n0, (int)gpuBase.deviceStride, perm);
    if (status == 0)
      status = gpuCudaPortMarkTail(gpuBase.deviceCoord, np, n0, (int)gpuBase.deviceStride, z, p0);
    if (status != 0)
      gpuFatalStatus("SAMPLE loss order", status);
  }
  gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
  /* the CPU reruns with the same random numbers in the fraction mode */
  if (samp->interval == 1)
    random_2_set_state(state);
  startCpuTimer();
  gpuPortVerifyLosses(np, sample_particles(gpuBase.coord, samp, n0, NULL, z, p0), n0, "sample_particles");
#else
  if (np < n0)
    gpuCopyApertureLossRowsToHost(np, n0, "SAMPLE lost particles");
#endif
  gpuRecordWallSeconds();
  return np;
}

/* CLEAN: remove_outlier_particles()'s centroids and deviations on the host from the copied columns, the
 * test on the device */
long gpu_track_clean(double **initial, void *element, long np, double **accepted, double z, double Po) {
  CLEAN *clean = (CLEAN *)element;
  static char *modeName[3] = {"stdeviation", "absdeviation", "absvalue"};
  int codes[7] = {0, 1, 2, 3, GPU_PORT_COLUMN_CLEAN_S, 5, GPU_PORT_COLUMN_FINITE};
  GPU_CLEAN_DATA d;
  unsigned long long counters[4];
  double *columns, centroid[6], stDev[6];
  long ip, j, count, nLeft, mode;
  float milliseconds = 0;
  int status;

  (void)initial;
  mode = match_string(clean->mode, modeName, 3, 0);
  if (np <= 0 || accepted || globalLossCoordOffset > 0 || mode < 0) {
    double **coord = gpuPortCpuCoord("CLEAN with an accepted beam or global loss coordinates");
    nLeft = remove_outlier_particles(coord, clean, np, accepted, z, Po);
    gpuPortCpuDone();
    return nLeft;
  }
  if (!(clean->xLimit > 0 || clean->xpLimit > 0 || clean->yLimit > 0 || clean->ypLimit > 0 || clean->tLimit > 0 ||
        clean->deltaLimit > 0))
    return np;
  startGpuTimer();
  columns = gpuPortHostColumns(np, 7, codes, Po, "CLEAN columns");
  for (j = 0; j < 6; j++)
    centroid[j] = stDev[j] = 0;
  for (ip = count = 0; ip < np; ip++) {
    if (!columns[6 * np + ip])
      continue;
    count++;
    for (j = 0; j < 6; j++)
      centroid[j] += columns[j * np + ip];
  }
  if (!count) {
    status = gpuCudaPortMarkTail(gpuBase.deviceCoord, 0, np, (int)gpuBase.deviceStride, z, Po);
    if (status != 0)
      gpuFatalStatus("CLEAN lost particles", status);
    gpuMarkDeviceChanged(0);
#ifdef GPU_VERIFY
    startCpuTimer();
    gpuPortVerifyLosses(0, remove_outlier_particles(gpuBase.coord, clean, np, NULL, z, Po), np,
                        "remove_outlier_particles");
#else
    gpuCopyApertureLossRowsToHost(0, np, "CLEAN lost particles");
#endif
    gpuRecordWallSeconds();
    return 0;
  }
  for (j = 0; j < 6; j++)
    centroid[j] /= count;
  if (mode == 0) {
    for (ip = 0; ip < np; ip++)
      for (j = 0; j < 6; j++) {
        double value = columns[j * np + ip];
        if (!isnan(value) && !isinf(value))
          stDev[j] += sqr(centroid[j] - value);
      }
    for (j = 0; j < 6; j++)
      stDev[j] = sqrt(stDev[j] / count);
  }
  memset(&d, 0, sizeof(d));
  d.limit[0] = clean->xLimit;
  d.limit[1] = clean->xpLimit;
  d.limit[2] = clean->yLimit;
  d.limit[3] = clean->ypLimit;
  d.limit[4] = clean->tLimit;
  d.limit[5] = clean->deltaLimit;
  memcpy(d.centroid, centroid, sizeof(centroid));
  memcpy(d.stDev, stDev, sizeof(stDev));
  d.z = z;
  d.Po = Po;
  d.cMks = c_mks;
  d.mode = (int)mode;
  status = gpuCudaClean(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &d, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CLEAN CUDA kernel", status);
  gpuRecordApertureKernel(milliseconds);
  gpuPortCounters(counters, "CLEAN counters");
  nLeft = counters[0] ? gpuPortSwapOrder(np, "CLEAN loss order") : np;
  gpuMarkDeviceChanged(nLeft);
#ifdef GPU_VERIFY
  startCpuTimer();
  gpuPortVerifyLosses(nLeft, remove_outlier_particles(gpuBase.coord, clean, np, NULL, z, Po), np,
                      "remove_outlier_particles");
#else
  if (nLeft < np)
    gpuCopyApertureLossRowsToHost(nLeft, np, "CLEAN lost particles");
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* EMITTANCE: transformEmittances()'s beam sums on a read-only copy, the transformation on the device */
long gpu_track_emittance(double **particle, long np, double pCentral, void *element) {
  EMITTANCEELEMENT *ee = (EMITTANCEELEMENT *)element;
  GPU_EMITTANCE_DATA d;
  double emit, emitc, pAverage, eta, **part;
  BEAM_SUMS *sums;
  long saveOnGpu, i;
  float milliseconds = 0;
  int status;

  (void)particle;
  if (np <= 0)
    return 0;
  startGpuTimer();
  part = copyParticlesToCpuReadOnly("EMITTANCE beam sums");
  saveOnGpu = gpuBase.elementOnGpu;
  gpuBase.elementOnGpu = 0;
  sums = allocateBeamSums(0, 1);
  zero_beam_sums(sums, 1);
  accumulate_beam_sums(sums, part, np, pCentral, 0, NULL, 0.0, 0.0, -1, -1, 0);
  gpuBase.elementOnGpu = saveOnGpu;
  pAverage = pCentral * (1 + sums->centroid[5]);
  memset(&d, 0, sizeof(d));
  for (i = 0; i < 2; i++) {
    computeEmitTwissFromSigmaMatrix(&emit, &emitc, NULL, NULL, sums->beamSums2->sigma, 2 * i);
    eta = 0;
    if (sums->beamSums2->sigma[5][5])
      eta = sums->beamSums2->sigma[2 * i + 0][5] / sums->beamSums2->sigma[5][5];
    if (ee->emit[i] >= 0) {
      if (emitc > 0)
        d.factor[i] = sqrt(ee->emit[i] / emitc);
      else
        continue;
    } else if (ee->emitn[i] >= 0) {
      emitc *= pAverage;
      if (emitc > 0)
        d.factor[i] = sqrt(ee->emitn[i] / emitc);
      else
        continue;
    } else
      continue;
    d.eta[i] = eta;
    d.doPlane[i] = 1;
  }
  freeBeamSums(sums, 1);
  if (d.doPlane[0] || d.doPlane[1]) {
    status = gpuCudaEmittance(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &d, &milliseconds);
    if (status != 0)
      gpuFatalStatus("EMITTANCE CUDA kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
    startCpuTimer();
    transformEmittances(gpuBase.coord, np, pCentral, ee);
    compareGpuCpu(np, "transformEmittances");
#endif
  }
  gpuRecordWallSeconds();
  return 1;
}

/* REMCOR: remove_correlations()'s sums on the host from the copied columns */
long gpu_track_remcor(double **part, void *element, long np) {
  REMCOR *remcor = (REMCOR *)element;
  GPU_REMCOR_DATA d;
  long removeFrom[4], ic, ip, wc, nColumns = 0, columnOf[4];
  int codes[5];
  double sumy2, sumxy, *columns;
  float milliseconds = 0;
  int status;

  (void)part;
  if (!np)
    return 1;
  removeFrom[0] = remcor->x;
  removeFrom[1] = remcor->xp;
  removeFrom[2] = remcor->y;
  removeFrom[3] = remcor->yp;
  wc = remcor->with - 1;
  startGpuTimer();
  codes[nColumns++] = (int)wc;
  for (ic = 0; ic < 4; ic++) {
    columnOf[ic] = -1;
    if (removeFrom[ic] && ic != wc && !remcor->ratioSet[ic]) {
      columnOf[ic] = nColumns;
      codes[nColumns++] = (int)ic;
    }
  }
  columns = gpuPortHostColumns(np, (int)nColumns, codes, 0.0, "REMCOR columns");
  for (ip = 0, sumy2 = 0; ip < np; ip++)
    sumy2 += columns[ip] * columns[ip];
  if (!sumy2) {
    gpuRecordWallSeconds();
    return 1;
  }
  memset(&d, 0, sizeof(d));
  d.wc = (int)wc;
  for (ic = 0; ic < 4; ic++) {
    if (!removeFrom[ic] || ic == wc)
      continue;
    if (!remcor->ratioSet[ic]) {
      for (ip = 0, sumxy = 0; ip < np; ip++)
        sumxy += columns[columnOf[ic] * np + ip] * columns[ip];
      d.ratio[ic] = sumxy / sumy2;
      if (remcor->onceOnly) {
        remcor->ratio[ic] = d.ratio[ic];
        remcor->ratioSet[ic] = 1;
      }
    } else
      d.ratio[ic] = remcor->ratio[ic];
    d.active[ic] = 1;
  }
  status = gpuCudaRemcor(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &d, &milliseconds);
  if (status != 0)
    gpuFatalStatus("REMCOR CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
  {
    REMCOR saved = *remcor;
    long i;
    /* the CPU computes again what this call did not store */
    for (i = 0; i < 4; i++)
      if (columnOf[i] >= 0 && remcor->onceOnly) {
        remcor->ratioSet[i] = 0;
      }
    startCpuTimer();
    remove_correlations(gpuBase.coord, remcor, np);
    *remcor = saved;
    compareGpuCpu(np, "remove_correlations");
  }
#endif
  gpuRecordWallSeconds();
  return 1;
}

/* MHISTOGRAM: findMinMax()'s results, with the time mean added on the host in particle order */
long gpu_mhist_minmax(long np, double Po, double *min, double *max, double *c0) {
  static double *time = NULL;
  static long capacity = 0;
  long i, j;
  int status;

  if (!getElementOnGpu() || np <= 0)
    return 0;
  if (capacity < np) {
    free(time);
    if (!(time = (double *)malloc(np * sizeof(*time))))
      gpuRequiredFailure("unable to allocate the MHISTOGRAM times");
    capacity = np;
  }
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaMhistMinMax(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, Po, c_mks, min, max, time);
  if (status != 0)
    gpuFatalStatus("MHISTOGRAM minima and maxima", status);
  for (j = 0; j < 6; j++)
    c0[j] = 0;
  for (i = 0; i < np; i++)
    c0[4] += time[i];
  c0[4] /= (double)np;
#ifdef GPU_VERIFY
  {
    double cpuMin[6], cpuMax[6], cpuC0[6];
    findMinMax(gpuBase.coord, np, cpuMin, cpuMax, cpuC0, Po);
    for (j = 0; j < 6; j++) {
      compareGpuCpuValue(min[j], cpuMin[j], 0, "findMinMax minimum");
      compareGpuCpuValue(max[j], cpuMax[j], 0, "findMinMax maximum");
    }
    compareGpuCpuValue(c0[4], cpuC0[4], 0, "findMinMax time mean");
  }
#endif
  gpuRecordWallSeconds();
  return 1;
}

static long gpuMhistBook1m(GPU_MHIST_BOOK *book, book1m *b, int32_t *xbins, long *total) {
  long i, j;

  memset(book, 0, sizeof(*book));
  book->kind = 0;
  book->nD = (int)b->nD;
  book->bins = (int)b->bins;
  for (i = j = 0; i < 6; i++) {
    if (!xbins[i])
      continue;
    book->column[j] = (int)i;
    book->xmin[j] = b->xmin[j];
    book->dx[j] = b->dx[j];
    j++;
  }
  book->countOffset = *total;
  *total += b->length * b->nD;
  return 1;
}

static long gpuMhistBookN(GPU_MHIST_BOOK *book, ntuple *b, long offset, long *total) {
  long i;

  if (b->nD > 6)
    return 0;
  memset(book, 0, sizeof(*book));
  book->kind = 1;
  book->nD = (int)b->nD;
  book->offset = (int)offset;
  for (i = 0; i < b->nD; i++) {
    book->xbins[i] = b->xbins[i];
    book->xmin[i] = b->xmin[i];
    book->dx[i] = b->dx[i];
  }
  book->countOffset = *total;
  *total += b->length;
  return 1;
}

/* mhist_table()'s fill loop: the counts of chfill1m() and chfilln() for np particles of weight 1 */
long gpu_mhist_fill(long np, double Po, double t0, void *book1d, int32_t *bins1d, void *x2d, void *y2d, void *z2d,
                    void *tr4d, void *full6d) {
  GPU_MHIST_BOOK books[GPU_MHIST_BOOKS];
  void *owner[GPU_MHIST_BOOKS];
  static unsigned long long *counts = NULL;
  static long capacity = 0;
  long nBooks = 0, total = 0, b, i, j;
  int status;

  if (!getElementOnGpu() || np <= 0)
    return 0;
  if (book1d && gpuMhistBook1m(&books[nBooks], (book1m *)book1d, bins1d, &total))
    owner[nBooks++] = book1d;
  if (x2d && gpuMhistBookN(&books[nBooks], (ntuple *)x2d, 0, &total))
    owner[nBooks++] = x2d;
  if (y2d && gpuMhistBookN(&books[nBooks], (ntuple *)y2d, 2, &total))
    owner[nBooks++] = y2d;
  if (z2d && gpuMhistBookN(&books[nBooks], (ntuple *)z2d, 4, &total))
    owner[nBooks++] = z2d;
  if (tr4d && gpuMhistBookN(&books[nBooks], (ntuple *)tr4d, 0, &total))
    owner[nBooks++] = tr4d;
  if (full6d && gpuMhistBookN(&books[nBooks], (ntuple *)full6d, 0, &total))
    owner[nBooks++] = full6d;
  if (!nBooks)
    return 1;
  if (capacity < total) {
    free(counts);
    if (!(counts = (unsigned long long *)malloc(total * sizeof(*counts))))
      gpuRequiredFailure("unable to allocate the MHISTOGRAM counts");
    capacity = total;
  }
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaMhistFill(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, Po, c_mks, t0, (int)nBooks, books,
                            total, counts);
  if (status != 0)
    gpuFatalStatus("MHISTOGRAM fill", status);
  /* each particle added 1 to one bin of each dimension (chfill1m()) or of the book (chfilln()) */
  for (b = 0; b < nBooks; b++) {
    unsigned long long *c = counts + books[b].countOffset;
    if (books[b].kind == 0) {
      book1m *bk = (book1m *)owner[b];
      for (i = 0; i < bk->length; i++)
        for (j = 0; j < bk->nD; j++)
          bk->value[i][j] += (double)c[i * bk->nD + j];
      bk->total += np;
      bk->count += np;
    } else {
      ntuple *bk = (ntuple *)owner[b];
      for (i = 0; i < bk->length; i++)
        bk->value[i] += (double)c[i];
      bk->total += np;
      bk->count += np;
    }
  }
  gpuRecordWallSeconds();
  return 1;
}

/* After the host changed one coordinate of a beam whose current copy it holds: that column to the device */
void gpu_port_upload_column(long np, long column) {
  static double *values = NULL;
  static long capacity = 0;
  long ip;
  int status;

  if (np <= 0 || !gpuBase.deviceCoord)
    return;
  if (capacity < np) {
    free(values);
    if (!(values = (double *)malloc(np * sizeof(*values))))
      gpuRequiredFailure("unable to allocate the column upload");
    capacity = np;
  }
  for (ip = 0; ip < np; ip++)
    values[ip] = gpuBase.coord[ip][column];
  status = gpuCudaPortSetColumn(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, (int)column, values);
  if (status != 0)
    gpuFatalStatus("column upload", status);
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 1;
}

long gpu_port_host_begin(void) {
  long saved = gpuBase.elementOnGpu;

  gpuBase.elementOnGpu = 0;
  return saved;
}

void gpu_port_host_end(long saved) {
  gpuBase.elementOnGpu = saved;
}

/* ---------------------------------------------------------------------------------------------------- */
/* Random elements: the host draws random_2() in the CPU's order and the device applies what it drew */

void distributionScatter(double **part, int64_t np, double Po, DSCATTER *scat, long iPass);
double radiationLength(long Z, double A, double rho);
#pragma weak scatter_ele
#pragma weak distributionScatter
#pragma weak track_through_matter
#pragma weak radiationLength
#pragma weak pepper_pot_plate
#pragma weak printWarning

extern int gpuCudaIncrement(void *coord, long nParticles, int stride, const GPU_INCREMENT_DATA *data,
                            const double *hostValues, float *milliseconds);
extern int gpuCudaMatter(void *coord, long nParticles, int stride, const GPU_MATTER_DATA *data,
                         const double *hostValues, const double *hostEvents, long nEvents, float *milliseconds);
extern int gpuCudaDrawIncrement(void *coord, long nParticles, int stride, const GPU_DRAW_DATA *data,
                                const double *hostStates, unsigned long long *counters, float *milliseconds);

/* SCATTER, DSCATTER and MATTER make their random numbers on the device from random_2's sequence: the host
 * keeps every decision the numbers make (which particles scatter, in which section, which are lost, and so
 * the order), walking the sequence with dlaran's own integer steps, and the device makes the numbers from
 * where each particle's start.  Every uniform is the CPU's; the Gaussian numbers differ by the rounding of
 * CUDA's log and sin.  ELEGANT_GPU_PORT_DEVICE_DRAWS=0 makes them on the host, as random_2 and gauss_rn(). */
static long gpuPortDeviceDraws(void) {
  static long enabled = -1;

  if (enabled < 0)
    enabled = !gpuEnvSet("ELEGANT_GPU_PORT_DEVICE_DRAWS") || gpuEnvFlag("ELEGANT_GPU_PORT_DEVICE_DRAWS");
  return enabled;
}

/* The next state of random_2's sequence */
#define GPU_PORT_NEXT_STATE(state) (((state) * GPU_DLARAN_MULTIPLIER) & GPU_DLARAN_MASK)

static void gpuPortDrawIncrement(long np, const GPU_DRAW_DATA *data, const double *states, unsigned long long *counters,
                                 const char *label) {
  float milliseconds = 0;
  int status;

  gpuCopyHostToDevice(np);
  status = gpuCudaDrawIncrement(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, data, states, counters,
                                &milliseconds);
  if (status != 0)
    gpuFatalStatus(label, status);
  gpuRecordHelperKernel(milliseconds);
  if (counters[3])
    gpuPortInvalidPending = 1;
  gpuMarkDeviceChanged(np);
}
extern int gpuCudaPeppot(void *coord, long nParticles, int stride, const GPU_PEPPOT_DATA *data, const double *hostX,
                         const double *hostY, unsigned char *hostFlags);

/* Buffers for the host's values and the loss order */
static double *gpuPortHostValues(long count) {
  static double *buffer = NULL;
  static long capacity = 0;

  if (capacity < count) {
    free(buffer);
    if (!(buffer = (double *)malloc(count * sizeof(*buffer))))
      gpuRequiredFailure("unable to allocate the host values");
    capacity = count;
  }
  return buffer;
}

static long long *gpuPortHostOrder(long nParticles) {
  static long long *buffer = NULL;
  static long capacity = 0;
  long i;

  if (capacity < nParticles) {
    free(buffer);
    if (!(buffer = (long long *)malloc(nParticles * sizeof(*buffer))))
      gpuRequiredFailure("unable to allocate the loss order");
    capacity = nParticles;
  }
  for (i = 0; i < nParticles; i++)
    buffer[i] = i;
  return buffer;
}

/* The host's increments to the device's current beam (gpuCopyHostToDevice() already called) */
static void gpuPortIncrement(long nParticles, const GPU_INCREMENT_DATA *data, const double *values,
                             const char *label) {
  unsigned long long counters[4];
  float milliseconds = 0;
  int status;

  status = gpuCudaIncrement(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, data, values, &milliseconds);
  if (status != 0)
    gpuFatalStatus(label, status);
  gpuRecordHelperKernel(milliseconds);
  gpuPortCounters(counters, label);
  if (counters[3])
    gpuPortInvalidPending = 1;
}

/* Rows in the order the host found (perm[i] is the row that ends up at i) */
static void gpuPortGather(long nParticles, const long long *perm, const char *label) {
  int status = gpuCudaPortPermute(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, perm);

  if (status != 0)
    gpuFatalStatus(label, status);
}

static long gpuScatterSupported(ELEMENT_LIST *eptr) {
  SCATTER *scat = (SCATTER *)eptr->p_elem;

  return scat->distribution &&
    (strcasecmp(scat->distribution, "gaussian") == 0 || strcasecmp(scat->distribution, "uniform") == 0);
}

static long gpuDscatterSupported(ELEMENT_LIST *eptr) {
  return !((DSCATTER *)eptr->p_elem)->oncePerParticle;
}

static long gpuMatterSupported(ELEMENT_LIST *eptr) {
  MATTER *matter = (MATTER *)eptr->p_elem;

  return particleIsElectron && !matter->nuclearBremsstrahlung && !matter->electronRecoil &&
    !(matter->spacing > 0 && matter->width > 0);
}

/* SCATTER: scatter_ele() */
void gpu_track_scatter(double **part, long np, double Po, void *element, long iPass) {
  SCATTER *scat = (SCATTER *)element;
  GPU_INCREMENT_DATA d;
  double sigma[4], *values, *v;
  long ip, i, gaussian, any = 0;
#ifdef GPU_VERIFY
  unsigned long long state;
#endif

  (void)part;
  if (!np)
    return;
  if (iPass < scat->startOnPass || (scat->endOnPass >= 0 && iPass > scat->endOnPass))
    return;
  gaussian = scat->distribution && strcasecmp(scat->distribution, "gaussian") == 0;
  if (!gaussian && !(scat->distribution && strcasecmp(scat->distribution, "uniform") == 0)) {
    scatter_ele(gpuPortCpuCoord("SCATTER distribution"), np, Po, scat, iPass);
    gpuPortCpuDone();
    return;
  }
  startGpuTimer();
#ifdef GPU_VERIFY
  state = random_2_get_state();
#endif
  sigma[0] = scat->x;
  sigma[1] = scat->xp;
  sigma[2] = scat->y;
  sigma[3] = scat->yp;
  memset(&d, 0, sizeof(d));
  d.Po = Po;
  for (i = 0; i < 4; i++)
    d.active[i] = sigma[i] != 0;
  d.dpActive = scat->dp != 0;
  if (gpuPortDeviceDraws()) {
    GPU_DRAW_DATA draw;
    unsigned long long S = random_2_get_state(), counters[4];
    long draws = 0;

    memset(&draw, 0, sizeof(draw));
    draw.inc = d;
    for (i = 0; i < 4; i++) {
      draw.sigma[i] = sigma[i];
      if (sigma[i])
        draws += gaussian ? 2 : 1;
    }
    draw.dp = scat->dp;
    if (scat->dp)
      draws += gaussian ? 2 : 1;
    draw.gaussian = (int)gaussian;
    draw.stride = gpuDlaranJump(1, draws);
    if (scat->probability < 1) {
      /* the CPU's draw for each particle, and its numbers when it scatters */
      double *states = gpuPortHostValues(np);
      for (ip = 0; ip < np; ip++) {
        S = GPU_PORT_NEXT_STATE(S);
        if (S * GPU_DLARAN_SCALE > scat->probability) {
          states[ip] = 0;
          continue;
        }
        states[ip] = (double)GPU_PORT_NEXT_STATE(S);
        S = (S * draw.stride) & GPU_DLARAN_MASK;
        any = 1;
      }
      random_2_set_state(S);
      if (any)
        gpuPortDrawIncrement(np, &draw, states, counters, "SCATTER CUDA kernel");
    } else {
      draw.count = np;
      draw.first = GPU_PORT_NEXT_STATE(S);
      random_2_set_state(gpuDlaranJump(S, (unsigned long long)np * draws));
      gpuPortDrawIncrement(np, &draw, NULL, counters, "SCATTER CUDA kernel");
    }
#ifdef GPU_VERIFY
    {
      unsigned long long after = random_2_get_state();
      random_2_set_state(state);
      startCpuTimer();
      scatter_ele(gpuBase.coord, np, Po, scat, iPass);
      compareGpuCpuValue((double)after, (double)random_2_get_state(), 0, "SCATTER random numbers used");
      compareGpuCpu(np, "scatter_ele");
    }
#endif
    gpuRecordWallSeconds();
    return;
  }
  values = gpuPortHostValues(np * GPU_INCREMENT_VALUES);
  for (ip = 0; ip < np; ip++) {
    v = values + ip * GPU_INCREMENT_VALUES;
    v[0] = 0;
    if (scat->probability < 1 && random_2(1) > scat->probability)
      continue;
    v[0] = 1;
    any = 1;
    for (i = 0; i < 4; i++) {
      if (!sigma[i])
        continue;
      v[1 + i] = gaussian ? gauss_rn(0, random_2) * sigma[i] : 2 * sigma[i] * (random_2(1) - 0.5);
    }
    if (scat->dp)
      v[5] = gaussian ? scat->dp * gauss_rn(0, random_2) : 2 * scat->dp * (random_2(1) - 0.5);
  }
  if (any) {
    gpuCopyHostToDevice(np);
    gpuPortIncrement(np, &d, values, "SCATTER CUDA kernel");
    gpuMarkDeviceChanged(np);
  }
#ifdef GPU_VERIFY
  random_2_set_state(state);
  startCpuTimer();
  scatter_ele(gpuBase.coord, np, Po, scat, iPass);
  compareGpuCpu(np, "scatter_ele");
#endif
  gpuRecordWallSeconds();
}

/* DSCATTER: distributionScatter() without ONCE_PER_PARTICLE */
void gpu_track_dscatter(double **part, long np, double Po, void *element, long iPass) {
  DSCATTER *scat = (DSCATTER *)element;
  GPU_INCREMENT_DATA d;
  TRACKING_CONTEXT context;
  double *values, amplitude, cdf;
  long ip, nLeftThisPass, nScattered, interpCode, column;
#ifdef GPU_VERIFY
  unsigned long long state;
  long nLeftBefore;
#endif

  if (!np)
    return;
  if (scat->oncePerParticle) {
    distributionScatter(gpuPortCpuCoord("DSCATTER with ONCE_PER_PARTICLE"), np, Po, scat, iPass);
    gpuPortCpuDone();
    return;
  }
  if (!scat->initialized)
    /* reads the file and sets up the element; a pass before STARTONPASS leaves the particles alone */
    distributionScatter(part, np, Po, scat, -1);
  if (iPass < scat->startOnPass)
    return;
  if (scat->endOnPass >= 0 && iPass > scat->endOnPass)
    return;
#ifdef GPU_VERIFY
  nLeftBefore = scat->nLeft;
#endif
  if (iPass == scat->startOnPass) {
    scat->nLeft = scat->limitTotal >= 0 ? scat->limitTotal : np;
    if (scat->nLeft > np)
      scat->nLeft = np;
  }
  nLeftThisPass = np;
  if (scat->limitTotal >= 0)
    nLeftThisPass = scat->nLeft;
  if (scat->limitPerPass >= 0 && scat->limitPerPass < nLeftThisPass)
    nLeftThisPass = scat->limitPerPass;
  if (nLeftThisPass == 0)
    return;
  startGpuTimer();
#ifdef GPU_VERIFY
  state = random_2_get_state();
#endif
  memset(&d, 0, sizeof(d));
  d.Po = Po;
  if (scat->iPlane == 5) {
    d.dpActive = 1;
    column = 5;
  } else {
    d.active[scat->iPlane] = 1;
    column = 1 + scat->iPlane;
  }
  if (gpuPortDeviceDraws()) {
    GPU_DRAW_DATA draw;
    unsigned long long S = random_2_get_state(), counters[4];
    double *states = NULL;

    memset(&draw, 0, sizeof(draw));
    draw.inc = d;
    draw.dscatter = 1;
    draw.column = (int)column;
    draw.randomSign = scat->randomSign ? 1 : 0;
    draw.factor = scat->factor;
    draw.indep = scat->indepData;
    draw.cdf = scat->cdfData;
    draw.nData = scat->nData;
    draw.stride = gpuDlaranJump(1, scat->randomSign ? 2 : 1);
    nScattered = 0;
    if (scat->probability < 1) {
      /* the CPU's draw for each particle, then its numbers if it scatters and the limits allow */
      states = gpuPortHostValues(np);
      for (ip = 0; ip < np; ip++)
        states[ip] = 0;
      for (ip = 0; ip < np; ip++) {
        S = GPU_PORT_NEXT_STATE(S);
        if (S * GPU_DLARAN_SCALE > scat->probability)
          continue;
        if (nLeftThisPass == 0)
          break;
        nScattered++;
        nLeftThisPass--;
        states[ip] = (double)GPU_PORT_NEXT_STATE(S);
        S = (S * draw.stride) & GPU_DLARAN_MASK;
      }
    } else {
      nScattered = nLeftThisPass < np ? nLeftThisPass : np;
      draw.count = nScattered;
      draw.first = GPU_PORT_NEXT_STATE(S);
      S = gpuDlaranJump(S, (unsigned long long)nScattered * (scat->randomSign ? 2 : 1));
    }
    random_2_set_state(S);
    scat->nLeft -= nScattered;
    if (nScattered) {
      gpuPortDrawIncrement(np, &draw, states, counters, "DSCATTER CUDA kernel");
      if (counters[1]) {
        /* interp() failed for some particles: the CPU's warnings, in its order */
        unsigned long long cdfState = draw.first;
        long scattered = 0;
        for (ip = 0; ip < np && scattered < nScattered; ip++) {
          if (states) {
            if (!states[ip])
              continue;
            cdfState = (unsigned long long)states[ip];
          } else if (ip)
            cdfState = (cdfState * draw.stride) & GPU_DLARAN_MASK;
          scattered++;
          cdf = cdfState * GPU_DLARAN_SCALE;
          interp(scat->indepData, scat->cdfData, scat->nData, cdf, 0, 1, &interpCode);
          if (!interpCode) {
            char buffer[16384];
            getTrackingContext(&context);
            snprintf(buffer, 16384, ". Element %s, CDF=%e", context.elementName, cdf);
            printWarning("DSCATTER interpolation error", buffer);
          }
        }
      }
    }
#ifdef GPU_VERIFY
    {
      long nLeftAfter = scat->nLeft;
      unsigned long long after = random_2_get_state();
      random_2_set_state(state);
      scat->nLeft = nLeftBefore;
      startCpuTimer();
      distributionScatter(gpuBase.coord, np, Po, scat, iPass);
      compareGpuCpuValue((double)nLeftAfter, (double)scat->nLeft, 0, "DSCATTER particles left to scatter");
      compareGpuCpuValue((double)after, (double)random_2_get_state(), 0, "DSCATTER random numbers used");
      compareGpuCpu(np, "distributionScatter");
    }
#endif
    gpuRecordWallSeconds();
    return;
  }
  values = gpuPortHostValues(np * GPU_INCREMENT_VALUES);
  for (ip = 0; ip < np; ip++)
    values[ip * GPU_INCREMENT_VALUES] = 0;
  nScattered = 0;
  for (ip = 0; ip < np; ip++) {
    if (scat->probability < 1 && random_2(1) > scat->probability)
      continue;
    if (nLeftThisPass == 0)
      break;
    nScattered++;
    nLeftThisPass--;
    cdf = random_2(1);
    amplitude = scat->factor * interp(scat->indepData, scat->cdfData, scat->nData, cdf, 0, 1, &interpCode);
    if (scat->randomSign)
      amplitude *= random_2(1) > 0.5 ? 1 : -1;
    if (!interpCode) {
      char buffer[16384];
      getTrackingContext(&context);
      snprintf(buffer, 16384, ". Element %s, CDF=%e", context.elementName, cdf);
      printWarning("DSCATTER interpolation error", buffer);
    }
    values[ip * GPU_INCREMENT_VALUES] = 1;
    values[ip * GPU_INCREMENT_VALUES + column] = amplitude;
  }
  scat->nLeft -= nScattered;
  if (nScattered) {
    gpuCopyHostToDevice(np);
    gpuPortIncrement(np, &d, values, "DSCATTER CUDA kernel");
    gpuMarkDeviceChanged(np);
  }
#ifdef GPU_VERIFY
  {
    long nLeftAfter = scat->nLeft;
    random_2_set_state(state);
    scat->nLeft = nLeftBefore;
    startCpuTimer();
    distributionScatter(gpuBase.coord, np, Po, scat, iPass);
    compareGpuCpuValue((double)nLeftAfter, (double)scat->nLeft, 0, "DSCATTER particles left to scatter");
    compareGpuCpu(np, "distributionScatter");
  }
#endif
  gpuRecordWallSeconds();
}

/* MATTER: track_through_matter() without slots, bremsstrahlung or electron recoil */
#define GPU_MATTER_SQRT_3 (1.7320508075688772)
#define GPU_MATTER_AMU (1.6605e-27)
#define GPU_MATTER_SQR_PI (PI * PI)
#define GPU_MATTER_ALPHA (1. / 137.036)
#define GPU_MATTER_MP_MKS 1.672648500000000e-27

static long gpuMatterOnCpu(long np, long iPass, MATTER *matter, double Po, double **accepted, double z0,
                           const char *reason) {
  long nLeft = track_through_matter(gpuPortCpuCoord(reason), np, iPass, matter, Po, accepted, z0);

  gpuPortCpuDone();
  return nLeft;
}

long gpu_track_matter(double **part, long np, long iPass, void *element, double Po, double **accepted, double z0) {
  MATTER *matter = (MATTER *)element;
  GPU_MATTER_DATA d;
  double L, L1, Nrad, theta_rms = 0, beta, beta4gamma2, rho, nDensity, Xo, dGammaFactor;
  double K1, K2 = 0, sigmaTotal, probScatter, prob = 0, P, gamma = 0, dgamma, dgamma1, *values, *v, *delta = NULL;
  static double *events = NULL;
  static long eventCapacity = 0;
  long sections0 = 1, multipleScattering, impulseMode, i, top, nLeft, nEvents = 0, is, count, deviceDraws;
  long long *perm, row;
  unsigned long long counters[4], state, S, drawStride;
  float milliseconds = 0;
  int status, code = 5;

  (void)part;
  if (np <= 0)
    return np;
  if (accepted || globalLossCoordOffset > 0 || !particleIsElectron || matter->nuclearBremsstrahlung ||
      matter->electronRecoil || (matter->spacing > 0 && matter->width > 0))
    return gpuMatterOnCpu(np, iPass, matter, Po, accepted, z0,
                          "MATTER with an accepted beam, global loss coordinates, slots or special scattering");
  if ((matter->startPass >= 0 && matter->startPass > iPass) || (matter->endPass >= 0 && matter->endPass < iPass)) {
    startGpuTimer();
    gpu_exactDrift(np, matter->length);
#ifdef GPU_VERIFY
    startCpuTimer();
    {
      long saved = gpu_port_host_begin();
      track_through_matter(gpuBase.coord, np, iPass, matter, Po, NULL, z0);
      gpu_port_host_end(saved);
    }
    compareGpuCpu(np, "track_through_matter");
#endif
    return np;
  }
  if (matter->length != 0) {
    L = matter->length;
    impulseMode = 0;
  } else if (matter->lEffective != 0) {
    L = matter->lEffective;
    impulseMode = 1;
  } else
    return np;
  L1 = L;

  /* the CPU's set-up, line for line */
  beta = Po / sqrt(sqr(Po) + 1);
  beta4gamma2 = sqr(beta) * sqr(Po);
  rho = matter->rho;
  if (matter->pressure > 0 && matter->temperature > 0)
    rho = matter->multiplicity * matter->pressure / (k_boltzmann_mks * matter->temperature) * GPU_MATTER_MP_MKS * matter->A;
  nDensity = rho / (GPU_MATTER_MP_MKS * matter->A);
  if (matter->Xo == 0) {
    if (matter->Z < 1 || matter->A < 1 || rho <= 0)
      return gpuMatterOnCpu(np, iPass, matter, Po, accepted, z0, "MATTER without a radiation length");
    Xo = radiationLength(matter->Z, matter->A, rho);
  } else {
    Xo = matter->Xo;
    if (matter->Z > 0 || matter->A > 0 || rho > 0) {
      if (matter->pressure && matter->temperature)
        printWarningForTracking("Redundant data supplied for MATTER element", "Xo, Z, A, temperature, pressure all specified");
      else
        printWarningForTracking("Redundant data supplied for MATTER element", "Xo, Z, A, rho all specified");
    }
  }
  Nrad = L / Xo;
  dGammaFactor = 1 - exp(-Nrad);
  if (Nrad < 1e-3 && !matter->forceMCS) {
    if (rho == 0)
      return gpuMatterOnCpu(np, iPass, matter, Po, accepted, z0, "MATTER without a density");
    K1 = 4 * matter->Z * (matter->Z + 1) * sqr(particleRadius / (beta * Po));
    K2 = sqr(pow(matter->Z, 1. / 3.) * GPU_MATTER_ALPHA / Po);
    sigmaTotal = K1 * pow(PI, 3) / (sqr(K2) + K2 * GPU_MATTER_SQR_PI);
    probScatter = rho / (GPU_MATTER_AMU * matter->A) * L * sigmaTotal;
    sections0 = probScatter / matter->pLimit + 1;
    Nrad /= sections0;
    multipleScattering = 0;
    L1 = L / sections0;
    prob = probScatter / sections0;
  } else {
    if (matter->forceMCS && Nrad < 1e-3) {
      multipleScattering = 2;
      theta_rms = sqrt(4 * PI * nDensity * L * sqr(particleRadius) / beta4gamma2 * matter->Z * (matter->Z + 1) *
                       log(184.15 / pow(matter->Z, 1. / 3.)));
    } else {
      multipleScattering = 1;
      theta_rms = 13.6 / particleMassMV / Po / sqr(beta) * sqrt(Nrad) * (1 + 0.038 * log(Nrad));
    }
  }
  if (!Nrad)
    return gpuMatterOnCpu(np, iPass, matter, Po, accepted, z0, "MATTER with no radiation lengths");
  if (impulseMode)
    L = L1 = 0;

  startGpuTimer();
  state = random_2_get_state();
  memset(&d, 0, sizeof(d));
  deviceDraws = gpuPortDeviceDraws();
  /* the device's numbers: four or two Gaussian numbers (two uniforms each) a particle in modes 1 and 2,
   * three uniforms an event in mode 0 */
  drawStride = gpuDlaranJump(1, multipleScattering == 1 ? 8 : (multipleScattering == 2 ? 4 : 3));
  S = state;
  if (deviceDraws && multipleScattering && !matter->energyDecay) {
    /* nothing to decide: particle ip's numbers start at first * stride^ip */
    gpuCopyHostToDevice(np);
    d.first = GPU_PORT_NEXT_STATE(S);
    d.stride = drawStride;
    random_2_set_state(gpuDlaranJump(S, (unsigned long long)np * (multipleScattering == 1 ? 8 : 4)));
    values = NULL;
    perm = NULL;
    top = np - 1;
  } else {
    if (matter->energyDecay)
      delta = gpuPortHostColumns(np, 1, &code, Po, "MATTER delta");
    else
      gpuCopyHostToDevice(np);
    values = gpuPortHostValues(np * GPU_MATTER_VALUES);
    perm = gpuPortHostOrder(np);
    /* the CPU's loop: a lost particle trades places with the last one, which is tracked next */
    i = 0;
    top = np - 1;
    while (i <= top) {
      row = perm[i];
      v = values + row * GPU_MATTER_VALUES;
      v[4] = v[5] = 0;
      if (matter->energyDecay) {
        P = (1 + delta[row]) * Po;
        gamma = sqrt(sqr(P) + 1);
        if (multipleScattering == 0 && gamma <= 1) {
          /* lost in the first section: the CPU's case */
          random_2_set_state(state);
          return gpuMatterOnCpu(np, iPass, matter, Po, accepted, z0, "MATTER with a particle at rest");
        }
      }
      if (deviceDraws && multipleScattering) {
        /* where this particle's numbers start, and past them */
        v[0] = (double)GPU_PORT_NEXT_STATE(S);
        S = (S * drawStride) & GPU_DLARAN_MASK;
      } else if (deviceDraws) {
        /* the CPU's draw for each section, and an event's three numbers made on the device */
        v[0] = nEvents;
        for (is = count = 0; is < sections0; is++) {
          S = GPU_PORT_NEXT_STATE(S);
          if (S * GPU_DLARAN_SCALE < prob) {
            double *e;
            if (eventCapacity <= nEvents) {
              eventCapacity = eventCapacity ? 2 * eventCapacity : 1024;
              if (!(events = (double *)realloc(events, eventCapacity * GPU_MATTER_EVENT_VALUES * sizeof(*events))))
                gpuRequiredFailure("unable to allocate the MATTER events");
            }
            e = events + nEvents * GPU_MATTER_EVENT_VALUES;
            e[0] = is;
            e[1] = (double)GPU_PORT_NEXT_STATE(S);
            e[2] = e[3] = 0;
            S = (S * drawStride) & GPU_DLARAN_MASK;
            nEvents++;
            count++;
          }
        }
        v[1] = count;
      } else if (multipleScattering == 1) {
        v[0] = gauss_rn(0, random_2);
        v[1] = gauss_rn(0, random_2);
        v[2] = gauss_rn(0, random_2);
        v[3] = gauss_rn(0, random_2);
      } else if (multipleScattering == 2) {
        v[0] = gauss_rn(0, random_2);
        v[1] = gauss_rn(0, random_2);
      } else {
        v[0] = nEvents;
        for (is = count = 0; is < sections0; is++) {
          if (random_2(1) < prob) {
            double zs, F, theta, phi, *e;
            zs = L1 * random_2(1);
            F = random_2(1);
            theta = sqrt((1 - F) * K2 * GPU_MATTER_SQR_PI / (K2 + F * GPU_MATTER_SQR_PI));
            phi = random_2(1) * PIx2;
            if (eventCapacity <= nEvents) {
              eventCapacity = eventCapacity ? 2 * eventCapacity : 1024;
              if (!(events = (double *)realloc(events, eventCapacity * GPU_MATTER_EVENT_VALUES * sizeof(*events))))
                gpuRequiredFailure("unable to allocate the MATTER events");
            }
            e = events + nEvents * GPU_MATTER_EVENT_VALUES;
            e[0] = is;
            e[1] = zs;
            e[2] = theta * sin(phi);
            e[3] = theta * cos(phi);
            nEvents++;
            count++;
          }
        }
        v[1] = count;
      }
      if (matter->energyDecay) {
        dgamma = gamma * dGammaFactor;
        if (matter->energyStraggle) {
          /* gauss_rn() itself, from where the walk is */
          if (deviceDraws)
            random_2_set_state(S);
          while ((dgamma1 = dgamma * (1 + 0.5 * gauss_rn(0, random_2))) < 0)
            ;
          if (deviceDraws)
            S = random_2_get_state();
          dgamma = dgamma1;
        }
        v[4] = dgamma;
        if (gamma - dgamma <= 1)
          v[5] = 1;
      }
      if (v[5]) {
        perm[i] = perm[top];
        perm[top] = row;
        top--;
      } else
        i++;
    }
    if (deviceDraws)
      random_2_set_state(S);
  }
  nLeft = top + 1;

  d.deviceDraws = (int)deviceDraws;
  d.K2 = K2;
  d.L = L;
  d.L1 = L1;
  d.sections = sections0;
  d.thetaRms = theta_rms;
  d.Po = Po;
  d.z0 = z0;
  d.multipleScattering = (int)multipleScattering;
  d.energyDecay = matter->energyDecay != 0;
  status = gpuCudaMatter(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &d, values, events, nEvents,
                         &milliseconds);
  if (status != 0)
    gpuFatalStatus("MATTER CUDA kernel", status);
  gpuRecordApertureKernel(milliseconds);
  gpuPortCounters(counters, "MATTER counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  if (nLeft < np)
    gpuPortGather(np, perm, "MATTER loss order");
  gpuMarkDeviceChanged(nLeft);
  if (nLeft < np)
    gpuCopyApertureLossRowsToHost(nLeft, np, "MATTER losses");
#ifdef GPU_VERIFY
  {
    unsigned long long after = random_2_get_state();
    long cpuLeft, saved;
    random_2_set_state(state);
    startCpuTimer();
    saved = gpu_port_host_begin();
    cpuLeft = track_through_matter(gpuBase.coord, np, iPass, matter, Po, NULL, z0);
    gpu_port_host_end(saved);
    compareGpuCpuValue((double)after, (double)random_2_get_state(), 0, "MATTER random numbers used");
    gpuPortVerifyLosses(nLeft, cpuLeft, np, "track_through_matter");
  }
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* PEPPOT: pepper_pot_plate().  One pass of its loop on the first nParticles rows: the device's hole test
 * (after the drift, on the second pass), the host's random numbers in the CPU's order, the kicks, then
 * the CPU's order; the particles left. */
static long gpuPeppotPass(long nParticles, GPU_PEPPOT_DATA *data, PEPPOT *peppot, double transmission, long seed,
                          const GPU_INCREMENT_DATA *kick) {
  static unsigned char *flags = NULL;
  static long capacity = 0;
  double *values, *v, scatter, rotate;
  long long *perm, row;
  long i, top, kicks = 0;
  int status;

  if (capacity < nParticles) {
    free(flags);
    if (!(flags = (unsigned char *)malloc(nParticles * sizeof(*flags))))
      gpuRequiredFailure("unable to allocate the PEPPOT flags");
    capacity = nParticles;
  }
  status = gpuCudaPeppot(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, data, peppot->x, peppot->y, flags);
  if (status != 0)
    gpuFatalStatus("PEPPOT CUDA kernel", status);
  values = gpuPortHostValues(nParticles * GPU_INCREMENT_VALUES);
  perm = gpuPortHostOrder(nParticles);
  for (i = 0; i < nParticles; i++)
    values[i * GPU_INCREMENT_VALUES] = 0;
  i = 0;
  top = nParticles - 1;
  while (i <= top) {
    row = perm[i];
    if (flags[row]) {
      if (transmission == 0 || random_2(seed) > transmission) {
        perm[i] = perm[top];
        perm[top] = row;
        top--;
        continue;
      }
      if (transmission && peppot->theta_rms) {
        /* particle made it through material--scatter it */
        scatter = gauss_rn(0, random_2) * peppot->theta_rms;
        rotate = random_2(1) * PIx2;
        v = values + row * GPU_INCREMENT_VALUES;
        v[0] = 1;
        v[2] = tan(scatter * cos(rotate));
        v[4] = tan(scatter * sin(rotate));
        kicks++;
      }
    }
    i++;
  }
  if (kicks)
    gpuPortIncrement(nParticles, kick, values, "PEPPOT kicks");
  if (top + 1 < nParticles)
    gpuPortGather(nParticles, perm, "PEPPOT loss order");
  return top + 1;
}

long gpu_track_peppot(double **part, void *element, long np, double **accepted) {
  PEPPOT *peppot = (PEPPOT *)element;
  GPU_PEPPOT_DATA data;
  GPU_INCREMENT_DATA kick;
  double transmission;
  long nLeft;
#ifdef GPU_VERIFY
  unsigned long long state;
#endif

  (void)part;
  if (np <= 0)
    return np;
  if (accepted || globalLossCoordOffset > 0) {
    nLeft = pepper_pot_plate(gpuPortCpuCoord("PEPPOT with an accepted beam or global loss coordinates"), peppot, np,
                             accepted);
    gpuPortCpuDone();
    return nLeft;
  }
  startGpuTimer();
#ifdef GPU_VERIFY
  state = random_2_get_state();
#endif
  if (peppot->length)
    transmission = sqrt(peppot->transmission);
  else
    transmission = peppot->transmission;
  memset(&data, 0, sizeof(data));
  data.nHoles = peppot->n_holes;
  data.radius2 = sqr(peppot->radii);
  memset(&kick, 0, sizeof(kick));
  kick.active[1] = kick.active[3] = 1;
  gpuCopyHostToDevice(np);
  nLeft = gpuPeppotPass(np, &data, peppot, transmission, 0, &kick);
  if (nLeft > 0 && peppot->length > 0) {
    data.length = peppot->length;
    nLeft = gpuPeppotPass(nLeft, &data, peppot, transmission, 1, &kick);
  }
  gpuMarkDeviceChanged(nLeft);
  if (nLeft < np)
    gpuCopyApertureLossRowsToHost(nLeft, np, "PEPPOT losses");
#ifdef GPU_VERIFY
  random_2_set_state(state);
  startCpuTimer();
  gpuPortVerifyLosses(nLeft, pepper_pot_plate(gpuBase.coord, peppot, np, NULL), np, "pepper_pot_plate");
#endif
  gpuRecordWallSeconds();
  return nLeft;
}

/* After the host changed some columns of a beam whose current copy it holds (CBSCAT): those columns to
 * the device */
void gpu_port_upload_columns(long np, const long *columns, long nColumns) {
  long i;

  for (i = 0; i < nColumns; i++)
    gpu_port_upload_column(np, columns[i]);
}

/* ---------------------------------------------------------------------------------------------------- */
/* IONEFFECTS: the ions on the host as elegant tracks them, on a current host copy; the Gaussian field's
 * kicks to the electrons on the device (ionEffects.cc calls here), other fields' kicks on the host copy
 * with x' and y' sent back (do_tracking.c) */

extern int gpuCudaIonKicks(void *coord, long nRows, int stride, const long long *hostRows,
                           const GPU_BEAMBEAM_DATA *data, float *milliseconds);

/* gaussianBeamKick(part[ip], ionCentroid, ionSigma, 0, kick, qIon, me_mks, 1), then
 * part[ip][1] += kick[0] / c_mks / Po and the same for y' */
void gpu_ioneffects_kicks(const int64_t *rows, long np, const double *centroid, const double *sigma, double charge,
                          double Po) {
  GPU_BEAMBEAM_DATA data;
  double ionCharge = 1;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return;
  memset(&data, 0, sizeof(data));
  data.centroid[0] = centroid[0];
  data.centroid[1] = centroid[1];
  data.size[0] = sigma[0];
  data.size[1] = sigma[1];
  data.charge = charge;
  data.P0 = Po;
  data.mass = me_mks;
  data.inverseMass = 1.0 / me_mks;
  data.C1 = c_mks * charge * re_mks * me_mks * ionCharge / e_mks;
  data.C1round = 2 * c_mks * charge * re_mks * me_mks * ionCharge / e_mks;
  data.twoPi = 2 * PI;
  data.cMks = c_mks;
  /* the device holds the beam the host copied from (copyParticlesToCpuReadOnly()) */
  status = gpuCudaIonKicks(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, (const long long *)rows, &data,
                           &milliseconds);
  if (status != 0)
    gpuFatalStatus("IONEFFECTS CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
}

/* After the device changed the beam whose host copy an element was reading (IONEFFECTS) */
void gpu_port_device_changed(long np) {
  gpuMarkDeviceChanged(np);
}

/* ---------------------------------------------------------------------------------------------------- */
/* BUMPER (KICKER): track_through_kicker()'s set-up and phase reference on the host (the fiducial time is its
 * sum over a read-only copy, in particle order), the particle loop on the device */

void set_up_kicker(KICKER *kicker);
#pragma weak set_up_kicker
#pragma weak track_through_kicker
extern int gpuCudaKicker(void *coord, long nParticles, int stride, const GPU_KICKER_DATA *data,
                         const double *tWaveform, const double *ampWaveform, const double *xpFactor,
                         const double *ypFactor, float *milliseconds);

static long gpuKickerSupported(ELEMENT_LIST *eptr) {
  (void)eptr;
  /* the spin rotation is made per particle from its kicks */
  return !spinCoordOffset;
}

long gpu_track_kicker(double **particle, long nParticles, void *element, double pCentral, long pass,
                      long defaultOrder) {
  KICKER *kicker = (KICKER *)element;
  GPU_KICKER_DATA data;
  unsigned long long counters[4], i;
  double timeOffset;
  float milliseconds = 0;
  int status;

  if (!kicker)
    gpuRequiredFailure("NULL BUMPER CUDA tracking data");
  if (!kicker->t_wf) {
    set_up_kicker(kicker);
    if (fabs(kicker->time_offset) > 100 * (kicker->tmax - kicker->tmin))
      printWarningForTracking("BUMPER time offset is large", "The TIME_OFFSET value for is much larger than the waveform length. Consider use of the FIRE_ON_PASS parameter to prevent problems with round-off error.");
  }
  if (nParticles <= 0)
    return 1;
  if (kicker->fire_on_pass > pass) {
    /* track_particles() keeps the drift on the device */
    drift_beam(particle, nParticles, kicker->length, defaultOrder);
    return 1;
  }
  if (!kicker->t_wf || !kicker->amp_wf || !kicker->n_wf || kicker->tmin >= kicker->tmax)
    return 0;
  if (kicker->deflectionMap && (!kicker->xpFactor || !kicker->ypFactor || kicker->nx < 1 || kicker->ny < 1))
    return 0;
  if (kicker->phase_reference == 0)
    kicker->phase_reference = unused_phase_reference();
  switch (get_phase_reference(&timeOffset, kicker->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!kicker->fiducial_seen) {
      double **part = copyParticlesToCpuReadOnly("BUMPER fiducial time");
      double t0 = 0;
      long ip;

      for (ip = 0; ip < nParticles; ip++)
        t0 += part[ip][4] / (c_mks * beta_from_delta(pCentral, part[ip][5]));
      t0 /= nParticles;
      t0 += trackingClockOffset();
      kicker->t_fiducial = t0;
      kicker->fiducial_seen = 1;
    }
    set_phase_reference(kicker->phase_reference, timeOffset = kicker->t_fiducial);
    break;
  default:
    return 0;
  }
  if (kicker->length < 0)
    return 1;
  timeOffset += kicker->time_offset;

  memset(&data, 0, sizeof(data));
  data.nWaveform = kicker->n_wf;
  data.tmin = kicker->tmin;
  data.tmax = kicker->tmax;
  data.angle = kicker->angle;
  data.length = kicker->length;
  data.b2 = kicker->b2;
  data.nKicks = kicker->n_kicks;
  data.pCentral = pCentral;
  data.cMks = c_mks;
  data.clockOffset = trackingClockOffset();
  data.timeOffset = timeOffset;
  data.periodic = kicker->periodic ? 1 : 0;
  data.doOffset = kicker->dx || kicker->dy || kicker->dz;
  data.dx = kicker->dx;
  data.dy = kicker->dy;
  data.dz = kicker->dz;
  data.doTilt = kicker->tilt != 0 && gpuPortRotation(kicker->tilt, &data.cosTilt, &data.sinTilt);
  if (data.doTilt)
    gpuPortRotation(-kicker->tilt, &data.cosUntilt, &data.sinUntilt);
  if ((data.deflectionMap = kicker->deflectionMap && kicker->n_kicks > 0)) {
    data.nx = kicker->nx;
    data.ny = kicker->ny;
    data.mapValues = kicker->nx * kicker->ny;
    data.xmin = kicker->xmin;
    data.ymin = kicker->ymin;
    data.dxg = kicker->dxg;
    data.dyg = kicker->dyg;
  }

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaKicker(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, kicker->t_wf,
                         kicker->amp_wf, kicker->xpFactor, kicker->ypFactor, &milliseconds);
  if (status != 0)
    gpuFatalStatus("BUMPER CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPortCounters(counters, "BUMPER counters");
  if (counters[3])
    gpuPortInvalidPending = 1;
  if (counters[2] != GPU_PORT_NO_ERROR) {
    double time = 0;
    status = gpuCudaPortValue((long)counters[2], &time);
    if (status != 0)
      gpuFatalStatus("BUMPER error time", status);
    printf("error: waveform interpolation problem in track_through_kicker()\n");
    fflush(stdout);
    printf("particle time is %21.15e\nwaveform limits are %21.15e to %21.15e\n", time, kicker->tmin, kicker->tmax);
    fflush(stdout);
    exitElegant(1);
  }
  gpuMarkDeviceChanged(nParticles);
#ifdef GPU_VERIFY
  startCpuTimer();
  track_through_kicker(gpuBase.coord, nParticles, kicker, pCentral, pass, defaultOrder);
  compareGpuCpu(nParticles, "track_through_kicker");
#endif
  (void)i;
  gpuRecordWallSeconds();
  return 1;
}

/* ---------------------------------------------------------------------------------------------------- */
/* SLICE: dump_slice_analysis() (slicePoint.c) with its numbers from the device */

extern int gpuCudaSlicePointTimes(const void *coord, long nParticles, int stride, double Po, double cMks,
                                  double clockOffset, int allPIDs, double startPID, double endPID, double *tMin,
                                  double *tMax, float *milliseconds);
extern int gpuCudaSlicePointMembers(const void *coord, long nParticles, int stride, int allPIDs, double startPID,
                                    double endPID, const double *t0, double dt0, long nSlices, long *count,
                                    long long *start, const double **members, long long *pairs, float *milliseconds);

static long gpuSlicePointUsable(long nParticles) {
  return gpuBase.initialized && gpuBase.activeDevice >= 0 && gpuBase.deviceCoord && nParticles > 0 &&
         nParticles == gpuBase.nParticles && nParticles < INT_MAX;
}

long gpu_slice_point_times(long nParticles, double Po, double clockOffset, long startPID, long endPID,
                           double *tMin, double *tMax) {
  float milliseconds = 0;
  int status;

  if (!gpuSlicePointUsable(nParticles))
    return 0;
  /* the device's copy, with any deferred drift applied */
  gpuCopyHostToDevice(nParticles);
  /* dump_slice_analysis() times the particles in its PID range, all of them when startPID >= endPID */
  status = gpuCudaSlicePointTimes(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, Po, c_mks,
                                  clockOffset, startPID >= endPID, (double)startPID, (double)endPID, tMin, tMax,
                                  &milliseconds);
  if (status != 0)
    gpuFatalStatus("SLICE times CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  return 1;
}

/* A range of slices' sums from the device's members, each in particle order with KahanPlus()'s steps as
 * accumulate_beam_sums() adds them: first the seven columns, then, about the means it divides out, the
 * BEAM_SUMS_SPARSE products.  A slice's chains are independent, so they run in one pass over its members. */
typedef struct {
  const double *members;
  long long pairs;
  const long long *start;
  const long *count;
  double *columnSum, *sigmaSum;
} GPU_SLICE_HOST_SUMS;

static void gpuSlicePointSumRange(long first, long last, void *context) {
  static const int pair[GPU_SLICE_SIGMA_PAIRS][2] = GPU_SLICE_SIGMA_PAIR_LIST;
  GPU_SLICE_HOST_SUMS *sums = (GPU_SLICE_HOST_SUMS *)context;
  long slice, m, n;
  int k, p;

  for (slice = first; slice < last; slice++) {
    const double *column[GPU_SLICE_COLUMNS];
    double sum[GPU_SLICE_COLUMNS], error[GPU_SLICE_COLUMNS], mean[GPU_SLICE_COLUMNS];
    double S[GPU_SLICE_SIGMA_PAIRS], E[GPU_SLICE_SIGMA_PAIRS];

    n = sums->count[slice];
    for (k = 0; k < GPU_SLICE_COLUMNS; k++) {
      column[k] = sums->members ? sums->members + k * sums->pairs + sums->start[slice] : NULL;
      sum[k] = error[k] = 0;
    }
    for (m = 0; m < n; m++)
      for (k = 0; k < GPU_SLICE_COLUMNS; k++) {
        double Y = column[k][m] + error[k], oldSum = sum[k];
        sum[k] = oldSum + Y;
        error[k] = Y - (sum[k] - oldSum);
      }
    /* the means as accumulate_beam_sums() divides them (an empty slice keeps its sum, 0) */
    for (k = 0; k < GPU_SLICE_COLUMNS; k++) {
      sums->columnSum[slice * GPU_SLICE_COLUMNS + k] = sum[k];
      mean[k] = sum[k];
      if (n)
        mean[k] /= n;
    }
    for (p = 0; p < GPU_SLICE_SIGMA_PAIRS; p++)
      S[p] = E[p] = 0;
    for (m = 0; m < n; m++)
      for (p = 0; p < GPU_SLICE_SIGMA_PAIRS; p++) {
        int i = pair[p][0], j = pair[p][1];
        double b = (column[i][m] - mean[i]) * (column[j][m] - mean[j]);
        double Y = b + E[p], oldSum = S[p];
        S[p] = oldSum + Y;
        E[p] = Y - (S[p] - oldSum);
      }
    for (p = 0; p < GPU_SLICE_SIGMA_PAIRS; p++)
      sums->sigmaSum[slice * GPU_SLICE_SIGMA_PAIRS + p] = S[p];
  }
}

long gpu_slice_point_sums(long nParticles, long startPID, long endPID, long nSlices, const double *t0, double dt0,
                          long *count, double *columnSum, double *sigmaSum) {
  GPU_SLICE_HOST_SUMS sums;
  float milliseconds = 0;
  long long pairs = 0, *start;
  const double *members = NULL;
  int status;

  if (!gpuSlicePointUsable(nParticles) || nSlices <= 0 || nSlices >= INT_MAX)
    return 0;
  if (!(start = (long long *)malloc(nSlices * sizeof(*start))))
    gpuRequiredFailure("unable to allocate the SLICE starts");
  /* accumulate_beam_sums() takes every PID when both limits are negative */
  status = gpuCudaSlicePointMembers(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                                    startPID < 0 && endPID < 0, (double)startPID, (double)endPID, t0, dt0, nSlices,
                                    count, start, &members, &pairs, &milliseconds);
  if (status == 1) { /* cudaErrorInvalidValue: more memberships than the sort takes */
    free(start);
    return 0;
  }
  if (status != 0)
    gpuFatalStatus("SLICE members CUDA kernels", status);
  gpuRecordHelperKernel(milliseconds);
  sums.members = members;
  sums.pairs = pairs;
  sums.start = start;
  sums.count = count;
  sums.columnSum = columnSum;
  sums.sigmaSum = sigmaSum;
  gpuHostParallelFor(nSlices, pairs >= GPU_HOST_THREAD_MIN_GAUSSIANS ? gpuHostThreads() : 1, gpuSlicePointSumRange,
                     &sums);
  free(start);
  return 1;
}
