#ifndef GPU_BASE_H
#define GPU_BASE_H

#ifdef __cplusplus
extern "C" {
#endif

#define ELEGANT_GPU_DEFAULT_MODE "auto"
#define ELEGANT_GPU_DEFAULT_MIN_PARTICLES 10000L

#ifndef GPU_BUNCHED_WAKE_UNSUPPORTED
#  define GPU_BUNCHED_WAKE_UNSUPPORTED 0
#  define GPU_BUNCHED_WAKE_TRACK 1
#  define GPU_BUNCHED_WAKE_SKIP 2
#  define GPU_BUNCHED_WAKE_MATCH_ONLY 3
#endif

typedef struct GPU_BASE {
  double **coord;
  double **accepted;
  double **lostPart;
  long nOriginal;
  long nParticles;
  long isMaster;
  long lossOutputNeeded;
  long orderSensitiveOutputNeeded;
  long reductionOutputNeeded;
  long backtrack;
  long elementOnGpu;
  long initialized;
  long deviceCount;
  long activeDevice;
  long requiredMode;
  long verifyMode;
  long minParticles;
  long matrixMinParticles;
  long helperMinParticles;
  long exactDriftMinParticles;
  long reductionMinParticles;
  long apertureMinParticles;
  long magnetMinParticles;
  long wakeMinParticles;
  long lscMinParticles;
  long csrMinParticles;
  long csrMinBins;
  long scmultMinParticles;
  void *element;
  void *hostCoordBase;
  void *deviceCoord;
  long deviceCapacity;
  long deviceStride;
  long deviceCurrent;
  long hostCurrent;
  long gpuElementCount;
  long gpuTrackParticleCount;
  long gpuExactDriftCount;
  long gpuLinearDriftCount;
  long gpuHelperCount;
  long gpuReductionCount;
  long gpuApertureCount;
  long gpuMagnetCount;
  long gpuWakeCount;
  long gpuLscCount;
  long gpuStandaloneLscCount;
  long gpuRfcwLscKickOnlyCount;
  long gpuRfcwLscFullCount;
  long gpuCsrCount;
  long gpuScmultCount;
  long gpuPassiveElementCount;
  double gpuKernelSeconds;
  double gpuTransferToDeviceSeconds;
  double gpuTransferToHostSeconds;
  double gpuWallSeconds;
  long gpuSyncRequestCount;
  long gpuSyncCopyCount;
  long gpuSyncReadOnlyCount;
  long gpuSyncMutableCount;
  long gpuSyncOutputCount;
  long gpuSyncCpuElementCount;
  long gpuSyncApertureLossCount;
  long gpuSyncMpiCount;
  long gpuSyncVerificationCount;
  long gpuSyncCollectiveCount;
  long gpuSyncReductionCount;
  long gpuSyncDeallocCount;
  long gpuSyncOtherCount;
  long gpuShortGpuIslandCpuCount;
} GPU_BASE;

typedef struct GPU_MATRIX_DATA {
  int order;
  int useSReference;
  double sReference;
  double C[6];
  double R[36];
  double T[126];
  double Q[336];
} GPU_MATRIX_DATA;

typedef struct GPU_BEAM_SUM_DATA {
  long count;
  double centroidSum[7];
  double productSum[28];
  double maxabs[7];
  double min[7];
  double max[7];
} GPU_BEAM_SUM_DATA;

typedef struct GPU_LONG_MIN_MAX_DATA {
  long count;
  long min;
  long max;
} GPU_LONG_MIN_MAX_DATA;

typedef struct GPU_DOUBLE_MIN_MAX_DATA {
  long count;
  double min;
  double max;
} GPU_DOUBLE_MIN_MAX_DATA;

typedef struct GPU_MULTIPOLE_DATA {
  long nSlices;
  int integrationOrder;
  int expandHamiltonian;
  int initialSlopeRoundTrip;
  int radiationBlock;
  double drift;
  double Po;
  double radCoef;
  double coordLimit;
  double slopeLimit;
  double KnL[3];
  long order[3];
  int skew[3];
  double xkick;
  double ykick;
  double dx;
  double dy;
  double dz;
  double cosTilt;
  double sinTilt;
} GPU_MULTIPOLE_DATA;

typedef struct GPU_CSBEND_DATA {
  long nSlices;
  int integrationOrder;
  int expandHamiltonian;
  int hasSkew;
  int hasNormal;
  int expansionOrder1;
  double length;
  double rho0;
  double rhoActual;
  double Po;
  double radCoef;
  double cosTilt;
  double sinTilt;
  int hasMisalignment;
  double dxi;
  double dyi;
  double dzi;
  double dxf;
  double dyf;
  double dzf;
  double dcoordEtilt[5];
  int edge1;
  int edge2;
  int edgeOrder;
  double e1;
  double e2;
  double he1;
  double he2;
  double psi1;
  double psi2;
  double fieldIndex;
  double edgeKickLimit1;
  double edgeKickLimit2;
  double coordLimit;
  double slopeLimit;
  double Fx[121];
  double Fy[121];
} GPU_CSBEND_DATA;

typedef struct GPU_WAKE_LONGITUDINAL_DATA {
  long bins;
  long wakePoints;
  long i0;
  int interpolate;
  int useBunchFilter;
  int bunchIndexColumn;
  long selectedBunch;
  double tmin;
  double dt;
  double pCentral;
  double factor;
  double particleMassMV;
  double particleRelSign;
  double cMks;
} GPU_WAKE_LONGITUDINAL_DATA;

typedef struct GPU_TRWAKE_DATA {
  long bins;
  long wakePoints;
  long i0;
  int interpolate;
  int useBunchFilter;
  int bunchIndexColumn;
  long selectedBunch;
  int hasWake[2];
  long driveExponent[2];
  long probeExponent[2];
  double tmin;
  double dt;
  double pCentral;
  double factor[2];
  double offset[2];
  int hasTilt;
  double cosTilt;
  double sinTilt;
  double particleMassMV;
  double particleRelSign;
  double cMks;
} GPU_TRWAKE_DATA;

#define GPU_COMBINED_WAKE_CHANNELS 7
#define GPU_COMBINED_WAKE_MODE_TIME 0
#define GPU_COMBINED_WAKE_MODE_IMPEDANCE 1

typedef struct GPU_COMBINED_WAKE_DATA {
  long bins;
  long tablePoints;
  long i0;
  int mode;
  int interpolate;
  int allowTimeFft;
  int useBunchFilter;
  int bunchIndexColumn;
  long selectedBunch;
  int enabled[GPU_COMBINED_WAKE_CHANNELS];
  int driver[GPU_COMBINED_WAKE_CHANNELS];
  int kickPlane[GPU_COMBINED_WAKE_CHANNELS];
  long probeExponent[GPU_COMBINED_WAKE_CHANNELS];
  double factor[GPU_COMBINED_WAKE_CHANNELS];
  double tmin;
  double dt;
  double pCentral;
  double offset[2];
  double particleMassMV;
  double particleRelSign;
  double cMks;
} GPU_COMBINED_WAKE_DATA;

typedef struct GPU_LSC_DATA {
  long bins;
  int interpolate;
  int doDrift;
  int backtrack;
  double tmin;
  double dt;
  double pCentral;
  double length;
  double particleMassMV;
  double particleRelSign;
  double cMks;
} GPU_LSC_DATA;

typedef struct GPU_SCMULT_LINEAR_DATA {
  int horizontal;
  int vertical;
  int uniformDistribution;
  double charge;
  double c1;
  double center[3];
  double sigma[3];
  double dmux;
  double dmuy;
  double betax;
  double betay;
} GPU_SCMULT_LINEAR_DATA;

typedef struct GPU_POLYNOMIAL_SERIES_DATA {
  long coordinateOffset[7];
  long totalTerms;
  double tilt;
  double dx;
  double dy;
  double dz;
  double pCentral;
  double coordinateLimit;
  double slopeLimit;
} GPU_POLYNOMIAL_SERIES_DATA;

typedef struct GPU_RFDF_DATA {
  long nKicks;
  long startPID;
  long endPID;
  int standingWave;
  int magneticDeflection;
  double length;
  double pCentral;
  double omega;
  double k;
  double ePhase;
  double dtLight;
  double eStrength;
  double b2;
  double cMks;
} GPU_RFDF_DATA;

typedef struct GPU_SREFFECTS_DATA {
  int lossOnly;
  int includeOffsets;
  double Fx;
  double Fy;
  double Fdelta;
  double Ddelta;
  double pCentral;
  double etax;
  double etapx;
  double etay;
  double etapy;
} GPU_SREFFECTS_DATA;

typedef struct GPU_BGGEXP_DATA {
  const void *tableOwner;
  unsigned long long tableSignature;
  long nz;
  long termCount;
  const int *m;
  const int *gradient;
  const int *radialPower;
  const double *coefficient;
  const double *multipoleFactor;
  const double *const *Cmn;
  const double *const *dCmnDz;
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
} GPU_BGGEXP_DATA;

typedef struct GPU_CWIGGLER_DATA {
  long periods;
  long stepsPerPeriod;
  int integrationOrder;
  int hasHorizontal;
  int hasVertical;
  double length;
  double periodLength;
  double kw;
  double horizontalCoefficient;
  double verticalCoefficient;
  double horizontalPhase;
  double verticalPhase;
  double zStartHorizontal;
  double zEndHorizontal;
  double zStartVertical;
  double zEndVertical;
  double poleFactor[3];
  double z1;
  double z2;
  double z3;
  double z4;
  double z5;
  double z6;
} GPU_CWIGGLER_DATA;

typedef struct GPU_FTABLE_DATA {
  const void *tableOwner;
  long dimensions[3];
  const double *field[3];
  double minimum[3];
  double maximum[3];
  double spacing[3];
  long nKicks;
  double length;
  double factor;
  double threshold;
  double pCentral;
  double eomc;
} GPU_FTABLE_DATA;

typedef struct GPU_KICKMAP_DATA {
  int undulator;
  long nKicks;
  long nx;
  long ny;
  double length;
  double halfLength;
  double xmin;
  double ymin;
  double dxg;
  double dyg;
  double kickScale;
  double pRef;
  double radCoef;
  double radiationKick;
} GPU_KICKMAP_DATA;

GPU_BASE *getGpuBase(void);
void gpuDescribeUsageSettings(char *buffer, unsigned long bufferSize);
void gpuBaseInit(double **coord, long nOriginal, double **accepted, double **lostPart,
                 long isMaster, long lossOutputNeeded,
                 long orderSensitiveOutputNeeded, long reductionOutputNeeded,
                 long alwaysChangeP0, long backtrack);
void gpuBaseDealloc(void);
void gpuDisableForRun(const char *reason);
void gpuSetTrackingSuppressed(long suppressed);
void setElementGpuData(void *eptr, long nParticles);
long getElementOnGpu(void);
void gpuSetCpuParticleArray(double **coord, long nParticles);
double **forceParticlesToCpu(const char *reason);
double **copyParticlesToCpuReadOnly(const char *reason);
long gpu_matrix_supported(void *M);
long gpu_reductions_enabled(long nParticles);
void startGpuTimer(void);
void startCpuTimer(void);
void displayTimings(void);
void compareGpuCpu(long nParticles, const char *label);
void copyReductionArrays(double *centroid, double *sigma);
void compareReductionArrays(double *centroid, double *sigma, void *sums, const char *label);
void *getGpuBeamSums(void *sums);
void sortByPID(long nParticles);

#ifdef __cplusplus
}
#endif

#endif
