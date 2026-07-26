#ifndef GPU_BASE_H
#define GPU_BASE_H

#ifdef __cplusplus
extern "C" {
#endif

#define ELEGANT_GPU_DEFAULT_MODE "auto"
#define ELEGANT_GPU_DEFAULT_MIN_PARTICLES 10000L

typedef struct GPU_OMP_TRACKING_WORKSPACE {
  unsigned char *survived;
  double *lossOffset;
  double *auxiliary;
  double **particleOrder;
  long capacity;
} GPU_OMP_TRACKING_WORKSPACE;

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
  long gpuScmultMomentCount;
  long gpuScmultMomentCacheHitCount;
  long gpuHistogramCount;
  long gpuPassiveElementCount;
  double gpuKernelSeconds;
  double gpuMatrixSeconds;
  double gpuReductionSeconds;
  double gpuScmultMomentSeconds;
  double gpuScmultKickSeconds;
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

#define GPU_MATRIX_MAX_TERMS 498

typedef struct GPU_MATRIX_TERM {
  double coefficient;
  unsigned char degree;
  unsigned char j;
  unsigned char k;
  unsigned char l;
} GPU_MATRIX_TERM;

typedef struct GPU_MATRIX_DATA {
  int order;
  int useSReference;
  double sReference;
  double C[6];
  double R[36];
  double T[126];
  int termOffset[7];
  int secondOrderOffset[37];
  GPU_MATRIX_TERM term[GPU_MATRIX_MAX_TERMS];
} GPU_MATRIX_DATA;

typedef struct GPU_BEAM_SUM_DATA {
  long count;
  double pSum;
  double gammaSum;
  double centroidSum[7];
  double productSum[28];
  double maxabs[7];
  double min[7];
  double max[7];
} GPU_BEAM_SUM_DATA;

typedef struct GPU_HISTOGRAM_RANGE_DATA {
  long count;
  double minimum[7];
  double maximum[7];
} GPU_HISTOGRAM_RANGE_DATA;

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

typedef struct GPU_APERTURE_LIMIT_DATA {
  int present;
  int elliptical;
  int xExponent;
  int yExponent;
  int openSide;
  double xMax;
  double yMax;
  double cosReverseTilt;
  double sinReverseTilt;
} GPU_APERTURE_LIMIT_DATA;

typedef struct GPU_MULTIPOLE_DATA {
  long nSlices;
  int integrationOrder;
  int expandHamiltonian;
  int initialSlopeRoundTrip;
  int radiationBlock;
  double drift;
  double Po;
  double lossZStart;
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
  GPU_APERTURE_LIMIT_DATA aperture;
  double endDrift;
  double k1;
  int edge1Effects;
  int edge2Effects;
  int edge1Linear;
  int edge2Linear;
  double edge1NonlinearFactor;
  double edge2NonlinearFactor;
  double edge1M1[8];
  double edge1M2[8];
  double edge2M1[8];
  double edge2M2[8];
} GPU_MULTIPOLE_DATA;

typedef struct GPU_EXACT_CORRECTOR_DATA {
  double length;
  double xkick;
  double ykick;
  double theta0;
  double rho0;
  double dx;
  double dy;
  double dz;
  double cosTilt;
  double sinTilt;
  double zStart;
  double pCentral;
} GPU_EXACT_CORRECTOR_DATA;

#define GPU_TAPER_APERTURE_CIRCULAR 1
#define GPU_TAPER_APERTURE_RECTANGULAR 2

typedef struct GPU_TAPER_APERTURE_DATA {
  int type;
  double length;
  double xStart;
  double xEnd;
  double yStart;
  double yEnd;
  double dx;
  double dy;
  double cosTilt;
  double sinTilt;
  double zStart;
  double pCentral;
} GPU_TAPER_APERTURE_DATA;

typedef struct GPU_SPEEDBUMP_DATA {
  double length;
  double chord;
  double dzCenter;
  double height;
  double position;
  double offset;
  double radius;
  double zStart;
  double pCentral;
  int plane;
  int plusDirection;
  int minusDirection;
  int scraperConvention;
} GPU_SPEEDBUMP_DATA;

#define GPU_BATCHED_APERTURE_MATRIX 1
#define GPU_BATCHED_APERTURE_MULTIPOLE 2
#define GPU_BATCHED_APERTURE_RCOL 3
#define GPU_BATCHED_APERTURE_MATRIX_DRIFT 4
#define GPU_BATCHED_APERTURE_MULTIPOLE_DRIFT 5

typedef struct GPU_BATCHED_APERTURE_DRIFT_DATA {
  double length;
  double coordLimit;
  double slopeLimit;
  int order;
  int expandHamiltonian;
} GPU_BATCHED_APERTURE_DRIFT_DATA;

typedef struct GPU_BATCHED_APERTURE_RCOL_DATA {
  double xmax;
  double ymax;
  double xCenter;
  double yCenter;
  double sStart;
} GPU_BATCHED_APERTURE_RCOL_DATA;

typedef union GPU_BATCHED_APERTURE_ELEMENT_DATA {
  GPU_MULTIPOLE_DATA multipole;
  GPU_BATCHED_APERTURE_RCOL_DATA rcol;
  GPU_BATCHED_APERTURE_DRIFT_DATA drift;
} GPU_BATCHED_APERTURE_ELEMENT_DATA;

typedef struct GPU_BATCHED_APERTURE_ELEMENT {
  int type;
  long elementIndex;
  GPU_BATCHED_APERTURE_ELEMENT_DATA data;
} GPU_BATCHED_APERTURE_ELEMENT;

/*
 * A tune program is rebuilt for each batched tune-tracking invocation after
 * element alteration and lazy aperture-file loading have completed.  The
 * opcode stream is deliberately small; immutable element data lives in
 * type-specific arrays so a fixed CUDA thread can follow one particle through
 * a complete turn without per-element launches, allocation, or compaction.
 */
#define GPU_TUNE_OP_NOP 0
#define GPU_TUNE_OP_EXACT_DRIFT 1
#define GPU_TUNE_OP_MATRIX 2
#define GPU_TUNE_OP_MULTIPOLE 3
#define GPU_TUNE_OP_EXACT_CORRECTOR 4
#define GPU_TUNE_OP_CSBEND 5
#define GPU_TUNE_OP_CCBEND 6
#define GPU_TUNE_OP_LGBEND 7
#define GPU_TUNE_OP_RCOL 8
#define GPU_TUNE_OP_ECOL 9
#define GPU_TUNE_OP_SCRAPER 10
#define GPU_TUNE_OP_TAPER_APERTURE 11
#define GPU_TUNE_OP_SPEEDBUMP 12

typedef struct GPU_TUNE_PROGRAM_OP {
  int opcode;
  int dataIndex;
  int auxiliaryIndex;
  int postApertureIndex;
  long elementIndex;
} GPU_TUNE_PROGRAM_OP;

typedef struct GPU_TUNE_DRIFT_DATA {
  double length;
} GPU_TUNE_DRIFT_DATA;

typedef struct GPU_TUNE_COLLIMATOR_DATA {
  double length;
  double xMax;
  double yMax;
  double xCenter;
  double yCenter;
  long xExponent;
  long yExponent;
  long openCode;
} GPU_TUNE_COLLIMATOR_DATA;

typedef struct GPU_TUNE_SCRAPER_DATA {
  double length;
  double center;
  double position;
  int plane;
  int sideSign;
  int secondSideSign;
} GPU_TUNE_SCRAPER_DATA;

typedef struct GPU_TUNE_PARTICLE_STATUS {
  long lossTurn;
  long lossElement;
  int alive;
} GPU_TUNE_PARTICLE_STATUS;

typedef struct GPU_CSBEND_DATA {
  long nSlices;
  int integrationOrder;
  int expandHamiltonian;
  int hasSkew;
  int hasNormal;
  int expansionOrder1;
  double length;
  double rho0;
  double invRho0;
  double rhoActual;
  double Po;
  double lossZStart;
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
  int edgeEffect1;
  int edgeEffect2;
  int edgeOrder;
  int edgeFlip;
  double e1;
  double e2;
  double K1;
  double fringeInt1[7];
  double fringeInt2[7];
  double he1;
  double he2;
  double psi1;
  double psi2;
  double fieldIndex;
  double edgeKickLimit1;
  double edgeKickLimit2;
  double coordLimit;
  double slopeLimit;
  GPU_APERTURE_LIMIT_DATA aperture;
  double Fx[121];
  double Fy[121];
} GPU_CSBEND_DATA;

typedef struct GPU_CCBEND_DATA {
  long nSlices;
  int integrationOrder;
  int fringeModel;
  int edgeOrder;
  int angleSign;
  double Po;
  double radCoef;
  int referenceCorrection;
  double chordLength;
  double angleHalf;
  double rho0;
  double KnL[3];
  double fringeInt1[8];
  double fringeInt2[8];
  double planeTan;
  double planeCos;
  double planeSin;
  double fringe1Tan;
  double fringe1Sin;
  double fringe1Sec;
  double fringe2Tan;
  double fringe2Sin;
  double fringe2Sec;
  double cosTilt;
  double sinTilt;
  double dxOffset;
  double xAdjust;
  double referenceTrajectory[5];
  double coordLimit;
  double slopeLimit;
  GPU_APERTURE_LIMIT_DATA aperture;
} GPU_CCBEND_DATA;

#define GPU_LGBEND_MAX_SEGMENTS 16
#define GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES 32

typedef struct GPU_LGBEND_SEGMENT_DATA {
  double length;
  double entryAngle;
  double exitAngle;
  double invRho;
  double K1;
  double KnL[3];
  double fringe1[8];
  double fringe2[8];
  double fringe1Tan;
  double fringe1Sin;
  double fringe1Sec;
  double fringe2Tan;
  double fringe2Sin;
  double fringe2Sec;
  int angleSign;
  int has1;
  int has2;
} GPU_LGBEND_SEGMENT_DATA;

typedef struct GPU_LGBEND_DATA {
  long nSegments;
  long nSlices;
  int integrationOrder;
  int edgeOrder;
  double predrift;
  double postdrift;
  double entryPosition;
  double entryAngle;
  double entryPlaneTan;
  double entryPlaneCos;
  double entryPlaneSin;
  double exitPosition;
  double exitAngle;
  double exitPlaneTan;
  double exitPlaneCos;
  double exitPlaneSin;
  double Po;
  double radCoef;
  double coordLimit;
  double slopeLimit;
  GPU_APERTURE_LIMIT_DATA aperture;
  GPU_LGBEND_SEGMENT_DATA segment[GPU_LGBEND_MAX_SEGMENTS];
} GPU_LGBEND_DATA;

typedef struct GPU_LGBEND_LOCAL_APERTURE_POINT {
  double xCenter;
  double yCenter;
  double xMax;
  double yMax;
} GPU_LGBEND_LOCAL_APERTURE_POINT;

/*
 * LGBEND aperture-data is sampled at each integration-slice boundary, just
 * as checkMultAperture() does on the CPU.  Keep this separate from
 * GPU_LGBEND_DATA so ordinary bends don't pay the per-element copy cost.
 */
typedef struct GPU_LGBEND_LOCAL_APERTURE_DATA {
  long nSegments;
  long nSlices;
  int present;
  GPU_LGBEND_LOCAL_APERTURE_POINT
    point[GPU_LGBEND_MAX_SEGMENTS]
         [GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES + 1];
} GPU_LGBEND_LOCAL_APERTURE_DATA;

typedef struct GPU_TUNE_PROGRAM {
  long opCount;
  long driftCount;
  long matrixCount;
  long multipoleCount;
  long exactCorrectorCount;
  long csbendCount;
  long ccbendCount;
  long lgbendCount;
  long lgbendLocalApertureCount;
  long collimatorCount;
  long scraperCount;
  long taperApertureCount;
  long speedbumpCount;
  long postApertureCount;
  long historyCertified;
  long lossCertified;
  GPU_TUNE_PROGRAM_OP *op;
  GPU_TUNE_DRIFT_DATA *drift;
  GPU_MATRIX_DATA *matrix;
  GPU_MULTIPOLE_DATA *multipole;
  GPU_EXACT_CORRECTOR_DATA *exactCorrector;
  GPU_CSBEND_DATA *csbend;
  GPU_CCBEND_DATA *ccbend;
  GPU_LGBEND_DATA *lgbend;
  GPU_LGBEND_LOCAL_APERTURE_DATA *lgbendLocalAperture;
  GPU_TUNE_COLLIMATOR_DATA *collimator;
  GPU_TUNE_SCRAPER_DATA *scraper;
  GPU_TAPER_APERTURE_DATA *taperAperture;
  GPU_SPEEDBUMP_DATA *speedbump;
  GPU_APERTURE_LIMIT_DATA *postAperture;
} GPU_TUNE_PROGRAM;

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

typedef struct GPU_RFMODE_DATA {
  long bins;
  long firstBin;
  long lastBin;
  int interpolate;
  long nCavities;
  double tmin;
  double dt;
  double pCentral;
  double particleMassMV;
  double particleRelSign;
  double cMks;
  double dx;
  double dy;
} GPU_RFMODE_DATA;

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
  int roundBeam;
  int swapXY;
  double charge;
  double c1;
  double center[3];
  double sigma[3];
  double dmux;
  double dmuy;
  double betax;
  double betay;
  double longitudinalScale;
  double roundSigma;
  double inverseSd;
  double minorMajorRatio;
  double majorMinorRatio;
  double inverseTwoMajorSigma2;
  double inverseTwoMinorSigma2;
  double kxScale;
  double kyScale;
} GPU_SCMULT_LINEAR_DATA;

typedef struct GPU_SCMULT_MOMENT_DATA {
  long count;
  double sum[3];
  double squareSum[3];
} GPU_SCMULT_MOMENT_DATA;

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
  int synchRad;
  double length;
  double periodLength;
  double kw;
  double horizontalCoefficient;
  double verticalCoefficient;
  double horizontalField;
  double verticalField;
  double pCentral;
  double srCoef;
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

typedef struct GPU_BMXYZ_DATA {
  const void *tableOwner;
  long dimensions[3];
  const double *field[3];
  double minimum[3];
  double spacing[3];
  double fieldLength;
  double integrationAccuracy;
  double strengthFactor;
  double fieldScale;
  int fieldIsMagnetic;
} GPU_BMXYZ_DATA;

#define GPU_LORENTZ_BMAPXY 1
#define GPU_LORENTZ_NIBEND 2
#define GPU_LORENTZ_NISEPT 3

typedef struct GPU_LORENTZ_DATA {
  int type;
  const void *tableOwner;
  long nx, ny;
  const double *field[2];
  double xmin, ymin, dx, dy;
  double length, integrationAccuracy;
  double strengthFactor, fieldScale;
  int fieldIsMagnetic;
  double angle, e1, e2, rho;
  double b1, q1Reference, q1Offset, fringeLength;
  double entranceSlope, entranceIntercept, fringeEntranceIntercept;
  double exitSlope, exitIntercept, fringeExitIntercept;
  double cosAlpha1, sinAlpha1, cosAlpha2, sinAlpha2;
} GPU_LORENTZ_DATA;

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
long gpuSetOmpTrackingThreads(long threads);
long gpuGetOmpTrackingThreads(void);
long gpuOmpTrackingEnabled(long particles);
long gpuOmpTrackingScopeActive(void);
GPU_OMP_TRACKING_WORKSPACE *gpuGetOmpTrackingWorkspace(long particles);
long gpuStableCompactParticles(double **particle, long particles,
                               const unsigned char *survived);
void gpuBaseInit(double **coord, long nOriginal, double **accepted, double **lostPart,
                 long isMaster, long lossOutputNeeded,
                 long orderSensitiveOutputNeeded, long reductionOutputNeeded,
                 long alwaysChangeP0, long backtrack);
void gpuBaseDealloc(void);
void gpuDisableForRun(const char *reason);
void gpuSetTrackingSuppressed(long suppressed);
void setElementGpuData(void *eptr, long nParticles);
long getElementOnGpu(void);
long gpu_track_through_exact_corrector(long nParticles, void *element,
                                       double pCentral, double **accepted,
                                       double zStart);
long gpu_track_through_taper_aperture(long nParticles, void *element,
                                      double pCentral, double **accepted,
                                      double zStart);
long gpu_track_through_speedbump(long nParticles, void *element,
                                 double pCentral, double **accepted,
                                 double zStart);
void gpuSetCpuParticleArray(double **coord, long nParticles);
double **forceParticlesToCpu(const char *reason);
double **copyParticlesToCpuReadOnly(const char *reason);
long gpu_matrix_supported(void *M);
long gpu_reductions_enabled(long nParticles);
long gpu_watch_parameters_supported(void *watch, long nParticles);
long gpu_watch_parameter_sums(long nParticles, long *count,
                              double *pSum, double *gammaSum);
long gpu_histogram_ranges(long nParticles, double pCentral,
                          long startPID, long endPID,
                          unsigned int coordinateMask,
                          double *minimum, double *maximum);
void gpu_histogram_bins(long nParticles, double pCentral,
                        long startPID, long endPID, long bins,
                        unsigned int coordinateMask,
                        double timeOffset,
                        const double *lower, const double *upper,
                        double *histogram);
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
