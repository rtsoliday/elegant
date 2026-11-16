#ifndef GPU_BASE_H
#define GPU_BASE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELEGANT_GPU_DEFAULT_MODE "auto"
#define ELEGANT_GPU_DEFAULT_MIN_PARTICLES 10000L

typedef struct GPU_OMP_TRACKING_WORKSPACE {
  unsigned char *survived;
  double *lossOffset;
  double *auxiliary;
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
  int64_t count;
  double pSum;
  double gammaSum;
  double centroidSum[7];
  double productSum[28];
  double maxabs[7];
  double min[7];
  double max[7];
} GPU_BEAM_SUM_DATA;

typedef struct GPU_HISTOGRAM_RANGE_DATA {
  int64_t count;
  double minimum[7];
  double maximum[7];
} GPU_HISTOGRAM_RANGE_DATA;

typedef struct GPU_LONG_MIN_MAX_DATA {
  int64_t count;
  long min;
  long max;
} GPU_LONG_MIN_MAX_DATA;

typedef struct GPU_DOUBLE_MIN_MAX_DATA {
  int64_t count;
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
  /* Set only by aperture_data files */
  double xCen;
  double yCen;
} GPU_APERTURE_LIMIT_DATA;

/* An APCONTOUR that stays in force through a magnet */
#define GPU_CONTOUR_LOGIC_AND 1
#define GPU_CONTOUR_LOGIC_OR 2
#define GPU_CONTOUR_LOGIC_NOT 3
#define GPU_CONTOUR_LOGIC_POP 4
#define GPU_CONTOUR_LOGIC_MAX_STACK 62
typedef struct GPU_APERTURE_CONTOUR_DATA {
  const double *x;   /* every contour's points, one contour after another */
  const double *y;
  const long *start; /* nContours+1 offsets into x and y */
  long nContours;
  const signed char *logic;  /* NULL, or each contour's GPU_CONTOUR_LOGIC_* operations in order */
  const long *logicStart;    /* nContours+1 offsets into logic */
  int invert;
  double dx;
  double dy;
  double dz; /* only the APCONTOUR element and its sticky checks; a magnet's check has no slope */
  double cosTilt;
  double sinTilt;
  double xFactor;
  double yFactor;
} GPU_APERTURE_CONTOUR_DATA;

/* First-order transforms of (x, qx, y, qy, s, delta) for MALIGN_METHOD 1 and 2, for the entrance [0] and exit [1] */
typedef struct GPU_LINEAR_MISALIGNMENT {
  double C[2][6];
  double R[2][6][6];
} GPU_LINEAR_MISALIGNMENT;

/* One face of the exact methods, in the layout of getExactMisalignmentFrame() */
typedef struct GPU_EXACT_MISALIGNMENT_FACE {
  double csi[3];
  double eta[3];
  double csiFinal[3];
  double etaFinal[3];
  double xAxis[3];
  double yAxis[3];
  double xFrame[3];
  double yFrame[3];
  double offset[3];
  double tanThetaW;
  double cosThetaW;
  double sinThetaW;
  double LD;
} GPU_EXACT_MISALIGNMENT_FACE;

typedef struct GPU_EXACT_MISALIGNMENT {
  /* Some CPU callers skip a face when there are no errors */
  int active[2];
  GPU_EXACT_MISALIGNMENT_FACE face[2];
} GPU_EXACT_MISALIGNMENT;

#define GPU_MULTIPOLE_MAX_TERM_ORDER 30
#define GPU_MULTIPOLE_MAX_TERMS 96

typedef struct GPU_MULTIPOLE_TERM {
  double strength;
  int order;
  int skew;
} GPU_MULTIPOLE_TERM;

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
  double isrCoef;
  const double *isrDraws; /* isrDrawsPerParticle per particle, in the order the CPU draws them */
  long isrDrawsPerParticle;
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
  GPU_APERTURE_CONTOUR_DATA contour;
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
  /* Terms from multipole files, in the order edge, steering, body */
  int nEdgeTerms;
  int nSteeringTerms;
  int nBodyTerms;
  int maxTermOrder;
  GPU_MULTIPOLE_TERM term[GPU_MULTIPOLE_MAX_TERMS];
  int hasLinearMisalignment;
  int hasExactMisalignment;
  union {
    GPU_LINEAR_MISALIGNMENT misalignment;
    GPU_EXACT_MISALIGNMENT exactMisalignment;
  };
  /* Spin, where the spin vector starts in the coordinates, and the constants
   * multipole_tracking2() passes to updateSpinQuaternionLocalFS() */
  long spinOffset;
  int spinRollOnly; /* MULT: multipole_tracking() rolls the spin with the tilt, nothing more */
  double spinRigidity;
  double spinCharge;
  double spinMassC;
  double spinAnomaly;
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
#define GPU_TAPER_APERTURE_ELLIPTICAL 3

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
  /* TAPERAPE: the semi-axes are xStart..xEnd and yStart..yEnd; a test that falls within a hair of the
   * boundary counts in *uncertain, and the call then goes to the CPU */
  double resolution, dadz, dbdz;
  int xExponent, yExponent;
  unsigned long long *uncertain;
  /* TAPERAPE with global loss coordinates: elegant converts a loss from the element's offset and tilted frame,
   * so x, x' and y there go into the three global columns from this one for the host to convert; else -1 */
  int localLossColumn;
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
#define GPU_BATCHED_APERTURE_APCONTOUR 6

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

/* An APCONTOUR element with no length; with HOLD_OFF it checks nothing */
typedef struct GPU_BATCHED_APERTURE_APCONTOUR_DATA {
  GPU_APERTURE_CONTOUR_DATA contour;
  double sStart;
  int check;
} GPU_BATCHED_APERTURE_APCONTOUR_DATA;

typedef union GPU_BATCHED_APERTURE_ELEMENT_DATA {
  GPU_MULTIPOLE_DATA multipole;
  GPU_BATCHED_APERTURE_RCOL_DATA rcol;
  GPU_BATCHED_APERTURE_DRIFT_DATA drift;
  GPU_BATCHED_APERTURE_APCONTOUR_DATA apContour;
} GPU_BATCHED_APERTURE_ELEMENT_DATA;

/* postContour: the sticky APCONTOUR that do_tracking() imposes after the element, at sEnd */
typedef struct GPU_BATCHED_APERTURE_ELEMENT {
  int type;
  long elementIndex;
  GPU_BATCHED_APERTURE_ELEMENT_DATA data;
  int postContourActive;
  GPU_APERTURE_CONTOUR_DATA postContour;
  double sEnd;
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
#define GPU_TUNE_OP_MALIGN 13

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

/* MALIGN, as offset_beam() applies it on pass onPass, or on every pass when onPass is -1 */
typedef struct GPU_TUNE_MALIGN_DATA {
  double dx, dxp, dy, dyp, dz, dt, dp, de;
  double pCentral, cMks;
  long onPass;
} GPU_TUNE_MALIGN_DATA;

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
  /* CSRCSBEND body slices: the rotation convertFromCSBendCoords() gives a lost particle's row */
  double lossCosTtilt;
  double lossSinTtilt;
  double radCoef;
  double cosTilt;
  double sinTilt;
  int hasMisalignment;
  int hasLinearMisalignment;
  int hasExactMisalignment;
  union {
    GPU_LINEAR_MISALIGNMENT misalignment;
    GPU_EXACT_MISALIGNMENT exactMisalignment;
  };
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
  /* Edge models 2 to 4: the full gap and FINT the CPU passes to dipoleFringeKHwang(), and each edge
   * angle's 1/cos, tan, sin and cos, taken on the host so they are the CPU's */
  double fringeGap;
  double fint1;
  double fint2;
  double edgeTrig1[4];
  double edgeTrig2[4];
  GPU_APERTURE_LIMIT_DATA aperture;
  GPU_APERTURE_CONTOUR_DATA contour;
  double isrCoef;
  const double *isrDraws; /* isrDrawsPerParticle per particle, in the order the CPU draws them */
  long isrDrawsPerParticle;
  /* Spin, where the spin vector starts in the coordinates, and the constants
   * integrate_csbend_ordn() passes to updateSpinQuaternionLocalFS() */
  long spinOffset;
  double spinFieldConversion;
  double spinH;
  double spinCharge;
  double spinMassC;
  double spinAnomaly;
  double Fx[121];
  double Fy[121];
  /* Sine, cosine and versine of each substep's drift angle, made on the host as integrate_csbend_ordn() makes
   * them (csbend.c: computeCsbendDriftSteps()), so that the two sides drift with the same numbers */
  double driftSin[8];
  double driftCos[8];
  double driftVers[8];
  /* The CPU divides QX and QY by 1+delta and multiplies them back after every kick, with radiation or without */
  int momentumRoundTrip;
} GPU_CSBEND_DATA;

/* Options of a CCBEND or LGBEND that only plain tracking sets, since batched programs refuse them */
typedef struct GPU_BEND_OPTIONS {
  double isrCoef;
  /* One per slice, in the order the CPU draws them.  The CPU tracks all particles through one LGBEND
   * segment before the next, so these go by segment, then particle: isrParticles per segment. */
  const double *isrDraws;
  long isrParticles;
  long isrFirstParticle;
  double *sliceRecord;    /* GPU_BEND_RECORD_SIZE values at each slice boundary, per segment, per particle */
  double globalZ1;
  double globalX1;
  double globalTheta1;
  double globalSin;
  double globalCos;
  double globalCosExtraTilt;
  double globalDxOffset;
  int spinOffset;
  double spinBrho;
  double spinCharge;
  double spinMassC;
  double spinAnomaly;
  GPU_APERTURE_CONTOUR_DATA contour;
} GPU_BEND_OPTIONS;

#define GPU_BEND_RECORD_SIZE 6

typedef struct GPU_CCBEND_DATA {
  long nSlices;
  int integrationOrder;
  int fringeModel;
  int edgeOrder;
  int angleSign;
  double Po;
  double radCoef;
  /* The CPU divides qx and qy by 1+dp and multiplies them back after every slice, with radiation or without */
  int momentumRoundTrip;
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
  /* The exit plane differs from the entrance plane only with YAW */
  double exitPlaneTan;
  double exitPlaneCos;
  double exitPlaneSin;
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
  /* Misalignments.  The two faces can differ, because the CPU leaves ETILT alone out of the entrance. */
  int malignMethod;
  int entryMisaligned;
  int exitMisaligned;
  int hasYaw;
  double dx;
  double dy;
  double dz;
  double cosEtilt;
  double sinEtilt;
  union {
    GPU_LINEAR_MISALIGNMENT misalignment;
    GPU_EXACT_MISALIGNMENT exactMisalignment;
  };
  /* K3 to K8 */
  int nHighKn;
  int highKnOrder[6];
  double highKnL[6];
  /* FRINGEMODEL=0: dipole, quadrupole and sextupole coefficients of the y' kick at each edge */
  double edge1Kick[3];
  double edge2Kick[3];
  /* Terms from multipole files, in the order edge 1, edge 2, body */
  int nEdge1Terms;
  int nEdge2Terms;
  int nBodyTerms;
  int maxTermOrder;
  double spinAngle1;
  double spinAngle2;
  GPU_BEND_OPTIONS options;
  GPU_MULTIPOLE_TERM term[GPU_MULTIPOLE_MAX_TERMS];
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
  double cosTilt;
  double sinTilt;
  /* The CPU always uses the exact body-centered method for an LGBEND */
  int misaligned;
  double apertureDx;
  double apertureDy;
  GPU_EXACT_MISALIGNMENT exactMisalignment;
  double zOffset[GPU_LGBEND_MAX_SEGMENTS];
  /* The segments this launch tracks, for a table of random numbers too large for one launch */
  long firstSegment;
  long lastSegment;
  GPU_BEND_OPTIONS options;
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
  /* An aperture_data file is evaluated once at each segment's center */
  int hasFileAperture;
  GPU_APERTURE_LIMIT_DATA fileAperture[GPU_LGBEND_MAX_SEGMENTS];
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
  long malignCount;
  long postApertureCount;
  long inexactCount; /* operations whose arithmetic isn't the CPU's, bends aside */
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
  GPU_TUNE_MALIGN_DATA *malign;
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
  int cpuOrderSums;  /* bin sums in particle order, as the CPU makes them */
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
  /* the CPU's numbers: bin sums in particle order, no device FFT */
  int cpuOrder;
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
  /* ZLONGIT's own grid (BIN_CENTERED=0): bin ib holds (t-tmin)/dt truncated, and its center, for the
   * longitudinal interpolation, is tminCenter+dt*ib with tminCenter = tmin+dt/2 formed on the host as
   * zlongit.c forms it.  Zero is the grid of every other user: bin centers at tmin+dt*ib. */
  int legacyGrid;
  double tminCenter;
  double pCentral;
  double offset[2];
  double particleMassMV;
  double particleRelSign;
  double cMks;
} GPU_COMBINED_WAKE_DATA;

/* Makes the voltages on the host from the three histograms (count, x and y sums, bins values each),
 * into voltage[channel * bins + bin] */
typedef void (*GPU_COMBINED_WAKE_HOST_VOLTAGE)(const double *histogram, double *voltage,
                                               void *context);

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
  int64_t count;
  double reference[3]; /* sums are relative to these values */
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
  int64_t startPID;
  int64_t endPID;
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
  /* Excitation, with three of the CPU's random numbers per particle unless lossOnly */
  double Srxp;
  double Sryp;
  double Srdelta;
  double cutoff;
  const double *draws;
} GPU_SREFFECTS_DATA;

/* ILMATRIX: the numbers trackWithIndividualizedLinearMatrix() takes from the element */
typedef struct GPU_ILMATRIX_DATA {
  double C[6];      /* the constant terms; C[4] is found for each particle */
  double length;    /* what the path-length factor multiplies */
  double tune[2], chrom[2], chrom2[2], chrom3[2];
  double tswax[2], tsway[2], tswax2[2], tsway2[2], tswaxay[2];
  double beta[2], beta1[2], alpha[2], alpha1[2];
  double eta[4], eta1[4];
  double alphac[3];
  double dsdA[2], dsdA2[2], dsdAxAy;
  /* rotateBeamCoordinatesForMisalignment() into the element's frame and, for the particles kept, back */
  double cosIn, sinIn, cosOut, sinOut;
  double z, pCentral;
  long tuneTwice[2];           /* (long)(2*tune[plane]), for the resonance-crossing test */
  int rotateIn, rotateOut;
  int allowResonanceCrossing;
} GPU_ILMATRIX_DATA;

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
  /* SYMPLECTIC=1: implicit midpoint rule on the vector potential (trackBGGExpansion) */
  int symplectic;
  int isBend;
  long zInterval;
  const double *potential; /* 3 per term: (-1)^ig m!, the denominators of A and of dAz */
  double step;
  double scaleA;
  double xEntry, zEntry, xExit, zExit;
  double sinEntry, cosEntry, sinExit, cosExit;
} GPU_BGGEXP_DATA;

typedef struct GPU_CWIGGLER_HARMONIC {
  double strength;
  double kx;
  double ky;
  double kz;
  double phase;
  double kwOverKz;
  double ratio;        /* kx/ky for By harmonics, ky/kx for Bx harmonics */
  double ratioSquared;
  long useSine;        /* use sin(k*q)/k rather than q*sinc(k*q) in the normal expansions */
} GPU_CWIGGLER_HARMONIC;

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
  /* For harmonic files.  The table holds the By harmonics, then the Bx harmonics. */
  const GPU_CWIGGLER_HARMONIC *harmonics;
  long horizontalHarmonics;
  long verticalHarmonics;
  int horizontalSplitPole;
  int verticalSplitPole;
  double horizontalAw;
  double verticalAw;
  double betaGamma;
  int hasTilt;
  double cosTilt;
  double sinTilt;
  /* Constant fields: the vector potential grows with z, and its end value comes off the slopes at the exit */
  double axConstant;
  double ayConstant;
  double constantBx;
  double constantBy;
  double exitAx;
  double exitAy;
  int hasOffset;
  double dx;
  double dy;
  double dz;
  /* Transverse gradient undulator: first By harmonic only, in the harmonic functions */
  double tguGradient;
  double tguCompFactor;
  double tguCharge;
  double tguFieldScale;
  int spinOffset;
  double isrCoef;
  const double *isrDraws; /* isrDrawsPerParticle per particle, drawn in the CPU's order */
  long isrDrawsPerParticle;
  double *fieldRecord;    /* GPU_CWIGGLER_FIELD_RECORD_SIZE values per row, one row per step plus the start, per particle */
} GPU_CWIGGLER_DATA;

#define GPU_CWIGGLER_FIELD_RECORD_SIZE 7

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
  /* field_table_tracking()'s frame changes (ftable_frame_converter() and the misalignments): entrance and
   * exit 0 none, 1 an exact drift back by L1 or L2, 2 the edge; the sines and cosines from the host */
  int entrance, exit, doOffset, doTilt;
  double l1, sinE1, cosE1, l2, sinE2, cosE2, dxExit;
  double dx, dy, dz, cosTilt, sinTilt, cosUntilt, sinUntilt;
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
  /* The aperture lorentz() checks at every derivative, as setupMultApertureData() left it at the
   * element's start, and what a lost particle's row gets */
  int apertures;
  GPU_APERTURE_LIMIT_DATA aperture;
  GPU_APERTURE_CONTOUR_DATA contour;
  double lossZStart;
  double pCentral;
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

/* LSRMDLTR with the non-adaptive Runge-Kutta integrator: motion.cc's derivatives_laserModulator(),
 * stochastic_laserModulator() without ISR, and rk_odeint3_na() */
typedef struct GPU_LSRMDLTR_DATA {
  double length, Bu, tguGradient, tguCompFactor, ku, k, omega, Escale, Bscale;
  double Ef0Laser, laserW0, ZRayleigh, laserPhase, laserX0, laserY0, laserZ0, laserTilt;
  double poleFactor1, poleFactor2, poleFactor3, t0, Zend, Zcenter, tolerance;
  double pCentral, gammaCentral, radCoef, cMks, eMks, meMks;
  long periods, nSteps, tProfilePoints, maxLoops;
  int laserM, laserN, helical, fieldCode, synchRad, laserOn;
  const double *timeValue, *amplitudeValue;
  /* ISR: gauss_rn_lim(0, 1, 3, random_2)'s numbers in the CPU's order; particle i of a pass takes
   * them from offsets[i] on, one a step */
  int isr;
  double isrCoef;
  const double *draws;
  long long drawCount;
} GPU_LSRMDLTR_DATA;

/* RFTMEZ0, TWLA and MAPSOLENOID as motion() (motion.cc) tracks them with the non-adaptive Runge-Kutta
 * integrator: the set-up is motionGpuSetup()'s, the fields derivatives_rftmEz0(), derivatives_tw_linac()
 * with its impulses, derivatives_mapSolenoid() */
#define GPU_MOTION_RFTMEZ0 1
#define GPU_MOTION_TWLA 2
#define GPU_MOTION_MAPSOLENOID 3
#define GPU_MOTION_TMCF 4  /* derivatives_tmcf_mode() */
#define GPU_MOTION_CEPL 5  /* derivatives_ce_plates() */
#define GPU_MOTION_TWPL 6  /* derivatives_tw_plates() */
#define GPU_MOTION_TWMTA 7 /* derivatives_twmta() with its impulses */
typedef struct GPU_MOTION_DATA {
  int type;
  /* motion()'s globals */
  double kscale, Zend, tauStart, Xlimit, Ylimit, tolerance, pCentral, zStart;
  double XapertureCenter, YapertureCenter, Xoffset, Yoffset;
  double Xcenter, Ycenter, Zcenter, Xcenter1, Ycenter1, Zcenter1;
  double partRot[9], fieldRot[9], partRot1[9], fieldRot1[9];
  long nSteps, maxLoops;
  int radialLimit;
  /* RFTMEZ0 */
  double dZ, EzPeak, Z0Sol, dZSol, dRSol, solenoidFactor, BxStrayScaled, ByStrayScaled;
  long nz, nzSol, nrSol;
  int radialOrder, hasSolenoid;
  const double *Ez, *dEzdZ, *BzSol, *BrSol, *dBzdZSol; /* the solenoid maps row ir at ir*nzSol */
  /* TWLA */
  double alphaS, EzS, betaWave, ErS, BphiS, BsolS, FrP;
  /* MAPSOLENOID */
  double dzMap, drMap, factor, zMap0, zMap1, BxUniformScaled, ByUniformScaled;
  long nzMap, nrMap;
  int hasUniform;
  const double *BzMap, *BrMap; /* row ir at ir*nzMap */
  /* TMCF: the scaled fields */
  double tmcfEr, tmcfEz, tmcfBphi;
  /* CEPL, TWPL */
  double Escaled, Estatic, cosTilt, sinTilt;
  /* TWMTA (with betaWave, EzS, BsolS and alphaS above) */
  double Kx, Ky, ExS, EyS, BxS, ByS;
} GPU_MOTION_DATA;

/* BEAMBEAM: applyBeamBeamKicks() (do_tracking.c) with gaussianBeamKick() and roundGaussianBeamKick(),
 * or ellipsoidalBeamKick() (ionEffects.cc).  The products of constants are the CPU's, formed on the
 * host in its order. */
typedef struct GPU_BEAMBEAM_DATA {
  int distribution; /* GAUSSIAN_BEAM_BEAM, UNIFORM_ELLIPSOIDAL_BEAM_BEAM, PARABOLIC_ELLIPSOIDAL_BEAM_BEAM */
  double centroid[2], size[2], charge, P0;
  /* gaussian: P0 * c_mks; particleMass and 1.0 / particleMass; c_mks * charge * re_mks * me_mks * ionCharge / e_mks and
   * the round beam's 2 * c_mks * ...; 2 * PI */
  double P0c, mass, inverseMass, C1, C1round, twoPi;
  /* ellipsoidal: -particleRelSign * particleCharge, particleMass * c_mks * P0, 4 * PI * epsilon_o,
   * 4 * charge, 8 * charge / 3, 8 * charge, c_mks */
  double pCharge, pMassCP0, fourPiEpsilon, fourCharge, eightChargeThirds, eightCharge, cMks;
} GPU_BEAMBEAM_DATA;

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
long gpuOmpTrackingRequested(long particles);
/* For loops that do a few arithmetic operations a particle (wake and cavity-mode kicks, time
 * coordinates, copies): host threads only from GPU_OMP_LIGHT_LOOP_MIN_PARTICLES particles */
long gpuOmpLightLoopRequested(long particles);
/* The one count of host threads: -ompThreads if given, else OMP_NUM_THREADS if set, else 1 */
void gpuSetHostThreads(long threads, const char *source);
/* True while a frequency-map batch is tracked whose RF references elegant would set again for each
 * particle on each turn; see gpu_batched_tune_own_rf_reference() */
long gpuTuneOwnRfReference(void);
long gpuOmpTrackingEnabled(long particles);
long gpuOmpTrackingScopeActive(void);
GPU_OMP_TRACKING_WORKSPACE *gpuGetOmpTrackingWorkspace(long particles);
long gpuCompactParticlesInCpuOrder(double **particle, long particles, unsigned char *survived);
void gpuBaseInit(double **coord, long nOriginal, double **accepted, double **lostPart,
                 long isMaster, long lossOutputNeeded,
                 long orderSensitiveOutputNeeded, long reductionOutputNeeded,
                 long alwaysChangeP0, long backtrack);
/* Notes whether the beam at the start of tracking is one bunch */
void gpuSetBunchAssignments(double **coord, long np, long idSlotsPerBunch);
/* For a beam between the resident floor and the general thresholds, lowers those thresholds for this
 * tracking call when the lattice's kick magnets would gain more on the device than the beam's trips to the
 * CPU cost.  Call after gpuBaseInit(). */
void gpuChooseSmallBeamThresholds(void *beamline, void *firstElement, long nParticles, long nPasses);
/* Notes whether a MAXAMP check follows the next element */
void gpuSetAfterElementAperture(long active);
void gpuBaseDealloc(void);
void gpuDisableForRun(const char *reason);
/* Whether do_tracking() must call the GPU layer at each element of this tracking call */
long gpuElementHooksNeeded(void);
void gpuSetTrackingSuppressed(long suppressed);
void setElementGpuData(void *eptr, long nParticles);
/* The pass do_tracking() is on, for elements that act on some passes only (gpuIdlePassElement()) */
void gpuSetTrackingPass(long iPass, long passOffset);
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
/* RFTMEZ0, TWLA, MAPSOLENOID as motion() tracks them; the particles left, or -1 when the CPU must track */
long gpu_track_motion(double **coord, double **accepted, long nParticles, void *field, long type,
                      double *pCentral, double zStart, long *changeP0);
/* BEAMBEAM's kicks (applyBeamBeamKicks()); 0 when the CPU must track */
long gpu_track_beambeam(long nParticles, void *element, double P0);
void gpuSetCpuParticleArray(double **coord, long nParticles);
double **forceParticlesToCpu(const char *reason);
double **copyParticlesToCpuReadOnly(const char *reason);
long gpu_matrix_supported(void *M);
long gpu_reductions_enabled(long nParticles);
long gpu_beam_sums_enabled(long nParticles, unsigned long flags);
long gpu_slice_moments_available(long nParticles);
long gpu_slice_moments(long nParticles, double Po, long nSlices, double *tMinAll, double *tMaxAll,
                       long *count, double *C, double *S);
long gpu_watch_parameters_supported(void *watch, long nParticles);
long gpu_watch_parameter_sums(long nParticles, int64_t *count,
                              double *pSum, double *gammaSum,
                              double *sigma56);
#ifdef GPU_VERIFY
void gpu_verify_watch_el(long nParticles, double pCentral, int64_t startPID, int64_t endPID,
                         double gpuEl, void *sums);
void gpu_verify_csr_drift_prep_coordinates(double **part, long nParticles);
#endif
int64_t gpu_histogram_ranges(long nParticles, double pCentral,
                             int64_t startPID, int64_t endPID,
                             unsigned int coordinateMask,
                             double *minimum, double *maximum);
void gpu_histogram_bins(long nParticles, double pCentral,
                        int64_t startPID, int64_t endPID, long bins,
                        unsigned int coordinateMask,
                        double timeOffset,
                        const double *lower, const double *upper,
                        double *histogram);
void startGpuTimer(void);
void startCpuTimer(void);
void displayTimings(void);
void compareGpuCpu(long nParticles, const char *label);
#ifdef GPU_VERIFY
void gpuVerifyResumeElementOnGpu(void);
void compareGpuCpuApertureResult(long gpuRemaining, long cpuRemaining, long nParticles,
                                 long pairByParticleID, const char *label);
void compareGpuCpuValue(double gpu, double cpu, double relTol, const char *label);
#endif
void copyReductionArrays(double *centroid, double *sigma);
void compareReductionArrays(double *centroid, double *sigma, void *sums, const char *label);
void *getGpuBeamSums(void *sums);
void sortByPID(long nParticles);

#ifdef __cplusplus
}
#endif

/* Element types moved to the device on November 15, 2026 */
#include "gpu_ports.h"

#endif
