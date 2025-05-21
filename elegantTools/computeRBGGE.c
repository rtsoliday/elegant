#include "mdb.h"
#include "SDDS.h"
#include "scan.h"
#include "fftpackC.h"
// #define DEBUG 1
// #define OLDFFT 1

#define TWOPI 6.28318530717958647692528676656

#if defined(linux) || (defined(_WIN32) && !defined(_MINGW))
#  include <omp.h>
#else
#  define NOTHREADS 1
#endif

typedef struct COMPLEX {
  double re;
  double im;
} COMPLEX;
COMPLEX **createComplexArray(long nxy, long nz, double *value);
void copyComplexArray(COMPLEX **target, COMPLEX **source, long nxy, long nz);
COMPLEX fourierCoeffIntegralTrap(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx);
COMPLEX fourierCoeffIntegralSimp(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx);
COMPLEX fourierCoeffIntegralLinInterp(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx);
COMPLEX fourierCoeffIntegralLinInterpSkew(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx);
COMPLEX calcGGtopbottomA(COMPLEX *beta, double k, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGtopbottomB(COMPLEX *beta, double k, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGrightA(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGrightB(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGleftA(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGleftB(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGtopbottomk0A(COMPLEX *beta, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGtopbottomk0B(COMPLEX *beta, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGrightk0(COMPLEX *beta, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGleftk0(COMPLEX *beta, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff);
COMPLEX calcGGallSidesBz(COMPLEX *beta, double k, double *lambda, double yMax, int32_t Ncoeff);
void FFT(COMPLEX *field, int32_t isign, int32_t npts);
unsigned long IntCeilingPowerOf2(unsigned long i);

typedef struct {
  short cached;
  COMPLEX **ByTop, **ByBottom, **BxRight, **BxLeft;
  COMPLEX **BzTop, **BzBottom, **BzRight, **BzLeft;
  COMPLEX **ByTopSaved, **ByBottomSaved, **BxRightSaved, **BxLeftSaved;
  COMPLEX **BzTopSaved, **BzBottomSaved, **BzRightSaved, **BzLeftSaved;
  long Nx, Ny, Nfft;
  double dx, dy, dz;
  double xMin, xMax;
  double yMin, yMax;
  double zMin, zMax;
  double xCenter, yCenter;
} FIELDS_ON_PLANES;

typedef struct {
  double rms, mad, max;
  double fracRms, fracMad, fracMax;
} ALL_RESIDUALS;

typedef struct {
  double *x, *y, *z;
  double *Bx, *By, *Bz;
  long n;
} FIELD_MAP;

typedef struct {
  short haveData;
  short skew;              /* if non-zero, these are skew terms */
  long nz;                 /* number of z points */
  double dz;               /* z spacing */
  double xCenter, yCenter; /* center of the expansion in magnet coordinate system */
  double xMax, yMax;       /* half-aperture of the field expansion in expansion coordinate system */
  double zMin, zMax;       /* minimum and maximum z values */
  long nm;                 /* number of values of m (angular harmonic) */
  long *m;                 /* value of m */
  long nGradients;         /* number of gradient functions per m */
  double ***Cmn;           /* generalized gradient: Cnms[im][in][iz] */
  double ***dCmn_dz;       /* z derivative of generalized gradient */
} BGGEXP_DATA;

void readBGGExpData(BGGEXP_DATA *bggexpData, char *filename, char *nameFragment, short skew);
void freeBGGExpData(BGGEXP_DATA *bggexpData);
int computeGGderiv(FIELDS_ON_PLANES *fieldsOnPlanes, char *outputFile, long derivatives, long multipoles, long fundamental,
                   long varyDerivatives);
int computeGGcos(FIELDS_ON_PLANES *fieldsOnPlanes, char *outputFile, long derivatives, long multipoles, long fundamental,
                 long varyDerivatives);
double evaluateGGEForFieldMap(FIELD_MAP *fmap, BGGEXP_DATA *bggexpData, FIELDS_ON_PLANES *fieldsOnPlanes,
                              long multipoles, long derivatives, double significance, double *coordLimit,
                              unsigned long flags, long varyDerivatives, ALL_RESIDUALS *allResiduals);
int ReadInputFiles(FIELDS_ON_PLANES *fieldsOnPlanes,
                   char *topFile, char *bottomFile, char *leftFile, char *rightFile, long needBz);
void readFieldMap(char *fieldMapFile, FIELD_MAP *fmData);
int evaluateGGEAndOutput(char *outputFile, char *normalFile, char *skewFile, FIELDS_ON_PLANES *fieldsOnPlanes);
void freeFieldsOnPlanes(FIELDS_ON_PLANES *fop);

#define SET_YMINUS 0
#define SET_YPLUS 1
#define SET_XMINUS 2
#define SET_XPLUS 3
#define SET_NORMAL 4
#define SET_SKEW 5
#define SET_DERIVATIVES 6
#define SET_MULTIPOLES 7
#define SET_FUNDAMENTAL 8
#define SET_AUTO_TUNE 9
#define SET_EVALUATE 10
#define SET_VERBOSE 11
#define SET_THREADS 12
#define SET_VARY_DERIVATIVES 13
#define N_OPTIONS 14

char *option[N_OPTIONS] = {
  "yminus", "yplus", "xminus", "xplus", "normal", "skew", "derivatives", "multipoles", "fundamental",
  "autotune", "evaluate", "verbose", "threads", "varyderivatives"};

#define USAGE "computeRBGGE -yminus=<filename> -yplus=<filename> -xminus=<filename> -xplus=<filename>\n\
             -normal=<output> [-skew=<output>] [-derivatives=<integer>] [-multipoles=<integer>] [-fundamental=<integer>]\n\
              [-varyDerivatives] [-evaluate=<filename>] [-verbose] [-threads=<integer>] \n\
              [-autotune=<3dMapFile>[,significance=<fieldValue>][,minimize={rms|mav|maximum}][,radiusLimit=<meters>][,xLimit=<meters>][,yLimit=<meters>][,increaseOnly][,verbose][,log=<filename>][,minDerivatives=<integer>][,minMultipoles=<integer>]]\n\
-yplus       (x, y, z, Bx, By, Bz) map for positive-y plane.\n\
-yminus      (x, y, z, Bx, By, Bz) map for negative-y plane.\n\
-xminus      (x, y, z, Bx, By, Bz) map for negative-x plane.\n\
-xplus       (x, y, z, Bx, By, Bz) map for positive-x plane.\n\
-normal      Output file for normal-component generalized gradients.\n\
-skew        Output file for skew-component generalized gradients. If the input data\n\
             has non-zero Bz on axis, this option is essential.\n\
-derivatives Number of derivatives vs z desired in output. Default: 7\n\
-multipoles  Number of multipoles desired in output. Default: 8\n\
-fundamental Fundamental multipole of sequence. 0=none (default), 1=dipole, 2=quadrupole, etc.\n\
-varyDerivatives\n\
             If given, the number of derivatives used varies with multipole order so as to\n\
             maintain an approximately consistent maximum transverse order.\n\
             Recommended to try if GGE does not fit data well.\n\
-evaluate    Evaluate the GGE over the interior region, including the four boundaries.\n\
-verbose     Print information while running.\n\
-threads     Specify number of threads to use for computations (default: 1).\n\
-autotune    Seeks to minimize the number of multipoles and derivatives to avoid using terms\n\
             that do not contribute to a good fit at the given level of significance. The user can\n\
             choose to minimize the maximum error (default), the rms error, or the mean absolute value\n\
             error. If 'increaseOnly' is given, the optimization method scans toward increasing values\n\
             of the number of multipoles and derivatives relative to the previous best; this is often much\n\
             faster.\n\n\
Rectangular Boundary Generalized Gradient Expansion by Ryan Lindberg, Robert Soliday, and Michael Borland.\n\
(" __DATE__ " " __TIME__ ", SVN revision: " SVN_VERSION ")\n"

#define AUTOTUNE_VERBOSE 0x0001UL
#define AUTOTUNE_RMS 0x0002UL
#define AUTOTUNE_MAXIMUM 0x0004UL
#define AUTOTUNE_MAV 0x0008UL
#define AUTOTUNE_EVALONLY 0x0010UL
#define AUTOTUNE_MODE_SET 0x0100UL
#define AUTOTUNE_LOG 0x0200UL
#define AUTOTUNE_INCRONLY 0x0400UL
char *modeOption[3] = {"rms", "maximum", "mav"};

int threads = 1;

int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  long i_arg;
  long multipoles, derivatives, fundamental = 0;
  long maxMultipoles = 8, maxDerivatives = 7;
  int32_t minMultipoles = -1, minDerivatives = -1;
  char *topFile = NULL, *bottomFile = NULL, *leftFile = NULL, *rightFile = NULL;
  char *normalOutputFile = NULL, *skewOutputFile = NULL;
  char *fieldMapFile = NULL;
  char *evaluationOutput = NULL;
  double autoTuneSignificance = 1e-12, autoTuneCoordLimit[3] = {0, 0, 0};
  FIELDS_ON_PLANES fieldsOnPlanes;
  FIELD_MAP fieldMap;
  double bestResidual;
  long bestMultipoles, bestDerivatives;
  unsigned long autoTuneFlags = 0;
  char *autoTuneModeString;
  BGGEXP_DATA bggexpData[2];
  SDDS_DATASET SDDS_autoTuneLog;
  char *autoTuneLogFile = NULL;
  long iAutoTuneLog = 0, verbose = 0, varyDerivatives = 0;
  ALL_RESIDUALS allResiduals;

#ifdef DEBUG
#  ifdef OLDFFT
  fprintf(stderr, "Using old FFT\n");
#  else
  fprintf(stderr, "Using new FFT\n");
#  endif
#endif

  argc = scanargs(&scanned, argc, argv);
  if (argc < 2 || argc > (2 + N_OPTIONS)) {
    fprintf(stderr, "%s\n", USAGE);
    return (1);
  }
  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (scanned[i_arg].arg_type == OPTION) {
      /* process options here */
      switch (match_string(scanned[i_arg].list[0], option, N_OPTIONS, 0)) {
      case SET_YPLUS:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -yplus syntax\n%s\n", USAGE);
          return (1);
        }
        topFile = scanned[i_arg].list[1];
        break;
      case SET_YMINUS:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -bottom syntax\n%s\n", USAGE);
          return (1);
        }
        bottomFile = scanned[i_arg].list[1];
        break;
      case SET_XMINUS:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -left syntax\n%s\n", USAGE);
          return (1);
        }
        leftFile = scanned[i_arg].list[1];
        break;
      case SET_XPLUS:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -right syntax\n%s\n", USAGE);
          return (1);
        }
        rightFile = scanned[i_arg].list[1];
        break;
      case SET_NORMAL:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -normal syntax\n%s\n", USAGE);
          return (1);
        }
        normalOutputFile = scanned[i_arg].list[1];
        break;
      case SET_SKEW:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -skew syntax\n%s\n", USAGE);
          return (1);
        }
        skewOutputFile = scanned[i_arg].list[1];
        break;
      case SET_EVALUATE:
        if (scanned[i_arg].n_items != 2) {
          fprintf(stderr, "invalid -evaluate syntax\n%s\n", USAGE);
          return (1);
        }
        evaluationOutput = scanned[i_arg].list[1];
        break;
      case SET_DERIVATIVES:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &maxDerivatives) != 1 ||
            maxDerivatives <= 0) {
          fprintf(stderr, "invalid -derivatives syntax\n%s\n", USAGE);
          return (1);
        }
        break;
      case SET_MULTIPOLES:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &maxMultipoles) != 1 ||
            maxMultipoles <= 0) {
          fprintf(stderr, "invalid -multipoles syntax\n%s\n", USAGE);
          return (1);
        }
        break;
      case SET_FUNDAMENTAL:
        if (scanned[i_arg].n_items != 2 ||
            sscanf(scanned[i_arg].list[1], "%ld", &fundamental) != 1 ||
            fundamental < 0) {
          fprintf(stderr, "invalid -fundamental syntax\n%s\n", USAGE);
          return (1);
        }
        break;
      case SET_AUTO_TUNE:
        if (scanned[i_arg].n_items < 2) {
          fprintf(stderr, "invalid -autotune syntax\n%s\n", USAGE);
          return (1);
        }
        fieldMapFile = scanned[i_arg].list[1];
        scanned[i_arg].n_items -= 2;
        autoTuneSignificance = 1e-12;
        autoTuneCoordLimit[0] = 0;
        autoTuneCoordLimit[1] = 0;
        autoTuneCoordLimit[2] = 0;
        autoTuneFlags = 0;
        minDerivatives = minMultipoles = -1;
        if (scanned[i_arg].n_items > 0 &&
            (!scanItemList(&autoTuneFlags, scanned[i_arg].list + 2, &scanned[i_arg].n_items, 0,
                           "verbose", -1, NULL, 0, AUTOTUNE_VERBOSE,
                           "increaseonly", -1, NULL, 0, AUTOTUNE_INCRONLY,
                           "evaluate", -1, NULL, 0, AUTOTUNE_EVALONLY,
                           "significance", SDDS_DOUBLE, &autoTuneSignificance, 1, 0,
                           "xlimit", SDDS_DOUBLE, &autoTuneCoordLimit[0], 1, 0,
                           "ylimit", SDDS_DOUBLE, &autoTuneCoordLimit[1], 1, 0,
                           "radiuslimit", SDDS_DOUBLE, &autoTuneCoordLimit[2], 1, 0,
                           "minimize", SDDS_STRING, &autoTuneModeString, 1, AUTOTUNE_MODE_SET,
                           "log", SDDS_STRING, &autoTuneLogFile, 1, AUTOTUNE_LOG,
                           "minmultipoles", SDDS_LONG, &minMultipoles, 1, 0,
                           "minderivatives", SDDS_LONG, &minDerivatives, 1, 0,
                           NULL) ||
             autoTuneSignificance <= 0)) {
          fprintf(stderr, "invalid -autotune syntax\n%s\n", USAGE);
          return (1);
        }
        if (autoTuneFlags & AUTOTUNE_MODE_SET) {
          switch (match_string(autoTuneModeString, modeOption, 3, 0)) {
          case 0:
            autoTuneFlags |= AUTOTUNE_RMS;
            break;
          case 1:
            autoTuneFlags |= AUTOTUNE_MAXIMUM;
            break;
          case 2:
            autoTuneFlags |= AUTOTUNE_MAV;
            break;
          default:
            SDDS_Bomb("invalid mode for autotune minimization. Use rms or maximum.");
            break;
          }
        } else
          autoTuneFlags |= AUTOTUNE_MAXIMUM;
        break;
      case SET_VERBOSE:
        verbose = 1;
        break;
      case SET_VARY_DERIVATIVES:
        varyDerivatives = 1;
        break;
      case SET_THREADS:
        if (scanned[i_arg].n_items != 2 || sscanf(scanned[i_arg].list[1], "%d", &threads) != 1 || threads <= 0)
          SDDS_Bomb("invalid -threads syntax: give an value greater than 0");
        break;
      default:
        fprintf(stderr, "unknown option given\n%s\n", USAGE);
        return (1);
        break;
      }
    } else {
      fprintf(stderr, "too many files listed\n%s\n", USAGE);
      return (1);
    }
  }

#if !defined(NOTHREADS)
  omp_set_num_threads(threads);
#endif

  init_stats();

  if ((topFile == NULL) || (bottomFile == NULL) || (leftFile == NULL) || (rightFile == NULL)) {
    fprintf(stderr, "%s\n", USAGE);
    return (1);
  }
#ifdef DEBUG
  fprintf(stderr, "topFile=%s\n", topFile);
  fprintf(stderr, "bottomFile=%s\n", bottomFile);
  fprintf(stderr, "leftFile=%s\n", leftFile);
  fprintf(stderr, "rightFile=%s\n", rightFile);
  fprintf(stderr, "derivatives=%ld\n", derivatives);
  fprintf(stderr, "multipoles=%ld\n", multipoles);
  fprintf(stderr, "normalOutputFile=%s\n", normalOutputFile);
  fprintf(stderr, "skewOutputFile=%s\n", skewOutputFile);
#endif

  if (autoTuneFlags & AUTOTUNE_LOG) {
    if (SDDS_InitializeOutput(&SDDS_autoTuneLog, SDDS_BINARY, 1, NULL, "computeRBGGE autotune output", autoTuneLogFile) != 1 ||
        SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalMultipoles", "m$bopt$n", NULL, "Optimal number of multipoles",
                             NULL, SDDS_LONG, NULL) == -1 ||
        SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalDerivatives", "d$bopt$n", NULL, "Optimal number of derivatives",
                             NULL, SDDS_LONG, NULL) == -1 ||
        SDDS_DefineParameter(&SDDS_autoTuneLog, "OptimalResidual", "r$bopt$n", "T", "Optimal residual",
                             NULL, SDDS_DOUBLE, NULL) == -1 ||
        !SDDS_DefineSimpleParameter(&SDDS_autoTuneLog, "OptimumLabel", NULL, SDDS_STRING) ||
        SDDS_DefineColumn(&SDDS_autoTuneLog, "m", NULL, NULL, "Number of multipoles", NULL, SDDS_LONG, 0) == -1 ||
        SDDS_DefineColumn(&SDDS_autoTuneLog, "d", NULL, NULL, "Number of derivatives", NULL, SDDS_LONG, 0) == -1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "RmsError", "T", SDDS_DOUBLE) != 1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "MaximumError", "T", SDDS_DOUBLE) != 1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "MadError", "T", SDDS_DOUBLE) != 1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalRmsError", NULL, SDDS_DOUBLE) != 1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalMaximumError", NULL, SDDS_DOUBLE) != 1 ||
        SDDS_DefineSimpleColumn(&SDDS_autoTuneLog, "FractionalMadError", NULL, SDDS_DOUBLE) != 1 ||
        !SDDS_WriteLayout(&SDDS_autoTuneLog) ||
        !SDDS_StartPage(&SDDS_autoTuneLog, maxMultipoles * maxDerivatives)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (verbose)
      report_stats(stdout, "Created auto-tune log file.");
  }
  bestResidual = DBL_MAX;
  bestDerivatives = maxDerivatives;
  bestMultipoles = maxMultipoles;

  bggexpData[0].haveData = bggexpData[1].haveData = 0;

  memset(&fieldsOnPlanes, 0, sizeof(fieldsOnPlanes));
  if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, skewOutputFile ? 1 : 0))
    return 1;
  if (verbose)
    report_stats(stdout, "Read input files");
  if (computeGGderiv(&fieldsOnPlanes, normalOutputFile, maxDerivatives, maxMultipoles, fundamental, varyDerivatives))
    return 1;
  if (verbose)
    report_stats(stdout, "Computed normal GGE");
  readBGGExpData(&bggexpData[0], normalOutputFile, "CnmS", 0);

  if (skewOutputFile) {
    /* Have to restore the data because it is modified by computeGGderiv and computeGGcos */
    if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, 1) ||
        computeGGcos(&fieldsOnPlanes, skewOutputFile, maxDerivatives, maxMultipoles, fundamental, varyDerivatives))
      return 1;
    readBGGExpData(&bggexpData[1], skewOutputFile, "CnmC", 1);
    if (verbose)
      report_stats(stdout, "Computed skew GGE");
  }

  /* Have to restore the data because it is modified by computeGGderiv and computeGGcos */
  if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, 1))
    return 1;

  if (fieldMapFile) {
    /* read 3D map */
    readFieldMap(fieldMapFile, &fieldMap);
    if (verbose)
      report_stats(stdout, "Read 3d field map from disk");
    if (minDerivatives < 1)
      minDerivatives = 1;
    if (minMultipoles < 1)
      minMultipoles = 1;
  } else {
    /* no auto-tuning */
    minDerivatives = maxDerivatives;
    fieldMap.n = 0;
  }
  if (autoTuneFlags & AUTOTUNE_EVALONLY)
    minDerivatives = maxDerivatives;

  for (derivatives = minDerivatives; derivatives <= maxDerivatives; derivatives++) {
    if (!(autoTuneFlags & AUTOTUNE_INCRONLY)) {
      if (!fieldMapFile) {
        /* no auto-tuning */
        minMultipoles = maxMultipoles;
      }
      if (autoTuneFlags & AUTOTUNE_EVALONLY)
        minMultipoles = maxMultipoles;
    }

    for (multipoles = minMultipoles; multipoles <= maxMultipoles; multipoles++) {
      if (fieldMapFile) {
        double residual;
        if ((residual = evaluateGGEForFieldMap(&fieldMap, &bggexpData[0], &fieldsOnPlanes,
                                               multipoles, derivatives,
                                               autoTuneSignificance, &autoTuneCoordLimit[0],
                                               autoTuneFlags, varyDerivatives, &allResiduals)) < bestResidual) {
          bestResidual = residual;
          bestMultipoles = multipoles;
          bestDerivatives = derivatives;
          if (autoTuneFlags & AUTOTUNE_INCRONLY) {
            minMultipoles = bestMultipoles;
            minDerivatives = bestDerivatives;
          }
          if (autoTuneFlags & AUTOTUNE_VERBOSE) {
            printf("New best residual of %le for m=%ld, d=%ld\n", residual, multipoles, derivatives);
            fflush(stdout);
          }
        } else {
          if (autoTuneFlags & AUTOTUNE_VERBOSE) {
            printf("Goodness of fit (%le) for m=%ld, d=%ld is not better than %le obtained for m=%ld, d=%ld\n",
                   residual, multipoles, derivatives, bestResidual, bestMultipoles, bestDerivatives);
            fflush(stdout);
          }
        }
        if (autoTuneFlags & AUTOTUNE_LOG) {
          if (!SDDS_SetRowValues(&SDDS_autoTuneLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                 iAutoTuneLog++,
                                 "m", multipoles, "d", derivatives, "RmsError", allResiduals.rms,
                                 "MaximumError", allResiduals.max, "MadError", allResiduals.mad,
                                 "FractionalRmsError", allResiduals.fracRms,
                                 "FractionalMadError", allResiduals.fracMad,
                                 "FractionalMaximumError", allResiduals.fracMax,
                                 NULL)) {
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
            return (1);
          }
        }
      }
    }
  }

  if (fieldMapFile) {
    if (verbose)
      report_stats(stdout, "Finished auto tuning");
    if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, skewOutputFile ? 1 : 0) ||
        computeGGderiv(&fieldsOnPlanes, normalOutputFile, bestDerivatives, bestMultipoles, fundamental, varyDerivatives))
      return 1;
    if (verbose)
      report_stats(stdout, "Saved best normal GGE result");
    if (skewOutputFile) {
      if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, 1) ||
          computeGGcos(&fieldsOnPlanes, skewOutputFile, bestDerivatives, bestMultipoles, fundamental, varyDerivatives))
        return 1;
      if (verbose)
        report_stats(stdout, "Saved best skew GGE result");
    }
  }

  if (evaluationOutput) {
    if (ReadInputFiles(&fieldsOnPlanes, topFile, bottomFile, leftFile, rightFile, skewOutputFile ? 1 : 0))
      return 1;
    evaluateGGEAndOutput(evaluationOutput, normalOutputFile, skewOutputFile, &fieldsOnPlanes);
    if (verbose)
      report_stats(stdout, "Wrote evaluation output to disk");
  }

  if (autoTuneFlags & AUTOTUNE_LOG) {
    char buffer[1024];
    snprintf(buffer, 1024, "m$bopt$n: %ld  d$bopt$n: %ld  r$bopt$n: %lg T",
             bestMultipoles, bestDerivatives, bestResidual);
    if (SDDS_SetParameters(&SDDS_autoTuneLog, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                           "OptimalMultipoles", bestMultipoles,
                           "OptimalDerivatives", bestDerivatives,
                           "OptimalResidual", bestResidual,
                           "OptimumLabel", buffer,
                           NULL) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    }
    if (SDDS_WritePage(&SDDS_autoTuneLog) != 1 || SDDS_Terminate(&SDDS_autoTuneLog) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (verbose)
      report_stats(stdout, "Wrote auto-tune log to disk");
  }

  return (0);
}

void freeFieldsOnPlanes(FIELDS_ON_PLANES *fop) {
  long ix, iy;

  if (fop->ByTop) {
    for (ix = 0; ix < fop->Nx; ix++) {
      free(fop->ByTop[ix]);
    }
    free(fop->ByTop);
    fop->ByTop = NULL;
  }
  if (fop->ByBottom) {
    for (ix = 0; ix < fop->Nx; ix++) {
      free(fop->ByBottom[ix]);
    }
    free(fop->ByBottom);
    fop->ByBottom = NULL;
  }

  if (fop->BzTop) {
    for (ix = 0; ix < fop->Nx; ix++) {
      free(fop->BzTop[ix]);
    }
    free(fop->BzTop);
    fop->BzTop = NULL;
  }
  if (fop->ByBottom) {
    for (ix = 0; ix < fop->Nx; ix++) {
      free(fop->ByBottom[ix]);
    }
    free(fop->ByBottom);
    fop->ByBottom = NULL;
  }

  if (fop->BxLeft) {
    for (iy = 0; iy < fop->Ny; iy++) {
      free(fop->BxLeft[iy]);
    }
    free(fop->BxLeft);
    fop->BxLeft = NULL;
  }
  if (fop->BxRight) {
    for (iy = 0; iy < fop->Ny; iy++) {
      free(fop->BxRight[iy]);
    }
    free(fop->BxRight);
    fop->BxRight = NULL;
  }

  if (fop->BzLeft) {
    for (iy = 0; iy < fop->Ny; iy++) {
      free(fop->BzLeft[iy]);
    }
    free(fop->BzLeft);
    fop->BzLeft = NULL;
  }
  if (fop->BzRight) {
    for (iy = 0; iy < fop->Ny; iy++) {
      free(fop->BzRight[iy]);
    }
    free(fop->BzRight);
    fop->BzRight = NULL;
  }

  fop->Nx = fop->Ny = fop->Nfft = 0;
  fop->dx = fop->dy = fop->dz = 0;
}

int ReadInputFiles(
                   FIELDS_ON_PLANES *fieldsOnPlanes,
                   char *topFile,
                   char *bottomFile,
                   char *leftFile,
                   char *rightFile,
                   long needBz /* if nonzero, reads Bz data */
                   ) {
  SDDS_DATASET SDDSInput;
  double *cvalues[2], *xvalues, *yvalues, *zvalues;
  int32_t rows, ix, iy;
  int32_t tmpNx, tmpNy, tmpNfft;
  double tmpdx, tmpdy, tmpdz;
  double xmintop, xminbottom, yminleft, yminright, xleft, xright, ytop, ybottom, xmaxtop, ymaxleft;

  if (!fieldsOnPlanes->cached) {
    /* Read in By and possibly Bz from the top face */
    if (SDDS_InitializeInput(&SDDSInput, topFile) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((SDDS_CheckColumn(&SDDSInput, "By", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (needBz &&
         SDDS_CheckColumn(&SDDSInput, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_ReadPage(&SDDSInput) != 1) {
      fprintf(stderr, "Unable to read SDDS page\n");
      return (1);
    }
    rows = SDDS_RowCount(&SDDSInput);
    cvalues[0] = SDDS_GetColumnInDoubles(&SDDSInput, "By");
    if (cvalues[0] == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    cvalues[1] = NULL;
    if (needBz) {
      cvalues[1] = SDDS_GetColumnInDoubles(&SDDSInput, "Bz");
      if (cvalues[1] == NULL) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
    }
    xvalues = SDDS_GetColumnInDoubles(&SDDSInput, "x");
    if (xvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    yvalues = SDDS_GetColumnInDoubles(&SDDSInput, "y");
    if (yvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    zvalues = SDDS_GetColumnInDoubles(&SDDSInput, "z");
    if (zvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_Terminate(&SDDSInput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((zvalues[0] != zvalues[1]) || (xvalues[0] == xvalues[1])) {
      fprintf(stderr, "Error: Sort the input file %s with sddssort -col=z -col=y -col=x\n", topFile);
      return (1);
    }
    xmintop = xvalues[0];
    xmaxtop = xvalues[rows - 1];
    ytop = yvalues[0];
    fieldsOnPlanes->xMin = xmintop;
    fieldsOnPlanes->xMax = xmaxtop;
    find_min_max(&(fieldsOnPlanes->zMin), &(fieldsOnPlanes->zMax), zvalues, rows);
    for (ix = 1; ix < rows; ix++) {
      if (zvalues[ix - 1] != zvalues[ix]) {
        fieldsOnPlanes->Nx = ix;
        fieldsOnPlanes->Nfft = rows / ix;
        fieldsOnPlanes->dx = (xvalues[ix - 1] - xvalues[0]) / (ix - 1);
        fieldsOnPlanes->dz = (zvalues[rows - 1] - zvalues[0]) / (fieldsOnPlanes->Nfft - 1);
        break;
      }
    }
    if (rows != fieldsOnPlanes->Nx * fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Unexpected row count (y plus file): Nx = %ld, Nz = %ld, rows = %ld\n",
              (long)fieldsOnPlanes->Nx, (long)fieldsOnPlanes->Nfft, (long)rows);
      return (1);
    }
#ifdef DEBUG
    fprintf(stderr, "Top file %s, Nx=%ld, Nfft=%ld, dx=%le, dz=%le\n",
            topFile, (long)fieldsOnPlanes->Nx, (long)fieldsOnPlanes->Nfft, fieldsOnPlanes->dx, fieldsOnPlanes->dz);
#endif

    fieldsOnPlanes->ByTop = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[0]);
    fieldsOnPlanes->ByTopSaved = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[0]);
    if (!needBz)
      fieldsOnPlanes->BzTop = fieldsOnPlanes->BzTopSaved = NULL;
    else {
      fieldsOnPlanes->BzTop = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[1]);
      fieldsOnPlanes->BzTopSaved = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[1]);
    }
    free(cvalues[0]);
    if (needBz)
      free(cvalues[1]);
    free(xvalues);
    free(yvalues);
    free(zvalues);
    tmpNx = fieldsOnPlanes->Nx;
    tmpNfft = fieldsOnPlanes->Nfft;
    tmpdx = fieldsOnPlanes->dx;
    tmpdz = fieldsOnPlanes->dz;

    /* Read in By from the bottom face */
    if (SDDS_InitializeInput(&SDDSInput, bottomFile) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((SDDS_CheckColumn(&SDDSInput, "By", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (needBz &&
         SDDS_CheckColumn(&SDDSInput, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_ReadPage(&SDDSInput) != 1) {
      fprintf(stderr, "Unable to read SDDS page\n");
      return (1);
    }
    rows = SDDS_RowCount(&SDDSInput);
    cvalues[0] = SDDS_GetColumnInDoubles(&SDDSInput, "By");
    if (cvalues[0] == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    cvalues[1] = NULL;
    if (needBz) {
      cvalues[1] = SDDS_GetColumnInDoubles(&SDDSInput, "Bz");
      if (cvalues[1] == NULL) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
    }
    xvalues = SDDS_GetColumnInDoubles(&SDDSInput, "x");
    if (xvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    yvalues = SDDS_GetColumnInDoubles(&SDDSInput, "y");
    if (yvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    zvalues = SDDS_GetColumnInDoubles(&SDDSInput, "z");
    if (zvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_Terminate(&SDDSInput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((zvalues[0] != zvalues[1]) || (xvalues[0] == xvalues[1])) {
      fprintf(stderr, "Error: Sort the input %s with sddssort -col=z -col=y -col=x\n", bottomFile);
      return (1);
    }
    xminbottom = xvalues[0];
    ybottom = yvalues[0];
    for (ix = 1; ix < rows; ix++) {
      if (zvalues[ix - 1] != zvalues[ix]) {
        fieldsOnPlanes->Nx = ix;
        fieldsOnPlanes->Nfft = rows / ix;
        fieldsOnPlanes->dx = (xvalues[ix - 1] - xvalues[0]) / (ix - 1);
        fieldsOnPlanes->dz = (zvalues[rows - 1] - zvalues[0]) / (fieldsOnPlanes->Nfft - 1);
        break;
      }
    }
    if (rows != fieldsOnPlanes->Nx * fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Unexpected row count (y minus file): Nx = %ld, Nz = %ld, rows = %ld\n",
              (long)fieldsOnPlanes->Nx, (long)fieldsOnPlanes->Nfft, (long)rows);
      return (1);
    }
#ifdef DEBUG
    fprintf(stderr, "Bottom file %s, Nx=%ld, Nfft=%ld, dx=%le, dz=%le\n",
            bottomFile, (long)fieldsOnPlanes->Nx, (long)fieldsOnPlanes->Nfft, fieldsOnPlanes->dx, fieldsOnPlanes->dz);
#endif

    if (tmpNx != fieldsOnPlanes->Nx) {
      fprintf(stderr, "Nx values differ in the input files\n");
      return (1);
    }
    if (tmpNfft != fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Nfft values differ in the input files\n");
      return (1);
    }
    if ((tmpdx + 1e-9 < fieldsOnPlanes->dx) || (tmpdx - 1e-9 > fieldsOnPlanes->dx)) {
      fprintf(stderr, "dx values differ in the input files (%21.15le vs %21.15le)\n", tmpdx, fieldsOnPlanes->dx);
      return (1);
    }
    if ((tmpdz + 1e-9 < fieldsOnPlanes->dz) || (tmpdz - 1e-9 > fieldsOnPlanes->dz)) {
      fprintf(stderr, "dz values differ in the input files (%21.15le vs %21.15le)\n", tmpdz, fieldsOnPlanes->dz);
      return (1);
    }

    fieldsOnPlanes->ByBottom = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[0]);
    fieldsOnPlanes->ByBottomSaved = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[0]);
    if (!needBz)
      fieldsOnPlanes->BzBottom = fieldsOnPlanes->BzBottomSaved = NULL;
    else {
      fieldsOnPlanes->BzBottom = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[1]);
      fieldsOnPlanes->BzBottomSaved = createComplexArray(fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft, cvalues[1]);
    }
    free(cvalues[0]);
    if (needBz)
      free(cvalues[1]);
    free(xvalues);
    free(yvalues);
    free(zvalues);

    /* Read in Bx from the left face */
    if (SDDS_InitializeInput(&SDDSInput, leftFile) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((SDDS_CheckColumn(&SDDSInput, "Bx", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (needBz &&
         SDDS_CheckColumn(&SDDSInput, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_ReadPage(&SDDSInput) != 1) {
      fprintf(stderr, "Unable to read SDDS page\n");
      return (1);
    }
    rows = SDDS_RowCount(&SDDSInput);
    cvalues[0] = SDDS_GetColumnInDoubles(&SDDSInput, "Bx");
    if (cvalues[0] == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    cvalues[1] = NULL;
    if (needBz) {
      cvalues[1] = SDDS_GetColumnInDoubles(&SDDSInput, "Bz");
      if (cvalues[1] == NULL) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
    }
    xvalues = SDDS_GetColumnInDoubles(&SDDSInput, "x");
    if (xvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    yvalues = SDDS_GetColumnInDoubles(&SDDSInput, "y");
    if (yvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    zvalues = SDDS_GetColumnInDoubles(&SDDSInput, "z");
    if (zvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_Terminate(&SDDSInput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((zvalues[0] != zvalues[1]) || (yvalues[0] == yvalues[1])) {
      fprintf(stderr, "Error: Sort the input file %s with sddssort -col=z -col=y -col=x\n", leftFile);
      return (1);
    }
    yminleft = yvalues[0];
    ymaxleft = yvalues[rows - 1];
    fieldsOnPlanes->yMin = yminleft;
    fieldsOnPlanes->yMax = ymaxleft;
    xleft = xvalues[0];
    for (iy = 1; iy < rows; iy++) {
      if (zvalues[iy - 1] != zvalues[iy]) {
        fieldsOnPlanes->Ny = iy;
        fieldsOnPlanes->Nfft = rows / iy;
        fieldsOnPlanes->dy = (yvalues[iy - 1] - yvalues[0]) / (iy - 1);
        fieldsOnPlanes->dz = (zvalues[rows - 1] - zvalues[0]) / (fieldsOnPlanes->Nfft - 1);
        break;
      }
    }
    if (rows != fieldsOnPlanes->Ny * fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Unexpected row count (x minus file): Ny = %ld, Nz = %ld, rows = %ld\n",
              (long)fieldsOnPlanes->Ny, (long)fieldsOnPlanes->Nfft, (long)rows);
      return (1);
    }

    if (tmpNfft != fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Nfft values differ in the input files\n");
      return (1);
    }
    if ((tmpdz + 1e-9 < fieldsOnPlanes->dz) || (tmpdz - 1e-9 > fieldsOnPlanes->dz)) {
      fprintf(stderr, "dz values differ in the input files\n");
      return (1);
    }
#ifdef DEBUG
    fprintf(stderr, "Left file %s, Ny=%ld, Nfft=%ld, dx=%le, dz=%le\n",
            leftFile, (long)fieldsOnPlanes->Ny, (long)fieldsOnPlanes->Nfft, fieldsOnPlanes->dx, fieldsOnPlanes->dz);
#endif

    fieldsOnPlanes->BxLeft = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[0]);
    fieldsOnPlanes->BxLeftSaved = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[0]);
    if (!needBz)
      fieldsOnPlanes->BzLeft = fieldsOnPlanes->BzLeftSaved = NULL;
    else {
      fieldsOnPlanes->BzLeft = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[1]);
      fieldsOnPlanes->BzLeftSaved = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[1]);
    }
    free(cvalues[0]);
    if (needBz)
      free(cvalues[1]);
    free(xvalues);
    free(yvalues);
    free(zvalues);
    tmpNy = fieldsOnPlanes->Ny;
    tmpdy = fieldsOnPlanes->dy;

    /* Read in Bx from the right face */
    if (SDDS_InitializeInput(&SDDSInput, rightFile) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((SDDS_CheckColumn(&SDDSInput, "Bx", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (needBz &&
         SDDS_CheckColumn(&SDDSInput, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) ||
        (SDDS_CheckColumn(&SDDSInput, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_ReadPage(&SDDSInput) != 1) {
      fprintf(stderr, "Unable to read SDDS page\n");
      return (1);
    }
    rows = SDDS_RowCount(&SDDSInput);
    cvalues[0] = SDDS_GetColumnInDoubles(&SDDSInput, "Bx");
    if (cvalues[0] == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    cvalues[1] = NULL;
    if (needBz) {
      cvalues[1] = SDDS_GetColumnInDoubles(&SDDSInput, "Bz");
      if (cvalues[1] == NULL) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
    }
    xvalues = SDDS_GetColumnInDoubles(&SDDSInput, "x");
    if (xvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    yvalues = SDDS_GetColumnInDoubles(&SDDSInput, "y");
    if (yvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    zvalues = SDDS_GetColumnInDoubles(&SDDSInput, "z");
    if (zvalues == NULL) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_Terminate(&SDDSInput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if ((zvalues[0] != zvalues[1]) || (yvalues[0] == yvalues[1])) {
      fprintf(stderr, "Error: Sort the input file %s with sddssort -col=z -col=y -col=x\n", rightFile);
      return (1);
    }
    yminright = yvalues[0];
    xright = xvalues[0];
    for (iy = 1; iy < rows; iy++) {
      if (zvalues[iy - 1] != zvalues[iy]) {
        fieldsOnPlanes->Ny = iy;
        fieldsOnPlanes->Nfft = rows / iy;
        fieldsOnPlanes->dy = (yvalues[iy - 1] - yvalues[0]) / (iy - 1);
        fieldsOnPlanes->dz = (zvalues[rows - 1] - zvalues[0]) / (fieldsOnPlanes->Nfft - 1);
        break;
      }
    }
    if (rows != fieldsOnPlanes->Ny * fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Unexpected row count (x plus file): Ny = %ld, Nz = %ld, rows = %ld\n",
              (long)fieldsOnPlanes->Ny, (long)fieldsOnPlanes->Nfft, (long)rows);
      return (1);
    }
#ifdef DEBUG
    fprintf(stderr, "Right file %s, Ny=%ld, Nfft=%ld, dx=%le, dz=%le\n",
            rightFile, (long)fieldsOnPlanes->Ny, (long)fieldsOnPlanes->Nfft, fieldsOnPlanes->dx, fieldsOnPlanes->dz);
#endif

    if (tmpNy != fieldsOnPlanes->Ny) {
      fprintf(stderr, "Ny values differ in the input files\n");
      return (1);
    }
    if (tmpNfft != fieldsOnPlanes->Nfft) {
      fprintf(stderr, "Nfft values differ in the input files\n");
      return (1);
    }
    if ((tmpdy + 1e-9 < fieldsOnPlanes->dy) || (tmpdy - 1e-9 > fieldsOnPlanes->dy)) {
      fprintf(stderr, "dy values differ in the x input files (%21.15le vs %21.15le)\n", tmpdy, fieldsOnPlanes->dy);
      return (1);
    }
    if ((tmpdz + 1e-9 < fieldsOnPlanes->dz) || (tmpdz - 1e-9 > fieldsOnPlanes->dz)) {
      fprintf(stderr, "dz values differ in the x input files (%21.15le vs %21.15le)\n", tmpdz, fieldsOnPlanes->dz);
      return (1);
    }

    fieldsOnPlanes->BxRight = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[0]);
    fieldsOnPlanes->BxRightSaved = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[0]);
    if (!needBz)
      fieldsOnPlanes->BzRight = fieldsOnPlanes->BzRightSaved = NULL;
    else {
      fieldsOnPlanes->BzRight = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[1]);
      fieldsOnPlanes->BzRightSaved = createComplexArray(fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft, cvalues[1]);
    }
    free(cvalues[0]);
    if (needBz)
      free(cvalues[1]);
    free(xvalues);
    free(yvalues);
    free(zvalues);

    if (xmintop != xminbottom) {
      fprintf(stderr, "Error: x range differs in yplus and yminus files\n");
      return (1);
    }
    if (yminleft != yminright) {
      fprintf(stderr, "Error: y range differs in xminus and xplus files\n");
      return (1);
    }
    if (xleft >= xright) {
      fprintf(stderr, "Error: x values in xminus file less than x values in xplus file\n");
      return (1);
    }
    if (ytop <= ybottom) {
      fprintf(stderr, "Error: y values in yplus file less than x values in yminus file\n");
      return (1);
    }
    if (xmintop != xleft) {
      fprintf(stderr, "Error: x values in xminus file don't match min x values in yplus and yminus files\n");
      return (1);
    }
    if (yminleft != ybottom) {
      fprintf(stderr, "Error: y values in yminus file don't match min y values in xminus and xplus files\n");
      return (1);
    }
    if (xmaxtop != xright) {
      fprintf(stderr, "Error: x values in xplus file don't match max x values in yplus and yminus files\n");
      return (1);
    }
    if (ymaxleft != ytop) {
      fprintf(stderr, "Error: y values in yplus file don't match max y values in xminus and xplus files\n");
      return (1);
    }

    fieldsOnPlanes->xCenter = (xleft + xright) * .5;
    fieldsOnPlanes->yCenter = (ytop + ybottom) * .5;

    fieldsOnPlanes->cached = 1;
  } else {
    copyComplexArray(fieldsOnPlanes->ByTop, fieldsOnPlanes->ByTopSaved, fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft);
    if (fieldsOnPlanes->BzTop)
      copyComplexArray(fieldsOnPlanes->BzTop, fieldsOnPlanes->BzTopSaved, fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft);
    copyComplexArray(fieldsOnPlanes->ByBottom, fieldsOnPlanes->ByBottomSaved, fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft);
    if (fieldsOnPlanes->BzBottom)
      copyComplexArray(fieldsOnPlanes->BzBottom, fieldsOnPlanes->BzBottomSaved, fieldsOnPlanes->Nx, fieldsOnPlanes->Nfft);
    copyComplexArray(fieldsOnPlanes->BxLeft, fieldsOnPlanes->BxLeftSaved, fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft);
    if (fieldsOnPlanes->BzLeft)
      copyComplexArray(fieldsOnPlanes->BzLeft, fieldsOnPlanes->BzLeftSaved, fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft);
    copyComplexArray(fieldsOnPlanes->BxRight, fieldsOnPlanes->BxRightSaved, fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft);
    if (fieldsOnPlanes->BzRight)
      copyComplexArray(fieldsOnPlanes->BzRight, fieldsOnPlanes->BzRightSaved, fieldsOnPlanes->Ny, fieldsOnPlanes->Nfft);
  }

  return (0);
}

int computeGGderiv(FIELDS_ON_PLANES *fieldsOnPlanes, char *outputFile, long derivatives, long multipoles,
                   long fundamental, long varyDerivatives) {
  COMPLEX **ByTop, **ByBottom, **BxRight, **BxLeft;

  COMPLEX **betaTop, **betaBottom, **betaRight, **betaLeft;
  COMPLEX **genGradr_k;
  COMPLEX **derivGG;
  COMPLEX genGradT, genGradB, genGradR, genGradL;

  double *lambda, *tau, *k, *x, *y;

  double xMax, yMax, xCenter, yCenter, zMin;
  double dx, dy, dz, dk, invNfft;

  int32_t n, ir, ix, Nx, iy, Ny, ik, Nfft, Nz;

  int32_t Ngrad = 8;
  int32_t Nderiv = 7;
  int32_t Ncoeff = 40;

  SDDS_DATASET SDDSOutput;
  char name[1024], units[1024];

  Ngrad = multipoles;
  Nderiv = 2 * derivatives - 1;

  Nfft = fieldsOnPlanes->Nfft;
  Nx = fieldsOnPlanes->Nx;
  Ny = fieldsOnPlanes->Ny;
  dx = fieldsOnPlanes->dx;
  dy = fieldsOnPlanes->dy;
  dz = fieldsOnPlanes->dz;
  ByTop = fieldsOnPlanes->ByTop;
  ByBottom = fieldsOnPlanes->ByBottom;
  BxRight = fieldsOnPlanes->BxRight;
  BxLeft = fieldsOnPlanes->BxLeft;
  xCenter = fieldsOnPlanes->xCenter;
  yCenter = fieldsOnPlanes->yCenter;
  zMin = fieldsOnPlanes->zMin;

  Nz = Nfft;

  dk = TWOPI / (dz * (double)Nfft);
#pragma omp parallel
  {
    int ix, iy, myid;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ix = 0; ix < Nx; ix++)
      if (ix % threads == myid) {
        FFT(ByTop[ix], -1, Nfft);
        FFT(ByBottom[ix], -1, Nfft);
      }
    for (iy = 0; iy < Ny; iy++)
      if (iy % threads == myid) {
        FFT(BxRight[iy], -1, Nfft);
        FFT(BxLeft[iy], -1, Nfft);
      }
#pragma omp barrier
  }

  xMax = dx * 0.5 * (double)(Nx - 1);
  yMax = dy * 0.5 * (double)(Ny - 1);
#ifdef DEBUG
  fprintf(stderr, "xMax = %e, yMax=%e\n", xMax, yMax);
#endif
  x = calloc(Nx, sizeof(double));
  for (ix = 0; ix < Nx; ix++)
    x[ix] = -xMax + dx * (double)ix;
  y = calloc(Ny, sizeof(double));
  for (iy = 0; iy < Ny; iy++)
    y[iy] = -yMax + dy * (double)iy;
  lambda = calloc(Ncoeff, sizeof(double));
  tau = calloc(Ncoeff, sizeof(double));
  for (n = 0; n < Ncoeff; n++) {
    lambda[n] = PI * (double)n / (2.0 * xMax);
    tau[n] = PI * (double)n / (2.0 * yMax);
  }
  betaTop = calloc(Nfft, sizeof(COMPLEX *));
  betaBottom = calloc(Nfft, sizeof(COMPLEX *));
  betaRight = calloc(Nfft, sizeof(COMPLEX *));
  betaLeft = calloc(Nfft, sizeof(COMPLEX *));
  for (ik = 0; ik < Nfft; ik++) {
    betaTop[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaBottom[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaRight[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaLeft[ik] = calloc(Ncoeff, sizeof(COMPLEX));
  }

#pragma omp parallel
  {
    int ik, ix, n, myid;
    COMPLEX *Bint;
    Bint = calloc(Nx, sizeof(COMPLEX));
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = ByTop[ix][ik].re;
          Bint[ix].im = ByTop[ix][ik].im;
        }
        betaTop[ik][0] = fourierCoeffIntegralTrap(Bint, x, lambda[0], dx, xMax, Nx);
        betaTop[ik][0].re = 0.5 * betaTop[ik][0].re;
        betaTop[ik][0].im = 0.5 * betaTop[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaTop[ik][n] = fourierCoeffIntegralLinInterp(Bint, x, lambda[n], dx, xMax, Nx);

        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = -ByBottom[ix][ik].re;
          Bint[ix].im = -ByBottom[ix][ik].im;
        }
        betaBottom[ik][0] = fourierCoeffIntegralTrap(Bint, x, lambda[0], dx, xMax, Nx);
        betaBottom[ik][0].re = 0.5 * betaBottom[ik][0].re;
        betaBottom[ik][0].im = 0.5 * betaBottom[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaBottom[ik][n] = fourierCoeffIntegralLinInterp(Bint, x, lambda[n], dx, xMax, Nx);
      }
#pragma omp barrier
    free(Bint);
  }

#pragma omp parallel
  {
    int ik, iy, n, myid;
    COMPLEX *Bint;
    Bint = calloc(Ny, sizeof(COMPLEX));
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = BxRight[iy][ik].re;
          Bint[iy].im = BxRight[iy][ik].im;
        }
        betaRight[ik][0] = fourierCoeffIntegralSimp(Bint, y, tau[0], dy, yMax, Ny);
        betaRight[ik][0].re = 0.5 * betaRight[ik][0].re;
        betaRight[ik][0].im = 0.5 * betaRight[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaRight[ik][n] = fourierCoeffIntegralLinInterp(Bint, y, tau[n], dy, yMax, Ny);

        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = -BxLeft[iy][ik].re;
          Bint[iy].im = -BxLeft[iy][ik].im;
        }
        betaLeft[ik][0] = fourierCoeffIntegralSimp(Bint, y, tau[0], dy, yMax, Ny);
        betaLeft[ik][0].re = 0.5 * betaLeft[ik][0].re;
        betaLeft[ik][0].im = 0.5 * betaLeft[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaLeft[ik][n] = fourierCoeffIntegralLinInterp(Bint, y, tau[n], dy, yMax, Ny);
      }
#pragma omp barrier
    free(Bint);
  }

  k = calloc(Nfft, sizeof(double));
  for (ik = 0; ik < Nfft / 2; ik++)
    k[ik] = dk * (double)ik;
  for (ik = Nfft / 2; ik < Nfft; ik++)
    k[ik] = -dk * (double)(Nfft - ik);
  genGradr_k = calloc(Ngrad, sizeof(COMPLEX *));
  for (ir = 0; ir < Ngrad; ir++)
    genGradr_k[ir] = calloc(Nfft, sizeof(COMPLEX));
  for (ir = 0; ir < Ngrad; ir++) {
    long ir1;
    if (fundamental)
      ir1 = fundamental * (2 * ir + 1);
    else
      ir1 = ir + 1;
    ik = 0;
    /* top and bottom need care for k->0 */
    genGradT = calcGGtopbottomk0A(betaTop[ik], lambda, yMax, ir1, Ncoeff);
    genGradB = calcGGtopbottomk0A(betaBottom[ik], lambda, yMax, ir1, Ncoeff);

    genGradR = calcGGrightA(betaRight[ik], k[ik], tau, xMax, ir1, Ncoeff);
    genGradL = calcGGleftA(betaLeft[ik], k[ik], tau, xMax, ir1, Ncoeff);

    genGradr_k[ir][ik].re = genGradT.re - genGradB.re + genGradR.re + genGradL.re;
    genGradr_k[ir][ik].im = genGradT.im - genGradB.im + genGradR.im + genGradL.im;
#pragma omp parallel
    {
      int ik, myid;
      COMPLEX genGradT, genGradB, genGradR, genGradL;
#if !defined(NOTHREADS)
      myid = omp_get_thread_num();
#else
      myid = 0;
#endif
      for (ik = 1; ik < Nfft; ik++)
        if (ik % threads == myid) {
          genGradT = calcGGtopbottomA(betaTop[ik], k[ik], lambda, yMax, ir1, Ncoeff);
          genGradB = calcGGtopbottomA(betaBottom[ik], k[ik], lambda, yMax, ir1, Ncoeff);
          genGradR = calcGGrightA(betaRight[ik], k[ik], tau, xMax, ir1, Ncoeff);
          genGradL = calcGGleftA(betaLeft[ik], k[ik], tau, xMax, ir1, Ncoeff);

          genGradr_k[ir][ik].re = genGradT.re - genGradB.re + genGradR.re + genGradL.re;
          genGradr_k[ir][ik].im = genGradT.im - genGradB.im + genGradR.im + genGradL.im;
        }
#pragma omp barrier
    }
  }

  invNfft = 1.0 / (double)Nfft;
  derivGG = calloc(Nderiv, sizeof(COMPLEX *));
  for (n = 0; n < Nderiv; n++)
    derivGG[n] = calloc(Nfft, sizeof(COMPLEX));

#ifdef DEBUG
  fprintf(stderr, "Printing results...\n");
#endif

  if (SDDS_InitializeOutput(&SDDSOutput, SDDS_BINARY, 1, NULL, "computeRBGGE normal output", outputFile) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  if ((SDDS_DefineSimpleParameter(&SDDSOutput, "m", NULL, SDDS_LONG) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleColumn(&SDDSOutput, "z", "m", SDDS_DOUBLE) != 1)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "CnmS%" PRId32, n);
    if ((2 * n - 1) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 1));
    else if ((2 * n - 1) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 1));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }
  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "dCnmS%" PRId32 "/dz", n);
    if ((2 * n - 2) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 2));
    else if ((2 * n - 2) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 2));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }
  if (SDDS_WriteLayout(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  for (ir = 0; ir < Ngrad; ir++) {
    long NderivLimited;
    NderivLimited = Nderiv;
    if (varyDerivatives) {
      long m;
      m = (int32_t)(fundamental ? fundamental * (2 * ir + 1) : ir + 1);
      NderivLimited = (derivatives - m / 2) * 2 - 1;
    }
    /* Take derivatives */
    for (ik = 0; ik < Nfft; ik++) {
      derivGG[0][ik].re = -k[ik] * genGradr_k[ir][ik].im;
      derivGG[0][ik].im = k[ik] * genGradr_k[ir][ik].re;
    }
    for (n = 1; n < NderivLimited; n++)
      for (ik = 0; ik < Nfft; ik++) {
        derivGG[n][ik].re = -k[ik] * derivGG[n - 1][ik].im;
        derivGG[n][ik].im = k[ik] * derivGG[n - 1][ik].re;
      }
    FFT(genGradr_k[ir], 1, Nfft);
    for (n = 0; n < NderivLimited; n++)
      FFT(derivGG[n], 1, Nfft);
    if (SDDS_StartPage(&SDDSOutput, Nz) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_SetParameters(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                           "m", (int32_t)(fundamental ? fundamental * (2 * ir + 1) : ir + 1),
                           "xCenter", xCenter,
                           "yCenter", yCenter,
                           "xMax", xMax,
                           "yMax", yMax,
                           NULL) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    for (ik = 0; ik < Nz; ik++) {
      if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                            "z", zMin + dz * (double)ik,
                            "CnmS0", genGradr_k[ir][ik].re * invNfft,
                            NULL) != 1) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
      for (n = 2; n < Nderiv; n += 2) {
        sprintf(name, "CnmS%" PRId32, n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              name, n >= NderivLimited ? 0.0 : derivGG[n - 1][ik].re * invNfft,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
      for (n = 0; n < Nderiv; n += 2) {
        sprintf(name, "dCnmS%" PRId32 "/dz", n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              name, n >= NderivLimited ? 0.0 : derivGG[n][ik].re * invNfft,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
    }
    if (SDDS_WritePage(&SDDSOutput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }
  if (SDDS_Terminate(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }

  return (0);
}

int computeGGcos(
                 FIELDS_ON_PLANES *fieldsOnPlanes,
                 char *outputFile,
                 long derivatives,
                 long multipoles,
                 long fundamental,
                 long varyDerivatives) {
  COMPLEX **ByTop, **ByBottom, **BxRight, **BxLeft;
  COMPLEX **BzTop, **BzBottom, **BzRight, **BzLeft;

  COMPLEX **betaTop, **betaBottom, **betaRight, **betaLeft;
  COMPLEX **genGradr_k;
  COMPLEX **derivGG;
  COMPLEX genGradT, genGradB, genGradR, genGradL;

  double *lambda, *tau, *k, *x, *y;

  double xMax, yMax, xCenter, yCenter, zMin;
  double dx, dy, dz, dk, invNfft;

  int32_t n, ir, ix, Nx, iy, Ny, ik, Nfft, Nz;

  int32_t Ngrad = 8;
  int32_t Nderiv = 7;
  int32_t Ncoeff = 40;

  SDDS_DATASET SDDSOutput;

  char name[1024], units[1024];

  Ngrad = multipoles;
  Nderiv = 2 * derivatives - 1;

  Nfft = fieldsOnPlanes->Nfft;
  Nx = fieldsOnPlanes->Nx;
  Ny = fieldsOnPlanes->Ny;
  dx = fieldsOnPlanes->dx;
  dy = fieldsOnPlanes->dy;
  dz = fieldsOnPlanes->dz;
  ByTop = fieldsOnPlanes->ByTop;
  ByBottom = fieldsOnPlanes->ByBottom;
  BxRight = fieldsOnPlanes->BxRight;
  BxLeft = fieldsOnPlanes->BxLeft;
  BzTop = fieldsOnPlanes->BzTop;
  BzBottom = fieldsOnPlanes->BzBottom;
  BzRight = fieldsOnPlanes->BzRight;
  BzLeft = fieldsOnPlanes->BzLeft;
  xCenter = fieldsOnPlanes->xCenter;
  yCenter = fieldsOnPlanes->yCenter;
  zMin = fieldsOnPlanes->zMin;

  Nz = Nfft;

  dk = TWOPI / (dz * (double)Nfft);
#pragma omp parallel
  {
    int ix, iy, myid;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ix = 0; ix < Nx; ix++)
      if (ix % threads == myid) {
        FFT(ByTop[ix], -1, Nfft);
        FFT(ByBottom[ix], -1, Nfft);
      }
    for (iy = 0; iy < Ny; iy++)
      if (iy % threads == myid) {
        FFT(BxRight[iy], -1, Nfft);
        FFT(BxLeft[iy], -1, Nfft);
      }
#pragma omp barrier
  }

  xMax = dx * 0.5 * (double)(Nx - 1);
  yMax = dy * 0.5 * (double)(Ny - 1);
#ifdef DEBUG
  printf("xMax = %e, yMax=%e\n", xMax, yMax);
#endif
  x = calloc(Nx, sizeof(double));
  for (ix = 0; ix < Nx; ix++)
    x[ix] = -xMax + dx * (double)ix;
  y = calloc(Ny, sizeof(double));
  for (iy = 0; iy < Ny; iy++)
    y[iy] = -yMax + dy * (double)iy;
  lambda = calloc(Ncoeff, sizeof(double));
  tau = calloc(Ncoeff, sizeof(double));
  for (n = 0; n < Ncoeff; n++) {
    lambda[n] = PI * (double)n / (2.0 * xMax);
    tau[n] = PI * (double)n / (2.0 * yMax);
  }
  betaTop = calloc(Nfft, sizeof(COMPLEX *));
  betaBottom = calloc(Nfft, sizeof(COMPLEX *));
  betaRight = calloc(Nfft, sizeof(COMPLEX *));
  betaLeft = calloc(Nfft, sizeof(COMPLEX *));
  for (ik = 0; ik < Nfft; ik++) {
    betaTop[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaBottom[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaRight[ik] = calloc(Ncoeff, sizeof(COMPLEX));
    betaLeft[ik] = calloc(Ncoeff, sizeof(COMPLEX));
  }

#pragma omp parallel
  {
    COMPLEX *Bint;
    int ik, ix, n, myid;

    Bint = calloc(Nx, sizeof(COMPLEX));
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = ByTop[ix][ik].re;
          Bint[ix].im = ByTop[ix][ik].im;
        }
        betaTop[ik][0] = fourierCoeffIntegralTrap(Bint, x, lambda[0], dx, xMax, Nx);
        betaTop[ik][0].re = 0.5 * betaTop[ik][0].re;
        betaTop[ik][0].im = 0.5 * betaTop[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaTop[ik][n] = fourierCoeffIntegralLinInterp(Bint, x, lambda[n], dx, xMax, Nx);

        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = -ByBottom[ix][ik].re;
          Bint[ix].im = -ByBottom[ix][ik].im;
        }
        betaBottom[ik][0] = fourierCoeffIntegralTrap(Bint, x, lambda[0], dx, xMax, Nx);
        betaBottom[ik][0].re = 0.5 * betaBottom[ik][0].re;
        betaBottom[ik][0].im = 0.5 * betaBottom[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaBottom[ik][n] = fourierCoeffIntegralLinInterp(Bint, x, lambda[n], dx, xMax, Nx);
      }
#pragma omp barrier
    free(Bint);
  }

#pragma omp parallel
  {
    COMPLEX *Bint;
    int ik, iy, n, myid;

    Bint = calloc(Ny, sizeof(COMPLEX));
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = BxRight[iy][ik].re;
          Bint[iy].im = BxRight[iy][ik].im;
        }
        betaRight[ik][0] = fourierCoeffIntegralTrap(Bint, y, tau[0], dy, yMax, Ny);
        betaRight[ik][0].re = 0.5 * betaRight[ik][0].re;
        betaRight[ik][0].im = 0.5 * betaRight[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaRight[ik][n] = fourierCoeffIntegralLinInterp(Bint, y, tau[n], dy, yMax, Ny);

        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = -BxLeft[iy][ik].re;
          Bint[iy].im = -BxLeft[iy][ik].im;
        }
        betaLeft[ik][0] = fourierCoeffIntegralTrap(Bint, y, tau[0], dy, yMax, Ny);
        betaLeft[ik][0].re = 0.5 * betaLeft[ik][0].re;
        betaLeft[ik][0].im = 0.5 * betaLeft[ik][0].im;
        for (n = 1; n < Ncoeff; n++)
          betaLeft[ik][n] = fourierCoeffIntegralLinInterp(Bint, y, tau[n], dy, yMax, Ny);
      }
#pragma omp barrier
    free(Bint);
  }

  k = calloc(Nfft, sizeof(double));
  for (ik = 0; ik < Nfft / 2; ik++)
    k[ik] = dk * (double)ik;
  for (ik = Nfft / 2; ik < Nfft; ik++)
    k[ik] = -dk * (double)(Nfft - ik);
  genGradr_k = calloc(Ngrad, sizeof(COMPLEX *));
  for (ir = 0; ir < Ngrad; ir++)
    genGradr_k[ir] = calloc(Nfft, sizeof(COMPLEX));

  /* calculate skew gradients C_ir for ir != 0 */
  for (ir = 1; ir < Ngrad; ir++) {
    long ir1;
    if (fundamental)
      ir1 = fundamental * (2 * ir + 1); /* Is this right ? */
    else
      ir1 = ir;
    ik = 0;
    /* top and bottom need care for k->0 */
    genGradT = calcGGtopbottomk0B(betaTop[ik], lambda, yMax, ir1, Ncoeff);
    genGradB = calcGGtopbottomk0B(betaBottom[ik], lambda, yMax, ir1, Ncoeff);

    genGradR = calcGGrightk0(betaRight[ik], tau, xMax, ir1, Ncoeff);
    genGradL = calcGGleftk0(betaLeft[ik], tau, xMax, ir1, Ncoeff);

    genGradr_k[ir][ik].re = genGradT.re + genGradB.re + genGradR.re + genGradL.re;
    genGradr_k[ir][ik].im = genGradT.im + genGradB.im + genGradR.im + genGradL.im;
#pragma omp parallel
    {
      int ik, myid;
      COMPLEX genGradT, genGradB, genGradR, genGradL;
#if !defined(NOTHREADS)
      myid = omp_get_thread_num();
#else
      myid = 0;
#endif
      for (ik = 1; ik < Nfft; ik++)
        if (ik % threads == myid) {
          genGradT = calcGGtopbottomB(betaTop[ik], k[ik], lambda, yMax, ir1, Ncoeff);
          genGradB = calcGGtopbottomB(betaBottom[ik], k[ik], lambda, yMax, ir1, Ncoeff);
          genGradR = calcGGrightB(betaRight[ik], k[ik], tau, xMax, ir1, Ncoeff);
          genGradL = calcGGleftB(betaLeft[ik], k[ik], tau, xMax, ir1, Ncoeff);

          genGradr_k[ir][ik].re = genGradT.re + genGradB.re + genGradR.re + genGradL.re;
          genGradr_k[ir][ik].im = genGradT.im + genGradB.im + genGradR.im + genGradL.im;
        }
#pragma omp barrier
    }
  }

#pragma omp parallel
  {
    int ix, iy, myid;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ix = 0; ix < Nx; ix++)
      if (ix % threads == myid) {
        FFT(BzTop[ix], -1, Nfft);
        FFT(BzBottom[ix], -1, Nfft);
      }
    for (iy = 0; iy < Ny; iy++)
      if (iy % threads == myid) {
        FFT(BzRight[iy], -1, Nfft);
        FFT(BzLeft[iy], -1, Nfft);
      }
#pragma omp barrier
  }

#pragma omp parallel
  {
    /* calculate Fourier coefficients associated with Bz to get C_0 */
    int ik, ix, n, myid;
    COMPLEX *Bint;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    Bint = calloc(Nx, sizeof(COMPLEX));
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = BzTop[ix][ik].re;
          Bint[ix].im = BzTop[ix][ik].im;
        }
        betaTop[ik][0].re = 0.0;
        betaTop[ik][0].im = 0.0;
        for (n = 1; n < Ncoeff; n++)
          betaTop[ik][n] = fourierCoeffIntegralLinInterpSkew(Bint, x, lambda[n], dx, xMax, Nx);

        for (ix = 0; ix < Nx; ix++) {
          Bint[ix].re = BzBottom[ix][ik].re;
          Bint[ix].im = BzBottom[ix][ik].im;
        }
        betaBottom[ik][0].re = 0.0;
        betaBottom[ik][0].im = 0.0;
        for (n = 1; n < Ncoeff; n++)
          betaBottom[ik][n] = fourierCoeffIntegralLinInterpSkew(Bint, x, lambda[n], dx, xMax, Nx);
      }
#pragma omp barrier
    free(Bint);
  }

#pragma omp parallel
  {
    /* calculate Fourier coefficients associated with Bz to get C_0 */
    int ik, iy, n, myid;
    COMPLEX *Bint;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    Bint = calloc(Ny, sizeof(COMPLEX));
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = BzRight[iy][ik].re;
          Bint[iy].im = BzRight[iy][ik].im;
        }
        betaRight[ik][0].re = 0.0;
        betaRight[ik][0].im = 0.0;
        for (n = 1; n < Ncoeff; n++)
          betaRight[ik][n] = fourierCoeffIntegralLinInterpSkew(Bint, y, tau[n], dy, yMax, Ny);

        for (iy = 0; iy < Ny; iy++) {
          Bint[iy].re = BzLeft[iy][ik].re;
          Bint[iy].im = BzLeft[iy][ik].im;
        }
        betaLeft[ik][0].re = 0.0;
        betaLeft[ik][0].im = 0.0;
        for (n = 1; n < Ncoeff; n++)
          betaLeft[ik][n] = fourierCoeffIntegralLinInterpSkew(Bint, y, tau[n], dy, yMax, Ny);
      }
    free(Bint);
#pragma omp barrier
  }

#pragma omp parallel
  {
    int ik, myid;
    COMPLEX genGradT, genGradB, genGradR, genGradL;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ik = 0; ik < Nfft; ik++)
      if (ik % threads == myid) {
        genGradT = calcGGallSidesBz(betaTop[ik], k[ik], lambda, yMax, Ncoeff);
        genGradB = calcGGallSidesBz(betaBottom[ik], k[ik], lambda, yMax, Ncoeff);
        genGradR = calcGGallSidesBz(betaRight[ik], k[ik], tau, xMax, Ncoeff);
        genGradL = calcGGallSidesBz(betaLeft[ik], k[ik], tau, xMax, Ncoeff);
        genGradr_k[0][ik].re = genGradT.re + genGradB.re + genGradR.re + genGradL.re;
        genGradr_k[0][ik].im = genGradT.im + genGradB.im + genGradR.im + genGradL.im;
      }
#pragma omp barrier
  }

  invNfft = 1.0 / (double)Nfft;
  derivGG = calloc(Nderiv, sizeof(COMPLEX *));
  for (n = 0; n < Nderiv; n++)
    derivGG[n] = calloc(Nfft, sizeof(COMPLEX));

#ifdef DEBUG
  printf("Printing results...\n");
#endif

  if (SDDS_InitializeOutput(&SDDSOutput, SDDS_BINARY, 1, NULL, "computeRBGGE skew output", outputFile) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  if ((SDDS_DefineSimpleParameter(&SDDSOutput, "m", NULL, SDDS_LONG) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yCenter", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "xMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleParameter(&SDDSOutput, "yMax", "m", SDDS_DOUBLE) != 1) ||
      (SDDS_DefineSimpleColumn(&SDDSOutput, "z", "m", SDDS_DOUBLE) != 1)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }

  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "CnmC%" PRId32, n);
    if ((2 * n - 1) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 1));
    else if ((2 * n - 1) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 1));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }
  for (n = 0; n < Nderiv; n += 2) {
    sprintf(name, "dCnmC%" PRId32 "/dz", n);
    if ((2 * n - 2) < 0)
      sprintf(units, "T/m$a(m-%d)$n", -(2 * n - 2));
    else if ((2 * n - 2) == 0)
      sprintf(units, "T/m$am$n");
    else
      sprintf(units, "T/m$a(m+%d)$n", (2 * n - 2));
    if (SDDS_DefineSimpleColumn(&SDDSOutput, name, units, SDDS_DOUBLE) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }

  if (SDDS_WriteLayout(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  for (ir = 0; ir < Ngrad; ir++) {
    long NderivLimited;
    NderivLimited = Nderiv;
    if (varyDerivatives) {
      long m;
      m = (int32_t)(fundamental ? fundamental * (2 * ir + 1) : ir + 1);
      NderivLimited = (derivatives - m / 2) * 2 - 1;
    }
    /* Take derivatives */
    if (ir == 0)
      for (ik = 0; ik < Nfft; ik++) {
        derivGG[0][ik].re = genGradr_k[ir][ik].re;
        derivGG[0][ik].im = genGradr_k[ir][ik].im;
        genGradr_k[ir][ik].re = 0.0;
        genGradr_k[ir][ik].im = 0.0;
      }
    else
      for (ik = 0; ik < Nfft; ik++) {
        derivGG[0][ik].re = -k[ik] * genGradr_k[ir][ik].im;
        derivGG[0][ik].im = k[ik] * genGradr_k[ir][ik].re;
      }
    for (n = 1; n < NderivLimited; n++)
      for (ik = 0; ik < Nfft; ik++) {
        derivGG[n][ik].re = -k[ik] * derivGG[n - 1][ik].im;
        derivGG[n][ik].im = k[ik] * derivGG[n - 1][ik].re;
      }
    FFT(genGradr_k[ir], 1, Nfft);
    for (n = 0; n < NderivLimited; n++)
      FFT(derivGG[n], 1, Nfft);

    if (SDDS_StartPage(&SDDSOutput, Nz) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    if (SDDS_SetParameters(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                           "m", (fundamental ? fundamental * (2 * ir + 1) : ir), /* Is this right? */
                           "xCenter", xCenter,
                           "yCenter", yCenter,
                           "xMax", xMax,
                           "yMax", yMax,
                           NULL) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
    for (ik = 0; ik < Nz; ik++) {
      if (ir == 0) {
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              "z", zMin + dz * (double)ik,
                              "CnmC0", 0.0,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      } else {
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              "z", zMin + dz * (double)ik,
                              "CnmC0", genGradr_k[ir][ik].re * invNfft,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
      for (n = 2; n < Nderiv; n += 2) {
        sprintf(name, "CnmC%" PRId32, n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              name, n >= NderivLimited ? 0.0 : derivGG[n - 1][ik].re * invNfft,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
      for (n = 0; n < NderivLimited; n += 2) {
        sprintf(name, "dCnmC%" PRId32 "/dz", n);
        if (SDDS_SetRowValues(&SDDSOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ik,
                              name, n >= NderivLimited ? 0.0 : derivGG[n][ik].re * invNfft,
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
    }
    if (SDDS_WritePage(&SDDSOutput) != 1) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }
  if (SDDS_Terminate(&SDDSOutput) != 1) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }

  return (0);
}

COMPLEX fourierCoeffIntegralTrap(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx) {
  COMPLEX integral;

  int32_t ix, NxM = Nx - 1;

  integral.re = 0.5 * Bint[0].re * cos((x[0] + xMax) * lambdaN);
  integral.im = 0.5 * Bint[0].im * cos((x[0] + xMax) * lambdaN);
  for (ix = 1; ix < NxM; ix++) {
    integral.re += Bint[ix].re * cos((x[ix] + xMax) * lambdaN);
    integral.im += Bint[ix].im * cos((x[ix] + xMax) * lambdaN);
  }
  integral.re += 0.5 * Bint[NxM].re * cos((x[NxM] + xMax) * lambdaN);
  integral.im += 0.5 * Bint[NxM].im * cos((x[NxM] + xMax) * lambdaN);
  integral.re = dx * integral.re / xMax;
  integral.im = dx * integral.im / xMax;

  return (integral);
}

COMPLEX fourierCoeffIntegralSimp(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx) {
  COMPLEX integral;

  int32_t ix;

  integral.re = Bint[0].re * cos((x[0] + xMax) * lambdaN);
  integral.im = Bint[0].im * cos((x[0] + xMax) * lambdaN);
  for (ix = 1; ix < Nx - 2; ix++) {
    integral.re += 4.0 * Bint[ix].re * cos((x[ix] + xMax) * lambdaN);
    integral.im += 4.0 * Bint[ix].im * cos((x[ix] + xMax) * lambdaN);
    ix++;
    integral.re += 2.0 * Bint[ix].re * cos((x[ix] + xMax) * lambdaN);
    integral.im += 2.0 * Bint[ix].im * cos((x[ix] + xMax) * lambdaN);
  }
  integral.re += 4.0 * Bint[ix].re * cos((x[ix] + xMax) * lambdaN);
  integral.im += 4.0 * Bint[ix].im * cos((x[ix] + xMax) * lambdaN);
  ix++;
  integral.re += Bint[ix].re * cos((x[ix] + xMax) * lambdaN);
  integral.im += Bint[ix].im * cos((x[ix] + xMax) * lambdaN);
  integral.re = dx * integral.re / (3.0 * xMax);
  integral.im = dx * integral.im / (3.0 * xMax);

  return (integral);
}

COMPLEX fourierCoeffIntegralLinInterp(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx) {
  COMPLEX integral;

  double slope, yint, x0, x1;

  int32_t ix;

  integral.re = 0.0;
  integral.im = 0.0;
  for (ix = 0; ix < Nx - 1; ix++) {
    x0 = x[ix] + xMax;
    x1 = x[ix + 1] + xMax;
    slope = (Bint[ix + 1].re - Bint[ix].re) / dx;
    yint = (x[ix + 1] * Bint[ix].re - x[ix] * Bint[ix + 1].re) / dx;
    integral.re += (yint / lambdaN) * (sin(x1 * lambdaN) - sin(x0 * lambdaN));
    integral.re += (slope / lambdaN) * (x[ix + 1] * sin(x1 * lambdaN) - x[ix] * sin(x0 * lambdaN) + (cos(x1 * lambdaN) - cos(x0 * lambdaN)) / lambdaN);
    slope = (Bint[ix + 1].im - Bint[ix].im) / dx;
    yint = (x[ix + 1] * Bint[ix].im - x[ix] * Bint[ix + 1].im) / dx;
    integral.im += (yint / lambdaN) * (sin(x1 * lambdaN) - sin(x0 * lambdaN));
    integral.im += (slope / lambdaN) * (x[ix + 1] * sin(x1 * lambdaN) - x[ix] * sin(x0 * lambdaN) + (cos(x1 * lambdaN) - cos(x0 * lambdaN)) / lambdaN);
  }
  integral.re = integral.re / xMax;
  integral.im = integral.im / xMax;

  return (integral);
}

COMPLEX fourierCoeffIntegralLinInterpSkew(COMPLEX *Bint, double *x, double lambdaN, double dx, double xMax, int32_t Nx) {
  COMPLEX integral;

  double slope, yint, x0, x1;

  int32_t ix;

  integral.re = 0.0;
  integral.im = 0.0;
  for (ix = 0; ix < Nx - 1; ix++) {
    x0 = x[ix] + xMax;
    x1 = x[ix + 1] + xMax;
    slope = (Bint[ix + 1].re - Bint[ix].re) / dx;
    yint = (x[ix + 1] * Bint[ix].re - x[ix] * Bint[ix + 1].re) / dx;
    integral.re -= (yint / lambdaN) * (cos(x1 * lambdaN) - cos(x0 * lambdaN));
    integral.re += (slope / lambdaN) * (-x[ix + 1] * cos(x1 * lambdaN) + x[ix] * cos(x0 * lambdaN) + (sin(x1 * lambdaN) - sin(x0 * lambdaN)) / lambdaN);
    slope = (Bint[ix + 1].im - Bint[ix].im) / dx;
    yint = (x[ix + 1] * Bint[ix].im - x[ix] * Bint[ix + 1].im) / dx;
    integral.im -= (yint / lambdaN) * (cos(x1 * lambdaN) - cos(x0 * lambdaN));
    integral.im += (slope / lambdaN) * (-x[ix + 1] * cos(x1 * lambdaN) + x[ix] * cos(x0 * lambdaN) + (sin(x1 * lambdaN) - sin(x0 * lambdaN)) / lambdaN);
  }
  integral.re = integral.re / xMax;
  integral.im = integral.im / xMax;

  return (integral);
}

COMPLEX calcGGtopbottomA(COMPLEX *beta, double k, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double klam, splus, sminus;
  double rcoef, factor;
  int32_t theta0r, deltan, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  power = grad_r / 2;
  for (i = 0; i < power; i++)
    rcoef = -1.0 * rcoef; /* (-1)^{floor[r/2]} */
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 0; n < Ncoeff; n++) {
    if (n % 2 == 0)
      deltan = theta0r;
    else
      deltan = 1 - theta0r;
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign;
    klam = sqrt(k * k + lambda[n] * lambda[n]);
    splus = lambda[n] + klam;
    sminus = lambda[n] - klam;
    for (i = 1; i < grad_r; i++) {
      splus = splus * (lambda[n] + klam);   /* s_{+}^r */
      sminus = sminus * (lambda[n] - klam); /* s_{-}^r */
    }
    factor = (double)(sign * deltan) * rcoef / (klam * 2.0 * cosh(yMax * klam));
    genGrad.re += (splus - sminus) * factor * beta[n].re;
    genGrad.im += (splus - sminus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGtopbottomB(COMPLEX *beta, double k, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double klam, splus, sminus;
  double rcoef, factor;
  int32_t sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */

  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 0; n < Ncoeff; n++) {
    if ((grad_r + n) % 2 == 0) {
      sign = -1;
      power = (grad_r + n) / 2;
      for (i = 1; i < power; i++)
        sign = -1 * sign; /* (-1)^{(ir+n)/2} */
    } else
      sign = 0;
    klam = sqrt(k * k + lambda[n] * lambda[n]);
    splus = lambda[n] + klam;
    sminus = lambda[n] - klam;
    for (i = 1; i < grad_r; i++) {
      splus = splus * (lambda[n] + klam);   /* s_{+}^r */
      sminus = sminus * (lambda[n] - klam); /* s_{-}^r */
    }
    factor = (double)sign * rcoef / (klam * 2.0 * sinh(yMax * klam));
    genGrad.re += (splus + sminus) * factor * beta[n].re;
    genGrad.im += (splus + sminus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGrightA(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double ktau, qplus, qminus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 1; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    ktau = sqrt(k * k + tau[n] * tau[n]);
    qplus = -(tau[n] + ktau);
    qminus = tau[n] - ktau;
    for (i = 1; i < grad_r; i++) {
      qplus = -qplus * (tau[n] + ktau);  /* (-s_{+})^r */
      qminus = qminus * (tau[n] - ktau); /* s_{-}^r */
    }
    factor = rcoef * (-(double)(sign * theta0r) / (ktau * 2.0 * sinh(xMax * ktau)) + (double)(sign * (1 - theta0r)) / (ktau * 2.0 * cosh(xMax * ktau)));
    genGrad.re += (qminus - qplus) * factor * beta[n].re;
    genGrad.im += (qminus - qplus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGrightB(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double ktau, qplus, qminus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 0; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    ktau = sqrt(k * k + tau[n] * tau[n]);
    qplus = -(tau[n] + ktau);
    qminus = tau[n] - ktau;
    for (i = 1; i < grad_r; i++) {
      qplus = -qplus * (tau[n] + ktau);  /* (-s_{+})^r */
      qminus = qminus * (tau[n] - ktau); /* s_{-}^r */
    }
    factor = rcoef * ((double)(sign * (1 - theta0r)) / (ktau * 2.0 * sinh(xMax * ktau)) - (double)(sign * theta0r) / (ktau * 2.0 * cosh(xMax * ktau)));
    genGrad.re += (qminus + qplus) * factor * beta[n].re;
    genGrad.im += (qminus + qplus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGleftA(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double ktau, qplus, qminus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 1; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    ktau = sqrt(k * k + tau[n] * tau[n]);
    qplus = -(tau[n] + ktau);
    qminus = tau[n] - ktau;
    for (i = 1; i < grad_r; i++) {
      qplus = -qplus * (tau[n] + ktau);  /* (-s_{+})^r */
      qminus = qminus * (tau[n] - ktau); /* s_{-}^r */
    }
    factor = rcoef * (-(double)(sign * theta0r) / (ktau * 2.0 * sinh(xMax * ktau)) - (double)(sign * (1 - theta0r)) / (ktau * 2.0 * cosh(xMax * ktau)));
    genGrad.re += (qminus - qplus) * factor * beta[n].re;
    genGrad.im += (qminus - qplus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGleftB(COMPLEX *beta, double k, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double ktau, qplus, qminus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 0; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    ktau = sqrt(k * k + tau[n] * tau[n]);
    qplus = -(tau[n] + ktau);
    qminus = tau[n] - ktau;
    for (i = 1; i < grad_r; i++) {
      qplus = -qplus * (tau[n] + ktau);  /* (-s_{+})^r */
      qminus = qminus * (tau[n] - ktau); /* s_{-}^r */
    }
    factor = rcoef * ((double)(sign * (1 - theta0r)) / (ktau * 2.0 * sinh(xMax * ktau)) + (double)(sign * theta0r) / (ktau * 2.0 * cosh(xMax * ktau)));
    genGrad.re += (qminus + qplus) * factor * beta[n].re;
    genGrad.im += (qminus + qplus) * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGtopbottomk0A(COMPLEX *beta, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double splus;
  double rcoef, factor;
  int32_t theta0r, deltan, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  power = grad_r / 2;
  for (i = 0; i < power; i++)
    rcoef = -1.0 * rcoef; /* (-1)^{floor[r/2]} */
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 0; n < Ncoeff; n++) {
    if (n % 2 == 0)
      deltan = theta0r;
    else
      deltan = 1 - theta0r;
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    splus = 1.0;
    for (i = 1; i < grad_r; i++)
      splus = splus * 2.0 * lambda[n]; /* (2*lambda)^(grad_r-1) */
    factor = (double)(sign * deltan) * rcoef * splus / (cosh(yMax * lambda[n]));
    genGrad.re += factor * beta[n].re;
    genGrad.im += factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGtopbottomk0B(COMPLEX *beta, double *lambda, double yMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double splus;
  double rcoef, factor;
  int32_t sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  // printf("1/rcoef = %e\n", 1.0/rcoef);
  if (grad_r == 2) {
    genGrad.re = -(rcoef / yMax) * beta[0].re;
    genGrad.im = -(rcoef / yMax) * beta[0].im;
  } else {
    genGrad.re = 0.0;
    genGrad.im = 0.0;
  }
  for (n = 1; n < Ncoeff; n++) {
    if ((grad_r + n) % 2 == 0) {
      sign = -1;
      power = (grad_r + n) / 2;
      for (i = 1; i < power; i++)
        sign = -1 * sign; /* (-1)^{(ir+n)/2} */
    } else
      sign = 0;
    splus = 1.0;
    for (i = 1; i < grad_r; i++)
      splus = splus * 2.0 * lambda[n]; /* (2*lambda)^(grad_r-1) */
    factor = (double)sign * rcoef * splus / (sinh(yMax * lambda[n]));
    genGrad.re += factor * beta[n].re;
    genGrad.im += factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGrightk0(COMPLEX *beta, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double qplus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  switch (grad_r) {
  case 1:
    genGrad.re = rcoef * beta[0].re;
    genGrad.im = rcoef * beta[0].im;
    break;
  case 2:
    genGrad.re = (rcoef * 2.0 / xMax) * beta[0].re;
    genGrad.im = (rcoef * 2.0 / xMax) * beta[0].im;
    break;
  default:
    genGrad.re = 0.0;
    genGrad.im = 0.0;
  }
  for (n = 2; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    qplus = -1.0;
    for (i = 1; i < grad_r; i++)
      qplus = -qplus * 2.0 * tau[n]; /* (-s_{+})^{r-1} */
    factor = rcoef * ((double)(sign * (1 - theta0r)) / sinh(xMax * tau[n]) - (double)(sign * theta0r) / cosh(xMax * tau[n]));

    genGrad.re += qplus * factor * beta[n].re;
    genGrad.im += qplus * factor * beta[n].im;
  }
  return (genGrad);
}

COMPLEX calcGGleftk0(COMPLEX *beta, double *tau, double xMax, int32_t grad_r, int32_t Ncoeff) {
  COMPLEX genGrad;
  double qplus;
  double rcoef, factor;
  int32_t theta0r, sign, power;

  int32_t n, i;

  rcoef = 1.0;
  for (i = grad_r; i >= 1; i--)
    rcoef *= 0.5 / (double)i; /* 1/(r!2^r) */
  if (grad_r % 2 == 0)
    theta0r = 0;
  else
    theta0r = 1;
  switch (grad_r) {
  case 1:
    genGrad.re = -rcoef * beta[0].re;
    genGrad.im = -rcoef * beta[0].im;
    break;
  case 2:
    genGrad.re = (rcoef * 2.0 / xMax) * beta[0].re;
    genGrad.im = (rcoef * 2.0 / xMax) * beta[0].im;
    break;
  default:
    genGrad.re = 0.0;
    genGrad.im = 0.0;
  }
  for (n = 2; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    qplus = -1.0;
    for (i = 1; i < grad_r; i++)
      qplus = -qplus * 2.0 * tau[n]; /* (-s_{+})^{r-1} */
    factor = rcoef * ((double)(sign * (1 - theta0r)) / sinh(xMax * tau[n]) + (double)(sign * theta0r) / cosh(xMax * tau[n]));
    genGrad.re += qplus * factor * beta[n].re;
    genGrad.im += qplus * factor * beta[n].im;
  }

  return (genGrad);
}

COMPLEX calcGGallSidesBz(COMPLEX *beta, double k, double *lambda, double yMax, int32_t Ncoeff) {
  COMPLEX genGrad;
  double klambda, factor;
  int32_t sign, power;

  int32_t n, i;

  genGrad.re = 0.0;
  genGrad.im = 0.0;
  for (n = 1; n < Ncoeff; n += 2) {
    sign = 1;
    power = n / 2;
    for (i = 0; i < power; i++)
      sign = -1 * sign; /* (-1)^{floor[n/2]} */
    klambda = sqrt(k * k + lambda[n] * lambda[n]);
    factor = (double)sign / (2.0 * cosh(yMax * klambda));
    genGrad.re += factor * beta[n].re;
    genGrad.im += factor * beta[n].im;
  }

  return (genGrad);
}

#ifndef OLDFFT
void FFT(COMPLEX *field, int32_t isign, int32_t npts) {
  double *real_imag;
  int32_t i;
#  ifdef DEBUG
  static FILE *fpfft = NULL;
  if (!fpfft) {
    fpfft = fopen("computeRBGGE.fft", "w");
    fprintf(fpfft, "SDDS1\n");
    fprintf(fpfft, "&column name=i type=long &end\n");
    fprintf(fpfft, "&column name=Real type=double &end\n");
    fprintf(fpfft, "&column name=Imag type=double &end\n");
    fprintf(fpfft, "&data mode=ascii &end\n");
  }
#  endif

  real_imag = tmalloc(sizeof(double) * (2 * npts + 2));
  for (i = 0; i < npts; i++) {
    real_imag[2 * i] = field[i].re;
    real_imag[2 * i + 1] = field[i].im;
  }
  if (isign == -1) {
    complexFFT(real_imag, npts, 0);
    for (i = 0; i < npts; i++) {
      field[i].re = npts * real_imag[2 * i];
      field[i].im = npts * real_imag[2 * i + 1];
    }
  } else {
    complexFFT(real_imag, npts, INVERSE_FFT);
    for (i = 0; i < npts; i++) {
      field[i].re = real_imag[2 * i];
      field[i].im = real_imag[2 * i + 1];
    }
  }
#  ifdef DEBUG
  fprintf(fpfft, "%" PRId32 "\n", npts);
  for (i = 0; i < npts; i++)
    fprintf(fpfft, "%" PRId32 " %le %le\n", i, field[i].re, field[i].im);
#  endif
  free(real_imag);
}
#else
void FFT(COMPLEX *field, int32_t isign, int32_t npts) {
  unsigned long mmax, m, hn, j, istep, i;
  double wtemp, wr, wpr, wpi, wi, theta, flt_isign;
  /* must be double to preserve accuracy */
  COMPLEX tempz;

  COMPLEX *fft;
  double tempr, tempi;

#  ifdef DEBUG
  static FILE *fpfft = NULL;
  if (!fpfft) {
    fpfft = fopen("computeRBGGE.fft", "w");
    fprintf(fpfft, "SDDS1\n");
    fprintf(fpfft, "&column name=i type=long &end\n");
    fprintf(fpfft, "&column name=Real type=double &end\n");
    fprintf(fpfft, "&column name=Imag type=double &end\n");
    fprintf(fpfft, "&data mode=ascii &end\n");
  }
#  endif

  fft = calloc(npts, sizeof(COMPLEX));

  hn = npts / 2;

  /* copy over */
  for (i = 0; i < npts; i++)
    fft[i] = field[i];

  /* Fourier Transform isign = -1, inverse isign = 1 */

  flt_isign = (double)(isign);
  /* first, sort into bit-reversed order */
  for (j = 0, i = 0; i < npts; i++) /* increment in regular order */
    {
      if (j > i) /* swap i and j = bit-reversal(i) once, if distinct */
        {
          tempz = fft[i];
          fft[i] = fft[j];
          fft[j] = tempz;
        }
      for (m = hn; ((j >= m) && (m >= 1)); m >>= 1)
        /* find bit-reversal of next i */
        {
          j -= m;
        }
      j += m;
    }

  /* next, apply Danielson-Lanczos algorithm */
  for (mmax = 1; (npts > mmax); mmax = istep)
    /* loop through log_base2(N) times */
    {
      istep = mmax << 1; /* = 2*mmax */
      /* initialize trig functions */
      theta = (flt_isign) * (PI / ((double)(mmax)));
      wtemp = sin(0.5 * theta);
      wpr = -2.0 * wtemp * wtemp;
      wpi = sin(theta);
      wr = 1.0;
      wi = 0.0;
      for (m = 0; m < mmax; m += 1) {
        for (i = m; i < npts; i += istep) {
          j = i + mmax;
          tempr = wr * fft[j].re - wi * fft[j].im;
          tempi = wr * fft[j].im + wi * fft[j].re;
          fft[j].re = fft[i].re - tempr;
          fft[j].im = fft[i].im - tempi;
          fft[i].re += tempr;
          fft[i].im += tempi;
        }
        /* update trig functions via recurrence relations */
        wr = wr + (wtemp = wr) * wpr - wi * wpi;
        wi = wi + wi * wpr + wtemp * wpi;
      }
    }

  /* copy */
  for (i = 0; i < npts; i++) {
    field[i] = fft[i];
  }
  free(fft);

#  ifdef DEBUG
  fprintf(fpfft, "%ld\n", npts);
  for (i = 0; i < npts; i++)
    fprintf(fpfft, "%ld %le %le\n", i, field[i].re, field[i].im);
#  endif
}
#endif

unsigned long IntCeilingPowerOf2(unsigned long i) {
  /* returns smallest non-negative x = 2^n  such that i <= x */

  unsigned long x;

  for (x = 1; (i > x); x <<= 1) {
  }
  return x;
}

void copyComplexArray(COMPLEX **target, COMPLEX **source, long nxy, long nz) {
  long ixy, iz;
  for (iz = 0; iz < nz; iz++)
    for (ixy = 0; ixy < nxy; ixy++) {
      target[ixy][iz].re = source[ixy][iz].re;
      target[ixy][iz].im = source[ixy][iz].im;
    }
}

COMPLEX **createComplexArray(long nxy, long nz, double *value) {
  COMPLEX **array;
  long ixy, iz, n;
  array = calloc(nxy, sizeof(COMPLEX *));
  for (ixy = 0; ixy < nxy; ixy++)
    array[ixy] = calloc(nz, sizeof(COMPLEX));
  n = 0;
  for (iz = 0; iz < nz; iz++) {
    for (ixy = 0; ixy < nxy; ixy++) {
      array[ixy][iz].re = value[n];
      n++;
    }
  }
  return array;
}

void readFieldMap(char *fieldMapFile, FIELD_MAP *fmData) {
  SDDS_DATASET SDDSin;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, fieldMapFile))
    SDDS_Bomb("unable to read field input file");
  if (SDDS_CheckColumn(&SDDSin, "Bx", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "By", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "Bz", "T", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)
    SDDS_Bomb("Didn't find required field columns Bx, By, Bz in T");
  if (SDDS_CheckColumn(&SDDSin, "x", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "y", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "z", "m", SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY)
    SDDS_Bomb("Didn't find required coordinate columns x, y, z in T");
  if (SDDS_ReadPage(&SDDSin) <= 0 ||
      !(fmData->x = SDDS_GetColumnInDoubles(&SDDSin, "x")) ||
      !(fmData->y = SDDS_GetColumnInDoubles(&SDDSin, "y")) ||
      !(fmData->z = SDDS_GetColumnInDoubles(&SDDSin, "z")) ||
      !(fmData->Bx = SDDS_GetColumnInDoubles(&SDDSin, "Bx")) ||
      !(fmData->By = SDDS_GetColumnInDoubles(&SDDSin, "By")) ||
      !(fmData->Bz = SDDS_GetColumnInDoubles(&SDDSin, "Bz")))
    SDDS_Bomb("unable to get data from field input file");
  if (!(fmData->n = SDDS_CountRowsOfInterest(&SDDSin)) || fmData->n < 1)
    SDDS_Bomb("field map file has insufficient data");
  SDDS_Terminate(&SDDSin);
}

void freeBGGExpData(BGGEXP_DATA *bggexpData) {
  long im, id;
  if (bggexpData->haveData) {
    for (im = 0; im < bggexpData->nm; im++) {
      for (id = 0; id < bggexpData->nGradients; id++) {
        if (bggexpData->Cmn[im][id])
          free(bggexpData->Cmn[im][id]);
        if (bggexpData->dCmn_dz[im][id])
          free(bggexpData->dCmn_dz[im][id]);
        bggexpData->Cmn[im][id] = bggexpData->dCmn_dz[im][id] = NULL;
      }
    }
    free(bggexpData->m);
    bggexpData->m = NULL;
    bggexpData->nm = bggexpData->nGradients = 0;
  }
}

#define BUFSIZE 1024

#ifdef DEBUG
void spewBGGExpData(BGGEXP_DATA *bgg) {
  FILE *fp;
  long im, ig, iz;
  fp = fopen("checkBGGExp.sdds", "w");
  fprintf(fp, "SDDS1\n&parameter name=m type=long &end\n");
  fprintf(fp, "&column name=z type=double units=m &end\n");
  for (ig = 0; ig < bgg->nGradients; ig++) {
    fprintf(fp, "&column name=CnmS%ld, type=double &end\n", 2 * ig);
    fprintf(fp, "&column name=dCnmS%ld/dz, type=double &end\n", 2 * ig);
  }
  fprintf(fp, "&data mode=ascii &end\n");
  for (im = 0; im < bgg->nm; im++) {
    fprintf(fp, "%ld\n%ld\n", bgg->m[im], bgg->nz);
    for (iz = 0; iz < bgg->nz; iz++) {
      fprintf(fp, "%le ", bgg->zMin + (bgg->zMax - bgg->zMin) / (bgg->nz - 1.0) * iz);
      for (ig = 0; ig < bgg->nGradients; ig++) {
        fprintf(fp, "%le %le ",
                bgg->Cmn[im][ig][iz], bgg->dCmn_dz[im][ig][iz]);
      }
      fputc('\n', fp);
    }
  }
  fclose(fp);
}
#endif

void readBGGExpData(BGGEXP_DATA *bggexpData, char *filename, char *nameFragment, short skew) {
  SDDS_DATASET SDDSin;
  char buffer[BUFSIZE];
  long im, ic, nc, readCode, nz;
  int32_t m;
  short xCenterPresent = 0, yCenterPresent = 0, xMaxPresent = 0, yMaxPresent = 0;

  if (!SDDS_InitializeInput(&SDDSin, filename)) {
    fprintf(stderr, "Unable to read file %s\n", filename);
    exit(1);
  }
  bggexpData->haveData = 1;

  /* Check presence of z column */
  if (SDDS_CheckColumn(&SDDSin, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OK) {
    fprintf(stderr, "Unable to find floating-point column \"z\" with units \"m\" in file %s\n", filename);
    exit(1);
  }

  /* Check presence of Cnm* columns */
  ic = 0;
  while (1) {
    snprintf(buffer, BUFSIZE, "%s%ld", nameFragment, 2 * ic);
    if (SDDS_CheckColumn(&SDDSin, buffer, NULL, SDDS_ANY_FLOATING_TYPE, NULL) != SDDS_CHECK_OK)
      break;
    ic++;
  }
  if (ic == 0) {
    fprintf(stderr, "Unable to find any floating-point columns %s* in file %s\n",
            nameFragment, filename);
    exit(1);
  }
  nc = ic;

  /* Check for presence of matching dCnmXXX/dz columns */
  for (ic = 0; ic < nc; ic++) {
    snprintf(buffer, BUFSIZE, "d%s%ld/dz", nameFragment, 2 * ic);
    if (SDDS_CheckColumn(&SDDSin, buffer, NULL, SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OK)
      break;
  }
  if (ic != nc) {
    fprintf(stderr, "Unable to find matching floating-point columns dCnm*/dz in file %s\n", filename);
    exit(1);
  }

  bggexpData->nm = bggexpData->nz = 0;
  bggexpData->nGradients = nc;
  bggexpData->m = NULL;
  bggexpData->Cmn = NULL;
  bggexpData->dCmn_dz = NULL;

  im = nz = 0;
  bggexpData->zMin = DBL_MAX;
  bggexpData->zMax = -DBL_MAX;
  bggexpData->xCenter = bggexpData->yCenter = 0;
  bggexpData->xMax = bggexpData->yMax = -1;
  while ((readCode = SDDS_ReadPage(&SDDSin)) > 0) {
    if (!SDDS_GetParameterAsLong(&SDDSin, "m", &m) || (m < 1 && !skew) || (m < 0 && skew)) {
      fprintf(stderr, "Problem with value of m (m<%d) for page %ld of file %s\n",
              (skew ? 0 : 1), readCode, filename);
      exit(1);
    }
    if (readCode == 1) {
      long iz;
      double dz0, dz, *z, zMin, zMax;
      if ((xCenterPresent && !SDDS_GetParameterAsDouble(&SDDSin, "xCenter", &(bggexpData->xCenter))) ||
          (yCenterPresent && !SDDS_GetParameterAsDouble(&SDDSin, "yCenter", &(bggexpData->yCenter)))) {
        fprintf(stderr, "Problem getting xCenter or yCenter values from file %s\n",
                filename);
        exit(1);
      }
      if ((xMaxPresent && !SDDS_GetParameterAsDouble(&SDDSin, "xMax", &(bggexpData->xMax))) ||
          (yMaxPresent && !SDDS_GetParameterAsDouble(&SDDSin, "yMax", &(bggexpData->yMax)))) {
        fprintf(stderr, "Problem getting xMax or yMax values from file %s\n",
                filename);
        exit(1);
      }
      if ((nz = SDDS_RowCount(&SDDSin)) <= 1) {
        fprintf(stderr, "Too few z values in file %s\n", filename);
        exit(1);
      }
      if (!(z = SDDS_GetColumnInDoubles(&SDDSin, "z"))) {
        fprintf(stderr, "Problem reading column z from %s\n", filename);
        exit(1);
      }
      find_min_max(&zMin, &zMax, z, nz);
      if (zMin < bggexpData->zMin)
        bggexpData->zMin = zMin;
      if (zMax > bggexpData->zMax)
        bggexpData->zMax = zMax;
      dz0 = z[1] - z[0];
      for (iz = 1; iz < nz; iz++) {
        dz = z[iz] - z[iz - 1];
        if (dz <= 0 || fabs(dz0 / dz - 1) > 1e-6) {
          fprintf(stderr, "Data not uniformly and monotonically increasing in z column from %s\n", filename);
          exit(1);
        }
      }
      free(z);
      bggexpData->dz = dz0;
    } else {
      if (nz != SDDS_RowCount(&SDDSin)) {
        fprintf(stderr, "Inconsistent number of z values in file %s\n", filename);
        exit(1);
      }
    }
    if (!(bggexpData->m = SDDS_Realloc(bggexpData->m,
                                       sizeof(*bggexpData->m) * (im + 1))) ||
        !(bggexpData->Cmn = SDDS_Realloc(bggexpData->Cmn,
                                         sizeof(*bggexpData->Cmn) * (im + 1))) ||
        !(bggexpData->dCmn_dz = SDDS_Realloc(bggexpData->dCmn_dz,
                                             sizeof(*bggexpData->dCmn_dz) * (im + 1)))) {
      fprintf(stderr, "Memory allocation failure (1) loading data from file %s\n", filename);
      exit(1);
    }
    bggexpData->Cmn[im] = NULL;
    bggexpData->dCmn_dz[im] = NULL;
    if (!(bggexpData->Cmn[im] = malloc(sizeof(*bggexpData->Cmn[im]) * nc)) ||
        !(bggexpData->dCmn_dz[im] = malloc(sizeof(*bggexpData->dCmn_dz[im]) * nc)))
      fprintf(stderr, "Memory allocation failure (2) loading data from file %s\n", filename);

    for (ic = 0; ic < nc; ic++) {
      snprintf(buffer, BUFSIZE, "%s%ld", nameFragment, 2 * ic);
      if (!(bggexpData->Cmn[im][ic] = SDDS_GetColumnInDoubles(&SDDSin, buffer))) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        fprintf(stderr, "Problem reading column %s from %s\n", buffer, filename);
        exit(1);
      }
      snprintf(buffer, BUFSIZE, "d%s%ld/dz", nameFragment, 2 * ic);
      if (!(bggexpData->dCmn_dz[im][ic] = SDDS_GetColumnInDoubles(&SDDSin, buffer))) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        fprintf(stderr, "Problem reading column %s from %s\n", buffer, filename);
        exit(1);
      }
    }

    bggexpData->m[im] = m;
    im++;
  }

  SDDS_Terminate(&SDDSin);

  bggexpData->nm = im;
  bggexpData->nz = nz;
}

double evaluateGGEForFieldMap(FIELD_MAP *fmap, BGGEXP_DATA *bggexpData, FIELDS_ON_PLANES *fieldsOnPlanes,
                              long multipoles, long derivatives, double significance, double *coordLimit,
                              unsigned long flags, long varyDerivatives, ALL_RESIDUALS *allResiduals) {
  double *residualSum, *residualSum2, *residualWorst, *maxField;
  long *residualCount, i;
#ifdef DEBUG
  FILE *fpdeb;

  fpdeb = fopen("computeRBGGE.debug", "w");
  fprintf(fpdeb, "SDDS1\n");
  fprintf(fpdeb, "&column name=x type=float units=m &end\n");
  fprintf(fpdeb, "&column name=y type=float units=m &end\n");
  fprintf(fpdeb, "&column name=z type=float units=m &end\n");
  fprintf(fpdeb, "&column name=Bx type=float units=T &end\n");
  fprintf(fpdeb, "&column name=By type=float units=T &end\n");
  fprintf(fpdeb, "&column name=Bz type=float units=T &end\n");
  fprintf(fpdeb, "&column name=BxRef type=float units=T &end\n");
  fprintf(fpdeb, "&column name=ByRef type=float units=T &end\n");
  fprintf(fpdeb, "&column name=BzRef type=float units=T &end\n");
  fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
#endif

#ifdef DEBUG
  spewBGGExpData(bggexpData);
#endif

  residualWorst = calloc(threads, sizeof(*residualWorst));
  residualSum = calloc(threads, sizeof(*residualSum));
  residualSum2 = calloc(threads, sizeof(*residualSum2));
  residualCount = calloc(threads, sizeof(*residualCount));
  maxField = calloc(threads, sizeof(*maxField));

#pragma omp parallel
  {
    double field;
    double B[3], Br, Bphi, residualTerm;
    double x, y, z, r, phi, dz;
    long ip, ns, iz, ig, m, im;
    int myid;
#if !defined(NOTHREADS)
    myid = omp_get_thread_num();
#else
    myid = 0;
#endif
    for (ip = 0; ip < fmap->n; ip++) {
      if (ip % threads == myid) {
        if (fmap->x[ip] > fieldsOnPlanes->xMax || fmap->x[ip] < fieldsOnPlanes->xMin ||
            fmap->y[ip] > fieldsOnPlanes->yMax || fmap->y[ip] < fieldsOnPlanes->yMin)
          continue;
        if ((coordLimit[0] > 0 && fabs(fmap->x[ip]) > coordLimit[0]) ||
            (coordLimit[1] > 0 && fabs(fmap->y[ip]) > coordLimit[1]))
          continue;
        if (coordLimit[2] > 0) {
          r = sqrt(sqr(fmap->x[ip]) + sqr(fmap->y[ip]));
          if (r > coordLimit[2])
            continue;
        }
        /* Compute fields */
        Br = Bphi = B[0] = B[1] = B[2] = 0;
        r = phi = dz = 0;
        iz = 0;
        for (ns = 0; ns < 2; ns++) {
          /* ns=0 => normal, ns=1 => skew */
          if (!bggexpData[ns].haveData)
            continue;
          x = fmap->x[ip] - bggexpData[ns].xCenter;
          y = fmap->y[ip] - bggexpData[ns].yCenter;
          z = fmap->z[ip];
          dz = (bggexpData[ns].zMax - bggexpData[ns].zMin) / (bggexpData[ns].nz - 1);
          iz = (z - bggexpData[ns].zMin) / dz + 0.5;
          if (fabs(iz * dz + bggexpData[ns].zMin - z) > 1e-3 * dz || iz < 0 || iz >= bggexpData[ns].nz) {
            fprintf(stderr, "evaluation points in the 3d field map need to be at the same z planes as the input data\n");
            fprintf(stderr, "no match for z=%15.8le, dz=%15.8le, zMin=%15.8le, iz=%ld\n", z, dz, bggexpData[ns].zMin, iz);
            exit(1);
          }

          r = sqrt(sqr(x) + sqr(y));
          phi = atan2(y, x);

          for (im = 0; im < bggexpData[ns].nm && im < multipoles; im++) {
            double mfact, term, sin_mphi, cos_mphi;
            long ndLimit;
            m = bggexpData[ns].m[im];
            mfact = dfactorial(m);
            sin_mphi = sin(m * phi);
            cos_mphi = cos(m * phi);
            ndLimit = MIN(bggexpData[ns].nGradients, derivatives);
            if (varyDerivatives)
              ndLimit -= m / 2;
            if (ns == 0) {
              /* normal */
              for (ig = 0; ig < ndLimit; ig++) {
                term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r * sin_mphi;
                term *= bggexpData[ns].Cmn[im][ig][iz];
                Br += term * (2 * ig + m) * sin_mphi;
                Bphi += m * term * cos_mphi;
              }
            } else {
              /* skew */
              if (m == 0) {
                B[2] += bggexpData[ns].dCmn_dz[im][0][iz]; // on-axis Bz from m=ig=0 term
                for (ig = 1; ig < ndLimit; ig++) {
                  term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                  B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r;
                  Br += term * (2 * ig + m) * bggexpData[ns].Cmn[im][ig][iz];
                }
              } else {
                for (ig = 0; ig < ndLimit; ig++) {
                  term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                  B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r * cos_mphi;
                  term *= bggexpData[ns].Cmn[im][ig][iz];
                  Br += term * (2 * ig + m) * cos_mphi;
                  Bphi -= m * term * sin_mphi;
                }
              }
            }
          }
        }
        B[0] = Br * cos(phi) - Bphi * sin(phi);
        B[1] = Br * sin(phi) + Bphi * cos(phi);
#ifdef DEBUG
        fprintf(fpdeb, "%le %le %le %le %le %le %le %le %le\n",
                fmap->x[ip], fmap->y[ip], iz * dz + bggexpData[0].zMin,
                B[0], B[1], B[2], fmap->Bx[ip], fmap->By[ip], fmap->Bz[ip]);
#endif
        field = sqrt(sqr(B[0]) + sqr(B[1]) + sqr(B[2]));
        if (field > maxField[myid])
          maxField[myid] = field;
        if ((residualTerm = sqrt(sqr(B[0] - fmap->Bx[ip]) + sqr(B[1] - fmap->By[ip]) + sqr(B[2] - fmap->Bz[ip]))) > residualWorst[myid])
          residualWorst[myid] = residualTerm;
        residualCount[myid]++;
        residualSum[myid] += fabs(residualTerm);
        residualSum2[myid] += sqr(residualTerm);
      }
    }
#pragma omp barrier
  }

  for (i = 1; i < threads; i++) {
    residualWorst[0] = MAX(residualWorst[0], residualWorst[i]);
    residualSum[0] += residualSum[i];
    residualSum2[0] += residualSum2[i];
    residualCount[0] += residualCount[i];
    maxField[0] = MAX(maxField[0], maxField[i]);
  }

#ifdef DEBUG
  fclose(fpdeb);
#endif

  allResiduals->max = residualWorst[0];
  if (residualCount[0])
    allResiduals->rms = sqrt(residualSum2[0] / residualCount[0]);
  else
    allResiduals->rms = DBL_MAX;
  if (residualCount[0])
    allResiduals->mad = residualSum[0] / residualCount[0];
  else
    allResiduals->mad = DBL_MAX;

  if (flags & AUTOTUNE_RMS) {
    residualWorst[0] = allResiduals->rms;
  } else if (flags & AUTOTUNE_MAV) {
    residualWorst[0] = allResiduals->mad;
  }

  if (maxField[0] > 0) {
    allResiduals->fracRms = allResiduals->rms / maxField[0];
    allResiduals->fracMad = allResiduals->mad / maxField[0];
    allResiduals->fracMax = allResiduals->max / maxField[0];
  } else
    allResiduals->fracRms =
      allResiduals->fracMad =
      allResiduals->fracMax = -DBL_MAX;

  return residualWorst[0] > significance ? residualWorst[0] : 0.0;
}

int evaluateGGEAndOutput(char *outputFile, char *normalFile, char *skewFile, FIELDS_ON_PLANES *fieldsOnPlanes) {
  double B[3], Br, Bphi;
  double x, y, z, r, phi;
  long ns, ig, m, im;
  BGGEXP_DATA bggexpData[2];
  char haveData[2] = {1, 0};
  SDDS_DATASET SDDSout;
  long ix, iy, iz, irow;

  bggexpData[0].haveData = bggexpData[1].haveData = 0;

  readBGGExpData(&bggexpData[0], normalFile, "CnmS", 0);
  if (skewFile) {
    readBGGExpData(&bggexpData[1], skewFile, "CnmC", 1);
    haveData[1] = 1;
  }

  if (SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 1, NULL, NULL, outputFile) != 1 ||
      !SDDS_DefineSimpleColumn(&SDDSout, "x", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "y", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "z", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Bx", "T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "By", "T", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Bz", "T", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSout) ||
      !SDDS_StartPage(&SDDSout, fieldsOnPlanes->Nfft * fieldsOnPlanes->Nx * fieldsOnPlanes->Ny)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return 1;
  }

  irow = 0;
  for (iz = 0; iz < fieldsOnPlanes->Nfft; iz++) {
    z = fieldsOnPlanes->zMin + fieldsOnPlanes->dz * iz;
    for (iy = 0; iy < fieldsOnPlanes->Ny; iy++) {
      y = fieldsOnPlanes->yMin + fieldsOnPlanes->dy * iy;
      for (ix = 0; ix < fieldsOnPlanes->Nx; ix++) {
        x = fieldsOnPlanes->xMin + fieldsOnPlanes->dx * ix;

        /* Compute fields */
        Br = Bphi = B[0] = B[1] = B[2] = 0;
        phi = 0;
        for (ns = 0; ns < 2; ns++) {
          /* ns=0 => normal, ns=1 => skew */
          if (!haveData[ns])
            continue;

          r = sqrt(sqr(x) + sqr(y));
          phi = atan2(y, x);

          for (im = 0; im < bggexpData[ns].nm; im++) {
            double mfact, term, sin_mphi, cos_mphi;
            long ndLimit;
            m = bggexpData[ns].m[im];
            mfact = dfactorial(m);
            sin_mphi = sin(m * phi);
            cos_mphi = cos(m * phi);
            ndLimit = bggexpData[ns].nGradients;
            if (ns == 0) {
              /* normal */
              for (ig = 0; ig < ndLimit; ig++) {
                term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r * sin_mphi;
                term *= bggexpData[ns].Cmn[im][ig][iz];
                Br += term * (2 * ig + m) * sin_mphi;
                Bphi += m * term * cos_mphi;
              }
            } else {
              /* skew */
              if (m == 0) {
                B[2] += bggexpData[ns].dCmn_dz[im][0][iz]; // on-axis Bz from m=ig=0 term
                for (ig = 1; ig < ndLimit; ig++) {
                  term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                  B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r;
                  Br += term * (2 * ig + m) * bggexpData[ns].Cmn[im][ig][iz];
                }
              } else {
                for (ig = 0; ig < ndLimit; ig++) {
                  term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
                  B[2] += term * bggexpData[ns].dCmn_dz[im][ig][iz] * r * cos_mphi;
                  term *= bggexpData[ns].Cmn[im][ig][iz];
                  Br += term * (2 * ig + m) * cos_mphi;
                  Bphi -= m * term * sin_mphi;
                }
              }
            }
          }
        }
        B[0] = Br * cos(phi) - Bphi * sin(phi);
        B[1] = Br * sin(phi) + Bphi * cos(phi);
        if (SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, irow++,
                              "x", x, "y", y, "z", z,
                              "Bx", B[0], "By", B[1], "Bz", B[2],
                              NULL) != 1) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          return (1);
        }
      }
    }
  }
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }

  freeBGGExpData(&bggexpData[0]);
  if (haveData[1])
    freeBGGExpData(&bggexpData[1]);
  return 0;
}
