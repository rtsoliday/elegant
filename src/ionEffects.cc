/************************************************************************* \
 * Copyright (c) 2017 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: ionEffects.c
 * purpose: simulation of ion interaction with the beam
 *
 * Joe Calvey, Michael Borland 2017
 */

#include <complex>
#include "mdb.h"
#include "track.h"
#include "ionEffects.h"
#include "constants.h"
#include "pressureData.h"

#include "poissonBuffer.h"

#define ION_FIELD_GAUSSIAN 0
#define ION_FIELD_BIGAUSSIAN 1
#define ION_FIELD_BILORENTZIAN 2
#define ION_FIELD_TRIGAUSSIAN 3
#define ION_FIELD_TRILORENTZIAN 4
#define ION_FIELD_EGAUSSIAN 5
#define ION_FIELD_GAUSSIANFIT 6
#define ION_FIELD_POISSON 7
#define N_ION_FIELD_METHODS 8
static char *ionFieldMethodOption[N_ION_FIELD_METHODS] = {
  (char *)"gaussian",
  (char *)"bigaussian",
  (char *)"bilorentzian",
  (char *)"trigaussian",
  (char *)"trilorentzian",
  (char *)"egaussian",
  (char *)"gaussianfit",
  (char *)"poisson"};
static long ionFieldMethod = -1;
static long isLorentzian = 0;

#define ION_FIT_RESIDUAL_SUM_ABS_DEV 0
#define ION_FIT_RESIDUAL_RMS_DEV 1
#define ION_FIT_RESIDUAL_MAX_ABS_DEV 2
#define ION_FIT_RESIDUAL_MAX_PLUS_RMS_DEV 3
#define ION_FIT_RESIDUAL_SUM_ABS_PLUS_RMS_DEV 4
#define ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_SUM 5
#define ION_FIT_RESIDUAL_SUM_ABS_PLUS_ABS_DEV_SUM 6
#define ION_FIT_RESIDUAL_RMS_DEV_PLUS_CENTROID 7
#define ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_CHARGE 8
#define ION_FIT_RESIDUAL_MAX_ABS_DEV_PLUS_ABS_DEV_CHARGE 9
#define N_ION_FIT_RESIDUAL_OPTIONS 10
static char *ionFitResidualOption[N_ION_FIT_RESIDUAL_OPTIONS] = {
  (char *)"sum-ad",
  (char *)"rms-dev",
  (char *)"max-ad",
  (char *)"max-ad-plus-rms-dev",
  (char *)"sum-ad-plus-rms-dev",
  (char *)"rms-dev-plus-ad-sum",
  (char *)"sum-ad-plus-ad-sum",
  (char *)"rms-dev-plus-centroid",
  (char *)"rms-dev-plus-ad-charge",
  (char *)"max-ad-plus-ad-charge",
};

static long residualType = -1;
static long ionsInitialized = 0;

static PRESSURE_DATA pressureData;

typedef struct {
  long nSpecies;
  char **ionName;
  double *mass, *chargeState; /* Atomic mass, charge state */
  long *sourceGasIndex;       /* If charge state=1, index in pressureData of the source gas */
  long *sourceIonIndex;       /* If charge state>1, index in this list of the ion that sources this ion */
  double *crossSection;
} ION_PROPERTIES;

static ION_PROPERTIES ionProperties;
void readIonProperties(char *filename);

void addIons(IONEFFECTS *ionEffects, long iSpecies, long nToAdd, double qToAdd, double centroid[4], double sigma[4], long symmetrize);

void addIon_point(IONEFFECTS *ionEffects, long iSpecies, double qToAdd, double x, double y);

void roundGaussianBeamKick(double *coord, double *center, double *sigma, long fromBeam, double kick[2], double charge,
                           double ionMass, double ionCharge);

void makeIonHistograms(IONEFFECTS *ionEffects, long nSpecies, double *bunchSigma, double *ionSigma);
void make2dIonHistogram(IONEFFECTS *ionEffects);
double findIonBinningRange(IONEFFECTS *ionEffects, long iPlane, long nSpecies);
void startSummaryDataOutputPage(IONEFFECTS *ionEffects, long iPass, long nPasses, long nBunches);
void computeIonEffectsElectronBunchParameters(double **part, double *time, int64_t np,
                                              CHARGE *charge, double *tNow, long *npTotal, double *qBunch,
                                              double centroid[4], double sigma[4]);
void setIonEffectsElectronBunchOutput(IONEFFECTS *ionEffects, double tNow, long iPass, long iBunch, double qBunch,
                                      long npTotal, double bunchSigma[4], double bunchCentroid[4]);
void setIonEffectsIonParameterOutput(IONEFFECTS *ionEffects, double tNow, long iPass, long iBunch, long nBunches, double qIon,
                                     double ionSigma[2], double ionCentroid[2]);
void advanceIonPositions(IONEFFECTS *ionEffects, long iPass, double tNow);
void computeIonOverallParameters(IONEFFECTS *ionEffects, double ionCentroid[2], double ionSigma[2], double *qIonReturn,
                                 long *nIonsTotal, double bunchCentroid[4], double bunchSigma[4], long iBunch);
void eliminateIonsOutsideSpan(IONEFFECTS *ionEffects);
void applyElectronBunchKicksToIons(IONEFFECTS *ionEffects, long iPass, double qBunch, double bunchCentroid[4], double bunchSigma[4],
                                   double dpSum[3]);
void generateIons(IONEFFECTS *ionEffects, long iPass, long iBunch, long nBunches,
                  double qBunch, double bunchCentroid[4], double bunchSigma[4]);
void applyIonKicksToElectronBunch(IONEFFECTS *ionEffects, double **part, int64_t np, double Po, long iBunch, long iPass, double qBunch,
                                  double bunchCentroid[4], double bunchSigma[4],
                                  double qIon, long nIonsTotal, double ionCentroid[2], double ionSigma[2],
                                  double dpSum[3]);
void doIonEffectsIonHistogramOutput(IONEFFECTS *ionEffects, long iBunch, long iPass, double tNow, long nBunches);
void flushIonEffectsSummaryOutput(IONEFFECTS *ionEffects);

#if USE_MPI
void shareIonHistograms(IONEFFECTS *ionEffects, short type);
static long simplexComparisonStep = 0, targetReached = 0;
void checkTargetIonFitting(double myResult, long invalid);
#endif
void determineOffsetAndActiveBins(double *histogram, long nBins, long *binOffset, long *activeBins);

static SDDS_DATASET *SDDS_beamOutput = NULL;
static SDDS_DATASET *SDDS_ionDensityOutput = NULL;
static SDDS_DATASET *SDDS_ionHistogramOutput = NULL;
static SDDS_DATASET *SDDS_ion2dHistogramOutput = NULL;
static long ionHistogramOutputInterval, ionHistogramMinOutputBins, ionHistogramMaxBins, ionHistogramMinPerBin;
static double ionHistogramOutput_sStart, ionHistogramOutput_sEnd;

static long iBeamOutput, iIonDensityOutput;
static IONEFFECTS *firstIonEffects = NULL; /* first in the lattice */
#if USE_MPI
static long leftIonCounter = 0;
extern void find_global_min_index(double *min, int *processor_ID, MPI_Comm comm);
#endif

// for fit (e.g., bi-gaussian)
static double *xData = NULL, *yData = NULL, *yFit = NULL, yDataSum, ionChargeData;
static long nData = 0;
static long nFunctions = 2; /* should be 2 or 3 */
static long mFunctions;

short multipleWhateverFit(double bunchSigma[4], double bunchCentroid[4], double paramValueX[9],
                          double paramValueY[9], IONEFFECTS *ionEffects, double ionSigma[2], double ionCentroid[2]);
double multiGaussianFunction(double *param, long *invalid);
double multiLorentzianFunction(double *param, long *invalid);

// for poisson solver
// static double **ionPotential, **xKickPoisson, **yKickPoisson;

// void report();
void report(double res, double *a, long pass, long n_eval, long n_dimen);

#if USE_MPI
void findGlobalMinIndex(double *min, int *processor_ID, MPI_Comm comm) {
  struct {
    double val;
    int rank;
  } in, out;
  in.val = *min;
  MPI_Comm_rank(comm, &(in.rank));
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
  *min = out.val;
  *processor_ID = out.rank;
}
#endif

char speciesNameBuffer[100];
char *makeSpeciesName(const char *prefix, char *suffix) {
  strcpy(speciesNameBuffer, prefix);
  strcat(speciesNameBuffer, suffix);
  return &speciesNameBuffer[0];
}

void closeIonEffectsOutputFiles() {
  if (SDDS_beamOutput) {
    SDDS_Terminate(SDDS_beamOutput);
    SDDS_beamOutput = NULL;
  }
  if (SDDS_ionDensityOutput) {
    SDDS_Terminate(SDDS_ionDensityOutput);
    SDDS_ionDensityOutput = NULL;
  }
  if (SDDS_ionHistogramOutput) {
    SDDS_Terminate(SDDS_ionHistogramOutput);
    SDDS_ionHistogramOutput = NULL;
  }
  if (SDDS_ion2dHistogramOutput) {
    SDDS_Terminate(SDDS_ion2dHistogramOutput);
    SDDS_ion2dHistogramOutput = NULL;
  }
}

void setUpIonEffectsOutputFiles(long nPasses) {
  long iSpecies;

  closeIonEffectsOutputFiles(); /* Shouldn't be needed, but won't hurt */

  if (beam_output) {
    /* Setup the beam output file */
#if USE_MPI
    if (myid == 0) {
#endif
      if (!SDDS_beamOutput) {
        SDDS_beamOutput = (SDDS_DATASET *)tmalloc(sizeof(*SDDS_beamOutput));
        if (!SDDS_InitializeOutputElegant(SDDS_beamOutput, SDDS_BINARY, 1, "electron beam output", NULL, beam_output)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_DefineSimpleColumn(SDDS_beamOutput, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_beamOutput, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Bunch", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "qBunch", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "npBunch", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_beamOutput, "s", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Sx", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Sy", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Sxp", "", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Syp", "", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Cx", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Cy", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Cxp", "", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_beamOutput, "Cyp", "", SDDS_DOUBLE) ||
            !SDDS_SaveLayout(SDDS_beamOutput) || !SDDS_WriteLayout(SDDS_beamOutput)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
#if USE_MPI
    }
#endif
  }

  if (ion_density_output) {
    /* Setup the ion density file */
#if USE_MPI
    if (myid == 0) {
#endif
      if (!SDDS_ionDensityOutput) {
        SDDS_ionDensityOutput = (SDDS_DATASET *)tmalloc(sizeof(*SDDS_ionDensityOutput));
        if (!SDDS_InitializeOutputElegant(SDDS_ionDensityOutput, SDDS_BINARY, 1, "ion density output", NULL, ion_density_output)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_DefineSimpleParameter(SDDS_ionDensityOutput, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "Bunch", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionDensityOutput, "s", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "qIons", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "nMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "nCoreMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "Sx", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "Sy", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "Cx", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "Cy", "m", SDDS_DOUBLE)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (ion_species_output) {
          for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
            if (!SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("qIons_", ionProperties.ionName[iSpecies]),
                                         "C", SDDS_DOUBLE) ||
                !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("nMacroIons_", ionProperties.ionName[iSpecies]),
                                         NULL, SDDS_LONG) ||
                !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("Cx_", ionProperties.ionName[iSpecies]),
                                         "m", SDDS_DOUBLE) ||
                !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("Cy_", ionProperties.ionName[iSpecies]),
                                         "m", SDDS_DOUBLE) ||
                !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("Sx_", ionProperties.ionName[iSpecies]),
                                         "m", SDDS_DOUBLE) ||
                !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput,
                                         makeSpeciesName("Sy_", ionProperties.ionName[iSpecies]),
                                         "m", SDDS_DOUBLE)) {
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
              exitElegant(1);
            }
          }
        }
#if USE_MPI
        if (!SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "nMacroIonsMin", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionDensityOutput, "nMacroIonsMax", NULL, SDDS_LONG)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
#endif
        if (!SDDS_SaveLayout(SDDS_ionDensityOutput) || !SDDS_WriteLayout(SDDS_ionDensityOutput)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
#if USE_MPI
    }
#endif
  }

  if (ion_histogram_output) {
    /* Setup the ion histogram output file */
#if USE_MPI
    if (myid == 0) {
#endif
      if (!SDDS_ionHistogramOutput) {
        SDDS_ionHistogramOutput = (SDDS_DATASET *)tmalloc(sizeof(*SDDS_ionHistogramOutput));
        if (!SDDS_InitializeOutputElegant(SDDS_ionHistogramOutput, SDDS_BINARY, 1,
                                          "ion histogram output", NULL, ion_histogram_output)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "Bunch", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "s", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "Plane", NULL, SDDS_STRING) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "qIons", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "qIonsOutside", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "fractionIonChargeOutside", NULL, SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "binSize", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "binRange", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nBins", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nCoreMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ionHistogramOutput, "Position", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "fitResidual", NULL, SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, isLorentzian ? "a1" : "sigma1", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "centroid1", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "q1", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionHistogramOutput, "Charge", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleColumn(SDDS_ionHistogramOutput, "ChargeFit", "C", SDDS_DOUBLE)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (ionFieldMethod != ION_FIELD_GAUSSIAN &&
            (!SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "fitType", NULL, SDDS_STRING) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "qIonsFromFit", "C", SDDS_DOUBLE) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "qIonsError", "C", SDDS_DOUBLE) ||
#if USE_MPI
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nEvaluationsBest", NULL, SDDS_LONG) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nEvaluationsMin", NULL, SDDS_LONG) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nEvaluationsMax", NULL, SDDS_LONG) ||
#else
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "nEvaluations", NULL, SDDS_LONG) ||
#endif
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, isLorentzian ? "a2" : "sigma2", "m", SDDS_DOUBLE) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "centroid2", "m", SDDS_DOUBLE) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "q2", "C", SDDS_DOUBLE))) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if ((ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionFieldMethod == ION_FIELD_TRILORENTZIAN) &&
            (!SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, isLorentzian ? "a3" : "sigma3", "m", SDDS_DOUBLE) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "centroid3", "m", SDDS_DOUBLE) ||
             !SDDS_DefineSimpleParameter(SDDS_ionHistogramOutput, "q3", "C", SDDS_DOUBLE))) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_SaveLayout(SDDS_ionHistogramOutput) || !SDDS_WriteLayout(SDDS_ionHistogramOutput)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
#if USE_MPI
    }
#endif
  }

  if (ion_2d_histogram_output) {
    /* Setup the ion 2d histogram output file */
#if USE_MPI
    if (myid == 0) {
#endif
      if (!SDDS_ion2dHistogramOutput) {
        SDDS_ion2dHistogramOutput = (SDDS_DATASET *)tmalloc(sizeof(*SDDS_ion2dHistogramOutput));
        if (!SDDS_InitializeOutputElegant(SDDS_ion2dHistogramOutput, SDDS_BINARY, 1,
                                          "ion 2d histogram output", NULL, ion_2d_histogram_output)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "Pass", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "Bunch", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "t", "s", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "s", "m", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "qIons", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "qIonsOutside", "C", SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "fractionIonChargeOutside", NULL, SDDS_DOUBLE) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "nxBins", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "nyBins", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "nMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleParameter(SDDS_ion2dHistogramOutput, "nCoreMacroIons", NULL, SDDS_LONG) ||
            !SDDS_DefineSimpleColumn(SDDS_ion2dHistogramOutput, "x", "m", SDDS_FLOAT) ||
            !SDDS_DefineSimpleColumn(SDDS_ion2dHistogramOutput, "y", "m", SDDS_FLOAT) ||
            !SDDS_DefineSimpleColumn(SDDS_ion2dHistogramOutput, "rho", "Coulomb/m$a2$n", SDDS_FLOAT)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (!SDDS_SaveLayout(SDDS_ion2dHistogramOutput) || !SDDS_WriteLayout(SDDS_ion2dHistogramOutput)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
#if USE_MPI
    }
#endif
  }
}

#  include <fftw3.h>
// FFTW makes 64byte aligned malloc for x86-64
static void **allocateFFTWArray(uint64_t size, int n1, int n2) {
  char **ptr0;
  char *buffer;

  ptr0 = (char **)malloc(sizeof(*ptr0) * n1);
  // buffer = (char *) aligned_alloc(64, n1*n2*size);
  buffer = (char *)fftw_malloc(n1 * n2 * size);
  for (int i = 0; i < n1; i++)
    ptr0[i] = buffer + i * size * n2;
  return ((void **)ptr0);
}

void setupIonEffects(NAMELIST_TEXT *nltext, VARY *control, RUN *run) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&ion_effects, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &ion_effects);

  /* Basic check of input values */
  if (macro_ions <= 0)
    bombElegant("macro_ions must be positive", NULL);
  if (generation_interval <= 0)
    bombElegant("generation_interval must be positive", NULL);
  if (!pressure_profile || !strlen(pressure_profile))
    bombElegant("pressure_profile undefined", NULL);
  if (!ion_properties || !strlen(ion_properties))
    bombElegant("ion_properties undefined", NULL);
  if (beam_output)
    beam_output = compose_filename(beam_output, run->rootname);
  if (ion_density_output)
    ion_density_output = compose_filename(ion_density_output, run->rootname);
  if (ion_histogram_output) {
    ion_histogram_output = compose_filename(ion_histogram_output, run->rootname);
    if (ion_histogram_output_s_start > ion_histogram_output_s_end)
      bombElegantVA((char *)"ion_histogram_s_start (%le) is > ion_histogram_s_end (%le)", ion_histogram_output_s_start,
                    ion_histogram_output_s_end);
    if (ion_histogram_output_interval <= 0)
      bombElegantVA((char *)"ion_histogram_s_interval (%ld) is <=0", ion_histogram_output_interval);
  }
  if (ion_2d_histogram_output) {
    ion_2d_histogram_output = compose_filename(ion_2d_histogram_output, run->rootname);
    if (ion_histogram_output_s_start > ion_histogram_output_s_end)
      bombElegantVA((char *)"ion_histogram_s_start (%le) is > ion_histogram_s_end (%le)", ion_histogram_output_s_start,
                    ion_histogram_output_s_end);
    if (ion_histogram_output_interval <= 0)
      bombElegantVA((char *)"ion_histogram_s_interval (%ld) is <=0", ion_histogram_output_interval);
  }
  ionHistogramOutputInterval = ion_histogram_output_interval;
  ionHistogramOutput_sStart = ion_histogram_output_s_start;
  ionHistogramOutput_sEnd = ion_histogram_output_s_end;
  ionHistogramMinOutputBins = ion_histogram_min_output_bins;
  ionHistogramMaxBins = ion_histogram_max_bins;
  ionHistogramMinPerBin = ion_histogram_min_per_bin;
  if (!field_calculation_method || !strlen(field_calculation_method))
    bombElegant("field_calculation_method undefined", NULL);
  if ((ionFieldMethod = match_string(field_calculation_method, ionFieldMethodOption, N_ION_FIELD_METHODS, EXACT_MATCH)) < 0)
    bombElegantVA((char *)"field_calculation_method=\"%s\" not recognized", field_calculation_method);
  if (ionFieldMethod == ION_FIELD_BIGAUSSIAN || ionFieldMethod == ION_FIELD_BILORENTZIAN)
    nFunctions = 2;
  else if (ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionFieldMethod == ION_FIELD_TRILORENTZIAN)
    nFunctions = 3;
  else if (ionFieldMethod == ION_FIELD_GAUSSIANFIT)
    nFunctions = 1;
  isLorentzian = 0;
  if (ionFieldMethod == ION_FIELD_BILORENTZIAN || ionFieldMethod == ION_FIELD_TRILORENTZIAN)
    isLorentzian = 1;

  if (!fit_residual_type || !strlen(fit_residual_type))
    residualType = ION_FIT_RESIDUAL_MAX_ABS_DEV_PLUS_ABS_DEV_CHARGE;
  else if ((residualType = match_string(fit_residual_type, ionFitResidualOption, N_ION_FIT_RESIDUAL_OPTIONS, EXACT_MATCH)) < 0)
    bombElegantVA((char *)"fit_residual_type=\"%s\" not recognized", fit_residual_type);

  for (int iPlane = 0; iPlane < 2; iPlane++) {
    if (ion_span[iPlane] <= 0)
      bombElegantVA((char *)"ion_span must be positive for both planes---%le given for %s plane\n",
                    ion_span[iPlane], iPlane ? "y" : "x");

    if (ion_bin_divisor[iPlane] <= 0)
      bombElegantVA((char *)"ion_bin_divisor must be positive for both planes---%le given for %s plane\n",
                    ion_bin_divisor[iPlane], iPlane ? "y" : "x");
  }

  readGasPressureData(pressure_profile, &pressureData, pressure_factor, 1);

  readIonProperties(ion_properties);

  setUpIonEffectsOutputFiles(control->n_passes);
}

void readIonProperties(char *filename) {
  /* Assumed file structure:
   * Columns:
   * IonName       --- SDDS_STRING, Name of the ion, e.g., "H2O+", "CO++"
   * Mass          --- SDDS_FLOAT or SDDS_DOUBLE, in AMU
   * ChargeState   --- SDDS_LONG or SDDS_SHORT, ion charge state (positive integer)
   * SourceName    --- SDDS_STRING, Name of the source molecule for this ion, e.g., "H2O", "CO+"
   * CrossSection  --- SDDS_FLOAT or SDDS_DOUBLE, Cross section for producing ion from source, in Mb (megabarns)
   */

  SDDS_DATASET SDDSin;
  long i;
  char **sourceName;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename)) {
    printf("Problem opening ion properties data file %s\n", filename);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (!check_sdds_column(&SDDSin, (char *)"Mass", (char *)"AMU"))
    bombElegantVA((char *)"Column \"Mass\" is missing, not floating-point type, or does not have units of \"AMU\" in %s\n",
                  filename);
  if (!check_sdds_column(&SDDSin, (char *)"CrossSection", (char *)"Mb") &&
      !check_sdds_column(&SDDSin, (char *)"CrossSection", (char *)"MBarns") &&
      !check_sdds_column(&SDDSin, (char *)"CrossSection", (char *)"megabarns"))
    bombElegantVA((char *)"Column \"CrossSection\" is missing, not floating-point type, or does not have units of megabarns (or Mbarns or Mb)",
                  filename);

  if (SDDS_ReadPage(&SDDSin) <= 0)
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  if ((ionProperties.nSpecies = SDDS_RowCount(&SDDSin)) <= 0)
    bombElegantVA((char *)"Ion properties file %s appears to have no rows.\n", filename);

  sourceName = NULL;
  if (!(ionProperties.ionName = (char **)SDDS_GetColumn(&SDDSin, (char *)"IonName")) ||
      !(ionProperties.mass = SDDS_GetColumnInDoubles(&SDDSin, (char *)"Mass")) ||
      !(ionProperties.chargeState = SDDS_GetColumnInDoubles(&SDDSin, (char *)"ChargeState")) ||
      !(ionProperties.crossSection = SDDS_GetColumnInDoubles(&SDDSin, (char *)"CrossSection")) ||
      !(sourceName = (char **)SDDS_GetColumn(&SDDSin, (char *)"SourceName")))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  if (!(ionProperties.sourceGasIndex = (long *)tmalloc(sizeof(*(ionProperties.sourceGasIndex)) * ionProperties.nSpecies)) ||
      !(ionProperties.sourceIonIndex = (long *)tmalloc(sizeof(*(ionProperties.sourceIonIndex)) * ionProperties.nSpecies)))
    bombElegantVA((char *)"Memory allocation failure allocating arrays for %ld ion species.\n", ionProperties.nSpecies);

  /* Figure out the source gas or source ion indices */
  for (i = 0; i < ionProperties.nSpecies; i++) {
    ionProperties.sourceGasIndex[i] = ionProperties.sourceIonIndex[i] = -1;
    if (ionProperties.chargeState[i] <= 0)
      bombElegantVA((char *)"Ion %s has non-positive charge state", ionProperties.ionName[i]);
    if (ionProperties.chargeState[i] == 1) {
      if ((ionProperties.sourceGasIndex[i] = match_string(sourceName[i], pressureData.gasName, pressureData.nGasses, EXACT_MATCH)) < 0) {
        if ((ionProperties.sourceIonIndex[i] = match_string(sourceName[i], ionProperties.ionName, ionProperties.nSpecies, EXACT_MATCH)) < 0) {
          bombElegantVA((char *)"Unable to find match to gas source \"%s\" for species \"%s\"",
                        sourceName[i], ionProperties.ionName[i]);
        }
      }
    } else {
      if ((ionProperties.sourceIonIndex[i] = match_string(sourceName[i], ionProperties.ionName, ionProperties.nSpecies, EXACT_MATCH)) < 0)
        bombElegantVA((char *)"Unable to find match to ion source \"%s\" for species \"%s\"",
                      sourceName[i], ionProperties.ionName[i]);
    }
  }

  printf("Finished reading ion properties file %s\n", filename);
  fflush(stdout);
}

void completeIonEffectsSetup(RUN *run, LINE_LIST *beamline) {
  /* scan through the beamline, find IONEFFECTS elements, set parameters */
  ELEMENT_LIST *eptr, *eptrLast;
  IONEFFECTS *ionEffects;
  short chargeSeen = 0, ionSeen = 0;
  long iPlane;

  eptr = beamline->elem;
  eptrLast = eptr;
  if (eptr->type == T_IONEFFECTS)
    bombElegant("ION_EFFECTS element cannot be the first element in the beamline", NULL);
  while (eptr) {
    if (eptr->type == T_CHARGE)
      chargeSeen = 1;
    if (eptr->type == T_IONEFFECTS) {
      if (!chargeSeen)
        bombElegant("ION_EFFECTS element preceeds the CHARGE element", NULL);
      /* Set the start of the s range for this element */
      ionEffects = (IONEFFECTS *)eptr->p_elem;
      ionEffects->sLocation = eptr->end_pos;
      if (!ionSeen) {
        ionEffects->sStart = 0;
        ionSeen = 1;
      } else
        ionEffects->sStart = (eptrLast->end_pos + eptr->end_pos) / 2;
      /* in case this is the last element in the beamline, set s so the range ends here */
      ionEffects->sEnd = eptr->end_pos;
      if (eptrLast && eptrLast->type == T_IONEFFECTS) {
        /* set the s range for the previous ion effects element */
        ionEffects = (IONEFFECTS *)eptrLast->p_elem;
        ionEffects->sEnd = (eptrLast->end_pos + eptr->end_pos) / 2;
      }
      eptrLast = eptr;
      for (iPlane = 0; iPlane < 2; iPlane++)
        ionEffects->xyIonHistogram[iPlane] = ionEffects->ionHistogram[iPlane] =
          ionEffects->ionHistogramFit[iPlane] = NULL;
    }
    if (!eptr->succ) {
      if (!ionSeen)
        bombElegant("No ION_EFFECTS elements seen", NULL);
      ionEffects = (IONEFFECTS *)eptrLast->p_elem;
      ionEffects->sEnd = eptr->end_pos;
    }
    eptr = eptr->succ;
  }

  eptr = beamline->elem;
  firstIonEffects = NULL;
  while (eptr) {
    if (eptr->type == T_IONEFFECTS) {
      ionEffects = (IONEFFECTS *)eptr->p_elem;
      if (!firstIonEffects)
        firstIonEffects = ionEffects;
      if (verbosity > 10)
        printf("IONEFFECTS element %s#%ld at s=%le m spans s:[%le, %le] m\n",
               eptr->name, eptr->occurence, eptr->end_pos, ionEffects->sStart, ionEffects->sEnd);

      /* Determine the average pressure for each gas */
      ionEffects->pressure = (double *)tmalloc(sizeof(*(ionEffects->pressure)) * pressureData.nGasses);
      if (use_local_pressure == 0) {
        computeAverageGasPressures(ionEffects->sStart, ionEffects->sEnd, ionEffects->pressure, &pressureData);
      } else {
        long iLocation, iGas;
        for (iLocation = 0; iLocation < pressureData.nLocations; iLocation++) {
          if (pressureData.s[iLocation] > ionEffects->sLocation) {
            for (iGas = 0; iGas < pressureData.nGasses; iGas++) {
              ionEffects->pressure[iGas] = pressureData.pressure[iGas][iLocation];
              //(ionEffects->sEnd - ionEffects->sStart);
            }
            break;
          }
        }
      }
      if (verbosity > 20) {
        long i;
        printf("Average pressures over s:[%le, %le] m\n", ionEffects->sStart, ionEffects->sEnd);
        for (i = 0; i < pressureData.nGasses; i++)
          printf("%s:%.2f nT  ", pressureData.gasName[i], ionEffects->pressure[i] * 1e9);
        printf("\n");
        fflush(stdout);
      }

      /* Allocate arrays (should really clean up the arrays first in case this is one in a series of runs) */
      ionEffects->coordinate = (double ***)calloc(ionProperties.nSpecies, sizeof(*(ionEffects->coordinate)));
      ionEffects->nIons = (long *)calloc(ionProperties.nSpecies, sizeof(*(ionEffects->nIons)));
      ionEffects->t = 0;
      if (ionEffects->macroIons <= 0)
        ionEffects->macroIons = macro_ions;
      if (ionEffects->generationInterval <= 0)
        ionEffects->generationInterval = generation_interval;
      for (iPlane = 0; iPlane < 2; iPlane++) {
        if (ionEffects->n2dGridIon[iPlane] <= 0)
          ionEffects->n2dGridIon[iPlane] = ion_poisson_bins[iPlane];
        if (ionEffects->span[iPlane] <= 0)
          ionEffects->span[iPlane] = ion_span[iPlane];
        if (ion_poisson_span[iPlane] <= 0)
          ionEffects->poisson_span[iPlane] = ion_span[iPlane];
        else
          ionEffects->poisson_span[iPlane] = ion_poisson_span[iPlane];
        if (ionEffects->binDivisor[iPlane] <= 0)
          ionEffects->binDivisor[iPlane] = ion_bin_divisor[iPlane];
        ionEffects->rangeMultiplier[iPlane] = ion_range_multiplier[iPlane];
        ionEffects->sigmaLimitMultiplier[iPlane] = ion_sigma_limit_multiplier[iPlane];
      }
      ionEffects->ion2dDensity = NULL;
      if ((ionEffects->n2dGridIon[0] > 0 && ionEffects->n2dGridIon[1] <= 0) || (ionEffects->n2dGridIon[0] <= 0 && ionEffects->n2dGridIon[1] > 0))
        bombElegant("Poisson grid parameters must be defined for both planes, or neither.", NULL);
      if (ionEffects->n2dGridIon[0] > 0 && ionEffects->n2dGridIon[1] > 0) {
        if (ionEffects->n2dGridIon[0] < 10 || ionEffects->n2dGridIon[1] < 10)
          bombElegant("Poisson grid size must be 10 or greater in x and y", NULL);
        printWarningForTracking((char *)"Poisson solver for IONEFFECTS is not fully implemented", (char *)"It is presently being tested and debugged");

        // Ensures 64 byte alignment for vectorization
        // For MKL, "Memory allocation function fftw_malloc returns memory aligned at a 16-byte boundary"
        // So,use manual align malloc calls [C99]
        // TODO: do we really need all 4? can reuse for kicks maybe?
        // TODO:
        ionEffects->ion2dDensity =
          (double **)allocateFFTWArray(sizeof(double), ionEffects->n2dGridIon[0], ionEffects->n2dGridIon[1]);
        ionEffects->ionPotential =
          (double **)allocateFFTWArray(sizeof(double), ionEffects->n2dGridIon[0], ionEffects->n2dGridIon[1]);
        ionEffects->xKickPoisson =
          (double **)allocateFFTWArray(sizeof(double), ionEffects->n2dGridIon[0], ionEffects->n2dGridIon[1]);
        ionEffects->yKickPoisson =
          (double **)allocateFFTWArray(sizeof(double), ionEffects->n2dGridIon[0], ionEffects->n2dGridIon[1]);
      }
    }
    eptr = eptr->succ;
  }

  ionsInitialized = 1;
}

/* converts Torr to 1/m^3 and mBarns to m^2 */
#define unitsFactor (1e-22 / (7.5006e-3 * k_boltzmann_mks * pressureData.temperature))

static double **speciesCentroid = NULL, *speciesCharge = NULL, **speciesSigma = NULL;
static long *speciesCount = NULL;

void trackWithIonEffects(
                         double **part0,         /* part0[i][j] is the jth coordinate (x,x',y,y',t,delta) for the ith particle */
                         long np0,               /* number of particles (on this processor) */
                         IONEFFECTS *ionEffects, /* ion effects element data */
                         double Po,              /* central momentum (beta*gamma) */
                         long iPass,             /* pass number */
                         long nPasses,           /* number of passes */
                         CHARGE *charge          /* beam charge structure */
                         ) {
  int64_t ip;
  long iBunch, nBunches = 0;
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double **part = NULL;    /* particle buffer for working bunch */
  double *time = NULL;     /* array to record arrival time of each particle in working bunch */
  long *ibParticle = NULL; /* array to record which bunch each particle is in */
  int64_t **ipBunch = NULL;   /* array to record particle indices in part0 array for all particles in each bunch */
  int64_t *npBunch = NULL;    /* array to record how many particles are in each bunch */
  int64_t np, max_np = 0;
  long npTotal;
  /* properties of the electron beam */
  double bunchCentroid[4], bunchSigma[4], tNow, qBunch;
  // double sigmatemp[2];
  /* properties of the ion cloud */
  double ionCentroid[2], ionSigma[2], qIon;
  long nIonsTotal;
  double dpSum[3];
#if USE_MPI
  MPI_Status mpiStatus;
#endif

  if (verbosity > 30) {
    printf("Running ION_EFFECTS\n");
    fflush(stdout);
  }

  if (!ionsInitialized) {
    bombElegant("IONEFFECTS element seen, but ion_effects command was not given to initialize ion modeling.", NULL);
  }
  if (ionEffects->disable) {
    if (verbosity > 30) {
      printf("ION_EFFECTS disabled, returning\n");
      fflush(stdout);
    }
    return;
  }
  if (disable_until_pass > iPass) {
    if (verbosity > 30) {
      printf("ION_EFFECTS disabled until pass %ld, returning\n", disable_until_pass);
      fflush(stdout);
    }
    return;
  }

  if (ionEffects->startPass < 0)
    ionEffects->startPass = 0;
  if (iPass == 0)
    ionEffects->xyFitSet[0] = ionEffects->xyFitSet[1] = 0;

  if ((ionEffects->startPass >= 0 && iPass < ionEffects->startPass) ||
      (ionEffects->endPass >= 0 && iPass > ionEffects->endPass) ||
      (ionEffects->passInterval >= 1 && (iPass - ionEffects->startPass) % ionEffects->passInterval != 0)) {
    if (verbosity > 30) {
      printf("ION_EFFECTS out of pass range (pass:%ld, start:%ld, end:%ld, interval:%ld), returning\n",
             iPass, ionEffects->startPass, ionEffects->endPass, ionEffects->passInterval);
      fflush(stdout);
    }
    return;
  }
  if (verbosity > 40) {
    printf("ION_EFFECTS within range (pass:%ld, start:%ld, end:%ld, interval:%ld), returning\n",
           iPass, ionEffects->startPass, ionEffects->endPass, ionEffects->passInterval);
    fflush(stdout);
  }

  if (isSlave || !distributedBeam) {
    /* Determine which bunch each particle is in */
    index_bunch_assignments(part0, np0, charge ? charge->idSlotsPerBunch : 0, Po, &time0, &ibParticle, &ipBunch, &npBunch, &nBunches, -1);
#if USE_MPI
    if (mpiAbort)
      return;
#endif
  }

#if USE_MPI
  /* Share the number of bunches with the master node */
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 1)
    MPI_Send(&nBunches, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
  if (myid == 0)
    MPI_Recv(&nBunches, 1, MPI_LONG, 1, 1, MPI_COMM_WORLD, &mpiStatus);
#endif
  if (nBunches == 0)
    return;
  if (verbosity > 30) {
    printf("Running ION_EFFECTS with %ld bunches\n", nBunches);
    fflush(stdout);
  }

  startSummaryDataOutputPage(ionEffects, iPass, nPasses, nBunches);

  for (iBunch = 0; iBunch < nBunches; iBunch++) {
    np = 0;
    /* Loop over all bunches */
    if (verbosity > 30) {
      printf("Working on bunch %ld of %ld\n", iBunch, nBunches);
      fflush(stdout);
    }

    /* set up data copy/pointers for this bunch */
    if (isSlave || !distributedBeam) {
      if (nBunches == 1) {
        time = time0;
        part = part0;
        np = np0;
      } else {
        if (npBunch)
          np = npBunch[iBunch];
        else
          np = 0;
        if (np && (!ibParticle || !ipBunch || !time0)) {
#if USE_MPI
          mpiAbort = MPI_ABORT_BUNCH_ASSIGNMENT_ERROR;
          return;
#else
          printf("Problem in index_bunch_assignments. Seek professional help.\n");
          exitElegant(1);
#endif
        }

        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)trealloc(time, sizeof(*time) * np);
          max_np = np;
        }
        if (np > 0) {
          for (ip = 0; ip < np; ip++) {
            time[ip] = time0[ipBunch[iBunch][ip]];
            memcpy(part[ip], part0[ipBunch[iBunch][ip]], sizeof(double) * totalPropertiesPerParticle);
          }
        }
      }
    }

#if USE_MPI
    if (myid == 0)
      np = 0;
#endif

    computeIonEffectsElectronBunchParameters(part, time, np, charge, &tNow, &npTotal, &qBunch, bunchCentroid, bunchSigma);

    setIonEffectsElectronBunchOutput(ionEffects, tNow, iPass, iBunch, qBunch, npTotal, bunchSigma, bunchCentroid);

    advanceIonPositions(ionEffects, iPass, tNow);

    eliminateIonsOutsideSpan(ionEffects);

    if (npTotal) {
      generateIons(ionEffects, iPass, iBunch, nBunches, qBunch, bunchCentroid, bunchSigma);

      applyElectronBunchKicksToIons(ionEffects, iPass, qBunch, bunchCentroid, bunchSigma, dpSum);
    }

    computeIonOverallParameters(ionEffects, ionCentroid, ionSigma, &qIon, &nIonsTotal, bunchCentroid, bunchSigma, iBunch);

    setIonEffectsIonParameterOutput(ionEffects, tNow, iPass, iBunch, nBunches, qIon, ionSigma, ionCentroid);

    if (npTotal && nIonsTotal)
      applyIonKicksToElectronBunch(ionEffects, part, np, Po, iBunch, iPass, qBunch, bunchCentroid, bunchSigma,
                                   qIon, nIonsTotal, ionCentroid, ionSigma, dpSum);

    if (isSlave || !distributedBeam) {
      if (nBunches != 1) {
        /*** Copy bunch coordinates back to original array */
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBunch[iBunch][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }

    doIonEffectsIonHistogramOutput(ionEffects, iBunch, iPass, tNow, nBunches);

    // write out coordinates of each ion, not presently used
    /*
      if ((verbosity > 20) && (ionEffects->sLocation > 688) && (ionEffects->sLocation < 692) && (iPass < 10)) {
      double xtemp, ytemp, qtemp;
      int jMacro = 0;
      FILE * fion;
      fion = fopen("ion_coord_all.dat", "a");
      for (int iSpecies=0; iSpecies<ionProperties.nSpecies; iSpecies++) {
      for (int jMacro=0; jMacro < ionEffects->nIons[iSpecies]; jMacro++) {
      xtemp = ionEffects->coordinate[iSpecies][jMacro][0];
      ytemp = ionEffects->coordinate[iSpecies][jMacro][2];
      qtemp = ionEffects->coordinate[iSpecies][jMacro][4];
      //fprintf(fion, "%f  %f  %f  %e  %d \n",  ionEffects->t, xtemp, ytemp, qtemp, iSpecies);
      fprintf(fion, "%d %d  %e  %e  %e  %d \n",  iPass, iBunch, xtemp, ytemp, qtemp, iSpecies);
      }
      }
      fclose(fion);
      }
    */

#if USE_MPI
#  ifdef DEBUG
    printf("Preparing to wait on barrier at end of loop for bucket %ld\n", iBunch);
    fflush(stdout);
#  endif
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef DEBUG
    printf("Done with bunch %ld\n", iBunch);
    fflush(stdout);
#endif
  } /* End of loop over bunches */

  flushIonEffectsSummaryOutput(ionEffects);

  if (time && time != time0)
    free(time);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBunch, npBunch, nBunches);
  if (speciesCentroid)
    free_zarray_2d((void **)speciesCentroid, ionProperties.nSpecies, 2);
  if (speciesSigma)
    free_zarray_2d((void **)speciesSigma, ionProperties.nSpecies, 2);
  speciesCentroid = speciesSigma = NULL;
  if (speciesCharge)
    free(speciesCharge);
  speciesCharge = NULL;
  if (speciesCount)
    free(speciesCount);
  speciesCount = NULL;
}

void addIons(IONEFFECTS *ionEffects, long iSpecies, long nToAdd, double qToAdd,
             double bunchCentroid[4], double bunchSigma[4], long symmetrize) {
  long iNew;

  /* Allocate space for ion coordinates */
  if (ionEffects->coordinate[iSpecies] == NULL)
    ionEffects->coordinate[iSpecies] = (double **)czarray_2d(sizeof(**(ionEffects->coordinate[iSpecies])), nToAdd, COORDINATES_PER_ION);
  else
    ionEffects->coordinate[iSpecies] = (double **)resize_czarray_2d((void **)ionEffects->coordinate[iSpecies],
                                                                    sizeof(**(ionEffects->coordinate[iSpecies])),
                                                                    ionEffects->nIons[iSpecies] + nToAdd, COORDINATES_PER_ION);

  iNew = ionEffects->nIons[iSpecies];
  ionEffects->nIons[iSpecies] += nToAdd;
  for (; iNew < ionEffects->nIons[iSpecies]; iNew++) {
    ionEffects->coordinate[iSpecies][iNew][0] = gauss_rn_lim(bunchCentroid[0], bunchSigma[0], 3, random_4); /* initial x position */
    ionEffects->coordinate[iSpecies][iNew][1] = 0;                                                          /* initial x velocity */
    ionEffects->coordinate[iSpecies][iNew][2] = gauss_rn_lim(bunchCentroid[2], bunchSigma[2], 3, random_4); /* initial y position */
    ionEffects->coordinate[iSpecies][iNew][3] = 0;                                                          /* initial y velocity */
    ionEffects->coordinate[iSpecies][iNew][4] = qToAdd;                                                     /* macroparticle charge */
    if (symmetrize) {
      iNew++;
      ionEffects->coordinate[iSpecies][iNew][0] = bunchCentroid[0] - (ionEffects->coordinate[iSpecies][iNew - 1][0] - bunchCentroid[0]);
      ionEffects->coordinate[iSpecies][iNew][1] = 0;
      ionEffects->coordinate[iSpecies][iNew][2] = bunchCentroid[2] - (ionEffects->coordinate[iSpecies][iNew - 1][2] - bunchCentroid[2]);
      ionEffects->coordinate[iSpecies][iNew][3] = 0;
      ionEffects->coordinate[iSpecies][iNew][4] = qToAdd;
    }

    // ionEffects->qIon[iSpecies][iNew] = qToAdd;
  }
}

void makeIonHistograms(IONEFFECTS *ionEffects, long nSpecies, double *bunchSigma, double *ionSigma) {
  long iSpecies, iBin[2], iPlane;
  long iIon;
  double qTotal = 0;
  long nIons = 0;
  double delta[2], bStart[2];

  for (iSpecies = qTotal = 0; iSpecies < nSpecies; iSpecies++)
    nIons += ionEffects->nIons[iSpecies];

#if USE_MPI
  long nIonsTotal;
  MPI_Allreduce(&nIons, &nIonsTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  nIons = nIonsTotal;
#endif

  for (iPlane = 0; iPlane < 2; iPlane++) {
    qTotal = 0;
    if (verbosity > 200) {
      printf("making ion histogram for plane %c\nbeamSize = %le, binDivisior=%le, rangeMult=%le, ionSigma=%le, maxBins=%ld\n",
             iPlane ? 'y' : 'x',
             bunchSigma[2 * iPlane], ionEffects->binDivisor[iPlane],
             ionEffects->rangeMultiplier[iPlane], ionSigma[iPlane], ionHistogramMaxBins);
      fflush(stdout);
    }
    if (bunchSigma[2 * iPlane] > 0 && ionEffects->binDivisor[iPlane] > 1)
      ionEffects->ionDelta[iPlane] = bunchSigma[2 * iPlane] / ionEffects->binDivisor[iPlane];
    else
      ionEffects->ionDelta[iPlane] = 1e-3;
    if (ionEffects->rangeMultiplier[iPlane] < 0)
      ionEffects->ionRange[iPlane] = 2 * abs(ionSigma[iPlane] * ionEffects->rangeMultiplier[iPlane]);
    else
      ionEffects->ionRange[iPlane] = findIonBinningRange(ionEffects, iPlane, nSpecies);
    ionEffects->ionBins[iPlane] = ionEffects->ionRange[iPlane] / ionEffects->ionDelta[iPlane] + 0.5;
    if (nIons != 0 && ionHistogramMinPerBin > 1 && (nIons / ionEffects->ionBins[iPlane]) < ionHistogramMinPerBin) {
      if ((ionEffects->ionBins[iPlane] = nIons / ionHistogramMinPerBin + 3) < 6)
        ionEffects->ionBins[iPlane] = 6;
      ionEffects->ionDelta[iPlane] = ionEffects->ionRange[iPlane] / (ionEffects->ionBins[iPlane] - 1.0);
    }
    if (ionEffects->ionBins[iPlane] > ionHistogramMaxBins) {
      ionEffects->ionBins[iPlane] = ionHistogramMaxBins;
      ionEffects->ionRange[iPlane] = ionEffects->ionDelta[iPlane] * (ionHistogramMaxBins - 1.0);
    }
    if (ionEffects->ionRange[iPlane] > 2 * ionEffects->span[iPlane]) {
      ionEffects->ionRange[iPlane] = 2 * ionEffects->span[iPlane];
      ionEffects->ionDelta[iPlane] = ionEffects->ionRange[iPlane] / (ionEffects->ionBins[iPlane] - 1.0);
    }

    if (verbosity > 100) {
      printf("making ion histogram for plane %c with %ld bins, bins size of %le, and half-range of %le\n",
             iPlane ? 'y' : 'x', ionEffects->ionBins[iPlane], ionEffects->ionDelta[iPlane], ionEffects->ionRange[iPlane]);
      fflush(stdout);
    }

    /* allocate and set x or y coordinates of histogram (for output to file) */
    if (ionEffects->xyIonHistogram[iPlane])
      free(ionEffects->xyIonHistogram[iPlane]);
    ionEffects->xyIonHistogram[iPlane] = (double *)tmalloc(sizeof(*ionEffects->xyIonHistogram[iPlane]) * ionEffects->ionBins[iPlane]);
    for (int ixy = 0; ixy < ionEffects->ionBins[iPlane]; ixy++)
      ionEffects->xyIonHistogram[iPlane][ixy] = ixy * ionEffects->ionDelta[iPlane] - ionEffects->ionRange[iPlane] / 2;

    if (ionEffects->ionHistogram[iPlane])
      free(ionEffects->ionHistogram[iPlane]);
    ionEffects->ionHistogram[iPlane] = (double *)tmalloc(sizeof(*ionEffects->ionHistogram[iPlane]) * ionEffects->ionBins[iPlane]);
    memset(ionEffects->ionHistogram[iPlane], 0, sizeof(*ionEffects->ionHistogram[iPlane]) * ionEffects->ionBins[iPlane]);

    if (ionEffects->ionHistogramFit[iPlane])
      free(ionEffects->ionHistogramFit[iPlane]);
    ionEffects->ionHistogramFit[iPlane] = (double *)tmalloc(sizeof(*ionEffects->ionHistogramFit[iPlane]) * ionEffects->ionBins[iPlane]);

    ionEffects->ionHistogramMissed[iPlane] = 0;

    // double delta, xyStart;
    delta[iPlane] = ionEffects->ionDelta[iPlane];
    bStart[iPlane] = -ionEffects->ionRange[iPlane] / 2;
  }

  /* histogram ion charge, if it's in range in both planes */
  for (iSpecies = qTotal = 0; iSpecies < nSpecies; iSpecies++) {
    for (iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
      for (iPlane = 0; iPlane < 2; iPlane++)
        iBin[iPlane] = floor((ionEffects->coordinate[iSpecies][iIon][2 * iPlane] - bStart[iPlane]) / delta[iPlane]);
      if (iBin[0] < 0 || iBin[0] > (ionEffects->ionBins[0] - 1) || iBin[1] < 0 || iBin[1] > (ionEffects->ionBins[1] - 1)) {
        ionEffects->ionHistogramMissed[0] += ionEffects->coordinate[iSpecies][iIon][4];
        ionEffects->ionHistogramMissed[1] += ionEffects->coordinate[iSpecies][iIon][4];
      } else {
        for (iPlane = 0; iPlane < 2; iPlane++)
          ionEffects->ionHistogram[iPlane][iBin[iPlane]] += ionEffects->coordinate[iSpecies][iIon][4];
        qTotal += ionEffects->coordinate[iSpecies][iIon][4];
      }
    }
  }
  ionEffects->qTotal = qTotal;

#if USE_MPI
#  if MPI_DEBUG
  printf("local histogram qTotal = %le\n", qTotal);
  fflush(stdout);
#  endif
  shareIonHistograms(ionEffects, 1);
#endif
}

void make2dIonHistogram(IONEFFECTS *ionEffects) {
  long iSpecies, iBin[2], iPlane;
  long iIon, ix, iy;
  double qTotal = 0;
  double delta[2];
#if USE_MPI
  long nIons = 0;
  long nIonsTotal;

  for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++)
    nIons += ionEffects->nIons[iSpecies];

  MPI_Allreduce(&nIons, &nIonsTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  nIons = nIonsTotal;
#endif

  for (iPlane = 0; iPlane < 2; iPlane++) {
    delta[iPlane] = 2 * ionEffects->poisson_span[iPlane] / (ionEffects->n2dGridIon[iPlane] - 1.0);
    ionEffects->ionHistogramMissed[iPlane] = 0;
  }

  memset(ionEffects->ion2dDensity[0], 0, sizeof(double) * ionEffects->n2dGridIon[0] * ionEffects->n2dGridIon[1]);

  /* histogram ion charge, if it's in range in both planes */
  for (iSpecies = qTotal = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
    for (iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
      for (iPlane = 0; iPlane < 2; iPlane++)
        iBin[iPlane] = floor((ionEffects->coordinate[iSpecies][iIon][2 * iPlane] + ionEffects->poisson_span[iPlane]) / delta[iPlane]);
      if (iBin[0] < 0 || iBin[0] > (ionEffects->n2dGridIon[0] - 1) ||
          iBin[1] < 0 || iBin[1] > (ionEffects->n2dGridIon[1] - 1)) {
        ionEffects->ionHistogramMissed[0] += ionEffects->coordinate[iSpecies][iIon][4];
      } else {
        ionEffects->ion2dDensity[iBin[0]][iBin[1]] += ionEffects->coordinate[iSpecies][iIon][4];
        qTotal += ionEffects->coordinate[iSpecies][iIon][4];
      }
    }
  }
  ionEffects->qTotal = qTotal;
  //  int total_len = ionEffects->n2dGridIon[0] * ionEffects->n2dGridIon[1];
  double f = 1.0 / (delta[0] * delta[1]);
  //  double* temp = ionEffects->ion2dDensity[0];
  //  for (ix = 0; ix < total_len; ix++)
  //      temp[ix] *= f;
  for (ix = 0; ix < ionEffects->n2dGridIon[0]; ix++)
    for (iy = 0; iy < ionEffects->n2dGridIon[1]; iy++)
      ionEffects->ion2dDensity[ix][iy] *= f;

#if USE_MPI
#  if MPI_DEBUG
  printf("local histogram qTotal = %le\n", qTotal);
  fflush(stdout);
#  endif
  shareIonHistograms(ionEffects, 2);
#endif
}

double findIonBinningRange(IONEFFECTS *ionEffects, long iPlane, long nSpecies) {
  double *histogram;
  double min, max, hrange, delta;
  long i, quickBins = 0, nIons;
#if USE_MPI
  long nIonsMissed;
#endif

  max = -(min = DBL_MAX);
  nIons = 0;
  for (long iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    nIons += ionEffects->nIons[iSpecies];
    for (long iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
      if (ionEffects->coordinate[iSpecies][iIon][2 * iPlane] < min)
        min = ionEffects->coordinate[iSpecies][iIon][2 * iPlane];
      if (ionEffects->coordinate[iSpecies][iIon][2 * iPlane] > max)
        max = ionEffects->coordinate[iSpecies][iIon][2 * iPlane];
    }
  }
#if USE_MPI
  double gmin, gmax;
  long nIonsGlobal;
  MPI_Allreduce(&min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nIons, &nIonsGlobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  min = gmin;
  max = gmax;
  nIons = nIonsGlobal;
#endif
  if (abs(max) > abs(min))
    hrange = abs(max) + ionEffects->ionDelta[iPlane];
  else
    hrange = abs(min) + ionEffects->ionDelta[iPlane];
  /*  printf("nIons = %ld, min = %le, max = %le, hrange = %le\n", nIons, min, max, hrange); */
  if (ionEffects->rangeMultiplier[iPlane] == 0) {
    /* printf("Using full range [%le, %le] for ion binning\n", -hrange, hrange); fflush(stdout); */
    return 2 * hrange;
  }
  delta = 10 * ionEffects->ionDelta[iPlane];
  quickBins = (2 * hrange) / delta + 0.5;
  if (quickBins < 50 || nIons < 8) {
    /* printf("Using full range [%le, %le] for ion binning\n", -hrange, hrange); fflush(stdout); */
    hrange *= abs(ionEffects->rangeMultiplier[iPlane]);
    if (hrange > ionEffects->span[iPlane])
      hrange = ionEffects->span[iPlane];
    return 2 * hrange;
  }

  /* printf("Using %ld bins, delta = %le for quick ion binning over range [%le, %le]\n",
     quickBins, delta, -hrange, hrange); fflush(stdout); */

  histogram = (double *)calloc(quickBins, sizeof(*histogram));

  /* make charge-weighted histogram */
#if USE_MPI
  nIonsMissed = 0;
#endif
  for (long iSpecies = 0; iSpecies < nSpecies; iSpecies++) {
    for (long iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
      long iBin;
      iBin = floor((ionEffects->coordinate[iSpecies][iIon][2 * iPlane] + hrange) / delta);
      if (iBin >= 0 && iBin < quickBins)
        histogram[iBin] += 1; /* ionEffects->coordinate[iSpecies][iIon][4]; */
      else {
#if USE_MPI
        nIonsMissed += 1;
#endif
      }
    }
  }

#if USE_MPI
  double *histogramGlobal;
  long nIonsMissedGlobal;
  histogramGlobal = (double *)calloc(quickBins, sizeof(*histogramGlobal));
  MPI_Allreduce(histogram, histogramGlobal, quickBins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nIonsMissed, &nIonsMissedGlobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  /* printf("nIonsMissedGlobal = %ld\n", nIonsMissedGlobal); fflush(stdout); */
  for (i = 0; i < quickBins; i++)
    histogram[i] = histogramGlobal[i];
  free(histogramGlobal);
  nIonsMissed = nIonsMissedGlobal;
#endif

  /* find cumulative distribution */
  /*
    for (i=0; i<quickBins; i++)
    printf("histogram[%ld] = %le\n", i, histogram[i]);
  */
  for (i = 1; i < quickBins; i++)
    histogram[i] += histogram[i - 1];
  for (i = 0; i < quickBins; i++)
    histogram[i] /= histogram[quickBins - 1];
  /*
    for (i=0; i<quickBins; i++)
    printf("cdf[%ld] = %le\n", i, histogram[i]);
  */

  /* find 10% and 90% points */
  min = -hrange;
  max = hrange;
  for (i = 0; i < quickBins; i++) {
    if (histogram[i] > 0.1) {
      min = -hrange + (i - 1) * delta;
      break;
    }
  }
  for (; i < quickBins; i++) {
    if (histogram[i] > 0.9) {
      max = -hrange + (i - 1) * delta;
      break;
    }
  }

  if (abs(max) > abs(min))
    hrange = abs(max);
  else
    hrange = abs(min);

  hrange *= abs(ionEffects->rangeMultiplier[iPlane]);
  if (hrange > ionEffects->span[iPlane])
    hrange = ionEffects->span[iPlane];
  /* printf("Using hrange=%le for ion binning\n", hrange); fflush(stdout); */

  free(histogram);
  return 2 * hrange;
}

#if USE_MPI
void shareIonHistograms(IONEFFECTS *ionEffects, short type) {
  double *buffer, partBuffer3[3], sumBuffer3[3];
  long iPlane;

  if (verbosity > 100) {
    printf("Sharing ion histogram(s)\n");
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  partBuffer3[0] = ionEffects->qTotal;
  partBuffer3[1] = ionEffects->ionHistogramMissed[0];
  partBuffer3[2] = ionEffects->ionHistogramMissed[1];
  MPI_Allreduce(partBuffer3, sumBuffer3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  ionEffects->qTotal = sumBuffer3[0];
  ionEffects->ionHistogramMissed[0] = sumBuffer3[1];
  ionEffects->ionHistogramMissed[1] = sumBuffer3[2];
  if (verbosity > 100) {
    printf("qTotal = %le, missed x/y: %le, %le\n",
           ionEffects->qTotal, ionEffects->ionHistogramMissed[0], ionEffects->ionHistogramMissed[1]);
    fflush(stdout);
  }

  if (type == 1) {
    /* share two 1-d histograms */
    if (ionEffects->ionBins[0] <= 0)
      return;

    for (iPlane = 0; iPlane < 2; iPlane++) {
      buffer = (double *)calloc(sizeof(*buffer), ionEffects->ionBins[iPlane]);
      MPI_Allreduce(ionEffects->ionHistogram[iPlane], buffer, ionEffects->ionBins[iPlane],
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      memcpy(ionEffects->ionHistogram[iPlane], buffer, sizeof(*buffer) * ionEffects->ionBins[iPlane]);
      free(buffer);
    }
  } else if (type == 2) {
    /* share one 2-d histogram */
    long nBinsTotal;

    nBinsTotal = ionEffects->n2dGridIon[0] * ionEffects->n2dGridIon[1];
    buffer = (double *)calloc(sizeof(*buffer), nBinsTotal);
    MPI_Allreduce(ionEffects->ion2dDensity[0], buffer, nBinsTotal, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(ionEffects->ion2dDensity[0], buffer, sizeof(*buffer) * nBinsTotal);
    free(buffer);
  } else
    bombElegant("shareIonHistograms called with invalid parameters. Please report this error.", NULL);
  if (verbosity > 100) {
    printf("Done sharing histograms\n");
    fflush(stdout);
  }
}
#endif

void determineOffsetAndActiveBins(double *histogram, long nBins, long *binOffset, long *activeBins) {
  long i, j;
  double min, max, threshold;

  if (nBins < ionHistogramMinOutputBins || nBins < 20) {
    *binOffset = 0;
    *activeBins = nBins;
    return;
  }

  find_min_max(&min, &max, histogram, nBins);
  threshold = max / 1000;

  for (i = 0; i < nBins; i++)
    if (histogram[i] > threshold)
      break;
  if ((*binOffset = i - 1) < 0)
    *binOffset = 0;

  for (j = nBins - 1; j > i; j--)
    if (histogram[j] > threshold)
      break;
  *activeBins = j - *binOffset;
  if (*activeBins <= 0) {
    *binOffset = 0;
    *activeBins = nBins;
    return;
  }

  if (*activeBins < ionHistogramMinOutputBins) {
    j = ionHistogramMinOutputBins - *activeBins;
    *binOffset -= j / 2;
    *activeBins += j;
    if (*binOffset < 0 || (*binOffset + *activeBins) >= nBins) {
      *binOffset = 0;
      *activeBins = ionHistogramMinOutputBins;
      return;
    }
  }
  return;
}

void addIon_point(IONEFFECTS *ionEffects, long iSpecies, double qToAdd, double x, double y) {
  long iNew;

  /* Allocate space for ion coordinates */
  if (ionEffects->coordinate[iSpecies] == NULL)
    ionEffects->coordinate[iSpecies] = (double **)czarray_2d(sizeof(**(ionEffects->coordinate[iSpecies])), 1, COORDINATES_PER_ION);
  else
    ionEffects->coordinate[iSpecies] = (double **)resize_czarray_2d((void **)ionEffects->coordinate[iSpecies],
                                                                    sizeof(**(ionEffects->coordinate[iSpecies])),
                                                                    ionEffects->nIons[iSpecies] + 1, COORDINATES_PER_ION);

  iNew = ionEffects->nIons[iSpecies];
  ionEffects->nIons[iSpecies] += 1;
  ionEffects->coordinate[iSpecies][iNew][0] = x;
  ionEffects->coordinate[iSpecies][iNew][1] = 0; /* initial x velocity */
  ionEffects->coordinate[iSpecies][iNew][2] = y;
  ionEffects->coordinate[iSpecies][iNew][3] = 0;      /* initial y velocity */
  ionEffects->coordinate[iSpecies][iNew][4] = qToAdd; /* macroparticle charge */
}

// From http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package under MIT license
#  include "Faddeeva.hh"

void gaussianBeamKick(
                      double *coord,   /* x, xp, y, yp, s, delta of kicked particle */
                      double *center,  /* center of kicking gaussian */
                      double *sigma,   /* sigma of kicking gaussian */
                      long fromBeam,   // If nonzero, center and sigma arrays have (x, x', y, y') data. Otherwise, just (x, y)
                      double kick[2],  /* for return of velocity change (m/s) */
                      double charge,   /* total charge of kicking distribution in Coulomb */
                      double ionMass,  /* mass of kicked particle in kg */
                      double ionCharge /* charge of kicked particle in units of electron charge */
                      ) {
  // calculate beam kick on ion, assuming Gaussian beam
  double sx, sy, x, y, sd, Fx, Fy, C1, C2, C3, ay;
  std::complex<double> Fc, w1, w2, erf1, erf2;
  long swapXY;

  kick[0] = 0;
  kick[1] = 0;
  //  return;

  sx = sigma[0];
  sy = sigma[fromBeam ? 2 : 1];

  if (abs(sx - sy) / sx < 1e-9)
    return roundGaussianBeamKick(coord, center, sigma, fromBeam, kick, charge, ionMass, ionCharge);

  x = coord[0] - center[0];
  y = coord[2] - center[fromBeam ? 2 : 1];

  C1 = c_mks * charge * re_mks * me_mks * ionCharge / e_mks;

  swapXY = 0;
  if (sx < sy) {
    double tmp;
    swapXY = 1;
    SWAP_DOUBLE(sx, sy);
    tmp = x;
    x = y;
    y = -tmp;
  }

  ay = abs(y);
  sd = sqrt(2.0 * (sqr(sx) - sqr(sy)));
  w1 = std::complex<double>(x / sd, ay / sd);
  w2 = std::complex<double>(x / sd * sy / sx, ay / sd * sx / sy);

  C2 = sqrt(2 * PI / (sqr(sx) - sqr(sy)));
  C3 = exp(-sqr(x) / (2 * sqr(sx)) - sqr(y) / (2 * sqr(sy)));

  // TODO: test relaxing relative error value (see library for details)
  erf1 = Faddeeva::w(w1);
  erf2 = Faddeeva::w(w2);

  Fc = C1 * C2 * (erf1 - C3 * erf2);
  Fx = Fc.imag();
  if (y > 0)
    Fy = Fc.real();
  else
    Fy = -Fc.real();

  if (swapXY) {
    double tmp;
    tmp = Fx;
    Fx = -Fy;
    Fy = tmp;
  }

  kick[0] = -Fx * (1.0 / ionMass);
  kick[1] = -Fy * (1.0 / ionMass);
}

void roundGaussianBeamKick(
                           double *coord,
                           double *center,
                           double *sigma,
                           long fromBeam, // If nonzero, center and sigma arrays have (x, x', y, y') data. Otherwise, just (x, y)
                           double kick[2],
                           double charge,
                           double ionMass,
                           double ionCharge) {
  // calculate beam kick on ion, assuming round Gaussian beam
  double sx, sy, x, y, sig, r, C1, dp, theta;

  kick[0] = 0;
  kick[1] = 0;
  //  return;

  sx = sigma[0];
  sy = sigma[fromBeam ? 2 : 1];
  sig = (sx + sy) / 2;

  x = coord[0] - center[0];
  y = coord[2] - center[fromBeam ? 2 : 1];
  r = sqrt(sqr(x) + sqr(y));

  C1 = 2 * c_mks * charge * re_mks * me_mks * ionCharge / e_mks;

  dp = C1 / r * (1 - exp(-sqr(r) / (2 * sqr(sig))));

  theta = atan2(y, x);
  kick[0] = -dp * cos(theta) / ionMass;
  kick[1] = -dp * sin(theta) / ionMass;
}

short multipleWhateverFit(double bunchSigma[4], double bunchCentroid[4], double *paramValueX, double *paramValueY,
                          IONEFFECTS *ionEffects, double ionSigma[2], double ionCentroid[2]) {
  double result = 0;
  double paramValue[9], paramDelta[9], paramDeltaSave[9], lowerLimit[9], upperLimit[9];
  long nSignificant;
  int32_t nEvalMax, nPassMax;
  unsigned long simplexFlags = SIMPLEX_NO_1D_SCANS;
  double peakVal, minVal, xMin, xMax, fitTolerance, fitTarget;
  long fitReturn, dummy, nEvaluations = 0;
  int nTries;
#if USE_MPI
  double bestResult, lastBestResult;
  int min_location;
#else
  double lastResult = DBL_MAX;
#endif
  long plane, pFunctions;

  fitTolerance = distribution_fit_tolerance;
  ionChargeData = ionEffects->qTotal;

  for (int i = 0; i < 3 * nFunctions; i++)
    paramValueX[i] = paramValueY[i] = 0;

  for (plane = 0; plane < 2; plane++) {
    fitReturn = 0;
#if MPI_DEBUG
    printf("Performing fit for %c plane\n", plane ? 'y' : 'x');
    fflush(stdout);
#endif

    for (int i = 0; i < 9; i++)
      paramValue[i] = paramDelta[i] = lowerLimit[i] = upperLimit[i] = 0;

    nData = ionEffects->ionBins[plane];
    nSignificant = 0;
    for (int i = 0; i < nData; i++)
      if (ionEffects->ionHistogram[plane][i] > 0)
        nSignificant++;
    if ((nSignificant += 2) > nData)
      nSignificant = nData;

    if (verbosity > 100) {
      printf("multipleWhateverFit, nData = %ld, nSignificant = %ld, plane=%c\n", nData, nSignificant, plane ? 'y' : 'x');
      fflush(stdout);
    }

    pFunctions = 2;
    for (mFunctions = 1; mFunctions <= nFunctions; mFunctions++) {
      if ((mFunctions == 1) && (nFunctions > 1))
        continue;
      if (verbosity > 100) {
        printf("multipleWhateverFit, mFunctions=%ld\n", mFunctions);
        fflush(stdout);
      }
      pFunctions = mFunctions;
      xData = ionEffects->xyIonHistogram[plane];
      yData = ionEffects->ionHistogram[plane];
      yFit = ionEffects->ionHistogramFit[plane];
      yDataSum = 0;
      for (int i = 0; i < nData; i++)
        yDataSum += yData[i];
      result = find_min_max(&minVal, &peakVal, yData, nData);
      find_min_max(&xMin, &xMax, xData, nData);

      // subtract baseline (minimum point) before fitting
      if (ion_fit_subtract_baseline) {
        for (int i = 0; i < nData; i++)
          yData[i] -= minVal;
      }

      /* smaller sigma is close to the beam size, larger is close to ion sigma */

      if (mFunctions == 1) { // single gaussian fit
#if USE_MPI
        if (myid % 2 == 0 && ionEffects->xyFitSet[plane] & 0x01) {
          memcpy(paramValue, ionEffects->xyFitParameter2[plane], 6 * sizeof(double));
        } else {
          paramValue[0] = ionSigma[plane];
          paramValue[1] = ionCentroid[plane];
          paramValue[2] = peakVal;
        }
#else
        if (ionEffects->xyFitSet[plane] & 0x01) {
          memcpy(paramValue, ionEffects->xyFitParameter2[plane], 6 * sizeof(double));
        } else {
          paramValue[0] = ionSigma[plane];
          paramValue[1] = ionCentroid[plane];
          paramValue[2] = peakVal;
        }
#endif
        paramDelta[0] = paramValue[0] / 2;
        paramDelta[1] = abs(bunchCentroid[2 * plane]) / 2;
        paramDelta[2] = peakVal / 4;
        lowerLimit[0] = paramValue[0] / 100;
        if (ionEffects->sigmaLimitMultiplier[plane] > 0 &&
            lowerLimit[0] < (ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane]))
          lowerLimit[0] = ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane];
        lowerLimit[1] = xMin / 10;
        lowerLimit[2] = peakVal / 20;
        // upperLimit[0] = paramValue[0]*10;
        upperLimit[0] = 20 * bunchSigma[2 * plane];
        if (upperLimit[0] < lowerLimit[0])
          upperLimit[0] = 2 * lowerLimit[0];
        upperLimit[1] = xMax / 10;
        upperLimit[2] = 2 * peakVal;

      } else if (mFunctions == 2) {
#if USE_MPI
        if (myid % 2 == 0 && ionEffects->xyFitSet[plane] & 0x01) {
          memcpy(paramValue, ionEffects->xyFitParameter2[plane], 6 * sizeof(double));
        } else {
          paramValue[0] = bunchSigma[2 * plane];
          paramValue[1] = bunchCentroid[2 * plane];
          paramValue[2] = peakVal / 2;
          paramValue[3] = ionSigma[plane];
          paramValue[4] = ionCentroid[plane];
          paramValue[5] = paramValue[2] / 3;
        }
#else
        if (ionEffects->xyFitSet[plane] & 0x01) {
          memcpy(paramValue, ionEffects->xyFitParameter2[plane], 6 * sizeof(double));
        } else {
          paramValue[0] = bunchSigma[2 * plane];
          paramValue[1] = bunchCentroid[2 * plane];
          paramValue[2] = peakVal / 2;
          paramValue[3] = ionSigma[plane];
          paramValue[4] = ionCentroid[plane];
          paramValue[5] = paramValue[2] / 3;
        }
#endif
        paramDelta[0] = paramValue[0] / 2;
        paramDelta[1] = abs(bunchCentroid[2 * plane]) / 2;
        paramDelta[2] = peakVal / 4;
        lowerLimit[0] = paramValue[0] / 100;
        if (ionEffects->sigmaLimitMultiplier[plane] > 0 &&
            lowerLimit[0] < (ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane]))
          lowerLimit[0] = ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane];
        lowerLimit[1] = xMin / 10;
        lowerLimit[2] = peakVal / 20;
        // upperLimit[0] = paramValue[0]*10;
        // upperLimit[0] = 20*bunchSigma[2*plane];
        upperLimit[0] = ionEffects->ionDelta[plane] * nData;
        if (upperLimit[0] < lowerLimit[0])
          upperLimit[0] = 2 * lowerLimit[0];
        upperLimit[1] = xMax / 10;
        upperLimit[2] = peakVal;

        paramDelta[3] = paramValue[3] / 2;
        paramDelta[4] = abs(ionCentroid[plane]) / 2;
        paramDelta[5] = peakVal / 4;
        lowerLimit[3] = paramValue[3] / 100;
        if (ionEffects->sigmaLimitMultiplier[plane] > 0 &&
            lowerLimit[3] < (ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane]))
          lowerLimit[3] = ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane];
        lowerLimit[4] = xMin / 10;
        lowerLimit[5] = paramDelta[5] / 5;
        // upperLimit[3] = paramValue[3]*10;
        // upperLimit[3] = 20*bunchSigma[2*plane];
        upperLimit[3] = ionEffects->ionDelta[plane] * nData;
        if (upperLimit[3] < lowerLimit[3])
          upperLimit[3] = 2 * lowerLimit[3];
        upperLimit[4] = xMax / 10;
        upperLimit[5] = peakVal;
      } else if (mFunctions == 3) {
        if (ionEffects->xyFitSet[plane] & 0x02) {
#if USE_MPI
          /* In this case, the odd processors will use the held-over values from 2-function fit */
          if (myid % 2 == 0) {
            memcpy(paramValue, ionEffects->xyFitParameter3[plane], 9 * sizeof(double));
            if (paramValue[6] <= 0)
              paramValue[6] = 10 * ionSigma[plane];
          } else {
            paramValue[6] = 10 * ionSigma[plane];
            paramValue[7] = 0;
            paramValue[8] = 0;
          }
#else
          memcpy(paramValue, ionEffects->xyFitParameter3[plane], 9 * sizeof(double));
#endif
        } else {
          /* paramValue[0-5] are held over from 2-function fit */
          paramValue[6] = 10 * ionSigma[plane];
          paramValue[7] = 0;
          paramValue[8] = 0;
        }
        paramDelta[6] = paramValue[6] / 2;
        paramDelta[7] = abs(ionCentroid[plane]) / 2;
        paramDelta[8] = peakVal / 20;
        lowerLimit[6] = paramValue[6] / 100;
        if (ionEffects->sigmaLimitMultiplier[plane] > 0 &&
            lowerLimit[6] < (ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane]))
          lowerLimit[6] = ionEffects->sigmaLimitMultiplier[plane] * ionEffects->ionDelta[plane];
        lowerLimit[7] = xMin / 10;
        lowerLimit[8] = 0;
        upperLimit[6] = paramValue[6] * 10;
        if (upperLimit[6] < lowerLimit[6])
          upperLimit[6] = 2 * lowerLimit[6];
        upperLimit[7] = xMax / 10;
        upperLimit[8] = peakVal;
      }

      for (int i = 0; i < 3 * mFunctions; i++) {
        if (lowerLimit[i] > paramValue[i]) {
          paramValue[i] = 1.1 * lowerLimit[i];
        }
        /*
          if (paramValue[i]<0)
          lowerLimit[i] = 10*paramValue[i];
          else
          lowerLimit[i] = paramValue[i]/10;
          }
        */
        if (upperLimit[i] < paramValue[i]) {
          paramValue[i] = 0.9 * upperLimit[i];
        }
        /*
          if (paramValue[i]<0)
          upperLimit[i] = paramValue[i]/10;
          else
          upperLimit[i] = paramValue[i]*10;
        */
        //}
      }

#if USE_MPI
      /* Randomize step sizes and starting points */
      for (int i = 0; i < 3 * mFunctions; i++) {
        if (myid != 0 && myid != 1) {
          paramValue[i] *= (1 + (random_2(0) - 0.5) / 5);
          if (paramValue[i] < lowerLimit[i])
            paramValue[i] = lowerLimit[i];
          if (paramValue[i] > upperLimit[i])
            paramValue[i] = upperLimit[i];
          paramDelta[i] *= random_2(0) * 9.9 + 0.1;
        }
      }
      lastBestResult = DBL_MAX;
#endif
      nTries = distribution_fit_restarts;
      nEvalMax = distribution_fit_evaluations;
      nPassMax = distribution_fit_passes;
#if !USE_MPI
      if (nTries < 10)
        nTries = 10;
      lastResult = DBL_MAX;
#endif
      fitTarget = distribution_fit_target;
      if (mFunctions == 2 && nFunctions == 3) {
        // For this stage, just go for a rough 2-function fit
        nTries = 1;
      }
      while (nTries--) {
        long nVariables;
        nVariables = 3 * mFunctions;
        if (nSignificant < 3) {
          nVariables = 3;
          paramValue[3] = paramValue[6] = -1; // size parameter: value doesn't matter, but can't be zero
          paramValue[5] = paramValue[8] = 0;  // height parameter: must be zero
        }
        memcpy(paramDeltaSave, paramDelta, sizeof(*paramDelta) * 9);
#if USE_MPI
        targetReached = 0;
        simplexComparisonStep = 0;
#endif
        fitReturn += simplexMin(&result, paramValue, paramDelta, lowerLimit, upperLimit,
                                NULL, nVariables, fitTarget, fitTolerance,
                                ionEffects->ionFieldMethod == ION_FIELD_BIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_GAUSSIANFIT
                                ? multiGaussianFunction
                                : multiLorentzianFunction,
                                (verbosity > 200 ? report : NULL), nEvalMax, nPassMax, 12, 3, 1.0, simplexFlags);
        memcpy(paramDelta, paramDeltaSave, sizeof(*paramDelta) * 9);
        if (fitReturn >= 0)
          nEvaluations += fitReturn;
        else {
          for (int i = 0; i < 9; i++)
            paramValue[i] = 0;
          for (int i = 0; i < nData; i++)
            yFit[i] = 0;
        }
        if (verbosity > 100) {
          printf("Exited simplexMin, return=%ld, result=%le\nparam: ", fitReturn, result);
          for (int i = 0; i < 3 * mFunctions; i++)
            printf("[%d]=%le, ", i, paramValue[i]);
          printf("\n");
          fflush(stdout);
        }
#if USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&result, &bestResult, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (verbosity > 100) {
          printf("bestResult=%le, result=%le\n", bestResult, result);
          fflush(stdout);
        }
        if (bestResult < distribution_fit_target || fabs(lastBestResult - bestResult) < fitTolerance) {
          if (verbosity > 100) {
            printf("Best result is good enough or hasn't changed, breaking from while loop\n");
            for (int i = 0; i < 3 * mFunctions; i++)
              printf("[%d]=%le, ", i, paramValue[i]);
            printf("\n");
            fflush(stdout);
          }
          break;
        }
        lastBestResult = bestResult;
#else
        if (result < distribution_fit_target || (lastResult - result) < fitTolerance) {
          if (verbosity > 100) {
            printf("Best result is good enough or hasn't changed, breaking from while loop\n");
            for (int i = 0; i < 3 * mFunctions; i++)
              printf("[%d]=%le, ", i, paramValue[i]);
            printf("\n");
            fflush(stdout);
          }
          break;
        }
        lastResult = result;
#endif

#if USE_MPI
        if (nTries != 0) {
          MPI_Barrier(MPI_COMM_WORLD);
          min_location = 1;
          findGlobalMinIndex(&result, &min_location, MPI_COMM_WORLD);
          if (verbosity > 100) {
            printf("distributing and randomizing bestResult from processor %d\n", min_location);
            fflush(stdout);
          }
          MPI_Bcast(paramValue, 3 * mFunctions, MPI_DOUBLE, min_location, MPI_COMM_WORLD);
          if (myid != 0) {
            for (int i = 0; i < 3 * mFunctions; i++) {
              paramValue[i] *= (1 + (random_2(0) - 0.5) / 20);
              if (paramValue[i] < lowerLimit[i])
                paramValue[i] = lowerLimit[i];
              if (paramValue[i] > upperLimit[i])
                paramValue[i] = upperLimit[i];
            }
          }
        }
#endif
      }

      if (verbosity > 100) {
        printf("Exited nTries loop (nTries=%d)\n", nTries);
        for (int i = 0; i < 3 * mFunctions; i++)
          printf("[%d]=%le, ", i, paramValue[i]);
        printf("\n");
        fflush(stdout);
      }

#if USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
      min_location = 1;
      findGlobalMinIndex(&result, &min_location, MPI_COMM_WORLD);
      if (verbosity > 100) {
        printf("distributing best result from processor %d, mFunctions=%ld\n", min_location, mFunctions);
        fflush(stdout);
      }
      MPI_Bcast(&paramValue[0], 9, MPI_DOUBLE, min_location, MPI_COMM_WORLD);
      if (verbosity > 100) {
        printf("1 (2) MPI_Bcast succeeded\n");
        fflush(stdout);
      }
      MPI_Bcast(&fitReturn, 1, MPI_LONG, min_location, MPI_COMM_WORLD);
      if (verbosity > 100) {
        printf("2 (2) MPI_Bcast succeeded: fitReturn = %ld\n", fitReturn);
        fflush(stdout);
      }
#endif

      if (fitReturn > 0) {
        ionEffects->xyFitSet[plane] |= mFunctions == 2 ? 0x01 : 0x02;
        for (int i = 0; i < 3 * mFunctions; i++) {
          if (mFunctions < 3)
            ionEffects->xyFitParameter2[plane][i] = paramValue[i];
          else
            ionEffects->xyFitParameter3[plane][i] = paramValue[i];
        }
      }

      if (result < distribution_fit_target) {
        if (verbosity > 100) {
          printf("Result %le better than target %le, breaking from mFunctions loop with mFunctions=%ld\n",
                 result, distribution_fit_target, mFunctions);
          fflush(stdout);
        }
        break;
      }
      if (nSignificant < 6) {
        // Don't run loop with mFunctions=3 when we don't have at least 6 valid points
        if (verbosity > 100) {
          printf("Terminating with mFunctions=%ld due to too few (%ld) significant points\n",
                 mFunctions, nSignificant);
          fflush(stdout);
        }
        break;
      }
    }

    mFunctions = pFunctions; // Last values used, regardless of how exit from loop occurred

    if (mFunctions == 2 && nFunctions == 3) {
      if (verbosity > 100) {
        printf("Setting up ionEffects->xyFitParameter3 for plane=%ld with mFunctions=2 data\n", plane);
        fflush(stdout);
      }
      ionEffects->xyFitSet[plane] |= 0x02;
      for (int i = 0; i < 3 * mFunctions; i++)
        ionEffects->xyFitParameter3[plane][i] = paramValue[i];
      for (int i = 3 * mFunctions; i < 3 * nFunctions; i++) {
        paramValue[i] = 0;
        ionEffects->xyFitParameter3[plane][i] = 0;
      }
    }

    if (verbosity > 100) {
      printf("Exited plane = %c optimization loop, result=%le, fitReturn=%ld, nEvaluations=%ld\n",
             plane ? 'y' : 'x', result, fitReturn, nEvaluations);
      for (int i = 0; i < 3 * nFunctions; i++)
        printf("[%d]=%le, ", i, paramValue[i]);
      printf("\n");
      fflush(stdout);
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&nEvaluations, &ionEffects->nEvaluationsMin[plane], 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&nEvaluations, &ionEffects->nEvaluationsMax[plane], 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    // min_location = -1;
    // findGlobalMinIndex(&result, &min_location, MPI_COMM_WORLD);
    // MPI_Bcast(&nEvaluations, 1, MPI_LONG, min_location, MPI_COMM_WORLD);
    ionEffects->nEvaluationsBest[plane] = nEvaluations;
#else
    ionEffects->nEvaluations[plane] = nEvaluations;
#endif
    if (ionEffects->ionFieldMethod == ION_FIELD_BIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_GAUSSIANFIT)
      ionEffects->xyFitResidual[plane] = multiGaussianFunction(paramValue, &dummy);
    else
      ionEffects->xyFitResidual[plane] = multiLorentzianFunction(paramValue, &dummy);

#if USE_MPI
    if (verbosity > 100) {
      printf("residual is %le\n", ionEffects->xyFitResidual[plane]);
      fflush(stdout);
    }
#endif

    if (fitReturn > 0) {
      for (int i = 0; i < nData; i++)
        ionEffects->ionHistogramFit[plane][i] = yFit[i];
    } else {
      ionEffects->xyFitSet[plane] = 0;
      for (int i = 0; i < nData; i++)
        ionEffects->ionHistogramFit[plane][i] = -1;
    }

    for (int i = 0; i < 3 * nFunctions; i++) {
      // Use of nFunctions is fine here since the arrays are sized to 9 and we initialized to zero.
      if (plane == 0)
        paramValueX[i] = paramValue[i];
      else
        paramValueY[i] = paramValue[i];
    }
  }
  mFunctions = nFunctions; // ok for both planes since we filled parameters with zeros to start.

  return ionEffects->ionFieldMethod;
}

double multiGaussianFunction(double *param, long *invalid) {
  double sum = 0, sum2 = 0, tmp = 0, result, max = 0, yFitSum = 0;
  double wSumData = 0, wSumFit = 0, peak = 0;
  double charge, dx;
  double centroid[3], sigma[3], height[3];

  *invalid = 0;

  for (int j = 0; j < mFunctions; j++) {
    // param[3*j+0] = sig[j]
    // param[3*j+1] = cen[j]
    // param[3*j+2] = h[j]
    sigma[j] = param[3 * j];
    centroid[j] = param[3 * j + 1];
    height[j] = param[3 * j + 2];
    if (height[j] > 0 && sigma[j] <= 0) {
      *invalid = 1;
      printf("invalid values in multiGaussianFunction: j=%d, sigma=%le, c=%le, h=%le\n",
             j, sigma[j], centroid[j], height[j]);
      fflush(stdout);
      result = DBL_MAX;
#if USE_MPI
      checkTargetIonFitting(result, *invalid);
#endif
      return result;
    }
  }

  dx = xData[1] - xData[0];

  for (int i = 0; i < nData; i++) {
    double z;
    if (yData[i] > peak)
      peak = yData[i];
    wSumData += xData[i] * yData[i];
    yFit[i] = 0;
    for (int j = 0; j < mFunctions; j++) {
      z = (xData[i] - centroid[j]) / sigma[j];
      if (z < 6 && z > -6 && height[j])
        yFit[i] += height[j] * exp(-z * z / 2);
    }
    tmp = abs(yFit[i] - yData[i]);
    sum += tmp;
    sum2 += sqr(tmp);
    yFitSum += yFit[i];
    wSumFit += xData[i] * yFit[i];
    if (tmp > max)
      max = tmp;
  }

  switch (residualType) {
  case ION_FIT_RESIDUAL_RMS_DEV:
    result = sqrt(sum2) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_MAX_ABS_DEV:
    result = max / yDataSum;
    break;
  case ION_FIT_RESIDUAL_MAX_PLUS_RMS_DEV:
    result = sqrt(sum2) / yDataSum + max / yDataSum;
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_PLUS_RMS_DEV:
    result = sqrt(sum2) / yDataSum + sum / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_SUM:
    result = sqrt(sum2) / yDataSum + abs(yDataSum - yFitSum) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_PLUS_ABS_DEV_SUM:
    result = sum / yDataSum + abs(yDataSum - yFitSum) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_CENTROID:
    if (yFitSum)
      result = sqrt(sum2) / yDataSum + abs(wSumFit / yFitSum - wSumData / yDataSum);
    else
      result = sqrt(sum2) / yDataSum + abs(wSumData / yDataSum);
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_DEV:
    result = sum / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_CHARGE:
    charge = 0;
    for (int j = 0; j < mFunctions; j++)
      charge += param[3 * j + 2] * sqrt(PIx2) * param[3 * j + 0];
    result = sqrt(sum2) / yDataSum + abs(charge / dx - ionChargeData) / ionChargeData;
    break;
  case ION_FIT_RESIDUAL_MAX_ABS_DEV_PLUS_ABS_DEV_CHARGE:
    charge = 0;
    for (int j = 0; j < mFunctions; j++)
      charge += param[3 * j + 2] * sqrt(PIx2) * param[3 * j + 0];
    result = max / peak + abs(charge / dx - ionChargeData) / ionChargeData;
    break;
  default:
    result = 0;
    bombElegant("Invalid residual code in multiGaussianFunction---seek professional help", NULL);
    break;
  }

  if (result < 0 || isnan(result) || isinf(result)) {
    *invalid = 1;
    result = DBL_MAX;
  }

#if USE_MPI
  checkTargetIonFitting(result, *invalid);
#endif

  return result;
}

double multiLorentzianFunction(double *param, long *invalid) {
  double sum = 0, sum2 = 0, tmp = 0, result, max = 0, yFitSum = 0;
  double wSumData = 0, wSumFit = 0, peak = 0;
  double charge, dx;
  double a[3], centroid[3], height[3];

  *invalid = 0;

  dx = xData[1] - xData[0];

  /* The parameters are different from the standard Lorentzian.
   * Instead of A/(pi*a*(1 + (x/a)^2) we use height/(1 + (x/a)^2)
   */
  for (int j = 0; j < mFunctions; j++) {
    // param[3*j+0] = a[j]
    // param[3*j+1] = cen[j]
    // param[3*j+2] = height[j]
    a[j] = param[3 * j];
    centroid[j] = param[3 * j + 1];
    height[j] = param[3 * j + 2];
    if (height[j] > 0 && a[j] <= 0) {
      *invalid = 1;
      printf("invalid values in multiLorentzianFunction: j=%d, a=%le, c=%le, h=%le\n",
             j, a[j], centroid[j], height[j]);
      fflush(stdout);
      result = DBL_MAX;
#if USE_MPI
      checkTargetIonFitting(result, *invalid);
#endif
      return result;
    }
  }

  for (int i = 0; i < nData; i++) {
    double z;
    if (yData[i] > peak)
      peak = yData[i];
    wSumData += xData[i] * yData[i];
    yFit[i] = 0;
    for (int j = 0; j < mFunctions; j++) {
      z = (xData[i] - centroid[j]) / a[j];
      yFit[i] += height[j] / (1 + sqr(z));
    }
    tmp = abs(yFit[i] - yData[i]);
    sum += tmp;
    sum2 += sqr(tmp);
    yFitSum += yFit[i];
    wSumFit += xData[i] * yFit[i];
    if (tmp > max)
      max = tmp;
  }

  switch (residualType) {
  case ION_FIT_RESIDUAL_RMS_DEV:
    result = sqrt(sum2) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_MAX_ABS_DEV:
    result = max / yDataSum;
    break;
  case ION_FIT_RESIDUAL_MAX_PLUS_RMS_DEV:
    result = sqrt(sum2) / yDataSum + max / yDataSum;
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_PLUS_RMS_DEV:
    result = sqrt(sum2) / yDataSum + sum / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_SUM:
    result = sqrt(sum2) / yDataSum + abs(yDataSum - yFitSum) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_PLUS_ABS_DEV_SUM:
    result = sum / yDataSum + abs(yDataSum - yFitSum) / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_CENTROID:
    if (yFitSum)
      result = sqrt(sum2) / yDataSum + abs(wSumFit / yFitSum - wSumData / yDataSum);
    else
      result = sqrt(sum2) / yDataSum + abs(wSumData / yDataSum);
    break;
  case ION_FIT_RESIDUAL_SUM_ABS_DEV:
    result = sum / yDataSum;
    break;
  case ION_FIT_RESIDUAL_RMS_DEV_PLUS_ABS_DEV_CHARGE:
    charge = 0;
    for (int j = 0; j < mFunctions; j++)
      charge += param[3 * j + 2] * PI * param[3 * j + 0];
    result = sqrt(sum2) / yDataSum + abs(charge / dx - ionChargeData) / ionChargeData;
    break;
  case ION_FIT_RESIDUAL_MAX_ABS_DEV_PLUS_ABS_DEV_CHARGE:
    charge = 0;
    for (int j = 0; j < mFunctions; j++)
      charge += param[3 * j + 2] * PI * param[3 * j + 0];
    // result = max/(yDataSum/nData) + abs(charge/dx-yDataSum)/yDataSum;
    result = max / peak + abs(charge / dx - ionChargeData) / ionChargeData;
    break;
  default:
    result = 0;
    bombElegant("Invalid residual code in multiLorentzianFunction---seek professional help", NULL);
    break;
  }

  if (result < 0 || isnan(result) || isinf(result)) {
    *invalid = 1;
    result = DBL_MAX;
  }

  if (verbosity > 300) {
    printf("fitting %ld points with %ld-lorentzian, returning %le\n", nData, mFunctions, result);
    printf("sum = %le, sum2 = %le, max = %le, yFitSum = %le, wSumData = %le, wSumFit = %le\n",
           sum, sum2, max, yFitSum, wSumData, wSumFit);
    for (int i = 0; i < mFunctions; i++) {
      printf("param[%d]: ", i);
      for (int j = 0; j < 3; j++)
        printf("%le%c", param[3 * i + j], j == 2 ? '\n' : ',');
    }
    fflush(stdout);
  }

#if USE_MPI
  checkTargetIonFitting(result, *invalid);
#endif

  return result;
}

#if USE_MPI
void checkTargetIonFitting(double myResult, long invalid) {
  MPI_Status status;
  static short *targetBuffer = NULL;
  int targetTag = 1;
  if (hybrid_simplex_comparison_interval <= 0)
    return;
  if (!targetBuffer)
    targetBuffer = (short *)tmalloc(sizeof(*targetBuffer) * n_processors);
  if (targetReached) {
    simplexMinAbort(1);
    return;
  }
  if (!invalid && distribution_fit_target > myResult) {
    int i;
    MPI_Request request;
    targetReached = 1;
    /* send message to other processors */
    for (i = 0; i < n_processors; i++) {
      targetBuffer[i] = targetReached; /* Isend operations can't share memory */
      if (i != myid)
        MPI_Isend(&targetBuffer[i], 1, MPI_SHORT, i, targetTag, MPI_COMM_WORLD, &request);
    }
  }
  if (!targetReached && simplexComparisonStep % hybrid_simplex_comparison_interval == 0) {
    int i, flag;
    /* check for messages from other processors */
    targetReached = 0;
    for (i = 0; i < n_processors; i++) {
      if (i != myid) {
        MPI_Iprobe(i, targetTag, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
          MPI_Recv(&targetBuffer[i], 1, MPI_SHORT, i, targetTag, MPI_COMM_WORLD, &status);
          targetReached += targetBuffer[i];
        }
      }
    }
  }
  if (targetReached)
    simplexMinAbort(1);
  simplexComparisonStep++;
}
#endif

void report(double y, double *x, long pass, long nEval, long n_dimen) {
  long i;

  fprintf(stderr, "pass %ld, after %ld evaluations: result = %.16e\na = ", pass, nEval, y);
  for (i = 0; i < n_dimen; i++)
    fprintf(stderr, "%.8e ", x[i]);
  fputc('\n', stderr);
}

void startSummaryDataOutputPage(IONEFFECTS *ionEffects, long iPass, long nPasses, long nBunches) {
#if USE_MPI
  if (myid == 0) {
#endif
    if (beam_output_all_locations || ionEffects == firstIonEffects) {
      if (SDDS_beamOutput) {
        if (verbosity > 10) {
          printf("Starting page (%ld rows) for ion-related electron beam output\n",
                 nBunches);
          fflush(stdout);
        }
        if (!SDDS_StartPage(SDDS_beamOutput, nBunches)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        iBeamOutput = 0;
      }
    }
    if (ion_output_all_locations == 1 || ionEffects == firstIonEffects) {
      if (SDDS_ionDensityOutput) {
        if (verbosity > 10) {
          printf("Starting page (%ld rows) for ion density output\n",
                 nBunches);
          fflush(stdout);
        }
        if (!SDDS_StartPage(SDDS_ionDensityOutput, nBunches)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        iIonDensityOutput = 0;
      }
    }
#if USE_MPI
  }
#endif
}

void computeIonEffectsElectronBunchParameters(
                                              double **part,
                                              double *time,
                                              int64_t np,
                                              CHARGE *charge,
                                              double *tNow,
                                              long *npTotal,
                                              double *qBunch,
                                              double bunchCentroid[4],
                                              double bunchSigma[4]) {
  if (verbosity > 30) {
    printf("Computing bunch parameters\n");
    fflush(stdout);
  }

  for (int i = 0; i < 4; i++) {
    bunchCentroid[i] = 0;
    bunchSigma[i] = 1;
  }
#if USE_MPI
  rms_emittance_p(part, 0, 1, np, &bunchSigma[0], NULL, &bunchSigma[1], &bunchCentroid[0], &bunchCentroid[1], npTotal);
  rms_emittance_p(part, 2, 3, np, &bunchSigma[2], NULL, &bunchSigma[3], &bunchCentroid[2], &bunchCentroid[3], npTotal);
  *tNow = computeAverage_p(time, np, MPI_COMM_WORLD);
#else
  *npTotal = np;
  rms_emittance(part, 0, 1, np, &bunchSigma[0], NULL, &bunchSigma[1], &bunchCentroid[0], &bunchCentroid[1]);
  rms_emittance(part, 2, 3, np, &bunchSigma[2], NULL, &bunchSigma[3], &bunchCentroid[2], &bunchCentroid[3]);
  compute_average(tNow, time, np);
#endif
  /* index_bunch_assignments() returns register-reduced time; add the per-step
     macro offset so tNow is the true absolute time.  ionEffects uses tNow only
     for ion drift intervals (advanceIonPositions) and time outputs -- never for
     an RF phase or a coord[4] write-back -- so true time is both correct and
     what the drift across step boundaries requires.  Zero unless step_frequency
     is in use, so legacy behavior is unchanged. */
  *tNow += trackingClockOffset();
  for (int i = 0; i < 4; i++) {
    if (isnan(bunchCentroid[i]) || isinf(bunchCentroid[i]))
      bunchCentroid[i] = 0;
    if (bunchSigma[i] > 0 && !isnan(bunchSigma[i]) && !isinf(bunchSigma[i]))
      bunchSigma[i] = sqrt(bunchSigma[i]);
    else
      bunchSigma[i] = 1;
  }
  *qBunch = (*npTotal) * charge->macroParticleCharge;

  if (verbosity > 40) {
    printf("np: %ld, <t>: %le, sigma x,y: %le, %le,  sigma x',y': %le, %le, centroid x,y: %le, %le,  q: %le\n",
           *npTotal, *tNow, bunchSigma[0], bunchSigma[2],
           bunchSigma[1], bunchSigma[3],
           bunchCentroid[0], bunchCentroid[2], *qBunch);
    fflush(stdout);
  }
}

void setIonEffectsElectronBunchOutput(
                                      IONEFFECTS *ionEffects,
                                      double tNow,
                                      long iPass,
                                      long iBunch,
                                      double qBunch,
                                      long npTotal,
                                      double bunchSigma[4],
                                      double bunchCentroid[4]) {
  if (verbosity > 30) {
    printf("Setting SDDS file for electron bunch output\n");
    fflush(stdout);
  }

#if USE_MPI
  if (myid == 0) {
#endif
    if (SDDS_beamOutput) {
      if ((beam_output_all_locations || ionEffects == firstIonEffects) &&
          (!SDDS_SetRowValues(SDDS_beamOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iBeamOutput++,
                              "t", tNow,
                              "Bunch", iBunch, "qBunch", qBunch, "npBunch", npTotal,
                              "Sx", bunchSigma[0], "Sy", bunchSigma[2],
                              "Cx", bunchCentroid[0], "Cy", bunchCentroid[2],
                              "Sxp", bunchSigma[1], "Syp", bunchSigma[3],
                              "Cxp", bunchCentroid[1], "Cyp", bunchCentroid[3],
                              NULL) ||
           !SDDS_SetParameters(SDDS_beamOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "s", ionEffects->sLocation,
                               "Pass", iPass, NULL))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
#if USE_MPI
  }
#endif
}

void setIonEffectsIonParameterOutput(
                                     IONEFFECTS *ionEffects,
                                     double tNow,
                                     long iPass,
                                     long iBunch,
                                     long nBunches,
                                     double qIon,
                                     double ionSigma[2],
                                     double ionCentroid[2]) {
  long iSpecies;
#if USE_MPI
  if (myid == 0) {
#endif
    if ((SDDS_ionDensityOutput) && (((iPass - ionEffects->startPass) * nBunches + iBunch) % ion_output_interval == 0)) {
      if (ion_output_all_locations || ionEffects == firstIonEffects) {
        if (!SDDS_SetParameters(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, &"s", ionEffects->sLocation, "Pass", iPass, NULL) ||
            !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                               "t", tNow, "Bunch", iBunch,
                               "qIons", qIon, "Sx", ionSigma[0], "Sy", ionSigma[1],
                               "Cx", ionCentroid[0], "Cy", ionCentroid[1], "nMacroIons", ionEffects->nTotalIons,
                               "nCoreMacroIons", ionEffects->nCoreIons,
#if USE_MPI
                               "nMacroIonsMin", ionEffects->nMin,
                               "nMacroIonsMax", ionEffects->nMax,
#endif
                               NULL)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        if (ion_species_output) {
          for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
            if (!SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("qIons_", ionProperties.ionName[iSpecies]),
                                   speciesCharge[iSpecies], NULL) ||
                !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("nMacroIons_", ionProperties.ionName[iSpecies]),
                                   speciesCount[iSpecies], NULL) ||
                !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("Cx_", ionProperties.ionName[iSpecies]),
                                   speciesCentroid[iSpecies][0], NULL) ||
                !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("Cy_", ionProperties.ionName[iSpecies]),
                                   speciesCentroid[iSpecies][1], NULL) ||
                !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("Sx_", ionProperties.ionName[iSpecies]),
                                   speciesSigma[iSpecies][0], NULL) ||
                !SDDS_SetRowValues(SDDS_ionDensityOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iIonDensityOutput,
                                   makeSpeciesName("Sy_", ionProperties.ionName[iSpecies]),
                                   speciesSigma[iSpecies][1], NULL)) {
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
              exitElegant(1);
            }
          }
        }
        iIonDensityOutput++;
      }
    }
#if USE_MPI
  }
#endif
}

void advanceIonPositions(IONEFFECTS *ionEffects, long iPass, double tNow) {
  long iSpecies, iIon;

  if (verbosity > 30) {
    printf("Advancing ion positions\n");
    fflush(stdout);
  }

  if (isSlave || !distributedBeam) {
    /*** Advance the ion positions */
    if (iPass >= freeze_ions_until_pass) {
      for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
        for (iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
          ionEffects->coordinate[iSpecies][iIon][0] += (tNow - ionEffects->t) * ionEffects->coordinate[iSpecies][iIon][1];
          ionEffects->coordinate[iSpecies][iIon][2] += (tNow - ionEffects->t) * ionEffects->coordinate[iSpecies][iIon][3];
        }
      }
    }
    ionEffects->t = tNow;
  }
}

void eliminateIonsOutsideSpan(IONEFFECTS *ionEffects) {
  long iSpecies, iIon;
  if (verbosity > 30) {
    printf("Eliminating outside ions\n");
    fflush(stdout);
  }
  if (isSlave || !distributedBeam) {
    if (ionEffects->span[0] || ionEffects->span[1]) {
      long ionCount0, ionCount1;
      /*** Eliminate ions that are outside the simulation region */
      ionCount0 = ionCount1 = 0;
      for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
        ionCount0 += ionEffects->nIons[iSpecies];
        for (iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
          if ((ionEffects->span[0] && fabs(ionEffects->coordinate[iSpecies][iIon][0]) > ionEffects->span[0]) ||
              (ionEffects->span[1] && fabs(ionEffects->coordinate[iSpecies][iIon][2]) > ionEffects->span[1])) {
            if (ionEffects->nIons[iSpecies] > 1) {
              /* Move ion at the top of the array into this slot in the array */
              long k;
              for (k = 0; k < 5; k++)
                ionEffects->coordinate[iSpecies][iIon][k] = ionEffects->coordinate[iSpecies][ionEffects->nIons[iSpecies] - 1][k];
            }
            ionEffects->nIons[iSpecies] -= 1;
            iIon -= 1;
            ionCount1--;
          }
        }
      }
      ionCount1 += ionCount0;
      if (verbosity > 40) {
        printf("Number of ions reduced from %ld to %ld\n", ionCount0, ionCount1);
        fflush(stdout);
      }
    }
  }
}

void generateIons(IONEFFECTS *ionEffects, long iPass, long iBunch, long nBunches,
                  double qBunch, double bunchCentroid[4], double bunchSigma[4]) {
  long iSpecies, index, nToAdd;
  double qToAdd;

  if (verbosity > 30) {
    printf("Generating ions\n");
    fflush(stdout);
  }

  if (isSlave || !distributedBeam) {
    if (((iPass - ionEffects->startPass) * nBunches + iBunch) % ionEffects->generationInterval == 0) {
      /*** Generate ions */
      for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
        nToAdd = 0;
        qToAdd = 0;
        if ((index = ionProperties.sourceGasIndex[iSpecies]) >= 0) {
          /* this is a singly-ionized molecule, so use source gas
             nToAdd =  someFunctionOfPressure(ionEffects->pressure[index], ...);
          */
#if USE_MPI
          /* The macroIons parameter is the number for all processors, so we need to
           * apportion the ions among the working processors
           */
          int64_t nLeft;
          nToAdd = ionEffects->macroIons / (n_processors - 1.0);
          nLeft = ionEffects->macroIons - nToAdd * (n_processors - 1);
          for (long iLeft = 0; iLeft < nLeft; iLeft++) {
            if (leftIonCounter % (n_processors - 1) == (myid - 1))
              nToAdd++;
            leftIonCounter++; /* This counter will be the same on all processors */
          }
#else
          nToAdd = ionEffects->macroIons;
#endif

          if (nToAdd) {
            qToAdd = unitsFactor * qBunch * ionEffects->pressure[index] * ionEffects->generationInterval *
              ionProperties.crossSection[iSpecies] * (ionEffects->sEnd - ionEffects->sStart) / ionEffects->macroIons;
            if (symmetrize) {
              nToAdd *= 2;
              qToAdd /= 2;
            }
            addIons(ionEffects, iSpecies, nToAdd, qToAdd, bunchCentroid, bunchSigma, symmetrize);
          }
        } else if (((index = ionProperties.sourceIonIndex[iSpecies]) >= 0) &&
                   (((iPass - ionEffects->startPass) * nBunches + iBunch) % multiple_ionization_interval == 0)) {
          /* This is a multiply-ionized molecule, so use source ion density.
           * Relevant quantities:
           * ionEffects->nIons[index] --- Number of ions of the source species
           * ionEffects->coordinate[index][j][k] --- kth coordinate of jth source ion
           * ionProperties.crossSection[iSpecies] --- Cross section for producing new ion from the source ions
           */
          /*
            nToAdd = someFunctionOfExistingNumberOfIons(...);
          */

          double beamFact, jx, jy, Pmi, rnd;
          beamFact = jx = jy = Pmi = rnd = 0;
          beamFact = multiple_ionization_interval * 1e-22 * qBunch / e_mks / (2 * PI * bunchSigma[0] * bunchSigma[2]);
          for (int jMacro = 0; jMacro < ionEffects->nIons[index]; jMacro++) {
            jx = ionEffects->coordinate[index][jMacro][0] - bunchCentroid[0];
            jy = ionEffects->coordinate[index][jMacro][2] - bunchCentroid[2];
            Pmi = beamFact * ionProperties.crossSection[iSpecies] *
              exp(-sqr(jx) / (2 * sqr(bunchSigma[0])) - sqr(jy) / (2 * sqr(bunchSigma[2])));

            rnd = random_2(0);
            if (rnd < Pmi) { // multiple ionization occurs
              double qToAdd, mx, my;
              qToAdd = ionEffects->coordinate[index][jMacro][4];
              mx = ionEffects->coordinate[index][jMacro][0];
              my = ionEffects->coordinate[index][jMacro][2];
              addIon_point(ionEffects, iSpecies, qToAdd, mx, my); // add multiply ionized ion

              // Initial kinetic energy
              double vmag, ionMass, vx, vy, rangle, Emi;
              ionMass = 1.672621898e-27 * ionProperties.mass[iSpecies];
              Emi = fabs(gauss_rn_lim(multiple_ionization_energy_peak, multiple_ionization_energy_rms, 3, random_4));
              // Emi = fabs(gauss_rn_lim(20, 10, 3, random_4));
              // Emi = 0;
              vmag = sqrt(2 * Emi * e_mks / ionMass);
              rangle = random_2(0) * 2 * PI;
              vx = vmag * cos(rangle);
              vy = vmag * sin(rangle);
              ionEffects->coordinate[iSpecies][ionEffects->nIons[iSpecies] - 1][1] = vx;
              ionEffects->coordinate[iSpecies][ionEffects->nIons[iSpecies] - 1][3] = vy;

              // delete source ion
              long k;
              for (k = 0; k < 5; k++)
                ionEffects->coordinate[index][jMacro][k] = ionEffects->coordinate[index][ionEffects->nIons[index] - 1][k];
              ionEffects->nIons[index] -= 1;
              jMacro -= 1;
            }
          }
        }
      }
    }
  }
}

void applyElectronBunchKicksToIons(IONEFFECTS *ionEffects, long iPass, double qBunch, double bunchCentroid[4], double bunchSigma[4],
                                   double dpSum[3]) {
  // long localCount;
  double ionMass, ionCharge, *coord, kick[2];
  double tempkick[2], maxkick[2], tempart[4];
  long iSpecies, iIon;

  if (verbosity > 30) {
    printf("Applying bunch kicks to ions\n");
    fflush(stdout);
  }

  memset(&dpSum[0], 0, sizeof(dpSum[0]) * 3);

  if (isSlave || !distributedBeam) {
    if (iPass >= freeze_ions_until_pass) {
      /*** Determine and apply kicks from beam to ions */
      // localCount = 0;
      for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
        kick[0] = kick[1] = 0;

        /* Relevant quantities:
         * ionProperties.chargeState[iSpecies] --- Charge state of the ion (integer)
         * ionProperties.mass[iSpecies] --- Mass of the ion (AMUs)
         * qBunch --- bunch charge (C)
         * sigma[0], sigma[1] --- x, y sigma (m)
         * centroid[0], centroid[1] --- x , y centroid (m)
         * ionEffects->nIons[index] --- Number of ions of the source species
         * ionEffects->coordinate[index][j][k] --- kth coordinate of jth source ion
         */

        ionMass = 1.672621898e-27 * ionProperties.mass[iSpecies];
        ionCharge = (double)ionProperties.chargeState[iSpecies];

        maxkick[0] = maxkick[1] = 0;
        tempkick[0] = tempkick[1] = 0;
        tempart[0] = bunchSigma[0] + bunchCentroid[0];
        tempart[2] = 0;
        gaussianBeamKick(tempart, bunchCentroid, bunchSigma, 1, tempkick, qBunch, ionMass, ionCharge);
        maxkick[0] = 2 * abs(tempkick[0]);

        tempart[2] = bunchSigma[2] + bunchCentroid[2];
        tempart[0] = 0;
        gaussianBeamKick(tempart, bunchCentroid, bunchSigma, 1, tempkick, qBunch, ionMass, ionCharge);
        maxkick[1] = 2 * abs(tempkick[1]);

        // localCount += ionEffects->nIons[iSpecies];
        for (iIon = 0; iIon < ionEffects->nIons[iSpecies]; iIon++) {
          coord = ionEffects->coordinate[iSpecies][iIon];
          kick[0] = kick[1] = 0;
          gaussianBeamKick(coord, bunchCentroid, bunchSigma, 1, kick, qBunch, ionMass, ionCharge);

          if (abs(kick[0]) < maxkick[0] && abs(kick[1]) < maxkick[1]) {
            ionEffects->coordinate[iSpecies][iIon][1] += kick[0];
            ionEffects->coordinate[iSpecies][iIon][3] += kick[1];
            // Transverse momentum change in kg*m/s, weighted by macro-ion charge
            // in order to account for fact that different macro-ions may represent different
            // numbers of actual ions

            // dpSum[0] += kick[0]*ionMass*(ionEffects->coordinate[iSpecies][iIon][4]/e_mks/ionCharge);
            // dpSum[1] += kick[1]*ionMass*(ionEffects->coordinate[iSpecies][iIon][4]/e_mks/ionCharge);
            // dpSum[2] += ionEffects->coordinate[iSpecies][iIon][4]/e_mks/ionCharge;

            // momentum change = (velocity change) x (mass of ion) x (number of ions in macroparticle)
            dpSum[0] += kick[0] * ionMass * (ionEffects->coordinate[iSpecies][iIon][4] / e_mks / ionCharge);
            dpSum[1] += kick[1] * ionMass * (ionEffects->coordinate[iSpecies][iIon][4] / e_mks / ionCharge);
          }
        }
      }
    }
  }

#if USE_MPI
  // Share dpSum across all cores
  double dpSumGlobal[3];
  MPI_Allreduce(dpSum, dpSumGlobal, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  memcpy(&dpSum[0], &dpSumGlobal[0], sizeof(dpSum[0]) * 3);
  // if (dpSum[2]) {
  // dpSum[0] /= dpSum[2];
  // dpSum[1] /= dpSum[2];
  // }
#endif
}

void computeIonOverallParameters(
                                 IONEFFECTS *ionEffects, double ionCentroid[2], double ionSigma[2], double *qIonReturn, long *nIonsTotal,
                                 double bunchCentroid[4], double bunchSigma[4], long iBunch) {
  long mTot, nTot, jMacro;
  double bx1, bx2, by1, by2;
  double qIon;
  long iSpecies;

  if (verbosity > 30) {
    printf("Computing centroid and sigma of ions\n");
    fflush(stdout);
  }

  /* use these as limits on the ion coordinates to include in the centroid and rms calculations,
   * i.e., the core ions
   */
  if (ionFieldMethod == ION_FIELD_GAUSSIAN) {
    /*
      bx1 = bunchCentroid[0] - 3*bunchSigma[0];
      bx2 = bunchCentroid[0] + 3*bunchSigma[0];
      by1 = bunchCentroid[2] - 3*bunchSigma[2];
      by2 = bunchCentroid[2] + 3*bunchSigma[2];
    */
    bx1 = bunchCentroid[0] - gaussian_ion_range * bunchSigma[0];
    bx2 = bunchCentroid[0] + gaussian_ion_range * bunchSigma[0];
    by1 = bunchCentroid[2] - gaussian_ion_range * bunchSigma[2];
    by2 = bunchCentroid[2] + gaussian_ion_range * bunchSigma[2];
  } else {
    bx1 = by1 = -DBL_MAX;
    bx2 = by2 = DBL_MAX;
  }

  if (ion_species_output) {
    if (!speciesCentroid)
      speciesCentroid = (double **)zarray_2d(sizeof(double), ionProperties.nSpecies, 2);
    if (!speciesSigma)
      speciesSigma = (double **)zarray_2d(sizeof(double), ionProperties.nSpecies, 2);
    if (!speciesCharge)
      speciesCharge = (double *)tmalloc(sizeof(*speciesCharge) * ionProperties.nSpecies);
    if (!speciesCount)
      speciesCount = (long *)tmalloc(sizeof(*speciesCount) * ionProperties.nSpecies);
  }

  /* Compute charge-weighted centroids */
  ionCentroid[0] = ionCentroid[1] = 0;
  qIon = *qIonReturn = 0;
  nTot = 0; /* counts the total number of ions */
  mTot = 0; /* counts the core number of ions */
  if (isSlave || !distributedBeam) {
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      /* Relevant quantities:
       * ionProperties.chargeState[iSpecies] --- Charge state of the ion (integer)
       * ionEffects->nIons[index] --- Number of ions of the source species
       * ionEffects->coordinate[index][j][k] --- kth coordinate of jth source ion
       */
      if (ion_species_output) {
        speciesCentroid[iSpecies][0] = speciesCentroid[iSpecies][1] = speciesCharge[iSpecies] = 0;
        speciesCount[iSpecies] = 0;
      }
      nTot += ionEffects->nIons[iSpecies];
      for (jMacro = 0; jMacro < ionEffects->nIons[iSpecies]; jMacro++) {
        if ((ionEffects->coordinate[iSpecies][jMacro][0] > bx1) && (ionEffects->coordinate[iSpecies][jMacro][0] < bx2) &&
            (ionEffects->coordinate[iSpecies][jMacro][2] > by1) && (ionEffects->coordinate[iSpecies][jMacro][2] < by2)) {
          if (ion_species_output) {
            speciesCentroid[iSpecies][0] += ionEffects->coordinate[iSpecies][jMacro][0] * ionEffects->coordinate[iSpecies][jMacro][4];
            speciesCentroid[iSpecies][1] += ionEffects->coordinate[iSpecies][jMacro][2] * ionEffects->coordinate[iSpecies][jMacro][4];
            speciesCharge[iSpecies] += ionEffects->coordinate[iSpecies][jMacro][4];
            speciesCount[iSpecies] += 1;
          }
          ionCentroid[0] += ionEffects->coordinate[iSpecies][jMacro][0] * ionEffects->coordinate[iSpecies][jMacro][4];
          ionCentroid[1] += ionEffects->coordinate[iSpecies][jMacro][2] * ionEffects->coordinate[iSpecies][jMacro][4];
          qIon += ionEffects->coordinate[iSpecies][jMacro][4];
          mTot++;
        }
      }
    }
  } else {
    /* clear out the arrays on the master processor */
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (ion_species_output) {
        speciesCentroid[iSpecies][0] = speciesCentroid[iSpecies][1] = speciesCharge[iSpecies] = 0;
        speciesCount[iSpecies] = 0;
      }
    }
  }

#if USE_MPI
  /* Sum ion centroid and charge data over all nodes */

  /* Find min and max ion counts */
  long nTotMin, nTotMax;
  if (myid == 0)
    nTot = LONG_MAX;
  MPI_Allreduce(&nTot, &nTotMin, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  ionEffects->nMin = nTotMin;
  if (myid == 0)
    nTot = LONG_MIN;
  MPI_Allreduce(&nTot, &nTotMax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  ionEffects->nMax = nTotMax;
  if (myid == 0)
    nTot = 0;

  /* total count, total charge, and centroids */
  double inBuffer[5], sumBuffer[5]; // to increase MPI efficiency
  double ionCentroidTotal[2];
  inBuffer[0] = nTot;
  inBuffer[1] = qIon;
  inBuffer[2] = ionCentroid[0];
  inBuffer[3] = ionCentroid[1];
  inBuffer[4] = mTot;
  MPI_Allreduce(inBuffer, sumBuffer, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  ionEffects->nTotalIons = nTot = sumBuffer[0];
  *qIonReturn = qIon = sumBuffer[1];
  ionCentroidTotal[0] = sumBuffer[2];
  ionCentroidTotal[1] = sumBuffer[3];
  ionEffects->nCoreIons = mTot = sumBuffer[4];
  if (qIon != 0) {
    ionCentroid[0] = ionCentroidTotal[0] / qIon;
    ionCentroid[1] = ionCentroidTotal[1] / qIon;
  } else
    ionCentroid[0] = ionCentroid[1] = 0; // Not really needed
  if (verbosity > 100) {
    printf("qIon = %le, centroids = %le, %le, counts: %ld, %ld\n",
           qIon, ionCentroid[0], ionCentroid[1], nTotMin, nTotMax);
    fflush(stdout);
  }
  if (ion_species_output) {
    double *speciesInBuffer, *speciesSumBuffer;
    speciesInBuffer = (double *)tmalloc(sizeof(*speciesInBuffer) * 4 * ionProperties.nSpecies);
    speciesSumBuffer = (double *)tmalloc(sizeof(*speciesSumBuffer) * 4 * ionProperties.nSpecies);
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (myid != 0) {
        speciesInBuffer[iSpecies * 4 + 0] = speciesCentroid[iSpecies][0];
        speciesInBuffer[iSpecies * 4 + 1] = speciesCentroid[iSpecies][1];
        speciesInBuffer[iSpecies * 4 + 2] = speciesCharge[iSpecies];
        speciesInBuffer[iSpecies * 4 + 3] = speciesCount[iSpecies];
      } else {
        speciesInBuffer[iSpecies * 4 + 0] = speciesInBuffer[iSpecies * 4 + 1] =
          speciesInBuffer[iSpecies * 4 + 2] = speciesInBuffer[iSpecies * 4 + 3] = 0;
      }
    }
    MPI_Allreduce(speciesInBuffer, speciesSumBuffer, 4 * ionProperties.nSpecies, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      speciesCentroid[iSpecies][0] = speciesSumBuffer[iSpecies * 4 + 0];
      speciesCentroid[iSpecies][1] = speciesSumBuffer[iSpecies * 4 + 1];
      speciesCharge[iSpecies] = speciesSumBuffer[iSpecies * 4 + 2];
      speciesCount[iSpecies] = speciesSumBuffer[iSpecies * 4 + 3];
      if (speciesCharge[iSpecies]) {
        speciesCentroid[iSpecies][0] /= speciesCharge[iSpecies];
        speciesCentroid[iSpecies][1] /= speciesCharge[iSpecies];
      } else
        speciesCentroid[iSpecies][0] = speciesCentroid[iSpecies][1] = 0;
    }
    free(speciesInBuffer);
    free(speciesSumBuffer);
  }
#else // Non-MPI
  *qIonReturn = qIon;
  ionEffects->nTotalIons = nTot;
  ionEffects->nCoreIons = mTot;
  if (qIon) {
    ionCentroid[0] = ionCentroid[0] / qIon;
    ionCentroid[1] = ionCentroid[1] / qIon;
  } else
    ionCentroid[0] = ionCentroid[1] = 0;
  if (ion_species_output) {
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (speciesCharge[iSpecies]) {
        speciesCentroid[iSpecies][0] /= speciesCharge[iSpecies];
        speciesCentroid[iSpecies][1] /= speciesCharge[iSpecies];
      } else
        speciesCentroid[iSpecies][0] = speciesCentroid[iSpecies][1] = 0;
    }
  }
#endif
  *nIonsTotal = nTot;

  /* Compute charge-weighted rms size */
  ionSigma[0] = ionSigma[1] = 0;
  if (isSlave || !distributedBeam) {
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      /* Relevant quantities:
       * ionProperties.chargeState[iSpecies] --- Charge state of the ion (integer)
       * ionEffects->nIons[index] --- Number of ions of the source species
       * ionEffects->coordinate[index][j][k] --- kth coordinate of jth source ion
       */

      if (ion_species_output)
        speciesSigma[iSpecies][0] = speciesSigma[iSpecies][1] = 0;
      for (jMacro = 0; jMacro < ionEffects->nIons[iSpecies]; jMacro++) {
        if ((ionEffects->coordinate[iSpecies][jMacro][0] > bx1) && (ionEffects->coordinate[iSpecies][jMacro][0] < bx2) &&
            (ionEffects->coordinate[iSpecies][jMacro][2] > by1) && (ionEffects->coordinate[iSpecies][jMacro][2] < by2)) {
          ionSigma[0] += sqr(ionEffects->coordinate[iSpecies][jMacro][0] - ionCentroid[0]) * ionEffects->coordinate[iSpecies][jMacro][4];
          ionSigma[1] += sqr(ionEffects->coordinate[iSpecies][jMacro][2] - ionCentroid[1]) * ionEffects->coordinate[iSpecies][jMacro][4];
          if (ion_species_output) {
            speciesSigma[iSpecies][0] += sqr(ionEffects->coordinate[iSpecies][jMacro][0] - speciesCentroid[iSpecies][0]) * ionEffects->coordinate[iSpecies][jMacro][4];
            speciesSigma[iSpecies][1] += sqr(ionEffects->coordinate[iSpecies][jMacro][2] - speciesCentroid[iSpecies][1]) * ionEffects->coordinate[iSpecies][jMacro][4];
          }
        }
      }
    }
  } else {
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (ion_species_output)
        speciesSigma[iSpecies][0] = speciesSigma[iSpecies][1] = 0;
    }
  }

#if USE_MPI
  double ionSigmaTotal[2];
  MPI_Allreduce(ionSigma, ionSigmaTotal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (qIon) {
    ionSigma[0] = sqrt(ionSigmaTotal[0] / qIon);
    ionSigma[1] = sqrt(ionSigmaTotal[1] / qIon);
  }
  if (verbosity > 100) {
    printf("sigmas = %le, %le\n",
           ionSigma[0], ionSigma[1]);
    fflush(stdout);
  }
  if (ion_species_output) {
    double *speciesInBuffer, *speciesSumBuffer;
    speciesInBuffer = (double *)tmalloc(sizeof(*speciesInBuffer) * 2 * ionProperties.nSpecies);
    speciesSumBuffer = (double *)tmalloc(sizeof(*speciesSumBuffer) * 2 * ionProperties.nSpecies);
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (myid != 0) {
        speciesInBuffer[2 * iSpecies + 0] = speciesSigma[iSpecies][0];
        speciesInBuffer[2 * iSpecies + 1] = speciesSigma[iSpecies][1];
      } else {
        speciesInBuffer[2 * iSpecies + 0] = speciesInBuffer[2 * iSpecies + 1] = 0;
      }
    }
    MPI_Allreduce(speciesInBuffer, speciesSumBuffer, 2 * ionProperties.nSpecies, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (speciesCharge[iSpecies]) {
        speciesSigma[iSpecies][0] = sqrt(speciesSumBuffer[2 * iSpecies + 0] / speciesCharge[iSpecies]);
        speciesSigma[iSpecies][1] = sqrt(speciesSumBuffer[2 * iSpecies + 1] / speciesCharge[iSpecies]);
      }
    }
    free(speciesInBuffer);
    free(speciesSumBuffer);
  }
#else
  if (qIon) {
    ionSigma[0] = sqrt(ionSigma[0] / qIon);
    ionSigma[1] = sqrt(ionSigma[1] / qIon);
  }
  if (ion_species_output) {
    for (iSpecies = 0; iSpecies < ionProperties.nSpecies; iSpecies++) {
      if (speciesCharge[iSpecies]) {
        speciesSigma[iSpecies][0] = sqrt(speciesSigma[iSpecies][0] / speciesCharge[iSpecies]);
        speciesSigma[iSpecies][1] = sqrt(speciesSigma[iSpecies][1] / speciesCharge[iSpecies]);
      }
    }
  }
#endif
}

void applyIonKicksToElectronBunch(
                                  IONEFFECTS *ionEffects,
                                  double **part,
                                  int64_t np,
                                  double Po,
                                  long iBunch,
                                  long iPass,
                                  double qBunch,
                                  double bunchCentroid[4],
                                  double bunchSigma[4],
                                  double qIon,
                                  long nIonsTotal,
                                  double ionCentroid[2],
                                  double ionSigma[2],
                                  double dpSum[3] // sum of momentum change applied to ions
                                  ) {
  double kick[2], dpSumBunch[2];
  int64_t ip;
  double paramValueX[9], paramValueY[9];
  long circuitBreaker[9];
  double tempCentroid[9][2], tempSigma[9][2], tempkick[2];
  double tempQ[9];
  double normX, normY;
  double slopeChange[2] = {0, 0};

  if (verbosity > 30) {
    printf("Applying ion kicks to bunch\n");
    fflush(stdout);
  }

  if (ionEffects->ion2dDensity) {
    /* Use Poisson solver for ion fields */

    /* - Create 2d histogram of ion density */
    make2dIonHistogram(ionEffects);

    /* - Solve poisson equation */

    // double C1;
    double delta[2];

    // C1 = e_mks * re_mks / me_mks / c_mks;

    for (int iPlane = 0; iPlane < 2; iPlane++) {
      delta[iPlane] = 2 * ionEffects->poisson_span[iPlane] / (ionEffects->n2dGridIon[iPlane] - 1.0);
    }

    // These buffers are never read, just overwritten in the wrapper, no need to clear them out

    poissonSolverWrapper(ionEffects->ion2dDensity, ionEffects->ionPotential,
                         ionEffects->n2dGridIon[0], ionEffects->n2dGridIon[1],
                         ionEffects->xKickPoisson, ionEffects->yKickPoisson, delta);

    // testFunc();
    //  debug output
    /*
      FILE *findex1, *findex2, *findex3, *findex4;
      double C3;
      double temp3[2];
      //if ((iPass >= 80) && (iBunch > 180) && (ionEffects->sLocation > 900)) {
      //if ((iPass == 0) && (iBunch == 0) && (ionEffects->sLocation < 100)) {
      //if ((iPass%81 == 0) && (iBunch == 0)) {
      //if ((ionSigma[1] > 3e-4) || (iBunch == 1)) {
      if ((iBunch == 0) && (ionEffects->sLocation < 200) && (iPass%10==1)) {
      printf("pass %d, bunch %d \n", iPass, iBunch);
      C3 =  -e_mks / (Po * me_mks * c_mks * c_mks * 8.85e-12);
      findex1 = fopen("dens.dat", "a");
      findex2 = fopen("potential.dat", "a");
      findex3 = fopen("xkick.dat", "a");
      findex4 = fopen("ykick.dat", "a");
      for(int i=0; i<ionEffects->n2dGridIon[0]; i++) {
      for(int j=0; j<ionEffects->n2dGridIon[1]; j++) {
      fprintf(findex1, "%4.3e \n ", ionEffects->ion2dDensity[i][j]);
      fprintf(findex2, "%4.3e \n ", ionEffects->ionPotential[i][j]);

      temp3[0] = C3 * ionEffects->xKickPoisson[i][j];
      temp3[1] = C3 * ionEffects->yKickPoisson[i][j];
      fprintf(findex3, "%4.3e \n ", temp3[0]);
      fprintf(findex4, "%4.3e \n ", temp3[1]);
      }
      //fprintf(findex, "\n");
      }
      fclose(findex1);
      fclose(findex2);
      fclose(findex3);
      fclose(findex4);

      //gaussian comp
      double fkick[2], fpart[4], xdelta, ydelta, C1, tempk[2];
      FILE * fcalc, *fgauss;

      fgauss = fopen("gauss_kick.dat", "a");

      for (int i=0; i<ionEffects->n2dGridIon[0]; i++) {
      fpart[0] = -ionEffects->span[0] + i*delta[0];

      for (int j=0; j<ionEffects->n2dGridIon[1]; j++) {
      fpart[2] = -ionEffects->span[1] + j*delta[1];
      fkick[0] = 0;
      fkick[1] = 0;

      gaussianBeamKick(fpart, ionCentroid, ionSigma, 0, kick, qIon, me_mks, 1);
      tempk[0] = kick[0] / c_mks / Po;
      tempk[1] = kick[1] / c_mks / Po;
      fprintf(fgauss, "%4.3e %4.3e \n", tempk[0], tempk[1]);

      }
      }

      fclose(fgauss);
      printf("output written \n");

      }
    */

    /* - Compute field and apply kicks to electrons */
    double C2;
    long iBin[2];
    C2 = e_mks / (Po * me_mks * c_mks * c_mks * 8.85e-12);

    for (ip = 0; ip < np; ip++) {
      if ((abs(part[ip][0]) < ionEffects->poisson_span[0]) && (abs(part[ip][2]) < ionEffects->poisson_span[1])) {
        iBin[0] = floor((part[ip][0] + ionEffects->poisson_span[0]) / delta[0]);
        iBin[1] = floor((part[ip][2] + ionEffects->poisson_span[1]) / delta[1]);
        part[ip][1] -= C2 * ionEffects->xKickPoisson[iBin[0]][iBin[1]];
        part[ip][3] -= C2 * ionEffects->yKickPoisson[iBin[0]][iBin[1]];
        // debug
        /*
          double kickampy;
          kickampy = abs(C2 * ionEffects->yKickPoisson[iBin[0]][iBin[1]]);
          if (kickampy > 5e-7) {
          int tempint = 0;
          //printf("large kickampy: %3.2e, Pass %d, bunch %d, sLoc %3.1f \n", kickampy, iPass, iBunch, ionEffects->sLocation);
          }
        */
      }
    }

  } else {
    /* Use 1D fits for ion fields */

    ionEffects->ionChargeFromFit[0] = ionEffects->ionChargeFromFit[1] = -1;

    makeIonHistograms(ionEffects, ionProperties.nSpecies, bunchSigma, ionSigma);

    if ((ionEffects->ionFieldMethod = ionFieldMethod) == ION_FIELD_EGAUSSIAN || conserve_momentum) {
      if (iPass >= freeze_electrons_until_pass) {
        // The kicks from ions are the same for all electrons;
        // Using conservation of momentum, the total is equal and opposite to the kick from
        // the electrons to the ions.
        if (qBunch) {
          // momentum change per electron = -(total momentum change of ions) / (number of electrons)
          slopeChange[0] = -dpSum[0] / (qBunch / e_mks) / (me_mks * c_mks * Po);
          slopeChange[1] = -dpSum[1] / (qBunch / e_mks) / (me_mks * c_mks * Po);
        }
        if (!conserve_momentum) {
          for (ip = 0; ip < np; ip++) {
            part[ip][1] += slopeChange[0];
            part[ip][3] += slopeChange[1];
          }
          return;
        }
      }
    }

    if ((ionEffects->ionFieldMethod = ionFieldMethod) != ION_FIELD_GAUSSIAN) {
      // multi-gaussian or multi-lorentzian kick

      /* We take a return value here for future improvement in which the fitting function is automatically selected. */

      ionEffects->ionFieldMethod =
        multipleWhateverFit(bunchSigma, bunchCentroid, paramValueX, paramValueY, ionEffects, ionSigma, ionCentroid);

      /* determine the charge implied by the fits */
      for (int iPlane = 0; iPlane < 2; iPlane++) {
        ionEffects->ionChargeFromFit[iPlane] = 0;
        if (ionEffects->ionFieldMethod == ION_FIELD_BIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_TRIGAUSSIAN) {
          for (int i = 0; i < nFunctions; i++)
            ionEffects->ionChargeFromFit[iPlane] += iPlane == 0 ? paramValueX[3 * i + 2] * sqrt(PIx2) * paramValueX[3 * i + 0] / ionEffects->ionDelta[iPlane] : paramValueY[3 * i + 2] * sqrt(PIx2) * paramValueY[3 * i + 0] / ionEffects->ionDelta[iPlane];
        } else {
          for (int i = 0; i < nFunctions; i++)
            ionEffects->ionChargeFromFit[iPlane] += iPlane == 0 ? paramValueX[3 * i + 2] * PI * paramValueX[3 * i + 0] / ionEffects->ionDelta[iPlane] : paramValueY[3 * i + 2] * PI * paramValueY[3 * i + 0] / ionEffects->ionDelta[iPlane];
        }
      }

      if (verbosity > 40) {
        long i, j;
        double sum[2];
        for (i = 0; i < 2; i++) {
          sum[i] = 0;
          for (j = 0; j < ionEffects->ionBins[i]; j++)
            sum[i] += ionEffects->ionHistogram[i][j];
        }
        printf("Charge check on histograms: x=%le, y=%le, q=%le\n", sum[0], sum[1], qIon);
        printf("Residual of fits: x=%le, y=%le\n", ionEffects->xyFitResidual[0], ionEffects->xyFitResidual[1]);
        printf("Charge from fits: x=%le, y=%le\n", ionEffects->ionChargeFromFit[0], ionEffects->ionChargeFromFit[1]);
        fflush(stdout);
      }

      /* these factors needed because we fit charge histograms instead of charge densities */
      normX = ionEffects->ionDelta[0];
      normY = ionEffects->ionDelta[1];

      if (ionEffects->ionFieldMethod == ION_FIELD_BIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionEffects->ionFieldMethod == ION_FIELD_GAUSSIANFIT) {
        /* paramValueX[0..8] = sigma1, centroid1, height1, sigma2, centroid2, height2, [sigma3, centroid3, height3] */
        /* paramValueY[0..8] = sigma1, centroid1, height1, sigma2, centroid2, height2, [sigma3, centroid3, height3] */
        for (int ix = 0; ix < nFunctions; ix++) {
          for (int iy = 0; iy < nFunctions; iy++) {
            tempCentroid[ix + iy * nFunctions][0] = paramValueX[1 + ix * 3];
            tempCentroid[ix + iy * nFunctions][1] = paramValueY[1 + iy * 3];
            tempSigma[ix + iy * nFunctions][0] = paramValueX[0 + ix * 3];
            tempSigma[ix + iy * nFunctions][1] = paramValueY[0 * iy * 3];

            /* We need 2*Pi*SigmaX*Sigmay here because in biGaussianFunction() the factor 1/(sqrt(2*pi)*sigma) is
             * hidden in the height parameter. We need to remove it for use in the B-E formula.
             * Dividing by qIon is needed because we are making a product of two functions, rho(x) and rho(y),
             * each of which will integrate to qIon.
             */

            tempQ[ix + iy * nFunctions] =
              paramValueX[2 + ix * 3] / normX * paramValueY[2 + iy * 3] / normY *
              2 * PI * paramValueX[0 + ix * 3] * paramValueY[0 + iy * 3] / ionEffects->qTotal;

            // tempQ[ix+iy*nFunctions] =
            //  paramValueX[2+ix*3] / normX * paramValueY[2+iy*3] / normY *
            //  2 * PI * paramValueX[0+ix*3] * paramValueY[0+iy*3] / qIon;

            // tempQ[ix+iy*nFunctions] =
            //   paramValueX[2+ix*3] / normX / ionEffects->ionChargeFromFit[0] *
            //   paramValueY[2+iy*3] / normY /  ionEffects->ionChargeFromFit[1] *
            //   2 * PI * paramValueX[0+ix*3] * paramValueY[0+iy*3] * ionEffects->qTotal;

            if (tempQ[ix + iy * nFunctions] < 1e-6 * ionEffects->qTotal)
              // Ignore these contributions, as they are likely to have wild values for centroid and size that
              // can lead to numerical problems
              tempQ[ix + iy * nFunctions] = 0;
          }
        }
      } else if (ionEffects->ionFieldMethod == ION_FIELD_BILORENTZIAN || ionEffects->ionFieldMethod == ION_FIELD_TRILORENTZIAN) {
        /* paramValueX[0..8] = ax1, centroidx1, heightx1, ax2, centroidx2, heightx2, [ax3, centroidx3, heightx3] */
        /* paramValueY[0..8] = ay1, centroidy1, heighty1, ay2, centroidy2, heighty2, [ay3, centroidy3, heighty3] */
        for (int ix = 0; ix < nFunctions; ix++) {
          for (int iy = 0; iy < nFunctions; iy++) {
            tempCentroid[ix + iy * nFunctions][0] = paramValueX[1 + ix * 3];
            tempCentroid[ix + iy * nFunctions][1] = paramValueY[1 + iy * 3];
            tempSigma[ix + iy * nFunctions][0] = paramValueX[0 + ix * 3];
            tempSigma[ix + iy * nFunctions][1] = paramValueY[0 * iy * 3];

            /* Here we account for the bin sizes (normX and normY) and convert the height parameters to those
             * used in a standard Lorentzian, PI*L(0)*a. We also divide out the total
             * charge since otherwise the 2d integral will be qIon^2, instead of qIon.
             */
            tempQ[ix + iy * nFunctions] =
              paramValueX[2 + ix * 3] * PI * paramValueX[0 + ix * 3] / normX *
              paramValueY[2 + iy * 3] * PI * paramValueY[0 + iy * 3] / normY / ionEffects->qTotal;
            if (tempQ[ix + iy * nFunctions] < 1e-6 * ionEffects->qTotal)
              // Ignore these contributions, as they are likely to have wild values for centroid and size that
              // can lead to numerical problems
              tempQ[ix + iy * nFunctions] = 0;
          }
        }
      } else
        bombElegant("invalid field method used for ION_EFFECTS, seek professional help", NULL);
    } else {
      ionEffects->qTotal = qIon;
      for (int iPlane = 0; iPlane < 2; iPlane++) {
        double residual;
        residual = 0;
        for (int i = 0; i < ionEffects->ionBins[iPlane]; i++) {
          double xy;
          xy = (i + 0.5) * ionEffects->ionDelta[iPlane] - ionEffects->ionRange[iPlane] / 2 - ionCentroid[iPlane];
          ionEffects->ionHistogramFit[iPlane][i] = qIon * ionEffects->ionDelta[iPlane] / sqrt(PIx2) / ionSigma[iPlane] * exp(-sqr(xy / ionSigma[iPlane]) / 2);
          residual += sqr(ionEffects->ionHistogramFit[iPlane][i] - ionEffects->ionHistogram[iPlane][i]);
        }
        ionEffects->xyFitResidual[iPlane] = sqrt(residual) / (qIon * ionEffects->ionDelta[iPlane] / sqrt(PIx2) / ionSigma[iPlane]);
        ionEffects->xyFitParameter2[iPlane][0] = ionSigma[iPlane];
        ionEffects->xyFitParameter2[iPlane][1] = ionCentroid[iPlane];
        ionEffects->xyFitParameter2[iPlane][2] = qIon * ionEffects->ionDelta[iPlane] / sqrt(PIx2) / ionSigma[iPlane];
      }
    }

    for (int i = 0; i < 9; i++)
      circuitBreaker[i] = 0;
    dpSumBunch[0] = dpSumBunch[1] = 0;
    if (isSlave || !distributedBeam) {
      /*** Determine and apply kicks to beam from the total ion field */
#if MPI_DEBUG
      printf("Applying kicks to electron beam\n");
#endif
      if (qIon && ionSigma[0] > 0 && ionSigma[1] > 0 && nIonsTotal > 10 && iPass >= freeze_electrons_until_pass) {
        switch (ionEffects->ionFieldMethod) {
        case ION_FIELD_GAUSSIAN:
          for (ip = 0; ip < np; ip++) {
            kick[0] = kick[1] = 0;

            gaussianBeamKick(part[ip], ionCentroid, ionSigma, 0, kick, qIon, me_mks, 1);
            part[ip][1] += kick[0] / c_mks / Po;
            part[ip][3] += kick[1] / c_mks / Po;
            dpSumBunch[0] += kick[0] * me_mks;
            dpSumBunch[1] += kick[1] * me_mks;
          }
          break;
        case ION_FIELD_BIGAUSSIAN:
        case ION_FIELD_TRIGAUSSIAN:
        case ION_FIELD_GAUSSIANFIT:
          double maxkick[2], tempart[4];
          maxkick[0] = maxkick[1] = 0;
          for (int i = 0; i < nFunctions * nFunctions; i++) {
            tempart[0] = tempCentroid[i][0] + tempSigma[i][0];
            tempart[2] = 0;
            gaussianBeamKick(tempart, tempCentroid[i], tempSigma[i], 0, tempkick, tempQ[i], me_mks, 1);
            maxkick[0] += 4 * abs(tempkick[0]);

            tempart[2] = tempCentroid[i][1] + tempSigma[i][1];
            tempart[0] = 0;
            gaussianBeamKick(tempart, tempCentroid[i], tempSigma[i], 0, tempkick, tempQ[i], me_mks, 1);
            maxkick[1] += 4 * abs(tempkick[1]);
          }
          for (ip = 0; ip < np; ip++) {
            kick[0] = kick[1] = 0;
            for (int i = 0; i < nFunctions * nFunctions; i++) {
              if (tempQ[i]) {
                gaussianBeamKick(part[ip], tempCentroid[i], tempSigma[i], 0, tempkick, tempQ[i], me_mks, 1);
                if (!isnan(tempkick[0]) && !isinf(tempkick[0]) && !isnan(tempkick[1]) && !isinf(tempkick[1]) &&
                    (abs(tempkick[0]) < maxkick[0]) && (abs(tempkick[1]) < maxkick[1])) {
                  kick[0] += tempkick[0];
                  kick[1] += tempkick[1];
                } else {
                  // printf("kick %3.2e,%3.2e > maxkick %3.2e,%3.2e: turn %ld , bunch %ld , cx1=%3.2e, cy=%3.2e, cx2=%3.2e,
                  // cy2=%3.2e, sx1=%3.2e, sy1=%3.2e, sx2=%3.2e, sy2=%3.2e, x=%3.2e, y=%3.2e \n",
                  // tempkick[0], tempkick[1], maxkick[0], maxkick[1], iPass, iBunch, tempCentroid[0][0],
                  // tempCentroid[0][1],  tempCentroid[1][0], tempCentroid[1][1], tempSigma[0][0],
                  // tempSigma[0][1], tempSigma[1][0], tempSigma[1][1], part[ip][0], part[ip][2]);
                  circuitBreaker[i]++;
                }
              }
            }
            part[ip][1] += kick[0] / c_mks / Po;
            part[ip][3] += kick[1] / c_mks / Po;
            dpSumBunch[0] += kick[0] * me_mks;
            dpSumBunch[1] += kick[1] * me_mks;
          }
          break;
        case ION_FIELD_BILORENTZIAN:
        case ION_FIELD_TRILORENTZIAN:
          for (ip = 0; ip < np; ip++) {
            kick[0] = kick[1] = 0;
            for (int i = 0; i < nFunctions * nFunctions; i++) {
              if (tempQ[i]) {
                evaluateVoltageFromLorentzian(tempkick,
                                              tempSigma[i][0], tempSigma[i][1],
                                              part[ip][0] - tempCentroid[i][0], part[ip][2] - tempCentroid[i][1]);
                if (!isnan(tempkick[0]) && !isinf(tempkick[0]) && !isnan(tempkick[1]) && !isinf(tempkick[1])) {
                  kick[0] += tempQ[i] * tempkick[0];
                  kick[1] += tempQ[i] * tempkick[1];
                } else
                  circuitBreaker[i]++;
              }
            }
            part[ip][1] -= kick[0] / (Po * particleMassMV * 1e6 * particleRelSign);
            part[ip][3] -= kick[1] / (Po * particleMassMV * 1e6 * particleRelSign);
            dpSumBunch[0] += kick[0] * e_mks / c_mks;
            dpSumBunch[1] += kick[1] * e_mks / c_mks;
          }
          break;
        default:
          bombElegant("invalid field method used for ION_EFFECTS, seek professional help", NULL);
          break;
        }
      }
    }

    if (conserve_momentum && qBunch) {
#if USE_MPI
      // Share dpSumBunch across all cores
      double dpSumBunchGlobal[2];
      MPI_Allreduce(dpSumBunch, dpSumBunchGlobal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      memcpy(&dpSumBunch[0], &dpSumBunchGlobal[0], sizeof(dpSumBunch[0]) * 2);
#endif
      if (isSlave || !distributedBeam) {
        // Slope corrections to force momentum conservation
        slopeChange[0] = -(dpSumBunch[0] + dpSum[0]) / (qBunch / e_mks) / (me_mks * c_mks * Po);
        slopeChange[1] = -(dpSumBunch[1] + dpSum[1]) / (qBunch / e_mks) / (me_mks * c_mks * Po);
        for (ip = 0; ip < np; ip++) {
          part[ip][1] += slopeChange[0];
          part[ip][3] += slopeChange[1];
        }
      }
    }

    if (verbosity) {
#if USE_MPI
      long circuitBreakerGlobal[9];
      MPI_Allreduce((long *)&circuitBreaker, (long *)&circuitBreakerGlobal, 9, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      memcpy(circuitBreaker, circuitBreakerGlobal, sizeof(*circuitBreaker) * 9);
#endif
      int count = 0;
      for (int i = 0; i < nFunctions * nFunctions; i++) {
        if (circuitBreaker[i])
          count++;
      }
      if (count) {
        printf("Warning: circuit breaker invoked for %d of %ld functions, bunch %ld, pass %ld\n",
               count, nFunctions * nFunctions, iBunch, iPass);
        fflush(stdout);
      }
    }
  }
}

void doIonEffectsIonHistogramOutput(IONEFFECTS *ionEffects, long iBunch, long iPass, double tNow, long nBunches) {
  long iPlane;
  long binOffset, activeBins;

  if (verbosity > 30) {
    printf("Setting SDDS output of ion histogram\n");
    fflush(stdout);
  }

  if (SDDS_ionHistogramOutput && !ionEffects->ion2dDensity && ((iPass - ionEffects->startPass) * nBunches + iBunch) % ionHistogramOutputInterval == 0 && (ionHistogramOutput_sStart < 0 || ionEffects->sLocation >= ionHistogramOutput_sStart) && (ionHistogramOutput_sEnd < 0 || ionEffects->sLocation <= ionHistogramOutput_sEnd)) {
    /* output ion density histogram */
#if USE_MPI
    if (myid == 0) {
#endif
      for (iPlane = 0; iPlane < 2; iPlane++) {
        /*
          determineOffsetAndActiveBins(ionEffects->ionHistogram[iPlane], ionEffects->ionBins[iPlane],
          &binOffset, &activeBins);
        */
        binOffset = 0;
        activeBins = ionEffects->ionBins[iPlane];
        if (!SDDS_StartPage(SDDS_ionHistogramOutput, activeBins) ||
            !SDDS_SetParameters(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "Pass", iPass, "Bunch", iBunch, "t", tNow, "s", ionEffects->sLocation,
                                "qIons", ionEffects->qTotal,
                                "qIonsOutside", ionEffects->ionHistogramMissed[iPlane],
                                "fractionIonChargeOutside",
                                ionEffects->qTotal ? ionEffects->ionHistogramMissed[iPlane] / ionEffects->qTotal : -1,
                                "binSize", ionEffects->ionDelta[iPlane],
                                "binRange", ionEffects->ionRange[iPlane],
                                "nBins", ionEffects->ionBins[iPlane],
                                "nMacroIons", ionEffects->nTotalIons,
                                "nCoreMacroIons", ionEffects->nCoreIons,
                                "Plane", iPlane == 0 ? "x" : "y",
                                "fitResidual", ionEffects->xyFitResidual[iPlane],
                                isLorentzian ? "a1" : "sigma1",
                                nFunctions < 3 ? ionEffects->xyFitParameter2[iPlane][0] : ionEffects->xyFitParameter3[iPlane][0],
                                "centroid1",
                                nFunctions < 3 ? ionEffects->xyFitParameter2[iPlane][1] : ionEffects->xyFitParameter3[iPlane][1],
                                "q1",
                                nFunctions < 3 ? ionEffects->xyFitParameter2[iPlane][2] : ionEffects->xyFitParameter3[iPlane][2],
                                NULL) ||
            !SDDS_SetColumn(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME,
                            ionEffects->xyIonHistogram[iPlane] + binOffset, activeBins, "Position") ||
            !SDDS_SetColumn(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME,
                            ionEffects->ionHistogram[iPlane] + binOffset, activeBins, "Charge") ||
            !SDDS_SetColumn(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME,
                            ionEffects->ionHistogramFit[iPlane] + binOffset, activeBins, "ChargeFit")) {
          SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"Problem writing ion histogram data");
        }
        if (ionFieldMethod != ION_FIELD_GAUSSIAN) {
          if (!SDDS_SetParameters(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "qIonsFromFit", ionEffects->ionChargeFromFit[iPlane],
                                  "qIonsError", ionEffects->ionChargeFromFit[iPlane] - ionEffects->qTotal,
#if USE_MPI
                                  "nEvaluationsBest", ionEffects->nEvaluationsBest[iPlane],
                                  "nEvaluationsMin", ionEffects->nEvaluationsMin[iPlane],
                                  "nEvaluationsMax", ionEffects->nEvaluationsMax[iPlane],
#else
                                  "nEvaluations", ionEffects->nEvaluations[iPlane],
#endif
                                  isLorentzian ? "a2" : "sigma2",
                                  nFunctions == 2 ? ionEffects->xyFitParameter2[iPlane][3] : ionEffects->xyFitParameter3[iPlane][3],
                                  "centroid2",
                                  nFunctions == 2 ? ionEffects->xyFitParameter2[iPlane][4] : ionEffects->xyFitParameter3[iPlane][4],
                                  "q2",
                                  nFunctions == 2 ? ionEffects->xyFitParameter2[iPlane][5] : ionEffects->xyFitParameter3[iPlane][5],
                                  NULL)) {
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb((char *)"Problem writing ion histogram data");
          }
          if ((ionFieldMethod == ION_FIELD_TRIGAUSSIAN || ionFieldMethod == ION_FIELD_TRILORENTZIAN) &&
              !SDDS_SetParameters(SDDS_ionHistogramOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  isLorentzian ? "a3" : "sigma3", ionEffects->xyFitParameter3[iPlane][6],
                                  "centroid3", ionEffects->xyFitParameter3[iPlane][7],
                                  "q3", ionEffects->xyFitParameter3[iPlane][8],
                                  NULL)) {
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb((char *)"Problem writing ion histogram data");
          }
        }
        if (!SDDS_WritePage(SDDS_ionHistogramOutput)) {
          SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
          SDDS_Bomb((char *)"Problem writing ion histogram data");
        }
      }
#if USE_MPI
    }
#endif
  }

  if (SDDS_ion2dHistogramOutput && ionEffects->ion2dDensity && ((iPass - ionEffects->startPass) * nBunches + iBunch) % ionHistogramOutputInterval == 0 && (ionHistogramOutput_sStart < 0 || ionEffects->sLocation >= ionHistogramOutput_sStart) && (ionHistogramOutput_sEnd < 0 || ionEffects->sLocation <= ionHistogramOutput_sEnd)) {
    double dx, dy;
    long ix, iy;
    /* output 2d ion density histogram */
#if USE_MPI
    if (myid == 0) {
#endif
      if (!SDDS_StartPage(SDDS_ion2dHistogramOutput, ionEffects->n2dGridIon[0] * ionEffects->n2dGridIon[1]) ||
          !SDDS_SetParameters(SDDS_ion2dHistogramOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Pass", iPass, "Bunch", iBunch, "t", tNow, "s", ionEffects->sLocation,
                              "qIons", ionEffects->qTotal,
                              "qIonsOutside", ionEffects->ionHistogramMissed[0],
                              "fractionIonChargeOutside",
                              ionEffects->qTotal ? ionEffects->ionHistogramMissed[0] / ionEffects->qTotal : -1,
                              "nxBins", ionEffects->n2dGridIon[0],
                              "nyBins", ionEffects->n2dGridIon[1],
                              "nMacroIons", ionEffects->nTotalIons,
                              "nCoreMacroIons", ionEffects->nCoreIons,
                              NULL)) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"Problem writing ion 2d histogram data");
      }
      dx = 2 * ionEffects->poisson_span[0] / (ionEffects->n2dGridIon[0] - 1.0);
      dy = 2 * ionEffects->poisson_span[1] / (ionEffects->n2dGridIon[1] - 1.0);
      for (ix = 0; ix < ionEffects->n2dGridIon[0]; ix++) {
        for (iy = 0; iy < ionEffects->n2dGridIon[1]; iy++) {
          if (!SDDS_SetRowValues(SDDS_ion2dHistogramOutput, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                 ix + iy * ionEffects->n2dGridIon[0],
                                 "x", (float)(ix * dx - ionEffects->poisson_span[0]),
                                 "y", (float)(iy * dy - ionEffects->poisson_span[1]),
                                 "rho", (float)ionEffects->ion2dDensity[ix][iy],
                                 NULL)) {
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            SDDS_Bomb((char *)"Problem writing ion 2d histogram data");
          }
        }
      }
      if (!SDDS_WritePage(SDDS_ion2dHistogramOutput)) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        SDDS_Bomb((char *)"Problem writing ion 2d histogram data");
      }
#if USE_MPI
    }
#endif
  }
}

void flushIonEffectsSummaryOutput(IONEFFECTS *ionEffects) {
#if USE_MPI
  if (myid == 0) {
#endif
    if ((beam_output_all_locations || ionEffects == firstIonEffects) && SDDS_beamOutput && !SDDS_WritePage(SDDS_beamOutput)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    iBeamOutput = 0;
    if ((ion_output_all_locations || ionEffects == firstIonEffects) && SDDS_ionDensityOutput && !SDDS_WritePage(SDDS_ionDensityOutput)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    iIonDensityOutput = 0;
#if USE_MPI
  }
#endif
}

/* This is for performing beam-beam kicks and really doesn't belong here, but it is a convenient place. */
/* Based on M. Furman, LBL-34682 */
void ellipsoidalBeamKick(
                         double *coord,      /* particle coordinates */
                         double P0,          /* reference beta*gamma */
                         double pMass,       /* mass of particle (kg) */
                         double pCharge,     /* charge of particle (C) */
                         double centroid[2], /* centroid of opposing beam */
                         double size[2],     /* semi-axes of opposing beam */
                         double charge,      /* charge of opposing beam (C) */
                         short parabolic     /* if non-zero, density is a parabolic function of x/a and y/b */
                         ) {
  double x, y, a, b, ELx, ELy;
  double xsign = 1, ysign = 1;

  x = coord[0] - centroid[0];
  y = coord[2] - centroid[1];
  a = size[0];
  b = size[1];

  if ((sqr(x / a) + sqr(y / b)) > 1) {
    /* Electric field times length, outside the ellipsoid */
    if (!parabolic) {
      std::complex<double> EL, zbar;
      if (x < 0)
        x *= (xsign = -1);
      if (y < 0)
        y *= (ysign = -1);
      zbar = std::complex<double>(x, -y);
      EL = 4 * charge / (zbar + sqrt(zbar * zbar - a * a + b * b)) / (4 * PI * epsilon_o);
      ELx = xsign * EL.real();
      ELy = ysign * EL.imag();
    } else {
      std::complex<double> EL, zbar, c1, c2;
      if (x < 0)
        x *= (xsign = -1);
      if (y < 0)
        y *= (ysign = -1);
      zbar = std::complex<double>(x, -y);
      c1 = sqrt(zbar * zbar - a * a + b * b);
      c2 = zbar + c1;
      EL = (8 * charge / 3) * (zbar + 2.0 * c1) / (c2 * c2) / (4 * PI * epsilon_o);
      ELx = xsign * EL.real();
      ELy = ysign * EL.imag();
    }
  } else {
    /* Electric field times length, inside the ellipsoid */
    if (!parabolic) {
      ELx = 4 * charge / (a + b) * (x / a) / (4 * PI * epsilon_o);
      ELy = 4 * charge / (a + b) * (y / b) / (4 * PI * epsilon_o);
    } else {
      std::complex<double> EL, xi, wbar, zbar;
      if (x < 0)
        x *= (xsign = -1);
      if (y < 0)
        y *= (ysign = -1);
      zbar = std::complex<double>(x, -y);
      wbar = std::complex<double>(b * x / a, -a * y / b);
      xi = std::complex<double>(x / a, y / b);
      EL = 8 * charge * xi / (a + b) * (1.0 - ((2.0 * zbar + wbar) * xi) / (3.0 * (a + b))) / (4 * PI * epsilon_o);
      ELx = xsign * EL.real();
      ELy = ysign * EL.imag();
    }
  }

  double qx, qy, qz, delta, denom, p, beta, betaz, dpx, dpy;
  delta = coord[5];
  p = P0 * (1 + delta);
  beta = p / sqrt(p * p + 1);
  betaz = beta / sqrt(1 + sqr(coord[1]) + sqr(coord[3]));

  dpx = ELx * pCharge / (betaz * c_mks);
  dpy = ELy * pCharge / (betaz * c_mks);

  denom = sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
  delta = coord[5];
  qx = coord[1] * (1 + delta) / denom;
  qy = coord[3] * (1 + delta) / denom;
  qz = sqrt(sqr(1 + delta) - sqr(qx) - sqr(qy));

  qx += dpx / (pMass * c_mks * P0);
  qy += dpy / (pMass * c_mks * P0);

  delta = sqrt(qx * qx + qy * qy + qz * qz) - 1;
  coord[1] = qx / qz;
  coord[3] = qy / qz;
  coord[5] = delta;
}
