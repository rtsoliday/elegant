/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* program: sddsemitmeas
 * purpose: process elegant runs to determine emittance of beam.
 *          Also processes experimental data using elegant-computed
 *          matrices.
 * Michael Borland, 1997.
 */

#include "mdb.h"
#include "matlib.h"
#include "match_string.h"
#include "SDDS.h"
#include "scan.h"
#include "table.h"
#include <time.h>
#include <memory.h>

#define SET_ERROR_LEVEL 0
#define SET_N_ERROR_SETS 1
#define SET_SEED 2
#define SET_VERBOSITY 3
#define SET_DEVIATION_LIMIT 4
#define SET_RESOLUTION 5
#define SET_LIMIT_MODE 6
#define SET_IGNORE_PLANE 7
#define SET_UNCERTAINTY_FRAC 8
#define SET_MINIMUM_UNCERT 9
#define SET_FIND_UNCERT 10
#define SET_FIXED_UNCERT 11
#define SET_CONSTANT_WEIGHTING 12
#define SET_SIGMA_DATA 13
#define SET_PIPE 14
#define SET_ENERGYSPREAD 15
#define N_OPTIONS 16

char *option[N_OPTIONS] = {
  "errorlevel",
  "nerrorsets",
  "seed",
  "verbosity",
  "deviationlimit",
  "resolution",
  "limitmode",
  "ignoreplane",
  "uncertaintyfraction",
  "minimumuncertainty",
  "finduncertainties",
  "fixeduncertainty",
  "constantweighting",
  "sigmadata",
  "pipe",
  "energyspread",
};

#define USAGE "sddsemitmeas\n\
 [<inputfile>] [<outputfile>] [-pipe=[input][,output]]\n\
 -sigmaData=<xName>,<yName>\n\
 [-energySpread=<fractionalRMSValue>]\n\
 [-errorLevel=<valueInm>,[{gaussian,<nSigmas> | uniform}]]\n\
 [-nErrorSets=<number>]\n\
 [-limitMode={resolution | zero}[{,reject}]\n\
 [-deviationLimit=<xLevelm>,<yLevelm>]\n\
 [-resolution=<xResolutionm>,<yResolutionm>]\n\
 [-uncertaintyFraction=<xValue>,<yValue>]\n\
 [-fixedUncertainty=<xValuem>,<yValuem>]\n\
 [-findUncertainties]\n\
 [-minimumUncertainty=<xValuem>,<yValuem>]\n\
 [-constantWeighting]\n\
 [-seed=integer] [-ignorePlane={x | y}]\n\
 [-verbosity=level]\n\n\
Program by Michael Borland. (This is version 3, November 1998.)"

static char *additional_help[] = {
  USAGE,
  "\nThis program computes the emittance from simulated or experimental",
  "beamsize data, using simulated data for response matrices.",
  "inputfile is an elegant \"final\" parameters output file (SDDS format),",
  "which contains the R matrix and simulated beam sigmas as a function",
  "of some variable or variables.  It may also contain experimental data",
  "that has been obtained separately and put into the elegant output file",
  "(e.g., using sddsxref). \n"
  "-sigmaData is used to name the columns in <inputfile> that contain the",
  "    beam sizes to be used in fitting.",
  "-errorLevel and -nErrorSets allow the addition of random measurement",
  "    errors to the sigmas. -deviationLimit allows exclusion from the ",
  "    fit of bad data.  -limitMode allows you to specify what is done",
  "    with data at or below the resolution limit.",
  "-resolution allows specification of the measurement resolution,",
  "    which is subtracted in quadrature from the sigma or width values.",
  "-uncertaintyFraction specifies fractional uncertainties in beam sizes.",
  "    -fixedUncertainy specifies fixed values for same, while -findUncertainties",
  "    uses preliminary fit to estimate uncertainties.",
  "    -minimumUncertainty specifies minimum allowed values for uncertainies.",
  "    -constantWeighting specifies that all points have the same weighting",
  "    in the fit, which is not the same as having the same uncertainty for",
  "    each point.",
  "-energySpread is used if there is dispersion at the measurement point.",
  "    In this case, you must put the dispersion (etax, etay) at the measurement",
  "    point in <inputfile>.  This can be done using the elegant twiss",
  "    output plus some SDDS tools.",
  "\n*** Note: use of sddsemitproc is recommended instead of this program! ***",
  NULL};

#define N_INCREMENT 10

double solve_normal_form(MATRIX *F, MATRIX *sF, MATRIX *P, MATRIX *M, MATRIX *C, double *s2_fit);
double solve_normal_form_opt(MATRIX *F, MATRIX *sF, MATRIX *P, MATRIX *M, MATRIX *C, double dev_limit,
                             int *n_used, double *s2_fit);
void print_fit(char *filename, double *variable_data, char *variable_name,
               double *sigma2_fit, double resol, char *sigma_name, int n_pts);
double propagate_errors_for_emittance(double **Sigma, double **Covar);
void set_up_covariance_matrix(MATRIX *K, double *sigma, double *uncert, int n_configs, int equal_weights);
double estimate_uncertainty(double *uncert, MATRIX *S, MATRIX *sS, MATRIX *R, MATRIX *s2, MATRIX *K,
                            double dev_limit, int n_configs, double uncert_min, double *fit_sig2_return);
int make_tweeked_data_set(MATRIX *s2, double *sigma, double error_level, double error_sigmas, int error_type_code,
                          int n_configs, double resol, int reject_at_limit, double limit, int *n_at_resol);
long SetSigmaData(SDDS_DATASET *SDDSout, char *dataName, MATRIX *s2, char *fitName, double *fitSigSqr,
                  long configs);

#define GAUSSIAN_ERRORS 0
#define UNIFORM_ERRORS 1
#define N_ERROR_TYPES 2
char *error_type[N_ERROR_TYPES] = {
  "gaussian", "uniform"};

#define LIMIT_AT_RESOLUTION 0
#define LIMIT_AT_ZERO 1
#define N_LIMIT_OPTIONS 2
char *limit_option[N_LIMIT_OPTIONS] = {
  "resolution", "zero"};

#define MAX_N_TRIES 100

int main(
         int argc,
         char **argv) {
  SDDS_TABLE SDDSin, SDDSout;
  double *R11;                                        /* R11 matrix element for ith configuration */
  double *R12 = NULL;                                 /* R12 matrix element for ith configuration */
  double *etax = NULL, *etay = NULL;                  /* dispersion at measurement point, if provided */
  double *sigmax, *uncertx;                           /* sigma in x plane for ith configuration */
  double *R33 = NULL, *R34 = NULL, *sigmay, *uncerty; /* similar data for y plane */
  int n_configs, i_config;
  MATRIX *Rx, *Ry, *s2x, *s2y;
  MATRIX *Sx, *Sy, *sSx, *sSy, *Kx, *Ky;
  double S11_sum, S11_sum2, S12_sum, S12_sum2, S22_sum, S22_sum2;
  //double S33_sum, S33_sum2;
  double S34_sum, S34_sum2, S44_sum, S44_sum2;
  double betax, alphax, betax_sum, betax_sum2, alphax_sum, alphax_sum2;
  double betay, alphay, betay_sum, betay_sum2, alphay_sum, alphay_sum2;
  int i_variable;
  double emitx, emity;
  SCANNED_ARG *scanned;
  int i_arg, i;
  char *input, *output;
  char *variable_name;
  double *variable_data, *x_fit_sig2, *y_fit_sig2;
  int n_error_sets, seed, i_error, error_type_code;
  double error_sigmas;
  double emitx_max, emitx_min;
  double emity_max, emity_min;
  double emitx_sum, emity_sum;
  double emitx2_sum, emity2_sum;
  double error_level, x_error_level, y_error_level;
  double md_x, md_y, *dev_limit, contrib;
  // double md;
  int i_dev, n_dev_limits;
  int nx_used, nx_used_sum;
  int ny_used, ny_used_sum;
  int n_good_fits_x, n_good_fits_y;
  double x_resol, y_resol;
  double x_limit = 0.0, y_limit = 0.0;
  int limit_code, reject_at_limit;
  int n_xresol, n_yresol, ignore_x, ignore_y, error_output;
  double x_uncert_frac, y_uncert_frac;
  double x_uncert_min, y_uncert_min;
  double x_fixed_uncert, y_fixed_uncert;
  double energySpread, energySize;
  int equal_weights_x_fit = 0, equal_weights_y_fit = 0, constant_weighting;
  int find_uncert, verbosity;
  char *x_width_name, *y_width_name;
  unsigned long pipeFlags;
  char emitLabel[514], xEmitLabel[256], yEmitLabel[256];

  argc = scanargs(&scanned, argc, argv);
  if (argc < 2 || argc > (2 + N_OPTIONS)) {
    for (i = 0;; i++) {
      if (!additional_help[i])
        break;
      puts(additional_help[i]);
    }
    exit(1);
  }

  input = output = NULL;
  dev_limit = NULL;
  n_error_sets = 1;
  error_level = n_dev_limits = 0;
  seed = -1;
  variable_name = NULL;
  x_resol = y_resol = 0;
  error_type_code = UNIFORM_ERRORS;
  error_sigmas = 1;
  limit_code = LIMIT_AT_ZERO;
  reject_at_limit = 0;
  ignore_x = ignore_y = 0;
  x_uncert_frac = y_uncert_frac = 0;
  x_uncert_min = y_uncert_min = 0;
  find_uncert = constant_weighting = 0;
  x_fixed_uncert = y_fixed_uncert = verbosity = 0;
  x_width_name = "Sx";
  y_width_name = "Sy";
  energySpread = 0;
  pipeFlags = 0;

  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (scanned[i_arg].arg_type == OPTION) {
      switch (match_string(scanned[i_arg].list[0], option,
                           N_OPTIONS, 0)) {
        /* process options here */
      case SET_ERROR_LEVEL:
        if (scanned[i_arg].n_items == 2) {
          if (!sscanf(scanned[i_arg].list[1], "%lf", &error_level))
            bomb("invalid -errorLevel syntax", USAGE);
        } else if (scanned[i_arg].n_items >= 3 && scanned[i_arg].n_items <= 4) {
          if (!sscanf(scanned[i_arg].list[1], "%lf", &error_level))
            bomb("invalid -errorLevel syntax", USAGE);
          if ((error_type_code = match_string(scanned[i_arg].list[2], error_type, N_ERROR_TYPES, 0)) < 0)
            bomb("unknown error type", USAGE);
          if (scanned[i_arg].n_items == 4) {
            if (error_type_code != GAUSSIAN_ERRORS ||
                !sscanf(scanned[i_arg].list[3], "%lf", &error_sigmas) ||
                error_sigmas < 0)
              bomb("invalid -errorLevel syntax", USAGE);
          }
        } else
          bomb("invalid -errorLevel syntax", USAGE);
        break;
      case SET_N_ERROR_SETS:
        if (scanned[i_arg].n_items != 2 ||
            !sscanf(scanned[i_arg].list[1], "%d", &n_error_sets) ||
            n_error_sets < 0)
          bomb("invalid -n_error_sets syntax", USAGE);
        break;
      case SET_SEED:
        if (scanned[i_arg].n_items != 2 ||
            !sscanf(scanned[i_arg].list[1], "%d", &seed) ||
            seed < 0)
          bomb("invalid -seed syntax", USAGE);
        break;
      case SET_DEVIATION_LIMIT:
        if ((n_dev_limits = scanned[i_arg].n_items - 1) < 1)
          bomb("invalid -deviation_limit syntax", USAGE);
        dev_limit = tmalloc(sizeof(double) * n_dev_limits);
        for (i = 0; i < n_dev_limits; i++) {
          if (!sscanf(scanned[i_arg].list[i + 1], "%lf", dev_limit + i) ||
              dev_limit[i] == 0)
            bomb("invalid -deviation_limit syntax", USAGE);
        }
        break;
      case SET_RESOLUTION:
        if (scanned[i_arg].n_items != 3 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &x_resol) ||
            !sscanf(scanned[i_arg].list[2], "%lf", &y_resol) ||
            x_resol < 0 || y_resol < 0)
          bomb("invalid -resolution syntax", USAGE);
        break;
      case SET_LIMIT_MODE:
        if (scanned[i_arg].n_items < 2 || scanned[i_arg].n_items > 4 ||
            (limit_code = match_string(scanned[i_arg].list[1], limit_option,
                                       N_LIMIT_OPTIONS, 0)) < 0)
          bomb("invalid -limit_mode syntax", USAGE);
        if (scanned[i_arg].n_items == 3) {
          if (scanned[i_arg].list[2][0] == 'r')
            reject_at_limit = 1;
          else
            bomb("invalid -limit_mode syntax", USAGE);
        }
        break;
      case SET_IGNORE_PLANE:
        if (scanned[i_arg].n_items != 2 ||
            !((ignore_x = (scanned[i_arg].list[1][0] == 'x')) ||
              (ignore_y = (scanned[i_arg].list[1][0] == 'y'))))
          bomb("invalid -ignore_plane syntax", USAGE);
        break;
      case SET_UNCERTAINTY_FRAC:
        if (scanned[i_arg].n_items != 3 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &x_uncert_frac) ||
            x_uncert_frac <= 0 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &y_uncert_frac) ||
            y_uncert_frac <= 0)
          bomb("invalid -uncertainty_fraction syntax", USAGE);
        break;
      case SET_MINIMUM_UNCERT:
        if (scanned[i_arg].n_items != 3 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &x_uncert_min) ||
            x_uncert_min <= 0 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &y_uncert_min) ||
            y_uncert_min <= 0)
          bomb("invalid -minimum_uncertainty syntax", USAGE);
        break;
      case SET_FIND_UNCERT:
        find_uncert = 1;
        break;
      case SET_FIXED_UNCERT:
        if (scanned[i_arg].n_items != 3 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &x_fixed_uncert) ||
            !sscanf(scanned[i_arg].list[2], "%lf", &y_fixed_uncert) ||
            x_fixed_uncert < 0 || y_fixed_uncert < 0)
          bomb("invalid -fixed_uncertainty syntax", USAGE);
        break;
      case SET_CONSTANT_WEIGHTING:
        constant_weighting = 1;
        break;
      case SET_VERBOSITY:
        if (scanned[i_arg].n_items != 2 ||
            !sscanf(scanned[i_arg].list[1], "%d", &verbosity) ||
            verbosity < 0)
          bomb("invalid -verbosity syntax", USAGE);
        break;
      case SET_SIGMA_DATA:
        if (scanned[i_arg].n_items != 3 ||
            !strlen(x_width_name = scanned[i_arg].list[1]) ||
            !strlen(y_width_name = scanned[i_arg].list[2]))
          bomb("invalid -sigmaData syntax", USAGE);
        break;
      case SET_PIPE:
        if (!processPipeOption(scanned[i_arg].list + 1, scanned[i_arg].n_items - 1, &pipeFlags))
          SDDS_Bomb("invalid -pipe syntax");
        break;
      case SET_ENERGYSPREAD:
        if (scanned[i_arg].n_items != 2 ||
            !sscanf(scanned[i_arg].list[1], "%lf", &energySpread) ||
            energySpread < 0)
          SDDS_Bomb("invalid -energySpread syntax");
        break;
      default:
        bomb("unknown option given", USAGE);
        break;
      }
    } else {
      if (!input)
        input = scanned[i_arg].list[0];
      else if (!output)
        output = scanned[i_arg].list[0];
      else
        bomb("too many filenames given", USAGE);
    }
  }

  processFilenames("sddsemitmeas", &input, &output, pipeFlags, 0, NULL);

  if (ignore_x && ignore_y)
    bomb("can't ignore both x and y planes!", USAGE);

  if (n_error_sets && error_level) {
    x_limit = (limit_code == LIMIT_AT_ZERO ? 0.0 : x_resol);
    y_limit = (limit_code == LIMIT_AT_ZERO ? 0.0 : y_resol);
  } else if (error_level) {
    x_limit = (limit_code == LIMIT_AT_ZERO ? 0.0 : x_resol);
    y_limit = (limit_code == LIMIT_AT_ZERO ? 0.0 : y_resol);
  }

  if (seed < 0) {
    /* generate seed from system clock */
    seed = (int)time(NULL);
  }
  random_1(-seed);

  if (n_error_sets <= 0)
    n_error_sets = 1;
  if (n_dev_limits <= 0) {
    dev_limit = tmalloc(sizeof(double));
    n_dev_limits = 1;
    dev_limit[0] = 0;
  }

  error_output = 0;
  if (n_error_sets > 1 || constant_weighting || find_uncert ||
      x_uncert_frac || y_uncert_frac || x_fixed_uncert || y_fixed_uncert)
    error_output = 1;

  if (!SDDS_InitializeInput(&SDDSin, input))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  if (!SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 1, NULL, NULL, output))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  if (!ignore_x) {
    if (!SDDS_DefineSimpleParameter(&SDDSout, "ex", "$gp$rm", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S11", "m$a2$n", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S12", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S22", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "betax", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "alphax", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "xRMSDeviation", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "xGoodFits", "", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "xAverageFitPoints", "", SDDS_DOUBLE) ||
        (error_output &&
         (!SDDS_DefineSimpleParameter(&SDDSout, "exSigma", "$gp$rm", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S11Sigma", "m$a2$n", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S12Sigma", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S22Sigma", "", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "betaxSigma", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "alphaxSigma", "m", SDDS_DOUBLE))))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (!SDDS_DefineSimpleColumn(&SDDSout, "xSigmaData", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "xSigmaFit", "m", SDDS_DOUBLE)) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
  }
  if (!ignore_y) {
    if (!SDDS_DefineSimpleParameter(&SDDSout, "ey", "$gp$rm", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S33", "m$a2$n", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S34", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "S44", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "betay", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "alphay", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "yRMSDeviation", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "yGoodFits", "", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDSout, "yAverageFitPoints", "", SDDS_DOUBLE) ||
        (error_output &&
         (!SDDS_DefineSimpleParameter(&SDDSout, "eySigma", "$gp$rm", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S33Sigma", "m$a2$n", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S34Sigma", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "S44Sigma", "", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "betaySigma", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(&SDDSout, "alphaySigma", "m", SDDS_DOUBLE))))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (!SDDS_DefineSimpleColumn(&SDDSout, "ySigmaData", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "ySigmaFit", "m", SDDS_DOUBLE)) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
  }

  if (SDDS_GetColumnIndex(&SDDSin, x_width_name) < 0 ||
      SDDS_GetColumnIndex(&SDDSin, y_width_name) < 0 ||
      SDDS_GetColumnIndex(&SDDSin, "R11") < 0 ||
      SDDS_GetColumnIndex(&SDDSin, "R12") < 0 ||
      SDDS_GetColumnIndex(&SDDSin, "R33") < 0 ||
      SDDS_GetColumnIndex(&SDDSin, "R34") < 0)
    if (energySpread &&
        (SDDS_GetColumnIndex(&SDDSin, "etax") < 0 ||
         SDDS_GetColumnIndex(&SDDSin, "etay") < 0))
      SDDS_Bomb("input file missing etax or etay");

  if (variable_name) {
    if ((i_variable = SDDS_GetColumnIndex(&SDDSin, variable_name)) < 0)
      bomb("no match for variable for fit output/filtering", NULL);
  } else {
    char **column_name;
    int32_t column_names;
    if (!(column_name = SDDS_GetColumnNames(&SDDSin, &column_names)))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    for (i_variable = 0; i_variable < column_names; i_variable++)
      if (column_name[i_variable][0] == 'Q')
        break;
    if (i_variable == column_names) {
      for (i_variable = 0; i_variable < column_names; i_variable++)
        if (strchr(column_name[i_variable], 'Q'))
          break;
    }
    if (i_variable == column_names) {
      for (i_variable = 0; i_variable < column_names; i_variable++)
        if (strcmp(column_name[i_variable], "Step") == 0)
          break;
    }
    if (i_variable == column_names)
      bomb("you did not specify -variable_name, and there is no obvious choice in the input data", NULL);
    variable_name = column_name[i_variable];
  }
  if (!SDDS_TransferColumnDefinition(&SDDSout, &SDDSin, variable_name, NULL) ||
      !SDDS_DefineSimpleParameter(&SDDSout, "EmittanceLabel", "", SDDS_STRING) ||
      !SDDS_WriteLayout(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  uncertx = uncerty = x_fit_sig2 = y_fit_sig2 = NULL;
  while (SDDS_ReadTable(&SDDSin) > 0) {
    n_configs = SDDS_CountRowsOfInterest(&SDDSin);
    if (!(R11 = SDDS_GetColumn(&SDDSin, "R11")) ||
        !(R12 = SDDS_GetColumn(&SDDSin, "R12")) ||
        !(R33 = SDDS_GetColumn(&SDDSin, "R33")) ||
        !(R34 = SDDS_GetColumn(&SDDSin, "R34")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (energySpread &&
        (!(etax = SDDS_GetColumn(&SDDSin, "etax")) ||
         !(etay = SDDS_GetColumn(&SDDSin, "etay"))))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    if (!(sigmax = SDDS_GetColumn(&SDDSin, x_width_name)))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (!(sigmay = SDDS_GetColumn(&SDDSin, y_width_name)))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    if (!(variable_data = SDDS_GetColumn(&SDDSin, variable_name)))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    uncertx = SDDS_Realloc(uncertx, sizeof(double) * n_configs);
    uncerty = SDDS_Realloc(uncerty, sizeof(double) * n_configs);
    x_fit_sig2 = SDDS_Realloc(x_fit_sig2, sizeof(double) * n_configs);
    y_fit_sig2 = SDDS_Realloc(y_fit_sig2, sizeof(double) * n_configs);

    for (i_config = 0; i_config < n_configs; i_config++) {
      uncertx[i_config] = uncerty[i_config] = 0;
    }
    if (n_configs < 4)
      continue;

    if (!SDDS_StartPage(&SDDSout, n_configs) || !SDDS_CopyColumns(&SDDSout, &SDDSin))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    if (!ignore_x) {
      for (i_config = 0; i_config < n_configs; i_config++) {
        if (energySpread && etax[i_config]) {
          energySize = sqr(energySpread * etax[i_config]);
          if (energySize > sqr(sigmax[i_config]))
            SDDS_Bomb("size due to energy spread exceeds measured horizontal size");
          sigmax[i_config] = sqrt(sqr(sigmax[i_config]) - energySize);
        }
      }
      if (x_uncert_frac && x_fixed_uncert)
        bomb("can't specify uncertainty fraction and fixed uncertainty together", NULL);
      if (constant_weighting) {
        for (i_config = 0; i_config < n_configs; i_config++) {
          if (sigmax[i_config] == 0)
            bomb("can't do constant weighting--there is a sigma that is zero", NULL);
          uncertx[i_config] = 1;
        }
      } else if ((x_uncert_frac || x_fixed_uncert) && sigmax[0] != 0) {
        equal_weights_x_fit = 0;
        for (i_config = 0; i_config < n_configs; i_config++)
          uncertx[i_config] = (x_uncert_frac ? sigmax[i_config] * x_uncert_frac : x_fixed_uncert);
      } else if (uncertx[0] == 0 || sigmax[0] == 0) {
        equal_weights_x_fit = 1;
        for (i_config = 0; i_config < n_configs; i_config++)
          uncertx[i_config] = 1;
      } else {
        equal_weights_x_fit = 0;
      }
      if (x_uncert_min) {
        for (i_config = 0; i_config < n_configs; i_config++)
          if (uncertx[i_config] < x_uncert_min)
            uncertx[i_config] = x_uncert_min;
      }
    }
    if (!ignore_y) {
      for (i_config = 0; i_config < n_configs; i_config++) {
        if (energySpread && etay[i_config]) {
          energySize = sqr(energySpread * etay[i_config]);
          if (energySize > sqr(sigmay[i_config]))
            SDDS_Bomb("size due to energy spread exceeds measured vertical size");
          sigmay[i_config] = sqrt(sqr(sigmay[i_config]) - energySize);
        }
      }
      if (y_uncert_frac && y_fixed_uncert)
        bomb("can't specify both uncertainty fraction and fixed uncertainty", NULL);
      if (constant_weighting) {
        for (i_config = 0; i_config < n_configs; i_config++) {
          if (sigmay[i_config] == 0)
            bomb("can't do constant weighting--there is a sigma that is zero", NULL);
          uncerty[i_config] = 1;
        }
      } else if ((y_uncert_frac || y_fixed_uncert) && sigmay[0] != 0) {
        equal_weights_y_fit = 0;
        for (i_config = 0; i_config < n_configs; i_config++) {
          uncerty[i_config] = (y_uncert_frac ? sigmay[i_config] * y_uncert_frac : y_fixed_uncert);
        }
      } else if (uncerty[0] == 0 || sigmay[0] == 0) {
        equal_weights_y_fit = 1;
        for (i_config = 0; i_config < n_configs; i_config++)
          uncerty[i_config] = 1;
      } else {
        equal_weights_y_fit = 0;
      }
      if (y_uncert_min) {
        for (i_config = 0; i_config < n_configs; i_config++)
          if (uncerty[i_config] < y_uncert_min)
            uncerty[i_config] = y_uncert_min;
      }
    }

    m_alloc(&Rx, n_configs, 3);
    m_alloc(&Ry, n_configs, 3);
    m_alloc(&s2x, n_configs, 1);
    m_alloc(&s2y, n_configs, 1);
    m_alloc(&Sx, 3, 1);
    m_alloc(&Sy, 3, 1);
    m_alloc(&sSx, 3, 3);
    m_alloc(&sSy, 3, 3);
    m_alloc(&Kx, n_configs, n_configs);
    m_zero(Kx);
    m_alloc(&Ky, n_configs, n_configs);
    m_zero(Ky);

    for (i_config = 0; i_config < n_configs; i_config++) {
      Rx->a[i_config][0] = sqr(R11[i_config]);
      Rx->a[i_config][1] = 2 * R11[i_config] * R12[i_config];
      Rx->a[i_config][2] = sqr(R12[i_config]);
      Ry->a[i_config][0] = sqr(R33[i_config]);
      Ry->a[i_config][1] = 2 * R33[i_config] * R34[i_config];
      Ry->a[i_config][2] = sqr(R34[i_config]);
    }

    x_error_level = y_error_level = 0;

    for (i_dev = 0; i_dev < n_dev_limits; i_dev++) {
      emitx_sum = emitx2_sum = 0;
      emity_sum = emity2_sum = 0;
      emitx_max = -(emitx_min = DBL_MAX);
      emity_max = -(emity_min = DBL_MAX);
      S11_sum = S11_sum2 = S12_sum = S12_sum2 = S22_sum = S22_sum2 = 0;
      //S33_sum = S33_sum2 = 0;
      S34_sum = S34_sum2 = S44_sum = S44_sum2 = 0;
      betax_sum = betax_sum2 = alphax_sum = alphax_sum2 = 0;
      betay_sum = betay_sum2 = alphay_sum = alphay_sum2 = 0;
      md_x = md_y = nx_used_sum = ny_used_sum = 0;
      n_good_fits_x = n_good_fits_y = 0;
      n_xresol = n_yresol = 0;
      for (i_error = 0; i_error < n_error_sets; i_error++) {
        if (verbosity > 1) {
          fprintf(stderr, "Error set %d...\n", i_error);
        }
        for (i_config = 0; i_config < n_configs; i_config++) {
          s2x->a[i_config][0] = sqr(sigmax[i_config]) - sqr(x_resol);
          s2y->a[i_config][0] = sqr(sigmay[i_config]) - sqr(y_resol);
        }

        if (error_level < 0) {
          if (!ignore_x) {
            if (x_error_level == 0) {
              /* must do initial fit to find error level */
              set_up_covariance_matrix(Kx, sigmax, uncertx, n_configs, equal_weights_x_fit);
              x_error_level = estimate_uncertainty(uncertx, Sx, sSx, Rx, s2x, Kx, dev_limit[i_dev],
                                                   n_configs, x_uncert_min, x_fit_sig2);
            }
            if (!make_tweeked_data_set(s2x, sigmax, x_error_level, error_sigmas, error_type_code, n_configs,
                                       x_resol, reject_at_limit, x_limit, &n_xresol)) {
              bomb("fatal error: failed to get acceptable error set\n", NULL);
            }
          }
          if (!ignore_y) {
            if (y_error_level == 0) {
              /* must do initial fit to find error level */
              set_up_covariance_matrix(Ky, sigmay, uncerty, n_configs, equal_weights_y_fit);
              y_error_level = estimate_uncertainty(uncerty, Sy, sSy, Ry, s2y, Ky, dev_limit[i_dev],
                                                   n_configs, y_uncert_min, y_fit_sig2);
            }
            if (!make_tweeked_data_set(s2y, sigmay, y_error_level, error_sigmas, error_type_code, n_configs,
                                       y_resol, reject_at_limit, y_limit, &n_yresol)) {
              bomb("fatal error: failed to get acceptable error set\n", NULL);
            }
          }
        } else if (error_level > 0) {
          if (!ignore_x) {
            if (!make_tweeked_data_set(s2x, sigmax, error_level, error_sigmas, error_type_code, n_configs,
                                       x_resol, reject_at_limit, x_limit, &n_xresol)) {
              bomb("fatal error: failed to get acceptable error set\n", NULL);
            }
          }
          if (!ignore_y) {
            if (!make_tweeked_data_set(s2y, sigmay, error_level, error_sigmas, error_type_code, n_configs,
                                       y_resol, reject_at_limit, y_limit, &n_yresol)) {
              bomb("fatal error: failed to get acceptable error set\n", NULL);
            }
          }
        }

        if (!ignore_x)
          set_up_covariance_matrix(Kx, sigmax, uncertx, n_configs, equal_weights_x_fit);
        if (!ignore_y)
          set_up_covariance_matrix(Ky, sigmay, uncerty, n_configs, equal_weights_y_fit);

        if (!ignore_x) {
          if (find_uncert) {
            // md = estimate_uncertainty(uncertx, Sx, sSx, Rx, s2x, Kx, dev_limit[i_dev], n_configs, x_uncert_min, NULL);
          }
          if ((contrib = solve_normal_form_opt(Sx, sSx, Rx, s2x, Kx, dev_limit[i_dev], &nx_used, x_fit_sig2)) < 0) {
            fprintf(stderr, "Problem fitting for x\n");
          } else {
            md_x += contrib;
            if (nx_used && (emitx = Sx->a[0][0] * Sx->a[2][0] - sqr(Sx->a[1][0])) > 0) {
              emitx = sqrt(emitx);
              if (verbosity > 2) {
                fprintf(stderr, "Vertical emittance: %e\n", emitx);
              }
              S11_sum += Sx->a[0][0];
              S11_sum2 += sqr(Sx->a[0][0]);
              S12_sum += Sx->a[1][0];
              S12_sum2 += sqr(Sx->a[1][0]);
              S22_sum += Sx->a[2][0];
              S22_sum2 += sqr(Sx->a[2][0]);
              betax = Sx->a[0][0] / emitx;
              betax_sum += betax;
              betax_sum2 += sqr(betax);
              alphax = -Sx->a[1][0] / emitx;
              alphax_sum += alphax;
              alphax_sum2 += sqr(alphax);
              emitx_sum += emitx;
              emitx2_sum += sqr(emitx);
              if (emitx_max < emitx)
                emitx_max = emitx;
              if (emitx_min > emitx)
                emitx_min = emitx;
              nx_used_sum += nx_used;
              n_good_fits_x++;
            }
          }
        }

        if (!ignore_y) {
          if (find_uncert) {
            // md = estimate_uncertainty(uncerty, Sy, sSy, Ry, s2y, Ky, dev_limit[i_dev], n_configs, y_uncert_min, NULL);
          }
          if ((contrib = solve_normal_form_opt(Sy, sSy, Ry, s2y, Ky, dev_limit[i_dev], &ny_used, y_fit_sig2)) < 0) {
            fprintf(stderr, "Problem fitting for y\n");
          } else {
            md_y += contrib;
            if (ny_used && (emity = Sy->a[0][0] * Sy->a[2][0] - sqr(Sy->a[1][0])) > 0) {
              emity = sqrt(emity);
              if (verbosity > 2) {
                fprintf(stderr, "Vertical emittance: %e\n", emity);
              }
              //S33_sum += Sy->a[0][0];
              //S33_sum2 += sqr(Sy->a[0][0]);
              S34_sum += Sy->a[1][0];
              S34_sum2 += sqr(Sy->a[1][0]);
              S44_sum += Sy->a[2][0];
              S44_sum2 += sqr(Sy->a[2][0]);
              betay = Sy->a[0][0] / emity;
              betay_sum += betay;
              betay_sum2 += sqr(betay);
              alphay = -Sy->a[1][0] / emity;
              alphay_sum += alphay;
              alphay_sum2 += sqr(alphay);
              emity_sum += emity;
              emity2_sum += sqr(emity);
              if (emity_max < emity)
                emity_max = emity;
              if (emity_min > emity)
                emity_min = emity;
              ny_used_sum += ny_used;
              n_good_fits_y++;
            }
          }
        }
      }

      xEmitLabel[0] = yEmitLabel[1] = 0;
      if (n_error_sets > 1) {
        if (!ignore_x && n_good_fits_x) {
          if (!error_output)
            sprintf(xEmitLabel, "$ge$r$bx$n = %.3g um", emitx_sum / n_good_fits_x * 1e6);
          else
            sprintf(xEmitLabel, "$ge$r$bx$n = %.3g $sa$e %.3g um",
                    1e6 * emitx_sum / n_good_fits_x,
                    1e6 * sqrt(emitx2_sum / n_good_fits_x - sqr(emitx_sum / n_good_fits_x)));
          if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "ex", emitx_sum / n_good_fits_x, "S11", S11_sum / n_good_fits_x,
                                  "S12", S12_sum / n_good_fits_x, "S22", S22_sum / n_good_fits_x,
                                  "betax", betax_sum / n_good_fits_x, "alphax", alphax_sum / n_good_fits_x,
                                  "xRMSDeviation", md_x / n_good_fits_x, "xGoodFits", n_good_fits_x,
                                  "xAverageFitPoints", (double)nx_used_sum / n_good_fits_x,
                                  NULL) ||
              (error_output &&
               !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                   "exSigma", sqrt(emitx2_sum / n_good_fits_x - sqr(emitx_sum / n_good_fits_x)),
                                   "S11Sigma", sqrt(S11_sum2 / n_good_fits_x - sqr(S11_sum / n_good_fits_x)),
                                   "S12Sigma", sqrt(S12_sum2 / n_good_fits_x - sqr(S12_sum / n_good_fits_x)),
                                   "S22Sigma", sqrt(S22_sum2 / n_good_fits_x - sqr(S22_sum / n_good_fits_x)),
                                   "betaxSigma", sqrt(betax_sum2 / n_good_fits_x - sqr(betax_sum / n_good_fits_x)),
                                   "alphaxSigma", sqrt(alphax_sum2 / n_good_fits_x - sqr(alphax_sum / n_good_fits_x)),
                                   NULL)))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (SetSigmaData(&SDDSout, "xSigmaData", s2x, "xSigmaFit", x_fit_sig2, n_configs))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        if (!ignore_y && n_good_fits_y) {
          if (!error_output)
            sprintf(yEmitLabel, "$ge$r$by$n = %.3g um", emity_sum / n_good_fits_y * 1e6);
          else
            sprintf(yEmitLabel, "$ge$r$by$n = %.3g $sa$e %.3g um",
                    1e6 * emity_sum / n_good_fits_y,
                    1e6 * sqrt(emity2_sum / n_good_fits_y - sqr(emity_sum / n_good_fits_y)));
          if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "ey", emity_sum / n_good_fits_y, "S22", S22_sum / n_good_fits_y,
                                  "S34", S34_sum / n_good_fits_y, "S44", S44_sum / n_good_fits_y,
                                  "betay", betay_sum / n_good_fits_y, "alphay", alphay_sum / n_good_fits_y,
                                  "yRMSDeviation", md_y / n_good_fits_y, "yGoodFits", n_good_fits_y,
                                  "yAverageFitPoints", (double)ny_used_sum / n_good_fits_y,
                                  NULL) ||
              (error_output &&
               !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                   "eySigma", sqrt(emity2_sum / n_good_fits_y - sqr(emity_sum / n_good_fits_y)),
                                   "S22Sigma", sqrt(S22_sum2 / n_good_fits_y - sqr(S22_sum / n_good_fits_y)),
                                   "S34Sigma", sqrt(S34_sum2 / n_good_fits_y - sqr(S34_sum / n_good_fits_y)),
                                   "S44Sigma", sqrt(S44_sum2 / n_good_fits_y - sqr(S44_sum / n_good_fits_y)),
                                   "betaySigma", sqrt(betay_sum2 / n_good_fits_y - sqr(betay_sum / n_good_fits_y)),
                                   "alphaySigma", sqrt(alphay_sum2 / n_good_fits_y - sqr(alphay_sum / n_good_fits_y)),
                                   NULL)))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (!SetSigmaData(&SDDSout, "ySigmaData", s2y, "ySigmaFit", y_fit_sig2, n_configs))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      } else {
        if (!ignore_x) {
          double S11, S12, S22, sS11, sS12, sS22, s_emitx;
          S11 = Sx->a[0][0];
          sS11 = sqrt(sSx->a[0][0]);
          S12 = Sx->a[1][0];
          sS12 = sqrt(sSx->a[1][1]);
          S22 = Sx->a[2][0];
          sS22 = sqrt(sSx->a[2][2]);
          if ((emitx = sqrt(S11 * S22 - sqr(S12))))
            s_emitx = propagate_errors_for_emittance(Sx->a, sSx->a);
          else
            s_emitx = 0;
          sprintf(xEmitLabel, "$ge$r$bx$n = %.3g $sa$e %.3g um",
                  1e6 * emitx_sum / n_good_fits_x,
                  1e6 * s_emitx);
          if (S11 <= 0)
            S11 = 1e-100;
          if (S22 <= 0)
            S22 = 1e-100;
          if (emitx <= 0)
            emitx = 1e-100;
          if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "ex", emitx, "S11", S11, "S12", S12, "S22", S22,
                                  "betax", S11 / emitx, "alphax", -S12 / emitx,
                                  "xRMSDeviation", md_x, "xGoodFits", 1,
                                  "xAverageFitPoints", (double)nx_used,
                                  NULL) ||
              (error_output &&
               !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                   "exSigma", s_emitx, "S11Sigma", sS11, "S12Sigma", sS12,
                                   "S22Sigma", sS22,
                                   "betaxSigma", (S11 / emitx) * sqrt(sqr(s_emitx / emitx) + sqr(sS11 / S11)),
                                   "alphaxSigma", (S12 / emitx) * sqrt(sqr(s_emitx / emitx) + sqr(sS12 / S12)),
                                   NULL)))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (!SetSigmaData(&SDDSout, "xSigmaData", s2x, "xSigmaFit", x_fit_sig2, n_configs))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        if (!ignore_y) {
          double S33, S34, S44, sS33, sS34, sS44, s_emity;
          S33 = Sy->a[0][0];
          sS33 = sqrt(sSy->a[0][0]);
          S34 = Sy->a[1][0];
          sS34 = sqrt(sSy->a[1][1]);
          S44 = Sy->a[2][0];
          sS44 = sqrt(sSy->a[2][2]);
          if ((emity = sqrt(S33 * S44 - sqr(S34))))
            s_emity = propagate_errors_for_emittance(Sy->a, sSy->a);
          else
            s_emity = 0;
          sprintf(yEmitLabel, "$ge$r$by$n = %.3g $sa$e %.3g um",
                  1e6 * emity_sum / n_good_fits_y,
                  1e6 * s_emity);
          if (S33 <= 0)
            S33 = 1e-100;
          if (S44 <= 0)
            S44 = 1e-100;
          if (emity <= 0)
            emity = 1e-100;
          if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "ey", emity, "S33", S33, "S34", S34, "S44", S44,
                                  "betay", S33 / emity, "alphay", -S34 / emity,
                                  "yRMSDeviation", md_y, "yGoodFits", 1,
                                  "yAverageFitPoints", (double)ny_used,
                                  NULL) ||
              (error_output &&
               !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                   "eySigma", s_emity, "S33Sigma", sS33, "S34Sigma", sS34,
                                   "S44Sigma", sS44,
                                   "betaySigma", (S33 / emity) * sqrt(sqr(s_emity / emity) + sqr(sS33 / S33)),
                                   "alphaySigma", (S34 / emity) * sqrt(sqr(s_emity / emity) + sqr(sS34 / S34)),
                                   NULL)))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          if (!SetSigmaData(&SDDSout, "ySigmaData", s2y, "ySigmaFit", y_fit_sig2, n_configs))
            SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
      snprintf(emitLabel, sizeof(emitLabel), "%s %s", xEmitLabel, yEmitLabel);
      if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "EmittanceLabel", emitLabel, NULL) ||
          !SDDS_WritePage(&SDDSout))
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }

    m_free(&Rx);
    m_free(&Ry);
    m_free(&s2x);
    m_free(&s2y);
    m_free(&Sx);
    m_free(&Sy);
    m_free(&sSx);
    m_free(&sSy);
    m_free(&Kx);
    m_free(&Ky);
  }

  if (!SDDS_Terminate(&SDDSin) || !SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

  return (0);
}

void print_fit(
               char *filename, double *variable_data, char *variable_name,
               double *sigma2_fit, double resol, char *sigma_name,
               int n_pts) {
  int i, n_good;
  FILE *fp;

  for (i = n_good = 0; i < n_pts; i++)
    if (sigma2_fit[i] >= 0)
      n_good++;

  fp = fopen_e(filename, "w", 0);
  fprintf(fp, "%s\n%s\nEMITMEAS fit output\n\n%d\n", variable_name, sigma_name,
          n_good);

  for (i = 0; i < n_pts; i++)
    if (sigma2_fit[i] >= 0)
      fprintf(fp, "%e %e\n", variable_data[i], sqrt(sigma2_fit[i] + sqr(resol)));

  fclose(fp);
}

double solve_normal_form_opt(
                             MATRIX *F,  /* Mx1 matrix of Fit coefficients (returned) */
                             MATRIX *sF, /* MxM matrix of errors in Fit coefficients (returned) */
                             MATRIX *P,  /* NxM matrix of Parameters of fit (provided) */
                             MATRIX *M,  /* Nx1 column matrix of Measured quantities (provided) */
                             /* M = P.F is the equation being solved */
                             MATRIX *K,        /* NxN inverse covariance matrix for Measured quantities.
                                                  K[i][j] = delta(i,j)/uncert[i]^2 */
                             double dev_limit, /* limit on deviation for any point used in final fit */
                             int *n_used,      /* number of points used in final fit */
                             double *s2_fit    /* sigma^2 from fit (returned) */
                             ) {
  int i, j, i_good;
  double rms_error, error;
  double *s2_fit2;
  int *index, n_good, *good;
  MATRIX *Pc, *Mc, *Kc;

  rms_error = solve_normal_form(F, sF, P, M, K, s2_fit);

  if (dev_limit == 0) {
    *n_used = M->n;
    return (rms_error);
  }

  good = tmalloc(sizeof(*good) * M->n);
  if (dev_limit < 0)
    dev_limit = -dev_limit * rms_error;
  for (i = n_good = 0; i < M->n; i++) {
    error = fabs(sqrt(s2_fit[i]) - sqrt(M->a[i][0]));
    if (dev_limit > error) {
      n_good++;
      good[i] = 1;
    } else {
      s2_fit[i] = -1; /* used to mark excluded points for caller */
      good[i] = 0;
    }
  }
  if (n_good == 0) {
    *n_used = 0;
    free(good);
    return (0.0);
  }

  m_alloc(&Pc, n_good, P->m);
  m_alloc(&Mc, n_good, 1);
  m_alloc(&Kc, n_good, n_good);
  m_zero(Kc);
  index = tmalloc(sizeof(int) * n_good);
  s2_fit2 = tmalloc(sizeof(*s2_fit2) * n_good);

  for (i = i_good = 0; i < M->n; i++) {
    if (good[i]) {
      index[i_good] = i;
      for (j = 0; j < P->m; j++)
        Pc->a[i_good][j] = P->a[i][j];
      for (j = 0; j < Mc->m; j++)
        Mc->a[i_good][j] = M->a[i][j];
      Kc->a[i_good][i_good] = K->a[i][i];
      i_good++;
    }
  }

  *n_used = n_good;
  rms_error = solve_normal_form(F, sF, Pc, Mc, Kc, s2_fit2);
  for (i = 0; i < n_good; i++)
    s2_fit[index[i]] = s2_fit2[i];

  free(good);
  free(s2_fit2);
  free(index);
  m_free(&Pc);
  m_free(&Mc);
  m_free(&Kc);

  return (rms_error);
}

double solve_normal_form(
                         MATRIX *F,  /* Mx1 matrix of Fit coefficients (returned) */
                         MATRIX *sF, /* MxM covariance matrix for Fit coefficients (returned) */
                         MATRIX *P,  /* NxM matrix of Parameters of fit (provided) */
                         MATRIX *M,  /* Nx1 column matrix of Measured sigma^2 (provided) */
                         MATRIX *K,  /* NxN inverse covariance matrix for Measured quantities (provided) .
                                        K[i][j] = delta(i,j)/uncert[i]^2 */
                         /* M = P.F is the equation being solved                          */
                         /* the solution is F = inverse(transpose(P) K P) tranpose(P) K M = T.M */
                         /*           and  sF = T.K.transpose(T) */
                         double *s2_fit /* sigma^2 from fit (returned) */
                         ) {
  int n, m, i;
  MATRIX *Pt, *Pt_K, *Pt_K_P, *Inv_Pt_K_P, *Inv_PtKP_PtK;
  MATRIX *Mp, *Tt, *TC, *C;
  double error, rms_error;

  n = M->n; /* n is the number of experimental measurements */
  m = P->m; /* m is 3 */
  m_alloc(&Pt, m, n);
  m_alloc(&Pt_K, m, n);
  m_alloc(&Pt_K_P, m, m);
  m_alloc(&Inv_Pt_K_P, m, m);
  m_alloc(&Inv_PtKP_PtK, m, n);
  m_alloc(&Mp, M->n, M->m);
  m_alloc(&Tt, n, m);
  m_alloc(&TC, m, n);
  m_alloc(&C, K->n, K->m);

  /* find the fit */
  if (!m_trans(Pt, P)) {
    fprintf(stderr, "matrix error--call was: m_trans(Pt, P)");
    return -1;
  }
  if (!m_mult(Pt_K, Pt, K)) {
    fprintf(stderr, "matrix error--call was: m_mult(Pt_K, Pt, K)");
    return -1;
  }
  if (!m_mult(Pt_K_P, Pt_K, P)) {
    fprintf(stderr, "matrix error--call was: m_mult(Pt_K_P, Pt_K, P)");
    return -1;
  }
  if (!m_invert(Inv_Pt_K_P, Pt_K_P)) {
    fprintf(stderr, "matrix error--call was: m_invert(Inv_Pt_K_P, Pt_K_P)");
    return -1;
  }
  if (!m_mult(Inv_PtKP_PtK, Inv_Pt_K_P, Pt_K)) {
    fprintf(stderr, "matrix error--call was: m_mult(Inv_PtKP_PtK, Inv_Pt_K_P, Pt_K)");
    return -1;
  }
  if (!m_mult(F, Inv_PtKP_PtK, M)) {
    fprintf(stderr, "matrix error--call was: m_mult(F, Inv_PtKP_PtK, M)");
    return -1;
  }
  m_zero(sF);
  if (m_invert(C, K)) {
    if (!m_trans(Tt, Inv_PtKP_PtK)) {
      fprintf(stderr, "matrix error--call was: m_trans(Tt, Inv_PtKP_PtK)");
      return -1;
    }
    if (!m_mult(TC, Inv_PtKP_PtK, C)) {
      fprintf(stderr, "matrix error--call was: m_mult(TC, Inv_PtKP_PtK, C)");
      return -1;
    }
    if (!m_mult(sF, TC, Tt)) {
      fprintf(stderr, "matrix error--call was: m_mult(sF, TC, Tt)");
      return -1;
    }
  }

  /* evaluate the fit */
  if (!m_mult(Mp, P, F)) {
    fprintf(stderr, "matrix error--call was: m_mult(Mp, P, F)");
    return -1;
  }
  for (i = rms_error = 0; i < Mp->n; i++) {
    if ((s2_fit[i] = Mp->a[i][0]) < 0) {
      s2_fit[i] = Mp->a[i][0] = 0;
      fprintf(stderr, "bad fit--negative sigma^2!\n");
    }
    error = sqrt(Mp->a[i][0]) - sqrt(M->a[i][0]);
    rms_error += sqr(error);
  }
  if (Mp->n > 3)
    rms_error = sqrt(rms_error / (Mp->n - 3));

  m_free(&Pt);
  m_free(&Pt_K);
  m_free(&Pt_K_P);
  m_free(&Inv_Pt_K_P);
  m_free(&Inv_PtKP_PtK);
  m_free(&Mp);
  m_free(&Tt);
  m_free(&TC);
  m_free(&C);

  return (rms_error);
}

double propagate_errors_for_emittance(double **Sigma, double **Covar) {
  double S11, S12, S22, emit;
  double sum;

  S11 = Sigma[0][0];
  S12 = Sigma[1][0];
  S22 = Sigma[2][0];
  if (!(emit = sqrt(S11 * S22 - sqr(S12))))
    return (-1.0);
  sum = 0;
  /* diagonal terms */
  sum += (sqr(S22) * Covar[0][0] / 4 + sqr(S11) * Covar[2][2] / 4 +
          sqr(S12) * Covar[1][1]) /
    sqr(emit);
  /* off-diagonal terms */
  sum += 2 * (-S12 * S22 / 2 * Covar[0][1] + S11 * S22 / 4 * Covar[0][2] + -S11 * S12 / 2 * Covar[1][2]) / sqr(emit);
  return (sqrt(sum));
}

int make_tweeked_data_set(MATRIX *s2, double *sigma, double error_level, double error_sigmas, int error_type_code,
                          int n_configs, double resol, int reject_at_limit, double limit, int *n_at_resol) {
  int i_config, n_tries;
  double tweek;

  for (i_config = n_tries = 0; i_config < n_configs && n_tries < MAX_N_TRIES; i_config++) {
    tweek = (error_type_code == UNIFORM_ERRORS ? 2 * error_level * (random_1(1) - .5) : gauss_rn_lim(0.0, error_level, error_sigmas, random_1));
    s2->a[i_config][0] = sqr(sigma[i_config] + tweek) - sqr(resol);
    if (s2->a[i_config][0] < sqr(limit)) {
      if (reject_at_limit) {
        n_tries++;
        i_config--;
        continue;
      } else {
        *n_at_resol += 1;
        s2->a[i_config][0] = sqr(limit);
      }
    }
  }
  if (n_tries == MAX_N_TRIES)
    return (0);
  return (1);
}

void set_up_covariance_matrix(MATRIX *K, double *sigma, double *uncert, int n_configs, int equal_weights)
/* actually, K is the inverse of the covariance matrix */
{
  int i_config;

  for (i_config = 0; i_config < n_configs; i_config++) {
    if (sigma[i_config] == 0 || uncert[i_config] == 0 || equal_weights)
      K->a[i_config][i_config] = 1;
    else
      K->a[i_config][i_config] = 1. / sqr(2 * sigma[i_config] * uncert[i_config]);
  }
}

double estimate_uncertainty(double *uncert, MATRIX *S, MATRIX *sS, MATRIX *R, MATRIX *s2,
                            MATRIX *K, double dev_limit, int n_configs, double uncert_min, double *fit_sig2_return) {
  double md, *fit_sig2;
  int n_used, i_config;

  if (fit_sig2_return)
    fit_sig2 = fit_sig2_return;
  else
    fit_sig2 = tmalloc(sizeof(*fit_sig2) * n_configs);

  /* find initial fit with supplied covariance matrix */
  md = solve_normal_form_opt(S, sS, R, s2, K, 0.0, &n_used, fit_sig2);
  if (!md || !n_used)
    bomb("unable to find initial fit (1)", NULL);

  /* calculate new covariance matrix */
  for (i_config = 0; i_config < n_configs; i_config++)
    K->a[i_config][i_config] = 1. / sqr(2 * md) / s2->a[i_config][0];

  /* do second fit, excluding points that lie to far out */
  md = solve_normal_form_opt(S, sS, R, s2, K, dev_limit, &n_used, fit_sig2);
  if (!md || !n_used)
    bomb("unable to find initial fit (2)", NULL);

  /* calculate new covariance matrix */
  if (uncert_min && md < uncert_min)
    md = uncert_min;
  for (i_config = 0; i_config < n_configs; i_config++) {
    K->a[i_config][i_config] = 1. / sqr(2 * md) / s2->a[i_config][0];
    uncert[i_config] = md;
  }

  if (!fit_sig2_return)
    free(fit_sig2);

  return (md);
}

long SetSigmaData(SDDS_DATASET *SDDSout, char *dataName, MATRIX *s2,
                  char *fitName, double *fitSigSqr, long configs) {
  static double *buffer = NULL;
  long i;

  if (!(buffer = SDDS_Realloc(buffer, configs * sizeof(*buffer))))
    return 0;

  for (i = 0; i < configs; i++) {
    if (s2->a[i][0] < 0)
      buffer[i] = 0;
    else
      buffer[i] = sqrt(s2->a[i][0]);
  }
  if (!SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, buffer, configs, dataName))
    return 0;

  for (i = 0; i < configs; i++) {
    if (fitSigSqr[i] < 0)
      buffer[i] = 0;
    else
      buffer[i] = sqrt(fitSigSqr[i]);
  }
  if (!SDDS_SetColumn(SDDSout, SDDS_SET_BY_NAME, buffer, configs, fitName))
    return 0;

  return 1;
}
