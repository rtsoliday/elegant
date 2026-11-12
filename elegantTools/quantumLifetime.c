/*************************************************************************
 * Copyright (c) 2024 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
 *************************************************************************/

#include <stdio.h>
#include "mdb.h"
#include "scan.h"
#include "match_string.h"
#include "SDDS.h"
#include "constants.h"

#define CLO_PLANE 0
#define CLO_SCAN_APERTURE 1
#define CLO_ONE_DIMENSIONAL 2
#define CLO_EMITTANCE_RATIO 3
#define CLO_VERBOSE 4
#define CLO_NOWARNING 5
#define CLO_IGNORE_DISPERSION 6
#define CLO_LIFETIME_LIMIT 7
#define N_OPTIONS 8

char *option[N_OPTIONS] = {
  "plane", "scanAperture", "oneDimensional", "emittanceRatio", "verbose", "nowarning", "ignoreDispersion",
  "lifetimeLimit"};

static char *USAGE = "quantumLifetime <twissFile> <outputFile> [-plane={x|y}]\n\
-emittanceRatio=<value> -scanAperture=location=<elementName>,begin=<meters>,end=<meters>,points=<number>\n\
[-oneDimensional [-ignoreDispersion]] [-lifetimeLimit=<valueInDays>(7)] [-nowarning] [-verbose]\n\n\
twissFile         elegant twiss_output output file with radiation integral computation included.\n\
outputFile        Output file containing lifetime vs aperture.\n\
-plane            Specify plane of calculations and aperture scan. Default: x.\n\
-emittanceRatio   Ratio of vertical to horizontal emittance. Used to partition natural emittance.\n\
-scanAperture     Specify location, range, and number of points for aperture scan.\n\
-oneDimensional   Perform a one-dimensional calculation.\n\
-ignoreDispersion Ignore dispersion for one-dimensional calculation.\n\
-lifetimeLimit    Lifetime values in excess of this will not appear in the output file.\n\
                  Set to 0 to disable.\n\
-nowarning        Suppress warnings.\n\
-verbose          Provide informational printouts.\n\n\
Program by M. Borland, based on A. Chao, PAC 1977, p. 1885.\n";

#define X_PLANE_GIVEN 0x01UL
#define Y_PLANE_GIVEN 0x02UL

#define SCAN_LOCATION_GIVEN 0x001UL
#define SCAN_BEGIN_GIVEN 0x002UL
#define SCAN_END_GIVEN 0x004UL
#define SCAN_POINTS_GIVEN 0x008UL

#ifdef DEBUG
FILE *fpdetail = NULL;
#endif

double x, y, n, alphax, alphaDelta;
long nCalls = 0;
double integrand(double u) {
  double value;
  value = (u + x / y) * (1 / y - x * u) * ((alphax + alphaDelta * x * x) / y + (alphaDelta - alphax) * x * u) * exp(-sqr(n * u * y) / 2);
#ifdef DEBUG
  fprintf(fpdetail, "%21.15e %21.15e\n", u, value);
#endif
  nCalls++;
  return value;
}

int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  char *twissFile = NULL, *outputFile = NULL;
  SDDS_DATASET SDDStwi, SDDSout;
  short oneDimensional = 0, verbose = 0, noWarning = 0, ignoreDispersion = 0;
  char *scanLocation = NULL;
  double scanBegin = 0, scanEnd = 0, dA;
  long scanPoints = 0, iPoint, iRow, i;
  char **elementName = NULL;
  double emittance, ex0, emittanceRatio, *beta = NULL, *eta = NULL, taux, tauy, tauDelta, lifetimeLimit = 7 * 24 * 3600;
  long elementIndex, elementRows;
  unsigned long planeFlags = X_PLANE_GIVEN, scanFlags = 0;
  double A, Sx, etaAbs, Sdelta0;
  double ST, tau, integral;
  long skipSmall = 0, skipLong = 0;

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&scanned, argc, argv);
  if (argc == 1)
    bomb(NULL, USAGE);

  for (i = 1; i < argc; i++) {
    if (scanned[i].arg_type == OPTION) {
      delete_chars(scanned[i].list[0], "_");
      switch (match_string(scanned[i].list[0], option, N_OPTIONS, UNIQUE_MATCH)) {
      case CLO_EMITTANCE_RATIO:
        if (scanned[i].n_items != 2)
          bomb("invalid -emittanceRatio syntax", "-emittanceRatio=<value>[0,1]");
        if (!get_double(&emittanceRatio, scanned[i].list[1]) || emittanceRatio < 0 || emittanceRatio > 1)
          bomb("invalid -emittanceRatio value", "-emittanceRatio=<value>[0,1]");
        break;
      case CLO_LIFETIME_LIMIT:
        if (scanned[i].n_items != 2)
          bomb("invalid -lifetimeLimit syntax", "-lifetimeLimit=<valueInHours>");
        if (!get_double(&lifetimeLimit, scanned[i].list[1]) || lifetimeLimit < 0)
          bomb("invalid -lifetimeLimit syntax", "-lifetimeLimit=<valueInHours>");
        /* convert to seconds */
        lifetimeLimit *= 3600 * 24;
        break;
      case CLO_ONE_DIMENSIONAL:
        oneDimensional = 1;
        break;
      case CLO_IGNORE_DISPERSION:
        ignoreDispersion = 1;
        break;
      case CLO_VERBOSE:
        verbose = 1;
        break;
      case CLO_NOWARNING:
        noWarning = 1;
        break;
      case CLO_PLANE:
        if (scanned[i].n_items != 2)
          bomb("invalid -plane syntax", "-plane={x|y}");
        scanned[i].n_items -= 1;
        planeFlags = 0;
        if (!scanItemList(&planeFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "x", -1, NULL, 0, X_PLANE_GIVEN,
                          "y", -1, NULL, 0, Y_PLANE_GIVEN,
                          NULL) ||
            bitsSet(planeFlags) != 1)
          bomb("invalid -plane syntax", "-plane={x|y}");
        break;
      case CLO_SCAN_APERTURE:
        if (scanned[i].n_items != 5)
          bomb("invalid -scanAperture syntax", "-scanAperture=location=<elementName>,begin=<meters>,end=<meters>,points=<number>");
        scanned[i].n_items -= 1;
        scanFlags = 0;
        if (!scanItemList(&scanFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "location", SDDS_STRING, &scanLocation, 1, SCAN_LOCATION_GIVEN,
                          "begin", SDDS_DOUBLE, &scanBegin, 1, SCAN_BEGIN_GIVEN,
                          "end", SDDS_DOUBLE, &scanEnd, 1, SCAN_END_GIVEN,
                          "points", SDDS_LONG, &scanPoints, 1, SCAN_POINTS_GIVEN,
                          NULL) ||
            bitsSet(scanFlags) != 4)
          bomb("invalid -scanAperture syntax", "-scanAperture=location=<elementName>,begin=<meters>,end=<meters>,points=<number>");
        if (scanPoints < 1)
          bomb("invalid -scanAperture syntax", "points must be nonzero");
        break;
      default:
        fprintf(stderr, "unknown option \"%s\" given\n", scanned[i].list[0]);
        exit(1);
        break;
      }
    } else {
      if (!twissFile)
        twissFile = scanned[i].list[0];
      else if (!outputFile)
        outputFile = scanned[i].list[0];
      else
        bomb("too many filenames given", NULL);
    }
  }
  if (!oneDimensional)
    ignoreDispersion = 0;

  if (!SDDS_InitializeInput(&SDDStwi, twissFile) || !SDDS_ReadPage(&SDDStwi))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if ((elementRows = SDDS_RowCount(&SDDStwi)) <= 0)
    SDDS_Bomb("too little data in input file");
  if (!(elementName = (char **)SDDS_GetColumn(&SDDStwi, "ElementName")) ||
      !(beta = (double *)SDDS_GetColumnInDoubles(&SDDStwi, planeFlags & X_PLANE_GIVEN ? "betax" : "betay")) ||
      !(eta = (double *)SDDS_GetColumnInDoubles(&SDDStwi, planeFlags & X_PLANE_GIVEN ? "etax" : "etay")) ||
      !SDDS_GetParameterAsDouble(&SDDStwi, "ex0", &ex0) ||
      !SDDS_GetParameterAsDouble(&SDDStwi, "Sdelta0", &Sdelta0) ||
      !SDDS_GetParameterAsDouble(&SDDStwi, "taux", &taux) ||
      !SDDS_GetParameterAsDouble(&SDDStwi, "tauy", &tauy) ||
      !SDDS_GetParameterAsDouble(&SDDStwi, "taudelta", &tauDelta))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  for (elementIndex = 0; elementIndex < elementRows; elementIndex++) {
    if (strcmp(elementName[elementIndex], scanLocation) == 0)
      break;
  }
  if (elementIndex == elementRows)
    SDDS_Bomb("no match for named element");

  if (!SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 1, "Quantum lifetime calculation",
                             "Quantum lifetime calculation", outputFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (!SDDS_TransferParameterDefinition(&SDDSout, &SDDStwi, "ex0", NULL) ||
      !SDDS_TransferParameterDefinition(&SDDSout, &SDDStwi, "Sdelta0", NULL) ||
      !SDDS_TransferParameterDefinition(&SDDSout, &SDDStwi, "taux", NULL) ||
      !SDDS_TransferParameterDefinition(&SDDSout, &SDDStwi, "tauy", NULL) ||
      !SDDS_TransferParameterDefinition(&SDDSout, &SDDStwi, "taudelta", NULL) ||
      !SDDS_Terminate(&SDDStwi))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (0 > SDDS_DefineParameter(&SDDSout, "emittanceRatio", NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "emittance", "$ge$r", "m", NULL, NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "beta", "$gb$r", "m", "Beta function", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "eta", "$gc$r", "m", "Dispersion function", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "SigmaM", "$gs$r$bM$n", "m", "Mono-energetic beam size", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "SigmaT", "$gs$r$bT$n", "m", "Total beam size", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "ElementName", NULL, NULL, "Name of element at which aperture is scanned.",
                               NULL, SDDS_STRING, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "plane", NULL, NULL, "Plane in which aperture is scanned.",
                               NULL, SDDS_STRING, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "dispersionIncluded", NULL, NULL, "Is dispersion considered in the calculation?",
                               NULL, SDDS_STRING, NULL) ||
      0 > SDDS_DefineParameter(&SDDSout, "oneDimensional", NULL, NULL, "Is a one-dimensional formula used?",
                               NULL, SDDS_STRING, NULL) ||
      0 > SDDS_DefineColumn(&SDDSout, "aperture", NULL, "m", "Value of scanned aperture", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&SDDSout, "quantumLifetime", NULL, "s", "Value of quantum lifetime", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&SDDSout, "nEvaluations", NULL, NULL, "Number of function evaluations", NULL, SDDS_LONG, 0))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (planeFlags & X_PLANE_GIVEN)
    emittance = ex0 / (1 + emittanceRatio * taux / tauy);
  else
    emittance = emittanceRatio * ex0 / (1 + emittanceRatio * taux / tauy);

  etaAbs = fabs(eta[elementIndex]);
  Sx = sqrt(emittance * beta[elementIndex]);
  ST = sqrt(sqr(Sx) + sqr(etaAbs * Sdelta0));
  if (!SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, scanPoints) ||
      !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ex0", ex0, "Sdelta0", Sdelta0, "taux", taux, "tauy", tauy, "taudelta", tauDelta,
                          "emittanceRatio", emittanceRatio, "emittance", emittance,
                          "ElementName", scanLocation, "plane", planeFlags & X_PLANE_GIVEN ? "x" : "y",
                          "oneDimensional", oneDimensional ? "y" : "n",
                          "dispersionIncluded", ignoreDispersion ? "n" : "y",
                          "SigmaM", Sx, "SigmaT", ST,
                          "beta", beta[elementIndex], "eta", eta[elementIndex],
                          NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  tau = planeFlags & X_PLANE_GIVEN ? taux : tauy;
  alphax = 1 / tau;
  alphaDelta = 1 / tauDelta;
  x = etaAbs * Sdelta0 / Sx;
  y = sqrt(1 + x * x);

#ifdef DEBUG
  fpdetail = fopen("qlDetail.sdds", "w");
  fprintf(fpdetail, "SDDS1\n&column name=u type=double &end\n&column name=K type=double &end\n");
  fprintf(fpdetail, "&parameter name=A type=double &end\n");
  fprintf(fpdetail, "&parameter name=n type=double &end\n");
  fprintf(fpdetail, "&parameter name=C type=double &end\n");
  fprintf(fpdetail, "&data mode=ascii no_row_counts=1 &end\n");
#endif

  dA = (scanEnd - scanBegin) / (scanPoints == 1 ? 1 : scanPoints - 1);
  for (iPoint = iRow = 0; iPoint < scanPoints; iPoint++) {
    double tauQ;
    A = scanBegin + iPoint * dA;
    if (oneDimensional) {
      double sum = 0, term, product, xi;
      long k;
      k = 1;
      if (ignoreDispersion) {
        if (A / Sx < 3) {
          skipSmall++;
          continue;
        }
        xi = sqr(A / Sx) / 2;
      } else {
        if (A / ST < 3) {
          skipSmall++;
          continue;
        }
        xi = sqr(A / ST) / 2;
      }
      if (verbose) {
        printf("Working on A = %le (x = %le) ignoring dispersion\n", A, x);
        fflush(stdout);
      }
      product = 1;
      while (1) {
        product *= xi / k; /* accumulates x^k/k! */
        term = product / k;
        sum += term;
        if (term / sum < 1e-6)
          break;
        k++;
      }
      tauQ = tau / 2 * sum;
    } else {
      n = A / ST;
      if (n < 3) {
        skipSmall++;
        continue;
      }
      if (verbose) {
        printf("Working on A = %le (n = %le) using two-dimensional method\n", A, n);
        fflush(stdout);
      }
#ifdef DEBUG
      fprintf(fpdetail, "%le\n%le\n%le\n", A, n, ipow(n, 4) * exp(-sqr(n) / 2));
#endif
      integral = 0;
      gaussianQuadrature(integrand, -x / y, 1 / (x * y), 4, 1e-4, &integral);
      tauQ = 1 / (ipow(n, 4) * exp(-sqr(n) / 2) * integral);
    }
    if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iRow++,
                           "aperture", A, "quantumLifetime", tauQ, "nEvaluations", nCalls, NULL)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    nCalls = 0;
#ifdef DEBUG
    fflush(fpdetail);
#endif
    if (lifetimeLimit > 0 && tauQ > lifetimeLimit)
      break;
  }
#ifdef DEBUG
  fclose(fpdetail);
#endif
  skipLong = scanPoints - iPoint;
  if (!noWarning) {
    if (skipSmall)
      fprintf(stderr, "Warning: %ld points skipped because approximation is invalid for aperture too small.\n",
              skipSmall);
    if (skipLong)
      fprintf(stderr, "Warning: %ld points skipped because lifetime exceeds %lg days.\n", skipLong,
              lifetimeLimit / (3600 * 24));
  }
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return (0);
}
