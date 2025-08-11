/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/*
 * sdds program to return Touschek lifetime.
 * Using A. Piwinski's formula, DESY 98-179/ISSN 0418-9833.
 * Input files are elegant twiss file with radiation integrals
 * parameters, and a momentum aperture file.
 * Calculated parameters will go in an output sdds file.
 */

#include <stdio.h>
#include "mdb.h"
#include "scan.h"
#include "match_string.h"
#include "SDDS.h"
#include "constants.h"

static char *USAGE = "touschekLifetime <resultsFile>\n\
 -twiss=<twissFile> -aperture=<momentumApertureFile> [-beam=<beamProfile> | -sliceAnalysis=<filename>] \n\
 {-charge=<nC>|-particles=<number>} {-coupling=<value>|-emityInput=<meters>}\n\
 [-deltaLimit=<percent>]\n\
 {-rf=voltage=<MV>,harmonic=<value>[,limit][,superperiods=<number>] | -length=<mm>}\n\
 [-emitInput=<valueInMeters>] [-deltaInput=<value>] [-verbosity=<value>]\n\
 [-method=[0/1] 0-direct; 1-variable substitution] [-ignoreMismatch[=silently]]\n\n\
twiss          Give &twiss_output output file from elegant, with radiation_integrals=1.\n\
aperture       Give &momentum_aperture output file from elegant.\n\
beam           Give beam profile file from elegant2genesis.\n\
sliceAnalysis  Give slice analysis file from elegant SLICE element.\n\
charge         Charge of bunch in nanocoulombs.\n\
particles      Number of electrons in the bunch.\n\
coupling       Ratio between vertical and horizontal emittances.\n\
emityInput     Vertical emittance in meters.\n\
emitInput      Natural emittance in meters. By default, taken from twiss file.\n\
deltaInput     Rms fractional energy spread. By default, taken from twiss file.\n\
deltaLimit     Maximum value (in percent) of stable momentum deviation, imposed as a cap\n\
               over the momentum aperture in the aperture file.\n\
RF             Give rf voltage (in MV) and harmonic number. If limit qualifier\n\
               is given, then deltaLimit is computed from the bucket half-height.\n\
               Otherwise, used only to compute the bunch length.\n\
               If the aperture and twiss data represents only part of the ring,\n\
               the number of superperiods should be specified.\n\
length         Give rms bunch length in mm.\n\
verbosity      Higher values result in more output during computations.\n\
ignoreMismatch Ignore mismatch between names of elements in the Twiss and aperture files.\n\
               By default, exits if mismatch is detected. If 'silently' is given\n\
               ignores with no warning messages.\n\
method         Choose integration method, direct or variable substitution.\n\n\
Program by A. Xiao, M. Borland.  (This is version 8, March 2017, M. Borland)\n";

#define VERBOSE 0
#define CHARGE 1
#define PARTICLES 2
#define COUPLING 3
#define RF 4
#define LENGTH 5
#define EMITINPUT 6
#define DELTAINPUT 7
#define TWISSFILE 8
#define APERFILE 9
#define DELTALIMIT 10
#define IGNORE_MISMATCH 11
#define EMITXINPUT 12
#define EMITYINPUT 13
#define METHOD 14
#define BEAMPROF 15
#define SLICEANAL 16
#define N_OPTIONS 17

char *option[N_OPTIONS] = {
  "verbose",
  "charge",
  "particles",
  "coupling",
  "rf",
  "length",
  "emitinput",
  "deltainput",
  "twiss",
  "aperture",
  "deltalimit",
  "ignoreMismatch",
  "emitxinput",
  "emityinput",
  "method",
  "beam",
  "sliceAnalysis",
};

void TouschekLifeCalc(long verbosity);
double FIntegral_0(double *tm, double *B1, double *B2, long index, long verbosity);
double Fvalue_0(double t, double tm, double b1, double b2);
double FIntegral_1(double *tm, double *B1, double *B2, long index, long verbosity);
double Fvalue_1(double t, double tm, double b1, double b2);
double linear_interpolation(double *y, double *t, long n, double t0, long i);
void limitMomentumAperture(double *dpp, double *dpm, double limit, long n);

/* global varibles */
long plane_orbit = 0;
double *s, *s2, *dpp, *dpm;
double *betax, *alphax, *etax, *etaxp;
double *betay, *alphay, *etay, *etayp;
double *tmP, *tmN, *B1, *B2, *FP, *FN;
double tLife;
long elements, elem2;
long *eOccur1;
long nSlice;
double pCentral, gamma0, NP, *npSlice, *szSlice, *sSlice, *sigmapSlice, *exSlice, *eySlice, *gammaSlice;
double taux, tauy;
char **eName1, **eName2, **eType1;
long ignoreMismatch = 0, method = 0;

#define NDIV 10000;
#define MAXREGION 30;
#define STEPS 100;

#define RF_LIMIT_APERTURE 0x01UL

int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  char *twissInput, *MAInput, *outputfile, *beamInput, *sliceAnalysis;
  SDDS_DATASET twissPage, aperPage, resultsPage, beamProfPage, sliceAnalysisPage;
  long verbosity;
  double etaymin, etaymax;
  long i, superperiods = 1;
  unsigned long rfFlags;
  double emitInput, sigmaDeltaInput, rfVoltage, rfHarmonic, eyInput;
  double alphac, U0, circumference, EMeV;
  double coupling, emitx0, charge, sz, sigmap, emitx, emity;
  short has_ex0 = 0, has_Sdelta0 = 0, has_beam = 0;
  double deltaLimit = 0;

  /****************************************************\
   * read from command line                           *
   \****************************************************/

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&scanned, argc, argv);
  if (argc == 1)
    bomb(NULL, USAGE);

  twissInput = MAInput = outputfile = beamInput = sliceAnalysis = NULL;
  verbosity = 0;
  NP = 0;
  charge = 0;
  coupling = 0;
  sz = 0;
  emitInput = eyInput = emitx = emity = 0;
  sigmaDeltaInput = 0;
  rfVoltage = rfHarmonic = 0;
  rfFlags = 0;

  for (i = 1; i < argc; i++) {
    if (scanned[i].arg_type == OPTION) {
      delete_chars(scanned[i].list[0], "_");
      switch (match_string(scanned[i].list[0], option, N_OPTIONS, UNIQUE_MATCH)) {
      case VERBOSE:
        if (scanned[i].n_items > 1) {
          get_long(&verbosity, scanned[i].list[1]);
        } else {
          verbosity = 1;
        }
        break;
      case CHARGE:
        if (scanned[i].n_items != 2)
          bomb("invalid -charge syntax/values", "-charge=<nC>");
        get_double(&charge, scanned[i].list[1]);
        break;
      case EMITINPUT:
      case EMITXINPUT:
        if (scanned[i].n_items != 2)
          bomb("invalid -emitInput syntax/values", "-emitInput=<value>");
        get_double(&emitInput, scanned[i].list[1]);
        break;
      case DELTAINPUT:
        if (scanned[i].n_items != 2)
          bomb("invalid -deltaInput syntax/values", "-deltaInput=<value>");
        get_double(&sigmaDeltaInput, scanned[i].list[1]);
        break;
      case LENGTH:
        if (scanned[i].n_items != 2)
          bomb("invalid -length syntax/values", "-length=<mm>");
        get_double(&sz, scanned[i].list[1]);
        sz /= 1000; /* convert input length from mm to m */
        break;
      case COUPLING:
        if (scanned[i].n_items != 2)
          bomb("invalid -coupling syntax/values", "-coupling=<value>");
        get_double(&coupling, scanned[i].list[1]);
        break;
      case EMITYINPUT:
        if (scanned[i].n_items != 2)
          bomb("invalid -eyInput syntax/values", "-eyInput=<meters>");
        get_double(&eyInput, scanned[i].list[1]);
        break;
      case PARTICLES:
        if (scanned[i].n_items != 2)
          bomb("invalid -particles syntax/values", "-particles=<value>");
        get_double(&NP, scanned[i].list[1]);
        break;
      case RF:
        if (scanned[i].n_items < 2)
          bomb("invalid -rf syntax", NULL);
        scanned[i].n_items--;
        if (!scanItemList(&rfFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "voltage", SDDS_DOUBLE, &rfVoltage, 1, 0,
                          "harmonic", SDDS_DOUBLE, &rfHarmonic, 1, 0,
                          "superperiods", SDDS_LONG, &superperiods, 1, 0,
                          "limit", -1, NULL, 0, RF_LIMIT_APERTURE,
                          NULL) ||
            rfVoltage <= 0 || rfHarmonic <= 0 || superperiods <= 0)
          bomb("invalid -rf syntax/values", "-rf=voltage=<MV>,harmonic=<value>[,limit][,superperiods=<number>]");
        break;
      case TWISSFILE:
        if (scanned[i].n_items < 2)
          bomb("invalid -twiss syntax", NULL);
        twissInput = scanned[i].list[1];
        break;
      case APERFILE:
        if (scanned[i].n_items < 2)
          bomb("invalid -twiss syntax", NULL);
        MAInput = scanned[i].list[1];
        break;
      case BEAMPROF:
        if (scanned[i].n_items < 2)
          bomb("invalid -beam syntax", NULL);
        beamInput = scanned[i].list[1];
        has_beam = 1;
        break;
      case SLICEANAL:
        if (scanned[i].n_items < 2)
          bomb("invalid -sliceAnalysis syntax", NULL);
        sliceAnalysis = scanned[i].list[1];
        has_beam = 1;
        break;
      case DELTALIMIT:
        if (scanned[i].n_items != 2 || !get_double(&deltaLimit, scanned[i].list[1]) || deltaLimit <= 0)
          bomb("invalid -deltaLimit syntax/values", "-deltaLimit=<percent>");
        deltaLimit /= 100.0;
        break;
      case IGNORE_MISMATCH:
        ignoreMismatch = 1;
	if (scanned[i].n_items==2 &&
	    strncmp(scanned[i].list[1], "silently", strlen(scanned[i].list[1]))==0)
	  ignoreMismatch = 2;
        break;
      case METHOD:
        if (scanned[i].n_items > 1) {
          get_long(&method, scanned[i].list[1]);
        }
        break;
      default:
        fprintf(stderr, "unknown option \"%s\" given\n", scanned[i].list[0]);
        exit(1);
        break;
      }
    } else {
      if (!outputfile)
        outputfile = scanned[i].list[0];
      else
        bomb("too many filenames given", NULL);
    }
  }
  if (charge && NP)
    bomb("Give only one of -charge and -particles", NULL);
  if (!(charge || NP || has_beam))
    bomb("Give one of -charge, -particles, -beam, or -sliceAnalysis", NULL);
  if ((!coupling && !eyInput && !has_beam) || (coupling && eyInput))
    bomb("Give one and only one of coupling, eyInput, beam or sliceAnalysis", NULL);
  if (!sz && !rfVoltage && !has_beam)
    bomb("Specify either the bunch length or the rf voltage or provide the bunch profile SDDS file.", NULL);

  if (!charge)
    charge = NP * e_mks;
  if (!NP) {
    /* command line input value is in units of nC */
    charge /= 1e9;
    NP = charge / e_mks;
  }

  /****************************************************\
   * Check input twissfile                            *
   \****************************************************/
  if (verbosity)
    fprintf(stdout, "Opening \"%s\" for checking presence of parameters.\n", twissInput);
  if (!SDDS_InitializeInput(&twissPage, twissInput))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  SDDS_ReadPage(&twissPage);

  switch (SDDS_CheckParameter(&twissPage, "I1", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL)) {
  case SDDS_CHECK_NONEXISTENT:
    if (verbosity)
      fprintf(stdout, "\tParameter I1 not found in input file.\n");
    break;
  case SDDS_CHECK_WRONGTYPE:
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf(stdout, "Unexpected result from SDDS_CheckParameter routine while checking parameter Type.\n");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  }

  /****************************************************\
   * Check input aperturefile                         *
   \****************************************************/
  if (verbosity)
    fprintf(stdout, "Opening \"%s\" for checking presence of parameters.\n", MAInput);
  if (!SDDS_InitializeInput(&aperPage, MAInput))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  /* Check presence of momentum aperture */
  SDDS_ReadPage(&aperPage);
  switch (SDDS_CheckColumn(&aperPage, "deltaPositive", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL)) {
  case SDDS_CHECK_NONEXISTENT:
    if (verbosity)
      fprintf(stdout, "\tColumn deltaPositive is not found in input file.\n");
    exit(1);
    break;
  case SDDS_CHECK_WRONGTYPE:
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf(stdout, "Unexpected result from SDDS_CheckColumn routine.\n");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  }
  /****************************************************\
   * Check output file and write file layout          *
   \****************************************************/
  if (verbosity)
    fprintf(stdout, "Opening \"%s\" for writing...\n", outputfile);
  if (!SDDS_InitializeOutput(&resultsPage, SDDS_BINARY, 1, "Touschek lifetime calculation",
                             "Touschek lifetime calculation", outputfile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* Ignore error returns from these three statements */
  SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "pCentral", NULL);
  SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "Sdelta0", NULL);
  SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "ex0", NULL);
  SDDS_ClearErrors();

  if (0 > SDDS_DefineParameter(&resultsPage, "coupling", NULL, NULL,
                               "Coupling", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "emitx", "$ge$r$bx$n", "m",
                               "Horizontal emittance with coupling", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "emity", "$ge$r$by$n", "m",
                               "Vertical emittance with coupling", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Sdelta", NULL, NULL,
                               "Fractional momentum spread", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Particles", NULL, NULL,
                               "Particles", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Charge", NULL, "nC",
                               "Charge", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "sigmaz", "$gs$r$bz$n", "m",
                               "Bunch length", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "tLifetime", NULL, "hour",
                               "Touschek half lifetime, which is equivalent to the initial exponential decay lifetime",
                               NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "deltaLimit", NULL, "",
                               "Limit on maximum momentum acceptance from -deltaLimit or -RF with limit qualifier.",
                               NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "RfVoltage", "V$brf$n", "MV", "Rf voltage", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Superperiods", NULL, NULL, "Number of superperiods", NULL, SDDS_LONG, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (!SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "s", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "betax", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "alphax", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "etax", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "etaxp", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "betay", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "alphay", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "etay", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "etayp", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "ElementName", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "ElementType", NULL) ||
      !SDDS_TransferColumnDefinition(&resultsPage, &twissPage, "ElementOccurence", NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (0 > SDDS_DefineColumn(&resultsPage, "tmP", NULL, NULL,
                            "Local positive momentum aperture (beta*dp/p)^2", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&resultsPage, "tmN", NULL, NULL,
                            "Local negative momentum aperture (beta*dp/p)^2", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&resultsPage, "B1", NULL, NULL,
                            "Piwinski's parameter B1", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&resultsPage, "B2", NULL, NULL,
                            "Piwinski's parameter B2", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&resultsPage, "FP", NULL, "1/s/m",
                            "Local particle loss rate for positive momentum particle", NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&resultsPage, "FN", NULL, "1/s/m",
                            "Local particle loss rate for negative momentum particle", NULL, SDDS_DOUBLE, 0))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (!SDDS_WriteLayout(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (verbosity)
    fprintf(stdout, "Finished setting up \"%s\".\n", outputfile);

  /****************************************************\
   * read from twiss file                             *
   \****************************************************/

  if (verbosity) {
    fprintf(stdout, "Reading twiss file...");
    fflush(stdout);
  }
  if (!SDDS_GetParameters(&twissPage,
                          "pCentral", &pCentral, "taux", &taux, "tauy", &tauy,
                          NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  elements = SDDS_CountRowsOfInterest(&twissPage);
  s = SDDS_GetColumnInDoubles(&twissPage, "s");
  elem2 = SDDS_CountRowsOfInterest(&aperPage);
  s2 = SDDS_GetColumnInDoubles(&aperPage, "s");
  if (elements < elem2)
    fprintf(stdout, "warning: Twiss file is shorter than Aperture file\n");
  if (!has_beam) {
    if (emitInput) {
      emitx = emitInput / (1 + coupling * taux / tauy);
    } else {
      if (!SDDS_GetParameters(&twissPage, "ex0", &emitx0, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      emitx = emitx0 / (1 + coupling * taux / tauy);
      has_ex0 = 1;
    }
    if (eyInput)
      emity = eyInput;
    else
      emity = emitx * coupling;
    if (sigmaDeltaInput) {
      sigmap = sigmaDeltaInput;
    } else {
      if (!SDDS_GetParameters(&twissPage, "Sdelta0", &sigmap, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      has_Sdelta0 = 1;
    }
    circumference = s[elements - 1];
    EMeV = sqrt(sqr(pCentral) + 1) * me_mev;
    if (!sz) {
      /* compute length in m from rf voltage, energy spread, etc */
      if (!SDDS_GetParameters(&twissPage, "alphac", &alphac,
                              "U0", &U0, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      sz =
        superperiods * circumference * sigmap *
        sqrt(alphac * EMeV / (PIx2 * rfHarmonic * sqrt(sqr(rfVoltage) - sqr(U0 * superperiods))));
      if (rfFlags & RF_LIMIT_APERTURE) {
        double q, rfLimit = 0;
        q = rfVoltage / (superperiods * U0);
        if (q < 1 || (rfLimit = sqrt(2 * U0 * superperiods / (PI * alphac * rfHarmonic * EMeV) * (sqrt(q * q - 1) - acos(1 / q)))) == 0)
          SDDS_Bomb("rf voltage too low compared to energy loss per turn");
        if (deltaLimit == 0 || rfLimit < deltaLimit)
          deltaLimit = rfLimit;
        if (verbosity) {
          printf("momentum acceptance limited to +/-%le by rf\n", deltaLimit);
          fflush(stdout);
        }
      }
    }
  }

  betax = SDDS_GetColumnInDoubles(&twissPage, "betax");
  betay = SDDS_GetColumnInDoubles(&twissPage, "betay");
  alphax = SDDS_GetColumnInDoubles(&twissPage, "alphax");
  alphay = SDDS_GetColumnInDoubles(&twissPage, "alphay");
  etax = SDDS_GetColumnInDoubles(&twissPage, "etax");
  etaxp = SDDS_GetColumnInDoubles(&twissPage, "etaxp");
  etay = SDDS_GetColumnInDoubles(&twissPage, "etay");
  etayp = SDDS_GetColumnInDoubles(&twissPage, "etayp");
  eName1 = SDDS_GetColumn(&twissPage, "ElementName");
  eType1 = SDDS_GetColumn(&twissPage, "ElementType");
  eOccur1 = SDDS_GetColumn(&twissPage, "ElementOccurence");

  if (verbosity) {
    fprintf(stdout, "done.\n");
    fflush(stdout);
  }

  if (verbosity) {
    fprintf(stdout, "Reading aperture file...");
    fflush(stdout);
  }
  if (!(eName2 = SDDS_GetColumn(&aperPage, "ElementName"))) {
    fprintf(stderr, "Error reading ElementName from aperture file\n");
    exit(1);
  }

  for (i = 0; i < elem2; i++)
    trim_spaces(eName2[i]);

  dpp = SDDS_GetColumnInDoubles(&aperPage, "deltaPositive");
  dpm = SDDS_GetColumnInDoubles(&aperPage, "deltaNegative");
  if (deltaLimit > 0)
    limitMomentumAperture(dpp, dpm, deltaLimit, elem2);

  if (verbosity) {
    fprintf(stdout, "done.\n");
    fflush(stdout);
  }

  /****************************************************\
   * Check and read beam profile input                *
  \****************************************************/
  if (has_beam) {
    double sSum, s2Sum;
    short increasing = 0;
    if (beamInput) {
      if (verbosity) {
        fprintf(stdout, "Opening \"%s\" for checking presence of parameters.\n", beamInput);
        fflush(stdout);
      }
      if (!SDDS_InitializeInput(&beamProfPage, beamInput))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      SDDS_ReadPage(&beamProfPage);
      if (!SDDS_CheckColumn(&beamProfPage, "Ne", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&beamProfPage, "s", "m", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&beamProfPage, "Sdelta", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&beamProfPage, "xemit", "m", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&beamProfPage, "yemit", "m", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&beamProfPage, "gamma", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      nSlice = SDDS_CountRowsOfInterest(&beamProfPage);
      npSlice = SDDS_GetColumnInDoubles(&beamProfPage, "Ne");
      sSlice = SDDS_GetColumnInDoubles(&beamProfPage, "s");
      sigmapSlice = SDDS_GetColumnInDoubles(&beamProfPage, "Sdelta");
      exSlice = SDDS_GetColumnInDoubles(&beamProfPage, "xemit");
      eySlice = SDDS_GetColumnInDoubles(&beamProfPage, "yemit");
      gammaSlice = SDDS_GetColumnInDoubles(&beamProfPage, "gamma");

      /* Convert normalized emittance to geometric emittance */
      for (i = 0; i < nSlice; i++)
        if (gammaSlice[i] > 0) {
          exSlice[i] /= gammaSlice[i];
          eySlice[i] /= gammaSlice[i];
        }
      free(gammaSlice);

    } else {
      /* slice analysis file */
      if (verbosity) {
        fprintf(stdout, "Opening \"%s\" for checking presence of parameters.\n", sliceAnalysis);
        fflush(stdout);
      }
      if (!SDDS_InitializeInput(&sliceAnalysisPage, sliceAnalysis))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      SDDS_ReadPage(&sliceAnalysisPage);
      if (!SDDS_CheckColumn(&sliceAnalysisPage, "Charge", "C", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&sliceAnalysisPage, "Ct", "s", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&sliceAnalysisPage, "Sdelta", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&sliceAnalysisPage, "ecx", "m", SDDS_DOUBLE, verbosity ? stdout : NULL) ||
          !SDDS_CheckColumn(&sliceAnalysisPage, "ecy", "m", SDDS_DOUBLE, verbosity ? stdout : NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      nSlice = SDDS_CountRowsOfInterest(&sliceAnalysisPage);
      npSlice = SDDS_GetColumnInDoubles(&sliceAnalysisPage, "Charge");
      sSlice = SDDS_GetColumnInDoubles(&sliceAnalysisPage, "Ct");
      sigmapSlice = SDDS_GetColumnInDoubles(&sliceAnalysisPage, "Sdelta");
      exSlice = SDDS_GetColumnInDoubles(&sliceAnalysisPage, "ecx");
      eySlice = SDDS_GetColumnInDoubles(&sliceAnalysisPage, "ecy");

      /* Convert Coulombs to # electrons */
      for (i = 0; i < nSlice; i++)
        npSlice[i] /= e_mks;
      /* Convert seconds to meters */
      for (i = 0; i < nSlice; i++)
        sSlice[i] *= c_mks;
    }

    NP = sz = sigmap = emitx = emity = 0;
    sSum = s2Sum = 0;
    szSlice = tmalloc(sizeof(*szSlice) * nSlice);

    /* Ensure that s values are monotonically increasing */
    for (i = 1; i < nSlice; i++) {
      if (sSlice[i] > sSlice[i - 1])
        increasing++;
    }
    if (increasing != (nSlice - 1) && increasing != 0) {
      fprintf(stdout, "Error: slice data is not monotonic in the 's' coordinate\n");
      exit(1);
    }
    if (increasing == 0)
      for (i = 0; i < nSlice; i++)
        sSlice[i] *= -1;
    /* suppress large values of s */
    for (i = nSlice - 1; i >= 0; i--)
      sSlice[i] -= sSlice[0];

    /* Compute number of particles, weighted emittances, weight energy spread */
    for (i = 0; i < nSlice; i++) {
      NP += npSlice[i];
      sSum += npSlice[i] * sSlice[i];
      s2Sum += npSlice[i] * sqr(sSlice[i]);
      if (i + 1 == nSlice) {
        szSlice[i] = szSlice[i - 1];
      } else {
        szSlice[i] = (sSlice[i + 1] - sSlice[i]) / 2. / sqrt(PI);
      }
      sigmap += npSlice[i] * sigmapSlice[i];
      emitx += npSlice[i] * exSlice[i];
      emity += npSlice[i] * eySlice[i];
    }
    sigmap /= NP;
    emitx /= NP;
    emity /= NP;
    sz = sqrt(s2Sum / NP - sqr(sSum / NP));
    coupling = emity / emitx;
    charge = NP * e_mks;
    free(sSlice);
  } else {
    nSlice = 1;
    npSlice = calloc(sizeof(double), 1);
    szSlice = calloc(sizeof(double), 1);
    sigmapSlice = calloc(sizeof(double), 1);
    exSlice = calloc(sizeof(double), 1);
    eySlice = calloc(sizeof(double), 1);
    npSlice[0] = NP;
    szSlice[0] = sz;
    sigmapSlice[0] = sigmap;
    exSlice[0] = emitx;
    eySlice[0] = emity;
  }

  if (verbosity > 0)
    fprintf(stdout, "Beam parameters set up.\n");

  /****************************************************\
   * calculate Touschek Lifetime                      *
   \****************************************************/
  find_min_max(&etaymin, &etaymax, etay, elements);
  if ((etaymax - etaymin) < 1e-6)
    plane_orbit = 1;

  if (!(tmP = SDDS_Malloc(sizeof(*tmP) * elements)) ||
      !(tmN = SDDS_Malloc(sizeof(*tmN) * elements)) ||
      !(B1 = SDDS_Malloc(sizeof(*B1) * elements)) ||
      !(B2 = SDDS_Malloc(sizeof(*B2) * elements)) ||
      !(FP = SDDS_Malloc(sizeof(*FP) * elements)) ||
      !(FN = SDDS_Malloc(sizeof(*FN) * elements)))
    bomb("memory allocation failure (integration arrays)", NULL);

  TouschekLifeCalc(verbosity);

  /****************************************************\
   * Write output file                                *
   \****************************************************/
  if (0 > SDDS_StartPage(&resultsPage, elements) ||
      !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "pCentral", pCentral,
                          "Sdelta", sigmap,
                          "coupling", coupling,
                          "emitx", emitx,
                          "emity", emity,
                          "Particles", NP,
                          "Charge", (1e9 * charge),
                          "tLifetime", tLife,
                          "deltaLimit", deltaLimit,
                          "sigmaz", sz,
                          "RfVoltage", rfVoltage,
                          "Superperiods", superperiods, NULL) ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, s, elements, "s") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, betax, elements, "betax") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, alphax, elements, "alphax") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, etax, elements, "etax") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, etaxp, elements, "etaxp") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, betay, elements, "betay") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, alphay, elements, "alphay") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, etay, elements, "etay") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, etayp, elements, "etayp") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, eName1, elements, "ElementName") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, eType1, elements, "ElementType") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, eOccur1, elements, "ElementOccurence") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, tmP, elements, "tmP") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, tmN, elements, "tmN") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, B1, elements, "B1") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, B2, elements, "B2") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, FP, elements, "FP") ||
      !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, FN, elements, "FN"))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (has_ex0 &&
      !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ex0", emitx0, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (has_Sdelta0 &&
      !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Sdelta0", sigmap, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_WritePage(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (SDDS_ReadPage(&twissPage) > 0)
    fprintf(stdout, "The code doesn't support multi twiss pages.\n");
  if (SDDS_ReadPage(&aperPage) > 0)
    fprintf(stdout, "The code doesn't support multi aperture pages.\n");

  if (!SDDS_Terminate(&twissPage) ||
      !SDDS_Terminate(&aperPage) ||
      !SDDS_Terminate(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (has_beam) {
    if (beamInput) {
      if (SDDS_ReadPage(&beamProfPage) > 0)
        fprintf(stdout, "The code doesn't support multiple beam pages.\n");
      if (!SDDS_Terminate(&beamProfPage))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    } else {
      if (SDDS_ReadPage(&sliceAnalysisPage) > 0)
        fprintf(stdout, "The code doesn't support multiple slice analysis pages.\n");
      if (!SDDS_Terminate(&sliceAnalysisPage))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  return (0);
}

void TouschekLifeCalc(long verbosity) {
  long i, j, k;
  double sp2, sp4, beta2, betagamma2, sh2;
  double sx2, sxb2, dx2, dx_2;
  double sy2, syb2, dy2, dy_2;
  double a0, a1, c0, c1, c2, c3;
  double pm, pp;
  double fnSlice, fpSlice, coeff;

  beta2 = sqr(pCentral) / (sqr(pCentral) + 1);
  betagamma2 = sqr(pCentral);
  a1 = sqr(re_mks) * c_mks / (4 * sqrt(PI) * (sqr(pCentral) + 1));
  i = j = k = 0;
  if (verbosity > 1)
    fprintf(stderr, "In TouschekLifeCalc\n");
  for (i = 0; i < elements; i++) {
    if (verbosity > 1)
      fprintf(stderr, "Working on i=%ld, j=%ld\n", i, j);

    /* remove zero length elements from beamline. Except the first element. */
    if (i > 0) {
      while (s[i] == s[i - 1]) {
        tmP[i] = tmP[i - 1];
        tmN[i] = tmN[i - 1];
        B1[i] = B1[i - 1];
        B2[i] = B2[i - 1];
        FP[i] = FP[i - 1];
        FN[i] = FN[i - 1];
        if (++i == elements)
          break;
      }
    }
    if (i == elements)
      break;

    if (j > 0) {
      while (s2[j] == s2[j - 1]) {
        dpp[j] = dpp[j - 1];
        dpm[j] = dpm[j - 1];
        if (++j == elem2) {
          j--;
          break;
        }
      }
    }
    /* compare two files if it's for same beamline. */
    if (s[i] > s2[j])
      j++;
    if (j == elem2)
      j--;
    /* Normally the first element for twiss file is _BEG_, which is not true for aperture file.
       But we need it for starting the calculation. */
    if (s[i] != 0 && s[i] == s2[j] && strcmp(eName1[i], eName2[j]) != 0) {
      if (ignoreMismatch) {
	if (ignoreMismatch==1)
	  printf("warning: element1 \"%s\" and elem2 \"%s\" at s1 %21.15e s2 %21.15e don't match\n", eName1[i], eName2[j], s[i], s2[j]);
      } else {
        printf("error: element1 \"%s\" and elem2 \"%s\" at s1 %21.15e s2 %21.15e don't match\n", eName1[i], eName2[j], s[i], s2[j]);
        bomb("Twiss and Aperture file are not for same beamline", NULL);
      }
    }

    if (verbosity > 1)
      fprintf(stderr, "Interpolating for s[i=%ld] = %e, s2[j=%ld to %ld] = %le, %le\n", i, s[i],
              j - 1, j, s2[j - 1], s2[j]);
    pp = linear_interpolation(dpp, s2, elem2, s[i], j - 1);
    pm = linear_interpolation(dpm, s2, elem2, s[i], j - 1);

    tmP[i] = beta2 * pp * pp;
    tmN[i] = beta2 * pm * pm;
    if (verbosity > 2)
      fprintf(stderr, "pp = %le, pm = %le, tmP[%ld] = %le, tmN[%ld] = %le\n", pp, pm, i, tmP[i], i, tmN[i]);

    if (tmP[i] == 0 || tmN[i] == 0) {
      B1[i] = B2[i] = 0;
      FP[i] = FN[i] = 0;
      /* Bombing is not good behavior. Just return 0 lifetime so scripts can behave appropriately. */
      /* bomb("zero momentum aperture??? This should never happen",NULL); */
    } else {
      for (k = 0; k < nSlice; k++) {
        sp2 = sqr(sigmapSlice[k]);
        sp4 = sqr(sp2);
        a0 = a1 * npSlice[k] / szSlice[k];
        sxb2 = betax[i] * exSlice[k];
        syb2 = betay[i] * eySlice[k];
        dx2 = ipow2(etax[i]);
        sx2 = sxb2 + dx2 * sp2;

        dx_2 = ipow2(alphax[i] * etax[i] + betax[i] * etaxp[i]);
        c1 = sqr(betax[i]) / (2 * betagamma2 * sxb2);
        c2 = sqr(betay[i]) / (2 * betagamma2 * syb2);

        if (verbosity > 3)
          fprintf(stderr, "slice %ld: sp2=%le, a0=%le, sxb2=%le, syb2=%le, dx2=%le, sx2=%le, dx_2=%le, c1=%le, c2=%le\n",
                  k, sp2, a0, sxb2, syb2, dx2, sx2, dx_2, c1, c2);
        if (isnan(c1) || isnan(c2) || isinf(c1) || isinf(c2)) {
          if (verbosity > 3)
            fprintf(stderr, "Slice invalid (probably low population)---ignored\n");
          continue;
        }

        if (plane_orbit) {
          sh2 = 1 / (1 / sp2 + (dx2 + dx_2) / sxb2);
          B1[i] = c1 * (1 - sh2 * dx_2 / sxb2) + c2;
          c0 = sqrt(sh2) / (sigmapSlice[k] * betagamma2 * exSlice[k] * eySlice[k]);
          c3 = sx2 * syb2;
          B2[i] = sqr(B1[i]) - sqr(c0) * c3;
          if (B2[i] < 0) {
            if (fabs(B2[i] / sqr(B1[i])) < 1e-7) {
              fprintf(stdout, "warning: B2^2<0 at \"%s\" occurence %ld. Please seek experts help.\n", eName1[i], eOccur1[i]);
            } else {
              B2[i] = 0;
            }
          }
          B2[i] = sqrt(B2[i]);
        }

        if (!plane_orbit) {
          dy2 = ipow2(etay[i]);
          sy2 = syb2 + dy2 * sp2;
          dy_2 = ipow2(alphay[i] * etay[i] + betay[i] * etayp[i]);
          sh2 = 1 / (1 / sp2 + (dx2 + dx_2) / sxb2 + (dy2 + dy_2) / syb2);
          c0 = sqrt(sh2) / (sigmapSlice[k] * betagamma2 * exSlice[k] * eySlice[k]);
          c3 = sx2 * sy2 - sp4 * dx2 * dy2;
          B1[i] = c1 * (1 - sh2 * dx_2 / sxb2) + c2 * (1 - sh2 * dy_2 / syb2);
          B2[i] = sqr(B1[i]) - sqr(c0) * c3;
          if (B2[i] < 0) {
            if (fabs(B2[i] / sqr(B1[i])) < 1e-7) {
              fprintf(stdout, "warning: B2^2<0 at \"%s\" occurence %ld. Please seek experts help.\n", eName1[i], eOccur1[i]);
            } else {
              B2[i] = 0;
            }
          }
          B2[i] = sqrt(B2[i]);
        }

        if (verbosity > 1)
          fprintf(stderr, "Computing F integrals for i=%ld, k=%ld\n", i, k);
        if (method) {
          if (verbosity > 2)
            fprintf(stderr, "Computing F integral P (%le, %le, %le)\n", tmP[i], B1[i], B2[i]);
          fpSlice = FIntegral_1(tmP, B1, B2, i, verbosity);
          if (verbosity > 2)
            fprintf(stderr, "Computing F integral N (%le, %le, %le)\n", tmN[i], B1[i], B2[i]);
          fnSlice = FIntegral_1(tmN, B1, B2, i, verbosity);
          coeff = a0 * c0;
        } else {
          if (verbosity > 2)
            fprintf(stderr, "Computing F integral P (%le, %le, %le)\n", tmP[i], B1[i], B2[i]);
          fpSlice = FIntegral_0(tmP, B1, B2, i, verbosity);
          if (verbosity > 2)
            fprintf(stderr, "Computing F integral N (%le, %le, %le)\n", tmN[i], B1[i], B2[i]);
          fnSlice = FIntegral_0(tmN, B1, B2, i, verbosity);
          coeff = a0 * c0 / 2;
        }
        FP[i] += coeff * fpSlice * npSlice[k] / NP;
        FN[i] += coeff * fnSlice * npSlice[k] / NP;
      }
    }
  }

  if (verbosity > 1)
    fprintf(stderr, "Exited loop\n");

  /* Normalize FN and FP so the units are 1/s/m */
  for (i = 0; i < elements; i++) {
    FN[i] /= s[elements - 1];
    FP[i] /= s[elements - 1];
  }

  tLife = 0;
  for (i = 1; i < elements; i++) {
    if (s[i] > s[i - 1]) {
      tLife += (s[i] - s[i - 1]) * (FP[i] + FN[i] + FP[i - 1] + FN[i - 1]) / 4.;
    }
  }
  if (verbosity > 1)
    fprintf(stderr, "Exited second loop\n");

  tLife = 1 / tLife / 3600;
  return;
}

double FIntegral_1(double *tm, double *B1, double *B2, long index, long verbosity) {
  double f0, f1, sum;
  double HPI, step;
  long converge = 0, pass, f1NonzeroSeen;
  double t1, k0, k1;
  double tstart, km, b1, b2;

  HPI = PI / 2;
  step = HPI / NDIV;
  tstart = tm[index];
  b1 = B1[index];
  b2 = B2[index];
  k0 = km = atan(sqrt(tstart));
  f0 = Fvalue_1(tstart, tstart, b1, b2);
  sum = 0;
  f1 = 0;
  pass = 0;
  f1NonzeroSeen = 0;

  while (!converge) {
    if (verbosity > 2)
      fprintf(stderr, "Computing F integral pass %ld: k0=%21.15e, f1=%21.15e\n",
              pass, k0, f1);
    pass++;

    k1 = k0 + step;
    t1 = sqr(tan(k1));
    if (isnan(t1)) {
      fprintf(stdout, "Integration failed at Element# %ld", index);
      break;
    }

    f1 = Fvalue_1(t1, tstart, b1, b2);
    if (f1 > 0)
      f1NonzeroSeen = 1;

    if ((f1NonzeroSeen && f1 == 0) || (f1 > 0 && f1 * (HPI - k1) < 1e-3 * sum))
      converge = 1;

    sum += (f1 + f0) / 2 * step;
    k0 = k1;
    f0 = f1;
  }
  return sum;
}

double Fvalue_1(double t, double tm, double b1, double b2) {
  double c0, c1, c2, result;

  c0 = (sqr(2 * t + 1) * (t / tm / (1 + t) - 1) / t + t - sqrt(t * tm * (1 + t)) - (2 + 1 / (2 * t)) * log(t / tm / (1 + t))) * sqrt(1 + t);
  c1 = exp(-b1 * t);
  c2 = dbesi0(b2 * t);
  result = c0 * c1 * c2;
  /* If overflow/underflow use approximate equation for modified bessel function. */
  if (isnan(result) || fabs(result) > FLT_MAX || fabs(result) == 0) {
    result = c0 * exp(b2 * t - b1 * t) / sqrt(PIx2 * b2 * t);
  }
  return result;
}

double FIntegral_0(double *tm, double *B1, double *B2, long index, long verbosity) {
  long maxRegion = MAXREGION;
  long steps = STEPS; /* number of integration steps per decade */
  long i, j;
  double test = 1e-5, simpsonCoeff[2] = {2., 4.};
  double h, tau0, tau, tstart, intF;
  double cof, sum, fun;
  double b1, b2;
  /* Using simpson's rule to do the integral.
     split integral into region with "steps" steps per region.
     integration intervals be [1,3], [3,9], [9,27], and so on.
     i is the index over these intervals. j is the index over each step.
  */
  b1 = B1[index];
  b2 = B2[index];
  tstart = tm[index];
  tau0 = tstart;
  intF = 0.0;
  for (i = 0; i < maxRegion; i++) {
    if (i == 0) {
      h = tau0 * 2. / steps;
    } else {
      h = tau0 * 3. / steps;
    }
    sum = 0.0;
    for (j = 0; j <= steps; j++) {
      tau = tau0 + h * j;
      cof = simpsonCoeff[j % 2];
      if (j == 0 || j == steps)
        cof = 1.;
      fun = Fvalue_0(tau, tstart, b1, b2);
      sum += cof * fun;
    }
    tau0 = tau;
    sum = (sum / 3.0) * h;
    intF += sum;
    if (FABS(sum / intF) < test)
      break;
    if (i == maxRegion)
      fprintf(stdout, "**Warning** Integral did not converge till tau= %g.\n", tau);
  }

  return intF;
}

double Fvalue_0(double t, double tm, double b1, double b2) {
  double c0, c1, c2, result;

  c0 = (sqr(2. + 1. / t) * (t / tm / (1. + t) - 1.) + 1. - sqrt(tm * (1. + t) / t) - (4. + 1. / t) / (2. * t) * log(t / tm / (1. + t))) * sqrt(t / (1. + t));
  c1 = exp(-b1 * t);
  c2 = dbesi0(b2 * t);
  result = c0 * c1 * c2;
  /* If overflow/underflow use approximate equation for modified bessel function. */
  if (isnan(result) || fabs(result) > FLT_MAX) {
    result = c0 * exp(b2 * t - b1 * t) / sqrt(PIx2 * b2 * t);
  }
  return result;
}

/* Only linear_interpolate from i=0 to i=n-2. For i<0 using i=0. For i>n-2, using i=n-1 */
double linear_interpolation(double *y, double *t, long n, double t0, long i) {
  if (i < 0)
    i = 0;
  if (i >= n - 1)
    i = n - 2;
  while (i <= n - 2 && t0 > t[i + 1])
    i++;
  if (i == n - 1)
    return (y[n - 1]);
  while (i >= 0 && t0 < t[i])
    i--;
  if (i == -1)
    return (y[0]);
  if (!(t0 >= t[i] && t0 <= t[i + 1])) {
    fprintf(stdout, "failure to bracket point in time array: t0=%e, t[0] = %e, t[n-1] = %e\n",
            t0, t[0], t[n - 1]);
    fflush(stdout);
    abort();
  }
  /* this handles zero length elements */
  if (t[i + 1] == t[i])
    return y[i];

  return (y[i] + (y[i + 1] - y[i]) / (t[i + 1] - t[i]) * (t0 - t[i]));
}

void limitMomentumAperture(double *dpp, double *dpm, double limit, long n) {
  long i;
  double limitn;

  limitn = -limit;
  for (i = 0; i < n; i++) {
    if (dpp[i] > limit)
      dpp[i] = limit;
    if (dpm[i] < limitn)
      dpm[i] = limitn;
  }
}
