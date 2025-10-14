/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* 
 * $Log: not supported by cvs2svn $
 * Revision 1.34  2011/08/02 13:27:53  borland
 * Previously, -emitxInput was renamed -emitInput, in order to be more accurate.
 * However, this breaks existing scripts.
 * Restored the -emitxInput option for backward compatibility.
 * It has the same function as -emitInput, but is not documented in order to
 * discourage use.
 *
 * Revision 1.33  2011/07/28 19:56:12  xiaoam
 * Fixed the bug for forced coupling calculation. Add calculation which takes different taux tauy. Modified equibrium emittance calculation which suit for coupling case.
 *
 * Revision 1.32  2010/11/23 22:02:45  xiaoam
 * Add forceCoupling option to force coupling to be constant (default).
 *
 * Revision 1.31  2010/08/12 15:32:20  borland
 * Added exitElegant() routine so that any exit will result in creation of
 * semaphore files if requested.
 *
 * Revision 1.30  2010/02/23 20:15:10  borland
 * Emittance given on commandline is treated in same way as emittance from input file.
 *
 * Revision 1.29  2010/02/05 22:05:38  soliday
 * Made some changes to reduce the number or compiler warnings on Linux.
 *
 * Revision 1.28  2008/10/23 19:14:31  xiaoam
 * Remove not used parameter coupling inside IBSCATTER.
 *
 * Revision 1.27  2008/05/20 20:45:29  xiaoam
 * Change code exits to warning message.
 *
 * Revision 1.26  2008/05/20 19:30:17  xiaoam
 * updated for new IBS calculation.
 *
 * Revision 1.25  2008/03/24 14:51:57  emery
 * Rewrote usage message to indicate that the -growthRateOnly option
 * is incompatible with the -integrate option.
 * Added -noWarning option.
 * Added SDDS parameters for initial values of growth rates.
 * Added a call to IBSGrowthRates just to get initial values of
 * growth rate calculated and put in output file parameters (this
 * was necessary for -integration option).
 * For -integrate option, set parameter variables from the arrays resulting
 * from integration.
 *
 * Revision 1.24  2008/03/19 23:04:53  emery
 * Added parameter sigmaDeltaInput to the output.
 * Removed periods in SDDS column and parameter descriptions.
 *
 * Revision 1.23  2008/03/06 23:07:23  borland
 * Fixed problem on 64 bit machines.
 *
 * Revision 1.22  2007/01/24 17:17:56  emery
 * Added comment clarifying the meaning of G and
 * the returned value of IBSequations.
 *
 * Revision 1.21  2005/11/10 15:38:48  soliday
 * Added changes to get it to compile properly with 64 bit compilers.
 *
 * Revision 1.20  2004/11/10 20:50:14  borland
 * Reduced the default target to 1e-6, and made the tolerance 1/100 of
 * the target.
 *
 * Revision 1.19  2004/10/08 07:17:29  borland
 * Added rf voltage to the output file.
 *
 * Revision 1.18  2003/05/07 03:11:14  emery
 * For growthRateOnly option, omit from the outputfile
 * the emittance, energy spread and bunch length, which are not changed.
 *
 * Revision 1.17  2003/01/21 18:48:07  borland
 * Now output dIBSRate in 1/(m*s), which is more useful.
 * Also, fixed units on IBSRate parameters (should be 1/s).
 *
 * Revision 1.16  2003/01/18 03:04:53  borland
 * Include contribution to total rate vs s in columns.
 *
 * Revision 1.15  2003/01/16 02:20:36  borland
 * Added simplex_divisor parameter to optimization_setup namelist.
 * Updated calls to simplexMin().
 *
 * Revision 1.14  2002/11/12 02:33:51  borland
 * In integrate mode, now outputs the bunch length and energy spread.
 *
 * Revision 1.13  2002/09/24 22:01:48  borland
 * Added FLOOR element for setting floor coordinates inside a beamline.
 * Improved algorithm for finding closed orbit with path-length constraint.
 * Added x and y error plus delta value to closed orbit output.
 * Added verbosity features for IBS.
 * Added optimization methods (random walk and random sample).
 *
 * Revision 1.12  2002/08/14 20:23:40  soliday
 * Added Open License
 *
 * Revision 1.11  2001/10/15 20:37:03  soliday
 * Cleaned up for Linux.
 *
 * Revision 1.10  2001/08/02 14:45:28  borland
 * Added search_path feature to run_setup namelist and to many (all?) elements and
 * commands that take input files.
 * FITPOINT beam statistics data now includes x, y, and z emittances.
 *
 * Revision 1.9  2001/05/16 19:02:58  borland
 * Modified calls to simplexMin() to accomodate new argument.
 * Added simplified Rosenzweig/Serafini model for rf focusing in the
 * body of RFCA and RFCW elements.
 *
 * Revision 1.8  2000/10/23 18:57:08  borland
 * Implemented -energy option.
 *
 * Revision 1.7  1999/10/12 21:49:55  borland
 * All printouts now go to the stdout rather than stderr.  fflush statements,
 * some unnecessary, were added in a mostly automated fashion.
 *
 * Revision 1.6  1999/08/05 15:35:52  soliday
 * Added WIN32 and Linux support
 *
 * Revision 1.5  1999/03/18 21:02:44  borland
 * Implemented the -rf option.
 *
 * Revision 1.4  1999/02/15 14:41:26  borland
 * Added integration vs time.
 *
 * Revision 1.3  1999/02/11 20:46:35  borland
 * Now uses target parameter of simplexMin to accept any solution with
 * 1e-4 or better convergence.
 *
 * Revision 1.2  1999/01/27 17:55:15  borland
 * Removed prototypes from ibsEmittance.c and zibs.c and put them in common
 * file zibs.h
 *
 * Revision 1.1  1999/01/25 10:51:05  emery
 * First installation of ibsEmittance.c zibs.c
 * for intra-beam scattering rate calculation
 * and equilibrium emittance calculation.
 *
 * sdds program to return value of emittance
 * with intra-beam scattering included.
 * Input file is an elegant twiss file with
 * radiation integrals parameters.
 * Commmand line argument is energy in MeV.
 * Calculated parameters will go in an output
 * sdds file.
*/
#include <stdio.h>
#include "mdb.h"
#include "scan.h"
#include "match_string.h"
#include "SDDS.h"
#include "constants.h"
static char *USAGE = "ibsEmittance <twissFile> <resultsFile>\n\
 {-charge=<nC>|-particles=<value>} {-coupling=<value>|-emityInput=<meters>}\n\
 [-emitInput=<value>] [-deltaInput=<value>] \n\
 [-emit0=<value>] [-delta0=<value>] \n\
 [-superperiods=<value>] [-isRing=1|0] [-forceCoupling=1|0] [-fixEmity=0|1] \n\
 {-RF=Voltage=<MV>,harmonic=<value>|-length=<mm>|-distribution=<timeHistogramFile>,<t-column>,<hist-column>}\n\
 [-energy=<MeV>] \n\
 [ {-growthRatesOnly | -integrate=turns=<number>[,stepSize=<number>][,verbose] } ]\n\
 [-noWarning]\n\n\
-charge          Bunch charge in nanoCoulombs.\n\
-particles       Number of electrons per bunch.\n\
-coupling        Ratio of vertical to horizontal emittance.\n\
-emityInput      Initial rms vertical emittance.\n\
-deltaInput      Initial rms energy spread.\n\
-emit0           Natural emittance. By default, use ex0 parameter in <twissFile>.\n\
-delta0          Natural rms energy spread. By default, use Sdelta0 parameter in <twissFile>.\n\
-superperiods    If <twissFile> was computed for only a fraction of the ring, set to indicate\n\
                 how many repetitions of the <twissFile> are needed to make a full ring.\n\
-isRing          Default is 1. Set to 0 for linac or transport line.\n\
-forceCoupling   Default is 1, meaning that ratio given with -coupling option is enforced.\n\
-fixEmity        Default is 0, meaning that vertical emittance changes due to IBS.\n\
                 If 1, then it is assumed that some other process maintains fixed vertical\n\
                 emittance at the initial value. Ignored for -integration mode.\n\
-RF              Specify rf parameters for bunch length calculation.\n\
-length          Specify initial bunch length. Ratio of energy spread to bunch length is held fixed.\n\
-distribution    Give slice-by-slice distribution of charge density in the bunch.\n\
-energy          Change the beam energy by scaling radiation-integrals in <twissFile>.\n\
-growthRatesOnly Compute only the growth rates, not the equilibrium.\n\
-integrate       Integrate vs turn to show evolution of beam parameters.\n\
-noWarning       Turn off warning messages.\n\n\
By L. Emery, A. Xiao, and M. Borland. Version 6, January 2016\n";

#define SET_ENERGY 0
#define VERBOSE 1
#define CHARGE 2
#define PARTICLES 3
#define COUPLING 4
#define RF 5
#define LENGTH 6
#define SUPERPERIOD 7
#define METHOD 8
#define EMITINPUT 9
#define DELTAINPUT 10
#define GROWTHRATESONLY 11
#define SET_TARGET 12
#define SET_INTEGRATE 13
#define NO_WARNING 14
#define ISRING 15
#define FORCECOUPLING 16
#define EMITXINPUT 17
#define EMITYINPUT 18
#define EMIT0 19
#define DELTA0 20
#define DISTRIBUTION 21
#define FIXEMITY 22
#define N_OPTIONS 23
char *option[N_OPTIONS] = {
  "energy",
  "verbose",
  "charge",
  "particles",
  "coupling",
  "rf",
  "length",
  "superperiod",
  "method",
  "emitinput",
  "deltainput",
  "growthratesonly",
  "target",
  "integrate",
  "noWarning",
  "isRing",
  "forceCoupling",
  "emitxinput", /* For backward compatibility---identical to -emitInput */
  "emityinput",
  "emit0",
  "delta0",
  "distribution",
  "fixemity"};

#include "zibs.h"
void exitElegant(long status);
double IBSequations(double *x, long *invalid);
void IBSsimplexReport(double ymin, double *xmin, long pass, long evals, long dims);

void IBSIntegrate(double *exInteg, double *eyInteg, double *elInteg, int32_t *passInteg,
                  double *SdeltaInteg, double *SzInteg,
                  double *xRateInteg, double *yRateInteg, double *zRateInteg,
                  long integTurns, long integStepSize,
                  double P, double emitx, double emity,
                  double sigmaDelta, double sigmaz,
                  double particles,
                  double emitx0, double sigmaDelta0,
                  double xSRdampRate, double ySRdampRate, double longSRdampRate,
                  double coupling,
                  double *s, double *pCentral, double *betax, double *alphax, double *betay,
                  double *alphay, double *etax, double *etaxp, double *etay, double *etayp, long elements,
                  long superperiods, long verbosity, long isRing, long forceCoupling);

void IBSRateForDistribution(double particles,
                            long elements, long superperiods, long verbosity, long isRing,
                            double emitx0, double emity0, double sigmaDelta0, double sigmaz,
                            double *s, double *pCentral, double *betax, double *alphax, double *betay,
                            double *alphay, double *etax, double *etaxp, double *etay, double *etayp,
                            double *xRateVsSTotal, double *yRateVsSTotal, double *zRateVsSTotal,
                            double *xGrowthRateTotal, double *yGrowthRateTotal, double *zGrowthRateTotal, long isElegant,
                            double sTotal,
                            double *distData, long nDist);

/* global variables */
double *s, *pCentral, *betax, *alphax, *betay, *alphay, *etax, *etaxp, *etay, *etayp;
long isRing;

double *distHistData = NULL, dz;
long nDistData = 0;
double sigmaz0;

int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  char *inputfile, *outputfile;
  SDDS_DATASET twissPage, resultsPage;
  double particles, charge, length;
  long verbosity, noWarning, i, elements, superperiods, growthRatesOnly, forceCoupling, fixEmity;
  double pCentral0, I1, I2, I3, I4, I5, taux, tauy, taudelta;
  double EMeV;
  double emitx0 = 0, emitx, emitxInput, emityInput, emity, coupling, sigmaz;
  double sigmaDelta0 = 0, sigmaDelta, sigmaDeltaInput, xGrowthRate, yGrowthRate, zGrowthRate;
  double xGrowthRateInitial, yGrowthRateInitial, zGrowthRateInitial;
  /* double emitxOld, sigmaDeltaOld; */
  long method, converged;
  /* used in simplex minimization */
  double yReturn, *xGuess, *dxGuess, *xLowerLimit, *xUpperLimit;
  short *disable;
  long dimensions = 17, maxEvaluations = 500, maxPasses = 2;
  double target = 1e-6;
  int32_t integrationTurns, integrationStepSize;
  long integrationPoints = 0;
  double *exInteg = NULL, *eyInteg = NULL, *elInteg = NULL, *xRateInteg = NULL, *yRateInteg = NULL, *zRateInteg = NULL;
  double *SdeltaInteg = NULL, *SzInteg = NULL;
  int32_t *passInteg = NULL;
  unsigned long dummyFlags;
  double rfVoltage, rfHarmonic;
  double alphac, U0, circumference, energy;
  double *xRateVsS, *yRateVsS, *zRateVsS;
  char *distributionFile = NULL, *tColumn = NULL, *histColumn = NULL;

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&scanned, argc, argv);
  if (argc == 1)
    bomb(NULL, USAGE);

  xRateVsS = yRateVsS = zRateVsS = NULL;
  inputfile = NULL;
  outputfile = NULL;
  energy = 0;
  verbosity = 0;
  isRing = 1;
  particles = 0;
  charge = 0;
  coupling = emityInput = 0;
  forceCoupling = 1;
  fixEmity = 0;
  length = 0;
  superperiods = 1;
  method = 0;
  emitxInput = 0;
  sigmaDeltaInput = 0;
  growthRatesOnly = 0;
  integrationTurns = 0;
  rfVoltage = rfHarmonic = 0;
  noWarning = 0;
  emitx0 = 0;
  sigmaDelta0 = 0;
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
      case ISRING:
        if (scanned[i].n_items > 1) {
          get_long(&isRing, scanned[i].list[1]);
        } else {
          isRing = 1;
        }
        break;
      case CHARGE:
        get_double(&charge, scanned[i].list[1]);
        break;
      case EMITXINPUT:
        /* This is really the emitx+emity, not emitx */
        get_double(&emitxInput, scanned[i].list[1]);
        break;
      case EMITINPUT:
        get_double(&emitxInput, scanned[i].list[1]);
        break;
      case DELTAINPUT:
        get_double(&sigmaDeltaInput, scanned[i].list[1]);
        break;
      case EMIT0:
        get_double(&emitx0, scanned[i].list[1]);
        break;
      case DELTA0:
        get_double(&sigmaDelta0, scanned[i].list[1]);
        break;
      case LENGTH:
        get_double(&length, scanned[i].list[1]);
        length /= 1000; /* convert input length from mm to m */
        break;
      case COUPLING:
        get_double(&coupling, scanned[i].list[1]);
        break;
      case EMITYINPUT:
        get_double(&emityInput, scanned[i].list[1]);
        break;
      case FORCECOUPLING:
        get_long(&forceCoupling, scanned[i].list[1]);
        break;
      case FIXEMITY:
        get_long(&fixEmity, scanned[i].list[1]);
        break;
      case PARTICLES:
        get_double(&particles, scanned[i].list[1]);
        break;
      case SUPERPERIOD:
        get_long(&superperiods, scanned[i].list[1]);
        break;
      case METHOD:
        get_long(&method, scanned[i].list[1]);
        break;
      case GROWTHRATESONLY:
        growthRatesOnly = 1;
        break;
      case SET_TARGET:
        if (scanned[i].n_items != 2 ||
            !get_double(&target, scanned[i].list[1]) ||
            target < 0)
          bomb("invalid -target syntax", NULL);
        break;
      case RF:
        if (scanned[i].n_items < 2)
          bomb("invalid -rf syntax", NULL);
        scanned[i].n_items--;
        rfVoltage = rfHarmonic = 0;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "voltage", SDDS_DOUBLE, &rfVoltage, 1, 0,
                          "harmonic", SDDS_DOUBLE, &rfHarmonic, 1, 0,
                          NULL) ||
            rfVoltage <= 0 || rfHarmonic <= 0)
          bomb("invalid -rf syntax/values", "-rf=voltage=MV,harmonic=<value>");
        break;
      case SET_ENERGY:
        if (scanned[i].n_items != 2)
          bomb("invalid -energy syntax", NULL);
        if (!sscanf(scanned[i].list[1], "%lf", &energy) || energy <= 0)
          bomb("invalid -energy syntax/values", "-energy=<MeV>");
        break;
      case SET_INTEGRATE:
        if (scanned[i].n_items < 2)
          bomb("invalid -integrate syntax", NULL);
        integrationTurns = 0;
        integrationStepSize = 1;
        scanned[i].n_items--;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "turns", SDDS_LONG, &integrationTurns, 1, 0,
                          "stepsize", SDDS_LONG, &integrationStepSize, 1, 0,
                          NULL) ||
            integrationTurns <= 0 || integrationStepSize < 1)
          bomb("invalid -integrate syntax", NULL);
        break;
      case NO_WARNING:
        noWarning = 1;
        break;
      case DISTRIBUTION:
        if (scanned[i].n_items != 4 ||
            !strlen(distributionFile = scanned[i].list[1]) ||
            !strlen(tColumn = scanned[i].list[2]) ||
            !strlen(histColumn = scanned[i].list[3]))
          bomb("invalid -distribution syntax", NULL);
        if (!fexists(distributionFile))
          bomb("distribution file not found", NULL);
        break;
      default:
        fprintf(stderr, "Unknown option %s given", scanned[i].list[0]);
        exit(1);
        break;
      }
    } else {
      if (!inputfile)
        inputfile = scanned[i].list[0];
      else if (!outputfile)
        outputfile = scanned[i].list[0];
      else
        bomb("too many filenames given", NULL);
    }
  }
  if (charge && particles) {
    bomb("Options charge and particles cannot be both specified.", NULL);
  }
  if (!charge)
    charge = particles * e_mks;
  if (!particles) {
    /* command line input value is in units of nC */
    charge /= 1e9;
    particles = charge / e_mks;
  }
  if ((!coupling && !emityInput) || (coupling && emityInput))
    bomb("Give -coupling or -emityInput (but not both)", NULL);
  if (!length && !rfVoltage && !distributionFile)
    bomb("Specify either the bunch length or the rf voltage.", NULL);

  if (growthRatesOnly && integrationTurns) {
    growthRatesOnly = 0;
    if (!noWarning)
      fprintf(stdout, "*Warning* -growthRatesOnly option is incompatiable with -integrate option. The -growthRatesOnly will be disabled.\n");
  }

  if (!growthRatesOnly && !integrationTurns && !noWarning)
    fprintf(stdout, "*Warning* The growth rate contribution columns in the results file will be those calculated from the equilibrium (or final) condition.\n");
  if (integrationTurns && !isRing) {
    integrationTurns = 0;
    fprintf(stdout, "*Warning* -isRing=0 is incompatiable with -integrate option. The -integrate will be disabled.\n");
  }
  if (energy && !isRing) {
    energy = 0;
    fprintf(stdout, "*Warning* you can not scale energy for linac beam. Scaling will be disabled.\n");
  }
  if (integrationTurns) {
    if (fixEmity && !noWarning)
      fprintf(stdout, "*Warning*: -fixEmity=1 is ignored in integration mode.\n");
  } else {
    if (!forceCoupling && !noWarning)
      fprintf(stdout, "*Warning*: -forceCoupling=0 is ignored in non-integration mode. Perhaps -fixEmity=1 will do?\n");
    forceCoupling = 1;
  }

  /***************************************************\
   * get parameter information from first input file  *
   \***************************************************/
  if (verbosity)
    fprintf(stdout, "Opening \"%s\" for checking presence of parameters.\n", inputfile);
  if (!SDDS_InitializeInput(&twissPage, inputfile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  /* read first page of input file to get parameters 
     I1 I2 I3 I4 I5.
     Check presence of first radiation integral.
     */
  SDDS_ReadPage(&twissPage);
  /* parameter Type */
  if (isRing) {
    switch (SDDS_CheckParameter(&twissPage, "I1", NULL, SDDS_DOUBLE, verbosity ? stdout : NULL)) {
    case SDDS_CHECK_NONEXISTENT:
      if (verbosity)
	fprintf(stdout, "\tParameter I1 not found in input file.\n");
      break;
    case SDDS_CHECK_WRONGTYPE:
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      exitElegant(1);
      break;
    case SDDS_CHECK_OKAY:
      break;
    default:
      fprintf(stdout, "Unexpected result from SDDS_CheckParameter routine while checking parameter Type.\n");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      exitElegant(1);
      break;
    }
  }
  if (verbosity)
    fprintf(stdout, "Opening \"%s\" for writing...\n", outputfile);
  if (!SDDS_InitializeOutput(&resultsPage, SDDS_BINARY, 1, "Intra-beam scattering rates",
                             "Intra-beam scattering rates", outputfile) ||
      !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "pCentral", NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (isRing && 
      (!SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "I1", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "I2", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "I3", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "I4", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "I5", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "taux", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "tauy", NULL) ||
       !SDDS_TransferParameterDefinition(&resultsPage, &twissPage, "taudelta", NULL)))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (0 > SDDS_DefineParameter(&resultsPage, "Superperiods", NULL, NULL, "Superperiods", NULL, SDDS_LONG, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Energy", "E", "MeV", "Total Energy", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Particles", NULL, NULL, "Particles", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "Charge", NULL, "nC", "Charge", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "PeakCurrent", "I$bp$n", "A", "Peak Current", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "RfVoltage", NULL, "MV", "Rf Voltage", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "xGrowthRateInitial", "g$bIBS,x$n", "1/s", "Initial IBS emittance growth rate in the horizontal plane", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "yGrowthRateInitial", "g$bIBS,y$n", "1/s", "Initial IBS emittance growth rate in the vertical plane", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "zGrowthRateInitial", "g$bIBS,z$n", "1/s", "Initial IBS emittance growth rate in the longitudinal plane", NULL, SDDS_DOUBLE, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (0 > SDDS_DefineParameter(&resultsPage, "Convergence", NULL, NULL, "Convergence state of emittance calculations", NULL, SDDS_STRING, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "emitx0", "$ge$r$bx,0$n", "m", "Equilibrium horizontal emittance with no coupling and no IBS", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "emitxInput", "$ge$r$bx,Input$n", "m", "Initial horizontal emittance with coupling", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "emityInput", "$ge$r$by,Input$n", "m", "Initial vertical emittance with coupling", NULL, SDDS_DOUBLE, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* requested equilibrium emittances or integrate emittances turn-by-turn*/
  if (!growthRatesOnly) {
    if (0 > SDDS_DefineParameter(&resultsPage, "xGrowthRate", "g$bIBS,x$n", "1/s", "IBS emittance growth rate in the horizontal plane", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "yGrowthRate", "g$bIBS,y$n", "1/s", "IBS emittance growth rate in the vertical plane", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "zGrowthRate", "g$bIBS,z$n", "1/s", "IBS emittance growth rate in the longitudinal plane", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "emitx", "$ge$r$bx$n", "m", "Horizontal emittance with coupling and with IBS", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "emity", "$ge$r$by$n", "m", "Vertical emittance with coupling and with IBS", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "sigmaDelta", "$gs$r$bd$n", NULL, "Relative momentum spread with IBS", NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&resultsPage, "sigmaz", "$gs$r$bz$n", "m", "Bunch length with IBS", NULL, SDDS_DOUBLE, NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (0 > SDDS_DefineParameter(&resultsPage, "sigmaDelta0", "$gs$r$bd,0$n", NULL, "Equilibrium relative momentum spread without IBS", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "sigmaDeltaInput", "$gs$r$bd,0$n", NULL, "Initial relative momentum spread", NULL, SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(&resultsPage, "sigmaz0", "$gs$r$bz,0$n", "m", "Bunch length without IBS", NULL, SDDS_DOUBLE, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (verbosity)
    fprintf(stdout, "Opening for reading \"%s\"\n", inputfile);
  if (!SDDS_InitializeInput(&twissPage, inputfile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (integrationTurns) {
    if (SDDS_DefineColumn(&resultsPage, "ex", "$ge$r$bx$n", "m", "Horizontal Emittance", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "ey", "$ge$r$by$n", "m", "Vertical Emittance", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "el", "$ge$r$bl$n", "s", "Longitudinal Emittance", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "Sdelta", "$gs$bd$n$r", "", "Fractional RMS Energy Spread", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "Sz", "$gs$r$bz$n", "m", "RMS Bunch Length", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "IBSRatex", NULL, "1/s", "Horizontal IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "IBSRatey", NULL, "1/s", "Vertical IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "IBSRatel", NULL, "1/s", "Longitudinal IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "Pass", NULL, NULL, NULL, NULL, SDDS_LONG, 0) < 0)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    integrationPoints = integrationTurns / integrationStepSize + 1;
    if (!(exInteg = SDDS_Malloc(sizeof(*exInteg) * integrationPoints)) ||
        !(eyInteg = SDDS_Malloc(sizeof(*eyInteg) * integrationPoints)) ||
        !(elInteg = SDDS_Malloc(sizeof(*elInteg) * integrationPoints)) ||
        !(SdeltaInteg = SDDS_Malloc(sizeof(*SdeltaInteg) * integrationPoints)) ||
        !(SzInteg = SDDS_Malloc(sizeof(*SzInteg) * integrationPoints)) ||
        !(xRateInteg = SDDS_Malloc(sizeof(*xRateInteg) * integrationPoints)) ||
        !(yRateInteg = SDDS_Malloc(sizeof(*yRateInteg) * integrationPoints)) ||
        !(zRateInteg = SDDS_Malloc(sizeof(*zRateInteg) * integrationPoints)) ||
        !(passInteg = SDDS_Malloc(sizeof(*passInteg) * integrationPoints)))
      bomb("memory allocation failure (integration arrays)", NULL);
  } else {
    if (SDDS_DefineColumn(&resultsPage, "s", NULL, "m", "Position", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "dIBSRatex", NULL, "1/s", "Local Horizontal IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "dIBSRatey", NULL, "1/s", "Local Vertical IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&resultsPage, "dIBSRatel", NULL, "1/s", "Local Longitudinal IBS Emittance Growth Rate", NULL, SDDS_DOUBLE, 0) < 0)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_WriteLayout(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (distributionFile) {
    /* Read in the distributionFile
     * Store distribution in array distHistData, spacing in variable dz, # points in variable nDistData
     */
    SDDS_DATASET SDDS_df;
    long i;
    double sum, sum2, *distZData, t0;
    if (!SDDS_InitializeInput(&SDDS_df, distributionFile) || !SDDS_ReadPage(&SDDS_df))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if ((nDistData = SDDS_RowCount(&SDDS_df)) <= 2)
      bomb("too few rows in distribution file", NULL);
    if (SDDS_CheckColumn(&SDDS_df, tColumn, "s", SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OKAY)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!(distZData = SDDS_GetColumnInDoubles(&SDDS_df, tColumn)) ||
        !(distHistData = SDDS_GetColumnInDoubles(&SDDS_df, histColumn)))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    SDDS_Terminate(&SDDS_df);

    /* Normalize the distribution to unit sum */
    for (i = sum = 0; i < nDistData; i++)
      sum += distHistData[i];
    if (sum <= 0)
      bomb("distribution sum is not positive", NULL);
    t0 = distZData[0];
    for (i = 0; i < nDistData; i++) {
      distHistData[i] /= sum;
      distZData[i] = c_mks * (distZData[i] - t0); /* convert time to distance */
    }

    /* compute the rms bunch duration, which is used as a scaling parameter and for output */
    for (i = sum = 0; i < nDistData; i++)
      sum += distHistData[i] * distZData[i];
    for (i = sum2 = 0; i < nDistData; i++)
      sum2 += distHistData[i] * sqr(distZData[i] - sum);
    sigmaz0 = sqrt(sum2);

    /* Find the spacing of the z points */
    if ((dz = distZData[1] - distZData[0]) == 0) {
      fprintf(stderr, "Error: time data spacing seems not to be monotonic:\n");
      for (i = 0; i < nDistData; i++)
        fprintf(stderr, "%21.15e\n", distZData[i]);
      bomb("Check spacing of time data in distribution file. Must be monotonically increasing or decreasing.", NULL);
    }
    for (i = 2; i < nDistData; i++) {
      double dz1;
      dz1 = distZData[i] - distZData[i - 1];
      if (fabs((dz1 - dz) / dz) > 1e-3)
        bomb("Check spacing of time data in distribution file. Must change monotonically and be approximately equispaced.", NULL);
    }
    dz = fabs(dz);
    free(distZData);
  }

  while (SDDS_ReadPage(&twissPage) > 0) {
    if (!SDDS_GetParameters(&twissPage,
                            "pCentral", &pCentral0,
                            NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    I1 = I2 = I3 = I4 = I5 = 0;
    taux = tauy = taudelta = 1;
    alphac = U0 = 0;
    if (isRing && !SDDS_GetParameters(&twissPage,
				      "I1", &I1,
				      "I2", &I2,
				      "I3", &I3,
				      "I4", &I4,
				      "I5", &I5,
				      "taux", &taux,
				      "tauy", &tauy,
				      "taudelta", &taudelta,
				      "alphac", &alphac,
				      "U0", &U0,
				      NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    EMeV = sqrt(sqr(pCentral0) + 1) * me_mev;
    elements = SDDS_CountRowsOfInterest(&twissPage);
    s = SDDS_GetColumnInDoubles(&twissPage, "s");
    pCentral = SDDS_GetColumnInDoubles(&twissPage, "pCentral0");
    circumference = s[elements - 1] * superperiods;
    U0 *= superperiods;
    if (energy != 0) {
      /* scale to new energy */
      pCentral0 = sqrt(sqr(energy / me_mev) - 1);
      taux /= ipow3(energy / EMeV);
      tauy /= ipow3(energy / EMeV);
      taudelta /= ipow3(energy / EMeV);
      U0 *= ipow4(energy / EMeV);
      for (i = 0; i < elements; i++)
        pCentral[i] = pCentral0;
      EMeV = energy;
    }

    if (!length && !distributionFile && U0 > rfVoltage)
      bomb("energy loss per turn is greater than rf voltage", NULL);
    betax = SDDS_GetColumnInDoubles(&twissPage, "betax");
    betay = SDDS_GetColumnInDoubles(&twissPage, "betay");
    alphax = SDDS_GetColumnInDoubles(&twissPage, "alphax");
    alphay = SDDS_GetColumnInDoubles(&twissPage, "alphay");
    etax = SDDS_GetColumnInDoubles(&twissPage, "etax");
    etaxp = SDDS_GetColumnInDoubles(&twissPage, "etaxp");
    etay = SDDS_GetColumnInDoubles(&twissPage, "etay");
    etayp = SDDS_GetColumnInDoubles(&twissPage, "etayp");

    /* emitx0 and sigmaDelta0 are unperturbed quantities 
       (i.e. no coupling and no IBS) that
       zibs requires to internally calculate the quantum excitation.
       (zibs doesn't use the radiation integrals but should!) 
       */
    if (isRing) {
      if (!emitx0)
	emitx0 = 55.0 / (32. * sqrt(3.)) * hbar_mks * sqr(pCentral0) / (me_mks * c_mks) * I5 / (I2 - I4);
      if (!sigmaDelta0)
	sigmaDelta0 = sqrt(55.0 / (32. * sqrt(3.)) * hbar_mks * sqr(pCentral0) / (me_mks * c_mks) * I3 / (2 * I2 + I4));
    }
    /* use unperturbed quantities in no input supplied. */
    if (!sigmaDeltaInput)
      sigmaDeltaInput = sigmaDelta0;
    if (!emitxInput)
      emitxInput = emitx0 / (1 + coupling * taux / tauy);
    else
      /* The emitxInput value is really emit=emitx+emity */
      emitxInput = emitxInput / (1 + coupling * taux / tauy);
    if (!emityInput)
      emityInput = emitxInput * coupling;
    else
      coupling = emityInput / emitxInput;
    sigmaDelta = sigmaDeltaInput;
    if (length)
      sigmaz0 = length;
    else if (!nDistData) {
      /* compute length in m from rf voltage, energy spread, etc */
      sigmaz0 =
        circumference * sigmaDelta *
        sqrt(alphac * EMeV / (PIx2 * rfHarmonic * sqrt(sqr(rfVoltage) - sqr(U0))));
    }
    sigmaz = sigmaz0;
    emity = emityInput;
    emitx = emitxInput;

    if (integrationPoints) {
      IBSIntegrate(exInteg, eyInteg, elInteg, passInteg,
                   SdeltaInteg, SzInteg,
                   xRateInteg, yRateInteg, zRateInteg,
                   integrationTurns, integrationStepSize,
                   pCentral0, emitx, emity, sigmaDelta, sigmaz, particles,
                   emitx0, sigmaDelta0, 2. / taux, 2. / tauy, 2. / taudelta, coupling,
                   s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp, elements,
                   superperiods, verbosity, isRing, forceCoupling);
    } else {
      if (!(xRateVsS = SDDS_Realloc(xRateVsS, sizeof(*xRateVsS) * elements)) ||
          !(yRateVsS = SDDS_Realloc(yRateVsS, sizeof(*yRateVsS) * elements)) ||
          !(zRateVsS = SDDS_Realloc(zRateVsS, sizeof(*zRateVsS) * elements)))
        bomb("memory allocation failure", NULL);
    }

    /* This call is to get the initial growth rates for writing to results file.
       This applies for any running option selected in the commandline */
    if (nDistData)
      IBSRateForDistribution(particles, elements, superperiods, verbosity, isRing,
                             emitx, emity, sigmaDelta, sigmaz,
                             s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
                             NULL, NULL, NULL,
                             &xGrowthRateInitial, &yGrowthRateInitial, &zGrowthRateInitial, 0, -1,
                             distHistData, nDistData);
    else
      IBSRate(particles, elements, superperiods, verbosity, isRing,
              emitx, emity, sigmaDelta, sigmaz,
              s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
              NULL, NULL, NULL,
              &xGrowthRateInitial, &yGrowthRateInitial, &zGrowthRateInitial, 0, -1);

    /* iterating for equilibrium emittances and final growth rates */
    if (!integrationTurns && !growthRatesOnly && isRing) {
      if (verbosity > 1) {
        fprintf(stdout, "Starting values:\nemitx: %10.5g sigmaDelta %10.5g.\n", emitx, sigmaDelta);
      }
      /* emitxOld = emitx; */
      /* sigmaDeltaOld = sigmaDelta; */
      xGuess = (double *)malloc(sizeof(double) * dimensions);
      dxGuess = (double *)malloc(sizeof(double) * dimensions);
      xLowerLimit = (double *)malloc(sizeof(double) * dimensions);
      xUpperLimit = (double *)malloc(sizeof(double) * dimensions);
      disable = (short *)malloc(sizeof(short) * dimensions);
      xGuess[0] = MAX(emitx, emitx0 / (1 + coupling));
      xGuess[1] = MAX(sigmaDelta, sigmaDelta0);
      dxGuess[0] = emitx * 0.1;
      dxGuess[1] = sigmaDelta * 0.1;
      xLowerLimit[0] = emitx0 / (1 + coupling);
      xLowerLimit[1] = sigmaDelta0;
      xUpperLimit[0] = emitx0 / (1 + coupling) * 200;
      xUpperLimit[1] = MIN(sigmaDelta0 * 100, 1.0);
      /* assign other variables to array which are not supoosed
         to be varied by simplex minimization
         */
      xGuess[2] = pCentral0;
      xGuess[3] = emity;
      xGuess[4] = sigmaz0;
      xGuess[5] = particles;
      xGuess[6] = emitx0;
      xGuess[7] = sigmaDelta0;
      xGuess[8] = taux;
      xGuess[9] = tauy;
      xGuess[10] = taudelta;
      xGuess[11] = coupling;
      xGuess[12] = elements;
      xGuess[13] = superperiods;
      xGuess[14] = verbosity;
      xGuess[15] = forceCoupling;
      xGuess[16] = fixEmity;
      xLowerLimit[2] = pCentral0;
      xLowerLimit[3] = emity;
      xLowerLimit[4] = sigmaz0;
      xLowerLimit[5] = particles;
      xLowerLimit[6] = emitx0;
      xLowerLimit[7] = sigmaDelta0;
      xLowerLimit[8] = taux;
      xLowerLimit[9] = tauy;
      xLowerLimit[10] = taudelta;
      xLowerLimit[11] = coupling;
      xLowerLimit[12] = elements;
      xLowerLimit[13] = superperiods;
      xLowerLimit[14] = verbosity;
      xLowerLimit[15] = forceCoupling;
      xLowerLimit[16] = fixEmity;
      xUpperLimit[2] = pCentral0;
      xUpperLimit[3] = emity;
      xUpperLimit[4] = sigmaz0;
      xUpperLimit[5] = particles;
      xUpperLimit[6] = emitx0;
      xUpperLimit[7] = sigmaDelta0;
      xUpperLimit[8] = taux;
      xUpperLimit[9] = tauy;
      xUpperLimit[10] = taudelta;
      xUpperLimit[11] = coupling;
      xUpperLimit[12] = elements;
      xUpperLimit[13] = superperiods;
      xUpperLimit[14] = verbosity;
      xUpperLimit[15] = forceCoupling;
      xUpperLimit[16] = fixEmity;
      disable[0] = 0;
      disable[1] = 0;
      for (i = 2; i < dimensions; i++) {
        dxGuess[i] = 0.0;
        disable[i] = 1;
      }
      if (verbosity) {
        fprintf(stdout, "Doing simplex minimization...\n");
      }
      simplexMin(&yReturn, xGuess, dxGuess, xLowerLimit, xUpperLimit, disable, dimensions,
                 target, target / 100.0, IBSequations, verbosity ? IBSsimplexReport : NULL,
                 maxEvaluations, maxPasses, 12, 3.0, 1.0, 0);
      /* final answers */
      emitx = xGuess[0];
      sigmaDelta = xGuess[1];
      if (!fixEmity)
        emity = emitx * coupling;
      sigmaz = sigmaz0 * (sigmaDelta / sigmaDelta0);
      dz *= (sigmaDelta / sigmaDelta0);
    }

    /* calculate growth rates contributions at equilibrium or
     just one time (-growthRateOnly option) */
    if (!integrationPoints) {
      if (nDistData)
        IBSRateForDistribution(particles, elements, superperiods, verbosity, isRing,
                               emitx, emity, sigmaDelta, sigmaz,
                               s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
                               xRateVsS, yRateVsS, zRateVsS,
                               &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1,
                               distHistData, nDistData);
      else
        IBSRate(particles, elements, superperiods, verbosity, isRing,
                emitx, emity, sigmaDelta, sigmaz,
                s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
                xRateVsS, yRateVsS, zRateVsS,
                &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1);
    } else {
      /* final growth rates and emittances after integration */
      xGrowthRate = xRateInteg[integrationPoints - 1];
      yGrowthRate = yRateInteg[integrationPoints - 1];
      zGrowthRate = zRateInteg[integrationPoints - 1];
      emitx = exInteg[integrationPoints - 1];
      emity = eyInteg[integrationPoints - 1];
      sigmaDelta = SdeltaInteg[integrationPoints - 1];
      sigmaz = SzInteg[integrationPoints - 1];
    }

    converged = 1;

    if (0 > SDDS_StartPage(&resultsPage, integrationPoints ? integrationPoints : elements) ||
        !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Convergence", converged ? "Emittance converged" : "Emittance did not converge",
                            "pCentral", pCentral0, "RfVoltage", rfVoltage,
                            "Energy", EMeV,
                            "Particles", particles,
                            "Charge", (1e9 * charge),
                            "PeakCurrent", (charge * c_mks / (sqrt(2 * PI) * sigmaz)),
                            "Superperiods", superperiods,
                            "emitx0", emitx0,
                            "emitxInput", emitxInput,
                            "emityInput", emityInput,
                            "xGrowthRateInitial", xGrowthRateInitial,
                            "yGrowthRateInitial", yGrowthRateInitial,
                            "zGrowthRateInitial", zGrowthRateInitial,
                            "sigmaDeltaInput", sigmaDeltaInput,
                            "sigmaDelta0", sigmaDelta0,
                            "sigmaz0", sigmaz0, NULL) ||
        (isRing &&
	 !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "I1", I1,
                            "I2", I2,
                            "I3", I3,
                            "I4", I4,
                            "I5", I5,
                            "taux", taux,
                            "tauy", tauy,
			     "taudelta", taudelta)) ||
        (!growthRatesOnly &&
         !SDDS_SetParameters(&resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             "xGrowthRate", xGrowthRate,
                             "yGrowthRate", yGrowthRate,
                             "zGrowthRate", zGrowthRate,
                             "emitx", emitx,
                             "emity", emity,
                             "sigmaDelta", sigmaDelta,
                             "sigmaz", sigmaz, NULL)) ||
        (integrationPoints &&
         (!SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, exInteg, integrationPoints, "ex") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, eyInteg, integrationPoints, "ey") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, elInteg, integrationPoints, "el") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, SdeltaInteg, integrationPoints, "Sdelta") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, SzInteg, integrationPoints, "Sz") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, xRateInteg, integrationPoints, "IBSRatex") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, yRateInteg, integrationPoints, "IBSRatey") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, zRateInteg, integrationPoints, "IBSRatel") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, passInteg, integrationPoints, "Pass"))) ||
        (!integrationPoints &&
         (!SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, s, elements, "s") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, xRateVsS, elements, "dIBSRatex") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, yRateVsS, elements, "dIBSRatey") ||
          !SDDS_SetColumn(&resultsPage, SDDS_SET_BY_NAME, zRateVsS, elements, "dIBSRatel"))) ||
        !SDDS_WritePage(&resultsPage))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_Terminate(&twissPage) || !SDDS_Terminate(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return (0);
}

double IBSequations(double *x, long *invalid) {
  double emitx, sigmaDelta;
  /* double pCentral0; */
  double emity, sigmaz, sigmaz0, particles, emitx0,
    sigmaDelta0, taux, tauy, taudelta, coupling;
  long elements, superperiods, verbosity, fixEmity;
  double xGrowthRate, yGrowthRate, zGrowthRate;
  double a, b, c, d, e, f, func1, func2;
  /* long forceCoupling;   */

  emitx = x[0];
  sigmaDelta = x[1];
  /* pCentral0 = x[2]; */
  sigmaz0 = x[4];
  particles = x[5];
  emitx0 = x[6];
  sigmaDelta0 = x[7];
  taux = x[8];
  tauy = x[9];
  taudelta = x[10];
  coupling = x[11];
  elements = x[12];
  superperiods = x[13];
  verbosity = x[14];
  /* forceCoupling = x[15]; */ /* not used */
  fixEmity = x[16];

  /* zap code requires damping rate for horizontal and longitudinal emittances
       which is twice the damping rate for one coordinate.
       The quantities emitx0 and 2/taux are used to determined
       the quantum excitation inside the zap algortihm.
     */

  /* During iterations maintain the ratios
       sigmaz/sigmaDelta and emity/emitx constant.
       Their original values are
       sigmaz0/sigmaDelta0 and emityInput/emitxInput respectively.       
       */

  /* equations to solve (using ZAP manual notation) and throwing in coupling
       terms.
       damping term     quantum excitation      IBS terms
                       (a constant)                                              
        SR             SR   0    1              IBS       1         IBS       k  
       g   e        = g    e   -----       +   g    e   -----  +   g    e   -----
        x   x          x    x  1 + k            x    x  1 + k       y    y  1 + k
     
       The quantum excitation term is a constant and is reduced in the x plane 
       because of coupling. Also the IBS growth rate in x is reduced because of
       coupling. 
     
       In the y-plane:
       damping term     quantum excitation      IBS terms
                       (a constant)                                              
        SR             SR   0    k              IBS       1         IBS       k  
       g   e        = g    e   -----       +   g    e   -----  +   g    e   -----
        y   y          y    y  1 + k            y    y  1 + k       x    x  1 + k
     
       In the longitudinal plane,
       damping term     quantum excitation      IBS term
                       (a constant)                                              
        SR      2      SR        2              IBS      2
       g   delta    = g    delta0          +   g    delta
        z              z                        z    
     
       Assume that g^IBS will have the approximate dependence which will help finding
       solutions:
                    Input Input          2  
                   e    e     deltaInput          
        IBS         x    y                          1
       g    = G' ----------------------- = G ----------------
                               2                           2               
                   e  e   delta                e  e   delta 
                    x  y                        x  y        
       where G doesn't change much during the iterations and where
       G' = g^IBS on the very first calculation.

       Correspondence with our variables:
       g^SR_x    -> 2/taux
       g^IBS_x   -> xGrowthRate
       e^0_x     -> emitx0
       k         -> coupling

       N.B.: These equations are misleading for x and y planes. The correct thing to 
       do is to sum the equations and solve with the requirement that emity=coupling*emitx.
       See technical note by R. Lindberg.

       One can ignore the y equation and simply put emity = coupling * emitx 
       during and after the calculation of emitx.
       */

  if (!fixEmity)
    /* enforce emity/emitx ratio. Otherwise, emity is constant */
    emity = emitx * coupling;
  else
    emity = x[3];

  sigmaz = sigmaz0 * (sigmaDelta / sigmaDelta0);
  if (nDistData)
    IBSRateForDistribution(particles, elements, superperiods, verbosity, isRing,
                           emitx, emity, sigmaDelta, sigmaz,
                           s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
                           NULL, NULL, NULL,
                           &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1,
                           distHistData, nDistData);
  else
    IBSRate(particles, elements, superperiods, verbosity, isRing,
            emitx, emity, sigmaDelta, sigmaz,
            s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
            NULL, NULL, NULL,
            &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1);
  a = -2. / taux * emitx - 2. / tauy * emity;
  b = 2. / taux * emitx0;
  c = xGrowthRate * emitx + yGrowthRate * emity;
  d = -2. / taudelta * sqr(sigmaDelta);
  e = 2. / taudelta * sqr(sigmaDelta0);
  f = zGrowthRate * sqr(sigmaDelta);
  func1 = a + b + c;
  func2 = d + e + f;
  *invalid = 0;
  /* returns an equation evaluation that, at the end of convergence, should be zero. */
  return (sqr(func1 / b) + sqr(func2 / e));
}

void IBSsimplexReport(double ymin, double *xmin, long pass, long evals, long dims) {
  fprintf(stdout, "IBS Simplex Report:\nMinimum value obtained: %e.\n", ymin);
  fprintf(stdout, "    %ld passes, %ld evaluations.\n", pass, evals);
  fprintf(stdout, "    emitx = %e   sigmaDelta = %e.\n", xmin[0], xmin[1]);
  return;
}

void IBSIntegrate(double *exInteg, double *eyInteg, double *elInteg, int32_t *passInteg,
                  double *SdeltaInteg, double *SzInteg,
                  double *xRateInteg, double *yRateInteg, double *zRateInteg,
                  long integTurns, long integStepSize,
                  double P, double emitx, double emity,
                  double sigmaDelta, double sigmaz,
                  double particles,
                  double emitx0, double sigmaDelta0,
                  double xSRdampRate, double ySRdampRate, double longitSRdampRate,
                  double coupling,
                  double *s, double *pCentral, double *betax, double *alphax, double *betay,
                  double *alphay, double *etax, double *etaxp, double *etay, double *etayp, long elements,
                  long superperiods, long verbosity, long isRing, long forceCoupling) {
  long turn, slot;
  double dT, gamma, vz, emitz, zRatio;
  double xGrowthRate, yGrowthRate, zGrowthRate, emitz0;

  gamma = sqrt(sqr(P) + 1);
  vz = c_mks * P / gamma;
  dT = s[elements - 1] * superperiods * integStepSize / vz;

  emitz = sigmaDelta * sigmaz;
  emitz0 = sigmaDelta0 * sigmaz;
  zRatio = sigmaDelta / sigmaz;
  for (turn = slot = 0; turn < integTurns; turn += integStepSize, slot++) {
    if (verbosity) {
      fprintf(stdout, "Integrating turn %ld, emitx=%le, emity=%le, emitz=%le\n",
              turn, emitx, emity, emitz);
      fflush(stdout);
    }
    exInteg[slot] = emitx;
    eyInteg[slot] = emity;
    elInteg[slot] = emitz / vz;
    SdeltaInteg[slot] = sigmaDelta;
    SzInteg[slot] = sigmaz;
    passInteg[slot] = turn;
    if (nDistData)
      IBSRateForDistribution(particles, elements, superperiods, verbosity, isRing,
                             emitx, emity, sigmaDelta, sigmaz,
                             s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
                             NULL, NULL, NULL,
                             &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1,
                             distHistData, nDistData);
    else
      IBSRate(particles, elements, superperiods, verbosity, isRing,
              emitx, emity, sigmaDelta, sigmaz,
              s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
              NULL, NULL, NULL,
              &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1);
    xRateInteg[slot] = xGrowthRate;
    yRateInteg[slot] = yGrowthRate;
    zRateInteg[slot] = zGrowthRate;
    emitx += (xGrowthRate - xSRdampRate) * emitx * dT + xSRdampRate * emitx0 * dT / (1 + coupling*ySRdampRate/xSRdampRate);
    emity += (yGrowthRate - ySRdampRate) * emity * dT + ySRdampRate * emitx0 * coupling * dT / (1 + coupling*ySRdampRate/xSRdampRate);
    emitz += (zGrowthRate - longitSRdampRate) * emitz * dT + longitSRdampRate * emitz0 * dT;
    if (forceCoupling) {
      /* Not sure this is right if Jx!=1 */
      emitx = (emitx + emity) / (1 + coupling);
      emity = emitx * coupling;
    }
    sigmaDelta = sqrt(emitz * zRatio);
    sigmaz = emitz / sigmaDelta;
  }
  exInteg[slot] = emitx;
  eyInteg[slot] = emity;
  elInteg[slot] = emitz / vz;
  SdeltaInteg[slot] = sigmaDelta;
  SzInteg[slot] = sigmaz;
  passInteg[slot] = turn;
  IBSRate(particles, elements, superperiods, verbosity, isRing,
          emitx, emity, sigmaDelta, sigmaz,
          s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
          NULL, NULL, NULL,
          &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1);
  xRateInteg[slot] = xGrowthRate;
  yRateInteg[slot] = yGrowthRate;
  zRateInteg[slot] = zGrowthRate;
}

void IBSRateForDistribution(double particles,
                            long elements, long superperiods, long verbosity, long isRing,
                            double emitx, double emity, double sigmaDelta, double sigmaz,
                            double *s, double *pCentral, double *betax, double *alphax, double *betay,
                            double *alphay, double *etax, double *etaxp, double *etay, double *etayp,
                            double *xRateVsSTotal, double *yRateVsSTotal, double *zRateVsSTotal,
                            double *xGrowthRateTotal, double *yGrowthRateTotal, double *zGrowthRateTotal, long isElegant,
                            double sTotal, double *distData, long nDist) {
  long i, j;
  double *xRateVsS, *yRateVsS, *zRateVsS;
  double xGrowthRate, yGrowthRate, zGrowthRate;

  xGrowthRate = yGrowthRate = zGrowthRate = 0;
  xRateVsS = yRateVsS = zRateVsS = NULL;

  if (xRateVsSTotal) {
    xRateVsS = tmalloc(sizeof(*xRateVsS) * elements);
    for (i = 0; i < elements; i++)
      xRateVsSTotal[i] = 0;
  }
  if (yRateVsSTotal) {
    yRateVsS = tmalloc(sizeof(*yRateVsS) * elements);
    for (i = 0; i < elements; i++)
      yRateVsSTotal[i] = 0;
  }
  if (zRateVsSTotal) {
    zRateVsS = tmalloc(sizeof(*zRateVsS) * elements);
    for (i = 0; i < elements; i++)
      zRateVsSTotal[i] = 0;
  }

  *xGrowthRateTotal = *yGrowthRateTotal = *zGrowthRateTotal = 0;

  for (i = 0; i < nDist; i++) {
    xGrowthRate = yGrowthRate = zGrowthRate = 0;
    IBSRate(particles * distData[i], elements, superperiods, verbosity, isRing,
            emitx, emity, sigmaDelta,
            sigmaz / sigmaz0 * dz,
            s, pCentral, betax, alphax, betay, alphay, etax, etaxp, etay, etayp,
            xRateVsS, yRateVsS, zRateVsS,
            &xGrowthRate, &yGrowthRate, &zGrowthRate, 0, -1);

    *xGrowthRateTotal += xGrowthRate * distData[i];
    *yGrowthRateTotal += yGrowthRate * distData[i];
    *zGrowthRateTotal += zGrowthRate * distData[i];
    if (xRateVsSTotal)
      for (j = 0; j < elements; j++)
        xRateVsSTotal[j] += xRateVsS[j] * distData[i];
    if (yRateVsSTotal)
      for (j = 0; j < elements; j++)
        yRateVsSTotal[j] += yRateVsS[j] * distData[i];
    if (zRateVsSTotal)
      for (j = 0; j < elements; j++)
        zRateVsSTotal[j] += zRateVsS[j] * distData[i];
  }
  *xGrowthRateTotal *= 2 * sqrt(PI);
  *yGrowthRateTotal *= 2 * sqrt(PI);
  *zGrowthRateTotal *= 2 * sqrt(PI);
  if (xRateVsSTotal) {
    for (j = 0; j < elements; j++)
      xRateVsSTotal[j] *= 2 * sqrt(PI);
    free(xRateVsS);
  }
  if (yRateVsSTotal) {
    for (j = 0; j < elements; j++)
      yRateVsSTotal[j] *= 2 * sqrt(PI);
    free(yRateVsS);
  }
  if (zRateVsSTotal) {
    for (j = 0; j < elements; j++)
      zRateVsSTotal[j] *= 2 * sqrt(PI);
    free(zRateVsS);
  }
  if (verbosity > 4) {
    printf("emitx = %le, emity=%le, sigmaz/sigmaz0 = %le\n", emitx, emity, sigmaz / sigmaz0);
    printf("rates = %le, %le, %le\n", *xGrowthRateTotal, *yGrowthRateTotal, *zGrowthRateTotal);
  }
}
