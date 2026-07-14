/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* Copyright 1994 by Louis Emery and Argonne National Laboratory,
 * all rights reserved.
 * Solves the Haissinski equation for density distribution
 * of a bunch.
 * Louis Emery, 2000
 */

#include <stdio.h>
#include "mdb.h"
#include "scan.h"
#include "match_string.h"
#include "SDDS.h"
#include "constants.h"
#include "fftpackC.h"

static char *USAGE1 = "haissinski <twissFile> <resultsFile>\n\
 {-semaphoreFile=<filename>}\n\
 {-wakeFunction=<file>,tColumn=<name>,wColumn=<name> |\n\
  -model=[L=<Henry>|Zn=<Ohms>],R=<Ohm> |\n\
  -BBResonator=Rs=<Ohm>,frequency=<Hz>,Q=<value>[,wall=<Ohms>]} \n\
 {-charge=<C>|-particles=<value>|-bunchCurrent=<A>}\n\
 {-steps=<numberOfChargeSteps>} {-outputLastStepOnly}\n\
 {-RF=Voltage=<V>,harmonic=<value>,phase=<radians>|-length=<s>}\n\
 {-harmonicCavity=Voltage=<V>,harmonicFactor=<harmonicFactor>,phase=<radians>}\n\
 {-superPeriods=<number>} {-energy=<GeV>} \n\
 -integrationParameters=deltaTime=<s>,points=<number>,\n\
iterations=<number>,fraction=<value>,tolerance=<value> \n\
 {-fftConvolution}\n\
Calculation of steady-state longitudinal bunch density distribution \n\
in an electron storage ring. \n\
<twissFile>    Elegant twiss file that contains ring parameters required\n\
               for calculation: pCentral (beam momentum), \n\
               alphac (momentum compaction), U0 (energy loss per turn),\n\
               Sdelta0 (rms of momentum distribution), etc.\n\
<resultsFile>  Output file with bunch distribution, wake potential for the\n\
               solved bunch distribution. The time coordinate is the\n\
               time advance from a reference point.\n\
semaphoreFile  If given, an empty file that is created when the computation finishes.\n\
wakeFunction   Input the wake function for an impulse response of a 1 C charge.\n\
               Time and wake column names should be specified with \n\
               units s and V/C respectively.\n";
static char *USAGE2 = "model          Instead of a wake function, a circuit model can be entered. The inductive\n\
               part may be entered with L (inductance) or with |Z/n| value.\n\
               R is the resistance.\n\
BBResonator    Instead of a wake function or circuit model, one can specify the\n\
               parameters of a broad-band resonator, optionally with a resistive\n\
               wall term (implemented as a simple resistor as in -model).\n\
RF             RF parameters that control the length of the zero-current beam.\n\
               If one want to solve from a potential with the main rf at a particular phase,\n\
               then the phase (in radians) must be given.\n\
length         Alternatively, one can specify the zero-current bunch length (rms)\n\
               in seconds directly. This case is considered a single rf system\n\
               and no harmonic cavities can be entered.\n\
harmonicCavity If -RF is given, one may also specify a higher-harmonic cavity\n\
               for possible bunch shortening or lengthening.\n\
               In general the bunch length should be solved from a non-harmonic potential,\n\
               thus the phase (in radians) of this harmonic cavity must be given.\n\
superPeriods   The number of superperiods of lattice file in the actual machine.\n\
               Default is 1.\n\
energy         The beam energy at which to perform computations, if it is desired\n\
               that this be different than the energy in the Twiss file.\n\
intermediateSolutions   write all intermediate solutions to the results file.\n\
               Used for debugging.\n\
integrationParameters   Specifies integration parameters for solving the \n\
               Haissinski integral equation. deltaTime specifies the time \n\
               interval in seconds for evaluating wake potential and charge density.\n\
               points is the number of points requested for the charge density.\n\
               iterationLimits is the number of iterations for solving the \n\
               integral equation. fraction is the relaxation fraction between \n\
               0 and 1 to reduce numerical instabilities.\n\
fftConvolution Compute the wake/density convolution with an FFT (O(n log n))\n\
               instead of the default direct sum (O(n^2)). Gives the same\n\
               result and is much faster for large -integrationParameters points;\n\
               only affects the -wakeFunction and -BBResonator paths.";

#define VERBOSE 0
#define CHARGE 1
#define PARTICLES 2
#define RF 3
#define LENGTH 4
#define INTEGRATION 5
#define STEPS 6
#define WAKE 7
#define MODEL 8
#define INTERMEDIATE_SOLUTIONS 9
#define SUPERPERIODS 10
#define ENERGY 11
#define OUTPUT_LAST_STEP_ONLY 12
#define HARMONIC_CAVITY 13
#define BUNCH_CURRENT 14
#define BBRESONATOR 15
#define SEMAPHORE_FILE 16
#define FFTCONVOLUTION 17
#define N_OPTIONS 18
char *option[N_OPTIONS] = {
  "verbose",
  "charge",
  "particles",
  "rf",
  "length",
  "integrationParameters",
  "steps",
  "wakeFunction",
  "model",
  "intermediateSolutions",
  "superperiods",
  "energy",
  "outputlaststeponly",
  "harmoniccavity",
  "bunchcurrent",
  "bbresonator",
  "semaphorefile",
  "fftconvolution"
};

typedef struct {
  double *y, xStart, xDelta;
  long points;
  long offset; /* time offset for all arrays, position where x is zero. */
  /* Different units from SI may be used.
     However to be compliant with elegant, the units of
     output files should be SI units.
     These factors allows conversion to SI units. */
  double xFactor, yFactor;
} FUNCTION;
typedef struct {
  long superPeriods;
  double desiredEnergyMeV;
  double energyMeV;
  double momentumCompaction;
  double U0;
  double sigmaDelta;
  double circumference;
  double revFrequency;
} RINGPARAMETERS;
typedef struct {
  double mainRfVoltage;
  double mainRfHarmonic;
  double mainRfPhase;
  double HHCvoltage;
  double HHCharmonicFactor;
  double HHCphase;
} RFPARAMETERS;

#define ITERATION_LIMITS 1000

void printFunction(char *label, FUNCTION *data);
void initializeFunction(FUNCTION *data);
void readRingParameters(char *twissFile, double desiredEnergyMeV, RINGPARAMETERS *parameters);

/* returned results is the first parameter. */
void getWakeFunction(FUNCTION *wake, char *wakeFile, char *tCol, char *wCol,
                     double delta, long points);
void integrateWakeFunction(FUNCTION *stepResponse, FUNCTION *wake);
void setupResultsFile(SDDS_TABLE *resultsPage, char *resultsFile, long points);
void getInitialDensity(FUNCTION *density, long points, double deltaTime, double charge, double length);
void copyFunction(FUNCTION *target, FUNCTION *source);
void getRfVoltage(FUNCTION *rfVoltageFn, RFPARAMETERS *rfParameters, RINGPARAMETERS *parameters);
void getRfPotential(FUNCTION *rfPotential, FUNCTION *rfVoltageFn, RINGPARAMETERS *parameters);
void getPotentialDistortion(FUNCTION *potentialDistortion,
                            FUNCTION *Vinduced,
                            FUNCTION *density, FUNCTION *wake);
void getPotentialDistortionFromModel(FUNCTION *potentialDistortion,
                                     FUNCTION *Vinduced,
                                     FUNCTION *density, double L, double R);
void calculateDistribution(FUNCTION *distribution, FUNCTION *potential,
                           FUNCTION *rfPotentialFn, FUNCTION *potentialDistortion,
                           double length, double VrfDot, long singleRF,
                           RINGPARAMETERS *ringParameters);
void normalizeDensityFunction(FUNCTION *density, FUNCTION *distribution, double charge);
void writeResults(SDDS_TABLE *resultsPage, FUNCTION *density,
                  FUNCTION *potential, FUNCTION *rfPotential, FUNCTION *rfVoltageFn, FUNCTION *potentialDistortion,
                  FUNCTION *Vind, double charge, double averageCurrent,
                  long converged, RFPARAMETERS *rfParameters, RINGPARAMETERS *parameters);
void makeBBRWakeFunction(FUNCTION *wake, double dt, long points,
                         double Q, double R, double omega, double rw, double T0);

long verbosity;
long useFFTConvolution = 0; /* -fftConvolution: use O(n log n) FFT for the wake convolution */

int main(int argc, char **argv) {
  SCANNED_ARG *scanned;
  long i, j, k, outputLastStepOnly;
  char *twissFile, *resultsFile, *semaphoreFile=NULL;
  SDDS_DATASET resultsPage;
  double particles, charge, finalCharge, length;
  long steps, converged;
  int32_t points, iterationLimits;
  long useWakeFunction = 0, intermediateSolutions;
  RFPARAMETERS rfParameters;
  RINGPARAMETERS ringParameters;
  double deltaTime;
  unsigned long dummyFlags;
  FUNCTION density, densityOld, diff, wake, stepResponse;
  FUNCTION potential, rfVoltageFn, rfPotential, potentialDistortion, distribution, Vinduced;
  char *wakeFile, *tCol, *wCol;
  double syncPhase = 0, syncTune, syncAngFrequency;
  double VrfDot, ZoverN = 0, inductance = 0, resistance;
  double maxDifference, rmsDifference, madDifference, maxTolerance, fraction, lastMaxDifference;
  double maxDensity;
  double averageCurrent = 0.0, bunchCurrent = 0.0, desiredEnergy;
  long useBBR = 0;
  double BBR_R, BBR_Q, BBR_frequency, BBR_rw;
  long singleRF = 0;

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&scanned, argc, argv);
  if (argc == 1) {
    fprintf(stderr, "%s%s\n", USAGE1, USAGE2);
    exit(1);
  }

  /* initialize the FUNCTION structures */
  initializeFunction(&density);
  initializeFunction(&densityOld);
  initializeFunction(&diff);
  initializeFunction(&wake);
  initializeFunction(&stepResponse);
  initializeFunction(&potential);
  initializeFunction(&rfPotential);
  initializeFunction(&potentialDistortion);
  initializeFunction(&distribution);
  initializeFunction(&Vinduced);
  initializeFunction(&rfVoltageFn);

  twissFile = NULL;
  resultsFile = NULL;
  verbosity = 0;
  particles = finalCharge = bunchCurrent = 0;
  length = 0;
  rfParameters.mainRfVoltage = rfParameters.mainRfHarmonic = rfParameters.mainRfPhase = rfParameters.HHCvoltage = rfParameters.HHCharmonicFactor = rfParameters.HHCphase = 0;
  steps = 1;
  points = 1000;
  iterationLimits = ITERATION_LIMITS;
  deltaTime = 0;
  maxTolerance = 0.0001; /* fraction of maximum */
  intermediateSolutions = 0;
  fraction = 0.01;
  wakeFile = tCol = wCol = NULL;
  desiredEnergy = 0.0;
  outputLastStepOnly = 0;
  ringParameters.superPeriods = 1;

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
        if (scanned[i].n_items < 2)
          bomb("invalid -charge syntax", NULL);
        get_double(&finalCharge, scanned[i].list[1]);
        break;
      case BUNCH_CURRENT:
        if (scanned[i].n_items < 2)
          bomb("invalid -bunchCurrent syntax", NULL);
        get_double(&bunchCurrent, scanned[i].list[1]);
        break;
      case PARTICLES:
        if (scanned[i].n_items < 2)
          bomb("invalid -particles syntax", NULL);
        get_double(&particles, scanned[i].list[1]);
        break;
      case STEPS:
        if (scanned[i].n_items < 2)
          bomb("invalid -steps syntax", NULL);
        get_long(&steps, scanned[i].list[1]);
        break;
      case SUPERPERIODS:
        if (scanned[i].n_items < 2)
          bomb("invalid -superPeriods syntax", NULL);
        get_long(&(ringParameters.superPeriods), scanned[i].list[1]);
        break;
      case ENERGY:
        if (scanned[i].n_items < 2)
          bomb("invalid -energy syntax", NULL);
        get_double(&desiredEnergy, scanned[i].list[1]);
        break;
      case INTERMEDIATE_SOLUTIONS:
        intermediateSolutions = 1;
        break;
      case OUTPUT_LAST_STEP_ONLY:
        outputLastStepOnly = 1;
        break;
      case LENGTH:
        if (scanned[i].n_items < 2)
          bomb("invalid -length syntax", NULL);
        get_double(&length, scanned[i].list[1]);
        singleRF = 1;
        break;
      case RF:
        if (scanned[i].n_items < 2)
          bomb("invalid -rf syntax", NULL);
        scanned[i].n_items--;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "voltage", SDDS_DOUBLE, &rfParameters.mainRfVoltage, 1, 0,
                          "harmonic", SDDS_DOUBLE, &rfParameters.mainRfHarmonic, 1, 0,
                          "phase", SDDS_DOUBLE, &rfParameters.mainRfPhase, 1, 0,
                          NULL) ||
            rfParameters.mainRfVoltage <= 0 || rfParameters.mainRfHarmonic <= 0)
          bomb("invalid -rf syntax/values", "-rf=voltage=<V>,harmonic=<value>,phase=<value>");
        break;
      case HARMONIC_CAVITY:
        if (scanned[i].n_items < 2)
          bomb("invalid -harmonicCavity syntax", NULL);
        scanned[i].n_items--;
        singleRF = 0;
        /* harmonic and factor are the same thing, a low integer */
        /* maybe later we'll change the type  to integer */
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "voltage", SDDS_DOUBLE, &rfParameters.HHCvoltage, 1, 0,
                          "harmonic", SDDS_DOUBLE, &rfParameters.HHCharmonicFactor, 1, 0,
                          "factor", SDDS_DOUBLE, &rfParameters.HHCharmonicFactor, 1, 0,
                          "phase", SDDS_DOUBLE, &rfParameters.HHCphase, 1, 0,
                          NULL) ||
            rfParameters.HHCharmonicFactor <= 0)
          bomb("invalid -harmonicCavity syntax/values",
               "-harmonicCavity=voltage=<V>,harmonicFactor=<value>,phase=<value>");
        break;
      case INTEGRATION:
        scanned[i].n_items--;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "points", SDDS_LONG, &points, 1, 0,
                          "deltaTime", SDDS_DOUBLE, &deltaTime, 1, 0,
                          "fraction", SDDS_DOUBLE, &fraction, 1, 0,
                          "tolerance", SDDS_DOUBLE, &maxTolerance, 1, 0,
                          "iterations", SDDS_LONG, &iterationLimits, 1, 0,
                          NULL) ||
            points <= 0 || iterationLimits <= 0 || maxTolerance <= 0 || deltaTime < 0) {
          bomb("invalid -integrationParameters syntax/values", "-integrationParameters=deltaTime=<s>,points=<number>,iterations=<number>");
        }

        break;
      case WAKE:
        if (!strlen(wakeFile = scanned[i].list[1]))
          bomb("bad -wake syntax", "-wakeFunction=<file>");
        scanned[i].n_items -= 2;
        if (!scanItemList(&dummyFlags, scanned[i].list + 2, &scanned[i].n_items, 0,
                          "tColumn", SDDS_STRING, &tCol, 1, 0,
                          "wColumn", SDDS_STRING, &wCol, 1, 0,
                          NULL) ||
            !tCol || !wCol)
          bomb("invalid -wake syntax/values", "-wakeFunction=<file>,tColumn=<name>,wColumn=<name>");
        useWakeFunction = 1;
        useBBR = 0;
        break;
      case MODEL:
        if (scanned[i].n_items < 2)
          bomb("invalid -model syntax", NULL);
        scanned[i].n_items--;
        inductance = ZoverN = resistance = 0.0;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "L", SDDS_DOUBLE, &inductance, 1, 0,
                          "Zn", SDDS_DOUBLE, &ZoverN, 1, 0,
                          "R", SDDS_DOUBLE, &resistance, 1, 0,
                          NULL) ||
            inductance < 0 || resistance < 0 || ZoverN < 0)
          bomb("invalid -model syntax/values", "-model=[L=<Henry>|Zn=<Ohms>],R=<Ohm>");
        useWakeFunction = useBBR = 0;
        break;
      case BBRESONATOR:
        if (scanned[i].n_items < 3)
          bomb("invalid -BBResonator syntax", NULL);
        scanned[i].n_items--;
        BBR_Q = BBR_frequency = BBR_R = BBR_rw = 0;
        if (!scanItemList(&dummyFlags, scanned[i].list + 1, &scanned[i].n_items, 0,
                          "R", SDDS_DOUBLE, &BBR_R, 1, 0,
                          "Q", SDDS_DOUBLE, &BBR_Q, 1, 0,
                          "frequency", SDDS_DOUBLE, &BBR_frequency, 1, 0,
                          "wall", SDDS_DOUBLE, &BBR_rw, 1, 0,
                          NULL) ||
            inductance < 0 || resistance < 0 || ZoverN < 0)
          bomb("invalid -model syntax/values", "-model=[L=<Henry>|Zn=<Ohms>],R=<Ohm>");
        useWakeFunction = 0;
        useBBR = 1;
        break;
      case SEMAPHORE_FILE:
        if (scanned[i].n_items!=2 || !strlen(semaphoreFile = scanned[i].list[1]))
          bomb("bad -semaphoreFile syntax", "-semaphoreFile=<file>");
	if (fexists(semaphoreFile))
	  remove(semaphoreFile);
	break;
      case FFTCONVOLUTION:
        useFFTConvolution = 1;
        break;
      default:
        bomb("unknown option given.", NULL);
      }
    } else {
      if (!twissFile)
        twissFile = scanned[i].list[0];
      else if (!resultsFile)
        resultsFile = scanned[i].list[0];
      else
        bomb("too many filenames given", NULL);
    }
  }
  if (((finalCharge != 0) + (particles != 0) + (bunchCurrent != 0)) > 1)
    bomb("Specify only one of -charge, -particles, or -bunchCurrent.", NULL);

  if (length && rfParameters.mainRfVoltage)
    bomb("Options length and RF cannot be both specified.", NULL);
  if (rfParameters.HHCharmonicFactor) {
    if (!rfParameters.mainRfVoltage)
      bomb("You must give -rf if you give -harmonicCavity", NULL);
  }
  readRingParameters(twissFile, desiredEnergy * 1e3, &ringParameters);
  ringParameters.revFrequency = c_mks / ringParameters.circumference;

  if (bunchCurrent)
    finalCharge = bunchCurrent / ringParameters.revFrequency;
  if (!finalCharge)
    finalCharge = particles * e_mks;
  if (!particles)
    particles = finalCharge / e_mks;

  if (!finalCharge)
    bomb("Specify at least one of -charge, -particles, or -bunchCurrent.", NULL);

  if (!length) { /* that is, main RF is specified on the command line */
    /* These calculations are correct only in the case of harmonic
       potential, for which it is the user's responsibility to understand,
       that, depending on the harmonic cavity's settings, the potential can be very
       non-harmonic */
    /* this phase is the ring definition of phase, i.e. V sin(omega.t), not the same as elegant's usual
       (linac) definition of phase */
    syncPhase = asin(ringParameters.U0 / rfParameters.mainRfVoltage);
    syncTune = sqrt(ringParameters.momentumCompaction * rfParameters.mainRfHarmonic * cos(syncPhase) /
                    2 / PI * rfParameters.mainRfVoltage / (ringParameters.energyMeV * 1e6));
    /*    if (rfParameters.HHCharmonicFactor)
          syncTune
          *= sqrt(1 + (rfParameters.HHCvoltage/rfParameters.mainRfVoltage)*rfParameters.HHCharmonicFactor/cos(syncPhase)); */
    syncAngFrequency = syncTune * 2 * PI * ringParameters.revFrequency;
    /* Even with HHC, start with a length from main rf system running only. */
    length = ringParameters.momentumCompaction * ringParameters.sigmaDelta / syncAngFrequency;

    /* length is seconds */
    /* derivative w.r.t time. Again this is valid only when the total rf potential is harmonic */
    VrfDot = rfParameters.mainRfVoltage * 2 * PI * rfParameters.mainRfHarmonic * ringParameters.revFrequency *
      (cos(syncPhase) + rfParameters.HHCvoltage / rfParameters.mainRfVoltage * rfParameters.HHCharmonicFactor);
  } else {
    /* only useful when the total rf potential is harmonic */
    /* When length is specified we don't know the RF voltage or the harmonic. However Vrfdot
       and f_s can be calculated. The rf potential can be constructed. */
    syncAngFrequency = ringParameters.momentumCompaction * ringParameters.sigmaDelta / length; /* length is seconds */
    syncTune = syncAngFrequency / 2 / PI / ringParameters.revFrequency;
    VrfDot = sqr(2 * PI) * ringParameters.revFrequency * sqr(syncTune) * (ringParameters.energyMeV * 1e6) /
      ringParameters.momentumCompaction;
  }
  if (deltaTime == 0)
    deltaTime = length / 100;

  if (!useWakeFunction && !useBBR) {
    if (inductance == 0 && ZoverN != 0) {
      inductance = ZoverN * ringParameters.circumference / 2 / PI / c_mks;
    }
  }

  if (useWakeFunction || useBBR) {
    useWakeFunction = 1;
    if (useBBR) {
      /* generate the wake function for a BBR
       * W = omega*R/Q * exp(-omega*t/(2*Q)) * (cos (omega*t) - sin(omega'*t)/(2*Q'))
       * units of W are V/C
       */
#ifdef DEBUG
      FILE *fp;
      long i;
      fprintf(stderr, "Making BBR wake\n");
#endif

      makeBBRWakeFunction(&wake, deltaTime, points, BBR_Q, BBR_R, BBR_frequency * PIx2, BBR_rw, 1 / ringParameters.revFrequency);

#ifdef DEBUG
      fp = fopen("BBRWake.sdds", "w");
      fprintf(fp, "SDDS1\n&column name=Time, type=double, units=s &end\n");
      fprintf(fp, "&column name=Wakefield, type=double, units=V/C &end\n");
      fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
      for (i = 0; i < points; i++)
        fprintf(fp, "%e %e\n", deltaTime * i, wake.y[i]);
      fclose(fp);
#endif
    } else {
      /*
        determine wake function from file data or from parameters.
        May require interpolation.
      */
      getWakeFunction(&wake, wakeFile, tCol, wCol, deltaTime, points);
    }

    /* integrate Wake function for step response of the special solver
     */
    integrateWakeFunction(&stepResponse, &wake);
  }

  setupResultsFile(&resultsPage, resultsFile, points);

  /* Loop over charge steps. The iterations in strength are required for convergence.
   */
  for (i = 1; i <= steps; i++) {
    charge = 1.0 * i / steps * finalCharge;
    fflush(stdout);
    if (i == 1) { /* these rf-related calls are done just once. */
      /* with HHC on, the value of length is from main rf system alone. */
      getInitialDensity(&density, points, deltaTime, charge, length);

      if (verbosity > 1)
        printFunction("Initial density", &density);
      copyFunction(&rfVoltageFn, &density);
      /* this phase has been calculated above to be between 0 and pi/2,
         which is not standard. It should be between pi/2 and pi for
         positive alpha rings. */
      /* adding syncPhase and commandline main rf phase together */
      rfParameters.mainRfPhase += syncPhase;
      getRfVoltage(&rfVoltageFn, &rfParameters, &ringParameters);
      if (verbosity > 1)
        printFunction("rf voltage", &rfVoltageFn);
      getRfPotential(&rfPotential, &rfVoltageFn, &ringParameters);
      if (verbosity > 1)
        printFunction("rf potential", &rfPotential);
    } else {
      /* use the previous solution (of lower charge) to make a new starting
         distribution */
      for (j = 0; j < points; j++) {
        density.y[j] *= 1.0 * i / (i - 1);
      }
    }
    /* Loop integration until convergence
     */
    converged = 0;
    if (verbosity) {
      fprintf(stdout, "Iterations for solution for charge %g\n", charge);
      fflush(stdout);
    }
    lastMaxDifference = DBL_MAX;
    for (j = 0; j < iterationLimits; j++) {
      copyFunction(&densityOld, &density);
      if (verbosity > 2)
        printFunction("density", &density);
      /* Calculate potential well distortion term.
         Induced voltage is returned as well.
      */
      if (useWakeFunction) {
        getPotentialDistortion(&potentialDistortion, &Vinduced, &density, &wake);
      } else {
        getPotentialDistortionFromModel(&potentialDistortion, &Vinduced,
                                        &density, inductance, resistance);
      }
      if (verbosity > 2) {
        printFunction("V induced", &Vinduced);
        printFunction("Potential distortion", &potentialDistortion);
      }
      /* Calculate distribution exponential (Use K. Bane's expression in
         SLAC-PUB-5177 p. 30) Even if we have harmonic cavities, this call
         will produce an initial estimate of length and distribution to
         enter into the iterations */
      calculateDistribution(&distribution, &potential, &rfPotential, &potentialDistortion,
                            length, VrfDot, singleRF, &ringParameters);
      if (verbosity > 2) {
        printFunction("Total potential", &potential);
        printFunction("New distribution", &distribution);
      }
      /* Normalize for new density function
       */
      normalizeDensityFunction(&density, &distribution, charge);
      if (verbosity > 2)
        printFunction("New density", &density);
      /* mix old density with new density to see if numerical instability
         is reduced. This may or may not work. It may be removed later, or
         a different convergence scheme could be dreamed up. */
      for (k = 0; k < points; k++) {
        density.y[k] = fraction * density.y[k] + (1 - fraction) * densityOld.y[k];
      }

      /* Compare new density function with
         old one and check for convergence
      */
      maxDifference = 0.0;
      madDifference = 0.0;
      rmsDifference = 0.0;
      diff = density;
      diff.y = SDDS_Malloc(sizeof(*diff.y) * diff.points);
      maxDensity = 0.0;
      for (k = 0; k < diff.points; k++) {
        if (density.y[k] > maxDensity)
          maxDensity = density.y[k];
        diff.y[k] = density.y[k] - densityOld.y[k];
        if (FABS(diff.y[k]) > maxDifference) {
          maxDifference = FABS(diff.y[k]);
        }
        rmsDifference += sqr(diff.y[k]);
        madDifference += FABS(diff.y[k]);
      }
      rmsDifference /= points;
      rmsDifference = sqrt(rmsDifference);
      madDifference /= points;
      if (verbosity) {
        fprintf(stdout, "maxDifference %10.2g madDifference %10.2g rmsDifference %10.2g.\n",
                maxDifference / maxDensity, madDifference / maxDensity, rmsDifference / maxDensity);
        fflush(stdout);
      }
      if (maxDifference / maxDensity < maxTolerance) {
        if (verbosity) {
          fprintf(stdout, "Converged for charge %g after %ld iterations.\n", charge, j);
          fflush(stdout);
        }
        converged = 1;
        break;
      }
      free(diff.y);
      if (intermediateSolutions) {
        writeResults(&resultsPage, &density, &potential, &rfPotential, &rfVoltageFn, &potentialDistortion,
                     &Vinduced, charge, averageCurrent, converged, &rfParameters, &ringParameters);
      }
      if (maxDifference > lastMaxDifference)
        fraction /= 2;
      lastMaxDifference = maxDifference;
    }
    /* write results whether converged or not */
    averageCurrent = charge * ringParameters.revFrequency;
    if (!outputLastStepOnly || i == steps)
      writeResults(&resultsPage, &density, &potential, &rfPotential, &rfVoltageFn, &potentialDistortion,
                   &Vinduced, charge, averageCurrent, converged, &rfParameters, &ringParameters);
  }
  if (!SDDS_Terminate(&resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
  if (semaphoreFile && strlen(semaphoreFile))
    fclose(fopen(semaphoreFile, "w"));
    
  return 0;
}

void initializeFunction(FUNCTION *data) {
  data->y = NULL;
  data->xStart = data->xDelta = 0.0;
  data->points = data->offset = 0;
  data->xFactor = data->yFactor = 0.0;
}

void readRingParameters(char *twissFile, double desiredEnergyMeV, RINGPARAMETERS *parameters) {
  SDDS_TABLE twissPage;
  double pCentral, *s;
  long elements;

  if (verbosity) {
    fprintf(stdout, "Opening for reading \"%s\"\n", twissFile);
    fflush(stdout);
  }
  if (!SDDS_InitializeInput(&twissPage, twissFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* read first page of input file to get parameters
   */
  SDDS_ReadPage(&twissPage);
  if (!SDDS_GetParameters(&twissPage,
                          "pCentral", &pCentral,
                          "alphac", &(parameters->momentumCompaction),
                          "U0", &(parameters->U0),
                          "Sdelta0", &(parameters->sigmaDelta),
                          NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  parameters->energyMeV = sqrt(sqr(pCentral) + 1) * me_mev;
  elements = SDDS_CountRowsOfInterest(&twissPage);
  s = SDDS_GetColumnInDoubles(&twissPage, "s");
  parameters->circumference = s[elements - 1] * parameters->superPeriods;
  parameters->U0 *= 1e6 * parameters->superPeriods; /* units in eV */

  if (desiredEnergyMeV > 0) {
    if (verbosity) {
      fprintf(stdout, "Scaling from %f MeV to %f MeV\n",
              parameters->energyMeV, desiredEnergyMeV);
      fflush(stdout);
    }
    parameters->U0 *= ipow4(desiredEnergyMeV / (parameters->energyMeV));
    parameters->sigmaDelta *= desiredEnergyMeV / (parameters->energyMeV);
    parameters->energyMeV = desiredEnergyMeV;
  }

  if (!SDDS_Terminate(&twissPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
  free(s);
}

void getWakeFunction(FUNCTION *wake, char *wakeFile, char *tCol, char *wCol,
                     double deltaTime, long pointsCalledFor) {
  SDDS_TABLE wakePage;
  double time, *t, *V;
  long i, rows, pointsThatCoversRangeOfData, returnCode;

  if (verbosity) {
    fprintf(stdout, "Opening for reading \"%s\"\n", wakeFile);
    fflush(stdout);
  }

  if (!SDDS_InitializeInput(&wakePage, wakeFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  SDDS_ReadPage(&wakePage);
  rows = SDDS_CountRowsOfInterest(&wakePage);
  switch (SDDS_CheckColumn(&wakePage, tCol, "s", SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_NONEXISTENT:
  case SDDS_CHECK_WRONGTYPE:
  case SDDS_CHECK_WRONGUNITS:
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  }
  switch (SDDS_CheckColumn(&wakePage, wCol, "V/C", SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_NONEXISTENT:
  case SDDS_CHECK_WRONGTYPE:
  case SDDS_CHECK_WRONGUNITS:
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exit(1);
    break;
  }
  t = SDDS_GetColumnInDoubles(&wakePage, tCol);
  V = SDDS_GetColumnInDoubles(&wakePage, wCol);

  /* start as close as possible to first point in file. In
     case of wake function, there may be some negative t values
     included for some reason (even though this might correspond
     to a non-causal wake.) */
  wake->xDelta = deltaTime;
  if (t[0] < 0) {
    wake->offset = ((long)(-t[0] / deltaTime)) - 1;
    wake->xStart = -wake->offset * deltaTime; /* xStart should be negative */
  } else {                                    /* is t[0] > 0, then we have some serious problems in that we
                                                 don't know the wake at zero. Leave the offset at zero in order
                                                 not to break things.*/
    wake->offset = 0;
    wake->xStart = 0.0;
  }
  pointsThatCoversRangeOfData = (long)((t[rows - 1] - t[0]) / deltaTime) + 1;
  /* points generated for wake function cannot give a greater range that
     what is provided by SDDS file */
  wake->points = MIN(pointsCalledFor, pointsThatCoversRangeOfData);
  /* I don't think we should remove points that are non-causal */
  /* wake->points = points - wake->offset; */
  wake->y = SDDS_Malloc(sizeof(*wake->y) * wake->points);
  wake->xFactor = 1;
  wake->yFactor = 1;
  /* interpolate to produce points spaced by deltaTime. */
  for (i = 0; i < wake->points; i++) {
    time = wake->xStart + i * deltaTime;
    wake->y[i] = interp(V, t, rows, time, 1, 1, &returnCode);
    if (verbosity > 1)
      fprintf(stderr, "%ld %14.3le %14.3le\n", i, time, wake->y[i]);
  }
  if (verbosity > 1)
    printFunction("interpolated wake function", wake);
}

void integrateWakeFunction(FUNCTION *stepResponse, FUNCTION *wake) {
  long i;

  *stepResponse = *wake;
  stepResponse->y = SDDS_Malloc(sizeof(*stepResponse->y) * stepResponse->points);
  stepResponse->yFactor = 1; /* using units (V/C)-s */
  stepResponse->y[0] = 0.0;
  for (i = 1; i < wake->points; i++) {
    stepResponse->y[i] = stepResponse->y[i - 1] + wake->xDelta *
      (wake->y[i] + wake->y[i - 1]) / 2.0;
  }
  if (verbosity > 1)
    printFunction("wake step function", stepResponse);
}

void setupResultsFile(SDDS_TABLE *resultsPage, char *resultsFile, long points) {
  /* define three columns but no parameters */
  if (!SDDS_InitializeOutput(resultsPage, SDDS_BINARY, 1,
                             "Bunch density distribution",
                             "Bunch density distribution",
                             resultsFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (0 > SDDS_DefineParameter(resultsPage, "Charge", "Q", "C",
                               "Total Charge", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "AverageCurrent", "I$bave$n", "A",
                               "Average current", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "Convergence", NULL, NULL,
                               "Convergence of iterations of integral equation", NULL,
                               SDDS_STRING, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "SuperPeriods", NULL, NULL,
                               "Superperiods of the lattice selected", NULL,
                               SDDS_LONG, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "DesiredEnergy", NULL, "Mev",
                               "Desired Energy from the command line", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "Energy", NULL, "Mev",
                               "Energy", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "AverageTau", "C$gt$r", "s",
                               "Bunch position in time", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "AverageZ", "Cs", "s",
                               "Bunch position in distance", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "SigmaTau", "$gs$bt$n$r", "s",
                               "Bunch length in time", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "SigmaZ", "$gs$r$bz$n", "s",
                               "Bunch length", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "MainRfVoltage", "V$brf$n", "V",
                               "Voltage of main RF system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "MainRfHarmonic", "h", NULL,
                               "Harmonic number of main rf system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "MainRfPhase", "$gf$e$bs$r", NULL,
                               "Phase of main rf system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "HarmonicRfVoltage", "V$bh$n", "V",
                               "Voltage of higher-harmonic RF system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "HarmonicFactor", "N", NULL,
                               "Harmonic factor of higher-harmonic rf system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineParameter(resultsPage, "HarmonicRfPhase", "$gf$r$bs$n", NULL,
                               "Phase of higher-harmonic rf system", NULL,
                               SDDS_DOUBLE, NULL) ||
      0 > SDDS_DefineColumn(resultsPage, "Time", "$gt$r", "s",
                            "Time relative to synchronous phase at zero current",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "Density", "$gr$r", "C/m",
                            "Charge density",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "Current", "I", "A",
                            "Instantaneous current",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "WakeField", NULL, "V",
                            "Wake field for particular bunch distribution",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "PotentialWellDistortion", NULL, NULL,
                            "Wake potential distortion term in distribution",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "RfPotentialWell", NULL, NULL,
                            "Potential well from RF voltage",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "RfVoltage", NULL, "V",
                            "RF voltage",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(resultsPage, "PotentialWell", NULL, NULL,
                            "Total Wake potential term in distribution",
                            NULL, SDDS_DOUBLE, 0))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_WriteLayout(resultsPage) || !SDDS_StartPage(resultsPage, points))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
}

void getInitialDensity(FUNCTION *density, long points, double deltaTime,
                       double charge, double length) {
  long i;
  double C;

  density->points = points;
  density->xDelta = deltaTime;
  density->y = SDDS_Malloc(sizeof(*density->y) * density->points);
  density->offset = -((long)points / 2);
  density->xStart = density->offset * density->xDelta;
  density->xFactor = 1; /* using seconds */
  density->yFactor = 1; /* using C/s, i.e. density is really a current */

  C = charge / sqrt(2 * PI) / length;
  for (i = 0; i < points; i++) {
    density->y[i] = C * exp(-sqr((i + density->offset) * density->xDelta) /
                            2.0 / sqr(length));
  }
}

void copyFunction(FUNCTION *target, FUNCTION *source) {
  long i;
  double *ptr;

  ptr = target->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we don't want. We want to keep the original
     pointer from creation in order not to cause memory leaks. Espcially if this
     function is called repeatedly for the same FUNCTION structure.
  */
  *target = *source;
  target->y = ptr;
  if (!target->y) {
    target->y = SDDS_Malloc(sizeof(*target->y) * source->points);
  }
  for (i = 0; i < source->points; i++) {
    target->y[i] = source->y[i];
  }
}

void getRfVoltage(FUNCTION *rfVoltageFn, RFPARAMETERS *rfParameters, RINGPARAMETERS *parameters) {

  long i;
  double time;
  /* array y was allocated from the previous call to copyFunction */

  /* calculate rf voltage */
  for (i = 0; i < rfVoltageFn->points; i++) {
    time = (i + rfVoltageFn->offset) * rfVoltageFn->xDelta;
    /* using the sine function, the main rf phase would be between 0 and
       pi/2, a reflection of a value in the range pi/2 and pi relative to
       pi/2. This effectively makes the time quantity have the same sign as
       the phase space time coordinate. The voltage is rising for
       increasing "tau" */
    rfVoltageFn->y[i] = rfParameters->mainRfVoltage * sin(2 * PI * rfParameters->mainRfHarmonic * parameters->revFrequency * time + rfParameters->mainRfPhase);
    if (rfParameters->HHCvoltage)
      rfVoltageFn->y[i] += rfParameters->HHCvoltage * sin(2 * PI * rfParameters->mainRfHarmonic * rfParameters->HHCharmonicFactor * parameters->revFrequency * time + rfParameters->HHCphase);
  }
}

void getRfPotential(FUNCTION *pot, FUNCTION *rfVoltage, RINGPARAMETERS *parameters) {

  long i;
  double *ptr, P0;

  ptr = pot->y;
  *pot = *rfVoltage;
  pot->y = ptr; /* to prevent memory leak */
  pot->y = SDDS_Malloc(sizeof(*pot->y) * pot->points);
  pot->yFactor = 1;

  /* integrate (rfVoltage - U0) */
  pot->y[0] = 0.0;
  for (i = 1; i < pot->points; i++)
    pot->y[i] = pot->y[i - 1] + pot->xDelta *
      ((rfVoltage->y[i] + rfVoltage->y[i - 1]) / 2.0 - parameters->U0);

  /* potential is zero at the index of offset, which is the midpoint of the array */
  P0 = pot->y[-pot->offset]; /* a variable needs to be created for modifying the array */
  for (i = 0; i < pot->points; i++)
    pot->y[i] -= P0;
}

/* Compute the induced voltage (Vind) as the wake/density convolution using an
   FFT, i.e. O(n log n) instead of the O(n^2) direct sum. This reproduces the
   direct loop below to machine precision. The sum is a cross-correlation
   (density is indexed at i+index-offset), computed as
   IFFT(conj(FFT(wake)) * FFT(density)); the fftpack convention here requires
   multiplying the inverse transform by the (padded) length L. */
static void inducedVoltageFFT(FUNCTION *Vind, FUNCTION *density, FUNCTION *wake) {
  long N = density->points, M = wake->points, woffset = wake->offset;
  long L = 1, i, k, p, q;
  double *A, *B, *C;

  while (L < N + M)
    L <<= 1;
  A = calloc(2 * L, sizeof(*A)); /* complex, interleaved re,im; zero-padded */
  B = calloc(2 * L, sizeof(*B));
  C = calloc(2 * L, sizeof(*C));
  for (p = 0; p < M; p++)
    A[2 * p] = wake->y[p];
  for (q = 0; q < N; q++)
    B[2 * q] = density->y[q];
  complexFFT(A, L, 0);
  complexFFT(B, L, 0);
  for (k = 0; k < L; k++) {
    double ar = A[2 * k], ai = A[2 * k + 1], br = B[2 * k], bi = B[2 * k + 1];
    C[2 * k] = ar * br + ai * bi;     /* Re[ conj(A) * B ] */
    C[2 * k + 1] = ar * bi - ai * br; /* Im[ conj(A) * B ] */
  }
  complexFFT(C, L, INVERSE_FFT);
  for (i = 0; i < N; i++) {
    long lag = i - woffset;
    long idx = lag >= 0 ? lag : L + lag; /* negative lags wrap around */
    Vind->y[i] = -density->xDelta * C[2 * idx] * (double)L;
  }
  free(A);
  free(B);
  free(C);
}

void getPotentialDistortion(FUNCTION *potentialDistortion,
                            FUNCTION *Vind,
                            FUNCTION *density, FUNCTION *wake) {
  long i, j, index;
  double *ptr;

  /* There are two results from this call, the induced voltage Vind and its
     integral, the potential potentialDistortion. */

  ptr = potentialDistortion->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we want to keep.
  */
  *potentialDistortion = *density;
  potentialDistortion->y = ptr;
  /* the potential well should have the same range as the density,
     i.e. the abscissa of both functions line up. */
  if (!potentialDistortion->y) {
    potentialDistortion->y = SDDS_Malloc(sizeof(*potentialDistortion->y) * potentialDistortion->points);
  }
  ptr = Vind->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we want to keep.
  */
  *Vind = *density;
  Vind->y = ptr;
  /*  Vind should have the same range as the density,
      i.e. the abscissa of both functions line up. */
  if (!Vind->y) {
    Vind->y = SDDS_Malloc(sizeof(*Vind->y) * Vind->points);
  }

  /* calculate induced voltage as a convolution. i.e. a n^2 calculation
     (unless -fftConvolution requests the O(n log n) FFT version). */
  if (useFFTConvolution) {
    inducedVoltageFFT(Vind, density, wake);
  } else
  for (i = 0; i < density->points; i++) {
    Vind->y[i] = 0.0;                                /* i is the index for tau in internal note (where the
                                                        induced voltage is to be calculated*/
    for (index = 0; index < wake->points; index++) { /* wake function array element */
      j = index + i - wake->offset;
      if (j < 0 || j >= density->points)
        continue;
      /* j is the tau' index in internal note */
      /* start integrating with the first element of wake->y even though
         it may correspond to negative time, i.e. non-causal data which
         may be zero, btw. */
      /* Integral as a sum of integrands. We can optimize this later,
         e.g. move the xDelta outside the inner loop, and use the extended
         trapezoid integral, which requires more index calculations.*/
      /* While delta-function wake is defined as positive, the induced
         voltage is expected to be negative. Use a minus sign here. */
      /* Also the "j" index appears positive in both density and wake. That
         is because wake and density are defined with opposite signs */
      Vind->y[i] -= wake->y[index] * density->y[j] * density->xDelta;
    }
  }

  /* integrate Vind to get potentialDistortion */
  if (potentialDistortion->offset > 0) {
    bomb("problem with offset value of FUNCTION potentialDistortion.", NULL);
  }
  potentialDistortion->y[0] = 0.0;
  for (i = 1; i < potentialDistortion->points; i++) {
    /* integral with trapeze rule. Can improve this with modified trapezoid integral. p. 885, Abramowitz and Stegun*/
    potentialDistortion->y[i] = potentialDistortion->y[i - 1] + potentialDistortion->xDelta *
      (Vind->y[i - 1] + Vind->y[i]) / 2.0;
  }

  /* potential at synchronous phase and middle of original
     density distribution is defined to be zero. Why the statements below are commented? */
  /*
    P0 = potentialDistortion->y[-potentialDistortion->offset];
    for (i=0; i<potentialDistortion->points; i++) {
    potentialDistortion->y[i] -= P0;
    }
  */
}

void getPotentialDistortionFromModel(FUNCTION *potentialDistortion,
                                     FUNCTION *Vind,
                                     FUNCTION *density, double L, double R) {
  long i;
  double *ptr, *charge;
  /* potential distortion would normally oppose that of the external rf potential */

  ptr = potentialDistortion->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we preserved.
  */
  *potentialDistortion = *density;
  potentialDistortion->y = ptr;
  /* the potential well should have the same range as the density,
     i.e. the abscissa of both functions line up. */
  if (!potentialDistortion->y) {
    potentialDistortion->y = SDDS_Malloc(sizeof(*potentialDistortion->y) * potentialDistortion->points);
  }

  ptr = Vind->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we preserved.
  */
  *Vind = *density;
  Vind->y = ptr;
  /* Vind should have the same range as the density,
     i.e. the abscissa of both functions line up. */
  if (!Vind->y) {
    Vind->y = SDDS_Malloc(sizeof(*Vind->y) * Vind->points);
  }

  charge = SDDS_Malloc(sizeof(*charge) * density->points);
  charge[0] = 0.0; /* cumulative charge along the bunch profile */
  for (i = 1; i < potentialDistortion->points; i++) {
    charge[i] = charge[i - 1] + density->xDelta * (density->y[i - 1] + density->y[i]) / 2.0;
  }
  /* For a model that includes inductance, the potential is calculated first, then
     the Vind is the derivative. */
  /* terms in R should be negative, right? */
  for (i = 0; i < potentialDistortion->points; i++) {
    potentialDistortion->y[i] = L * density->y[i] - R * charge[i];
  }
  for (i = 1; i < potentialDistortion->points - 1; i++) {
    Vind->y[i] = L * (density->y[i + 1] - density->y[i - 1]) / 2.0 / density->xDelta - R * density->y[i];
  }
  /* end points */
  Vind->y[0] = L * (density->y[1] - density->y[0]) /
    density->xDelta -
    R * density->y[0];
  Vind->y[potentialDistortion->points - 1] =
    L * (density->y[potentialDistortion->points - 1] - density->y[potentialDistortion->points - 2]) /
    density->xDelta -
    R * density->y[potentialDistortion->points - 1];
  free(charge);
}

void calculateDistribution(FUNCTION *distribution, FUNCTION *potential,
                           FUNCTION *rfPotentialFn, FUNCTION *potentialDistortion,
                           double length, double VrfDot, long singleRF,
                           RINGPARAMETERS *parameters) {
  long i;
  double rfPotential, *ptr, time;

  ptr = potential->y;
  /* This assignment transfers all values of the data structure, including
     the pointer to the double array, which we want to keep.
  */
  *potential = *potentialDistortion;
  potential->y = ptr;
  /* the potential should have the same abscissa range as the density,
     i.e. the abscissa of both functions line up. */
  if (!potential->y) {
    potential->y = SDDS_Malloc(sizeof(*potential->y) * potential->points);
  }

  ptr = distribution->y;
  /* This assignment transfers all values, including
     the pointer to the double array, which we preserved.
  */
  *distribution = *potentialDistortion;
  distribution->y = ptr;
  /* the distribution should have the same abscissa range as the density,
     i.e. the abscissa of both functions line up. */
  if (!distribution->y) {
    distribution->y = SDDS_Malloc(sizeof(*distribution->y) * distribution->points);
  }
  for (i = 0; i < potential->points; i++) {
    time = distribution->xStart + i * distribution->xDelta;
    /* this works only with single RF and a harmonic potential. With a harmonic cavity
       where the total potential is non-harmonic this is not right. */
    if (singleRF) {
      rfPotential = sqr(time) / 2.0 / sqr(length);
      potential->y[i] = rfPotential + 1.0 / VrfDot / sqr(length) * potentialDistortion->y[i];
    } else {
      potential->y[i] = parameters->revFrequency / (parameters->energyMeV * 1e6 * parameters->momentumCompaction * sqr(parameters->sigmaDelta)) * (rfPotentialFn->y[i] + potentialDistortion->y[i]);
    }
    distribution->y[i] = exp(-potential->y[i]);
  }
}

void normalizeDensityFunction(FUNCTION *density, FUNCTION *distribution, double charge) {
  long i;
  double integral;

  /* integrate exp(-H) to find normalization constant */
  integral = 0.0;
  for (i = 0; i < distribution->points; i++) {
    /* could use more sophistication */
    integral += distribution->y[i];
  }
  integral *= distribution->xDelta;
  for (i = 0; i < distribution->points; i++) {
    density->y[i] = (charge / integral) * distribution->y[i];
  }
}

void writeResults(SDDS_TABLE *resultsPage, FUNCTION *density,
                  FUNCTION *potential, FUNCTION *rfPotential, FUNCTION *rfVoltageFn, FUNCTION *potentialDistortion,
                  FUNCTION *Vind, double charge, double averageCurrent,
                  long converged, RFPARAMETERS *rfParameters, RINGPARAMETERS *parameters) {
  long i;
  double *current, *actualDensity, *time, averageTau, sigmaTau;
  double timeSum, timeSqrSum, currentSum;

  time = SDDS_Malloc(sizeof(*time) * density->points);
  current = SDDS_Malloc(sizeof(*current) * density->points);
  actualDensity = SDDS_Malloc(sizeof(*actualDensity) * density->points);
  timeSum = 0.0;
  timeSqrSum = 0.0;
  currentSum = 0.0;
  for (i = 0; i < density->points; i++) {
    time[i] = density->xStart + i * density->xDelta;
    current[i] = density->y[i] * density->yFactor;               /* convert to A units */
    actualDensity[i] = density->y[i] * density->yFactor / c_mks; /* convert to C/m units */
    timeSum += time[i] * current[i];
    timeSqrSum += sqr(time[i]) * current[i];
    currentSum += current[i];
  }
  averageTau = timeSum / currentSum;
  sigmaTau = sqrt(timeSqrSum / currentSum - sqr(averageTau));

  if (!SDDS_SetParameters(resultsPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Convergence", converged ? "Solution converged" : "Solution did not converge",
                          "Charge", charge,
                          "AverageCurrent", averageCurrent,
                          "SuperPeriods", parameters->superPeriods,
                          "Energy", parameters->energyMeV,
                          "SigmaTau", sigmaTau,
                          "SigmaZ", sigmaTau * c_mks,
                          "AverageTau", averageTau,
                          "AverageZ", averageTau * c_mks,
                          "MainRfVoltage", rfParameters->mainRfVoltage,
                          "MainRfHarmonic", rfParameters->mainRfHarmonic,
                          "MainRfPhase", rfParameters->mainRfPhase,
                          "HarmonicRfVoltage", rfParameters->HHCvoltage,
                          "HarmonicFactor", rfParameters->HHCharmonicFactor,
                          "HarmonicRfPhase", rfParameters->HHCphase,
                          NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (density->points > 0) {
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, current, density->points, "Current"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, time, density->points, "Time"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, actualDensity, density->points, "Density"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, Vind->y, density->points, "WakeField"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, rfVoltageFn->y, density->points, "RfVoltage"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, potential->y, density->points, "PotentialWell"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, rfPotential->y, density->points, "RfPotentialWell"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!SDDS_SetColumn(resultsPage, SDDS_SET_BY_NAME, potentialDistortion->y, density->points, "PotentialWellDistortion"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_WritePage(resultsPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (density->points > 0) {
    free(time);
    free(current);
    free(actualDensity);
  }
}

void printFunction(char *label, FUNCTION *data) {
  long i;

  fprintf(stderr, "Structure %s:\n", label);
  fprintf(stderr, "\t%s.xStart: %g\n", label, data->xStart);
  fprintf(stderr, "\t%s.xDelta: %g\n", label, data->xDelta);
  fprintf(stderr, "\t%s.points: %ld\n", label, data->points);
  fprintf(stderr, "\t%s.offset: %ld\n", label, data->offset);
  for (i = 0; i < 3; i++) {
    fprintf(stderr, "\t%s.y[%ld]: %g\n", label, i, data->y[i]);
  }
  fprintf(stderr, "\t...\n");
  for (i = data->points / 2 - 2; i < data->points / 2 + 3; i++) {
    fprintf(stderr, "\t%s.y[%ld]: %g\n", label, i, data->y[i]);
  }
  fprintf(stderr, "\t...\n");
  for (i = data->points - 3; i < data->points; i++) {
    fprintf(stderr, "\t%s.y[%ld]: %g\n", label, i, data->y[i]);
  }
  fflush(stderr);
}

void makeBBRWakeFunction(FUNCTION *wake, double dt, long points,
                         double Q, double R, double omega, double rw, double T0) {
  double Qp, omegap, t;
  long i;

  wake->xStart = 0;
  wake->xDelta = dt;
  wake->points = points;
  wake->offset = 0;
  wake->xFactor = wake->yFactor = 1;

  Qp = sqrt(Q * Q - 0.25);
  omegap = omega * Qp / Q;

  wake->y = tmalloc(sizeof(*(wake->y)) * points);
  for (i = 0; i < points; i++) {
    t = i * dt;
    wake->y[i] = (omega * R / Q) * exp(-omega * t / (2 * Q)) * (cos(omegap * t) - sin(omegap * t) / (2 * Qp));
  }
  if (rw)
    wake->y[0] += rw / dt;
}
