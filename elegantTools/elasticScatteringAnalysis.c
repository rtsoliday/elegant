/*************************************************************************\
 * Copyright (c) 2018 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2018 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* program: elasticScatteringAnalysis
 * purpose: take elastic scattering tracking output from elegant along with
 *          twiss output and pressure distribution, compute elastic scattering
 *          loss distribution etc.
 *
 * Michael Borland, 2018
 *
 */
#include "mdb.h"
#include "scan.h"
#include "SDDS.h"
#include "pressureData.h"

#define SET_TWISS 0
#define SET_PRESSURE 1
#define SET_VERBOSE 2
#define SET_STORED_CHARGE 3
#define N_OPTIONS 4

char *option[N_OPTIONS] = {
  "twiss",
  "pressure",
  "verbose",
  "storedcharge",
};

char *USAGE1 = "elasticScatteringAnalysis <SDDSinputfile> <outputRootname>\n\
  [-twiss=<filename>] [-pressure=<filename>[,periodic]]\n\
  [-storedCharge=<nanoCoulombs>] [-verbose]\n\n\
The input file should be created by the &elastic_scattering command in Pelegant.\n\
Three output files are created with different extensions:\n\
 + .sdds : out-scattering rate and other quantities as a function of scattering location s.\n\
 + .full : contribution to out-scattering rate from each lost particle.\n\
-twiss     Give name of file from twiss_output. Used to determine length of accelerator, so must\n\
           cover the entire ring.\n\
-pressure  Give name of file with pressure data vs s. If non-periodic, should cover the full circumference\n\
           as defined by the 's' column in the twiss file. Otherwise, should cover a full lattice period.\n\
  The file should contain the following:\n\
  Parameters\n\
   Gasses --- SDDS_STRING giving comma- or space-separated list of gas species, e.g., \"H2O H2 N2 O2 CO2 CO CH4\"\n\
   Temperature --- SDDS_FLOAT or SDDS_DOUBLE giving temperature in degrees K. Defaults to 293.\n\
  Columns\n\
   s         --- SDDS_FLOAT or SDDS_DOUBLE giving location in the lattice in meters\n\
   <gasName> --- SDDS_FLOAT or SDDS_DOUBLE giving pressure of <gasName> in Torr or nT\n\
-storedCharge Charge in beam, in nanocoulombs. Used to compute the loss distribution in physical units.\n\
-verbose      If given, possibly useful intermediate information is printed.\n\n\
Program by Michael Borland.  ("__DATE__
  ")\n";

#define PERIODIC_GIVEN 0x0001UL

long findMinMaxStep(double *pmin, double *pmax, double *dp, double *data, long nd);

int main(int argc, char **argv) {
  SDDS_DATASET SDDSin;
  char *inputFile, *outputRootname, *twissFile, *pressureFile;
  long i_arg, verbose, twissRows;
  SCANNED_ARG *s_arg;
  unsigned long pressureFlags;
  double storedCharge, *sTwiss, *betax, *betay, LTotal, pCentral, rateFactor;
  PRESSURE_DATA pressureData;
  char buffer[16384];
  long is, is2, ns2, ig, ic;
  /* Variables for scattering simulation output */
  long nScatter, nScatterPeriods;
  double *sScatter, *thetaScatter, *sLost, sScatterMax, *phiScatter;
  double phiMin, phiMax, dphi;
  double thetaMin, thetaMax, dtheta;
  long nScatterUnique, *indexScatterUnique, *nPhiThetaScatterUnique;
  double *sScatterUnique, *sScatterStart, *sScatterEnd;
  double *G, GWAve, ds2;
  double *rate, rateTotal, lifetime;
  /* output files */
  SDDS_DATASET SDDSmain, SDDSfull;

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&s_arg, argc, argv);
  if (argc < 2) {
    fprintf(stderr, "%s\n", USAGE1);
    return (1);
  }

  sTwiss = betax = betay = NULL;
  twissRows = 0;
  sScatter = thetaScatter = phiScatter = sLost = NULL;
  nScatter = 0;
  inputFile = outputRootname = twissFile = pressureFile = NULL;
  verbose = pressureFlags = 0;
  storedCharge = 0;

  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (s_arg[i_arg].arg_type == OPTION) {
      switch (match_string(s_arg[i_arg].list[0], option, N_OPTIONS, 0)) {
      case SET_TWISS:
        if (s_arg[i_arg].n_items != 2)
          SDDS_Bomb("give filename for -twiss option");
        twissFile = s_arg[i_arg].list[1];
        break;
      case SET_PRESSURE:
        pressureFlags = 0;
        if (s_arg[i_arg].n_items < 2)
          SDDS_Bomb("give filename for -pressure option");
        pressureFile = s_arg[i_arg].list[1];
        s_arg[i_arg].n_items -= 2;
        if (!scanItemList(&pressureFlags, s_arg[i_arg].list + 2, &s_arg[i_arg].n_items, 0,
                          "periodic", -1, NULL, 0, PERIODIC_GIVEN,
                          NULL))
          SDDS_Bomb("invalid -pressure syntax/values");
        break;
      case SET_STORED_CHARGE:
        if (s_arg[i_arg].n_items != 2)
          SDDS_Bomb("invalid -storedCharge syntax/values: use -storedCharge=<nC>");
        get_double(&storedCharge, s_arg[i_arg].list[1]);
        storedCharge *= 1e-9;
        break;
      case SET_VERBOSE:
        verbose = 1;
        break;
      default:
        fprintf(stdout, "error: unknown switch: %s\n", s_arg[i_arg].list[0]);
        fflush(stdout);
        exit(1);
        break;
      }
    } else {
      if (inputFile == NULL)
        inputFile = s_arg[i_arg].list[0];
      else if (outputRootname == NULL)
        outputRootname = s_arg[i_arg].list[0];
      else
        SDDS_Bomb("too many filenames");
    }
  }

  if (!twissFile)
    SDDS_Bomb("give -twissInput=<filename> option");
  if (!pressureFile)
    SDDS_Bomb("give -pressureInput=<filename> option");
  if (!inputFile)
    SDDS_Bomb("give input file");
  if (!outputRootname)
    SDDS_Bomb("output rootname");

  /* Read data from twiss file */
  if (!SDDS_InitializeInput(&SDDSin, twissFile)) {
    SDDS_SetError("Problem initializing input from twiss file.");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckColumn(&SDDSin, "s", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "betax", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "betay", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY) {
    fprintf(stderr,
            "elasticScatteringAnalysis: didn't find 's', 'betax', and 'betay' columns with units 'm' in %s",
            twissFile);
    exit(1);
  }
  if (SDDS_ReadPage(&SDDSin) <= 0 ||
      (twissRows = SDDS_RowCount(&SDDSin)) < 2 ||
      !(betax = SDDS_GetColumnInDoubles(&SDDSin, "betax")) ||
      !(betay = SDDS_GetColumnInDoubles(&SDDSin, "betay")) ||
      !(sTwiss = SDDS_GetColumnInDoubles(&SDDSin, "s")) ||
      !SDDS_GetParameterAsDouble(&SDDSin, "pCentral", &pCentral) ||
      !SDDS_Terminate(&SDDSin))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  /* Check for negative drifts, remove zero-length intervals */
  for (is = 1; is < twissRows; is++) {
    if (fabs(sTwiss[is] - sTwiss[is - 1]) < 1e-6) {
      if (is != (twissRows - 1)) {
        for (is2 = is; is2 < (twissRows - 1); is2++) {
          betax[is2] = betax[is2 + 1];
          betay[is2] = betay[is2 + 1];
          sTwiss[is2] = sTwiss[is2 + 1];
        }
      }
      twissRows--;
      is--;
    } else if (sTwiss[is] < sTwiss[is - 1])
      bombVA("Column s is not monotonically increasing in file %s: s[%ld]=%le, s[%ld]=%le, %ld rows total\n", twissFile,
             is - 1, sTwiss[is - 1], is, sTwiss[is], twissRows);
  }
  find_min_max(NULL, &LTotal, sTwiss, twissRows);

  readGasPressureData(pressureFile, &pressureData, 1.0, 0);

  if (!SDDS_InitializeInput(&SDDSin, inputFile)) {
    SDDS_SetError("Problem initializing input from scattering file.");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* check the input file for valid data */
  if (SDDS_CheckColumn(&SDDSin, "s", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "theta", NULL, SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "phi", NULL, SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
      SDDS_CheckColumn(&SDDSin, "sLost", NULL, SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY) {
    fprintf(stderr,
            "elasticScatteringAnalysis: one or more data quantities have the wrong units or are not present in %s",
            inputFile);
    exit(1);
  }
  if (SDDS_ReadPage(&SDDSin) <= 0 ||
      (nScatter = SDDS_RowCount(&SDDSin)) < 2 ||
      !(sScatter = SDDS_GetColumnInDoubles(&SDDSin, "s")) ||
      !(sLost = SDDS_GetColumnInDoubles(&SDDSin, "sLost")) ||
      !(thetaScatter = SDDS_GetColumnInDoubles(&SDDSin, "theta")) ||
      !(phiScatter = SDDS_GetColumnInDoubles(&SDDSin, "phi")) ||
      !SDDS_Terminate(&SDDSin))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  /* check for properly-ordered data --- would be nice to sort this internally */
  for (is = 1; is < nScatter; is++) {
    if (sScatter[is] < sScatter[is - 1]) {
      SDDS_Bomb("Please pre-sort the scattering data using sddssort -column=s");
    }
  }
  rate = tmalloc(sizeof(*rate) * nScatter);

  /* Find unique scattering locations */
  nScatterUnique = 1;
  sScatterMax = sScatter[0];
  for (is = 1; is < nScatter; is++) {
    if (sScatter[is] != sScatter[is - 1])
      nScatterUnique++;
    if (sScatterMax < sScatter[is])
      sScatterMax = sScatter[is];
  }
  nScatterPeriods = LTotal / sScatterMax + 0.5;
  if (verbose) {
    printf("%ld unique scattering locations\n", nScatterUnique);
    printf("Scattering data covers 1/%ld of the total region, assumed periodic\n", nScatterPeriods);
    fflush(stdout);
  }
  sScatterUnique = tmalloc(sizeof(*sScatterUnique) * nScatterUnique);
  indexScatterUnique = tmalloc(sizeof(*indexScatterUnique) * nScatterUnique);
  nPhiThetaScatterUnique = tmalloc(sizeof(*nPhiThetaScatterUnique) * nScatterUnique);
  sScatterUnique[0] = sScatter[0];
  indexScatterUnique[0] = 0;
  is2 = 1;
  for (is = 0; is < nScatter; is++)
    if (sScatter[is] != sScatterUnique[is2 - 1]) {
      if (is2 > (nScatterUnique - 1))
        bombVA("is2 = %ld, nScatterUnique=%ld\n", is2, nScatterUnique);
      nPhiThetaScatterUnique[is2 - 1] = is - indexScatterUnique[is2 - 1];
      sScatterUnique[is2] = sScatter[is];
      indexScatterUnique[is2] = is;
      is2++;
    }
  nPhiThetaScatterUnique[is2 - 1] = is - indexScatterUnique[is2 - 1];
  if (verbose) {
    printf("%ld unique scattering locations from s=%le to %le\n",
           nScatterUnique, sScatterUnique[0], sScatterUnique[nScatterUnique - 1]);
  }

  /* Figure out dphi step */
  phiMin = phiMax = dphi = 0;
  findMinMaxStep(&phiMin, &phiMax, &dphi, phiScatter, nScatter);
  if (verbose) {
    printf("dphi = %le\n", dphi);
    fflush(stdout);
  }

  /* Set up output files */
  /* -- main file: quantities vs unique scattering location */
  sprintf(buffer, "%s.sdds", outputRootname);
  if (fexists(buffer))
    bombVA("output filename in use: %s\n", buffer);
  if (!SDDS_InitializeOutput(&SDDSmain, SDDS_BINARY, 1, NULL, NULL, buffer) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "s", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "sStart", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "sEnd", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "GWInteg", "1/(m*s)", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "dphi", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "dtheta", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSmain, "rateSum", "1/s", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&SDDSmain, "lifetime", "h", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSmain) ||
      !SDDS_StartPage(&SDDSmain, nScatterUnique))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* -- full file: data for each scattered (lost) particle
   *     This file can be xrefed with the input file to recover other associations
   */
  sprintf(buffer, "%s.full", outputRootname);
  if (fexists(buffer))
    bombVA("output filename in use: %s\n", buffer);
  if (!SDDS_InitializeOutput(&SDDSfull, SDDS_BINARY, 1, NULL, NULL, buffer) ||
      !SDDS_DefineSimpleColumn(&SDDSfull, "s", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSfull, "sLost", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSfull, "phi", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSfull, "theta", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSfull, "rate", "1/s", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleParameter(&SDDSfull, "lifetime", "h", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSfull) ||
      !SDDS_StartPage(&SDDSfull, nScatter))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* See AOP-TN-2017-040 for calculation method */

  G = tmalloc(sizeof(*G) * pressureData.nLocations);
  for (is = 0; is < pressureData.nLocations; is++)
    G[is] = 0;
  for (ig = 0; ig < pressureData.nGasses; ig++) {
    double s1;
    s1 = 0;
    for (ic = 0; ic < pressureData.gasData[ig].nConstituents; ic++) {
      s1 += pressureData.gasData[ig].nAtoms[ic] * sqr(pressureData.gasData[ig].Z[ic]);
    }
    for (is = 0; is < pressureData.nLocations; is++) {
      G[is] += pressureData.pressure[ig][is] * s1;
    }
  }
  for (is = 0; is < pressureData.nLocations; is++) {
    /* 133.3224 converts Torr to Pascal */
    G[is] *= sqr(re_mks / (2 * pCentral)) * 133.3224 / (k_boltzmann_mks * pressureData.temperature);
  }
  /* The factor of 2 is because the phi angles are assumed on [0, pi] instead of [0, 2*pi] */
  rateFactor = 2 * c_mks / LTotal;

  /* Find endpoints surrounding each scattering point */
  sScatterStart = tmalloc(sizeof(*sScatterStart) * nScatterUnique);
  sScatterEnd = tmalloc(sizeof(*sScatterEnd) * nScatterUnique);
  for (is = 0; is < nScatterUnique; is++) {
    if (is == 0)
      sScatterStart[is] = 0;
    else
      sScatterStart[is] = (sScatterUnique[is - 1] + sScatterUnique[is]) / 2;
    if (is == (nScatterUnique - 1))
      sScatterEnd[is] = LTotal / nScatterPeriods;
    else
      sScatterEnd[is] = (sScatterUnique[is + 1] + sScatterUnique[is]) / 2;
  }

  rateTotal = 0;
  for (is = 0; is < nScatterUnique; is++) {
    double rateSum, betax0, betay0, betax1, betay1, fint;
    long iAngle, index;
    unsigned long code;

    ds2 = 0.001;
    ns2 = (sScatterEnd[is] - sScatterStart[is]) / ds2;
    if (ns2 < 2)
      bombVA("scattering points too closely spaced: ns2=%ld, sScatter: %le,  s:[%le, %le]",
             ns2, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
    ds2 = (sScatterEnd[is] - sScatterStart[is]) / (ns2 - 1);
    GWAve = 0;
    betax0 = interpolate(betax, sTwiss, twissRows, sScatterUnique[is], NULL, NULL, 1, &code, 1);
    if (code)
      bombVA("Interpolation failed (%ld) for betax: sScatter: %le,  s:[%le, %le]",
             code, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
    betay0 = interpolate(betay, sTwiss, twissRows, sScatterUnique[is], NULL, NULL, 1, &code, 1);
    if (code)
      bombVA("Interpolation failed (%ld) for betay: sScatter: %le,  s:[%le, %le]",
             code, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
    for (is2 = 0; is2 < ns2; is2++) {
      betax1 = interpolate(betax, sTwiss, twissRows, sScatterStart[is] + is2 * ds2, NULL, NULL, 1, &code, 1);
      if (code)
        bombVA("Interpolation failed (%ld) for betax: s=%le, sScatter: %le,  s:[%le, %le]",
               code, sScatterStart[is] + is2 * ds2, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
      betay1 = interpolate(betay, sTwiss, twissRows, sScatterStart[is] + is2 * ds2, NULL, NULL, 1, &code, 1);
      if (code)
        bombVA("Interpolation failed (%ld) for betay: s=%le, sScatter: %le,  s:[%le, %le]",
               code, sScatterStart[is] + is2 * ds2, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
      GWAve += sqrt(betax1 * betay1 / (betax0 * betay0)) *
        interpolate(G, pressureData.s, pressureData.nLocations,
                    fmod(sScatterStart[is] + is2 * ds2, pressureData.s[pressureData.nLocations - 1]), NULL, NULL,
                    1, &code, 1);
      if (code)
        bombVA("Interpolation failed (%ld) for G: s=%le, sScatter: %le,  s:[%le, %le]",
               code, sScatterStart[is] + is2 * ds2, sScatterUnique[is], sScatterStart[is], sScatterEnd[is]);
    }
    GWAve /= ns2;
    rateSum = 0;
    thetaMin = thetaMax = dtheta = 0;
    findMinMaxStep(&thetaMin, &thetaMax, &dtheta,
                   thetaScatter + indexScatterUnique[is],
                   nPhiThetaScatterUnique[is]);
    if (verbose) {
      printf("dtheta = %le for s=%le (%ld points)\n", dtheta, sScatterUnique[is], nPhiThetaScatterUnique[is]);
      fflush(stdout);
    }
    fint = dphi * dtheta * GWAve * rateFactor * (sScatterEnd[is] - sScatterStart[is]);
    for (iAngle = 0; iAngle < nPhiThetaScatterUnique[is]; iAngle++) {
      index = indexScatterUnique[is] + iAngle;
      rate[index] = fint * sin(thetaScatter[index]) / ipow4(sin(thetaScatter[index] / 2));
      rateSum += rate[index];
      /* Output rate vs (sScatterUnique, delta) */
      if (!SDDS_SetRowValues(&SDDSfull, SDDS_PASS_BY_VALUE | SDDS_SET_BY_NAME, index,
                             "s", sScatterUnique[is],
                             "sLost", sLost[index],
                             "theta", thetaScatter[index],
                             "phi", phiScatter[index],
                             "rate", rate[index], NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    /* Output rateSum vs sScatterUnique */
    if (!SDDS_SetRowValues(&SDDSmain, SDDS_PASS_BY_VALUE | SDDS_SET_BY_NAME, is,
                           "s", sScatterUnique[is],
                           "sStart", sScatterStart[is],
                           "sEnd", sScatterEnd[is],
                           "GWInteg", GWAve * (sScatterEnd[is] - sScatterStart[is]),
                           "dphi", dphi,
                           "dtheta", dtheta,
                           "rateSum", rateSum, NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    rateTotal += rateSum;
  }
  /* Compute lifetime in  hours */
  lifetime = 1. / (rateTotal * 3600.0);
  if (verbose) {
    printf("Lifetime is %le h\n", lifetime);
    fflush(stdout);
  }
  if (!SDDS_SetParameters(&SDDSmain, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "lifetime", lifetime, NULL) ||
      !SDDS_SetParameters(&SDDSfull, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "lifetime", lifetime, NULL) ||
      !SDDS_WritePage(&SDDSmain) ||
      !SDDS_WritePage(&SDDSfull) ||
      !SDDS_Terminate(&SDDSmain) ||
      !SDDS_Terminate(&SDDSfull))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /* Free arrays */
  free(sScatter);
  free(sLost);
  free(thetaScatter);
  free(phiScatter);
  free(rate);
  free(G);
  free(sScatterUnique);
  free(indexScatterUnique);
  free(nPhiThetaScatterUnique);
  free(sScatterStart);
  free(sScatterEnd);

  /* Free pressure data */
  free(pressureData.s);
  free_czarray_2d((void **)pressureData.pressure, pressureData.nGasses, pressureData.nLocations);
  free(pressureData.gasName);

  free_scanargs(&s_arg, argc);
  return 0;
}

long findMinMaxStep(double *pmin, double *pmax, double *dp, double *data, long nd) {
  long i;
  double p1;
  if (nd < 2)
    return 0;
  find_min_max(pmin, pmax, data, nd);
  p1 = *pmax;
  for (i = 0; i < nd; i++)
    if (data[i] < p1 && data[i] > *pmin && fabs(data[i] - *pmin) > 1e-6 * (*pmax))
      p1 = data[i];
  *dp = (p1 - *pmin);
  return 1;
}
