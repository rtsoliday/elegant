/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: pressureData.c
 * purpose: read and average residual-gas pressure data for ion/scattering
 *          calculations
 *
 * Joe Calvey, Michael Borland 2017
 */
#include "mdb.h"
#include "SDDS.h"
#include "constants.h"
#include "pressureData.h"

long checkSddsColumn(SDDS_TABLE *SDDS_table, char *name, char *units)
{
  char *units1;
  if (SDDS_GetColumnIndex(SDDS_table, name)<0)
    return(0);
  if (SDDS_GetColumnInformation(SDDS_table, "units", &units1, SDDS_GET_BY_NAME, name)!=SDDS_STRING) {
    SDDS_SetError("units field of column has wrong data type!");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
  }
  if (!units || SDDS_StringIsBlank(units)) {
    if (!units1)
      return(1);
    if (SDDS_StringIsBlank(units1)) {
      free(units1);
      return(1);
    }
    return(0);
  }
  if (!units1)
    return(0);
  if (strcmp(units, units1)==0) {
    free(units1);
    return(1);
  }
  free(units1);
  return(0);
}

static long isKnownGasName(char *name);

void readGasPressureData(char *filename, PRESSURE_DATA *pressureData, double factor, long verbosity)
{
  /* Assumed file structure:
   * Parameters:
   * Gasses --- SDDS_STRING giving comma- or space-separated list of gas species, e.g., "H2O H2 N2 O2 CO2 CO CH4".
   *            Optional: if absent, every column whose name matches a known species is used.
   * Temperature --- SDDS_FLOAT or SDDS_DOUBLE giving temperature in degrees K. Defaults to 293.
   * Columns:
   * s         --- SDDS_FLOAT or SDDS_DOUBLE giving location in the lattice
   * <gasName> --- SDDS_FLOAT or SDDS_DOUBLE giving pressure of <gasName> in Torr or nT
   */

  SDDS_DATASET SDDSin;
  char *gasColumnList, *ptr, **columnName=NULL;
  int32_t nColumns=0;
  long i, gasesGiven = 0;
  double dsMin, dsMax, ds, pressureMultiplier;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename))
    bombVA("Failed to initialize input from %s", filename);

  if (!checkSddsColumn(&SDDSin, "s", "m"))
    bombVA("Column 's' is missing or does not have units of 'm' in %s", filename);
  switch (SDDS_CheckParameter(&SDDSin, "Gasses", NULL, SDDS_STRING, NULL)) {
  case SDDS_CHECK_OK:
    gasesGiven = 1;
    break;
  case SDDS_CHECK_NONEXISTENT:
    /* Gasses is optional; if absent, the species are inferred from the column names below. */
    gasesGiven = 0;
    break;
  default:
    bombVA("Parameter \"Gasses\" in %s is present but not of string type", filename);
    break;
  }

  if (SDDS_ReadPage(&SDDSin)<=0)
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);

  switch (SDDS_CheckParameter(&SDDSin, "Temperature", "K", SDDS_ANY_FLOATING_TYPE, NULL)) {
  case SDDS_CHECK_OK:
    if (!SDDS_GetParameterAsDouble(&SDDSin, "Temperature", &pressureData->temperature) ||
        pressureData->temperature<=0)
      bombVA("Problem reading 'Temperature' from %s. Check for valid value (got %le).\n", filename,
                    pressureData->temperature);
    break;
  case SDDS_CHECK_NONEXISTENT:
    pressureData->temperature = 273+20;
    printf("Parameter 'Temperature' missing from %s, assuming %le K\n", filename, pressureData->temperature);
    break;
  case SDDS_CHECK_WRONGTYPE:
    bombVA("Parameter 'Temperature' in %s has wrong type. Expect SDDS_DOUBLE or SDDS_FLOAT.\n", filename);
    break;
  case SDDS_CHECK_WRONGUNITS:
    bombVA("Parameter 'Temperature' in %s has wrong units. Expect 'K'.\n", filename);
    break;
  default:
    bombVA("Unexpected value checking existence, units, and type for 'Temperature' in %s\n", filename);
    break;
  }

  pressureData->nGasses = 0;
  pressureData->gasName = NULL;
  if (gasesGiven) {
    if (!SDDS_GetParameter(&SDDSin, "Gasses", &gasColumnList))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
    while ((ptr=get_token(gasColumnList))!=NULL) {
      pressureData->gasName = (char**)SDDS_Realloc(pressureData->gasName, sizeof(*(pressureData->gasName))*(pressureData->nGasses+1));
      cp_str(&pressureData->gasName[pressureData->nGasses], ptr);
      pressureData->nGasses += 1;
    }
    free(gasColumnList);
  } else {
    /* No Gasses parameter: use any column whose name matches a known species. */
    if (!(columnName=SDDS_GetColumnNames(&SDDSin, &nColumns)))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
    for (i=0; i<nColumns; i++) {
      if (isKnownGasName(columnName[i])) {
        pressureData->gasName = (char**)SDDS_Realloc(pressureData->gasName, sizeof(*(pressureData->gasName))*(pressureData->nGasses+1));
        cp_str(&pressureData->gasName[pressureData->nGasses], columnName[i]);
        pressureData->nGasses += 1;
      }
    }
    SDDS_FreeStringArray(columnName, nColumns);
    free(columnName);
    if (pressureData->nGasses==0)
      bombVA("No \"Gasses\" parameter in %s and no columns match a known gas species", filename);
  }
  if (verbosity) {
    printf("%ld gasses %s in %s: ", pressureData->nGasses, gasesGiven ? "listed" : "found", filename);
    for (i=0; i<pressureData->nGasses; i++)
      printf("%s%c", pressureData->gasName[i], i==(pressureData->nGasses-1) ? '\n' : ' ');
  }
  pressureData->gasData = tmalloc(sizeof(*(pressureData->gasData))*pressureData->nGasses);
  for (i=0; i<pressureData->nGasses; i++) {
    if (!identifyGas(pressureData->gasData+i, pressureData->gasName[i])) {
      fprintf(stderr, "unknown gas: \"%s\"\n", pressureData->gasName[i]);
    }
  }

  pressureData->nLocations = SDDS_RowCount(&SDDSin);
  if (verbosity)
    printf("Gas data provided at %ld s locations\n", pressureData->nLocations);
  if (!(pressureData->s=SDDS_GetColumnInDoubles(&SDDSin, "s")))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);

  /* One row pointer per gas; each row is filled below by SDDS_GetColumnInDoubles
     (which allocates its own nLocations-long array), so only the pointer array is needed. */
  pressureData->pressure = (double**)tmalloc(sizeof(*(pressureData->pressure))*pressureData->nGasses);
  for (i=0; i<pressureData->nGasses; i++) {
    pressureMultiplier = 1;
    if (!checkSddsColumn(&SDDSin, pressureData->gasName[i], "Torr") && !checkSddsColumn(&SDDSin, pressureData->gasName[i], "T")) {
      pressureMultiplier = 1e-9;
      if (!checkSddsColumn(&SDDSin, pressureData->gasName[i], "nT") && !checkSddsColumn(&SDDSin, pressureData->gasName[i], "nTorr"))
        bombVA("Column \"%s\" is missing, not floating-point type, or does not have units of \"Torr\" or \"nT\" in %s",
                      pressureData->gasName[i], filename);
    }
    if (!(pressureData->pressure[i] = SDDS_GetColumnInDoubles(&SDDSin, pressureData->gasName[i]))) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
    }
    /* Convert to Torr and include external factor supplied by caller */
    pressureMultiplier *= factor;
    if (pressureMultiplier!=1) {
      long j;
      for (j=0; j<pressureData->nLocations; j++) {
        pressureData->pressure[i][j] *= pressureMultiplier;
      }
    }
  }

  dsMax = -(dsMin = DBL_MAX);
  for (i=1; i<pressureData->nLocations; i++) {
    ds = pressureData->s[i] - pressureData->s[i-1];
    if (ds<=0)
      bombVA("s data is not monotonically increasing in pressure data file %s (%le, %le)", filename, pressureData->s[i-1], pressureData->s[i]);
    if (dsMin>ds)
      dsMin = ds;
    if (dsMax<ds)
      dsMax = ds;
  }
  if (fabs(1-dsMin/dsMax)>1e-3)
    bombVA("s data is not uniformly spaced to within desired 0.1% in pressure data file %s (dsMin=%le, dsMax=%le)", filename,
                  dsMin, dsMax);

  if (verbosity) {
    printf("Finished reading pressure data file %s\n", filename);
    fflush(stdout);
  }
}

void computeAverageGasPressures(double sStart, double sEnd, double *pressure, PRESSURE_DATA *pressureData)
{
  double sum;
  long iGas, iLocation, iStart, iEnd;

  /* Could improve this by interpolating the pressure at sStart and sEnd */

  /* Find the indices spanning the desired region */
  iStart = iEnd = -1;
  for (iLocation=0; iLocation<pressureData->nLocations; iLocation++) {
    if (pressureData->s[iLocation]>=sStart && iStart==-1)
      iStart = iLocation;
    if (pressureData->s[iLocation]<=sEnd)
      iEnd = iLocation;
    else
      break;
  }
  if (iStart==-1 || iEnd==-1 || iEnd<=iStart)
    bombVA("Failed to find indices corresponding to pressure region s:[%le, %le] m\n",
                  sStart, sEnd);

  for (iGas=0; iGas<pressureData->nGasses; iGas++) {
    sum = 0;
    for (iLocation=iStart; iLocation<=iEnd; iLocation++)
      sum += pressureData->pressure[iGas][iLocation];
    pressure[iGas] = sum/(iEnd-iStart+1);
  }
}

/* Composition of each supported gas species, broken into constituent elements:
   nAtoms[j] atoms of the element with atomic number Z[j] and mass number A[j].
   Only elements H, C, N, O, and F appear. To add a species, add a row here. */
#define MAX_GAS_CONSTITUENTS 3
typedef struct {
  char *name;
  long nConstituents;
  long nAtoms[MAX_GAS_CONSTITUENTS];
  double Z[MAX_GAS_CONSTITUENTS];
  double A[MAX_GAS_CONSTITUENTS];
} GAS_SPECIES;

static GAS_SPECIES gasSpeciesList[] = {
  /* name    nCons  nAtoms        Z            A          */
  {"H2",     1,     {2},          {1},         {1}},
  {"H2O",    2,     {2, 1},       {1, 8},      {1, 16}},
  {"N2",     1,     {2},          {7},         {14}},
  {"O2",     1,     {2},          {8},         {16}},
  {"CO2",    2,     {1, 2},       {6, 8},      {12, 16}},
  {"CO",     2,     {1, 1},       {6, 8},      {12, 16}},
  {"CH4",    2,     {1, 4},       {6, 1},      {12, 1}},
  {"C",      1,     {1},          {6},         {12}},
  {"CH3",    2,     {1, 3},       {6, 1},      {12, 1}},
  {"OH",     2,     {1, 1},       {8, 1},      {16, 1}},
  {"C2H2",   2,     {2, 2},       {6, 1},      {12, 1}},
  {"C2H4",   2,     {2, 4},       {6, 1},      {12, 1}},
  {"COH",    3,     {1, 1, 1},    {6, 8, 1},   {12, 16, 1}},
  {"C2H6",   2,     {2, 6},       {6, 1},      {12, 1}},
  {"C2H5",   2,     {2, 5},       {6, 1},      {12, 1}},
  {"CF3",    2,     {1, 3},       {6, 9},      {12, 19}},
};
#define N_GAS_SPECIES ((long)(sizeof(gasSpeciesList)/sizeof(gasSpeciesList[0])))

static long isKnownGasName(char *name)
{
  long i;
  for (i=0; i<N_GAS_SPECIES; i++)
    if (strcmp(name, gasSpeciesList[i].name)==0)
      return 1;
  return 0;
}

long identifyGas(GAS_DATA *gasData, char *name)
{
  long i, j;
  for (i=0; i<N_GAS_SPECIES; i++) {
    if (strcmp(name, gasSpeciesList[i].name)==0) {
      gasData->nConstituents = gasSpeciesList[i].nConstituents;
      gasData->nAtoms = tmalloc(sizeof(*(gasData->nAtoms))*(gasData->nConstituents));
      gasData->Z = tmalloc(sizeof(*(gasData->Z))*(gasData->nConstituents));
      gasData->A = tmalloc(sizeof(*(gasData->A))*(gasData->nConstituents));
      for (j=0; j<gasData->nConstituents; j++) {
        gasData->nAtoms[j] = gasSpeciesList[i].nAtoms[j];
        gasData->Z[j] = gasSpeciesList[i].Z[j];
        gasData->A[j] = gasSpeciesList[i].A[j];
      }
      return 1;
    }
  }
  return 0;
}
