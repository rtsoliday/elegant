/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

#if !defined(PRESSURE_DATA_H)
#define PRESSURE_DATA_H 1
/*
 * Joe Calvey, Michael Borland 2017
 */

typedef struct {
  long nConstituents;
  long *nAtoms;
  double *Z, *A;
} GAS_DATA;


typedef struct {
  long nGasses;
  char **gasName;
  GAS_DATA *gasData;
  long nLocations;
  double *s;         /* s[j] is the location of the jth set of pressure samples */
  double **pressure; /* pressure[i][j] is the pressure of the ith species at the jth location */
  double temperature; /* in degrees K */
} PRESSURE_DATA;

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_WIN32) && defined(_MSC_VER))
__declspec(dllexport) void readGasPressureData(char *filename, PRESSURE_DATA *pressureData, double factor, long verbosity);
#else
void readGasPressureData(char *filename, PRESSURE_DATA *pressureData, double factor, long verbosity);
#endif
void computeAverageGasPressures(double sStart, double sEnd, double *pressure, PRESSURE_DATA *pressureData);
long identifyGas(GAS_DATA *gasData, char *name);

#ifdef __cplusplus
}
#endif
#endif
