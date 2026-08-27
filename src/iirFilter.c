/*************************************************************************\
* Copyright (c) 2015 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2015 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"

int readIIRFilter(IIRFILTER *filterBank, long maxFilters, char *inputFile) {
  SDDS_DATASET SDDSin;
  int i, j, code;
  double a0;

  if (!inputFile)
    return 0;
  if (maxFilters <= 0 || filterBank == NULL)
    bombElegantVA("Error: readIIRFilter called with invalid arguments for file %s\n", inputFile);
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, inputFile))
    bombElegantVA("Error: Problem with IIR filter file %s\n", inputFile);
  if (SDDS_CheckColumn(&SDDSin, (char *)"denominator", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"numerator", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
    bombElegantVA("Error: Problem with existence/type of columns in IIR filter file %s\n", inputFile);

  for (i = 0; i < maxFilters; i++) {
    filterBank[i].nTerms = filterBank[i].iBuffer = 0;
    if (filterBank[i].an)
      free(filterBank[i].an);
    if (filterBank[i].bn)
      free(filterBank[i].bn);
    if (filterBank[i].xn)
      free(filterBank[i].xn);
    if (filterBank[i].yn)
      free(filterBank[i].yn);
    filterBank[i].an = filterBank[i].bn = filterBank[i].xn = filterBank[i].yn = NULL;
  }

  i = 0;
  while ((code = SDDS_ReadPage(&SDDSin)) > 0 && i < maxFilters) {
    filterBank[i].nTerms = SDDS_RowCount(&SDDSin);

    if (!(filterBank[i].an = SDDS_GetColumnInDoubles(&SDDSin, (char *)"denominator")) ||
        !(filterBank[i].bn = SDDS_GetColumnInDoubles(&SDDSin, (char *)"numerator")))
      bombElegantVA("Error: Problem retrieving column data for IIR filter file %s, page %d\n", inputFile, code);

    if (!(filterBank[i].xn = calloc(filterBank[i].nTerms, sizeof(*(filterBank[i].xn)))) ||
        !(filterBank[i].yn = calloc(filterBank[i].nTerms, sizeof(*(filterBank[i].yn)))))
      bombElegantVA("Error: Memory allocation problem for IIR filter file %s, page %d, %ld terms\n", inputFile, code, filterBank[i].nTerms);

    if ((a0 = filterBank[i].an[0]) == 0)
      bombElegantVA("Error: first denominator filter coefficient is zero in IIR filter file %s, page %d\n", inputFile, code);
    for (j = 0; j < filterBank[i].nTerms; j++) {
      filterBank[i].an[j] /= a0;
      filterBank[i].bn[j] /= a0;
    }
    i++;
  }
  if (code == 0)
    bombElegantVA("Error: IIR filter file %s has invalid page(s)\n", inputFile);
  if (i >= maxFilters)
    bombElegantVA("Error: IIR filter file %s has too many pages (%ld maximum allowed)\n", inputFile, maxFilters);

  if (!SDDS_Terminate(&SDDSin))
    bombElegantVA("Error: Problem terminating IIR filter file %s\n", inputFile);

  return i;
}

double applyIIRFilter(IIRFILTER *filterBank, long nFilters, double x) {
  double y, output;
  long i, j, ib, d;

  output = 0;

  for (i = 0; i < nFilters; i++) {
    if (filterBank[i].nTerms <= 0)
      continue;
    if ((--(filterBank[i].iBuffer)) < 0)
      filterBank[i].iBuffer = filterBank[i].nTerms - 1;

    ib = filterBank[i].iBuffer;
    filterBank[i].xn[ib] = x;
    filterBank[i].yn[ib] = 0; /* for convenience in doing sums */

    y = 0;
    d = 0;
    for (j = ib; j < filterBank[i].nTerms; j++, d++) {
      y += filterBank[i].xn[j] * filterBank[i].bn[d];
      y -= filterBank[i].yn[j] * filterBank[i].an[d];
    }
    for (j = 0; j < ib; j++, d++) {
      if (d >= filterBank[i].nTerms)
        bombElegantVA("Problem (1) with circular buffer handling for IIR filter\n");
      y += filterBank[i].xn[j] * filterBank[i].bn[d];
      y -= filterBank[i].yn[j] * filterBank[i].an[d];
    }
    if (d != filterBank[i].nTerms) {
      bombElegantVA("Problem (2) with circular buffer handling for IIR filter\n");
    }
    filterBank[i].yn[ib] = y;
    output += y;
  }

  return output;
}

void freeIIRFilterMemory(IIRFILTER *filterBank, long nFilters) {
  long i;
  for (i = 0; i < nFilters; i++) {
    if (filterBank[i].an)
      free(filterBank[i].an);
    if (filterBank[i].bn)
      free(filterBank[i].bn);
    if (filterBank[i].xn)
      free(filterBank[i].xn);
    if (filterBank[i].yn)
      free(filterBank[i].yn);
    filterBank[i].an = filterBank[i].bn = filterBank[i].xn = filterBank[i].yn = NULL;
    filterBank[i].iBuffer = filterBank[i].nTerms = 0;
  }
}
