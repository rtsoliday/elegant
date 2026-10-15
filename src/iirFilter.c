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
  short haveGain;

  if (!inputFile)
    return 0;
  if (maxFilters <= 0 || filterBank == NULL)
    bombElegantVA("Error: readIIRFilter called with invalid arguments for file %s\n", inputFile);
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, inputFile))
    bombElegantVA("Error: Problem with IIR filter file %s\n", inputFile);
  if (SDDS_CheckColumn(&SDDSin, (char *)"denominator", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, (char *)"numerator", NULL, SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK)
    bombElegantVA("Error: Problem with existence/type of columns in IIR filter file %s\n", inputFile);
  /* The "gain" parameter is optional; when present its per-page value scales that stage's output. */
  haveGain = (SDDS_CheckParameter(&SDDSin, (char *)"gain", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK);

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
    filterBank[i].gain = 1.0;
    if (haveGain && !SDDS_GetParameterAsDouble(&SDDSin, (char *)"gain", &filterBank[i].gain))
      bombElegantVA("Error: Problem retrieving gain parameter for IIR filter file %s, page %d\n", inputFile, code);
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

/* Apply one IIR filter stage to input x, advancing its circular-buffer state, and
   return that stage's output y.  Factored out of applyIIRFilter so the parallel bank
   (applyIIRFilter) and the series cascade (applyIIRFilterCascade) share the identical
   per-stage arithmetic---and therefore produce bit-for-bit identical results for a
   single stage.  The caller must ensure filter->nTerms > 0. */
static double applyOneIIRStage(IIRFILTER *filter, double x) {
  double y;
  long j, ib, d;

  if ((--(filter->iBuffer)) < 0)
    filter->iBuffer = filter->nTerms - 1;

  ib = filter->iBuffer;
  filter->xn[ib] = x;
  filter->yn[ib] = 0; /* for convenience in doing sums */

  y = 0;
  d = 0;
  for (j = ib; j < filter->nTerms; j++, d++) {
    y += filter->xn[j] * filter->bn[d];
    y -= filter->yn[j] * filter->an[d];
  }
  for (j = 0; j < ib; j++, d++) {
    if (d >= filter->nTerms)
      bombElegantVA("Problem (1) with circular buffer handling for IIR filter\n");
    y += filter->xn[j] * filter->bn[d];
    y -= filter->yn[j] * filter->an[d];
  }
  if (d != filter->nTerms) {
    bombElegantVA("Problem (2) with circular buffer handling for IIR filter\n");
  }
  /* Store the unscaled output in the feedback buffer so the stage's own transfer function
     H(z) is unchanged; the optional gain multiplies only the value passed on (to the next
     cascade stage, or into the parallel-bank sum, or to the final output). */
  filter->yn[ib] = y;
  return y * filter->gain;
}

/* Parallel filter bank: every stage sees the same input x, and the stage outputs are
   summed.  Multiple pages in the filter file thus form one composite transfer function
   H(z) = Sum_i H_i(z).  This is the long-standing behavior (used e.g. by RFMODE LLRF
   loops); it is unchanged. */
double applyIIRFilter(IIRFILTER *filterBank, long nFilters, double x) {
  double output;
  long i;

  output = 0;
  for (i = 0; i < nFilters; i++) {
    if (filterBank[i].nTerms <= 0)
      continue;
    output += applyOneIIRStage(&filterBank[i], x);
  }

  return output;
}

/* Series cascade: each stage's output feeds the next stage's input, and the value
   returned is the last stage's output.  Multiple pages in the filter file thus form one
   composite transfer function H(z) = Prod_i H_i(z), with the number of coefficients free
   to vary from stage to stage.  A stage with no terms is a pass-through (x unchanged).
   For a single stage this is bit-for-bit identical to applyIIRFilter. */
double applyIIRFilterCascade(IIRFILTER *filterBank, long nFilters, double x) {
  long i;

  for (i = 0; i < nFilters; i++) {
    if (filterBank[i].nTerms <= 0)
      continue;
    x = applyOneIIRStage(&filterBank[i], x);
  }

  return x;
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
