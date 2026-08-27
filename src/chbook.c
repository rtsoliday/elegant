/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include <stdio.h>
#include "mdb.h"
#include "SDDS.h"
#include "constants.h"
#include "chbook.h"

void exitElegant(long status);
void bombElegant(char *error, char *usage);
void printWarning(char *s1, char *s2);
int32_t SDDS_InitializeOutputElegant(SDDS_DATASET *SDDS_dataset, int32_t data_mode,
                                     int32_t lines_per_row, const char *description,
                                     const char *contents, const char *filename);

void findBit(long value, long *Bit, long inc, long length);

book1 *chbook1(char *vName, char *units, double xmin, double xmax, int32_t xbins) {
  book1 *book;
  book = (book1 *)malloc(sizeof(*book));

  book->vname = vName;
  book->units = units;

  book->xmin = xmin;
  book->xmax = xmax;
  book->dx = (xmax - xmin) / (double)xbins;
  book->xbins = xbins;
  book->length = xbins;
  book->count = 0;
  book->total = 0;

  book->value = calloc(sizeof(*book->value), book->length);

  return book;
}

void chfill1(book1 *bName, double x, double Frequency) {
  long index;

  index = (long)((x - bName->xmin) / bName->dx);
  if (index < 0)
    index = 0;
  if (index > bName->xbins - 1)
    index = bName->xbins - 1;

  bName->value[index] += Frequency;
  bName->total += Frequency;
  bName->count++;

  return;
}

void free_hbook1(book1 *x) {
  free(x->value);
  free(x);
  return;
}

book2 *chbook2(char *xName, char *yName, char *xunits, char *yunits,
               double xmin, double xmax, double ymin, double ymax, int32_t xbins, int32_t ybins) {
  book2 *book;

  book = (book2 *)malloc(sizeof(*book));

  book->xname = xName;
  book->yname = yName;
  book->xunits = xunits;
  book->yunits = yunits;

  book->xmin = xmin;
  book->xmax = xmax;
  book->dx = (xmax - xmin) / (double)xbins;
  book->xbins = xbins;
  book->ymin = ymin;
  book->ymax = ymax;
  book->dy = (ymax - ymin) / (double)ybins;
  book->ybins = ybins;

  book->length = xbins * ybins;
  book->count = 0;
  book->total = 0;

  book->value = calloc(sizeof(*book->value), book->length);

  return book;
}

void chfill2(book2 *bName, double x, double y, double Frequency) {
  long index[2], i;

  index[0] = (long)((x - bName->xmin) / bName->dx);
  if (index[0] < 0)
    index[0] = 0;
  if (index[0] > (bName->xbins - 1))
    index[0] = bName->xbins - 1;

  index[1] = (long)((y - bName->ymin) / bName->dy);
  if (index[1] < 0)
    index[1] = 0;
  if (index[1] > (bName->ybins - 1))
    index[1] = bName->ybins - 1;

  i = index[0] * bName->xbins + index[1];
  bName->value[i] += Frequency;
  bName->total += Frequency;
  bName->count++;

  return;
}

void free_hbook2(book2 *x) {
  free(x->value);
  free(x);
  return;
}

ntuple *chbookn(char **vName, char **units, long ND, double *xmin, double *xmax, int32_t *xbins, long offset) {
  ntuple *book;
  long i;

  book = (ntuple *)malloc(sizeof(*book));

  book->nD = ND;

  book->vname = calloc(sizeof(*book->vname), ND);
  book->units = calloc(sizeof(*book->units), ND);
  book->xmin = calloc(sizeof(*book->xmin), ND);
  book->xmax = calloc(sizeof(*book->xmax), ND);
  book->dx = calloc(sizeof(*book->dx), ND);
  book->xbins = calloc(sizeof(*book->xbins), ND);

  book->count = 0;
  book->total = 0;
  book->length = 1;
  for (i = 0; i < ND; i++) {
    book->vname[i] = vName[i + offset];
    book->units[i] = units[i + offset];
    book->length *= xbins[i + offset];
    book->xbins[i] = xbins[i + offset];
    book->xmin[i] = xmin[i + offset];
    book->xmax[i] = xmax[i + offset];
    book->dx[i] = (xmax[i + offset] - xmin[i + offset]) / (double)xbins[i + offset];
  }

  book->value = calloc(sizeof(*book->value), book->length);

  return book;
}

void chfilln(ntuple *bName, double *x, double Frequency, long offset) {
  long index;
  long i, temp;

  index = 0;
  for (i = 0; i < bName->nD; i++) {
    temp = (long)((x[i + offset] - bName->xmin[i]) / bName->dx[i]);
    if (temp < 0)
      temp = 0;
    if (temp > bName->xbins[i] - 1)
      temp = bName->xbins[i] - 1;
    if (i > 0)
      index = index * bName->xbins[i - 1] + temp;
    if (i == 0)
      index = temp;
  }

  bName->value[index] += Frequency;
  bName->total += Frequency;
  bName->count++;
  return;
}

ntuple *readbookn(char *inputfile, long i_page) {
  ntuple *book;
  long i, iPage;
  SDDS_DATASET mhist;
  PARAMETER_DEFINITION *para;
  int32_t nD;
  char buffer[1024], dimensionString[100];

  if (!SDDS_InitializeInput(&mhist, inputfile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  while (1) {
    if ((iPage = SDDS_ReadPage(&mhist)) <= 0)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (iPage == i_page)
      break;
  }
  switch (SDDS_CheckParameter(&mhist, "ND", NULL, SDDS_LONG, stderr)) {
  case SDDS_CHECK_NONEXISTENT:
    printf("\tParameter ND not found in file %s.\n", inputfile);
    exitElegant(1);
    break;
  case SDDS_CHECK_WRONGTYPE:
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exitElegant(1);
    break;
  case SDDS_CHECK_OKAY:
    break;
  default:
    printf("Unexpected result from SDDS_CheckParameter routine while checking parameter ND in file %s.\n", inputfile);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    exitElegant(1);
    break;
  }

  book = (ntuple *)malloc(sizeof(*book));
  if (!SDDS_GetParameter(&mhist, "ND", &nD))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  book->nD = (long)nD;

  book->vname = calloc(sizeof(*book->vname), book->nD);
  book->units = calloc(sizeof(*book->units), book->nD);
  book->xmin = calloc(sizeof(*book->xmin), book->nD);
  book->xmax = calloc(sizeof(*book->xmax), book->nD);
  book->dx = calloc(sizeof(*book->dx), book->nD);
  book->xbins = calloc(sizeof(*book->xbins), book->nD);

  book->length = SDDS_RowCount(&mhist);
  book->total = 0;
  for (i = 0; i < book->nD; i++) {
    sprintf(dimensionString, "%02ld", i);
    sprintf(buffer, "Variable%sName", dimensionString);
    if (!(para = SDDS_GetParameterDefinition(&mhist, buffer)))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    book->vname[i] = para->name;
    book->units[i] = para->units;
    sprintf(buffer, "Variable%sDimension", dimensionString);
    if (!SDDS_GetParameter(&mhist, buffer, &book->xbins[i]))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    sprintf(buffer, "Variable%sMin", dimensionString);
    if (!SDDS_GetParameter(&mhist, buffer, &book->xmin[i]))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    sprintf(buffer, "Variable%sMax", dimensionString);
    if (!SDDS_GetParameter(&mhist, buffer, &book->xmax[i]))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    sprintf(buffer, "Variable%sInterval", dimensionString);
    if (!SDDS_GetParameter(&mhist, buffer, &book->dx[i]))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* book->value = (double*) calloc(sizeof(*book->value), book->length); */
  if (!(book->value = SDDS_GetColumnInDoubles(&mhist, "Frequency")))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  for (book->total = 0, i = 0; i < book->length; i++)
    book->total += book->value[i];

  if (!SDDS_Terminate(&mhist))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return book;
}

/* The x0 is normalized input [0-1]; x is actual input [xmin, xmax]. */
double interpolate_bookn(ntuple *bName, double *x0, double *x, long offset,
                         long normalize, long normalInput, long zero_Edge, long verbose) {
  static double **Coef = NULL;
  static long **grid = NULL, *Bit = NULL;
  static long max_nD = 0;
  double value, result;
  long j, vIndex, temp, flag = 0;
  int64_t i, np;
  char warningBuffer[16384];

  np = (long)ipow(2, bName->nD);
  if (max_nD < bName->nD) {
    if (Coef)
      free_czarray_2d((void **)Coef, max_nD, 2);
    Coef = NULL;
    if (grid)
      free_czarray_2d((void **)grid, max_nD, 2);
    grid = NULL;
    if (Bit)
      free(Bit);
    Bit = NULL;
  }
  if (!Coef)
    Coef = (double **)czarray_2d(sizeof(double), bName->nD, 2);
  if (!grid)
    grid = (long **)czarray_2d(sizeof(long), bName->nD, 2);
  if (!Bit)
    Bit = calloc(sizeof(long), bName->nD);
  memset(Bit, 0, bName->nD * sizeof(*Bit));
  if (max_nD < bName->nD)
    max_nD = bName->nD;

  for (i = 0; i < bName->nD; i++) {
    if (normalInput) {
      x[i + offset] = x0[i + offset] * (bName->xmax[i] - bName->xmin[i]) + bName->xmin[i];
    } else {
      if (x[i + offset] > bName->xmax[i]) {
        if (verbose) {
          snprintf(warningBuffer, 16384,
                   "quantity is %s, xmax=%g, x=%g",
                   bName->vname[i], bName->xmax[i], x[i + offset]);
          printWarning("interpolate_bookn: value is outside upper boundary", warningBuffer);
        }
        if (zero_Edge) {
          flag = 1;
          if (verbose)
            printf("set output to zero\n");
        } else {
          x[i + offset] = bName->xmax[i];
          if (verbose)
            printf("set output to xmax\n");
        }
      }
      if (x[i + offset] < bName->xmin[i]) {
        if (verbose) {
          snprintf(warningBuffer, 16384,
                   "quantity is %s, xmin=%g, x=%g",
                   bName->vname[i], bName->xmin[i], x[i + offset]);
          printWarning("interpolate_bookn: value is outside lower boundary", warningBuffer);
        }
        if (zero_Edge) {
          flag = 1;
          if (verbose)
            printf("set output to zero\n");
        } else {
          x[i + offset] = bName->xmin[i];
          if (verbose)
            printf("set output to xmin\n");
        }
      }
      x0[i + offset] = (x[i + offset] - bName->xmin[i]) / (bName->xmax[i] - bName->xmin[i]);
    }
    if (flag)
      return 0;

    grid[i][0] = (long)(x0[i + offset] * bName->xbins[i] - 0.5);
    Coef[i][1] = (x[i + offset] - bName->xmin[i]) / bName->dx[i] - grid[i][0] - 0.5;
    if (Coef[i][1] < 0) {
      grid[i][0] -= 1;
      Coef[i][1] += 1;
    }
    Coef[i][0] = 1. - Coef[i][1];

    grid[i][1] = grid[i][0] + 1;
    if (grid[i][1] > bName->xbins[i])
      bombElegant("interpolate_bookn --- This should not be happen", NULL);
    if (grid[i][0] < 0)
      grid[i][0] = 0;
    if (grid[i][1] == bName->xbins[i])
      grid[i][1]--;
  }

  result = 0.;
  for (i = 0; i < np; i++) {
    value = 1.;
    vIndex = 0;
    temp = 1;
    findBit(i, Bit, 2, bName->nD);
    for (j = bName->nD - 1; j >= 0; j--) {
      value *= Coef[j][Bit[j]];
      vIndex += grid[j][Bit[j]] * temp;
      temp = temp * bName->xbins[j];
    }
    result += value * bName->value[vIndex];
  }

  if (normalize) {
    for (i = 0; i < bName->nD; i++)
      result /= bName->dx[i];
    return (result / bName->total);
  } else
    return result;
}

void findBit(long value, long *Bit, long inc, long length) {
  long i;

  for (i = 0; i < length; i++)
    Bit[i] = 0;

  for (i = length - 1; i >= 0; i--) {
    Bit[i] = value % inc;
    if (!(value /= inc))
      break;
  }
  return;
}

void free_hbookn(ntuple *x) {
  free(x->vname);
  free(x->units);
  free(x->xmin);
  free(x->xmax);
  free(x->dx);
  free(x->xbins);
  free(x->value);
  free(x);
  return;
}

book1m *chbook1m(char **vName, char **units, double *xmin, double *xmax, int32_t *xbins, long column_number) {
  book1m *book;
  long i, j;

  book = (book1m *)malloc(sizeof(*book));

  book->nD = book->bins = 0;
  for (i = 0; i < column_number; i++) {
    if (!xbins[i])
      continue;
    book->nD++;
    if (book->bins < xbins[i])
      book->bins = xbins[i];
  }

  book->vname = calloc(sizeof(*book->vname), book->nD);
  book->units = calloc(sizeof(*book->units), book->nD);
  book->xmin = calloc(sizeof(*book->xmin), book->nD);
  book->xmax = calloc(sizeof(*book->xmax), book->nD);
  book->dx = calloc(sizeof(*book->dx), book->nD);

  book->count = 0;
  book->total = 0;
  book->length = book->bins;
  for (i = j = 0; i < column_number; i++) {
    if (!xbins[i])
      continue;
    book->vname[j] = vName[i];
    book->units[j] = units[i];
    book->xmin[j] = xmin[i];
    book->xmax[j] = xmax[i];
    book->dx[j] = (xmax[i] - xmin[i]) / (double)book->bins;
    j++;
  }

  book->value = (double **)czarray_2d(sizeof(double), book->length, book->nD);
  for (i = 0; i < book->length; i++)
    for (j = 0; j < book->nD; j++)
      book->value[i][j] = 0;

  return book;
}

void chfill1m(book1m *bName, double *x, double Frequency, int32_t *xbins, long column_number) {
  long index, i, j;

  for (i = j = 0; i < column_number; i++) {
    if (!xbins[i])
      continue;
    index = (long)((x[i] - bName->xmin[j]) / bName->dx[j]);
    if (index < 0)
      index = 0;
    if (index > bName->bins - 1)
      index = bName->bins - 1;
    bName->value[index][j] += Frequency;
    j++;
  }
  bName->total += Frequency;
  bName->count++;
  return;
}

void free_hbook1m(book1m *x) {
  free(x->vname);
  free(x->units);
  free(x->xmin);
  free(x->xmax);
  free(x->dx);
  free_czarray_2d((void **)x->value, x->length, x->nD);
  free(x);
  return;
}

void chprint1(book1 *bName, char *filename, char *description, char *frequencyUnits, SDDS_DEFINITION *parameter_definition,
              void **sdds_value, long n_parameters, long normalize, long verbosity, long append) {
  SDDS_DATASET outPage;
  double value;
  long i, last_index, index;
  char buffer[1024];

  last_index = -1;
  if (!append) {
    /* Open file for writting */
    if (verbosity)
      fprintf(stdout, "Opening \"%s\" for writing...\n", filename);

    if (!SDDS_InitializeOutputElegant(&outPage, SDDS_BINARY, 1,
                                      description, description, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    for (i = 0; i < n_parameters; i++) {
      if (!SDDS_ProcessParameterString(&outPage, parameter_definition[i].text, 0) ||
          (index = SDDS_GetParameterIndex(&outPage, parameter_definition[i].name) < 0)) {
        printf("Unable to define SDDS parameter for chprint1--name was:\n%s\n",
               parameter_definition[i].name);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        index = 0; /* eliminate spurious warning */
        exitElegant(1);
      }
      if (index != (last_index + 1))
        printWarning("chprint1: parameter indices for SDDS file are not sequential. This will probably cause unexpected results. File",
                     filename);
      last_index = index;
    }

    if (0 > SDDS_DefineParameter(&outPage, "VariableName", NULL, NULL,
                                 NULL, NULL, SDDS_STRING, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "VariableInterval", NULL, bName->units,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "VariableMinimum", NULL, bName->units,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "VariableMaximum", NULL, bName->units,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "VariableDimension", NULL, NULL,
                                 NULL, NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "ND", NULL, NULL,
                                 "n_Dimension Table", NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    sprintf(buffer, "%sFrequency", bName->vname);
    if (0 > SDDS_DefineColumn(&outPage, bName->vname, NULL, bName->units,
                              NULL, NULL, SDDS_DOUBLE, 0) ||
        0 > SDDS_DefineColumn(&outPage, buffer, NULL, frequencyUnits,
                              NULL, NULL, SDDS_DOUBLE, 0))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    if (!SDDS_WriteLayout(&outPage))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  } else {
    if (!SDDS_InitializeAppend(&outPage, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* Write to output file */
  if (0 > SDDS_StartPage(&outPage, bName->length) ||
      !SDDS_SetParameters(&outPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "VariableName", bName->vname,
                          "VariableInterval", bName->dx,
                          "VariableMinimum", bName->xmin,
                          "VariableMaximum", bName->xmax,
                          "VariableDimension", bName->length,
                          "ND", 1, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  for (i = 0; i < n_parameters; i++) {
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_REFERENCE,
                            i, sdds_value[i], -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  for (i = 0; i < bName->length; i++) {
    value = ((double)i + 0.5) * bName->dx + bName->xmin;
    if (normalize) {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, value, 1, bName->value[i] / bName->total, -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    } else {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, value, 1, bName->value[i], -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WritePage(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_Terminate(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  return;
}

void chprint2(book2 *bName, char *filename, char *description, SDDS_DEFINITION *parameter_definition,
              void **sdds_value, long n_parameters, long normalize, long verbosity, long append) {
  SDDS_DATASET outPage;
  char buffer[1024];
  long i, last_index, index;

  last_index = -1;
  if (!append) {
    /* Open file for writting */
    if (verbosity)
      fprintf(stdout, "Opening \"%s\" for writing...\n", filename);
    if (!SDDS_InitializeOutputElegant(&outPage, SDDS_BINARY, 1,
                                      description, description, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    for (i = 0; i < n_parameters; i++) {
      if (!SDDS_ProcessParameterString(&outPage, parameter_definition[i].text, 0) ||
          (index = SDDS_GetParameterIndex(&outPage, parameter_definition[i].name)) < 0) {
        printf("Unable to define SDDS parameter for chprint2--name was:\n%s\n",
               parameter_definition[i].name);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        index = 0; /* eliminate spurious warning */
        exitElegant(1);
      }
      if (index != (last_index + 1))
        printWarning("chprint2: parameter indices for SDDS file are not sequential. This will probably cause unexpected results. File",
                     filename);
      last_index = index;
    }

    if (0 > SDDS_DefineParameter(&outPage, "Variable1Name", NULL, bName->xunits,
                                 NULL, NULL, SDDS_STRING, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable1Interval", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable1Minimum", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable1Maximum", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable1Dimension", NULL, NULL,
                                 NULL, NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable2Name", NULL, bName->yunits,
                                 NULL, NULL, SDDS_STRING, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable2Interval", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable2Minimum", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable2Maximum", NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "Variable2Dimension", NULL, NULL,
                                 NULL, NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "ND", NULL, NULL,
                                 "n_Dimension Table", NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    sprintf(buffer, "%s-%sFrequency", bName->xname, bName->yname);
    if (0 > SDDS_DefineColumn(&outPage, buffer, NULL, NULL,
                              NULL, NULL, SDDS_DOUBLE, 0))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    if (!SDDS_WriteLayout(&outPage))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  } else {
    if (!SDDS_InitializeAppend(&outPage, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* Write to output file */
  if (0 > SDDS_StartPage(&outPage, bName->length) ||
      !SDDS_SetParameters(&outPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Variable1Name", bName->xname,
                          "Variable2Name", bName->yname,
                          "Variable1Interval", bName->dx,
                          "Variable1Minimum", bName->xmin,
                          "Variable1Maximum", bName->xmax,
                          "Variable1Dimension", bName->xbins,
                          "Variable2Interval", bName->dy,
                          "Variable2Minimum", bName->ymin,
                          "Variable2Maximum", bName->ymax,
                          "Variable2Dimension", bName->ybins,
                          "ND", 2, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  for (i = 0; i < n_parameters; i++) {
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_REFERENCE,
                            i, sdds_value[i], -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  for (i = 0; i < bName->length; i++) {
    if (normalize) {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, bName->value[i] / bName->total, -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    } else {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, bName->value[i], -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (!SDDS_WritePage(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_Terminate(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return;
}

void chprintn(ntuple *bName, char *filename, char *description, SDDS_DEFINITION *parameter_definition,
              void **sdds_value, long n_parameters, long normalize, long verbosity, long append) {
  SDDS_DATASET outPage;
  char buffer[1024], dimensionString[100];
  long i, Index[5], last_index, index;

  last_index = -1;
  if (!append) {
    /* Open file for writting */
    if (verbosity)
      fprintf(stdout, "Opening \"%s\" for writing...\n", filename);
    if (!SDDS_InitializeOutputElegant(&outPage, SDDS_BINARY, 1,
                                      description, description, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    for (i = 0; i < n_parameters; i++) {
      if (!SDDS_ProcessParameterString(&outPage, parameter_definition[i].text, 0) ||
          (index = SDDS_GetParameterIndex(&outPage, parameter_definition[i].name)) < 0) {
        printf("Unable to define SDDS parameter for chprintn--name was:\n%s\n",
               parameter_definition[i].name);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        index = 0; /* eliminate spurious warning */
        exitElegant(1);
      }
      if (index != (last_index + 1))
        printWarning("chprintn: parameter indices for SDDS file are not sequential. This will probably cause unexpected results. File",
                     filename);
      fflush(stdout);
      last_index = index;
    }

    for (i = 0; i < bName->nD; i++) {
      sprintf(dimensionString, "%02ld", i);

      sprintf(buffer, "Variable%sName", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, bName->units[i],
                                   NULL, NULL, SDDS_STRING, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sMin", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sMax", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sInterval", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sDimension", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_LONG, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    if (0 > SDDS_DefineParameter(&outPage, "ND", NULL, NULL,
                                 "n_Dimension Table", NULL, SDDS_LONG, NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (0 > SDDS_DefineColumn(&outPage, "Index", NULL, NULL,
                              NULL, NULL, SDDS_LONG, 0) ||
        0 > SDDS_DefineColumn(&outPage, "Frequency", NULL, NULL,
                              NULL, NULL, SDDS_DOUBLE, 0) ||
        0 > SDDS_DefineParameter(&outPage, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    if (!SDDS_WriteLayout(&outPage))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  } else {
    if (!SDDS_InitializeAppend(&outPage, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* Write to output file */
  if (0 > SDDS_StartPage(&outPage, bName->length))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  for (i = 0; i < n_parameters; i++) {
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_REFERENCE,
                            i, sdds_value[i], -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  Index[4] = n_parameters - 1;
  for (i = 0; i < bName->nD; i++) {
    Index[0] = Index[4] + 1;
    Index[1] = Index[0] + 1;
    Index[2] = Index[1] + 1;
    Index[3] = Index[2] + 1;
    Index[4] = Index[3] + 1;
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                            Index[0], bName->vname[i],
                            Index[1], bName->xmin[i],
                            Index[2], bName->xmax[i],
                            Index[3], bName->dx[i],
                            Index[4], bName->xbins[i], -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ND", bName->nD, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  for (i = 0; i < bName->length; i++) {
    if (normalize) {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, i, 1, bName->value[i] / bName->total, -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    } else {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, i, 1, bName->value[i], -1))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WritePage(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_Terminate(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return;
}

void chprint1m(book1m *bName, char *filename, char *description, SDDS_DEFINITION *parameter_definition,
               void **sdds_value, long n_parameters, long normalize, long verbosity, long append) {
  SDDS_DATASET outPage;
  char name[100], units[100], freq[100], buffer[1024], dimensionString[100];
  long i, j, Index[5], last_index, index;
  double *value1, *value2;

  last_index = -1;
  if (!append) {
    /* Open file for writting */
    if (verbosity)
      fprintf(stdout, "Opening \"%s\" for writing...\n", filename);
    if (!SDDS_InitializeOutputElegant(&outPage, SDDS_BINARY, 1,
                                      description, description, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    for (i = 0; i < n_parameters; i++) {
      if (!SDDS_ProcessParameterString(&outPage, parameter_definition[i].text, 0) ||
          (index = SDDS_GetParameterIndex(&outPage, parameter_definition[i].name)) < 0) {
        printf("Unable to define SDDS parameter for chprint1m--name was:\n%s\n",
               parameter_definition[i].name);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        index = 0; /* eliminate spurious warning */
        exitElegant(1);
      }
      if (index != (last_index + 1))
        printWarning("chprint1m: parameter indices for SDDS file are not sequential. This will probably cause unexpected results. File",
                     filename);
      last_index = index;
    }

    for (i = 0; i < bName->nD; i++) {
      sprintf(dimensionString, "%02ld", i);

      sprintf(buffer, "Variable%sName", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, bName->units[i],
                                   NULL, NULL, SDDS_STRING, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sMin", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sMax", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sInterval", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_DOUBLE, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

      sprintf(buffer, "Variable%sDimension", dimensionString);
      if (0 > SDDS_DefineParameter(&outPage, buffer, NULL, NULL,
                                   NULL, NULL, SDDS_LONG, NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    if (0 > SDDS_DefineParameter(&outPage, "ND", NULL, NULL,
                                 "n_Dimension Table", NULL, SDDS_LONG, NULL) ||
        0 > SDDS_DefineParameter(&outPage, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    for (i = 0; i < bName->nD; i++) {
      sprintf(name, "%s", bName->vname[i]);
      sprintf(units, "%s", bName->units[i]);
      sprintf(freq, "%sFrequency", bName->vname[i]);
      if (0 > SDDS_DefineColumn(&outPage, name, NULL, units,
                                NULL, NULL, SDDS_DOUBLE, 0) ||
          0 > SDDS_DefineColumn(&outPage, freq, NULL, NULL,
                                NULL, NULL, SDDS_DOUBLE, 0))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_WriteLayout(&outPage))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  } else {
    if (!SDDS_InitializeAppend(&outPage, filename))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  /* Write to output file */
  if (0 > SDDS_StartPage(&outPage, bName->length))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  for (i = 0; i < n_parameters; i++) {
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_REFERENCE,
                            i, sdds_value[i], -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  Index[4] = n_parameters - 1;
  for (i = 0; i < bName->nD; i++) {
    Index[0] = Index[4] + 1;
    Index[1] = Index[0] + 1;
    Index[2] = Index[1] + 1;
    Index[3] = Index[2] + 1;
    Index[4] = Index[3] + 1;
    if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                            Index[0], bName->vname[i],
                            Index[1], bName->xmin[i],
                            Index[2], bName->xmax[i],
                            Index[3], bName->dx[i],
                            Index[4], bName->bins, -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_SetParameters(&outPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "ND", bName->nD, NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  Index[1] = -1;
  value1 = calloc(sizeof(*value1), bName->length);
  value2 = calloc(sizeof(*value2), bName->length);
  for (i = 0; i < bName->nD; i++) {
    sprintf(name, "%s", bName->vname[i]);
    sprintf(freq, "%sFrequency", bName->vname[i]);
    for (j = 0; j < bName->length; j++) {
      value1[j] = ((double)j + 0.5) * bName->dx[i] + bName->xmin[i];
      value2[j] = bName->value[j][i];
      if (normalize)
        value2[j] /= bName->total;
    }
    Index[0] = Index[1] + 1;
    Index[1] = Index[0] + 1;
    for (j = 0; j < bName->length; j++) {
      if (!SDDS_SetRowValues(&outPage, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             j, name, value1[j], freq, value2[j], NULL))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WritePage(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_Terminate(&outPage))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  free(value1);
  free(value2);
  return;
}
