/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "sliceAnalysis.h"
#ifdef HAVE_GPU
#  include <gpu_base.h>
#endif

static double tmp_safe_sqrt;
#define SAFE_SQRT(x) ((tmp_safe_sqrt = (x)) < 0 ? 0.0 : sqrt(tmp_safe_sqrt))

static SLICE_OUTPUT *sliceOutput = NULL;

long defineSliceParameters(SLICE_OUTPUT *sliceOutput, long slice);

void clearSliceAnalysis() {
#if USE_MPI
  return;
#endif
  if (sliceOutput && sliceOutput->active && sliceOutput->filename) {
    SDDS_Terminate(&(sliceOutput->SDDSout));
    SDDS_ClearErrors();
    if (sliceOutput->enx)
      free(sliceOutput->enx);
    if (sliceOutput->eny)
      free(sliceOutput->eny);
    if (sliceOutput->ecnx)
      free(sliceOutput->ecnx);
    if (sliceOutput->ecny)
      free(sliceOutput->ecny);
    if (sliceOutput->betacx)
      free(sliceOutput->betacx);
    if (sliceOutput->betacy)
      free(sliceOutput->betacy);
    if (sliceOutput->alphacx)
      free(sliceOutput->alphacx);
    if (sliceOutput->alphacy)
      free(sliceOutput->alphacy);
    if (sliceOutput->charge)
      free(sliceOutput->charge);
    if (sliceOutput->particles)
      free(sliceOutput->particles);
    if (sliceOutput->duration)
      free(sliceOutput->duration);
    if (sliceOutput->Sdelta)
      free(sliceOutput->Sdelta);
    if (sliceOutput->Cx)
      free(sliceOutput->Cx);
    if (sliceOutput->Cy)
      free(sliceOutput->Cy);
    if (sliceOutput->Cxp)
      free(sliceOutput->Cxp);
    if (sliceOutput->Cyp)
      free(sliceOutput->Cyp);
    if (sliceOutput->Cdelta)
      free(sliceOutput->Cdelta);
    if (sliceOutput->Ct)
      free(sliceOutput->Ct);

    if (sliceOutput->enxIndex)
      free(sliceOutput->enxIndex);
    if (sliceOutput->enyIndex)
      free(sliceOutput->enyIndex);
    if (sliceOutput->ecnxIndex)
      free(sliceOutput->ecnxIndex);
    if (sliceOutput->ecnyIndex)
      free(sliceOutput->ecnyIndex);
    if (sliceOutput->betacxIndex)
      free(sliceOutput->betacxIndex);
    if (sliceOutput->betacyIndex)
      free(sliceOutput->betacyIndex);
    if (sliceOutput->alphacxIndex)
      free(sliceOutput->alphacxIndex);
    if (sliceOutput->alphacyIndex)
      free(sliceOutput->alphacyIndex);
    if (sliceOutput->chargeIndex)
      free(sliceOutput->chargeIndex);
    if (sliceOutput->particlesIndex)
      free(sliceOutput->particlesIndex);
    if (sliceOutput->durationIndex)
      free(sliceOutput->durationIndex);
    if (sliceOutput->SdeltaIndex)
      free(sliceOutput->SdeltaIndex);
    if (sliceOutput->CxIndex)
      free(sliceOutput->CxIndex);
    if (sliceOutput->CyIndex)
      free(sliceOutput->CyIndex);
    if (sliceOutput->CxpIndex)
      free(sliceOutput->CxpIndex);
    if (sliceOutput->CypIndex)
      free(sliceOutput->CypIndex);
    if (sliceOutput->CdeltaIndex)
      free(sliceOutput->CdeltaIndex);
    if (sliceOutput->CtIndex)
      free(sliceOutput->CtIndex);

    if (sliceOutput->enxMemNum)
      free(sliceOutput->enxMemNum);
    if (sliceOutput->enyMemNum)
      free(sliceOutput->enyMemNum);
    if (sliceOutput->ecnxMemNum)
      free(sliceOutput->ecnxMemNum);
    if (sliceOutput->ecnyMemNum)
      free(sliceOutput->ecnyMemNum);
    if (sliceOutput->betacxMemNum)
      free(sliceOutput->betacxMemNum);
    if (sliceOutput->betacyMemNum)
      free(sliceOutput->betacyMemNum);
    if (sliceOutput->alphacxMemNum)
      free(sliceOutput->alphacxMemNum);
    if (sliceOutput->alphacyMemNum)
      free(sliceOutput->alphacyMemNum);
    if (sliceOutput->chargeMemNum)
      free(sliceOutput->chargeMemNum);
    if (sliceOutput->particlesMemNum)
      free(sliceOutput->particlesMemNum);
    if (sliceOutput->durationMemNum)
      free(sliceOutput->durationMemNum);
    if (sliceOutput->SdeltaMemNum)
      free(sliceOutput->SdeltaMemNum);
    if (sliceOutput->CxMemNum)
      free(sliceOutput->CxMemNum);
    if (sliceOutput->CyMemNum)
      free(sliceOutput->CyMemNum);
    if (sliceOutput->CxpMemNum)
      free(sliceOutput->CxpMemNum);
    if (sliceOutput->CypMemNum)
      free(sliceOutput->CypMemNum);
    if (sliceOutput->CdeltaMemNum)
      free(sliceOutput->CdeltaMemNum);
    if (sliceOutput->CtMemNum)
      free(sliceOutput->CtMemNum);

    if (sliceOutput->sliceFound)
      free(sliceOutput->sliceFound);
  }
}

void setupSliceAnalysis(NAMELIST_TEXT *nltext, RUN *run,
                        OUTPUT_FILES *output_data) {
  SDDS_DATASET *SDDSout;

#if USE_MPI
  /* This function will be parallelized in the future */
  printf("Slice analysis is not supported in this version of parallel elegant.\n");
  MPI_Barrier(MPI_COMM_WORLD);
  return;
#endif

  /* process namelist text */
  if (processNamelist(&slice_analysis, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &slice_analysis);

  sliceOutput = &(output_data->sliceAnalysis);

  if (!output || !strlen(output))
    bombElegant("provide a filename for slice_analysis output", NULL);
  if (n_slices < 0)
    bombElegant("invalid n_slice value", NULL);
  if (s_start < 0 || s_start > s_end)
    bombElegant("invalid s_start and/or s_end values", NULL);
  if ((s_start > 0 || s_end != DBL_MAX) && final_values_only)
    bombElegant("giving s_start or s_end with final_values_only=1 is not meaningful.", NULL);
  clearSliceAnalysis();

  sliceOutput->active = 1;
  sliceOutput->nSlices = n_slices;
  sliceOutput->finalValuesOnly = final_values_only;
  sliceOutput->sStart = s_start;
  sliceOutput->sEnd = s_end;

  if (!(sliceOutput->enx = malloc(sizeof(*(sliceOutput->enx)) * (n_slices + 2))) ||
      !(sliceOutput->eny = malloc(sizeof(*(sliceOutput->eny)) * (n_slices + 2))) ||
      !(sliceOutput->ecnx = malloc(sizeof(*(sliceOutput->ecnx)) * (n_slices + 2))) ||
      !(sliceOutput->ecny = malloc(sizeof(*(sliceOutput->ecny)) * (n_slices + 2))) ||
      !(sliceOutput->betacx = malloc(sizeof(*(sliceOutput->betacx)) * (n_slices + 2))) ||
      !(sliceOutput->betacy = malloc(sizeof(*(sliceOutput->betacy)) * (n_slices + 2))) ||
      !(sliceOutput->alphacx = malloc(sizeof(*(sliceOutput->alphacx)) * (n_slices + 2))) ||
      !(sliceOutput->alphacy = malloc(sizeof(*(sliceOutput->alphacy)) * (n_slices + 2))) ||
      !(sliceOutput->charge = malloc(sizeof(*(sliceOutput->charge)) * (n_slices + 2))) ||
      !(sliceOutput->particles = malloc(sizeof(*(sliceOutput->particles)) * (n_slices + 2))) ||
      !(sliceOutput->duration = malloc(sizeof(*(sliceOutput->duration)) * (n_slices + 2))) ||
      !(sliceOutput->Sdelta = malloc(sizeof(*(sliceOutput->Sdelta)) * (n_slices + 2))) ||
      !(sliceOutput->Cx = malloc(sizeof(*(sliceOutput->Cx)) * (n_slices + 2))) ||
      !(sliceOutput->Cy = malloc(sizeof(*(sliceOutput->Cy)) * (n_slices + 2))) ||
      !(sliceOutput->Cxp = malloc(sizeof(*(sliceOutput->Cxp)) * (n_slices + 2))) ||
      !(sliceOutput->Cyp = malloc(sizeof(*(sliceOutput->Cyp)) * (n_slices + 2))) ||
      !(sliceOutput->Cdelta = malloc(sizeof(*(sliceOutput->Cdelta)) * (n_slices + 2))) ||
      !(sliceOutput->Ct = malloc(sizeof(*(sliceOutput->Ct)) * (n_slices + 2))))
    bombElegant("memory allocation failure (setupSliceAnalysis)", NULL);

  if (!(sliceOutput->enxIndex = malloc(sizeof(*(sliceOutput->enxIndex)) * (n_slices + 2))) ||
      !(sliceOutput->enyIndex = malloc(sizeof(*(sliceOutput->enyIndex)) * (n_slices + 2))) ||
      !(sliceOutput->ecnxIndex = malloc(sizeof(*(sliceOutput->ecnxIndex)) * (n_slices + 2))) ||
      !(sliceOutput->ecnyIndex = malloc(sizeof(*(sliceOutput->ecnyIndex)) * (n_slices + 2))) ||
      !(sliceOutput->betacxIndex = malloc(sizeof(*(sliceOutput->betacxIndex)) * (n_slices + 2))) ||
      !(sliceOutput->betacyIndex = malloc(sizeof(*(sliceOutput->betacyIndex)) * (n_slices + 2))) ||
      !(sliceOutput->alphacxIndex = malloc(sizeof(*(sliceOutput->alphacxIndex)) * (n_slices + 2))) ||
      !(sliceOutput->alphacyIndex = malloc(sizeof(*(sliceOutput->alphacyIndex)) * (n_slices + 2))) ||
      !(sliceOutput->chargeIndex = malloc(sizeof(*(sliceOutput->chargeIndex)) * (n_slices + 2))) ||
      !(sliceOutput->particlesIndex = malloc(sizeof(*(sliceOutput->particlesIndex)) * (n_slices + 2))) ||
      !(sliceOutput->durationIndex = malloc(sizeof(*(sliceOutput->durationIndex)) * (n_slices + 2))) ||
      !(sliceOutput->SdeltaIndex = malloc(sizeof(*(sliceOutput->SdeltaIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CxIndex = malloc(sizeof(*(sliceOutput->CxIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CyIndex = malloc(sizeof(*(sliceOutput->CyIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CxpIndex = malloc(sizeof(*(sliceOutput->CxpIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CypIndex = malloc(sizeof(*(sliceOutput->CypIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CdeltaIndex = malloc(sizeof(*(sliceOutput->CdeltaIndex)) * (n_slices + 2))) ||
      !(sliceOutput->CtIndex = malloc(sizeof(*(sliceOutput->CtIndex)) * (n_slices + 2))) ||
      !(sliceOutput->sliceFound = malloc(sizeof(*(sliceOutput->sliceFound)) * (n_slices + 2))))
    bombElegant("memory allocation failure (setupSliceAnalysis)", NULL);

  if (!(sliceOutput->enxMemNum = malloc(sizeof(*(sliceOutput->enxMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->enyMemNum = malloc(sizeof(*(sliceOutput->enyMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->ecnxMemNum = malloc(sizeof(*(sliceOutput->ecnxMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->ecnyMemNum = malloc(sizeof(*(sliceOutput->ecnyMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->betacxMemNum = malloc(sizeof(*(sliceOutput->betacxMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->betacyMemNum = malloc(sizeof(*(sliceOutput->betacyMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->alphacxMemNum = malloc(sizeof(*(sliceOutput->alphacxMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->alphacyMemNum = malloc(sizeof(*(sliceOutput->alphacyMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->chargeMemNum = malloc(sizeof(*(sliceOutput->chargeMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->particlesMemNum = malloc(sizeof(*(sliceOutput->particlesMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->durationMemNum = malloc(sizeof(*(sliceOutput->durationMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->SdeltaMemNum = malloc(sizeof(*(sliceOutput->SdeltaMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CxMemNum = malloc(sizeof(*(sliceOutput->CxMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CyMemNum = malloc(sizeof(*(sliceOutput->CyMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CxpMemNum = malloc(sizeof(*(sliceOutput->CxpMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CypMemNum = malloc(sizeof(*(sliceOutput->CypMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CdeltaMemNum = malloc(sizeof(*(sliceOutput->CdeltaMemNum)) * (n_slices + 2))) ||
      !(sliceOutput->CtMemNum = malloc(sizeof(*(sliceOutput->CtMemNum)) * (n_slices + 2))))
    bombElegant("memory allocation failure (setupSliceAnalysis)", NULL);

  sliceOutput->filename = compose_filename(output, run->rootname);
  SDDSout = &(sliceOutput->SDDSout);
  if (!SDDS_InitializeOutputElegant(SDDSout, SDDS_BINARY, 0, NULL, NULL, sliceOutput->filename) ||
      0 > SDDS_DefineParameter(SDDSout, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
      !SDDS_DefineSimpleParameter(SDDSout, "Step", NULL, SDDS_LONG) ||
      !SDDS_DefineSimpleColumn(SDDSout, "ElementName", NULL, SDDS_STRING) ||
      !SDDS_DefineSimpleColumn(SDDSout, "s", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "etax", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "etaxp", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "etay", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "etayp", NULL, SDDS_DOUBLE)) {
    fprintf(stderr, "Unable define SDDS parameter for file %s\n", sliceOutput->filename);
    fflush(stderr);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (n_slices <= 1) {
    if (!defineSliceParameters(sliceOutput, 0)) {
      fprintf(stderr, "Unable define SDDS parameters for file %s\n", sliceOutput->filename);
      fflush(stderr);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  } else {
    long slice;
    /* slice 0 is the nominal (no slicing)
     * slice N+1 is the average over the slices
     */
    for (slice = 0; slice <= n_slices + 1; slice++) {
      if (!defineSliceParameters(sliceOutput, slice)) {
        fprintf(stderr, "Unable define SDDS parameters for file %s\n", sliceOutput->filename);
        fflush(stderr);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  if (!SDDS_WriteLayout(SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
}

long defineSliceParameters(SLICE_OUTPUT *sliceOutput, long slice) {
  SDDS_DATASET *SDDSout;
  char buffer[100], sliceNumString[20];

#if USE_MPI
  return 1;
#endif

  SDDSout = &(sliceOutput->SDDSout);
  if (slice && sliceOutput->nSlices > 1) {
    if (slice <= sliceOutput->nSlices) {
      sprintf(sliceNumString, "Slice%02ld", slice);
    } else {
      /* "slice" N+1 is the average over all slices */
      sprintf(sliceNumString, "Ave");
    }
  } else
    sliceNumString[0] = 0;

  sprintf(buffer, "enx%s", sliceNumString);
  if ((sliceOutput->enxIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->enxMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "eny%s", sliceNumString);
  if ((sliceOutput->enyIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->enyMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "ecnx%s", sliceNumString);
  if ((sliceOutput->ecnxIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->ecnxMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "ecny%s", sliceNumString);
  if ((sliceOutput->ecnyIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->ecnyMemNum[slice] = rpn_create_mem(buffer, 0);

  sprintf(buffer, "betacx%s", sliceNumString);
  if ((sliceOutput->betacxIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->betacxMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "betacy%s", sliceNumString);
  if ((sliceOutput->betacyIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->betacyMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "alphacx%s", sliceNumString);
  if ((sliceOutput->alphacxIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->alphacxMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "alphacy%s", sliceNumString);
  if ((sliceOutput->alphacyIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->alphacyMemNum[slice] = rpn_create_mem(buffer, 0);

  sprintf(buffer, "charge%s", sliceNumString);
  if ((sliceOutput->chargeIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "C", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->chargeMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "particles%s", sliceNumString);
  if ((sliceOutput->particlesIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->particlesMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "duration%s", sliceNumString);
  if ((sliceOutput->durationIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->durationMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Sdelta%s", sliceNumString);
  if ((sliceOutput->SdeltaIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->SdeltaMemNum[slice] = rpn_create_mem(buffer, 0);

  sprintf(buffer, "Cx%s", sliceNumString);
  if ((sliceOutput->CxIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CxMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Cy%s", sliceNumString);
  if ((sliceOutput->CyIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CyMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Cxp%s", sliceNumString);
  if ((sliceOutput->CxpIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CxpMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Cyp%s", sliceNumString);
  if ((sliceOutput->CypIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CypMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Cdelta%s", sliceNumString);
  if ((sliceOutput->CdeltaIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CdeltaMemNum[slice] = rpn_create_mem(buffer, 0);
  sprintf(buffer, "Ct%s", sliceNumString);
  if ((sliceOutput->CtIndex[slice] =
       SDDS_DefineColumn(SDDSout, buffer, NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sliceOutput->CtMemNum[slice] = rpn_create_mem(buffer, 0);

  return 1;
}

void performSliceAnalysisOutput(SLICE_OUTPUT *sliceOutput, double **particle, long particles,
                                long newPage, long step, double Po, double charge,
                                char *elementName, double elementPosition,
                                long timeGiven) {
  SDDS_DATASET *SDDSout;
  long slice;

#if USE_MPI
  return;
#endif

  if (!sliceOutput || !sliceOutput->active)
    return;
  if (sliceOutput->sStart < sliceOutput->sEnd &&
      (sliceOutput->sStart > elementPosition ||
       sliceOutput->sEnd < elementPosition))
    return;
  SDDSout = &(sliceOutput->SDDSout);
  if (newPage) {
    sliceOutput->rows = 0;
    if (!SDDS_StartPage(SDDSout, 1) ||
        !SDDS_SetParameters(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Step", step,
                            NULL)) {
      fprintf(stderr, "Unable write data to file %s\n", sliceOutput->filename);
      fflush(stderr);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  performSliceAnalysis(sliceOutput, particle, particles, Po, charge, timeGiven);
  if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sliceOutput->rows,
                         "ElementName", elementName,
                         "s", elementPosition,
                         "etax", sliceOutput->eta[0],
                         "etaxp", sliceOutput->eta[1],
                         "etay", sliceOutput->eta[2],
                         "etayp", sliceOutput->eta[3],
                         NULL)) {
    fprintf(stderr, "Unable write data to file %s\n", sliceOutput->filename);
    fflush(stderr);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (slice = 0; slice <= sliceOutput->nSlices + 1; slice++) {
    if (slice == 1 && sliceOutput->nSlices == 0)
      break;
    if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                           sliceOutput->rows,
                           sliceOutput->enxIndex[slice], sliceOutput->enx[slice],
                           sliceOutput->enyIndex[slice], sliceOutput->eny[slice],
                           sliceOutput->ecnxIndex[slice], sliceOutput->ecnx[slice],
                           sliceOutput->ecnyIndex[slice], sliceOutput->ecny[slice],
                           sliceOutput->betacxIndex[slice], sliceOutput->betacx[slice],
                           sliceOutput->betacyIndex[slice], sliceOutput->betacy[slice],
                           sliceOutput->alphacxIndex[slice], sliceOutput->alphacx[slice],
                           sliceOutput->alphacyIndex[slice], sliceOutput->alphacy[slice],
                           sliceOutput->chargeIndex[slice], sliceOutput->charge[slice],
                           sliceOutput->particlesIndex[slice], sliceOutput->particles[slice],
                           sliceOutput->durationIndex[slice], sliceOutput->duration[slice],
                           sliceOutput->SdeltaIndex[slice], sliceOutput->Sdelta[slice],
                           sliceOutput->CxIndex[slice], sliceOutput->Cx[slice],
                           sliceOutput->CyIndex[slice], sliceOutput->Cy[slice],
                           sliceOutput->CxpIndex[slice], sliceOutput->Cxp[slice],
                           sliceOutput->CypIndex[slice], sliceOutput->Cyp[slice],
                           sliceOutput->CdeltaIndex[slice], sliceOutput->Cdelta[slice],
                           sliceOutput->CtIndex[slice], sliceOutput->Ct[slice],
                           -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    rpn_store(sliceOutput->enx[slice], NULL, sliceOutput->enxMemNum[slice]);
    rpn_store(sliceOutput->eny[slice], NULL, sliceOutput->enyMemNum[slice]);
    rpn_store(sliceOutput->ecnx[slice], NULL, sliceOutput->ecnxMemNum[slice]);
    rpn_store(sliceOutput->ecny[slice], NULL, sliceOutput->ecnyMemNum[slice]);
    rpn_store(sliceOutput->betacx[slice], NULL, sliceOutput->betacxMemNum[slice]);
    rpn_store(sliceOutput->betacy[slice], NULL, sliceOutput->betacyMemNum[slice]);
    rpn_store(sliceOutput->alphacx[slice], NULL, sliceOutput->alphacxMemNum[slice]);
    rpn_store(sliceOutput->alphacy[slice], NULL, sliceOutput->alphacyMemNum[slice]);
    rpn_store(sliceOutput->charge[slice], NULL, sliceOutput->chargeMemNum[slice]);
    rpn_store(sliceOutput->particles[slice], NULL, sliceOutput->particlesMemNum[slice]);
    rpn_store(sliceOutput->duration[slice], NULL, sliceOutput->durationMemNum[slice]);
    rpn_store(sliceOutput->Sdelta[slice], NULL, sliceOutput->SdeltaMemNum[slice]);
    rpn_store(sliceOutput->Cx[slice], NULL, sliceOutput->CxMemNum[slice]);
    rpn_store(sliceOutput->Cxp[slice], NULL, sliceOutput->CxpMemNum[slice]);
    rpn_store(sliceOutput->Cy[slice], NULL, sliceOutput->CyMemNum[slice]);
    rpn_store(sliceOutput->Cyp[slice], NULL, sliceOutput->CypMemNum[slice]);
    rpn_store(sliceOutput->Cdelta[slice], NULL, sliceOutput->CdeltaMemNum[slice]);
    rpn_store(sliceOutput->Ct[slice], NULL, sliceOutput->CtMemNum[slice]);
  }
  sliceOutput->rows += 1;
  if (!SDDS_UpdatePage(SDDSout, FLUSH_TABLE))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
}

/* The device's numbers for a slice when it has them (gpu_slice_moments()), else computeSliceMoments().
 * They are the CPU's, which GPU_VERIFY checks by computing both */
static long sliceMoments(long slice, double aveCoord[6], double S[6][6], double **particle, long particles,
                         double tMin, double tMax, long onDevice, long *deviceCount, double *deviceC,
                         double *deviceS) {
  long count, j, k;

  if (onDevice) {
    for (j = 0; j < 6; j++) {
      aveCoord[j] = deviceC[slice * 6 + j];
      for (k = 0; k < 6; k++)
        S[j][k] = deviceS[slice * 36 + j * 6 + k];
    }
    return deviceCount[slice];
  }
  count = computeSliceMoments(aveCoord, S, particle, particles, tMin, tMax);
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  if (deviceCount) {
    long same = count == deviceCount[slice];
    for (j = 0; j < 6 && same; j++) {
      same = aveCoord[j] == deviceC[slice * 6 + j];
      for (k = 0; k < 6 && same; k++)
        same = S[j][k] == deviceS[slice * 36 + j * 6 + k];
    }
    if (!same) {
      fprintf(stderr, "elegant CUDA VERIFY failed for slice analysis slice %ld: CPU count %ld, GPU count %ld\n",
              slice, count, deviceCount[slice]);
      exit(1);
    }
  }
#endif
  return count;
}

void performSliceAnalysis(SLICE_OUTPUT *sliceOutput, double **particle, long particles,
                          double Po, double charge, long timeGiven) {
  double *sSave = NULL;
  long i, slice, nSlices;
  double tMinAll, tMaxAll, tMin, tMax;
  long count, slicesFound = 0;
  double aveCoord[6], S[6][6];
  long onDevice = 0, *deviceCount = NULL;
  double *deviceC = NULL, *deviceS = NULL, deviceTMin = 0, deviceTMax = 0;

#if USE_MPI
  return;
#endif

  if (!particles) {
    printf("no particles left---can't compute slice analysis");
    fflush(stdout);
    /* fill in some dummy values */
    for (slice = 0; slice < sliceOutput->nSlices + 1; slice++) {
      sliceOutput->enx[slice] = sliceOutput->eny[slice] =
        sliceOutput->ecnx[slice] = sliceOutput->ecny[slice] =
        sliceOutput->betacx[slice] = sliceOutput->betacy[slice] =
        sliceOutput->alphacx[slice] = sliceOutput->alphacy[slice] =
        sliceOutput->Cx[slice] = sliceOutput->Cy[slice] =
        sliceOutput->Cxp[slice] = sliceOutput->Cyp[slice] =
        sliceOutput->Cdelta[slice] = sliceOutput->duration[slice] =
        sliceOutput->Sdelta[slice] = sliceOutput->charge[slice] =
        sliceOutput->particles[slice] = DBL_MAX;
    }
    return;
  }

#ifdef HAVE_GPU
  if (!timeGiven) {
    long slices = sliceOutput->nSlices > 1 ? sliceOutput->nSlices + 1 : 1;

    deviceCount = tmalloc(sizeof(*deviceCount) * slices);
    deviceC = tmalloc(sizeof(*deviceC) * 6 * slices);
    deviceS = tmalloc(sizeof(*deviceS) * 36 * slices);
    if (!gpu_slice_moments(particles, Po, sliceOutput->nSlices, &deviceTMin, &deviceTMax,
                           deviceCount, deviceC, deviceS)) {
      free(deviceCount);
      free(deviceC);
      free(deviceS);
      deviceCount = NULL;
      deviceC = deviceS = NULL;
    } else {
#  ifdef GPU_VERIFY
      particle = copyParticlesToCpuReadOnly("performSliceAnalysis GPU-VERIFY");
#  else
      onDevice = 1;
#  endif
    }
  }
#endif

  if (!timeGiven && !onDevice) {
    /* compute the time coordinate and replace 's' with this value.
     * save s values in a new array
     */
    if (!(sSave = malloc(sizeof(*sSave) * particles)))
      SDDS_Bomb("memory allocation failure (performSliceAnalysis)");
    computeTimeCoordinatesOnly(sSave, Po, particle, particles);
    for (i = 0; i < particles; i++)
      SWAP_DOUBLE(sSave[i], particle[i][4]);
  }

  /* compute normal values (over entire beam) */
  sliceMoments(0, aveCoord, S, particle, particles, 0, 0, onDevice, deviceCount, deviceC, deviceS);
  sliceOutput->charge[0] = charge;
  sliceOutput->particles[0] = particles;
  sliceOutput->Cx[0] = aveCoord[0];
  sliceOutput->Cxp[0] = aveCoord[1];
  sliceOutput->Cy[0] = aveCoord[2];
  sliceOutput->Cyp[0] = aveCoord[3];
  sliceOutput->Ct[0] = aveCoord[4];
  sliceOutput->Cdelta[0] = aveCoord[5];
  sliceOutput->Sdelta[0] = sqrt(S[5][5]);
  for (i = 0; i < 4; i++)
    sliceOutput->eta[i] = 0;
  if (S[5][5])
    for (i = 0; i < 4; i++)
      sliceOutput->eta[i] = S[i][5] / S[5][5];
  sliceOutput->enx[0] = Po * (1 + aveCoord[5]) * SAFE_SQRT(S[0][0] * S[1][1] - sqr(S[0][1]));
  sliceOutput->eny[0] = Po * (1 + aveCoord[5]) * SAFE_SQRT(S[2][2] * S[3][3] - sqr(S[2][3]));
  sliceOutput->ecnx[0] = Po * (1 + aveCoord[5]) * correctedEmittance(S, sliceOutput->eta, 0, 1, &sliceOutput->betacx[0], &sliceOutput->alphacx[0]);
  sliceOutput->ecny[0] = Po * (1 + aveCoord[5]) * correctedEmittance(S, sliceOutput->eta, 2, 3, &sliceOutput->betacy[0], &sliceOutput->alphacy[0]);
  /* find total bunch duration */
  if (onDevice) {
    tMinAll = deviceTMin;
    tMaxAll = deviceTMax;
  } else {
    tMaxAll = -(tMinAll = DBL_MAX);
    for (i = 0; i < particles; i++) {
      if (tMinAll > particle[i][4])
        tMinAll = particle[i][4];
      if (tMaxAll < particle[i][4])
        tMaxAll = particle[i][4];
    }
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
    if (deviceCount && (tMinAll != deviceTMin || tMaxAll != deviceTMax)) {
      fprintf(stderr, "elegant CUDA VERIFY failed for slice analysis time span: CPU %.17e %.17e GPU %.17e %.17e\n",
              tMinAll, tMaxAll, deviceTMin, deviceTMax);
      exit(1);
    }
#endif
  }
  sliceOutput->duration[0] = tMaxAll - tMinAll;

  if (sliceOutput->nSlices > 1) {
    /* compute values for each slice, plus average */
    nSlices = sliceOutput->nSlices;

    sliceOutput->enx[nSlices + 1] = sliceOutput->eny[nSlices + 1] = 0;
    sliceOutput->ecnx[nSlices + 1] = sliceOutput->ecny[nSlices + 1] = 0;
    sliceOutput->betacx[nSlices + 1] = sliceOutput->betacy[nSlices + 1] = 0;
    sliceOutput->alphacx[nSlices + 1] = sliceOutput->alphacy[nSlices + 1] = 0;
    sliceOutput->charge[nSlices + 1] = sliceOutput->duration[nSlices + 1] = 0;
    sliceOutput->particles[nSlices + 1] = sliceOutput->Sdelta[nSlices + 1] = 0;
    sliceOutput->Cx[nSlices + 1] = sliceOutput->Cxp[nSlices + 1] = 0;
    sliceOutput->Cy[nSlices + 1] = sliceOutput->Cyp[nSlices + 1] = 0;
    sliceOutput->Cdelta[nSlices + 1] = sliceOutput->Ct[nSlices + 1] = 0;

    for (slice = 1; slice <= sliceOutput->nSlices; slice++) {
      /* find boundaries of slice in time */
      tMin = (slice - 1) * (tMaxAll - tMinAll) / sliceOutput->nSlices + tMinAll;
      tMax = tMin + (tMaxAll - tMinAll) / sliceOutput->nSlices;

      sliceOutput->duration[slice] = tMax - tMin;

      count = sliceMoments(slice, aveCoord, S, particle, particles, tMin, tMax, onDevice, deviceCount,
                           deviceC, deviceS);
      sliceOutput->particles[slice] = count;
      if (count < 2)
        sliceOutput->sliceFound[slice] = 0;
      else {
        slicesFound++;
        sliceOutput->sliceFound[slice] = 1;

        sliceOutput->charge[slice] = (charge * count) / particles;
        sliceOutput->Cx[slice] = aveCoord[0];
        sliceOutput->Cxp[slice] = aveCoord[1];
        sliceOutput->Cy[slice] = aveCoord[2];
        sliceOutput->Cyp[slice] = aveCoord[3];
        sliceOutput->Ct[slice] = aveCoord[4];
        sliceOutput->Cdelta[slice] = aveCoord[5];
        sliceOutput->Sdelta[slice] = sqrt(S[5][5]);
        sliceOutput->enx[slice] = Po * SAFE_SQRT(S[0][0] * S[1][1] - sqr(S[0][1]));
        sliceOutput->eny[slice] = Po * SAFE_SQRT(S[2][2] * S[3][3] - sqr(S[2][3]));
        sliceOutput->ecnx[slice] = Po * correctedEmittance(S, sliceOutput->eta, 0, 1,
                                                           &sliceOutput->betacx[slice],
                                                           &sliceOutput->alphacx[slice]);
        sliceOutput->ecny[slice] = Po * correctedEmittance(S, sliceOutput->eta, 2, 3,
                                                           &sliceOutput->betacy[slice],
                                                           &sliceOutput->alphacy[slice]);
        sliceOutput->enx[nSlices + 1] += sliceOutput->enx[slice];
        sliceOutput->eny[nSlices + 1] += sliceOutput->eny[slice];
        sliceOutput->ecnx[nSlices + 1] += sliceOutput->ecnx[slice];
        sliceOutput->ecny[nSlices + 1] += sliceOutput->ecny[slice];
        sliceOutput->betacx[nSlices + 1] += sliceOutput->betacx[slice];
        sliceOutput->betacy[nSlices + 1] += sliceOutput->betacy[slice];
        sliceOutput->alphacx[nSlices + 1] += sliceOutput->alphacx[slice];
        sliceOutput->alphacy[nSlices + 1] += sliceOutput->alphacy[slice];
        sliceOutput->charge[nSlices + 1] += sliceOutput->charge[slice];
        sliceOutput->particles[nSlices + 1] += sliceOutput->particles[slice];
        sliceOutput->duration[nSlices + 1] += sliceOutput->duration[slice];
        sliceOutput->Sdelta[nSlices + 1] += sliceOutput->Sdelta[slice];
        sliceOutput->Cx[nSlices + 1] += sliceOutput->Cx[slice];
        sliceOutput->Cxp[nSlices + 1] += sliceOutput->Cxp[slice];
        sliceOutput->Cy[nSlices + 1] += sliceOutput->Cy[slice];
        sliceOutput->Cyp[nSlices + 1] += sliceOutput->Cyp[slice];
        sliceOutput->Cdelta[nSlices + 1] += sliceOutput->Cdelta[slice];
        sliceOutput->Ct[nSlices + 1] += sliceOutput->Ct[slice];
      }
    }

    if (!slicesFound)
      bombElegant("No valid slices found for slice property computation.", NULL);

    sliceOutput->enx[nSlices + 1] /= slicesFound;
    sliceOutput->eny[nSlices + 1] /= slicesFound;
    sliceOutput->ecnx[nSlices + 1] /= slicesFound;
    sliceOutput->ecny[nSlices + 1] /= slicesFound;
    sliceOutput->betacx[nSlices + 1] /= slicesFound;
    sliceOutput->betacy[nSlices + 1] /= slicesFound;
    sliceOutput->alphacx[nSlices + 1] /= slicesFound;
    sliceOutput->alphacy[nSlices + 1] /= slicesFound;
    sliceOutput->Sdelta[nSlices + 1] /= slicesFound;
    sliceOutput->charge[nSlices + 1] /= slicesFound;
    sliceOutput->particles[nSlices + 1] /= slicesFound;
    sliceOutput->duration[nSlices + 1] /= slicesFound;
    sliceOutput->Cx[nSlices + 1] /= slicesFound;
    sliceOutput->Cxp[nSlices + 1] /= slicesFound;
    sliceOutput->Cy[nSlices + 1] /= slicesFound;
    sliceOutput->Cyp[nSlices + 1] /= slicesFound;
    sliceOutput->Cdelta[nSlices + 1] /= slicesFound;
    sliceOutput->Ct[nSlices + 1] /= slicesFound;
  }

  if (!timeGiven && !onDevice) {
    for (i = 0; i < particles; i++)
      SWAP_DOUBLE(sSave[i], particle[i][4]);
    free(sSave);
  }
  free(deviceCount);
  free(deviceC);
  free(deviceS);
}

double correctedEmittance(double S[6][6], double eta[4], long i1, long i2,
                          double *beta, double *alpha) {
  double T1, T2, T3, ec2;
  T1 = S[i1][i1] + sqr(eta[i1]) * S[5][5] - 2 * eta[i1] * S[i1][5];
  T2 = S[i2][i2] + sqr(eta[i2]) * S[5][5] - 2 * eta[i2] * S[i2][5];
  T3 = S[i1][i2] - S[i2][5] * eta[i1] - S[i1][5] * eta[i2] + S[5][5] * eta[i1] * eta[i2];
  if ((ec2 = T1 * T2 - sqr(T3)) < 0)
    ec2 = 0;
  if (beta) {
    if (ec2)
      *beta = (S[i1][i1] + sqr(eta[i1]) * S[5][5] - 2 * eta[i1] * S[i1][5]) / sqrt(ec2);
    else
      *beta = DBL_MAX;
  }
  if (alpha) {
    if (ec2)
      *alpha = -(S[i1][i2] + eta[i1] * eta[i2] * S[5][5] -
                 eta[i1] * S[i2][5] - eta[i2] * S[i1][5]) /
        sqrt(ec2);
    else
      *alpha = DBL_MAX;
  }
  return sqrt(ec2);
}
