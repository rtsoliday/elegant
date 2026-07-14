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
#include "sasefel.h"

#define BEAMSIZE_GEOMETRIC_MEAN 0
#define BEAMSIZE_ARITHMETIC_MEAN 1
#define BEAMSIZE_HORIZONTAL 2
#define BEAMSIZE_VERTICAL 3
#define BEAMSIZE_MAXIMUM 4
#define BEAMSIZE_OPTIONS 5
#define ANALYSIS_BINS 10000

char *beamsizeOption[BEAMSIZE_OPTIONS] = {
  "geometric mean", "arithmetic mean", "horizontal", "vertical", "maximum"};

long DefineSASEParameters(SASEFEL_OUTPUT *sasefelOutput, long slice);
void SetSASEBetaEmitValues(double *emitToUse, double *betaToUse,
                           long beamsizeMode,
                           double emitx, double S11, double S12,
                           double emity, double S33, double S34, double userBeta);

void setupSASEFELAtEnd(NAMELIST_TEXT *nltext, RUN *run, OUTPUT_FILES *output_data) {
  SASEFEL_OUTPUT *sasefelOutput;
  SDDS_DATASET *SDDSout;

  /* process namelist text */
  if (processNamelist(&sasefel, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &sasefel);

  sasefelOutput = &(output_data->sasefel);

  if (beta < 0)
    bombElegant("beta < 0", NULL);
  if (undulator_K <= 0)
    bombElegant("undulator_K <= 0", NULL);
  if (undulator_period <= 0)
    bombElegant("undulator_period <= 0", NULL);
  if (n_slices < 0 || slice_fraction < 0 ||
      (n_slices == 0 && slice_fraction > 0) ||
      (n_slices > 0 && slice_fraction <= 0) ||
      n_slices * slice_fraction > 1)
    bombElegant("invalid slice parameters", NULL);
  sasefelOutput->beamsizeMode = 0;
  if (strlen(beamsize_mode) &&
      (sasefelOutput->beamsizeMode = match_string(beamsize_mode, beamsizeOption, BEAMSIZE_OPTIONS, 0)) < 0)
    bombElegant("beamsize_mode not 'geometric mean', 'arithmetic mean', 'horizontal', 'vertical', or 'maximum'",
                NULL);

  if (sasefelOutput->active && sasefelOutput->filename) {
    SDDS_Terminate(&(sasefelOutput->SDDSout));
    SDDS_ClearErrors();
    if (sasefelOutput->betaToUse)
      free(sasefelOutput->betaToUse);
    if (sasefelOutput->betaxBeam)
      free(sasefelOutput->betaxBeam);
    if (sasefelOutput->alphaxBeam)
      free(sasefelOutput->alphaxBeam);
    if (sasefelOutput->betayBeam)
      free(sasefelOutput->betayBeam);
    if (sasefelOutput->alphayBeam)
      free(sasefelOutput->alphayBeam);
    if (sasefelOutput->enx)
      free(sasefelOutput->enx);
    if (sasefelOutput->eny)
      free(sasefelOutput->eny);
    if (sasefelOutput->charge)
      free(sasefelOutput->charge);
    if (sasefelOutput->pCentral)
      free(sasefelOutput->pCentral);
    if (sasefelOutput->rmsBunchLength)
      free(sasefelOutput->rmsBunchLength);
    if (sasefelOutput->Sdelta)
      free(sasefelOutput->Sdelta);
    if (sasefelOutput->emit)
      free(sasefelOutput->emit);
    if (sasefelOutput->Cx)
      free(sasefelOutput->Cx);
    if (sasefelOutput->Cy)
      free(sasefelOutput->Cy);
    if (sasefelOutput->Cxp)
      free(sasefelOutput->Cxp);
    if (sasefelOutput->Cyp)
      free(sasefelOutput->Cyp);
    if (sasefelOutput->Ct)
      free(sasefelOutput->Ct);
    if (sasefelOutput->Cdelta)
      free(sasefelOutput->Cdelta);
    if (sasefelOutput->lightWavelength)
      free(sasefelOutput->lightWavelength);
    if (sasefelOutput->saturationLength)
      free(sasefelOutput->saturationLength);
    if (sasefelOutput->gainLength)
      free(sasefelOutput->gainLength);
    if (sasefelOutput->noisePower)
      free(sasefelOutput->noisePower);
    if (sasefelOutput->saturationPower)
      free(sasefelOutput->saturationPower);
    if (sasefelOutput->PierceParameter)
      free(sasefelOutput->PierceParameter);
    if (sasefelOutput->etaDiffraction)
      free(sasefelOutput->etaDiffraction);
    if (sasefelOutput->etaEmittance)
      free(sasefelOutput->etaEmittance);
    if (sasefelOutput->etaEnergySpread)
      free(sasefelOutput->etaEnergySpread);
    if (sasefelOutput->betaToUseIndex)
      free(sasefelOutput->betaToUseIndex);
    if (sasefelOutput->chargeIndex)
      free(sasefelOutput->chargeIndex);
    if (sasefelOutput->pCentralIndex)
      free(sasefelOutput->pCentralIndex);
    if (sasefelOutput->rmsBunchLengthIndex)
      free(sasefelOutput->rmsBunchLengthIndex);
    if (sasefelOutput->SdeltaIndex)
      free(sasefelOutput->SdeltaIndex);
    if (sasefelOutput->emitIndex)
      free(sasefelOutput->emitIndex);
    if (sasefelOutput->lightWavelengthIndex)
      free(sasefelOutput->lightWavelengthIndex);
    if (sasefelOutput->saturationLengthIndex)
      free(sasefelOutput->saturationLengthIndex);
    if (sasefelOutput->gainLengthIndex)
      free(sasefelOutput->gainLengthIndex);
    if (sasefelOutput->noisePowerIndex)
      free(sasefelOutput->noisePowerIndex);
    if (sasefelOutput->saturationPowerIndex)
      free(sasefelOutput->saturationPowerIndex);
    if (sasefelOutput->PierceParameterIndex)
      free(sasefelOutput->PierceParameterIndex);
    if (sasefelOutput->etaDiffractionIndex)
      free(sasefelOutput->etaDiffractionIndex);
    if (sasefelOutput->etaEmittanceIndex)
      free(sasefelOutput->etaEmittanceIndex);
    if (sasefelOutput->etaEnergySpreadIndex)
      free(sasefelOutput->etaEnergySpreadIndex);
    if (sasefelOutput->sliceFound)
      free(sasefelOutput->sliceFound);
  }

  sasefelOutput->active = 1;
  sasefelOutput->beta = beta;
  sasefelOutput->undulatorK = undulator_K;
  sasefelOutput->undulatorPeriod = undulator_period;
  sasefelOutput->nSlices = n_slices;
  sasefelOutput->sliceFraction = slice_fraction;

  if (!(sasefelOutput->betaToUse = malloc(sizeof(*(sasefelOutput->betaToUse)) * (n_slices + 2))) ||
      !(sasefelOutput->betaxBeam = malloc(sizeof(*(sasefelOutput->betaxBeam)) * (n_slices + 2))) ||
      !(sasefelOutput->alphaxBeam = malloc(sizeof(*(sasefelOutput->alphaxBeam)) * (n_slices + 2))) ||
      !(sasefelOutput->betayBeam = malloc(sizeof(*(sasefelOutput->betayBeam)) * (n_slices + 2))) ||
      !(sasefelOutput->alphayBeam = malloc(sizeof(*(sasefelOutput->alphayBeam)) * (n_slices + 2))) ||
      !(sasefelOutput->enx = malloc(sizeof(*(sasefelOutput->enx)) * (n_slices + 2))) ||
      !(sasefelOutput->eny = malloc(sizeof(*(sasefelOutput->eny)) * (n_slices + 2))) ||
      !(sasefelOutput->charge = malloc(sizeof(*(sasefelOutput->charge)) * (n_slices + 2))) ||
      !(sasefelOutput->pCentral = malloc(sizeof(*(sasefelOutput->pCentral)) * (n_slices + 2))) ||
      !(sasefelOutput->rmsBunchLength = malloc(sizeof(*(sasefelOutput->rmsBunchLength)) * (n_slices + 2))) ||
      !(sasefelOutput->Sdelta = malloc(sizeof(*(sasefelOutput->Sdelta)) * (n_slices + 2))) ||
      !(sasefelOutput->emit = malloc(sizeof(*(sasefelOutput->emit)) * (n_slices + 2))) ||
      !(sasefelOutput->Cx = malloc(sizeof(*(sasefelOutput->Cx)) * (n_slices + 2))) ||
      !(sasefelOutput->Cy = malloc(sizeof(*(sasefelOutput->Cy)) * (n_slices + 2))) ||
      !(sasefelOutput->Cxp = malloc(sizeof(*(sasefelOutput->Cxp)) * (n_slices + 2))) ||
      !(sasefelOutput->Cyp = malloc(sizeof(*(sasefelOutput->Cyp)) * (n_slices + 2))) ||
      !(sasefelOutput->Ct = malloc(sizeof(*(sasefelOutput->Ct)) * (n_slices + 2))) ||
      !(sasefelOutput->Cdelta = malloc(sizeof(*(sasefelOutput->Cdelta)) * (n_slices + 2))) ||
      !(sasefelOutput->lightWavelength = malloc(sizeof(*(sasefelOutput->lightWavelength)) * (n_slices + 2))) ||
      !(sasefelOutput->saturationLength = malloc(sizeof(*(sasefelOutput->saturationLength)) * (n_slices + 2))) ||
      !(sasefelOutput->gainLength = malloc(sizeof(*(sasefelOutput->gainLength)) * (n_slices + 2))) ||
      !(sasefelOutput->noisePower = malloc(sizeof(*(sasefelOutput->noisePower)) * (n_slices + 2))) ||
      !(sasefelOutput->saturationPower = malloc(sizeof(*(sasefelOutput->saturationPower)) * (n_slices + 2))) ||
      !(sasefelOutput->PierceParameter = malloc(sizeof(*(sasefelOutput->PierceParameter)) * (n_slices + 2))) ||
      !(sasefelOutput->etaDiffraction = malloc(sizeof(*(sasefelOutput->etaDiffraction)) * (n_slices + 2))) ||
      !(sasefelOutput->etaEmittance = malloc(sizeof(*(sasefelOutput->etaEmittance)) * (n_slices + 2))) ||
      !(sasefelOutput->etaEnergySpread = malloc(sizeof(*(sasefelOutput->etaEnergySpread)) * (n_slices + 2))))
    bombElegant("memory allocation failure (setupSASEFELAtEnd)", NULL);

  if (!(sasefelOutput->betaToUseIndex = malloc(sizeof(*(sasefelOutput->betaToUseIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->betaxBeamIndex = malloc(sizeof(*(sasefelOutput->betaxBeamIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->alphaxBeamIndex = malloc(sizeof(*(sasefelOutput->alphaxBeamIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->betayBeamIndex = malloc(sizeof(*(sasefelOutput->betayBeamIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->alphayBeamIndex = malloc(sizeof(*(sasefelOutput->alphayBeamIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->enxIndex = malloc(sizeof(*(sasefelOutput->enxIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->enyIndex = malloc(sizeof(*(sasefelOutput->enyIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->chargeIndex = malloc(sizeof(*(sasefelOutput->chargeIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->pCentralIndex = malloc(sizeof(*(sasefelOutput->pCentralIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->rmsBunchLengthIndex = malloc(sizeof(*(sasefelOutput->rmsBunchLengthIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->SdeltaIndex = malloc(sizeof(*(sasefelOutput->SdeltaIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->emitIndex = malloc(sizeof(*(sasefelOutput->emitIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CxIndex = malloc(sizeof(*(sasefelOutput->CxIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CyIndex = malloc(sizeof(*(sasefelOutput->CyIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CxpIndex = malloc(sizeof(*(sasefelOutput->CxpIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CypIndex = malloc(sizeof(*(sasefelOutput->CypIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CtIndex = malloc(sizeof(*(sasefelOutput->CtIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->CdeltaIndex = malloc(sizeof(*(sasefelOutput->CdeltaIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->lightWavelengthIndex = malloc(sizeof(*(sasefelOutput->lightWavelengthIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->saturationLengthIndex = malloc(sizeof(*(sasefelOutput->saturationLengthIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->gainLengthIndex = malloc(sizeof(*(sasefelOutput->gainLengthIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->noisePowerIndex = malloc(sizeof(*(sasefelOutput->noisePowerIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->saturationPowerIndex = malloc(sizeof(*(sasefelOutput->saturationPowerIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->PierceParameterIndex = malloc(sizeof(*(sasefelOutput->PierceParameterIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->etaDiffractionIndex = malloc(sizeof(*(sasefelOutput->etaDiffractionIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->etaEmittanceIndex = malloc(sizeof(*(sasefelOutput->etaEmittanceIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->etaEnergySpreadIndex = malloc(sizeof(*(sasefelOutput->etaEnergySpreadIndex)) * (n_slices + 2))) ||
      !(sasefelOutput->sliceFound = malloc(sizeof(*(sasefelOutput->sliceFound)) * (n_slices + 2))))
    bombElegant("memory allocation failure (setupSASEFELAtEnd)", NULL);

  if (isMaster)
    if (output) {

      sasefelOutput->filename = compose_filename(output, run->rootname);
      SDDSout = &(sasefelOutput->SDDSout);
      if (!SDDS_InitializeOutputElegant(SDDSout, SDDS_BINARY, 0, NULL, NULL, sasefelOutput->filename) ||
          !SDDS_DefineSimpleParameter(SDDSout, "Step", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(SDDSout, "undulatorK", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(SDDSout, "undulatorPeriod", "m", SDDS_DOUBLE)) {
        printf("Unable define SDDS parameter for file %s\n", sasefelOutput->filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (n_slices <= 1) {
        if (!DefineSASEParameters(sasefelOutput, 0)) {
          printf("Unable define SDDS parameters for file %s\n", sasefelOutput->filename);
          fflush(stdout);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      } else {
        long slice;
        /* slice 0 is the nominal (no slicing)
         * slice N+1 is the average over the slices
         */
        for (slice = 0; slice <= n_slices + 1; slice++) {
          if (!DefineSASEParameters(sasefelOutput, slice)) {
            printf("Unable define SDDS parameters for file %s\n", sasefelOutput->filename);
            fflush(stdout);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      }
      if (0 > SDDS_DefineParameter(SDDSout, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_WriteLayout(SDDSout))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
}

long DefineSASEParameters(SASEFEL_OUTPUT *sasefelOutput, long slice) {
  SDDS_DATASET *SDDSout;
  char buffer[100], sliceNumString[20];

  SDDSout = &(sasefelOutput->SDDSout);
  if (slice && sasefelOutput->nSlices > 1) {
    if (slice <= sasefelOutput->nSlices) {
      sprintf(sliceNumString, sasefelOutput->nSlices < 100 ? "Slice%02ld" : "Slice%03ld", slice);
    } else {
      /* "slice" N+1 is the average over all slices */
      sprintf(sliceNumString, "Ave");
    }
  } else
    sliceNumString[0] = 0;

  sprintf(buffer, "betaBeam%s", sliceNumString);
  if ((sasefelOutput->betaToUseIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  sprintf(buffer, "Cx%s", sliceNumString);
  if ((sasefelOutput->CxIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Cy%s", sliceNumString);
  if ((sasefelOutput->CyIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Cxp%s", sliceNumString);
  if ((sasefelOutput->CxpIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Cyp%s", sliceNumString);
  if ((sasefelOutput->CypIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Ct%s", sliceNumString);
  if ((sasefelOutput->CtIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "s", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Cdelta%s", sliceNumString);
  if ((sasefelOutput->CdeltaIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  sprintf(buffer, "betaxBeam%s", sliceNumString);
  if ((sasefelOutput->betaxBeamIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "alphaxBeam%s", sliceNumString);
  if ((sasefelOutput->alphaxBeamIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "betayBeam%s", sliceNumString);
  if ((sasefelOutput->betayBeamIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "alphayBeam%s", sliceNumString);
  if ((sasefelOutput->alphayBeamIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  sprintf(buffer, "enx%s", sliceNumString);
  if ((sasefelOutput->enxIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "eny%s", sliceNumString);
  if ((sasefelOutput->enyIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  sprintf(buffer, "charge%s", sliceNumString);
  if ((sasefelOutput->chargeIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "C", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "rmsBunchLength%s", sliceNumString);
  if ((sasefelOutput->rmsBunchLengthIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "s", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "Sdelta%s", sliceNumString);
  if ((sasefelOutput->SdeltaIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "emit%s", sliceNumString);
  if ((sasefelOutput->emitIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "pCentral%s", sliceNumString);
  if ((sasefelOutput->pCentralIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m$be$nc", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "lightWavelength%s", sliceNumString);
  if ((sasefelOutput->lightWavelengthIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "gainLength%s", sliceNumString);
  if ((sasefelOutput->gainLengthIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "noisePower%s", sliceNumString);
  if ((sasefelOutput->noisePowerIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "W", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "saturationPower%s", sliceNumString);
  if ((sasefelOutput->saturationPowerIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "W", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "saturationLength%s", sliceNumString);
  if ((sasefelOutput->saturationLengthIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, "m", NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "PierceParameter%s", sliceNumString);
  if ((sasefelOutput->PierceParameterIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "etaDiffraction%s", sliceNumString);
  if ((sasefelOutput->etaDiffractionIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "etaEmittance%s", sliceNumString);
  if ((sasefelOutput->etaEmittanceIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  sprintf(buffer, "etaEnergySpread%s", sliceNumString);
  if ((sasefelOutput->etaEnergySpreadIndex[slice] =
       SDDS_DefineParameter(SDDSout, buffer, NULL, NULL, NULL, NULL, SDDS_DOUBLE, NULL)) < 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  return 1;
}

void doSASEFELAtEndOutput(SASEFEL_OUTPUT *sasefelOutput, long step) {
  SDDS_DATASET *SDDSout;
  long slice;

  if (!sasefelOutput || !sasefelOutput->active)
    SDDS_Bomb("doSASEFELAtEndOutput called without proper setup!");
  if (!sasefelOutput->filename)
    return;
  SDDSout = &(sasefelOutput->SDDSout);
#if SDDS_MPI_IO
  SDDSout->parallel_io = 0;
#endif
  if (!SDDS_StartPage(SDDSout, 0) ||
      !SDDS_SetParameters(SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Step", step,
                          "undulatorK", sasefelOutput->undulatorK,
                          "undulatorPeriod", sasefelOutput->undulatorPeriod,
                          NULL)) {
    printf("Unable write data to file %s\n", sasefelOutput->filename);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (slice = 0; slice <= sasefelOutput->nSlices + 1; slice++) {
    if (slice == 1 && sasefelOutput->nSlices == 0)
      break;
    if (!SDDS_SetParameters(SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                            sasefelOutput->betaToUseIndex[slice], sasefelOutput->betaToUse[slice],
                            sasefelOutput->betaxBeamIndex[slice], sasefelOutput->betaxBeam[slice],
                            sasefelOutput->alphaxBeamIndex[slice], sasefelOutput->alphaxBeam[slice],
                            sasefelOutput->betayBeamIndex[slice], sasefelOutput->betayBeam[slice],
                            sasefelOutput->alphayBeamIndex[slice], sasefelOutput->alphayBeam[slice],
                            sasefelOutput->enxIndex[slice], sasefelOutput->enx[slice],
                            sasefelOutput->enyIndex[slice], sasefelOutput->eny[slice],
                            sasefelOutput->chargeIndex[slice], sasefelOutput->charge[slice],
                            sasefelOutput->pCentralIndex[slice], sasefelOutput->pCentral[slice],
                            sasefelOutput->rmsBunchLengthIndex[slice], sasefelOutput->rmsBunchLength[slice],
                            sasefelOutput->SdeltaIndex[slice], sasefelOutput->Sdelta[slice],
                            sasefelOutput->emitIndex[slice], sasefelOutput->emit[slice],
                            sasefelOutput->CxIndex[slice], sasefelOutput->Cx[slice],
                            sasefelOutput->CyIndex[slice], sasefelOutput->Cy[slice],
                            sasefelOutput->CxpIndex[slice], sasefelOutput->Cxp[slice],
                            sasefelOutput->CypIndex[slice], sasefelOutput->Cyp[slice],
                            sasefelOutput->CtIndex[slice], sasefelOutput->Ct[slice],
                            sasefelOutput->CdeltaIndex[slice], sasefelOutput->Cdelta[slice],
                            sasefelOutput->lightWavelengthIndex[slice], sasefelOutput->lightWavelength[slice],
                            sasefelOutput->saturationLengthIndex[slice], sasefelOutput->saturationLength[slice],
                            sasefelOutput->gainLengthIndex[slice], sasefelOutput->gainLength[slice],
                            sasefelOutput->noisePowerIndex[slice], sasefelOutput->noisePower[slice],
                            sasefelOutput->saturationPowerIndex[slice], sasefelOutput->saturationPower[slice],
                            sasefelOutput->PierceParameterIndex[slice], sasefelOutput->PierceParameter[slice],
                            sasefelOutput->etaDiffractionIndex[slice], sasefelOutput->etaDiffraction[slice],
                            sasefelOutput->etaEmittanceIndex[slice], sasefelOutput->etaEmittance[slice],
                            sasefelOutput->etaEnergySpreadIndex[slice], sasefelOutput->etaEnergySpread[slice],
                            -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_WritePage(SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDSout);
}

void computeSASEFELAtEnd(SASEFEL_OUTPUT *sasefelOutput, double **particle, long particles,
                         double Po, double charge) {
  double emitx, emity;
  double *time, deltaAve, deltaRMS, tRMS;
  double S11, S12, S22, S33, S34, S44;
  long i, slice, nSlices;
  double xLimit[2], percentLevel[2];
  long count, slicesFound = 0, j;
  double aveCoord[7], rmsCoord[7];

#if USE_MPI
  long total_count;
  long total_particles;

  MPI_Allreduce(&particles, &total_particles, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
  if (!particles) {
    printf("no particles left---can't compute FEL parameters");
    fflush(stdout);
    /* fill in some dummy values */
    for (slice = 0; slice < sasefelOutput->nSlices + 1; slice++) {
      sasefelOutput->betaToUse[slice] = sasefelOutput->charge[slice] = DBL_MAX;
      sasefelOutput->betaxBeam[slice] = sasefelOutput->alphaxBeam[slice] = DBL_MAX;
      sasefelOutput->betayBeam[slice] = sasefelOutput->alphayBeam[slice] = DBL_MAX;
      sasefelOutput->enx[slice] = sasefelOutput->eny[slice] = DBL_MAX;
      sasefelOutput->Cx[slice] = sasefelOutput->Cy[slice] = DBL_MAX;
      sasefelOutput->Cxp[slice] = sasefelOutput->Cyp[slice] = DBL_MAX;
      sasefelOutput->Ct[slice] = sasefelOutput->Cdelta[slice] = DBL_MAX;
      sasefelOutput->pCentral[slice] = sasefelOutput->rmsBunchLength[slice] = DBL_MAX;
      sasefelOutput->Sdelta[slice] = sasefelOutput->emit[slice] = DBL_MAX;
      sasefelOutput->lightWavelength[slice] = sasefelOutput->saturationLength[slice] = DBL_MAX;
      sasefelOutput->gainLength[slice] = sasefelOutput->noisePower[slice] = DBL_MAX;
      sasefelOutput->saturationPower[slice] = sasefelOutput->PierceParameter[slice] = DBL_MAX;
      sasefelOutput->etaDiffraction[slice] = sasefelOutput->etaEmittance[slice] = DBL_MAX;
      sasefelOutput->etaEnergySpread[slice] = DBL_MAX;
    }
    return;
  }
#endif
  if (!(time = malloc(sizeof(*time) * particles)))
    SDDS_Bomb("memory allocation failure (computeSASEFELAtEnd)");
  computeTimeCoordinatesOnly(time, Po, particle, particles);

  /* compute normal values (over entire beam) */
  /* find center of energy distribution */
  for (j = 0; j < 7; j++)
    aveCoord[j] = 0;
  for (i = 0; i < particles; i++) {
    for (j = 0; j < 6; j++)
      aveCoord[j] += particle[i][j];
    aveCoord[j] += time[i];
  }

#if USE_MPI
  if (USE_MPI) {
    double aveCoord_tmp[7];

    memcpy(aveCoord_tmp, aveCoord, 7 * sizeof(aveCoord[0]));
    MPI_Allreduce(aveCoord_tmp, aveCoord, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (j = 0; j < 7; j++)
      aveCoord[j] /= total_particles;
  }
#else
  for (j = 0; j < 7; j++)
    aveCoord[j] /= particles;
#endif
  deltaAve = aveCoord[5];
  sasefelOutput->Cx[0] = aveCoord[0];
  sasefelOutput->Cxp[0] = aveCoord[1];
  sasefelOutput->Cy[0] = aveCoord[2];
  sasefelOutput->Cyp[0] = aveCoord[3];
  sasefelOutput->Cdelta[0] = aveCoord[5];
  sasefelOutput->Ct[0] = aveCoord[6];

  /* compute rms energy spread */
  for (i = deltaRMS = 0; i < particles; i++)
    deltaRMS += sqr(particle[i][5] - deltaAve);
#if !USE_MPI
  sasefelOutput->Sdelta[0] = deltaRMS = sqrt(deltaRMS / particles);
#else
  if (USE_MPI) {
    double deltaRMS_sum;

    MPI_Reduce(&deltaRMS, &deltaRMS_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (isMaster)
      sasefelOutput->Sdelta[0] = deltaRMS = sqrt(deltaRMS_sum / total_particles);
  }
#endif
  /* compute rms-equivalent time value so that Q/(sqrt(2*PI)*tRMS) is
   * a good estimate of peak current.  I use the 10% and 90% points of
   * the distribution to compute peak current, then get equivalent tRMS.
   */
  percentLevel[0] = 10;
  percentLevel[1] = 90;
#if USE_MPI
  approximate_percentiles_p(xLimit, percentLevel, 2, time, particles, ANALYSIS_BINS);
#else
  compute_percentiles(xLimit, percentLevel, 2, time, particles);
#endif
  sasefelOutput->rmsBunchLength[0] = tRMS = (xLimit[1] - xLimit[0]) / (0.8 * sqrt(2 * PI));

#if USE_MPI
  emitx = rms_emittance_p(particle, 0, 1, particles, &S11, &S12, NULL, NULL, NULL, NULL);
  emity = rms_emittance_p(particle, 2, 3, particles, &S33, &S34, NULL, NULL, NULL, NULL);
#else
  emitx = rms_emittance(particle, 0, 1, particles, &S11, &S12, NULL, NULL, NULL);
  emity = rms_emittance(particle, 2, 3, particles, &S33, &S34, NULL, NULL, NULL);
#endif

  SetSASEBetaEmitValues(sasefelOutput->emit + 0, sasefelOutput->betaToUse + 0,
                        sasefelOutput->beamsizeMode,
                        emitx, S11, S12,
                        emity, S33, S34, sasefelOutput->beta);

  sasefelOutput->pCentral[0] = Po * (1 + deltaAve);
  sasefelOutput->charge[0] = charge;
  sasefelOutput->enx[0] = emitx * sasefelOutput->pCentral[0];
  sasefelOutput->eny[0] = emity * sasefelOutput->pCentral[0];

  sasefelOutput->betaxBeam[0] = S11 / emitx;
  sasefelOutput->betayBeam[0] = S33 / emity;
  sasefelOutput->alphaxBeam[0] = -S12 / emitx;
  sasefelOutput->alphayBeam[0] = -S34 / emity;

  ComputeSASEFELParameters(&sasefelOutput->lightWavelength[0], &sasefelOutput->saturationLength[0],
                           &sasefelOutput->gainLength[0],
                           &sasefelOutput->noisePower[0], &sasefelOutput->saturationPower[0],
                           &sasefelOutput->PierceParameter[0],
                           &sasefelOutput->etaDiffraction[0], &sasefelOutput->etaEmittance[0],
                           &sasefelOutput->etaEnergySpread[0],
                           charge, tRMS, sasefelOutput->undulatorPeriod,
                           sasefelOutput->undulatorK,
                           sasefelOutput->betaToUse[0], sasefelOutput->emit[0],
                           sasefelOutput->Sdelta[0], sasefelOutput->pCentral[0],
                           1);
  if (sasefelOutput->nSlices > 1) {
    /* compute values for each slice, plus average */
    nSlices = sasefelOutput->nSlices;

    sasefelOutput->betaToUse[nSlices + 1] = sasefelOutput->charge[nSlices + 1] = 0;
    sasefelOutput->betaxBeam[nSlices + 1] = sasefelOutput->betayBeam[nSlices + 1] = 0;
    sasefelOutput->alphaxBeam[nSlices + 1] = sasefelOutput->alphayBeam[nSlices + 1] = 0;
    sasefelOutput->enx[nSlices + 1] = sasefelOutput->eny[nSlices + 1] = 0;
    sasefelOutput->pCentral[nSlices + 1] = sasefelOutput->rmsBunchLength[nSlices + 1] = 0;
    sasefelOutput->Sdelta[nSlices + 1] = sasefelOutput->emit[nSlices + 1] = 0;
    sasefelOutput->lightWavelength[nSlices + 1] = sasefelOutput->saturationLength[nSlices + 1] = 0;
    sasefelOutput->gainLength[nSlices + 1] = sasefelOutput->noisePower[nSlices + 1] = 0;
    sasefelOutput->saturationPower[nSlices + 1] = sasefelOutput->PierceParameter[nSlices + 1] = 0;
    sasefelOutput->etaDiffraction[nSlices + 1] = sasefelOutput->etaEmittance[nSlices + 1] = 0;
    sasefelOutput->etaEnergySpread[nSlices + 1] = 0;
    sasefelOutput->Cx[nSlices + 1] = sasefelOutput->Cxp[nSlices + 1] = 0;
    sasefelOutput->Cy[nSlices + 1] = sasefelOutput->Cyp[nSlices + 1] = 0;
    sasefelOutput->Ct[nSlices + 1] = sasefelOutput->Cdelta[nSlices + 1] = 0;

    for (slice = 1; slice <= sasefelOutput->nSlices; slice++) {
      /* find boundaries of slice in time */
      percentLevel[0] = 100 * (0.5 - sasefelOutput->nSlices * sasefelOutput->sliceFraction / 2.0 +
                               (slice - 1) * sasefelOutput->sliceFraction);
      if (percentLevel[0] < 0)
        percentLevel[0] = 0;
      percentLevel[1] = percentLevel[0] + 100 * sasefelOutput->sliceFraction;
      if (percentLevel[1] > 100)
        percentLevel[1] = 100;

      /* compute rms-equivalent time value so that Q/(sqrt(2*PI)*tRMS) is
       * the average current in the slice
       */
#if USE_MPI
      approximate_percentiles_p(xLimit, percentLevel, 2, time, particles, ANALYSIS_BINS);
#else
      compute_percentiles(xLimit, percentLevel, 2, time, particles);
#endif
      sasefelOutput->rmsBunchLength[slice] = (xLimit[1] - xLimit[0]) / sqrt(2 * PI);

      /* find centroids of slice */
      for (j = 0; j < 7; j++)
        aveCoord[j] = 0;
      for (i = count = 0; i < particles; i++) {
        if (time[i] > xLimit[0] && time[i] < xLimit[1]) {
          count++;
          for (j = 0; j < 6; j++)
            aveCoord[j] += particle[i][j];
          aveCoord[j] += time[i];
        }
      }
#if USE_MPI
      MPI_Allreduce(&count, &total_count, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      count = total_count;
#endif
      if (count < 2) {
        /* fill in some dummy values */
        sasefelOutput->sliceFound[slice] = 0;
        sasefelOutput->betaxBeam[nSlices + 1] = sasefelOutput->betayBeam[nSlices + 1] = DBL_MAX;
        sasefelOutput->alphaxBeam[nSlices + 1] = sasefelOutput->alphayBeam[nSlices + 1] = DBL_MAX;
        sasefelOutput->enx[nSlices + 1] = sasefelOutput->eny[nSlices + 1] = DBL_MAX;
        sasefelOutput->betaToUse[slice] = sasefelOutput->charge[slice] = DBL_MAX;
        sasefelOutput->pCentral[slice] = sasefelOutput->rmsBunchLength[slice] = DBL_MAX;
        sasefelOutput->Sdelta[slice] = sasefelOutput->emit[slice] = DBL_MAX;
        sasefelOutput->lightWavelength[slice] = sasefelOutput->saturationLength[slice] = DBL_MAX;
        sasefelOutput->gainLength[slice] = sasefelOutput->noisePower[slice] = DBL_MAX;
        sasefelOutput->saturationPower[slice] = sasefelOutput->PierceParameter[slice] = DBL_MAX;
        sasefelOutput->etaDiffraction[slice] = sasefelOutput->etaEmittance[slice] = DBL_MAX;
        sasefelOutput->etaEnergySpread[slice] = DBL_MAX;
        sasefelOutput->Cx[nSlices + 1] = sasefelOutput->Cxp[nSlices + 1] = DBL_MAX;
        sasefelOutput->Cy[nSlices + 1] = sasefelOutput->Cyp[nSlices + 1] = DBL_MAX;
        sasefelOutput->Ct[nSlices + 1] = sasefelOutput->Cdelta[nSlices + 1] = DBL_MAX;
      }
      slicesFound++;
      sasefelOutput->sliceFound[slice] = 1;
#if USE_MPI
      if (USE_MPI) {
        double aveCoord_tmp[7];

        memcpy(aveCoord_tmp, aveCoord, 7 * sizeof(aveCoord[0]));
        MPI_Allreduce(aveCoord_tmp, aveCoord, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      }
#endif
      for (j = 0; j < 7; j++) {
        aveCoord[j] /= count;
        rmsCoord[j] = 0;
      }
      S12 = S34 = 0;
      sasefelOutput->Cx[slice] = aveCoord[0];
      sasefelOutput->Cxp[slice] = aveCoord[1];
      sasefelOutput->Cy[slice] = aveCoord[2];
      sasefelOutput->Cyp[slice] = aveCoord[3];
      sasefelOutput->Cdelta[slice] = aveCoord[5];
      sasefelOutput->Ct[slice] = aveCoord[6];

      /* compute rms energy spread and transverse moments */
      for (i = deltaRMS = 0; i < particles; i++)
        if (time[i] > xLimit[0] && time[i] < xLimit[1]) {
          for (j = 0; j < 6; j++)
            rmsCoord[j] += sqr(particle[i][j] - aveCoord[j]);
          S12 += (particle[i][0] - aveCoord[0]) * (particle[i][1] - aveCoord[1]);
          S34 += (particle[i][2] - aveCoord[2]) * (particle[i][3] - aveCoord[3]);
        }
#if USE_MPI
      if (USE_MPI) {
        double rmsCoord_tmp[6],
          S12_tmp = S12, S34_tmp = S34;

        memcpy(rmsCoord_tmp, rmsCoord, 6 * sizeof(rmsCoord[0]));
        MPI_Reduce(rmsCoord_tmp, rmsCoord, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&S12_tmp, &S12, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&S34_tmp, &S34, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      }
#endif
      for (j = 0; j < 6; j++)
        rmsCoord[j] = sqrt(rmsCoord[j] / count);
      S12 /= count;
      S34 /= count;
      S11 = sqr(rmsCoord[0]);
      S22 = sqr(rmsCoord[1]);
      S33 = sqr(rmsCoord[2]);
      S44 = sqr(rmsCoord[3]);

      sasefelOutput->Sdelta[slice] = rmsCoord[5];
      emitx = sqrt(S11 * S22 - sqr(S12));
      emity = sqrt(S33 * S44 - sqr(S34));

      SetSASEBetaEmitValues(sasefelOutput->emit + slice, sasefelOutput->betaToUse + slice,
                            sasefelOutput->beamsizeMode,
                            emitx, S11, S12,
                            emity, S33, S34, sasefelOutput->beta);

      sasefelOutput->pCentral[slice] = Po * (1 + aveCoord[5]);
      sasefelOutput->charge[slice] = charge * sasefelOutput->sliceFraction;
      sasefelOutput->enx[slice] = emitx * sasefelOutput->pCentral[0];
      sasefelOutput->eny[slice] = emity * sasefelOutput->pCentral[0];

      sasefelOutput->betaxBeam[slice] = S11 / emitx;
      sasefelOutput->betayBeam[slice] = S33 / emity;
      sasefelOutput->alphaxBeam[slice] = -S12 / emitx;
      sasefelOutput->alphayBeam[slice] = -S34 / emity;

      ComputeSASEFELParameters(&sasefelOutput->lightWavelength[slice],
                               &sasefelOutput->saturationLength[slice],
                               &sasefelOutput->gainLength[slice],
                               &sasefelOutput->noisePower[slice],
                               &sasefelOutput->saturationPower[slice],
                               &sasefelOutput->PierceParameter[slice],
                               &sasefelOutput->etaDiffraction[slice],
                               &sasefelOutput->etaEmittance[slice],
                               &sasefelOutput->etaEnergySpread[slice],
                               sasefelOutput->charge[slice],
                               sasefelOutput->rmsBunchLength[slice],
                               sasefelOutput->undulatorPeriod,
                               sasefelOutput->undulatorK,
                               sasefelOutput->betaToUse[slice], sasefelOutput->emit[slice],
                               sasefelOutput->Sdelta[slice], sasefelOutput->pCentral[slice],
                               1);

      sasefelOutput->lightWavelength[nSlices + 1] += sasefelOutput->lightWavelength[slice];
      sasefelOutput->saturationLength[nSlices + 1] += sasefelOutput->saturationLength[slice];
      sasefelOutput->gainLength[nSlices + 1] += sasefelOutput->gainLength[slice];
      sasefelOutput->noisePower[nSlices + 1] += sasefelOutput->noisePower[slice];
      sasefelOutput->saturationPower[nSlices + 1] += sasefelOutput->saturationPower[slice];
      sasefelOutput->PierceParameter[nSlices + 1] += sasefelOutput->PierceParameter[slice];
      sasefelOutput->etaDiffraction[nSlices + 1] += sasefelOutput->etaDiffraction[slice];
      sasefelOutput->etaEmittance[nSlices + 1] += sasefelOutput->etaEmittance[slice];
      sasefelOutput->etaEnergySpread[nSlices + 1] += sasefelOutput->etaEnergySpread[slice];
      sasefelOutput->betaToUse[nSlices + 1] += sasefelOutput->betaToUse[slice];
      sasefelOutput->betaxBeam[nSlices + 1] += sasefelOutput->betaxBeam[slice];
      sasefelOutput->betayBeam[nSlices + 1] += sasefelOutput->betayBeam[slice];
      sasefelOutput->alphaxBeam[nSlices + 1] += sasefelOutput->alphaxBeam[slice];
      sasefelOutput->alphayBeam[nSlices + 1] += sasefelOutput->alphayBeam[slice];
      sasefelOutput->enx[nSlices + 1] += sasefelOutput->enx[slice];
      sasefelOutput->eny[nSlices + 1] += sasefelOutput->eny[slice];
      sasefelOutput->emit[nSlices + 1] += sasefelOutput->emit[slice];
      sasefelOutput->Sdelta[nSlices + 1] += sasefelOutput->Sdelta[slice];
      sasefelOutput->pCentral[nSlices + 1] += sasefelOutput->pCentral[slice];
      sasefelOutput->charge[nSlices + 1] += sasefelOutput->charge[slice];
      sasefelOutput->rmsBunchLength[nSlices + 1] += sasefelOutput->rmsBunchLength[slice];
      sasefelOutput->Cx[nSlices + 1] += sasefelOutput->Cx[slice];
      sasefelOutput->Cxp[nSlices + 1] += sasefelOutput->Cxp[slice];
      sasefelOutput->Cy[nSlices + 1] += sasefelOutput->Cy[slice];
      sasefelOutput->Cyp[nSlices + 1] += sasefelOutput->Cyp[slice];
      sasefelOutput->Cdelta[nSlices + 1] += sasefelOutput->Cdelta[slice];
      sasefelOutput->Ct[nSlices + 1] += sasefelOutput->Ct[slice];
    }

    if (!slicesFound)
      bombElegant("No valid slices found for SASE FEL computation.", NULL);

    sasefelOutput->lightWavelength[nSlices + 1] /= slicesFound;
    sasefelOutput->saturationLength[nSlices + 1] /= slicesFound;
    sasefelOutput->saturationPower[nSlices + 1] /= slicesFound;
    sasefelOutput->gainLength[nSlices + 1] /= slicesFound;
    sasefelOutput->noisePower[nSlices + 1] /= slicesFound;
    sasefelOutput->PierceParameter[nSlices + 1] /= slicesFound;
    sasefelOutput->etaDiffraction[nSlices + 1] /= slicesFound;
    sasefelOutput->etaEmittance[nSlices + 1] /= slicesFound;
    sasefelOutput->etaEnergySpread[nSlices + 1] /= slicesFound;
    sasefelOutput->betaToUse[nSlices + 1] /= slicesFound;
    sasefelOutput->betaxBeam[nSlices + 1] /= slicesFound;
    sasefelOutput->betayBeam[nSlices + 1] /= slicesFound;
    sasefelOutput->alphaxBeam[nSlices + 1] /= slicesFound;
    sasefelOutput->alphayBeam[nSlices + 1] /= slicesFound;
    sasefelOutput->enx[nSlices + 1] /= slicesFound;
    sasefelOutput->eny[nSlices + 1] /= slicesFound;
    sasefelOutput->emit[nSlices + 1] /= slicesFound;
    sasefelOutput->Sdelta[nSlices + 1] /= slicesFound;
    sasefelOutput->pCentral[nSlices + 1] /= slicesFound;
    sasefelOutput->charge[nSlices + 1] /= slicesFound;
    sasefelOutput->rmsBunchLength[nSlices + 1] /= slicesFound;
    sasefelOutput->Cx[nSlices + 1] /= slicesFound;
    sasefelOutput->Cxp[nSlices + 1] /= slicesFound;
    sasefelOutput->Cy[nSlices + 1] /= slicesFound;
    sasefelOutput->Cyp[nSlices + 1] /= slicesFound;
    sasefelOutput->Ct[nSlices + 1] /= slicesFound;
    sasefelOutput->Cdelta[nSlices + 1] /= slicesFound;
  }

  free(time);
}

void storeSASEFELAtEndInRPN(SASEFEL_OUTPUT *sasefelOutput) {
  rpn_store(sasefelOutput->lightWavelength[0], NULL,
            rpn_create_mem("SASE.lightWavelength", 0));
  rpn_store(sasefelOutput->gainLength[0], NULL,
            rpn_create_mem("SASE.gainLength", 0));
  rpn_store(sasefelOutput->saturationPower[0], NULL,
            rpn_create_mem("SASE.saturationPower", 0));
  rpn_store(sasefelOutput->saturationLength[0], NULL,
            rpn_create_mem("SASE.saturationLength", 0));
}

void SetSASEBetaEmitValues(double *emitToUse, double *betaToUse,
                           long beamsizeMode,
                           double emitx, double S11, double S12,
                           double emity, double S33, double S34, double userBeta) {
  if (userBeta) {
    switch (beamsizeMode) {
    case BEAMSIZE_GEOMETRIC_MEAN:
      *emitToUse = sqrt(emitx * emity);
      break;
    case BEAMSIZE_ARITHMETIC_MEAN:
      *emitToUse = (emitx + emity) / 2;
      break;
    case BEAMSIZE_HORIZONTAL:
      *emitToUse = emitx;
      break;
    case BEAMSIZE_VERTICAL:
      *emitToUse = emity;
      break;
    case BEAMSIZE_MAXIMUM:
      *emitToUse = emitx > emity ? emitx : emity;
      break;
    default:
      bombElegant("invalid beamsize mode (SetSASEBetaEmitValues)", NULL);
      break;
    }
    *betaToUse = userBeta;
  } else {
    double Sx, Sy, S = 0.0;
    Sx = sqrt(S11);
    Sy = sqrt(S33);
    switch (beamsizeMode) {
    case BEAMSIZE_GEOMETRIC_MEAN:
      S = sqrt(Sx * Sy);
      break;
    case BEAMSIZE_ARITHMETIC_MEAN:
      S = (Sx + Sy) / 2;
      break;
    case BEAMSIZE_HORIZONTAL:
      S = Sx;
      break;
    case BEAMSIZE_VERTICAL:
      S = Sy;
      break;
    case BEAMSIZE_MAXIMUM:
      S = Sx > Sy ? Sx : Sy;
      break;
    default:
      bombElegant("invalid beamsize mode (SetSASEBetaEmitValues)", NULL);
      break;
    }
    /* sase fel code just computes beamsize from sqrt(emit*beta),
     * so to some extent it is arbitrary what we choose */
    *emitToUse = sqrt(emitx * emity);
    *betaToUse = S * S / (*emitToUse);
  }
}
