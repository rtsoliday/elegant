/*************************************************************************\
* Copyright (c) 2010 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2010 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "modulate.h"
#ifdef HAVE_GPU
#  include <gpu_base.h>
#  include <gpu_simple_rfca.h>
#endif

long loadModulationTable(double **t, double **value, char *file, char *timeColumn, char *amplitudeColumn);

void addModulationElements(MODULATION_DATA *modData, NAMELIST_TEXT *nltext, LINE_LIST *beamline, RUN *run) {
  long n_items, n_added, firstIndexInGroup;
  ELEMENT_LIST *context;
  double sMin = -DBL_MAX, sMax = DBL_MAX;
  double *tData, *AData;
  long nData = 0;

  modData->beamline = beamline;

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&modulate_elements, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (name == NULL) {
    if (!type)
      bombElegant("element name missing in modulate_elements namelist", NULL);
    SDDS_CopyString(&name, "*");
  }
  if (!expression && !(filename && time_column && amplitude_column))
    bombElegant("either expression or filename, time_column, and amplitude_column must all be given", NULL);
  if (expression && filename)
    bombElegant("only one of expression and filename may be given", NULL);
  if (item == NULL)
    bombElegant("item name missing in modulate_elements namelist", NULL);
  if (echoNamelists)
    print_namelist(stdout, &modulate_elements);

  if (filename) {
    /* Read data file */
    if ((nData = loadModulationTable(&tData, &AData, filename, time_column, amplitude_column)) <= 2)
      bombElegant("too few items in modulation table", NULL);
  } else
    nData = 0;

  n_added = 0;
  n_items = modData->nItems;
  context = NULL;
  if (after && strlen(after)) {
    if (!(context = find_element(after, &context, beamline->elem))) {
      printf("Element %s not found in beamline.\n", after);
      exitElegant(1);
    }
    sMin = context->end_pos;
    if (find_element(after, &context, beamline->elem)) {
      printf("Element %s found in beamline more than once.\n", after);
      exitElegant(1);
    }
    printf("%s found at s = %le m\n", after, sMin);
    fflush(stdout);
  }
  context = NULL;
  if (before && strlen(before)) {
    if (!(context = find_element(before, &context, beamline->elem))) {
      printf("Element %s not found in beamline.\n", before);
      exitElegant(1);
    }
    sMax = context->end_pos;
    if (find_element(before, &context, beamline->elem)) {
      printf("Element %s found in beamline more than once.\n", after);
      exitElegant(1);
    }
    printf("%s found at s = %le m\n", before, sMax);
    fflush(stdout);
  }
  if (after && before && sMin > sMax) {
    printf("Element %s is not upstream of %s!\n",
           before, after);
    exitElegant(1);
  }
  if (type && has_wildcards(type) && strchr(type, '-'))
    type = expand_ranges(type);
  if (has_wildcards(name)) {
    if (strchr(name, '-'))
      name = expand_ranges(name);
    str_toupper(name);
  }
  if (start_pass > end_pass) {
    printf("start_pass > end_pass!\n");
    exitElegant(1);
  }
  firstIndexInGroup = -1;
  while ((context = wfind_element(name, &context, beamline->elem))) {
    if (type && !wild_match(entity_name[context->type], type))
      continue;
    if ((sMin >= 0 && context->end_pos < sMin) ||
        (sMax >= 0 && context->end_pos > sMax) ||
        (s_start >= 0 && context->end_pos < s_start) ||
        (s_end >= 0 && context->end_pos > s_end) ||
        (start_occurence && context->occurence < start_occurence) ||
        (end_occurence && context->occurence > end_occurence))
      continue;
    printf("Adding modulation for %s#%ld at s=%le\n", context->name, context->occurence, context->end_pos);

    modData->element = SDDS_Realloc(modData->element, sizeof(*modData->element) * (n_items + 1));
    modData->expression = SDDS_Realloc(modData->expression, sizeof(*modData->expression) * (n_items + 1));
    modData->parameterNumber = SDDS_Realloc(modData->parameterNumber, sizeof(*modData->parameterNumber) * (n_items + 1));
    modData->flags = SDDS_Realloc(modData->flags, sizeof(*modData->flags) * (n_items + 1));
    modData->factor = SDDS_Realloc(modData->factor, sizeof(*modData->factor)*(n_items+1));
    modData->verboseThreshold = SDDS_Realloc(modData->verboseThreshold, sizeof(*modData->verboseThreshold) * (n_items + 1));
    modData->lastVerboseValue = SDDS_Realloc(modData->lastVerboseValue, sizeof(*modData->lastVerboseValue) * (n_items + 1));
    modData->unperturbedValue = SDDS_Realloc(modData->unperturbedValue, sizeof(*modData->unperturbedValue) * (n_items + 1));
    modData->nData = SDDS_Realloc(modData->nData, sizeof(*modData->nData) * (n_items + 1));
    modData->dataIndex = SDDS_Realloc(modData->dataIndex, sizeof(*modData->dataIndex) * (n_items + 1));
    modData->timeData = SDDS_Realloc(modData->timeData, sizeof(*modData->timeData) * (n_items + 1));
    modData->modulationData = SDDS_Realloc(modData->modulationData, sizeof(*modData->modulationData) * (n_items + 1));
    modData->record = SDDS_Realloc(modData->record, sizeof(*modData->record) * (n_items + 1));
    modData->flushRecord = SDDS_Realloc(modData->flushRecord, sizeof(*modData->flushRecord) * (n_items + 1));
    modData->fpRecord = SDDS_Realloc(modData->fpRecord, sizeof(*modData->fpRecord) * (n_items + 1));
    modData->convertPassToTime = SDDS_Realloc(modData->convertPassToTime, sizeof(*modData->convertPassToTime) * (n_items + 1));
    modData->startPass = SDDS_Realloc(modData->startPass, sizeof(*modData->startPass) * (n_items + 1));
    modData->endPass = SDDS_Realloc(modData->endPass, sizeof(*modData->endPass) * (n_items + 1));
    modData->passDelay = SDDS_Realloc(modData->passDelay, sizeof(*modData->passDelay) * (n_items + 1));
    modData->timeDelay = SDDS_Realloc(modData->timeDelay, sizeof(*modData->timeDelay) * (n_items + 1));

    modData->element[n_items] = context;
    modData->flags[n_items] = (multiplicative ? MULTIPLICATIVE_MOD : 0) + (differential ? DIFFERENTIAL_MOD : 0) + (verbose ? VERBOSE_MOD : 0) + (refresh_matrix ? REFRESH_MATRIX_MOD : 0);
    modData->verboseThreshold[n_items] = verbose_threshold;
    modData->timeData[n_items] = modData->modulationData[n_items] = NULL;
    modData->expression[n_items] = NULL;
    modData->fpRecord[n_items] = NULL;
    modData->nData[n_items] = 0;
    modData->flushRecord[n_items] = flush_record;
    modData->convertPassToTime[n_items] = convert_pass_to_time;
    modData->startPass[n_items] = start_pass;
    modData->endPass[n_items] = end_pass;
    modData->passDelay[n_items] = pass_delay;
    modData->timeDelay[n_items] = time_delay;
    modData->factor[n_items] = factor;
    if (pass_delay!=0 && !convert_pass_to_time) 
      bombElegant("If pass_delay is non-zero, must set convert_pass_to_time=1.", NULL);

    if (filename) {
      if ((modData->dataIndex[n_items] = firstIndexInGroup) == -1) {
        modData->timeData[n_items] = tData;
        modData->modulationData[n_items] = AData;
        modData->nData[n_items] = nData;
      }
    } else
      cp_str(&modData->expression[n_items], expression);

    if ((modData->parameterNumber[n_items] = confirm_parameter(item, context->type)) < 0) {
      printf("error: cannot modulate %s---no such parameter for %s (wildcard name: %s)\n", item, context->name, name);
      fflush(stdout);
      exitElegant(1);
    }

    if (record
#if USE_MPI
        && myid == 0
#endif
    ) {
      modData->record[n_items] = compose_filename(record, run->rootname);
      record = NULL;
      if (!(modData->fpRecord[n_items] = fopen(modData->record[n_items], "w")))
        SDDS_Bomb("problem setting up  modulation record file");
      fprintf(modData->fpRecord[n_items], "SDDS1\n&column name=t, units=s, type=double &end\n");
      fprintf(modData->fpRecord[n_items], "&column name=Pass, type=long &end\n");
      fprintf(modData->fpRecord[n_items], "&column name=Amplitude, type=double &end\n");
      fprintf(modData->fpRecord[n_items], "&column name=OriginalValue, type=double &end\n");
      fprintf(modData->fpRecord[n_items], "&column name=NewValue, type=double &end\n");
      fprintf(modData->fpRecord[n_items], "&data mode=ascii, no_row_counts=1 &end\n");
    }

    modData->nItems = ++n_items;
    n_added++;
    if (firstIndexInGroup == -1)
      firstIndexInGroup = n_items - 1;
  }

  if (!n_added) {
    printf("error: no match given modulation\n");
    fflush(stdout);
    exitElegant(1);
  }
}

long loadModulationTable(double **t, double **value, char *file, char *timeColumn, char *amplitudeColumn) {
  SDDS_TABLE SDDS_table;
  long i, count = 0;

  if (!t || !value || !file || !timeColumn || !amplitudeColumn)
    bombElegant("NULL value pointer passed (loadModulationTable)", NULL);

  if (!SDDS_InitializeInputFromSearchPath(&SDDS_table, file) ||
      SDDS_ReadTable(&SDDS_table) != 1 ||
      !(count = SDDS_CountRowsOfInterest(&SDDS_table)) ||
      !(*value = SDDS_GetColumnInDoubles(&SDDS_table, amplitudeColumn)) ||
      !(*t = SDDS_GetColumnInDoubles(&SDDS_table, timeColumn)) ||
      !SDDS_Terminate(&SDDS_table)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    bombElegant("Unable to read required data from modulation file.", NULL);
  }
  for (i = 1; i < count; i++)
    if ((*t)[i - 1] >= (*t)[i])
      bombElegant("time values not monotonically increasing for modulation data", NULL);
  return (count);
}

long applyElementModulations(MODULATION_DATA *modData, LINE_LIST *beamline, double pCentral,
                             double **coord, long np, RUN *run, long iPass) {
  long iMod, code, matricesUpdated, jMod;
  /* short modulationValid = 0; */
  double modulation, value, t, tBeam, lastValue, beta;
  long type, param;
  char *p_elem;

  if (modData->nItems <= 0)
    return 0;

#ifdef DEBUG
  printf("applyElementModulations\n");
  fflush(stdout);
#endif

#if USE_MPI
  long np_total = 0;
  if (notSinglePart) {
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (np_total == 0)
      return 0;
  } else if (np == 0)
    return 0;
#  ifdef DEBUG
  printf("applyElementModulations: np_total = %ld\n", np_total);
  fflush(stdout);
#  endif
#else
  if (np == 0)
    return 0;
#endif

  beta = pCentral / sqrt(pCentral * pCentral + 1);

#ifdef HAVE_GPU
  if (getGpuBase()->elementOnGpu) {
#  if USE_MPI
    if (notSinglePart) {
      coord = forceParticlesToCpu("findFiducialTime MPI fallback");
      tBeam = findFiducialTime(coord, np, 0, 0, pCentral, FID_MODE_TMEAN | FID_MODE_FULLBEAM);
    } else
#  endif
    tBeam = gpu_findFiducialTime(np, 0.0, 0.0, pCentral, FID_MODE_TMEAN | FID_MODE_FULLBEAM);
  } else
#endif
    tBeam = findFiducialTime(coord, np, 0, 0, pCentral, FID_MODE_TMEAN | FID_MODE_FULLBEAM);
#ifdef DEBUG
  printf("applyElementModulations: tBeam = %le\n", tBeam);
  fflush(stdout);
#endif

  matricesUpdated = 0;

  if (modData->valuesInitialized==0) {
    for (iMod = 0; iMod < modData->nItems; iMod++) {
#ifdef DEBUG
      printf("applyElementModulations: iMod = %ld\n", iMod);
      fflush(stdout);
#endif

      modData->lastVerboseValue[iMod] = modData->unperturbedValue[iMod] = parameter_value(modData->element[iMod]->name, modData->element[iMod]->type,
                                                                                          modData->parameterNumber[iMod], beamline);
      printf("Unperturbed value of modulated quantity %s.%s is %21.15le\n",
             modData->element[iMod]->name,
             entity_description[modData->element[iMod]->type].parameter[modData->parameterNumber[iMod]].name,
             modData->unperturbedValue[iMod]);
      if (modData->unperturbedValue[iMod] == 0 && modData->flags[iMod] & MULTIPLICATIVE_MOD) {
        char buffer[1024];
        snprintf(buffer, 1024, "Quantity is %s.%s. This may be an error.",
                 modData->element[iMod]->name,
                 entity_description[modData->element[iMod]->type].parameter[modData->parameterNumber[iMod]].name);
        printWarning("Multiplicative modulation specified but unperturbed value is zero.", buffer);
      }
    }
    modData->valuesInitialized = 1;
  }

  for (iMod = 0; iMod < modData->nItems; iMod++) {
    if (iPass < modData->startPass[iMod] || iPass > modData->endPass[iMod])
      continue;
#ifdef DEBUG
    printf("applyElementModulations: iMod = %ld active\n", iMod);
    fflush(stdout);
#endif
    if (modData->convertPassToTime[iMod]) {
      double s0;
      s0 = modData->beamline->elem_recirc ? modData->beamline->elem_recirc->end_pos : 0;
      t = ((iPass - modData->passDelay[iMod]) * modData->beamline->revolution_length + (modData->element[iMod]->end_pos - s0)) / (beta * c_mks);
    } else
      t = tBeam;
    t -= modData->timeDelay[iMod];
#ifdef DEBUG
    printf("applyElementModulations: t = %le\n", t);
    fflush(stdout);
#endif

    type = modData->element[iMod]->type;
#ifdef DEBUG
    printf("applyElementModulations: type = %ld\n", type);
    fflush(stdout);
#endif
    param = modData->parameterNumber[iMod];
#ifdef DEBUG
    printf("applyElementModulations: param = %ld\n", param);
    fflush(stdout);
#endif
    p_elem = (char *)(modData->element[iMod]->p_elem);
#ifdef DEBUG
    print_elem(stdout, modData->element[iMod]);
    printf("applyElementModulations: p_elem = %x\n", p_elem);
    fflush(stdout);
#endif
    value = modData->unperturbedValue[iMod];
#ifdef DEBUG
    printf("applyElementModulations: value = %le\n", value);
    fflush(stdout);
#endif

    modulation = 0;

#ifdef DEBUG
    printf("applyElementModulations: expression = %s\n", modData->expression[iMod]);
    fflush(stdout);
#endif
    if (!modData->expression[iMod]) {
      jMod = iMod; /* jMod is the index of the modulation that stores the tabular data, which may be shared */
      if (modData->dataIndex[iMod] != -1)
        jMod = modData->dataIndex[iMod];
      code = 1;
      if (t <= modData->timeData[jMod][0]) {
        char buffer[16384];
        modulation = modData->modulationData[jMod][0];
        snprintf(buffer, 16384, "t=%21.15le is below modulation table range [%21.15le, %21.15le] for element %s, parameter %s. Value is taken from first table entry.",
                 t, modData->timeData[jMod][0], modData->timeData[jMod][modData->nData[jMod] - 1], modData->element[jMod]->name,
                 entity_description[type].parameter[param].name);
        printWarning("Interpolation below modulation table range.", buffer);
      } else if (t >= modData->timeData[jMod][modData->nData[jMod] - 1]) {
        char buffer[16384];
        modulation = modData->modulationData[jMod][modData->nData[jMod] - 1];
        snprintf(buffer, 16384, "t=%21.15le is above modulation table range [%21.15le, %21.15le] for element %s, parameter %s. Value is taken from last table entry.",
                 t, modData->timeData[jMod][0], modData->timeData[jMod][modData->nData[jMod] - 1], modData->element[jMod]->name,
                 entity_description[type].parameter[param].name);
        printWarning("Interpolation above modulation table range.", buffer);
      } else
        modulation = interp(modData->modulationData[jMod], modData->timeData[jMod], modData->nData[jMod], t, 0, 1, &code);
      if (code == 0) {
        fprintf(stderr, "Error: interpolation failed for t=%21.15le for element %s, parameter %s\n",
                         t, modData->element[jMod]->name, entity_description[type].parameter[param].name);
        exitElegant(1);
      }
      /* modulationValid = 1; */
    } else {
#ifdef DEBUG
      printf("applyElementModulations: evaluating expression (1)\n");
      fflush(stdout);
#endif
      push_num(t);
#ifdef DEBUG
      printf("applyElementModulations: evaluating expression (2)\n");
      fflush(stdout);
#endif
      modulation = rpn(modData->expression[iMod]);
#ifdef DEBUG
      printf("applyElementModulations: modulation = %le, evaluating expression (3)\n", modulation);
      fflush(stdout);
#endif
      rpn_clear();
#ifdef DEBUG
      printf("applyElementModulations: modulation from expression = %le\n", modulation);
      fflush(stdout);
#endif
    }

    modulation *= modData->factor[iMod];
    
    if (modData->flags[iMod] & DIFFERENTIAL_MOD) {
      if (modData->flags[iMod] & MULTIPLICATIVE_MOD)
        value = (1 + modulation) * value;
      else
        value = value + modulation;
    } else {
      if (modData->flags[iMod] & MULTIPLICATIVE_MOD)
        value = value * modulation;
      else
        value = modulation;
    }

#ifdef DEBUG
    printf("applyElementModulations: value = %le\n", value);
    fflush(stdout);
#endif

    switch (entity_description[type].parameter[param].type) {
    case IS_DOUBLE:
      lastValue = modData->lastVerboseValue[iMod];
      *((double *)(p_elem + entity_description[type].parameter[param].offset)) = value;
      if (modData->flags[iMod] & VERBOSE_MOD && fabs(value - lastValue) > modData->verboseThreshold[iMod] * (fabs(value) + fabs(lastValue)) / 2) {
        printf("Modulation value for element %s#%ld, parameter %s changed to %21.15le at t = %21.15le (originally %21.15le)\n",
               modData->element[iMod]->name, modData->element[iMod]->occurence,
               entity_description[type].parameter[param].name, value, t, modData->unperturbedValue[iMod]);
        modData->lastVerboseValue[iMod] = value;
      }
      break;
    case IS_LONG:
      lastValue = modData->lastVerboseValue[iMod];
      value = (long)(value + 0.5);
      *((long *)(p_elem + entity_description[type].parameter[param].offset)) = value;
      if (modData->flags[iMod] & VERBOSE_MOD && value != lastValue) {
        printf("Modulation value for element %s#%ld, parameter %s changed to %ld at t = %21.15le (originally %ld)\n",
               modData->element[iMod]->name, modData->element[iMod]->occurence,
               entity_description[type].parameter[param].name, (long)(value + 0.5), t, (long)(modData->unperturbedValue[iMod]));
        modData->lastVerboseValue[iMod] = value;
      }
      break;
    case IS_SHORT:
      lastValue = modData->lastVerboseValue[iMod];
      value = (short)(value + 0.5);
      *((short *)(p_elem + entity_description[type].parameter[param].offset)) = value;
      if (modData->flags[iMod] & VERBOSE_MOD && value != lastValue) {
        printf("Modulation value for element %s#%ld, parameter %s changed to %hd at t = %21.15le (originally %hd)\n",
               modData->element[iMod]->name, modData->element[iMod]->occurence,
               entity_description[type].parameter[param].name, (short)(value + 0.5), t, (short)(modData->unperturbedValue[iMod]));
        modData->lastVerboseValue[iMod] = value;
      }
      break;
    default:
      break;
    }
    
#ifdef DEBUG
    printf("applyElementModulations: Set parameter value on elements(s)\n");
    fflush(stdout);
#endif

    if (modData->fpRecord[iMod]
#if USE_MPI
        && myid == 0
#endif
    ) {
      fprintf(modData->fpRecord[iMod], "%21.15le %ld %21.15le %21.15le %21.15le\n",
              t, iPass, modulation, modData->unperturbedValue[iMod], value);
      if (modData->flushRecord[iMod] > 0 && (iPass % modData->flushRecord[iMod]) == 0)
        fflush(modData->fpRecord[iMod]);
#ifdef DEBUG
      printf("applyElementModulations: Updated record file\n");
      fflush(stdout);
#endif
    }

    if (entity_description[type].flags & HAS_MATRIX &&
        entity_description[type].parameter[param].flags & PARAM_CHANGES_MATRIX &&
        ((modData->flags[iMod] & REFRESH_MATRIX_MOD) ||
         (entity_description[type].flags & (MATRIX_TRACKING + HYBRID_TRACKING)))) {
      /* update the matrix */
      if (modData->element[iMod]->matrix) {
        free_matrices(modData->element[iMod]->matrix);
        tfree(modData->element[iMod]->matrix);
        modData->element[iMod]->matrix = NULL;
      }
      compute_matrix(modData->element[iMod], run, NULL);
      matricesUpdated++;
#ifdef DEBUG
      printf("applyElementModulations: Updated matrices\n");
      fflush(stdout);
#endif
    }
  }
#ifdef DEBUG
  printf("applyElementModulations: returning\n");
  fflush(stdout);
#endif

  return matricesUpdated;
}
