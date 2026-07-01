/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: error.c
 * contents: error_setup(), add_error_element(), perturbation()
 *
 * Michael Borland, 1991
 */
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#endif
#include "mdb.h"
#include "track.h"
#include "error.h"
#include "knobs.h"

long duplicate_name(char **list, long n_list, char *name);
void assignRandomizedIntegersToArray(long *sequence, long n, long min, long step);

#define DEBUG 0

void error_setup(ERRORVAL *errcon, NAMELIST_TEXT *nltext, RUN *run_cond, LINE_LIST *beamline) {
  long i;

  log_entry("error_setup");

  /* reset namelist variables to defaults */
  /*
    clear_error_settings = 1;
    summarize_error_settings = 0;
    error_log = NULL;
 */

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&error_control, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &error_control);

  if (summarize_error_settings) {
    printf("summary of random error settings: \n");
    fflush(stdout);
    if (errcon->no_errors_first_step) {
      printf("No errors will be generated for the first step.\n");
      fflush(stdout);
    }
    for (i = 0; i < errcon->n_items; i++) {
      switch (errcon->error_type[i]) {
      case UNIFORM_ERRORS:
      case GAUSSIAN_ERRORS:
        printf("%8s:  %sadditive %s errors with amplitude %e %s and cutoff %e %s\n",
               errcon->quan_name[i], (errcon->flags[i] & NONADDITIVE_ERRORS ? "non-" : ""),
               known_error_type[errcon->error_type[i]],
               (errcon->flags[i] & FRACTIONAL_ERRORS ? 100 : 1) * errcon->error_level[i],
               errcon->flags[i] & FRACTIONAL_ERRORS ? "%" : errcon->quan_unit[i],
               errcon->error_cutoff[i], (errcon->flags[i] & POST_CORRECTION ? "(post-correction)" : ""));
        fflush(stdout);
        break;
      case PLUS_OR_MINUS_ERRORS:
        printf("%8s:  %sadditive %s errors with amplitude %e\n",
               errcon->quan_name[i],
               (errcon->flags[i] & NONADDITIVE_ERRORS ? "non-" : ""),
               known_error_type[errcon->error_type[i]],
               errcon->error_level[i]);
        fflush(stdout);
        break;
      case SAMPLED_ERRORS:
        printf("%8s:  %serrors sampled (%s) with multiplicative amplitude %e\n",
               errcon->quan_name[i],
               (errcon->flags[i] & NONADDITIVE_ERRORS ? "non-" : ""),
               sampleModeChoice[errcon->errorSamples[errcon->sampleIndex[i]].mode],
               errcon->error_level[i]);
        fflush(stdout);
        break;
      default:
        bombElegant("Invalid error type seen. Seek professional help!", NULL);
        break;
      }
    }
    log_exit("error_setup");
    return;
  }

  errcon->no_errors_first_step = no_errors_for_first_step;

  compute_end_positions(beamline);

  if (errcon->fp_log)
    fclose(errcon->fp_log);
#if USE_MPI
  if (isSlave)
    error_log = NULL;
#endif
  if (error_log) {
    errcon->fp_log = fopen_e(compose_filename(error_log, run_cond->rootname), "w", 0);
    fprintf(errcon->fp_log, "SDDS1\n");
    fprintf(errcon->fp_log,
            "&description text=\"Error log--input: %s  lattice: %s\", contents=\"error log, elegant output\" &end\n",
            run_cond->runfile, run_cond->lattice);
    fprintf(errcon->fp_log, "&associate filename=\"%s\", path=\"%s\", contents=\"elegant input, parent\" &end\n",
            run_cond->runfile, getenv("PWD"));
    fprintf(errcon->fp_log, "&associate filename=\"%s\", path=\"%s\", contents=\"elegant lattice, parent\" &end\n",
            run_cond->lattice, getenv("PWD"));
    fprintf(errcon->fp_log, "&parameter name=Step, type=long, description=\"simulation step\" &end\n");

    fprintf(errcon->fp_log, "&parameter name=When, type=string, description=\"phase of simulation when errors were asserted\" &end\n");
    fprintf(errcon->fp_log, "&column name=ParameterValue, type=double, description=\"Perturbed value\" &end\n");
    fprintf(errcon->fp_log, "&column name=ParameterError, type=double, description=\"Perturbation value\" &end\n");
    fprintf(errcon->fp_log, "&column name=ElementParameter, type=string, description=\"Parameter name\" &end\n");
    fprintf(errcon->fp_log, "&column name=ElementName, type=string, description=\"Element name\" &end\n");
    fprintf(errcon->fp_log, "&column name=ElementOccurence, type=long, description=\"Element occurence\" &end\n");
    fprintf(errcon->fp_log, "&column name=ElementType, type=string, description=\"Element type\" &end\n");
    fprintf(errcon->fp_log, "&data mode=ascii, lines_per_row=1, no_row_counts=1 &end\n");
    fflush(errcon->fp_log);
  } else
    errcon->fp_log = NULL;

  if (clear_error_settings) {
    /* indicate that no error data has been asserted */
    errcon->new_data_read = 0;
    /* clean up the error control structure and flags on elements */
    if (errcon->n_items)
      set_element_flags(beamline, errcon->name, NULL, NULL, NULL, errcon->n_items, PARAMETERS_ARE_STATIC, 0, 1, 0);
    errcon->n_items = 0;
  }
  printf("\n*** Cleared error settings\n");
  fflush(stdout);
  log_exit("error_setup");
}

void add_error_element(ERRORVAL *errcon, NAMELIST_TEXT *nltext, LINE_LIST *beamline) {
  long n_items, n_added, i_start, firstIndexInGroup, errorDistCode;
  ELEMENT_LIST *context;
  long sampleModeCode = -1, nSampleValues = 0;
  double *sampleValue = NULL;
  char *sampleFile = NULL;
  double sMin = -DBL_MAX, sMax = DBL_MAX;

  log_entry("add_error_element");

  if ((n_items = errcon->n_items) == 0) {
    if (errcon->new_data_read)
      bombElegant("improper sequencing of error specifications and tracking", NULL);
    errcon->new_data_read = 1;
  }

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&error_element, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (name == NULL) {
    if (!element_type)
      bombElegant("element name missing in error namelist", NULL);
    SDDS_CopyString(&name, "*");
  }
  if (echoNamelists)
    print_namelist(stdout, &error_element);

  /* check for valid input and copy to errcon arrays.
   * Note: item= is required for element targets, but ignored when name
   * refers to a knob (handled in the knob shortcut below). */
  if (item == NULL && name != NULL && !has_wildcards(name)) {
    char *upTmp = tmalloc(sizeof(*upTmp) * (strlen(name) + 1));
    strcpy(upTmp, name);
    str_toupper(upTmp);
    if (find_knob(upTmp) < 0)
      bombElegant("item name missing in error namelist", NULL);
    free(upTmp);
  } else if (item == NULL) {
    bombElegant("item name missing in error namelist", NULL);
  }
  if ((errorDistCode = match_string(type, known_error_type, N_ERROR_TYPES, 0)) < 0)
    bombElegant("unknown error type specified", NULL);
  if (errorDistCode == SAMPLED_ERRORS) {
    SDDS_DATASET SDDSin;
    if (!sample_file || !strlen(sample_file))
      bombElegant("give valid value for sample_file in sampled error mode", NULL);
    if (!sample_file_column || !strlen(sample_file_column))
      bombElegant("give valid value for sample_file_column in sampled error mode", NULL);
    if (!sample_mode || (sampleModeCode = match_string(sample_mode, sampleModeChoice, N_SAMPLE_MODES, 0)) < 0)
      bombElegant("give valid value for sample_mode in sampled error mode", NULL);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, sample_file))
      bombElegant(NULL, NULL);
    switch (SDDS_ReadPage(&SDDSin)) {
    case 0:
    case -1:
      bombElegantVA("Error: error_element: no data in file %s\n", sample_file);
      break;
    default:
      if ((nSampleValues = SDDS_RowCount(&SDDSin)) < 1)
        bombElegantVA("Error: error_element: insufficient data in file %s\n", sample_file);
      if (!(sampleValue = SDDS_GetColumnInDoubles(&SDDSin, sample_file_column)))
        bombElegant(NULL, NULL);
      printf("Read %ld values from %s\n", nSampleValues, sample_file);
      break;
    }
  } else {
    if (sample_file && strlen(sample_file))
      printWarning("error_element", "sample_file given but error type is not \"sampled\"");
    if (sample_file_column && strlen(sample_file_column))
      printWarning("error_element", "sample_file_column given but error type is not \"sampled\"");
    if (sample_mode && strlen(sample_mode))
      printWarning("error_element", "sample_mode given but error type is not \"sampled\"");
  }
  if (bind_number < 0)
    bombElegant("bind_number < 0", NULL);
  if (!additive && fractional)
    bombElegant("fractional errors must be additive", NULL);

  context = NULL;
  n_added = 0;
  i_start = n_items;

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
  if (element_type && has_wildcards(element_type) && strchr(element_type, '-'))
    element_type = expand_ranges(element_type);

  /* Knob shortcut: if `name` (no wildcards) matches a knob, register a single
   * ERRORVAL slot encoded with elem_type=T_KNOB and param_number=knobIndex;
   * skip element iteration entirely. */
  if (!has_wildcards(name)) {
    long knobIndex;
    char *upName = tmalloc(sizeof(*upName) * (strlen(name) + 1));
    strcpy(upName, name);
    str_toupper(upName);
    knobIndex = find_knob(upName);
    if (knobIndex >= 0) {
      if (item && strlen(item))
        printWarning("error_element: item= is ignored when name= matches a knob.", upName);
      if (fractional)
        bombElegant("error_element: fractional errors are not supported for knobs", upName);
      errcon->name = trealloc(errcon->name, sizeof(*errcon->name) * (n_items + 1));
      errcon->item = trealloc(errcon->item, sizeof(*errcon->item) * (n_items + 1));
      errcon->quan_name = trealloc(errcon->quan_name, sizeof(*errcon->quan_name) * (n_items + 1));
      errcon->quan_final_index = trealloc(errcon->quan_final_index, sizeof(*errcon->quan_final_index) * (n_items + 1));
      errcon->quan_unit = trealloc(errcon->quan_unit, sizeof(*errcon->quan_unit) * (n_items + 1));
      errcon->error_level = trealloc(errcon->error_level, sizeof(*errcon->error_level) * (n_items + 1));
      errcon->error_cutoff = trealloc(errcon->error_cutoff, sizeof(*errcon->error_cutoff) * (n_items + 1));
      errcon->error_type = trealloc(errcon->error_type, sizeof(*errcon->error_type) * (n_items + 1));
      errcon->elem_type = trealloc(errcon->elem_type, sizeof(*errcon->elem_type) * (n_items + 1));
      errcon->param_number = trealloc(errcon->param_number, sizeof(*errcon->param_number) * (n_items + 1));
      errcon->sampleIndex = trealloc(errcon->sampleIndex, sizeof(*errcon->sampleIndex) * (n_items + 1));
      errcon->flags = trealloc(errcon->flags, sizeof(*errcon->flags) * (n_items + 1));
      errcon->bind_number = trealloc(errcon->bind_number, sizeof(*errcon->bind_number) * (n_items + 1));
      errcon->boundTo = trealloc(errcon->boundTo, sizeof(*errcon->boundTo) * (n_items + 1));
      errcon->unperturbed_value = trealloc(errcon->unperturbed_value, sizeof(*errcon->unperturbed_value) * (n_items + 1));
      errcon->error_value = trealloc(errcon->error_value, sizeof(*errcon->error_value) * (n_items + 1));
      errcon->sMin = trealloc(errcon->sMin, sizeof(*errcon->sMin) * (n_items + 1));
      errcon->sMax = trealloc(errcon->sMax, sizeof(*errcon->sMax) * (n_items + 1));

      cp_str(errcon->name + n_items, upName);
      cp_str(errcon->item + n_items, "value");
      errcon->quan_name[n_items] = tmalloc(sizeof(char) * (strlen(upName) + 8));
      sprintf(errcon->quan_name[n_items], "d%s.value", upName);
      errcon->quan_final_index[n_items] = -1;
      cp_str(&errcon->quan_unit[n_items], "");
      errcon->error_level[n_items] = amplitude * error_factor;
      errcon->error_cutoff[n_items] = cutoff;
      errcon->error_type[n_items] = errorDistCode;
      errcon->elem_type[n_items] = T_KNOB;
      errcon->param_number[n_items] = knobIndex;
      errcon->sampleIndex[n_items] = -1;
      errcon->flags[n_items] = 0;
      errcon->flags[n_items] += (bind == 0 ? 0 : (bind == -1 ? ANTIBIND_ERRORS : BIND_ERRORS));
      errcon->flags[n_items] += (post_correction ? POST_CORRECTION : PRE_CORRECTION);
      /* Note: NONADDITIVE_ERRORS and FRACTIONAL_ERRORS are not honored on
       * knobs; perturb_knob always adds delta to the current knob value. */
      errcon->bind_number[n_items] = bind_number;
      errcon->boundTo[n_items] = -1;
      errcon->unperturbed_value[n_items] = 0.0;
      errcon->error_value[n_items] = 0.0;
      errcon->sMin[n_items] = sMin;
      errcon->sMax[n_items] = sMax;
      errcon->n_items = n_items + 1;
      free(upName);
      log_exit("add_error_element");
      return;
    }
    free(upName);
  }

  if (has_wildcards(name)) {
    if (strchr(name, '-'))
      name = expand_ranges(name);
    str_toupper(name);
    firstIndexInGroup = -1;
    while ((context = wfind_element(name, &context, beamline->elem))) {
      if (element_type &&
          !wild_match(entity_name[context->type], element_type))
        continue;
      if (exclude &&
          wild_match(context->name, exclude))
        continue;
      if (i_start != n_items && duplicate_name(errcon->name + i_start, n_items - i_start, context->name))
        continue;
      if ((sMin >= 0 && context->end_pos < sMin) ||
          (sMax >= 0 && context->end_pos > sMax))
        continue;
      errcon->name = trealloc(errcon->name, sizeof(*errcon->name) * (n_items + 1));
      errcon->item = trealloc(errcon->item, sizeof(*errcon->item) * (n_items + 1));
      errcon->quan_name = trealloc(errcon->quan_name, sizeof(*errcon->quan_name) * (n_items + 1));
      errcon->quan_final_index = trealloc(errcon->quan_final_index, sizeof(*errcon->quan_final_index) * (n_items + 1));
      errcon->quan_unit = trealloc(errcon->quan_unit, sizeof(*errcon->quan_unit) * (n_items + 1));
      errcon->error_level = trealloc(errcon->error_level, sizeof(*errcon->error_level) * (n_items + 1));
      errcon->error_cutoff = trealloc(errcon->error_cutoff, sizeof(*errcon->error_cutoff) * (n_items + 1));
      errcon->error_type = trealloc(errcon->error_type, sizeof(*errcon->error_type) * (n_items + 1));
      errcon->elem_type = trealloc(errcon->elem_type, sizeof(*errcon->elem_type) * (n_items + 1));
      errcon->error_value = trealloc(errcon->error_value, sizeof(*errcon->error_value) * (n_items + 1));
      errcon->unperturbed_value = trealloc(errcon->unperturbed_value, sizeof(*errcon->unperturbed_value) * (n_items + 1));
      errcon->param_number = trealloc(errcon->param_number, sizeof(*errcon->param_number) * (n_items + 1));
      errcon->bind_number = trealloc(errcon->bind_number, sizeof(*errcon->bind_number) * (n_items + 1));
      errcon->flags = trealloc(errcon->flags, sizeof(*errcon->flags) * (n_items + 1));
      errcon->sMin = trealloc(errcon->sMin, sizeof(*errcon->sMin) * (n_items + 1));
      errcon->sMax = trealloc(errcon->sMax, sizeof(*errcon->sMax) * (n_items + 1));
      errcon->boundTo = trealloc(errcon->boundTo, sizeof(*errcon->boundTo) * (n_items + 1));
      errcon->sampleIndex = trealloc(errcon->sampleIndex, sizeof(*errcon->sampleIndex) * (n_items + 1));
      errcon->sampleIndex[n_items] = -1;

      if (errorDistCode == SAMPLED_ERRORS) {
        long nSeq;
        if (firstIndexInGroup == -1) {
          errcon->sampleIndex[n_items] = nSeq = errcon->nErrorSampleSets;
          errcon->errorSamples = trealloc(errcon->errorSamples, sizeof(*errcon->errorSamples) * (nSeq + 1));
          errcon->errorSamples[nSeq].sourceData = sampleValue;
          if (sampleModeCode == SAMPLE_RANDOM_EXHAUST_REUSE) {
            errcon->errorSamples[nSeq].sequence = tmalloc(sizeof(*errcon->errorSamples[nSeq].sequence) * nSampleValues);
            assignRandomizedIntegersToArray(errcon->errorSamples[nSeq].sequence, nSampleValues, 0, 1);
          }
          errcon->errorSamples[nSeq].nValues = nSampleValues;
          errcon->errorSamples[nSeq].iSequence = 0;
          errcon->errorSamples[nSeq].mode = sampleModeCode;
          errcon->nErrorSampleSets += 1;
        } else
          errcon->sampleIndex[n_items] = errcon->nErrorSampleSets - 1;
      }

      cp_str(errcon->item + n_items, str_toupper(item));
      cp_str(errcon->name + n_items, context->name);
      errcon->error_level[n_items] = amplitude * error_factor;
      errcon->error_cutoff[n_items] = cutoff;
      errcon->error_type[n_items] = errorDistCode; /* match_string(type, known_error_type, N_ERROR_TYPES, 0); */
      errcon->quan_name[n_items] = tmalloc(sizeof(char *) * (strlen(context->name) + strlen(item) + 4));
      errcon->quan_final_index[n_items] = -1;
      sprintf(errcon->quan_name[n_items], "d%s.%s", context->name, item);
      errcon->flags[n_items] = (fractional ? FRACTIONAL_ERRORS : 0);
      errcon->flags[n_items] += (bind == 0 ? 0 : (bind == -1 ? ANTIBIND_ERRORS : BIND_ERRORS));
      errcon->flags[n_items] += (post_correction ? POST_CORRECTION : PRE_CORRECTION);
      errcon->flags[n_items] += (additive ? 0 : NONADDITIVE_ERRORS);
      errcon->bind_number[n_items] = bind_number;
      /* boundTo must be -1 when there is no cross-name binding or when this element is the
             * first of a series 
             */
      errcon->boundTo[n_items] = bind_across_names ? firstIndexInGroup : -1;
      /*
            if (errcon->boundTo[n_items]!=-1)
              printf("%s bound to %s\n", 
                      errcon->name[n_items], errcon->name[errcon->boundTo[n_items]]);
*/
      errcon->sMin[n_items] = sMin;
      errcon->sMax[n_items] = sMax;
      errcon->elem_type[n_items] = context->type;

      if ((errcon->param_number[n_items] = confirm_parameter(item, context->type)) < 0) {
        printf("error: cannot vary %s--no such parameter for %s (wildcard name: %s)\n", item, context->name, name);
        fflush(stdout);
        exitElegant(1);
      }
      cp_str(&errcon->quan_unit[n_items],
             errcon->flags[n_items] & FRACTIONAL_ERRORS ? "fr" : entity_description[errcon->elem_type[n_items]].parameter[errcon->param_number[n_items]].unit);
      errcon->unperturbed_value[n_items] = parameter_value(errcon->name[n_items], errcon->elem_type[n_items], errcon->param_number[n_items],
                                                           beamline);
      if (errcon->unperturbed_value[n_items] == 0 && errcon->flags[n_items] & FRACTIONAL_ERRORS) {
        char warningText[1024];
        snprintf(warningText, 1024, "Element %s, item %s",
                 errcon->name[n_items], errcon->item[n_items]);
        printWarning("Fractional error specified for unperturbed value of zero.", warningText);
      }
      if (duplicate_name(errcon->quan_name, n_items, errcon->quan_name[n_items])) {
        char warningText[1024];
        snprintf(warningText, 1024, "Element %s, item %s",
                 errcon->name[n_items], errcon->item[n_items]);
        printWarning("Errors specified more than once for the same quantity.",
                     warningText);
      }
      errcon->n_items = ++n_items;
      n_added++;
      if (firstIndexInGroup == -1)
        firstIndexInGroup = n_items - 1;
    }
  } else {
    str_toupper(name);
    if (!(context = find_element(name, &context, beamline->elem))) {
      printf("error: cannot add errors to element %s--not in beamline\n", name);
      fflush(stdout);
      exitElegant(1);
    }
    errcon->name = trealloc(errcon->name, sizeof(*errcon->name) * (n_items + 1));
    errcon->item = trealloc(errcon->item, sizeof(*errcon->item) * (n_items + 1));
    errcon->quan_name = trealloc(errcon->quan_name, sizeof(*errcon->quan_name) * (n_items + 1));
    errcon->quan_final_index = trealloc(errcon->quan_final_index, sizeof(*errcon->quan_final_index) * (n_items + 1));
    errcon->quan_unit = trealloc(errcon->quan_unit, sizeof(*errcon->quan_unit) * (n_items + 1));
    errcon->error_level = trealloc(errcon->error_level, sizeof(*errcon->error_level) * (n_items + 1));
    errcon->error_cutoff = trealloc(errcon->error_cutoff, sizeof(*errcon->error_cutoff) * (n_items + 1));
    errcon->error_type = trealloc(errcon->error_type, sizeof(*errcon->error_type) * (n_items + 1));
    errcon->elem_type = trealloc(errcon->elem_type, sizeof(*errcon->elem_type) * (n_items + 1));
    errcon->error_value = trealloc(errcon->error_value, sizeof(*errcon->error_value) * (n_items + 1));
    errcon->unperturbed_value = trealloc(errcon->unperturbed_value, sizeof(*errcon->unperturbed_value) * (n_items + 1));
    errcon->param_number = trealloc(errcon->param_number, sizeof(*errcon->param_number) * (n_items + 1));
    errcon->bind_number = trealloc(errcon->bind_number, sizeof(*errcon->bind_number) * (n_items + 1));
    errcon->flags = trealloc(errcon->flags, sizeof(*errcon->flags) * (n_items + 1));
    errcon->sMin = trealloc(errcon->sMin, sizeof(*errcon->sMin) * (n_items + 1));
    errcon->sMax = trealloc(errcon->sMax, sizeof(*errcon->sMax) * (n_items + 1));
    errcon->boundTo = trealloc(errcon->boundTo, sizeof(*errcon->boundTo) * (n_items + 1));
    errcon->sampleIndex = trealloc(errcon->sampleIndex, sizeof(*errcon->sampleIndex) * (n_items + 1));
    errcon->sampleIndex[n_items] = -1;

    if (errorDistCode == SAMPLED_ERRORS) {
      long nSeq;
      nSeq = errcon->nErrorSampleSets;
      errcon->sampleIndex[n_items] = nSeq;
      errcon->errorSamples = trealloc(errcon->errorSamples, sizeof(*errcon->errorSamples) * (nSeq + 1));
      errcon->errorSamples[nSeq].sourceData = sampleValue;
      if (sampleModeCode == SAMPLE_RANDOM_EXHAUST_REUSE) {
        long ii;
        errcon->errorSamples[nSeq].sequence = tmalloc(sizeof(*errcon->errorSamples[nSeq].sequence) * nSampleValues);
        assignRandomizedIntegersToArray(errcon->errorSamples[nSeq].sequence, nSampleValues, 0, 1);
        for (ii = 0; ii < nSampleValues; ii++)
          printf("%ld: %ld\n", ii, errcon->errorSamples[nSeq].sequence[ii]);
        exit(0);
      } else
        errcon->errorSamples[nSeq].sequence = NULL;
      errcon->errorSamples[nSeq].nValues = nSampleValues;
      errcon->errorSamples[nSeq].iSequence = 0;
      errcon->errorSamples[nSeq].mode = sampleModeCode;
      errcon->nErrorSampleSets += 1;
    }

    cp_str(errcon->item + n_items, str_toupper(item));
    cp_str(errcon->name + n_items, context->name);
    errcon->error_level[n_items] = amplitude;
    errcon->error_cutoff[n_items] = cutoff;
    errcon->error_type[n_items] = match_string(type, known_error_type, N_ERROR_TYPES, 0);
    errcon->quan_name[n_items] = tmalloc(sizeof(char *) * (strlen(context->name) + strlen(item) + 4));
    sprintf(errcon->quan_name[n_items], "d%s.%s", context->name, item);
    errcon->flags[n_items] = (fractional ? FRACTIONAL_ERRORS : 0);
    errcon->flags[n_items] += (bind == 0 ? 0 : (bind == -1 ? ANTIBIND_ERRORS : BIND_ERRORS));
    errcon->flags[n_items] += (post_correction ? POST_CORRECTION : PRE_CORRECTION);
    errcon->flags[n_items] += (additive ? 0 : NONADDITIVE_ERRORS);
    errcon->bind_number[n_items] = bind_number;
    errcon->sMin[n_items] = sMin;
    errcon->sMax[n_items] = sMax;
    errcon->boundTo[n_items] = -1; /* not used when there are no wildcards */

    errcon->elem_type[n_items] = context->type;
    if ((errcon->param_number[n_items] = confirm_parameter(item, context->type)) < 0) {
      printf("error: cannot vary %s--no such parameter for %s\n", item, name);
      fflush(stdout);
      exitElegant(1);
    }
    cp_str(&errcon->quan_unit[n_items],
           errcon->flags[n_items] & FRACTIONAL_ERRORS ? "fr" : entity_description[errcon->elem_type[n_items]].parameter[errcon->param_number[n_items]].unit);
    errcon->unperturbed_value[n_items] = parameter_value(errcon->name[n_items], errcon->elem_type[n_items], errcon->param_number[n_items],
                                                         beamline);
    if (errcon->unperturbed_value[n_items] == 0 && errcon->flags[n_items] & FRACTIONAL_ERRORS) {
      char warningText[1024];
      snprintf(warningText, 1024, "Element %s, item %s",
               errcon->name[n_items], errcon->item[n_items]);
      printWarning("Fractional error specified for unperturbed value of zero.", warningText);
    }
    if (duplicate_name(errcon->quan_name, n_items, errcon->quan_name[n_items])) {
      char warningText[1024];
      snprintf(warningText, 1024, "Element %s, item %s",
               errcon->name[n_items], errcon->item[n_items]);
      printWarning("Errors specified more than once for the same quantity.",
                   warningText);
    }
    errcon->n_items = ++n_items;
    n_added++;
  }

  if (!n_added && !allow_missing_elements) {
    printf("error: no match for name %s\n", name);
    fflush(stdout);
    exitElegant(1);
  }
  if (sampleFile)
    free(sampleFile);
  log_exit("add_error_element");
}

double parameter_value(char *pname, long elem_type, long param, LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  long lresult;
  short sresult;
  double dresult;
  char *p_elem;
  long data_type;

  log_entry("parameter_value");

  eptr = NULL;
  data_type = entity_description[elem_type].parameter[param].type;
  if (find_element(pname, &eptr, beamline->elem)) {
    p_elem = eptr->p_elem;
    switch (data_type) {
    case IS_DOUBLE:
      dresult = *((double *)(p_elem + entity_description[elem_type].parameter[param].offset));
      log_exit("parameter_value");
      return (dresult);
    case IS_LONG:
      lresult = *((long *)(p_elem + entity_description[elem_type].parameter[param].offset));
      log_exit("parameter_value");
      return ((double)lresult);
    case IS_SHORT:
      sresult = *((short *)(p_elem + entity_description[elem_type].parameter[param].offset));
      log_exit("parameter_value");
      return ((double)sresult);
    case IS_STRING:
    default:
      bombElegant("unknown/invalid variable quantity (parameter_value)", NULL);
      exitElegant(1);
    }
  }
  printf("error: unable to find value of parameter %ld for element %s of type %ld\n",
         param, pname, elem_type);
  fflush(stdout);
  exitElegant(1);
  return (0.0);
}

double perturbation(double xamplitude, double xcutoff, long xerror_type, long sampleIndex, ERROR_SAMPLES *errorSamples) {
  double value;
  switch (xerror_type) {
  case UNIFORM_ERRORS:
    return (2 * xamplitude * (random_1_elegant(0) - 0.5));
  case GAUSSIAN_ERRORS:
    return (gauss_rn_lim(0.0, xamplitude, xcutoff, random_1_elegant));
  case PLUS_OR_MINUS_ERRORS:
    /* return either -x or x */
    return (xamplitude * (random_1_elegant(0) > 0.5 ? 1.0 : -1.0));
  case SAMPLED_ERRORS:
    if (errorSamples[sampleIndex].iSequence >= errorSamples[sampleIndex].nValues) {
      errorSamples[sampleIndex].iSequence = 0;
      if (errorSamples[sampleIndex].mode == SAMPLE_RANDOM_EXHAUST_REUSE)
        assignRandomizedIntegersToArray(errorSamples[sampleIndex].sequence, errorSamples[sampleIndex].nValues, 0, 1);
    }
    value = 0;
    switch (errorSamples[sampleIndex].mode) {
    case SAMPLE_RANDOM_REPLACE:
      value = errorSamples[sampleIndex].sourceData[(long)(errorSamples[sampleIndex].nValues * random_1_elegant(0))];
      break;
    case SAMPLE_RANDOM_EXHAUST_REUSE:
      value = errorSamples[sampleIndex].sourceData[errorSamples[sampleIndex].sequence[errorSamples[sampleIndex].iSequence]];
      break;
    case SAMPLE_SEQUENTIAL_REUSE:
      value = errorSamples[sampleIndex].sourceData[errorSamples[sampleIndex].iSequence];
      break;
    }
    if (errorSamples[sampleIndex].mode != SAMPLE_RANDOM_REPLACE)
      errorSamples[sampleIndex].iSequence += 1;
    return value;
    break;
  default:
    bombElegant("unknown error type in perturbation()", NULL);
    exitElegant(1);
    break;
  }
  return (0.0);
}

long duplicate_name(char **list, long n_list, char *name) {
  while (n_list--)
    if (strcmp(*list++, name) == 0)
      return (1);
  return (0);
}

void assignRandomizedIntegersToArray(long *sequence, long n, long min, long step)
/* generate sequential integers from min to min+(n-1)*step, then randomize the order */
{
  double *randomValue;
  long *sortedIndex, i;

  if (n < 2)
    return;

  /* generate random values and sort them to get the (randomized) indices */
  randomValue = tmalloc(sizeof(*randomValue) * n);
  for (i = 0; i < n; i++)
    randomValue[i] = random_1_elegant(0);
  sortedIndex = sort_and_return_index(randomValue, SDDS_DOUBLE, n, 0);

  /* assign sequential integers (with a step size) in random order */
  for (i = 0; i < n; i++)
    sequence[sortedIndex[i]] = min + step * i;

  free(randomValue);
  free(sortedIndex);
}
