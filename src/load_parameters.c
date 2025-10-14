/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: load_parameters.c
 * contents: setup_load_parameters(), do_load_parameters()
 *
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "track.h"
#include "load_parameters.h"
#include "SDDS.h"
#include "match_string.h"

#define DEBUG 0

/* structure to store and manage a load_parameters request */
typedef struct {
  SDDS_TABLE table;
  char *filename;
  unsigned long flags;
#define COMMAND_FLAG_CHANGE_DEFINITIONS 0x0001UL
#define COMMAND_FLAG_IGNORE             0x0002UL
#define COMMAND_FLAG_IGNORE_OCCURENCE   0x0004UL
#define COMMAND_FLAG_USE_FIRST          0x0008UL
#define ALLOW_MISSING_ELEMENTS          0x0010UL
#define ALLOW_MISSING_PARAMETERS        0x0020UL
#define NUMERICAL_DATA_PRESENT          0x0040UL
#define STRING_DATA_PRESENT             0x0080UL
#define COMMAND_FLAG_SCRIPT_TRIGGERED   0x0100UL
  char **includeNamePattern, **includeItemPattern, **includeTypePattern;
  long includeNamePatterns, includeItemPatterns, includeTypePatterns;
  char **excludeNamePattern, **excludeItemPattern, **excludeTypePattern;
  long excludeNamePatterns, excludeItemPatterns, excludeTypePatterns;
  char *editNameCommand;
  short repeat_first_page_at_each_step; /* If non-zero, the file is expected to be single-page and is asserted at each step */
  long skip_pages;           /* if >0, pages are skipped, used as counter */
  long last_code;            /* return code from SDDS_ReadTable */
  double *starting_value;    /* only for numerical data */
  char *use_start;           /* if nonzero, starting_value will be used to restore to initial state */
  void **reset_address;
  short *value_type;
  ELEMENT_LIST **element;
  long *element_flags;
  long values;
  double prefactor;          /* multiply values by this factor before using */
} LOAD_PARAMETERS;

/* variables to store and manage load_parameters requests */
static LOAD_PARAMETERS *load_request = NULL;
static long load_requests = 0;
static long load_parameters_setup = 0;

/* names of the SDDS columns that will be used */
static char *Element_ColumnName = "ElementName";
static char *Occurence_ColumnName = "ElementOccurence";
static char *Parameter_ColumnName = "ElementParameter";
static char *Value_ColumnName = "ParameterValue";
static char *ValueString_ColumnName = "ParameterValueString";
static char *Mode_ColumnName = "ParameterMode";
static char *ElementType_ColumnName = "ElementType";

/* the order here must match the order of the #define's in track.h */
#define LOAD_MODES 4
static char *load_mode[LOAD_MODES] = {
  "absolute", "differential", "ignore", "fractional"};

long setup_load_parameters_for_file(char *filename, RUN *run, LINE_LIST *beamline);

static long printingEnabled = 1;

long setup_load_parameters(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  long i = 0;

#if !USE_MPI
  printingEnabled = 1;
#else
  printingEnabled = myid == 1 ? 1 : 0;
#endif

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&load_parameters, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &load_parameters);

  if (force_occurence_data && use_first)
    bombElegant("Error: force_occurence_data=1 and use_first=1 not meaningful\n", NULL);

  if (filename_list && strlen(filename_list)) {
    printf("Using list of filenames for parameter loading\n");
  } else {
    if (!filename && !clear_settings)
      bombElegant("filename or filename_list must be given for load_parameters unless you are clearing settings", NULL);
    printf("Using single filename for parameter loading\n");
  }

  if (repeat_first_page_at_each_step) {
    if (change_defined_values)
      bombElegant("change_defined_values and repeat_first_page_at_each_step modes are incompatible", NULL);
    if (skip_pages)
      bombElegant("skip_pages feature and repeat_first_page_at_each_step mode are incompatible", NULL);
  }

  if (clear_settings && load_requests) {
    for (i = 0; i < load_requests; i++) {
      if (!SDDS_Terminate(&load_request[i].table))
        bombElegant("problem terminating load_parameters table", NULL);
    }
    load_requests = 0;
  }
  if (clear_settings)
    return 0;
  load_parameters_setup = 1;

  if (filename_list) {
    char *filename0;
    while ((filename0 = get_token(filename_list)) != NULL)
      i = setup_load_parameters_for_file(filename0, run, beamline);
    return i;
  } else
    return setup_load_parameters_for_file(filename, run, beamline);
}

long setup_load_parameters_for_file(char *filename, RUN *run, LINE_LIST *beamline) {
  long index;

  load_request = trealloc(load_request, sizeof(*load_request) * (load_requests + 1));
  load_request[load_requests].flags = (change_defined_values ? COMMAND_FLAG_CHANGE_DEFINITIONS : 0) +
    (allow_missing_elements ? ALLOW_MISSING_ELEMENTS : 0) +
    (allow_missing_parameters ? ALLOW_MISSING_PARAMETERS : 0) +
    (script_triggered ? COMMAND_FLAG_SCRIPT_TRIGGERED : 0);
  load_request[load_requests].filename = compose_filename(filename, run->rootname);
  if (change_defined_values && !force_occurence_data)
    load_request[load_requests].flags |= COMMAND_FLAG_IGNORE_OCCURENCE;
  if (use_first)
    load_request[load_requests].flags |= COMMAND_FLAG_USE_FIRST;
  load_request[load_requests].includeNamePattern = addPatterns(&(load_request[load_requests].includeNamePatterns), include_name_pattern);
  load_request[load_requests].includeNamePattern = addPatterns(&(load_request[load_requests].includeNamePatterns), include_name_pattern);
  load_request[load_requests].includeItemPattern = addPatterns(&(load_request[load_requests].includeItemPatterns), include_item_pattern);
  load_request[load_requests].includeTypePattern = addPatterns(&(load_request[load_requests].includeTypePatterns), include_type_pattern);
  load_request[load_requests].excludeNamePattern = addPatterns(&(load_request[load_requests].excludeNamePatterns), exclude_name_pattern);
  load_request[load_requests].excludeItemPattern = addPatterns(&(load_request[load_requests].excludeItemPatterns), exclude_item_pattern);
  load_request[load_requests].excludeTypePattern = addPatterns(&(load_request[load_requests].excludeTypePatterns), exclude_type_pattern);
  load_request[load_requests].skip_pages = skip_pages;
  load_request[load_requests].repeat_first_page_at_each_step = repeat_first_page_at_each_step;
#ifdef USE_MPE /* use the MPE library */
  int event1a, event1b;
  event1a = MPE_Log_get_event_number();
  event1b = MPE_Log_get_event_number();
  if (isMaster)
    MPE_Describe_state(event1a, event1b, "load_parameters", "blue");
  MPE_Log_event(event1a, 0, "start load_parameters"); /* record time spent on reading input */
#endif

  SDDS_ClearErrors();
#if SDDS_MPI_IO
  /* All the processes will read the wake file, but not in parallel.
     Zero the Memory when call  SDDS_InitializeInput */
  load_request[load_requests].table.parallel_io = 0;
#endif

  if (!SDDS_InitializeInputFromSearchPath(&load_request[load_requests].table,
                                          load_request[load_requests].filename)) {
    if (printingEnabled && allow_missing_files)
      printWarning("load_parameters: Couldn't initialize SDDS input for a file.",
                   load_request[load_requests].filename);
    if (!allow_missing_files) {
      printf("Error: couldn't initialize SDDS input for load_parameters file %s\n",
             load_request[load_requests].filename);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDS_ClearErrors();
    return 1;
  }
  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, Element_ColumnName)) < 0 ||
      SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_STRING) {
    if (printingEnabled) {
      printf("Column \"%s\" is not in file %s or is not of string type.\n", Element_ColumnName,
             load_request[load_requests].filename);
      fflush(stdout);
    }
    exitElegant(1);
  }
  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, Parameter_ColumnName)) < 0 ||
      SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_STRING) {
    if (printingEnabled) {
      printf("Column \"%s\" is not in file %s or is not of string type.\n", Parameter_ColumnName,
             load_request[load_requests].filename);
      fflush(stdout);
    }
    exitElegant(1);
  }

  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, Value_ColumnName)) >= 0) {
    if (SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_DOUBLE) {
      if (printingEnabled) {
        printf("Column \"%s\" in file %s is not of double-precision type.\n",
               Value_ColumnName, load_request[load_requests].filename);
        fflush(stdout);
      }
      exitElegant(1);
    }
    load_request[load_requests].flags |= NUMERICAL_DATA_PRESENT;
  }
  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, ValueString_ColumnName)) >= 0) {
    if (SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_STRING) {
      if (printingEnabled) {
        printf("Column \"%s\" is in file %s not of string type.\n",
               ValueString_ColumnName, load_request[load_requests].filename);
        fflush(stdout);
      }
      exitElegant(1);
    }
    load_request[load_requests].flags |= STRING_DATA_PRESENT;
  }
  if ((load_request[load_requests].flags & NUMERICAL_DATA_PRESENT) &&
      (load_request[load_requests].flags & STRING_DATA_PRESENT)) {
    printf("Both numerical and string data columns present in file %s. Will use the latter for actual string quantities only.\n",
           load_request[load_requests].filename);
  }

  if ((include_type_pattern || exclude_type_pattern) &&
      ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, ElementType_ColumnName)) < 0 ||
       SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_STRING)) {
    if (printingEnabled) {
      printf("include_type_pattern and/or exclude_type_pattern given, but\n");
      printf("column \"%s\" is not in file %s or is not of string type.\n",
             ElementType_ColumnName, load_request[load_requests].filename);
      fflush(stdout);
    }
    exitElegant(1);
  }

  /* The Mode column is optional: */
  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, Mode_ColumnName)) >= 0 &&
      SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_STRING) {
    if (printingEnabled) {
      printf("Column \"%s\" is in file %s but is not of string type.\n",
             Mode_ColumnName, load_request[load_requests].filename);
      fflush(stdout);
    }
    exitElegant(1);
  }
  /* The Occurence column is optional: */
  if ((index = SDDS_GetColumnIndex(&load_request[load_requests].table, Occurence_ColumnName)) >= 0 &&
      SDDS_GetColumnType(&load_request[load_requests].table, index) != SDDS_LONG) {
    if (printingEnabled) {
      printf("Column \"%s\" is in file %s but is not of long-integer type.\n",
             Occurence_ColumnName, load_request[load_requests].filename);
      fflush(stdout);
    }
    exitElegant(1);
  }

  if (SDDS_NumberOfErrors()) {
    if (printingEnabled) {
      printf("error: an uncaught error occured in SDDS routines (setup_load_parameters):\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    }
    exitElegant(1);
  }

  load_request[load_requests].last_code = 0;
  load_request[load_requests].starting_value = NULL;
  load_request[load_requests].use_start = NULL;
  load_request[load_requests].element = NULL;
  load_request[load_requests].reset_address = NULL;
  load_request[load_requests].value_type = NULL;
  load_request[load_requests].element_flags = NULL;
  load_request[load_requests].values = 0;
  load_request[load_requests].editNameCommand = edit_name_command;
  load_requests++;
  if (load_request[load_requests - 1].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) {
    /* do this right away so that it gets propagated into error and vary operations */
    do_load_parameters(beamline, 1, NULL);
    if (printingEnabled) {
      printf("New length per pass: %21.15e m\n",
             compute_end_positions(beamline));
    }
#ifdef USE_MPE
    MPE_Log_event(event1b, 0, "end load_parameters");
#endif
    /* No reason to keep this, so just decrement the counter */
    SDDS_Terminate(&load_request[load_requests - 1].table);
    load_requests--;
    return 1;
  }
#ifdef USE_MPE
  MPE_Log_event(event1b, 0, "end load_parameters");
#endif
  return 0;
}

long do_load_parameters(LINE_LIST *beamline, long change_definitions, char *scriptFile) {
  long i, j, mode_flags, code, rows, param, allFilesRead, allFilesIgnored;
  char **element, **parameter, **type, **mode, *p_elem, *p_elem0, *ptr, lastMissingElement[100];
  double *value, newValue;
  char **valueString;
  ELEMENT_LIST *eptr;
  long element_missing;
  int32_t numberChanged, totalNumberChanged = 0;
  long lastMissingOccurence = 0;
  int32_t *occurence;
  char elem_param[1024], inHash=0;
  htab *hash_table;
  short wildMatch;
  
  if (!load_requests || !load_parameters_setup)
    return NO_LOAD_PARAMETERS;
  allFilesRead = 1;
  allFilesIgnored = 1;

  for (i = 0; i < load_requests; i++) {
    if ((load_request[i].flags & COMMAND_FLAG_IGNORE) ||
        (!(load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) && change_definitions) ||
        ((load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) && !change_definitions))
      continue;
    if (load_request[i].flags&COMMAND_FLAG_SCRIPT_TRIGGERED) {
      /* If no filename given, or doesn't match this load request, skip this load request */
      if (!scriptFile || !strlen(scriptFile) || strcmp(scriptFile, load_request[i].filename)!=0)
	continue;
      /* Close and reopen the file to ensure we read the first page */
      SDDS_Terminate(&load_request[i].table);
      SDDS_InitializeInputFromSearchPath(&load_request[i].table, load_request[i].filename);
    } else {
      /* If this request is script triggered but no trigger filename is passed, skip this load request */
      if (scriptFile && strlen(scriptFile))
	continue;
    }
    hash_table = hcreate(12); /* create a hash table with the size of 2^12, it can grow automatically if necessary */

    allFilesIgnored = 0;
    if (load_request[i].last_code) {
      printf("Reasserting starting values before loading parameters\n");
      for (j = 0; j < load_request[i].values; j++) {
        load_request[i].element[j]->flags = load_request[i].element_flags[j];
        if (load_request[i].use_start[j]) {
          switch (load_request[i].value_type[j]) {
          case IS_DOUBLE:
            *((double *)(load_request[i].reset_address[j])) = load_request[i].starting_value[j];
            break;
          case IS_LONG:
            *((long *)(load_request[i].reset_address[j])) = nearestInteger(load_request[i].starting_value[j]);
            break;
          case IS_SHORT:
            *((short *)(load_request[i].reset_address[j])) = nearestInteger(load_request[i].starting_value[j]);
            break;
          }
        }
      }
    }
    while (load_request[i].skip_pages > 0) {
      printf("Skipping page\n");
      code = load_request[i].last_code = SDDS_ReadTable(&load_request[i].table);
      load_request[i].skip_pages--;
    }
    if ((code = load_request[i].last_code = SDDS_ReadTable(&load_request[i].table)) < 1) {
      free(load_request[i].reset_address);
      load_request[i].reset_address = NULL;
      free(load_request[i].value_type);
      load_request[i].value_type = NULL;
      free(load_request[i].element_flags);
      load_request[i].element_flags = NULL;
      free(load_request[i].starting_value);
      load_request[i].starting_value = NULL;
      free(load_request[i].use_start);
      load_request[i].use_start = NULL;
      free(load_request[i].element);
      load_request[i].element = NULL;
      if (code < 0) {
        if (printingEnabled)
          printWarning("load_parameters: input file ends unexpectedly.", load_request[i].filename);
        load_request[i].flags |= COMMAND_FLAG_IGNORE;
        continue;
      }
      if (printingEnabled) {
        printf("Error: problem reading data from load_parameters file %s\n", load_request[i].filename);
        fflush(stdout);
      }
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    allFilesRead = 0;
    SDDS_SetRowFlags(&load_request[i].table, 1);
    if ((rows = SDDS_CountRowsOfInterest(&load_request[i].table)) <= 0) {
      load_request[i].last_code = 0;
      load_request[i].flags |= COMMAND_FLAG_IGNORE;
      continue;
    }
    mode = NULL;
    if (!(element = (char **)SDDS_GetColumn(&load_request[i].table, Element_ColumnName)) ||
        !(parameter = (char **)SDDS_GetColumn(&load_request[i].table, Parameter_ColumnName)) ||
        (SDDS_GetColumnIndex(&load_request[i].table, Mode_ColumnName) >= 0 &&
         !(mode = (char **)SDDS_GetColumn(&load_request[i].table, Mode_ColumnName)))) {
      if (printingEnabled) {
        printf("Error: problem accessing data from load_parameters file %s\n", load_request[i].filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      parameter = element = NULL; /* suppress compiler warning */
      exitElegant(1);
    }
    type = NULL;
    if ((load_request[i].includeTypePattern || load_request[i].excludeTypePattern) &&
        !(type = (char **)SDDS_GetColumn(&load_request[i].table, ElementType_ColumnName))) {
      if (printingEnabled) {
        printf("Error: problem accessing data from load_parameters file %s\n", load_request[i].filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      parameter = element = NULL; /* suppress compiler warning */
      exitElegant(1);
    }
    valueString = NULL;
    value = NULL;
    if ((load_request[i].flags & NUMERICAL_DATA_PRESENT &&
         !(value = (double *)SDDS_GetColumn(&load_request[i].table, Value_ColumnName))) ||
        (load_request[i].flags & STRING_DATA_PRESENT &&
         !(valueString = (char **)SDDS_GetColumn(&load_request[i].table, ValueString_ColumnName)))) {
      if (printingEnabled) {
        printf("Error: problem accessing data from load_parameters file %s\n", load_request[i].filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      parameter = element = NULL; /* suppress compiler warning */
      exitElegant(1);
    }
    if (value && prefactor!=1)
      for (j=0; j<rows; j++)
        value[j] *= prefactor;
    
    occurence = NULL;
    if (!(load_request[i].flags & COMMAND_FLAG_IGNORE_OCCURENCE)) {
      if (verbose)
        printf("Using occurence data.\n");
      if (SDDS_GetColumnIndex(&load_request[i].table, Occurence_ColumnName) >= 0) {
        if (!(occurence = (int32_t *)SDDS_GetColumn(&load_request[i].table, Occurence_ColumnName))) {
          if (printingEnabled) {
            printf("Error: problem accessing data from load_parameters file %s\n", load_request[i].filename);
            fflush(stdout);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          }
          exitElegant(1);
        }
      }
    }

    if (load_request[i].repeat_first_page_at_each_step) {
      SDDS_Terminate(&load_request[i].table);
      SDDS_InitializeInputFromSearchPath(&load_request[i].table,
                                         load_request[i].filename);
    }

    load_request[i].values = 0;
    element_missing = 0;
    lastMissingElement[0] = 0;
    lastMissingOccurence = 0;
    for (j = 0; j < rows; j++) {
      char warningText[16384];
      
      /* If the user gives the use_first flag, then we load only the first instance for
       * any parameter.   If occurence data is present, then we load the first instance
       * for each occurence.  Otherwise, we load the first instance ignoring occurrence
       * (this would happen in change_defined_values mode).
       * If use_first is not given, then we load all values, which is the slowest option
       * but also the original default behavior.
       */
      eptr = NULL;
      if (occurence) {
        if (occurence[j] <= lastMissingOccurence &&
            strcmp(lastMissingElement, element[j]) == 0)
          continue;
      } else {
        if (strcmp(lastMissingElement, element[j]) == 0)
          continue;
      }
      if (load_request[i].includeNamePattern && !matchesPatternList(load_request[i].includeNamePattern, load_request[i].includeNamePatterns, element[j]))
        continue;
      if (load_request[i].includeItemPattern && !matchesPatternList(load_request[i].includeItemPattern, load_request[i].includeItemPatterns, parameter[j]))
        continue;
      if (load_request[i].includeTypePattern && !matchesPatternList(load_request[i].includeTypePattern, load_request[i].includeTypePatterns, type[j]))
        continue;

      if (load_request[i].excludeNamePattern && matchesPatternList(load_request[i].excludeNamePattern, load_request[i].excludeNamePatterns, element[j]))
        continue;
      if (load_request[i].excludeItemPattern && matchesPatternList(load_request[i].excludeItemPattern, load_request[i].excludeItemPatterns, parameter[j]))
        continue;
      if (load_request[i].excludeTypePattern && matchesPatternList(load_request[i].excludeTypePattern, load_request[i].excludeTypePatterns, type[j]))
        continue;

      element_missing = 0;

      /* If edit command given, change the name now (after matching) */
      if (load_request[i].editNameCommand) {
        char buffer[16384];
        strcpy(buffer, element[j]);
        edit_string(buffer, load_request[i].editNameCommand);
        free(element[j]);
        cp_str(element + j, buffer);
      }

      wildMatch = 0;
      if (has_wildcards(element[j])) {
        if (strchr(element[j], '-')) {
          char *tmp;
          tmp = element[j];
          element[j] = expand_ranges(element[j]);
          free(tmp);
        }
        if (!(eptr = wfind_element(element[j], &eptr, beamline->elem))) {
          if (load_request[i].flags & ALLOW_MISSING_ELEMENTS) {
            snprintf(warningText, 16384, "Unable to find match to element pattern %s, listed in file %s.",
                     element[j], load_request[i].filename);
            if (load_request[i].flags & ALLOW_MISSING_ELEMENTS) {
              if (printingEnabled)
                printWarning("load_parameters: Data provided for element that is missing from beamline.", warningText);
            } else
              bombElegant("Data provided for element that is missing from beamline.", warningText);
          }
        }
        wildMatch = 1;
      } else {
        if (!occurence) {
          strcpy(elem_param, element[j]);
          strcat(elem_param, parameter[j]);
        } else {
          sprintf(elem_param, "%s%s%" PRId32, element[j], parameter[j], occurence[j]);
        }
        inHash = !hadd(hash_table, (unsigned char *)elem_param, strlen(elem_param), NULL);
        /* Compare to hash table to see if we've already done this element/parameter */
        if (load_request[i].flags & COMMAND_FLAG_USE_FIRST && inHash)
          continue;
        
        /* if occurence is available, we can take advantage of hash table */
        if ((occurence && (!find_element_hash(element[j], occurence[j], &eptr, beamline->elem))) ||
            (!occurence && (!find_element_hash(element[j], 1, &eptr, beamline->elem)))) {
          if (occurence) {
            snprintf(warningText, 16384,
                     "Unable to find occurence %" PRId32 " of element %s, listed in file %s.",
                     occurence[j], element[j], load_request[i].filename);
            if (load_request[i].flags & ALLOW_MISSING_ELEMENTS) {
              if (printingEnabled)
                printWarning("load_parameters: Data provided for element occurence that is missing from beamline.", warningText);
            } else
              bombElegant("Data provided for element occurence that is missing from beamline.", warningText);
          } else {
            snprintf(warningText, 16384, "Unable to find element %s, listed in file %s.",
                     element[j], load_request[i].filename);
            if (load_request[i].flags & ALLOW_MISSING_ELEMENTS) {
              if (printingEnabled)
                printWarning("load_parameters: Data provided for element that is missing from beamline.", warningText);
            } else
              bombElegant("Data provided for element that is missing from beamline.", warningText);
          }
          element_missing = 1;
        }
      }
      if (element_missing) {
        if (occurence)
          lastMissingOccurence = occurence[j];
        strncpy(lastMissingElement, element[j], 99);
        continue;
      }
      lastMissingElement[0] = 0;
      lastMissingOccurence = 0;
      if ((param = confirm_parameter(parameter[j], eptr->type)) < 0) {
        char warningText[16384];
        snprintf(warningText, 16384,
                 "Element %s does not have a parameter %s (input file %s)",
                 eptr->name, parameter[j], load_request[i].filename);
        if (load_request[i].flags & ALLOW_MISSING_PARAMETERS) {
          if (printingEnabled)
            printWarning("load_parameters: Attempt to load parameter that element lacks.", warningText);
        } else
          bombElegantVA("Attempt to load parameter that element lacks: %s", warningText);
        continue;
      }
      if (entity_description[eptr->type].parameter[param].flags & PARAM_IS_LOCKED) {
        printWarningWithContext(entity_name[eptr->type],
                                entity_description[eptr->type].parameter[param].name,
                                "load_parameters: ignoring load_parameter attempt for locked parameter",
                                NULL);
        continue;
      }
      mode_flags = 0;
      if (mode)
        while ((ptr = get_token_t(mode[j], " \t,+"))) {
          long k;
          if ((k = match_string(ptr, load_mode, LOAD_MODES, UNIQUE_MATCH)) < 0) {
            if (printingEnabled) {
              printf("Error: unknown/ambiguous mode specifier %s (do_load_parameters)\nKnown specifiers are:\n",
                     ptr);
              fflush(stdout);
              for (k = 0; k < LOAD_MODES; k++)
                printf("    %s\n", load_mode[k]);
              fflush(stdout);
            }
            exitElegant(1);
          }
          mode_flags |= 1 << k;
          free(ptr);
        }
      if (mode_flags == 0)
        mode_flags = LOAD_FLAG_ABSOLUTE;
      if (mode_flags & LOAD_FLAG_IGNORE)
        continue;
      if (verbose && printingEnabled)
        printf("Working on row %ld of file\n", j);

      if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) {
        if (load_request[i].flags & NUMERICAL_DATA_PRESENT &&
            entity_description[eptr->type].parameter[param].type != IS_STRING)
          change_defined_parameter(element[j], param, eptr->type,
                                   value ? value[j] : 0, NULL, mode_flags + (verbose ? LOAD_FLAG_VERBOSE : 0));
        else
          change_defined_parameter(element[j], param, eptr->type,
                                   value ? value[j] : 0,
                                   valueString ? valueString[j] : NULL,
                                   mode_flags + (verbose ? LOAD_FLAG_VERBOSE : 0));
      }
      numberChanged = 0;
      do {
        numberChanged++;
        p_elem = eptr->p_elem;
        p_elem0 = eptr->p_elem0;
        load_request[i].reset_address = trealloc(load_request[i].reset_address,
                                                 sizeof(*load_request[i].reset_address) * (load_request[i].values + 1));
        load_request[i].value_type = trealloc(load_request[i].value_type,
                                              sizeof(*load_request[i].value_type) * (load_request[i].values + 1));
        load_request[i].element_flags = trealloc(load_request[i].element_flags,
                                                 sizeof(*load_request[i].element_flags) * (load_request[i].values + 1));
        load_request[i].starting_value = trealloc(load_request[i].starting_value,
                                                  sizeof(*load_request[i].starting_value) * (load_request[i].values + 1));
        load_request[i].use_start = trealloc(load_request[i].use_start,
                                             sizeof(*load_request[i].use_start) * (load_request[i].values + 1));
        load_request[i].element = trealloc(load_request[i].element,
                                           sizeof(*load_request[i].element) * (load_request[i].values + 1));
        load_request[i].reset_address[load_request[i].values] = ((double *)(p_elem + entity_description[eptr->type].parameter[param].offset));
        load_request[i].element[load_request[i].values] = eptr;
        load_request[i].use_start[load_request[i].values] = !inHash; /* don't use start value if it is already recorded */
        switch (entity_description[eptr->type].parameter[param].type) {
        case IS_DOUBLE:
          if (valueString && !value) {
            if (!sscanf(valueString[j], "%lf", &newValue)) {
              if (printingEnabled) {
                printf("Error: unable to scan double from \"%s\"\n", valueString[j]);
                fflush(stdout);
              }
              exitElegant(1);
            }
          } else {
            newValue = value[j];
          }
          if (eptr->divisions > 1 && (entity_description[eptr->type].parameter[param].flags & PARAM_DIVISION_RELATED))
            newValue /= eptr->divisions;
          if (verbose && printingEnabled) {
            printf("Changing %s.%s #%" PRId32 "  from %21.15e to ",
                   eptr->name,
                   entity_description[eptr->type].parameter[param].name,
                   occurence ? occurence[j] : numberChanged,
                   *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            fflush(stdout);
          }
          load_request[i].starting_value[load_request[i].values] = *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset));
          load_request[i].value_type[load_request[i].values] = IS_DOUBLE;
          if (mode_flags & LOAD_FLAG_ABSOLUTE) {
            *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)) = newValue;
            if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
              *((double *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) = newValue;

          } else if (mode_flags & LOAD_FLAG_DIFFERENTIAL) {
            *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)) += newValue;
            if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
              *((double *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) += newValue;
          } else if (mode_flags & LOAD_FLAG_FRACTIONAL) {
            *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
            if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
              *((double *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
          }
          if (verbose && printingEnabled) {
            printf("%21.15e (%21.15e)\n",
                   *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)),
                   *((double *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            fflush(stdout);
          }
          break;
        case IS_LONG:
        case IS_SHORT:
          if (valueString && !value) {
            if (!sscanf(valueString[j], "%lf", &newValue)) {
              if (printingEnabled) {
                printf("Error: unable to scan double from \"%s\"\n", valueString[j]);
                fflush(stdout);
              }
              exitElegant(1);
            }
          } else {
            newValue = value[j];
          }
          if (verbose && printingEnabled) {
            printf("Changing %s.%s #%" PRId32 " ",
                   eptr->name,
                   entity_description[eptr->type].parameter[param].name, numberChanged);
            if (entity_description[eptr->type].parameter[param].type == IS_LONG)
              printf("%ld  to ",
                     *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            else
              printf("%hd  to ",
                     *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            fflush(stdout);
          }
          if (entity_description[eptr->type].parameter[param].type == IS_LONG) {
            load_request[i].starting_value[load_request[i].values] = *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset));
            load_request[i].value_type[load_request[i].values] = IS_LONG;
            if (mode_flags & LOAD_FLAG_ABSOLUTE) {
              *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset)) =
                nearestInteger(newValue);
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((long *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) =
                  nearestInteger(newValue);
            } else if (mode_flags & LOAD_FLAG_DIFFERENTIAL) {
              *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset)) +=
                nearestInteger(newValue);
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((long *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) =
                  nearestInteger(newValue);
            } else if (mode_flags & LOAD_FLAG_FRACTIONAL) {
              *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((long *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
            }
          } else {
            load_request[i].starting_value[load_request[i].values] = *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset));
            load_request[i].value_type[load_request[i].values] = IS_SHORT;
            if (mode_flags & LOAD_FLAG_ABSOLUTE) {
              *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset)) =
                nearestInteger(newValue);
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((short *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) =
                  nearestInteger(newValue);
            } else if (mode_flags & LOAD_FLAG_DIFFERENTIAL) {
              *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset)) +=
                nearestInteger(newValue);
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((short *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) =
                  nearestInteger(newValue);
            } else if (mode_flags & LOAD_FLAG_FRACTIONAL) {
              *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
              if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS)
                *((short *)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) *= 1 + newValue;
            }
          }
          if (verbose && printingEnabled) {
            if (entity_description[eptr->type].parameter[param].type == IS_LONG)
              printf("%ld \n",
                     *((long *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            else
              printf("%hd \n",
                     *((short *)(p_elem + entity_description[eptr->type].parameter[param].offset)));
            fflush(stdout);
          }
          break;
        case IS_STRING:
          if (!valueString) {
            printf("Error: attempt to change %s.%s #%" PRId32 ", which is a string value, but no ParameterValueString column given in file %s\n",
                   eptr->name,
                   entity_description[eptr->type].parameter[param].name, numberChanged,
                   load_request[i].filename);
            exit(1);
          }
          load_request[i].value_type[load_request[i].values] = IS_STRING;
          if (verbose && printingEnabled) {
            printf("Changing %s.%s #%" PRId32 "  from %s to ",
                   eptr->name,
                   entity_description[eptr->type].parameter[param].name, numberChanged,
                   *((char **)(p_elem + entity_description[eptr->type].parameter[param].offset)) ? *((char **)(p_elem + entity_description[eptr->type].parameter[param].offset)) : "NULL");
            fflush(stdout);
          }
          if (strlen(valueString[j])) {
            if (!SDDS_CopyString((char **)(p_elem + entity_description[eptr->type].parameter[param].offset),
                                 valueString[j])) {
              if (printingEnabled) {
                printf("Error (do_load_parameters): unable to copy value string\n");
                fflush(stdout);
              }
              exitElegant(1);
            }
          } else
            *((char **)(p_elem + entity_description[eptr->type].parameter[param].offset)) = NULL;
          if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) {
            if (strlen(valueString[j])) {
              if (!SDDS_CopyString((char **)(p_elem0 + entity_description[eptr->type].parameter[param].offset),
                                   valueString[j])) {
                if (printingEnabled) {
                  printf("Error (do_load_parameters): unable to copy value string\n");
                  fflush(stdout);
                }
                exitElegant(1);
              }
            } else
              *((char **)(p_elem0 + entity_description[eptr->type].parameter[param].offset)) = NULL;
          }
          if (verbose && printingEnabled) {
            printf("%s\n",
                   *((char **)(p_elem + entity_description[eptr->type].parameter[param].offset)) ? *((char **)(p_elem + entity_description[eptr->type].parameter[param].offset)) : "NULL");
            fflush(stdout);
          }
          break;
        default:
          if (printingEnabled) {
            fprintf(stdout,
                    "Error: can't load parameter value for parameter %s of %s--not numeric parameter (do_load_parameters)\n",
                    parameter[j], element[j]);
            fflush(stdout);
          }
          break;
        }
        eptr->flags |=
          PARAMETERS_ARE_PERTURBED |
          ((entity_description[eptr->type].parameter[param].flags & PARAM_CHANGES_MATRIX) ? VMATRIX_IS_PERTURBED : 0);
        if ((eptr->flags & PARAMETERS_ARE_PERTURBED) && (entity_description[eptr->type].flags & HAS_MATRIX) && eptr->matrix) {
          free_matrices(eptr->matrix);
          free(eptr->matrix);
          eptr->matrix = NULL;
        }
        load_request[i].element_flags[load_request[i].values] = eptr->flags;
        load_request[i].values++;
        if (!wildMatch) {
          if (occurence || !find_element_hash(element[j], numberChanged + 1, &eptr, beamline->elem))
            break;
          /*
          if (!(!occurence && find_element_hash(element[j], numberChanged + 1, &eptr, beamline->elem)))
            break;
          */
        } else {
          if (!(eptr=wfind_element(element[j], &eptr, beamline->elem)))
            break;
        }
      } while (eptr);
      totalNumberChanged += numberChanged;
    }

    for (j=0; j<rows; j++) {
      free(element[j]);
      element[j] = NULL;
      free(parameter[j]);
      parameter[j] = NULL;
      if (mode) {
        free(mode[j]);
        mode[j] = NULL;
      }
      if (valueString && valueString[j]) {
        free(valueString[j]);
        valueString[j] = NULL;
      }
    }
    free(element);
    free(parameter);
    if (mode)
      free(mode);
    if (value)
      free(value);
    if (valueString)
      free(valueString);
    if (occurence)
      free(occurence);
    element = parameter = mode = valueString = NULL;
    value = NULL;
    occurence = NULL;
    if (load_request[i].flags & COMMAND_FLAG_CHANGE_DEFINITIONS) {
      free(load_request[i].reset_address);
      load_request[i].reset_address = NULL;
      free(load_request[i].value_type);
      load_request[i].value_type = NULL;
      free(load_request[i].element_flags);
      load_request[i].element_flags = NULL;
      free(load_request[i].starting_value);
      load_request[i].starting_value = NULL;
      free(load_request[i].element);
      load_request[i].element = NULL;
      load_request[i].flags |= COMMAND_FLAG_IGNORE; /* ignore hereafter */
    }

    if (hash_table) {
      hdestroy(hash_table); /* destroy hash table */
      hash_table = NULL;
    }
  }

  if (!allFilesRead || allFilesIgnored) {
    compute_end_positions(beamline);
    if (printingEnabled)
      printf("%" PRId32 "  parameter values loaded\n", totalNumberChanged);
    return PARAMETERS_LOADED;
  }
  return PARAMETERS_ENDED;
}

void finish_load_parameters() {
  long i;
  for (i = 0; i < load_requests; i++) {
    if (!SDDS_Terminate(&load_request[i].table)) {
      if (printingEnabled) {
        printf("Error: unable to terminate load_parameters SDDS file %s\n", load_request[i].filename);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      exitElegant(1);
    }
    free(load_request[i].filename);
  }
  if (load_requests)
    free(load_request);
  load_request = NULL;
  load_requests = 0;
}

static long dumpingLatticeParameters = 0;
static SDDS_DATASET SDDS_dumpLattice;
void dumpLatticeParameters(char *filename, RUN *run, LINE_LIST *beamline, long suppressDefaults) {
  SDDS_DATASET *SDDSout;
  long iElem, iParam;
  ELEMENT_LIST *eptr;
  PARAMETER *parameter;
  long row, maxRows, doSave;
  double value = 0.0;
  char *string_value = NULL;
  static long iElementName, iElementParameter, iParameterValue, iElementType, iOccurence, iElementGroup, iParameterValueString;

  if (suppressDefaults) {
    printWarning("Use of suppress_parameter_defaults (run_setup) can lead to unexpected results.",
                 "Problems may particularly appear in conjunction with alter_elements or load_parameters.");
  }

  SDDSout = &SDDS_dumpLattice;
  if (!dumpingLatticeParameters) {
    if (!SDDS_InitializeOutputElegant(SDDSout, SDDS_BINARY, 0, NULL, NULL, filename) ||
        (iElementName = SDDS_DefineColumn(SDDSout, Element_ColumnName, NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementParameter = SDDS_DefineColumn(SDDSout, Parameter_ColumnName, NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iParameterValue = SDDS_DefineColumn(SDDSout, Value_ColumnName, NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iParameterValueString = SDDS_DefineColumn(SDDSout, ValueString_ColumnName, NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementType = SDDS_DefineColumn(SDDSout, ElementType_ColumnName, NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iOccurence = SDDS_DefineColumn(SDDSout, Occurence_ColumnName, NULL, NULL, NULL, NULL, SDDS_LONG, 0)) < 0 ||
        (iElementGroup = SDDS_DefineColumn(SDDSout, "ElementGroup", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        !SDDS_WriteLayout(SDDSout)) {
      printf("Problem setting up parameter output file\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    dumpingLatticeParameters = 1;
  }

  if (!SDDS_StartPage(SDDSout, maxRows = beamline->n_elems * 10))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  row = 0;
  printf("Saving lattice parameters to %s...", filename);
  fflush(stdout);

  eptr = beamline->elem;
  for (iElem = 0; iElem < beamline->n_elems; iElem++) {
    /*
      printf("name=%s, divisions=%" PRId32 "   first=%hd\n",
      eptr->name, eptr->divisions, eptr->firstOfDivGroup);
      */
    if (eptr->divisions > 1 && !eptr->firstOfDivGroup) {
      /* don't emit data for every member of a divided group */
      eptr = eptr->succ;
      continue;
    }
    if (!(parameter = entity_description[eptr->type].parameter))
      SDDS_Bomb("parameter entry is NULL for entity description (dumpLatticeParameters)");
    if (!(eptr->name))
      SDDS_Bomb("element name is NULL (dumpLatticeParameters)");
    for (iParam = 0; iParam < entity_description[eptr->type].n_params; iParam++) {
      value = 0;
      string_value = NULL;
      doSave = 1;
      if (parameter[iParam].flags & PARAM_IS_ALIAS)
        continue;
      switch (parameter[iParam].type) {
      case IS_DOUBLE:
        value = *(double *)(eptr->p_elem + parameter[iParam].offset);
        if (parameter[iParam].flags & PARAM_DIVISION_RELATED &&
            eptr->divisions > 1)
          value *= eptr->divisions;
        if (parameter[iParam].flags & HAS_LENGTH && run->backtrack && iParam == 0)
          value *= -1; /* don't want to save internal negative length values */
        if (suppressDefaults && value == parameter[iParam].number)
          doSave = 0;
        if (!doSave) {
          double refValue;
          refValue = *(double *)(eptr->p_elem0 + parameter[iParam].offset);
          if (parameter[iParam].flags & PARAM_DIVISION_RELATED &&
              eptr->divisions > 1)
            refValue *= eptr->divisions;
          if (parameter[iParam].flags & HAS_LENGTH && run->backtrack && iParam == 0)
            refValue *= -1; /* don't want to save internal negative length values */
          if (value != refValue)
            doSave = 1;
        }
        break;
      case IS_LONG:
        value = *(long *)(eptr->p_elem + parameter[iParam].offset);
        if (suppressDefaults && value == parameter[iParam].integer)
          doSave = 0;
        if (!doSave) {
          double refValue;
          refValue = *(long *)(eptr->p_elem0 + parameter[iParam].offset);
          if (value != refValue)
            doSave = 1;
        }
        break;
      case IS_SHORT:
        value = *(short *)(eptr->p_elem + parameter[iParam].offset);
        if (suppressDefaults && value == parameter[iParam].integer)
          doSave = 0;
        if (!doSave) {
          double refValue;
          refValue = *(short *)(eptr->p_elem0 + parameter[iParam].offset);
          if (value != refValue)
            doSave = 0;
        }
        break;
      case IS_STRING:
        string_value = *(char **)(eptr->p_elem + parameter[iParam].offset);
        if (suppressDefaults) {
          if (!string_value) {
            if (!parameter[iParam].string)
              doSave = 0;
          } else {
            if (parameter[iParam].string && strcmp(string_value, parameter[iParam].string) == 0)
              doSave = 0;
          }
        }
        if (!doSave) {
          char *ref_value;
          ref_value = *(char **)(eptr->p_elem0 + parameter[iParam].offset);
          if ((!string_value && ref_value) || (string_value && !ref_value) || (ref_value && string_value && strcmp(string_value, ref_value) != 0))
            doSave = 1;
        }
        break;
      default:
        value = 0;
        string_value = NULL;
        doSave = 0;
        break;
      }
      /* some kludges to avoid saving things that shouldn't be saved as the user didn't
         set them in the first place
         */
      if (strcmp(parameter[iParam].name, "PHASE_REFERENCE") == 0 && value > LONG_MAX / 2) {
        doSave = 0;
      }

      if (doSave) {
        if (!(parameter[iParam].name))
          SDDS_Bomb("parameter name is NULL (dumpLatticeParameters)");
        if (row >= maxRows) {
          if (!SDDS_LengthenTable(SDDSout, 1000))
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          maxRows += 1000;
        }
        if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                               iOccurence, eptr->occurence,
                               iElementName, eptr->name,
                               iElementParameter, parameter[iParam].name,
                               iParameterValue, value,
                               iParameterValueString, string_value ? string_value : "",
                               iElementType, entity_name[eptr->type],
                               iElementGroup, eptr->group ? eptr->group : "",
                               -1)) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        row++;
      }
    }
    eptr = eptr->succ;
  }
  if (!SDDS_WritePage(SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDSout);
  printf("done.\n");
  fflush(stdout);
}

void finishLatticeParametersFile() {
  if (dumpingLatticeParameters && !SDDS_Terminate(&SDDS_dumpLattice))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  dumpingLatticeParameters = 0;
}

static long dumpingRfcData = 0;
static SDDS_DATASET SDDS_rfcData;

void dumpRfcReferenceData(char *filename, RUN *run, LINE_LIST *beamline) {
  SDDS_DATASET *SDDSout;
  long iElem, row, maxRows;
  ELEMENT_LIST *eptr;
  RFCA *rfca;
  static long iElementName, iElementParameter, iParameterValue, iElementType, iOccurence;

  SDDSout = &SDDS_rfcData;
  if (!dumpingRfcData) {
    if (!SDDS_InitializeOutputElegant(SDDSout, SDDS_BINARY, 0, NULL, NULL, filename) ||
        (iElementName = SDDS_DefineColumn(SDDSout, "ElementName", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iElementParameter = SDDS_DefineColumn(SDDSout, "ElementParameter", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iParameterValue = SDDS_DefineColumn(SDDSout, "ParameterValue", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
        (iElementType = SDDS_DefineColumn(SDDSout, "ElementType", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
        (iOccurence = SDDS_DefineColumn(SDDSout, "ElementOccurence", NULL, NULL, NULL, NULL, SDDS_LONG, 0)) < 0 ||
        !SDDS_WriteLayout(SDDSout)) {
      printf("Problem setting up parameter output file got RFCA and RFCW reference\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    dumpingRfcData = 1;
  }

  if (!SDDS_StartPage(SDDSout, maxRows = beamline->n_elems * 10))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  row = 0;
  printf("Saving RFCA/RFCW timing parameters to %s...", filename);
  fflush(stdout);

  eptr = beamline->elem;
  for (iElem = 0; iElem < beamline->n_elems; iElem++) {
    if (eptr->type == T_RFCW)
      rfca = &(((RFCW *)(eptr->p_elem))->rfca);
    else if (eptr->type == T_RFCA)
      rfca = ((RFCA *)(eptr->p_elem));
    else {
      eptr = eptr->succ;
      continue;
    }
    if (!(eptr->name))
      SDDS_Bomb("element name is NULL (dumpRfcReferenceData)");
    if (row >= maxRows) {
      if (!SDDS_LengthenTable(SDDSout, 100))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      maxRows += 100;
    }
    if (!SDDS_SetRowValues(SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                           iOccurence, eptr->occurence,
                           iElementName, eptr->name,
                           iElementParameter, "T_REFERENCE",
                           iParameterValue, rfca->t_fiducial,
                           iElementType, entity_name[eptr->type],
                           -1)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    row++;
    eptr = eptr->succ;
  }
  if (!SDDS_WritePage(SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDSout);
}

void finishRfcDataFile() {
  if (dumpingRfcData && !SDDS_Terminate(&SDDS_rfcData))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  dumpingRfcData = 0;
}
