/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: optimize.c
 *
 * Michael Borland, 1991
 */
#include "mdb.h"
#include "track.h"
#include "SDDS.h"

#include "optimize.h"
#include "match_string.h"
#include "chromDefs.h"
#include "tuneDefs.h"
#include "correctDefs.h"

#define COMPARE_PARTICLE_SUM_ABSDEV 0x0001UL
#define COMPARE_PARTICLE_MAX_ABSDEV 0x0002UL
#define COMPARE_PARTICLE_SUM_SQR 0x0004UL
#define N_PARTICLE_COMPARISON_MODES 3
static char *particleDeviationComparisonMode[N_PARTICLE_COMPARISON_MODES] = {
  "sum-ad",
  "max-ad",
  "sum-sqr",
};

#define STATISTIC_SUM_ABS 0
#define STATISTIC_MAXIMUM 1
#define STATISTIC_MINIMUM 2
#define STATISTIC_SUM_SQR 3
#define STATISTIC_SUM 4
#define N_OPTIM_STATS 5
static char *optimize_statistic[N_OPTIM_STATS] = {
  "sum-absolute-value", "maximum", "minimum", "sum-squares", "plain-sum"};

static long stopOptimization = 0;
long checkForOptimRecord(double *value, long values, long *again);
void storeOptimRecord(double *value, long values, long invalid, double result);
void rpnStoreHigherMatrixElements(VMATRIX *M, long **TijkMem, long **UijklMem, long maxOrder);
double particleComparisonForOptimization(BEAM *beam, OPTIMIZATION_DATA *optimData, long *invalid);

void initializeOptimizationStatistics(double *sum, double *sumAbs, double *sum2, double *min, double *max);
void updateOptimizationStatistics(double *sum, double *sumAbs, double *sum2, double *min, double *max, double value);
double chooseOptimizationStatistic(double sum, double sumAbs, double sum2, double min, double max, long stat);

#if USE_MPI
/* Find the global minimal value and its location across all the processors */
void find_global_min_index(double *min, int *processor_ID, MPI_Comm comm);
/* Print out the best individuals for all the processors */
void SDDS_PrintPopulations(SDDS_TABLE *popLogPtr, double result, double *variable, long dimensions);
/* Print statistics after each iteration */
void SDDS_PrintStatistics(SDDS_TABLE *popLogPtr, long iteration, double best_value, double worst_value, double median, double avarage, double spread, double *variable, long n_variables, double *covariable, long n_covariables, long print_all);
FILE *fpSimplexLog = NULL;
static long simplexLogStep = 0;
static long simplexComparisonStep = 0;
static short targetReached = 0;
void checkTarget(double myResult, long invalid);
#endif

static time_t interrupt_file_mtime = 0;
static double interrupt_file_check_time = 0; /* last time the interrupt file was checked */
static short interrupt_in_progress = 0;
time_t get_mtime(char *filename) {
  struct stat statbuf;
  if (stat(filename, &statbuf) == -1)
    return 0;
  return statbuf.st_mtime;
}

#if !USE_MPI
long myid = 0;
#endif

void do_optimization_setup(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  log_entry("do_optimization_setup");

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&optimization_setup, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimization_setup);

  /* check validity of input values, and copy into structure */
  if ((optimization_data->mode = match_string(mode, optimize_mode, N_OPTIM_MODES, EXACT_MATCH)) < 0)
    bombElegant("unknown optimization mode", NULL);
  optimization_data->equation = equation;
  if ((optimization_data->method = match_string(method, optimize_method, N_OPTIM_METHODS, EXACT_MATCH)) < 0)
    bombElegant("unknown optimization method", NULL);
  if (statistic == NULL ||
      (optimization_data->statistic = match_string(statistic, optimize_statistic, N_OPTIM_STATS, EXACT_MATCH)) < 0) {
    long i;
    fprintf(stderr, "Unknown optimization statistic: %s.  Known values are ", statistic);
    for (i = 0; i < N_OPTIM_STATS; i++)
      fprintf(stderr, "\"%s\"%s", optimize_statistic[i], i == (N_OPTIM_STATS - 1) ? "\n" : ", ");
    exit(1);
  }
  if ((optimization_data->tolerance = tolerance) == 0)
    bombElegant("tolerance == 0", NULL);
  if ((optimization_data->n_passes = n_passes) <= 0)
    bombElegant("n_passes <= 0", NULL);
  if ((optimization_data->n_evaluations = n_evaluations) <= 0)
    bombElegant("n_evaluations <= 0", NULL);
  if ((optimization_data->n_restarts = n_restarts) < 0)
    bombElegant("n_restarts < 0", NULL);
  if ((optimization_data->restart_reset_threshold = restart_reset_threshold)<0)
    bombElegant("restart_reset_threshold < 0", NULL);
  if ((optimization_data->matrix_order = matrix_order) < 1 ||
      matrix_order > 3)
    bombElegant("matrix_order must be 1, 2, or 3", NULL);
  optimization_data->soft_failure = soft_failure;
  if (output_sparsing_factor <= 0)
    output_sparsing_factor = 1;
#if USE_MPI
  if (!writePermitted)
    log_file = NULL;
  optimization_data->random_factor = random_factor;
  /* The output files are disabled for most of the optimization methods in Pelegant except for the simplex method, which runs in serial mode */
  if (optimization_data->method == OPTIM_METHOD_SIMPLEX)
    enableOutput = 1;
  if ((optimization_data->hybrid_simplex_tolerance = hybrid_simplex_tolerance) == 0)
    bombElegant("hybrid_simplex_tolerance == 0", NULL);
  if ((optimization_data->hybrid_simplex_tolerance_count = hybrid_simplex_tolerance_count) <= 0)
    bombElegant("hybrid_simplex_tolerance_count <= 0", NULL);
  optimization_data->hybrid_simplex_comparison_interval = hybrid_simplex_comparison_interval;
#endif
  if (log_file) {
    if (str_in(log_file, "%s"))
      log_file = compose_filename(log_file, run->rootname);
    if (strcmp(log_file, "/dev/tty") == 0 || strcmp(log_file, "tt:") == 0)
      optimization_data->fp_log = stdout;
    else if ((optimization_data->fp_log = fopen_e(log_file, "w", FOPEN_RETURN_ON_ERROR)) == NULL)
      bombElegant("unable to open log file", NULL);
  }
  if (term_log_file) {
    if (str_in(term_log_file, "%s"))
      term_log_file = compose_filename(term_log_file, run->rootname);
  }
  optimization_data->verbose = verbose;
  if (optimization_data->mode == OPTIM_MODE_MAXIMUM && target != -DBL_MAX)
    target = -target;
  optimization_data->target = target;
  optimization_data->simplexPassRangeFactor = simplex_pass_range_factor;
  optimization_data->simplexDivisor = simplex_divisor;
  optimization_data->includeSimplex1dScans = include_simplex_1d_scans;
  optimization_data->startFromSimplexVertex1 = start_from_simplex_vertex1;
  optimization_data->rcdsStepFactor = rcds_step_factor;
  if ((optimization_data->restart_worst_term_factor = restart_worst_term_factor) <= 0)
    bombElegant("restart_worst_term_factor <= 0", NULL);
  if ((optimization_data->restart_worst_terms = restart_worst_terms) <= 0)
    bombElegant("restart_worst_terms <= 0", NULL);
  if (interrupt_file && strlen(interrupt_file)) {
    if (str_in(interrupt_file, "%s"))
      interrupt_file = compose_filename(interrupt_file, run->rootname);
    if (fexists(interrupt_file)) {
      interrupt_file_mtime = get_mtime(interrupt_file);
    } else {
      interrupt_file_mtime = 0;
    }
  } else {
    interrupt_file = NULL;
    interrupt_file_mtime = 0;
  }

  /* reset flags for elements that may have been varied previously */
  if (optimization_data->variables.n_variables)
    set_element_flags(beamline, optimization_data->variables.element, NULL,
                      optimization_data->variables.varied_type,
                      NULL, optimization_data->variables.n_variables,
                      PARAMETERS_ARE_STATIC, 0, 1, 0);

  /* initialize other elements of the structure */
  optimization_data->new_data_read = 0;
  optimization_data->balance_terms = balance_terms;
  optimization_data->variables.n_variables = 0;
  optimization_data->covariables.n_covariables = 0;
  optimization_data->constraints.n_constraints = 0;
  optimization_data->TijkMem = NULL;
  optimization_data->UijklMem = NULL;
  log_exit("do_optimization_setup");
}

#if USE_MPI
void do_parallel_optimization_setup(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  do_optimization_setup(optimization_data, nltext, run, beamline);

  parallelTrackingBasedMatrices = 0;

  if (optimization_data->method != OPTIM_METHOD_SIMPLEX)
    runInSinglePartMode = 1; /* All the processors will track the same particles with different parameters */

  if (optimization_data->method == OPTIM_METHOD_SWARM || optimization_data->method == OPTIM_METHOD_GENETIC) {
    if (population_size < n_processors) {
      char warningBuffer[1024];
      snprintf(warningBuffer, 1024,
               "The population size will be increased to %d.", n_processors);
      printWarning("parallel_optimization_setup: The population is less than the number of processors.",
                   warningBuffer);
      population_size = n_processors;
    }
  }

  if (optimization_data->method == OPTIM_METHOD_SWARM) /* The n_restarts is used to control n_iterations for particle swarm optimization */
  optimization_data->n_restarts = n_iterations - 1;
  optimization_data->n_iterations = n_iterations;
  optimization_data->max_no_change = max_no_change;
  optimization_data->population_size = population_size;
  optimization_data->print_all_individuals = print_all_individuals;
  if (population_log && strlen(population_log)) {
    if (str_in(population_log, "%s"))
      population_log = compose_filename(population_log, run->rootname);
  } else {
    population_log = NULL;
  }
  if (interrupt_file && strlen(interrupt_file)) {
    if (str_in(interrupt_file, "%s"))
      interrupt_file = compose_filename(interrupt_file, run->rootname);
    if (fexists(interrupt_file)) {
      interrupt_file_mtime = get_mtime(interrupt_file);
    } else {
      interrupt_file_mtime = 0;
    }
  } else {
    interrupt_file = NULL;
    interrupt_file_mtime = 0;
  }

  if (simplex_log)
    simplex_log = compose_filename(simplex_log, run->rootname);

  if (optimization_data->method == OPTIM_METHOD_GENETIC)
    /* The crossover type defined in PGAPACK started from 1, instead of 0. */
    if ((optimization_data->crossover_type = (match_string(crossover, crossover_type, N_CROSSOVER_TYPES, EXACT_MATCH) + 1)) < 1)
      bombElegant("unknown genecic optimization crossover type ", NULL);
}
#endif

void add_optimization_variable(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  long n_variables;
  OPTIM_VARIABLES *variables;
  ELEMENT_LIST *context;
  /* these are used to append a dummy name to the variables list for use with final parameters output: */
  static char *extra_name[3] = {"optimized", "optimizationFunction", "bestOptimizationFunction"};
  static char *extra_unit[3] = {"", "", ""};
  char warningBuffer[1024];
  long i, extras = 3;
  char *ptr;

  log_entry("add_optimization_variable");

  /* process namelist text */
  /* can't use automatic defaults, because of DBL_MAX being a nonconstant object */
  set_namelist_processing_flags(0);
  set_print_namelist_flags(0);
  name = item = NULL;
  step_size = 1;
  fractional_step_size = -1;
  lower_limit = -(upper_limit = DBL_MAX);
  differential_limits = 0;
  force_inside = 0;
  no_element = 0;
  initial_value = 0;
  if (processNamelist(&optimization_variable, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimization_variable);

  if (disable)
    return;

  if ((n_variables = optimization_data->variables.n_variables) == 0) {
    if (optimization_data->new_data_read)
      bombElegant("improper sequencing of variation and tracking", NULL);
    optimization_data->new_data_read = 1;
  }

  variables = &(optimization_data->variables);
  variables->element = trealloc(variables->element, sizeof(*variables->element) * (n_variables + extras + 1));
  variables->item = trealloc(variables->item, sizeof(*variables->item) * (n_variables + extras + 1));
  variables->lower_limit = trealloc(variables->lower_limit, sizeof(*variables->lower_limit) * (n_variables + extras + 1));
  variables->upper_limit = trealloc(variables->upper_limit, sizeof(*variables->upper_limit) * (n_variables + extras + 1));
  variables->step = trealloc(variables->step, sizeof(*variables->step) * (n_variables + extras + 1));
  variables->orig_step = trealloc(variables->orig_step, sizeof(*variables->orig_step) * (n_variables + extras + 1));
  variables->varied_quan_name = trealloc(variables->varied_quan_name, sizeof(*variables->varied_quan_name) * (n_variables + extras + 1));
  variables->varied_quan_unit = trealloc(variables->varied_quan_unit, sizeof(*variables->varied_quan_unit) * (n_variables + extras + 1));
  variables->varied_type = trealloc(variables->varied_type, sizeof(*variables->varied_type) * (n_variables + extras + 1));
  variables->varied_param = trealloc(variables->varied_param, sizeof(*variables->varied_param) * (n_variables + extras + 1));
  variables->varied_quan_value = trealloc(variables->varied_quan_value, sizeof(*variables->varied_quan_value) * (n_variables + extras + 1));
  variables->initial_value = trealloc(variables->initial_value, sizeof(*variables->initial_value) * (n_variables + extras + 1));
  variables->memory_number = trealloc(variables->memory_number, sizeof(*variables->memory_number) * (n_variables + extras + 1));
  variables->force_inside = trealloc(variables->force_inside, sizeof(*variables->force_inside) * (n_variables + extras + 1));

  /* check for valid input */
  if (name == NULL)
    bombElegant("element name missing in optimization_variable namelist", NULL);
  str_toupper(name);
  str_toupper(item);
  if (!no_element) {
    context = NULL;
    if (!find_element(name, &context, beamline->elem)) {
      printf("error: cannot vary element %s--not in beamline\n", name);
      fflush(stdout);
      exitElegant(1);
    }
    cp_str(&variables->element[n_variables], name);
    variables->varied_type[n_variables] = context->type;
    if (item == NULL)
      bombElegant("item name missing in optimization_variable list", NULL);

    if ((variables->varied_param[n_variables] = confirm_parameter(item, context->type)) < 0) {
      printf("error: cannot vary %s--no such parameter for %s\n", item, name);
      fflush(stdout);
      exitElegant(1);
    }
    if (entity_description[context->type].parameter[variables->varied_param[n_variables]].flags & PARAM_IS_LOCKED)
      bombElegantVA("Error: parameter %s of %s cannot be changed via optimization_variable\n",
                    entity_description[context->type].parameter[variables->varied_param[n_variables]].name,
                    entity_name[context->type]);

    cp_str(&variables->item[n_variables], item);
    cp_str(&variables->varied_quan_unit[n_variables],
           entity_description[context->type].parameter[variables->varied_param[n_variables]].unit);
    if (!get_parameter_value(variables->varied_quan_value + n_variables, name, variables->varied_param[n_variables],
                             context->type, beamline))
      bombElegant("unable to get initial value for parameter", NULL);
    if (differential_limits) {
      lower_limit += variables->varied_quan_value[n_variables];
      upper_limit += variables->varied_quan_value[n_variables];
    }
  } else {
    variables->varied_type[n_variables] = T_FREEVAR;
    cp_str(&variables->element[n_variables], name);
    cp_str(&variables->item[n_variables], item);
    cp_str(&variables->varied_quan_unit[n_variables], "");
    variables->varied_quan_value[n_variables] = initial_value;
  }

  if (lower_limit >= upper_limit)
    bombElegant("lower_limit >= upper_limit", NULL);

  variables->initial_value[n_variables] = variables->varied_quan_value[n_variables];
  variables->force_inside[n_variables] = force_inside;
  if (variables->initial_value[n_variables] > upper_limit) {
    if (force_inside) {
      snprintf(warningBuffer, 1024,
               "Initial value (%e) is greater than the upper limit (%le) for %s.%s. Set to upper limit.",
               variables->initial_value[n_variables], upper_limit,
               name, item);
      printWarning("optimization_variable: Initial value is greater than the upper limit.", warningBuffer);
      variables->initial_value[n_variables] = variables->varied_quan_value[n_variables] = upper_limit;
    } else {
      printf("Error: Initial value (%e) is greater than upper limit.\n", variables->initial_value[n_variables]);
      exit(1);
    }
  }
  if (variables->initial_value[n_variables] < lower_limit) {
    if (force_inside) {
      snprintf(warningBuffer, 1024,
               "Initial value (%e) is smaller than the lower limit (%le) for %s.%s. Set to lower limit.",
               variables->initial_value[n_variables], lower_limit,
               name, item);
      printWarning("optimization_variable: Initial value is smaller than the lower limit.", warningBuffer);
      variables->initial_value[n_variables] = variables->varied_quan_value[n_variables] = lower_limit;
    } else {
      printf("Error: Initial value (%e) is smaller than lower limit.\n", variables->initial_value[n_variables]);
      exit(1);
    }
  }
  if (fractional_step_size<=0) 
    variables->orig_step[n_variables] = step_size;
  else
    variables->orig_step[n_variables] = (upper_limit - lower_limit)*fractional_step_size;
  variables->varied_quan_name[n_variables] = tmalloc(sizeof(char) * (strlen(name) + strlen(item) + 3));
  sprintf(variables->varied_quan_name[n_variables], "%s.%s", name, item);
  variables->lower_limit[n_variables] = lower_limit;
  variables->upper_limit[n_variables] = upper_limit;
  rpn_store(variables->initial_value[n_variables], NULL,
            variables->memory_number[n_variables] =
              rpn_create_mem(variables->varied_quan_name[n_variables], 0));

  ptr = tmalloc(sizeof(char) * (strlen(name) + strlen(item) + 4));
  sprintf(ptr, "%s.%s0", name, item);
  rpn_store(variables->initial_value[n_variables], NULL, rpn_create_mem(ptr, 0));
  free(ptr);

  for (i = 0; i < extras; i++) {
    variables->varied_quan_name[n_variables + i + 1] = extra_name[i];
    variables->varied_quan_unit[n_variables + i + 1] = extra_unit[i];
  }

  optimization_data->variables.n_variables += 1;
  log_exit("add_optimization_variable");
}

void add_optimization_term(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run,
                           LINE_LIST *beamline) {
  long field_value = 0, n_field_values = 0, index, nFileTerms = 0;
  char s[16834], value[100];
  char **fileTerm = NULL;

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&optimization_term, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimization_term);

  if (optimization_term_struct.weight == 0)
    return;

  if (optimization_term_struct.term == NULL && optimization_term_struct.input_file == NULL)
    bombElegant("term is invalid and no input file given", NULL);
  if (optimization_data->equation)
    bombElegant("you've already given an optimization equation, so you can't give individual terms", NULL);
  if (optimization_term_struct.field_string && optimization_term_struct.input_file)
    bombElegant("you can't use field substitution and file input together", NULL);
  if (optimization_term_struct.field_string) {
    if ((n_field_values = (optimization_term_struct.field_final_value - optimization_term_struct.field_initial_value) / optimization_term_struct.field_interval + 1) <= 0)
      bombElegant("something strage about field_final_value, field_initial_value, and field_interval", NULL);
    field_value = optimization_term_struct.field_initial_value;
  } else if (optimization_term_struct.input_file) {
    SDDS_DATASET SDDSin;
    if (!optimization_term_struct.input_column)
      bombElegant("you must give input_column when giving input_file", NULL);
    if (!SDDS_InitializeInputFromSearchPath(&SDDSin, optimization_term_struct.input_file) ||
        SDDS_ReadPage(&SDDSin) != 1)
      SDDS_Bomb("problem reading optimization term input file");
    if ((nFileTerms = SDDS_RowCount(&SDDSin)) == 0) {
      printWarning("optimization_term: No rows loaded from optimization term file.", NULL);
      return;
    }
    if (SDDS_GetNamedColumnType(&SDDSin, optimization_term_struct.input_column) != SDDS_STRING) {
      printf("Error: column %s is nonexistent or not string type.\n", optimization_term_struct.input_column);
      exitElegant(1);
    }
    if (!(fileTerm = SDDS_GetColumn(&SDDSin, optimization_term_struct.input_column)))
      SDDS_Bomb("error getting optimization term column from file");
    SDDS_Terminate(&SDDSin);
  } else {
    optimization_term_struct.field_initial_value = optimization_term_struct.field_final_value = 0;
    optimization_term_struct.field_interval = 1;
    n_field_values = 1;
    field_value = 0;
    nFileTerms = 0;
  }

  if (!(optimization_data->term = SDDS_Realloc(optimization_data->term,
                                               sizeof(*optimization_data->term) * (optimization_data->terms + n_field_values + nFileTerms))) ||
      !(optimization_data->termValue = SDDS_Realloc(optimization_data->termValue,
                                                    sizeof(*optimization_data->termValue) * (optimization_data->terms + n_field_values + nFileTerms))) ||
      !(optimization_data->termWeight = SDDS_Realloc(optimization_data->termWeight,
                                                     sizeof(*optimization_data->termWeight) * (optimization_data->terms + n_field_values + nFileTerms))) ||
      !(optimization_data->usersTermWeight = SDDS_Realloc(optimization_data->usersTermWeight,
                                                          sizeof(*optimization_data->usersTermWeight) * (optimization_data->terms + n_field_values + nFileTerms))))
    bombElegant("memory allocation failure", NULL);

  if (!nFileTerms) {
    for (index = 0; index < n_field_values; index++) {
      if (optimization_term_struct.field_string && strlen(optimization_term_struct.field_string)) {
        sprintf(value, "%ld", field_value);
        replaceString(s, optimization_term_struct.term, optimization_term_struct.field_string, value, -1, 0);
        field_value += optimization_term_struct.field_interval;
      } else
        strcpy(s, optimization_term_struct.term);
      if (!SDDS_CopyString(&optimization_data->term[optimization_data->terms + index], s))
        bombElegant("memory allocation failure", NULL);
      optimization_data->termWeight[optimization_data->terms + index] = 1;
      optimization_data->usersTermWeight[optimization_data->terms + index] = optimization_term_struct.weight;
      if (optimization_term_struct.verbose)
        printf("Added term: %s\n", optimization_data->term[optimization_data->terms + index]);
    }
    optimization_data->terms += n_field_values;
  } else {
    for (index = 0; index < nFileTerms; index++) {
      optimization_data->term[optimization_data->terms + index] = fileTerm[index];
      optimization_data->termWeight[optimization_data->terms + index] = 1;
      optimization_data->usersTermWeight[optimization_data->terms + index] = optimization_term_struct.weight;
      if (optimization_term_struct.verbose)
        printf("Added term: %s\n", optimization_data->term[optimization_data->terms + index]);
    }
    optimization_data->terms += nFileTerms;
    free(fileTerm);
  }
}

void add_optimization_covariable(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
#include "optim_covariable.h"
  long n_covariables;
  OPTIM_COVARIABLES *covariables;
  ELEMENT_LIST *context;
  char nameBuffer[100], *ptr;
  static long nameIndex = 0;

  log_entry("add_optimization_covariable");

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&optimization_covariable, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimization_covariable);
  if (disable)
    return;

  n_covariables = optimization_data->covariables.n_covariables;

  covariables = &(optimization_data->covariables);
  covariables->element = trealloc(covariables->element, sizeof(*covariables->element) * (n_covariables + 1));
  covariables->item = trealloc(covariables->item, sizeof(*covariables->item) * (n_covariables + 1));
  covariables->equation = trealloc(covariables->equation, sizeof(*covariables->equation) * (n_covariables + 1));
  covariables->pcode = trealloc(covariables->pcode, sizeof(*covariables->pcode) * (n_covariables + 1));
  covariables->varied_quan_name = trealloc(covariables->varied_quan_name, sizeof(*covariables->varied_quan_name) * (n_covariables + 1));
  covariables->varied_quan_unit = trealloc(covariables->varied_quan_unit, sizeof(*covariables->varied_quan_unit) * (n_covariables + 1));
  covariables->varied_type = trealloc(covariables->varied_type, sizeof(*covariables->varied_type) * (n_covariables + 1));
  covariables->varied_param = trealloc(covariables->varied_param, sizeof(*covariables->varied_param) * (n_covariables + 1));
  covariables->varied_quan_value = trealloc(covariables->varied_quan_value, sizeof(*covariables->varied_quan_value) * (n_covariables + 1));
  covariables->memory_number = trealloc(covariables->memory_number, sizeof(*covariables->memory_number) * (n_covariables + 1));

  /* check for valid input */
  if (name == NULL)
    bombElegant("element name missing in optimization_variable namelist", NULL);
  str_toupper(name);
  context = NULL;
  if (!find_element(name, &context, beamline->elem)) {
    printf("error: cannot vary element %s--not in beamline\n", name);
    fflush(stdout);
    exitElegant(1);
  }
  cp_str(&covariables->element[n_covariables], name);
  covariables->varied_type[n_covariables] = context->type;
  if (item == NULL)
    bombElegant("item name missing in optimization_variable list", NULL);
  str_toupper(item);
  if ((covariables->varied_param[n_covariables] = confirm_parameter(item, context->type)) < 0) {
    printf("error: cannot vary %s--no such parameter for %s\n", item, name);
    fflush(stdout);
    exitElegant(1);
  }
  if (entity_description[context->type].parameter[covariables->varied_param[n_covariables]].flags & PARAM_IS_LOCKED)
    bombElegantVA("Error: parameter %s of %s cannot be changed via optimization_covariable\n",
                  entity_description[context->type].parameter[covariables->varied_param[n_covariables]].name,
                  entity_name[context->type]);

  cp_str(&covariables->item[n_covariables], item);
  cp_str(&covariables->varied_quan_unit[n_covariables],
         entity_description[context->type].parameter[covariables->varied_param[n_covariables]].unit);

  if (!get_parameter_value(covariables->varied_quan_value + n_covariables, name, covariables->varied_param[n_covariables],
                           context->type, beamline))
    bombElegant("unable to get initial value for parameter", NULL);

  ptr = tmalloc(sizeof(char) * (strlen(name) + strlen(item) + 4));
  sprintf(ptr, "%s.%s0", name, item);
  rpn_store(covariables->varied_quan_value[n_covariables], NULL, rpn_create_mem(ptr, 0));
  free(ptr);

  covariables->varied_quan_name[n_covariables] = tmalloc(sizeof(char) * (strlen(name) + strlen(item) + 3));
  sprintf(covariables->varied_quan_name[n_covariables], "%s.%s", name, item);
  cp_str(&covariables->equation[n_covariables], equation);
  rpn_store(covariables->varied_quan_value[n_covariables] = rpn(equation), NULL,
            covariables->memory_number[n_covariables] = rpn_create_mem(covariables->varied_quan_name[n_covariables], 0));
  if (rpn_check_error())
    exitElegant(1);
  printf("Initial value of %s is %e %s\n",
         covariables->varied_quan_name[n_covariables], covariables->varied_quan_value[n_covariables],
         covariables->varied_quan_unit[n_covariables]);
  fflush(stdout);

  sprintf(nameBuffer, "AOCEqn%ld", nameIndex++);
  create_udf(nameBuffer, equation);
  if (!SDDS_CopyString(&ptr, nameBuffer))
    SDDS_PrintErrors(stdout, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  covariables->pcode[n_covariables] = ptr;
  optimization_data->covariables.n_covariables += 1;
  log_exit("add_optimization_covariable");
}

void add_optimization_constraint(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  long n_constraints;
  OPTIM_CONSTRAINTS *constraints;

  log_entry("add_optimization_constraint");

  constraints = &(optimization_data->constraints);
  n_constraints = constraints->n_constraints;

  constraints->quantity = trealloc(constraints->quantity, sizeof(*constraints->quantity) * (n_constraints + 1));
  constraints->index = trealloc(constraints->index, sizeof(*constraints->index) * (n_constraints + 1));
  constraints->lower = trealloc(constraints->lower, sizeof(*constraints->lower) * (n_constraints + 1));
  constraints->upper = trealloc(constraints->upper, sizeof(*constraints->upper) * (n_constraints + 1));

  /*
    quantity = NULL;
    lower = upper = 0;
 */

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&optimization_constraint, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimization_constraint);

  /* check for valid input */
  if (quantity == NULL)
    bombElegant("quantity name missing in optimization_constraint namelist", NULL);
  constraints->quantity[n_constraints] = quantity;
  if ((constraints->index[n_constraints] = get_final_property_index(quantity)) < 0)
    bombElegant("no match for quantity to be constrained", NULL);
  if (lower > upper)
    bombElegant("lower > upper for constraint", NULL);
  constraints->lower[n_constraints] = lower;
  constraints->upper[n_constraints] = upper;

  constraints->n_constraints += 1;
  log_exit("add_optimization_constraint");
}

void do_set_reference_particle_output(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  SDDS_DATASET SDDSin;
  char s[16834];
  double *coord, beta, maxWeight;
  long i;
  char *name[COORDINATES_PER_PARTICLE + 1] = {"x", "xp", "y", "yp", "t", "p", "particleID"};

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&set_reference_particle_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &set_reference_particle_output);

  /* check for valid input */
  if (set_reference_particle_output_struct.weight <= 0)
    return;
  if (!set_reference_particle_output_struct.match_to)
    bombElegant("Provide fileame with match_to parameter", NULL);

  maxWeight = 0;
  for (i = 0; i < 6; i++) {
    if ((optimization_data->particleMatchingWeight[i] = set_reference_particle_output_struct.weight[i]) < 0)
      bombElegant("weight cannot be negative", NULL);
    if (optimization_data->particleMatchingWeight[i] > maxWeight)
      maxWeight = optimization_data->particleMatchingWeight[i];
  }
  if (maxWeight == 0)
    bombElegant("weights cannot all be zero", NULL);

  optimization_data->particleMatchingMode = COMPARE_PARTICLE_SUM_ABSDEV;
  if (set_reference_particle_output_struct.comparison_mode && strlen(set_reference_particle_output_struct.comparison_mode)) {
    if ((i = match_string(set_reference_particle_output_struct.comparison_mode,
                          particleDeviationComparisonMode, N_PARTICLE_COMPARISON_MODES, EXACT_MATCH)) < 0) {
      fprintf(stderr, "unknown comparison mode \"%s\".  Known modes are ", set_reference_particle_output_struct.comparison_mode);
      for (i = 0; i < N_PARTICLE_COMPARISON_MODES; i++)
        fprintf(stderr, "\"%s\"%s", particleDeviationComparisonMode[i],
                i == (N_PARTICLE_COMPARISON_MODES - 1) ? "\n" : ", ");
      exit(1);
    }
    optimization_data->particleMatchingMode = COMPARE_PARTICLE_SUM_ABSDEV << i;
  }

  printf("Reading reference particle data from %s\n", set_reference_particle_output_struct.match_to);
  fflush(stdout);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, set_reference_particle_output_struct.match_to)) {
    sprintf(s, "Problem opening beam input file %s", set_reference_particle_output_struct.match_to);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if (!check_sdds_column(&SDDSin, "x", "m") ||
      !check_sdds_column(&SDDSin, "y", "m") ||
      !check_sdds_column(&SDDSin, "xp", NULL) ||
      !check_sdds_column(&SDDSin, "yp", NULL) ||
      !check_sdds_column(&SDDSin, "t", "s") ||
      (!check_sdds_column(&SDDSin, "p", "m$be$nc") && !check_sdds_column(&SDDSin, "p", NULL)) ||
      !check_sdds_column(&SDDSin, "particleID", NULL)) {
    sprintf(s, "Problem with existence or units of columns in beam input file %s", set_reference_particle_output_struct.match_to);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if (SDDS_ReadPage(&SDDSin) != 1 || (optimization_data->nParticlesToMatch = SDDS_RowCount(&SDDSin)) <= 0) {
    sprintf(s, "Problem with empty or nonexistent data page in %s", set_reference_particle_output_struct.match_to);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  optimization_data->coordinatesToMatch =
    (double **)czarray_2d(sizeof(double), optimization_data->nParticlesToMatch, totalPropertiesPerParticle);

  for (i = 0; i < COORDINATES_PER_PARTICLE + 1; i++) {
    long j;
    if (!(coord = SDDS_GetColumnInDoubles(&SDDSin, name[i]))) {
      sprintf(s, "Problem getting column data from %s", set_reference_particle_output_struct.match_to);
      SDDS_SetError(s);
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    for (j = 0; j < optimization_data->nParticlesToMatch; j++)
      optimization_data->coordinatesToMatch[j][i] = coord[j];
    free(coord);
  }

  /* convert (t, p) to (s, delta) */
  for (i = 0; i < optimization_data->nParticlesToMatch; i++) {
    beta = optimization_data->coordinatesToMatch[i][5] / sqrt(sqr(optimization_data->coordinatesToMatch[i][5]) + 1);
    optimization_data->coordinatesToMatch[i][4] *= beta * c_mks;
    optimization_data->coordinatesToMatch[i][5] = optimization_data->coordinatesToMatch[i][5] / run->p_central - 1;
  }

  SDDS_Terminate(&SDDSin);

  printf("%ld particles read from first page of %s\n\n", optimization_data->nParticlesToMatch,
         set_reference_particle_output_struct.match_to);
  fflush(stdout);
}

void summarize_optimization_setup(OPTIMIZATION_DATA *optimization_data) {
  OPTIM_VARIABLES *variables;
  OPTIM_CONSTRAINTS *constraints;
  OPTIM_COVARIABLES *covariables;
  long i;

  log_entry("summarize_optimization_setup");

  constraints = &(optimization_data->constraints);
  variables = &(optimization_data->variables);
  covariables = &(optimization_data->covariables);

  if (optimization_data->fp_log) {
    fprintf(optimization_data->fp_log, "\nOptimization to be performed using method '%s' in mode '%s' with tolerance %e.\n",
            optimize_method[optimization_data->method], optimize_mode[optimization_data->mode], optimization_data->tolerance);
#if USE_MPI
#  if MPI_DEBUG
    fprintf(optimization_data->fp_log, "Mode settings for optimization: parallelTrackingBasedMatrices = %ld, partOnMaster=%d, parallelStatus=%d, lessPartAllowed=%ld, isSlave=%ld, isMaster=%ld, notSinglePart=%ld, runInSinglePartMode=%ld, trajectoryTracking=%ld\n",
            parallelTrackingBasedMatrices, partOnMaster, parallelStatus, lessPartAllowed, isSlave, isMaster, notSinglePart, runInSinglePartMode, trajectoryTracking);
#  endif
    if (runInSinglePartMode) /* For genetic optimization */
      fprintf(optimization_data->fp_log, "    As many as %ld generations will be performed on each of %ld passes.\n",
              optimization_data->n_iterations, optimization_data->n_passes);
    else
#endif
      fprintf(optimization_data->fp_log, "    As many as %ld function evaluations will be performed on each of %ld passes.\n",
              optimization_data->n_evaluations, optimization_data->n_passes);

    if (optimization_data->terms == 0)
      fprintf(optimization_data->fp_log, "    The quantity to be optimized is defined by the equation\n        %s\n    subject to %ld constraints%c\n",
              optimization_data->equation, constraints->n_constraints,
              constraints->n_constraints > 0 ? ':' : '.');
    else
      fprintf(optimization_data->fp_log, "    The quantity to be optimized is defined by the sum of %ld terms,\nsubject to %ld constraints%c\n",
              optimization_data->terms, constraints->n_constraints,
              constraints->n_constraints > 0 ? ':' : '.');
    for (i = 0; i < constraints->n_constraints; i++)
      fprintf(optimization_data->fp_log, "        %13.6e <= %10s <= %13.6e\n",
              constraints->lower[i], constraints->quantity[i], constraints->upper[i]);

    fprintf(optimization_data->fp_log, "    The following variables will be used in the optimization:\n");
    fprintf(optimization_data->fp_log, "    name      initial value   lower limit    upper limit     step size\n");
    fprintf(optimization_data->fp_log, "--------------------------------------------------------------------------\n");
    for (i = 0; i < variables->n_variables; i++) {
      fprintf(optimization_data->fp_log, "%12s  %13.6e", variables->varied_quan_name[i], variables->initial_value[i]);
      if (variables->lower_limit[i] == variables->upper_limit[i])
        fprintf(optimization_data->fp_log, "        -              -      ");
      else
        fprintf(optimization_data->fp_log, "  %13.6e  %13.6e", variables->lower_limit[i], variables->upper_limit[i]);
      if (variables->orig_step[i] == 0)
        fprintf(optimization_data->fp_log, "        -\n");
      else
        fprintf(optimization_data->fp_log, "  %13.6e\n", variables->orig_step[i]);
    }
    fputc('\n', optimization_data->fp_log);
    if (covariables->n_covariables) {
      fprintf(optimization_data->fp_log, "    The following covariables will be used in the optimization:\n");
      fprintf(optimization_data->fp_log, "    name      equation\n");
      fprintf(optimization_data->fp_log, "--------------------------------------------------------------------------\n");
      for (i = 0; i < covariables->n_covariables; i++)
        fprintf(optimization_data->fp_log, "%12s  %s\n", covariables->varied_quan_name[i], covariables->equation[i]);
      fputc('\n', optimization_data->fp_log);
    }
    fputc('\n', optimization_data->fp_log);
    fflush(optimization_data->fp_log);
  }
  if (!log_file || optimization_data->fp_log != stdout) {
    printf("\nOptimization to be performed using method '%s' in mode '%s' with tolerance %e.\n",
           optimize_method[optimization_data->method], optimize_mode[optimization_data->mode], optimization_data->tolerance);
    fflush(stdout);
    printf("    As many as %ld function evaluations will be performed on each of %ld passes.\n",
           optimization_data->n_evaluations, optimization_data->n_passes);
    fflush(stdout);
    printf("    The quantity to be optimized is defined by the equation\n        %s\n    subject to %ld constraints%c\n",
           optimization_data->equation, constraints->n_constraints,
           constraints->n_constraints > 0 ? ':' : '.');
    fflush(stdout);
    for (i = 0; i < constraints->n_constraints; i++) {
      printf("        %13.6e <= %10s <= %13.6e\n",
             constraints->lower[i], constraints->quantity[i], constraints->upper[i]);
      fflush(stdout);
    }
    printf("    The following variables will be used in the optimization:\n");
    fflush(stdout);
    printf("    name      initial value   lower limit    upper limit     step size\n");
    fflush(stdout);
    printf("--------------------------------------------------------------------------\n");
    fflush(stdout);
    for (i = 0; i < variables->n_variables; i++) {
      printf("%12s  %13.6e", variables->varied_quan_name[i], variables->initial_value[i]);
      fflush(stdout);
      if (variables->lower_limit[i] == variables->upper_limit[i]) {
        printf("        -              -      ");
        fflush(stdout);
      } else {
        printf("  %13.6e  %13.6e", variables->lower_limit[i], variables->upper_limit[i]);
        fflush(stdout);
      }
      if (variables->orig_step[i] == 0) {
        printf("        -\n");
        fflush(stdout);
      } else {
        printf("  %13.6e\n", variables->orig_step[i]);
        fflush(stdout);
      }
    }
    fputc('\n', stdout);
    if (covariables->n_covariables) {
      printf("    The following covariables will be used in the optimization:\n");
      fflush(stdout);
      printf("    name      equation\n");
      fflush(stdout);
      printf("--------------------------------------------------------------------------\n");
      fflush(stdout);
      for (i = 0; i < covariables->n_covariables; i++) {
        printf("%12s  %s\n", covariables->varied_quan_name[i], covariables->equation[i]);
        fflush(stdout);
      }
      fputc('\n', stdout);
    }
    fputc('\n', stdout);
  }
  log_exit("summarize_optimization_setup");
}

/* variables needed to do tracking for optimization--this data has to be held in global
 * variables like this since the optimization function has a simple calling syntax
 */
static RUN *run;
static VARY *control;
static ERRORVAL *error;
static BEAM *beam;
static OPTIMIZATION_DATA *optimization_data;
static OUTPUT_FILES *output;
static LINE_LIST *beamline;
static CHROM_CORRECTION *chromCorrData;
static void *orbitCorrData;
static long orbitCorrMode;
static TUNE_CORRECTION *tuneCorrData;
static long beam_type_code, n_evaluations_made, n_passes_made;
static double *final_property_value;
static long final_property_values;
static double charge;
static unsigned long optim_func_flags;
static long force_output;
static long doClosedOrbit, doChromCorr, doTuneCorr, doFindAperture, doResponse;

/* structure to keep results of last N optimization function
 * evaluations, so we don't track the same thing twice.
 */
#define MAX_OPTIM_RECORDS 50
typedef struct {
  long invalid;
  double *variableValue;
  double result;
  long usedBefore;
} OPTIM_RECORD;
static long optimRecords = 0, nextOptimRecordSlot = 0, balanceTerms = 0, ignoreOptimRecords = 0;
static OPTIM_RECORD optimRecord[MAX_OPTIM_RECORDS];
static double bestResult = DBL_MAX;

#if defined(__linux__) || (__APPLE__)
#  include <signal.h>
void traceback_handler(int signal);

void optimizationInterruptHandler(int signal) {
  simplexMinAbort(1);
  optimAbort(1);
  printf("Aborting optimization...");
}
#endif

int variableAtLimit(double value, double lower, double upper) {
  double range = upper - lower;
  if (range == 0 || (value - lower) / range < 0.01 || (upper - value) / range < 0.01)
    return 1;
  return 0;
}

void do_optimize(NAMELIST_TEXT *nltext, RUN *run1, VARY *control1, ERRORVAL *error1,
                 LINE_LIST *beamline1, BEAM *beam1, OUTPUT_FILES *output1,
                 OPTIMIZATION_DATA *optimization_data1,
                 void *chromCorrData1, long beam_type1,
                 long doClosedOrbit1, long doChromCorr1,
                 void *orbitCorrData1, long orbitCorrMode1,
                 void *tuneCorrData1, long doTuneCorr1,
                 long doFindAperture1, long doResponse1) {
  static long optUDFcount = 0;
  void optimization_report(double result, double *value, long pass, long n_evals, long n_dim);
  void (*optimization_report_ptr)(double result, double *value, long pass, long n_evals, long n_dim) = optimization_report;
  OPTIM_VARIABLES *variables;
  OPTIM_COVARIABLES *covariables;
  OPTIM_CONSTRAINTS *constraints;
  double result, lastResult;
  long i, startsLeft, i_step_saved;
#if USE_MPI
  long hybrid_simplex_tolerance_counter;
  long n_total_evaluations_made = 0;
  double scale_factor = optimization_data1->random_factor;
  int min_location = 0;
  double worst_result, median, average, spread = 0.0;
  static double *covariables_global = NULL, *result_array;
  simplexComparisonStep = 0;
  targetReached = 0;
#endif

#if USE_MPI && defined(MPI_DEBUG)
  FILE *fpdebug = NULL;
  char sdebug[100];
  long stepDebug = 0;
#endif
  log_entry("do_optimize");

  optimRecords = ignoreOptimRecords = nextOptimRecordSlot = 0;
  bestResult = DBL_MAX;

  stopOptimization = 0;
  run = run1;
  control = control1;
  error = error1;
  beamline = beamline1;
  beam = beam1;
  output = output1,
  optimization_data = optimization_data1;
  chromCorrData = (CHROM_CORRECTION *)chromCorrData1;
  beam_type_code = beam_type1;
  doClosedOrbit = doClosedOrbit1;
  doChromCorr = doChromCorr1;
  orbitCorrData = orbitCorrData1;
  orbitCorrMode = orbitCorrMode1;
  tuneCorrData = (TUNE_CORRECTION *)tuneCorrData1;
  doTuneCorr = doTuneCorr1;
  doFindAperture = doFindAperture1;
  doResponse = doResponse1;

  n_evaluations_made = 0;
  n_passes_made = 0;
  i_step_saved = control->i_step;

  variables = &(optimization_data->variables);
  covariables = &(optimization_data->covariables);
  constraints = &(optimization_data->constraints);

  if (variables->n_variables == 0)
    bombElegant("no variables specified for optimization", NULL);

  for (i = 0; i < MAX_OPTIM_RECORDS; i++)
    optimRecord[i].variableValue = tmalloc(sizeof(*optimRecord[i].variableValue) *
                                           variables->n_variables);

  /* set the end-of-optimization hidden variable to 0 */
  variables->varied_quan_value[variables->n_variables] = 0;
  /* set the optimization function hidden variable to 0 */
  variables->varied_quan_value[variables->n_variables + 1] = 0;

  if (optimization_data->equation == NULL || !strlen(optimization_data->equation)) {
    long i, length;
    if (optimization_data->terms == 0)
      bombElegant("give an optimization equation or at least one optimization term", NULL);
    for (i = length = 0; i < optimization_data->terms; i++)
      length += strlen(optimization_data->term[i]) + 4;
    if (!(optimization_data->equation = SDDS_Malloc(sizeof(*optimization_data->equation) * length)))
      bombElegant("memory allocation failue", NULL);
    optimization_data->equation[0] = '\0';
    for (i = 0; i < optimization_data->terms; i++) {
      strcat(optimization_data->equation, optimization_data->term[i]);
      if (i)
        strcat(optimization_data->equation, " + ");
      else
        strcat(optimization_data->equation, " ");
    }
  }

  /* process namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&optimize, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &optimize);

  if (summarize_setup)
    summarize_optimization_setup(optimization_data);

#if USE_MPI
  if (simplex_log) {
    char buffer[2048];
    if (fpSimplexLog)
      fclose(fpSimplexLog);
    snprintf(buffer, 2048, "%s-%04d", simplex_log, myid);
    fpSimplexLog = fopen(buffer, "w");
    fprintf(fpSimplexLog, "SDDS1\n&column name=Step type=long &end\n");
    fprintf(fpSimplexLog, "&column name=ElapsedTime type=float units=s &end\n");
    fprintf(fpSimplexLog, "&column name=optimizationFunction type=double &end\n");
    fprintf(fpSimplexLog, "&column name=bestOptimizationFunction type=double &end\n");
    fprintf(fpSimplexLog, "&column name=invalid type=short &end\n");
    fprintf(fpSimplexLog, "&column name=state type=short &end\n");
    simplexLogStep = 0;
    for (i = 0; i<variables->n_variables; i++)
      fprintf(fpSimplexLog, "&column name=%s.%s type=double &end\n",
              variables->element[i], variables->item[i]);
    for (i = 0; i<optimization_data->terms; i++)
      fprintf(fpSimplexLog, "&column name=Term%03ld type=double description=\"%s\" &end\n",
              i, optimization_data->term[i]);
  }
#endif

  for (i = 0; i < variables->n_variables; i++) {
    if (variables->varied_type[i] != T_FREEVAR) {
      if (!get_parameter_value(variables->varied_quan_value + i,
                               variables->element[i],
                               variables->varied_param[i],
                               variables->varied_type[i], beamline))
        bombElegant("unable to get initial value for parameter", NULL);
      variables->initial_value[i] = variables->varied_quan_value[i];
    }
    if (variables->initial_value[i] > variables->upper_limit[i]) {
      if (variables->force_inside[i])
        variables->varied_quan_value[i] = variables->initial_value[i] = variables->upper_limit[i];
      else {
        printf("Initial value (%e) is greater than upper limit for %s.%s\n",
               variables->initial_value[i],
               variables->element[i], variables->item[i]);
        exitElegant(1);
      }
    }
    if (variables->initial_value[i] < variables->lower_limit[i]) {
      if (variables->force_inside[i])
        variables->varied_quan_value[i] = variables->initial_value[i] = variables->lower_limit[i];
      else {
        printf("Initial value (%e) is smaller than lower limit for %s.%s\n",
               variables->initial_value[i],
               variables->element[i], variables->item[i]);
        exitElegant(1);
      }
    }
    printf("Initial value for %s.%s is %e\n",
           variables->element[i], variables->item[i],
           variables->initial_value[i]);
    rpn_store(variables->initial_value[i], NULL, variables->memory_number[i]);
    variables->step[i] = variables->orig_step[i];
    if (variables->step[i] == 0) {
      if (variables->lower_limit[i] == variables->upper_limit[i])
        printf("Note: step size for %s set to %e.\n",
               variables->varied_quan_name[i],
               variables->orig_step[i] = variables->step[i] = 1);
      else
        printf("Note: step size for %s set to %e.\n", variables->varied_quan_name[i],
               variables->orig_step[i] = variables->step[i] =
                 (variables->upper_limit[i] - variables->lower_limit[i]) / 10);
      fflush(stdout);
    }
  }

#if USE_MPI
  if (fpSimplexLog)
    fprintf(fpSimplexLog, "&data mode=ascii no_row_counts=1 &end\n");
  if (scale_factor != 1.0)
    for (i = 0; i < variables->n_variables; i++) {
      variables->step[i] *= scale_factor;
    }
#endif

  for (i = 0; i < covariables->n_covariables; i++) {
    if (!get_parameter_value(covariables->varied_quan_value + i,
                             covariables->element[i],
                             covariables->varied_param[i],
                             covariables->varied_type[i],
                             beamline))
      bombElegant("unable to get initial value for parameter", NULL);
    rpn_store(covariables->varied_quan_value[i] = rpn(covariables->equation[i]),
              NULL, covariables->memory_number[i]);
  }

  final_property_values = count_final_properties();
  final_property_value = tmalloc(sizeof(*final_property_value) * final_property_values);

  if (!output->sums_vs_z) {
    output->sums_vs_z = tmalloc(sizeof(*output->sums_vs_z));
    output->n_z_points = 0;
  }

  i = variables->n_variables;
  variables->varied_quan_value[i] = 0; /* end-of-optimization indicator */
  control->i_step = 0;
  optim_func_flags = FINAL_SUMS_ONLY + INHIBIT_FILE_OUTPUT + SILENT_RUNNING + OPTIMIZING;

  if (!optimization_data->UDFcreated) {
    char UDFname[100];
    sprintf(UDFname, "optUDF%ld", optUDFcount++);
    create_udf(UDFname, optimization_data->equation);
    cp_str(&optimization_data->UDFname, UDFname);
    optimization_data->UDFcreated = 1;
  }

#if USE_MPI && defined(MPI_DEBUG)
  sprintf(sdebug, "debug-%03d.sdds", myid);
  fpdebug = fopen(sdebug, "w");
  fprintf(fpdebug, "SDDS1\n&column name=Stage type=string &end\n");
  fprintf(fpdebug, "&column name=Step type=long &end\n");
  fprintf(fpdebug, "&column name=Result type=double &end\n");
  for (i = 0; i < variables->n_variables; i++)
    fprintf(fpdebug, "&column name=V%03ld type=double &end\n", i);
  fprintf(fpdebug, "&data mode=ascii no_row_counts=1 &end\n");
#endif

  startsLeft = optimization_data->n_restarts + 1;
  interrupt_in_progress = 0;
  result = DBL_MAX;
  balanceTerms = 1;
#if USE_MPI
  hybrid_simplex_tolerance_counter = 0;
#endif
#if defined(__linux__) || defined(__APPLE__)
  if (optimization_data->method != OPTIM_METHOD_POWELL)
    signal(SIGINT, optimizationInterruptHandler);
#endif
#if USE_MPI
  if (optimization_data->method != OPTIM_METHOD_SIMPLEX)
    SDDS_PopulationSetup(population_log, &(optimization_data->popLog), &(optimization_data->variables), &(optimization_data->covariables));
#endif
  while (startsLeft-- && !stopOptimization) {
    lastResult = result;
    switch (optimization_data->method) {
    case OPTIM_METHOD_SIMPLEX:
      fputs("Starting simplex optimization.\n", stdout);
#if USE_MPI
    case OPTIM_METHOD_HYBSIMPLEX:
      hybrid_simplex_comparison_interval = optimization_data1->hybrid_simplex_comparison_interval;
      if (optimization_data->method == OPTIM_METHOD_HYBSIMPLEX) {
        fputs("Starting hybrid simplex optimization.\n", stdout);
        for (i = 0; i < variables->n_variables; i++)
          variables->step[i] = (random_2(0) - 0.5) * variables->orig_step[i] * scale_factor;
        /* Disabling the report from simplexMin routine, as it will print result from the Master only.
	     We print the best result across all the processors in a higher level routine */
        optimization_report_ptr = NULL;
      }
#  if MPI_DEBUG
      fprintf(fpdebug, "Before %ld %g ", stepDebug, result);
      for (i = 0; i < variables->n_variables; i++)
        fprintf(fpdebug, "%g ", variables->varied_quan_value[i]);
      fprintf(fpdebug, "\n");
      fflush(fpdebug);
#  endif
#endif
      if (simplexMin(&result, variables->varied_quan_value, variables->step,
                     variables->lower_limit, variables->upper_limit, NULL,
                     variables->n_variables, optimization_data->target,
                     optimization_data->tolerance, optimization_function, optimization_report_ptr,
                     optimization_data->n_evaluations, optimization_data->n_passes, 12,
                     optimization_data->simplexDivisor,
                     optimization_data->simplexPassRangeFactor,
                     (optimization_data->includeSimplex1dScans ? 0 : SIMPLEX_NO_1D_SCANS) +
                       (optimization_data->verbose > 1 ? SIMPLEX_VERBOSE_LEVEL1 : 0) +
                       (optimization_data->verbose > 2 ? SIMPLEX_VERBOSE_LEVEL2 : 0) +
                       (optimization_data->startFromSimplexVertex1 ? SIMPLEX_START_FROM_VERTEX1 : 0)) < 0) {
        if (result > optimization_data->tolerance) {
          if (!optimization_data->soft_failure)
            bombElegant("optimization unsuccessful--aborting", NULL);
          else
            printWarning("optimize: Simplex optimization unsuccessful.", "Continuing.");
        } else
          printWarning("optimize: Maximum number of passes reached in simplex optimization", NULL);
      }
#ifdef DEBUG
      printf("Exited from simplexMin\n");
      fflush(stdout);
#endif
#if USE_MPI
      /* check if the result meets the requirement for each point across all the processors here */
      if (optimization_data->method == OPTIM_METHOD_HYBSIMPLEX) {
#  if MPI_DEBUG
        fprintf(fpdebug, "After %ld %g ", stepDebug, result);
        for (i = 0; i < variables->n_variables; i++)
          fprintf(fpdebug, "%g ", variables->varied_quan_value[i]);
        fprintf(fpdebug, "\n");
        fflush(fpdebug);
#  endif
#  if MPI_DEBUG
        fprintf(stdout, "minimal value is %21.15e on %d\n", result, myid);
        fflush(stdout);
#  endif
        MPI_Barrier(MPI_COMM_WORLD);
        if (population_log) {
          if (optimization_data->print_all_individuals)
            SDDS_PrintPopulations(&(optimization_data->popLog), result, variables->varied_quan_value, variables->n_variables);
            /* Compute statistics of the results over all processors */
#  if MPI_DEBUG
          fprintf(stdout, "Computing statistics across cores\n");
          fflush(stdout);
#  endif
          MPI_Reduce(&result, &worst_result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MPI_Reduce(&result, &average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          if (isMaster) {
            average /= n_processors;
            if (!result_array)
              result_array = tmalloc(n_processors * sizeof(*result_array));
          }
          MPI_Gather(&result, 1, MPI_DOUBLE, result_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          if (isMaster) {
            compute_median(&median, result_array, n_processors);
            spread = 0.0;
            for (i = 0; i < n_processors; i++) {
              if (!isnan(result_array[i]) && !isinf(result_array[i]))
                spread += sqr(result_array[i] - average);
            }
            spread = sqrt(spread / n_processors);
          }
        }
#  if MPI_DEBUG
        fprintf(stdout, "Finding min/max across cores\n");
        fflush(stdout);
#  endif
        find_global_min_index(&result, &min_location, MPI_COMM_WORLD);
        MPI_Bcast(variables->varied_quan_value, variables->n_variables, MPI_DOUBLE, min_location, MPI_COMM_WORLD);

#  if MPI_DEBUG
        fprintf(fpdebug, "Sync %ld %g ", stepDebug, result);
        for (i = 0; i < variables->n_variables; i++)
          fprintf(fpdebug, "%g ", variables->varied_quan_value[i]);
        fprintf(fpdebug, "\n");
        fflush(fpdebug);

        result = optimization_function(variables->varied_quan_value, &i);
        fprintf(fpdebug, "Check %ld %g ", stepDebug, result);
        for (i = 0; i < variables->n_variables; i++)
          fprintf(fpdebug, "%g ", variables->varied_quan_value[i]);
        fprintf(fpdebug, "\n");
        fflush(fpdebug);

        stepDebug++;
#  endif
        if (optimization_data->fp_log && optimization_data->verbose > 1)
          optimization_report(result, variables->varied_quan_value, optimization_data->n_restarts + 1 - startsLeft, n_evaluations_made, variables->n_variables);
      }
#endif
      if (simplexMinAbort(0) || result < optimization_data->target)
        stopOptimization = 1;
#if USE_MPI
      if (optimization_data->method == OPTIM_METHOD_HYBSIMPLEX && optimization_data->hybrid_simplex_tolerance > 0) {
        if ((lastResult - result) < optimization_data->hybrid_simplex_tolerance) {
          if (++hybrid_simplex_tolerance_counter > optimization_data->hybrid_simplex_tolerance_count)
            stopOptimization = 1;
        } else
          hybrid_simplex_tolerance_counter = 0;
      }
#endif
#ifdef DEBUG
      printf("End of simplex method case\n");
      fflush(stdout);
#endif
      break;
    case OPTIM_METHOD_1DSCANS:
      fputs("Starting 1d scan optimization.\n", stdout);
      if (OneDScanOptimize(&result, variables->varied_quan_value, variables->step,
                           variables->lower_limit, variables->upper_limit, NULL,
                           variables->n_variables, optimization_data->target,
                           optimization_data->tolerance, optimization_function, optimization_report_ptr,
                           optimization_data->n_evaluations, optimization_data->n_passes, 1,
                           0)<0) {
        if (result > optimization_data->tolerance) {
          if (!optimization_data->soft_failure)
            bombElegant("optimization unsuccessful--aborting", NULL);
          else
            printWarning("optimize: 1D scan optimization unsuccessful.", "Continuing.");
        } else
          printWarning("optimize: Maximum number of passes reached in 1d scan optimization", NULL);
      }
      if (optimization_data->fp_log && optimization_data->verbose > 1)
        optimization_report(result, variables->varied_quan_value, optimization_data->n_restarts + 1 - startsLeft, n_evaluations_made, variables->n_variables);
      if (simplexMinAbort(0) || result < optimization_data->target)
        stopOptimization = 1;
      break;
    case OPTIM_METHOD_RCDS:
      {
      double *start;
      start = tmalloc(sizeof(*start)*variables->n_variables);
      memcpy(start, variables->varied_quan_value, sizeof(*start)*variables->n_variables);
      fputs("Starting RCDS optimization.\n", stdout);
      if (rcdsMin(&result, variables->varied_quan_value, start, variables->step,
                  variables->lower_limit, variables->upper_limit, NULL,
                  variables->n_variables, optimization_data->target,
                  optimization_data->tolerance, optimization_function, optimization_report_ptr,
                  optimization_data->n_evaluations, optimization_data->n_passes, 
                  0.0, optimization_data->rcdsStepFactor, 0)<0) {
        if (result > optimization_data->tolerance) {
          if (!optimization_data->soft_failure)
            bombElegant("optimization unsuccessful--aborting", NULL);
          else
            printWarning("optimize: RCDS optimization unsuccessful.", "Continuing.");
        } else
          printWarning("optimize: Maximum number of passes reached in RCDS optimization", NULL);
      }
      if (optimization_data->fp_log && optimization_data->verbose > 1)
        optimization_report(result, variables->varied_quan_value, optimization_data->n_restarts + 1 - startsLeft, n_evaluations_made, variables->n_variables);
      if (rcdsMinAbort(0) || result < optimization_data->target)
        stopOptimization = 1;
      free(start);
      }
      break;
    case OPTIM_METHOD_POWELL:
      fputs("Starting Powell optimization.\n", stdout);
      if (powellMin(&result, variables->varied_quan_value, variables->step,
                    variables->lower_limit, variables->upper_limit,
                    variables->n_variables, optimization_data->target,
                    optimization_data->tolerance, optimization_function,
                    optimization_report,
                    optimization_data->n_evaluations / optimization_data->n_passes + 1,
                    optimization_data->n_evaluations *
                      (optimization_data->n_evaluations / optimization_data->n_passes + 1),
                    3) < 0) {
        if (result > optimization_data->tolerance) {
          if (!optimization_data->soft_failure)
            bombElegant("optimization unsuccessful--aborting", NULL);
          else
            printWarning("optimize: Powell optimization unsuccessful.", "Continuing.");
        } else
          printWarning("optimize: Maximum number of passes reached in powell optimization.", NULL);
      }
      break;
    case OPTIM_METHOD_GRID:
      fputs("Starting grid-search optimization.", stdout);
      if (!grid_search_min(&result, variables->varied_quan_value, variables->lower_limit,
                           variables->upper_limit, variables->step,
                           variables->n_variables, optimization_data->target,
                           optimization_function)) {
        if (!optimization_data->soft_failure)
          bombElegant("optimization unsuccessful--aborting", NULL);
        else
          printWarning("optimize: grid-search optimization unsuccessful.", "Continuing.");
      }
      if (optimAbort(0))
        stopOptimization = 1;
      break;
    case OPTIM_METHOD_SAMPLE:
      fputs("Starting grid-sample optimization.", stdout);
      if (!grid_sample_min(&result, variables->varied_quan_value, variables->lower_limit,
                           variables->upper_limit, variables->step,
                           variables->n_variables, optimization_data->target,
                           optimization_function, optimization_data->n_evaluations * 1.0,
                           random_1_elegant)) {
        if (!optimization_data->soft_failure)
          bombElegant("optimization unsuccessful--aborting", NULL);
        else
          printWarning("optimize: grid-sample optimization unsuccessful.", "Continuing.");
      }
      if (optimAbort(0))
        stopOptimization = 1;
      break;
    case OPTIM_METHOD_RANSAMPLE:
      fputs("Starting random-sampled optimization.", stdout);
      if (!randomSampleMin(&result, variables->varied_quan_value, variables->lower_limit,
                           variables->upper_limit, variables->n_variables,
                           optimization_data->target, optimization_function,
                           optimization_data->n_evaluations, random_1_elegant)) {
        if (!optimization_data->soft_failure)
          bombElegant("optimization unsuccessful--aborting", NULL);
        else
          printWarning("optimize: random-sample optimization unsuccessful.", "Continuing.");
      }
      if (optimAbort(0))
        stopOptimization = 1;
      break;
    case OPTIM_METHOD_RANWALK:
      fputs("Starting random-sampled optimization.", stdout);
      if (!randomWalkMin(&result, variables->varied_quan_value,
                         variables->lower_limit, variables->upper_limit,
                         variables->step,
                         variables->n_variables, optimization_data->target,
                         optimization_function,
                         optimization_data->n_evaluations, random_1_elegant)) {
        if (!optimization_data->soft_failure)
          bombElegant("optimization unsuccessful--aborting", NULL);
        else
          printWarning("optimize: random-walk optimization unsuccessful.", "Continuing.");
      }
      if (optimAbort(0))
        stopOptimization = 1;
      break;
#if USE_MPI
    case OPTIM_METHOD_GENETIC:
      fputs("Starting genetic optimization.\n", stdout);
      n_total_evaluations_made = geneticMin(&result, variables->varied_quan_value,
                                            variables->lower_limit, variables->upper_limit, variables->step,
                                            variables->n_variables, optimization_data->target,
                                            optimization_function, optimization_data->n_iterations,
                                            optimization_data->max_no_change,
                                            optimization_data->population_size, output_sparsing_factor,
                                            optimization_data->print_all_individuals, population_log,
                                            &(optimization_data->popLog), optimization_data->verbose,
                                            optimization_data->crossover_type, variables, covariables);
      if (optimAbort(0))
        stopOptimization = 1;
      break;
    case OPTIM_METHOD_SWARM:
      if (optimization_data->n_restarts + 1 - startsLeft == 1)
        fputs("Starting particle swarm optimization.\n", stdout);
      swarmMin(&result, variables->varied_quan_value,
               variables->lower_limit, variables->upper_limit, variables->step,
               variables->n_variables, optimization_data->target,
               optimization_function, optimization_data->population_size,
               optimization_data->n_restarts + 1 - startsLeft, optimization_data->n_restarts);

      /* check if the result meets the requirement for each point across all the processors here */
      if (USE_MPI) {
#  if MPI_DEBUG
        fprintf(stdout, "minimal value is %g for iteration %ld on %d\n", result, optimization_data->n_restarts + 1 - startsLeft, myid);
#  endif
        if (population_log) {
          if (optimization_data->print_all_individuals)
            SDDS_PrintPopulations(&(optimization_data->popLog), result, variables->varied_quan_value, variables->n_variables);
          /* Compute statistics of the results over all processors */
          MPI_Reduce(&result, &worst_result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          MPI_Reduce(&result, &average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          if (isMaster) {
            average /= n_processors;
            if (!result_array)
              result_array = tmalloc(n_processors * sizeof(*result_array));
          }
          MPI_Gather(&result, 1, MPI_DOUBLE, result_array, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          if (isMaster) {
            compute_median(&median, result_array, n_processors);
            spread = 0.0;
            for (i = 0; i < n_processors; i++) {
              if (!isnan(result_array[i]) && !isinf(result_array[i]))
                spread += sqr(result_array[i] - average);
            }
            spread = sqrt(spread / n_processors);
          }
        }

        find_global_min_index(&result, &min_location, MPI_COMM_WORLD);
#  if MPI_DEBUG
        fprintf(stdout, "min_location=%d\n", min_location);
#  endif
        MPI_Bcast(variables->varied_quan_value, variables->n_variables, MPI_DOUBLE, min_location, MPI_COMM_WORLD);
      }
      if (optimAbort(0))
        stopOptimization = 1;
      break;
#else
    case OPTIM_METHOD_GENETIC:
    case OPTIM_METHOD_SWARM:
      bombElegant("This optimization method can only be used in Pelegant.", NULL);
      break;
#endif
    default:
      bombElegant("unknown optimization method code (do_optimize())", NULL);
      break;
    }

#ifdef DEBUG
    printf("Outside optimization method switch\n");
    fflush(stdout);
#endif

#if USE_MPI
    if ((optimization_data->method == OPTIM_METHOD_SWARM) || (optimization_data->method == OPTIM_METHOD_HYBSIMPLEX)) {
      /* The covariables are updated locally after each optimization_function call no matter if a better result 
	   is achieved, which might not match the optimal variables for current iteration. So we keep the global best 
	   covariable values in a separate array. */
      if (covariables->n_covariables) {
        if ((optimization_data->verbose > 1) && optimization_data->fp_log && (optimization_data->n_restarts == startsLeft))
          fprintf(optimization_data->fp_log, "Warning: The covariable values might not match the calculated result from the variable values when there are more "
                                             "than one individual per processor. While the correctness of the final result is not affected.\n");
        /* Only one memory is allocated for each covariable on each processor. It is updated as needed in the optimization function. */
      }
      if (covariables->n_covariables && (optimization_data->verbose > 1) && (result < lastResult || (optimization_data->n_restarts == startsLeft)))
        MPI_Bcast(covariables->varied_quan_value, covariables->n_covariables, MPI_DOUBLE, min_location, MPI_COMM_WORLD);
      if ((optimization_data->verbose > 1) && optimization_data->fp_log) {
        fprintf(optimization_data->fp_log, "Minimal value is %.15g after %ld iterations.\n", result, optimization_data->n_restarts + 1 - startsLeft);
        fprintf(optimization_data->fp_log, "\nNew variable values for iteration %ld\n", optimization_data->n_restarts + 1 - startsLeft);
        fflush(optimization_data->fp_log);
        for (i = 0; i < variables->n_variables; i++)
          fprintf(optimization_data->fp_log, "    %10s: %23.15e\n", variables->varied_quan_name[i], variables->varied_quan_value[i]);
        fflush(optimization_data->fp_log);
      }
      if (covariables->n_covariables) {
        if (optimization_data->n_restarts == startsLeft) {
          covariables_global = trealloc(covariables_global, sizeof(*covariables_global) * (covariables->n_covariables + 1));
        }
        if (result < lastResult || (optimization_data->n_restarts == startsLeft)) {
          for (i = 0; i < covariables->n_covariables; i++)
            covariables_global[i] = covariables->varied_quan_value[i];
        }
        if ((optimization_data->verbose > 1) && optimization_data->fp_log) {
          fprintf(optimization_data->fp_log, "new covariable values:\n");
          for (i = 0; i < covariables->n_covariables; i++)
            fprintf(optimization_data->fp_log, "    %10s: %23.15e\n", covariables->varied_quan_name[i], covariables_global[i]);
          fflush(optimization_data->fp_log);
        }
      }
#  if MPI_DEBUG
      if ((isSlave || !notSinglePart) && (optimization_data->verbose > 1)) {
        fprintf(stdout, "Minimal value is %.15g after %ld iterations.\n", result, optimization_data->n_restarts + 1 - startsLeft);
        printf("\nNew variable values for iteration %ld\n", optimization_data->n_restarts + 1 - startsLeft);
        fflush(stdout);
        for (i = 0; i < variables->n_variables; i++)
          printf("    %10s: %23.15e\n", variables->varied_quan_name[i], variables->varied_quan_value[i]);
        fflush(stdout);
        if (covariables->n_covariables) {
          if (optimization_data->n_restarts == startsLeft)
            covariables_global = trealloc(covariables_global, sizeof(*covariables_global) * (covariables->n_covariables + 1));
          if (result < lastResult || (optimization_data->n_restarts == startsLeft)) {
            for (i = 0; i < covariables->n_covariables; i++)
              covariables_global[i] = covariables->varied_quan_value[i];
          }
          printf("new covariable values:\n");
          for (i = 0; i < covariables->n_covariables; i++)
            printf("    %10s: %23.15e\n", covariables->varied_quan_name[i], covariables_global[i]);
        }
        fflush(stdout);
      }
#  endif
      if (population_log) {
        fputs("Writing population statistics\n", stdout);
        SDDS_PrintStatistics(&(optimization_data->popLog), optimization_data->n_restarts + 1 - startsLeft, result, worst_result, median, average, spread, variables->varied_quan_value, variables->n_variables, covariables_global, covariables->n_covariables, optimization_data->print_all_individuals);
      }
    }
#else

#  ifdef DEBUG
    printf("Evaluating at optimum point\n");
    fflush(stdout);
#  endif

    /* This part looks like redundant, as this is repeated after exiting the while loop. -- Y. Wang */
    /* evaluate once more at the optimimum point to get all parameters right and to get additional output */
    force_output = 1;
    ignoreOptimRecords = 1; /* to force re-evaluation */
    result = optimization_function(variables->varied_quan_value, &i);
    ignoreOptimRecords = 0;
    force_output = 0;
#endif

#ifdef DEBUG
    printf("Done evaluating at optimum point (if needed)\n");
    fflush(stdout);
#endif

    if (result <= optimization_data->target) {
      if (optimization_data->verbose > 1) {
        printf("Target value reached, terminating optimization\n");
      }
      break;
    }
#if USE_MPI
    if ((optimization_data->method != OPTIM_METHOD_SWARM) && (optimization_data->method != OPTIM_METHOD_HYBSIMPLEX))
#endif
#ifdef DEBUG
      printf("Doing MPI post-optimization checks\n");
    fflush(stdout);
#endif
    if (fabs(result - lastResult) < optimization_data->tolerance) {
      if (optimization_data->verbose > 1) {
        printf("New result (%21.15e) not sufficiently different from old result (%21.15e), terminating optimization\n", result, lastResult);
      }
      break;
    }
    lastResult = result;
    if (interrupt_file) {
      interrupt_file_check_time = delapsed_time();
      if (fexists(interrupt_file) &&
          (interrupt_file_mtime == 0 || interrupt_file_mtime < get_mtime(interrupt_file))) {
        printf("Interrupt file %s was created or changed---terminating optimization loop\n", interrupt_file);
        simplexMinAbort(1);
        interrupt_in_progress = 1;
        startsLeft = 0;
        break;
      }
    }
    if (startsLeft && !stopOptimization) {
#ifdef DEBUG
      printf("Preparing to restart\n");
      fflush(stdout);
#endif
      for (i = 0; i < variables->n_variables; i++) {
        variables->step[i] = variables->orig_step[i];
      }
      if (optimization_data->restart_reset_threshold>0) {
        /* reset variables back to their original values if they have not changed much (compared to their range) */
        for (i=0; i<variables->n_variables; i++) {
          double deltaFrac;
          deltaFrac = fabs(variables->varied_quan_value[i]-variables->initial_value[i])
            /fabs(variables->upper_limit[i]-variables->lower_limit[i]);
          if (deltaFrac!=0 && deltaFrac < optimization_data->restart_reset_threshold) {
            if (optimization_data->verbose) {
              printf("Resetting %s to %le (was %le)\n",
                     variables->varied_quan_name[i], variables->initial_value[i], variables->varied_quan_value[i]);
            }
            variables->varied_quan_value[i] = variables->initial_value[i];
          }
        }
      }
      if (optimization_data->restart_worst_term_factor != 1 && optimization_data->terms > 1) {
        int64_t imax, imin, iworst;
        double *savedTermValue;
        /* double newResult; */
        if (!(savedTermValue = malloc(sizeof(*savedTermValue) * optimization_data->terms)))
          bombElegant("memory allocation failure (saving term values)", NULL);
        memcpy(savedTermValue, optimization_data->termValue, optimization_data->terms * sizeof(*savedTermValue));
        /* newResult = lastResult; */
        for (iworst = 0; iworst < optimization_data->restart_worst_terms; iworst++) {
          if (index_min_max(&imin, &imax, optimization_data->termValue, optimization_data->terms)) {
            optimization_data->termWeight[imax] *= optimization_data->restart_worst_term_factor;
            printf("Adjusted weight for term from %le to %le: %s\n",
                   optimization_data->termWeight[imax] / optimization_data->restart_worst_term_factor,
                   optimization_data->termWeight[imax], optimization_data->term[imax]);
            /* just to be sure it doesn't get picked as the max again */
            optimization_data->termValue[imax] = -DBL_MAX;
          } else
            break;
        }
        memcpy(optimization_data->termValue, savedTermValue, optimization_data->terms * sizeof(*savedTermValue));
        index_min_max(&imin, &imax, optimization_data->termWeight, optimization_data->terms);
        if (optimization_data->termWeight[imin])
          for (i = 0; i < optimization_data->terms; i++)
            optimization_data->termWeight[i] /= optimization_data->termWeight[imin];
      }
      if (optimization_data->verbose > 1 && startsLeft > 1) {
        printf("Redoing optimization\n");
        fflush(stdout);
      }
    }
#ifdef DEBUG
    printf("Bottom of optimization restarts loop\n");
    fflush(stdout);
#endif
  }

#if defined(__linux__) || (__APPLE__)
  if (optimization_data->method != OPTIM_METHOD_POWELL)
    signal(SIGINT, traceback_handler);
#endif

  printf("Exited optimization loop\n");
  fflush(stdout);

#if USE_MPI
  last_optimize_function_call = 1;
  min_value_location = min_location;
#endif

  /* evaluate once more at the optimimum point to get all parameters right and to get additional output */
  optim_func_flags = 0;
  force_output = 1;
  ignoreOptimRecords = 1;                                   /* to force re-evaluation */
  variables->varied_quan_value[variables->n_variables] = 1; /* indicates end-of-optimization */
  result = optimization_function(variables->varied_quan_value, &i);
  ignoreOptimRecords = 0;
  force_output = 0;

  for (i = 0; i < MAX_OPTIM_RECORDS; i++)
    free(optimRecord[i].variableValue);

  /* change values in element definitions so that new lattice can be saved */
  change_defined_parameter_values(variables->element, variables->varied_param, variables->varied_type,
                                  variables->varied_quan_value, variables->n_variables);
  if (control->n_elements_to_vary)
    change_defined_parameter_values(control->element, control->varied_param, control->varied_type,
                                    control->varied_quan_value, control->n_elements_to_vary);
  if (covariables->n_covariables)
    change_defined_parameter_values(covariables->element, covariables->varied_param, covariables->varied_type,
                                    covariables->varied_quan_value, covariables->n_covariables);

  outputTuneFootprint(control);

#if USE_MPI
  MPI_Reduce(&n_evaluations_made, &n_total_evaluations_made, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#  if MPI_DEBUG
  if (isSlave) {
    fprintf(stdout, "Minimal value is %.15g after %ld iterations.\n", result, optimization_data->n_restarts + 1 - startsLeft);
    printf("\nNew variable values for iteration %ld\n", optimization_data->n_restarts + 1 - startsLeft);
    fflush(stdout);
    for (i = 0; i < variables->n_variables; i++)
      printf("    %10s: %23.15e\n", variables->varied_quan_name[i], variables->varied_quan_value[i]);
    fflush(stdout);
  }
#  endif
#endif

  /* Master only starting here ... */
  if (isMaster) {
    if (optimization_data->fp_log) {
      fprintf(optimization_data->fp_log, "Optimization results:\n  optimization function has value %.15g\n",
              optimization_data->mode == OPTIM_MODE_MAXIMUM ? -result : result);
      if (optimization_data->terms) {
        double sum;
        fprintf(optimization_data->fp_log, "Terms of equation: \n");
        for (i = sum = 0; i < optimization_data->terms; i++) {
          rpn_clear();
          fprintf(optimization_data->fp_log, "%g*(%20s): %23.15e\n",
                  optimization_data->termWeight[i], optimization_data->term[i], 
                  rpn(optimization_data->term[i]) * optimization_data->termWeight[i]);
          sum += rpn(optimization_data->term[i]) * optimization_data->termWeight[i];
        }
      }
#if !USE_MPI
      fprintf(optimization_data->fp_log, "    A total of %ld function evaluations were made.\n", n_evaluations_made);
#else
      fprintf(optimization_data->fp_log, "    A total of %ld function evaluations were made with an average of %ld function evaluations per processor\n", n_total_evaluations_made, n_total_evaluations_made / n_processors);
#endif
      if (constraints->n_constraints) {
        fprintf(optimization_data->fp_log, "Constraints:\n");
        for (i = 0; i < constraints->n_constraints; i++)
          fprintf(optimization_data->fp_log, "%10s: %23.15e\n", constraints->quantity[i],
                  final_property_value[constraints->index[i]]);
      }
      fprintf(optimization_data->fp_log, "Optimum values of variables and changes from initial values:\n");
      for (i = 0; i < variables->n_variables; i++)
        fprintf(optimization_data->fp_log, "%10s: %23.15e  %23.15e (was %23.15e) %s\n", variables->varied_quan_name[i],
                variables->varied_quan_value[i], variables->varied_quan_value[i] - variables->initial_value[i],
                variables->initial_value[i],
                variableAtLimit(variables->varied_quan_value[i], variables->lower_limit[i],
                                variables->upper_limit[i])
                  ? "(near limit)"
                  : "");
      for (i = 0; i < covariables->n_covariables; i++)
        fprintf(optimization_data->fp_log, "%10s: %23.15e\n", covariables->varied_quan_name[i], covariables->varied_quan_value[i]);
      fflush(optimization_data->fp_log);
    }
    if (!log_file || optimization_data->fp_log != stdout) {
      printf("Optimization results:\n    optimization function has value %.15g\n",
             optimization_data->mode == OPTIM_MODE_MAXIMUM ? -result : result);
      if (optimization_data->terms) {
        double sum;
        printf("Terms of equation: \n");
        for (i = sum = 0; i < optimization_data->terms; i++) {
          rpn_clear();
          printf("%g*(%20s): %23.15e\n",
                 optimization_data->termWeight[i], optimization_data->term[i],
                 optimization_data->termWeight[i] * rpn(optimization_data->term[i]));
          sum += optimization_data->termWeight[i] * rpn(optimization_data->term[i]);
        }
      }
      fflush(stdout);
#if !USE_MPI
      printf("    A total of %ld function evaluations were made.\n", n_evaluations_made);
#else
      if (isMaster)
        printf("    A total of %ld function evaluations were made with an average of %ld function evaluations per processor\n", n_total_evaluations_made, n_total_evaluations_made / n_processors);
#endif
      fflush(stdout);
      if (constraints->n_constraints) {
        printf("Constraints:\n");
        fflush(stdout);
        for (i = 0; i < constraints->n_constraints; i++)
          printf("%10s: %23.15e\n", constraints->quantity[i], final_property_value[constraints->index[i]]);
        fflush(stdout);
      }
      printf("Optimum values of variables and changes from initial values:\n");
      fflush(stdout);
      for (i = 0; i < variables->n_variables; i++)
        printf("%10s: %23.15e  %23.15e\n", variables->varied_quan_name[i],
               variables->varied_quan_value[i], variables->varied_quan_value[i] - variables->initial_value[i]);
      fflush(stdout);
      for (i = 0; i < covariables->n_covariables; i++)
        printf("%10s: %23.15e\n", covariables->varied_quan_name[i], covariables->varied_quan_value[i]);
      fflush(stdout);
    }
    if (term_log_file && strlen(term_log_file) && optimization_data->terms) {
      SDDS_DATASET termLog;
      long i, iterm=0, ivalue=0, iweight=0;
      if (!SDDS_InitializeOutputElegant(&termLog, SDDS_BINARY, 0, NULL, NULL, term_log_file) ||
          (ivalue = SDDS_DefineColumn(&termLog, "Contribution", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (iweight = SDDS_DefineColumn(&termLog, "Weight", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (iterm = SDDS_DefineColumn(&termLog, "Term", NULL, NULL, NULL, NULL, SDDS_STRING, 0)) < 0 ||
          !SDDS_WriteLayout(&termLog) ||
          !SDDS_StartPage(&termLog, optimization_data->terms)) {
        printf("Problem writing optimization term log\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        iterm = ivalue = 0; /* suppress spurious compiler warning */
        exitElegant(1);
      }
      for (i = 0; i < optimization_data->terms; i++) {
        if (!SDDS_SetRowValues(&termLog, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                               iterm, optimization_data->term[i],
                               iweight, optimization_data->termWeight[i],
                               ivalue, optimization_data->termWeight[i] * rpn(optimization_data->term[i]),
                               -1)) {
          printf("Problem writing optimization term log\n");
          SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      }
      if (!SDDS_WritePage(&termLog) || !SDDS_Terminate(&termLog)) {
        printf("Problem writing optimization term log\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
  }
  /* ... Master only ending here */

  for (i = 0; i < variables->n_variables; i++)
    variables->initial_value[i] = variables->varied_quan_value[i];
  control->i_step = i_step_saved;

  if (optimization_data->coordinatesToMatch) {
    free_czarray_2d((void **)optimization_data->coordinatesToMatch, optimization_data->nParticlesToMatch, totalPropertiesPerParticle);
    optimization_data->coordinatesToMatch = NULL;
  }

  log_exit("do_optimize");
}

/* Next three lines from elegant.c: */
#define SET_AWE_BEAM 5
#define SET_BUNCHED_BEAM 6
#define SET_SDDS_BEAM 33

#define N_TWISS_QUANS (89 + 18 + 2)
static char *twiss_name[N_TWISS_QUANS] = {
  "betax", "alphax", "nux", "etax", "etapx",
  "betay", "alphay", "nuy", "etay", "etapy",
  "max.betax", "max.etax", "max.etapx",
  "max.betay", "max.etay", "max.etapy",
  "min.betax", "min.etax", "min.etapx",
  "min.betay", "min.etay", "min.etapy",
  "dnux/dp", "dnuy/dp", "alphac", "alphac2", "alphac3",
  "ave.betax", "ave.betay",
  "etaxp", "etayp",
  "waistsx", "waistsy",
  "dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy",
  "dnux/dp2", "dnux/dp3",
  "dnuy/dp2", "dnuy/dp3",
  "etax2", "etax3",
  "etay2", "etay3",
  "nuxChromLower", "nuxChromUpper",
  "nuyChromLower", "nuyChromUpper",
  "dbetax/dp", "dbetay/dp", "dalphax/dp", "dalphay/dp",
  "dnux/dAx2", "dnux/dAy2", "dnuy/dAx2", "dnuy/dAy2",
  "dnux/dAxAy", "dnuy/dAxAy",
  "nuxTswaLower", "nuxTswaUpper",
  "nuyTswaLower", "nuyTswaUpper",
  "couplingIntegral", "emittanceRatio",
  "h21000", "h30000", "h10110", "h10020", "h10200",
  "dnux/dJx", "dnux/dJy", "dnuy/dJy",
  "h11001", "h00111", "h20001", "h00201", "h10002",
  "h22000", "h11110", "h00220", "h31000", "h40000",
  "h20110", "h11200", "h20020", "h20200", "h00310", "h00400",
  "p99.betax", "p99.etax", "p99.etapx",
  "p99.betay", "p99.etay", "p99.etapy",
  "p98.betax", "p98.etax", "p98.etapx",
  "p98.betay", "p98.etay", "p98.etapy",
  "p96.betax", "p96.etax", "p96.etapx",
  "p96.betay", "p96.etay", "p96.etapy",
  "Ax", "Ay"};
static long twiss_mem[N_TWISS_QUANS] = {-1};

static char *radint_name[14] = {
  "ex0",
  "Sdelta0",
  "Jx",
  "Jy",
  "Jdelta",
  "taux",
  "tauy",
  "taudelta",
  "U0",
  "I1",
  "I2",
  "I3",
  "I4",
  "I5",
};
static long radint_mem[14] = {
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
};
static char *floorCoord_name[7] = {
  "X",
  "Y",
  "Z",
  "theta",
  "phi",
  "psi",
  "sTotal",
};
static long floorCoord_mem[7] = {
  -1, -1, -1, -1, -1, -1, -1};
static char *floorStat_name[6] = {
  "min.X",
  "min.Y",
  "min.Z",
  "max.X",
  "max.Y",
  "max.Z",
};
static long floorStat_mem[6] = {
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
};

static char *tuneFootprintName[20] = {
  "FP.nuxSpreadChrom",
  "FP.nuySpreadChrom",
  "FP.deltaLimit",
  "FP.nuxSpreadAmp",
  "FP.nuySpreadAmp",
  "FP.xSpread",
  "FP.ySpread",
  "FP.diffusionRateMaxChrom",
  "FP.diffusionRateMaxAmp",
  "FP.xyArea",
  "FP.nuxChromMin",
  "FP.nuxChromMax",
  "FP.nuyChromMin",
  "FP.nuyChromMax",
  "FP.nuxAmpMin",
  "FP.nuxAmpMax",
  "FP.nuyAmpMin",
  "FP.nuyAmpMax",
  "FP.chromx1",
  "FP.chromy1",
};
static long tuneFootprintMem[20] = {
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
};

static char *spin_name[3] = {
  "Cspx", "Cspy", "Cspz",
};
static long spin_mem[3] = {
  -1, -1, -1
};

int showTwissMemories(FILE *fp) {
  long i;

  for (i = 0; i < N_TWISS_QUANS; i++) {
    if (twiss_mem[i] == -1)
      break;
    else
      fprintf(fp, "%s = %21.15e\n",
              twiss_name[i], rpn_recall(twiss_mem[i]));
  }
  fflush(fp);
  return 0;
}

double optimization_function(double *value, long *invalid) {
  double rpn(char *expression);
  OPTIM_VARIABLES *variables;
  OPTIM_CONSTRAINTS *constraints;
  OPTIM_COVARIABLES *covariables;
  double conval, result = DBL_MAX, psum;
  long i, iRec, recordUsedAgain;
  unsigned long unstable;
  VMATRIX *M = NULL;
  TWISS twiss_ave, twiss_min, twiss_max;
  TWISS twiss_p99, twiss_p98, twiss_p96;
  double XYZ[3], Angle[3], XYZMin[3], XYZMax[3];
  double startingOrbitCoord[6] = {0, 0, 0, 0, 0, 0};
  long rpnError = 0;
  TUNE_FOOTPRINTS tuneFP;
  static long nLostMemory = -1;
#if USE_MPI
  long beamNoToTrack;
  long nLostTotal;
#endif

  log_entry("optimization_function");

  if (nLostMemory == -1)
    nLostMemory = rpn_create_mem("nLost", 0);
#if DEBUG
  printf("optimization_function: In optimization function\n");
  printf("Beamline flags: %lx\n", beamline->flags);
  fflush(stdout);
#endif

  if (restart_random_numbers)
    seedElegantRandomNumbers(0, RESTART_RN_ALL);

  if (interrupt_file) {
    if (interrupt_file_check_interval==0 || (interrupt_file_check_time+interrupt_file_check_interval)<delapsed_time()) {
      if (fexists(interrupt_file) &&
          (interrupt_file_mtime == 0 || interrupt_file_mtime < get_mtime(interrupt_file))) {
        if (!interrupt_in_progress) {
          printf("Interrupt file %s was created or changed---beginning termination of optimization loop\n", interrupt_file);
          fflush(stdout);
        }
        interrupt_in_progress = 1;
        simplexMinAbort(1);
      }
      interrupt_file_check_time = delapsed_time();
    }
  }

  *invalid = 0;
  unstable = 0;
  n_evaluations_made++;

  variables = &(optimization_data->variables);
  constraints = &(optimization_data->constraints);
  covariables = &(optimization_data->covariables);

  /* assert variable values and store in rpn memories */
  delete_phase_references();
  reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);

  if (beamline->links && beamline->links->n_links)
    reset_element_links(beamline->links, run, beamline);

  assert_parameter_values(variables->element, variables->varied_param, variables->varied_type,
                          value, variables->n_variables, beamline);
  for (i = 0; i < variables->n_variables; i++)
    rpn_store(value[i], NULL, variables->memory_number[i]);

  /* set element flags to indicate variation of parameters */
  set_element_flags(beamline, variables->element, NULL, variables->varied_type, variables->varied_param,
                    variables->n_variables, PARAMETERS_ARE_VARIED, VMATRIX_IS_VARIED, 0, 0);

  if (covariables->n_covariables) {
#if DEBUG
    printf("optimization_function: Computing covariables\n");
    fflush(stdout);
#endif

    /* calculate values of covariables and assert these as well */
    for (i = 0; i < covariables->n_covariables; i++) {
      rpn_clear();
      rpn_store(covariables->varied_quan_value[i] = rpn(covariables->pcode[i]), NULL, covariables->memory_number[i]);
      if (rpn_check_error())
        exitElegant(1);
    }
    assert_parameter_values(covariables->element, covariables->varied_param, covariables->varied_type,
                            covariables->varied_quan_value, covariables->n_covariables, beamline);
    /* set element flags to indicate variation of parameters */
    set_element_flags(beamline, covariables->element, NULL, covariables->varied_type, covariables->varied_param,
                      covariables->n_covariables, PARAMETERS_ARE_VARIED, VMATRIX_IS_VARIED, 0, 0);
  }

  if (optimization_data->verbose && optimization_data->fp_log) {
    fprintf(optimization_data->fp_log, "\nNew variable values for evaluation %ld of pass %ld:\n", n_evaluations_made, n_passes_made + 1);
    fflush(optimization_data->fp_log);
    for (i = 0; i < variables->n_variables; i++)
      fprintf(optimization_data->fp_log, "    %10s: %23.15e\n", variables->varied_quan_name[i], value[i]);
    fflush(optimization_data->fp_log);
    if (covariables->n_covariables) {
      fprintf(optimization_data->fp_log, "new covariable values:\n");
      for (i = 0; i < covariables->n_covariables; i++)
        fprintf(optimization_data->fp_log, "    %10s: %23.15e\n", covariables->varied_quan_name[i], covariables->varied_quan_value[i]);
    }
    fflush(optimization_data->fp_log);
  }

  if ((iRec = checkForOptimRecord(value, variables->n_variables, &recordUsedAgain)) >= 0) {
#if USE_MPI
    if (!runInSinglePartMode) { /* For parallel genetic optimization, all the individuals for the first iteration are same */
#endif
      if (recordUsedAgain > 20) {
        printf("record used too many times---stopping optimization\n");
        stopOptimization = 1;
        simplexMinAbort(1);
      }
      if (optimization_data->verbose && optimization_data->fp_log)
        fprintf(optimization_data->fp_log, "Using previously computed value %23.15e\n\n",
                optimRecord[iRec].result);
      *invalid = optimRecord[iRec].invalid;
#if USE_MPI
    }
    checkTarget(optimRecord[iRec].result, *invalid);
#endif
    return optimRecord[iRec].result;
  }

  /* compute matrices for perturbed elements */
#if DEBUG
  printf("optimization_function: Computing matrices\n");
  fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
  printf("Mode before compute_changed_matrices: parallelTrackingBasedMatrices = %ld, partOnMaster=%d, parallelStatus=%d, lessPartAllowed=%ld, isSlave=%ld, isMaster=%ld, notSinglePart=%ld, runInSinglePartMode=%ld, trajectoryTracking=%ld\n",
         parallelTrackingBasedMatrices, partOnMaster, parallelStatus, lessPartAllowed, isSlave, isMaster, notSinglePart, runInSinglePartMode, trajectoryTracking);
  fflush(stdout);
#endif

  if (beamline->links && beamline->links->n_links)
    rebaseline_element_links(beamline->links, run, beamline);
  i = assert_element_links(beamline->links, run, beamline, STATIC_LINK + DYNAMIC_LINK + LINK_ELEMENT_DEFINITION);
  i += compute_changed_matrices(beamline, run);
#if USE_MPI && MPI_DEBUG
  printf("Mode after compute_changed_matrices: parallelTrackingBasedMatrices = %ld, partOnMaster=%d, parallelStatus=%d, lessPartAllowed=%ld, isSlave=%ld, isMaster=%ld, notSinglePart=%ld, runInSinglePartMode=%ld, trajectoryTracking=%ld\n",
         parallelTrackingBasedMatrices, partOnMaster, parallelStatus, lessPartAllowed, isSlave, isMaster, notSinglePart, runInSinglePartMode, trajectoryTracking);
  fflush(stdout);
#endif
#if DEBUG
  printf("optimization_function: Computed %ld matrices\n", i);
  fflush(stdout);
#endif
  if (beamline->flags & BEAMLINE_CONCAT_DONE)
    free_elements1(beamline->ecat);
  beamline->flags &= ~(BEAMLINE_CONCAT_CURRENT + BEAMLINE_CONCAT_DONE +
                       BEAMLINE_TWISS_CURRENT + BEAMLINE_TWISS_DONE +
                       BEAMLINE_RADINT_CURRENT + BEAMLINE_RADINT_DONE);

  if (i && beamline->matrix) {
    free_matrices(beamline->matrix);
    free(beamline->matrix);
    beamline->matrix = NULL;
  }

  if (optimization_data->verbose && optimization_data->fp_log) {
    fprintf(optimization_data->fp_log, "%ld matrices (re)computed\n", i);
    fflush(optimization_data->fp_log);
  }

#if DEBUG
  printf("optimization_function: Generating beam\n");
  fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
  printf("Generating beam\n");
  fflush(stdout);
#endif
  /* generate initial beam distribution and track it */
  switch (beam_type_code) {
  case SET_AWE_BEAM:
    bombElegant("beam type code of SET_AWE_BEAM in optimization_function--this shouldn't happen", NULL);
    break;
  case SET_BUNCHED_BEAM:
    new_bunched_beam(beam, run, control, output, 0);
    break;
  case SET_SDDS_BEAM:
    if (new_sdds_beam(beam, run, control, output, 0) < 0)
      bombElegant("unable to get beam for tracking (is file empty or out of pages?)", NULL);
    break;
  default:
    bombElegant("unknown beam type code in optimization", NULL);
    break;
  }
  control->i_step++; /* to prevent automatic regeneration of beam */
  zero_beam_sums(output->sums_vs_z, output->n_z_points + 1);

  if (doClosedOrbit) {
    if (!run_closed_orbit(run, beamline, startingOrbitCoord, beam, 0)) {
      *invalid = 1;
      printWarning("optimize: failed to find closed orbit while optimizing.", NULL);
    }
#if DEBUG
    printf("closed orbit computed: (%le, %le, %le, %le)\n",
           beamline->closed_orbit[0].centroid[0],
           beamline->closed_orbit[0].centroid[1],
           beamline->closed_orbit[0].centroid[2],
           beamline->closed_orbit[0].centroid[3]);
#endif
  }
  if (!*invalid && orbitCorrMode != -1 &&
      !do_correction(orbitCorrData, run, beamline, startingOrbitCoord, beam, control->i_step,
                     INITIAL_CORRECTION + NO_OUTPUT_CORRECTION)) {
    *invalid = 1;
    printWarning("optimize: failed to perform orbit correction while optimizing.", NULL);
  }
  if (!*invalid && doTuneCorr &&
      !do_tune_correction(tuneCorrData, run, NULL, beamline, startingOrbitCoord, doClosedOrbit,
                          0, 0)) {
    *invalid = 1;
    printWarning("optimize: failed to do tune correction while optimizing.", NULL);
  }
  if (!*invalid && doChromCorr &&
      !do_chromaticity_correction(chromCorrData, run, beamline, startingOrbitCoord, doClosedOrbit,
                                  0, 0)) {
    *invalid = 1;
    printWarning("optimize: failed to do chromaticity correction while optimizing.", NULL);
  }
  if (!*invalid && doClosedOrbit &&
      !run_closed_orbit(run, beamline, startingOrbitCoord, beam, 0)) {
    *invalid = 1;
    printWarning("optimize: failed to find closed orbit while optimizing.", NULL);
  }

  if (!*invalid && doResponse) {
    update_response(run, beamline, orbitCorrData);
  }

  if (!*invalid && beamline->flags & BEAMLINE_TWISS_WANTED) {
#if USE_MPI && MPI_DEBUG
    printf("Computing twiss parameters\n");
    fflush(stdout);
#endif
    if (twiss_mem[0] == -1) {
      for (i = 0; i < N_TWISS_QUANS; i++)
        twiss_mem[i] = rpn_create_mem(twiss_name[i], 0);
    }
    /* get twiss mode and (beta, alpha, eta, etap) for both planes */
    if (optimization_data->verbose && optimization_data->fp_log) {
      fprintf(optimization_data->fp_log, "Computing twiss parameters for optimization\n");
      fflush(optimization_data->fp_log);
    }
    for (i = 0; i < N_TWISS_QUANS; i++)
      rpn_store(DBL_MAX/2, NULL, twiss_mem[i]);
    update_twiss_parameters(run, beamline, &unstable);
    run_coupled_twiss_output(run, beamline, startingOrbitCoord);
    if (unstable)
      *invalid = 1;
#if DEBUG || MPI_DEBUG
    printf("Twiss parameters done.\n");
    printf("betax=%g, alphax=%g, etax=%g\n",
           beamline->elast->twiss->betax,
           beamline->elast->twiss->alphax,
           beamline->elast->twiss->etax);
    printf("betay=%g, alphay=%g, etay=%g\n",
           beamline->elast->twiss->betay,
           beamline->elast->twiss->alphay,
           beamline->elast->twiss->etay);
    printf("nux=%10.6f, nuy=%10.6f\n",
           beamline->tune[0], beamline->tune[1]);
    fflush(stdout);
#endif
    if (!unstable) {
      /* store twiss parameters for last element */
      for (i = 0; i < 5; i++) {
	rpn_store(*((&beamline->elast->twiss->betax) + i) / (i == 2 ? PIx2 : 1), NULL, twiss_mem[i]);
	rpn_store(*((&beamline->elast->twiss->betay) + i) / (i == 2 ? PIx2 : 1), NULL, twiss_mem[i + 5]);
      }
      /* store statistics */
      compute_twiss_statistics(beamline, &twiss_ave, &twiss_min, &twiss_max);
      rpn_store(twiss_max.betax, NULL, twiss_mem[10]);
      rpn_store(twiss_max.etax, NULL, twiss_mem[11]);
      rpn_store(twiss_max.etapx, NULL, twiss_mem[12]);
      rpn_store(twiss_max.betay, NULL, twiss_mem[13]);
      rpn_store(twiss_max.etay, NULL, twiss_mem[14]);
      rpn_store(twiss_max.etapy, NULL, twiss_mem[15]);
      rpn_store(twiss_min.betax, NULL, twiss_mem[16]);
      rpn_store(twiss_min.etax, NULL, twiss_mem[17]);
      rpn_store(twiss_min.etapx, NULL, twiss_mem[18]);
      rpn_store(twiss_min.betay, NULL, twiss_mem[19]);
      rpn_store(twiss_min.etay, NULL, twiss_mem[20]);
      rpn_store(twiss_min.etapy, NULL, twiss_mem[21]);
      /* chromaticity */
      rpn_store(beamline->chromaticity[0], NULL, twiss_mem[22]);
      rpn_store(beamline->chromaticity[1], NULL, twiss_mem[23]);
      /* first and second-order momentum compaction */
      rpn_store(beamline->alpha[0], NULL, twiss_mem[24]);
      rpn_store(beamline->alpha[1], NULL, twiss_mem[25]);
      rpn_store(beamline->alpha[2], NULL, twiss_mem[26]);
      /* more statistics */
      rpn_store(twiss_ave.betax, NULL, twiss_mem[27]);
      rpn_store(twiss_ave.betay, NULL, twiss_mem[28]);
      /* alternate names for etapx and etapy */
      rpn_store(beamline->elast->twiss->etapx, NULL, twiss_mem[29]);
      rpn_store(beamline->elast->twiss->etapy, NULL, twiss_mem[30]);
      /* number of waists per plane */
      rpn_store((double)beamline->waists[0], NULL, twiss_mem[31]);
      rpn_store((double)beamline->waists[1], NULL, twiss_mem[32]);
      /* amplitude-dependent tune shifts */
      rpn_store(beamline->dnux_dA[1][0], NULL, twiss_mem[33]);
      rpn_store(beamline->dnux_dA[0][1], NULL, twiss_mem[34]);
      rpn_store(beamline->dnuy_dA[1][0], NULL, twiss_mem[35]);
      rpn_store(beamline->dnuy_dA[0][1], NULL, twiss_mem[36]);
      /* higher-order chromaticities */
      rpn_store(beamline->chrom2[0], NULL, twiss_mem[37]);
      rpn_store(beamline->chrom3[0], NULL, twiss_mem[38]);
      rpn_store(beamline->chrom2[1], NULL, twiss_mem[39]);
      rpn_store(beamline->chrom3[1], NULL, twiss_mem[40]);
      /* higher-order dispersion */
      rpn_store(beamline->eta2[0], NULL, twiss_mem[41]);
      rpn_store(beamline->eta3[0], NULL, twiss_mem[42]);
      rpn_store(beamline->eta2[2], NULL, twiss_mem[43]);
      rpn_store(beamline->eta3[2], NULL, twiss_mem[44]);
      /* limits of tunes due to chromatic effects */
      rpn_store(beamline->tuneChromLower[0], NULL, twiss_mem[45]);
      rpn_store(beamline->tuneChromUpper[0], NULL, twiss_mem[46]);
      rpn_store(beamline->tuneChromLower[1], NULL, twiss_mem[47]);
      rpn_store(beamline->tuneChromUpper[1], NULL, twiss_mem[48]);
      /* derivative of beta functions with momentum offset */
      rpn_store(beamline->dbeta_dPoP[0], NULL, twiss_mem[49]);
      rpn_store(beamline->dbeta_dPoP[1], NULL, twiss_mem[50]);
      rpn_store(beamline->dalpha_dPoP[0], NULL, twiss_mem[51]);
      rpn_store(beamline->dalpha_dPoP[1], NULL, twiss_mem[52]);
      /* higher-order tune shifts with amplitude */
      rpn_store(beamline->dnux_dA[2][0], NULL, twiss_mem[53]);
      rpn_store(beamline->dnux_dA[0][2], NULL, twiss_mem[54]);
      rpn_store(beamline->dnuy_dA[2][0], NULL, twiss_mem[55]);
      rpn_store(beamline->dnuy_dA[0][2], NULL, twiss_mem[56]);
      rpn_store(beamline->dnux_dA[1][1], NULL, twiss_mem[57]);
      rpn_store(beamline->dnuy_dA[1][1], NULL, twiss_mem[58]);
      /* tune extrema due to TSWA */
      rpn_store(beamline->nuxTswaExtrema[0], NULL, twiss_mem[59]);
      rpn_store(beamline->nuxTswaExtrema[1], NULL, twiss_mem[60]);
      rpn_store(beamline->nuyTswaExtrema[0], NULL, twiss_mem[61]);
      rpn_store(beamline->nuyTswaExtrema[1], NULL, twiss_mem[62]);
      /* coupling parameters */
      rpn_store(beamline->couplingFactor[0], NULL, twiss_mem[63]);
      rpn_store(beamline->couplingFactor[2], NULL, twiss_mem[64]);
      /* geometric driving terms */
      rpn_store(beamline->drivingTerms.h21000[0], NULL, twiss_mem[65]);
      rpn_store(beamline->drivingTerms.h30000[0], NULL, twiss_mem[66]);
      rpn_store(beamline->drivingTerms.h10110[0], NULL, twiss_mem[67]);
      rpn_store(beamline->drivingTerms.h10020[0], NULL, twiss_mem[68]);
      rpn_store(beamline->drivingTerms.h10200[0], NULL, twiss_mem[69]);
      rpn_store(beamline->drivingTerms.dnux_dJx, NULL, twiss_mem[70]);
      rpn_store(beamline->drivingTerms.dnux_dJy, NULL, twiss_mem[71]);
      rpn_store(beamline->drivingTerms.dnuy_dJy, NULL, twiss_mem[72]);
      rpn_store(beamline->drivingTerms.h11001[0], NULL, twiss_mem[73]);
      rpn_store(beamline->drivingTerms.h00111[0], NULL, twiss_mem[74]);
      rpn_store(beamline->drivingTerms.h20001[0], NULL, twiss_mem[75]);
      rpn_store(beamline->drivingTerms.h00201[0], NULL, twiss_mem[76]);
      rpn_store(beamline->drivingTerms.h10002[0], NULL, twiss_mem[77]);
      rpn_store(beamline->drivingTerms.h22000[0], NULL, twiss_mem[78]);
      rpn_store(beamline->drivingTerms.h11110[0], NULL, twiss_mem[79]);
      rpn_store(beamline->drivingTerms.h00220[0], NULL, twiss_mem[80]);
      rpn_store(beamline->drivingTerms.h31000[0], NULL, twiss_mem[81]);
      rpn_store(beamline->drivingTerms.h40000[0], NULL, twiss_mem[82]);
      rpn_store(beamline->drivingTerms.h20110[0], NULL, twiss_mem[83]);
      rpn_store(beamline->drivingTerms.h11200[0], NULL, twiss_mem[84]);
      rpn_store(beamline->drivingTerms.h20020[0], NULL, twiss_mem[85]);
      rpn_store(beamline->drivingTerms.h20200[0], NULL, twiss_mem[86]);
      rpn_store(beamline->drivingTerms.h00310[0], NULL, twiss_mem[87]);
      rpn_store(beamline->drivingTerms.h00400[0], NULL, twiss_mem[88]);

      compute_twiss_percentiles(beamline, &twiss_p99, &twiss_p98, &twiss_p96);
      rpn_store(twiss_p99.betax, NULL, twiss_mem[89]);
      rpn_store(twiss_p99.etax, NULL,  twiss_mem[90]);
      rpn_store(twiss_p99.etapx, NULL, twiss_mem[91]);
      rpn_store(twiss_p99.betay, NULL, twiss_mem[92]);
      rpn_store(twiss_p99.etay, NULL,  twiss_mem[93]);
      rpn_store(twiss_p99.etapy, NULL, twiss_mem[94]);
      rpn_store(twiss_p98.betax, NULL, twiss_mem[95]);
      rpn_store(twiss_p98.etax, NULL,  twiss_mem[96]);
      rpn_store(twiss_p98.etapx, NULL, twiss_mem[97]);
      rpn_store(twiss_p98.betay, NULL, twiss_mem[98]);
      rpn_store(twiss_p98.etay, NULL,  twiss_mem[99]);
      rpn_store(twiss_p98.etapy, NULL, twiss_mem[100]);
      rpn_store(twiss_p96.betax, NULL, twiss_mem[101]);
      rpn_store(twiss_p96.etax, NULL,  twiss_mem[102]);
      rpn_store(twiss_p96.etapx, NULL, twiss_mem[103]);
      rpn_store(twiss_p96.betay, NULL, twiss_mem[104]);
      rpn_store(twiss_p96.etay, NULL,  twiss_mem[105]);
      rpn_store(twiss_p96.etapy, NULL, twiss_mem[106]);
      
      rpn_store(beamline->acceptance[0], NULL, twiss_mem[107]);
      rpn_store(beamline->acceptance[1], NULL, twiss_mem[108]);
    }
    
#if DEBUG || MPI_DEBUG
    printf("Twiss parameters stored.\n");
    fflush(stdout);
#endif
  }

  if (beamline->flags & BEAMLINE_RADINT_WANTED) {    
    if (radint_mem[0] == -1) {
      for (i = 0; i < 14; i++)
	radint_mem[i] = rpn_create_mem(radint_name[i], 0);
    }
    if (!*invalid) {
      if (optimization_data->verbose && optimization_data->fp_log) {
	fprintf(optimization_data->fp_log, "Updating radiation integral values for optimization\n");
	fflush(optimization_data->fp_log);
      }
      /* radiation integrals already updated by update_twiss_parameters above
	 which is guaranteed to be called
      */
      rpn_store(beamline->radIntegrals.ex0 > 0 ? beamline->radIntegrals.ex0 : sqrt(DBL_MAX), NULL,
		radint_mem[0]);
      rpn_store(beamline->radIntegrals.sigmadelta, NULL, radint_mem[1]);
      rpn_store(beamline->radIntegrals.Jx, NULL, radint_mem[2]);
      rpn_store(beamline->radIntegrals.Jy, NULL, radint_mem[3]);
      rpn_store(beamline->radIntegrals.Jdelta, NULL, radint_mem[4]);
      rpn_store(beamline->radIntegrals.taux, NULL, radint_mem[5]);
      rpn_store(beamline->radIntegrals.tauy, NULL, radint_mem[6]);
      rpn_store(beamline->radIntegrals.taudelta, NULL, radint_mem[7]);
      rpn_store(beamline->radIntegrals.Uo, NULL, radint_mem[8]);
      for (i = 0; i < 5; i++)
	rpn_store(beamline->radIntegrals.RI[i], NULL, radint_mem[i + 9]);
#if DEBUG
      printf("Radiation integrals stored.\n");
      fflush(stdout);
#endif
    } else {
      for (i = 0; i < 14; i++)
	rpn_store(DBL_MAX/2, NULL, radint_mem[i]);
    }
  }

#if DEBUG
  printf("Starting moments_output if requested\n");
  fflush(stdout);
#endif
  runMomentsOutput(run, beamline, startingOrbitCoord, 1, 0);

#if DEBUG
  printMessageAndTime(stdout, "Starting tune_footprint if requested\n");
#endif
  if (doTuneFootprint(run, control, startingOrbitCoord, beamline, &tuneFP)) {
#if DEBUG
    printMessageAndTime(stdout, "Done computing tune footprint\n");
#endif
    if (tuneFootprintMem[0] == -1) {
      for (i = 0; i < 20; i++)
        tuneFootprintMem[i] = rpn_create_mem(tuneFootprintName[i], 0);
    }
    rpn_store(tuneFP.chromaticTuneRange[0], NULL, tuneFootprintMem[0]);
    rpn_store(tuneFP.chromaticTuneRange[1], NULL, tuneFootprintMem[1]);
    rpn_store(tuneFP.deltaRange[2], NULL, tuneFootprintMem[2]);
    rpn_store(tuneFP.amplitudeTuneRange[0], NULL, tuneFootprintMem[3]);
    rpn_store(tuneFP.amplitudeTuneRange[1], NULL, tuneFootprintMem[4]);
    rpn_store(tuneFP.positionRange[0], NULL, tuneFootprintMem[5]);
    rpn_store(tuneFP.positionRange[1], NULL, tuneFootprintMem[6]);
    rpn_store(tuneFP.chromaticDiffusionMaximum, NULL, tuneFootprintMem[7]);
    rpn_store(tuneFP.amplitudeDiffusionMaximum, NULL, tuneFootprintMem[8]);
    rpn_store(tuneFP.xyArea, NULL, tuneFootprintMem[9]);
    rpn_store(tuneFP.nuxChromLimit[0], NULL, tuneFootprintMem[10]);
    rpn_store(tuneFP.nuxChromLimit[1], NULL, tuneFootprintMem[11]);
    rpn_store(tuneFP.nuyChromLimit[0], NULL, tuneFootprintMem[12]);
    rpn_store(tuneFP.nuyChromLimit[1], NULL, tuneFootprintMem[13]);
    rpn_store(tuneFP.nuxAmpLimit[0], NULL, tuneFootprintMem[14]);
    rpn_store(tuneFP.nuxAmpLimit[1], NULL, tuneFootprintMem[15]);
    rpn_store(tuneFP.nuyAmpLimit[0], NULL, tuneFootprintMem[16]);
    rpn_store(tuneFP.nuyAmpLimit[1], NULL, tuneFootprintMem[17]);
    rpn_store(tuneFP.chrom1[0], NULL, tuneFootprintMem[18]);
    rpn_store(tuneFP.chrom1[1], NULL, tuneFootprintMem[19]);
    if (optimization_data->verbose > 1) {
      for (i = 0; i < 20; i++)
        printf("%s = %le\n", tuneFootprintName[i], rpn_recall(tuneFootprintMem[i]));
    }

#if DEBUG
    printf("Done setting values from tune footprint\n");
    fflush(stdout);
#endif
  }

  if (floorCoord_mem[0] == -1)
    for (i = 0; i < 7; i++)
      floorCoord_mem[i] = rpn_create_mem(floorCoord_name[i], 0);
  if (floorStat_mem[0] == -1)
    for (i = 0; i < 6; i++)
      floorStat_mem[i] = rpn_create_mem(floorStat_name[i], 0);
  final_floor_coordinates(beamline, XYZ, Angle, XYZMin, XYZMax);
  for (i = 0; i < 3; i++) {
    rpn_store(XYZ[i], NULL, floorCoord_mem[i]);
    rpn_store(Angle[i], NULL, floorCoord_mem[i + 3]);
    rpn_store(XYZMin[i], NULL, floorStat_mem[i]);
    rpn_store(XYZMax[i], NULL, floorStat_mem[i + 3]);
  }
#if DEBUG
  printf("Floor coordinates stored.\n");
  fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
  printf("Floor coordinates stored.\n");
  fflush(stdout);
#endif

  compute_end_positions(beamline);
  rpn_store(beamline->revolution_length, NULL, floorCoord_mem[6]);

  for (i = 0; i < variables->n_variables; i++)
    variables->varied_quan_value[i] = value[i];

  if (!*invalid) {
    output->n_z_points = 0;
    M = accumulate_matrices(beamline->elem, run, NULL,
                            optimization_data->matrix_order < 1 ? 1 : optimization_data->matrix_order, 0);

#if USE_MPI && MPI_DEBUG
  printf("Accumulated matrices\n");
  fflush(stdout);
#endif

#if SDDS_MPI_IO
    if (isMaster && notSinglePart)
      if (beam->n_to_track_total < (n_processors - 1)) {
        printf("*************************************************************************************************\n");
        printf("* Warning! The number of particles (%ld) shouldn't be less than the number of processors (%d)! *\n", beam->n_to_track, n_processors - 1);
        printf("* Use of fewer processors is recommended!                                                       *\n");
        printf("*************************************************************************************************\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
#endif
    if (center_on_orbit)
      center_beam_on_coords(beam->particle, beam->n_to_track, startingOrbitCoord, center_momentum_also);
    if (!inhibit_tracking) {
      if (optimization_data->verbose && optimization_data->fp_log) {
        fprintf(optimization_data->fp_log, "Tracking for optimization\n");
        fflush(optimization_data->fp_log);
      }
#if USE_MPI && MPI_DEBUG
      printf("About to track beam. parallelStatus = %d, partOnMaster = %d, notSinglePart = %ld, runInSinglePartMode = %ld\n",
             parallelStatus, partOnMaster, notSinglePart, runInSinglePartMode);
      fflush(stdout);
#endif
#if DEBUG
      printMessageAndTime(stdout, "About to track beam\n");
      fflush(stdout);
#endif
      if (output->sums_vs_z) {
        free(output->sums_vs_z);
        output->sums_vs_z = NULL;
      }
      track_beam(run, control, error, variables, beamline, beam, output, optim_func_flags, 1,
                 &charge);
#if DEBUG
      printMessageAndTime(stdout, "Done tracking beam\n");
#endif
#if USE_MPI && MPI_DEBUG
      printf("Done tracking beam.\n");
      fflush(stdout);
#endif
      /* Store number of lost particles in memory for use in optimization expressions */
#if USE_MPI
      if (optimization_data->method != OPTIM_METHOD_HYBSIMPLEX && optimization_data->method != OPTIM_METHOD_SWARM &&
          optimization_data->method != OPTIM_METHOD_GENETIC) {
        MPI_Allreduce(&beam->n_lost, &nLostTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        /*
          printf("nLostTotal = %ld\n", nLostTotal);
          fflush(stdout);
        */
        rpn_store(nLostTotal, NULL, nLostMemory);
      } else
        rpn_store(beam->n_lost, NULL, nLostMemory);
#else
      rpn_store(beam->n_lost, NULL, nLostMemory);
#endif
    }

    if (doFindAperture) {
      double area;
      do_aperture_search(run, control, startingOrbitCoord, error, beamline, &area);
#if USE_MPI
      double area1;
      area1 = area;
      MPI_Allreduce(&area1, &area, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#if MPI_DEBUG
      printf("DA area is %le\n", area);
      fflush(stdout);
#endif
#endif
      rpn_store(area, NULL, rpn_create_mem("DaArea", 0));
      if (optimization_data->verbose)
        printf("DA area is %e\n", area);
    }
  }

  if (!*invalid) {
    if (output->sasefel.active)
      storeSASEFELAtEndInRPN(&(output->sasefel));

      /* compute final parameters and store in rpn memories */
#if DEBUG
    printMessageAndTime(stdout, "Computing final parameters\n");
#endif
    if (!output->sums_vs_z)
      bombElegant("sums_vs_z element of output structure is NULL--programming error (optimization_function)", NULL);
#if USE_MPI
#if MPI_DEBUG
      printf("computing final properties.\n");
      fflush(stdout);
#endif

    if (notSinglePart)
      beamNoToTrack = beam->n_to_track_total;
    else
      beamNoToTrack = beam->n_to_track;
    if ((i = compute_final_properties(final_property_value, output->sums_vs_z + output->n_z_points,
                                      beamNoToTrack, beam->p0, M, beam->particle,
                                      control->i_step, control->indexLimitProduct * control->n_steps,
                                      charge)) != final_property_values) {
#else
    if ((i = compute_final_properties(final_property_value, output->sums_vs_z + output->n_z_points,
                                      beam->n_to_track, beam->p0, M, beam->particle,
                                      control->i_step, control->indexLimitProduct * control->n_steps,
                                      charge)) != final_property_values) {
#endif
      printf("error: compute_final_properties computed %ld quantities when %ld were expected (optimization_function)\n",
             i, final_property_values);
      fflush(stdout);
      abort();
    }

#if DEBUG
    printf("Done computing final parameters, invalid = %ld\n", *invalid);
    fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
    printf("Done computing final parameters, invalid = %ld\n", *invalid);
    fflush(stdout);
#endif

    psum = -1;
#if USE_MPI
    if (!partOnMaster)
#endif
      if (optimization_data->nParticlesToMatch) {
        psum = particleComparisonForOptimization(beam, optimization_data, invalid);
#if DEBUG
        printf("optimization_function: returned from particleComparisonForOptimization, invalid = %ld\n", *invalid);
        fflush(stdout);
#endif
      }

    if (isMaster || !notSinglePart) { /* Only the master will execute the block */
      rpn_store_final_properties(final_property_value, final_property_values);
      if (spinCoordOffset) {
	int im;
	if (spin_mem[0]==-1) {
	  for (im=0; im<3; im++)
	    spin_mem[im] = rpn_create_mem(spin_name[im], 0);
	}
	for (im=0; im<3; im++)
	  rpn_store(output->sums_vs_z[output->n_z_points].spinSums->centroid[im], NULL, spin_mem[im]);
      }
      if (optimization_data->matrix_order > 1 && !*invalid)
        rpnStoreHigherMatrixElements(M, &optimization_data->TijkMem,
                                     &optimization_data->UijklMem,
                                     optimization_data->matrix_order);
      free_matrices(M);
      free(M);
      M = NULL;

#if DEBUG
      printf("optimization_function: Checking constraints, invalid = %ld\n", *invalid);
      fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
      printf("optimization_function: Checking constraints, invalid = %ld\n", *invalid);
    fflush(stdout);
#endif
      /* check constraints */
      if (optimization_data->verbose && optimization_data->fp_log && constraints->n_constraints) {
        fprintf(optimization_data->fp_log, "    Constraints:\n");
        fflush(optimization_data->fp_log);
      }
      for (i = 0; i < constraints->n_constraints; i++) {
        if (optimization_data->verbose && optimization_data->fp_log)
          fprintf(optimization_data->fp_log, "    %10s: %23.15e", constraints->quantity[i],
                  final_property_value[constraints->index[i]]);
        if ((conval = final_property_value[constraints->index[i]]) < constraints->lower[i] ||
            conval > constraints->upper[i]) {
#if DEBUG
          printMessageAndTime(stdout, "optimization_function: constraint violated\n");
#endif
          *invalid = 1;
          if (optimization_data->verbose && optimization_data->fp_log) {
            fprintf(optimization_data->fp_log, " ---- invalid\n\n");
            fflush(optimization_data->fp_log);
          }
          log_exit("optimization_function");
          break;
        }
        if (optimization_data->verbose && optimization_data->fp_log) {
          fputc('\n', optimization_data->fp_log);
          fflush(optimization_data->fp_log);
        }
      }

#if DEBUG
      printf("optimization_function: Computing rpn function, invalid = %ld\n", *invalid);
      fflush(stdout);
#endif
#if USE_MPI && MPI_DEBUG
      printf("optimization_function: Computing rpn function, invalid = %ld\n", *invalid);
    fflush(stdout);
#endif
      result = 0;
      rpn_clear();
#if USE_MPI
      if (partOnMaster)
#endif
        if (optimization_data->nParticlesToMatch) {
#if DEBUG
          printMessageAndTime(stdout, "optimization_function: Computing sum over particle coordinates\n");
#endif
          psum = particleComparisonForOptimization(beam, optimization_data, invalid);
#if DEBUG
          printf("psum =  %le, invalid = %ld\n", psum, *invalid);
          fflush(stdout);
#endif
        }
      if (!*invalid) {
        long i = 0, terms = 0;
        double value, sum, min, max, sum2, sumAbs;
        initializeOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max);
        if (balanceTerms && optimization_data->balance_terms && optimization_data->terms) {
          for (i = 0; i < optimization_data->terms; i++) {
            rpn_clear();
            if ((value = rpn(optimization_data->term[i])) != 0) {
              optimization_data->termWeight[i] = 1 / fabs(value);
              terms++;
            } else
              optimization_data->termWeight[i] = 0;
            updateOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max, value);
            if (rpn_check_error()) {
              printf("Problem evaluating expression: %s\n", optimization_data->term[i]);
              rpn_clear_error();
              rpnError++;
            }
          }
          if (rpnError) {
            printf("RPN expression errors prevent balancing terms\n");
            exitElegant(1);
          }
          if (terms)
            for (i = 0; i < optimization_data->terms; i++) {
              if (optimization_data->termWeight[i])
                optimization_data->termWeight[i] *= optimization_data->usersTermWeight[i] * sum / terms;
              else
                optimization_data->termWeight[i] = optimization_data->usersTermWeight[i] * sum / terms;
            }
          balanceTerms = 0;
          printf("\nOptimization terms balanced.\n");
          fflush(stdout);
        } else {
          for (i = 0; i < optimization_data->terms; i++)
            optimization_data->termWeight[i] = optimization_data->usersTermWeight[i];
        }

        /* compute and return quantity to be optimized */
        if (optimization_data->terms) {
          long i;
          double sum, min, max, sum2, sumAbs;
          initializeOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max);
          if (psum >= 0)
            updateOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max, psum);
          for (i = 0; i < optimization_data->terms; i++) {
            rpn_clear();
            value = optimization_data->termValue[i] = optimization_data->termWeight[i] * rpn(optimization_data->term[i]);
            updateOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max, value);
            if (rpn_check_error()) {
              printf("Problem evaluating expression: %s\n", optimization_data->term[i]);
              rpn_clear_error();
              rpnError++;
            }
          }
          result = chooseOptimizationStatistic(sum, sumAbs, sum2, min, max, optimization_data->statistic);
        } else {
          double sum, min, max, sum2, sumAbs;
          initializeOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max);
          if (psum >= 0)
            updateOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max, psum);
          rpn_clear(); /* clear rpn stack */
          updateOptimizationStatistics(&sum, &sumAbs, &sum2, &min, &max, rpn(optimization_data->UDFname));
          result = chooseOptimizationStatistic(sum, sumAbs, sum2, min, max, optimization_data->statistic);
          if (rpn_check_error()) {
            printf("Problem evaluating expression: %s\n", optimization_data->term[i]);
            rpnError++;
          }
        }
        if (rpnError) {
          printf("RPN expression errors prevent optimization\n");
          exitElegant(1);
        }
        if (isnan(result) || isinf(result)) {
          *invalid = 1;
        } else {
          /* #if !USE_MPI  The information here is from a processor locally, we print the result across all the processors after an iteration */
          if (optimization_data->verbose && optimization_data->fp_log) {
            fprintf(optimization_data->fp_log, "equation evaluates to %23.15e\n", result);
            fflush(optimization_data->fp_log);
            if (optimization_data->terms && !*invalid) {
              fprintf(optimization_data->fp_log, "Terms of equation: \n");
              if (optimization_data->nParticlesToMatch)
                fprintf(optimization_data->fp_log, "1*(particle comparison): %23.15e\n",
                        psum);
              for (i = 0; i < optimization_data->terms; i++) {
                rpn_clear();
                fprintf(optimization_data->fp_log, "%g*(%20s): %23.15e\n",
                        optimization_data->termWeight[i],
                        optimization_data->term[i],
                        optimization_data->termValue[i]);
              }
            }
            fprintf(optimization_data->fp_log, "\n\n");
          }
          /* #endif */
        }
      }
    } /* End of Master only */

    if (*invalid) {
      result = sqrt(DBL_MAX);
      if (*invalid && optimization_data->verbose && optimization_data->fp_log) {
        fprintf(optimization_data->fp_log, "Result is invalid\n");
        fflush(optimization_data->fp_log);
      }
    }

    if (optimization_data->mode == OPTIM_MODE_MAXIMUM)
      result *= -1;

    /* copy the result into the "hidden" slot in the varied quantities array for output
       * to final properties file
       */
    variables->varied_quan_value[variables->n_variables + 1] =
      optimization_data->mode == OPTIM_MODE_MAXIMUM ? -1 * result : result;
    if (!*invalid && bestResult > result) {
      if (optimization_data->verbose && optimization_data->fp_log)
        fprintf(optimization_data->fp_log, "** Result %21.15e is new best\n", result);
      bestResult = result;
    }
    variables->varied_quan_value[variables->n_variables + 2] =
      optimization_data->mode == OPTIM_MODE_MAXIMUM ? -1 * bestResult : bestResult;

#if USE_MPI
    if (notSinglePart || enableOutput) /* Disable the beam output (except for simplex) when all the processors track independently */
#endif
      if (!*invalid && (force_output || (control->i_step - 2) % output_sparsing_factor == 0)) {
        if (center_on_orbit)
          center_beam_on_coords(beam->particle, beam->n_to_track, startingOrbitCoord, center_momentum_also);
        do_track_beam_output(run, control, error, variables, beamline, beam, output, optim_func_flags,
                             charge);
        if (optimization_data->verbose && optimization_data->fp_log) {
          fprintf(optimization_data->fp_log, "Completed post-tracking output\n");
          fflush(optimization_data->fp_log);
        }
      }
  }

#if USE_MPI
  if (notSinglePart) {
    MPI_Bcast(invalid, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  ++simplexLogStep;
  if (fpSimplexLog && (simplex_log_interval <= 0 || (simplexLogStep - 1) % simplex_log_interval == 0)) {
    fprintf(fpSimplexLog, "%ld %21.15le %21.15le %21.15le %ld 0 ", simplexLogStep, delapsed_time(), result, bestResult, *invalid);
    for (i = 0; i < variables->n_variables; i++)
      fprintf(fpSimplexLog, "%21.15le ", value[i]);
    for (i = 0; i < optimization_data->terms; i++)
      fprintf(fpSimplexLog, "%21.15le ", optimization_data->termWeight[i]*rpn(optimization_data->term[i]));
    fprintf(fpSimplexLog, "\n");
    fflush(fpSimplexLog);
  }
#endif

  storeOptimRecord(value, variables->n_variables, *invalid, result);

#if DEBUG
  printf("optimization_function: Returning %le,  invalid=%ld\n", result, *invalid);
  fflush(stdout);
#endif

  log_exit("optimization_function");
  if (*invalid)
    result = DBL_MAX;
#if USE_MPI
  if (result < bestResult && !*invalid)
    bestResult = result;
  checkTarget(bestResult, *invalid);
#endif
  return (result);
}

#if USE_MPI
void checkTarget(double myResult, long invalid) {
  MPI_Status status;
  static short *targetBuffer = NULL;
  int targetTag = 1;
  if (hybrid_simplex_comparison_interval <= 0)
    return;
#  if MPI_DEBUG
  printf("checkTarget(%le, %ld) called\n", myResult, invalid);
  fflush(stdout);
#  endif
  if (!targetBuffer)
    targetBuffer = tmalloc(sizeof(*targetBuffer) * n_processors);
  if (targetReached) {
    simplexMinAbort(1);
    return;
  }
#  if MPI_DEBUG
  printf("checkTarget (1), optimization_data->target = %le\n", optimization_data->target);
  fflush(stdout);
#  endif
  if (!invalid && optimization_data->target > myResult) {
    int i;
    MPI_Request request;
#  if MPI_DEBUG
    printf("Sending message on step %ld from %d: I dominate!\n", simplexComparisonStep, myid);
    fflush(stdout);
#  endif
    targetReached = 1;
    /* send message to other processors */
    for (i = 0; i < n_processors; i++) {
      targetBuffer[i] = targetReached; /* Isend operations can't share memory */
      if (i != myid)
        MPI_Isend(&targetBuffer[i], 1, MPI_SHORT, i, targetTag, MPI_COMM_WORLD, &request);
    }
  }
  if (!targetReached && simplexComparisonStep % hybrid_simplex_comparison_interval == 0) {
    int i, flag;
    /* check for messages from other processors */
#  if MPI_DEBUG
    printf("Probing for messages on step %ld\n", simplexComparisonStep);
    fflush(stdout);
#  endif
    targetReached = 0;
    for (i = 0; i < n_processors; i++) {
      if (i != myid) {
#  if MPI_DEBUG
        printf("Probing for message from %d\n", i);
        fflush(stdout);
#  endif
        MPI_Iprobe(i, targetTag, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
#  if MPI_DEBUG
          printf("Receiving message from %d\n", i);
          fflush(stdout);
#  endif
          MPI_Recv(&targetBuffer[i], 1, MPI_SHORT, i, targetTag, MPI_COMM_WORLD, &status);
          targetReached += targetBuffer[i];
#  if MPI_DEBUG
          printf("Message from %d is %hd\n", i, targetBuffer[i]);
          fflush(stdout);
#  endif
        } else {
#  if MPI_DEBUG
          printf("No message\n");
          fflush(stdout);
#  endif
        }
      }
    }
  }
#  if MPI_DEBUG
  printf("targetReached (global) = %hd\n", targetReached);
#  endif
  if (targetReached)
    simplexMinAbort(1);
  simplexComparisonStep++;
}
#endif

long checkForOptimRecord(double *value, long values, long *again) {
  long iRecord, iValue;
  double diff;
  if (ignoreOptimRecords)
    return -1;
  for (iRecord = 0; iRecord < optimRecords; iRecord++) {
    for (iValue = 0; iValue < values; iValue++) {
      diff = fabs(value[iValue] - optimRecord[iRecord].variableValue[iValue]);
      if (diff != 0)
        break;
    }
    if (iValue == values)
      break;
  }
  if (iRecord == optimRecords) {
    *again = 0;
    return -1;
  }
  optimRecord[iRecord].usedBefore += 1;
  *again = optimRecord[iRecord].usedBefore - 1;
  return iRecord;
}

void storeOptimRecord(double *value, long values, long invalid, double result) {
  long i;
  if (ignoreOptimRecords)
    return;
  for (i = 0; i < values; i++)
    optimRecord[nextOptimRecordSlot].variableValue[i] = value[i];
  optimRecord[nextOptimRecordSlot].usedBefore = 0;
  optimRecord[nextOptimRecordSlot].invalid = invalid;
  optimRecord[nextOptimRecordSlot].result = result;
  if (++nextOptimRecordSlot >= MAX_OPTIM_RECORDS)
    nextOptimRecordSlot = 0;
  if (++optimRecords >= MAX_OPTIM_RECORDS)
    optimRecords = MAX_OPTIM_RECORDS;
}

void optimization_report(double result, double *value, long pass, long n_evals, long n_dim) {
  OPTIM_VARIABLES *variables;
  /* OPTIM_COVARIABLES *covariables; */
  OPTIM_CONSTRAINTS *constraints;
  long i;

#if (!USE_MPI)
  if (!optimization_data->fp_log)
    return;
#endif

  /* update internal values (particularly rpn) */
  ignoreOptimRecords = 1; /* force reevaluation */
  result = optimization_function(value, &i);
  ignoreOptimRecords = 0;

#if (USE_MPI)
  if (!optimization_data->fp_log)
    return;
#endif

  variables = &(optimization_data->variables);
  /* covariables = &(optimization_data->covariables); */
  constraints = &(optimization_data->constraints);

#if !USE_MPI
  fprintf(optimization_data->fp_log, "Optimization pass %ld completed:\n    optimization function has value %23.15e\n",
          pass, optimization_data->mode == OPTIM_MODE_MAXIMUM ? -result : result);
#else
  fprintf(optimization_data->fp_log, "Optimization restart %ld completed:\n    optimization function has value %23.15e\n",
          pass, optimization_data->mode == OPTIM_MODE_MAXIMUM ? -result : result);
#endif
  n_passes_made = pass;
  if (optimization_data->terms) {
    fprintf(optimization_data->fp_log, "Terms of equation: \n");
    for (i = 0; i < optimization_data->terms; i++) {
      rpn_clear();
      fprintf(optimization_data->fp_log, "%g*(%20s): %23.15e\n",
              optimization_data->termWeight[i], optimization_data->term[i],
              rpn(optimization_data->term[i]) * optimization_data->termWeight[i]);
    }
  }

  if (constraints->n_constraints) {
    fprintf(optimization_data->fp_log, "    Constraints:\n");
    for (i = 0; i < constraints->n_constraints; i++)
      fprintf(optimization_data->fp_log, "    %10s: %23.15e\n", constraints->quantity[i],
              final_property_value[constraints->index[i]]);
  }

  fprintf(optimization_data->fp_log, "    Values of variables:\n");
  for (i = 0; i < n_dim; i++)
    fprintf(optimization_data->fp_log, "    %10s: %23.15e %s\n", variables->varied_quan_name[i], value[i],
            variableAtLimit(value[i], variables->lower_limit[i], variables->upper_limit[i]) ? "(near limit)" : "");
  fflush(optimization_data->fp_log);
}

void rpnStoreHigherMatrixElements(VMATRIX *M, long **TijkMem, long **UijklMem, long maxOrder) {
  long order, i, j, k, l, count;
  char buffer[10];

  if (!*TijkMem && maxOrder >= 2) {
    for (i = count = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          count++;
    if (!(*TijkMem = malloc(sizeof(**TijkMem) * count)))
      bombElegant("memory allocation failure (rpnStoreHigherMatrixElements)", NULL);
    for (i = count = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++, count++) {
          sprintf(buffer, "T%ld%ld%ld", i + 1, j + 1, k + 1);
          (*TijkMem)[count] = rpn_create_mem(buffer, 0);
        }
  }
  if (!*UijklMem && maxOrder >= 3) {
    for (i = count = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          for (l = 0; l <= k; l++)
            count++;
    if (!(*UijklMem = malloc(sizeof(**UijklMem) * count)))
      bombElegant("memory allocation failure (rpnStoreHigherMatrixElements)", NULL);
    for (i = count = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          for (l = 0; l <= k; l++, count++) {
            sprintf(buffer, "U%ld%ld%ld%ld", i + 1, j + 1, k + 1, l + 1);
            (*UijklMem)[count] = rpn_create_mem(buffer, 0);
          }
  }

  for (order = 2; order <= maxOrder && order <= M->order; order++) {
    switch (order) {
    case 2:
      if (!M->T)
        bombElegant("second order matrix is missing!", NULL);
      for (i = count = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
          for (k = 0; k <= j; k++, count++)
            rpn_store(M->T[i][j][k], NULL, (*TijkMem)[count]);
      break;
    case 3:
      if (!M->Q)
        bombElegant("third order matrix is missing!", NULL);
      for (i = count = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
          for (k = 0; k <= j; k++)
            for (l = 0; l <= k; l++, count++)
              rpn_store(M->Q[i][j][k][l], NULL, (*UijklMem)[count]);
      break;
    default:
      break;
    }
  }
}

#if USE_MPI
void find_global_min_index(double *min, int *processor_ID, MPI_Comm comm) {
  struct {
    double val;
    int rank;
  } in, out;
#  ifdef MPI_DEBUG
  static int fgmiCounter = 0;
  fgmiCounter++;
  printf("Call %d to find_global_min_index...\n", fgmiCounter);
  fflush(stdout);
#  endif
  in.val = *min;
  MPI_Comm_rank(comm, &(in.rank));
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
#  ifdef MPI_DEBUG
  printf("...call succeeded\n");
  fflush(stdout);
#  endif
  *min = out.val;
  *processor_ID = out.rank;
}

void SDDS_PopulationSetup(char *population_log, SDDS_TABLE *popLogPtr, OPTIM_VARIABLES *optim, OPTIM_COVARIABLES *co_optim) {
  if (isMaster) {
    if (population_log && strlen(population_log)) {
      if (!SDDS_InitializeOutputElegant(popLogPtr, SDDS_BINARY, 1, NULL, NULL, population_log) ||
          0 > SDDS_DefineParameter(popLogPtr, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "Iteration", NULL, SDDS_LONG) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "ElapsedTime", "s", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "ElapsedCoreTime", "s", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "OptimizationValue", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "WorstOptimizationValue", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "MedianOptimizationValue", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "AverageOptimizationValue", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(popLogPtr, "OptimizationValueSpread", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameters(popLogPtr, optim->n_variables, optim->varied_quan_name,
                                       optim->varied_quan_unit, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumn(popLogPtr, "OptimizationValue", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumns(popLogPtr, optim->n_variables, optim->varied_quan_name,
                                    optim->varied_quan_unit, SDDS_DOUBLE) ||
          !SDDS_WriteLayout(popLogPtr)) {
        printf("Problem setting up population output file %s\n", population_log);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
  }
}

/* Function to print populations */
void SDDS_PrintPopulations(SDDS_TABLE *popLogPtr, double result, double *variable, long dimensions) {
  static double **individuals = NULL, *results = NULL;
  /* For both the swarm and hybrid simplex methods, only one best individual from each process will be printed */
  long pop_size = n_processors;
  long row, j;
  printf(" SDDS_PrintPopulations is called\n");
  if (!popLogPtr)
    return;
  if (!individuals)
    individuals = (double **)czarray_2d(sizeof(double), pop_size, dimensions);
  if (!results)
    results = (double *)tmalloc(sizeof(double) * pop_size);

  MPI_Gather(variable, dimensions, MPI_DOUBLE, &individuals[0][0], dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&result, 1, MPI_DOUBLE, results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (isMaster) {
    if (!SDDS_StartPage(popLogPtr, pop_size))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    for (row = 0; row < pop_size; row++) {
      if (!SDDS_SetRowValues(popLogPtr, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row, 0, results[row], -1)) {
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        fflush(stderr);
      }
      for (j = 0; j < dimensions; j++) {
        if (!SDDS_SetRowValues(popLogPtr, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row, j + 1, individuals[row][j], -1)) {
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
          fflush(stderr);
        }
      }
    }
  }
}

/* Function to print the statistics of the populations after each optimization iteration */
void SDDS_PrintStatistics(SDDS_TABLE *popLogPtr, long iteration, double best_value, double worst_value, double median, double average, double spread, double *best_individual, long dimensions, double *covariable, long n_covariables, long print_all) {
  int i;
  long offset = 9;

  if (!popLogPtr)
    return;
  if (isMaster) {
    if (!print_all)
      if (!SDDS_StartPage(popLogPtr, 1))
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    if (!SDDS_SetParameters(popLogPtr, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Iteration", iteration,
                            "ElapsedTime", delapsed_time(),
#  if USE_MPI
                            "ElapsedCoreTime", delapsed_time() * n_processors,
#  else
                            "ElapsedCoreTime", delapsed_time(),
#  endif
                            "OptimizationValue", best_value,
                            "WorstOptimizationValue", worst_value,
                            "MedianOptimizationValue", median,
                            "AverageOptimizationValue", average,
                            "OptimizationValueSpread", spread, NULL))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    for (i = 0; i < dimensions; i++) {
      if (!SDDS_SetParameters(popLogPtr, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, offset + i, best_individual[i], -1))
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }

    /* The covariable information won't be printed to be consistent with the genetic optimization
    if (n_covariables) {
      for (i=0; i<n_covariables; i++) {
	if (!SDDS_SetParameters(popLogPtr, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, offset+dimensions+i, covariable[i], -1))
	  SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
      }
    } 
    */

    if (!SDDS_WritePage(popLogPtr))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

    SDDS_DoFSync(popLogPtr);
  }
}
#endif

double particleComparisonForOptimization(BEAM *beam, OPTIMIZATION_DATA *optimData, long *invalid) {
  long i, j;

  *invalid = 0;

  /* determine if number of particles is correct and if particle IDs match up */
#if USE_MPI
  if (!partOnMaster) {
    long k, nLeft, nTotalLeft, globalInvalid;
    double particleStat[6] = {0, 0, 0, 0, 0, 0}, value;
    double globalValue;
    if (myid == 0)
      nLeft = 0;
    else
      nLeft = beam->n_to_track;
    MPI_Allreduce(&nLeft, &nTotalLeft, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fflush(stdout);
    if (nTotalLeft != optimData->nParticlesToMatch) {
      *invalid = 1;
      return 0;
    }

    for (i = 0; i < nLeft; i++) {
      for (j = 0; j < optimData->nParticlesToMatch; j++) {
        if (beam->particle[i][6] == optimData->coordinatesToMatch[j][6]) {
          for (k = 0; k < 6; k++) {
            if (optimData->particleMatchingWeight[j]) {
              if (optimData->particleMatchingMode & COMPARE_PARTICLE_MAX_ABSDEV) {
                if ((value = fabs(beam->particle[i][k] - optimData->coordinatesToMatch[j][k])) > particleStat[k])
                  particleStat[k] = value;
              } else if (optimData->particleMatchingMode & COMPARE_PARTICLE_SUM_ABSDEV) {
                particleStat[k] += fabs(beam->particle[i][k] - optimData->coordinatesToMatch[j][k]);
              } else {
                particleStat[k] += sqr(beam->particle[i][k] - optimData->coordinatesToMatch[j][k]);
              }
            }
          }
          break;
        }
      }
      if (j == optimData->nParticlesToMatch) {
        printf("No particle found for ID = %.0lf\n", beam->particle[i][6]);
        *invalid = 1;
        break;
      }
    }
    MPI_Allreduce(invalid, &globalInvalid, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (globalInvalid) {
      *invalid = 1;
      return 0;
    }
    *invalid = 0;
    for (k = 0; k < 6; k++) {
      value = particleStat[k] * optimData->particleMatchingWeight[k];
      if (optimData->particleMatchingMode & COMPARE_PARTICLE_MAX_ABSDEV)
        MPI_Allreduce(&value, &globalValue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      else
        MPI_Allreduce(&value, &globalValue, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      particleStat[k] = globalValue;
    }
    if (optimData->particleMatchingMode & COMPARE_PARTICLE_MAX_ABSDEV) {
      find_min_max(NULL, &value, particleStat, 6);
    } else {
      for (k = value = 0; k < 6; k++)
        value += particleStat[k];
    }
    return value;
  } else {
#endif
    /* In this case, each processor (maybe only 1) tracks all the particles */

    if (beam->n_to_track != optimData->nParticlesToMatch) {
      printf("Mismatch in comparison of particles after tracking: %ld needed but %ld present\n",
             optimData->nParticlesToMatch, beam->n_to_track);
      fflush(stdout);
      *invalid = 1;
      return 0;
    }

    for (i = 0; i < optimData->nParticlesToMatch; i++) {
      if (beam->particle[i][6] != optimData->coordinatesToMatch[i][6]) {
        printf("Mismatch of particle ID for particle %ld: %.0lf expected but %.0lf found\n",
               i, beam->particle[i][6], optimData->coordinatesToMatch[i][6]);
        fflush(stdout);
        *invalid = 1;
        return 0;
      }
    }

#if USE_MPI
  }
#endif

  if (optimData->particleMatchingMode & COMPARE_PARTICLE_MAX_ABSDEV) {
    /* maximum absolute deviation */
    double maxAbsDev, absDev;
    maxAbsDev = 0;
    for (j = 0; j < 6; j++) {
      if (!optimData->particleMatchingWeight[j])
        continue;
      for (i = 0; i < optimData->nParticlesToMatch; i++) {
        absDev = fabs(beam->particle[i][j] - optimData->coordinatesToMatch[i][j]) * optimData->particleMatchingWeight[j];
        if (absDev > maxAbsDev)
          maxAbsDev = absDev;
      }
    }
    return maxAbsDev;
  } else if (optimData->particleMatchingMode & COMPARE_PARTICLE_SUM_ABSDEV) {
    /* sum of absolute deviations */
    double sum, sum1;
    sum = 0;
    for (j = 0; j < 6; j++) {
      if (!optimData->particleMatchingWeight[j])
        continue;
      sum1 = 0;
      for (i = 0; i < optimData->nParticlesToMatch; i++)
        sum1 += fabs(beam->particle[i][j] - optimData->coordinatesToMatch[i][j]);
      sum += sum1 * optimData->particleMatchingWeight[j];
    }
    return sum;
  } else {
    /* sum of squared deviations */
    double sum, sum1;
    sum = 0;
    for (j = 0; j < 6; j++) {
      if (!optimData->particleMatchingWeight[j])
        continue;
      sum1 = 0;
      for (i = 0; i < optimData->nParticlesToMatch; i++)
        sum1 += sqr(beam->particle[i][j] - optimData->coordinatesToMatch[i][j]);
      sum += sum1 * optimData->particleMatchingWeight[j];
    }
    return sum;
  }
}

void initializeOptimizationStatistics(double *sum, double *sumAbs, double *sum2, double *min, double *max) {
  *sum = *sum2 = *sumAbs = 0;
  *min = DBL_MAX;
  *max = -DBL_MAX;
}

void updateOptimizationStatistics(double *sum, double *sumAbs, double *sum2, double *min, double *max, double value) {
  *sum += value;
  *sumAbs += fabs(value);
  *sum2 += sqr(value);
  if (value > *max)
    *max = value;
  if (value < *min)
    *min = value;
}

double chooseOptimizationStatistic(double sum, double sumAbs, double sum2, double min, double max, long stat) {
  double result;
  switch (stat) {
  case STATISTIC_SUM_ABS:
    result = sumAbs;
    break;
  case STATISTIC_MAXIMUM:
    result = max;
    break;
  case STATISTIC_MINIMUM:
    result = min;
    break;
  case STATISTIC_SUM_SQR:
    result = sum2;
    break;
  default:
    result = sum;
    break;
  }
  return result;
}
