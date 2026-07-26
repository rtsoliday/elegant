/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* program: elegant
 * purpose: accelerator simulation
 * Michael Borland, 1989-2018
 */
#include "mdb.h"
#include "track.h"
#include "elegant.h"
#include "SDDS.h"
#include "scan.h"
#include <ctype.h>
#include "match_string.h"
#include <signal.h>
#include <time.h>
#if defined(__linux__) || defined(_WIN32)
#  include <malloc.h>
#endif
#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

#if defined(__linux__)
#  include <sched.h>
#  include <linux/version.h>
#else
#  define KERNEL_VERSION(a, b, c) 0
#endif

#include "chromDefs.h"
#include "correctDefs.h"
#include "tuneDefs.h"
#include "correctorStash.h"
#if HAVE_GPU
#  include "gpu_base.h"
#endif

void doMacroOutput(NAMELIST_TEXT *nltext, RUN *run, char **macroTag, char **macroValue, long macros);
void traceback_handler(int code);
void createSemaphoreFile(char *filename);
void manageSemaphoreFiles(char *semaphore_file, char *rootname, char *semaphoreFile[3]);
void processApertureInput(NAMELIST_TEXT *nltext, RUN *run);
void initialize_structures(RUN *run_conditions, VARY *run_control, ERRORVAL *error_control, CORRECTION *correct,
                           BEAM *beam, OUTPUT_FILES *output_data, OPTIMIZATION_DATA *optimize,
                           CHROM_CORRECTION *chrom_corr_data, TUNE_CORRECTION *tune_corr_data,
                           ELEMENT_LINKS *links);
void do_semaphore_setup(char **semaphoreFile, char *rootname, long run_setuped, NAMELIST_TEXT *nltext);
void printFarewell(FILE *fp);
void closeBeamlineOutputFiles(LINE_LIST *beamline);
void setSigmaIndices();
void process_particle_command(NAMELIST_TEXT *nltext);
void process_change_start(NAMELIST_TEXT *nltext, CHANGE_START_SPEC *css);
void process_change_end(NAMELIST_TEXT *nltext, CHANGE_END_SPEC *ces);
void freeInputObjects();
void runFiducialParticle(RUN *run, VARY *control, double *startCoord, LINE_LIST *beamline, short final, short mustSurvive);

#define DESCRIBE_INPUT 0
#define DEFINE_MACRO 1
#define DEFINE_CPU_LIST 2
#define DEFINE_PIPE 3
#define DEFINE_RPN_DEFNS 4
#define DEFINE_VERBOSE 5
#define DEFINE_CONFIGURATION 6
#if HAVE_GPU && !USE_MPI
#  define DEFINE_OMP_THREADS 7
#  define N_OPTIONS 8
#else
#  define N_OPTIONS 7
#endif
char *option[N_OPTIONS] = {
  "describeinput",
  "macro",
  "cpulist",
  "pipe",
  "rpndefns",
  "verbose",
  "configuration",
#if HAVE_GPU && !USE_MPI
  "ompthreads",
#endif
};

#define SHOW_USAGE 0x0001
#define SHOW_GREETING 0x0002

void showUsageOrGreeting(unsigned long mode) {
#if USE_MPI
#  if HAVE_GPU
  char *USAGE = "usage: mpirun -np <number of processes> gpu-Pelegant <inputfile> [-macro=<tag>=<value>,[...]] [-rpnDefns=<filename>] [-configuration=<filename>]";
  char *GREETING = "This is gpu-Pelegant 2026.4Beta1 ALPHA RELEASE, "__DATE__
    ", by M. Borland, K. Amyx, J. Calvey, M. Carla', N. Carmignani, AJ Dick, Z. Duan, M. Ehrlichman, L. Emery, W. Guo, J.R. King, N. Kuklev, R. Lindberg, I.V. Pogorelov, V. Sajaev, R. Soliday, Y.-P. Sun, M. Wallbank, C.-X. Wang, Y. Wang, Y. Wu, and A. Xiao.\nParallelized by Y. Wang, H. Shang, and M. Borland.";
#  else
  char *USAGE = "usage: mpirun -np <number of processes> Pelegant <inputfile> [-macro=<tag>=<value>,[...]] [-rpnDefns=<filename>] [-configuration=<filename>]";
  char *GREETING = "This is elegant 2026.4Beta1 "__DATE__
    ", by M. Borland, J. Calvey, M. Carla', N. Carmignani, AJ Dick, Z. Duan, M. Ehrlichman, L. Emery, W. Guo, N. Kuklev, R. Lindberg, V. Sajaev, R. Soliday, Y.-P. Sun, M. Wallbank, C.-X. Wang, Y. Wang, Y. Wu, and A. Xiao.\nParallelized by Y. Wang, H. Shang, and M. Borland.";
#  endif
#else
#  if HAVE_GPU
  char *USAGE = "usage: gpu-elegant {<inputfile>|-pipe=in} [-macro=<tag>=<value>,[...]] [-rpnDefns=<filename>] [-configuration=<filename>] [-ompThreads=<positive-integer>]";
  char *GREETING = "This is gpu-elegant 2026.4Beta1 ALPHA RELEASE, "__DATE__
    ", by M. Borland, K. Amyx, J. Calvey, M. Carla', N. Carmignani, AJ Dick, Z. Duan, M. Ehrlichman, L. Emery, W. Guo, J.R. King, N. Kuklev, R. Lindberg, I.V. Pogorelov, V. Sajaev, R. Soliday, Y.-P. Sun, M. Wallbank, C.-X. Wang, Y. Wang, Y. Wu, and A. Xiao.";
#  else
  char *USAGE = "usage: elegant {<inputfile>|-pipe=in} [-macro=<tag>=<value>,[...]] [-rpnDefns=<filename>] [-configuration=<filename>]";
  char *GREETING = "This is elegant 2026.4Beta1, "__DATE__
    ", by M. Borland, J. Calvey, M. Carla', N. Carmignani, AJ Dick, Z. Duan, M. Ehrlichman, L. Emery, W. Guo, N. Kuklev, R. Lindberg, V. Sajaev, R. Soliday, Y.-P. Sun, M. Wallbank, C.-X. Wang, Y. Wang, Y. Wu, and A. Xiao.";
#  endif
#endif
  time_t timeNow;
  char *timeNowString;
  timeNow = time(NULL);
  timeNowString = ctime(&timeNow);
#if USE_MPI
  printf("Running Pelegant on %d cores at %s\n", n_processors, timeNowString ? timeNowString : "?");
#else
  printf("Running elegant at %s\n", timeNowString ? timeNowString : "?");
#endif
  if (mode & SHOW_GREETING)
    puts(GREETING);
  if (mode & SHOW_USAGE) {
    puts(USAGE);
#if HAVE_GPU
    {
      char gpuUsage[1024];
      gpuDescribeUsageSettings(gpuUsage, sizeof(gpuUsage));
      puts(gpuUsage);
    }
#endif
  }
  printFarewell(stdout);
}

#define RUN_SETUP 0
#define RUN_CONTROL 1
#define VARY_ELEMENT 2
#define ERROR_CONTROL 3
#define ERROR_ELEMENT 4
#define SET_AWE_BEAM 5
#define SET_BUNCHED_BEAM 6
#define CORRECTION_SETUP 7
#define MATRIX_OUTPUT 8
#define TWISS_OUTPUT 9
#define TRACK 10
#define STOP 11
#define OPTIMIZATION_SETUP 12
#define OPTIMIZE_CMD 13
#define OPTIMIZATION_VARIABLE 14
#define OPTIMIZATION_CONSTRAINT 15
#define OPTIMIZATION_COVARIABLE 16
#define SAVE_LATTICE 17
#define RPN_EXPRESSION 18
#define PROGRAM_TRACE 19
#define CHROMATICITY 20
#define CLOSED_ORBIT 21
#define FIND_APERTURE 22
#define ANALYZE_MAP 23
#define CORRECT_TUNES 24
#define LINK_CONTROL 25
#define LINK_ELEMENTS 26
#define STEERING_ELEMENT 27
#define AMPLIF_FACTORS 28
#define PRINT_DICTIONARY 29
#define FLOOR_COORDINATES 30
#define CORRECTION_MATRIX_OUTPUT 31
#define LOAD_PARAMETERS 32
#define SET_SDDS_BEAM 33
#define SUBPROCESS 34
#define FIT_TRACES 35
#define SASEFEL_AT_END 36
#define ALTER_ELEMENTS 37
#define OPTIMIZATION_TERM 38
#define SLICE_ANALYSIS 39
#define DIVIDE_ELEMENTS 40
#define TUNE_SHIFT_WITH_AMPLITUDE 41
#define TRANSMUTE_ELEMENTS 42
#define TWISS_ANALYSIS 43
#define SEMAPHORES 44
#define FREQUENCY_MAP 45
#define INSERT_SCEFFECTS 46
#define MOMENTUM_APERTURE 47
#define APERTURE_INPUT 48
#define COUPLED_TWISS_OUTPUT 49
#define LINEAR_CHROMATIC_TRACKING_SETUP 50
#define RPN_LOAD 51
#define MOMENTS_OUTPUT 52
#define TOUSCHEK_SCATTER 53
#define INSERT_ELEMENTS 54
#define CHANGE_PARTICLE 55
#define GLOBAL_SETTINGS 56
#define REPLACE_ELEMENTS 57
#define APERTURE_DATAX 58
#define MODULATE_ELEMENTS 59
#define PARALLEL_OPTIMIZATION_SETUP 60
#define RAMP_ELEMENTS 61
#define RF_SETUP 62
#define CHAOS_MAP 63
#define TUNE_FOOTPRINT 64
#define ION_EFFECTS 65
#define ELASTIC_SCATTERING 66
#define INELASTIC_SCATTERING 67
#define IGNORE_ELEMENTS 68
#define SET_REFERENCE_PARTICLE_OUTPUT 69
#define OBSTRUCTION_DATA 70
#define SET_BUNCHED_BEAM_MOMENTS 71
#define CHANGE_START 72
#define CHANGE_END 73
#define INCLUDE_COMMANDS 74
#define PARTICLE_TUNES 75
#define MACRO_OUTPUT 76
#define CORRECT_COUPLING 77
#define COMPUTE_COUPLING_RESPONSE_MATRIX 78
#define LOAD_COUPLING_RESPONSE_MATRIX 79
#define CORRECT_LATTICE 80
#define COMPUTE_LATTICE_RESPONSE_MATRIX 81
#define LOAD_LATTICE_RESPONSE_MATRIX 82
#define UNDULATOR_BRIGHTNESS 83
#define LOAD_KNOBS 84
#define N_COMMANDS 85

char *command[N_COMMANDS] = {
  "run_setup",
  "run_control",
  "vary_element",
  "error_control",
  "error_element",
  "awe_beam",
  "bunched_beam",
  "correct",
  "matrix_output",
  "twiss_output",
  "track",
  "stop",
  "optimization_setup",
  "optimize",
  "optimization_variable",
  "optimization_constraint",
  "optimization_covariable",
  "save_lattice",
  "rpn_expression",
  "trace",
  "chromaticity",
  "closed_orbit",
  "find_aperture",
  "analyze_map",
  "correct_tunes",
  "link_control",
  "link_elements",
  "steering_element",
  "amplification_factors",
  "print_dictionary",
  "floor_coordinates",
  "correction_matrix_output",
  "load_parameters",
  "sdds_beam",
  "subprocess",
  "fit_traces",
  "sasefel",
  "alter_elements",
  "optimization_term",
  "slice_analysis",
  "divide_elements",
  "tune_shift_with_amplitude",
  "transmute_elements",
  "twiss_analysis",
  "semaphores",
  "frequency_map",
  "insert_sceffects",
  "momentum_aperture",
  "aperture_input",
  "coupled_twiss_output",
  "linear_chromatic_tracking_setup",
  "rpn_load",
  "moments_output",
  "touschek_scatter",
  "insert_elements",
  "change_particle",
  "global_settings",
  "replace_elements",
  "aperture_data",
  "modulate_elements",
  "parallel_optimization_setup",
  "ramp_elements",
  "rf_setup",
  "chaos_map",
  "tune_footprint",
  "ion_effects",
  "elastic_scattering",
  "inelastic_scattering",
  "ignore_elements",
  "set_reference_particle_output",
  "obstruction_data",
  "bunched_beam_moments",
  "change_start",
  "change_end",
  "include_commands",
  "particle_tunes",
  "macro_output",
  "correct_coupling",
  "compute_coupling_response_matrix",
  "load_coupling_response_matrix",
  "correct_lattice",
  "compute_lattice_response_matrix",
  "load_lattice_response_matrix",
  "undulator_brightness",
  "load_knobs",
};

char *description[N_COMMANDS] = {
  "run_setup                        defines lattice input file, primary output files, tracking order, etc.",
  "run_control                      defines number of steps, number of passes, number of indices, etc.",
  "vary_element                     defines element family and item to vary with an index",
  "error_control                    sets up and control random errors",
  "error_element                    defines an element family and item to add errors to",
  "awe_beam                         defines name of input beam data file, type of data, and some preprocessing",
  "bunched_beam                     defines beam distribution",
  "correct                          requests orbit or trajectory correction and define parameters",
  "matrix_output                    requests SDDS-format output or printed output of the matrix",
  "twiss_output                     requests output of Twiss parameters, chromaticity, and acceptance",
  "track                            command to begin tracking",
  "stop                             command to stop reading input file and end the run",
  "optimization_setup               requests running of optimization mode and sets it up",
  "optimize                         command to begin optimization",
  "optimization_variable            defines an element family and item to vary for optimization",
  "optimization_constraint          defines a constraint on optimization",
  "optimization_covariable          defines an element family and item to compute from optimization variables",
  "optimization_term                specifies an individual term in the optimization equation",
  "save_lattice                     command to save the current lattice",
  "rpn_expression                   command to execute an rpn expression (useful for optimization)",
  "trace                            requests tracing of program calls and defines parameters of trace",
  "chromaticity                     requests correction of the chromaticity",
  "closed_orbit                     requests output of the closed orbit",
  "find_aperture                    command to do aperture searches",
  "analyze_map                      command to do map analysis",
  "correct_tunes                    requests correction of the tunes",
  "link_control                     sets up and control element links",
  "link_elements                    defines a link between two sets of elements",
  "steering_element                 defines an element (group) and item for steering the beam",
  "amplification_factors            computes orbit/trajectory amplification factors",
  "print_dictionary                 prints a list of accelerator element types and allowed parameters",
  "floor_coordinates                computes the floor coordinates for the ends of elements",
  "correction_matrix_output         prints response matrices and their inverses",
  "load_parameters                  sets up loading of parameter values for elements",
  "sdds_beam                        defines name of input beam data file",
  "subprocess                       executes a string in a sub-shell",
  "fit_traces                       obtains a lattice model by fitting to multiple tracks through a beamline",
  "sasefel                          computes parameters of SASE FEL at end of system",
  "alter_elements                   alters a common parameter for one or more elements",
  "slice_analysis                   computes and outputs slice analysis of the beam",
  "divide_elements                  sets up parser to automatically divide specified elements into parts",
  "tune_shift_with_amplitude        sets up twiss module for computation of tune shift with amplitude",
  "transmute_elements               defines transmutation of one element type into another",
  "twiss_analysis                   requests twiss analysis of regions of a beamline (for optimization)",
  "semaphores                       requests use of semaphore files to indicate run start and end",
  "frequency_map                    command to perform frequency map analysis",
  "insert_sceffects                 add space charge element to beamline and set calculation flags",
  "momentum_aperture                determine momentum aperture from tracking as a function of position in the ring",
  "aperture_data                    provide an SDDS file with the physical aperture vs s",
  "coupled_twiss_output             compute coupled beamsizes and twiss parameters",
  "linear_chromatic_tracking_setup  set up chromatic derivatives for linear chromatic tracking",
  "rpn_load                         load SDDS data into rpn variables",
  "moments_output                   perform moments computations and output to file",
  "touschek_scatter                 calculate Touschek lifetime, simulate touschek scattering effects, find out momentum aperture through tracking",
  "insert_elements                  insert elements into already defined beamline",
  "change_particle                  change the particle type",
  "global_settings                  change various global settings",
  "replace_elements                 remove or replace elements inside beamline",
  "aperture_input                   provide an SDDS file with the physical aperture vs s (same as aperture_data)",
  "modulate_elements                modulate values of elements as a function of time",
  "parallel_optimization_setup      requests running of parallel optimization mode and sets it up",
  "ramp_elements                    ramp values of elements as a function of pass",
  "rf_setup                         set rf cavity frequency, phase, and voltage for ring simulation",
  "chaos_map                        command to perform chaos map analysis",
  "tune_footprint                   command to perform tune footprint tracking",
  "ion_effects                      command to set up modeling of ion effects",
  "elastic_scattering               determine x', y' aperture from tracking as a function of position in the ring",
  "inelastic_scattering             determine inelastic scattering aperture from tracking as a function of position in the ring",
  "ignore_elements                  declare that elements with certain names and types should be ignored in tracking",
  "set_reference_particle_output    set reference particle coordinates to be matched via optimization",
  "obstruction_data                 set (Z,X) contours of obstructions in the vertical midplane",
  "bunched_beam_moments             defines beam distribution using user-supplied beam moments",
  "change_start                     modify the beamline to start at a named location",
  "change_end                       modify the beamline to end at a named location",
  "include_commands                 include commands from a file",
  "particle_tunes                   accumulate data and find tunes for each particle in a multi-particle beam",
  "macro_output                     output values of commandline macros to a file",
  "correct_coupling                 correct vertical dispersion by adjusting skew quadrupoles (LOCO-style)",
  "compute_coupling_response_matrix  compute and save the response matrix for coupling correction",
  "load_coupling_response_matrix  load a previously-saved coupling-correction response matrix",
  "correct_lattice                 correct linear lattice (beta_x, beta_y, eta_x) by adjusting normal quadrupoles",
  "compute_lattice_response_matrix compute and save the response matrix for lattice correction",
  "load_lattice_response_matrix    load a previously-saved lattice-correction response matrix",
  "undulator_brightness            compute Lindberg undulator brightness as an RPN scalar (one per invocation)",
  "load_knobs                      load knob definitions from SDDS; each knob is a scalar usable as a vary/optimization/error target"};

#define NAMELIST_BUFLEN 65536

#define DEBUG 0

static VARY run_control;
static RUN run_conditions;
static char *semaphoreFile[3];

long writePermitted = 1;
/* For serial version, isMaster and isSlave are both set to 1 */
long isMaster = 1;
long isSlave = 1;
long distributedBeam = 0;       /* All the processors will do the same thing by default */
long independentRunPerRank = 0; /* This flag will be set as true under some special cases, such as genetic optimization */
double memDistFactor = 1.0;   /* In serial version, the memory will be allocted for all the particles */
long do_find_aperture = 0;

long remaining_sequence_No, orig_sequence_No; /* For Pelegant regression test */

#if USE_MPI
int n_processors = 1;
int myid;
int dumpAcceptance = 0;
parallelMode parallelStatus = trueParallel;
int partOnMaster = 1; /* indicate if the particle information is available on master */
long watch_not_allowed = 0;
long enableOutput = 0;    /* This flag is used to enforce output for the simplex method in Pelegant  */
long lessPartAllowed = 0; /* By default, the number of particles is required to be at least n_processors-1 */
MPI_Comm workers;
int fdStdout; /* save the duplicated file descriptor stdout to use it latter */
long last_optimize_function_call = 0;
int min_value_location = 0;
MPI_Group world_group, worker_group;
int ranks[1];
#endif

#ifdef SET_DOUBLE
void set_fpu(unsigned int mode) {
  __asm__("fldcw %0"
          :
          : "m"(*&mode));
}
#endif

int run_coupled_twiss_output(RUN *run, LINE_LIST *beamline, double *starting_coord);
void finish_coupled_twiss_output();
void run_rpn_load(NAMELIST_TEXT *nltext, RUN *run);
void setup_undulator_brightness(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void setup_load_knobs(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void setupLinearChromaticTracking(NAMELIST_TEXT *nltext, LINE_LIST *beamline);
void setup_coupled_twiss_output(NAMELIST_TEXT *nltext, RUN *run,
                                LINE_LIST *beamline, long *do_coupled_twiss_output,
                                long default_order);
void setup_correct_coupling(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long do_correct_coupling(RUN *run, LINE_LIST *beamline);
void finish_correct_coupling(void);
void setup_compute_coupling_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void setup_load_coupling_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void setup_correct_lattice(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long do_correct_lattice(RUN *run, LINE_LIST *beamline);
void finish_correct_lattice(void);
void setup_compute_lattice_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void setup_load_lattice_response_matrix(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void reset_alter_specifications();
void finishCorrectionOutput();
void setupElasticScattering(NAMELIST_TEXT *nltext, RUN *run, VARY *control, long twissFlag);
long runElasticScattering(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline, double *startingCoord);
void finishElasticScattering();
void setupInelasticScattering(NAMELIST_TEXT *nltext, RUN *run, VARY *control);
long runInelasticScattering(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline, double *startingCoord);
void finishInelasticScattering();

char *getNamelist(char *s, long n, FILE *fp, long *errorCode) {
#if USE_MPI
  if (fp==stdin) {
    /* only the master can read stdin */
    s[0] = 0;
    if (myid==0) {
      while (!get_namelist_e(s, NAMELIST_BUFLEN, fp, errorCode)) {
        s[0] = 0;
      }
      fprintf(stderr, "namelist error code = %ld\n", *errorCode);
    }
    MPI_Bcast(s, NAMELIST_BUFLEN, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(errorCode, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    return s;
  } else
#endif
    return get_namelist_e(s, NAMELIST_BUFLEN, fp, errorCode);
}

int main(int argc, char **argv)
{
  char **macroTag, **macroValue = NULL;
  long macros;
  LINE_LIST *beamline = NULL; /* pointer to root of linked list */
  FILE **fpIn = NULL;
  char *inputfile, *inputFileArray[2] = {NULL, NULL};
  long iInput, fpIndex, maxIncludedFiles;
  SCANNED_ARG *scanned;
  char s[NAMELIST_BUFLEN], *ptr;
  long i, verbose = 0;
  CORRECTION correct;
  ERRORVAL error_control;
  BEAM beam;
  OUTPUT_FILES *output_data;
  OPTIMIZATION_DATA optimize;
  CHROM_CORRECTION chrom_corr_data;
  TUNE_CORRECTION tune_corr_data;
  ELEMENT_LINKS links;
  char *saved_lattice = NULL;
  long correction_setuped, run_setuped, run_controled, error_controled, beam_type, commandCode;
  long do_chromatic_correction = 0, do_twiss_output = 0, fl_do_tune_correction = 0, do_coupled_twiss_output = 0;
  long fl_do_coupling_correction = 0;
  long fl_do_lattice_correction = 0;
  long do_rf_setup = 0, do_floor_coordinates = 0, sceffects_inserted = 0;
  long do_moments_output = 0;
  long do_closed_orbit = 0, do_matrix_output = 0, do_response_output = 0;
  long last_default_order = 0, new_beam_flags, twiss_computed = 0;
  /* long correctionDone, moments_computed = 0, links_present; */
  long linear_chromatic_tracking_setup_done = 0, ionEffectsSeen = 0;
  double *starting_coord = NULL, finalCharge;
  long namelists_read = 0, failed, firstPass, namelistErrorCode = 0;
  long lastCommandCode = 0;
  unsigned long pipeFlags = 0;
  double apertureReturn;
  char *rpnDefns = NULL, *configurationFile = NULL;
#if USE_MPI
#  ifdef MPI_DEBUG
  FILE *fpError;
  char fileName[15];
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int namelen;
#  endif
#endif
  CHANGE_START_SPEC changeStart = {0, NULL, -1, 0};
  CHANGE_END_SPEC changeEnd = {0, NULL, -1};
  output_data = &(run_conditions.outputFiles);

  if (0) {
    long i, j, missing = 0;
    for (i = 0; i < N_COMMANDS; i++) {
      for (j = 0; j < N_COMMANDS; j++) {
        if (strncmp(command[i], description[j], strlen(command[i])) == 0) {
          printf("Match of command[%ld]=%s and description[%ld]\n",
                 i, command[i], j);
          break;
        }
      }
      if (j == N_COMMANDS) {
        printf("No match for command[%ld]=%s\n", i, command[i]);
        missing++;
      }
    }
    printf("%ld commands are missing descriptions\n", missing);
  }

  semaphoreFile[0] = semaphoreFile[1] = semaphoreFile[2] = NULL;

  if (!SDDS_CheckDatasetStructureSize(sizeof(SDDS_DATASET)))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);

#if USE_MPI
  MPI_Init(&argc, &argv);
  /* get the total number of processors */
  MPI_Comm_size(MPI_COMM_WORLD, &n_processors);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (n_processors <= 1) {
    // NOTE: MPICH will eat the bomb elegant error and instead print weird "Fatal error in PMPI_Comm_free"
    isSlave = 0;
    bombElegant("You must specify at least 2 processors to run Pelegant.", "mpiexec -np N Pelegant, where N is greater than 1.");
  }
  
  /* create a new communicator group with the slave processors only */
  ranks[0] = 0; /* first process is master */
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_excl(world_group, 1, ranks, &worker_group);
  MPI_Comm_create(MPI_COMM_WORLD, worker_group, &workers);

  if (myid != 0) {
#  ifdef MPI_DEBUG
    sprintf(fileName, "error.%04d", myid);
    fpError = fopen(fileName, "w");
    freopen(fileName, "w", stdout);
    freopen(fileName, "w", stderr);
    MPI_Get_processor_name(processor_name, &namelen);
    printf("Process %d on %s\n", myid, processor_name);
#  else
    /* duplicate an open file descriptor, tested with gcc */
    /* redirect output, only the master processor and first slave will write on screen */
    fdStdout = dup(fileno(stdout)); /* duplicate and save stdout in case needed */
#    if defined(_WIN32)
    freopen("NUL", "w", stdout);
    /* freopen("NUL","w",stderr); */
#    else
    if (!freopen("/dev/null", "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
    /*    freopen("/dev/null","w",stderr); */
#    endif
#  endif
    writePermitted = isMaster = 0;
    isSlave = 1;
  } else
    isSlave = 0;

  if (sizeof(int) < 4) { /* The size of integer is assumed to be 4 bytes to handle a large number of particles */
    printWarning("The INT_MAX could be too small to record the number of particles.", NULL);
  }
#  if !SDDS_MPI_IO
  if (isSlave && (n_processors > 3))          /* This will avoid wasting memory on a laptop with a small number of cores */
    memDistFactor = 2.0 / (n_processors - 1); /* In parallel version, a portion of memory will be allocated on each slave */
#  endif
#endif

#ifdef SET_DOUBLE
  set_fpu(0x27F); /* use double-precision rounding */
#endif

  signal(SIGINT, traceback_handler);
  signal(SIGILL, traceback_handler);
  signal(SIGABRT, traceback_handler);
  signal(SIGFPE, traceback_handler);
  signal(SIGSEGV, traceback_handler);
#ifndef _WIN32
  signal(SIGHUP, traceback_handler);
  signal(SIGQUIT, traceback_handler);
  signal(SIGTRAP, traceback_handler);
  signal(SIGBUS, traceback_handler);
#endif

  log_entry("main");
  if (!SDDS_CheckTableStructureSize(sizeof(SDDS_TABLE))) {
    fprintf(stderr, "table structure size is inconsistent\n");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  load_hash = NULL;
  compute_offsets();
  set_max_name_length(100);
  setSigmaIndices();

  macros = 1;
  if (!(macroTag = malloc(sizeof(*macroTag))) ||
      !(macroValue = malloc(sizeof(*macroValue))))
    bombElegant("memory allocation failure setting up default macro tag/value pairs", NULL);
  macroTag[0] = "INPUTFILENAME";
  macroValue[0] = NULL; /* will fill in later */

  init_stats();

  argc = scanargs(&scanned, argc, argv);
  if (argc < 2) {
    showUsageOrGreeting(SHOW_USAGE | SHOW_GREETING);
    fflush(stdout);
    link_date();
    exitElegant(0);
  }

  showUsageOrGreeting(SHOW_GREETING);
  fflush(stdout);
  link_date();

  inputfile = inputFileArray[0] = inputFileArray[1] = NULL;
  for (i = 1; i < argc; i++) {
    if (scanned[i].arg_type == OPTION) {
      switch (match_string(scanned[i].list[0], option, N_OPTIONS, 0)) {
      case DEFINE_VERBOSE:
        printf("Set to verbose mode for argument processing\n");
        verbose = 1;
        break;
      case DESCRIBE_INPUT:
        show_namelists_fields(stdout, namelist_pointer, namelist_name, n_namelists);
        if (argc == 2)
          exitElegant(0);
        break;
      case DEFINE_PIPE:
        if (!processPipeOption(scanned[i].list + 1, scanned[i].n_items - 1, &pipeFlags))
          bombElegant("invalid -pipe syntax", NULL);
        break;
      case DEFINE_MACRO:
        if ((scanned[i].n_items -= 1) < 1) {
          printf("Invalid -macro syntax\n");
          showUsageOrGreeting(SHOW_USAGE);
          exitElegant(1);
        }
        if (!(macroTag = SDDS_Realloc(macroTag, sizeof(*macroTag) * (macros + scanned[i].n_items))) ||
            !(macroValue = SDDS_Realloc(macroValue, sizeof(*macroValue) * (macros + scanned[i].n_items))))
          bombElegant("memory allocation failure (-macro)", NULL);
        else {
          long j;
          for (j = 0; j < scanned[i].n_items; j++) {
            macroTag[macros] = scanned[i].list[j + 1];
            if (!(macroValue[macros] = strchr(macroTag[macros], '='))) {
              printf("Invalid -macro syntax\n");
              showUsageOrGreeting(SHOW_USAGE);
              exitElegant(1);
            }
            macroValue[macros][0] = 0;
            macroValue[macros] += 1;
            if (verbose) {
#if USE_MPI
              if (myid == 0)
#endif
                printf("macro: %s --> %s\n", macroTag[macros], macroValue[macros]);
            }
            macros++;
          }
        }
        break;
      case DEFINE_CPU_LIST:
#if (!USE_MPI && defined(__linux__))
#  if (LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 0))
        if (scanned[i].n_items < 2) {
          printf("Invalid -cpuList syntax\n");
          showUsageOrGreeting(SHOW_USAGE);
          exitElegant(1);
        } else {
          long j, k;
          unsigned long processorMask = 0;
          for (j = 1; j < scanned[i].n_items; j++) {
            if (sscanf(scanned[i].list[j], "%ld", &k) != 1 && k < 0) {
              printf("Invalid -cpuList syntax\n");
              showUsageOrGreeting(SHOW_USAGE);
              exitElegant(1);
            }
            processorMask |= (unsigned long)(ipow(2, k) + 0.5);
          }
          /*
            This code isn't valid with the current version of sched_setaffinity
            printf("processorMask = %lx\n", processorMask);
            sched_setaffinity(0, sizeof(processorMask), &processorMask);
          */
        }
#  else  /* (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0)) */
        printWarning("CPU list ignored", "Incompatible with Linux kernal version");
#  endif /* (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,0)) */
#else    /*(!USE_MPI && defined(linux)) */
        printWarning("CPU list ignored", "Incompatible with MPI version");
#endif   /*(!USE_MPI && defined(linux)) */
        break;
      case DEFINE_RPN_DEFNS:
        if (scanned[i].n_items != 2 || !strlen(rpnDefns = scanned[i].list[1]))
          bombElegant("invalid -rpnDefns syntax", NULL);
        break;
      case DEFINE_CONFIGURATION:
        if (scanned[i].n_items != 2 || !strlen(configurationFile = scanned[i].list[1]))
          bombElegant("invalid -configurationFile syntax", NULL);
        break;
#if HAVE_GPU && !USE_MPI
      case DEFINE_OMP_THREADS:
        {
          long ompThreads;
          char trailing;
          if (scanned[i].n_items != 2 ||
              sscanf(scanned[i].list[1], "%ld%c",
                     &ompThreads, &trailing) != 1 ||
              ompThreads < 1)
            bombElegant("invalid -ompThreads syntax: expected a positive integer",
                        NULL);
          if (!gpuSetOmpTrackingThreads(ompThreads))
            bombElegant("-ompThreads requires OpenMP support in gpu-elegant",
                        NULL);
          if (ompThreads > 1)
            fprintf(stderr,
                    "gpu-elegant: experimental loss-sensitive CPU tracking "
                    "enabled with %ld OpenMP threads. Use only on a GPU node "
                    "whose requested CPU cores are not shared with other jobs.\n",
                    ompThreads);
        }
        break;
#endif
      default:
        printf("Unknown option given.\n");
        showUsageOrGreeting(SHOW_USAGE);
        break;
      }
    } else {
      /* filenames */
      if (!inputfile)
        inputfile = scanned[i].list[0];
      else {
        printf("Too many file names listed.\n");
        showUsageOrGreeting(SHOW_USAGE);
        exitElegant(1);
      }
    }
  }

  if (!rpnDefns) {
    if (getenv("RPN_DEFNS")) {
      rpn(getenv("RPN_DEFNS"));
      if (rpn_check_error()) {
        bombElegant("RPN_DEFNS environment variable invalid", NULL);
      }
    } else {
      bombElegant("RPN_DEFNS environment variable undefined. Must define or provide -rpnDefns commandline option.", NULL);
    }
  } else {
    rpn(rpnDefns);
    if (rpn_check_error()) {
      bombElegant("rpn definitions file invalid", NULL);
    }
  }

  if (!configurationFile) {
    if ((configurationFile = getenv("ELEGANT_CONFIGURATION")) &&
        !strlen(configurationFile = getenv("ELEGANT_CONFIGURATION")))
      configurationFile = NULL;
  }

#if defined(_WIN32)
  /*Default number of allowed open files is 512, max allowed on windows is 2048*/
  _setmaxstdio(1024);
#endif

  if (pipeFlags & USE_STDIN) {
    if (inputfile)
      bombElegant("both -pipe=input and input file given", NULL);
#if defined(_WIN32)
    if (_setmode(_fileno(stdin), _O_BINARY) == -1)
      bombElegant("unable to set stdin to binary mode", NULL);
#endif
    inputfile = "stdin";
  } else {
    if (!inputfile) {
      printf("No input file was given.\n");
      showUsageOrGreeting(SHOW_USAGE);
      exitElegant(1);
    }
  }

  cp_str(&macroValue[0], inputfile);

#if defined(CONDOR_COMPILE)
  sprintf(s, "%s.ckpt", inputfile);
  init_image_with_file_name(s);
#else
#  ifndef _WIN32
  signal(SIGUSR2, SIG_IGN);
#  endif
#endif

  initialize_structures(&run_conditions, &run_control, &error_control, &correct, &beam, output_data,
                        &optimize, &chrom_corr_data, &tune_corr_data, &links);

  run_setuped = run_controled = error_controled = correction_setuped = ionEffectsSeen = 0;
  changeStart.active = changeEnd.active = 0;

  beam_type = -1;
  if (configurationFile) {
    inputFileArray[0] = configurationFile;
    inputFileArray[1] = inputfile;
  } else {
    inputFileArray[0] = inputfile;
    inputFileArray[1] = NULL;
  }

  iInput = 0;
  fpIn = tmalloc(sizeof(*fpIn) * 1);
  maxIncludedFiles = 1;
  fpIndex = 0;
  while (iInput < 2 && (inputfile = inputFileArray[iInput++]) != NULL) {
    fpIn[0] = NULL;
    fpIndex = 0;
    if (strcmp(inputfile, "stdin") == 0)
      fpIn[0] = stdin;
    else
      fpIn[0] = fopen_e(inputfile, "r", 0);
    while (fpIndex >= 0) {
      commandCode = -1;
      while (getNamelist(s, NAMELIST_BUFLEN, fpIn[fpIndex], &namelistErrorCode)) {
        if (namelistErrorCode != NAMELIST_NO_ERROR)
          break;
        substituteTagValue(s, NAMELIST_BUFLEN, macroTag, macroValue, macros);
#if DEBUG
        fprintf(stderr, "%s\n", s);
#endif
        report_stats(stdout, "statistics: ");
        fflush(stdout);
        if (namelists_read)
          free_namelist_text(&namelist_text);
        scan_namelist(&namelist_text, s);
        namelists_read = 1;
#if USE_MPI /* synchronize all the processes before execute an input statement */
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        lastCommandCode = commandCode;
        switch ((commandCode = match_string(namelist_text.group_name, command, N_COMMANDS, EXACT_MATCH))) {
        case INCLUDE_COMMANDS:
          include_commands_struct.filename = NULL;
          include_commands_struct.disable = 0;
          set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
          set_print_namelist_flags(0);
          if (processNamelist(&include_commands, &namelist_text) == NAMELIST_ERROR)
            bombElegant(NULL, NULL);
          if (echoNamelists)
            print_namelist(stdout, &include_commands);
          fflush(stdout);
          if (!include_commands_struct.disable) {
            fpIndex++;
            if (fpIndex >= maxIncludedFiles)
              fpIn = SDDS_Realloc(fpIn, sizeof(*fpIn) * (++maxIncludedFiles));
            if (!(fpIn[fpIndex] = fopen_e(include_commands_struct.filename, "r", 0)))
              bombElegant(NULL, NULL);
          }
          break;
        case CHANGE_PARTICLE:
          if (run_setuped)
            bombElegant("particle command should precede run_setup", NULL); /* to ensure nothing is inconsistent */
          process_particle_command(&namelist_text);
          break;
        case CHANGE_START:
          if (run_setuped)
            bombElegant("change_start command must preceed run_setup", NULL);
          process_change_start(&namelist_text, &changeStart);
          break;
        case CHANGE_END:
          if (run_setuped)
            bombElegant("change_end command must preceed run_setup", NULL);
          process_change_end(&namelist_text, &changeEnd);
          break;
        case RUN_SETUP:
          beam_type = -1;
          initialize_structures(NULL, &run_control, &error_control, &correct, &beam, output_data,
                                &optimize, &chrom_corr_data, &tune_corr_data, &links);
          /* Drop any prior per-step corrector snapshots so the next
           * vary_beamline() tick takes a fresh baseline. */
          corrector_stash_invalidate_all();
          reset_alter_specifications();
          clearSliceAnalysis();
          finish_load_parameters();
          run_setuped = run_controled = error_controled = correction_setuped = ionEffectsSeen = 0;

          run_setuped = run_controled = error_controled = correction_setuped = do_closed_orbit = do_chromatic_correction =
            fl_do_tune_correction = fl_do_coupling_correction = fl_do_lattice_correction = do_floor_coordinates = 0;
          do_twiss_output = do_matrix_output = do_response_output = do_coupled_twiss_output = do_moments_output =
            do_find_aperture = do_rf_setup = 0;
          linear_chromatic_tracking_setup_done = losses_include_global_coordinates = 0;
          losses_s_limit[0] = -(losses_s_limit[1] = DBL_MAX);
          search_path = NULL;
          s_start = 0;

          set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
          set_print_namelist_flags(0);
          if (processNamelist(&run_setup, &namelist_text) == NAMELIST_ERROR)
            bombElegant(NULL, NULL);

          if ((!search_path || !strlen(search_path)) && searchPath && strlen(searchPath)) {
            search_path = searchPath;
          }
          if (echoNamelists)
            print_namelist(stdout, &run_setup);
          setSearchPath(search_path);

          if (concat_order != 0)
            printWarning("concat_order is non-zero in run_setup.",
                         "Using matrix concatenation is rarely needed and reduces accuracy.");

          /* check for validity of namelist inputs */
          if (lattice == NULL) {
            if (!saved_lattice)
              bombElegant("no lattice supplied", NULL);
            if (default_order != last_default_order)
              delete_matrix_data(NULL);
          } else {
            /* free previous lattice info */
            if (beamline)
              closeBeamlineOutputFiles(beamline);
            free_elements(NULL);
            free_beamlines(NULL);
            freeInputObjects();
            saved_lattice = lattice;
          }
          free_beamdata(&beam);
          if (default_order < 1 || default_order > 3)
            bombElegant("default_order is out of range", NULL);
          if (concat_order > 3)
            bombElegant("concat_order is out of range", NULL);
          if (p_central && p_central_mev)
            bombElegant("give only one of p_central and p_central_mev", NULL);
          if (p_central_mev != 0 && p_central == 0)
            p_central = p_central_mev / particleMassMV;
          if (p_central <= 0 && !expand_for)
            bombElegant("p_central<=0 and p_central_mev<=0", NULL);
          if (expand_for)
            p_central = find_beam_p_central(expand_for);
          if (random_number_seed == 0) {
            random_number_seed = (long)time(NULL);
            random_number_seed = 2 * (random_number_seed / 2) + 1;
            printf("clock-generated random_number_seed = %ld\n", random_number_seed);
            fflush(stdout);
          }

          /* copy run data into run_conditions structure */
          run_conditions.ideal_gamma = sqrt(sqr(p_central) + 1);
          run_conditions.p_central = p_central;
          run_conditions.default_order = default_order;
          run_conditions.concat_order = concat_order;
          run_conditions.print_statistics = print_statistics;
          run_conditions.combine_bunch_statistics = combine_bunch_statistics;
          run_conditions.wrap_around = wrap_around;
          run_conditions.showElementTiming = show_element_timing;
          run_conditions.monitorMemoryUsage = monitor_memory_usage;
          run_conditions.backtrack = back_tracking;
          sStart = s_start;
          if ((run_conditions.lossLimit[0] = losses_s_limit[0]) > (run_conditions.lossLimit[1] = losses_s_limit[1]))
            bombElegant("losses_s_limit[0] can't be greater than losses_s_limit[1]", NULL);
          totalPropertiesPerParticle = COORDINATES_PER_PARTICLE + BASIC_PROPERTIES_PER_PARTICLE;
          globalLossCoordOffset = spinCoordOffset = 0;
          if ((run_conditions.lossesIncludeGlobalCoordinates = losses_include_global_coordinates)) {
            globalLossCoordOffset = COORDINATES_PER_PARTICLE + BASIC_PROPERTIES_PER_PARTICLE;
            totalPropertiesPerParticle += GLOBAL_LOSS_PROPERTIES_PER_PARTICLE;
          } 
	  if ((run_conditions.spinTracking = spin_tracking)) {
	    spinCoordOffset = COORDINATES_PER_PARTICLE + BASIC_PROPERTIES_PER_PARTICLE + (globalLossCoordOffset>0 ? GLOBAL_LOSS_PROPERTIES_PER_PARTICLE: 0);
	    totalPropertiesPerParticle += SPIN_PROPERTIES_PER_PARTICLE;
	  }
          sizeOfParticle = totalPropertiesPerParticle * sizeof(double);
          if (starting_coord)
            free(starting_coord);
          starting_coord = calloc(1, sizeOfParticle);
          if ((run_conditions.final_pass = final_pass))
            run_conditions.wrap_around = 1;
          run_conditions.tracking_updates = tracking_updates;
          run_conditions.always_change_p0 = always_change_p0;
          run_conditions.load_balancing_on = load_balancing_on;
          run_conditions.random_sequence_No = random_sequence_No;
          remaining_sequence_No = random_sequence_No; /* For Pelegant regression test */

          /* extract the root filename from the input filename */
          strcpy_ss(s, inputfile);
          if (rootname == NULL) {
            clean_filename(s);
            if ((ptr = strrchr(s, '.')))
              *ptr = 0;
            cp_str(&rootname, s);
            run_conditions.rootname = rootname;
            run_conditions.runfile = inputfile;
          } else {
            run_conditions.rootname = rootname;
            run_conditions.runfile = compose_filename(inputfile, rootname);
          }

          seedElegantRandomNumbers(random_number_seed, 0);

          /* In the version with parallel I/O, these file names need to be known by all the processors, otherwise there
             will be a synchronization issue when calculated accumulated sum in do_tracking */
          run_conditions.acceptance = compose_filename(acceptance, rootname);
#if USE_MPI
          if (run_conditions.acceptance)
            dumpAcceptance = 1;
#endif
          run_conditions.centroid = compose_filename(centroid, rootname);
          run_conditions.bpmCentroid = compose_filename(bpm_centroid, rootname);
          run_conditions.sigma = compose_filename(sigma, rootname);

          if (countIgnoreElementsSpecs(0) != 0) {
            if ((run_conditions.centroid && strlen(run_conditions.centroid)) ||
                (run_conditions.bpmCentroid && strlen(run_conditions.bpmCentroid)))
              bombElegant("Can't request centroid output if ignore_elements command with completely=0 is in force", NULL);
            if (run_conditions.sigma && strlen(run_conditions.sigma))
              bombElegant("Can't request sigma output if ignore_elements command with completely=0 is in force", NULL);
          }

          if (run_conditions.apertureData.initialized && !run_conditions.apertureData.persistent)
            resetApertureData(&(run_conditions.apertureData));

          /* parse the lattice file and create the beamline */
          run_conditions.lattice = compose_filename(saved_lattice, rootname);
          if (element_divisions > 1)
            addDivisionSpec("*", NULL, NULL, NULL, element_divisions, 0.0);
#ifdef USE_MPE /* use the MPE library */
          if (USE_MPE) {
            int event1a, event1b;
            event1a = MPE_Log_get_event_number();
            event1b = MPE_Log_get_event_number();
            if (isMaster)
              MPE_Describe_state(event1a, event1b, "get_beamline", "blue");
            MPE_Log_event(event1a, 0, "start get_beamline"); /* record time spent on reading input */
#endif
            beamline = get_beamline(lattice, use_beamline, p_central, echo_lattice, back_tracking, &changeStart, &changeEnd);
            changeStart.active = 0;
#ifdef USE_MPE
            MPE_Log_event(event1b, 0, "end get_beamline");
          }
#endif
          printf("length of beamline %s per pass: %21.15e m\n", beamline->name, beamline->revolution_length);
          fflush(stdout);
          lattice = saved_lattice;

          if ((final && strlen(final)) || sceffects_inserted)
            beamline->flags |= BEAMLINE_MATRICES_NEEDED;

#if !SDDS_MPI_IO
          if (isMaster)
#endif
            {
              run_conditions.output = compose_filename(output, rootname);
              run_conditions.losses = compose_filename(losses, rootname);
              run_conditions.final = compose_filename(final, rootname);
            }
#if !SDDS_MPI_IO
          else
            run_conditions.final = run_conditions.output = run_conditions.losses = NULL;
#endif

          if (isMaster) { /* These files will be written by master */
            magnets = compose_filename(magnets, rootname);
            profile = compose_filename(profile, rootname);
            semaphore_file = compose_filename(semaphore_file, rootname);
            parameters = compose_filename(parameters, rootname);
            rfc_reference_output = compose_filename(rfc_reference_output, rootname);
          } else {
            magnets = profile = semaphore_file = parameters = rfc_reference_output = NULL;
          }

          manageSemaphoreFiles(semaphore_file, rootname, semaphoreFile);

          /* output the magnet layout */
          if (magnets)
            output_magnets(magnets, lattice, beamline);
          if (profile)
            output_profile(profile, lattice, beamline);

          delete_phase_references(); /* necessary for multi-step runs */
          reset_special_elements(beamline, RESET_INCLUDE_ALL);
          reset_driftCSR();
          finishSCSpecs();
          last_default_order = default_order;
          run_setuped = 1;
          break;
        case GLOBAL_SETTINGS:
          processGlobalSettings(&namelist_text);
          break;
        case RUN_CONTROL:
          if (!run_setuped)
            bombElegant("run_setup must precede run_control namelist", NULL);
          vary_setup(&run_control, &namelist_text, &run_conditions, beamline);
          run_control.ready = 1;
          run_controled = 1;
          beamline->fiducial_flag = run_control.fiducial_flag;
          break;
        case VARY_ELEMENT:
          if (!run_controled)
            bombElegant("run_control must precede vary_element namelists", NULL);
          if (beam_type != -1)
            bombElegant("vary_element statements must come before beam definition", NULL);
          add_varied_element(&run_control, &namelist_text, &run_conditions, beamline);
          break;
        case ERROR_CONTROL:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede error_control namelist", NULL);
          if (beam_type != -1)
            bombElegant("error specifications must be completed before beam type is specified", NULL);
          error_setup(&error_control, &namelist_text, &run_conditions, beamline);
          error_controled = 1;
          break;
        case ERROR_ELEMENT:
          if (beam_type != -1)
            bombElegant("error_element statements must come before beam definition", NULL);
          if (!error_controled)
            bombElegant("error_control namelist must precede error_element namelists", NULL);
          add_error_element(&error_control, &namelist_text, beamline);
          break;
        case CORRECTION_SETUP:
          if (!run_setuped)
            bombElegant("run_setup must precede correction", NULL);
          if (beam_type != -1)
            bombElegant("beam setup (bunched_beam or sdds_beam) must follow correction setup", NULL);
          correction_setuped = 1;
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          correction_setup(&correct, &namelist_text, &run_conditions, beamline);
          delete_phase_references();
          reset_special_elements(beamline, RESET_INCLUDE_RF);
          reset_driftCSR();
          break;
        case SET_AWE_BEAM:
          printf("This program no longer supports awe-format files.\n");
          printf("Use awe2sdds to convert your data files, and use\n");
          printf("the sdds_beam command instead of awe_beam.\n");
          exit(1);
          break;
        case SET_BUNCHED_BEAM:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede bunched_beam namelist", NULL);
          if (beam_type != -1)
            bombElegant("a beam definition was already given for this run sequence", NULL);
          setup_bunched_beam(&beam, &namelist_text, &run_conditions, &run_control, &error_control, &optimize.variables,
                             output_data, beamline, beamline->n_elems,
                             correct.mode != -1 &&
                             (correct.track_before_and_after || correct.start_from_centroid));
          setup_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables, beamline);
          beam_type = SET_BUNCHED_BEAM;
          break;
        case SET_BUNCHED_BEAM_MOMENTS:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede bunched_beam_moments namelist", NULL);
          if (beam_type != -1)
            bombElegant("a beam definition was already given for this run sequence", NULL);
          setup_bunched_beam_moments(&beam, &namelist_text, &run_conditions, &run_control, &error_control, &optimize.variables,
                                     output_data, beamline, beamline->n_elems,
                                     correct.mode != -1 &&
                                     (correct.track_before_and_after || correct.start_from_centroid));
          setup_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables, beamline);
          beam_type = SET_BUNCHED_BEAM;
          break;
        case SET_SDDS_BEAM:
#if USE_MPI
          distributedBeam = 1;
#endif
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede sdds_beam namelist", NULL);
          if (beam_type != -1)
            bombElegant("a beam definition was already given for this run sequence", NULL);
          setup_sdds_beam(&beam, &namelist_text, &run_conditions, &run_control, &error_control,
                          &optimize.variables, output_data, beamline, beamline->n_elems,
                          correct.mode != -1 &&
                          (correct.track_before_and_after || correct.start_from_centroid));
          setup_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables, beamline);
          beam_type = SET_SDDS_BEAM;
          break;
        case ION_EFFECTS:
          if (!run_setuped)
            bombElegant("run_setup must precede ion_effects namelist", NULL);
          setupIonEffects(&namelist_text, &run_control, &run_conditions);
          ionEffectsSeen = 1;
          break;
        case TRACK:
        case ANALYZE_MAP:
        case TOUSCHEK_SCATTER:
          switch (commandCode) {
          case TRACK:
            if (!run_setuped || !run_controled || beam_type == -1)
              bombElegant("run_setup, run_control, and beam definition must precede track namelist", NULL);
            set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
            set_print_namelist_flags(0);
            if (processNamelist(&track, &namelist_text) == NAMELIST_ERROR)
              bombElegant(NULL, NULL);
            if (echoNamelists)
              print_namelist(stdout, &track);
            run_conditions.stopTrackingParticleLimit = stop_tracking_particle_limit;
            run_conditions.checkBeamStructure = check_beam_structure;
            if (interrupt_file && strlen(interrupt_file)) {
              run_conditions.trackingInterruptFile = compose_filename(interrupt_file, rootname);
              run_conditions.trackingInterruptFileMtime = 0;
              if (fexists(run_conditions.trackingInterruptFile))
                run_conditions.trackingInterruptFileMtime = get_mtime(run_conditions.trackingInterruptFile);
            }
            /*
              #if USE_MPI
              if (stop_tracking_particle_limit!=-1)
              bombElegant("stop_tracking_particle_limit feature not supported in Pelegant", NULL);
              #endif
            */
            if (use_linear_chromatic_matrix &&
                !(linear_chromatic_tracking_setup_done || twiss_computed || do_twiss_output))
              bombElegant("you must compute twiss parameters or give linear_chromatic_tracking_setup to do linear chromatic tracking", NULL);
            if (longitudinal_ring_only && !(twiss_computed || do_twiss_output))
              bombElegant("you must compute twiss parameters to do longitudinal ring tracking", NULL);
            if (use_linear_chromatic_matrix && longitudinal_ring_only)
              bombElegant("can't do linear chromatic tracking and longitudinal-only tracking together", NULL);
            if (beam_type == -1)
              bombElegant("beam must be defined prior to tracking", NULL);
            if (ionEffectsSeen)
              completeIonEffectsSetup(&run_conditions, beamline);
            break;
          case ANALYZE_MAP:
            if (!run_setuped || !run_controled)
              bombElegant("run_setup and run_control must precede analyze_map namelist", NULL);
            setup_transport_analysis(&namelist_text, &run_conditions, &run_control, &error_control);
            break;
          case TOUSCHEK_SCATTER:
            if (!run_setuped || !(twiss_computed || do_twiss_output))
              bombElegant("run_setup and twiss_output must precede touschek_scatter namelist", NULL);
            break;
          }
          firstPass = 1;
          if (run_controled)
            soft_failure = !(run_control.terminate_on_failure);
          else
            soft_failure = 0;
          while (vary_beamline(&run_control, &error_control, &run_conditions, beamline)) {
            if (run_control.restartFiles)
              setup_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables, beamline);
            /* vary_beamline asserts changes due to vary_element, error_element, and load_parameters */
            fill_double_array(starting_coord, 6, 0.0);
            /* correctionDone = 0; */
            new_beam_flags = 0;
            if (correct.mode != -1 && (correct.track_before_and_after || correct.start_from_centroid)) {
              if (beam_type == SET_SDDS_BEAM) {
                if (new_sdds_beam(&beam, &run_conditions, &run_control, output_data, 0) < 0)
                  break;
              } else
                new_bunched_beam(&beam, &run_conditions, &run_control, output_data, 0);
              new_beam_flags = TRACK_PREVIOUS_BUNCH;
              if (commandCode == TRACK && correct.track_before_and_after) {
                track_beam(&run_conditions, &run_control, &error_control, &optimize.variables,
                           beamline, &beam, output_data,
                           PRECORRECTION_BEAM, 0, &finalCharge);
                /* This is needed to put the original bunch back in the tracking buffer, since it may
                 * be needed as the starting point for orbit/trajectory correction
                 */
                if (beam_type == SET_SDDS_BEAM) {
                  if (new_sdds_beam(&beam, &run_conditions, &run_control, output_data, 0) < 0)
                    break;
                } else
                  new_bunched_beam(&beam, &run_conditions, &run_control, output_data, 0);
                new_beam_flags = TRACK_PREVIOUS_BUNCH;
              }
            }

            /* If needed, find closed orbit, twiss parameters, moments, and response matrix, but don't write
             * output unless requested to do so "pre-correction"
             */
            /* If closed orbit is calculated, starting_coord will store the closed orbit at the start of
             * the beamline
             */
            if (do_closed_orbit &&
                !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
              if (soft_failure) {
                printWarning("Closed orbit not found.", "Continuing to next step.");
                continue;
              } else
                bombElegant("Closed orbit not found", NULL);
            }
            /* Compute twiss parameters with starting_coord as the start of the orbit */
	    /*
            if (do_twiss_output) {
              update_twiss_parameters(&run_conditions, beamline, &unstable);
	      if (unstable) {
                if (soft_failure) {
                  printWarning("Twiss parameters not defined.", "Continuing to next step.");
                  continue;
                } else
                  bombElegant("Twiss parameters not defined", NULL);
              }
	    }
	    */
            if (do_rf_setup)
              run_rf_setup(&run_conditions, beamline, 0);
            /* compute moments with starting_coord as the start of the orbit/trajectory */
            if (do_moments_output)
              runMomentsOutput(&run_conditions, beamline, starting_coord, 0, 1);
            if (do_response_output)
              run_response_output(&run_conditions, beamline, &correct, 0);

            if (correct.mode != -1 || fl_do_tune_correction || fl_do_coupling_correction || fl_do_lattice_correction || do_chromatic_correction) {
              /* Perform orbit, tune, and/or chromaticity correction */
              if (correct.use_actual_beam && correct.mode == TRAJECTORY_CORRECTION) {
                if (beam_type == SET_SDDS_BEAM) {
                  if (new_sdds_beam(&beam, &run_conditions, &run_control, output_data, 0) < 0)
                    break;
                } else
                  new_bunched_beam(&beam, &run_conditions, &run_control, output_data, 0);
                new_beam_flags = TRACK_PREVIOUS_BUNCH;
              }
              for (i = failed = 0; i < correction_iterations; i++) {
                if (correction_iterations > 1) {
                  printf("\nOrbit/tune/chromaticity correction iteration %ld\n", i + 1);
                  fflush(stdout);
                }
                /* Orbit/trajectory correction */
                if (correct.mode != -1 &&
                    !do_correction(&correct, &run_conditions, beamline, starting_coord, &beam,
                                   run_control.i_step,
                                   (i == 0 ? INITIAL_CORRECTION : 0) + (i == correction_iterations - 1 ? FINAL_CORRECTION : 0))) {
                  printWarning("Orbit correction failed.", "Continuing with next step.");
                  continue;
                }
                if (fl_do_coupling_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_correct_coupling(&run_conditions, beamline)) {
                    if (soft_failure) {
                      printWarning("Coupling correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Coupling correction failed", NULL);
                  }
                }
                if (fl_do_lattice_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_correct_lattice(&run_conditions, beamline)) {
                    if (soft_failure) {
                      printWarning("Lattice correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Lattice correction failed", NULL);
                  }
                }
                if (fl_do_tune_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_tune_correction(&tune_corr_data, &run_conditions, &run_control, beamline, starting_coord, do_closed_orbit,
                                          run_control.i_step, i == correction_iterations - 1)) {
                    if (soft_failure) {
                      printWarning("Tune correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Tune correction failed", NULL);
                  }
                }
                if (do_chromatic_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_chromaticity_correction(&chrom_corr_data, &run_conditions, beamline, starting_coord, do_closed_orbit,
                                                  run_control.i_step, i == correction_iterations - 1)) {
                    if (!soft_failure) {
                      printWarning("Chromaticity correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Chromaticity correction failed", NULL);
                  }
                }
                /* correctionDone = 1; */
              }
              if (failed)
                continue;
            }

            if (correct.mode != -1 && (correct.track_before_and_after || (correct.start_from_centroid && correct.mode == TRAJECTORY_CORRECTION))) {
              /* If we are performing orbit/trajectory correction and tracking before/after correction, we need to
                 generate a beam (will in fact just restore the beam generated above.
                 Also, if we need the beam to give the starting point for trajectory correction, we need to genrate a beam.
              */
              if (beam_type == SET_SDDS_BEAM) {
                if (new_sdds_beam(&beam, &run_conditions, &run_control, output_data, new_beam_flags) < 0)
                  break;
              } else if (beam_type == SET_BUNCHED_BEAM)
                new_bunched_beam(&beam, &run_conditions, &run_control, output_data, new_beam_flags);
            }

            /* Assert post-correction perturbations */
            perturb_beamline(&run_control, &error_control, &run_conditions, beamline);

            /* Do post-correction output */
            if (do_closed_orbit && !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 1)) {
              if (soft_failure) {
                printWarning("Closed orbit not found.", "Continuing to next step.");
                continue;
              } else
                bombElegant("Closed orbit not found", NULL);
            }
            if (do_twiss_output && !run_twiss_output(&run_conditions, beamline, starting_coord, 1)) {
              if (soft_failure) {
                printWarning("Twiss parameters not defined.", "Continuing to next step.");
                continue;
              } else
                bombElegant("Twiss parameters not defined", NULL);
            }
            if (do_rf_setup)
              run_rf_setup(&run_conditions, beamline, 1);
            if (do_moments_output)
              runMomentsOutput(&run_conditions, beamline, starting_coord, 1, 1);
            if (do_coupled_twiss_output &&
                run_coupled_twiss_output(&run_conditions, beamline, starting_coord)) {
              if (soft_failure) {
                printWarning("Coupled twiss parameters computation failed.", NULL);
                continue;
              } else
                bombElegant("Coupled twiss parameters computation failed.", NULL);
            }
            if (do_response_output)
              run_response_output(&run_conditions, beamline, &correct, 1);

            if (!(correct.mode != -1 &&
                  (correct.track_before_and_after || (correct.start_from_centroid && correct.mode == TRAJECTORY_CORRECTION)))) {
              /* This is where we normally generate the beam, unless it was needed prior to trajectory correction */
              if (beam_type == SET_SDDS_BEAM) {
                if (new_sdds_beam(&beam, &run_conditions, &run_control, output_data, new_beam_flags) < 0)
                  break;
              } else if (beam_type == SET_BUNCHED_BEAM)
                new_bunched_beam(&beam, &run_conditions, &run_control, output_data, new_beam_flags);
            }

            if (center_on_orbit)
              center_beam_on_coords(beam.particle, beam.n_to_track, starting_coord, center_momentum_also);
            else if (offset_by_orbit)
              offset_beam_by_coords(beam.particle, beam.n_to_track, starting_coord, offset_momentum_also);
            run_matrix_output(&run_conditions, &run_control, beamline);
            if (firstPass) {
              /* prevent fiducialization of RF etc. by correction etc. */
              delete_phase_references();
              reset_special_elements(beamline, RESET_INCLUDE_RF);
              reset_driftCSR();
            }
            firstPass = 0;
            switch (commandCode) {
            case TRACK:
              /* Finally, do tracking */
              track_beam(&run_conditions, &run_control, &error_control, &optimize.variables,
                         beamline, &beam, output_data,
                         (use_linear_chromatic_matrix ? LINEAR_CHROMATIC_MATRIX : 0) +
                         (longitudinal_ring_only ? LONGITUDINAL_RING_ONLY : 0) +
                         (ibs_only ? IBS_ONLY_TRACKING : 0),
                         0, &finalCharge);
              break;
            case ANALYZE_MAP:
              do_transport_analysis(&run_conditions, &run_control, &error_control, beamline,
                                    (do_closed_orbit || correct.mode != -1 ? starting_coord : NULL));
              break;
            case TOUSCHEK_SCATTER:
              TouschekEffect(&run_conditions, &run_control, &error_control, beamline, &namelist_text);
              break;
            }
            if (parameters)
              dumpLatticeParameters(parameters, &run_conditions, beamline, suppress_parameter_defaults);
            if (rfc_reference_output)
              dumpRfcReferenceData(rfc_reference_output, &run_conditions, beamline);
            /* Reset corrector magnets for before/after tracking mode */
            if (correct.mode != -1 && commandCode == TRACK && correct.track_before_and_after)
              zero_correctors(beamline->elem_recirc ? beamline->elem_recirc : beamline->elem, &run_conditions, &correct);
            if (run_control.restartFiles) {
              if (commandCode == TRACK)
                finish_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables,
                              beamline, beamline->n_elems, &beam, finalCharge);
              if (do_closed_orbit)
                finish_clorb_output();
              if (do_twiss_output)
                finish_twiss_output(beamline);
              if (do_moments_output)
                finishMomentsOutput();
              if (do_coupled_twiss_output)
                finish_coupled_twiss_output();
              if (do_response_output)
                finish_response_output();
              if (parameters)
                finishLatticeParametersFile();
              if (rfc_reference_output)
                finishRfcDataFile();
              if (correct.mode != -1)
                finishCorrectionOutput();
            }
            if (run_control.stepDoneSemaphore && strlen(run_control.stepDoneSemaphore)) {
              FILE *fpsem;
#if USE_MPI
              if (myid == 0) {
#endif
                printf("Creating step-done semaphore file %s\n", run_control.stepDoneSemaphore);
                fpsem = fopen(run_control.stepDoneSemaphore, "w");
                fclose(fpsem);
#if USE_MPI
              }
              MPI_Barrier(MPI_COMM_WORLD);
#endif
            }
          }
          if (!run_control.restartFiles) {
            if (commandCode == TRACK)
              finish_output(output_data, &run_conditions, &run_control, &error_control, &optimize.variables,
                            beamline, beamline->n_elems, &beam, finalCharge);
            if (beam_type)
              free_beamdata(&beam);
            if (do_closed_orbit)
              finish_clorb_output();
            if (do_twiss_output)
              finish_twiss_output(beamline);
            if (do_moments_output)
              finishMomentsOutput();
            if (do_coupled_twiss_output)
              finish_coupled_twiss_output();
            if (do_response_output)
              finish_response_output();
            if (parameters)
              finishLatticeParametersFile();
            if (rfc_reference_output)
              finishRfcDataFile();
            if (correct.mode != -1)
              finishCorrectionOutput();
          }

          switch (commandCode) {
          case TRACK:
            printf("Finished tracking.\n");
            break;
          case ANALYZE_MAP:
            printf("Finished transport analysis.\n");
            break;
          }
          fflush(stdout);
          /* reassert defaults for namelist run_setup */
          lattice = use_beamline = acceptance = centroid = bpm_centroid = sigma = final = output = rootname = losses =
            parameters = NULL;
          combine_bunch_statistics = 0;
          random_number_seed = 987654321;
          wrap_around = 1;
          final_pass = 0;
          default_order = 2;
          concat_order = 0;
          tracking_updates = 1;
          show_element_timing = monitor_memory_usage = 0;
          concat_order = print_statistics = p_central = 0;
          run_setuped = run_controled = error_controled = correction_setuped = do_chromatic_correction =
            fl_do_tune_correction = fl_do_coupling_correction = fl_do_lattice_correction = do_closed_orbit = do_twiss_output = do_coupled_twiss_output = do_response_output =
            ionEffectsSeen = back_tracking = losses_include_global_coordinates = 0;
          element_divisions = 0;
          run_conditions.rampData.valuesInitialized =
            run_conditions.modulationData.valuesInitialized = 0;
          break;
        case MATRIX_OUTPUT:
          if (!run_setuped)
            bombElegant("run_setup must precede matrix_output namelist", NULL);
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          setup_matrix_output(&namelist_text, &run_conditions, run_controled ? &run_control : NULL, beamline);
          do_matrix_output = 1;
          break;
        case TWISS_OUTPUT:
          if (!run_setuped)
            bombElegant("run_setup must precede twiss_output namelist", NULL);
          setup_twiss_output(&namelist_text, &run_conditions, beamline, &do_twiss_output,
                             run_conditions.default_order);
          if (!do_twiss_output) {
            twiss_computed = 1;
            run_twiss_output(&run_conditions, beamline, starting_coord, -1);
            delete_phase_references();
            reset_special_elements(beamline, RESET_INCLUDE_RF);
            reset_driftCSR();
            finish_twiss_output(beamline);
          }
          break;
        case RF_SETUP:
          if (!run_setuped)
            bombElegant("run_setup must precede rf_setup namelist", NULL);
          setup_rf_setup(&namelist_text, &run_conditions, beamline, do_twiss_output, &do_rf_setup);
          break;
        case MOMENTS_OUTPUT:
          if (!run_setuped)
            bombElegant("run_setup must precede moments_output namelist", NULL);
          setupMomentsOutput(&namelist_text, &run_conditions, beamline, &do_moments_output,
                             run_conditions.default_order);
          if (!do_moments_output) {
            /* moments_computed = 1; */
            runMomentsOutput(&run_conditions, beamline, NULL, -1, 1);
            delete_phase_references();
            reset_special_elements(beamline, RESET_INCLUDE_RF);
            reset_driftCSR();
            finishMomentsOutput();
          }
          break;
        case COUPLED_TWISS_OUTPUT:
          if (!run_setuped)
            bombElegant("run_setup must precede coupled_twiss_output namelist", NULL);
          setup_coupled_twiss_output(&namelist_text, &run_conditions, beamline, &do_coupled_twiss_output,
                                     run_conditions.default_order);
          if (!do_coupled_twiss_output) {
            run_coupled_twiss_output(&run_conditions, beamline, NULL);
            delete_phase_references();
            reset_special_elements(beamline, RESET_INCLUDE_RF);
            reset_driftCSR();
            finish_coupled_twiss_output();
          }
          break;
        case CORRECT_COUPLING:
          if (!run_setuped)
            bombElegant("run_setup and twiss_output must precede correct_coupling namelist", NULL);
          setup_correct_coupling(&namelist_text, &run_conditions, beamline);
          fl_do_coupling_correction = 1;
          break;
        case COMPUTE_COUPLING_RESPONSE_MATRIX:
          if (!run_setuped || (!do_twiss_output && !twiss_computed))
            bombElegant("run_setup and twiss_output must precede compute_coupling_response_matrix namelist", NULL);
          setup_compute_coupling_response_matrix(&namelist_text, &run_conditions, beamline);
          break;
        case LOAD_COUPLING_RESPONSE_MATRIX:
          if (!run_setuped)
            bombElegant("run_setup must precede load_coupling_response_matrix namelist", NULL);
          setup_load_coupling_response_matrix(&namelist_text, &run_conditions, beamline);
          break;
        case CORRECT_LATTICE:
          if (!run_setuped || (!do_twiss_output && !twiss_computed))
            bombElegant("run_setup and twiss_output must precede correct_lattice namelist", NULL);
          setup_correct_lattice(&namelist_text, &run_conditions, beamline);
          fl_do_lattice_correction = 1;
          break;
        case COMPUTE_LATTICE_RESPONSE_MATRIX:
          if (!run_setuped || (!do_twiss_output && !twiss_computed))
            bombElegant("run_setup and twiss_output must precede compute_lattice_response_matrix namelist", NULL);
          setup_compute_lattice_response_matrix(&namelist_text, &run_conditions, beamline);
          break;
        case LOAD_LATTICE_RESPONSE_MATRIX:
          if (!run_setuped)
            bombElegant("run_setup must precede load_lattice_response_matrix namelist", NULL);
          setup_load_lattice_response_matrix(&namelist_text, &run_conditions, beamline);
          break;
        case TUNE_SHIFT_WITH_AMPLITUDE:
          if (do_twiss_output)
            bombElegant("you must give tune_shift_with_amplitude before twiss_output", NULL);
          setupTuneShiftWithAmplitude(&namelist_text, &run_conditions);
          break;
        case SEMAPHORES:
          do_semaphore_setup(semaphoreFile, run_conditions.rootname, run_setuped, &namelist_text);
          break;
        case STOP:
          lorentz_report();
          finish_load_parameters();
          /* if (semaphore_file)
             createSemaphoreFile(semaphore_file);
             if (semaphoreFile[1])
             createSemaphoreFile(semaphoreFile[1]);
          */
          free_beamdata(&beam);
          printFarewell(stdout);
          exitElegant(0);
          break;
        case OPTIMIZATION_SETUP:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          do_optimization_setup(&optimize, &namelist_text, &run_conditions, beamline);
          break;
#if USE_MPI
        case PARALLEL_OPTIMIZATION_SETUP:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          do_parallel_optimization_setup(&optimize, &namelist_text, &run_conditions, beamline);
          break;
#endif
        case OPTIMIZE_CMD:
          if (beam_type == -1)
            bombElegant("beam definition must come before optimize command", NULL);
          while (vary_beamline(&run_control, &error_control, &run_conditions, beamline)) {
            do_optimize(&namelist_text, &run_conditions, &run_control, &error_control, beamline, &beam,
                        output_data, &optimize, &chrom_corr_data, beam_type, do_closed_orbit,
                        do_chromatic_correction, &correct, correct.mode, &tune_corr_data,
                        fl_do_tune_correction, do_find_aperture, do_response_output);
            if (parameters)
              dumpLatticeParameters(parameters, &run_conditions, beamline, suppress_parameter_defaults);
          }
          if (parameters)
            finishLatticeParametersFile();
          /* reassert defaults for namelist run_setup */
          lattice = use_beamline = acceptance = centroid = bpm_centroid = sigma = final = output = rootname = losses =
            parameters = NULL;
          combine_bunch_statistics = 0;
          random_number_seed = 987654321;
          wrap_around = 1;
          final_pass = 0;
          default_order = 2;
          concat_order = 0;
          tracking_updates = 1;
          show_element_timing = monitor_memory_usage = 0;
          concat_order = print_statistics = p_central = 0;
          run_setuped = run_controled = error_controled = correction_setuped = do_chromatic_correction =
            fl_do_tune_correction = fl_do_coupling_correction = fl_do_lattice_correction = do_closed_orbit = do_twiss_output = do_coupled_twiss_output = do_response_output =
            ionEffectsSeen = 0;
#if USE_MPI
          independentRunPerRank = 0; /* We should set the flag to the normal parallel tracking after parallel optimization */
#endif
          break;
        case OPTIMIZATION_VARIABLE:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          add_optimization_variable(&optimize, &namelist_text, &run_conditions, beamline);
          break;
        case OPTIMIZATION_CONSTRAINT:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          add_optimization_constraint(&optimize, &namelist_text, &run_conditions, beamline);
          break;
        case OPTIMIZATION_COVARIABLE:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          add_optimization_covariable(&optimize, &namelist_text, &run_conditions, beamline);
          break;
        case SET_REFERENCE_PARTICLE_OUTPUT:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          do_set_reference_particle_output(&optimize, &namelist_text, &run_conditions, beamline);
          break;
        case OPTIMIZATION_TERM:
          if (beam_type != -1)
            bombElegant("optimization statements must come before beam definition", NULL);
          add_optimization_term(&optimize, &namelist_text, &run_conditions, beamline);
          break;
        case SAVE_LATTICE:
          do_save_lattice(&namelist_text, &run_conditions, beamline);
          break;
        case RPN_EXPRESSION:
          run_rpn_expression(&namelist_text);
          break;
        case RPN_LOAD:
          run_rpn_load(&namelist_text, &run_conditions);
          break;
        case UNDULATOR_BRIGHTNESS:
          setup_undulator_brightness(&namelist_text, &run_conditions, beamline);
          break;
        case LOAD_KNOBS:
          setup_load_knobs(&namelist_text, &run_conditions, beamline);
          break;
        case PROGRAM_TRACE:
          process_trace_request(&namelist_text);
          break;
        case CHROMATICITY:
          setup_chromaticity_correction(&namelist_text, &run_conditions, beamline, &chrom_corr_data);
          do_chromatic_correction = 1;
          break;
        case CORRECT_TUNES:
          setup_tune_correction(&namelist_text, &run_conditions, beamline, &tune_corr_data);
          fl_do_tune_correction = 1;
          break;
        case CLOSED_ORBIT:
          do_closed_orbit = setup_closed_orbit(&namelist_text, &run_conditions, beamline);
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          if (do_closed_orbit == 2) {
            /* immediate mode */
            run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 1);
            do_closed_orbit = 0;
          }
          /*
            if (correction_setuped)
            printWarning("You've asked to do both closed-orbit calculation and orbit correction, which may duplicate effort.\n");
          */
          fflush(stdout);
          break;
        case PARTICLE_TUNES:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede particle_tunes namelist", NULL);
          setupParticleTunes(&namelist_text, &run_conditions, &run_control, &(output_data->particleTunes));
          break;
        case MACRO_OUTPUT:
          if (!run_setuped)
            bombElegant("run_setup must precede macro_output namelist", NULL);
          doMacroOutput(&namelist_text, &run_conditions, macroTag, macroValue, macros);
          break;
        case TUNE_FOOTPRINT:
        case FIND_APERTURE:
        case FREQUENCY_MAP:
        case MOMENTUM_APERTURE:
        case ELASTIC_SCATTERING:
        case INELASTIC_SCATTERING:
        case CHAOS_MAP:
          switch (commandCode) {
            long code;
          case TUNE_FOOTPRINT:
            if (!run_setuped || !run_controled)
              bombElegant("run_setup and run_control must precede tune_footprint namelist", NULL);
            code = setupTuneFootprint(&namelist_text, &run_conditions, &run_control);
            if (code == 1) {
              /* immediate output to files */
              doTuneFootprint(&run_conditions, &run_control, starting_coord, beamline, NULL);
              outputTuneFootprint(&run_control);
              continue;
            } else if (code == 2) {
              /* used by optimizer (or other), no output */
              continue;
            }
            /* will compute footprint and output below */
            break;
          case FIND_APERTURE:
            setup_aperture_search(&namelist_text, &run_conditions, &run_control, &do_find_aperture);
            if (do_find_aperture)
              continue;
            break;
          case FREQUENCY_MAP:
            setupFrequencyMap(&namelist_text, &run_conditions, &run_control);
            break;
          case MOMENTUM_APERTURE:
            setupMomentumApertureSearch(&namelist_text, &run_conditions, &run_control);
            break;
          case ELASTIC_SCATTERING:
            setupElasticScattering(&namelist_text, &run_conditions, &run_control, do_twiss_output || twiss_computed);
            beamline->flags |= BEAMLINE_MATRICES_NEEDED;
            break;
          case INELASTIC_SCATTERING:
            setupInelasticScattering(&namelist_text, &run_conditions, &run_control);
            beamline->flags |= BEAMLINE_MATRICES_NEEDED;
            break;
          case CHAOS_MAP:
            setupChaosMap(&namelist_text, &run_conditions, &run_control);
            break;
          }
          if (run_controled)
            soft_failure = !(run_control.terminate_on_failure);
          else
            soft_failure = 0;
          while (vary_beamline(&run_control, &error_control, &run_conditions, beamline)) {
#if DEBUG
            printf("semaphore_file = %s\n", semaphore_file ? semaphore_file : NULL);
#endif
            fill_double_array(starting_coord, 6, 0.0);
            if (correct.mode != -1 || fl_do_tune_correction || fl_do_coupling_correction || fl_do_lattice_correction || do_chromatic_correction) {
              for (i = failed = 0; i < correction_iterations; i++) {
                if (run_control.reset_rf_each_step)
                  delete_phase_references();
                reset_special_elements(beamline, run_control.reset_rf_each_step ? RESET_INCLUDE_RF : 0);
                runFiducialParticle(&run_conditions, &run_control, starting_coord, beamline, 0, 0);
                if (correction_iterations > 1) {
                  printf("\nOrbit/tune/chromaticity correction iteration %ld\n", i + 1);
                  fflush(stdout);
                }
                if (correct.mode != -1 &&
                    !do_correction(&correct, &run_conditions, beamline, starting_coord, &beam,
                                   run_control.i_step,
                                   (i == 0 ? INITIAL_CORRECTION : 0) + (i == correction_iterations - 1 ? FINAL_CORRECTION : 0))) {
                  printWarning("Orbit correction failed.", "Continuing with next step.");
                  continue;
                }
                if (fl_do_coupling_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_correct_coupling(&run_conditions, beamline)) {
                    if (soft_failure) {
                      printWarning("Coupling correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Coupling correction failed", NULL);
                  }
                }
                if (fl_do_lattice_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_correct_lattice(&run_conditions, beamline)) {
                    if (soft_failure) {
                      printWarning("Lattice correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Lattice correction failed", NULL);
                  }
                }
                if (fl_do_tune_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_tune_correction(&tune_corr_data, &run_conditions, &run_control, beamline, starting_coord, do_closed_orbit,
                                          run_control.i_step, i == correction_iterations - 1)) {
                    if (soft_failure) {
                      printWarning("Tune correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Tune correction failed", NULL);
                  }
                }
                if (do_chromatic_correction) {
                  if (do_closed_orbit &&
                      !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 0)) {
                    if (soft_failure) {
                      printWarning("Closed orbit not found.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Closed orbit not found", NULL);
                  }
                  if (!do_chromaticity_correction(&chrom_corr_data, &run_conditions, beamline, starting_coord, do_closed_orbit,
                                                  run_control.i_step, i == correction_iterations - 1)) {
                    if (soft_failure) {
                      printWarning("Chromaticity correction failed.", "Continuing to next step.");
                      failed = 1;
                      break;
                    } else
                      bombElegant("Chromaticity correction failed", NULL);
                  }
                }
                /* correctionDone = 1; */
              }
              if (failed)
                continue;
            }

            if (run_control.reset_rf_each_step)
              delete_phase_references();
            reset_special_elements(beamline, run_control.reset_rf_each_step ? RESET_INCLUDE_RF : 0);
            runFiducialParticle(&run_conditions, &run_control, starting_coord, beamline, 1, 1);
            perturb_beamline(&run_control, &error_control, &run_conditions, beamline);
            if (do_closed_orbit &&
                !run_closed_orbit(&run_conditions, beamline, starting_coord, NULL, 1)) {
              if (soft_failure) {
                printWarning("Closed orbit not found.", "Continuing to next step.");
                continue;
              } else
                bombElegant("Closed orbit not found", NULL);
            }
            if (do_twiss_output &&
                !run_twiss_output(&run_conditions, beamline, starting_coord, 1)) {
              if (soft_failure) {
                printWarning("Twiss parameters not defined.", "Continuing to next step.");
                continue;
              } else
                bombElegant("Twiss parameters not defined", NULL);
            }
            if (do_coupled_twiss_output &&
                run_coupled_twiss_output(&run_conditions, beamline, starting_coord)) {
              if (soft_failure) {
                printWarning("Coupled twiss parameters calculation failed.", NULL);
                /* Should we continue to next step ?? */
              } else
                bombElegant("Coupled twiss parameters calculation failed", NULL);
            }
            run_matrix_output(&run_conditions, &run_control, beamline);
            if (do_response_output)
              run_response_output(&run_conditions, beamline, &correct, 1);
            if (do_rf_setup)
              run_rf_setup(&run_conditions, beamline, 0);
            if (do_moments_output)
              runMomentsOutput(&run_conditions, beamline, starting_coord, 1, 1);
            if (parameters)
              dumpLatticeParameters(parameters, &run_conditions, beamline, suppress_parameter_defaults);
            switch (commandCode) {
            case FIND_APERTURE:
              do_aperture_search(&run_conditions, &run_control, starting_coord,
                                 &error_control, beamline, &apertureReturn);
              break;
            case FREQUENCY_MAP:
              doFrequencyMap(&run_conditions, &run_control, starting_coord, &error_control, beamline);
              break;
            case MOMENTUM_APERTURE:
              doMomentumApertureSearch(&run_conditions, &run_control, &error_control, beamline,
                                       (correct.mode != -1 || do_closed_orbit) ? starting_coord : NULL);
              break;
            case ELASTIC_SCATTERING:
              runElasticScattering(&run_conditions, &run_control, &error_control, beamline,
                                   (correct.mode != -1 || do_closed_orbit) ? starting_coord : NULL);
              break;
            case INELASTIC_SCATTERING:
              runInelasticScattering(&run_conditions, &run_control, &error_control, beamline,
                                     (correct.mode != -1 || do_closed_orbit) ? starting_coord : NULL);
              break;
            case CHAOS_MAP:
              doChaosMap(&run_conditions, &run_control, starting_coord, &error_control, beamline);
              break;
            case TUNE_FOOTPRINT:
              doTuneFootprint(&run_conditions, &run_control, starting_coord, beamline, NULL);
              outputTuneFootprint(&run_control);
              break;
            }
          }
          printf("Finished all tracking steps.\n");
          fflush(stdout);
          fflush(stdout);
          switch (commandCode) {
          case FIND_APERTURE:
            finish_aperture_search(&run_conditions, &run_control, &error_control, beamline);
            break;
          case FREQUENCY_MAP:
            finishFrequencyMap();
            break;
          case MOMENTUM_APERTURE:
            finishMomentumApertureSearch();
            break;
          case ELASTIC_SCATTERING:
            finishElasticScattering();
            break;
          case INELASTIC_SCATTERING:
            finishInelasticScattering();
            break;
          case CHAOS_MAP:
            finishChaosMap();
            break;
          }
          if (do_closed_orbit)
            finish_clorb_output();
          if (parameters)
            finishLatticeParametersFile();
          if (beam_type != -1)
            free_beamdata(&beam);
          if (do_closed_orbit)
            finish_clorb_output();
          if (do_twiss_output)
            finish_twiss_output(beamline);
          if (do_response_output)
            finish_response_output();
          if (correct.mode != -1)
            finishCorrectionOutput();

          switch (commandCode) {
          case FIND_APERTURE:
            printf("Finished dynamic aperture search.\n");
            break;
          case FREQUENCY_MAP:
            printf("Finished frequency map analysis.\n");
            break;
          case MOMENTUM_APERTURE:
            printf("Finished momentum aperture search.\n");
            break;
          case ELASTIC_SCATTERING:
            printf("Finished elastic scattering.\n");
            break;
          case INELASTIC_SCATTERING:
            printf("Finished inelastic scattering.\n");
            break;
          case CHAOS_MAP:
            printf("Finished chaos map analysis.\n");
            break;
          }
#if DEBUG
          printf("semaphore_file = %s\n", semaphore_file ? semaphore_file : NULL);
#endif
          fflush(stdout);
          /* reassert defaults for namelist run_setup */
          lattice = use_beamline = acceptance = centroid = bpm_centroid = sigma = final = output = rootname = losses =
            parameters = NULL;
          combine_bunch_statistics = 0;
          random_number_seed = 987654321;
          wrap_around = 1;
          final_pass = 0;
          default_order = 2;
          concat_order = 0;
          tracking_updates = 1;
          show_element_timing = monitor_memory_usage = 0;
          concat_order = print_statistics = p_central = 0;
          run_setuped = run_controled = error_controled = correction_setuped = do_chromatic_correction =
            fl_do_tune_correction = fl_do_coupling_correction = fl_do_lattice_correction = do_closed_orbit = do_twiss_output = do_coupled_twiss_output = do_response_output =
            ionEffectsSeen = 0;
          break;
        case LINK_CONTROL:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede link_control namelist", NULL);
          element_link_control(&links, &namelist_text, &run_conditions, beamline);
          break;
        case LINK_ELEMENTS:
          if (!run_setuped || !run_controled)
            bombElegant("run_setup and run_control must precede link_elements namelist", NULL);
          if (!beamline)
            bombElegant("beamline not defined--can't add element links", NULL);
          add_element_links(&links, &namelist_text, beamline);
          /* links_present = 1; */
          beamline->links = &links;
          break;
        case STEERING_ELEMENT:
          if (correction_setuped)
            bombElegant("you must define steering elements prior to giving the 'correct' namelist", NULL);
          add_steering_element(&correct, beamline, &run_conditions, &namelist_text);
          break;
        case AMPLIF_FACTORS:
          if (!correction_setuped)
            bombElegant("You must specify orbit or trajectory correction prior to the amplification_factors command",
                        NULL);
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          if (parameters)
            dumpLatticeParameters(parameters, &run_conditions, beamline, suppress_parameter_defaults);
          compute_amplification_factors(&namelist_text, &run_conditions, &correct, do_closed_orbit, beamline);
          break;
        case PRINT_DICTIONARY:
          set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
          set_print_namelist_flags(0);
          if (processNamelist(&print_dictionary, &namelist_text) == NAMELIST_ERROR)
            bombElegant(NULL, NULL);
          if (echoNamelists)
            print_namelist(stdout, &print_dictionary);
          do_print_dictionary(filename, latex_form, SDDS_form);
          break;
        case FLOOR_COORDINATES:
          if (!run_setuped)
            bombElegant("run_setup must precede floor_coordinates namelist", NULL);
          output_floor_coordinates(&namelist_text, &run_conditions, beamline);
          do_floor_coordinates = 1;
          break;
        case CORRECTION_MATRIX_OUTPUT:
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          if (!run_setuped)
            bombElegant("run setup must precede correction_matrix_output namelist", NULL);
          setup_correction_matrix_output(&namelist_text, &run_conditions, beamline, &correct,
                                         &do_response_output,
                                         do_twiss_output + do_matrix_output + twiss_computed);
          if (!do_response_output) {
            run_response_output(&run_conditions, beamline, &correct, -1);
            delete_phase_references();
            reset_special_elements(beamline, RESET_INCLUDE_RF);
            reset_driftCSR();
            finish_response_output();
          }
          break;
        case LOAD_PARAMETERS:
          if (!run_setuped)
            bombElegant("run_setup must precede load_parameters namelists", NULL);
          if (run_controled)
            bombElegant("load_parameters namelists must precede run_control namelist", NULL);
          if (error_controled)
            bombElegant("load_parameters namelists must precede error_control and error namelists", NULL);
          if (setup_load_parameters(&namelist_text, &run_conditions, beamline)) {
            if (magnets)
              output_magnets(magnets, lattice, beamline);
            if (profile)
              output_profile(profile, lattice, beamline);
          }
          break;
        case SUBPROCESS:
          if (isMaster)
            run_subprocess(&namelist_text, &run_conditions);
          break;
        case FIT_TRACES:
#if 0
          do_fit_trace_data(&namelist_text, &run_conditions, beamline);
          if (parameters) {
            dumpLatticeParameters(parameters, &run_conditions, beamline, suppress_parameter_defaults);
            finishLatticeParametersFile();
          }
          /* reassert defaults for namelist run_setup */
          lattice = use_beamline = acceptance = centroid = bpm_centroid = sigma = final = output = rootname = losses =
            parameters = NULL;
          combine_bunch_statistics = 0;
          random_number_seed = 987654321;
          wrap_around = 1;
          final_pass = 0;
          default_order = 2;
          concat_order = 0;
          tracking_updates = 1;
          show_element_timing = monitor_memory_usage = 0;
          concat_order = print_statistics = p_central = 0;
          run_setuped = run_controled = error_controled = correction_setuped = do_chromatic_correction =
            fl_do_tune_correction = do_closed_orbit = do_twiss_output = do_coupled_twiss_output = do_response_output = 
            ionEffectsSeed = 0;
#endif
          break;
        case SASEFEL_AT_END:
          if (!run_setuped)
            bombElegant("run_setup must precede sasefel namelist", NULL);
          if (beam_type != -1)
            bombElegant("sasefel namelist must precede beam definition", NULL);
          setupSASEFELAtEnd(&namelist_text, &run_conditions, output_data);
          break;
        case ALTER_ELEMENTS:
          if (!run_setuped)
            bombElegant("run_setup must precede alter_element namelist", NULL);
          setup_alter_element(&namelist_text, &run_conditions, beamline);
          break;
        case SLICE_ANALYSIS:
#if USE_MPI
          printf("\n********\nslice_analysis not supported for parallel version\n********\n\n");
          fflush(stdout);
#else
          if (!run_setuped)
            bombElegant("run_setup must precede slice_analysis namelist", NULL);
          if (beam_type != -1)
            bombElegant("slice_analysis namelist must precede beam definition", NULL);
          setupSliceAnalysis(&namelist_text, &run_conditions, output_data);
#endif
          break;
        case DIVIDE_ELEMENTS:
          if (run_setuped)
            bombElegant("divide_elements must precede run_setup", NULL);
          setupDivideElements(&namelist_text, &run_conditions, beamline);
          break;
        case TRANSMUTE_ELEMENTS:
          if (run_setuped)
            bombElegant("transmute_elements must precede run_setup", NULL);
          setupTransmuteElements(&namelist_text, &run_conditions, beamline);
          break;
        case IGNORE_ELEMENTS:
          if (run_setuped)
            bombElegant("ignore_elements must precede run_setup", NULL);
          setupIgnoreElements(&namelist_text, &run_conditions, beamline);
          break;
        case INSERT_SCEFFECTS:
          if (run_setuped)
            bombElegant("insert_sceffects must precede run_setup", NULL);
          setupSCEffect(&namelist_text, &run_conditions, beamline);
          sceffects_inserted = 1;
          break;
        case INSERT_ELEMENTS:
          if (!run_setuped)
            bombElegant("run_setup must precede insert_element namelist", NULL);
          if (lastCommandCode != RUN_SETUP && lastCommandCode != INSERT_ELEMENTS && lastCommandCode != REPLACE_ELEMENTS)
            printWarning("To avoid possible calculation errors, insert_elements commands should immediately follow run_setup.", NULL);
          do_insert_elements(&namelist_text, &run_conditions, beamline);
          break;
        case REPLACE_ELEMENTS:
          if (!run_setuped)
            bombElegant("run_setup must precede replace_element namelist", NULL);
          if (lastCommandCode != RUN_SETUP && lastCommandCode != INSERT_ELEMENTS && lastCommandCode != REPLACE_ELEMENTS)
            printWarning("To avoid possible calculation errors, replace_elements commands should immediately follow run_setup.", NULL);
          do_replace_elements(&namelist_text, &run_conditions, beamline);
          break;
        case TWISS_ANALYSIS:
          if (do_twiss_output)
            bombElegant("twiss_analysis must come before twiss_output", NULL);
          setupTwissAnalysisRequest(&namelist_text, &run_conditions, beamline);
          break;
        case APERTURE_INPUT:
        case APERTURE_DATAX:
          processApertureInput(&namelist_text, &run_conditions);
          break;
        case OBSTRUCTION_DATA:
          if (!run_setuped)
            bombElegant("run_setup must precede obstruction_data namelist", NULL);
          if (!do_floor_coordinates)
            bombElegant("floor_coordinate command required for obstruction_data to work", NULL);
#if HAVE_GPU
          bombElegant("The obstruction_data command is not implemented for the GPU version of elegant.", NULL);
#endif
          readObstructionInput(&namelist_text, &run_conditions);
          break;
        case LINEAR_CHROMATIC_TRACKING_SETUP:
          beamline->flags |= BEAMLINE_MATRICES_NEEDED;
          if (do_twiss_output)
            bombElegant("you can't give twiss_output and linear_chromatic_tracking_setup together", NULL);
          if (!run_setuped)
            bombElegant("run_setup must precede linear_chromatic_tracking_setup", NULL);
          setupLinearChromaticTracking(&namelist_text, beamline);
          linear_chromatic_tracking_setup_done = 1;
          break;
        case MODULATE_ELEMENTS:
          if (!run_setuped)
            bombElegant("run_setup must precede modulate_elements", NULL);
          addModulationElements(&(run_conditions.modulationData), &namelist_text, beamline, &run_conditions);
          break;
        case RAMP_ELEMENTS:
          if (!run_setuped)
            bombElegant("run_setup must precede ramp_elements", NULL);
          addRampElements(&(run_conditions.rampData), &namelist_text, beamline, &run_conditions);
          break;
        default:
          printf("unknown namelist %s given.  Known namelists are:\n", namelist_text.group_name);
          fflush(stdout);
          for (i = 0; i < N_COMMANDS; i++)
            printf("%s\n", description[i]);
          fflush(stdout);
          exitElegant(1);
          break;
        }
      }

      switch (namelistErrorCode) {
      case NAMELIST_NO_ERROR:
        break;
      case NAMELIST_BUFFER_TOO_SMALL:
        bombElegant("Error: namelist buffer too small. Check for improper construction.\n", NULL);
        break;
      case NAMELIST_IMPROPER_CONSTRUCTION:
        bombElegant("Error: namelist construction error. Check for missing &end.\n", NULL);
        break;
      default:
        bombElegant("Error: Invalid return from namelist scan. Seek expert help.\n", NULL);
        break;
      }

      fclose(fpIn[fpIndex--]);
    }
  }

#if DEBUG
  printf("semaphore_file = %s\n", semaphore_file ? semaphore_file : NULL);
#endif
  printf("End of input data encountered.\n");
  fflush(stdout);
  fflush(stdout);
  lorentz_report();
  finish_load_parameters();
  free_beamdata(&beam);
  free(macroTag);
  free(macroValue);
  free(starting_coord);
#if USE_MPI
  printf("Terminating run with %d total processors\n", n_processors);
#endif
  report_stats(stdout, "statistics: ");
  fflush(stdout);
  log_exit("main");
  printFarewell(stdout);
  if (load_hash)
    hdestroy(load_hash); /* destroy hash table */
  free_scanargs(&scanned, argc);
  exitElegant(0);
  return 0; /* suppresses compiler warning */
}

void printFarewell(FILE *fp) {
  summarizeWarnings();
#if (USE_MPI)
  printf("=====================================================================================\n");
  printf("Thanks for using Pelegant.  Please cite the following references in your publications:\n");
  printf("  M. Borland, \"elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,\"\n");
  printf("  Advanced Photon Source LS-287, September 2000.\n");
  printf("  Y. Wang and M. Borland, \"Pelegant: A Parallel Accelerator Simulation Code for  \n");
  printf("  Electron Generation and Tracking\", Proceedings of the 12th Advanced Accelerator  \n");
  printf("  Concepts Workshop, AIP Conf. Proc. 877, 241 (2006).\n");
  printf("If you use a modified version, please indicate this in all publications.\n");
  printf("=====================================================================================\n");
#elif HAVE_GPU
  printf("=====================================================================================\n");
  printf("Thanks for using gpu-elegant.  Please cite the following references in your publications:\n");
  printf("  M. Borland, \"elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,\"\n");
  printf("  Advanced Photon Source LS-287, September 2000.\n");
  printf("  J. R. King, I. V. Pogorelov, M. Borland, R. Soliday, K. Amyx,\n");
  printf("  \"Current status of the GPU-Accelerated version of elegant,\" Proc. IPAC15, 623 (2015).\n");
  printf("If you use a modified version, please indicate this in all publications.\n");
  printf("=====================================================================================\n");
#else
  printf("=====================================================================================\n");
  printf("Thanks for using elegant.  Please cite the following reference in your publications:\n");
  printf("  M. Borland, \"elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,\"\n");
  printf("  Advanced Photon Source LS-287, September 2000.\n");
  printf("If you use a modified version, please indicate this in all publications.\n");
  printf("=====================================================================================\n");
#endif
}

double find_beam_p_central(char *input) {
  SDDS_DATASET SDDSin;
  char s[SDDS_MAXLINE];
  double *p = NULL, psum;
  long i, rows;
#if SDDS_MPI_IO
  /* All the processes will read the wake file, but not in parallel.
     Zero the Memory when call  SDDS_InitializeInput */
  SDDSin.parallel_io = 0;
#endif

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, input) || !SDDS_ReadPage(&SDDSin)) {
    sprintf(s, "Problem opening beam input file %s", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if ((rows = SDDS_RowCount(&SDDSin)) <= 0 || !(p = SDDS_GetColumnInDoubles(&SDDSin, "p"))) {
    sprintf(s, "No data in input file %s", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  for (i = psum = 0; i < rows; i++)
    psum += p[i];
  if (SDDS_ReadPage(&SDDSin) > 0) {
    char buffer[16384];
    snprintf(buffer, 16384, ": %s", input);
    printWarning("Only the first page of a file is used by expand_for.", buffer);
    fflush(stdout);
  }
  SDDS_Terminate(&SDDSin);
  printf("Expanding about p = %21.15e\n", psum / rows);
  fflush(stdout);
  return psum / rows;
}

void center_beam_on_coords(double **part, long np, double *coord, long center_dp) {
  double offset;
  long i, j, lim;
  double centroid[6];

  compute_centroids(centroid, part, np);

  if (center_dp)
    lim = 5;
  else
    lim = 4;

  for (j = 0; j <= lim; j++) {
    offset = centroid[j] - coord[j];
    for (i = 0; i < np; i++)
      part[i][j] -= offset;
  }

#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void offset_beam_by_coords(double **part, long np, double *coord, long offset_dp) {
  long i, j, lim;

  if (!np)
    return;

  if (offset_dp)
    lim = 5;
  else
    lim = 4;

  for (j = 0; j <= lim; j++) {
    for (i = 0; i < np; i++)
      part[i][j] += coord[j];
  }
}

typedef struct {
  char *elementName;
  long index;
} DICTIONARY_ENTRY;

int dictionaryEntryCmp(const void *p1, const void *p2) {
  DICTIONARY_ENTRY *de1, *de2;
  de1 = (DICTIONARY_ENTRY *)p1;
  de2 = (DICTIONARY_ENTRY *)p2;
  return strcmp(de1->elementName, de2->elementName);
}

void do_print_dictionary(char *filename, long latex_form, long SDDS_form) {
  FILE *fp;
  long i;
  DICTIONARY_ENTRY *dictList;

  if (!filename)
    bombElegant("filename invalid (do_print_dictionary)", NULL);
  if (latex_form && SDDS_form)
    bombElegant("give latex_form=1 or SDDS_form=1, not both", NULL);

  if (!(dictList = malloc(sizeof(*dictList) * N_TYPES)))
    bombElegant("memory allocation failure", NULL);
  for (i = 1; i < N_TYPES; i++) {
    dictList[i - 1].elementName = entity_name[i];
    dictList[i - 1].index = i;
  }
  qsort((void *)dictList, N_TYPES - 1, sizeof(*dictList), dictionaryEntryCmp);
  fp = fopen_e(filename, "w", 0);
#if DEBUG
  fprintf(stderr, "Opened file to print dictionary entries\n");
#endif
  if (latex_form) {
    fprintf(fp, "\\newlength{\\descwidth}\n");
    fprintf(fp, "\\setlength{\\descwidth}{2in}\n");
  }
  if (SDDS_form) {
    fprintf(fp, "SDDS1\n");
    fprintf(fp, "&parameter name=ElementType, type=string &end\n");
    fprintf(fp, "&parameter name=ParallelCapable, type=short &end\n");
    fprintf(fp, "&parameter name=GPUCapable, type=short &end\n");
    fprintf(fp, "&parameter name=BacktrackCapable, type=short &end\n");
    fprintf(fp, "&parameter name=SpinTrackingCapable, type=short &end\n");
    fprintf(fp, "&parameter name=Size, type=long, units=bytes &end\n");
    fprintf(fp, "&column name=ParameterName, type=string &end\n");
    fprintf(fp, "&column name=Units, type=string &end\n");
    fprintf(fp, "&column name=Type, type=string &end\n");
    fprintf(fp, "&column name=Default, type=string &end\n");
    fprintf(fp, "&column name=Description, type=string &end\n");
    fprintf(fp, "&data mode=ascii &end\n");
  }
#if DEBUG
  fprintf(stderr, "Beginning loop to print dictionary entries\n");
#endif
  for (i = 0; i < N_TYPES - 1; i++) {
#if DEBUG
    fprintf(stderr, "Printing dictionary entry %ld (%s) of %ld\n", i,
            dictList[i].elementName, (long)(N_TYPES - 1));
#endif
    if (entity_description[dictList[i].index].flags & NO_DICT_OUTPUT)
      continue;
    print_dictionary_entry(fp, dictList[i].index, latex_form, SDDS_form);
  }
#if DEBUG
  fprintf(stderr, "Free'ing dictList\n");
#endif
  free(dictList);
#if DEBUG
  fprintf(stderr, "Closing file\n");
#endif
  fclose(fp);
#if DEBUG
  fprintf(stderr, "Exiting do_print_dictionary\n");
#endif
}

#define PRINTABLE_NULL(s) (s ? s : "NULL")
char *translateUnitsToTex(char *source);
char *makeTexSafeString(char *source, long checkMath);

void print_dictionary_entry(FILE *fp, long type, long latex_form, long SDDS_form) {
  char *type_name[4] = {"double", "long", "STRING", "short"};
  char *description;
  long j, texLines, specialEntry, xyWaveforms;
  char buffer[16384];
  char *specialDescription = "Optionally used to assign an element to a group, with a user-defined name.  Group names will appear in the parameter output file in the column ElementGroup";
  if (latex_form) {
    fprintf(fp, "\\newpage\n\\begin{center}{\\Large\\verb|%s|}\\end{center}\n\\subsection{%s---%s}\n",
            entity_name[type], entity_name[type], makeTexSafeString(entity_text[type], 0));
    fprintf(fp, "%s\n\\\\\n", makeTexSafeString(entity_text[type], 0));
    fprintf(fp, "Parallel capable? : %s\\\\\n", entity_description[type].flags & UNIPROCESSOR ? "no" : "yes");
    fprintf(fp, "GPU capable? : %s\\\\\n", entity_description[type].flags & GPU_SUPPORT ? "yes" : "no");
    fprintf(fp, "Back-tracking capable? : %s\\\\\n", entity_description[type].flags & BACKTRACK ? "yes" : "no");
    fprintf(fp, "Spin-tracking capable? : %s\\\\\n", entity_description[type].flags & SPIN_TRACKING ? "yes" : "no");
    fprintf(fp, "\\begin{tabular}{|l|l|l|l|p{\\descwidth}|} \\hline\n");
    fprintf(fp, "Parameter Name & Units & Type & Default & Description \\\\ \\hline \n");
  } else {
    fprintf(fp, "%c***** element type %s:\n", SDDS_form ? '!' : '*', entity_name[type]);
    if (SDDS_form) {
      strcpy_ss(buffer, entity_name[type]);
      replace_chars(buffer, "\n\t", "  ");
      fprintf(fp, "%s\n", buffer);
      fprintf(fp, "%ld\n", (long)(entity_description[type].flags & UNIPROCESSOR ? 0 : 1));
      fprintf(fp, "%ld\n", (long)(entity_description[type].flags & GPU_SUPPORT ? 1 : 0));
      fprintf(fp, "%ld\n", (long)(entity_description[type].flags & BACKTRACK ? 1 : 0));
      fprintf(fp, "%ld\n", (long)(entity_description[type].flags & SPIN_TRACKING ? 1 : 0));
      fprintf(fp, "%ld\n", entity_description[type].structure_size);
      fprintf(fp, "%ld\n", entity_description[type].n_params + 1);
    } else
      fprintf(fp, "%s\n", entity_text[type]);
  }
  specialEntry = 0;
  xyWaveforms = 0;
  for (j = texLines = 0; j <= entity_description[type].n_params; j++) {
    if (j == entity_description[type].n_params)
      specialEntry = 1;
    /* 35 lines fits on a latex page */
    if (latex_form && texLines > 35) {
      texLines = 0;
      fprintf(fp, "\\end{tabular}\n\n");
      fprintf(fp, "\\newpage\n\\begin{center}{\\Large\\verb|%s| continued}\\end{center}\n",
              entity_name[type]);
      fprintf(fp, "%s\n\\\\\n", makeTexSafeString(entity_text[type], 0));
      fprintf(fp, "\\begin{tabular}{|l|l|l|l|p{\\descwidth}|} \\hline\n");
      fprintf(fp, "Parameter Name & Units & Type & Default & Description \\\\ \\hline \n");
    }
    if (SDDS_form) {
      fprintf(fp, "\"%s\" \"%s\" \"%s\"",
              PRINTABLE_NULL(specialEntry ? "GROUP" : entity_description[type].parameter[j].name),
              PRINTABLE_NULL(specialEntry ? "" : entity_description[type].parameter[j].unit),
              PRINTABLE_NULL(specialEntry ? "string" : type_name[entity_description[type].parameter[j].type - 1]));
    } else if (!latex_form)
      fprintf(fp, "%20s %20s %10s",
              PRINTABLE_NULL(specialEntry ? "GROUP" : entity_description[type].parameter[j].name),
              PRINTABLE_NULL(specialEntry ? "" : entity_description[type].parameter[j].unit),
              PRINTABLE_NULL(specialEntry ? "string" : type_name[entity_description[type].parameter[j].type - 1]));
    else {
      fprintf(fp, "%s ",
              makeTexSafeString(PRINTABLE_NULL(specialEntry ? "GROUP" : entity_description[type].parameter[j].name), 0));
      fprintf(fp, "& %s ",
              translateUnitsToTex(PRINTABLE_NULL(specialEntry ? "" : entity_description[type].parameter[j].unit)));
      fprintf(fp, "& %s & ",
              makeTexSafeString(PRINTABLE_NULL(specialEntry ? "string" : type_name[entity_description[type].parameter[j].type - 1]), 0));
    }
    if (!specialEntry) {
      if (entity_description[type].parameter[j].flags & PARAM_XY_WAVEFORM)
        xyWaveforms++;
      switch (entity_description[type].parameter[j].type) {
      case IS_DOUBLE:
        if (latex_form && entity_description[type].parameter[j].number == 0)
          fprintf(fp, " 0.0");
        else
          fprintf(fp, "  %.15g", entity_description[type].parameter[j].number);
        break;
      case IS_LONG:
      case IS_SHORT:
        if (latex_form && entity_description[type].parameter[j].integer == 0)
          fprintf(fp, " \\verb|0|");
        else
          fprintf(fp, "  %-15ld", entity_description[type].parameter[j].integer);
        break;
      case IS_STRING:
        if (SDDS_form)
          fprintf(fp, " \"%s\"",
                  PRINTABLE_NULL(entity_description[type].parameter[j].string));
        else
          fprintf(fp, "  %-15s",
                  PRINTABLE_NULL(entity_description[type].parameter[j].string));
        break;
      default:
        printf("Invalid parameter type for %s item of %s\n",
               PRINTABLE_NULL(entity_description[type].parameter[j].name),
               entity_name[type]);
        fflush(stdout);
        exitElegant(1);
      }
    } else {
      fprintf(fp, "NULL");
    }
    if (specialEntry)
      description = specialDescription;
    else
      description = entity_description[type].parameter[j].description;
    if (latex_form) {
      char *ptr0;
      strcpy_ss(buffer, description);
      if (strlen(ptr0 = buffer)) {
        /* add to lines counter based on estimate of wrap-around lines
           in the latex parbox. 28 is approximate number of char. in 2 in */
        texLines += strlen(ptr0) / 28;
        if (*ptr0) {
          fprintf(fp, " & %s ",
                  makeTexSafeString(ptr0, 1));
          fprintf(fp, " \\\\ \\hline \n");
          texLines++;
        }
      } else {
        fprintf(fp, " & \\\\ \\hline \n");
        texLines++;
      }
    } else if (SDDS_form)
      fprintf(fp, "  \"%s\"\n", description);
    else
      fprintf(fp, "  %s\n", description);
  }
  if (latex_form) {
    char physicsFile[1024];

    fprintf(fp, "\\end{tabular}\n\n");

    sprintf(physicsFile, "%s.tex", entity_name[type]);
    str_tolower(physicsFile);
    if (fexists(physicsFile)) {
      fprintf(stderr, "Including file %s\n", physicsFile);
      fprintf(fp, "\\vspace*{0.5in}\n\\input{%s}\n", physicsFile);
    }
    if (xyWaveforms)
      fprintf(fp, "\\vspace*{0.5in}\n\\input{xyWaveforms.tex}\n");
  }
}

#include <memory.h>

void initialize_structures(RUN *run_conditions, VARY *run_control, ERRORVAL *error_control, CORRECTION *correct,
                           BEAM *beam, OUTPUT_FILES *output_data, OPTIMIZATION_DATA *optimize,
                           CHROM_CORRECTION *chrom_corr_data, TUNE_CORRECTION *tune_corr_data,
                           ELEMENT_LINKS *links) {
  if (run_conditions)
    memset((void *)run_conditions, 0, sizeof(*run_conditions));
  if (run_control)
    memset((void *)run_control, 0, sizeof(*run_control));
  run_control->bunch_frequency = 2856e6;
  if (error_control)
    memset((void *)error_control, 0, sizeof(*error_control));
  if (correct)
    memset((void *)correct, 0, sizeof(*correct));
  correct->mode = correct->method = -1;
  if (beam)
    memset((void *)beam, 0, sizeof(*beam));
  if (output_data)
    memset((void *)output_data, 0, sizeof(*output_data));
  if (optimize)
    memset((void *)optimize, 0, sizeof(*optimize));
  if (chrom_corr_data)
    memset((void *)chrom_corr_data, 0, sizeof(*chrom_corr_data));
  if (tune_corr_data)
    memset((void *)tune_corr_data, 0, sizeof(*tune_corr_data));
  if (links)
    memset((void *)links, 0, sizeof(*links));
}

char *makeTexSafeString(char *source, long checkMath) {
  static char buffer[1024];
  long index = 0, math = 0;
  if (!source)
    return source;
  buffer[0] = 0;
  while (*source) {
    if (*source == '$' && checkMath) {
      math = !math;
      buffer[index++] = *source++;
      continue;
    }
    if (!math) {
      if (*source == '_' || *source == '^' || *source == '{' || *source == '}' || *source == '%') {
        buffer[index++] = '\\';
        buffer[index++] = *source++;
      } else if (*source == '<' || *source == '>' || *source == '|') {
        buffer[index++] = '$';
        buffer[index++] = *source++;
        buffer[index++] = '$';
      } else
        buffer[index++] = *source++;
    } else {
      buffer[index++] = *source++;
    }
  }
  buffer[index] = 0;
  return buffer;
}

char *translateUnitsToTex(char *source) {
  static char buffer[1024];
  char *ptr, *ptr1;

  if (strlen(source) == 0)
    return source;
  buffer[0] = '$';
  buffer[1] = 0;
  ptr = source;
  while (*ptr) {
    if (*ptr == '$') {
      switch (*(ptr1 = ptr + 1)) {
      case 'a':
      case 'b':
      case 'A':
      case 'B':
        while (*ptr1 && *ptr1 != '$')
          ptr1++;
        if (*ptr1 && (*(ptr1 + 1) == 'n' || *(ptr1 + 1) == 'N')) {
          if (*(ptr + 1) == 'a' || *(ptr + 1) == 'A')
            strcat(buffer, "^{");
          else
            strcat(buffer, "_{");
          strncat(buffer, ptr + 2, (ptr1 - 1) - (ptr + 2) + 1);
          strcat(buffer, "}");
          ptr = ptr1 + 1;
        } else
          strncat(buffer, ptr, 1);
        break;
      default:
        printf("Unrecognized $ sequence: %s\n", ptr);
        fflush(stdout);
        strncat(buffer, ptr, 1);
        break;
      }
    } else
      strncat(buffer, ptr, 1);
    ptr++;
  }
  strcat(buffer, "$");
  return buffer;
}

void free_beamdata(BEAM *beam) {
  if (beam->particle)
    free_czarray_2d((void **)beam->particle, beam->n_particle, totalPropertiesPerParticle);
  if (beam->accepted)
    free_czarray_2d((void **)beam->accepted, beam->n_particle, totalPropertiesPerParticle);
  if (beam->original && beam->original != beam->particle)
    free_czarray_2d((void **)beam->original, beam->n_original, totalPropertiesPerParticle);

  beam->particle = beam->accepted = beam->original = NULL;
  beam->n_original = beam->n_to_track = beam->n_accepted = beam->n_saved = beam->n_particle = 0;
  beam->p0_original = beam->p0 = 0.;
  beam->n_lost = 0;
}

long getTableFromSearchPath(TABLE *tab, char *file, long sampleInterval, long flags) {
  char *filename;
  long value;
  if (!(filename = findFileInSearchPath(file)))
    return 0;
  value = get_table(tab, filename, sampleInterval, flags);
  free(filename);
  return value;
}

void do_semaphore_setup(char **semaphoreFile, char *rootname, long run_setuped, NAMELIST_TEXT *nltext) {

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&semaphores, nltext) == NAMELIST_ERROR)
    bombElegant("Problem processing semaphores command", NULL);
  if (echoNamelists)
    print_namelist(stdout, &semaphores);

  if (immediate && strlen(immediate)) {
    FILE *fp;
    immediate = compose_filename(immediate, rootname);
    fp = fopen(immediate, "w");
    fclose(fp);
  }

  if ((started && strlen(started)) || (done && strlen(done)) || (failed && strlen(failed))) {
    if (run_setuped)
      bombElegant("You must give the semaphores command before run_setup except if using only the immediate parameter", NULL);
  }

  semaphoreFile[0] = semaphoreFile[1] = semaphoreFile[2] = NULL;
  if (writePermitted) {
    if (started)
      SDDS_CopyString(&semaphoreFile[0], started);
    if (done)
      SDDS_CopyString(&semaphoreFile[1], done);
    if (failed)
      SDDS_CopyString(&semaphoreFile[2], failed);
  }
}

void getRunControlContext(VARY *context) {
  *context = run_control;
}

void getRunSetupContext(RUN *context) {
  *context = run_conditions;
}

void swapParticles(double *p1, double *p2) {
  double buffer[MAX_PROPERTIES_PER_PARTICLE];
  if (p1 == p2)
    return;
  memcpy(buffer, p1, sizeOfParticle);
  memcpy(p1, p2, sizeOfParticle);
  memcpy(p2, buffer, sizeOfParticle);
}

void createSemaphoreFile(char *filename) {
  FILE *fp;
#if USE_MPI
  if (!isMaster)
    return;
#endif
  if (filename && strlen(filename) && !is_blank(filename)) {
    printf("Creating semaphore file \"%s\"\n", filename);
  } else
    return;
  if (!(fp = fopen(filename, "w"))) {
    printf("Problem creating semaphore file \"%s\"\n", filename);
    exitElegant(1);
  } else { /* Put the CPU time in the .done file */
    if (wild_match(filename, "*.done"))
      fprintf(fp, "%8.2f\n", cpu_time() / 100.0);
  }
  fclose(fp);
}

void processApertureInput(NAMELIST_TEXT *nltext, RUN *run) {
#include "aperture_data.h"

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&aperture_data, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &aperture_data);

  resetApertureData(&(run->apertureData));

  run->apertureData.initialized = 0;

  if (disable)
    return;

  readApertureInput(&(run->apertureData), input, 0);
  run->apertureData.periodic = periodic;
  run->apertureData.persistent = persistent;

  printf("\n** %ld points of aperture data read from file\n\n", run->apertureData.points);
}

void readApertureInput(APERTURE_DATA *apData, char *input, short zmode) {
  SDDS_DATASET SDDSin;
  char s[16384];
  long i;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, input)) {
    sprintf(s, "Problem opening aperture input file %s", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if (!check_sdds_column(&SDDSin, "xHalfAperture", "m") ||
      !check_sdds_column(&SDDSin, "yHalfAperture", "m") ||
      !check_sdds_column(&SDDSin, "xCenter", "m") ||
      !check_sdds_column(&SDDSin, "yCenter", "m")) {
    printf("Necessary data quantities (xHalfAperture, yHalfAperture, xCenter, and yCenter) have wrong units or are not present in %s\n",
           input);
    printf("Note that units must be \"m\" on all quantities\n");
    fflush(stdout);
    exitElegant(1);
  }

  if ((!zmode && !check_sdds_column(&SDDSin, "s", "m")) ||
      (zmode && !check_sdds_column(&SDDSin, "z", "m"))) {
    printf("Necessary data quantity %s has wrong units or not present in %s\n",
           zmode ? "z" : "s", input);
    printf("Note that units must be \"m\" on all quantities\n");
    fflush(stdout);
    exitElegant(1);
  }
  apData->zmode = zmode;

  if (!SDDS_ReadPage(&SDDSin)) {
    sprintf(s, "Problem reading aperture input file %s---seems to be empty", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if ((apData->points = SDDS_RowCount(&SDDSin)) < 2) {
    char buffer[16384];
    snprintf(buffer, 16384, "Aperture input file %s has only %ld rows",
             input, apData->points);
    printWarning(buffer, NULL);
  }

  if (!(apData->sz = SDDS_GetColumnInDoubles(&SDDSin, zmode ? "z" : "s")) ||
      !(apData->xMax = SDDS_GetColumnInDoubles(&SDDSin, "xHalfAperture")) ||
      !(apData->yMax = SDDS_GetColumnInDoubles(&SDDSin, "yHalfAperture")) ||
      !(apData->dx = SDDS_GetColumnInDoubles(&SDDSin, "xCenter")) ||
      !(apData->dy = SDDS_GetColumnInDoubles(&SDDSin, "yCenter"))) {
    sprintf(s, "Problem getting data from aperture input file %s", input);
    SDDS_SetError(s);
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (apData->sz[0] != 0) {
    printf("The first value of %s in %s is not zero.\n", input, zmode ? "z" : "s");
    exitElegant(1);
  }
  for (i = 0; i < apData->points; i++) {
    if (i && apData->sz[i] < apData->sz[i - 1]) {
      printf("%s values in %s are not monotonically increasing.\n", zmode ? "z" : "s", input);
      exitElegant(1);
    }
    if (apData->xMax[i] < 0 || apData->yMax[i] < 0) {
      printf("One or more xHalfAperture and yHalfAperture values in %s are negative.\n", input);
      exitElegant(1);
    }
  }

  apData->initialized = 1;
  apData->periodic = apData->persistent = 0;

  return;
}

void resetApertureData(APERTURE_DATA *apData) {
  if (apData->initialized) {
    if (apData->sz)
      free(apData->sz);
    if (apData->xMax)
      free(apData->xMax);
    if (apData->yMax)
      free(apData->yMax);
    if (apData->dx)
      free(apData->dx);
    if (apData->dy)
      free(apData->dy);
    apData->sz = apData->xMax = apData->yMax = apData->dx = apData->dy = NULL;
    apData->initialized = 0;
  }
}

/* Routine to close output files that are associated with beamline elements
 */

void closeBeamlineOutputFiles(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  CSRCSBEND *CsrCsBend;

  eptr = beamline->elem;
  while (eptr) {
    switch (eptr->type) {
    case T_WATCH:
      if (((WATCH *)(eptr->p_elem))->initialized) {
        SDDS_Terminate(((WATCH *)(eptr->p_elem))->SDDS_table);
        free(((WATCH *)(eptr->p_elem))->SDDS_table);
      }
      ((WATCH *)(eptr->p_elem))->initialized = 0;
      ((WATCH *)(eptr->p_elem))->SDDS_table = NULL;
      break;
    case T_HISTOGRAM:
      if (((HISTOGRAM *)(eptr->p_elem))->initialized) {
        SDDS_Terminate(((HISTOGRAM *)(eptr->p_elem))->SDDS_table);
        free(((HISTOGRAM *)(eptr->p_elem))->SDDS_table);
      }
      ((HISTOGRAM *)(eptr->p_elem))->initialized = 0;
      ((HISTOGRAM *)(eptr->p_elem))->SDDS_table = NULL;
      break;
    case T_CSRCSBEND:
      CsrCsBend = (CSRCSBEND *)(eptr->p_elem);
      if (CsrCsBend->histogramFile && CsrCsBend->SDDSout) {
        SDDS_Terminate(CsrCsBend->SDDSout);
        free(CsrCsBend->SDDSout);
      }
      CsrCsBend->SDDSout = NULL;
      if (CsrCsBend->particleOutputFile && CsrCsBend->SDDSpart) {
        SDDS_Terminate(CsrCsBend->SDDSpart);
        free(CsrCsBend->SDDSpart);
      }
      CsrCsBend->SDDSpart = NULL;
      break;
    case T_RFMODE:
      if (((RFMODE *)(eptr->p_elem))->record && ((RFMODE *)(eptr->p_elem))->fileInitialized) {
        SDDS_Terminate(((RFMODE *)(eptr->p_elem))->SDDSrec);
        free(((RFMODE *)(eptr->p_elem))->SDDSrec);
      }
      ((RFMODE *)(eptr->p_elem))->SDDSrec = NULL;
      ((RFMODE *)(eptr->p_elem))->fileInitialized = 0;
      break;
    case T_FRFMODE:
      if (((FRFMODE *)(eptr->p_elem))->outputFile && ((FRFMODE *)(eptr->p_elem))->initialized) {
        SDDS_Terminate(((FRFMODE *)(eptr->p_elem))->SDDSout);
        free(((FRFMODE *)(eptr->p_elem))->SDDSout);
      }
      ((FRFMODE *)(eptr->p_elem))->SDDSout = NULL;
      ((FRFMODE *)(eptr->p_elem))->initialized = 0;
      break;
    case T_TRFMODE:
      if (((TRFMODE *)(eptr->p_elem))->record && ((TRFMODE *)(eptr->p_elem))->fileInitialized) {
        SDDS_Terminate(((TRFMODE *)(eptr->p_elem))->SDDSrec);
        free(((TRFMODE *)(eptr->p_elem))->SDDSrec);
      }
      ((TRFMODE *)(eptr->p_elem))->SDDSrec = NULL;
      ((TRFMODE *)(eptr->p_elem))->fileInitialized = 0;
      break;
    case T_FTRFMODE:
      if (((FTRFMODE *)(eptr->p_elem))->outputFile && ((FTRFMODE *)(eptr->p_elem))->initialized) {
        SDDS_Terminate(((FTRFMODE *)(eptr->p_elem))->SDDSout);
        free(((FTRFMODE *)(eptr->p_elem))->SDDSout);
      }
      ((FTRFMODE *)(eptr->p_elem))->SDDSout = NULL;
      ((FTRFMODE *)(eptr->p_elem))->initialized = 0;
      break;
    case T_ZLONGIT:
      if (((ZLONGIT *)(eptr->p_elem))->wakes && ((ZLONGIT *)(eptr->p_elem))->SDDS_wake_initialized) {
        SDDS_Terminate(((ZLONGIT *)(eptr->p_elem))->SDDS_wake);
        free(((ZLONGIT *)(eptr->p_elem))->SDDS_wake);
      }
      ((ZLONGIT *)(eptr->p_elem))->SDDS_wake = NULL;
      ((ZLONGIT *)(eptr->p_elem))->SDDS_wake_initialized = 0;
      break;
    case T_ZTRANSVERSE:
      if (((ZTRANSVERSE *)(eptr->p_elem))->wakes && ((ZTRANSVERSE *)(eptr->p_elem))->SDDS_wake_initialized) {
        SDDS_Terminate(((ZTRANSVERSE *)(eptr->p_elem))->SDDS_wake);
        free(((ZTRANSVERSE *)(eptr->p_elem))->SDDS_wake);
      }
      ((ZTRANSVERSE *)(eptr->p_elem))->SDDS_wake = NULL;
      ((ZTRANSVERSE *)(eptr->p_elem))->SDDS_wake_initialized = 0;
      break;
    case T_TFBDRIVER:
      if (((TFBDRIVER *)(eptr->p_elem))->outputFile && (((TFBDRIVER *)(eptr->p_elem)))->SDDSout) {
        SDDS_Terminate(((TFBDRIVER *)(eptr->p_elem))->SDDSout);
        free(((TFBDRIVER *)(eptr->p_elem))->SDDSout);
      }
      ((TFBDRIVER *)(eptr->p_elem))->SDDSout = NULL;
      break;
    default:
      break;
    }
    eptr = eptr->succ;
  }
}

void setSigmaIndices() {
  long i, j, k;

  for (i = k = 0; i < 6; i++)
    for (j = i; j < 6; j++, k++) {
      sigmaIndex3[i][j] = sigmaIndex3[j][i] = k;
      sigmaIndex1[k] = i;
      sigmaIndex2[k] = j;
    }
}

#define TYPE_ELECTRON 0
#define TYPE_PROTON 1
#define TYPE_MUON 2
#define TYPE_POSITRON 3
#define TYPE_OTHER 4
#define N_PARTICLE_TYPES 5
static char *particleTypeName[N_PARTICLE_TYPES] = {
  "electron", "proton", "muon", "positron", "custom"};
static double charge[N_PARTICLE_TYPES] = {
  -e_mks, e_mks, -e_mks, e_mks, 0};
static double mass[N_PARTICLE_TYPES] = {
  me_mks, 1.6726485e-27, 1.88353109e-28, me_mks, 0};
// Values from F. Meot article in Polarized Beam Dynamics and Instrumentation in Particle Accelerators
static double anomalousMagneticMoment[N_PARTICLE_TYPES] = {
  0.00115965218128, 1.792847, 1.165921e-3, 0.00115965218128, 0};

void process_particle_command(NAMELIST_TEXT *nltext) {
  long code, i;

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&change_particle, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &change_particle);

  code = match_string(change_particle_struct.name, particleTypeName, N_PARTICLE_TYPES, EXACT_MATCH);

  if (code == TYPE_ELECTRON) {
    particleIsElectron = 1;
    /* ensure exact backward compatibility */
    particleMass = me_mks;
    particleCharge = e_mks;
    particleMassMV = me_mev;
    particleRadius = re_mks;
    particleRelSign = 1;
    particleAnomalousMagneticMoment = 0.00115965218128;
  } else if (code == TYPE_POSITRON) {
    particleIsElectron = 0;
    /* ensure exact reversal from electron */
    particleMass = me_mks;
    particleCharge = -e_mks;
    particleMassMV = me_mev;
    particleRadius = re_mks;
    particleRelSign = -1;
    particleAnomalousMagneticMoment = 0.00115965218128;
  } else {
    particleIsElectron = 0;
    switch (code) {
    case TYPE_PROTON:
    case TYPE_MUON:
      particleMass = mass[code];
      /* minus sign is a legacy of time when this was an electron-only code */
      particleCharge = -charge[code];
      particleAnomalousMagneticMoment = anomalousMagneticMoment[code];
      break;
    case TYPE_OTHER:
      if (change_particle_struct.mass_ratio <= 0 || change_particle_struct.charge_ratio == 0)
        bombElegant("Must have mass_ratio>0 and charge_ratio nonzero", NULL);
      particleMass = mass[TYPE_ELECTRON] * change_particle_struct.mass_ratio;
      /* minus sign is a legacy of time when this was an electron-only code */
      particleCharge = -charge[TYPE_ELECTRON] * change_particle_struct.charge_ratio;
      break;
    default:
      fprintf(stderr, "Unknown particle type.  Known types are \n");
      for (i = 0; i < N_PARTICLE_TYPES; i++)
        fprintf(stderr, "%s%c", particleTypeName[i], i == (N_PARTICLE_TYPES - 1) ? '\n' : ' ');
      exitElegant(1);
      break;
    }
    particleRelSign = -SIGN(particleCharge) / SIGN(charge[TYPE_ELECTRON]);
    particleMassMV = particleMass * sqr(c_mks) / fabs(particleCharge) / 1e6;
    particleRadius = sqr(particleCharge) / (4 * PI * epsilon_o * particleMass * sqr(c_mks));
  }
  /*
    printf("particleMass = %e, particleRadius = %e, particleCharge = %e, particleMassMV = %e, particleRelSign = %e\n",
    particleMass, particleRadius, particleCharge, particleMassMV, particleRelSign);
  printWarning("Changing the particle type is not a fully tested feature",
               ". Please be alert for and report results that don't make sense.");
  */
}

void bombTracking(const char *error) {
  TRACKING_CONTEXT tc;
  getTrackingContext(&tc);
#if USE_MPI
  /* allow slaves to print messages. may get many copies */
  dup2(fdStdout, fileno(stdout));
#endif
  printf("error:  %s\n", error);
  if (strlen(tc.elementName))
    printf("Tracking through %s#%ld at s=%lem\n",
           tc.elementName, tc.elementOccurrence, tc.zEnd);
  else
    printf("Tracking through unidentified element\n");
  show_traceback(stdout);
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (isSlave)
    MPI_Comm_free(&workers);
  MPI_Group_free(&worker_group);
  close(fdStdout);
  MPI_Finalize();
#endif
  exit(1);
}

/* This version accepts a printf-style template and variable number of arguments to be printed */
void bombElegantVA(char *template, ...) {
  char *p;
  char c, *s;
  int i;
  long j;
  va_list argp;
  double d;

#if USE_MPI
  /* allow slaves to print messages. may get many copies */
  dup2(fdStdout, fileno(stdout));
#endif

  va_start(argp, template);
  p = template;
  while (*p) {
    if (*p == '%') {
      switch (*++p) {
      case 'l':
        switch (*++p) {
        case 'd':
          j = va_arg(argp, long int);
          printf("%ld", j);
          break;
        case 'e':
          d = va_arg(argp, double);
          printf("%21.15le", d);
          break;
        case 'f':
          d = va_arg(argp, double);
          printf("%lf", d);
          break;
        case 'g':
          d = va_arg(argp, double);
          printf("%21.15lg", d);
          break;
        default:
          printf("%%l%c", *p);
          break;
        }
        break;
      case 'c':
        c = va_arg(argp, int);
        putchar(c);
        break;
      case 'd':
        i = va_arg(argp, int);
        printf("%d", i);
        break;
      case 's':
        s = va_arg(argp, char *);
        fputs(s, stdout);
        break;
      case 'e':
        d = va_arg(argp, double);
        printf("%21.15e", d);
        break;
      case 'f':
        d = va_arg(argp, double);
        printf("%f", d);
        break;
      case 'g':
        d = va_arg(argp, double);
        printf("%21.15g", d);
        break;
      default:
        printf("%%%c", *p);
        break;
      }
    } else {
      putchar(*p);
    }
    p++;
  }
  va_end(argp);
  bombElegant(NULL, NULL);
}

void bombElegant(const char *error, const char *usage) {
#if USE_MPI
  /* allow slaves to print messages. may get many copies */
  if (error || usage)
    dup2(fdStdout, fileno(stdout));
#endif
  if (error)
    printf("error: %s\n", error);
  if (usage)
    printf("usage: %s\n", usage);
  if (semaphoreFile[2])
    createSemaphoreFile(semaphoreFile[2]);
  show_traceback(stdout);
#if USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (isSlave)
    MPI_Comm_free(&workers);
  if (worker_group)
    MPI_Group_free(&worker_group);
  close(fdStdout);
  MPI_Finalize();
#endif
  summarizeWarnings();
  exit(1);
}

#if USE_MPI
void mpiSetAbort() {
  mpiAbort = 1;
}
#endif

void exitElegant(long status) {
#if USE_MPI
  /* allow slaves to print messages. may get many copies */
  if (status)
    dup2(fdStdout, fileno(stdout));
#endif
  if (status && semaphoreFile[2])
    createSemaphoreFile(semaphoreFile[2]);
  if (!status && semaphoreFile[1])
    createSemaphoreFile(semaphoreFile[1]);
  if (!status && semaphore_file)
    createSemaphoreFile(semaphore_file);
#if USE_MPI
  if (status) {
    /* Abnormal/error exit (including from the SIGSEGV traceback handler): the
     * other ranks are not guaranteed to reach this same collective path -- they
     * may be blocked in a different collective (e.g. MPI_Allreduce in
     * new_sdds_beam). A Barrier/Finalize here would therefore deadlock the whole
     * job and hide the failure as a hang. Abort the entire MPI job instead so the
     * error terminates immediately and visibly. */
    fflush(stdout);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, (int)status);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (isSlave)
    MPI_Comm_free(&workers);
  MPI_Group_free(&worker_group);
  close(fdStdout);
  MPI_Finalize();
#endif
  exit(status);
}

void runFiducialParticle(RUN *run, VARY *control, double *startCoord, LINE_LIST *beamline, short final, short mustSurvive) {
  double **coord, pCentral;
  long code;
#if USE_MPI
  long distributedBeam0, partOnMaster0;
  distributedBeam0 = distributedBeam;
  partOnMaster0 = partOnMaster;
  distributedBeam = 0;
  partOnMaster = 1;
#endif

  /* Prevent do_tracking() from recognizing these flags. Instead, we'll control behavior directly */
  /* beamline->fiducial_flag = 0; */

  coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);
  if (startCoord)
    memcpy(coord[0], startCoord, sizeof(double) * 6);
  else
    memset(coord[0], 0, sizeof(**coord) * 6);
  coord[0][6] = 1;
  pCentral = run->p_central;
  printf("Tracking fiducial particle (runFiducialParticle)\n");
  fflush(stdout);
  if (!(code = do_tracking(NULL, coord, 1, NULL, beamline, &pCentral,
                           NULL, NULL, NULL, NULL, run, control->i_step,
                           (control->fiducial_flag &
                            (LINEAR_CHROMATIC_MATRIX + LONGITUDINAL_RING_ONLY + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + FIDUCIAL_BEAM_SEEN + RESTRICT_FIDUCIALIZATION + PRECORRECTION_BEAM + IBS_ONLY_TRACKING + RESET_RF_FOR_EACH_STEP)) |
                           ALLOW_MPI_ABORT_TRACKING | INHIBIT_FILE_OUTPUT,
                           1, 0, NULL, NULL, NULL, NULL, NULL))) {
    if (mustSurvive) {
      printf("Fiducial particle lost. Don't know what to do.\n");
      exitElegant(1);
    } else {
      printWarning("Fiducial particle lost", NULL);
    }
  } else {
    printf("Tracking fiducial particle completed.\n");
    fflush(stdout);
  }
#if USE_MPI
  distributedBeam = distributedBeam0;
  partOnMaster = partOnMaster0;
#endif
  if (control->fiducial_flag & FIRST_BEAM_IS_FIDUCIAL && final) {
    control->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
    beamline->fiducial_flag |= FIDUCIAL_BEAM_SEEN; /* This is the one that matters */
  }
}

void watchMemory(long *buffer, char *description, long report) {
  if (!report) {
    *buffer = memoryUsage();
  } else {
    long temp;
    temp = memoryUsage();
    if (temp != *buffer) {
      printf("%s: memory changed by %ld\n", description, temp - *buffer);
    }
    *buffer = temp;
  }
}

void process_change_start(NAMELIST_TEXT *nltext, CHANGE_START_SPEC *css) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&change_start, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &change_start);

  if (css->elementName)
    free(css->elementName);

  css->active = 1;
  css->elementName = change_start_struct.element_name;
  css->ringMode = change_start_struct.ring_mode;
  css->elementOccurence = change_start_struct.element_occurence;
  css->deltaPosition = change_start_struct.delta_position;
}

void process_change_end(NAMELIST_TEXT *nltext, CHANGE_END_SPEC *ces) {
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&change_end, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &change_end);

  if (ces->elementName)
    free(ces->elementName);

  ces->active = 1;
  ces->elementName = change_end_struct.element_name;
  ces->elementOccurence = change_end_struct.element_occurence;
  ces->deltaPosition = change_end_struct.delta_position;
}

void manageSemaphoreFiles(char *semaphore_file, char *rootname, char *semaphoreFile[3]) {
  char *lastSem;
  long i;

#if USE_MPI
  if (myid != 0)
    return;
#endif

  if (semaphore_file && fexists(semaphore_file)) {
    if (allowOverwrite)
      remove(semaphore_file);
    else {
      printf("Error: file %s already exists\n", semaphore_file);
      exit(1);
    }
  }

  if (semaphoreFile[0]) {
    /* "started" */
    cp_str(&lastSem, semaphoreFile[0]);
    semaphoreFile[0] = compose_filename(semaphoreFile[0], rootname);
  }
  if (semaphoreFile[1]) {
    /* "done" */
    semaphoreFile[1] = compose_filename(semaphoreFile[1], rootname);
  }
  if (semaphoreFile[2]) {
    /* "failed" */
    semaphoreFile[2] = compose_filename(semaphoreFile[2], rootname);
  }

  for (i = 0; i < 3; i++)
    if (semaphoreFile[i] && strlen(semaphoreFile[i]) && fexists(semaphoreFile[i])) {
      if (allowOverwrite)
        remove(semaphoreFile[i]);
      else {
        printf("Error: file %s already exists\n", semaphoreFile[i]);
        exit(1);
      }
    }

  if (semaphoreFile[0]) {
    /* "started" */
    if (strcmp(lastSem, semaphoreFile[0]))
      createSemaphoreFile(semaphoreFile[0]);
    free(lastSem);
  }
}

long SDDS_IdentifyTypeOfData(char *data0)
{
#define DATA_IS_INT 0x01UL
#define DATA_IS_FLOAT 0x02UL
  unsigned long flags = DATA_IS_INT|DATA_IS_FLOAT;
  short exponentialSeen=0, periodSeen=0;
  char *data;
  long longValue;
  double doubleValue;

  data = data0;
  if (*data=='+' || *data=='-')
    data++;

  while (*data && flags && !exponentialSeen) {
    if (*data<='/' && !(*data=='.' && !periodSeen)) {
      flags = 0;
      break;
    }
    if (*data>=':' || *data=='.') {
      flags &= ~DATA_IS_INT;
      if (*data=='.' && flags&DATA_IS_FLOAT) {
        if (periodSeen)
          /* can't have two periods */
          flags &= ~DATA_IS_FLOAT;
        periodSeen = 1;
        if (data!=data0 && !(isdigit(*(data-1)) || *(data-1)=='+' || *(data-1)=='-'))
          /* Period is not preceed by a number or sign */
          flags &= ~DATA_IS_FLOAT;
      }
      if ((*data=='e' || *data=='E') && flags&DATA_IS_FLOAT) {
	/* exponent if [.0123456789]e[+-0123456789] */
	if (data==data0 || (!isdigit(*(data-1)) && *(data-1)!='.'))
	  flags &= ~DATA_IS_FLOAT;
	else if (*(data+1)=='+' || *(data+1)=='-' || isdigit(*(data+1))) {
	  data ++;
	  exponentialSeen = 1;
	} else {
	  flags &= ~DATA_IS_FLOAT;
	}
      }
    }
    data++;
  }
  while (*data && exponentialSeen) {
    if (!isdigit(*data)) {
      flags = 0;
      break;
    }
    data++;
  }
  if ((flags&DATA_IS_INT) && sscanf(data0, "%ld", &longValue)==1)
    return SDDS_LONG;
  if ((flags&DATA_IS_FLOAT) && sscanf(data0, "%lf", &doubleValue)==1)
    return SDDS_DOUBLE;
  return SDDS_STRING;
}

void doMacroOutput(NAMELIST_TEXT *nltext, RUN *run, char **macroTag, char **macroValue, long macros)
{
  char *filename;
#define COLUMN_MACRO_MODE 0
#define PARAMETER_MACRO_MODE 1
#define DICTIONARY_MACRO_MODE 2
#define N_MACRO_MODES 3
  char *option[N_MACRO_MODES] = {"columns", "parameters", "dictionary"};
  SDDS_DATASET SDDSout;
  long i, *type, longValue;
  double doubleValue;

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&macro_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &macro_output);

  if (macros==0)
    bombElegant("Requested macro output but no macros defined", NULL);

  filename = compose_filename(macro_output_struct.filename, run->rootname);
  type = tmalloc(sizeof(*type)*macros);

#if USE_MPI 
  if (myid==0) {
#endif
    if (!SDDS_InitializeOutputElegant(&SDDSout, SDDS_BINARY, 0, NULL, NULL, filename)) {
#if USE_MPI
      fprintf(stderr,  "Problem creating macro output file %s\n", filename);
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      bombElegantVA("Problem creating macro output file %s\n", filename);
#endif
    }
#if USE_MPI
  }
#endif

  switch (match_string(macro_output_struct.mode, option, N_MACRO_MODES, 0)) {
  case COLUMN_MACRO_MODE:
#if USE_MPI 
    if (myid==0) {
#endif
      for (i=0; i<macros; i++) {
        type[i] = SDDS_IdentifyTypeOfData(macroValue[i]);
        if (!SDDS_DefineSimpleColumn(&SDDSout, macroTag[i], NULL, type[i])) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
          MPI_Abort(MPI_COMM_WORLD, 1);
#else
          exitElegant(1);
#endif       
        }
      }
      if (!SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, 1)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exitElegant(1);
#endif       
      }
      for (i=0; i<macros; i++) {
        switch (type[i]) {
        case SDDS_LONG:
          if (!sscanf(macroValue[i], "%ld", &longValue) ||
              !SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, 0, i, longValue, -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        case SDDS_DOUBLE:
          if (!sscanf(macroValue[i], "%lf", &doubleValue) ||
              !SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, 0, i, doubleValue, -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        case SDDS_STRING:
          if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, 0, i, macroValue[i], -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        default:
          break;
        }
      }
#if USE_MPI 
    }
#endif
    break;
  case PARAMETER_MACRO_MODE:
#if USE_MPI 
    if (myid==0) {
#endif
      for (i=0; i<macros; i++) {
        type[i] = SDDS_IdentifyTypeOfData(macroValue[i]);
        if (!SDDS_DefineSimpleParameter(&SDDSout, macroTag[i], NULL, type[i])) {
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
          MPI_Abort(MPI_COMM_WORLD, 1);
#else
          exitElegant(1);
#endif       
        }
      }
      if (!SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, 1)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exitElegant(1);
#endif       
      }
      for (i=0; i<macros; i++) {
        switch (type[i]) {
        case SDDS_LONG:
          if (!sscanf(macroValue[i], "%ld", &longValue) ||
              !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, i, longValue, -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        case SDDS_DOUBLE:
          if (!sscanf(macroValue[i], "%lf", &doubleValue) ||
              !SDDS_SetParameters(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, i, doubleValue, -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        case SDDS_STRING: 
          if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, i, macroValue[i], -1)) {
            fprintf(stderr, "Problem setting or scanning value %s for macro %s\n",
                    macroValue[i], macroTag[i]);
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exitElegant(1);
#endif       
          }
          break;
        default:
#if USE_MPI
          fprintf(stderr, "Invalid data type writing macro file\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
#else
          bombElegantVA("invalid data type %ld writing macro file (%s=%s)\n", type[i], macroTag[i], macroValue[i]);
#endif       
          break;
        }
      }
#if USE_MPI 
    }
#endif
    break;
  case DICTIONARY_MACRO_MODE:
#if USE_MPI 
    if (myid==0) {
#endif
      if (!SDDS_DefineSimpleColumn(&SDDSout, "MacroTag", NULL, SDDS_STRING) ||
          !SDDS_DefineSimpleColumn(&SDDSout, "MacroStringValue", NULL, SDDS_STRING)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exitElegant(1);
#endif       
      }
      if (!SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, macros) ||
          !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, macroTag, macros, 0) ||
          !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, macroValue, macros, 1)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exitElegant(1);
#endif       
      }
#if USE_MPI 
    }
#endif
    break;
  default:
    bombElegantVA("unknown mode \"%s\" for macro_output\n", macro_output_struct.mode);
    break;
  }

#if USE_MPI
  if (myid==0) {
#endif
    if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout)) {
      fprintf(stderr, "Problem writing macro_output file %s\n", filename);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#else
      bombElegant(NULL, NULL);
#endif       
    }
#if USE_MPI
  }
#endif
}
