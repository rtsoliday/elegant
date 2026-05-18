/* file: error.nl
 * purpose: namelist for random errors
 * 
 * Michael Borland, 1991
 */
#include "namelist.h"

#namelist error_control static
    long clear_error_settings = 1;
    long summarize_error_settings = 0;
    STRING error_log = NULL;
    long no_errors_for_first_step = 0;
    double error_factor = 1;
#end

#namelist error_element static
    STRING name = NULL;
    STRING exclude = NULL;
    STRING item = NULL;
    STRING element_type = NULL;
    STRING type = "gaussian";
    double amplitude = 0.0;
    double cutoff = 3.0;
    long bind = 1;
    long bind_number = 0;
    long bind_across_names = 0;
    long fractional = 0;
    long post_correction = 0;
    long additive = 1;
    long allow_missing_elements = 0;
    STRING before = NULL;
    STRING after = NULL;
    STRING sample_file = NULL;
    STRING sample_file_column = NULL;
    STRING sample_mode = NULL;
#end

#define UNIFORM_ERRORS 0
#define GAUSSIAN_ERRORS 1
#define PLUS_OR_MINUS_ERRORS 2
#define SAMPLED_ERRORS 3
#define N_ERROR_TYPES 4
static char *known_error_type[N_ERROR_TYPES] = {
  "uniform", "gaussian", "plus_or_minus", "sampled"
};

#define SAMPLE_RANDOM_REPLACE 0
#define SAMPLE_RANDOM_EXHAUST_REUSE 1
#define SAMPLE_SEQUENTIAL_REUSE 2
#define N_SAMPLE_MODES 3
static char *sampleModeChoice[N_SAMPLE_MODES] = {
  "random", "shuffle", "sequential",
};
