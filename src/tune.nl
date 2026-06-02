/* file: tune.nl
 * contents: namelist for tune correction
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist correct_tunes
    STRING quadrupoles = NULL;
    STRING lower_limits = NULL;
    STRING upper_limits = NULL;
    STRING items = NULL;
    STRING exclude = NULL;
    double tune_x = -1;
    double tune_y = -1;
    long n_iterations = 5;
    double correction_fraction = 0.9;
    double tolerance = 1e-8;
    long step_up_interval = 0;
    double max_correction_fraction = 0.9;
    double delta_correction_fraction = 0.1;
    STRING strength_log = NULL;
    long change_defined_values = 0;
    long use_perturbed_matrix = 0;
    double dK1_weight = 1;
    long update_orbit = 0;
    long reset_correctors_each_step = 1;
    long verbosity = 1;
    STRING response_matrix_output = NULL;
    STRING correction_matrix_output = NULL;
    long fse_units = 0;
#end


