/* file: chrom.nl
 * contents: namelist for chromaticity correction
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist chromaticity
    STRING sextupoles = NULL;
    STRING lower_limits = NULL;
    STRING upper_limits = NULL;
    STRING items = NULL;
    STRING exclude = NULL;
    double dnux_dp = 0;
    double dnuy_dp = 0;
    double sextupole_tweek = 1e-3;
    double correction_fraction = 0.9;
    double min_correction_fraction = 0;
    long n_iterations = 5;
    double tolerance = 0;
    STRING strength_log = NULL;
    long change_defined_values = 0;
    double strength_limit = 0;
    long use_perturbed_matrix = 0;    
    long exit_on_failure = 0;
    long update_orbit = 0;
    long reset_correctors_each_step = 1;
    long verbosity = 1;
    double dK2_weight = 1;
    STRING response_matrix_output = NULL;
    STRING correction_matrix_output = NULL;
    long fse_units = 0;
#end


