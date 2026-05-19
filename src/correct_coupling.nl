/* file: correct_coupling.nl
 * contents: namelist for correct_coupling command
 */
#include "namelist.h"

#namelist correct_coupling
    STRING name_pattern = NULL;
    STRING type_pattern = NULL;
    STRING item = "K1";
    STRING bpm_name_pattern = NULL;
    STRING bpm_type_pattern = "MONI";
    long n_iterations = 3;
    double convergence = 1e-9;
    double correction_fraction = 0.5;
    long use_perturbed_matrix = 0;
    long adaptive_step = 0;
    double response_perturbation = 1e-6;
    double svd_threshold = 1e-6;
    long n_singular_values = 0;
    double strength_limit = 0;
    STRING strength_log = NULL;
    STRING etay_file = NULL;
    STRING response_file = NULL;
    long verbosity = 0;
#end

#namelist compute_coupling_correction_matrix,struct
    STRING filename = NULL;
    STRING name_pattern = NULL;
    STRING type_pattern = NULL;
    STRING item = "K1";
    STRING bpm_name_pattern = NULL;
    STRING bpm_type_pattern = "MONI";
    double response_perturbation = 1e-6;
    long verbosity = 0;
#end

#namelist load_coupling_correction_matrix,struct
    STRING filename = NULL;
    long verbosity = 0;
#end
