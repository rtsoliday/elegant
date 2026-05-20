/* file: correct_lattice.nl
 * contents: namelist for correct_lattice command
 */
#include "namelist.h"

#namelist correct_lattice
    STRING name_pattern = NULL;
    STRING type_pattern = NULL;
    STRING bind_name_pattern = NULL;
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
    double beta_measurement_noise = 0.0;
    double eta_measurement_noise = 0.0;
    double measurement_noise_cutoff = 3.0;
    STRING reference_file = NULL;
    double betax_weight = 1.0;
    double betay_weight = 1.0;
    double etax_weight = 1.0;
    STRING strength_log = NULL;
    STRING response_file = NULL;
    STRING rms_log = NULL;
    long verbosity = 0;
#end

#namelist compute_lattice_response_matrix,struct
    STRING filename = NULL;
    STRING name_pattern = NULL;
    STRING type_pattern = NULL;
    STRING bind_name_pattern = NULL;
    STRING item = "K1";
    STRING bpm_name_pattern = NULL;
    STRING bpm_type_pattern = "MONI";
    double response_perturbation = 1e-6;
    double measurement_noise = 0.0;
    double measurement_noise_cutoff = 3.0;
    long verbosity = 0;
#end

#namelist load_lattice_response_matrix,struct
    STRING filename = NULL;
    long verbosity = 0;
#end
