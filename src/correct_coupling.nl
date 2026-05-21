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
    double measurement_noise = 0.0;
    double measurement_noise_cutoff = 3.0;
    double etay_weight = 1.0;
    STRING cross_h_steering = NULL;
    STRING cross_v_steering = NULL;
    STRING cross_x_bpm_name_pattern = NULL;
    STRING cross_x_bpm_type_pattern = "MONI HMON";
    STRING cross_y_bpm_name_pattern = NULL;
    STRING cross_y_bpm_type_pattern = "MONI VMON";
    double cross_response_weight = 0.0;
    double cross_steering_kick = 1e-5;
    double cross_measurement_noise = 0.0;
    STRING strength_log = NULL;
    STRING etay_file = NULL;
    STRING response_file = NULL;
    STRING rms_log = NULL;
    long verbosity = 0;
#end

#namelist compute_coupling_response_matrix,struct
    STRING filename = NULL;
    STRING cross_filename = NULL;
    STRING name_pattern = NULL;
    STRING type_pattern = NULL;
    STRING item = "K1";
    STRING bpm_name_pattern = NULL;
    STRING bpm_type_pattern = "MONI";
    STRING cross_h_steering = NULL;
    STRING cross_v_steering = NULL;
    STRING cross_x_bpm_name_pattern = NULL;
    STRING cross_x_bpm_type_pattern = "MONI HMON";
    STRING cross_y_bpm_name_pattern = NULL;
    STRING cross_y_bpm_type_pattern = "MONI VMON";
    double cross_steering_kick = 1e-5;
    double response_perturbation = 1e-6;
    double measurement_noise = 0.0;
    double cross_measurement_noise = 0.0;
    double measurement_noise_cutoff = 3.0;
    long verbosity = 0;
#end

#namelist load_coupling_response_matrix,struct
    STRING filename = NULL;
    STRING cross_filename = NULL;
    long verbosity = 0;
#end
