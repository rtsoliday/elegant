/* file: twiss.nl
 * contents: namelist for twiss_output
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist setup_linear_chromatic_tracking,struct
    double nux[4] = {-1, 0, 0, 0};
    double betax[2] = {1.0, 0.0};
    double alphax[2] = {0.0, 0.0};
    double etax[2] = {0.0, 0.0};
    double etapx[2] = {0.0, 0.0};
    double nuy[4] = {-1, 0, 0, 0};
    double betay[2] = {1.0, 0.0};
    double alphay[2] = {0.0, 0.0};
    double etay[2] = {0.0, 0.0};
    double etapy[2] = {0.0, 0.0};
    double alphac[2] = {0.0, 0.0};
#end

#namelist tune_shift_with_amplitude,struct
    long turns = 2048;
    double x0 = 1e-6;
    double y0 = 1e-6;
    double x1 = 3e-4;
    double y1 = 3e-4;
    long lines_only = 1;
    long grid_size = 6;
    long sparse_grid = 0;
    long spread_only = 0;
    long exclude_lost_particles = 1;
    double nux_roi_width = 0.02;
    double nuy_roi_width = 0.02;
    double scale_down_factor = 2;
    double scale_up_factor = 1.05;
    double scale_down_limit = 0.01;
    double scale_up_limit = 1e-4;
    long scaling_iterations = 10;
    long use_concatenation = 0;
    long verbose = 0;
    long order = 2;
    STRING tune_output = NULL;
#end

#namelist twiss_output
    STRING filename = NULL;
    long matched = 1;
    long output_at_each_step = 0;
    long output_before_tune_correction = 0;
    long final_values_only = 0;
    long statistics = 0;
    long radiation_integrals = 0;
    double beta_x = 1;
    double alpha_x = 0;
    double eta_x = 0;
    double etap_x = 0;
    double beta_y = 1;
    double alpha_y = 0;
    double eta_y = 0;
    double etap_y = 0;
    STRING reference_file = NULL;
    STRING reference_element = NULL;
    long reference_element_occurrence = 0;
    long reflect_reference_values = 0;
    long concat_order = 3;
    long higher_order_chromaticity = 0;
    long higher_order_chromaticity_points = 5;
    double higher_order_chromaticity_range = 4e-4;
    long quick_higher_order_chromaticity = 0;
    double chromatic_tune_spread_half_range = 0;
    long cavities_are_drifts_if_matched = 1;
    long compute_driving_terms = 0;
    long leading_order_driving_terms_only = 0;
    STRING s_dependent_driving_terms_file = NULL;
    long local_dispersion = 1;
    long n_periods = 1;
#end

#namelist twiss_analysis,struct
        STRING match_name = NULL;
        STRING start_name = NULL;
        STRING end_name = NULL;
        long start_occurence = 1;
        long end_occurence = 1;
        double s_start = -1;
        double s_end = -1;
        STRING tag = NULL;
        long verbosity = 0;
        long clear = 0;
#end

#namelist rf_setup,struct
        STRING filename = NULL;
        STRING bucket_filename = NULL;
        long bucket_points = 200;
        STRING name = NULL;
        long start_occurence = -1;
        long end_occurence = -1;
        double s_start = -1;
        double s_end = -1;
        long set_for_each_step = 0;
        double near_frequency = 0;
	double fractional_frequency_change = 0;
        long harmonic = -1;
        double bucket_half_height = 0;
        double over_voltage = 0;
	double total_voltage = 0;
	long disable = 0;
        long output_only = 0;
        long track_for_frequency = 0;
        double phase_offset = 0;
#end
