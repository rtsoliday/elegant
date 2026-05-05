/* file: bunched_beam.nl
 * contents: namelist and structures for bunched beam specification
 *           and generation
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist bunched_beam static
    STRING bunch = NULL;
    long n_particles_per_bunch = 1;
    long multiply_np_by_cores = 0;
    double time_start = 0;
    STRING matched_to_cell = NULL;
    double emit_x  = 0;
    double emit_nx  = 0;
    double beta_x  = 1.0;
    double alpha_x = 0.0;
    double eta_x   = 0.0;
    double etap_x  = 0.0;
    double emit_y  = 0;
    double emit_ny  = 0;
    double beta_y  = 1.0;
    double alpha_y = 0.0;
    double eta_y   = 0.0;
    double etap_y  = 0.0;
    double spin[3] = {0.0, 0.0, 1.0};
    long use_twiss_command_values = 0;
    long use_moments_output_values = 0;
    double Po = 0.0;
    double sigma_dp = 0.0;
    double sigma_s = 0.0;
    double dp_s_coupling = 0;
    double emit_z = 0;
    double beta_z = 0;
    double alpha_z = 0;
    double momentum_chirp = 0;
    long one_random_bunch = 1;
    long save_initial_coordinates = 1;
    long limit_invariants = 0;
    long symmetrize = 0;
    long halton_sequence[3] = {0, 0, 0};
    int32_t halton_radix[6] = {0, 0, 0, 0, 0, 0};
    long optimized_halton = 0;
    long randomize_order[3] = {0, 0, 0};
    long limit_in_4d = 0;
    long enforce_rms_values[3] = {0, 0, 0};
    double distribution_cutoff[3] = {2, 2, 2};
    STRING distribution_type[3] = {"gaussian","gaussian","gaussian"};
    double centroid[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    long first_is_fiducial = 0;
#end

