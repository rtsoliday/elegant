/* file: bunched_bea2m.nl
 * contents: namelist and structures for bunched beam specification
 *           and generation
 * 
 * Michael Borland, 2020
 */
#include "namelist.h"


#namelist bunched_beam_moments static,struct
    STRING bunch = NULL;
    long n_particles_per_bunch = 1;
    long multiply_np_by_cores = 0;
    long use_moments_output_values = 0;

    double S1_beta = 0;
    double S2_beta = 0;
    double S12_beta = 0;
    double S16 = 0;
    double S26 = 0;

    double S3_beta = 0;
    double S4_beta = 0;
    double S34_beta = 0;
    double S36 = 0;
    double S46 = 0;

    double S5 = 0;
    double S6 = 0;
    double S56 = 0;
    
    double spin[3] = {0.0, 0.0, 1.0};

    double time_start = 0;
    double Po = 0.0;
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

