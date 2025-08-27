/* file: moments.nl
 * contents: namelist for moments calculation
 * 
 * Michael Borland, 2007
 */
#include "namelist.h"

#namelist moments_output
    STRING filename = NULL;
    STRING matrix_output = NULL;
    long output_at_each_step = 0;
    long output_before_tune_correction = 0;
    long final_values_only = 0;
    long verbosity = 0;
    long matched = 1;
    long equilibrium = 1;
    long radiation = 1;
    long ibs_iterations = 0;
    long n_slices = 10;
    long slice_etilted = 1;
    long tracking_based_diffusion_matrix_particles = 1000;
    double emit_x = 0;
    double beta_x = 0;
    double alpha_x = 0;
    double eta_x = 0;
    double etap_x = 0;
    double emit_y = 0;
    double beta_y = 0;
    double alpha_y = 0;
    double eta_y = 0;
    double etap_y = 0;
    double emit_z = 0;
    double beta_z = 0;
    double alpha_z = 0;
    STRING reference_file = NULL;
    STRING reference_element = NULL;
    long reference_element_occurrence = 0;
    long reflect_reference_values = 0;
#end

