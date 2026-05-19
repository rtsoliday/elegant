/* file: coupled_sigma.nl
 * contents: namelist for coupled_sigma_matrix
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist coupled_twiss_output
    STRING filename = NULL;
    long output_at_each_step = 0;
    long emittances_from_twiss_command = 1;
    double emittance_ratio = 0.01;
    double emit_x = 0;
    double sigma_dp = 0;
    long calculate_3d_coupling = 1;
    long verbosity = 0;
    long concat_order = 2;
    long output_sigma_matrix = 0;
    long matched = 1;
    double beta_x1 = 1.0;
    double beta_x2 = 0.0;
    double beta_y1 = 0.0;
    double beta_y2 = 1.0;
    double alpha_x1 = 0.0;
    double alpha_x2 = 0.0;
    double alpha_y1 = 0.0;
    double alpha_y2 = 0.0;
    double eta_x = 0.0;
    double etap_x = 0.0;
    double eta_y = 0.0;
    double etap_y = 0.0;
    double gamma_x1 = -1.0;
    double gamma_x2 = -1.0;
    double gamma_y1 = -1.0;
    double gamma_y2 = -1.0;
    double A_xy_1 = 0.0;
    double A_xpy_1 = 0.0;
    double A_xyp_1 = 0.0;
    double A_xpyp_1 = 0.0;
    double A_xy_2 = 0.0;
    double A_xpy_2 = 0.0;
    double A_xyp_2 = 0.0;
    double A_xpyp_2 = 0.0;
    STRING reference_file = NULL;
    STRING reference_element = NULL;
    long reference_element_occurrence = 0;
    long reflect_reference_values = 0;
#end

