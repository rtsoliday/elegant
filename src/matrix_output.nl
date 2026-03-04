/* file: matrix_output.nl
 * contents: namelist for matrix_output
 * 
 * Michael Borland, 1989
 */
#include "namelist.h"

#namelist matrix_output
    STRING printout = NULL;
    long printout_order = 1;
    STRING printout_format = "%22.15e ";
    long full_matrix_only = 0;
    long print_element_data = 1;
    long mathematica_full_matrix = 0;
    STRING mathematica_matrix_name = "MFull";
    STRING mathematica_matrix_file = NULL;
    STRING SDDS_output = NULL;
    long SDDS_output_order = 1;
    long individual_matrices = 0;
    long output_at_each_step = 0;
    STRING start_from = NULL;
    long start_from_occurence = 1;
    STRING SDDS_output_match = NULL;
    double suppress_below = 0;
#end






