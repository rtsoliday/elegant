/* file: global.nl
 * contents: namelist for global settings
 * 
 * Michael Borland, 2023
 */
#include "namelist.h"

#namelist global_settings static
     long inhibit_fsync = 0;
     long allow_overwrite = 1;
     long echo_namelists = 1;
     long mpi_randomization_mode = 3;
     long exact_normalized_emittance = 0;
     double SR_gaussian_limit = 3.0;
     long inhibit_seed_permutation = 0;
     STRING log_file = NULL;
     STRING error_log_file = NULL;
     long share_tracking_based_matrices = 1;
     long tracking_based_matrices_store_limit = 5000;
     long parallel_tracking_based_matrices = 1;
     long mpi_io_force_file_sync = 0;
     long mpi_io_read_buffer_size = 0;
     long mpi_io_write_buffer_size = 0;
     long usleep_mpi_io_kludge = 0;
     double tracking_matrix_step_factor = 1;
     double tracking_matrix_points = 9;
     double tracking_matrix_max_fit_order = 4;
     double tracking_matrix_step_size[6] = {5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5};
     short tracking_matrix_cleanup = 0;
     long warning_limit = 10;
     short malign_method = 0;
     double slope_limit = SLOPE_LIMIT;
     double coord_limit = COORD_LIMIT;
     STRING search_path = NULL;
     long namelist_buffer_size_factor = 1;
#end

