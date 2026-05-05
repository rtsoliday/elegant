/*************************************************************************\
* Copyright (c) 2023 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2023 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
//#include "mdbsun.h"
#include "track.h"
#include "global_settings.h"

void processGlobalSettings(NAMELIST_TEXT *nltext) {
  memcpy(tracking_matrix_step_size, trackingMatrixStepSize, sizeof(*trackingMatrixStepSize) * 6);
  malign_method = misalignmentMethod;
  inhibit_fsync = inhibitFileSync;
  allow_overwrite = allowOverwrite;
  echo_namelists = echoNamelists;
  mpi_randomization_mode = mpiRandomizationMode;
  SR_gaussian_limit = srGaussianLimit;
  exact_normalized_emittance = exactNormalizedEmittance;
  share_tracking_based_matrices = shareTrackingBasedMatrices;
  tracking_based_matrices_store_limit = trackingBasedMatricesStoreLimit;
  warning_limit = warningCountLimit;
  tracking_matrix_step_factor = trackingMatrixStepFactor;
  tracking_matrix_points = trackingMatrixPoints;
  tracking_matrix_max_fit_order = trackingMatrixMaxFitOrder;
  tracking_matrix_cleanup = trackingMatrixCleanUp;
  parallel_tracking_based_matrices = parallelTrackingBasedMatrices;
  slope_limit = slopeLimit;
  coord_limit = coordLimit;
  
  set_namelist_processing_flags(0);
  set_print_namelist_flags(0);
  if (processNamelist(&global_settings, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  if (echoNamelists)
    print_namelist(stdout, &global_settings);

  set_namelist_buffer_size(namelist_buffer_size_factor*16384);
  misalignmentMethod = malign_method;
  inhibitFileSync = inhibit_fsync;
  allowOverwrite = allow_overwrite;
  echoNamelists = echo_namelists;
  mpiRandomizationMode = mpi_randomization_mode;
  srGaussianLimit = SR_gaussian_limit;
  exactNormalizedEmittance = exact_normalized_emittance;
  inhibitRandomSeedPermutation(inhibit_seed_permutation);
  shareTrackingBasedMatrices = share_tracking_based_matrices;
  trackingBasedMatricesStoreLimit = tracking_based_matrices_store_limit;
  warningCountLimit = warning_limit;
  trackingMatrixStepFactor = tracking_matrix_step_factor;
  trackingMatrixPoints = tracking_matrix_points;
  trackingMatrixMaxFitOrder = tracking_matrix_max_fit_order;
  trackingMatrixCleanUp = tracking_matrix_cleanup;
  cp_str(&searchPath, search_path);
  memcpy(trackingMatrixStepSize, tracking_matrix_step_size, sizeof(*trackingMatrixStepSize) * 6);
  parallelTrackingBasedMatrices = parallel_tracking_based_matrices;
  slopeLimit = slope_limit;
  coordLimit = coord_limit;
#if SDDS_MPI_IO
  SDDS_MPI_SetWriteKludgeUsleep(usleep_mpi_io_kludge);
  SDDS_MPI_SetFileSync(mpi_io_force_file_sync);
  if (mpi_io_read_buffer_size)
    SDDS_SetDefaultReadBufferSize(mpi_io_read_buffer_size);
  if (mpi_io_write_buffer_size)
    SDDS_SetDefaultWriteBufferSize(mpi_io_write_buffer_size);
#endif
  if (log_file)
    if (!freopen(log_file, "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
  if (error_log_file)
    if (!freopen(error_log_file, "w", stderr)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
}

