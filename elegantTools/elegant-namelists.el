;;; elegant-namelists.el --- Generated ELEGANT namelist tables -*- lexical-binding: t; -*-
;;;
;;; Auto-generated. Do not edit by hand.
;;;
;;; Provides:
;;;   elegant-namelist-commands
;;;   elegant-namelist-qualifiers
;;;   elegant-namelist-qualifier-types
;;;

(defconst elegant-namelist-commands
  (list
   "alter_elements"
   "amplification_factors"
   "analyze_map"
   "aperture_data"
   "find_aperture"
   "bunched_beam"
   "bunched_beam_moments"
   "chaos_map"
   "chromaticity"
   "closed_orbit"
   "correct"
   "correct_coupling"
   "compute_coupling_response_matrix"
   "load_coupling_response_matrix"
   "correct_lattice"
   "compute_lattice_response_matrix"
   "load_lattice_response_matrix"
   "coupled_twiss_output"
   "divide_elements"
   "elastic_scattering"
   "change_start"
   "change_end"
   "run_setup"
   "change_particle"
   "track"
   "print_dictionary"
   "semaphores"
   "include_commands"
   "macro_output"
   "error_control"
   "error_element"
   "fit_traces"
   "floor_coordinates"
   "frequency_map"
   "global_settings"
   "ignore_elements"
   "inelastic_scattering"
   "insert_sceffects"
   "insert_elements"
   "ion_effects"
   "link_control"
   "link_elements"
   "load_parameters"
   "matrix_output"
   "modulate_elements"
   "moments_output"
   "momentum_aperture"
   "obstruction_data"
   "optimization_covariable"
   "optimization_term"
   "optimization_setup"
   "optimization_variable"
   "optimization_constraint"
   "optimize"
   "set_reference_particle_output"
   "particle_tunes"
   "ramp_elements"
   "replace_elements"
   "correction_matrix_output"
   "rpn_expression"
   "rpn_load"
   "sasefel"
   "save_lattice"
   "sdds_beam"
   "slice_analysis"
   "steering_element"
   "subprocess"
   "touschek_scatter"
   "trace"
   "transmute_elements"
   "correct_tunes"
   "tune_footprint"
   "setup_linear_chromatic_tracking"
   "tune_shift_with_amplitude"
   "twiss_output"
   "twiss_analysis"
   "rf_setup"
   "undulator_brightness"
   "run_control"
   "vary_element"
   )
  "List of ELEGANT #namelist command names.")

(defconst elegant-namelist-qualifiers
  (list
   (cons "alter_elements"
         (list
          "name"
          "item"
          "type"
          "exclude"
          "value"
          "string_value"
          "disable"
          "differential"
          "multiplicative"
          "alter_at_each_step"
          "alter_before_load_parameters"
          "verbose"
          "allow_missing_elements"
          "allow_missing_parameters"
          "start_occurence"
          "end_occurence"
          "occurence_step"
          "s_start"
          "s_end"
          "after"
          "before"
          ))
   (cons "amplification_factors"
         (list
          "output"
          "uncorrected_orbit_function"
          "corrected_orbit_function"
          "kick_function"
          "change"
          "name"
          "type"
          "item"
          "plane"
          "number_to_do"
          "maximum_z"
          ))
   (cons "analyze_map"
         (list
          "output"
          "output_order"
          "printout"
          "printout_order"
          "printout_format"
          "delta_x"
          "delta_xp"
          "delta_y"
          "delta_yp"
          "delta_s"
          "delta_dp"
          "accuracy_factor"
          "center_on_orbit"
          "verbosity"
          "n_points"
          "max_fit_order"
          "canonical_variables"
          "periodic"
          "beta_x"
          "alpha_x"
          "eta_x"
          "etap_x"
          "beta_y"
          "alpha_y"
          "eta_y"
          "etap_y"
          ))
   (cons "aperture_data"
         (list
          "input"
          "periodic"
          "persistent"
          "disable"
          ))
   (cons "find_aperture"
         (list
          "output"
          "search_output"
          "boundary"
          "mode"
          "xmin"
          "xmax"
          "xpmin"
          "xpmax"
          "ymin"
          "ymax"
          "ypmin"
          "ypmax"
          "nx"
          "n_splits"
          "split_fraction"
          "desired_resolution"
          "ny"
          "deltamin"
          "deltamax"
          "ndelta"
          "verbosity"
          "assume_nonincreasing"
          "offset_by_orbit"
          "n_lines"
          "full_plane"
          "optimization_mode"
          ))
   (cons "bunched_beam"
         (list
          "bunch"
          "n_particles_per_bunch"
          "multiply_np_by_cores"
          "time_start"
          "matched_to_cell"
          "emit_x"
          "emit_nx"
          "beta_x"
          "alpha_x"
          "eta_x"
          "etap_x"
          "emit_y"
          "emit_ny"
          "beta_y"
          "alpha_y"
          "eta_y"
          "etap_y"
          "spin"
          "use_twiss_command_values"
          "use_moments_output_values"
          "Po"
          "sigma_dp"
          "sigma_s"
          "dp_s_coupling"
          "emit_z"
          "beta_z"
          "alpha_z"
          "momentum_chirp"
          "one_random_bunch"
          "save_initial_coordinates"
          "limit_invariants"
          "symmetrize"
          "halton_sequence"
          "halton_radix"
          "optimized_halton"
          "randomize_order"
          "limit_in_4d"
          "enforce_rms_values"
          "distribution_cutoff"
          "distribution_type"
          "centroid"
          "first_is_fiducial"
          ))
   (cons "bunched_beam_moments"
         (list
          "bunch"
          "n_particles_per_bunch"
          "multiply_np_by_cores"
          "use_moments_output_values"
          "S1_beta"
          "S2_beta"
          "S12_beta"
          "S16"
          "S26"
          "S3_beta"
          "S4_beta"
          "S34_beta"
          "S36"
          "S46"
          "S5"
          "S6"
          "S56"
          "spin"
          "time_start"
          "Po"
          "one_random_bunch"
          "save_initial_coordinates"
          "limit_invariants"
          "symmetrize"
          "halton_sequence"
          "halton_radix"
          "optimized_halton"
          "randomize_order"
          "limit_in_4d"
          "enforce_rms_values"
          "distribution_cutoff"
          "distribution_type"
          "centroid"
          "first_is_fiducial"
          ))
   (cons "chaos_map"
         (list
          "output"
          "xmin"
          "xmax"
          "ymin"
          "ymax"
          "delta_min"
          "delta_max"
          "nx"
          "ny"
          "ndelta"
          "forward_backward"
          "change_x"
          "change_y"
          "verbosity"
          ))
   (cons "chromaticity"
         (list
          "sextupoles"
          "lower_limits"
          "upper_limits"
          "items"
          "exclude"
          "dnux_dp"
          "dnuy_dp"
          "sextupole_tweek"
          "correction_fraction"
          "min_correction_fraction"
          "n_iterations"
          "tolerance"
          "strength_log"
          "change_defined_values"
          "strength_limit"
          "use_perturbed_matrix"
          "exit_on_failure"
          "update_orbit"
          "reset_correctors_each_step"
          "verbosity"
          "dK2_weight"
          "response_matrix_output"
          "correction_matrix_output"
          "fse_units"
          ))
   (cons "closed_orbit"
         (list
          "output"
          "start_from_centroid"
          "start_from_dp_centroid"
          "closed_orbit_accuracy"
          "closed_orbit_accuracy_requirement"
          "closed_orbit_iterations"
          "fixed_length"
          "start_from_recirc"
          "verbosity"
          "iteration_fraction"
          "fraction_multiplier"
          "fraction_divisor"
          "multiplier_interval"
          "update_matrix"
          "output_monitors_only"
          "tracking_turns"
          "disable"
          "immediate"
          ))
   (cons "correct"
         (list
          "disable"
          "mode"
          "method"
          "trajectory_output"
          "corrector_output"
          "bpm_output"
          "statistics"
          "corrector_tweek"
          "corrector_limit"
          "correction_fraction"
          "correction_accuracy"
          "do_correction"
          "remove_smallest_SVs"
          "keep_largest_SVs"
          "minimum_SV_ratio"
          "auto_limit_SVs"
          "Tikhonov_relative_alpha"
          "Tikhonov_n"
          "remove_pegged"
          "threading_divisor"
          "threading_correctors"
          "bpm_noise"
          "bpm_noise_cutoff"
          "bpm_noise_distribution"
          "verbose"
          "fixed_length"
          "fixed_length_matrix"
          "n_xy_cycles"
          "minimum_cycles"
          "force_alternation"
          "n_iterations"
          "prezero_correctors"
          "reset_correctors_each_step"
          "track_before_and_after"
          "start_from_centroid"
          "use_actual_beam"
          "closed_orbit_accuracy"
          "closed_orbit_accuracy_requirement"
          "closed_orbit_iterations"
          "closed_orbit_iteration_fraction"
          "closed_orbit_fraction_multiplier"
          "closed_orbit_multiplier_interval"
          "closed_orbit_tracking_turns"
          "use_perturbed_matrix"
          "use_response_from_computed_orbits"
          "rpn_store_response_matrix"
          ))
   (cons "correct_coupling"
         (list
          "correction_elements"
          "items"
          "lower_limits"
          "upper_limits"
          "exclude"
          "bpm_name_pattern"
          "bpm_type_pattern"
          "n_iterations"
          "convergence"
          "change_tolerance"
          "correction_fraction"
          "use_perturbed_matrix"
          "adaptive_step"
          "response_perturbation"
          "svd_threshold"
          "n_singular_values"
          "auto_sv_threshold"
          "auto_sv_threshold_factor"
          "measurement_noise"
          "measurement_noise_cutoff"
          "etay_weight"
          "cross_h_steering"
          "cross_v_steering"
          "cross_x_bpm_name_pattern"
          "cross_x_bpm_type_pattern"
          "cross_y_bpm_name_pattern"
          "cross_y_bpm_type_pattern"
          "cross_response_weight"
          "cross_steering_kick"
          "cross_measurement_noise"
          "strength_log"
          "etay_file"
          "response_file"
          "rms_log"
          "reset_correctors_each_step"
          "verbosity"
          ))
   (cons "compute_coupling_response_matrix"
         (list
          "filename"
          "cross_filename"
          "correction_elements"
          "items"
          "exclude"
          "bpm_name_pattern"
          "bpm_type_pattern"
          "cross_h_steering"
          "cross_v_steering"
          "cross_x_bpm_name_pattern"
          "cross_x_bpm_type_pattern"
          "cross_y_bpm_name_pattern"
          "cross_y_bpm_type_pattern"
          "cross_steering_kick"
          "response_perturbation"
          "measurement_noise"
          "cross_measurement_noise"
          "measurement_noise_cutoff"
          "verbosity"
          ))
   (cons "load_coupling_response_matrix"
         (list
          "filename"
          "cross_filename"
          "verbosity"
          ))
   (cons "correct_lattice"
         (list
          "correction_elements"
          "items"
          "lower_limits"
          "upper_limits"
          "exclude"
          "bind_name_pattern"
          "measurement_elements"
          "measurement_types"
          "n_iterations"
          "convergence"
          "change_tolerance"
          "correction_fraction"
          "use_perturbed_matrix"
          "adaptive_step"
          "response_perturbation"
          "svd_threshold"
          "n_singular_values"
          "auto_sv_threshold"
          "auto_sv_threshold_factor"
          "beta_measurement_noise"
          "eta_measurement_noise"
          "measurement_noise_cutoff"
          "reference_file"
          "betax_weight"
          "betay_weight"
          "etax_weight"
          "strength_log"
          "response_file"
          "rms_log"
          "reset_correctors_each_step"
          "verbosity"
          ))
   (cons "compute_lattice_response_matrix"
         (list
          "filename"
          "correction_elements"
          "items"
          "exclude"
          "bind_name_pattern"
          "measurement_elements"
          "measurement_types"
          "response_perturbation"
          "measurement_noise"
          "measurement_noise_cutoff"
          "verbosity"
          ))
   (cons "load_lattice_response_matrix"
         (list
          "filename"
          "verbosity"
          ))
   (cons "coupled_twiss_output"
         (list
          "filename"
          "output_at_each_step"
          "emittances_from_twiss_command"
          "emittance_ratio"
          "emit_x"
          "sigma_dp"
          "calculate_3d_coupling"
          "verbosity"
          "concat_order"
          "output_sigma_matrix"
          "matched"
          "beta_x1"
          "beta_x2"
          "beta_y1"
          "beta_y2"
          "alpha_x1"
          "alpha_x2"
          "alpha_y1"
          "alpha_y2"
          "eta_x"
          "etap_x"
          "eta_y"
          "etap_y"
          "gamma_x1"
          "gamma_x2"
          "gamma_y1"
          "gamma_y2"
          "A_xy_1"
          "A_xpy_1"
          "A_xyp_1"
          "A_xpyp_1"
          "A_xy_2"
          "A_xpy_2"
          "A_xyp_2"
          "A_xpyp_2"
          "reference_file"
          "reference_element"
          "reference_element_occurrence"
          "reflect_reference_values"
          ))
   (cons "divide_elements"
         (list
          "name"
          "type"
          "exclude"
          "exclude_name_pattern"
          "exclude_type_pattern"
          "divisions"
          "maximum_length"
          "clear"
          ))
   (cons "elastic_scattering"
         (list
          "losses"
          "output"
          "log_file"
          "theta_min"
          "theta_max"
          "n_theta"
          "quadratic_theta_spacing"
          "n_phi"
          "twiss_scaling"
          "s_start"
          "s_end"
          "include_name_pattern"
          "include_type_pattern"
          "verbosity"
          "soft_failure"
          "allow_watch_file_output"
          ))
   (cons "change_start"
         (list
          "element_name"
          "ring_mode"
          "element_occurence"
          "delta_position"
          ))
   (cons "change_end"
         (list
          "element_name"
          "element_occurence"
          "delta_position"
          ))
   (cons "run_setup"
         (list
          "lattice"
          "use_beamline"
          "rootname"
          "output"
          "centroid"
          "bpm_centroid"
          "sigma"
          "final"
          "acceptance"
          "losses"
          "losses_include_global_coordinates"
          "losses_s_limit"
          "magnets"
          "profile"
          "semaphore_file"
          "parameters"
          "suppress_parameter_defaults"
          "rfc_reference_output"
          "combine_bunch_statistics"
          "wrap_around"
          "final_pass"
          "default_order"
          "concat_order"
          "print_statistics"
          "show_element_timing"
          "monitor_memory_usage"
          "random_number_seed"
          "correction_iterations"
          "echo_lattice"
          "p_central"
          "p_central_mev"
          "always_change_p0"
          "load_balancing_on"
          "random_sequence_No"
          "expand_for"
          "tracking_updates"
          "search_path"
          "element_divisions"
          "back_tracking"
          "s_start"
          "spin_tracking"
          ))
   (cons "change_particle"
         (list
          "name"
          "mass_ratio"
          "charge_ratio"
          ))
   (cons "track"
         (list
          "center_on_orbit"
          "center_momentum_also"
          "offset_by_orbit"
          "offset_momentum_also"
          "soft_failure"
          "use_linear_chromatic_matrix"
          "longitudinal_ring_only"
          "ibs_only"
          "stop_tracking_particle_limit"
          "check_beam_structure"
          "interrupt_file"
          ))
   (cons "print_dictionary"
         (list
          "filename"
          "latex_form"
          "SDDS_form"
          ))
   (cons "semaphores"
         (list
          "started"
          "done"
          "failed"
          "immediate"
          ))
   (cons "include_commands"
         (list
          "filename"
          "disable"
          ))
   (cons "macro_output"
         (list
          "filename"
          "mode"
          ))
   (cons "error_control"
         (list
          "clear_error_settings"
          "summarize_error_settings"
          "error_log"
          "no_errors_for_first_step"
          "error_factor"
          ))
   (cons "error_element"
         (list
          "name"
          "exclude"
          "item"
          "element_type"
          "type"
          "amplitude"
          "cutoff"
          "bind"
          "bind_number"
          "bind_across_names"
          "fractional"
          "post_correction"
          "additive"
          "allow_missing_elements"
          "before"
          "after"
          "sample_file"
          "sample_file_column"
          "sample_mode"
          ))
   (cons "fit_traces"
         (list
          "trace_data_file"
          "fit_parameters_file"
          "iterations"
          "sub_iterations"
          "use_SVD"
          "SVs_to_keep"
          "SVs_to_remove"
          "BPM_threshold"
          "convergence_factor"
          "convergence_factor_divisor"
          "convergence_factor_multiplier"
          "convergence_increase_steps"
          "convergence_factor_min"
          "convergence_factor_max"
          "position_change_limit"
          "slope_change_limit"
          "trace_sub_iterations"
          "trace_convergence_factor"
          "trace_fractional_target"
          "fit_output_file"
          "trace_output_file"
          "target"
          "tolerance"
          "reject_BPM_common_mode"
          "n_restarts"
          "restart_randomization_level"
          ))
   (cons "floor_coordinates"
         (list
          "filename"
          "X0"
          "Y0"
          "Z0"
          "theta0"
          "phi0"
          "psi0"
          "include_vertices"
          "include_arc_centers"
          "vertices_only"
          "magnet_centers"
          "store_vertices"
          "store_centers"
          ))
   (cons "frequency_map"
         (list
          "output"
          "xmin"
          "xmax"
          "ymin"
          "ymax"
          "delta_min"
          "delta_max"
          "nx"
          "ny"
          "ndelta"
          "verbosity"
          "include_changes"
          "quadratic_spacing"
          "full_grid_output"
          ))
   (cons "global_settings"
         (list
          "inhibit_fsync"
          "allow_overwrite"
          "echo_namelists"
          "mpi_randomization_mode"
          "exact_normalized_emittance"
          "SR_gaussian_limit"
          "inhibit_seed_permutation"
          "log_file"
          "error_log_file"
          "share_tracking_based_matrices"
          "tracking_based_matrices_store_limit"
          "parallel_tracking_based_matrices"
          "mpi_io_force_file_sync"
          "mpi_io_read_buffer_size"
          "mpi_io_write_buffer_size"
          "usleep_mpi_io_kludge"
          "tracking_matrix_step_factor"
          "tracking_matrix_points"
          "tracking_matrix_max_fit_order"
          "tracking_matrix_step_size"
          "tracking_matrix_cleanup"
          "warning_limit"
          "malign_method"
          "slope_limit"
          "coord_limit"
          "search_path"
          "namelist_buffer_size_factor"
          ))
   (cons "ignore_elements"
         (list
          "name"
          "type"
          "exclude"
          "completely"
          "disable"
          "clear_all"
          ))
   (cons "inelastic_scattering"
         (list
          "losses"
          "output"
          "log_file"
          "k_min"
          "momentum_aperture"
          "momentum_aperture_scale"
          "momentum_aperture_periodicity"
          "n_k"
          "s_start"
          "s_end"
          "include_name_pattern"
          "include_type_pattern"
          "verbosity"
          "soft_failure"
          "allow_watch_file_output"
          ))
   (cons "insert_sceffects"
         (list
          "name"
          "type"
          "exclude"
          "disable"
          "clear"
          "element_prefix"
          "skip"
          "vertical"
          "horizontal"
          "longitudinal"
          "nonlinear"
          "uniform_distribution"
          "slice_duration"
          "slice_threshold"
          "slice_interpolation"
          "verbosity"
          "averaging_factor"
          ))
   (cons "insert_elements"
         (list
          "name"
          "type"
          "exclude"
          "skip"
          "disable"
          "insert_before"
          "add_at_start"
          "add_at_end"
          "s_start"
          "s_end"
          "occurrence_start"
          "occurrence_end"
          "start_at_element"
          "end_at_element"
          "element_def"
          "verbose"
          "total_occurrences"
          "occurrence"
          "allow_no_matches"
          "allow_no_insertions"
          "insertion_count_variable"
          ))
   (cons "ion_effects"
         (list
          "pressure_profile"
          "pressure_factor"
          "ion_properties"
          "beam_output"
          "beam_output_all_locations"
          "ion_density_output"
          "ion_output_all_locations"
          "ion_species_output"
          "ion_output_interval"
          "field_calculation_method"
          "conserve_momentum"
          "distribution_fit_target"
          "distribution_fit_tolerance"
          "distribution_fit_evaluations"
          "distribution_fit_passes"
          "distribution_fit_restarts"
          "hybrid_simplex_comparison_interval"
          "fit_residual_type"
          "macro_ions"
          "symmetrize"
          "generation_interval"
          "multiple_ionization_interval"
          "multiple_ionization_energy_peak"
          "multiple_ionization_energy_rms"
          "ion_span"
          "ion_poisson_bins"
          "ion_poisson_span"
          "ion_bin_divisor"
          "ion_range_multiplier"
          "ion_sigma_limit_multiplier"
          "ion_histogram_max_bins"
          "ion_histogram_min_per_bin"
          "ion_histogram_output"
          "ion_2d_histogram_output"
          "ion_histogram_output_s_start"
          "ion_histogram_output_s_end"
          "ion_histogram_output_interval"
          "ion_histogram_min_output_bins"
          "disable_until_pass"
          "freeze_ions_until_pass"
          "freeze_electrons_until_pass"
          "ion_fit_subtract_baseline"
          "gaussian_ion_range"
          "verbosity"
          "use_local_pressure"
          ))
   (cons "link_control"
         (list
          "clear_links"
          "summarize_links"
          "verbosity"
          ))
   (cons "link_elements"
         (list
          "target"
          "target_occurence"
          "exclude"
          "item"
          "source"
          "source_from_target_edit"
          "source_position"
          "mode"
          "equation"
          "minimum"
          "maximum"
          "exclude_self"
          ))
   (cons "load_parameters"
         (list
          "filename"
          "filename_list"
          "include_name_pattern"
          "include_item_pattern"
          "include_type_pattern"
          "exclude_name_pattern"
          "exclude_item_pattern"
          "exclude_type_pattern"
          "edit_name_command"
          "change_defined_values"
          "script_triggered"
          "repeat_first_page_at_each_step"
          "clear_settings"
          "allow_missing_files"
          "allow_missing_elements"
          "allow_missing_parameters"
          "force_occurence_data"
          "verbose"
          "skip_pages"
          "use_first"
          "prefactor"
          ))
   (cons "matrix_output"
         (list
          "printout"
          "printout_order"
          "printout_format"
          "full_matrix_only"
          "print_element_data"
          "mathematica_full_matrix"
          "mathematica_matrix_name"
          "mathematica_matrix_file"
          "SDDS_output"
          "SDDS_output_order"
          "individual_matrices"
          "output_at_each_step"
          "start_from"
          "start_from_occurence"
          "SDDS_output_match"
          "suppress_below"
          ))
   (cons "modulate_elements"
         (list
          "name"
          "item"
          "type"
          "expression"
          "filename"
          "time_column"
          "convert_pass_to_time"
          "amplitude_column"
          "refresh_matrix"
          "differential"
          "multiplicative"
          "factor"
          "start_pass"
          "end_pass"
          "pass_delay"
          "time_delay"
          "start_occurence"
          "end_occurence"
          "s_start"
          "s_end"
          "before"
          "after"
          "verbose"
          "verbose_threshold"
          "record"
          "flush_record"
          ))
   (cons "moments_output"
         (list
          "filename"
          "matrix_output"
          "output_at_each_step"
          "output_before_tune_correction"
          "final_values_only"
          "verbosity"
          "matched"
          "equilibrium"
          "force_e1_gt_e2"
          "radiation"
          "ibs_iterations"
          "ibs_output_iterations"
          "ibs_iteration_fraction"
          "ibs_coulomb_log"
          "n_slices"
          "slice_etilted"
          "tracking_based_diffusion_matrix_particles"
          "emit_x"
          "beta_x"
          "alpha_x"
          "eta_x"
          "etap_x"
          "emit_y"
          "beta_y"
          "alpha_y"
          "eta_y"
          "etap_y"
          "emit_z"
          "beta_z"
          "alpha_z"
          "reference_file"
          "reference_element"
          "reference_element_occurrence"
          "reflect_reference_values"
          ))
   (cons "momentum_aperture"
         (list
          "output"
          "x_initial"
          "y_initial"
          "delta_negative_limit"
          "delta_positive_limit"
          "delta_negative_start"
          "delta_positive_start"
          "delta_step_size"
          "splits"
          "steps_back"
          "split_step_divisor"
          "skip_elements"
          "process_elements"
          "s_start"
          "s_end"
          "include_name_pattern"
          "include_type_pattern"
          "fiducialize"
          "verbosity"
          "soft_failure"
          "output_mode"
          "forbid_resonance_crossing"
          "allow_watch_file_output"
          ))
   (cons "obstruction_data"
         (list
          "input"
          "periods"
          "disable"
          "y_spacing"
          "y_limit"
          ))
   (cons "optimization_covariable"
         (list
          "name"
          "item"
          "equation"
          "disable"
          ))
   (cons "optimization_term"
         (list
          "term"
          "weight"
          "field_string"
          "field_initial_value"
          "field_final_value"
          "field_interval"
          "input_file"
          "input_column"
          "verbose"
          ))
   (cons "optimization_setup"
         (list
          "equation"
          "mode"
          "method"
          "statistic"
          "tolerance"
          "hybrid_simplex_tolerance"
          "hybrid_simplex_tolerance_count"
          "hybrid_simplex_comparison_interval"
          "target"
          "center_on_orbit"
          "center_momentum_also"
          "soft_failure"
          "n_passes"
          "n_evaluations"
          "n_restarts"
          "restart_reset_threshold"
          "restart_worst_term_factor"
          "restart_worst_terms"
          "matrix_order"
          "log_file"
          "term_log_file"
          "verbose"
          "output_sparsing_factor"
          "crossover"
          "balance_terms"
          "simplex_divisor"
          "simplex_pass_range_factor"
          "include_simplex_1d_scans"
          "start_from_simplex_vertex1"
          "restart_random_numbers"
          "random_factor"
          "rcds_step_factor"
          "n_iterations"
          "max_no_change"
          "population_size"
          "print_all_individuals"
          "population_log"
          "interrupt_file"
          "interrupt_file_check_interval"
          "inhibit_tracking"
          "simplex_log"
          "simplex_log_interval"
          ))
   (cons "optimization_variable"
         (list
          "name"
          "item"
          "lower_limit"
          "upper_limit"
          "step_size"
          "fractional_step_size"
          "disable"
          "force_inside"
          "differential_limits"
          "no_element"
          "initial_value"
          ))
   (cons "optimization_constraint"
         (list
          "quantity"
          "lower"
          "upper"
          ))
   (cons "optimize"
         (list
          "summarize_setup"
          ))
   (cons "set_reference_particle_output"
         (list
          "match_to"
          "weight"
          "comparison_mode"
          ))
   (cons "particle_tunes"
         (list
          "filename"
          "start_pid"
          "end_pid"
          "pid_interval"
          "include_x"
          "include_y"
          "include_s"
          "include_spin"
          "start_pass"
          "segment_length"
          ))
   (cons "ramp_elements"
         (list
          "name"
          "item"
          "type"
          "start_pass"
          "end_pass"
          "start_value"
          "end_value"
          "refresh_matrix"
          "differential"
          "multiplicative"
          "start_occurence"
          "end_occurence"
          "exponent"
          "s_start"
          "s_end"
          "before"
          "after"
          "verbose"
          "record"
          ))
   (cons "replace_elements"
         (list
          "name"
          "type"
          "exclude"
          "skip"
          "disable"
          "element_def"
          "total_occurrences"
          "occurrence"
          "verbose"
          ))
   (cons "correction_matrix_output"
         (list
          "response"
          "inverse"
          "slope_response"
          "full_names"
          "KnL_units"
          "BnL_units"
          "output_at_each_step"
          "output_before_tune_correction"
          "fixed_length"
          "coupled"
          "use_response_from_computed_orbits"
          ))
   (cons "rpn_expression"
         (list
          "expression"
          ))
   (cons "rpn_load"
         (list
          "filename"
          "match_column"
          "match_column_value"
          "matching_row_number"
          "match_parameter"
          "match_parameter_value"
          "use_row"
          "use_page"
          "load_parameters"
          "tag"
          ))
   (cons "sasefel"
         (list
          "output"
          "model"
          "beamsize_mode"
          "beta"
          "undulator_K"
          "undulator_period"
          "slice_fraction"
          "n_slices"
          ))
   (cons "save_lattice"
         (list
          "filename"
          "suppress_defaults"
          "output_seq"
          ))
   (cons "sdds_beam"
         (list
          "input"
          "input_list"
          "input_type"
          "selection_parameter"
          "selection_string"
          "one_random_bunch"
          "n_particles_per_ring"
          "reuse_bunch"
          "prebunched"
          "track_pages_separately"
          "use_bunched_mode"
          "fiducialization_bunch"
          "sample_interval"
          "n_tables_to_skip"
          "center_transversely"
          "center_arrival_time"
          "reverse_t_sign"
          "sample_fraction"
          "p_lower"
          "p_upper"
          "save_initial_coordinates"
          "n_duplicates"
          "duplicate_stagger"
          ))
   (cons "slice_analysis"
         (list
          "output"
          "n_slices"
          "s_start"
          "s_end"
          "final_values_only"
          ))
   (cons "steering_element"
         (list
          "name"
          "element_type"
          "item"
          "plane"
          "tweek"
          "limit"
          "start_occurence"
          "end_occurence"
          "occurence_step"
          "s_start"
          "s_end"
          "after"
          "before"
          "verbose"
          ))
   (cons "subprocess"
         (list
          "command"
          ))
   (cons "touschek_scatter"
         (list
          "charge"
          "frequency"
          "emit_x"
          "emit_nx"
          "emit_y"
          "emit_ny"
          "sigma_dp"
          "sigma_s"
          "distribution_cutoff"
          "Momentum_Aperture_scale"
          "Momentum_Aperture"
          "XDist"
          "YDist"
          "ZDist"
          "TranDist"
          "FullDist"
          "bunch"
          "loss"
          "distribution"
          "initial"
          "output"
          "nbins"
          "sbin_step"
          "n_simulated"
          "ignored_portion"
          "i_start"
          "i_end"
          "match_position_only"
          "do_track"
          "verbosity"
          "overwrite_files"
          ))
   (cons "trace"
         (list
          "traceback_on"
          "trace_on"
          "heap_verify_depth"
          "immediate"
          "filename"
          "memory_log"
          "record_allocation"
          ))
   (cons "transmute_elements"
         (list
          "name"
          "type"
          "exclude"
          "new_type"
          "disable"
          "clear_all"
          ))
   (cons "correct_tunes"
         (list
          "quadrupoles"
          "lower_limits"
          "upper_limits"
          "items"
          "exclude"
          "tune_x"
          "tune_y"
          "n_iterations"
          "correction_fraction"
          "tolerance"
          "step_up_interval"
          "max_correction_fraction"
          "delta_correction_fraction"
          "strength_log"
          "change_defined_values"
          "use_perturbed_matrix"
          "dK1_weight"
          "update_orbit"
          "reset_correctors_each_step"
          "verbosity"
          "response_matrix_output"
          "correction_matrix_output"
          "fse_units"
          ))
   (cons "tune_footprint"
         (list
          "delta_output"
          "xy_output"
          "xmin"
          "xmax"
          "ymin"
          "ymax"
          "x_for_delta"
          "y_for_delta"
          "separate_xy_for_delta"
          "delta_min"
          "delta_max"
          "nx"
          "ny"
          "ndelta"
          "verbosity"
          "quadratic_spacing"
          "compute_diffusion"
          "diffusion_rate_limit"
          "immediate"
          "filtered_output"
          "ignore_half_integer"
          "chromaticity_fit_order"
          ))
   (cons "setup_linear_chromatic_tracking"
         (list
          "nux"
          "betax"
          "alphax"
          "etax"
          "etapx"
          "nuy"
          "betay"
          "alphay"
          "etay"
          "etapy"
          "alphac"
          ))
   (cons "tune_shift_with_amplitude"
         (list
          "turns"
          "x0"
          "y0"
          "x1"
          "y1"
          "lines_only"
          "grid_size"
          "sparse_grid"
          "spread_only"
          "exclude_lost_particles"
          "nux_roi_width"
          "nuy_roi_width"
          "scale_down_factor"
          "scale_up_factor"
          "scale_down_limit"
          "scale_up_limit"
          "scaling_iterations"
          "use_concatenation"
          "verbose"
          "order"
          "tune_output"
          ))
   (cons "twiss_output"
         (list
          "filename"
          "matched"
          "output_at_each_step"
          "output_before_tune_correction"
          "final_values_only"
          "statistics"
          "radiation_integrals"
          "beta_x"
          "alpha_x"
          "eta_x"
          "etap_x"
          "beta_y"
          "alpha_y"
          "eta_y"
          "etap_y"
          "reference_file"
          "reference_element"
          "reference_element_occurrence"
          "reflect_reference_values"
          "concat_order"
          "higher_order_chromaticity"
          "higher_order_chromaticity_points"
          "higher_order_chromaticity_range"
          "quick_higher_order_chromaticity"
          "chromatic_tune_spread_half_range"
          "cavities_are_drifts_if_matched"
          "compute_driving_terms"
          "leading_order_driving_terms_only"
          "s_dependent_driving_terms_file"
          "local_dispersion"
          "n_periods"
          ))
   (cons "twiss_analysis"
         (list
          "match_name"
          "start_name"
          "end_name"
          "start_occurence"
          "end_occurence"
          "s_start"
          "s_end"
          "tag"
          "verbosity"
          "clear"
          ))
   (cons "rf_setup"
         (list
          "filename"
          "name"
          "start_occurence"
          "end_occurence"
          "s_start"
          "s_end"
          "set_for_each_step"
          "near_frequency"
          "fractional_frequency_change"
          "harmonic"
          "bucket_half_height"
          "over_voltage"
          "total_voltage"
          "disable"
          "output_only"
          "track_for_frequency"
          "phase_offset"
          ))
   (cons "undulator_brightness"
         (list
          "tag"
          "wavelength"
          "photon_energy"
          "harmonic"
          "detuning"
          "period_length"
          "n_periods"
          "total_length"
          "current"
          "use_twiss_output_values"
          "coupling"
          "twiss_element"
          "twiss_occurence"
          "emitx"
          "emity"
          "betax"
          "alphax"
          "betay"
          "alphay"
          "etax"
          "etaxp"
          "etay"
          "etayp"
          "Sdelta"
          ))
   (cons "run_control"
         (list
          "n_steps"
          "bunch_frequency"
          "n_indices"
          "n_passes"
          "n_passes_fiducial"
          "terminate_on_failure"
          "reset_rf_for_each_step"
          "first_is_fiducial"
          "restrict_fiducialization"
          "reset_scattering_seed"
          "wait_for_step_semaphore"
          "step_done_semaphore"
          "semaphore_check_interval"
          "restart_files"
          ))
   (cons "vary_element"
         (list
          "index_number"
          "index_limit"
          "name"
          "item"
          "initial"
          "final"
          "differential"
          "geometric"
          "multiplicative"
          "enumeration_file"
          "enumeration_column"
          "disable"
          ))
   )
  "Alist mapping namelist command -> list of qualifier names.")

(defconst elegant-namelist-qualifier-types
  (list
   (cons "alter_elements"
         (list
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "value" "double")
          (cons "string_value" "STRING")
          (cons "disable" "long")
          (cons "differential" "long")
          (cons "multiplicative" "long")
          (cons "alter_at_each_step" "long")
          (cons "alter_before_load_parameters" "long")
          (cons "verbose" "long")
          (cons "allow_missing_elements" "long")
          (cons "allow_missing_parameters" "long")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "occurence_step" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "after" "STRING")
          (cons "before" "STRING")
          ))
   (cons "amplification_factors"
         (list
          (cons "output" "STRING")
          (cons "uncorrected_orbit_function" "STRING")
          (cons "corrected_orbit_function" "STRING")
          (cons "kick_function" "STRING")
          (cons "change" "double")
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "item" "STRING")
          (cons "plane" "STRING")
          (cons "number_to_do" "long")
          (cons "maximum_z" "double")
          ))
   (cons "analyze_map"
         (list
          (cons "output" "STRING")
          (cons "output_order" "long")
          (cons "printout" "STRING")
          (cons "printout_order" "long")
          (cons "printout_format" "STRING")
          (cons "delta_x" "double")
          (cons "delta_xp" "double")
          (cons "delta_y" "double")
          (cons "delta_yp" "double")
          (cons "delta_s" "double")
          (cons "delta_dp" "double")
          (cons "accuracy_factor" "double")
          (cons "center_on_orbit" "long")
          (cons "verbosity" "long")
          (cons "n_points" "long")
          (cons "max_fit_order" "long")
          (cons "canonical_variables" "long")
          (cons "periodic" "long")
          (cons "beta_x" "double")
          (cons "alpha_x" "double")
          (cons "eta_x" "double")
          (cons "etap_x" "double")
          (cons "beta_y" "double")
          (cons "alpha_y" "double")
          (cons "eta_y" "double")
          (cons "etap_y" "double")
          ))
   (cons "aperture_data"
         (list
          (cons "input" "STRING")
          (cons "periodic" "long")
          (cons "persistent" "long")
          (cons "disable" "long")
          ))
   (cons "find_aperture"
         (list
          (cons "output" "STRING")
          (cons "search_output" "STRING")
          (cons "boundary" "STRING")
          (cons "mode" "STRING")
          (cons "xmin" "double")
          (cons "xmax" "double")
          (cons "xpmin" "double")
          (cons "xpmax" "double")
          (cons "ymin" "double")
          (cons "ymax" "double")
          (cons "ypmin" "double")
          (cons "ypmax" "double")
          (cons "nx" "long")
          (cons "n_splits" "long")
          (cons "split_fraction" "double")
          (cons "desired_resolution" "double")
          (cons "ny" "long")
          (cons "deltamin" "double")
          (cons "deltamax" "double")
          (cons "ndelta" "long")
          (cons "verbosity" "long")
          (cons "assume_nonincreasing" "long")
          (cons "offset_by_orbit" "long")
          (cons "n_lines" "long")
          (cons "full_plane" "long")
          (cons "optimization_mode" "long")
          ))
   (cons "bunched_beam"
         (list
          (cons "bunch" "STRING")
          (cons "n_particles_per_bunch" "long")
          (cons "multiply_np_by_cores" "long")
          (cons "time_start" "double")
          (cons "matched_to_cell" "STRING")
          (cons "emit_x" "double")
          (cons "emit_nx" "double")
          (cons "beta_x" "double")
          (cons "alpha_x" "double")
          (cons "eta_x" "double")
          (cons "etap_x" "double")
          (cons "emit_y" "double")
          (cons "emit_ny" "double")
          (cons "beta_y" "double")
          (cons "alpha_y" "double")
          (cons "eta_y" "double")
          (cons "etap_y" "double")
          (cons "spin" "double")
          (cons "use_twiss_command_values" "long")
          (cons "use_moments_output_values" "long")
          (cons "Po" "double")
          (cons "sigma_dp" "double")
          (cons "sigma_s" "double")
          (cons "dp_s_coupling" "double")
          (cons "emit_z" "double")
          (cons "beta_z" "double")
          (cons "alpha_z" "double")
          (cons "momentum_chirp" "double")
          (cons "one_random_bunch" "long")
          (cons "save_initial_coordinates" "long")
          (cons "limit_invariants" "long")
          (cons "symmetrize" "long")
          (cons "halton_sequence" "long")
          (cons "halton_radix" "int32_t")
          (cons "optimized_halton" "long")
          (cons "randomize_order" "long")
          (cons "limit_in_4d" "long")
          (cons "enforce_rms_values" "long")
          (cons "distribution_cutoff" "double")
          (cons "distribution_type" "STRING")
          (cons "centroid" "double")
          (cons "first_is_fiducial" "long")
          ))
   (cons "bunched_beam_moments"
         (list
          (cons "bunch" "STRING")
          (cons "n_particles_per_bunch" "long")
          (cons "multiply_np_by_cores" "long")
          (cons "use_moments_output_values" "long")
          (cons "S1_beta" "double")
          (cons "S2_beta" "double")
          (cons "S12_beta" "double")
          (cons "S16" "double")
          (cons "S26" "double")
          (cons "S3_beta" "double")
          (cons "S4_beta" "double")
          (cons "S34_beta" "double")
          (cons "S36" "double")
          (cons "S46" "double")
          (cons "S5" "double")
          (cons "S6" "double")
          (cons "S56" "double")
          (cons "spin" "double")
          (cons "time_start" "double")
          (cons "Po" "double")
          (cons "one_random_bunch" "long")
          (cons "save_initial_coordinates" "long")
          (cons "limit_invariants" "long")
          (cons "symmetrize" "long")
          (cons "halton_sequence" "long")
          (cons "halton_radix" "int32_t")
          (cons "optimized_halton" "long")
          (cons "randomize_order" "long")
          (cons "limit_in_4d" "long")
          (cons "enforce_rms_values" "long")
          (cons "distribution_cutoff" "double")
          (cons "distribution_type" "STRING")
          (cons "centroid" "double")
          (cons "first_is_fiducial" "long")
          ))
   (cons "chaos_map"
         (list
          (cons "output" "STRING")
          (cons "xmin" "double")
          (cons "xmax" "double")
          (cons "ymin" "double")
          (cons "ymax" "double")
          (cons "delta_min" "double")
          (cons "delta_max" "double")
          (cons "nx" "long")
          (cons "ny" "long")
          (cons "ndelta" "long")
          (cons "forward_backward" "long")
          (cons "change_x" "double")
          (cons "change_y" "double")
          (cons "verbosity" "long")
          ))
   (cons "chromaticity"
         (list
          (cons "sextupoles" "STRING")
          (cons "lower_limits" "STRING")
          (cons "upper_limits" "STRING")
          (cons "items" "STRING")
          (cons "exclude" "STRING")
          (cons "dnux_dp" "double")
          (cons "dnuy_dp" "double")
          (cons "sextupole_tweek" "double")
          (cons "correction_fraction" "double")
          (cons "min_correction_fraction" "double")
          (cons "n_iterations" "long")
          (cons "tolerance" "double")
          (cons "strength_log" "STRING")
          (cons "change_defined_values" "long")
          (cons "strength_limit" "double")
          (cons "use_perturbed_matrix" "long")
          (cons "exit_on_failure" "long")
          (cons "update_orbit" "long")
          (cons "reset_correctors_each_step" "long")
          (cons "verbosity" "long")
          (cons "dK2_weight" "double")
          (cons "response_matrix_output" "STRING")
          (cons "correction_matrix_output" "STRING")
          (cons "fse_units" "long")
          ))
   (cons "closed_orbit"
         (list
          (cons "output" "STRING")
          (cons "start_from_centroid" "long")
          (cons "start_from_dp_centroid" "long")
          (cons "closed_orbit_accuracy" "double")
          (cons "closed_orbit_accuracy_requirement" "double")
          (cons "closed_orbit_iterations" "long")
          (cons "fixed_length" "long")
          (cons "start_from_recirc" "long")
          (cons "verbosity" "long")
          (cons "iteration_fraction" "double")
          (cons "fraction_multiplier" "double")
          (cons "fraction_divisor" "double")
          (cons "multiplier_interval" "double")
          (cons "update_matrix" "long")
          (cons "output_monitors_only" "long")
          (cons "tracking_turns" "long")
          (cons "disable" "long")
          (cons "immediate" "long")
          ))
   (cons "correct"
         (list
          (cons "disable" "long")
          (cons "mode" "STRING")
          (cons "method" "STRING")
          (cons "trajectory_output" "STRING")
          (cons "corrector_output" "STRING")
          (cons "bpm_output" "STRING")
          (cons "statistics" "STRING")
          (cons "corrector_tweek" "double")
          (cons "corrector_limit" "double")
          (cons "correction_fraction" "double")
          (cons "correction_accuracy" "double")
          (cons "do_correction" "long")
          (cons "remove_smallest_SVs" "long")
          (cons "keep_largest_SVs" "long")
          (cons "minimum_SV_ratio" "double")
          (cons "auto_limit_SVs" "long")
          (cons "Tikhonov_relative_alpha" "double")
          (cons "Tikhonov_n" "long")
          (cons "remove_pegged" "long")
          (cons "threading_divisor" "long")
          (cons "threading_correctors" "long")
          (cons "bpm_noise" "double")
          (cons "bpm_noise_cutoff" "double")
          (cons "bpm_noise_distribution" "STRING")
          (cons "verbose" "long")
          (cons "fixed_length" "long")
          (cons "fixed_length_matrix" "long")
          (cons "n_xy_cycles" "long")
          (cons "minimum_cycles" "long")
          (cons "force_alternation" "long")
          (cons "n_iterations" "long")
          (cons "prezero_correctors" "long")
          (cons "reset_correctors_each_step" "long")
          (cons "track_before_and_after" "long")
          (cons "start_from_centroid" "long")
          (cons "use_actual_beam" "long")
          (cons "closed_orbit_accuracy" "double")
          (cons "closed_orbit_accuracy_requirement" "double")
          (cons "closed_orbit_iterations" "long")
          (cons "closed_orbit_iteration_fraction" "double")
          (cons "closed_orbit_fraction_multiplier" "double")
          (cons "closed_orbit_multiplier_interval" "double")
          (cons "closed_orbit_tracking_turns" "long")
          (cons "use_perturbed_matrix" "long")
          (cons "use_response_from_computed_orbits" "long")
          (cons "rpn_store_response_matrix" "long")
          ))
   (cons "correct_coupling"
         (list
          (cons "correction_elements" "STRING")
          (cons "items" "STRING")
          (cons "lower_limits" "STRING")
          (cons "upper_limits" "STRING")
          (cons "exclude" "STRING")
          (cons "bpm_name_pattern" "STRING")
          (cons "bpm_type_pattern" "STRING")
          (cons "n_iterations" "long")
          (cons "convergence" "double")
          (cons "change_tolerance" "double")
          (cons "correction_fraction" "double")
          (cons "use_perturbed_matrix" "long")
          (cons "adaptive_step" "long")
          (cons "response_perturbation" "double")
          (cons "svd_threshold" "double")
          (cons "n_singular_values" "long")
          (cons "auto_sv_threshold" "long")
          (cons "auto_sv_threshold_factor" "double")
          (cons "measurement_noise" "double")
          (cons "measurement_noise_cutoff" "double")
          (cons "etay_weight" "double")
          (cons "cross_h_steering" "STRING")
          (cons "cross_v_steering" "STRING")
          (cons "cross_x_bpm_name_pattern" "STRING")
          (cons "cross_x_bpm_type_pattern" "STRING")
          (cons "cross_y_bpm_name_pattern" "STRING")
          (cons "cross_y_bpm_type_pattern" "STRING")
          (cons "cross_response_weight" "double")
          (cons "cross_steering_kick" "double")
          (cons "cross_measurement_noise" "double")
          (cons "strength_log" "STRING")
          (cons "etay_file" "STRING")
          (cons "response_file" "STRING")
          (cons "rms_log" "STRING")
          (cons "reset_correctors_each_step" "long")
          (cons "verbosity" "long")
          ))
   (cons "compute_coupling_response_matrix"
         (list
          (cons "filename" "STRING")
          (cons "cross_filename" "STRING")
          (cons "correction_elements" "STRING")
          (cons "items" "STRING")
          (cons "exclude" "STRING")
          (cons "bpm_name_pattern" "STRING")
          (cons "bpm_type_pattern" "STRING")
          (cons "cross_h_steering" "STRING")
          (cons "cross_v_steering" "STRING")
          (cons "cross_x_bpm_name_pattern" "STRING")
          (cons "cross_x_bpm_type_pattern" "STRING")
          (cons "cross_y_bpm_name_pattern" "STRING")
          (cons "cross_y_bpm_type_pattern" "STRING")
          (cons "cross_steering_kick" "double")
          (cons "response_perturbation" "double")
          (cons "measurement_noise" "double")
          (cons "cross_measurement_noise" "double")
          (cons "measurement_noise_cutoff" "double")
          (cons "verbosity" "long")
          ))
   (cons "load_coupling_response_matrix"
         (list
          (cons "filename" "STRING")
          (cons "cross_filename" "STRING")
          (cons "verbosity" "long")
          ))
   (cons "correct_lattice"
         (list
          (cons "correction_elements" "STRING")
          (cons "items" "STRING")
          (cons "lower_limits" "STRING")
          (cons "upper_limits" "STRING")
          (cons "exclude" "STRING")
          (cons "bind_name_pattern" "STRING")
          (cons "measurement_elements" "STRING")
          (cons "measurement_types" "STRING")
          (cons "n_iterations" "long")
          (cons "convergence" "double")
          (cons "change_tolerance" "double")
          (cons "correction_fraction" "double")
          (cons "use_perturbed_matrix" "long")
          (cons "adaptive_step" "long")
          (cons "response_perturbation" "double")
          (cons "svd_threshold" "double")
          (cons "n_singular_values" "long")
          (cons "auto_sv_threshold" "long")
          (cons "auto_sv_threshold_factor" "double")
          (cons "beta_measurement_noise" "double")
          (cons "eta_measurement_noise" "double")
          (cons "measurement_noise_cutoff" "double")
          (cons "reference_file" "STRING")
          (cons "betax_weight" "double")
          (cons "betay_weight" "double")
          (cons "etax_weight" "double")
          (cons "strength_log" "STRING")
          (cons "response_file" "STRING")
          (cons "rms_log" "STRING")
          (cons "reset_correctors_each_step" "long")
          (cons "verbosity" "long")
          ))
   (cons "compute_lattice_response_matrix"
         (list
          (cons "filename" "STRING")
          (cons "correction_elements" "STRING")
          (cons "items" "STRING")
          (cons "exclude" "STRING")
          (cons "bind_name_pattern" "STRING")
          (cons "measurement_elements" "STRING")
          (cons "measurement_types" "STRING")
          (cons "response_perturbation" "double")
          (cons "measurement_noise" "double")
          (cons "measurement_noise_cutoff" "double")
          (cons "verbosity" "long")
          ))
   (cons "load_lattice_response_matrix"
         (list
          (cons "filename" "STRING")
          (cons "verbosity" "long")
          ))
   (cons "coupled_twiss_output"
         (list
          (cons "filename" "STRING")
          (cons "output_at_each_step" "long")
          (cons "emittances_from_twiss_command" "long")
          (cons "emittance_ratio" "double")
          (cons "emit_x" "double")
          (cons "sigma_dp" "double")
          (cons "calculate_3d_coupling" "long")
          (cons "verbosity" "long")
          (cons "concat_order" "long")
          (cons "output_sigma_matrix" "long")
          (cons "matched" "long")
          (cons "beta_x1" "double")
          (cons "beta_x2" "double")
          (cons "beta_y1" "double")
          (cons "beta_y2" "double")
          (cons "alpha_x1" "double")
          (cons "alpha_x2" "double")
          (cons "alpha_y1" "double")
          (cons "alpha_y2" "double")
          (cons "eta_x" "double")
          (cons "etap_x" "double")
          (cons "eta_y" "double")
          (cons "etap_y" "double")
          (cons "gamma_x1" "double")
          (cons "gamma_x2" "double")
          (cons "gamma_y1" "double")
          (cons "gamma_y2" "double")
          (cons "A_xy_1" "double")
          (cons "A_xpy_1" "double")
          (cons "A_xyp_1" "double")
          (cons "A_xpyp_1" "double")
          (cons "A_xy_2" "double")
          (cons "A_xpy_2" "double")
          (cons "A_xyp_2" "double")
          (cons "A_xpyp_2" "double")
          (cons "reference_file" "STRING")
          (cons "reference_element" "STRING")
          (cons "reference_element_occurrence" "long")
          (cons "reflect_reference_values" "long")
          ))
   (cons "divide_elements"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "exclude_name_pattern" "STRING")
          (cons "exclude_type_pattern" "STRING")
          (cons "divisions" "long")
          (cons "maximum_length" "double")
          (cons "clear" "long")
          ))
   (cons "elastic_scattering"
         (list
          (cons "losses" "STRING")
          (cons "output" "STRING")
          (cons "log_file" "STRING")
          (cons "theta_min" "double")
          (cons "theta_max" "double")
          (cons "n_theta" "long")
          (cons "quadratic_theta_spacing" "long")
          (cons "n_phi" "long")
          (cons "twiss_scaling" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "include_name_pattern" "STRING")
          (cons "include_type_pattern" "STRING")
          (cons "verbosity" "long")
          (cons "soft_failure" "long")
          (cons "allow_watch_file_output" "long")
          ))
   (cons "change_start"
         (list
          (cons "element_name" "STRING")
          (cons "ring_mode" "long")
          (cons "element_occurence" "long")
          (cons "delta_position" "long")
          ))
   (cons "change_end"
         (list
          (cons "element_name" "STRING")
          (cons "element_occurence" "long")
          (cons "delta_position" "long")
          ))
   (cons "run_setup"
         (list
          (cons "lattice" "STRING")
          (cons "use_beamline" "STRING")
          (cons "rootname" "STRING")
          (cons "output" "STRING")
          (cons "centroid" "STRING")
          (cons "bpm_centroid" "STRING")
          (cons "sigma" "STRING")
          (cons "final" "STRING")
          (cons "acceptance" "STRING")
          (cons "losses" "STRING")
          (cons "losses_include_global_coordinates" "long")
          (cons "losses_s_limit" "double")
          (cons "magnets" "STRING")
          (cons "profile" "STRING")
          (cons "semaphore_file" "STRING")
          (cons "parameters" "STRING")
          (cons "suppress_parameter_defaults" "long")
          (cons "rfc_reference_output" "STRING")
          (cons "combine_bunch_statistics" "long")
          (cons "wrap_around" "long")
          (cons "final_pass" "long")
          (cons "default_order" "long")
          (cons "concat_order" "long")
          (cons "print_statistics" "long")
          (cons "show_element_timing" "long")
          (cons "monitor_memory_usage" "long")
          (cons "random_number_seed" "long")
          (cons "correction_iterations" "long")
          (cons "echo_lattice" "long")
          (cons "p_central" "double")
          (cons "p_central_mev" "double")
          (cons "always_change_p0" "long")
          (cons "load_balancing_on" "long")
          (cons "random_sequence_No" "long")
          (cons "expand_for" "STRING")
          (cons "tracking_updates" "long")
          (cons "search_path" "STRING")
          (cons "element_divisions" "long")
          (cons "back_tracking" "long")
          (cons "s_start" "double")
          (cons "spin_tracking" "long")
          ))
   (cons "change_particle"
         (list
          (cons "name" "STRING")
          (cons "mass_ratio" "double")
          (cons "charge_ratio" "double")
          ))
   (cons "track"
         (list
          (cons "center_on_orbit" "long")
          (cons "center_momentum_also" "long")
          (cons "offset_by_orbit" "long")
          (cons "offset_momentum_also" "long")
          (cons "soft_failure" "long")
          (cons "use_linear_chromatic_matrix" "long")
          (cons "longitudinal_ring_only" "long")
          (cons "ibs_only" "long")
          (cons "stop_tracking_particle_limit" "long")
          (cons "check_beam_structure" "long")
          (cons "interrupt_file" "STRING")
          ))
   (cons "print_dictionary"
         (list
          (cons "filename" "STRING")
          (cons "latex_form" "long")
          (cons "SDDS_form" "long")
          ))
   (cons "semaphores"
         (list
          (cons "started" "STRING")
          (cons "done" "STRING")
          (cons "failed" "STRING")
          (cons "immediate" "STRING")
          ))
   (cons "include_commands"
         (list
          (cons "filename" "STRING")
          (cons "disable" "long")
          ))
   (cons "macro_output"
         (list
          (cons "filename" "STRING")
          (cons "mode" "STRING")
          ))
   (cons "error_control"
         (list
          (cons "clear_error_settings" "long")
          (cons "summarize_error_settings" "long")
          (cons "error_log" "STRING")
          (cons "no_errors_for_first_step" "long")
          (cons "error_factor" "double")
          ))
   (cons "error_element"
         (list
          (cons "name" "STRING")
          (cons "exclude" "STRING")
          (cons "item" "STRING")
          (cons "element_type" "STRING")
          (cons "type" "STRING")
          (cons "amplitude" "double")
          (cons "cutoff" "double")
          (cons "bind" "long")
          (cons "bind_number" "long")
          (cons "bind_across_names" "long")
          (cons "fractional" "long")
          (cons "post_correction" "long")
          (cons "additive" "long")
          (cons "allow_missing_elements" "long")
          (cons "before" "STRING")
          (cons "after" "STRING")
          (cons "sample_file" "STRING")
          (cons "sample_file_column" "STRING")
          (cons "sample_mode" "STRING")
          ))
   (cons "fit_traces"
         (list
          (cons "trace_data_file" "STRING")
          (cons "fit_parameters_file" "STRING")
          (cons "iterations" "long")
          (cons "sub_iterations" "long")
          (cons "use_SVD" "long")
          (cons "SVs_to_keep" "long")
          (cons "SVs_to_remove" "long")
          (cons "BPM_threshold" "double")
          (cons "convergence_factor" "double")
          (cons "convergence_factor_divisor" "double")
          (cons "convergence_factor_multiplier" "double")
          (cons "convergence_increase_steps" "long")
          (cons "convergence_factor_min" "double")
          (cons "convergence_factor_max" "double")
          (cons "position_change_limit" "double")
          (cons "slope_change_limit" "double")
          (cons "trace_sub_iterations" "long")
          (cons "trace_convergence_factor" "double")
          (cons "trace_fractional_target" "double")
          (cons "fit_output_file" "STRING")
          (cons "trace_output_file" "STRING")
          (cons "target" "double")
          (cons "tolerance" "double")
          (cons "reject_BPM_common_mode" "long")
          (cons "n_restarts" "long")
          (cons "restart_randomization_level" "double")
          ))
   (cons "floor_coordinates"
         (list
          (cons "filename" "STRING")
          (cons "X0" "double")
          (cons "Y0" "double")
          (cons "Z0" "double")
          (cons "theta0" "double")
          (cons "phi0" "double")
          (cons "psi0" "double")
          (cons "include_vertices" "long")
          (cons "include_arc_centers" "long")
          (cons "vertices_only" "long")
          (cons "magnet_centers" "long")
          (cons "store_vertices" "long")
          (cons "store_centers" "long")
          ))
   (cons "frequency_map"
         (list
          (cons "output" "STRING")
          (cons "xmin" "double")
          (cons "xmax" "double")
          (cons "ymin" "double")
          (cons "ymax" "double")
          (cons "delta_min" "double")
          (cons "delta_max" "double")
          (cons "nx" "long")
          (cons "ny" "long")
          (cons "ndelta" "long")
          (cons "verbosity" "long")
          (cons "include_changes" "long")
          (cons "quadratic_spacing" "long")
          (cons "full_grid_output" "long")
          ))
   (cons "global_settings"
         (list
          (cons "inhibit_fsync" "long")
          (cons "allow_overwrite" "long")
          (cons "echo_namelists" "long")
          (cons "mpi_randomization_mode" "long")
          (cons "exact_normalized_emittance" "long")
          (cons "SR_gaussian_limit" "double")
          (cons "inhibit_seed_permutation" "long")
          (cons "log_file" "STRING")
          (cons "error_log_file" "STRING")
          (cons "share_tracking_based_matrices" "long")
          (cons "tracking_based_matrices_store_limit" "long")
          (cons "parallel_tracking_based_matrices" "long")
          (cons "mpi_io_force_file_sync" "long")
          (cons "mpi_io_read_buffer_size" "long")
          (cons "mpi_io_write_buffer_size" "long")
          (cons "usleep_mpi_io_kludge" "long")
          (cons "tracking_matrix_step_factor" "double")
          (cons "tracking_matrix_points" "double")
          (cons "tracking_matrix_max_fit_order" "double")
          (cons "tracking_matrix_step_size" "double")
          (cons "tracking_matrix_cleanup" "short")
          (cons "warning_limit" "long")
          (cons "malign_method" "short")
          (cons "slope_limit" "double")
          (cons "coord_limit" "double")
          (cons "search_path" "STRING")
          (cons "namelist_buffer_size_factor" "long")
          ))
   (cons "ignore_elements"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "completely" "long")
          (cons "disable" "long")
          (cons "clear_all" "long")
          ))
   (cons "inelastic_scattering"
         (list
          (cons "losses" "STRING")
          (cons "output" "STRING")
          (cons "log_file" "STRING")
          (cons "k_min" "double")
          (cons "momentum_aperture" "STRING")
          (cons "momentum_aperture_scale" "double")
          (cons "momentum_aperture_periodicity" "double")
          (cons "n_k" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "include_name_pattern" "STRING")
          (cons "include_type_pattern" "STRING")
          (cons "verbosity" "long")
          (cons "soft_failure" "long")
          (cons "allow_watch_file_output" "long")
          ))
   (cons "insert_sceffects"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "disable" "long")
          (cons "clear" "long")
          (cons "element_prefix" "STRING")
          (cons "skip" "long")
          (cons "vertical" "long")
          (cons "horizontal" "long")
          (cons "longitudinal" "long")
          (cons "nonlinear" "long")
          (cons "uniform_distribution" "long")
          (cons "slice_duration" "double")
          (cons "slice_threshold" "long")
          (cons "slice_interpolation" "long")
          (cons "verbosity" "long")
          (cons "averaging_factor" "double")
          ))
   (cons "insert_elements"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "skip" "long")
          (cons "disable" "long")
          (cons "insert_before" "long")
          (cons "add_at_start" "long")
          (cons "add_at_end" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "occurrence_start" "long")
          (cons "occurrence_end" "long")
          (cons "start_at_element" "STRING")
          (cons "end_at_element" "STRING")
          (cons "element_def" "STRING")
          (cons "verbose" "long")
          (cons "total_occurrences" "long")
          (cons "occurrence" "long")
          (cons "allow_no_matches" "long")
          (cons "allow_no_insertions" "long")
          (cons "insertion_count_variable" "STRING")
          ))
   (cons "ion_effects"
         (list
          (cons "pressure_profile" "STRING")
          (cons "pressure_factor" "double")
          (cons "ion_properties" "STRING")
          (cons "beam_output" "STRING")
          (cons "beam_output_all_locations" "long")
          (cons "ion_density_output" "STRING")
          (cons "ion_output_all_locations" "long")
          (cons "ion_species_output" "long")
          (cons "ion_output_interval" "long")
          (cons "field_calculation_method" "STRING")
          (cons "conserve_momentum" "long")
          (cons "distribution_fit_target" "double")
          (cons "distribution_fit_tolerance" "double")
          (cons "distribution_fit_evaluations" "long")
          (cons "distribution_fit_passes" "long")
          (cons "distribution_fit_restarts" "long")
          (cons "hybrid_simplex_comparison_interval" "long")
          (cons "fit_residual_type" "STRING")
          (cons "macro_ions" "long")
          (cons "symmetrize" "long")
          (cons "generation_interval" "long")
          (cons "multiple_ionization_interval" "long")
          (cons "multiple_ionization_energy_peak" "double")
          (cons "multiple_ionization_energy_rms" "double")
          (cons "ion_span" "double")
          (cons "ion_poisson_bins" "long")
          (cons "ion_poisson_span" "double")
          (cons "ion_bin_divisor" "double")
          (cons "ion_range_multiplier" "double")
          (cons "ion_sigma_limit_multiplier" "double")
          (cons "ion_histogram_max_bins" "long")
          (cons "ion_histogram_min_per_bin" "long")
          (cons "ion_histogram_output" "STRING")
          (cons "ion_2d_histogram_output" "STRING")
          (cons "ion_histogram_output_s_start" "double")
          (cons "ion_histogram_output_s_end" "double")
          (cons "ion_histogram_output_interval" "long")
          (cons "ion_histogram_min_output_bins" "long")
          (cons "disable_until_pass" "long")
          (cons "freeze_ions_until_pass" "long")
          (cons "freeze_electrons_until_pass" "long")
          (cons "ion_fit_subtract_baseline" "long")
          (cons "gaussian_ion_range" "double")
          (cons "verbosity" "long")
          (cons "use_local_pressure" "long")
          ))
   (cons "link_control"
         (list
          (cons "clear_links" "long")
          (cons "summarize_links" "long")
          (cons "verbosity" "long")
          ))
   (cons "link_elements"
         (list
          (cons "target" "STRING")
          (cons "target_occurence" "long")
          (cons "exclude" "STRING")
          (cons "item" "STRING")
          (cons "source" "STRING")
          (cons "source_from_target_edit" "STRING")
          (cons "source_position" "STRING")
          (cons "mode" "STRING")
          (cons "equation" "STRING")
          (cons "minimum" "double")
          (cons "maximum" "double")
          (cons "exclude_self" "long")
          ))
   (cons "load_parameters"
         (list
          (cons "filename" "STRING")
          (cons "filename_list" "STRING")
          (cons "include_name_pattern" "STRING")
          (cons "include_item_pattern" "STRING")
          (cons "include_type_pattern" "STRING")
          (cons "exclude_name_pattern" "STRING")
          (cons "exclude_item_pattern" "STRING")
          (cons "exclude_type_pattern" "STRING")
          (cons "edit_name_command" "STRING")
          (cons "change_defined_values" "long")
          (cons "script_triggered" "long")
          (cons "repeat_first_page_at_each_step" "long")
          (cons "clear_settings" "long")
          (cons "allow_missing_files" "long")
          (cons "allow_missing_elements" "long")
          (cons "allow_missing_parameters" "long")
          (cons "force_occurence_data" "long")
          (cons "verbose" "long")
          (cons "skip_pages" "long")
          (cons "use_first" "long")
          (cons "prefactor" "double")
          ))
   (cons "matrix_output"
         (list
          (cons "printout" "STRING")
          (cons "printout_order" "long")
          (cons "printout_format" "STRING")
          (cons "full_matrix_only" "long")
          (cons "print_element_data" "long")
          (cons "mathematica_full_matrix" "long")
          (cons "mathematica_matrix_name" "STRING")
          (cons "mathematica_matrix_file" "STRING")
          (cons "SDDS_output" "STRING")
          (cons "SDDS_output_order" "long")
          (cons "individual_matrices" "long")
          (cons "output_at_each_step" "long")
          (cons "start_from" "STRING")
          (cons "start_from_occurence" "long")
          (cons "SDDS_output_match" "STRING")
          (cons "suppress_below" "double")
          ))
   (cons "modulate_elements"
         (list
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "type" "STRING")
          (cons "expression" "STRING")
          (cons "filename" "STRING")
          (cons "time_column" "STRING")
          (cons "convert_pass_to_time" "long")
          (cons "amplitude_column" "STRING")
          (cons "refresh_matrix" "long")
          (cons "differential" "long")
          (cons "multiplicative" "long")
          (cons "factor" "double")
          (cons "start_pass" "long")
          (cons "end_pass" "long")
          (cons "pass_delay" "long")
          (cons "time_delay" "double")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "before" "STRING")
          (cons "after" "STRING")
          (cons "verbose" "long")
          (cons "verbose_threshold" "double")
          (cons "record" "STRING")
          (cons "flush_record" "long")
          ))
   (cons "moments_output"
         (list
          (cons "filename" "STRING")
          (cons "matrix_output" "STRING")
          (cons "output_at_each_step" "long")
          (cons "output_before_tune_correction" "long")
          (cons "final_values_only" "long")
          (cons "verbosity" "long")
          (cons "matched" "long")
          (cons "equilibrium" "long")
          (cons "force_e1_gt_e2" "long")
          (cons "radiation" "long")
          (cons "ibs_iterations" "long")
          (cons "ibs_output_iterations" "long")
          (cons "ibs_iteration_fraction" "double")
          (cons "ibs_coulomb_log" "double")
          (cons "n_slices" "long")
          (cons "slice_etilted" "long")
          (cons "tracking_based_diffusion_matrix_particles" "long")
          (cons "emit_x" "double")
          (cons "beta_x" "double")
          (cons "alpha_x" "double")
          (cons "eta_x" "double")
          (cons "etap_x" "double")
          (cons "emit_y" "double")
          (cons "beta_y" "double")
          (cons "alpha_y" "double")
          (cons "eta_y" "double")
          (cons "etap_y" "double")
          (cons "emit_z" "double")
          (cons "beta_z" "double")
          (cons "alpha_z" "double")
          (cons "reference_file" "STRING")
          (cons "reference_element" "STRING")
          (cons "reference_element_occurrence" "long")
          (cons "reflect_reference_values" "long")
          ))
   (cons "momentum_aperture"
         (list
          (cons "output" "STRING")
          (cons "x_initial" "double")
          (cons "y_initial" "double")
          (cons "delta_negative_limit" "double")
          (cons "delta_positive_limit" "double")
          (cons "delta_negative_start" "double")
          (cons "delta_positive_start" "double")
          (cons "delta_step_size" "double")
          (cons "splits" "long")
          (cons "steps_back" "long")
          (cons "split_step_divisor" "long")
          (cons "skip_elements" "long")
          (cons "process_elements" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "include_name_pattern" "STRING")
          (cons "include_type_pattern" "STRING")
          (cons "fiducialize" "long")
          (cons "verbosity" "long")
          (cons "soft_failure" "long")
          (cons "output_mode" "long")
          (cons "forbid_resonance_crossing" "long")
          (cons "allow_watch_file_output" "long")
          ))
   (cons "obstruction_data"
         (list
          (cons "input" "STRING")
          (cons "periods" "long")
          (cons "disable" "long")
          (cons "y_spacing" "double")
          (cons "y_limit" "double")
          ))
   (cons "optimization_covariable"
         (list
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "equation" "STRING")
          (cons "disable" "long")
          ))
   (cons "optimization_term"
         (list
          (cons "term" "STRING")
          (cons "weight" "double")
          (cons "field_string" "STRING")
          (cons "field_initial_value" "long")
          (cons "field_final_value" "long")
          (cons "field_interval" "long")
          (cons "input_file" "STRING")
          (cons "input_column" "STRING")
          (cons "verbose" "long")
          ))
   (cons "optimization_setup"
         (list
          (cons "equation" "STRING")
          (cons "mode" "STRING")
          (cons "method" "STRING")
          (cons "statistic" "STRING")
          (cons "tolerance" "double")
          (cons "hybrid_simplex_tolerance" "double")
          (cons "hybrid_simplex_tolerance_count" "double")
          (cons "hybrid_simplex_comparison_interval" "long")
          (cons "target" "double")
          (cons "center_on_orbit" "long")
          (cons "center_momentum_also" "long")
          (cons "soft_failure" "long")
          (cons "n_passes" "long")
          (cons "n_evaluations" "long")
          (cons "n_restarts" "long")
          (cons "restart_reset_threshold" "double")
          (cons "restart_worst_term_factor" "double")
          (cons "restart_worst_terms" "long")
          (cons "matrix_order" "long")
          (cons "log_file" "STRING")
          (cons "term_log_file" "STRING")
          (cons "verbose" "long")
          (cons "output_sparsing_factor" "long")
          (cons "crossover" "STRING")
          (cons "balance_terms" "long")
          (cons "simplex_divisor" "double")
          (cons "simplex_pass_range_factor" "double")
          (cons "include_simplex_1d_scans" "long")
          (cons "start_from_simplex_vertex1" "long")
          (cons "restart_random_numbers" "long")
          (cons "random_factor" "double")
          (cons "rcds_step_factor" "double")
          (cons "n_iterations" "long")
          (cons "max_no_change" "long")
          (cons "population_size" "long")
          (cons "print_all_individuals" "long")
          (cons "population_log" "STRING")
          (cons "interrupt_file" "STRING")
          (cons "interrupt_file_check_interval" "double")
          (cons "inhibit_tracking" "long")
          (cons "simplex_log" "STRING")
          (cons "simplex_log_interval" "long")
          ))
   (cons "optimization_variable"
         (list
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "lower_limit" "double")
          (cons "upper_limit" "double")
          (cons "step_size" "double")
          (cons "fractional_step_size" "double")
          (cons "disable" "long")
          (cons "force_inside" "long")
          (cons "differential_limits" "long")
          (cons "no_element" "long")
          (cons "initial_value" "double")
          ))
   (cons "optimization_constraint"
         (list
          (cons "quantity" "STRING")
          (cons "lower" "double")
          (cons "upper" "double")
          ))
   (cons "optimize"
         (list
          (cons "summarize_setup" "long")
          ))
   (cons "set_reference_particle_output"
         (list
          (cons "match_to" "STRING")
          (cons "weight" "double")
          (cons "comparison_mode" "STRING")
          ))
   (cons "particle_tunes"
         (list
          (cons "filename" "STRING")
          (cons "start_pid" "long")
          (cons "end_pid" "long")
          (cons "pid_interval" "long")
          (cons "include_x" "short")
          (cons "include_y" "short")
          (cons "include_s" "short")
          (cons "include_spin" "short")
          (cons "start_pass" "long")
          (cons "segment_length" "long")
          ))
   (cons "ramp_elements"
         (list
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "type" "STRING")
          (cons "start_pass" "long")
          (cons "end_pass" "long")
          (cons "start_value" "double")
          (cons "end_value" "double")
          (cons "refresh_matrix" "long")
          (cons "differential" "long")
          (cons "multiplicative" "long")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "exponent" "double")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "before" "STRING")
          (cons "after" "STRING")
          (cons "verbose" "long")
          (cons "record" "STRING")
          ))
   (cons "replace_elements"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "skip" "long")
          (cons "disable" "long")
          (cons "element_def" "STRING")
          (cons "total_occurrences" "long")
          (cons "occurrence" "long")
          (cons "verbose" "long")
          ))
   (cons "correction_matrix_output"
         (list
          (cons "response" "STRING")
          (cons "inverse" "STRING")
          (cons "slope_response" "STRING")
          (cons "full_names" "long")
          (cons "KnL_units" "long")
          (cons "BnL_units" "long")
          (cons "output_at_each_step" "long")
          (cons "output_before_tune_correction" "long")
          (cons "fixed_length" "long")
          (cons "coupled" "long")
          (cons "use_response_from_computed_orbits" "long")
          ))
   (cons "rpn_expression"
         (list
          (cons "expression" "STRING")
          ))
   (cons "rpn_load"
         (list
          (cons "filename" "STRING")
          (cons "match_column" "STRING")
          (cons "match_column_value" "STRING")
          (cons "matching_row_number" "long")
          (cons "match_parameter" "STRING")
          (cons "match_parameter_value" "STRING")
          (cons "use_row" "long")
          (cons "use_page" "long")
          (cons "load_parameters" "long")
          (cons "tag" "STRING")
          ))
   (cons "sasefel"
         (list
          (cons "output" "STRING")
          (cons "model" "STRING")
          (cons "beamsize_mode" "STRING")
          (cons "beta" "double")
          (cons "undulator_K" "double")
          (cons "undulator_period" "double")
          (cons "slice_fraction" "double")
          (cons "n_slices" "long")
          ))
   (cons "save_lattice"
         (list
          (cons "filename" "STRING")
          (cons "suppress_defaults" "long")
          (cons "output_seq" "long")
          ))
   (cons "sdds_beam"
         (list
          (cons "input" "STRING")
          (cons "input_list" "STRING")
          (cons "input_type" "STRING")
          (cons "selection_parameter" "STRING")
          (cons "selection_string" "STRING")
          (cons "one_random_bunch" "long")
          (cons "n_particles_per_ring" "long")
          (cons "reuse_bunch" "long")
          (cons "prebunched" "long")
          (cons "track_pages_separately" "long")
          (cons "use_bunched_mode" "long")
          (cons "fiducialization_bunch" "long")
          (cons "sample_interval" "long")
          (cons "n_tables_to_skip" "long")
          (cons "center_transversely" "long")
          (cons "center_arrival_time" "long")
          (cons "reverse_t_sign" "long")
          (cons "sample_fraction" "double")
          (cons "p_lower" "double")
          (cons "p_upper" "double")
          (cons "save_initial_coordinates" "long")
          (cons "n_duplicates" "long")
          (cons "duplicate_stagger" "double")
          ))
   (cons "slice_analysis"
         (list
          (cons "output" "STRING")
          (cons "n_slices" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "final_values_only" "long")
          ))
   (cons "steering_element"
         (list
          (cons "name" "STRING")
          (cons "element_type" "STRING")
          (cons "item" "STRING")
          (cons "plane" "STRING")
          (cons "tweek" "double")
          (cons "limit" "double")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "occurence_step" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "after" "STRING")
          (cons "before" "STRING")
          (cons "verbose" "long")
          ))
   (cons "subprocess"
         (list
          (cons "command" "STRING")
          ))
   (cons "touschek_scatter"
         (list
          (cons "charge" "double")
          (cons "frequency" "double")
          (cons "emit_x" "double")
          (cons "emit_nx" "double")
          (cons "emit_y" "double")
          (cons "emit_ny" "double")
          (cons "sigma_dp" "double")
          (cons "sigma_s" "double")
          (cons "distribution_cutoff" "double")
          (cons "Momentum_Aperture_scale" "double")
          (cons "Momentum_Aperture" "STRING")
          (cons "XDist" "STRING")
          (cons "YDist" "STRING")
          (cons "ZDist" "STRING")
          (cons "TranDist" "STRING")
          (cons "FullDist" "STRING")
          (cons "bunch" "STRING")
          (cons "loss" "STRING")
          (cons "distribution" "STRING")
          (cons "initial" "STRING")
          (cons "output" "STRING")
          (cons "nbins" "long")
          (cons "sbin_step" "double")
          (cons "n_simulated" "long")
          (cons "ignored_portion" "double")
          (cons "i_start" "long")
          (cons "i_end" "long")
          (cons "match_position_only" "long")
          (cons "do_track" "long")
          (cons "verbosity" "long")
          (cons "overwrite_files" "long")
          ))
   (cons "trace"
         (list
          (cons "traceback_on" "long")
          (cons "trace_on" "long")
          (cons "heap_verify_depth" "long")
          (cons "immediate" "long")
          (cons "filename" "STRING")
          (cons "memory_log" "STRING")
          (cons "record_allocation" "long")
          ))
   (cons "transmute_elements"
         (list
          (cons "name" "STRING")
          (cons "type" "STRING")
          (cons "exclude" "STRING")
          (cons "new_type" "STRING")
          (cons "disable" "long")
          (cons "clear_all" "long")
          ))
   (cons "correct_tunes"
         (list
          (cons "quadrupoles" "STRING")
          (cons "lower_limits" "STRING")
          (cons "upper_limits" "STRING")
          (cons "items" "STRING")
          (cons "exclude" "STRING")
          (cons "tune_x" "double")
          (cons "tune_y" "double")
          (cons "n_iterations" "long")
          (cons "correction_fraction" "double")
          (cons "tolerance" "double")
          (cons "step_up_interval" "long")
          (cons "max_correction_fraction" "double")
          (cons "delta_correction_fraction" "double")
          (cons "strength_log" "STRING")
          (cons "change_defined_values" "long")
          (cons "use_perturbed_matrix" "long")
          (cons "dK1_weight" "double")
          (cons "update_orbit" "long")
          (cons "reset_correctors_each_step" "long")
          (cons "verbosity" "long")
          (cons "response_matrix_output" "STRING")
          (cons "correction_matrix_output" "STRING")
          (cons "fse_units" "long")
          ))
   (cons "tune_footprint"
         (list
          (cons "delta_output" "STRING")
          (cons "xy_output" "STRING")
          (cons "xmin" "double")
          (cons "xmax" "double")
          (cons "ymin" "double")
          (cons "ymax" "double")
          (cons "x_for_delta" "double")
          (cons "y_for_delta" "double")
          (cons "separate_xy_for_delta" "double")
          (cons "delta_min" "double")
          (cons "delta_max" "double")
          (cons "nx" "long")
          (cons "ny" "long")
          (cons "ndelta" "long")
          (cons "verbosity" "long")
          (cons "quadratic_spacing" "long")
          (cons "compute_diffusion" "long")
          (cons "diffusion_rate_limit" "long")
          (cons "immediate" "long")
          (cons "filtered_output" "long")
          (cons "ignore_half_integer" "long")
          (cons "chromaticity_fit_order" "long")
          ))
   (cons "setup_linear_chromatic_tracking"
         (list
          (cons "nux" "double")
          (cons "betax" "double")
          (cons "alphax" "double")
          (cons "etax" "double")
          (cons "etapx" "double")
          (cons "nuy" "double")
          (cons "betay" "double")
          (cons "alphay" "double")
          (cons "etay" "double")
          (cons "etapy" "double")
          (cons "alphac" "double")
          ))
   (cons "tune_shift_with_amplitude"
         (list
          (cons "turns" "long")
          (cons "x0" "double")
          (cons "y0" "double")
          (cons "x1" "double")
          (cons "y1" "double")
          (cons "lines_only" "long")
          (cons "grid_size" "long")
          (cons "sparse_grid" "long")
          (cons "spread_only" "long")
          (cons "exclude_lost_particles" "long")
          (cons "nux_roi_width" "double")
          (cons "nuy_roi_width" "double")
          (cons "scale_down_factor" "double")
          (cons "scale_up_factor" "double")
          (cons "scale_down_limit" "double")
          (cons "scale_up_limit" "double")
          (cons "scaling_iterations" "long")
          (cons "use_concatenation" "long")
          (cons "verbose" "long")
          (cons "order" "long")
          (cons "tune_output" "STRING")
          ))
   (cons "twiss_output"
         (list
          (cons "filename" "STRING")
          (cons "matched" "long")
          (cons "output_at_each_step" "long")
          (cons "output_before_tune_correction" "long")
          (cons "final_values_only" "long")
          (cons "statistics" "long")
          (cons "radiation_integrals" "long")
          (cons "beta_x" "double")
          (cons "alpha_x" "double")
          (cons "eta_x" "double")
          (cons "etap_x" "double")
          (cons "beta_y" "double")
          (cons "alpha_y" "double")
          (cons "eta_y" "double")
          (cons "etap_y" "double")
          (cons "reference_file" "STRING")
          (cons "reference_element" "STRING")
          (cons "reference_element_occurrence" "long")
          (cons "reflect_reference_values" "long")
          (cons "concat_order" "long")
          (cons "higher_order_chromaticity" "long")
          (cons "higher_order_chromaticity_points" "long")
          (cons "higher_order_chromaticity_range" "double")
          (cons "quick_higher_order_chromaticity" "long")
          (cons "chromatic_tune_spread_half_range" "double")
          (cons "cavities_are_drifts_if_matched" "long")
          (cons "compute_driving_terms" "long")
          (cons "leading_order_driving_terms_only" "long")
          (cons "s_dependent_driving_terms_file" "STRING")
          (cons "local_dispersion" "long")
          (cons "n_periods" "long")
          ))
   (cons "twiss_analysis"
         (list
          (cons "match_name" "STRING")
          (cons "start_name" "STRING")
          (cons "end_name" "STRING")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "tag" "STRING")
          (cons "verbosity" "long")
          (cons "clear" "long")
          ))
   (cons "rf_setup"
         (list
          (cons "filename" "STRING")
          (cons "name" "STRING")
          (cons "start_occurence" "long")
          (cons "end_occurence" "long")
          (cons "s_start" "double")
          (cons "s_end" "double")
          (cons "set_for_each_step" "long")
          (cons "near_frequency" "double")
          (cons "fractional_frequency_change" "double")
          (cons "harmonic" "long")
          (cons "bucket_half_height" "double")
          (cons "over_voltage" "double")
          (cons "total_voltage" "double")
          (cons "disable" "long")
          (cons "output_only" "long")
          (cons "track_for_frequency" "long")
          (cons "phase_offset" "double")
          ))
   (cons "undulator_brightness"
         (list
          (cons "tag" "STRING")
          (cons "wavelength" "double")
          (cons "photon_energy" "double")
          (cons "harmonic" "long")
          (cons "detuning" "double")
          (cons "period_length" "double")
          (cons "n_periods" "long")
          (cons "total_length" "double")
          (cons "current" "double")
          (cons "use_twiss_output_values" "long")
          (cons "coupling" "double")
          (cons "twiss_element" "STRING")
          (cons "twiss_occurence" "long")
          (cons "emitx" "double")
          (cons "emity" "double")
          (cons "betax" "double")
          (cons "alphax" "double")
          (cons "betay" "double")
          (cons "alphay" "double")
          (cons "etax" "double")
          (cons "etaxp" "double")
          (cons "etay" "double")
          (cons "etayp" "double")
          (cons "Sdelta" "double")
          ))
   (cons "run_control"
         (list
          (cons "n_steps" "long")
          (cons "bunch_frequency" "double")
          (cons "n_indices" "long")
          (cons "n_passes" "long")
          (cons "n_passes_fiducial" "long")
          (cons "terminate_on_failure" "long")
          (cons "reset_rf_for_each_step" "long")
          (cons "first_is_fiducial" "long")
          (cons "restrict_fiducialization" "long")
          (cons "reset_scattering_seed" "long")
          (cons "wait_for_step_semaphore" "STRING")
          (cons "step_done_semaphore" "STRING")
          (cons "semaphore_check_interval" "double")
          (cons "restart_files" "long")
          ))
   (cons "vary_element"
         (list
          (cons "index_number" "long")
          (cons "index_limit" "long")
          (cons "name" "STRING")
          (cons "item" "STRING")
          (cons "initial" "double")
          (cons "final" "double")
          (cons "differential" "long")
          (cons "geometric" "long")
          (cons "multiplicative" "long")
          (cons "enumeration_file" "STRING")
          (cons "enumeration_column" "STRING")
          (cons "disable" "long")
          ))
   )
  "Alist mapping namelist command -> alist of (qualifier . base-type).")

(provide 'elegant-namelists)
;;; elegant-namelists.el ends here
