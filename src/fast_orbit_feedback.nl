/* file: fast_orbit_feedback.nl
 * purpose: namelist definition for fast orbit feedback (FOFB) simulation
 *
 * The command models fast orbit feedback combined with realistic beam
 * dynamics.  Each FOFB iteration tracks the beam for run_control->n_passes
 * turns (accumulating each BPM's turn-by-turn position through a digital IIR
 * filter to form a "tick" readout), computes steering-corrector setpoint
 * changes from a response (inverse) matrix through a per-actuator-class PID
 * controller, and lets the physically applied kick evolve turn-by-turn toward
 * the new setpoint through a per-actuator z-transform step response.  The
 * number of FOFB iterations is run_control->n_steps.
 *
 * See fast_orbit_feedback.c for the implementation.
 */
#include "namelist.h"

#namelist fast_orbit_feedback static
    STRING inverse[2] = {NULL, NULL};
    STRING response[4] = {NULL, NULL, NULL, NULL};
    long invert_response = 0;
    long keep_largest_SVs = 0;
    long remove_smallest_SVs = 0;
    double minimum_SV_ratio = 0;
    double Tikhonov_relative_alpha = 0;
    long Tikhonov_n = -1;
    double steering_Kp = 0;
    double steering_Ki = 1;
    double steering_Kd = 0;
    double rf_Kp = 0;
    double rf_Ki = 0;
    double rf_Kd = 0;
    double corrector_limit = 0;
    double rf_frequency_limit = 0;
    double rf_response_scale = 0;
    long anti_windup = 1;
    long reset_filters_each_step = 0;
    double bpm_noise = 0;
    double bpm_noise_cutoff = 0;
    STRING bpm_noise_distribution = "gaussian";
    long include_rf_frequency = 0;
    STRING bpm_filter_file = NULL;
    STRING steering_filter_file = NULL;
    STRING rf_filter_file = NULL;
    STRING output = NULL;
    long output_interval = 1;
    long center_on_orbit = 0;
    long center_momentum_also = 1;
    long offset_by_orbit = 0;
    long offset_momentum_also = 1;
    long verbosity = 1;
#end
