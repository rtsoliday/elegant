#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OUTPUT_ROOT="$SCRIPT_DIR/output"

MODE=quick
TARGET_SECONDS=60
RUN_LABEL=
CASE_FILTER=
ELEGANT_BIN=${ELEGANT_BIN:-}
RPN_DEFNS_FILE=${RPN_DEFNS:-}
SEED=987654321
DRY_RUN=0
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
MPI_RANKS=${MPI_RANKS:-0}
MPIRUN=${MPIRUN:-mpirun}
PRODUCTION_SMOKE=0
PARTICLES_OVERRIDE=
PASSES_OVERRIDE=

CASES="matrix phase19_matrix_load_balance exact_drift phase2_helpers phase2_time_center phase2_special_matrix phase2_matrix_extended phase2_residency phase3_limit_amplitudes phase3_limit_loss phase3_elimit_amplitudes phase3_elimit_loss phase3_ecol phase3_ecol_loss phase3_scraper phase3_scraper_loss phase15_aperture_data_loss phase15_remove_invalid_loss phase15_rcol_open_loss phase15_ecol_mixed_loss phase15_ecol_open_global_loss phase15_rcol_open_global_loss phase15_scraper_global_loss phase15_scraper_two_sided_global_loss multipole phase17_multipole_misalignment phase55_mult_deterministic phase56_mult_loss_compaction phase57_mult_loss_accepted_compaction phase59_mult_loss_output_fallback phase60_mult_global_loss_fallback phase4_dqcor phase17_dqcor_misalignment csbend phase17_csbend_misalignment phase58_csbend_loss_compaction phase61_csbend_advanced_fallback phase62_kickmap_loss_compaction phase63_kickmap_loss_output_fallback phase64_kickmap_global_loss_fallback phase4_csbend_expanded phase4_csbend_ho_edge aperture_loss phase5_wake wake_trwake wake_trwake_fixed_bins phase16_wake_smoothing phase16_wake_change_p0 phase16_trwake_smoothing phase16_trwake_tilt phase16_lsc_smoothing_filter phase16_lsc_kick_mode phase16_lsc_auto_leffective phase16_lsc_backtrack phase16_lsc_low_frequency_filter phase16_fiducial_modulate phase16_bunched_wake_single phase16_bunched_wake_filter_skip phase16_bunched_wake_multibucket_skip phase22_bunched_wake_filter_select phase23_bunched_wake_filter_range phase24_bunched_wake_change_p0_skip lsc csr phase6_csr_csbend phase6_csr_bins_512 phase6_csr_bins_4096 phase6_csr_short_bunch phase6_csr_long_bunch phase14_csr_last_wake phase14_csr_filters phase14_csr_saldin54 phase14_csr_noop_drift_aperture phase14_csr_entry_edge phase14_csr_linear_drift phase7_scmult_linear phase65_scmult_nonlinear_fallback phase66_scmult_sliced_fallback phase67_scmult_multibunch_fallback phase18_rfca_thin phase26_rfca_thin_change_p0 phase27_rfca_thin_fiducial_modes phase28_rfca_thin_offset phase29_rfca_matrix_rf_only phase30_rfca_matrix_fiducial_modes phase39_rfca_kick_rf_only phase21_rfcw_rf_only phase25_rfcw_rf_only_offset phase31_rfcw_rf_only_fiducial_modes phase32_rfcw_matrix_wake phase33_rfcw_kick_wake phase34_rfcw_wakes_at_end phase35_rfcw_matrix_wakes_at_end phase36_rfcw_lsc phase37_rfcw_multikick phase38_rfcw_kick_rf_only phase41_rfcw_wake_pmaximum_fiducial phase42_rfcw_fixed_wake_bins phase43_rfcw_lsc_only phase44_rfcw_single_wake_planes phase45_rf_kick_treference phase46_rfcw_wake_treference phase47_rf_selected_tmean_fiducial phase48_rf_selected_pmaximum_fiducial phase49_rfcw_wake_selected_fiducial phase50_rf_first_fiducial phase51_rf_standing_wave_single phase52_rf_standing_wave_multikick_treference phase53_rfca_standing_wave_multikick_fiducial phase54_rfcw_standing_wave_multikick_fiducial rfcw lcls0"
PRODUCTION_SMOKE_CASES="lcls0 lcls1 clic1 csbend1 maxamp1 collimate1 collimate2 collimate3 dqcor1"

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --quick                 Run short smoke tests (default)
  --baseline              Scale CPU pass counts toward --target-seconds
  --target-seconds N      Target CPU runtime per baseline case (default: 60)
  --case NAME             Run one case only
  --production-smoke      Run the curated bounded production smoke subset
  --particles N           Override the default particle count for selected cases
  --passes N              Override the default pass count for selected cases
  --elegant PATH          elegant binary to run
  --rpn-defns PATH        RPN definitions file
  --output DIR            Output root (default: test/gpu_cuda/output)
  --label NAME            Run label under the output root
  --seed N                Random seed passed to all benchmark inputs
  --mpi-ranks N           Launch each run with mpirun -np N
  --dry-run               Print commands without running elegant
  -h, --help              Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick)
      MODE=quick
      ;;
    --baseline)
      MODE=baseline
      ;;
    --target-seconds)
      TARGET_SECONDS=$2
      shift
      ;;
    --case)
      CASE_FILTER=$2
      shift
      ;;
    --production-smoke)
      PRODUCTION_SMOKE=1
      ;;
    --particles)
      PARTICLES_OVERRIDE=$2
      shift
      ;;
    --passes)
      PASSES_OVERRIDE=$2
      shift
      ;;
    --elegant)
      ELEGANT_BIN=$2
      shift
      ;;
    --rpn-defns)
      RPN_DEFNS_FILE=$2
      shift
      ;;
    --output)
      OUTPUT_ROOT=$2
      shift
      ;;
    --label)
      RUN_LABEL=$2
      shift
      ;;
    --seed)
      SEED=$2
      shift
      ;;
    --mpi-ranks)
      MPI_RANKS=$2
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

find_elegant() {
  if [ -n "$ELEGANT_BIN" ]; then
    absolute_path "$ELEGANT_BIN"
    return
  fi
  if [ -x "$REPO_ROOT/bin/Linux-x86_64/elegant" ]; then
    printf '%s\n' "$REPO_ROOT/bin/Linux-x86_64/elegant"
    return
  fi
  if [ -x "$REPO_ROOT/src/O.Linux-x86_64/elegant" ]; then
    printf '%s\n' "$REPO_ROOT/src/O.Linux-x86_64/elegant"
    return
  fi
  if command -v elegant >/dev/null 2>&1; then
    command -v elegant
    return
  fi
  echo "unable to find elegant; use --elegant PATH" >&2
  exit 1
}

absolute_path() {
  local path=$1
  local dir
  local base

  case "$path" in
    /*)
      printf '%s\n' "$path"
      ;;
    */*)
      dir=$(cd "$(dirname "$path")" && pwd)
      base=$(basename "$path")
      printf '%s/%s\n' "$dir" "$base"
      ;;
    *)
      if command -v "$path" >/dev/null 2>&1; then
        command -v "$path"
      else
        printf '%s\n' "$path"
      fi
      ;;
  esac
}

find_rpn_defns() {
  if [ -n "$RPN_DEFNS_FILE" ]; then
    absolute_path "$RPN_DEFNS_FILE"
    return
  fi
  for candidate in \
    "$REPO_ROOT/../SDDS/defns.rpn" \
    "$REPO_ROOT/defns.rpn" \
    /usr/local/share/defns.rpn \
    /usr/share/defns.rpn
  do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "unable to find defns.rpn; use --rpn-defns PATH or set RPN_DEFNS" >&2
  exit 1
}

case_defaults() {
  case "$1:$MODE" in
    matrix:quick) printf '2000 5\n' ;;
    matrix:baseline) printf '20000 20\n' ;;
    phase19_matrix_load_balance:quick) printf '12000 10\n' ;;
    phase19_matrix_load_balance:baseline) printf '20000 20\n' ;;
    exact_drift:quick) printf '2000 10\n' ;;
    exact_drift:baseline) printf '30000 40\n' ;;
    phase2_helpers:quick) printf '2000 5\n' ;;
    phase2_helpers:baseline) printf '20000 20\n' ;;
    phase2_time_center:quick) printf '2000 5\n' ;;
    phase2_time_center:baseline) printf '20000 200\n' ;;
    phase2_special_matrix:quick) printf '2000 5\n' ;;
    phase2_special_matrix:baseline) printf '20000 20\n' ;;
    phase2_matrix_extended:quick) printf '2000 5\n' ;;
    phase2_matrix_extended:baseline) printf '20000 20\n' ;;
    phase2_residency:quick) printf '2000 5\n' ;;
    phase2_residency:baseline) printf '20000 20\n' ;;
    phase3_limit_amplitudes:quick) printf '3000 5\n' ;;
    phase3_limit_amplitudes:baseline) printf '30000 20\n' ;;
    phase3_limit_loss:quick) printf '3000 3\n' ;;
    phase3_limit_loss:baseline) printf '30000 10\n' ;;
    phase3_elimit_amplitudes:quick) printf '3000 5\n' ;;
    phase3_elimit_amplitudes:baseline) printf '30000 20\n' ;;
    phase3_elimit_loss:quick) printf '3000 3\n' ;;
    phase3_elimit_loss:baseline) printf '30000 10\n' ;;
    phase3_rcol:quick) printf '3000 5\n' ;;
    phase3_rcol:baseline) printf '30000 20\n' ;;
    phase3_rcol_loss:quick) printf '3000 3\n' ;;
    phase3_rcol_loss:baseline) printf '30000 10\n' ;;
    phase3_ecol:quick) printf '3000 5\n' ;;
    phase3_ecol:baseline) printf '30000 20\n' ;;
    phase3_ecol_loss:quick) printf '3000 3\n' ;;
    phase3_ecol_loss:baseline) printf '30000 10\n' ;;
    phase3_scraper:quick) printf '3000 5\n' ;;
    phase3_scraper:baseline) printf '30000 20\n' ;;
    phase3_scraper_loss:quick) printf '3000 3\n' ;;
    phase3_scraper_loss:baseline) printf '30000 10\n' ;;
    phase15_aperture_data_loss:quick) printf '3000 3\n' ;;
    phase15_aperture_data_loss:baseline) printf '30000 10\n' ;;
    phase15_remove_invalid_loss:quick) printf '0 3\n' ;;
    phase15_remove_invalid_loss:baseline) printf '0 100\n' ;;
    phase15_rcol_open_loss:quick) printf '3000 3\n' ;;
    phase15_rcol_open_loss:baseline) printf '30000 10\n' ;;
    phase15_ecol_mixed_loss:quick) printf '3000 3\n' ;;
    phase15_ecol_mixed_loss:baseline) printf '30000 10\n' ;;
    phase15_elimit_loss_no_output:quick) printf '3000 3\n' ;;
    phase15_elimit_loss_no_output:baseline) printf '30000 10\n' ;;
    phase15_ecol_open_global_loss:quick) printf '3000 3\n' ;;
    phase15_ecol_open_global_loss:baseline) printf '30000 10\n' ;;
    phase15_rcol_open_global_loss:quick) printf '3000 3\n' ;;
    phase15_rcol_open_global_loss:baseline) printf '30000 10\n' ;;
    phase15_scraper_global_loss:quick) printf '3000 3\n' ;;
    phase15_scraper_global_loss:baseline) printf '30000 10\n' ;;
    phase15_scraper_two_sided_global_loss:quick) printf '3000 3\n' ;;
    phase15_scraper_two_sided_global_loss:baseline) printf '30000 10\n' ;;
    multipole:quick) printf '2000 2\n' ;;
    multipole:baseline) printf '20000 10\n' ;;
    phase17_multipole_misalignment:quick) printf '3000 2\n' ;;
    phase17_multipole_misalignment:baseline) printf '30000 10\n' ;;
    phase55_mult_deterministic:quick) printf '3000 2\n' ;;
    phase55_mult_deterministic:baseline) printf '30000 10\n' ;;
    phase56_mult_loss_compaction:quick) printf '3000 1\n' ;;
    phase56_mult_loss_compaction:baseline) printf '30000 5\n' ;;
    phase57_mult_loss_accepted_compaction:quick) printf '3000 1\n' ;;
    phase57_mult_loss_accepted_compaction:baseline) printf '30000 5\n' ;;
    phase59_mult_loss_output_fallback:quick) printf '3000 1\n' ;;
    phase59_mult_loss_output_fallback:baseline) printf '30000 5\n' ;;
    phase60_mult_global_loss_fallback:quick) printf '3000 1\n' ;;
    phase60_mult_global_loss_fallback:baseline) printf '30000 5\n' ;;
    phase58_csbend_loss_compaction:quick) printf '3000 1\n' ;;
    phase58_csbend_loss_compaction:baseline) printf '30000 5\n' ;;
    phase61_csbend_advanced_fallback:quick) printf '3000 2\n' ;;
    phase61_csbend_advanced_fallback:baseline) printf '30000 8\n' ;;
    phase62_kickmap_loss_compaction:quick) printf '3000 3\n' ;;
    phase62_kickmap_loss_compaction:baseline) printf '30000 20\n' ;;
    phase63_kickmap_loss_output_fallback:quick) printf '3000 3\n' ;;
    phase63_kickmap_loss_output_fallback:baseline) printf '30000 20\n' ;;
    phase64_kickmap_global_loss_fallback:quick) printf '3000 3\n' ;;
    phase64_kickmap_global_loss_fallback:baseline) printf '30000 20\n' ;;
    phase4_dqcor:quick) printf '2000 3\n' ;;
    phase4_dqcor:baseline) printf '20000 20\n' ;;
    phase17_dqcor_misalignment:quick) printf '3000 3\n' ;;
    phase17_dqcor_misalignment:baseline) printf '30000 10\n' ;;
    csbend:quick) printf '2000 2\n' ;;
    csbend:baseline) printf '20000 8\n' ;;
    phase17_csbend_misalignment:quick) printf '3000 2\n' ;;
    phase17_csbend_misalignment:baseline) printf '30000 8\n' ;;
    phase4_csbend_expanded:quick) printf '2000 2\n' ;;
    phase4_csbend_expanded:baseline) printf '20000 8\n' ;;
    phase4_csbend_ho_edge:quick) printf '2000 2\n' ;;
    phase4_csbend_ho_edge:baseline) printf '20000 8\n' ;;
    aperture_loss:quick) printf '3000 2\n' ;;
    aperture_loss:baseline) printf '30000 8\n' ;;
    phase5_wake:quick) printf '3000 1\n' ;;
    phase5_wake:baseline) printf '30000 5\n' ;;
    wake_trwake:quick) printf '3000 1\n' ;;
    wake_trwake:baseline) printf '30000 5\n' ;;
    wake_trwake_fixed_bins:quick) printf '3000 1\n' ;;
    wake_trwake_fixed_bins:baseline) printf '30000 5\n' ;;
    phase16_wake_smoothing:quick) printf '3000 1\n' ;;
    phase16_wake_smoothing:baseline) printf '30000 5\n' ;;
    phase16_wake_change_p0:quick) printf '3000 1\n' ;;
    phase16_wake_change_p0:baseline) printf '30000 5\n' ;;
    phase16_trwake_smoothing:quick) printf '3000 1\n' ;;
    phase16_trwake_smoothing:baseline) printf '30000 5\n' ;;
    phase16_trwake_tilt:quick) printf '3000 1\n' ;;
    phase16_trwake_tilt:baseline) printf '30000 5\n' ;;
    phase16_lsc_smoothing_filter:quick) printf '3000 1\n' ;;
    phase16_lsc_smoothing_filter:baseline) printf '30000 5\n' ;;
    phase16_lsc_kick_mode:quick) printf '3000 1\n' ;;
    phase16_lsc_kick_mode:baseline) printf '30000 5\n' ;;
    phase16_lsc_auto_leffective:quick) printf '3000 1\n' ;;
    phase16_lsc_auto_leffective:baseline) printf '30000 5\n' ;;
    phase16_lsc_backtrack:quick) printf '3000 1\n' ;;
    phase16_lsc_backtrack:baseline) printf '30000 5\n' ;;
    phase16_lsc_low_frequency_filter:quick) printf '3000 1\n' ;;
    phase16_lsc_low_frequency_filter:baseline) printf '30000 5\n' ;;
    phase16_fiducial_modulate:quick) printf '3000 5\n' ;;
    phase16_fiducial_modulate:baseline) printf '30000 20\n' ;;
    phase16_bunched_wake_single:quick) printf '3000 1\n' ;;
    phase16_bunched_wake_single:baseline) printf '30000 5\n' ;;
    phase16_bunched_wake_filter_skip:quick) printf '3000 5\n' ;;
    phase16_bunched_wake_filter_skip:baseline) printf '30000 40\n' ;;
    phase16_bunched_wake_multibucket_skip:quick) printf '1000 5\n' ;;
    phase16_bunched_wake_multibucket_skip:baseline) printf '10000 40\n' ;;
    phase22_bunched_wake_filter_select:quick) printf '1000 5\n' ;;
    phase22_bunched_wake_filter_select:baseline) printf '10000 40\n' ;;
    phase23_bunched_wake_filter_range:quick) printf '1200 4\n' ;;
    phase23_bunched_wake_filter_range:baseline) printf '12000 30\n' ;;
    phase24_bunched_wake_change_p0_skip:quick) printf '1200 5\n' ;;
    phase24_bunched_wake_change_p0_skip:baseline) printf '12000 40\n' ;;
    lsc:quick) printf '3000 1\n' ;;
    lsc:baseline) printf '30000 5\n' ;;
    csr:quick) printf '2000 1\n' ;;
    csr:baseline) printf '20000 3\n' ;;
    phase6_csr_csbend:quick) printf '2000 1\n' ;;
    phase6_csr_csbend:baseline) printf '20000 1\n' ;;
    phase6_csr_bins_512:quick) printf '2000 1\n' ;;
    phase6_csr_bins_512:baseline) printf '20000 1\n' ;;
    phase6_csr_bins_4096:quick) printf '2000 1\n' ;;
    phase6_csr_bins_4096:baseline) printf '20000 1\n' ;;
    phase6_csr_short_bunch:quick) printf '2000 1\n' ;;
    phase6_csr_short_bunch:baseline) printf '20000 1\n' ;;
    phase6_csr_long_bunch:quick) printf '2000 1\n' ;;
    phase6_csr_long_bunch:baseline) printf '20000 1\n' ;;
    phase14_csr_last_wake:quick) printf '2000 1\n' ;;
    phase14_csr_last_wake:baseline) printf '20000 1\n' ;;
    phase14_csr_filters:quick) printf '2000 1\n' ;;
    phase14_csr_filters:baseline) printf '20000 1\n' ;;
    phase14_csr_saldin54:quick) printf '2000 1\n' ;;
    phase14_csr_saldin54:baseline) printf '20000 1\n' ;;
    phase14_csr_noop_drift_aperture:quick) printf '2000 1\n' ;;
    phase14_csr_noop_drift_aperture:baseline) printf '20000 1\n' ;;
    phase14_csr_entry_edge:quick) printf '2000 1\n' ;;
    phase14_csr_entry_edge:baseline) printf '20000 1\n' ;;
    phase14_csr_linear_drift:quick) printf '2000 1\n' ;;
    phase14_csr_linear_drift:baseline) printf '20000 1\n' ;;
    phase7_scmult_linear:quick) printf '3000 1\n' ;;
    phase7_scmult_linear:baseline) printf '30000 3\n' ;;
    phase65_scmult_nonlinear_fallback:quick) printf '2000 2\n' ;;
    phase65_scmult_nonlinear_fallback:baseline) printf '10000 10\n' ;;
    phase66_scmult_sliced_fallback:quick) printf '2000 2\n' ;;
    phase66_scmult_sliced_fallback:baseline) printf '10000 10\n' ;;
    phase67_scmult_multibunch_fallback:quick) printf '1000 2\n' ;;
    phase67_scmult_multibunch_fallback:baseline) printf '10000 10\n' ;;
    phase18_rfca_thin:quick) printf '3000 8\n' ;;
    phase18_rfca_thin:baseline) printf '30000 80\n' ;;
    phase26_rfca_thin_change_p0:quick) printf '3000 4\n' ;;
    phase26_rfca_thin_change_p0:baseline) printf '30000 40\n' ;;
    phase27_rfca_thin_fiducial_modes:quick) printf '3000 4\n' ;;
    phase27_rfca_thin_fiducial_modes:baseline) printf '30000 40\n' ;;
    phase28_rfca_thin_offset:quick) printf '3000 4\n' ;;
    phase28_rfca_thin_offset:baseline) printf '30000 40\n' ;;
    phase29_rfca_matrix_rf_only:quick) printf '3000 4\n' ;;
    phase29_rfca_matrix_rf_only:baseline) printf '30000 40\n' ;;
    phase30_rfca_matrix_fiducial_modes:quick) printf '3000 4\n' ;;
    phase30_rfca_matrix_fiducial_modes:baseline) printf '30000 40\n' ;;
    phase39_rfca_kick_rf_only:quick) printf '3000 4\n' ;;
    phase39_rfca_kick_rf_only:baseline) printf '30000 40\n' ;;
    phase40_rf_pmaximum_fiducial:quick) printf '3000 4\n' ;;
    phase40_rf_pmaximum_fiducial:baseline) printf '30000 40\n' ;;
    phase21_rfcw_rf_only:quick) printf '3000 4\n' ;;
    phase21_rfcw_rf_only:baseline) printf '30000 40\n' ;;
    phase25_rfcw_rf_only_offset:quick) printf '3000 4\n' ;;
    phase25_rfcw_rf_only_offset:baseline) printf '30000 40\n' ;;
    phase31_rfcw_rf_only_fiducial_modes:quick) printf '3000 4\n' ;;
    phase31_rfcw_rf_only_fiducial_modes:baseline) printf '30000 40\n' ;;
    phase32_rfcw_matrix_wake:quick) printf '3000 4\n' ;;
    phase32_rfcw_matrix_wake:baseline) printf '30000 40\n' ;;
    phase33_rfcw_kick_wake:quick) printf '3000 4\n' ;;
    phase33_rfcw_kick_wake:baseline) printf '30000 40\n' ;;
    phase34_rfcw_wakes_at_end:quick) printf '3000 4\n' ;;
    phase34_rfcw_wakes_at_end:baseline) printf '30000 40\n' ;;
    phase35_rfcw_matrix_wakes_at_end:quick) printf '3000 4\n' ;;
    phase35_rfcw_matrix_wakes_at_end:baseline) printf '30000 40\n' ;;
    phase36_rfcw_lsc:quick) printf '3000 3\n' ;;
    phase36_rfcw_lsc:baseline) printf '30000 30\n' ;;
    phase37_rfcw_multikick:quick) printf '3000 3\n' ;;
    phase37_rfcw_multikick:baseline) printf '30000 30\n' ;;
    phase38_rfcw_kick_rf_only:quick) printf '3000 4\n' ;;
    phase38_rfcw_kick_rf_only:baseline) printf '30000 40\n' ;;
    phase41_rfcw_wake_pmaximum_fiducial:quick) printf '3000 3\n' ;;
    phase41_rfcw_wake_pmaximum_fiducial:baseline) printf '30000 30\n' ;;
    phase42_rfcw_fixed_wake_bins:quick) printf '3000 3\n' ;;
    phase42_rfcw_fixed_wake_bins:baseline) printf '30000 30\n' ;;
    phase43_rfcw_lsc_only:quick) printf '3000 3\n' ;;
    phase43_rfcw_lsc_only:baseline) printf '30000 30\n' ;;
    phase44_rfcw_single_wake_planes:quick) printf '3000 3\n' ;;
    phase44_rfcw_single_wake_planes:baseline) printf '30000 30\n' ;;
    phase45_rf_kick_treference:quick) printf '3000 4\n' ;;
    phase45_rf_kick_treference:baseline) printf '30000 40\n' ;;
    phase46_rfcw_wake_treference:quick) printf '3000 3\n' ;;
    phase46_rfcw_wake_treference:baseline) printf '30000 30\n' ;;
    phase47_rf_selected_tmean_fiducial:quick) printf '2000 3\n' ;;
    phase47_rf_selected_tmean_fiducial:baseline) printf '20000 30\n' ;;
    phase48_rf_selected_pmaximum_fiducial:quick) printf '2000 3\n' ;;
    phase48_rf_selected_pmaximum_fiducial:baseline) printf '20000 30\n' ;;
    phase49_rfcw_wake_selected_fiducial:quick) printf '2000 3\n' ;;
    phase49_rfcw_wake_selected_fiducial:baseline) printf '20000 30\n' ;;
    phase50_rf_first_fiducial:quick) printf '2000 3\n' ;;
    phase50_rf_first_fiducial:baseline) printf '20000 30\n' ;;
    phase51_rf_standing_wave_single:quick) printf '2000 3\n' ;;
    phase51_rf_standing_wave_single:baseline) printf '20000 30\n' ;;
    phase52_rf_standing_wave_multikick_treference:quick) printf '2000 3\n' ;;
    phase52_rf_standing_wave_multikick_treference:baseline) printf '20000 30\n' ;;
    phase53_rfca_standing_wave_multikick_fiducial:quick) printf '2000 3\n' ;;
    phase53_rfca_standing_wave_multikick_fiducial:baseline) printf '20000 30\n' ;;
    phase54_rfcw_standing_wave_multikick_fiducial:quick) printf '2000 3\n' ;;
    phase54_rfcw_standing_wave_multikick_fiducial:baseline) printf '20000 30\n' ;;
    rfcw:quick) printf '3000 1\n' ;;
    rfcw:baseline) printf '30000 5\n' ;;
    lcls0:quick) printf '0 1\n' ;;
    lcls0:baseline) printf '0 1\n' ;;
    lcls1:quick) printf '0 1\n' ;;
    lcls1:baseline) printf '0 1\n' ;;
    clic1:quick) printf '2000 1\n' ;;
    clic1:baseline) printf '20000 1\n' ;;
    csbend1:quick) printf '3000 2\n' ;;
    csbend1:baseline) printf '30000 8\n' ;;
    maxamp1:quick) printf '3000 3\n' ;;
    maxamp1:baseline) printf '30000 10\n' ;;
    collimate1:quick) printf '3000 1\n' ;;
    collimate1:baseline) printf '30000 3\n' ;;
    collimate2:quick) printf '3000 1\n' ;;
    collimate2:baseline) printf '30000 3\n' ;;
    collimate3:quick) printf '3000 1\n' ;;
    collimate3:baseline) printf '30000 3\n' ;;
    dqcor1:quick) printf '2000 3\n' ;;
    dqcor1:baseline) printf '20000 20\n' ;;
    spinTest2:quick) printf '100 128\n' ;;
    spinTest2:baseline) printf '1000 1024\n' ;;
    scRing2:quick) printf '0 8\n' ;;
    scRing2:baseline) printf '0 64\n' ;;
    scRing2_no_watch:quick) printf '0 8\n' ;;
    scRing2_no_watch:baseline) printf '0 64\n' ;;
    ionEffectsPoisson:quick) printf '2000 3\n' ;;
    ionEffectsPoisson:baseline) printf '10000 10\n' ;;
    bmapxy1:quick) printf '1000 1\n' ;;
    bmapxy1:baseline) printf '10000 3\n' ;;
    bmxyz1:quick) printf '25 1\n' ;;
    bmxyz1:baseline) printf '100 1\n' ;;
    boffaxe1:quick) printf '100 1\n' ;;
    boffaxe1:baseline) printf '1000 1\n' ;;
    cwiggler10:quick) printf '100 10\n' ;;
    cwiggler10:baseline) printf '1000 20\n' ;;
    cwiggler10_radiation:quick) printf '100 2\n' ;;
    cwiggler10_radiation:baseline) printf '1000 10\n' ;;
    uKickMap4_radiation:quick) printf '100 1\n' ;;
    uKickMap4_radiation:baseline) printf '1000 1\n' ;;
    uKickMap1:quick) printf '3000 2\n' ;;
    uKickMap1:baseline) printf '30000 5\n' ;;
    latticeErrors6:quick) printf '3000 2\n' ;;
    latticeErrors6:baseline) printf '30000 5\n' ;;
    latticeErrors6_loss_output:quick) printf '3000 2\n' ;;
    latticeErrors6_loss_output:baseline) printf '30000 5\n' ;;
    latticeErrors6_global_loss:quick) printf '3000 2\n' ;;
    latticeErrors6_global_loss:baseline) printf '30000 5\n' ;;
    *) printf '2000 1\n' ;;
  esac
}

case_directory() {
  if [ -f "$SCRIPT_DIR/cases/$1/benchmark.ele" ]; then
    printf '%s\n' "$SCRIPT_DIR/cases/$1"
    return
  fi
  if [ -f "$SCRIPT_DIR/production_cases/$1/benchmark.ele" ]; then
    printf '%s\n' "$SCRIPT_DIR/production_cases/$1"
    return
  fi
  return 1
}

initial_extra_macros() {
  case "$1" in
    lcls0) printf 'sample_fraction=0.1\n' ;;
    lcls1) printf 'sample_fraction=0.05\n' ;;
    *) printf '\n' ;;
  esac
}

scaled_extra_macros() {
  case "$1" in
    lcls0)
      awk -v elapsed="$2" -v target="$TARGET_SECONDS" '
        BEGIN {
          fraction = 0.1
          if (elapsed > 0)
            fraction = 0.1 * target / elapsed
          if (fraction > 1)
            fraction = 1
          if (fraction < 0.001)
            fraction = 0.001
          printf "sample_fraction=%.8g\n", fraction
        }'
      ;;
    lcls1)
      awk -v elapsed="$2" -v target="$TARGET_SECONDS" '
        BEGIN {
          fraction = 0.05
          if (elapsed > 0)
            fraction = 0.05 * target / elapsed
          if (fraction > 1)
            fraction = 1
          if (fraction < 0.001)
            fraction = 0.001
          printf "sample_fraction=%.8g\n", fraction
        }'
      ;;
    *) printf '\n' ;;
  esac
}

scale_passes_for_case() {
  case "$1" in
    lcls0|lcls1) printf '%s\n' "$2" ;;
    *) scale_passes "$2" "$3" ;;
  esac
}

read_real_time() {
  if [ ! -f "$1" ]; then
    printf '0\n'
    return
  fi
  awk '/^real[[:space:]]+/ { value=$2 } END { if (value == "") value=0; print value }' "$1"
}

scale_passes() {
  awk -v passes="$1" -v elapsed="$2" -v target="$TARGET_SECONDS" '
    BEGIN {
      if (elapsed <= 0) {
        scaled = passes
      } else {
        scaled = int(passes * target / elapsed + 0.5)
      }
      if (scaled < 1)
        scaled = 1
      if (scaled > 100000)
        scaled = 100000
      print scaled
    }'
}

run_case_once() {
  case_name=$1
  particles=$2
  passes=$3
  destination=$4
  extra_macros=${5:-}

  case_dir=$(case_directory "$case_name")
  mkdir -p "$destination"
  root="$destination/$case_name"
  stdout_file="$destination/elegant.stdout"
  stderr_file="$destination/elegant.stderr"
  time_file="$destination/time.txt"

  macro_string="root=$root,n_particles=$particles,n_passes=$passes,seed=$SEED"
  if [ -n "$extra_macros" ]; then
    macro_string="$macro_string,$extra_macros"
  fi
  cmd=()
  if [ "$MPI_RANKS" -gt 0 ]; then
    cmd=("$MPIRUN" -np "$MPI_RANKS")
  fi
  cmd+=("$ELEGANT_BIN" "$case_dir/benchmark.ele" "-macro=$macro_string" "-rpnDefns=$RPN_DEFNS_FILE")
  echo "case=$case_name particles=$particles passes=$passes root=$root"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  (
    cd "$case_dir"
    /usr/bin/time -p timeout "$TIMEOUT_SECONDS" "${cmd[@]}"
  ) >"$stdout_file" 2>"$stderr_file.time"
  status=$?
  awk '/^(real|user|sys)[[:space:]]/ { print }' "$stderr_file.time" >"$time_file"
  awk '!/^(real|user|sys)[[:space:]]/ { print }' "$stderr_file.time" >"$stderr_file"
  rm -f "$stderr_file.time"
  return "$status"
}

ELEGANT_BIN=$(find_elegant)
RPN_DEFNS_FILE=$(find_rpn_defns)
if [ ! -x "$ELEGANT_BIN" ]; then
  echo "elegant is not executable: $ELEGANT_BIN" >&2
  exit 1
fi
if [ ! -f "$RPN_DEFNS_FILE" ]; then
  echo "RPN definitions file not found: $RPN_DEFNS_FILE" >&2
  exit 1
fi

if [ -z "$RUN_LABEL" ]; then
  RUN_LABEL="cpu-$MODE"
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/manifest.tsv"
printf 'case\tmode\tmpi_ranks\tparticles\tpasses\textra_macros\tstatus\treal_seconds\toutput_dir\n' >"$MANIFEST"

selected_cases=$CASES
if [ "$PRODUCTION_SMOKE" -eq 1 ]; then
  selected_cases=$PRODUCTION_SMOKE_CASES
fi
if [ -n "$CASE_FILTER" ]; then
  selected_cases=$CASE_FILTER
fi

overall_status=0
for case_name in $selected_cases; do
  if ! case_directory "$case_name" >/dev/null; then
    echo "unknown or incomplete case: $case_name" >&2
    overall_status=1
    continue
  fi

  set -- $(case_defaults "$case_name")
  particles=$1
  passes=$2
  if [ -n "$PARTICLES_OVERRIDE" ]; then
    particles=$PARTICLES_OVERRIDE
  fi
  if [ -n "$PASSES_OVERRIDE" ]; then
    passes=$PASSES_OVERRIDE
  fi
  extra_macros=$(initial_extra_macros "$case_name")
  final_dir="$RUN_DIR/$case_name"

  if [ "$MODE" = baseline ]; then
    sample_dir="$RUN_DIR/${case_name}.sample"
    sample_status=0
    run_case_once "$case_name" "$particles" "$passes" "$sample_dir" "$extra_macros" || sample_status=$?
    sample_elapsed=$(read_real_time "$sample_dir/time.txt")
    if [ "$sample_status" -ne 0 ]; then
      printf '%s\t%s-sample\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$case_name" "$MODE" "$MPI_RANKS" "$particles" "$passes" "$extra_macros" "$sample_status" "$sample_elapsed" "$sample_dir" >>"$MANIFEST"
      overall_status=1
      continue
    fi
    passes=$(scale_passes_for_case "$case_name" "$passes" "$sample_elapsed")
    extra_macros=$(scaled_extra_macros "$case_name" "$sample_elapsed")
  fi

  status=0
  run_case_once "$case_name" "$particles" "$passes" "$final_dir" "$extra_macros" || status=$?
  elapsed=$(read_real_time "$final_dir/time.txt")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$case_name" "$MODE" "$MPI_RANKS" "$particles" "$passes" "$extra_macros" "$status" "$elapsed" "$final_dir" >>"$MANIFEST"
  if [ "$status" -ne 0 ]; then
    overall_status=1
  fi
done

echo "wrote $MANIFEST"
exit "$overall_status"
