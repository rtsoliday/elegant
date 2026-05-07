#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

CPU_ELEGANT=${CPU_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64/elegant}
GPU_ELEGANT=${GPU_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-elegant}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/output}
LABEL_PREFIX=${LABEL_PREFIX:-production}
TOLERANCE=${TOLERANCE:-1e-11}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
GPU_MODE=${ELEGANT_GPU_MODE:-auto}
REPORT_FILE=${REPORT_FILE:-}
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Run the curated Phase 11 production smoke subset once with the CPU binary,
once with gpu-elegant, then compare the SDDS outputs.

Options:
  --cpu-elegant PATH      CPU elegant binary (default: bin/Linux-x86_64/elegant)
  --gpu-elegant PATH      CUDA gpu-elegant binary (default: bin/Linux-x86_64-gpu/gpu-elegant)
  --output DIR            Output root (default: test/gpu_cuda/output)
  --label-prefix NAME     Label prefix for output directories (default: production)
  --tolerance N           SDDS comparison tolerance (default: 1e-11)
  --timeout N             Per-case timeout in seconds (default: 180)
  --report PATH           Write a Markdown benchmark report after comparison
  --require-gpu           Fail if the CUDA binary cannot use a GPU device
  --dry-run               Print commands without executing them
  -h, --help              Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-elegant)
      CPU_ELEGANT=$2
      shift
      ;;
    --gpu-elegant)
      GPU_ELEGANT=$2
      shift
      ;;
    --output)
      OUTPUT_ROOT=$2
      shift
      ;;
    --label-prefix)
      LABEL_PREFIX=$2
      shift
      ;;
    --tolerance)
      TOLERANCE=$2
      shift
      ;;
    --timeout)
      TIMEOUT_SECONDS=$2
      shift
      ;;
    --report)
      REPORT_FILE=$2
      shift
      ;;
    --require-gpu)
      GPU_MODE=required
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

quote_cmd() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

metadata_value() {
  local name=$1
  if [ "${!name+x}" = x ]; then
    printf '%s\n' "${!name}"
  else
    printf 'unset\n'
  fi
}

run_cmd() {
  quote_cmd "$@"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

require_executable() {
  local path=$1
  if [ "$DRY_RUN" -eq 1 ]; then
    return
  fi
  if [ ! -x "$path" ]; then
    echo "missing executable: $path" >&2
    exit 1
  fi
}

CPU_ELEGANT=$(absolute_path "$CPU_ELEGANT")
GPU_ELEGANT=$(absolute_path "$GPU_ELEGANT")
OUTPUT_ROOT=$(absolute_path "$OUTPUT_ROOT")
cpu_label="$LABEL_PREFIX-cpu-production-smoke"
gpu_label="$LABEL_PREFIX-gpu-production-smoke"

require_executable "$CPU_ELEGANT"
require_executable "$GPU_ELEGANT"

gpu_env=(
  "TIMEOUT_SECONDS=$TIMEOUT_SECONDS"
  "ELEGANT_GPU_MODE=$GPU_MODE"
  "ELEGANT_GPU_VERBOSE=1"
  "ELEGANT_GPU_MIN_PARTICLES=1"
)
for gpu_env_name in \
  ELEGANT_GPU_ENABLE_APERTURE_COMPACTION \
  ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION \
  ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE
do
  if [ "${!gpu_env_name+x}" = x ]; then
    gpu_env+=("$gpu_env_name=${!gpu_env_name}")
  fi
done

run_cmd env TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
  "$SCRIPT_DIR/run_benchmarks.sh" --production-smoke --quick \
  --elegant "$CPU_ELEGANT" \
  --output "$OUTPUT_ROOT" \
  --label "$cpu_label"

run_cmd env "${gpu_env[@]}" \
  "$SCRIPT_DIR/run_benchmarks.sh" --production-smoke --quick \
  --elegant "$GPU_ELEGANT" \
  --output "$OUTPUT_ROOT" \
  --label "$gpu_label"

run_cmd python3 "$SCRIPT_DIR/compare_sdds.py" \
  "$OUTPUT_ROOT/$cpu_label" \
  "$OUTPUT_ROOT/$gpu_label" \
  --tolerance "$TOLERANCE"

if [ -n "$REPORT_FILE" ]; then
  run_cmd python3 "$SCRIPT_DIR/report_benchmarks.py" \
    --cpu-manifest "$OUTPUT_ROOT/$cpu_label/manifest.tsv" \
    --gpu-manifest "$OUTPUT_ROOT/$gpu_label/manifest.tsv" \
    --cpu-output-root "$OUTPUT_ROOT/$cpu_label" \
    --gpu-output-root "$OUTPUT_ROOT/$gpu_label" \
    --gpu-binary "$GPU_ELEGANT" \
    --run-compare \
    --tolerance "$TOLERANCE" \
    --title "elegant CUDA Production Smoke Report" \
    --metadata "CPU_ELEGANT=$CPU_ELEGANT" \
    --metadata "GPU_ELEGANT=$GPU_ELEGANT" \
    --metadata "TIMEOUT_SECONDS=$TIMEOUT_SECONDS" \
    --metadata "ELEGANT_GPU_MODE=$GPU_MODE" \
    --metadata "ELEGANT_GPU_VERBOSE=1" \
    --metadata "ELEGANT_GPU_MIN_PARTICLES=1" \
    --metadata "ELEGANT_GPU_ENABLE_APERTURE_COMPACTION=$(metadata_value ELEGANT_GPU_ENABLE_APERTURE_COMPACTION)" \
    --metadata "ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION=$(metadata_value ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION)" \
    --metadata "ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE=$(metadata_value ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE)" \
    --build-command "make" \
    --output "$REPORT_FILE"
fi
