#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

JOBS=${JOBS:-4}
NVCC_PATH=${NVCC:-}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
CASE_NAME=${CASE_NAME:-matrix}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/output}
TARGET_SECONDS=${TARGET_SECONDS:-60}
TOLERANCE=${TOLERANCE:-1e-11}
MPI_RANKS=${MPI_RANKS:-2}
RUN_LABEL_PREFIX=${RUN_LABEL_PREFIX:-ci}
CPU_ELEGANT=${CPU_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64/elegant}
GPU_ELEGANT=${GPU_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-elegant}
GPU_VERIFY_ELEGANT=${GPU_VERIFY_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu-verify/gpu-elegant}
CPU_PELEGANT=${CPU_PELEGANT:-$REPO_ROOT/bin/Linux-x86_64/Pelegant}
GPU_PELEGANT=${GPU_PELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-Pelegant}

DO_CPU_BUILD=0
DO_CUDA_BUILD=0
DO_CUDA_VERIFY_BUILD=0
DO_QUICK=0
DO_VERIFY_QUICK=0
DO_BASELINE=0
DO_MPI_SMOKE=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Build options:
  --cpu-build             Build the ordinary CPU binary
  --cuda-build            Build the CUDA gpu-elegant binary
  --cuda-verify-build     Build the GPU_VERIFY CUDA binary

Run options:
  --quick                 Run CPU and CUDA quick smoke, then compare outputs
  --verify-quick          Run a quick smoke with the GPU_VERIFY CUDA binary
  --baseline              Run CPU and CUDA timing jobs aimed at --target-seconds
  --mpi-smoke             Run fixed-rank CPU and CUDA Pelegant smoke, then compare

Controls:
  --case NAME             Benchmark case to run (default: matrix)
  --output DIR            Benchmark output root (default: test/gpu_cuda/output)
  --target-seconds N      Baseline target, in seconds (default: 60)
  --tolerance N           SDDS comparison tolerance (default: 1e-11)
  --mpi-ranks N           MPI ranks for --mpi-smoke (default: 2)
  --jobs N                Parallel make jobs (default: 4)
  --nvcc PATH             CUDA compiler path
  --cuda-arch ARCH        CUDA architecture, e.g. sm_86 (default: sm_86)
  --label-prefix NAME     Prefix for benchmark output labels (default: ci)
  --dry-run               Print commands without executing them
  -h, --help              Show this help

If no build or run option is given, --cpu-build is used.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-build)
      DO_CPU_BUILD=1
      ;;
    --cuda-build)
      DO_CUDA_BUILD=1
      ;;
    --cuda-verify-build)
      DO_CUDA_VERIFY_BUILD=1
      ;;
    --quick)
      DO_QUICK=1
      ;;
    --verify-quick)
      DO_VERIFY_QUICK=1
      ;;
    --baseline)
      DO_BASELINE=1
      ;;
    --mpi-smoke)
      DO_MPI_SMOKE=1
      ;;
    --case)
      CASE_NAME=$2
      shift
      ;;
    --output)
      OUTPUT_ROOT=$2
      shift
      ;;
    --target-seconds)
      TARGET_SECONDS=$2
      shift
      ;;
    --tolerance)
      TOLERANCE=$2
      shift
      ;;
    --mpi-ranks)
      MPI_RANKS=$2
      shift
      ;;
    --jobs)
      JOBS=$2
      shift
      ;;
    --nvcc)
      NVCC_PATH=$2
      shift
      ;;
    --cuda-arch)
      CUDA_ARCH=$2
      shift
      ;;
    --label-prefix)
      RUN_LABEL_PREFIX=$2
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

if [ "$DO_CPU_BUILD$DO_CUDA_BUILD$DO_CUDA_VERIFY_BUILD$DO_QUICK$DO_VERIFY_QUICK$DO_BASELINE$DO_MPI_SMOKE" = "0000000" ]; then
  DO_CPU_BUILD=1
fi

quote_cmd() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
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

find_nvcc() {
  local candidate

  if [ -n "$NVCC_PATH" ]; then
    printf '%s\n' "$NVCC_PATH"
    return
  fi

  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return
  fi

  for candidate in \
    /usr/local/cuda-12.4/bin/nvcc \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-13.0/bin/nvcc \
    /usr/local/cuda-12.5/bin/nvcc \
    /usr/local/cuda-12.3/bin/nvcc
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "unable to find nvcc; use --nvcc PATH or set NVCC" >&2
  exit 1
}

run_compare() {
  local cpu_label=$1
  local gpu_label=$2
  run_cmd python3 "$SCRIPT_DIR/compare_sdds.py" \
    "$OUTPUT_ROOT/$cpu_label/$CASE_NAME" \
    "$OUTPUT_ROOT/$gpu_label/$CASE_NAME" \
    --tolerance "$TOLERANCE"
}

if [ "$DO_CPU_BUILD" -eq 1 ]; then
  run_cmd make -C "$REPO_ROOT/src" -j "$JOBS"
fi

if [ "$DO_CUDA_BUILD" -eq 1 ]; then
  NVCC_PATH=$(find_nvcc)
  run_cmd make -C "$REPO_ROOT/src" HAVE_CUDA=1 NVCC="$NVCC_PATH" CUDA_ARCH="$CUDA_ARCH" -j "$JOBS"
fi

if [ "$DO_CUDA_VERIFY_BUILD" -eq 1 ]; then
  NVCC_PATH=$(find_nvcc)
  run_cmd make -C "$REPO_ROOT/src" HAVE_CUDA=1 GPU_VERIFY=1 NVCC="$NVCC_PATH" CUDA_ARCH="$CUDA_ARCH" -j "$JOBS"
fi

if [ "$DO_QUICK" -eq 1 ]; then
  cpu_label="$RUN_LABEL_PREFIX-cpu-$CASE_NAME-quick"
  gpu_label="$RUN_LABEL_PREFIX-gpu-$CASE_NAME-quick"
  require_executable "$CPU_ELEGANT"
  require_executable "$GPU_ELEGANT"
  run_cmd "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --output "$OUTPUT_ROOT" --elegant "$CPU_ELEGANT" --label "$cpu_label"
  run_cmd env ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
    "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --output "$OUTPUT_ROOT" --elegant "$GPU_ELEGANT" --label "$gpu_label"
  run_compare "$cpu_label" "$gpu_label"
fi

if [ "$DO_VERIFY_QUICK" -eq 1 ]; then
  verify_label="$RUN_LABEL_PREFIX-gpu-$CASE_NAME-verify"
  require_executable "$GPU_VERIFY_ELEGANT"
  run_cmd env ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_VERIFY=1 ELEGANT_GPU_MIN_PARTICLES=1 \
    "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --output "$OUTPUT_ROOT" --elegant "$GPU_VERIFY_ELEGANT" --label "$verify_label"
fi

if [ "$DO_BASELINE" -eq 1 ]; then
  cpu_label="$RUN_LABEL_PREFIX-cpu-$CASE_NAME-baseline"
  gpu_label="$RUN_LABEL_PREFIX-gpu-$CASE_NAME-baseline"
  require_executable "$CPU_ELEGANT"
  require_executable "$GPU_ELEGANT"
  run_cmd "$SCRIPT_DIR/run_benchmarks.sh" --baseline --target-seconds "$TARGET_SECONDS" --case "$CASE_NAME" --output "$OUTPUT_ROOT" --elegant "$CPU_ELEGANT" --label "$cpu_label"
  run_cmd env ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
    "$SCRIPT_DIR/run_benchmarks.sh" --baseline --target-seconds "$TARGET_SECONDS" --case "$CASE_NAME" --output "$OUTPUT_ROOT" --elegant "$GPU_ELEGANT" --label "$gpu_label"
  echo "baseline labels may use different autoscaled pass counts; use --quick or an explicit common-pass run for SDDS comparison"
fi

if [ "$DO_MPI_SMOKE" -eq 1 ]; then
  cpu_label="$RUN_LABEL_PREFIX-cpu-pelegant-$CASE_NAME"
  gpu_label="$RUN_LABEL_PREFIX-gpu-pelegant-$CASE_NAME"
  require_executable "$CPU_PELEGANT"
  require_executable "$GPU_PELEGANT"
  run_cmd "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --output "$OUTPUT_ROOT" --mpi-ranks "$MPI_RANKS" --elegant "$CPU_PELEGANT" --label "$cpu_label"
  run_cmd env ELEGANT_GPU_MODE=auto ELEGANT_GPU_VERBOSE=1 ELEGANT_GPU_MIN_PARTICLES=1 \
    "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --output "$OUTPUT_ROOT" --mpi-ranks "$MPI_RANKS" --elegant "$GPU_PELEGANT" --label "$gpu_label"
  run_compare "$cpu_label" "$gpu_label"
fi
