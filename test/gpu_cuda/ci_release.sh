#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

JOBS=${JOBS:-4}
NVCC_PATH=${NVCC:-}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
CASE_NAME=${CASE_NAME:-matrix}
TIMING_CASE=${TIMING_CASE:-}
TARGET_SECONDS=${TARGET_SECONDS:-60}
TOLERANCE=${TOLERANCE:-1e-11}
MPI_RANKS=${MPI_RANKS:-2}
LABEL_PREFIX=${LABEL_PREFIX:-phase20-ci}
RUN_OUTPUT_ROOT=${RUN_OUTPUT_ROOT:-$SCRIPT_DIR/output}
ARTIFACT_DIR=${ARTIFACT_DIR:-$RUN_OUTPUT_ROOT/ci_artifacts/$LABEL_PREFIX}
REFERENCE_ELEGANT=${REFERENCE_ELEGANT:-}
REFERENCE_PELEGANT=${REFERENCE_PELEGANT:-}

DO_CPU_BUILD=0
DO_CUDA_COMPILE=0
DO_GPU_SMOKE=0
DO_TIMING=0
DO_PELEGANT_SMOKE=0
DO_RELEASE_INVARIANCE=0
DO_FALLBACK_REPORT=0
DO_ALL_AVAILABLE=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Stages:
  --cpu-build             Run ordinary CPU build stage
  --cuda-compile          Run CUDA and GPU_VERIFY compile stages
  --gpu-smoke             Run CPU/GPU quick and GPU_VERIFY quick smoke
  --timing                Run bounded CPU/GPU timing baseline and report
  --pelegant-smoke        Run fixed-rank CPU/GPU Pelegant smoke
  --release-invariance    Run release-invariance layout/dictionary checks
  --fallback-report       Generate CUDA fallback/synchronization summary
  --all-available         Run stages that are supported by local tools

Controls:
  --case NAME             Correctness smoke case (default: matrix)
  --timing-case NAME      Timing baseline case (default: --case)
  --target-seconds N      CPU-target timing goal (default: 60)
  --tolerance N           SDDS comparison tolerance (default: 1e-11)
  --mpi-ranks N           MPI ranks for Pelegant smoke (default: 2)
  --jobs N                Parallel make jobs (default: 4)
  --nvcc PATH             CUDA compiler path
  --cuda-arch ARCH        CUDA architecture, e.g. sm_86 (default: sm_86)
  --label-prefix NAME     Output label prefix (default: phase20-ci)
  --output-root DIR       Benchmark output root (default: test/gpu_cuda/output)
  --artifact-dir DIR      Artifact directory (default: output/ci_artifacts/<label>)
  --reference-elegant PATH
  --reference-pelegant PATH
  --dry-run               Print commands without executing them
  -h, --help              Show this help

If no stage is selected, --cpu-build and --fallback-report are used.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-build)
      DO_CPU_BUILD=1
      ;;
    --cuda-compile)
      DO_CUDA_COMPILE=1
      ;;
    --gpu-smoke)
      DO_GPU_SMOKE=1
      ;;
    --timing)
      DO_TIMING=1
      ;;
    --pelegant-smoke)
      DO_PELEGANT_SMOKE=1
      ;;
    --release-invariance)
      DO_RELEASE_INVARIANCE=1
      ;;
    --fallback-report)
      DO_FALLBACK_REPORT=1
      ;;
    --all-available)
      DO_ALL_AVAILABLE=1
      ;;
    --case)
      CASE_NAME=$2
      shift
      ;;
    --timing-case)
      TIMING_CASE=$2
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
      LABEL_PREFIX=$2
      shift
      ;;
    --output-root)
      RUN_OUTPUT_ROOT=$2
      shift
      ;;
    --artifact-dir)
      ARTIFACT_DIR=$2
      shift
      ;;
    --reference-elegant)
      REFERENCE_ELEGANT=$2
      shift
      ;;
    --reference-pelegant)
      REFERENCE_PELEGANT=$2
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

LOG_DIR="$ARTIFACT_DIR/logs"
REPORT_DIR="$ARTIFACT_DIR/reports"
BENCHMARK_ARTIFACT_DIR="$ARTIFACT_DIR/benchmark-output"

if [ -z "$TIMING_CASE" ]; then
  TIMING_CASE=$CASE_NAME
fi

quote_cmd() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_logged() {
  local stage=$1
  shift
  mkdir -p "$LOG_DIR"
  quote_cmd "$@" | tee "$LOG_DIR/$stage.log"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@" 2>&1 | tee -a "$LOG_DIR/$stage.log"
  fi
}

find_nvcc_quiet() {
  local candidate
  if [ -n "$NVCC_PATH" ]; then
    [ -x "$NVCC_PATH" ] && printf '%s\n' "$NVCC_PATH"
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
  return 1
}

find_nvcc() {
  if ! find_nvcc_quiet; then
    echo "unable to find nvcc; use --nvcc PATH or set NVCC" >&2
    exit 1
  fi
}

if [ "$DO_ALL_AVAILABLE" -eq 1 ]; then
  DO_CPU_BUILD=1
  if find_nvcc_quiet >/dev/null 2>&1; then
    DO_CUDA_COMPILE=1
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DO_GPU_SMOKE=1
    if command -v mpirun >/dev/null 2>&1; then
      DO_PELEGANT_SMOKE=1
    fi
  fi
  DO_FALLBACK_REPORT=1
fi

if [ "$DO_CPU_BUILD$DO_CUDA_COMPILE$DO_GPU_SMOKE$DO_TIMING$DO_PELEGANT_SMOKE$DO_RELEASE_INVARIANCE$DO_FALLBACK_REPORT" = "0000000" ]; then
  DO_CPU_BUILD=1
  DO_FALLBACK_REPORT=1
fi

run_cpu_build() {
  run_logged cpu-build "$SCRIPT_DIR/ci_smoke.sh" --cpu-build --jobs "$JOBS" --output "$RUN_OUTPUT_ROOT" --label-prefix "$LABEL_PREFIX"
}

run_cuda_compile() {
  NVCC_PATH=$(find_nvcc)
  run_logged cuda-compile "$SCRIPT_DIR/ci_smoke.sh" \
    --cuda-build --cuda-verify-build \
    --output "$RUN_OUTPUT_ROOT" \
    --jobs "$JOBS" --nvcc "$NVCC_PATH" --cuda-arch "$CUDA_ARCH" \
    --label-prefix "$LABEL_PREFIX"
}

run_gpu_smoke() {
  run_logged gpu-smoke "$SCRIPT_DIR/ci_smoke.sh" \
    --quick --verify-quick --case "$CASE_NAME" \
    --output "$RUN_OUTPUT_ROOT" \
    --tolerance "$TOLERANCE" --label-prefix "$LABEL_PREFIX"
}

run_pelegant_smoke() {
  run_logged pelegant-smoke "$SCRIPT_DIR/ci_smoke.sh" \
    --mpi-smoke --case "$CASE_NAME" --mpi-ranks "$MPI_RANKS" \
    --output "$RUN_OUTPUT_ROOT" \
    --tolerance "$TOLERANCE" --label-prefix "$LABEL_PREFIX"
}

run_timing() {
  run_logged timing "$SCRIPT_DIR/ci_smoke.sh" \
    --baseline --case "$TIMING_CASE" --target-seconds "$TARGET_SECONDS" \
    --output "$RUN_OUTPUT_ROOT" \
    --label-prefix "$LABEL_PREFIX"
  mkdir -p "$REPORT_DIR"
  run_logged timing-report python3 "$SCRIPT_DIR/report_benchmarks.py" \
    --cpu-manifest "$RUN_OUTPUT_ROOT/$LABEL_PREFIX-cpu-$TIMING_CASE-baseline/manifest.tsv" \
    --gpu-manifest "$RUN_OUTPUT_ROOT/$LABEL_PREFIX-gpu-$TIMING_CASE-baseline/manifest.tsv" \
    --title "CI Timing: $TIMING_CASE" \
    --gpu-binary "$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-elegant" \
    --metadata "Label prefix=$LABEL_PREFIX" \
    --metadata "Target seconds=$TARGET_SECONDS" \
    --metadata "Tolerance=$TOLERANCE" \
    --output "$REPORT_DIR/$LABEL_PREFIX-$TIMING_CASE-timing.md"
}

run_release_invariance() {
  local args=(
    --candidate-elegant "$REPO_ROOT/bin/Linux-x86_64/elegant"
    --candidate-pelegant "$REPO_ROOT/bin/Linux-x86_64/Pelegant"
    --gpu-elegant "$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-elegant"
    --gpu-pelegant "$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-Pelegant"
    --require-cuda-layout
    --case "$CASE_NAME"
    --mpi-ranks "$MPI_RANKS"
    --tolerance "$TOLERANCE"
    --output "$ARTIFACT_DIR/release_invariance"
  )
  if [ -n "$REFERENCE_ELEGANT" ]; then
    args+=(--reference-elegant "$REFERENCE_ELEGANT")
  else
    args+=(--skip-serial)
  fi
  if [ -n "$REFERENCE_PELEGANT" ]; then
    args+=(--reference-pelegant "$REFERENCE_PELEGANT")
  else
    args+=(--skip-mpi)
  fi
  run_logged release-invariance "$SCRIPT_DIR/release_invariance.sh" "${args[@]}"
}

run_fallback_report() {
  mkdir -p "$REPORT_DIR"
  run_logged fallback-report python3 "$SCRIPT_DIR/summarize_fallbacks.py" \
    --output-root "$RUN_OUTPUT_ROOT" \
    --label-prefix "$LABEL_PREFIX" \
    --output "$REPORT_DIR/$LABEL_PREFIX-fallbacks.md" \
    --tsv "$REPORT_DIR/$LABEL_PREFIX-fallbacks.tsv"
}

collect_artifacts() {
  mkdir -p "$BENCHMARK_ARTIFACT_DIR" "$REPORT_DIR"
  local dir
  for dir in "$RUN_OUTPUT_ROOT"/"$LABEL_PREFIX"*; do
    if [ -d "$dir" ]; then
      cp -a "$dir" "$BENCHMARK_ARTIFACT_DIR/"
    fi
  done
  cp "$SCRIPT_DIR/release_notes_template.md" "$ARTIFACT_DIR/release_notes_template.md"
  find "$ARTIFACT_DIR" -type f | sort > "$ARTIFACT_DIR/artifacts.txt"
}

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$REPORT_DIR"

if [ "$DO_CPU_BUILD" -eq 1 ]; then
  run_cpu_build
fi
if [ "$DO_CUDA_COMPILE" -eq 1 ]; then
  run_cuda_compile
fi
if [ "$DO_GPU_SMOKE" -eq 1 ]; then
  run_gpu_smoke
fi
if [ "$DO_TIMING" -eq 1 ]; then
  run_timing
fi
if [ "$DO_PELEGANT_SMOKE" -eq 1 ]; then
  run_pelegant_smoke
fi
if [ "$DO_RELEASE_INVARIANCE" -eq 1 ]; then
  run_release_invariance
fi
if [ "$DO_FALLBACK_REPORT" -eq 1 ]; then
  run_fallback_report
fi

if [ "$DRY_RUN" -eq 0 ]; then
  collect_artifacts
else
  quote_cmd collect_artifacts "$ARTIFACT_DIR"
fi

echo "CI/release artifacts: $ARTIFACT_DIR"
