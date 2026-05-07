#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

CASE=matrix
RANKS=2
PARTICLES=20000
PASSES=20
TOLERANCE=1e-11
LABEL_PREFIX=phase19-pelegant-single-gpu
CPU_PELEGANT=${CPU_PELEGANT:-"$REPO_ROOT/bin/Linux-x86_64/Pelegant"}
GPU_PELEGANT=${GPU_PELEGANT:-"$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-Pelegant"}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --case NAME              Benchmark case name (default: $CASE)
  --ranks N                MPI ranks to run (default: $RANKS)
  --particles N            Particles per bunch (default: $PARTICLES)
  --passes N               Passes (default: $PASSES)
  --tolerance VALUE        SDDS comparison tolerance (default: $TOLERANCE)
  --label-prefix NAME      Output label prefix (default: $LABEL_PREFIX)
  --cpu-pelegant PATH      CPU Pelegant binary (default: $CPU_PELEGANT)
  --gpu-pelegant PATH      GPU Pelegant binary (default: $GPU_PELEGANT)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      CASE=$2
      shift 2
      ;;
    --ranks)
      RANKS=$2
      shift 2
      ;;
    --particles)
      PARTICLES=$2
      shift 2
      ;;
    --passes)
      PASSES=$2
      shift 2
      ;;
    --tolerance)
      TOLERANCE=$2
      shift 2
      ;;
    --label-prefix)
      LABEL_PREFIX=$2
      shift 2
      ;;
    --cpu-pelegant)
      CPU_PELEGANT=$2
      shift 2
      ;;
    --gpu-pelegant)
      GPU_PELEGANT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$CPU_PELEGANT" ]]; then
  echo "CPU Pelegant binary is not executable: $CPU_PELEGANT" >&2
  exit 1
fi

if [[ ! -x "$GPU_PELEGANT" ]]; then
  echo "GPU Pelegant binary is not executable: $GPU_PELEGANT" >&2
  exit 1
fi

CPU_LABEL="${LABEL_PREFIX}-cpu-r${RANKS}-${CASE}"
GPU_LABEL="${LABEL_PREFIX}-gpu-r${RANKS}-${CASE}"
CPU_DIR="$SCRIPT_DIR/output/$CPU_LABEL/$CASE"
GPU_DIR="$SCRIPT_DIR/output/$GPU_LABEL/$CASE"

COMMON_ARGS=(
  --quick
  --case "$CASE"
  --particles "$PARTICLES"
  --passes "$PASSES"
  --mpi-ranks "$RANKS"
)

echo "Running CPU Pelegant: $CPU_PELEGANT"
"$SCRIPT_DIR/run_benchmarks.sh" "${COMMON_ARGS[@]}" --elegant "$CPU_PELEGANT" --label "$CPU_LABEL"

echo "Running GPU Pelegant: $GPU_PELEGANT"
env \
  ELEGANT_GPU_MODE="${ELEGANT_GPU_MODE:-required}" \
  ELEGANT_GPU_VERBOSE="${ELEGANT_GPU_VERBOSE:-1}" \
  ELEGANT_GPU_MIN_PARTICLES="${ELEGANT_GPU_MIN_PARTICLES:-1}" \
  "$SCRIPT_DIR/run_benchmarks.sh" "${COMMON_ARGS[@]}" --elegant "$GPU_PELEGANT" --label "$GPU_LABEL"

echo "Comparing CPU and GPU SDDS output at tolerance $TOLERANCE"
python3 "$SCRIPT_DIR/compare_sdds.py" "$CPU_DIR" "$GPU_DIR" --tolerance "$TOLERANCE"

echo "CPU manifest: $SCRIPT_DIR/output/$CPU_LABEL/manifest.tsv"
echo "GPU manifest: $SCRIPT_DIR/output/$GPU_LABEL/manifest.tsv"
