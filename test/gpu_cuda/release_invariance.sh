#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

CANDIDATE_ELEGANT=${CANDIDATE_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64/elegant}
REFERENCE_ELEGANT=${REFERENCE_ELEGANT:-}
CANDIDATE_PELEGANT=${CANDIDATE_PELEGANT:-$REPO_ROOT/bin/Linux-x86_64/Pelegant}
REFERENCE_PELEGANT=${REFERENCE_PELEGANT:-}
GPU_ELEGANT=${GPU_ELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-elegant}
GPU_PELEGANT=${GPU_PELEGANT:-$REPO_ROOT/bin/Linux-x86_64-gpu/gpu-Pelegant}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/output/release_invariance}
CASE_NAME=${CASE_NAME:-matrix}
MPI_RANKS=${MPI_RANKS:-2}
TOLERANCE=${TOLERANCE:-1e-11}
RPN_DEFNS_FILE=${RPN_DEFNS:-}

DO_LAYOUT=1
DO_DICTIONARY=1
DO_SERIAL=1
DO_MPI=1
DO_CLEAN_CHECK=0
REQUIRE_CUDA_LAYOUT=0
REQUIRE_MPI_LAYOUT=0
ALLOW_DICTIONARY_DIFF=0
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Phase 10 release-hardening checks for CPU invariance and CUDA layout.

Reference/candidate binaries:
  --candidate-elegant PATH    Candidate CPU elegant (default: bin/Linux-x86_64/elegant)
  --reference-elegant PATH    Clean-baseline CPU elegant to compare against
  --candidate-pelegant PATH   Candidate CPU Pelegant (default: bin/Linux-x86_64/Pelegant)
  --reference-pelegant PATH   Clean-baseline CPU Pelegant to compare against
  --gpu-elegant PATH          CUDA serial binary for layout checks
  --gpu-pelegant PATH         CUDA Pelegant binary for layout checks

Checks:
  --skip-layout               Skip binary layout checks
  --skip-dictionary           Skip dictionary generation and metadata checks
  --skip-serial               Skip serial tracking output comparison
  --skip-mpi                  Skip MPI tracking output comparison
  --clean-check               Run make CUDA_AUTO=0 clean and make clean, then verify object dirs are gone
  --require-cuda-layout       Fail if gpu-elegant or gpu-Pelegant is missing
  --require-mpi-layout        Fail if candidate Pelegant is missing
  --allow-dictionary-diff     Continue after printing SDDS/LaTeX dictionary diffs

Controls:
  --case NAME                 Benchmark case for tracking comparisons (default: matrix)
  --mpi-ranks N               MPI ranks for Pelegant comparison (default: 2)
  --tolerance N               SDDS comparison tolerance (default: 1e-11)
  --rpn-defns PATH            RPN definitions file
  --output DIR                Output directory (default: test/gpu_cuda/output/release_invariance)
  --dry-run                   Print commands without executing them
  -h, --help                  Show this help

Serial and MPI tracking comparisons require reference binaries.  Dictionary
metadata checks run on the candidate CPU elegant binary and compare SDDS/LaTeX
dictionary files only when --reference-elegant is provided.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --candidate-elegant)
      CANDIDATE_ELEGANT=$2
      shift
      ;;
    --reference-elegant)
      REFERENCE_ELEGANT=$2
      shift
      ;;
    --candidate-pelegant)
      CANDIDATE_PELEGANT=$2
      shift
      ;;
    --reference-pelegant)
      REFERENCE_PELEGANT=$2
      shift
      ;;
    --gpu-elegant)
      GPU_ELEGANT=$2
      shift
      ;;
    --gpu-pelegant)
      GPU_PELEGANT=$2
      shift
      ;;
    --skip-layout)
      DO_LAYOUT=0
      ;;
    --skip-dictionary)
      DO_DICTIONARY=0
      ;;
    --skip-serial)
      DO_SERIAL=0
      ;;
    --skip-mpi)
      DO_MPI=0
      ;;
    --clean-check)
      DO_CLEAN_CHECK=1
      ;;
    --require-cuda-layout)
      REQUIRE_CUDA_LAYOUT=1
      ;;
    --require-mpi-layout)
      REQUIRE_MPI_LAYOUT=1
      ;;
    --allow-dictionary-diff)
      ALLOW_DICTIONARY_DIFF=1
      ;;
    --case)
      CASE_NAME=$2
      shift
      ;;
    --mpi-ranks)
      MPI_RANKS=$2
      shift
      ;;
    --tolerance)
      TOLERANCE=$2
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

run_chdir_cmd() {
  local dir=$1
  shift
  printf '+ cd %q &&' "$dir"
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    (cd "$dir" && "$@")
  fi
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
  local candidate
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

fail_missing_executable() {
  local label=$1
  local path=$2
  echo "missing executable for $label: $path" >&2
  return 1
}

require_executable() {
  local label=$1
  local path=$2
  if [ "$DRY_RUN" -eq 1 ]; then
    return
  fi
  if [ ! -x "$path" ]; then
    fail_missing_executable "$label" "$path"
  fi
}

check_optional_executable() {
  local label=$1
  local path=$2
  local required=$3
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN layout check: $label -> $path"
    return
  fi
  if [ -x "$path" ]; then
    echo "PASS layout: $label -> $path"
  elif [ "$required" -eq 1 ]; then
    fail_missing_executable "$label" "$path"
  else
    echo "SKIP layout: $label not present at $path"
  fi
}

run_dictionary() {
  local binary=$1
  local label=$2
  local destination="$OUTPUT_ROOT/$label/dictionary"
  mkdir -p "$destination"
  run_chdir_cmd "$destination" "$binary" "$SCRIPT_DIR/release_checks/print_dictionary.ele" "-rpnDefns=$RPN_DEFNS_FILE"
}

compare_dictionary_files() {
  local reference="$OUTPUT_ROOT/reference/dictionary"
  local candidate="$OUTPUT_ROOT/candidate/dictionary"
  compare_dictionary_file "$reference/dictionary.sdds" "$candidate/dictionary.sdds"
  compare_dictionary_file "$reference/dictionary.tex" "$candidate/dictionary.tex"
}

compare_dictionary_file() {
  local reference=$1
  local candidate=$2
  local status
  quote_cmd diff -u "$reference" "$candidate"
  if [ "$DRY_RUN" -eq 1 ]; then
    return
  fi
  set +e
  diff -u "$reference" "$candidate"
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    return
  fi
  if [ "$ALLOW_DICTIONARY_DIFF" -eq 1 ]; then
    echo "ALLOW dictionary diff: $reference vs $candidate"
    return
  fi
  return "$status"
}

run_dictionary_checks() {
  require_executable "candidate elegant" "$CANDIDATE_ELEGANT"
  if [ -n "$REFERENCE_ELEGANT" ]; then
    require_executable "reference elegant" "$REFERENCE_ELEGANT"
    run_dictionary "$REFERENCE_ELEGANT" reference
  fi
  run_dictionary "$CANDIDATE_ELEGANT" candidate
  run_cmd python3 "$SCRIPT_DIR/check_dictionary_gpu_support.py" "$OUTPUT_ROOT/candidate/dictionary/dictionary.sdds"
  if [ -n "$REFERENCE_ELEGANT" ]; then
    compare_dictionary_files
  else
    echo "SKIP dictionary diff: no --reference-elegant provided"
  fi
}

run_tracking_compare() {
  local reference_binary=$1
  local candidate_binary=$2
  local prefix=$3
  shift 3
  local extra_args=("$@")
  local reference_label="release-${prefix}-reference-${CASE_NAME}"
  local candidate_label="release-${prefix}-candidate-${CASE_NAME}"

  run_cmd "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --elegant "$reference_binary" \
    --output "$OUTPUT_ROOT" --label "$reference_label" "${extra_args[@]}"
  run_cmd "$SCRIPT_DIR/run_benchmarks.sh" --quick --case "$CASE_NAME" --elegant "$candidate_binary" \
    --output "$OUTPUT_ROOT" --label "$candidate_label" "${extra_args[@]}"
  run_cmd python3 "$SCRIPT_DIR/compare_sdds.py" \
    "$OUTPUT_ROOT/$reference_label/$CASE_NAME" \
    "$OUTPUT_ROOT/$candidate_label/$CASE_NAME" \
    --tolerance "$TOLERANCE"
}

run_serial_compare() {
  if [ -z "$REFERENCE_ELEGANT" ]; then
    echo "SKIP serial tracking comparison: no --reference-elegant provided"
    return
  fi
  require_executable "reference elegant" "$REFERENCE_ELEGANT"
  require_executable "candidate elegant" "$CANDIDATE_ELEGANT"
  run_tracking_compare "$REFERENCE_ELEGANT" "$CANDIDATE_ELEGANT" serial
}

run_mpi_compare() {
  if [ -z "$REFERENCE_PELEGANT" ]; then
    echo "SKIP MPI tracking comparison: no --reference-pelegant provided"
    return
  fi
  require_executable "reference Pelegant" "$REFERENCE_PELEGANT"
  require_executable "candidate Pelegant" "$CANDIDATE_PELEGANT"
  run_tracking_compare "$REFERENCE_PELEGANT" "$CANDIDATE_PELEGANT" mpi --mpi-ranks "$MPI_RANKS"
}

check_layout() {
  check_optional_executable "candidate elegant" "$CANDIDATE_ELEGANT" 1
  check_optional_executable "candidate Pelegant" "$CANDIDATE_PELEGANT" "$REQUIRE_MPI_LAYOUT"
  check_optional_executable "gpu-elegant" "$GPU_ELEGANT" "$REQUIRE_CUDA_LAYOUT"
  check_optional_executable "gpu-Pelegant" "$GPU_PELEGANT" "$REQUIRE_CUDA_LAYOUT"
}

verify_clean_dirs_removed() {
  local failed=0
  local dir
  for dir in \
    "$REPO_ROOT/src/O.$OS_NAME-$ARCH_NAME" \
    "$REPO_ROOT/src/O.$OS_NAME-$ARCH_NAME.gpu" \
    "$REPO_ROOT/src/O.$OS_NAME-$ARCH_NAME.gpu.verify" \
    "$REPO_ROOT/src/O.$OS_NAME-$ARCH_NAME.gpu.mpi" \
    "$REPO_ROOT/src/O.$OS_NAME-$ARCH_NAME.gpu.mpi.verify"
  do
    if [ -e "$dir" ]; then
      echo "FAIL clean: object directory remains: $dir" >&2
      failed=1
    else
      echo "PASS clean: removed $dir"
    fi
  done
  return "$failed"
}

run_clean_check() {
  echo "Running opt-in clean checks. This removes local build object directories."
  run_cmd make -C "$REPO_ROOT" CUDA_AUTO=0 clean
  verify_clean_dirs_removed
  run_cmd make -C "$REPO_ROOT" clean
  verify_clean_dirs_removed
}

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(absolute_path "$OUTPUT_ROOT")
if [ "$DO_DICTIONARY" -eq 1 ]; then
  RPN_DEFNS_FILE=$(find_rpn_defns)
fi
CANDIDATE_ELEGANT=$(absolute_path "$CANDIDATE_ELEGANT")
CANDIDATE_PELEGANT=$(absolute_path "$CANDIDATE_PELEGANT")
GPU_ELEGANT=$(absolute_path "$GPU_ELEGANT")
GPU_PELEGANT=$(absolute_path "$GPU_PELEGANT")
if [ -n "$REFERENCE_ELEGANT" ]; then
  REFERENCE_ELEGANT=$(absolute_path "$REFERENCE_ELEGANT")
fi
if [ -n "$REFERENCE_PELEGANT" ]; then
  REFERENCE_PELEGANT=$(absolute_path "$REFERENCE_PELEGANT")
fi

if [ "$DO_LAYOUT" -eq 1 ]; then
  check_layout
fi

if [ "$DO_DICTIONARY" -eq 1 ]; then
  run_dictionary_checks
fi

if [ "$DO_SERIAL" -eq 1 ]; then
  run_serial_compare
fi

if [ "$DO_MPI" -eq 1 ]; then
  run_mpi_compare
fi

if [ "$DO_CLEAN_CHECK" -eq 1 ]; then
  run_clean_check
fi

echo "release invariance checks completed"
