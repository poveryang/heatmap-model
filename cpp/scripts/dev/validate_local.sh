#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"

TARGET_PLATFORM="x86"
CONTEXT="cpu"
PRECISION="uint8"
MODEL_PATH=""
IMAGE_PATH=""
BUILD_DIR="$CPP_DIR/build/x86"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
CMAKE_ARGS=()

usage() {
  cat <<'USAGE'
Usage: cpp/scripts/dev/validate_local.sh [options] [-- extra cmake args]

Options:
  --platform VALUE   TARGET_PLATFORM for CMake. Default: x86
  --context VALUE    Runtime context for HMAP-TEST. Default: cpu
  --precision VALUE  Runtime precision for HMAP-TEST. Default: uint8
  --model PATH       tmfile path (required)
  --image PATH       image path (required)
  --build-dir PATH   CMake build dir. Default: cpp/build/x86
  --jobs VALUE       Build parallelism. Default: detected CPU count
  -h, --help         Show this help.
USAGE
}

resolve_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s\n' "$PWD/$raw"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      TARGET_PLATFORM="$2"
      BUILD_DIR="$CPP_DIR/build/$2"
      shift 2
      ;;
    --context)
      CONTEXT="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --image)
      IMAGE_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$(resolve_path "$2")"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --)
      shift
      CMAKE_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$IMAGE_PATH" ]]; then
  echo "--model and --image are required." >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image file not found: $IMAGE_PATH" >&2
  exit 1
fi

cmake -S "$CPP_DIR" -B "$BUILD_DIR" -DTARGET_PLATFORM="$TARGET_PLATFORM" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target HMAP-TEST -j "$JOBS"

(
  cd "$BUILD_DIR"
  "$BUILD_DIR/HMAP-TEST" -c "$CONTEXT" -p "$PRECISION" -m "$MODEL_PATH" -i "$IMAGE_PATH"
)

echo "Generated outputs in $BUILD_DIR: heatmap.png, blend.png"
