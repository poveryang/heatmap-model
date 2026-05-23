#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"

PROFILE_PATH="$CPP_DIR/profiles/imx8plus.env"
ARGS=()

usage() {
  cat <<'USAGE'
Usage: cpp/scripts/convert/quantize_uint8.sh [options]

Options:
  --profile PATH  Conversion profile. Default: cpp/profiles/imx8plus.env
  --fp32 PATH     Input fp32 tmfile path. Default: profile DEFAULT_FP32_TMFILE
  --out PATH      Output uint8 tmfile path. Default: profile DEFAULT_UINT8_TMFILE
  --calib PATH    Calibration image directory (required unless DEFAULT_CALIB_DATASET in profile)
  --tool PATH      Tengine quant_tool_uint8 path. Default: profile TENGINE_QUANT_TOOL
  --shape VALUE    Input shape passed to -g. Default: profile TENGINE_INPUT_SHAPE
  --mean VALUE     Values passed to -w. Default: profile TENGINE_MEAN_VALUES
  --scale VALUE    Values passed to -s. Default: profile TENGINE_SCALE_VALUES
  --scale-file PATH
                  External calibration scale table passed to -f, for example
                  MQBench *_for_tengine.scale.
  --threads VALUE  Thread count passed to -t. Default: profile TENGINE_QUANT_THREADS
  --algo VALUE     Algorithm id passed to -a. Default: profile TENGINE_QUANT_ALGORITHM
  -h, --help       Show this help.
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

resolve_repo_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s\n' "$REPO_ROOT/$raw"
  fi
}

to_docker_path() {
  local host_path="$1"
  local workdir="${MODEL_CONVERT_DOCKER_WORKDIR:-${TOOLCHAIN_DOCKER_WORKDIR:-/workspace}}"
  case "$host_path" in
    "$REPO_ROOT")
      printf '%s\n' "$workdir"
      ;;
    "$REPO_ROOT"/*)
      printf '%s/%s\n' "$workdir" "${host_path#"$REPO_ROOT"/}"
      ;;
    *)
      echo "Docker runner expects model paths under repo: $host_path" >&2
      exit 1
      ;;
  esac
}

to_docker_path_if_in_repo() {
  local host_path="$1"
  local workdir="${MODEL_CONVERT_DOCKER_WORKDIR:-${TOOLCHAIN_DOCKER_WORKDIR:-/workspace}}"
  case "$host_path" in
    "$REPO_ROOT")
      printf '%s\n' "$workdir"
      ;;
    "$REPO_ROOT"/*)
      printf '%s/%s\n' "$workdir" "${host_path#"$REPO_ROOT"/}"
      ;;
    *)
      return 1
      ;;
  esac
}

collect_quant_scale() {
  local out_dir
  out_dir="$(dirname "$OUT_TMFILE")"
  local scale_src=""
  if [[ -f "$REPO_ROOT/table_kl.scale" ]]; then
    scale_src="$REPO_ROOT/table_kl.scale"
  elif [[ -f "$PWD/table_kl.scale" ]]; then
    scale_src="$PWD/table_kl.scale"
  fi

  if [[ -n "$scale_src" ]]; then
    mkdir -p "$out_dir"
    mv "$scale_src" "$out_dir/table_kl.scale"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --no-profile)
      PROFILE_PATH=""
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROFILE_PATH" ]]; then
  if [[ ! -f "$PROFILE_PATH" ]]; then
    echo "Model convert profile not found: $PROFILE_PATH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$PROFILE_PATH"
fi

MODEL_CONVERT_RUNNER="${MODEL_CONVERT_RUNNER:-${TOOLCHAIN_RUNNER:-docker}}"
MODEL_CONVERT_DOCKER_IMAGE="${MODEL_CONVERT_DOCKER_IMAGE:-${TOOLCHAIN_DOCKER_IMAGE:-}}"
MODEL_CONVERT_DOCKER_PLATFORM="${MODEL_CONVERT_DOCKER_PLATFORM:-${TOOLCHAIN_DOCKER_PLATFORM:-}}"
MODEL_CONVERT_DOCKER_WORKDIR="${MODEL_CONVERT_DOCKER_WORKDIR:-${TOOLCHAIN_DOCKER_WORKDIR:-/workspace}}"
MODEL_CONVERT_DOCKER_ARGS="${MODEL_CONVERT_DOCKER_ARGS:-${TOOLCHAIN_DOCKER_ARGS:-}}"
MODEL_CONVERT_DOCKER_CALIB_DIR="${MODEL_CONVERT_DOCKER_CALIB_DIR:-${TOOLCHAIN_DOCKER_CALIB_DIR:-/calib}}"

QUANT_TOOL="${TENGINE_QUANT_TOOL:-$HOME/CProjs/tengine-lite/build_quant/install/bin/quant_tool_uint8}"
FP32_TMFILE="$(resolve_repo_path "${DEFAULT_FP32_TMFILE:-cpp/artifacts/tmfile/model-fp32.tmfile}")"
OUT_TMFILE="$(resolve_repo_path "${DEFAULT_UINT8_TMFILE:-cpp/artifacts/tmfile/model-uint8.tmfile}")"
CALIB_DATASET=""
INPUT_SHAPE="${TENGINE_INPUT_SHAPE:-1,400,640}"
MEAN_VALUES="${TENGINE_MEAN_VALUES:-110.3895,110.3895,110.3895}"
SCALE_VALUES="${TENGINE_SCALE_VALUES:-0.01669463,0.01669463,0.01669463}"
THREADS="${TENGINE_QUANT_THREADS:-64}"
ALGORITHM="${TENGINE_QUANT_ALGORITHM:-1}"
SCALE_FILE=""

set -- "${ARGS[@]}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp32)
      FP32_TMFILE="$(resolve_path "$2")"
      shift 2
      ;;
    --out)
      OUT_TMFILE="$(resolve_path "$2")"
      shift 2
      ;;
    --calib)
      CALIB_DATASET="$(resolve_path "$2")"
      shift 2
      ;;
    --tool)
      QUANT_TOOL="$(resolve_path "$2")"
      shift 2
      ;;
    --shape)
      INPUT_SHAPE="$2"
      shift 2
      ;;
    --mean)
      MEAN_VALUES="$2"
      shift 2
      ;;
    --scale)
      SCALE_VALUES="$2"
      shift 2
      ;;
    --scale-file)
      SCALE_FILE="$(resolve_path "$2")"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --algo)
      ALGORITHM="$2"
      shift 2
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

if [[ -z "$CALIB_DATASET" && -n "${DEFAULT_CALIB_DATASET:-}" ]]; then
  CALIB_DATASET="$(resolve_repo_path "$DEFAULT_CALIB_DATASET")"
fi

if [[ -z "$CALIB_DATASET" ]]; then
  echo "--calib is required (or set DEFAULT_CALIB_DATASET in profile)." >&2
  exit 1
fi

if [[ ! -f "$FP32_TMFILE" ]]; then
  echo "fp32 tmfile not found: $FP32_TMFILE" >&2
  exit 1
fi

if [[ ! -d "$CALIB_DATASET" ]]; then
  echo "Calibration dataset directory not found: $CALIB_DATASET" >&2
  exit 1
fi

if [[ -n "$SCALE_FILE" && ! -f "$SCALE_FILE" ]]; then
  echo "Scale file not found: $SCALE_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_TMFILE")"

if [[ "$MODEL_CONVERT_RUNNER" == "host" ]]; then
  if [[ ! -x "$QUANT_TOOL" ]]; then
    echo "Tengine quant_tool_uint8 is not executable: $QUANT_TOOL" >&2
    echo "Set TENGINE_QUANT_TOOL, pass --tool, or use a Docker profile." >&2
    exit 1
  fi
  QUANT_ARGS=(
    -m "$FP32_TMFILE"
    -o "$OUT_TMFILE"
    -i "$CALIB_DATASET"
    -g "$INPUT_SHAPE"
    -w "$MEAN_VALUES"
    -s "$SCALE_VALUES"
    -t "$THREADS"
    -a "$ALGORITHM"
  )
  if [[ -n "$SCALE_FILE" ]]; then
    QUANT_ARGS+=(-f "$SCALE_FILE")
  fi
  "$QUANT_TOOL" "${QUANT_ARGS[@]}"
elif [[ "$MODEL_CONVERT_RUNNER" == "docker" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found; install Docker or use a host profile." >&2
    exit 1
  fi
  if [[ -z "$MODEL_CONVERT_DOCKER_IMAGE" ]]; then
    echo "MODEL_CONVERT_DOCKER_IMAGE is required for Docker runner." >&2
    exit 1
  fi

  DOCKER_EXTRA_ARGS=()
  if [[ -n "$MODEL_CONVERT_DOCKER_ARGS" ]]; then
    read -r -a DOCKER_EXTRA_ARGS <<< "$MODEL_CONVERT_DOCKER_ARGS"
  fi

  DOCKER_MOUNTS=(-v "$REPO_ROOT:$MODEL_CONVERT_DOCKER_WORKDIR")
  if CALIB_DOCKER_PATH="$(to_docker_path_if_in_repo "$CALIB_DATASET")"; then
    :
  else
    CALIB_DOCKER_PATH="$MODEL_CONVERT_DOCKER_CALIB_DIR"
    DOCKER_MOUNTS+=(-v "$CALIB_DATASET:$CALIB_DOCKER_PATH:ro")
  fi

  DOCKER_CMD=(
    docker run --rm
    "${DOCKER_MOUNTS[@]}"
    -w "$MODEL_CONVERT_DOCKER_WORKDIR"
  )
  if [[ -n "$MODEL_CONVERT_DOCKER_PLATFORM" ]]; then
    DOCKER_CMD+=(--platform "$MODEL_CONVERT_DOCKER_PLATFORM")
  fi
  if [[ "${#DOCKER_EXTRA_ARGS[@]}" -gt 0 ]]; then
    DOCKER_CMD+=("${DOCKER_EXTRA_ARGS[@]}")
  fi
  DOCKER_CMD+=(
    "$MODEL_CONVERT_DOCKER_IMAGE" \
    "$QUANT_TOOL" \
    -m "$(to_docker_path "$FP32_TMFILE")" \
    -o "$(to_docker_path "$OUT_TMFILE")" \
    -i "$CALIB_DOCKER_PATH" \
    -g "$INPUT_SHAPE" \
    -w "$MEAN_VALUES" \
    -s "$SCALE_VALUES" \
    -t "$THREADS" \
    -a "$ALGORITHM"
  )
  if [[ -n "$SCALE_FILE" ]]; then
    DOCKER_CMD+=(-f "$(to_docker_path "$SCALE_FILE")")
  fi
  "${DOCKER_CMD[@]}"
else
  echo "Unsupported MODEL_CONVERT_RUNNER: $MODEL_CONVERT_RUNNER" >&2
  exit 1
fi

collect_quant_scale
echo "$OUT_TMFILE"
