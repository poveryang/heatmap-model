#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"

PROFILE_PATH="$CPP_DIR/profiles/imx8plus.env"
ARGS=()

usage() {
  cat <<'USAGE'
Usage: cpp/scripts/convert/onnx_to_tmfile.sh [options]

Options:
  --profile PATH  Conversion profile. Default: cpp/profiles/imx8plus.env
  --onnx PATH     Input ONNX path (required unless DEFAULT_ONNX_PATH in profile)
  --out PATH      Output fp32 tmfile path. Default: profile DEFAULT_FP32_TMFILE
  --tool PATH     Tengine convert_tool path. Default: profile TENGINE_CONVERT_TOOL
  -h, --help      Show this help.
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
      echo "Docker runner expects paths under repo: $host_path" >&2
      exit 1
      ;;
  esac
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

CONVERT_TOOL="${TENGINE_CONVERT_TOOL:-$HOME/CProjs/tengine-lite/build_cvt_tool/install/bin/convert_tool}"
ONNX_PATH=""
OUT_PATH="$(resolve_repo_path "${DEFAULT_FP32_TMFILE:-cpp/artifacts/tmfile/model-fp32.tmfile}")"

set -- "${ARGS[@]}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)
      ONNX_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --out)
      OUT_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --tool)
      CONVERT_TOOL="$(resolve_path "$2")"
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

if [[ -z "$ONNX_PATH" && -n "${DEFAULT_ONNX_PATH:-}" ]]; then
  ONNX_PATH="$(resolve_repo_path "$DEFAULT_ONNX_PATH")"
fi

if [[ -z "$ONNX_PATH" ]]; then
  echo "--onnx is required (or set DEFAULT_ONNX_PATH in profile)." >&2
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ONNX file not found: $ONNX_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_PATH")"

if [[ "$MODEL_CONVERT_RUNNER" == "host" ]]; then
  if [[ ! -x "$CONVERT_TOOL" ]]; then
    echo "Tengine convert_tool is not executable: $CONVERT_TOOL" >&2
    echo "Set TENGINE_CONVERT_TOOL, pass --tool, or use a Docker profile." >&2
    exit 1
  fi
  "$CONVERT_TOOL" -f onnx -m "$ONNX_PATH" -o "$OUT_PATH"
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
  DOCKER_CMD=(
    docker run --rm
    -v "$REPO_ROOT:$MODEL_CONVERT_DOCKER_WORKDIR"
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
    "$CONVERT_TOOL" \
    -f onnx \
    -m "$(to_docker_path "$ONNX_PATH")" \
    -o "$(to_docker_path "$OUT_PATH")"
  )
  "${DOCKER_CMD[@]}"
else
  echo "Unsupported MODEL_CONVERT_RUNNER: $MODEL_CONVERT_RUNNER" >&2
  exit 1
fi

echo "$OUT_PATH"
