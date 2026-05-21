#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"

CONFIG_PATH="${HEATMAP_BOARD_CONFIG:-${HEATMAP_DEVICE_CONFIG:-$CPP_DIR/board.env}}"

BOARD_HOST="${BOARD_HOST:-${DEVICE_HOST:-10.80.184.167}}"
BOARD_SSH_PORT="${BOARD_SSH_PORT:-${DEVICE_SSH_PORT:-201}}"
BOARD_USER="${BOARD_USER:-${DEVICE_USER:-root}}"
BOARD_PASSWORD="${BOARD_PASSWORD:-${DEVICE_PASSWORD:-}}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/heatmap-model}"
BINARY_PATH="$CPP_DIR/build/imx8plus/HMAP-TEST"
MODEL_PATH="$CPP_DIR/artifacts/tmfile/model-uint8.tmfile"
IMAGE_PATH=""
OUT_DIR="$CPP_DIR/artifacts/imx8plus"
OPENCV_RUNTIME_DIR="$CPP_DIR/build/imx8plus/runtime/opencv"
GCC_RUNTIME_DIR="$CPP_DIR/build/imx8plus/runtime/gcc"
TENGINE_RUNTIME_DIR="$CPP_DIR/thirdparty/tengine/lib/aarch64"
TENGINE_RUNTIME_MODE="${TENGINE_RUNTIME_MODE:-minimal}"
CONTEXT="timvx"
PRECISION="uint8"
MIN_HOT_INTENSITY="${MIN_HOT_INTENSITY:-}"

usage() {
  cat <<'USAGE'
Usage: cpp/scripts/deploy/run_on_board.sh [options]

Options:
  --config PATH       Board config. Default: cpp/board.env or HEATMAP_BOARD_CONFIG
  --ip VALUE          Board IP. Default: BOARD_HOST or 10.80.184.167
  --port VALUE        SSH port. Default: BOARD_SSH_PORT or 201
  --user VALUE        SSH user. Default: BOARD_USER or root
  --password VALUE    SSH password. Empty means SSH key or interactive auth.
  --remote-dir PATH   Remote working directory. Default: /tmp/heatmap-model
  --binary PATH       Local HMAP-TEST binary. Default: cpp/build/imx8plus/HMAP-TEST
  --model PATH        Local tmfile. Default: cpp/artifacts/tmfile/model-uint8.tmfile
  --image PATH        Local input image (required unless set in board.env)
  --out-dir PATH      Local output directory. Default: cpp/artifacts/imx8plus
  --opencv-dir PATH   OpenCV aarch64 runtime. Default: cpp/build/imx8plus/runtime/opencv
  --gcc-dir PATH      GCC aarch64 runtime. Default: cpp/build/imx8plus/runtime/gcc
  --tengine-runtime VALUE
                      Tengine runtime deploy mode: all|minimal|none. Default: minimal
  --context VALUE     Runtime context. Default: timvx
  --precision VALUE   Runtime precision. Default: uint8
  --min-hot VALUE     Fail if max_hot intensity is below VALUE. Default: disabled
  -h, --help          Show this help.
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

preparse_config() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="$(resolve_path "$2")"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
}

load_config() {
  if [[ -f "$CONFIG_PATH" ]]; then
    :
  elif [[ -f "$CPP_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$CPP_DIR/$CONFIG_PATH"
  elif [[ -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
  fi
  if [[ -f "$CONFIG_PATH" ]]; then
    # shellcheck source=/dev/null
    set -a
    source "$CONFIG_PATH"
    set +a
  fi
  # Map legacy DEVICE_* names from older configs.
  BOARD_HOST="${BOARD_HOST:-${DEVICE_HOST:-}}"
  BOARD_SSH_PORT="${BOARD_SSH_PORT:-${DEVICE_SSH_PORT:-${DEVICE_PORT:-}}}"
  BOARD_USER="${BOARD_USER:-${DEVICE_USER:-}}"
  BOARD_PASSWORD="${BOARD_PASSWORD:-${DEVICE_PASSWORD:-}}"
  DEVICE_HOST="$BOARD_HOST"
  DEVICE_SSH_PORT="$BOARD_SSH_PORT"
  DEVICE_USER="$BOARD_USER"
  DEVICE_PASSWORD="$BOARD_PASSWORD"
}

run_with_password() {
  if [[ -z "$BOARD_PASSWORD" ]]; then
    "$@"
    return
  fi
  if ! command -v expect >/dev/null 2>&1; then
    echo "BOARD_PASSWORD is set but expect is not installed: brew install expect" >&2
    exit 1
  fi
  export HEATMAP_EXPECT_CMD
  HEATMAP_EXPECT_CMD="$(printf '%q ' "$@")"
  export HEATMAP_EXPECT_PASSWORD="$BOARD_PASSWORD"
  expect <<'EXPECT_EOF'
set timeout 120
log_user 1
spawn bash -c $env(HEATMAP_EXPECT_CMD)
expect {
  -re "(?i)password:" {
    send "$env(HEATMAP_EXPECT_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_EOF
}

remote_ssh() {
  run_with_password ssh -p "$DEVICE_SSH_PORT" "$DEVICE_USER@$DEVICE_HOST" "$1"
}

remote_ssh_tty() {
  run_with_password ssh -tt -p "$DEVICE_SSH_PORT" "$DEVICE_USER@$DEVICE_HOST" "$1"
}

remote_scp_to() {
  run_with_password scp -P "$DEVICE_SSH_PORT" -O "$1" "$DEVICE_USER@$DEVICE_HOST:$2"
}

remote_scp_from() {
  run_with_password scp -P "$DEVICE_SSH_PORT" -O "$DEVICE_USER@$DEVICE_HOST:$1" "$2"
}

remote_tar_upload() {
  local src_dir="$1"
  local dst_dir="$2"
  if [[ -z "$BOARD_PASSWORD" ]]; then
    tar -C "$src_dir" -cf - . | ssh -p "$DEVICE_SSH_PORT" "$DEVICE_USER@$DEVICE_HOST" "tar -C '$dst_dir' -xf -"
  else
    local archive
    archive="$(mktemp -t heatmap-device-libs.XXXXXX.tar)"
    tar -C "$src_dir" -cf "$archive" .
    remote_scp_to "$archive" "$dst_dir/$(basename "$archive")"
    remote_ssh "tar -C '$dst_dir' -xf '$dst_dir/$(basename "$archive")' && rm -f '$dst_dir/$(basename "$archive")'"
    rm -f "$archive"
  fi
}

upload_tengine_runtime() {
  case "$TENGINE_RUNTIME_MODE" in
    all)
      remote_tar_upload "$TENGINE_RUNTIME_DIR" "$REMOTE_DIR/lib"
      ;;
    minimal)
      local tmp_dir
      tmp_dir="$(mktemp -d -t heatmap-tengine-runtime.XXXXXX)"
      cp "$TENGINE_RUNTIME_DIR"/libtengine-lite.so* "$tmp_dir"/
      remote_tar_upload "$tmp_dir" "$REMOTE_DIR/lib"
      rm -rf "$tmp_dir"
      ;;
    none)
      ;;
    *)
      echo "Unsupported TENGINE_RUNTIME_MODE: $TENGINE_RUNTIME_MODE" >&2
      echo "Expected: all, minimal, or none." >&2
      exit 2
      ;;
  esac
}

preparse_config "$@"
load_config

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --ip)
      BOARD_HOST="$2"
      DEVICE_HOST="$2"
      shift 2
      ;;
    --port)
      BOARD_SSH_PORT="$2"
      DEVICE_SSH_PORT="$2"
      shift 2
      ;;
    --user)
      BOARD_USER="$2"
      DEVICE_USER="$2"
      shift 2
      ;;
    --password)
      BOARD_PASSWORD="$2"
      DEVICE_PASSWORD="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --binary)
      BINARY_PATH="$(resolve_path "$2")"
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
    --out-dir)
      OUT_DIR="$(resolve_path "$2")"
      shift 2
      ;;
    --opencv-dir)
      OPENCV_RUNTIME_DIR="$(resolve_path "$2")"
      shift 2
      ;;
    --gcc-dir)
      GCC_RUNTIME_DIR="$(resolve_path "$2")"
      shift 2
      ;;
    --tengine-runtime)
      TENGINE_RUNTIME_MODE="$2"
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
    --min-hot)
      MIN_HOT_INTENSITY="$2"
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

BINARY_PATH="$(resolve_repo_path "$BINARY_PATH")"
MODEL_PATH="$(resolve_repo_path "$MODEL_PATH")"
IMAGE_PATH="$(resolve_repo_path "$IMAGE_PATH")"
OUT_DIR="$(resolve_repo_path "$OUT_DIR")"
OPENCV_RUNTIME_DIR="$(resolve_repo_path "$OPENCV_RUNTIME_DIR")"
GCC_RUNTIME_DIR="$(resolve_repo_path "$GCC_RUNTIME_DIR")"
TENGINE_RUNTIME_DIR="$(resolve_repo_path "$TENGINE_RUNTIME_DIR")"

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "Executable HMAP-TEST not found: $BINARY_PATH" >&2
  echo "Build the imx8plus platform binary first, or pass --binary." >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$IMAGE_PATH" ]]; then
  echo "--image is required (or set IMAGE_PATH in board.env)." >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image file not found: $IMAGE_PATH" >&2
  exit 1
fi

if [[ ! -d "$OPENCV_RUNTIME_DIR" ]]; then
  echo "OpenCV runtime dir not found: $OPENCV_RUNTIME_DIR" >&2
  echo "Run cpp/scripts/build/cross_build.sh first, or pass --opencv-dir." >&2
  exit 1
fi

if [[ ! -d "$GCC_RUNTIME_DIR" ]]; then
  echo "GCC runtime dir not found: $GCC_RUNTIME_DIR" >&2
  echo "Run cpp/scripts/build/cross_build.sh first, or pass --gcc-dir." >&2
  exit 1
fi

if [[ "$TENGINE_RUNTIME_MODE" != "none" && ! -d "$TENGINE_RUNTIME_DIR" ]]; then
  echo "Tengine runtime dir not found: $TENGINE_RUNTIME_DIR" >&2
  exit 1
fi

REMOTE_MODEL="$REMOTE_DIR/model/$(basename "$MODEL_PATH")"
REMOTE_IMAGE="$REMOTE_DIR/input/$(basename "$IMAGE_PATH")"
REMOTE_BINARY="$REMOTE_DIR/HMAP-TEST"
mkdir -p "$OUT_DIR"

echo "board: $BOARD_USER@$BOARD_HOST:$BOARD_SSH_PORT"
echo "remote: $REMOTE_DIR"

remote_ssh "rm -rf '$REMOTE_DIR/lib' && mkdir -p '$REMOTE_DIR/bin' '$REMOTE_DIR/lib' '$REMOTE_DIR/model' '$REMOTE_DIR/input' '$REMOTE_DIR/output'"
remote_scp_to "$BINARY_PATH" "$REMOTE_BINARY"
remote_scp_to "$MODEL_PATH" "$REMOTE_MODEL"
remote_scp_to "$IMAGE_PATH" "$REMOTE_IMAGE"
upload_tengine_runtime
remote_tar_upload "$OPENCV_RUNTIME_DIR" "$REMOTE_DIR/lib"
remote_tar_upload "$GCC_RUNTIME_DIR" "$REMOTE_DIR/lib"

remote_ssh_tty "$(cat <<EOF
set -e
export LD_LIBRARY_PATH="$REMOTE_DIR/lib:\${LD_LIBRARY_PATH:-}"
cd "$REMOTE_DIR/output"
chmod +x "$REMOTE_BINARY"
"$REMOTE_BINARY" -c "$CONTEXT" -p "$PRECISION" -m "$REMOTE_MODEL" -i "$REMOTE_IMAGE" > "$REMOTE_DIR/output/run.log" 2>&1
cat "$REMOTE_DIR/output/run.log"
EOF
)"

remote_scp_from "$REMOTE_DIR/output/heatmap.png" "$OUT_DIR/heatmap.png"
remote_scp_from "$REMOTE_DIR/output/blend.png" "$OUT_DIR/blend.png"
remote_scp_from "$REMOTE_DIR/output/run.log" "$OUT_DIR/run.log"

if [[ -n "$MIN_HOT_INTENSITY" ]]; then
  MAX_HOT="$(awk '/max_hot intensity:/ {value=$NF} END {print value}' "$OUT_DIR/run.log")"
  if [[ -z "$MAX_HOT" ]]; then
    echo "Could not find max_hot intensity in $OUT_DIR/run.log" >&2
    exit 1
  fi
  awk -v value="$MAX_HOT" -v min="$MIN_HOT_INTENSITY" 'BEGIN { exit !(value + 0 >= min + 0) }' || {
    echo "max_hot intensity too low: $MAX_HOT < $MIN_HOT_INTENSITY" >&2
    exit 1
  }
fi

echo "Remote inference done."
echo "Heatmap: $OUT_DIR/heatmap.png"
echo "Blend: $OUT_DIR/blend.png"
echo "Log: $OUT_DIR/run.log"
