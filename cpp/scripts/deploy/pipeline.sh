#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CPP_DIR/.." && pwd)"

PROFILE_PATH="$CPP_DIR/profiles/imx8plus.env"
ONNX_PATH=""
FP32_TMFILE="$CPP_DIR/artifacts/tmfile/model-fp32.tmfile"
UINT8_TMFILE="$CPP_DIR/artifacts/tmfile/model-uint8.tmfile"
CALIB_DATASET=""
IMAGE_PATH=""
OUT_DIR="$CPP_DIR/artifacts/imx8plus"
CONTEXT="timvx"
TENGINE_RUNTIME_MODE="minimal"
MIN_HOT_INTENSITY="1"
CONFIG_PATH="${HEATMAP_BOARD_CONFIG:-${HEATMAP_DEVICE_CONFIG:-}}"
SKIP_BUILD=0

usage() {
  cat <<'USAGE'
Usage: cpp/scripts/deploy/pipeline.sh [options]

Runs imx8plus deploy pipeline:
ONNX -> fp32 tmfile -> uint8 tmfile -> cross build -> on-board inference.

Required:
  --onnx PATH       Input ONNX model
  --calib PATH      Calibration image directory (for uint8 quantization)
  --image PATH      Test image for on-board inference

Options:
  --profile PATH          Conversion profile. Default: cpp/profiles/imx8plus.env
  --fp32 PATH             Output fp32 tmfile. Default: cpp/artifacts/tmfile/model-fp32.tmfile
  --uint8 PATH            Output uint8 tmfile. Default: cpp/artifacts/tmfile/model-uint8.tmfile
  --out-dir PATH          On-board result directory. Default: cpp/artifacts/imx8plus
  --context VALUE         Runtime context. Default: timvx
  --tengine-runtime VALUE Runtime deploy mode: all|minimal|none. Default: minimal
  --min-hot VALUE         Minimum max_hot intensity. Default: 1
  --config PATH           Board config (cpp/board.env). Default: HEATMAP_BOARD_CONFIG
  --skip-build            Reuse existing cpp/build/imx8plus/HMAP-TEST
  -h, --help              Show this help.
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
    --profile)
      PROFILE_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --onnx)
      ONNX_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --fp32)
      FP32_TMFILE="$(resolve_path "$2")"
      shift 2
      ;;
    --uint8)
      UINT8_TMFILE="$(resolve_path "$2")"
      shift 2
      ;;
    --calib)
      CALIB_DATASET="$(resolve_path "$2")"
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
    --context)
      CONTEXT="$2"
      shift 2
      ;;
    --tengine-runtime)
      TENGINE_RUNTIME_MODE="$2"
      shift 2
      ;;
    --min-hot)
      MIN_HOT_INTENSITY="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$(resolve_path "$2")"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
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

if [[ -z "$ONNX_PATH" ]]; then
  echo "--onnx is required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$CALIB_DATASET" ]]; then
  echo "--calib is required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$IMAGE_PATH" ]]; then
  echo "--image is required." >&2
  usage >&2
  exit 2
fi

"$CPP_DIR/scripts/convert/onnx_to_tmfile.sh" \
  --profile "$PROFILE_PATH" \
  --onnx "$ONNX_PATH" \
  --out "$FP32_TMFILE"

"$CPP_DIR/scripts/convert/quantize_uint8.sh" \
  --profile "$PROFILE_PATH" \
  --fp32 "$FP32_TMFILE" \
  --calib "$CALIB_DATASET" \
  --out "$UINT8_TMFILE"

if [[ "$SKIP_BUILD" != "1" ]]; then
  "$CPP_DIR/scripts/build/cross_build.sh"
fi

RUN_ARGS=(
  --context "$CONTEXT"
  --precision uint8
  --tengine-runtime "$TENGINE_RUNTIME_MODE"
  --model "$UINT8_TMFILE"
  --image "$IMAGE_PATH"
  --out-dir "$OUT_DIR"
  --min-hot "$MIN_HOT_INTENSITY"
)
if [[ -n "$CONFIG_PATH" ]]; then
  RUN_ARGS=(--config "$CONFIG_PATH" "${RUN_ARGS[@]}")
fi

"$CPP_DIR/scripts/deploy/run_on_board.sh" "${RUN_ARGS[@]}"
