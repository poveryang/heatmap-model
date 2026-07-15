#!/bin/sh
set -eu

APP_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
INPUT_DIR="${1:?Usage: $0 INPUT_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$APP_DIR/results}"
mkdir -p "$OUTPUT_DIR"

export VIVANTE_SDK_DIR="$APP_DIR/vivante-sdk"
export LD_LIBRARY_PATH="$APP_DIR/lib"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

exec "$APP_DIR/HMAP-TEST" -c timvx -p uint8 \
  -m "$APP_DIR/model/barcode-yolov8n-gray-final-uint8.tmfile" \
  -d "$INPUT_DIR" -o "$OUTPUT_DIR" -w "${YOLO_WARMUP:-10}"
