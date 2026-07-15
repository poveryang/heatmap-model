# YOLOv8n Gray Barcode Detector

This is the frozen training and export pipeline for the one-channel barcode
detector used by the VS1000Pro image-adjustment feature. It detects `bar`, `qr`,
and `dm`.

## Model

- Architecture: `python/configs/det/barcode-yolov8n-gray.yaml`
- Input: fixed grayscale `1x1x640x640`
- Pretraining: COCO `yolov8n.pt`; shared layers 0-21 are transferred by name.
- RGB stem conversion: the three pretrained input-channel kernels are summed into
  one grayscale kernel.
- Deployment export: raw box and class logits. DFL decode, sigmoid, thresholding,
  and class-aware NMS execute in C++.

## Dataset

```bash
python python/scripts/det/convert_hmap_to_yolo.py \
  --source-root /home/yjunj/data/barcode \
  --out-root /home/yjunj/data/barcode_yolo_det
```

The generated data YAML must retain `channels: 1`.

## Train

```bash
python python/scripts/det/train_barcode_detector.py \
  --data python/configs/det/barcode-data.yaml \
  --model python/configs/det/barcode-yolov8n-gray.yaml \
  --pretrained yolov8n.pt \
  --pretrained-max-layer 21 \
  --name barcode-yolov8n-gray
```

Use `--dry-run` to validate the one-channel model and grayscale pretraining
transfer before starting a run.

## Export

```bash
python python/scripts/det/export_raw_head_onnx.py \
  --checkpoint python/runs/det/<run>/weights/best.pt \
  --split-free-c2f \
  --output deploy/vs1000pro/yolov8n-gray/onnx/barcode-yolov8n-gray.onnx
```

Use `prepare_gray_calibration.py` to letterbox representative grayscale training
images exactly as the board does. `equalize_concat_scales.py` repairs TIM-VX
Concat qparams before UINT8 quantization.

## Board Validation

`prepare_board_validation_split.py` creates SHA256 content-disjoint quantization
validation and final-test views. `evaluate_board_detections.py` evaluates the
board detection CSV against YOLO labels. The frozen model and full release
results are under `deploy/vs1000pro/yolov8n-gray/`.
