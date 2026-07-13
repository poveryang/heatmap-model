# Lightweight Barcode Detector

This folder contains the reproducible object-detection-only pipeline used for the
lightweight barcode detector experiment.

## Model Choice

Primary model: `yolo12n.pt`.

Rationale:
- nano-size detector: about 2.6M parameters and 6.5 GFLOPs at 640 input.
- strong current accuracy/latency trade-off among Ultralytics-compatible nano
  detectors.
- simple deployment path through Ultralytics export to ONNX/TensorRT.

Fallback if absolute latency is more important than accuracy: benchmark
`yolov10n.pt`, but treat it as a speed fallback rather than the default.

## Dataset Conversion

The source HMap labels store rotated rectangles in text files such as:

```text
relative/image.png;cx,cy,w,h,angle,class;...
```

Convert them to standard YOLO axis-aligned detection labels:

```bash
/home/yjunj/miniforge3/envs/hmap/bin/python python/scripts/det/convert_hmap_to_yolo.py \
  --source-root /home/yjunj/data/barcode \
  --out-root /home/yjunj/data/barcode_yolo_det
```

The converter writes `data.yaml`, split image/label folders, and `stats.json`.

## Training

Stable training recipe used for the main run:

```bash
/home/yjunj/miniforge3/envs/hmap/bin/python python/scripts/det/train_yolo12n.py \
  --data /home/yjunj/data/barcode_yolo_det/data.yaml \
  --project python/runs/det \
  --name yolo12n-barcode-det-adamw \
  --epochs 20 \
  --batch 128 \
  --device 0,1,2,3 \
  --workers 12 \
  --patience 8 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --warmup-epochs 1 \
  --mosaic 0.3 \
  --scale 0.3 \
  --degrees 5 \
  --translate 0.03 \
  --shear 1 \
  --close-mosaic 5
```

The recipe intentionally uses moderate geometric augmentation for barcode-like
targets and closes mosaic near the end so final epochs match deployment images
more closely.

## Evaluation And Export

Run an independent validation summary and optional ONNX export:

```bash
/home/yjunj/miniforge3/envs/hmap/bin/python python/scripts/det/eval_yolo_detector.py \
  --weights /home/yjunj/projects/heatmap-model/python/runs/det/yolo12n-barcode-det-adamw/weights/best.pt \
  --data /home/yjunj/data/barcode_yolo_det/data.yaml \
  --project python/runs/det_eval \
  --name yolo12n-barcode-det-adamw \
  --imgsz 640 \
  --batch 64 \
  --device 0 \
  --export-onnx
```

The summary is written to the evaluation run directory as `summary.json`.
