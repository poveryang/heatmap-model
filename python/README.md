# YOLOv8 Training and Export

## Environment

```bash
conda create -n yolo-barcode python=3.10 -y
conda activate yolo-barcode
pip install -r python/requirements.txt
```

## Dataset Conversion

```bash
python python/scripts/det/convert_barcode_to_yolo.py \
  --source-root /home/yjunj/data/barcode \
  --out-root /home/yjunj/data/barcode_yolo_det
```

## Train

```bash
python python/scripts/det/train_barcode_detector.py \
  --data python/configs/det/barcode-data.yaml \
  --model python/configs/det/barcode-yolov8n-gray.yaml \
  --pretrained yolov8n.pt \
  --pretrained-max-layer 21 \
  --name barcode-yolov8n-gray
```

The frozen ONNX, tmfile, scale table, and validation evidence are under
`deploy/vs1000pro/yolov8n-gray/`.
