# YOLOv8 Gray Barcode Detector

This repository contains the frozen one-channel YOLOv8n detector for barcode,
QR, and Data Matrix localization. It includes training/export tooling, a
Tengine/TIM-VX C++ runtime, and the validated VS1000Pro release artifact.

## Release

- Runtime model: `deploy/vs1000pro/yolov8n-gray/tmfile/barcode-yolov8n-gray-final-uint8.tmfile`
- Input: fixed grayscale `1x1x640x640`
- Board result: 95.68% mAP50, 93.90% recall, 79.091 ms end-to-end mean
- AT integration: `deploy/vs1000pro/yolov8n-gray/AT_INTEGRATION.md`

## Layout

- `python/`: YOLOv8 training, export, calibration, and validation tools
- `cpp/`: Tengine/TIM-VX detector implementation for VS1000Pro
- `deploy/vs1000pro/yolov8n-gray/`: frozen model, ONNX export, checkpoint, and release evidence
