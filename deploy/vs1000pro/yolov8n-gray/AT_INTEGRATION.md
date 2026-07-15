# AT Integration Guide

## Release Assets

- Runtime model: `tmfile/barcode-yolov8n-gray-final-uint8.tmfile`
- Cross-platform export: `onnx/barcode-yolov8n-gray-best-raw-splitfree-opset11.onnx`
- C++ API: `cpp/include/yolo_detector.h`
- Implementation: `cpp/src/yolo_detector.cpp`
- Board wrapper: `run.sh`

The model accepts one grayscale image and letterboxes it internally to fixed
`1x1x640x640`. It returns axis-aligned image-space boxes for `bar`, `qr`, and
`dm`.

## AT Lifecycle

1. When the image-adjustment page opens, construct `YoloDetector("timvx", "uint8")`
   and call `Init()` with the final tmfile path.
2. Reuse that instance for all adjustment frames by calling `Infer()`.
3. When adjustment completes, destroy the instance before deep decoding begins.

The initialized process uses about 82 MB PSS (83 MB RSS) while idle and does not
consume NPU compute until `Infer()` is called. Releasing it after adjustment
returns its graph, buffers, and TIM-VX context before the decoder phase.

## C++ Usage

```cpp
YoloDetector detector("timvx", "uint8");
if (!detector.Init("/usr/scanner/yolov8n-gray-test/model/barcode-yolov8n-gray-final-uint8.tmfile")) {
    return false;
}

std::vector<YoloDetection> detections;
YoloTiming timing;
if (!detector.Infer(gray_image, detections, timing)) {
    return false;
}
```

`gray_image` should be a one-channel `cv::Mat`. Production defaults are
confidence `0.25` and NMS IoU `0.45`.

## Board Result

The frozen UINT8 model has one TIM-VX NPU subgraph with 384 nodes and no CPU
fallback nodes. Sustained board latency is 79.091 ms end-to-end (12.644 FPS),
including preprocessing and C++ postprocessing. Full validation details are in
`README.md` and `validation/finaltest.metrics.json`.
