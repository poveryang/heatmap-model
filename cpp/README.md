# YOLOv8 Tengine Runtime

## Sources

- `include/yolo_detector.h`: public detector API
- `src/yolo_detector.cpp`: Tengine/TIM-VX execution, letterbox preprocessing,
  raw-head decode, thresholding, and class-aware NMS
- `test/test_yolo.cpp`: standalone validation executable

## Build for VS1000Pro

```bash
cmake -S cpp -B cpp/build/imx8plus -DTARGET_PLATFORM=imx8plus
cmake --build cpp/build/imx8plus --target YOLO-DETECTOR-TEST -j
```

`YOLO_DETECTOR` is a static library intended for AT integration. Runtime needs
`libtengine-lite.so`, OpenCV core/imgproc/imgcodecs, and matching Vivante
OpenVX/TIM-VX libraries on the board. See
`deploy/vs1000pro/yolov8n-gray/AT_INTEGRATION.md` for lifecycle details.
