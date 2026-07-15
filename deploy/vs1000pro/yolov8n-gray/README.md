# YOLOv8n-gray VS1000Pro deployment

## Model

- Training run: `20260713_223145_barcode-yolov8n-gray`
- Selected checkpoint: epoch 116 `best.pt`
- Training mAP50 / mAP50-95: `0.97199 / 0.89799`
- Input: fixed `1x1x640x640` grayscale
- Raw outputs: box logits `[1,64,8400]`, class logits `[1,3,8400]`
- DFL Softmax, box decode, thresholding, and NMS: C++
- Calibration: 256 representative grayscale images, MIN-MAX, UINT8 per-channel
- TIM-VX compatibility: all NPU Concat inputs use their output tensor qparams
- Export optimization: mathematically equivalent split-free C2f blocks
- Final class-logit qparam: `scale=0.5`, `zero_point=245`

## Historical 128-image smoke test

The test used the same 128 images as the earlier dual-model smoke test, 10 warmups,
and disabled output image writing.

| Metric | Average | P50 | P95 |
| --- | ---: | ---: | ---: |
| Preprocess | 6.375 ms | 5.156 ms | 11.349 ms |
| Model execution | 65.438 ms | 65.744 ms | 67.429 ms |
| C++ postprocess | 2.277 ms | 2.215 ms | 2.642 ms |
| End to end | 74.090 ms | 73.729 ms | 80.584 ms |

- Pipeline throughput: `13.497 FPS`
- Model load + graph preparation: `15610.570 ms`
- Warmup model average: `66.004 ms`
- TIM-VX graph: 1 subgraph, 384 nodes
- CPU fallback: 0 nodes
- The load time is graph compilation, not model-file I/O; a RAM-backed model
  did not materially reduce initialization time. Production should keep one
  detector instance resident and reuse the prepared graph.

## Release validation (2026-07-15)

The release candidate was selected on a content-disjoint quantization-validation
set and evaluated once on a separate, content-disjoint final test set. The split
was created with SHA256 grouping so identical images cannot appear in both sets.
All image labels came from the original converted YOLO validation annotations.

| Set | Images | Role | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | F1 @ 0.25 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Quantization validation | 2,028 | Choose quantization candidate | 95.40% | 85.24% | 94.32% | 93.86% | 94.09% |
| Final test | 8,358 | One-time blind release evaluation | **95.68%** | **85.09%** | **94.92%** | **93.90%** | **94.41%** |

The final test contains 14,074 labelled targets. At the production confidence
threshold, the board produced 13,215 true positives, 707 false positives, and
859 false negatives. The weakest known scenes remain curved and inverted barcodes;
they are a training-data priority, not evidence of a conversion error.

### Sustained VS1000Pro performance

Five runs over the same 1,000 NFS-hosted grayscale images, with 10 warmups per
run and output-image writing disabled, were used for the release measurement.

| Metric | Mean | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| Preprocess | 5.388 ms | 5.093 ms | 7.952 ms | 8.117 ms |
| TIM-VX model execution | 68.514 ms | 68.464 ms | 68.688 ms | 70.011 ms |
| C++ postprocess | 5.188 ms | 4.942 ms | 6.247 ms | 9.135 ms |
| End to end | **79.091 ms** | 78.549 ms | 81.974 ms | 83.540 ms |

- Sustained end-to-end throughput: **12.644 FPS**.
- Per-run end-to-end means: 79.11, 79.11, 79.09, 79.05, and 79.09 ms.
- Board thermal zones rose from 39/40 C to 42/43 C across the five runs; no
  material performance drift was observed.
- Twenty process-cold starts measured model load and graph preparation at
  15,655.763 ms mean, 15,660.127 ms P50, and 15,730.975 ms P95. This is a
  process restart measurement with normal OS file cache, not a power-cycle test.
- After graph initialization and before any warmup or inference, the isolated
  detector process measured 83.0 MB RSS and 81.7 MB PSS (69.5 MB anonymous,
  13.5 MB file-backed). It does not consume NPU compute while idle, but it keeps
  the graph and associated driver resources alive until the process exits.
- TIM-VX graph: 1 NPU subgraph, 384 nodes; CPU fallback: 0 nodes.

The selected final file remains
`tmfile/barcode-yolov8n-gray-final-uint8.tmfile`. It is deployed on the board at
`/usr/scanner/yolov8n-gray-test`; the prior executable is retained there as
`HMAP-TEST.pre-validation-20260715`.

## Correctness investigation

The original MIN-MAX tmfile was correct on the Tengine UINT8 CPU backend but
incorrect on TIM-VX. CPU and TIM-VX tensors matched through `model.4/Split`; at
`model.6/Split`, the TIM-VX activation saturated (`raw_max=255`, mean `42.394`)
while CPU stayed at `raw_max=175`, mean `9.571`. The first failing interval
contains the first four-input C2f Concat, whose inputs had different qparams.

The selected model equalizes the input qparams of the 13 NPU Concat nodes with
their respective output qparams before quantization. In the diagnostic precursor,
the repaired TIM-VX tensor at `model.6/Split` was `raw_max=172`, mean `9.580`.
The final export then removes all eight C2f Split operators while preserving the
same math, so the severe edge-clipped false boxes disappear without CPU fallback.

Against FP32 ONNX on the same 128 images, 122 images have exactly the same
detection count. At class-aware IoU >= 0.5, 250 of 252 board detections match
the 254 ONNX detections: precision `99.21%`, reference recall `98.43%`, mean
matched IoU `0.9638`, and median matched IoU `0.9701`.

## Quantization selection

| Candidate | Model ms | ONNX precision | ONNX recall | Mean IoU | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 MinMax, original output qparam | **57.342** | 98.80% | 97.24% | 0.9582 | speed profile |
| 512 MinMax | 65.492 | 98.02% | 97.64% | 0.9627 | rejected |
| 256 KL | 65.434 | 99.59% | 94.49% | 0.9530 | rejected |
| 256 MinMax, class scale 0.5 | 65.438 | **99.21%** | **98.43%** | **0.9638** | selected |

Increasing calibration from 256 to 512 images did not produce a favorable
accuracy/speed tradeoff. KL lost 3.94 percentage points of ONNX-relative recall.
The selected output qparam preserves a useful range of class logits while giving
the quantized head finer confidence resolution than the original `scale=2.29`.

The release validation above supersedes the earlier small-sample operating-point
estimate. Confidence 0.25 remains the production default.

## Training policy

Do not train on the current validation set: it is also configured as `test`, so
doing so would remove the only independent accuracy measurement. For a future
training revision, first create a source-grouped held-out test set, then the old
validation images may be folded into training. The highest-value improvements
are hard-example mining for curved and inverted barcodes, correcting duplicate
labels, source/class-balanced sampling, and QAT focused on the terminal
class-logit quantization. Input 640 and the three P3-P5 heads remain fixed for
this release because P2 and larger inputs cost too much on VS1000Pro.

## Files

- `checkpoints/barcode-yolov8n-gray-best.pt` (training checkpoint)
- `onnx/barcode-yolov8n-gray-best-raw-splitfree-opset11.onnx` (portable raw-head export)
- `onnx/barcode-yolov8n-gray-best-raw-splitfree-opset11.operators.json` (operator audit)
- `tmfile/barcode-yolov8n-gray-final-uint8.tmfile` (selected board runtime model)
- `quant/splitfree256/table_splitfree256_minmax_concat_score05.scale` (selected qparams)
- `AT_INTEGRATION.md` (AT lifecycle and C++ integration)
- `validation/VALIDATION.md`, `validation/finaltest.metrics.json`, and
  `validation/qval/*.metrics.json` (release evidence)

Raw labels, images, visualization PNGs, per-image timing CSVs, and rejected
quantization candidates remain local evidence and are intentionally not versioned.

Board deployment: `/usr/scanner/yolov8n-gray-test`.

## SHA256

```text
6083fe665e261b188ed056971ff148f0e6db91be17520a78d67425412b5671d6  barcode-yolov8n-gray-best.pt
3733eacd51b48218b245c807804a8fca4e02fee8e6744e112d4cca396977ecfc  barcode-yolov8n-gray-best-raw-splitfree-opset11.onnx
92717f32e365405f80f9e168f97bf89165e810fe5bf40a3f9f8487e9f7b8691b  barcode-yolov8n-gray-best-raw-splitfree-fp32.tmfile
51219a7fa9cb55e4e34d0eef0eadae6947d791852695e12b33f65b2d7e7e5b31  barcode-yolov8n-gray-final-uint8.tmfile
```
