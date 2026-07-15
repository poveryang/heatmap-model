# Release Validation Summary

The frozen UINT8 model was selected from three quantization candidates on a
SHA256 content-disjoint quantization-validation set of 2,028 images. The chosen
candidate achieved 95.40% mAP50 and 93.86% recall at confidence 0.25.

It was then evaluated once on a separate, content-disjoint final set of 8,358
images containing 14,074 labelled targets. Board results were 95.68% mAP50,
85.09% mAP50-95, 94.92% precision, and 93.90% recall at confidence 0.25.

`finaltest.metrics.json` is the machine-readable final result. The candidate
metric JSON files in `qval/` support the release selection. Raw labels, images,
visualizations, timing CSV files, and tar archives remain local validation
evidence and are deliberately not versioned.
