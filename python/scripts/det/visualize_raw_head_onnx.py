#!/usr/bin/env python3
"""Visualize fixed-shape raw YOLO head ONNX outputs using board-equivalent postprocess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ("bar", "qr", "dm")
COLORS = ((36, 176, 72), (224, 144, 32), (190, 58, 210))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--skip-output-images", action="store_true")
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(size / gray.shape[1], size / gray.shape[0])
    width = max(1, round(gray.shape[1] * scale))
    height = max(1, round(gray.shape[0] * scale))
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    left = (size - width) // 2
    top = (size - height) // 2
    canvas = np.full((size, size), 114, dtype=np.uint8)
    canvas[top : top + height, left : left + width] = resized
    return canvas[None, None].astype(np.float32) / 255.0, scale, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size and len(kept) < 300:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        a = boxes[current]
        b = boxes[remaining]
        x0 = np.maximum(a[0], b[:, 0])
        y0 = np.maximum(a[1], b[:, 1])
        x1 = np.minimum(a[2], b[:, 2])
        y1 = np.minimum(a[3], b[:, 3])
        intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
        suppress = (classes[remaining] == classes[current]) & (iou > threshold)
        order = remaining[~suppress]
    return kept


def decode(
    box_logits: np.ndarray,
    class_logits: np.ndarray,
    image_shape: tuple[int, int],
    scale: float,
    left: int,
    top: int,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_logits = box_logits[0]
    class_logits = class_logits[0]
    candidates = box_logits.shape[1]
    reg_max = box_logits.shape[0] // 4
    strides = (4, 8, 16, 32) if candidates == 34000 else (8, 16, 32)

    class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -80.0, 80.0)))
    classes = class_scores.argmax(axis=0)
    scores = class_scores[classes, np.arange(candidates)]
    selected = np.flatnonzero(scores >= conf)
    if not selected.size:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int32)

    if reg_max == 1:
        distances = box_logits[:, selected].T
    else:
        logits = box_logits[:, selected].T.reshape(-1, 4, reg_max)
        logits -= logits.max(axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * np.arange(reg_max, dtype=np.float32)).sum(axis=2)

    anchors = np.empty((selected.size, 2), dtype=np.float32)
    scales = np.empty(selected.size, dtype=np.float32)
    for output_index, candidate in enumerate(selected):
        level_index = int(candidate)
        for stride in strides:
            side = 640 // stride
            level_size = side * side
            if level_index < level_size:
                anchors[output_index] = (level_index % side + 0.5, level_index // side + 0.5)
                scales[output_index] = stride
                break
            level_index -= level_size

    boxes = np.empty((selected.size, 4), dtype=np.float32)
    boxes[:, 0] = (anchors[:, 0] - distances[:, 0]) * scales
    boxes[:, 1] = (anchors[:, 1] - distances[:, 1]) * scales
    boxes[:, 2] = (anchors[:, 0] + distances[:, 2]) * scales
    boxes[:, 3] = (anchors[:, 1] + distances[:, 3]) * scales
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / scale
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / scale
    height, width = image_shape
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, width - 1)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, height - 1)
    scores = scores[selected]
    classes = classes[selected].astype(np.int32)
    kept = nms(boxes, scores, classes, iou)
    return boxes[kept], scores[kept], classes[kept]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.limit]
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    records = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        tensor, scale, left, top = letterbox(image)
        box_logits, class_logits = session.run(None, {session.get_inputs()[0].name: tensor})
        boxes, scores, classes = decode(
            box_logits, class_logits, image.shape[:2], scale, left, top, args.conf, args.iou
        )
        output = None
        if not args.skip_output_images:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
            for box, score, class_id in zip(boxes, scores, classes):
                x0, y0, x1, y1 = box.astype(int)
                color = COLORS[int(class_id)]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{CLASS_NAMES[int(class_id)]} {score:.2f}", (x0, max(16, y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            output = args.output_dir / f"{path.stem}.det.png"
            cv2.imwrite(str(output), canvas)
        detection_records = [
            {
                "class_id": int(class_id),
                "score": float(score),
                "box": [float(value) for value in box],
            }
            for box, score, class_id in zip(boxes, scores, classes)
        ]
        records.append({
            "image": path.name,
            "detections": int(len(boxes)),
            "detection_records": detection_records,
            "output": str(output) if output is not None else None,
        })

    report = {"onnx": str(args.onnx), "images": records, "confidence": args.conf, "iou": args.iou}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
