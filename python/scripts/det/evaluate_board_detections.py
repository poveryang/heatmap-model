#!/usr/bin/env python3
"""Evaluate board detection CSVs against the YOLO label files used for validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = ("bar", "qr", "dm")
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-confidence", type=float, default=0.001)
    return parser.parse_args()


def load_manifest(path: Path) -> list[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0].split("\t")[0] != "relative":
        raise ValueError(f"Unexpected manifest format: {path}")
    return [line.split("\t", maxsplit=1)[0] for line in rows[1:] if line]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union > 0.0 else 0.0


def load_ground_truth(relatives: list[str], labels_root: Path, images_root: Path) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    ground_truth: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for index, relative in enumerate(relatives, start=1):
        image_path = images_root / relative
        label_path = (labels_root / relative).with_suffix(".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = []
        for row in label_path.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            class_id, center_x, center_y, box_width, box_height = map(float, row.split())
            class_index = int(class_id)
            x0 = (center_x - box_width / 2.0) * width
            y0 = (center_y - box_height / 2.0) * height
            x1 = (center_x + box_width / 2.0) * width
            y1 = (center_y + box_height / 2.0) * height
            boxes.append((class_index, (x0, y0, x1, y1)))
        ground_truth[relative] = boxes
        if index % 1000 == 0:
            print(f"loaded_ground_truth={index}/{len(relatives)}", flush=True)
    return ground_truth


def load_predictions(path: Path, allowed: set[str], minimum_confidence: float) -> list[tuple[str, int, float, tuple[float, float, float, float]]]:
    predictions = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in allowed:
                raise ValueError(f"Prediction outside manifest: {row['image']}")
            score = float(row["score"])
            if score < minimum_confidence:
                continue
            predictions.append(
                (
                    row["image"],
                    int(row["class_id"]),
                    score,
                    (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])),
                )
            )
    return predictions


def match_predictions(
    predictions: list[tuple[str, int, float, tuple[float, float, float, float]]],
    ground_truth: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    class_id: int,
    threshold: float,
) -> tuple[list[int], list[int], int]:
    candidates = [prediction for prediction in predictions if prediction[1] == class_id]
    candidates.sort(key=lambda item: item[2], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    true_positive: list[int] = []
    false_positive: list[int] = []
    total_ground_truth = sum(sum(item[0] == class_id for item in boxes) for boxes in ground_truth.values())
    for image, _, _, box in candidates:
        best_index = -1
        best_iou = threshold
        for index, (target_class, target_box) in enumerate(ground_truth[image]):
            if target_class != class_id or index in matched[image]:
                continue
            overlap = iou(box, target_box)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            matched[image].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    return true_positive, false_positive, total_ground_truth


def average_precision(true_positive: list[int], false_positive: list[int], total_ground_truth: int) -> float:
    if total_ground_truth == 0:
        return 0.0
    cumulative_tp = []
    cumulative_fp = []
    tp = fp = 0
    for current_tp, current_fp in zip(true_positive, false_positive):
        tp += current_tp
        fp += current_fp
        cumulative_tp.append(tp)
        cumulative_fp.append(fp)
    recalls = [value / total_ground_truth for value in cumulative_tp]
    precisions = [tp_count / max(1, tp_count + fp_count) for tp_count, fp_count in zip(cumulative_tp, cumulative_fp)]
    result = 0.0
    for point in range(101):
        recall = point / 100.0
        result += max((precision for precision, current_recall in zip(precisions, recalls) if current_recall >= recall), default=0.0)
    return result / 101.0


def operating_metrics(
    predictions: list[tuple[str, int, float, tuple[float, float, float, float]]],
    ground_truth: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    confidence: float,
    class_id: int | None = None,
) -> dict[str, float | int]:
    selected = [prediction for prediction in predictions if prediction[2] >= confidence and (class_id is None or prediction[1] == class_id)]
    selected.sort(key=lambda item: item[2], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    tp = fp = 0
    gt_total = 0
    for boxes in ground_truth.values():
        gt_total += sum(class_id is None or item[0] == class_id for item in boxes)
    for image, predicted_class, _, box in selected:
        best_index = -1
        best_iou = 0.5
        for index, (target_class, target_box) in enumerate(ground_truth[image]):
            if index in matched[image] or target_class != predicted_class:
                continue
            overlap = iou(box, target_box)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            matched[image].add(best_index)
            tp += 1
        else:
            fp += 1
    fn = gt_total - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_total if gt_total else 0.0
    return {
        "ground_truth": gt_total,
        "predictions": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    args = parse_args()
    relatives = load_manifest(args.manifest)
    ground_truth = load_ground_truth(relatives, args.labels_root, args.images_root)
    predictions = load_predictions(args.detections, set(relatives), args.minimum_confidence)
    per_class = {}
    class_maps = []
    class_ap50 = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        ap_by_iou = {}
        for threshold in IOU_THRESHOLDS:
            tp, fp, total = match_predictions(predictions, ground_truth, class_id, threshold)
            ap_by_iou[f"{threshold:.2f}"] = average_precision(tp, fp, total)
        operating = operating_metrics(predictions, ground_truth, args.operating_confidence, class_id)
        per_class[class_name] = {
            "ap50": ap_by_iou["0.50"],
            "map50_95": sum(ap_by_iou.values()) / len(ap_by_iou),
            "ap_by_iou": ap_by_iou,
            "operating": operating,
        }
        class_ap50.append(ap_by_iou["0.50"])
        class_maps.append(sum(ap_by_iou.values()) / len(ap_by_iou))

    per_category = {}
    for category in sorted({Path(relative).parts[0] for relative in relatives}):
        category_ground_truth = {
            relative: boxes for relative, boxes in ground_truth.items() if Path(relative).parts[0] == category
        }
        category_predictions = [prediction for prediction in predictions if Path(prediction[0]).parts[0] == category]
        per_category[category] = operating_metrics(
            category_predictions, category_ground_truth, args.operating_confidence
        )

    summary = {
        "images": len(relatives),
        "predictions_at_minimum_confidence": len(predictions),
        "minimum_confidence": args.minimum_confidence,
        "operating_confidence": args.operating_confidence,
        "metrics": {"map50": sum(class_ap50) / len(class_ap50), "map50_95": sum(class_maps) / len(class_maps)},
        "operating": operating_metrics(predictions, ground_truth, args.operating_confidence),
        "per_class": per_class,
        "per_category": per_category,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
