#!/usr/bin/env python3
"""Convert the existing HMap rotated-rectangle labels to YOLO detection labels.

The source dataset uses lines like:
  rel/image.png;cx,cy,w,h,angle_deg,class_id;...

This converter writes an Ultralytics-compatible detection dataset:
  out/
    data.yaml
    images/{train,val}/...
    labels/{train,val}/...
    stats.json

Images are symlinked by default to avoid duplicating large datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


CLASS_NAMES = ("bar", "qr", "dm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/home/yjunj/data/barcode"))
    parser.add_argument("--out-root", type=Path, default=Path("/home/yjunj/data/barcode_yolo_det"))
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="Debug limit; 0 means all training images.")
    parser.add_argument("--limit-val", type=int, default=0, help="Debug limit; 0 means all validation images.")
    return parser.parse_args()


def iter_label_rows(label_file: Path, limit: int = 0) -> Iterable[tuple[str, list[list[float]]]]:
    with label_file.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            instances = []
            for item in parts[1:]:
                values = [float(x) for x in item.split(",")]
                if len(values) != 6:
                    raise ValueError(f"{label_file}:{idx + 1}: expected 6 values, got {len(values)}")
                instances.append(values)
            yield parts[0], instances


def rrect_to_xyxy(cx: float, cy: float, w: float, h: float, angle: float) -> tuple[float, float, float, float]:
    points = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle)))
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def clamp_xyxy(
    box: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    x2 = max(0.0, min(float(image_w), x2))
    y2 = max(0.0, min(float(image_h), y2))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(box: tuple[float, float, float, float], image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    bx = x1 + bw / 2.0
    by = y1 + bh / 2.0
    return bx / image_w, by / image_h, bw / image_w, bh / image_h


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def convert_split(
    source_root: Path,
    out_root: Path,
    split: str,
    source_split: str,
    limit: int,
    copy_images: bool,
) -> dict:
    label_file = source_root / source_split / f"{source_split}.txt"
    rows = list(iter_label_rows(label_file, limit))
    class_counts: Counter[int] = Counter()
    image_count = 0
    instance_count = 0
    skipped_instances = 0
    missing_images = []
    areas = []

    for rel_path, instances in rows:
        src_img = source_root / source_split / rel_path
        if not src_img.is_file():
            missing_images.append(str(src_img))
            continue

        image = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing_images.append(str(src_img))
            continue
        image_h, image_w = image.shape[:2]

        label_lines = []
        for cx, cy, w, h, angle, class_id in instances:
            class_id = int(class_id)
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                skipped_instances += 1
                continue
            box = clamp_xyxy(rrect_to_xyxy(cx, cy, w, h, angle), image_w, image_h)
            if box is None:
                skipped_instances += 1
                continue
            x, y, bw, bh = xyxy_to_yolo(box, image_w, image_h)
            label_lines.append(f"{class_id} {x:.8f} {y:.8f} {bw:.8f} {bh:.8f}")
            class_counts[class_id] += 1
            instance_count += 1
            areas.append(float(bw * bh))

        if not label_lines:
            continue

        dst_img = out_root / "images" / split / rel_path
        dst_label = (out_root / "labels" / split / rel_path).with_suffix(".txt")
        link_or_copy(src_img, dst_img, copy_images)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        image_count += 1

    if missing_images:
        sample = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} missing/unreadable images, first entries:\n{sample}")

    mean_area = float(np.mean(areas)) if areas else 0.0
    median_area = float(np.median(areas)) if areas else 0.0
    return {
        "images": image_count,
        "instances": instance_count,
        "skipped_instances": skipped_instances,
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(class_counts.items())},
        "mean_normalized_area": mean_area,
        "median_normalized_area": median_area,
    }


def write_data_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_root} is not empty; pass --overwrite to rebuild it.")
    out_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "classes": list(CLASS_NAMES),
        "splits": {
            "train": convert_split(source_root, out_root, "train", "train", args.limit_train, args.copy_images),
            "val": convert_split(source_root, out_root, "val", "test", args.limit_val, args.copy_images),
        },
    }
    stats["total_images"] = sum(split["images"] for split in stats["splits"].values())
    stats["total_instances"] = sum(split["instances"] for split in stats["splits"].values())
    write_data_yaml(out_root)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
