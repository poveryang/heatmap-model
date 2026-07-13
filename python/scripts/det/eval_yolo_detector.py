#!/usr/bin/env python3
"""Evaluate accuracy, speed, and export size for an Ultralytics detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/home/yjunj/data/barcode_yolo_det/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("python/runs/det_eval"))
    parser.add_argument("--name", default="yolo12n-barcode-det")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="1")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        plots=True,
    )

    result_dir = Path(metrics.save_dir)
    summary = {
        "weights": str(args.weights),
        "weights_mb": args.weights.stat().st_size / 1024 / 1024,
        "imgsz": args.imgsz,
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "save_dir": str(result_dir),
    }

    if args.benchmark:
        benchmark = model.benchmark(
            data=str(args.data),
            imgsz=args.imgsz,
            half=True,
            int8=False,
            device=args.device,
            verbose=False,
        )
        summary["benchmark"] = str(benchmark)

    if args.export_onnx:
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, half=False, dynamic=False, simplify=True)
        summary["onnx"] = {
            "path": str(onnx_path),
            "size_mb": Path(onnx_path).stat().st_size / 1024 / 1024,
        }

    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
