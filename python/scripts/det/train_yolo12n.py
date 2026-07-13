#!/usr/bin/env python3
"""Train a lightweight YOLO12n detector on the converted barcode dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/yjunj/data/barcode_yolo_det/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("python/runs/det"))
    parser.add_argument("--name", default="yolo12n-barcode-det")
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--device", default="1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1.0e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--mosaic", type=float, default=0.30)
    parser.add_argument("--scale", type=float, default=0.30)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.03)
    parser.add_argument("--shear", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        project=str(project),
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.10,
        hsv_v=0.20,
        fliplr=0.5,
        flipud=0.0,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
