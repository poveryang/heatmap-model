#!/usr/bin/env python3
"""Train the deployment-oriented single-channel barcode detector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


PYTHON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_ROOT))

from yolo_detector import (  # noqa: E402
    GrayscalePretrainedDetectionTrainer,
    build_grayscale_model,
    resolve_pretrained_weights,
)
from yolo_detector.trainer import (  # noqa: E402
    PRETRAINED_BACKBONE_LAYERS_ENV,
    PRETRAINED_ENV,
)


DEFAULT_MODEL = PYTHON_ROOT / "configs" / "det" / "barcode-yolov8n-gray.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PYTHON_ROOT / "configs" / "det" / "barcode-data.yaml",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--pretrained-max-layer", type=int, default=21)
    parser.add_argument("--project", type=Path, default=Path("python/runs/det"))
    parser.add_argument("--name", default="barcode-yolov8n-gray")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val-conf", type=float, default=0.01)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1.0e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--mosaic", type=float, default=0.50)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_pythonpath() -> None:
    current = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(PYTHON_ROOT) if not current else f"{PYTHON_ROOT}{os.pathsep}{current}"


def dry_run(args: argparse.Namespace, pretrained: Path) -> None:
    model, transfer = build_grayscale_model(
        args.model,
        nc=3,
        pretrained=pretrained,
        pretrained_max_layer=args.pretrained_max_layer,
    )
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, args.imgsz, args.imgsz))
    prediction = output[0] if isinstance(output, tuple) else output
    summary = {
        "model": str(args.model),
        "input_shape": [1, 1, args.imgsz, args.imgsz],
        "output_shape": list(prediction.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "end2end": bool(model.end2end),
        "reg_max": int(model.model[-1].reg_max),
        "strides": [int(value) for value in model.stride.tolist()],
        "transfer": transfer,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    configure_pythonpath()

    if args.resume:
        YOLO(str(args.resume.expanduser().resolve())).train(resume=True)
        return

    model_config = args.model.expanduser().resolve()
    pretrained = resolve_pretrained_weights(args.pretrained)
    os.environ[PRETRAINED_ENV] = str(pretrained)
    os.environ[PRETRAINED_BACKBONE_LAYERS_ENV] = str(args.pretrained_max_layer)

    if args.dry_run:
        dry_run(args, pretrained)
        return

    model = YOLO(str(model_config))
    model.train(
        trainer=GrayscalePretrainedDetectionTrainer,
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=True,
        amp=True,
        cache=False,
        plots=args.plots,
        save=True,
        save_period=5,
        val=True,
        conf=args.val_conf,
        pretrained=False,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=0.0002,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        fliplr=0.5,
        flipud=0.5,
    )


if __name__ == "__main__":
    main()
