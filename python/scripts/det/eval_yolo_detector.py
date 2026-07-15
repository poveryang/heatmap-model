#!/usr/bin/env python3
"""Evaluate accuracy, speed, and export size for an Ultralytics detector."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PYTHON_ROOT / "configs" / "det" / "barcode-data.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=Path("python/runs/det_eval"))
    parser.add_argument("--name", default="barcode-yolo26n-p2-gray")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--visual-conf", type=float, default=0.25)
    parser.add_argument(
        "--operating-conf",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25],
        help="Confidence thresholds to sample from the validation P/R curves.",
    )
    parser.add_argument("--visual-samples", type=int, default=16)
    parser.add_argument("--timing-warmup", type=int, default=30)
    parser.add_argument("--timing-iters", type=int, default=200)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def resolve_validation_images(data_yaml: Path, limit: int) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val = config["val"]
    val_entries = val if isinstance(val, list) else [val]
    images: list[Path] = []
    for entry in val_entries:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return sorted(images)[:limit]


def torch_device(device: str) -> torch.device:
    first = device.split(",", maxsplit=1)[0].strip()
    if first.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{first}")


def benchmark_single_image(
    model: YOLO,
    device_arg: str,
    imgsz: int,
    warmup: int,
    iterations: int,
    half: bool,
) -> dict[str, float | int | str | bool]:
    device = torch_device(device_arg)
    network = model.model.to(device).eval()
    use_half = half and device.type == "cuda"
    network.half() if use_half else network.float()
    channels = int(network.model[0].conv.in_channels)
    sample = torch.zeros(1, channels, imgsz, imgsz, device=device)
    sample = sample.half() if use_half else sample.float()

    with torch.inference_mode():
        for _ in range(warmup):
            network(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        samples_ms: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            network(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    p95_index = min(len(samples_ms) - 1, int(len(samples_ms) * 0.95))
    result = {
        "device": str(device),
        "half": use_half,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": samples_ms[p95_index],
        "fps_from_mean": 1000.0 / statistics.fmean(samples_ms),
    }
    network.float()
    return result


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    data = args.data.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    result_dir = Path(metrics.save_dir)
    class_names = metrics.names
    per_class = {
        str(class_names[index]): {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "map50": float(metrics.box.ap50[index]),
            "map50_95": float(metrics.box.ap[index].mean()),
        }
        for index in range(len(metrics.box.p))
    }
    summary = {
        "weights": str(weights),
        "weights_mb": weights.stat().st_size / 1024 / 1024,
        "imgsz": args.imgsz,
        "confidence_floor": args.conf,
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "save_dir": str(result_dir),
    }

    confidence_axis = metrics.box.px
    for threshold in args.operating_conf:
        index = int(abs(confidence_axis - threshold).argmin())
        per_class_at_threshold = {
            str(class_names[class_index]): {
                "precision": float(metrics.box.p_curve[class_index, index]),
                "recall": float(metrics.box.r_curve[class_index, index]),
                "f1": float(metrics.box.f1_curve[class_index, index]),
            }
            for class_index in range(len(class_names))
        }
        summary.setdefault("operating_points", {})[f"{threshold:.3f}"] = {
            "confidence": float(confidence_axis[index]),
            "precision": float(metrics.box.p_curve[:, index].mean()),
            "recall": float(metrics.box.r_curve[:, index].mean()),
            "f1": float(metrics.box.f1_curve[:, index].mean()),
            "per_class": per_class_at_threshold,
        }

    samples = resolve_validation_images(data, args.visual_samples)
    if samples:
        predictions = model.predict(
            source=[str(path) for path in samples],
            imgsz=args.imgsz,
            conf=args.visual_conf,
            device=args.device,
            project=str(result_dir),
            name="predictions",
            exist_ok=True,
            save=True,
            verbose=False,
        )
        summary["visualizations"] = {
            "count": len(predictions),
            "source_images": [str(path) for path in samples],
            "save_dir": str(result_dir / "predictions"),
        }

    if args.benchmark:
        benchmark = model.benchmark(
            data=str(data),
            imgsz=args.imgsz,
            half=args.half,
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

    # Validation may fuse layers under torch.inference_mode(), producing tensors
    # that cannot be reused by a later eager timing loop. Reload for timing.
    timing_model = YOLO(str(weights))
    summary["single_image_model_timing"] = benchmark_single_image(
        timing_model,
        args.device,
        args.imgsz,
        args.timing_warmup,
        args.timing_iters,
        args.half,
    )

    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
