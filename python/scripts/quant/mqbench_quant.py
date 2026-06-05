#!/usr/bin/env python3
"""Export a MQBench Tengine_u8 calibration table for the heatmap model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PYTHON_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "build" / "matplotlib"))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import cv2
import numpy as np
import onnx
import torch
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.utils.state import enable_calibration, enable_quantization

from hmap import CONFIGS_DIR  # noqa: E402
from hmap.model import CSPPAFPNNet  # noqa: E402
from hmap.utils.misc import load_configs  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def iter_images(root: Path) -> list[Path]:
    images = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    ]
    images.sort()
    return images


def select_images(images: list[Path], limit: int | None, mode: str) -> list[Path]:
    if limit is None or limit <= 0 or len(images) <= limit:
        return images
    if mode == "head":
        return images[:limit]
    if mode != "even":
        raise ValueError(f"Unsupported sample mode: {mode}")
    if limit == 1:
        return [images[0]]
    max_index = len(images) - 1
    return [images[round(index * max_index / (limit - 1))] for index in range(limit)]


def batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def preprocess_image(path: Path, height: int, width: int, mean: float, std: float) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return torch.from_numpy(image).unsqueeze(0)


def load_model(exp_name: str, ckpt_path: Path) -> torch.nn.Module:
    configs = load_configs(CONFIGS_DIR / f"{exp_name}.yaml")
    model_conf = dict(configs["model"])
    model_conf.pop("gamma", None)
    model_conf.pop("alpha", None)
    model_conf.pop("init_lr", None)

    model = CSPPAFPNNet(**model_conf)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("generator."):
            state_dict[key.replace("generator.", "", 1)] = value
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_config(observer: str, weight_qscheme: str) -> dict:
    extra_qconfig_dict = {
        "w_observer": "MinMaxObserver",
    }
    if observer != "default":
        extra_qconfig_dict["a_observer"] = observer
    if weight_qscheme == "symmetric":
        extra_qconfig_dict["w_qscheme"] = {
            "symmetry": True,
            "per_channel": False,
            "pot_scale": False,
            "bit": 8,
            "symmetric_range": True,
        }
    elif weight_qscheme != "tengine-default":
        raise ValueError(f"Unsupported weight qscheme: {weight_qscheme}")
    return {"extra_qconfig_dict": extra_qconfig_dict}


def force_input_qparams(scale_path: Path, input_name: str, mean: float, std: float) -> tuple[float, int]:
    input_scale = 1.0 / (std * 255.0)
    input_zero_point = int(round(mean * 255.0))

    lines = []
    replaced = False
    if scale_path.exists():
        for line in scale_path.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == input_name:
                lines.append(f"{input_name} {input_scale:.9g} {input_zero_point}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.insert(0, f"{input_name} {input_scale:.9g} {input_zero_point}")

    scale_path.write_text("\n".join(lines) + "\n")
    return input_scale, input_zero_point


def normalize_onnx_input_name(onnx_path: Path, input_name: str) -> str | None:
    model = onnx.load(onnx_path)
    if not model.graph.input:
        return None

    old_input_name = model.graph.input[0].name
    if old_input_name == input_name:
        return old_input_name

    model.graph.input[0].name = input_name
    for node in model.graph.node:
        for index, node_input in enumerate(node.input):
            if node_input == old_input_name:
                node.input[index] = input_name
    for value_info in model.graph.value_info:
        if value_info.name == old_input_name:
            value_info.name = input_name
    onnx.save(model, onnx_path)
    return old_input_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Input Lightning checkpoint.")
    parser.add_argument("--exp", default="hmap-v2", help="Config name under python/configs.")
    parser.add_argument("--calib", required=True, help="Calibration image directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for ONNX and scale files.")
    parser.add_argument("--model-name", default="hmap_mqbench", help="Output artifact base name.")
    parser.add_argument(
        "--observer",
        default="default",
        choices=["default", "MinMaxObserver", "EMAMinMaxObserver", "MSEObserver"],
        help="MQBench activation observer. default uses Tengine_u8 defaults.",
    )
    parser.add_argument(
        "--weight-qscheme",
        choices=["symmetric", "tengine-default"],
        default="symmetric",
        help=(
            "Weight quantization scheme. The MQBench Tengine_u8 default is "
            "asymmetric uint8, which can clip negative weights during Tengine "
            "export; symmetric is the safer default for this convolutional detector."
        ),
    )
    parser.add_argument("--calib-limit", type=int, default=512,
                        help="Maximum calibration images. Use <=0 for all images.")
    parser.add_argument("--sample-mode", choices=["even", "head"], default="even",
                        help="How to select --calib-limit images. Default: even.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--mean", type=float, default=0.4329)
    parser.add_argument("--std", type=float, default=0.2349)
    parser.add_argument("--input-name", default="data")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--no-force-input-qparams", action="store_true",
                        help="Keep MQBench-observed input qparams instead of deployment preprocessing qparams.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt)
    calib_dir = resolve_path(args.calib)
    out_dir = resolve_path(args.out_dir)
    limit = None if args.calib_limit <= 0 else args.calib_limit

    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not calib_dir.is_dir():
        raise SystemExit(f"Calibration directory not found: {calib_dir}")

    images = select_images(iter_images(calib_dir), limit, args.sample_mode)
    if not images:
        raise SystemExit(f"No calibration images found under: {calib_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.exp, ckpt_path)
    model = prepare_by_platform(
        model,
        BackendType.Tengine_u8,
        prepare_custom_config_dict=prepare_config(args.observer, args.weight_qscheme),
    )
    model.eval()

    enable_calibration(model)
    with torch.no_grad():
        for index, batch_paths in enumerate(batches(images, args.batch_size), start=1):
            batch = torch.stack([
                preprocess_image(path, args.height, args.width, args.mean, args.std)
                for path in batch_paths
            ])
            model(batch)
            print(f"calibration {index}: {len(batch_paths)} images", flush=True)

    enable_quantization(model)
    with torch.no_grad():
        batch_paths = images[: args.batch_size]
        batch = torch.stack([
            preprocess_image(path, args.height, args.width, args.mean, args.std)
            for path in batch_paths
        ])
        model(batch)

    scale_path = out_dir / f"{args.model_name}_for_tengine.scale"
    onnx_path = out_dir / f"{args.model_name}_for_tengine.onnx"
    dummy_input = torch.zeros(1, 1, args.height, args.width)
    try:
        convert_deploy(
            model,
            BackendType.Tengine_u8,
            input_shape_dict={args.input_name: [1, 1, args.height, args.width]},
            dummy_input=dummy_input,
            output_path=str(out_dir),
            model_name=args.model_name,
            input_names=[args.input_name],
            output_names=[args.output_name],
        )
    except IndexError:
        # MQBench 0.0.6 can raise after writing Tengine files when the final
        # DequantizeLinear feeds graph output directly. Keep the usable files.
        if not scale_path.is_file() or not onnx_path.is_file():
            raise

    exported_input_name = normalize_onnx_input_name(onnx_path, args.input_name)
    input_qparams = None
    if not args.no_force_input_qparams:
        input_qparams = force_input_qparams(scale_path, args.input_name, args.mean, args.std)

    metadata = {
        "tool": "MQBench",
        "backend": "Tengine_u8",
        "observer": args.observer,
        "weight_qscheme": args.weight_qscheme,
        "checkpoint": str(ckpt_path),
        "calibration_dir": str(calib_dir),
        "calibration_images": len(images),
        "sample_mode": args.sample_mode,
        "input_shape": [1, 1, args.height, args.width],
        "preprocess": {
            "resize": [args.height, args.width],
            "mean": args.mean,
            "std": args.std,
            "exported_input_name": exported_input_name,
            "forced_input_qparams": None if input_qparams is None else {
                "scale": input_qparams[0],
                "zero_point": input_qparams[1],
            },
        },
        "outputs": {
            "onnx": str(onnx_path),
            "scale": str(scale_path),
        },
    }
    (out_dir / "mqbench_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(scale_path)


if __name__ == "__main__":
    main()
