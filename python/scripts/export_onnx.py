#!/usr/bin/env python3
"""Export a heatmap Lightning checkpoint to ONNX with the legacy exporter."""

import argparse
import os
import shutil
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))


def resolve_path(raw_path):
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a heatmap checkpoint to ONNX using hmap.utils.export_model.export_onnx."
    )
    parser.add_argument("--exp", required=True, help="Experiment config name, for example hmap-v2.")
    parser.add_argument("--ckpt", required=True, help="Path to the Lightning .ckpt file.")
    parser.add_argument(
        "--out",
        help="Optional output ONNX path. By default the legacy exporter writes next to the checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt)
    out_path = resolve_path(args.out) if args.out else None

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    from hmap.utils.export_model import export_onnx

    old_cwd = Path.cwd()
    os.chdir(PYTHON_ROOT)
    try:
        export_onnx(args.exp, str(ckpt_path))
    finally:
        os.chdir(old_cwd)

    generated_path = Path(str(ckpt_path).replace(".ckpt", ".onnx"))
    if out_path is not None and generated_path.resolve() != out_path.resolve():
        if out_path.exists() and not args.force:
            raise SystemExit(f"Output already exists, pass --force to overwrite: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(generated_path), str(out_path))
        generated_path = out_path

    print(generated_path)


if __name__ == "__main__":
    main()
