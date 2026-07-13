#!/usr/bin/env python3
"""Collect representative calibration images for detector quantization."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Copy a deterministic sample of validation images for uint8 calibration."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/yjunj/data/barcode_yolo_det/images/val"),
        help="Source image directory. Defaults to the local barcode validation split.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=package_dir / "calibration" / "images",
        help="Output calibration image directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=512,
        help="Maximum number of calibration images to copy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
        help="Seed for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output images before copying.",
    )
    return parser


def collect_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.resolve()
    out_dir = args.out.resolve()

    if not source.is_dir():
        raise SystemExit(f"Source image directory not found: {source}")
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    images = collect_images(source)
    if not images:
        raise SystemExit(f"No images found under: {source}")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    selected = sorted(images[: min(args.count, len(images))])

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir.parent / "calibration_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(f"source={source}\n")
        manifest.write(f"count={len(selected)}\n")
        manifest.write(f"seed={args.seed}\n\n")
        for src in selected:
            rel = src.relative_to(source)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.write(f"{rel.as_posix()}\n")

    print(f"Copied {len(selected)} calibration images to {out_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
