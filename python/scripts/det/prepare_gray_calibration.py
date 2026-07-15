#!/usr/bin/env python3
"""Prepare grayscale letterbox images that exactly match board preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--fill", type=int, default=114)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in args.input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, path in enumerate(images):
        image = Image.open(path).convert("L")
        scale = min(args.size / image.width, args.size / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        left = (args.size - width) // 2
        top = (args.size - height) // 2
        canvas = Image.new("L", (args.size, args.size), color=args.fill)
        canvas.paste(resized, (left, top))
        output = args.output_dir / f"{index:04d}__{path.stem}.png"
        canvas.save(output)
    print(f"prepared={len(images)} size={args.size}x{args.size} channels=1 output={args.output_dir}")


if __name__ == "__main__":
    main()
