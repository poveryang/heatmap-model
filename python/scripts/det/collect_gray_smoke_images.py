#!/usr/bin/env python3
"""Collect deterministic, category-balanced grayscale calibration or test images."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
EXCLUDED_PARTS = {"viz", "visualization", "visualizations"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/home/yjunj/data/barcode"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--exclude-manifest", type=Path)
    return parser.parse_args()


def excluded_paths(manifest: Path | None) -> set[str]:
    if manifest is None:
        return set()
    excluded: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        excluded.add(line.split("\t", maxsplit=1)[0])
    return excluded


def collect(source: Path, excluded: set[str]) -> dict[str, deque[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(source)
        if any(part.lower() in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() in excluded:
            continue
        group = relative.parent.as_posix()
        grouped[group].append(path)
    return {group: deque(paths) for group, paths in grouped.items()}


def balanced_sample(groups: dict[str, deque[Path]], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    for paths in groups.values():
        items = list(paths)
        rng.shuffle(items)
        paths.clear()
        paths.extend(items)

    selected: list[Path] = []
    group_names = sorted(groups)
    rng.shuffle(group_names)
    while len(selected) < count:
        added = False
        for group in group_names:
            if groups[group]:
                selected.append(groups[group].popleft())
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
    return selected


def safe_name(path: Path, index: int) -> str:
    context = "__".join(path.parts[-3:])
    context = re.sub(r"[^A-Za-z0-9_.-]+", "_", context)
    return f"{index:04d}__{context}"


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    excluded = excluded_paths(args.exclude_manifest)
    groups = collect(source, excluded)
    selected = balanced_sample(groups, args.count, args.seed)
    if len(selected) < args.count:
        raise SystemExit(f"Only found {len(selected)} usable images, requested {args.count}")

    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# source={source}\n# count={len(selected)}\n# seed={args.seed}\n")
        for index, path in enumerate(selected):
            relative = path.relative_to(source).as_posix()
            destination = images_dir / safe_name(path.relative_to(source), index)
            shutil.copy2(path, destination)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{relative}\t{destination.name}\t{path.stat().st_size}\t{digest}\n")

    print(f"sampled={len(selected)} groups={len(groups)} output={output}")


if __name__ == "__main__":
    main()
