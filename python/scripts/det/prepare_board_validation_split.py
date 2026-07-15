#!/usr/bin/env python3
"""Create content-disjoint NFS views for board quantization validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quant-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="20260715-yolov8n-gray-final")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_score(seed: str, digest: str) -> float:
    value = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
    return int.from_bytes(value[:8], "big") / float(1 << 64)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.quant_fraction < 1.0:
        raise SystemExit("--quant-fraction must be between 0 and 1")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    paths = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(paths) != len(set(paths)):
        raise SystemExit("Manifest contains duplicate paths")

    records = []
    digest_split: dict[str, str] = {}
    for index, relative_text in enumerate(paths, start=1):
        relative = Path(relative_text)
        image = source_root / relative
        label = image.with_suffix(".json")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing image or label: {relative}")
        digest = file_sha256(image)
        split = digest_split.setdefault(
            digest,
            "quant-val" if split_score(args.seed, digest) < args.quant_fraction else "final-test",
        )
        records.append(
            {
                "relative": relative.as_posix(),
                "category": relative.parts[0],
                "sha256": digest,
                "bytes": image.stat().st_size,
                "split": split,
            }
        )
        if index % 500 == 0:
            print(f"hashed={index}/{len(paths)}", flush=True)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    for split in ("quant-val", "final-test"):
        split_root = output_root / split
        split_root.mkdir()
        selected = [record for record in records if record["split"] == split]
        with (output_root / f"{split}.tsv").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("relative\tcategory\tbytes\tsha256\n")
            for record in selected:
                relative = Path(record["relative"])
                source = source_root / relative
                destination = split_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(os.path.relpath(source, destination.parent))
                handle.write(
                    f"{record['relative']}\t{record['category']}\t{record['bytes']}\t{record['sha256']}\n"
                )

    counts = {
        split: dict(Counter(record["category"] for record in records if record["split"] == split))
        for split in ("quant-val", "final-test")
    }
    duplicate_files = len(records) - len(digest_split)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "source_root": str(source_root),
        "seed": args.seed,
        "quant_fraction": args.quant_fraction,
        "total_images": len(records),
        "unique_content_hashes": len(digest_split),
        "duplicate_files": duplicate_files,
        "split_counts": {split: sum(values.values()) for split, values in counts.items()},
        "category_counts": counts,
        "content_hash_overlap": 0,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
