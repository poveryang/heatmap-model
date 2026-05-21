#!/usr/bin/env python3
"""Convert labelme png/json pairs under root_dir into train/test layout for HMapDataModule."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

LABEL_MAP = {
    'bar': 0, '1d': 0,
    'qr': 1,
    'dm': 2,
}


def polygon_to_rrect(points):
    pts = np.array(points, dtype=np.float32)
    (x, y), (w, h), angle = cv2.minAreaRect(pts)
    return x, y, w, h, angle


def load_pairs(source_root: Path):
    pairs = []
    for json_path in source_root.rglob('*.json'):
        png_path = json_path.with_suffix('.png')
        if not png_path.exists():
            continue
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        instances = []
        for shape in data.get('shapes', []):
            label = shape.get('label', '').lower()
            if label not in LABEL_MAP:
                continue
            inst = list(polygon_to_rrect(shape['points'])) + [LABEL_MAP[label]]
            instances.append(inst)
        if instances:
            pairs.append((png_path, instances))
    return pairs


def write_label_file(pairs, dest_dir: Path, txt_name: str):
    lines = []
    for src_png, instances in pairs:
        dst_png = dest_dir / src_png.name
        if dst_png.resolve() != src_png.resolve():
            shutil.copy2(src_png, dst_png)
        inst_strs = [','.join(str(v) for v in inst) for inst in instances]
        lines.append(f"{src_png.name};{';'.join(inst_strs)}")
    (dest_dir / txt_name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True, help='Dataset root with train/test/sample subdirs')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pairs = load_pairs(args.source)
    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[:args.max_samples]
    if len(pairs) < 4:
        raise SystemExit(f'Need at least 4 annotated samples, found {len(pairs)}')

    split = max(1, int(len(pairs) * args.train_ratio))
    train_pairs = pairs[:split]
    test_pairs = pairs[split:] or pairs[-1:]

    train_dir = args.root / 'train'
    test_dir = args.root / 'test'
    sample_dir = args.root / 'sample'
    for d in (train_dir, test_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob('*.png'):
            old.unlink()
        for old in d.glob('*.txt'):
            old.unlink()

    write_label_file(train_pairs, train_dir, 'train.txt')
    write_label_file(test_pairs, test_dir, 'test.txt')
    shutil.copy2(train_pairs[0][0], sample_dir / train_pairs[0][0].name)

    print(f'Prepared {len(train_pairs)} train / {len(test_pairs)} test samples under {args.root}')


if __name__ == '__main__':
    main()
