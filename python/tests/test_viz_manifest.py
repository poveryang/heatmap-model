from pathlib import Path

import pytest

from hmap.dataset.hmap_dataset import parse_viz_manifest
from hmap.utils.visualization import infer_grid_cols


def test_infer_grid_cols():
    assert infer_grid_cols(1) == 1
    assert infer_grid_cols(6) == 3
    assert infer_grid_cols(9) == 3


class _FakeValDataset:
    def __init__(self, tmp_path):
        self.data_dir = tmp_path / 'test'
        self.data_dir.mkdir(parents=True)
        self.img_paths = []
        for name in ('a/001.png', 'b/002.png', 'c/003.png'):
            path = self.data_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'')
            self.img_paths.append(path)


def test_parse_viz_manifest_reads_entries(tmp_path):
    dataset = _FakeValDataset(tmp_path)
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('b/002.png\n# comment\na/001.png\n', encoding='utf-8')

    indices = parse_viz_manifest(manifest, dataset, max_images=4, manifest_label='qroi')
    assert indices == [1, 0]


def test_parse_viz_manifest_requires_file(tmp_path):
    dataset = _FakeValDataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        parse_viz_manifest(tmp_path / 'missing.txt', dataset, max_images=4, manifest_label='heatmap')


def test_parse_viz_manifest_requires_valid_entries(tmp_path):
    dataset = _FakeValDataset(tmp_path)
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('# only comments\n', encoding='utf-8')
    with pytest.raises(ValueError):
        parse_viz_manifest(manifest, dataset, max_images=4, manifest_label='qroi')
