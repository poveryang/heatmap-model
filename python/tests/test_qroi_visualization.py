import cv2
import numpy as np
import pytest
import torch

from hmap.utils.visualization import (
    compute_qroi_viz_summary,
    format_qroi_summary_text,
    rrect_to_corners,
    visualize_batch_qroi,
)


def _make_batch_gt():
    roi_boxes = torch.tensor([
        [0, 120.0, 80.0, 60.0, 30.0, 15.0],
        [0, 220.0, 140.0, 50.0, 40.0, -20.0],
        [1, 300.0, 200.0, 45.0, 35.0, 0.0],
    ], dtype=torch.float32)
    return {
        'object_hmap': torch.zeros((2, 3, 64, 64)),
        'geo_target': torch.zeros((2, 6, 64, 64)),
        'geo_weight': torch.zeros((2, 6, 64, 64)),
        'roi_boxes': roi_boxes,
        'roi_labels': torch.tensor([0, 1, 2], dtype=torch.long),
        'roi_quality': torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
        'roi_quality_mask': torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        'roi_ids': torch.tensor([0, 1, 2], dtype=torch.long),
    }


def test_rrect_to_corners_matches_cv2():
    cx, cy, box_w, box_h, angle = 120.5, 80.25, 60.0, 30.0, 15.0
    expected = cv2.boxPoints(((cx, cy), (box_w, box_h), angle))
    actual = rrect_to_corners(cx, cy, box_w, box_h, angle)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mae_ignores_masked_rois():
    summary = compute_qroi_viz_summary(
        gt_quality=[0.2, 0.8, 0.5],
        pred_quality=[0.3, 0.1, 0.7],
        quality_mask=[True, False, True],
    )
    assert summary['valid_count'] == 2
    assert summary['masked_count'] == 1
    assert summary['mae'] == pytest.approx((0.1 + 0.2) / 2)
    assert summary['gt_mean'] == pytest.approx((0.2 + 0.5) / 2)
    assert summary['pred_mean'] == pytest.approx((0.3 + 0.7) / 2)


def test_summary_marks_no_valid_qroi():
    summary = compute_qroi_viz_summary(
        gt_quality=[0.2, 0.8],
        pred_quality=[0.3, 0.1],
        quality_mask=[False, False],
    )
    assert summary['no_valid'] is True
    assert summary['valid_count'] == 0
    assert summary['masked_count'] == 2
    assert summary['mae'] is None


def test_format_qroi_summary_single_line():
    summary = compute_qroi_viz_summary(
        gt_quality=[0.2, 0.8],
        pred_quality=[0.3, 0.1],
        quality_mask=[True, True],
    )
    lines = format_qroi_summary_text(summary)
    assert len(lines) == 1
    assert ' | ' in lines[0]
    assert 'valid ROI MAE' in lines[0]


def test_visualize_batch_qroi_keeps_grid_width(tmp_path):
    batch_gt = _make_batch_gt()
    batch_in_tensor = torch.full((2, 1, 64, 64), 0.5, dtype=torch.float32)
    q_pred = torch.tensor([0.25, 0.9, 0.55], dtype=torch.float32)
    image = visualize_batch_qroi(batch_in_tensor, batch_gt, q_pred)
    assert image.shape[1] >= 64 * 2


def test_visualize_batch_qroi_writes_image(tmp_path):
    batch_gt = _make_batch_gt()
    batch_in_tensor = torch.full((2, 1, 64, 64), 0.5, dtype=torch.float32)
    q_pred = torch.tensor([0.25, 0.9, 0.55], dtype=torch.float32)
    image = visualize_batch_qroi(batch_in_tensor, batch_gt, q_pred)

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8

    output_path = tmp_path / 'qroi_test.png'
    assert cv2.imwrite(str(output_path), image)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_visualize_batch_qroi_no_valid_qroi(tmp_path):
    batch_gt = _make_batch_gt()
    batch_gt['roi_quality_mask'] = torch.zeros((3,), dtype=torch.float32)
    batch_in_tensor = torch.full((1, 1, 64, 64), 0.5, dtype=torch.float32)
    q_pred = torch.tensor([0.25, 0.9, 0.55], dtype=torch.float32)
    image = visualize_batch_qroi(batch_in_tensor, batch_gt, q_pred)

    summary = compute_qroi_viz_summary(
        batch_gt['roi_quality'].tolist(),
        q_pred.tolist(),
        batch_gt['roi_quality_mask'].tolist(),
    )
    assert summary['no_valid'] is True

    output_path = tmp_path / 'qroi_no_valid.png'
    assert cv2.imwrite(str(output_path), image)
