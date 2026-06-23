import torch

from hmap.model.hmap_model import _build_rotated_roi_grid, rotated_roi_align


def _rotated_roi_align_reference(features, roi_boxes, output_size=7):
    if roi_boxes is None or roi_boxes.numel() == 0:
        return features.new_zeros((0, features.shape[1], output_size, output_size))

    device = features.device
    dtype = features.dtype
    roi_boxes = roi_boxes.to(device=device, dtype=dtype)
    batch_idx = roi_boxes[:, 0].long().clamp(0, features.shape[0] - 1)
    cx, cy, box_w, box_h, angle = [roi_boxes[:, i] for i in range(1, 6)]
    _, _, h, w = features.shape
    grid = _build_rotated_roi_grid(cx, cy, box_w, box_h, angle, h, w, device, dtype, output_size)
    selected = features[batch_idx]
    return torch.nn.functional.grid_sample(
        selected, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def test_rotated_roi_align_matches_reference():
    torch.manual_seed(0)
    features = torch.randn(4, 8, 40, 64, requires_grad=True)
    roi_boxes = torch.tensor([
        [0, 12.0, 18.0, 20.0, 10.0, 15.0],
        [0, 30.0, 22.0, 16.0, 8.0, -20.0],
        [1, 8.0, 9.0, 12.0, 6.0, 45.0],
        [3, 50.0, 30.0, 24.0, 14.0, 90.0],
    ], dtype=torch.float32)
    output_size = 7

    ref = _rotated_roi_align_reference(features, roi_boxes, output_size=output_size)
    out = rotated_roi_align(features, roi_boxes, output_size=output_size, max_rois_per_chunk=2)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)

    loss = out.sum()
    loss.backward()
    features_ref = features.detach().clone().requires_grad_(True)
    ref = _rotated_roi_align_reference(features_ref, roi_boxes, output_size=output_size)
    ref.sum().backward()
    assert torch.allclose(features.grad, features_ref.grad, atol=1e-5, rtol=1e-5)
