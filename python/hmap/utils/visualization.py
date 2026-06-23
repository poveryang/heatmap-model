import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.ops

id_cls_map = {0: 'Input image', 1: '1D heatmap', 2: 'QR heatmap', 3: 'DM heatmap'}
QROI_CLASS_NAMES = {0: 'BAR', 1: 'QR', 2: 'DM'}
DEFAULT_NORM_MEAN = 0.4330
DEFAULT_NORM_STD = 0.2349


def infer_grid_cols(num_panels):
    if num_panels <= 1:
        return 1
    cols = int(math.ceil(math.sqrt(num_panels)))
    while cols > 1 and num_panels % cols != 0:
        cols -= 1
    if num_panels % cols != 0:
        cols = int(math.ceil(math.sqrt(num_panels)))
    return cols


def stack_panels_grid(panels, ncols=None, pad=8, bg_color=(245, 245, 245)):
    if ncols is None:
        ncols = infer_grid_cols(len(panels))
    if not panels:
        return np.full((400, 640, 3), 128, dtype=np.uint8)
    if len(panels) == 1:
        return panels[0]

    target_w = max(panel.shape[1] for panel in panels)
    norm_panels = []
    for panel in panels:
        if panel.shape[1] != target_w:
            scale = target_w / panel.shape[1]
            new_h = max(1, int(round(panel.shape[0] * scale)))
            panel = cv2.resize(panel, (target_w, new_h), interpolation=cv2.INTER_AREA)
        norm_panels.append(panel)

    rows = []
    for row_start in range(0, len(norm_panels), ncols):
        row_panels = norm_panels[row_start:row_start + ncols]
        if len(row_panels) < ncols:
            blank = np.full_like(norm_panels[0], bg_color, dtype=np.uint8)
            row_panels = row_panels + [blank] * (ncols - len(row_panels))
        row_h = max(panel.shape[0] for panel in row_panels)
        padded_row = []
        for panel_idx, panel in enumerate(row_panels):
            if panel.shape[0] < row_h:
                panel = cv2.vconcat([
                    panel,
                    np.full((row_h - panel.shape[0], target_w, 3), bg_color, dtype=np.uint8),
                ])
            padded_row.append(panel)
            if panel_idx < len(row_panels) - 1:
                padded_row.append(np.full((row_h, pad, 3), bg_color, dtype=np.uint8))
        rows.append(cv2.hconcat(padded_row))

    stacked = rows[0]
    for row_idx in range(1, len(rows)):
        stacked = cv2.vconcat([
            stacked,
            np.full((pad, stacked.shape[1], 3), bg_color, dtype=np.uint8),
            rows[row_idx],
        ])
    return stacked


def visualize_batch_hmaps(batch_hmap_tensor, batch_img_tensor, input_is_normalized=False,
                          mean=DEFAULT_NORM_MEAN, std=DEFAULT_NORM_STD, grid_cols=1):
    hmaps_array = batch_hmap_tensor.detach().cpu().numpy()
    imgs_array = batch_img_tensor.detach().cpu().numpy()
    if input_is_normalized:
        imgs_array = imgs_array * std + mean

    imgs_with_hmap = []
    for hmap_array, img_array in zip(hmaps_array, imgs_array):
        fig, axes = plt.subplots(2, 4, figsize=(30, 6), gridspec_kw={'height_ratios': [30, 1]})
        blended_image = blend_img_with_hmap(hmap_array, img_array)
        img_ax = axes[0, 0]
        cbar_ax = axes[1, 0]
        img_ax.imshow(blended_image)
        img_ax.set_aspect('auto')
        img_ax.axis('off')
        img_ax.set_title(id_cls_map[0])
        cbar_ax.remove()
        for i in range(3):
            hmap_ax = axes[0, i + 1]
            cbar_ax = axes[1, i + 1]
            sns.heatmap(hmap_array[i], cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=1, ax=hmap_ax,
                        cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})

            hmap_ax.set_aspect('auto')
            hmap_ax.set_title(id_cls_map[i + 1])
            cbar_ax.set_aspect(1 / 40, adjustable='box')
            cbar_ax.tick_params(labelsize=12)

        fig.tight_layout()
        canvas = fig.canvas
        canvas.draw()
        fig_img = np.array(canvas.renderer.buffer_rgba())
        imgs_with_hmap.append(fig_img[:, :, :3])
        plt.close(fig)
    if len(imgs_with_hmap) == 1:
        visualize_hmaps = imgs_with_hmap[0]
    else:
        visualize_hmaps = stack_panels_grid(
            [cv2.cvtColor(panel, cv2.COLOR_RGB2BGR) for panel in imgs_with_hmap],
            ncols=grid_cols)
        return visualize_hmaps
    visualize_hmaps = cv2.cvtColor(visualize_hmaps, cv2.COLOR_RGB2BGR)
    return visualize_hmaps


def visualize_single_hmap(hmap_tensor, img_tensor):
    hmap_array = hmap_tensor.detach().cpu().numpy().squeeze(0)
    img_array = img_tensor.detach().cpu().numpy().squeeze(0)

    fig, axes = plt.subplots(2, 4, figsize=(30, 6), gridspec_kw={'height_ratios': [30, 1]})
    blended_image = blend_img_with_hmap(hmap_array, img_array)
    img_ax = axes[0, 0]
    cbar_ax = axes[1, 0]
    img_ax.imshow(blended_image)
    img_ax.set_aspect('auto')
    img_ax.axis('off')
    img_ax.set_title(id_cls_map[0])
    cbar_ax.remove()
    for i in range(3):
        hmap_ax = axes[0, i + 1]
        cbar_ax = axes[1, i + 1]
        sns.heatmap(hmap_array[i], cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=1, ax=hmap_ax,
                    cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})

        hmap_ax.set_aspect('auto')
        hmap_ax.set_title(id_cls_map[i + 1])
        cbar_ax.set_aspect(1 / 40, adjustable='box')
        cbar_ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return fig_img


def blend_img_with_hmap(image, heatmap):
    image = np.asarray(image, dtype=np.float32).transpose((1, 2, 0))
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 3 and heatmap.shape[0] in (1, 3):
        heatmap = heatmap.transpose((1, 2, 0))
    if heatmap.ndim == 2:
        heatmap = heatmap[..., None]
    if heatmap.shape[2] > 1:
        heatmap = heatmap.max(axis=2, keepdims=True)
    heatmap = np.repeat(heatmap, 3, axis=2)
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return blended_image


def draw_img_with_labels(image, target):
    label_name = {
        0: '1d',
        1: 'qr',
        2: 'dm'
    }
    if isinstance(target, dict):
        label_strs = [label_name[int(x)] for x in target['labels']]
        boxes = target['boxes']
    else:
        label_strs = [label_name[int(x[4])] for x in target]
        boxes = [list(map(int, x[:4])) for x in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)

    image = torch.asarray(image * 255, dtype=torch.uint8)
    image = torchvision.utils.draw_bounding_boxes(image, boxes, labels=label_strs, colors=(255, 0, 0), width=2)
    image = np.asarray(image).transpose((1, 2, 0))
    return image


def qroi_pred_to_bgr(q_roi):
    q = float(np.clip(q_roi, 0.0, 1.0))
    return (0, int(255 * q), int(255 * (1.0 - q)))


def rrect_to_corners(cx, cy, box_w, box_h, angle_deg):
    return cv2.boxPoints(((float(cx), float(cy)), (float(box_w), float(box_h)), float(angle_deg)))


def draw_dashed_line(image, pt1, pt2, color, thickness=2, gap=8):
    pt1 = np.asarray(pt1, dtype=np.float32)
    pt2 = np.asarray(pt2, dtype=np.float32)
    dist = float(np.linalg.norm(pt2 - pt1))
    if dist <= 1e-6:
        return
    direction = (pt2 - pt1) / dist
    pos = 0.0
    draw = True
    while pos < dist:
        end = min(pos + gap, dist)
        if draw:
            start_pt = tuple(np.round(pt1 + direction * pos).astype(int))
            end_pt = tuple(np.round(pt1 + direction * end).astype(int))
            cv2.line(image, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
        pos = end
        draw = not draw


def draw_rotated_roi(image, corners, color, thickness=2, dashed=False):
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if dashed:
        for idx in range(len(corners)):
            draw_dashed_line(
                image, corners[idx], corners[(idx + 1) % len(corners)], color, thickness=thickness)
    else:
        cv2.polylines(
            image,
            [corners.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA)


def denorm_input_tensor(image_tensor, mean=DEFAULT_NORM_MEAN, std=DEFAULT_NORM_STD):
    image = image_tensor.detach().cpu().float()
    if image.ndim == 3:
        image = image.squeeze(0)
    image = image * std + mean
    image = np.clip(image.numpy() * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)


def compute_qroi_viz_summary(gt_quality, pred_quality, quality_mask):
    mask = np.asarray(quality_mask, dtype=bool)
    gt = np.asarray(gt_quality, dtype=np.float32)
    pred = np.asarray(pred_quality, dtype=np.float32)
    valid_count = int(mask.sum())
    masked_count = int((~mask).sum())
    if valid_count == 0:
        return {
            'valid_count': valid_count,
            'masked_count': masked_count,
            'mae': None,
            'gt_mean': None,
            'pred_mean': None,
            'no_valid': True,
        }
    valid_gt = gt[mask]
    valid_pred = pred[mask]
    return {
        'valid_count': valid_count,
        'masked_count': masked_count,
        'mae': float(np.abs(valid_pred - valid_gt).mean()),
        'gt_mean': float(valid_gt.mean()),
        'pred_mean': float(valid_pred.mean()),
        'no_valid': False,
    }


def format_qroi_summary_text(summary):
    parts = [
        f"valid ROI: {summary['valid_count']}",
        f"masked ROI: {summary['masked_count']}",
    ]
    if summary['no_valid']:
        parts.append('No valid Q_roi')
    else:
        parts.extend([
            f"valid ROI MAE: {summary['mae']:.4f}",
            f"GT mean: {summary['gt_mean']:.4f}",
            f"Pred mean: {summary['pred_mean']:.4f}",
        ])
    return [' | '.join(parts)]


def draw_qroi_summary_panel(image, summary, origin=(12, 28), line_height=24):
    for line_idx, line in enumerate(format_qroi_summary_text(summary)):
        y = origin[1] + line_idx * line_height
        cv2.putText(
            image, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_qroi_on_image(image, roi_entries):
    for entry in roi_entries:
        corners = rrect_to_corners(
            entry['cx'], entry['cy'], entry['w'], entry['h'], entry['angle_deg'])
        class_name = QROI_CLASS_NAMES.get(int(entry['class_id']), str(entry['class_id']))
        if entry['quality_mask']:
            color = qroi_pred_to_bgr(entry['pred_q'])
            draw_rotated_roi(image, corners, color, thickness=2, dashed=False)
            err = abs(float(entry['pred_q']) - float(entry['gt_q']))
            label = (
                f"{class_name} id={entry['roi_id']} "
                f"GT={entry['gt_q']:.3f} Pred={entry['pred_q']:.3f} "
                f"|err|={err:.3f} mask=1")
        else:
            color = (160, 160, 160)
            draw_rotated_roi(image, corners, color, thickness=2, dashed=True)
            label = (
                f"{class_name} id={entry['roi_id']} "
                f"GT={entry['gt_q']:.3f} Pred={entry['pred_q']:.3f} MASKED")
        anchor = tuple(np.round(corners.min(axis=0)).astype(int))
        anchor = (max(anchor[0], 0), max(anchor[1], 14))
        cv2.putText(
            image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def collect_batch_qroi_entries(batch_gt, q_pred):
    roi_boxes = batch_gt['roi_boxes']
    if roi_boxes.numel() == 0:
        return {}

    if roi_boxes.shape[1] == 6:
        batch_indices = roi_boxes[:, 0].long()
        boxes = roi_boxes[:, 1:6]
    else:
        batch_indices = torch.zeros(roi_boxes.shape[0], dtype=torch.long)
        boxes = roi_boxes

    q_pred = q_pred.detach().cpu().float()
    grouped = {}
    for roi_idx in range(boxes.shape[0]):
        batch_idx = int(batch_indices[roi_idx].item())
        grouped.setdefault(batch_idx, []).append({
            'cx': float(boxes[roi_idx, 0]),
            'cy': float(boxes[roi_idx, 1]),
            'w': float(boxes[roi_idx, 2]),
            'h': float(boxes[roi_idx, 3]),
            'angle_deg': float(boxes[roi_idx, 4]),
            'class_id': int(batch_gt['roi_labels'][roi_idx].item()),
            'roi_id': int(batch_gt['roi_ids'][roi_idx].item()),
            'gt_q': float(batch_gt['roi_quality'][roi_idx].item()),
            'pred_q': float(q_pred[roi_idx].item()),
            'quality_mask': bool(batch_gt['roi_quality_mask'][roi_idx].item() > 0.5),
        })
    return grouped


def visualize_single_qroi(
        image_tensor, roi_entries, mean=DEFAULT_NORM_MEAN, std=DEFAULT_NORM_STD, draw_summary=True):
    image = denorm_input_tensor(image_tensor, mean=mean, std=std)
    annotate_qroi_on_image(image, roi_entries)
    gt = [entry['gt_q'] for entry in roi_entries]
    pred = [entry['pred_q'] for entry in roi_entries]
    mask = [entry['quality_mask'] for entry in roi_entries]
    summary = compute_qroi_viz_summary(gt, pred, mask)
    if draw_summary:
        draw_qroi_summary_panel(image, summary)
    return image, summary


def visualize_batch_qroi(batch_in_tensor, batch_gt, q_pred, mean=DEFAULT_NORM_MEAN, std=DEFAULT_NORM_STD,
                         grid_cols=None):
    grouped = collect_batch_qroi_entries(batch_gt, q_pred)
    batch_size = batch_in_tensor.shape[0]
    panels = []
    global_gt, global_pred, global_mask = [], [], []

    for batch_idx in range(batch_size):
        roi_entries = grouped.get(batch_idx, [])
        panel, _ = visualize_single_qroi(
            batch_in_tensor[batch_idx],
            roi_entries,
            mean=mean,
            std=std,
            draw_summary=False)
        panels.append(panel)
        for entry in roi_entries:
            global_gt.append(entry['gt_q'])
            global_pred.append(entry['pred_q'])
            global_mask.append(entry['quality_mask'])

    summary = compute_qroi_viz_summary(global_gt, global_pred, global_mask)
    if batch_size == 0:
        panel = np.full((400, 640, 3), 128, dtype=np.uint8)
        draw_qroi_summary_panel(panel, summary)
        return panel

    body = stack_panels_grid(panels, ncols=grid_cols)
    header_h = 40
    header_w = body.shape[1]
    header = np.full((header_h, header_w, 3), 245, dtype=np.uint8)
    draw_qroi_summary_panel(header, summary, origin=(12, 28))
    return cv2.vconcat([header, body])


if __name__ == '__main__':
    pass
