import cv2
import numpy as np
import torch


def hmap_to_bboxes(hmap_tensor, intensity_threshold=0.2):
    hmap_bboxes = []
    for ch_id, ch_hmap in enumerate(hmap_tensor):
        # Filter out the channel with the max intensity less than intensity_threshold
        if ch_hmap.max() < intensity_threshold:
            hmap_bboxes.append([])
            continue

        ch_hmap = ch_hmap.cpu().numpy()
        ch_hmap = (ch_hmap * 255).astype('uint8')  # convert to 8UC3

        # Adaptive thresholding
        _, binary_map = cv2.threshold(ch_hmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map.astype('uint8'), connectivity=8)
        ch_bboxes = []
        for i in range(1, num_labels):
            # Filter out the connected component with area less than 100
            if stats[i][4] < 100:
                continue

            # Filter out the connected component with mean intensity less than intensity_threshold
            mean_intensity = ch_hmap[labels == i].mean()
            if mean_intensity < (intensity_threshold * 255):
                continue

            # Find bounding box for each connected component
            x, y, w, h, _ = stats[i]
            ch_bboxes.append((x, y, w, h))
        hmap_bboxes.append(ch_bboxes)
    return hmap_bboxes


def calc_confusion_matrix(pred_boxes, gt_boxes, iou_threshold=0.5):
    # Initialize confusion matrix elements
    tp, fp, fn = 0, 0, 0

    # Iterate each channel
    for ch_pred_boxes, ch_gt_boxes in zip(pred_boxes, gt_boxes):
        ch_tp, ch_fp, ch_fn = calc_channel_confusion_matrix(ch_pred_boxes, ch_gt_boxes, iou_threshold)
        tp += ch_tp
        fp += ch_fp
        fn += ch_fn
    return tp, fp, fn


def calc_channel_confusion_matrix(pred_boxes, gt_boxes, iou_threshold=0.2):
    # Initialize confusion matrix elements
    tp, fp, fn = 0, 0, 0

    # Calculate True Positive and False Positive
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1  # True Positive
                gt_boxes.remove(gt_box)
                break
        else:
            fp += 1  # False Positive
    fn = len(gt_boxes)  # False Negative

    return tp, fp, fn


def calculate_iou(bbox1, bbox2):
    # Convert bbox to x, y, w, h
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Get the coordinates of intersection rectangle
    x1, y1, x2, y2 = max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    # Calculate the IOU
    if x1 >= x2 or y1 >= y2:
        return 0
    intersection = (x2 - x1) * (y2 - y1)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def dense_output_to_rotated_rois(
        dense_tensor,
        object_classes=3,
        geometry_channels=6,
        intensity_threshold=0.2,
        min_area=100):
    """Decode a single image dense output into rotated ROI proposals."""
    if dense_tensor.ndim != 3:
        raise ValueError(f'Expected CxHxW dense tensor, got shape {tuple(dense_tensor.shape)}')
    object_hmap = torch.sigmoid(dense_tensor[:object_classes]).detach().cpu()
    geo = dense_tensor[object_classes:object_classes + geometry_channels].detach().cpu()
    if geo.shape[0] < 6:
        return []

    _, h, w = dense_tensor.shape
    proposals = []
    for class_id, ch_hmap in enumerate(object_hmap):
        if ch_hmap.max() < intensity_threshold:
            continue
        ch_np = (ch_hmap.numpy() * 255).astype('uint8')
        _, binary_map = cv2.threshold(ch_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map.astype('uint8'), connectivity=8)
        for label_id in range(1, num_labels):
            if stats[label_id][4] < min_area:
                continue
            component_mask = labels == label_id
            mean_intensity = ch_np[component_mask].mean()
            if mean_intensity < (intensity_threshold * 255):
                continue
            ys, xs = np.where(component_mask)
            scores = ch_hmap[ys, xs]
            peak_idx = int(torch.argmax(scores))
            y = int(ys[peak_idx])
            x = int(xs[peak_idx])
            score = float(ch_hmap[y, x])
            vals = geo[:, y, x]
            roi_w = float(torch.exp(vals[2]) * w)
            roi_h = float(torch.exp(vals[3]) * h)
            cx = float(x + vals[0] * roi_w)
            cy = float(y + vals[1] * roi_h)
            angle = float(0.5 * torch.atan2(vals[4], vals[5]) * 180.0 / torch.pi)
            proposals.append({
                'class_id': class_id,
                'score': score,
                'peak': [x, y],
                'rrect': [cx, cy, roi_w, roi_h, angle],
            })
    return proposals
