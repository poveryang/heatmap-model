import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from hmap.model.loss import FocalLoss
from hmap.model.csp_pafpn import CSPPAFPNNet
from hmap.utils import (
    visualize_batch_hmaps,
    visualize_batch_qroi,
    warmup_lr,
    hmap_to_bboxes,
    calc_confusion_matrix,
    dense_output_to_rotated_rois,
)


class HMapLitModel(LightningModule):
    def __init__(
            self,
            init_lr,
            gamma,
            alpha,
            object_classes=None,
            geometry_channels=0,
            enable_qroi=False,
            lambda_geo=1.0,
            lambda_qroi=1.0,
            qroi_pool_size=7,
            qroi_hidden_channels=None,
            qroi_eps=0.2,
            **model_conf):
        super().__init__()
        if 'out_channels' not in model_conf:
            raise ValueError('HMapLitModel requires model out_channels')
        self.object_classes = int(object_classes)
        self.geometry_channels = int(geometry_channels)
        self.enable_qroi = bool(enable_qroi)
        self.lambda_geo = float(lambda_geo)
        self.lambda_qroi = float(lambda_qroi)
        self.qroi_pool_size = int(qroi_pool_size)
        self.qroi_eps = float(qroi_eps)

        self.generator = CSPPAFPNNet(
            **model_conf,
            object_classes=self.object_classes,
            geometry_channels=self.geometry_channels)
        self.loss = FocalLoss(gamma, alpha)

        self.qroi_head = None
        if self.enable_qroi:
            feature_channels = getattr(
                self.generator,
                'feature_channels',
                model_conf.get('feature_channels'))
            if feature_channels is None:
                raise ValueError('Q_roi head requires generator.feature_channels or model feature_channels')
            feature_channels = int(feature_channels)
            hidden_channels = int(qroi_hidden_channels or max(feature_channels, 16))
            self.qroi_head = nn.Sequential(
                nn.Linear(feature_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1),
            )

        self.init_lr = float(init_lr)
        self.predict_dataloader = None
        self.heatmap_viz_dataloader = None
        self.qroi_viz_dataloader = None
        self.log_dir = None
        self.save_hyperparameters(
            ignore=['predict_dataloader', 'heatmap_viz_dataloader', 'qroi_viz_dataloader', 'log_dir'])

        self.test_step_outputs = []
        self.val_geo_outputs = []

    def forward(self, x, roi_boxes=None, return_dict=False):
        if self.enable_qroi or return_dict:
            dense_output, features = self.generator(x, return_features=True)
            q_roi = self.predict_qroi(features, roi_boxes) if roi_boxes is not None else None
            if return_dict or roi_boxes is not None:
                return {'dense_output': dense_output, 'features': features, 'q_roi': q_roi}
            return dense_output
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        batch_in_tensor, batch_gt = batch
        batch_size = batch_in_tensor.shape[0]
        if isinstance(batch_gt, dict):
            outputs = self(batch_in_tensor, roi_boxes=batch_gt['roi_boxes'], return_dict=True)
            loss, metrics = self.compute_geo_qroi_loss(outputs, batch_gt)
            self.log_metrics(metrics, prefix='train', batch_size=batch_size)
            return loss

        batch_hmap_tensor = self(batch_in_tensor)
        loss = self.loss(batch_hmap_tensor, batch_gt)
        self.log(
            'train_loss',
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_in_tensor, batch_gt = batch
        batch_size = batch_in_tensor.shape[0]
        logged_val_loss = False
        if isinstance(batch_gt, dict):
            outputs = self(batch_in_tensor, roi_boxes=batch_gt['roi_boxes'], return_dict=True)
            loss, metrics = self.compute_geo_qroi_loss(outputs, batch_gt)
            batch_hmap_tensor = outputs['dense_output'][:, :self.object_classes]
            batch_gt_tensor = batch_gt['object_hmap']
            geo_metrics = self.calc_geometry_metrics(
                outputs['dense_output'][:, self.object_classes:self.object_classes + self.geometry_channels],
                batch_gt)
            if geo_metrics is not None:
                self.val_geo_outputs.append(geo_metrics.detach())
            self.log_metrics(metrics, prefix='val', batch_size=batch_size)
            logged_val_loss = True
        else:
            batch_gt_tensor = batch_gt
            batch_hmap_tensor = self(batch_in_tensor)
            loss = self.loss(batch_hmap_tensor, batch_gt_tensor)
        if not logged_val_loss:
            self.log(
                'val_loss',
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                prog_bar=True)

        # calculate precision and recall
        batch_hmap_tensor = torch.sigmoid(batch_hmap_tensor)
        tp, fp, fn = 0, 0, 0
        for hmap_tensor, gt_tensor in zip(batch_hmap_tensor, batch_gt_tensor):
            hmap_bboxes = hmap_to_bboxes(hmap_tensor)
            gt_bboxes = hmap_to_bboxes(gt_tensor)
            tp_, fp_, fn_ = calc_confusion_matrix(hmap_bboxes, gt_bboxes)
            tp += tp_
            fp += fp_
            fn += fn_
        self.test_step_outputs.append(torch.tensor([tp, fp, fn], dtype=torch.float32))
        return loss

    def on_validation_epoch_end(self):
        totals = torch.stack(self.test_step_outputs).sum(dim=0).to(self.device)
        self.test_step_outputs = []
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        tp, fp, fn = totals[0], totals[1], totals[2]
        precision = tp / torch.clamp(tp + fp, min=1)
        recall = tp / torch.clamp(tp + fn, min=1)
        if self.trainer.is_global_zero:
            print(f'\nPrecision: {precision:.4f}, Recall: {recall:.4f}')
        self.log('val_precision', precision, sync_dist=True)
        self.log('val_recall', recall, sync_dist=True)
        if self.val_geo_outputs:
            geo_totals = torch.stack(self.val_geo_outputs).sum(dim=0).to(self.device)
            geo_count = torch.tensor(float(len(self.val_geo_outputs)), device=self.device)
            self.val_geo_outputs = []
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(geo_totals, op=dist.ReduceOp.SUM)
                dist.all_reduce(geo_count, op=dist.ReduceOp.SUM)
            center_err, size_err, angle_err = geo_totals / torch.clamp(geo_count, min=1)
            self.log('val_geo_center_error', center_err, sync_dist=True, prog_bar=False)
            self.log('val_geo_size_error', size_err, sync_dist=True, prog_bar=False)
            self.log('val_geo_angle_error', angle_err, sync_dist=True, prog_bar=False)

        if self.logger is None or not self.trainer.is_global_zero:
            return
        log_dir = self._get_log_dir()
        if self.heatmap_viz_dataloader is not None:
            batch_in_tensor, batch_gt = next(iter(self.heatmap_viz_dataloader()))
            batch_in_tensor = batch_in_tensor.to(self.device)
            batch_hmap_tensor = self.generator.forward(batch_in_tensor)
            if isinstance(batch_hmap_tensor, tuple):
                batch_hmap_tensor = batch_hmap_tensor[0]
            batch_hmap_tensor = batch_hmap_tensor[:, :self.object_classes]
            batch_hmap_tensor = torch.sigmoid(batch_hmap_tensor)
            visualize_hmaps = visualize_batch_hmaps(
                batch_hmap_tensor, batch_in_tensor, input_is_normalized=True)
            hmap_path = f'{log_dir}/hmap_{self.current_epoch}.png'
            cv2.imwrite(hmap_path, visualize_hmaps)
            self._log_viz_image(hmap_path, 'viz/heatmap')
        elif self.predict_dataloader is not None:
            batch_in_tensor, batch_img_tensor = next(iter(self.predict_dataloader()))
            batch_in_tensor = batch_in_tensor.to(self.device)
            batch_hmap_tensor = self.generator.forward(batch_in_tensor)
            if isinstance(batch_hmap_tensor, tuple):
                batch_hmap_tensor = batch_hmap_tensor[0]
            batch_hmap_tensor = batch_hmap_tensor[:, :self.object_classes]
            batch_hmap_tensor = torch.sigmoid(batch_hmap_tensor)
            visualize_hmaps = visualize_batch_hmaps(
                batch_hmap_tensor, batch_img_tensor)
            hmap_path = f'{log_dir}/hmap_{self.current_epoch}.png'
            cv2.imwrite(hmap_path, visualize_hmaps)
            self._log_viz_image(hmap_path, 'viz/heatmap')
        self._save_qroi_visualization()

    def test_step(self, batch, batch_idx):
        batch_in_tensor, batch_gt = batch
        if isinstance(batch_gt, dict):
            outputs = self(batch_in_tensor, roi_boxes=batch_gt['roi_boxes'], return_dict=True)
            batch_gt_tensor = batch_gt['object_hmap']
            batch_hmap_tensor = outputs['dense_output'][:, :self.object_classes]
        else:
            batch_gt_tensor = batch_gt
            batch_hmap_tensor = self(batch_in_tensor)
        batch_hmap_tensor = torch.sigmoid(batch_hmap_tensor)

        tp, fp, fn = 0, 0, 0
        for hmap_tensor, gt_tensor in zip(batch_hmap_tensor, batch_gt_tensor):
            hmap_bboxes = hmap_to_bboxes(hmap_tensor)
            gt_bboxes = hmap_to_bboxes(gt_tensor)
            tp_, fp_, fn_ = calc_confusion_matrix(hmap_bboxes, gt_bboxes)
            tp += tp_
            fp += fp_
            fn += fn_
        self.test_step_outputs.append(torch.tensor([tp, fp, fn], dtype=torch.float32))
        return tp, fp, fn

    def on_test_epoch_end(self):
        tp, fp, fn = torch.stack(self.test_step_outputs).mean(dim=0)
        self.test_step_outputs = []
        precision = tp / torch.clamp(tp + fp, min=1)
        recall = tp / torch.clamp(tp + fn, min=1)
        print(f'\nPrecision: {precision:.4f}, Recall: {recall:.4f}')

    def configure_optimizers(self):
        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs, warmup_factor=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [lr_scheduler]

    def set_predict_dataloader(self, dataloader):
        self.predict_dataloader = dataloader

    def set_heatmap_viz_dataloader(self, dataloader):
        self.heatmap_viz_dataloader = dataloader

    def set_qroi_viz_dataloader(self, dataloader):
        self.qroi_viz_dataloader = dataloader

    def set_log_dir(self, log_dir):
        self.log_dir = str(log_dir)

    def _get_log_dir(self):
        if self.log_dir:
            return self.log_dir
        if self.logger is not None and hasattr(self.logger, 'log_dir'):
            return self.logger.log_dir
        if self.trainer is not None and getattr(self.trainer, 'loggers', None):
            for logger in self.trainer.loggers:
                if hasattr(logger, 'log_dir') and logger.log_dir:
                    return logger.log_dir
        raise RuntimeError('No log directory configured for visualization output')

    def _log_viz_image(self, image_path, key):
        if self.trainer is None:
            return
        for logger in getattr(self.trainer, 'loggers', []):
            if logger.__class__.__name__ == 'WandbLogger':
                import wandb
                logger.experiment.log(
                    {key: wandb.Image(str(image_path))},
                    step=self.global_step)

    @torch.no_grad()
    def _save_qroi_visualization(self):
        if not self.enable_qroi or self.qroi_viz_dataloader is None:
            return
        batch_in_tensor, batch_gt = next(iter(self.qroi_viz_dataloader()))
        if not isinstance(batch_gt, dict):
            return
        batch_in_tensor = batch_in_tensor.to(self.device)
        roi_boxes = batch_gt['roi_boxes']
        if roi_boxes.numel() > 0:
            roi_boxes = roi_boxes.to(self.device)
        outputs = self(batch_in_tensor, roi_boxes=roi_boxes, return_dict=True)
        q_pred = outputs['q_roi']
        if q_pred is None:
            q_pred = torch.zeros((0,), device=batch_in_tensor.device)
        visualize_qroi = visualize_batch_qroi(
            batch_in_tensor.cpu(), batch_gt, q_pred.cpu())
        qroi_path = f'{self._get_log_dir()}/qroi_{self.current_epoch}.png'
        cv2.imwrite(qroi_path, visualize_qroi)
        self._log_viz_image(qroi_path, 'viz/qroi')

    def predict_qroi(self, features, roi_boxes):
        if self.qroi_head is None:
            return None
        if roi_boxes is None or roi_boxes.numel() == 0:
            return features.new_zeros((0,))
        pooled = rotated_roi_align(features, roi_boxes, output_size=self.qroi_pool_size)
        pooled = pooled.mean(dim=(-1, -2))
        return torch.sigmoid(self.qroi_head(pooled).squeeze(1))

    @torch.no_grad()
    def predict_instances(self, x, intensity_threshold=0.2, min_area=100):
        outputs = self(x, return_dict=True)
        dense_output = outputs['dense_output']
        instances = []
        for batch_idx, dense_item in enumerate(dense_output):
            proposals = dense_output_to_rotated_rois(
                dense_item,
                object_classes=self.object_classes,
                geometry_channels=self.geometry_channels,
                intensity_threshold=intensity_threshold,
                min_area=min_area)
            if proposals and self.qroi_head is not None:
                roi_boxes = dense_item.new_tensor(
                    [[batch_idx] + proposal['rrect'] for proposal in proposals],
                    dtype=dense_item.dtype)
                q_scores = self.predict_qroi(outputs['features'], roi_boxes).detach().cpu().tolist()
            else:
                q_scores = [1.0 for _ in proposals]
            for proposal, q_score in zip(proposals, q_scores):
                final_score = proposal['score'] * (self.qroi_eps + (1.0 - self.qroi_eps) * float(q_score))
                proposal['q_roi'] = float(q_score)
                proposal['final_score'] = float(final_score)
            instances.append(proposals)
        return instances

    def compute_geo_qroi_loss(self, outputs, target):
        dense_output = outputs['dense_output']
        object_logits = dense_output[:, :self.object_classes]
        object_loss = self.loss(object_logits, target['object_hmap'])

        geo_loss = dense_output.sum() * 0
        if self.geometry_channels:
            geo_pred = dense_output[:, self.object_classes:self.object_classes + self.geometry_channels]
            geo_target = target['geo_target']
            geo_weight = target['geo_weight']
            geo_raw = F.smooth_l1_loss(geo_pred, geo_target, reduction='none')
            denom = torch.clamp(geo_weight.sum(), min=1.0)
            geo_loss = (geo_raw * geo_weight).sum() / denom

        qroi_loss = dense_output.sum() * 0
        qroi_mae = dense_output.new_tensor(0.0)
        valid_qroi_ratio = dense_output.new_tensor(0.0)
        q_pred = outputs.get('q_roi')
        if q_pred is not None:
            q_target = target['roi_quality']
            q_mask = target['roi_quality_mask']
            if q_target.numel() > 0:
                valid_qroi_ratio = q_mask.mean()
                denom = torch.clamp(q_mask.sum(), min=1.0)
                q_raw = F.smooth_l1_loss(q_pred, q_target, reduction='none')
                qroi_loss = (q_raw * q_mask).sum() / denom
                if q_mask.sum() > 0:
                    qroi_mae = (torch.abs(q_pred - q_target) * q_mask).sum() / denom

        total = object_loss + self.lambda_geo * geo_loss + self.lambda_qroi * qroi_loss
        metrics = {
            'loss': total.detach(),
            'object_loss': object_loss.detach(),
            'geometry_loss': geo_loss.detach(),
            'qroi_loss': qroi_loss.detach(),
            'valid_qroi_ratio': valid_qroi_ratio.detach(),
            'qroi_mae': qroi_mae.detach(),
        }
        return total, metrics

    def log_metrics(self, metrics, prefix, batch_size):
        for name, value in metrics.items():
            self.log(
                f'{prefix}_{name}',
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                prog_bar=name in ('loss', 'object_loss', 'geometry_loss', 'qroi_loss'))

    def calc_geometry_metrics(self, geo_pred, target):
        roi_boxes = target['roi_boxes']
        if geo_pred.numel() == 0 or roi_boxes.numel() == 0:
            return None
        _, _, h, w = geo_pred.shape
        batch_idx = roi_boxes[:, 0].long()
        gt_cx = roi_boxes[:, 1]
        gt_cy = roi_boxes[:, 2]
        gt_w = roi_boxes[:, 3]
        gt_h = roi_boxes[:, 4]
        gt_angle = roi_boxes[:, 5]
        xs = torch.clamp(torch.round(gt_cx).long(), 0, w - 1)
        ys = torch.clamp(torch.round(gt_cy).long(), 0, h - 1)
        vals = geo_pred[batch_idx, :, ys, xs]
        pred_w = torch.exp(vals[:, 2]) * w
        pred_h = torch.exp(vals[:, 3]) * h
        pred_cx = xs.float() + vals[:, 0] * pred_w
        pred_cy = ys.float() + vals[:, 1] * pred_h
        pred_angle = 0.5 * torch.atan2(vals[:, 4], vals[:, 5]) * 180.0 / torch.pi

        center_err = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).mean()
        size_err = (torch.abs(pred_w - gt_w) + torch.abs(pred_h - gt_h)).mean() * 0.5
        angle_err = torch.abs(torch.remainder(pred_angle - gt_angle + 90.0, 180.0) - 90.0).mean()
        return torch.stack([center_err, size_err, angle_err])


def _build_rotated_roi_grid(cx, cy, box_w, box_h, angle, h, w, device, dtype, output_size):
    y_lin = torch.linspace(-0.5, 0.5, output_size, device=device, dtype=dtype)
    x_lin = torch.linspace(-0.5, 0.5, output_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
    local_x = grid_x.unsqueeze(0) * box_w.view(-1, 1, 1)
    local_y = grid_y.unsqueeze(0) * box_h.view(-1, 1, 1)

    theta = angle * torch.pi / 180.0
    cos_t = torch.cos(theta).view(-1, 1, 1)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    sample_x = cx.view(-1, 1, 1) + local_x * cos_t - local_y * sin_t
    sample_y = cy.view(-1, 1, 1) + local_x * sin_t + local_y * cos_t

    if w > 1:
        sample_x = sample_x / (w - 1) * 2 - 1
    else:
        sample_x = sample_x * 0
    if h > 1:
        sample_y = sample_y / (h - 1) * 2 - 1
    else:
        sample_y = sample_y * 0
    return torch.stack([sample_x, sample_y], dim=-1)


def rotated_roi_align(features, roi_boxes, output_size=7, max_rois_per_chunk=64):
    if roi_boxes is None or roi_boxes.numel() == 0:
        return features.new_zeros((0, features.shape[1], output_size, output_size))

    device = features.device
    dtype = features.dtype
    roi_boxes = roi_boxes.to(device=device, dtype=dtype)
    batch_idx = roi_boxes[:, 0].long().clamp(0, features.shape[0] - 1)
    _, _, h, w = features.shape

    outputs = []
    for batch_id in range(features.shape[0]):
        mask = batch_idx == batch_id
        if not mask.any():
            continue
        boxes = roi_boxes[mask]
        cx, cy, box_w, box_h, angle = [boxes[:, i] for i in range(1, 6)]
        grid = _build_rotated_roi_grid(cx, cy, box_w, box_h, angle, h, w, device, dtype, output_size)
        feature = features[batch_id:batch_id + 1]
        for start in range(0, grid.shape[0], max_rois_per_chunk):
            end = start + max_rois_per_chunk
            grid_chunk = grid[start:end]
            chunk_size = grid_chunk.shape[0]
            sampled = F.grid_sample(
                feature.expand(chunk_size, -1, -1, -1),
                grid_chunk,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True)
            outputs.append(sampled)
    if not outputs:
        return features.new_zeros((0, features.shape[1], output_size, output_size))
    return torch.cat(outputs, dim=0)
