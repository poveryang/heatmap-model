from pathlib import Path
import json

import cv2
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch

from hmap.dataset.hmap_transform import HeatMapTransform


class HeatMapDataset(Dataset):
    def __init__(self, data_dir, transform=None, quality_path=None):
        self.data_dir = data_dir
        self.transform = transform
        self.quality_labels = self.load_quality_labels(quality_path)
        self.img_paths, self.labels = self.load_items()

    def __getitem__(self, idx):
        # load image and instances
        image = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        instances = self.labels[idx]
        # transform to image tensor and heatmap tensor
        image_tensor, hmap_tensor = self.transform(image, instances)

        return image_tensor, hmap_tensor

    def __len__(self):
        return len(self.img_paths)

    def load_items(self):
        img_paths, labels = [], []
        mode = self.data_dir.name
        label_file = self.data_dir / f'{mode}.txt'
        with open(str(label_file), mode='r', encoding='utf-8') as f:
            for item in f.readlines():
                # extract image path
                item_parts = item.strip().split(';')
                img_rel = item_parts[0]
                img_path = self.data_dir / img_rel
                image_id = f'{mode}/{img_rel}'
                # extract instances
                instances_str = item_parts[1:]
                instances = []
                for roi_id, instance in enumerate(instances_str):
                    instance_parts = instance.split(",")
                    rrect = [float(r) for r in instance_parts[:-1]]
                    label_id = int(instance_parts[-1])
                    q_roi, quality_mask = self.quality_labels.get((image_id, roi_id), (0.0, False))
                    instance = rrect + [label_id, roi_id, q_roi, quality_mask]
                    instances.append(instance)
                # load to list
                img_paths.append(img_path)
                labels.append(instances)
        return img_paths, labels

    @staticmethod
    def load_quality_labels(quality_path):
        if quality_path is None:
            return {}
        quality_path = Path(quality_path)
        if not quality_path.exists():
            raise FileNotFoundError(f'Quality label file not found: {quality_path}')

        labels = {}
        with open(str(quality_path), mode='r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_id = row.get('image_id')
                roi_id = row.get('roi_id')
                if image_id is None or roi_id is None:
                    raise ValueError(f'{quality_path}:{line_no}: missing image_id/roi_id')
                q_roi = row.get('q_roi', row.get('q_decode_free', row.get('quality_score')))
                quality_mask = row.get('quality_mask', q_roi is not None)
                labels[(image_id, int(roi_id))] = (
                    float(q_roi) if q_roi is not None else 0.0,
                    bool(quality_mask) and q_roi is not None,
                )
        return labels


class HeatMapInferDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.img_paths = list(Path(data_dir).rglob('*.png'))
        self.img_paths.sort()
        self.transform = transform

    def __getitem__(self, idx):
        # load image and instances
        image = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        norm_tensor, image_tensor = self.transform(image)
        return norm_tensor, image_tensor

    def __len__(self):
        return len(self.img_paths)


def hmap_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    if not isinstance(targets[0], dict):
        return images, torch.stack(targets, dim=0)

    collated = {}
    for key in ('object_hmap', 'geo_target', 'geo_weight'):
        collated[key] = torch.stack([target[key] for target in targets], dim=0)

    roi_boxes, roi_labels, roi_quality, roi_quality_mask, roi_ids = [], [], [], [], []
    for batch_idx, target in enumerate(targets):
        num_rois = target['roi_boxes'].shape[0]
        if num_rois == 0:
            continue
        batch_col = torch.full((num_rois, 1), batch_idx, dtype=torch.float32)
        roi_boxes.append(torch.cat([batch_col, target['roi_boxes']], dim=1))
        roi_labels.append(target['roi_labels'])
        roi_quality.append(target['roi_quality'])
        roi_quality_mask.append(target['roi_quality_mask'])
        roi_ids.append(target['roi_ids'])

    if roi_boxes:
        collated['roi_boxes'] = torch.cat(roi_boxes, dim=0)
        collated['roi_labels'] = torch.cat(roi_labels, dim=0)
        collated['roi_quality'] = torch.cat(roi_quality, dim=0)
        collated['roi_quality_mask'] = torch.cat(roi_quality_mask, dim=0)
        collated['roi_ids'] = torch.cat(roi_ids, dim=0)
    else:
        collated['roi_boxes'] = torch.zeros((0, 6), dtype=torch.float32)
        collated['roi_labels'] = torch.zeros((0,), dtype=torch.long)
        collated['roi_quality'] = torch.zeros((0,), dtype=torch.float32)
        collated['roi_quality_mask'] = torch.zeros((0,), dtype=torch.float32)
        collated['roi_ids'] = torch.zeros((0,), dtype=torch.long)
    return images, collated


class HMapDataModule(LightningDataModule):
    def __init__(
            self,
            root_dir,
            input_size,
            batch_size,
            num_workers=8,
            task='geo_qroi',
            quality_path=None,
            img_aug=True,
            geo_aug=False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.quality_path = quality_path
        self.img_aug = img_aug
        self.geo_aug = geo_aug
        self.hmap_train = self.hmap_val = self.hmap_test = self.hmap_infer = None

    def setup(self, stage=None):
        # set image path
        train_dir = self.root_dir / 'train'
        val_dir = self.root_dir / 'test'
        test_dir = self.root_dir / 'test'
        infer_dir = self.root_dir / 'sample'
        enable_geo_qroi = self.task in ('geo_qroi', 'geometry_qroi')
        output_mode = 'geo_qroi' if enable_geo_qroi else 'heatmap'
        # set transform
        train_transform = HeatMapTransform(
            self.input_size,
            img_aug=self.img_aug,
            geo_aug=(self.geo_aug and not enable_geo_qroi),
            output_mode=output_mode)
        val_transform = HeatMapTransform(self.input_size, img_aug=False, geo_aug=False, output_mode=output_mode)
        # set dataset
        quality_path = self.quality_path if enable_geo_qroi else None
        self.hmap_train = HeatMapDataset(train_dir, transform=train_transform, quality_path=quality_path)
        self.hmap_val = HeatMapDataset(val_dir, transform=val_transform, quality_path=quality_path)
        self.hmap_test = HeatMapDataset(test_dir, transform=val_transform, quality_path=quality_path)
        self.hmap_infer = HeatMapInferDataset(infer_dir, transform=val_transform)

    def train_dataloader(self):
        collate_fn = hmap_collate if self.task in ('geo_qroi', 'geometry_qroi') else None
        return DataLoader(self.hmap_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        collate_fn = hmap_collate if self.task in ('geo_qroi', 'geometry_qroi') else None
        return DataLoader(self.hmap_val, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        collate_fn = hmap_collate if self.task in ('geo_qroi', 'geometry_qroi') else None
        return DataLoader(self.hmap_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def predict_dataloader(self):
        batch_size = min(8, len(self.hmap_infer)) or 1
        return DataLoader(self.hmap_infer, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
