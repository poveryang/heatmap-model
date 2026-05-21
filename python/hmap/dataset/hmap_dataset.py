from pathlib import Path

import cv2
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from hmap.dataset.hmap_transform import HeatMapTransform


class HeatMapDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
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
        f = open(str(label_file), mode='r', encoding='utf-8')
        for item in f.readlines():
            # extract image path
            item_parts = item.strip().split(';')
            img_path = self.data_dir / item_parts[0]
            # extract instances
            instances_str = item_parts[1:]
            instances = []
            for instance in instances_str:
                instance_parts = instance.split(",")
                rrect = [float(r) for r in instance_parts[:-1]]
                label_id = int(instance_parts[-1])
                instance = rrect + [label_id]
                instances.append(instance)
            # load to list
            img_paths.append(img_path)
            labels.append(instances)
        return img_paths, labels


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


class HMapDataModule(LightningDataModule):
    def __init__(self, root_dir, input_size, batch_size, num_workers=8):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hmap_train = self.hmap_val = self.hmap_test = self.hmap_infer = None

    def setup(self, stage=None):
        # set image path
        train_dir = self.root_dir / 'train'
        val_dir = self.root_dir / 'test'
        test_dir = self.root_dir / 'test'
        infer_dir = self.root_dir / 'sample'
        # set transform
        train_transform = HeatMapTransform(self.input_size, img_aug=True, geo_aug=True)
        val_transform = HeatMapTransform(self.input_size, img_aug=False, geo_aug=False)
        # set dataset
        self.hmap_train = HeatMapDataset(train_dir, transform=train_transform)
        self.hmap_val = HeatMapDataset(val_dir, transform=val_transform)
        self.hmap_test = HeatMapDataset(test_dir, transform=val_transform)
        self.hmap_infer = HeatMapInferDataset(infer_dir, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.hmap_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.hmap_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.hmap_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def predict_dataloader(self):
        batch_size = min(8, len(self.hmap_infer)) or 1
        return DataLoader(self.hmap_infer, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
