import math
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.utils import make_grid


def load_configs(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def set_callbacks(exp_name, run_dir):
    model_ckpt = ModelCheckpoint(
        dirpath=str(run_dir / 'checkpoints'),
        filename=f'{exp_name}-{{epoch:03d}}-{{val_loss:.3e}}',
        monitor='val_loss', mode='min',
        save_top_k=150, save_weights_only=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    return [model_ckpt, lr_monitor]


def blend_image_hmap_tensor(img, hmap, alpha=0.5):
    img = img * 0.2349 + 0.4330
    hmap = torch.sigmoid(hmap)
    blended_batch = img * alpha + hmap * (1 - alpha)
    blended_batch = (blended_batch - blended_batch.min()) / \
                    (blended_batch.max() - blended_batch.min())
    blended_grid = make_grid(blended_batch, nrow=3)
    return blended_grid


def concat_image_hmap_tensor(img, hmap):
    img = img * 0.2349 + 0.4330
    img = img.repeat(1, 3, 1, 1)  # image n1hw -> n3hw
    hmap = torch.sigmoid(hmap)
    concat_batch = torch.cat([img, hmap], dim=0)
    concat_grid = make_grid(concat_batch, nrow=3)
    return concat_grid


def warmup_lr(max_epochs, warmup_epochs=None, warmup_factor=0.1):
    if not warmup_epochs:
        warmup_epochs = max_epochs // 20

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        else:
            return 1 / 2 * (1 + math.cos((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * math.pi))

    return lr_lambda
