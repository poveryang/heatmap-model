import math
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torchvision.utils import make_grid


TRAINER_PASSTHROUGH_KEYS = frozenset({
    'accelerator', 'strategy', 'devices', 'num_nodes', 'precision', 'logger',
    'callbacks', 'fast_dev_run', 'max_epochs', 'min_epochs', 'max_steps', 'min_steps',
    'max_time', 'limit_train_batches', 'limit_val_batches', 'limit_test_batches',
    'limit_predict_batches', 'overfit_batches', 'val_check_interval',
    'check_val_every_n_epoch', 'num_sanity_val_steps', 'log_every_n_steps',
    'enable_checkpointing', 'enable_progress_bar', 'enable_model_summary',
    'accumulate_grad_batches', 'gradient_clip_val', 'gradient_clip_algorithm',
    'deterministic', 'benchmark', 'inference_mode', 'use_distributed_sampler',
    'profiler', 'detect_anomaly', 'barebones', 'plugins', 'sync_batchnorm',
    'reload_dataloaders_every_n_epochs', 'default_root_dir',
})


def trainer_kwargs_from_config(trainer_conf):
    trainer_conf = dict(trainer_conf or {})
    return {key: value for key, value in trainer_conf.items() if key in TRAINER_PASSTHROUGH_KEYS}


def load_configs(config_file):
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def set_callbacks(exp_name, run_dir, trainer_conf=None):
    trainer_conf = trainer_conf or {}
    save_top_k = int(trainer_conf.get('save_top_k', 150))
    model_ckpt = ModelCheckpoint(
        dirpath=str(run_dir / 'checkpoints'),
        filename=f'{exp_name}-{{epoch:03d}}-{{val_loss:.3e}}',
        monitor='val_loss', mode='min',
        save_top_k=save_top_k, save_weights_only=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [model_ckpt, lr_monitor]
    early_stopping = trainer_conf.get('early_stopping')
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor=early_stopping.get('monitor', 'val_loss'),
            mode=early_stopping.get('mode', 'min'),
            patience=int(early_stopping.get('patience', 15)),
            min_delta=float(early_stopping.get('min_delta', 1e-4)),
            check_on_train_epoch_end=False,
        ))
    return callbacks


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
