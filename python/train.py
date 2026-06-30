import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch

torch.set_float32_matmul_precision('medium')

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from hmap import CONFIGS_DIR, RUNS_DIR
from hmap.dataset import HMapDataModule
from hmap.model import HMapLitModel
from hmap.utils import load_configs, load_pl_model, set_callbacks, trainer_kwargs_from_config


def make_run_dir(exp_name: str) -> Path:
    run_name = os.environ.get('HMAP_RUN_NAME')
    if not run_name:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp_name}"
        os.environ['HMAP_RUN_NAME'] = run_name
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loggers(run_dir: Path, exp_name: str, configs: dict):
    loggers = [CSVLogger(save_dir=str(run_dir), name='', version='')]
    wandb_conf = configs.get('wandb') or {}
    if wandb_conf.get('enabled', False):
        run_name = os.environ.get('HMAP_RUN_NAME', run_dir.name)
        loggers.append(WandbLogger(
            project=wandb_conf.get('project', 'heatmap-model'),
            entity=wandb_conf.get('entity'),
            name=wandb_conf.get('name') or run_name,
            save_dir=str(run_dir),
            log_model=wandb_conf.get('log_model', False),
            config=configs,
        ))
    return loggers


def main(exp_name, pretrained_path=None, resume_path=None, wandb_enabled=None):
    """
    This is the main function to start training.

    Args:
        exp_name: experiment name
        pretrained_path: the path of pretrained model
        resume_path: the path of checkpoint to resume training
        wandb_enabled: override wandb.enabled from config (True/False/None)

    Returns: None

    """

    # Load experiment configurations
    if resume_path is None:
        config_path = CONFIGS_DIR / f'{exp_name}.yaml'
    else:
        config_path = CONFIGS_DIR / f'{exp_name}-resume.yaml'
    configs = load_configs(config_path)
    if wandb_enabled is not None:
        configs.setdefault('wandb', {})['enabled'] = wandb_enabled

    # Set up data module
    datamodule = HMapDataModule(**configs['data'])
    datamodule.setup()

    # Initialize model
    hmap_model = load_pl_model(HMapLitModel, pretrained_path, **configs['model'])
    hmap_model.set_heatmap_viz_dataloader(datamodule.heatmap_viz_dataloader)
    hmap_model.set_qroi_viz_dataloader(datamodule.qroi_viz_dataloader)

    run_dir = make_run_dir(exp_name)
    hmap_model.set_log_dir(run_dir)
    callbacks = set_callbacks(exp_name, run_dir, configs.get('trainer'))
    loggers = build_loggers(run_dir, exp_name, configs)
    trainer = Trainer(logger=loggers, callbacks=callbacks, **trainer_kwargs_from_config(configs['trainer']))

    print(f"Run directory: {run_dir}")
    if any(logger.__class__.__name__ == 'WandbLogger' for logger in loggers):
        wandb_logger = next(logger for logger in loggers if logger.__class__.__name__ == 'WandbLogger')
        print(f"W&B run: {wandb_logger.experiment.url}")

    # Start training
    trainer.fit(hmap_model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train heatmap model')
    parser.add_argument('--exp', default='hmap-smoke', help='Config name under python/configs/')
    parser.add_argument('--pretrained', help='Path to pretrained checkpoint')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', dest='wandb_enabled', action='store_true', help='Enable W&B logging')
    parser.add_argument('--no-wandb', dest='wandb_enabled', action='store_false', help='Disable W&B logging')
    parser.set_defaults(wandb_enabled=None)
    args = parser.parse_args()
    main(args.exp, pretrained_path=args.pretrained, resume_path=args.resume, wandb_enabled=args.wandb_enabled)
