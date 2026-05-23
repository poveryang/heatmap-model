import argparse
import sys
from datetime import datetime
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from hmap import CONFIGS_DIR, RUNS_DIR
from hmap.dataset import HMapDataModule
from hmap.model import HMapLitModel
from hmap.utils import load_configs, load_pl_model, set_callbacks


def make_run_dir(exp_name: str) -> Path:
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp_name}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(exp_name, pretrained_path=None, resume_path=None):
    """
    This is the main function to start training.

    Args:
        exp_name: experiment name
        pretrained_path: the path of pretrained model
        resume_path: the path of checkpoint to resume training

    Returns: None

    """

    # Load experiment configurations
    if resume_path is None:
        config_path = CONFIGS_DIR / f'{exp_name}.yaml'
    else:
        config_path = CONFIGS_DIR / f'{exp_name}-resume.yaml'
    configs = load_configs(config_path)

    # Set up data module
    datamodule = HMapDataModule(**configs['data'])
    datamodule.setup()

    # Initialize model
    hmap_model = load_pl_model(HMapLitModel, pretrained_path, **configs['model'])
    hmap_model.set_predict_dataloader(datamodule.predict_dataloader)

    run_dir = make_run_dir(exp_name)
    callbacks = set_callbacks(exp_name, run_dir)
    # name/version empty: metrics and plots go directly under run_dir
    logger = CSVLogger(save_dir=str(run_dir), name='', version='')
    trainer = Trainer(logger=logger, **configs['trainer'], callbacks=callbacks)

    print(f"Run directory: {run_dir}")

    # Start training
    trainer.fit(hmap_model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train heatmap model')
    parser.add_argument('--exp', default='hmap-smoke', help='Config name under python/configs/')
    parser.add_argument('--pretrained', help='Path to pretrained checkpoint')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args.exp, pretrained_path=args.pretrained, resume_path=args.resume)
