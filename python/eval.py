import argparse
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from lightning import Trainer

from hmap import CONFIGS_DIR
from hmap.dataset import HMapDataModule
from hmap.model import HMapLitModel
from hmap.utils import load_configs, load_pl_model, trainer_kwargs_from_config


def main(exp_name, model_path=None):
    configs = load_configs(CONFIGS_DIR / f'{exp_name}.yaml')

    datamodule = HMapDataModule(**configs['data'])
    datamodule.setup()

    hmap_model = load_pl_model(HMapLitModel, model_path, **configs['model'])

    trainer_conf = trainer_kwargs_from_config(configs['trainer'])
    trainer_conf['devices'] = 1
    trainer = Trainer(**trainer_conf)
    trainer.test(hmap_model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a heatmap model.')
    parser.add_argument('--exp', default='hmap-v2', help='Experiment config name.')
    parser.add_argument('--ckpt', required=True, help='Path to Lightning checkpoint.')
    args = parser.parse_args()
    main(args.exp, args.ckpt)
