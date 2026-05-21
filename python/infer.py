import argparse
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import cv2
import torch

from hmap import CONFIGS_DIR
from hmap.dataset import HeatMapTransform
from hmap.model import HMapLitModel
from hmap.utils import load_configs, load_pl_model, visualize_single_hmap


def infer_single_image(image_path, hmap_model, visualize=True):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hmap_transform = HeatMapTransform(
        input_size=(400, 640), img_aug=False, geo_aug=False)
    in_tensor, image_tensor = hmap_transform(image)
    in_tensor = in_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    hmap_tensor = hmap_model(in_tensor)
    hmap_tensor = torch.sigmoid(hmap_tensor)

    if visualize:
        vis_fig = visualize_single_hmap(hmap_tensor, image_tensor)
        vis_fig = cv2.cvtColor(vis_fig, cv2.COLOR_RGB2BGR)
        cv2.imshow('vis_fig', vis_fig)
        cv2.waitKey(0)

    return hmap_tensor


def main(exp_name, ckpt_path, image_path, visualize=True):
    configs = load_configs(CONFIGS_DIR / f'{exp_name}.yaml')
    hmap_model = load_pl_model(HMapLitModel, ckpt_path, **configs['model'])
    hmap_model.eval()
    infer_single_image(image_path, hmap_model, visualize=visualize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Python heatmap inference on one image.')
    parser.add_argument('--exp', default='hmap-v2', help='Experiment config name.')
    parser.add_argument('--ckpt', required=True, help='Path to Lightning checkpoint.')
    parser.add_argument('--image', required=True, help='Path to input grayscale image.')
    parser.add_argument('--no-vis', action='store_true', help='Skip OpenCV visualization.')
    args = parser.parse_args()
    main(args.exp, args.ckpt, args.image, visualize=not args.no_vis)
