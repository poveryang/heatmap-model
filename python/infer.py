import argparse
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import cv2
import torch
from tqdm import tqdm

from hmap import CONFIGS_DIR
from hmap.dataset import HeatMapTransform
from hmap.model import HMapLitModel
from hmap.utils import load_configs, load_pl_model, visualize_single_hmap


def infer_single_image(image_path, hmap_model, hmap_transform, visualize=True, output_path=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Failed to read image: {image_path}')

    in_tensor, image_tensor = hmap_transform(image)
    in_tensor = in_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        dense_tensor = hmap_model(in_tensor)
        object_classes = getattr(hmap_model, 'object_classes', min(dense_tensor.shape[1], 3))
        hmap_tensor = torch.sigmoid(dense_tensor[:, :object_classes])
        instances = []
        if getattr(hmap_model, 'geometry_channels', 0) >= 6:
            instances = hmap_model.predict_instances(in_tensor)[0]

    if visualize or output_path is not None:
        vis_fig = visualize_single_hmap(hmap_tensor, image_tensor)
        vis_fig = cv2.cvtColor(vis_fig, cv2.COLOR_RGB2BGR)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_fig)
            if instances:
                roi_path = output_path.with_name(f'{output_path.stem}_rois{output_path.suffix}')
                cv2.imwrite(str(roi_path), draw_instances(image_tensor, instances))
        elif visualize:
            cv2.imshow('vis_fig', vis_fig)
            if instances:
                cv2.imshow('rois', draw_instances(image_tensor, instances))
            cv2.waitKey(0)

    return hmap_tensor, instances


def draw_instances(image_tensor, instances):
    image = image_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    image = (image * 255).clip(0, 255).astype('uint8')
    draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for inst in instances:
        cx, cy, w, h, angle = inst['rrect']
        pts = cv2.boxPoints(((cx, cy), (w, h), angle)).astype('int32')
        cv2.polylines(draw, [pts], True, (0, 255, 0), 2)
        text = f"c{inst['class_id']} q={inst.get('q_roi', 0):.2f} f={inst.get('final_score', 0):.2f}"
        cv2.putText(draw, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return draw


def infer_directory(input_dir, output_dir, hmap_model, input_size, visualize=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_paths = sorted(input_dir.rglob('*.png'))
    if not image_paths:
        raise ValueError(f'No PNG images found under: {input_dir}')

    hmap_transform = HeatMapTransform(input_size=input_size, img_aug=False, geo_aug=False)
    for image_path in tqdm(image_paths, desc='Inferring'):
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.png')
        infer_single_image(
            image_path, hmap_model, hmap_transform,
            visualize=visualize, output_path=output_path)

    return len(image_paths)


def main(exp_name, ckpt_path, image_path=None, input_dir=None, output_dir=None, visualize=True):
    configs = load_configs(CONFIGS_DIR / f'{exp_name}.yaml')
    hmap_model = load_pl_model(HMapLitModel, ckpt_path, **configs['model'])
    hmap_model.eval()
    input_size = tuple(configs['data']['input_size'])
    hmap_transform = HeatMapTransform(input_size=input_size, img_aug=False, geo_aug=False)

    if input_dir is not None:
        if output_dir is None:
            output_dir = Path(input_dir).parent / f'{Path(input_dir).name}-results'
        count = infer_directory(input_dir, output_dir, hmap_model, input_size, visualize=visualize)
        print(f'Finished inference on {count} images. Results saved to: {output_dir}')
        return

    if image_path is None:
        raise ValueError('Provide either --image or --input-dir.')

    infer_single_image(image_path, hmap_model, hmap_transform, visualize=visualize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Python heatmap inference on one image or a directory.')
    parser.add_argument('--exp', default='hmap-v2', help='Experiment config name.')
    parser.add_argument('--ckpt', required=True, help='Path to Lightning checkpoint.')
    parser.add_argument('--image', help='Path to input grayscale image.')
    parser.add_argument('--input-dir', help='Directory containing input PNG images.')
    parser.add_argument('--output-dir', help='Directory to save visualization results for batch inference.')
    parser.add_argument('--no-vis', action='store_true', help='Skip interactive OpenCV visualization.')
    args = parser.parse_args()
    main(
        args.exp, args.ckpt,
        image_path=args.image,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        visualize=not args.no_vis and args.input_dir is None,
    )
