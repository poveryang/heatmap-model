import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.ops

id_cls_map = {0: 'Input image', 1: '1D heatmap', 2: 'QR heatmap', 3: 'DM heatmap'}


def visualize_batch_hmaps(batch_hmap_tensor, batch_img_tensor):
    # TODO: call visualize_single_hmap recursively
    hmaps_array = batch_hmap_tensor.detach().cpu().numpy()
    imgs_array = batch_img_tensor.detach().cpu().numpy()

    imgs_with_hmap = []
    for hmap_array, img_array in zip(hmaps_array, imgs_array):
        fig, axes = plt.subplots(2, 4, figsize=(30, 6), gridspec_kw={'height_ratios': [30, 1]})
        blended_image = blend_img_with_hmap(hmap_array, img_array)
        img_ax = axes[0, 0]
        cbar_ax = axes[1, 0]
        img_ax.imshow(blended_image)
        img_ax.set_aspect('auto')
        img_ax.axis('off')
        img_ax.set_title(id_cls_map[0])
        cbar_ax.remove()
        for i in range(3):
            hmap_ax = axes[0, i + 1]
            cbar_ax = axes[1, i + 1]
            sns.heatmap(hmap_array[i], cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=1, ax=hmap_ax,
                        cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})

            hmap_ax.set_aspect('auto')
            hmap_ax.set_title(id_cls_map[i + 1])
            cbar_ax.set_aspect(1 / 40, adjustable='box')
            cbar_ax.tick_params(labelsize=12)

        fig.tight_layout()
        canvas = fig.canvas
        canvas.draw()
        fig_img = np.array(canvas.renderer.buffer_rgba())
        imgs_with_hmap.append(fig_img[:, :, :3])
    if len(imgs_with_hmap) == 1:
        visualize_hmaps = imgs_with_hmap[0]
    else:
        visualize_hmaps = np.concatenate(imgs_with_hmap, axis=0)
    visualize_hmaps = cv2.cvtColor(visualize_hmaps, cv2.COLOR_RGB2BGR)
    return visualize_hmaps


def visualize_single_hmap(hmap_tensor, img_tensor):
    hmap_array = hmap_tensor.detach().cpu().numpy().squeeze(0)
    img_array = img_tensor.detach().cpu().numpy().squeeze(0)

    fig, axes = plt.subplots(2, 4, figsize=(30, 6), gridspec_kw={'height_ratios': [30, 1]})
    blended_image = blend_img_with_hmap(hmap_array, img_array)
    img_ax = axes[0, 0]
    cbar_ax = axes[1, 0]
    img_ax.imshow(blended_image)
    img_ax.set_aspect('auto')
    img_ax.axis('off')
    img_ax.set_title(id_cls_map[0])
    cbar_ax.remove()
    for i in range(3):
        hmap_ax = axes[0, i + 1]
        cbar_ax = axes[1, i + 1]
        sns.heatmap(hmap_array[i], cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=1, ax=hmap_ax,
                    cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})

        hmap_ax.set_aspect('auto')
        hmap_ax.set_title(id_cls_map[i + 1])
        cbar_ax.set_aspect(1 / 40, adjustable='box')
        cbar_ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    return fig_img


def blend_img_with_hmap(image, heatmap):
    image = image.transpose((1, 2, 0))
    heatmap = heatmap.transpose((1, 2, 0))
    blended_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return blended_image


def draw_img_with_labels(image, target):
    label_name = {
        0: '1d',
        1: 'qr',
        2: 'dm'
    }
    if isinstance(target, dict):
        label_strs = [label_name[int(x)] for x in target['labels']]
        boxes = target['boxes']
    else:
        label_strs = [label_name[int(x[4])] for x in target]
        boxes = [list(map(int, x[:4])) for x in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)

    image = torch.asarray(image * 255, dtype=torch.uint8)
    image = torchvision.utils.draw_bounding_boxes(image, boxes, labels=label_strs, colors=(255, 0, 0), width=2)
    image = np.asarray(image).transpose((1, 2, 0))
    return image


if __name__ == '__main__':
    pass
