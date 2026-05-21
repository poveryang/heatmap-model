from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image

valid_pids_map = {
    (0, 0, 0, 0): 0,  # 1d
    (8, 9, 10, 11): 0,  # 1d
    (1, 1, 2, 3): 1,  # qr
    (1, 1, 2, 3, 7, 7, 7): 1,  # qr
    (4, 4, 5, 6): 2  # dm
}


def draw_boxes(img, boxes):
    """
    Draw boxes
    Args:
        img: Image
        boxes: List of boxes, shape (n, 4, 2)

    Returns:
        img: Image
    """
    for box in boxes:
        img = draw_box_vertices(img, box)
    return img


def draw_box_vertices(img, box_vertices, color=(0, 255, 0), thickness=2):
    """
    Draw box vertices
    Args:
        img: Image
        box_vertices: Box points, shape (4, 2)
        color: Color
        thickness: Thickness

    Returns:
        img: Image
    """
    for i in range(4):
        cv2.line(img, tuple(map(int, box_vertices[i])), tuple(map(int, box_vertices[(i + 1) % 4])),
                 (0, 255, 0), 2)
        cv2.circle(img, tuple(map(int, box_vertices[i])), 2, (0, 0, 255), 2)
        cv2.putText(img, str(i), tuple(map(int, box_vertices[i])), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness)

    return img


def blend_heatmap(img, heatmap, alpha=0.5, clamp_min=0.0, clamp_max=1.0):
    # heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if clamp_min != 0.0 or clamp_max != 1.0:
        # heatmap = np.clip(heatmap, clamp_min, clamp_max)
        heatmap = np.where(clamp_min < heatmap, 255, 0).astype(np.uint8)
    else:
        heatmap = (heatmap * 255).astype(np.uint8)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_GRAY2RGB)
    blended_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return blended_img


def cal_mean_std(dir_name):
    """
    Statistics the mean and std of the dataset
    Args:
        dir_name: the root dir of the dataset
    """
    mean = np.zeros(1)
    std = np.zeros(1)
    all_img_paths = list(dir_name.rglob(r"**/*.png"))
    for img_path in tqdm.tqdm(all_img_paths, ncols=80):
        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.float64)
        img /= 255.0
        mean += img.mean()
    mean /= len(all_img_paths)

    for img_path in tqdm.tqdm(all_img_paths, ncols=80):
        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.float64)
        img /= 255.0
        diff = (img - mean).mean()
        std += diff * diff
    std /= len(all_img_paths)
    std = np.sqrt(std)

    print(f"mean: {mean}, std: {std}")
    with open(dir_name.parent / 'mean_std.txt', 'w', encoding='utf-8') as f:
        f.writelines(f"mean: {mean}, std: {std}")


def save_all_paths(dir_name, label_file):
    """
    Save all image paths and labels to a txt file
    Args:
        dir_name: the root dir of the dataset
        label_file: the txt file to save
    """
    dir_name = Path(dir_name)
    all_img_paths = list(dir_name.rglob(r"**/*.png"))
    with open(label_file, 'w', encoding='utf-8') as f:
        for img_path in all_img_paths:
            f.write(str('/'.join(img_path.parts[-3:])) + '\n')
    f.close()


def split_train_test(label_file, train_ratio=0.8, shuffle=True):
    train_label_file = label_file.replace('_all.txt', '_train.txt')
    test_label_file = label_file.replace('_all.txt', '_test.txt')

    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if shuffle:
        import random
        random.shuffle(lines)

    train_lines = lines[:int(len(lines) * train_ratio)]
    test_lines = lines[int(len(lines) * train_ratio):]

    with open(train_label_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(test_label_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)


if __name__ == '__main__':
    # cal_mean_std(r'D:\Barcode-Detection-Data\train')
    # save_all_paths(r'D:\Barcode-Detection-Data\image', r'D:\Barcode-Detection-Data\hmap_all.txt')
    # split_train_test(r'D:\Barcode-Detection-Data\hmap_all.txt')

    split_train_test(r"D:\Barcode-Detection-Data\bbox\bbox_all.txt")
