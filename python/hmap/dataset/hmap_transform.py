import random

import cv2
import numpy
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image, \
    adjust_brightness, adjust_contrast, gaussian_blur, \
    hflip, vflip, rotate, crop, resize


class HeatMapTransform(torch.nn.Module):
    def __init__(self, input_size, img_aug=False, geo_aug=False, mean=0.4330, std=0.2349):
        super(HeatMapTransform, self).__init__()
        self.input_size = input_size
        self.img_aug = img_aug
        self.geo_aug = geo_aug
        self.mean = [mean, ]
        self.std = [std, ]

    def forward(self, image, instances=None):
        # Random jitter image
        if self.img_aug:
            image = self.random_jitter(image)

        # Generate heatmap by instances and image
        if instances is not None:
            # Generate heatmap by instances and image
            hmap = self.generate_hmap(instances, image)
            gt_tensor = to_tensor(hmap)
            in_tensor = self.norm_to_tensor(image)
        else:
            gt_tensor = to_tensor(image)  # Set gt as image for convenience in inference when instances is None
            in_tensor = self.norm_to_tensor(image)

        # Random geometric transformation and resize image and heatmap
        if self.geo_aug:  # Resize image and heatmap
            in_tensor, gt_tensor = self.geometric_trans(in_tensor, gt_tensor)
        else:
            in_tensor, gt_tensor = self.aspect_resize(in_tensor, gt_tensor)

        return in_tensor, gt_tensor

    @staticmethod
    def random_jitter(image):
        # To tensor
        image = to_tensor(image)

        # Random adjust image's brightness, contrast, blur
        if random.random() < 0.2:
            factor = random.uniform(0.5, 2)
            image = adjust_brightness(image, factor)
        if random.random() < 0.2:
            factor = random.uniform(0.5, 2)
            image = adjust_contrast(image, factor)
        if random.random() < 0.2:
            ksize = random.choice((3, 7, 11))
            image = gaussian_blur(image, kernel_size=[ksize, ksize])

        # To numpy array
        image = to_pil_image(image)
        image = np.array(image)
        return image

    def norm_to_tensor(self, image):
        # ToTensor and Normalize
        image = to_tensor(image)
        image = normalize(image, mean=self.mean, std=self.std)
        return image

    def geometric_trans(self, img_tensor, hmap_tensor):
        # Random horizontal flip
        if random.random() < 0.5:
            img_tensor = hflip(img_tensor)
            hmap_tensor = hflip(hmap_tensor) if hmap_tensor is not None else None

        # Random vertical flip
        if random.random() < 0.5:
            img_tensor = vflip(img_tensor)
            hmap_tensor = vflip(hmap_tensor) if hmap_tensor is not None else None

        # Random rotation
        if random.random() < 0.2:
            angle = random.randint(-90, 90)
            img_tensor = rotate(img_tensor, angle)
            hmap_tensor = rotate(hmap_tensor, angle) if hmap_tensor is not None else None

        # Random crop resize
        if random.random() < 0.8:
            img_tensor, hmap_tensor = self.crop_resize(img_tensor, hmap_tensor)
        else:
            img_tensor, hmap_tensor = self.aspect_resize(img_tensor, hmap_tensor)
        return img_tensor, hmap_tensor

    def crop_resize(self, img_tensor, hmap_tensor=None, crop_prob=0.2):
        # random crop
        if random.random() < crop_prob:
            img_size = list(img_tensor.shape[-2:])

            crop_rate = random.uniform(0.7, 0.9)
            crop_size = [int(img_size[0] * crop_rate), int(img_size[1] * crop_rate)]

            offset_rate = random.uniform(-0.2, 0.2)
            dy, dx = [int(img_size[0] * offset_rate), int(img_size[1] * offset_rate)]

            img_tensor = crop(img_tensor, dy, dx, crop_size[0], crop_size[1])
            hmap_tensor = crop(hmap_tensor, dy, dx, crop_size[0], crop_size[1]) if hmap_tensor is not None else None
        # resize
        img_tensor = resize(img_tensor, [self.input_size[0], self.input_size[1]], antialias=True)
        hmap_tensor = resize(hmap_tensor, [self.input_size[0], self.input_size[1]], antialias=True) \
            if hmap_tensor is not None else None
        return img_tensor, hmap_tensor

    def aspect_resize(self, img_tensor, hmap_tensor=None):
        img_h, img_w = img_tensor.shape[-2:]
        scale = min(self.input_size[0] / img_h, self.input_size[1] / img_w)
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        dx = (self.input_size[1] - scaled_w) // 2
        dy = (self.input_size[0] - scaled_h) // 2

        scaled_img_tensor = resize(img_tensor, [scaled_h, scaled_w], antialias=True)
        dst_img_tensor = torch.full((1, self.input_size[0], self.input_size[1]), 0.5, dtype=torch.float32)
        dst_img_tensor[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_img_tensor

        scaled_hmap_tensor = resize(hmap_tensor, [scaled_h, scaled_w], antialias=True) \
            if hmap_tensor is not None else None
        if scaled_hmap_tensor is not None:
            dst_hmap_tensor = torch.zeros((3, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            dst_hmap_tensor[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_hmap_tensor
        else:
            dst_hmap_tensor = None

        return dst_img_tensor, dst_hmap_tensor

    def generate_hmap(self, instances, image: numpy.ndarray):
        img_h, img_w = image.shape[:2]
        x_range = np.arange(0, img_w)
        y_range = np.arange(0, img_h)
        x_map, y_map = np.meshgrid(x_range, y_range)

        heatmap = np.zeros((img_h, img_w, 3), dtype=np.float32)

        for instance in instances:
            # Finds the four vertices and rotation angle of the rotated rectangle
            x_ctr, y_ctr, rrect_w, rrect_h, angle_deg = instance[:5]
            rrect_pts = cv2.boxPoints(((x_ctr, y_ctr), (rrect_w, rrect_h), angle_deg))

            # Bounding-box of the rotated rectangle
            box_x, box_y, box_w, box_h = cv2.boundingRect(rrect_pts)
            box_x1 = max(box_x, 0)
            box_y1 = max(box_y, 0)
            box_x2 = min(box_x + box_w, img_w)
            box_y2 = min(box_y + box_h, img_h)
            if box_x1 >= box_x2 or box_y1 >= box_y2:
                continue

            # Mask of the rotated rectangle
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [rrect_pts.astype(np.int32)], 1)

            # Mask and crop the image, then calculate the instance weight
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            crop_img = masked_img[box_y1:box_y2, box_x1:box_x2]
            inst_weight = self.calc_instance_weight(crop_img)

            x = x_map[box_y1:box_y2, box_x1:box_x2]
            y = y_map[box_y1:box_y2, box_x1:box_x2]
            angle_rad = np.deg2rad(angle_deg)

            # The line function of the box's vertical axis, and distance to the axis
            a1 = np.cos(angle_rad)
            b1 = np.sin(angle_rad)
            c1 = -a1 * x_ctr - b1 * y_ctr
            d1 = np.abs(a1 * x + b1 * y + c1) / np.sqrt(a1 ** 2 + b1 ** 2)

            # The line function of the box's horizontal axis, and distance to the axis
            a2 = np.cos(angle_rad + np.pi / 2)
            b2 = np.sin(angle_rad + np.pi / 2)
            c2 = -a2 * x_ctr - b2 * y_ctr
            d2 = np.abs(a2 * x + b2 * y + c2) / np.sqrt(a2 ** 2 + b2 ** 2)

            # 3. Calculate the distance of each pixel to the box's axes line to generate the gaussian heatmap
            # adjust the kernel size according to the patch weight
            kernel_w = rrect_w * inst_weight
            kernel_h = rrect_h * inst_weight
            sigma1 = 0.3 * ((kernel_w - 1) * 0.5 - 1) + 0.8
            sigma2 = 0.3 * ((kernel_h - 1) * 0.5 - 1) + 0.8

            g1 = np.exp(-d1 ** 2 / (2 * sigma1 ** 2))  # gaussian distribution along the first axis
            g2 = np.exp(-d2 ** 2 / (2 * sigma2 ** 2))  # gaussian distribution along the second axis
            g = g1 * g2  # gaussian heatmap

            # 4. Mask the heatmap with box boundary
            inst_type = instance[5]
            heatmap[box_y1:box_y2, box_x1:box_x2, inst_type] = np.maximum(
                heatmap[box_y1:box_y2, box_x1:box_x2, inst_type], g)

        return heatmap

    @staticmethod
    def calc_instance_weight(gray_patch: numpy.ndarray):
        # otsu thresholding
        thres, _ = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # calculate mean value of pixels less than threshold and greater than threshold
        under_thresh = gray_patch[gray_patch <= thres]
        over_thresh = gray_patch[gray_patch > thres]

        mean_left = np.mean(under_thresh) if len(under_thresh) > 0 else 0
        mean_right = np.mean(over_thresh) if len(over_thresh) > 0 else 0
        contrast_ratio = (mean_right - mean_left) / 32
        contrast_weight = np.power(contrast_ratio, 0.2)
        contrast_weight = np.clip(contrast_weight, 0.2, 1.2)

        # calculate distribution of pixels less than threshold and greater than threshold
        n_under_thresh = len(under_thresh)
        n_over_thresh = len(over_thresh)
        balance_ratio = (n_under_thresh - n_over_thresh) / (n_under_thresh + n_over_thresh)
        balance_weight = 1.2 * np.power(np.cos(np.pi / 3 * balance_ratio), 0.8)
        balance_weight = np.clip(balance_weight, 0.5, 1.2)

        weight = contrast_weight * balance_weight
        return weight
