import random

import cv2
import numpy
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image, \
    adjust_brightness, adjust_contrast, gaussian_blur, \
    hflip, vflip, rotate, crop, resize


class HeatMapTransform(torch.nn.Module):
    def __init__(
            self,
            input_size,
            img_aug=False,
            geo_aug=False,
            mean=0.4330,
            std=0.2349,
            output_mode='heatmap',
            geometry_min_weight=0.05,
            square_angle_weight=0.2):
        super(HeatMapTransform, self).__init__()
        self.input_size = input_size
        self.img_aug = img_aug
        self.geo_aug = geo_aug
        self.mean = [mean, ]
        self.std = [std, ]
        self.output_mode = output_mode
        self.geometry_min_weight = geometry_min_weight
        self.square_angle_weight = square_angle_weight

    def forward(self, image, instances=None):
        # Random jitter image
        if self.img_aug:
            image = self.random_jitter(image)

        if instances is not None and self.output_mode == 'geo_qroi':
            resized_image, resized_instances = self.aspect_resize_image_and_instances(image, instances)
            in_tensor = self.norm_to_tensor(resized_image)
            target = self.generate_geo_qroi_targets(resized_instances, resized_image)
            return in_tensor, target

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

    def aspect_resize_image_and_instances(self, image: numpy.ndarray, instances):
        img_h, img_w = image.shape[:2]
        out_h, out_w = self.input_size
        scale = min(out_h / img_h, out_w / img_w)
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        dx = (out_w - scaled_w) // 2
        dy = (out_h - scaled_h) // 2

        resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        dst = np.full((out_h, out_w), 128, dtype=image.dtype)
        dst[dy:dy + scaled_h, dx:dx + scaled_w] = resized

        resized_instances = []
        for instance in instances:
            inst = self.parse_instance(instance)
            x_ctr = inst['x_ctr'] * scale + dx
            y_ctr = inst['y_ctr'] * scale + dy
            box_w = inst['w'] * scale
            box_h = inst['h'] * scale
            x_ctr, y_ctr, box_w, box_h, angle_deg = self.canonical_rrect(
                x_ctr, y_ctr, box_w, box_h, inst['angle_deg'])
            resized_instances.append({
                **inst,
                'x_ctr': x_ctr,
                'y_ctr': y_ctr,
                'w': box_w,
                'h': box_h,
                'angle_deg': angle_deg,
            })
        return dst, resized_instances

    @staticmethod
    def parse_instance(instance):
        if isinstance(instance, dict):
            return {
                'x_ctr': float(instance['x_ctr']),
                'y_ctr': float(instance['y_ctr']),
                'w': float(instance['w']),
                'h': float(instance['h']),
                'angle_deg': float(instance['angle_deg']),
                'class_id': int(instance['class_id']),
                'roi_id': int(instance.get('roi_id', 0)),
                'q_roi': float(instance.get('q_roi', 0.0)),
                'quality_mask': bool(instance.get('quality_mask', False)),
            }
        q_roi = float(instance[7]) if len(instance) > 7 else 0.0
        quality_mask = bool(instance[8]) if len(instance) > 8 else False
        roi_id = int(instance[6]) if len(instance) > 6 else 0
        return {
            'x_ctr': float(instance[0]),
            'y_ctr': float(instance[1]),
            'w': float(instance[2]),
            'h': float(instance[3]),
            'angle_deg': float(instance[4]),
            'class_id': int(instance[5]),
            'roi_id': roi_id,
            'q_roi': q_roi,
            'quality_mask': quality_mask,
        }

    @staticmethod
    def normalize_angle(angle_deg):
        while angle_deg < -90:
            angle_deg += 180
        while angle_deg >= 90:
            angle_deg -= 180
        return angle_deg

    @classmethod
    def canonical_rrect(cls, x_ctr, y_ctr, box_w, box_h, angle_deg):
        if box_w < box_h:
            box_w, box_h = box_h, box_w
            angle_deg += 90
        return x_ctr, y_ctr, box_w, box_h, cls.normalize_angle(angle_deg)

    def generate_geo_qroi_targets(self, instances, image: numpy.ndarray):
        img_h, img_w = image.shape[:2]
        object_hmap = np.zeros((img_h, img_w, 3), dtype=np.float32)
        geo_target = np.zeros((6, img_h, img_w), dtype=np.float32)
        geo_weight = np.zeros((6, img_h, img_w), dtype=np.float32)
        assign_weight = np.zeros((img_h, img_w), dtype=np.float32)

        roi_boxes, roi_labels, roi_quality, roi_quality_mask, roi_ids = [], [], [], [], []

        x_range = np.arange(0, img_w)
        y_range = np.arange(0, img_h)
        x_map, y_map = np.meshgrid(x_range, y_range)

        for inst in instances:
            x_ctr, y_ctr, rrect_w, rrect_h, angle_deg = (
                inst['x_ctr'], inst['y_ctr'], inst['w'], inst['h'], inst['angle_deg'])
            class_id = int(inst['class_id'])
            if class_id < 0 or class_id >= object_hmap.shape[-1]:
                continue

            rrect_pts = cv2.boxPoints(((x_ctr, y_ctr), (rrect_w, rrect_h), angle_deg))
            box_x, box_y, box_w, box_h = cv2.boundingRect(rrect_pts)
            box_x1 = max(box_x, 0)
            box_y1 = max(box_y, 0)
            box_x2 = min(box_x + box_w, img_w)
            box_y2 = min(box_y + box_h, img_h)
            if box_x1 >= box_x2 or box_y1 >= box_y2:
                continue

            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [rrect_pts.astype(np.int32)], 1)
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            crop_img = masked_img[box_y1:box_y2, box_x1:box_x2]
            inst_weight = self.calc_instance_weight(crop_img)

            x = x_map[box_y1:box_y2, box_x1:box_x2]
            y = y_map[box_y1:box_y2, box_x1:box_x2]
            angle_rad = np.deg2rad(angle_deg)

            a1 = np.cos(angle_rad)
            b1 = np.sin(angle_rad)
            c1 = -a1 * x_ctr - b1 * y_ctr
            d1 = np.abs(a1 * x + b1 * y + c1) / np.sqrt(a1 ** 2 + b1 ** 2)

            a2 = np.cos(angle_rad + np.pi / 2)
            b2 = np.sin(angle_rad + np.pi / 2)
            c2 = -a2 * x_ctr - b2 * y_ctr
            d2 = np.abs(a2 * x + b2 * y + c2) / np.sqrt(a2 ** 2 + b2 ** 2)

            kernel_w = rrect_w * inst_weight
            kernel_h = rrect_h * inst_weight
            sigma1 = 0.3 * ((kernel_w - 1) * 0.5 - 1) + 0.8
            sigma2 = 0.3 * ((kernel_h - 1) * 0.5 - 1) + 0.8
            g = np.exp(-d1 ** 2 / (2 * sigma1 ** 2)) * np.exp(-d2 ** 2 / (2 * sigma2 ** 2))

            object_crop = object_hmap[box_y1:box_y2, box_x1:box_x2, class_id]
            object_hmap[box_y1:box_y2, box_x1:box_x2, class_id] = np.maximum(object_crop, g)

            assign_crop_weight = assign_weight[box_y1:box_y2, box_x1:box_x2]
            assign = (g > self.geometry_min_weight) & (g > assign_crop_weight)
            if np.any(assign):
                safe_w = max(rrect_w, 1.0)
                safe_h = max(rrect_h, 1.0)
                theta = np.deg2rad(angle_deg)
                aspect_ratio = min(rrect_w, rrect_h) / max(rrect_w, rrect_h)
                angle_scale = self.square_angle_weight if aspect_ratio > 0.9 else 1.0
                values = np.stack([
                    (x_ctr - x) / safe_w,
                    (y_ctr - y) / safe_h,
                    np.full_like(x, np.log(safe_w / img_w), dtype=np.float32),
                    np.full_like(x, np.log(safe_h / img_h), dtype=np.float32),
                    np.full_like(x, np.sin(2 * theta), dtype=np.float32),
                    np.full_like(x, np.cos(2 * theta), dtype=np.float32),
                ], axis=0)
                for ch in range(6):
                    target_view = geo_target[ch, box_y1:box_y2, box_x1:box_x2]
                    weight_view = geo_weight[ch, box_y1:box_y2, box_x1:box_x2]
                    target_view[assign] = values[ch][assign]
                    ch_weight = g * angle_scale if ch >= 4 else g
                    weight_view[assign] = ch_weight[assign]
                assign_crop_weight[assign] = g[assign]

            roi_boxes.append([x_ctr, y_ctr, rrect_w, rrect_h, angle_deg])
            roi_labels.append(class_id)
            roi_quality.append(float(inst.get('q_roi', 0.0)))
            roi_quality_mask.append(float(bool(inst.get('quality_mask', False))))
            roi_ids.append(int(inst.get('roi_id', len(roi_ids))))

        if roi_boxes:
            roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32)
            roi_labels = torch.tensor(roi_labels, dtype=torch.long)
            roi_quality = torch.tensor(roi_quality, dtype=torch.float32)
            roi_quality_mask = torch.tensor(roi_quality_mask, dtype=torch.float32)
            roi_ids = torch.tensor(roi_ids, dtype=torch.long)
        else:
            roi_boxes = torch.zeros((0, 5), dtype=torch.float32)
            roi_labels = torch.zeros((0,), dtype=torch.long)
            roi_quality = torch.zeros((0,), dtype=torch.float32)
            roi_quality_mask = torch.zeros((0,), dtype=torch.float32)
            roi_ids = torch.zeros((0,), dtype=torch.long)

        return {
            'object_hmap': to_tensor(object_hmap),
            'geo_target': torch.from_numpy(geo_target),
            'geo_weight': torch.from_numpy(geo_weight),
            'roi_boxes': roi_boxes,
            'roi_labels': roi_labels,
            'roi_quality': roi_quality,
            'roi_quality_mask': roi_quality_mask,
            'roi_ids': roi_ids,
        }

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
