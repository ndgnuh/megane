import math
from itertools import count
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import ops


def prepare_input(image, W, H):
    image = image.resize((W, H), Image.Resampling.LANCZOS).convert("RGB")
    image_np = np.array(image, dtype="float32") / 255
    image_np = image_np.transpose(2, 0, 1)
    return image_np


def visualize(image, boxes, classes):
    image = image.copy()

    W, H = image.size
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 0, 0),
        (125, 0, 0),
        (0, 125, 0),
        (0, 0, 125),
    ]
    i = 0
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        x1 = x1 * W
        y1 = y1 * H
        x2 = x2 * W
        y2 = y2 * H
        outline = colors[c]
        i = i + 1
        draw.rectangle((x1, y1, x2, y2), outline=outline)
    return image


def visualize_feature_maps(maps, normalize=False):
    from matplotlib import pyplot as plt

    images = []
    w = 0
    h = 0
    for m in maps:
        img = np.concatenate(m, axis=-1)
        if normalize:
            a = img.min()
            b = img.max()
            img = np.nan_to_num((img - a) / (b - a))
        img = (img * 255).round().astype("uint8")
        image = Image.fromarray(img)
        images.append(image)
        w = w + image.width
        h = max(h, image.height)

    big = Image.new("L", (w, h), 0)
    x = 0
    for image in images:
        big.paste(image, (x, 0))
        x = x + image.width
    plt.imshow(big)
    plt.show()


def encode_fcos(
    image: Image.Image,
    boxes: List[Tuple[float, float, float, float]],
    classes: List[int],
    C: int,
    W: int = 800,
    H: int = 1024,
    strides: List[int] = [8, 16, 32, 64, 128],
    max_regression_thresholds: List[float] = [64, 128, 256, 512, 999999],
):
    # Create target maps at different scales
    centerness_maps = []
    regression_maps = []
    classification_maps = []
    training_maps = []
    for stride in strides:
        ft_width = int(W // stride)
        ft_height = int(H // stride)
        centerness_maps.append(np.zeros((C, ft_height, ft_width), dtype="float32"))
        regression_maps.append(np.zeros((C, 4, ft_height, ft_width), dtype="float32"))
        classification_maps.append(np.zeros((C, ft_height, ft_width), dtype="float32"))
        training_maps.append(np.zeros((C, ft_height, ft_width), dtype="bool"))

    # Denormalize bboxes
    boxes = [
        (
            int(x0 * W),
            int(y0 * H),
            int(x1 * W),
            int(y1 * H),
        )
        for (x0, y0, x1, y1) in boxes
    ]

    # Draw targets
    max_regression_thresholds_1 = [-1] + max_regression_thresholds[:-1]
    max_regression_thresholds_2 = max_regression_thresholds
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        # Select level
        # Use this instead of max to negate the extreme aspect ratio
        max_regression_value = math.sqrt(w * h)
        idx = 0
        stride = strides[0]
        for i, s, m1, m2 in zip(
            count(),
            strides,
            max_regression_thresholds_1,
            max_regression_thresholds_2,
        ):
            if m1 < max_regression_value and max_regression_value <= m2:
                stride = s
                idx = i
                break

        # metrics
        _, _, ft_height, ft_width = regression_maps[idx].shape
        x1_ = int(x1 / stride + 0.5)
        x2_ = int(x2 / stride + 0.5)
        y1_ = int(y1 / stride + 0.5)
        y2_ = int(y2 / stride + 0.5)
        if x1_ >= x2_ or y1_ >= y2_:
            continue
        w_ = x2_ - x1_
        h_ = y2_ - y1_

        # Grid
        xs = np.arange(x1_, x2_)[None, :]
        ys = np.arange(y1_, y2_)[:, None]
        xs = xs * stride + math.floor(stride / 2)
        ys = ys * stride + math.floor(stride / 2)

        # Regression
        l = (xs - x1) / stride
        r = (x2 - xs) / stride
        t = (ys - y1) / stride
        b = (y2 - ys) / stride
        reg = np.broadcast_arrays(l, t, r, b)
        reg = np.stack(reg, axis=0)

        # centerness
        ctn = np.fmin(l, r) * np.fmin(r, b) / np.fmax(l, r) / np.fmax(t, b)
        ctn = np.sqrt(ctn)
        if ctn.size == 0:
            continue
        ctn = ctn / ctn.max()

        # Draw
        x1_ = max(x1_, 0)
        y1_ = max(y1_, 0)
        x2_ = min(x2_, ft_width)
        y2_ = min(y2_, ft_height)
        c_pos = [slice(y1_, y2_), slice(x1_, x2_)]  # position on canvas
        t_pos = [slice(0, y2_ - y1_), slice(0, x2_ - x1_)]  # brush trim position
        centerness_maps[idx][c, *c_pos] = np.fmax(
            centerness_maps[idx][c, *c_pos],
            ctn[*t_pos],
        )
        regression_maps[idx][c, :, *c_pos] = reg[:, *t_pos]
        classification_maps[idx][c, *c_pos] = 1
        training_maps[idx][c, *c_pos] = True

    # Convert image to tensor
    image = image.resize((W, H), Image.Resampling.LANCZOS).convert("RGB")
    image_np = np.array(image, dtype="float32") / 255
    image_np = image_np.transpose(2, 0, 1)

    # visualize_feature_maps(regression_maps)
    # # visualize_feature_maps(classification_maps)
    # visualize_feature_maps(centerness_maps)
    return (
        image_np,
        regression_maps,
        centerness_maps,
        classification_maps,
        training_maps,
    )


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

    Args:
      box1 (List[int, int, int, int]): bounding boxes, sized (4,).
      box2 (List[int, int, int, int]): bounding boxes, sized (4,).

    Return:
      iou (float): IoU.
    """
    lt = np.zeros(2, "float32")
    rb = np.zeros(2, "float32")  # get inter-area left_top/right_bottom
    for i in range(2):
        if box1[i] > box2[i]:
            lt[i] = box1[i]
        else:
            lt[i] = box2[i]
        if box1[i + 2] < box2[i + 2]:
            rb[i] = box1[i + 2]
        else:
            rb[i] = box2[i + 2]
    wh = rb - lt
    wh[wh < 0] = 0  # if no overlapping
    inter = wh[0] * wh[1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter)
    return iou


def decode_fcos(
    regression_maps,
    centerness_maps,
    classification_maps,
    strides: List[int] = [8, 16, 32, 64, 128],
    min_score: float = 0.5,
    nms_threshold: float = 0.6,
):
    # Result containers
    boxes = []
    classes = []
    scores = []

    # Decode each level
    for rgs_map, ctn_map, cls_map, stride in zip(
        regression_maps,
        centerness_maps,
        classification_maps,
        strides,
    ):
        # Decode each class
        for class_id, cls_map_c, rgs_map_c, ctn_map_c in zip(
            count(), cls_map, rgs_map, ctn_map
        ):
            score_map = ctn_map_c * cls_map_c
            y, x = np.where(cls_map_c >= 0.5)
            scores_ = score_map[y, x]

            # Unpack regression map
            # l = (xs - x1) / stride -> x1 = xs - l * stride
            # r = (x2 - xs) / stride -> x2 = xs + r * stride
            l, t, r, b = rgs_map_c
            xs = x * stride + math.floor(stride / 2)
            ys = y * stride + math.floor(stride / 2)
            x1 = xs - l[y, x] * stride
            x2 = xs + r[y, x] * stride
            y1 = ys - t[y, x] * stride
            y2 = ys + b[y, x] * stride

            boxes_ = np.stack([x1, y1, x2, y2], axis=1)

            # No box, ignore
            if len(boxes_) == 0:
                continue

            # Ignore low score boxes
            mask = scores_ > min_score
            boxes_ = boxes_[mask]
            scores_ = scores_[mask]

            # NMS
            boxes_ = torch.tensor(boxes_.astype("float32"))
            scores_ = torch.tensor(scores_.astype("float32"))
            keep_ = ops.nms(boxes_, scores_, iou_threshold=nms_threshold)
            boxes_ = boxes_[keep_].tolist()
            scores_ = scores_[keep_].tolist()

            # Descale boxes
            H, W = cls_map_c.shape
            H = H * stride
            W = W * stride
            boxes_ = [(x1 / W, y1 / H, x2 / W, y2 / H) for (x1, y1, x2, y2) in boxes_]

            # Append results
            boxes.extend(boxes_)
            classes.extend([class_id] * len(boxes_))
            scores.extend(scores_)

    return boxes, classes, scores


if __name__ == "__main__":
    import random

    from icecream import ic
    from labelme import LabelMeDataset
    from matplotlib import pyplot as plt

    class2str = ["dog", "cat"]
    class2str = ["table", "column", "text"]

    dataset = LabelMeDataset("index.txt", class2str)
    dataset = LabelMeDataset("data/all.txt", class2str)

    image, gt_boxes, gt_classes = random.choice(dataset)
    # image, gt_boxes, gt_classes = dataset[0]
    image_np, rmaps, cmaps = encode_fcos(
        image, gt_boxes, gt_classes, C=len(class2str), W=800
    )
    boxes, classes, scores = decode_fcos(rmaps, cmaps)
    ic(boxes)
    ic(gt_boxes)

    im1 = visualize(image, boxes)
    im2 = visualize(image, gt_boxes)
    plt.subplot(1, 2, 1)
    plt.imshow(im2)
    plt.title("Ground truth")
    plt.subplot(1, 2, 2)
    plt.imshow(im1)
    plt.title("Decoded")
    plt.show()
