from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import ops


def visualize(image, boxes):
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
    for x1, y1, x2, y2 in boxes:
        x1 = x1 * W
        y1 = y1 * H
        x2 = x2 * W
        y2 = y2 * H
        outline = colors[i % len(colors)]
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
    W: int = 1024,
    H: int = 1024,
    s: List[int] = [8, 16, 32, 64, 128],
    m: List[float] = [64, 128, 256, 512, 999999],
):
    """
    Encode the ground truth boxes and classes into FCOS target maps.

    Args:
        image (Image.Image):
            The input image in Pillow format.
        boxes (List[Tuple[float, float, float, float]]):
            List of bounding boxes in (x0, y0, x1, y1) format.
        classes (List[int]):
            List of class indices corresponding to each bounding box.
        C (int):
            The number of classes.
        W (int, optional):
            The width of the target maps. Default is 1024.
        H (int, optional):
            The height of the target maps. Default is 1024.
        s (List[int], optional):
            List of strides for each scale level. Default is [8, 16, 32, 64, 128].
        m (List[float], optional):
            List of maximum regression values for each scale level. Default is [64, 128, 256, 512, 999999].

    Returns:
        image (np.ndarray):
            Image in tensor format, shape (C, H, W), value in 0..1.
        regression_maps (List[np.ndarray]):
            Regression target feature maps of all strides.
        centerness_maps (List[np.ndarray]):
            Centerness target feature maps of all strides.
    """
    centerness_maps = []
    regression_maps = []

    # Denormalize bounding boxes
    boxes = [
        (
            int(x0 * W),
            int(y0 * H),
            int(x1 * W),
            int(y1 * H),
        )
        for (x0, y0, x1, y1) in boxes
    ]

    # Create different target maps at current scales
    for stride in s:
        ft_width = W // stride
        ft_height = H // stride
        centerness_maps.append(np.zeros((C, ft_height, ft_width), dtype="float32"))
        regression_maps.append(np.zeros((4, ft_height, ft_width), dtype="float32"))

    # Draw target maps
    m_prev = [0] + m[:-1]
    for (x0, y0, x1, y1), c in zip(boxes, classes):
        # Multi level prediction assignment
        for i, (stride, m_i1, m_i) in enumerate(zip(s, m_prev, m)):
            _, Hs, Ws = regression_maps[i].shape

            # Scale bboxes
            x0s = int(x0 / stride + 0.5)
            y0s = int(y0 / stride + 0.5)
            x1s = int(x1 / stride + 0.5)
            y1s = int(y1 / stride + 0.5)

            # Select stride level
            max_regression_value = max(x1 - x0, y1 - y0)
            if max_regression_value >= m_i or max_regression_value < m_i1:
                continue

            # Grid
            xs = np.arange(x0s, x1s)[None, :]
            ys = np.arange(y0s, y1s)[:, None]
            xs = xs * stride + stride // 2
            ys = ys * stride + stride // 2

            # Box encode
            l = xs - x0
            t = ys - y0
            r = x1 - xs
            b = y1 - ys

            # Draw Position
            x0s = max(x0s, 0)
            y0s = max(y0s, 0)
            x1s = min(x1s, Ws - 1)
            y1s = min(y1s, Hs - 1)
            pos = [slice(y0s, y1s), slice(x0s, x1s)]
            trim = [slice(0, y1s - y0s), slice(0, x1s - x0s)]

            # Regression map, too lazy to handle overlap box tho
            reg = np.stack(np.broadcast_arrays(l, t, r, b), axis=0)
            regression_maps[i][:, *pos] = np.fmax(
                reg[:, *trim],
                regression_maps[i][:, *pos],
            )

            # Centerness map
            ctn = np.fmin(l, r) * np.fmin(t, b) / np.fmax(l, r) / np.fmax(t, b)
            if np.any(np.isnan(ctn)):
                ic(ctn)
            ctn = np.sqrt(ctn)
            ctn = ctn / ctn.max()
            centerness_maps[i][c, *pos] = np.fmax(
                ctn[*trim], centerness_maps[i][c, *pos]
            )

            # Break because we encode 1 level per box only
            break

    # Convert image to tensor
    image = image.resize((W, H), Image.Resampling.LANCZOS).convert("RGB")
    image_np = np.array(image, dtype="float32") / 255
    image_np = image_np.transpose(2, 0, 1)

    # visualize_feature_maps(regression_maps, normalize=True)
    # visualize_feature_maps(centerness_maps, normalize=False)
    return (image_np, regression_maps, centerness_maps)


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
    rmaps,
    cmaps,
    threshold=0.05,
    strides: List[int] = [8, 16, 32, 64, 128],
):
    """
    Decode FCOS regression and centerness maps to obtain bounding boxes, classes, and scores.

    Args:
        rmaps (List[np.ndarray]):
            List of regression maps for each scale level.
        cmaps (List[np.ndarray]):
            List of centerness maps for each scale level.
        threshold (float, optional):
            Threshold for centerness scores. Default is 0.05.
        strides (List[int], optional):
            List of strides for each scale level. Default is [8, 16, 32, 64, 128].

    Returns:
        boxes (List[Tuple[float, float, float, float]]):
            List of bounding boxes in xyxy format, normalized to 0..1 range.
        classes (List[int]):
            Object class indices.
        scores (List[float]):
            Object scores.
    """
    boxes = []
    classes = []
    scores = []
    for rmap, cmap, s in zip(rmaps, cmaps, strides):
        c, y, x = np.where(cmap >= threshold)
        centerness = cmap[c, y, x]

        # Unpack regression map
        l, t, r, b = rmap
        l = l[y, x]
        t = t[y, x]
        r = r[y, x]
        b = b[y, x]

        # Remap to image
        x = x * s + s // 2
        y = y * s + s // 2

        # Box
        x0 = x - l
        y0 = y - t
        x1 = x + r
        y1 = y + b
        for x0_, y0_, x1_, y1_, c_, score_ in zip(x0, y0, x1, y1, c, centerness):
            # x0_ = x0_ * s + s // 2
            # y0_ = y0_ * s + s // 2
            # x1_ = x1_ * s + s // 2
            # y1_ = y1_ * s + s // 2
            scores.append(score_)
            boxes.append([x0_, y0_, x1_, y1_])
            classes.append(c_)

    # if False:
    if len(boxes) > 0:
        out_boxes = []
        out_scores = []
        out_classes = []

        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        classes = torch.tensor(classes)
        for i in torch.unique(classes):
            mask = classes == i
            scores_ = scores[mask]
            boxes_ = boxes[mask]
            keep = ops.nms(boxes_.type(torch.float32), scores_, 0.25)

            out_boxes.extend(boxes_[keep].tolist())
            out_scores.extend(classes[mask][keep].tolist())
            out_classes.extend([i] * len(keep))

        boxes = out_boxes
        scores = out_scores
        classes = out_classes

    # Descale boxes
    _, W, H = rmaps[0].shape
    W = W * strides[0]
    H = H * strides[0]
    boxes = [(x0 / W, y0 / H, x1 / W, y1 / H) for (x0, y0, x1, y1) in boxes]
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
