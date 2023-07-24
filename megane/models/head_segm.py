import math
from dataclasses import dataclass
from typing import List, Tuple

import simpoly
import numpy as np
import cv2
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import registry
from megane.data import Sample
from megane.models import losses as L
from megane.models.head_rrm import ResidualRefinementModule


@torch.jit.script
def polygon2xyxy_single(polygon: List[Tuple[float, float]]):
    x = [pt[0] for pt in polygon]
    y = [pt[1] for pt in polygon]
    return [int(min(x)), int(min(y)), int(max(x)), int(max(y))]


@torch.jit.script
def polygon2xyxy(polygons: List[List[Tuple[float, float]]]):
    return [polygon2xyxy_single(polygon) for polygon in polygons]


@torch.jit.script
def encode_clsm_straight(
    W: int,
    H: int,
    C: int,
    boxes: List[Tuple[int, int, int, int]],
    classes: List[int],
):
    targets = torch.zeros((C, H, W), dtype=torch.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = classes[i]
        targets[c, y1:y2, x1:x2] = 1

    return targets


@torch.jit.script
def xyxy2polygon(box: Tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


@torch.jit.script
def scale_polygons(polygons: List[List[Tuple[float, float]]], W: int, H: int):
    return [[(x * W, y * H) for (x, y) in box] for box in polygons]


@registry.target_encoders.register(name="clsm")
@dataclass
class CLSMEncoder:
    num_classes: int
    straight: bool

    def __call__(self, sample: Sample):
        # Unpack
        W, H = sample.image.size
        C = self.num_classes
        boxes = sample.boxes
        # boxes = scale_polygons(boxes, W, H)
        # boxes = polygon2xyxy(boxes)
        classes = sample.classes

        # Encode
        images = TF.to_tensor(sample.image)
        targets = encode_ellipse(W, H, boxes, classes, C)
        return images, targets


@registry.target_decoders.register(name="clsm")
@dataclass
class CLSMDecoder:
    straight: bool

    def __call__(self, inputs, outputs, ground_truth=False):
        # Post process
        if not ground_truth:
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs = torch.sigmoid(outputs)

        # Decode image
        image = TF.to_pil_image(inputs)
        W, H = image.size

        outputs = outputs.detach().cpu().numpy()
        boxes, classes = decode_masks(outputs)
        return Sample(image, classes=classes, boxes=boxes)


@registry.heads.register(name="clsm")
class ClassifierSegmenter(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, num_classes, 4, stride=4),
        )
        self.conv_reflect = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, num_classes, 4, stride=4),
        )

    def forward(self, ft, *args, **kwargs):
        x = ft
        x = self.conv_1(x)
        if self.infer:
            return torch.sigmoid(x)
        else:
            reflect = self.conv_reflect(ft)
            return x, reflect

    def compute_loss(self, outputs, targets):
        probas, reflects = outputs

        # Informations
        # loss = 0
        # pos = targets > 0
        # neg = ~pos
        # pred = torch.sigmoid(probas) > 0.5
        # unsure = (pred < 0.8) & (pred > 0.2)
        gt_corrects = ((probas > 0.5) == (targets > 0)) * 1.0
        bce = F.binary_cross_entropy_with_logits

        # confusions
        # fp = (pred == 1) * (targets == 0)
        # fn = (pred == 0) * (targets == 1)

        # Reflection loss
        losses = []
        for i in range(probas.shape[1]):
            r_loss = bce(reflects[:, i], gt_corrects[:, i])

            # Mask loss
            m_loss = bce(probas[:, i], targets[:, i])

            # Total loss
            loss = (m_loss + r_loss) / 2
            losses.append(loss)

        loss = sum(losses) / len(losses)
        return loss

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth=False):
        if not ground_truth:
            outputs = torch.sigmoid(outputs[0])

        if outputs.ndim == 4:
            outputs = torch.cat(list(outputs), dim=-2)
        outputs = torch.cat(list(outputs), dim=-1)
        outputs = outputs.unsqueeze(0)
        logger.add_image(tag, outputs, step)


def decode_masks(masks: np.ndarray):
    num_classes = len(masks)
    boxes = []
    classes = []
    for i in range(num_classes):
        mask = masks[i]
        H, W = mask.shape
        cnts, _ = cv2.findContours(
            (mask * 255).astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts:
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2 = x1 + w
            y2 = y1 + h

            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            polygon = simpoly.scale_from(polygon, W, H)

            boxes.append(polygon)
            classes.append(i)
    return boxes, classes


def encode_ellipse(
    W: int,
    H: int,
    boxes: List[List[Tuple[float, float]]],
    classes: List[int],
    num_class: int,
):
    targets = [np.zeros((H, W), "float32") for _ in range(num_class)]
    n = len(boxes)
    for i in range(n):
        # Scale
        box = simpoly.scale_to(boxes[i], W, H)
        cls = classes[i]

        #
        x, y, w, h = cv2.boundingRect(np.array(box, int))
        cxcy = (x + w // 2, y + h // 2)
        canvas = draw_gradient_ellipse((H, W), cxcy, (w // 2, h // 2), 0)
        targets[cls] = np.maximum(targets[cls], canvas)
    targets = np.clip(targets, 0, 1)
    return targets


def draw_gradient_ellipse(
    HW: Tuple[int, int],
    cxcy: Tuple[int, int],
    ab: Tuple[int, int],
    theta: float,
):
    # Precomputed constants
    H, W = HW
    cx, cy = cxcy
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = ab[0] ** 2, ab[1] ** 2

    # Use grid and broadcasting instead of loops
    x = np.arange(HW[1])[:, None]
    y = np.arange(HW[0])

    # https://stackoverflow.com/questions/49829783/draw-a-gradual-change-ellipse-in-skimage
    dist = (
        np.power((x - cx) * ct + (y - cy) * st, 2) / aa
        + np.power((x - cy) * st - (y - cy) * ct, 2) / bb
    )
    dist = np.clip(1 - dist, 0, 1)
    dist = dist.T
    return dist
