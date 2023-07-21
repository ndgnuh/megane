from dataclasses import dataclass
from typing import List, Tuple

import cv2
import simpoly
import torch
from lenses import bind
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import registry, utils
from megane.data import Sample
from megane.models import losses as L
from megane.models.head_rrm import HeadWithRRM, ResidualRefinementModule
from megane.registry import heads, target_decoders, target_encoders


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
        boxes = scale_polygons(boxes, W, H)
        boxes = polygon2xyxy(boxes)
        classes = sample.classes

        # Encode
        images = TF.to_tensor(sample.image)
        targets = encode_clsm_straight(W, H, C, boxes, classes)
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

        # Decode image
        image = TF.to_pil_image(inputs)
        W, H = image.size

        outputs = outputs.detach().cpu().numpy()
        boxes = []
        classes = []
        for i in range(outputs.shape[-3]):
            mask = (outputs[i] * 255).astype("uint8")
            stats = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, s in stats[2]:
                box = (x / W, y / H, (x + w) / W, (y + h) / H)
                box = xyxy2polygon(box)
                boxes.append(box)
                classes.append(i)
        return Sample(image, classes=classes, boxes=boxes)


@registry.heads.register(name="clsm")
class ClassifierSegmenter(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, num_classes, 4, stride=4),
        )
        self.rrm = nn.Sequential(
            ResidualRefinementModule(num_classes),
        )

    def forward(self, x, *args, **kwargs):
        x = self.conv_1(x)
        prerefine = x
        x = self.rrm(x)
        if self.infer:
            return x
        else:
            return x, prerefine

    def ms_ce(self, outputs: torch.Tensor, targets: torch.Tensor):
        loss = F.cross_entropy(outputs, targets)
        for scale in [0.5, 0.75, 1.5, 2]:
            pr = F.interpolate(outputs, scale_factor=scale)
            gt = F.interpolate(targets, scale_factor=scale)
            loss = loss + F.cross_entropy(pr, gt)
        return loss / 5

    def loss_sampling(self, losses, masks, weights):
        loss = 0
        total = 0
        for mask, weight in zip(masks, weights):
            if torch.count_nonzero(mask) == 0:
                continue
            loss += losses[mask].mean() * weight
            total += weight
        loss = loss / total
        return loss

    def loss_sampling2(self, losses, positives, negatives=None, n_pos=None, n_neg=None):
        if negatives is None:
            negatives = ~positives
        if n_pos is None:
            n_pos = torch.count_nonzero(positives)
        if n_neg is None:
            n_neg = min(torch.count_nonzero(negatives), n_pos * 3)
        if n_pos == 0 or n_neg == 0:
            return losses.mean()

        p_losses, _ = torch.topk(losses[positives], n_pos)
        n_losses, _ = torch.topk(losses[negatives], n_neg)
        loss = p_losses.mean() + n_losses.mean()
        return loss

    def compute_loss(self, outputs, targets):
        refined, unrefined = outputs
        unrefined = torch.sigmoid(unrefined)
        refined = torch.sigmoid(refined)
        positives = (torch.sigmoid(refined) > 0.75) == targets
        # positives = targets > 0
        # tp = (predicts == 1) & (targets == 1)
        # fp = (predicts == 1) & (targets == 0)
        # tn = (predicts == 0) & (targets == 0)
        # fn = (predicts == 0) & (targets == 1)
        # unsure = (refined <= 0.9) & (refined >= 0.1)
        negatives = ~positives
        n_pos = torch.count_nonzero(positives)
        n_neg = min(torch.count_nonzero(negatives), n_pos * 3)

        loss = 0
        bce = F.binary_cross_entropy_with_logits
        losses = bce(unrefined, targets, reduction="none")
        loss += self.loss_sampling2(losses, positives, negatives, n_pos, n_neg)
        losses = bce(refined, targets, reduction="none")
        loss += self.loss_sampling2(losses, positives, negatives, n_pos, n_neg)
        losses = L.dice_ssim_loss(refined, targets, reduction="none")
        loss += self.loss_sampling2(losses, positives, negatives, n_pos, n_neg)
        losses = L.dice_ssim_loss(unrefined, targets, reduction="none")
        loss += self.loss_sampling2(losses, positives, negatives, n_pos, n_neg)
        return loss / 4

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth=False):
        if not ground_truth:
            outputs = torch.sigmoid(outputs[0])
        outputs = torch.cat(list(outputs), dim=-2)
        outputs = torch.cat(list(outputs), dim=-1)
        outputs = outputs.unsqueeze(0)
        logger.add_image(tag, outputs, step)
