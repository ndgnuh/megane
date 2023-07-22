from dataclasses import dataclass
from typing import List, Tuple

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
            # outputs = torch.sigmoid(outputs)

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
            nn.Sigmoid(),
        )
        self.rrm = nn.Sequential(
            ResidualRefinementModule(num_classes),
            nn.Sigmoid(),
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

    @torch.jit.script
    def loss_reduce(
        losses: Tensor,
        tp: Tensor,
        tn: Tensor,
        fp: Tensor,
        fn: Tensor,
    ):
        weights = [1, 1, 3, 3]
        masks = [tp, tn, fp, fn]
        loss = torch.tensor(0.0, dtype=torch.float32, device=losses.device)
        total = 0
        for i in range(4):
            mask = masks[i]
            weight = weights[i]
            if torch.count_nonzero(mask) == 0:
                continue
            loss = loss + losses[mask].mean()
            total = total + weight

        loss = loss / total
        return loss

    def compute_loss(self, outputs, targets):
        refined, unrefined = outputs

        # # positives and negatives
        # pr_pos = refined > 0.5
        # pr_neg = ~pr_pos
        # gt_pos = targets > 0
        # gt_neg = ~gt_pos

        # # Confusions
        # tp = pr_pos & gt_pos
        # fp = pr_pos & gt_neg
        # tn = pr_neg & gt_neg
        # fn = pr_neg & gt_pos
        # conf = (tp, tn, fp, fn)

        loss = 0
        # bce = F.binary_cross_entropy
        # pos = targets > 0
        # neg = ~pos
        # losses = bce(unrefined, targets, reduction="none")
        # loss += losses[pos].mean() + losses[neg].mean()
        # loss += self.loss_reduce(losses, *conf)
        # losses = bce(refined, targets, reduction="none")
        # loss += losses[pos].mean() + losses[neg].mean()
        # loss += self.loss_reduce(losses, *conf)
        loss += L.dice_ssim_loss(refined, targets, reduction="mean")
        loss += L.dice_ssim_loss(unrefined, targets, reduction="mean")
        return loss / 2

    def visualize_outputs(self, outputs, logger, tag, step, ground_truth=False):
        if not ground_truth:
            outputs = outputs[0]
        outputs = torch.cat(list(outputs), dim=-2)
        outputs = torch.cat(list(outputs), dim=-1)
        outputs = outputs.unsqueeze(0)
        logger.add_image(tag, outputs, step)
