# Reference: https://arxiv.org/abs/1911.08947
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple

import cv2
import numpy as np
import simpoly
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from megane import utils
from megane.data import Sample
from megane.debug import with_timer
from megane.registry import heads, target_decoders, target_encoders


@target_encoders.register("dbnet")
@dataclass
class DBNetEncoder:
    num_classes: int
    shrink_rate: float = 0.4
    shrink: bool = True
    min_scale: float = 0.01
    fixed_distance: Optional[float] = None

    def __call__(self, sample: Sample):
        targets = encode_dbnet(
            sample,
            num_classes=self.num_classes,
            r=self.shrink_rate,
            shrink=self.shrink,
            fixed_dist=self.fixed_distance,
        )
        images = TF.to_tensor(sample.image)
        return images, targets


@target_decoders.register("dbnet")
@dataclass
class DBNetDecoder:
    expand_rate: float = 1.5
    expand: bool = True
    min_scale: float = 0.01
    fixed_distance: float = None
    morph_open: Optional[int] = None

    @cached_property
    def morph_kernel(self):
        if self.morph_open is None:
            return None
        ksize = [self.morph_open] * 2
        return cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    def __post_init__(self):
        print("[head_dbnet.py: 52] TODO: rework the dbnet decode and encoder")

    def __call__(self, images, outputs, ground_truth=False):
        outputs = outputs[0].detach().cpu()
        if not ground_truth:
            outputs = torch.sigmoid(outputs * 50)

        image = TF.to_pil_image(images.detach().cpu())
        boxes, classes, scores = decode_dbnet(
            outputs.numpy(),
            self.expand_rate,
            self.expand,
            self.fixed_distance,
            self.morph_kernel,
        )
        sample = Sample(
            image=image,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        return sample


@with_timer
def encode_dbnet(
    sample: Sample,
    num_classes: int,
    r: float = 0.4,
    shrink: bool = True,
    fixed_dist: int = None,
):
    boxes = sample.boxes
    W, H = sample.image.size
    classes = sample.classes

    proba_maps = np.zeros((num_classes, H, W), dtype="float32")
    threshold_maps = np.zeros((num_classes, H, W), dtype="float32")

    # Negative sample
    if len(classes) == 0:
        return proba_maps, threshold_maps

    for xy, cls in zip(boxes, classes):
        box = simpoly.scale_to(xy, W, H)

        # Fixed distance or dynamic
        if fixed_dist is None:
            d = simpoly.get_shrink_dist(box, r)
        else:
            d = fixed_dist

        if shrink:
            sbox = np.array(simpoly.offset(box, d), dtype=int)
        else:
            sbox = np.array(box, dtype=int)
        ebox = simpoly.offset(box, -d)

        # Draw probability map
        cv2.fillConvexPoly(proba_maps[cls], sbox, 1)

        # Threshold map draw
        # Draw through a canvas
        draw_threshold(threshold_maps[cls], box, ebox, d)

    proba_maps = np.clip(proba_maps, 0, 1)
    threshold_maps = np.clip(threshold_maps, 0, 1)
    return proba_maps, threshold_maps


def _compute_score(proba_map, rect):
    canvas = np.zeros_like(proba_map, dtype="float32")
    poly = cv2.boxPoints(rect)
    canvas = cv2.fillConvexPoly(canvas, poly.astype(int), 1)
    score_map = canvas * proba_map
    score = score_map.sum() / np.count_nonzero(score_map)
    return score


def decode_dbnet(
    proba_maps,
    expand_rate: float = 1.5,
    shrink: bool = True,
    fixed_dist: float = None,
    morph_kernel=None,
):
    C, H, W = proba_maps.shape

    classes = []
    scores = []
    boxes = []
    for cls, proba_map in enumerate(proba_maps):
        mask = (proba_map > 0.2).astype("uint8")
        if morph_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=morph_kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            A = cv2.contourArea(cnt)
            L = cv2.arcLength(cnt, True)
            D = abs(A * expand_rate / (L + 1e-3))
            if D == 0 and expand_rate > 0:
                continue

            if fixed_dist is not None:
                D = fixed_dist

            # cnt = cv2.approxPolyDP(cnt, L * 0.01, True)
            box = cnt[:, 0, :].tolist()
            if shrink:
                box = simpoly.offset(box, D)
            box = simpoly.scale_from(box, W, H)
            score = 0.5

            boxes.append(box)
            scores.append(score)
            classes.append(cls)

    # boxes[..., 0] /= W
    # boxes[..., 1] /= H
    return boxes, classes, scores


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(1, -1)
        return x


def loss_mining(losses, mask, k=3):
    positive = mask
    negative = ~mask
    num_positives = torch.count_nonzero(positive)
    num_negatives = torch.count_nonzero(negative)
    if num_positives == 0 or num_negatives == 0:
        return losses.mean()

    num_negatives = min(int(k * num_positives), num_negatives)
    p_loss = torch.topk(losses[positive], num_positives).values.mean()
    n_loss = torch.topk(losses[negative], num_negatives).values.mean()
    return p_loss + n_loss


def generate_background(x, dim=-3, logit=False):
    if logit:
        x = torch.sigmoid(x)
    bg = 1 - x.sum(dim=dim, keepdim=True)
    bg = torch.clamp(bg, 1e-6, 1 - 1e-6)
    if logit:
        bg = torch.log(bg / (1 - bg))
    return bg


def with_background(x, dim=-3, logit=False):
    bg = generate_background(x, dim=dim, logit=logit)
    return torch.cat([bg, x], dim=dim)


class PredictionConv(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        aux_size = hidden_size // 4
        self.conv_1 = nn.Sequential(
            nn.Conv2d(hidden_size, aux_size, 3, padding=1, bias=False),
            nn.InstanceNorm2d(aux_size),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(
                aux_size,
                aux_size,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.InstanceNorm2d(aux_size),
            nn.ReLU(),
        )
        self.conv_3 = nn.ConvTranspose2d(
            aux_size,
            num_classes,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


@heads.register("dbnet")
class DBNet(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.probas = PredictionConv(hidden_size, num_classes)
        self.thresholds = PredictionConv(hidden_size, num_classes)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            return probas
        else:
            thresholds = self.thresholds(features)
            return (probas, thresholds)

    @staticmethod
    def binarize(proba: Tensor, threshold: Tensor, k: int = 50) -> Tensor:
        binmap = k * (proba - threshold)
        return torch.sigmoid(binmap)

    @staticmethod
    def compute_loss_single(
        pr_proba: Tensor,
        pr_threshold: Tensor,
        pr_bin: Tensor,
        gt_proba: Tensor,
        gt_threshold: Tensor,
        proba_scale: float = 5,
        bin_scale: float = 1,
        threshold_scale: float = 10,
    ) -> Tensor:
        # Probamap loss
        p_losses = F.binary_cross_entropy_with_logits(
            pr_proba,
            gt_proba,
            reduction="none",
        )
        p_loss = compute_loss_mining(p_losses, gt_proba > 0)

        # Binary map loss
        b = p_losses.max()
        a = p_losses.min()
        weights = (p_losses - a) / (b - a) + 1
        b_loss = dice_loss(pr_bin, gt_proba, weights=weights)

        # Threshold map loss
        t_loss = F.l1_loss(torch.sigmoid(pr_threshold), gt_threshold)

        loss = proba_scale * p_loss + b_loss * bin_scale + t_loss * threshold_scale
        return loss

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets
        pr_bins = self.binarize(pr_probas, pr_thresholds)

        N, C = gt_probas.shape[:2]

        # Compute loss per sample and classes
        loss = 0
        # for b_idx in range(N):
        #     for c_idx in range(C):
        loss = self.compute_loss_single(
            pr_probas,
            pr_thresholds,
            pr_bins,
            gt_probas,
            gt_thresholds,
        )
        # loss = loss / N / C
        return loss

    def post_process(self, probas):
        return torch.sigmoid(probas)

    @torch.no_grad()
    def visualize_outputs(
        self,
        outputs,
        logger,
        tag,
        step,
        ground_truth: bool = False,
    ):
        probas, thresholds = outputs
        images = torch.cat([probas, thresholds], dim=-3).cpu()
        if not ground_truth:
            images = self.post_process(images * 50)

        images = utils.stack_image_batch(images)
        logger.add_image(tag, images, step)


def draw_threshold(
    canvas: np.ndarray,
    polygon: List[Tuple[float, float]],
    outer_polygon: List[Tuple[float, float]],
    distance: float,
):
    H, W = canvas.shape

    # Polygon bounding rects
    xmin, ymin = np.min(outer_polygon, axis=0)
    xmax, ymax = np.max(outer_polygon, axis=0)

    # Squeeze to int and limit to the canvas
    xmin = max(0, int(round(xmin)))
    xmax = min(W, int(round(xmax)))
    ymin = max(0, int(round(ymin)))
    ymax = min(H, int(round(ymax)))

    # Draw threshold
    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    distances = []
    n = len(polygon)
    for i in range(n):
        xa, ya = polygon[i]
        xb, yb = polygon[(i + 1) % n]
        dist = point_segment_distance(xs[:, None], ys[None, :], xa, ya, xb, yb)
        dist = np.clip(dist / distance, 0, 1)
        distances.append(dist)

    # Paste on the canvas
    try:
        thresholds = 1 - np.min(distances, axis=0).T
        thresholds = np.maximum(thresholds, canvas[ymin:ymax, xmin:xmax])
        canvas[ymin:ymax, xmin:xmax] = thresholds
    except ValueError as e:
        print("xyxy", xmin, xmax, ymin, ymax)
        raise e
    return canvas


def point_segment_distance(
    x: np.ndarray,
    y: np.ndarray,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
):
    # M is the point where we want to compute distance from
    # AB is the segment
    MA2 = np.square(x - xa) + np.square(y - ya)
    MB2 = np.square(x - xb) + np.square(y - yb)
    AB2 = np.square(xa - xb) + np.square(ya - yb)

    # Cos of AMB = cos
    # c = sqrt(a^2 + b^2 - ab cos(α))
    cos = (MA2 + MB2 - AB2) / (2 * np.sqrt(MA2 * MB2) + 1e-6)

    # T is the extension of MB so that ATB is square
    # S_MAB = AT * MB / 2= AB * MH / 2
    # and AT = sin(alpha) * AM
    # therefore MA * MB * sin(π - AMB) = AB * MH
    # and MH = MA * MB * sin(π - AMB) / AB2
    sin2 = 1 - np.square(cos)
    sin2 = np.nan_to_num(sin2)
    dists = np.sqrt(MA2 * MB2 * sin2 / AB2)

    # Cos > 0 means that this AMB is acute
    # therefore the distance should be the height of the triangle from M
    cos_mask = cos >= 0
    dists[cos_mask] = np.sqrt(np.minimum(MA2, MB2)[cos_mask])
    return dists


def dice_loss(
    pr: Tensor,
    gt: Tensor,
    reduction: str = "mean",
    weights: Optional[Tensor] = None,
):
    # Apply weights if any
    if weights is None:
        inter = pr * gt
    else:
        inter = pr * gt * weights
    union = (pr + gt).clamp(0, 1)
    losses = 1 - (2 * inter) / (union + 1e-12)

    # Reduce and return
    if reduction == "mean":
        return losses.mean()
    elif reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    else:
        raise NotImplementedError(f"Unknown reduction {reduction}")


def compute_loss_mining(
    losses: Tensor,
    pos: Tensor,
    neg: Optional[Tensor] = None,
    k: int = 3,
):
    # Negative mask,
    # you can optionally passed it to this function to reduce computation
    if neg is None:
        neg = ~pos

    # Counting
    num_pos = torch.count_nonzero(pos)
    num_neg = torch.min(num_pos * k, torch.count_nonzero(neg))

    # Postive and negative losses
    pos_losses = torch.topk(losses[pos], k=num_pos).values
    neg_losses = torch.topk(losses[neg], k=num_neg).values
    pos_loss = pos_losses.sum()
    neg_loss = neg_losses.sum()

    # Reduction
    loss = (pos_loss + neg_loss) / (num_pos + num_neg + 1e-6)
    return loss
