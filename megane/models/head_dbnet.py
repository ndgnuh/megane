# Reference: https://arxiv.org/abs/1911.08947
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import simpoly
from typing import Optional, Tuple, List
from functools import cached_property
from math import sqrt

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from lenses import bind
from megane import utils
from megane.data import Sample
from megane.models.api import ModelAPI
from megane.models.head_rrm import HeadWithRRM, ResidualRefinementModule
from megane.models import losses as L
from megane.debug import with_timer


from megane.registry import heads, target_decoders, target_encoders


def offset(poly, offset):
    scale = 1000
    n = len(poly)
    offset = offset * scale
    offset_lines = []
    new_poly = []

    total = n
    for i in range(n):
        # Line endpoints
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if x1 == x2 and y1 == y2:
            total = total - 1
            continue

        # Rescale for accuracy
        x1 = x1 * scale
        x2 = x2 * scale
        y1 = y1 * scale
        y2 = y2 * scale

        # Calculate the direction vector & normal vector
        vx, vy = x2 - x1, y2 - y1
        vx, vy = vy, -vx

        # normalize the normal vector
        length = sqrt(vx**2 + vy**2)
        vx, vy = vx / length, vy / length

        # Offset endpoints -> offset lines
        x1 = x1 - vx * offset
        y1 = y1 - vy * offset
        x2 = x2 - vx * offset
        y2 = y2 - vy * offset
        offset_lines.append((x1, y1, x2, y2))

    # Find intersections
    # New poly vertices are the intersection of the offset lines
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    n = total
    for i in range(n):
        (x1, y1, x2, y2) = offset_lines[i]
        (x3, y3, x4, y4) = offset_lines[(i + 1) % n]
        deno = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-6
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - x4 * y3)) / deno
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - x4 * y3)) / deno
        new_poly.append((x / scale, y / scale))

    return new_poly


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

    boxes = [[(x * W, y * H) for (x, y) in box] for box in boxes]

    # Negative sample
    if len(classes) == 0:
        return proba_maps, threshold_maps

    max_size = max(H, W) * 2
    for xy, cls in zip(boxes, classes):
        box = xy

        # Fixed distance or dynamic
        if fixed_dist is None:
            d = simpoly.get_shrink_dist(box, r)
        else:
            d = fixed_dist

        if shrink:
            sbox = np.array(offset(box, d), dtype=int)
        else:
            sbox = np.array(box, dtype=int)
        ebox = np.clip(offset(box, -d), 0, np.inf).astype(int)

        # Draw probability map
        cv2.fillConvexPoly(proba_maps[cls], sbox, 1)

        # Threshold map draw
        # Draw through a canvas
        canvas = draw_threshold(W, H, box, sbox, ebox, d)
        threshold_maps[cls] = np.maximum(threshold_maps[cls], canvas)

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
    def __init__(self, hidden_size: int, num_classes: int, refine: str = None):
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
        if refine == "rrm":
            self.refine = ResidualRefinementModule(num_classes)
        elif refine == "fast":
            self.refine = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 21, padding=21 // 2),
                nn.ReLU(),
            )
        else:
            self.refine = None

        # self.proj = nn.Conv2d(hidden_size, aux_size, 1)
        # self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        if self.refine is not None:
            x = self.refine(x) + x
        return x


@heads.register("dbnet")
class DBNet(ModelAPI):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        shrink_rate: float = 0.4,
        expand_rate: float = 1.5,
        shrink: bool = True,
        resize_mode: str = "resize",
        fixed_dist: float = None,
        refine: str = None,
    ):
        super().__init__()
        utils.save_args()

        # 0 = background
        self.probas = PredictionConv(hidden_size, num_classes, refine=refine)
        # 0 = threshold
        self.thresholds = PredictionConv(hidden_size, num_classes, refine=refine)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            return probas
        else:
            thresholds = self.thresholds(features)
        return (probas, thresholds)

    def db(self, P, T, k=50, logits=True):
        x = k * (P - T)
        if logits:
            return torch.sigmoid(x)
        else:
            return x

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds = outputs
        gt_probas, gt_thresholds = targets

        loss = 0
        count = 0
        for i in range(self.num_classes):
            # Prepare
            pr_proba = pr_probas[:, i]
            pr_threshold = pr_thresholds[:, i]
            gt_proba = gt_probas[:, i]
            gt_threshold = gt_thresholds[:, i]
            # DB map loss

            # Training mask is needed because the surrounding will be 0.5
            pr_bin = self.db(pr_proba, pr_threshold, logits=True)
            gt_bin = self.db(gt_proba, gt_threshold, logits=True)
            training_mask = (gt_proba + gt_threshold) > 0
            pr = pr_bin[training_mask]
            gt = gt_bin[training_mask]
            loss += F.binary_cross_entropy_with_logits(pr, gt)

            # Proba map loss
            pr = torch.sigmoid(pr_proba).unsqueeze(0)
            gt = gt_proba.unsqueeze(0)
            loss += F.binary_cross_entropy_with_logits(pr, gt)

            # Threshold map loss
            pr = pr_threshold
            gt = gt_threshold
            losses = F.l1_loss(torch.sigmoid(pr), gt)
            loss += losses * 10

            count = count + 1

        loss = loss / count
        return loss

    def encode_sample(self, sample: Sample):
        sz = sample.image.size
        num_classes = self.num_classes
        shrink_rate = self.shrink_rate
        image = TF.to_tensor(sample.image)
        targets = encode_dbnet(
            sample,
            num_classes,
            shrink_rate,
            shrink=self.shrink,
            fixed_dist=self.fixed_dist,
        )
        return image, targets

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

    @torch.no_grad()
    def decode_sample(self, inputs, outputs, ground_truth: bool = False):
        # post process
        outputs = outputs[0].detach().cpu()
        if not ground_truth:
            outputs = self.post_process(outputs)

        image = TF.to_pil_image(inputs.detach().cpu())
        boxes, classes, scores = decode_dbnet(
            outputs.numpy(),
            self.expand_rate,
            self.shrink,
            self.fixed_dist,
        )
        sample = Sample(
            image=image,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        return sample


def point_line_distance(
    xy: Tuple[float, float], P1: Tuple[float, float], P2: Tuple[float, float]
) -> float:
    x0, y0 = xy
    x1, y1 = P1
    x2, y2 = P2
    n = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return n / d


def min_polygon_distance(
    xy: Tuple[float, float], polygons: List[Tuple[float, float]], n: int
) -> float:
    min_d = 9999
    for i in range(n):
        P1 = polygons[i]
        P2 = polygons[(i + 1) % n]
        d = point_line_distance(xy, P1, P2)
        min_d = min(d, min_d)
    return min_d


def draw_threshold(
    W: int,
    H: int,
    polygon: List[Tuple[float, float]],
    inner_polygon: List[Tuple[float, float]],
    outer_polygon: List[Tuple[float, float]],
    expand_dist: float,
):
    canvas = np.zeros((H, W), "float32")
    cv2.fillConvexPoly(canvas, np.array(outer_polygon, int), 1)
    cv2.fillConvexPoly(canvas, np.array(inner_polygon, int), 0)
    I, J = np.where(canvas)
    n = len(polygon)
    for i, j in zip(I, J):
        d = min_polygon_distance((j, i), polygon, n)
        d = max(min(d / expand_dist, 1), 0)
        canvas[i, j] = 1 - d
    return canvas
