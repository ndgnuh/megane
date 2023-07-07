# Reference: https://arxiv.org/abs/1911.08947
import cv2
import numpy as np
import torch
import simpoly
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from lenses import bind
from megane import utils
from megane.data import Sample
from megane.models.api import ModelAPI
from megane.models.losses import dice_loss
from megane.debug import with_timer


# @with_timer
def encode_dbnet(sample: Sample, num_classes: int, r: float = 0.4, shrink: bool = True):
    boxes = sample.boxes
    W, H = sample.image.size
    classes = sample.classes

    proba_maps = np.zeros((num_classes, H, W), dtype="float32")
    threshold_maps = np.zeros((num_classes, H, W), dtype="float32")

    boxes = [[(x * W, y * H) for (x, y) in box] for box in boxes]

    # Negative sample
    if len(classes) == 0:
        return proba_maps, threshold_maps

    canvas = np.zeros((H, W), dtype="float32")
    for xy, cls in zip(boxes, classes):
        box = xy
        d = simpoly.get_shrink_dist(box, r)
        if shrink:
            sbox = np.array(simpoly.offset(box, d), dtype=int)
        else:
            sbox = np.array(box, dtype=int)
        ebox = np.array(simpoly.offset(box, -d), dtype=int)

        # Draw probability map
        cv2.fillConvexPoly(proba_maps[cls], sbox, 1)

        # Threshold map draw
        # Draw through a canvas
        canvas = cv2.fillConvexPoly(canvas, ebox, 1)
        canvas = cv2.fillConvexPoly(canvas, sbox, 0)
        threshold_maps[cls] += canvas

        # clear canvas
        canvas = cv2.fillConvexPoly(canvas, ebox, 0)

    proba_maps = np.clip(proba_maps, 0, 1)
    threshold_maps = np.clip(threshold_maps, 0, 1)
    backgrounds = 1 - proba_maps
    return proba_maps, threshold_maps, backgrounds


def _compute_score(proba_map, rect):
    canvas = np.zeros_like(proba_map, dtype="float32")
    poly = cv2.boxPoints(rect)
    canvas = cv2.fillConvexPoly(canvas, poly.astype(int), 1)
    score_map = canvas * proba_map
    score = score_map.sum() / np.count_nonzero(score_map)
    return score


def decode_dbnet(proba_maps, expand_rate: float = 1.5, shrink: bool = True):
    C, H, W = proba_maps.shape

    classes = []
    scores = []
    boxes = []
    for cls, proba_map in enumerate(proba_maps):
        mask = (proba_map > 0.2).astype("uint8")
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            A = cv2.contourArea(cnt)
            L = cv2.arcLength(cnt, True)
            D = abs(A * expand_rate / (L + 1e-3))
            if D == 0:
                continue

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


def _prediction_conv(hidden_size: int, num_classes: int):
    aux_size = hidden_size // 4
    return nn.Sequential(
        # First conv
        nn.Conv2d(hidden_size, aux_size, 3, padding=1, bias=False),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        # Second conv
        nn.ConvTranspose2d(
            aux_size,
            aux_size,
            kernel_size=2,
            stride=2,
            bias=False,
        ),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        # Prediction conv
        nn.ConvTranspose2d(
            aux_size,
            num_classes,
            kernel_size=2,
            stride=2,
        ),
    )


class DBNet(ModelAPI):
    def __init__(
        self,
        image_size: int,
        hidden_size: int,
        num_classes: int,
        shrink_rate: float = 0.4,
        expand_rate: float = 1.5,
        shrink: bool = True,
        resize_mode: str = "resize",
    ):
        super().__init__()
        utils.save_args()

        # 0 = background
        self.probas = _prediction_conv(hidden_size, num_classes)
        self.thresholds = _prediction_conv(hidden_size, num_classes)
        self.backgrounds = _prediction_conv(hidden_size, num_classes)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            thresholds = None
            backgrounds = None
        else:
            thresholds = self.thresholds(features)
            backgrounds = self.backgrounds(features)
        return (probas, thresholds, backgrounds)

    def db(self, P, T, k=50, logits=True):
        x = k * (P - T)
        if logits:
            return torch.sigmoid(x)
        else:
            return x

    def compute_loss(self, outputs, targets):
        pr_probas, pr_thresholds, pr_backgrounds = outputs
        gt_probas, gt_thresholds, gt_backgrounds = targets
        loss = 0

        count = 0
        loss_fn = dice_loss
        for i in range(self.num_classes):
            # Prepare
            pr_proba = pr_probas[:, i]
            pr_threshold = pr_thresholds[:, i]
            pr_bg = pr_backgrounds[:, i]
            gt_proba = gt_probas[:, i]
            gt_threshold = gt_thresholds[:, i]
            gt_bg = gt_backgrounds[:, i]

            # Training mask
            # classification loss
            pr = torch.stack([pr_bg, pr_proba], dim=1)
            gt = torch.stack([gt_bg, gt_proba], dim=1)
            loss += F.cross_entropy(pr, gt)

            # DB map loss
            # Training mask is needed because the surrounding will be 0.5
            pr_bin = self.db(pr_proba, pr_threshold, logits=False)
            gt_bin = self.db(gt_proba, gt_threshold, logits=True)
            training_mask = (gt_proba + gt_threshold) > 0
            if torch.count_nonzero(training_mask) > 0:
                pr = pr_bin[training_mask]
                gt = gt_bin[training_mask]
                loss += loss_fn(pr, gt)
                # loss += torch.abs(pr_bin[~training_mask] - 0.5).mean()
                # loss += torch.abs(pr_probas[~training_mask]).mean()

            # Proba map loss
            # pr = with_background(pr_proba.unsqueeze(1), logit=True)
            # gt = with_background(gt_proba.unsqueeze(1))
            pr = pr_proba
            gt = gt_proba
            loss += dice_loss(pr, gt)
            # losses = loss_fn(pr, gt, reduction="none")
            # loss += loss_mining(losses, proba_mask)

            # Threshold map loss
            # pr = with_background(pr_threshold.unsqueeze(1), logit=True)
            # gt = with_background(gt_threshold.unsqueeze(1))
            pr = pr_threshold
            gt = gt_threshold
            losses = F.l1_loss(torch.sigmoid(pr), gt, reduction="mean")
            loss += losses * 10
            # loss += loss_mining(losses, threshold_mask) * 10
            # loss += F.l1_loss(torch.sigmoid(pr), gt, reduction="mean") * 10

            count = count + 1

        loss = loss / count
        return loss

    def encode_sample(self, sample: Sample):
        sz = self.image_size
        num_classes = self.num_classes
        shrink_rate = self.shrink_rate
        sample = bind(sample).image.set(sample.image.resize([sz, sz]))
        image = utils.prepare_input(sample.image, sz, sz, self.resize_mode)
        targets = encode_dbnet(sample, num_classes, shrink_rate, shrink=self.shrink)
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
        probas, thresholds, backgrounds = outputs
        images = torch.cat([probas, thresholds, backgrounds], dim=-3).cpu()
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
        )
        sample = Sample(
            image=image,
            boxes=boxes,
            classes=classes,
            scores=scores,
        )
        return sample