# Reference: https://arxiv.org/abs/1911.08947
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from lenses import bind
from megane import utils
from megane.data import Sample
from megane.models.api import ModelAPI


def shrink_dist(w, h, r):
    A = w * h
    L = (w + h) * 2
    d = A / L * (1 - r**2)
    return d


def expand_dist(w, h, r):
    A = w * h
    L = (w + h) * 2
    d = A * r / L
    return d


def encode_dbnet(sample: Sample, num_classes: int, r: float = 0.4):
    boxes = sample.boxes
    W, H = sample.image.size
    classes = sample.classes

    proba_maps = np.zeros((num_classes, H, W), dtype="float32")
    threshold_maps = np.zeros((num_classes, H, W), dtype="float32")

    # Negative sample
    if len(classes) == 0:
        return proba_maps, threshold_maps

    boxes = np.array(sample.boxes, dtype="float32")
    boxes[..., 0] *= W
    boxes[..., 1] *= H

    for box, cls in zip(boxes, classes):
        rect = cv2.minAreaRect(box)
        (x, y), (w, h), a = rect

        d = shrink_dist(w, h, r=r)
        ebox = cv2.boxPoints([(x, y), (w + d, h + d), a]).astype(int)
        sbox = cv2.boxPoints([(x, y), (w - d, h - d), a]).astype(int)

        # Probamap draw
        cv2.fillConvexPoly(proba_maps[cls], sbox, 1)

        # Threshold map draw
        canvas = np.zeros_like(threshold_maps[cls])
        canvas = cv2.fillConvexPoly(canvas, ebox, 1)
        canvas = cv2.fillConvexPoly(canvas, sbox, 0)
        # canvas = cv2.distanceTransform(canvas.astype("uint8"), cv2.DIST_L2, 5)
        threshold_maps[cls] = threshold_maps[cls] + canvas

    threshold_maps = np.clip(threshold_maps, 0, 1)
    return proba_maps, threshold_maps


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

        # self.proj = nn.Conv2d(hidden_size, aux_size, 1)
        # self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class DBNet(ModelAPI):
    def __init__(
        self,
        image_size: int,
        hidden_size: int,
        num_classes: int,
        shrink_rate: float = 0.4,
        expand_rate: float = 1.5,
        resize_mode: str = "resize",
    ):
        super().__init__()
        utils.save_args()

        # 0 = background
        self.probas = PredictionConv(hidden_size, num_classes)
        # 0 = threshold
        self.thresholds = PredictionConv(hidden_size, num_classes)

    def forward(self, features, targets=None):
        probas = self.probas(features)
        if self.infer:
            thresholds = None
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

        def dice_loss(pr, gt, reduction="mean"):
            pr = torch.sigmoid(pr)
            losses = 1 - (pr * gt * 2 + 1) / (pr + gt + 1)
            if reduction == "mean":
                return losses.mean()
            elif reduction == "none":
                return losses
            else:
                raise NotImplementedError(f"Unknown reduction {reduction}")

        loss = 0
        count = 0
        loss_fn = dice_loss
        for i in range(self.num_classes):
            # Prepare
            pr_proba = pr_probas[:, i]
            pr_threshold = pr_thresholds[:, i]
            gt_proba = gt_probas[:, i]
            gt_threshold = gt_thresholds[:, i]

            # Training mask
            proba_mask = (torch.sigmoid(pr_proba * 50) > 0.5) & (gt_proba > 0.2)
            threshold_mask = (torch.sigmoid(pr_threshold * 50) > 0.5) & (
                gt_threshold > 0.2
            )

            # Full classification loss
            # pr = with_background(
            #     torch.stack([pr_proba, pr_threshold], dim=1),
            #     logit=True,
            # )
            # gt = with_background(
            #     torch.stack([gt_proba, gt_threshold], dim=1),
            #     logit=False,
            # )
            # loss += F.cross_entropy(pr, gt)

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
        targets = encode_dbnet(sample, num_classes, shrink_rate)
        return image, targets

    def post_process(self, probas):
        return torch.clamp(torch.tanh(probas * 50), 0, 1)

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
        image = TF.to_pil_image(inputs.detach().cpu())

        # post process
        outputs = outputs[0].detach().cpu()
        if not ground_truth:
            outputs = self.post_process(outputs)

        #  adapt
        masks = outputs.numpy()

        # decode
        final_classes = []
        final_scores = []
        final_polygons = []
        r = self.expand_rate
        for class_idx in range(self.num_classes):
            mask = masks[class_idx]
            polygons, scores = utils.mask_to_rect(mask)
            if r > 0:
                areas = utils.polygon_area(polygons, batch=True)
                lengths = utils.polygon_perimeter(polygons, batch=True)
                dists = [A * r / L for (A, L) in zip(areas, lengths)]
                polygons = [
                    utils.offset_polygon(p, d) for (p, d) in zip(polygons, dists)
                ]

            final_polygons.extend(polygons)
            final_classes.extend([class_idx] * len(polygons))
            final_scores.extend(scores)

        # output sample
        sample = Sample(
            image=image,
            boxes=final_polygons,
            scores=final_scores,
            classes=final_classes,
        )
        return sample
