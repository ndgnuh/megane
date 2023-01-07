import torch
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class BalancedBCEWithLogitsLoss:
    k: int = 3  # negative / positive ratio

    def __call__(self, predicts, targets):
        target_mask = targets > 0.5
        n_positive = torch.count_nonzero(target_mask)
        n_negative = torch.min(torch.count_nonzero(
            ~target_mask), self.k * n_positive)
        losses = F.binary_cross_entropy_with_logits(
            predicts,
            targets,
            reduction="none"
        )

        pos_losses = torch.sum(losses[target_mask])
        neg_losses = torch.sum(losses[~target_mask].sort(
            descending=True).values[:n_negative]
        )
        balanced_loss = (pos_losses + neg_losses) / (n_positive + n_negative)
        return balanced_loss


@dataclass
class BalancedBCELoss:
    k: int = 3  # negative / positive ratio

    def __call__(self, predicts, targets):
        assert targets.min() >= 0 and targets.max() <= 1
        assert predicts.min() >= 0 and predicts.max() <= 1
        target_mask = targets >= 0.5
        n_positive = torch.count_nonzero(target_mask)
        n_negative = torch.min(torch.count_nonzero(
            ~target_mask), self.k * n_positive)
        losses = F.binary_cross_entropy(predicts, targets, reduction="none")
        pos_losses = torch.sum(losses[target_mask].sort(
            descending=True).values[:n_positive])
        neg_losses = torch.sum(losses[~target_mask].sort(
            descending=True).values[:n_negative])
        balanced_loss = (pos_losses + neg_losses) / (n_positive + n_negative)
        return balanced_loss


class DBLoss(nn.Module):
    # L = Ls + α×Lb + β×Lt
    # L: total loss
    # Ls: loss of probability map
    # Lb: loss of binary map
    # Lt: loss of threshold map
    # α, β: scale of each loss
    def __init__(self,
                 alpha: float = 1,
                 beta: float = 10,
                 r: float = 50,
                 k: int = 3):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.bce = BalancedBCEWithLogitsLoss(k=3)
        # self.bce = BalancedBCELoss(k=3)
        # self.bce = nn.BCEWithLogitsLoss()
        # self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    # def forward(self,
    #             proba_map: Tensor,
    #             target_proba_map: Tensor,
    #             thresh_map: Tensor,
    #             target_thresh_map: Tensor):
    def forward(self, outputs, annotations):
        # Compat with training interface
        proba_map, thresh_map = outputs
        target_proba_map, target_thresh_map = annotations

        # Segmentation map
        # Skip the 1 / (1 + exp(...)) here, because we're using bce with logits
        r = self.r
        bin_map = r * (proba_map - thresh_map)
        target_bin_map = r * (target_proba_map - target_thresh_map)

        # Probability map loss
        # Ls = self.bce(torch.sigmoid(proba_map),
        #               torch.sigmoid(target_proba_map))
        Ls = self.bce((proba_map),
                      (target_proba_map))

        # Binary map loss
        Lb = self.bce((bin_map),
                      (target_bin_map))
        # Lb = self.bce(torch.sigmoid(bin_map),
        #               torch.sigmoid(target_bin_map))

        # Threshold map loss
        Lt = self.l1(thresh_map, target_thresh_map)

        # Total loss
        L = Ls + self.alpha * Lb + self.beta * Lt
        # ic(Ls, self.alpha * Lb, self.beta * Lt)
        # L = torch.sign(L) * torch.log(torch.abs(L))
        return L


class RetinaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def raise_unsupported_mode(mode):
    raise ValueError("Unsupported loss mode " + str(mode))


class LossMixin(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "db":
            self.loss = DBLoss()
        elif mode == "retina":
            self.loss = RetinaLoss()
        else:
            raise raise_unsupported_mode(mode)

    def forward(self, outputs, annotations):
        mode = self.mode
        if mode == 'db':
            prob_map, thres_map = outputs
            t_prob_map, t_thres_map = annotations
            return self.loss(prob_map, t_prob_map, thres_map, t_thres_map)
        elif mode == "retina":
            raise raise_unsupported_mode(self.mode)
        else:
            raise raise_unsupported_mode(self.mode)
