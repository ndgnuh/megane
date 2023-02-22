# the argument are sorted so that we can do something like this:
# outputs = model(inputs)
# loss = criterion(*outputs, *labels)
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import torch

from torch import nn

@dataclass
class DBLoss:
    proba_loss_scale: float = 5.0
    theshold_loss_scale: float = 10.0
    r: float = 50.0
    k: float = 3.0  # negative / positive  rate

    def __call__(
        self,
        proba_maps: Tensor,  # Float
        threshold_maps: Tensor,  # Float
        target_proba_maps: Tensor,  # Bool
        proba_masks: Tensor,  # Bool
        target_threshold_maps: Tensor,  # Bool
        threshold_masks: Tensor,  # Bool
    ):
        # Fallback losses, in case there are no boxes
        proba_map_loss = torch.zeros(1, device=proba_maps.device)
        bin_map_loss = torch.zeros(1, device=proba_maps.device)
        theshold_map_loss = torch.zeros(1, device=proba_maps.device)

        if torch.any(proba_masks):
            # Balanced probability map loss
            proba_map_losses = F.binary_cross_entropy_with_logits(
                proba_maps,
                target_proba_maps * 1.0,
                reduction="none"
            )[proba_masks]

            positive_targets = 1 * target_proba_maps[proba_masks]
            negative_targets = 1 - positive_targets
            positive_count = target_proba_maps[proba_masks].sum()
            negative_count = torch.minimum(
                negative_targets.sum(), self.k * positive_count)
            negative_count = int(negative_count.item())

            negative_loss = proba_map_losses * negative_targets
            negative_loss = negative_loss.sort().values[-negative_count:]
            positive_loss = proba_map_losses * positive_targets
            proba_map_loss = (negative_loss.sum() + positive_loss.sum()
                              ) / (positive_count + negative_count + 1e-6)

            # Dice loss for binary map loss
            binary_maps = torch.sigmoid(
                (proba_maps[proba_masks] - threshold_maps[proba_masks]) *
                self.r
            )
            bce_min = proba_map_losses.min()
            weights = (proba_map_losses - bce_min) / \
                (proba_map_losses.max() - bce_min) + 1.0
            inter = torch.sum(
                binary_maps * target_proba_maps[proba_masks] * weights)
            union = torch.sum(binary_maps) + \
                torch.sum(target_proba_maps[proba_masks]) + 1e-8
            bin_map_loss = 1 - 2.0 * inter / union

        # Theshold map loss, if not all of them are masked
        if torch.any(threshold_masks):
            theshold_map_loss = F.l1_loss(
                threshold_maps[threshold_masks],
                target_proba_maps[threshold_masks] * 1.0
            )

        total_loss = (
            proba_map_loss * self.proba_loss_scale +
            theshold_map_loss * self.theshold_loss_scale +
            bin_map_loss
        )
        return total_loss




class GSpadeLoss(nn.Module):

    def __init__(self, no_prob=0.1, yes_prob=1):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=torch.tensor([no_prob, yes_prob])).to("cuda:0")
        self.l2= nn.L1Loss()
    def forward(self, prob, thres, 
        target_proba_maps: Tensor,  # Bool
            proba_masks: Tensor,  # Bool
            target_threshold_maps: Tensor,  # Bool
            threshold_masks: Tensor,  # Bool
            spade_loss_mask: Tensor,  # Bool,
        ):

        # print("spade_loss_mask: ", spade_loss_mask.dtype)
        # print("prob: ",prob.dtype)
        labels = spade_loss_mask.type(torch.long)
        scores=torch.cat((thres, prob), dim=1)
        score_cross=self.loss(scores, labels)
        score_l2=self.l2(prob.squeeze(1), spade_loss_mask.float())
        # score_l2=self.l2(thres.squeeze(1), 1 - spade_loss_mask.float())
        # if scores.dim() == 3:
        #     scores = scores.unsqueeze(0)
        # if labels.dim() == 2:
        #     labels = labels.unsqueeze(0)
        

        return score_cross #+score_l2