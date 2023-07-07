import torch
from torch import nn


class Contour(nn.Module):
    """Calculate contour of a feature map.

    Args:
        kernel_size:
            Dilate/erode kernel size. Default: 3

    Inputs:
        images:
            Tensor of shape N, C, H, W.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.dilate = nn.MaxPool2d(kernel_size, stride=1)
        self.relu = nn.ReLU()

    def forward(self, images):
        d = self.dilate(images)
        e = -self.dilate(-images)
        return self.relu(d - e)


class ContourLoss(nn.Module):
    """Calculate loss between contour of feature maps.
    The loss applied on contours is MSE.

    Args:
        kernel_size:
            Dilate/erode kernel size. Default: 3

    Inputs:
        outputs:
            Tensor of shape N, C, H, W.
        targets:
            Tensor of shape N, C, H, W.

    References:
        - https://proceedings.mlr.press/v143/el-jurdi21a/el-jurdi21a.pdf
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.contour = Contour(kernel_size)
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        c_outputs = self.contour(outputs)
        c_targets = self.contour(targets)
        return self.mse(c_outputs, c_targets)


class DiceLoss(nn.Module):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
    """

    def __init__(self, smooth: float = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)

        smooth = self.smooth
        return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


class LogCoshDiceLoss(nn.Module):
    """Log(Cosh(DiceLoss))"""

    def __init__(self, smooth: float = 1):
        super().__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, pr, gt):
        dice = self.dice(pr, gt)
        return torch.log(torch.cosh(dice))


def dice_loss(pr, gt, reduction="mean"):
    pr = torch.sigmoid(pr)
    losses = 1 - (pr * gt * 2 + 1) / (pr + gt + 1)
    if reduction == "mean":
        return losses.mean()
    elif reduction == "none":
        return losses
    else:
        raise NotImplementedError(f"Unknown reduction {reduction}")


def lc_dice_loss(pr, gt, reduction="mean"):
    torch.log(torch.cosh(dice_loss(pr, gt, reduction)))
