import torch
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM, ssim


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
    losses = 1 - (pr * gt * 2 + 1) / (pr + gt + 1)
    if reduction == "mean":
        return losses.mean()
    elif reduction == "none":
        return losses
    else:
        raise NotImplementedError(f"Unknown reduction {reduction}")


def bce_dice_loss(pr, gt, reduction="mean"):
    bce = F.binary_cross_entropy(pr, gt, reduction=reduction)
    dice = dice_loss(pr, gt, reduction)
    return (bce + dice) * 0.5


def lc_dice_loss(pr, gt, reduction="mean", logit=True):
    return torch.log(torch.cosh(dice_loss(pr, gt, reduction, logit)))


def dice_ssim_loss(pr, gt, reduction="mean", c=1e-6):
    dice = dice_loss(pr, gt, reduction)
    # ssim = ssim_loss(pr, gt, reduction, c)
    ssim = 1 - ms_ssim(pr, gt, data_range=1, size_average=True)
    return (dice + ssim) * 0.5


def ms_ssim_loss(pr, gt, *args, **kwargs):
    return 1 - ssim(pr, gt, data_range=1, size_average=True)


def ssim_loss(pr, gt, reduction="mean", c=1e-6):
    if pr.ndim > 2:
        return sum(ssim_loss(pr_i, gt_i, c=c) for pr_i, gt_i in zip(pr, gt))

    # Statistics
    s_pr, m_pr = torch.std_mean(pr)
    s_gt, m_gt = torch.std_mean(gt)
    s_pr_gt = torch.mean((pr - m_pr) * (gt - m_gt))

    # Auxiliary computations
    m_pr_2 = torch.pow(m_pr, 2)
    s_pr_2 = torch.pow(s_pr, 2)
    m_gt_2 = torch.pow(m_gt, 2)
    s_gt_2 = torch.pow(s_gt, 2)

    # Image similarity
    c1 = c2 = c3 = c
    l = (2 * m_pr * m_gt + c1) / (m_pr_2 + m_gt_2 + c1)
    c = (2 * s_pr * s_gt + c2) / (s_pr_2 + s_gt_2 + c2)
    s = (s_pr_gt + c3) / (s_pr_2 + s_gt_2 + c3)
    ssim = 1 - l * s * c

    # Reduction
    if reduction == "mean":
        ssim = ssim.mean()
    elif reduction == "sum":
        ssim = ssim.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported reduction {reduction}")
    return ssim
