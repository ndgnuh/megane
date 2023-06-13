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
