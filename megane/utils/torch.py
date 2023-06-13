import torch
from torch import Tensor


def stack_image_batch(images: Tensor):
    """Stack images tensor from N C H W to 1 (N H) (C W).
    Useful for visualizing them on tensorboard.

    Args:
        images:
            Torch Tensor of shape N C H W.

    Returns:
        stacked:
            Torch tensor of shape 1 (N H) (C W)
    """
    images = torch.cat([image for image in images], dim=-2)
    images = torch.cat([image for image in images], dim=-1)
    images = images.unsqueeze(0)
    return images
