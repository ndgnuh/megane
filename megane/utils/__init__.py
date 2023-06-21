from io import BytesIO
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from megane.utils.image import prepare_input
from megane.utils.masks import *
from megane.utils.meanap import *
from megane.utils.misc import init_from_ns, save_args, with_batch_mode
from megane.utils.polygons import *

assert prepare_input

try:
    from megane.utils.torch import stack_image_batch

    # Silent linter
    assert stack_image_batch
except ImportError:
    print("Torch is not installed, torch related utilities is not available")


def bytes2pillow(bs: bytes) -> Image:
    """
    Convert byte buffer to Pillow Image

    Args:
        bs: The byte buffer

    Returns:
        image: A PIL Image
    """
    io = BytesIO(bs)
    image = Image.open(io)
    image.load()
    io.close()
    return image


def polygon2xyxy(polygon):
    xmin = min(x for x, y in polygon)
    xmax = max(x for x, y in polygon)
    ymin = min(y for x, y in polygon)
    ymax = max(y for x, y in polygon)
    return xmin, ymin, xmax, ymax


def xyxy2polygon(xyxy):
    xmin, ymin, xmax, ymax = xyxy
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
