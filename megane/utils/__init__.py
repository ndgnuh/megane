from io import BytesIO
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from megane.utils.image import prepare_input
from megane.utils.masks import *
from megane.utils.meanap import *
from megane.utils.misc import *
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


def init_from_ns(ns, config: Dict):
    """Helper function that takes a namespace and a dictionary
    to initialize an instance.

    Args:
        ns:
            A namespace of any type, must support `getattr`.
        config:
            A dict with the keyword arguments.
            Must contain the `type` key.
            The `type` is the reflection key to determine the
            type name in the specified namespace.
        *args:
            Extra positional arguments

    Returns:
        The initialized instance.
    """
    kind = config.pop("type")
    return getattr(ns, kind)(**config)
