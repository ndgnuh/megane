from io import BytesIO
from typing import Dict

import numpy as np
from PIL import Image

from megane.utils.masks import *
from megane.utils.meanap import *
from megane.utils.misc import *
from megane.utils.polygons import *

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


def prepare_input(image: Image, image_width: int, image_height: int):
    """Prepare the input to be fed to the model

    Args:
        image:
            Pillow image
        image_width:
            The image width W that model expects
        image_height:
            The image height H that model expects

    Returns:
        A numpy array of shape [3, H, W], type `float32`, value normalized to [0, 1] range.
    """
    image = image.convert("RGB")
    image = image.resize((image_width, image_height))
    image = np.array(image)
    image = image.astype("float32") / 255
    h, w, c = 0, 1, 2
    image = image.transpose([c, h, w])
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

    Returns:
        The initialized instance.
    """
    kind = config.pop("type")
    return getattr(ns, kind)(**config)
