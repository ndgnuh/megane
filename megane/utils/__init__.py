from typing import Dict

import cv2
import numpy as np

from megane.utils.image import prepare_input
from megane.utils.masks import *
from megane.utils.meanap import *
from megane.utils.misc import init_from_ns, save_args, with_batch_mode
from megane.utils.polygons import *
from megane.utils.convert import *

assert prepare_input

try:
    from megane.utils.torch import stack_image_batch

    # Silent linter
    assert stack_image_batch
except ImportError:
    print("Torch is not installed, torch related utilities is not available")
