import random
from functools import partial
from typing import Union, Tuple

import cv2
import numpy as np
import albumentations as A


def BloomFilter(
    white_threshold=(220, 240),
    blur: int = (2, 20),
    gain: int = (0.3, 3),
    **kwargs,
):
    fn = partial(
        bloom_filter,
        white_threshold=white_threshold,
        blur=blur,
        gain=gain,
    )
    return A.Lambda(image=fn, name="BloomFilter", **kwargs)


def bloom_filter(
    img,
    white_threshold: Union[Tuple[int, int], int],
    blur: Union[Tuple[int, int], int],
    gain: Union[Tuple[int, int], int],
    **options,
):
    # adapt input
    if isinstance(white_threshold, (list, tuple)):
        white_threshold = random.uniform(*white_threshold)
    if isinstance(blur, (list, tuple)):
        blur = random.uniform(*blur)
    if isinstance(gain, (list, tuple)):
        gain = random.uniform(*gain)

    # convert image to hsv colorspace as floats
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    h, s, v = cv2.split(hsv)

    # Desire low saturation and high brightness for white
    # So invert saturation and multiply with brightness
    sv = ((255 - s) * v / 255).clip(0, 255).astype(np.uint8)

    # threshold
    thresh = cv2.threshold(sv, white_threshold, 255, cv2.THRESH_BINARY)[1]

    # blur and make 3 channels
    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=blur, sigmaY=blur)
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # blend blur and image using gain on blur
    result = cv2.addWeighted(img, 1, blur, gain, 0)

    # output image
    return result
