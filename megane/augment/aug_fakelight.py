import random
from typing import Tuple, Callable
from dataclasses import dataclass
from functools import partial

import albumentations as A
import numpy as np
import cv2


@dataclass
class ShaderBasicLight:
    min_deg_x: int = 0
    min_deg_y: int = 0
    max_deg_x: int = 3
    max_deg_y: int = 3
    red: Tuple = (0, 255)
    green: Tuple = (0, 255)
    blue: Tuple = (0, 255)

    def __post_init__(self):
        self.deg_x = random.randint(self.min_deg_x, self.max_deg_x)
        self.deg_y = random.randint(self.min_deg_y, self.max_deg_y)
        self.flip_x = random.choice((True, False))
        self.flip_y = random.choice((True, False))
        self.r = random.choice(self.red)
        self.g = random.choice(self.green)
        self.b = random.choice(self.blue)

    def __call__(self, x, y, w, h):
        deg_x = self.deg_x
        deg_y = self.deg_y
        px = x**deg_x / w**deg_x
        py = y**deg_y / h**deg_y
        if self.flip_x:
            px = 1 - px
        if self.flip_y:
            py = 1 - py
        r = self.r * (1 - px * py)
        g = self.g * (1 - px * py)
        b = self.b * (1 - px * py)
        return (int(r), int(g), int(b))


def fake_light(
    image: np.ndarray, shader_factory: Callable, tile_size: int, alpha: int, **opts
):
    # Prepare
    H, W = image.shape[:2]
    shader_fn = shader_factory()
    if isinstance(tile_size, tuple):
        tile_size = random.randint(*tile_size)
    if isinstance(alpha, tuple):
        alpha = random.uniform(*alpha)

    # Convert image to RGB if it's gray
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # Tiles
    canvas = np.zeros((H, W, 3))
    for x in range(0, W, tile_size):
        for y in range(0, H, tile_size):
            br = shader_fn(x, y, W, H)
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            cv2.rectangle(canvas, (x, y), (x2, y2), br, -1)

    # alpha composite
    image = (image * (1 - alpha) + canvas * alpha).round().astype(image.dtype)
    return image


def FakeLight(tile_size=(20, 50), alpha=(0.2, 0.6), **kw):
    fn = partial(
        fake_light, shader_factory=ShaderBasicLight, tile_size=tile_size, alpha=alpha
    )
    return A.Lambda(image=fn, **kw)
