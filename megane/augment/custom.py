import random
from dataclasses import dataclass
from typing import List

import numpy as np
from lenses import bind
from PIL import Image

from megane.data import Sample
from megane.utils import denormalize_polygon, draw_mask


def replace_background(image, background, polygons):
    w, h = image.size
    polygons = denormalize_polygon(polygons, w, h, batch=True)
    polygons = [np.array(p) for p in polygons]
    mask = draw_mask(w, h, polygons).astype(bool)[:, :, None]
    image = np.array(image)
    background = np.array(background.resize((w, h)).convert("RGB"))
    image = image * mask + (~mask) * background
    return Image.fromarray(image)


@dataclass
class ReplaceBackground:
    background_images: List[str]
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.uniform(0, 1) > self.p:
            return sample

        # Load random background
        bg = random.choice(self.background_images)
        bg = Image.open(bg)
        bg.load()

        # Replace background
        image = replace_background(sample.image, bg, sample.boxes)
        # sample.image.close()
        bg.close()

        # Replace image
        new_sample = bind(sample).GetAttr("image").set(image)
        return new_sample
