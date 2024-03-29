import random
from dataclasses import dataclass
from typing import List, Callable

import numpy as np
from lenses import bind
from PIL import Image

from megane.data import Sample
from megane.utils import denormalize_polygon, draw_mask


def replace_background(image, background, polygons):
    w, h = image.size
    blend = random.uniform(0.5, 1)
    polygons = denormalize_polygon(polygons, w, h, batch=True)
    polygons = [np.array(p) for p in polygons]
    mask = draw_mask(w, h, polygons).astype(bool)[:, :, None]
    mask = mask * blend
    image = np.array(image)
    background = np.array(background.resize((w, h)).convert("RGB"))
    image = image * mask + (1 - mask) * background
    image = image.astype("uint8")
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


@dataclass
class ReplaceNegative:
    background_images: List[str]
    p: float = 0.5

    def __call__(self, sample):
        if random.uniform(0, 1) > self.p:
            return sample

        bg = random.choice(self.background_images)
        bg = Image.open(bg)
        return Sample(image=bg.convert("RGB"))


@dataclass
class OneOf:
    transforms: List[Callable]
    p: float = 0.5

    def __call__(self, sample):
        if random.uniform(0, 1) > self.p:
            return sample
        weights = [t.p for t in self.transforms]
        transform = random.choices(self.transforms, weights, k=1)[0]
        return transform(sample)
