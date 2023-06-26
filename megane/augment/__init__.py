import random
import numpy as np
from PIL import Image

from megane.augment import albumen as A
from megane.augment import custom as CA
from megane.data import Sample


class Augmentation:
    def __init__(
        self,
        prob=0.33333,
        background_images: str = [],
        domain_images: str = [],
        **other
    ):
        self.albumen_transform = A.default_transform(
            prob=prob,
            background_images=background_images,
            domain_images=domain_images,
        )
        if len(background_images) > 0:
            self.custom_transform = CA.ReplaceBackground(
                p=prob,
                background_images=background_images,
            )
        else:
            self.custom_transform = lambda x: x

    def __call__(self, sample: Sample) -> Sample:
        sample = self.custom_transform(sample)
        image = np.array(sample.image)
        augmented = self.albumen_transform(image=image)
        sample.image = Image.fromarray(augmented["image"])
        return sample
