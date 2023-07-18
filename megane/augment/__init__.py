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
            **other,
        )
        if len(background_images) > 0:
            self.custom_transform = CA.OneOf(
                [
                    CA.ReplaceBackground(
                        background_images=background_images,
                    ),
                    CA.ReplaceNegative(
                        background_images=background_images,
                    ),
                ],
                p=prob,
            )
        else:
            self.custom_transform = lambda x: x

    def _albumen_transform(self, sample):
        enc = A.encode(sample)
        enc = self.albumen_transform(**enc)
        return A.decode(enc)

    def __call__(self, sample: Sample) -> Sample:
        random.shuffle(self.albumen_transform.transforms)
        if random.choice((True, False)):
            sample = self.custom_transform(sample)
            sample = self._albumen_transform(sample)
        else:
            sample = self._albumen_transform(sample)
            sample = self.custom_transform(sample)
        return sample
