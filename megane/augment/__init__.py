import random

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
        self.custom_transform = CA.ReplaceBackground(
            p=prob,
            background_images=background_images,
        )

    def __call__(self, sample: Sample) -> Sample:
        if random.choice((True, False)):
            enc = A.encode(sample)
            enc = self.albumen_transform(**enc)
            dec = A.decode(sample, enc)
            dec = self.custom_transform(dec)
        else:
            sample = self.custom_transform(sample)
            enc = A.encode(sample)
            enc = self.albumen_transform(**enc)
            dec = A.decode(sample, enc)
        return dec
