from megane.augment import albumen as A
from megane.data import Sample


class Augmentation:
    def __init__(
        self,
        prob=0.33333,
        background_images: str = [],
        domain_images: str = [],
    ):
        self.albumen_transform = A.default_transform(
            prob=prob,
            background_images=background_images,
            domain_images=domain_images,
        )
        self.custom_transform = lambda x: x

    def __call__(self, sample: Sample) -> Sample:
        enc = A.encode(sample)
        enc = self.albumen_transform(**enc)
        dec = A.decode(sample, enc)
        dec = self.custom_transform(dec)
        return dec
