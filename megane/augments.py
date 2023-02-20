import albumentations as A
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass
from typing import Callable
from PIL import Image, ImageDraw
from functools import lru_cache, cached_property
from os import path, listdir, environ
import cv2


class CopyPaste:
    def __init__(self, bg_dir=None, p=0.5):
        if bg_dir is None:
            bg_dir = environ["COPY_PASTE_BG_DIR"]
        self.background_dir = bg_dir
        self.p = p

    @cached_property
    def background_files(self):
        return [
            np.array(Image.open(
                path.join(self.background_dir, file)
            ).convert("RGB"))
            for file in listdir(self.background_dir)
        ]

    def __call__(self, image, keypoints: np.ndarray):
        if random.uniform(0, 1) >= self.p:
            return dict(image=image, keypoints=keypoints)

        # Background mask
        h, w = image.shape[:2]
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        for polygon in keypoints:
            mask_draw.polygon(polygon, fill=255)
        mask = np.array(mask) > 0
        mask = mask[:, :, None]

        # Random background
        bg = random.choice(self.background_files)
        bg = cv2.resize(bg, (w, h))
        image = image * mask + (~mask) * bg
        return dict(image=image, keypoints=keypoints)


def no_augment():
    return None


def default_augment(p=0.3):
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)

    return A.Compose([
        CopyPaste(p=0.7),

        # Changing image coloring
        A.OneOf([
            A.CLAHE(p=p),
            A.ColorJitter(p=p),
            A.Emboss(p=p),
            A.HueSaturationValue(p=p),
            A.RandomBrightnessContrast(p=p),
            A.InvertImg(p=p),
            A.RGBShift(p=p),
            A.ToSepia(p=p),
            A.ToGray(p=p),
        ]),

        # Noises
        A.OneOf([
            A.ISONoise(p=p),
            A.MultiplicativeNoise(p=p),
        ]),

        # Dropouts
        A.OneOf([
            A.PixelDropout(p=p),
            A.ChannelDropout(p=p),
        ]),

        # Image degration
        A.OneOf([
            A.ImageCompression(p=p),
            A.GaussianBlur(p=p),
            A.Posterize(p=p),
            A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, p=p),
            A.MedianBlur(blur_limit=1, p=p),
            A.MotionBlur(p=p),
        ]),

        # Spatial transform
        A.OneOf([
            # Doesn't work on keypoints
            # A.ElasticTransform(alpha=1, sigma=1, alpha_affine=1, p=p),
            A.Perspective(fit_output=True, p=p),

            # Removed due
            # A.PiecewiseAffine(nb_rows=3, nb_cols=3, p=p),

            # Removed due to making the output out of range
            A.ShiftScaleRotate(p=p),
            # A.SafeRotate((-5, 5), p=p),
        ])
    ], keypoint_params=keypoint_params)


@dataclass
class Augment:
    transform: Optional[Callable]

    def __call__(self, image, annotation):
        if self.transform is None:
            return image, annotation

        width, height = image.size
        polygons = np.array(annotation['polygons'])
        num_polygons = polygons.shape[0]

        # Use keypoints xy format to augment
        # The polygons are 0-1 normalized
        polygons = polygons.reshape(-1, 2)
        polygons[:, 0] *= width
        polygons[:, 1] *= height
        result = self.transform(
            image=np.array(image),
            keypoints=polygons
        )

        # Outputs
        # Returns to the old annotion format
        image = Image.fromarray(result['image'])
        polygons = np.array(result['keypoints'])
        polygons[:, 0] /= width
        polygons[:, 1] /= height
        polygons = polygons.reshape(num_polygons, 4, 2)
        annotation['polygons'] = polygons.tolist()
        return image, annotation

    @classmethod
    def from_string(cls, augment: str):
        transform = dict(
            default=default_augment,
            none=no_augment,
            yes=default_augment,
            no=no_augment,
        )[augment]()
        return cls(transform)
