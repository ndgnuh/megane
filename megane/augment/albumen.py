from itertools import product
from typing import List

import albumentations as A
import cv2
import numpy as np
import simpoly
import toolz
from PIL import Image

from megane.data import Sample

from megane.augment.aug_bloom import BloomFilter
from megane.augment.aug_chromatic_aberration import ChromaticAberration
from megane.augment.aug_fakelight import FakeLight


def rgb_range(step=10):
    r = range(0, 255, step)
    g = range(0, 255, step)
    b = range(0, 255, step)
    return product(r, g, b)


def CoarseDropout(*a, **kw):
    orig = A.CoarseDropout(*a, **kw)
    patched = A.Lambda(name="CoarseDropoutImg", image=orig.apply)
    return patched


def idendity(**kw):
    return kw


def encode(sample: Sample):
    """
    Encodes a Sample to albumentation inputs.

    Args:
        sample (Sample):
            The Sample object containing image and keypoints data.

    Returns:
        Dict:
            Albumentation input.

    Example:
        sample = Sample(image, boxes)
        enc = encode(sample)
        transform(**enc)
    """
    image = np.array(sample.image)
    classes = sample.classes
    boxes = sample.boxes
    h, w = image.shape[:2]
    masks = []
    keypoints = []
    for i, polygon in enumerate(boxes):
        polygon = simpoly.scale_to(polygon, w, h)
        keypoints.extend(polygon)
        masks.extend([i] * len(polygon))

    if len(masks) > 0:
        assert max(masks) == len(classes) - 1
    else:
        assert len(classes) == 0
    return dict(image=image, keypoints=keypoints, box_mapping=masks, classes=classes)


def decode(outputs):
    """
    Decodes the albumenation outputs into a Sample object.

    Args:
        outputs (dict):
            The outputs obtained from the encoding process.

    Returns:
        Sample:
            The decoded Sample object.

    Example:
        sample = Sample(image, boxes, classes, scores)
        outputs = {"image": encoded_image,
            "box_mapping": box_mapping, "keypoints": keypoints}
        decoded_sample = decode(sample, outputs)
    """
    image = Image.fromarray(outputs["image"])
    w, h = image.size
    classes = outputs["classes"]
    masks = outputs["box_mapping"]
    xy = outputs["keypoints"]

    # Conver keypoints to bounding boxes
    groups = toolz.groupby(lambda i: i[1], enumerate(masks))
    out_classes = []
    boxes = []
    for i, idx in groups.items():
        box = simpoly.scale_from([xy[j] for j, _ in idx], w, h)
        if len(box) > 2:
            boxes.append(box)
            out_classes.append(classes[i])

    # Correctness checking
    assert len(boxes) == len(out_classes)

    # Reconstruct another sample
    return Sample(
        image=image,
        boxes=boxes,
        classes=out_classes,
    )


def default_transform(
    prob,
    rotate: bool = False,
    flip: bool = False,
    light_fx: bool = False,
    background_images: List[str] = [],
    domain_images: List[str] = [],
):
    transformations = [
        # Cropping related
        A.OneOf(
            [
                A.RandomCropFromBorders(),
                A.CropAndPad(percent=(0.025, 0.25)),
            ],
            p=prob,
        ),
        # Color effects
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.ToGray(),
                A.Equalize(),
                A.ChannelDropout(),
                A.ChannelShuffle(),
                A.FancyPCA(),
                A.ToSepia(),
                A.ColorJitter(),
                A.RandomGamma(),
                A.RGBShift(),
            ],
            p=prob,
        ),
        # Degrade
        A.OneOf(
            [
                A.PixelDropout(),
                A.OneOf(
                    [
                        CoarseDropout(fill_value=color, max_width=32, max_height=32)
                        for color in range(0, 255)
                    ]
                ),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
                A.Blur(),
                A.MedianBlur(),
                A.Posterize(),
                A.Spatter(),
                A.ISONoise(),
                A.MultiplicativeNoise(),
                A.ImageCompression(quality_lower=50),
                A.GaussNoise(),
            ],
            p=prob,
        ),
    ]

    if light_fx:
        # Lighting
        transformations.append(A.OneOf([FakeLight()], p=prob))
        transformations.append(
            A.OneOf(
                [
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1)),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 1),
                        p=prob,
                        src_radius=32,
                        num_flare_circles_upper=20,
                    ),
                    BloomFilter(),
                    ChromaticAberration(),
                ],
                p=prob,
            )
        )

    if flip:
        # group: Flipping around
        flip_transform = A.OneOf(
            [
                A.RandomRotate90(p=prob),
                A.VerticalFlip(p=prob),
                A.HorizontalFlip(p=prob),
            ],
            p=prob,
        )
        transformations.append(flip_transform)
        # group: Geometric transform

    if rotate:
        rotate_transform = A.OneOf(
            [
                *[
                    A.Perspective(fit_output=True, pad_val=(r, g, b))
                    for (r, g, b) in rgb_range(10)
                ],
                *[
                    A.Affine(
                        scale=(0.3, 1),
                        rotate=(-180, 180),
                        translate_percent=(0.2, 0.2),
                        shear=(-30, 30),
                        fit_output=True,
                        cval=(r, g, b),
                    )
                    for (r, g, b) in rgb_range(10)
                ],
            ],
            p=prob,
        )
        transformations.append(rotate_transform)

    if len(domain_images) > 0:
        domain_transforms = A.OneOf(
            [
                A.FDA(domain_images, beta_limit=0.025),
                A.HistogramMatching(domain_images),
                A.PixelDistributionAdaptation(domain_images),
            ],
            p=prob,
        )
        transformations.append(domain_transforms)
    return A.Compose(
        transformations,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
