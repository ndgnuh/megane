import albumentations as A
import numpy as np
from PIL import Image

from megane import utils
from megane.data import Sample


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
    w, h = sample.image.size
    boxes = utils.denormalize_polygon(sample.boxes, w, h, batch=True)
    masks = []
    keypoints = []
    for i, polygon in enumerate(boxes):
        keypoints.extend(polygon)
        masks.extend([i] * len(polygon))
    return dict(image=image, keypoints=keypoints, box_mapping=masks)


def decode(sample: Sample, outputs):
    """
    Decodes the albumenation outputs into a Sample object.

    Args:
        sample (Sample):
            The original Sample object.
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
    w, h = sample.image.size
    image = Image.fromarray(outputs["image"])

    # Convert keypoints to bounding boxes
    boxes = [[]] * len(sample.boxes)
    for mask, xy in zip(outputs["box_mapping"], outputs["keypoints"]):
        x, y = xy
        boxes[mask].append((x, y))

    # Remove invalid bounding boxes
    keeps = [len(box) > 2 for box in boxes]
    boxes = utils.normalize_polygon(boxes, w, h, batch=True)
    boxes = [c for (c, keep) in zip(boxes, keeps) if keep]
    classes = [c for (c, keep) in zip(sample.classes, keeps) if keep]
    if sample.scores is None:
        scores = None
    else:
        scores = [c for (c, keep) in zip(sample.scores, keeps) if keep]

    # Reconstruct another sample
    return Sample(
        image=image,
        boxes=boxes,
        classes=classes,
        scores=scores,
    )


def default_transform(prob, background_images, domain_images):
    transformations = [
        # Color fx
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.InvertImg(),
                A.ToGray(),
                # A.Solarize(),
                # A.ToSepia(),
                # A.ColorJitter(),
                # A.RandomGamma(),
                # A.RandomShadow(),
                # A.RandomSunFlare(),
                # A.RGBShift(),
            ],
            p=prob,
        ),
        # Degrade
        A.OneOf(
            [
                A.Downscale(),
                A.Blur(),
                A.MedianBlur(),
            ],
            p=prob,
        ),
        # Channel fx
        A.OneOf(
            [
                A.ChannelDropout(),
                A.ChannelShuffle(),
            ],
            p=prob,
        ),
        # Noise
        A.OneOf(
            [
                A.ISONoise(),
                A.MultiplicativeNoise(),
                A.GaussNoise(),
            ],
            p=prob,
        ),
        # Geometric transform/rotate
        A.OneOf(
            [
                A.RandomRotate90(),
                A.SafeRotate((-180, 180)),
            ],
            p=prob,
        )
        # A.OneOf(
        #     [
        #         A.Affine(
        #             scale=(0.4, 1),
        #             rotate=(-30, 30),
        #             translate_percent=0.0,
        #             shear=0,
        #             keep_ratio=True,
        #             fit_output=True,
        #         ),
        #         # A.RandomRotate90(),
        #         # A.Transpose(),
        #     ],
        #     p=prob,
        # ),
    ]

    if len(domain_images) > 0:
        domain_transforms = A.OneOf(
            [
                A.FDA(domain_images, beta_limit=0.025),
                # A.HistogramMatching(domain_images, p=0.2),
                # A.PixelDistributionAdaptation(domain_images, p=0.4),
            ],
            p=prob,
        )
        transformations.append(domain_transforms)
    return A.Compose(
        transformations,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
