from dataclasses import dataclass
from typing import Optional, get_type_hints, List, Dict

import numpy as np
from PIL import Image

from ..data import Sample


class BatchEncode(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Encoded(**{k: v[idx] for k, v in self.items()})
        else:
            return dict.__getitem__(self, idx)


@dataclass
class Encoded:
    image: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    class_scores: np.ndarray
    box_scores: np.ndarray
    image_width: np.ndarray
    image_height: np.ndarray

    def to_tensor(self):
        import torch

        return Encoded(**{k: torch.tensor(v) for k, v in vars(self).items()})

    def to_batch(self):
        return Encoded(**{k: v.unsqueeze(0) for k, v in vars(self).items()})

    def to_numpy(self):
        import torch

        return Encoded(**{k: v.cpu().detach().numpy() for k, v in vars(self).items()})


@dataclass
class DetrProcessor:
    image_width: int
    image_height: int

    def encode(self, sample: Sample) -> Encoded:
        return encode(
            sample=sample, image_width=self.image_width, image_height=self.image_height
        )

    def decode(self, encoded: Encoded) -> Sample:
        return decode(encoded)

    def collate(self, encoded: List[Encoded]) -> BatchEncode:
        import torch

        encoded = [enc.to_tensor() for enc in encoded]
        image = torch.stack([enc.image for enc in encoded])
        keys = [key for key in get_type_hints(Encoded).keys() if key != "image"]
        batch = dict(
            **{k: [getattr(enc, k) for enc in encoded] for k in keys},
            image=image,
        )
        batch = {k: v for k, v in batch.items() if len(v)}
        return BatchEncode(batch)


def encode(sample: Sample, image_width, image_height) -> Encoded:
    orig_image_width = sample.image.width
    orig_image_height = sample.image.height

    # encode image
    image = sample.image.resize([image_width, image_height])
    image = np.array(image)
    if image.dtype == "uint8":
        image = (image / 255).astype("float32")
        image = np.clip(image, 0, 1)

    h, w, c = 0, 1, 2
    image = image.transpose((c, h, w))

    # encode bounding boxes
    return Encoded(
        image=image,
        boxes=np.array(sample.boxes),
        classes=np.array(sample.classes),
        scores=np.array(sample.scores),
        image_width=np.array(orig_image_width),
        image_height=np.array(orig_image_height),
    )


def decode(enc: Encoded) -> Sample:
    # decode image
    c, h, w = 0, 1, 2
    image = enc.image.transpose([h, w, c])
    image = (image * 255).astype("uint8")
    image = Image.fromarray(image).resize([enc.image_width, enc.image_height])

    return Sample(
        image=image,
        boxes=enc.boxes.tolist(),
        classes=enc.classes.tolist(),
        class_scores=enc.class_scores.tolist(),
        box_scores=enc.box_scores.tolist(),
    )
