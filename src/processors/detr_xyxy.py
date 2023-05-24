from dataclasses import dataclass
from typing import Optional, get_type_hints, List, Dict

import numpy as np
from PIL import Image

from ..data import Sample
from . import common


class BatchEncode(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Encoded(**{k: v[idx] for k, v in self.items() if v is not None})
        else:
            return dict.__getitem__(self, idx)


@dataclass
class Encoded:
    image: np.ndarray
    image_width: np.ndarray
    image_height: np.ndarray
    boxes: Optional[np.ndarray] = None # NA when infer
    classes: Optional[np.ndarray] = None # NA when infer
    scores: Optional[np.ndarray] = None # NA when input

    def to_tensor(self):
        import torch

        return Encoded(**{k: torch.tensor(v) for k, v in vars(self).items() if v is not None})

    def to_batch(self):
        return Encoded(**{k: v.unsqueeze(0) for k, v in vars(self).items()})

    def to_numpy(self):
        return Encoded(**{k: v.cpu().detach().numpy() for k, v in vars(self).items() if v is not None})


@dataclass
class DetrProcessor:
    image_width: int
    image_height: int

    def encode(self, sample: Sample) -> Encoded:
        width = sample.image_width
        height = sample.image_height
        # encode image
        image = common.pil_to_np(
            sample.image.resize([self.image_width, self.image_height])
        )

        # encode boxes to xyxy
        boxes = sample.boxes
        if boxes is not None:
            boxes = np.array(boxes)
            boxes = common.normalize(boxes, width, height)
            boxes = common.points_to_xyxy(boxes)

        # encode scores and classes
        # plus 1 because of background
        scores = None if sample.scores is None else np.array(sample.scores)
        classes = None if sample.classes is None else np.array(sample.classes) + 1

        return Encoded(
            image=image,
            boxes=boxes,
            scores=scores,
            classes=classes,
            image_width=width,
            image_height=height,
        )

    def decode(self, encoded: Encoded) -> Sample:
        # decode image
        image = common.pil_to_np(encoded.image)
        image = image.resize([encoded.image_width, encoded.image_height])

        # decode boxes
        boxes = encoded.boxes
        if boxes is not None:
            boxes = common.xyxy_to_points(boxes)
            boxes = common.denormalize(boxes, encoded.image_width, encoded.image_height)
            boxes = boxes.tolist()

        # decode scores
        scores = encoded.scores
        if scores is not None:
            scores = scores.tolist()

        # decode classes
        classes = encoded.classes
        if classes is not None:
            classes = classes - 1
            classes = classes.tolist()

        return Sample(image=image, boxes=boxes, scores=scores, classes=classes)

    def collate(self, encoded: List[Encoded]):
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
