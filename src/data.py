import numpy as np
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Any
from doctr.datasets import SROIE
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

Point = Tuple[int, int]
Box = Tuple[Point, Point, Point, Point]


class Sample(BaseModel):
    image: Any
    boxes: Optional[List[Box]] = None
    classes: Optional[List[int]] = None
    scores: Optional[List[float]] = None

    @property
    def image_width(self):
        return self.image.width

    @property
    def image_height(self):
        return self.image.width


class MeganeDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.sroie = SROIE(train=train, download=True, use_polygons=True)
        self.transform = transform

    def __getitem__(self, idx):
        image, targets = self.sroie[idx]
        image = TF.to_pil_image(image)
        boxes = targets["boxes"]
        boxes[..., 0] = boxes[..., 0] * image.width
        boxes[..., 1] = boxes[..., 1] * image.height
        boxes = boxes.round().astype(int)

        sample = Sample(
            image=image,
            boxes=boxes.tolist(),
            classes=[1] * len(boxes),
        )

        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return len(self.sroie)


def pretty(sample: Sample) -> Image:
    if sample.boxes is None:
        return sample.image

    image = sample.image.copy()
    w, h = image.size
    ctx = ImageDraw.Draw(image)
    for box in sample.boxes:
        polygon = [(int(x), int(y)) for x, y in box]
        ctx.polygon(polygon, outline=(255, 0, 0), width=2)
    return image


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    data = MeganeDataset(train=True)
    image = pretty(data[0])
    plt.imshow(image)
    plt.savefig("test.png")
