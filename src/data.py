import numpy as np
from pydantic import BaseModel
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Any
from doctr.datasets import SROIE
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

Point = Tuple[float, float]
Box = Tuple[Point, Point, Point, Point]


class Sample(BaseModel):
    image: Any
    boxes: List[Box] = Field(default_factor=[])
    classes: List[int] = Field(default_factor=[])
    box_scores: List[float] = Field(default_factor=[])
    class_scores: List[float] = Field(default_factor=[])


class MeganeDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.sroie = SROIE(train=train, download=True, use_polygons=True)

    def __getitem__(self, idx):
        image, targets = self.sroie[idx]
        boxes = targets["boxes"]
        return Sample(
            image=TF.to_pil_image(image),
            boxes=boxes.tolist(),
            classes=[1] * len(boxes),
        )

    def __len__(self):
        return len(self.sroie)


def pretty(sample: Sample) -> Image:
    if sample.boxes is None:
        return sample.image

    image = sample.image.copy()
    w, h = image.size
    ctx = ImageDraw.Draw(image)
    boxes = np.array(sample.boxes)
    boxes[..., 0] = boxes[..., 0] * w
    boxes[..., 1] = boxes[..., 1] * h
    for box in boxes:
        polygon = [(int(x), int(y)) for x, y in box]
        ctx.polygon(polygon, outline=(255, 0, 0), width=2)
    return image


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    data = MeganeDataset(train=True)
    image = pretty(data[0])
    plt.imshow(image)
    plt.savefig("test.png")
