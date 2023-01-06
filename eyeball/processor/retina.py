from ..tools import remember
from torchvision.transforms import functional as TF
import torch


class RetinaProcessor:
    @remember
    def __init__(
        self,
        threshold: float = 0.6
    ):
        pass

    def pre(self, image, annotation):
        image = TF.to_tensor(image)
        boxes = torch.tensor(annotation)
        return image, boxes

    def post(self):
        pass

    def collate(self, batch):
        pass
