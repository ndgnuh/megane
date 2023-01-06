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

    def pre(self, image, boxes):
        padding_box = (0, 0, 0, 0)
        width, height = image.size
        num_boxes = width * height // 16
        padding_boxes = [padding_box] * (num_boxes - len(boxes))
        boxes = boxes + padding_boxes
        image = TF.to_tensor(image)
        boxes = torch.tensor(boxes)
        return image, boxes

    def post(self):
        pass

    def collate(self, batch):
        pass
