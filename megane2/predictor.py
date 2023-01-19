import cv2
import torch
from megane2.loaders import megane_dataloader
from megane2 import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image


class Predictor:
    def __init__(
        self,
        backbone: str,
        hidden_size: int,
        image_width: int,
        image_height: int,
        num_classes: int = 1,
        min_score: float = 0.8,
        weights: str = None,
        device: str = None
    ):
        self.model = models.DBNet(backbone, hidden_size).eval()
        self.image_width = image_width
        self.image_height = image_height

        if weights is not None:
            print("Loaded", weights)
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))

        if device is not None:
            self.model.to(device)

        self.post_processor = transforms.DBPostprocess(
            expand_ratio=1.5,
            min_score=min_score
        )

    @torch.no_grad()
    def predict(self, image: Image.Image):
        image = image.resize((self.image_width, self.image_height))
        pt_image = TF.to_tensor(image).unsqueeze(0)
        proba_maps, _ = self.model(pt_image)
        proba_maps = torch.sigmoid(proba_maps).squeeze(0).cpu().numpy()
        polygons, labels, scores = self.post_processor(proba_maps)
        return polygons, labels, scores, proba_maps
