import cv2
import torch
from megane2.loaders import megane_dataloader
from megane2.configs import read_config
from megane2 import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image


class Predictor:
    def __init__(
        self,
        model_config: str,
        device: str = None
    ):
        if isinstance(model_config, str):
            model_config = read_config(model_config)
        self.model = models.DBNet.from_config(model_config)
        self.post_processor = transforms.DBPostprocess.from_config(
            model_config
        )
        self.image_width = model_config["image_width"]
        self.image_height = model_config["image_height"]
        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def predict(self, image: Image.Image):
        image = image.resize((self.image_width, self.image_height))
        pt_image = TF.to_tensor(image).unsqueeze(0)
        proba_maps, _ = self.model(pt_image)
        proba_maps = torch.sigmoid(proba_maps).squeeze(0).cpu().numpy()
        polygons, labels, scores = self.post_processor(proba_maps)
        return polygons, labels, scores, proba_maps
