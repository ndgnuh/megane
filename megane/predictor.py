import torch
from PIL import Image

from megane.configs import ModelConfig
from megane.data import Sample
from megane.models import Model
from megane.utils import prepare_input


class Predictor:
    def __init__(self, model_config: ModelConfig):
        self.model = Model(model_config)
        self.model.load_state_dict(
            torch.load(model_config.inference_weight, map_location="cpu")
        )
        self.image_size = model_config.image_size

    def predict(self, image: Image) -> Sample:
        input_image = prepare_input(image, self.image_size, self.image_size)
        input_image = torch.tensor(input_image).unsqueeze(0)
        raw_outputs = self.model(input_image)
        sample = self.model.decode_sample(input_image[0], raw_outputs[0])
        sample.image = image
        return sample
