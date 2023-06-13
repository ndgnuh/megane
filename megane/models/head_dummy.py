from torch import nn
from torchvision.transforms import functional as TF

from .. import utils
from ..configs import ModelConfig
from .api import ModelAPI


class DummyHead(ModelAPI):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size

    def forward(self, outputs, targets=None):
        from icecream import ic

        ic(outputs.shape)
        return outputs

    def compute_loss(self, outputs, targets):
        return outputs.sum()

    def encode_sample(self, sample):
        image = utils.prepare_input(sample.image, config.image_size, config.image_size)
        return image, image

    def decode_sample(self, inputs, outputs):
        return Sample(image=TF.to_pil_image(inputs))
