from .common import *
from .detr_poly import DetrProcessor as DetrProcessor8
from .detr_xyxy import DetrProcessor as DetrProcessor4
from ..structures import ModelConfig, ModelType

def get_processor(config: ModelConfig):
    if config.type == ModelType.DETR_POLYGON:
        return DetrProcessor8(config.image_width,
                             config.image_height)
    elif config.type == ModelType.DETR_XYXY:
        return DetrProcessor4(config.image_width, config.image_height)
    else:
        raise NotImplementedError(f"Unsupported model type {config.type}")

