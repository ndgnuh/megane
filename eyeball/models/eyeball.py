from torch import nn


from . import heads
from ..tools.init import init_from_config, init_from_ns


class backbones:
    # Stub
    def mobilenet_v3_large(



class EyeBall(nn.Sequential):
    def __init__(self, config: Dict):
        super().__init__()
        backbone_config = config['backbone']
        backbone_name = backbone_config.pop("name")
        self.backbone = init_from_ns(backbones, config['backbone'])
        self.head = init_from_config(heads, config['head'])
