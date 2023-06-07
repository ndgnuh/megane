from os import path
from typing import List, Union, Dict, Optional
from pydantic import BaseModel as _BaseModel, Field


class BaseModel(_BaseModel):
    @classmethod
    def from_file(cls, config_path):
        return cls.parse_obj(read(config_path))


def read(config_path):
    import yaml

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if "name" not in config:
        config["name"] = path.splitext(path.basename(config_path))[0]
    return config


def default_fabric_config():
    return dict(accelerator="auto")


class TrainConfig(BaseModel):
    """Training configuration schema.

    Attributes:
        train_data:
            Path to train data annotation
        val_data:
            Path to validate data annotation
        lr:
            Base learning rate
        total_steps:
            Number of training iterations
        print_every:
            Logging interval in steps
        validate_every:
            Validate interval in steps
        dataloader:
            Dataloader config kwargs, default to `{}`
        fabric:
            Torch Fabric config, default to `dict(accelerator='auto')`
    """

    train_data: str
    val_data: str

    lr: float = 1e-4

    total_steps: int
    print_every: int
    validate_every: int

    dataloader: Dict = Field(default_factory=dict)
    fabric: Dict = Field(default_factory=default_fabric_config)


class HeadConfig(BaseModel):
    classes: List[str]
    contour_loss_kernel: int = 3

    @property
    def num_classes(self):
        return len(self.classes)


class FPNConfig(BaseModel):
    arch: str
    layers: List[str]
    hidden_size: int
    feature_module: Optional[str] = None


class FViTConfig(BaseModel):
    patch_size: int
    num_blocks: List[int] = [3, 6, 3]
    hidden_sizes: List[int] = [32, 64, 128, 96]
    num_attention_heads: List[int] = [4, 8, 16]

    @property
    def hidden_size(self):
        last_hidden = self.hidden_sizes[-1]
        last_hidden = last_hidden // (2**self.num_stages)
        return last_hidden

    @property
    def num_stages(self):
        return len(self.num_blocks)


class ModelConfig(BaseModel):
    name: str
    image_size: int
    head: Union[HeadConfig]
    backbone: Union[FPNConfig, FViTConfig]

    continue_weight: Optional[str] = None
    inference_weight: Optional[str] = None

    # Properties that must be provided by the model head configs
    @property
    def classes(self):
        return self.head.classes

    # Properties that must be provided by the model backbone configs
    @property
    def hidden_size(self):
        return self.backbone.hidden_size

    # Stuffs for trainer
    @property
    def latest_weight_name(self):
        return f"{self.name}.latest.pt"

    @property
    def best_weight_name(self):
        return f"{self.name}.best.pt"
