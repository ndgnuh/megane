import yaml
from .tools.dict import munchify
from argparse import Namespace
import logging
from os import path, listdir

thisdir = path.dirname(__file__)
config_path = path.join(thisdir, "eyeball_configs")
config_files = [
    "db_efficientnet_b3.yml",
    "db_mobilenet_v2.yml",
    "db_mobilenet_v3_large.yml",
    "db_mobilenet_v3_small.yml",
    "db_resnet18.yml",
    "db_resnet34.yml",
    "db_resnet50_small.yml",
    "db_resnet50.yml",
    "db_shufflenet_v2.yml",
]
config_files = [path.join(config_path, file) for file in config_files]
configs = {
    path.splitext(path.basename(c))[0]: c for c in config_files
}


def read_yaml(file):
    with open(file) as f:
        ret = yaml.load(f, Loader=yaml.FullLoader)
        ret['name'] = get_name(file)
        ret = munchify(ret)
        return ret


def get_name(file):
    return path.splitext(path.basename(file))[0]


MODEL_CONFIG = [
    ("weights", False),
    ("backbone", True),
    ("backbone_options", True),
    ("head", True),
    ("head_options", True),
    ("input_size", True),
    ("activation", True),
    ("processor", True),
    ("processor_options", True),
    ("loss", True),
    ("loss_options", True),
    ("image_width", True),
    ("image_height", True),
]

schemas = Namespace(MODEL_CONFIG=MODEL_CONFIG)


def validate_config(config, schema, strict=False):
    keys = {}
    for key, is_optional in schema:
        if not is_optional:
            assert key in config
        keys[key] = False
    for k, v in config.items():
        if keys.get(k, True):
            if strict:
                raise ValueError(f"Unexpected key {key}")
            else:
                logging.warning(f"Warning: Unexpected key {key}")
    return True
