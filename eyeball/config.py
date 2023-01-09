import yaml
from os import path
from .tools.dict import munchify
from argparse import Namespace
import logging


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
