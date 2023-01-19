import yaml
from copy import copy
from typing import List, Dict
from os import path


def get_name(file):
    return path.splitext(path.basename(file))[0]


def read_config(file):
    with open(file) as f:
        ret = yaml.load(f, Loader=yaml.FullLoader)
        ret['name'] = get_name(file)
        return ret


def init_from_config(cls, config: Dict, keys: List, **options: Dict):
    # Can't use config.get(k, options[k])
    # Because options[k] is eager
    kwargs = {}
    for k, v in options.items():
        if k not in configs:
            configs[k] = options[v]

    kwargs = {
        for k in keys
    }
    return cls(**kwargs)
