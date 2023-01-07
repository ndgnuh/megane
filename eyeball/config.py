import yaml
from os import path
from .tools.dict import munchify


def read_yaml(file):
    with open(file) as f:
        ret = yaml.load(f, Loader=yaml.FullLoader)
        ret['name'] = get_name(file)
        ret = munchify(ret)
        return ret


def get_name(file):
    return path.splitext(path.basename(file))[0]
