from os import path, listdir

thisdir = path.dirname(__file__)


def list_configs():
    return [path.join(thisdir, file) for file in listdir(thisdir)]


config_files = list_configs()
