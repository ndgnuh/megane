from .db import DBProcessor
from .retina import RetinaProcessor


def ProcessorMixin(mode):
    if mode == "db":
        return DBProcessor()
    elif mode == "retina":
        return RetinaProcessor()
    else:
        raise ValueError(f"Unsupported mode {mode}")
