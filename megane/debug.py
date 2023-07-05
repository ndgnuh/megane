import sys
import logging
import time
from functools import wraps


logger = logging.getLogger("megane-debug")
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
formatter = logging.Formatter()


def with_timer(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        a = time.perf_counter()
        ret = f(*args, **kwargs)
        b = time.perf_counter()
        logger.debug(f"[timer] ({f.__name__}) {b - a:.4e}s")
        return ret

    return wrapped
