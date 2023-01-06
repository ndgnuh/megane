from functools import wraps
import sys
import inspect


def remeber(init_method):
    @wraps(init_method)
    def wrap_init(self, *arg, **kwargs):
        super(type(self), self).__init__()
        allkwargs = inspect.getcallargs(init_method, self, *arg, **kwargs)
        for k, v in allkwargs.items():
            setattr(self, k, v)
        init_method(self, *arg, **kwargs)
    return wrap_init


sys.modules[__name__] = remeber
