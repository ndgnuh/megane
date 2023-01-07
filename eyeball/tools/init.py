from copy import copy


def init_from_config(namespace, config):
    name = config.pop("name")
    Class = getattr(namespace, name)
    return Class(**config)


def init_from_ns(namespace, name, options, *a, **extra_options):
    options = copy(options)
    options.update(extra_options)
    Class = getattr(namespace, name)
    return Class(*a, **options)
