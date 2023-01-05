def init_from_config(namespace, config):
    name = config.pop("name")
    Class = getattr(namespace, name)
    return Class(**config)
