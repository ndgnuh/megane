class Registry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def register(self, name=None, force=False):
        def register_(obj):
            nonlocal name
            if name is None:
                name = obj.__name__
            assert name not in self
            self[name] = obj
            return obj

        return register_

    def __getattribute__(self, name):
        if name == "register":
            return object.__getattribute__(self, name)
        return self[name]


# Input preprocessors
# Transform input but keep data type
processors = Registry()

# Model backbone
backbones = Registry()

# Model heads
heads = Registry()

# Target encoders
# Transform input encode to torch tensor
target_encoders = Registry()

# Target decoders
# Decode torch tensor to input date type
target_decoders = Registry()
