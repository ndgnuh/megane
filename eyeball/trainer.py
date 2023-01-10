from pytorch_lightning.lite import LightningLite
from itertools import cycle
from torch import nn, optim

from . import ops
from .models import losses
from .models.detector import Detector
from .tools.init import init_from_config


def init_loss_fn(config):
    name = config['loss']
    options = config.get('loss_options', {})
    return init_from_config(losses, name, options)


class Trainer(LightningLite):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.train_config = train_config
        self.model_config = model_config
        self.name = model_config['name']
        self.best_weight_path = path.join(
            "storage/weights/",
            f"{self.name}-best.pt"
        )
        self.latest_weight_path = path.join(
            "storage/weights/",
            f"{self.name}-latest.pt"
        )

        self.model = Detector(model_config)
        self.loss = init_loss_fn(model_config)

        self.optimizer = init_from_ns(
            optim,
            train_config['optimizer'],
            train_config.get('optimizer_options', {}),
            self.model.parameters()
        )
        if 'lr_scheduler' in train_config:
            self.lr_scheduler = init.init_from_ns(
                optim.lr_scheduler,
                train_config['lr_scheduler'],
                train_config.get('lr_scheduler_options', {}),
                self.optimizer
            )

        for k in ["total_steps", "print_every", "validate_every"]:
            setattr(self, k, train_config[k])

        self.train_loader = self.mk_dataloader(train_config['train_data'])
        self.val_loader = self.mk_dataloader(train_config['validate_data'])
