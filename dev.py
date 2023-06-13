import torch
from icecream import install
from matplotlib import pyplot as plt

from src import ModelConfig, TrainConfig, Trainer, read_config

install()

model_config = ModelConfig.from_file("configs/db_fpn_base.yml")
train_config = TrainConfig.from_file("configs/training.yml")
Trainer(train_config, model_config).train()
