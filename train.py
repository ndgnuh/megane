from megane import Trainer, read_yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model-config", "-c",
                    dest="model_config",
                    required=True, )
args = parser.parse_args()

train_config = read_yaml("configs/train.yaml")
model_config = read_yaml(args.model_config)
trainer = Trainer(model_config, train_config)
trainer.run()
