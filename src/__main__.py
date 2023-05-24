from argparse import ArgumentParser
from .structures.configs import ModelConfig, TrainConfig
from .trainer import Trainer

def main():
    from icecream import install; install()
    parser = ArgumentParser()
    parser.add_argument("-m", dest="model_config", required=True)
    parser.add_argument("-e", dest="train_config", required=True)

    args = parser.parse_args()
    model_config = ModelConfig.from_file(args.model_config)
    train_config = TrainConfig.from_file(args.train_config)

    trainer = Trainer(train_config, model_config)
    trainer.train()


if __name__ == "__main__":
    main()
