from argparse import ArgumentParser

from megane import ModelConfig, TrainConfig, Trainer
from megane.configs import MeganeConfig


def main():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        import icecream

        icecream.install()

    config = MeganeConfig.from_file(args.config)
    Trainer(config).train()


if __name__ == "__main__":
    main()
