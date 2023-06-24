from argparse import ArgumentParser

from megane import ModelConfig, TrainConfig, Trainer


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", required=True)
    parser.add_argument("-m", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        import icecream

        icecream.install()

    model_config = ModelConfig.from_file(args.m)
    train_config = TrainConfig.from_file(args.e)
    Trainer(train_config, model_config).train()


if __name__ == "__main__":
    main()
