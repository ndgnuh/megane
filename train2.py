from argparse import ArgumentParser
from megane2.loaders import megane_dataloader
from megane2 import transforms, losses, configs
from megane2.trainer import Trainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c",
                        dest="model_config",
                        help="Model configuration",
                        required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--wd", type=int, default=1024)
    parser.add_argument("--ht", type=int, default=1024)
    parser.add_argument("--total-steps", "-N", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(
        configs.read_config(args.model_config),
        train_data=args.train_data,
        val_data=args.val_data,
        total_steps=args.total_steps,
        num_workers=args.num_workers,
        validate_every=args.validate_every,
    )
    trainer.train()


if __name__ == "__main__":
    main()
