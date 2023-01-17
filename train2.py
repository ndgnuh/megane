from argparse import ArgumentParser
from megane2.loaders import megane_dataloader
from megane2 import transforms


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validate-data", required=True)
    parser.add_argument("--wd", type=int, default=1024)
    parser.add_argument("--ht", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    transform = transforms.Compose([
        transforms.Resize(args.wd, args.ht),
        transforms.DBPreprocess(),
    ])
    train_loader = megane_dataloader(
        args.train_data,
        transform=transform
    )
    validate_loader = megane_dataloader(
        args.validate_data,
        transform=transform
    )
    print(next(iter(train_loader)))


if __name__ == "__main__":
    main()
