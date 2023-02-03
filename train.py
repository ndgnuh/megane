import questionary as Q
from argparse import ArgumentParser
from os import path


def get_option_interactive():

    def ask_int(prompt, default):
        q = Q.text(prompt, validate=str.isnumeric, default=str(default))
        return int(q.ask())

    def ask_file(prompt, default=""):
        q = Q.path(prompt, validate=path.isfile, default=str(default))
        return q.ask()

    def ask_float(prompt, default):
        def isfloat(x: str):
            try:
                float(x)
                return True
            except Exception:
                return False
        q = Q.text(prompt, validate=isfloat, default=str(default))
        return q.ask
    options = dict()
    options["model_config"] = ask_file("Model config (.yml):")
    options["train_data"] = ask_file("Train data index:")
    train_data_dir = path.dirname(options["train_data"])
    options["val_data"] = ask_file("Validate data index:", train_data_dir)
    options["batch_size"] = ask_int("Batch size:", 4)
    options["num_workers"] = ask_int("Number of workers", 1)
    options["total_steps"] = ask_int("Total training steps:", 1000)
    options["validate_every"] = ask_int("Validate every:", 50)
    options["learning_rate"] = ask_float("Learning rate:", 3e-4)

    return options


def get_option_shell(args):
    parser = ArgumentParser()
    parser.add_argument("-c",
                        dest="model_config",
                        help="Model configuration",
                        required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--batch-size", default=4, type=int, required=True)
    parser.add_argument("--total-steps", "-N", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    options = parser.parse_args(args)
    options = {k: getattr(options, k) for k in vars(options)}
    print(options)
    return options


def main():
    from megane.trainer import Trainer
    parser = ArgumentParser()
    parser.add_argument("-i",
                        action="store_true",
                        help="interactive configuration",
                        dest="interactive",
                        default=False)

    args, unknowns = parser.parse_known_args()
    if args.interactive:
        options = get_option_interactive()
    else:
        options = get_option_shell(unknowns)

    trainer = Trainer(**options)
    trainer.train()


if __name__ == "__main__":
    main()
