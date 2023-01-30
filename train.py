import questionary as Q
from os import path
from megane2.trainer import Trainer
from megane2.configs import read_config


def ask_int(prompt, default):
    q = Q.text(prompt, validate=str.isnumeric, default=str(default))
    return int(q.ask())


def ask_file(prompt, default=""):
    q = Q.path(prompt, validate=path.isfile, default=str(default))
    return q.ask()


def main():
    options = dict()
    options["model_config"] = read_config(ask_file("Model config (.yml):"))
    options["train_data"] = ask_file("Train data index:")
    train_data_dir = path.dirname(options["train_data"])
    options["val_data"] = ask_file("Validate data index:", train_data_dir)
    options["batch_size"] = ask_int("Batch size:", 4)
    options["num_workers"] = ask_int("Number of workers", 1)
    options["total_steps"] = ask_int("Total training steps:", 1000)
    options["validate_every"] = ask_int("Validate every:", 50)

    trainer = Trainer(**options)
    trainer.run()


if __name__ == "__main__":
    main()
