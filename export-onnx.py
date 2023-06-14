from argparse import ArgumentParser
from os import path

import torch
from torch.onnx import export

from megane import Model, ModelConfig


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o")

    args = parser.parse_args()
    config = ModelConfig.from_file(args.config)
    assert (
        config.inference_weight is not None
    ), "No pretrained weight, the model is useless"

    # Load model
    model = Model(config)
    weight = torch.load(config.inference_weight, map_location="cpu")
    model.load_state_dict(weight)
    model.set_infer(True)

    # OUtput file
    if args.output:
        output_file = args.output
    else:
        output_file = path.splitext(args.config)[0]
        output_file = path.basename(f"{output_file}.onnx")

    # Example input
    sz = (1, 3, config.image_size, config.image_size)
    example_input = torch.zeros(sz)
    export(
        model,
        example_input,
        output_file,
        do_constant_folding=True,
        input_names=["images"],
    )
    print(f"Exported to {output_file}")


if __name__ == "__main__":
    main()
