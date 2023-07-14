from argparse import ArgumentParser
from os import path

import torch
from torchvision.transforms import functional as TF
from torch.onnx import export
from PIL import Image

from megane import Model, MeganeConfig, Sample, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output", default=None)

    args = parser.parse_args()
    config = MeganeConfig.from_file(args.config)

    # Load model
    model, processor, _, _ = init_model(config)
    model.set_infer(True)

    # OUtput file
    if args.output:
        output_file = args.output
    else:
        output_file = path.splitext(args.config)[0]
        output_file = path.basename(f"{output_file}.onnx")

    # Pseudo input
    image = Image.new("RGB", (1000, 1000), 0)
    sample = Sample(image)
    sample = processor(sample)
    inputs = TF.to_tensor(sample.image).unsqueeze(0)

    # Export
    export(
        model,
        inputs,
        output_file,
        do_constant_folding=True,
        input_names=["images"],
    )
    print(f"Exported to {output_file}")


if __name__ == "__main__":
    main()
