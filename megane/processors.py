"""
There are two thing that need to be distinguished here.

- processor: the one which process the input, however, sometime processing the input means changing the ground truth, therefore the processor has access to full context.
- encoder/decoder: the one which process the target, it needs the input to know the image size, therefore has the access to the full context.

So.... their signature should be:

processor: Sample -> Sample
encoder: Sample -> Tuple[Input, GroundTruth]
decoder: Tuple[Input, GroundTruth | ModelOutput] -> Sample

With this, some weird problem arise though.
1. So the encoder need to call the processor?
2. The predictor only need to call the processor?
3. The ground truth and the model outputs are different from each other, where should we bridge them, maybe in `model.head.post_process`?
4. This makes me sound like java people, and I don't like it
"""
from dataclasses import dataclass
from typing import Tuple

import simpoly
from PIL import Image
from lenses import bind

from megane.registry import processors
from megane.data import Sample
from megane.utils import init_from_ns


@processors.register(name="letterbox")
@dataclass
class Letterbox:
    image_size: Tuple[int, int]
    fill_value: Tuple[int, int, int] = (127, 127, 127)

    def __call__(self, sample: Sample) -> Sample:
        # Unpack data
        image = sample.image
        boxes = sample.boxes
        fill_value = self.fill_value
        width, height = self.image_size
        original_width, original_height = image.size

        # New size of the image while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > width / height:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_width = int(height * aspect_ratio)
            new_height = height

        # Calculate the padding for both width and height
        pad_width = (width - new_width) // 2
        pad_height = (height - new_height) // 2

        # Resize and paste the resized image onto the letterboxed canvas
        final = Image.new("RGB", (width, height), fill_value)
        resized = image.resize((new_width, new_height))
        final.paste(resized, (pad_width, pad_height))

        # Bounding box letterboxing
        new_boxes = []
        for box in boxes:
            box = simpoly.scale_to(box, new_width, new_height)
            box = [(x + pad_width, y + pad_height) for (x, y) in box]
            box = simpoly.scale_from(box, width, height)
            new_boxes.append(box)

        # Create new sample
        sample = bind(sample).image.set(final)
        sample = bind(sample).boxes.set(new_boxes)
        return sample


@processors.register(name="resize")
@dataclass
class Resize:
    """Resize the image inside a sample, returns a new sample.

    # Args:
        image_size (Tuple[int, int]): The expected width and height of the image
    """

    image_size: Tuple[int, int]

    def __call__(self, sample: Sample) -> Sample:
        image = sample.image.resize(self.image_size)
        new_sample = bind(sample).image.set(image)
        return new_sample


def get_processor(config):
    return init_from_ns(processors, config.input_process)
