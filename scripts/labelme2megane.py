import json
from PIL import Image
from io import BytesIO
from base64 import b64decode
from os import path, listdir, makedirs


def image_from_bytes(bs):
    io = BytesIO(bs)
    return Image.open(io)


def image_to_bytes(bs):
    io = BytesIO(bs)
    return Image.open(io)


def readlabelme(file):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    shapes = data['shapes']
    width = data['imageWidth']
    height = data['imageHeight']
    image_path = data['imagePath']
    image_data = data['imageData']

    polygons = []
    labels = []
    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            [x1, y1], [x2, y2] = shape['points']
            polygon = [
                (x1 / width, y1 / height),
                (x2 / width, y1 / height),
                (x2 / width, y2 / height),
                (x1 / width, y2 / height),
            ]
        elif shape['shape_type'] == 'polygon':
            polygon = [
                (x / width, y / height)
                for (x, y) in shape['points']
            ]

        labels.append(shape['label'])
        polygons.append(polygon)
    label_maps = sorted(list(set(list(labels))))
    labels = [label_maps.index(label) for label in labels]

    data = dict(
        width=width,
        height=height,
        polygons=polygons,
        labels=labels,
        image_path=image_path,
    )
    if image_data is not None:
        image = image_from_bytes(b64decode(image_data))
    else:
        image_path = path.join(path.dirname(file), image_path)
        image = Image.open(image_path)
    return data, image


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    makedirs(args.output, exist_ok=True)
    for file in listdir(args.input):
        if not file.endswith(".json"):
            continue
        in_json_path = path.join(args.input, file)
        data, image = readlabelme(in_json_path)
        out_json_path = path.join(args.output, path.basename(file))
        out_image_path = path.join(args.output, data['image_path'])
        with open(out_json_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        image.save(out_image_path)
