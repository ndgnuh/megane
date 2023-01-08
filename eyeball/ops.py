import json


def draw_rects(image, rects, **kwargs):
    from PIL import ImageDraw
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for rect in rects:
        draw.rectangle(rect, **kwargs)
    return result


def read_json(f):
    with open(f) as io:
        return json.load(io)


def top_left_letterbox(image):
    width, height = image.size
    size = max(width, height)
    output = Image.new("RGB", (size, size), (125, 125, 125))
    output.paste(image, (0, 0))
    return output


def xyxy_to_ccwh(box):
    x1, y1, x2, y2 = box
    w = (x2 - x1)
    h = (y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


def preprocess_image(image, width, height):
    image = top_left_letterbox(image)
    # image.thumbnail((width, height), resample=Image.BILINEAR)
    image = image.resize((width, height), resample=Image.BILINEAR)
    return image


def normalize(box, width, height, norm_constant=1000):
    x1, y1, x2, y2 = box
    x1 = x1 * norm_constant / width
    y1 = y1 * norm_constant / height
    x2 = x2 * norm_constant / width
    y2 = y2 * norm_constant / height
    return x1, y1, x2, y2


def denormalize(box, width, height, norm_constant=1000):
    x1, y1, x2, y2 = box
    x1 = x1 / norm_constant * width
    y1 = y1 / norm_constant * height
    x2 = x2 / norm_constant * width
    y2 = y2 / norm_constant * height
    return x1, y1, x2, y2


def offset_rect(xyxy, r, expand=False):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1

    # Expand distance
    A = w * h
    L = 2 * (w + h)
    d = A * (1 - r**2) / L

    if expand:
        d = -d

    # Expanded
    x1 = max(x1 + d, 0)
    y1 = max(y1 + d, 0)
    x2 = x2 - d
    y2 = y2 - d
    return x1, y1, x2, y2
