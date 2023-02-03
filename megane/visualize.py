from PIL import Image, ImageDraw


def draw_polygons(image: Image.Image, polygons, **options):
    options.setdefault("outline", (255, 0, 0))
    w, h = image.size
    polygons = [
        [
            (int(x * w), int(y * h))
            for (x, y) in polygon
        ]
        for polygon in polygons
    ]
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, **options)
    return image
