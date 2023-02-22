import numpy as np
import cv2
from PIL import Image


def polygon_area(poly):
    # https://en.wikipedia.org/wiki/Polygon#Area
    n = len(poly)
    area = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    area /= 2
    return area


def polygon_perimeter(poly):
    n = len(poly)
    peri = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        peri += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return peri


def offset_poly(poly, offset=1):
    n = len(poly)
    offset_lines = []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Calculate the direction vector
        vx, vy = x2 - x1, y2 - y1

        # flip direction vector by 90deg
        vx, vy = vy, -vx

        # normalize
        length = (vx**2 + vy**2)**0.5 + 1e-6
        vx, vy = vx / length, vy / length

        # Offset line
        nx1 = x1 + vx * offset
        ny1 = y1 + vy * offset
        nx2 = x2 + vx * offset
        ny2 = y2 + vy * offset

        offset_lines.append((nx1, ny1, nx2, ny2))

    # New poly vertices are the intersection of the offset lines
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    new_poly = []
    for i in range(n):
        (x1, y1, x2, y2) = offset_lines[i]
        (x3, y3, x4, y4) = offset_lines[(i + 1) % n]
        deno = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-6
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
             (x1 - x2) * (x3 * y4 - x4 * y3)) / deno
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
             (y1 - y2) * (x3 * y4 - x4 * y3)) / deno
        new_poly.append((x, y))
    return new_poly


def build_db_target(
    polygons: np.ndarray,
    image_width: int,
    image_height: int,
    shrink_ratio: float = 0.4,
    min_thresh: float = 0.3,
    max_thresh: float = 0.7,
    min_box_size: int = 10,
):
    # The mask are here
    # Polygons must be in the shape: [n, 4, 2]
    # Denormalize polygon
    polygons = polygons.copy()
    polygons[:, :, 0] *= image_width
    polygons[:, :, 1] *= image_height

    areas = [polygon_area(poly) for poly in polygons]
    lengths = np.array([polygon_perimeter(poly) for poly in polygons])
    xmaxs = polygons[..., 0].max(axis=1)
    xmins = polygons[..., 0].min(axis=1)
    mask_polygons = (xmaxs - xmins) >= min_box_size

    # Shrinks
    shrink_distances = np.array([
        (1 - shrink_ratio**2) * A / L
        for poly, A, L in zip(polygons, areas, lengths)
    ])
    shrinked_polygons = np.stack([
        offset_poly(polygon, -distance)
        for (polygon, distance) in zip(polygons, shrink_distances)
    ])
    expanded_polygons = np.stack([
        offset_poly(polygon, distance)
        for (polygon, distance) in zip(polygons, shrink_distances)
    ])

    # To int
    polygons = polygons.round().astype(int)
    shrinked_polygons = shrinked_polygons.round().astype(int)
    expanded_polygons = expanded_polygons.round().astype(int)

    # Draw maps and masks
    proba_map = np.zeros((image_height, image_width), dtype='uint8')
    proba_mask = np.ones((image_height, image_width), dtype='uint8') * 255
    thresh_map = np.zeros((image_height, image_width), dtype='float32')
    thresh_mask = np.ones((image_height, image_width), dtype='uint8') * 255

    # Proba map/mask
    cv2.fillPoly(proba_map, shrinked_polygons[mask_polygons], (255,))
    cv2.fillPoly(proba_mask, polygons[~mask_polygons], (0,))

    # Threshold map/mask
    # Draw on a canvas and add it because drawing everything at once
    # will create holes in the border of threshold boxes.
    for (ex, sh) in zip(
            expanded_polygons[mask_polygons],
            shrinked_polygons[mask_polygons]
    ):
        canvas = np.zeros_like(thresh_map, dtype='uint8')
        cv2.fillPoly(canvas, [ex], (255,))
        cv2.fillPoly(canvas, [sh], (0,))
        canvas = cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_3)
        masked = canvas[canvas > 0]
        if masked.size < 1:
            continue
        max_dist = masked.max()
        min_dist = masked.min()
        canvas = ((canvas - min_dist) /
                  (max_dist - min_dist + 1e-6)) * (canvas > 0)
        thresh_map = np.clip(thresh_map + canvas, 0, 1)

    thresh_map = (255 * thresh_map).astype('uint8')
    cv2.fillPoly(thresh_mask, polygons[~mask_polygons], (0,))

    return (
        Image.fromarray(proba_map),
        Image.fromarray(proba_mask),
        Image.fromarray(thresh_map),
        Image.fromarray(thresh_mask),
    )


def mask_to_polygons(
    proba_map: np.ndarray,
    min_box_size: float = 10.0,
    min_score: float = 0.6,
    expand_ratio: float = 1.5,
    min_threshold: float = 0.7
):
    h, w = proba_map.shape
    mask = (proba_map > min_threshold).astype('uint8')
    polygons = []
    angles = []
    scores = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # max_cnt=1000
    for cnt in cnts:
        # if max_cnt==0:
        #     break
        # else:
        #     max_cnt-=1
        # Make sure the polygon has 4 corners
        if cnt.shape[0] < 4:
            continue

        # Angle is deegree
        # Convert to a rotated rectangle
        xy, wh, angle = cv2.minAreaRect(cnt)
        polygon = cv2.boxPoints((xy, wh, angle)).round().astype(int)

        # Score thresholding
        score = polygon_score(proba_map, polygon)
        if score < min_score:
            continue

        # Filter small polygon
        A = polygon_area(polygon)
        if np.power(A, 2) < min_box_size:
            continue

        # Expand polygon
        L = polygon_perimeter(polygon)
        D = expand_ratio * A / L
        polygon = offset_poly(polygon, D)
        polygons.append(polygon)
        angles.append(angle)
        scores.append(score)

    # normalize polygons
    polygons = [
        [(x / w, y / h) for (x, y) in points]
        for points in polygons
    ]
    return polygons, scores, angles


def simplify_contour(contour, min_corners=4, max_corners=6):
    '''
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    if contour.shape[0] >= min_corners and contour.shape[0] <= max_corners:
        return contour, True

    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour, False

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > max_corners:
            lb = (lb + ub)/2.
        elif len(approx) < min_corners:
            ub = (lb + ub)/2.
        else:
            return approx, True


PERMUTATIONS = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2]
]
PERMUTATIONS = PERMUTATIONS + [list(reversed(p)) for p in PERMUTATIONS]


def match_polygon(p1, p2):
    dists = [np.linalg.norm(p2[p, :] - p1, axis=1).mean()
             for p in PERMUTATIONS]
    idx = np.argmin(dists)
    return p2[PERMUTATIONS[idx]]


def crop_from_polygon(image, polygon):
    # Bounding rectangle
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    w = x2 - x1
    h = y2 - y1
    dst_pts = np.array([(0, h), (0, 0), (w, 0), (w, h)], dtype='float32')
    dst_pts = match_polygon(polygon, dst_pts)

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(
        polygon.astype('float32'), dst_pts.astype('float32'))

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(
        image, M, (int(w), int(h)), cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
    return warped


def crop_from_mask(image, mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    crops = []
    height, width = image.shape[:2]
    for cnt in cnts:
        polygon, success = simplify_contour(cnt, max_corners=4, min_corners=4)
        polygon = polygon[:, 0, :]  # 4, 1, 2 -> 4, 2
        if not success:
            continue

        # Bounding rectangle
        warped = crop_from_polygon(image, polygon)
        crops.append(warped)
    return crops


def polygon_score(proba_map, polygon):
    mask = np.zeros_like(proba_map)
    cv2.fillPoly(mask, [polygon], 1)
    return (mask * proba_map).sum() / np.count_nonzero(mask)


# import matplotlib.pyplot as plt
# def plot_img(img,size=(8,8)):
#     plt.figure(figsize=size)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
def polygon_to_mask_segment(image, annotation,new_w,new_h):
    polygon=np.array(annotation['shapes'][0]['points'])
    image=np.array(image)
    ano_h,ano_w=annotation['imageHeight'],annotation['imageWidth']
    mask = np.zeros((new_w,new_h), dtype=np.uint8)
    new_points=[]
    for i in range(len(polygon)):
        x=int(polygon[i][0]/ano_w*new_w)
        y=int(polygon[i][1]/ano_h*new_h)
        # print("before: ",polygon[i])
        # print("after: ",x,y)
        new_points.append([x,y])
    cv2.fillPoly(mask,  pts=[np.array(new_points)], color=(255,255,255))
    
    # Draw the element wit a color depending on the Class
    # cv2.drawContours(mask, [arr.astype(np.int32)], -1, (255, 255, 255), -1, cv2.LINE_AA)

    return mask

def mask_to_polygons_segment(mix_maps):
    h,w=mix_maps.shape
    cnts, _ = cv2.findContours(mix_maps.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt=cnts[np.argmax([cv2.contourArea(x) for x in cnts])]
    xy, wh, angle = cv2.minAreaRect(cnt)
    polygons,_ = simplify_contour(cnt,4,4)
    polygons = [
            [(x / w, y / h) for (x, y) in points]
            for points in [polygons[:,0,:]]
        ]
    return polygons