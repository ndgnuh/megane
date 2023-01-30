from shapely.geometry import Polygon


def get_polygons_iou(p1, p2):
    try:
        if not isinstance(p1, Polygon):
            p1 = Polygon(p1)
        if not isinstance(p2, Polygon):
            p2 = Polygon(p2)
        inter = p1.intersection(p2).area
        uni = p1.union(p2).area
        return inter / (uni + 1e-6)
    except Exception:
        return 0


def get_tp_fp_fn(predicts, targets, iou_threshold: float = 0.8):
    num_predicts = len(predicts)
    num_targets = len(targets)

    # Special cases
    if num_predicts == 0 and num_targets == 0:
        return 0, 0, 0
    if num_predicts == 0:
        return 0, 0, num_targets
    if num_targets == 0:
        return 0, num_predicts, 0

    # Normal cases
    ious = [
        (i, j, get_polygons_iou(p1, p2))
        for (i, p1) in enumerate(predicts)
        for (j, p2) in enumerate(targets)
    ]
    ious = [(i, j, iou) for (i, j, iou) in ious if iou >= iou_threshold]
    ious = sorted(ious, key=lambda x: x[-1], reverse=True)
    predicts_visited = [False] * num_predicts
    targets_visited = [False] * num_targets
    tp = 0  # True position = correctly guess the box position
    for i, j, iou in ious:
        if iou < iou_threshold or predicts_visited[i] or targets_visited[j]:
            continue
        tp = tp + 1
        predicts_visited[i] = targets_visited[j] = True

    # False positive, guess a box but it's not there
    fp = num_predicts - tp

    # False negative, there's a box there but didn't get it
    fn = num_targets - tp
    return tp, fp, fn


def f1_score(predicts, targets, iou_threshold: float = 0.5):
    tp, fp, fn = get_tp_fp_fn(predicts, targets, iou_threshold)
    return tp / (tp + 0.5 * (fp + fn) + 1e-6)
