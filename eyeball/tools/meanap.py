import numpy as np


def rect_iou(pr, gt):
    x1_t, y1_t, x2_t, y2_t = gt
    x1_p, y1_p, x2_p, y2_p = pr

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def confusion(prs, gts, iou_threshold=0.5):
    """
    Returns [TP, FP, FN], in this exact order.
    True negative is not used, therefore is not returned.

    prs = predictions
    gts = ground truths

    prs, gts: List[Tuple[int, int, int, int]]
    """
    num_prs = len(prs)
    num_gts = len(gts)

    # Empty predictions
    if num_prs == 0:
        tp = 0
        fp = 0
        fn = num_gts
        return tp, fp, fn

    # Empty ground truths
    if num_gts == 0:
        tp = 0
        fp = num_prs
        fn = 0
        return tp, fp, fn

    # IOU matrix between GT and PR
    gt_idx = []
    pr_idx = []
    ious = []
    for i_pr, pr in enumerate(prs):
        for i_gt, gt in enumerate(gts):
            iou = rect_iou(pr, gt)
            if iou > iou_threshold:
                gt_idx.append(i_gt)
                pr_idx.append(i_pr)
                ious.append(iou)

    # No matches
    if len(ious) == 0:
        tp = 0
        fp = num_prs
        fn = num_gts
        return tp, fp, fn

    # Final case
    # Count all the match
    # Use a lookup table to avoid searching in the loop
    num_matches = 0  # This is TP, named to num_matches for the semantic
    pr_unmatched = [True] * num_prs
    gt_unmatched = [True] * num_gts
    for i in np.argsort(ious)[::-1]:
        pr_i = pr_idx[i]
        gt_i = gt_idx[i]
        if pr_unmatched[pr_i] and gt_unmatched[gt_i]:
            num_matches += 1
            # If the boxes are unmatched, add them to matches
            pr_unmatched[pr_i] = gt_unmatched[gt_i] = False
    tp = num_matches
    fp = num_prs - num_matches
    fn = num_gts - num_matches

    return tp, fp, fn


def calc_mean_ap(batch_prs,
                 batch_gts,
                 score_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
    """
    Return Mean Average Precision.

    """
    # Calculate precisions and recalls
    precisions = []
    recalls = []
    for prs, gts in zip(batch_prs, batch_gts):
        prs = [box for box, score in zip(prs['boxes'], prs['scores'])
               if score >= score_threshold]
        gts = [box for box in gts['boxes']]
        tp, fp, fn = confusion(prs, gts, iou_threshold=iou_threshold)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        precisions.append(precision)
        recalls.append(recall)

    # Calculate mean AP
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ap = []
    for recall_level in np.arange(0, 1, step=0.1):
        try:
            precision_idx = np.argwhere(recalls >= recall_level).flatten()
            precision = precisions[precision_idx].max()
        except ValueError:
            precision = 0.0
        ap.append(precision)
    mean_ap = np.mean(ap)

    return mean_ap
