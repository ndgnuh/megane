def get_matching_index(pr_boxes: np.ndarray, gt_boxes: np.ndarray):
    pr_polygons = [Polygon(box) for box in pr_boxes]
    gt_polygons = [Polygon(box) for box in gt_boxes]
    n, m = len(pr_polygons), len(gt_polygons)

    # Calculate scoring matrix
    iou_matrix = np.zeros([n, m])
    for i in range(n):
        pr = pr_polygons[i]
        if not pr.is_valid:
            continue

        for j in range(m):
            gt = gt_polygons[j]
            inter = pr.intersection(gt).area
            uni = pr.union(gt).area
            iou_matrix[i, j] = inter / uni

    # Get the maximal matching based on score matrix
    mask = np.zeros(n, bool)
    gt_order = np.zeros(m, int)
    count = 0
    while count < m:
        idx = np.argmax(iou_matrix)
        i, j = np.unravel_index(idx, (n, m))

        mask[i] = True
        gt_order[count] = j
        iou_matrix[i, :] = -np.inf
        iou_matrix[:, j] = -np.inf

        count += 1
    return mask, gt_order


class HungarianMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def to_np(self, x):
        return x.detach().cpu().numpy()

    def forward(self, pr_boxes, gt_boxes, pr_classes_logits, gt_classes):
        device = pr_boxes.device
        l_loss = 0
        c_loss = 0

        count = 0
        for pr_boxes_i, gt_boxes_i, pr_classes_logits_i, gt_classes_i in zip(
            pr_boxes, gt_boxes, pr_classes_logits, gt_classes
        ):
            count = count + 1
            pr_boxes_np = self.to_np(pr_boxes_i)
            gt_boxes_np = self.to_np(gt_boxes_i)
            mask, gt_order = get_matching_index(pr_boxes_np, gt_boxes_np)
            mask = torch.BoolTensor(mask).to(device)
            gt_order = torch.LongTensor(gt_order).to(device)

            # Localization loss
            neg = pr_boxes_i[~mask]
            l_loss += self.ce(pr_boxes_i[mask], gt_boxes_i[gt_order])
            l_loss += self.ce(neg, torch.zeros_like(neg))

            # Classification loss
            neg = pr_classes_logits_i[mask]
            # ic(type(gt_classes), np.array(gt_classes).shape)
            c_loss += self.ce(pr_classes_logits_i[mask], gt_classes_i[gt_order])
            c_loss += self.ce(neg, torch.zeros_like(neg))

        loss = l_loss + c_loss
        loss = loss / count / 2
        return loss
