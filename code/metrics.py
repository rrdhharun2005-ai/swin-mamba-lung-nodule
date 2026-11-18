"""Loss and segmentation metrics."""
import torch
import torch.nn.functional as F


def dice_bce_loss(logits, target, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + smooth) / (p.sum() + target.sum() + smooth)
    return bce + dice


class SegMeter:
    """Accumulates pixel- and slice-level metrics over a dataset.

    dice_global : 2TP / (2TP+FP+FN) over all pixels of all slices
    dice_pos    : mean per-slice Dice over slices that contain a nodule
    iou_global, precision, recall (sensitivity): pixel level, all slices
    fp_slice_rate: share of nodule-free slices where the model predicts >= min_area pixels
    """

    def __init__(self, thr=0.5, min_area=10):
        self.thr, self.min_area = thr, min_area
        self.tp = self.fp = self.fn = 0.0
        self.pos_dice, self.neg_total, self.neg_fp = [], 0, 0

    @torch.no_grad()
    def update(self, logits, target):
        pred = (torch.sigmoid(logits.float()) > self.thr).float()
        tp = (pred * target).flatten(1).sum(1)
        ps, ts = pred.flatten(1).sum(1), target.flatten(1).sum(1)
        self.tp += tp.sum().item()
        self.fp += (ps - tp).sum().item()
        self.fn += (ts - tp).sum().item()
        pos = ts > 0
        if pos.any():
            self.pos_dice += (2 * tp[pos] / (ps[pos] + ts[pos])).tolist()
        neg = ~pos
        self.neg_total += int(neg.sum())
        self.neg_fp += int((ps[neg] >= self.min_area).sum())

    def compute(self):
        eps = 1e-8
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "dice_global": 2 * tp / (2 * tp + fp + fn + eps),
            "dice_pos": sum(self.pos_dice) / max(len(self.pos_dice), 1),
            "iou_global": tp / (tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "fp_slice_rate": self.neg_fp / max(self.neg_total, 1),
        }
