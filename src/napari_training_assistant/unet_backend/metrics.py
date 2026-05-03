from __future__ import annotations

import torch


def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8):
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    tgt = targets.float()

    tp = (pred * tgt).sum().item()
    fp = (pred * (1.0 - tgt)).sum().item()
    fn = ((1.0 - pred) * tgt).sum().item()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "f1": float(f1),
    }


def multiclass_mean_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-8):
    pred = torch.argmax(logits, dim=1)
    tgt = targets.long()

    dices = []
    ious = []
    f1s = []

    for cls in range(1, num_classes):  # skip background in mean
        p = (pred == cls).float()
        t = (tgt == cls).float()

        tp = (p * t).sum().item()
        fp = (p * (1.0 - t)).sum().item()
        fn = ((1.0 - p) * t).sum().item()

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)

        dices.append(dice)
        ious.append(iou)
        f1s.append(f1)

    if len(dices) == 0:
        return {"dice": 0.0, "iou": 0.0, "f1": 0.0}

    return {
        "dice": float(sum(dices) / len(dices)),
        "iou": float(sum(ious) / len(ious)),
        "f1": float(sum(f1s) / len(f1s)),
    }