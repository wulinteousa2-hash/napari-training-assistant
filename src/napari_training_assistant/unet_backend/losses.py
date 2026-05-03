from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BinaryBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets.float()) + self.dice(logits, targets.float())


class MulticlassDiceCELoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets.long())

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, -1, *range(1, targets.ndim))
        one_hot = one_hot.float()

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return ce + dice_loss