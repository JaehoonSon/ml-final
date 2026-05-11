"""Segmentation metrics."""

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.flatten().cpu()
        target = target.flatten().cpu()
        keep = (target >= 0) & (target < self.n)
        idx = self.n * target[keep] + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self):
        m = self.mat.float()
        diag = m.diag()
        pa = diag.sum() / m.sum().clamp(min=1)
        denom = m.sum(1) + m.sum(0) - diag
        iou = torch.where(denom > 0, diag / denom.clamp(min=1), torch.zeros_like(diag))
        valid = denom > 0
        miou = iou[valid].mean() if valid.any() else torch.tensor(0.0)
        return pa.item(), miou.item()
