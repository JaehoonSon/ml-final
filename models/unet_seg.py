"""U-Net segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class UNetSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=32):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, base_channels=base_channels, depth=4)
        self.head = nn.Conv2d(self.unet.out_channels, num_classes, 1)

    def forward(self, x):
        return self.head(self.unet(x))

    def loss(self, batch):
        x, y = batch
        return F.cross_entropy(self.forward(x), y)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    @torch.no_grad()
    def sample_pixels(self, x):
        logits = self.forward(x)
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
        return torch.multinomial(probs, 1).reshape(B, H, W)
