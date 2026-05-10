"""
Plain U-Net backbone. Shared by PixelSeg and Probabilistic U-Net.

Returns a feature map at input resolution. The caller adds a task-specific
head (logits, PixelCNN, latent injection, etc.) on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, depth=4):
        super().__init__()
        chs = [base_channels * (2**i) for i in range(depth + 1)]

        # Encoder
        self.down_blocks = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.down_blocks.append(conv_block(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up_convs.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.up_blocks.append(conv_block(chs[i + 1], chs[i]))

        self.out_channels = base_channels  # features returned at input resolution

    def forward(self, x):
        skips = []
        h = x
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            if i < len(self.down_blocks) - 1:
                skips.append(h)
                h = self.pool(h)

        for up_conv, up_block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            h = up_conv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = up_block(h)
        return h  # (B, base_channels, H, W)
