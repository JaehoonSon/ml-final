"""PixelSeg model for binary VOC."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet

LOW_RES = 56


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.shape
        mask = torch.ones_like(self.weight)
        # PixelCNN mask: A excludes the current pixel, B keeps it.
        mask[:, :, kH // 2, kW // 2 + (1 if mask_type == "B" else 0):] = 0
        mask[:, :, kH // 2 + 1:, :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, num_classes, cond_channels, hidden=64, n_layers=4, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.cond = nn.Conv2d(cond_channels, hidden, 1)
        self.input = MaskedConv2d("A", num_classes, hidden, kernel_size, padding=pad)
        self.body = nn.ModuleList([
            MaskedConv2d("B", hidden, hidden, kernel_size, padding=pad) for _ in range(n_layers - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in range(n_layers - 1)])
        self.out = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, y_onehot, cond):
        c = self.cond(cond)
        h = F.relu(self.input(y_onehot) + c)
        for layer, bn in zip(self.body, self.bns):
            h = F.relu(bn(layer(h)) + c)
        return self.out(h)


class PixelSeg(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=32, pixelcnn_layers=4):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(in_channels=in_channels, base_channels=base_channels, depth=4)
        feat_ch = self.unet.out_channels
        self.pixelcnn = PixelCNN(num_classes, cond_channels=feat_ch, n_layers=pixelcnn_layers)
        self.target_head = nn.Sequential(
            nn.Conv2d(feat_ch + num_classes, feat_ch, 3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, num_classes, 1),
        )

    def _onehot(self, y):
        return F.one_hot(y, self.num_classes).permute(0, 3, 1, 2).float()

    def forward(self, x, y):
        features = self.unet(x)
        features_low = F.adaptive_avg_pool2d(features, LOW_RES)
        # Max-pool keeps small foreground regions from disappearing at 56x56.
        y_low = F.adaptive_max_pool2d(y.float().unsqueeze(1), LOW_RES).squeeze(1).long()
        low_logits = self.pixelcnn(self._onehot(y_low), features_low)

        y_low_up = F.interpolate(self._onehot(y_low), size=x.shape[-2:], mode="nearest")
        high_logits = self.target_head(torch.cat([features, y_low_up], dim=1))
        return low_logits, high_logits, y_low

    def loss(self, batch):
        x, y = batch
        low, high, y_low = self(x, y)
        return F.cross_entropy(low, y_low) + F.cross_entropy(high, y)

    @torch.no_grad()
    def _decode_low(self, features_low, greedy=False):
        B, _, H, W = features_low.shape
        y = torch.zeros(B, H, W, dtype=torch.long, device=features_low.device)
        total = H * W
        pos = 0
        while pos < total:
            r, c = pos // W, pos % W
            logits = self.pixelcnn(self._onehot(y), features_low)
            if greedy:
                step = logits[:, :, r, c].argmax(dim=1)
            else:
                probs = F.softmax(logits[:, :, r, c], dim=1)
                step = torch.multinomial(probs, 1).squeeze(1)
            y[:, r, c] = step
            pos += 1

            if (step == 0).all():
                # Most pixels are background, so skip easy background runs.
                argmax_full = logits.argmax(dim=1).view(B, -1)
                while pos < total and (argmax_full[:, pos] == 0).all():
                    nr, nc = pos // W, pos % W
                    y[:, nr, nc] = 0
                    pos += 1
        return y

    @torch.no_grad()
    def _high_from_low(self, x, features, y_low):
        y_low_up = F.interpolate(self._onehot(y_low), size=x.shape[-2:], mode="nearest")
        high_logits = self.target_head(torch.cat([features, y_low_up], dim=1))
        return high_logits.argmax(dim=1)

    @torch.no_grad()
    def sample(self, x):
        features = self.unet(x)
        features_low = F.adaptive_avg_pool2d(features, LOW_RES)
        y_low = self._decode_low(features_low, greedy=False)
        return self._high_from_low(x, features, y_low)

    @torch.no_grad()
    def predict_greedy(self, x):
        features = self.unet(x)
        features_low = F.adaptive_avg_pool2d(features, LOW_RES)
        y_low = self._decode_low(features_low, greedy=True)
        return self._high_from_low(x, features, y_low)

    @torch.no_grad()
    def predict_vote(self, x, n_samples):
        features = self.unet(x)
        features_low = F.adaptive_avg_pool2d(features, LOW_RES)
        votes = torch.zeros(x.shape[0], *x.shape[-2:], device=x.device)
        for _ in range(n_samples):
            y_low = self._decode_low(features_low, greedy=False)
            votes += self._high_from_low(x, features, y_low).float()
        return (votes / n_samples > 0.5).long()
