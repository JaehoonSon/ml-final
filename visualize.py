"""
Qualitative comparison: a few val images, each shown as
    Input | Ground Truth | Deterministic | Softmax sampling | PixelSeg

    uv run python visualize.py
    uv run python visualize.py --num-images 4 --pixelseg-tag pixelseg_lr1e-5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import VOCDataset
from models.deterministic import Deterministic
from models.pixelseg import PixelSeg

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(image_tensor):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * IMG_STD + IMG_MEAN, 0, 1)


def overlay(image, mask, color=(1.0, 0.2, 0.2), alpha=0.5):
    out = image.copy()
    m = mask.cpu().numpy().astype(bool) if torch.is_tensor(mask) else mask.astype(bool)
    for c, v in enumerate(color):
        out[..., c] = np.where(m, (1 - alpha) * out[..., c] + alpha * v, out[..., c])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--det-tag", type=str, default="det")
    p.add_argument("--pixelseg-tag", type=str, default="pixelseg")
    p.add_argument("--output", type=str, default="runs/figures/comparison.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    det = Deterministic().to(device)
    det.load_state_dict(torch.load(f"runs/{args.det_tag}/best.pt", map_location=device))
    det.eval()

    pixelseg = PixelSeg().to(device)
    pixelseg.load_state_dict(torch.load(f"runs/{args.pixelseg_tag}/best.pt", map_location=device))
    pixelseg.eval()

    dataset = VOCDataset("val")
    indices = np.linspace(0, len(dataset) - 1, args.num_images).astype(int)

    titles = ["Input", "Ground Truth", "Deterministic", "Softmax sampling", "PixelSeg"]
    _, axes = plt.subplots(args.num_images, len(titles),
                           figsize=(len(titles) * 2.5, args.num_images * 2.5))
    if args.num_images == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        x = image.unsqueeze(0).to(device)

        with torch.no_grad():
            det_pred = det.predict_argmax(x)[0]
            softmax_pred = det.sample_softmax(x)[0]
            pixelseg_pred = pixelseg.sample(x)[0]

        img_np = unnormalize(image)
        panels = [
            img_np,
            overlay(img_np, mask),
            overlay(img_np, det_pred),
            overlay(img_np, softmax_pred),
            overlay(img_np, pixelseg_pred),
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=11)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
