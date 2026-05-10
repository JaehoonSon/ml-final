"""
Stochastic comparison figures.

Produces two PNGs in runs/figures/:
    samples.png      — multiple samples per method side by side (shows coherence)
    uncertainty.png  — per-pixel variance heatmaps for each stochastic method

    uv run python visualize_stochastic.py
    uv run python visualize_stochastic.py --num-images 4 --pixelseg-tag pixelseg_lr1e-5
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


def gather_samples(det, pixelseg, dataset, indices, n_samples, device):
    """For each image, draw n_samples from softmax sampling and from PixelSeg."""
    results = []
    for idx in indices:
        image, mask = dataset[idx]
        x = image.unsqueeze(0).to(device)
        with torch.no_grad():
            softmax_samples = torch.stack([det.sample_softmax(x)[0] for _ in range(n_samples)])
            pixelseg_samples = torch.stack([pixelseg.sample(x)[0] for _ in range(n_samples)])
        results.append({
            "image": image,
            "mask": mask,
            "softmax": softmax_samples,
            "pixelseg": pixelseg_samples,
        })
    return results


def plot_samples_grid(results, output, n_show=4):
    """Each row = one image. Columns: Input, GT, softmax × n_show, pixelseg × n_show."""
    n_rows = len(results)
    titles = ["Input", "GT"] + \
        [f"Softmax {i + 1}" for i in range(n_show)] + \
        [f"PixelSeg {i + 1}" for i in range(n_show)]
    n_cols = len(titles)

    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes[None, :]

    for row, r in enumerate(results):
        img = unnormalize(r["image"])
        panels = [img, overlay(img, r["mask"])]
        panels += [overlay(img, r["softmax"][i]) for i in range(n_show)]
        panels += [overlay(img, r["pixelseg"][i]) for i in range(n_show)]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def plot_uncertainty(results, output):
    """Per-pixel variance across samples, comparing softmax vs PixelSeg."""
    n_rows = len(results)
    titles = ["Input", "Ground Truth", "Softmax variance", "PixelSeg variance"]
    n_cols = len(titles)

    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[None, :]

    for row, r in enumerate(results):
        img = unnormalize(r["image"])
        soft_var = r["softmax"].float().var(dim=0).cpu().numpy()
        ps_var = r["pixelseg"].float().var(dim=0).cpu().numpy()
        vmax = max(soft_var.max(), ps_var.max(), 1e-3)

        panels = [(img, None), (overlay(img, r["mask"]), None),
                  (soft_var, "hot"), (ps_var, "hot")]
        for col, (panel, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--num-samples", type=int, default=8,
                   help="Samples drawn per method (used for variance computation).")
    p.add_argument("--num-show", type=int, default=4,
                   help="How many of those samples to display in samples.png.")
    p.add_argument("--det-tag", type=str, default="det")
    p.add_argument("--pixelseg-tag", type=str, default="pixelseg")
    p.add_argument("--output-dir", type=str, default="runs/figures")
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

    print(f"Drawing {args.num_samples} samples per method × {args.num_images} images")
    results = gather_samples(det, pixelseg, dataset, indices, args.num_samples, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_samples_grid(results, out_dir / "samples.png", n_show=args.num_show)
    plot_uncertainty(results, out_dir / "uncertainty.png")


if __name__ == "__main__":
    main()
