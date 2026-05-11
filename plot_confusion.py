"""Plot confusion matrices from saved checkpoints."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NUM_CLASSES, VOCDataset
from metrics import ConfusionMatrix
from models.pixelseg import PixelSeg
from models.unet_seg import UNetSeg


LABELS = ["background", "foreground"]


def load_model(mode, tag, device):
    if mode in ("det", "softmax"):
        model = UNetSeg().to(device)
        run_dir = Path("runs") / (tag or "det")
    else:
        model = PixelSeg().to(device)
        run_dir = Path("runs") / (tag or "pixelseg")

    ckpt = run_dir / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, run_dir


def predict(model, mode, x):
    if mode == "det":
        return model.predict(x)
    if mode == "softmax":
        return model.sample_pixels(x)
    return model.sample(x)


def plot_matrix(matrix, title, output):
    mat = matrix.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(NUM_CLASSES), LABELS, rotation=20, ha="right")
    ax.set_yticks(range(NUM_CLASSES), LABELS)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(mat[i, j])})",
                    ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["det", "softmax", "pixelseg"], required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, run_dir = load_model(args.mode, args.tag, device)
    loader = DataLoader(VOCDataset("val"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    cm = ConfusionMatrix(NUM_CLASSES)
    n_passes = 1 if args.mode == "det" else args.num_samples
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            for _ in range(n_passes):
                cm.update(predict(model, args.mode, x), y)

    tag = run_dir.name
    output = Path(args.output or f"runs/figures/confusion_{args.mode}_{tag}.png")
    plot_matrix(cm.mat.numpy(), f"{args.mode} ({tag})", output)


if __name__ == "__main__":
    main()
