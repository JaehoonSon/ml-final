"""Save side-by-side prediction examples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import VOCDataset
from models.pixelseg import PixelSeg
from models.unet_seg import UNetSeg

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


def tag_title(tag):
    prefix = "pixelseg_lr"
    if tag.startswith(prefix):
        return f"PixelSeg {tag[len(prefix):]}"
    return f"PixelSeg {tag}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--det-tag", type=str, default="det")
    parser.add_argument("--pixelseg-tag", type=str, default="pixelseg")
    parser.add_argument("--pixelseg-tags", nargs="+", default=None,
                        help="One or more PixelSeg run directories to compare. Overrides --pixelseg-tag.")
    parser.add_argument("--output", type=str, default="runs/figures/comparison.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixelseg_tags = args.pixelseg_tags or [args.pixelseg_tag]

    unet = UNetSeg().to(device)
    unet.load_state_dict(torch.load(f"runs/{args.det_tag}/best.pt", map_location=device))
    unet.eval()

    pixelseg_models = []
    for tag in pixelseg_tags:
        model = PixelSeg().to(device)
        model.load_state_dict(torch.load(f"runs/{tag}/best.pt", map_location=device))
        model.eval()
        pixelseg_models.append(model)

    dataset = VOCDataset("val")
    indices = np.linspace(0, len(dataset) - 1, args.num_images).astype(int)

    titles = ["Input", "GT", "U-Net", "Softmax"] + [tag_title(tag) for tag in pixelseg_tags]
    _, axes = plt.subplots(args.num_images, len(titles),
                           figsize=(len(titles) * 2.5, args.num_images * 2.5))
    if args.num_images == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        x = image.unsqueeze(0).to(device)

        with torch.no_grad():
            unet_pred = unet.predict(x)[0]
            pixel_sample = unet.sample_pixels(x)[0]
            pixelseg_preds = [model.sample(x)[0] for model in pixelseg_models]

        img_np = unnormalize(image)
        panels = [
            img_np,
            overlay(img_np, mask),
            overlay(img_np, unet_pred),
            overlay(img_np, pixel_sample),
        ] + [overlay(img_np, pred) for pred in pixelseg_preds]

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
