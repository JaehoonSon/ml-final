"""Evaluate a checkpoint on the VOC validation split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NUM_CLASSES, VOCDataset
from metrics import ConfusionMatrix
from models.pixelseg import PixelSeg
from models.unet_seg import UNetSeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["det", "softmax", "pixelseg"], required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tag", type=str, default=None,
                        help="Run directory to load. Defaults to 'det' or 'pixelseg' based on --mode.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode in ("det", "softmax"):
        model = UNetSeg().to(device)
        run_dir = Path("runs") / (args.tag or "det")
    else:
        model = PixelSeg().to(device)
        run_dir = Path("runs") / (args.tag or "pixelseg")
    ckpt = run_dir / "best.pt"
    print(f"Loading checkpoint from {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loader = DataLoader(VOCDataset("val"), batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    cm = ConfusionMatrix(NUM_CLASSES)
    n_passes = 1 if args.mode == "det" else args.num_samples

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        for _ in range(n_passes):
            if args.mode == "det":
                pred = model.predict(x)
            elif args.mode == "softmax":
                pred = model.sample_pixels(x)
            else:
                pred = model.sample(x)
            cm.update(pred, y)

    pa, miou = cm.compute()
    results = {
        "mode": args.mode,
        "pixel_accuracy": pa,
        "mean_iou": miou,
        "num_samples": n_passes,
    }
    print(json.dumps(results, indent=2))

    out_dir = Path(ckpt).parent
    (out_dir / f"results_{args.mode}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
