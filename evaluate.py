"""
Evaluate any of the three runs on the Pascal VOC val set, reporting Pixel
Accuracy and Mean IoU computed from a global confusion matrix (matches the
evaluation protocol in https://github.com/Lstsk/advML_lab2).

    uv run python evaluate.py --mode det                 # Run 1: argmax
    uv run python evaluate.py --mode softmax             # Run 2: naive softmax sampling (uses det checkpoint)
    uv run python evaluate.py --mode pixelseg            # Run 3: PixelSeg autoregressive samples
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NUM_CLASSES, VOCDataset
from metrics import ConfusionMatrix
from models.deterministic import Deterministic
from models.pixelseg import PixelSeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["det", "softmax", "pixelseg"], required=True)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tag", type=str, default=None,
                   help="Run directory to load. Defaults to 'det' or 'pixelseg' based on --mode.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode in ("det", "softmax"):
        model = Deterministic().to(device)
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
                pred = model.predict_argmax(x)
            elif args.mode == "softmax":
                pred = model.sample_softmax(x)
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
