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
    parser.add_argument(
        "--inference",
        choices=["sample", "greedy", "vote"],
        default="sample",
        help="Only used when --mode pixelseg. sample: N independent samples each contributing to "
             "the confusion matrix (E[IoU per sample]). greedy: 1 deterministic prediction per image. "
             "vote: N samples per image, per-pixel majority vote → 1 prediction per image.",
    )
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

    if args.mode == "det":
        passes_per_image = 1
    elif args.mode == "softmax":
        passes_per_image = args.num_samples
    elif args.inference == "sample":
        passes_per_image = args.num_samples
    else:
        passes_per_image = 1  # greedy and vote produce one prediction per image

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        for _ in range(passes_per_image):
            if args.mode == "det":
                pred = model.predict(x)
            elif args.mode == "softmax":
                pred = model.sample_pixels(x)
            elif args.inference == "sample":
                pred = model.sample(x)
            elif args.inference == "greedy":
                pred = model.predict_greedy(x)
            else:
                pred = model.predict_vote(x, args.num_samples)
            cm.update(pred, y)

    pa, miou = cm.compute()
    results = {
        "mode": args.mode,
        "inference": args.inference if args.mode == "pixelseg" else None,
        "pixel_accuracy": pa,
        "mean_iou": miou,
        "num_samples": args.num_samples if args.mode != "det" and args.inference != "greedy" else 1,
    }
    print(json.dumps(results, indent=2))

    out_dir = Path(ckpt).parent
    if args.mode == "pixelseg":
        if args.inference == "greedy":
            fname = "results_pixelseg_greedy.json"
        elif args.inference == "vote":
            fname = f"results_pixelseg_vote_N{args.num_samples}.json"
        else:
            fname = "results_pixelseg.json"
    else:
        fname = f"results_{args.mode}.json"
    (out_dir / fname).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
