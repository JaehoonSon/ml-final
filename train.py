"""Train on binary Pascal VOC."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import VOCDataset
from models.unet_seg import UNetSeg
from models.pixelseg import PixelSeg

MODELS = {"det": UNetSeg, "pixelseg": PixelSeg}


def run_epoch(model, loader, optimizer, device, is_train):
    model.train(is_train)
    total, n = 0.0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        loss = model.loss((x, y))
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom run directory name. Defaults to the model name.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(VOCDataset("train"), batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(VOCDataset("val"), batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    run_dir = Path("runs") / (args.tag or args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}")
    best = float("inf")

    history = {
        "model": args.model,
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "epochs": [],
    }
    history_path = run_dir / "history.json"

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, is_train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, device, is_train=False)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
        history["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
