"""Plot training/validation loss curves from runs/<tag>/history.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(run_dir: Path):
    with open(run_dir / "history.json") as f:
        return json.load(f)


def plot_one(history, ax):
    epochs = [e["epoch"] for e in history["epochs"]]
    train = [e["train_loss"] for e in history["epochs"]]
    val = [e["val_loss"] for e in history["epochs"]]
    ax.plot(epochs, train, label="train", linewidth=1.8)
    ax.plot(epochs, val, label="val", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()


def title_for(history):
    hp = history["hparams"]
    return (
        f"{history['model']}  "
        f"(lr={hp['lr']:g}, bs={hp['batch_size']}, epochs={hp['epochs']})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more run directories under runs/, e.g. runs/pixelseg_lr1e-4",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <run>/loss_curve.png for a single run, "
             "or runs/figures/loss_curves.png when multiple runs are given.",
    )
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    histories = [load_history(r) for r in run_dirs]

    if len(histories) == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_one(histories[0], ax)
        ax.set_title(title_for(histories[0]))
        out = Path(args.out) if args.out else run_dirs[0] / "loss_curve.png"
    else:
        n = len(histories)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
        for ax, h in zip(axes.flat, histories):
            plot_one(h, ax)
            ax.set_title(title_for(h), fontsize=10)
        for ax in axes.flat[len(histories):]:
            ax.axis("off")
        out = Path(args.out) if args.out else Path("runs/figures/loss_curves.png")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
