"""Plot vote-IoU-vs-N for a PixelSeg run, with greedy + sample reference lines.

Reads:
  runs/<tag>/results_pixelseg.json           (single-sample, contributes N=16 passes)
  runs/<tag>/results_pixelseg_greedy.json    (greedy decoding)
  runs/<tag>/results_pixelseg_vote_N<N>.json (vote with N samples)

Optionally overlays the U-Net argmax baseline from runs/det/results_det.json.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def vote_points(run_dir):
    points = []
    pat = re.compile(r"results_pixelseg_vote_N(\d+)\.json$")
    for p in sorted(run_dir.glob("results_pixelseg_vote_N*.json")):
        m = pat.search(p.name)
        if not m:
            continue
        n = int(m.group(1))
        data = load(p)
        points.append((n, data["pixel_accuracy"], data["mean_iou"]))
    points.sort()
    return points


def plot_metric(ax, points, sample, greedy, baseline, metric, ylabel):
    ns = [p[0] for p in points]
    ys = [p[1 if metric == "pixel_accuracy" else 2] for p in points]
    ax.plot(ns, ys, marker="o", label="vote (vs N)", linewidth=2)

    if sample is not None:
        ax.axhline(sample[metric], linestyle="--", color="tab:orange",
                   label=f"single-sample E[IoU] = {sample[metric]:.3f}")
    if greedy is not None:
        ax.axhline(greedy[metric], linestyle="--", color="tab:green",
                   label=f"greedy = {greedy[metric]:.3f}")
    if baseline is not None:
        ax.axhline(baseline[metric], linestyle=":", color="black",
                   label=f"U-Net argmax = {baseline[metric]:.3f}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("vote sample count N")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pixelseg_lr1e-3",
                        help="PixelSeg run dir under runs/")
    parser.add_argument("--out", default="runs/figures/vote_vs_n.png")
    args = parser.parse_args()

    run_dir = Path("runs") / args.tag
    points = vote_points(run_dir)
    if not points:
        raise SystemExit(f"No vote results found in {run_dir}")

    sample = load(run_dir / "results_pixelseg.json")
    greedy = load(run_dir / "results_pixelseg_greedy.json")
    baseline = load(Path("runs/det/results_det.json"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_metric(axes[0], points, sample, greedy, baseline,
                "mean_iou", "mean IoU")
    plot_metric(axes[1], points, sample, greedy, baseline,
                "pixel_accuracy", "pixel accuracy")
    fig.suptitle(f"PixelSeg inference comparison ({args.tag})", fontsize=12)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print(f"  vote points: {[(n, round(iou, 3)) for n, _, iou in points]}")
    if sample: print(f"  single sample: IoU={sample['mean_iou']:.3f}")
    if greedy: print(f"  greedy:        IoU={greedy['mean_iou']:.3f}")
    if baseline: print(f"  U-Net argmax:  IoU={baseline['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
