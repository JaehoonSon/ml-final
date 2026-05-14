# PixelSeg on Binary Pascal VOC

This repo adapts PixelSeg (Zhang et al., ACM MM 2022) to Pascal VOC 2012 after
collapsing the masks into foreground/background.

The main thing we wanted to check was whether autoregressive mask sampling gives
cleaner binary VOC samples than sampling each pixel independently from a U-Net
softmax.

VOC only has one annotation per image, so this is not a multi-annotator
uncertainty experiment. The visual comparison matters as much as the raw IoU.

## Runs

| Run | Model | Inference | Purpose |
|---|---|---|---|
| 1 | U-Net | argmax | standard segmentation baseline |
| 2 | U-Net | per-pixel softmax samples | noisy stochastic baseline |
| 3 | PixelSeg | PixelCNN mask samples | structured stochastic samples |

The softmax-sampling run uses the U-Net checkpoint; only inference changes.

## Reference baselines (advML_lab2, 21-class VOC)

These come from [Lstsk/advML_lab2](https://github.com/Lstsk/advML_lab2) for
context. They evaluate on **21-class** VOC, so the numbers are not directly
comparable to our binary setup, but the metric protocol is identical.

| Model | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| TransUNet (10 epoch, vanilla CE) | 73.34% | 4.15% |
| U-Net (10 epoch, vanilla CE) | 73.82% | 4.15% |
| TransUNet (50 epoch, weighted) | 59.80% | 8.63% |
| U-Net (50 epoch, weighted) | 60.82% | 9.86% |
| U-Net (100 epoch, weighted) | 63.24% | 13.70% |
| TransUNet (300 epoch, weighted) | 57.37% | 11.77% |
| TransUNet (300 epoch, weighted - Carson) | 85.72% | 48.22% |
| U-Net (300 epoch, weighted) | 77.09% | 19.75% |
| TransUNet (100 epoch, weighted, lr 1e-3) | 52.06% | 6.56% |
| TransUNet (100 epoch, weighted, lr 1e-4) | 82.66% | 45.33% |
| **TransUNet (100 epoch, weighted, lr 1e-5)** | **91.13%** | **62.65%** |

## Results

Binary VOC validation set:

| Method | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| U-Net argmax | 83.33% | 62.27% |
| U-Net softmax samples (16) | 77.19% | 55.06% |
| PixelSeg, lr 1e-3 (16) | 81.87% | 62.14% |

PixelSeg learning-rate sweep:

| Learning rate | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| 1e-3 | 81.87% | 62.14% |
| 1e-4 | 79.74% | 57.32% |
| 1e-5 | 74.04% | 39.71% |

## Project layout

```
ml-final/
├── data/
│   ├── prepare_voc.py        # torchvision download trigger
│   └── dataset.py            # binary VOC dataset
├── models/
│   ├── unet.py
│   ├── unet_seg.py
│   └── pixelseg.py
├── metrics.py
├── train.py                  # --model {det, pixelseg}
├── evaluate.py               # --mode {det, softmax, pixelseg}, --inference {sample, greedy, vote}
├── plot_history.py           # train/val loss curves from runs/<tag>/history.json
├── plot_confusion.py         # confusion matrices per inference mode
├── visualize.py
└── visualize_stochastic.py
```

# Reproduce everything

One copy-paste block that trains every model, evaluates every inference mode,
runs the two hyperparameter sweeps (learning rate, vote sample count), and
generates all figures. Each block can be re-run individually if interrupted.

```bash
# ---------- Setup ----------
uv sync
uv run python data/prepare_voc.py

# ---------- Train: U-Net baseline ----------
uv run python train.py --model det --epochs 100 --lr 1e-4 --batch-size 16

# ---------- Train: PixelSeg learning-rate sweep (hyperparameter #1) ----------
for LR in 1e-3 1e-4 1e-5; do
  uv run python train.py \
    --model pixelseg --epochs 100 --batch-size 16 \
    --lr $LR --tag pixelseg_lr$LR
done

# ---------- Loss curves (over/underfitting analysis) ----------
uv run python plot_history.py runs/det
uv run python plot_history.py \
  runs/pixelseg_lr1e-3 runs/pixelseg_lr1e-4 runs/pixelseg_lr1e-5 \
  --out runs/figures/pixelseg_loss_curves.png

# ---------- Evaluate: U-Net (argmax + per-pixel softmax samples) ----------
uv run python evaluate.py --mode det
uv run python evaluate.py --mode softmax --num-samples 16

# ---------- Evaluate PixelSeg: every LR × every inference mode × every N ----------
for TAG in pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5; do
  uv run python evaluate.py --mode pixelseg --inference sample \
    --tag $TAG --num-samples 16
  uv run python evaluate.py --mode pixelseg --inference greedy --tag $TAG
  for N in 1 2 4 8 16 32; do
    uv run python evaluate.py --mode pixelseg --inference vote \
      --tag $TAG --num-samples $N
  done
done

# ---------- Confusion matrices ----------
uv run python plot_confusion.py --mode det
uv run python plot_confusion.py --mode softmax --num-samples 16
for TAG in pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5; do
  uv run python plot_confusion.py --mode pixelseg        --tag $TAG --num-samples 16
  uv run python plot_confusion.py --mode pixelseg_greedy --tag $TAG
  uv run python plot_confusion.py --mode pixelseg_vote   --tag $TAG --num-samples 16
done

# ---------- Qualitative figures ----------
uv run python visualize.py \
  --pixelseg-tags pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5 \
  --output runs/figures/comparison_all_pixelseg.png
uv run python visualize_stochastic.py --pixelseg-tag pixelseg_lr1e-3
```

### What you get

Per-run files:
- `runs/det/{best.pt, history.json, results_det.json, results_softmax.json}`
- `runs/pixelseg_lr{1e-3,1e-4,1e-5}/{best.pt, history.json, results_pixelseg.json}`
- `runs/pixelseg_lr1e-3/results_pixelseg_greedy.json`
- `runs/pixelseg_lr1e-3/results_pixelseg_vote_N{1,2,4,8,16,32}.json`

Figures:
- `runs/det/loss_curve.png`, `runs/figures/pixelseg_loss_curves.png`
- `runs/figures/confusion_*.png` (one per mode × tag)
- `runs/figures/comparison_all_pixelseg.png`, `samples.png`, `uncertainty.png`

### Inference modes (PixelSeg)

| `--inference` | What it does | Predictions / image |
|---|---|---|
| `sample` (default) | N iid autoregressive samples, each contributes to the confusion matrix → E[IoU per sample] | N |
| `greedy` | argmax at every PixelCNN step (joint-MAP approximation) — deterministic | 1 |
| `vote` | N iid samples → per-pixel majority vote at high-res (marginal-mode estimate) | 1 |

### Quick rsync example

```bash
# From computer A, push the deterministic run to computer B
rsync -avz runs/det/ user@computer-B:/path/to/ml-final/runs/det/
```

## Takeaway

PixelSeg does not clearly beat the U-Net argmax mask on IoU, but the best run is
close to it and much better than independent softmax sampling. The sample grids
and variance heatmaps are the main evidence for whether the stochastic masks are
actually cleaner.
