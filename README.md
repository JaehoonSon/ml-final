# PixelSeg-style Stochastic Segmentation on Pascal VOC (binary)

Adapts PixelSeg (Zhang et al., ACM MM 2022) to Pascal VOC 2012 reframed as
**binary** foreground/background segmentation. The question:

> Can PixelSeg-style autoregressive mask sampling produce more spatially coherent
> stochastic VOC masks than naive independent softmax sampling?

VOC has one annotation per image, so we are **not** claiming to learn multiple
plausible human annotations. We are testing whether modeling pixel-to-pixel
dependencies in the output mask gives cleaner samples on natural images.

## Three runs

| Run | Model | Inference | What it shows |
|---|---|---|---|
| 1 | Deterministic U-Net | argmax (one mask) | Standard segmentation upper bound |
| 2 | Same trained U-Net | sample from softmax per pixel (independent) | The "noisy sampling" failure mode PixelSeg motivates against |
| 3 | PixelSeg | autoregressive PixelCNN sampling at 56×56 + decoder upsample | The proposed fix |

Run 2 reuses the Run 1 checkpoint — only the inference function changes.

## Reference baselines (advML_lab2, 21-class VOC)

These numbers come from [Lstsk/advML_lab2](https://github.com/Lstsk/advML_lab2),
included as context for how U-Net / TransUNet behave on the harder 21-class
version of the task. They are **not directly comparable** to our binary results
(different number of classes), but show the metric format we're matching.

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

## Our results (binary VOC, val set)

To be filled in after training. Same metrics, computed from a global confusion
matrix, exactly as in advML_lab2.

| Run | Method | Pixel Accuracy | Mean IoU |
|---|---|---:|---:|
| 1 | Deterministic U-Net (argmax) | TBD | TBD |
| 2 | Naive softmax sampling (16 samples) | TBD | TBD |
| 3 | PixelSeg autoregressive (16 samples) | TBD | TBD |

## Layout

```
ml-final/
├── data/
│   ├── prepare_voc.py        # torchvision download trigger
│   └── dataset.py            # binarized VOC, 224x224
├── models/
│   ├── unet.py               # shared backbone
│   ├── deterministic.py      # Run 1 + Run 2 (argmax / softmax sampling)
│   └── pixelseg.py           # Run 3 (PixelCNN head, fast autoregressive sampling)
├── metrics.py                # ConfusionMatrix → PA, mIoU
├── train.py                  # --model {det, pixelseg}
├── evaluate.py               # --mode {det, softmax, pixelseg}
└── README.md
```

Six source files. No baseline reimplementation — context numbers come from advML_lab2.

## End-to-end commands (copy-paste)

Each block below is standalone. Run top-to-bottom for a clean overnight run.

### Step 1 — One-time setup (~5 min)

```bash
uv sync
uv run python data/prepare_voc.py
```

This installs dependencies and downloads VOC 2012 (~2 GB) to `data/voc/`.

### Step 2 — Train Run 1: Deterministic U-Net (~1 hour)

```bash
uv run python train.py \
    --model det \
    --epochs 100 \
    --lr 1e-4 \
    --batch-size 16
```

Writes `runs/det/best.pt`.

### Step 3 — Train Run 3: PixelSeg (~3–5 hours)

```bash
uv run python train.py \
    --model pixelseg \
    --epochs 100 \
    --lr 1e-4 \
    --batch-size 16
```

Writes `runs/pixelseg/best.pt`.

### Step 4 — Evaluate all three runs (~20–40 min total)

```bash
uv run python evaluate.py --mode det
uv run python evaluate.py --mode softmax --num-samples 16
uv run python evaluate.py --mode pixelseg --num-samples 16
```

Each writes `runs/<model>/results_<mode>.json` and prints PA + mIoU to stdout.

### Run everything sequentially (single command)

If you want to kick the whole pipeline off in one command and walk away:

```bash
uv sync && \
uv run python data/prepare_voc.py && \
uv run python train.py --model det --epochs 100 --lr 1e-4 --batch-size 16 && \
uv run python train.py --model pixelseg --epochs 100 --lr 1e-4 --batch-size 16 && \
uv run python evaluate.py --mode det && \
uv run python evaluate.py --mode softmax --num-samples 16 && \
uv run python evaluate.py --mode pixelseg --num-samples 16
```

### Step 5 — PixelSeg hyperparameter sweep (optional)

To showcase PixelSeg under different settings without losing earlier checkpoints, use `--tag` to give each config its own run directory:

```bash
# Two PixelSeg configs, different learning rates
uv run python train.py --model pixelseg --epochs 100 --lr 1e-4 --tag pixelseg_lr1e-4
uv run python train.py --model pixelseg --epochs 100 --lr 1e-5 --tag pixelseg_lr1e-5

# Evaluate each with the matching tag
uv run python evaluate.py --mode pixelseg --tag pixelseg_lr1e-4
uv run python evaluate.py --mode pixelseg --tag pixelseg_lr1e-5
```

Each tagged run writes to `runs/<tag>/best.pt` and `runs/<tag>/results_pixelseg.json`. Without `--tag`, the default `runs/pixelseg/` is used (backwards-compatible).

### Step 6 — Visualization

Three figures, in order of impact for your presentation:

**A. Basic comparison** — one prediction per method, side by side:
```bash
uv run python visualize.py
```
Output: `runs/figures/comparison.png` (4 images × [Input | GT | Det | Softmax | PixelSeg]).

**B. Multiple stochastic samples** — shows that softmax samples are noisy while PixelSeg samples are coherent:
```bash
uv run python visualize_stochastic.py
```
Output: `runs/figures/samples.png` (3 images × [Input | GT | 4 softmax samples | 4 PixelSeg samples]).

**C. Uncertainty heatmaps** — produced by the same script as B:

Output: `runs/figures/uncertainty.png` (3 images × [Input | GT | Softmax variance | PixelSeg variance]).

To visualize a tagged PixelSeg run instead of the default:
```bash
uv run python visualize.py --pixelseg-tag pixelseg_lr1e-5 --output runs/figures/lr1e-5_comparison.png
uv run python visualize_stochastic.py --pixelseg-tag pixelseg_lr1e-5 --output-dir runs/figures/lr1e-5
```

## Hyperparameter reference

| Setting | Value |
|---|---|
| Image size | 224×224 |
| Low-res mask (PixelCNN domain) | 56×56 |
| PixelCNN depth | 4 layers |
| PixelCNN hidden | 64 |
| Samples per image (Runs 2 and 3) | 16 |
| Optimizer | Adam |
| Default learning rate | 1e-4 |
| Default epochs | 100 |
| Default batch size | 16 |
| Augmentation (train only) | resize 1.1× → random crop 224×224, h-flip p=0.5, brightness/contrast p=0.5 |

## Expected timeline

| Step | Approx. time |
|---|---|
| VOC download | 5–10 min |
| Train deterministic (Run 1) | ~1 h |
| Train PixelSeg (Run 3) | ~3–5 h |
| Eval all three runs | ~30 min |
| **Total** | **~5–7 h, fits one night** |

## Final claim we're testing

> On Pascal VOC, PixelSeg-style autoregressive sampling does not necessarily
> beat deterministic segmentation on raw IoU, but it produces more spatially
> coherent stochastic masks than independent softmax sampling — meaning the
> structured-output idea generalizes from medical to natural images.
