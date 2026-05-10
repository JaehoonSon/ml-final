# PixelSeg-style Stochastic Segmentation on Pascal VOC (binary)

Adapts PixelSeg (Zhang et al., ACM MM 2022) to Pascal VOC 2012 reframed as
**binary** foreground/background segmentation. The question:

> Can PixelSeg-style autoregressive mask sampling produce more spatially coherent
> stochastic VOC masks than naive independent softmax sampling?

VOC has one annotation per image, so we are **not** claiming to learn multiple
plausible human annotations. We are testing whether modeling pixel-to-pixel
dependencies in the output mask gives cleaner samples on natural images.

## Three runs at a glance

| Run | Model | Inference | Purpose |
|---|---|---|---|
| 1 | Deterministic U-Net | argmax (one mask) | Standard segmentation upper bound |
| 2 | Same trained U-Net | sample softmax per pixel (independent) | The "noisy sampling" failure mode |
| 3 | PixelSeg | autoregressive PixelCNN sampling | The proposed fix |

Run 2 reuses Run 1's checkpoint — only the inference function changes.

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

## Our results (binary VOC, val set, to fill in after training)

### Main comparison

| Method | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| Run 1: Deterministic U-Net (argmax) | TBD | TBD |
| Run 2: Naive softmax sampling (16 samples) | TBD | TBD |
| Run 3: PixelSeg autoregressive (16 samples) | TBD | TBD |

### PixelSeg learning-rate ablation

| PixelSeg variant | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| lr 1e-3 | TBD | TBD |
| lr 1e-4 | TBD | TBD |
| lr 1e-5 | TBD | TBD |

## Project layout

```
ml-final/
├── data/
│   ├── prepare_voc.py        # torchvision download trigger
│   └── dataset.py            # binarized VOC, 224x224, joint augmentations
├── models/
│   ├── unet.py               # shared backbone
│   ├── deterministic.py      # Run 1 + Run 2
│   └── pixelseg.py           # Run 3
├── metrics.py                # ConfusionMatrix → PA, mIoU
├── train.py                  # --model {det, pixelseg}
├── evaluate.py               # --mode {det, softmax, pixelseg}
├── visualize.py              # basic side-by-side comparison figure
└── visualize_stochastic.py   # samples grid + uncertainty heatmaps
```

---

# Section 1 — Computer A: Deterministic (Run 1 + Run 2)

Trains the U-Net once and evaluates both argmax (Run 1) and naive softmax sampling (Run 2). Wall time: ~1 h training + ~5 min eval.

```bash
uv sync && \
uv run python data/prepare_voc.py && \
uv run python train.py --model det --epochs 100 --lr 1e-4 --batch-size 16 && \
uv run python evaluate.py --mode det && \
uv run python evaluate.py --mode softmax --num-samples 16
```

**Outputs after this run finishes**:
- `runs/det/best.pt` — checkpoint
- `runs/det/results_det.json` — Run 1 metrics
- `runs/det/results_softmax.json` — Run 2 metrics

---

# Section 2 — Computer B: PixelSeg (Run 3, three lr variants)

Trains PixelSeg three times with different learning rates, then evaluates each. Wall time: ~9–15 h training + ~30–90 min eval.

```bash
uv sync && \
uv run python data/prepare_voc.py && \
uv run python train.py --model pixelseg --epochs 100 --lr 1e-3 --batch-size 16 --tag pixelseg_lr1e-3 && \
uv run python train.py --model pixelseg --epochs 100 --lr 1e-4 --batch-size 16 --tag pixelseg_lr1e-4 && \
uv run python train.py --model pixelseg --epochs 100 --lr 1e-5 --batch-size 16 --tag pixelseg_lr1e-5 && \
uv run python evaluate.py --mode pixelseg --tag pixelseg_lr1e-3 --num-samples 16 && \
uv run python evaluate.py --mode pixelseg --tag pixelseg_lr1e-4 --num-samples 16 && \
uv run python evaluate.py --mode pixelseg --tag pixelseg_lr1e-5 --num-samples 16
```

**Outputs after this run finishes**:
- `runs/pixelseg_lr1e-3/best.pt` and `results_pixelseg.json`
- `runs/pixelseg_lr1e-4/best.pt` and `results_pixelseg.json`
- `runs/pixelseg_lr1e-5/best.pt` and `results_pixelseg.json`

---

# Section 3 — After both computers finish: combine + visualize

The visualization scripts need **both** the deterministic checkpoint and at least one PixelSeg checkpoint in the same `runs/` tree. Copy the missing folder over (e.g., from computer A → computer B), then run:

```bash
uv run python visualize.py --pixelseg-tag pixelseg_lr1e-5 && \
uv run python visualize_stochastic.py --pixelseg-tag pixelseg_lr1e-5
```

Replace `pixelseg_lr1e-5` with whichever variant scored best. **Outputs**:
- `runs/figures/comparison.png` — 4 images × [Input | GT | Det | Softmax | PixelSeg]
- `runs/figures/samples.png` — 3 images × [Input | GT | softmax×4 | PixelSeg×4]
- `runs/figures/uncertainty.png` — 3 images × [Input | GT | Softmax variance | PixelSeg variance]

### Quick rsync example

```bash
# From computer A, push the deterministic run to computer B
rsync -avz runs/det/ user@computer-B:/path/to/ml-final/runs/det/
```

## Final claim we're testing

> On Pascal VOC, PixelSeg-style autoregressive sampling does not necessarily
> beat deterministic segmentation on raw IoU, but it produces more spatially
> coherent stochastic masks than independent softmax sampling — meaning the
> structured-output idea generalizes from medical to natural images.
