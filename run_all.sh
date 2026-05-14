#!/usr/bin/env bash
# Train every model, run both hyperparameter sweeps, evaluate all inference modes,
# and generate every figure. Safe to leave overnight in a tmux session.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh

set -euo pipefail
exec > >(tee -a run.log) 2>&1

echo "=== started $(date) ==="

# ---------- Setup ----------
uv sync
uv run python data/prepare_voc.py

# ---------- Train: U-Net baseline ----------
uv run python train.py --model det --epochs 100 --lr 1e-4 --batch-size 16

# ---------- Train: PixelSeg learning-rate sweep (hyperparameter #1) ----------
for LR in 1e-3 1e-4 1e-5; do
  uv run python train.py \
    --model pixelseg --epochs 100 --batch-size 16 \
    --lr "$LR" --tag "pixelseg_lr$LR"
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
# Inference modes: sample (current), greedy (joint MAP), vote (marginal mode).
# Vote sweeps N ∈ {1,2,4,8,16,32} to study how voting converges (hyperparameter #2).
for TAG in pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5; do
  uv run python evaluate.py --mode pixelseg --inference sample \
    --tag "$TAG" --num-samples 16
  uv run python evaluate.py --mode pixelseg --inference greedy --tag "$TAG"
  for N in 1 2 4 8 16 32; do
    uv run python evaluate.py --mode pixelseg --inference vote \
      --tag "$TAG" --num-samples "$N"
  done
done

# ---------- Confusion matrices: every model × headline inference modes ----------
# (We skip per-N vote matrices — those are summarized by the vote-vs-N curve.)
uv run python plot_confusion.py --mode det
uv run python plot_confusion.py --mode softmax --num-samples 16
for TAG in pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5; do
  uv run python plot_confusion.py --mode pixelseg        --tag "$TAG" --num-samples 16
  uv run python plot_confusion.py --mode pixelseg_greedy --tag "$TAG"
  uv run python plot_confusion.py --mode pixelseg_vote   --tag "$TAG" --num-samples 16
done

# ---------- Qualitative figures ----------
uv run python visualize.py \
  --pixelseg-tags pixelseg_lr1e-3 pixelseg_lr1e-4 pixelseg_lr1e-5 \
  --output runs/figures/comparison_all_pixelseg.png
uv run python visualize_stochastic.py --pixelseg-tag pixelseg_lr1e-3

echo "=== finished $(date) ==="
