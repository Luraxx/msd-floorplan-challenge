#!/usr/bin/env bash
# Full-scale U-Net training + predict + eval on the GPU box.
# Override any knob via env vars, e.g.:  SIZE=512 EPOCHS=150 bash scripts/train_full.sh
set -euo pipefail

MSD=${MSD:-data/modified-swiss-dwellings-v2}
SIZE=${SIZE:-256}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-32}
NTRAIN=${NTRAIN:-4572}     # full train split
NTEST=${NTEST:-800}        # full test split

echo "==> Training U-Net  (size=$SIZE epochs=$EPOCHS batch=$BATCH n=$NTRAIN)"
python src/model/unet_seg.py train \
  --train "$MSD/train" --size "$SIZE" --n "$NTRAIN" --epochs "$EPOCHS" --batch "$BATCH"

echo "==> Predicting test split"
python src/model/unet_seg.py predict \
  --test "$MSD/test" --size "$SIZE" --n "$NTEST" --out outputs/generated_unet

echo "==> Evaluating (FID / density / coverage)"
python src/eval/run_eval.py \
  --real "$MSD/test/graph_out" --fake outputs/generated_unet --n "$NTEST"
