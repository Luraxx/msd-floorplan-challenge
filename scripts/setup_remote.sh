#!/usr/bin/env bash
# One-shot setup on a fresh GPU box (run from the repo root after `git clone`).
# Installs Python deps and downloads the MSD dataset.
#
# Prereq for the dataset step: a Kaggle API token at ~/.kaggle/kaggle.json
#   (kaggle.com -> Account -> Create New API Token), then: chmod 600 ~/.kaggle/kaggle.json
set -euo pipefail

echo "==> Installing Python dependencies"
pip install -r requirements.txt

DATA_DIR="data/modified-swiss-dwellings-v2"
if [ -d "$DATA_DIR" ]; then
  echo "==> Dataset already present at $DATA_DIR"
else
  echo "==> Downloading MSD dataset from Kaggle (needs ~/.kaggle/kaggle.json)"
  pip install -q kaggle
  mkdir -p data
  kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings -p data --unzip
fi

echo "==> Verifying"
python - <<'PY'
import glob
for split in ("train", "test"):
    n = len(glob.glob(f"data/modified-swiss-dwellings-v2/{split}/struct_in/*.npy"))
    print(f"  {split}: {n} struct_in files")
import torch
print("  torch", torch.__version__, "| cuda:", torch.cuda.is_available())
PY
echo "==> Setup done. Next: bash scripts/train_full.sh"
