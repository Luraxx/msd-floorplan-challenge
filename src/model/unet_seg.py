"""
Strategy 5 — Graph-informed U-Net (segmentation) for MSD.

A conditional GENERATOR (unlike retrieval): it predicts a per-pixel room-type map
from the building structure + access graph, then vectorizes it to a `graph_out`.

  input  : struct_in -> 3 channels [free mask, norm col-coord, norm row-coord]
  condition: graph_in -> 12-d descriptor -> projected into the U-Net bottleneck
  output : (size,size) room-type logits (10 classes: background + 9 room types)
  loss   : per-pixel weighted cross-entropy (background down-weighted)
  emit   : argmax label map -> unet_common.vectorize -> graph_out pickle

Honest expectation (distributional FID): a vectorized segmentation reads as jaggier
than a real plan, so this will likely NOT beat the retrieval baseline on FID. It is
the real conditional-generation deliverable. Don't chase CPU convergence.

Usage:
    python src/model/unet_seg.py train   --train <MSD>/train --size 128 --n 500 --epochs 8
    python src/model/unet_seg.py predict --train <MSD>/train --test <MSD>/test \
        --size 128 --n 400 --out outputs/generated_unet
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unet_common as U

WEIGHTS = "outputs/unet.pt"


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ids(split_dir, limit=None):
    files = sorted(glob.glob(os.path.join(split_dir, "graph_in", "*.pickle")))
    if limit:
        files = files[:limit]
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_input(struct_path, size):
    """3-channel input: free mask + normalized col/row coordinate planes."""
    free, col, row = U.load_struct(struct_path, size)
    cc = np.clip(col / 40.0, -1, 1)
    rr = np.clip(row / 40.0, -1, 1)
    colp = np.broadcast_to(cc[None, :], (size, size))
    rowp = np.broadcast_to(rr[:, None], (size, size))
    x = np.stack([free, colp, rowp], axis=0).astype(np.float32)
    return x, col, row


# ---------------------------------------------------------------- model
def build_model():
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    class GUNet(nn.Module):
        def __init__(self, base=24, feat=12, n_classes=U.N_CLASSES):
            super().__init__()
            self.e1 = block(3, base)
            self.e2 = block(base, base * 2)
            self.e3 = block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.bott = block(base * 4, base * 8)
            self.gcond = nn.Sequential(nn.Linear(feat, base * 4), nn.ReLU(inplace=True),
                                       nn.Linear(base * 4, base * 8))
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.d3 = block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = block(base * 2, base)
            self.head = nn.Conv2d(base, n_classes, 1)

        def forward(self, x, feat):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b = self.bott(self.pool(e3))
            b = b + self.gcond(feat)[:, :, None, None]   # broadcast graph conditioning
            d3 = self.d3(torch.cat([self.up3(b), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)

    return GUNet()


# ---------------------------------------------------------------- data
def build_dataset(split_dir, ids, size):
    import torch
    X, Y, F = [], [], []
    for j, i in enumerate(ids):
        x, col, row = make_input(os.path.join(split_dir, "struct_in", f"{i}.npy"), size)
        g = _load(os.path.join(split_dir, "graph_out", f"{i}.pickle"))
        y = U.rasterize_target(g, col, row, size)
        gin = _load(os.path.join(split_dir, "graph_in", f"{i}.pickle"))
        X.append(x); Y.append(y); F.append(U.graph_in_feature(gin))
        if (j + 1) % 100 == 0:
            print(f"  prepared {j + 1}/{len(ids)}")
    return (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)),
            torch.tensor(np.stack(F), dtype=torch.float32))


# ---------------------------------------------------------------- train
def train(args):
    import torch
    import torch.nn as nn
    dev = _device()
    print(f"device={dev}")
    ids = _ids(args.train, args.n)
    print(f"Preparing {len(ids)} training samples at {args.size}px ...")
    X, Y, F = build_dataset(args.train, ids, args.size)
    # normalize graph features
    mu, sd = F.mean(0), F.std(0) + 1e-6
    F = (F - mu) / sd

    model = build_model().to(dev)
    w = torch.ones(U.N_CLASSES); w[0] = 0.3            # down-weight background
    crit = nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N = len(ids); bs = args.batch
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for k in range(0, N, bs):
            idx = perm[k:k + bs]
            xb, yb, fb = X[idx].to(dev), Y[idx].to(dev), F[idx].to(dev)
            opt.zero_grad()
            out = model(xb, fb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{args.epochs}  loss={tot / N:.4f}")

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "size": args.size}, WEIGHTS)
    print(f"Saved weights -> {WEIGHTS}")


# ---------------------------------------------------------------- predict
def predict(args):
    import torch
    dev = _device()
    ckpt = torch.load(WEIGHTS, map_location=dev)
    size = ckpt["size"]; mu, sd = ckpt["mu"].cpu(), ckpt["sd"].cpu()
    model = build_model().to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()

    ids = _ids(args.test, args.n)
    os.makedirs(args.out, exist_ok=True)
    print(f"Predicting {len(ids)} test plans ...")
    for j, i in enumerate(ids):
        x, col, row = make_input(os.path.join(args.test, "struct_in", f"{i}.npy"), size)
        gin = _load(os.path.join(args.test, "graph_in", f"{i}.pickle"))
        f = (torch.tensor(U.graph_in_feature(gin), dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            out = model(torch.tensor(x[None]).to(dev), f[None].to(dev))
            lab = out.argmax(1)[0].cpu().numpy().astype(np.int64)
        # restrict to free space so we don't paint rooms outside the building
        free = x[0] > 0.5
        lab = np.where(free, lab, 0)
        g = U.vectorize(lab, col, row)
        with open(os.path.join(args.out, f"{i}.pickle"), "wb") as fh:
            pickle.dump(g, fh)
        if (j + 1) % 50 == 0:
            print(f"  predicted {j + 1}/{len(ids)}")
    print(f"Wrote predictions -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True); t.add_argument("--size", type=int, default=128)
    t.add_argument("--n", type=int, default=500); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=8)
    p = sub.add_parser("predict")
    p.add_argument("--train", required=False); p.add_argument("--test", required=True)
    p.add_argument("--size", type=int, default=128); p.add_argument("--n", type=int, default=400)
    p.add_argument("--out", default="outputs/generated_unet")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
