"""
GSDiff-style wall-junction generator for MSD (baseline v4).

Two stages, in the structural-graph representation (wallgraph.py):
  1. NODE DIFFUSION — a Transformer DDPM denoises a padded set of wall junctions,
     each a 13-vector [x, y, room_multihot(9), balcony, valid]. Trained with the
     usual noise-MSE plus an ALIGNMENT loss (grid-snap on x/y) so junctions land
     on a shared grid => axis-aligned walls (the GSDiff idea, simplified).
  2. EDGE PREDICTION — an MLP scores each junction pair; trained with BCE on the
     real wall segments. At generation it reconnects the sampled junctions.

Generate: reverse-diffuse junctions -> keep valid>0.5 -> predict wall segments ->
wallgraph.reconstruct -> graph_out.

    python src/model/gsdiff_model.py train    --data outputs/gsdiff_train.npz --epochs 300
    python src/model/gsdiff_model.py generate  --test <MSD>/test --out outputs/generated_gsdiff --n 800
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wallgraph import N_ROOM, reconstruct  # noqa: E402

FEAT = 2 + N_ROOM + 1          # x,y, rooms(9), balcony = 12 (data channels)
DCH = FEAT + 1                 # + valid  = 13 diffused channels
WEIGHTS = os.environ.get("GSDIFF_WEIGHTS", "outputs/gsdiff.pt")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_models(d_model=256, depth=6, heads=8):
    import torch
    import torch.nn as nn

    class TimeEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t):  # t: (B,) in [0,1]
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            a = t[:, None] * freqs[None] * 1000.0
            emb = torch.cat([a.sin(), a.cos()], dim=-1)
            return self.mlp(emb)

    class Denoiser(nn.Module):
        """Transformer that predicts the noise on a padded junction set."""
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(DCH, d_model)
            self.temb = TimeEmb(d_model)
            layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4,
                                               batch_first=True, activation="gelu", dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, depth)
            self.out = nn.Linear(d_model, DCH)

        def forward(self, x, t):           # x: (B,N,DCH), t: (B,)
            h = self.inp(x) + self.temb(t)[:, None, :]
            h = self.enc(h)
            return self.out(h)

    class EdgeNet(nn.Module):
        """Scores each junction pair for a wall segment."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(FEAT, 128)
            self.mlp = nn.Sequential(
                nn.Linear(128 * 2 + 3, 256), nn.SiLU(),
                nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

        def forward(self, feat, ij):       # feat:(B,N,FEAT), ij:(B,P,2) long
            h = self.proj(feat)
            B, P, _ = ij.shape
            bi = torch.arange(B, device=feat.device)[:, None]
            fi = h[bi, ij[..., 0]]
            fj = h[bi, ij[..., 1]]
            pi = feat[bi, ij[..., 0], :2]
            pj = feat[bi, ij[..., 1], :2]
            d = pj - pi
            geo = torch.cat([d, (d.norm(dim=-1, keepdim=True))], dim=-1)
            return self.mlp(torch.cat([fi + fj, (fi - fj).abs(), geo], dim=-1)).squeeze(-1)

    return Denoiser(), EdgeNet()


# --------------------------------------------------------------------------- #
# diffusion schedule
# --------------------------------------------------------------------------- #
def schedule(T, device):
    import torch
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    a = 1.0 - betas
    abar = torch.cumprod(a, 0)
    return betas, a, abar


def grid_align_loss(x0_xy, valid):
    """Encourage x/y to land on a shared grid -> axis-aligned walls."""
    import torch
    G = 24.0
    snapped = torch.round(x0_xy * G) / G
    err = (x0_xy - snapped).abs().sum(-1)
    return (err * valid).sum() / valid.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(args):
    import torch
    import torch.nn.functional as F
    dev = _device()
    print(f"device={dev}")
    z = np.load(args.data)
    feat = torch.tensor(z["feat"], dtype=torch.float32)        # (M,N,12)
    valid = torch.tensor(z["valid"], dtype=torch.float32)      # (M,N)
    edges = torch.tensor(z["edges"].astype(np.int64))          # (M,E,2)
    ecount = torch.tensor(z["ecount"].astype(np.int64))        # (M,)
    M, N, _ = feat.shape
    print(f"data: {M} plans, N={N}")

    x_data = torch.cat([feat, valid[..., None]], dim=-1)       # (M,N,13)
    denoiser, edgenet = build_models()
    denoiser.to(dev); edgenet.to(dev)
    T = 1000
    betas, a, abar = schedule(T, dev)
    opt = torch.optim.AdamW(list(denoiser.parameters()) + list(edgenet.parameters()), lr=2e-4)

    bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(M)
        tot_d = tot_e = 0.0
        denoiser.train(); edgenet.train()
        for k in range(0, M, bs):
            idx = perm[k:k + bs]
            x0 = x_data[idx].to(dev)               # (B,N,13)
            vmask = valid[idx].to(dev)             # (B,N)
            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=dev)
            ab = abar[t][:, None, None]
            noise = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = denoiser(xt, t.float() / T)
            loss_mse = F.mse_loss(pred, noise)
            # reconstruct x0_hat for alignment
            x0_hat = (xt - (1 - ab).sqrt() * pred) / ab.sqrt().clamp(min=1e-4)
            loss_al = grid_align_loss(x0_hat[..., :2], vmask)

            # edge head (teacher forcing on real nodes): sample pos + neg pairs
            ej = edges[idx].to(dev)                # (B,E,2)
            ec = ecount[idx].to(dev)
            P = 256
            B2 = B
            pos = torch.zeros(B2, P, 2, dtype=torch.long, device=dev)
            for bb in range(B2):
                e = ej[bb][: max(int(ec[bb]), 1)]
                sel = e[torch.randint(0, e.shape[0], (P,), device=dev)]
                pos[bb] = sel
            ni = torch.randint(0, N, (B2, P, 2), device=dev)
            ij = torch.cat([pos, ni], dim=1)        # (B,2P,2)
            lbl = torch.cat([torch.ones(B2, P, device=dev),
                             torch.zeros(B2, P, device=dev)], dim=1)
            logit = edgenet(x0[..., :FEAT], ij)
            # ignore neg pairs that touch padding or are self-loops handled loosely
            loss_e = F.binary_cross_entropy_with_logits(logit, lbl)

            loss = loss_mse + 0.5 * loss_al + loss_e
            opt.zero_grad(); loss.backward(); opt.step()
            tot_d += (loss_mse + 0.5 * loss_al).item() * B
            tot_e += loss_e.item() * B
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  diff={tot_d/M:.4f}  edge={tot_e/M:.4f}")

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    import torch as T_
    T_.save({"denoiser": denoiser.state_dict(), "edgenet": edgenet.state_dict()}, WEIGHTS)
    print(f"saved -> {WEIGHTS}")


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
@__import__("functools").lru_cache(maxsize=1)
def _load_gen(dev):
    import torch
    ckpt = torch.load(WEIGHTS, map_location=dev)
    den, edge = build_models()
    den.load_state_dict(ckpt["denoiser"]); edge.load_state_dict(ckpt["edgenet"])
    den.to(dev).eval(); edge.to(dev).eval()
    return den, edge


@__import__("torch").no_grad()
def sample_plan(dev, N=256, T=1000, edge_thresh=0.5):
    import torch
    den, edge = _load_gen(dev)
    betas, a, abar = schedule(T, dev)
    x = torch.randn(1, N, DCH, device=dev)
    for ti in range(T - 1, -1, -1):
        t = torch.full((1,), ti, device=dev)
        eps = den(x, t.float() / T)
        abar_t = abar[ti]
        x0 = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
        x0 = x0.clamp(-2, 2)
        if ti > 0:                                   # proper DDPM posterior step
            abar_p = abar[ti - 1]
            mean = (abar_p.sqrt() * betas[ti] / (1 - abar_t)) * x0 + \
                   (a[ti].sqrt() * (1 - abar_p) / (1 - abar_t)) * x
            var = betas[ti] * (1 - abar_p) / (1 - abar_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0
    feat = x[0, :, :FEAT]
    valid = (x[0, :, FEAT] > 0.5)
    keep = torch.where(valid)[0]
    if len(keep) < 4:
        return None
    feat = feat[keep]
    pos = feat[:, :2].cpu().numpy()
    rmh = (feat[:, 2:2 + N_ROOM] > 0.5).float().cpu().numpy()
    K = len(keep)
    # edge prediction over all pairs
    ii, jj = torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing="ij")
    mask = ii < jj
    ij = torch.stack([ii[mask], jj[mask]], dim=-1)[None]
    logit = edge(feat[None], ij)[0]
    prob = torch.sigmoid(logit)
    sel = prob > edge_thresh
    pairs = ij[0][sel].cpu().numpy()
    edges = [(int(a), int(b)) for a, b in pairs]
    return pos, rmh, edges


def generate(args):
    import torch, glob, pickle
    dev = _device()
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(args.test, "graph_in", "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, 0
    for tid in ids:
        try:
            s = sample_plan(dev)
            if s is None:
                failed += 1; continue
            pos, rmh, edges = s
            G = reconstruct(pos, edges, rmh)
            if G.number_of_nodes() < 2:
                failed += 1; continue
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception:
            failed += 1
    print(f"Wrote {written}/{len(ids)} (failed {failed}) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/gsdiff_train.npz")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=32)
    g = sub.add_parser("generate")
    g.add_argument("--test", required=True)
    g.add_argument("--out", default="outputs/generated_gsdiff")
    g.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else generate(a)


if __name__ == "__main__":
    main()
