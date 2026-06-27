"""
Conditional RASTER diffusion for MSD floor plans (baseline v6) — the working
diffusion model.

GSDiff-style junction-graph diffusion couldn't close plans. So instead we diffuse
the ROOM-TYPE RASTER (which always closes via vectorize) and condition on the
building envelope. This is a standard, reliable image-diffusion setup and it is
genuinely GENERATIVE: many diverse plausible plans per envelope (unlike the
deterministic U-Net).

  x0   = 2*one_hot(room_type, 10) - 1   (10 channels, in [-1,1]) at 64x64
  cond = building interior mask          (1 channel)
  DDPM U-Net denoiser; DDIM sampling; argmax -> vectorize -> graph_out.

    python src/model/diffusion_model.py train    --data outputs/refine_train.npz --epochs 200
    python src/model/diffusion_model.py generate  --test <MSD>/test --out outputs/generated_diff --n 800
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unet_seg as U          # noqa: E402
import unet_common as C       # noqa: E402

N_CLASS = C.N_CLASSES          # 10
RES = 64
WEIGHTS = os.environ.get("DIFF_WEIGHTS", "outputs/diffusion.pt")


def _down(a, res):
    idx = np.linspace(0, a.shape[-1] - 1, res).round().astype(int)
    return a[..., idx][..., idx, :] if a.ndim == 3 else a[np.ix_(idx, idx)]


def build_unet(base=64, ch=N_CLASS, cond=1, tdim=256):
    import torch
    import torch.nn as nn

    def temb(t):
        half = tdim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * fr[None] * 1000
        return torch.cat([a.sin(), a.cos()], -1)

    class Res(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, padding=1)
            self.temb = nn.Linear(tdim, co)
            self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, padding=1)
            self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

        def forward(self, x, t):
            h = self.c1(nn.functional.silu(self.n1(x)))
            h = h + self.temb(t)[:, :, None, None]
            h = self.c2(nn.functional.silu(self.n2(h)))
            return h + self.skip(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
            self.inp = nn.Conv2d(ch + cond, base, 3, padding=1)
            self.d1a = Res(base, base); self.d1b = Res(base, base)
            self.down1 = nn.Conv2d(base, base, 4, 2, 1)
            self.d2a = Res(base, base * 2); self.d2b = Res(base * 2, base * 2)
            self.down2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
            self.d3a = Res(base * 2, base * 4); self.d3b = Res(base * 4, base * 4)
            self.down3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)
            self.m1 = Res(base * 4, base * 4); self.m2 = Res(base * 4, base * 4)
            self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
            self.u3a = Res(base * 8, base * 4); self.u3b = Res(base * 4, base * 2)
            self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
            self.u2a = Res(base * 4, base * 2); self.u2b = Res(base * 2, base)
            self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
            self.u1a = Res(base * 2, base); self.u1b = Res(base, base)
            self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, ch, 3, padding=1))

        def forward(self, x, cond, t):
            te = self.tmlp(temb(t))
            h = self.inp(torch.cat([x, cond], 1))
            h1 = self.d1b(self.d1a(h, te), te)
            h2 = self.d2b(self.d2a(self.down1(h1), te), te)
            h3 = self.d3b(self.d3a(self.down2(h2), te), te)
            h = self.down3(h3)
            h = self.m2(self.m1(h, te), te)
            h = self.up3(h)
            h = self.u3b(self.u3a(torch.cat([h, h3], 1), te), te)
            h = self.up2(h)
            h = self.u2b(self.u2a(torch.cat([h, h2], 1), te), te)
            h = self.up1(h)
            h = self.u1b(self.u1a(torch.cat([h, h1], 1), te), te)
            return self.out(h)

    return Net()


def schedule(T, dev):
    import torch
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    a = 1 - betas
    return betas, a, torch.cumprod(a, 0)


def train(args):
    import torch
    import torch.nn.functional as F
    dev = U._device()
    print(f"device={dev}")
    z = np.load(args.data)
    tgt = z["target"]; free = z["free"]
    M = len(tgt)
    tgt = np.stack([_down(t, RES) for t in tgt]).astype(np.int64)
    free = np.stack([_down(f, RES) for f in free]).astype(np.float32)
    tgt = torch.tensor(tgt); free = torch.tensor(free)
    print(f"data: {M} maps at {RES}px")

    net = build_unet().to(dev)
    T = 1000
    _, _, abar = schedule(T, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(M); tot = 0.0
        net.train()
        for k in range(0, M, bs):
            idx = perm[k:k + bs]
            oh = F.one_hot(tgt[idx], N_CLASS).permute(0, 3, 1, 2).float() * 2 - 1
            x0 = oh.to(dev); cond = free[idx][:, None].to(dev)
            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=dev)
            ab = abar[t][:, None, None, None]
            noise = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            pred = net(xt, cond, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * B
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={tot/M:.4f}")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save({"net": net.state_dict()}, WEIGHTS)
    print(f"saved -> {WEIGHTS}")


import functools  # noqa: E402


@functools.lru_cache(maxsize=1)
def _load(dev):
    import torch
    net = build_unet()
    net.load_state_dict(torch.load(WEIGHTS, map_location=dev)["net"])
    net.to(dev).eval()
    return net


def sample(dev, cond, steps=100):
    """DDIM sampling conditioned on a (1,1,RES,RES) envelope -> (RES,RES) argmax labels."""
    import torch
    net = _load(dev)
    T = 1000
    _, _, abar = schedule(T, dev)
    ts = list(range(0, T, max(1, T // steps)))[::-1]
    with torch.no_grad():
        x = torch.randn(1, N_CLASS, RES, RES, device=dev)
        for i, ti in enumerate(ts):
            t = torch.full((1,), ti, device=dev)
            eps = net(x, cond, t)
            ab = abar[ti]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.5, 1.5)
            tp = ts[i + 1] if i + 1 < len(ts) else None
            if tp is None:
                x = x0
            else:
                abp = abar[tp]
                x = abp.sqrt() * x0 + (1 - abp).sqrt() * eps
        return x[0].argmax(0).cpu().numpy().astype(np.int64)


def generate(args):
    import torch
    dev = U._device()
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(args.test, "graph_in", "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)
    size = 256
    idx = np.linspace(0, size - 1, RES).round().astype(int)
    written = 0
    for j, tid in enumerate(ids):
        try:
            x, col, row = U.make_input(os.path.join(args.test, "struct_in", f"{tid}.npy"), size)
            free = (x[0] > 0.5)
            cond = torch.tensor(free[np.ix_(idx, idx)].astype(np.float32))[None, None].to(dev)
            lab = sample(dev, cond, steps=args.steps)
            free64 = free[np.ix_(idx, idx)]
            lab = np.where(free64, lab, 0)
            g = C.vectorize(lab, col[idx], row[idx])
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(g, fh)
            written += 1
        except Exception:
            continue
        if (j + 1) % 200 == 0:
            print(f"  {j + 1}/{len(ids)}")
    print(f"Wrote {written}/{len(ids)} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/refine_train.npz")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=64)
    g = sub.add_parser("generate")
    g.add_argument("--test", required=True)
    g.add_argument("--out", default="outputs/generated_diff")
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else generate(a)


if __name__ == "__main__":
    main()
