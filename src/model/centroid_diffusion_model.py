"""
Centroid Graph Diffusion — Stage 1 (outline-only).

Given ONLY the apartment outline, a Transformer denoises a SET of room centroids
(continuous x0-diffusion) and reads off each room's TYPE and a VALIDITY flag (how
many rooms). No access graph is given — the nodes self-attend and CROSS-ATTEND to
the outline-vertex encoder. v1 reconstructs polygons with weighted Voronoi; the
separator algorithm is the v2 upgrade.

    python src/model/centroid_diffusion_model.py train    --data outputs/centroid_train.npz --epochs 600
    python src/model/centroid_diffusion_model.py generate --data outputs/centroid_train.npz --out outputs/models/centroid-v1/generated
    python src/model/centroid_diffusion_model.py real      --data outputs/centroid_train.npz --out outputs/models/centroid-v1/real   # FID reference
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))
sys.path.insert(0, os.path.join(_HERE, "..", "msd_vendor"))

WEIGHTS = os.environ.get("CENTROID_WEIGHTS", "outputs/centroid_diffusion.pt")
N_TYPE = 9
RES_T = 1000


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(d=256, enc_layers=4, dec_layers=6, heads=8):
    import torch
    import torch.nn as nn

    def temb(t, dim):
        half = dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * fr[None]
        return torch.cat([a.sin(), a.cos()], -1)

    class MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.dh = heads, d // heads
            self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
            self.o = nn.Linear(d, d)

        def forward(self, x, ctx, kpm):
            B, Q, _ = x.shape; K = ctx.shape[1]
            q = self.q(x).reshape(B, Q, self.h, self.dh).transpose(1, 2)
            k = self.k(ctx).reshape(B, K, self.h, self.dh).transpose(1, 2)
            v = self.v(ctx).reshape(B, K, self.h, self.dh).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / self.dh ** 0.5
            if kpm is not None:
                att = att.masked_fill(~kpm[:, None, None, :].bool(), -1e4)
            att = att.softmax(-1)
            return self.o((att @ v).transpose(1, 2).reshape(B, Q, d))

    class EncBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.sa = MHA()
            self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x, m):
            x = x + self.sa(self.n1(x), self.n1(x), m)
            return x + self.mlp(self.n2(x))

    class DecBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.sa = MHA()
            self.n2 = nn.LayerNorm(d); self.nc = nn.LayerNorm(d); self.ca = MHA()
            self.n3 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x, ctx, ctx_m):
            x = x + self.sa(self.n1(x), self.n1(x), None)
            x = x + self.ca(self.n2(x), self.nc(ctx), ctx_m)
            return x + self.mlp(self.n3(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.vert_in = nn.Linear(4, d)
            self.vpos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
            self.enc = nn.ModuleList([EncBlock() for _ in range(enc_layers)])
            self.xy_in = nn.Linear(2, d)
            self.tmlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
            self.dec = nn.ModuleList([DecBlock() for _ in range(dec_layers)])
            self.x0_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
            self.type_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_TYPE))
            self.valid_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.count_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

        def encode(self, verts, vmask):
            V = verts.shape[1]
            h = self.vert_in(verts) + self.vpos[:, :V]
            for b in self.enc:
                h = b(h, vmask)
            return h

        def count(self, out_emb, out_mask):
            """Predict the room count from the outline encoding (pooled over vertices)."""
            m = out_mask[:, :, None]
            pooled = (out_emb * m).sum(1) / (m.sum(1) + 1e-6)
            return self.count_head(pooled).squeeze(-1)

        def forward(self, noisy_xy, t, out_emb, out_mask):
            h = self.xy_in(noisy_xy) + self.tmlp(temb(t, h_dim := out_emb.shape[-1]))[:, None, :]
            for b in self.dec:
                h = b(h, out_emb, out_mask)
            return self.x0_head(h), self.type_head(h), self.valid_head(h).squeeze(-1)

    return Net()


def schedule(T, dev):
    import torch
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    return torch.cumprod(1 - betas, 0)


def train(args):
    import torch
    import torch.nn.functional as F
    dev = _device(); print(f"device={dev}")
    z = np.load(args.data)
    pid = z["plan_id"]
    tr = pid % 10 != 0                                    # hold out plan_id%10==0
    V = torch.tensor(z["out_verts"][tr]); VM = torch.tensor(z["out_mask"][tr])
    C = torch.tensor(z["cents"][tr]); TY = torch.tensor(z["types"][tr]); VA = torch.tensor(z["valid"][tr])
    M = len(C); print(f"{M} train apartments")

    net = build_model().to(dev)
    abar = schedule(RES_T, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(M); tot = 0.0
        net.train()
        for k in range(0, M, bs):
            idx = perm[k:k + bs]
            verts = V[idx].to(dev); vmask = VM[idx].to(dev)
            x0 = C[idx].to(dev); ty = TY[idx].to(dev); va = VA[idx].to(dev)
            B = x0.shape[0]
            t = torch.randint(0, RES_T, (B,), device=dev)
            ab = abar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
            out_emb = net.encode(verts, vmask)
            px0, plog, pval = net(xt, t, out_emb, vmask)
            w = va[:, :, None]
            l_pos = ((px0 - x0) ** 2 * w).sum() / (w.sum() * 2 + 1e-6)
            l_type = (F.cross_entropy(plog.reshape(-1, N_TYPE), ty.reshape(-1), reduction="none")
                      .reshape(B, -1) * va).sum() / (va.sum() + 1e-6)
            l_val = F.binary_cross_entropy_with_logits(pval, va)
            l_count = F.mse_loss(net.count(out_emb, vmask), va.sum(1))      # global room count from outline
            loss = l_pos + 0.5 * l_type + 0.5 * l_val + 0.1 * l_count
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * B
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={tot/M:.4f}  (pos={l_pos.item():.3f} type={l_type.item():.3f} val={l_val.item():.3f} count={l_count.item():.2f})")
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save({"net": net.state_dict()}, WEIGHTS)
    print(f"saved -> {WEIGHTS}")


import functools  # noqa: E402


@functools.lru_cache(maxsize=1)
def _load_net(dev):
    import torch
    net = build_model()
    net.load_state_dict(torch.load(WEIGHTS, map_location=dev)["net"])
    return net.to(dev).eval()


def sample(dev, verts, vmask, steps=100):
    """Predict the room count K from the outline, then denoise exactly K centroids.
    Returns (centroids[K,2], types[K], valid[K]=all True) in the normalized frame."""
    import torch
    net = _load_net(dev)
    abar = schedule(RES_T, dev)
    ts = list(range(0, RES_T, max(1, RES_T // steps)))[::-1]
    v = torch.tensor(verts, device=dev)[None]; vm = torch.tensor(vmask, device=dev)[None]
    out_emb = net.encode(v, vm)
    K = max(2, min(16, int(round(float(net.count(out_emb, vm).item())))))
    with torch.no_grad():
        x = torch.randn(1, K, 2, device=dev)
        plog = None
        for i, ti in enumerate(ts):
            t = torch.full((1,), ti, device=dev)
            ab = abar[ti]
            px0, plog, _ = net(x, t, out_emb, vm)
            px0 = px0.clamp(-0.7, 0.7)
            eps = (x - ab.sqrt() * px0) / (1 - ab).sqrt()
            tp = ts[i + 1] if i + 1 < len(ts) else None
            x = px0 if tp is None else abar[tp].sqrt() * px0 + (1 - abar[tp]).sqrt() * eps
    cents = x[0].cpu().numpy()
    types = plog[0].argmax(-1).cpu().numpy()
    return cents, types, np.ones(K, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/centroid_train.npz")
    t.add_argument("--epochs", type=int, default=600); t.add_argument("--batch", type=int, default=128)
    for name in ("generate", "real"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="outputs/centroid_train.npz")
        p.add_argument("--out", default="outputs/models/centroid-v1/generated")
        p.add_argument("--n", type=int, default=None); p.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        from centroid_reconstruct import run_generate, run_real
        (run_real if a.cmd == "real" else run_generate)(a, sample, _device)


if __name__ == "__main__":
    main()
