"""
Corner diffusion (Weg B / HouseDiffusion-style) — the learned realization of the
"outer points + denoiser" idea.

Each room is a token carrying its 4 outer-corner box (x0,y0,x1,y1). A Transformer
DENOISES all room boxes jointly, conditioned on room type and the ACCESS GRAPH via
graph-relational attention: door/passage/entrance edges add a learned per-head bias
so connected rooms attend to (and end up beside) each other. Continuous diffusion,
x0-prediction (the trick that fixed our raster diffusion).

    python src/model/corner_diffusion_model.py train    --data outputs/corner_train.npz --epochs 400
    python src/model/corner_diffusion_model.py generate --test <MSD>/test --train <MSD>/train --out outputs/models/corner-v1/generated --n 800
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

WEIGHTS = os.environ.get("CORNER_WEIGHTS", "outputs/corner_diffusion.pt")
N_TYPE = 10          # 9 room types + 1 pad
N_CONN = 4           # none/passage/door/entrance
RES_T = 1000


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(d=256, layers=6, heads=8):
    import torch
    import torch.nn as nn

    def temb(t, dim):
        half = dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * fr[None]
        return torch.cat([a.sin(), a.cos()], -1)

    class GraphAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.dh = heads, d // heads
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)

        def forward(self, x, bias, kpm):
            B, R, _ = x.shape
            qkv = self.qkv(x).reshape(B, R, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]               # [B,h,R,dh]
            att = (q @ k.transpose(-1, -2)) / self.dh ** 0.5 + bias
            att = att.masked_fill(~kpm[:, None, None, :].bool(), -1e4)
            att = att.softmax(-1)
            out = (att @ v).transpose(1, 2).reshape(B, R, d)
            return self.o(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.attn = GraphAttn()
            self.n2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x, bias, kpm):
            x = x + self.attn(self.n1(x), bias, kpm)
            x = x + self.mlp(self.n2(x))
            return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.box_in = nn.Linear(4, d)
            self.type_emb = nn.Embedding(N_TYPE, d)
            self.tmlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
            self.conn_bias = nn.Embedding(N_CONN, heads)
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))

        def forward(self, boxes, t, types, adj, mask):
            h = self.box_in(boxes) + self.type_emb(types) + self.tmlp(temb(t, d))[:, None, :]
            bias = self.conn_bias(adj).permute(0, 3, 1, 2)     # [B,R,R,h] -> [B,h,R,R]
            for b in self.blocks:
                h = b(h, bias, mask)
            return self.out(h)

    return Net()


def schedule(T, dev):
    import torch
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    abar = torch.cumprod(1 - betas, 0)
    return abar


def train(args):
    import torch
    import torch.nn.functional as F
    dev = _device(); print(f"device={dev}")
    z = np.load(args.data)
    boxes = torch.tensor(z["boxes"]); types = torch.tensor(z["types"])
    adj = torch.tensor(z["adj"]); mask = torch.tensor(z["mask"])
    types = torch.where(mask.bool(), types, torch.full_like(types, N_TYPE - 1))   # pad -> type 9
    M = len(boxes); print(f"{M} floors")

    net = build_model().to(dev)
    abar = schedule(RES_T, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(M); tot = 0.0
        net.train()
        for k in range(0, M, bs):
            idx = perm[k:k + bs]
            x0 = (boxes[idx].to(dev) * 2 - 1)               # [-1,1]
            ty = types[idx].to(dev); ad = adj[idx].to(dev); mk = mask[idx].to(dev)
            B = x0.shape[0]
            t = torch.randint(0, RES_T, (B,), device=dev)
            ab = abar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
            pred = net(xt, t, ty, ad, mk)
            w = mk[:, :, None]
            loss = ((pred - x0) ** 2 * w).sum() / (w.sum() * 4 + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * B
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={tot/M:.4f}")
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


def sample(dev, types, adj, steps=100):
    """DDIM-sample boxes for one floor. types,adj: numpy. Returns boxes [R,4] in [0,1]."""
    import torch
    net = _load_net(dev)
    R = len(types)
    abar = schedule(RES_T, dev)
    ts = list(range(0, RES_T, max(1, RES_T // steps)))[::-1]
    ty = torch.tensor(types, device=dev)[None]
    ad = torch.tensor(adj, device=dev)[None]
    mk = torch.ones(1, R, device=dev)
    with torch.no_grad():
        x = torch.randn(1, R, 4, device=dev)
        for i, ti in enumerate(ts):
            t = torch.full((1,), ti, device=dev)
            ab = abar[ti]
            x0 = net(x, t, ty, ad, mk).clamp(-1.2, 1.2)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            tp = ts[i + 1] if i + 1 < len(ts) else None
            if tp is None:
                x = x0
            else:
                abp = abar[tp]
                x = abp.sqrt() * x0 + (1 - abp).sqrt() * eps
    return ((x[0].cpu().numpy() + 1) / 2)


def frame_of_struct(struct_in):
    """Building frame of a test floor, consistent with corner_diffusion_data.floor_to_boxes."""
    from baseline_rect import interior_mask, building_angle
    interior, col_x, row_y = interior_mask(struct_in)
    theta = building_angle(interior)
    ys, xs = np.where(interior)
    wx, wy = col_x[xs], row_y[ys]
    cen = (float(wx.mean()), float(wy.mean()))
    ct, st = math.cos(-theta), math.sin(-theta)
    u = cen[0] + (wx - cen[0]) * ct - (wy - cen[1]) * st
    v = cen[1] + (wx - cen[0]) * st + (wy - cen[1]) * ct
    umin, vmin = float(u.min()), float(v.min())
    return theta, cen, (umin, vmin, float(u.max()) - umin, float(v.max()) - vmin)


def assemble_boxes(struct_in, boxes01, nodes, types, graph_in):
    """Tile the real interior with the generated boxes (paint pixels) so rooms fill
    the envelope with no gaps — the geometry fix the raw boxes lack."""
    import networkx as nx
    import torch
    from baseline_rect import interior_mask, building_angle, region_to_poly
    interior, col_x, row_y = interior_mask(struct_in)
    theta = building_angle(interior)
    ys, xs = np.where(interior)
    wx, wy = col_x[xs], row_y[ys]
    cen = (float(wx.mean()), float(wy.mean()))
    ct, st = math.cos(-theta), math.sin(-theta)
    u = cen[0] + (wx - cen[0]) * ct - (wy - cen[1]) * st
    v = cen[1] + (wx - cen[0]) * st + (wy - cen[1]) * ct
    umin, vmin = float(u.min()), float(v.min())
    du, dv = (float(u.max()) - umin) or 1.0, (float(v.max()) - vmin) or 1.0

    R = len(nodes); nid = {n: i for i, n in enumerate(nodes)}
    rects, centers = [], []
    for i in range(R):
        x0, x1 = sorted((boxes01[i][0], boxes01[i][2]))
        y0, y1 = sorted((boxes01[i][1], boxes01[i][3]))
        r = (umin + x0 * du, vmin + y0 * dv, umin + x1 * du, vmin + y1 * dv)
        rects.append(r); centers.append((0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])))

    P = len(u); best = np.full(P, -1, int); inside = np.zeros(P, bool)
    for i in sorted(range(R), key=lambda j: (rects[j][2] - rects[j][0]) * (rects[j][3] - rects[j][1])):
        x0, y0, x1, y1 = rects[i]
        hit = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1) & ~inside
        best[hit] = i; inside |= hit
    miss = ~inside
    cs = np.array(centers)
    if miss.any():
        d = (u[miss][:, None] - cs[None, :, 0]) ** 2 + (v[miss][:, None] - cs[None, :, 1]) ** 2
        best[miss] = np.argmin(d, 1)
    for i in range(R):
        if not (best == i).any():
            d = (u - cs[i, 0]) ** 2 + (v - cs[i, 1]) ** 2
            k = max(12, P // (4 * R)); best[np.argpartition(d, min(k, P - 1))[:k]] = i

    lab = np.zeros(interior.shape, int); lab[ys, xs] = best + 1
    G = nx.Graph(); G.graph.update(graph_in.graph)
    for n in nodes:
        poly = region_to_poly(lab == (nid[n] + 1), col_x, row_y)
        if poly is None:
            raise ValueError(f"empty cell node {n}")
        G.add_node(n, geometry=list(zip(*poly.exterior.coords.xy)), room_type=int(types[nid[n]]),
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
    for a, b, d in graph_in.edges(data=True):
        G.add_edge(a, b, connectivity=d.get("connectivity"))
    return G


def generate(args):
    import torch
    import networkx as nx
    from labeling import learn_mapping, fallback_mapping, label
    from corner_diffusion_data import CONN
    from validate import validate_graph_out
    dev = _device()
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(args.test, "graph_in", "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, []
    for tid in ids:
        try:
            gi = pickle.load(open(os.path.join(args.test, "graph_in", f"{tid}.pickle"), "rb"))
            st = np.load(os.path.join(args.test, "struct_in", f"{tid}.npy"))
            nodes = list(gi.nodes)
            if len(nodes) < 2 or len(nodes) > 64:
                raise ValueError(f"R={len(nodes)} out of range")
            nid = {n: i for i, n in enumerate(nodes)}
            types = np.array([(int(gi.nodes[n]["room_type"]) if gi.nodes[n].get("room_type") is not None
                               else label(gi.nodes[n].get("zoning_type"), mapping)) for n in nodes])
            A = np.zeros((len(nodes), len(nodes)), np.int64)
            for u, v, d in gi.edges(data=True):
                if u in nid and v in nid:
                    c = CONN.get(d.get("connectivity"), 0)
                    A[nid[u], nid[v]] = c; A[nid[v], nid[u]] = c
            bx = sample(dev, types, A, steps=args.steps)            # [R,4] in [0,1]
            G = assemble_boxes(st, bx, nodes, types, gi)            # tile the interior
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:
            failed.append((tid, str(e)))
        if (written + len(failed)) % 200 == 0:
            print(f"  {written + len(failed)}/{len(ids)}")
    print(f"Wrote {written}/{len(ids)} -> {args.out}")
    if failed:
        print(f"[!] {len(failed)} failed; first: {failed[:4]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/corner_train.npz")
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--batch", type=int, default=64)
    g = sub.add_parser("generate")
    g.add_argument("--test", required=True); g.add_argument("--train")
    g.add_argument("--out", default="outputs/models/corner-v1/generated")
    g.add_argument("--n", type=int, default=None); g.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else generate(a)


if __name__ == "__main__":
    main()
