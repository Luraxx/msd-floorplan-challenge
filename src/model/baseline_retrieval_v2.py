"""
Baseline v2 — structure-aware conditioned retrieval.

v1 retrieved the nearest train sample by the ACCESS GRAPH only (node count +
zoning histogram + connectivity counts). v2 adds a STRUCTURE descriptor read
from `struct_in` so the retrieved plan also matches the given building shell:

  - building footprint **aspect ratio** (orientation-agnostic: max_dim / min_dim)
  - footprint **area** (convex-hull area of the wall pixels, world coords)

Why these two and not absolute position/scale: the official renderer auto-frames
each plan (`set_aspect('equal')` + `margins`), so translating or uniformly scaling
a plan does NOT change its rendered image — only its *shape* (aspect ratio) and
relative layout do. So the honest lever is to retrieve a real plan whose footprint
shape matches the input, not to warp a mismatched one.

The footprint comes from the wall pixels of `struct_in` (ch0 == 0), mapped to world
coordinates via ch1 (world-y per row) / ch2 (world-x per col).

Usage:
    python src/model/baseline_retrieval_v2.py \
        --train <MSD>/train --test <MSD>/test --out outputs/generated_v2 --n 400
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

MAX_ZONING = 8
CONN_KINDS = ["door", "passage", "entrance"]
# weight of the structure block relative to the (standardized) graph block
STRUCT_WEIGHT = 1.5


def graph_feature(G) -> np.ndarray:
    """Access-graph descriptor: [n_nodes, zoning hist(8), conn counts(3)] -> 12-d."""
    zhist = np.zeros(MAX_ZONING, dtype=np.float32)
    for _, att in G.nodes(data=True):
        z = att.get("zoning_type")
        if z is not None and 0 <= z < MAX_ZONING:
            zhist[z] += 1
    conn = Counter(d.get("connectivity") for _, _, d in G.edges(data=True))
    cvec = np.array([conn.get(k, 0) for k in CONN_KINDS], dtype=np.float32)
    n = np.float32(G.number_of_nodes())
    return np.concatenate([[n], zhist, cvec])


def struct_feature(npy_path: str) -> np.ndarray:
    """Footprint descriptor from struct_in walls: [aspect_ratio, area] -> 2-d.

    aspect_ratio is orientation-agnostic (max/min side of the wall bbox), so the
    exact ch1/ch2 axis convention does not matter. area is the convex-hull area of
    the wall pixels in world units.
    """
    s = np.load(npy_path).astype(np.float32)
    wall = s[..., 0] < 127
    if wall.sum() < 8:
        return np.array([1.0, 0.0], dtype=np.float32)
    ys = s[..., 1][wall]
    xs = s[..., 2][wall]
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    lo, hi = min(w, h), max(w, h)
    aspect = hi / lo if lo > 1e-3 else 1.0
    try:
        pts = np.stack([xs, ys], axis=1)
        # subsample for speed; hull is scale-stable
        if len(pts) > 4000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
        area = float(ConvexHull(pts).volume)  # 2-D hull "volume" == area
    except Exception:
        area = w * h
    return np.array([aspect, area], dtype=np.float32)


def _ids(path: str, limit: int | None = None):
    files = sorted(glob.glob(os.path.join(path, "graph_in", "*.pickle")))
    if limit:
        files = files[:limit]
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _load_graph(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_matrix(split_dir: str, ids: list[str]) -> np.ndarray:
    feats = []
    for i in ids:
        g = _load_graph(os.path.join(split_dir, "graph_in", f"{i}.pickle"))
        gf = graph_feature(g)
        sf = struct_feature(os.path.join(split_dir, "struct_in", f"{i}.npy"))
        feats.append(np.concatenate([gf, sf]))
    return np.vstack(feats)


def main() -> None:
    import torch  # noqa: F401  (graph_out centroids are torch tensors → needed to (un)pickle)

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="outputs/generated_v2")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    print("Indexing train (graph + structure) ...")
    train_ids = _ids(args.train)
    Xtr = build_matrix(args.train, train_ids)

    print("Loading test queries ...")
    test_ids = _ids(args.test, args.n)
    Xte = build_matrix(args.test, test_ids)

    # standardize each column on train stats, then upweight the 2 structure cols
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    Xtr_n[:, -2:] *= STRUCT_WEIGHT
    Xte_n[:, -2:] *= STRUCT_WEIGHT

    nn = NearestNeighbors(n_neighbors=1).fit(Xtr_n)
    _, idx = nn.kneighbors(Xte_n)

    os.makedirs(args.out, exist_ok=True)
    for i, tid in enumerate(test_ids):
        match_id = train_ids[int(idx[i, 0])]
        with open(os.path.join(args.train, "graph_out", f"{match_id}.pickle"), "rb") as fh:
            pred = pickle.load(fh)
        with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
            pickle.dump(pred, fh)
    print(f"Wrote {len(test_ids)} predictions to {args.out}")


if __name__ == "__main__":
    main()
