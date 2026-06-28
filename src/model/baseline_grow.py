"""
Baseline v7 — sequential constructive room DISCOVERY along a graph traversal.

The architect's-eye idea (WallPlan / Raster-to-Graph style): don't partition the
whole envelope at once; DISCOVER rooms one at a time following a DFS of the access
graph. Start at the entrance, grow a room, walk to a neighbour, grow the next room
ADJACENT to it, ... reach a leaf, backtrack up the graph, continue. Each room gets
a fixed size (learned per-type area) when it is discovered; leftovers go to the
nearest room. This v1 is fully deterministic (no learning yet) to test whether the
constructive-traversal concept produces sensible plans.

    python src/model/baseline_grow.py --test <MSD>/test --train <MSD>/train --out outputs/generated_grow --n 400
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import networkx as nx
import torch
from shapely.geometry import Polygon

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

from labeling import learn_mapping, fallback_mapping, label  # noqa: E402
from baseline_rect import interior_mask, region_to_poly, DEFAULT_AREA, building_angle  # noqa: E402
from validate import validate_graph_out  # noqa: E402


def pick_start(G):
    for u, v, d in G.edges(data=True):
        if d.get("connectivity") == "entrance":
            return u
    return max(G.nodes, key=lambda n: G.degree(n)) if G.number_of_nodes() else None


def dfs_order(G, start):
    order, parent, seen, st = [], {start: None}, {start}, [start]
    while st:
        n = st.pop()
        order.append(n)
        for nb in sorted(G.neighbors(n)):
            if nb not in seen:
                seen.add(nb); parent[nb] = n; st.append(nb)
    for n in G.nodes:
        if n not in seen:
            seen.add(n); parent[n] = None; order.append(n)
    return order, parent


def grow_bfs(seed, free, target):
    """Grow a connected region from seed over `free` pixels up to ~target area."""
    from collections import deque
    H, W = free.shape
    r0, c0 = seed
    if not free[r0, c0]:
        return None
    free[r0, c0] = False
    got = [(r0, c0)]
    q = deque([(r0, c0)])
    while q and len(got) < target:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc]:
                free[nr, nc] = False
                got.append((nr, nc)); q.append((nr, nc))
    m = np.zeros_like(free)
    rr = np.fromiter((g[0] for g in got), int); cc = np.fromiter((g[1] for g in got), int)
    m[rr, cc] = True
    return m


def _boundary_seed(free, interior):
    from scipy import ndimage
    edge = free & ~ndimage.binary_erosion(interior)
    ys, xs = np.where(edge)
    if len(ys):
        return int(ys[0]), int(xs[0])
    ys, xs = np.where(free)
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def _adjacent_seed(parent_mask, free):
    from scipy import ndimage
    cand = ndimage.binary_dilation(parent_mask) & free
    ys, xs = np.where(cand)
    if len(ys):
        c = len(ys) // 2
        return int(ys[c]), int(xs[c])
    ys, xs = np.where(free)
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def predict(graph_in, struct_in, mapping, rules):
    interior, col_x, row_y = interior_mask(struct_in)
    H, W = interior.shape
    total = int(interior.sum())
    if total < 50:
        raise ValueError("interior too small")
    nodes = list(graph_in.nodes)
    rts = {n: (int(graph_in.nodes[n]["room_type"]) if graph_in.nodes[n].get("room_type") is not None
               else label(graph_in.nodes[n].get("zoning_type"), mapping)) for n in nodes}
    af = rules.get("area_frac", DEFAULT_AREA)

    def area_of(rt):
        return float(af.get(str(rt), af.get(rt, 0.05)))

    ssum = sum(area_of(rts[n]) for n in nodes) or 1.0
    # leave ~15% slack so growth never starves; the leftover fill distributes it
    target = {n: max(int(area_of(rts[n]) / ssum * total * 0.85), 30) for n in nodes}

    start = pick_start(graph_in)
    order, parent = dfs_order(graph_in, start)
    free = interior.copy()
    masks = {}
    for n in order:
        p = parent.get(n)
        seed = _adjacent_seed(masks[p], free) if (p in masks) else _boundary_seed(free, interior)
        if seed is None:
            raise ValueError("no free space to place room")
        m = grow_bfs(seed, free, target[n])
        if m is None or not m.any():
            raise ValueError(f"failed to grow node {n}")
        masks[n] = m

    # label map + fill leftover interior with nearest room
    label_map = np.zeros((H, W), dtype=int)
    nid = {n: i + 1 for i, n in enumerate(order)}
    for n in order:
        label_map[masks[n]] = nid[n]
    from scipy import ndimage
    leftover = interior & (label_map == 0)
    if leftover.any() and (label_map > 0).any():
        ind = ndimage.distance_transform_edt(label_map == 0, return_distances=False, return_indices=True)
        label_map = np.where(leftover, label_map[tuple(ind)], label_map)

    G = nx.Graph()
    G.graph.update(graph_in.graph)
    for n in nodes:
        poly = region_to_poly(label_map == nid[n], col_x, row_y)
        if poly is None:
            raise ValueError(f"empty region for node {n}")
        G.add_node(n, geometry=list(zip(*poly.exterior.coords.xy)), room_type=rts[n],
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
    for u, v, d in graph_in.edges(data=True):
        G.add_edge(u, v, connectivity=d.get("connectivity"))
    return G


def _load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _struct(test_dir, tid):
    for ext in (".npy", ".npz"):
        p = os.path.join(test_dir, "struct_in", f"{tid}{ext}")
        if os.path.exists(p):
            a = np.load(p)
            return a[a.files[0]] if hasattr(a, "files") else a
    raise FileNotFoundError(tid)


def run(args):
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(args.test, "graph_in", "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, []
    for tid in ids:
        try:
            gi = _load(os.path.join(args.test, "graph_in", f"{tid}.pickle"))
            G = predict(gi, _struct(args.test, tid), mapping, {})
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:
            failed.append((tid, str(e)))
    print(f"Wrote {written}/{len(ids)} -> {args.out}")
    if failed:
        print(f"[!] {len(failed)} failed; first: {failed[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train")
    ap.add_argument("--out", default="outputs/generated_grow")
    ap.add_argument("--n", type=int, default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
