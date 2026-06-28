"""
Corner-diffusion dataset (Weg B / HouseDiffusion-style).

We represent each room by its OUTER POINTS. MSD rooms are ~99% rectangular, so v1
uses the room's axis-aligned bounding box in the BUILDING frame = its 4 outer
corners (x0,y0,x1,y1). A floor becomes a set of room-boxes + types + the access
graph; the diffusion model later denoises these boxes from the graph.

Per floor we cache:
  boxes  [R,4]   normalized to [0,1] in the building frame (axis-aligned)
  types  [R]     room_type 0..8
  adj    [R,R]   0=none, 1=passage, 2=door, 3=entrance  (symmetric)
  mask   [R]     1=real room (for padding to R_MAX)

    python src/model/corner_diffusion_data.py --src <MSD>/train/graph_out --out outputs/corner_train.npz
    python src/model/corner_diffusion_data.py --roundtrip <MSD>/train  # render box-approx vs real
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate

R_MAX = 64
CONN = {None: 0, "passage": 1, "door": 2, "entrance": 3}


def _load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def building_theta(union: Polygon) -> float:
    mrr = union.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    e = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    lo = max(e, key=lambda s: (s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2)
    return math.atan2(lo[1][1] - lo[0][1], lo[1][0] - lo[0][0])


def floor_to_boxes(go):
    """Return (nodes, boxes[R,4] in [0,1] building frame, types[R], theta, (gx0,gy0,sx,sy), centroid)."""
    polys = {n: Polygon(d["geometry"]) for n, d in go.nodes(data=True)
             if d.get("geometry") and len(d["geometry"]) >= 3}
    polys = {n: p for n, p in polys.items() if p.is_valid and p.area > 0}
    if len(polys) < 2:
        return None
    union = unary_union(list(polys.values()))
    theta = building_theta(union)
    cen = union.centroid
    raw = {}
    for n, p in polys.items():
        pr = rotate(p, -theta, origin=cen, use_radians=True)
        raw[n] = pr.bounds                      # (x0,y0,x1,y1) axis-aligned in building frame
    allb = np.array(list(raw.values()))
    gx0, gy0 = allb[:, 0].min(), allb[:, 1].min()
    gx1, gy1 = allb[:, 2].max(), allb[:, 3].max()
    sx, sy = (gx1 - gx0) or 1.0, (gy1 - gy0) or 1.0
    nodes = list(polys.keys())
    boxes = np.array([[(raw[n][0] - gx0) / sx, (raw[n][1] - gy0) / sy,
                       (raw[n][2] - gx0) / sx, (raw[n][3] - gy0) / sy] for n in nodes], dtype=np.float32)
    types = np.array([int(go.nodes[n].get("room_type", 0)) for n in nodes], dtype=np.int64)
    return nodes, boxes, types, theta, (gx0, gy0, sx, sy), (cen.x, cen.y)


def boxes_to_world(box, theta, norm, cen):
    """Inverse of floor_to_boxes for one room: normalized building-frame box -> world rectangle."""
    from shapely.geometry import box as shbox
    from shapely.affinity import rotate as rot
    gx0, gy0, sx, sy = norm
    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    bx = shbox(x0 * sx + gx0, y0 * sy + gy0, x1 * sx + gx0, y1 * sy + gy0)
    return rot(bx, theta, origin=(cen[0], cen[1]), use_radians=True)


def adjacency(go, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for u, v, d in go.edges(data=True):
        if u in idx and v in idx:
            c = CONN.get(d.get("connectivity"), 0)
            A[idx[u], idx[v]] = c
            A[idx[v], idx[u]] = c
    return A


def build(args):
    files = sorted(glob.glob(os.path.join(args.src, "*.pickle")))
    if args.n:
        files = files[: args.n]
    B, T, Adj, M = [], [], [], []
    kept = 0
    for f in files:
        try:
            go = _load(f)
            r = floor_to_boxes(go)
            if r is None:
                continue
            nodes, boxes, types, *_ = r
            R = len(nodes)
            if R > R_MAX:
                continue
            A = adjacency(go, nodes)
            pb = np.zeros((R_MAX, 4), np.float32); pb[:R] = boxes
            pt = np.zeros((R_MAX,), np.int64); pt[:R] = types
            pa = np.zeros((R_MAX, R_MAX), np.int64); pa[:R, :R] = A
            pm = np.zeros((R_MAX,), np.float32); pm[:R] = 1.0
            B.append(pb); T.append(pt); Adj.append(pa); M.append(pm)
            kept += 1
        except Exception:
            continue
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, boxes=np.stack(B), types=np.stack(T),
                        adj=np.stack(Adj), mask=np.stack(M))
    print(f"Cached {kept}/{len(files)} floors (R<= {R_MAX}) -> {args.out}")
    rr = np.array([m.sum() for m in M])
    print(f"rooms/floor: min={rr.min():.0f} median={np.median(rr):.0f} max={rr.max():.0f}")


def roundtrip(args):
    """Render the box-approximation of a few real plans vs the real plan."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "msd_vendor"))
    import torch
    from render import render_plan
    from PIL import Image
    SCR = "/tmp/claude-0/-root/aecf661b-1f68-405d-adbc-c759eb4f175f/scratchpad"
    files = sorted(glob.glob(os.path.join(args.roundtrip, "graph_out", "*.pickle")))[:4]
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        go = _load(f)
        r = floor_to_boxes(go)
        if r is None:
            continue
        nodes, boxes, types, theta, (gx0, gy0, sx, sy), cen = r
        G = nx.Graph(); G.graph.update(go.graph)
        for i, n in enumerate(nodes):
            bx = boxes_to_world(boxes[i], theta, (gx0, gy0, sx, sy), cen)
            G.add_node(n, geometry=list(bx.exterior.coords), room_type=int(types[i]),
                       centroid=torch.tensor([bx.centroid.x, bx.centroid.y]))
        for u, v, d in go.edges(data=True):
            G.add_edge(u, v, connectivity=d.get("connectivity"))
        Image.fromarray(render_plan(G)).save(f"{SCR}/box_{tid}_approx.png")
        Image.fromarray(render_plan(go)).save(f"{SCR}/box_{tid}_real.png")
        print(f"rendered {tid}: {len(nodes)} rooms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--out", default="outputs/corner_train.npz")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--roundtrip")
    a = ap.parse_args()
    if a.roundtrip:
        roundtrip(a)
    else:
        build(a)


if __name__ == "__main__":
    main()
