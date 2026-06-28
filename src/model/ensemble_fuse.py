"""
Ensemble fusion (the "probability cloud" idea): overlay every graph-conditioned
model's room recommendation per node-id, and let a per-pixel vote decide where the
boundary points actually lie. Each model votes where it is strong (LLM=topology,
Rectilinear=areas, Partition=density); disagreement concentrates on room EDGES
(the outer points) — exactly what we want the consensus to resolve.

All graph-conditioned generators preserve graph_in node-ids, so room n in every
model is the same room — we can stack them.

    python src/model/ensemble_fuse.py --test <MSD>/test --train <MSD>/train \
        --models llm-v1:outputs/models/llm-v1/generated rect:RECT part:PART \
        --out outputs/models/ensemble-v1/generated
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
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

from labeling import learn_mapping, fallback_mapping, label  # noqa: E402
from baseline_rect import interior_mask, region_to_poly, predict as rect_predict  # noqa: E402
from partition import envelope_from_struct_in, partition_envelope  # noqa: E402
from validate import validate_graph_out  # noqa: E402


def _load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _struct(test_dir, tid):
    a = np.load(os.path.join(test_dir, "struct_in", f"{tid}.npy"))
    return a


def rasterize(go, nodes, nid, col_x, row_y):
    """Fill each node's polygon onto the (H,W) interior grid -> int label image (nid+1)."""
    H, W = len(row_y), len(col_x)
    cx0, cx1 = float(col_x[0]), float(col_x[-1])
    ry0, ry1 = float(row_y[0]), float(row_y[-1])
    img = Image.new("I", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for n in nodes:
        if n not in go.nodes:
            continue
        poly = go.nodes[n].get("geometry")
        if poly is None or len(poly) < 3:
            continue
        pts = []
        for x, y in poly:
            ci = (x - cx0) / (cx1 - cx0) * (W - 1) if cx1 != cx0 else 0
            ri = (y - ry0) / (ry1 - ry0) * (H - 1) if ry1 != ry0 else 0
            pts.append((ci, ri))
        dr.polygon(pts, fill=int(nid[n]) + 1)
    return np.array(img, dtype=np.int32)


def fuse(graph_in, struct_in, model_graphs, mapping):
    interior, col_x, row_y = interior_mask(struct_in)
    nodes = list(graph_in.nodes)
    nid = {n: i for i, n in enumerate(nodes)}
    rts = {n: (int(graph_in.nodes[n]["room_type"]) if graph_in.nodes[n].get("room_type") is not None
               else label(graph_in.nodes[n].get("zoning_type"), mapping)) for n in nodes}

    layers = [rasterize(go, nodes, nid, col_x, row_y) for go in model_graphs if go is not None]
    if len(layers) < 2:
        raise ValueError("need >=2 model layers")
    stack = np.stack(layers, 0)                       # (K, H, W), 0 = unassigned
    K, H, W = stack.shape

    # per-pixel majority vote over the K layers (the probability cloud -> argmax)
    flat = stack.reshape(K, -1).T                     # (HW, K)
    nlab = len(nodes) + 1
    out = np.zeros(H * W, dtype=np.int32)
    inside = interior.reshape(-1)
    idx = np.where(inside)[0]
    votes = flat[idx]                                 # (P, K)
    # count votes per label via bincount per row is slow; vectorize with one-hot sum
    counts = np.zeros((len(idx), nlab), dtype=np.int16)
    for k in range(K):
        counts[np.arange(len(idx)), votes[:, k]] += 1
    counts[:, 0] = -1                                 # never pick "unassigned"
    out[idx] = counts.argmax(1)
    lab = out.reshape(H, W)

    # agreement = mean (max vote count) / K over interior pixels
    agree = float((counts.max(1)).mean() / K)

    # fill interior pixels that stayed unassigned -> nearest assigned
    miss = inside & (lab.reshape(-1) == 0)
    if miss.any() and (lab > 0).any():
        from scipy import ndimage
        ind = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(miss.reshape(H, W), lab[tuple(ind)], lab)

    G = nx.Graph()
    G.graph.update(graph_in.graph)
    for n in nodes:
        poly = region_to_poly(lab == (nid[n] + 1), col_x, row_y)
        if poly is None:
            raise ValueError(f"empty cell for node {n}")
        G.add_node(n, geometry=list(zip(*poly.exterior.coords.xy)), room_type=rts[n],
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
    for a, b, d in graph_in.edges(data=True):
        G.add_edge(a, b, connectivity=d.get("connectivity"))
    return G, agree


def _partition_graph(graph_in, struct_in):
    try:
        env = envelope_from_struct_in(struct_in, mode="concave")
        cells = partition_envelope(env, graph_in)
        g = nx.Graph()
        for n, poly in cells.items():
            if poly.is_empty:
                continue
            g.add_node(n, geometry=list(zip(*poly.exterior.coords.xy)))
        return g
    except Exception:
        return None


def run(args):
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    base = args.models[0].split(":", 1)[1]
    tids = [os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(os.path.join(base, "*.pickle")))]
    extra_dirs = {m.split(":", 1)[0]: m.split(":", 1)[1] for m in args.models[1:] if ":" in m and m.split(":", 1)[1] not in ("RECT", "PART")}
    os.makedirs(args.out, exist_ok=True)
    written, agrees, failed = 0, [], []
    for tid in tids:
        try:
            gi = _load(os.path.join(args.test, "graph_in", f"{tid}.pickle"))
            st = _struct(args.test, tid)
            graphs = [_load(os.path.join(base, f"{tid}.pickle"))]
            for m in args.models[1:]:
                key, src = m.split(":", 1)
                if src == "RECT":
                    try:
                        graphs.append(rect_predict(gi, st, mapping, {}))
                    except Exception:
                        graphs.append(None)
                elif src == "PART":
                    graphs.append(_partition_graph(gi, st))
                else:
                    p = os.path.join(src, f"{tid}.pickle")
                    graphs.append(_load(p) if os.path.exists(p) else None)
            G, agree = fuse(gi, st, graphs, mapping)
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
            agrees.append(agree)
        except Exception as e:
            failed.append((tid, str(e)))
    print(f"Wrote {written}/{len(tids)} -> {args.out}")
    if agrees:
        print(f"Mean model agreement (max-vote/K): {np.mean(agrees):.1%}")
    if failed:
        print(f"[!] {len(failed)} failed; first: {failed[:4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train")
    ap.add_argument("--models", nargs="+", required=True,
                    help="key:dir entries; dir can be RECT or PART to generate on the fly")
    ap.add_argument("--out", default="outputs/models/ensemble-v1/generated")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
