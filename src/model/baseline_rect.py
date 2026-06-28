"""
Baseline v3 — rule-based RECTILINEAR partition.

Real MSD rooms are rectangular (median rectangularity 0.99, ~7 vertices) — except
corridors (~55% L-shaped) and living rooms (~51%), which are the flexible shapes
that absorb leftover space. So instead of Voronoi blobs (baseline v2), we
partition the apartment envelope into axis-aligned rectangles — ONE per graph
node — sized by learned per-room-type areas, laid out along the access graph
(spring-layout ordering keeps adjacent rooms neighbours). Each interior pixel is
assigned to the NEAREST rectangle, so concave / slanted-wall leftovers are
absorbed by the closest room (the corridor usually).

Rules mined from train (overridable via rules.json):
  * per-type area fractions (cell sizing)
  * balconies hug the exterior (exterior_types pushed to the boundary slices)

    python src/model/baseline_rect.py --test <MSD>/test --train <MSD>/train \
        --out outputs/generated_rect --n 400
"""
from __future__ import annotations

import argparse
import glob
import json
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
from partition import _largest_polygon  # noqa: E402
from validate import validate_graph_out  # noqa: E402

# per-room-type median area fractions, mined from 400 train plans
DEFAULT_AREA = {0: 0.3521, 1: 0.1862, 2: 0.0852, 3: 0.0516, 4: 0.1701, 5: 0.0077,
                6: 0.0180, 7: 0.0552, 8: 0.0729, 9: 0.005, 10: 0.005, 11: 0.005, 12: 0.005}
EXTERIOR_TYPES = {8}  # balconies hug the outside (mined: exterior frac 0.59 vs ~0.25)


# --------------------------------------------------------------------------- #
def interior_mask(struct_in: np.ndarray):
    """Enclosed building interior (concave footprint) + world coord ramps."""
    from scipy import ndimage
    s = np.asarray(struct_in, dtype=float)
    ch0 = s[..., 0]
    col_x = s[0, :, 1]
    row_y = s[:, 0, 2]
    free = ch0 > 127
    walls = ndimage.binary_dilation(~free, iterations=2)
    lab, _ = ndimage.label(~walls)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    outside = np.isin(lab, list(border)) if border else np.zeros_like(free)
    interior = ndimage.binary_dilation((~walls) & ~outside, iterations=2)
    if interior.sum() < 0.02 * interior.size:
        interior = free
    return interior, col_x, row_y


def spring_order(graph: nx.Graph, nodes: list) -> list:
    """1D ordering of nodes that keeps graph-adjacent rooms near each other."""
    if len(nodes) <= 2:
        return list(nodes)
    pos = nx.spring_layout(graph, seed=7)
    P = np.array([pos[n] for n in nodes], dtype=float)
    P = P - P.mean(0)
    try:
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        proj = P @ vt[0]
    except Exception:
        proj = P[:, 0]
    return [nodes[i] for i in np.argsort(proj)]


def building_angle(interior) -> float:
    """Dominant wall direction = angle of the LONGEST edge of the interior's
    minimum rotated rectangle. We slice along this axis so the rooms come out
    parallel to the outer walls instead of to the image axes."""
    from skimage import measure
    cs = measure.find_contours(interior.astype(float), 0.5)
    if not cs:
        return 0.0
    poly = Polygon([(float(c), float(r)) for r, c in max(cs, key=len)])  # x=col, y=row
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = _largest_polygon(poly)
    if poly.is_empty or poly.area <= 0:
        return 0.0
    xy = list(poly.minimum_rotated_rectangle.exterior.coords)
    best_len, best_ang = -1.0, 0.0
    for i in range(len(xy) - 1):
        (x0, y0), (x1, y1) = xy[i], xy[i + 1]
        L = float(np.hypot(x1 - x0, y1 - y0))
        if L > best_len:
            best_len, best_ang = L, float(np.arctan2(y1 - y0, x1 - x0))
    return best_ang


def rectilinearize_graph(lab, interior, col_x, row_y, min_frac=1 / 400):
    """Turn a (jagged) room-type label map into rectangular rooms in the building
    frame, then a graph_out. Each connected room region -> its bounding rectangle
    (rotated to the longest wall); overlaps resolved small-on-top; gaps -> nearest;
    rooms linked by an MST. Combines a learned layout with clean geometry."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from unet_common import N_CLASSES
    H, W = lab.shape
    theta = building_angle(interior)
    ys, xs = np.where(interior)
    if len(ys) < 20:
        return None
    ct, st = np.cos(theta), np.sin(theta)
    u = xs * ct + ys * st
    v = -xs * st + ys * ct
    boxes = []  # (room_type, umin, umax, vmin, vmax, area)
    for cls in range(1, N_CLASSES):
        m = lab == cls
        if not m.any():
            continue
        cc, n = ndimage.label(m)
        for k in range(1, n + 1):
            cm = cc == k
            a = int(cm.sum())
            if a < max(8, int(H * W * min_frac)):
                continue
            cys, cxs = np.where(cm)
            cu = cxs * ct + cys * st
            cv = -cxs * st + cys * ct
            boxes.append((cls - 1, cu.min(), cu.max(), cv.min(), cv.max(), a))
    if not boxes:
        return None
    boxes.sort(key=lambda b: -b[5])           # big first -> small painted on top
    assign = np.zeros(len(ys), dtype=int)
    rtof = {}
    for i, (rt, umin, umax, vmin, vmax, _a) in enumerate(boxes, start=1):
        inb = (u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)
        assign[inb] = i
        rtof[i] = rt
    un = assign == 0
    if un.any() and (~un).any():
        pts = np.c_[ys, xs]
        tree = cKDTree(pts[~un])
        assign[un] = assign[~un][tree.query(pts[un])[1]]
    rmap = np.zeros((H, W), dtype=int)
    rmap[ys, xs] = assign

    G = nx.Graph()
    nid = 1
    for i in range(1, len(boxes) + 1):
        poly = region_to_poly(rmap == i, col_x, row_y)
        if poly is None:
            continue
        G.add_node(nid, geometry=list(zip(*poly.exterior.coords.xy)), room_type=rtof[i],
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
        nid += 1
    if G.number_of_nodes() < 2:
        return G
    polys = {n: Polygon(d["geometry"]) for n, d in G.nodes(data=True)}
    ids = list(polys)
    adj = nx.Graph(); adj.add_nodes_from(ids)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa, pb = polys[ids[a]], polys[ids[b]]
            if pa.buffer(0.35).intersection(pb.buffer(0.35)).length > 0.6:
                ca, cb = G.nodes[ids[a]]["centroid"], G.nodes[ids[b]]["centroid"]
                adj.add_edge(ids[a], ids[b], weight=float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])))
    for x, y in nx.minimum_spanning_tree(adj).edges():
        G.add_edge(x, y, connectivity="door")
    return G


def region_to_poly(mask, col_x, row_y):
    from skimage import measure
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs:
        return None
    cont = measure.approximate_polygon(max(cs, key=len), tolerance=1.5)
    if len(cont) < 3:
        return None
    nc, nr = len(col_x), len(row_y)
    pts = [(float(col_x[min(int(round(c)), nc - 1)]),
            float(row_y[min(int(round(r)), nr - 1)])) for r, c in cont]
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    p = _largest_polygon(p)
    return p if (not p.is_empty and p.area > 0) else None


def predict(graph_in: nx.Graph, struct_in: np.ndarray, mapping: dict, rules: dict) -> nx.Graph:
    interior, col_x, row_y = interior_mask(struct_in)
    return layout(graph_in, interior, col_x, row_y, mapping, rules)


def layout(graph_in: nx.Graph, interior, col_x, row_y, mapping: dict, rules: dict) -> nx.Graph:
    """Rectilinear layout of a given interior mask for a given access graph.

    Used both with a struct_in envelope (dataset) and a hand-drawn envelope +
    a parametric room program (Studio). Node room_type is taken directly if set,
    else labelled from zoning_type; per-node 'area' overrides the learned size."""
    import numpy as np
    ys, xs = np.where(interior)
    if len(ys) < 10:
        raise ValueError("interior too small")
    # rotate the slicing into the building's own frame (parallel to the walls)
    theta = building_angle(interior)
    ct, st = np.cos(theta), np.sin(theta)
    u_all = xs * ct + ys * st      # along the longest outer wall
    v_all = -xs * st + ys * ct     # perpendicular to it
    nodes = list(graph_in.nodes)
    rts = {n: (int(graph_in.nodes[n]["room_type"]) if graph_in.nodes[n].get("room_type") is not None
               else label(graph_in.nodes[n].get("zoning_type"), mapping)) for n in nodes}
    area_frac = rules.get("area_frac", DEFAULT_AREA)
    ext_types = set(rules.get("exterior_types", EXTERIOR_TYPES))

    def af(rt):
        return float(area_frac.get(str(rt), area_frac.get(rt, 0.05)))

    order = spring_order(graph_in, nodes)
    # push exterior-preferring rooms (balconies) toward the ends -> outer bands
    ext = [n for n in order if rts[n] in ext_types]
    mid = [n for n in order if rts[n] not in ext_types]
    h = len(ext) // 2
    order = ext[:h] + mid + ext[h:]
    items = [(n, max(float(graph_in.nodes[n].get("area") or af(rts[n])), 1e-3)) for n in order]
    idx_of = {n: i for i, n in enumerate(order)}

    # Recursive rectilinear split BY INTERIOR-PIXEL COUNT (not bbox geometry), so
    # every room gets its area share of real interior and no cell is empty;
    # concave / slanted leftovers fall into whichever band covers them.
    labels = np.full(len(ys), -1, dtype=int)

    def rec(idx, it):
        if len(it) == 1:
            labels[idx] = idx_of[it[0][0]]
            return
        if len(idx) <= len(it):                      # too few pixels: one each
            for j, (n, _) in enumerate(it):
                if j < len(idx):
                    labels[idx[j]] = idx_of[n]
            return
        tot = sum(w for _, w in it) or 1.0
        acc, k = 0.0, 0
        for k in range(len(it) - 1):
            acc += it[k][1]
            if acc >= tot / 2:
                break
        left, right = it[: k + 1], it[k + 1:]
        fl = sum(w for _, w in left) / tot
        uu, vv = u_all[idx], v_all[idx]
        o = np.argsort(uu, kind="stable") if (uu.max() - uu.min()) >= (vv.max() - vv.min()) \
            else np.argsort(vv, kind="stable")
        cut = max(1, min(len(idx) - 1, int(round(fl * len(idx)))))
        rec(idx[o[:cut]], left)
        rec(idx[o[cut:]], right)

    rec(np.arange(len(ys)), items)

    lab = np.zeros(interior.shape, dtype=int)
    lab[ys, xs] = labels + 1

    G = nx.Graph()
    G.graph.update(graph_in.graph)
    for n in order:
        i = idx_of[n]
        poly = region_to_poly(lab == (i + 1), col_x, row_y)
        if poly is None:
            raise ValueError(f"empty cell for node {n}")
        coords = list(zip(*poly.exterior.coords.xy))
        G.add_node(n, geometry=coords, room_type=rts[n],
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
    for u, v, d in graph_in.edges(data=True):
        G.add_edge(u, v, connectivity=d.get("connectivity"))
    return G


# --------------------------------------------------------------------------- #
def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_struct(test_dir, tid):
    for ext in (".npy", ".npz"):
        p = os.path.join(test_dir, "struct_in", f"{tid}{ext}")
        if os.path.exists(p):
            arr = np.load(p)
            return arr[arr.files[0]] if hasattr(arr, "files") else arr
    raise FileNotFoundError(f"no struct_in for {tid}")


def run(args):
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    rules = {}
    if args.rules and os.path.exists(args.rules):
        rules = json.load(open(args.rules))
    print(f"mapping: {mapping}  | rules: {'loaded' if rules else 'defaults'}")

    in_dir = os.path.join(args.test, "graph_in")
    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(in_dir, "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)

    written, failed = 0, []
    for tid in ids:
        try:
            gi = _load(os.path.join(in_dir, f"{tid}.pickle"))
            st = _load_struct(args.test, tid)
            G = predict(gi, st, mapping, rules)
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:
            failed.append((tid, str(e)))
    print(f"Wrote {written}/{len(ids)} predictions to {args.out}")
    if failed:
        print(f"[!] {len(failed)} failed; first few:")
        for tid, err in failed[:8]:
            print(f"    {tid}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train")
    ap.add_argument("--out", default="outputs/generated_rect")
    ap.add_argument("--rules", default="outputs/rect_rules.json")
    ap.add_argument("--n", type=int, default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
