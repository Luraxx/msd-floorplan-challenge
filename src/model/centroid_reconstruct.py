"""
Centroid graph -> room polygons (v1: weighted-Voronoi reconstruction) + the
generate / real-reference drivers for the outline-only Centroid Diffusion model.

v1 reconstruction: clip a (size-weighted) Voronoi of the generated centroids to the
apartment outline; adjacency is derived from the resulting geometry (rooms that
share a wall); a boundary corridor/stairs becomes the entrance. The separator
algorithm is the v2 upgrade.
"""
from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np
import networkx as nx
import torch
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import rotate, translate, scale as shscale

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

from partition import _voronoi_cells, _dejitter  # noqa: E402
from validate import validate_graph_out  # noqa: E402

TYPE_AREA = {0: 1.0, 1: 1.3, 2: 0.7, 3: 0.8, 4: 0.7, 5: 0.5, 6: 0.4, 7: 0.4, 8: 0.6}  # rel. room sizes


def _largest(g):
    return max(g.geoms, key=lambda x: x.area) if isinstance(g, MultiPolygon) else g


def inverse_canon(geom, tf):
    """Normalized building frame -> world (inverse of centroid_diffusion_data.canon)."""
    theta, cx, cy, scale = float(tf[0]), float(tf[1]), float(tf[2]), float(tf[3])
    g = shscale(geom, scale, scale, origin=(0, 0))
    g = translate(g, cx, cy)
    return rotate(g, theta, origin=(cx, cy), use_radians=True)


def outline_polygon(verts, vmask):
    pts = verts[vmask.astype(bool)][:, :2]
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    return _largest(p) if (p and not p.is_empty) else None


def reconstruct(cents, types, valid, outline_norm, tf):
    """Generated normalized centroids -> world graph_out via weighted Voronoi."""
    keep = [i for i in range(len(valid)) if valid[i]]
    # robustness: keep at least 2 nodes (the most confident if validity collapsed)
    if len(keep) < 2:
        keep = list(range(min(4, len(cents))))
    seeds = {}
    for i in keep:
        p = Point(float(cents[i][0]), float(cents[i][1]))
        if not outline_norm.contains(p):                 # project inside (containment)
            p = outline_norm.exterior.interpolate(outline_norm.exterior.project(p))
        seeds[i] = np.array([p.x, p.y])
    seeds = _dejitter(seeds)
    cells = _voronoi_cells(seeds, outline_norm)          # plain Voronoi tiles exactly (round-trip-proven)

    G = nx.Graph()
    polys = {}
    for i in keep:
        c = cells.get(i)
        if c is None or c.is_empty or c.area <= 0:
            continue
        cw = _largest(inverse_canon(_largest(c), tf))
        if cw.is_empty or cw.area <= 0:
            continue
        polys[i] = cw
        G.add_node(i, geometry=list(cw.exterior.coords), room_type=int(types[i]),
                   centroid=torch.tensor([cw.centroid.x, cw.centroid.y]))
    # adjacency from geometry
    ids = list(polys)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            u, v = ids[a], ids[b]
            if polys[u].buffer(0.15).intersection(polys[v].buffer(0.15)).length > 0.4:
                G.add_edge(u, v, connectivity="door")
    _mark_entrance(G, polys, types)
    # contiguous 0..K-1 ids — the MSD renderer hardcodes G.nodes[1]
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    return G


def _weighted_voronoi(seeds, env, weights):
    """Multiplicatively-weighted Voronoi via per-seed sampling on a fine grid, clipped to env."""
    # fall back to plain Voronoi if anything degenerate
    try:
        from shapely.ops import unary_union
        minx, miny, maxx, maxy = env.bounds
        res = 110
        xs = np.linspace(minx, maxx, res); ys = np.linspace(miny, maxy, res)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        ids = list(seeds)
        S = np.array([seeds[i] for i in ids])
        w = np.array([weights[i] for i in ids])
        d = np.sqrt(((pts[:, None, :] - S[None]) ** 2).sum(-1)) / w[None]   # weighted distance
        owner = np.array(ids)[d.argmin(1)].reshape(res, res)
        from shapely.geometry import box as shbox
        cw = (maxx - minx) / (res - 1); ch = (maxy - miny) / (res - 1)
        cells = {}
        for i in ids:
            mask = owner == i
            if not mask.any():
                continue
            boxes = [shbox(xs[c] - cw / 2, ys[r] - ch / 2, xs[c] + cw / 2, ys[r] + ch / 2)
                     for r, c in zip(*np.where(mask))]
            cell = unary_union(boxes).intersection(env)
            if not cell.is_empty:
                cells[i] = _largest(cell)
        if len(cells) >= max(2, int(0.6 * len(ids))):
            return cells
    except Exception:
        pass
    return _voronoi_cells(seeds, env)


def _mark_entrance(G, polys, types):
    """Boundary corridor/stairs -> entrance edge (render parity)."""
    cand = [i for i in polys if int(types[i]) in (4, 5)] or list(polys)
    if not cand or G.number_of_edges() == 0:
        return
    # pick the candidate with a neighbour; mark one incident edge entrance
    for i in cand:
        nbrs = list(G.neighbors(i))
        if nbrs:
            G[i][nbrs[0]]["connectivity"] = "entrance"
            return


# ----------------------------------------------------------------------------- drivers

def _save(G, gi_like, out, tid):
    if G.number_of_nodes() < 2:
        raise ValueError("degenerate")
    with open(os.path.join(out, f"{tid}.pickle"), "wb") as fh:
        pickle.dump(G, fh)


def run_generate(args, sample_fn, device_fn):
    dev = device_fn()
    z = np.load(args.data)
    pid = z["plan_id"]; test = np.where(pid % 10 == 0)[0]
    if args.n:
        test = test[: args.n]
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, []
    for k in test:
        try:
            verts = z["out_verts"][k]; vmask = z["out_mask"][k]; tf = z["tf"][k]
            outline = outline_polygon(verts, vmask)
            if outline is None:
                raise ValueError("no outline")
            cents, types, valid = sample_fn(dev, verts.astype(np.float32), vmask.astype(np.float32), steps=args.steps)
            G = reconstruct(cents, types, valid, outline, tf)
            if G.number_of_nodes() < 2:
                raise ValueError("degenerate")
            with open(os.path.join(args.out, f"{int(k)}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:
            failed.append((int(k), str(e)))
    print(f"Wrote {written}/{len(test)} -> {args.out}")
    if failed:
        print(f"[!] {len(failed)} failed; first: {failed[:4]}")


def run_real(args, *_):
    """Render the REAL test apartments (world polygons) as the FID reference."""
    from centroid_diffusion_data import iter_apartments, TYPE2I  # noqa
    os.makedirs(args.out, exist_ok=True)
    written = 0
    cap = args.n or 100000
    for uid, pid, rooms in iter_apartments(args.csv if hasattr(args, "csv") else "data/mds_V2_5.372k.csv"):
        if pid % 10 != 0:
            continue
        if len(rooms) < 2:
            continue
        G = nx.Graph()
        for i, (p, t) in enumerate(rooms):
            pp = _largest(p)
            G.add_node(i, geometry=list(pp.exterior.coords), room_type=int(t),
                       centroid=torch.tensor([pp.centroid.x, pp.centroid.y]))
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rooms[i][0].buffer(0.1).intersection(rooms[j][0].buffer(0.1)).length > 0.3:
                    G.add_edge(i, j, connectivity="door")
        _mark_entrance(G, {i: _largest(p) for i, (p, _) in enumerate(rooms)},
                       {i: t for i, (_, t) in enumerate(rooms)})
        with open(os.path.join(args.out, f"{int(uid)}.pickle"), "wb") as fh:
            pickle.dump(G, fh)
        written += 1
        if written >= cap:
            break
    print(f"Wrote {written} real apartments -> {args.out}")
