"""
Baseline v9 — LLM-as-layout-solver (foundation-model / combination approach).

Every geometric generator we built scatters the given door graph: our best,
Rectilinear, honours only 41% of door edges as shared walls (real = 100%). The
problem is topological REASONING, which is exactly what a large language model is
good at. So we hand the access graph to an LLM and let it place the rooms.

This is a COMBINATION of models:
  * the LLM (Claude, a foundation model) does the hard part — read the
    corridor-centred access graph and lay out non-overlapping rectangles that
    honour every adjacency;
  * our geometric pipeline does the rest — rotate into the building frame, clip
    the LLM's rectangles to the real (rotated, possibly L-shaped) interior mask
    by painting interior pixels, vectorise to polygons, validate, render, score.

Pipeline (two stages, because the "model" is an LLM driven by the Workflow tool):

  1. `specs`  : build a compact spec per floor (building-frame bbox + rooms with
                types + required adjacencies) -> specs.json
  2. <Workflow>: one agent per floor turns its spec into room rectangles
                -> layouts.json   (see scripts/llm_layout_workflow notes)
  3. `build`  : paint each interior pixel with the LLM rectangle that contains it
                (in the building frame) -> graph_out pickles

    python src/model/llm_layout.py specs --test <MSD>/test --train <MSD>/train --n 80 --out outputs/llm/specs.json
    # ... run the workflow to produce outputs/llm/layouts.json ...
    python src/model/llm_layout.py build --test <MSD>/test --train <MSD>/train --layouts outputs/llm/layouts.json --out outputs/models/llm-v1/generated
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

from labeling import learn_mapping, fallback_mapping, label  # noqa: E402
from baseline_rect import interior_mask, building_angle, region_to_poly, DEFAULT_AREA  # noqa: E402
from validate import validate_graph_out  # noqa: E402

NAMES = {0: "Bedroom", 1: "Livingroom", 2: "Kitchen", 3: "Dining", 4: "Corridor",
         5: "Stairs", 6: "Storeroom", 7: "Bathroom", 8: "Balcony"}


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


def _frame(struct_in):
    """Interior pixels in the building (axis-aligned) frame. Returns dicts of arrays."""
    interior, col_x, row_y = interior_mask(struct_in)
    ys, xs = np.where(interior)
    if len(ys) < 10:
        raise ValueError("interior too small")
    theta = building_angle(interior)
    ct, st = np.cos(theta), np.sin(theta)
    u = xs * ct + ys * st
    v = -xs * st + ys * ct
    return dict(interior=interior, col_x=col_x, row_y=row_y, ys=ys, xs=xs, u=u, v=v)


def _room_types(graph_in, mapping):
    return {n: (int(graph_in.nodes[n]["room_type"]) if graph_in.nodes[n].get("room_type") is not None
                else label(graph_in.nodes[n].get("zoning_type"), mapping)) for n in graph_in.nodes}


def build_spec(tid, graph_in, struct_in, mapping, area_frac=None):
    f = _frame(struct_in)
    rts = _room_types(graph_in, mapping)
    umin, umax = float(f["u"].min()), float(f["u"].max())
    vmin, vmax = float(f["v"].min()), float(f["v"].max())
    # learned per-type area as a fraction of the floor (the combination signal)
    af = area_frac or DEFAULT_AREA
    raw = {n: float(af.get(str(rts[n]), af.get(rts[n], 0.05))) for n in graph_in.nodes}
    tot = sum(raw.values()) or 1.0
    rooms = [{"id": int(n), "type": NAMES.get(rts[n], "Bedroom"),
              "target_area_frac": round(raw[n] / tot, 4)} for n in graph_in.nodes]
    adj = [{"a": int(u), "b": int(v), "via": d.get("connectivity")}
           for u, v, d in graph_in.edges(data=True)]
    comps = [sorted(int(x) for x in c) for c in nx.connected_components(graph_in)]
    return {"id": tid, "bbox": [round(umin, 2), round(vmin, 2), round(umax, 2), round(vmax, 2)],
            "rooms": rooms, "adjacencies": adj, "apartments": comps}


def cmd_specs(args):
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in sorted(glob.glob(os.path.join(args.test, "graph_in", "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    specs = []
    for tid in ids:
        try:
            specs.append(build_spec(tid, _load(os.path.join(args.test, "graph_in", f"{tid}.pickle")),
                                    _struct(args.test, tid), mapping))
        except Exception as e:
            print(f"[skip] {tid}: {e}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(specs, open(args.out, "w"))
    print(f"Wrote {len(specs)} specs -> {args.out}")


def assemble(graph_in, struct_in, rects, mapping):
    """Paint interior pixels with the LLM rectangle (building frame) containing them."""
    f = _frame(struct_in)
    rts = _room_types(graph_in, mapping)
    nodes = list(graph_in.nodes)
    nid = {n: i for i, n in enumerate(nodes)}
    by_id = {int(r["id"]): r for r in rects}

    u, v = f["u"], f["v"]
    P = len(u)
    best = np.full(P, -1, dtype=int)            # node index per pixel
    inside = np.zeros(P, dtype=bool)
    centers, cleaned = {}, {}
    for n in nodes:
        r = by_id.get(int(n))
        if r is None:
            continue
        x0, y0, x1, y1 = float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        centers[n] = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        cleaned[n] = (x0, y0, x1, y1)

    # smaller rectangles claim their pixels first, so big rooms can't swallow small ones
    for n in sorted(cleaned, key=lambda m: (cleaned[m][2] - cleaned[m][0]) * (cleaned[m][3] - cleaned[m][1])):
        x0, y0, x1, y1 = cleaned[n]
        hit = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1) & ~inside
        best[hit] = nid[n]
        inside |= hit

    cs = np.array([centers[n] for n in nodes if n in centers])
    cidx = np.array([nid[n] for n in nodes if n in centers])
    # pixels not covered by any rect -> nearest rectangle centre
    miss = ~inside
    if miss.any() and len(cs):
        du = u[miss][:, None] - cs[None, :, 0]
        dv = v[miss][:, None] - cs[None, :, 1]
        best[miss] = cidx[np.argmin(du * du + dv * dv, axis=1)]

    # guarantee every node owns a cell: an empty node steals the pixels nearest its centre
    for n in nodes:
        if n not in centers or (best == nid[n]).any():
            continue
        cx, cy = centers[n]
        d = (u - cx) ** 2 + (v - cy) ** 2
        k = max(12, P // (4 * len(nodes)))
        best[np.argpartition(d, min(k, P - 1))[:k]] = nid[n]

    lab = np.zeros(f["interior"].shape, dtype=int)
    lab[f["ys"], f["xs"]] = best + 1

    G = nx.Graph()
    G.graph.update(graph_in.graph)
    for n in nodes:
        poly = region_to_poly(lab == (nid[n] + 1), f["col_x"], f["row_y"])
        if poly is None:
            raise ValueError(f"empty cell for node {n}")
        G.add_node(n, geometry=list(zip(*poly.exterior.coords.xy)), room_type=rts[n],
                   centroid=torch.tensor([poly.centroid.x, poly.centroid.y]))
    for a, b, d in graph_in.edges(data=True):
        G.add_edge(a, b, connectivity=d.get("connectivity"))
    return G


def cmd_build(args):
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    layouts = json.load(open(args.layouts))
    if isinstance(layouts, list):                      # [{id/tid, rooms}]
        layouts = {str(x.get("tid", x.get("id"))): x["rooms"] for x in layouts}
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, []
    for tid, rects in layouts.items():
        try:
            gi = _load(os.path.join(args.test, "graph_in", f"{tid}.pickle"))
            G = assemble(gi, _struct(args.test, tid), rects, mapping)
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:
            failed.append((tid, str(e)))
    print(f"Wrote {written}/{len(layouts)} -> {args.out}")
    if failed:
        print(f"[!] {len(failed)} failed; first: {failed[:4]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specs"); s.add_argument("--test", required=True); s.add_argument("--train")
    s.add_argument("--n", type=int, default=None); s.add_argument("--out", default="outputs/llm/specs.json")
    b = sub.add_parser("build"); b.add_argument("--test", required=True); b.add_argument("--train")
    b.add_argument("--layouts", required=True); b.add_argument("--out", default="outputs/models/llm-v1/generated")
    a = ap.parse_args()
    cmd_specs(a) if a.cmd == "specs" else cmd_build(a)


if __name__ == "__main__":
    main()
