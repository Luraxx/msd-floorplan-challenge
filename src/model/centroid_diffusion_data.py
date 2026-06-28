"""
Per-APARTMENT dataset for the outline-only Centroid Graph Diffusion model (Weg B-2).

From the Kaggle CSV (mds_V2_5.372k.csv) we group rooms by unit_id (one apartment),
derive the apartment OUTLINE (room-union buffer trick), and build the target graph:
room centroids + types + adjacency. Everything is canonicalized to the building
frame (rotate longest axis horizontal, centre, scale longest axis = 1) so the
diffusion sees a normalized shape; the inverse transform is stored for rendering.

Cached per apartment (padded):
  out_verts [V,4]  outline vertices (x,y,sin θ,cos θ) in the normalized frame
  out_mask  [V]    1 = real vertex
  cents     [N,2]  room centroids (normalized)
  types     [N]    room_type 0..8
  valid     [N]    1 = real room (pad to N)
  adj       [N,N]  1 = rooms share a boundary  (for v2 / eval; v1 ignores it)
  tf        [4]    (theta, cx, cy, scale) inverse transform to world
  plan_id          for the train/test split

    python src/model/centroid_diffusion_data.py --csv data/mds_V2_5.372k.csv --out outputs/centroid_train.npz
    python src/model/centroid_diffusion_data.py --csv data/mds_V2_5.372k.csv --roundtrip   # Voronoi-from-real-centroids vs real
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.affinity import rotate, translate, scale as shscale

TYPE2I = {"Bedroom": 0, "Livingroom": 1, "Kitchen": 2, "Dining": 3, "Corridor": 4,
          "Stairs": 5, "Storeroom": 6, "Bathroom": 7, "Balcony": 8}
N_MAX = 16          # rooms/apartment p95=15
V_MAX = 48          # outline vertices (padded)
BRIDGE = 0.3        # m, wall-gap bridge for the outline


def _largest(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def building_theta(poly: Polygon) -> float:
    mrr = poly.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    e = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    lo = max(e, key=lambda s: (s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2)
    return math.atan2(lo[1][1] - lo[0][1], lo[1][0] - lo[0][0])


def canon(geom, theta, cen, scale):
    """World -> normalized building frame (rotate -theta about cen, centre, /scale)."""
    g = rotate(geom, -theta, origin=cen, use_radians=True)
    g = translate(g, -cen[0], -cen[1])
    return shscale(g, 1.0 / scale, 1.0 / scale, origin=(0, 0))


def vertex_tokens(outline_norm: Polygon):
    """[V,4] = (x, y, sin(interior angle), cos(interior angle)) along the exterior ring."""
    ring = list(outline_norm.exterior.coords)[:-1]
    K = len(ring)
    toks = []
    for i in range(K):
        p = np.array(ring[i]); a = np.array(ring[(i - 1) % K]); b = np.array(ring[(i + 1) % K])
        va, vb = a - p, b - p
        ang = math.atan2(va[0] * vb[1] - va[1] * vb[0], va[0] * vb[0] + va[1] * vb[1])
        toks.append([p[0], p[1], math.sin(ang), math.cos(ang)])
    return np.array(toks, dtype=np.float32)


def apartment_sample(rooms):
    """rooms: list of (Polygon, type_idx). Returns the cached arrays or None."""
    polys = [p for p, _ in rooms if p is not None and not p.is_empty and p.area > 0.5]
    if len(polys) < 2 or len(rooms) > N_MAX:
        return None
    union = _largest(unary_union([p.buffer(BRIDGE) for p in polys]).buffer(-BRIDGE))
    if union.is_empty or union.area <= 0:
        return None
    theta = building_theta(union)
    cen = (union.centroid.x, union.centroid.y)
    mrr = union.minimum_rotated_rectangle
    cc = list(mrr.exterior.coords)[:4]
    sides = [math.dist(cc[i], cc[(i + 1) % 4]) for i in range(4)]
    scale = max(sides) or 1.0

    out_norm = _largest(canon(union, theta, cen, scale))
    if not isinstance(out_norm, Polygon) or len(out_norm.exterior.coords) > V_MAX + 1:
        out_norm = out_norm.simplify(0.01)
        if len(out_norm.exterior.coords) > V_MAX + 1:
            return None
    vt = vertex_tokens(out_norm)

    cents, types = [], []
    for p, t in rooms:
        c = canon(p.centroid, theta, cen, scale)
        cents.append([c.x, c.y]); types.append(t)
    cents = np.array(cents, dtype=np.float32)

    R = len(rooms)
    A = np.zeros((R, R), np.int64)
    for i in range(R):
        for j in range(i + 1, R):
            pi, pj = rooms[i][0], rooms[j][0]
            if pi.buffer(0.1).intersection(pj.buffer(0.1)).length > 0.3:
                A[i, j] = A[j, i] = 1
    return dict(vt=vt, cents=cents, types=np.array(types, np.int64), adj=A,
                tf=np.array([theta, cen[0], cen[1], scale], np.float32))


def _pad(sample):
    vt = np.zeros((V_MAX, 4), np.float32); vm = np.zeros((V_MAX,), np.float32)
    k = min(len(sample["vt"]), V_MAX); vt[:k] = sample["vt"][:k]; vm[:k] = 1
    R = len(sample["types"])
    ce = np.zeros((N_MAX, 2), np.float32); ce[:R] = sample["cents"]
    ty = np.zeros((N_MAX,), np.int64); ty[:R] = sample["types"]
    va = np.zeros((N_MAX,), np.float32); va[:R] = 1
    ad = np.zeros((N_MAX, N_MAX), np.int64); ad[:R, :R] = sample["adj"]
    return vt, vm, ce, ty, va, ad


def iter_apartments(csv, limit=None):
    df = pd.read_csv(csv, usecols=["unit_id", "plan_id", "entity_type", "roomtype", "geom"])
    df = df[(df.entity_type == "area") & (df.roomtype != "Structure")]
    df = df[df.roomtype.isin(TYPE2I)]
    groups = df.groupby("unit_id")
    n = 0
    for uid, g in groups:
        rooms = []
        for _, r in g.iterrows():
            try:
                geom = _largest(wkt.loads(r.geom))
                rooms.append((geom, TYPE2I[r.roomtype]))
            except Exception:
                pass
        if len(rooms) >= 2:
            yield uid, int(g.plan_id.iloc[0]), rooms
            n += 1
            if limit and n >= limit:
                return


def build(args):
    VT, VM, CE, TY, VA, AD, TF, PID, UID = [], [], [], [], [], [], [], [], []
    kept = 0
    for uid, pid, rooms in iter_apartments(args.csv, args.n):
        s = apartment_sample(rooms)
        if s is None:
            continue
        vt, vm, ce, ty, va, ad = _pad(s)
        VT.append(vt); VM.append(vm); CE.append(ce); TY.append(ty); VA.append(va)
        AD.append(ad); TF.append(s["tf"]); PID.append(pid); UID.append(int(uid))
        kept += 1
        if kept % 2000 == 0:
            print(f"  {kept} apartments")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, out_verts=np.stack(VT), out_mask=np.stack(VM),
                        cents=np.stack(CE), types=np.stack(TY), valid=np.stack(VA),
                        adj=np.stack(AD), tf=np.stack(TF), plan_id=np.array(PID),
                        unit_id=np.array(UID))
    rr = np.array([v.sum() for v in VA])
    print(f"Cached {kept} apartments -> {args.out}")
    print(f"rooms/apt: median={int(np.median(rr))} max={int(rr.max())}; V_MAX={V_MAX} N_MAX={N_MAX}")


def roundtrip(args):
    """Render Voronoi-from-REAL-centroids vs the real apartment (validate the representation)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "msd_vendor"))
    sys.path.insert(0, os.path.dirname(__file__))
    import networkx as nx, torch
    from render import render_plan
    from partition import _voronoi_cells, _dejitter
    from PIL import Image
    SCR = "/tmp/claude-0/-root/aecf661b-1f68-405d-adbc-c759eb4f175f/scratchpad"
    done = 0
    for uid, pid, rooms in iter_apartments(args.csv, 400):
        s = apartment_sample(rooms)
        if s is None or len(rooms) < 5:
            continue
        # real apartment graph
        Greal = nx.Graph()
        for i, (p, t) in enumerate(rooms):
            Greal.add_node(i, geometry=list(p.exterior.coords), room_type=t,
                           centroid=torch.tensor([p.centroid.x, p.centroid.y]))
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rooms[i][0].buffer(0.1).intersection(rooms[j][0].buffer(0.1)).length > 0.3:
                    Greal.add_edge(i, j, connectivity="door")
        # voronoi from the real centroids, clipped to the outline
        union = _largest(unary_union([p.buffer(BRIDGE) for p, _ in rooms]).buffer(-BRIDGE))
        seeds = _dejitter({i: np.array([p.centroid.x, p.centroid.y]) for i, (p, _) in enumerate(rooms)})
        cells = _voronoi_cells(seeds, union)
        Gv = nx.Graph()
        for i, (p, t) in enumerate(rooms):
            c = cells.get(i)
            if c is None or c.is_empty:
                continue
            c = _largest(c)
            Gv.add_node(i, geometry=list(c.exterior.coords), room_type=t,
                        centroid=torch.tensor([c.centroid.x, c.centroid.y]))
        for u, v, d in Greal.edges(data=True):
            if u in Gv and v in Gv:
                Gv.add_edge(u, v, connectivity="door")
        Image.fromarray(render_plan(Greal)).save(f"{SCR}/apt_{uid}_real.png")
        Image.fromarray(render_plan(Gv)).save(f"{SCR}/apt_{uid}_voronoi.png")
        print(f"apt {uid} (plan {pid}): {len(rooms)} rooms")
        done += 1
        if done >= 4:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/mds_V2_5.372k.csv")
    ap.add_argument("--out", default="outputs/centroid_train.npz")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--roundtrip", action="store_true")
    a = ap.parse_args()
    roundtrip(a) if a.roundtrip else build(a)


if __name__ == "__main__":
    main()
