"""
Shared helpers for the Graph-informed U-Net (strategy 5).

Coordinate convention (VERIFIED on the data, do not change without re-checking):
  struct_in[...,1] (ch1) is constant down each column  -> ch1 = f(col): world coord 'a'
  struct_in[...,2] (ch2) is constant along each row    -> ch2 = f(row): world coord 'b'
  A graph_out geometry/centroid point is (a, b): a maps to a COLUMN via ch1,
  b maps to a ROW via ch2. (Checked: 100% of room centroids land in free space.)

So: world (a,b) -> pixel (row, col) with
      col = argmin |ch1[0, :] - a|
      row = argmin |ch2[:, 0] - b|
   and pixel (row, col) -> world (a, b) = (ch1[0, col], ch2[row, 0]).
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import networkx as nx
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

N_ROOM_TYPES = 9          # room_type ids 0..8
N_CLASSES = N_ROOM_TYPES + 1   # + background (class 0); room_type r -> class r+1
MAX_ZONING = 8
CONN_KINDS = ["door", "passage", "entrance"]


def load_struct(struct_path: str, size: int):
    """Return (wallfree (size,size) float in [0,1], col_vals(size), row_vals(size))."""
    s = np.load(struct_path).astype(np.float32)
    H = s.shape[0]
    idx = np.linspace(0, H - 1, size).round().astype(int)
    sub = s[np.ix_(idx, idx)]                       # (size,size,3)
    free = (sub[..., 0] > 127).astype(np.float32)   # 1=free, 0=wall
    col_vals = sub[0, :, 1].astype(np.float32)      # world 'a' per column (ch1)
    row_vals = sub[:, 0, 2].astype(np.float32)      # world 'b' per row    (ch2)
    return free, col_vals, row_vals


def _w2p(a, b, col_vals, row_vals):
    col = int(np.argmin(np.abs(col_vals - a)))
    row = int(np.argmin(np.abs(row_vals - b)))
    return row, col


def rasterize_target(G, col_vals, row_vals, size: int) -> np.ndarray:
    """graph_out -> (size,size) int label map. class 0 = background, room_type r -> r+1.

    Larger rooms are drawn first so smaller rooms sit on top (avoid occlusion)."""
    img = Image.new("I", (size, size), 0)
    draw = ImageDraw.Draw(img)
    nodes = []
    for _, d in G.nodes(data=True):
        try:
            poly = np.asarray(d["geometry"], dtype=np.float32)
            area = Polygon(poly).area
        except Exception:
            continue
        nodes.append((area, poly, int(d["room_type"])))
    for _area, poly, rt in sorted(nodes, key=lambda t: -t[0]):
        if len(poly) < 3:
            continue
        cols = np.abs(col_vals[None, :] - poly[:, 0:1]).argmin(1)   # vectorized w->px
        rows = np.abs(row_vals[None, :] - poly[:, 1:2]).argmin(1)
        draw.polygon(list(zip(cols.tolist(), rows.tolist())), fill=rt + 1)
    return np.array(img, dtype=np.int64)


def graph_in_feature(G) -> np.ndarray:
    """12-d access-graph descriptor (same as the retrieval baseline)."""
    zhist = np.zeros(MAX_ZONING, dtype=np.float32)
    for _, att in G.nodes(data=True):
        z = att.get("zoning_type")
        if z is not None and 0 <= z < MAX_ZONING:
            zhist[z] += 1
    conn = Counter(d.get("connectivity") for _, _, d in G.edges(data=True))
    cvec = np.array([conn.get(k, 0) for k in CONN_KINDS], dtype=np.float32)
    n = np.float32(G.number_of_nodes())
    return np.concatenate([[n], zhist, cvec])


def vectorize(label: np.ndarray, col_vals: np.ndarray, row_vals: np.ndarray,
              min_px: int = 8, buffer_px: int = 5):
    """Predicted (size,size) label map -> graph_out networkx graph.

    Each connected component of a room-type class becomes a node with a polygon
    (in world coords), room_type and centroid. Edges connect components whose
    bounding regions are within `buffer_px` (the MSD raster->graph rule)."""
    from scipy import ndimage
    from skimage import measure

    G = nx.Graph()
    regions = []   # (node_id, room_type, dilated_mask, centroid_world, polygon_world)
    nid = 0
    for cls in range(1, N_CLASSES):
        mask = label == cls
        if not mask.any():
            continue
        lab, n = ndimage.label(mask)
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < min_px:
                continue
            contours = measure.find_contours(comp.astype(float), 0.5)
            if not contours:
                continue
            cont = max(contours, key=len)          # (row, col) pairs
            cont = measure.approximate_polygon(cont, tolerance=1.5)
            if len(cont) < 3:
                continue
            poly_world = [(float(col_vals[min(int(round(c)), len(col_vals) - 1)]),
                           float(row_vals[min(int(round(r)), len(row_vals) - 1)]))
                          for r, c in cont]
            rr, cc = np.where(comp)
            cw = float(col_vals[int(round(cc.mean()))])
            rw = float(row_vals[int(round(rr.mean()))])
            G.add_node(nid, geometry=poly_world, room_type=cls - 1, centroid=(cw, rw))
            regions.append((nid, ndimage.binary_dilation(comp, iterations=buffer_px)))
            nid += 1

    # adjacency: rooms whose dilated masks overlap, weighted by centroid distance
    adj = nx.Graph()
    adj.add_nodes_from(G.nodes)
    for a in range(len(regions)):
        for b in range(a + 1, len(regions)):
            if (regions[a][1] & regions[b][1]).any():
                na, nb = regions[a][0], regions[b][0]
                ca, cb = G.nodes[na]["centroid"], G.nodes[nb]["centroid"]
                w = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                adj.add_edge(na, nb, weight=w)
    # real access graphs are near-trees (~n-1 edges); take a minimum spanning
    # forest so the rendered edge density matches real plans (helps FID).
    mst = nx.minimum_spanning_tree(adj)
    for u, v in mst.edges():
        G.add_edge(u, v, connectivity="door")
    return G
