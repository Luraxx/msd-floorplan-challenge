"""
GSDiff-style STRUCTURAL GRAPH representation for MSD (baseline v4 foundation).

GSDiff (Hu et al. 2025) models a floor plan not as rooms but as a graph of
WALL JUNCTIONS (nodes) and WALL SEGMENTS (edges). Each node carries:
  c = (x, y)        position
  r = multi-hot[9]  which MSD room types surround this junction
  b = {0,1}         balcony-boundary flag (balcony edges are treated as walls,
                    their junctions get b=1 — per the project's balcony rule)

MSD ships room polygons, not wall graphs, so we EXTRACT the structural graph by
snapping nearby room-polygon corners into shared junctions and turning room
boundary edges into shared wall segments. We RECONSTRUCT rooms from a structural
graph with shapely.polygonize (the "minimal polygonal loops" of the paper),
labelling each loop by the majority room semantics of its junctions.

This module is the data layer; the generative model trains in this representation.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

N_ROOM = 9  # MSD room types 0..8 (Bedroom..Balcony)
BALCONY = 8


def _rings(go):
    rooms = []
    for _, d in go.nodes(data=True):
        g = d.get("geometry")
        if g is None:
            continue
        P = np.asarray(g, dtype=float)
        if len(P) < 3:
            continue
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        rooms.append((P, int(d.get("room_type", 0))))
    return rooms


def extract_wall_graph(go, snap: float = 0.45):
    """graph_out -> (nodes (K,2), room_multihot (K,9), balcony (K,), edges [(i,j)])."""
    rooms = _rings(go)
    if not rooms:
        raise ValueError("no room polygons")

    verts = np.concatenate([P for P, _ in rooms], axis=0)
    # greedy spatial clustering of corners into shared junctions
    reps: list[np.ndarray] = []
    assign = np.empty(len(verts), dtype=int)
    for i, v in enumerate(verts):
        best, bd = -1, snap
        for j, r in enumerate(reps):
            d = float(np.hypot(v[0] - r[0], v[1] - r[1]))
            if d <= bd:
                best, bd = j, d
        if best < 0:
            reps.append(v.copy())
            best = len(reps) - 1
        assign[i] = best
    nodes = np.array([r for r in reps], dtype=float)
    K = len(nodes)
    # average cluster members for a stable junction position
    sums = np.zeros((K, 2))
    cnt = np.zeros(K)
    for i, v in enumerate(verts):
        sums[assign[i]] += v
        cnt[assign[i]] += 1
    nodes = sums / np.maximum(cnt[:, None], 1)

    rmh = np.zeros((K, N_ROOM), dtype=np.float32)
    bal = np.zeros(K, dtype=np.float32)
    edges = set()
    off = 0
    for P, rt in rooms:
        idx = assign[off: off + len(P)]
        off += len(P)
        for k in idx:
            if 0 <= rt < N_ROOM:
                rmh[k, rt] = 1.0
            if rt == BALCONY:
                bal[k] = 1.0
        m = len(idx)
        for k in range(m):
            a, b = int(idx[k]), int(idx[(k + 1) % m])
            if a != b:
                edges.add((min(a, b), max(a, b)))
    return nodes, rmh, bal, sorted(edges)


def reconstruct(nodes, edges, rmh, min_area: float = 1.0):
    """structural graph -> graph_out (rooms = polygonized minimal loops).

    Drops thin wall-gap slivers (< min_area) and connects rooms that share a wall
    into a sparse, near-tree access graph (minimum spanning tree) so the render
    matches real plans instead of a fully-connected hairball."""
    import torch
    lines = [LineString([nodes[a], nodes[b]]) for a, b in edges
             if not np.allclose(nodes[a], nodes[b])]
    faces = [f for f in polygonize(lines) if f.area >= min_area]
    G = nx.Graph()
    nid = 1
    for f in faces:
        ring = np.asarray(f.exterior.coords)
        votes = np.zeros(N_ROOM)
        for p in ring:
            d = np.hypot(nodes[:, 0] - p[0], nodes[:, 1] - p[1])
            votes += rmh[int(d.argmin())]
        rt = int(votes.argmax()) if votes.any() else 0
        G.add_node(nid, geometry=list(f.exterior.coords), room_type=rt,
                   centroid=torch.tensor([f.centroid.x, f.centroid.y]))
        nid += 1

    # sparse access graph: MST over rooms that share a wall
    polys = {n: Polygon(d["geometry"]) for n, d in G.nodes(data=True)}
    ids = list(polys)
    adj = nx.Graph()
    adj.add_nodes_from(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            shared = polys[a].buffer(0.35).intersection(polys[b].buffer(0.35)).length
            if shared > 0.6:
                ca, cb = G.nodes[a]["centroid"], G.nodes[b]["centroid"]
                w = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                adj.add_edge(a, b, weight=w)
    for u, v in nx.minimum_spanning_tree(adj).edges():
        G.add_edge(u, v, connectivity="door")
    return G
