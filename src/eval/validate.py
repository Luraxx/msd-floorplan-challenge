"""
Validate a graph_out before it is written — no hidden failures.

The official renderer (src/msd_vendor/plot.py) indexes node key **1** to pick its
colormap (`list(G.nodes[1].keys())`) and reads `geometry`, `room_type`,
`centroid` on every node plus `connectivity` on every edge. A graph that is
missing any of these does not necessarily crash — it can render blank or silently
switch to the *zoning* colormap (a hidden distribution shift that tanks FID). So
we assert the contract loudly and let the caller fall back per-id instead of
emitting an unscorable pickle.
"""
from __future__ import annotations

from shapely.geometry import Polygon

CONN = {"door", "passage", "entrance"}


def validate_graph_out(G, graph_in=None) -> list[str]:
    """Return a list of contract violations (empty list == valid)."""
    problems: list[str] = []

    if G.number_of_nodes() < 2:
        problems.append(f"need >=2 nodes (renderer indexes node 1); got {G.number_of_nodes()}")
    if 1 not in G.nodes:
        problems.append("node key 1 missing (renderer reads G.nodes[1] to choose the colormap)")
    elif "room_type" not in G.nodes[1]:
        problems.append("node 1 lacks 'room_type' -> renderer silently falls back to zoning colormap")

    for n in G.nodes:
        att = G.nodes[n]
        for k in ("geometry", "room_type", "centroid"):
            if k not in att:
                problems.append(f"node {n} missing '{k}'")
        geom = att.get("geometry")
        if geom is not None:
            try:
                p = Polygon(geom)
                if not p.is_valid or p.area <= 0:
                    problems.append(f"node {n} geometry is not a positive-area polygon")
            except Exception as e:  # geometry not coercible to a polygon
                problems.append(f"node {n} geometry not polygon-able: {e}")
        c = att.get("centroid")
        if c is not None:
            try:
                if len(list(c)) != 2:
                    problems.append(f"node {n} centroid is not length-2")
            except TypeError:
                problems.append(f"node {n} centroid is not iterable")

    for u, v, d in G.edges(data=True):
        if d.get("connectivity") not in CONN:
            problems.append(f"edge ({u},{v}) connectivity {d.get('connectivity')!r} not in {sorted(CONN)}")

    if graph_in is not None and set(G.nodes) != set(graph_in.nodes):
        problems.append("node set not aligned with graph_in")

    return problems


def assert_valid(G, graph_in=None):
    problems = validate_graph_out(G, graph_in)
    if problems:
        raise ValueError("invalid graph_out:\n  - " + "\n  - ".join(problems))
    return G
