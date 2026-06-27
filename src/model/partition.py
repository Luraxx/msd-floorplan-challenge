"""
Graph-conditioned partition of an apartment envelope into one labelled cell per
graph node — the geometry half of the "label every graph vertex" baseline.

Framing (per the challenge): the load-bearing structure only gives the outer
envelope, NOT the partition walls between rooms. So we GENERATE the partition:
place one seed per graph node, lay the seeds out with a spring layout driven by
the room-adjacency graph (adjacent rooms attract -> their cells end up sharing a
wall), then take the Voronoi cells clipped to the envelope. The cell boundaries
ARE the generated partition walls, and the cells tile the envelope by
construction.

The spring layout is the cheap "constraint solver": it makes Voronoi adjacency
approximate the target graph adjacency. A few Lloyd iterations regularise the
cell shapes.

ponytail: spring-layout seeds + Lloyd is the minimal solver that produces valid,
tiling, graph-shaped cells. Upgrade path when geometry fidelity matters: replace
_seed_positions/_lloyd with an explicit optimiser that (a) penalises Voronoi
adjacencies that disagree with the graph edges and (b) drives each cell toward a
target area learned per room_type from train.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points, unary_union, voronoi_diagram

WALL_BRIDGE_DISTANCE = 0.3  # meters; matches README outline derivation


# --------------------------------------------------------------------------- #
# Envelope sources
# --------------------------------------------------------------------------- #
def envelope_from_polygons(polys) -> Polygon:
    """Outline = morphological close of room polygons (README's outline derivation)."""
    shapes = [Polygon(p) for p in polys]
    closed = unary_union(shapes).buffer(WALL_BRIDGE_DISTANCE).buffer(-WALL_BRIDGE_DISTANCE)
    return _largest_polygon(closed)


def envelope_from_struct_in(struct_in: np.ndarray, mode: str = "concave") -> Polygon:
    """
    Apartment envelope from the struct_in tensor.

    Coordinate convention (verified, matches unet_common): ch0 = wall/free mask
    (0 = wall, 255 = free); ch1 = world x per COLUMN; ch2 = world y per ROW.

    mode="concave" (default): trace the ENCLOSED building interior (flood-fill the
      free space from the border = outside; the rest is the real, concave
      footprint). This is the FID lever vs. a convex hull, which spills cells
      outside the true plan shape.
    mode="convex": convex hull of the structure pixels (legacy fallback).
    """
    s = np.asarray(struct_in, dtype=float)
    ch0 = s[..., 0]
    col_x = s[0, :, 1]   # world x for each column (ch1 is constant down columns)
    row_y = s[:, 0, 2]   # world y for each row    (ch2 is constant along rows)
    nc, nr = len(col_x), len(row_y)

    if mode == "concave":
        try:
            from scipy import ndimage
            from skimage import measure
            free = ch0 > 127
            walls = ndimage.binary_dilation(~free, iterations=2)   # close small gaps
            free2 = ~walls
            lab, _ = ndimage.label(free2)
            border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
            border.discard(0)
            outside = np.isin(lab, list(border)) if border else np.zeros_like(free2)
            interior = ndimage.binary_dilation(free2 & ~outside, iterations=2)
            if interior.sum() >= 0.02 * interior.size:
                lab2, n2 = ndimage.label(interior)
                if n2 > 1:
                    sizes = ndimage.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))
                    interior = lab2 == int(np.argmax(sizes) + 1)
                contours = measure.find_contours(interior.astype(float), 0.5)
                if contours:
                    cont = measure.approximate_polygon(max(contours, key=len), tolerance=2.0)
                    pts = [(float(col_x[min(int(round(c)), nc - 1)]),
                            float(row_y[min(int(round(r)), nr - 1)])) for r, c in cont]
                    if len(pts) >= 3:
                        poly = Polygon(pts)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        poly = _largest_polygon(poly)
                        if not poly.is_empty and poly.area > 0:
                            return poly
        except Exception:
            pass  # fall through to the convex hull

    mask = ch0 < 128  # walls/structure footprint
    ys, xs = np.where(mask)
    if len(xs) < 3:
        raise ValueError("struct_in has too few structure pixels for an envelope")
    pts = np.column_stack([col_x[xs], row_y[ys]])
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    if hull.geom_type != "Polygon":
        raise ValueError("could not form an envelope polygon from struct_in")
    return hull


# --------------------------------------------------------------------------- #
# Seed placement (the "constraint solver")
# --------------------------------------------------------------------------- #
def _nearest_inside(env: Polygon, p: Point) -> Point:
    """Pull a point that landed outside the envelope just inside it."""
    edge = nearest_points(env, p)[0]
    c = np.array(env.representative_point().coords[0])
    q = np.array(edge.coords[0])
    inward = Point(q + 0.01 * (c - q))
    return inward if env.contains(inward) else env.representative_point()


def _dejitter(seeds: dict, eps: float = 1e-3) -> dict:
    """Voronoi needs distinct generators — nudge coincident seeds apart."""
    rng = np.random.default_rng(0)
    seen: list = []
    out: dict = {}
    for n, s in seeds.items():
        s = np.asarray(s, dtype=float)
        while any(np.allclose(s, t, atol=eps) for t in seen):
            s = s + rng.normal(0, eps * 10, 2)
        seen.append(s)
        out[n] = s
    return out


def _seed_positions(graph: nx.Graph, envelope: Polygon, seed: int = 7) -> dict:
    """One 2D seed per node, laid out by the adjacency graph, fit inside the envelope."""
    nodes = list(graph.nodes)
    if len(nodes) == 1:
        return {nodes[0]: np.array(envelope.representative_point().coords[0])}

    pos = nx.spring_layout(graph, seed=seed)  # node -> [x, y] in ~[-1, 1]
    P = np.array([pos[n] for n in nodes])
    minx, miny, maxx, maxy = envelope.bounds
    box = np.array([maxx - minx, maxy - miny])
    margin = 0.05
    lo = np.array([minx, miny]) + margin * box
    hi = np.array([maxx, maxy]) - margin * box
    span = np.maximum(P.max(0) - P.min(0), 1e-9)
    Q = lo + (P - P.min(0)) / span * (hi - lo)

    seeds = {}
    for n, q in zip(nodes, Q):
        p = Point(q)
        seeds[n] = np.array((p if envelope.contains(p) else _nearest_inside(envelope, p)).coords[0])
    return _dejitter(seeds)


def _lloyd(seeds: dict, envelope: Polygon, iters: int) -> dict:
    """Centroidal relaxation: move each seed to its cell's centroid."""
    for _ in range(iters):
        cells = _voronoi_cells(seeds, envelope)
        moved = {
            n: (np.array(c.representative_point().coords[0]) if not c.is_empty and c.area > 0 else seeds[n])
            for n, c in cells.items()
        }
        seeds = _dejitter(moved)
    return seeds


# --------------------------------------------------------------------------- #
# Voronoi cells
# --------------------------------------------------------------------------- #
def _largest_polygon(geom) -> Polygon:
    """Reduce any geometry to its largest Polygon part (intersections can split)."""
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(polys, key=lambda g: g.area) if polys else Polygon()


def _voronoi_cells(seeds: dict, envelope: Polygon) -> dict:
    """{node: cell clipped to envelope}. Each Voronoi region contains its own seed."""
    nodes = list(seeds)
    if len(nodes) == 1:
        return {nodes[0]: envelope}

    regions = list(voronoi_diagram(MultiPoint([tuple(seeds[n]) for n in nodes]),
                                   envelope=envelope).geoms)
    cells = {}
    for n in nodes:
        sp = Point(seeds[n])
        region = next((g for g in regions if g.contains(sp)), None)
        if region is None:  # numerical edge case: pick the closest region
            region = min(regions, key=lambda g: g.distance(sp))
        cells[n] = region.intersection(envelope)
    return cells


def partition_envelope(envelope: Polygon, graph: nx.Graph,
                       lloyd_iters: int = 2, seed: int = 7) -> dict:
    """
    Return {node: shapely Polygon} — one room cell per graph node, tiling the
    envelope, with cell adjacency shaped by the graph's edges.
    """
    if envelope.is_empty or envelope.area <= 0:
        raise ValueError("empty envelope")
    seeds = _seed_positions(graph, envelope, seed=seed)
    if lloyd_iters:
        seeds = _lloyd(seeds, envelope, lloyd_iters)
    return {n: _largest_polygon(c) for n, c in _voronoi_cells(seeds, envelope).items()}
