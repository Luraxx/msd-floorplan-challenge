"""
Baseline v2 — graph-conditioned partition + labelling.

Instead of retrieving a whole real plan (baseline v1), we GENERATE geometry per
graph node: partition the apartment envelope into one cell per node (cell walls =
generated partition walls) and label each node's room_type from its zoning_type.
The output is a real `graph_out` (geometry + room_type + centroid; edges = the
input connectivity), which natively satisfies the scored format and renders
cleanly through the official plot_floor.

Pipeline (matches the intended design):
    graph_in (+ envelope)  ->  partition envelope into cells   (partition.py)
                           ->  label each cell from zoning      (labeling.py)
                           ->  assemble graph_out, validate     (eval/validate.py)
                           ->  [optional] place door geometry    (place_doors)

Doors: connectivity edges (door/passage/entrance) are carried over from the input
graph, so door TOPOLOGY is always preserved. `--doors` additionally places a door
point on the shared wall between two connected cells (geometry the scorer ignores
but downstream/visualisation wants).

Usage:
    # no data yet — verify the whole pipeline end-to-end on a synthetic apartment
    python src/model/baseline_partition.py --selfcheck

    # real data: generate predictions for the test split
    python src/model/baseline_partition.py --test <MSD>/test --train <MSD>/train \
        --out outputs/generated_v2 --n 400
    python src/eval/run_eval.py --real <MSD>/test/graph_out --fake outputs/generated_v2 --n 400
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle  # MSD ships graph_in/out as pickles (Kaggle, ECCV 2024) — dataset's native, trusted format
import sys

import networkx as nx
import numpy as np
import torch  # graph_out centroids are torch tensors (matches MSD format / baseline_retrieval)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # model/
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))       # eval/

from labeling import learn_mapping, fallback_mapping, label  # noqa: E402
from partition import (  # noqa: E402
    envelope_from_polygons,
    envelope_from_struct_in,
    partition_envelope,
)
from validate import validate_graph_out  # noqa: E402


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_graph_out(graph_in: nx.Graph, cells: dict, mapping: dict) -> nx.Graph:
    """Assemble a graph_out: per-node geometry + room_type + centroid; edges keep connectivity."""
    G = nx.Graph()
    G.graph.update(graph_in.graph)
    for n in graph_in.nodes:
        cell = cells[n]
        coords = list(zip(*cell.exterior.coords.xy))
        rt = label(graph_in.nodes[n].get("zoning_type"), mapping)
        G.add_node(n, geometry=coords, room_type=rt,
                   centroid=torch.tensor([cell.centroid.x, cell.centroid.y]))
    for u, v, d in graph_in.edges(data=True):
        G.add_edge(u, v, connectivity=d.get("connectivity"))
    return G


def place_doors(G: nx.Graph) -> nx.Graph:
    """
    Post-step: put a door point on the shared wall of each door/entrance edge.
    Stored as edge attr `door_geometry` ([x, y]); passages get none (no wall).
    Note: the scorer renders edges by centroid and ignores this — it is for
    downstream use / visualisation.
    """
    from shapely.geometry import Polygon

    for u, v, d in G.edges(data=True):
        if d.get("connectivity") not in ("door", "entrance"):
            continue
        shared = Polygon(G.nodes[u]["geometry"]).intersection(Polygon(G.nodes[v]["geometry"]))
        if shared.is_empty:
            shared = Polygon(G.nodes[u]["geometry"]).boundary.intersection(
                Polygon(G.nodes[v]["geometry"]).boundary)
        if not shared.is_empty:
            pt = shared.interpolate(0.5, normalized=True) if shared.length else shared.representative_point()
            d["door_geometry"] = [float(pt.x), float(pt.y)]
    return G


def predict(graph_in: nx.Graph, envelope, mapping: dict, doors: bool = False) -> nx.Graph:
    cells = partition_envelope(envelope, graph_in)
    G = build_graph_out(graph_in, cells, mapping)
    return place_doors(G) if doors else G


# --------------------------------------------------------------------------- #
# Real-data run
# --------------------------------------------------------------------------- #
def _load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_envelope(test_dir: str, tid: str, source: str, mode: str = "concave"):
    """Envelope for one test id. `struct_in` for the real task; `graph_out` for sanity runs."""
    if source == "graph_out":
        go = _load_pickle(os.path.join(test_dir, "graph_out", f"{tid}.pickle"))
        return envelope_from_polygons([n for _, n in go.nodes("geometry")])
    for ext in (".npy", ".npz"):
        p = os.path.join(test_dir, "struct_in", f"{tid}{ext}")
        if os.path.exists(p):
            arr = np.load(p)
            arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
            return envelope_from_struct_in(arr, mode=mode)
    raise FileNotFoundError(f"no struct_in for {tid} in {test_dir}/struct_in")


def run(args) -> None:
    mapping = learn_mapping(args.train) if args.train and os.path.isdir(args.train) else fallback_mapping()
    print(f"zoning->room mapping: {mapping}  ({'learned' if args.train else 'fallback'})")

    in_dir = os.path.join(args.test, "graph_in")
    ids = [os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(os.path.join(in_dir, "*.pickle")))]
    if args.n:
        ids = ids[: args.n]
    os.makedirs(args.out, exist_ok=True)

    written, failed = 0, []
    for tid in ids:
        try:
            gi = _load_pickle(os.path.join(in_dir, f"{tid}.pickle"))
            env = _load_envelope(args.test, tid, args.envelope_source, mode=args.envelope_mode)
            G = predict(gi, env, mapping, doors=args.doors)
            problems = validate_graph_out(G, gi)
            if problems:
                raise ValueError("; ".join(problems))
            with open(os.path.join(args.out, f"{tid}.pickle"), "wb") as fh:
                pickle.dump(G, fh)
            written += 1
        except Exception as e:  # fail loudly per-id; caller can backfill (e.g. retrieval)
            failed.append((tid, str(e)))

    print(f"Wrote {written}/{len(ids)} predictions to {args.out}")
    if failed:
        print(f"[!] {len(failed)} ids failed (NOT written — backfill with retrieval):")
        for tid, err in failed[:10]:
            print(f"    {tid}: {err}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")


# --------------------------------------------------------------------------- #
# Self-check (runs without the dataset)
# --------------------------------------------------------------------------- #
def selfcheck() -> None:
    """Synthetic apartment -> full pipeline -> assert valid + renders non-blank."""
    from shapely.geometry import Polygon

    # L-shaped envelope (non-trivial outline)
    envelope = Polygon([(0, 0), (10, 0), (10, 6), (6, 6), (6, 10), (0, 10)])

    # 5-room access graph (zoning_type per node, typed connectivity per edge)
    gi = nx.Graph()
    gi.graph["ID"] = "selfcheck"
    for n, z in zip(range(5), [0, 1, 2, 3, 1]):  # Zone1,Zone2,Zone3,Zone4,Zone2
        gi.add_node(n, zoning_type=z)
    gi.add_edges_from([
        (0, 1, {"connectivity": "door"}),
        (1, 2, {"connectivity": "passage"}),
        (1, 4, {"connectivity": "door"}),
        (2, 3, {"connectivity": "door"}),
        (0, 4, {"connectivity": "entrance"}),
    ])

    G = predict(gi, envelope, fallback_mapping(), doors=True)

    problems = validate_graph_out(G, gi)
    assert not problems, "validation failed:\n" + "\n".join(problems)

    # cells tile the envelope (areas sum ~ envelope area, no big gaps/overlaps)
    total = sum(Polygon(G.nodes[n]["geometry"]).area for n in G.nodes)
    cover = total / envelope.area
    assert 0.9 <= cover <= 1.1, f"cells do not tile the envelope (coverage {cover:.2f})"

    # renders to a non-blank, multi-colour image through the OFFICIAL renderer
    from render import render_plan  # noqa: E402
    img = render_plan(G)
    assert img.shape == (512, 512, 3)
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 3, "render looks blank (too few colours)"

    n_doors = sum("door_geometry" in d for _, _, d in G.edges(data=True))
    print(f"OK selfcheck: {G.number_of_nodes()} rooms, coverage {cover:.3f}, "
          f"{len(np.unique(img.reshape(-1, 3), axis=0))} colours, {n_doors} door points placed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", help="MSD test split dir (has graph_in/ and struct_in/)")
    ap.add_argument("--train", help="MSD train split dir — learn zoning->room mapping (optional)")
    ap.add_argument("--out", default="outputs/generated_v2", help="dir to write predicted graph_out pickles")
    ap.add_argument("--n", type=int, default=None, help="cap number of test ids (use equal N at eval)")
    ap.add_argument("--envelope-source", choices=["struct_in", "graph_out"], default="struct_in",
                    help="struct_in = real task; graph_out = sanity run (train-as-test)")
    ap.add_argument("--envelope-mode", choices=["concave", "convex"], default="concave",
                    help="concave = trace real interior footprint (better FID); convex = legacy hull")
    ap.add_argument("--doors", action="store_true", help="also place door-opening geometry (post-step)")
    ap.add_argument("--selfcheck", action="store_true", help="run the synthetic pipeline test and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.test:
        ap.error("--test is required (or use --selfcheck)")
    run(args)


if __name__ == "__main__":
    main()
