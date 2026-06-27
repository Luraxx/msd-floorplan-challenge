"""
Per-node room-type labeling: map a `zoning_type` (input graph node attr) to a
`room_type` (output graph node attr).

This is the "labeling" half of baseline v2 — assign every graph vertex a room
class. The mapping is data-driven when a train split is available (we learn the
empirical argmax P(room_type | zoning_type) from the node-aligned
graph_in / graph_out pairs, which handles the zoning->room fan-out, e.g. Zone2
-> {Livingroom, Kitchen, Dining, Corridor}); it falls back to the fixed
zoning->room grouping from the MSD constants when no train data is present.

`room_type` / `zoning_type` are integer indices into ROOM_NAMES / ZONING_NAMES
(see src/msd_vendor/constants.py).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

# vendored MSD constants (ZONING_NAMES, ROOM_NAMES, ZONING_ROOMS)
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "msd_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from constants import ROOM_NAMES, ZONING_NAMES, ZONING_ROOMS  # noqa: E402

STRUCTURE = ROOM_NAMES.index("Structure")


def fallback_mapping() -> dict[int, int]:
    """zoning_type idx -> room_type idx via the fixed MSD grouping (first room in each zone)."""
    m: dict[int, int] = {}
    for zi, zname in enumerate(ZONING_NAMES):
        rooms = ZONING_ROOMS.get(zname, [])
        rname = next((r for r in rooms if r in ROOM_NAMES), None)
        m[zi] = ROOM_NAMES.index(rname) if rname else STRUCTURE
    return m


def learn_mapping(train_dir: str, limit: int | None = None) -> dict[int, int]:
    """
    Learn argmax room_type per zoning_type from node-aligned graph_in/graph_out.
    Falls back to the fixed grouping for any zoning_type unseen in train.
    """
    import torch  # noqa: F401  (graph_out centroids unpickle as torch tensors)

    in_files = sorted(glob.glob(os.path.join(train_dir, "graph_in", "*.pickle")))
    if limit:
        in_files = in_files[:limit]
    counts: dict[int, Counter] = defaultdict(Counter)
    for f in in_files:
        tid = os.path.splitext(os.path.basename(f))[0]
        out_f = os.path.join(train_dir, "graph_out", f"{tid}.pickle")
        if not os.path.exists(out_f):
            continue
        with open(f, "rb") as fh:
            gi = pickle.load(fh)
        with open(out_f, "rb") as fh:
            go = pickle.load(fh)
        for n in gi.nodes:
            if n not in go.nodes:
                continue
            z, r = gi.nodes[n].get("zoning_type"), go.nodes[n].get("room_type")
            if z is not None and r is not None:
                counts[int(z)][int(r)] += 1

    mapping = fallback_mapping()
    for z, c in counts.items():
        if c:
            mapping[z] = c.most_common(1)[0][0]
    return mapping


def label(zoning_type, mapping: dict[int, int]) -> int:
    """Room-type index for a node's zoning_type (Structure if unknown)."""
    if zoning_type is None:
        return STRUCTURE
    return int(mapping.get(int(zoning_type), STRUCTURE))


def save_mapping(mapping: dict[int, int], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({str(k): v for k, v in mapping.items()}, fh, indent=2)


def load_mapping(path: str) -> dict[int, int]:
    with open(path) as fh:
        return {int(k): int(v) for k, v in json.load(fh).items()}
