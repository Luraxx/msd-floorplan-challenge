"""
Challenge evaluation CLI.

Renders a set of REAL plans and a set of GENERATED plans through the official
MSD renderer, then reports FID + density + coverage (the leaderboard metrics).

Plans are networkx.Graph pickles in MSD `graph_out` format (node attrs
`geometry`, `room_type`, `centroid`; edge attr `connectivity`).

Examples:
    # Compare your generated plans against the real test set
    python src/eval/run_eval.py --real <dir_of_real_pickles> --fake <dir_of_generated_pickles>

    # Quick self-check against the local MSD dataset (real vs real holdout)
    python src/eval/run_eval.py --real "$MSD/test/graph_out" --fake "$MSD/train/graph_out" --n 500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render import render_plans          # noqa: E402
from metrics import compute_metrics      # noqa: E402


def load_graphs(path: str, limit: int | None = None):
    """Load graph pickles from a directory (or a single .pickle file)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.pickle")))
    else:
        files = [path]
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"[!] No .pickle graphs found at {path}")
    graphs = []
    for f in files:
        with open(f, "rb") as fh:
            graphs.append(pickle.load(fh))
    return graphs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="dir/file of real graph pickles")
    ap.add_argument("--fake", required=True, help="dir/file of generated graph pickles")
    ap.add_argument("--n", type=int, default=None, help="cap samples per side (use equal N)")
    ap.add_argument("--nearest-k", type=int, default=5)
    ap.add_argument("--size", type=int, default=512, help="render canvas px")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--out", default=None, help="optional path to write metrics JSON")
    args = ap.parse_args()

    real_graphs = load_graphs(args.real, args.n)
    fake_graphs = load_graphs(args.fake, args.n)
    n = min(len(real_graphs), len(fake_graphs))
    real_graphs, fake_graphs = real_graphs[:n], fake_graphs[:n]
    print(f"Evaluating {n} real vs {n} generated plans (k={args.nearest_k}, size={args.size})")

    t = time.time()
    real_imgs = render_plans(real_graphs, size=args.size)
    fake_imgs = render_plans(fake_graphs, size=args.size)
    print(f"Rendered {2 * n} images in {time.time() - t:.1f}s")

    t = time.time()
    m = compute_metrics(real_imgs, fake_imgs, nearest_k=args.nearest_k, device=args.device)
    print(f"Computed metrics in {time.time() - t:.1f}s\n")

    print("=" * 40)
    print(f"  FID       {m['fid']:.3f}   (lower better)")
    print(f"  Density   {m['density']:.3f}   (higher better)")
    print(f"  Coverage  {m['coverage']:.3f}   (higher better)")
    print("-" * 40)
    print(f"  precision {m['precision']:.3f}   recall {m['recall']:.3f}")
    print("=" * 40)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(m, fh, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
