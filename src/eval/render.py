"""
Render MSD floor-plan graphs to fixed-size RGB images using the OFFICIAL
MSD plot.py (plot_floor). Real and generated plans MUST go through this same
function so the evaluation images share one distribution.

A "plan" here is a networkx.Graph with node attrs `geometry` (polygon coords),
`room_type` (int) and `centroid`, and edge attr `connectivity` — exactly the
format of the MSD `graph_out` pickles.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt

# Make the vendored MSD package importable (plot.py does `from constants import ...`)
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "msd_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
import plot as msd_plot  # noqa: E402

IMG_SIZE = 512  # px (square); Inception resizes to 299 internally


def _fig_to_array(fig) -> np.ndarray:
    """Grab the canvas as an (H, W, 3) uint8 RGB array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def render_plan(G, size: int = IMG_SIZE) -> np.ndarray:
    """Render one floor-plan graph to a (size, size, 3) uint8 RGB array."""
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the whole canvas, no margins
    ax.set_facecolor("white")
    try:
        msd_plot.plot_floor(G, ax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.02)
        img = _fig_to_array(fig)
    finally:
        plt.close(fig)
    return img


def render_plans(graphs, size: int = IMG_SIZE, progress: bool = True) -> np.ndarray:
    """Render a list of graphs -> (N, size, size, 3) uint8 array."""
    out = np.empty((len(graphs), size, size, 3), dtype=np.uint8)
    for i, G in enumerate(graphs):
        out[i] = render_plan(G, size)
        if progress and (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{len(graphs)}")
    return out
