"""
Studio: turn a hand-drawn wall sketch into a generated floor plan.

Takes a PNG where the user drew walls (dark strokes on a light canvas), builds a
struct_in-equivalent input, runs the trained U-Net, vectorizes to a graph_out,
renders the colored plan, and emits a "what's where" documentation JSON.

    python scripts/canvas_predict.py --in sketch.png --out plan.png [--rooms 6]

Prints a JSON document to stdout (consumed by /api/canvas).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
import unet_common as U      # noqa: E402
import unet_seg as S         # noqa: E402

ROOM_NAMES = {0: "Bedroom", 1: "Livingroom", 2: "Kitchen", 3: "Dining", 4: "Corridor",
              5: "Stairs", 6: "Storeroom", 7: "Bathroom", 8: "Balcony"}
ROOM_COLORS = {0: "#1f77b4", 1: "#e6550d", 2: "#fd8d3c", 3: "#fdae6b", 4: "#fdd0a2",
               5: "#72246c", 6: "#5254a3", 7: "#6b6ecf", 8: "#2ca02c"}


def build_input(png_path, size):
    """Hand-drawn walls PNG -> (3,size,size) model input + coord ramps + free mask."""
    from scipy import ndimage
    img = Image.open(png_path).convert("L").resize((size, size), Image.NEAREST)
    a = np.asarray(img).astype(np.float32)        # 0=black stroke (wall) .. 255=white
    free_all = (a >= 127)                         # not a wall stroke
    # Keep only the ENCLOSED building interior: flood-fill free space from the
    # canvas border (that reachable region is "outside"); rooms go in the rest.
    lab, _ = ndimage.label(free_all)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    outside = np.isin(lab, list(border)) if border else np.zeros_like(free_all)
    interior = free_all & ~outside
    if interior.mean() < 0.04:                     # outline not closed -> use full sketch
        interior = free_all
    free = interior.astype(np.float32)             # 1 free (inside), 0 wall/outside
    # synthetic world-coordinate ramps at a plausible training-like scale
    col_vals = np.linspace(-20.0, 20.0, size).astype(np.float32)
    row_vals = np.linspace(-20.0, 20.0, size).astype(np.float32)
    cc = np.clip(col_vals / 40.0, -1, 1)
    rr = np.clip(row_vals / 40.0, -1, 1)
    colp = np.broadcast_to(cc[None, :], (size, size))
    rowp = np.broadcast_to(rr[:, None], (size, size))
    x = np.stack([free, colp, rowp], axis=0).astype(np.float32)
    return x, col_vals, row_vals, free


def render_aligned(G, col_vals, row_vals, free, size=512):
    """Render the generated plan in the SKETCH's coordinate frame.

    The official render_plan autoscales to the room bounding box, so a small plan
    drawn in a corner gets blown up to fill the image and no longer matches the
    sketch. Here we (a) fix the view to the sketch's full world extent and (b)
    orient y like the canvas (row 0 = top), and (c) overlay the drawn walls — so
    the result sits exactly where you drew it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(ROOT, "src", "msd_vendor"))
    import plot as msd_plot  # noqa
    from skimage import measure

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    try:
        if G.number_of_nodes() > 0:
            msd_plot.plot_floor(G, ax)
    except Exception:
        pass
    # overlay the walls you drew (contour of the free/interior mask)
    try:
        nc, nr = len(col_vals), len(row_vals)
        for cont in measure.find_contours(free.astype(float), 0.5):
            xs = [float(col_vals[min(int(round(c)), nc - 1)]) for _, c in cont]
            ys = [float(row_vals[min(int(round(r)), nr - 1)]) for r, _ in cont]
            ax.plot(xs, ys, color="#111111", lw=1.8, zorder=5)
    except Exception:
        pass
    # frame to the sketch extent, y oriented like the canvas (top = row 0)
    ax.set_xlim(float(col_vals.min()), float(col_vals.max()))
    ax.set_ylim(float(row_vals.max()), float(row_vals.min()))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.path.join(ROOT, "outputs", "unet.pt"))
    ap.add_argument("--rooms", type=int, default=0, help="optional target #rooms hint (0=auto)")
    args = ap.parse_args()

    import torch
    dev = S._device()
    if not os.path.exists(args.weights):
        print(json.dumps({"error": f"no trained weights at {args.weights} — train a model first"}))
        sys.exit(0)
    ckpt = torch.load(args.weights, map_location=dev)
    size = ckpt["size"]
    mu, sd = ckpt["mu"].cpu(), ckpt["sd"].cpu()
    base = ckpt.get("base", 24)
    model = S.build_model(base) if "base" in ckpt else S.build_model()
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    x, col_vals, row_vals, free = build_input(args.inp, size)

    # graph conditioning: neutral (training mean -> normalized 0), optional #rooms hint
    feat = mu.clone().float()
    if args.rooms and args.rooms > 0:
        feat[0] = float(args.rooms)
    fn = (feat - mu) / sd

    with torch.no_grad():
        out = model(torch.tensor(x[None]).to(dev), fn[None].to(dev))
        lab = out.argmax(1)[0].cpu().numpy().astype(np.int64)
    lab = np.where(free > 0.5, lab, 0)            # rooms only in free space
    # Clean the noisy argmax for a presentable sketch result: majority/modal
    # smoothing collapses salt-and-pepper fragments into coherent rooms.
    try:
        from skimage.filters.rank import modal
        from skimage.morphology import disk
        sm = modal(lab.astype(np.uint8), disk(max(3, size // 48)))
        lab = np.where(free > 0.5, sm.astype(np.int64), 0)
    except Exception:
        pass
    # larger min room size for hand sketches -> fewer, cleaner rooms
    g = U.vectorize(lab, col_vals, row_vals, min_px=max(8, (size * size) // 700))

    # render the generated plan IN THE SAME FRAME as the sketch so it lines up
    img = render_aligned(g, col_vals, row_vals, free, size=512)
    Image.fromarray(img).save(args.out)

    # documentation: what's where
    from collections import Counter
    cnt = Counter()
    area = Counter()
    for _, d in g.nodes(data=True):
        rt = int(d.get("room_type", 0))
        cnt[rt] += 1
    total_free = float((free > 0.5).mean())
    rooms_doc = [
        {"type": rt, "name": ROOM_NAMES.get(rt, str(rt)), "color": ROOM_COLORS.get(rt, "#333"),
         "count": cnt[rt]}
        for rt in sorted(cnt, key=lambda r: -cnt[r])
    ]
    doc = {
        "ok": True,
        "n_rooms": g.number_of_nodes(),
        "n_connections": g.number_of_edges(),
        "free_fraction": round(total_free, 3),
        "rooms": rooms_doc,
        "size": size,
        "weights": os.path.basename(args.weights),
    }
    print(json.dumps(doc))


if __name__ == "__main__":
    main()
