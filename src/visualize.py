"""
MSD Floor Plan — data exploration & visualization.

Loads the Modified Swiss Dwellings CSV, prints a structural overview, and
renders sample apartments as (Outline -> Rooms) pairs — the input/target of
the generative challenge.

Usage:
    python src/visualize.py
    python src/visualize.py --csv data/mds_V2_5.372k.csv --n 6
    python src/visualize.py --unit-id 64314          # render one specific unit
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "mds_V2_5.372k.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

# Distance (meters) used to bridge wall gaps when deriving the outline.
WALL_BRIDGE_DISTANCE = 0.3


def load_gdf(csv_path: str) -> gpd.GeoDataFrame:
    """Read the CSV and parse the WKT 'geom' column into a GeoDataFrame."""
    if not os.path.exists(csv_path):
        sys.exit(
            f"\n[!] CSV not found: {csv_path}\n"
            "    Download it from Kaggle and place it in data/:\n"
            "    https://www.kaggle.com/datasets/caspervanengelenburg/"
            "modified-swiss-dwellings/data\n"
        )
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["geom"] = df["geom"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geom")


def overview(gdf: gpd.GeoDataFrame) -> None:
    """Print a quick structural summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows (geometries): {len(gdf):,}")
    print(f"Columns: {list(gdf.columns)}")

    for key in ("plan_id", "unit_id"):
        if key in gdf.columns:
            print(f"Unique {key}: {gdf[key].nunique():,}")

    if "entity_type" in gdf.columns:
        print("\nentity_type counts:")
        print(gdf["entity_type"].value_counts().to_string())

    # Try to find a room-type column (name varies between dataset versions).
    type_cols = [c for c in gdf.columns
                 if c not in ("geom", "plan_id", "unit_id", "entity_type")
                 and gdf[c].dtype == object]
    for c in type_cols:
        n = gdf[c].nunique()
        if 1 < n < 60:  # looks categorical
            print(f"\nLikely room-type column '{c}' ({n} categories):")
            print(gdf[c].value_counts().head(20).to_string())
    print("=" * 60 + "\n")


def rooms_of(gdf: gpd.GeoDataFrame, unit_id) -> gpd.GeoDataFrame:
    """Return the room ('area') geometries for one apartment."""
    apt = gdf[gdf["unit_id"] == unit_id]
    if "entity_type" in apt.columns:
        apt = apt[apt["entity_type"] == "area"]
    return apt


def derive_outline(rooms_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Morphological close: dilate -> union -> erode to get the apartment shell."""
    geom = (
        rooms_gdf.geometry.buffer(WALL_BRIDGE_DISTANCE)
        .union_all()
        .buffer(-WALL_BRIDGE_DISTANCE)
    )
    return gpd.GeoDataFrame(geometry=[geom], crs=rooms_gdf.crs)


def color_arg(rooms_gdf: gpd.GeoDataFrame):
    """Color rooms by type if a categorical type column exists, else by index."""
    for c in ("room_type", "roomtype", "type", "category", "label"):
        if c in rooms_gdf.columns:
            return {"column": c, "legend": True, "cmap": "Set3"}
    return {"cmap": "Set3"}


def plot_pair(rooms_gdf: gpd.GeoDataFrame, unit_id, ax1, ax2) -> None:
    outline = derive_outline(rooms_gdf)

    outline.plot(ax=ax1, facecolor="#f4f4f4", edgecolor="black", linewidth=3)
    ax1.set_title("Input: Apartment Outline", fontsize=12, fontweight="bold")
    ax1.axis("equal"); ax1.axis("off")

    rooms_gdf.plot(ax=ax2, edgecolor="white", linewidth=1.5, **color_arg(rooms_gdf))
    outline.plot(ax=ax2, facecolor="none", edgecolor="black", linewidth=2, alpha=0.4)
    ax2.set_title(f"Target: Rooms (unit {unit_id})", fontsize=12, fontweight="bold")
    ax2.axis("equal"); ax2.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--n", type=int, default=6, help="number of sample apartments")
    p.add_argument("--unit-id", type=int, default=None, help="render one specific unit")
    args = p.parse_args()

    gdf = load_gdf(args.csv)
    overview(gdf)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.unit_id is not None:
        rooms = rooms_of(gdf, args.unit_id)
        if rooms.empty:
            sys.exit(f"[!] No rooms found for unit_id {args.unit_id}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plot_pair(rooms, args.unit_id, ax1, ax2)
        out = os.path.join(OUT_DIR, f"unit_{args.unit_id}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        return

    # Grid of N sample apartments (pick units with several rooms).
    counts = gdf[gdf.get("entity_type", "area") == "area"].groupby("unit_id").size()
    sample_units = counts[counts >= 3].sort_values(ascending=False).head(args.n).index.tolist()

    fig, axes = plt.subplots(args.n, 2, figsize=(12, 5 * args.n))
    if args.n == 1:
        axes = [axes]
    for row, uid in zip(axes, sample_units):
        plot_pair(rooms_of(gdf, uid), uid, row[0], row[1])

    out = os.path.join(OUT_DIR, "samples_overview.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}  ({len(sample_units)} apartments)")


if __name__ == "__main__":
    main()
