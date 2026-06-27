# EHL Paris Hackathon — Floor Plan Generation Challenge

Conditional generation of apartment floor plans on the **Modified Swiss Dwellings (MSD)**
dataset. Given an empty apartment **outline**, the model must generate the **room layout**
(rooms + types) that fills it.

> Org: Davis (commercialdeckdavis.com) · Started: 2026-06-27

---

## 1. The Task

| | |
|---|---|
| **Input (condition)** | Clean apartment outline (the building envelope) — one solid polygon |
| **Output (target)** | Room layout: individual room polygons, each with a room type |
| **Constraint** | The generated rooms must tile the given outline |

It is a **conditional** generation problem (`outline → rooms`), not unconditional
sampling. The input outline is *derived* from the ground-truth rooms (see §3), so every
apartment yields a training pair automatically.

---

## 2. Data

- **Source:** [Kaggle — Modified Swiss Dwellings](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings/data)
- **File:** `mds_V2_5.372k.csv` (383 MB) in `data/` — **1,086,846 geometries**, **5,372 floors** (`plan_id`), **18,902 apartments** (`unit_id`).
- **Format:** each row is a geometry stored as a **WKT** string; load with GeoPandas/Shapely.

Confirmed columns: `apartment_id, site_id, building_id, plan_id, floor_id, unit_id,
area_id, unit_usage, entity_type, entity_subtype, geom, elevation, height, zoning, roomtype`.

| Column | Meaning |
|---|---|
| `geom` | Geometry (polygon) as WKT |
| `plan_id` | A whole **floor** (may contain several apartments) |
| `unit_id` | A single **apartment** = one training example |
| `entity_type` | `area` = room · `separator` = wall · `opening` = door/window |
| `roomtype` | Balcony, Bathroom, Bedroom, Corridor, Kitchen, Livingroom, Storeroom, Structure |

`entity_type` breakdown: `separator` 602,196 · `opening` 281,320 · `area` 203,330.

### Two representations (important)

| | `mds_V2_5.372k.csv` | `modified-swiss-dwellings-v2/` (16 GB) |
|---|---|---|
| Form | flat **polygons** (WKT) per `unit_id` | **graphs + tensors**, `train`/`test` splits |
| Use | quick exploration, the Discord starter snippet | the canonical ML-ready format |

Inside the v2 folder (4,572 train samples), per id:

| Subfolder | Content |
|---|---|
| `graph_in` / `graph_out` | networkx graphs — **input** access graph / **target** floor plan |
| `struct_in` | `(512,512,3)` float16 — model input condition (the structure) |
| `full_out` | `(512,512,3)` float16 — **model-ready tensor, NOT a viewable image** (values ≈ −16…16) |

A `graph_out` node has `geometry` (polygon coords), `room_type` (int), `centroid`;
edges have `connectivity` (`door`/`passage`/`entrance`). This is exactly what MSD's
`plot_floor` renders — so **`graph_out` is the canonical thing we score** (see §4).
The v2 folder lives on disk at `~/Downloads/archive/`, not in this repo.

```python
import pandas as pd, geopandas as gpd
from shapely import wkt
df = pd.read_csv("data/mds_V2_5.372k.csv")
df["geom"] = df["geom"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry="geom")
```

---

## 3. Deriving the input outline

The outline is computed from the rooms via a **morphological close**
(dilate → union → erode), which bridges the wall gaps between rooms into one shell:

```python
WALL_BRIDGE_DISTANCE = 0.3  # 30 cm (MSD units are meters)
outline = (rooms.geometry
           .buffer(WALL_BRIDGE_DISTANCE)   # 1. inflate rooms to close gaps
           .union_all()                    # 2. merge into one solid shape
           .buffer(-WALL_BRIDGE_DISTANCE)) # 3. shrink back to original scale
```

---

## 4. Evaluation (fixed by the organizers)

The pipeline: **generate rooms → render with `plot.py` → InceptionV3 features →
metrics vs. real plans.**

| Metric | Tool | Goal |
|---|---|---|
| **FID** | torchmetrics `FrechetInceptionDistance` | distribution match (lower better) |
| **Density** | [clovaai/generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc) `compute_prdc(..., nearest_k=5)` | quality (higher better) |
| **Coverage** | same `compute_prdc` | diversity / mode coverage (higher better) |
| **Rendering** | [caspervanengelenburg/msd](https://github.com/caspervanengelenburg/msd) `plot.py` | consistent images for both real & generated |

> ⚠️ Render generated **and** real plans with the *same* `plot.py` script — otherwise
> render-style differences corrupt FID / density / coverage.

**This repo implements the full pipeline** (`src/eval/`), validated end-to-end:

```bash
# Compare your generated graph pickles against the real test set
python src/eval/run_eval.py --real <real_graph_out_dir> --fake <your_generated_dir> --n 500
```

- `src/msd_vendor/` — vendored MSD `plot.py` + `constants.py` (with a matplotlib≥3.9
  `get_cmap` compat shim). `plot_floor(G, ax)` renders a graph natively.
- `src/eval/render.py` — graph → fixed 512×512 uint8 RGB via the official `plot_floor`.
- `src/eval/metrics.py` — **one** InceptionV3 (torchmetrics') feeds both FID and prdc,
  so real & fake share identical preprocessing. FID uses float64 covariance.
- `src/eval/prdc.py` — vendored density/coverage (MIT).

**Sanity checks (passed):** `metrics(A, A)` → FID ≈ 0, density/coverage = 1.0.
**Real-vs-real reference** (test vs train, N=200): FID ≈ 67, density ≈ 0.97,
coverage ≈ 0.88 — roughly the best a model can reach at this sample size.

> ⚠️ Local eval is **self-consistent** (real & fake share our renderer). For the
> official leaderboard number, render with the organizers' exact `plot.py` settings.
> Keep N equal on both sides; FID is biased upward below a few thousand samples.

---

## 5. Pipeline overview

```
CSV (WKT polygons)
   │  per unit_id
   ▼
Rooms (entity_type='area')  ──buffer/union/buffer──►  Outline (Input)
   │ (Target)                                              │
   └──────────────►  MODEL: Outline ─► Rooms  ◄────────────┘
                              │
                     plot.py renders rooms to image
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
         FID (torchmetrics)        Density / Coverage (PRDC, k=5)
         vs. real plans            on InceptionV3 features
```

---

## 6. Repo structure

```
.
├── README.md            ← this file (single source of truth)
├── requirements.txt
├── data/                ← raw CSV (gitignored)
├── src/
│   ├── visualize.py     ← data exploration + (outline → rooms) plots
│   ├── msd_vendor/      ← vendored MSD plot.py + constants.py (rendering)
│   └── eval/            ← evaluation pipeline
│       ├── render.py    ← graph → image (official plot_floor)
│       ├── metrics.py   ← FID + density/coverage (shared InceptionV3)
│       ├── prdc.py      ← vendored precision/recall/density/coverage
│       └── run_eval.py  ← CLI: real vs generated → metrics
├── outputs/             ← rendered figures (gitignored)
└── notebooks/           ← scratch / experiments
```

---

## 7. Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Data (gitignored, not in repo): download from Kaggle into data/
#    data/mds_V2_5.372k.csv                  (polygon CSV)
#    data/modified-swiss-dwellings-v2/       (16 GB graph/image dataset; train/ test/)
#    Locally we use MSD = data/modified-swiss-dwellings-v2
# 2) Explore + visualize
python src/visualize.py                 # overview + grid of sample apartments
python src/visualize.py --unit-id 64314 # one specific apartment

# 3) Evaluate generated plans (graph pickles) against real ones
python src/eval/run_eval.py --real <real_dir> --fake <generated_dir> --n 500
```

---

## 8. The generative task (precise)

`graph_in` and `graph_out` are **node-aligned** (same node count every id, 0/4572
mismatches). The model gets, per id:
- `graph_in` — the access graph: nodes with `zoning_type`, edges with `connectivity`
  (`door`/`passage`/`entrance`). The "bubble diagram".
- `struct_in` — `(512,512,3)`: `ch0` = **wall mask** (0 = wall), `ch1`/`ch2` =
  world y/x coordinate per row/col (meters, same frame as `graph_out`).

It must predict, per node: `geometry` (room polygon), `room_type`, `centroid`.

**Regime (tested):** overlaying real `graph_out` polygons on the `struct_in` wall
mask shows room boundaries sit *on* the given walls — so the **walls are given**;
the task is segment-the-walls-into-rooms + label, not invent partitions.
`zoning_type → room_type` is grouped, not 1:1 (zoning 0→Bedroom, 3→Bathroom are
clean; 1 and 2 fan out).

## 9. Baseline v1 — retrieval

`src/model/baseline_retrieval.py`: for each test `graph_in`, retrieve the most
structurally similar train sample (feature = node count + zoning histogram +
connectivity counts) and emit its `graph_out`.

```bash
python src/model/baseline_retrieval.py --train <MSD>/train --test <MSD>/test --out outputs/generated --n 400
python src/eval/run_eval.py --real <MSD>/test/graph_out --fake outputs/generated --n 400
```

**Result (N=400):** FID **36.0** · density **0.91** · coverage **0.89** · prec 0.85 · rec 0.87.

This ≈ the real-vs-real ceiling (it outputs real plans), which closes the
submission→eval loop and shows the metrics are **distributional and partially
gameable** — any method emitting plausible real-shaped layouts scores well.

**Nuance on "walls are given":** `struct_in` carries the **load-bearing** structure
(outer walls + columns), *not* the thin partition walls between rooms. So the room
boundaries are **not** fully determined by the input — the partition walls have to be
**generated**. Two readings of v2 follow from this:
- if a clean room tiling can be polygonized straight from `ch0`, segment-and-label it;
- otherwise treat it as a generation problem: partition the envelope into one cell per
  graph node, where the cell boundaries *are* the generated partition walls.

## 10. Baseline v2 — graph-conditioned partition + labelling

`src/model/baseline_partition.py`: don't retrieve a whole plan — **generate geometry
per graph node**. The idea is "label every vertex of the graph": give each node a room
type *and* a room cell.

```
graph_in (+ envelope) ─partition→ one Voronoi cell per node   (src/model/partition.py)
                      ─label────→ room_type from zoning_type   (src/model/labeling.py)
                      ─assemble─→ graph_out + validate          (src/eval/validate.py)
                      ─[doors]──→ door point on shared wall      (place_doors, optional)
```

- **Partition (the "constraint solver"):** one seed per node, laid out by a spring
  layout of the adjacency graph (adjacent rooms attract → their cells share a wall),
  then Voronoi cells clipped to the envelope + a couple of Lloyd iterations. Cells tile
  the envelope by construction; their borders are the partition walls.
- **Labelling:** `zoning_type → room_type` learned as argmax `P(room_type|zoning_type)`
  from the node-aligned train pairs (handles the Zone2/Zone3 fan-out); falls back to the
  fixed MSD grouping with no train data.
- **Doors:** connectivity edges are carried over (topology preserved); `--doors` also
  places door geometry on shared walls (the scorer ignores it; downstream wants it).
- **No hidden failures:** every plan is checked against the renderer's contract
  (`src/eval/validate.py` — node `1` exists with `room_type`; every node has
  `geometry`/`room_type`/`centroid`; edges typed) before it is written; failing ids are
  **not** emitted (backfill them with retrieval).

```bash
# no dataset needed — verify the full pipeline on a synthetic apartment
python src/model/baseline_partition.py --selfcheck

# real data
python src/model/baseline_partition.py --test <MSD>/test --train <MSD>/train \
    --out outputs/generated_v2 --n 400 --doors
python src/eval/run_eval.py --real <MSD>/test/graph_out --fake outputs/generated_v2 --n 400
```

> ⚠️ `envelope_from_struct_in` (convex hull of structure pixels) is the one piece not
> yet validated against real `struct_in` — refine it once the 16 GB split is on disk.
> Until then, `--envelope-source graph_out` runs the pipeline against derived outlines.

## 11. Roadmap

- [x] Repo structure + README
- [x] Data visualization (`src/visualize.py`) → `outputs/samples_overview.png`
- [x] Download & inspect real CSV columns (confirmed: `roomtype`, walls/openings)
- [x] Evaluation pipeline (render → features → FID + density/coverage) — validated
- [x] Baseline v1 — retrieval (FID 36 @ N=400)
- [x] Baseline v2 — graph-conditioned partition + labelling (`baseline_partition.py`; self-check passes)
- [ ] Validate `envelope_from_struct_in` against real `struct_in`; run v2 vs retrieval at equal N
- [ ] Confirm official submission format vs Amine's slides
