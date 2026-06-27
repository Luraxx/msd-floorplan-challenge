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
- **File:** `mds_V2_5.372k.csv` (~5,372 apartments) — place it in `data/`
- **Format:** each row is a geometry stored as a **WKT** string; load with GeoPandas/Shapely.

| Column | Meaning |
|---|---|
| `geom` | Geometry (polygon) as WKT |
| `plan_id` | A whole **floor** (may contain several apartments) |
| `unit_id` | A single **apartment** = one training example |
| `entity_type` | Kind of geometry; `'area'` = a room |

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

```python
# Density & Coverage
from prdc import compute_prdc
m = compute_prdc(real_features=real_feats, fake_features=fake_feats, nearest_k=5)
# -> {'precision', 'recall', 'density', 'coverage'}

# FID
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048)
fid.update(real_imgs, real=True); fid.update(fake_imgs, real=False)
fid.compute()
```

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
│   └── visualize.py     ← data exploration + (outline → rooms) plots
├── outputs/             ← rendered figures (gitignored)
└── notebooks/           ← scratch / experiments
```

---

## 7. Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Download mds_V2_5.372k.csv from Kaggle into data/
# 2) Explore + visualize
python src/visualize.py                 # overview + grid of sample apartments
python src/visualize.py --unit-id 64314 # one specific apartment
```

---

## 8. Roadmap

- [x] Repo structure + README
- [x] Data visualization (`src/visualize.py`)
- [ ] Download & inspect real CSV columns (confirm room-type field)
- [ ] Evaluation pipeline (render → features → FID + density/coverage)
- [ ] Baseline model (outline → rooms)
- [ ] Improve & iterate
