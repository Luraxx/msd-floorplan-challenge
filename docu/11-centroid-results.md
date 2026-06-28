# Centroid Diffusion — v1 results (outline-only)

Built per the plan in [10-centroid-diffusion-plan.md](10-centroid-diffusion-plan.md):
outline-only, per-apartment, Voronoi reconstruction. It works.

## Result (n=1890 test apartments) — two registered models

| metric | **centroid-v1** (no count head) | **centroid-v2** (count head) |
|---|---|---|
| FID | 58.0 | **44.8** |
| Density | 0.284 | 0.265 |
| Coverage | 0.269 | **0.299** |
| rooms/apt (median) | 10 | **8** (= real) |

Both are registered on the site as separate models so the count-head ablation is
visible. (FID is computed strictly on the held-out test split, plan_id % 10 == 0;
train ∩ test = 0, no leakage.)

**centroid-v2 — global room-count head.** v1's per-node validity flag was near-random
(BCE 0.64): with an exchangeable node set and symmetric noise, "which node is real"
is ill-posed, so the model over-generated (~10 rooms vs 8 real). Fix: predict the
room count **K globally from the outline encoding** (pooled) and denoise *exactly K*
centroids — no validity needed. The count head learned almost perfectly (MSE
5.9 → 0.21 ≈ 0.5 rooms). Room count now matches the real distribution (median 8),
which alone cut **FID 58.0 → 44.8** and lifted coverage 0.269 → 0.299.

The model is given **only the apartment outline** and generates the room count,
centroids, types, and adjacency — then Voronoi reconstruction tiles the outline.
Generated apartments are coherent, space-filling, with a full access graph + an
entrance edge, and adapt to the outline shape.

> ⚠️ **Separate track.** This FID is on the **outline-only, per-apartment** setting
> with its own real reference (1890 real apartments). It is **NOT** comparable to the
> per-floor, graph-conditioned leaderboard (Rectilinear 80.9 etc.): apartments are
> smaller (~8 rooms vs ~24) and the reference set differs, so the number is naturally
> lower. The comparison that matters is against future versions of this same model.

## What worked

- **Box/centroid representation + Voronoi** tiles the outline cleanly (round-trip:
  Voronoi from real centroids ≈ the real plan).
- **x0-diffusion on centroids** with outline cross-attention places rooms sensibly
  (position loss 0.046); the apartment-scale task (median 8 rooms) is tractable.
- **18,580 apartments** from the CSV is a big, clean training set.

## What is weak (→ v2)

- **Room-type head (CE ≈ 1.65)** — types are hard to infer from position alone;
  apartments over-predict some types (lots of kitchen/dining orange).
- **Validity / room count** — with an exchangeable node set and symmetric noise,
  per-node validity is ill-posed (BCE ≈ 0.64, near random); the count still lands
  near the mean (~10 vs real 8) because the outline size carries it, but it is not
  precise. A global count head off the outline encoder would be cleaner.
- **Voronoi rooms are blobby** vs the real rectangles — the **separator algorithm
  (v2)** is the geometry upgrade, and **Stage-2 learned edges** would replace the
  geometry-derived adjacency.

## Bugs fixed on the way

- Weighted-Voronoi grid only covered 14% of the outline → reverted to plain
  `_voronoi_cells` (which tiles exactly). Weighted is a careful v2 upgrade.
- The MSD renderer hardcodes `G.nodes[1]`; generated node-ids had gaps (validity
  filtering) → re-index to contiguous `0..K-1`.
