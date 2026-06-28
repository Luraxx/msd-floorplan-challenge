# Corner Diffusion (Weg B / HouseDiffusion-style)

The learned realization of the "outer points + a denoiser that finds where they
actually lie" idea — and the version of the ensemble instinct that works (fuse a
*learned model*, not a pixel cloud). It is also the only one of our models to use
the **corner representation** that the MSD-challenge top baseline (HouseDiffusion)
wins with.

## Representation

Each room → its **4 outer corners** = the axis-aligned bounding box `(x0,y0,x1,y1)`
in the **building frame** (rotated so the building is axis-aligned), normalized to
[0,1]. MSD rooms are ~99% rectangular, so the box *is* the room. A round-trip test
(real plan → boxes → render) reproduces the plans almost exactly — the
representation is faithful. See [corner_diffusion_data.py](../src/model/corner_diffusion_data.py).

## Model

A Transformer that **denoises all room boxes jointly** (continuous diffusion,
x0-prediction — the trick that fixed our raster diffusion):

- one token per room: its noisy box + a room-type embedding + the diffusion time;
- **graph-relational attention**: each door/passage/entrance edge adds a learned
  per-head bias to the attention score, so connected rooms attend to — and end up
  beside — each other (this is what injects the access-graph topology);
- predicts the clean boxes; DDIM sampling from noise → boxes → tile the interior.

Trained from scratch on 4,382 real floors, 400 epochs on the AMD MI300X
(loss 0.27 → 0.064, no divergence). See [corner_diffusion_model.py](../src/model/corner_diffusion_model.py).

## Tiling the interior (the geometry fix)

Raw generated boxes leave gaps (they don't perfectly tile), which wrecks
density/coverage. So we **paint every interior pixel with the box that contains it**
(smallest-box-first + nearest-fill), exactly like the LLM pipeline — the rooms then
fill the real envelope with no gaps. This alone moved FID **138 → 96**.

## Result (n=321, R≤64)

| model | FID ↓ | density ↑ | coverage ↑ | adjacency ↑ |
|---|---|---|---|---|
| **Corner Diffusion** | **96.1** | **0.254** | **0.312** | **59 %** |
| Raster diffusion | 103.1 | 0.10 | 0.12 | ~41 % |
| Refinement | 102.1 | 0.09 | 0.10 | ~41 % |
| U-Net | 145.7 | — | — | ~41 % |
| Rectilinear (rule-based) | 80.9 | 0.16 | 0.24 | 41 % |

**The best LEARNED, generative model we have** — beats raster diffusion on FID *and*
density/coverage *and* adjacency. It is genuinely generative (samples diverse plans
per envelope) and graph-faithful. Rule-based Rectilinear still wins raw FID, but the
corner model is the strongest *generative* approach and the one with headroom.

## Tiling vs adjacency trade-off

Painting the interior boosts FID/coverage but costs adjacency (raw boxes were 71%;
tiled is 59%) — the nearest-fill reassigns boundary pixels and breaks some shared
walls. Both modes are useful; tiled is the better all-rounder.

## Limitations / next steps

- **Assembly success 80 %** (321/400) — some rooms get no pixels (empty cell);
  R>64 floors are skipped. A more robust empty-cell fallback + larger R_MAX would
  lift coverage of the test set.
- **L-shaped rooms**: v1 is box-only; corridors/living rooms are sometimes L-shaped
  (the real corner representation would use >4 corners per room).
- **Envelope shape**: conditioned only via the normalization + post-clip; a CNN
  envelope encoder (cross-attention) would sharpen the fit.
- **Adjacency loss**: training is pure box-MSE; adding a differentiable
  shared-wall / overlap term would push adjacency back toward the raw-box 71%.
