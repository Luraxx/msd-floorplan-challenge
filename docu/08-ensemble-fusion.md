# Ensemble Fusion — the "probability cloud" experiment

Idea (proposed): overlay every graph-conditioned model's room recommendation, build
a per-pixel probability cloud of where the room boundaries (outer points) are, and
let a vote / diffusion resolve the true positions. Each model votes where it is
strong (LLM = topology, Rectilinear = areas, Partition = density). All our
graph-conditioned models preserve `graph_in` node-ids, so room *n* is the same room
in every model — they can be stacked.

Implemented in [src/model/ensemble_fuse.py](../src/model/ensemble_fuse.py): rasterise
each model's polygons to a node-label image, per-pixel **majority vote** across
models → consensus → vectorise.

## Result — naive pixel fusion FAILS

On the 36 LLM floors, fusing {LLM, Rectilinear, Partition}:

- **Model agreement: 37.5 %** (mean max-vote / K over interior pixels). With K=3 that
  is barely above the 33 % floor where all three disagree — the models almost never
  place a pixel's room the same way.
- **Only 7 / 36 produced a valid plan** (the rest collapse: too few pixels win a
  majority for some nodes → empty cells).
- On those 7, **adjacency 39 %** vs **LLM 67 %** / Rect 42 % — the fusion is *worse*
  than its own ingredients. Visually: rooms scattered, fragmented, access-graph
  edges crisscrossing (positions mutually inconsistent).

## Why — the problem is under-determined

Graph + envelope admit MANY valid layouts. Each model commits to a *different* valid
one (the LLM reasons, Rectilinear slices, Partition spring-relaxes). Averaging two
*different valid* layouts pixel-wise yields an *invalid* one — mush.

This is a general principle: **ensemble averaging helps regression/classification,
but not structured generation with multiple valid modes.** You cannot overlay two
different floor plans.

## The corrected path — fuse PROPERTIES, not pixels

The underlying instinct (room corners + a denoiser that finds the true points) is
exactly **HouseDiffusion**, the MSD-challenge top baseline. The fix is *what* gets
combined:

- **A. Per-property fusion** — take each model's *strength*, not its pixels:
  LLM **topology** (which rooms touch) + Rectilinear **areas** (how big) → a solver
  places corners satisfying both. This is the LLM + learned-areas combination
  already built ([docu/07](07-llm-layout.md)).
- **B. Corner diffusion (HouseDiffusion)** — a *learned* model denoises room corners
  from the access graph; other models can *condition* it, not be argmax-ed. (~2–3 d
  training.)
- **C. Best-of-ensemble selection** — pick the single best model per floor (works
  under multi-modality; needs a no-ground-truth critic, e.g. LayoutGKN). (~1 d.)

Pixel/point averaging is only valid when the models share a mode — these do not.
The negative result is kept (`ensemble_fuse.py`); no model is registered.
