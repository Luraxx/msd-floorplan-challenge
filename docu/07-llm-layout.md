# LLM-as-Layout-Solver (foundation-model approach)

Every geometric generator we built scatters the given access graph: our best,
**Rectilinear, honours only ~41 % of door edges** as shared walls (real = 100 %).
The bottleneck is *topological reasoning* — read a corridor-centred access graph
and place rooms so connected pairs touch. That is exactly what a large language
model is good at. So we tried the obvious thing nobody on our list had: **hand the
access graph to an LLM (Claude) and let it lay out the rooms.**

This is a **combination of models**, exactly the direction worth pushing:

| stage | model | job |
|---|---|---|
| reason + place | **LLM (Claude)** | read the access graph, output non-overlapping room rectangles honouring every adjacency |
| realise geometry | **our pipeline** | rotate into the building frame, clip rectangles to the real (rotated, L-shaped) interior by painting pixels, vectorise, validate, render, score |

Pipeline: `llm_layout.py specs` → a Workflow agent per floor (`scripts/...workflow`)
→ `llm_layout.py build`. See [src/model/llm_layout.py](../src/model/llm_layout.py).

## Result 1 — the LLM solves topology that geometry could not

Single hardest metric, the one all our generators fail and retrieval gets for free:
fraction of door/passage/entrance edges realised as a shared wall.

| | door-adjacency faithfulness |
|---|---|
| Real plans | 100 % |
| **LLM (Claude)** | **71.8 %** |
| Rectilinear (*identical* rectangle geometry) | 40.1 % |

**The LLM nearly doubles adjacency faithfulness at the same geometric style.** On a
small single-apartment floor it hits a perfect 100 % / 0 overlap / 100 % coverage;
the 71.8 % average is dragged down by the 50–90-room floors, where even the LLM
struggles to tile everything consistently. (n = 36; see caveat below.)

## Result 2 — but it does NOT yet win the scored metrics

Same 36 floors, FID / density / coverage on the official renderer:

| model | FID ↓ | density ↑ | coverage ↑ | adjacency ↑ |
|---|---|---|---|---|
| Rectilinear | **137.2** | **0.539** | **0.750** | 0.401 |
| LLM (Claude) | 151.8 | 0.400 | 0.611 | **0.718** |

The LLM **gets the topology right but the proportions wrong**: it *guesses* room
areas ("Bedroom large, Bathroom small"), while Rectilinear uses **learned per-type
area fractions** matched to the real distribution. FID/density/coverage are
dominated by room-size/shape realism, so the learned areas win them — even though
the layout is structurally less faithful. (This mirrors the MSD-benchmark paper:
methods that win topology can lose pixel realism.)

> ⚠️ Caveat: this eval is **n = 36**, not the n≈800 leaderboard. We hit the LLM
> session limit at 36/80 floors. FID at n=36 is noisy and **not comparable** to the
> leaderboard numbers (Rectilinear is 80.9 at n=800 but 137 here). Only the
> same-n LLM-vs-Rectilinear comparison above is meaningful.

## The combination that should win: LLM topology + learned areas

The two models are complementary — the LLM owns adjacency, Rectilinear owns
proportions. The fix is to give the LLM our learned numbers: each room's spec now
carries a **`target_area`** (its learned per-type fraction of the floor), and the
prompt instructs the LLM to size rooms to those targets. That keeps the
adjacency win while matching the real area distribution that drives FID. Ready to
run on the full 80+ floors when the session limit resets.

## Why this matters

It is the first of our approaches that **reasons about the problem** instead of
pattern-matching geometry, and the first to use a foundation model. It validates a
broader bet: decompose the task and route each part to the model that is best at
it — LLM for combinatorial/topological reasoning, learned statistics for metric
realism, our renderer/validator for execution.
