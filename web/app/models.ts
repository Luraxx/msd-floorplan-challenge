// Model registry — the single source of truth documenting every model we build.
// Add a new entry here whenever a model is created; the /models page renders it.

export type ModelDoc = {
  id: string;
  name: string;
  family: "retrieval" | "generative" | "partition" | "llm";
  status: "baseline" | "trained" | "experimental" | "planned";
  generator: boolean; // true = can power the Studio (draw -> generate)
  date: string;
  summary: string;
  approach: string;
  config: { label: string; value: string }[];
  metrics?: { fid?: number; density?: number; coverage?: number; note?: string };
  strengths: string[];
  limitations: string[];
};

export const MODELS: ModelDoc[] = [
  {
    id: "llm-v1",
    name: "LLM Layout · Claude (foundation model)",
    family: "llm",
    status: "experimental",
    generator: false,
    date: "2026-06-28",
    summary:
      "The first approach that REASONS about the problem instead of pattern-matching geometry. Every geometric generator scatters the given access graph — our best (Rectilinear) realises only 41% of door edges as shared walls. So we hand the access graph to an LLM (Claude) and let it place the rooms. It nearly DOUBLES door-adjacency faithfulness (41% → 72%) at the same rectangle geometry — a combination of a foundation model for topological reasoning + our pipeline for geometry.",
    approach:
      "Per floor we build a compact spec (building-frame bbox + rooms with types + required adjacencies + apartments). A Workflow agent (Claude) turns it into non-overlapping room rectangles honouring every adjacency, with corridors as spines. Our pipeline then rotates into the building frame, clips the rectangles to the real (rotated, L-shaped) interior by painting pixels (smallest-rectangle-wins + non-empty-cell guarantee), vectorises, validates and renders.",
    config: [
      { label: "Reasoner", value: "Claude (LLM, no training)" },
      { label: "Input", value: "access graph + envelope bbox" },
      { label: "Geometry", value: "pixel-clip to real interior" },
      { label: "Eval", value: "n=36 (LLM session limit)" },
    ],
    metrics: { fid: 151.8, density: 0.40, coverage: 0.61, note: "n=36, NOT comparable to the n=800 leaderboard (Rectilinear is 137 at this n=36); door-adjacency 72% vs Rectilinear 40%" },
    strengths: [
      "Nearly doubles door-adjacency faithfulness (72% vs 40%) — solves the topology all geometric models fail",
      "Reasons about the access graph (corridor-centred stars) instead of pattern-matching",
      "100% adjacency / 0 overlap / 100% coverage on small single-apartment floors",
      "Zero training — a foundation model + our geometry pipeline (a true model combination)",
    ],
    limitations: [
      "Guesses room AREAS → loses FID/density/coverage to Rectilinear (the combination fix: feed learned per-type target areas — built, ready to run)",
      "Struggles to tile the largest 50–90-room floors fully consistently",
      "One LLM call per floor → rate-limited; eval is partial (36/80)",
    ],
  },
  {
    id: "diffusion-v1",
    name: "Conditional raster diffusion",
    family: "generative",
    status: "trained",
    generator: false,
    date: "2026-06-28",
    summary:
      "A working, genuinely generative diffusion model — DIVERSE plausible plans for the same footprint (unlike the deterministic U-Net). GSDiff's junction-graph diffusion couldn't close plans; diffusing the room-type RASTER conditioned on the envelope always closes and works reliably.",
    approach:
      "Diffuse the 10-channel one-hot room-type map (in [-1,1]) at 64 px, conditioned on the interior mask. A DDPM U-Net predicts x0 (clean map) + a cross-entropy term on the room classes (predicting noise gave a degenerate all-uniform sample). DDIM sampling → argmax → vectorize → graph_out.",
    config: [
      { label: "Type", value: "DDPM raster diffusion (x0-pred)" },
      { label: "Conditioning", value: "building envelope" },
      { label: "Resolution", value: "64 px" },
      { label: "Sampler", value: "DDIM, 100 steps" },
      { label: "Hardware", value: "AMD MI300X" },
    ],
    metrics: { fid: 103.1, density: 0.100, coverage: 0.115, note: "n=800; diverse samples per envelope" },
    strengths: [
      "Genuinely generative — diverse plans per envelope (every other model here is deterministic)",
      "Best density / coverage of the learned generators",
      "Closes reliably (raster → vectorize), unlike the GSDiff graph approach",
    ],
    limitations: [
      "64 px → jagged geometry; FID still behind the rule-based Rectilinear (80.9)",
      "Background-heavy (unweighted classes) → somewhat sparse rooms",
      "128 px + class-weighted loss should push it further (next step)",
    ],
  },
  {
    id: "refine-v1",
    name: "Refinement (U-Net → clean)",
    family: "generative",
    status: "trained",
    generator: false,
    date: "2026-06-27",
    summary:
      "A hybrid that keeps the U-Net's learned room distribution + sizes but cleans the geometry: a second U-Net learns to map the first U-Net's rough, blobby layout to the real clean layout. Improves the raw U-Net's FID by ~44 points.",
    approach:
      "Run the trained U-Net → argmax room-type map; feed it (one-hot) + the building interior to a refinement U-Net trained against the real rasterized graph_out (weighted cross-entropy); vectorize the cleaned map to a graph_out. Reliable closure (raster → vectorize), no graph-closure problem.",
    config: [
      { label: "Type", value: "learned refinement of the U-Net" },
      { label: "Input", value: "U-Net argmax map + envelope" },
      { label: "Resolution", value: "128 px" },
      { label: "Test plans", value: "800" },
      { label: "Hardware", value: "AMD MI300X" },
    ],
    metrics: { fid: 102.1, density: 0.092, coverage: 0.095, note: "n=800; raw U-Net was 145.7" },
    strengths: [
      "Cleans the U-Net's blobby output — +44 FID over the raw U-Net (145.7 → 102.1)",
      "Keeps the U-Net's learned placement + room sizes",
      "Closes reliably (unlike the GSDiff graph approach)",
    ],
    limitations: [
      "Still raster-based (128 px) — not perfectly rectilinear; behind the rule-based Rectilinear (80.9)",
      "Inherits the U-Net's mistakes where its layout is wrong",
      "A rectilinear snap on top could push it further (next step)",
    ],
  },
  {
    id: "gsdiff-v4",
    name: "GSDiff structural-graph diffusion",
    family: "generative",
    status: "experimental",
    generator: false,
    date: "2026-06-27",
    summary:
      "A from-scratch take on the GSDiff paper: model the plan as a graph of wall JUNCTIONS + wall SEGMENTS (not rooms), generated by a Transformer diffusion model conditioned on the building envelope. The representation is validated and the pipeline trains end-to-end — but generation does not yet close into rooms (the hard core of GSDiff). Documented as work-in-progress.",
    approach:
      "Extract wall-junction structural graphs from MSD room polygons (each junction carries a 9-way room-semantics multi-hot + a balcony flag; balcony boundaries are treated as walls with b=1). A Transformer DDPM denoises the padded junction set while CROSS-ATTENDING to a CNN encoding of the building envelope; an MLP predicts the wall segments; rooms = polygonized minimal loops. Trained with noise-MSE + a grid-snap alignment loss on the MI300X.",
    config: [
      { label: "Representation", value: "wall junctions + segments" },
      { label: "Conditioning", value: "building envelope (CNN + cross-attention)" },
      { label: "Node semantics", value: "9 MSD room types + balcony flag" },
      { label: "Train graphs", value: "3,696" },
      { label: "Hardware", value: "AMD MI300X" },
    ],
    strengths: [
      "Validated the representation: MSD → wall graph → rooms round-trips faithfully (the part GSDiff's authors called hard for MSD)",
      "Envelope conditioning works — generated junctions land inside the given footprint",
      "True vector-graph generation with our room types + the balcony rule",
    ],
    limitations: [
      "Generated graphs don't yet close into rooms (~0–1 vs ~30) — no usable plans yet",
      "Training diverged late (the alignment loss destabilized); needs grad-clip / LR-decay / best-checkpoint",
      "Needs GSDiff's real mixed-base alignment + edge-perception trick to make walls close",
    ],
  },
  {
    id: "rect-v1",
    name: "Rectilinear (wall-aligned)",
    family: "partition",
    status: "trained",
    generator: false,
    date: "2026-06-27",
    summary:
      "Rule-based generator that exploits the strongest data fact: real rooms are rectangular (median rectangularity 0.99). It slices the building interior into rectangular rooms ALIGNED to the longest outer wall, sized by learned per-type areas, with balconies pushed to the facade. Best FID of any generator so far.",
    approach:
      "Trace the concave interior → estimate the building axis (longest edge of its min-rotated-rectangle) → recursively split the interior BY PIXEL-AREA along that axis into one rectangle per graph node (areas = learned per-type fractions; balconies ordered to the boundary) → leftover concave bits fall to the covering band → label from zoning.",
    config: [
      { label: "Type", value: "rule-based rectilinear partition" },
      { label: "Key rule", value: "slice parallel to the longest outer wall" },
      { label: "Test plans", value: "800" },
      { label: "Compute", value: "CPU, ~seconds/plan, no training" },
    ],
    metrics: { fid: 80.9, density: 0.16, coverage: 0.24, note: "n=800; wall-aligned (axis-aligned was 100.0)" },
    strengths: [
      "Best generator FID (80.9) — rectangular rooms read as real",
      "Wall-alignment alone cut FID from 100.0 → 80.9",
      "Encodes mined rules: per-type areas, balconies-outside, rooms rectangular",
      "Cheap, CPU-only, no training",
    ],
    limitations: [
      "Lower density/coverage than the Voronoi partition (0.16/0.24 vs 0.28/0.27)",
      "Needs an input access graph — not a bare-sketch Studio generator",
      "Slice-and-dice gives strict bands; real plans nest rooms more freely",
    ],
  },
  {
    id: "unet-graph-v1",
    name: "Graph-informed U-Net",
    family: "generative",
    status: "trained",
    generator: true,
    date: "2026-06-27",
    summary:
      "A true conditional generator: predicts a per-pixel room-type map from the wall structure plus a 12-d access-graph descriptor, then vectorizes it into a graph_out. This is the model that powers the Studio.",
    approach:
      "3-channel input (free mask + normalized col/row coords) → U-Net (base=24) with the graph descriptor projected into the bottleneck → per-pixel softmax over 10 classes → argmax → connected-component vectorization → MST access graph.",
    config: [
      { label: "Backbone", value: "U-Net, base=24 (~1.1M params)" },
      { label: "Resolution", value: "256 px" },
      { label: "Epochs", value: "100" },
      { label: "Batch", value: "32" },
      { label: "Train / test", value: "4572 / 800" },
      { label: "Hardware", value: "1× AMD MI300X (ROCm)" },
      { label: "Final loss", value: "0.252" },
    ],
    metrics: { fid: 145.7, density: 0.063, coverage: 0.056 },
    strengths: [
      "Genuinely generative — works on hand-drawn structures, not just dataset retrieval",
      "Honest, end-to-end conditional generation",
      "Trains in minutes on the MI300X",
    ],
    limitations: [
      "Distributionally behind retrieval on FID (vectorized segmentation reads jaggier)",
      "Under-capacity for the GPU (uses ~1.4% of HBM) — the main improvement lever",
      "Same-type adjacent rooms merge in vectorization (~65% node-recovery ceiling)",
    ],
  },
  {
    id: "retrieval-v2",
    name: "Structure-aware retrieval",
    family: "retrieval",
    status: "baseline",
    generator: false,
    date: "2026-06-27",
    summary:
      "Retrieves the nearest real training plan by a structure- and graph-aware descriptor. Strong distributional scores, but copies rather than generates.",
    approach:
      "Encode each plan's structure + access graph into a descriptor; for a query, return the closest real graph_out from the train split.",
    config: [
      { label: "Type", value: "Non-parametric retrieval" },
      { label: "Index", value: "Train split (4572)" },
      { label: "Compute", value: "CPU, no training" },
    ],
    metrics: { fid: 34.1, density: 0.87, coverage: 0.91 },
    strengths: ["Near the real-vs-real FID ceiling", "Instant, no training", "High density/coverage"],
    limitations: ["Not generative — returns existing plans", "Cannot honor a novel drawn structure"],
  },
  {
    id: "retrieval-v1",
    name: "Retrieval baseline",
    family: "retrieval",
    status: "baseline",
    generator: false,
    date: "2026-06-26",
    summary: "First retrieval baseline on a simpler graph descriptor — the original reference point.",
    approach: "Nearest-neighbour over a 12-d access-graph descriptor.",
    config: [
      { label: "Type", value: "Non-parametric retrieval" },
      { label: "Compute", value: "CPU, no training" },
    ],
    metrics: { fid: 36.0, density: 0.91, coverage: 0.89 },
    strengths: ["Simple, strong reference", "Instant"],
    limitations: ["Not generative"],
  },
  {
    id: "partition-concave",
    name: "Graph-conditioned partition (concave)",
    family: "partition",
    status: "trained",
    generator: false,
    date: "2026-06-27",
    summary:
      "A near-non-learned generator that beats the U-Net at scale: partitions the real building footprint into one Voronoi cell per graph node, then labels each cell's room type from its zoning. The concave-envelope upgrade traces the true (notched, elongated) footprint instead of a convex blob.",
    approach:
      "Trace the enclosed building interior from struct_in (flood-fill → concave footprint) → one seed per access-graph node (spring layout + 2 Lloyd iterations) → clip Voronoi cells to the footprint → label room_type = argmax P(room_type | zoning_type) learned on train.",
    config: [
      { label: "Type", value: "Voronoi partition + learned labeling" },
      { label: "Envelope", value: "concave interior (vs convex hull)" },
      { label: "Test plans", value: "800" },
      { label: "Compute", value: "CPU, ~seconds/plan, no training" },
    ],
    metrics: { fid: 81.5, density: 0.28, coverage: 0.27, note: "n=800; convex envelope = FID 87.5" },
    strengths: [
      "Best generator so far — FID 81.5 vs the U-Net's 145.7",
      "Exactly one room per graph node — no over/under-segmentation",
      "Concave footprint lifts FID 87.5 → 81.5 over the convex hull",
      "Cheap, CPU-only, no training",
    ],
    limitations: [
      "Needs an input access graph (zoning + connectivity) — can't run from a bare canvas sketch",
      "Voronoi cells are blobby vs. real rectangular rooms",
      "argmax labeling collapses room-type diversity",
    ],
  },
];
