// Model registry — the single source of truth documenting every model we build.
// Add a new entry here whenever a model is created; the /models page renders it.

export type ModelDoc = {
  id: string;
  name: string;
  family: "retrieval" | "generative" | "partition";
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
