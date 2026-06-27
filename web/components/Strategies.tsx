"use client";

import { useState } from "react";

type Status = "done" | "next" | "stretch";

type Idea = {
  n: number;
  title: string;
  status: Status;
  best?: boolean;
  output: string;
  effort: string;
  fid: string;
  fit: number;
  summary: string;
  points: string[];
};

const IDEAS: Idea[] = [
  {
    n: 1, title: "Conditioned retrieval", status: "done", best: true,
    output: "real graph_out", effort: "Low (done)", fid: "36.0", fit: 90,
    summary: "Match each test graph to the nearest train apartment and emit its real layout. Current best model — shown above.",
    points: ["Always emits a valid, real-shaped plan", "Scores near the real-vs-real ceiling", "Ignores the given walls — its main weakness"],
  },
  {
    n: 2, title: "Wall-mask segmentation → labelled rooms", status: "next",
    output: "true polygons", effort: "Medium", fid: "target < 36", fit: 95,
    summary: "The honest conditioning lever. The walls are GIVEN in struct_in ch0 — polygonize the wall mask into room faces, map pixels to world coords via ch1/ch2, label faces from the access graph, build graph_out. Geometry then matches each specific input.",
    points: ["Uses the actual input structure, not just retrieval", "Output is true tiling polygons → matches graph_out", "Mostly classical CV + labelling, little training", "Must handle doorway gaps when polygonizing"],
  },
  {
    n: 3, title: "HouseDiffusion / Modified HouseDiffusion", status: "stretch",
    output: "vector polygons", effort: "High (training)", fid: "best topology", fit: 88,
    summary: "Diffusion on polygon corners; the official MSD diffusion baseline (MHD) adds wall cross-attention + a GAT for room types. Output is exactly graph_out and topology is preserved best.",
    points: ["Output format is exactly right (vector polygons)", "Best topology / connectivity retention", "Heavy to train; reported geometry on MSD is weak", "Start from aminshabani/house_diffusion + arXiv 2312.03938"],
  },
  {
    n: 4, title: "Graph2Plan (GNN + CNN + retrieval)", status: "stretch",
    output: "axis-aligned boxes", effort: "Medium", fid: "decent", fit: 70,
    summary: "Boundary + layout graph → raster, then refined room bounding boxes with type labels. Faster to stand up than diffusion, but emits boxes, not arbitrary tiling polygons.",
    points: ["Boundary + graph conditioned (matches inputs)", "Faster than a from-scratch diffusion model", "Output is boxes + raster — lower geometric fidelity", "Trained on RPLAN, would need adapting to MSD"],
  },
  {
    n: 5, title: "Graph-informed U-Net (segmentation)", status: "stretch",
    output: "raster image", effort: "Medium", fid: "best pixels", fit: 60,
    summary: "The official MSD segmentation baseline: U-Net + GCN at the bottleneck, optional Segment Anything pre-processing. Best pixel accuracy (MIoU 42.4) but raster output yields no clean polygon/graph.",
    points: ["Best pixel overlap of the MSD baselines", "Since scoring is image-based, raster may still score well", "No reliable polygon/graph extraction", "Doesn't directly satisfy the graph_out format"],
  },
];

function MiniViz({ n }: { n: number }) {
  const arrow = (
    <defs>
      <marker id={`mv-${n}`} markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#94a3b8" />
      </marker>
    </defs>
  );
  const A = ({ x }: { x: number }) => (
    <line x1={x} y1={46} x2={x + 26} y2={46} stroke="#94a3b8" strokeWidth={1.4} markerEnd={`url(#mv-${n})`} />
  );
  const cls = "h-[88px] w-full";
  const vb = "0 0 340 92";

  if (n === 1)
    return (
      <svg viewBox={vb} className={cls} role="img" aria-label="Query graph matched to a similar plan, emit it">
        {arrow}
        <circle cx="22" cy="34" r="6" fill="#cbd5e1" /><circle cx="44" cy="50" r="6" fill="#cbd5e1" /><circle cx="26" cy="62" r="6" fill="#cbd5e1" />
        <line x1="22" y1="34" x2="44" y2="50" stroke="#cbd5e1" strokeWidth="2" /><line x1="44" y1="50" x2="26" y2="62" stroke="#cbd5e1" strokeWidth="2" />
        <text x="34" y="84" textAnchor="middle" fontSize="11" fill="#64748b">query</text>
        <A x={64} />
        <rect x="98" y="28" width="58" height="36" rx="8" fill="#eef2ff" stroke="#c7d2fe" /><text x="127" y="51" textAnchor="middle" fontSize="16" fill="#4338ca">≈</text>
        <text x="127" y="84" textAnchor="middle" fontSize="11" fill="#64748b">match</text>
        <A x={160} />
        <g transform="translate(196,22)">
          <rect width="48" height="48" rx="4" fill="#ecfdf5" stroke="#a7f3d0" />
          <rect x="4" y="4" width="20" height="26" fill="#6ee7b7" /><rect x="26" y="4" width="18" height="14" fill="#fcd34d" /><rect x="26" y="20" width="18" height="14" fill="#93c5fd" /><rect x="4" y="32" width="40" height="12" fill="#c4b5fd" />
        </g>
        <text x="220" y="84" textAnchor="middle" fontSize="11" fill="#047857">real plan</text>
      </svg>
    );

  if (n === 2)
    return (
      <svg viewBox={vb} className={cls} role="img" aria-label="Wall mask polygonized into faces, then labelled rooms">
        {arrow}
        <g transform="translate(8,22)">
          <rect width="48" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
          <line x1="0" y1="20" x2="48" y2="20" stroke="#334155" strokeWidth="2" /><line x1="24" y1="20" x2="24" y2="48" stroke="#334155" strokeWidth="2" /><line x1="24" y1="8" x2="24" y2="14" stroke="#334155" strokeWidth="2" />
        </g>
        <text x="32" y="84" textAnchor="middle" fontSize="11" fill="#64748b">walls</text>
        <A x={64} />
        <g transform="translate(98,22)">
          <rect width="48" height="48" rx="4" fill="#f1f5f9" stroke="#cbd5e1" />
          <rect x="2" y="2" width="20" height="16" fill="none" stroke="#6366f1" strokeWidth="1.5" /><rect x="24" y="2" width="22" height="16" fill="none" stroke="#6366f1" strokeWidth="1.5" /><rect x="2" y="20" width="44" height="26" fill="none" stroke="#6366f1" strokeWidth="1.5" />
        </g>
        <text x="122" y="84" textAnchor="middle" fontSize="11" fill="#64748b">faces</text>
        <A x={160} />
        <g transform="translate(196,22)">
          <rect width="48" height="48" rx="4" fill="#ecfdf5" stroke="#a7f3d0" />
          <rect x="2" y="2" width="20" height="16" fill="#93c5fd" /><rect x="24" y="2" width="22" height="16" fill="#fcd34d" /><rect x="2" y="20" width="44" height="26" fill="#6ee7b7" />
        </g>
        <text x="220" y="84" textAnchor="middle" fontSize="11" fill="#047857">labelled</text>
      </svg>
    );

  if (n === 3)
    return (
      <svg viewBox={vb} className={cls} role="img" aria-label="Noise denoised step by step into a polygon">
        {arrow}
        <g transform="translate(10,24)">
          <rect width="46" height="46" rx="4" fill="#faf5ff" stroke="#e9d5ff" />
          {[14, 22, 30, 18, 34, 26].map((cx, i) => (
            <circle key={i} cx={cx} cy={10 + ((i * 9) % 30)} r="2.4" fill="#c084fc" />
          ))}
        </g>
        <text x="33" y="84" textAnchor="middle" fontSize="11" fill="#64748b">noise</text>
        <A x={62} />
        <text x="96" y="50" textAnchor="middle" fontSize="11" fill="#7c3aed">denoise</text>
        <text x="96" y="64" textAnchor="middle" fontSize="11" fill="#7c3aed">×T</text>
        <A x={118} />
        <g transform="translate(152,24)">
          <rect width="46" height="46" rx="4" fill="#faf5ff" stroke="#e9d5ff" />
          <polygon points="6,8 40,8 40,40 22,40 22,26 6,26" fill="none" stroke="#9333ea" strokeWidth="2" />
        </g>
        <text x="175" y="84" textAnchor="middle" fontSize="11" fill="#7c3aed">polygon</text>
        <A x={206} />
        <g transform="translate(244,24)">
          <rect width="46" height="46" rx="4" fill="#ecfdf5" stroke="#a7f3d0" />
          <rect x="4" y="4" width="38" height="38" fill="#6ee7b7" opacity="0.6" />
        </g>
        <text x="267" y="84" textAnchor="middle" fontSize="11" fill="#047857">vector</text>
      </svg>
    );

  if (n === 4)
    return (
      <svg viewBox={vb} className={cls} role="img" aria-label="Graph plus boundary turned into room boxes">
        {arrow}
        <g transform="translate(10,22)">
          <rect width="48" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
          <circle cx="14" cy="16" r="5" fill="#a5b4fc" /><circle cx="34" cy="14" r="5" fill="#a5b4fc" /><circle cx="24" cy="34" r="5" fill="#a5b4fc" />
          <line x1="14" y1="16" x2="34" y2="14" stroke="#a5b4fc" strokeWidth="2" /><line x1="34" y1="14" x2="24" y2="34" stroke="#a5b4fc" strokeWidth="2" />
        </g>
        <text x="34" y="84" textAnchor="middle" fontSize="11" fill="#64748b">graph</text>
        <A x={64} />
        <rect x="98" y="30" width="58" height="32" rx="8" fill="#eef2ff" stroke="#c7d2fe" /><text x="127" y="50" textAnchor="middle" fontSize="11" fill="#4338ca">GNN+CNN</text>
        <text x="127" y="84" textAnchor="middle" fontSize="11" fill="#64748b">retrieve</text>
        <A x={160} />
        <g transform="translate(196,22)">
          <rect width="48" height="48" rx="4" fill="#f1f5f9" stroke="#cbd5e1" />
          <rect x="4" y="4" width="22" height="20" fill="none" stroke="#f59e0b" strokeWidth="2" /><rect x="28" y="4" width="16" height="20" fill="none" stroke="#f59e0b" strokeWidth="2" /><rect x="4" y="26" width="40" height="18" fill="none" stroke="#f59e0b" strokeWidth="2" />
        </g>
        <text x="220" y="84" textAnchor="middle" fontSize="11" fill="#b45309">boxes</text>
      </svg>
    );

  return (
    <svg viewBox={vb} className={cls} role="img" aria-label="Image through a U-Net into a segmentation">
      {arrow}
      <g transform="translate(10,22)">
        <rect width="48" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
        <line x1="0" y1="22" x2="48" y2="22" stroke="#334155" strokeWidth="1.5" /><line x1="22" y1="22" x2="22" y2="48" stroke="#334155" strokeWidth="1.5" />
      </g>
      <text x="34" y="84" textAnchor="middle" fontSize="11" fill="#64748b">image</text>
      <A x={64} />
      <path d="M100,28 L108,28 L114,56 L120,56 L126,28 L134,28" fill="none" stroke="#6366f1" strokeWidth="2.5" />
      <text x="117" y="84" textAnchor="middle" fontSize="11" fill="#4338ca">U-Net</text>
      <A x={160} />
      <g transform="translate(196,22)">
        <rect width="48" height="48" rx="4" fill="#ecfdf5" stroke="#a7f3d0" />
        <rect x="2" y="2" width="44" height="20" fill="#93c5fd" /><rect x="2" y="24" width="22" height="22" fill="#6ee7b7" /><rect x="26" y="24" width="20" height="22" fill="#fcd34d" />
      </g>
      <text x="220" y="84" textAnchor="middle" fontSize="11" fill="#047857">raster</text>
    </svg>
  );
}

const STATUS: Record<Status, { label: string; chip: string; bar: string }> = {
  done: { label: "done · live", chip: "border-emerald-200 bg-emerald-50 text-emerald-700", bar: "bg-emerald-500" },
  next: { label: "next up", chip: "border-indigo-200 bg-indigo-50 text-indigo-700", bar: "bg-indigo-500" },
  stretch: { label: "stretch", chip: "border-violet-200 bg-violet-50 text-violet-700", bar: "bg-violet-500" },
};

const FILTERS: { key: "all" | Status; label: string }[] = [
  { key: "all", label: "All" },
  { key: "done", label: "Done" },
  { key: "next", label: "Next up" },
  { key: "stretch", label: "Stretch" },
];

export default function Strategies() {
  const [filter, setFilter] = useState<"all" | Status>("all");
  const shown = IDEAS.filter((i) => filter === "all" || i.status === filter);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
                filter === f.key
                  ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex flex-wrap gap-4 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />done</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-indigo-500" />next</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-violet-500" />stretch</span>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {shown.map((i) => (
          <div
            key={i.n}
            className={`flex flex-col rounded-2xl border bg-white p-6 shadow-sm ${
              i.best ? "border-2 border-indigo-200" : "border-slate-200"
            }`}
          >
            <div className="flex flex-wrap items-center gap-3">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-slate-100 text-sm font-semibold text-slate-700">{i.n}</span>
              <h3 className="text-base font-semibold text-slate-900">{i.title}</h3>
              <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${STATUS[i.status].chip}`}>
                {STATUS[i.status].label}
              </span>
              {i.best && (
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">best model</span>
              )}
            </div>

            <div className="mt-4 overflow-hidden rounded-xl border border-slate-100 bg-slate-50/60 px-2 py-1">
              <MiniViz n={i.n} />
            </div>

            <p className="mt-3 text-sm leading-relaxed text-slate-600">{i.summary}</p>

            <ul className="mt-3 space-y-1.5">
              {i.points.map((p, idx) => (
                <li key={idx} className="flex gap-2 text-sm text-slate-600">
                  <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-slate-300" />
                  {p}
                </li>
              ))}
            </ul>

            <div className="mt-auto pt-5">
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div><div className="text-slate-400">Output</div><div className="mt-0.5 font-medium text-slate-700">{i.output}</div></div>
                <div><div className="text-slate-400">Effort</div><div className="mt-0.5 font-medium text-slate-700">{i.effort}</div></div>
                <div><div className="text-slate-400">Expected FID</div><div className="mt-0.5 font-medium text-slate-700">{i.fid}</div></div>
              </div>
              <div className="mt-3">
                <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
                  <span>Fit to the task</span><span>{i.fit}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div className={`h-full rounded-full ${STATUS[i.status].bar}`} style={{ width: `${i.fit}%` }} />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
