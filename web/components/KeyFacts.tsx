const STATS = [
  { v: "5,372", l: "Floor plans" },
  { v: "18.9K+", l: "Units (apartments)" },
  { v: "165.3K+", l: "Areas (rooms)" },
];

const SPLITS = [
  { v: "4,167", l: "Train floor plans", sub: "with ground truth" },
  { v: "1,205", l: "Test floor plans", sub: "inputs only" },
];

const SPEC = [
  ["License", "CC BY 4.0"],
  ["DOI", "10.4121/e1d89cb5-…-v2"],
  ["Modalities", "Raster · vector/geometry · access graphs"],
  ["Container", "NetworkX Graph / PyTorch Geometric"],
  ["File formats", ".pickle (graphs) · .npy (rasters)"],
  ["Source", "Derived from Swiss Dwellings v3.0.0"],
];

export default function KeyFacts() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-3">
        {STATS.map((s) => (
          <div key={s.l} className="rounded-2xl border border-slate-200 bg-white p-5 text-center shadow-sm">
            <div className="text-3xl font-semibold tracking-tight text-slate-900">{s.v}</div>
            <div className="mt-1 text-xs font-medium uppercase tracking-wide text-slate-500">{s.l}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            {SPLITS.map((s) => (
              <div key={s.l} className="rounded-2xl border border-slate-200 bg-paper p-5">
                <div className="text-2xl font-semibold tracking-tight text-indigo-700">{s.v}</div>
                <div className="mt-1 text-sm font-medium text-slate-700">{s.l}</div>
                <div className="text-xs text-slate-500">{s.sub}</div>
              </div>
            ))}
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <dl className="divide-y divide-slate-100">
              {SPEC.map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between gap-4 py-2 first:pt-0 last:pb-0">
                  <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{k}</dt>
                  <dd className="text-right text-sm font-medium text-slate-800">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-900">What makes it unique</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Built for benchmark comparability at <strong>building-complex scale</strong> — many rooms,
              irregular geometries, multiple apartments per floor, shared circulation constraints and
              connectivity across units. That sets it apart from single-unit datasets like RPLAN and LIFULL.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-900">The core task</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              Models receive <strong>(1)</strong> a functional diagram (access graph) and{" "}
              <strong>(2)</strong> the necessary building structure (binary image), and predict the full
              layout — floor-plan auto-completion / generation at building-complex scale.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm">
            <a className="rounded-lg border border-slate-300 bg-white px-4 py-2 font-medium text-slate-700 hover:border-slate-400" href="https://archilyse.standfest.science/modified-swiss-dwellings" target="_blank" rel="noopener noreferrer">
              Dataset page →
            </a>
            <a className="rounded-lg border border-slate-300 bg-white px-4 py-2 font-medium text-slate-700 hover:border-slate-400" href="https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9" target="_blank" rel="noopener noreferrer">
              Download (4TU) →
            </a>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Key facts from the official dataset page (Archilyse · ECCV 2024). Note: the splits quoted here are
        the dataset&apos;s published split; the local ML-ready copy used in this repo subsets it differently.
      </p>
    </div>
  );
}
