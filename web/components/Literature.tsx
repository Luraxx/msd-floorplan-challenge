type Paper = {
  title: string;
  authors: string;
  year: number;
  venue: string;
  group: "core" | "using";
  pdf?: string;
  links: { label: string; href: string }[];
  note?: string;
};

const PAPERS: Paper[] = [
  {
    title: "MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes",
    authors: "van Engelenburg et al.", year: 2024, venue: "ECCV", group: "core",
    pdf: "/literatur/msd-benchmark-2024.pdf",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2407.10121" },
      { label: "GitHub", href: "https://github.com/caspervanengelenburg/msd" },
    ],
  },
  {
    title: "Floor plan generation: the interplay among data, machine, and designer",
    authors: "Mostafavi et al.", year: 2024, venue: "IJAC", group: "core",
    pdf: "/literatur/floorplan-interplay-2024.pdf",
    links: [{ label: "DOI", href: "https://doi.org/10.1177/14780771241290649" }],
  },
  {
    title: "LayoutGKN: Graph Similarity Learning of Floor Plans",
    authors: "van Engelenburg et al.", year: 2025, venue: "BMVC", group: "using",
    pdf: "/literatur/layoutgkn-2025.pdf",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2509.03737" },
      { label: "GitHub", href: "https://github.com/caspervanengelenburg/LayoutGKN" },
    ],
  },
  {
    title: "GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation",
    authors: "Hu et al.", year: 2025, venue: "AAAI", group: "using",
    pdf: "/literatur/gsdiff-2025.pdf",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2408.16258" },
      { label: "GitHub", href: "https://github.com/SizheHu/GSDiff" },
    ],
  },
  {
    title: "DoorDet: Semi-Automated Multi-Class Door Detection Dataset via Object Detection and LLMs",
    authors: "—", year: 2025, venue: "arXiv", group: "using",
    pdf: "/literatur/doordet-2025.pdf",
    links: [{ label: "arXiv", href: "https://arxiv.org/abs/2508.07714" }],
  },
  {
    title: "Generating accessible multi-occupancy floor plans with fine-grained control using a diffusion model",
    authors: "—", year: 2025, venue: "Automation in Construction", group: "using",
    note: "Paywalled — link only",
    links: [{ label: "DOI", href: "https://doi.org/10.1016/j.autcon.2025.106332" }],
  },
  {
    title: "MRED-14: A Benchmark for Low-Energy Residential Floor Plan Generation with 14 Flexible Inputs",
    authors: "Zeng et al.", year: 2025, venue: "ACM Multimedia", group: "using",
    note: "Paywalled — link only",
    links: [{ label: "DOI", href: "https://doi.org/10.1145/3746027.3754949" }],
  },
  {
    title: "Semi-Automated Dataset Generation for Residential Buildings Using Graph-Based Topological Modelling",
    authors: "—", year: 2025, venue: "Buildings (MDPI)", group: "using",
    note: "Open access — grab via the publisher (CDN blocks automated download)",
    links: [{ label: "MDPI", href: "https://www.mdpi.com/2075-5309/15/8/1283" }],
  },
];

function Card({ p }: { p: Paper }) {
  return (
    <div className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2 text-xs">
        <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 font-medium text-slate-600">{p.venue}</span>
        <span className="text-slate-400">{p.year}</span>
        {p.pdf ? (
          <span className="ml-auto rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700">PDF stored</span>
        ) : (
          <span className="ml-auto rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 font-medium text-amber-700">link only</span>
        )}
      </div>

      <h3 className="mt-3 text-[15px] font-semibold leading-snug text-slate-900">{p.title}</h3>
      <p className="mt-1 text-sm text-slate-500">{p.authors}</p>
      {p.note && <p className="mt-2 text-xs text-slate-500">{p.note}</p>}

      <div className="mt-auto flex flex-wrap gap-2 pt-4">
        {p.pdf && (
          <a
            href={p.pdf}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Open PDF →
          </a>
        )}
        {p.links.map((l) => (
          <a
            key={l.label}
            href={l.href}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:border-slate-400"
          >
            {l.label}
          </a>
        ))}
      </div>
    </div>
  );
}

export default function Literature() {
  const core = PAPERS.filter((p) => p.group === "core");
  const using = PAPERS.filter((p) => p.group === "using");
  const stored = PAPERS.filter((p) => p.pdf).length;

  return (
    <div className="space-y-8">
      <p className="max-w-2xl text-slate-600">
        The reading list behind the project, sourced from the official dataset page. {stored} papers are
        stored locally as PDFs — open them right here; the rest link out to the publisher.
      </p>

      <div>
        <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-indigo-600">Core MSD</div>
        <div className="grid gap-5 md:grid-cols-2">
          {core.map((p) => <Card key={p.title} p={p} />)}
        </div>
      </div>

      <div>
        <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-indigo-600">Research using MSD</div>
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {using.map((p) => <Card key={p.title} p={p} />)}
        </div>
      </div>
    </div>
  );
}
