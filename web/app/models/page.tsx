import Link from "next/link";
import { Section, Chip } from "../../components/ui";
import Nav from "../../components/Nav";
import Footer from "../../components/Footer";
import { MODELS, ModelDoc } from "../models";

const FAMILY_COLOR: Record<ModelDoc["family"], string> = {
  generative: "#4f46e5",
  retrieval: "#0ea5e9",
  partition: "#16a34a",
  llm: "#db2777",
};
const STATUS_LABEL: Record<ModelDoc["status"], string> = {
  baseline: "Baseline",
  trained: "Trained",
  experimental: "Experimental",
  planned: "Planned",
};

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-2.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="font-mono text-lg font-semibold text-slate-900">{value}</div>
      {hint && <div className="text-[10px] text-slate-400">{hint}</div>}
    </div>
  );
}

function Card({ m }: { m: ModelDoc }) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: FAMILY_COLOR[m.family] }} />
        <h3 className="text-lg font-semibold text-slate-900">{m.name}</h3>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">{STATUS_LABEL[m.status]}</span>
        {m.generator && <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[11px] font-semibold text-indigo-700">powers Studio</span>}
        <span className="ml-auto font-mono text-xs text-slate-400">{m.date}</span>
      </div>

      <p className="mt-3 text-sm leading-relaxed text-slate-600">{m.summary}</p>

      {m.metrics && (
        <div className="mt-4 grid grid-cols-3 gap-2">
          <Metric label="FID ↓" value={m.metrics.fid?.toFixed(1) ?? "—"} hint={m.metrics.note} />
          <Metric label="Density ↑" value={m.metrics.density?.toFixed(2) ?? "—"} />
          <Metric label="Coverage ↑" value={m.metrics.coverage?.toFixed(2) ?? "—"} />
        </div>
      )}

      <div className="mt-4">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">How it works</div>
        <p className="mt-1 text-sm text-slate-600">{m.approach}</p>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Config</div>
          <dl className="mt-1 space-y-1">
            {m.config.map((c) => (
              <div key={c.label} className="flex justify-between gap-3 text-sm">
                <dt className="text-slate-500">{c.label}</dt>
                <dd className="text-right font-medium text-slate-800">{c.value}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="grid gap-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-emerald-600">Strengths</div>
            <ul className="mt-1 space-y-1 text-sm text-slate-600">
              {m.strengths.map((s, i) => <li key={i} className="flex gap-1.5"><span className="text-emerald-500">+</span>{s}</li>)}
            </ul>
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-rose-500">Limitations</div>
            <ul className="mt-1 space-y-1 text-sm text-slate-600">
              {m.limitations.map((s, i) => <li key={i} className="flex gap-1.5"><span className="text-rose-400">−</span>{s}</li>)}
            </ul>
          </div>
        </div>
      </div>
    </article>
  );
}

export default function ModelsPage() {
  const sorted = [...MODELS].sort((a, b) => (a.metrics?.fid ?? 1e9) - (b.metrics?.fid ?? 1e9));
  return (
    <main className="min-h-screen">
      <Nav />

      <div className="grid-bg border-b border-slate-200">
        <div className="mx-auto max-w-6xl px-5 py-16 md:py-20">
          <Chip color="#4f46e5">Models · documented</Chip>
          <h1 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight text-slate-900 md:text-5xl">
            Every model we build, documented
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-600">
            One card per model — what it is, how it works, its config and its measured FID / density /
            coverage. New models land here automatically. Train one yourself on the{" "}
            <Link href="/live" className="text-indigo-600 underline">Live</Link> page.
          </p>
        </div>
      </div>

      <Section eyebrow="Registry" title="The model catalogue">
        <div className="grid gap-6">
          {sorted.map((m) => <Card key={m.id} m={m} />)}
        </div>
      </Section>

      <Footer />
    </main>
  );
}
