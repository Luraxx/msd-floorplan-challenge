"use client";

import { useEffect, useMemo, useState } from "react";

type Run = {
  dir: string;
  label: string;
  count: number;
  ids: string[];
  metrics: Record<string, number> | null;
  updatedAt: number | null;
};
type Baseline = { name: string; fid: number; density: number; coverage: number };
type Results = { runs: Run[]; baselines: Baseline[] };

export default function ResultsPanel() {
  const [data, setData] = useState<Results | null>(null);
  const [runIdx, setRunIdx] = useState(0);
  const [pos, setPos] = useState(0);

  const load = () =>
    fetch("/api/results", { cache: "no-store" }).then((r) => r.json()).then(setData).catch(() => {});

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const withPlans = (data?.runs ?? []).filter((r) => r.count > 0);
  const run = withPlans[runIdx];
  const id = run?.ids?.[pos];

  const cards = useMemo(() => {
    const rows: { name: string; fid?: number; density?: number; coverage?: number; live?: boolean }[] = [];
    for (const r of data?.runs ?? []) {
      if (r.metrics) rows.push({ name: r.label, fid: r.metrics.fid, density: r.metrics.density, coverage: r.metrics.coverage, live: true });
    }
    for (const b of data?.baselines ?? []) rows.push(b);
    return rows.sort((a, b) => (a.fid ?? 1e9) - (b.fid ?? 1e9));
  }, [data]);

  if (!data) return <div className="py-12 text-center text-sm text-slate-400">Loading results…</div>;
  if (!data.runs.length)
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-slate-500">
        No generated plans yet. Start a training run above — when it finishes, the predictions and metrics
        appear here automatically.
      </div>
    );

  return (
    <div className="space-y-6">
      {/* leaderboard */}
      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-2.5">Model</th>
              <th className="px-4 py-2.5">FID ↓</th>
              <th className="px-4 py-2.5">Density ↑</th>
              <th className="px-4 py-2.5">Coverage ↑</th>
            </tr>
          </thead>
          <tbody>
            {cards.map((c, i) => (
              <tr key={i} className={`border-b border-slate-100 ${c.live ? "bg-indigo-50/40" : ""}`}>
                <td className="px-4 py-2.5 font-medium text-slate-900">
                  {c.name} {c.live && <span className="ml-1 rounded bg-indigo-100 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-700">LIVE</span>}
                </td>
                <td className="px-4 py-2.5 font-mono">{c.fid != null ? c.fid.toFixed(2) : "—"}</td>
                <td className="px-4 py-2.5 font-mono">{c.density != null ? c.density.toFixed(3) : "—"}</td>
                <td className="px-4 py-2.5 font-mono">{c.coverage != null ? c.coverage.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* comparison browser */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <select
            value={runIdx}
            onChange={(e) => { setRunIdx(Number(e.target.value)); setPos(0); }}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm"
          >
            {withPlans.map((r, i) => (
              <option key={r.dir} value={i}>{r.label} ({r.count} plans)</option>
            ))}
          </select>
          <button onClick={() => setPos((p) => Math.max(0, p - 1))} className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm hover:bg-slate-50">← Prev</button>
          <button onClick={() => setPos((p) => Math.min((run?.ids.length ?? 1) - 1, p + 1))} className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm hover:bg-slate-50">Next →</button>
          <button onClick={() => run && setPos(Math.floor(Math.random() * run.ids.length))} className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm hover:bg-slate-50">🎲</button>
          <div className="ml-auto text-sm text-slate-500">
            test <span className="font-mono font-semibold text-slate-900">#{id ?? "—"}</span>
            <span className="mx-2 text-slate-300">|</span>{run ? `${pos + 1} / ${run.ids.length}` : ""}
          </div>
        </div>

        {id && run && (
          <div className="grid gap-4 md:grid-cols-2">
            <Figure title="Ground truth (real)" note="real reference"
              src={`/api/sample?id=${id}&kind=truth&pred=${run.dir}&size=512`} />
            <Figure title={`Generated · ${run.label}`} note={`${run.dir} · predicted`}
              src={`/api/sample?split=test&id=${id}&kind=pred&pred=${run.dir}&size=512`} />
          </div>
        )}
      </div>
    </div>
  );
}

function Figure({ title, note, src }: { title: string; note: string; src: string }) {
  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      <div className="aspect-square w-full bg-white">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img key={src} src={src} alt={title} className="h-full w-full object-contain" loading="lazy" />
      </div>
      <figcaption className="border-t border-slate-200 px-3 py-2">
        <div className="text-sm font-medium text-slate-900">{title}</div>
        <div className="font-mono text-xs text-slate-400">{note}</div>
      </figcaption>
    </figure>
  );
}
