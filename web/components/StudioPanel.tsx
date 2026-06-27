"use client";

import { useEffect, useRef, useState } from "react";
import DrawCanvas, { DrawCanvasHandle } from "./DrawCanvas";

type Room = { type: number; name: string; color: string; count: number };
type Doc = { n_rooms: number; n_connections: number; free_fraction: number; rooms: Room[]; weights: string };
type Model = { id: string; name: string; status: string; hasWeights: boolean; metrics?: { fid?: number } | null };

export default function StudioPanel() {
  const canvas = useRef<DrawCanvasHandle | null>(null);
  const [rooms, setRooms] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [plan, setPlan] = useState<string | null>(null);
  const [doc, setDoc] = useState<Doc | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState<string>("");

  useEffect(() => {
    fetch("/api/models", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        const usable: Model[] = (d.models || []).filter((m: Model) => m.hasWeights);
        setModels(usable);
        if (usable.length) setModel((cur) => cur || usable[0].id);
      })
      .catch(() => {});
  }, []);

  const generate = async () => {
    const image = canvas.current?.getPNG();
    if (!image) return;
    setBusy(true); setErr(""); setPlan(null); setDoc(null);
    try {
      const r = await fetch("/api/canvas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, rooms, model }),
      });
      const d = await r.json();
      if (!d.ok) { setErr(d.error || "generation failed"); return; }
      setPlan(d.image);
      setDoc(d.doc);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* left: drawing */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-1 text-sm font-semibold text-slate-900">1 · Draw the structure</h3>
        <p className="mb-4 text-xs text-slate-500">
          Sketch the outer walls as a <strong>closed shape</strong>, then add interior walls. Everything
          enclosed becomes the apartment; the model lays out the rooms inside.
        </p>
        <DrawCanvas ref={canvas} />
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-medium text-slate-500">
            Model
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-slate-900"
            >
              {models.length === 0 && <option value="">no trained model</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}{m.metrics?.fid != null ? ` · FID ${m.metrics.fid.toFixed(0)}` : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs font-medium text-slate-500">
            Room hint
            <input
              type="number" min={0} max={40} value={rooms}
              onChange={(e) => setRooms(Number(e.target.value))}
              className="w-20 rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-slate-900"
            />
            <span className="text-slate-400">(0 = auto)</span>
          </label>
          <button
            onClick={generate}
            disabled={busy || !model}
            title={!model ? "train a model first" : ""}
            className="ml-auto rounded-lg bg-indigo-600 px-5 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {busy ? "Generating…" : "✨ Generate floor plan"}
          </button>
        </div>
        {err && <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{err}</p>}
      </div>

      {/* right: result + documentation */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-1 text-sm font-semibold text-slate-900">2 · Generated apartment</h3>
        <p className="mb-4 text-xs text-slate-500">
          The trained U-Net turns your structure into a room layout, then documents what goes where.
        </p>

        <div className="grid aspect-square w-full place-items-center overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
          {busy ? (
            <div className="flex flex-col items-center gap-2 text-slate-400">
              <span className="h-6 w-6 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-500" />
              <span className="text-sm">running the model…</span>
            </div>
          ) : plan ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={plan} alt="generated plan" className="h-full w-full object-contain" />
          ) : (
            <span className="text-sm text-slate-400">Draw a structure and hit Generate</span>
          )}
        </div>

        {doc && (
          <div className="mt-4">
            <div className="mb-3 flex gap-3 text-sm">
              <Stat label="Rooms" v={doc.n_rooms} />
              <Stat label="Connections" v={doc.n_connections} />
              <Stat label="Model" v={doc.weights} mono />
            </div>
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Room breakdown</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {doc.rooms.map((r) => (
                <span key={r.type} className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: r.color }} />
                  {r.name} <span className="text-slate-400">×{r.count}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, v, mono }: { label: string; v: number | string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`font-semibold text-slate-900 ${mono ? "font-mono text-xs" : ""}`}>{v}</div>
    </div>
  );
}
