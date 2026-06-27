"use client";

import { useEffect, useRef, useState } from "react";

const MODELS = [
  { name: "Retrieval v2 (structure-aware)", fid: 34.1, tag: "retrieves real plans", color: "#0ea5e9" },
  { name: "Rectilinear · wall-aligned", fid: 80.9, tag: "rule-based generator", color: "#4f46e5" },
  { name: "Partition · concave", fid: 81.5, tag: "rule-based generator", color: "#16a34a" },
  { name: "U-Net · 256px (MI300X)", fid: 145.7, tag: "learned generator", color: "#f59e0b" },
].sort((a, b) => a.fid - b.fid);

export default function FidBars() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={ref} className="space-y-5">
      {MODELS.map((m, i) => {
        const score = Math.max(0.05, (180 - m.fid) / 180); // lower FID -> longer bar
        return (
          <div key={m.name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="text-sm font-semibold text-slate-900 sm:text-base">{m.name}</span>
              <span className="font-mono text-sm text-slate-500">
                FID <span className="font-semibold text-slate-900">{m.fid.toFixed(1)}</span>
              </span>
            </div>
            <div className="h-3.5 overflow-hidden rounded-full bg-slate-200/80">
              <div
                className="h-full rounded-full transition-[width] duration-[1100ms] ease-out"
                style={{
                  width: shown ? `${score * 100}%` : "0%",
                  transitionDelay: `${i * 140}ms`,
                  background: m.color,
                }}
              />
            </div>
            <div className="mt-1 text-xs text-slate-400">{m.tag}</div>
          </div>
        );
      })}
      <p className="pt-2 text-xs text-slate-400">
        FID = Fréchet Inception Distance, lower is better. Retrieval copies a real plan; the others
        <em> generate</em> geometry. Live numbers on the Live page.
      </p>
    </div>
  );
}
