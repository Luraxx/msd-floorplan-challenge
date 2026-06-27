"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export type DrawCanvasHandle = {
  getPNG: () => string;
  clear: () => void;
  isEmpty: () => boolean;
};

type Pt = { x: number; y: number };
const SIZE = 512;
const RDP_EPS = 7;       // simplification tolerance (px)
const SNAP_DEG = 13;     // snap a segment to horizontal/vertical within this angle

// --- geometry: Ramer–Douglas–Peucker + axis snapping ------------------------
function perpDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  return Math.abs((p.x - a.x) * dy - (p.y - a.y) * dx) / len;
}
function rdp(pts: Pt[], eps: number): Pt[] {
  if (pts.length < 3) return pts;
  let dmax = 0, idx = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = perpDist(pts[i], pts[0], pts[pts.length - 1]);
    if (d > dmax) { dmax = d; idx = i; }
  }
  if (dmax > eps) {
    const left = rdp(pts.slice(0, idx + 1), eps);
    const right = rdp(pts.slice(idx), eps);
    return left.slice(0, -1).concat(right);
  }
  return [pts[0], pts[pts.length - 1]];
}
function straighten(points: Pt[]): Pt[] {
  if (points.length < 2) return points;
  const simp = rdp(points, RDP_EPS);
  if (simp.length < 2) return [points[0], points[points.length - 1]];
  const out: Pt[] = [simp[0]];
  let cur = simp[0];
  for (let i = 1; i < simp.length; i++) {
    const nx = simp[i];
    const ang = (Math.atan2(nx.y - cur.y, nx.x - cur.x) * 180) / Math.PI;
    const a = ((ang % 180) + 180) % 180; // 0..180
    const snapped: Pt = { x: nx.x, y: nx.y };
    if (a < SNAP_DEG || a > 180 - SNAP_DEG) snapped.y = cur.y;       // ~horizontal
    else if (Math.abs(a - 90) < SNAP_DEG) snapped.x = cur.x;          // ~vertical
    out.push(snapped);
    cur = snapped;
  }
  return out;
}

const DrawCanvas = forwardRef<DrawCanvasHandle, { className?: string }>(function DrawCanvas(
  { className = "" },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const shapes = useRef<Pt[][]>([]);       // committed, straightened walls
  const raw = useRef<Pt[]>([]);            // current freehand stroke
  const drawing = useRef(false);
  const [brush, setBrush] = useState(6);
  const brushRef = useRef(6);
  brushRef.current = brush;

  const ctx = () => canvasRef.current?.getContext("2d") || null;

  const redraw = () => {
    const c = ctx();
    if (!c) return;
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, SIZE, SIZE);
    c.strokeStyle = "#111111";
    c.lineCap = "round";
    c.lineJoin = "round";
    c.lineWidth = brushRef.current;
    for (const s of shapes.current) {
      if (s.length < 2) continue;
      c.beginPath();
      c.moveTo(s[0].x, s[0].y);
      for (let i = 1; i < s.length; i++) c.lineTo(s[i].x, s[i].y);
      c.stroke();
    }
  };

  useEffect(() => { redraw(); /* eslint-disable-next-line */ }, []);
  useEffect(() => { redraw(); /* eslint-disable-next-line */ }, [brush]);

  const pos = (e: React.PointerEvent): Pt => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  };

  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    canvasRef.current?.setPointerCapture(e.pointerId);
    drawing.current = true;
    raw.current = [pos(e)];
  };

  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const c = ctx();
    if (!c) return;
    const p = pos(e);
    const prev = raw.current[raw.current.length - 1];
    raw.current.push(p);
    // faint live preview of the freehand stroke
    c.strokeStyle = "rgba(99,102,241,0.45)";
    c.lineWidth = brushRef.current;
    c.lineCap = "round";
    c.beginPath();
    c.moveTo(prev.x, prev.y);
    c.lineTo(p.x, p.y);
    c.stroke();
  };

  const up = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    drawing.current = false;
    try { canvasRef.current?.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    const s = straighten(raw.current);
    if (s.length >= 2) shapes.current.push(s);
    raw.current = [];
    redraw(); // replace the faint freehand with the clean straightened wall
  };

  useImperativeHandle(ref, () => ({
    getPNG: () => { redraw(); return canvasRef.current?.toDataURL("image/png") || ""; },
    clear: () => { shapes.current = []; redraw(); },
    isEmpty: () => shapes.current.length === 0,
  }));

  const undo = () => { shapes.current.pop(); redraw(); };

  return (
    <div className={className}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white">✏️ Wall</span>
        <label className="flex items-center gap-2 text-xs text-slate-500">
          thickness
          <input type="range" min={3} max={16} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
        </label>
        <button onClick={undo} className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">↶ Undo</button>
        <button onClick={() => { shapes.current = []; redraw(); }} className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">Clear</button>
        <span className="ml-auto text-xs text-slate-400">drawn lines snap straight ↗</span>
      </div>
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerLeave={up}
        onPointerCancel={up}
        className="aspect-square w-full touch-none rounded-xl border border-slate-300 bg-white shadow-inner"
        style={{ touchAction: "none" }}
      />
    </div>
  );
});

export default DrawCanvas;
