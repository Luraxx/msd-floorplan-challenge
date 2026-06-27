import { NextRequest } from "next/server";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { openSync } from "node:fs";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const execFileP = promisify(execFile);

// Detect ANY training process on the box (web-started OR manual), so we never
// launch a second run and overload the single GPU.
async function trainingProcess(): Promise<boolean> {
  try {
    const { stdout } = await execFileP("pgrep", ["-f", "unet_seg.py train"], { timeout: 5000 });
    return stdout.trim().length > 0;
  } catch {
    return false; // pgrep exits 1 when nothing matches
  }
}

const ROOT = path.resolve(process.cwd(), "..");
const OUTPUTS = path.join(ROOT, "outputs");
const STATE = path.join(OUTPUTS, "web_train.json");
const LOG = path.join(OUTPUTS, "web_train.log");
const SCRIPT = path.join(ROOT, "scripts", "web_train.sh");

type Config = { size: number; epochs: number; batch: number; ntrain: number; ntest: number };
type State = {
  pid: number | null;
  config: Config;
  startedAt: number;
  finishedAt: number | null;
  status: "running" | "done" | "stopped" | "error";
  runId?: string;
  modelName?: string;
};

function clamp(v: unknown, lo: number, hi: number, dflt: number): number {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n)) return dflt;
  return Math.min(Math.max(n, lo), hi);
}

function alive(pid: number | null): boolean {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
}

async function readState(): Promise<State | null> {
  try { return JSON.parse(await readFile(STATE, "utf8")); } catch { return null; }
}

function parseLog(text: string) {
  const lines = text.split("\n").filter(Boolean);
  const epochs: { epoch: number; total: number; loss: number }[] = [];
  let phase = "";
  let metrics: Record<string, number> | null = null;
  for (const ln of lines) {
    const e = ln.match(/^epoch\s+(\d+)\/(\d+)\s+loss=([\d.]+)/);
    if (e) epochs.push({ epoch: +e[1], total: +e[2], loss: +e[3] });
    const p = ln.match(/\[web-train\]\s+PHASE\s+(\w+)/);
    if (p) phase = p[1];
    const f = ln.match(/FID\s+([\d.]+)/);
    if (f) metrics = { ...(metrics || {}), fid: +f[1] };
    const d = ln.match(/Density\s+([\d.]+)/);
    if (d) metrics = { ...(metrics || {}), density: +d[1] };
    const c = ln.match(/Coverage\s+([\d.]+)/);
    if (c) metrics = { ...(metrics || {}), coverage: +c[1] };
  }
  return { phase, epochs, metrics, tail: lines.slice(-40) };
}

export async function GET() {
  const st = await readState();
  let log = "";
  try { log = await readFile(LOG, "utf8"); } catch { /* none yet */ }
  const running = !!st && alive(st.pid);
  const gpuBusy = await trainingProcess();
  // A training is happening that this panel did not start (e.g. a manual run)
  const foreignRun = gpuBusy && !running;
  const parsed = parseLog(log);
  let status = st?.status ?? "idle";
  if (st && st.status === "running" && !running) status = "done"; // process exited
  return Response.json({
    running,
    gpuBusy,
    foreignRun,
    status,
    config: st?.config ?? null,
    startedAt: st?.startedAt ?? null,
    phase: parsed.phase,
    epochs: parsed.epochs,
    metrics: parsed.metrics,
    tail: parsed.tail,
  });
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const action = body?.action;

  if (action === "stop") {
    const st = await readState();
    if (st?.pid && alive(st.pid)) {
      try { process.kill(-st.pid, "SIGTERM"); } catch { try { process.kill(st.pid, "SIGTERM"); } catch { /* gone */ } }
    }
    if (st) await writeFile(STATE, JSON.stringify({ ...st, status: "stopped", finishedAt: Date.now() }));
    return Response.json({ ok: true, stopped: true });
  }

  if (action === "start") {
    const existing = await readState();
    if (existing && alive(existing.pid)) {
      return Response.json({ ok: false, error: "A training run is already active — stop it first." }, { status: 409 });
    }
    // Guard against ANY training process (e.g. a manually started run) so we
    // never run two trainings at once and overload the single GPU.
    if (await trainingProcess()) {
      return Response.json({ ok: false, error: "The GPU is already busy with another training run. Wait for it to finish." }, { status: 409 });
    }
    const config: Config = {
      size: clamp(body?.size, 64, 512, 128),
      epochs: clamp(body?.epochs, 1, 300, 20),
      batch: clamp(body?.batch, 1, 128, 32),
      ntrain: clamp(body?.ntrain, 1, 4572, 1000),
      ntest: clamp(body?.ntest, 1, 800, 200),
    };
    await mkdir(OUTPUTS, { recursive: true });
    const startedAt = Date.now();
    const runId = `web-${startedAt}`;
    const modelName = `U-Net · ${config.size}px · ${config.epochs}ep`;
    const fd = openSync(LOG, "w"); // truncate
    const child = spawn("bash", [SCRIPT], {
      cwd: ROOT,
      detached: true,
      stdio: ["ignore", fd, fd],
      env: {
        ...process.env,
        SIZE: String(config.size),
        EPOCHS: String(config.epochs),
        BATCH: String(config.batch),
        NTRAIN: String(config.ntrain),
        NTEST: String(config.ntest),
        RUN_ID: runId,
        MODEL_NAME: modelName,
      },
    });
    child.unref();
    const st: State = { pid: child.pid ?? null, config, startedAt, finishedAt: null, status: "running", runId, modelName };
    await writeFile(STATE, JSON.stringify(st));
    return Response.json({ ok: true, pid: child.pid, runId, config });
  }

  return Response.json({ ok: false, error: "unknown action" }, { status: 400 });
}
