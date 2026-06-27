import { NextRequest } from "next/server";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, mkdir, access } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const execFileP = promisify(execFile);
const ROOT = path.resolve(process.cwd(), "..");
const PY = path.join(ROOT, ".venv", "bin", "python");
const SCRIPT = path.join(ROOT, "scripts", "web_render.py");
const CACHE = path.join(ROOT, "outputs", "render-cache");

const KINDS = new Set(["walls", "rooms", "graph", "pred"]);

// GET /api/sample?split=train&id=10000&kind=rooms&size=512[&pred=<dir>]
export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const split = sp.get("split") === "test" ? "test" : "train";
  const id = (sp.get("id") || "").replace(/[^0-9]/g, "");
  const kind = KINDS.has(sp.get("kind") || "") ? sp.get("kind")! : "rooms";
  const size = Math.min(Math.max(parseInt(sp.get("size") || "512", 10) || 512, 128), 768);
  const predDir = (sp.get("pred") || "").replace(/[^a-zA-Z0-9_\-]/g, "");
  if (!id) return new Response("missing id", { status: 400 });

  const tag = kind === "pred" ? `pred-${predDir}` : kind;
  const outPath = path.join(CACHE, `${split}_${id}_${tag}_${size}.png`);

  try {
    await access(outPath);
  } catch {
    await mkdir(CACHE, { recursive: true });
    const args = ["--split", split, "--id", id, "--kind", kind, "--out", outPath, "--size", String(size)];
    if (kind === "pred") {
      if (!predDir) return new Response("missing pred model", { status: 400 });
      // pred = model id -> its generated predictions in the model store
      args.push("--pred-dir", path.join(ROOT, "outputs", "models", predDir, "generated"));
    }
    try {
      await execFileP(PY, [SCRIPT, ...args], { cwd: ROOT, timeout: 30000, maxBuffer: 1 << 20 });
    } catch (e: unknown) {
      const msg = (e as { stderr?: string; message?: string })?.stderr ||
        (e as { message?: string })?.message || "unknown error";
      return new Response("render failed: " + msg, { status: 500 });
    }
  }

  const buf = await readFile(outPath);
  return new Response(new Uint8Array(buf), {
    headers: { "Content-Type": "image/png", "Cache-Control": "public, max-age=3600" },
  });
}
