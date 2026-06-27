import { NextRequest } from "next/server";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { writeFile, readFile, mkdir, access } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const execFileP = promisify(execFile);
const ROOT = path.resolve(process.cwd(), "..");
const PY = path.join(ROOT, ".venv", "bin", "python");
const SCRIPT = path.join(ROOT, "scripts", "canvas_predict.py");
const CANVAS_DIR = path.join(ROOT, "outputs", "canvas");

// POST { image: dataURL(png of drawn walls), rooms?: number, model?: id }
//   -> { ok, doc, image: dataURL(generated plan) }
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));

  // resolve which model's weights to use (selected model from the store, else default)
  const modelId = String(body?.model || "").replace(/[^a-zA-Z0-9_\-]/g, "");
  const weights = modelId
    ? path.join(ROOT, "outputs", "models", modelId, "weights.pt")
    : path.join(ROOT, "outputs", "unet.pt");
  try { await access(weights); }
  catch { return Response.json({ ok: false, error: "Selected model has no weights yet. Pick a finished model or train one." }, { status: 400 }); }

  const dataUrl: string = body?.image || "";
  const m = dataUrl.match(/^data:image\/png;base64,(.+)$/);
  if (!m) return Response.json({ ok: false, error: "expected a PNG data URL in 'image'" }, { status: 400 });
  const rooms = Math.min(Math.max(Math.floor(Number(body?.rooms) || 0), 0), 40);

  await mkdir(CANVAS_DIR, { recursive: true });
  const stamp = `${Date.now()}_${Math.floor(performance.now())}`;
  const sketchPath = path.join(CANVAS_DIR, `sketch_${stamp}.png`);
  const planPath = path.join(CANVAS_DIR, `plan_${stamp}.png`);
  await writeFile(sketchPath, Buffer.from(m[1], "base64"));

  let doc: unknown;
  try {
    const { stdout } = await execFileP(
      PY,
      [SCRIPT, "--in", sketchPath, "--out", planPath, "--weights", weights, "--rooms", String(rooms)],
      { cwd: ROOT, timeout: 120000, maxBuffer: 4 << 20 },
    );
    const line = stdout.trim().split("\n").filter(Boolean).pop() || "{}";
    doc = JSON.parse(line);
  } catch (e: unknown) {
    const msg = (e as { stderr?: string; message?: string })?.stderr ||
      (e as { message?: string })?.message || "inference failed";
    return Response.json({ ok: false, error: String(msg).slice(-400) }, { status: 500 });
  }

  if (doc && typeof doc === "object" && (doc as { error?: string }).error) {
    return Response.json({ ok: false, error: (doc as { error: string }).error }, { status: 400 });
  }

  let image = "";
  try {
    const buf = await readFile(planPath);
    image = `data:image/png;base64,${buf.toString("base64")}`;
  } catch {
    return Response.json({ ok: false, error: "plan was not generated" }, { status: 500 });
  }

  return Response.json({ ok: true, doc, image });
}
