import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ROOT = path.resolve(process.cwd(), "..");
const STORE = path.join(ROOT, "outputs", "models");

// GET /api/models -> every trained model in the store (newest first)
export async function GET() {
  let dirs: string[] = [];
  try { dirs = await readdir(STORE); } catch { return Response.json({ models: [] }); }

  const models = [];
  for (const id of dirs) {
    try {
      const meta = JSON.parse(await readFile(path.join(STORE, id, "meta.json"), "utf8"));
      let nGenerated = meta.nGenerated ?? 0;
      try { nGenerated = (await readdir(path.join(STORE, id, "generated"))).filter((f) => f.endsWith(".pickle")).length; } catch { /* keep */ }
      let hasWeights = false;
      try { await stat(path.join(STORE, id, "weights.pt")); hasWeights = true; } catch { /* no weights */ }
      models.push({ ...meta, id, nGenerated, hasWeights });
    } catch { /* skip incomplete dirs */ }
  }
  models.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
  return Response.json({ models });
}
