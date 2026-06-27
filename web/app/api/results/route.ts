import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ROOT = path.resolve(process.cwd(), "..");
const STORE = path.join(ROOT, "outputs", "models");

async function listIds(id: string): Promise<string[]> {
  try {
    const files = await readdir(path.join(STORE, id, "generated"));
    return files.filter((f) => f.endsWith(".pickle")).map((f) => f.slice(0, -7))
      .sort((a, b) => Number(a) - Number(b));
  } catch { return []; }
}

// GET /api/results -> every trained model with its metrics + generated ids
export async function GET() {
  let dirs: string[] = [];
  try { dirs = await readdir(STORE); } catch { /* empty store */ }

  const runs = [];
  for (const id of dirs) {
    let meta: { name?: string; metrics?: Record<string, number> | null; createdAt?: number; status?: string };
    try { meta = JSON.parse(await readFile(path.join(STORE, id, "meta.json"), "utf8")); }
    catch { continue; }
    const ids = await listIds(id);
    runs.push({
      dir: id,                       // used as the /api/sample pred param
      label: meta.name || id,
      count: ids.length,
      ids: ids.slice(0, 200),
      metrics: meta.metrics ?? null,
      status: meta.status ?? "done",
      createdAt: meta.createdAt ?? 0,
    });
  }
  runs.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));

  const baselines = [
    { name: "Retrieval v1", fid: 36.0, density: 0.91, coverage: 0.89 },
    { name: "Retrieval v2 (structure-aware)", fid: 34.1, density: 0.87, coverage: 0.91 },
  ];
  return Response.json({ runs, baselines });
}
