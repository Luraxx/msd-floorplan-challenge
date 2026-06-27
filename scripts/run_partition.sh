#!/usr/bin/env bash
# Generate + eval + register a Graph-conditioned Partition model into the store.
#   RUN_ID=partition-concave MODEL_NAME="Partition (concave)" MODE=concave N=800 bash scripts/run_partition.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1

MSD=${MSD:-data/modified-swiss-dwellings-v2}
RUN_ID=${RUN_ID:?set RUN_ID}
MODEL_NAME=${MODEL_NAME:?set MODEL_NAME}
MODE=${MODE:-concave}
N=${N:-800}
DIR="outputs/models/$RUN_ID"
mkdir -p "$DIR/generated"

echo "[partition] id=$RUN_ID mode=$MODE n=$N"
python -u src/model/baseline_partition.py --test "$MSD/test" --train "$MSD/train" \
  --out "$DIR/generated" --n "$N" --envelope-mode "$MODE"

echo "[partition] eval"
python -u src/eval/run_eval.py --real "$MSD/test/graph_out" --fake "$DIR/generated" \
  --n "$N" --out "$DIR/metrics.json"

python - "$RUN_ID" "$MODEL_NAME" "$MODE" "$N" <<'PY'
import sys, json, os, time
rid, name, mode, n = sys.argv[1:5]; d = f"outputs/models/{rid}"
meta = {"id": rid, "name": name, "family": "partition", "status": "done",
        "config": {"type": f"Voronoi partition + learned labeling", "envelope": mode, "ntest": int(n)},
        "metrics": None, "createdAt": int(time.time() * 1000), "source": "partition"}
try:
    m = json.load(open(os.path.join(d, "metrics.json")))
    meta["metrics"] = {k: m[k] for k in ("fid", "density", "coverage") if k in m}
except Exception:
    pass
meta["nGenerated"] = len([f for f in os.listdir(os.path.join(d, "generated")) if f.endswith(".pickle")])
json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
print("[partition] metrics:", json.dumps(meta["metrics"]))
PY
echo "[partition] DONE $RUN_ID"
