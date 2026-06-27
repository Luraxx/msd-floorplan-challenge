#!/usr/bin/env bash
# Web-triggered training run -> persisted into the model store at
# outputs/models/$RUN_ID/{weights.pt, generated/, metrics.json, meta.json}.
# Fully UNBUFFERED so the website can stream live progress. Never clobbers
# other models (each run has its own id).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1

MSD=${MSD:-data/modified-swiss-dwellings-v2}
SIZE=${SIZE:-128}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-32}
NTRAIN=${NTRAIN:-1000}
NTEST=${NTEST:-200}
RUN_ID=${RUN_ID:-webrun}
MODEL_NAME=${MODEL_NAME:-U-Net run}

DIR="outputs/models/$RUN_ID"
export UNET_WEIGHTS="$DIR/weights.pt"
mkdir -p "$DIR/generated"

# write meta.json up front (status=running) so it shows immediately
python - "$RUN_ID" "$MODEL_NAME" "$SIZE" "$EPOCHS" "$BATCH" "$NTRAIN" "$NTEST" <<'PY'
import sys, json, os, time
rid, name, size, ep, batch, ntr, nte = sys.argv[1:8]
d = f"outputs/models/{rid}"
json.dump({"id": rid, "name": name, "family": "generative", "status": "running",
           "config": {"size": int(size), "epochs": int(ep), "batch": int(batch),
                      "ntrain": int(ntr), "ntest": int(nte), "base": 24},
           "metrics": None, "createdAt": int(time.time() * 1000), "source": "web"},
          open(os.path.join(d, "meta.json"), "w"), indent=2)
PY

echo "[web-train] START id=$RUN_ID size=$SIZE epochs=$EPOCHS batch=$BATCH ntrain=$NTRAIN ntest=$NTEST"
echo "[web-train] PHASE train"
python -u src/model/unet_seg.py train \
  --train "$MSD/train" --size "$SIZE" --n "$NTRAIN" --epochs "$EPOCHS" --batch "$BATCH"

echo "[web-train] PHASE predict"
python -u src/model/unet_seg.py predict \
  --test "$MSD/test" --size "$SIZE" --n "$NTEST" --out "$DIR/generated"

echo "[web-train] PHASE eval"
python -u src/eval/run_eval.py \
  --real "$MSD/test/graph_out" --fake "$DIR/generated" --n "$NTEST" --out "$DIR/metrics.json"

# finalize meta.json (merge metrics, status=done)
python - "$RUN_ID" <<'PY'
import sys, json, os
rid = sys.argv[1]; d = f"outputs/models/{rid}"
meta = json.load(open(os.path.join(d, "meta.json")))
try:
    m = json.load(open(os.path.join(d, "metrics.json")))
    meta["metrics"] = {k: m[k] for k in ("fid", "density", "coverage") if k in m}
except Exception:
    pass
meta["status"] = "done"
meta["nGenerated"] = len([f for f in os.listdir(os.path.join(d, "generated")) if f.endswith(".pickle")])
json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
PY
echo "[web-train] DONE id=$RUN_ID"
