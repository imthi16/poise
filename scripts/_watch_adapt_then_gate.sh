#!/usr/bin/env bash
# Wait for the running LayerSkip adaptation (PID $1) to finish, then auto-run the
# quality gate with the new adapter so the depth->KL curve is ready without babysitting.
set -uo pipefail
cd "$(dirname "$0")/.."
PID="$1"
ADAPTER="${2:-./data/results/layerskip_adapter_v2}"

echo "[watch] waiting for adaptation PID $PID ..."
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[watch] adaptation process exited at $(date -u +%H:%M:%S)UTC"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[watch] ERROR: no adapter saved at $ADAPTER — training likely failed; skipping gate."
  exit 1
fi

echo "[watch] re-running gate with $ADAPTER ..."
POISE_DEVICE=cuda POISE_TELEMETRY_BACKEND=jtop POISE_ADAPTER_PATH="$ADAPTER" \
  python3.10 -u -m poise.bringup --steps model,gate --device cuda
echo "[watch] gate complete at $(date -u +%H:%M:%S)UTC"
