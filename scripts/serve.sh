#!/usr/bin/env bash
# Serve the POISE API + Prometheus metrics (CLAUDE.md §5 step 12, §7).
# On the Jetson this loads the real model; off-device it serves the mock engine so the
# dashboard works without weights.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${POISE_API_HOST:-0.0.0.0}"
PORT="${POISE_API_PORT:-8000}"

# Ensure synthetic RAG corpus exists for the /v1/rag/query demo (synthetic only).
[ -f data/synthetic/corpus.jsonl ] || python3 scripts/make_synthetic_data.py

echo "[POISE] serving on ${HOST}:${PORT}  (/health /v1/* /metrics)"
exec uvicorn "poise.serving.api:create_app" --factory --host "${HOST}" --port "${PORT}"
