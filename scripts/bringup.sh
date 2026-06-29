#!/usr/bin/env bash
# Universal, turnkey POISE bring-up (CLAUDE.md §5). Works on Jetson, a generic NVIDIA
# GPU box, Apple Silicon, or CPU-only. Auto-detects device + telemetry and adapts the
# depth config to whatever model is loaded.
#
#   bash scripts/bringup.sh                         # profile + deps + model + GATE + eval
#   bash scripts/bringup.sh --model sshleifer/tiny-gpt2   # full pipeline on a CPU laptop
#   bash scripts/bringup.sh --steps profile,deps          # just inspect the machine
#
# The step-3 GATE is the go/no-go: it measures depth->quality on the real model and
# prints the usable LAYER_MIN. Nothing downstream is trusted before it runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure synthetic prompts exist for the gate / eval (synthetic/public only).
[ -f data/synthetic/eval_prompts.jsonl ] || python3 scripts/make_synthetic_data.py >/dev/null 2>&1 || true

exec python3 -m poise.bringup "$@"
