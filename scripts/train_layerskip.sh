#!/usr/bin/env bash
# LayerSkip-style adaptation (CLAUDE.md §9 a): LoRA + early-exit loss so intermediate
# layers become calibrated to the LM head and variable-depth execution keeps quality.
#
# Run this on the GPU (Jetson / Kaggle) AFTER the step-3 gate shows naive early-exit
# collapses. Then set POISE_ADAPTER_PATH to the saved adapter and re-run the gate to
# measure the new (gentler) depth->quality curve.
#
#   pip install peft
#   bash scripts/train_layerskip.sh --steps 500 --corpus /path/to/corpus.txt
#   POISE_ADAPTER_PATH=./data/results/layerskip_adapter bash scripts/bringup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure a (synthetic) corpus exists; for a real run pass --corpus <public-corpus.txt>.
[ -f data/synthetic/ppl_corpus.txt ] || python3 scripts/make_synthetic_data.py >/dev/null 2>&1 || true

exec python3 -m poise.adaptation.layerskip "$@"
