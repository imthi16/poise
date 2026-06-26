#!/usr/bin/env bash
# Stress benchmark / headline comparison (CLAUDE.md §5 step 11). ON-DEVICE for real
# numbers; runs against the mock engine off-device for a pipeline smoke test.
#
# Every headline number is produced by eval/report.py with variance across seeds —
# nothing is asserted ahead of measurement.
set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT="${1:-Explain thermal throttling in one paragraph.}"
MAX_TOKENS="${2:-512}"

# Ensure synthetic eval data exists (synthetic/public only).
[ -f data/synthetic/eval_prompts.jsonl ] || python3 scripts/make_synthetic_data.py

echo "[POISE] stress benchmark: max_new_tokens=${MAX_TOKENS}"
python3 -m poise.eval.benchmark --prompt "${PROMPT}" --max-new-tokens "${MAX_TOKENS}"

echo "[POISE] aggregate a multi-seed comparison (static-full-32 / static x2 / pid / ppo)"
echo "        and emit tables+plots via poise.eval.report.emit_report(...)."
