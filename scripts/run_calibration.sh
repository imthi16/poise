#!/usr/bin/env bash
# Thermal-power calibration sweep (CLAUDE.md §5 step 5). ON-DEVICE (Jetson).
#
# Drives fixed-depth load across power modes and records telemetry, then fits the RC
# params + depth->power map and writes them into configs/simulator.yaml (calibrated).
#
# Off-device this produces clearly-labeled SYNTHETIC traces (mock backend) — useful
# only to exercise the sweep->fit pipeline, never as real calibration.
set -euo pipefail
cd "$(dirname "$0")/.."

DURATION="${1:-120}"
MODES="${2:-MAXN}"

echo "[POISE] calibration sweep: duration=${DURATION}s modes=${MODES}"
python3 -m poise.calibration.sweep --duration "${DURATION}" --modes "${MODES}" --out data/calibration

echo "[POISE] fit RC params from the sweeps with:"
echo "    python3 -c 'from poise.calibration import fit; ...'  # see calibration/fit.py"
echo "[POISE] (fit writes configs/simulator.yaml only from REAL on-board data)."
