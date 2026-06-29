#!/usr/bin/env bash
# Diagnose the POISE environment and print the version-matched fix (e.g. the correct
# JetPack torch wheel on a Jetson). Run this first if bring-up resolves to CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m poise.doctor
