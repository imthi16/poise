# POISE: Hardware-State-Conditioned Variable-Depth Inference on the Edge

*Paper skeleton — keep in sync with the build. No performance number appears here until
`eval/report.py` has produced it with variance (CLAUDE.md §9).*

## Abstract

We present POISE, an on-device inference orchestrator that varies per-token transformer
depth in response to **live hardware physics** — junction temperature, power draw, GPU
clock, and throttle status — on an NVIDIA Jetson AGX Orin. Unlike input-difficulty
adaptive-depth methods (CALM, LayerSkip, AdaInfer, DASH), POISE closes the loop on
**hardware state**. A two-tier controller (a reactive PID and an anticipatory PPO policy
trained off-device against a calibrated RC thermal simulator) trades a small, measured
amount of quality to hold target throughput and cut energy-per-token under sustained
thermal load where a fixed-depth model throttles and collapses. *[All quantitative
results are pending the eval harness and will be reported with variance.]*

## 1. Introduction

- Edge LLM inference is thermally constrained; fixed-depth models throttle under load.
- Contribution: **hardware-state-conditioned adaptive computation** (the depth decision
  is driven by device physics, not content). We do not claim adaptive depth itself as
  novel.

## 2. Related work

| Family | Depth conditioned on |
|---|---|
| CALM, LayerSkip, AdaInfer, DASH, early-exit | input difficulty (confidence/content) |
| **POISE (ours)** | **hardware state (temp/power/throttle)** |

The axes are orthogonal and composable; we contribute the hardware-state loop.

## 3. Method

### 3.1 Variable-depth execution
Per-token forward through layers `0..budget-1`, exiting via the LM head
(`engine/early_exit`). ⚠ Intermediate states are uncalibrated to the head — §5 gate.

### 3.2 KV-cache consistency (⚠)
Variable depth creates KV holes; we implement and **measure** three strategies
(recompute-on-demand, propagate-hidden, monotone-non-increasing). Constant-budget output
is bit-identical to standard decoding.

### 3.3 Two-tier control
PID (reactive; lags slow thermal dynamics) + PPO policy (anticipatory), with an
unconditional `TEMP_MAX` safety override.

### 3.4 Calibrated simulator + off-device RL
First-order RC thermal model fit from on-board sweeps; depth→power map; PPO trains
against it with domain randomization; quality term from the measured depth→KL table.

## 4. Experimental setup

- Model: 32-layer decoder-only LLM (fp16 adaptive path; Q4_K_M GGUF as the llama.cpp
  baseline only).
- Stress protocol: sustained generation to thermal steady state (`eval/benchmark`).
- Comparison set: static-full-32, ≥2 static reduced depths, PID, PPO.
- Metrics: tok/s, TTFT, energy/token, peak temp, time-above-setpoint, throttle count,
  mean budget; quality via mean KL vs full-32, perplexity, task accuracy.

## 5. Results (the step-3 gate first)

> **Pending measurement.** The depth→quality table (gate) determines whether the
> ≤2–3% quality-loss hypothesis holds and what the usable `LAYER_MIN` is. Headline
> energy/quality/throughput comparisons are emitted by `eval/report.py` with spread
> across ≥5 seeds. Nothing is asserted here ahead of those outputs.

## 6. Limitations

- Static depth→quality mapping is an approximation; on-board validation uses true
  per-token KL.
- First-order RC model; domain randomization mitigates but does not eliminate the
  sim-to-real gap, which we quantify on the board.
- "neuromorphic-inspired" (if used) refers only to hardware-state-conditioned
  variable-depth execution — never a silicon claim.

## 7. Reproducibility

Seeds, config snapshots (`runs.config_json`, `ppo_checkpoints.*_json`), and the
run-book (`README.md`) reproduce calibration → (Kaggle) training → on-board validation →
eval.
