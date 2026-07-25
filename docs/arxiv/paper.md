# POISE: Hardware-State-Conditioned Variable-Depth Inference on the Edge — A Characterization

*Measured characterization. Every number below is an output of the eval harness on the real
device (DeepSeek-R1-Distill-Llama-8B, 32 layers, fp16, NVIDIA Jetson AGX Orin, MAXN), reported
with variance where repeated. Results that refute the project's initial hypothesis are reported
as found (CLAUDE.md §9).*

## Abstract

We set out to build POISE, an on-device inference orchestrator that varies per-token transformer
depth in response to **live hardware physics** — junction temperature, power, throttle status —
to hold throughput and cut energy under sustained thermal load where a fixed-depth model
throttles. We calibrated a first-order RC thermal model on the board, measured the per-depth
quality cost on a real task benchmark, and ran the closed control loop on the device. **Our
central empirical finding refutes the thermal premise for this platform:** across single-stream
decode, long-context prefill, and batched inference, transformer depth changes power by ≤2 W
over 16→32 layers while throughput scales ~2×. Depth is therefore an **energy/throughput knob,
not a thermal actuator** — a PID controller conditioned on junction temperature correctly sheds
depth but *cannot* lower temperature, and at ~47 W the chip never reaches its throttle point at
all. We report what hardware-state-conditioned depth control *does* deliver — ~2× throughput and
~2× lower energy-per-token at a measured quality cost — and characterize precisely why the
thermal-regulation mechanism fails on a memory-/power-ceiling-bound accelerator.

## 1. Introduction

Edge LLM inference is widely assumed to be thermally constrained. The POISE hypothesis was that
by shedding transformer layers when the device gets hot, an inference engine could avoid
throttling and hold throughput. We built the full apparatus to test this — variable-depth engine,
on-board RC calibration, a PID controller, and an eval harness — and measured the mechanism end
to end. This paper reports the measurements, including the ones that contradict the hypothesis.

**Contribution framing.** We do not claim adaptive depth itself as novel (CALM, LayerSkip,
AdaInfer, DASH condition depth on input difficulty). Our intended contribution was the
*hardware-state* loop. What we can honestly contribute is a **characterization**: when and why
hardware-state-conditioned depth control helps (energy/throughput) and why it fails as thermal
control for bandwidth-/power-bound decode on the Jetson Orin.

## 2. Related work

| Family | Depth conditioned on |
|---|---|
| CALM, LayerSkip, AdaInfer, DASH, early-exit | input difficulty (confidence/content) |
| **POISE (ours)** | **hardware state (temp/power/throttle)** |

The axes are orthogonal and composable; we studied the hardware-state loop.

## 3. Method

- **Variable-depth execution.** Per-token forward through layers `0..budget-1`, projected to
  logits via the model's final norm + LM head (`engine/early_exit`). Intermediate states are
  uncalibrated to the head — quantified by the step-3 gate.
- **On-board calibration.** A load driver (`calibration/driver.py`) drives sustained fixed-depth
  inference while sampling telemetry, producing one heating trace per depth; a first-order RC
  model (`dT/dt = (P − (T−T_amb)/R_th)/C_th`) is fit from the warm-up curves.
- **Two-tier control.** A reactive PID mapping thermal error → budget, plus an (unbuilt-for-this-
  report) PPO policy. An unconditional `TEMP_MAX` safety override.

## 4. Experimental setup

- **Model:** DeepSeek-R1-Distill-Llama-8B (Llama-3.1, 32 layers), fp16, eager attention.
- **Device:** NVIDIA Jetson AGX Orin, MAXN power mode, jtop telemetry.
- **Quality:** per-depth multiple-choice accuracy on 300 AI2 ARC items (150 easy + 150
  challenge) via length-normalized log-likelihood scoring (standard `acc_norm`); KL/perplexity/
  top-1 vs full-32.
- **Throughput/energy/thermal:** sustained generation with real telemetry; energy-per-token =
  ∫P dt ÷ tokens; steady-state tails; ≥3 repeats for the depth sweep.

## 5. Results

### 5.1 RC thermal calibration (holds)
Real on-board calibration fits the first-order model well: **R_th = 0.807 K/W, C_th = 87.6 J/K,
τ = 64.8 s, fit RMSE = 0.32 °C**. The thermal model is accurate; the problem is not the model.

### 5.2 Depth→power is flat across ALL regimes (central finding)
Steady-state power vs depth, by workload regime:

| regime | P(depth-16) | P(depth-32) | ΔP (16→32) | slope |
|---|---:|---:|---:|---:|
| decode (single stream) | ~46 W | ~48 W | ~2 W | flat |
| prefill (seq 2048) | 50.1 W | 51.3 W | 1.2 W | 0.08 W/layer |
| batched (8×512) | 58.8 W | 59.6 W | 0.9 W | 0.05 W/layer |

Doubling the layer count moves power by ≤2 W in every regime, while throughput falls ~2×. At
MAXN the module runs at a power/clock ceiling for any heavy workload, so additional layers take
proportionally longer at the *same* power. **Depth does not modulate heat.**

### 5.3 Throughput and energy scale with depth (the real benefit)
MAXN, 3 repeats, std < 0.1 tok/s:

| depth | tok/s | J/token | tok/s per watt | ×vs full |
|---:|---:|---:|---:|---:|
| 32 | 12.09 | 3.896 | 0.257 | 1.00× |
| 28 | 13.80 | 3.413 | 0.293 | 1.14× |
| 24 | 15.97 | 2.924 | 0.342 | 1.33× |
| 16 | 23.95 | 1.898 | 0.527 | **2.05×** |

Halving depth ≈ 2× throughput and 2× lower energy-per-token (tok/s·W⁻¹ = 1 / J·token⁻¹).

### 5.4 Quality cost of depth reduction (real, steep)
Per-depth ARC accuracy (`acc_norm`). The LayerSkip LoRA adapter (v1) does not damage full depth
and lifts the whole curve ~2–5 points, but opens no free band:

| depth | base | v1 adapter | Δ vs full-32 (adapter) |
|---:|---:|---:|---:|
| 32 | 0.527 | 0.557 | — |
| 28 | 0.457 | 0.477 | −8.0 pts (−14%) |
| 24 | 0.430 | 0.423 | −13.3 pts (−24%) |
| 16 | 0.297 | 0.347 | −21.0 pts (−38%) |

Even shedding 2 layers (32→30) costs ~7 pts / 12.6% relative — far above the initial ≤2–3%
hypothesis. (Task loss is nonetheless *milder* than the KL signal implied: KL called depth-28
"unusable" at 1.08 nats, yet its task loss is only ~14% relative.)

### 5.5 On-board closed loop: depth is not a thermal actuator (refutation)
Sustained load, PID setpoint 50 °C, 130 s per mode, normal cooling:

| mode | tok/s | peak tj | mean depth | J/token |
|---|---:|---:|---:|---:|
| static-32 | 11.74 | 60.8 °C | 32 | 4.07 |
| static-16 | 23.23 | **62.2 °C** | 16 | 2.04 |
| PID | 18.74 | **62.7 °C** | 21 | 2.55 |

Three facts settle the hypothesis: (i) **static-16 runs hotter than static-32** despite half the
depth, because 2× throughput keeps the GPU more continuously active; (ii) **PID correctly sheds
depth (32→21)** when tj exceeds setpoint yet runs the **hottest** of the three — the actuator has
no authority over the controlled variable; (iii) at ~47 W the chip plateaus near 59–63 °C and
never approaches the ~87 °C passive-throttle point, so there is no throttle to avoid.

## 6. Discussion

Hardware-state-conditioned depth control on the Jetson Orin is an **energy-per-token / throughput
knob**, not a thermal governor. The clean way to state the value: *at a fixed operating point,
reducing depth yields ~linear throughput and energy gains at a measured, non-trivial quality
cost* — a controller can pick the depth that meets a latency/energy target for the current
workload. It cannot regulate temperature, because on this class of accelerator power is bound by
a memory/clock/power ceiling that depth does not move. This is a property of the
hardware+workload, not of the controller: the same conclusion holds across decode, prefill, and
batched inference. Whether any accelerator/workload exists where depth *does* move power enough to
regulate temperature remains open, but it is not this platform in any regime we tested.

## 7. Limitations & honesty

- The thermal-regulation objective as originally framed is **not supported** by our measurements;
  we report it as refuted rather than narrowing silently (CLAUDE.md §9).
- ARC via MC-likelihood underrates a chain-of-thought model in absolute terms; we rely on the
  per-depth deltas, which are internally consistent.
- Single device (one AGX Orin), MAXN mode; a hard nvpmodel power cap requires a reboot and, given
  flat depth→power, would not create differential thermal benefit.
- "neuromorphic-inspired," if used anywhere, refers only to hardware-state-conditioned
  variable-depth execution — never a silicon claim.

## 8. Reproducibility

Calibration: `python3.10 -m poise.calibration.driver`. Quality gate: `scripts/prong1_task_gate.py`
(+ `scripts/build_arc_suite.py`). Throughput/energy: `scripts/depth_bench.py`. Closed-loop stress:
`scripts/thermal_stress.py`. Compute-bound probe: `scripts/compute_bound_probe.py`. Results JSON
under `data/results/`. Interpreter: python3.10 (JetPack stack).
