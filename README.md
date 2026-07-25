<div align="center">

<img src="docs/assets/hero.svg" alt="POISE — Power-Optimized Inference via State-aware Execution: an on-device LLM engine that varies per-token transformer depth in response to live hardware physics" width="100%">

<br>

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-158%20passing-2ea44f)](tests/)
[![Off-device](https://img.shields.io/badge/off--device-mock%20mode-2ea44f)](#-off-device-by-design)
[![PyTorch](https://img.shields.io/badge/adaptive%20path-PyTorch%20eager-ee4c2c?logo=pytorch&logoColor=white)](poise/engine/)
[![Control](https://img.shields.io/badge/control-PID%20%2B%20PPO-764abc)](poise/control/)
[![Serving](https://img.shields.io/badge/serving-FastAPI%20%2B%20Prometheus-009688?logo=fastapi&logoColor=white)](poise/serving/)
[![License](https://img.shields.io/badge/license-personal%20IP-lightgrey)](#-license)

*Conditioning per-token compute on the temperature, power, and throttle state of the chip running it.*

</div>

---

## TL;DR

- **What it is** — an on-device LLM engine that runs a *variable number of transformer layers per
  token*, choosing the depth from the chip's live physical state (junction temperature, power,
  throttle) rather than from input difficulty.
- **The question it set out to answer** — when an edge device gets hot, can shedding layers avoid
  thermal throttling and hold throughput?
- **The measured answer** — **no, and that itself is the result.** On a Jetson AGX Orin, depth
  barely moves power (~2 W across 16→32 layers), so it can't cool the chip. But shedding depth
  *does* buy **~2× throughput and ~2× lower energy per token** — at a measured quality cost. POISE
  is an **energy/throughput knob, not a thermal governor.** [Jump to the numbers ↓](#-what-we-measured)

This README reports what the hardware actually did, including the part that refuted the original
hypothesis. Every figure below is an output of the eval harness, with variance where repeated.

---

## 🌡️ The question

Run a fixed-depth LLM on a thermally-constrained edge device under sustained load and the physics
wins: the junction heats up, the hardware **throttles** the clock to protect the silicon, and
throughput **collapses** — exactly when you need it most.

The POISE hypothesis: instead of always executing all 32 layers, execute a **variable number of
layers per token** — a *layer budget* — chosen in real time from the device's measured state, and
shed depth under heat to stay ahead of the throttle cliff.

```
   fixed-depth model                          POISE (hypothesis)
   throughput                                 throughput
   ▲                                          ▲
   │████████▓▓▒▒░░       ← throttle cliff      │████████▓▓▓▓▓▓▓▓   ← shed depth, hold throughput
   │              ░░░░░                        │
   └──────────────────▶ heat                   └──────────────────▶ heat

   The motivating hypothesis — NOT a result. Whether it holds is an empirical
   question, and the answer (below) is measured, not assumed.
```

---

## 🔬 What we measured

> **Setup.** DeepSeek-R1-Distill-Llama-8B (32 layers, fp16) on an NVIDIA Jetson AGX Orin, real
> jtop telemetry. Throughput/energy is ≥3 repeats (std < 0.1 tok/s); task quality is 300 AI2 ARC
> items scored by length-normalized log-likelihood (standard `acc_norm`). Full write-up:
> [`docs/arxiv/paper.md`](docs/arxiv/paper.md).

### The headline: depth is an energy/throughput knob, not a thermal one

Halving the layer count roughly **doubles throughput and halves energy per token** — at a real,
steep accuracy cost:

| depth | tok/s | energy/token | tok/s per watt | ARC accuracy |
| --: | --: | --: | --: | --: |
| **32** (full) | 12.09 | 3.90 J | 0.257 (1.00×) | 0.557 |
| 24 | 15.97 | 2.92 J | 0.342 (1.33×) | 0.423 |
| **16** | 23.95 | 1.90 J | 0.527 (**2.05×**) | 0.347 |

But it **does not cool the chip.** Across single-stream decode, long-context prefill (2048), and
batched forwards (8×512), doubling depth moves power by **≤2 W** while throughput scales ~2× — the
GPU pegs at a power/clock ceiling, so depth only sets *how fast* work clears, not *how much heat* it
makes. Three facts settle it:

- **`static-16` runs hotter than `static-32`** — the shallower model is faster, so it keeps the GPU
  more continuously busy.
- **A PID controller conditioned on junction temperature sheds depth (32 → 21) yet runs the
  hottest of all** — the actuator has no authority over the variable it's trying to control.
- At ~47 W the chip plateaus near 60 °C and **never reaches the ~87 °C throttle point** — there is
  no throttle to dodge.

The RC thermal model itself is accurate (R_th 0.807 K/W, τ 64.8 s, fit RMSE 0.32 °C) — the model
isn't the problem; the mechanism is.

### The quality cost is real (and the adapter helps only a little)

Depth reduction is not free on quality. On ARC (`acc_norm`, v1 LayerSkip adapter): shedding to
depth-28 costs 14% relative, depth-24 costs 24%, depth-16 costs 38% — all far above the ≤2–3% the
project originally hoped for. A LoRA early-exit adapter lifts the whole curve ~2–5 points and
doesn't damage full depth, but opens no "free" reduced-depth band.

<details>
<summary><b>Earlier signal — the distributional quality gate (KL / perplexity)</b></summary>

<br>

Before the task-level measurement, the go/no-go gate compared each depth's output distribution to
the full-32 model over a 301-token set. Naive early-exit collapses; the LoRA adapter recovers ≈3×
in KL but still doesn't clear a strict bar — which is why the honest task metric above matters more.

**Naive early-exit (stock model):**

| depth | mean KL ↓ | perplexity ↓ | top-1 agree ↑ |
| --: | --: | --: | --: |
| 16 | 7.24 | 615,631 | 1.7% |
| 24 | 5.20 | 109,711 | 13.0% |
| 28 | 3.91 | 34,098 | 14.6% |
| 32 | 0.00 | 1,959 | 100% |

**With the LayerSkip LoRA adapter:**

| depth | mean KL ↓ | perplexity ↓ | top-1 agree ↑ | vs. naive |
| --: | --: | --: | --: | :-- |
| 16 | 2.78 | 12,982 | 9.0% | KL −62% |
| 24 | 1.55 | 4,253 | 29.2% | KL −70% |
| 28 | 1.08 | 2,981 | 35.5% | KL −72% |
| 32 | 0.00 | 1,874 | 100% | anchor held |

The KL signal *overstated* the collapse: it called depth-28 "unusable" (KL 1.08), yet that depth's
actual task-accuracy loss is only ~14%.

</details>

### Honest status

The thermal-regulation goal as originally framed is **refuted for single-stream decode on the
Orin** — reported as found, not buried. What holds up, measured with variance: hardware-state
depth control as an **energy/throughput knob** at a fixed operating point, at a quantified quality
cost. PPO against the calibrated simulator was *not* pursued — with flat depth→power the policy has
no thermal signal to learn. The defensible contribution is a **characterization**: when and why
depth control helps (energy/throughput), and why it fails as thermal control on a
power-ceiling-bound accelerator.

---

## 💡 The framing

> ### Hardware-state-conditioned adaptive computation.

Existing adaptive-depth / early-exit methods condition execution depth on **input difficulty**
(token / sequence confidence, content). POISE instead conditions depth on **hardware state** —
closing the loop between live device physics and per-token compute.

| Method | Conditions depth on |
| :--- | :--- |
| CALM · LayerSkip · AdaInfer · DASH · (early-exit literature) | **input difficulty** — confidence / entropy / content |
| **⚡ POISE (this work)** | **hardware state** — junction temp · power · GPU clock · throttle |

The two axes are orthogonal and composable; the hardware-state loop is the angle studied here.
"Adaptive layer skipping" by itself is **not** claimed as novel — that field is crowded; the
question was *what drives the depth decision*. Any "neuromorphic-inspired" analogy refers only to
hardware-state-conditioned variable-depth execution — never a silicon claim.

---

## 🧠 How it works — a two-tier controller

```
                    ┌──────────────────── telemetry/ ─────────────────────┐
                    │  jtop · tegrastats · mock  →  TelemetrySample        │
                    │  (temp · power · clock · util · throttled)           │
                    └─────────────────────────┬───────────────────────────┘
                                              │  live hardware state
                                              ▼
        control/  (POLICY)            ┌────────────────┐         engine/  (MECHANISM)
   ┌────────────────────────┐         │ BudgetAllocator│ budget  ┌───────────────────────────┐
   │  PID  +  PPO policy     │ ──────▶ │ static·pid·ppo │ ──────▶ │ adaptive_runner            │
   │  (obs == training obs)  │         │  🔒 TEMP_MAX    │ /token  │  layers 0..budget-1        │
   └────────────────────────┘         │   override      │         │  → early_exit → logits     │
              ▲                        └────────────────┘         │  + variable-depth KV cache │
              │ trained off-device                                └──────────────┬────────────┘
              │ against the simulator                                            │ tokens + per-token trace
   ┌──────────┴───────────────────────────────────┐                             ▼
   │  rl/  RC simulator (calibrated) + Gym env +   │                  eval/ · serving/ · storage/
   │  reward (quality from the step-3 gate) + PPO  │
   └───────────────────────────────────────────────┘
```

- **PID feedback controller** — reactively maps a thermal error signal to a layer budget. (Built
  and run on-board; its measured behavior is in [What we measured](#-what-we-measured).)
- **PPO policy** — the anticipatory tier, designed to act ahead of throttling. *Not pursued*: the
  flat depth→power finding leaves it no thermal signal to learn.

🔒 **The engine is mechanism, `control/` is policy** — the runner contains no control logic, and
thermal safety (`TEMP_MAX`) overrides throughput, always.

---

## ✨ At a glance

- 🎚️ **Per-token variable depth** — clean per-layer loop with early-exit projection.
- 🌡️ **Closed on physics** — depth follows temperature / power / throttle, not content.
- 🧩 **Honest KV cache** — three documented strategies; constant-budget output is bit-identical.
- 🚦 **Two-tier control** — reactive PID + anticipatory PPO, with a hard thermal-safety override.
- 🔬 **Measured before claimed** — the depth→quality cost is profiled before the controller is built.
- 🖥️ **Live dashboard + API** — FastAPI (`/v1/*`) + Prometheus + a React/Recharts UI.
- 🌐 **Universal** — auto-detects cuda/mps/cpu + jtop/nvml/mock; depth adapts to any model.
- 🧪 **Fully testable off-device** — 158 tests green with no GPU, no model, no board.

---

## 🖥️ The dashboard

The React/Recharts UI is a **thermal-governor instrument**: a single accent colour is interpolated
from live junction temperature, so the gauge, the depth-budget ladder, and the page itself *warm*
as the chip heats. The junction gauge and the depth ladder move in opposition — heat rises on the
left, depth sheds on the right — making the design intent (`temp ↑ ⇒ depth ↓`) legible at a glance.

<div align="center">

<img src="docs/assets/dashboard.svg" alt="POISE Thermal Governor dashboard: a junction-temperature gauge at 82.5°C over the 80° setpoint, a depth-budget ladder with the 20-layer rung active, live telemetry readouts, and temperature/budget traces" width="100%">

<sub><b>Vector rendering</b> of the live dashboard, captured at a "governor holding over setpoint" moment (PID mode, depth shed to 20). Derived from the app's own thermochromic logic — not a mockup, not measured performance.</sub>

</div>

> **Run it:** `bash scripts/serve.sh` (FastAPI on :8000), then `cd dashboard && npm run dev`
> (Vite on :5173). Off-device it serves a mock engine that reuses the real control loop, so the
> dashboard works without weights.

---

## 🚀 Quickstart

```bash
# 1. Install the lightweight core (no torch — runs fully off-device)
pip install -r requirements.txt

# 2. Prove it works in mock mode
pytest                                  # green, no board required

# 3. Universal bring-up — auto-detects device + telemetry, runs profile/deps/gate/eval.
#    Works on Jetson, NVIDIA GPU box, Apple Silicon, or CPU-only.
bash scripts/bringup.sh                              # uses your configured model
bash scripts/bringup.sh --model sshleifer/tiny-gpt2  # validate the WHOLE pipeline on CPU

# 4. Serve the API + live dashboard (real engine on-device, mock engine off-device)
bash scripts/serve.sh                   # FastAPI on :8000, Prometheus at /metrics
cd dashboard && npm install && npm run dev   # → http://localhost:5173

# For on-device use, copy and fill in credentials (never committed):
cp .env.example .env                    # HF_TOKEN; POISE_DEVICE/TELEMETRY stay 'auto'
```

> **Universal:** `POISE_DEVICE=auto` resolves cuda → mps → cpu; `POISE_TELEMETRY_BACKEND=auto`
> picks jtop → nvml → mock; and the layer-budget config rescales to **any** decoder-only model's
> actual layer count. See [`docs/bringup.md`](docs/bringup.md).

---

## 🧊 Off-device by design

Every hardware-touching module ships a **mock path**, so the whole stack is developable, testable,
and demoable without the Jetson:

| Concern | On the board | Off-device (laptop / CI / Kaggle) |
| :--- | :--- | :--- |
| Device | `auto` → cuda (Jetson/GPU) / mps | `auto` → cpu |
| Telemetry | `auto` → jtop / nvml / tegrastats | `auto` → mock thermal curve |
| Model | any HF decoder-only (depth auto-adapts) | tiny model on CPU, or mock engine |
| Thermal model | calibrated RC params | clearly-labeled **placeholder** params |
| RL training | — | RC simulator + Gymnasium env (no model) |
| Serving / dashboard | real `AdaptiveRunner` | mock engine reusing the **real** control loop |
| Heavy deps (`torch`, `transformers`, `faiss`, `llama.cpp`) | installed | optional extras |

---

## 🗺️ Build order

Built in dependency order (module map in [`docs/architecture.md`](docs/architecture.md)).
**Step 3 is a go/no-go gate.**

| | Stage | | | Stage |
| :-- | :-- | :-- | :-- | :-- |
| 0 | Scaffold · config · storage | | 8 | PPO training (off-device) |
| 1 | Telemetry (+ mock) | | 9 | PPO on-board validation |
| 2 | Engine: load + early-exit | | 10 | Fixed-depth baselines |
| **3** | **🚧 GATE — depth→quality profile** | | 11 | Eval harness + variance report |
| 4 | Adaptive runner + KV cache | | 12 | Serving + metrics + dashboard |
| 5 | Calibration → RC params | | 13 | RAG demo (synthetic only) |
| 6 | Simulator + env + reward | | 14 | Repro packaging |
| 7 | PID + budget allocator | | | |

> 🚧 **The gate (step 3)** measures the depth→quality cost *before* the controller is built. The
> ≤2–3% quality-loss target was a **hypothesis to be measured, not an assumption** — see
> [What we measured](#-what-we-measured) for how it turned out and
> [`docs/novelty.md`](docs/novelty.md) for the prior-art positioning.

---

## 📟 Run-book

<details>
<summary><b>Calibration → training → benchmark → serve</b> (click to expand)</summary>

```bash
# Calibration sweep (on the Jetson) → fits RC params + depth→power map
bash scripts/run_calibration.sh

# PPO training (off-device against the simulator)
#   → trains the policy, sweeps reward weights, reports the converged depth distribution

# Stress benchmark / headline comparison (static-full-32 · static×2 · pid · ppo)
bash scripts/run_benchmark.sh

# Serve the API + Prometheus metrics
bash scripts/serve.sh                 # /health  /v1/generate  /v1/telemetry  /metrics

# Live dashboard — temp / power / current-budget / tok-s + per-token budget trace
cd dashboard && npm install && npm run dev
```

</details>

---

## 📂 Repository layout

```
poise/
├── telemetry/     # TelemetrySample + jtop/tegrastats readers + mock
├── engine/        # model_loader · early_exit · adaptive_runner · kv_cache   (mechanism)
├── control/       # pid · budget_allocator · policy                           (policy)
├── rl/            # simulator · env · reward · domain_random · train_ppo
├── calibration/   # quality_profile (the GATE) · driver · sweep · fit
├── baselines/     # static_llamacpp (Q4_K_M) · static_trtllm   (fixed-depth only)
├── eval/          # benchmark · quality · report   (the only place numbers are produced)
├── serving/       # FastAPI api · routes · Prometheus metrics
├── storage/       # SQLite schema + migrations + accessors
└── rag/           # synthetic FAISS index + LangGraph demo   (engine as a black box)
```

See [`docs/architecture.md`](docs/architecture.md) for the data-flow diagram and full module map,
and [`docs/novelty.md`](docs/novelty.md) for the prior-art positioning.

---

## 🔒 Governance

- **Personal portfolio / research project.** Personal IP ownership of all code.
- **No company / proprietary data anywhere.** The RAG demo uses only public or synthetic data — a
  synthetic-doc generator ships with the repo.
- **No secrets in the repo.** All tokens / keys via `.env` (git-ignored).
- **No fabricated results.** Calibration params stay labeled *placeholder* until fit from real
  data; headline numbers exist only as eval outputs with variance.

---

## 📜 License

Proprietary — personal IP. All rights reserved.

<div align="center">
<sub>POISE — depth follows physics.</sub>
</div>
