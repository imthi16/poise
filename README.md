<div align="center">

# ⚡ POISE

### Power-Optimized Inference via State-aware Execution

**An on-device LLM engine that varies per-token transformer depth in response to live hardware physics.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-158%20passing-2ea44f)](tests/)
[![Off-device](https://img.shields.io/badge/off--device-mock%20mode-2ea44f)](#-off-device-by-design)
[![PyTorch](https://img.shields.io/badge/adaptive%20path-PyTorch%20eager-ee4c2c?logo=pytorch&logoColor=white)](poise/engine/)
[![RL](https://img.shields.io/badge/control-PID%20%2B%20PPO-764abc)](poise/control/)
[![Serving](https://img.shields.io/badge/serving-FastAPI%20%2B%20Prometheus-009688?logo=fastapi&logoColor=white)](poise/serving/)
[![License](https://img.shields.io/badge/license-personal%20IP-lightgrey)](#-license)

*Conditioning per-token compute on the temperature, power, and throttle state of the chip running it.*

</div>

---

## 🌡️ The problem

Run a fixed-depth LLM on a thermally-constrained edge device under sustained load and the
physics wins: the junction heats up, the hardware **throttles** the clock to protect the
silicon, and throughput **collapses** — exactly when you need it most.

POISE refuses that cliff. Instead of always executing all 32 transformer layers, it executes a
**variable number of layers per token** — a *layer budget* — chosen in real time from the
device's measured physical state. Under heat it **gracefully sheds depth** to hold a target
tokens/sec and cut energy-per-token, trading a small, **measured** amount of quality rather than
falling off a throttling cliff.

```
   fixed-depth model                          POISE (design goal)
   throughput                                 throughput
   ▲                                          ▲
   │████████▓▓▒▒░░       ← throttle cliff      │████████▓▓▓▓▓▓▓▓   ← depth shed gracefully
   │              ░░░░░                        │
   └──────────────────▶ heat                   └──────────────────▶ heat

   Schematic of the motivating problem + the design goal — NOT measured results.
   Whether POISE achieves this (and at what quality cost) is an empirical question
   answered only by the eval harness.
```

---

## 💡 The contribution

> ### Hardware-state-conditioned adaptive computation.

Existing adaptive-depth / early-exit methods condition execution depth on **input difficulty**
(token / sequence confidence, content). POISE instead conditions execution depth on **hardware
state** — it closes the loop between live device physics and per-token compute.

| Method | Conditions depth on |
| :--- | :--- |
| CALM · LayerSkip · AdaInfer · DASH · (early-exit literature) | **input difficulty** — confidence / entropy / content |
| **⚡ POISE (this work)** | **hardware state** — junction temp · power · GPU clock · throttle |

The two axes are orthogonal and composable; the **hardware-state loop is the contribution.**

- "Adaptive layer skipping" by itself is **not** claimed as novel — that field is crowded. The
  novelty is *what drives the depth decision*.
- The mechanism is literally **hardware-state-conditioned variable-depth execution**. Any
  "neuromorphic-inspired" analogy, if used at all, refers to exactly that mechanism — never a
  hardware/silicon claim.

> [!IMPORTANT]
> **No performance numbers appear in this README, the docs, or the paper until
> `poise/eval/report.py` has produced them with variance across multiple runs.** Every headline
> figure (energy %, quality %, throughput) is a measured output of the eval harness, reported
> with spread — never asserted ahead of measurement.

---

## 🔬 Measured — a characterization

> **Scope of this evidence.** All numbers below are eval-harness outputs on the real model
> (DeepSeek-R1-Distill-Llama-8B, fp16) on an NVIDIA Jetson AGX Orin, reported with variance where
> repeated. This section covers the go/no-go quality gate **and** the on-device throughput/energy/
> thermal characterization. Findings that refute the project's initial hypothesis are reported as
> found (CLAUDE.md §9); see [Honest status](#still-pending).

**Setup.** DeepSeek-R1-Distill-Llama-8B (32 layers, fp16) on an NVIDIA Jetson AGX Orin, 2026-07-24.
For each candidate depth *d*, we measure depth-*d* output vs the full-32 output over a 301-token
calibration set: **mean KL**, **perplexity**, and **top-1 agreement**. Reproducible with
`python3.10 -m poise.bringup --steps model,gate`; every row is persisted to the `quality_profile`
table in `data/results/poise.db`.

**Finding 1 — naive early-exit collapses.** On the stock model, dropping even 4 of 32 layers
destroys agreement with the full model. There is no "free" reduced depth: the usable `LAYER_MIN`
at KL ≤ 0.1 is 32. *This is the quality-collapse CLAUDE.md §9 warns about — measured, not assumed.*

| depth | mean KL ↓ | perplexity ↓ | top-1 agree ↑ |
| --: | --: | --: | --: |
| 16 | 7.24 | 615,631 | 1.7% |
| 20 | 6.30 | 278,734 | 7.6% |
| 24 | 5.20 | 109,711 | 13.0% |
| 28 | 3.91 | 34,098 | 14.6% |
| 32 | 0.00 | 1,959 | 100% |

**Finding 2 — LayerSkip adaptation recovers ≈3×.** A LoRA early-exit / self-distillation adapter
(CLAUDE.md §9 option a) re-calibrates the intermediate layers to the LM head, cutting KL roughly
threefold across the band while the full-depth anchor holds (depth-32 perplexity 1,959 → 1,874):

| depth | mean KL ↓ | perplexity ↓ | top-1 agree ↑ | vs. naive |
| --: | --: | --: | --: | :-- |
| 16 | 2.78 | 12,982 | 9.0% | KL −62% |
| 20 | 2.06 | 6,734 | 15.6% | KL −67% |
| 24 | 1.55 | 4,253 | 29.2% | KL −70% |
| 28 | 1.08 | 2,981 | 35.5% | KL −72% |
| 32 | 0.00 | 1,874 | 100% | anchor held |

**Finding 3 — task-level cost (ARC).** On 300 AI2 ARC items (length-normalized log-likelihood, the
standard `acc_norm`), depth reduction has a real but *milder* cost than KL implied — even 4 shed
layers stay well above the ≤2–3% hypothesis. v1 adapter, `acc_norm`: 32 → 0.557, 28 → 0.477
(−14%), 24 → 0.423 (−24%), 16 → 0.347 (−38%). The adapter does not damage full depth.

**Finding 4 — depth is an energy/throughput knob, not a thermal one (the decisive result).**
Real on-board RC calibration holds (R_th 0.807 K/W, τ 64.8 s, fit RMSE 0.32 °C), but the thermal
premise does **not**. Measured on the Jetson (≥3 repeats, std < 0.1 tok/s):

| depth | tok/s | J/token | tok/s·W⁻¹ | ARC acc |
| --: | --: | --: | --: | --: |
| 32 | 12.09 | 3.896 | 0.257 (1.00×) | 0.557 |
| 24 | 15.97 | 2.924 | 0.342 (1.33×) | 0.423 |
| 16 | 23.95 | 1.898 | 0.527 (**2.05×**) | 0.347 |

Halving depth buys ~2× throughput and ~2× lower energy/token — but **it does not cool the chip.**
Across decode, prefill (2048), and batched (8×512) forwards, doubling depth moves power by ≤2 W
while throughput scales ~2× (the GPU pegs at a power/clock ceiling; depth only sets how fast work
clears). In the closed-loop stress test, `static-16` runs *hotter* than `static-32` (higher
throughput keeps the GPU busier), and a PID conditioned on junction temp correctly sheds depth
(32→21) yet runs the **hottest** of all — the actuator has no thermal authority. At ~47 W the chip
plateaus ~60 °C and never reaches the ~87 °C throttle point. Full write-up: [`docs/arxiv/paper.md`](docs/arxiv/paper.md).

<a name="still-pending"></a>
**Honest status.** The thermal-regulation objective as originally framed is **refuted for
single-stream decode on the Orin** (reported as found — CLAUDE.md §9). What holds up, measured with
variance, is hardware-state-conditioned depth as an **energy/throughput knob** at a fixed operating
point, at a quantified quality cost. PPO against the calibrated simulator is *not* pursued: with
flat depth→power the policy would have no thermal signal to learn. The defensible contribution is a
**characterization** — when/why depth control helps (energy/throughput) and why it fails as thermal
control on a power-ceiling-bound accelerator.

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

1. **PID feedback controller** — reactively maps a thermal error signal to a layer budget. PID
   lags slow thermal dynamics; that lag is the *motivation* for the second tier, not a bug.
2. **PPO-trained policy** — given the full hardware-state vector, learns an *anticipatory* budget
   allocation that acts ahead of throttling. Trained off-device against a calibrated RC thermal
   simulator with domain randomization, then validated on the real board.

🔒 **The engine is mechanism, `control/` is policy** — the runner contains no control logic, and
thermal safety (`TEMP_MAX`) overrides throughput, always.

---

## ✨ At a glance

- 🎚️ **Per-token variable depth** — clean per-layer loop with early-exit projection.
- 🌡️ **Closed on physics** — depth follows temperature / power / throttle, not content.
- 🧩 **Honest KV cache** — three documented strategies; constant-budget output is bit-identical.
- 🚦 **Two-tier control** — reactive PID + anticipatory PPO, with a hard thermal-safety override.
- 🔬 **A go/no-go quality gate** — depth→quality cost is *measured before* the controller is built.
- 🖥️ **Live dashboard + API** — FastAPI (`/v1/*`) + Prometheus + a React/Recharts UI.
- 🌐 **Universal** — auto-detects cuda/mps/cpu + jtop/nvml/mock; depth adapts to any model.
- 🧪 **Fully testable off-device** — 158 tests green with no GPU, no model, no board.

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
> picks jtop → nvml → mock; and the layer-budget config rescales to **any** decoder-only
> model's actual layer count. See [`docs/bringup.md`](docs/bringup.md).

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

Built in dependency order (full spec in [`CLAUDE.md`](CLAUDE.md) §5). **Step 3 is a go/no-go gate.**

| | Stage | | | Stage |
| :-- | :-- | :-- | :-- | :-- |
| 0 | Scaffold · config · storage | | 8 | PPO training (Kaggle) |
| 1 | Telemetry (+ mock) | | 9 | PPO on-board validation |
| 2 | Engine: load + early-exit | | 10 | Fixed-depth baselines |
| **3** | **🚧 GATE — depth→quality profile** | | 11 | Eval harness + variance report |
| 4 | Adaptive runner + KV cache | | 12 | Serving + metrics + dashboard |
| 5 | Calibration → RC params | | 13 | RAG demo (synthetic only) |
| 6 | Simulator + env + reward | | 14 | Repro packaging |
| 7 | PID + budget allocator | | | |

> 🚧 **The gate (step 3)** measures the depth→quality cost *before* the controller is built. The
> ≤2–3% quality-loss target is a **hypothesis to be measured, not an assumption** — see
> [`docs/novelty.md`](docs/novelty.md) and `CLAUDE.md` §9 for the honest options if it doesn't hold.

---

## 📟 Run-book

<details>
<summary><b>Calibration → training → benchmark → serve</b> (click to expand)</summary>

```bash
# Calibration sweep (on the Jetson) → fits RC params + depth→power map
bash scripts/run_calibration.sh

# PPO training (Kaggle, off-device against the simulator)
#   open scripts/train_ppo_kaggle.ipynb
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
├── calibration/   # quality_profile (the GATE) · sweep · fit
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

- **Personal portfolio / research project.** Assumes **personal IP ownership** of all code.
- **No company / proprietary data anywhere.** The RAG demo uses only public or synthetic data — a
  synthetic-doc generator ships with the repo.
- **No secrets in the repo.** All tokens / keys via `.env` (git-ignored).
- **No fabricated results.** Calibration params stay labeled *placeholder* until fit from real
  data; headline numbers exist only as eval outputs with variance.

---

## 📜 License

Proprietary — personal IP. See [`CLAUDE.md`](CLAUDE.md) §1 governance.

<div align="center">
<sub>POISE — depth follows physics.</sub>
</div>
