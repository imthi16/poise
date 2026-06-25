# CLAUDE.md — POISE

> **Power-Optimized Inference via State-aware Execution**
> On-device LLM engine that varies per-token transformer depth in response to live hardware physics.

This file is the single source of truth for an AI coding agent building POISE. Build from this file. If something here conflicts with a casual instruction in chat, **ask before deviating** on anything marked ⚠.

---

## 0. How to read this file (conventions)

- **⚠ RESEARCH-CRITICAL** — this is an open research decision, not settled engineering. Implement the specified *starting point*, but treat the values/approach as a hypothesis to be measured and tuned. **Never fabricate, assume, or hardcode a "result" here.** Build the measurement, run it, report what actually happens.
- **🔒 HARD CONSTRAINT** — non-negotiable. Violating it breaks the project's correctness, claims, or governance.
- **Existing input** — three Phase-0 files already exist and are tested: `poise_telemetry.py`, `poise_calibrate.py`, `fit_calibration.py`. **Integrate/refactor these into the package structure below; do not rewrite from scratch.**

The repo must run **off-device** (laptop, CI, Kaggle) in mock mode AND **on-device** (Jetson) in real mode. Every module that touches hardware must have a mock path so the rest of the system is developable and testable without the board.

---

## 1. Project Overview

POISE is an inference orchestrator for a decoder-only LLM running fully on-device on an NVIDIA Jetson AGX Orin. Instead of always executing all 32 transformer layers, POISE executes a **variable number of layers per token (a "layer budget")**, chosen in real time from the device's measured physical state — junction temperature, power draw, GPU clock, and throttle status.

The budget is set by a two-tier controller:

1. A **PID feedback controller** that reactively maps a thermal error signal to a layer budget.
2. A **PPO-trained RL policy** that, given the full hardware-state vector, learns an anticipatory budget allocation (PID is reactive and lags slow thermal dynamics; the RL policy can learn the thermal time-constants and act ahead of throttling).

**Objective:** under sustained thermal load where a fixed-depth model is forced into hardware throttling and its throughput collapses, POISE instead *gracefully reduces depth* to hold a target tokens/sec and cut energy-per-token, trading a small, measured amount of quality.

### The novelty — keep this exact framing everywhere 🔒

Existing adaptive-depth / early-exit methods (**CALM, LayerSkip, AdaInfer, DASH**, and the wider early-exit literature) condition execution depth on **input difficulty** (token/sequence confidence, content). POISE conditions execution depth on **hardware state** — it closes the loop between live device physics and per-token compute. **The contribution is hardware-state-conditioned adaptive computation.**

- 🔒 Do **not** present "adaptive layer skipping" itself as novel — that field is crowded.
- 🔒 Do **not** use the word "neuromorphic" as a standalone claim. If "neuromorphic-inspired" appears at all, it must be immediately paired with the literal mechanism ("hardware-state-conditioned variable-depth execution"). It is an analogy, never a hardware claim.
- All headline numbers (energy %, quality %, throughput) are **measured outputs of the eval harness**, reported with variance — never asserted in prose, READMEs, or the paper ahead of measurement.

### Governance 🔒

- This is a **personal portfolio/research project** that runs on employer-provided hardware. Written IP-ownership confirmation must exist before substantial building. The agent does not need to verify this, but **must not** introduce anything that assumes company ownership.
- 🔒 **No company/proprietary data anywhere.** The RAG demo uses **only public or synthetic data**. No internal docs, no scraped proprietary corpora.
- 🔒 No secrets in the repo. All tokens/keys via `.env` (git-ignored).

---

## 2. Full Tech Stack

| Concern | Choice | Notes |
|---|---|---|
| Adaptive inference path | **PyTorch + HuggingFace `transformers`, eager mode** | Full control over the per-layer loop is required for per-token variable depth. This is the primary engine. |
| Model | **DeepSeek-R1-Distill-Llama-8B** (Llama-3.1 base, 32 layers) | Gated → needs `HF_TOKEN`. |
| Adaptive-path weights | **fp16** (default) or **bnb-4bit** | ⚠ See §6 `model_loader` — **`Q4_K_M` is a GGUF/llama.cpp format and is NOT loadable in the HF eager path.** Q4_K_M is reserved for the llama.cpp baseline only. The PyTorch adaptive path runs fp16 (~16 GB, fits 64 GB easily) or bitsandbytes-4bit. |
| Static baseline | **llama.cpp** (GGUF **Q4_K_M**), optionally **TensorRT-LLM** | Baselines only. |
| RL | **PPO** (use a maintained impl: `stable-baselines3` or `cleanrl`) + **Gymnasium** env | Trained off-device on Kaggle against the simulator. |
| Control | Custom **PID** (no heavy dep) | |
| Thermal/power model | **RC model** fit from calibration (`scipy`, `numpy`) | The simulator core. |
| Serving | **FastAPI** + `uvicorn` | |
| Metrics | **Prometheus** (`prometheus-client`) | |
| Telemetry source | **jtop** (`jetson-stats`) / `tegrastats` parse / **mock** | |
| Vector store (demo) | **FAISS** (`faiss-cpu`) | Public/synthetic data only. |
| Demo orchestration | **LangGraph** | 🔒 **Demo/application layer ONLY** — never inside the inference engine. |
| Experiment store | **SQLite** (DuckDB acceptable for analytics) | See §8. |
| Dashboard | **React** (Vite) + Recharts | Reads the API + `/metrics`. |
| Config | **YAML** files in `configs/` + env overrides | |
| Testing | **pytest** | |

🔒 **TensorRT-LLM must never appear in the adaptive path.** It compiles a static graph; mid-inference per-token depth changes are incompatible with it. TensorRT-LLM exists in this repo only to produce a fast *fixed-depth* baseline number.

---

## 3. Exact Folder Structure to create

See the repository tree. The package root is `poise/`, with submodules: `telemetry/`,
`engine/`, `control/`, `rl/`, `calibration/`, `baselines/`, `eval/`, `serving/`,
`storage/`, `rag/`. Plus `configs/`, `dashboard/`, `scripts/`, `data/`, `tests/`, `docs/`.

---

## 5. Implementation Order (step by step)

Build in this order. Each step depends on the previous. **Step 3 is a go/no-go gate — do not build the controller until it passes.**

0. **Scaffold** — repo tree, `pyproject.toml`, `requirements.txt`, `.env.example`, `config.py`, `configs/*.yaml`, `storage/db.py` with migrations. Everything importable; `pytest` green on empty stubs.
1. **Telemetry** — refactor `poise_telemetry.py` into `telemetry/`. Real reader + mock + `TelemetrySample` schema. Testable off-device against mock.
2. **Engine — model loading + early exit** — `model_loader.py` exposes the 32 decoder layers as an indexable list and the LM head. `early_exit.py` projects a hidden state at depth *d* to logits. Verify you can run a forward pass through layers `0..d` and obtain a logit vector.
3. ⚠ **GATE — per-depth quality profile** — `calibration/quality_profile.py`. On a calibration prompt set, for each candidate depth *d*, measure mean KL-divergence (and perplexity, and a small task-accuracy suite) of depth-*d* output vs full-32 output. **Output: a depth→quality-cost table.** This determines whether the project's quality premise holds and what the real usable `LAYER_MIN` is.
4. **Adaptive runner + KV cache** — `adaptive_runner.py` generates tokens at a *given fixed* budget first (correctness), then accepts a per-token budget from a callback. `kv_cache.py` handles the ⚠ KV-consistency problem.
5. **Calibration → simulator params** *(on real board)* — refactor `poise_calibrate.py`/`fit_calibration.py` into `calibration/`. Produce fitted RC params + a depth→power map.
6. **Simulator + env + reward** — `rl/simulator.py` (RC model + depth→power + depth→quality-cost from step 3), `rl/env.py` (Gymnasium), `rl/reward.py`.
7. **PID + budget allocator** *(validate on board)* — `control/pid.py`, `control/budget_allocator.py`.
8. **PPO training (Kaggle)** — `rl/train_ppo.py` against the simulator with domain randomization. Produce `ppo_policy.zip`.
9. **PPO integration** *(validate on board)* — `control/policy.py` loads the policy; allocator runs in `ppo` mode. **Validate the sim-trained policy on the real board.**
10. **Baselines** — `baselines/static_llamacpp.py` (Q4_K_M), optional `static_trtllm.py`.
11. **Eval harness** — `eval/benchmark.py`, `eval/quality.py`, `eval/report.py`.
12. **Serving + metrics + dashboard** — FastAPI, Prometheus, React dashboard.
13. **RAG demo** — `rag/index.py` + `rag/graph.py` (LangGraph), synthetic data only.
14. **Repro packaging** — README run-book, seeds, config snapshots, results in `docs/arxiv/`.

---

## 9. Constraints / Do Not Do

🔒 **Framing & claims**
- Do **not** claim "adaptive layer skipping" as novel. The contribution is **hardware-state-conditioned** depth (vs input-difficulty methods CALM/LayerSkip/AdaInfer/DASH).
- Do **not** use "neuromorphic" unqualified. Only "neuromorphic-inspired", always immediately followed by the literal mechanism.
- Do **not** write any performance number anywhere until `eval/report.py` has produced it with variance.

🔒 **Architecture**
- Do **not** put TensorRT-LLM in the adaptive path. TRT-LLM = fixed-depth baseline only.
- Do **not** load `Q4_K_M` (GGUF) in the PyTorch adaptive path. Adaptive path = fp16 / bnb-4bit.
- Do **not** put LangGraph, FAISS, or RAG logic inside the inference engine. Demo/application layer only.
- Do **not** put control/policy logic inside `adaptive_runner.py`. Runner = mechanism; `control/` = policy.

🔒 **Correctness**
- Do **not** ship the variable-depth path without the KV-cache consistency tests passing (bit-identical at constant budget; characterized divergence per strategy).
- Do **not** let any budget choice exceed `TEMP_MAX`-driven safety override. Thermal safety beats throughput, always.

⚠ **Research-critical — the central honesty risk**
- The **≤2–3% quality-loss target is a hypothesis, not an assumption.** Run the step-3 quality profile **before** building the controller. If quality collapses outside a narrow depth band, the honest options are: (a) add a **LayerSkip-style adaptation step**; (b) **narrow the skip range** and report the measured loss honestly; or (c) re-scope the claim. Do not paper over a quality collapse.
- Reward shaping is iterative and hack-prone. Always report the converged **depth distribution**; a policy that collapses to minimum depth is a failed reward, not a success.
- The simulator's static depth→quality mapping is an approximation; the RC model is first-order. Use domain randomization, and **validate the policy on the real board**.

🔒 **Governance**
- No company/proprietary data anywhere. RAG = public/synthetic only.
- No secrets committed. All via `.env`.
- Assume **personal IP ownership** of all code.

---

## Quick reference — the one-paragraph pitch (keep accurate)

POISE runs a 32-layer LLM on a Jetson AGX Orin and **conditions per-token execution depth on the device's live physical state** (temperature, power, throttle) via a PID controller plus a PPO policy trained off-device against a calibrated RC thermal simulator. Unlike input-difficulty adaptive-depth methods, POISE closes the loop on **hardware state** — holding target throughput and cutting measured energy-per-token under sustained thermal load where a fixed-depth model throttles and collapses. Every claim is a measured eval output; the model's robustness to variable depth (and thus the achievable quality loss) is an empirical result gated early in the build.

---

*This is a condensed in-repo copy. The authoritative, full specification (with the
complete module-by-module rules, API schema §7, DB schema §8, and Definition of Done
§10) is the project brief from which this repo was built.*
