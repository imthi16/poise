# POISE — Power-Optimized Inference via State-aware Execution

> On-device LLM engine that varies per-token transformer depth in response to live hardware physics.

POISE runs a 32-layer decoder-only LLM on an NVIDIA Jetson AGX Orin and **conditions
per-token execution depth on the device's live physical state** (junction temperature,
power draw, GPU clock, throttle status) via a PID controller plus a PPO policy trained
off-device against a calibrated RC thermal simulator.

## The contribution

**Hardware-state-conditioned adaptive computation.**

Existing adaptive-depth / early-exit methods (CALM, LayerSkip, AdaInfer, DASH, and the
wider early-exit literature) condition execution depth on **input difficulty** (token /
sequence confidence, content). POISE instead conditions execution depth on **hardware
state** — it closes the loop between live device physics and per-token compute.

- "Adaptive layer skipping" by itself is **not** claimed as novel; that field is crowded.
  The novelty is *what drives the depth decision*: hardware state, not input difficulty.
- The mechanism is literally **hardware-state-conditioned variable-depth execution**. Any
  "neuromorphic-inspired" analogy, if used at all, refers to exactly that mechanism — it is
  never a hardware/silicon claim.

## What it does

Under sustained thermal load, a fixed-depth model is forced into hardware throttling and
its throughput collapses. POISE instead *gracefully reduces depth* to hold a target
tokens/sec and cut energy-per-token, trading a small, **measured** amount of quality.

> **No performance numbers appear in this README, the docs, or the paper until
> `poise/eval/report.py` has produced them with variance across multiple runs.** Every
> headline figure (energy %, quality %, throughput) is a measured output of the eval
> harness, reported with spread — never asserted ahead of measurement.

## Architecture (two-tier controller)

1. **PID feedback controller** — reactively maps a thermal error signal to a layer budget.
   PID lags slow thermal dynamics; that lag is the motivation for the second tier.
2. **PPO-trained RL policy** — given the full hardware-state vector, learns an anticipatory
   budget allocation, acting ahead of throttling. Trained off-device against the simulator
   with domain randomization, then validated on the real board.

```
telemetry ─▶ control/ (PID + PPO policy) ─▶ budget ─▶ engine/adaptive_runner ─▶ tokens
   ▲                                                          │
   └──────────────── live hardware state ◀────────────────────┘
```

The **engine is mechanism, `control/` is policy** — the runner contains no control logic.

## Off-device vs on-device

Every hardware-touching module has a **mock path**, so the whole stack is developable and
testable without the Jetson:

- `POISE_TELEMETRY_BACKEND=mock` drives a plausible thermal curve from a load level.
- The RC simulator + Gymnasium env let the PPO policy train on Kaggle with no board.
- Heavy/gated deps (`torch`, `transformers`, `faiss`, `llama.cpp`) are optional; the core
  (config, telemetry-mock, control, simulator, env, serving, storage) installs and tests
  green off-device.

```bash
pip install -r requirements.txt        # core + serving + RL env (no torch)
pytest                                 # green off-device in mock mode
cp .env.example .env                   # then fill in HF_TOKEN etc. for on-device use
```

## Build / run order

See `CLAUDE.md` §5 for the authoritative step-by-step build order. Summary:

1. Scaffold → 2. Telemetry → 3. **GATE: per-depth quality profile** → 4. Adaptive runner +
KV cache → 5. Calibration (board) → 6. Simulator + env + reward → 7. PID → 8. PPO (Kaggle)
→ 9. PPO on board → 10. Baselines → 11. Eval → 12. Serving + dashboard → 13. RAG demo.

**Step 3 is a go/no-go gate**: measure the depth→quality cost before building the
controller. The ≤2–3% quality-loss target is a hypothesis to be measured, not an assumption
(see `docs/novelty.md` and `CLAUDE.md` §9).

## Run-book

```bash
# Calibration (on the Jetson):
bash scripts/run_calibration.sh

# PPO training (Kaggle, off-device against the simulator):
#   scripts/train_ppo_kaggle.ipynb

# Benchmark / stress protocol:
bash scripts/run_benchmark.sh

# Serve the API + Prometheus metrics:
bash scripts/serve.sh        # FastAPI on :8000, /metrics for Prometheus

# Dashboard (live temp / power / budget / tok-s):
cd dashboard && npm install && npm run dev
```

## Governance

- Personal portfolio/research project. Assumes **personal IP ownership** of all code.
- **No company/proprietary data anywhere.** The RAG demo uses only public or synthetic data
  (a synthetic-doc generator ships with the repo).
- **No secrets in the repo.** All tokens/keys via `.env` (git-ignored).

## Repository layout

See `CLAUDE.md` §3 for the full tree and `docs/architecture.md` for the module map.

## License

Proprietary — personal IP. See `CLAUDE.md` §1 governance.
