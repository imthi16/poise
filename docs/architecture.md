# POISE — architecture & module map

> The contribution is **hardware-state-conditioned adaptive computation**: per-token
> execution depth follows live device physics, not input difficulty. See `novelty.md`.

## Data flow

```
                    ┌──────────────────────── telemetry/ ───────────────────────┐
                    │  jtop | tegrastats | mock  →  TelemetrySample (temp/power/  │
                    │  clock/util/throttled), sampled on a background thread      │
                    └───────────────┬───────────────────────────────────────────┘
                                    │ live hardware state
                                    ▼
   control/ (POLICY)        ┌───────────────┐         engine/ (MECHANISM)
   ┌───────────────────┐    │ BudgetAllocator│  budget  ┌──────────────────────────┐
   │ PID  +  PPO Policy │──▶ │  static|pid|ppo│ ───────▶ │ adaptive_runner.generate │
   │ (obs == train obs) │    │  +TEMP_MAX 🔒  │   per     │  layers 0..budget-1 →    │
   └───────────────────┘    └───────────────┘   token   │  early_exit → logits     │
            ▲                                            │  kv_cache (variable depth)│
            │ trained off-device                         └───────────┬──────────────┘
   rl/ (OFF-DEVICE)                                                  │ tokens + per-token trace
   ┌───────────────────────────────────────────┐                    ▼
   │ simulator (RC, calibrated) + env (Gym) +   │            eval/ + serving/ + storage/
   │ reward (quality table from the step-3 gate)│
   │ + domain_random  →  train_ppo (PPO)        │
   └───────────────────────────────────────────┘
```

## Package map (`poise/`)

| Module | Role | Off-device? |
|---|---|---|
| `config.py` | typed config from YAML + `.env` + `POISE_*`, fail-loud validation | yes |
| `telemetry/` | `TelemetrySample` + jtop/tegrastats readers + **mock** | mock |
| `engine/model_loader` | load HF model (fp16/bnb-4bit), expose 32 layers + head + norm | import-only |
| `engine/early_exit` | project intermediate hidden state → logits (⚠ uncalibrated) | yes (math) |
| `engine/adaptive_runner` | per-token variable-depth loop; `budget_fn` only; trace | loop tested w/ fake fwd |
| `engine/kv_cache` | ⚠ KV consistency under variable depth (3 strategies) | bookkeeping tested |
| `control/pid` | reactive PID (lags slow thermal dynamics → motivates PPO) | yes |
| `control/budget_allocator` | state→budget; 🔒 clamp + `TEMP_MAX` override | yes |
| `control/policy` | PPO wrapper; obs shared with env (valid transfer) | fallback to PID |
| `rl/simulator` | RC thermal+power model (calibrated params only) | placeholder labeled |
| `rl/env` | Gymnasium env; quality term from the **gate** table | yes |
| `rl/reward` | ⚠ reward + reward-hacking depth-histogram guard | yes |
| `rl/domain_random` | per-episode R_th/C_th/power/ambient/noise randomization | yes |
| `rl/train_ppo` | PPO (Kaggle); converged depth dist + weight sweep | SB3 lazy |
| `calibration/quality_profile` | ⚠ **step-3 gate**: depth→{KL, ppl, acc} | math tested |
| `calibration/sweep` + `fit` | on-board sweeps → RC params + depth→power | sweep/fit tested |
| `baselines/` | 🔒 fixed-depth llama.cpp (Q4_K_M) + optional TRT-LLM | lazy |
| `eval/` | stress benchmark + quality + 🔒 variance report | aggregation tested |
| `serving/` | FastAPI §7 + Prometheus; mock engine off-device | yes |
| `storage/` | SQLite schema + migrations + accessors | yes |
| `rag/` | 🔒 demo only; synthetic FAISS + LangGraph; engine as black box | yes |

## Hard separations (CLAUDE.md §9)

- **runner = mechanism, control = policy** — no control logic in `adaptive_runner`.
- **TensorRT-LLM / Q4_K_M = baselines only** — never in the adaptive path.
- **LangGraph / FAISS / RAG = demo layer** — consume the engine as a black box.
- **thermal safety beats throughput** — `TEMP_MAX` override is unconditional.
- **no number before measurement** — headline figures come only from `eval/report.py`.

## Off-device vs on-device

Every hardware-touching module has a mock path. The full off-device suite
(`pytest`, 122 tests) runs without torch / the gated model / the board, exercising
config, telemetry-mock, the engine accessors + runner loop + KV bookkeeping, the gate
math, calibration fit, the simulator/env/reward/PPO diagnostics, PID + allocator,
eval aggregation, the API (mock engine), and the RAG demo.
