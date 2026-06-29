# POISE bring-up — universal, turnkey

POISE runs on **any** machine and **any** HF decoder-only model. The same command
adapts to what you have:

| Machine | Device | Telemetry | What runs |
|---|---|---|---|
| Jetson AGX Orin | `cuda` | `jtop` / `tegrastats` | everything, incl. real calibration |
| Generic NVIDIA GPU box | `cuda` | `nvml` | model · gate · eval (calibration if sensors) |
| Apple Silicon | `mps` | `mock` | model · gate · eval |
| CPU-only laptop | `cpu` | `mock` | model · gate · eval (use a tiny model) |

Device (`auto` → cuda > mps > cpu) and telemetry (`auto` → jtop > nvml > mock) are
detected automatically; the depth/budget config is rescaled to the loaded model's
actual layer count.

## One command

```bash
bash scripts/bringup.sh
```

Runs: **profile → deps → model → GATE → eval**, printing ok / skipped / failed for
each and *why*. Steps that need hardware you don't have are skipped cleanly.

### Validate the whole pipeline on a CPU laptop (no GPU, no gated model)

```bash
pip install -r requirements.txt
pip install torch transformers            # CPU wheels are fine
bash scripts/bringup.sh --model sshleifer/tiny-gpt2
```

A tiny model exercises load → depth-adaptation → the GATE → the adaptive runner →
eval end to end, so you can prove the plumbing anywhere before touching the Jetson.

### Just inspect the machine

```bash
bash scripts/bringup.sh --steps profile,deps
```

## The full on-device sequence (Jetson)

```bash
# 0. credentials + extras
cp .env.example .env          # set HF_TOKEN; POISE_DEVICE/TELEMETRY stay 'auto'
pip install -r requirements.txt
pip install -e '.[engine]'    # torch (JetPack build) + transformers + accelerate
pip install jetson-stats      # jtop telemetry

# 1. bring-up: profile, deps, load model, run the GATE, smoke eval
bash scripts/bringup.sh

#    -> the GATE prints the depth->{KL, perplexity, accuracy} table and the usable
#       LAYER_MIN. If quality collapses outside a narrow band, STOP and read
#       CLAUDE.md §9 (LayerSkip adaptation / narrow-and-report / re-scope).

# 2. calibrate the thermal model (writes configs/simulator.yaml, real params only)
bash scripts/run_calibration.sh
python3 -c "from poise.calibration import fit; print('fit RC params from data/calibration/*.csv')"

# 3. train the PPO policy OFF-DEVICE (Kaggle) against the calibrated simulator
#    scripts/train_ppo_kaggle.ipynb  -> copy ppo_policy.zip back, set POISE_POLICY_PATH

# 4. full eval comparison (static-full / static x2 / pid / ppo) with variance
bash scripts/run_benchmark.sh

# 5. serve the API + live dashboard (loads the REAL engine on-device)
bash scripts/serve.sh
cd dashboard && npm install && npm run dev
```

## Notes

- **Nothing is trusted before the GATE.** The ≤2–3% quality-loss target is a
  hypothesis the GATE tests; no performance number is asserted before `eval/report.py`.
- **Calibration uses real sensors only.** On a machine with `mock` telemetry the
  calibrate step is skipped — synthetic sweeps must never be used as real RC params.
- **PPO trains off-device.** On-board RL is far too slow (thermal time-constants are
  minutes); `--train` runs only a tiny local smoke-train.
- **`bnb-4bit` needs CUDA.** On mps/cpu the loader falls back to fp16/fp32 with a note.
