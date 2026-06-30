# Novelty & positioning — hardware-state-conditioned adaptive computation

> Keep this framing exact across code comments, README, and the paper (CLAUDE.md §1, §9).

## The one-sentence contribution

POISE **conditions per-token transformer execution depth on the device's live
physical state** (junction temperature, power draw, GPU clock, throttle status),
closing the loop between hardware physics and per-token compute.

## What is and is not claimed

- ✅ **Claimed:** hardware-state-conditioned adaptive computation — *what drives the
  depth decision* is live device physics.
- ❌ **Not claimed:** that "adaptive layer skipping" / early-exit is itself novel. That
  field is crowded; we build on it.
- ❌ **Not claimed:** any "neuromorphic" hardware property. If the analogy
  "neuromorphic-inspired" is ever used, it refers *only* to the literal mechanism —
  hardware-state-conditioned variable-depth execution — never silicon.

## Prior art — what each method conditions depth on

| Method | Depth decision conditioned on | Signal type |
|---|---|---|
| **CALM** (Schuster et al.) | per-token confidence / early-exit classifier | input difficulty |
| **LayerSkip** (Elhoushi et al.) | trained early-exit + self-speculative decoding | input difficulty |
| **AdaInfer** | input features / statistical exit criteria | input difficulty |
| **DASH** | content-/confidence-driven dynamic depth | input difficulty |
| early-exit literature (broadly) | token/sequence confidence, entropy, content | input difficulty |
| **POISE (this work)** | **junction temp, power, GPU clock, throttle** | **hardware state** |

The orthogonal axis is the point: every row above decides depth from *what is being
computed*; POISE decides depth from *the physical state of the device computing it*.
The two are composable — a hardware-state controller can sit on top of an
input-difficulty exit criterion — but the hardware-state loop is the contribution here.

## The central honesty risk (⚠ — the step-3 gate)

Naively projecting an un-calibrated intermediate hidden state through the final LM
head (`engine/early_exit.py`) is **not** guaranteed to be cheap in quality. The
≤2–3% quality-loss target is a **hypothesis**, tested by `calibration/quality_profile.py`
*before* the controller is built.

If the measured depth→quality table shows quality collapsing outside a narrow band,
the honest responses (CLAUDE.md §9) are:

1. **(a)** add a LayerSkip-style adaptation step (LoRA + layer-dropout + early-exit
   loss) to make the model robust to variable depth — real added scope, flag to human;
2. **(b)** narrow the usable skip range and report the measured loss honestly;
3. **(c)** re-scope the claim.

We do **not** paper over a quality collapse to hit the target.

## Measured depth→quality results — the step-3 gate

### (1) Stock model, NAIVE early-exit — the gate's go/no-go finding

Measured by `calibration/quality_profile.py` on **DeepSeek-R1-Distill-Llama-8B** (32
layers, fp16) on a Jetson AGX Orin GPU, naive early-exit (projecting an un-calibrated
intermediate hidden state through the final LM head):

| depth | mean KL vs full-32 | perplexity | top-1 agreement |
|---|---|---|---|
| 16 | 7.24 | 615,631 | 0.017 |
| 20 | 6.30 | 278,734 | 0.076 |
| 24 | 5.20 | 109,711 | 0.130 |
| 28 | 3.91 | 34,098 | 0.146 |
| 32 (full) | 0.00 | 1,959 | 1.000 |

**`recommend_layer_min(KL ≤ 0.1) = 32` → no usable reduced depth.** Dropping even 4 of
32 layers gives ~17× worse perplexity and 15% next-token agreement. **The ≤2–3%
quality-loss hypothesis is *falsified* for naive layer-skipping on the stock model.**
(Absolute perplexity is inflated by a short synthetic eval set; the *relative KL* is the
architecture-driven signal and is what matters.) This is exactly the central risk §9
predicted — and the gate caught it before any controller was built on a false premise.

### (2) Decision (§9 option a): LayerSkip-style adaptation — IN PROGRESS

Because there is no usable band, we add a LayerSkip-style adaptation step
(`poise/adaptation/layerskip.py`): a LoRA fine-tune with an **early-exit loss** that
trains the shared norm+head to predict the next token from intermediate hidden states,
calibrating them to the head. The mechanism is validated on a tiny model
(`test_layerskip.py`: training cuts the shallow-depth loss >50%); the full adaptation on
the 8B + the **post-adaptation gate table** are the next measurement.

| depth | mean KL vs full-32 (ADAPTED) | usable? |
|---|---|---|
| _pending adaptation run_ | — | — |

The post-adaptation gate is the evidence that determines the *real* usable `LAYER_MIN`
and whether the quality premise holds after calibration. No headline energy/throughput/
quality number is asserted until `eval/report.py` produces it with variance.
