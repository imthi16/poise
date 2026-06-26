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

## Measured depth→quality results

> **Empty by design.** No KL / perplexity / accuracy numbers are written here until
> `calibration/quality_profile.py` (the gate) and `eval/report.py` have produced them
> with variance on the real model + board. The usable `LAYER_MIN` will be read off the
> measured table via `recommend_layer_min(...)`, not guessed.

| depth | mean KL vs full-32 | perplexity | top-1 agreement | task acc | usable? |
|---|---|---|---|---|---|
| _pending gate run_ | — | — | — | — | — |

When populated, this table is the evidence that determines whether the quality
premise holds and what `LAYER_MIN` actually is.
