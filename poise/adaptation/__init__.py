"""LayerSkip-style adaptation (CLAUDE.md §9 option a) — ADDED SCOPE.

The step-3 gate measured that naive early-exit collapses quality on the stock model.
This package fine-tunes the model with an EARLY-EXIT LOSS over a LoRA adapter so
intermediate layers become calibrated to the LM head, making variable-depth execution
actually viable. Train off-device (Kaggle/GPU) or on the Jetson; then re-run the gate.

The loss + curriculum are pure and unit-tested (incl. a tiny-model test that proves the
intermediate-depth loss actually decreases — the exact effect that lowers the gate KL).
peft (LoRA) and the training loop are the heavier, on-GPU parts.
"""

from __future__ import annotations

from .layerskip import (
    early_exit_loss,
    distill_loss,
    loss_weights,
    evaluate_depth_losses,
    add_lora,
    load_text_blocks,
    train_layerskip,
)

__all__ = [
    "early_exit_loss",
    "distill_loss",
    "loss_weights",
    "evaluate_depth_losses",
    "add_lora",
    "load_text_blocks",
    "train_layerskip",
]
