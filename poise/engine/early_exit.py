"""Project an intermediate hidden state to logits (CLAUDE.md §6).

``project_to_logits(hidden_state, model)`` applies the model's final norm and LM
head to a hidden state taken at an arbitrary depth ``d``.

⚠ RESEARCH-CRITICAL
-------------------
Intermediate hidden states are **not** calibrated to the final LM head in a stock
model. Naively applying the head at shallow depth degrades quality — possibly
badly. This module makes the *mechanism* available; the *quality cost* of exiting
at depth ``d`` is exactly what ``calibration/quality_profile.py`` measures and what
CLAUDE.md §9 flags as the project's central risk. Do not assume the cost is small —
measure it (the step-3 gate) before building the controller.
"""

from __future__ import annotations

from typing import Any

from .model_loader import get_lm_head, get_norm


def project_to_logits(hidden_state: Any, model: Any, *, apply_norm: bool = True) -> Any:
    """Map a hidden state ``[*, hidden]`` to logits ``[*, vocab]``.

    Parameters
    ----------
    hidden_state : the residual-stream activation at depth ``d`` (any leading dims).
    model        : the loaded causal-LM (provides ``norm`` + ``lm_head``).
    apply_norm   : apply the model's final norm before the head (default True —
                   this matches how the full-depth path produces logits).
    """
    head = get_lm_head(model)
    if apply_norm:
        norm = get_norm(model)
        hidden_state = norm(hidden_state)
    return head(hidden_state)


def last_token_logits(hidden_states: Any, model: Any, *, apply_norm: bool = True) -> Any:
    """Convenience: logits for the final sequence position from ``[batch, seq, hidden]``."""
    last = hidden_states[:, -1, :]
    return project_to_logits(last, model, apply_norm=apply_norm)


__all__ = ["project_to_logits", "last_token_logits"]
