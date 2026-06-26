"""POISE inference engine — the *mechanism* of variable-depth execution.

This package is pure mechanism: it loads the model, exposes the per-layer stack,
projects intermediate hidden states to logits, runs the per-token variable-depth
loop, and manages the KV cache under depth changes. It contains **no control or
policy logic** — that lives in ``poise/control/`` (CLAUDE.md §9).
"""

from __future__ import annotations

from .model_loader import (
    load_model,
    get_layers,
    get_num_layers,
    get_lm_head,
    get_norm,
    get_embeddings,
    ModelLoadError,
)
from .early_exit import project_to_logits, last_token_logits
from .kv_cache import (
    KVStrategy,
    KVBookkeeping,
    VariableDepthKVCache,
    plan_for_depth,
)
from .adaptive_runner import (
    AdaptiveRunner,
    BudgetContext,
    TokenTrace,
    GenerationResult,
)

__all__ = [
    "load_model",
    "get_layers",
    "get_num_layers",
    "get_lm_head",
    "get_norm",
    "get_embeddings",
    "ModelLoadError",
    "project_to_logits",
    "last_token_logits",
    "KVStrategy",
    "KVBookkeeping",
    "VariableDepthKVCache",
    "plan_for_depth",
    "AdaptiveRunner",
    "BudgetContext",
    "TokenTrace",
    "GenerationResult",
]
