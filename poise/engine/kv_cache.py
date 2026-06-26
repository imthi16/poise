"""KV-cache management under variable depth (⚠ CLAUDE.md §6 — hardest correctness problem).

Each decoder layer holds its own KV cache. If token *t* runs only layers ``0..k``,
then layers ``k+1..L-1`` have **no KV entry for token t**. A later token *t+1* that
runs deeper will attend, at those layers, to a cache with a **hole**.

Three strategies are implemented behind a flag. Their quality/throughput tradeoff
must be **measured**, not assumed (the runner records per-token ``kl_vs_full`` so
``eval`` can characterize divergence per strategy):

1. ``recompute_on_demand`` — when depth increases, recompute the missing positions'
   KV for the newly-needed layers (correct; cost spikes on depth-up transitions).
2. ``propagate_hidden`` — for skipped layers, cheaply populate KV from the carried
   hidden state via that layer's k/v projections only (no attention/MLP).
   Approximate; must be quality-tested.
3. ``monotone_nonincreasing_within_window`` — restrict depth to be non-increasing
   within a window so holes are never attended (restrictive; a clean correctness
   baseline).

This module splits the *bookkeeping/policy* (pure Python, fully tested off-device)
from the *tensor ops* (torch, validated on-device). The contract enforced here is:
**at a constant budget no hole is ever created**, which is what makes the
constant-budget output bit-identical to a normal HF generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class KVStrategy(str, Enum):
    RECOMPUTE_ON_DEMAND = "recompute_on_demand"
    PROPAGATE_HIDDEN = "propagate_hidden"
    MONOTONE_NONINCREASING = "monotone_nonincreasing_within_window"


# --------------------------------------------------------------------------- #
# Pure-Python bookkeeping — which (layer, position) KV entries are valid.
# --------------------------------------------------------------------------- #
@dataclass
class KVBookkeeping:
    """Tracks, per layer, which token positions have a valid KV entry.

    Invariant we rely on for correctness: a token at depth ``d`` populates layers
    ``0..d-1`` for its position. Attention at layer ``L`` for a query position ``q``
    reads keys/values at all positions ``p <= q``. A *hole* exists if any such ``p``
    lacks a KV entry at layer ``L``.
    """

    num_layers: int
    # filled[layer] = set of positions with valid KV at this layer.
    filled: list[set[int]] = field(default_factory=list)
    depth_history: list[int] = field(default_factory=list)  # depth chosen per position

    def __post_init__(self):
        if not self.filled:
            self.filled = [set() for _ in range(self.num_layers)]

    def reset(self) -> None:
        self.filled = [set() for _ in range(self.num_layers)]
        self.depth_history = []

    def record_token(self, position: int, depth: int) -> None:
        """Mark layers ``0..depth-1`` as filled for ``position``."""
        for layer in range(min(depth, self.num_layers)):
            self.filled[layer].add(position)
        # pad history if positions were skipped
        while len(self.depth_history) <= position:
            self.depth_history.append(0)
        self.depth_history[position] = depth

    def fill_layer_position(self, layer: int, position: int) -> None:
        """Mark a single (layer, position) filled (used by recompute/propagate)."""
        self.filled[layer].add(position)

    def holes_for_query(self, layer: int, query_position: int) -> list[int]:
        """Positions ``p <= query_position`` missing a KV entry at ``layer``."""
        present = self.filled[layer]
        return [p for p in range(query_position + 1) if p not in present]

    def has_hole_for_query(self, depth: int, query_position: int) -> bool:
        """True if attending at any layer ``0..depth-1`` for ``query_position`` hits a hole."""
        for layer in range(min(depth, self.num_layers)):
            if self.holes_for_query(layer, query_position):
                return True
        return False

    def seq_len(self) -> int:
        return len(self.depth_history)


# --------------------------------------------------------------------------- #
# Strategy policy — what depth is actually admissible, and what repair is needed.
# --------------------------------------------------------------------------- #
@dataclass
class KVRepairPlan:
    """What the runner must do before attending at ``target_depth`` for ``position``."""
    admissible_depth: int
    recompute_layers: list[int] = field(default_factory=list)   # layers needing recompute
    recompute_positions: list[int] = field(default_factory=list)
    propagate_layers: list[int] = field(default_factory=list)   # layers to fill via k/v proj


def plan_for_depth(
    strategy: KVStrategy,
    book: KVBookkeeping,
    position: int,
    requested_depth: int,
    *,
    window: Optional[int] = None,
) -> KVRepairPlan:
    """Translate a requested depth into an admissible depth + a repair plan.

    Pure function — no tensors. The runner consumes the plan to do the actual work.
    """
    L = book.num_layers
    requested_depth = max(1, min(requested_depth, L))

    if strategy is KVStrategy.MONOTONE_NONINCREASING:
        # Clamp so depth never increases within the window => no hole is ever attended.
        prev = _window_min_depth(book, position, window)
        admissible = requested_depth if prev is None else min(requested_depth, prev)
        return KVRepairPlan(admissible_depth=max(1, admissible))

    if strategy is KVStrategy.RECOMPUTE_ON_DEMAND:
        # Allow the requested depth; recompute KV at any newly-needed layer for the
        # earlier positions that lack it.
        layers, positions = [], set()
        for layer in range(requested_depth):
            missing = book.holes_for_query(layer, position - 1) if position > 0 else []
            # exclude the current position (it has not been computed yet at all)
            missing = [p for p in missing if p < position]
            if missing:
                layers.append(layer)
                positions.update(missing)
        return KVRepairPlan(
            admissible_depth=requested_depth,
            recompute_layers=layers,
            recompute_positions=sorted(positions),
        )

    if strategy is KVStrategy.PROPAGATE_HIDDEN:
        # Skipped layers for *previous* tokens are filled cheaply from carried hidden
        # state via k/v projections (done at skip time by the runner). Here we just
        # surface which layers of the current step are "skipped" beyond the budget so
        # the runner can propagate them after computing layers 0..requested_depth-1.
        propagate = list(range(requested_depth, L))
        return KVRepairPlan(admissible_depth=requested_depth, propagate_layers=propagate)

    raise ValueError(f"unknown KV strategy: {strategy!r}")  # pragma: no cover


def _window_min_depth(
    book: KVBookkeeping, position: int, window: Optional[int]
) -> Optional[int]:
    """Min depth used over the recent window of already-decoded positions."""
    if position <= 0 or not book.depth_history:
        return None
    start = 0 if window is None else max(0, position - window)
    hist = book.depth_history[start:position]
    hist = [d for d in hist if d > 0]
    return min(hist) if hist else None


# --------------------------------------------------------------------------- #
# Tensor-backed cache (torch path, validated on-device).
# --------------------------------------------------------------------------- #
class VariableDepthKVCache:
    """Per-layer key/value tensors plus the bookkeeping above.

    Tensors are kept as ``[batch, heads, seq, head_dim]`` and concatenated along the
    sequence axis as tokens are appended. Off-device the tensor methods are never
    exercised; the bookkeeping is what tests assert.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.keys: list[Optional[Any]] = [None] * num_layers
        self.values: list[Optional[Any]] = [None] * num_layers
        self.book = KVBookkeeping(num_layers=num_layers)

    def reset(self) -> None:
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.book.reset()

    def get(self, layer: int):
        return self.keys[layer], self.values[layer]

    def append(self, layer: int, k: Any, v: Any, position: int) -> None:
        """Append one (or more) positions' KV at ``layer`` along the seq axis."""
        import torch  # local import; tensor path only

        if self.keys[layer] is None:
            self.keys[layer] = k
            self.values[layer] = v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
        self.book.fill_layer_position(layer, position)

    def record_token(self, position: int, depth: int) -> None:
        self.book.record_token(position, depth)

    def seq_len(self) -> int:
        return self.book.seq_len()


__all__ = [
    "KVStrategy",
    "KVBookkeeping",
    "KVRepairPlan",
    "plan_for_depth",
    "VariableDepthKVCache",
]
