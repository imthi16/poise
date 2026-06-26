"""Control = POLICY (CLAUDE.md §9: runner = mechanism, control = policy).

Maps live hardware state to a layer budget via PID and/or a trained PPO policy, with
an unconditional thermal-safety override. The ``BudgetAllocator`` is the ``budget_fn``
the engine runner calls per token.
"""

from __future__ import annotations

from .pid import PID
from .budget_allocator import BudgetAllocator, snap_to_budget_set, make_allocator
from .policy import Policy

__all__ = ["PID", "BudgetAllocator", "snap_to_budget_set", "make_allocator", "Policy"]
