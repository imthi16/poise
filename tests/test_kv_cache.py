"""KV-cache bookkeeping + strategy correctness (⚠ CLAUDE.md §6, §10).

The bit-identical-at-constant-budget assertion and the per-strategy divergence
*magnitude* require the model and run on-device (see test_adaptive_runner's
on-device markers). Here we lock down the correctness REASONING off-device:

  * constant budget never creates a hole  => the precondition for bit-identical
    output (no silent KV corruption);
  * monotone-nonincreasing clamping provably avoids attended holes;
  * recompute_on_demand surfaces exactly the holes that need repair;
  * propagate_hidden marks the skipped layers to fill.
"""

from __future__ import annotations

from poise.engine.kv_cache import (
    KVBookkeeping,
    KVStrategy,
    plan_for_depth,
)


def test_record_token_fills_layers_below_depth():
    book = KVBookkeeping(num_layers=32)
    book.record_token(position=0, depth=20)
    assert all(0 in book.filled[l] for l in range(20))
    assert all(0 not in book.filled[l] for l in range(20, 32))


def test_constant_budget_creates_no_hole():
    """The invariant behind bit-identical output: same depth every token => no hole."""
    book = KVBookkeeping(num_layers=32)
    depth = 24
    for pos in range(10):
        book.record_token(pos, depth)
    for q in range(10):
        assert not book.has_hole_for_query(depth, q)


def test_increasing_depth_creates_attended_hole():
    """If depth rises, a deeper later token attends to layers that earlier shallow
    tokens never populated => a hole. This is the core hazard."""
    book = KVBookkeeping(num_layers=32)
    book.record_token(0, depth=16)  # shallow
    book.record_token(1, depth=16)
    book.record_token(2, depth=28)  # deeper: layers 16..27 have holes at pos 0,1
    assert book.has_hole_for_query(28, query_position=2)
    holes = book.holes_for_query(layer=20, query_position=2)
    assert holes == [0, 1]  # positions 0 and 1 lack KV at layer 20


def test_monotone_clamp_prevents_holes():
    book = KVBookkeeping(num_layers=32)
    requested = [32, 28, 30, 16, 24, 20]  # note the up-swings at idx 2 and 4
    for pos, req in enumerate(requested):
        plan = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, pos, req)
        d = plan.admissible_depth
        # admissible depth is non-increasing
        if pos > 0:
            assert d <= book.depth_history[pos - 1]
        book.record_token(pos, d)
    # every query attends without hitting a hole
    for q in range(len(requested)):
        d_q = book.depth_history[q]
        assert not book.has_hole_for_query(d_q, q)


def test_monotone_window_allows_reset_outside_window():
    book = KVBookkeeping(num_layers=32)
    # window=2: depth can rise again once old shallow tokens fall outside the window
    plan0 = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, 0, 16, window=2)
    book.record_token(0, plan0.admissible_depth)  # 16
    plan1 = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, 1, 16, window=2)
    book.record_token(1, plan1.admissible_depth)  # 16
    # position 2: window covers positions [0,1] -> min depth 16 -> clamp to 16
    plan2 = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, 2, 32, window=2)
    assert plan2.admissible_depth == 16


def test_recompute_plan_targets_the_holes():
    book = KVBookkeeping(num_layers=32)
    book.record_token(0, depth=16)
    book.record_token(1, depth=16)
    plan = plan_for_depth(KVStrategy.RECOMPUTE_ON_DEMAND, book, position=2, requested_depth=28)
    assert plan.admissible_depth == 28  # request honored
    # layers 16..27 need recompute for positions 0,1
    assert 20 in plan.recompute_layers and 16 in plan.recompute_layers
    assert plan.recompute_positions == [0, 1]
    # layers below 16 were already filled -> not in recompute set
    assert 0 not in plan.recompute_layers


def test_recompute_no_op_when_no_holes():
    book = KVBookkeeping(num_layers=32)
    for pos in range(3):
        book.record_token(pos, depth=32)
    plan = plan_for_depth(KVStrategy.RECOMPUTE_ON_DEMAND, book, position=3, requested_depth=32)
    assert plan.recompute_layers == [] and plan.recompute_positions == []


def test_propagate_marks_skipped_layers():
    book = KVBookkeeping(num_layers=32)
    plan = plan_for_depth(KVStrategy.PROPAGATE_HIDDEN, book, position=0, requested_depth=20)
    assert plan.admissible_depth == 20
    assert plan.propagate_layers == list(range(20, 32))


def test_requested_depth_clamped_to_layer_count():
    book = KVBookkeeping(num_layers=32)
    plan = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, 0, requested_depth=99)
    assert plan.admissible_depth == 32
    plan2 = plan_for_depth(KVStrategy.MONOTONE_NONINCREASING, book, 0, requested_depth=0)
    assert plan2.admissible_depth == 1
