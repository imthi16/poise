"""Step-3 GATE math + reporting helpers (CLAUDE.md §5 step 3, §9).

The model-running profiler needs torch + the gated model (an on-device step), so
here we lock down the pure-numpy measurement math and the LAYER_MIN recommendation
logic that decides whether the project's quality premise holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from poise.calibration.quality_profile import (
    DepthQuality,
    format_table,
    kl_divergence_logits,
    load_prompts,
    log_softmax,
    perplexity_from_logits,
    recommend_layer_min,
)


def test_log_softmax_normalizes():
    x = np.array([[1.0, 2.0, 3.0]])
    lp = log_softmax(x)
    np.testing.assert_allclose(np.sum(np.exp(lp), axis=-1), 1.0, rtol=1e-9)


def test_kl_zero_for_identical():
    logits = np.array([[0.3, 1.2, -0.7, 2.0]])
    kl = kl_divergence_logits(logits, logits)
    np.testing.assert_allclose(kl, 0.0, atol=1e-12)


def test_kl_known_value():
    # P = [0.5, 0.5];  Q = [0.9, 0.1]
    p_logits = np.array([[0.0, 0.0]])
    q_logits = np.array([[math.log(0.9), math.log(0.1)]])
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    np.testing.assert_allclose(kl_divergence_logits(p_logits, q_logits)[0], expected, rtol=1e-9)


def test_kl_is_nonnegative():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((20, 50))
    q = rng.standard_normal((20, 50))
    assert np.all(kl_divergence_logits(p, q) >= -1e-9)


def test_perplexity_uniform_equals_vocab():
    logits = np.zeros((10, 7))  # uniform over 7 classes
    targets = np.arange(10) % 7
    assert abs(perplexity_from_logits(logits, targets) - 7.0) < 1e-6


def test_recommend_layer_min_picks_smallest_within_threshold():
    results = [
        DepthQuality(16, mean_kl_vs_full=0.40, perplexity=9, top1_agreement=0.7,
                     task_accuracy=None, n_tokens=100),
        DepthQuality(24, mean_kl_vs_full=0.08, perplexity=7, top1_agreement=0.9,
                     task_accuracy=None, n_tokens=100),
        DepthQuality(32, mean_kl_vs_full=0.00, perplexity=6, top1_agreement=1.0,
                     task_accuracy=None, n_tokens=100),
    ]
    assert recommend_layer_min(results, max_kl=0.10) == 24
    assert recommend_layer_min(results, max_kl=0.50) == 16


def test_recommend_layer_min_signals_collapse():
    """No depth under threshold => None (a quality-collapse signal, not a silent pass)."""
    results = [
        DepthQuality(16, 1.2, 20, 0.4, None, 100),
        DepthQuality(24, 0.9, 15, 0.5, None, 100),
        DepthQuality(32, 0.0, 6, 1.0, None, 100),
    ]
    # Exclude full depth from "usable reduced depth" by setting a tight threshold.
    assert recommend_layer_min([r for r in results if r.depth < 32], max_kl=0.10) is None


def test_format_table_renders_all_depths():
    results = [
        DepthQuality(16, 0.4, 9.0, 0.7, 0.55, 100),
        DepthQuality(32, 0.0, 6.0, 1.0, 0.91, 100),
    ]
    txt = format_table(results)
    assert "depth" in txt and "16" in txt and "32" in txt


def test_load_prompts_txt_and_jsonl(tmp_path):
    txt = tmp_path / "p.txt"
    txt.write_text("hello world\nsecond prompt\n\n")
    assert load_prompts(txt) == ["hello world", "second prompt"]

    jl = tmp_path / "p.jsonl"
    jl.write_text('{"prompt": "a"}\n{"prompt": "b"}\n')
    assert load_prompts(jl) == ["a", "b"]
