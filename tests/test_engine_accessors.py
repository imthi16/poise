"""Engine accessors + early-exit wiring, tested off-device with a fake model.

We cannot load the 8B gated model (or torch) in CI, so these tests verify the
*structural* contract: layer indexing, head/norm/embedding lookup, the q4_k_m
rejection, and that project_to_logits applies norm-then-head in the right order.
"""

from __future__ import annotations

import numpy as np
import pytest

from poise.engine import model_loader as ml
from poise.engine.early_exit import project_to_logits, last_token_logits
from poise.engine.model_loader import ModelLoadError


# --- a minimal stand-in for a HF LlamaForCausalLM ------------------------- #
class _Norm:
    """Identity-plus-scale norm so we can detect whether it was applied."""

    def __init__(self, scale=2.0):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class _Head:
    def __init__(self, hidden, vocab, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((hidden, vocab)).astype(np.float32)

    def __call__(self, x):
        return x @ self.W


class _Base:
    def __init__(self, n_layers, hidden, vocab):
        self.layers = [f"decoder_block_{i}" for i in range(n_layers)]
        self.norm = _Norm()
        self.embed_tokens = "embed"


class _FakeModel:
    def __init__(self, n_layers=32, hidden=8, vocab=16):
        self.model = _Base(n_layers, hidden, vocab)
        self.lm_head = _Head(hidden, vocab)


def test_layer_accessors():
    m = _FakeModel(n_layers=32)
    assert ml.get_num_layers(m) == 32
    layers = ml.get_layers(m)
    assert layers[0] == "decoder_block_0" and layers[-1] == "decoder_block_31"
    assert ml.get_lm_head(m) is m.lm_head
    assert ml.get_norm(m) is m.model.norm
    assert ml.get_embeddings(m) == "embed"


def test_project_to_logits_applies_norm_then_head():
    m = _FakeModel(hidden=4, vocab=5)
    h = np.ones((1, 4), dtype=np.float32)
    with_norm = project_to_logits(h, m, apply_norm=True)
    without_norm = project_to_logits(h, m, apply_norm=False)
    # norm scales by 2.0, so logits must differ and be exactly 2x (head is linear)
    assert with_norm.shape == (1, 5)
    np.testing.assert_allclose(with_norm, without_norm * 2.0, rtol=1e-5)


def test_last_token_logits_picks_final_position():
    m = _FakeModel(hidden=4, vocab=5)
    seq = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = last_token_logits(seq, m, apply_norm=False)
    expected = seq[:, -1, :] @ m.lm_head.W
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_load_model_rejects_q4km_without_torch(cfg, monkeypatch):
    """q4_k_m must be rejected with a pointer to the baseline, independent of torch."""
    # Build a ModelConfig with q4_k_m directly (config loader would also reject it).
    from dataclasses import replace

    bad = replace(cfg, model=replace(cfg.model, dtype="q4_k_m"))
    with pytest.raises(ModelLoadError) as ei:
        ml.load_model(bad)
    assert "llama.cpp" in str(ei.value)


def test_load_model_without_torch_gives_clear_error(cfg):
    """Off-device (no torch) load_model must fail loud with install guidance."""
    try:
        import torch  # noqa: F401

        pytest.skip("torch is installed; skipping the missing-torch path")
    except Exception:
        pass
    with pytest.raises(ModelLoadError) as ei:
        ml.load_model(cfg)
    assert "torch" in str(ei.value).lower()


def test_wrong_layer_count_helper():
    m = _FakeModel(n_layers=28)
    assert ml.get_num_layers(m) == 28  # loader would reject this vs layer_total=32
