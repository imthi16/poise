"""On-device engine validation on a REAL (tiny) Llama — runs wherever torch exists.

Guarded by importorskip so off-device CI skips it cleanly (universal testing: the
suite adapts to the machine). Where torch + transformers are installed, this exercises
the paths that the fake-forward tests cannot: the structural accessors on a real Llama,
the step-3 GATE (output_hidden_states + early-exit projection + KL/perplexity), and the
adaptive runner's real manual layer loop + KV cache — incl. a bit-identical check that
the full-depth prefill reproduces HF's own next-token logits.

Weights are random, so the *numbers* are meaningless (and nothing here is a quality
result); the assertions are structural correctness only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from poise.config import load_config  # noqa: E402
from poise.hardware import adapt_depth_to_model  # noqa: E402


def _tiny_llama(n_layers=4, vocab=128, hidden=64):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


class _StubTok:
    """Deterministic stub tokenizer: fixed ids per prompt (no vocab files needed)."""

    eos_token_id = None

    def __init__(self, vocab=128, length=12):
        self.vocab = vocab
        self.length = length

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None,
                 add_special_tokens=True):
        g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
        return {"input_ids": torch.randint(1, self.vocab, (1, self.length), generator=g)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def model():
    return _tiny_llama()


@pytest.fixture
def cfg(model):
    from poise.engine.model_loader import get_num_layers

    return adapt_depth_to_model(load_config(), get_num_layers(model))


def test_accessors_on_real_llama(model):
    from poise.engine.model_loader import (
        get_num_layers, get_norm, get_lm_head, get_embeddings,
    )

    assert get_num_layers(model) == 4
    assert type(get_norm(model)).__name__ == "LlamaRMSNorm"
    assert type(get_lm_head(model)).__name__ == "Linear"
    assert type(get_embeddings(model)).__name__ == "Embedding"


def test_gate_runs_on_real_model(model, cfg):
    """The step-3 GATE harness: full-depth KL == 0; full-depth top-1 agreement == 1."""
    from poise.calibration.quality_profile import QualityProfiler

    prof = QualityProfiler(cfg, model, _StubTok())
    results = prof.profile(["alpha", "beta", "gamma"], dataset_tag="test")
    by_depth = {r.depth: r for r in results}
    full = cfg.depth.layer_total
    assert abs(by_depth[full].mean_kl_vs_full) < 1e-6        # reference vs itself
    assert by_depth[full].top1_agreement == 1.0
    assert all(r.perplexity > 0 for r in results)
    # shallower depths diverge at least as much as full depth (>= 0 and full is the min)
    assert min(r.mean_kl_vs_full for r in results) == by_depth[full].mean_kl_vs_full


def test_adaptive_runner_real_forward_generates(model, cfg):
    from poise.engine.adaptive_runner import AdaptiveRunner

    runner = AdaptiveRunner(cfg, model, _StubTok())
    res = runner.generate("hello world", 6, budget_fn=lambda ctx: cfg.depth.layer_total)
    assert res.tokens == 6
    assert len(res.trace) == 6
    assert all(t.budget == cfg.depth.layer_total for t in res.trace)


def test_full_depth_prefill_matches_hf_logits(model, cfg):
    """Bit-identical precondition: the full-depth prefill must reproduce HF's own
    next-token argmax for the prompt's last position."""
    from poise.engine.adaptive_runner import AdaptiveRunner

    tok = _StubTok()
    input_ids = tok("fixed-prompt")["input_ids"]

    with torch.no_grad():
        hf_logits = model(input_ids).logits[0, -1, :]
    expected_first = int(torch.argmax(hf_logits).item())

    class _FixedTok(_StubTok):
        def __call__(self, text, **kw):
            return {"input_ids": input_ids}

    runner = AdaptiveRunner(cfg, model, _FixedTok())
    res = runner.generate("anything", 1, budget_fn=lambda ctx: cfg.depth.layer_total)
    assert res.token_ids[0] == expected_first  # prefill logits == HF logits


def test_runner_variable_depth_runs(model, cfg):
    """Variable per-token depth executes end-to-end (monotone strategy clamps holes)."""
    from poise.engine.adaptive_runner import AdaptiveRunner

    runner = AdaptiveRunner(cfg, model, _StubTok())
    depths = cfg.depth.budget_set
    res = runner.generate("vary", 5, budget_fn=lambda ctx: depths[min(ctx.step, len(depths) - 1)])
    assert res.tokens == 5
    assert all(d <= cfg.depth.layer_total for d in [t.budget for t in res.trace])
