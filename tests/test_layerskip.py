"""LayerSkip adaptation: early-exit loss + curriculum (CLAUDE.md §9 a).

The headline test proves the MECHANISM on a real (tiny) model: training with the
early-exit loss makes the intermediate-depth next-token loss drop sharply — i.e. the
shallow exit becomes a good predictor. That is exactly the effect that turns the gate's
catastrophic depth->KL curve into a usable one on the 8B. Torch-guarded so off-device
CI without torch skips cleanly.
"""

from __future__ import annotations

import pytest

from poise.adaptation.layerskip import load_text_blocks, loss_weights

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

try:
    from transformers import LlamaConfig, LlamaForCausalLM
except Exception as e:  # noqa: BLE001
    pytest.skip(f"transformers Llama unavailable: {e}", allow_module_level=True)

from poise.adaptation.layerskip import (  # noqa: E402
    distill_loss, early_exit_loss, evaluate_depth_losses,
)


def _tiny_llama(n_layers=4, vocab=64, hidden=32):
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


# --- pure curriculum logic (no torch) -------------------------------------- #
def test_loss_weights_linear_curriculum():
    w = loss_weights([2, 3, 4], full_depth=4, full_weight=1.0, ee_weight=1.0, curriculum="linear")
    assert w[4] == 1.0                       # full depth keeps full weight
    assert w[2] == pytest.approx(0.5)        # shallow exit weighted ~ d/full
    assert w[3] == pytest.approx(0.75)
    assert w[2] < w[3] < w[4]                # deeper exits weighted more


def test_loss_weights_none_uniform():
    w = loss_weights([2, 4], full_depth=4, ee_weight=1.0, curriculum="none")
    assert w[2] == 1.0 and w[4] == 1.0


def test_load_text_blocks_chunks(tmp_path):
    class _Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": list(range(50))}

    corpus = tmp_path / "c.txt"
    corpus.write_text("any text; the fake tokenizer returns 50 ids regardless")
    blocks = load_text_blocks(corpus, _Tok(), max_length=10)
    assert all(len(b) == 10 for b in blocks)
    assert len(blocks) == 4  # floor((50-10)/10) windows


# --- the mechanism on a real tiny model ------------------------------------ #
def test_early_exit_loss_shapes_and_grad():
    model = _tiny_llama()
    ids = torch.randint(1, 64, (1, 16))
    total, per = early_exit_loss(model, ids, [2, 4], full_depth=4)
    assert set(per) == {2, 4}
    assert torch.isfinite(total)
    assert per[4] >= 0 and per[2] >= 0
    total.backward()  # differentiable


def test_full_depth_loss_matches_model_loss():
    """At full depth the early-exit loss must equal the model's own next-token CE."""
    import torch.nn.functional as F

    model = _tiny_llama()
    ids = torch.randint(1, 64, (1, 16))
    with torch.no_grad():
        _, per = early_exit_loss(model, ids, [4], full_depth=4)
        logits = model(ids).logits[:, :-1, :]
        ref = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1))
    assert float(per[4]) == pytest.approx(float(ref), rel=1e-4)


def test_training_calibrates_intermediate_depth():
    """THE point of LayerSkip adaptation: training the early-exit loss makes the
    shallow-depth next-token loss drop sharply (intermediate states get calibrated to
    the head) — the same effect that fixes the gate's KL on the real model."""
    model = _tiny_llama(n_layers=4)
    model.train()
    ids = torch.randint(1, 64, (1, 24))  # overfit a fixed batch to isolate the effect
    depths = [2, 4]
    w = loss_weights(depths, 4)

    with torch.no_grad():
        _, per0 = early_exit_loss(model, ids, depths, full_depth=4)
    before = float(per0[2])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(60):
        total, _ = early_exit_loss(model, ids, depths, full_depth=4, weights=w)
        opt.zero_grad()
        total.backward()
        opt.step()

    with torch.no_grad():
        _, per1 = early_exit_loss(model, ids, depths, full_depth=4)
    after = float(per1[2])

    # the shallow exit becomes a much better predictor (this is what lowers gate KL)
    assert after < before * 0.5, f"shallow-depth loss did not improve: {before:.3f} -> {after:.3f}"


def test_distill_full_is_zero_and_shallow_equals_gate_kl():
    """Distillation objective: full-depth KL==0 (base preserved) and the shallow term
    IS the gate metric (KL of full vs the shallow projection)."""
    from poise.calibration.quality_profile import kl_divergence_logits
    from poise.engine.early_exit import project_to_logits

    model = _tiny_llama()
    ids = torch.randint(1, 64, (1, 16))
    with torch.no_grad():
        _, per = distill_loss(model, ids, [2, 4], full_depth=4)
        out = model(ids, output_hidden_states=True, use_cache=False)
        full = out.logits[:, :-1, :].float().numpy().reshape(-1, out.logits.size(-1))
        shallow = project_to_logits(out.hidden_states[2][:, :-1, :], model)
        shallow = shallow.float().numpy().reshape(-1, shallow.size(-1))
    assert float(per[4]) < 1e-5  # teacher full == student full => base anchored
    gate_kl = float(kl_divergence_logits(full, shallow).mean())
    assert float(per[2]) == pytest.approx(gate_kl, rel=1e-3)


def test_distill_training_reduces_shallow_kl():
    """Training the distill objective drives the shallow exit toward the full-depth
    distribution — i.e. directly lowers the gate KL (without a CE base-quality hit)."""
    model = _tiny_llama(n_layers=4)
    model.train()
    ids = torch.randint(1, 64, (1, 24))
    with torch.no_grad():
        _, p0 = distill_loss(model, ids, [2, 4], full_depth=4)
    before = float(p0[2])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(60):
        total, _ = distill_loss(model, ids, [2, 4], full_depth=4)
        opt.zero_grad()
        total.backward()
        opt.step()
    with torch.no_grad():
        _, p1 = distill_loss(model, ids, [2, 4], full_depth=4)
    assert float(p1[2]) < before * 0.7, f"shallow KL did not drop: {before:.3f} -> {float(p1[2]):.3f}"


def test_evaluate_depth_losses_runs():
    model = _tiny_llama()
    batches = [torch.randint(1, 64, (1, 16)) for _ in range(3)]
    losses = evaluate_depth_losses(model, batches, [2, 4], full_depth=4)
    assert set(losses) == {2, 4} and all(v >= 0 for v in losses.values())
