"""LayerSkip-style adaptation: early-exit loss + LoRA training (CLAUDE.md §9 a).

Core idea (Elhoushi et al., "LayerSkip"): make a model robust to reduced depth by
supervising its INTERMEDIATE exits — apply the shared final-norm + LM head to the
hidden state at depth ``d`` and train that to predict the next token. This calibrates
intermediate hidden states to the head, which is precisely what the step-3 gate found
missing in the stock model. A LoRA adapter keeps it parameter-efficient.

Split: the loss + curriculum are pure/torch and unit-tested (incl. a tiny-model test
that the depth-``d`` loss actually goes DOWN with training — the same effect that pulls
the gate's KL down). peft + the training loop are the on-GPU parts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ..engine.early_exit import project_to_logits

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Loss + curriculum (torch; unit-tested with a tiny model)
# --------------------------------------------------------------------------- #
def loss_weights(
    depths: Sequence[int],
    full_depth: int,
    *,
    full_weight: float = 1.0,
    ee_weight: float = 1.0,
    curriculum: str = "linear",
) -> dict[int, float]:
    """Per-depth loss weights. ``linear`` weights deeper exits more (~ d/full_depth),
    so shallow exits aren't over-penalized early while full-depth quality is preserved."""
    w: dict[int, float] = {}
    for d in depths:
        if d >= full_depth:
            w[d] = full_weight
        elif curriculum == "linear":
            w[d] = ee_weight * (d / full_depth)
        else:
            w[d] = ee_weight
    return w


def early_exit_loss(
    model: Any,
    input_ids: Any,
    depths: Sequence[int],
    *,
    full_depth: int,
    attention_mask: Any = None,
    weights: Optional[Mapping[int, float]] = None,
):
    """Weighted next-token loss across exit ``depths``.

    For each depth ``d``: full depth uses the model's own logits; a shallower depth
    projects ``hidden_states[d]`` through the shared norm + LM head (``project_to_logits``)
    — the same path the gate and the adaptive runner use, so training and inference agree.

    Returns ``(total_loss, {depth: loss})``.
    """
    import torch.nn.functional as F

    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    hidden_states = out.hidden_states  # len = num_layers + 1
    targets = input_ids[:, 1:].contiguous()

    per_depth = {}
    for d in depths:
        if d >= full_depth:
            logits = out.logits[:, :-1, :]
        else:
            logits = project_to_logits(hidden_states[d][:, :-1, :], model)
        per_depth[d] = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
        )

    w = dict(weights) if weights is not None else {d: 1.0 for d in depths}
    total = sum(w[d] * per_depth[d] for d in depths)
    return total, per_depth


def evaluate_depth_losses(
    model: Any, batches: Sequence[Any], depths: Sequence[int], *, full_depth: int
) -> dict[int, float]:
    """Mean per-depth cross-entropy over ``batches`` (a quick before/after adaptation
    check). Lower at shallow depths == intermediate states better calibrated to the head."""
    import torch

    sums = {d: 0.0 for d in depths}
    n = 0
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for input_ids in batches:
            _, per = early_exit_loss(model, input_ids, depths, full_depth=full_depth)
            for d in depths:
                sums[d] += float(per[d])
            n += 1
    if model_was_training:
        model.train()
    return {d: (sums[d] / max(1, n)) for d in depths}


# --------------------------------------------------------------------------- #
# LoRA + data (on-GPU)
# --------------------------------------------------------------------------- #
def add_lora(model: Any, lora_cfg: Mapping[str, Any]):
    """Wrap ``model`` with a LoRA adapter (peft). Returns the peft model."""
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover - off-device
        raise RuntimeError(
            "LayerSkip adaptation needs peft (pip install peft). It trains a small LoRA "
            "adapter; run on the GPU box / Jetson / Kaggle."
        ) from e
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules",
                                         ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_text_blocks(
    corpus_path: str | Path, tokenizer: Any, max_length: int = 512,
    max_blocks: Optional[int] = None,
) -> list:
    """Tokenize a text corpus and chunk it into fixed-length blocks for LM training."""
    text = Path(corpus_path).read_text()
    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    blocks = [ids[i:i + max_length] for i in range(0, max(0, len(ids) - max_length), max_length)]
    if not blocks and ids:
        blocks = [ids]
    if max_blocks:
        blocks = blocks[:max_blocks]
    return blocks


# --------------------------------------------------------------------------- #
# Config + training entrypoint
# --------------------------------------------------------------------------- #
def load_adaptation_config(config_dir: str | Path | None = None) -> dict:
    import yaml

    from ..config import DEFAULT_CONFIG_DIR

    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    path = cdir / "adaptation.yaml"
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("adaptation", {})


def train_layerskip(
    cfg: "PoiseConfig",
    adaptation: Optional[dict] = None,
    *,
    out_dir: Optional[str] = None,
    steps: Optional[int] = None,
) -> dict:
    """Fine-tune a LoRA adapter with the early-exit loss. Requires torch + peft + a GPU.

    Saves the adapter and returns before/after per-depth losses so you can see the
    intermediate exits improve (which is what lowers the gate KL)."""
    import torch

    from ..engine.model_loader import load_model, get_num_layers
    from ..hardware import adapt_depth_to_model

    adaptation = adaptation or load_adaptation_config()
    ee = adaptation.get("early_exit", {})
    tr = adaptation.get("train", {})
    out_dir = out_dir or tr.get("out_dir", "./data/results/layerskip_adapter")
    steps = steps or int(tr.get("steps", 500))

    model, tokenizer = load_model(cfg)
    n_layers = get_num_layers(model)
    cfg = adapt_depth_to_model(cfg, n_layers)
    full_depth = n_layers
    depths = [min(d, full_depth) for d in ee.get("depths", list(cfg.depth.budget_set))]
    depths = sorted(set(depths) | {full_depth})
    weights = loss_weights(
        depths, full_depth,
        full_weight=float(ee.get("full_weight", 1.0)),
        ee_weight=float(ee.get("ee_weight", 1.0)),
        curriculum=str(ee.get("curriculum", "linear")),
    )

    model = add_lora(model, adaptation.get("lora", {}))
    model.train()
    device = next(model.parameters()).device

    blocks = load_text_blocks(tr.get("corpus", cfg.rag.data_dir + "/ppl_corpus.txt"),
                              tokenizer, max_length=int(tr.get("max_length", 512)))
    if not blocks:
        raise RuntimeError("training corpus produced no blocks; provide a larger corpus")

    def batches(n):
        import random

        rng = random.Random(int(tr.get("seed", 0)))
        for _ in range(n):
            blk = rng.choice(blocks)
            yield torch.tensor([blk], device=device)

    before = evaluate_depth_losses(model, list(batches(8)), depths, full_depth=full_depth)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=float(tr.get("lr", 1e-4)))
    grad_accum = int(tr.get("grad_accum", 8))
    log_every = int(tr.get("log_every", 25))
    model.train()
    step = 0
    for input_ids in batches(steps * grad_accum):
        total, per = early_exit_loss(model, input_ids, depths,
                                     full_depth=full_depth, weights=weights)
        (total / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad()
        if step % (log_every * grad_accum) == 0:
            shallow = min(depths)
            print(f"step {step // grad_accum:4d}  total={float(total):.4f}  "
                  f"d{shallow}={float(per[shallow]):.4f}  d{full_depth}={float(per[full_depth]):.4f}")
        step += 1

    after = evaluate_depth_losses(model, list(batches(8)), depths, full_depth=full_depth)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"[POISE] saved LayerSkip adapter to {out_dir}")
    print("per-depth loss  (before -> after):")
    for d in depths:
        print(f"  depth {d:3d}:  {before[d]:.4f} -> {after[d]:.4f}")
    print("Set POISE_ADAPTER_PATH to this dir and re-run the gate "
          "(bash scripts/bringup.sh) to measure the new depth->quality curve.")
    return {"out_dir": out_dir, "depths": depths, "before": before, "after": after}


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser(description="POISE LayerSkip adaptation (LoRA + early-exit loss)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()

    cfg = load_config()
    adaptation = load_adaptation_config()
    if args.corpus:
        adaptation.setdefault("train", {})["corpus"] = args.corpus
    train_layerskip(cfg, adaptation, out_dir=args.out, steps=args.steps)


__all__ = [
    "loss_weights",
    "early_exit_loss",
    "evaluate_depth_losses",
    "add_lora",
    "load_text_blocks",
    "load_adaptation_config",
    "train_layerskip",
    "main",
]
