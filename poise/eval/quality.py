"""Quality evaluation: perplexity, task accuracy, mean KL vs full-32 (CLAUDE.md §6, §11).

Mean KL vs full-depth is the hard-to-game signal (shared math with the step-3 gate).
At eval time we use TRUE per-token KL recorded during actual generation (vs the
static depth→quality approximation used for RL training) — this is the on-board
validation of the quality cost (§9). Pure aggregation here is unit-tested off-device;
perplexity/accuracy over the model are the on-device parts (lazy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

# reuse the gate's pure math (single source of truth for KL / perplexity)
from ..calibration.quality_profile import kl_divergence_logits, perplexity_from_logits

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..engine.adaptive_runner import TokenTrace


def mean_kl_from_trace(trace: Sequence["TokenTrace"]) -> float | None:
    """Mean recorded per-token KL vs full depth (None if not computed)."""
    kls = [t.kl_vs_full for t in trace if t.kl_vs_full is not None]
    return float(np.mean(kls)) if kls else None


def exact_match_accuracy(predictions: Iterable[str], golds: Iterable[str]) -> float:
    preds = list(predictions)
    gold = list(golds)
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(preds, gold) if _norm(p) == _norm(g))
    return correct / len(gold)


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def quality_loss_pct(metric_full: float, metric_reduced: float) -> float:
    """Relative degradation of a higher-is-better metric (e.g., accuracy), in percent.

    Returns the MEASURED loss; it is never asserted ahead of measurement. Positive =
    worse than full depth.
    """
    if metric_full == 0:
        return 0.0
    return 100.0 * (metric_full - metric_reduced) / abs(metric_full)


class QualityEvaluator:
    """On-device quality metrics over a prompt/task set (needs the model)."""

    def __init__(self, cfg: "PoiseConfig", model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def perplexity(self, texts: Sequence[str], depth: int | None = None, max_length: int = 512) -> float:
        """Perplexity of the (optionally depth-d) model over ``texts``."""
        import torch

        from ..engine.early_exit import project_to_logits

        full = self.cfg.depth.layer_total
        depth = full if depth is None else depth
        device = next(self.model.parameters()).device
        nlls, ntok = [], 0
        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = self.model(ids, output_hidden_states=(depth != full), use_cache=False)
            if depth == full:
                logits = out.logits[0, :-1, :]
            else:
                logits = project_to_logits(out.hidden_states[depth][0, :-1, :], self.model)
            targets = ids[0, 1:].detach().cpu().numpy()
            lp = logits.float().detach().cpu().numpy()
            ppl = perplexity_from_logits(lp, targets)
            nlls.append(np.log(ppl) * targets.shape[0])
            ntok += int(targets.shape[0])
        return float(np.exp(sum(nlls) / ntok)) if ntok else float("nan")

    def task_accuracy(self, suite: Sequence[tuple[str, str]], budget_fn, max_new_tokens: int = 32) -> float:
        """Exact-match accuracy generating with the adaptive runner under ``budget_fn``."""
        from ..engine.adaptive_runner import AdaptiveRunner

        runner = AdaptiveRunner(self.cfg, self.model, self.tokenizer)
        preds, golds = [], []
        for prompt, gold in suite:
            res = runner.generate(prompt, max_new_tokens, budget_fn)
            preds.append(res.text)
            golds.append(gold)
        return exact_match_accuracy(preds, golds)


__all__ = [
    "mean_kl_from_trace",
    "exact_match_accuracy",
    "quality_loss_pct",
    "QualityEvaluator",
    "kl_divergence_logits",
    "perplexity_from_logits",
]
