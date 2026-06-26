"""⚠ STEP-3 GATE — per-depth quality profile (CLAUDE.md §5 step 3, §6, §9).

For each candidate depth ``d`` we measure, against a full-32 reference:
  * mean KL-divergence  KL(P_full || P_depth)   — the primary, hard-to-game signal
  * perplexity of the depth-``d`` model on the calibration text
  * top-1 agreement with the full-depth model (a behavioral accuracy proxy)
  * optional task accuracy on a small provided suite

Output: a depth→quality-cost table, persisted to the ``quality_profile`` DB table
and summarized for ``docs/novelty.md``. **The usable LAYER_MIN is read off this
table — it is never guessed.**

HONESTY CONTRACT (🔒 / ⚠)
------------------------
This file computes quality cost from REAL forward passes. It contains NO hardcoded
KL/perplexity numbers and writes none. Off-device (no torch / no gated model) the
measurement cannot run and raises — it must never emit fabricated results. The
≤2–3% quality-loss target is a HYPOTHESIS this gate tests; if quality collapses
outside a narrow band, report it honestly (see §9 options a/b/c).

The model side uses ``output_hidden_states=True``: a single teacher-forced forward
yields the post-layer hidden state at every depth, which ``early_exit`` projects to
logits. No KV bookkeeping is needed here (that is generation — adaptive_runner).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np

from ..engine.early_exit import project_to_logits

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Pure math (numpy) — unit-testable off-device
# --------------------------------------------------------------------------- #
def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - m
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse


def kl_divergence_logits(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Per-row KL(P || Q) in nats, where P/Q are softmax(p_logits)/softmax(q_logits).

    ``p`` is the reference (full-depth); ``q`` is the depth-``d`` distribution.
    Returns an array with the class axis reduced.
    """
    logp = log_softmax(p_logits, axis=-1)
    logq = log_softmax(q_logits, axis=-1)
    p = np.exp(logp)
    return np.sum(p * (logp - logq), axis=-1)


def perplexity_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Perplexity = exp(mean NLL) of ``targets`` under ``logits`` ([N, vocab], [N])."""
    logp = log_softmax(logits, axis=-1)
    idx = np.asarray(targets, dtype=np.int64)
    nll = -logp[np.arange(idx.shape[0]), idx]
    return float(np.exp(np.mean(nll)))


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #
@dataclass
class DepthQuality:
    depth: int
    mean_kl_vs_full: float
    perplexity: float
    top1_agreement: float          # fraction of positions matching full-depth argmax
    task_accuracy: float | None    # only when a task suite is provided
    n_tokens: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Profiler (model path — requires torch + the loaded model)
# --------------------------------------------------------------------------- #
class QualityProfiler:
    """Measures the depth→quality-cost table for a loaded model.

    Parameters
    ----------
    cfg        : POISE config (depths default to budget_set ∪ {layer_total}).
    model      : loaded causal-LM exposing ``output_hidden_states``.
    tokenizer  : matching tokenizer.
    """

    def __init__(self, cfg: "PoiseConfig", model: Any, tokenizer: Any):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.layer_total = cfg.depth.layer_total

    def candidate_depths(self, depths: Sequence[int] | None = None) -> list[int]:
        if depths is not None:
            ds = sorted({int(d) for d in depths})
        else:
            ds = sorted({*(cfg_b for cfg_b in self.cfg.depth.budget_set), self.layer_total})
        full = self.layer_total
        # Always include the full-depth reference (cost = 0 by construction).
        if full not in ds:
            ds.append(full)
        return sorted(d for d in ds if 1 <= d <= full)

    def _forward_hidden_states(self, input_ids: Any):
        """Return (hidden_states tuple, full-depth logits) for a teacher-forced pass."""
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        # hidden_states: len L+1; hs[0]=embeddings, hs[d]=after d decoder layers.
        return out.hidden_states, out.logits

    def profile(
        self,
        prompts: Iterable[str],
        depths: Sequence[int] | None = None,
        *,
        dataset_tag: str = "calibration",
        task_suite: Sequence[tuple[str, str]] | None = None,
        max_length: int = 512,
    ) -> list[DepthQuality]:
        """Run the gate over ``prompts``. Returns one DepthQuality per candidate depth."""
        import torch

        ds = self.candidate_depths(depths)
        full = self.layer_total

        # accumulators per depth
        kl_sum = {d: 0.0 for d in ds}
        agree_sum = {d: 0 for d in ds}
        nll_sum = {d: 0.0 for d in ds}
        ntok = {d: 0 for d in ds}

        device = next(self.model.parameters()).device
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            hs, full_logits = self._forward_hidden_states(input_ids)

            # targets for perplexity: next-token over positions 0..T-2
            targets = input_ids[0, 1:].detach().cpu().numpy()
            full_lp = full_logits[0, :-1, :]
            full_argmax = torch.argmax(full_lp, dim=-1).detach().cpu().numpy()
            full_np = full_lp.float().detach().cpu().numpy()

            for d in ds:
                if d == full:
                    logits_d = full_lp
                else:
                    logits_d = project_to_logits(hs[d][0, :-1, :], self.model)
                q_np = logits_d.float().detach().cpu().numpy()
                kl = kl_divergence_logits(full_np, q_np)  # per position
                kl_sum[d] += float(np.sum(kl))
                d_argmax = np.argmax(q_np, axis=-1)
                agree_sum[d] += int(np.sum(d_argmax == full_argmax))
                logp = log_softmax(q_np, axis=-1)
                nll_sum[d] += float(-np.sum(logp[np.arange(targets.shape[0]), targets]))
                ntok[d] += int(targets.shape[0])

        task_acc = self._task_accuracy(task_suite, ds) if task_suite else {d: None for d in ds}

        results = []
        for d in ds:
            n = max(1, ntok[d])
            results.append(
                DepthQuality(
                    depth=d,
                    mean_kl_vs_full=kl_sum[d] / n,
                    perplexity=float(np.exp(nll_sum[d] / n)),
                    top1_agreement=agree_sum[d] / n,
                    task_accuracy=task_acc[d],
                    n_tokens=ntok[d],
                )
            )
        return results

    def _task_accuracy(
        self, task_suite: Sequence[tuple[str, str]], depths: Sequence[int]
    ) -> dict[int, float | None]:
        """Exact-match of the depth-``d`` greedy *first answer token* vs the gold answer's
        first token. A lightweight, model-only accuracy proxy for the gate; the richer
        generation-based suite lives in eval/quality.py (needs the runner)."""
        import torch

        device = next(self.model.parameters()).device
        correct = {d: 0 for d in depths}
        total = 0
        full = self.layer_total
        for prompt, answer in task_suite:
            ans_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            if not ans_ids:
                continue
            gold_first = ans_ids[0]
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            hs, full_logits = self._forward_hidden_states(input_ids)
            total += 1
            for d in depths:
                if d == full:
                    last = full_logits[0, -1, :]
                else:
                    last = project_to_logits(hs[d][0, -1, :], self.model)
                pred = int(torch.argmax(last).item())
                if pred == gold_first:
                    correct[d] += 1
        if total == 0:
            return {d: None for d in depths}
        return {d: correct[d] / total for d in depths}


# --------------------------------------------------------------------------- #
# Persistence + reporting
# --------------------------------------------------------------------------- #
def persist_profile(
    conn, results: Sequence[DepthQuality], dataset_tag: str = "calibration"
) -> None:
    from ..storage import models

    for r in results:
        models.insert_quality_profile(
            conn,
            depth=r.depth,
            mean_kl_vs_full=r.mean_kl_vs_full,
            perplexity=r.perplexity,
            task_accuracy=r.task_accuracy,
            dataset_tag=dataset_tag,
        )


def recommend_layer_min(
    results: Sequence[DepthQuality], max_kl: float
) -> int | None:
    """Smallest depth whose mean KL vs full is <= ``max_kl``.

    This is how the *usable* LAYER_MIN is set from measurement (CLAUDE.md §10).
    Returns None if no depth meets the threshold (a quality-collapse signal — do
    not paper over it; see §9).
    """
    ok = sorted(r.depth for r in results if r.mean_kl_vs_full <= max_kl)
    return ok[0] if ok else None


def format_table(results: Sequence[DepthQuality]) -> str:
    lines = [
        f"{'depth':>6} {'mean_KL':>10} {'ppl':>10} {'top1_agree':>11} "
        f"{'task_acc':>9} {'n_tok':>8}",
        "-" * 58,
    ]
    for r in sorted(results, key=lambda x: x.depth):
        ta = "-" if r.task_accuracy is None else f"{r.task_accuracy:.3f}"
        lines.append(
            f"{r.depth:>6} {r.mean_kl_vs_full:>10.4f} {r.perplexity:>10.3f} "
            f"{r.top1_agreement:>11.3f} {ta:>9} {r.n_tokens:>8}"
        )
    return "\n".join(lines)


def load_prompts(path: str | Path) -> list[str]:
    """Load calibration prompts: one per line (.txt) or {'prompt': ...} per line (.jsonl)."""
    import json

    p = Path(path)
    prompts: list[str] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if p.suffix == ".jsonl":
                obj = json.loads(line)
                prompts.append(obj["prompt"] if isinstance(obj, dict) else str(obj))
            else:
                prompts.append(line)
    return prompts


def run_gate(
    cfg: "PoiseConfig",
    prompts_path: str | Path,
    *,
    depths: Sequence[int] | None = None,
    dataset_tag: str = "calibration",
    max_kl: float = 0.10,
    persist: bool = True,
) -> list[DepthQuality]:
    """End-to-end gate: load model, profile, persist, print the table + LAYER_MIN.

    Requires torch + the (gated) model — this is an ON-DEVICE / GPU-box step.
    """
    from ..engine.model_loader import load_model

    model, tokenizer = load_model(cfg)
    profiler = QualityProfiler(cfg, model, tokenizer)
    prompts = load_prompts(prompts_path)
    results = profiler.profile(prompts, depths=depths, dataset_tag=dataset_tag)

    print(format_table(results))
    rec = recommend_layer_min(results, max_kl=max_kl)
    if rec is None:
        print(
            f"\n⚠ No depth meets mean_KL <= {max_kl}. This is a quality-collapse signal "
            f"— do NOT narrow silently. See CLAUDE.md §9 (LayerSkip adaptation / narrow "
            f"the range and report honestly / re-scope)."
        )
    else:
        print(f"\nRecommended usable LAYER_MIN (mean_KL <= {max_kl}): {rec}")

    if persist:
        from ..storage.db import init_db

        conn = init_db(cfg.storage.db_path)
        persist_profile(conn, results, dataset_tag=dataset_tag)
        conn.close()
    return results


__all__ = [
    "log_softmax",
    "kl_divergence_logits",
    "perplexity_from_logits",
    "DepthQuality",
    "QualityProfiler",
    "persist_profile",
    "recommend_layer_min",
    "format_table",
    "load_prompts",
    "run_gate",
]
