"""PRONG 1 — per-depth TASK-LEVEL quality cost on a real public benchmark (CLAUDE.md §5 step 3, §9).

The step-3 gate (``quality_profile.py``) measured KL / perplexity / top-1 agreement vs
full-32. Those are distributional signals; the open question is whether the depth-reduction
"quality collapse" survives at the *task* level. This script measures **multiple-choice
accuracy per depth** on ARC using length-normalized log-likelihood scoring — the standard,
non-gameable MC eval (lm-eval "acc_norm"), NOT a first-token proxy.

For each question + candidate answer, we run ONE teacher-forced forward with
``output_hidden_states=True`` and, at each depth ``d``, project the depth-``d`` hidden
states to logits (``early_exit.project_to_logits``) and sum the log-prob of the answer's
continuation tokens. The choice with the highest (length-normalized) log-prob is the
prediction. Accuracy is compared against the full-32 reference.

🔒 Honesty: every number here is a measured forward-pass output. No hardcoded results.
Run on a GPU box (4090) or the Jetson with the real gated model. Set --adapter to profile
the LayerSkip-adapted model; omit it for the raw base model.

    POISE_ADAPTER_PATH=data/results/layerskip_adapter \\
        python3.10 scripts/prong1_task_gate.py --suite data/eval/arc_mc.jsonl \\
        --depths 16,20,24,28,30,32 --tag arc-v1adapter
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_suite(path: str | Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def mc_accuracy_by_depth(model, tokenizer, suite, depths, layer_total, max_ctx=1024):
    """Length-normalized log-likelihood MC accuracy at each depth. Returns
    (acc_norm[d], acc_raw[d]) dicts."""
    import numpy as np
    import torch

    from poise.engine.early_exit import project_to_logits

    device = next(model.parameters()).device
    correct_norm = {d: 0 for d in depths}
    correct_raw = {d: 0 for d in depths}
    total = 0

    for item in suite:
        q = item["question"]
        choices = item["choices"]
        gold = int(item["answer_idx"])
        # context: "Question: <q>\nAnswer:" — choices scored as the continuation.
        ctx = f"Question: {q}\nAnswer:"
        ctx_ids = tokenizer(ctx, return_tensors="pt", truncation=True,
                            max_length=max_ctx)["input_ids"][0]
        n_ctx = ctx_ids.shape[0]

        # per-depth score for each choice
        scores_norm = {d: [] for d in depths}
        scores_raw = {d: [] for d in depths}
        for ch in choices:
            cont_ids = tokenizer(" " + ch.strip(), add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
            if cont_ids.shape[0] == 0:
                for d in depths:
                    scores_norm[d].append(-1e9)
                    scores_raw[d].append(-1e9)
                continue
            full_ids = torch.cat([ctx_ids, cont_ids]).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True, use_cache=False)
            # continuation targets sit at positions [n_ctx .. end]; predicted from
            # the logits at the preceding position.
            tgt = full_ids[0, n_ctx:]
            for d in depths:
                if d == layer_total:
                    logits = out.logits[0]
                else:
                    logits = project_to_logits(out.hidden_states[d][0], model)
                logp = torch.log_softmax(logits.float(), dim=-1)
                # log p(tgt_i | ..tgt_{i-1}) uses logp at index (n_ctx + i - 1)
                pos = torch.arange(n_ctx - 1, n_ctx - 1 + tgt.shape[0], device=device)
                tok_lp = logp[pos, tgt].sum().item()
                scores_raw[d].append(tok_lp)
                scores_norm[d].append(tok_lp / tgt.shape[0])
        total += 1
        for d in depths:
            if int(np.argmax(scores_norm[d])) == gold:
                correct_norm[d] += 1
            if int(np.argmax(scores_raw[d])) == gold:
                correct_raw[d] += 1

    acc_norm = {d: correct_norm[d] / max(1, total) for d in depths}
    acc_raw = {d: correct_raw[d] / max(1, total) for d in depths}
    return acc_norm, acc_raw, total


def main() -> None:
    ap = argparse.ArgumentParser(description="PRONG 1: per-depth MC task accuracy gate")
    ap.add_argument("--suite", default="data/eval/arc_mc.jsonl")
    ap.add_argument("--depths", default="16,20,24,28,30,32")
    ap.add_argument("--adapter", default=None,
                    help="LayerSkip adapter dir (sets POISE_ADAPTER_PATH)")
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--tag", default="arc")
    ap.add_argument("--out", default="data/results/prong1")
    args = ap.parse_args()

    if args.adapter:
        os.environ["POISE_ADAPTER_PATH"] = args.adapter

    from poise.config import load_config
    from poise.engine.model_loader import load_model, get_num_layers

    cfg = load_config()
    model, tokenizer = load_model(cfg)
    layer_total = get_num_layers(model)
    depths = sorted({int(x) for x in args.depths.split(",")} | {layer_total})
    depths = [d for d in depths if 1 <= d <= layer_total]

    suite = load_suite(args.suite)
    if args.limit:
        suite = suite[: args.limit]

    print(f"[prong1] model={cfg.model.model_id} adapter={args.adapter or 'NONE (base)'}")
    print(f"[prong1] suite={args.suite} n={len(suite)} depths={depths} L={layer_total}")

    acc_norm, acc_raw, total = mc_accuracy_by_depth(
        model, tokenizer, suite, depths, layer_total
    )

    full = layer_total
    print(f"\n{'depth':>6} {'acc_norm':>9} {'acc_raw':>9} {'Δacc_norm_vs_full':>18} "
          f"{'rel_loss_%':>11}")
    print("-" * 58)
    rows = []
    for d in depths:
        d_an = acc_norm[full] - acc_norm[d]
        rel = 100.0 * d_an / acc_norm[full] if acc_norm[full] else 0.0
        print(f"{d:>6} {acc_norm[d]:>9.4f} {acc_raw[d]:>9.4f} {d_an:>18.4f} {rel:>11.2f}")
        rows.append({"depth": d, "acc_norm": acc_norm[d], "acc_raw": acc_raw[d],
                     "delta_acc_norm_vs_full": d_an, "rel_loss_pct": rel})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "model_id": cfg.model.model_id,
        "adapter": args.adapter,
        "suite": args.suite,
        "n_questions": total,
        "layer_total": layer_total,
        "acc_norm_full": acc_norm[full],
        "rows": rows,
    }
    fp = out / f"prong1_{args.tag}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[prong1] wrote {fp}")


if __name__ == "__main__":
    main()
