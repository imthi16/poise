"""Build a real public multiple-choice task suite from AI2 ARC (CLAUDE.md §9: public data only).

Writes ``data/eval/arc_mc.jsonl`` with one normalized MC item per line:
    {"question": str, "choices": [str, ...], "answer_idx": int, "source": "arc-easy"|"arc-challenge"}

ARC is CC-BY-SA public data (allenai/ai2_arc). This replaces the 5 toy items in
``data/synthetic/task_suite.jsonl`` so the step-3 gate can measure a real *task-level*
quality cost per depth (PRONG 1), not just KL/top-1 vs full-32.

    python3.10 scripts/build_arc_suite.py --n-easy 150 --n-challenge 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(split_name: str, n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", split_name, split="test")
    ds = ds.shuffle(seed=seed)
    items = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        ans = row["answerKey"]
        if ans not in labels:
            continue  # a handful of malformed rows
        items.append(
            {
                "question": row["question"].strip(),
                "choices": [t.strip() for t in texts],
                "answer_idx": labels.index(ans),
                "source": split_name.lower().replace("_", "-"),
            }
        )
        if len(items) >= n:
            break
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ARC multiple-choice task suite")
    ap.add_argument("--n-easy", type=int, default=150)
    ap.add_argument("--n-challenge", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/eval/arc_mc.jsonl")
    args = ap.parse_args()

    items = []
    if args.n_easy:
        items += _load("ARC-Easy", args.n_easy, args.seed)
    if args.n_challenge:
        items += _load("ARC-Challenge", args.n_challenge, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    n_e = sum(1 for it in items if it["source"] == "arc-easy")
    n_c = sum(1 for it in items if it["source"] == "arc-challenge")
    print(f"wrote {len(items)} items to {out}  (easy={n_e} challenge={n_c})")


if __name__ == "__main__":
    main()
