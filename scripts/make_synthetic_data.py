#!/usr/bin/env python3
"""Generate all SYNTHETIC datasets POISE needs (CLAUDE.md §9 — public/synthetic only).

Writes under POISE_RAG_DATA_DIR (default data/synthetic):
  * corpus.jsonl       — RAG demo corpus
  * eval_prompts.jsonl — stress/eval generation prompts
  * ppl_corpus.txt     — perplexity corpus (one passage per line)
  * task_suite.jsonl   — small exact-match task suite {prompt, answer}

🔒 No proprietary data anywhere. Everything here is generated from public-knowledge
templates so the repo reproduces with zero external corpora.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Allow running as `python3 scripts/make_synthetic_data.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poise.config import load_config  # noqa: E402
from poise.rag.index import generate_synthetic_corpus  # noqa: E402

_EVAL_PROMPTS = [
    "Explain thermal throttling in one paragraph.",
    "Summarize how a PID controller regulates temperature.",
    "Why does executing fewer transformer layers save energy?",
    "Describe the trade-off between inference depth and output quality.",
    "What is energy-per-token and how is it measured?",
    "How does an RC model approximate chip temperature under load?",
    "Contrast input-difficulty early exit with hardware-state conditioning.",
    "What happens to throughput when a processor hits its thermal ceiling?",
]

_PPL_PASSAGES = [
    "Sustained inference load raises the junction temperature toward a steady state set "
    "by power draw and thermal resistance.",
    "Reducing the number of executed layers lowers compute and power at a measured cost "
    "in output quality.",
    "A reactive controller lags slow thermal dynamics because the time constants span "
    "seconds to minutes.",
    "Energy per token integrates instantaneous power over the generation window and "
    "divides by the number of tokens produced.",
    "Hardware-state-conditioned execution depth closes the loop between device physics "
    "and per-token compute.",
]

_TASK_SUITE = [
    {"prompt": "The capital of France is", "answer": "Paris"},
    {"prompt": "Two plus two equals", "answer": "4"},
    {"prompt": "Water freezes at zero degrees", "answer": "Celsius"},
    {"prompt": "The opposite of hot is", "answer": "cold"},
    {"prompt": "A triangle has this many sides:", "answer": "3"},
]


def main() -> None:
    cfg = load_config()
    out = Path(cfg.rag.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    corpus = generate_synthetic_corpus(out, n_docs=60, seed=0)

    with open(out / "eval_prompts.jsonl", "w") as f:
        for i in range(24):
            f.write(json.dumps({"prompt": rng.choice(_EVAL_PROMPTS)}) + "\n")

    with open(out / "ppl_corpus.txt", "w") as f:
        for _ in range(40):
            f.write(rng.choice(_PPL_PASSAGES) + "\n")

    with open(out / "task_suite.jsonl", "w") as f:
        for item in _TASK_SUITE:
            f.write(json.dumps(item) + "\n")

    print(f"Synthetic data written to {out}/ (corpus={corpus.name}, eval_prompts, "
          f"ppl_corpus, task_suite). All synthetic/public — no proprietary data.")


if __name__ == "__main__":
    main()
