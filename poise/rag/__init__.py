"""RAG demo (CLAUDE.md §9) — DEMO/APPLICATION LAYER ONLY.

🔒 Consumes POISE as a black box via the serving API; never touches engine internals.
🔒 Synthetic/public data only (ships a synthetic-doc generator).
"""

from __future__ import annotations

from .index import (
    generate_synthetic_corpus,
    load_corpus,
    HashingEmbedder,
    VectorIndex,
    build_index,
)
from .graph import run_rag, build_graph

__all__ = [
    "generate_synthetic_corpus",
    "load_corpus",
    "HashingEmbedder",
    "VectorIndex",
    "build_index",
    "run_rag",
    "build_graph",
]
