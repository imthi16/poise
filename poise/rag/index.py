"""FAISS index over SYNTHETIC/PUBLIC docs (CLAUDE.md §6, §9 — DEMO LAYER ONLY).

🔒 Synthetic/public data only. This module ships a synthetic-doc generator so the demo
needs no external corpus and never touches proprietary data.

To stay self-contained and testable offline, embeddings use a dependency-free hashing
embedder (public bag-of-words hashing, no downloaded model) and the index falls back to
a numpy brute-force search when faiss is unavailable. On a real box you can swap in a
sentence-transformer + faiss-cpu without changing the interface.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


# --------------------------------------------------------------------------- #
# Synthetic corpus generator (public-knowledge templates; clearly synthetic)
# --------------------------------------------------------------------------- #
_TOPICS = [
    ("thermal throttling", "When a processor's junction temperature exceeds a safe "
     "threshold, hardware throttling reduces clock frequency to protect the silicon, "
     "which lowers throughput."),
    ("layer budget", "Executing fewer transformer layers per token reduces compute and "
     "power at the cost of some output quality; the trade-off is measured per depth."),
    ("Jetson AGX Orin", "The NVIDIA Jetson AGX Orin is an edge module with a power budget "
     "selectable via nvpmodel; sustained load raises temperature toward a steady state."),
    ("RC thermal model", "A first-order RC model approximates a chip's temperature as it "
     "charges toward an equilibrium set by power draw and thermal resistance."),
    ("PID control", "A PID controller maps a temperature error to an actuator signal; it "
     "is reactive and lags slow thermal dynamics with long time constants."),
    ("energy per token", "Energy per token is the integral of power over time divided by "
     "tokens generated; reducing depth under load can lower it."),
    ("early exit", "Early-exit methods stop computation at an intermediate layer; classic "
     "approaches gate on input difficulty rather than device state."),
    ("hardware-state conditioning", "Conditioning execution depth on live hardware state "
     "such as temperature and power closes the loop between physics and compute."),
]


def generate_synthetic_corpus(
    out_dir: str | Path, n_docs: int = 40, seed: int = 0
) -> Path:
    """Write a synthetic JSONL corpus (id, text). Public-knowledge templates only."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "corpus.jsonl"
    with open(path, "w") as f:
        for i in range(n_docs):
            topic, fact = rng.choice(_TOPICS)
            qualifier = rng.choice(
                ["In practice, ", "Empirically, ", "On edge devices, ", "Under load, ",
                 "For on-device inference, "]
            )
            text = f"{qualifier}{fact} This note concerns {topic}."
            f.write(json.dumps({"id": f"doc-{i:03d}", "topic": topic, "text": text}) + "\n")
    return path


def load_corpus(path: str | Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# --------------------------------------------------------------------------- #
# Dependency-free hashing embedder (public; deterministic)
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            vec[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


# --------------------------------------------------------------------------- #
# Vector index (faiss if available; numpy brute force otherwise)
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    doc_id: str
    score: float
    text: str


class VectorIndex:
    def __init__(self, embedder: Optional[HashingEmbedder] = None):
        self.embedder = embedder or HashingEmbedder()
        self.docs: list[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._faiss = None

    def build(self, docs: Sequence[dict]) -> "VectorIndex":
        self.docs = list(docs)
        self._matrix = self.embedder.embed_many([d["text"] for d in self.docs])
        try:  # pragma: no cover - faiss optional
            import faiss

            index = faiss.IndexFlatIP(self.embedder.dim)
            index.add(self._matrix)
            self._faiss = index
        except Exception:
            self._faiss = None  # numpy brute-force fallback
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self._matrix is None or len(self.docs) == 0:
            return []
        q = self.embedder.embed(query).reshape(1, -1)
        k = min(k, len(self.docs))
        if self._faiss is not None:  # pragma: no cover - faiss optional
            scores, idx = self._faiss.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
        else:
            sims = (self._matrix @ q[0])
            order = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [Hit(self.docs[i]["id"], float(s), self.docs[i]["text"]) for i, s in pairs]


def build_index(cfg: "PoiseConfig", *, n_docs: int = 40, seed: int = 0) -> VectorIndex:
    """Ensure a synthetic corpus exists under the RAG data dir, then build the index."""
    data_dir = Path(cfg.rag.data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.exists():
        generate_synthetic_corpus(data_dir, n_docs=n_docs, seed=seed)
    docs = load_corpus(corpus_path)
    return VectorIndex().build(docs)


__all__ = [
    "generate_synthetic_corpus",
    "load_corpus",
    "HashingEmbedder",
    "VectorIndex",
    "Hit",
    "build_index",
]
