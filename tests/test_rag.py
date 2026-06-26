"""RAG demo: synthetic corpus + retrieval + black-box engine call (CLAUDE.md §9, §10).

🔒 Synthetic data only; engine consumed via the serving service. Runs fully off-device
(hashing embedder + numpy index + mock engine).
"""

from __future__ import annotations

import numpy as np

from poise.rag.index import (
    HashingEmbedder,
    VectorIndex,
    build_index,
    generate_synthetic_corpus,
    load_corpus,
)
from poise.rag.graph import run_rag


def test_synthetic_corpus_generation(tmp_path):
    path = generate_synthetic_corpus(tmp_path, n_docs=20, seed=0)
    docs = load_corpus(path)
    assert len(docs) == 20
    assert all("text" in d and "id" in d for d in docs)
    # deterministic given the seed
    path2 = generate_synthetic_corpus(tmp_path / "b", n_docs=20, seed=0)
    assert [d["text"] for d in load_corpus(path2)] == [d["text"] for d in docs]


def test_hashing_embedder_deterministic_and_normalized():
    emb = HashingEmbedder(dim=128)
    a = emb.embed("thermal throttling on the jetson")
    b = emb.embed("thermal throttling on the jetson")
    np.testing.assert_array_equal(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6


def test_index_retrieves_relevant_doc():
    docs = [
        {"id": "d0", "text": "Thermal throttling reduces clock frequency to protect silicon."},
        {"id": "d1", "text": "A PID controller maps temperature error to an actuator signal."},
        {"id": "d2", "text": "Energy per token is power integrated over time per token."},
    ]
    idx = VectorIndex(HashingEmbedder(dim=512)).build(docs)
    hits = idx.search("what causes throttling and clock reduction?", k=2)
    assert hits[0].doc_id == "d0"  # most relevant doc ranked first
    assert len(hits) == 2


def test_build_index_generates_corpus_if_missing(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("POISE_RAG_DATA_DIR", str(tmp_path / "syn"))
    from poise.config import load_config

    cfg2 = load_config()
    idx = build_index(cfg2, n_docs=15, seed=1)
    assert len(idx.docs) == 15
    assert (tmp_path / "syn" / "corpus.jsonl").exists()


def test_run_rag_end_to_end_black_box(cfg, tmp_path, monkeypatch):
    """retrieve -> build context -> call engine via the serving service (black box)."""
    monkeypatch.setenv("POISE_RAG_DATA_DIR", str(tmp_path / "syn"))
    monkeypatch.setenv("POISE_DB_PATH", str(tmp_path / "rag.db"))
    from poise.config import load_config
    from poise.serving.api import InferenceService

    cfg2 = load_config()
    service = InferenceService(cfg2)  # mock engine
    out = run_rag(cfg2, "How does reducing depth affect energy per token?", k=3,
                  service=service, max_new_tokens=8)
    assert "answer" in out and out["run_id"] is not None
    assert len(out["sources"]) == 3
    assert all("doc_id" in s and "score" in s for s in out["sources"])


def test_api_rag_endpoint(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from poise.config import load_config
    from poise.serving.api import InferenceService, create_app

    monkeypatch.setenv("POISE_RAG_DATA_DIR", str(tmp_path / "syn"))
    monkeypatch.setenv("POISE_DB_PATH", str(tmp_path / "rag_api.db"))
    cfg = load_config()
    client = TestClient(create_app(cfg, service=InferenceService(cfg)))
    r = client.post("/v1/rag/query", json={"query": "explain layer budgets", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and len(body["sources"]) == 2
