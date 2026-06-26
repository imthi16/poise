"""LangGraph retrieve→generate demo (CLAUDE.md §6, §9 — DEMO LAYER ONLY).

🔒 LangGraph and RAG never touch the inference engine internals. This graph consumes
POISE through the serving ``InferenceService`` (a black box) — retrieve top-k from the
synthetic FAISS index, build a context prompt, and call ``service.generate``.

🔒 Synthetic/public data only (see ``index.py``'s generator).

LangGraph is imported lazily; if it is not installed the same two-node pipeline runs
as a plain function, so the demo works either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypedDict

from .index import VectorIndex, build_index

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..serving.api import InferenceService


class RagState(TypedDict, total=False):
    query: str
    k: int
    hits: list
    answer: str
    sources: list
    run_id: Optional[str]
    max_new_tokens: int


def _build_prompt(query: str, hits) -> str:
    context = "\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
    return (
        "Use the retrieved context to answer the question concisely.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def make_nodes(index: VectorIndex, service: "InferenceService", mode: Optional[str] = None):
    def retrieve(state: RagState) -> RagState:
        hits = index.search(state["query"], k=state.get("k", 4))
        return {**state, "hits": hits,
                "sources": [{"doc_id": h.doc_id, "score": h.score} for h in hits]}

    def generate(state: RagState) -> RagState:
        prompt = _build_prompt(state["query"], state.get("hits", []))
        # 🔒 engine consumed as a black box via the serving service
        resp = service.generate(
            prompt, state.get("max_new_tokens", 128), mode=mode, return_trace=False
        )
        return {**state, "answer": resp["text"], "run_id": resp["run_id"]}

    return retrieve, generate


def build_graph(index: VectorIndex, service: "InferenceService", mode: Optional[str] = None):
    """Compile a LangGraph retrieve→generate graph (or return a plain-pipeline shim)."""
    retrieve, generate = make_nodes(index, service, mode)
    try:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(RagState)
        g.add_node("retrieve", retrieve)
        g.add_node("generate", generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()
    except Exception:
        # Fallback: identical two-step pipeline without the langgraph dependency.
        class _Pipeline:
            def invoke(self, state: RagState) -> RagState:
                return generate(retrieve(state))

        return _Pipeline()


def run_rag(
    cfg: "PoiseConfig",
    query: str,
    k: int = 4,
    *,
    service: Optional["InferenceService"] = None,
    index: Optional[VectorIndex] = None,
    mode: Optional[str] = None,
    max_new_tokens: int = 128,
) -> dict:
    """Run the demo end-to-end. Returns {answer, sources, run_id}."""
    if index is None:
        index = build_index(cfg)
    if service is None:
        from ..serving.api import InferenceService

        service = InferenceService(cfg)
    graph = build_graph(index, service, mode)
    out = graph.invoke({"query": query, "k": k, "max_new_tokens": max_new_tokens})
    return {
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "run_id": out.get("run_id"),
    }


__all__ = ["RagState", "make_nodes", "build_graph", "run_rag"]
