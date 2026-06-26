"""Fixed-depth baselines (CLAUDE.md §9): comparison numbers ONLY.

🔒 llama.cpp (Q4_K_M GGUF) and TensorRT-LLM are static, fixed-depth baselines. Neither
is ever wired into the adaptive path. Q4_K_M is llama.cpp-only; TRT-LLM compiles a
static graph incompatible with per-token depth changes.
"""

from __future__ import annotations

from .static_llamacpp import LlamaCppBaseline
from .static_trtllm import TensorRTLLMBaseline

__all__ = ["LlamaCppBaseline", "TensorRTLLMBaseline"]
