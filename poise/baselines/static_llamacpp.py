"""Static llama.cpp baseline (GGUF Q4_K_M) — fixed-depth comparison (CLAUDE.md §6, §11).

Runs the GGUF **Q4_K_M** model at FIXED (full) depth via llama.cpp under the same
stress protocol, producing throughput / energy / thermal metrics for comparison
against the adaptive path.

🔒 This is a BASELINE ONLY. Q4_K_M is a GGUF/llama.cpp format and lives exclusively
here — it is NEVER loaded into the PyTorch adaptive path (that path is fp16/bnb-4bit).
llama.cpp runs the whole network at fixed depth; this is the "static-full-32" point in
the comparison set.

llama-cpp-python is imported lazily (``pip install -e .[baselines]``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..eval.benchmark import RunMetrics, summarize_trace, trace_from_token_latencies

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..telemetry.reader import BaseTelemetryReader


class LlamaCppBaseline:
    def __init__(
        self,
        cfg: "PoiseConfig",
        telemetry_reader: "BaseTelemetryReader",
        *,
        gguf_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        clock=time.perf_counter,
    ):
        self.cfg = cfg
        self.reader = telemetry_reader
        self.gguf_path = gguf_path or cfg.raw.get("baseline", {}).get("gguf_path")
        self.n_gpu_layers = n_gpu_layers
        self._clock = clock
        self._llm = None

    def load(self):  # pragma: no cover - requires llama-cpp-python + GGUF
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "llama-cpp-python not installed (pip install -e .[baselines]). The "
                "Q4_K_M GGUF baseline runs via llama.cpp only."
            ) from e
        if not self.gguf_path:
            raise RuntimeError("no GGUF path configured (POISE_GGUF_PATH / model.yaml baseline).")
        self._llm = Llama(
            model_path=self.gguf_path,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )
        return self._llm

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, RunMetrics]:
        """Stream-generate, timing each token and sampling telemetry per token."""
        if self._llm is None:  # pragma: no cover - on-device
            self.load()

        latencies_s: list[float] = []
        text_parts: list[str] = []
        t0 = self._clock()
        ttft_ms = 0.0
        last = t0
        # llama-cpp-python streaming API
        stream = self._llm(  # pragma: no cover - on-device
            prompt, max_tokens=max_new_tokens, stream=True, temperature=0.0
        )
        for i, chunk in enumerate(stream):  # pragma: no cover - on-device
            now = self._clock()
            if i == 0:
                ttft_ms = (now - t0) * 1000.0
            latencies_s.append(now - last)
            last = now
            text_parts.append(chunk["choices"][0]["text"])

        return "".join(text_parts), self._summarize(latencies_s, ttft_ms)

    def _summarize(self, latencies_s, ttft_ms) -> RunMetrics:
        # Full-depth fixed baseline => budget == layer_total for every token.
        trace = trace_from_token_latencies(
            latencies_s, self.reader, budget=self.cfg.depth.layer_total
        )
        return summarize_trace(
            trace, ttft_ms=ttft_ms, temp_setpoint_c=self.cfg.thermal.temp_setpoint_c
        )


__all__ = ["LlamaCppBaseline"]
