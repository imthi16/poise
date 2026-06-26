"""Optional static TensorRT-LLM baseline — fixed-depth ONLY (CLAUDE.md §2, §9, §11).

🔒 HARD CONSTRAINT: TensorRT-LLM compiles a STATIC graph; mid-inference per-token
depth changes are incompatible with it. It exists in this repo ONLY to produce a fast
FIXED-DEPTH baseline number and must NEVER be wired into the adaptive path.

tensorrt_llm is imported lazily and is optional; the harness/aggregation mirror the
llama.cpp baseline so results are comparable under the same stress protocol.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..eval.benchmark import RunMetrics, summarize_trace, trace_from_token_latencies

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..telemetry.reader import BaseTelemetryReader


class TensorRTLLMBaseline:
    """Fixed-depth TRT-LLM baseline. Optional; requires a prebuilt TRT engine."""

    def __init__(
        self,
        cfg: "PoiseConfig",
        telemetry_reader: "BaseTelemetryReader",
        *,
        engine_dir: Optional[str] = None,
        fixed_depth: Optional[int] = None,
        clock=time.perf_counter,
    ):
        self.cfg = cfg
        self.reader = telemetry_reader
        self.engine_dir = engine_dir
        # 🔒 fixed depth only — a single constant for the whole run.
        self.fixed_depth = fixed_depth or cfg.depth.layer_total
        self._clock = clock
        self._runner = None

    def load(self):  # pragma: no cover - requires tensorrt_llm + a built engine
        try:
            import tensorrt_llm  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "tensorrt_llm not installed; the TRT-LLM baseline is optional. It is a "
                "FIXED-DEPTH baseline only and must never touch the adaptive path."
            ) from e
        if not self.engine_dir:
            raise RuntimeError("no TRT-LLM engine_dir provided.")
        # Engine/runner construction is environment-specific; wire your built engine here.
        raise NotImplementedError(
            "Provide a prebuilt TRT-LLM engine and instantiate its ModelRunner here. "
            "Kept unimplemented to avoid pinning a fragile TRT-LLM version; the metric "
            "path below is identical to the llama.cpp baseline."
        )

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, RunMetrics]:  # pragma: no cover
        if self._runner is None:
            self.load()
        # On-device: stream tokens from the TRT runner, timing each + sampling telemetry,
        # then summarize identically to the other baselines.
        raise NotImplementedError("TRT-LLM generation requires a built engine (on-device).")

    def _summarize(self, latencies_s, ttft_ms) -> RunMetrics:
        trace = trace_from_token_latencies(latencies_s, self.reader, budget=self.fixed_depth)
        return summarize_trace(
            trace, ttft_ms=ttft_ms, temp_setpoint_c=self.cfg.thermal.temp_setpoint_c
        )


__all__ = ["TensorRTLLMBaseline"]
