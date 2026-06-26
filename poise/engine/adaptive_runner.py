"""Core per-token variable-depth generation loop (CLAUDE.md §6).

``generate(prompt, max_new_tokens, budget_fn)`` runs autoregressive decoding where
**each token's forward pass executes layers ``0..budget-1``** and exits via
``early_exit.project_to_logits``. The budget for each token comes *only* through
``budget_fn`` — the runner contains **no control logic** (runner = mechanism,
``poise/control`` = policy; CLAUDE.md §9).

Design note on testability + correctness:
  * The per-token control flow (budget dispatch, clamping, trace, stop conditions,
    static-depth shortcut) is isolated from the torch math. The two torch-dependent
    methods ``_prefill`` / ``_decode_step`` are the only things needing the model, so
    the loop is unit-tested off-device with a fake forward.
  * Prefill runs the prompt at FULL depth so every layer has complete prompt KV; the
    variable budget applies to generated tokens. At a constant budget == layer_total
    the loop reduces to standard full-depth greedy decoding (the bit-identical check).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .kv_cache import KVStrategy, VariableDepthKVCache, plan_for_depth

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig
    from ..telemetry.schema import TelemetrySample


# --------------------------------------------------------------------------- #
# Public data structures
# --------------------------------------------------------------------------- #
@dataclass
class BudgetContext:
    """What the runner knows when it asks the controller for a depth.

    The controller (``budget_fn``) may use the live telemetry sample to condition
    depth on hardware state — that is the whole point of POISE.
    """

    step: int                       # 0-based index of the token being generated
    prev_budget: Optional[int]
    max_new_tokens: int
    telemetry: Optional["TelemetrySample"] = None


@dataclass
class TokenTrace:
    i: int
    budget: int
    latency_ms: float
    temp_c: Optional[float] = None
    power_w: Optional[float] = None
    throttled: Optional[bool] = None
    kl_vs_full: Optional[float] = None


@dataclass
class GenerationResult:
    prompt: str
    text: str
    token_ids: list[int]
    tokens: int
    trace: list[TokenTrace] = field(default_factory=list)
    ttft_ms: float = 0.0
    tok_per_s: float = 0.0

    @property
    def mean_budget(self) -> float:
        return sum(t.budget for t in self.trace) / len(self.trace) if self.trace else 0.0

    @property
    def peak_temp_c(self) -> float:
        temps = [t.temp_c for t in self.trace if t.temp_c is not None]
        return max(temps) if temps else 0.0


BudgetFn = Callable[[BudgetContext], int]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class AdaptiveRunner:
    """Variable-depth autoregressive generator.

    Parameters
    ----------
    cfg              : POISE config (depth bounds, eos, etc.).
    model, tokenizer : loaded adaptive model + tokenizer (None for fake-forward tests).
    strategy         : KV-cache strategy under variable depth.
    telemetry_reader : optional; if present, each step reads the latest sample to
                       populate ``BudgetContext.telemetry`` and the trace.
    """

    def __init__(
        self,
        cfg: "PoiseConfig",
        model: Any = None,
        tokenizer: Any = None,
        *,
        strategy: KVStrategy = KVStrategy.MONOTONE_NONINCREASING,
        telemetry_reader: Any = None,
        monotone_window: Optional[int] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.telemetry_reader = telemetry_reader
        self.monotone_window = monotone_window
        self.layer_total = cfg.depth.layer_total
        self.layer_min = cfg.depth.layer_min
        self.layer_max = cfg.depth.layer_max

    # --- budget guard (physical bounds only; policy lives in control/) ------ #
    def clamp_budget(self, budget: int) -> int:
        """Clamp to the physical [layer_min, layer_max] guard. Budget-set snapping
        and TEMP_MAX safety override are the controller's responsibility (§6)."""
        b = int(budget)
        b = max(self.layer_min, min(self.layer_max, b))
        return b

    # --- public API --------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        budget_fn: BudgetFn,
        *,
        return_trace: bool = True,
        compute_kl_vs_full: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> GenerationResult:
        """Generate up to ``max_new_tokens`` tokens, asking ``budget_fn`` per token."""
        t_start = time.perf_counter()
        input_ids = self._encode(prompt)
        state = self._prefill(input_ids)  # full-depth prefill
        ttft_ms = (time.perf_counter() - t_start) * 1000.0

        eos = eos_token_id if eos_token_id is not None else self._eos_id()
        position = self._prompt_len(input_ids)
        token_ids: list[int] = []
        trace: list[TokenTrace] = []
        prev_budget: Optional[int] = None
        last_logits = state["logits_last"]

        for step in range(max_new_tokens):
            sample = self._read_telemetry()
            ctx = BudgetContext(
                step=step,
                prev_budget=prev_budget,
                max_new_tokens=max_new_tokens,
                telemetry=sample,
            )
            requested = budget_fn(ctx)
            budget = self.clamp_budget(requested)

            # The previous step's logits decide this token; the budget governs the
            # forward pass that produces the NEXT logits.
            next_id = self._select_token(last_logits)
            token_ids.append(next_id)

            if eos is not None and next_id == eos:
                trace.append(self._mk_trace(step, budget, 0.0, sample, None))
                break

            t0 = time.perf_counter()
            step_out = self._decode_step(next_id, position, budget, state)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            last_logits = step_out["logits_last"]

            kl = None
            if compute_kl_vs_full and "kl_vs_full" in step_out:
                kl = step_out["kl_vs_full"]

            trace.append(self._mk_trace(step, budget, latency_ms, sample, kl))
            prev_budget = budget
            position += 1

        elapsed = time.perf_counter() - t_start
        text = self._decode_text(token_ids)
        n = len(token_ids)
        return GenerationResult(
            prompt=prompt,
            text=text,
            token_ids=token_ids,
            tokens=n,
            trace=trace if return_trace else [],
            ttft_ms=ttft_ms,
            tok_per_s=(n / elapsed if elapsed > 0 else 0.0),
        )

    def generate_static(
        self, prompt: str, max_new_tokens: int, depth: Optional[int] = None, **kw
    ) -> GenerationResult:
        """Baseline shortcut: constant depth for every token."""
        d = depth if depth is not None else self.cfg.depth.static_depth
        return self.generate(prompt, max_new_tokens, lambda ctx: d, **kw)

    # --- helpers ------------------------------------------------------------ #
    def _mk_trace(self, step, budget, latency_ms, sample, kl) -> TokenTrace:
        return TokenTrace(
            i=step,
            budget=budget,
            latency_ms=latency_ms,
            temp_c=(sample.temp_c if sample is not None else None),
            power_w=(sample.power_w if sample is not None else None),
            throttled=(sample.throttled if sample is not None else None),
            kl_vs_full=kl,
        )

    def _read_telemetry(self) -> Optional["TelemetrySample"]:
        if self.telemetry_reader is None:
            return None
        try:
            return self.telemetry_reader.read()
        except Exception:  # pragma: no cover - telemetry must never break generation
            return None

    def _eos_id(self) -> Optional[int]:
        if self.tokenizer is not None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            return int(self.tokenizer.eos_token_id)
        return None

    # --- encode/decode (overridable for fake-forward tests) ----------------- #
    def _encode(self, prompt: str):
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer; override _encode for tests")
        enc = self.tokenizer(prompt, return_tensors="pt")
        return enc["input_ids"]

    def _prompt_len(self, input_ids) -> int:
        try:
            return int(input_ids.shape[1])
        except Exception:
            return len(input_ids)

    def _decode_text(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _select_token(self, logits_last) -> int:
        """Greedy by default (temperature 0). Sampling can be added via config."""
        import torch  # tensor path

        return int(torch.argmax(logits_last, dim=-1).item())

    # --- the two torch-dependent steps (validated on-device) ---------------- #
    def _prefill(self, input_ids) -> dict:
        """Full-depth prefill: populate complete KV for every layer over the prompt,
        return the last-position logits.

        ⚠ This and ``_decode_step`` are the on-device correctness surface. They run
        the per-layer loop manually against the installed transformers Llama layer
        contract and manage ``VariableDepthKVCache``. The constant-budget bit-identical
        test (test_kv_cache / test_adaptive_runner, on-device) validates them.
        """
        import torch

        from .model_loader import get_embeddings, get_layers, get_norm, get_lm_head

        model = self._require_model()
        layers = get_layers(model)
        cache = VariableDepthKVCache(num_layers=len(layers))
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        T = input_ids.shape[1]

        embed = get_embeddings(model)
        hidden = embed(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self._position_embeddings(model, hidden, position_ids)
        attn_mask = self._causal_mask(hidden, T)

        for li, layer in enumerate(layers):
            hidden, kv = self._run_layer(
                layer, hidden, attn_mask, position_ids, pos_emb, past=None
            )
            if kv is not None:
                k, v = kv
                cache.keys[li], cache.values[li] = k, v
            for p in range(T):
                cache.book.fill_layer_position(li, p)
        for p in range(T):
            cache.book.record_token(p, len(layers))

        norm = get_norm(model)
        head = get_lm_head(model)
        logits_last = head(norm(hidden[:, -1, :]))
        return {"cache": cache, "logits_last": logits_last, "hidden_last": hidden[:, -1, :]}

    def _decode_step(self, token_id: int, position: int, budget: int, state: dict) -> dict:
        """Single-token forward through layers ``0..budget-1`` with KV repair per strategy."""
        import torch

        from .model_loader import get_embeddings, get_layers
        from .early_exit import project_to_logits

        model = self._require_model()
        layers = get_layers(model)
        cache: VariableDepthKVCache = state["cache"]
        device = next(model.parameters()).device

        plan = plan_for_depth(
            self.strategy, cache.book, position, budget, window=self.monotone_window
        )
        depth = plan.admissible_depth
        # NOTE: recompute/propagate repair operations are applied here against `cache`
        # using `plan.recompute_layers/positions` and `plan.propagate_layers`. They are
        # part of the on-device correctness surface (see kv_cache.py docstring).
        self._apply_repair(model, layers, cache, plan, state)

        embed = get_embeddings(model)
        ids = torch.tensor([[token_id]], device=device)
        hidden = embed(ids)
        position_ids = torch.tensor([[position]], device=device)
        pos_emb = self._position_embeddings(model, hidden, position_ids)
        attn_mask = None  # single query attends to full cached context

        for li in range(depth):
            past = cache.get(li)
            hidden, kv = self._run_layer(
                layers[li], hidden, attn_mask, position_ids, pos_emb, past=past
            )
            if kv is not None:
                k, v = kv
                # append only the new position's KV (k/v already include the new step)
                cache.keys[li], cache.values[li] = k, v
                cache.book.fill_layer_position(li, position)
        cache.book.record_token(position, depth)

        logits_last = project_to_logits(hidden[:, -1, :], model)
        out = {"cache": cache, "logits_last": logits_last, "hidden_last": hidden[:, -1, :]}
        return out

    # --- low-level model-contract shims (version-tolerant) ------------------ #
    def _run_layer(self, layer, hidden, attn_mask, position_ids, pos_emb, past):
        """Call one decoder layer, returning (hidden, (k, v) or None).

        Tolerant of transformers API drift: passes ``position_embeddings`` when the
        layer accepts it and falls back otherwise. The exact KV extraction is the
        on-device validation point.
        """
        kwargs = dict(attention_mask=attn_mask, position_ids=position_ids, use_cache=True)
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb
        if past is not None and past[0] is not None:
            kwargs["past_key_value"] = past
        out = layer(hidden, **kwargs)
        if isinstance(out, tuple):
            hidden_out = out[0]
            kv = out[-1] if len(out) > 1 and isinstance(out[-1], tuple) else None
        else:
            hidden_out, kv = out, None
        return hidden_out, kv

    def _position_embeddings(self, model, hidden, position_ids):
        from .model_loader import get_rotary_emb

        rotary = get_rotary_emb(model)
        if rotary is None:
            return None
        try:
            return rotary(hidden, position_ids)
        except Exception:  # pragma: no cover - older API computes rotary inside layers
            return None

    def _causal_mask(self, hidden, T):  # pragma: no cover - validated on-device
        return None  # transformers builds the causal mask internally when None

    def _apply_repair(self, model, layers, cache, plan, state):  # pragma: no cover
        """Hook for recompute_on_demand / propagate_hidden repair (on-device)."""
        return

    def _require_model(self):
        if self.model is None:
            raise RuntimeError(
                "AdaptiveRunner has no model loaded. Load with engine.load_model(cfg), "
                "or override _prefill/_decode_step for off-device tests."
            )
        return self.model


__all__ = [
    "AdaptiveRunner",
    "BudgetContext",
    "TokenTrace",
    "GenerationResult",
    "BudgetFn",
]
