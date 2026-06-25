"""Load the HF model and expose its internals for variable-depth execution.

CLAUDE.md §6 rules:
  * 🔒 Adaptive path loads **fp16 or bnb-4bit** only. ``q4_k_m`` (GGUF) is rejected
    here with a pointer to the llama.cpp baseline.
  * Confirm the decoder block count equals ``POISE_LAYER_TOTAL`` (32); fail otherwise.

torch/transformers are imported lazily inside the functions so this module (and
the package) imports fine off-device where the heavy deps / gated model are absent.
The rest of the stack depends only on these accessors' *shapes*, not on a live model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoiseConfig


class ModelLoadError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:  # pragma: no cover - off-device
        raise ModelLoadError(
            "The adaptive engine requires torch + transformers (install the extras: "
            "`pip install -e .[engine]`, or the CUDA torch build matching your Jetson). "
            "Off-device, develop against the mock telemetry / simulator paths instead."
        ) from e
    import torch
    import transformers

    return torch, transformers


def load_model(cfg: "PoiseConfig") -> Tuple[Any, Any]:
    """Load model + tokenizer per config. Returns ``(model, tokenizer)``.

    Enforces the fp16 / bnb-4bit constraint and the exact layer count.
    """
    # Enforce the dtype constraint BEFORE importing torch so the rejection is
    # independent of whether the heavy deps are installed (defense in depth — the
    # config loader also rejects this).
    dtype = cfg.model.dtype.lower()
    if dtype in ("q4_k_m", "q4km", "gguf"):
        raise ModelLoadError(
            "Refusing to load q4_k_m in the PyTorch adaptive path: that is a GGUF / "
            "llama.cpp format and is NOT loadable here. Use fp16 or bnb-4bit for the "
            "adaptive path; q4_k_m is reserved for poise/baselines/static_llamacpp.py."
        )

    torch, transformers = _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ref = cfg.model.model_path or cfg.model.model_id
    token = cfg.model.hf_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if token:
        kwargs["token"] = token

    if dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "bnb-4bit":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise ModelLoadError(
                "bnb-4bit requested but bitsandbytes/transformers quant config is "
                "unavailable. Install `pip install -e .[quant]`."
            ) from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ModelLoadError(f"unsupported adaptive dtype: {cfg.model.dtype!r}")

    if cfg.model.device == "cuda":
        kwargs["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(model_ref, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    model.eval()

    n = get_num_layers(model)
    if n != cfg.depth.layer_total:
        raise ModelLoadError(
            f"loaded model has {n} decoder layers but config expects "
            f"POISE_LAYER_TOTAL={cfg.depth.layer_total}. Refusing to proceed — the "
            f"depth/budget math assumes exactly {cfg.depth.layer_total} layers."
        )
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Structural accessors — navigate common decoder-only layouts (Llama-family).
# --------------------------------------------------------------------------- #
def _base(model: Any) -> Any:
    """Return the inner base model that holds ``.layers`` / ``.norm`` / embeddings."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer  # GPT-style fallback
    if hasattr(model, "layers"):
        return model
    raise ModelLoadError("could not locate decoder layer stack on the model object")


def get_layers(model: Any) -> List[Any]:
    """The decoder blocks as an indexable list (the variable-depth budget indexes this)."""
    base = _base(model)
    layers = getattr(base, "layers", None)
    if layers is None:
        layers = getattr(base, "h", None)  # GPT-style
    if layers is None:
        raise ModelLoadError("decoder layer list not found")
    return list(layers)


def get_num_layers(model: Any) -> int:
    return len(get_layers(model))


def get_lm_head(model: Any) -> Any:
    for attr in ("lm_head", "embed_out", "output"):
        head = getattr(model, attr, None)
        if head is not None:
            return head
    raise ModelLoadError("LM head not found on the model object")


def get_norm(model: Any) -> Any:
    """Final norm applied before the LM head (RMSNorm for Llama)."""
    base = _base(model)
    for attr in ("norm", "final_layernorm", "ln_f"):
        norm = getattr(base, attr, None)
        if norm is not None:
            return norm
    raise ModelLoadError("final norm not found on the model object")


def get_embeddings(model: Any) -> Any:
    base = _base(model)
    for attr in ("embed_tokens", "wte"):
        emb = getattr(base, attr, None)
        if emb is not None:
            return emb
    raise ModelLoadError("token embedding not found on the model object")


def get_rotary_emb(model: Any) -> Any | None:
    """Model-level rotary embedding module if the architecture exposes one
    (newer Llama impls compute position embeddings once at the base model)."""
    base = _base(model)
    return getattr(base, "rotary_emb", None)


__all__ = [
    "load_model",
    "get_layers",
    "get_num_layers",
    "get_lm_head",
    "get_norm",
    "get_embeddings",
    "get_rotary_emb",
    "ModelLoadError",
]
