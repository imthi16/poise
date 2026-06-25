"""POISE — Power-Optimized Inference via State-aware Execution.

Hardware-state-conditioned variable-depth LLM inference. The contribution is that
per-token execution depth is conditioned on live device physics (temperature,
power, throttle), not on input difficulty (cf. CALM / LayerSkip / AdaInfer / DASH).

See CLAUDE.md for the authoritative spec.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import load_config, PoiseConfig  # noqa: E402

__all__ = ["load_config", "PoiseConfig", "__version__"]
