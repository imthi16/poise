"""Universal hardware detection + model-agnostic depth adaptation.

POISE must bring up on ANY machine — Jetson, a generic NVIDIA GPU box, Apple
Silicon, or CPU-only — with ANY decoder-only model (not just the 32-layer 8B). This
module auto-detects the compute device and the best telemetry backend, and adapts the
layer-budget configuration to whatever model is actually loaded.

Everything here degrades gracefully and imports without torch / pynvml / jtop, so the
off-device test suite exercises the detection + adaptation logic with mocks.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import PoiseConfig

VALID_DEVICES = ("cuda", "mps", "cpu")
VALID_TELEMETRY = ("jtop", "nvml", "tegrastats", "mock")


# --------------------------------------------------------------------------- #
# Capability probes (all guarded; safe off-device)
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def is_jetson() -> bool:
    """True on NVIDIA Tegra/Jetson (Orin, Xavier, Nano, ...)."""
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode(errors="ignore").lower()
        return any(k in model for k in ("jetson", "tegra", "orin", "xavier"))
    except Exception:
        return False


def parse_l4t(line: str) -> str | None:
    """Parse an L4T version like 'R36.4' from an /etc/nv_tegra_release header line."""
    import re

    m = re.search(r"R(\d+)\D+REVISION:\s*([\d.]+)", line)
    return f"R{m.group(1)}.{m.group(2)}" if m else None


def _l4t_jetpack(l4t: str | None) -> str | None:
    if not l4t:
        return None
    table = {
        "R36.4": "6.1/6.2", "R36.3": "6.0", "R36.2": "6.0 DP",
        "R35.5": "5.1.3", "R35.4": "5.1.2", "R35.3": "5.1.1", "R35.2": "5.1",
    }
    for prefix, jp in table.items():
        if l4t.startswith(prefix):
            return jp
    if l4t.startswith("R36"):
        return "6.x"
    if l4t.startswith("R35"):
        return "5.x"
    return None


def cuda_toolkit_version() -> str | None:
    """Best-effort CUDA toolkit version from the JetPack install (no torch needed)."""
    import glob
    import json

    for p in ["/usr/local/cuda/version.json", *glob.glob("/usr/local/cuda-*/version.json")]:
        try:
            with open(p) as f:
                return json.load(f).get("cuda", {}).get("version")
        except Exception:
            continue
    return None


def jetpack_info() -> dict | None:
    """JetPack / L4T / CUDA info on a Jetson (None off-Jetson). Best-effort, guarded."""
    if not is_jetson():
        return None
    l4t = None
    try:
        with open("/etc/nv_tegra_release") as f:
            l4t = parse_l4t(f.readline())
    except Exception:
        pass
    return {"l4t": l4t, "jetpack": _l4t_jetpack(l4t), "cuda": cuda_toolkit_version()}


def jtop_available() -> bool:
    try:
        import jtop  # noqa: F401

        return True
    except Exception:
        return False


def nvml_available() -> bool:
    """True if an NVML-queryable NVIDIA GPU is present (generic desktop/server GPUs)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        ok = pynvml.nvmlDeviceGetCount() > 0
        pynvml.nvmlShutdown()
        return ok
    except Exception:
        return False


def tegrastats_available() -> bool:
    return shutil.which("tegrastats") is not None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def detect_device(prefer: str = "auto") -> str:
    """Resolve a concrete device. ``auto`` => cuda > mps > cpu."""
    if prefer in VALID_DEVICES:
        return prefer
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def detect_telemetry_backend(device: str | None = None) -> str:
    """Pick the richest available telemetry source for this machine."""
    if is_jetson():
        if jtop_available():
            return "jtop"
        if tegrastats_available():
            return "tegrastats"
        return "mock"
    if (device == "cuda" or cuda_available()) and nvml_available():
        return "nvml"
    return "mock"


def accelerator_name(device: str) -> str:
    if is_jetson():
        try:
            with open("/proc/device-tree/model", "rb") as f:
                return f.read().decode(errors="ignore").strip("\x00").strip()
        except Exception:
            return "NVIDIA Jetson"
    if device == "cuda":
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:
            return "NVIDIA GPU"
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    accelerator: str
    is_jetson: bool
    telemetry_backend: str
    torch_available: bool
    cuda_available: bool
    mps_available: bool

    def summary(self) -> str:
        return (
            f"device={self.device} ({self.accelerator}) | jetson={self.is_jetson} | "
            f"telemetry={self.telemetry_backend} | torch={self.torch_available}"
        )


def probe(prefer_device: str = "auto") -> HardwareProfile:
    device = detect_device(prefer_device)
    return HardwareProfile(
        device=device,
        accelerator=accelerator_name(device),
        is_jetson=is_jetson(),
        telemetry_backend=detect_telemetry_backend(device),
        torch_available=torch_available(),
        cuda_available=cuda_available(),
        mps_available=mps_available(),
    )


# --------------------------------------------------------------------------- #
# Model-agnostic depth adaptation
# --------------------------------------------------------------------------- #
def default_budget_set(layer_min: int, layer_max: int, n_steps: int = 5) -> tuple[int, ...]:
    """Evenly-spaced, de-duplicated discrete depths in ``[layer_min, layer_max]``."""
    import numpy as np

    layer_min = max(1, int(layer_min))
    layer_max = max(layer_min + 1, int(layer_max))
    vals = np.linspace(layer_min, layer_max, max(2, n_steps))
    out = sorted({int(round(v)) for v in vals})
    if out[0] != layer_min:
        out[0] = layer_min
    if out[-1] != layer_max:
        out[-1] = layer_max
    return tuple(sorted(set(out)))


def recommend_depth(num_layers: int, min_frac: float = 0.5, n_steps: int = 5) -> dict:
    """A sensible depth/budget configuration for a model with ``num_layers`` layers."""
    if num_layers < 2:
        raise ValueError(f"need >= 2 decoder layers to vary depth, got {num_layers}")
    layer_total = int(num_layers)
    layer_min = max(1, int(round(layer_total * min_frac)))
    layer_min = min(layer_min, layer_total - 1)  # keep room to vary
    return {
        "layer_total": layer_total,
        "layer_min": layer_min,
        "layer_max": layer_total,
        "static_depth": layer_total,
        "budget_set": default_budget_set(layer_min, layer_total, n_steps),
    }


def _placeholder_maps(budget_set, layer_total):
    """Regenerate clearly-labeled placeholder power/latency maps for new budgets."""
    power = {int(b): round(18.0 + (35.0 - 18.0) * (b / layer_total), 3) for b in budget_set}
    latency = {int(b): round(0.018 + (0.036 - 0.018) * (b / layer_total), 5) for b in budget_set}
    return power, latency


def adapt_depth_to_model(cfg: "PoiseConfig", num_layers: int) -> "PoiseConfig":
    """Return a config whose depth/budget config matches the loaded model's layer count.

    No-op when the model already has ``layer_total`` layers. When the count differs
    (any HF decoder-only model), the budget set and bounds are rescaled, and the
    *placeholder* simulator maps are regenerated for the new budgets (calibrated maps
    are left untouched — real params still come only from calibration).
    """
    if num_layers == cfg.depth.layer_total:
        return cfg
    rec = recommend_depth(num_layers)
    depth = replace(
        cfg.depth,
        layer_total=rec["layer_total"],
        layer_min=rec["layer_min"],
        layer_max=rec["layer_max"],
        budget_set=tuple(rec["budget_set"]),
        static_depth=rec["static_depth"],
    )
    sim = cfg.sim
    if cfg.sim.params_source == "placeholder":
        power, latency = _placeholder_maps(rec["budget_set"], rec["layer_total"])
        sim = replace(cfg.sim, depth_power_map=power, depth_latency_map=latency)
    return replace(cfg, depth=depth, sim=sim)


__all__ = [
    "HardwareProfile",
    "probe",
    "detect_device",
    "detect_telemetry_backend",
    "accelerator_name",
    "is_jetson",
    "torch_available",
    "cuda_available",
    "mps_available",
    "nvml_available",
    "jtop_available",
    "tegrastats_available",
    "jetpack_info",
    "parse_l4t",
    "cuda_toolkit_version",
    "default_budget_set",
    "recommend_depth",
    "adapt_depth_to_model",
    "VALID_DEVICES",
]
