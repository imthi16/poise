"""``poise doctor`` — diagnose the environment and prescribe the fix.

The most common on-device gotcha: installing the **generic PyPI torch** on a Jetson.
That wheel cannot use the integrated GPU (driver/CUDA mismatch), so ``cuda`` is
unavailable and POISE silently runs on CPU. This module detects that (and other
setup issues) and prints the exact, version-matched fix — so nobody gets stuck.

    python -m poise.doctor        # or: bash scripts/doctor.sh
"""

from __future__ import annotations

from . import hardware


def diagnose() -> dict:
    hw = hardware.probe("auto")
    info = {
        "device": hw.device,
        "accelerator": hw.accelerator,
        "is_jetson": hw.is_jetson,
        "telemetry_backend": hw.telemetry_backend,
        "torch_installed": hardware.torch_present(),       # package present (may not import)
        "torch_import_error": hardware.torch_import_error(),
        "cuda_available": hardware.cuda_available(),
        "jetpack": hardware.jetpack_info(),
    }
    info["issues"] = _issues(info)
    return info


_MISSING_LIB_FIX = {
    "libcusparseLt": (
        "the GPU torch links cuSPARSELt — install it:\n"
        "        sudo apt-get update && sudo apt-get install -y libcusparselt0 libcusparselt-dev"
    ),
    "libcudnn": (
        "missing cuDNN — install JetPack's cuDNN:\n"
        "        sudo apt-get install -y libcudnn9-cuda-12 || sudo apt-get install -y nvidia-jetpack"
    ),
    "libnvinfer": (
        "missing TensorRT libs:  sudo apt-get install -y nvidia-jetpack"
    ),
}


def _issues(info: dict) -> list[str]:
    issues: list[str] = []
    err = info.get("torch_import_error")
    if not info["torch_installed"]:
        issues.append("no_torch")
    elif err:
        # torch is installed but cannot import — usually a missing CUDA .so on Jetson.
        issues.append("torch_import_broken")
    elif info["is_jetson"] and not info["cuda_available"]:
        issues.append("jetson_torch_cpu_only")
    if info["is_jetson"] and info["telemetry_backend"] == "mock":
        issues.append("jetson_no_telemetry")
    return issues


def _missing_lib_hint(err: str) -> str | None:
    for lib, fix in _MISSING_LIB_FIX.items():
        if lib.lower() in (err or "").lower():
            return fix
    return None


def recommend_torch_install(jetpack: dict | None) -> list[str]:
    """The version-matched torch install commands for this Jetson.

    Gives the NVIDIA CDN (most reliable) as primary and the jetson-ai-lab index as an
    alternative, and warns about the two real-world gotchas: the index host may not
    resolve, and transformers/accelerate will drag the generic (GPU-incompatible)
    PyPI torch back unless torch is installed first / with --no-deps.
    """
    cuda = (jetpack or {}).get("cuda") or ""
    l4t = (jetpack or {}).get("l4t") or ""
    if l4t.startswith("R35"):
        cu, jp = "cu114", "jp5"
    else:  # JetPack 6.x
        cu = "cu122" if cuda.startswith("12.2") else "cu126"
        jp = "jp6"
    docs = "https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
    return [
        "pip uninstall -y torch torchvision torchaudio   # remove the GPU-incompatible build FIRST",
        "# Get YOUR JetPack's torch wheel URL from NVIDIA's doc (the CDN dir listing is",
        "# disabled, but the .whl files download fine), then install it directly:",
        f"#   {docs}",
        f'pip install --no-cache-dir "<the torch-...{cu}...aarch64.whl URL from the doc>"',
        f"# community-index alternative IF it resolves: --index-url https://pypi.jetson-ai-lab.io/{jp}/{cu}",
        "pip install --no-deps accelerate   # so it cannot pull the generic torch back",
        "# bulletproof fallback: run in NVIDIA's container  nvcr.io/nvidia/l4t-pytorch (matches your L4T)",
        'python3 -c "import torch; print(torch.cuda.is_available())"   # must be True',
    ]


_FIX = {
    "jetson_torch_cpu_only": (
        "torch is installed but CANNOT use the Jetson GPU (generic PyPI wheel /\n"
        "    driver mismatch) — POISE will run on CPU. Install the JetPack-matched wheel:"
    ),
    "no_torch": (
        "torch is not installed — the adaptive engine needs it.\n"
        "    On a Jetson use the JetPack wheel (below); elsewhere `pip install torch transformers`."
    ),
    "jetson_no_telemetry": (
        "on a Jetson but no telemetry backend resolved — install jetson-stats for jtop:\n"
        "    sudo pip install jetson-stats   (then reboot or `sudo systemctl restart jtop`)"
    ),
    "torch_import_broken": (
        "torch is installed but FAILS to import — almost always a missing CUDA system\n"
        "    library that the Jetson wheel links against. Fix the specific lib below."
    ),
}


def format_report(info: dict) -> str:
    lines = ["POISE doctor", "=" * 48]
    lines.append(f"  device            : {info['device']} ({info['accelerator']})")
    lines.append(f"  jetson            : {info['is_jetson']}")
    if info["jetpack"]:
        jp = info["jetpack"]
        lines.append(f"  jetpack / L4T     : {jp.get('jetpack')} / {jp.get('l4t')} "
                     f"(CUDA {jp.get('cuda')})")
    lines.append(f"  telemetry backend : {info['telemetry_backend']}")
    lines.append(f"  torch installed   : {info['torch_installed']}")
    lines.append(f"  cuda available    : {info['cuda_available']}")
    if info.get("torch_import_error"):
        lines.append(f"  torch import error: {info['torch_import_error'][:80]}")

    if not info["issues"]:
        lines.append("\n  ✓ no issues detected — ready to bring up.")
        return "\n".join(lines)

    lines.append("\n  Issues:")
    for issue in info["issues"]:
        lines.append(f"  ✗ {_FIX.get(issue, issue)}")
        if issue == "torch_import_broken":
            hint = _missing_lib_hint(info.get("torch_import_error") or "")
            lines.append(f"        {hint}" if hint else
                         "        (paste the import error and re-run; install the missing lib via apt)")
        if issue in ("jetson_torch_cpu_only", "no_torch"):
            for cmd in recommend_torch_install(info["jetpack"]):
                lines.append(f"        {cmd}")
            lines.append("    Authoritative wheels: "
                         "https://docs.nvidia.com/deeplearning/frameworks/"
                         "install-pytorch-jetson-platform/")
    lines.append("\n  Re-run `python -m poise.doctor` until there are no issues, "
                 "then `bash scripts/bringup.sh`.")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    info = diagnose()
    print(format_report(info))
    return 1 if info["issues"] else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
