"""poise doctor — environment diagnosis + version-matched fixes."""

from __future__ import annotations

from dataclasses import dataclass

from poise import hardware
from poise.doctor import diagnose, format_report, recommend_torch_install


def test_parse_l4t():
    assert hardware.parse_l4t("# R36 (release), REVISION: 4.0, GCID: 1").startswith("R36.4")
    assert hardware.parse_l4t("# R35 (release), REVISION: 4.1").startswith("R35.4")
    assert hardware.parse_l4t("no version here") is None


def test_l4t_to_jetpack():
    assert hardware._l4t_jetpack("R36.4.0") == "6.1/6.2"
    assert hardware._l4t_jetpack("R35.4.1") == "5.1.2"
    assert hardware._l4t_jetpack("R36.9") == "6.x"
    assert hardware._l4t_jetpack(None) is None


def test_recommend_torch_install_jetpack6_cuda126():
    cmds = recommend_torch_install({"l4t": "R36.4.0", "cuda": "12.6.0"})
    assert any("jp6/cu126" in c for c in cmds)
    assert any("uninstall" in c for c in cmds)


def test_recommend_torch_install_cuda122():
    cmds = recommend_torch_install({"l4t": "R36.3.0", "cuda": "12.2.1"})
    assert any("jp6/cu122" in c for c in cmds)


def test_recommend_torch_install_jetpack5():
    cmds = recommend_torch_install({"l4t": "R35.4.1", "cuda": "11.4"})
    assert any("jp5" in c for c in cmds)


@dataclass
class _FakeHW:
    device: str = "cpu"
    accelerator: str = "NVIDIA Jetson AGX Orin"
    is_jetson: bool = True
    telemetry_backend: str = "jtop"
    torch_available: bool = True
    cuda_available: bool = False
    mps_available: bool = False


def _patch(monkeypatch, *, present, import_err, cuda, hw=None):
    monkeypatch.setattr(hardware, "probe", lambda *_a, **_k: hw or _FakeHW())
    monkeypatch.setattr(hardware, "torch_present", lambda: present)
    monkeypatch.setattr(hardware, "torch_import_error", lambda: import_err)
    monkeypatch.setattr(hardware, "cuda_available", lambda: cuda)
    monkeypatch.setattr(hardware, "jetpack_info",
                        lambda: {"l4t": "R36.4.0", "jetpack": "6.1/6.2", "cuda": "12.6.0"})


def test_diagnose_flags_jetson_cpu_only_torch(monkeypatch):
    """Torch imports fine on a Jetson but cuda unavailable -> CPU-only (wrong wheel)."""
    _patch(monkeypatch, present=True, import_err=None, cuda=False)
    info = diagnose()
    assert "jetson_torch_cpu_only" in info["issues"]
    report = format_report(info)
    assert "jp6/cu126" in report and "CPU" in report


def test_diagnose_flags_broken_torch_import_cusparselt(monkeypatch):
    """Torch installed but import fails on a missing CUDA .so -> prescribe the lib fix."""
    _patch(monkeypatch, present=True,
           import_err="libcusparseLt.so.0: cannot open shared object file", cuda=False)
    info = diagnose()
    assert "torch_import_broken" in info["issues"]
    report = format_report(info)
    assert "libcusparselt0" in report      # the exact apt fix is shown
    assert "cuSPARSELt" in report or "cusparselt" in report.lower()


def test_diagnose_clean_when_no_issues(monkeypatch):
    _patch(monkeypatch, present=True, import_err=None, cuda=True,
           hw=_FakeHW(device="cuda", is_jetson=True, cuda_available=True))
    info = diagnose()
    assert info["issues"] == []
    assert "no issues" in format_report(info)


def test_diagnose_returns_expected_keys():
    info = diagnose()  # real environment
    assert {"device", "is_jetson", "torch_installed", "cuda_available", "issues"} <= set(info)
