"""Universal hardware detection + model-agnostic depth adaptation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from poise import hardware
from poise.config import load_config


# --- device detection ------------------------------------------------------ #
def test_detect_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_available", lambda: True)
    monkeypatch.setattr(hardware, "mps_available", lambda: True)
    assert hardware.detect_device("auto") == "cuda"

    monkeypatch.setattr(hardware, "cuda_available", lambda: False)
    assert hardware.detect_device("auto") == "mps"

    monkeypatch.setattr(hardware, "mps_available", lambda: False)
    assert hardware.detect_device("auto") == "cpu"


def test_detect_device_respects_explicit():
    assert hardware.detect_device("cpu") == "cpu"
    assert hardware.detect_device("cuda") == "cuda"


def test_detect_telemetry_backend_off_device_is_mock(monkeypatch):
    monkeypatch.setattr(hardware, "is_jetson", lambda: False)
    monkeypatch.setattr(hardware, "nvml_available", lambda: False)
    monkeypatch.setattr(hardware, "cuda_available", lambda: False)
    assert hardware.detect_telemetry_backend("cpu") == "mock"


def test_detect_telemetry_backend_nvml_on_nvidia(monkeypatch):
    monkeypatch.setattr(hardware, "is_jetson", lambda: False)
    monkeypatch.setattr(hardware, "nvml_available", lambda: True)
    assert hardware.detect_telemetry_backend("cuda") == "nvml"


def test_detect_telemetry_backend_jtop_on_jetson(monkeypatch):
    monkeypatch.setattr(hardware, "is_jetson", lambda: True)
    monkeypatch.setattr(hardware, "jtop_available", lambda: True)
    assert hardware.detect_telemetry_backend("cuda") == "jtop"


def test_probe_returns_valid_profile():
    hw = hardware.probe("auto")
    assert hw.device in hardware.VALID_DEVICES
    assert hw.telemetry_backend in ("jtop", "nvml", "tegrastats", "mock")
    assert isinstance(hw.summary(), str)


# --- depth adaptation ------------------------------------------------------ #
def test_default_budget_set_spans_range():
    bs = hardware.default_budget_set(16, 32, 5)
    assert bs == (16, 20, 24, 28, 32)
    assert hardware.default_budget_set(6, 12, 5)[0] == 6
    assert hardware.default_budget_set(6, 12, 5)[-1] == 12


def test_recommend_depth_scales():
    rec = hardware.recommend_depth(22)
    assert rec["layer_total"] == 22 and rec["layer_max"] == 22
    assert 1 <= rec["layer_min"] < 22
    assert rec["budget_set"][0] == rec["layer_min"]
    assert rec["budget_set"][-1] == 22


def test_recommend_depth_rejects_tiny():
    with pytest.raises(ValueError):
        hardware.recommend_depth(1)


def test_adapt_depth_noop_when_match():
    cfg = load_config()
    same = hardware.adapt_depth_to_model(cfg, cfg.depth.layer_total)
    assert same is cfg  # unchanged


def test_adapt_depth_rescales_to_model():
    """Any model size: budgets + bounds rescale and stay valid (the universal piece)."""
    cfg = load_config()  # default layer_total=32
    adapted = hardware.adapt_depth_to_model(cfg, 12)  # e.g. GPT-2 small
    d = adapted.depth
    assert d.layer_total == 12 and d.layer_max == 12
    assert d.layer_min < d.layer_max <= d.layer_total
    assert all(d.layer_min <= b <= d.layer_total for b in d.budget_set)
    assert d.static_depth == 12
    # placeholder sim maps regenerated for the new budgets
    assert set(adapted.sim.depth_power_map.keys()) == set(d.budget_set)
    assert set(adapted.sim.depth_latency_map.keys()) == set(d.budget_set)


def test_adapt_depth_keeps_invariants_for_many_sizes():
    cfg = load_config()
    for n in (6, 12, 24, 28, 40, 80):
        a = hardware.adapt_depth_to_model(cfg, n)
        d = a.depth
        assert d.layer_min < d.layer_max <= d.layer_total == n
        assert d.budget_set[0] >= d.layer_min and d.budget_set[-1] == n


def test_config_resolves_auto_device(monkeypatch):
    """`device: auto` must resolve to a concrete target on any machine."""
    monkeypatch.setenv("POISE_DEVICE", "auto")
    cfg = load_config()
    assert cfg.model.device in ("cuda", "mps", "cpu")
