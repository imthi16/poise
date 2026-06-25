"""Config loader: validation invariants + env overrides (CLAUDE.md §6 / §10)."""

from __future__ import annotations

import pytest

from poise.config import load_config, ConfigError


def test_loads_defaults():
    cfg = load_config()
    assert cfg.depth.layer_total == 32
    assert cfg.depth.budget_set == (16, 20, 24, 28, 32)
    assert cfg.thermal.temp_setpoint_c < cfg.thermal.temp_max_c
    assert cfg.model.dtype in ("fp16", "bnb-4bit")
    assert cfg.telemetry.backend == "mock"  # off-device default


def test_depth_invariant_enforced(monkeypatch):
    monkeypatch.setenv("POISE_LAYER_MIN", "33")
    with pytest.raises(ConfigError):
        load_config()


def test_thermal_invariant_enforced(monkeypatch):
    monkeypatch.setenv("POISE_TEMP_SETPOINT_C", "90.0")
    monkeypatch.setenv("POISE_TEMP_MAX_C", "87.0")
    with pytest.raises(ConfigError):
        load_config()


def test_q4km_rejected_in_adaptive_path(monkeypatch):
    """Q4_K_M is GGUF/llama.cpp-only; must be rejected for the HF eager path."""
    monkeypatch.setenv("POISE_DTYPE", "q4_k_m")
    with pytest.raises(ConfigError) as ei:
        load_config()
    assert "llama.cpp" in str(ei.value)


def test_env_override_precedence(monkeypatch):
    monkeypatch.setenv("POISE_CONTROL_MODE", "ppo")
    monkeypatch.setenv("POISE_TEMP_SETPOINT_C", "75.5")
    cfg = load_config()
    assert cfg.control.mode == "ppo"
    assert cfg.thermal.temp_setpoint_c == 75.5


def test_budget_set_override(monkeypatch):
    monkeypatch.setenv("POISE_BUDGET_SET", "8,16,32")
    cfg = load_config()
    assert cfg.depth.budget_set == (8, 16, 32)


def test_snapshot_redacts_secret(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secretvalue")
    cfg = load_config()
    snap = cfg.snapshot()
    assert snap["model"]["hf_token"] == "***redacted***"


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv("POISE_CONTROL_MODE", "bogus")
    with pytest.raises(ConfigError):
        load_config()


def test_placeholder_sim_params_not_marked_calibrated():
    """Off-device sim params must be clearly placeholder, never a 'result'."""
    cfg = load_config()
    assert cfg.sim.params_source == "placeholder"
    assert cfg.sim.calibrated_valid is False
