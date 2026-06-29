"""Universal bring-up orchestrator — runs/skip logic off-device (CLAUDE.md §5)."""

from __future__ import annotations

from poise.bringup import StepResult, run_checklist
from poise.config import load_config


def _by_name(results):
    return {r.name: r for r in results}


def test_profile_and_deps_always_ok():
    cfg = load_config()
    res = _by_name(run_checklist(cfg, ["profile", "deps"]))
    assert res["profile"].status == "ok"
    assert res["deps"].status == "ok"
    # device is reported in the profile detail
    assert any(d in res["profile"].detail for d in ("cuda", "mps", "cpu"))


def test_model_step_skips_without_torch_or_model():
    """Off-device (no torch / no gated model) the model step skips cleanly, not crash."""
    cfg = load_config()
    res = _by_name(run_checklist(cfg, ["model"]))
    # either skipped (no torch / model) or ok (if a model happens to be available)
    assert res["model"].status in ("skipped", "ok")
    if res["model"].status == "skipped":
        assert "tiny-gpt2" in res["model"].detail or "MODEL_ID" in res["model"].detail


def test_gate_skips_without_model():
    cfg = load_config()
    res = _by_name(run_checklist(cfg, ["gate"]))
    assert res["gate"].status == "skipped"
    assert "model" in res["gate"].detail.lower()


def test_calibrate_skips_on_mock_telemetry():
    """Calibration must refuse mock sensors — synthetic sweeps are not real params."""
    cfg = load_config()
    res = _by_name(run_checklist(cfg, ["profile", "calibrate"]))
    assert res["calibrate"].status == "skipped"
    assert "mock" in res["calibrate"].detail.lower()


def test_eval_runs_against_mock_engine_offdevice():
    """The eval step works anywhere: off-device it runs the mock engine and labels it."""
    cfg = load_config()
    res = _by_name(run_checklist(cfg, ["eval"], eval_tokens=8))
    assert res["eval"].status == "ok"
    assert "MOCK" in res["eval"].detail  # clearly flagged as not an eval result
    assert "tok/s" in res["eval"].detail


def test_full_default_checklist_no_crash():
    cfg = load_config()
    results = run_checklist(cfg, None)  # default steps
    names = [r.name for r in results]
    assert names == ["profile", "deps", "model", "gate", "eval"]
    assert all(r.status in ("ok", "skipped", "failed") for r in results)
    # nothing should hard-fail off-device
    assert not any(r.status == "failed" for r in results), [r.line() for r in results]


def test_step_result_line_format():
    assert "✓" in StepResult("x", "ok").line()
    assert "•" in StepResult("x", "skipped").line()
    assert "✗" in StepResult("x", "failed").line()
