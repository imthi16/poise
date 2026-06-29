"""Turnkey, universal on-device bring-up for POISE.

Runs the build/validation pipeline end to end and adapts to whatever machine it is
on — Jetson, a generic NVIDIA GPU box, Apple Silicon, or CPU-only — with ANY HF
decoder-only model. Each step runs if the machine supports it and is clearly reported
as ok / skipped / failed otherwise, so the same command is safe everywhere.

    python -m poise.bringup                      # profile + deps + model + gate + eval
    python -m poise.bringup --model sshleifer/tiny-gpt2   # validate the WHOLE pipeline
                                                          # on a CPU laptop, no GPU/gated model
    python -m poise.bringup --steps profile,deps          # just inspect the machine

The step-3 GATE is the go/no-go: it measures the depth->quality cost on the real model
and prints the usable LAYER_MIN. Nothing downstream is trusted before it runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import hardware

DEFAULT_STEPS = ["profile", "deps", "model", "gate", "eval"]
ALL_STEPS = ["profile", "deps", "model", "gate", "calibrate", "train", "eval"]


@dataclass
class StepResult:
    name: str
    status: str  # ok | skipped | failed
    detail: str = ""
    data: dict = field(default_factory=dict)

    def line(self) -> str:
        icon = {"ok": "✓", "skipped": "•", "failed": "✗"}.get(self.status, "?")
        return f"  [{icon}] {self.name:<10} {self.status:<8} {self.detail}"


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def step_profile(ctx: dict) -> StepResult:
    hw = hardware.probe(ctx["cfg"].model.device)
    ctx["hw"] = hw
    return StepResult("profile", "ok", hw.summary(), {"profile": hw.__dict__})


def step_deps(ctx: dict) -> StepResult:
    mods = {
        "torch": "engine", "transformers": "engine", "accelerate": "engine",
        "bitsandbytes": "quant (bnb-4bit)", "stable_baselines3": "rl (PPO)",
        "faiss": "rag", "llama_cpp": "baselines (Q4_K_M)", "pynvml": "telemetry (NVML)",
        "jtop": "telemetry (Jetson)",
    }
    present, missing = [], []
    for m, _ in mods.items():
        try:
            __import__(m)
            present.append(m)
        except Exception:
            missing.append(m)
    ctx["deps_present"] = set(present)
    detail = f"present: {', '.join(present) or 'none'} | missing: {', '.join(missing) or 'none'}"
    return StepResult("deps", "ok", detail, {"present": present, "missing": missing})


def step_model(ctx: dict) -> StepResult:
    cfg = ctx["cfg"]
    try:
        from .engine.model_loader import load_model, get_num_layers
        from .hardware import adapt_depth_to_model
    except Exception as e:  # pragma: no cover
        return StepResult("model", "failed", f"import error: {e}")
    try:
        model, tokenizer = load_model(cfg)
    except Exception as e:
        return StepResult("model", "skipped",
                          f"{type(e).__name__}: install extras / set POISE_MODEL_ID "
                          f"(try --model sshleifer/tiny-gpt2 on CPU). [{str(e)[:80]}]")
    n = get_num_layers(model)
    cfg = adapt_depth_to_model(cfg, n)
    ctx["cfg"] = cfg
    ctx["model"] = model
    ctx["tokenizer"] = tokenizer
    return StepResult("model", "ok",
                      f"{cfg.model.model_id} on {cfg.model.device}: {n} layers, "
                      f"budgets {list(cfg.depth.budget_set)}", {"num_layers": n})


def step_gate(ctx: dict) -> StepResult:
    if "model" not in ctx:
        return StepResult("gate", "skipped", "no model loaded (run the model step first)")
    cfg = ctx["cfg"]
    try:
        from .calibration.quality_profile import (
            QualityProfiler, format_table, recommend_layer_min, persist_profile,
        )
        from .storage.db import init_db
    except Exception as e:  # pragma: no cover
        return StepResult("gate", "failed", f"import error: {e}")

    prompts = ctx.get("prompts") or _default_prompts(cfg)
    max_kl = ctx.get("max_kl", 0.10)
    try:
        profiler = QualityProfiler(cfg, ctx["model"], ctx["tokenizer"])
        results = profiler.profile(prompts, dataset_tag="bringup")
    except Exception as e:
        return StepResult("gate", "failed", f"{type(e).__name__}: {str(e)[:100]}")

    print(format_table(results))
    rec = recommend_layer_min(results, max_kl=max_kl)
    try:
        conn = init_db(cfg.storage.db_path)
        persist_profile(conn, results, dataset_tag="bringup")
        conn.close()
    except Exception:
        pass
    if rec is None:
        return StepResult("gate", "ok",
                          f"⚠ no depth meets mean_KL<= {max_kl} (quality-collapse signal — "
                          f"see CLAUDE.md §9; do NOT narrow silently)", {"layer_min": None})
    ctx["recommended_layer_min"] = rec
    return StepResult("gate", "ok", f"usable LAYER_MIN (mean_KL<= {max_kl}) = {rec}",
                      {"layer_min": rec})


def step_calibrate(ctx: dict) -> StepResult:
    cfg = ctx["cfg"]
    hw = ctx.get("hw")
    backend = hw.telemetry_backend if hw else cfg.telemetry.backend
    if backend == "mock":
        return StepResult("calibrate", "skipped",
                          "telemetry is mock (no real sensors) — calibration needs a board "
                          "with jtop/NVML; synthetic sweeps must not be used as real params")
    try:
        from .calibration.sweep import SweepSpec, run_sweep
    except Exception as e:  # pragma: no cover
        return StepResult("calibrate", "failed", f"import error: {e}")
    try:
        spec = SweepSpec(depths=list(cfg.depth.budget_set), duration_s=ctx.get("calib_s", 60.0),
                         hz=cfg.telemetry.hz)
        paths = run_sweep(cfg, spec, out_dir="data/calibration")
        return StepResult("calibrate", "ok",
                          f"wrote {len(paths)} sweep CSVs; fit RC params with calibration.fit "
                          f"-> configs/simulator.yaml", {"paths": [str(p) for p in paths]})
    except Exception as e:
        return StepResult("calibrate", "failed", f"{type(e).__name__}: {str(e)[:100]}")


def step_train(ctx: dict) -> StepResult:
    if "stable_baselines3" not in ctx.get("deps_present", set()):
        return StepResult("train", "skipped",
                          "PPO trains OFF-DEVICE (Kaggle): pip install -e .[rl] then run "
                          "scripts/train_ppo_kaggle.ipynb (on-board RL is far too slow)")
    if not ctx.get("do_train"):
        return StepResult("train", "skipped",
                          "pass --train to run a short local PPO smoke-train (normally on Kaggle)")
    cfg = ctx["cfg"]
    try:
        from .rl.train_ppo import train
        from .storage.db import init_db

        conn = init_db(cfg.storage.db_path)
        res = train(cfg, "ppo_policy.zip", conn=conn, total_timesteps=ctx.get("train_steps", 20000))
        conn.close()
        return StepResult("train", "ok",
                          f"reward={res.final_mean_reward:.2f} collapsed_to_min="
                          f"{res.collapsed_to_min} dist={res.converged_depth_hist}",
                          {"collapsed": res.collapsed_to_min})
    except Exception as e:
        return StepResult("train", "failed", f"{type(e).__name__}: {str(e)[:100]}")


def step_eval(ctx: dict) -> StepResult:
    cfg = ctx["cfg"]
    try:
        from .eval.benchmark import StressBenchmark, StressSpec
        from .control.budget_allocator import make_allocator
    except Exception as e:  # pragma: no cover
        return StepResult("eval", "failed", f"import error: {e}")

    real = "model" in ctx
    if real:
        from .engine.adaptive_runner import AdaptiveRunner
        from .telemetry import make_reader

        reader = make_reader(cfg)
        runner = AdaptiveRunner(cfg, ctx["model"], ctx["tokenizer"], telemetry_reader=reader)
        label = cfg.control.mode
    else:
        from .serving.api import MockAdaptiveRunner
        from .telemetry import make_reader

        reader = make_reader(cfg)
        runner = MockAdaptiveRunner(cfg, telemetry_reader=reader)
        label = "mock"

    try:
        bench = StressBenchmark(cfg, runner, make_allocator(cfg))
        _, metrics = bench.run(ctx.get("eval_prompt", "Explain thermal throttling."),
                               StressSpec(max_new_tokens=ctx.get("eval_tokens", 32)))
        note = "" if real else " (MOCK — not an eval result; real numbers need the model)"
        return StepResult("eval", "ok",
                          f"{label}: {metrics.tok_per_s:.1f} tok/s, "
                          f"{metrics.energy_per_token_j:.3f} J/tok, mean depth "
                          f"{metrics.mean_budget:.1f}{note}", metrics.as_dict())
    except Exception as e:
        return StepResult("eval", "failed", f"{type(e).__name__}: {str(e)[:100]}")


_STEP_FNS: dict[str, Callable[[dict], StepResult]] = {
    "profile": step_profile, "deps": step_deps, "model": step_model, "gate": step_gate,
    "calibrate": step_calibrate, "train": step_train, "eval": step_eval,
}


def _default_prompts(cfg) -> list[str]:
    from pathlib import Path

    from .calibration.quality_profile import load_prompts

    p = Path(cfg.rag.data_dir) / "eval_prompts.jsonl"
    if p.exists():
        return load_prompts(p)
    return [
        "Explain thermal throttling in one paragraph.",
        "Why does executing fewer transformer layers save energy?",
        "Describe the trade-off between inference depth and output quality.",
    ]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_checklist(cfg, steps: Optional[list[str]] = None, **ctx_kw) -> list[StepResult]:
    """Run the selected steps in order, threading context (cfg/model/...) between them."""
    steps = steps or DEFAULT_STEPS
    ctx: dict[str, Any] = {"cfg": cfg, **ctx_kw}
    results: list[StepResult] = []
    for name in steps:
        fn = _STEP_FNS.get(name)
        if fn is None:
            results.append(StepResult(name, "failed", "unknown step"))
            continue
        try:
            results.append(fn(ctx))
        except Exception as e:  # pragma: no cover - a step should report, not crash
            results.append(StepResult(name, "failed", f"{type(e).__name__}: {str(e)[:100]}"))
    return results


def main() -> None:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="POISE universal on-device bring-up")
    ap.add_argument("--model", help="override POISE_MODEL_ID (e.g. sshleifer/tiny-gpt2 for CPU)")
    ap.add_argument("--device", help="override POISE_DEVICE (auto|cuda|mps|cpu)")
    ap.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                    help=f"comma list of {ALL_STEPS}")
    ap.add_argument("--max-kl", type=float, default=0.10, help="gate threshold for LAYER_MIN")
    ap.add_argument("--train", action="store_true", help="run a short local PPO smoke-train")
    args = ap.parse_args()

    if args.model:
        os.environ["POISE_MODEL_ID"] = args.model
        os.environ.setdefault("POISE_MODEL_PATH", "")
    if args.device:
        os.environ["POISE_DEVICE"] = args.device

    from .config import load_config

    cfg = load_config()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    print("=" * 64)
    print("POISE bring-up")
    print("=" * 64)

    # Preflight: catch the #1 on-device gotcha (CPU-only torch on a Jetson) and print
    # the version-matched fix BEFORE running the pipeline.
    from .doctor import diagnose, format_report

    dx = diagnose()
    if dx["issues"]:
        print(format_report(dx))
        print("=" * 64)

    results = run_checklist(cfg, steps, max_kl=args.max_kl, do_train=args.train)
    print("\nSummary:")
    for r in results:
        print(r.line())
    failed = [r for r in results if r.status == "failed"]
    print(f"\n{sum(r.status=='ok' for r in results)} ok, "
          f"{sum(r.status=='skipped' for r in results)} skipped, {len(failed)} failed.")
    if "gate" in steps and not any(r.name == "gate" and r.status == "ok" for r in results):
        print("NOTE: the step-3 GATE did not complete — the quality premise is unproven "
              "until it runs on the real model (CLAUDE.md §9).")


if __name__ == "__main__":  # pragma: no cover
    main()
