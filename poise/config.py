"""POISE configuration loader.

Single entrypoint: ``load_config() -> PoiseConfig``.

Loads ``configs/*.yaml`` (deep-merged) and ``.env`` (via python-dotenv), then
applies ``POISE_*`` environment-variable overrides on top of the YAML. Returns
typed config dataclasses.

Rules (CLAUDE.md §6):
  * validate ranges (layer_min < layer_max <= layer_total; temp_setpoint < temp_max)
  * fail loud on an invalid / out-of-range value
  * the adaptive PyTorch path accepts dtype fp16 | bnb-4bit ONLY — ``q4_k_m`` is a
    GGUF/llama.cpp format and is rejected here with a pointer to the baseline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is a core dep, but degrade gracefully if absent.
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover

    def _load_dotenv(*_a, **_k):  # type: ignore
        return False


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

# YAML files merged (in order) into one resolved dict. Keys are largely disjoint.
_CONFIG_FILES = (
    "default.yaml",
    "model.yaml",
    "controller_pid.yaml",
    "ppo.yaml",
    "simulator.yaml",
    "eval.yaml",
)

_VALID_DTYPES_ADAPTIVE = ("fp16", "bnb-4bit")
_VALID_DEVICES = ("cuda", "cpu")
_VALID_MODES = ("static", "pid", "ppo")
_VALID_TELEMETRY = ("jtop", "tegrastats", "mock")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """Raised on any invalid / inconsistent configuration. Always fails loud."""


# --------------------------------------------------------------------------- #
# Typed config dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_path: str | None
    dtype: str  # fp16 | bnb-4bit  (NOT q4_k_m — see model_loader)
    device: str  # cuda | cpu
    trust_remote_code: bool
    gguf_path: str  # llama.cpp baseline ONLY
    hf_token: str | None  # secret, env-only
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class DepthConfig:
    layer_total: int
    layer_min: int
    layer_max: int
    budget_set: tuple[int, ...]
    static_depth: int


@dataclass(frozen=True)
class ThermalConfig:
    temp_setpoint_c: float
    temp_max_c: float
    ambient_c: float


@dataclass(frozen=True)
class PIDConfig:
    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_limit: float
    period_ms: int


@dataclass(frozen=True)
class ControlConfig:
    mode: str  # static | pid | ppo
    period_ms: int
    policy_path: str | None  # empty/None => PID-only
    pid: PIDConfig


@dataclass(frozen=True)
class TelemetryConfig:
    backend: str  # jtop | tegrastats | mock
    hz: int


@dataclass(frozen=True)
class SimConfig:
    calibrated_valid: bool
    r_th_c_per_w: float
    c_th_j_per_c: float
    ambient_c: float
    init_temp_c: float
    depth_power_map: dict[int, float]      # depth -> steady-state watts
    depth_latency_map: dict[int, float]    # depth -> per-token seconds
    params_source: str                     # 'placeholder' | 'calibrated'
    fit_rmse_c: float | None


@dataclass(frozen=True)
class RewardConfig:
    w_thr: float
    w_q: float
    w_therm: float
    w_energy: float
    w_throttle: float


@dataclass(frozen=True)
class EnvConfig:
    dt_s: float
    episode_seconds: float
    throughput_norm_tok_s: float
    energy_norm_j: float


@dataclass(frozen=True)
class DomainRandConfig:
    enabled: bool
    r_th_rel_range: tuple[float, float]
    c_th_rel_range: tuple[float, float]
    power_rel_range: tuple[float, float]
    ambient_c_range: tuple[float, float]
    sensor_noise_c: float


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    clip_range: float
    ent_coef: float
    seed: int
    policy: str
    net_arch: tuple[int, ...]
    reward: RewardConfig
    env: EnvConfig
    domain_random: DomainRandConfig


@dataclass(frozen=True)
class ServeConfig:
    host: str
    port: int
    prometheus_port: int


@dataclass(frozen=True)
class StorageConfig:
    db_path: str
    log_token_events: bool


@dataclass(frozen=True)
class RAGConfig:
    data_dir: str
    index_path: str


@dataclass(frozen=True)
class PoiseConfig:
    model: ModelConfig
    depth: DepthConfig
    thermal: ThermalConfig
    control: ControlConfig
    telemetry: TelemetryConfig
    sim: SimConfig
    ppo: PPOConfig
    serving: ServeConfig
    storage: StorageConfig
    rag: RAGConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable snapshot for storage (secrets redacted)."""
        d = {
            k: asdict(getattr(self, k))
            for k in (
                "model",
                "depth",
                "thermal",
                "control",
                "telemetry",
                "sim",
                "ppo",
                "serving",
                "storage",
                "rag",
            )
        }
        if d["model"].get("hf_token"):
            d["model"]["hf_token"] = "***redacted***"
        return d


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_dir(config_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in _CONFIG_FILES:
        p = config_dir / name
        if not p.exists():
            continue
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p} did not parse to a mapping")
        merged = _deep_merge(merged, data)
    return merged


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _int_keyed_map(d: dict | None) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in (d or {}).items():
        out[int(k)] = float(v)
    return out


# env var -> (dotted path in resolved dict, caster)
_ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "POISE_MODEL_ID": ("model.model_id", str),
    "POISE_MODEL_PATH": ("model.model_path", str),
    "POISE_GGUF_PATH": ("baseline.gguf_path", str),
    "POISE_DEVICE": ("device", str),
    "POISE_DTYPE": ("model.dtype", str),
    "POISE_LAYER_TOTAL": ("depth.layer_total", int),
    "POISE_LAYER_MIN": ("depth.layer_min", int),
    "POISE_LAYER_MAX": ("depth.layer_max", int),
    "POISE_BUDGET_SET": ("depth.budget_set", "int_list"),
    "POISE_STATIC_DEPTH": ("depth.static_depth", int),
    "POISE_TEMP_SETPOINT_C": ("thermal.temp_setpoint_c", float),
    "POISE_TEMP_MAX_C": ("thermal.temp_max_c", float),
    "POISE_AMBIENT_C": ("thermal.ambient_c", float),
    "POISE_PID_KP": ("pid.kp", float),
    "POISE_PID_KI": ("pid.ki", float),
    "POISE_PID_KD": ("pid.kd", float),
    "POISE_CONTROL_PERIOD_MS": ("control.period_ms", int),
    "POISE_CONTROL_MODE": ("control_mode", str),
    "POISE_POLICY_PATH": ("control.policy_path", str),
    "POISE_TELEMETRY_BACKEND": ("telemetry.backend", str),
    "POISE_TELEMETRY_HZ": ("telemetry.hz", int),
    "POISE_API_HOST": ("serving.host", str),
    "POISE_API_PORT": ("serving.port", int),
    "POISE_PROMETHEUS_PORT": ("serving.prometheus_port", int),
    "POISE_DB_PATH": ("storage.db_path", str),
    "POISE_RAG_DATA_DIR": ("rag.data_dir", str),
    "POISE_RAG_INDEX_PATH": ("rag.index_path", str),
}


def _set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _get_path(d: dict, dotted: str, default: Any = None) -> Any:
    node: Any = d
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _apply_env_overrides(resolved: dict) -> None:
    for env_key, (path, caster) in _ENV_OVERRIDES.items():
        if env_key not in os.environ:
            continue
        raw = os.environ[env_key]
        if raw == "":
            continue
        if caster == "int_list":
            value: Any = tuple(int(x) for x in raw.split(",") if x.strip() != "")
        elif caster is bool:
            value = _as_bool(raw)
        else:
            value = caster(raw)  # type: ignore[operator]
        _set_path(resolved, path, value)


# --------------------------------------------------------------------------- #
# Build + validate
# --------------------------------------------------------------------------- #
def _build(resolved: dict) -> PoiseConfig:
    depth_cfg = resolved.get("depth", {})
    budget_set = tuple(int(x) for x in depth_cfg.get("budget_set", []))

    sim = resolved.get("calibrated", {}) or {}
    placeholder = resolved.get("placeholder", {}) or {}
    calibrated_valid = bool(sim.get("valid", False))
    if calibrated_valid and sim.get("r_th_c_per_w") is not None:
        r_th = float(sim["r_th_c_per_w"])
        c_th = float(sim["c_th_j_per_c"])
        amb = float(sim.get("ambient_c", placeholder.get("ambient_c", 25.0)))
        init_t = float(sim.get("init_temp_c", placeholder.get("init_temp_c", 40.0)))
        params_source = "calibrated"
        fit_rmse = sim.get("fit_rmse_c")
    else:
        # Off-device development path: clearly-labeled NOMINAL placeholders, never
        # presented as a measured calibration result.
        r_th = float(placeholder.get("r_th_c_per_w", 1.5))
        c_th = float(placeholder.get("c_th_j_per_c", 40.0))
        amb = float(placeholder.get("ambient_c", 25.0))
        init_t = float(placeholder.get("init_temp_c", 40.0))
        params_source = "placeholder"
        fit_rmse = None

    sim_cfg = SimConfig(
        calibrated_valid=calibrated_valid,
        r_th_c_per_w=r_th,
        c_th_j_per_c=c_th,
        ambient_c=amb,
        init_temp_c=init_t,
        depth_power_map=_int_keyed_map(_get_path(resolved, "depth_power_map.watts")),
        depth_latency_map=_int_keyed_map(_get_path(resolved, "depth_latency_map.seconds")),
        params_source=params_source,
        fit_rmse_c=(float(fit_rmse) if fit_rmse is not None else None),
    )

    model_cfg = ModelConfig(
        model_id=str(_get_path(resolved, "model.model_id", "")),
        model_path=(_get_path(resolved, "model.model_path") or None),
        dtype=str(_get_path(resolved, "model.dtype", "fp16")),
        device=str(resolved.get("device", "cuda")),
        trust_remote_code=bool(_get_path(resolved, "model.trust_remote_code", False)),
        gguf_path=str(_get_path(resolved, "baseline.gguf_path", "")),
        hf_token=os.environ.get("HF_TOKEN") or None,
        max_new_tokens=int(_get_path(resolved, "generation.max_new_tokens", 256)),
        temperature=float(_get_path(resolved, "generation.temperature", 0.0)),
        top_p=float(_get_path(resolved, "generation.top_p", 1.0)),
        seed=int(_get_path(resolved, "generation.seed", 0)),
    )

    depth = DepthConfig(
        layer_total=int(depth_cfg.get("layer_total", 32)),
        layer_min=int(depth_cfg.get("layer_min", 16)),
        layer_max=int(depth_cfg.get("layer_max", 32)),
        budget_set=budget_set,
        static_depth=int(depth_cfg.get("static_depth", 32)),
    )

    thermal_cfg = resolved.get("thermal", {})
    thermal = ThermalConfig(
        temp_setpoint_c=float(thermal_cfg.get("temp_setpoint_c", 80.0)),
        temp_max_c=float(thermal_cfg.get("temp_max_c", 87.0)),
        ambient_c=float(thermal_cfg.get("ambient_c", 25.0)),
    )

    pid_raw = resolved.get("pid", {})
    pid = PIDConfig(
        kp=float(pid_raw.get("kp", 0.8)),
        ki=float(pid_raw.get("ki", 0.05)),
        kd=float(pid_raw.get("kd", 0.2)),
        output_min=float(pid_raw.get("output_min", depth.layer_min)),
        output_max=float(pid_raw.get("output_max", depth.layer_max)),
        integral_limit=float(pid_raw.get("integral_limit", 16.0)),
        period_ms=int(pid_raw.get("period_ms", _get_path(resolved, "control.period_ms", 250))),
    )

    policy_path = _get_path(resolved, "control.policy_path") or os.environ.get(
        "POISE_POLICY_PATH"
    )
    control = ControlConfig(
        mode=str(resolved.get("control_mode", "pid")),
        period_ms=int(_get_path(resolved, "control.period_ms", 250)),
        policy_path=(policy_path or None),
        pid=pid,
    )

    tel_raw = resolved.get("telemetry", {})
    telemetry = TelemetryConfig(
        backend=str(tel_raw.get("backend", "mock")),
        hz=int(tel_raw.get("hz", 4)),
    )

    reward_raw = resolved.get("reward", {})
    env_raw = resolved.get("env", {})
    dr_raw = resolved.get("domain_randomization", {})
    ppo_raw = resolved.get("ppo", {})
    ppo = PPOConfig(
        total_timesteps=int(ppo_raw.get("total_timesteps", 1_000_000)),
        n_envs=int(ppo_raw.get("n_envs", 8)),
        n_steps=int(ppo_raw.get("n_steps", 2048)),
        batch_size=int(ppo_raw.get("batch_size", 256)),
        gamma=float(ppo_raw.get("gamma", 0.99)),
        gae_lambda=float(ppo_raw.get("gae_lambda", 0.95)),
        learning_rate=float(ppo_raw.get("learning_rate", 3e-4)),
        clip_range=float(ppo_raw.get("clip_range", 0.2)),
        ent_coef=float(ppo_raw.get("ent_coef", 0.01)),
        seed=int(ppo_raw.get("seed", 0)),
        policy=str(ppo_raw.get("policy", "MlpPolicy")),
        net_arch=tuple(int(x) for x in ppo_raw.get("net_arch", [128, 128])),
        reward=RewardConfig(
            w_thr=float(reward_raw.get("w_thr", 1.0)),
            w_q=float(reward_raw.get("w_q", 1.0)),
            w_therm=float(reward_raw.get("w_therm", 0.5)),
            w_energy=float(reward_raw.get("w_energy", 0.3)),
            w_throttle=float(reward_raw.get("w_throttle", 10.0)),
        ),
        env=EnvConfig(
            dt_s=float(env_raw.get("dt_s", 0.25)),
            episode_seconds=float(env_raw.get("episode_seconds", 120.0)),
            throughput_norm_tok_s=float(env_raw.get("throughput_norm_tok_s", 30.0)),
            energy_norm_j=float(env_raw.get("energy_norm_j", 5.0)),
        ),
        domain_random=DomainRandConfig(
            enabled=bool(dr_raw.get("enabled", True)),
            r_th_rel_range=tuple(dr_raw.get("r_th_rel_range", [0.8, 1.2])),  # type: ignore
            c_th_rel_range=tuple(dr_raw.get("c_th_rel_range", [0.8, 1.2])),  # type: ignore
            power_rel_range=tuple(dr_raw.get("power_rel_range", [0.9, 1.1])),  # type: ignore
            ambient_c_range=tuple(dr_raw.get("ambient_c_range", [20.0, 35.0])),  # type: ignore
            sensor_noise_c=float(dr_raw.get("sensor_noise_c", 0.5)),
        ),
    )

    serve_raw = resolved.get("serving", {})
    serving = ServeConfig(
        host=str(serve_raw.get("host", "0.0.0.0")),
        port=int(serve_raw.get("port", 8000)),
        prometheus_port=int(serve_raw.get("prometheus_port", 9090)),
    )

    store_raw = resolved.get("storage", {})
    storage = StorageConfig(
        db_path=str(store_raw.get("db_path", "./data/results/poise.db")),
        log_token_events=bool(store_raw.get("log_token_events", True)),
    )

    rag_raw = resolved.get("rag", {})
    rag = RAGConfig(
        data_dir=str(rag_raw.get("data_dir", "./data/synthetic")),
        index_path=str(rag_raw.get("index_path", "./data/synthetic/faiss.index")),
    )

    cfg = PoiseConfig(
        model=model_cfg,
        depth=depth,
        thermal=thermal,
        control=control,
        telemetry=telemetry,
        sim=sim_cfg,
        ppo=ppo,
        serving=serving,
        storage=storage,
        rag=rag,
        raw=resolved,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: PoiseConfig) -> None:
    d = cfg.depth
    if not (d.layer_min < d.layer_max <= d.layer_total):
        raise ConfigError(
            f"depth invariant violated: require layer_min < layer_max <= layer_total, "
            f"got min={d.layer_min}, max={d.layer_max}, total={d.layer_total}"
        )
    if not d.budget_set:
        raise ConfigError("depth.budget_set must be non-empty")
    if any(b < 1 or b > d.layer_total for b in d.budget_set):
        raise ConfigError(
            f"depth.budget_set values must be in [1, {d.layer_total}], got {d.budget_set}"
        )
    if not (d.layer_min <= d.static_depth <= d.layer_total):
        raise ConfigError(
            f"depth.static_depth={d.static_depth} must be in "
            f"[{d.layer_min}, {d.layer_total}]"
        )

    t = cfg.thermal
    if not (t.temp_setpoint_c < t.temp_max_c):
        raise ConfigError(
            f"thermal invariant violated: require temp_setpoint < temp_max, "
            f"got setpoint={t.temp_setpoint_c}, max={t.temp_max_c}"
        )

    if cfg.model.device not in _VALID_DEVICES:
        raise ConfigError(f"device must be one of {_VALID_DEVICES}, got {cfg.model.device!r}")

    # Adaptive PyTorch path: fp16 | bnb-4bit only. q4_k_m is GGUF/llama.cpp-only.
    dtype = cfg.model.dtype.lower()
    if dtype in ("q4_k_m", "q4km", "gguf"):
        raise ConfigError(
            "POISE_DTYPE/model.dtype is the ADAPTIVE-path dtype and must be fp16 or "
            "bnb-4bit. 'q4_k_m' is a GGUF/llama.cpp format and is NOT loadable in the HF "
            "eager path — it is reserved for the llama.cpp baseline "
            "(poise/baselines/static_llamacpp.py)."
        )
    if dtype not in _VALID_DTYPES_ADAPTIVE:
        raise ConfigError(
            f"model.dtype must be one of {_VALID_DTYPES_ADAPTIVE}, got {cfg.model.dtype!r}"
        )

    if cfg.control.mode not in _VALID_MODES:
        raise ConfigError(f"control_mode must be one of {_VALID_MODES}, got {cfg.control.mode!r}")

    if cfg.telemetry.backend not in _VALID_TELEMETRY:
        raise ConfigError(
            f"telemetry.backend must be one of {_VALID_TELEMETRY}, got {cfg.telemetry.backend!r}"
        )
    if cfg.telemetry.hz <= 0:
        raise ConfigError(f"telemetry.hz must be > 0, got {cfg.telemetry.hz}")

    if cfg.control.period_ms <= 0:
        raise ConfigError(f"control.period_ms must be > 0, got {cfg.control.period_ms}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
    load_env: bool = True,
) -> PoiseConfig:
    """Load + validate the full POISE configuration.

    Order of precedence (low -> high): YAML defaults -> ``.env`` file ->
    process environment variables.
    """
    if load_env:
        # Load .env if present (does not override already-set process env vars).
        if env_file is not None:
            _load_dotenv(env_file, override=False)
        else:
            default_env = REPO_ROOT / ".env"
            if default_env.exists():
                _load_dotenv(default_env, override=False)

    cdir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not cdir.exists():
        raise ConfigError(f"config directory not found: {cdir}")

    resolved = _load_yaml_dir(cdir)
    _apply_env_overrides(resolved)
    return _build(resolved)


__all__ = [
    "load_config",
    "PoiseConfig",
    "ModelConfig",
    "DepthConfig",
    "ThermalConfig",
    "PIDConfig",
    "ControlConfig",
    "TelemetryConfig",
    "SimConfig",
    "PPOConfig",
    "RewardConfig",
    "EnvConfig",
    "DomainRandConfig",
    "ServeConfig",
    "StorageConfig",
    "RAGConfig",
    "ConfigError",
]
