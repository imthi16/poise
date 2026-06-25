"""Real telemetry readers for the Jetson (CLAUDE.md §6, from `poise_telemetry.py`).

Two backends behind one interface: ``jtop`` (jetson-stats) and ``tegrastats``
(parse the system binary's stdout). Both sample on a **background thread** and
expose the latest sample without blocking the inference thread.

Off-device these backends are unavailable; use ``telemetry.mock.MockTelemetryReader``
(selected automatically by ``make_reader`` when backend == "mock").
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Iterator, Optional

from .schema import TelemetrySample


class BaseTelemetryReader:
    """Background-sampling base: a worker thread refreshes ``_latest`` so callers
    never block. Subclasses implement ``_sample_once()``.
    """

    def __init__(self, hz: float = 4.0):
        self.hz = max(0.5, float(hz))
        self._latest: Optional[TelemetrySample] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- subclass hook ---------------------------------------------------- #
    def _sample_once(self) -> TelemetrySample:  # pragma: no cover - hardware path
        raise NotImplementedError

    # --- public API ------------------------------------------------------- #
    def read(self) -> TelemetrySample:
        """Latest sample.

        When the background thread is running, return its most recent sample
        without blocking the caller (the inference thread must never stall on
        telemetry). With no live thread, sample synchronously on each call so
        simple/test usage still advances.
        """
        if self._thread is not None and self._thread.is_alive():
            with self._lock:
                if self._latest is not None:
                    return self._latest
        sample = self._sample_once()
        with self._lock:
            self._latest = sample
        return sample

    def latest(self) -> Optional[TelemetrySample]:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def stream(self, hz: Optional[float] = None) -> Iterator[TelemetrySample]:
        period = 1.0 / (hz or self.hz)
        while not self._stop.is_set():
            yield self.read()
            time.sleep(period)

    def _run(self) -> None:  # pragma: no cover - hardware path
        period = 1.0 / self.hz
        while not self._stop.is_set():
            try:
                sample = self._sample_once()
                with self._lock:
                    self._latest = sample
            except Exception:
                # Never let a telemetry hiccup kill the inference process.
                pass
            time.sleep(period)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class JtopReader(BaseTelemetryReader):
    """jetson-stats backend. Reads junction temp, total power, GPU clock/util and
    a real throttle flag from the ``jtop`` service."""

    def __init__(self, hz: float = 4.0):
        super().__init__(hz)
        try:
            from jtop import jtop  # type: ignore
        except Exception as e:  # pragma: no cover - off-device
            raise RuntimeError(
                "jetson-stats not available; install with `pip install jetson-stats` on the "
                "Jetson, or use POISE_TELEMETRY_BACKEND=mock off-device."
            ) from e
        self._jtop_cls = jtop
        self._jtop = jtop()
        self._jtop.start()

    def _sample_once(self) -> TelemetrySample:  # pragma: no cover - hardware path
        j = self._jtop
        stats = j.stats
        temps = j.temperature or {}
        # Prefer junction/SOC temps; fall back to the hottest reported zone.
        temp_c = _pick_temp(temps)
        power = j.power or {}
        power_w = float(power.get("tot", {}).get("power", 0.0)) / 1000.0  # mW -> W
        gpu = j.gpu or {}
        first_gpu = next(iter(gpu.values()), {}) if isinstance(gpu, dict) else {}
        gpu_clock = float(first_gpu.get("freq", {}).get("cur", 0.0)) / 1000.0  # kHz->MHz
        gpu_util = float(stats.get("GPU", 0.0))
        # Real throttle signal: jtop exposes thermal/power throttling flags.
        throttled = bool(_jtop_throttled(j))
        return TelemetrySample(
            ts=time.time(),
            temp_c=temp_c,
            power_w=power_w,
            gpu_clock_mhz=gpu_clock,
            gpu_util=gpu_util,
            throttled=throttled,
        )

    def stop(self) -> None:  # pragma: no cover - hardware path
        super().stop()
        try:
            self._jtop.close()
        except Exception:
            pass


class TegrastatsReader(BaseTelemetryReader):
    """Parse ``tegrastats`` stdout. Slower-format but available without jtop.

    Example line fragment:
      ``... GR3D_FREQ 99%@1300 ... tj@72.5C VDD_GPU_SOC 4500mW/4500mW ...``
    """

    _RE_TEMP = re.compile(r"tj@([0-9.]+)C")
    _RE_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%(?:@(\d+))?")
    _RE_POWER = re.compile(r"(VDD_GPU_SOC|POM_5V_GPU|VDD_IN)\s+(\d+)mW")

    def __init__(self, hz: float = 4.0, interval_ms: Optional[int] = None):
        super().__init__(hz)
        self._interval_ms = interval_ms or int(1000.0 / self.hz)
        self._proc: Optional[subprocess.Popen] = None

    def _ensure_proc(self):  # pragma: no cover - hardware path
        if self._proc is None:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        return self._proc

    def _sample_once(self) -> TelemetrySample:  # pragma: no cover - hardware path
        proc = self._ensure_proc()
        assert proc.stdout is not None
        line = proc.stdout.readline()
        return self.parse_line(line)

    @classmethod
    def parse_line(cls, line: str) -> TelemetrySample:
        """Parse one tegrastats line into a TelemetrySample (pure; unit-testable)."""
        temp_m = cls._RE_TEMP.search(line)
        gr3d_m = cls._RE_GR3D.search(line)
        pow_m = cls._RE_POWER.search(line)
        temp_c = float(temp_m.group(1)) if temp_m else 0.0
        gpu_util = float(gr3d_m.group(1)) if gr3d_m else 0.0
        gpu_clock = float(gr3d_m.group(2)) if (gr3d_m and gr3d_m.group(2)) else 0.0
        power_w = (float(pow_m.group(2)) / 1000.0) if pow_m else 0.0
        # tegrastats has no explicit throttle column; infer from a clamped clock is
        # unreliable, so report False here. jtop is preferred when a real flag matters.
        return TelemetrySample(
            ts=time.time(),
            temp_c=temp_c,
            power_w=power_w,
            gpu_clock_mhz=gpu_clock,
            gpu_util=gpu_util,
            throttled=False,
        )

    def stop(self) -> None:  # pragma: no cover - hardware path
        super().stop()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pick_temp(temps: dict) -> float:  # pragma: no cover - hardware path
    """Pick a representative junction temperature from a jtop temp dict."""
    def _val(x):
        return float(x.get("temp", x)) if isinstance(x, dict) else float(x)

    for key in ("junction", "Tj", "tj", "SOC", "soc0", "CPU", "GPU"):
        if key in temps:
            try:
                return _val(temps[key])
            except (TypeError, ValueError):
                continue
    vals = []
    for v in temps.values():
        try:
            vals.append(_val(v))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else 0.0


def _jtop_throttled(j) -> bool:  # pragma: no cover - hardware path
    """Best-effort real throttle flag across jetson-stats versions."""
    for attr in ("throttling", "throttle"):
        val = getattr(j, attr, None)
        if isinstance(val, dict) and val:
            return any(bool(x) for x in val.values())
        if isinstance(val, bool):
            return val
    return False


__all__ = ["BaseTelemetryReader", "JtopReader", "TegrastatsReader"]
