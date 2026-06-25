"""Telemetry: mock thermal dynamics + tegrastats parsing + factory (CLAUDE.md §10)."""

from __future__ import annotations

from poise.telemetry import MockTelemetryReader, TelemetrySample, make_reader
from poise.telemetry.reader import TegrastatsReader


class FakeClock:
    """Manually-advanced monotonic clock for deterministic thermal stepping."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float):
        self.t += dt


def test_sample_schema_immutable():
    s = TelemetrySample(ts=1.0, temp_c=50.0, power_w=20.0, gpu_clock_mhz=1300.0,
                        gpu_util=80.0, throttled=False)
    assert s.to_dict()["temp_c"] == 50.0
    try:
        s.temp_c = 60.0  # type: ignore[misc]
        assert False, "TelemetrySample must be frozen"
    except Exception:
        pass


def test_mock_warms_up_under_load():
    clk = FakeClock()
    r = MockTelemetryReader(ambient_c=25.0, temp_max_c=87.0, init_temp_c=30.0,
                            load_level=1.0, budget=32, layer_total=32, clock_fn=clk)
    first = r.read().temp_c
    for _ in range(60):
        clk.advance(1.0)
        last = r.read().temp_c
    assert last > first  # temperature rises under sustained load
    assert last > 60.0   # and reaches a meaningfully hot steady state


def test_mock_cools_when_load_drops():
    clk = FakeClock()
    r = MockTelemetryReader(ambient_c=25.0, init_temp_c=30.0, load_level=1.0,
                            tau_s=10.0, clock_fn=clk)
    for _ in range(60):
        clk.advance(1.0)
        hot = r.read().temp_c
    r.set_load(0.0)
    r.set_budget(16)
    for _ in range(120):
        clk.advance(1.0)
        cool = r.read().temp_c
    assert cool < hot


def test_mock_throttles_at_ceiling():
    clk = FakeClock()
    r = MockTelemetryReader(ambient_c=25.0, temp_max_c=70.0, init_temp_c=30.0,
                            load_level=1.0, budget=32, clock_fn=clk)
    throttled_seen = False
    nominal_clock = r.nominal_clock_mhz
    for _ in range(200):
        clk.advance(1.0)
        s = r.read()
        if s.throttled:
            throttled_seen = True
            assert s.gpu_clock_mhz < nominal_clock  # clock clamps on throttle
            break
    assert throttled_seen


def test_lower_budget_runs_cooler():
    clk_hi, clk_lo = FakeClock(), FakeClock()
    hi = MockTelemetryReader(init_temp_c=30.0, load_level=0.8, budget=32, clock_fn=clk_hi)
    lo = MockTelemetryReader(init_temp_c=30.0, load_level=0.8, budget=16, clock_fn=clk_lo)
    for _ in range(80):
        clk_hi.advance(1.0); clk_lo.advance(1.0)
        t_hi = hi.read().temp_c
        t_lo = lo.read().temp_c
    assert t_lo < t_hi  # fewer layers => cooler steady state


def test_tegrastats_parse():
    line = ("RAM 1000/1000MB GR3D_FREQ 99%@1300 tj@72.5C "
            "VDD_GPU_SOC 4500mW/4500mW")
    s = TegrastatsReader.parse_line(line)
    assert abs(s.temp_c - 72.5) < 1e-6
    assert abs(s.gpu_clock_mhz - 1300.0) < 1e-6
    assert abs(s.power_w - 4.5) < 1e-6
    assert s.gpu_util == 99.0


def test_factory_returns_mock_off_device(cfg):
    r = make_reader(cfg)
    assert isinstance(r, MockTelemetryReader)
    assert isinstance(r.read(), TelemetrySample)
