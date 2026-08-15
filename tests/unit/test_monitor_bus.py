"""Unit tests for MonitorChaosBus ring + publish latency."""

from __future__ import annotations

import time

from jlens_qwen.chaos.monitor_bus import MonitorChaosBus


def test_monitor_event_ring_and_subscribe():
    bus = MonitorChaosBus()
    received: list[dict] = []
    bus.subscribe(received.append)
    bus.begin_invocation("s1", "i1", device="cpu")

    payload = {
        "A_vocab": {"entropy": 0.5, "margin": 0.8},
        "B_vocab": {
            "code_mass": 0.5,
            "distress_mass": 0.2,
            "stub_mass": 0.15,
            "junk_mass": 0.0,
        },
        "C_vocab": {"flip_rate": 0.1, "oscillation": 0.0},
        "probes": {},
    }
    flags = bus.record_token(0, 100, payload)
    fired = bus.record_monitor_event(0, 100, flags, payload)
    assert fired is not None
    assert fired["recipe_id"] == "B_STUB_DISTRESS"
    assert len(received) == 1
    assert received[0]["step"] == 0

    snap = bus.snapshot()
    assert snap["monitor"]["n_recipe_fires"] >= 1
    live = bus.live_state()
    assert live["recipe_id"] == "B_STUB_DISTRESS"


def test_monitor_publish_speed_budget():
    """Publish path (eval+ring+notify) must stay ≤100µs p95 without subscribers."""
    bus = MonitorChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    payload = {
        "A_vocab": {"entropy": 2.6, "margin": 0.1},
        "B_vocab": {
            "code_mass": 0.1,
            "distress_mass": 0.2,
            "stub_mass": 0.15,
            "junk_mass": 0.0,
        },
        "C_vocab": {"flip_rate": 0.6, "oscillation": 0.5},
        "probes": {},
    }
    samples: list[float] = []
    for step in range(2000):
        flags = bus.record_token(step, 100 + step, payload)
        t0 = time.perf_counter_ns()
        bus.record_monitor_event(step, 100 + step, flags, payload)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    assert p95 <= 100.0, f"publish p95={p95:.2f}µs exceeds 100µs budget"


def test_finalize_includes_recipe_timeline():
    bus = MonitorChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    payload = {
        "A_vocab": {"entropy": 0.5, "margin": 0.8},
        "B_vocab": {
            "code_mass": 0.5,
            "distress_mass": 0.2,
            "stub_mass": 0.15,
            "junk_mass": 0.0,
        },
        "C_vocab": {"flip_rate": 0.1, "oscillation": 0.0},
        "probes": {},
    }
    for step in range(10):
        flags = bus.record_token(step, step, payload)
        bus.record_monitor_event(step, step, flags, payload)
    bus.set_gen_ms(10.0)
    rec = bus.finalize()
    assert "monitor" in rec
    assert rec["monitor"]["n_recipe_fires"] >= 1
    assert len(rec["monitor"]["recipe_timeline"]) >= 1
