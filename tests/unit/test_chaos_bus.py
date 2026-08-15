"""Unit tests for ChaosBus ring buffer + downsample."""

from __future__ import annotations

from jlens_qwen.chaos.bus import ChaosBus


def test_bus_record_and_snapshot():
    bus = ChaosBus()
    bus.begin_invocation("s1", "i1", {"task": "t"}, device="cpu")
    for step in range(5):
        flags = bus.record_token(
            step,
            token_id=100 + step,
            payload={
                "A_vocab": {"entropy": 1.0 + 0.1 * step, "margin": 0.5},
                "B_vocab": {
                    "code_mass": 0.8,
                    "distress_mass": 0.05,
                    "stub_mass": 0.0,
                    "junk_mass": 0.0,
                },
                "C_vocab": {"flip_rate": 0.1, "oscillation": 0.0},
                "probes": {},
            },
        )
        assert isinstance(flags, list)
    snap = bus.snapshot()
    assert snap["n_tokens"] == 5
    assert snap["active"] is True
    bus.set_gen_ms(100.0)
    rec = bus.finalize()
    assert rec["version"] == 2
    assert len(rec["tokens"]) == 5
    assert rec["token_ids"] == [100, 101, 102, 103, 104]
    assert rec["n_tokens_full"] == 5
    assert "timing" in rec
    assert rec["composites"]["PLDS"] is not None
    hints = rec.get("divergence_hints") or {}
    assert hints.get("d_lexicon") is None
    assert hints.get("b_distress_rate") == 0.0


def test_bus_divergence_hints_first_cross():
    bus = ChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    thr_d = 0.12
    thr_s = 0.10
    for step in range(50):
        hot = step == 42
        bus.record_token(
            step,
            token_id=100 + step,
            payload={
                "A_vocab": {"entropy": 0.5, "margin": 0.5},
                "B_vocab": {
                    "code_mass": 0.5,
                    "distress_mass": thr_d if hot else 0.0,
                    "stub_mass": thr_s if hot else 0.0,
                    "junk_mass": 0.0,
                },
                "C_vocab": {"flip_rate": 0.0, "oscillation": 0.0},
                "probes": {},
            },
        )
    rec = bus.finalize()
    hints = rec["divergence_hints"]
    assert hints["d_lexicon"] == 42
    assert hints["d_flags"] == 42
    assert hints["d_distress_only"] == 42
    assert hints["d_stub_only"] == 42
    assert len(rec["token_ids"]) == 50


def test_bus_downsample_keeps_flagged():
    bus = ChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    # Force many tokens; inject flags sparingly (avoid A_low_margin on calm rows)
    for step in range(600):
        hot = step % 17 == 0
        bus.record_token(
            step,
            token_id=step,
            payload={
                "A_vocab": {
                    "entropy": 2.6 if hot else 0.5,
                    "margin": 0.5,
                },
                "B_vocab": {
                    "code_mass": 0.8,
                    "distress_mass": 0.9 if hot else 0.01,
                    "stub_mass": 0.0,
                    "junk_mass": 0.0,
                },
                "C_vocab": {"flip_rate": 0.0, "oscillation": 0.0},
                "probes": {},
            },
        )
    bus.set_gen_ms(50.0)
    rec = bus.finalize()
    assert len(rec["tokens"]) < 600
    assert len(rec["token_ids"]) == 600
    assert rec["n_tokens_full"] == 600
    assert rec["token_ids"] == list(range(600))
    flagged_in = {t["step"] for t in rec["tokens"] if t.get("flags")}
    assert flagged_in, "expected some flags in synthetic high-distress rows"
    # every originally flagged step must survive (odd flagged kept even if not even-index)
    for step in range(0, 600, 17):
        assert any(t["step"] == step and t.get("flags") for t in rec["tokens"])


def test_bus_estimate_cliff_layer():
    bus = ChaosBus()
    bus.begin_invocation("s", "i", device="cpu")
    for step in range(12):
        bus.record_token(
            step,
            token_id=step,
            payload={
                "A_vocab": {"entropy": 0.5, "margin": 0.5},
                "B_vocab": {
                    "code_mass": 0.8,
                    "distress_mass": 0.01,
                    "stub_mass": 0.0,
                    "junk_mass": 0.0,
                },
                "C_vocab": {"flip_rate": 0.0, "oscillation": 0.0},
                "probes": {
                    "10": {"A": {"entropy": 0.1}, "B_transport": {"code_mass": 0.5}},
                    "16": {"A": {"entropy": 0.1}, "B_transport": {"code_mass": 0.5}},
                    "22": {"A": {"entropy": 0.9}, "B_transport": {"code_mass": 0.5}},
                },
            },
        )
    cliff = bus.estimate_cliff_layer()
    assert cliff is not None
    assert 8 <= cliff <= 26
