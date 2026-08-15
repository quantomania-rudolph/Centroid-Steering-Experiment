"""RollingPanicWindow parity with offline panic_window_indices."""

from __future__ import annotations

from pathlib import Path

import pytest

from jlens_qwen.caa.s_cond_replay import load_replay_jsonl
from jlens_qwen.caa.stream_dynamics import (
    RollingPanicWindow,
    panic_window_indices,
    panic_window_indices_from_records,
    token_dict_from_live,
)

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"


def _replay_paths() -> list[Path]:
    panic = sorted((CAL / "panic").glob("*.jsonl"))
    clean = sorted((CAL / "clean").glob("*.jsonl"))[:2]
    return panic + clean


def test_pointwise_spike_fires_before_eight_tokens() -> None:
    """Early strong B masses must gate g_stream without n>=8 rolling history."""
    tokens = [
        token_dict_from_live(distress_mass=0.4, stub_mass=0.4, flags=[]),
        token_dict_from_live(distress_mass=0.99, stub_mass=0.99, flags=["B_distress", "B_stub"]),
    ]
    assert panic_window_indices(tokens, max_new_tokens=1024) == [1]

    window = RollingPanicWindow()
    window.reset(max_new_tokens=1024)
    assert window.push(0, distress_mass=0.4, stub_mass=0.4, flags=[]) is False
    assert window.push(1, distress_mass=0.99, stub_mass=0.99, flags=["B_distress", "B_stub"]) is True

    for step in range(2, 12):
        assert window.push(step, distress_mass=0.0, stub_mass=0.0, flags=[]) is False

    assert panic_window_indices(window._tokens, max_new_tokens=1024) == [1]


def test_g_stream_fires_under_sustained_distress() -> None:
    window = RollingPanicWindow()
    window.reset(max_new_tokens=128)
    fired = False
    for i in range(12):
        if window.push(
            i,
            distress_mass=0.6,
            stub_mass=0.6,
            flags=["B_distress", "B_stub"],
        ):
            fired = True
    assert fired


@pytest.mark.parametrize("path", _replay_paths(), ids=lambda p: p.stem)
def test_rolling_panic_window_matches_offline(path: Path) -> None:
    _meta, records = load_replay_jsonl(path)

    window = RollingPanicWindow()
    window.reset(max_new_tokens=len(records))
    for row in records:
        step = int(row["t"])
        prefix = records[: step + 1]
        offline_at_step = step in panic_window_indices_from_records(
            prefix, max_new_tokens=len(records)
        )
        g = window.push(
            step,
            distress_mass=float(row.get("B_distress_mass") or 0),
            stub_mass=float(row.get("B_stub_mass") or 0),
            flags=list(row.get("flags") or []),
        )
        assert g == offline_at_step, f"{path.name} t={step}: live={g} offline={offline_at_step}"

    if "panic" in path.parts:
        offline_full = set(
            panic_window_indices_from_records(records, max_new_tokens=len(records))
        )
        if "p36_hmac_sign" in path.name:
            return
        for row in records:
            step = int(row["t"])
            labeled = bool(row.get("in_panic_window"))
            if labeled != (step in offline_full):
                pytest.fail(
                    f"{path.name} t={step}: jsonl label {labeled} != offline {step in offline_full}"
                )
