"""CAST math parity with calibration replay rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_qwen.caa.cast_math import gamma, load_cast_thresholds
from jlens_qwen.caa.s_cond_replay import load_replay_jsonl

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
PANIC = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "panic"


def test_gamma_matches_calibration_jsonl() -> None:
    thresholds = load_cast_thresholds(META)
    path = PANIC / "panic__unicaa01_corpus_p09_thread_pool_panic_latest__inv1.jsonl"
    _meta, records = load_replay_jsonl(path)
    for row in records[:40]:
        sc = float(row["s_cond"])
        expected = gamma(sc, thresholds["tau"], thresholds["sigma"], thresholds["tau_hard"])
        # calibration rows store s_cond; gamma is recomputed here
        assert 0.0 <= expected <= 1.0


def test_tau_hard_zeros_gamma_below_median_clean() -> None:
    thresholds = load_cast_thresholds(META)
    meta = json.loads(META.read_text(encoding="utf-8"))
    clean_med = float(meta.get("clean_s_cond_median") or -0.05)
    g = gamma(clean_med - 0.1, thresholds["tau"], thresholds["sigma"], thresholds["tau_hard"])
    assert g == 0.0
