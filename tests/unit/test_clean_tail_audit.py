"""Unit tests for clean-tail FPR audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.clean_tail_audit import (
    PER_TASK_FPR_CAP,
    clean_tail_gate_passes,
    episode_clean_fpr,
    select_tau_fit_clean_ids,
)


def test_episode_clean_fpr_counts() -> None:
    rows = [{"s_cond": 0.1}, {"s_cond": -0.1}, {"s_cond": 0.01}]
    stats = episode_clean_fpr(rows, tau=0.02, sigma=0.001, tau_hard=-0.05)
    assert stats["n_tokens"] == 3
    assert stats["fpr_s_above_tau"] == 1 / 3


def test_clean_tail_gate_passes_quiet_corpus() -> None:
    audit = {
        "n_clean_episodes": 20,
        "corpus_fpr_gamma_high": 0.03,
        "p61": {"fpr_gamma_high": 0.05},
        "tau_excluded_noisy": [],
        "episodes": [{"fpr_gamma_high": 0.04} for _ in range(20)],
        "thresholds": {
            "min_clean_episodes": 20,
            "corpus_fpr_cap": 0.05,
            "per_task_fpr_cap": PER_TASK_FPR_CAP,
            "p61_fpr_cap": PER_TASK_FPR_CAP,
        },
    }
    ok, checks = clean_tail_gate_passes(audit)
    assert ok
    assert len(checks) == 5


def test_select_tau_fit_excludes_noisy() -> None:
    quiet_rows = [{"s_cond": -0.1 + 0.001 * i} for i in range(40)]
    noisy_rows = [{"s_cond": 0.2 - 0.001 * i} for i in range(40)]
    quiet = ({"run_id": "quiet"}, quiet_rows)
    noisy = ({"run_id": "noisy"}, noisy_rows)
    pool, excluded = select_tau_fit_clean_ids(
        [quiet, noisy],
        {"quiet", "noisy"},
        per_task_cap=0.08,
        token_cap=256,
    )
    assert "quiet" in pool
    assert "noisy" not in pool
    assert excluded
