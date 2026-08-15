"""Per-task clean-tail FPR audit for CAST tau calibration (p61-class noisy neutrals)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jlens_qwen.caa.cast_math import gamma, load_cast_thresholds
from jlens_qwen.caa.s_cond_replay import load_replay_jsonl

GAMMA_HIGH = 0.5
PER_TASK_FPR_CAP = 0.08
CORPUS_CLEAN_FPR_CAP = 0.05
MIN_CLEAN_EPISODES = 20
P61_TASK_SLUG = "unicaa01_p61_neutral"
P61_FPR_CAP = PER_TASK_FPR_CAP
QUIET_RUN_P95_CAP = 0.018


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def episode_clean_fpr(
    rows: list[dict[str, Any]],
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
    gamma_floor: float = GAMMA_HIGH,
) -> dict[str, Any]:
    """Per-episode false-positive rate: fraction of tokens with gamma >= gamma_floor."""
    n = len(rows)
    if n == 0:
        return {
            "n_tokens": 0,
            "fpr_s_above_tau": float("nan"),
            "fpr_gamma_high": float("nan"),
            "median_s_cond": float("nan"),
            "p95_s_cond": float("nan"),
        }
    sc = [float(r["s_cond"]) for r in rows]
    above_tau = sum(1 for s in sc if s >= tau)
    above_gamma = sum(
        1 for s in sc if gamma(s, tau, sigma, tau_hard) >= gamma_floor
    )
    return {
        "n_tokens": n,
        "fpr_s_above_tau": above_tau / n,
        "fpr_gamma_high": above_gamma / n,
        "median_s_cond": _percentile(sc, 50),
        "p95_s_cond": _percentile(sc, 95),
    }


def audit_clean_tail(
    calib_dir: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Audit every clean replay JSONL for per-task and corpus-level FPR."""
    thresholds = load_cast_thresholds(metadata_path)
    tau = thresholds["tau"]
    sigma = thresholds["sigma"]
    tau_hard = thresholds["tau_hard"]

    episodes: list[dict[str, Any]] = []
    clean_dir = calib_dir / "clean"
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"missing clean corpus dir: {clean_dir}")

    for path in sorted(clean_dir.glob("*.jsonl")):
        meta, rows = load_replay_jsonl(path)
        if not rows:
            continue
        stats = episode_clean_fpr(
            rows, tau=tau, sigma=sigma, tau_hard=tau_hard
        )
        episodes.append(
            {
                "run_id": str(meta.get("run_id") or path.stem),
                "task_slug": str(meta.get("task_slug") or ""),
                "path": str(path),
                **stats,
            }
        )

    total_tokens = sum(int(e["n_tokens"]) for e in episodes)
    total_gamma_high = sum(
        int(round(float(e["fpr_gamma_high"]) * int(e["n_tokens"])))
        for e in episodes
        if e["n_tokens"] > 0
    )
    corpus_fpr_gamma = total_gamma_high / max(1, total_tokens)
    total_above_tau = sum(
        int(round(float(e["fpr_s_above_tau"]) * int(e["n_tokens"])))
        for e in episodes
        if e["n_tokens"] > 0
    )
    corpus_fpr_tau = total_above_tau / max(1, total_tokens)

    worst = max(episodes, key=lambda e: float(e.get("fpr_gamma_high") or 0.0))
    p61_ep = next(
        (e for e in episodes if e.get("task_slug") == P61_TASK_SLUG),
        None,
    )

    manifest_path = calib_dir / "split_manifest.json"
    tau_excluded: list[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tau_excluded = list(manifest.get("tau_excluded_noisy") or [])

    return {
        "tau": tau,
        "sigma": sigma,
        "tau_hard": tau_hard,
        "n_clean_episodes": len(episodes),
        "n_clean_tokens": total_tokens,
        "corpus_fpr_gamma_high": corpus_fpr_gamma,
        "corpus_fpr_s_above_tau": corpus_fpr_tau,
        "worst_task_slug": worst.get("task_slug"),
        "worst_fpr_gamma_high": worst.get("fpr_gamma_high"),
        "p61": p61_ep,
        "tau_excluded_noisy": tau_excluded,
        "thresholds": {
            "per_task_fpr_cap": PER_TASK_FPR_CAP,
            "corpus_fpr_cap": CORPUS_CLEAN_FPR_CAP,
            "min_clean_episodes": MIN_CLEAN_EPISODES,
            "gamma_high": GAMMA_HIGH,
            "p61_fpr_cap": P61_FPR_CAP,
        },
        "episodes": episodes,
    }


def _episode_fpr_value(ep: dict[str, Any]) -> float:
    val = ep.get("fpr_gamma_high")
    if val is None:
        return 1.0
    return float(val)


def clean_tail_gate_passes(audit: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    n_eps = int(audit.get("n_clean_episodes") or 0)
    min_eps = int(
        (audit.get("thresholds") or {}).get("min_clean_episodes") or MIN_CLEAN_EPISODES
    )
    checks.append(
        {
            "id": "CT-1",
            "pass": n_eps >= min_eps,
            "detail": f"n_clean_episodes={n_eps} min={min_eps}",
        }
    )

    corp = float(audit.get("corpus_fpr_gamma_high") or 1.0)
    corp_cap = float(
        (audit.get("thresholds") or {}).get("corpus_fpr_cap") or CORPUS_CLEAN_FPR_CAP
    )
    checks.append(
        {
            "id": "CT-2",
            "pass": corp <= corp_cap + 1e-4,
            "detail": f"corpus_fpr_gamma_high={corp:.4f} cap={corp_cap}",
        }
    )

    cap = float(
        (audit.get("thresholds") or {}).get("per_task_fpr_cap") or PER_TASK_FPR_CAP
    )
    violators = [
        e
        for e in (audit.get("episodes") or [])
        if _episode_fpr_value(e) > cap + 1e-4
    ]
    checks.append(
        {
            "id": "CT-3",
            "pass": not violators,
            "detail": (
                "all tasks within cap"
                if not violators
                else "; ".join(
                    f"{v['task_slug']}={_episode_fpr_value(v):.4f}"
                    for v in violators[:5]
                )
            ),
        }
    )

    p61 = audit.get("p61")
    p61_cap = float((audit.get("thresholds") or {}).get("p61_fpr_cap") or P61_FPR_CAP)
    if p61 is None:
        checks.append(
            {
                "id": "CT-4",
                "pass": False,
                "detail": f"missing {P61_TASK_SLUG} in clean corpus",
            }
        )
    else:
        p61_fpr = _episode_fpr_value(p61)
        checks.append(
            {
                "id": "CT-4",
                "pass": p61_fpr <= p61_cap + 1e-4,
                "detail": f"p61_fpr_gamma_high={p61_fpr:.4f} cap={p61_cap}",
            }
        )

    excluded = list(audit.get("tau_excluded_noisy") or [])
    checks.append(
        {
            "id": "CT-5",
            "pass": bool(excluded) or corp <= corp_cap + 1e-4,
            "detail": (
                f"tau_excluded_noisy={excluded}"
                if excluded
                else "no exclusions (corpus quiet enough)"
            ),
        }
    )

    return all(c["pass"] for c in checks), checks


def write_clean_tail_tsv(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "run_id\ttask_slug\tn_tokens\tfpr_tau\tfpr_gamma\tmed_s\tp95_s"
    ]
    for ep in audit.get("episodes") or []:
        lines.append(
            "\t".join(
                [
                    str(ep.get("run_id") or ""),
                    str(ep.get("task_slug") or ""),
                    str(ep.get("n_tokens") or 0),
                    f"{float(ep.get('fpr_s_above_tau') or 0):.4f}",
                    f"{float(ep.get('fpr_gamma_high') or 0):.4f}",
                    f"{float(ep.get('median_s_cond') or 0):.4f}",
                    f"{float(ep.get('p95_s_cond') or 0):.4f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_tau_fit_clean_ids(
    clean_runs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    seed_ids: set[str],
    *,
    per_task_cap: float = PER_TASK_FPR_CAP,
    max_iters: int = 8,
    token_cap: int = 256,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Iteratively drop noisy clean runs from the tau fit pool."""
    pool = set(seed_ids)
    excluded: list[dict[str, Any]] = []
    sigma_probe = 0.001

    def _sconds(rid: str) -> list[float]:
        for meta, rows in clean_runs:
            if str(meta["run_id"]) != rid:
                continue
            capped = rows[:token_cap]
            return [float(r["s_cond"]) for r in capped]
        return []

    for _ in range(max_iters):
        vals: list[float] = []
        for rid in pool:
            vals.extend(_sconds(rid))
        if not vals:
            break
        tau = _percentile(vals, 95)
        tau_hard = _percentile(vals, 50)
        newly: list[str] = []
        for rid in sorted(pool):
            sc = _sconds(rid)
            if not sc:
                continue
            fpr = sum(
                1
                for s in sc
                if gamma(s, tau, sigma_probe, tau_hard) >= GAMMA_HIGH
            ) / len(sc)
            if fpr > per_task_cap + 1e-4:
                newly.append(rid)
        if not newly:
            break
        for rid in newly:
            pool.discard(rid)
            task_slug = ""
            for meta, _ in clean_runs:
                if str(meta["run_id"]) == rid:
                    task_slug = str(meta.get("task_slug") or "")
                    break
            excluded.append(
                {"run_id": rid, "task_slug": task_slug, "reason": "per_task_fpr_cap"}
            )
    return pool, excluded


def filter_quiet_tau_fit_ids(
    clean_runs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    pool_ids: set[str],
    *,
    quiet_p95_cap: float = QUIET_RUN_P95_CAP,
    token_cap: int = 256,
    min_pool: int = 8,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Drop high-tail clean runs from tau fit (noisy houses like p61)."""
    run_p95s: list[tuple[str, str, float]] = []
    for meta, rows in clean_runs:
        rid = str(meta.get("run_id") or "")
        if rid not in pool_ids:
            continue
        sc = [float(r["s_cond"]) for r in rows[:token_cap]]
        if not sc:
            continue
        run_p95s.append((rid, str(meta.get("task_slug") or ""), _percentile(sc, 95)))

    if not run_p95s:
        return set(pool_ids), []

    adaptive_cap = _percentile([p for _, _, p in run_p95s], 75)
    cap = min(quiet_p95_cap, adaptive_cap)
    quiet = {rid for rid, _, p95 in run_p95s if p95 <= cap + 1e-6}
    if len(quiet) < min_pool:
        return set(pool_ids), []

    dropped = [
        {"run_id": rid, "task_slug": slug, "reason": "quiet_p95_cap", "run_p95": p95}
        for rid, slug, p95 in run_p95s
        if rid not in quiet
    ]
    return quiet, dropped


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
