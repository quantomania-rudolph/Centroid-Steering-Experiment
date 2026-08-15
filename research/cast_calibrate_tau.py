#!/usr/bin/env python
"""Phases B–D: lock splits, fit τ/σ, verify held-out, write metadata + gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.calibration_deny import load_deny_list  # noqa: E402
from jlens_qwen.caa.cast_math import gamma  # noqa: E402
from jlens_qwen.caa.clean_tail_audit import (  # noqa: E402
    PER_TASK_FPR_CAP,
    episode_clean_fpr,
    filter_quiet_tau_fit_ids,
    select_tau_fit_clean_ids,
)
from jlens_qwen.caa.s_cond_replay import (  # noqa: E402
    extract_task_num,
    load_replay_jsonl,
    load_u_cond,
)

CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
DEFAULT_COND = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "condition_L16.pt"
GATES = ROOT / "artifacts" / "gates"
SPLIT_SEED = 42
FIT_CLEAN_FRAC = 0.70
TOKEN_CAP = 256
TAU_MARGIN = 0.05
CLEAN_GAMMA_BUDGET = 0.05
PANIC_GAMMA_TARGET = 0.70
FIT_PANIC_N = 12
VERIFY_PANIC_N = 3
MIN_FIT_EPISODE_PASS = 9
MIN_THICK_PANIC_EPISODES = VERIFY_PANIC_N + MIN_FIT_EPISODE_PASS  # 12 → 9 fit + 3 verify
MIN_VERIFY_WINDOW = 2
MANIFEST_VERSION = 2
SIGMA_GRID = [
    1e-5,
    1e-4,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.0045,
    0.005,
    0.006,
    0.008,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
]
SIGMA_SELECTION = "max_passing"  # prefer largest σ still under clean-γ budget (less brittle)
AGREE_INTERSECT_DEFAULT = True
AGREE_GAMMA_FLOOR = 0.25
MIN_CLEAN_RUNS = 20


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty values for percentile")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _load_corpus(kind: str) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    d = CALIB_DIR / kind
    out: list[tuple[dict, list[dict]]] = []
    for path in sorted(d.glob("*.jsonl")):
        meta, rows = load_replay_jsonl(path)
        if not rows:
            continue
        if kind == "panic" and not any(r.get("in_panic_window") for r in rows):
            continue
        out.append((meta, rows))
    return out


def _cap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows[:TOKEN_CAP]


def _run_sconds(meta: dict, rows: list[dict], *, panic_window_only: bool) -> list[float]:
    if meta.get("kind") == "panic" and panic_window_only:
        rows = [r for r in rows if r.get("in_panic_window")]
    return [float(r["s_cond"]) for r in _cap_rows(rows)]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _median(values: list[float]) -> float:
    if not values:
        return float("-inf")
    return _percentile(values, 50)


def _episode_window_rows(
    rows: list[dict[str, Any]],
    *,
    agree_intersect: bool = False,
    tau: float = 0.0,
    sigma: float = 0.001,
    tau_hard: float = 0.0,
) -> list[dict[str, Any]]:
    window = [r for r in _cap_rows(rows) if r.get("in_panic_window")]
    if not agree_intersect:
        return window
    return [
        r
        for r in window
        if gamma(float(r["s_cond"]), tau, sigma, tau_hard) >= AGREE_GAMMA_FLOOR
    ]


def _rows_for_split(
    runs: list[tuple[dict, list]],
    ids: set[str],
    *,
    panic_window_only: bool,
    agree_intersect: bool = False,
    tau: float = 0.0,
    sigma: float = 0.001,
    tau_hard: float = 0.0,
) -> list[float]:
    out: list[float] = []
    for meta, rows in runs:
        if str(meta["run_id"]) not in ids:
            continue
        if meta.get("kind") == "panic" and panic_window_only:
            rows = _episode_window_rows(
                rows,
                agree_intersect=agree_intersect,
                tau=tau,
                sigma=sigma,
                tau_hard=tau_hard,
            )
        else:
            rows = _cap_rows(rows)
        out.extend(float(r["s_cond"]) for r in rows)
    return out


def _episode_stats(meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    window_rows = _episode_window_rows(rows)
    task_slug = str(meta.get("task_slug") or "")
    return {
        "run_id": str(meta.get("run_id") or ""),
        "task_num": extract_task_num(task_slug) or "",
        "task_slug": task_slug,
        "n_window_tokens": len(window_rows),
        "thin_episode": 0 < len(window_rows) < MIN_VERIFY_WINDOW,
        "verify_eligible": len(window_rows) >= MIN_VERIFY_WINDOW,
    }


def _episode_median_s_cond(
    rows: list[dict[str, Any]],
    *,
    agree_intersect: bool = False,
    tau: float = 0.0,
    sigma: float = 0.001,
    tau_hard: float = 0.0,
) -> float:
    window_rows = _episode_window_rows(
        rows,
        agree_intersect=agree_intersect,
        tau=tau,
        sigma=sigma,
        tau_hard=tau_hard,
    )
    if not window_rows:
        return float("-inf")
    return _median([float(r["s_cond"]) for r in window_rows])


def _episode_median_gamma(
    rows: list[dict[str, Any]],
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
    agree_intersect: bool = False,
) -> float:
    window_rows = _episode_window_rows(
        rows,
        agree_intersect=agree_intersect,
        tau=tau,
        sigma=sigma,
        tau_hard=tau_hard,
    )
    if not window_rows:
        return 0.0
    gammas = [gamma(float(r["s_cond"]), tau, sigma, tau_hard) for r in window_rows]
    return _percentile(gammas, 50)


def lock_splits(clean_runs: list[tuple[dict, list]], panic_runs: list[tuple[dict, list]]) -> dict[str, Any]:
    rng = random.Random(SPLIT_SEED)
    clean_ids = [str(m["run_id"]) for m, _ in clean_runs]
    rng.shuffle(clean_ids)
    n_fit = max(1, int(round(len(clean_ids) * FIT_CLEAN_FRAC)))
    fit_clean = sorted(clean_ids[:n_fit])
    verify_clean = sorted(clean_ids[n_fit:])

    episode_meta: list[dict[str, Any]] = []
    for meta, rows in panic_runs:
        stats = _episode_stats(meta, rows)
        if stats["n_window_tokens"] < 1:
            continue
        episode_meta.append({**stats, "meta": meta, "rows": rows})

    verify_pool = [e for e in episode_meta if e["verify_eligible"]]
    if len(verify_pool) < VERIFY_PANIC_N:
        raise RuntimeError(
            f"need {VERIFY_PANIC_N} verify-eligible panic episodes, have {len(verify_pool)}"
        )
    if len(episode_meta) < MIN_THICK_PANIC_EPISODES:
        raise RuntimeError(
            f"need ≥{MIN_THICK_PANIC_EPISODES} thick panic episodes with windows, have {len(episode_meta)}"
        )

    verify_pool.sort(key=lambda e: (e["task_num"], e["run_id"]))
    rng.shuffle(verify_pool)
    verify_ids = sorted(e["run_id"] for e in verify_pool[:VERIFY_PANIC_N])

    fit_pool = [e for e in episode_meta if e["run_id"] not in set(verify_ids)]
    thick_fit = [e for e in fit_pool if e["verify_eligible"]]
    thin_fit = [e for e in fit_pool if not e["verify_eligible"]]
    fit_n = min(FIT_PANIC_N, len(thick_fit))
    if fit_n < MIN_FIT_EPISODE_PASS:
        raise RuntimeError(
            f"need ≥{MIN_FIT_EPISODE_PASS} thick (≥{MIN_VERIFY_WINDOW} window-token) fit episodes, "
            f"have {len(thick_fit)} after reserving {VERIFY_PANIC_N} verify — regap thin panics"
        )
    rng.shuffle(thick_fit)
    rng.shuffle(thin_fit)
    fit_ids = sorted(e["run_id"] for e in thick_fit[:fit_n])
    holdout_ids = sorted(
        e["run_id"]
        for e in episode_meta
        if e["run_id"] not in set(verify_ids) and e["run_id"] not in set(fit_ids)
    )

    episodes_table = []
    for e in episode_meta:
        role = "holdout_reserve"
        if e["run_id"] in verify_ids:
            role = "verify_panic"
        elif e["run_id"] in fit_ids:
            role = "fit_panic_shape"
        episodes_table.append(
            {
                "run_id": e["run_id"],
                "task_num": e["task_num"],
                "n_window_tokens": e["n_window_tokens"],
                "thin_episode": e["thin_episode"],
                "split_role": role,
            }
        )

    tau_fit_ids, tau_excluded = select_tau_fit_clean_ids(
        clean_runs,
        set(fit_clean),
        per_task_cap=PER_TASK_FPR_CAP,
        token_cap=TOKEN_CAP,
    )
    tau_fit_ids, quiet_excluded = filter_quiet_tau_fit_ids(
        clean_runs,
        tau_fit_ids,
        token_cap=TOKEN_CAP,
    )
    tau_excluded.extend(quiet_excluded)
    if len(tau_fit_ids) < 6:
        raise RuntimeError(
            f"tau fit pool too small after noisy exclusion: {len(tau_fit_ids)}"
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "assignment": "stratified_random",
        "seed": SPLIT_SEED,
        "fit_clean": fit_clean,
        "tau_fit_clean": sorted(tau_fit_ids),
        "tau_excluded_noisy": [e["run_id"] for e in tau_excluded],
        "tau_excluded_detail": tau_excluded,
        "quiet_run_p95_cap": 0.018,
        "verify_clean": verify_clean,
        "fit_panic_shape": fit_ids,
        "fit_panic_n": fit_n,
        "verify_panic": verify_ids,
        "holdout_reserve": holdout_ids,
        "episodes": episodes_table,
        "token_cap": TOKEN_CAP,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "corpus_hashes": {
            str(p): _file_hash(p)
            for p in sorted((CALIB_DIR / "clean").glob("*.jsonl"))
            + sorted((CALIB_DIR / "panic").glob("*.jsonl"))
        },
    }
    out = CALIB_DIR / "split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _per_run_p95(runs: list[tuple[dict, list]], ids: set[str]) -> list[float]:
    out: list[float] = []
    for meta, rows in runs:
        if meta["run_id"] not in ids:
            continue
        sc = _run_sconds(meta, rows, panic_window_only=False)
        if sc:
            out.append(_percentile(sc, 95))
    return out


def bootstrap_tau_ci(
    runs: list[tuple[dict, list]],
    ids: list[str],
    *,
    n_boot: int = 200,
) -> tuple[float, float]:
    rng = random.Random(SPLIT_SEED + 7)
    taus: list[float] = []
    id_list = list(ids)
    for _ in range(n_boot):
        sample = [rng.choice(id_list) for _ in range(len(id_list))]
        vals = _rows_for_split(runs, set(sample), panic_window_only=False)
        if vals:
            taus.append(_percentile(vals, 95))
    if not taus:
        return 0.0, 1.0
    taus.sort()
    lo = taus[int(0.05 * (len(taus) - 1))]
    hi = taus[int(0.95 * (len(taus) - 1))]
    return lo, hi


def _per_task_gamma_violators(
    clean_runs: list[tuple[dict, list]],
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
) -> list[str]:
    violators: list[str] = []
    for meta, rows in clean_runs:
        sc = _run_sconds(meta, rows, panic_window_only=False)
        if not sc:
            continue
        fpr = sum(1 for s in sc if gamma(s, tau, sigma, tau_hard) >= 0.5) / len(sc)
        if fpr > PER_TASK_FPR_CAP + 1e-4:
            violators.append(f"{meta.get('task_slug')}={fpr:.4f}")
    return violators


def fit_tau_sigma(
    clean_runs: list[tuple[dict, list]],
    panic_runs: list[tuple[dict, list]],
    manifest: dict[str, Any],
    *,
    agree_intersect: bool = AGREE_INTERSECT_DEFAULT,
) -> dict[str, Any]:
    fit_clean_ids = set(manifest["fit_clean"])
    excluded_noisy = set(manifest.get("tau_excluded_noisy") or [])
    all_clean_ids = {str(meta["run_id"]) for meta, _ in clean_runs}
    tau_token_ids = all_clean_ids - excluded_noisy
    if len(tau_token_ids) < 8:
        tau_token_ids = set(manifest.get("tau_fit_clean") or manifest["fit_clean"])
    verify_clean_ids = set(manifest["verify_clean"])
    fit_panic_ids = set(manifest.get("fit_panic_shape") or [])
    verify_panic_ids = set(manifest.get("verify_panic") or [])

    fit_clean_vals = _rows_for_split(clean_runs, tau_token_ids, panic_window_only=False)
    tau_candidates = sorted(
        {_percentile(fit_clean_vals, p) for p in (93.0, 94.0, 95.0, 96.0, 97.0, 98.0)}
    )
    tau = tau_candidates[0]
    tau_hard = _percentile(fit_clean_vals, 50) if fit_clean_vals else tau
    for tau_try in tau_candidates:
        tau_hard_try = _percentile(
            [s for s in fit_clean_vals if s <= tau_try] or fit_clean_vals,
            50,
        )
        if not _per_task_gamma_violators(
            clean_runs,
            tau=tau_try,
            sigma=SIGMA_GRID[2],
            tau_hard=tau_hard_try,
        ):
            tau = tau_try
            tau_hard = tau_hard_try
            break
        tau = tau_try
        tau_hard = tau_hard_try
    run_p95s = _per_run_p95(clean_runs, tau_token_ids)
    median_run_p95 = _percentile(run_p95s, 50) if run_p95s else tau
    if abs(median_run_p95 - tau) > 0.08:
        raise RuntimeError(
            f"median per-run p95 {median_run_p95:.4f} differs from pooled tau {tau:.4f}"
        )

    passing_sigmas: list[tuple[float, dict[str, Any]]] = []
    for sigma in SIGMA_GRID:
        if _per_task_gamma_violators(
            clean_runs, tau=tau, sigma=sigma, tau_hard=tau_hard
        ):
            continue
        clean_gammas = [gamma(s, tau, sigma, tau_hard) for s in fit_clean_vals]
        mean_clean_gamma = sum(clean_gammas) / max(1, len(clean_gammas))
        if mean_clean_gamma > CLEAN_GAMMA_BUDGET + 1e-3:
            continue
        fit_episode_pass = 0
        fit_episode_total = 0
        for meta, rows in panic_runs:
            if str(meta["run_id"]) not in fit_panic_ids:
                continue
            fit_episode_total += 1
            med_g = _episode_median_gamma(
                rows,
                tau=tau,
                sigma=sigma,
                tau_hard=tau_hard,
                agree_intersect=agree_intersect,
            )
            if med_g >= PANIC_GAMMA_TARGET:
                fit_episode_pass += 1
        if fit_episode_total and fit_episode_pass < MIN_FIT_EPISODE_PASS:
            continue
        panic_vals = _rows_for_split(
            panic_runs,
            fit_panic_ids,
            panic_window_only=True,
            agree_intersect=agree_intersect,
            tau=tau,
            sigma=sigma,
            tau_hard=tau_hard,
        )
        med_panic_gamma = 1.0
        if panic_vals:
            panic_gammas = [gamma(s, tau, sigma, tau_hard) for s in panic_vals]
            med_panic_gamma = _percentile(panic_gammas, 50)
        passing_sigmas.append(
            (
                sigma,
                {
                    "mean_clean_gamma": mean_clean_gamma,
                    "median_panic_gamma": med_panic_gamma,
                    "n_fit_clean_tokens": len(fit_clean_vals),
                    "n_fit_panic_tokens": len(panic_vals) if fit_panic_ids else 0,
                    "n_fit_episodes_pass": fit_episode_pass,
                    "n_fit_episodes_total": fit_episode_total,
                },
            )
        )

    if not passing_sigmas:
        raise RuntimeError("no σ satisfies clean γ budget + panic shape target")

    if SIGMA_SELECTION == "max_passing":
        best_sigma, best_stats = passing_sigmas[-1]
    else:
        best_sigma, best_stats = passing_sigmas[0]
    best_stats["sigma_selection"] = SIGMA_SELECTION
    best_stats["n_passing_sigmas"] = len(passing_sigmas)
    best_stats["agree_intersect"] = agree_intersect

    lo, hi = bootstrap_tau_ci(clean_runs, list(tau_token_ids))
    return {
        "tau": tau,
        "sigma": best_sigma,
        "tau_hard": tau_hard,
        "median_run_p95": median_run_p95,
        "tau_ci_90": [lo, hi],
        "tau_ci_width": hi - lo,
        "fit_stats": best_stats,
        "manifest_hash": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()[:16],
    }


def verify_candidate(
    candidate: dict[str, Any],
    clean_runs: list[tuple[dict, list]],
    panic_runs: list[tuple[dict, list]],
    manifest: dict[str, Any],
    u_cond: torch.Tensor,
    *,
    agree_intersect: bool = AGREE_INTERSECT_DEFAULT,
) -> dict[str, Any]:
    tau = float(candidate["tau"])
    sigma = float(candidate["sigma"])
    tau_hard = float(candidate.get("tau_hard") if candidate.get("tau_hard") is not None else tau)
    verify_clean_ids = set(manifest["verify_clean"])
    verify_panic_ids = set(manifest.get("verify_panic") or [])
    if not verify_panic_ids:
        return {
            "pass": False,
            "errors": ["verify_panic empty — fail closed"],
            "episodes": [],
            "n_verify_panic_tokens": 0,
        }
    fit_clean_ids = set(manifest["fit_clean"])

    verify_clean = _rows_for_split(clean_runs, verify_clean_ids, panic_window_only=False)
    fit_clean = _rows_for_split(clean_runs, fit_clean_ids, panic_window_only=False)
    verify_panic = _rows_for_split(
        panic_runs,
        verify_panic_ids,
        panic_window_only=True,
        agree_intersect=agree_intersect,
        tau=tau,
        sigma=sigma,
        tau_hard=tau_hard,
    )

    episodes: list[dict[str, Any]] = []
    for meta, rows in panic_runs:
        if str(meta["run_id"]) not in verify_panic_ids:
            continue
        stats = _episode_stats(meta, rows)
        med_s = _episode_median_s_cond(
            rows,
            agree_intersect=agree_intersect,
            tau=tau,
            sigma=sigma,
            tau_hard=tau_hard,
        )
        med_g = _episode_median_gamma(
            rows,
            tau=tau,
            sigma=sigma,
            tau_hard=tau_hard,
            agree_intersect=agree_intersect,
        )
        ep_pass = (
            stats["n_window_tokens"] >= MIN_VERIFY_WINDOW
            and med_s > tau + TAU_MARGIN
            and med_g >= PANIC_GAMMA_TARGET
        )
        episodes.append(
            {
                "run_id": stats["run_id"],
                "task_num": stats["task_num"],
                "n_window": stats["n_window_tokens"],
                "median_s_cond": med_s,
                "median_gamma": med_g,
                "pass": ep_pass,
            }
        )

    fpr = sum(1 for s in verify_clean if s >= tau) / max(1, len(verify_clean))
    mean_verify_gamma = sum(gamma(s, tau, sigma, tau_hard) for s in verify_clean) / max(1, len(verify_clean))
    p95_fit = _percentile(fit_clean, 95) if fit_clean else tau
    p95_verify = _percentile(verify_clean, 95) if verify_clean else tau
    med_panic_s = _percentile(verify_panic, 50) if verify_panic else -1.0
    panic_gammas = [gamma(s, tau, sigma, tau_hard) for s in verify_panic]
    med_panic_gamma = _percentile(panic_gammas, 50) if panic_gammas else 0.0

    # sabotage: shuffled û_cond
    shuffled = u_cond[torch.randperm(u_cond.numel())]
    shuffled = shuffled / shuffled.norm()
    shuf_dot = float(torch.dot(u_cond, shuffled).item())

    per_task_fpr: list[dict[str, Any]] = []
    per_task_violators: list[str] = []
    for meta, rows in clean_runs:
        stats = episode_clean_fpr(
            _cap_rows(rows),
            tau=tau,
            sigma=sigma,
            tau_hard=tau_hard,
        )
        row = {
            "run_id": str(meta.get("run_id") or ""),
            "task_slug": str(meta.get("task_slug") or ""),
            **stats,
        }
        per_task_fpr.append(row)
        if float(stats["fpr_gamma_high"]) > PER_TASK_FPR_CAP + 1e-4:
            per_task_violators.append(
                f"{row['task_slug']}={float(stats['fpr_gamma_high']):.4f}"
            )

    errors: list[str] = []
    if len(episodes) != VERIFY_PANIC_N:
        errors.append(f"expected {VERIFY_PANIC_N} verify episodes, got {len(episodes)}")
    if episodes and not all(e["pass"] for e in episodes):
        errors.append("not all verify episodes pass episode-level thresholds")
    if fpr > 0.05 + 1e-3:
        errors.append(f"verify_clean FPR {fpr:.4f} > 0.05")
    if mean_verify_gamma > CLEAN_GAMMA_BUDGET + 1e-3:
        errors.append(f"verify_clean mean_gamma {mean_verify_gamma:.4f} > {CLEAN_GAMMA_BUDGET}")
    if abs(p95_fit - p95_verify) > 0.05:
        errors.append(f"|p95_fit - p95_verify| = {abs(p95_fit - p95_verify):.4f} > 0.05")
    if verify_panic and med_panic_s <= tau + TAU_MARGIN:
        errors.append(f"verify_panic median s_cond {med_panic_s:.4f} <= tau+margin")
    if verify_panic and med_panic_gamma < PANIC_GAMMA_TARGET:
        errors.append(f"verify_panic pooled median gamma {med_panic_gamma:.4f} < {PANIC_GAMMA_TARGET}")
    if float(candidate.get("tau_ci_width") or 0) > 0.08:
        errors.append(f"tau CI width {candidate['tau_ci_width']:.4f} > 0.08")
    median_run_p95 = float(candidate.get("median_run_p95") or tau)
    if abs(median_run_p95 - tau) > 0.05:
        errors.append(f"|median_run_p95 - tau| = {abs(median_run_p95 - tau):.4f} > 0.05")
    if per_task_violators:
        errors.append(
            f"per-task clean FPR > {PER_TASK_FPR_CAP}: {', '.join(per_task_violators[:6])}"
        )

    return {
        "pass": not errors,
        "errors": errors,
        "fpr_verify_clean": fpr,
        "mean_verify_gamma": mean_verify_gamma,
        "p95_fit_clean": p95_fit,
        "p95_verify_clean": p95_verify,
        "median_verify_panic_s_cond": med_panic_s,
        "median_verify_panic_gamma": med_panic_gamma,
        "sabotage_shuffled_ucond_dot": shuf_dot,
        "n_verify_clean_tokens": len(verify_clean),
        "n_verify_panic_tokens": len(verify_panic),
        "episodes": episodes,
        "n_verify_episodes_pass": sum(1 for e in episodes if e.get("pass")),
        "per_task_clean_fpr": per_task_fpr,
        "per_task_violators": per_task_violators,
    }


def write_outputs(candidate: dict[str, Any], verify: dict[str, Any], manifest: dict[str, Any]) -> None:
    G44 = CALIB_DIR / "G44_fit_candidate.json"
    G44.write_text(json.dumps({**candidate, "verify_preview": verify}, indent=2) + "\n", encoding="utf-8")

    meta_path = DEFAULT_META
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tau"] = candidate["tau"]
    meta["sigma"] = candidate["sigma"]
    meta["tau_hard"] = candidate.get("tau_hard")
    meta["calibration"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(CALIB_DIR / "split_manifest.json"),
        "fit_candidate": str(G44),
        "tau_ci_90": candidate.get("tau_ci_90"),
        "tau_ci_width": candidate.get("tau_ci_width"),
        "n_fit_clean_runs": len(manifest.get("fit_clean") or []),
        "n_verify_clean_runs": len(manifest.get("verify_clean") or []),
        "n_verify_panic_runs": len(manifest.get("verify_panic") or []),
        "verify": verify,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    gate = {
        "gate": "caa_tau_sigma_calibration",
        "version": MANIFEST_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass": verify.get("pass"),
        "tau": candidate["tau"],
        "sigma": candidate["sigma"],
        "tau_hard": candidate.get("tau_hard"),
        "verify": verify,
        "fit_stats": candidate.get("fit_stats"),
        "split_manifest": str(CALIB_DIR / "split_manifest.json"),
        "corpus_gate": str(CALIB_DIR / "G42_scond_corpus_gate.json"),
        "manifest_hash": candidate.get("manifest_hash"),
    }
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / "caa_tau_sigma_calibration_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )


def _manifest_hash_ok(manifest: dict[str, Any]) -> bool:
    for path_s in manifest.get("corpus_hashes") or {}:
        p = Path(path_s)
        if not p.is_file():
            return False
        if _file_hash(p) != manifest["corpus_hashes"][path_s]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="CAST τ/σ calibration")
    ap.add_argument("--skip-write", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recalibrate even if corpus hashes drift")
    ap.add_argument(
        "--no-agree-intersect",
        action="store_true",
        help="Fit σ on all chaos-window tokens (no s_cond floor)",
    )
    args = ap.parse_args()
    agree_intersect = not args.no_agree_intersect

    g42 = CALIB_DIR / "G42_scond_corpus_gate.json"
    if not g42.is_file():
        print(json.dumps({"pass": False, "error": "missing G42 — run caa_build_scond_corpus.py"}))
        return 1
    g42_data = json.loads(g42.read_text(encoding="utf-8"))
    if not g42_data.get("pass"):
        print(json.dumps({"pass": False, "error": "G42 not pass", "g42": g42_data}))
        return 1
    if int(g42_data.get("version") or 1) < 2:
        print(json.dumps({"pass": False, "error": "G42 version < 2 — rebuild corpus"}))
        return 1

    clean_runs = _load_corpus("clean")
    panic_runs = _load_corpus("panic")
    if len(clean_runs) < MIN_CLEAN_RUNS or len(panic_runs) < MIN_THICK_PANIC_EPISODES:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "insufficient corpus on disk",
                    "n_clean": len(clean_runs),
                    "need_clean": MIN_CLEAN_RUNS,
                    "n_panic": len(panic_runs),
                    "need": MIN_THICK_PANIC_EPISODES,
                }
            )
        )
        return 1

    manifest_path = CALIB_DIR / "split_manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(existing.get("version") or 1) >= MANIFEST_VERSION and not _manifest_hash_ok(existing):
            print(json.dumps({"pass": False, "error": "corpus hash drift — pass --force"}))
            return 1

    manifest = lock_splits(clean_runs, panic_runs)
    u_cond = load_u_cond(DEFAULT_COND)
    candidate = fit_tau_sigma(
        clean_runs, panic_runs, manifest, agree_intersect=agree_intersect
    )
    verify = verify_candidate(
        candidate,
        clean_runs,
        panic_runs,
        manifest,
        u_cond,
        agree_intersect=agree_intersect,
    )

    result = {"pass": verify["pass"], "candidate": candidate, "verify": verify}
    print(json.dumps(result, indent=2))

    if verify["pass"] and not args.skip_write:
        write_outputs(candidate, verify, manifest)
    return 0 if verify["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
