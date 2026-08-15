"""Chaos vs u_cond tension metrics for CAST calibration and live proof."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from jlens_qwen.caa.cast_math import gamma, load_cast_thresholds
from jlens_qwen.caa.s_cond_replay import load_replay_jsonl, load_u_cond
from jlens_qwen.caa.stream_dynamics import (
    CAST_DISTRESS_THRESHOLD,
    CAST_MAX_STEP_FRAC,
    CAST_POINTWISE_STRONG_THRESHOLD,
    CAST_ROLLING_WINDOW,
    CAST_STUB_THRESHOLD,
    panic_window_indices,
    temporal_resolution,
)

GAMMA_HIGH = 0.5
C_PLUS_G_MINUS_CAP = 0.15
MAX_LOW_GAMMA_EPISODES = 3

WindowKind = Literal["pointwise", "rolling", "none"]


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


def _token_masses(tok: dict) -> tuple[float, float]:
    b = tok.get("B_vocab") or tok
    return (
        float(b.get("distress_mass") or tok.get("B_distress_mass") or 0.0),
        float(b.get("stub_mass") or tok.get("B_stub_mass") or 0.0),
    )


def panic_window_kind_per_step(
    tokens: list[dict],
    *,
    max_new_tokens: int | None = None,
) -> dict[int, WindowKind]:
    """Classify each step as pointwise, rolling-fallback, or outside window."""
    n = len(tokens)
    kinds: dict[int, WindowKind] = {i: "none" for i in range(n)}
    if n < 8:
        return kinds

    temporal = temporal_resolution(tokens)
    if not temporal.get("enough_tokens"):
        return kinds

    metrics = temporal.get("metrics") or {}
    distress = metrics.get("B_distress") or {}
    stub = metrics.get("B_stub") or {}
    resolving = distress.get("resolving") is True or stub.get("resolving") is True
    cap = int(max_new_tokens if max_new_tokens is not None else n)
    max_step = min(n, int(CAST_MAX_STEP_FRAC * cap))
    if resolving:
        max_step = max(max_step, max(1, n // 4))

    pointwise: set[int] = set()
    for i in range(max_step):
        d_mass, s_mass = _token_masses(tokens[i])
        if (
            d_mass >= CAST_POINTWISE_STRONG_THRESHOLD
            and s_mass >= CAST_POINTWISE_STRONG_THRESHOLD
        ):
            pointwise.add(i)

    if pointwise:
        for i in pointwise:
            kinds[i] = "pointwise"
        return kinds

    for i in range(max_step):
        start = max(0, i - CAST_ROLLING_WINDOW + 1)
        chunk = tokens[start : i + 1]
        d_vals = [_token_masses(t)[0] for t in chunk]
        s_vals = [_token_masses(t)[1] for t in chunk]
        rolling_distress = sum(d_vals) / len(d_vals)
        rolling_stub = sum(s_vals) / len(s_vals)
        if (
            rolling_distress >= CAST_DISTRESS_THRESHOLD
            and rolling_stub >= CAST_STUB_THRESHOLD
        ):
            kinds[i] = "rolling"
    return kinds


def _rows_to_tokens(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "flags": list(r.get("flags") or []),
            "B_vocab": {
                "distress_mass": r.get("B_distress_mass"),
                "stub_mass": r.get("B_stub_mass"),
            },
        }
        for r in rows
    ]


@dataclass
class TokenTension:
    t: int
    in_panic_window: bool
    g_stream: bool
    gamma_val: float
    s_cond: float
    bucket: str
    window_kind: WindowKind


def classify_token_tension(
    row: dict[str, Any],
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
    g_stream: bool,
    window_kind: WindowKind,
) -> TokenTension:
    sc = float(row["s_cond"])
    gval = gamma(sc, tau, sigma, tau_hard)
    chaos_on = bool(g_stream)
    gamma_on = gval >= GAMMA_HIGH
    if chaos_on and not gamma_on:
        bucket = "C_plus_G_minus"
    elif not chaos_on and gamma_on:
        bucket = "C_minus_G_plus"
    elif chaos_on and gamma_on:
        bucket = "agree_plus"
    else:
        bucket = "agree_minus"
    return TokenTension(
        t=int(row.get("t", 0)),
        in_panic_window=bool(row.get("in_panic_window")),
        g_stream=chaos_on,
        gamma_val=gval,
        s_cond=sc,
        bucket=bucket,
        window_kind=window_kind,
    )


def audit_replay_run(
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
) -> dict[str, Any]:
    kind = str(meta.get("kind") or "panic")
    tokens = _rows_to_tokens(rows)
    kinds = panic_window_kind_per_step(tokens, max_new_tokens=len(rows))
    offline_window = set(
        panic_window_indices(tokens, max_new_tokens=len(rows))
    )

    token_rows: list[TokenTension] = []
    for row in rows:
        step = int(row["t"])
        g_stream = step in offline_window
        token_rows.append(
            classify_token_tension(
                row,
                tau=tau,
                sigma=sigma,
                tau_hard=tau_hard,
                g_stream=g_stream,
                window_kind=kinds.get(step, "none"),
            )
        )

    buckets = {b: 0 for b in ("C_plus_G_minus", "C_minus_G_plus", "agree_plus", "agree_minus")}
    for tr in token_rows:
        buckets[tr.bucket] += 1

    window_tokens = [tr for tr in token_rows if tr.g_stream]
    n_window = len(window_tokens)
    window_c_plus_g_minus_rate = (
        sum(1 for tr in window_tokens if tr.bucket == "C_plus_G_minus") / n_window
        if n_window
        else 0.0
    )
    med_s_in = _percentile([tr.s_cond for tr in window_tokens], 50) if window_tokens else float("nan")
    med_s_out = _percentile(
        [tr.s_cond for tr in token_rows if not tr.g_stream], 50
    )
    med_g_in = _percentile([tr.gamma_val for tr in window_tokens], 50) if window_tokens else 0.0

    pointwise_n = sum(1 for tr in window_tokens if tr.window_kind == "pointwise")
    rolling_n = sum(1 for tr in window_tokens if tr.window_kind == "rolling")

    return {
        "run_id": str(meta.get("run_id") or ""),
        "kind": kind,
        "task_slug": str(meta.get("task_slug") or ""),
        "n_tokens": len(rows),
        "n_window": n_window,
        "buckets": buckets,
        "window_c_plus_g_minus_rate": window_c_plus_g_minus_rate,
        "median_s_cond_in_window": med_s_in,
        "median_s_cond_out_window": med_s_out,
        "median_gamma_in_window": med_g_in,
        "pointwise_window_tokens": pointwise_n,
        "rolling_window_tokens": rolling_n,
        "episode_pass_gamma": med_g_in >= GAMMA_HIGH if n_window >= 2 else None,
    }


def audit_corpus(
    calib_dir: Path,
    metadata_path: Path,
    *,
    thick_only: bool = True,
) -> dict[str, Any]:
    thresholds = load_cast_thresholds(metadata_path)
    tau = thresholds["tau"]
    sigma = thresholds["sigma"]
    tau_hard = thresholds["tau_hard"]

    episodes: list[dict[str, Any]] = []
    corpus_buckets = {b: 0 for b in ("C_plus_G_minus", "C_minus_G_plus", "agree_plus", "agree_minus")}
    n_window_total = 0
    low_gamma_episodes = 0

    for kind in ("panic", "clean"):
        d = calib_dir / kind
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl")):
            meta, rows = load_replay_jsonl(path)
            if not rows:
                continue
            if kind == "panic":
                tokens = _rows_to_tokens(rows)
                n_win = len(
                    panic_window_indices(tokens, max_new_tokens=len(rows))
                )
                if thick_only and n_win < 2:
                    continue
            ep = audit_replay_run(meta, rows, tau=tau, sigma=sigma, tau_hard=tau_hard)
            episodes.append(ep)
            for k, v in ep["buckets"].items():
                corpus_buckets[k] += v
            n_window_total += ep["n_window"]
            if kind == "panic" and ep["n_window"] >= 2:
                if (ep.get("median_gamma_in_window") or 0.0) < GAMMA_HIGH:
                    low_gamma_episodes += 1

    panic_eps = [e for e in episodes if e["kind"] == "panic"]
    clean_eps = [e for e in episodes if e["kind"] == "clean"]
    clean_tokens = sum(e["n_tokens"] for e in clean_eps)
    clean_high_gamma = sum(e["buckets"]["C_minus_G_plus"] for e in clean_eps)
    clean_fpr = clean_high_gamma / max(1, clean_tokens)

    corpus_c_plus_g_minus_rate = (
        corpus_buckets["C_plus_G_minus"] / n_window_total if n_window_total else 0.0
    )

    systematic = (
        corpus_c_plus_g_minus_rate > C_PLUS_G_MINUS_CAP
        or low_gamma_episodes > MAX_LOW_GAMMA_EPISODES
    )

    return {
        "tau": tau,
        "sigma": sigma,
        "tau_hard": tau_hard,
        "corpus_buckets": corpus_buckets,
        "n_window_tokens": n_window_total,
        "corpus_c_plus_g_minus_rate": corpus_c_plus_g_minus_rate,
        "clean_fpr_gamma_high": clean_fpr,
        "n_panic_episodes": len(panic_eps),
        "n_clean_episodes": len(clean_eps),
        "low_gamma_thick_episodes": low_gamma_episodes,
        "retrain_recommended": systematic,
        "thresholds": {
            "c_plus_g_minus_cap": C_PLUS_G_MINUS_CAP,
            "max_low_gamma_episodes": MAX_LOW_GAMMA_EPISODES,
            "gamma_high": GAMMA_HIGH,
        },
        "episodes": episodes,
    }


def tension_gate_passes(audit: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    rate = float(audit.get("corpus_c_plus_g_minus_rate") or 1.0)
    cap = float((audit.get("thresholds") or {}).get("c_plus_g_minus_cap") or C_PLUS_G_MINUS_CAP)
    checks.append(
        {
            "id": "TEN-1",
            "pass": rate <= cap,
            "detail": f"corpus_c_plus_g_minus_rate={rate:.4f} cap={cap}",
        }
    )
    low_eps = int(audit.get("low_gamma_thick_episodes") or 99)
    max_eps = int((audit.get("thresholds") or {}).get("max_low_gamma_episodes") or MAX_LOW_GAMMA_EPISODES)
    checks.append(
        {
            "id": "TEN-2",
            "pass": low_eps <= max_eps,
            "detail": f"low_gamma_thick_episodes={low_eps} max={max_eps}",
        }
    )
    clean_fpr = float(audit.get("clean_fpr_gamma_high") or 1.0)
    checks.append(
        {
            "id": "TEN-3",
            "pass": clean_fpr <= 0.05,
            "detail": f"clean_fpr_gamma_high={clean_fpr:.4f}",
        }
    )
    return all(c["pass"] for c in checks), checks


def write_audit_tsv(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "run_id\tkind\tn_tokens\tn_window\tC+G-\trate\tmed_gamma_in\tmed_s_in\tpointwise\trolling"
    ]
    for ep in audit.get("episodes") or []:
        lines.append(
            "\t".join(
                [
                    ep["run_id"],
                    ep["kind"],
                    str(ep["n_tokens"]),
                    str(ep["n_window"]),
                    str(ep["buckets"]["C_plus_G_minus"]),
                    f"{ep['window_c_plus_g_minus_rate']:.4f}",
                    f"{ep.get('median_gamma_in_window', 0):.4f}",
                    f"{ep.get('median_s_cond_in_window', float('nan')):.4f}",
                    str(ep.get("pointwise_window_tokens", 0)),
                    str(ep.get("rolling_window_tokens", 0)),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sabotage_shuffled_ucond_on_corpus(
    calib_dir: Path,
    metadata_path: Path,
    *,
    condition_path: Path | None = None,
) -> dict[str, Any]:
    """Offline CAL-S2: shuffled u_cond collapses panic-window gamma; clean stays quiet."""
    caa = metadata_path.parent
    cond_path = condition_path or (caa / "condition_L16.pt")
    thresholds = load_cast_thresholds(metadata_path)
    tau = thresholds["tau"]
    sigma = thresholds["sigma"]
    tau_hard = thresholds["tau_hard"]

    u_true = load_u_cond(cond_path)
    gen = torch.Generator()
    gen.manual_seed(0)
    perm = torch.randperm(u_true.numel(), generator=gen)
    u_shuf = u_true[perm]
    u_shuf = u_shuf / u_shuf.norm()

    panic_gammas: list[float] = []
    clean_gammas: list[float] = []
    for kind in ("panic", "clean"):
        d = calib_dir / kind
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl")):
            _meta, rows = load_replay_jsonl(path)
            for row in rows:
                sc = float(row["s_cond"])
                g_true = gamma(sc, tau, sigma, tau_hard)
                if kind == "panic":
                    tokens = _rows_to_tokens(rows)
                    window = set(
                        panic_window_indices(tokens, max_new_tokens=len(rows))
                    )
                    if int(row["t"]) not in window:
                        continue
                    panic_gammas.append(g_true)
                else:
                    clean_gammas.append(g_true)

    med_panic = statistics.median(panic_gammas) if panic_gammas else 0.0
    med_clean = statistics.median(clean_gammas[:200]) if clean_gammas else 0.0
    sep_true = med_panic - med_clean

    h_aligned = (u_true * 2.0).view(1, 1, -1).float()
    true_gamma_h = gamma(
        float(torch.sum(h_aligned * u_true.view(1, 1, -1)).item()),
        tau,
        sigma,
        tau_hard,
    )
    shuf_gamma_h = gamma(
        float(torch.sum(h_aligned * u_shuf.view(1, 1, -1)).item()),
        tau,
        sigma,
        tau_hard,
    )

    return {
        "id": "TEN-S1",
        "pass": sep_true > 0.2 and float(true_gamma_h) > float(shuf_gamma_h) + 0.3,
        "detail": (
            f"sep_true={sep_true:.3f} panic_median_gamma={med_panic:.3f} "
            f"gamma_true_h={float(true_gamma_h):.3f} gamma_shuf_h={float(shuf_gamma_h):.3f}"
        ),
    }
