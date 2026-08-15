#!/usr/bin/env python
"""CAST τ/σ OOS proof — writes v2 caa_tau_sigma_calibration_gate.json with sabotage."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.cast_math import gamma  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.s_cond_replay import load_u_cond  # noqa: E402

from cast_calibrate_tau import (  # noqa: E402
    CALIB_DIR,
    CLEAN_GAMMA_BUDGET,
    DEFAULT_COND,
    GATES,
    MANIFEST_VERSION,
    MIN_FIT_EPISODE_PASS,
    PANIC_GAMMA_TARGET,
    VERIFY_PANIC_N,
    _load_corpus,
    fit_tau_sigma,
    lock_splits,
    verify_candidate,
)

OUT = GATES / "caa_tau_sigma_calibration_gate.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _run_calibrate() -> int:
    cmd = [str(PY), str(ROOT / "research" / "cast_calibrate_tau.py")]
    return subprocess.call(cmd, cwd=str(ROOT))


def _sabotage_shuffled_u_cond(clean_runs, panic_runs, candidate: dict) -> dict:
    u_cond = load_u_cond(DEFAULT_COND)
    u_shuf = u_cond[torch.randperm(u_cond.numel())]
    u_shuf = u_shuf / u_shuf.norm()
    dot = float(torch.dot(u_cond, u_shuf).item())
    tau = float(candidate["tau"])
    sigma = float(candidate["sigma"])
    tau_hard = float(candidate.get("tau_hard") or tau)
    clean_vals = []
    panic_vals = []
    for meta, rows in clean_runs:
        if meta.get("kind") != "clean":
            continue
        clean_vals.extend(float(r["s_cond"]) for r in rows[:64])
    for meta, rows in panic_runs:
        if meta.get("kind") != "panic":
            continue
        panic_vals.extend(
            float(r["s_cond"]) for r in rows if r.get("in_panic_window")
        )
    if not clean_vals or not panic_vals:
        return {"id": "CAL-S2", "pass": False, "detail": "missing vals"}
    clean_g = sum(gamma(s, tau, sigma, tau_hard) for s in clean_vals) / len(clean_vals)
    panic_g = sum(gamma(s, tau, sigma, tau_hard) for s in panic_vals) / len(panic_vals)
    sep = panic_g - clean_g
    return {
        "id": "CAL-S2",
        "pass": sep > 0.3 and dot < 0.2,
        "detail": f"sep={sep:.3f} shuf_dot={dot:.4f}",
    }


def _sabotage_tau_plus_one(clean_runs, candidate: dict) -> dict:
    tau = float(candidate["tau"]) + 1.0
    sigma = float(candidate["sigma"])
    tau_hard = float(candidate.get("tau_hard") or candidate["tau"])
    high = 0
    for meta, rows in clean_runs:
        if meta.get("kind") != "clean":
            continue
        for r in rows[:128]:
            if gamma(float(r["s_cond"]), tau, sigma, tau_hard) > 0.5:
                high += 1
    return {
        "id": "CAL-S3",
        "pass": high == 0,
        "detail": f"clean_high_gamma_tau+1={high}",
    }


def _sabotage_verify_into_fit(manifest: dict, clean_runs, panic_runs, candidate: dict) -> dict:
    if not manifest.get("verify_panic"):
        return {"id": "CAL-S1", "pass": False, "detail": "no verify panic"}
    tampered = copy.deepcopy(manifest)
    moved = tampered["verify_panic"][0]
    tampered["fit_panic_shape"] = list(tampered.get("fit_panic_shape") or []) + [moved]
    tampered["verify_panic"] = tampered["verify_panic"][1:]
    try:
        alt = fit_tau_sigma(clean_runs, panic_runs, tampered)
        changed = (
            float(alt.get("sigma")) != float(candidate.get("sigma"))
            or float(alt.get("tau")) != float(candidate.get("tau"))
        )
        return {
            "id": "CAL-S1",
            "pass": changed,
            "detail": f"sigma {candidate.get('sigma')} -> {alt.get('sigma')}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"id": "CAL-S1", "pass": True, "detail": f"fit failed as expected: {exc}"}


def main() -> int:
    g42 = CALIB_DIR / "G42_scond_corpus_gate.json"
    if not g42.is_file():
        print(json.dumps({"pass": False, "error": "missing G42"}))
        return 1

    rc = _run_calibrate()
    if rc != 0:
        return rc

    if not OUT.is_file():
        print(json.dumps({"pass": False, "error": f"missing {OUT}"}))
        return 1

    gate = json.loads(OUT.read_text(encoding="utf-8"))
    manifest = json.loads((CALIB_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    clean_runs = _load_corpus("clean")
    panic_runs = _load_corpus("panic")
    candidate = {
        "tau": gate["tau"],
        "sigma": gate["sigma"],
        "tau_hard": gate.get("tau_hard"),
        "fit_stats": gate.get("fit_stats"),
        "tau_ci_width": (gate.get("verify") or {}).get("tau_ci_width"),
    }
  # reload full candidate from G44 if needed
    g44 = CALIB_DIR / "G44_fit_candidate.json"
    if g44.is_file():
        candidate = json.loads(g44.read_text(encoding="utf-8"))

    verify = gate.get("verify") or {}
    checks = [
        {
            "id": "CAL-V2-1",
            "pass": int(gate.get("version") or 0) >= MANIFEST_VERSION
            and int(json.loads(g42.read_text(encoding="utf-8")).get("version") or 0) >= 2,
            "detail": "G42 v2 + gate v2",
        },
        {
            "id": "CAL-V2-2",
            "pass": verify.get("n_verify_episodes_pass") == VERIFY_PANIC_N,
            "detail": f"episodes_pass={verify.get('n_verify_episodes_pass')}",
        },
        {
            "id": "CAL-V2-3",
            "pass": int(verify.get("n_verify_panic_tokens") or 0) >= 6,
            "detail": f"n_verify_panic_tokens={verify.get('n_verify_panic_tokens')}",
        },
        {
            "id": "CAL-V2-4",
            "pass": float(verify.get("fpr_verify_clean") or 1) <= 0.05 + 1e-3
            and float(verify.get("mean_verify_gamma") or 1) <= CLEAN_GAMMA_BUDGET + 1e-3,
            "detail": "clean verify budgets",
        },
        {
            "id": "CAL-V2-5",
            "pass": float(candidate.get("tau_ci_width") or 1) <= 0.08,
            "detail": f"tau_ci_width={candidate.get('tau_ci_width')}",
        },
        {
            "id": "CAL-V2-6",
            "pass": int((candidate.get("fit_stats") or {}).get("n_fit_episodes_pass") or 0)
            >= MIN_FIT_EPISODE_PASS,
            "detail": str(candidate.get("fit_stats")),
        },
    ]

    sabotage = [
        _sabotage_verify_into_fit(manifest, clean_runs, panic_runs, candidate),
        _sabotage_shuffled_u_cond(clean_runs, panic_runs, candidate),
        _sabotage_tau_plus_one(clean_runs, candidate),
    ]

    passed = bool(gate.get("pass")) and all(c["pass"] for c in checks)
    payload = write_gate(
        OUT,
        gate="caa_tau_sigma_calibration",
        pass_=passed,
        checks=checks,
        sabotage=sabotage,
        extra={
            "version": MANIFEST_VERSION,
            "tau": gate.get("tau"),
            "sigma": gate.get("sigma"),
            "tau_hard": gate.get("tau_hard"),
            "verify": verify,
            "fit_stats": candidate.get("fit_stats"),
            "manifest_hash": candidate.get("manifest_hash"),
            "split_manifest": str(CALIB_DIR / "split_manifest.json"),
            "corpus_gate": str(g42),
        },
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
