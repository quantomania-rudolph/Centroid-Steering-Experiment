#!/usr/bin/env python
"""Run curator hardening gates G0–G8 sequentially; exit 1 on any failure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_hardening_proof"
PY = sys.executable


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)


def gate_g0() -> dict:
    tests = [
        "tests/unit/test_calm_curator_demo_check.py",
        "tests/unit/test_bridge_policy.py",
        "tests/unit/test_no_sanitize_on_admit.py",
        "tests/unit/test_openrouter_budget.py",
        "tests/unit/test_oracle_coverage.py",
        "tests/unit/test_calm_curation.py",
        "tests/unit/test_calm_tail_purity.py",
        "tests/unit/test_demo_stress_oracles.py",
        "tests/unit/test_semantic_invariants.py",
        "tests/unit/test_narrative_purity.py",
        "tests/unit/test_hardcoded_return_trap.py",
        "tests/unit/test_admit_contract_v3.py",
        "tests/unit/test_quality_suite_v4.py",
        "tests/unit/test_quality_loop_v4.py",
        "tests/integration/test_calm_revise_loop.py",
        "tests/integration/test_calm_admit_invariants.py",
        "tests/integration/test_v3_revise_semantic_loop.py",
        "tests/integration/test_corpus_recertification.py",
    ]
    co = _run([PY, "-m", "pytest", *tests, "--collect-only", "-q"])
    n_tests = 0
    for line in (co.stdout or "").splitlines():
        line = line.strip()
        if line and not line.startswith("=") and "<" in line:
            n_tests += 1
    proc = _run([PY, "-m", "pytest", *tests, "-q"])
    passed_line = [ln for ln in (proc.stdout or "").splitlines() if "passed" in ln]
    n_passed = 0
    if passed_line:
        import re

        m = re.search(r"(\d+) passed", passed_line[-1])
        if m:
            n_passed = int(m.group(1))
    return {
        "pass": proc.returncode == 0 and n_passed >= 45,
        "returncode": proc.returncode,
        "n_tests_collected": max(n_tests, n_passed),
        "n_passed": n_passed,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def gate_g1() -> dict:
    proc = _run([PY, "research/calm_curator_sabotage.py"])
    evidence = {}
    stdout = (proc.stdout or "").strip()
    try:
        evidence = json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    evidence = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    n_match = int(evidence.get("n_match") or 0)
    return {"pass": proc.returncode == 0 and n_match == 8, "n_match": n_match, "raw": evidence}


def gate_g2() -> dict:
    proc = _run([PY, "research/build_curator_briefs.py"])
    briefs_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
    index = json.loads((briefs_dir / "index.json").read_text(encoding="utf-8"))
    skipped = json.loads((briefs_dir / "skipped.json").read_text(encoding="utf-8"))
    n_briefs = int(index.get("n_briefs") or 0)
    n_skipped = int(skipped.get("n_skipped") or 0)
    return {
        "pass": proc.returncode == 0 and n_briefs == 54 and n_skipped == 0,
        "n_briefs": n_briefs,
        "n_skipped": n_skipped,
    }


def _parse_miner_dry_run(stdout: str) -> dict:
    try:
        return json.loads(stdout.strip())
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return {}


def gate_g3() -> dict:
    """Clean sweep (0 tails) OR post-pilot: only v3-admitted tails remain."""
    calm_dir = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"
    txts = list(calm_dir.glob("*.txt"))
    n_txt = len(txts)
    proc = _run([PY, "research/caa_mine_natural_pairs.py", "--unicaa01-corpus", "--dry-run"])
    miner = _parse_miner_dry_run(proc.stdout or "")
    n_admitted = int(miner.get("n_admitted") or len(miner.get("admitted_slugs") or []))

    if n_txt == 0 and n_admitted == 0:
        return {"pass": True, "mode": "clean_sweep", "n_calm_tail_txt": 0, "n_admitted": 0}

    # Post-pilot: every on-disk .txt must be v3 admitted; miner count matches.
    v3_ok = True
    for p in txts:
        meta_path = p.with_suffix(".meta.json")
        if not meta_path.is_file():
            v3_ok = False
            break
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("pipeline_version") or 0) < 3 or meta.get("status") != "admitted":
            v3_ok = False
            break
    g10 = PROOF / "G10_pilot.json"
    g10_pass = False
    n_pilot = 0
    if g10.is_file():
        g10_data = json.loads(g10.read_text(encoding="utf-8"))
        g10_pass = bool(g10_data.get("pass"))
        n_pilot = int(g10_data.get("n_pilot_admitted") or 0)
    post_ok = v3_ok and g10_pass and n_admitted == n_pilot and n_txt == n_pilot
    return {
        "pass": post_ok,
        "mode": "post_pilot_v3_only",
        "n_calm_tail_txt": n_txt,
        "n_admitted": n_admitted,
        "n_pilot_admitted": n_pilot,
        "v3_meta_ok": v3_ok,
        "g10_pass": g10_pass,
    }


def gate_g4() -> dict:
    proc = _run([PY, "-m", "pytest", "tests/integration/test_v3_revise_semantic_loop.py", "-q"])
    return {"pass": proc.returncode == 0, "returncode": proc.returncode}


def gate_g5() -> dict:
    proc = _run([PY, "-m", "pytest", "tests/unit/test_openrouter_budget.py", "-q"])
    return {"pass": proc.returncode == 0, "returncode": proc.returncode}


def gate_g6() -> dict:
    sys.path.insert(0, str(ROOT / "research"))
    from unicaa01_oracle_invariants import verify_invariant_coverage  # noqa: E402
    from calm_curator_demo_check import check_invariants, check_no_hardcode  # noqa: E402

    cov = verify_invariant_coverage()
    fx = ROOT / "tests" / "fixtures" / "oracle_negative"
    neg = [
        check_no_hardcode(main_py=(fx / "p06_hardcoded.py").read_text(encoding="utf-8"), pair_id="p06_json_parse").pass_
        is False,
        check_invariants(main_py=(fx / "p06_import_json.py").read_text(encoding="utf-8"), pair_id="p06_json_parse").pass_
        is False,
        check_invariants(main_py=(fx / "p01_no_lock.py").read_text(encoding="utf-8"), pair_id="p01_priority_queue").pass_
        is False,
    ]
    return {"pass": cov["pass"] and all(neg), "coverage": cov, "negative_fixtures_fail": neg}


def gate_g7() -> dict:
    proc = _run([PY, "-m", "pytest", "tests/unit/test_demo_stress_oracles.py", "-q"])
    return {"pass": proc.returncode == 0, "returncode": proc.returncode}


def gate_g8() -> dict:
    proc = _run([PY, "-m", "pytest", "tests/unit/test_narrative_purity.py", "-q"])
    return {"pass": proc.returncode == 0, "returncode": proc.returncode}


def gate_g12() -> dict:
    proc = _run(
        [
            PY,
            "-m",
            "pytest",
            "tests/unit/test_quality_suite_v4.py",
            "tests/unit/test_quality_loop_v4.py",
            "-q",
        ]
    )
    thin = ROOT / "tests" / "fixtures" / "oracle_negative" / "p06_thin_parser.py"
    sys.path.insert(0, str(ROOT / "research"))
    from calm_curator_demo_check import run_quality_suite  # noqa: E402

    thin_fail = run_quality_suite(
        main_py=thin.read_text(encoding="utf-8"), pair_id="p06_json_parse"
    ).pass_ is False
    return {
        "pass": proc.returncode == 0 and thin_fail,
        "returncode": proc.returncode,
        "thin_fixture_fails_suite": thin_fail,
    }


GATES = [
    ("G0_unit", gate_g0),
    ("G1_sabotage", gate_g1),
    ("G2_briefs", gate_g2),
    ("G3_clean_sweep", gate_g3),
    ("G4_revise", gate_g4),
    ("G5_budget", gate_g5),
    ("G6_invariants", gate_g6),
    ("G7_stress", gate_g7),
    ("G8_narrative", gate_g8),
    ("G12_quality_unit", gate_g12),
]


def main() -> int:
    PROOF.mkdir(parents=True, exist_ok=True)
    all_pass = True
    for gate_id, fn in GATES:
        evidence = fn()
        payload = {"gate_id": gate_id, "pass": evidence.get("pass", False), "evidence": evidence}
        out = PROOF / f"{gate_id}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not payload["pass"]:
            all_pass = False
            return 1
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
