#!/usr/bin/env python
"""Sabotage fixtures for calm curator mechanical + v3 narrative gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import BRIDGE_CATALOG  # noqa: E402
from build_curator_briefs import build_all_briefs  # noqa: E402
from calm_curator_demo_check import check_no_hardcode  # noqa: E402
from validate_calm_curator_output import validate_calm_curator_output  # noqa: E402
from validate_narrative_purity import narrative_all_pass, validate_narrative_purity  # noqa: E402

OUT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_curator_sabotage_report.json"
FIXTURES = ROOT / "tests" / "fixtures" / "oracle_negative"


def _p06_brief(briefs: list) -> dict[str, Any]:
    for b in briefs:
        if b.pair_id == "p06_json_parse":
            return b.to_dict()
    return briefs[0].to_dict()


def _sabotage_cases(brief: dict[str, Any]) -> list[tuple[str, str, bool]]:
    pl = brief["prefix_lock"]
    prefix = str(pl["prefix_text"])
    good_suffix = (
        "NotImplementedError` by implementing a minimal JSON parser.\n\n"
        f"{brief['output_contract']['task_dir']}/main.py\n```python\n"
        '"""Minimal JSON array parser."""\n'
        "def parse_array(s: str) -> list:\n"
        "    s = s.strip()\n"
        "    if not (s.startswith('[') and s.endswith(']')):\n"
        "        raise ValueError('array only')\n"
        "    inner = s[1:-1].strip()\n"
        "    if not inner:\n"
        "        return []\n"
        "    return [int(x.strip()) for x in inner.split(',')]\n\n"
        "def demo():\n"
        '    return parse_array("[1,2]")\n```\n'
    )
    good = prefix + good_suffix
    bridge_echo = prefix + BRIDGE_CATALOG[0] + "\n" + good_suffix
    cross_task = prefix + "thread-safe run_next with heap pop logic.\n" + good_suffix
    cases: list[tuple[str, str, bool]] = [
        ("S1_prefix_break", prefix + "@@@PREFIX_LOCK_BREAK@@@", False),
        ("S2_notimplemented", prefix + "fix stub\n```python\nraise NotImplementedError\n```", False),
        ("S3_sorry", prefix + "sorry I cannot fix this\n```\npass\n```", False),
        ("S4_pip", prefix + "run pip install foo\n```\npass\n```", False),
        ("S5_good_shape", good, True),
        ("S6_bridge_catalog", bridge_echo, False),
        ("S7_hardcoded_p06", prefix + good_suffix, False),
        ("S8_cross_task_p06", cross_task, False),
    ]
    return cases


def _would_admit(case_id: str, brief: dict[str, Any], text: str) -> bool:
    if case_id == "S7_hardcoded_p06":
        main_py = (FIXTURES / "p06_hardcoded.py").read_text(encoding="utf-8")
        return check_no_hardcode(main_py=main_py, pair_id="p06_json_parse").pass_

    if case_id in ("S6_bridge_catalog", "S8_cross_task_p06"):
        return narrative_all_pass(
            validate_narrative_purity(
                full_text=text,
                prefix_text=brief["prefix_lock"]["prefix_text"],
                pair_id=brief["pair_id"],
            )
        )

    rep = validate_calm_curator_output(
        brief=brief,
        candidate_text=text,
        attempt_repair=False,
        sanitize=False,
        strict_assembly=(case_id != "S5_good_shape"),
        allow_bridge=False,
        run_miner_parity=False,
        run_demo_oracle=False,
    )
    if case_id == "S5_good_shape":
        rep = validate_calm_curator_output(brief=brief, candidate_text=text)
    return rep.pass_


def main() -> int:
    briefs = build_all_briefs()
    if not briefs:
        print(json.dumps({"error": "no briefs"}))
        return 1
    brief = _p06_brief(briefs)
    rows: list[dict[str, Any]] = []
    for case_id, text, expect_pass in _sabotage_cases(brief):
        actual_pass = _would_admit(case_id, brief, text)
        rows.append(
            {
                "case": case_id,
                "expect_pass": expect_pass,
                "actual_pass": actual_pass,
                "match": actual_pass == expect_pass,
            }
        )
    report = {"n_cases": len(rows), "n_match": sum(1 for r in rows if r["match"]), "cases": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["n_match"] == report["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
