#!/usr/bin/env python
"""Line-by-line senior review of downstream-ready (admitted) calm tails."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from calm_curator_demo_check import run_quality_suite  # noqa: E402
from verify_admitted_tail import verify_one  # noqa: E402

TAILS = ROOT / "artifacts/caa/UNI_CAA01_corpus/calm_tails"
BRIEFS = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs"
PROOF = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof"

HARD = [
    ("stub_in_prefix", re.compile(r"\bstub\b", re.I)),
    ("i_will_prefix", re.compile(r"\bI will\b")),
    ("i_asked_prefix", re.compile(r"\bI asked\b", re.I)),
    ("fix_colon_prefix", re.compile(r"^Fix:\s")),
]
SOFT = [
    ("broken_in_prefix", re.compile(r"\bbroken\b", re.I)),
    ("error_to_fix_prefix", re.compile(r"^The error", re.I)),
    ("very_short_narr", None),
]


def main() -> int:
    rows = []
    for txt in sorted(TAILS.glob("unicaa01_*.txt")):
        meta = json.loads(txt.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if meta.get("status") != "admitted":
            continue
        brief = json.loads((BRIEFS / f"{meta['brief_id']}.json").read_text(encoding="utf-8"))
        text = txt.read_text(encoding="utf-8")
        pref = brief["prefix_lock"]["prefix_text"]
        narr = text.split("```")[0]
        narr = "\n".join(
            ln
            for ln in narr.splitlines()
            if not ln.strip().endswith("/main.py") and not ln.strip().endswith("\\main.py")
        ).strip()
        code = extract_main_py(text)
        hard = [n for n, rx in HARD if rx.search(pref)]
        soft = [n for n, rx in SOFT if rx and rx.search(pref)]
        if len(narr) < 40:
            soft.append("very_short_narr")
        if re.search(r"NotImplementedError|TODO|FIXME|\"\"\"Broken", code):
            hard.append("code_defect")
        v = verify_one(txt)
        suite = run_quality_suite(main_py=code, pair_id=brief["pair_id"])
        grade = "FAIL" if hard or not v.get("pass") or not suite.pass_ else ("SOFT" if soft else "PASS")
        rows.append(
            {
                "grade": grade,
                "brief_id": meta["brief_id"],
                "pair_id": brief["pair_id"],
                "hard": hard,
                "soft": soft,
                "verify": bool(v.get("pass")),
                "suite": suite.to_dict(),
                "narrative": narr.replace("\n", " | ")[:200],
                "prefix": pref[:140],
                "code_chars": len(code),
            }
        )

    fails = [r for r in rows if r["grade"] == "FAIL"]
    softs = [r for r in rows if r["grade"] == "SOFT"]
    passes = [r for r in rows if r["grade"] == "PASS"]
    payload = {
        "n": len(rows),
        "n_pass": len(passes),
        "n_soft": len(softs),
        "n_fail": len(fails),
        "fails": fails,
        "softs": softs,
        "passes": [{"pair_id": r["pair_id"], "brief_id": r["brief_id"]} for r in passes],
        "all": rows,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / "G23_personal_review_ready.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "n_pass": len(passes), "n_soft": len(softs), "n_fail": len(fails),
                      "fail_ids": [r["brief_id"] for r in fails],
                      "soft_ids": [r["brief_id"] for r in softs]}, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
