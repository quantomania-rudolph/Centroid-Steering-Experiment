"""SEM_no_hardcode gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import check_no_hardcode  # noqa: E402


def test_hardcoded_p06_fixture_fails():
    fx = ROOT / "tests" / "fixtures" / "oracle_negative" / "p06_hardcoded.py"
    r = check_no_hardcode(main_py=fx.read_text(encoding="utf-8"), pair_id="p06_json_parse")
    assert r.pass_ is False


def test_non_literal_demo_passes():
    main_py = (
        'def parse_array(s: str):\n'
        "    return [int(x) for x in s.strip('[]').split(',')]\n"
        'def demo():\n'
        '    return parse_array("[1,2]")\n'
    )
    r = check_no_hardcode(main_py=main_py, pair_id="p06_json_parse")
    assert r.pass_ is True
