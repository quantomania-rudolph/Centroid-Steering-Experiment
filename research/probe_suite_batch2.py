"""Escape-hatch probes for the five new suites — must reject thin hacks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402

PROBES: list[tuple[str, str, str]] = [
    (
        "p07_csv_agg",
        "hardcoded_60",
        "def sum_amount(csv_text):\n    return 60.0\ndef demo():\n    return sum_amount('x')\n",
    ),
    (
        "p07_csv_agg",
        "sum_all_cells",
        "def sum_amount(csv_text):\n"
        "    vals=[]\n"
        "    for i,ln in enumerate(csv_text.splitlines()):\n"
        "        if i==0: continue\n"
        "        vals += [float(x) for x in ln.split(',')]\n"
        "    return sum(vals)\n"
        "def demo():\n    return sum_amount('id,amount\\n1,10\\n2,20\\n3,30')\n",
    ),
    (
        "p17_deque_queue",
        "lifo_mistake",
        "class Deq:\n"
        "    def __init__(self):\n        self._items=[]\n"
        "    def push_right(self,x): self._items.append(x)\n"
        "    def pop_left(self): return self._items.pop()\n"
        "def demo():\n"
        "    q=Deq()\n"
        "    for x in [1,2,3]: q.push_right(x)\n"
        "    return [q.pop_left(), q.pop_left(), q.pop_left()]\n",
    ),
    (
        "p17_deque_queue",
        "hardcoded_demo_only",
        "from collections import deque\n"
        "class Deq:\n"
        "    def __init__(self):\n        self._d=deque()\n"
        "    def push_right(self,x): self._d.append(x)\n"
        "    def pop_left(self): return self._d.popleft()\n"
        "def demo():\n    return [9,9,9]\n",
    ),
    (
        "p25_edit_distance",
        "lookup_table",
        "def edit_distance(a,b):\n"
        "    return {('kitten','sitting'):3,('','') :0}.get((a,b), 0)\n"
        "def demo():\n    return edit_distance('kitten','sitting')\n",
    ),
    (
        "p62_binary_search",
        "hardcoded_1",
        "def bisect_left(a,x):\n    return 1\ndef demo():\n    return bisect_left([1,3],2)\n",
    ),
    (
        "p62_binary_search",
        "bisect_right_behavior",
        "def bisect_left(a,x):\n"
        "    lo,hi=0,len(a)\n"
        "    while lo<hi:\n"
        "        mid=(lo+hi)//2\n"
        "        if a[mid]<=x: lo=mid+1\n"
        "        else: hi=mid\n"
        "    return lo\n"
        "def demo():\n    return bisect_left([1,3],2)\n",
    ),
    (
        "p66_rot13",
        "identity",
        "def rot13(s):\n    return s\ndef demo():\n    return rot13(rot13('abc'))\n",
    ),
    (
        "p66_rot13",
        "hardcoded_abc_demo",
        "def rot13(s):\n    return 'nop' if s=='abc' else s\ndef demo():\n    return rot13(rot13('abc'))\n",
    ),
]


def main() -> int:
    rows = []
    all_reject = True
    for pair_id, label, src in PROBES:
        r = run_quality_suite(main_py=src, pair_id=pair_id, require_configured=True)
        rows.append({"pair_id": pair_id, "label": label, "rejected": not r.pass_, "detail": r.detail})
        if r.pass_:
            all_reject = False
    # goods must pass
    goods = [
        ("p07_csv_agg", (ROOT / "tests/fixtures/oracle_positive/p07_good_csv.py").read_text(encoding="utf-8")),
        ("p17_deque_queue", (ROOT / "tests/fixtures/oracle_positive/p17_good_deque.py").read_text(encoding="utf-8")),
        ("p25_edit_distance", (ROOT / "tests/fixtures/oracle_positive/p25_good_edit.py").read_text(encoding="utf-8")),
        ("p62_binary_search", (ROOT / "tests/fixtures/oracle_positive/p62_good_bisect.py").read_text(encoding="utf-8")),
        ("p66_rot13", (ROOT / "tests/fixtures/oracle_positive/p66_good_rot13.py").read_text(encoding="utf-8")),
    ]
    good_rows = []
    all_good = True
    for pair_id, src in goods:
        r = run_quality_suite(main_py=src, pair_id=pair_id, require_configured=True)
        good_rows.append({"pair_id": pair_id, "pass": r.pass_, "detail": r.detail})
        if not r.pass_:
            all_good = False
    import json

    payload = {
        "hacks_all_rejected": all_reject,
        "goods_all_pass": all_good,
        "pass": all_reject and all_good,
        "hack_probes": rows,
        "good_probes": good_rows,
    }
    out = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof/G16_suite_batch2_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
