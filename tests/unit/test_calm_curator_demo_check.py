"""Unit tests for calm_curator_demo_check subprocess oracle runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_demo_check  # noqa: E402
from unicaa01_demo_oracles import get_oracle  # noqa: E402

P01_GOOD = '''
def demo():
    return ["alert", "backup", "email"]
'''

P01_STUB = '''
def demo():
    raise NotImplementedError("stub")
'''

P06_GOOD = '''
def parse_json(s):
    import json
    return json.loads(s)

def demo():
    return parse_json("[1,2]")
'''

P06_BAD = '''
def parse_json(s):
    return None

def demo():
    return parse_json("[1,2]")
'''


def test_p01_eq_pass():
    oracle = get_oracle("p01_priority_queue")
    assert oracle is not None
    r = run_demo_check(main_py=P01_GOOD, oracle=oracle)
    assert r.pass_


def test_p01_stub_fail():
    oracle = get_oracle("p01_priority_queue")
    r = run_demo_check(main_py=P01_STUB, oracle=oracle)
    assert not r.pass_


def test_p06_eq_pass():
    oracle = get_oracle("p06_json_parse")
    r = run_demo_check(main_py=P06_GOOD, oracle=oracle)
    assert r.pass_
    assert r.actual_repr == "[1, 2]"


def test_wrong_list_fail():
    oracle = get_oracle("p06_json_parse")
    r = run_demo_check(main_py=P06_BAD, oracle=oracle)
    assert not r.pass_


def test_timeout_fail():
    oracle = get_oracle("p06_json_parse")
    slow = "import time\ndef demo():\n    time.sleep(10)\n    return [1,2]\n"
    r = run_demo_check(main_py=slow, oracle=oracle, timeout_sec=0.2)
    assert not r.pass_
    assert "timeout" in r.error.lower()
