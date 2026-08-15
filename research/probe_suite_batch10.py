"""Probe the 10 new suites — hacks must fail, goods must pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402

GOOD: dict[str, str] = {
    "p10_config_validate": '''
def validate_config(cfg):
    p = cfg.get("port")
    return isinstance(p, int) and 1 <= p <= 65535
def demo():
    return validate_config({"port": 99999})
''',
    "p23_knapsack": '''
def knapsack01(weights, values, cap):
    n=len(weights)
    dp=[0]*(cap+1)
    for i in range(n):
        w,v=weights[i],values[i]
        for c in range(cap, w-1, -1):
            dp[c]=max(dp[c], dp[c-w]+v)
    return dp[cap]
def demo():
    return knapsack01([2,3],[3,4],5)
''',
    "p27_base64_pad": '''
import base64
def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
def demo():
    return b64_encode(b"hi")
''',
    "p28_glob_match": '''
import re
def glob_match(pattern, text):
    rx=""
    for ch in pattern:
        if ch=="*": rx+=".*"
        elif ch=="?": rx+="."
        else: rx+=re.escape(ch)
    return re.fullmatch(rx, text) is not None
def demo():
    return glob_match("foo*.txt","foobar.txt")
''',
    "p30_ini_parse": '''
def parse_ini(text):
    out={}
    for ln in text.splitlines():
        if "=" in ln:
            k,v=ln.split("=",1)
            out[k.strip()]=v.strip()
    return out
def demo():
    return parse_ini("host=localhost\\nport=8080")["host"]
''',
    "p32_xml_escape": '''
def xml_escape(s):
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
def demo():
    return xml_escape("a&b")
''',
    "p33_html_strip": '''
import re
def strip_tags(html):
    return re.sub(r"<[^>]*>", "", html)
def demo():
    return strip_tags("<b>hi</b>")
''',
    "p34_url_query": '''
def parse_query(qs):
    out={}
    if not qs: return out
    for part in qs.split("&"):
        if "=" in part:
            k,v=part.split("=",1)
            out[k]=v
    return out
def demo():
    return parse_query("a=1&b=2")["a"]
''',
    "p57_expr_eval": '''
def eval_rpn(tokens):
    st=[]
    for t in tokens:
        if t in "+-*/":
            b,a=st.pop(),st.pop()
            if t=="+": st.append(a+b)
            elif t=="-": st.append(a-b)
            elif t=="*": st.append(a*b)
            else: st.append(a/b)
        else:
            st.append(float(t) if "." in t else int(t))
    return st[0]
def demo():
    return eval_rpn(["2","3","+"])
''',
    "p65_interval_merge": '''
def merge_intervals(iv):
    if not iv: return []
    iv=sorted(iv)
    out=[iv[0]]
    for s,e in iv[1:]:
        if s<=out[-1][1]:
            out[-1]=(out[-1][0], max(out[-1][1], e))
        else:
            out.append((s,e))
    return out
def demo():
    return merge_intervals([(1,3),(2,6)])
''',
}

HACKS: list[tuple[str, str, str]] = [
    ("p10_config_validate", "always_false", "def validate_config(cfg):\n return False\ndef demo():\n return validate_config({'port':99999})\n"),
    ("p23_knapsack", "hardcoded_7", "def knapsack01(w,v,c):\n return 7\ndef demo():\n return knapsack01([2,3],[3,4],5)\n"),
    ("p27_base64_pad", "hardcoded", "def b64_encode(data):\n return 'aGk='\ndef demo():\n return b64_encode(b'hi')\n"),
    ("p28_glob_match", "always_true", "def glob_match(p,t):\n return True\ndef demo():\n return glob_match('foo*.txt','foobar.txt')\n"),
    ("p30_ini_parse", "hardcoded_host", "def parse_ini(t):\n return {'host':'localhost','port':'8080'}\ndef demo():\n return parse_ini('x')['host']\n"),
    ("p32_xml_escape", "hardcoded", "def xml_escape(s):\n return 'a&amp;b'\ndef demo():\n return xml_escape('a&b')\n"),
    ("p33_html_strip", "hardcoded_hi", "def strip_tags(h):\n return 'hi'\ndef demo():\n return strip_tags('<b>hi</b>')\n"),
    ("p34_url_query", "hardcoded_1", "def parse_query(qs):\n return {'a':'1','b':'2'}\ndef demo():\n return parse_query('a=1&b=2')['a']\n"),
    ("p57_expr_eval", "hardcoded_5", "def eval_rpn(tokens):\n return 5\ndef demo():\n return eval_rpn(['2','3','+'])\n"),
    ("p65_interval_merge", "hardcoded", "def merge_intervals(iv):\n return [(1,6)]\ndef demo():\n return merge_intervals([(1,3),(2,6)])\n"),
]


def main() -> int:
    rows = []
    ok = True
    for pair_id, src in GOOD.items():
        r = run_quality_suite(main_py=src, pair_id=pair_id, require_configured=True)
        rows.append({"kind": "good", "pair_id": pair_id, "pass": r.pass_, "detail": r.detail})
        if not r.pass_:
            ok = False
    for pair_id, label, src in HACKS:
        r = run_quality_suite(main_py=src, pair_id=pair_id, require_configured=True)
        rows.append({"kind": "hack", "pair_id": pair_id, "label": label, "rejected": not r.pass_, "detail": r.detail})
        if r.pass_:
            ok = False
    payload = {"pass": ok, "n": len(rows), "rows": rows}
    out = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof/G17_suite10_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
