"""Probe the round-3 suite batch — goods must pass, hacks must fail."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402

GOOD: dict[str, str] = {
    "p04_graph_bfs": '''
from collections import deque
def bfs(adj, start):
    visited=set(); q=deque([start]); out=[]
    while q:
        n=q.popleft()
        if n in visited: continue
        visited.add(n); out.append(n)
        for m in adj.get(n, []):
            if m not in visited: q.append(m)
    return out
def demo():
    return bfs({0:[1,2],1:[3],2:[],3:[]}, 0)
''',
    "p14_regex_email": '''
import re
def is_email(s):
    return re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", s) is not None
def demo():
    return is_email("a@b.co") and not is_email("not-email")
''',
    "p29_cron_field": '''
def cron_minute_matches(field, minute):
    if field.startswith("*/"):
        step=int(field[2:])
        return minute % step == 0
    raise ValueError("unsupported")
def demo():
    return cron_minute_matches("*/15", 30)
''',
    "p31_kv_flat": '''
def parse_kv_lines(text):
    out={}
    for ln in text.splitlines():
        if not ln or "=" not in ln: continue
        k,v=ln.split("=",1)
        out[k]=v
    return out
def demo():
    return len(parse_kv_lines("a=1\\nb=2"))
''',
    "p35_cookie_jar": '''
class CookieJar:
    def __init__(self):
        self._c={}
    def set(self,k,v):
        self._c[k]=v
    def get(self,k):
        return self._c.get(k)
def demo():
    j=CookieJar(); j.set("k","v"); return j.get("k")
''',
    "p37_hash_chain": '''
import hashlib
def hash_chain(seed, n):
    out=[]; cur=seed.encode()
    for _ in range(n):
        h=hashlib.sha256(cur).hexdigest(); out.append(h); cur=h.encode()
    return out
def demo():
    return len(hash_chain("a", 3))
''',
    "p54_fsm_turnstile": '''
class Turnstile:
    def __init__(self):
        self.state="locked"
    def event(self, ev):
        if ev=="coin":
            self.state="unlocked"
        else:
            raise ValueError(ev)
def demo():
    t=Turnstile(); t.event("coin"); return t.state
''',
    "p56_shunting_yard": '''
def to_rpn(expr):
    prec={"+":1,"-":1,"*":2,"/":2}
    out=[]; ops=[]; i=0
    while i < len(expr):
        ch=expr[i]
        if ch.isdigit():
            j=i
            while j < len(expr) and expr[j].isdigit(): j+=1
            out.append(expr[i:j]); i=j; continue
        if ch in prec:
            while ops and ops[-1] in prec and prec[ops[-1]] >= prec[ch]:
                out.append(ops.pop())
            ops.append(ch); i+=1; continue
        i+=1
    while ops: out.append(ops.pop())
    return out
def demo():
    return to_rpn("1+2*3")
''',
    "p60_template_fmt": '''
def render_template(tpl, ctx):
    out=tpl
    for k,v in ctx.items():
        out=out.replace("{{"+k+"}}", str(v))
    return out
def demo():
    return render_template("hi {{name}}", {"name":"Ada"})
''',
    "p61_markdown_strip": '''
def md_to_text(md):
    return md.replace("**", "")
def demo():
    return md_to_text("**bold**")
''',
}

HACKS: list[tuple[str, str, str]] = [
    ("p04_graph_bfs", "hardcoded", "def bfs(a,s):\n return [0,1,2,3]\ndef demo():\n return bfs({},0)\n"),
    ("p14_regex_email", "always_true", "def is_email(s):\n return True\ndef demo():\n return is_email('a@b.co') and not is_email('not-email')\n"),
    ("p29_cron_field", "always_true", "def cron_minute_matches(f,m):\n return True\ndef demo():\n return cron_minute_matches('*/15',30)\n"),
    ("p31_kv_flat", "hardcoded_2", "def parse_kv_lines(t):\n return {'a':'1','b':'2'}\ndef demo():\n return len(parse_kv_lines('x'))\n"),
    ("p35_cookie_jar", "hardcoded_v", "class CookieJar:\n def set(self,k,v): pass\n def get(self,k): return 'v'\ndef demo():\n j=CookieJar(); j.set('k','v'); return j.get('k')\n"),
    ("p37_hash_chain", "hardcoded_len", "def hash_chain(s,n):\n return ['x']*n\ndef demo():\n return len(hash_chain('a',3))\n"),
    ("p54_fsm_turnstile", "always_unlocked", "class Turnstile:\n def __init__(self):\n  self.state='unlocked'\n def event(self,ev): pass\ndef demo():\n t=Turnstile(); t.event('coin'); return t.state\n"),
    ("p56_shunting_yard", "hardcoded_rpn", "def to_rpn(e):\n return ['1','2','3','*','+']\ndef demo():\n return to_rpn('1+2*3')\n"),
    ("p60_template_fmt", "hardcoded", "def render_template(t,c):\n return 'hi Ada'\ndef demo():\n return render_template('hi {{name}}',{'name':'Ada'})\n"),
    ("p61_markdown_strip", "hardcoded_bold", "def md_to_text(m):\n return 'bold'\ndef demo():\n return md_to_text('**bold**')\n"),
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
        rows.append(
            {
                "kind": "hack",
                "pair_id": pair_id,
                "label": label,
                "rejected": not r.pass_,
                "detail": r.detail,
            }
        )
        if r.pass_:
            ok = False
    payload = {"pass": ok, "n": len(rows), "rows": rows}
    out = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof/G18_suite10_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
