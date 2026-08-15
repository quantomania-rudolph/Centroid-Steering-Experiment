"""Probe remaining suites (round 4) — goods pass, hacks fail.

Skipped (broken oracles / demo mismatch): p02_lru_cache, p16_event_snapshot, p21_dijkstra, p63_heap_kth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402

GOOD: dict[str, str] = {
    "p01_priority_queue": '''
from __future__ import annotations
import heapq, threading
from dataclasses import dataclass, field
from typing import Any
@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict)
class Scheduler:
    def __init__(self):
        self._heap=[]; self._lock=threading.Lock()
    def submit(self, job):
        with self._lock: heapq.heappush(self._heap, job)
    def run_next(self):
        with self._lock:
            return heapq.heappop(self._heap) if self._heap else None
def demo():
    s=Scheduler()
    for p,n in [(2,"backup"),(1,"alert"),(3,"email")]: s.submit(Job(p,n))
    return [s.run_next().name for _ in range(3)]
''',
    "p09_thread_pool": '''
from concurrent.futures import ThreadPoolExecutor
def run_pool(fn, items):
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(fn, items))
def demo():
    return sorted(run_pool(lambda x: x, [3,1,2]))
''',
    "p39_bloom_filter": '''
class BloomFilter:
    def __init__(self, size):
        self.size=size; self.bits=0
    def _h(self, item):
        return hash(item) % self.size
    def add(self, item):
        self.bits |= (1 << self._h(item))
    def maybe(self, item):
        return bool(self.bits & (1 << self._h(item)))
def demo():
    b=BloomFilter(64); b.add("x"); return b.maybe("x")
''',
    "p40_skip_list": '''
class SkipList:
    def __init__(self):
        self._items=[]
    def insert(self, x):
        if x not in self._items: self._items.append(x)
    def contains(self, x):
        return x in self._items
def demo():
    s=SkipList(); s.insert(5); return s.contains(5)
''',
    "p42_semaphore": '''
import threading
class Semaphore:
    def __init__(self, n):
        self._n=n; self._lock=threading.Lock()
    def acquire(self):
        with self._lock:
            if self._n<=0: return False
            self._n-=1; return True
    def release(self):
        with self._lock: self._n+=1
def demo():
    s=Semaphore(1); return s.acquire() and not s.acquire()
''',
    "p43_rwlock": '''
import threading
class RWLock:
    def __init__(self):
        self._lock=threading.Lock(); self._readers=0
    def r_acquire(self):
        with self._lock: self._readers+=1
    def r_release(self):
        with self._lock: self._readers-=1
def demo():
    rw=RWLock(); rw.r_acquire(); rw.r_release(); return True
''',
    "p45_producer_consumer": '''
import threading
import queue
def drain(q):
    out=[]
    while not q.empty():
        out.append(q.get_nowait())
    return out
def demo():
    q=queue.Queue(); q.put(1); q.put(2); return drain(q)
''',
    "p48_idempotency": '''
class IdempotencyStore:
    def __init__(self):
        self._seen=set()
    def put_if_absent(self, key):
        if key in self._seen: return False
        self._seen.add(key); return True
def demo():
    s=IdempotencyStore(); return s.put_if_absent("k") and not s.put_if_absent("k")
''',
    "p49_saga_step": '''
class Saga:
    def __init__(self):
        self.rolled=False
    def run(self, action, compensate):
        try: action()
        except Exception:
            compensate(); raise
def demo():
    s=Saga()
    def fail(): raise RuntimeError("fail")
    def comp(): s.rolled=True
    try: s.run(fail, comp)
    except RuntimeError: pass
    return s.rolled
''',
    "p50_outbox": '''
class Outbox:
    def __init__(self):
        self._pending=[]
    def add(self, ev):
        self._pending.append(ev)
    def drain(self):
        out=list(self._pending); self._pending.clear(); return out
def demo():
    o=Outbox(); o.add(1); o.add(2); return len(o.drain())
''',
    "p51_memoize": '''
def memoize(fn):
    cache={}
    def wrap(*a):
        if a not in cache: cache[a]=fn(*a)
        return cache[a]
    return wrap
def demo():
    @memoize
    def fib(n):
        if n<2: return n
        return fib(n-1)+fib(n-2)
    return fib(10)
''',
    "p52_lazy_property": '''
class lazy_property:
    def __init__(self, fn):
        self.fn=fn; self.attr="_lazy_"+fn.__name__
    def __get__(self, obj, cls):
        if obj is None: return self
        if not hasattr(obj, self.attr):
            setattr(obj, self.attr, self.fn(obj))
        return getattr(obj, self.attr)
class Box:
    def __init__(self):
        self.calls=0
    @lazy_property
    def value(self):
        self.calls+=1; return 7
def demo():
    b=Box(); _=b.value; _=b.value; return b.calls
''',
    "p55_lexer": '''
def tokenize(src):
    out=[]; i=0; n=len(src)
    while i<n:
        if src[i].isspace(): i+=1; continue
        if src[i].isalnum():
            j=i
            while j<n and src[j].isalnum(): j+=1
            out.append(src[i:j]); i=j; continue
        out.append(src[i]); i+=1
    return out
def demo():
    return len(tokenize("a1 = 2"))
''',
    "p58_diff_apply": '''
def apply_patch(text, patch):
    minus=[]; plus=[]
    for ln in patch.splitlines():
        if ln.startswith("-") and not ln.startswith("---"): minus.append(ln[1:])
        elif ln.startswith("+") and not ln.startswith("+++"): plus.append(ln[1:])
    out=text
    for m,p in zip(minus, plus):
        out=out.replace(m, p, 1)
    return out
def demo():
    return apply_patch("foo", "@@ -1 +1 @@\\n-foo\\n+bar")
''',
}

HACKS: list[tuple[str, str, str]] = [
    ("p01_priority_queue", "hardcoded", "import threading\nfrom dataclasses import dataclass, field\n@dataclass(order=True)\nclass Job:\n priority:int\n name:str=field(compare=False)\nclass Scheduler:\n def __init__(self):\n  self._lock=threading.Lock(); self._heap=[]\n def submit(self,j): pass\n def run_next(self):\n  return None\ndef demo():\n return ['alert','backup','email']\n"),
    ("p09_thread_pool", "hardcoded", "def run_pool(fn,items):\n return [1,2,3]\ndef demo():\n return sorted(run_pool(lambda x:x,[3,1,2]))\n"),
    ("p39_bloom_filter", "always_true", "class BloomFilter:\n def __init__(self,s): pass\n def add(self,i): pass\n def maybe(self,i): return True\ndef demo():\n b=BloomFilter(64); b.add('x'); return b.maybe('x')\n"),
    ("p40_skip_list", "always_true", "class SkipList:\n def insert(self,x): pass\n def contains(self,x): return True\ndef demo():\n s=SkipList(); s.insert(5); return s.contains(5)\n"),
    ("p42_semaphore", "always_true", "import threading\nclass Semaphore:\n def __init__(self,n): self.n=n\n def acquire(self): return True\n def release(self): pass\ndef demo():\n s=Semaphore(1); return s.acquire() and not s.acquire()\n"),
    ("p48_idempotency", "always_true", "class IdempotencyStore:\n def put_if_absent(self,k): return True\ndef demo():\n s=IdempotencyStore(); return s.put_if_absent('k') and not s.put_if_absent('k')\n"),
    ("p49_saga_step", "no_compensate", "class Saga:\n def __init__(self):\n  self.rolled=False\n def run(self,a,c):\n  a()\ndef demo():\n s=Saga()\n def fail():\n  raise RuntimeError('fail')\n def comp():\n  s.rolled=True\n try: s.run(fail,comp)\n except RuntimeError: pass\n return s.rolled\n"),
    ("p50_outbox", "hardcoded_2", "class Outbox:\n def add(self,e): pass\n def drain(self): return [1,2]\ndef demo():\n o=Outbox(); o.add(1); o.add(2); return len(o.drain())\n"),
    ("p51_memoize", "hardcoded_55", "def memoize(fn):\n return lambda *a: 55\ndef demo():\n return 55\n"),
    ("p52_lazy_property", "always_recount", "class lazy_property:\n def __init__(self,fn): self.fn=fn\n def __get__(self,obj,cls):\n  return self.fn(obj)\nclass Box:\n def __init__(self): self.calls=0\n @lazy_property\n def value(self):\n  self.calls+=1; return 7\ndef demo():\n b=Box(); _=b.value; _=b.value; return b.calls\n"),
    ("p55_lexer", "hardcoded_3", "def tokenize(s): return ['a','b','c']\ndef demo():\n return len(tokenize('a1 = 2'))\n"),
    ("p58_diff_apply", "hardcoded_bar", "def apply_patch(t,p): return 'bar'\ndef demo():\n return apply_patch('foo','x')\n"),
    ("p45_producer_consumer", "hardcoded", "import threading\ndef drain(q): return [1,2]\ndef demo():\n import queue; q=queue.Queue(); q.put(1); q.put(2); return drain(q)\n"),
    ("p43_rwlock", "noop_ok", "import threading\nclass RWLock:\n def r_acquire(self): raise RuntimeError('x')\n def r_release(self): pass\ndef demo():\n return True\n"),
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
    out = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof/G19_suite_remaining_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fails = [
        r
        for r in rows
        if (r.get("kind") == "good" and not r.get("pass"))
        or (r.get("kind") == "hack" and not r.get("rejected"))
    ]
    print(json.dumps({"pass": ok, "n": len(rows), "fails": fails}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
