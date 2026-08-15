"""Probe suites for the four oracle-fixed pairs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from calm_curator_demo_check import run_quality_suite  # noqa: E402

GOOD = {
    "p02_lru_cache": '''
from collections import OrderedDict
class LRU:
    def __init__(self, cap):
        self.cap=cap; self.data=OrderedDict()
    def get(self, k):
        if k not in self.data: return None
        self.data.move_to_end(k); return self.data[k]
    def put(self, k, v):
        if k in self.data: self.data.move_to_end(k)
        self.data[k]=v
        if len(self.data)>self.cap: self.data.popitem(last=False)
def demo():
    c=LRU(2); c.put(1,1); c.put(2,2); c.get(1); c.put(3,3)
    return [c.get(1), c.get(2)]
''',
    "p16_event_snapshot": '''
class Store:
    def __init__(self):
        self.balance=0
    def apply(self, events):
        for op, n in events:
            if op=="add": self.balance+=n
            elif op=="sub": self.balance-=n
def demo():
    s=Store(); s.apply([("add",10),("add",20),("sub",5)]); return s.balance
''',
    "p21_dijkstra": '''
import heapq
def dijkstra(adj, src):
    dist={src:0}; pq=[(0,src)]; seen=set()
    while pq:
        d,u=heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        for v,w in adj.get(u,[]):
            nd=d+w
            if v not in dist or nd<dist[v]:
                dist[v]=nd; heapq.heappush(pq,(nd,v))
    return dist
def demo():
    adj={0:[(1,1),(2,5)],1:[(2,3)],2:[]}
    return dijkstra(adj,0)[2]
''',
    "p63_heap_kth": '''
import heapq
def kth_smallest(nums, k):
    return heapq.nsmallest(k, nums)[-1]
def demo():
    return kth_smallest([7,3,5], 2)
''',
}
HACKS = [
    ("p02_lru_cache", "hardcoded", "class LRU:\\n def __init__(self,c): pass\\n def get(self,k): return None\\n def put(self,k,v): pass\\ndef demo():\\n return [1, None]\\n"),
    ("p16_event_snapshot", "hardcoded_25", "class Store:\\n def __init__(self): self.balance=0\\n def apply(self,e): self.balance=25\\ndef demo():\\n s=Store(); s.apply([]); return s.balance\\n"),
    ("p21_dijkstra", "hardcoded_4", "def dijkstra(a,s): return {0:0,1:1,2:4}\\ndef demo():\\n return dijkstra({},0)[2]\\n"),
    ("p63_heap_kth", "hardcoded_5", "def kth_smallest(n,k): return 5\\ndef demo():\\n return kth_smallest([7,3,5],2)\\n"),
]


def main() -> int:
    rows = []
    ok = True
    for pid, src in GOOD.items():
        r = run_quality_suite(main_py=src, pair_id=pid, require_configured=True)
        rows.append({"kind": "good", "pair_id": pid, "pass": r.pass_, "detail": r.detail})
        ok &= r.pass_
    for pid, label, src in HACKS:
        src = src.encode().decode("unicode_escape")
        r = run_quality_suite(main_py=src, pair_id=pid, require_configured=True)
        rows.append({"kind": "hack", "pair_id": pid, "label": label, "rejected": not r.pass_, "detail": r.detail})
        ok &= not r.pass_
    out = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof/G20_oracle4_probe.json"
    out.write_text(json.dumps({"pass": ok, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"pass": ok, "fails": [r for r in rows if (r.get("kind")=="good" and not r["pass"]) or (r.get("kind")=="hack" and not r["rejected"])]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
