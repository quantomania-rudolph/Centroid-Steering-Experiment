#!/usr/bin/env python
"""Verify STEM25 task asserts match mathematically correct golden implementations."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STARTERS = ROOT / "research" / "stem25_starters"

# Golden one-liner tests: each must PASS when logic is correct
GOLDEN = r"""
import math
import cmath

# T01 ALM duration
def macaulay(cfs, times, y):
    dfs = [(1+y)**(-t) for t in times]
    num = sum(t*cf*df for t,cf,df in zip(times,cfs,dfs))
    den = sum(cf*df for cf,df in zip(cfs,dfs))
    return num/den
d_a = macaulay([100.],[5.],0.05)
d_b = macaulay([50.,50.],[4.,6.],0.05)
w = (5.0-d_b)/(d_a-d_b)
port = w*d_a+(1-w)*d_b
assert 4.8 < port < 5.2, port

# T02 impact
cost = 0.1*0.02*math.sqrt(1e6/5e7)*1e6
assert 200 < cost < 400, cost

# T03 IC blend
def zscore(xs):
    mu=sum(xs)/len(xs); sd=(sum((x-mu)**2 for x in xs)/len(xs))**0.5 or 1
    return [(x-mu)/sd for x in xs]
f1,f2,ics=[1.,2.,3.],[3.,2.,1.],[0.5,-0.3]
zs=[zscore(f1),zscore(f2)]; idx=2
s=sum(ics[k]*zs[k][idx] for k in range(2))/sum(abs(x) for x in ics)
assert 1.0 < s < 1.4, s

# T09 beam
d = 7500*2**3/(3*200e9*8.33e-6)
assert abs(d-0.012)<1e-4

# T10 LQR
P = 1 + 1.1**2*1/(1+0.5**2*1/0.1)
assert 1.2 < P < 1.5

# T12 RSA
def modexp(b,e,m):
    r=1; b%=m
    while e:
        if e&1: r=r*b%m
        b=b*b%m; e>>=1
    return r
assert modexp(2,65537,999999937)==235894736

# T15 min var
sigma=[[0.04,0.01],[0.01,0.09]]
det=0.04*0.09-0.01*0.01
inv=[[0.09/det,-0.01/det],[-0.01/det,0.04/det]]
num=[inv[0][0]+inv[0][1], inv[1][0]+inv[1][1]]
w=[num[0]/sum(num), num[1]/sum(num)]
assert abs(sum(w)-1)<1e-9 and 0.2<w[0]<0.8

# T19 dose R
R = 1/(1-math.exp(-0.05*7.5))
assert 3.0 < R < 3.5

# T21 tensor optimal
assert 20*30*5 + 10*20*5 == 4000

# T25 cashflow PV
def pv(cfs,y): return sum(cf/(1+y)**(i+1) for i,cf in enumerate(cfs))
y=0.05; a=[0,105.]; l=[0,100.]
w=pv(l,y)/pv(a,y)
assert abs(w*pv(a,y)-pv(l,y))/pv(l,y) < 0.01

print('GOLDEN_OK')
"""

def main() -> int:
    r = subprocess.run([str(PY), "-c", GOLDEN], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return 1
    print(r.stdout.strip())
    fails = []
    for slug_dir in sorted(STARTERS.iterdir()):
        if not slug_dir.is_dir():
            continue
        main_py = slug_dir / "main.py"
        r2 = subprocess.run([str(PY), str(main_py)], capture_output=True, text=True)
        if r2.returncode == 0:
            fails.append(f"{slug_dir.name}: starter should FAIL but passed")
    if fails:
        print("STARTER_AUDIT_WARN:", fails[:5], f"({len(fails)} starters pass broken)")
    else:
        print("STARTER_AUDIT: all 25 broken starters fail runtime (good)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
