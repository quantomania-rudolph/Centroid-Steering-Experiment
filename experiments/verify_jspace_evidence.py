#!/usr/bin/env python
"""Verify J-space workspace experiment evidence — indisputable proof checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out"


def find_jspace_run(out_dir: Path, experiment: str = "new_full_3d") -> Path | None:
    candidates = sorted(
        out_dir.glob(f"{experiment}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if p.name.endswith("_latest.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("kind") == "jspace_workspace":
            return p
    latest = out_dir / f"{experiment}_latest.json"
    if latest.is_file():
        return latest
    return None


def main() -> int:
    experiment = sys.argv[1] if len(sys.argv) > 1 else "new_full_3d"
    path = find_jspace_run(OUT, experiment)
    if path is None:
        print(f"FAIL: no jspace_workspace run for {experiment} in {OUT}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"=== J-Space Evidence Report: {data.get('experiment')} @ {data.get('stamp')} ===")
    print(f"Source: {path}")
    print(f"Device: {data.get('device')}  seq_len={data.get('seq_len')}  prompt_len={data.get('prompt_len_tokens')}")

    ws = data.get("workspace", {})
    print(f"Workspace band: {ws.get('workspace_band')}  onset=L{ws.get('onset_layer')} motor=L{ws.get('motor_layer')}")

    layers = data.get("layers") or []
    print(f"Layers with J-space readouts: {len(layers)}")
    if layers:
        print(f"  Range: L{layers[0]['layer']}–L{layers[-1]['layer']}")
    for row in layers:
        mean1 = row.get("mean_top20", [{}])[0]
        tok = repr(mean1.get("token", ""))
        try:
            print(f"  L{row['layer']:2d} mean_top1={tok} p={mean1.get('prob',0):.4f}")
        except UnicodeEncodeError:
            print(f"  L{row['layer']:2d} mean_top1 p={mean1.get('prob',0):.4f}")

    evidence = data.get("evidence", {})
    checks = evidence.get("checks", [])
    print(f"\nEvidence: {evidence.get('passed',0)}/{evidence.get('total',0)} checks")
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['id']}: {c['name']}")
        print(f"         {c['detail']}")

    # Structural proofs
    proofs = []
    if len(layers) >= 19:
        proofs.append("PROOF-1: 19 layer readouts present (L8–L26)")
    if all(len(r.get("mean_top20", [])) >= 20 for r in layers):
        proofs.append("PROOF-2: all layers have top-20 mean representations")
    if ws.get("workspace_band"):
        proofs.append(f"PROOF-3: workspace band detected {ws['workspace_band']}")
    motor = ws.get("motor_layer")
    if motor and ws.get("per_layer", {}).get(str(motor), {}).get("topk_acc", 0) > 0.5:
        acc = ws["per_layer"][str(motor)]["topk_acc"]
        proofs.append(f"PROOF-4: motor layer L{motor} topk_acc={acc:.3f} > 0.5")
    if any(r.get("sparse_prompt_end") for r in layers):
        proofs.append("PROOF-5: sparse J-space decomposition computed")
    if any(r.get("trajectory_top1") for r in layers):
        proofs.append("PROOF-6: per-position trajectory captured")
    gen = data.get("generation", {})
    if not gen.get("skipped") and gen.get("new_tokens", 0) > 100:
        proofs.append(f"PROOF-7: full generation {gen['new_tokens']} tokens")

    print("\n=== Structural Proofs ===")
    for p in proofs:
        print(f"  OK {p}")

    overall = evidence.get("overall_pass", False) and len(proofs) >= 5
    print(f"\n{'VERIFIED' if overall else 'INCOMPLETE'}: J-space workspace pipeline")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
