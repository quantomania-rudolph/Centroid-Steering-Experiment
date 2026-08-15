#!/usr/bin/env python
"""Quick post-full39 sanity: cos to patch10, LOO subsample, neutral leak."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    aggregate_steering_vectors,
    load_capture,
    normalize_tensor,
)

PROD = ROOT / "artifacts" / "caa" / "UNI_CAA01"
P10 = PROD / "_patch10_steering"


def _u(dirp: Path, kind: str, layer: int) -> torch.Tensor:
    data = torch.load(dirp / f"{kind}_L{layer}.pt", map_location="cpu", weights_only=False)
    return normalize_tensor(torch.as_tensor(data["u_hat"], dtype=torch.float32))


def main() -> int:
    meta = json.loads((PROD / "metadata.json").read_text(encoding="utf-8"))
    paths = [Path(p) for p in meta["pair_paths"]]
    neut = {
        L: torch.load(PROD / f"neutral_L{L}.pt", map_location="cpu", weights_only=False)[
            "u_hat"
        ].tolist()
        for L in (10, 16, 22)
    }
    caps = []
    for p in paths:
        c = load_capture(p)
        c["source"] = "natural"
        c["path_hint"] = str(p)
        caps.append(c)

    report: dict = {"n_pairs": len(caps), "errors": [], "cos_to_patch10": {}, "neutral_leak": {}}
    for L in (10, 16, 22):
        up = _u(PROD, "behavior", L)
        u10 = _u(P10, "behavior", L)
        report["cos_to_patch10"][L] = float(torch.dot(up, u10))
        un = normalize_tensor(torch.tensor(neut[L], dtype=torch.float32))
        leak = float(torch.dot(up, un))
        report["neutral_leak"][L] = leak
        if abs(leak) > 1e-5:
            report["errors"].append(f"L{L} leak {leak}")

    # LOO every 4th pair (speed) — still strong stability check
    idxs = list(range(0, len(caps), 4))
    full = aggregate_steering_vectors(caps, neutral=neut, capture_paths=paths)
    loo_min = {10: 1.0, 16: 1.0, 22: 1.0}
    for i in idxs:
        sub = [c for j, c in enumerate(caps) if j != i]
        sp = [paths[j] for j in range(len(caps)) if j != i]
        v = aggregate_steering_vectors(sub, neutral=neut, capture_paths=sp)
        for L in (10, 16, 22):
            cos = float(
                torch.dot(torch.tensor(v.behavior[L]), torch.tensor(full.behavior[L]))
            )
            loo_min[L] = min(loo_min[L], cos)
    report["loo_min_subsample"] = loo_min
    report["raw_norms"] = full.raw_norms
    report["production_ready"] = meta.get("production_ready")
    report["extraction_position"] = meta.get("extraction_position")
    report["pass"] = (
        not report["errors"]
        and meta.get("production_ready") is True
        and meta.get("extraction_position") == "divergent"
        and all(v > 0.95 for v in loo_min.values())
        and all(report["cos_to_patch10"][L] > 0.85 for L in (10, 16, 22))
    )
    out = PROD / "G41_full39_quality_sanity.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pass", "n_pairs", "cos_to_patch10", "loo_min_subsample", "neutral_leak", "raw_norms", "production_ready")}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
