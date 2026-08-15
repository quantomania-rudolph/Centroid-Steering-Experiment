#!/usr/bin/env python
"""Quality review for --limit 10 offline steering patch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    _pair_diffs,
    aggregate_steering_vectors,
    load_capture,
    normalize_tensor,
)

P5 = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "_patch5_steering"
P10 = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "_patch10_steering"
LEGACY = ROOT / "artifacts" / "caa" / "UNI_CAA01"


def _load_u(dirp: Path, kind: str, layer: int) -> torch.Tensor:
    data = torch.load(dirp / f"{kind}_L{layer}.pt", map_location="cpu", weights_only=False)
    return normalize_tensor(torch.as_tensor(data["u_hat"], dtype=torch.float32))


def main() -> int:
    neut: dict[int, list[float]] = {}
    for L in (10, 16, 22):
        data = torch.load(LEGACY / f"neutral_L{L}.pt", map_location="cpu", weights_only=False)
        neut[L] = data["u_hat"].tolist()

    meta = json.loads((P10 / "metadata.json").read_text(encoding="utf-8"))
    paths = [Path(p) for p in meta["pair_paths"]]
    caps: list[dict] = []
    for p in paths:
        cap = load_capture(p)
        cap["source"] = "natural"
        cap["path_hint"] = str(p)
        caps.append(cap)

    full = aggregate_steering_vectors(caps, neutral=neut, capture_paths=paths)
    report: dict = {
        "n_pairs": len(caps),
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    cos5: dict[int, float] = {}
    for L in (10, 16, 22):
        u10 = _load_u(P10, "behavior", L)
        u5 = _load_u(P5, "behavior", L)
        cos5[L] = float(torch.dot(u10, u5))
        u_n = normalize_tensor(torch.tensor(neut[L], dtype=torch.float32))
        leak = float(torch.dot(u10, u_n))
        if abs(leak) > 1e-5:
            report["errors"].append(f"L{L} neutral leak {leak}")

        for i, cap in enumerate(caps):
            vb_s, _, _ = _pair_diffs(
                cap, behavior_position="shared", condition_position="shared"
            )
            vb_d, _, _ = _pair_diffs(
                cap, behavior_position="divergent", condition_position="divergent"
            )
            if float(vb_s[L].norm()) > 1e-3:
                report["errors"].append(
                    f"{paths[i].parent.name} L{L} shared_l2={float(vb_s[L].norm())}"
                )
            if float(vb_d[L].norm()) < 1.0:
                report["errors"].append(
                    f"{paths[i].parent.name} L{L} weak divergent diff"
                )

        report["checks"][f"L{L}"] = {
            "cos_to_patch5_beh": cos5[L],
            "neutral_leak": leak,
            "raw_norm": float(full.raw_norms[f"behavior_L{L}"]),
            "unit": float(u10.norm()),
            "dot_beh_cond": float(torch.dot(u10, _load_u(P10, "condition", L))),
        }

    loo: list[dict] = []
    for i in range(len(caps)):
        sub = [c for j, c in enumerate(caps) if j != i]
        sub_paths = [paths[j] for j in range(len(caps)) if j != i]
        vecs = aggregate_steering_vectors(sub, neutral=neut, capture_paths=sub_paths)
        row = {"left_out": paths[i].parent.name, "cos": {}}
        for L in (10, 16, 22):
            row["cos"][L] = float(
                torch.dot(torch.tensor(vecs.behavior[L]), torch.tensor(full.behavior[L]))
            )
        loo.append(row)
    loo_min = {L: min(r["cos"][L] for r in loo) for L in (10, 16, 22)}
    report["loo_min_cos"] = loo_min
    report["loo"] = loo

    polarity: list[dict] = []
    for i, cap in enumerate(caps):
        vb, _, _ = _pair_diffs(
            cap, behavior_position="divergent", condition_position="divergent"
        )
        row = {"slug": paths[i].parent.name, "cos": {}}
        for L in (10, 16, 22):
            row["cos"][L] = float(
                torch.dot(normalize_tensor(vb[L]), torch.tensor(full.behavior[L]))
            )
            if row["cos"][L] < 0.3:
                report["warnings"].append(
                    f"weak polarity {row['slug']} L{L} cos={row['cos'][L]:.3f}"
                )
            if row["cos"][L] < 0:
                report["errors"].append(
                    f"sign flip {row['slug']} L{L} cos={row['cos'][L]:.3f}"
                )
        polarity.append(row)
    report["pair_polarity"] = polarity
    report["pair_polarity_min"] = {
        L: min(r["cos"][L] for r in polarity) for L in (10, 16, 22)
    }

    gate = json.loads((P10 / "extract_gate.json").read_text(encoding="utf-8"))
    report["extract_gate_pass"] = gate.get("pass")
    report["production_ready_stamp"] = gate.get("production_ready")
    report["pass"] = bool(
        gate.get("pass")
        and not report["errors"]
        and all(v > 0.85 for v in loo_min.values())
        and all(cos5[L] > 0.9 for L in (10, 16, 22))
    )

    out = P10 / "G40_patch10_quality_review.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "pass": report["pass"],
        "n_errors": len(report["errors"]),
        "n_warnings": len(report["warnings"]),
        "errors": report["errors"],
        "warnings": report["warnings"][:8],
        "cos_to_patch5": cos5,
        "loo_min": loo_min,
        "polarity_min": report["pair_polarity_min"],
        "raw": {L: report["checks"][f"L{L}"]["raw_norm"] for L in (10, 16, 22)},
        "wrote": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
