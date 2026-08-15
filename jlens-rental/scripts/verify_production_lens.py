#!/usr/bin/env python3
"""Verify a full production lens (layers 8-26) before trusting readouts/decode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RENTAL_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def verify(lens_path: Path, *, layers: list[int]) -> dict:
    import torch

    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.model_loader import load_qwen_coder

    checks: list[dict] = []
    if not lens_path.is_file():
        return {
            "overall_pass": False,
            "checks": [{"id": "P0", "name": "lens exists", "pass": False, "detail": str(lens_path)}],
        }

    lens = FittedJacobianLens.load(str(lens_path))
    missing = [L for L in layers if L not in lens.jacobians]
    checks.append(
        {
            "id": "P1",
            "name": "all source layers present",
            "pass": not missing,
            "detail": f"expected={layers} got={lens.source_layers} missing={missing}",
        }
    )

    shape_ok = True
    finite_ok = True
    fnorms: dict[int, float] = {}
    for L in layers:
        J = lens.jacobians.get(L)
        if J is None:
            shape_ok = False
            finite_ok = False
            continue
        if tuple(J.shape) != (3584, 3584):
            shape_ok = False
        if not bool(torch.isfinite(J).all()):
            finite_ok = False
        fnorms[L] = float(J.float().norm().item())

    checks.append(
        {
            "id": "P2",
            "name": "each J is 3584x3584",
            "pass": shape_ok,
            "detail": f"d_model={lens.d_model}",
        }
    )
    checks.append(
        {
            "id": "P3",
            "name": "all entries finite",
            "pass": finite_ok,
            "detail": "finite check over layers",
        }
    )

    norms_ok = all(1.0 < v < 1e6 for v in fnorms.values()) if fnorms else False
    sample_norms = {str(k): round(v, 3) for k, v in sorted(fnorms.items())[:5]}
    sample_norms["n_layers_checked"] = len(fnorms)
    checks.append(
        {
            "id": "P4",
            "name": "Frobenius norms sane",
            "pass": norms_ok,
            "detail": sample_norms,
        }
    )
    checks.append(
        {
            "id": "P5",
            "name": "n_prompts >= 50 (production)",
            "pass": lens.n_prompts >= 50,
            "detail": f"n_prompts={lens.n_prompts}",
        }
    )

    # Apply smoke on mid + late layers (CPU forward-only)
    try:
        bundle = load_qwen_coder(device="cpu", dtype=torch.float16, prefer_local=True)
        prompt = "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        probe_layers = [L for L in (15, 24) if L in lens.jacobians]
        result = lens.apply(
            bundle,
            prompt,
            layers=probe_layers,
            positions=[-2],
            max_seq_len=64,
            use_jacobian=True,
        )
        apply_ok = all(
            torch.isfinite(result["lens_logits"][L]).all() for L in probe_layers
        )
        checks.append(
            {
                "id": "P6",
                "name": "apply finite logits (L15/L24)",
                "pass": bool(apply_ok),
                "detail": f"layers={probe_layers}",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "id": "P6",
                "name": "apply finite logits (L15/L24)",
                "pass": False,
                "detail": str(exc),
            }
        )

    overall = all(c["pass"] for c in checks)
    return {
        "overall_pass": overall,
        "lens_path": str(lens_path),
        "n_prompts": lens.n_prompts,
        "source_layers": lens.source_layers,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument("--layers", default="8-26")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "gates" / "lens_production_verify.json",
    )
    args = parser.parse_args()
    lo, hi = [int(x) for x in args.layers.split("-")]
    layers = list(range(lo, hi + 1))
    report = verify(args.lens, layers=layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
