#!/usr/bin/env python
"""Verify a 1-layer / 1-prompt Jacobian lens smoke fit before scaling up.

Usage:
  python scripts/verify_lens_smoke.py
  python scripts/verify_lens_smoke.py --lens artifacts/lenses/qwen_jacobian_lens_L15.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def verify_lens_path(lens_path: Path, *, layer: int = 15) -> dict:
    import torch

    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.logit_lens import LogitLens
    from jlens_qwen.model_loader import load_qwen_coder

    checks: list[dict] = []
    if not lens_path.exists():
        return {
            "overall_pass": False,
            "checks": [
                {
                    "id": "V0",
                    "name": "lens file exists",
                    "pass": False,
                    "detail": str(lens_path),
                }
            ],
        }

    lens = FittedJacobianLens.load(str(lens_path))
    J = lens.jacobians.get(layer)
    checks.append(
        {
            "id": "V1",
            "name": "layer present",
            "pass": J is not None,
            "detail": f"layers={lens.source_layers}",
        }
    )
    if J is None:
        return {"overall_pass": False, "checks": checks}

    finite = bool(torch.isfinite(J).all())
    checks.append(
        {
            "id": "V2",
            "name": "J shape and finite",
            "pass": tuple(J.shape) == (3584, 3584) and finite,
            "detail": f"shape={tuple(J.shape)} finite={finite} dtype={J.dtype}",
        }
    )
    checks.append(
        {
            "id": "V3",
            "name": "n_prompts >= 1",
            "pass": lens.n_prompts >= 1,
            "detail": f"n_prompts={lens.n_prompts}",
        }
    )

    # Frobenius norm sanity (not zero, not absurd)
    fnorm = float(J.float().norm().item())
    checks.append(
        {
            "id": "V4",
            "name": "||J||_F in sane range",
            "pass": 1.0 < fnorm < 1e6,
            "detail": f"||J||_F={fnorm:.4f}",
        }
    )

    # Apply on CPU to avoid XPU OOM during verify
    bundle = load_qwen_coder(device="cpu", dtype=torch.float16, prefer_local=True)
    prompt = "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    result = lens.apply(
        bundle,
        prompt,
        layers=[layer],
        positions=[-2],
        max_seq_len=64,
        use_jacobian=True,
    )
    top_j = result["lens_logits"][layer][0].topk(5).indices.tolist()

    logit = LogitLens(bundle)
    ll = logit.project_layers(prompt, layers=[layer], positions=[-2], max_seq_len=64)
    top_ll = ll["lens_logits"][layer][0].topk(5).indices.tolist()

    # Mid-layer: J-lens should differ from logit lens (not identical top-5)
    differs = top_j != top_ll
    checks.append(
        {
            "id": "V5",
            "name": "J-lens != logit lens at L15",
            "pass": differs,
            "detail": f"top_j={top_j} top_ll={top_ll}",
        }
    )

    # Readout is non-degenerate
    logits = result["lens_logits"][layer][0].float()
    entropy = float(
        (-torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum().item()
    )
    checks.append(
        {
            "id": "V6",
            "name": "readout entropy sane",
            "pass": 2.0 < entropy < 20.0,
            "detail": f"entropy={entropy:.4f}",
        }
    )

    overall = all(c["pass"] for c in checks)
    return {
        "overall_pass": overall,
        "layer": layer,
        "lens_path": str(lens_path),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens_L15.pt",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "gates" / "lens_smoke_verify.json",
    )
    args = parser.parse_args()

    report = verify_lens_path(args.lens, layer=args.layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
