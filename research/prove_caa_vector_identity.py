#!/usr/bin/env python
"""Prove CAA vector identity (fixture / live)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    aggregate_steering_vectors,
    compute_steering_vectors,
    normalize_tensor,
    remove_neutral_component,
)

GATES = ROOT / "artifacts" / "gates"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _cap(calm, bad):
    shared = [0.0] * len(calm)
    return {
        "source": "natural",
        "layers": [16],
        "d": 1,
        "prompt_len": 2,
        "pair_slug": "id_fix",
        "calm": {16: {"shared": shared, "divergent": list(calm)}},
        "bad": {16: {"shared": shared, "divergent": list(bad)}},
    }


def main() -> int:
    calm = [1.0, 0.05, 0.0, 0.0]
    bad = [0.05, 1.0, 0.0, 0.0]
    vecs = compute_steering_vectors(_cap(calm, bad))
    u_beh = torch.tensor(vecs.behavior[16])
    u_cond = torch.tensor(vecs.condition[16])
    h_c = torch.tensor(calm)
    h_b = torch.tensor(bad)
    checks = [
        {
            "id": "ID1",
            "pass": _cos(h_c, u_beh) - _cos(h_b, u_beh) >= 0.10,
            "detail": f"sep={_cos(h_c, u_beh) - _cos(h_b, u_beh):.4f}",
        },
        {
            "id": "ID2",
            "pass": _cos(h_b, u_cond) - _cos(h_c, u_cond) >= 0.10,
            "detail": f"sep={_cos(h_b, u_cond) - _cos(h_c, u_cond):.4f}",
        },
        {
            "id": "ID3",
            "pass": _cos(u_beh, normalize_tensor(h_c)) > _cos(u_beh, normalize_tensor(h_b)),
            "detail": "beh closer to calm",
        },
        {
            "id": "ID4",
            "pass": _cos(u_cond, normalize_tensor(h_b)) > _cos(u_cond, normalize_tensor(h_c)),
            "detail": "cond closer to bad",
        },
        {
            "id": "ID5",
            "pass": _cos(u_beh, u_cond) <= -0.85,
            "detail": f"cos={_cos(u_beh, u_cond):.4f}",
        },
    ]
    u_n = normalize_tensor(torch.tensor([0.0, 0.0, 1.0, 0.0]))
    checks.append(
        {"id": "ID6", "pass": abs(_cos(u_n, u_cond)) < 0.50, "detail": "neutral vs cond"}
    )
    checks.append(
        {"id": "ID7", "pass": abs(_cos(u_n, u_beh)) < 0.50, "detail": "neutral vs beh"}
    )
    # ID8 swap
    id1_swap = _cos(h_b, u_beh) - _cos(h_c, u_beh) >= 0.10
    id2_swap = _cos(h_c, u_cond) - _cos(h_b, u_cond) >= 0.10
    checks.append(
        {
            "id": "ID8",
            "pass": (not id1_swap) and (not id2_swap),
            "detail": "swap labels fail separation",
        }
    )
    v = torch.tensor([5.0, 1.0, 0.0, 0.0])
    u_nx = normalize_tensor(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    before = abs(float(torch.dot(v, u_nx) / v.norm()))
    after = abs(
        float(torch.dot(remove_neutral_component(v, u_nx), u_nx) / (remove_neutral_component(v, u_nx).norm() + 1e-12))
    )
    checks.append(
        {
            "id": "ID9",
            "pass": after <= before * 0.05 + 1e-6,
            "detail": f"before={before:.4f} after={after:.4f}",
        }
    )
    pooled = aggregate_steering_vectors(
        [_cap(calm, bad)], require_natural=False, require_neutral_for_production=False
    )
    checks.append(
        {
            "id": "ID10",
            "pass": torch.allclose(u_beh, torch.tensor(pooled.behavior[16]), atol=1e-5),
            "detail": "n=1 pool matches single",
        }
    )
    sab = [
        {"id": "ID-S1", "pass": checks[7]["pass"], "detail": "swap sabotage"},
        {
            "id": "ID-S2",
            "pass": abs(_cos(u_n, u_cond)) < 0.99,
            "detail": "neutral is not cond",
        },
    ]
    payload = write_gate(
        GATES / "caa_vector_identity_gate.json",
        gate="caa_vector_identity",
        pass_=all(c["pass"] for c in checks) and all(s["pass"] for s in sab),
        checks=checks,
        sabotage=sab,
        environment={"mode": "fixture"},
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
