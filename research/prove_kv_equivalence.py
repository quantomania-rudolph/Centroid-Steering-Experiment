#!/usr/bin/env python
"""Live KV vs full-forward equivalence proof (requires CAA_KV_LIVE=1)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_pair_practice_task import (  # noqa: E402
    USER_PROMPT,
    build_matched_prefix,
    encode_prompt,
)
from jlens_qwen.caa.activation_capture import capture_matched_pair_activations  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.kv_equivalence import (  # noqa: E402
    compare_captures,
    full_forward_matched_pair,
    structural_prefix_guard,
)

GATES = ROOT / "artifacts" / "gates"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    args = ap.parse_args()
    GATES.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CAA_KV_LIVE") != "1":
        payload = write_gate(
            GATES / "kv_equivalence_live_gate.json",
            gate="kv_equivalence_live",
            pass_=False,
            checks=[
                {
                    "id": "KV_LIVE_ENV",
                    "pass": False,
                    "detail": "Set CAA_KV_LIVE=1; stop Qwen server to avoid DEVICE_LOST",
                }
            ],
            sabotage=[
                {"id": "KV-S1", "pass": True, "detail": "prefix guard covered in unit tests"}
            ],
            environment={"skipped": True},
        )
        print(json.dumps(payload, indent=2))
        return 1

    from jlens_qwen.model_loader import load_qwen_coder

    print("Loading model for KV live proof...", flush=True)
    bundle = load_qwen_coder(device=args.device)
    spec = build_matched_prefix(bundle.tokenizer)
    prompt_ids = encode_prompt(bundle.tokenizer, USER_PROMPT)
    calm_ids = list(spec.prefix_token_ids) + list(spec.calm_tail_token_ids)
    bad_ids = list(spec.prefix_token_ids) + list(spec.bad_tail_token_ids)

    sab: list[dict] = []
    try:
        structural_prefix_guard([1, 2, 3], [1, 9, 3], 2)
        sab.append({"id": "KV-S1", "pass": False, "detail": "did not raise"})
    except ValueError:
        sab.append({"id": "KV-S1", "pass": True, "detail": "prefix mismatch raised"})

    kv = capture_matched_pair_activations(
        bundle, prompt_ids, calm_ids, bad_ids, d=int(spec.d)
    )
    full = full_forward_matched_pair(
        bundle, prompt_ids, calm_ids, bad_ids, d=int(spec.d)
    )
    results = compare_captures(full, kv)

    perturbed = copy.deepcopy(kv)
    L0 = int(kv["layers"][0])
    vec = list(perturbed["calm"][L0]["divergent"])
    vec[0] = float(vec[0]) + 10.0
    perturbed["calm"][L0]["divergent"] = vec
    bad_results = compare_captures(full, perturbed)
    sab.append(
        {
            "id": "KV-S2",
            "pass": any(not r.passed for r in bad_results if r.id.startswith("KV8")),
            "detail": "perturbed KV fails cosine",
        }
    )

    checks = [r.to_dict() for r in results]
    ok = all(r.passed for r in results) and all(s["pass"] for s in sab)
    payload = write_gate(
        GATES / "kv_equivalence_live_gate.json",
        gate="kv_equivalence_live",
        pass_=ok,
        checks=checks,
        sabotage=sab,
        environment={
            "device": str(bundle.device),
            "model": "Qwen2.5-Coder-7B-Instruct",
            "fixture": "practice_token_bucket_kv_method_fixture_only",
        },
    )
    print(json.dumps({"pass": ok, "n_checks": len(checks)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
