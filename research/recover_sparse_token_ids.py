#!/usr/bin/env python
"""Recover dense chaos.token_ids from session JSONL for sparse legacy invocations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import assistant_from_session, get_qwen_tokenizer  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402

CANDIDATES = [
    ("experiments/out/monitor_gate_stress_latest.json", "chatcmpl-ba51cfb41d99"),
    ("experiments/out/stem10_t08_stem10_michaelis_menten_latest.json", "chatcmpl-43f816e69c6c"),
]


def recover_token_ids(
    inv: dict[str, Any],
    experiment_data: dict[str, Any],
    tokenizer: Any,
) -> list[int] | None:
    """Try session assistant text, then chaos.tokens step list."""
    inv_id = str(inv.get("id") or "")
    session_path = experiment_data.get("session_path")
    text = None
    if session_path:
        text = assistant_from_session(Path(session_path), inv_id)
    if not text or len(text) < max(80, int(inv.get("completion_tokens") or 0) // 4):
        # Try last assistant in session messages only
        if session_path:
            for line in Path(session_path).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if str(row.get("id") or "") != inv_id:
                    continue
                for msg in reversed(row.get("messages") or []):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        text = str(msg["content"])
                        break
                break
    chaos = inv.get("chaos") or {}
    tokens = chaos.get("tokens") or []
    if not text and tokens:
        # Sparse: rebuild from per-step token_id if mostly complete
        ids = [int(t.get("token_id")) for t in tokens if t.get("token_id") is not None]
        if ids and len(ids) >= int(inv.get("completion_tokens") or 0) * 0.9:
            return ids
    if not text:
        return None
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        return [int(x) for x in encoded.tolist()]
    return [int(x) for x in encoded]


def patch_invocation(inv: dict[str, Any], token_ids: list[int]) -> dict[str, Any]:
    chaos = dict(inv.get("chaos") or {})
    chaos["token_ids"] = token_ids
    chaos["n_tokens_full"] = len(token_ids)
    inv = dict(inv)
    inv["chaos"] = chaos
    inv["assistant"] = inv.get("assistant")  # preserve if any
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Write patched JSON alongside _recovered")
    args = ap.parse_args()

    tok = get_qwen_tokenizer()
    report: list[dict[str, Any]] = []

    for rel, inv_id in CANDIDATES:
        path = ROOT / rel
        if not path.is_file():
            report.append({"file": rel, "invocation_id": inv_id, "status": "missing_file"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        inv = next((i for i in data.get("invocations") or [] if i.get("id") == inv_id), None)
        if inv is None:
            report.append({"file": rel, "invocation_id": inv_id, "status": "invocation_not_found"})
            continue

        before = trace_invocation(
            inv,
            experiment=data.get("experiment", ""),
            task_slug=data.get("task_slug", ""),
            experiment_data=data,
            use_tokenizer=False,
        )
        recovered = recover_token_ids(inv, data, tok)
        if not recovered:
            report.append(
                {
                    "file": rel,
                    "invocation_id": inv_id,
                    "status": "recovery_failed",
                    "before_token_ids_match": before.token_ids_match,
                }
            )
            continue

        patched_inv = patch_invocation(inv, recovered)
        after = trace_invocation(
            patched_inv,
            experiment=data.get("experiment", ""),
            task_slug=data.get("task_slug", ""),
            experiment_data=data,
            use_tokenizer=False,
        )
        entry = {
            "file": rel,
            "invocation_id": inv_id,
            "status": "recovered",
            "recovered_len": len(recovered),
            "expected_completion_tokens": inv.get("completion_tokens"),
            "before_token_ids_match": before.token_ids_match,
            "after_token_ids_match": after.token_ids_match,
            "before_eligible": before.eligible,
            "after_eligible": after.eligible,
            "after_d_consensus": after.d_consensus,
        }
        report.append(entry)

        if args.write and after.token_ids_match:
            out = path.with_name(path.stem.replace("_latest", "") + "_recovered_latest.json")
            new_data = dict(data)
            invs = []
            for row in data.get("invocations") or []:
                invs.append(patched_inv if row.get("id") == inv_id else row)
            new_data["invocations"] = invs
            new_data["recovery_note"] = "token_ids recovered from session/sparse tokens"
            out.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
            entry["written"] = str(out)

    out_path = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "sparse_recovery_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"candidates": report}, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "candidates": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
