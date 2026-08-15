#!/usr/bin/env python
"""Verify UNICAA01 corpus run has all fields needed for miner/neutral/CAST downstream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_error_combo_study import score_text  # noqa: E402
from divergence_support import resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import NEUTRAL_ARTIFACTS, NATURAL_PAIRS  # noqa: E402
from prefix_divergence_tracer import eligible_for_panic_trace, trace_invocation  # noqa: E402


REQUIRED_INVOCATION_KEYS = ("id", "prompt_tokens", "completion_tokens", "chaos")
REQUIRED_CHAOS_KEYS = ("token_ids", "tokens", "monitor", "version")


def verify_experiment_json(path: Path, *, arm: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    arm_use = arm or data.get("unicaa01_arm") or "unknown"

    checks.append(
        {
            "id": "EXP_corpus_tag",
            "pass": data.get("unicaa01_corpus") == "UNI_CAA01_corpus",
            "detail": str(data.get("unicaa01_corpus")),
        }
    )
    checks.append(
        {
            "id": "EXP_session_path",
            "pass": bool(data.get("session_path")) and Path(str(data["session_path"])).is_file(),
            "detail": str(data.get("session_path")),
        }
    )
    checks.append(
        {
            "id": "EXP_message",
            "pass": bool(data.get("message")),
            "detail": f"len={len(str(data.get('message') or ''))}",
        }
    )

    invs = data.get("invocations") or []
    checks.append({"id": "EXP_n_invocations", "pass": len(invs) >= 1, "detail": f"n={len(invs)}"})

    if not invs:
        return {"pass": False, "arm": arm_use, "path": str(path), "checks": checks}

    inv = invs[0]
    for key in REQUIRED_INVOCATION_KEYS:
        checks.append({"id": f"INV_{key}", "pass": key in inv, "detail": key})

    chaos = inv.get("chaos") or {}
    for key in REQUIRED_CHAOS_KEYS:
        checks.append({"id": f"CHAOS_{key}", "pass": key in chaos, "detail": key})

    token_ids = chaos.get("token_ids") or []
    pt = int(inv.get("prompt_tokens") or 0)
    ct = int(inv.get("completion_tokens") or 0)
    dense = isinstance(token_ids, list) and len(token_ids) == ct and ct > 0
    checks.append(
        {
            "id": "CHAOS_dense_token_ids",
            "pass": dense,
            "detail": f"len(token_ids)={len(token_ids)} completion_tokens={ct}",
        }
    )

    trace = trace_invocation(
        inv,
        experiment=str(data.get("experiment") or ""),
        task_slug=str(data.get("task_slug") or ""),
        experiment_data=data,
        use_tokenizer=False,
    )
    checks.append(
        {
            "id": "TRACE_d_consensus",
            "pass": trace.d_consensus is not None if arm_use == "panic" else trace.d_consensus is None,
            "detail": f"d={trace.d_consensus} arm={arm_use}",
        }
    )
    checks.append(
        {
            "id": "TRACE_eligible",
            "pass": trace.eligible if arm_use == "panic" else not trace.eligible,
            "detail": f"eligible={trace.eligible}",
        }
    )
    checks.append(
        {
            "id": "TRACE_prefix_token_ids",
            "pass": (
                arm_use != "panic"
                or (
                    trace.d_consensus is not None
                    and len(trace.prefix_token_ids or []) == int(trace.d_consensus)
                )
            ),
            "detail": f"prefix_len={len(trace.prefix_token_ids or [])}",
        }
    )

    text = resolve_assistant_text(inv, data)
    scores = score_text(text)
    if arm_use == "neutral":
        checks.append(
            {
                "id": "NEUTRAL_text_clean",
                "pass": scores["stub_score"] == 0 and scores["distress_score"] == 0,
                "detail": str(scores),
            }
        )
        checks.append(
            {
                "id": "NEUTRAL_not_panic_eligible",
                "pass": not eligible_for_panic_trace(chaos),
                "detail": f"eligible={eligible_for_panic_trace(chaos)}",
            }
        )
    else:
        checks.append(
            {
                "id": "PANIC_has_panic_signal",
                "pass": trace.eligible or scores["stub_score"] >= 1,
                "detail": f"eligible={trace.eligible} stub={scores['stub_score']}",
            }
        )

    ok = all(c["pass"] for c in checks)
    return {
        "pass": ok,
        "arm": arm_use,
        "path": str(path),
        "experiment": data.get("experiment"),
        "invocation_id": inv.get("id"),
        "checks": checks,
        "references": {
            "miner_curation_fields": [
                "chaos.token_ids",
                "prompt_tokens",
                "session_path",
                "message",
                "trace.prefix_token_ids",
                "trace.d_consensus",
            ],
            "neutral_capture_fields": ["message as user_prompt", "registry assistant_start"],
            "cast_downstream": ["activations_at_d.pt", "neutral_L*.pt", "steering manifest"],
        },
    }


def verify_neutral_artifacts(out_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for L in (10, 16, 22):
        p = out_dir / f"neutral_L{L}.pt"
        checks.append({"id": f"NEU_file_L{L}", "pass": p.is_file(), "detail": str(p)})
    cap = out_dir / "neutral_capture.json"
    checks.append({"id": "NEU_capture_json", "pass": cap.is_file(), "detail": str(cap)})
    if cap.is_file():
        data = json.loads(cap.read_text(encoding="utf-8"))
        checks.append(
            {
                "id": "NEU_n_tasks",
                "pass": int(data.get("n_tasks") or 0) >= 1,
                "detail": f"n_tasks={data.get('n_tasks')}",
            }
        )
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Experiment JSON paths")
    ap.add_argument("--neutral-artifacts", type=Path, default=NEUTRAL_ARTIFACTS)
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else []
    if not paths:
        corpus = ROOT / "experiments" / "unicaa01_corpus" / "out"
        paths = sorted(corpus.glob("*_latest.json"))

    reports = [verify_experiment_json(p) for p in paths]
    neu = verify_neutral_artifacts(args.neutral_artifacts) if args.neutral_artifacts.is_dir() else None

    out = {"experiments": reports, "neutral_artifacts": neu}
    out_path = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "verification_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), **out}, indent=2))
    return 0 if all(r["pass"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
