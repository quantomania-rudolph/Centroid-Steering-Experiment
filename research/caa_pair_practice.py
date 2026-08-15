#!/usr/bin/env python
"""End-to-end UNI_CAA01 Method B practice pair — calm vs bad forced chaos replay."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

import torch  # noqa: E402

from audit_divergence_metrics import audit_invocation  # noqa: E402
from caa_pair_practice_task import (  # noqa: E402
    EXPERIMENT,
    TASK_SLUG,
    USER_PROMPT,
    build_matched_prefix,
    encode_prompt,
)
from jlens_qwen.caa.activation_capture import capture_matched_pair_activations  # noqa: E402
from jlens_qwen.caa.method_b_databus import build_method_b_record  # noqa: E402
from jlens_qwen.caa.steering_vectors import compute_steering_vectors, save_steering_artifacts  # noqa: E402
from jlens_qwen.chaos.live_generate import live_generate  # noqa: E402
from jlens_qwen.chaos.monitor_bus import MonitorChaosBus  # noqa: E402
from jlens_qwen.chaos.probe import ProbeEngine  # noqa: E402
from jlens_qwen.jacobian_lens import FittedJacobianLens  # noqa: E402
from jlens_qwen.model_loader import load_qwen_coder  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402
from validate_caa_pair import validate_pair  # noqa: E402

DEFAULT_LENS = ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt"
OUT_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "practice_pair"
VECTOR_OUT = ROOT / "artifacts" / "caa" / "UNI_CAA01"
EXTRACT_LAYERS = [10, 16, 22]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run_forced(
    bundle: Any,
    probe: ProbeEngine,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    tail_kind: str,
    invocation_id: str,
    expected_d: int | None = None,
    forced_prefix_len: int | None = None,
    free_generate_after: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int | None = None,
    probe_ready: bool = False,
) -> dict[str, Any]:
    if free_generate_after:
        raise NotImplementedError(
            "free_generate_after desyncs tails from spec — use full forced_token_ids replay"
        )
    bus = MonitorChaosBus()
    if not probe_ready:
        probe.setup()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
    out = live_generate(
        bundle,
        input_ids,
        max_new_tokens=max_new_tokens or len(completion_ids),
        temperature=temperature,
        chaos_enabled=True,
        probe=probe,
        bus=bus,
        session_id="caa_practice",
        invocation_id=invocation_id,
        meta={"task_slug": TASK_SLUG, "tail_kind": tail_kind, "experiment": EXPERIMENT},
        forced_token_ids=completion_ids,
    )
    assistant_text = out["text"]
    chaos = out.get("chaos") or {}
    inv = {
        "id": invocation_id,
        "completion_tokens": out.get("completion_tokens"),
        "assistant": assistant_text,
        "chaos": chaos,
    }
    traced = trace_invocation(
        inv,
        experiment=EXPERIMENT,
        task_slug=TASK_SLUG,
        use_tokenizer=True,
    ).to_dict()
    audit = audit_invocation(inv, experiment=EXPERIMENT, task_slug=TASK_SLUG)
    d_split = traced.get("d_consensus")
    if d_split is None and expected_d is not None:
        # Text-LCP divergence (Method B); do not substitute chaos-lagged d.
        d_split = int(expected_d)
    tail_ids = completion_ids[int(d_split) :] if d_split is not None else completion_ids
    prefix_for_mb = list(completion_ids[: int(d_split)]) if d_split is not None and d_split > 0 else []
    tail_text = bundle.tokenizer.decode(tail_ids, skip_special_tokens=True) if tail_ids else ""
    mb = build_method_b_record(
        task_slug=TASK_SLUG,
        experiment=EXPERIMENT,
        invocation_id=invocation_id,
        user_prompt=USER_PROMPT,
        divergence=traced,
        tail_token_ids=tail_ids,
        tail_text=tail_text,
        tail_kind=tail_kind,  # type: ignore[arg-type]
        prompt_len=len(prompt_ids),
        extract_layers=EXTRACT_LAYERS,
        fallback_d=d_split,
        fallback_prefix_token_ids=prefix_for_mb,
    )
    return {
        "invocation_id": invocation_id,
        "tail_kind": tail_kind,
        "assistant_text": assistant_text,
        "token_ids": out.get("token_ids"),
        "chaos": chaos,
        "invocation": inv,
        "traced": traced,
        "audit": audit,
        "method_b_record": mb.to_dict() if mb else None,
        "prefix_token_ids": (
            (mb.prefix_token_ids if mb else None)
            or traced.get("prefix_token_ids")
            or prefix_for_mb
            or []
        ),
        "experiment": EXPERIMENT,
        "task_slug": TASK_SLUG,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run UNI_CAA01 Method B practice pair")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="Build spec only, no model")
    ap.add_argument(
        "--extract-vectors",
        action="store_true",
        help="Also compute û_beh/û_cond and save to artifacts/caa/UNI_CAA01/",
    )
    ap.add_argument(
        "--vectors-only",
        action="store_true",
        help="Skip model run; extract vectors from existing activations_at_d.pt/json",
    )
    args = ap.parse_args()

    from divergence_support import get_qwen_tokenizer

    tokenizer = get_qwen_tokenizer()
    spec = build_matched_prefix(tokenizer)
    prompt_ids = encode_prompt(tokenizer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.out_dir / "matched_prefix_spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "d": spec.d,
                    "prefix_len": len(spec.prefix_token_ids),
                    "calm_tail_len": len(spec.calm_tail_token_ids),
                    "bad_tail_len": len(spec.bad_tail_token_ids),
                    "spec_path": str(spec_path),
                },
                indent=2,
            )
        )
        return 0

    if args.vectors_only:
        return _extract_vectors_from_disk(args.out_dir, VECTOR_OUT)

    device = _pick_device(args.device)
    print(f"Loading Qwen on {device} ...", flush=True)
    bundle = load_qwen_coder(device=device, dtype=torch.float16, prefer_local=True)
    if device == "xpu" and str(bundle.device) != "xpu":
        raise RuntimeError(f"Requested xpu but model landed on {bundle.device}")
    print(f"Model device confirmed: {bundle.device}", flush=True)
    lens = FittedJacobianLens.load(str(DEFAULT_LENS)) if DEFAULT_LENS.is_file() else None
    probe = ProbeEngine(bundle, lens)
    probe.setup()

    calm_id = f"practice-calm-{uuid.uuid4().hex[:8]}"
    bad_id = f"practice-bad-{uuid.uuid4().hex[:8]}"

    calm_run = _run_forced(
        bundle,
        probe,
        prompt_ids=prompt_ids,
        completion_ids=spec.calm_full_token_ids,
        tail_kind="calm",
        invocation_id=calm_id,
        expected_d=spec.d,
        probe_ready=True,
    )
    bad_run = _run_forced(
        bundle,
        probe,
        prompt_ids=prompt_ids,
        completion_ids=spec.bad_full_token_ids,
        tail_kind="bad",
        invocation_id=bad_id,
        expected_d=spec.d,
        probe_ready=True,
    )

    # Fast shared-prefix activation capture (1 KV prefill + 2 decode steps).
    capture = capture_matched_pair_activations(
        bundle,
        prompt_ids,
        spec.calm_full_token_ids,
        spec.bad_full_token_ids,
        spec.d,
        EXTRACT_LAYERS,
    )
    extract_positions = list(capture.get("positions", {}).values())

    validation = validate_pair(
        spec=spec.to_dict(),
        calm_run=calm_run,
        bad_run=bad_run,
        prompt_len=len(prompt_ids),
    )

    result = {
        "experiment": EXPERIMENT,
        "task_slug": TASK_SLUG,
        "user_prompt": USER_PROMPT,
        "spec": spec.to_dict(),
        "prompt_len": len(prompt_ids),
        "extract_positions": extract_positions,
        "extract_layers": EXTRACT_LAYERS,
        "calm_run": {
            k: v
            for k, v in calm_run.items()
            if k not in ("chaos",)
        },
        "bad_run": {
            k: v
            for k, v in bad_run.items()
            if k not in ("chaos",)
        },
        "validation": validation,
        "capture_method": capture.get("capture_method"),
    }

    out_path = args.out_dir / "practice_pair_result.json"
    act_path = args.out_dir / "activations_at_d.pt"
    torch.save(capture, act_path)
    # Compact JSON index for humans (no huge vectors).
    act_index = {
        "pt_path": str(act_path),
        "layers": capture.get("layers"),
        "positions": capture.get("positions"),
        "d": capture.get("d"),
        "prompt_len": capture.get("prompt_len"),
        "capture_method": capture.get("capture_method"),
    }
    (args.out_dir / "activations_at_d.json").write_text(
        json.dumps(act_index, indent=2), encoding="utf-8"
    )
    result["activations_path"] = str(act_path)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Compact chaos summaries for review.
    for kind, run in ("calm", calm_run), ("bad", bad_run):
        chaos = run.get("chaos") or {}
        summary = {
            "eligible": validation[kind]["eligible"],
            "d_consensus": run.get("traced", {}).get("d_consensus"),
            "confidence": run.get("traced", {}).get("confidence"),
            "divergence_hints": chaos.get("divergence_hints"),
            "recipe_timeline": (chaos.get("monitor") or {}).get("recipe_timeline"),
            "n_tokens_full": chaos.get("n_tokens_full"),
        }
        (args.out_dir / f"chaos_summary_{kind}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    gate_path = args.out_dir / "practice_pair_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "pass": validation["pass"],
                "calm_pass": validation["calm"]["pass"],
                "bad_pass": validation["bad"]["pass"],
                "d": spec.d,
                "out": str(out_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "pass": validation["pass"],
                "d": spec.d,
                "calm_eligible": validation["calm"]["eligible"],
                "bad_eligible": validation["bad"]["eligible"],
                "bad_d": validation["bad"]["traced"].get("d_consensus"),
                "capture_method": capture.get("capture_method"),
                "out": str(out_path),
                "gate": str(gate_path),
            },
            indent=2,
        )
    )

    if args.extract_vectors and validation["pass"]:
        return _extract_vectors_from_disk(args.out_dir, VECTOR_OUT)
    return 0 if validation["pass"] else 1


def _extract_vectors_from_disk(pair_dir: Path, vector_dir: Path) -> int:
    pt = pair_dir / "activations_at_d.pt"
    legacy = pair_dir / "activations_at_d.json"
    from jlens_qwen.caa.steering_vectors import load_capture

    capture_path = pt if pt.is_file() else legacy
    if not capture_path.is_file():
        print(f"No activations at {pt} or {legacy}", file=sys.stderr)
        return 1
    capture = load_capture(capture_path)
    vectors = compute_steering_vectors(capture)
    written = save_steering_artifacts(
        vectors, vector_dir, metadata_extra={"source_capture": str(capture_path)}
    )
    gate = {
        "pass": bool(vectors.behavior and vectors.condition),
        "n_layers_beh": len(vectors.behavior),
        "n_layers_cond": len(vectors.condition),
        "notes": vectors.notes,
        "artifacts": written,
    }
    (vector_dir / "steering_extract_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
