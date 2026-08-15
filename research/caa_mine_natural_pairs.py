#!/usr/bin/env python
"""Mine Method B natural pairs — Phase 3A fixture; Phase 3B tracer miner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.admission import admission_checks, verify_capture_integrity  # noqa: E402
from jlens_qwen.caa.capture_manifest import (  # noqa: E402
    CaptureManifest,
    assert_not_practice_path,
    write_admission_gate,
    write_manifest,
)
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.kv_equivalence import extract_positions  # noqa: E402
from jlens_qwen.caa.miner_pairing import build_clean_chaos_from_token_ids  # noqa: E402
from jlens_qwen.caa.natural_miner import mine_corpus, scan_corpus  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import (  # noqa: E402
    CALM_TAILS as UNICAA01_CALM_TAILS,
    CORPUS_GLOB,
    NATURAL_PAIRS as UNICAA01_NATURAL_PAIRS,
)
from jlens_qwen.chaos.constants import PROBE_LAYERS  # noqa: E402
from prefix_divergence_tracer import trace_all_glob, trace_invocation  # noqa: E402
from validate_caa_pair import validate_pair  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
FIXTURE_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "fixtures" / "natural_pair_synth"
NATURAL_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "pairs" / "natural"
CALM_TAILS = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calm_tails"
CAA_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01"


def _synth_capture(*, d: int = 3, prompt_len: int = 8, dim: int = 8) -> dict[str, Any]:
    layers = list(PROBE_LAYERS)
    positions = extract_positions(prompt_len, d)
    shared = [0.25] * dim
    calm_d = [1.0] + [0.0] * (dim - 1)
    bad_d = [0.0, 1.0] + [0.0] * (dim - 2)
    return {
        "layers": layers,
        "prompt_len": prompt_len,
        "d": d,
        "positions": positions,
        "calm": {L: {"shared": list(shared), "divergent": list(calm_d)} for L in layers},
        "bad": {L: {"shared": list(shared), "divergent": list(bad_d)} for L in layers},
        "capture_method": "shared_kv_prefill",
        "source": "natural",
        "pair_slug": "fixture_synth_pair",
        "task_slug": "fixture_synth_pair",
    }


def _synth_pair_runs(d: int = 3) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = list(range(100, 100 + d))
    calm_tail = [200, 201]
    bad_tail = [300, 301]
    calm_ids = prefix + calm_tail
    bad_ids = prefix + bad_tail

    def _tokens(n: int, panic_at: int | None = None):
        toks = []
        for i in range(n):
            flags = []
            distress = stub = 0.0
            if panic_at is not None and i >= panic_at:
                flags = ["B_distress", "B_stub"]
                distress = stub = 0.2
            toks.append(
                {
                    "step": i,
                    "token_id": (calm_ids if panic_at is None else bad_ids)[i],
                    "B_vocab": {"distress_mass": distress, "stub_mass": stub},
                    "flags": flags,
                }
            )
        return toks

    calm_chaos = build_clean_chaos_from_token_ids(calm_ids)
    bad_chaos = {
        "tokens": _tokens(len(bad_ids), panic_at=d),
        "token_ids": bad_ids,
        "n_tokens_full": len(bad_ids),
        "divergence_hints": {"d_lexicon": d, "d_flags": d, "b_distress_rate": 0.5, "b_stub_rate": 0.5},
        "monitor": {
            "recipe_timeline": [
                {"recipe_id": "B_DISTRESS", "step": d},
                {"recipe_id": "B_STUB", "step": d},
            ]
        },
    }
    spec = {
        "d": d,
        "prefix_token_ids": prefix,
        "calm_tail_token_ids": calm_tail,
        "bad_tail_token_ids": bad_tail,
        "calm_tail_text": "return True",
        "bad_tail_text": "Sorry pip install raise NotImplementedError",
    }
    calm_run = {
        "assistant_text": "```python\ndef ok():\n    return True\n```",
        "tail_text": "return True",
        "chaos": calm_chaos,
        "invocation": {"id": "calm_fx", "completion_tokens": len(calm_ids), "assistant": "return True", "chaos": calm_chaos},
        "prefix_token_ids": prefix,
        "experiment": "fixture",
        "task_slug": "fixture_synth_pair",
        "method_b_record": {
            "d": d, "prefix_token_ids": prefix, "tail_token_ids": calm_tail,
            "full_completion_token_ids": calm_ids, "extract_positions": [8 + d - 1, 8 + d],
            "extract_layers": list(PROBE_LAYERS), "tail_kind": "calm",
        },
    }
    bad_run = {
        "assistant_text": "Sorry, cannot. pip install x. raise NotImplementedError",
        "chaos": bad_chaos,
        "invocation": {"id": "bad_fx", "completion_tokens": len(bad_ids), "assistant": "Sorry NotImplementedError pip install", "chaos": bad_chaos},
        "prefix_token_ids": prefix,
        "experiment": "fixture",
        "task_slug": "fixture_synth_pair",
        "method_b_record": {
            "d": d, "prefix_token_ids": prefix, "tail_token_ids": bad_tail,
            "full_completion_token_ids": bad_ids, "extract_positions": [8 + d - 1, 8 + d],
            "extract_layers": list(PROBE_LAYERS), "tail_kind": "observed",
        },
    }
    return spec, calm_run, bad_run


def run_fixture() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    prompt_len = 8
    d = 3
    spec, calm_run, bad_run = _synth_pair_runs(d=d)
    pair = validate_pair(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
    capture = _synth_capture(d=d, prompt_len=prompt_len)
    import torch

    torch.save(capture, FIXTURE_DIR / "activations_at_d.pt")
    (FIXTURE_DIR / "matched_prefix_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (FIXTURE_DIR / "method_b_calm.json").write_text(json.dumps(calm_run["method_b_record"], indent=2), encoding="utf-8")
    (FIXTURE_DIR / "method_b_bad.json").write_text(json.dumps(bad_run["method_b_record"], indent=2), encoding="utf-8")
    (FIXTURE_DIR / "purity_gate.json").write_text(json.dumps({"pass": pair["pass"], "calm": pair["calm"], "bad": pair["bad"]}, indent=2), encoding="utf-8")

    manifest = CaptureManifest(
        source="natural", pair_slug="fixture_synth_pair", d=d, prompt_len=prompt_len,
        layers=list(PROBE_LAYERS), capture_method="shared_kv_prefill", extraction_position="divergent",
        calm_purity_pass=bool(pair["calm"]["pass"]), bad_purity_pass=bool(pair["bad"]["pass"]),
        pair_gate_pass=bool(pair["pass"]), kv_live_gate_pass=False,
        notes=["fixture_synth — Phase 3A; pool_eligible false until kv_live"],
    )
    cap = verify_capture_integrity(capture, manifest=manifest)
    adm = admission_checks(pair_validate_pass=pair["pass"], cap_pass=cap["pass"], manifest=manifest, capture_dir=FIXTURE_DIR)
    fixture_ok = pair["pass"] and cap["pass"] and not manifest.pool_eligible(capture_dir=FIXTURE_DIR)
    write_admission_gate(FIXTURE_DIR, source="natural", pass_=adm["pass"], checks=adm["checks"] + cap["checks"], pool_eligible=adm["pool_eligible"], extra={"phase": "3A_fixture", "pair_validate": pair["pass"]})
    write_manifest(manifest, FIXTURE_DIR)

    sab = []
    try:
        assert_not_practice_path(ROOT / "artifacts/caa/UNI_CAA01/practice_pair")
        sab.append({"id": "NM-S1", "pass": False, "detail": "should have raised"})
    except ValueError:
        sab.append({"id": "NM-S1", "pass": True, "detail": "practice path refused"})

    bad_spec = dict(spec)
    bad_spec["prefix_token_ids"] = list(spec["prefix_token_ids"])
    bad_spec["prefix_token_ids"][-1] = 999
    mismatch = validate_pair(spec=bad_spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
    sab.append({"id": "NM-S2", "pass": mismatch["pass"] is False, "detail": "token mismatch fails pair gate"})

    # NM-S3: bad without stub+distress should fail bad T gates
    weak_bad = dict(bad_run)
    weak_chaos = build_clean_chaos_from_token_ids(bad_run["chaos"]["token_ids"])
    weak_bad["chaos"] = weak_chaos
    weak_bad["assistant_text"] = "def ok(): return 1"
    weak_pair = validate_pair(spec=spec, calm_run=calm_run, bad_run=weak_bad, prompt_len=prompt_len)
    sab.append({"id": "NM-S3", "pass": weak_pair["bad"]["pass"] is False, "detail": "bad without panic fails T gates"})

    corrupt = dict(capture)
    corrupt["positions"] = {"shared": 0, "divergent": 1}
    cap_bad = verify_capture_integrity(corrupt, manifest=manifest)
    sab.append({"id": "NM-S4", "pass": cap_bad["pass"] is False, "detail": "corrupt positions fail CAP3"})

    checks = [
        {"id": "3A_pair_validate", "pass": pair["pass"], "detail": "fixture pair"},
        {"id": "3A_cap", "pass": cap["pass"], "detail": json.dumps(cap["checks"])[:200]},
        {"id": "3A_pool_blocked_without_kv_live", "pass": not manifest.pool_eligible(capture_dir=FIXTURE_DIR), "detail": "pool_eligible false until kv live"},
        {"id": "3A_fixture_dir", "pass": FIXTURE_DIR.is_dir(), "detail": str(FIXTURE_DIR)},
    ]
    payload = write_gate(GATES / "caa_natural_pair_admission_gate.json", gate="caa_natural_pair_admission", pass_=fixture_ok and all(s["pass"] for s in sab), checks=checks, sabotage=sab, environment={"mode": "fixture", "n_natural_admitted": 0}, extra={"fixture_dir": str(FIXTURE_DIR)})
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


def run_live(
    *,
    scan_only: bool = False,
    dry_run: bool = False,
    device: str = "xpu",
    unicaa01_corpus: bool = False,
    max_pairs: int | None = None,
    only_tasks: set[str] | None = None,
) -> int:
    kv_live = GATES / "kv_equivalence_live_gate.json"
    blockers: list[str] = []
    if not kv_live.is_file() or not json.loads(kv_live.read_text(encoding="utf-8")).get("pass"):
        blockers.append("Phase 1 live KV gate missing or failed")

    corpus_glob = CORPUS_GLOB if unicaa01_corpus else "experiments/out/*_latest.json"
    calm_tails = UNICAA01_CALM_TAILS if unicaa01_corpus else CALM_TAILS
    natural_root = UNICAA01_NATURAL_PAIRS if unicaa01_corpus else NATURAL_ROOT

    mode = "scan"
    if dry_run:
        mode = "dry_run"
    elif os.environ.get("CAA_MINE_LIVE") == "1" and not scan_only:
        mode = "capture"

    if mode == "capture" and blockers:
        payload = write_gate(
            GATES / "caa_natural_pair_live_gate.json",
            gate="caa_natural_pair_live",
            pass_=False,
            checks=[{"id": "3B_prerequisites", "pass": False, "detail": "; ".join(blockers)}],
            sabotage=[{"id": "3B-block", "pass": True, "detail": "blocked when prerequisites missing"}],
            environment={"mode": "live"},
            extra={"blockers": blockers, "n_candidates": 0, "n_rejected": 0, "n_admitted": 0},
        )
        print(json.dumps(payload, indent=2))
        return 1

    bundle = None
    tokenizer = None
    if mode in ("dry_run", "capture"):
        from divergence_support import get_qwen_tokenizer

        tokenizer = get_qwen_tokenizer()
    if mode == "capture":
        from jlens_qwen.model_loader import load_qwen_coder
        import torch

        if str(device).startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
        bundle = load_qwen_coder(device=device)

    if mode == "scan":
        result = scan_corpus(
            root=ROOT,
            corpus_glob=corpus_glob,
            trace_invocation_fn=trace_invocation,
            calm_tails_root=calm_tails,
        )
    else:
        result = mine_corpus(
            root=ROOT,
            natural_root=natural_root,
            corpus_glob=corpus_glob,
            trace_invocation_fn=trace_invocation,
            validate_pair_fn=validate_pair,
            resolve_assistant_text_fn=resolve_assistant_text,
            calm_tails_root=calm_tails,
            bundle=bundle,
            mode=mode,
            tokenizer=tokenizer,
            max_pairs=max_pairs,
            only_tasks=only_tasks,
        )

    corpus_eligible = sum(
        1
        for t in trace_all_glob(corpus_glob, use_tokenizer=False)
        if t.eligible and t.d_consensus is not None
    )
    with_calm = sum(1 for c in result.candidates if c.calm_tail_path)
    validate_fail = sum(
        1 for r in result.rejections if r.reason in {"validate_pair", "d_coherence", "calm_tail_invalid"}
    )
    capture_fail = sum(
        1 for r in result.rejections if r.reason in {"capture_error", "capture_write_error", "admission"}
    )
    # Capture: require requested new admits (default ≥1) and zero hard validate fails.
    admit_target = max_pairs if (mode == "capture" and max_pairs is not None) else 1
    if mode == "capture":
        pass_ = result.n_admitted >= admit_target and validate_fail == 0 and capture_fail == 0
    elif mode == "dry_run":
        # Dry-run passes only if we validated at least one pair and had no hard validate failures
        # among candidates that had calm tails (missing_calm_tail is expected quarantine).
        pass_ = result.n_scanned > 0 and result.n_admitted >= 1 and validate_fail == 0
    else:
        pass_ = result.n_candidates >= 0

    payload = write_gate(
        GATES / "caa_natural_pair_live_gate.json",
        gate="caa_natural_pair_live",
        pass_=pass_ and mode != "scan",
        checks=[
            {"id": "3B_tracer_scan", "pass": result.n_scanned > 0, "detail": f"scanned={result.n_scanned}"},
            {"id": "3B_eligible", "pass": result.n_eligible > 0, "detail": f"eligible={result.n_eligible}"},
            {"id": "3B_candidates", "pass": result.n_candidates > 0, "detail": f"candidates={result.n_candidates}"},
            {
                "id": "3B_with_calm_tail",
                "pass": with_calm > 0,
                "detail": f"with_calm={with_calm}",
            },
            {
                "id": "3B_validated_or_captured",
                "pass": result.n_admitted >= admit_target,
                "detail": (
                    f"n_admitted={result.n_admitted} target={admit_target} mode={mode} "
                    f"(dry_run=pair_validated; capture=pool_eligible_on_disk)"
                ),
            },
            {
                "id": "3B_no_validate_failures",
                "pass": validate_fail == 0,
                "detail": f"validate_or_coherence_fails={validate_fail}",
            },
            {
                "id": "3B_no_capture_failures",
                "pass": mode != "capture" or capture_fail == 0,
                "detail": f"capture_or_admission_fails={capture_fail}",
            },
        ],
        sabotage=[{"id": "3B-block", "pass": True, "detail": "miner runs without crash"}],
        environment={
            "mode": mode,
            "n_pair_validated_or_captured": result.n_admitted,
            "n_pool_eligible_on_disk": result.n_admitted if mode == "capture" else 0,
            "admit_target": admit_target,
            "metric_note": (
                "dry_run n_admitted = pair-validated count; "
                "capture n_admitted = pool-eligible activations on disk; "
                "max_pairs = new admits this run (skips already pool-eligible)"
            ),
        },
        extra={
            **result.to_dict(),
            "n_corpus_eligible": corpus_eligible,
            "n_with_calm_tail": with_calm,
            "blockers": blockers,
            "corpus_glob": corpus_glob,
            "natural_root": str(natural_root),
            "max_pairs": max_pairs,
            "only_tasks": sorted(only_tasks) if only_tasks else None,
        },
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] or mode == "scan" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="CAA natural pair miner")
    ap.add_argument("--fixture", action="store_true", help="Phase 3A fixture mode")
    ap.add_argument("--live", action="store_true", help="Phase 3B live capture (CAA_MINE_LIVE=1)")
    ap.add_argument("--scan", action="store_true", help="Tracer scan only — list corpus candidates")
    ap.add_argument("--dry-run", action="store_true", help="Validate pairs without XPU capture")
    ap.add_argument("--unicaa01-corpus", action="store_true", help="Scan UNICAA01 corpus only")
    ap.add_argument("--device", default="xpu")
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Stop after N successful dry-run validations or NEW live captures "
        "(already pool-eligible dirs are skipped and do not count)",
    )
    ap.add_argument(
        "--only-tasks",
        default=None,
        help="Comma-separated task_slug filter (e.g. unicaa01_p16_panic,unicaa01_p21_panic)",
    )
    args = ap.parse_args()
    only_tasks = None
    if args.only_tasks:
        only_tasks = {t.strip() for t in str(args.only_tasks).split(",") if t.strip()}
    if args.fixture:
        return run_fixture()
    if args.live or args.scan or args.dry_run:
        return run_live(
            scan_only=args.scan,
            dry_run=args.dry_run,
            device=args.device,
            unicaa01_corpus=args.unicaa01_corpus,
            max_pairs=args.max_pairs,
            only_tasks=only_tasks,
        )
    return run_fixture()


if __name__ == "__main__":
    raise SystemExit(main())
