#!/usr/bin/env python
"""Debug validate_pair for scheduler natural candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import get_qwen_tokenizer, resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_file  # noqa: E402
from jlens_qwen.caa.method_b_databus import build_method_b_record  # noqa: E402
from jlens_qwen.caa.miner_pairing import build_clean_chaos_from_token_ids, extract_d_from_lcp  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402
from validate_caa_pair import validate_pair  # noqa: E402


def main() -> int:
    tok = get_qwen_tokenizer()
    data = json.loads(
        (ROOT / "experiments/out/divergence_corpus_stress_b_latest.json").read_text(encoding="utf-8")
    )
    inv = data["invocations"][0]
    trace = trace_invocation(
        inv,
        experiment=data["experiment"],
        task_slug=data["task_slug"],
        experiment_data=data,
        use_tokenizer=False,
    )
    bad_ids = list(inv["chaos"]["token_ids"])
    prefix = bad_ids[:55]
    calm_path = (
        ROOT
        / "artifacts/caa/UNI_CAA01/calm_tails/live_panic_demo_scheduler__chatcmpl_b757de025c4c__d55.txt"
    )
    calm_spec = build_calm_tail_from_file(
        prefix_token_ids=prefix, d=55, calm_completion_path=calm_path, tokenizer=tok
    )
    d_extract = extract_d_from_lcp(calm_spec.calm_full_token_ids, bad_ids)
    bad_text = resolve_assistant_text(inv, data)
    prompt_len = int(inv["prompt_tokens"])
    spec = {
        "d": d_extract,
        "prefix_token_ids": list(calm_spec.prefix_token_ids)[:d_extract],
        "calm_tail_token_ids": calm_spec.calm_tail_token_ids,
        "bad_tail_token_ids": bad_ids[d_extract:],
        "calm_tail_text": calm_spec.calm_text,
        "bad_tail_text": bad_text[d_extract:] if bad_text else "",
    }
    calm_chaos = build_clean_chaos_from_token_ids(calm_spec.calm_full_token_ids)
    calm_mb = build_method_b_record(
        task_slug=data["task_slug"],
        experiment=data["experiment"],
        invocation_id=f"{inv['id']}_calm",
        user_prompt=data["message"],
        divergence=trace.to_dict(),
        tail_token_ids=calm_spec.calm_tail_token_ids,
        tail_text=calm_spec.calm_text,
        tail_kind="calm",
        prompt_len=prompt_len,
        fallback_d=d_extract,
        fallback_prefix_token_ids=spec["prefix_token_ids"],
    )
    bad_mb = build_method_b_record(
        task_slug=data["task_slug"],
        experiment=data["experiment"],
        invocation_id=inv["id"],
        user_prompt=data["message"],
        divergence=trace.to_dict(),
        tail_token_ids=bad_ids[d_extract:],
        tail_text=spec["bad_tail_text"],
        tail_kind="observed",
        prompt_len=prompt_len,
    )
    calm_run = {
        "assistant_text": calm_spec.calm_text,
        "tail_text": calm_spec.calm_text,
        "chaos": calm_chaos,
        "invocation": {
            "id": f"{inv['id']}_calm",
            "completion_tokens": len(calm_spec.calm_full_token_ids),
            "assistant": calm_spec.calm_text,
            "chaos": calm_chaos,
        },
        "experiment": data["experiment"],
        "task_slug": data["task_slug"],
        "method_b_record": calm_mb.to_dict() if calm_mb else None,
    }
    bad_run = {
        "assistant_text": bad_text,
        "chaos": inv["chaos"],
        "invocation": inv,
        "experiment": data["experiment"],
        "task_slug": data["task_slug"],
        "method_b_record": bad_mb.to_dict() if bad_mb else None,
    }
    pair = validate_pair(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
    print("pass", pair["pass"])
    for c in pair.get("pair_checks", []):
        if not c["pass"]:
            print("PAIR FAIL", c["id"], c["detail"])
    for kind in ("calm", "bad"):
        t = pair[kind]
        print("---", kind, "pass", t["pass"])
        for c in t.get("checks", []):
            if not c["pass"]:
                print(" ", c["id"], c["detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
