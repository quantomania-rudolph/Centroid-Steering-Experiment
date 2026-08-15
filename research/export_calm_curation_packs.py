#!/usr/bin/env python
"""Export calm-tail curation packs for dense tracer-eligible panic invocations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import get_qwen_tokenizer, resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.calm_tail_registry import calm_tail_filename  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import CALM_TAILS, CURATION_PACKS, ensure_dirs  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402


def _load_experiments(globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in globs:
        paths.extend(sorted(ROOT.glob(pattern)))
    return paths


def build_pack(
    *,
    data: dict[str, Any],
    inv: dict[str, Any],
    trace: Any,
    tokenizer: Any,
) -> dict[str, Any] | None:
    if not trace.eligible or trace.d_consensus is None:
        return None
    if not trace.token_ids_match:
        return None
    bad_ids = list((inv.get("chaos") or {}).get("token_ids") or [])
    d = int(trace.d_consensus)
    prefix_ids = list(trace.prefix_token_ids or bad_ids[:d])
    if len(prefix_ids) < d:
        return None
    prefix_text = tokenizer.decode(prefix_ids) if tokenizer else trace.prefix_text
    bad_text = resolve_assistant_text(inv, data)
    task_slug = str(data.get("task_slug") or "")
    inv_id = str(inv.get("id") or "")
    calm_name = calm_tail_filename(task_slug, inv_id, d, ext="txt")
    return {
        "task_slug": task_slug,
        "invocation_id": inv_id,
        "experiment": str(data.get("experiment") or ""),
        "pair_id": data.get("unicaa01_pair_id"),
        "arm": data.get("unicaa01_arm"),
        "user_message": str(data.get("message") or ""),
        "session_path": str(data.get("session_path") or ""),
        "prompt_tokens": int(inv.get("prompt_tokens") or 0),
        "completion_tokens": int(inv.get("completion_tokens") or 0),
        "d_tracer": d,
        "prefix_len": len(prefix_ids),
        "prefix_token_ids": prefix_ids,
        "prefix_text": prefix_text,
        "bad_token_at_d": bad_ids[d] if len(bad_ids) > d else None,
        "bad_token_at_d_text": (
            tokenizer.decode([bad_ids[d]]) if tokenizer and len(bad_ids) > d else ""
        ),
        "bad_tail_text_preview": bad_text[d:] if bad_text else "",
        "calm_tail_filename": calm_name,
        "calm_tail_path": str(CALM_TAILS / calm_name),
        "token_ids_match": bool(trace.token_ids_match),
        "confidence": str(trace.confidence),
        "notes": [
            "Composer senior-editor: write calm completion text only.",
            "Prefix lock: encode(text)[:d] must equal prefix_token_ids.",
            "No pip/sorry/stub/NotImplementedError in calm tail.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export UNICAA01 calm curation packs")
    ap.add_argument(
        "--glob",
        action="append",
        default=[
            "experiments/unicaa01_corpus/out/*_latest.json",
            "experiments/out/*_latest.json",
        ],
    )
    ap.add_argument("--out", type=Path, default=CURATION_PACKS)
    args = ap.parse_args()

    ensure_dirs()
    tok = get_qwen_tokenizer()
    packs: list[dict[str, Any]] = []
    skipped = 0

    for path in _load_experiments(args.glob):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        experiment = str(data.get("experiment") or path.stem)
        task_slug = str(data.get("task_slug") or "")
        for inv in data.get("invocations") or []:
            trace = trace_invocation(
                inv,
                experiment=experiment,
                task_slug=task_slug,
                experiment_data=data,
                use_tokenizer=False,
            )
            pack = build_pack(data=data, inv=inv, trace=trace, tokenizer=tok)
            if pack is None:
                skipped += 1
                continue
            pack["source_experiment_json"] = str(path)
            packs.append(pack)

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    for pack in packs:
        out_file = args.out / f"{pack['experiment']}__{pack['invocation_id']}__d{pack['d_tracer']}.json"
        out_file.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    index = {
        "n_packs": len(packs),
        "n_skipped": skipped,
        "packs": [p["calm_tail_filename"] for p in packs],
        "files": [str(args.out / f"{p['experiment']}__{p['invocation_id']}__d{p['d_tracer']}.json") for p in packs],
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
