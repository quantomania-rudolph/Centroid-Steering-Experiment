#!/usr/bin/env python
"""Build CuratorBrief JSON from UNICAA01 curation packs + registry starters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_schemas import (  # noqa: E402
    CuratorBrief,
    DemoOracle,
    DivergenceContext,
    OutputContract,
    PrefixLock,
)
from divergence_support import get_qwen_tokenizer, resolve_assistant_text  # noqa: E402
from jlens_qwen.caa.calm_tail_registry import calm_tail_filename  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import CALM_TAILS, CURATION_PACKS, ensure_dirs  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402
from unicaa01_registry import get_pair  # noqa: E402
from unicaa01_demo_oracles import get_oracle  # noqa: E402
from unicaa01_oracle_invariants import sanitize_bad_tail_preview  # noqa: E402

BRIEFS_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_briefs"
STAGING_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "curator_staging"

FORBIDDEN = [
    "NotImplementedError",
    "TODO",
    "FIXME",
    "pip install",
    "sorry",
    "cannot",
    "unable to",
]


def extract_success_criteria(user_message: str) -> str:
    m = re.search(r"(demo\(\)[^.]*\.|must [^.]+\.)", user_message, re.I)
    if m:
        return m.group(1).strip()[:200]
    parts = user_message.split(".")
    for p in parts:
        if "must" in p.lower():
            return p.strip()[:200]
    return parts[0].strip()[:120] if parts else user_message[:120]


def task_summary_from_message(user_message: str) -> str:
    first = user_message.split(".")[0].strip()
    return first[:100]


def pack_to_brief(pack: dict[str, Any], *, pair: Any) -> CuratorBrief:
    task_slug = str(pack["task_slug"])
    inv_id = str(pack["invocation_id"])
    d = int(pack["d_tracer"])
    pair_id = str(pack.get("pair_id") or pair.pair_id)
    oracle_row = get_oracle(pair_id)
    if oracle_row is None:
        raise ValueError(f"missing demo oracle for pair_id={pair_id}")
    demo_oracle = DemoOracle(
        pair_id=oracle_row.pair_id,
        check_mode=oracle_row.check_mode,
        expect_repr=oracle_row.expect_repr,
    )
    calm_name = calm_tail_filename(task_slug, inv_id, d, ext="txt")
    if calm_name != str(pack.get("calm_tail_filename") or ""):
        raise ValueError(f"calm_tail_filename mismatch pack={pack.get('calm_tail_filename')} expected={calm_name}")

    bad_preview = sanitize_bad_tail_preview(
        pair_id, str(pack.get("bad_tail_text_preview") or "")
    )
    user_message = str(pack.get("user_message") or "")
    brief_id = f"{task_slug}__{inv_id.replace('-', '_')}__d{d}"

    return CuratorBrief(
        brief_version=1,
        brief_id=brief_id,
        task_slug=task_slug,
        pair_id=pair_id,
        invocation_id=inv_id,
        domain=str(pair.domain),
        user_message=user_message,
        panic_starter_main_py=str(pair.panic_starter),
        prefix_lock=PrefixLock(
            d=d,
            prefix_token_ids=list(pack["prefix_token_ids"]),
            prefix_text=str(pack["prefix_text"]),
            immutable=True,
        ),
        divergence_context=DivergenceContext(
            bad_token_at_d=pack.get("bad_token_at_d"),
            bad_token_at_d_text=str(pack.get("bad_token_at_d_text") or "")[:20],
            bad_tail_summary=bad_preview,
            forbidden_patterns=list(FORBIDDEN),
        ),
        output_contract=OutputContract(
            format="aider_listing",
            target_staging_path=str(STAGING_ROOT / brief_id / "candidate.txt"),
            target_admitted_path=str(CALM_TAILS / calm_name),
            task_dir=task_slug,
        ),
        success_criteria=extract_success_criteria(user_message),
        task_summary=task_summary_from_message(user_message),
        provenance={
            "source_pack": str(pack.get("_pack_path") or ""),
            "source_experiment_json": str(pack.get("source_experiment_json") or ""),
            "tracer_confidence": str(pack.get("confidence") or "medium"),
        },
        tracer_confidence=str(pack.get("confidence") or "medium"),
        demo_oracle=demo_oracle,
    )


def verify_pack_trace(pack: dict[str, Any]) -> bool:
    src = Path(str(pack.get("source_experiment_json") or ""))
    if not src.is_file():
        return False
    data = json.loads(src.read_text(encoding="utf-8"))
    inv_id = str(pack.get("invocation_id") or "")
    inv = next((i for i in data.get("invocations") or [] if str(i.get("id")) == inv_id), None)
    if inv is None:
        return False
    trace = trace_invocation(
        inv,
        experiment=str(data.get("experiment") or ""),
        task_slug=str(data.get("task_slug") or ""),
        experiment_data=data,
        use_tokenizer=False,
    )
    return list(trace.prefix_token_ids or []) == list(pack.get("prefix_token_ids") or [])


def build_all_briefs() -> list[CuratorBrief]:
    ensure_dirs()
    BRIEFS_ROOT.mkdir(parents=True, exist_ok=True)
    briefs: list[CuratorBrief] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(CURATION_PACKS.glob("unicaa01_corpus_*.json")):
        pack = json.loads(path.read_text(encoding="utf-8"))
        pack["_pack_path"] = str(path)
        if not verify_pack_trace(pack):
            skipped.append({"pack": str(path), "reason": "trace_mismatch"})
            continue
        try:
            pair = get_pair(str(pack["pair_id"]))
            briefs.append(pack_to_brief(pack, pair=pair))
        except (KeyError, ValueError) as exc:
            skipped.append({"pack": str(path), "reason": str(exc)})
    skipped_path = BRIEFS_ROOT / "skipped.json"
    skipped_path.write_text(
        json.dumps({"n_skipped": len(skipped), "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    return briefs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BRIEFS_ROOT)
    args = ap.parse_args()
    briefs = build_all_briefs()
    skipped_path = args.out / "skipped.json"
    n_skipped = 0
    if skipped_path.is_file():
        n_skipped = int(json.loads(skipped_path.read_text(encoding="utf-8")).get("n_skipped") or 0)
    if n_skipped > 0:
        print(json.dumps({"error": "skipped packs", "n_skipped": n_skipped}, indent=2))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for b in briefs:
        out = args.out / f"{b.brief_id}.json"
        out.write_text(json.dumps(b.to_dict(), indent=2), encoding="utf-8")
        files.append(str(out))
    index = {"n_briefs": len(briefs), "files": files}
    (args.out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
