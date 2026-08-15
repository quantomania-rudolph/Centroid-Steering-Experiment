"""Find BPE+CD-safe narrative openings for residual stub tails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import assemble_calm_completion, extract_main_py  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402

BRIEFS = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs"
TAILS = ROOT / "artifacts/caa/UNI_CAA01_corpus/calm_tails"

CANDIDATES: dict[str, list[str]] = {
    "unicaa01_p07_panic__chatcmpl_30255b182620__d3": [
        "stub` for sum_amount now parses the CSV and sums the amount column.\n\n",
        "CSV parser` that totals the amount column.\n\n",
        "amount column` parser that totals values from the CSV.\n\n",
        "column total` for the CSV amount field.\n\n",
    ],
    "unicaa01_p10_panic__chatcmpl_ec2610a87318__d3": [
        "stub` validate_config path to reject invalid ports.\n\n",
        "validate_config` so invalid ports are rejected.\n\n",
        "config check` so invalid ports are rejected.\n\n",
        "port check` that rejects out-of-range values.\n\n",
    ],
    "unicaa01_p30_panic__chatcmpl_d222a9ac16f1__d3": [
        "stub` for parse_ini now splits key=value lines into a dict.\n",
        "parse_ini` that splits key=value lines into a dict.\n",
        "INI parser` that splits key=value lines into a dict.\n",
        "key=value parser` for simple INI lines.\n",
    ],
    "unicaa01_p31_panic__chatcmpl_1b6f807a66ad__d11": [
        "missing return` because it never builds the dict.\n",
        "RuntimeError` — build and return the key/value dict.\n",
        "empty return` — build and return the key/value dict.\n",
        "hole` — build and return the key/value dict.\n",
    ],
    "unicaa01_p34_panic__chatcmpl_0e4fa30316ca__d3": [
        "stub` for parse_query now splits ampersand pairs into a dict.\n\n",
        "parse_query` that splits ampersand pairs into a dict.\n\n",
        "query parser` that splits ampersand pairs into a dict.\n\n",
        "ampersand split` into a query dict.\n\n",
    ],
    "unicaa01_p37_panic__chatcmpl_4e836a4fb67c__d3": [
        "stub` for hash_chain walks an iterative SHA-256 loop from the seed.\n\n",
        "hash_chain` that walks an iterative SHA-256 loop from the seed.\n\n",
        "SHA-256 chain` walked iteratively from the seed.\n\n",
        "digest chain` walked iteratively from the seed.\n\n",
    ],
    "unicaa01_p65_panic__chatcmpl_b9ffc3b14249__d3": [
        "stub` for merge_intervals merges overlapping ranges after sorting by start.\n\n",
        "merge_intervals` that merges overlapping ranges after sorting by start.\n\n",
        "interval merge` that collapses overlaps after sorting by start.\n\n",
        "overlap merge` after sorting ranges by start.\n\n",
    ],
}


def main() -> None:
    tok = get_qwen_tokenizer()
    for brief_id, cands in CANDIDATES.items():
        brief = json.loads((BRIEFS / f"{brief_id}.json").read_text(encoding="utf-8"))
        code = extract_main_py((TAILS / f"{brief_id}.txt").read_text(encoding="utf-8"))
        bad = brief["divergence_context"]["bad_token_at_d"]
        bad_text = brief["divergence_context"]["bad_token_at_d_text"]
        print(f"=== {brief_id.split('__')[0]} bad={bad_text!r} ({bad})")
        for suf in cands:
            body = f"{suf}{brief['output_contract']['task_dir']}/main.py\n```python\n{code}\n```\n"
            try:
                assembled = assemble_calm_completion(
                    prefix_token_ids=brief["prefix_lock"]["prefix_token_ids"],
                    prefix_text=brief["prefix_lock"]["prefix_text"],
                    suffix_text=body,
                    tokenizer=tok,
                    d=brief["prefix_lock"]["d"],
                    bad_token_at_d=bad,
                    bad_token_at_d_text=bad_text,
                    strict=True,
                    allow_bridge=False,
                )
                print("  OK ", repr(suf[:48]), "bridge", assembled.bridge_used)
            except Exception as exc:  # noqa: BLE001
                print("  FAIL", repr(suf[:48]), str(exc)[:80])
        print()


if __name__ == "__main__":
    main()
