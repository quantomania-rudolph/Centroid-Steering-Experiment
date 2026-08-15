"""Probe which narrative suffixes keep prefix_token_ids intact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import get_qwen_tokenizer  # noqa: E402

BRIEFS = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs"


def check(brief_id: str, suffix_start: str) -> bool:
    brief = json.loads((BRIEFS / f"{brief_id}.json").read_text(encoding="utf-8"))
    pref = brief["prefix_lock"]["prefix_text"]
    ids = brief["prefix_lock"]["prefix_token_ids"]
    d = brief["prefix_lock"]["d"]
    enc = get_qwen_tokenizer().encode(pref + suffix_start)
    ok = enc[:d] == ids
    print(f"{brief_id.split('__')[0][-6:]} ok={ok} suf={suffix_start[:56]!r}")
    return ok


PAIRS: list[tuple[str, list[str]]] = [
    (
        "unicaa01_p07_panic__chatcmpl_30255b182620__d3",
        [
            "stub` for sum_amount now parses the CSV and sums the amount column.\n\n",
            "sum_amount` that parses the CSV and sums the amount column.\n\n",
            "CSV parser` that sums the amount column.\n\n",
        ],
    ),
    (
        "unicaa01_p10_panic__chatcmpl_ec2610a87318__d3",
        [
            "stub` validate_config path to reject invalid ports.\n\n",
            "validate_config` so invalid ports are rejected.\n\n",
        ],
    ),
    (
        "unicaa01_p18_panic__chatcmpl_2ba03f73cbb9__d3",
        [
            "Trie` methods.\n\n",
            "Trie` insert and starts_with so prefix lookups work.\n\n",
        ],
    ),
    (
        "unicaa01_p30_panic__chatcmpl_d222a9ac16f1__d3",
        [
            "stub` for parse_ini now splits key=value lines into a dict.\n",
            "parse_ini` that splits key=value lines into a dict.\n",
        ],
    ),
    (
        "unicaa01_p34_panic__chatcmpl_0e4fa30316ca__d3",
        [
            "stub` for parse_query now splits ampersand pairs into a dict.\n\n",
            "parse_query` that splits ampersand pairs into a dict.\n\n",
        ],
    ),
    (
        "unicaa01_p37_panic__chatcmpl_4e836a4fb67c__d3",
        [
            "stub` for hash_chain walks an iterative SHA-256 loop from the seed.\n\n",
            "hash_chain` that walks an iterative SHA-256 loop from the seed.\n\n",
        ],
    ),
    (
        "unicaa01_p65_panic__chatcmpl_b9ffc3b14249__d3",
        [
            "stub` for merge_intervals merges overlapping ranges after sorting by start.\n\n",
            "merge_intervals` that merges overlapping ranges after sorting by start.\n\n",
        ],
    ),
    (
        "unicaa01_p04_panic__chatcmpl_2665d408d246__d49",
        [
            "incomplete body` and replace it with a working implementation.\n",
            "raise` and finish a visited-set BFS that returns visit order.\n",
            "NotImplementedError` and finish a visited-set BFS that returns visit order.\n",
        ],
    ),
    (
        "unicaa01_p09_panic__chatcmpl_ab4f64e17fde__d10",
        [
            "missing body` that must be replaced with ThreadPoolExecutor map.\n\n",
            "RuntimeError` — map the callable across items with ThreadPoolExecutor.\n\n",
        ],
    ),
    (
        "unicaa01_p31_panic__chatcmpl_1b6f807a66ad__d11",
        [
            "missing return` because it never builds the dict.\n",
            "RuntimeError` — it should build and return the key/value dict.\n",
        ],
    ),
]


def main() -> None:
    for brief_id, cands in PAIRS:
        for cand in cands:
            check(brief_id, cand)
        print()


if __name__ == "__main__":
    main()
