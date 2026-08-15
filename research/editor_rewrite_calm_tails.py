"""Senior-editor rewrite: clean prose/docs, keep verified code bodies."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_file  # noqa: E402
from verify_admitted_tail import verify_one  # noqa: E402

TAILS = ROOT / "artifacts/caa/UNI_CAA01_corpus/calm_tails"
BRIEFS = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs"
PROOF = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof"

# Suffix after the locked prefix (must close open backticks; no NotImplementedError/TODO).
CONTINUATIONS: dict[str, str] = {
    "unicaa01_p01_panic__chatcmpl_8b67ddd03286__d16": "incomplete body` that needs a thread-safe pop from the heap.\n\n",
    # Locked prefix already rambles; close the open tick with a concrete fault name.
    "unicaa01_p04_panic__chatcmpl_2665d408d246__d49": "broken raise` and ship a visited-set BFS that returns visit order from node 0.\n",
    "unicaa01_p07_panic__chatcmpl_30255b182620__d3": "CSV parser` that totals the amount column.\n\n",
    "unicaa01_p09_panic__chatcmpl_ab4f64e17fde__d10": "RuntimeError` — map the callable across items with ThreadPoolExecutor.\n\n",
    "unicaa01_p10_panic__chatcmpl_ec2610a87318__d3": "config check` so invalid ports are rejected.\n\n",
    "unicaa01_p12_panic__chatcmpl_9fcda159903f__d14": "incomplete body`.\n",
    "unicaa01_p14_panic__chatcmpl_fe69aa9a2766__d10": "incomplete matcher` by using a simple local-part@domain.tld pattern.\n\n",
    "unicaa01_p18_panic__chatcmpl_2ba03f73cbb9__d3": "Trie` insert and starts_with so prefix lookups work.\n\n",
    "unicaa01_p23_panic__chatcmpl_223fdd2deb20__d3": "max_val` knapsack path to return the optimal 0/1 value.\n",
    "unicaa01_p27_panic__chatcmpl_df7c0f5ed153__d3": "b64_encode` with a working base64 implementation.\n\n",
    "unicaa01_p30_panic__chatcmpl_d222a9ac16f1__d3": "INI parser` that splits key=value lines into a dict.\n",
    "unicaa01_p31_panic__chatcmpl_1b6f807a66ad__d11": "RuntimeError` — fold lines into a returned key/value dict.\n",
    "unicaa01_p33_panic__chatcmpl_5d81120833b1__d3": "strip_tags` to convert simple HTML tags to plain text.\n",
    "unicaa01_p34_panic__chatcmpl_0e4fa30316ca__d3": "query parser` that splits ampersand pairs into a dict.\n\n",
    "unicaa01_p37_panic__chatcmpl_4e836a4fb67c__d3": "SHA-256 chain` walked iteratively from the seed.\n\n",
    "unicaa01_p42_panic__chatcmpl_6d7feca070ed__d8": "acquire path` — return True only while permits remain.\n",
    "unicaa01_p45_panic__chatcmpl_e8dcf81d0af7__d8": "drain path` — empty the queue into a list.\n\n",
    "unicaa01_p48_panic__chatcmpl_4aabf5e536c8__d3": "put_if_absent` to track seen keys and reject repeats.\n\n",
    "unicaa01_p50_panic__chatcmpl_6f078e4d2153__d3": "drain` method to return pending events.\n",
    "unicaa01_p51_panic__chatcmpl_de8dd51a0f3e__d3": "memoize` by caching results in a dictionary.\n\n",
    "unicaa01_p54_panic__chatcmpl_1170b3ad6d58__d8": "event path` — a coin should unlock the turnstile.\n",
    "unicaa01_p62_panic__chatcmpl_55716bdd6c36__d3": "bisect_left` with a manual binary search.\n\n",
    "unicaa01_p65_panic__chatcmpl_b9ffc3b14249__d3": "interval merge` that collapses overlaps after sorting by start.\n\n",
}

DOC_FIXES: dict[str, str] = {
    "unicaa01_p04_panic__chatcmpl_2665d408d246__d49": '"""BFS visit order with a visited set."""',
    "unicaa01_p07_panic__chatcmpl_30255b182620__d3": '"""Sum the amount column from a small CSV string."""',
    "unicaa01_p09_panic__chatcmpl_ab4f64e17fde__d10": '"""Map a callable across items with a thread pool."""',
    "unicaa01_p10_panic__chatcmpl_ec2610a87318__d3": '"""Config validator for port range."""',
    "unicaa01_p12_panic__chatcmpl_9fcda159903f__d14": '"""Compute Fibonacci iteratively."""',
    "unicaa01_p14_panic__chatcmpl_fe69aa9a2766__d10": '"""Email address check via regex."""',
    "unicaa01_p18_panic__chatcmpl_2ba03f73cbb9__d3": '"""Trie with insert and prefix search."""',
    "unicaa01_p23_panic__chatcmpl_223fdd2deb20__d3": '"""0/1 knapsack via DP."""',
    "unicaa01_p27_panic__chatcmpl_df7c0f5ed153__d3": '"""Base64 encode with stdlib."""',
    "unicaa01_p30_panic__chatcmpl_d222a9ac16f1__d3": '"""Parse simple key=value INI lines."""',
    "unicaa01_p31_panic__chatcmpl_1b6f807a66ad__d11": '"""Parse flat key=value lines."""',
    "unicaa01_p33_panic__chatcmpl_5d81120833b1__d3": '"""Strip HTML tags to plain text."""',
    "unicaa01_p34_panic__chatcmpl_0e4fa30316ca__d3": '"""Parse URL query strings."""',
    "unicaa01_p37_panic__chatcmpl_4e836a4fb67c__d3": '"""Iterative SHA-256 hash chain."""',
    "unicaa01_p42_panic__chatcmpl_6d7feca070ed__d8": '"""Counting semaphore with non-blocking acquire."""',
    "unicaa01_p45_panic__chatcmpl_e8dcf81d0af7__d8": '"""Drain a queue into a list."""',
    "unicaa01_p48_panic__chatcmpl_4aabf5e536c8__d3": '"""Idempotency key store."""',
    "unicaa01_p50_panic__chatcmpl_6f078e4d2153__d3": '"""Transactional outbox drain."""',
    "unicaa01_p51_panic__chatcmpl_de8dd51a0f3e__d3": '"""Memoize decorator with a simple cache."""',
    "unicaa01_p54_panic__chatcmpl_1170b3ad6d58__d8": '"""Coin/pass turnstile FSM."""',
    "unicaa01_p62_panic__chatcmpl_55716bdd6c36__d3": '"""Leftmost insertion point via binary search."""',
    "unicaa01_p65_panic__chatcmpl_b9ffc3b14249__d3": '"""Merge overlapping intervals."""',
}


def _fix_docstring(code: str, new_doc: str | None) -> str:
    if not new_doc:
        # Generic cleanup of Broken docs
        code2 = re.sub(r'^"""Broken[^"]*"""', '"""Task fix."""', code, count=1)
        if code2 == code:
            code2 = re.sub(r"^'''Broken[^']*'''", "'''Task fix.'''", code, count=1)
        return code2
    code2 = re.sub(r'^"""[^"]*"""', new_doc, code, count=1)
    if code2 == code and not code.lstrip().startswith('"""'):
        return new_doc + "\n" + code
    return code2


def build_text(brief_id: str, suffix: str) -> str:
    brief = json.loads((BRIEFS / f"{brief_id}.json").read_text(encoding="utf-8"))
    prefix = brief["prefix_lock"]["prefix_text"]
    old = (TAILS / f"{brief_id}.txt").read_text(encoding="utf-8")
    code = extract_main_py(old)
    code = _fix_docstring(code, DOC_FIXES.get(brief_id))
    task = brief["output_contract"]["task_dir"]
    return f"{prefix}{suffix}{task}/main.py\n```python\n{code}\n```\n"


def apply_one(brief_id: str, suffix: str) -> dict:
    brief = json.loads((BRIEFS / f"{brief_id}.json").read_text(encoding="utf-8"))
    text = build_text(brief_id, suffix)
    path = TAILS / f"{brief_id}.txt"
    backup = path.read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    tok = get_qwen_tokenizer()
    try:
        build_calm_tail_from_file(
            prefix_token_ids=brief["prefix_lock"]["prefix_token_ids"],
            d=brief["prefix_lock"]["d"],
            calm_completion_path=path,
            tokenizer=tok,
        )
        result = verify_one(path)
    except Exception as exc:  # noqa: BLE001
        path.write_text(backup, encoding="utf-8")
        return {"id": brief_id, "pass": False, "error": str(exc)}
    if not result.get("pass"):
        path.write_text(backup, encoding="utf-8")
        try:
            build_calm_tail_from_file(
                prefix_token_ids=brief["prefix_lock"]["prefix_token_ids"],
                d=brief["prefix_lock"]["d"],
                calm_completion_path=path,
                tokenizer=tok,
            )
        except Exception:
            pass
        failed = [k for k, v in (result.get("checks") or {}).items() if not v]
        return {"id": brief_id, "pass": False, "failed_checks": failed, "detail": result.get("error")}
    # refresh meta narrative note
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["editorial_rewrite"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"id": brief_id, "pass": True}


def main() -> int:
    rows = []
    for brief_id, suffix in CONTINUATIONS.items():
        row = apply_one(brief_id, suffix)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))
    payload = {
        "n": len(rows),
        "n_pass": sum(1 for r in rows if r.get("pass")),
        "rows": rows,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / "G20_editorial_rewrite.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("PASS", payload["n_pass"], "/", payload["n"])
    return 0 if payload["n_pass"] == payload["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
