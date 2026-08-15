"""Debug why sum_amount suffix trips ASM_no_bridge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import assemble_calm_completion, extract_main_py  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402

BID = "unicaa01_p07_panic__chatcmpl_30255b182620__d3"


def enc(tok, text: str) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)
    return list(ids) if not hasattr(ids, "tolist") else ids.tolist()


def main() -> None:
    brief = json.loads(
        (ROOT / f"artifacts/caa/UNI_CAA01_corpus/curator_briefs/{BID}.json").read_text(encoding="utf-8")
    )
    path = ROOT / f"artifacts/caa/UNI_CAA01_corpus/calm_tails/{BID}.txt"
    code = extract_main_py(path.read_text(encoding="utf-8"))
    pref = brief["prefix_lock"]["prefix_text"]
    d = brief["prefix_lock"]["d"]
    prefix_ids = brief["prefix_lock"]["prefix_token_ids"]
    bad = brief["divergence_context"]["bad_token_at_d"]
    bad_text = brief["divergence_context"]["bad_token_at_d_text"]
    tok = get_qwen_tokenizer()

    for label, suf in [
        ("stub", "stub` for sum_amount now parses the CSV and sums the amount column.\n\n"),
        ("sum", "sum_amount` that parses the CSV and sums the amount column.\n\n"),
        ("csv", "CSV parser` that sums the amount column.\n\n"),
    ]:
        body = f"{suf}{brief['output_contract']['task_dir']}/main.py\n```python\n{code}\n```\n"
        ids = enc(tok, pref + body)
        print(
            label,
            "pref_ok",
            ids[:d] == prefix_ids,
            "at_d",
            ids[d] if len(ids) > d else None,
            tok.decode([ids[d]]) if len(ids) > d else None,
            "is_bad",
            ids[d] == bad if len(ids) > d else None,
            "bad",
            bad,
            bad_text,
        )
        try:
            assembled = assemble_calm_completion(
                prefix_token_ids=prefix_ids,
                prefix_text=pref,
                suffix_text=body,
                tokenizer=tok,
                d=d,
                bad_token_at_d=bad,
                bad_token_at_d_text=bad_text,
                strict=True,
                allow_bridge=False,
            )
            print("  assemble ok bridge", assembled.bridge_used)
        except Exception as exc:  # noqa: BLE001
            print("  assemble FAIL", exc)


if __name__ == "__main__":
    main()
