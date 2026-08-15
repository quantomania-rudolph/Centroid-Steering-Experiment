from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from divergence_support import get_qwen_tokenizer  # noqa: E402
from editor_rewrite_calm_tails import apply_one  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402

BID = "unicaa01_p43_panic__chatcmpl_0c1386eff216__d7"


def main() -> None:
    brief = json.loads(
        (ROOT / f"artifacts/caa/UNI_CAA01_corpus/curator_briefs/{BID}.json").read_text(encoding="utf-8")
    )
    path = ROOT / f"artifacts/caa/UNI_CAA01_corpus/calm_tails/{BID}.txt"
    code = extract_main_py(path.read_text(encoding="utf-8"))
    pref = brief["prefix_lock"]["prefix_text"]
    for suf in (
        "rwlock methods` — wire acquire/release under a condition variable.\n",
        "rwlock` acquire/release paths under a condition variable.\n",
        "missing body` — wire acquire/release under a condition variable.\n",
    ):
        text = f"{pref}{suf}{brief['output_contract']['task_dir']}/main.py\n```python\n{code}\n```\n"
        narr = validate_narrative_purity(
            full_text=text, prefix_text=pref, pair_id=brief["pair_id"]
        )
        bad = [(r.gate_id, r.detail) for r in narr if not r.pass_]
        print(repr(suf[:55]), "bad", bad)

    # Prefer methods` wording (avoids 'stubs').
    print(apply_one(BID, "rwlock methods` — wire acquire/release under a condition variable.\n"))


if __name__ == "__main__":
    main()
