"""Audit admitted calm tails for prose slop."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TAILS = ROOT / "artifacts/caa/UNI_CAA01_corpus/calm_tails"


def main() -> None:
    rows = []
    for p in sorted(TAILS.glob("unicaa01_p*_panic__*.txt")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^```", text, re.M)
        narr = (text[: m.start()] if m else text[:200]).strip()
        code = text[m.end() :] if m else ""
        flags: list[str] = []
        if "`stub`" in narr or re.search(r"\bstub\b", narr, re.I):
            flags.append("stub_word")
        if re.search(r'"""Broken', code):
            flags.append("broken_doc")
        if re.search(r"placeholder|NotImplemented", narr, re.I):
            flags.append("placeholder")
        if len(narr) > 220:
            flags.append("long_narr")
        if narr.count("`") % 2:
            flags.append("odd_ticks")
        if re.search(r"raises a `stub`|the `stub`", narr):
            flags.append("the_stub")
        if "missing backtick" in narr.lower():
            flags.append("meta_backtick")
        rows.append((len(flags), p.name, flags, narr.replace("\n", " | ")[:200]))
    rows.sort(reverse=True)
    for n, name, flags, narr in rows:
        if flags:
            print(f"{name}\n  {flags}\n  {narr}\n")
    print(f"flagged {sum(1 for r in rows if r[0])}/{len(rows)}")


if __name__ == "__main__":
    main()
