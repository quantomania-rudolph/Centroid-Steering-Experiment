#!/usr/bin/env python
"""Calm-tail CLI — validate curated calm completions (no free-gen)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.calm_tail import (  # noqa: E402
    build_calm_tail_from_ids,
    reject_free_generate,
    validate_prefix_lock,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate prefix-locked calm tails")
    ap.add_argument("--prefix-json", type=Path, required=True, help="JSON list of prefix token ids")
    ap.add_argument("--calm-ids-json", type=Path, help="JSON list of full calm completion token ids")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--free-generate", action="store_true", help="Must fail (forbidden)")
    args = ap.parse_args()

    if args.free_generate:
        try:
            reject_free_generate()
        except NotImplementedError as e:
            print(json.dumps({"pass": False, "error": str(e)}))
            return 1

    prefix = json.loads(args.prefix_json.read_text(encoding="utf-8"))
    if not args.calm_ids_json:
        print("need --calm-ids-json", file=sys.stderr)
        return 1
    calm = json.loads(args.calm_ids_json.read_text(encoding="utf-8"))
    gate = validate_prefix_lock(calm, prefix, args.d)
    if not gate["pass"]:
        print(json.dumps(gate, indent=2))
        return 1
    spec = build_calm_tail_from_ids(
        prefix_token_ids=prefix, calm_full_token_ids=calm, d=args.d
    )
    print(json.dumps(spec.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
