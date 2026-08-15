#!/usr/bin/env python
"""Generate L{layer}_readout.txt/json for each layer in a fitted lens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument("--layers", default="8-26", help="e.g. 8-26")
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "corpus.txt")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "readouts" / "production",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "readouts" / "production_manifest.json",
    )
    args = parser.parse_args()

    if not args.lens.exists():
        print(f"Lens not found: {args.lens}", file=sys.stderr)
        return 1

    lo, hi = [int(x) for x in args.layers.split("-")]
    layers = list(range(lo, hi + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for layer in layers:
        print(f"=== layer {layer} ===")
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "show_layer_readout.py"),
                "--lens",
                str(args.lens),
                "--layer",
                str(layer),
                "--prompts",
                str(args.prompts),
                "--prompt-index",
                str(args.prompt_index),
                "--max-seq-len",
                str(args.max_seq_len),
                "--out-dir",
                str(args.out_dir),
            ],
            cwd=str(ROOT),
        )
        ok = proc.returncode == 0
        results.append(
            {
                "layer": layer,
                "ok": ok,
                "txt": str(args.out_dir / f"L{layer}_readout.txt"),
                "json": str(args.out_dir / f"L{layer}_readout.json"),
            }
        )
        if not ok:
            print(f"readout failed for layer {layer}", file=sys.stderr)

    manifest = {
        "lens": str(args.lens),
        "layers": layers,
        "prompts": str(args.prompts),
        "prompt_index": args.prompt_index,
        "out_dir": str(args.out_dir),
        "results": results,
        "all_ok": all(r["ok"] for r in results),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest {args.manifest}")
    return 0 if manifest["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
