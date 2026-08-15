#!/usr/bin/env python
"""Wait for L15 smoke fit, verify, show layer readout, optionally launch production."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens_L15.pt",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument(
        "--skip-readout",
        action="store_true",
        help="Skip human-readable layer readout after verify",
    )
    parser.add_argument(
        "--launch-production",
        action="store_true",
        help="After verify + readout, start 0-26 x 100 prompts fit",
    )
    args = parser.parse_args()

    print(f"Waiting for {args.lens} ...")
    while not args.lens.exists():
        time.sleep(args.poll_sec)
        print(f"  still waiting ({args.lens.name} not found)")

    print("Lens file found. Running smoke verification...")
    verify = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_lens_smoke.py"),
            "--lens",
            str(args.lens),
            "--layer",
            str(args.layer),
        ]
    )
    if verify.returncode != 0:
        print("Smoke verification FAILED.")
        return 1

    print("Smoke verification PASSED.")

    if not args.skip_readout:
        print(f"Generating layer {args.layer} readout for one prompt...")
        readout = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "show_layer_readout.py"),
                "--lens",
                str(args.lens),
                "--layer",
                str(args.layer),
            ]
        )
        if readout.returncode != 0:
            print("Layer readout FAILED — production blocked.", file=sys.stderr)
            return 1
        print("Layer readout written to artifacts/readouts/")

    if not args.launch_production:
        print(
            "Production NOT started (requires explicit approval). "
            "When ready: python scripts/run_production_fit.py"
        )
        return 0

    prod = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_production_fit.py"),
            "--smoke-lens",
            str(args.lens),
        ]
    )
    return prod.returncode


if __name__ == "__main__":
    raise SystemExit(main())
