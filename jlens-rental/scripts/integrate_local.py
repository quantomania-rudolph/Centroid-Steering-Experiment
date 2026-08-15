#!/usr/bin/env python3
"""Copy rental lens into project artifacts, production-verify, run readouts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RENTAL_ROOT.parent
SCRIPTS = RENTAL_ROOT / "scripts"
LENS_SRC = RENTAL_ROOT / "output" / "qwen_jacobian_lens.pt"
LENS_DST = PROJECT_ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt"


def main() -> int:
    if not LENS_SRC.is_file():
        print(f"Missing {LENS_SRC}. Run download_from_instance.py first.", flush=True)
        return 1

    LENS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LENS_SRC, LENS_DST)
    print(f"Copied lens -> {LENS_DST}", flush=True)

    meta_src = RENTAL_ROOT / "output" / "fit_meta.json"
    if meta_src.is_file():
        meta_dst = PROJECT_ROOT / "artifacts" / "gates" / "rental_fit_meta.json"
        shutil.copy2(meta_src, meta_dst)
        print(f"Copied metadata -> {meta_dst}", flush=True)

    verify = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "verify_production_lens.py"),
            "--lens",
            str(LENS_DST),
            "--layers",
            "8-26",
            "--out",
            str(PROJECT_ROOT / "artifacts" / "gates" / "lens_production_verify.json"),
        ],
        cwd=str(RENTAL_ROOT),
    )
    if verify.returncode != 0:
        print("Production lens verify FAILED — not running readouts.", flush=True)
        return verify.returncode
    print("Production lens verify PASSED.", flush=True)

    readouts = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_all_layer_readouts.py"),
            "--lens",
            str(LENS_DST),
            "--layers",
            "8-26",
        ],
        cwd=str(PROJECT_ROOT),
    )
    return readouts.returncode


if __name__ == "__main__":
    raise SystemExit(main())
