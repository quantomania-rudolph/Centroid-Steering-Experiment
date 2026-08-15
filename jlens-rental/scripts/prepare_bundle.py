#!/usr/bin/env python3
"""Zip prompts + fit scripts for upload to GPU instance."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RENTAL_ROOT.parent
BUNDLE_DIR = RENTAL_ROOT / "bundle"
BUNDLE_ZIP = BUNDLE_DIR / "rental_bundle.zip"


def main() -> int:
    staging = BUNDLE_DIR / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    prompts_src = PROJECT_ROOT / "data" / "prompts" / "corpus_100.json"
    if not prompts_src.is_file():
        print(f"Missing prompts: {prompts_src}", flush=True)
        return 1

    def copy_text_lf(src: Path, dst: Path) -> None:
        data = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        dst.write_bytes(data)

    shutil.copy2(prompts_src, staging / "corpus_100.json")
    copy_text_lf(RENTAL_ROOT / "fit_config.yaml", staging / "fit_config.yaml")
    copy_text_lf(RENTAL_ROOT / "fit_on_gpu.py", staging / "fit_on_gpu.py")
    copy_text_lf(RENTAL_ROOT / "checkpoint_fit.py", staging / "checkpoint_fit.py")
    copy_text_lf(RENTAL_ROOT / "run_fit.sh", staging / "run_fit.sh")

    secrets_src = RENTAL_ROOT / "secrets" / "secrets.env"
    if secrets_src.is_file():
        secrets_staging = staging / "secrets"
        secrets_staging.mkdir()
        copy_text_lf(secrets_src, secrets_staging / "secrets.env")
        print("Included secrets/secrets.env in bundle (HF_TOKEN for remote login, LF).")
    else:
        print("No secrets.env — upload HF_TOKEN manually on the instance.")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    print(f"Bundle: {BUNDLE_ZIP} ({BUNDLE_ZIP.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
