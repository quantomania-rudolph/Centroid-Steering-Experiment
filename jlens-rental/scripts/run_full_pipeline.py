#!/usr/bin/env python3
"""
End-to-end rental pipeline with guaranteed Vast destroy in finally.

Order:
  1. prepare_bundle
  2. upload
  3. remote fit (with optional max wall-clock)
  4. download lens
  5. integrate locally (verify + readouts)
  finally: destroy Vast instance if AUTO_DESTROY=1 (default)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RENTAL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = RENTAL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from load_secrets import load_secrets  # noqa: E402


def run_step(name: str, script: str, *, timeout: float | None = None) -> int:
    cmd = [sys.executable, str(SCRIPTS / script)]
    print(f"\n===== {name} =====\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(RENTAL_ROOT), timeout=timeout)


def destroy(*, reason: str) -> int:
    print(f"\n===== DESTROY Vast ({reason}) =====", flush=True)
    return subprocess.call(
        [sys.executable, str(SCRIPTS / "destroy_vast_instance.py")],
        cwd=str(RENTAL_ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Bundle already on instance",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Fit already done; download + integrate only",
    )
    parser.add_argument(
        "--no-destroy",
        action="store_true",
        help="Do not destroy instance (dangerous for billing)",
    )
    parser.add_argument(
        "--destroy-only",
        action="store_true",
        help="Only destroy instance and exit",
    )
    args = parser.parse_args()

    secrets = load_secrets()
    auto_destroy = secrets.get("AUTO_DESTROY", "1").strip() not in ("0", "false", "False")
    if args.no_destroy:
        auto_destroy = False

    if args.destroy_only:
        return destroy(reason="destroy-only")

    max_min = float(secrets.get("MAX_BILLING_MINUTES", "90") or "90")
    fit_timeout_s = max_min * 60.0

    destroy_flag = {"done": False}
    watchdog_stop = threading.Event()

    def watchdog() -> None:
        # Hard ceiling: destroy even if SSH hangs
        if not watchdog_stop.wait(timeout=fit_timeout_s + 600):
            print(
                f"\nWATCHDOG: exceeded MAX_BILLING_MINUTES={max_min} + 10m grace — destroying",
                flush=True,
            )
            if auto_destroy and not destroy_flag["done"]:
                destroy(reason="watchdog")
                destroy_flag["done"] = True

    wd = threading.Thread(target=watchdog, daemon=True)
    if auto_destroy:
        wd.start()

    rc = 1
    try:
        rc = run_step("prepare_bundle", "prepare_bundle.py")
        if rc != 0:
            pass
        elif not args.skip_upload:
            rc = run_step("upload", "upload_to_instance.py")

        if rc == 0 and not args.skip_fit:
            print(
                f"Remote fit starting (timeout {max_min:.0f} min from MAX_BILLING_MINUTES)...",
                flush=True,
            )
            try:
                rc = run_step("remote_fit", "run_remote_fit.py", timeout=fit_timeout_s)
            except subprocess.TimeoutExpired:
                print("Fit timed out — will still try download then destroy.", flush=True)
                rc = 124
            if rc not in (0, 124):
                print(f"Fit exited {rc}; attempting download anyway...", flush=True)

        if rc in (0, 124) or (RENTAL_ROOT / "output" / "qwen_jacobian_lens.pt").is_file():
            dl = run_step("download", "download_from_instance.py")
            if dl != 0 and rc == 0:
                rc = dl

        if (RENTAL_ROOT / "output" / "qwen_jacobian_lens.pt").is_file():
            integ = run_step("integrate_local", "integrate_local.py")
            if integ != 0:
                rc = integ
        elif rc == 0:
            print("No lens file downloaded — integrate skipped.", flush=True)
            rc = 2
    finally:
        watchdog_stop.set()
        if auto_destroy and not destroy_flag["done"]:
            d_rc = destroy(reason="pipeline finally")
            destroy_flag["done"] = True
            if d_rc != 0:
                print(
                    "CRITICAL: auto-destroy failed — open Vast console and Destroy manually NOW.",
                    flush=True,
                )
                if rc == 0:
                    rc = d_rc
        elif not auto_destroy:
            print(
                "AUTO_DESTROY disabled — YOU must destroy the instance in Vast console.",
                flush=True,
            )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
