#!/usr/bin/env python
"""Restart Qwen OpenAI server on XPU (OOM recovery between corpus batches)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVE = ROOT / "experiments" / "serve_qwen_openai.py"
PORT = 8767
SERVER_LOG = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "server_shadow.log"
# Leave ~4–5 GiB on Arc 140V for KV cache / activations during chaos decode.
SERVER_ENV = {
    **os.environ,
    "JLENS_XPU_MAX_MEMORY": os.environ.get("JLENS_XPU_MAX_MEMORY", "11GiB"),
    "JLENS_CPU_MAX_MEMORY": os.environ.get("JLENS_CPU_MAX_MEMORY", "14GiB"),
    "PYTORCH_XPU_ALLOC_CONF": os.environ.get("PYTORCH_XPU_ALLOC_CONF", "max_split_size_mb:128"),
}


def _kill_port() -> None:
    if sys.platform == "win32":
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-NetTCPConnection -LocalPort {PORT} -ErrorAction SilentlyContinue | "
                f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}",
            ],
            check=False,
            capture_output=True,
        )
    else:
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"], check=False, capture_output=True)


def stop_server(*, settle_s: float = 5.0) -> None:
    """Kill the Qwen server and wait for XPU memory to settle."""
    print("Stopping Qwen server...", flush=True)
    _kill_port()
    time.sleep(settle_s)
    try:
        import gc

        import torch

        gc.collect()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
    except Exception:
        pass


def _health_ok(
    h: dict,
    *,
    cast_shadow: bool | None = None,
    cast_inject: bool | None = None,
    device: str | None = None,
) -> bool:
    if not h.get("ok"):
        return False
    if cast_shadow is not None and h.get("cast_shadow") is not cast_shadow:
        return False
    if cast_inject is not None and h.get("cast_inject") is not cast_inject:
        return False
    if device is not None and str(h.get("device")) != device:
        return False
    return True


def fetch_health() -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def wait_health(
    timeout_s: float = 300.0,
    *,
    cast_shadow: bool | None = None,
    cast_inject: bool | None = None,
    device: str | None = None,
) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = fetch_health()
        if h is not None and _health_ok(
            h,
            cast_shadow=cast_shadow,
            cast_inject=cast_inject,
            device=device,
        ):
            return True
        time.sleep(3)
    return False


def restart_server(
    *,
    block: bool = True,
    device: str = "xpu",
    shadow: bool = False,
    inject: bool = False,
    cast_shadow: bool | None = None,
    cast_inject: bool | None = None,
    alpha_0: float | None = None,
) -> bool:
    alpha_label = f" alpha_0={alpha_0}" if alpha_0 is not None else ""
    label = (
        f"XPU shadow={shadow} inject={inject}{alpha_label}"
        if shadow or inject
        else "XPU"
    )
    print(f"Restarting Qwen server ({label})...", flush=True)
    stop_server(settle_s=3.0)
    env = dict(SERVER_ENV)
    if shadow:
        env["CAA_CAST_SHADOW"] = "1"
        env["CAA_CAST_INJECT"] = "1" if inject else "0"
    if alpha_0 is not None:
        env["CAA_ALPHA_0"] = str(alpha_0)
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(SERVER_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(PY), str(SERVE), "--device", device, "--port", str(PORT), "--chaos"],
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if not block:
        return True
    ok = wait_health(
        cast_shadow=cast_shadow if cast_shadow is not None else (True if shadow else None),
        cast_inject=cast_inject if cast_inject is not None else (False if shadow and not inject else None),
        device=device,
    )
    if not ok:
        proc.kill()
    print(f"Server ready={ok} pid={proc.pid}", flush=True)
    return ok


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Restart Qwen OpenAI server on XPU")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--shadow", action="store_true", help="Enable CAST shadow (CAA_CAST_SHADOW=1)")
    parser.add_argument("--no-inject", action="store_true", help="Disable CAST inject (CAA_CAST_INJECT=0)")
    parser.add_argument(
        "--alpha0",
        type=float,
        default=None,
        help="Override CAA_ALPHA_0 for CAST live session (sweep / proof)",
    )
    args = parser.parse_args()
    inject = not args.no_inject if args.shadow else False
    return (
        0
        if restart_server(
            device=args.device,
            shadow=args.shadow,
            inject=inject,
            alpha_0=args.alpha0,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
