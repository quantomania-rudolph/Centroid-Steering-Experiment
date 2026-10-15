"""Preflight checks before CAST alpha0 live harness runs."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jlens_qwen.caa.cast_panic_stubs import (
    PANIC_SLUGS_CONFIRM,
    PANIC_SLUGS_FIT,
    restore_all_panic_stubs,
    restore_panic_stub,
)

ROOT_PANIC_GLOB = "unicaa01_*_panic"

# Orphan lab workers (not the harness itself — tree-kill on Windows can take down siblings).
_STRAY_CMD_MARKERS: tuple[tuple[str, bool], ...] = (
    ("serve_qwen_openai.py", True),
    ("run_aider_agent.py", True),
    ("-m aider", True),
)


@dataclass
class PreflightResult:
    ok: bool
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _purge_repo_root_panic_dirs(repo_root: Path) -> list[str]:
    actions: list[str] = []
    for path in sorted(repo_root.glob(ROOT_PANIC_GLOB)):
        if not path.is_dir():
            continue
        shutil.rmtree(path)
        actions.append(f"removed {path}")
    return actions


def _win_process_rows() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine } | "
        "Select-Object ProcessId, ParentProcessId, CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            timeout=30,
        ).strip()
        if not out:
            return []
        data = json.loads(out)
        return [data] if isinstance(data, dict) else list(data)
    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return []


def list_stray_lab_pids(*, repo_root: Path) -> list[int]:
    """PIDs for orphan serve_qwen / aider workers under this repo."""
    repo = str(repo_root.resolve()).lower()
    my_pid = os.getpid()
    pids: list[int] = []
    for row in _win_process_rows():
        cmd = str(row.get("CommandLine") or "")
        pid = int(row.get("ProcessId") or 0)
        if pid <= 0 or pid == my_pid:
            continue
        cmd_l = cmd.lower()
        if repo not in cmd_l:
            continue
        if any(marker.lower() in cmd_l for marker, _ in _STRAY_CMD_MARKERS):
            pids.append(pid)
    return sorted(set(pids))


def _kill_port(port: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
                f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}",
            ],
            check=False,
            capture_output=True,
        )
    else:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, capture_output=True)


def _taskkill_pid(pid: int, *, tree: bool = True) -> bool:
    args = ["taskkill", "/PID", str(pid)]
    if tree:
        args.append("/T")
    args.append("/F")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    out = f"{result.stdout or ''}{result.stderr or ''}".lower()
    if result.returncode == 0:
        return True
    return "no running instance" in out or "not found" in out


def release_local_xpu_cache() -> None:
    """Best-effort XPU cache release in the calling process (does not free other PIDs)."""
    try:
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


def kill_stray_lab_processes(
    *,
    repo_root: Path,
    port: int = 8767,
    max_rounds: int = 3,
    settle_s: float = 2.0,
) -> list[str]:
    """Stop port server and orphan aider/serve workers; verify PIDs are gone."""
    actions: list[str] = []
    _kill_port(port)
    actions.append(f"kill_port({port})")

    if sys.platform != "win32":
        return actions

    for round_i in range(max_rounds):
        pids = list_stray_lab_pids(repo_root=repo_root)
        if not pids:
            actions.append("verified_no_stray_lab_processes" if round_i else "no_stray_lab_processes")
            break
        actions.append(f"round={round_i + 1} targets={pids}")
        _kill_port(port)
        # Parent venv launcher + child interpreter: tree-kill every match, parents first.
        for pid in sorted(pids):
            ok = _taskkill_pid(pid, tree=True)
            actions.append(f"taskkill pid={pid} ok={ok}")
            if not ok:
                actions.append(f"WARN: taskkill failed for pid={pid}")
        time.sleep(settle_s)
    else:
        pids = list_stray_lab_pids(repo_root=repo_root)
        if pids:
            actions.append(f"ERROR: stray_pids_still_alive={pids}")

    release_local_xpu_cache()
    return actions


def run_preflight(
    *,
    repo_root: Path,
    qwen_code: Path,
    extra_slugs: tuple[str, ...] = (),
    min_alphas: int = 3,
    alphas: list[float] | None = None,
    require_stubs: bool = True,
) -> PreflightResult:
    """Restore stubs, purge stray dirs, validate layout. No server check (harness does that)."""
    result = PreflightResult(ok=True)
    result.actions.extend(kill_stray_lab_processes(repo_root=repo_root))
    result.actions.extend(_purge_repo_root_panic_dirs(repo_root))

    double_path = qwen_code / "qwen code"
    if double_path.is_dir():
        shutil.rmtree(double_path)
        result.actions.append(f"removed {double_path}")

    slugs = tuple(dict.fromkeys((*PANIC_SLUGS_FIT, *PANIC_SLUGS_CONFIRM, *extra_slugs)))
    if require_stubs:
        restore_all_panic_stubs(qwen_code)
        for slug in slugs:
            stub = qwen_code / slug / "main.py"
            if not stub.is_file():
                result.ok = False
                result.errors.append(f"missing stub {stub}")
                continue
            if "NotImplementedError" not in stub.read_text(encoding="utf-8"):
                result.ok = False
                result.errors.append(f"stub not broken: {stub}")

    if alphas is not None and len(alphas) < min_alphas:
        result.ok = False
        result.errors.append(f"alphas grid needs >={min_alphas} values, got {alphas}")

    stray = list_stray_lab_pids(repo_root=repo_root)
    if stray:
        result.ok = False
        result.errors.append(f"stray lab processes still alive: {stray}")

    return result


def assert_aider_path_under_qwen_code(path: Path, qwen_code: Path) -> None:
    resolved = path.resolve()
    base = qwen_code.resolve()
    if base not in resolved.parents and resolved != base:
        raise ValueError(f"aider path must be under {base}, got {resolved}")
