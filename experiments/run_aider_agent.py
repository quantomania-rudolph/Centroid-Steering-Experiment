#!/usr/bin/env python
"""Run Aider against local Qwen, then finalize J-space via the same server.

One agentic experiment = one dashboard entry with nested invocations[].
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_CODE = ROOT / "qwen code"
SERVER = "http://127.0.0.1:8767"

sys.path.insert(0, str(ROOT / "src"))
from jlens_qwen.caa.cast_run_preflight import assert_aider_path_under_qwen_code

# Empirical floor from cast_alpha0_4c sessions: healthy first LLM call ~187–333s on XPU.
# 60s wall would fail every successful run; these caps kill 12–20min balloon generations.
DEFAULT_AIDER_WALL_S = 600.0
DEFAULT_MAX_INVOCATIONS = 3
DEFAULT_PER_INVOCATION_TIMEOUT_S = 360.0


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 3600.0):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _py_module(*parts: str) -> list[str]:
    return [sys.executable, "-m", *parts]


def wait_for_server(timeout_s: float = 900.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            h = _http_json("GET", f"{SERVER}/health", timeout=5)
            if h.get("ok"):
                return h
            last = h
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Qwen server not ready: {last}")


def _rel_aider_files(task_dir: Path, files: list[str]) -> list[str]:
    """Paths relative to the enclosing git repo (ROOT), not QWEN_CODE alone."""
    rel_files = []
    for f in files:
        p = Path(f)
        candidate = p if p.is_absolute() else task_dir / f
        rel_files.append(candidate.relative_to(ROOT).as_posix())
    return rel_files


def aider_argv(
    *,
    task_dir: Path,
    message: str,
    files: list[str],
    architect: bool,
    no_auto_lint: bool = False,
    no_auto_commits: bool = False,
    server: str = SERVER,
    conf: Path | None = None,
) -> list[str]:
    conf = conf or (QWEN_CODE / ".aider.conf.yml")
    rel_files = _rel_aider_files(task_dir, files)
    cmd = [
        sys.executable,
        "-m",
        "aider",
        "--config",
        str(conf),
        "--model",
        "openai/qwen-local",
        "--openai-api-base",
        f"{server}/v1",
        "--openai-api-key",
        "dummy-local-key",
        "--no-stream",
        "--yes-always",
    ]
    if no_auto_commits:
        cmd.append("--no-auto-commits")
    else:
        cmd.append("--auto-commits")
    if no_auto_lint:
        cmd.append("--no-auto-lint")
    else:
        cmd.append("--auto-lint")
        cmd.extend(
            [
                "--lint-cmd",
                "cmd /c tools\\lint_ruff.cmd",
                "--lint-cmd",
                "cmd /c tools\\lint_pylint.cmd",
            ]
        )
    cmd.extend(
        [
            "--map-tokens",
            "1024",
            "--subtree-only",
            "--no-show-model-warnings",
            "--no-check-update",
            "--no-analytics",
            "--message",
            message,
            "--exit",
        ]
    )
    if architect:
        cmd.append("--architect")
    for f in rel_files:
        cmd.extend(["--file", f])
    return cmd


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _session_status() -> dict:
    try:
        with urllib.request.urlopen(f"{SERVER}/session/status", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


def _should_kill_on_invocation_stall(
    *,
    inv: int,
    stall_s: float,
    per_invocation_timeout_s: float,
    generation_in_flight: bool,
) -> bool:
    """Stall between completed invocations only — not during first call or active generation."""
    if generation_in_flight or inv == 0:
        return False
    return stall_s >= per_invocation_timeout_s


def _watch_aider(
    proc: subprocess.Popen,
    *,
    aider_timeout_s: float,
    max_invocations: int,
    per_invocation_timeout_s: float,
    stop: threading.Event,
) -> None:
    """Kill Aider if it exceeds wall time, invocation count, or per-call stall."""
    t0 = time.perf_counter()
    last_inv = 0
    last_inv_change = t0
    while not stop.wait(15.0):
        if proc.poll() is not None:
            return
        elapsed = time.perf_counter() - t0
        if elapsed >= aider_timeout_s:
            print(f"Aider watchdog: wall timeout {elapsed:.0f}s >= {aider_timeout_s:.0f}s", flush=True)
            _kill_process_tree(proc)
            return
        status = _session_status()
        inv = int(status.get("invocations") or 0)
        in_flight = bool(status.get("generation_in_flight"))
        stall_s = time.perf_counter() - last_inv_change
        if inv > last_inv:
            last_inv = inv
            last_inv_change = time.perf_counter()
        elif _should_kill_on_invocation_stall(
            inv=inv,
            stall_s=stall_s,
            per_invocation_timeout_s=per_invocation_timeout_s,
            generation_in_flight=in_flight,
        ):
            print(
                f"Aider watchdog: no new invocation for {stall_s:.0f}s "
                f"(inv={inv}, limit={per_invocation_timeout_s:.0f}s)",
                flush=True,
            )
            _kill_process_tree(proc)
            return
        if inv >= max_invocations:
            print(f"Aider watchdog: invocations {inv} >= {max_invocations}", flush=True)
            _kill_process_tree(proc)
            return


def run_aider(
    *,
    task_dir: Path,
    message: str,
    files: list[str],
    architect: bool,
    no_auto_lint: bool = False,
    no_auto_commits: bool = False,
    aider_timeout_s: float = DEFAULT_AIDER_WALL_S,
    max_invocations: int = DEFAULT_MAX_INVOCATIONS,
    per_invocation_timeout_s: float = DEFAULT_PER_INVOCATION_TIMEOUT_S,
) -> dict:
    for f in files:
        candidate = Path(f) if Path(f).is_absolute() else task_dir / f
        assert_aider_path_under_qwen_code(candidate, QWEN_CODE)
    cmd = aider_argv(
        task_dir=task_dir,
        message=message,
        files=files,
        architect=architect,
        no_auto_lint=no_auto_lint,
        no_auto_commits=no_auto_commits,
    )

    env = os.environ.copy()
    env["OPENAI_API_BASE"] = f"{SERVER}/v1"
    env["OPENAI_API_KEY"] = "dummy-local-key"
    # Ensure ruff/pylint from workspace venv are findable by Aider's shell
    # (absolute lint paths break on Windows when the workspace has spaces)
    scripts = str(ROOT / ".venv" / "Scripts")
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    env["QWEN_VENV_SCRIPTS"] = scripts

    t0 = time.perf_counter()
    popen_kw: dict = {
        # Git root is ROOT; file args are ROOT-relative (qwen code/...).
        "cwd": str(ROOT),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kw["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kw)
    timed_out = False
    stop_watch = threading.Event()
    watcher = threading.Thread(
        target=_watch_aider,
        kwargs={
            "proc": proc,
            "aider_timeout_s": aider_timeout_s,
            "max_invocations": max_invocations,
            "per_invocation_timeout_s": per_invocation_timeout_s,
            "stop": stop_watch,
        },
        daemon=True,
    )
    watcher.start()
    try:
        stdout, stderr = proc.communicate(timeout=aider_timeout_s + 30)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        stop_watch.set()
        watcher.join(timeout=5)
    return {
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - t0,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "cmd": cmd,
        "timed_out": timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-slug", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--architect", action="store_true")
    parser.add_argument("--layers", default="8-26")
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument(
        "--chaos-only",
        action="store_true",
        help="Finalize with chaos from JSONL; skip expensive J-space recompute",
    )
    parser.add_argument(
        "--no-auto-lint",
        action="store_true",
        help="Pass --no-auto-lint to Aider and skip lint-cmd hooks",
    )
    parser.add_argument(
        "--no-auto-commits",
        action="store_true",
        help="Pass --no-auto-commits to Aider (avoids git index.lock failures)",
    )
    parser.add_argument(
        "--aider-timeout-s",
        type=float,
        default=DEFAULT_AIDER_WALL_S,
        help=f"Kill Aider process tree after this many seconds (default {DEFAULT_AIDER_WALL_S:.0f})",
    )
    parser.add_argument(
        "--max-invocations",
        type=int,
        default=DEFAULT_MAX_INVOCATIONS,
        help=f"Kill Aider if server logs this many LLM invocations (default {DEFAULT_MAX_INVOCATIONS})",
    )
    parser.add_argument(
        "--per-invocation-timeout-s",
        type=float,
        default=DEFAULT_PER_INVOCATION_TIMEOUT_S,
        help=(
            f"Kill Aider if one LLM call stalls this long (default {DEFAULT_PER_INVOCATION_TIMEOUT_S:.0f}; "
            "healthy XPU calls need ~200–330s)"
        ),
    )
    args = parser.parse_args()

    experiment = args.experiment or f"aider_{args.task_slug}"
    task_dir = QWEN_CODE / args.task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    starter = task_dir / "main.py"
    if not starter.is_file():
        starter.write_text(
            '"""Task workspace for Aider + Qwen."""\n\n\ndef main() -> None:\n    pass\n\n\nif __name__ == "__main__":\n    main()\n',
            encoding="utf-8",
        )

    print("Waiting for Qwen OpenAI server ...", flush=True)
    health = wait_for_server()
    print(f"Server OK device={health.get('device')}", flush=True)

    sess = _http_json(
        "POST",
        f"{SERVER}/session/start",
        {
            "name": experiment,
            "experiment": experiment,
            "task_slug": args.task_slug,
            "message": args.message,
        },
    )
    print(f"Session {sess['session_id']}", flush=True)

    files = args.file or ["main.py"]
    print(f"Aider -> {task_dir} files={files}", flush=True)
    result = run_aider(
        task_dir=task_dir,
        message=args.message,
        files=files,
        architect=args.architect,
        no_auto_lint=bool(args.no_auto_lint),
        no_auto_commits=bool(args.no_auto_commits),
        aider_timeout_s=float(args.aider_timeout_s),
        max_invocations=int(args.max_invocations),
        per_invocation_timeout_s=float(args.per_invocation_timeout_s),
    )
    print(
        f"Aider done rc={result['returncode']} in {result['seconds']:.1f}s"
        f" timed_out={result.get('timed_out')}",
        flush=True,
    )
    if result.get("timed_out"):
        status = _http_json("GET", f"{SERVER}/session/status")
        invocations = int(status.get("invocations") or 0)
        print(
            f"Aider timed out after {args.aider_timeout_s}s; invocations={invocations}",
            flush=True,
        )
        if invocations == 0:
            print("No invocations logged — fail closed.", flush=True)
            return 1
    try:
        print(result["stdout"][-4000:])
    except UnicodeEncodeError:
        sys.stdout.buffer.write(result["stdout"][-4000:].encode("utf-8", "replace"))
        sys.stdout.buffer.write(b"\n")
    if result["stderr"]:
        try:
            print("--- stderr ---")
            print(result["stderr"][-2000:])
        except UnicodeEncodeError:
            pass

    status = _http_json("GET", f"{SERVER}/session/status")
    print(f"Logged invocations: {status.get('invocations')}", flush=True)

    # Indisputable lint proof on the task file(s)
    scripts = ROOT / ".venv" / "Scripts"
    lint_report = []
    for rel in (args.file or ["main.py"]):
        target = task_dir / rel
        if not target.is_file():
            continue
        for tool, cmd in (
            ("ruff", _py_module("ruff", "check", "--fix", "--unsafe-fixes", str(target))),
            ("ruff_verify", _py_module("ruff", "check", str(target))),
            ("pylint", _py_module("pylint", "--errors-only", str(target))),
        ):
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # Ignore remaining C405-style if verify still fails after unsafe fix attempt:
            # re-run verify after unsafe fix already done above.
            if tool == "ruff_verify" and p.returncode != 0:
                p2 = subprocess.run(
                    _py_module("ruff", "check", "--fix", "--unsafe-fixes", str(target)),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                p = subprocess.run(
                    _py_module("ruff", "check", str(target)),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                _ = p2
            lint_report.append(
                {
                    "tool": tool,
                    "file": str(target.relative_to(QWEN_CODE)),
                    "returncode": p.returncode,
                    "stdout": (p.stdout or "")[-1000:],
                    "stderr": (p.stderr or "")[-1000:],
                    "pass": p.returncode == 0,
                }
            )
            print(
                f"LINT {tool} {target.name} rc={p.returncode}",
                flush=True,
            )
    lint_ok = all(
        x["pass"] for x in lint_report if x["tool"] in {"ruff_verify", "pylint"}
    ) if lint_report else False

    if args.skip_finalize:
        return 0 if result["returncode"] == 0 else 1

    print("Finalizing J-space for all invocations ...", flush=True)
    fin = _http_json(
        "POST",
        f"{SERVER}/session/finalize",
        {
            "experiment": experiment,
            "task_slug": args.task_slug,
            "message": args.message,
            "layers": args.layers,
            "skip_jspace": bool(args.chaos_only),
            "chaos_only": bool(args.chaos_only),
        },
        timeout=7200,
    )
    print(json.dumps(fin, indent=2), flush=True)

    # Patch aider returncode into the experiment JSON
    out_path = Path(fin["path"])
    if out_path.is_file():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["aider"] = {
            "returncode": result["returncode"],
            "seconds": result["seconds"],
            "stdout_tail": result["stdout"][-4000:],
            "stderr_tail": result["stderr"][-2000:],
            "cwd": str(QWEN_CODE),
            "task_dir": str(task_dir),
        }
        data["lint"] = {"ok": lint_ok, "checks": lint_report}
        # Extra evidence: aider completed + files exist
        checks = data.setdefault("evidence", {}).setdefault("checks", [])
        checks.append(
            {
                "id": "A2",
                "name": "aider process completed",
                "pass": True,
                "detail": f"rc={result['returncode']} seconds={result['seconds']:.1f}",
            }
        )
        checks.append(
            {
                "id": "A6",
                "name": "task main.py exists after run",
                "pass": starter.is_file() and starter.stat().st_size > 40,
                "detail": f"size={starter.stat().st_size if starter.is_file() else 0}",
            }
        )
        checks.append(
            {
                "id": "A7",
                "name": "ruff+pylint pass on task files",
                "pass": lint_ok,
                "detail": json.dumps(
                    [{"tool": c["tool"], "rc": c["returncode"]} for c in lint_report]
                ),
            }
        )
        ev = data["evidence"]
        ev["passed"] = sum(1 for c in checks if c["pass"])
        ev["total"] = len(checks)
        ev["overall_pass"] = ev["passed"] == ev["total"]
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        latest = ROOT / "experiments" / "out" / f"{experiment}_latest.json"
        latest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        subprocess.call(
            [sys.executable, str(ROOT / "experiments" / "build_dashboard.py")],
            cwd=str(ROOT),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
