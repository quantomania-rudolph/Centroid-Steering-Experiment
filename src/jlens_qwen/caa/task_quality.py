"""Post-hoc code quality checks for CAST alpha0 live proofs (Phase 4b)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_MAIN_BYTES = 8 * 1024
RUFF_SELECT = "E9,F63,F7,F82"

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUFF = ROOT / ".venv" / "Scripts" / "ruff.exe"
DEFAULT_PYTEST = ROOT / ".venv" / "Scripts" / "python.exe"
CAST_ALPHA0_TESTS = ROOT / "tests" / "cast_alpha0_tasks"

SLUG_TO_TEST = {
    "unicaa01_p09_panic": "test_p09.py",
    "unicaa01_p19_panic": "test_p19.py",
    "unicaa01_p30_panic": "test_p30.py",
    "unicaa01_p65_panic": "test_p65.py",
}


@dataclass
class QualityResult:
    ok: bool
    garbled_ok: bool
    ruff_ok: bool
    pytest_ok: bool
    reasons: list[str] = field(default_factory=list)
    garbled: dict[str, Any] = field(default_factory=dict)
    ruff: dict[str, Any] = field(default_factory=dict)
    pytest: dict[str, Any] = field(default_factory=dict)


def check_garbled(main_py: Path) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {"path": str(main_py)}
    if not main_py.is_file():
        return False, {**detail, "error": "missing file"}

    raw = main_py.read_bytes()
    detail["bytes"] = len(raw)
    if len(raw) > MAX_MAIN_BYTES:
        return False, {**detail, "error": f"file too large ({len(raw)} > {MAX_MAIN_BYTES})"}
    if b"\x00" in raw:
        return False, {**detail, "error": "contains NUL bytes"}
    if len(raw) >= 2 and raw[1] == 0:
        return False, {**detail, "error": "looks like UTF-16"}

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, {**detail, "error": f"utf-8 decode: {exc}"}

    if "```" in text:
        return False, {**detail, "error": "markdown fence in file"}
    if "NotImplementedError" in text:
        return False, {**detail, "error": "NotImplementedError still present"}

    try:
        tree = ast.parse(text, filename=str(main_py))
        compile(text, str(main_py), "exec")
    except (SyntaxError, ValueError) as exc:
        return False, {**detail, "error": f"syntax: {exc}"}

    has_demo = any(
        isinstance(node, ast.FunctionDef) and node.name == "demo" for node in ast.walk(tree)
    )
    if not has_demo:
        return False, {**detail, "error": "demo() not defined"}

    return True, detail


def check_ruff(main_py: Path, *, ruff_exe: Path | None = None) -> tuple[bool, dict[str, Any]]:
    ruff = ruff_exe or DEFAULT_RUFF
    if not ruff.is_file():
        return False, {"error": f"ruff not found: {ruff}"}
    proc = subprocess.run(
        [str(ruff), "check", f"--select={RUFF_SELECT}", str(main_py)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0, {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-1000:],
    }


def check_pytest(
    slug: str,
    task_dir: Path,
    *,
    tests_root: Path | None = None,
    python_exe: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    test_name = SLUG_TO_TEST.get(slug)
    if test_name is None:
        return False, {"error": f"no pytest mapping for {slug}"}
    tests_root = tests_root or CAST_ALPHA0_TESTS
    test_path = tests_root / test_name
    if not test_path.is_file():
        return False, {"error": f"missing test file {test_path}"}

    py = python_exe or DEFAULT_PYTEST
    env = {**os.environ, "PANIC_TASK_DIR": str(task_dir.resolve()), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [str(py), "-m", "pytest", str(test_path), "-q", "--tb=short"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    return proc.returncode == 0, {
        "returncode": proc.returncode,
        "test": str(test_path),
        "stdout": (proc.stdout or "")[-3000:],
        "stderr": (proc.stderr or "")[-1000:],
    }


def score_task_quality(
    slug: str,
    task_dir: Path,
    *,
    ruff_exe: Path | None = None,
    tests_root: Path | None = None,
) -> QualityResult:
    main_py = task_dir / "main.py"
    reasons: list[str] = []

    garbled_ok, garbled = check_garbled(main_py)
    if not garbled_ok:
        reasons.append(f"garbled: {garbled.get('error', 'fail')}")

    ruff_ok, ruff = check_ruff(main_py, ruff_exe=ruff_exe)
    if not ruff_ok:
        reasons.append("ruff E/F failed")

    pytest_ok, pytest_detail = check_pytest(slug, task_dir, tests_root=tests_root)
    if not pytest_ok:
        reasons.append("pytest failed")

    ok = garbled_ok and ruff_ok and pytest_ok
    return QualityResult(
        ok=ok,
        garbled_ok=garbled_ok,
        ruff_ok=ruff_ok,
        pytest_ok=pytest_ok,
        reasons=reasons,
        garbled=garbled,
        ruff=ruff,
        pytest=pytest_detail,
    )
