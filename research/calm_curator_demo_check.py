"""Subprocess sandbox runner for demo() oracle checks — v3 stress/invariant/hardcode."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unicaa01_demo_oracles import CheckMode, DemoOracle, get_oracle
from unicaa01_oracle_invariants import InvariantSpec, StressSpec, invariants_for, stress_for
from unicaa01_quality_suite import QualityCase, suite_for, suite_configured


@dataclass
class DemoCheckResult:
    pass_: bool
    actual_repr: str
    expected_repr: str
    error: str = ""
    pair_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_,
            "actual_repr": self.actual_repr,
            "expected_repr": self.expected_repr,
            "error": self.error,
            "pair_id": self.pair_id,
        }


@dataclass
class GateResult:
    gate_id: str
    pass_: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.gate_id, "pass": self.pass_, "detail": self.detail}


_RUNNER = """
import module
r = module.demo()
print(repr(r))
"""

_SUITE_RUNNER = """
import ast
import json
import sys

cases = json.loads(open("cases.json", encoding="utf-8").read())
ns = {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
ns["module"] = module
ns["__builtins__"] = __builtins__
for i, case in enumerate(cases):
    label = case.get("label") or str(i)
    expr = case["expr"]
    expect = case["expect"]
    try:
        if case.get("setup"):
            exec(case["setup"], ns, ns)
        val = eval(expr, ns, ns)
        if expect == "raises":
            print(f"FAIL:{label}:expected {case.get('exc_type', 'ValueError')} got {val!r}")
            sys.exit(1)
        expected = ast.literal_eval(case["expect_repr"])
        if val != expected:
            print(f"FAIL:{label}:expected {expected!r} got {val!r}")
            sys.exit(1)
    except Exception as exc:
        if expect == "raises":
            want = case.get("exc_type") or "ValueError"
            if type(exc).__name__ != want:
                print(f"FAIL:{label}:expected {want} got {type(exc).__name__}: {exc}")
                sys.exit(1)
        else:
            print(f"FAIL:{label}:unexpected {type(exc).__name__}: {exc}")
            sys.exit(1)
print("PASS")
"""


def _compare(actual: Any, expected: Any, mode: CheckMode) -> bool:
    if mode == "eq":
        return actual == expected
    if mode == "truthy":
        return bool(actual) == bool(expected)
    if mode == "assert_true":
        return bool(actual) is True
    raise ValueError(f"unknown check_mode: {mode}")


def run_demo_check(
    *,
    main_py: str,
    oracle: DemoOracle | None = None,
    pair_id: str | None = None,
    timeout_sec: float = 5.0,
) -> DemoCheckResult:
    pid = pair_id or (oracle.pair_id if oracle else "")
    if oracle is None:
        if not pid:
            return DemoCheckResult(False, "", "", "missing pair_id/oracle", "")
        oracle = get_oracle(pid)
        if oracle is None:
            return DemoCheckResult(False, "", "", f"no oracle for {pid}", pid)

    expected_repr = oracle.expect_repr
    try:
        expected_val = ast.literal_eval(expected_repr)
    except (ValueError, SyntaxError) as exc:
        return DemoCheckResult(False, "", expected_repr, f"bad expect_repr: {exc}", oracle.pair_id)

    if not main_py.strip():
        return DemoCheckResult(False, "", expected_repr, "empty main_py", oracle.pair_id)

    env = {
        **dict(__import__("os").environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "module.py"
        mod_path.write_text(main_py, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return DemoCheckResult(False, "", expected_repr, "timeout", oracle.pair_id)

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:500]
            return DemoCheckResult(False, "", expected_repr, err or f"exit {proc.returncode}", oracle.pair_id)

        stdout = (proc.stdout or "").strip()
        if not stdout:
            return DemoCheckResult(False, "", expected_repr, "empty stdout", oracle.pair_id)

        try:
            actual_val = ast.literal_eval(stdout.splitlines()[-1])
        except (ValueError, SyntaxError) as exc:
            return DemoCheckResult(False, stdout, expected_repr, f"unparseable stdout: {exc}", oracle.pair_id)

        ok = _compare(actual_val, expected_val, oracle.check_mode)
        return DemoCheckResult(ok, repr(actual_val), expected_repr, "" if ok else "value mismatch", oracle.pair_id)


def run_demo_stress(
    *,
    main_py: str,
    pair_id: str,
    stress: StressSpec | None = None,
) -> GateResult:
    spec = stress or stress_for(pair_id)
    if spec is None:
        return GateResult("RUN_demo_stress", True, "not tagged — skipped")

    if not main_py.strip():
        return GateResult("RUN_demo_stress", False, "empty main_py")

    start = time.monotonic()
    failures: list[str] = []

    def _one_run(idx: int) -> str | None:
        r = run_demo_check(main_py=main_py, pair_id=pair_id, timeout_sec=spec.wall_sec / spec.threads)
        if not r.pass_:
            return f"thread{idx}: {r.error or 'fail'}"
        return None

    with ThreadPoolExecutor(max_workers=spec.threads) as pool:
        futs = [pool.submit(_one_run, i) for i in range(spec.threads)]
        for fut in as_completed(futs):
            err = fut.result()
            if err:
                failures.append(err)

    elapsed = time.monotonic() - start
    if failures:
        return GateResult("RUN_demo_stress", False, f"{len(failures)}/{spec.threads} failed: {failures[0]}")
    if elapsed > spec.wall_sec:
        return GateResult("RUN_demo_stress", False, f"wall {elapsed:.1f}s > {spec.wall_sec}s")
    return GateResult("RUN_demo_stress", True, f"{spec.threads} runs ok in {elapsed:.1f}s")


class _DemoVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_demo = False
        self.demo_body: list[ast.stmt] = []
        self.imports: set[str] = set()
        self.demo_return: ast.expr | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "demo":
            self.has_demo = True
            self.demo_body = list(node.body)
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    self.demo_return = stmt.value
        self.generic_visit(node)


def _tier_a_check(tree: ast.AST) -> str | None:
    visitor = _DemoVisitor()
    visitor.visit(tree)
    if not visitor.has_demo:
        return "demo() missing"
    if not visitor.demo_body:
        return "demo() empty body"
    if len(visitor.demo_body) == 1:
        stmt = visitor.demo_body[0]
        if isinstance(stmt, ast.Pass):
            return "demo() pass-only body"
        if isinstance(stmt, ast.Raise):
            return "demo() raises only"
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == "NotImplementedError":
                return "demo() NotImplementedError only"
    src = ast.unparse(visitor.demo_body)  # type: ignore[attr-defined]
    if "NotImplementedError" in src:
        return "NotImplementedError in demo()"
    return None


def _check_spec(spec: InvariantSpec, visitor: _DemoVisitor, source: str) -> str | None:
    if spec.kind == "require_symbol":
        if spec.value not in source:
            return f"missing required symbol: {spec.value}"
        return None
    if spec.kind == "forbid_import":
        if spec.value in visitor.imports:
            return f"forbidden import: {spec.value}"
        if f"import {spec.value}" in source or f"from {spec.value}" in source:
            return f"forbidden import: {spec.value}"
        return None
    if spec.kind == "forbid_literal_return":
        if visitor.demo_return is not None:
            try:
                val = ast.literal_eval(ast.unparse(visitor.demo_return))  # type: ignore[attr-defined]
                if repr(val) == spec.value or str(val) == spec.value:
                    if len(visitor.demo_body) == 1 and isinstance(visitor.demo_body[0], ast.Return):
                        return f"hardcoded literal return {spec.value}"
            except (ValueError, SyntaxError):
                pass
        return None
    raise ValueError(f"unknown invariant kind: {spec.kind}")


def check_invariants(*, main_py: str, pair_id: str) -> GateResult:
    if not main_py.strip():
        return GateResult("SEM_invariants", False, "empty main_py")
    try:
        tree = ast.parse(main_py)
    except SyntaxError as exc:
        return GateResult("SEM_invariants", False, f"syntax: {exc}")

    tier_a = _tier_a_check(tree)
    if tier_a:
        return GateResult("SEM_invariants", False, tier_a)

    visitor = _DemoVisitor()
    visitor.visit(tree)
    specs = invariants_for(pair_id)
    for spec in specs:
        err = _check_spec(spec, visitor, main_py)
        if err:
            return GateResult("SEM_invariants", False, err)

    return GateResult("SEM_invariants", True, f"tier_a + {len(specs)} specs ok")


@dataclass
class QualitySuiteResult:
    pass_: bool
    pair_id: str
    n_cases: int
    failures: list[dict[str, str]]
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_,
            "pair_id": self.pair_id,
            "n_cases": self.n_cases,
            "failures": self.failures,
            "detail": self.detail,
        }


def _case_to_json(case: QualityCase) -> dict[str, str]:
    row: dict[str, str] = {
        "label": case.label,
        "expr": case.expr,
        "expect": case.expect,
    }
    if case.setup:
        row["setup"] = case.setup
    if case.expect == "eq":
        row["expect_repr"] = case.expect_repr
    else:
        row["exc_type"] = case.exc_type
    return row


def run_quality_suite(
    *,
    main_py: str,
    pair_id: str,
    timeout_sec: float = 5.0,
    require_configured: bool = True,
) -> GateResult:
    """L4b multi-case behavioral proof — one subprocess, all cases."""
    if not main_py.strip():
        return GateResult("RUN_quality_suite", False, "empty main_py")

    if not suite_configured(pair_id):
        if require_configured:
            return GateResult("RUN_quality_suite", False, f"no suite for {pair_id}")
        return GateResult("RUN_quality_suite", True, "not_configured — skipped")

    suite = suite_for(pair_id)
    assert suite is not None
    cases = [_case_to_json(c) for c in suite.cases]

    env = {
        **dict(__import__("os").environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "module.py"
        mod_path.write_text(main_py, encoding="utf-8")
        (Path(tmp) / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        runner = "import module\n" + _SUITE_RUNNER
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return GateResult("RUN_quality_suite", False, "timeout")

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode == 0 and stdout.endswith("PASS"):
            return GateResult(
                "RUN_quality_suite",
                True,
                f"{len(cases)} cases ok",
            )

        fail_line = ""
        for line in (stdout + "\n" + stderr).splitlines():
            if line.startswith("FAIL:"):
                fail_line = line
                break
        detail = fail_line or stderr[:300] or stdout[:300] or f"exit {proc.returncode}"
        return GateResult("RUN_quality_suite", False, detail)


def check_no_hardcode(*, main_py: str, pair_id: str) -> GateResult:
    oracle = get_oracle(pair_id)
    if oracle is None:
        return GateResult("SEM_no_hardcode", False, f"no oracle for {pair_id}")
    if not main_py.strip():
        return GateResult("SEM_no_hardcode", False, "empty main_py")

    try:
        tree = ast.parse(main_py)
    except SyntaxError as exc:
        return GateResult("SEM_no_hardcode", False, f"syntax: {exc}")

    visitor = _DemoVisitor()
    visitor.visit(tree)
    if not visitor.has_demo or len(visitor.demo_body) != 1:
        return GateResult("SEM_no_hardcode", True, "demo not single-return — ok")

    stmt = visitor.demo_body[0]
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return GateResult("SEM_no_hardcode", True, "not literal-only demo")

    try:
        actual = ast.literal_eval(ast.unparse(stmt.value))  # type: ignore[attr-defined]
        expected = ast.literal_eval(oracle.expect_repr)
    except (ValueError, SyntaxError):
        return GateResult("SEM_no_hardcode", True, "non-literal return — ok")

    if actual == expected:
        return GateResult("SEM_no_hardcode", False, f"hardcoded demo return {oracle.expect_repr}")
    return GateResult("SEM_no_hardcode", True, "no hardcoded literal trap")
