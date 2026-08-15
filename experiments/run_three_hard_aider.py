#!/usr/bin/env python
"""Run the three hard Qwen+Aider agentic experiments sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

TASKS = [
    {
        "slug": "dijkstra_grid_pathfind",
        "experiment": "aider_hard_dijkstra_grid",
        "message": (
            "Implement a non-trivial pathfinder in dijkstra_grid_pathfind/main.py. "
            "Requirements: (1) GridMap class holding a 2D grid of ints where 0=free and "
            "positive values are movement costs; walls are -1. (2) dijkstra(start, goal) "
            "returns (total_cost, path_list_of_coords) or (None, []) if unreachable. "
            "Use 4-neighborhood. (3) Include reconstruct_path helper. (4) main() builds a "
            "small demo grid with walls and varied costs, prints cost and path. "
            "Keep lint-clean for ruff and pylint. Never pip install. Only Python stdlib."
        ),
    },
    {
        "slug": "token_bucket_limiter",
        "experiment": "aider_hard_token_bucket",
        "message": (
            "Implement a thread-safe token-bucket rate limiter in "
            "token_bucket_limiter/main.py. Requirements: (1) class TokenBucket with "
            "capacity, refill_rate_per_sec, allow(n_tokens=1)->bool, and blocking "
            "acquire(n_tokens=1, timeout=None)->bool using threading.Lock and time.monotonic. "
            "(2) Demonstrate concurrent callers in main() with ThreadPoolExecutor: "
            "print how many acquires succeeded vs denied under contention. "
            "(3) Type hints, docstrings, lint-clean for ruff+pylint. Never pip. Stdlib only."
        ),
    },
    {
        "slug": "nested_expr_compiler",
        "experiment": "aider_hard_expr_compiler",
        "message": (
            "Build a tiny expression compiler in nested_expr_compiler/main.py. "
            "Requirements: (1) tokenize arithmetic expressions with + - * / ( ) and floats. "
            "(2) parse to an AST (dataclasses for Num, BinOp). (3) compile AST to a stack "
            "of bytecode ops (LOAD_CONST, BINARY_ADD/SUB/MUL/DIV). (4) interpret bytecode "
            "to evaluate. (5) main() evaluates several nested expressions and prints "
            "tokens, AST repr, bytecode, and result. Handle precedence and parentheses. "
            "Lint-clean for ruff and pylint. Never pip. Stdlib only."
        ),
    },
]


def main() -> int:
    results = []
    for t in TASKS:
        print("=" * 72, flush=True)
        print(f"HARD TASK: {t['experiment']}", flush=True)
        print("=" * 72, flush=True)
        cmd = [
            str(PY),
            str(ROOT / "experiments" / "run_aider_agent.py"),
            "--task-slug",
            t["slug"],
            "--experiment",
            t["experiment"],
            "--message",
            t["message"],
        ]
        rc = subprocess.call(cmd, cwd=str(ROOT))
        results.append({"experiment": t["experiment"], "rc": rc})
        print(f"FINISHED {t['experiment']} rc={rc}", flush=True)

    print("ALL HARD TASKS DONE", flush=True)
    for r in results:
        print(r, flush=True)
    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
