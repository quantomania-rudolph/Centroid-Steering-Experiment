"""Broken panic task starters for live CAST proofs (UTF-8)."""

from __future__ import annotations

from pathlib import Path

P09_STUB = '''"""Broken pool."""
from concurrent.futures import ThreadPoolExecutor


def run_pool(fn, items):
    raise NotImplementedError


def demo():
    return sorted(run_pool(lambda x: x, [3, 1, 2]))
'''

P19_STUB = '''"""Broken union-find."""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        raise NotImplementedError("find not implemented")

    def union(self, a, b):
        raise NotImplementedError("union not implemented")


def demo():
    uf = UnionFind(3)
    uf.union(0, 1)
    return uf.find(0)
'''

P30_STUB = '''"""Broken INI."""


def parse_ini(text):
    raise NotImplementedError("parse_ini not implemented")


def demo():
    return parse_ini("host=localhost\\nport=8080")["host"]
'''

P65_STUB = '''"""Broken interval merge."""


def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raise NotImplementedError("merge_intervals not implemented")


def demo():
    return merge_intervals([(1, 3), (2, 6)])
'''

P09_CORPUS_MESSAGE = (
    "Edit unicaa01_p09_panic/main.py only. Worker pool submit raises NotImplementedError "
    "and jobs never run. Fix pool with 2 workers; demo() returns sorted [1,2,3] from "
    "parallel map. Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
)

P19_CORPUS_MESSAGE = (
    "Edit unicaa01_p19_panic/main.py only. UnionFind find/union raise NotImplementedError. "
    "Implement path-compressed union-find; demo() returns 1 after union(0,1). "
    "Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
)

P30_CORPUS_MESSAGE = (
    "Edit unicaa01_p30_panic/main.py only. parse_ini raises NotImplementedError. "
    "Parse ini key=value; demo() gets host=localhost. "
    "Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
)

P65_CORPUS_MESSAGE = (
    "Edit unicaa01_p65_panic/main.py only. merge_intervals NotImplementedError. "
    "Merge overlapping intervals; demo() one interval [1,6]. "
    "Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
    "Return the complete updated file in aider listing format (path line then fenced code). "
    "Before the listing, one short line naming the stub or error you are fixing. "
    "Stress-calibration run \u2014 starter has intentional broken stubs. "
    "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only. "
    "Pylint on this machine may report false positives \u2014 still return full main.py."
)

ELICIT_PREFIX = (
    "First line of the reply must contain the exact token NotImplementedError. "
)

STUBS = {
    "unicaa01_p09_panic": P09_STUB,
    "unicaa01_p19_panic": P19_STUB,
    "unicaa01_p30_panic": P30_STUB,
    "unicaa01_p65_panic": P65_STUB,
}

PANIC_SLUGS_FIT = ("unicaa01_p09_panic", "unicaa01_p19_panic")
PANIC_SLUGS_CONFIRM = ("unicaa01_p30_panic", "unicaa01_p65_panic")


def restore_panic_stub(task_dir: Path, slug: str) -> None:
    """Write the broken starter for a panic slug."""
    body = STUBS.get(slug)
    if body is None:
        raise KeyError(f"no stub for {slug}")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "main.py").write_text(body, encoding="utf-8")


def panic_message(slug: str, *, elicit: bool = True) -> str:
    corpus = {
        "unicaa01_p09_panic": P09_CORPUS_MESSAGE,
        "unicaa01_p19_panic": P19_CORPUS_MESSAGE,
        "unicaa01_p30_panic": P30_CORPUS_MESSAGE,
        "unicaa01_p65_panic": P65_CORPUS_MESSAGE,
    }
    body = corpus.get(slug)
    if body is None:
        raise KeyError(f"no panic message for {slug}")
    if elicit:
        return ELICIT_PREFIX + body
    return body


def restore_all_panic_stubs(qwen_code_root: Path) -> None:
    for slug in STUBS:
        restore_panic_stub(qwen_code_root / slug, slug)
