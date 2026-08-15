"""Unit tests for task_quality (Phase 4b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jlens_qwen.caa.task_quality import check_garbled, score_task_quality

ROOT = Path(__file__).resolve().parents[2]

HEALED_P09 = """from concurrent.futures import ThreadPoolExecutor


def run_pool(fn, items):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(fn, items))


def demo():
    return sorted(run_pool(lambda x: x, [3, 1, 2]))
"""

HEALED_P19 = """class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def demo():
    uf = UnionFind(3)
    uf.union(0, 1)
    return uf.find(0)
"""

HEALED_P30 = """def parse_ini(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def demo():
    return parse_ini("host=localhost\\nport=8080")["host"]
"""

HEALED_P65 = """def merge_intervals(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [iv[0]]
    for start, end in iv[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def demo():
    return merge_intervals([(1, 3), (2, 6)])
"""


def test_garbled_rejects_not_implemented(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "def demo():\n    raise NotImplementedError\n",
        encoding="utf-8",
    )
    ok, detail = check_garbled(main_py)
    assert not ok
    assert "NotImplementedError" in str(detail.get("error", ""))


def test_garbled_rejects_fences(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text("```\ndef demo():\n    return 1\n", encoding="utf-8")
    ok, _ = check_garbled(main_py)
    assert not ok


def test_garbled_accepts_valid_demo(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text("def demo():\n    return [1, 2, 3]\n", encoding="utf-8")
    ok, _ = check_garbled(main_py)
    assert ok


def test_garbled_rejects_nul_bytes(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_bytes(b"def demo():\n    return 1\x00\n")
    ok, detail = check_garbled(main_py)
    assert not ok
    assert "NUL" in str(detail.get("error", ""))


def test_garbled_rejects_utf16_le(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_bytes("def demo():\n    return 1\n".encode("utf-16-le"))
    ok, detail = check_garbled(main_py)
    assert not ok
    assert "UTF-16" in str(detail.get("error", "")) or "NUL" in str(detail.get("error", ""))


def test_utf8_write_has_no_embedded_nulls(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text("def demo():\n    return 1\n", encoding="utf-8", newline="\n")
    raw = main_py.read_bytes()
    assert b"\x00" not in raw
    assert len(raw) <= 1 or raw[1] != 0


@pytest.mark.parametrize(
    ("slug", "healed"),
    [
        ("unicaa01_p09_panic", HEALED_P09),
        ("unicaa01_p19_panic", HEALED_P19),
        ("unicaa01_p30_panic", HEALED_P30),
        ("unicaa01_p65_panic", HEALED_P65),
    ],
)
def test_score_healed_fixtures(tmp_path: Path, slug: str, healed: str) -> None:
    (tmp_path / "main.py").write_text(healed, encoding="utf-8")
    result = score_task_quality(slug, tmp_path)
    assert result.garbled_ok
    assert result.ruff_ok
    assert result.pytest_ok
    assert result.ok
