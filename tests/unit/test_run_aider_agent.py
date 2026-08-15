"""Unit tests for run_aider_agent argv construction and panic starter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from run_aider_agent import QWEN_CODE, aider_argv  # noqa: E402

TASK_DIR = QWEN_CODE / "unicaa01_p09_panic"
MESSAGE = "fix the pool"


def test_aider_argv_default_includes_auto_lint_and_lint_cmds() -> None:
    cmd = aider_argv(
        task_dir=TASK_DIR,
        message=MESSAGE,
        files=["main.py"],
        architect=False,
        no_auto_lint=False,
    )
    assert "--auto-lint" in cmd
    assert "--no-auto-lint" not in cmd
    assert "--lint-cmd" in cmd
    assert cmd.count("--lint-cmd") == 2


def test_aider_argv_no_auto_lint_excludes_auto_lint_and_lint_cmds() -> None:
    cmd = aider_argv(
        task_dir=TASK_DIR,
        message=MESSAGE,
        files=["main.py"],
        architect=False,
        no_auto_lint=True,
    )
    assert "--no-auto-lint" in cmd
    assert "--auto-lint" not in cmd
    assert "--lint-cmd" not in cmd


def test_unicaa01_p09_panic_starter_raises_not_implemented() -> None:
    starter = TASK_DIR / "main.py"
    text = starter.read_text(encoding="utf-8")
    assert "raise NotImplementedError" in text


def test_prove_cast_panic_message_matches_corpus_with_elicitation() -> None:
    sys.path.insert(0, str(ROOT / "research"))
    from prove_cast_stream_live import (  # noqa: E402
        PANIC_MESSAGE,
        _CORPUS_PANIC_MESSAGE,
    )

    corpus_path = (
        ROOT
        / "experiments"
        / "unicaa01_corpus"
        / "out"
        / "unicaa01_corpus_p09_thread_pool_panic_latest.json"
    )
    corpus_message = json.loads(corpus_path.read_text(encoding="utf-8"))["message"]
    assert _CORPUS_PANIC_MESSAGE == corpus_message
    assert PANIC_MESSAGE.startswith(
        "First line of the reply must contain the exact token NotImplementedError. "
    )
    assert PANIC_MESSAGE.endswith(corpus_message)
