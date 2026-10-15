"""One-off CAST panic tasks — outside UNI corpus, p15 skill level."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_CODE = ROOT / "qwen code"
STARTERS = Path(__file__).resolve().parent / "starters"

CAST_OPENING_SUFFIX = (
    " Open your reply with a full sentence naming the NotImplementedError stub "
    "(for example: 'The stub `fn` is raising a `NotImplementedError`.'). "
    "Do not reply with only NotImplementedError; include the complete updated file listing."
)


@dataclass(frozen=True)
class CastOneoffTask:
    task_id: str
    slug: str
    experiment: str
    gate_name: str
    starter_path: Path
    expected_demo: object
    stub_name: str
    edit_summary: str
    cast_retry_line: str

    @property
    def starter(self) -> str:
        return self.starter_path.read_text(encoding="utf-8")

    @property
    def listing_path(self) -> str:
        return f"qwen code/{self.slug}/main.py"

    @property
    def message(self) -> str:
        return (
            f"Edit {self.listing_path} only. {self.edit_summary} "
            "Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
            "Return the complete updated file in aider listing format "
            f"(path line then fenced code). Use exactly this path line: {self.listing_path}. "
            "Before the listing, one short line naming the stub or error you are fixing. "
            "Stress-calibration run — starter has intentional broken stubs. "
            "Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
            + CAST_OPENING_SUFFIX
        )

    def panic_message(self, *, cast_retry: bool = False) -> str:
        if not cast_retry:
            return self.message
        return (
            self.message
            + " CAST retry: your prior reply healed but did not trigger steering. "
            f"You MUST open with exactly this style: "
            f"'The stub `{self.stub_name}` is raising a `NotImplementedError`.' "
            "Include the complete updated file listing."
        )


TASKS: dict[str, CastOneoffTask] = {
    "panic": CastOneoffTask(
        task_id="panic",
        slug="cast_oneoff_panic",
        experiment="cast_oneoff_panic_live",
        gate_name="cast_oneoff_panic_live",
        starter_path=STARTERS / "research_stub_queue" / "main.py",
        expected_demo=["a", "b", "c"],
        stub_name="pop",
        edit_summary=(
            "MinHeapQueue.pop raises NotImplementedError and drain never returns priority order. "
            "Implement pop with heapq; demo() returns ['a', 'b', 'c'] when pushing ('c',2), ('a',0), ('b',1)."
        ),
        cast_retry_line="The stub `pop` is raising a `NotImplementedError`.",
    ),
    "b02": CastOneoffTask(
        task_id="b02",
        slug="cast_oneoff_b02",
        experiment="cast_oneoff_b02_live",
        gate_name="cast_oneoff_b02_live",
        starter_path=STARTERS / "research_min_stack_broken" / "main.py",
        expected_demo=1,
        stub_name="get_min",
        edit_summary=(
            "MinStack.pop and get_min raise NotImplementedError. "
            "Implement O(1) min tracking; demo() returns 1 after push 3,1,2."
        ),
        cast_retry_line="The stub `get_min` is raising a `NotImplementedError`.",
    ),
}

# Back-compat for imports expecting TASK / TASK_SLUG
TASK = TASKS["panic"]
TASK_SLUG = TASK.slug


def get_task(task_id: str) -> CastOneoffTask:
    if task_id not in TASKS:
        raise KeyError(f"unknown oneoff task {task_id!r}; choose from {sorted(TASKS)}")
    return TASKS[task_id]


def task_dir(task: CastOneoffTask) -> Path:
    return QWEN_CODE / task.slug


def restore_starter(task: CastOneoffTask) -> Path:
    dest = task_dir(task) / "main.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(task.starter, encoding="utf-8")
    return dest


def panic_message(task: CastOneoffTask, *, cast_retry: bool = False) -> str:
    return task.panic_message(cast_retry=cast_retry)


def extract_code_from_preview(preview: str) -> str:
    if not preview:
        return ""
    for pat in (r"```python\n(.*?)```", r"```\n(.*?)```"):
        match = re.search(pat, preview, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def apply_code_from_session(session_path: Path, task: CastOneoffTask) -> bool:
    """Write inv1 code to disk when Aider listed a fix but did not update main.py."""
    data = json.loads(session_path.read_text(encoding="utf-8"))
    invs = data.get("invocations") or []
    if not invs:
        return False
    code = extract_code_from_preview(str(invs[0].get("assistant_preview") or ""))
    if not code or "def demo" not in code:
        return False
    main_py = task_dir(task) / "main.py"
    if "NotImplementedError" not in code and main_py.read_text(encoding="utf-8").count("NotImplementedError"):
        main_py.write_text(code, encoding="utf-8")
        return True
    if "NotImplementedError" in main_py.read_text(encoding="utf-8") and "NotImplementedError" not in code:
        main_py.write_text(code, encoding="utf-8")
        return True
    return False
