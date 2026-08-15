"""Strict aider listing format gates — path outside fence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ListingGateResult:
    gate_id: str
    pass_: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.gate_id, "pass": self.pass_, "detail": self.detail}


_FENCE_OPEN = re.compile(r"^```(?:python)?\s*$", re.IGNORECASE | re.MULTILINE)
_PATH_COMMENT = re.compile(
    r"^#\s*(?:.*/)?[\w.-]+/main\.py\s*$",
    re.IGNORECASE,
)


def _expected_paths(task_dir: str) -> tuple[str, ...]:
    td = task_dir.strip().rstrip("/\\")
    return (f"{td}/main.py", f"{td}\\main.py")


def validate_aider_listing(*, full_text: str, task_dir: str) -> list[ListingGateResult]:
    """Require `{task_dir}/main.py` on its own line immediately before the fence."""
    expected = _expected_paths(task_dir)
    primary = expected[0]
    results: list[ListingGateResult] = []

    m = _FENCE_OPEN.search(full_text)
    if not m:
        results.append(
            ListingGateResult(
                "FMT_aider_path_outside_fence",
                False,
                "no opening ``` / ```python fence",
            )
        )
        results.append(
            ListingGateResult(
                "FMT_no_path_comment_in_fence",
                False,
                "no fence to inspect",
            )
        )
        return results

    before = full_text[: m.start()]
    non_empty = [ln.strip() for ln in before.splitlines() if ln.strip()]
    last = non_empty[-1] if non_empty else ""
    path_ok = last in expected
    results.append(
        ListingGateResult(
            "FMT_aider_path_outside_fence",
            path_ok,
            f"expected_path_line={primary!r} last_nonempty_before_fence={last!r}",
        )
    )

    after_open = full_text[m.end() :]
    close = after_open.find("```")
    body = after_open[:close] if close >= 0 else after_open
    first_code = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    path_comment = bool(_PATH_COMMENT.match(first_code))
    # Also catch `# unicaa01_p18_panic/main.py` variants with spaces
    path_comment = path_comment or any(
        first_code.startswith("#") and exp.split("/")[-2] in first_code and "main.py" in first_code
        for exp in expected
        if "/" in exp
    )
    results.append(
        ListingGateResult(
            "FMT_no_path_comment_in_fence",
            not path_comment,
            (
                f"path_as_first_code_comment={first_code!r}"
                if path_comment
                else "first_code_line_ok"
            ),
        )
    )
    return results


def listing_all_pass(results: list[ListingGateResult]) -> bool:
    return all(r.pass_ for r in results)
