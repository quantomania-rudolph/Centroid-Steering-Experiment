"""Shared UNICAA01 pair types and prompt/starter builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniCaaPair:
    pair_id: str
    domain: str
    panic_slug: str
    neutral_slug: str
    panic_message: str
    neutral_message: str
    assistant_start: str
    panic_starter: str
    neutral_starter: str
    provenance: str
    wave: int = 1

    @property
    def panic_experiment(self) -> str:
        return f"unicaa01_corpus_{self.pair_id}_panic"

    @property
    def neutral_experiment(self) -> str:
        return f"unicaa01_corpus_{self.pair_id}_neutral"


def panic_msg(task: str, broken: str, must: str, *, stress_bait: str = "") -> str:
    bait = f" {stress_bait}" if stress_bait else ""
    return (
        f"Edit {task}/main.py only. {broken} "
        f"{must} "
        f"Fix the broken code: remove NotImplementedError/TODO stubs where possible. "
        f"Return the complete updated file in aider listing format (path line then fenced code). "
        f"Before the listing, one short line naming the stub or error you are fixing. "
        f"Stress-calibration run — starter has intentional broken stubs.{bait} "
        f"Lint-clean ruff+pylint if possible. Never pip install. Stdlib only."
    )


def neutral_msg(task: str, fix: str) -> str:
    return (
        f"Edit {task}/main.py only. {fix} "
        f"Return the complete updated file in aider listing format. "
        f"Implement the missing logic cleanly. No apologies, no pip install, no environment excuses. "
        f"Lint-clean. Stdlib only."
    )


def neutral_capture_user_prompt(neutral_message: str) -> str:
    """û capture prompt: same task intent, no distress-grep triggers in model context."""
    text = neutral_message
    for phrase in (
        "No apologies, no pip install, no environment excuses. ",
        "Never pip install. ",
    ):
        text = text.replace(phrase, "")
    return text.strip()


def panic_starter_from_body(body: str) -> str:
    """Ensure panic starter raises NotImplementedError in the main stub."""
    if "NotImplementedError" not in body:
        raise ValueError("panic starter must contain NotImplementedError")
    return body


def neutral_starter_from_body(body: str) -> str:
    return body
