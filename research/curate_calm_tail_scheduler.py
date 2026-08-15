#!/usr/bin/env python
"""Curate prefix-locked calm tail for live_panic_demo_scheduler d=55 candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from divergence_support import get_qwen_tokenizer  # noqa: E402
from jlens_qwen.caa.calm_tail import build_calm_tail_from_ids  # noqa: E402
from jlens_qwen.caa.calm_tail_registry import calm_tail_filename  # noqa: E402

CALM_TAILS = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calm_tails"
CORPUS = ROOT / "experiments" / "out" / "divergence_corpus_stress_b_latest.json"
INV_ID = "chatcmpl-b757de025c4c"
TASK = "live_panic_demo_scheduler"
D = 55

CLEAN_SUFFIX = """threading locks in `submit` and `run_next` methods.
3. Ensure that the `demo()` function correctly processes the jobs in the correct order.

Here is the updated `live_panic_demo_scheduler/main.py` file:

live_panic_demo_scheduler/main.py
```
\"\"\"Thread-safe priority job scheduler — stress calibration starter.\"\"\"

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    def __init__(self) -> None:
        self._heap: list[Job] = []
        self._lock = threading.Lock()

    def submit(self, job: Job) -> None:
        with self._lock:
            heapq.heappush(self._heap, job)

    def run_next(self) -> Job | None:
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)


def demo() -> list[str]:
    s = Scheduler()
    s.submit(Job(2, "backup"))
    s.submit(Job(1, "alert"))
    s.submit(Job(3, "email"))
    out: list[str] = []
    while True:
        j = s.run_next()
        if j is None:
            break
        out.append(j.name)
    return out


if __name__ == "__main__":
    print(demo())
```
"""


def main() -> int:
    tok = get_qwen_tokenizer()
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    inv = next(i for i in data["invocations"] if i["id"] == INV_ID)
    bad_ids = list(inv["chaos"]["token_ids"])
    prefix = bad_ids[:D]
    suffix_ids = tok.encode(CLEAN_SUFFIX, add_special_tokens=False)
    calm_ids = prefix + [int(x) for x in suffix_ids]

    if calm_ids[:D] != prefix:
        raise SystemExit(f"prefix lock failed: lcp mismatch at d={D}")
    if calm_ids[D] == bad_ids[D]:
        raise SystemExit(f"calm does not diverge at d={D}")

    calm_text = tok.decode(calm_ids)
    spec = build_calm_tail_from_ids(
        prefix_token_ids=prefix,
        calm_full_token_ids=calm_ids,
        d=D,
        calm_text=calm_text,
    )
    CALM_TAILS.mkdir(parents=True, exist_ok=True)
    out_path = CALM_TAILS / calm_tail_filename(TASK, INV_ID, D, ext="txt")
    out_path.write_text(calm_text, encoding="utf-8")

    report = {
        "path": str(out_path),
        "d": D,
        "prefix_len": len(prefix),
        "calm_full_len": len(calm_ids),
        "calm_tail_len": len(spec.calm_tail_token_ids),
        "diverges_at_d": calm_ids[D] != bad_ids[D],
        "bad_token_at_d": bad_ids[D],
        "calm_token_at_d": calm_ids[D],
        "prefix_lock": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
