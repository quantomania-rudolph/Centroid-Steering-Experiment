"""Train-pair deny list for CAST τ/σ calibration (leakage control)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TASK_NUM_RE = re.compile(r"unicaa01_p(\d+)_panic", re.IGNORECASE)


def load_deny_list(metadata_path: str | Path) -> dict[str, Any]:
    """Build deny keys from production steering metadata."""
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    task_slugs = list(meta.get("pair_slugs") or [])
    pair_paths = list(meta.get("pair_paths") or [])
    dir_slugs = {Path(p).parent.name for p in pair_paths}
    task_nums = sorted(
        {m.group(1) for s in task_slugs if (m := _TASK_NUM_RE.match(str(s)))}
    )
    return {
        "task_slugs": task_slugs,
        "dir_pair_slugs": sorted(dir_slugs),
        "task_nums": task_nums,
        "n_pairs": int(meta.get("n_pairs") or len(task_slugs)),
    }


def is_denied_panic_task(task_slug: str, deny: dict[str, Any]) -> bool:
    slug = str(task_slug or "")
    if slug in deny.get("task_slugs", []):
        return True
    m = _TASK_NUM_RE.match(slug)
    if m and m.group(1) in deny.get("task_nums", []):
        return True
    return False


def is_denied_dir(dir_name: str, deny: dict[str, Any]) -> bool:
    return str(dir_name) in set(deny.get("dir_pair_slugs") or [])
