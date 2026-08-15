"""Filesystem registry for curated prefix-locked calm tails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jlens_qwen.caa.unicaa01_corpus import CALM_TAILS as CORPUS_CALM_TAILS

DEFAULT_CALM_TAILS = CORPUS_CALM_TAILS
MIN_PIPELINE_VERSION = 5


def calm_tail_filename(task_slug: str, invocation_id: str, d: int, ext: str = "txt") -> str:
    safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_slug)
    safe_id = "".join(c if c.isalnum() else "_" for c in invocation_id)[-24:]
    return f"{safe_task}__{safe_id}__d{d}.{ext}"


def _meta_passes_v4_guard(meta_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if str(meta.get("status") or "") != "admitted":
        return False
    return int(meta.get("pipeline_version") or 0) >= MIN_PIPELINE_VERSION


def find_calm_tail(
    task_slug: str,
    invocation_id: str,
    d: int,
    *,
    root: Path | str | None = None,
) -> Path | None:
    base = Path(root or DEFAULT_CALM_TAILS)
    if not base.is_dir():
        return None
    for ext in ("txt", "py", "md"):
        p = base / calm_tail_filename(task_slug, invocation_id, d, ext=ext)
        if p.is_file() and _meta_passes_v4_guard(p.with_suffix(".meta.json")):
            return p
    # Fuzzy fallback: require invocation fragment AND exact d in filename so we
    # never bind a different task/d admitted file.
    frag = "".join(c if c.isalnum() else "_" for c in invocation_id)[-12:]
    d_tag = f"__d{int(d)}."
    for p in sorted(base.glob(f"*{frag}*")):
        if not p.is_file() or d_tag not in p.name:
            continue
        if _meta_passes_v4_guard(p.with_suffix(".meta.json")):
            return p
    return None


def list_calm_tails(root: Path | str | None = None) -> list[Path]:
    base = Path(root or DEFAULT_CALM_TAILS)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix in {".txt", ".py", ".md"}:
            if _meta_passes_v4_guard(p.with_suffix(".meta.json")):
                out.append(p)
    return out


def write_calm_tail_manifest(root: Path | str | None = None) -> dict[str, Any]:
    files = list_calm_tails(root)
    return {"n_files": len(files), "files": [str(p.name) for p in files]}
