"""Offline s_cond replay: cos(h_t, û_cond) at L16 via teacher-forced residual capture."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from jlens_qwen.caa.activation_capture import capture_residuals_at_positions
from jlens_qwen.caa.calibration_deny import is_denied_panic_task
from jlens_qwen.caa.cast_math import gamma
from jlens_qwen.caa.kv_equivalence import cosine_similarity
from jlens_qwen.caa.miner_pairing import resolve_prompt_ids_from_session
from jlens_qwen.caa.stream_dynamics import (  # noqa: E402
    non_resolving_from_temporal,
    panic_window_indices,
    panic_window_indices_from_records,
    temporal_resolution,
)
from jlens_qwen.model_loader import ModelBundle

PRIMARY_LAYER = 16
PANIC_RECIPE = "B_STUB_DISTRESS"


@dataclass
class ReplayRun:
    run_id: str
    kind: Literal["clean", "panic"]
    experiment: str
    task_slug: str
    invocation_id: str
    prompt_len: int
    n_gen_tokens: int
    records: list[dict[str, Any]]
    temporal: dict[str, Any]
    recipe_fired: bool
    non_resolving: bool


def load_u_cond(condition_pt: str | Path) -> torch.Tensor:
    data = torch.load(condition_pt, map_location="cpu", weights_only=False)
    u = data.get("u_hat")
    if u is None:
        raise ValueError(f"missing u_hat in {condition_pt}")
    t = torch.as_tensor(u, dtype=torch.float32)
    n = float(t.norm().item())
    if n < 1e-12:
        raise ValueError("û_cond has zero norm")
    return t / n


def load_u_beh(behavior_pt: str | Path) -> torch.Tensor:
    data = torch.load(behavior_pt, map_location="cpu", weights_only=False)
    u = data.get("u_hat")
    if u is None:
        raise ValueError(f"missing u_hat in {behavior_pt}")
    t = torch.as_tensor(u, dtype=torch.float32)
    n = float(t.norm().item())
    if n < 1e-12:
        raise ValueError("û_beh has zero norm")
    return t / n


def _recipe_fired(chaos: dict[str, Any]) -> bool:
    timeline = (chaos.get("monitor") or {}).get("recipe_timeline") or []
    return any(
        x.get("recipe_id") == PANIC_RECIPE and x.get("severity") in ("warn", "critical")
        for x in timeline
    )


def _non_resolving(tokens: list[dict]) -> bool:
    return non_resolving_from_temporal(temporal_resolution(tokens))


@torch.no_grad()
def replay_invocation(
    bundle: ModelBundle,
    *,
    u_cond: torch.Tensor,
    experiment_data: dict[str, Any],
    inv: dict[str, Any],
    kind: Literal["clean", "panic"],
    run_id: str,
    max_gen_tokens: int = 256,
) -> ReplayRun:
    chaos = inv.get("chaos") or {}
    token_ids = [int(x) for x in (chaos.get("token_ids") or [])]
    if len(token_ids) < 8:
        raise ValueError(f"{run_id}: token_ids too short ({len(token_ids)})")
    token_ids = token_ids[:max_gen_tokens]
    tokens_meta = chaos.get("tokens") or []
    if len(tokens_meta) < len(token_ids):
        tokens_meta = tokens_meta + [{}] * (len(token_ids) - len(tokens_meta))

    prompt_ids = resolve_prompt_ids_from_session(inv, experiment_data, bundle.tokenizer)
    full_ids = prompt_ids + token_ids
    prompt_len = len(prompt_ids)
    positions = list(range(prompt_len, prompt_len + len(token_ids)))

    captured = capture_residuals_at_positions(
        bundle,
        torch.tensor(full_ids, dtype=torch.long),
        positions,
        layers=[PRIMARY_LAYER],
    )
    layer_vecs = captured.get(PRIMARY_LAYER) or {}
    if not layer_vecs:
        raise ValueError(f"{run_id}: no L{PRIMARY_LAYER} residuals captured")

    window_idxs = (
        panic_window_indices(
            tokens_meta[: len(token_ids)],
            max_new_tokens=len(token_ids),
        )
        if kind == "panic"
        else list(range(len(token_ids)))
    )

    records: list[dict[str, Any]] = []
    for gen_i, pos in enumerate(positions):
        h_list = layer_vecs.get(pos)
        if h_list is None:
            continue
        h = torch.tensor(h_list, dtype=torch.float32)
        if not torch.isfinite(h).all():
            raise ValueError(f"{run_id}: non-finite h at pos {pos}")
        s_cond = cosine_similarity(h, u_cond)
        tok_meta = tokens_meta[gen_i] if gen_i < len(tokens_meta) else {}
        b_vocab = tok_meta.get("B_vocab") or {}
        records.append(
            {
                "t": gen_i,
                "pos": pos,
                "s_cond": float(s_cond),
                "in_panic_window": gen_i in window_idxs,
                "B_distress_mass": float(b_vocab.get("distress_mass") or 0.0),
                "B_stub_mass": float(b_vocab.get("stub_mass") or 0.0),
                "flags": list(tok_meta.get("flags") or []),
            }
        )

    temporal = temporal_resolution(tokens_meta[: len(token_ids)])
    return ReplayRun(
        run_id=run_id,
        kind=kind,
        experiment=str(experiment_data.get("experiment") or ""),
        task_slug=str(experiment_data.get("task_slug") or inv.get("task_slug") or ""),
        invocation_id=str(inv.get("id") or ""),
        prompt_len=prompt_len,
        n_gen_tokens=len(token_ids),
        records=records,
        temporal=temporal,
        recipe_fired=_recipe_fired(chaos),
        non_resolving=_non_resolving(tokens_meta[: len(token_ids)]),
    )


def discover_clean_sessions(
    experiments_dir: str | Path,
    *,
    deny: dict[str, Any] | None = None,
    min_gen_tokens: int = 8,
) -> list[tuple[Path, int]]:
    """Return (session_json_path, invocation_index) for clean replay candidates."""
    root = Path(experiments_dir)
    found: list[tuple[Path, int, int]] = []
    patterns = ["*neutral_latest.json", "aider_chaos_proof_clean_latest.json"]
    seen: set[str] = set()
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            data = json.loads(path.read_text(encoding="utf-8"))
            for inv in data.get("invocations") or []:
                chaos = inv.get("chaos") or {}
                tids = chaos.get("token_ids") or []
                if len(tids) < min_gen_tokens:
                    continue
                key = f"{path.name}:{inv.get('id')}"
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    (path, int(inv.get("invocation_index") or 1), len(tids))
                )
    # Prefer longest completions first (one strong run per file ranks higher).
    found.sort(key=lambda x: (-x[2], x[0].name))
    return [(p, idx) for p, idx, _ in found]


def default_calibration_experiment_dirs(repo_root: Path | None = None) -> list[Path]:
    """Primary corpus panic dir + legacy experiments/out for clean neutral runs."""
    root = repo_root or Path(__file__).resolve().parents[3]
    return [
        root / "experiments" / "unicaa01_corpus" / "out",
        root / "experiments" / "out",
    ]


def discover_held_out_panic_sessions(
    experiments_dir: str | Path | list[str | Path],
    deny: dict[str, Any],
    *,
    min_gen_tokens: int = 8,
) -> list[tuple[Path, int]]:
    roots = (
        [Path(experiments_dir)]
        if not isinstance(experiments_dir, list)
        else [Path(p) for p in experiments_dir]
    )
    found: list[tuple[Path, int, int, str]] = []
    seen_slug: dict[str, tuple[Path, int, int]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("unicaa01_corpus_p*_panic_latest.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            task_slug = str(data.get("task_slug") or "")
            if is_denied_panic_task(task_slug, deny):
                continue
            for inv in data.get("invocations") or []:
                chaos = inv.get("chaos") or {}
                if not _recipe_fired(chaos):
                    continue
                tids = chaos.get("token_ids") or []
                if len(tids) < min_gen_tokens:
                    continue
                inv_idx = int(inv.get("invocation_index") or 1)
                n_tok = len(tids)
                prev = seen_slug.get(task_slug)
                if prev is None or n_tok > prev[2]:
                    seen_slug[task_slug] = (path, inv_idx, n_tok)
    for task_slug, (path, inv_idx, n_tok) in seen_slug.items():
        found.append((path, inv_idx, n_tok, task_slug))
    found.sort(key=lambda x: (-x[2], x[3]))
    return [(p, idx) for p, idx, _, _ in found]


def write_replay_jsonl(path: Path, run: ReplayRun, *, split: str = "unassigned") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        meta = {
            "type": "meta",
            "run_id": run.run_id,
            "kind": run.kind,
            "split": split,
            "experiment": run.experiment,
            "task_slug": run.task_slug,
            "invocation_id": run.invocation_id,
            "prompt_len": run.prompt_len,
            "n_gen_tokens": run.n_gen_tokens,
            "recipe_fired": run.recipe_fired,
            "non_resolving": run.non_resolving,
            "temporal": run.temporal,
        }
        f.write(json.dumps(meta) + "\n")
        for row in run.records:
            f.write(json.dumps({"type": "token", **row}) + "\n")


def load_replay_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "meta":
            meta = obj
        elif obj.get("type") == "token":
            rows.append(obj)
    return meta, rows


def relabel_panic_windows(path: Path) -> int:
    """Rewrite in_panic_window flags using CAST rolling gate (no model reload)."""
    meta, rows = load_replay_jsonl(path)
    if meta.get("kind") != "panic" or not rows:
        return 0
    window = set(
        panic_window_indices_from_records(rows, max_new_tokens=len(rows))
    )
    changed = 0
    for row in rows:
        new_flag = int(row.get("t") or 0) in window
        if bool(row.get("in_panic_window")) != new_flag:
            changed += 1
        row["in_panic_window"] = new_flag
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps({"type": "token", **row}) + "\n")
    return changed


def relabel_panic_agree_windows(
    path: Path,
    *,
    tau: float,
    sigma: float,
    tau_hard: float,
    gamma_floor: float = 0.5,
) -> int:
    """Intersect chaos windows with u_cond agreement (live AND semantics)."""
    meta, rows = load_replay_jsonl(path)
    if meta.get("kind") != "panic" or not rows:
        return 0
    changed = 0
    for row in rows:
        if not row.get("in_panic_window"):
            continue
        gval = gamma(float(row["s_cond"]), tau, sigma, tau_hard)
        new_flag = gval >= gamma_floor
        if bool(row.get("in_panic_window")) != new_flag:
            changed += 1
        row["in_panic_window"] = new_flag
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps({"type": "token", **row}) + "\n")
    return changed


def extract_task_num(task_slug: str) -> str | None:
    m = re.search(r"p(\d+)", task_slug)
    return m.group(1) if m else None
