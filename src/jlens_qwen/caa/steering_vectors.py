"""Compute and save UNI_CAA01 steering vectors from Method B activations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from jlens_qwen.caa.capture_manifest import (
    CaptureManifest,
    assert_not_practice_path,
    load_manifest,
    verify_on_disk,
)
from jlens_qwen.chaos.constants import PROBE_LAYERS

PositionKey = Literal["shared", "divergent"]


@dataclass
class SteeringVectorSet:
    """û_beh and û_cond per layer (UNI_CAA01 §2.2–2.3)."""

    vector_id: str = "UNI_CAA01"
    layers: list[int] = field(default_factory=lambda: list(PROBE_LAYERS))
    n_pairs: int = 0
    d: int | None = None
    prompt_len: int | None = None
    behavior: dict[int, list[float]] = field(default_factory=dict)
    condition: dict[int, list[float]] = field(default_factory=dict)
    neutral: dict[int, list[float]] = field(default_factory=dict)
    behavior_raw: dict[int, list[float]] = field(default_factory=dict)
    condition_raw: dict[int, list[float]] = field(default_factory=dict)
    raw_norms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    pair_slugs: list[str] = field(default_factory=list)
    source: str = "natural"
    neutral_applied: bool = False
    production_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_tensor(vec: list[float] | None) -> torch.Tensor | None:
    if not vec:
        return None
    return torch.tensor(vec, dtype=torch.float32)


def normalize_tensor(v: torch.Tensor) -> torch.Tensor:
    n = float(v.norm().item())
    if n < 1e-12:
        return v
    return v / n


def remove_neutral_component(v: torch.Tensor, u_neutral: torch.Tensor) -> torch.Tensor:
    """v' = v - (v · û_neutral) û_neutral  (UNI_CAA01 §2.4)."""
    u = normalize_tensor(u_neutral)
    return v - torch.dot(v, u) * u


def _pair_diffs(
    capture: dict[str, Any],
    *,
    behavior_position: PositionKey = "divergent",
    condition_position: PositionKey = "divergent",
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int]]:
    """Unnormalized per-layer v_beh and v_cond for one pair."""
    layers = list(capture.get("layers") or PROBE_LAYERS)
    calm = capture.get("calm") or {}
    bad = capture.get("bad") or {}
    v_beh: dict[int, torch.Tensor] = {}
    v_cond: dict[int, torch.Tensor] = {}
    used_layers: list[int] = []

    for L in layers:
        Ls = str(L)
        calm_h = (calm.get(L) or calm.get(Ls) or {}).get(behavior_position)
        bad_h = (bad.get(L) or bad.get(Ls) or {}).get(behavior_position)
        if calm_h is None or bad_h is None:
            continue
        calm_t = _to_tensor(calm_h)
        bad_t = _to_tensor(bad_h)
        if calm_t is None or bad_t is None:
            continue
        v_beh[L] = calm_t - bad_t

        calm_c = (calm.get(L) or calm.get(Ls) or {}).get(condition_position) or calm_h
        bad_c = (bad.get(L) or bad.get(Ls) or {}).get(condition_position) or bad_h
        calm_ct = _to_tensor(calm_c)
        bad_ct = _to_tensor(bad_c)
        if calm_ct is None or bad_ct is None:
            continue
        v_cond[L] = bad_ct - calm_ct
        used_layers.append(L)

    return v_beh, v_cond, used_layers


def discover_natural_captures(root: str | Path | list[str | Path]) -> list[Path]:
    """Only pairs/natural/**/activations_at_d.pt with pool_eligible admission.

    Accepts one root or many (e.g. legacy UNI_CAA01 + UNI_CAA01_corpus).
    Hard-rejects practice_pair paths. Empty list is correct when no natural pairs exist.
    """
    roots = [root] if not isinstance(root, list) else root
    found: list[Path] = []
    seen: set[str] = set()
    for root_item in roots:
        root_path = Path(root_item)
        pairs_dir = root_path / "pairs" / "natural"
        if not pairs_dir.is_dir():
            continue

        for sub in sorted(pairs_dir.iterdir()):
            if not sub.is_dir():
                continue
            assert_not_practice_path(sub)
            act = sub / "activations_at_d.pt"
            if not act.is_file():
                # Prefer .pt; allow json only if .pt missing
                act = sub / "activations_at_d.json"
                if not act.is_file():
                    continue
            key = str(act.resolve())
            if key in seen:
                continue
            try:
                manifest = load_manifest(sub)
            except FileNotFoundError:
                continue
            if not manifest.pool_eligible(capture_dir=sub):
                continue
            verify = verify_on_disk(sub, manifest)
            if not verify["pass"]:
                continue
            adm = sub / "admission_gate.json"
            if not adm.is_file():
                continue
            adm_data = json.loads(adm.read_text(encoding="utf-8"))
            if not adm_data.get("pass") or not adm_data.get("pool_eligible"):
                continue
            if adm_data.get("source") != "natural":
                continue
            seen.add(key)
            found.append(act)
    return found


def aggregate_steering_vectors(
    captures: list[dict[str, Any]],
    *,
    vector_id: str = "UNI_CAA01",
    behavior_position: PositionKey = "divergent",
    condition_position: PositionKey = "divergent",
    neutral: dict[int, list[float]] | None = None,
    require_natural: bool = True,
    require_neutral_for_production: bool = True,
    capture_paths: list[Path] | None = None,
) -> SteeringVectorSet:
    """Pool Method B pairs: v = mean_i(h_calm_i - h_bad_i), then normalize per layer.

    Fail-closed: empty captures raise. Practice paths raise when require_natural.
    """
    if not captures:
        raise ValueError(
            "aggregate_steering_vectors: empty natural set — fail closed "
            "(no silent practice fallback)"
        )

    if capture_paths is not None:
        for p in capture_paths:
            assert_not_practice_path(p)

    if require_natural:
        for i, cap in enumerate(captures):
            src = cap.get("source") or cap.get("manifest_source")
            if src != "natural":
                raise ValueError(
                    f"aggregate_steering_vectors: capture[{i}] source={src!r} "
                    f"— explicit source='natural' required (fail closed)"
                )
            path_hint = str(cap.get("path_hint") or "")
            if "practice_pair" in path_hint.replace("\\", "/"):
                raise ValueError(
                    f"aggregate_steering_vectors: practice_pair path in capture[{i}]"
                )

    layers = list(captures[0].get("layers") or PROBE_LAYERS)
    out = SteeringVectorSet(
        vector_id=vector_id,
        layers=layers,
        n_pairs=len(captures),
        d=captures[0].get("d"),
        prompt_len=captures[0].get("prompt_len"),
        source="natural",
    )

    beh_acc: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    cond_acc: dict[int, list[torch.Tensor]] = {L: [] for L in layers}

    for cap in captures:
        slug = cap.get("task_slug") or cap.get("pair_slug") or ""
        if slug:
            out.pair_slugs.append(str(slug))
        v_beh, v_cond, used = _pair_diffs(
            cap,
            behavior_position=behavior_position,
            condition_position=condition_position,
        )
        for L in used:
            if L in v_beh:
                beh_acc[L].append(v_beh[L])
            if L in v_cond:
                cond_acc[L].append(v_cond[L])

    u_neutral: dict[int, torch.Tensor] = {}
    if neutral:
        for L, vec in neutral.items():
            t = _to_tensor(vec)
            if t is not None:
                u_neutral[int(L)] = normalize_tensor(t)
        out.neutral = {int(L): v.tolist() for L, v in u_neutral.items()}

    for L in layers:
        if beh_acc[L]:
            mean_beh = torch.stack(beh_acc[L], dim=0).mean(dim=0)
            out.behavior_raw[L] = mean_beh.tolist()
            out.raw_norms[f"behavior_L{L}"] = float(mean_beh.norm().item())
            cleaned = mean_beh
            if L in u_neutral:
                cleaned = remove_neutral_component(cleaned, u_neutral[L])
            out.behavior[L] = normalize_tensor(cleaned).tolist()
        if cond_acc[L]:
            mean_cond = torch.stack(cond_acc[L], dim=0).mean(dim=0)
            out.condition_raw[L] = mean_cond.tolist()
            out.raw_norms[f"condition_L{L}"] = float(mean_cond.norm().item())
            cleaned = mean_cond
            if L in u_neutral:
                cleaned = remove_neutral_component(cleaned, u_neutral[L])
            out.condition[L] = normalize_tensor(cleaned).tolist()

    if u_neutral:
        out.neutral_applied = True
        out.notes.append("neutral subspace cleanup applied (§2.4)")

    if not out.behavior or not out.condition:
        out.notes.append("incomplete pooled vectors — check captures")

    if out.n_pairs < 3:
        out.notes.append(
            f"n_pairs={out.n_pairs}: bootstrap only — need ≥3 admitted natural "
            "pairs before production_ready"
        )
        out.production_ready = False
    else:
        out.production_ready = bool(out.neutral_applied) if require_neutral_for_production else True

    if require_neutral_for_production and not out.neutral_applied:
        out.production_ready = False
        out.notes.append("neutral_applied=false — production write blocked")

    return out


def compute_steering_vectors(
    capture: dict[str, Any],
    *,
    vector_id: str = "UNI_CAA01",
    behavior_position: PositionKey = "divergent",
    condition_position: PositionKey = "divergent",
    neutral: dict[int, list[float]] | None = None,
) -> SteeringVectorSet:
    """Build û_beh and û_cond from one matched-pair activation capture (debug helper)."""
    # Single-pair debug path: do not require natural source tag.
    cap = dict(capture)
    cap.setdefault("source", capture.get("source") or "natural")
    return aggregate_steering_vectors(
        [cap],
        vector_id=vector_id,
        behavior_position=behavior_position,
        condition_position=condition_position,
        neutral=neutral,
        require_natural=False,
        require_neutral_for_production=False,
    )


def save_steering_artifacts(
    vectors: SteeringVectorSet,
    out_dir: str | Path,
    *,
    metadata_extra: dict[str, Any] | None = None,
    require_neutral: bool = False,
) -> dict[str, str]:
    """Write behavior_L*.pt, condition_L*.pt, metadata.json per UNI_CAA01 §2.6."""
    if require_neutral and not vectors.neutral_applied:
        raise ValueError("production write requires neutral_applied=true (AGG8)")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for L in vectors.layers:
        if L in vectors.behavior:
            p = out / f"behavior_L{L}.pt"
            torch.save(
                {
                    "vector_id": vectors.vector_id,
                    "layer": L,
                    "kind": "behavior",
                    "u_hat": torch.tensor(vectors.behavior[L], dtype=torch.float32),
                    "n_pairs": vectors.n_pairs,
                },
                p,
            )
            written[f"behavior_L{L}"] = str(p)
        if L in vectors.condition:
            p = out / f"condition_L{L}.pt"
            torch.save(
                {
                    "vector_id": vectors.vector_id,
                    "layer": L,
                    "kind": "condition",
                    "u_hat": torch.tensor(vectors.condition[L], dtype=torch.float32),
                    "n_pairs": vectors.n_pairs,
                },
                p,
            )
            written[f"condition_L{L}"] = str(p)
        if L in vectors.neutral:
            p = out / f"neutral_L{L}.pt"
            torch.save(
                {
                    "vector_id": vectors.vector_id,
                    "layer": L,
                    "kind": "neutral",
                    "u_hat": torch.tensor(vectors.neutral[L], dtype=torch.float32),
                    "n_pairs": vectors.n_pairs,
                },
                p,
            )
            written[f"neutral_L{L}"] = str(p)

    meta = {
        "vector_id": vectors.vector_id,
        "layers": vectors.layers,
        "n_pairs": vectors.n_pairs,
        "d": vectors.d,
        "prompt_len": vectors.prompt_len,
        "raw_norms": vectors.raw_norms,
        "notes": vectors.notes,
        "pair_slugs": vectors.pair_slugs,
        "source": vectors.source,
        "neutral_applied": vectors.neutral_applied,
        "production_ready": vectors.production_ready,
        "default_inject_layer": 16,
        "alpha_0": 0.85,
        "tau": None,
        "sigma": None,
    }
    if metadata_extra:
        meta.update(metadata_extra)

    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["metadata"] = str(meta_path)
    return written


def load_capture(path: str | Path) -> dict[str, Any]:
    """Load activation capture from .pt (preferred) or legacy .json."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix == ".pt":
        data = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            return _normalize_capture_format(data)
        raise ValueError(f"unexpected capture payload in {p}")
    return _normalize_capture_format(json.loads(p.read_text(encoding="utf-8")))


def _normalize_capture_format(capture: dict[str, Any]) -> dict[str, Any]:
    """Accept new shared/divergent layout or legacy position-indexed JSON."""
    calm = capture.get("calm") or {}
    if not calm:
        return capture
    sample = next(iter(calm.values()), {})
    if isinstance(sample, dict) and ("shared" in sample or "divergent" in sample):
        return capture

    positions = list(capture.get("positions") or [])
    if len(positions) >= 2:
        shared_pos, div_pos = int(positions[0]), int(positions[1])
    elif len(positions) == 1:
        shared_pos = div_pos = int(positions[0])
    else:
        return capture

    new_calm: dict[int, dict[str, list[float]]] = {}
    new_bad: dict[int, dict[str, list[float]]] = {}

    def _pos_lookup(by_pos: dict, pos: int) -> list[float] | None:
        if pos in by_pos:
            return by_pos[pos]
        sp = str(pos)
        if sp in by_pos:
            return by_pos[sp]
        return None

    for key, by_pos in calm.items():
        L = int(key)
        new_calm[L] = {}
        sv = _pos_lookup(by_pos, shared_pos)
        dv = _pos_lookup(by_pos, div_pos)
        if sv is not None:
            new_calm[L]["shared"] = sv
        if dv is not None:
            new_calm[L]["divergent"] = dv
    for key, by_pos in (capture.get("bad") or {}).items():
        L = int(key)
        new_bad[L] = {}
        sv = _pos_lookup(by_pos, shared_pos)
        dv = _pos_lookup(by_pos, div_pos)
        if sv is not None:
            new_bad[L]["shared"] = sv
        if dv is not None:
            new_bad[L]["divergent"] = dv

    out = dict(capture)
    out["calm"] = new_calm
    out["bad"] = new_bad
    out["capture_method"] = out.get("capture_method") or "legacy_position_index"
    return out
