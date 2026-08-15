#!/usr/bin/env python
"""Extract UNI_CAA01 û_beh / û_cond — natural-only by default (fail closed).

Default geometry (UNI_CAA01 §2.2–2.3):
  layers = PROBE_LAYERS (L10, L16, L22)
  position = divergent (h at first divergent token d)
  v_beh = mean(h_calm - h_bad);  v_cond = mean(h_bad - h_calm)
  then optional û_neutral cleanup + L2 normalize per layer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    PositionKey,
    aggregate_steering_vectors,
    compute_steering_vectors,
    discover_natural_captures,
    load_capture,
    normalize_tensor,
    save_steering_artifacts,
)
from jlens_qwen.chaos.constants import PROBE_LAYERS  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "caa" / "UNI_CAA01"
SANDBOX = DEFAULT_OUT / "_sandbox_practice"
PATCH5_OUT = DEFAULT_OUT / "_patch5_steering"
NEUTRAL_DIR = DEFAULT_OUT
REQUIRED_LAYERS = tuple(PROBE_LAYERS)


def _load_neutral() -> dict[int, list[float]] | None:
    out: dict[int, list[float]] = {}
    for L in REQUIRED_LAYERS:
        p = NEUTRAL_DIR / f"neutral_L{L}.pt"
        if not p.is_file():
            return None
        data = torch.load(p, map_location="cpu", weights_only=False)
        u = data.get("u_hat")
        if u is None:
            return None
        out[int(L)] = u.tolist() if hasattr(u, "tolist") else list(u)
    return out


def _validate_capture(
    cap: dict,
    path: Path,
    *,
    position: PositionKey,
    required_layers: tuple[int, ...],
) -> list[str]:
    """Hard checks before pooling — return human-readable errors (empty = ok)."""
    errs: list[str] = []
    calm = cap.get("calm") or {}
    bad = cap.get("bad") or {}
    if not calm or not bad:
        errs.append(f"{path}: missing calm/bad blocks")
        return errs
    for L in required_layers:
        Ls = str(L)
        c_block = calm.get(L) or calm.get(Ls) or {}
        b_block = bad.get(L) or bad.get(Ls) or {}
        c_h = c_block.get(position) if isinstance(c_block, dict) else None
        b_h = b_block.get(position) if isinstance(b_block, dict) else None
        if c_h is None or b_h is None:
            errs.append(f"{path.name}: L{L} missing '{position}' on calm or bad")
            continue
        if len(c_h) != len(b_h):
            errs.append(
                f"{path.name}: L{L} dim mismatch calm={len(c_h)} bad={len(b_h)}"
            )
            continue
        if len(c_h) < 8:
            errs.append(f"{path.name}: L{L} absurdly short hidden dim={len(c_h)}")
            continue
        c_t = torch.tensor(c_h, dtype=torch.float32)
        b_t = torch.tensor(b_h, dtype=torch.float32)
        if not torch.isfinite(c_t).all() or not torch.isfinite(b_t).all():
            errs.append(f"{path.name}: L{L} non-finite activations")
            continue
        diff = c_t - b_t
        if float(diff.norm().item()) < 1e-6:
            errs.append(f"{path.name}: L{L} calm≈bad at {position} (||diff||~0)")
    d = cap.get("d")
    if d is None or int(d) < 0:
        errs.append(f"{path.name}: invalid d={d!r}")
    return errs


# Minimum ||mean(h_calm-h_bad)|| before normalize — catches near-zero / shared-position mistakes.
# Patch-5 observed ~43/58/151 at L10/16/22; floor is deliberately conservative.
RAW_NORM_FLOOR = {10: 5.0, 16: 5.0, 22: 10.0}


def _clear_vector_artifacts(out_dir: Path) -> list[str]:
    """Remove prior steering .pt/json stamps so re-runs cannot leave stale layers."""
    removed: list[str] = []
    if not out_dir.is_dir():
        return removed
    for pat in (
        "behavior_L*.pt",
        "condition_L*.pt",
        "neutral_L*.pt",
        "metadata.json",
        "extract_gate.json",
    ):
        for p in out_dir.glob(pat):
            p.unlink()
            removed.append(p.name)
    return removed


def _post_write_verify(
    out_dir: Path,
    *,
    expected_layers: tuple[int, ...],
    n_pairs: int,
    position: PositionKey,
    raw_norms: dict[str, float],
    patch_run: bool,
) -> dict:
    """Reload written artifacts and assert unit norms / signal / metadata."""
    errors: list[str] = []
    checks: list[dict] = []
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        return {"pass": False, "errors": ["missing metadata.json"], "checks": []}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("extraction_position") != position:
        errors.append(
            f"metadata extraction_position={meta.get('extraction_position')!r} "
            f"!= {position!r}"
        )
    if int(meta.get("n_pairs") or 0) != n_pairs:
        errors.append(f"metadata n_pairs={meta.get('n_pairs')} != {n_pairs}")
    if not meta.get("neutral_applied"):
        errors.append("metadata neutral_applied is false")
    if patch_run and meta.get("production_ready"):
        errors.append("patch_run must stamp production_ready=false")

    for L in expected_layers:
        key = f"behavior_L{L}"
        raw = float(raw_norms.get(key) or 0.0)
        floor = float(RAW_NORM_FLOOR.get(L, 5.0))
        ok_raw = raw >= floor
        if not ok_raw:
            errors.append(
                f"L{L}: raw_norm {raw:.4f} < floor {floor} "
                f"(near-zero mean-diff / wrong position?)"
            )

        bp = out_dir / f"behavior_L{L}.pt"
        cp = out_dir / f"condition_L{L}.pt"
        if not bp.is_file() or not cp.is_file():
            errors.append(f"missing behavior/condition L{L}")
            continue
        beh = torch.load(bp, map_location="cpu", weights_only=False)
        cond = torch.load(cp, map_location="cpu", weights_only=False)
        u_b = normalize_tensor(torch.as_tensor(beh["u_hat"], dtype=torch.float32))
        u_c = normalize_tensor(torch.as_tensor(cond["u_hat"], dtype=torch.float32))
        nb = float(u_b.norm().item())
        nc = float(u_c.norm().item())
        dot = float(torch.dot(u_b, u_c).item())
        ok_norm = abs(nb - 1.0) < 1e-4 and abs(nc - 1.0) < 1e-4
        # Consistency only (same-position math ⇒ antipodal); real signal = raw_norm floor.
        ok_anti = dot < -0.85
        ok_finite = bool(torch.isfinite(u_b).all() and torch.isfinite(u_c).all())
        detail = {
            "layer": L,
            "norm_beh": nb,
            "norm_cond": nc,
            "dot_beh_cond": dot,
            "raw_norm_beh": raw,
            "raw_norm_floor": floor,
            "ok_raw_signal": ok_raw,
            "ok_unit_norm": ok_norm,
            "ok_anti_aligned": ok_anti,
            "ok_finite": ok_finite,
            "dim": int(u_b.numel()),
        }
        checks.append(detail)
        if not ok_norm:
            errors.append(f"L{L}: non-unit norms beh={nb} cond={nc}")
        if not ok_anti:
            errors.append(f"L{L}: beh·cond={dot:.4f} expected < -0.85 (anti-aligned)")
        if not ok_finite:
            errors.append(f"L{L}: non-finite u_hat")
        if beh.get("kind") != "behavior" or cond.get("kind") != "condition":
            errors.append(f"L{L}: kind mismatch in .pt payloads")
        if int(beh.get("n_pairs") or 0) != n_pairs:
            errors.append(f"L{L}: behavior n_pairs stamp mismatch")

    return {"pass": not errors, "errors": errors, "checks": checks, "metadata": meta}


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract UNI_CAA01 steering vectors")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--vector-id", default="UNI_CAA01")
    ap.add_argument(
        "--sandbox-practice",
        action="store_true",
        help="Extract from practice_pair into _sandbox_practice/ only (never production)",
    )
    ap.add_argument("--capture", type=Path, default=None, help="Single capture (sandbox/debug)")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N discovered natural pairs (patch / smoke)",
    )
    ap.add_argument(
        "--position",
        choices=("divergent", "shared"),
        default="divergent",
        help="Activation position for v_beh / v_cond (default: divergent)",
    )
    ap.add_argument(
        "--allow-overwrite-production",
        action="store_true",
        help="Permit writing into the production UNI_CAA01 root (full extract only)",
    )
    ap.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also pool legacy UNI_CAA01/pairs/natural (default: corpus-only)",
    )
    args = ap.parse_args()
    position: PositionKey = args.position  # type: ignore[assignment]

    if args.sandbox_practice:
        capture_path = (
            args.capture
            or DEFAULT_OUT / "practice_pair" / "activations_at_d.pt"
        )
        if not capture_path.is_file():
            capture_path = DEFAULT_OUT / "practice_pair" / "activations_at_d.json"
        if not capture_path.is_file():
            print(f"Practice capture not found: {capture_path}", file=sys.stderr)
            return 1
        capture = load_capture(capture_path)
        capture["source"] = "practice"
        vectors = compute_steering_vectors(
            capture,
            vector_id=args.vector_id,
            behavior_position=position,
            condition_position=position,
        )
        written = save_steering_artifacts(
            vectors,
            SANDBOX,
            metadata_extra={
                "source_capture": str(capture_path),
                "source": "practice",
                "sandbox": True,
                "extraction_position": position,
            },
            require_neutral=False,
        )
        print(json.dumps({"pass": True, "sandbox": True, "artifacts": written}, indent=2))
        return 0

    # Production / patch path: natural only — corpus by default
    from jlens_qwen.caa.unicaa01_corpus import CAA_CORPUS_ROOT

    discover_roots: list[Path] = [CAA_CORPUS_ROOT]
    if args.include_legacy:
        discover_roots.append(DEFAULT_OUT)
    paths = discover_natural_captures(discover_roots)
    inclusion = {
        "roots": [str(r) for r in discover_roots],
        "include_legacy": bool(args.include_legacy),
        "n_discovered": len(paths),
        "discovered_slugs": [p.parent.name for p in paths],
    }
    if not paths:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "empty natural set — fail closed (AGG1)",
                    "n_pairs": 0,
                    "inclusion": inclusion,
                    "hint": "Complete Phase 3B after calm-tail curation; do not use practice",
                },
                indent=2,
            )
        )
        return 1

    if args.limit is not None:
        if args.limit < 1:
            print(json.dumps({"pass": False, "error": "--limit must be >= 1"}, indent=2))
            return 1
        paths = paths[: args.limit]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = PATCH5_OUT if args.limit is not None else DEFAULT_OUT

    # Guard: limited / patch runs must not clobber production root unless forced
    prod_resolved = DEFAULT_OUT.resolve()
    if out_dir.resolve() == prod_resolved and (
        args.limit is not None or not args.allow_overwrite_production
    ):
        if args.limit is not None:
            print(
                json.dumps(
                    {
                        "pass": False,
                        "error": (
                            "refusing to write limited (--limit) extract into "
                            "production UNI_CAA01 root; use --out-dir or default "
                            f"{PATCH5_OUT}"
                        ),
                    },
                    indent=2,
                )
            )
            return 1
        if not args.allow_overwrite_production:
            print(
                json.dumps(
                    {
                        "pass": False,
                        "error": (
                            "refusing production overwrite without "
                            "--allow-overwrite-production"
                        ),
                        "hint": f"pass --out-dir {PATCH5_OUT} for a safe test write",
                    },
                    indent=2,
                )
            )
            return 1

    captures: list[dict] = []
    validate_errors: list[str] = []
    used_paths: list[str] = []
    for p in paths:
        try:
            cap = load_capture(p)
        except Exception as exc:  # noqa: BLE001 — surface load failures explicitly
            validate_errors.append(f"{p}: load failed: {exc}")
            continue
        cap["source"] = "natural"
        cap["path_hint"] = str(p)
        errs = _validate_capture(
            cap, p, position=position, required_layers=REQUIRED_LAYERS
        )
        if errs:
            validate_errors.extend(errs)
            continue
        captures.append(cap)
        used_paths.append(str(p))

    if validate_errors:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "capture validation failed",
                    "n_ok": len(captures),
                    "n_requested": len(paths),
                    "validate_errors": validate_errors,
                    "inclusion": inclusion,
                },
                indent=2,
            )
        )
        return 1

    if args.limit is not None and len(captures) != args.limit:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": f"expected {args.limit} validated captures, got {len(captures)}",
                    "inclusion": inclusion,
                },
                indent=2,
            )
        )
        return 1

    neutral = _load_neutral()
    if neutral is None:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "neutral artifacts required for production write (AGG8)",
                },
                indent=2,
            )
        )
        return 1

    vectors = aggregate_steering_vectors(
        captures,
        vector_id=args.vector_id,
        behavior_position=position,
        condition_position=position,
        neutral=neutral,
        require_natural=True,
        require_neutral_for_production=True,
        capture_paths=[Path(p) for p in used_paths],
    )
    patch_run = args.limit is not None
    if patch_run:
        vectors.production_ready = False
        vectors.notes.append("patch_run: production_ready forced false")

    if not vectors.production_ready and not patch_run:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "production_ready false — need ≥3 admitted natural pairs",
                    "n_pairs": vectors.n_pairs,
                    "notes": vectors.notes,
                    "inclusion": inclusion,
                },
                indent=2,
            )
        )
        return 1
    if patch_run and (not vectors.behavior or not vectors.condition):
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "incomplete vectors after patch aggregate",
                    "n_pairs": vectors.n_pairs,
                    "notes": vectors.notes,
                },
                indent=2,
            )
        )
        return 1

    cleared = _clear_vector_artifacts(out_dir)
    d_values = sorted({int(c["d"]) for c in captures if c.get("d") is not None})
    written = save_steering_artifacts(
        vectors,
        out_dir,
        metadata_extra={
            "source": "natural",
            "n_paths": len(used_paths),
            "extraction_position": position,
            "required_layers": list(REQUIRED_LAYERS),
            "pair_paths": used_paths,
            "patch_limit": args.limit,
            "patch_run": patch_run,
            "production_ready": vectors.production_ready,
            "d_values": d_values,
            "inclusion": inclusion,
            "cleared_prior_artifacts": cleared,
        },
        require_neutral=True,
    )
    verify = _post_write_verify(
        out_dir,
        expected_layers=REQUIRED_LAYERS,
        n_pairs=vectors.n_pairs,
        position=position,
        raw_norms=vectors.raw_norms,
        patch_run=patch_run,
    )
    gate = {
        "pass": bool(vectors.behavior and vectors.condition) and verify["pass"],
        "n_pairs": vectors.n_pairs,
        "production_ready": vectors.production_ready,
        "neutral_applied": vectors.neutral_applied,
        "extraction_position": position,
        "layers": list(REQUIRED_LAYERS),
        "out_dir": str(out_dir),
        "pair_paths": used_paths,
        "inclusion": inclusion,
        "artifacts": written,
        "notes": vectors.notes,
        "post_write_verify": {
            "pass": verify["pass"],
            "errors": verify["errors"],
            "checks": verify["checks"],
        },
        "raw_norms": vectors.raw_norms,
        "d_values": d_values,
    }
    print(json.dumps(gate, indent=2))
    (out_dir / "extract_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
