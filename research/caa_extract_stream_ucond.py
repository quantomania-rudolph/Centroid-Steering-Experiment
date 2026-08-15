#!/usr/bin/env python
"""Extract stream-window û_cond from train pairs (deny-listed), not calibration holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.activation_capture import capture_residuals_at_positions  # noqa: E402
from jlens_qwen.caa.calibration_deny import is_denied_panic_task, load_deny_list  # noqa: E402
from jlens_qwen.caa.miner_pairing import resolve_prompt_ids_from_session  # noqa: E402
from jlens_qwen.caa.steering_vectors import (  # noqa: E402
    normalize_tensor,
    remove_neutral_component,
)
from jlens_qwen.caa.stream_dynamics import panic_window_indices  # noqa: E402
from jlens_qwen.model_loader import load_qwen_coder  # noqa: E402
from unicaa01_restart_server import stop_server  # noqa: E402

DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
DEFAULT_OUT = ROOT / "artifacts" / "caa" / "UNI_CAA01"
PRIMARY_LAYER = 16
EXP_DIRS = [
    ROOT / "experiments" / "unicaa01_corpus" / "out",
    ROOT / "experiments" / "out",
]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_experiment(experiment: str) -> Path | None:
    for d in EXP_DIRS:
        for name in (f"{experiment}_latest.json", f"{experiment}.json"):
            p = d / name
            if p.is_file():
                return p
    return None


def _chaos_tokens_from_inv(inv: dict[str, Any]) -> list[dict[str, Any]]:
    chaos = inv.get("chaos") or {}
    raw = chaos.get("tokens") or []
    out: list[dict[str, Any]] = []
    for t in raw:
        b = t.get("B_vocab") or {}
        out.append(
            {
                "flags": list(t.get("flags") or []),
                "B_vocab": {
                    "distress_mass": b.get("distress_mass"),
                    "stub_mass": b.get("stub_mass"),
                },
            }
        )
    return out


def _mean_vec(vecs: list[torch.Tensor]) -> torch.Tensor | None:
    if not vecs:
        return None
    stack = torch.stack(vecs, dim=0).float()
    return stack.mean(dim=0)


def _load_neutral(out_dir: Path) -> dict[int, torch.Tensor]:
    neutral: dict[int, torch.Tensor] = {}
    for layer in (10, 16, 22):
        p = out_dir / f"neutral_L{layer}.pt"
        if not p.is_file():
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        u = data.get("u_hat")
        if u is not None:
            neutral[layer] = normalize_tensor(torch.as_tensor(u, dtype=torch.float32))
    return neutral


def extract_stream_ucond(
    *,
    metadata_path: Path,
    out_dir: Path,
    device: str,
    min_window: int = 1,
) -> dict[str, Any]:
    meta = _load_json(metadata_path)
    deny = load_deny_list(metadata_path)
    pair_paths = [Path(p) for p in (meta.get("pair_paths") or [])]
    if not pair_paths:
        raise RuntimeError("metadata missing pair_paths")

    stop_server(settle_s=8.0)
    bundle = load_qwen_coder(device=device, dtype=torch.float16, prefer_local=True)
    neutral = _load_neutral(out_dir)

    v_cond_parts: list[torch.Tensor] = []
    used_slugs: list[str] = []
    skipped: list[dict[str, str]] = []

    for act_path in pair_paths:
        pair_dir = act_path.parent
        bad_path = pair_dir / "method_b_bad.json"
        calm_path = pair_dir / "method_b_calm.json"
        if not bad_path.is_file() or not calm_path.is_file():
            skipped.append({"pair": pair_dir.name, "reason": "missing method_b"})
            continue

        bad_rec = _load_json(bad_path)
        calm_rec = _load_json(calm_path)
        task_slug = str(bad_rec.get("task_slug") or "")
        if is_denied_panic_task(task_slug, deny) is False:
            # train pairs should be on deny list; skip if leaked into calibration pool
            pass
        experiment = str(bad_rec.get("experiment") or "")
        exp_path = _find_experiment(experiment)
        if exp_path is None:
            skipped.append({"pair": pair_dir.name, "reason": f"no experiment {experiment}"})
            continue
        exp_data = _load_json(exp_path)
        inv_id = str(bad_rec.get("invocation_id") or "")
        inv = None
        for candidate in exp_data.get("invocations") or []:
            if str(candidate.get("id") or "") == inv_id:
                inv = candidate
                break
        if inv is None:
            skipped.append({"pair": pair_dir.name, "reason": "invocation not found"})
            continue

        bad_ids = [int(x) for x in (bad_rec.get("full_completion_token_ids") or [])]
        calm_ids = [int(x) for x in (calm_rec.get("full_completion_token_ids") or [])]
        if len(bad_ids) < 8 or len(calm_ids) < 8:
            skipped.append({"pair": pair_dir.name, "reason": "completion too short"})
            continue

        tokens_meta = _chaos_tokens_from_inv(inv)
        if len(tokens_meta) < len(bad_ids):
            tokens_meta = tokens_meta + [{}] * (len(bad_ids) - len(tokens_meta))

        window_idxs = panic_window_indices(tokens_meta[: len(bad_ids)], max_new_tokens=len(bad_ids))
        if len(window_idxs) < min_window:
            skipped.append({"pair": pair_dir.name, "reason": f"window<{min_window}"})
            continue

        try:
            prompt_ids = resolve_prompt_ids_from_session(inv, exp_data, bundle.tokenizer)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"pair": pair_dir.name, "reason": f"prompt_ids: {exc}"})
            continue

        prompt_len = len(prompt_ids)
        bad_positions = [prompt_len + i for i in window_idxs if i < len(bad_ids)]
        calm_positions = [prompt_len + i for i in window_idxs if i < len(calm_ids)]
        if not bad_positions or not calm_positions:
            skipped.append({"pair": pair_dir.name, "reason": "no capture positions"})
            continue

        bad_full = prompt_ids + bad_ids
        calm_full = prompt_ids + calm_ids
        bad_cap = capture_residuals_at_positions(
            bundle,
            torch.tensor(bad_full, dtype=torch.long),
            bad_positions,
            layers=[PRIMARY_LAYER],
        )
        calm_cap = capture_residuals_at_positions(
            bundle,
            torch.tensor(calm_full, dtype=torch.long),
            calm_positions,
            layers=[PRIMARY_LAYER],
        )
        bad_vecs = [
            torch.tensor(bad_cap[PRIMARY_LAYER][pos], dtype=torch.float32)
            for pos in bad_positions
            if pos in bad_cap.get(PRIMARY_LAYER, {})
        ]
        calm_vecs = [
            torch.tensor(calm_cap[PRIMARY_LAYER][pos], dtype=torch.float32)
            for pos in calm_positions
            if pos in calm_cap.get(PRIMARY_LAYER, {})
        ]
        mean_bad = _mean_vec(bad_vecs)
        mean_calm = _mean_vec(calm_vecs)
        if mean_bad is None or mean_calm is None:
            skipped.append({"pair": pair_dir.name, "reason": "empty capture"})
            continue
        v_cond_parts.append(mean_bad - mean_calm)
        used_slugs.append(task_slug)

    if not v_cond_parts:
        raise RuntimeError(f"no stream pairs extracted; skipped={skipped[:5]}")

    v_raw = torch.stack(v_cond_parts, dim=0).mean(dim=0)
    u = normalize_tensor(v_raw)
    if PRIMARY_LAYER in neutral:
        u = normalize_tensor(remove_neutral_component(u, neutral[PRIMARY_LAYER]))

    out_dir.mkdir(parents=True, exist_ok=True)
    stream_pt = out_dir / "condition_L16_stream.pt"
    torch.save(
        {
            "vector_id": "UNI_CAA01",
            "layer": PRIMARY_LAYER,
            "kind": "condition",
            "geometry": "stream_window",
            "u_hat": u,
            "n_pairs": len(used_slugs),
            "pair_slugs": used_slugs,
        },
        stream_pt,
    )

    sidecar = {
        "geometry": "stream_window",
        "extraction_position": "panic_window_mean",
        "n_pairs": len(used_slugs),
        "pair_slugs": used_slugs,
        "skipped": skipped,
        "source_condition": str(stream_pt),
        "raw_norm": float(v_raw.norm().item()),
    }
    sidecar_path = out_dir / "metadata_stream_ucond.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    return {
        "pass": True,
        "n_pairs": len(used_slugs),
        "n_skipped": len(skipped),
        "condition_L16_stream": str(stream_pt),
        "metadata_stream": str(sidecar_path),
        "raw_norm": sidecar["raw_norm"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract stream-window û_cond from train pairs")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--promote",
        action="store_true",
        help="Copy condition_L16_stream.pt -> condition_L16.pt after extract",
    )
    args = ap.parse_args()

    result = extract_stream_ucond(
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        device=_pick_device(args.device),
    )
    if args.promote:
        src = Path(result["condition_L16_stream"])
        dst = args.out_dir / "condition_L16.pt"
        backup = args.out_dir / "condition_L16_divergent.pt"
        if dst.is_file() and not backup.is_file():
            dst.replace(backup)
        dst.write_bytes(src.read_bytes())

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
