"""Capture integrity (CAP) and admission (ADM) checks for Method B pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jlens_qwen.caa.capture_manifest import CaptureManifest
from jlens_qwen.chaos.constants import PROBE_LAYERS


def verify_capture_integrity(
    capture: dict[str, Any],
    *,
    manifest: CaptureManifest | None = None,
    d: int | None = None,
    prompt_len: int | None = None,
) -> dict[str, Any]:
    """CAP1–CAP6 on activations_at_d payload."""
    checks: list[dict[str, Any]] = []
    d_use = int(d if d is not None else (manifest.d if manifest else capture.get("d") or -1))
    pl = int(
        prompt_len
        if prompt_len is not None
        else (manifest.prompt_len if manifest else capture.get("prompt_len") or -1)
    )

    checks.append(
        {
            "id": "CAP1_d",
            "pass": int(capture.get("d", -2)) == d_use and d_use >= 0,
            "detail": f"capture.d={capture.get('d')} expected={d_use}",
        }
    )
    checks.append(
        {
            "id": "CAP2_prompt_len",
            "pass": int(capture.get("prompt_len", -2)) == pl and pl >= 0,
            "detail": f"capture.prompt_len={capture.get('prompt_len')} expected={pl}",
        }
    )

    positions = capture.get("positions") or {}
    exp_shared = pl + d_use - 1 if d_use > 0 else max(pl - 1, 0)
    exp_div = pl + d_use
    checks.append(
        {
            "id": "CAP3_shared_pos",
            "pass": positions.get("shared") == exp_shared,
            "detail": f"got={positions.get('shared')} expected={exp_shared}",
        }
    )
    checks.append(
        {
            "id": "CAP4_divergent_pos",
            "pass": positions.get("divergent") == exp_div,
            "detail": f"got={positions.get('divergent')} expected={exp_div}",
        }
    )

    layers = list(capture.get("layers") or PROBE_LAYERS)
    calm = capture.get("calm") or {}
    bad = capture.get("bad") or {}
    both_ok = True
    shared_eq = True
    for L in layers:
        Ls = str(L)
        c = calm.get(L) or calm.get(Ls) or {}
        b = bad.get(L) or bad.get(Ls) or {}
        if "shared" not in c or "divergent" not in c or "shared" not in b or "divergent" not in b:
            both_ok = False
        if c.get("shared") != b.get("shared"):
            shared_eq = False

    checks.append(
        {
            "id": "CAP5_both_arms_layers",
            "pass": both_ok and set(layers) == set(PROBE_LAYERS),
            "detail": f"layers={layers}",
        }
    )
    checks.append(
        {
            "id": "CAP6_shared_identical",
            "pass": shared_eq and both_ok,
            "detail": "calm[L].shared == bad[L].shared",
        }
    )

    finite_ok = True
    for arm_name, arm in (("calm", calm), ("bad", bad)):
        for L in layers:
            Ls = str(L)
            slot = arm.get(L) or arm.get(Ls) or {}
            for pos_key in ("shared", "divergent"):
                vec = slot.get(pos_key)
                if vec is None:
                    continue
                try:
                    vals = list(vec)
                except TypeError:
                    finite_ok = False
                    break
                if not vals or any(
                    (isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))))
                    or x is None
                    for x in vals
                ):
                    finite_ok = False
                    break
            if not finite_ok:
                break
        if not finite_ok:
            break
    checks.append(
        {
            "id": "CAP7_finite_activations",
            "pass": finite_ok and both_ok,
            "detail": "all shared/divergent activation vectors non-empty and finite",
        }
    )

    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def admission_checks(
    *,
    pair_validate_pass: bool,
    cap_pass: bool,
    manifest: CaptureManifest,
    capture_dir: Path,
) -> dict[str, Any]:
    """ADM1–ADM5."""
    path_s = str(capture_dir).replace("\\", "/")
    checks = [
        {
            "id": "ADM1_validate_pair",
            "pass": bool(pair_validate_pass),
            "detail": f"pair_validate_pass={pair_validate_pass}",
        },
        {
            "id": "ADM2_cap",
            "pass": bool(cap_pass),
            "detail": f"cap_pass={cap_pass}",
        },
        {
            "id": "ADM3_kv_live",
            "pass": bool(manifest.kv_live_gate_pass),
            "detail": f"kv_live_gate_pass={manifest.kv_live_gate_pass}",
        },
        {
            "id": "ADM4_source_natural",
            "pass": manifest.source == "natural",
            "detail": f"source={manifest.source}",
        },
        {
            "id": "ADM5_not_practice_path",
            "pass": "practice_pair" not in path_s,
            "detail": f"path={path_s}",
        },
    ]
    ok = all(c["pass"] for c in checks)
    pool = ok and manifest.pool_eligible(capture_dir=capture_dir)
    return {"pass": ok, "pool_eligible": pool, "checks": checks}
