"""Shared-KV vs full-forward capture equivalence (UNI_CAA01 Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from jlens_qwen.chaos.constants import PROBE_LAYERS


@dataclass
class CheckResult:
    id: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "pass": self.passed, "detail": self.detail}


def _as_tensor(vec: list[float] | None) -> torch.Tensor | None:
    if not vec:
        return None
    return torch.tensor(vec, dtype=torch.float32)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm().item())
    if na < 1e-12:
        return float((a - b).norm().item())
    return float((a - b).norm().item() / na)


def extract_positions(prompt_len: int, d: int) -> dict[str, int]:
    """Canonical Method B absolute positions."""
    d = int(d)
    prompt_len = int(prompt_len)
    if d > 0:
        pos_shared = prompt_len + d - 1
    else:
        pos_shared = max(prompt_len - 1, 0)
    return {"shared": pos_shared, "divergent": prompt_len + d}


def full_forward_matched_pair(
    bundle: Any,
    prompt_ids: list[int],
    calm_completion_ids: list[int],
    bad_completion_ids: list[int],
    d: int,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Reference capture via two full-sequence forwards (use_cache=False)."""
    from jlens_qwen.caa.activation_capture import capture_residuals_at_positions

    layers = list(layers or PROBE_LAYERS)
    d = int(d)
    prefix_c = list(calm_completion_ids[:d])
    prefix_b = list(bad_completion_ids[:d])
    if prefix_c != prefix_b:
        raise ValueError("calm/bad prefix token mismatch — cannot compare captures")

    prompt_len = len(prompt_ids)
    positions = extract_positions(prompt_len, d)
    pos_shared = positions["shared"]
    pos_div = positions["divergent"]

    # Shared: forward only through shared prefix (fair match to KV prefill length).
    shared_ids = list(prompt_ids) + list(calm_completion_ids[:d])
    if len(shared_ids) < pos_shared + 1:
        raise ValueError("shared prefix too short for pos_shared")
    shared_t = torch.tensor(
        [shared_ids[: pos_shared + 1]], dtype=torch.long, device=bundle.device
    )
    shared_raw = capture_residuals_at_positions(bundle, shared_t, [pos_shared], layers)

    calm_full = list(prompt_ids) + list(calm_completion_ids)
    bad_full = list(prompt_ids) + list(bad_completion_ids)
    need = pos_div + 1
    if len(calm_full) < need or len(bad_full) < need:
        raise ValueError(
            f"completion too short for d={d}: calm={len(calm_completion_ids)} "
            f"bad={len(bad_completion_ids)} need_abs={need}"
        )

    calm_ids_t = torch.tensor([calm_full[:need]], dtype=torch.long, device=bundle.device)
    bad_ids_t = torch.tensor([bad_full[:need]], dtype=torch.long, device=bundle.device)

    calm_raw = capture_residuals_at_positions(bundle, calm_ids_t, [pos_div], layers)
    bad_raw = capture_residuals_at_positions(bundle, bad_ids_t, [pos_div], layers)

    calm_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}
    bad_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}
    for L in layers:
        if pos_shared in shared_raw.get(L, {}):
            calm_out[L]["shared"] = shared_raw[L][pos_shared]
            bad_out[L]["shared"] = shared_raw[L][pos_shared]
        if pos_div in calm_raw.get(L, {}):
            calm_out[L]["divergent"] = calm_raw[L][pos_div]
        if pos_div in bad_raw.get(L, {}):
            bad_out[L]["divergent"] = bad_raw[L][pos_div]

    return {
        "layers": layers,
        "prompt_len": prompt_len,
        "d": d,
        "positions": positions,
        "calm": calm_out,
        "bad": bad_out,
        "capture_method": "full_forward",
    }


def compare_captures(
    full: dict[str, Any],
    kv: dict[str, Any],
    *,
    cos_min: float = 0.99999,
    rel_l2_max: float = 2e-3,
    layers: list[int] | None = None,
) -> list[CheckResult]:
    """Compare full-forward vs shared-KV captures.

    cos_min=0.99999 is the hard directional proof. rel_l2_max=2e-3 covers fp16
    residual noise observed on Qwen2.5-Coder-7B (plan linf ≈1e-3 budget).
    """
    layers = list(layers or full.get("layers") or PROBE_LAYERS)
    results: list[CheckResult] = []

    # KV6 / layers
    kv_layers = set(int(x) for x in (kv.get("layers") or []))
    full_layers = set(int(x) for x in (full.get("layers") or []))
    expected = set(layers)
    results.append(
        CheckResult(
            "KV4_layers",
            kv_layers == expected and full_layers == expected,
            f"kv={sorted(kv_layers)} full={sorted(full_layers)} expected={sorted(expected)}",
        )
    )

    results.append(
        CheckResult(
            "KV6_capture_method",
            bool(kv.get("capture_method")),
            f"kv_method={kv.get('capture_method')} full_method={full.get('capture_method')}",
        )
    )

    # Positions
    prompt_len = int(full.get("prompt_len") or kv.get("prompt_len") or 0)
    d = int(full.get("d") if full.get("d") is not None else kv.get("d") or 0)
    expected_pos = extract_positions(prompt_len, d)
    kv_pos = kv.get("positions") or {}
    full_pos = full.get("positions") or {}
    results.append(
        CheckResult(
            "KV2_positions",
            kv_pos.get("shared") == expected_pos["shared"]
            and kv_pos.get("divergent") == expected_pos["divergent"]
            and full_pos.get("shared") == expected_pos["shared"]
            and full_pos.get("divergent") == expected_pos["divergent"],
            f"expected={expected_pos} kv={kv_pos} full={full_pos}",
        )
    )
    results.append(
        CheckResult(
            "KV7_databus_positions",
            expected_pos["shared"] == (prompt_len + d - 1 if d > 0 else max(prompt_len - 1, 0))
            and expected_pos["divergent"] == prompt_len + d,
            f"pos={expected_pos}",
        )
    )

    # KV3 shared identical on KV path
    kv_calm = kv.get("calm") or {}
    kv_bad = kv.get("bad") or {}
    shared_ok = True
    shared_detail = []
    for L in layers:
        Ls = str(L)
        c_s = (kv_calm.get(L) or kv_calm.get(Ls) or {}).get("shared")
        b_s = (kv_bad.get(L) or kv_bad.get(Ls) or {}).get("shared")
        if c_s is None or b_s is None:
            shared_ok = False
            shared_detail.append(f"L{L}:missing")
            continue
        if c_s != b_s:
            shared_ok = False
            shared_detail.append(f"L{L}:mismatch")
    results.append(
        CheckResult("KV3_shared_identical", shared_ok, ";".join(shared_detail) or "ok")
    )

    # Numerical: KV8 / KV9 / KV10
    arms = ("calm", "bad")
    keys = ("shared", "divergent")
    for arm in arms:
        full_arm = full.get(arm) or {}
        kv_arm = kv.get(arm) or {}
        for L in layers:
            Ls = str(L)
            for key in keys:
                fh = (full_arm.get(L) or full_arm.get(Ls) or {}).get(key)
                kh = (kv_arm.get(L) or kv_arm.get(Ls) or {}).get(key)
                ft = _as_tensor(fh)
                kt = _as_tensor(kh)
                cid = f"KV8_cos_{arm}_L{L}_{key}"
                lid = f"KV9_relL2_{arm}_L{L}_{key}"
                if ft is None or kt is None:
                    results.append(CheckResult(cid, False, "missing vector"))
                    results.append(CheckResult(lid, False, "missing vector"))
                    continue
                cos = cosine_similarity(ft, kt)
                rel = relative_l2(ft, kt)
                results.append(
                    CheckResult(cid, cos >= cos_min, f"cos={cos:.8f} min={cos_min}")
                )
                results.append(
                    CheckResult(lid, rel <= rel_l2_max, f"rel_l2={rel:.8e} max={rel_l2_max}")
                )

    # KV10: divergent present for both arms on both methods
    for arm in arms:
        for method_name, capt in (("full", full), ("kv", kv)):
            arm_d = capt.get(arm) or {}
            ok = all(
                (arm_d.get(L) or arm_d.get(str(L)) or {}).get("divergent") is not None
                for L in layers
            )
            results.append(
                CheckResult(
                    f"KV10_divergent_{method_name}_{arm}",
                    ok,
                    f"layers={layers}",
                )
            )

    return results


def assert_kv_equivalent(
    full: dict[str, Any],
    kv: dict[str, Any],
    *,
    cos_min: float = 0.99999,
    rel_l2_max: float = 2e-3,
) -> list[CheckResult]:
    results = compare_captures(full, kv, cos_min=cos_min, rel_l2_max=rel_l2_max)
    failed = [r for r in results if not r.passed]
    if failed:
        detail = "; ".join(f"{r.id}:{r.detail}" for r in failed[:8])
        raise AssertionError(f"KV equivalence failed ({len(failed)} checks): {detail}")
    return results


def structural_prefix_guard(
    calm_completion_ids: list[int],
    bad_completion_ids: list[int],
    d: int,
) -> None:
    """KV1: mismatched prefixes must raise."""
    if list(calm_completion_ids[:d]) != list(bad_completion_ids[:d]):
        raise ValueError("calm/bad prefix token mismatch — cannot share KV prefill")
