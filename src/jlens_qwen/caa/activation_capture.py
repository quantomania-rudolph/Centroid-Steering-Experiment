"""Capture layer residuals at absolute sequence positions (Method B extraction)."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

from jlens_qwen.model_loader import ActivationRecorder, ModelBundle


def _vec_at_last(rec: ActivationRecorder, layer: int) -> list[float] | None:
    h = rec.activations.get(layer)
    if h is None:
        return None
    return h[0, -1, :].float().detach().cpu().tolist()


def _vec_at_index(rec: ActivationRecorder, layer: int, pos: int) -> list[float] | None:
    h = rec.activations.get(layer)
    if h is None:
        return None
    if pos < 0 or pos >= int(h.shape[1]):
        return None
    return h[0, pos, :].float().detach().cpu().tolist()


@torch.no_grad()
def capture_residuals_at_positions(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    positions: list[int],
    layers: list[int] | None = None,
) -> dict[int, dict[int, list[float]]]:
    """Forward full sequence; return {layer: {position: h_vec}} as CPU float lists."""
    layers = layers or [10, 16, 22]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(bundle.device)

    out: dict[int, dict[int, list[float]]] = {L: {} for L in layers}
    with ActivationRecorder(bundle.layers, at=layers) as rec:
        bundle.model(input_ids=input_ids, use_cache=False)
        seq_len = int(input_ids.shape[1])
        for L in layers:
            h = rec.activations.get(L)
            if h is None:
                continue
            for pos in positions:
                if 0 <= pos < seq_len:
                    vec = h[0, pos, :].float().detach().cpu()
                    out[L][pos] = vec.tolist()
    return out


def _clone_past_key_values(past: Any) -> Any:
    """Independent KV cache copy so calm/bad single-token decodes do not share state."""
    if past is None:
        return None
    # HuggingFace DynamicCache / StaticCache
    if hasattr(past, "to_legacy_cache") and hasattr(type(past), "from_legacy_cache"):
        legacy = past.to_legacy_cache()
        cloned = tuple(tuple(t.clone() for t in layer) for layer in legacy)
        return type(past).from_legacy_cache(cloned)
    if isinstance(past, tuple):
        return tuple(
            tuple(t.clone() if torch.is_tensor(t) else t for t in layer) for layer in past
        )

    return copy.deepcopy(past)


@torch.no_grad()
def _capture_matched_pair_dual_forward(
    bundle: ModelBundle,
    prompt_ids: list[int],
    calm_completion_ids: list[int],
    bad_completion_ids: list[int],
    d: int,
    layers: list[int],
) -> dict[str, Any]:
    """Slower but XPU-safe: two full forwards, no KV cache clone (Arc DEVICE_LOST prone)."""
    prompt_len = len(prompt_ids)
    pos_shared = prompt_len + d - 1 if d > 0 else max(prompt_len - 1, 0)
    pos_div = prompt_len + d

    calm_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}
    bad_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}

    calm_seq = list(prompt_ids) + list(calm_completion_ids[: d + 1])
    bad_seq = list(prompt_ids) + list(bad_completion_ids[: d + 1])
    if len(calm_seq) <= pos_div or len(bad_seq) <= pos_div:
        raise ValueError("completion too short for divergent position extraction")

    def _run_arm(seq: list[int], dest: dict[int, dict[str, list[float]]]) -> None:
        input_ids = torch.tensor([seq], dtype=torch.long, device=bundle.device)
        with ActivationRecorder(bundle.layers, at=layers) as rec:
            bundle.model(input_ids=input_ids, use_cache=False)
            for L in layers:
                shared = _vec_at_index(rec, L, pos_shared)
                div = _vec_at_index(rec, L, pos_div)
                if shared is not None:
                    dest[L]["shared"] = shared
                if div is not None:
                    dest[L]["divergent"] = div
        if str(bundle.device).startswith("xpu") and hasattr(torch, "xpu"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

    _run_arm(calm_seq, calm_out)
    _run_arm(bad_seq, bad_out)

    # Shared must match across arms at the LCP position (fail closed if not).
    for L in layers:
        c_s = (calm_out.get(L) or {}).get("shared")
        b_s = (bad_out.get(L) or {}).get("shared")
        if c_s is None or b_s is None:
            continue
        # Numerical noise on hybrid XPU offload: require close, not bit-identical.
        if len(c_s) != len(b_s):
            raise RuntimeError(f"shared activation length mismatch at layer {L}")
        max_abs = max(abs(a - b) for a, b in zip(c_s, b_s))
        if max_abs > 1e-2:
            raise RuntimeError(f"shared activations diverged at layer {L}: max_abs={max_abs}")
        # Canonicalize so CAP6 identical-shared check stays fail-closed exact.
        bad_out[L]["shared"] = list(c_s)

    return {
        "layers": layers,
        "prompt_len": prompt_len,
        "d": d,
        "positions": {"shared": pos_shared, "divergent": pos_div},
        "calm": calm_out,
        "bad": bad_out,
        "capture_method": "dual_full_forward",
    }


def _model_is_accelerated_hybrid(bundle: ModelBundle) -> bool:
    model = bundle.model
    if getattr(model, "hf_device_map", None):
        return True
    try:
        return any(p.device.type == "meta" for p in model.parameters())
    except Exception:
        return False


@torch.no_grad()
def capture_matched_pair_activations(
    bundle: ModelBundle,
    prompt_ids: list[int],
    calm_completion_ids: list[int],
    bad_completion_ids: list[int],
    d: int,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Method B capture: shared-KV prefill by default; dual-forward on hybrid XPU.

    Shared-KV is ~2-3x faster but Arc + accelerate CPU-offload often hits DEVICE_LOST
    when cloning past_key_values. Dual full-forward is the durable XPU path.
    """
    layers = layers or [10, 16, 22]
    d = int(d)
    prefix_c = list(calm_completion_ids[:d])
    prefix_b = list(bad_completion_ids[:d])
    if prefix_c != prefix_b:
        raise ValueError("calm/bad prefix token mismatch — cannot share KV prefill")

    use_dual = str(bundle.device).startswith("xpu") and _model_is_accelerated_hybrid(bundle)
    if use_dual:
        return _capture_matched_pair_dual_forward(
            bundle, prompt_ids, calm_completion_ids, bad_completion_ids, d, list(layers)
        )

    prompt_len = len(prompt_ids)
    pos_shared = prompt_len + d - 1 if d > 0 else max(prompt_len - 1, 0)
    pos_div = prompt_len + d

    calm_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}
    bad_out: dict[int, dict[str, list[float]]] = {L: {} for L in layers}

    prefill_ids = list(prompt_ids) + prefix_c
    input_ids = torch.tensor([prefill_ids], dtype=torch.long, device=bundle.device)

    try:
        with ActivationRecorder(bundle.layers, at=layers) as rec:
            out = bundle.model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            for L in layers:
                shared_vec = _vec_at_last(rec, L)
                if shared_vec is not None:
                    calm_out[L]["shared"] = shared_vec
                    bad_out[L]["shared"] = shared_vec

        past_calm = _clone_past_key_values(past)
        past_bad = _clone_past_key_values(past)

        if d < len(calm_completion_ids):
            tok = torch.tensor(
                [[int(calm_completion_ids[d])]], dtype=torch.long, device=bundle.device
            )
            with ActivationRecorder(bundle.layers, at=layers) as rec:
                bundle.model(tok, past_key_values=past_calm, use_cache=True)
                for L in layers:
                    div_vec = _vec_at_last(rec, L)
                    if div_vec is not None:
                        calm_out[L]["divergent"] = div_vec

        if d < len(bad_completion_ids):
            tok = torch.tensor(
                [[int(bad_completion_ids[d])]], dtype=torch.long, device=bundle.device
            )
            with ActivationRecorder(bundle.layers, at=layers) as rec:
                bundle.model(tok, past_key_values=past_bad, use_cache=True)
                for L in layers:
                    div_vec = _vec_at_last(rec, L)
                    if div_vec is not None:
                        bad_out[L]["divergent"] = div_vec
    except RuntimeError as exc:
        msg = str(exc)
        if "DEVICE_LOST" in msg or "OUT_OF_DEVICE_MEMORY" in msg:
            if str(bundle.device).startswith("xpu") and hasattr(torch, "xpu"):
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass
            return _capture_matched_pair_dual_forward(
                bundle, prompt_ids, calm_completion_ids, bad_completion_ids, d, list(layers)
            )
        raise

    return {
        "layers": layers,
        "prompt_len": prompt_len,
        "d": d,
        "positions": {"shared": pos_shared, "divergent": pos_div},
        "calm": calm_out,
        "bad": bad_out,
        "capture_method": "shared_kv_prefill",
    }


def diff_vectors(
    pos_h: dict[int, list[float]],
    neg_h: dict[int, list[float]],
) -> list[float]:
    """pos - neg elementwise (behavior contrast)."""
    n = min(len(pos_h), len(neg_h))
    if n == 0:
        return []
    diff = [pos_h[i] - neg_h[i] for i in range(n)]
    norm = math.sqrt(sum(x * x for x in diff))
    if norm < 1e-12:
        return diff
    return [x / norm for x in diff]
