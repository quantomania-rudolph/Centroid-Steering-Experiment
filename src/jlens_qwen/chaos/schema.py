"""JSON schema helpers for chaos persistence (v2).

v2 adds full-completion ``token_ids`` + ``n_tokens_full`` so prefix
reconstruction survives ``tokens[]`` downsampling (>512).
"""

from __future__ import annotations

from typing import Any

from jlens_qwen.chaos.constants import CHAOS_SCHEMA_VERSION, PROBE_LAYERS


def empty_chaos_record(*, device: str | None = None) -> dict[str, Any]:
    return {
        "version": CHAOS_SCHEMA_VERSION,
        "device": device,
        "probe_layers": list(PROBE_LAYERS),
        "cadence": {
            "F0": True,
            "F1": True,
            "F2_stride": 8,
            "F5_gated": True,
        },
        "tokens": [],
        "token_ids": [],
        "n_tokens_full": 0,
        "divergence_hints": {},
        "events": [],
        "curves_E": {},
        "composites": {},
        "steering_hooks": {
            "recommended_layers": [],
            "disagreement_F": None,
            "g_residual": None,
        },
        "timing": {},
        "degraded_mode": False,
        "flags_timeline": [],
    }
