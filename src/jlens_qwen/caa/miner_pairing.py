"""Method B natural pair pairing helpers — d-coherence, field remap, chaos builders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens_qwen.caa.calm_tail import longest_common_prefix
from jlens_qwen.chaos.constants import PROBE_LAYERS

CHAOS_DEBOUNCE_STEPS = 15


@dataclass
class CheckResult:
    id: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "pass": self.passed, "detail": self.detail}


def extract_d_from_lcp(calm_ids: list[int], bad_ids: list[int]) -> int:
    return longest_common_prefix(list(calm_ids), list(bad_ids))


def check_d_coherence(
    d_extract: int,
    tracer_d_consensus: int,
    *,
    debounce: int = CHAOS_DEBOUNCE_STEPS,
) -> CheckResult:
    delta = abs(int(d_extract) - int(tracer_d_consensus))
    ok = delta <= debounce
    return CheckResult(
        id="NM_d_coherence",
        passed=ok,
        detail=f"d_extract={d_extract} tracer={tracer_d_consensus} delta={delta} max={debounce}",
    )


def remap_invocation_fields(inv: dict[str, Any], experiment_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize corpus invocation + attach experiment_data for text resolution."""
    out = dict(inv)
    exp = experiment_data or {}
    pt = inv.get("prompt_tokens")
    if pt is not None:
        out["prompt_len"] = int(pt)
    out["_experiment_data"] = exp
    return out


def build_clean_chaos_from_token_ids(token_ids: list[int]) -> dict[str, Any]:
    """Synthetic chaos for calm-arm validation (zero distress/stub)."""
    ids = [int(x) for x in token_ids]
    return {
        "tokens": [
            {
                "step": i,
                "token_id": tid,
                "B_vocab": {"distress_mass": 0.0, "stub_mass": 0.0},
                "flags": [],
            }
            for i, tid in enumerate(ids)
        ],
        "token_ids": ids,
        "n_tokens_full": len(ids),
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }


def resolve_prompt_ids_from_session(
    inv: dict[str, Any],
    experiment_data: dict[str, Any],
    tokenizer: Any,
) -> list[int]:
    """Reconstruct prompt token IDs from session JSONL; fail if count mismatch."""
    session_path = experiment_data.get("session_path")
    inv_id = inv.get("id") or inv.get("invocation_id")
    expected = int(inv.get("prompt_tokens") or inv.get("prompt_len") or -1)
    if not session_path or not inv_id:
        raise ValueError("session_path and invocation id required for prompt reconstruction")

    messages: list[dict[str, str]] = []
    for line in Path(session_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("id") or "") != str(inv_id):
            continue
        for msg in row.get("messages") or []:
            role = msg.get("role")
            content = msg.get("content")
            if role and content is not None:
                messages.append({"role": str(role), "content": str(content)})
        break

    if not messages:
        raise ValueError(f"no messages for invocation {inv_id} in {session_path}")

    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "__getitem__") and "input_ids" in encoded:
            ids = encoded["input_ids"]
        else:
            ids = encoded
        if hasattr(ids, "tolist"):
            prompt_ids = [int(x) for x in ids.tolist()]
        else:
            prompt_ids = [int(x) for x in ids]
    else:
        raise ValueError("tokenizer must support apply_chat_template")

    if expected >= 0 and len(prompt_ids) != expected:
        raise ValueError(
            f"prompt token count mismatch: encoded={len(prompt_ids)} expected={expected}"
        )
    return prompt_ids


def make_pair_slug(experiment: str, invocation_id: str, d: int) -> str:
    safe_exp = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment)[:40]
    safe_id = "".join(c if c.isalnum() else "_" for c in invocation_id)[-16:]
    return f"{safe_exp}__{safe_id}__d{d}"


def dense_token_ids_ok(trace: Any) -> bool:
    return bool(getattr(trace, "token_ids_match", False)) and getattr(trace, "d_consensus", None) is not None
