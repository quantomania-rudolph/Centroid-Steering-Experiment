"""Neutral-task activation capture for û_neutral subspace cleanup (UNI_CAA01 §2.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from jlens_qwen.caa.activation_capture import capture_residuals_at_positions
from jlens_qwen.chaos.constants import PROBE_LAYERS
from jlens_qwen.model_loader import ModelBundle

MIN_NORM = 1e-8

# Provenance-clean routine coding tasks only.
# chaos_proof_meta_wipe was removed — it was designed as a distress-pressure probe.
NEUTRAL_TASKS: list[dict[str, str]] = [
    {
        "slug": "chaos_proof_clean",
        "provenance": "routine fibonacci kata; no distress-pressure design",
        "user_prompt": (
            "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) "
            "prints 55. Lint-clean."
        ),
        "assistant_start": "```python\nimport",
    },
    {
        "slug": "neutral_topo_sort",
        "provenance": "routine topo-sort kata; final_review_topo_sort starter",
        "user_prompt": (
            "Edit final_review_topo_sort/main.py only. Fix topo_sort so demo() "
            "prints a valid topological order for the sample DAG. Lint-clean."
        ),
        "assistant_start": "```python\nfrom",
    },
]


@dataclass
class NeutralCaptureResult:
    layers: list[int]
    n_tasks: int
    neutral: dict[int, list[float]]
    neutral_raw: dict[int, list[float]]
    per_task: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "n_tasks": self.n_tasks,
            "neutral": self.neutral,
            "neutral_raw": self.neutral_raw,
            "per_task": [
                {k: v for k, v in row.items() if not str(k).startswith("h_L")}
                for row in self.per_task
            ],
        }


def encode_chat_prompt(tokenizer: Any, user_prompt: str) -> list[int]:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "__getitem__") and "input_ids" in encoded:
            ids = encoded["input_ids"]
        else:
            ids = encoded
        if hasattr(ids, "tolist"):
            return [int(x) for x in ids.tolist()]
        return [int(x) for x in ids]
    return tokenizer.encode(f"User: {user_prompt}\nAssistant:", add_special_tokens=True)


def verify_gen_start_tokenization(
    tokenizer: Any,
    *,
    prompt_ids: list[int],
    assistant_start: str,
    first_token_id: int,
) -> tuple[bool, str]:
    """N8: spliced gen-start token must match in-context tokenization."""
    start_ids = tokenizer.encode(assistant_start, add_special_tokens=False)
    if hasattr(start_ids, "tolist"):
        start_ids = [int(x) for x in start_ids.tolist()]
    else:
        start_ids = [int(x) for x in start_ids]
    if not start_ids:
        return False, "assistant_start encodes empty"
    if int(start_ids[0]) != int(first_token_id):
        return False, f"isolated first token {start_ids[0]} != spliced {first_token_id}"

    # Round-trip: re-tokenize prompt+assistant_start prefix and compare gen token.
    messages = [{"role": "user", "content": ""}]
    if hasattr(tokenizer, "apply_chat_template"):
        # Use decoded prompt tail + assistant start for cross-check when possible.
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        combined = prompt_text + assistant_start
        round_ids = tokenizer.encode(combined, add_special_tokens=False)
        if hasattr(round_ids, "tolist"):
            round_ids = [int(x) for x in round_ids.tolist()]
        else:
            round_ids = [int(x) for x in round_ids]
        if len(round_ids) >= len(prompt_ids) + 1:
            rt_tok = int(round_ids[len(prompt_ids)])
            if rt_tok != int(first_token_id):
                return (
                    False,
                    f"round-trip token at gen_pos mismatch: {rt_tok} != {first_token_id}",
                )
    return True, "ok"


def _build_replay_chaos(token_ids: list[int]) -> dict[str, Any]:
    return {
        "tokens": [
            {
                "step": i,
                "token_id": int(tid),
                "B_vocab": {"distress_mass": 0.0, "stub_mass": 0.0},
                "flags": [],
            }
            for i, tid in enumerate(token_ids)
        ],
        "token_ids": [int(x) for x in token_ids],
        "n_tokens_full": len(token_ids),
        "divergence_hints": {"b_distress_rate": 0.0, "b_stub_rate": 0.0},
        "monitor": {"recipe_timeline": []},
    }


@torch.no_grad()
def capture_neutral_gen_start(
    bundle: ModelBundle,
    *,
    tasks: list[dict[str, str]] | None = None,
    layers: list[int] | None = None,
) -> NeutralCaptureResult:
    """Capture h at first assistant token (gen step 0) on clean neutral tasks."""
    tasks = tasks or NEUTRAL_TASKS
    if len(tasks) < 2:
        raise ValueError("û_neutral requires ≥2 neutral tasks (N-S2)")
    layers = list(layers or PROBE_LAYERS)
    if layers != list(PROBE_LAYERS):
        raise ValueError(f"neutral capture layers must be {PROBE_LAYERS}, got {layers}")

    per_task: list[dict[str, Any]] = []
    accum: dict[int, list[torch.Tensor]] = {L: [] for L in layers}

    for task in tasks:
        prompt_ids = encode_chat_prompt(bundle.tokenizer, task["user_prompt"])
        start_ids = bundle.tokenizer.encode(
            task["assistant_start"], add_special_tokens=False
        )
        if hasattr(start_ids, "tolist"):
            start_ids = [int(x) for x in start_ids.tolist()]
        else:
            start_ids = [int(x) for x in start_ids]
        if not start_ids:
            raise ValueError(f"task {task['slug']}: assistant_start encodes empty")

        first_token_id = int(start_ids[0])
        full_ids = list(prompt_ids) + [first_token_id]
        gen_pos = len(prompt_ids)

        tok_ok, tok_detail = verify_gen_start_tokenization(
            bundle.tokenizer,
            prompt_ids=prompt_ids,
            assistant_start=task["assistant_start"],
            first_token_id=first_token_id,
        )
        if not tok_ok:
            raise ValueError(f"task {task['slug']}: tokenization check failed: {tok_detail}")

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=bundle.device)
        captured = capture_residuals_at_positions(bundle, input_ids, [gen_pos], layers)

        row: dict[str, Any] = {
            "slug": task["slug"],
            "provenance": task.get("provenance", ""),
            "prompt_len": len(prompt_ids),
            "gen_position": gen_pos,
            "first_token_id": first_token_id,
            "user_prompt": task["user_prompt"],
            "tokenization_ok": tok_ok,
            "tokenization_detail": tok_detail,
            "replay_chaos": _build_replay_chaos(full_ids),
        }
        for L in layers:
            vec = captured.get(L, {}).get(gen_pos)
            if vec is None:
                raise ValueError(f"task {task['slug']}: missing activation at L{L} gen_pos={gen_pos}")
            accum[L].append(torch.tensor(vec, dtype=torch.float32))
            row[f"h_L{L}_norm"] = float(torch.tensor(vec).norm().item())
            row[f"h_L{L}"] = vec
        per_task.append(row)

    if len(per_task) < 2:
        raise ValueError(f"û_neutral captured only {len(per_task)} tasks — need ≥2")

    neutral: dict[int, list[float]] = {}
    neutral_raw: dict[int, list[float]] = {}
    for L in layers:
        if len(accum[L]) < 2:
            raise ValueError(f"N9: layer L{L} has only {len(accum[L])} task vectors — need ≥2")
        mean_h = torch.stack(accum[L], dim=0).mean(dim=0)
        norm = float(mean_h.norm().item())
        if norm <= MIN_NORM:
            raise ValueError(f"N10: layer L{L} mean norm too small ({norm}) — degenerate neutral")
        neutral_raw[L] = mean_h.tolist()
        neutral[L] = (mean_h / norm).tolist()

    return NeutralCaptureResult(
        layers=layers,
        n_tasks=len(per_task),
        neutral=neutral,
        neutral_raw=neutral_raw,
        per_task=per_task,
    )
