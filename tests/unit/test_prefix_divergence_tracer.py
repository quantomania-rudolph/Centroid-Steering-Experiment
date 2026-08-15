"""Unit tests for PrefixDivergenceTracer (≥8 cases from plan)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.chaos.flags import THRESH
from prefix_divergence_tracer import (
    eligible_for_panic_trace,
    trace_invocation,
)


def _token(
    step: int,
    *,
    distress: float = 0.0,
    stub: float = 0.0,
    flags: list[str] | None = None,
    token_id: int | None = None,
) -> dict:
    return {
        "step": step,
        "token_id": token_id if token_id is not None else (1000 + step),
        "A_vocab": {"entropy": 0.5, "margin": 0.5},
        "B_vocab": {
            "code_mass": 0.5,
            "distress_mass": distress,
            "stub_mass": stub,
            "junk_mass": 0.0,
        },
        "C_vocab": {"flip_rate": 0.0, "oscillation": 0.0},
        "flags": list(flags or []),
    }


def _inv(
    tokens: list[dict],
    *,
    recipe_step: int | None = None,
    token_ids: list[int] | None = None,
    assistant: str = "",
    completion_tokens: int | None = None,
) -> dict:
    chaos: dict = {"tokens": tokens, "monitor": {"recipe_timeline": []}}
    if recipe_step is not None:
        chaos["monitor"]["recipe_timeline"] = [
            {"step": recipe_step, "recipe_id": "B_STUB_DISTRESS", "severity": "warn"}
        ]
    if token_ids is not None:
        chaos["token_ids"] = token_ids
        chaos["n_tokens_full"] = len(token_ids)
    ct = completion_tokens
    if ct is None and token_ids is not None:
        ct = len(token_ids)
    if ct is None:
        ct = len(tokens)
    return {
        "invocation_index": 0,
        "id": "test-inv",
        "completion_tokens": ct,
        "assistant": assistant,
        "chaos": chaos,
    }


def test_simultaneous_distress_stub_at_42():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(50)]
    tokens[42] = _token(
        42,
        distress=thr_d,
        stub=thr_s,
        flags=["B_distress", "B_stub"],
    )
    # eligibility via flags rates — need ≥1% both; inject a few more
    for s in (43, 44):
        tokens[s] = _token(s, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"])
    ids = list(range(50))
    r = trace_invocation(_inv(tokens, recipe_step=42, token_ids=ids), use_tokenizer=False)
    assert r.eligible
    assert r.d_lexicon == 42
    assert r.d_flags == 42
    assert r.d_consensus == 42
    assert r.prefix_end == 41


def test_stub_after_distress():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(50)]
    tokens[40] = _token(40, distress=thr_d, flags=["B_distress"])
    tokens[45] = _token(
        45, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
    )
    for s in (46, 47):
        tokens[s] = _token(
            s, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
        )
    r = trace_invocation(_inv(tokens, recipe_step=45, token_ids=list(range(50))), use_tokenizer=False)
    assert r.d_consensus == 45
    assert r.divergence_class == "stub_after_distress"
    assert r.d_distress_only == 40
    assert r.d_stub_only == 45


def test_debounced_recipe_later_than_lexicon():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(60)]
    tokens[41] = _token(
        41, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
    )
    for s in (42, 43, 44):
        tokens[s] = _token(
            s, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
        )
    r = trace_invocation(_inv(tokens, recipe_step=50, token_ids=list(range(60))), use_tokenizer=False)
    assert r.d_consensus == 41
    assert r.d_recipe == 50
    assert r.d_recipe_lag == 9


def test_clean_tokens_not_eligible():
    tokens = [_token(i) for i in range(20)]
    r = trace_invocation(_inv(tokens, token_ids=list(range(20))), use_tokenizer=False)
    assert not r.eligible
    assert r.d_consensus is None
    assert r.confidence == "none"


def test_distress_only_not_eligible():
    thr_d = THRESH["B_distress"]
    tokens = [_token(i) for i in range(20)]
    for s in range(5, 10):
        tokens[s] = _token(s, distress=thr_d, flags=["B_distress"])
    r = trace_invocation(_inv(tokens, token_ids=list(range(20))), use_tokenizer=False)
    assert not r.eligible
    assert r.d_consensus is None
    assert r.confidence == "none"


def test_token_ids_prefix_slice():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i, token_id=200 + i) for i in range(20)]
    tokens[10] = _token(
        10,
        distress=thr_d,
        stub=thr_s,
        flags=["B_distress", "B_stub"],
        token_id=210,
    )
    for s in (11, 12):
        tokens[s] = _token(
            s,
            distress=thr_d,
            stub=thr_s,
            flags=["B_distress", "B_stub"],
            token_id=200 + s,
        )
    ids = [200 + i for i in range(20)]
    r = trace_invocation(_inv(tokens, recipe_step=10, token_ids=ids), use_tokenizer=False)
    assert r.d_consensus == 10
    assert r.prefix_token_ids == ids[:10]
    assert r.token_ids_match
    assert r.method_b_record is not None
    assert r.method_b_record["d"] == 10
    assert r.method_b_record["prefix_end"] == 9


def test_missing_token_ids_legacy_medium_with_recipe():
    """Legacy gold without token_ids can still reach medium when recipe lag is ok."""
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(30)]
    for s in (15, 16, 17):
        tokens[s] = _token(
            s, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
        )
    r = trace_invocation(_inv(tokens, recipe_step=15, token_ids=None), use_tokenizer=False)
    assert r.eligible
    assert r.d_consensus == 15
    assert r.confidence == "medium"
    assert any("token_ids missing" in n for n in r.notes)


def test_missing_token_ids_low_without_recipe():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    tokens = [_token(i) for i in range(30)]
    for s in (15, 16, 17):
        tokens[s] = _token(
            s, distress=thr_d, stub=thr_s, flags=["B_distress", "B_stub"]
        )
    # Eligible via flag rates, but no recipe → confidence low without token_ids
    r = trace_invocation(_inv(tokens, recipe_step=None, token_ids=None), use_tokenizer=False)
    assert r.eligible
    assert r.d_consensus == 15
    assert r.confidence == "low"


class _FakeTok:
    """Decode each id as a single printable char for d_text alignment tests."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_d_text_alignment_within_5():
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    # Build assistant text whose panic keyword starts at a known char offset.
    # token_ids map 1:1 to chars via FakeTok (chr(id)).
    # Use ids in printable ASCII range.
    prefix = "ok code here "
    panic = "sorry NotImplementedError"
    assistant = prefix + panic
    # Map each char to its ord as token id
    ids = [ord(c) for c in assistant]
    tokens = [_token(i, token_id=ids[i]) for i in range(len(ids))]
    # Fire both masses at the 's' of sorry
    panic_start = len(prefix)
    for s in range(panic_start, min(panic_start + 3, len(ids))):
        tokens[s] = _token(
            s,
            distress=thr_d,
            stub=thr_s,
            flags=["B_distress", "B_stub"],
            token_id=ids[s],
        )
    r = trace_invocation(
        _inv(
            tokens,
            recipe_step=panic_start,
            token_ids=ids,
            assistant=assistant,
        ),
        tokenizer=_FakeTok(),
        use_tokenizer=False,
    )
    assert r.d_text is not None
    assert abs(r.d_text - r.d_consensus) <= 5
    assert r.d_consensus == panic_start


def test_estimators_prefer_divergence_hints():
    """Bus hints survive tokens[] downsample blind spots."""
    tokens = [_token(i) for i in range(3)]
    chaos = {
        "tokens": tokens,
        "token_ids": list(range(20)),
        "n_tokens_full": 20,
        "divergence_hints": {
            "d_lexicon": 5,
            "d_flags": 5,
            "d_distress_only": 4,
            "d_stub_only": 5,
            "b_distress_rate": 0.15,
            "b_stub_rate": 0.12,
        },
        "monitor": {
            "recipe_timeline": [
                {"recipe_id": "B_STUB_DISTRESS", "step": 6, "severity": "warn"}
            ]
        },
    }
    inv = {
        "chaos": chaos,
        "completion_tokens": 20,
        "invocation_index": 0,
        "id": "hint-inv",
    }
    r = trace_invocation(inv, use_tokenizer=False)
    assert r.d_lexicon == 5
    assert r.d_flags == 5
    assert r.d_consensus == 5
    assert r.d_recipe == 6
    assert r.d_recipe_lag == 1


def test_eligible_uses_flags_thresh_not_hardcoded():
    """U3 / C2: tracer must use flags.THRESH for mass crosses."""
    assert THRESH["B_distress"] == 0.12
    assert THRESH["B_stub"] == 0.10
    tokens = [_token(0, distress=0.11, stub=0.09)]  # just under
    chaos = {"tokens": tokens, "monitor": {"recipe_timeline": []}}
    assert not eligible_for_panic_trace(chaos)
    # Over threshold but only one token with both flags → rate may be 100%
    tokens2 = [
        _token(0, distress=0.12, stub=0.10, flags=["B_distress", "B_stub"]),
    ]
    assert eligible_for_panic_trace(
        {"tokens": tokens2, "monitor": {"recipe_timeline": []}}
    )
